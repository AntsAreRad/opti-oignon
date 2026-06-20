#!/usr/bin/env python3
"""
Tests for the Benchmark Dashboard module (S60).

Covers BenchmarkHistory CRUD, comparison, trends, regression detection,
routes_benchmark endpoints (suites, tasks, runs, model config),
scoring logic, refusal detection, and run state management.

Target: 90+ tests, 0 regressions.

Uses importlib.util for direct module loading to avoid __init__.py
triggering hard ollama imports in this test environment.
"""

import asyncio
import importlib.util
import json
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# =============================================================================
# DIRECT MODULE LOADERS (bypass __init__.py)
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent


def _load_module(name, filepath):
    """Load a Python module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load benchmark_history module
_bh_mod = _load_module(
    "opti_oignon.benchmark_history",
    PROJECT_ROOT / "opti_oignon" / "benchmark_history.py",
)
BenchmarkHistory = _bh_mod.BenchmarkHistory
BenchmarkRunRecord = _bh_mod.BenchmarkRunRecord
BenchmarkResultRecord = _bh_mod.BenchmarkResultRecord

# Pre-load routes_benchmark module (mock package-level imports)
if "opti_oignon" not in sys.modules:
    sys.modules["opti_oignon"] = MagicMock()
if "opti_oignon.api" not in sys.modules:
    sys.modules["opti_oignon.api"] = MagicMock()

_routes_mod = _load_module(
    "opti_oignon.api.routes_benchmark",
    PROJECT_ROOT / "opti_oignon" / "api" / "routes_benchmark.py",
)
_calculate_score = _routes_mod._calculate_score
_is_refusal = _routes_mod._is_refusal
_load_benchmark_config = _routes_mod._load_benchmark_config
_load_models_config = _routes_mod._load_models_config
_save_models_config = _routes_mod._save_models_config
_RunState = _routes_mod._RunState
_router = _routes_mod.router


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="opti_bench_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def history(tmp_dir):
    return BenchmarkHistory(db_path=tmp_dir / "test_benchmark.db")


@pytest.fixture
def sample_run():
    return BenchmarkRunRecord(
        id=str(uuid.uuid4()), run_type="llm",
        started_at="2026-03-04T12:00:00Z", completed_at="2026-03-04T12:05:00Z",
        status="completed", models=["model-a", "model-b"],
        tasks=["task1", "task2"], total_tests=4,
        avg_score=7.5, best_model="model-a", duration_sec=300.0,
        config_snapshot={"temperature": 0.7},
    )


@pytest.fixture
def sample_result(sample_run):
    return BenchmarkResultRecord(
        id=str(uuid.uuid4()), run_id=sample_run.id,
        model="model-a", task="task1", task_name="Test Task 1",
        category="general", score=8.0, auto_score=8.0,
        time_seconds=5.5, status="success",
        response_preview="This is a test response.",
        keywords_found=["keyword1", "keyword2"], keywords_missing=["keyword3"],
    )


@pytest.fixture
def benchmark_config(tmp_dir):
    config = {
        "runner": {"timeout": 60, "temperature": 0.5},
        "scoring": {"user_weight": 0.6, "auto_weight": 0.4,
                    "regression_threshold": 1.5, "time_penalty_factor": 2.0,
                    "time_penalty_points": 2},
        "suites": {
            "test_suite": {"name": "Test Suite", "description": "A test suite",
                          "tasks": ["test_task_a", "test_task_b"]},
            "quick": {"name": "Quick", "description": "Quick check",
                     "tasks": ["test_task_a"]},
        },
        "tasks": {
            "test_task_a": {"name": "Task A", "description": "First task",
                           "category": "general", "prompt": "What is 2+2?",
                           "expected_keywords": ["four", "4"],
                           "max_expected_time": 30, "scoring_method": "keywords"},
            "test_task_b": {"name": "Task B", "description": "Second task",
                           "category": "code", "prompt": "Write hello world.",
                           "expected_keywords": ["print", "hello"],
                           "max_expected_time": 60, "scoring_method": "keywords"},
        },
    }
    path = tmp_dir / "benchmark.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def models_config(tmp_dir):
    config = {
        "routing": {
            "general": {"primary": "model-a", "fast": "model-b"},
            "code": {"primary": "model-c", "quality": "model-a"},
        },
        "fallback_order": ["model-a", "model-b", "model-c"],
        "special": {"embeddings": "embed-model"},
        "blacklist": [],
        "temperatures": {"code": 0.2, "general": 0.5},
        "timeouts": {"default": 300},
        "context_windows": {"model-a": 32768},
    }
    path = tmp_dir / "models.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


# =============================================================================
# BENCHMARK HISTORY -- INIT
# =============================================================================

class TestBenchmarkHistoryInit:

    def test_creates_db_file(self, tmp_dir):
        db_path = tmp_dir / "new.db"
        assert not db_path.exists()
        BenchmarkHistory(db_path=db_path)
        assert db_path.exists()

    def test_creates_tables(self, history):
        conn = history._get_conn()
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "benchmark_runs" in tables
        assert "benchmark_results" in tables

    def test_creates_indexes(self, history):
        conn = history._get_conn()
        indexes = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        conn.close()
        assert "idx_results_run_id" in indexes
        assert "idx_results_model" in indexes

    def test_idempotent_init(self, tmp_dir):
        db_path = tmp_dir / "idem.db"
        bh1 = BenchmarkHistory(db_path=db_path)
        bh2 = BenchmarkHistory(db_path=db_path)
        assert bh1.get_run_count() == 0
        assert bh2.get_run_count() == 0

    def test_parent_dir_created(self, tmp_dir):
        db_path = tmp_dir / "subdir" / "nested" / "test.db"
        BenchmarkHistory(db_path=db_path)
        assert db_path.exists()


# =============================================================================
# BENCHMARK HISTORY -- RUN CRUD
# =============================================================================

class TestBenchmarkHistoryRunCRUD:

    def test_save_run(self, history, sample_run):
        assert history.save_run(sample_run) == sample_run.id

    def test_get_runs_empty(self, history):
        assert history.get_runs() == []

    def test_get_runs_after_save(self, history, sample_run):
        history.save_run(sample_run)
        runs = history.get_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == sample_run.id

    def test_get_runs_filter_by_type(self, history, sample_run):
        history.save_run(sample_run)
        assert len(history.get_runs(run_type="llm")) == 1
        assert len(history.get_runs(run_type="perf")) == 0

    def test_get_runs_pagination(self, history):
        for i in range(5):
            history.save_run(BenchmarkRunRecord(
                id=str(uuid.uuid4()),
                started_at=f"2026-03-0{i+1}T12:00:00Z", status="completed"))
        assert len(history.get_runs(limit=3)) == 3
        assert len(history.get_runs(limit=3, offset=3)) == 2

    def test_get_runs_newest_first(self, history):
        ids = []
        for i in range(3):
            run = BenchmarkRunRecord(id=str(uuid.uuid4()),
                started_at=f"2026-03-0{i+1}T12:00:00Z", status="completed")
            history.save_run(run)
            ids.append(run.id)
        assert history.get_runs()[0]["id"] == ids[2]

    def test_get_run_detail(self, history, sample_run, sample_result):
        history.save_run(sample_run)
        history.save_result(sample_result)
        detail = history.get_run_detail(sample_run.id)
        assert detail is not None
        assert len(detail["results"]) == 1

    def test_get_run_detail_not_found(self, history):
        assert history.get_run_detail("nonexistent") is None

    def test_get_run_detail_has_ranking(self, history, sample_run, sample_result):
        history.save_run(sample_run)
        history.save_result(sample_result)
        detail = history.get_run_detail(sample_run.id)
        assert len(detail["global_ranking"]) == 1
        assert detail["global_ranking"][0]["rank"] == 1

    def test_get_run_detail_has_best_by_category(self, history, sample_run, sample_result):
        history.save_run(sample_run)
        history.save_result(sample_result)
        detail = history.get_run_detail(sample_run.id)
        assert detail["best_by_category"].get("general") == "model-a"

    def test_delete_run(self, history, sample_run):
        history.save_run(sample_run)
        assert history.delete_run(sample_run.id) is True
        assert len(history.get_runs()) == 0

    def test_delete_run_cascades(self, history, sample_run, sample_result):
        history.save_run(sample_run)
        history.save_result(sample_result)
        history.delete_run(sample_run.id)
        conn = history._get_conn()
        cnt = conn.execute("SELECT COUNT(*) as cnt FROM benchmark_results WHERE run_id = ?",
                          (sample_run.id,)).fetchone()["cnt"]
        conn.close()
        assert cnt == 0

    def test_delete_run_not_found(self, history):
        assert history.delete_run("nonexistent") is False

    def test_get_run_count(self, history, sample_run):
        assert history.get_run_count() == 0
        history.save_run(sample_run)
        assert history.get_run_count() == 1

    def test_save_run_upsert(self, history, sample_run):
        history.save_run(sample_run)
        sample_run.status = "error"
        history.save_run(sample_run)
        assert history.get_runs()[0]["status"] == "error"

    def test_run_models_json(self, history, sample_run):
        history.save_run(sample_run)
        assert isinstance(history.get_runs()[0]["models"], list)

    def test_run_config_snapshot(self, history, sample_run):
        history.save_run(sample_run)
        detail = history.get_run_detail(sample_run.id)
        assert detail["config_snapshot"]["temperature"] == 0.7


# =============================================================================
# BENCHMARK HISTORY -- RESULTS
# =============================================================================

class TestBenchmarkHistoryResults:

    def test_save_result(self, history, sample_run, sample_result):
        history.save_run(sample_run)
        assert history.save_result(sample_result) == sample_result.id

    def test_save_result_auto_id(self, history, sample_run):
        history.save_run(sample_run)
        r = BenchmarkResultRecord(run_id=sample_run.id, model="m", task="t")
        assert history.save_result(r)

    def test_result_keywords_json(self, history, sample_run, sample_result):
        history.save_run(sample_run)
        history.save_result(sample_result)
        detail = history.get_run_detail(sample_run.id)
        assert "keyword1" in detail["results"][0]["keywords_found"]

    def test_multiple_results(self, history, sample_run):
        history.save_run(sample_run)
        for i in range(5):
            history.save_result(BenchmarkResultRecord(
                run_id=sample_run.id, model=f"m-{i}", task="t1",
                score=float(i+5), auto_score=float(i+5), status="success"))
        assert len(history.get_run_detail(sample_run.id)["results"]) == 5

    def test_result_with_error(self, history, sample_run):
        history.save_run(sample_run)
        history.save_result(BenchmarkResultRecord(
            run_id=sample_run.id, model="m", task="t",
            status="error", error_message="Timeout"))
        assert history.get_run_detail(sample_run.id)["results"][0]["status"] == "error"

    def test_result_user_score(self, history, sample_run):
        history.save_run(sample_run)
        history.save_result(BenchmarkResultRecord(
            run_id=sample_run.id, model="m", task="t",
            score=8.2, auto_score=7.0, user_score=9.0, status="success"))
        assert history.get_run_detail(sample_run.id)["results"][0]["user_score"] == 9.0


# =============================================================================
# COMPARISON
# =============================================================================

class TestBenchmarkHistoryComparison:

    def _make(self, history, run_id, model, task, score):
        history.save_run(BenchmarkRunRecord(
            id=run_id, started_at=f"2026-03-04T1{run_id[-1]}:00:00Z", status="completed"))
        history.save_result(BenchmarkResultRecord(
            run_id=run_id, model=model, task=task,
            score=score, auto_score=score, status="success"))

    def test_compare_two_runs(self, history):
        self._make(history, "run-1", "m1", "t1", 7.0)
        self._make(history, "run-2", "m1", "t1", 9.0)
        comp = history.compare_runs(["run-1", "run-2"])
        assert comp.matrix["m1"]["t1"] == [7.0, 9.0]

    def test_compare_improvement(self, history):
        self._make(history, "run-a", "m1", "t1", 5.0)
        self._make(history, "run-b", "m1", "t1", 8.0)
        comp = history.compare_runs(["run-a", "run-b"])
        assert comp.deltas[0]["direction"] == "improved"

    def test_compare_regression(self, history):
        self._make(history, "run-x", "m1", "t1", 8.0)
        self._make(history, "run-y", "m1", "t1", 5.0)
        comp = history.compare_runs(["run-x", "run-y"])
        assert len(comp.regressions) == 1

    def test_compare_single_run(self, history):
        assert len(history.compare_runs(["only"]).runs) == 0

    def test_compare_nonexistent(self, history):
        assert len(history.compare_runs(["f1", "f2"]).runs) == 0

    def test_compare_stable(self, history):
        self._make(history, "run-s1", "m1", "t1", 7.0)
        self._make(history, "run-s2", "m1", "t1", 7.0)
        comp = history.compare_runs(["run-s1", "run-s2"])
        assert comp.deltas[0]["direction"] == "stable"


# =============================================================================
# TRENDS
# =============================================================================

class TestBenchmarkHistoryTrends:

    def test_trends_empty(self, history):
        assert history.get_model_trends("x").run_ids == []

    def test_trends_single(self, history):
        history.save_run(BenchmarkRunRecord(id="tr1", started_at="2026-03-01T12:00:00Z", status="completed"))
        history.save_result(BenchmarkResultRecord(
            run_id="tr1", model="tm", task="t1", score=8.0, auto_score=8.0,
            time_seconds=3.0, status="success"))
        t = history.get_model_trends("tm")
        assert t.avg_scores == [8.0]

    def test_trends_multiple(self, history):
        for i in range(3):
            history.save_run(BenchmarkRunRecord(
                id=f"tr-{i}", started_at=f"2026-03-0{i+1}T12:00:00Z", status="completed"))
            history.save_result(BenchmarkResultRecord(
                run_id=f"tr-{i}", model="tm", task="t1",
                score=float(6+i), auto_score=float(6+i), time_seconds=1.0, status="success"))
        assert history.get_model_trends("tm").avg_scores == [6.0, 7.0, 8.0]

    def test_trends_limit(self, history):
        for i in range(5):
            history.save_run(BenchmarkRunRecord(
                id=f"tl-{i}", started_at=f"2026-03-0{i+1}T12:00:00Z", status="completed"))
            history.save_result(BenchmarkResultRecord(
                run_id=f"tl-{i}", model="lm", task="t1",
                score=float(i), auto_score=float(i), time_seconds=1.0, status="success"))
        assert len(history.get_model_trends("lm", last_n_runs=3).run_ids) == 3


# =============================================================================
# RANKING
# =============================================================================

class TestRanking:

    def test_ranking_order(self, history):
        history.save_run(BenchmarkRunRecord(id="rk", started_at="2026-03-04T12:00:00Z", status="completed"))
        for m, s in [("ma", 9.0), ("mb", 5.0)]:
            history.save_result(BenchmarkResultRecord(
                run_id="rk", model=m, task="t1", score=s, auto_score=s, status="success", time_seconds=1.0))
        r = history.get_run_detail("rk")["global_ranking"]
        assert r[0]["model"] == "ma" and r[0]["rank"] == 1

    def test_ranking_excludes_errors(self, history):
        history.save_run(BenchmarkRunRecord(id="re", started_at="2026-03-04T12:00:00Z", status="completed"))
        history.save_result(BenchmarkResultRecord(
            run_id="re", model="m1", task="t1", score=8.0, auto_score=8.0, status="success", time_seconds=1.0))
        history.save_result(BenchmarkResultRecord(
            run_id="re", model="m1", task="t2", score=0.0, auto_score=0.0, status="error", time_seconds=1.0))
        assert history.get_run_detail("re")["global_ranking"][0]["tests"] == 1


# =============================================================================
# SCORING LOGIC
# =============================================================================

class TestScoringLogic:

    def test_all_keywords(self):
        s, f, m = _calculate_score(
            "The variety of species and life in the ecosystem shows great diversity.",
            ["variety", "species", "life", "ecosystem", "diversity"])
        assert s >= 8 and len(f) == 5

    def test_no_keywords_long(self):
        assert _calculate_score("word " * 100, [])[0] == 7

    def test_partial_keywords(self):
        s, f, m = _calculate_score("variety of species", ["variety", "species", "eco", "life"])
        assert len(f) == 2 and s == 3  # 5 - 2 short penalty

    def test_short_penalty(self):
        assert _calculate_score("Yes.", ["yes"])[0] == 8

    def test_bonus_completeness(self):
        assert _calculate_score("variety species life " + "x " * 200, ["variety", "species", "life"])[0] == 10

    def test_empty_response(self):
        assert _calculate_score("", ["kw"])[0] == 0

    def test_no_keywords_short(self):
        assert _calculate_score("Hi", [])[0] == 3

    def test_no_keywords_medium(self):
        assert _calculate_score("x " * 40, [])[0] == 5

    def test_case_insensitive(self):
        assert len(_calculate_score("VARIETY Species", ["variety", "species"])[1]) == 2


# =============================================================================
# REFUSAL DETECTION
# =============================================================================

class TestRefusalDetection:

    def test_sorry(self):
        assert _is_refusal("I'm sorry, I can't help.") is True

    def test_cannot(self):
        assert _is_refusal("I cannot provide that.") is True

    def test_as_ai(self):
        assert _is_refusal("As an AI, I don't have opinions.") is True

    def test_no_refusal(self):
        assert _is_refusal("The answer is 42.") is False

    def test_empty(self):
        assert _is_refusal("") is False

    def test_not_able(self):
        assert _is_refusal("I am not able to do that.") is True

    def test_normal(self):
        assert _is_refusal("Here is the code: print('hello')") is False


# =============================================================================
# CONFIG LOADING
# =============================================================================

class TestConfigLoading:

    def test_load_benchmark_config(self, benchmark_config):
        with patch.object(_routes_mod, "BENCHMARK_CONFIG_PATH", benchmark_config):
            c = _load_benchmark_config()
            assert "test_suite" in c["suites"]

    def test_load_missing_config(self, tmp_dir):
        with patch.object(_routes_mod, "BENCHMARK_CONFIG_PATH", tmp_dir / "no.yaml"):
            assert "suites" in _load_benchmark_config()

    def test_load_models_config(self, models_config):
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", models_config):
            assert "routing" in _load_models_config()

    def test_save_models_config(self, tmp_dir):
        path = tmp_dir / "save.yaml"
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", path):
            assert _save_models_config({"routing": {"t": {"primary": "m"}}}) is True
            with open(path) as f:
                assert yaml.safe_load(f)["routing"]["t"]["primary"] == "m"


# =============================================================================
# ROUTES -- SUITES
# =============================================================================

class TestSuiteEndpoints:

    @pytest.fixture
    def client(self, benchmark_config, models_config):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        app.include_router(_router)
        with patch.object(_routes_mod, "BENCHMARK_CONFIG_PATH", benchmark_config), \
             patch.object(_routes_mod, "MODELS_CONFIG_PATH", models_config):
            yield TestClient(app)

    def test_list_suites(self, client, benchmark_config):
        with patch.object(_routes_mod, "BENCHMARK_CONFIG_PATH", benchmark_config):
            r = client.get("/api/benchmark/suites")
            assert r.status_code == 200 and len(r.json()["suites"]) == 2

    def test_get_suite(self, client, benchmark_config):
        with patch.object(_routes_mod, "BENCHMARK_CONFIG_PATH", benchmark_config):
            r = client.get("/api/benchmark/suites/test_suite")
            assert r.status_code == 200 and r.json()["id"] == "test_suite"

    def test_suite_404(self, client, benchmark_config):
        with patch.object(_routes_mod, "BENCHMARK_CONFIG_PATH", benchmark_config):
            assert client.get("/api/benchmark/suites/fake").status_code == 404

    def test_list_tasks(self, client, benchmark_config):
        with patch.object(_routes_mod, "BENCHMARK_CONFIG_PATH", benchmark_config):
            assert len(client.get("/api/benchmark/tasks").json()["tasks"]) == 2

    def test_suite_categories(self, client, benchmark_config):
        with patch.object(_routes_mod, "BENCHMARK_CONFIG_PATH", benchmark_config):
            suites = client.get("/api/benchmark/suites").json()["suites"]
            s = next(s for s in suites if s["id"] == "test_suite")
            assert set(s["categories"]) == {"general", "code"}


# =============================================================================
# ROUTES -- STATUS & CANCEL
# =============================================================================

class TestRunStatusEndpoints:

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        app.include_router(_router)
        _routes_mod._state.finish("idle")
        yield TestClient(app)

    def test_status_idle(self, client):
        assert client.get("/api/benchmark/llm/status").json()["status"] != "running"

    def test_cancel_no_run(self, client):
        assert client.post("/api/benchmark/llm/cancel").status_code == 409


# =============================================================================
# ROUTES -- HISTORY
# =============================================================================

class TestHistoryEndpoints:

    @pytest.fixture
    def setup(self, tmp_dir):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        bh = BenchmarkHistory(db_path=tmp_dir / "test.db")
        bh.save_run(BenchmarkRunRecord(
            id="tr-1", started_at="2026-03-04T12:00:00Z",
            completed_at="2026-03-04T12:05:00Z", status="completed",
            models=["model-a"], tasks=["task1"], total_tests=1,
            avg_score=8.0, best_model="model-a", duration_sec=300.0))
        bh.save_result(BenchmarkResultRecord(
            run_id="tr-1", model="model-a", task="task1", task_name="Task 1",
            category="general", score=8.0, auto_score=8.0, status="success",
            time_seconds=5.0, response_preview="Test"))
        app = FastAPI()
        app.include_router(_router)
        with patch.object(_routes_mod, "HISTORY_AVAILABLE", True), \
             patch.object(_routes_mod, "benchmark_history", bh):
            yield TestClient(app), bh

    def test_list_runs(self, setup):
        c, bh = setup
        with patch.object(_routes_mod, "benchmark_history", bh), \
             patch.object(_routes_mod, "HISTORY_AVAILABLE", True):
            assert c.get("/api/benchmark/runs").json()["total"] == 1

    def test_get_detail(self, setup):
        c, bh = setup
        with patch.object(_routes_mod, "benchmark_history", bh), \
             patch.object(_routes_mod, "HISTORY_AVAILABLE", True):
            r = c.get("/api/benchmark/runs/tr-1")
            assert r.status_code == 200 and len(r.json()["results"]) == 1

    def test_detail_404(self, setup):
        c, bh = setup
        with patch.object(_routes_mod, "benchmark_history", bh), \
             patch.object(_routes_mod, "HISTORY_AVAILABLE", True):
            assert c.get("/api/benchmark/runs/fake").status_code == 404

    def test_delete(self, setup):
        c, bh = setup
        with patch.object(_routes_mod, "benchmark_history", bh), \
             patch.object(_routes_mod, "HISTORY_AVAILABLE", True):
            assert c.delete("/api/benchmark/runs/tr-1").json()["status"] == "deleted"

    def test_delete_404(self, setup):
        c, bh = setup
        with patch.object(_routes_mod, "benchmark_history", bh), \
             patch.object(_routes_mod, "HISTORY_AVAILABLE", True):
            assert c.delete("/api/benchmark/runs/fake").status_code == 404


# =============================================================================
# ROUTES -- MODEL CONFIG
# =============================================================================

class TestModelConfigEndpoints:

    @pytest.fixture
    def client(self, models_config, benchmark_config):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        app.include_router(_router)
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", models_config), \
             patch.object(_routes_mod, "BENCHMARK_CONFIG_PATH", benchmark_config):
            yield TestClient(app), models_config

    def test_get_config(self, client):
        c, p = client
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", p), \
             patch.object(_routes_mod, "_get_installed_models", return_value=["model-a"]):
            assert "config" in c.get("/api/benchmark/models/config").json()

    def test_get_roles(self, client):
        c, p = client
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", p), \
             patch.object(_routes_mod, "_get_installed_models", return_value=["model-a"]):
            roles = {r["role"]: r for r in c.get("/api/benchmark/models/config/roles").json()["roles"]}
            assert roles["general"]["primary"] == "model-a"

    def test_update_role(self, client):
        c, p = client
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", p):
            c.put("/api/benchmark/models/config/roles/general",
                  json={"primary": "new-m", "fast": "fast-m"})
            with open(p) as f:
                assert yaml.safe_load(f)["routing"]["general"]["primary"] == "new-m"

    def test_update_role_empty(self, client):
        c, p = client
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", p):
            assert c.put("/api/benchmark/models/config/roles/general",
                        json={"primary": ""}).status_code == 400

    def test_validate_warnings(self, client):
        c, p = client
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", p), \
             patch.object(_routes_mod, "_get_installed_models", return_value=["model-a"]):
            assert len(c.post("/api/benchmark/models/config/validate",
                             json={"config": {}}).json()["warnings"]) > 0

    def test_validate_all_ok(self, client):
        c, p = client
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", p), \
             patch.object(_routes_mod, "_get_installed_models",
                         return_value=["model-a", "model-b", "model-c", "embed-model"]):
            assert c.post("/api/benchmark/models/config/validate",
                         json={"config": {}}).json()["valid"] is True

    def test_installed_models(self, client):
        c, _ = client
        with patch.object(_routes_mod, "_get_installed_models", return_value=["m1", "m2"]):
            assert c.get("/api/benchmark/models/installed").json()["count"] == 2

    def test_save_full_config(self, client):
        c, p = client
        with patch.object(_routes_mod, "MODELS_CONFIG_PATH", p):
            c.put("/api/benchmark/models/config",
                  json={"config": {"routing": {"new": {"primary": "m"}}}})
            with open(p) as f:
                assert yaml.safe_load(f)["routing"]["new"]["primary"] == "m"

    def test_save_empty_rejected(self, client):
        c, _ = client
        assert c.put("/api/benchmark/models/config", json={"config": {}}).status_code == 400


# =============================================================================
# RUN STATE
# =============================================================================

class TestRunState:

    def test_initial(self):
        s = _RunState()
        assert not s.is_running() and s.status == "idle"

    def test_lifecycle(self):
        s = _RunState()
        s.start("r1")
        assert s.is_running() and s.current_run_id == "r1"
        s.finish("completed")
        assert not s.is_running()

    def test_cancel(self):
        s = _RunState()
        s.start("r1")
        s.request_cancel()
        assert s.is_cancelled()
        s.finish("cancelled")
        assert not s.is_cancelled()

    def test_broadcast(self):
        s = _RunState()
        q = asyncio.Queue()
        s.add_ws_client(q)
        s.broadcast({"type": "test"})
        assert q.get_nowait()["type"] == "test"
        s.remove_ws_client(q)
        s.broadcast({"type": "x"})
        assert q.empty()

    def test_multi_clients(self):
        s = _RunState()
        q1, q2 = asyncio.Queue(), asyncio.Queue()
        s.add_ws_client(q1)
        s.add_ws_client(q2)
        s.broadcast({"type": "m"})
        assert q1.get_nowait() and q2.get_nowait()


# =============================================================================
# THREAD SAFETY
# =============================================================================

class TestThreadSafety:

    def test_concurrent_writes(self, history):
        errors = []
        def w(i):
            try:
                history.save_run(BenchmarkRunRecord(
                    id=f"c-{i}", started_at=f"2026-03-04T{i:02d}:00:00Z", status="completed"))
            except Exception as e:
                errors.append(str(e))
        threads = [threading.Thread(target=w, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors and history.get_run_count() == 10


# =============================================================================
# SINGLETON
# =============================================================================

class TestSingleton:
    def test_exists(self):
        assert _bh_mod.benchmark_history is not None


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:

    def test_empty_detail(self, history):
        history.save_run(BenchmarkRunRecord(id="e", started_at="2026-03-04T12:00:00Z", status="completed"))
        d = history.get_run_detail("e")
        assert d["results"] == [] and d["global_ranking"] == []

    def test_none_avg(self, history):
        history.save_run(BenchmarkRunRecord(id="n", started_at="2026-03-04T12:00:00Z", status="running"))
        assert history.get_runs()[0]["avg_score"] is None

    def test_none_user_score(self, history, sample_run):
        history.save_run(sample_run)
        history.save_result(BenchmarkResultRecord(
            run_id=sample_run.id, model="m", task="t",
            score=7.0, auto_score=7.0, user_score=None, status="success"))
        assert history.get_run_detail(sample_run.id)["results"][0]["user_score"] is None

    def test_missing_tasks_comparison(self, history):
        history.save_run(BenchmarkRunRecord(id="c1", started_at="2026-03-04T12:00:00Z", status="completed"))
        history.save_run(BenchmarkRunRecord(id="c2", started_at="2026-03-04T13:00:00Z", status="completed"))
        history.save_result(BenchmarkResultRecord(run_id="c1", model="m", task="t1", score=8.0, auto_score=8.0, status="success"))
        history.save_result(BenchmarkResultRecord(run_id="c2", model="m", task="t1", score=9.0, auto_score=9.0, status="success"))
        history.save_result(BenchmarkResultRecord(run_id="c2", model="m", task="t2", score=7.0, auto_score=7.0, status="success"))
        comp = history.compare_runs(["c1", "c2"])
        assert comp.matrix["m"]["t2"][0] is None

    def test_special_model_name(self, history):
        history.save_run(BenchmarkRunRecord(id="sp", started_at="2026-03-04T12:00:00Z", status="completed"))
        history.save_result(BenchmarkResultRecord(
            run_id="sp", model="user/model:v1.2-beta", task="t1", score=7.0, auto_score=7.0, status="success"))
        assert history.get_run_detail("sp")["results"][0]["model"] == "user/model:v1.2-beta"

    def test_long_preview(self, history, sample_run):
        history.save_run(sample_run)
        history.save_result(BenchmarkResultRecord(
            run_id=sample_run.id, model="m", task="t", response_preview="x" * 1000, status="success"))
        assert len(history.get_run_detail(sample_run.id)["results"][0]["response_preview"]) == 1000

    def test_empty_keywords(self, history, sample_run):
        history.save_run(sample_run)
        history.save_result(BenchmarkResultRecord(
            run_id=sample_run.id, model="m", task="t",
            keywords_found=[], keywords_missing=[], status="success"))
        r = history.get_run_detail(sample_run.id)["results"][0]
        assert r["keywords_found"] == [] and r["keywords_missing"] == []
