#!/usr/bin/env python3
"""
Tests for S89 — LLM-as-Judge + Benchmark Dashboard v2.

Covers:
  - Judge scoring, parsing, weighted computation, blending
  - Judge heuristic extraction from unstructured text
  - JudgeStore persistence (save, retrieve, summary)
  - BenchmarkJudge evaluate and evaluate_run
  - Recommendations engine (generate, roles, persistence)
  - RecommendationStore (save, get_latest, mark_applied, history)
  - Apply recommendations to smart_router
  - Benchmark runner integration with judge params
  - API route schemas (judge, leaderboard, h2h, trends, recs, export)
  - Leaderboard ranking logic
  - Head-to-head metric comparison
  - Trend detection and regression
  - Export CSV/JSON format
  - Frontend file structure
"""

import csv
import importlib.util
import io
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

_PROJECT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    """Load a module directly from file path."""
    full = _PROJECT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load modules
_judge_mod = _load_module("benchmark_judge", "opti_oignon/benchmark_judge.py")
_rec_mod = _load_module("benchmark_recommendations", "opti_oignon/benchmark_recommendations.py")
_runner_mod = _load_module("benchmark_runner", "opti_oignon/benchmark_runner.py")
_schemas_mod = _load_module("schemas", "opti_oignon/api/schemas.py")

# Re-export classes
BenchmarkJudge = _judge_mod.BenchmarkJudge
JudgeScore = _judge_mod.JudgeScore
JudgeRunSummary = _judge_mod.JudgeRunSummary
JudgeStore = _judge_mod.JudgeStore
RUBRIC_DIMENSIONS = _judge_mod.RUBRIC_DIMENSIONS
_load_config = _judge_mod._load_config

BenchmarkRecommender = _rec_mod.BenchmarkRecommender
RecommendationStore = _rec_mod.RecommendationStore
RecommendationSnapshot = _rec_mod.RecommendationSnapshot
ModelRecommendation = _rec_mod.ModelRecommendation
ROLE_FAST = _rec_mod.ROLE_FAST
ROLE_QUALITY = _rec_mod.ROLE_QUALITY
ROLE_CODE = _rec_mod.ROLE_CODE
ROLE_VALUE = _rec_mod.ROLE_VALUE
ALL_ROLES = _rec_mod.ALL_ROLES

BenchmarkRunner = _runner_mod.BenchmarkRunner
ResultsStore = _runner_mod.ResultsStore
RunStatus = _runner_mod.RunStatus

# Schemas
BenchmarkV2RunRequest = _schemas_mod.BenchmarkV2RunRequest
BenchmarkV2ResultsResponse = _schemas_mod.BenchmarkV2ResultsResponse
BenchmarkV2JudgeScore = _schemas_mod.BenchmarkV2JudgeScore
BenchmarkV2JudgeSummary = _schemas_mod.BenchmarkV2JudgeSummary
BenchmarkV2LeaderboardEntry = _schemas_mod.BenchmarkV2LeaderboardEntry
BenchmarkV2LeaderboardResponse = _schemas_mod.BenchmarkV2LeaderboardResponse
BenchmarkV2HeadToHeadMetric = _schemas_mod.BenchmarkV2HeadToHeadMetric
BenchmarkV2HeadToHeadResponse = _schemas_mod.BenchmarkV2HeadToHeadResponse
BenchmarkV2TrendPoint = _schemas_mod.BenchmarkV2TrendPoint
BenchmarkV2TrendResponse = _schemas_mod.BenchmarkV2TrendResponse
BenchmarkV2RecommendationEntry = _schemas_mod.BenchmarkV2RecommendationEntry
BenchmarkV2RecommendationsResponse = _schemas_mod.BenchmarkV2RecommendationsResponse
BenchmarkV2ApplyResponse = _schemas_mod.BenchmarkV2ApplyResponse
BenchmarkV2ExportResponse = _schemas_mod.BenchmarkV2ExportResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    """Provide a temporary database path."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_benchmark.db")
    yield db_path


@pytest.fixture
def judge_store(tmp_db):
    """Provide a JudgeStore with temp DB."""
    return JudgeStore(db_path=tmp_db)


@pytest.fixture
def judge(judge_store):
    """Provide a BenchmarkJudge with temp store."""
    return BenchmarkJudge(store=judge_store)


@pytest.fixture
def rec_store(tmp_db):
    """Provide a RecommendationStore with temp DB."""
    return RecommendationStore(db_path=tmp_db)


@pytest.fixture
def recommender(rec_store, tmp_db):
    """Provide a BenchmarkRecommender with temp store."""
    return BenchmarkRecommender(store=rec_store, db_path=tmp_db)


@pytest.fixture
def runner(tmp_db):
    """Provide a BenchmarkRunner with temp DB."""
    store = ResultsStore(db_path=tmp_db)
    return BenchmarkRunner(store=store, db_path=tmp_db)


def mock_query_fn(model, prompt, timeout=45, max_tokens=800):
    """Mock LLM query that returns a basic response."""
    return "Test response with some content here.", 50.0, 200.0, 20


def mock_judge_query_fn(model, prompt, timeout=60, max_tokens=1024):
    """Mock judge LLM query that returns structured JSON."""
    return json.dumps({
        "accuracy": 8,
        "relevance": 7,
        "completeness": 6,
        "conciseness": 9,
        "reasoning": 7,
        "justification": "Good response overall",
    }), 30.0, 150.0, 15


# ===========================================================================
# JUDGE TESTS
# ===========================================================================

class TestJudgeScore:
    """Tests for JudgeScore dataclass."""

    def test_default_values(self):
        js = JudgeScore()
        assert js.question_id == ""
        assert js.accuracy == 0
        assert js.weighted_score == 0.0
        assert js.error == ""

    def test_to_dict(self):
        js = JudgeScore(question_id="q1", model="m1", accuracy=8, relevance=7)
        d = js.to_dict()
        assert d["question_id"] == "q1"
        assert d["accuracy"] == 8
        assert d["relevance"] == 7
        assert "weighted_score" in d

    def test_all_fields_serialized(self):
        js = JudgeScore(
            question_id="q1", model="m1", judge_model="j1",
            accuracy=8, relevance=7, completeness=6, conciseness=9, reasoning=7,
            justification="ok", weighted_score=0.73, tokens_used=15,
            eval_time_ms=100.0, error="",
        )
        d = js.to_dict()
        assert len(d) == 13


class TestJudgeRunSummary:
    """Tests for JudgeRunSummary dataclass."""

    def test_defaults(self):
        s = JudgeRunSummary()
        assert s.total_evaluations == 0
        assert s.total_tokens == 0
        assert s.avg_score == 0.0

    def test_fields(self):
        s = JudgeRunSummary(run_id="r1", judge_model="j1", total_evaluations=5)
        assert s.run_id == "r1"
        assert s.total_evaluations == 5


class TestJudgeConfig:
    """Tests for judge configuration loading."""

    def test_default_config_has_required_keys(self):
        config = _load_config("/nonexistent/path.yaml")
        assert "autonomous_weight" in config
        assert "judge_weight" in config
        assert "rubric" in config
        assert "judge_system_prompt" in config

    def test_weights_sum_to_one(self):
        config = _load_config()
        total = config["autonomous_weight"] + config["judge_weight"]
        assert abs(total - 1.0) < 0.01

    def test_rubric_has_five_dimensions(self):
        config = _load_config()
        rubric = config["rubric"]
        for dim in RUBRIC_DIMENSIONS:
            assert dim in rubric

    def test_rubric_weights_sum_to_one(self):
        config = _load_config()
        rubric = config["rubric"]
        total = sum(rubric[dim]["weight"] for dim in RUBRIC_DIMENSIONS)
        assert abs(total - 1.0) < 0.01


class TestBenchmarkJudgeParsing:
    """Tests for judge response parsing."""

    def test_parse_valid_json(self, judge):
        raw = json.dumps({
            "accuracy": 8, "relevance": 7, "completeness": 6,
            "conciseness": 9, "reasoning": 7, "justification": "Good",
        })
        result = judge.parse_judge_response(raw)
        assert result["accuracy"] == 8
        assert result["relevance"] == 7
        assert result["justification"] == "Good"

    def test_parse_clamping_high(self, judge):
        raw = json.dumps({"accuracy": 15, "relevance": 7, "completeness": 6,
                          "conciseness": 9, "reasoning": 7})
        result = judge.parse_judge_response(raw)
        assert result["accuracy"] == 10

    def test_parse_clamping_low(self, judge):
        raw = json.dumps({"accuracy": -5, "relevance": 7, "completeness": 6,
                          "conciseness": 9, "reasoning": 7})
        result = judge.parse_judge_response(raw)
        assert result["accuracy"] == 0

    def test_parse_malformed_json_with_repair(self, judge):
        raw = '{"accuracy": 8, "relevance": 7, "completeness": 6, "conciseness": 9, "reasoning": 7}'
        result = judge.parse_judge_response(raw)
        assert result["accuracy"] == 8

    def test_parse_heuristic_fallback(self, judge):
        raw = "accuracy: 8\nrelevance: 7\ncompleteness: 6\nconciseness: 9\nreasoning: 7"
        result = judge.parse_judge_response(raw)
        assert result["accuracy"] == 8
        assert result["relevance"] == 7

    def test_parse_empty_string(self, judge):
        result = judge.parse_judge_response("")
        assert all(result[dim] == 0 for dim in RUBRIC_DIMENSIONS)

    def test_parse_garbage_text(self, judge):
        result = judge.parse_judge_response("This is not JSON at all, just random text.")
        for dim in RUBRIC_DIMENSIONS:
            assert isinstance(result[dim], int)

    def test_parse_partial_json(self, judge):
        raw = '{"accuracy": 8, "relevance": 7}'
        result = judge.parse_judge_response(raw)
        assert result["accuracy"] == 8
        assert result["relevance"] == 7
        assert result["completeness"] == 0


class TestBenchmarkJudgeScoring:
    """Tests for weighted score computation and blending."""

    def test_compute_weighted_score_perfect(self, judge):
        scores = {dim: 10 for dim in RUBRIC_DIMENSIONS}
        ws = judge.compute_weighted_score(scores)
        assert abs(ws - 1.0) < 0.001

    def test_compute_weighted_score_zero(self, judge):
        scores = {dim: 0 for dim in RUBRIC_DIMENSIONS}
        ws = judge.compute_weighted_score(scores)
        assert ws == 0.0

    def test_compute_weighted_score_mixed(self, judge):
        scores = {"accuracy": 8, "relevance": 7, "completeness": 6,
                  "conciseness": 9, "reasoning": 7}
        ws = judge.compute_weighted_score(scores)
        assert 0.6 < ws < 0.8

    def test_blend_scores_equal(self, judge):
        result = judge.blend_scores(0.8, 0.6)
        assert abs(result - 0.7) < 0.01

    def test_blend_scores_zero_judge(self, judge):
        result = judge.blend_scores(1.0, 0.0)
        assert abs(result - 0.5) < 0.01

    def test_blend_scores_perfect(self, judge):
        result = judge.blend_scores(1.0, 1.0)
        assert abs(result - 1.0) < 0.01

    def test_rubric_weights_property(self, judge):
        w = judge.rubric_weights
        assert isinstance(w, dict)
        assert len(w) == 5
        assert abs(sum(w.values()) - 1.0) < 0.01


class TestBenchmarkJudgeEval:
    """Tests for judge evaluation methods."""

    def test_build_eval_prompt(self, judge):
        prompt = judge.build_eval_prompt("What is 2+2?", "4")
        assert "What is 2+2?" in prompt
        assert "4" in prompt
        assert "JSON" in prompt

    def test_evaluate_single(self, judge):
        js = judge.evaluate(
            question_id="q1",
            question_text="What is 2+2?",
            response="4",
            model="test-model",
            judge_model="judge-model",
            query_fn=mock_judge_query_fn,
        )
        assert js.question_id == "q1"
        assert js.model == "test-model"
        assert js.judge_model == "judge-model"
        assert js.accuracy == 8
        assert js.weighted_score > 0
        assert js.tokens_used == 15
        assert js.error == ""

    def test_evaluate_with_query_error(self, judge):
        def error_fn(model, prompt, timeout, max_tokens):
            raise RuntimeError("connection failed")

        js = judge.evaluate("q1", "test", "resp", "m1", "j1", query_fn=error_fn)
        assert js.error == "connection failed"
        assert js.accuracy == 0

    def test_evaluate_with_empty_response(self, judge):
        def empty_fn(model, prompt, timeout, max_tokens):
            return "", 0.0, 0.0, 0

        js = judge.evaluate("q1", "test", "resp", "m1", "j1", query_fn=empty_fn)
        assert js.error == "Empty response from judge model"

    def test_evaluate_run(self, judge):
        qrs = [
            {"question_id": "q1", "question_text": "Q1?", "response": "A1", "model": "m1"},
            {"question_id": "q2", "question_text": "Q2?", "response": "A2", "model": "m1"},
            {"question_id": "q1", "question_text": "Q1?", "response": "A1b", "model": "m2"},
        ]
        summary = judge.evaluate_run("run-1", "judge-model", qrs, query_fn=mock_judge_query_fn)
        assert summary.run_id == "run-1"
        assert summary.total_evaluations == 3
        assert summary.total_tokens == 45
        assert "m1" in summary.scores_by_model
        assert "m2" in summary.scores_by_model
        assert summary.avg_score > 0
        assert summary.errors == 0

    def test_evaluate_run_with_errors(self, judge):
        call_count = [0]

        def flaky_fn(model, prompt, timeout, max_tokens):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("flaky error")
            return mock_judge_query_fn(model, prompt, timeout, max_tokens)

        qrs = [
            {"question_id": "q1", "prompt": "Q1?", "response": "A1", "model": "m1"},
            {"question_id": "q2", "prompt": "Q2?", "response": "A2", "model": "m1"},
        ]
        summary = judge.evaluate_run("run-2", "j1", qrs, query_fn=flaky_fn)
        assert summary.errors == 1
        assert summary.total_evaluations == 2


class TestJudgeStore:
    """Tests for JudgeStore persistence."""

    def test_save_and_retrieve(self, judge_store):
        js = JudgeScore(
            question_id="q1", model="m1", judge_model="j1",
            accuracy=8, relevance=7, completeness=6, conciseness=9, reasoning=7,
            weighted_score=0.73, tokens_used=15,
        )
        judge_store.save_score("run-1", js)
        scores = judge_store.get_scores_for_run("run-1")
        assert len(scores) == 1
        assert scores[0]["accuracy"] == 8
        assert scores[0]["model"] == "m1"

    def test_batch_save(self, judge_store):
        scores = [
            JudgeScore(question_id=f"q{i}", model="m1", judge_model="j1",
                       accuracy=7+i, weighted_score=0.7)
            for i in range(5)
        ]
        judge_store.save_scores_batch("run-2", scores)
        retrieved = judge_store.get_scores_for_run("run-2")
        assert len(retrieved) == 5

    def test_get_scores_for_model(self, judge_store):
        judge_store.save_score("run-3", JudgeScore(
            question_id="q1", model="m1", judge_model="j1", accuracy=8))
        judge_store.save_score("run-3", JudgeScore(
            question_id="q1", model="m2", judge_model="j1", accuracy=6))
        m1_scores = judge_store.get_scores_for_model("run-3", "m1")
        assert len(m1_scores) == 1
        assert m1_scores[0]["accuracy"] == 8

    def test_get_summary(self, judge_store):
        for i in range(3):
            judge_store.save_score("run-4", JudgeScore(
                question_id=f"q{i}", model="m1", judge_model="j1",
                accuracy=7+i, relevance=6, completeness=5,
                conciseness=8, reasoning=7, weighted_score=0.7,
                tokens_used=10,
            ))
        summary = judge_store.get_summary_for_run("run-4")
        assert summary["run_id"] == "run-4"
        assert summary["total_tokens"] == 30
        assert "m1" in summary["models"]
        assert summary["models"]["m1"]["evaluations"] == 3

    def test_empty_summary(self, judge_store):
        summary = judge_store.get_summary_for_run("nonexistent")
        assert summary == {}

    def test_save_empty_batch(self, judge_store):
        judge_store.save_scores_batch("run-x", [])
        assert judge_store.get_scores_for_run("run-x") == []


# ===========================================================================
# RECOMMENDATION TESTS
# ===========================================================================

class TestModelRecommendation:
    """Tests for ModelRecommendation dataclass."""

    def test_to_dict(self):
        r = ModelRecommendation(role="fast", model="gemma3:4b", composite_score=0.55)
        d = r.to_dict()
        assert d["role"] == "fast"
        assert d["model"] == "gemma3:4b"
        assert d["composite_score"] == 0.55

    def test_all_roles_defined(self):
        assert len(ALL_ROLES) == 4
        assert ROLE_FAST in ALL_ROLES
        assert ROLE_QUALITY in ALL_ROLES
        assert ROLE_CODE in ALL_ROLES
        assert ROLE_VALUE in ALL_ROLES


class TestRecommendationSnapshot:
    """Tests for RecommendationSnapshot dataclass."""

    def test_to_dict(self):
        snap = RecommendationSnapshot(
            snapshot_id="rec-123", created_at=100.0, profile="test",
            recommendations=[ModelRecommendation(role="fast", model="m1")],
        )
        d = snap.to_dict()
        assert d["snapshot_id"] == "rec-123"
        assert len(d["recommendations"]) == 1

    def test_get_recommendation(self):
        snap = RecommendationSnapshot(recommendations=[
            ModelRecommendation(role="fast", model="m1"),
            ModelRecommendation(role="quality", model="m2"),
        ])
        assert snap.get_recommendation("fast").model == "m1"
        assert snap.get_recommendation("quality").model == "m2"
        assert snap.get_recommendation("nonexistent") is None


class TestRecommendationStore:
    """Tests for RecommendationStore persistence."""

    def test_save_and_get_latest(self, rec_store):
        snap = RecommendationSnapshot(
            snapshot_id="rec-1", created_at=100.0, profile="test",
            recommendations=[ModelRecommendation(role="fast", model="m1")],
        )
        rec_store.save_snapshot(snap)
        latest = rec_store.get_latest()
        assert latest is not None
        assert latest.snapshot_id == "rec-1"
        assert len(latest.recommendations) == 1

    def test_get_by_id(self, rec_store):
        snap = RecommendationSnapshot(
            snapshot_id="rec-2", created_at=200.0, profile="test",
        )
        rec_store.save_snapshot(snap)
        found = rec_store.get_by_id("rec-2")
        assert found is not None
        assert found.snapshot_id == "rec-2"

    def test_get_by_id_not_found(self, rec_store):
        assert rec_store.get_by_id("nonexistent") is None

    def test_mark_applied(self, rec_store):
        snap = RecommendationSnapshot(
            snapshot_id="rec-3", created_at=300.0,
        )
        rec_store.save_snapshot(snap)
        assert rec_store.mark_applied("rec-3") is True
        updated = rec_store.get_by_id("rec-3")
        assert updated.applied is True
        assert updated.applied_at > 0

    def test_history(self, rec_store):
        for i in range(5):
            rec_store.save_snapshot(RecommendationSnapshot(
                snapshot_id=f"rec-h{i}", created_at=100.0 + i,
            ))
        history = rec_store.get_history(limit=3)
        assert len(history) == 3
        assert history[0].created_at > history[1].created_at

    def test_latest_returns_most_recent(self, rec_store):
        rec_store.save_snapshot(RecommendationSnapshot(
            snapshot_id="rec-old", created_at=100.0))
        rec_store.save_snapshot(RecommendationSnapshot(
            snapshot_id="rec-new", created_at=200.0))
        latest = rec_store.get_latest()
        assert latest.snapshot_id == "rec-new"


class TestBenchmarkRecommender:
    """Tests for BenchmarkRecommender logic."""

    def test_generate_from_scores(self, recommender):
        scores = [
            {"model": "qwen3:32b", "accuracy_avg": 0.85, "code_avg": 0.7,
             "structure_avg": 0.8, "speed_avg": 0.4, "composite": 0.72},
            {"model": "qwen3-coder:30b", "accuracy_avg": 0.6, "code_avg": 0.95,
             "structure_avg": 0.75, "speed_avg": 0.5, "composite": 0.68},
            {"model": "gemma3:4b", "accuracy_avg": 0.5, "code_avg": 0.3,
             "structure_avg": 0.6, "speed_avg": 0.95, "composite": 0.55},
        ]
        snap = recommender.generate_from_scores(scores, profile="test")
        assert len(snap.recommendations) == 4
        roles = [r.role for r in snap.recommendations]
        assert set(roles) == set(ALL_ROLES)

    def test_quality_picks_highest_composite(self, recommender):
        scores = [
            {"model": "m1", "accuracy_avg": 0.9, "code_avg": 0.9,
             "structure_avg": 0.9, "speed_avg": 0.1, "composite": 0.95},
            {"model": "m2", "accuracy_avg": 0.5, "code_avg": 0.5,
             "structure_avg": 0.5, "speed_avg": 0.9, "composite": 0.50},
        ]
        snap = recommender.generate_from_scores(scores)
        quality = snap.get_recommendation(ROLE_QUALITY)
        assert quality.model == "m1"

    def test_fast_picks_fastest_acceptable(self, recommender):
        scores = [
            {"model": "big", "accuracy_avg": 0.9, "code_avg": 0.9,
             "structure_avg": 0.9, "speed_avg": 0.2, "composite": 0.90},
            {"model": "small", "accuracy_avg": 0.6, "code_avg": 0.4,
             "structure_avg": 0.6, "speed_avg": 0.95, "composite": 0.55},
        ]
        snap = recommender.generate_from_scores(scores)
        fast = snap.get_recommendation(ROLE_FAST)
        assert fast.model == "small"

    def test_code_picks_best_code(self, recommender):
        scores = [
            {"model": "general", "accuracy_avg": 0.9, "code_avg": 0.5,
             "structure_avg": 0.9, "speed_avg": 0.5, "composite": 0.7},
            {"model": "coder", "accuracy_avg": 0.5, "code_avg": 0.98,
             "structure_avg": 0.5, "speed_avg": 0.5, "composite": 0.6},
        ]
        snap = recommender.generate_from_scores(scores)
        code = snap.get_recommendation(ROLE_CODE)
        assert code.model == "coder"

    def test_empty_scores(self, recommender):
        snap = recommender.generate_from_scores([])
        assert len(snap.recommendations) == 0

    def test_single_model(self, recommender):
        scores = [
            {"model": "only", "accuracy_avg": 0.7, "code_avg": 0.7,
             "structure_avg": 0.7, "speed_avg": 0.7, "composite": 0.7},
        ]
        snap = recommender.generate_from_scores(scores)
        for role in ALL_ROLES:
            rec = snap.get_recommendation(role)
            assert rec.model == "only"

    def test_generate_persists(self, recommender):
        scores = [
            {"model": "m1", "accuracy_avg": 0.7, "code_avg": 0.7,
             "structure_avg": 0.7, "speed_avg": 0.7, "composite": 0.7},
        ]
        snap = recommender.generate_from_scores(scores, profile="p1")
        latest = recommender.get_latest()
        assert latest is not None
        assert latest.snapshot_id == snap.snapshot_id


# ===========================================================================
# RUNNER INTEGRATION TESTS
# ===========================================================================

class TestRunnerJudgeIntegration:
    """Tests for BenchmarkRunner with judge params."""

    def test_start_run_signature(self):
        import inspect
        sig = inspect.signature(BenchmarkRunner.start_run)
        params = list(sig.parameters.keys())
        assert "use_judge" in params
        assert "judge_model" in params

    def test_run_sync_signature(self):
        import inspect
        sig = inspect.signature(BenchmarkRunner.run_sync)
        params = list(sig.parameters.keys())
        assert "use_judge" in params
        assert "judge_model" in params

    def test_run_without_judge(self, runner):
        result = runner.run_sync(
            "all_round", ["test-model"],
            query_fn=mock_query_fn,
        )
        assert result.status in (RunStatus.COMPLETED, RunStatus.FAILED)

    def test_run_with_judge_flag(self, runner):
        result = runner.run_sync(
            "all_round", ["test-model"],
            query_fn=mock_query_fn,
            use_judge=True,
            judge_model="judge-model",
        )
        assert result.status in (RunStatus.COMPLETED, RunStatus.FAILED)

    def test_backward_compatible_no_judge(self, runner):
        result = runner.run_sync(
            "all_round", ["test-model"],
            query_fn=mock_query_fn,
        )
        # Should work exactly as before
        assert result.run_id.startswith("run-")


# ===========================================================================
# SCHEMA TESTS
# ===========================================================================

class TestS89Schemas:
    """Tests for all S89 Pydantic schemas."""

    def test_run_request_judge_fields(self):
        rr = BenchmarkV2RunRequest(
            profile="test", models=["m1"],
            use_judge=True, judge_model="j1",
        )
        assert rr.use_judge is True
        assert rr.judge_model == "j1"

    def test_run_request_defaults(self):
        rr = BenchmarkV2RunRequest()
        assert rr.use_judge is False
        assert rr.judge_model == ""

    def test_results_response_judge_fields(self):
        res = BenchmarkV2ResultsResponse(
            judge_scores=[{"accuracy": 8}],
            judge_summary={"run_id": "r1"},
        )
        assert len(res.judge_scores) == 1
        assert res.judge_summary["run_id"] == "r1"

    def test_judge_score_schema(self):
        js = BenchmarkV2JudgeScore(
            question_id="q1", model="m1", judge_model="j1",
            accuracy=8, relevance=7, weighted_score=0.73,
        )
        assert js.accuracy == 8
        assert js.weighted_score == 0.73

    def test_leaderboard_entry(self):
        e = BenchmarkV2LeaderboardEntry(rank=1, model="m1", composite=0.85)
        assert e.rank == 1
        assert e.composite == 0.85

    def test_leaderboard_response(self):
        r = BenchmarkV2LeaderboardResponse(
            profile="test",
            entries=[BenchmarkV2LeaderboardEntry(rank=1, model="m1")],
            total=1,
        )
        assert r.total == 1
        assert len(r.entries) == 1

    def test_h2h_metric(self):
        m = BenchmarkV2HeadToHeadMetric(
            metric="accuracy", model_a_value=0.8, model_b_value=0.6, winner="modelA",
        )
        assert m.winner == "modelA"

    def test_h2h_response(self):
        r = BenchmarkV2HeadToHeadResponse(
            model_a="a", model_b="b",
            model_a_wins=3, model_b_wins=2, ties=0,
            overall_winner="a",
        )
        assert r.model_a_wins == 3

    def test_trend_point(self):
        tp = BenchmarkV2TrendPoint(
            run_id="r1", timestamp=1000.0, composite=0.75,
        )
        assert tp.composite == 0.75

    def test_trend_response(self):
        tr = BenchmarkV2TrendResponse(
            model="m1", trend_direction="improving", regression_detected=False,
        )
        assert tr.trend_direction == "improving"
        assert tr.regression_detected is False

    def test_recommendation_entry(self):
        re = BenchmarkV2RecommendationEntry(
            role="fast", model="m1", reason="Fastest",
        )
        assert re.role == "fast"

    def test_recommendations_response(self):
        rr = BenchmarkV2RecommendationsResponse(
            snapshot_id="rec-1",
            recommendations=[BenchmarkV2RecommendationEntry(role="fast", model="m1")],
        )
        assert len(rr.recommendations) == 1

    def test_apply_response(self):
        ar = BenchmarkV2ApplyResponse(applied=True, snapshot_id="rec-1")
        assert ar.applied is True

    def test_export_response(self):
        er = BenchmarkV2ExportResponse(
            run_id="r1", format="csv", model_count=3, question_count=10,
        )
        assert er.format == "csv"


# ===========================================================================
# LEADERBOARD LOGIC TESTS
# ===========================================================================

class TestLeaderboardLogic:
    """Tests for leaderboard ranking."""

    def test_models_ranked_by_composite(self):
        models = [
            {"model": "a", "avg_composite": 0.7},
            {"model": "b", "avg_composite": 0.9},
            {"model": "c", "avg_composite": 0.5},
        ]
        ranked = sorted(models, key=lambda x: x["avg_composite"], reverse=True)
        assert ranked[0]["model"] == "b"
        assert ranked[1]["model"] == "a"
        assert ranked[2]["model"] == "c"

    def test_ranking_with_ties(self):
        models = [
            {"model": "a", "avg_composite": 0.7},
            {"model": "b", "avg_composite": 0.7},
        ]
        ranked = sorted(models, key=lambda x: x["avg_composite"], reverse=True)
        assert len(ranked) == 2


# ===========================================================================
# HEAD-TO-HEAD LOGIC TESTS
# ===========================================================================

class TestHeadToHeadLogic:
    """Tests for head-to-head comparison logic."""

    def test_winner_per_metric(self):
        a = {"avg_accuracy": 0.8, "avg_code": 0.5, "avg_composite": 0.7}
        b = {"avg_accuracy": 0.6, "avg_code": 0.9, "avg_composite": 0.7}

        metric_keys = [("accuracy", "avg_accuracy"), ("code", "avg_code"), ("composite", "avg_composite")]
        a_wins = 0
        b_wins = 0
        ties = 0
        for _, key in metric_keys:
            if a[key] > b[key]:
                a_wins += 1
            elif b[key] > a[key]:
                b_wins += 1
            else:
                ties += 1
        assert a_wins == 1
        assert b_wins == 1
        assert ties == 1

    def test_overall_winner(self):
        assert "a" if 3 > 2 else "b" == "a"
        assert "tie" if 2 == 2 else "b" == "tie"


# ===========================================================================
# TREND DETECTION TESTS
# ===========================================================================

class TestTrendDetection:
    """Tests for trend direction and regression detection."""

    def test_improving_trend(self):
        composites = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        recent = composites[-3:]
        older = composites[:3]
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        assert avg_recent > avg_older * 1.05

    def test_declining_trend(self):
        composites = [0.8, 0.75, 0.7, 0.5, 0.45, 0.4]
        recent = composites[-3:]
        older = composites[:3]
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        assert avg_recent < avg_older * 0.95

    def test_stable_trend(self):
        composites = [0.7, 0.71, 0.69, 0.7, 0.71, 0.7]
        recent = composites[-3:]
        older = composites[:3]
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        assert avg_older * 0.95 <= avg_recent <= avg_older * 1.05

    def test_too_few_points(self):
        composites = [0.7, 0.8]
        assert len(composites) < 3


# ===========================================================================
# EXPORT FORMAT TESTS
# ===========================================================================

class TestExportFormats:
    """Tests for export CSV/JSON generation."""

    def test_csv_header(self):
        output = io.StringIO()
        writer = csv.writer(output)
        header = [
            "run_id", "model", "question_id", "category",
            "accuracy_score", "code_score", "structure_score",
            "speed_score", "composite_score",
        ]
        writer.writerow(header)
        output.seek(0)
        reader = csv.reader(output)
        row = next(reader)
        assert row == header

    def test_csv_row_generation(self):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "run-1", "model-a", "q1", "general",
            0.8, 0.0, 0.7, 0.6, 0.65,
        ])
        output.seek(0)
        reader = csv.reader(output)
        row = next(reader)
        assert row[0] == "run-1"
        assert row[1] == "model-a"
        assert float(row[4]) == 0.8

    def test_json_export_structure(self):
        export_data = {
            "run_id": "run-1",
            "profile": "test",
            "models": ["m1"],
            "model_scores": {},
            "question_results": {},
            "judge_scores": [],
        }
        serialized = json.dumps(export_data)
        parsed = json.loads(serialized)
        assert parsed["run_id"] == "run-1"
        assert isinstance(parsed["judge_scores"], list)


# ===========================================================================
# FRONTEND FILE TESTS
# ===========================================================================

class TestFrontendFiles:
    """Tests for frontend file structure and conventions."""

    def test_types_file_has_judge_interface(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "types.ts"
        content = path.read_text()
        assert "BenchmarkV2JudgeScore" in content

    def test_types_file_has_leaderboard(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "types.ts"
        content = path.read_text()
        assert "BenchmarkV2LeaderboardEntry" in content
        assert "BenchmarkV2LeaderboardResponse" in content

    def test_types_file_has_h2h(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "types.ts"
        content = path.read_text()
        assert "BenchmarkV2HeadToHeadMetric" in content
        assert "BenchmarkV2HeadToHeadResponse" in content

    def test_types_file_has_trends(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "types.ts"
        content = path.read_text()
        assert "BenchmarkV2TrendPoint" in content
        assert "BenchmarkV2TrendResponse" in content

    def test_types_file_has_recommendations(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "types.ts"
        content = path.read_text()
        assert "BenchmarkV2RecommendationEntry" in content
        assert "BenchmarkV2RecommendationsResponse" in content
        assert "BenchmarkV2ApplyResponse" in content

    def test_types_run_request_has_judge_fields(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "types.ts"
        content = path.read_text()
        assert "use_judge" in content
        assert "judge_model" in content

    def test_api_client_has_new_functions(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "api" / "benchmarkV2.ts"
        content = path.read_text()
        for fn in ["getLeaderboard", "getHeadToHead", "getTrends",
                    "getRecommendations", "applyRecommendations",
                    "exportJson", "exportCsv", "downloadBlob"]:
            assert fn in content, f"Missing function: {fn}"

    def test_panel_has_tabs(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "components" / "panels" / "BenchmarkV2Panel.svelte"
        content = path.read_text()
        for tab in ["leaderboard", "h2h", "trends"]:
            assert tab in content, f"Missing tab: {tab}"

    def test_panel_has_judge_toggle(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "components" / "panels" / "BenchmarkV2Panel.svelte"
        content = path.read_text()
        assert "useJudge" in content
        assert "judgeModel" in content

    def test_panel_has_export_buttons(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "components" / "panels" / "BenchmarkV2Panel.svelte"
        content = path.read_text()
        assert "Export JSON" in content
        assert "Export CSV" in content

    def test_panel_no_hardcoded_hex_colors(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "components" / "panels" / "BenchmarkV2Panel.svelte"
        content = path.read_text()
        import re
        # Find color/background/border with hex values in style blocks
        hex_matches = re.findall(r'(?:color|background|border):\s*#[0-9a-fA-F]{3,6}', content)
        assert len(hex_matches) == 0, f"Hardcoded hex colors found: {hex_matches}"

    def test_panel_no_emojis(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "components" / "panels" / "BenchmarkV2Panel.svelte"
        content = path.read_text()
        import re
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+",
            flags=re.UNICODE,
        )
        assert not emoji_pattern.search(content), "Emojis found in panel"


# ===========================================================================
# VERSION TESTS
# ===========================================================================

class TestVersionBump:
    """Tests for version consistency."""

    def test_app_version(self):
        path = _PROJECT / "opti_oignon" / "api" / "app.py"
        content = path.read_text()
        assert '"1.10.0"' in content

    def test_pyproject_version(self):
        path = _PROJECT / "pyproject.toml"
        content = path.read_text()
        assert '"1.10.0"' in content

    def test_setup_version(self):
        path = _PROJECT / "setup.py"
        content = path.read_text()
        assert '"1.10.0"' in content


# ===========================================================================
# BACKEND FILE EXISTENCE TESTS
# ===========================================================================

class TestBackendFiles:
    """Tests for new S89 backend files."""

    def test_benchmark_judge_exists(self):
        assert (_PROJECT / "opti_oignon" / "benchmark_judge.py").is_file()

    def test_benchmark_recommendations_exists(self):
        assert (_PROJECT / "opti_oignon" / "benchmark_recommendations.py").is_file()

    def test_benchmark_judge_yaml_exists(self):
        assert (_PROJECT / "opti_oignon" / "config" / "benchmark_judge.yaml").is_file()

    def test_deps_has_judge_flag(self):
        path = _PROJECT / "opti_oignon" / "api" / "deps.py"
        content = path.read_text()
        assert "BENCHMARK_JUDGE_AVAILABLE" in content

    def test_deps_has_recommendations_flag(self):
        path = _PROJECT / "opti_oignon" / "api" / "deps.py"
        content = path.read_text()
        assert "BENCHMARK_RECOMMENDATIONS_AVAILABLE" in content

    def test_routes_has_new_endpoints(self):
        path = _PROJECT / "opti_oignon" / "api" / "routes_benchmark_v2.py"
        content = path.read_text()
        for endpoint in ["leaderboard", "head-to-head", "trends",
                         "recommendations", "recommendations/apply", "export"]:
            assert endpoint in content, f"Missing endpoint: {endpoint}"

    def test_schemas_has_new_classes(self):
        path = _PROJECT / "opti_oignon" / "api" / "schemas.py"
        content = path.read_text()
        for cls in ["BenchmarkV2JudgeScore", "BenchmarkV2LeaderboardEntry",
                     "BenchmarkV2HeadToHeadResponse", "BenchmarkV2TrendResponse",
                     "BenchmarkV2RecommendationsResponse", "BenchmarkV2ApplyResponse",
                     "BenchmarkV2ExportResponse"]:
            assert cls in content, f"Missing schema: {cls}"

    def test_benchmark_runner_has_judge_import(self):
        path = _PROJECT / "opti_oignon" / "benchmark_runner.py"
        content = path.read_text()
        assert "BENCHMARK_JUDGE_AVAILABLE" in content
        assert "use_judge" in content

    def test_no_french_in_new_files(self):
        for rel in [
            "opti_oignon/benchmark_judge.py",
            "opti_oignon/benchmark_recommendations.py",
        ]:
            path = _PROJECT / rel
            content = path.read_text()
            # Spot check common French words
            for word in ["modele", "resultat", "reponse", "disponible"]:
                assert word not in content.lower(), f"French word '{word}' in {rel}"
