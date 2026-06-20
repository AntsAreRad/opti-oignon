#!/usr/bin/env python3
"""
S193 F6b — benchmark periphery fixes (history/recommendations/custom_profiles/
auto_trigger) and the performance-benchmark exposure.

Covers:
  - PRF-01: the perf micro-bench runner is no longer shadowed by the S88 runner
  - BMK-11: recommendations aggregate only over the latest N completed runs
  - BMK-12: a crashing run is marked FAILED, not left zombie/busy
  - BMK-13: custom-profile numeric fields are validated (create + update)
  - BMK-14: hand-edited builtin-id custom profiles are ignored at load
  - BMK-16: a skipped auto-trigger keeps the change for re-detection
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    full = _PROJECT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_runner_mod = _load_module("s193b_benchmark_runner", "opti_oignon/benchmark_runner.py")
_rec_mod = _load_module("s193b_benchmark_recommendations", "opti_oignon/benchmark_recommendations.py")
_cp_mod = _load_module("s193b_benchmark_custom_profiles", "opti_oignon/benchmark_custom_profiles.py")
_at_mod = _load_module("s193b_benchmark_auto_trigger", "opti_oignon/benchmark_auto_trigger.py")
_perf_mod = _load_module("s193b_performance_benchmark", "opti_oignon/performance_benchmark.py")


# ---------------------------------------------------------------------------
# PRF-01 — perf runner no longer shadowed
# ---------------------------------------------------------------------------

class TestPRF01PerfRunnerExposure:
    def test_deps_renames_perf_runner(self):
        src = (_PROJECT / "opti_oignon/api/deps.py").read_text()
        assert "S193 PRF-01" in src
        assert 'perf_benchmark_runner = _LazyAttr("opti_oignon.performance_benchmark"' in src

    def test_routes_health_uses_renamed_runner(self):
        src = (_PROJECT / "opti_oignon/api/routes_health.py").read_text()
        assert "perf_benchmark_runner.run_all(" in src
        assert "perf_benchmark_runner.run(name" in src
        assert "    perf_benchmark_runner,\n" in src

    def test_two_runners_are_distinct_apis(self):
        # The perf runner exposes run_all/run; the S88 runner exposes start_run.
        assert hasattr(_perf_mod.benchmark_runner, "run_all")
        assert hasattr(_perf_mod.benchmark_runner, "run")
        assert hasattr(_runner_mod.benchmark_runner, "start_run")
        assert not hasattr(_runner_mod.benchmark_runner, "run_all")


# ---------------------------------------------------------------------------
# BMK-11 — recommendation scope and source attribution
# ---------------------------------------------------------------------------

class TestBMK11RecommendationScope:
    def _seed(self, db_path):
        store = _runner_mod.ResultsStore(db_path)
        RunResult = _runner_mod.RunResult
        ModelScore = _runner_mod.ModelScore
        RunStatus = _runner_mod.RunStatus
        # Old run: model X looks great (stale/garbage); recent run: model Y wins.
        store.save_run(RunResult(
            run_id="old-1", profile="p", models=["X"],
            status=RunStatus.COMPLETED, started_at=1000.0,
            model_scores={"X": ModelScore(model="X", composite=0.99,
                                          accuracy_avg=0.99, speed_avg=0.99)},
        ))
        store.save_run(RunResult(
            run_id="new-1", profile="p", models=["Y"],
            status=RunStatus.COMPLETED, started_at=2000.0,
            model_scores={"Y": ModelScore(model="Y", composite=0.80,
                                          accuracy_avg=0.80, speed_avg=0.80)},
        ))
        return store

    def test_history_limit_scopes_to_recent_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "bench.db")
            self._seed(db)
            rec = _rec_mod.BenchmarkRecommender(db_path=db)
            snap = rec.generate_from_history(profile="p", limit=1)
            assert snap is not None
            # Only the most recent run contributes; the stale "X" run is excluded.
            assert snap.source_run_ids == ["new-1"]
            models = {r.model for r in snap.recommendations}
            assert models == {"Y"}

    def test_history_wider_limit_includes_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "bench.db")
            self._seed(db)
            rec = _rec_mod.BenchmarkRecommender(db_path=db)
            snap = rec.generate_from_history(profile="p", limit=5)
            assert set(snap.source_run_ids) == {"old-1", "new-1"}


# ---------------------------------------------------------------------------
# BMK-12 — crashing run is FAILED, not zombie/busy
# ---------------------------------------------------------------------------

_BAD_QUESTIONS = """
general_knowledge:
  - id: q1
    prompt: "Q?"
    expected: ["A"]
    scoring: exact
"""

# expected_length_range with a single element crashes the structural eval's
# tuple unpack inside the run body.
_BAD_PROFILES = """
profiles:
  bad:
    name: Bad
    categories: [general_knowledge]
    weight_preset: balanced
    expected_length_range: [100]
weight_presets:
  balanced:
    accuracy: 0.35
    code: 0.25
    structure: 0.25
    speed: 0.15
"""


class TestBMK12CrashGuard:
    def test_crash_marks_failed_and_clears_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            qp = Path(tmp) / "q.yaml"
            pp = Path(tmp) / "p.yaml"
            qp.write_text(_BAD_QUESTIONS, encoding="utf-8")
            pp.write_text(_BAD_PROFILES, encoding="utf-8")
            ev = _runner_mod.benchmark_evaluator.__class__(
                questions_path=qp, profiles_path=pp,
            )
            store = _runner_mod.ResultsStore(os.path.join(tmp, "b.db"))
            runner = _runner_mod.BenchmarkRunner(evaluator=ev, store=store)
            result = runner.run_sync(
                "bad", ["m1"],
                query_fn=lambda m, p, t, mt: ("hello world", 100.0, 1000.0, 5),
            )
            assert result.status == _runner_mod.RunStatus.FAILED
            assert result.error
            # The runner must not be stuck busy (would 409-lock the v2 endpoint).
            assert runner.is_busy is False


# ---------------------------------------------------------------------------
# BMK-13 / BMK-14 — custom profile validation and load guard
# ---------------------------------------------------------------------------

class TestBMK13Validation:
    def _store(self, tmp):
        return _cp_mod.CustomProfileStore(path=Path(tmp) / "cp.yaml")

    def test_rejects_zero_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self._store(tmp)
            with pytest.raises(ValueError):
                s.create("P1", timeout=0)

    def test_rejects_malformed_length_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self._store(tmp)
            with pytest.raises(ValueError):
                s.create("P2", expected_length_range=[100])

    def test_rejects_inverted_length_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self._store(tmp)
            with pytest.raises(ValueError):
                s.create("P3", expected_length_range=[600, 10])

    def test_rejects_incomplete_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self._store(tmp)
            with pytest.raises(ValueError):
                s.create("P4", custom_weights={"accuracy": 0.5})

    def test_rejects_zero_sum_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self._store(tmp)
            with pytest.raises(ValueError):
                s.create("P5", custom_weights={
                    "accuracy": 0, "code": 0, "structure": 0, "speed": 0,
                })

    def test_valid_profile_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self._store(tmp)
            p = s.create("P6", timeout=30, max_response_tokens=500,
                         expected_length_range=[10, 600])
            assert p.profile_id.startswith("custom_")

    def test_update_rejects_bad_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self._store(tmp)
            p = s.create("P7")
            with pytest.raises(ValueError):
                s.update(p.profile_id, {"timeout": -5})


class TestBMK14LoadGuard:
    def test_builtin_id_entry_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.yaml"
            path.write_text(
                "profiles:\n"
                "  all_round:\n"
                "    name: Shadow\n"
                "    categories: [math]\n"
                "  custom_abc123:\n"
                "    name: Real\n"
                "    categories: [math]\n",
                encoding="utf-8",
            )
            store = _cp_mod.CustomProfileStore(path=path)
            ids = {p.profile_id for p in store.list_profiles()}
            assert ids == {"custom_abc123"}


# ---------------------------------------------------------------------------
# BMK-16 — skipped trigger keeps the change for re-detection
# ---------------------------------------------------------------------------

class _FakeRunner:
    def __init__(self):
        self.is_busy = False
        self.started = []

    def start_run(self, profile, models, use_judge=False, judge_model=""):
        self.started.append(list(models))
        return f"run-{len(self.started)}"


class TestBMK16SkippedTriggerRetries:
    def _trigger(self, tmp, runner, models_seq):
        path = Path(tmp) / "at.yaml"
        # Build a list_fn that returns the next snapshot on each call.
        seq = iter(models_seq)
        current = {"v": next(seq)}

        def list_fn():
            return [{"model": n, "digest": d} for n, d in current["v"].items()]

        at = _at_mod.AutoTrigger(
            config_path=path, benchmark_runner=runner, ollama_list_fn=list_fn,
        )
        # Disable cooldown / resource guard for the test.
        at._cooldown = 0.0
        at._last_trigger_time = 0.0
        return at, current, seq

    def test_busy_skip_then_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = _FakeRunner()
            # Baseline {m1}; then {m1, m2} appears.
            at, current, _ = self._trigger(
                tmp, runner, [{"m1": "d1"}],
            )
            at._last_snapshot = _at_mod.ModelSnapshot(models={"m1": "d1"})

            # m2 appears while the runner is busy -> trigger skipped.
            current["v"] = {"m1": "d1", "m2": "d2"}
            runner.is_busy = True
            at._poll_once()
            assert runner.started == []
            # Snapshot NOT advanced: m2 still unknown.
            assert "m2" not in at._last_snapshot.models

            # Runner free now -> next poll re-detects and triggers.
            runner.is_busy = False
            at._poll_once()
            assert runner.started == [["m2"]]
            assert "m2" in at._last_snapshot.models

            # No further changes -> no extra trigger.
            at._poll_once()
            assert runner.started == [["m2"]]
