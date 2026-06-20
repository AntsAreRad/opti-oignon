#!/usr/bin/env python3
"""
Tests for S91 — Auto-Trigger Polish + Custom Profile Weights in Runner.

Covers:
  - Auto-trigger: test_poll, busy-runner handling, cooldown_remaining,
    resource_guard_active in status, event log
  - Custom profiles: duplicate name prevention, name length validation
  - Benchmark runner: custom_weights threading, is_busy property,
    effective weights in RunResult metadata
  - API schemas: new/updated schemas
"""

import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Direct module loading (avoids __init__.py dependency chain)
# ---------------------------------------------------------------------------

_PROJECT = Path(__file__).resolve().parent.parent / "opti_oignon"


def _load_module(name: str, filename: str):
    path = _PROJECT / filename
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_auto_trigger_mod = _load_module(
    "_benchmark_auto_trigger_s91", "benchmark_auto_trigger.py"
)
AutoTrigger = _auto_trigger_mod.AutoTrigger
ModelSnapshot = _auto_trigger_mod.ModelSnapshot
TriggerEvent = _auto_trigger_mod.TriggerEvent

_custom_profiles_mod = _load_module(
    "_benchmark_custom_profiles_s91", "benchmark_custom_profiles.py"
)
CustomProfileStore = _custom_profiles_mod.CustomProfileStore
CustomProfile = _custom_profiles_mod.CustomProfile

_runner_mod = _load_module("_benchmark_runner_s91", "benchmark_runner.py")
BenchmarkRunner = _runner_mod.BenchmarkRunner
RunResult = _runner_mod.RunResult
RunStatus = _runner_mod.RunStatus
RunProgress = _runner_mod.RunProgress
ResultsStore = _runner_mod.ResultsStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_yaml(tmp_path):
    """Temporary YAML config for auto-trigger."""
    path = tmp_path / "auto_trigger.yaml"
    import yaml
    yaml.safe_dump(
        {"enabled": False, "cooldown_seconds": 60, "poll_interval_seconds": 30},
        open(path, "w"),
    )
    return path


@pytest.fixture
def mock_ollama_models():
    models = []
    for name, digest in [("qwen3:32b", "abc123"), ("llama3:8b", "def456")]:
        m = MagicMock()
        m.model = name
        m.digest = digest
        models.append(m)
    return models


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.start_run.return_value = f"run-{uuid.uuid4().hex[:12]}"
    runner.is_busy = False
    return runner


@pytest.fixture
def trigger(tmp_yaml, mock_runner, mock_ollama_models):
    def list_fn():
        return mock_ollama_models
    t = AutoTrigger(
        config_path=tmp_yaml,
        benchmark_runner=mock_runner,
        ollama_list_fn=list_fn,
    )
    yield t
    t.stop()


@pytest.fixture
def profiles_path(tmp_path):
    return tmp_path / "custom_profiles.yaml"


@pytest.fixture
def store(profiles_path):
    return CustomProfileStore(path=profiles_path)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_results.db"


@pytest.fixture
def results_store(db_path):
    return ResultsStore(db_path=db_path)


# ===========================================================================
# Part 1 — Auto-Trigger UX Polish
# ===========================================================================

class TestAutoTriggerTestPoll:
    """Tests for the test_poll method (S91)."""

    def test_test_poll_success(self, trigger):
        """test_poll returns ok=True and model count."""
        result = trigger.test_poll()
        assert result["ok"] is True
        assert result["snapshot_models"] == 2
        assert "qwen3:32b" in result["model_names"]
        assert "llama3:8b" in result["model_names"]

    def test_test_poll_no_diff_on_first(self, trigger):
        """First poll has no baseline, so diff should be None."""
        result = trigger.test_poll()
        assert result["diff"] is None

    def test_test_poll_detects_diff(self, trigger):
        """test_poll detects diff after baseline is established."""
        # Establish baseline via a poll_once
        trigger._poll_once()
        # Now test_poll should compare against baseline
        result = trigger.test_poll()
        assert result["ok"] is True
        assert result["diff"] is not None
        assert result["diff"]["has_changes"] is False

    def test_test_poll_does_not_modify_snapshot(self, trigger):
        """test_poll must not change the stored snapshot."""
        trigger._poll_once()
        old_snapshot = trigger._last_snapshot
        trigger.test_poll()
        assert trigger._last_snapshot is old_snapshot

    def test_test_poll_connection_failure(self, tmp_yaml):
        """test_poll returns ok=False when Ollama is unreachable."""
        def failing_list():
            return []
        t = AutoTrigger(
            config_path=tmp_yaml,
            ollama_list_fn=failing_list,
        )
        result = t.test_poll()
        assert result["ok"] is False
        assert result["snapshot_models"] == 0


class TestAutoTriggerCooldownStatus:
    """Tests for cooldown_remaining in status (S91)."""

    def test_status_cooldown_remaining_zero_initially(self, trigger):
        """Cooldown remaining is 0 when no trigger has fired."""
        status = trigger.status
        assert status["cooldown_remaining"] == 0.0

    def test_status_cooldown_remaining_after_trigger(self, trigger, mock_runner):
        """Cooldown remaining > 0 after a trigger fires."""
        trigger._poll_once()
        trigger._do_trigger(["test:model"], "new_model")
        status = trigger.status
        assert status["cooldown_remaining"] > 0

    def test_status_resource_guard_fields(self, trigger):
        """Status includes resource_guard_active and load_max."""
        status = trigger.status
        assert "resource_guard_active" in status
        assert "resource_guard_load_max" in status
        # Default config has no resource guard
        assert status["resource_guard_active"] is False


class TestAutoTriggerBusyRunner:
    """Tests for runner-busy handling in auto-trigger (S91)."""

    def test_trigger_skips_when_runner_busy(self, trigger, mock_runner):
        """Auto-trigger skips when runner.is_busy is True."""
        mock_runner.is_busy = True
        trigger._poll_once()
        event = trigger._do_trigger(["qwen3:32b"], "new_model")
        assert event.skipped is True
        assert "busy" in event.skip_reason.lower()
        mock_runner.start_run.assert_not_called()

    def test_trigger_proceeds_when_runner_free(self, trigger, mock_runner):
        """Auto-trigger proceeds when runner.is_busy is False."""
        mock_runner.is_busy = False
        trigger._poll_once()
        event = trigger._do_trigger(["qwen3:32b"], "new_model")
        assert not event.skipped
        mock_runner.start_run.assert_called_once()


class TestAutoTriggerEventLog:
    """Tests for event recording and retrieval (S91)."""

    def test_events_empty_initially(self, trigger):
        assert trigger.events == []

    def test_events_after_trigger(self, trigger, mock_runner):
        """Events list grows after a trigger."""
        trigger._poll_once()
        trigger._do_trigger(["test:7b"], "new_model")
        events = trigger.events
        assert len(events) == 1
        assert events[0]["trigger_type"] == "new_model"
        assert "test:7b" in events[0]["models"]

    def test_events_record_skipped(self, trigger, mock_runner):
        """Skipped events (busy runner) are recorded."""
        mock_runner.is_busy = True
        trigger._poll_once()
        trigger._do_trigger(["test:7b"], "new_model")
        events = trigger.events
        assert len(events) == 1
        assert events[0]["skipped"] is True


# ===========================================================================
# Part 2 — Custom Profile Validation
# ===========================================================================

class TestCustomProfileNameValidation:
    """Tests for profile name constraints (S91)."""

    def test_create_duplicate_name_raises(self, store):
        """Creating a profile with an existing name raises ValueError."""
        store.create(name="My Profile")
        with pytest.raises(ValueError, match="already exists"):
            store.create(name="My Profile")

    def test_create_duplicate_case_insensitive(self, store):
        """Duplicate detection is case-insensitive."""
        store.create(name="Test Profile")
        with pytest.raises(ValueError, match="already exists"):
            store.create(name="test profile")

    def test_create_empty_name_raises(self, store):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            store.create(name="")

    def test_create_whitespace_name_raises(self, store):
        """Whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            store.create(name="   ")

    def test_create_name_too_long_raises(self, store):
        """Name exceeding 64 chars raises ValueError."""
        long_name = "A" * 65
        with pytest.raises(ValueError, match="too long"):
            store.create(name=long_name)

    def test_create_name_at_limit(self, store):
        """Name exactly 64 chars is accepted."""
        name_64 = "B" * 64
        profile = store.create(name=name_64)
        assert profile.name == name_64

    def test_update_duplicate_name_raises(self, store):
        """Updating a profile to an existing name raises ValueError."""
        p1 = store.create(name="Alpha")
        p2 = store.create(name="Beta")
        with pytest.raises(ValueError, match="already exists"):
            store.update(p2.profile_id, {"name": "Alpha"})

    def test_update_same_name_ok(self, store):
        """Updating a profile keeping same name works (no self-conflict)."""
        p = store.create(name="Gamma")
        result = store.update(p.profile_id, {"name": "Gamma"})
        assert result is not None
        assert result.name == "Gamma"

    def test_update_name_too_long_raises(self, store):
        """Updating name to > 64 chars raises ValueError."""
        p = store.create(name="Short")
        with pytest.raises(ValueError, match="too long"):
            store.update(p.profile_id, {"name": "X" * 65})


# ===========================================================================
# Part 3 — Custom Weights in Runner Pipeline
# ===========================================================================

class TestBenchmarkRunnerIsBusy:
    """Tests for BenchmarkRunner.is_busy property (S91)."""

    def test_is_busy_false_when_idle(self, results_store):
        """is_busy is False when no runs are active."""
        runner = BenchmarkRunner(store=results_store)
        assert runner.is_busy is False

    def test_is_busy_true_when_running(self, results_store):
        """is_busy is True when a run is in RUNNING state."""
        runner = BenchmarkRunner(store=results_store)
        with runner._lock:
            runner._active_runs["test-run"] = RunProgress(
                run_id="test-run",
                status=RunStatus.RUNNING,
            )
        assert runner.is_busy is True

    def test_is_busy_false_when_completed(self, results_store):
        """is_busy is False when all runs are completed."""
        runner = BenchmarkRunner(store=results_store)
        with runner._lock:
            runner._active_runs["done-run"] = RunProgress(
                run_id="done-run",
                status=RunStatus.COMPLETED,
            )
        assert runner.is_busy is False


class TestRunResultCustomWeights:
    """Tests for custom_weights in RunResult (S91)."""

    def test_run_result_default_no_custom_weights(self):
        """RunResult.custom_weights is None by default."""
        r = RunResult(run_id="r1", profile="test")
        assert r.custom_weights is None

    def test_run_result_with_custom_weights(self):
        """RunResult stores custom_weights dict."""
        cw = {"accuracy": 0.5, "code": 0.2, "structure": 0.2, "speed": 0.1}
        r = RunResult(run_id="r2", profile="test", custom_weights=cw)
        assert r.custom_weights == cw

    def test_results_store_persists_custom_weights(self, results_store):
        """Custom weights survive save/load round-trip in SQLite."""
        cw = {"accuracy": 0.4, "code": 0.3, "structure": 0.2, "speed": 0.1}
        result = RunResult(
            run_id="run-cw-test",
            profile="test_profile",
            models=["model-a"],
            status=RunStatus.COMPLETED,
            started_at=time.time() - 10,
            finished_at=time.time(),
            duration_ms=10000.0,
            weight_preset="custom",
            custom_weights=cw,
        )
        results_store.save_run(result)
        loaded = results_store.get_run("run-cw-test")
        assert loaded is not None
        assert loaded["custom_weights"] == cw

    def test_results_store_no_custom_weights(self, results_store):
        """Run without custom_weights stores None."""
        result = RunResult(
            run_id="run-no-cw",
            profile="test_profile",
            models=["model-a"],
            status=RunStatus.COMPLETED,
            started_at=time.time() - 10,
            finished_at=time.time(),
            duration_ms=10000.0,
        )
        results_store.save_run(result)
        loaded = results_store.get_run("run-no-cw")
        assert loaded is not None
        assert loaded["custom_weights"] is None

    def test_history_includes_custom_weights(self, results_store):
        """get_history returns custom_weights for each run."""
        cw = {"accuracy": 0.6, "code": 0.1, "structure": 0.2, "speed": 0.1}
        result = RunResult(
            run_id="run-hist-cw",
            profile="all_round",
            models=["model-a"],
            status=RunStatus.COMPLETED,
            started_at=time.time(),
            finished_at=time.time(),
            duration_ms=5000.0,
            custom_weights=cw,
        )
        results_store.save_run(result)
        history = results_store.get_history(limit=10)
        assert len(history) >= 1
        found = [r for r in history if r["run_id"] == "run-hist-cw"]
        assert len(found) == 1
        assert found[0]["custom_weights"] == cw


# ===========================================================================
# Part 4 — Schema validation
# ===========================================================================

class TestSchemas:
    """Tests for new/updated Pydantic schemas (S91)."""

    @pytest.fixture(autouse=True)
    def _load_schemas(self):
        schemas_path = _PROJECT / "api" / "schemas.py"
        spec = importlib.util.spec_from_file_location("_schemas_s91", str(schemas_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._schemas = mod

    def test_auto_trigger_status_new_fields(self):
        S = self._schemas.BenchmarkV2AutoTriggerStatusResponse
        s = S(
            cooldown_remaining=42.5,
            resource_guard_active=True,
            resource_guard_load_max=4.0,
        )
        assert s.cooldown_remaining == 42.5
        assert s.resource_guard_active is True
        assert s.resource_guard_load_max == 4.0

    def test_test_poll_response_schema(self):
        R = self._schemas.BenchmarkV2AutoTriggerTestPollResponse
        r = R(
            ok=True,
            snapshot_models=5,
            model_names=["a", "b"],
        )
        assert r.ok is True
        assert r.snapshot_models == 5

    def test_results_response_custom_weights(self):
        R = self._schemas.BenchmarkV2ResultsResponse
        r = R(
            custom_weights={"accuracy": 0.5, "code": 0.2, "structure": 0.2, "speed": 0.1},
        )
        assert r.custom_weights["accuracy"] == 0.5

    def test_history_entry_custom_weights(self):
        E = self._schemas.BenchmarkV2HistoryEntry
        e = E(
            custom_weights={"accuracy": 0.3, "code": 0.3, "structure": 0.2, "speed": 0.2},
        )
        assert e.custom_weights is not None

    def test_run_request_custom_weights(self):
        R = self._schemas.BenchmarkV2RunRequest
        r = R(
            profile="test",
            models=["m1"],
            custom_weights={"accuracy": 0.4, "code": 0.3, "structure": 0.2, "speed": 0.1},
        )
        assert r.custom_weights is not None


class TestVersionBump:
    """Verify version was bumped to 1.9.3."""

    def test_app_version(self):
        app_path = _PROJECT / "api" / "app.py"
        content = app_path.read_text()
        assert '"1.10.0"' in content

    def test_pyproject_version(self):
        pyproject_path = _PROJECT.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        assert '"1.10.0"' in content

    def test_setup_version(self):
        setup_path = _PROJECT.parent / "setup.py"
        content = setup_path.read_text()
        assert '"1.10.0"' in content
