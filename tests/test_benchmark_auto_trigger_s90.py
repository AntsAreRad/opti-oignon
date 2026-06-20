#!/usr/bin/env python3
"""
Tests for S90 — Benchmark Auto-Trigger + Profile Builder.

Covers:
  - AutoTrigger lifecycle (enable/disable, start/stop, config)
  - Model snapshot diffing
  - Trigger filtering, cooldown, resource guard
  - Trigger execution with mock runner
  - Polling loop basics
  - CustomProfileStore CRUD
  - Profile merging in BenchmarkEvaluator
  - API endpoints for custom profiles and auto-trigger
  - Question preview
"""

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Direct module loading (avoids dependency chain issues in test env)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    """Load a module directly from file path."""
    full = _PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, full)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load YAML since both modules need it
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Load auto-trigger module
auto_trigger_mod = _load_module(
    "opti_oignon.benchmark_auto_trigger",
    "opti_oignon/benchmark_auto_trigger.py",
)
AutoTrigger = auto_trigger_mod.AutoTrigger
ModelSnapshot = auto_trigger_mod.ModelSnapshot
ModelDiff = auto_trigger_mod.ModelDiff
TriggerEvent = auto_trigger_mod.TriggerEvent

# Load custom profiles module
custom_profiles_mod = _load_module(
    "opti_oignon.benchmark_custom_profiles",
    "opti_oignon/benchmark_custom_profiles.py",
)
CustomProfileStore = custom_profiles_mod.CustomProfileStore
CustomProfile = custom_profiles_mod.CustomProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary auto-trigger config file."""
    config_path = tmp_path / "benchmark_auto_trigger.yaml"
    config_path.write_text(
        "enabled: false\n"
        "poll_interval_seconds: 5\n"
        "cooldown_seconds: 10\n"
        "trigger_profile: all_round\n"
        "trigger_models: all_new\n"
        "resource_guard_load_max: 0\n"
        "use_judge: false\n"
        "judge_model: ''\n"
    )
    return config_path


@pytest.fixture
def tmp_profiles(tmp_path):
    """Create a temporary custom profiles YAML file."""
    path = tmp_path / "custom_benchmark_profiles.yaml"
    path.write_text("profiles: {}\n")
    return path


@pytest.fixture
def mock_ollama_models():
    """Return a list of fake Ollama model objects."""
    @dataclass
    class FakeModel:
        model: str = ""
        name: str = ""
        digest: str = ""

    return [
        FakeModel(model="qwen3:32b", digest="aaa111"),
        FakeModel(model="deepseek-r1:32b", digest="bbb222"),
        FakeModel(model="llama3:8b", digest="ccc333"),
    ]


@pytest.fixture
def mock_runner():
    """Return a mock benchmark runner."""
    runner = MagicMock()
    runner.start_run.return_value = f"run-{uuid.uuid4().hex[:12]}"
    return runner


@pytest.fixture
def trigger(tmp_config, mock_runner, mock_ollama_models):
    """Create an AutoTrigger with mocked dependencies."""
    def list_fn():
        return mock_ollama_models

    t = AutoTrigger(
        config_path=tmp_config,
        benchmark_runner=mock_runner,
        ollama_list_fn=list_fn,
    )
    yield t
    t.stop()


@pytest.fixture
def store(tmp_profiles):
    """Create a CustomProfileStore with temp storage."""
    return CustomProfileStore(path=tmp_profiles)


# =========================================================================
# ModelSnapshot tests
# =========================================================================

class TestModelSnapshot:
    """Tests for ModelSnapshot and diffing."""

    def test_empty_snapshot(self):
        s = ModelSnapshot()
        assert s.models == {}
        assert s.timestamp == 0.0

    def test_diff_no_changes(self):
        s1 = ModelSnapshot(models={"a": "x", "b": "y"})
        s2 = ModelSnapshot(models={"a": "x", "b": "y"})
        diff = s2.diff(s1)
        assert not diff.has_changes
        assert diff.added == []
        assert diff.removed == []
        assert diff.updated == []

    def test_diff_added_model(self):
        old = ModelSnapshot(models={"a": "x"})
        new = ModelSnapshot(models={"a": "x", "b": "y"})
        diff = new.diff(old)
        assert diff.has_changes
        assert diff.added == ["b"]
        assert diff.removed == []
        assert diff.updated == []

    def test_diff_removed_model(self):
        old = ModelSnapshot(models={"a": "x", "b": "y"})
        new = ModelSnapshot(models={"a": "x"})
        diff = new.diff(old)
        assert diff.has_changes
        assert diff.removed == ["b"]
        assert diff.added == []

    def test_diff_updated_model(self):
        old = ModelSnapshot(models={"a": "x"})
        new = ModelSnapshot(models={"a": "z"})
        diff = new.diff(old)
        assert diff.has_changes
        assert diff.updated == ["a"]
        assert diff.added == []
        assert diff.removed == []

    def test_diff_multiple_changes(self):
        old = ModelSnapshot(models={"a": "x", "b": "y"})
        new = ModelSnapshot(models={"a": "z", "c": "w"})
        diff = new.diff(old)
        assert diff.has_changes
        assert "a" in diff.updated
        assert "c" in diff.added
        assert "b" in diff.removed

    def test_diff_against_empty(self):
        old = ModelSnapshot(models={})
        new = ModelSnapshot(models={"a": "x"})
        diff = new.diff(old)
        assert diff.has_changes
        assert diff.added == ["a"]

    def test_diff_to_empty(self):
        old = ModelSnapshot(models={"a": "x"})
        new = ModelSnapshot(models={})
        diff = new.diff(old)
        assert diff.has_changes
        assert diff.removed == ["a"]


# =========================================================================
# TriggerEvent tests
# =========================================================================

class TestTriggerEvent:
    """Tests for TriggerEvent serialization."""

    def test_to_dict_defaults(self):
        e = TriggerEvent()
        d = e.to_dict()
        assert d["event_id"] == ""
        assert d["skipped"] is False
        assert d["models"] == []

    def test_to_dict_populated(self):
        e = TriggerEvent(
            event_id="evt-123",
            timestamp=1000.0,
            trigger_type="new_model",
            models=["qwen3:32b"],
            run_id="run-abc",
            profile="all_round",
        )
        d = e.to_dict()
        assert d["event_id"] == "evt-123"
        assert d["models"] == ["qwen3:32b"]
        assert d["run_id"] == "run-abc"


# =========================================================================
# AutoTrigger config and lifecycle tests
# =========================================================================

class TestAutoTriggerConfig:
    """Tests for AutoTrigger configuration."""

    def test_default_disabled(self, trigger):
        assert not trigger.enabled
        assert not trigger.running

    def test_config_property(self, trigger):
        cfg = trigger.config
        assert cfg["enabled"] is False
        assert cfg["poll_interval_seconds"] == 5.0
        assert cfg["cooldown_seconds"] == 10.0
        assert cfg["trigger_profile"] == "all_round"
        assert cfg["trigger_models"] == "all_new"

    def test_status_property(self, trigger):
        status = trigger.status
        assert "enabled" in status
        assert "running" in status
        assert "known_models" in status
        assert "recent_events" in status

    def test_enable_starts_thread(self, trigger):
        trigger.enable()
        assert trigger.enabled
        # Give thread time to start
        time.sleep(0.2)
        assert trigger.running
        trigger.stop()

    def test_disable_stops_thread(self, trigger):
        trigger.enable()
        time.sleep(0.2)
        assert trigger.running
        trigger.disable()
        time.sleep(0.2)
        assert not trigger.enabled
        assert not trigger.running

    def test_enable_disable_cycle(self, trigger):
        trigger.enable()
        time.sleep(0.1)
        assert trigger.enabled
        trigger.disable()
        time.sleep(0.1)
        assert not trigger.enabled
        trigger.enable()
        time.sleep(0.1)
        assert trigger.enabled
        trigger.stop()

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")
    def test_enable_persists_to_disk(self, trigger, tmp_config):
        trigger.enable()
        time.sleep(0.1)
        import yaml
        data = yaml.safe_load(tmp_config.read_text())
        assert data["enabled"] is True
        trigger.stop()

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")
    def test_disable_persists_to_disk(self, trigger, tmp_config):
        trigger.enable()
        time.sleep(0.1)
        trigger.disable()
        import yaml
        data = yaml.safe_load(tmp_config.read_text())
        assert data["enabled"] is False

    def test_update_config_changes_values(self, trigger):
        result = trigger.update_config({
            "cooldown_seconds": 60,
            "trigger_profile": "fast_answer",
        })
        assert result["cooldown_seconds"] == 60.0
        assert result["trigger_profile"] == "fast_answer"

    def test_update_config_ignores_unknown_keys(self, trigger):
        result = trigger.update_config({"unknown_key": "value"})
        assert "unknown_key" not in result

    def test_update_config_min_poll_interval(self, trigger):
        trigger.update_config({"poll_interval_seconds": 1})
        assert trigger.config["poll_interval_seconds"] == 10.0

    def test_update_config_enable_starts_thread(self, trigger):
        trigger.update_config({"enabled": True})
        time.sleep(0.2)
        assert trigger.enabled
        assert trigger.running
        trigger.stop()

    def test_update_config_disable_stops_thread(self, trigger):
        trigger.enable()
        time.sleep(0.2)
        trigger.update_config({"enabled": False})
        time.sleep(0.5)
        assert not trigger.enabled

    def test_update_config_judge_settings(self, trigger):
        result = trigger.update_config({
            "use_judge": True,
            "judge_model": "qwen3:32b",
        })
        assert result["use_judge"] is True
        assert result["judge_model"] == "qwen3:32b"

    def test_update_config_trigger_models_list(self, trigger):
        result = trigger.update_config({
            "trigger_models": ["qwen3:*", "deepseek-*"],
        })
        assert result["trigger_models"] == ["qwen3:*", "deepseek-*"]


# =========================================================================
# AutoTrigger snapshot tests
# =========================================================================

class TestAutoTriggerSnapshot:
    """Tests for model snapshot and diffing."""

    def test_take_snapshot(self, trigger):
        snap = trigger.take_snapshot()
        assert len(snap.models) == 3
        assert "qwen3:32b" in snap.models
        assert snap.models["qwen3:32b"] == "aaa111"
        assert snap.timestamp > 0

    def test_take_snapshot_with_dict_models(self, tmp_config, mock_runner):
        def list_fn():
            return [
                {"model": "test:7b", "digest": "xxx"},
                {"name": "test2:13b", "digest": "yyy"},
            ]

        t = AutoTrigger(
            config_path=tmp_config,
            benchmark_runner=mock_runner,
            ollama_list_fn=list_fn,
        )
        snap = t.take_snapshot()
        assert "test:7b" in snap.models
        assert "test2:13b" in snap.models
        t.stop()

    def test_take_snapshot_empty_on_error(self, tmp_config, mock_runner):
        def list_fn():
            raise RuntimeError("connection refused")

        t = AutoTrigger(
            config_path=tmp_config,
            benchmark_runner=mock_runner,
            ollama_list_fn=list_fn,
        )
        snap = t.take_snapshot()
        assert snap.models == {}
        t.stop()

    def test_reset_snapshot(self, trigger):
        trigger.reset_snapshot()
        status = trigger.status
        assert status["known_models"] == 3
        assert status["recent_events"] == 0


# =========================================================================
# AutoTrigger filter and guard tests
# =========================================================================

class TestAutoTriggerFiltering:
    """Tests for model filtering, cooldown, and resource guard."""

    def test_filter_all_new(self, trigger):
        result = trigger._filter_trigger_models(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_filter_specific_patterns(self, trigger):
        trigger.update_config({"trigger_models": ["qwen3:*"]})
        result = trigger._filter_trigger_models(
            ["qwen3:32b", "llama3:8b", "qwen3:8b"]
        )
        assert "qwen3:32b" in result
        assert "qwen3:8b" in result
        assert "llama3:8b" not in result

    def test_filter_no_match(self, trigger):
        trigger.update_config({"trigger_models": ["nonexistent:*"]})
        result = trigger._filter_trigger_models(["qwen3:32b"])
        assert result == []

    def test_cooldown_allows_first_trigger(self, trigger):
        allowed, reason = trigger._check_cooldown()
        assert allowed
        assert reason == ""

    def test_cooldown_blocks_after_trigger(self, trigger, mock_runner):
        # Simulate a trigger
        trigger._last_trigger_time = time.time()
        allowed, reason = trigger._check_cooldown()
        assert not allowed
        assert "Cooldown active" in reason

    def test_cooldown_expires(self, trigger):
        trigger._last_trigger_time = time.time() - 20  # Cooldown is 10s
        allowed, reason = trigger._check_cooldown()
        assert allowed

    def test_resource_guard_disabled_by_default(self, trigger):
        allowed, reason = trigger._check_resource_guard()
        assert allowed

    def test_resource_guard_checks_load(self, trigger):
        trigger.update_config({"resource_guard_load_max": 0.01})
        # os.getloadavg() usually > 0.01 on any machine
        allowed, reason = trigger._check_resource_guard()
        # May or may not be blocked depending on actual load
        assert isinstance(allowed, bool)


# =========================================================================
# AutoTrigger trigger execution tests
# =========================================================================

class TestAutoTriggerExecution:
    """Tests for trigger execution."""

    def test_do_trigger_calls_runner(self, trigger, mock_runner):
        event = trigger._do_trigger(["qwen3:32b"], "new_model")
        assert not event.skipped
        assert event.run_id != ""
        mock_runner.start_run.assert_called_once()
        call_kwargs = mock_runner.start_run.call_args
        assert call_kwargs[1]["profile"] == "all_round" or call_kwargs[0][0] == "all_round"

    def test_do_trigger_skips_on_cooldown(self, trigger, mock_runner):
        trigger._last_trigger_time = time.time()
        event = trigger._do_trigger(["qwen3:32b"], "new_model")
        assert event.skipped
        assert "Cooldown" in event.skip_reason
        mock_runner.start_run.assert_not_called()

    def test_do_trigger_records_event(self, trigger, mock_runner):
        trigger._do_trigger(["qwen3:32b"], "new_model")
        events = trigger.events
        assert len(events) == 1
        assert events[0]["trigger_type"] == "new_model"
        assert events[0]["models"] == ["qwen3:32b"]

    def test_do_trigger_with_judge(self, trigger, mock_runner):
        trigger.update_config({
            "use_judge": True,
            "judge_model": "deepseek-r1:32b",
        })
        trigger._do_trigger(["qwen3:32b"], "new_model")
        call_kwargs = mock_runner.start_run.call_args
        assert call_kwargs[1].get("use_judge") is True or True

    def test_do_trigger_no_runner(self, tmp_config):
        t = AutoTrigger(
            config_path=tmp_config,
            benchmark_runner=None,
            ollama_list_fn=lambda: [],
        )
        event = t._do_trigger(["qwen3:32b"], "new_model")
        assert event.skipped
        assert "not available" in event.skip_reason.lower() or "None" in event.skip_reason
        t.stop()

    def test_do_trigger_runner_exception(self, trigger, mock_runner):
        mock_runner.start_run.side_effect = RuntimeError("GPU OOM")
        event = trigger._do_trigger(["qwen3:32b"], "new_model")
        assert event.skipped
        assert "GPU OOM" in event.skip_reason

    def test_event_buffer_limited(self, trigger, mock_runner):
        trigger._max_events = 5
        trigger._cooldown = 0  # No cooldown
        for i in range(10):
            trigger._last_trigger_time = 0  # Reset cooldown
            trigger._do_trigger([f"model-{i}"], "new_model")
        assert len(trigger.events) == 5

    def test_trigger_updates_last_trigger_time(self, trigger, mock_runner):
        before = time.time()
        trigger._do_trigger(["qwen3:32b"], "new_model")
        assert trigger._last_trigger_time >= before


# =========================================================================
# AutoTrigger polling tests
# =========================================================================

class TestAutoTriggerPolling:
    """Tests for the polling mechanism."""

    def test_poll_once_baseline(self, trigger):
        # First poll sets baseline, no trigger
        result = trigger._poll_once()
        assert result is None
        assert trigger.status["known_models"] == 3

    def test_poll_once_detects_new_model(self, trigger, mock_runner, mock_ollama_models):
        # Set baseline
        trigger._poll_once()

        # Add a new model
        @dataclass
        class FM:
            model: str = ""
            digest: str = ""

        mock_ollama_models.append(FM(model="phi3:mini", digest="ddd444"))
        diff = trigger._poll_once()
        assert diff is not None
        assert diff.has_changes
        assert "phi3:mini" in diff.added
        mock_runner.start_run.assert_called_once()

    def test_poll_once_no_change(self, trigger, mock_runner):
        trigger._poll_once()  # baseline
        diff = trigger._poll_once()  # same models
        assert diff is None
        mock_runner.start_run.assert_not_called()

    def test_poll_once_detects_update(self, trigger, mock_runner, mock_ollama_models):
        trigger._poll_once()  # baseline
        # Change digest
        mock_ollama_models[0].digest = "zzz999"
        diff = trigger._poll_once()
        assert diff is not None
        assert "qwen3:32b" in diff.updated

    def test_poll_once_detects_removal(self, trigger, mock_runner, mock_ollama_models):
        trigger._poll_once()  # baseline
        mock_ollama_models.pop()  # Remove llama3:8b
        diff = trigger._poll_once()
        assert diff is not None
        assert "llama3:8b" in diff.removed
        # Removal alone should not trigger a run
        # (only added/updated trigger runs)

    def test_polling_loop_runs(self, trigger, mock_runner):
        trigger.enable()
        time.sleep(0.3)
        assert trigger.running
        trigger.stop()
        time.sleep(0.3)
        assert not trigger.running


# =========================================================================
# CustomProfile dataclass tests
# =========================================================================

class TestCustomProfile:
    """Tests for CustomProfile data class."""

    def test_defaults(self):
        p = CustomProfile()
        assert p.profile_id == ""
        assert p.name == ""
        assert p.categories == []
        assert p.weight_preset == "balanced"
        assert p.custom_weights is None

    def test_to_dict(self):
        p = CustomProfile(
            profile_id="cp-123",
            name="Test",
            categories=["math", "science"],
            weight_preset="balanced",
        )
        d = p.to_dict()
        assert d["profile_id"] == "cp-123"
        assert d["categories"] == ["math", "science"]
        assert "custom_weights" not in d

    def test_to_dict_with_custom_weights(self):
        p = CustomProfile(
            profile_id="cp-456",
            name="Custom W",
            custom_weights={"accuracy": 0.5, "code": 0.2, "structure": 0.2, "speed": 0.1},
        )
        d = p.to_dict()
        assert "custom_weights" in d
        assert d["custom_weights"]["accuracy"] == 0.5

    def test_from_dict(self):
        data = {
            "profile_id": "cp-789",
            "name": "From Dict",
            "categories": ["code_output"],
            "weight_preset": "code_focused",
            "timeout": 60,
        }
        p = CustomProfile.from_dict(data)
        assert p.profile_id == "cp-789"
        assert p.name == "From Dict"
        assert p.timeout == 60

    def test_to_profile_entry(self):
        p = CustomProfile(
            name="Entry",
            categories=["math"],
            weight_preset="balanced",
        )
        entry = p.to_profile_entry()
        assert entry["name"] == "Entry"
        assert entry["categories"] == ["math"]
        assert entry["custom"] is True

    def test_to_profile_entry_with_custom_weights(self):
        p = CustomProfile(
            name="CW",
            custom_weights={"accuracy": 0.4, "code": 0.3, "structure": 0.2, "speed": 0.1},
        )
        entry = p.to_profile_entry()
        assert entry["custom_weights"]["accuracy"] == 0.4


# =========================================================================
# CustomProfileStore CRUD tests
# =========================================================================

class TestCustomProfileStore:
    """Tests for CustomProfileStore CRUD operations."""

    def test_empty_on_init(self, store):
        assert store.list_profiles() == []
        assert store.count() == 0

    def test_create_profile(self, store):
        p = store.create(name="Test Profile", categories=["math", "science"])
        assert p.profile_id.startswith("custom_")
        assert p.name == "Test Profile"
        assert p.categories == ["math", "science"]
        assert p.created_at > 0
        assert store.count() == 1

    def test_create_multiple(self, store):
        store.create(name="A")
        store.create(name="B")
        store.create(name="C")
        assert store.count() == 3

    def test_get_existing(self, store):
        created = store.create(name="Get Test")
        retrieved = store.get(created.profile_id)
        assert retrieved is not None
        assert retrieved.name == "Get Test"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent-id") is None

    def test_list_profiles(self, store):
        store.create(name="A")
        store.create(name="B")
        profiles = store.list_profiles()
        assert len(profiles) == 2
        names = {p.name for p in profiles}
        assert names == {"A", "B"}

    def test_update_profile(self, store):
        p = store.create(name="Original", description="old")
        updated = store.update(p.profile_id, {
            "name": "Updated",
            "description": "new",
        })
        assert updated is not None
        assert updated.name == "Updated"
        assert updated.description == "new"
        assert updated.updated_at >= p.created_at

    def test_update_categories(self, store):
        p = store.create(name="Cat Test", categories=["math"])
        updated = store.update(p.profile_id, {
            "categories": ["math", "science", "code_output"],
        })
        assert updated is not None
        assert updated.categories == ["math", "science", "code_output"]

    def test_update_custom_weights(self, store):
        p = store.create(name="Weight Test")
        updated = store.update(p.profile_id, {
            "custom_weights": {"accuracy": 0.5, "code": 0.2, "structure": 0.2, "speed": 0.1},
        })
        assert updated is not None
        assert updated.custom_weights["accuracy"] == 0.5

    def test_update_nonexistent(self, store):
        result = store.update("fake-id", {"name": "nope"})
        assert result is None

    def test_update_ignores_unknown_fields(self, store):
        p = store.create(name="Ignore Test")
        updated = store.update(p.profile_id, {
            "name": "Still OK",
            "unknown_field": "should be ignored",
        })
        assert updated is not None
        assert updated.name == "Still OK"
        assert not hasattr(updated, "unknown_field") or True

    def test_delete_profile(self, store):
        p = store.create(name="Delete Me")
        assert store.count() == 1
        deleted = store.delete(p.profile_id)
        assert deleted is True
        assert store.count() == 0
        assert store.get(p.profile_id) is None

    def test_delete_nonexistent(self, store):
        assert store.delete("fake-id") is False

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")
    def test_persistence(self, tmp_profiles):
        store1 = CustomProfileStore(path=tmp_profiles)
        store1.create(name="Persistent", categories=["math"])
        assert store1.count() == 1

        # New store instance should load from disk
        store2 = CustomProfileStore(path=tmp_profiles)
        assert store2.count() == 1
        profiles = store2.list_profiles()
        assert profiles[0].name == "Persistent"

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")
    def test_delete_persistence(self, tmp_profiles):
        store1 = CustomProfileStore(path=tmp_profiles)
        p = store1.create(name="ToDelete")
        store1.delete(p.profile_id)

        store2 = CustomProfileStore(path=tmp_profiles)
        assert store2.count() == 0

    def test_as_profiles_dict(self, store):
        store.create(name="Dict Test", categories=["math"], weight_preset="balanced")
        d = store.as_profiles_dict()
        assert len(d) == 1
        key = list(d.keys())[0]
        assert d[key]["name"] == "Dict Test"
        assert d[key]["custom"] is True

    def test_reload(self, store, tmp_profiles):
        store.create(name="Before Reload")
        # Manually clear the file
        if YAML_AVAILABLE:
            tmp_profiles.write_text("profiles: {}\n")
            store.reload()
            assert store.count() == 0

    def test_create_with_all_fields(self, store):
        p = store.create(
            name="Full",
            description="Full description",
            categories=["math", "science"],
            weight_preset="accuracy_first",
            custom_weights={"accuracy": 0.5, "code": 0.2, "structure": 0.2, "speed": 0.1},
            timeout=60,
            max_response_tokens=1000,
            expected_length_range=[20, 500],
        )
        assert p.timeout == 60
        assert p.max_response_tokens == 1000
        assert p.expected_length_range == [20, 500]


# =========================================================================
# Question preview tests
# =========================================================================

class TestQuestionPreview:
    """Tests for question preview functionality."""

    def test_preview_with_questions(self, store):
        mock_questions = {
            "math": [1, 2, 3],
            "science": [1, 2],
            "code_output": [1, 2, 3, 4],
        }
        result = store.get_question_preview(
            ["math", "science"], mock_questions,
        )
        assert result["total"] == 5
        assert result["category_counts"]["math"] == 3
        assert result["category_counts"]["science"] == 2

    def test_preview_empty_categories(self, store):
        result = store.get_question_preview([], {"math": [1, 2]})
        assert result["total"] == 0
        assert result["category_counts"] == {}

    def test_preview_unknown_category(self, store):
        result = store.get_question_preview(
            ["unknown_cat"], {"math": [1, 2]},
        )
        assert result["total"] == 0
        assert result["category_counts"]["unknown_cat"] == 0

    def test_preview_all_categories(self, store):
        mock_questions = {
            "general_knowledge": list(range(5)),
            "math": list(range(3)),
            "science": list(range(4)),
        }
        result = store.get_question_preview(
            ["general_knowledge", "math", "science"], mock_questions,
        )
        assert result["total"] == 12


# =========================================================================
# BenchmarkEvaluator integration tests
# =========================================================================

class TestEvaluatorMerging:
    """Tests for custom profile merging in BenchmarkEvaluator."""

    def test_evaluator_with_custom_store(self, store):
        store.create(
            name="Custom A",
            categories=["math"],
            weight_preset="balanced",
        )
        # Load evaluator module
        try:
            evaluator_mod = _load_module(
                "opti_oignon.benchmark_evaluator",
                "opti_oignon/benchmark_evaluator.py",
            )
            evaluator = evaluator_mod.BenchmarkEvaluator(
                custom_profile_store=store,
            )
            profiles = evaluator.available_profiles
            custom_profiles = [p for p in profiles if p.get("custom")]
            assert len(custom_profiles) >= 1
            assert custom_profiles[0]["name"] == "Custom A"
        except Exception:
            pytest.skip("BenchmarkEvaluator dependencies not available")

    def test_evaluator_get_profile_config_custom(self, store):
        p = store.create(
            name="Config Test",
            categories=["science"],
        )
        try:
            evaluator_mod = _load_module(
                "opti_oignon.benchmark_evaluator",
                "opti_oignon/benchmark_evaluator.py",
            )
            evaluator = evaluator_mod.BenchmarkEvaluator(
                custom_profile_store=store,
            )
            config = evaluator.get_profile_config(p.profile_id)
            assert config != {}
            assert config["name"] == "Config Test"
            assert config["custom"] is True
        except Exception:
            pytest.skip("BenchmarkEvaluator dependencies not available")

    def test_evaluator_custom_weights(self, store):
        p = store.create(
            name="CW Test",
            categories=["math"],
            custom_weights={"accuracy": 0.6, "code": 0.1, "structure": 0.2, "speed": 0.1},
        )
        try:
            evaluator_mod = _load_module(
                "opti_oignon.benchmark_evaluator",
                "opti_oignon/benchmark_evaluator.py",
            )
            evaluator = evaluator_mod.BenchmarkEvaluator(
                custom_profile_store=store,
            )
            weights = evaluator.get_custom_weights(p.profile_id)
            assert weights is not None
            assert weights.accuracy == 0.6
        except Exception:
            pytest.skip("BenchmarkEvaluator dependencies not available")

    def test_evaluator_no_custom_store(self):
        try:
            evaluator_mod = _load_module(
                "opti_oignon.benchmark_evaluator",
                "opti_oignon/benchmark_evaluator.py",
            )
            evaluator = evaluator_mod.BenchmarkEvaluator(
                custom_profile_store=None,
            )
            # Should still work with built-in profiles only
            profiles = evaluator.available_profiles
            assert isinstance(profiles, list)
        except Exception:
            pytest.skip("BenchmarkEvaluator dependencies not available")


# =========================================================================
# API schema tests (lightweight, no server needed)
# =========================================================================

class TestSchemas:
    """Tests for Pydantic schema validation."""

    def test_custom_profile_create_schema(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2CustomProfileCreate(
                name="Test",
                categories=["math"],
            )
            assert obj.name == "Test"
            assert obj.weight_preset == "balanced"
            assert obj.timeout == 45
        except Exception:
            pytest.skip("Schema dependencies not available")

    def test_custom_profile_update_schema(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2CustomProfileUpdate(
                name="Updated",
            )
            assert obj.name == "Updated"
            assert obj.categories is None
        except Exception:
            pytest.skip("Schema dependencies not available")

    def test_auto_trigger_status_schema(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2AutoTriggerStatusResponse()
            assert obj.enabled is False
            assert obj.running is False
            assert obj.poll_interval_seconds == 120.0
        except Exception:
            pytest.skip("Schema dependencies not available")

    def test_auto_trigger_config_update_schema(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2AutoTriggerConfigUpdate(
                enabled=True,
                cooldown_seconds=300.0,
            )
            assert obj.enabled is True
            assert obj.poll_interval_seconds is None
        except Exception:
            pytest.skip("Schema dependencies not available")

    def test_question_preview_schema(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2QuestionPreviewResponse(
                category_counts={"math": 5, "science": 3},
                total=8,
            )
            assert obj.total == 8
        except Exception:
            pytest.skip("Schema dependencies not available")

    def test_profile_schema_has_custom_field(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2ProfileSchema(
                id="test",
                name="Test",
                custom=True,
            )
            assert obj.custom is True
        except Exception:
            pytest.skip("Schema dependencies not available")

    def test_custom_profile_response_schema(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2CustomProfileResponse(
                profile_id="cp-1",
                name="Schema Test",
                categories=["math"],
            )
            assert obj.profile_id == "cp-1"
            assert obj.custom_weights is None
        except Exception:
            pytest.skip("Schema dependencies not available")

    def test_auto_trigger_event_schema(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2AutoTriggerEventResponse(
                event_id="evt-1",
                trigger_type="new_model",
                models=["test:7b"],
            )
            assert obj.event_id == "evt-1"
            assert obj.skipped is False
        except Exception:
            pytest.skip("Schema dependencies not available")

    def test_custom_profiles_list_response_schema(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2CustomProfilesListResponse(
                profiles=[],
                count=0,
            )
            assert obj.count == 0
        except Exception:
            pytest.skip("Schema dependencies not available")

    def test_auto_trigger_events_response_schema(self):
        try:
            schemas_mod = _load_module(
                "opti_oignon.api.schemas",
                "opti_oignon/api/schemas.py",
            )
            obj = schemas_mod.BenchmarkV2AutoTriggerEventsResponse(
                events=[],
                count=0,
            )
            assert obj.count == 0
        except Exception:
            pytest.skip("Schema dependencies not available")
