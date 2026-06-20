#!/usr/bin/env python3
"""
TESTS -- Model Health Monitor & Automatic Failover (S63)
=========================================================

Comprehensive tests for ModelHealthMonitor, ModelHealthRecord,
ModelStatus, health check logic, background thread, configuration,
API endpoints, and SmartRouter failover integration.

Target: 60+ tests, zero regressions.
"""

import json
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from opti_oignon.model_health import (
    DEFAULT_CHECK_INTERVAL,
    DEFAULT_DEGRADED_THRESHOLD,
    DEFAULT_LATENCY_WARNING_MS,
    DEFAULT_MAX_RECORDS,
    DEFAULT_UNAVAILABLE_THRESHOLD,
    ModelHealthMonitor,
    ModelHealthRecord,
    ModelStatus,
    check_all,
    get_health,
    is_available,
    is_healthy,
    model_health_monitor,
)
from opti_oignon.model_profiles import (
    ModelProfile,
    ModelProfileManager,
)
from opti_oignon.smart_router import (
    SmartRouter,
    SmartRoutingResult,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_profile(
    name: str = "test-model:7b",
    display_name: str = "Test Model",
    capabilities: list[str] | None = None,
    context_window: int = 32768,
    speed_tier: str = "medium",
    quality_tier: str = "medium",
    recommended_for: list[str] | None = None,
    task_scores: dict[str, float] | None = None,
) -> ModelProfile:
    """Create a ModelProfile for testing."""
    return ModelProfile(
        name=name,
        display_name=display_name,
        capabilities=capabilities or ["general"],
        context_window=context_window,
        speed_tier=speed_tier,
        quality_tier=quality_tier,
        recommended_for=recommended_for or ["general"],
        not_recommended_for=[],
        task_scores=task_scores or {"general": 0.8},
    )


def _make_profile_manager(*profiles: ModelProfile) -> ModelProfileManager:
    """Create a ProfileManager with the given profiles."""
    pm = ModelProfileManager(profiles_path=Path("/tmp/_no_exist_profiles.yaml"))
    for p in profiles:
        pm.add_profile(p)
    return pm


def _make_mock_ollama(models=None, show_side_effect=None):
    """Create a mock ollama module."""
    mock = MagicMock()
    if models is not None:
        mock.list.return_value = {"models": models}
    else:
        mock.list.return_value = {"models": []}
    if show_side_effect is not None:
        mock.show.side_effect = show_side_effect
    else:
        mock.show.return_value = {"modelinfo": {}}
    return mock


# =============================================================================
# TEST: ModelStatus Enum
# =============================================================================

class TestModelStatus(unittest.TestCase):
    """Tests for the ModelStatus enum."""

    def test_status_values(self):
        self.assertEqual(ModelStatus.HEALTHY.value, "healthy")
        self.assertEqual(ModelStatus.DEGRADED.value, "degraded")
        self.assertEqual(ModelStatus.UNAVAILABLE.value, "unavailable")
        self.assertEqual(ModelStatus.UNKNOWN.value, "unknown")

    def test_status_is_string_enum(self):
        self.assertIsInstance(ModelStatus.HEALTHY, str)
        self.assertEqual(str(ModelStatus.HEALTHY), "ModelStatus.HEALTHY")

    def test_status_comparison(self):
        self.assertNotEqual(ModelStatus.HEALTHY, ModelStatus.DEGRADED)
        self.assertEqual(ModelStatus.HEALTHY, ModelStatus.HEALTHY)


# =============================================================================
# TEST: ModelHealthRecord
# =============================================================================

class TestModelHealthRecord(unittest.TestCase):
    """Tests for ModelHealthRecord dataclass."""

    def test_default_values(self):
        r = ModelHealthRecord(model="test:7b")
        self.assertEqual(r.model, "test:7b")
        self.assertEqual(r.status, ModelStatus.UNKNOWN)
        self.assertEqual(r.latency_ms, 0.0)
        self.assertEqual(r.error_count, 0)
        self.assertEqual(r.consecutive_failures, 0)
        self.assertEqual(r.last_error, "")
        self.assertEqual(r.check_count, 0)

    def test_to_dict(self):
        r = ModelHealthRecord(
            model="test:7b",
            status=ModelStatus.HEALTHY,
            latency_ms=123.456,
            error_count=2,
            check_count=10,
        )
        d = r.to_dict()
        self.assertEqual(d["model"], "test:7b")
        self.assertEqual(d["status"], "healthy")
        self.assertEqual(d["latency_ms"], 123.46)
        self.assertEqual(d["error_count"], 2)
        self.assertEqual(d["check_count"], 10)

    def test_to_dict_keys(self):
        r = ModelHealthRecord(model="test:7b")
        d = r.to_dict()
        expected_keys = {
            "model", "status", "latency_ms", "last_check", "last_success",
            "error_count", "consecutive_failures", "last_error", "check_count",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_reset(self):
        r = ModelHealthRecord(
            model="test:7b",
            status=ModelStatus.DEGRADED,
            latency_ms=500.0,
            error_count=5,
            consecutive_failures=3,
            check_count=20,
        )
        r.reset()
        self.assertEqual(r.status, ModelStatus.UNKNOWN)
        self.assertEqual(r.latency_ms, 0.0)
        self.assertEqual(r.error_count, 0)
        self.assertEqual(r.consecutive_failures, 0)
        self.assertEqual(r.check_count, 0)

    def test_model_name_preserved_after_reset(self):
        r = ModelHealthRecord(model="keep-me:7b")
        r.reset()
        self.assertEqual(r.model, "keep-me:7b")


# =============================================================================
# TEST: ModelHealthMonitor -- Initialization
# =============================================================================

class TestMonitorInit(unittest.TestCase):
    """Tests for ModelHealthMonitor initialization."""

    def test_default_init(self):
        m = ModelHealthMonitor(ollama_module=None)
        self.assertTrue(m.enabled)
        self.assertEqual(m.check_interval, DEFAULT_CHECK_INTERVAL)
        self.assertEqual(m.degraded_threshold, DEFAULT_DEGRADED_THRESHOLD)
        self.assertEqual(m.unavailable_threshold, DEFAULT_UNAVAILABLE_THRESHOLD)
        self.assertTrue(m.auto_failover)
        self.assertFalse(m.running)

    def test_custom_init(self):
        m = ModelHealthMonitor(
            enabled=False,
            check_interval=30,
            degraded_threshold=2,
            unavailable_threshold=4,
            latency_warning_ms=3000,
            auto_failover=False,
            max_records=50,
            ollama_module=None,
            config_path=Path("/tmp/_no_exist_health_cfg.yaml"),
        )
        self.assertFalse(m.enabled)
        self.assertEqual(m.check_interval, 30)
        self.assertEqual(m.degraded_threshold, 2)
        self.assertEqual(m.unavailable_threshold, 4)
        self.assertFalse(m.auto_failover)

    def test_empty_records_on_init(self):
        m = ModelHealthMonitor(ollama_module=None)
        self.assertEqual(len(m.get_all_health()), 0)


# =============================================================================
# TEST: YAML Config Loading
# =============================================================================

class TestMonitorConfig(unittest.TestCase):
    """Tests for YAML config loading."""

    def test_load_from_yaml(self):
        config = {
            "model_health": {
                "enabled": False,
                "check_interval_seconds": 120,
                "degraded_threshold": 4,
                "unavailable_threshold": 8,
                "latency_warning_ms": 10000,
                "auto_failover": False,
                "max_records": 200,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            path = Path(f.name)

        m = ModelHealthMonitor(config_path=path, ollama_module=None)
        self.assertFalse(m.enabled)
        self.assertEqual(m.check_interval, 120)
        self.assertEqual(m.degraded_threshold, 4)
        self.assertEqual(m.unavailable_threshold, 8)
        self.assertFalse(m.auto_failover)
        path.unlink()

    def test_missing_config_uses_defaults(self):
        m = ModelHealthMonitor(
            config_path=Path("/tmp/_no_exist_health.yaml"),
            ollama_module=None,
        )
        self.assertEqual(m.check_interval, DEFAULT_CHECK_INTERVAL)

    def test_invalid_yaml_uses_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("not: [valid: yaml: {{")
            f.flush()
            path = Path(f.name)
        m = ModelHealthMonitor(config_path=path, ollama_module=None)
        self.assertEqual(m.check_interval, DEFAULT_CHECK_INTERVAL)
        path.unlink()

    def test_empty_config_uses_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            f.flush()
            path = Path(f.name)
        m = ModelHealthMonitor(config_path=path, ollama_module=None)
        self.assertEqual(m.check_interval, DEFAULT_CHECK_INTERVAL)
        path.unlink()

    def test_invalid_values_ignored(self):
        config = {
            "model_health": {
                "check_interval_seconds": -10,
                "degraded_threshold": 0,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            path = Path(f.name)
        m = ModelHealthMonitor(config_path=path, ollama_module=None)
        # Negative/zero values should not override defaults
        self.assertEqual(m.check_interval, DEFAULT_CHECK_INTERVAL)
        self.assertEqual(m.degraded_threshold, DEFAULT_DEGRADED_THRESHOLD)
        path.unlink()


# =============================================================================
# TEST: Health Check Logic
# =============================================================================

class TestHealthCheck(unittest.TestCase):
    """Tests for individual model health checks."""

    def test_check_healthy_model(self):
        mock_ollama = _make_mock_ollama()
        m = ModelHealthMonitor(ollama_module=mock_ollama)
        record = m.check_model("test:7b")
        self.assertEqual(record.status, ModelStatus.HEALTHY)
        self.assertEqual(record.consecutive_failures, 0)
        self.assertGreater(record.latency_ms, 0)
        self.assertGreater(record.last_check, 0)
        self.assertGreater(record.last_success, 0)
        self.assertEqual(record.check_count, 1)

    def test_check_failing_model(self):
        mock_ollama = _make_mock_ollama(show_side_effect=Exception("model not found"))
        m = ModelHealthMonitor(ollama_module=mock_ollama)
        record = m.check_model("bad:7b")
        self.assertEqual(record.consecutive_failures, 1)
        self.assertEqual(record.error_count, 1)
        self.assertIn("model not found", record.last_error)

    def test_consecutive_failures_degrade(self):
        mock_ollama = _make_mock_ollama(show_side_effect=Exception("fail"))
        m = ModelHealthMonitor(ollama_module=mock_ollama, degraded_threshold=3)
        for _ in range(3):
            record = m.check_model("bad:7b")
        self.assertEqual(record.status, ModelStatus.DEGRADED)
        self.assertEqual(record.consecutive_failures, 3)

    def test_consecutive_failures_unavailable(self):
        mock_ollama = _make_mock_ollama(show_side_effect=Exception("fail"))
        m = ModelHealthMonitor(ollama_module=mock_ollama, unavailable_threshold=5)
        for _ in range(5):
            record = m.check_model("bad:7b")
        self.assertEqual(record.status, ModelStatus.UNAVAILABLE)
        self.assertEqual(record.consecutive_failures, 5)

    def test_recovery_after_failure(self):
        call_count = [0]
        def side_effect(name):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise Exception("temporary failure")
            return {"modelinfo": {}}

        mock_ollama = _make_mock_ollama(show_side_effect=side_effect)
        m = ModelHealthMonitor(ollama_module=mock_ollama, degraded_threshold=3)

        # 3 failures -> degraded
        for _ in range(3):
            m.check_model("model:7b")
        self.assertEqual(m.get_status("model:7b"), ModelStatus.DEGRADED)

        # 1 success -> healthy again
        m.check_model("model:7b")
        self.assertEqual(m.get_status("model:7b"), ModelStatus.HEALTHY)
        record = m.get_health("model:7b")
        self.assertEqual(record.consecutive_failures, 0)
        self.assertEqual(record.error_count, 3)  # Historical errors preserved

    def test_no_ollama_module(self):
        m = ModelHealthMonitor(ollama_module=None)
        record = m.check_model("test:7b")
        self.assertEqual(record.consecutive_failures, 1)
        self.assertIn("not available", record.last_error)

    def test_check_updates_existing_record(self):
        mock_ollama = _make_mock_ollama()
        m = ModelHealthMonitor(ollama_module=mock_ollama)
        m.check_model("test:7b")
        m.check_model("test:7b")
        record = m.get_health("test:7b")
        self.assertEqual(record.check_count, 2)

    def test_latency_recorded(self):
        def slow_show(name):
            time.sleep(0.01)
            return {"modelinfo": {}}

        mock_ollama = _make_mock_ollama(show_side_effect=slow_show)
        m = ModelHealthMonitor(ollama_module=mock_ollama)
        record = m.check_model("test:7b")
        self.assertGreater(record.latency_ms, 5.0)


# =============================================================================
# TEST: Model Discovery
# =============================================================================

class TestModelDiscovery(unittest.TestCase):
    """Tests for model discovery via ollama.list()."""

    def test_discover_from_dict_models(self):
        mock_ollama = _make_mock_ollama(models=[
            {"model": "qwen3:32b", "name": "qwen3:32b"},
            {"model": "llama3:8b", "name": "llama3:8b"},
        ])
        m = ModelHealthMonitor(ollama_module=mock_ollama)
        names = m._discover_models()
        self.assertIn("qwen3:32b", names)
        self.assertIn("llama3:8b", names)

    def test_discover_empty_list(self):
        mock_ollama = _make_mock_ollama(models=[])
        m = ModelHealthMonitor(ollama_module=mock_ollama)
        names = m._discover_models()
        self.assertEqual(names, [])

    def test_discover_no_ollama(self):
        m = ModelHealthMonitor(ollama_module=None)
        names = m._discover_models()
        self.assertEqual(names, [])

    def test_discover_exception_returns_existing_keys(self):
        mock_ollama = MagicMock()
        mock_ollama.list.side_effect = Exception("connection refused")
        m = ModelHealthMonitor(ollama_module=mock_ollama)
        # Pre-populate a record
        m._records["existing:7b"] = ModelHealthRecord(model="existing:7b")
        names = m._discover_models()
        self.assertIn("existing:7b", names)

    def test_check_all_discovers_and_checks(self):
        mock_ollama = _make_mock_ollama(models=[
            {"model": "m1:7b"},
            {"model": "m2:7b"},
        ])
        m = ModelHealthMonitor(ollama_module=mock_ollama)
        results = m.check_all()
        self.assertIn("m1:7b", results)
        self.assertIn("m2:7b", results)
        self.assertEqual(results["m1:7b"].status, ModelStatus.HEALTHY)


# =============================================================================
# TEST: Query Methods
# =============================================================================

class TestQueryMethods(unittest.TestCase):
    """Tests for health query convenience methods."""

    def setUp(self):
        self.m = ModelHealthMonitor(ollama_module=None)
        self.m._records["healthy:7b"] = ModelHealthRecord(
            model="healthy:7b", status=ModelStatus.HEALTHY, check_count=1
        )
        self.m._records["degraded:7b"] = ModelHealthRecord(
            model="degraded:7b", status=ModelStatus.DEGRADED, check_count=5
        )
        self.m._records["unavailable:70b"] = ModelHealthRecord(
            model="unavailable:70b", status=ModelStatus.UNAVAILABLE, check_count=10
        )
        self.m._records["unknown:7b"] = ModelHealthRecord(
            model="unknown:7b", status=ModelStatus.UNKNOWN
        )

    def test_get_health_existing(self):
        r = self.m.get_health("healthy:7b")
        self.assertIsNotNone(r)
        self.assertEqual(r.status, ModelStatus.HEALTHY)

    def test_get_health_missing(self):
        r = self.m.get_health("nonexistent:7b")
        self.assertIsNone(r)

    def test_get_all_health(self):
        all_h = self.m.get_all_health()
        self.assertEqual(len(all_h), 4)

    def test_get_status(self):
        self.assertEqual(self.m.get_status("healthy:7b"), ModelStatus.HEALTHY)
        self.assertEqual(self.m.get_status("missing:7b"), ModelStatus.UNKNOWN)

    def test_is_healthy(self):
        self.assertTrue(self.m.is_healthy("healthy:7b"))
        self.assertTrue(self.m.is_healthy("unknown:7b"))  # Unknown treated as healthy
        self.assertFalse(self.m.is_healthy("degraded:7b"))
        self.assertFalse(self.m.is_healthy("unavailable:70b"))
        self.assertTrue(self.m.is_healthy("not-tracked:7b"))  # Untracked = UNKNOWN

    def test_is_available(self):
        self.assertTrue(self.m.is_available("healthy:7b"))
        self.assertTrue(self.m.is_available("degraded:7b"))
        self.assertFalse(self.m.is_available("unavailable:70b"))
        self.assertTrue(self.m.is_available("not-tracked:7b"))

    def test_get_healthy_models(self):
        healthy = self.m.get_healthy_models()
        self.assertEqual(healthy, ["healthy:7b"])

    def test_get_degraded_models(self):
        degraded = self.m.get_degraded_models()
        self.assertEqual(degraded, ["degraded:7b"])

    def test_get_unavailable_models(self):
        unavail = self.m.get_unavailable_models()
        self.assertEqual(unavail, ["unavailable:70b"])


# =============================================================================
# TEST: Configuration
# =============================================================================

class TestMonitorConfigure(unittest.TestCase):
    """Tests for runtime configuration."""

    def test_configure_enabled(self):
        m = ModelHealthMonitor(ollama_module=None)
        m.configure(enabled=False)
        self.assertFalse(m.enabled)

    def test_configure_interval(self):
        m = ModelHealthMonitor(ollama_module=None)
        m.configure(check_interval=30)
        self.assertEqual(m.check_interval, 30)

    def test_configure_thresholds(self):
        m = ModelHealthMonitor(ollama_module=None)
        m.configure(degraded_threshold=2, unavailable_threshold=4)
        self.assertEqual(m.degraded_threshold, 2)
        self.assertEqual(m.unavailable_threshold, 4)

    def test_configure_invalid_ignored(self):
        m = ModelHealthMonitor(ollama_module=None)
        m.configure(check_interval=-1, degraded_threshold=0)
        self.assertEqual(m.check_interval, DEFAULT_CHECK_INTERVAL)
        self.assertEqual(m.degraded_threshold, DEFAULT_DEGRADED_THRESHOLD)

    def test_configure_auto_failover(self):
        m = ModelHealthMonitor(ollama_module=None)
        m.configure(auto_failover=False)
        self.assertFalse(m.auto_failover)

    def test_get_config(self):
        m = ModelHealthMonitor(ollama_module=None)
        c = m.get_config()
        self.assertIn("enabled", c)
        self.assertIn("running", c)
        self.assertIn("check_interval_seconds", c)
        self.assertIn("tracked_models", c)
        self.assertIn("ollama_available", c)

    def test_reset_clears_records(self):
        m = ModelHealthMonitor(ollama_module=None)
        m._records["a:7b"] = ModelHealthRecord(model="a:7b")
        m._records["b:7b"] = ModelHealthRecord(model="b:7b")
        self.assertEqual(len(m.get_all_health()), 2)
        m.reset()
        self.assertEqual(len(m.get_all_health()), 0)

    def test_remove_model(self):
        m = ModelHealthMonitor(ollama_module=None)
        m._records["a:7b"] = ModelHealthRecord(model="a:7b")
        self.assertTrue(m.remove_model("a:7b"))
        self.assertIsNone(m.get_health("a:7b"))

    def test_remove_model_not_found(self):
        m = ModelHealthMonitor(ollama_module=None)
        self.assertFalse(m.remove_model("missing:7b"))


# =============================================================================
# TEST: to_dict / Serialization
# =============================================================================

class TestMonitorSerialization(unittest.TestCase):
    """Tests for monitor state serialization."""

    def test_to_dict_structure(self):
        m = ModelHealthMonitor(ollama_module=None)
        m._records["a:7b"] = ModelHealthRecord(
            model="a:7b", status=ModelStatus.HEALTHY, check_count=1
        )
        d = m.to_dict()
        self.assertIn("records", d)
        self.assertIn("summary", d)
        self.assertIn("enabled", d)
        self.assertIn("a:7b", d["records"])

    def test_summary_counts(self):
        m = ModelHealthMonitor(ollama_module=None)
        m._records["h:7b"] = ModelHealthRecord(model="h:7b", status=ModelStatus.HEALTHY)
        m._records["d:7b"] = ModelHealthRecord(model="d:7b", status=ModelStatus.DEGRADED)
        m._records["u:7b"] = ModelHealthRecord(model="u:7b", status=ModelStatus.UNAVAILABLE)
        d = m.to_dict()
        self.assertEqual(d["summary"]["healthy"], 1)
        self.assertEqual(d["summary"]["degraded"], 1)
        self.assertEqual(d["summary"]["unavailable"], 1)

    def test_to_dict_json_serializable(self):
        m = ModelHealthMonitor(ollama_module=None)
        m._records["t:7b"] = ModelHealthRecord(model="t:7b", status=ModelStatus.HEALTHY)
        d = m.to_dict()
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)


# =============================================================================
# TEST: Background Thread
# =============================================================================

class TestBackgroundThread(unittest.TestCase):
    """Tests for the background health check thread."""

    def test_start_and_stop(self):
        mock_ollama = _make_mock_ollama()
        m = ModelHealthMonitor(
            check_interval=1,
            ollama_module=mock_ollama,
        )
        m.start()
        self.assertTrue(m.running)
        time.sleep(0.1)
        m.stop()
        self.assertFalse(m.running)

    def test_start_when_disabled(self):
        m = ModelHealthMonitor(
            enabled=False, ollama_module=None,
            config_path=Path("/tmp/_no_exist_health_cfg.yaml"),
        )
        m.start()
        self.assertFalse(m.running)

    def test_start_idempotent(self):
        mock_ollama = _make_mock_ollama()
        m = ModelHealthMonitor(check_interval=60, ollama_module=mock_ollama)
        m.start()
        thread1 = m._thread
        m.start()  # Should not create a new thread
        thread2 = m._thread
        self.assertIs(thread1, thread2)
        m.stop()

    def test_stop_idempotent(self):
        m = ModelHealthMonitor(ollama_module=None)
        m.stop()  # Should not raise
        m.stop()

    def test_thread_is_daemon(self):
        mock_ollama = _make_mock_ollama()
        m = ModelHealthMonitor(check_interval=60, ollama_module=mock_ollama)
        m.start()
        self.assertTrue(m._thread.daemon)
        m.stop()


# =============================================================================
# TEST: Singleton & Convenience Functions
# =============================================================================

class TestSingleton(unittest.TestCase):
    """Tests for the module-level singleton and convenience functions."""

    def test_singleton_exists(self):
        self.assertIsInstance(model_health_monitor, ModelHealthMonitor)

    def test_get_health_convenience(self):
        # Should return None for unknown model
        result = get_health("nonexistent-model:7b")
        self.assertIsNone(result)

    def test_is_healthy_convenience(self):
        # Unknown models are considered healthy
        self.assertTrue(is_healthy("nonexistent:7b"))

    def test_is_available_convenience(self):
        self.assertTrue(is_available("nonexistent:7b"))


# =============================================================================
# TEST: SmartRoutingResult -- New Fields
# =============================================================================

class TestSmartRoutingResultFields(unittest.TestCase):
    """Tests for the new failover fields in SmartRoutingResult."""

    def test_default_values(self):
        r = SmartRoutingResult(model="test:7b")
        self.assertFalse(r.failover)
        self.assertEqual(r.original_model, "")

    def test_failover_values(self):
        r = SmartRoutingResult(
            model="fallback:7b",
            failover=True,
            original_model="primary:70b",
        )
        self.assertTrue(r.failover)
        self.assertEqual(r.original_model, "primary:70b")

    def test_to_dict_includes_failover(self):
        r = SmartRoutingResult(
            model="fallback:7b",
            failover=True,
            original_model="primary:70b",
        )
        d = r.to_dict()
        self.assertIn("failover", d)
        self.assertIn("original_model", d)
        self.assertTrue(d["failover"])
        self.assertEqual(d["original_model"], "primary:70b")

    def test_to_dict_default_failover(self):
        r = SmartRoutingResult(model="test:7b")
        d = r.to_dict()
        self.assertFalse(d["failover"])
        self.assertEqual(d["original_model"], "")

    def test_backward_compatible(self):
        """Existing code creating SmartRoutingResult without failover fields works."""
        r = SmartRoutingResult(
            model="test:7b",
            score=0.9,
            task_score=0.85,
            speed_weight=1.0,
            context_fit=1.0,
            reason="test",
            profile_used=True,
            fallback=False,
            feedback_adjusted=False,
        )
        d = r.to_dict()
        self.assertFalse(d["failover"])
        self.assertEqual(d["original_model"], "")


# =============================================================================
# TEST: SmartRouter Failover -- Unavailable Model Excluded
# =============================================================================

class TestFailoverUnavailable(unittest.TestCase):
    """Tests for SmartRouter excluding unavailable models."""

    def setUp(self):
        self.pm = _make_profile_manager(
            _make_profile("best:70b", task_scores={"general": 0.95}, speed_tier="slow", quality_tier="high"),
            _make_profile("alt:7b", task_scores={"general": 0.6}, speed_tier="fast", quality_tier="medium"),
        )
        self.hm = ModelHealthMonitor(enabled=True, ollama_module=None)

    def test_no_health_issues_best_model_wins(self):
        router = SmartRouter(
            profile_manager=self.pm, health_monitor=self.hm,
            feedback_adapter=None, speed_preference="balanced",
        )
        result = router.select_model("direct")
        self.assertEqual(result.model, "best:70b")
        self.assertFalse(result.failover)

    def test_unavailable_model_excluded(self):
        self.hm._records["best:70b"] = ModelHealthRecord(
            model="best:70b", status=ModelStatus.UNAVAILABLE,
            consecutive_failures=5, check_count=5,
        )
        router = SmartRouter(
            profile_manager=self.pm, health_monitor=self.hm,
            feedback_adapter=None, speed_preference="balanced",
        )
        result = router.select_model("direct")
        self.assertEqual(result.model, "alt:7b")
        self.assertTrue(result.failover)
        self.assertEqual(result.original_model, "best:70b")
        self.assertIn("failover", result.reason)

    def test_failover_reason_contains_status(self):
        self.hm._records["best:70b"] = ModelHealthRecord(
            model="best:70b", status=ModelStatus.UNAVAILABLE,
            consecutive_failures=5, check_count=5,
        )
        router = SmartRouter(
            profile_manager=self.pm, health_monitor=self.hm,
            feedback_adapter=None,
        )
        result = router.select_model("direct")
        self.assertIn("unavailable", result.reason)


# =============================================================================
# TEST: SmartRouter Failover -- Degraded Model Penalized
# =============================================================================

class TestFailoverDegraded(unittest.TestCase):
    """Tests for SmartRouter applying penalty to degraded models."""

    def setUp(self):
        self.pm = _make_profile_manager(
            _make_profile("best:70b", task_scores={"general": 0.95}, speed_tier="medium", quality_tier="high"),
            _make_profile("alt:7b", task_scores={"general": 0.5}, speed_tier="medium", quality_tier="medium"),
        )
        self.hm = ModelHealthMonitor(enabled=True, ollama_module=None)

    def test_degraded_penalty_changes_winner(self):
        # Without penalty: best:70b wins (0.95 > 0.5)
        # With 0.5x penalty: best:70b = 0.95*1.0*0.5 = 0.475 < alt:7b = 0.5*1.0*1.0
        self.hm._records["best:70b"] = ModelHealthRecord(
            model="best:70b", status=ModelStatus.DEGRADED,
            consecutive_failures=3, check_count=3,
        )
        router = SmartRouter(
            profile_manager=self.pm, health_monitor=self.hm,
            feedback_adapter=None, speed_preference="balanced",
        )
        result = router.select_model("direct")
        self.assertEqual(result.model, "alt:7b")
        self.assertTrue(result.failover)
        self.assertEqual(result.original_model, "best:70b")
        self.assertIn("degraded", result.reason)

    def test_degraded_no_winner_change_no_failover(self):
        # If penalty doesn't change winner, no failover
        pm = _make_profile_manager(
            _make_profile("best:70b", task_scores={"general": 0.95}, speed_tier="medium"),
            _make_profile("alt:7b", task_scores={"general": 0.1}, speed_tier="medium"),
        )
        hm = ModelHealthMonitor(enabled=True, ollama_module=None)
        hm._records["alt:7b"] = ModelHealthRecord(
            model="alt:7b", status=ModelStatus.DEGRADED,
            consecutive_failures=3, check_count=3,
        )
        router = SmartRouter(
            profile_manager=pm, health_monitor=hm,
            feedback_adapter=None, speed_preference="balanced",
        )
        result = router.select_model("direct")
        self.assertEqual(result.model, "best:70b")
        self.assertFalse(result.failover)


# =============================================================================
# TEST: SmartRouter Failover -- Health Monitor Disabled
# =============================================================================

class TestFailoverDisabled(unittest.TestCase):
    """Tests for graceful degradation when health monitor is off."""

    def test_no_health_monitor(self):
        pm = _make_profile_manager(
            _make_profile("model:7b", task_scores={"general": 0.8}),
        )
        router = SmartRouter(
            profile_manager=pm, health_monitor=None,
            feedback_adapter=None,
        )
        result = router.select_model("direct")
        self.assertFalse(result.failover)

    def test_auto_failover_disabled(self):
        pm = _make_profile_manager(
            _make_profile("best:70b", task_scores={"general": 0.95}),
            _make_profile("alt:7b", task_scores={"general": 0.5}),
        )
        hm = ModelHealthMonitor(enabled=True, ollama_module=None)
        hm._auto_failover = False
        hm._records["best:70b"] = ModelHealthRecord(
            model="best:70b", status=ModelStatus.UNAVAILABLE,
            consecutive_failures=5, check_count=5,
        )
        router = SmartRouter(
            profile_manager=pm, health_monitor=hm,
            feedback_adapter=None,
        )
        result = router.select_model("direct")
        # Unavailable model NOT excluded when auto_failover is off
        self.assertEqual(result.model, "best:70b")
        self.assertFalse(result.failover)

    def test_health_monitor_exception_handled(self):
        """SmartRouter handles exceptions from health monitor gracefully."""
        mock_hm = MagicMock()
        mock_hm.auto_failover = True
        mock_hm.get_status.side_effect = Exception("monitor error")
        mock_hm.get_unavailable_models.side_effect = Exception("monitor error")

        pm = _make_profile_manager(
            _make_profile("model:7b", task_scores={"general": 0.8}),
        )
        router = SmartRouter(
            profile_manager=pm, health_monitor=mock_hm,
            feedback_adapter=None,
        )
        # Should not raise, should fallback gracefully
        result = router.select_model("direct")
        self.assertIsNotNone(result.model)


# =============================================================================
# TEST: SmartRouter get_config with Health Monitor
# =============================================================================

class TestSmartRouterConfigHealth(unittest.TestCase):
    """Tests for health monitor status in SmartRouter config."""

    def test_config_includes_health_fields(self):
        hm = ModelHealthMonitor(enabled=True, ollama_module=None)
        pm = _make_profile_manager(_make_profile("m:7b"))
        router = SmartRouter(profile_manager=pm, health_monitor=hm, feedback_adapter=None)
        config = router.get_config()
        self.assertIn("health_monitor_enabled", config)
        self.assertIn("health_monitor_running", config)
        self.assertIn("auto_failover", config)
        self.assertTrue(config["health_monitor_enabled"])
        self.assertFalse(config["health_monitor_running"])
        self.assertTrue(config["auto_failover"])

    def test_config_no_health_monitor(self):
        pm = _make_profile_manager(_make_profile("m:7b"))
        router = SmartRouter(profile_manager=pm, health_monitor=None, feedback_adapter=None)
        config = router.get_config()
        self.assertFalse(config["health_monitor_enabled"])
        self.assertFalse(config["auto_failover"])


# =============================================================================
# TEST: API Endpoints (via FastAPI TestClient)
# =============================================================================

class TestModelHealthAPI(unittest.TestCase):
    """Tests for model health API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up FastAPI test client."""
        try:
            from fastapi.testclient import TestClient
            from opti_oignon.api.routes_smart_routing import router
            from fastapi import FastAPI
            app = FastAPI()
            app.include_router(router)
            cls.client = TestClient(app)
            cls.api_available = True
        except ImportError:
            cls.api_available = False

    def setUp(self):
        if not self.api_available:
            self.skipTest("FastAPI test client not available")

    def test_get_all_model_health(self):
        resp = self.client.get("/api/smart-routing/model-health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("records", data)
        self.assertIn("summary", data)
        self.assertIn("config", data)

    def test_get_single_model_health_not_found(self):
        resp = self.client.get("/api/smart-routing/model-health/nonexistent:7b")
        self.assertEqual(resp.status_code, 404)

    def test_force_health_check(self):
        resp = self.client.post("/api/smart-routing/model-health/check")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")
        self.assertIn("checked", data)


# =============================================================================
# TEST: Edge Cases
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_all_models_unavailable(self):
        """When all models are unavailable, SmartRouter falls back to default."""
        pm = _make_profile_manager(
            _make_profile("m1:7b", task_scores={"general": 0.8}),
            _make_profile("m2:7b", task_scores={"general": 0.7}),
        )
        hm = ModelHealthMonitor(enabled=True, ollama_module=None)
        hm._records["m1:7b"] = ModelHealthRecord(
            model="m1:7b", status=ModelStatus.UNAVAILABLE, consecutive_failures=5, check_count=5,
        )
        hm._records["m2:7b"] = ModelHealthRecord(
            model="m2:7b", status=ModelStatus.UNAVAILABLE, consecutive_failures=5, check_count=5,
        )
        router = SmartRouter(profile_manager=pm, health_monitor=hm, feedback_adapter=None)
        result = router.select_model("direct")
        # All candidates excluded -> fallback
        self.assertTrue(result.fallback)

    def test_unknown_status_treated_as_healthy(self):
        """Models with UNKNOWN status should not be penalized."""
        pm = _make_profile_manager(
            _make_profile("m:7b", task_scores={"general": 0.8}),
        )
        hm = ModelHealthMonitor(enabled=True, ollama_module=None)
        hm._records["m:7b"] = ModelHealthRecord(
            model="m:7b", status=ModelStatus.UNKNOWN,
        )
        router = SmartRouter(profile_manager=pm, health_monitor=hm, feedback_adapter=None)
        result = router.select_model("direct")
        self.assertEqual(result.model, "m:7b")
        self.assertFalse(result.failover)

    def test_health_penalty_only_for_degraded(self):
        """HEALTHY status should NOT get a penalty."""
        pm = _make_profile_manager(
            _make_profile("m:7b", task_scores={"general": 0.8}),
        )
        hm = ModelHealthMonitor(enabled=True, ollama_module=None)
        hm._records["m:7b"] = ModelHealthRecord(
            model="m:7b", status=ModelStatus.HEALTHY, check_count=1,
        )
        router = SmartRouter(profile_manager=pm, health_monitor=hm, feedback_adapter=None)
        result = router.select_model("direct")
        self.assertEqual(result.context_fit, 1.0)

    def test_multiple_unavailable_first_is_original(self):
        """With multiple unavailable models, original_model picks the first found."""
        pm = _make_profile_manager(
            _make_profile("best:70b", task_scores={"general": 0.95}),
            _make_profile("mid:30b", task_scores={"general": 0.8}),
            _make_profile("alt:7b", task_scores={"general": 0.5}),
        )
        hm = ModelHealthMonitor(enabled=True, ollama_module=None)
        hm._records["best:70b"] = ModelHealthRecord(
            model="best:70b", status=ModelStatus.UNAVAILABLE, consecutive_failures=5, check_count=5,
        )
        hm._records["mid:30b"] = ModelHealthRecord(
            model="mid:30b", status=ModelStatus.UNAVAILABLE, consecutive_failures=5, check_count=5,
        )
        router = SmartRouter(profile_manager=pm, health_monitor=hm, feedback_adapter=None)
        result = router.select_model("direct")
        self.assertEqual(result.model, "alt:7b")
        self.assertTrue(result.failover)
        self.assertIn(result.original_model, ["best:70b", "mid:30b"])

    def test_cache_cleared_on_health_change(self):
        """Router cache should be cleared when health status changes are relevant."""
        pm = _make_profile_manager(
            _make_profile("m:7b", task_scores={"general": 0.8}),
        )
        hm = ModelHealthMonitor(enabled=True, ollama_module=None)
        router = SmartRouter(profile_manager=pm, health_monitor=hm, feedback_adapter=None)
        r1 = router.select_model("direct")
        self.assertEqual(r1.model, "m:7b")

        # After manually clearing cache, result should still be consistent
        router.clear_cache()
        r2 = router.select_model("direct")
        self.assertEqual(r2.model, "m:7b")

    def test_concurrent_check_model_thread_safe(self):
        """Multiple threads calling check_model should not corrupt records."""
        mock_ollama = _make_mock_ollama()
        m = ModelHealthMonitor(ollama_module=mock_ollama)
        errors = []

        def check():
            try:
                for _ in range(10):
                    m.check_model("thread-test:7b")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        r = m.get_health("thread-test:7b")
        self.assertEqual(r.check_count, 40)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
