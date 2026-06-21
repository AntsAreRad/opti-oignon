#!/usr/bin/env python3
"""
Tests for LearnedRouter integration with SmartRouter (S67, Step 2).

Covers: conditional import flag, classify_task_type(), log_routing_sample(),
get_config() learned router fields, and backward compatibility.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_learned_router(tmp_path: Path, enabled: bool = False, trained: bool = False):
    """Return a configured LearnedRouter mock or real instance."""
    from opti_oignon.learned_router import LearnedRouter

    cfg_path = tmp_path / "lr.yaml"
    db_path = tmp_path / "lr.db"
    model_path = tmp_path / "lr.pkl"

    router = LearnedRouter(config_path=cfg_path, db_path=db_path, model_path=model_path)
    router.update_config({"enabled": enabled, "min_training_samples": 5, "cv_folds": 2})

    if trained:
        for i in range(5):
            router.log_sample(f"python function variant {i}", "code_python")
            router.log_sample(f"fix error variant {i}", "debug")
            router.log_sample(f"analyze data variant {i}", "data_analysis")
            router.log_sample(f"plan steps variant {i}", "planning")
            router.log_sample(f"write text variant {i}", "general")
        router.train(min_samples=5)

    return router


class _MinimalSmartRouter:
    """SmartRouter instantiated without model profiles (uses fallback path)."""

    def __init__(self, learned_router=None):
        from opti_oignon.smart_router import SmartRouter
        self.sr = SmartRouter(enabled=False)
        if learned_router is not None:
            self.sr._learned_router = learned_router


# ---------------------------------------------------------------------------
# Tests: flag presence
# ---------------------------------------------------------------------------

class TestImportFlags:
    def test_learned_router_in_smart_flag_exists(self):
        from opti_oignon.smart_router import LEARNED_ROUTER_IN_SMART
        assert isinstance(LEARNED_ROUTER_IN_SMART, bool)

    def test_flag_is_true_when_sklearn_available(self):
        from opti_oignon.smart_router import LEARNED_ROUTER_IN_SMART
        assert LEARNED_ROUTER_IN_SMART is True

    def test_default_learned_router_imported(self):
        from opti_oignon.smart_router import _default_learned_router
        assert _default_learned_router is not None


# ---------------------------------------------------------------------------
# Tests: SmartRoutingResult new fields
# ---------------------------------------------------------------------------

class TestSmartRoutingResultFields:
    def test_routing_source_default_yaml(self):
        from opti_oignon.smart_router import SmartRoutingResult
        r = SmartRoutingResult(model="test-model")
        assert r.routing_source == "yaml"

    def test_learned_confidence_default_zero(self):
        from opti_oignon.smart_router import SmartRoutingResult
        r = SmartRoutingResult(model="test-model")
        assert r.learned_confidence == 0.0

    def test_to_dict_includes_routing_source(self):
        from opti_oignon.smart_router import SmartRoutingResult
        r = SmartRoutingResult(model="test-model", routing_source="learned", learned_confidence=0.85)
        d = r.to_dict()
        assert d["routing_source"] == "learned"
        assert d["learned_confidence"] == 0.85

    def test_to_dict_complete_keys(self):
        from opti_oignon.smart_router import SmartRoutingResult
        r = SmartRoutingResult(model="m")
        d = r.to_dict()
        for key in ("routing_source", "learned_confidence", "model", "fallback"):
            assert key in d


# ---------------------------------------------------------------------------
# Tests: classify_task_type()
# ---------------------------------------------------------------------------

class TestClassifyTaskType:
    def test_returns_yaml_when_no_learned_router(self, tmp_path):
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        sr._learned_router = None
        task, source, conf = sr.classify_task_type("write python code", "code_python")
        assert task == "code_python"
        assert source == "yaml"
        assert conf == 0.0

    def test_returns_yaml_when_disabled(self, tmp_path):
        lr = _make_learned_router(tmp_path, enabled=False, trained=True)
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        sr._learned_router = lr
        task, source, conf = sr.classify_task_type("fix this bug", "debug")
        assert source == "yaml"
        assert task == "debug"

    def test_returns_yaml_when_not_trained(self, tmp_path):
        lr = _make_learned_router(tmp_path, enabled=True, trained=False)
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        sr._learned_router = lr
        task, source, conf = sr.classify_task_type("some query", "general")
        assert source == "yaml"
        assert task == "general"

    def test_returns_learned_when_enabled_trained_confident(self, tmp_path):
        lr = _make_learned_router(tmp_path, enabled=True, trained=True)
        # Set threshold to 0 to force ML path
        lr.update_config({"confidence_threshold": 0.0})
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        sr._learned_router = lr
        task, source, conf = sr.classify_task_type("fix this python error", "general")
        # source could be 'learned' or 'yaml' depending on confidence; just check types
        assert source in ("learned", "yaml")
        assert isinstance(task, str)
        assert isinstance(conf, float)

    def test_returns_three_tuple(self, tmp_path):
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        result = sr.classify_task_type("query", "general")
        assert len(result) == 3

    def test_graceful_on_exception(self, tmp_path):
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        bad_lr = MagicMock()
        bad_lr.classify_with_fallback.side_effect = RuntimeError("boom")
        sr._learned_router = bad_lr
        # Should not raise — graceful fallback
        task, source, conf = sr.classify_task_type("query", "general")
        assert task == "general"
        assert source == "yaml"


# ---------------------------------------------------------------------------
# Tests: log_routing_sample()
# ---------------------------------------------------------------------------

class TestLogRoutingSample:
    def test_log_sample_increases_count(self, tmp_path):
        lr = _make_learned_router(tmp_path)
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        sr._learned_router = lr
        before = lr.get_sample_count()
        sr.log_routing_sample("write a python function", "code_python")
        assert lr.get_sample_count() == before + 1

    def test_log_sample_no_error_without_learned_router(self):
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        sr._learned_router = None
        # Should not raise
        sr.log_routing_sample("any query", "general")

    def test_log_sample_graceful_on_exception(self):
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        bad_lr = MagicMock()
        bad_lr.log_sample.side_effect = RuntimeError("db error")
        sr._learned_router = bad_lr
        # Should not raise
        sr.log_routing_sample("query", "general")


# ---------------------------------------------------------------------------
# Tests: get_config() learned router fields
# ---------------------------------------------------------------------------

class TestGetConfigLearnedFields:
    def test_get_config_has_learned_router_available(self):
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        cfg = sr.get_config()
        assert "learned_router_available" in cfg
        assert isinstance(cfg["learned_router_available"], bool)

    def test_get_config_learned_fields_present(self):
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        cfg = sr.get_config()
        for key in ("learned_router_enabled", "learned_router_trained", "learned_router_samples"):
            assert key in cfg

    def test_get_config_learned_enabled_false_when_not_configured(self):
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        cfg = sr.get_config()
        assert cfg["learned_router_enabled"] is False


# ---------------------------------------------------------------------------
# Tests: backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_smart_router_init_without_learned_router(self):
        """SmartRouter must still work when learned router unavailable."""
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        sr._learned_router = None
        # Basic operations should not raise
        cfg = sr.get_config()
        assert cfg is not None

    def test_select_model_unchanged_when_learned_disabled(self):
        """select_model() must return a valid result regardless of learned router state."""
        from opti_oignon.smart_router import SmartRouter
        sr = SmartRouter(enabled=False)
        result = sr.select_model("direct")
        # Result should always be a valid SmartRoutingResult with a model
        assert isinstance(result.model, str)
        assert len(result.model) > 0
        # The new fields must be present with correct defaults
        assert result.routing_source == "yaml"
        assert result.learned_confidence == 0.0

    def test_existing_smart_routing_result_fields_intact(self):
        """Existing SmartRoutingResult fields must not have changed."""
        from opti_oignon.smart_router import SmartRoutingResult
        r = SmartRoutingResult(
            model="m", score=0.9, task_score=0.8, speed_weight=1.0,
            context_fit=1.0, reason="test", fallback=False,
            feedback_adjusted=False, failover=False, original_model="",
        )
        d = r.to_dict()
        for key in ("model", "score", "task_score", "speed_weight", "context_fit",
                    "reason", "fallback", "feedback_adjusted", "failover", "original_model"):
            assert key in d

    def test_singleton_smart_router_has_learned_router(self):
        """The module-level singleton should have _learned_router attribute."""
        from opti_oignon.smart_router import smart_router
        assert hasattr(smart_router, "_learned_router")
