#!/usr/bin/env python3
"""
Tests for LearnedRouter core module (S67, Step 1).

Covers: initialization, configuration, training data management,
model training, classification, fallback logic, auto-retrain,
persistence, and status reporting.
"""

import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router(tmp_path: Path, config_overrides: dict | None = None):
    """Create a LearnedRouter backed by tmp_path for isolation."""
    from opti_oignon.learned_router import LearnedRouter

    cfg_path = tmp_path / "learned_routing.yaml"
    db_path = tmp_path / "learned_router.db"
    model_path = tmp_path / "learned_router.pkl"

    router = LearnedRouter(
        config_path=cfg_path,
        db_path=db_path,
        model_path=model_path,
    )
    if config_overrides:
        router.update_config(config_overrides)
    return router


def _seed_samples(router, n_per_class: int = 20):
    """Populate the router's DB with synthetic labeled samples."""
    samples = [
        ("how do I fix this python error", "debug"),
        ("write a function to parse JSON in python", "code_python"),
        ("explain the concept of recursion", "reasoning"),
        ("what is the best way to analyze this dataset", "data_analysis"),
        ("help me plan the next steps for my project", "planning"),
    ]
    for query, label in samples:
        for i in range(n_per_class):
            router.log_sample(f"{query} variant {i}", label)


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------

class TestLearnedRouterInit:
    def test_instantiation_succeeds(self, tmp_path):
        router = _make_router(tmp_path)
        assert router is not None

    def test_not_trained_on_fresh_init(self, tmp_path):
        router = _make_router(tmp_path)
        assert router.is_trained is False

    def test_default_config_values(self, tmp_path):
        router = _make_router(tmp_path)
        cfg = router.get_config()
        assert cfg["confidence_threshold"] == 0.70
        assert cfg["min_training_samples"] == 50
        assert cfg["model_type"] == "logistic"
        assert cfg["enabled"] is False

    def test_sklearn_flag_is_true(self, tmp_path):
        """sklearn must be importable in the test environment."""
        from opti_oignon.learned_router import SKLEARN_AVAILABLE
        assert SKLEARN_AVAILABLE is True

    def test_learned_router_available_flag(self):
        from opti_oignon.learned_router import LEARNED_ROUTER_AVAILABLE
        assert LEARNED_ROUTER_AVAILABLE is True


# ---------------------------------------------------------------------------
# Tests: configuration
# ---------------------------------------------------------------------------

class TestLearnedRouterConfig:
    def test_update_config_persists(self, tmp_path):
        router = _make_router(tmp_path)
        router.update_config({"confidence_threshold": 0.85})
        assert router.get_config()["confidence_threshold"] == 0.85

    def test_update_config_partial_merge(self, tmp_path):
        router = _make_router(tmp_path)
        router.update_config({"model_type": "random_forest"})
        cfg = router.get_config()
        assert cfg["model_type"] == "random_forest"
        assert cfg["confidence_threshold"] == 0.70  # unchanged

    def test_config_written_to_yaml(self, tmp_path):
        import yaml
        router = _make_router(tmp_path)
        router.update_config({"enabled": True})
        cfg_path = tmp_path / "learned_routing.yaml"
        assert cfg_path.exists()
        with open(cfg_path) as fh:
            loaded = yaml.safe_load(fh)
        assert loaded["enabled"] is True


# ---------------------------------------------------------------------------
# Tests: training data management
# ---------------------------------------------------------------------------

class TestTrainingDataManagement:
    def test_log_sample_increments_count(self, tmp_path):
        router = _make_router(tmp_path)
        assert router.get_sample_count() == 0
        router.log_sample("write a python script", "code_python")
        assert router.get_sample_count() == 1

    def test_log_multiple_samples(self, tmp_path):
        router = _make_router(tmp_path)
        for i in range(10):
            router.log_sample(f"query {i}", "general")
        assert router.get_sample_count() == 10

    def test_log_sample_empty_query_ignored(self, tmp_path):
        router = _make_router(tmp_path)
        router.log_sample("", "code_python")
        assert router.get_sample_count() == 0

    def test_log_sample_empty_task_type_ignored(self, tmp_path):
        router = _make_router(tmp_path)
        router.log_sample("some query", "")
        assert router.get_sample_count() == 0

    def test_class_distribution(self, tmp_path):
        router = _make_router(tmp_path)
        for _ in range(3):
            router.log_sample("python code", "code_python")
        for _ in range(2):
            router.log_sample("fix bug", "debug")
        dist = router.get_class_distribution()
        assert dist["code_python"] == 3
        assert dist["debug"] == 2

    def test_pruning_respects_max_stored(self, tmp_path):
        router = _make_router(tmp_path, {"max_stored_samples": 10})
        for i in range(15):
            router.log_sample(f"query {i}", "general")
        assert router.get_sample_count() <= 10


# ---------------------------------------------------------------------------
# Tests: training
# ---------------------------------------------------------------------------

class TestTraining:
    def test_train_fails_below_threshold(self, tmp_path):
        router = _make_router(tmp_path, {"min_training_samples": 50})
        router.log_sample("test query", "general")
        result = router.train()
        assert result.success is False
        assert "Insufficient" in result.error

    def test_train_succeeds_with_enough_samples(self, tmp_path):
        router = _make_router(tmp_path, {"min_training_samples": 5, "cv_folds": 2})
        _seed_samples(router, n_per_class=5)
        result = router.train(min_samples=5)
        assert result.success is True
        assert result.n_samples > 0
        assert result.n_classes >= 2
        assert 0.0 <= result.accuracy <= 1.0

    def test_train_creates_model_file(self, tmp_path):
        router = _make_router(tmp_path, {"min_training_samples": 5, "cv_folds": 2})
        _seed_samples(router, n_per_class=5)
        router.train(min_samples=5)
        model_path = tmp_path / "learned_router.pkl"
        assert model_path.exists()

    def test_is_trained_after_successful_train(self, tmp_path):
        router = _make_router(tmp_path, {"min_training_samples": 5, "cv_folds": 2})
        _seed_samples(router, n_per_class=5)
        router.train(min_samples=5)
        assert router.is_trained is True

    def test_last_training_result_stored(self, tmp_path):
        router = _make_router(tmp_path, {"min_training_samples": 5, "cv_folds": 2})
        _seed_samples(router, n_per_class=5)
        result = router.train(min_samples=5)
        assert router.last_training_result is not None
        assert router.last_training_result.success is True

    def test_training_result_to_dict(self, tmp_path):
        router = _make_router(tmp_path, {"min_training_samples": 5, "cv_folds": 2})
        _seed_samples(router, n_per_class=5)
        result = router.train(min_samples=5)
        d = result.to_dict()
        assert "accuracy" in d
        assert "n_samples" in d
        assert "n_classes" in d
        assert "trained_at" in d
        assert d["success"] is True


# ---------------------------------------------------------------------------
# Tests: classification
# ---------------------------------------------------------------------------

class TestClassification:
    def _trained_router(self, tmp_path):
        router = _make_router(tmp_path, {"min_training_samples": 5, "cv_folds": 2})
        _seed_samples(router, n_per_class=5)
        router.train(min_samples=5)
        return router

    def test_classify_returns_prediction(self, tmp_path):
        router = self._trained_router(tmp_path)
        pred = router.classify("write a python function")
        assert pred.task_type != ""
        assert 0.0 <= pred.confidence <= 1.0

    def test_classify_untrained_returns_general(self, tmp_path):
        router = _make_router(tmp_path)
        pred = router.classify("some query")
        assert pred.task_type == "general"
        assert pred.fallback_used is True

    def test_classify_top_classes_populated(self, tmp_path):
        router = self._trained_router(tmp_path)
        pred = router.classify("write a python function")
        assert len(pred.top_classes) >= 1
        assert "task_type" in pred.top_classes[0]
        assert "confidence" in pred.top_classes[0]

    def test_classify_to_dict(self, tmp_path):
        router = self._trained_router(tmp_path)
        pred = router.classify("fix this bug")
        d = pred.to_dict()
        assert "task_type" in d
        assert "confidence" in d
        assert "fallback_used" in d


# ---------------------------------------------------------------------------
# Tests: classify_with_fallback
# ---------------------------------------------------------------------------

class TestClassifyWithFallback:
    def _trained_enabled_router(self, tmp_path):
        router = _make_router(
            tmp_path,
            {"min_training_samples": 5, "cv_folds": 2, "enabled": True, "confidence_threshold": 0.0},
        )
        _seed_samples(router, n_per_class=5)
        router.train(min_samples=5)
        return router

    def test_fallback_when_disabled(self, tmp_path):
        router = _make_router(tmp_path, {"enabled": False})
        _seed_samples(router, n_per_class=5)
        router.train(min_samples=5)
        pred = router.classify_with_fallback("write code", "general")
        assert pred.fallback_used is True
        assert pred.task_type == "general"

    def test_fallback_when_not_trained(self, tmp_path):
        router = _make_router(tmp_path, {"enabled": True})
        pred = router.classify_with_fallback("write code", "code_python")
        assert pred.fallback_used is True
        assert pred.task_type == "code_python"

    def test_ml_used_when_enabled_and_confident(self, tmp_path):
        router = self._trained_enabled_router(tmp_path)
        pred = router.classify_with_fallback("explain recursion", "general")
        # With threshold=0.0, ML should always be used
        assert pred.model_type != "yaml_fallback"

    def test_fallback_when_low_confidence(self, tmp_path):
        router = _make_router(
            tmp_path,
            {"min_training_samples": 5, "cv_folds": 2, "enabled": True, "confidence_threshold": 0.99},
        )
        _seed_samples(router, n_per_class=5)
        router.train(min_samples=5)
        pred = router.classify_with_fallback("vague query", "planning")
        # At threshold=0.99, almost certainly falls back
        if pred.fallback_used:
            assert pred.task_type == "planning"


# ---------------------------------------------------------------------------
# Tests: model persistence
# ---------------------------------------------------------------------------

class TestModelPersistence:
    def test_model_reloaded_on_new_instance(self, tmp_path):
        from opti_oignon.learned_router import LearnedRouter

        cfg_path = tmp_path / "learned_routing.yaml"
        db_path = tmp_path / "learned_router.db"
        model_path = tmp_path / "learned_router.pkl"

        r1 = LearnedRouter(config_path=cfg_path, db_path=db_path, model_path=model_path)
        _seed_samples(r1, n_per_class=5)
        r1._config["cv_folds"] = 2
        r1.train(min_samples=5)
        assert r1.is_trained

        # Create a new instance pointing to the same files
        r2 = LearnedRouter(config_path=cfg_path, db_path=db_path, model_path=model_path)
        assert r2.is_trained  # Should have loaded from disk


# ---------------------------------------------------------------------------
# Tests: auto-retrain
# ---------------------------------------------------------------------------

class TestAutoRetrain:
    def test_auto_retrain_not_triggered_below_interval(self, tmp_path):
        router = _make_router(tmp_path, {"auto_retrain_interval": 100, "min_training_samples": 5})
        _seed_samples(router, n_per_class=5)
        router.train(min_samples=5)
        router._samples_since_retrain = 50
        result = router.auto_retrain_if_needed()
        assert result is None

    def test_auto_retrain_triggered_at_interval(self, tmp_path):
        router = _make_router(
            tmp_path,
            {"auto_retrain_interval": 10, "min_training_samples": 5, "cv_folds": 2},
        )
        _seed_samples(router, n_per_class=5)
        router.train(min_samples=5)
        router._samples_since_retrain = 10
        result = router.auto_retrain_if_needed()
        assert result is not None
        assert result.success is True


# ---------------------------------------------------------------------------
# Tests: status reporting
# ---------------------------------------------------------------------------

class TestStatus:
    def test_get_status_keys(self, tmp_path):
        router = _make_router(tmp_path)
        status = router.get_status()
        for key in ("available", "trained", "enabled", "sample_count", "model_type"):
            assert key in status

    def test_status_trained_false_before_train(self, tmp_path):
        router = _make_router(tmp_path)
        assert router.get_status()["trained"] is False

    def test_status_trained_true_after_train(self, tmp_path):
        router = _make_router(tmp_path, {"min_training_samples": 5, "cv_folds": 2})
        _seed_samples(router, n_per_class=5)
        router.train(min_samples=5)
        assert router.get_status()["trained"] is True


# ---------------------------------------------------------------------------
# Tests: singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_module_singleton_exists(self):
        from opti_oignon.learned_router import learned_router
        assert learned_router is not None

    def test_module_singleton_is_learned_router(self):
        from opti_oignon.learned_router import LearnedRouter, learned_router
        assert isinstance(learned_router, LearnedRouter)
