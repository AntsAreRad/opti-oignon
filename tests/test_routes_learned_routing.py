#!/usr/bin/env python3
"""
Tests for routes_learned_routing.py API endpoints (S67, Step 4).

Uses TestClient with a fresh LearnedRouter injected via monkeypatching
to avoid any dependency on real Ollama or trained models.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixture: isolated router + patched app
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_learned_router(tmp_path):
    """Return a LearnedRouter backed by tmp_path."""
    from opti_oignon.learned_router import LearnedRouter
    return LearnedRouter(
        config_path=tmp_path / "lr.yaml",
        db_path=tmp_path / "lr.db",
        model_path=tmp_path / "lr.pkl",
    )


@pytest.fixture
def trained_router(tmp_path):
    """Return a trained LearnedRouter with minimal samples."""
    from opti_oignon.learned_router import LearnedRouter
    router = LearnedRouter(
        config_path=tmp_path / "lr.yaml",
        db_path=tmp_path / "lr.db",
        model_path=tmp_path / "lr.pkl",
    )
    router.update_config({"min_training_samples": 5, "cv_folds": 2})
    labels = ["code_python", "debug", "data_analysis", "planning", "general"]
    for label in labels:
        for i in range(5):
            router.log_sample(f"{label} query variant {i}", label)
    router.train(min_samples=5)
    return router


@pytest.fixture
def client(tmp_learned_router):
    """TestClient with the learned router singleton patched."""
    with patch(
        "opti_oignon.api.routes_learned_routing._get_learned_router",
        return_value=tmp_learned_router,
    ):
        from opti_oignon.api.app import app
        yield TestClient(app)


@pytest.fixture
def trained_client(trained_router):
    """TestClient with a trained learned router patched."""
    with patch(
        "opti_oignon.api.routes_learned_routing._get_learned_router",
        return_value=trained_router,
    ):
        from opti_oignon.api.app import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Tests: GET /api/routing/learned/status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_200(self, client):
        resp = client.get("/api/routing/learned/status")
        assert resp.status_code == 200

    def test_status_keys(self, client):
        data = client.get("/api/routing/learned/status").json()
        for key in ("available", "trained", "enabled", "sample_count", "model_type"):
            assert key in data

    def test_status_trained_false_initially(self, client):
        data = client.get("/api/routing/learned/status").json()
        assert data["trained"] is False

    def test_status_trained_true_after_training(self, trained_client):
        data = trained_client.get("/api/routing/learned/status").json()
        assert data["trained"] is True


# ---------------------------------------------------------------------------
# Tests: POST /api/routing/learned/train
# ---------------------------------------------------------------------------

class TestPostTrain:
    def test_train_fails_without_samples(self, client):
        resp = client.post("/api/routing/learned/train")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "Insufficient" in data["error"]

    def test_train_succeeds_with_samples(self, trained_client):
        resp = trained_client.post("/api/routing/learned/train")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["n_samples"] > 0

    def test_train_returns_accuracy(self, trained_client):
        data = trained_client.post("/api/routing/learned/train").json()
        assert "accuracy" in data
        assert 0.0 <= data["accuracy"] <= 1.0

    def test_train_response_fields(self, trained_client):
        data = trained_client.post("/api/routing/learned/train").json()
        for key in ("success", "accuracy", "n_samples", "n_classes", "trained_at", "model_type"):
            assert key in data


# ---------------------------------------------------------------------------
# Tests: GET /api/routing/learned/config
# ---------------------------------------------------------------------------

class TestGetConfig:
    def test_config_200(self, client):
        resp = client.get("/api/routing/learned/config")
        assert resp.status_code == 200

    def test_config_has_all_fields(self, client):
        data = client.get("/api/routing/learned/config").json()
        for key in (
            "enabled", "model_type", "confidence_threshold",
            "min_training_samples", "auto_retrain_interval",
            "feature_max_features", "feature_ngram_range",
        ):
            assert key in data

    def test_config_defaults(self, client):
        data = client.get("/api/routing/learned/config").json()
        assert data["enabled"] is False
        assert data["model_type"] == "logistic"
        assert data["confidence_threshold"] == 0.70


# ---------------------------------------------------------------------------
# Tests: PUT /api/routing/learned/config
# ---------------------------------------------------------------------------

class TestPutConfig:
    def test_update_confidence_threshold(self, client):
        resp = client.put(
            "/api/routing/learned/config",
            json={"confidence_threshold": 0.85},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_update_model_type(self, client):
        resp = client.put(
            "/api/routing/learned/config",
            json={"model_type": "random_forest"},
        )
        assert resp.status_code == 200

    def test_invalid_model_type_rejected(self, client):
        resp = client.put(
            "/api/routing/learned/config",
            json={"model_type": "svm"},
        )
        assert resp.status_code == 400

    def test_enable_untrained_router_rejected(self, client):
        resp = client.put(
            "/api/routing/learned/config",
            json={"enabled": True},
        )
        assert resp.status_code == 400
        assert "train" in resp.json()["detail"].lower()

    def test_enable_trained_router_accepted(self, trained_client):
        resp = trained_client.put(
            "/api/routing/learned/config",
            json={"enabled": True},
        )
        assert resp.status_code == 200

    def test_empty_body_rejected(self, client):
        resp = client.put("/api/routing/learned/config", json={})
        assert resp.status_code == 400

    def test_updated_keys_returned(self, client):
        resp = client.put(
            "/api/routing/learned/config",
            json={"confidence_threshold": 0.6, "cv_folds": 3},
        )
        data = resp.json()
        assert "confidence_threshold" in data["updated"]
        assert "cv_folds" in data["updated"]


# ---------------------------------------------------------------------------
# Tests: POST /api/routing/learned/classify
# ---------------------------------------------------------------------------

class TestPostClassify:
    def test_classify_200(self, client):
        resp = client.post(
            "/api/routing/learned/classify",
            json={"query": "write a python function", "yaml_task_type": "code_python"},
        )
        assert resp.status_code == 200

    def test_classify_response_fields(self, client):
        data = client.post(
            "/api/routing/learned/classify",
            json={"query": "explain recursion"},
        ).json()
        for key in ("ml_prediction", "yaml_task_type", "final_task_type",
                    "routing_source", "confidence"):
            assert key in data

    def test_classify_fallback_when_untrained(self, client):
        data = client.post(
            "/api/routing/learned/classify",
            json={"query": "fix this bug", "yaml_task_type": "debug"},
        ).json()
        # Untrained router must fall back to YAML
        assert data["routing_source"] == "yaml"
        assert data["final_task_type"] == "debug"

    def test_classify_uses_default_yaml_task_type(self, client):
        data = client.post(
            "/api/routing/learned/classify",
            json={"query": "some query"},
        ).json()
        assert data["yaml_task_type"] == "general"

    def test_classify_empty_query_rejected(self, client):
        resp = client.post(
            "/api/routing/learned/classify",
            json={"query": ""},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: GET /api/routing/learned/metrics
# ---------------------------------------------------------------------------

class TestGetMetrics:
    def test_metrics_200(self, client):
        resp = client.get("/api/routing/learned/metrics")
        assert resp.status_code == 200

    def test_metrics_fields(self, client):
        data = client.get("/api/routing/learned/metrics").json()
        for key in (
            "total_decisions", "learned_count", "yaml_count",
            "learned_ratio", "avg_ml_confidence", "class_agreement_rate",
            "top_disagreements", "confidence_histogram",
        ):
            assert key in data

    def test_metrics_empty_initially(self, client):
        data = client.get("/api/routing/learned/metrics").json()
        assert data["total_decisions"] == 0

    def test_metrics_window_hours_param(self, client):
        resp = client.get("/api/routing/learned/metrics?window_hours=48")
        assert resp.status_code == 200
        assert resp.json()["window_hours"] == 48.0

    def test_metrics_invalid_window_rejected(self, client):
        resp = client.get("/api/routing/learned/metrics?window_hours=0")
        assert resp.status_code == 400

    def test_metrics_confidence_histogram_list(self, client):
        data = client.get("/api/routing/learned/metrics").json()
        assert isinstance(data["confidence_histogram"], list)
