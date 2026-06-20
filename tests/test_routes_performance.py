#!/usr/bin/env python3
"""
Tests for Performance API routes and executor integration -- S72 Step 2.

Covers:
- GET /api/performance/summary
- GET /api/performance/latency
- GET /api/performance/drift
- GET /api/performance/recommendations
- GET /api/performance/history
- GET /api/performance/throughput
- GET /api/performance/utilization
- POST /api/performance/cleanup
- Executor integration (PERFORMANCE_MONITOR_AVAILABLE flag)
- Unavailable fallback responses
"""

import importlib.util
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Provide mock ollama before any project imports
# ---------------------------------------------------------------------------
sys.modules.setdefault("ollama", MagicMock())

# ---------------------------------------------------------------------------
# Direct module import for PerformanceMonitor (bypass __init__.py)
# ---------------------------------------------------------------------------
_mod_path = Path(__file__).resolve().parent.parent / "opti_oignon" / "performance_monitor.py"
_spec = importlib.util.spec_from_file_location("performance_monitor_mod", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PerformanceMonitor = _mod.PerformanceMonitor

# ---------------------------------------------------------------------------
# FastAPI test client setup
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None


def _make_monitor(tmp_path: Path) -> PerformanceMonitor:
    """Create a test PerformanceMonitor with temp DB."""
    import yaml
    cfg = {
        "enabled": True,
        "retention_days": 7,
        "default_window_seconds": 300,
        "drift": {
            "window_seconds": 100,
            "baseline_window_seconds": 1000,
            "threshold": 0.3,
        },
        "recommendation_rules": [
            {
                "metric": "latency_p95",
                "condition": "gt",
                "threshold": 1000,
                "message": "Model {model} p95 too high",
            }
        ],
    }
    cfg_path = tmp_path / "performance.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    db_path = tmp_path / "test_perf.db"
    return PerformanceMonitor(db_path=db_path, config_path=cfg_path)


def _seed_monitor(mon: PerformanceMonitor, count: int = 10):
    """Seed monitor with test data."""
    now = time.time()
    for i in range(count):
        mon.record_execution(
            model="test-model",
            task_type="code_python",
            latency_ms=500 + i * 100,
            tokens_in=100,
            tokens_out=300,
            quality_score=0.85,
            timestamp=now - i,
        )


# ---------------------------------------------------------------------------
# Test: Routes with mocked monitor
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestPerformanceRoutes:
    """Tests for /api/performance/* endpoints."""

    def _get_client(self, monitor):
        """Create a test client with mocked deps."""
        from fastapi import FastAPI
        # Import routes module
        routes_path = Path(__file__).resolve().parent.parent / "opti_oignon" / "api" / "routes_performance.py"
        spec = importlib.util.spec_from_file_location("routes_perf_mod", routes_path)
        routes_mod = importlib.util.module_from_spec(spec)

        # Patch deps before loading
        mock_deps = MagicMock()
        mock_deps.PERFORMANCE_MONITOR_AVAILABLE = True
        mock_deps.performance_monitor = monitor
        sys.modules["opti_oignon.api.deps"] = mock_deps
        # Also set as relative import target
        routes_mod.__package__ = "opti_oignon.api"

        spec.loader.exec_module(routes_mod)

        app = FastAPI()
        app.include_router(routes_mod.router)
        return TestClient(app)

    def test_summary_endpoint(self, tmp_path):
        mon = _make_monitor(tmp_path)
        _seed_monitor(mon)
        client = self._get_client(mon)
        resp = client.get("/api/performance/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert "throughput" in data
        assert "latency" in data
        assert "utilization" in data

    def test_latency_endpoint(self, tmp_path):
        mon = _make_monitor(tmp_path)
        _seed_monitor(mon)
        client = self._get_client(mon)
        resp = client.get("/api/performance/latency?model=test-model&window=300")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["model"] == "test-model"
        assert data["count"] == 10

    def test_latency_all_models(self, tmp_path):
        mon = _make_monitor(tmp_path)
        _seed_monitor(mon)
        client = self._get_client(mon)
        resp = client.get("/api/performance/latency")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["count"] == 10

    def test_drift_endpoint(self, tmp_path):
        mon = _make_monitor(tmp_path)
        client = self._get_client(mon)
        resp = client.get("/api/performance/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert "drifts" in data

    def test_recommendations_endpoint(self, tmp_path):
        mon = _make_monitor(tmp_path)
        _seed_monitor(mon)
        client = self._get_client(mon)
        resp = client.get("/api/performance/recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert "recommendations" in data

    def test_history_endpoint(self, tmp_path):
        mon = _make_monitor(tmp_path)
        _seed_monitor(mon, 5)
        client = self._get_client(mon)
        resp = client.get("/api/performance/history?hours=1&limit=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["count"] == 5

    def test_history_with_model_filter(self, tmp_path):
        mon = _make_monitor(tmp_path)
        _seed_monitor(mon, 5)
        client = self._get_client(mon)
        resp = client.get("/api/performance/history?model=test-model&hours=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 5

    def test_throughput_endpoint(self, tmp_path):
        mon = _make_monitor(tmp_path)
        _seed_monitor(mon)
        client = self._get_client(mon)
        resp = client.get("/api/performance/throughput?window=300")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["request_count"] == 10

    def test_utilization_endpoint(self, tmp_path):
        mon = _make_monitor(tmp_path)
        _seed_monitor(mon)
        client = self._get_client(mon)
        resp = client.get("/api/performance/utilization")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert "test-model" in data["models"]

    def test_cleanup_endpoint(self, tmp_path):
        mon = _make_monitor(tmp_path)
        _seed_monitor(mon)
        client = self._get_client(mon)
        resp = client.post("/api/performance/cleanup")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert "deleted" in data


# ---------------------------------------------------------------------------
# Test: Unavailable fallback
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestPerformanceRoutesUnavailable:
    """Test endpoints when monitor is unavailable."""

    def _get_client_unavailable(self):
        """Create client with unavailable monitor."""
        from fastapi import FastAPI
        routes_path = Path(__file__).resolve().parent.parent / "opti_oignon" / "api" / "routes_performance.py"
        spec = importlib.util.spec_from_file_location("routes_perf_unavail", routes_path)
        routes_mod = importlib.util.module_from_spec(spec)

        mock_deps = MagicMock()
        mock_deps.PERFORMANCE_MONITOR_AVAILABLE = False
        mock_deps.performance_monitor = None
        sys.modules["opti_oignon.api.deps"] = mock_deps
        routes_mod.__package__ = "opti_oignon.api"

        spec.loader.exec_module(routes_mod)

        app = FastAPI()
        app.include_router(routes_mod.router)
        return TestClient(app)

    def test_summary_unavailable(self):
        client = self._get_client_unavailable()
        resp = client.get("/api/performance/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False

    def test_drift_unavailable(self):
        client = self._get_client_unavailable()
        resp = client.get("/api/performance/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert data["drifts"] == []


# ---------------------------------------------------------------------------
# Test: Executor integration flag
# ---------------------------------------------------------------------------

class TestExecutorIntegration:
    """Tests for executor performance_monitor integration."""

    def test_performance_monitor_import_in_executor(self):
        """Verify executor has the PERFORMANCE_MONITOR_AVAILABLE flag."""
        executor_path = Path(__file__).resolve().parent.parent / "opti_oignon" / "executor.py"
        content = executor_path.read_text(encoding="utf-8")
        assert "PERFORMANCE_MONITOR_AVAILABLE" in content
        assert "_performance_monitor" in content
        assert "record_execution" in content

    def test_executor_records_after_completion(self):
        """Verify the recording block exists in executor flow."""
        executor_path = Path(__file__).resolve().parent.parent / "opti_oignon" / "executor.py"
        content = executor_path.read_text(encoding="utf-8")
        # Check the S72 integration block is present
        assert "S72: Record performance metrics" in content
        assert "_performance_monitor.record_execution(" in content
        assert "latency_ms=elapsed * 1000" in content
