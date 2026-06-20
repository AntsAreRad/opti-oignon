#!/usr/bin/env python3
"""
Tests for S114 — Profiler Dashboard UI + Telemetry Event History + Observability.

Test groups:
1. TelemetryHistoryStore — SQLite persistence, queries, retention, purge
2. TelemetryHistoryStore as consumer — event filtering, data extraction
3. Trends and model breakdown aggregation
4. Routes: history, trends, purge endpoints (schema validation)
5. Frontend: ProfilerDashboard, ObservabilityPanel, API client AST/content checks
6. Integration: full pipeline from event emission to history retrieval
"""

import ast
import importlib.util
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module isolation helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _stub_opti():
    """Ensure opti_oignon parent module is stubbed."""
    if "opti_oignon" not in sys.modules or not hasattr(sys.modules["opti_oignon"], "__path__"):
        stub = types.ModuleType("opti_oignon")
        stub.__path__ = [str(ROOT / "opti_oignon")]
        sys.modules["opti_oignon"] = stub

    if "opti_oignon.config" not in sys.modules:
        cfg = types.ModuleType("opti_oignon.config")
        cfg.DATA_DIR = tempfile.mkdtemp(prefix="oo_test_s114_")
        sys.modules["opti_oignon.config"] = cfg


def _load_module(name: str, filename: str):
    _stub_opti()
    filepath = ROOT / "opti_oignon" / filename
    spec = importlib.util.spec_from_file_location(f"opti_oignon.{name}", str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"opti_oignon.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeEvent:
    """Minimal telemetry event for testing."""
    def __init__(self, event_type, request_id, model="", data=None, timestamp=None):
        self.event_type = event_type
        self.request_id = request_id
        self.model = model
        self.data = data or {}
        self.timestamp = timestamp or time.time()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database path."""
    return str(tmp_path / "test_history.db")


@pytest.fixture
def history_mod():
    return _load_module("telemetry_history", "telemetry_history.py")


@pytest.fixture
def store(history_mod, tmp_db):
    """Create a fresh TelemetryHistoryStore for each test."""
    return history_mod.TelemetryHistoryStore(db_path=tmp_db, retention_days=7)


@pytest.fixture
def populated_store(store):
    """Store with 10 events across 2 models."""
    t0 = time.time()
    events = []
    for i in range(10):
        model = "llama3:8b" if i < 6 else "mistral:7b"
        events.append(FakeEvent(
            "inference_end", f"req-{i:03d}", model,
            data={
                "latency_ms": 500 + i * 100,
                "tokens_in": 30 + i * 5,
                "tokens_out": 20 + i * 3,
                "prompt_eval_ms": 100 + i * 10,
                "token_gen_ms": 200 + i * 20,
            },
            timestamp=t0 - (10 - i) * 600,  # spaced 10min apart
        ))
    store.consume(events)
    return store


# =========================================================================
# Test Group 1 — TelemetryHistoryStore basics
# =========================================================================

class TestHistoryStoreBasics:
    """Core TelemetryHistoryStore functionality."""

    def test_init_creates_db(self, history_mod, tmp_db):
        store = history_mod.TelemetryHistoryStore(db_path=tmp_db)
        assert os.path.exists(tmp_db)

    def test_init_creates_tables(self, store, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='inference_events'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_empty_store_stats(self, store):
        stats = store.get_stats()
        assert stats["available"] is True
        assert stats["total_stored"] == 0
        assert stats["retention_days"] == 7

    def test_empty_store_history(self, store):
        result = store.get_history()
        assert result["total"] == 0
        assert result["events"] == []

    def test_empty_store_trends(self, store):
        trends = store.get_trends()
        assert trends == []

    def test_no_db_path_degrades(self, history_mod):
        store = history_mod.TelemetryHistoryStore(db_path="")
        stats = store.get_stats()
        assert stats["available"] is False
        assert store.get_history()["events"] == []
        assert store.get_trends() == []
        assert store.get_model_breakdown() == []
        assert store.purge() == 0
        assert store.purge_all() == 0


# =========================================================================
# Test Group 2 — Consumer event filtering
# =========================================================================

class TestConsumerFiltering:
    """TelemetryHistoryStore.consume() event filtering."""

    def test_only_inference_end_persisted(self, store):
        events = [
            FakeEvent("inference_start", "r1", "m1"),
            FakeEvent("token_generated", "r1"),
            FakeEvent("inference_end", "r1", "m1", {"latency_ms": 500, "tokens_in": 10, "tokens_out": 5}),
        ]
        store.consume(events)
        assert store.get_history()["total"] == 1

    def test_no_request_id_skipped(self, store):
        events = [
            FakeEvent("inference_end", "", "m1", {"latency_ms": 100}),
        ]
        store.consume(events)
        assert store.get_history()["total"] == 0

    def test_tok_per_sec_computed(self, store):
        events = [
            FakeEvent("inference_end", "r1", "m1", {
                "latency_ms": 1000,
                "tokens_out": 50,
            }),
        ]
        store.consume(events)
        ev = store.get_history()["events"][0]
        assert ev["tok_per_sec"] == 50.0  # 50 tokens / 1 second

    def test_zero_latency_zero_toks(self, store):
        events = [
            FakeEvent("inference_end", "r1", "m1", {
                "latency_ms": 0,
                "tokens_out": 50,
            }),
        ]
        store.consume(events)
        ev = store.get_history()["events"][0]
        assert ev["tok_per_sec"] == 0.0

    def test_multiple_events_batch(self, store):
        events = [
            FakeEvent("inference_end", f"r{i}", "m1", {"latency_ms": 100 * i})
            for i in range(1, 6)
        ]
        store.consume(events)
        assert store.get_history()["total"] == 5

    def test_consumer_has_name(self, store):
        assert hasattr(store.consume, "__name__")
        assert "telemetry_history" in store.consume.__name__


# =========================================================================
# Test Group 3 — Query operations
# =========================================================================

class TestQueries:
    """History, trends, and model breakdown queries."""

    def test_history_pagination(self, populated_store):
        page1 = populated_store.get_history(limit=3, offset=0)
        assert len(page1["events"]) == 3
        assert page1["total"] == 10

        page2 = populated_store.get_history(limit=3, offset=3)
        assert len(page2["events"]) == 3

        # IDs should not overlap
        ids1 = {e["request_id"] for e in page1["events"]}
        ids2 = {e["request_id"] for e in page2["events"]}
        assert ids1.isdisjoint(ids2)

    def test_history_model_filter(self, populated_store):
        llama = populated_store.get_history(model="llama3:8b")
        assert llama["total"] == 6

        mistral = populated_store.get_history(model="mistral:7b")
        assert mistral["total"] == 4

    def test_history_order_desc(self, populated_store):
        result = populated_store.get_history(limit=10)
        timestamps = [e["timestamp"] for e in result["events"]]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_trends_returns_buckets(self, populated_store):
        trends = populated_store.get_trends(hours=24)
        assert len(trends) >= 1
        for bucket in trends:
            assert "bucket_label" in bucket
            assert "event_count" in bucket
            assert bucket["event_count"] > 0
            assert "avg_latency_ms" in bucket
            assert "avg_tok_per_sec" in bucket

    def test_trends_model_filter(self, populated_store):
        all_trends = populated_store.get_trends(hours=24)
        llama_trends = populated_store.get_trends(hours=24, model="llama3:8b")

        all_count = sum(b["event_count"] for b in all_trends)
        llama_count = sum(b["event_count"] for b in llama_trends)
        assert llama_count <= all_count
        assert llama_count == 6

    def test_model_breakdown(self, populated_store):
        breakdown = populated_store.get_model_breakdown()
        assert len(breakdown) == 2

        models = {m["model"] for m in breakdown}
        assert "llama3:8b" in models
        assert "mistral:7b" in models

        for entry in breakdown:
            assert entry["event_count"] > 0
            assert entry["avg_latency_ms"] > 0

    def test_model_breakdown_order(self, populated_store):
        breakdown = populated_store.get_model_breakdown()
        # Should be ordered by count DESC
        counts = [m["event_count"] for m in breakdown]
        assert counts == sorted(counts, reverse=True)


# =========================================================================
# Test Group 4 — Purge operations
# =========================================================================

class TestPurge:
    """Retention and purge operations."""

    def test_purge_all(self, populated_store):
        assert populated_store.get_history()["total"] == 10
        deleted = populated_store.purge_all()
        assert deleted == 10
        assert populated_store.get_history()["total"] == 0

    def test_purge_by_age(self, store):
        t0 = time.time()
        events = [
            FakeEvent("inference_end", "old", "m1", {"latency_ms": 100},
                      timestamp=t0 - 86400 * 10),  # 10 days ago
            FakeEvent("inference_end", "new", "m1", {"latency_ms": 200},
                      timestamp=t0),  # now
        ]
        store.consume(events)
        assert store.get_history()["total"] == 2

        deleted = store.purge(older_than_days=5)
        assert deleted == 1

        remaining = store.get_history()
        assert remaining["total"] == 1
        assert remaining["events"][0]["request_id"] == "new"

    def test_purge_nothing(self, populated_store):
        deleted = populated_store.purge(older_than_days=365)
        assert deleted == 0
        assert populated_store.get_history()["total"] == 10


# =========================================================================
# Test Group 5 — Event data integrity
# =========================================================================

class TestDataIntegrity:
    """Verify stored events contain all expected fields."""

    def test_event_fields(self, store):
        store.consume([
            FakeEvent("inference_end", "r1", "llama3:8b", {
                "latency_ms": 850.5,
                "tokens_in": 42,
                "tokens_out": 28,
                "prompt_eval_ms": 150.3,
                "token_gen_ms": 600.2,
            })
        ])

        ev = store.get_history()["events"][0]
        assert ev["request_id"] == "r1"
        assert ev["model"] == "llama3:8b"
        assert ev["latency_ms"] == 850.5
        assert ev["tokens_in"] == 42
        assert ev["tokens_out"] == 28
        assert ev["prompt_eval_ms"] == 150.3
        assert ev["token_gen_ms"] == 600.2
        assert ev["tok_per_sec"] > 0
        assert ev["timestamp"] > 0
        assert ev["id"] > 0

    def test_stats_after_inserts(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["total_stored"] == 10
        assert stats["available"] is True
        assert stats["oldest_event_ts"] > 0


# =========================================================================
# Test Group 6 — AST integrity of all S114 files
# =========================================================================

class TestASTIntegrity:
    """Verify all modified/new files parse correctly."""

    @pytest.mark.parametrize("filepath", [
        "opti_oignon/telemetry_history.py",
        "opti_oignon/plugin_loader.py",
        "opti_oignon/plugin_hooks.py",
        "opti_oignon/api/routes_telemetry.py",
        "opti_oignon/api/routes_plugins.py",
        "opti_oignon/api/routes_chat.py",
        "opti_oignon/api/schemas.py",
        "opti_oignon/api/deps.py",
    ])
    def test_python_ast(self, filepath):
        full = ROOT / filepath
        tree = ast.parse(full.read_text())
        assert tree is not None


# =========================================================================
# Test Group 7 — Frontend file checks
# =========================================================================

class TestFrontendFiles:
    """Verify frontend component and API client files."""

    def test_profiler_dashboard_exists(self):
        p = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "ProfilerDashboard.svelte"
        assert p.exists()
        content = p.read_text()
        assert "getProfilerSummary" in content
        assert "getRecentProfiles" in content
        assert "breakdown-bar" in content
        assert "profiler-table" in content
        assert "auto-refresh" in content.lower() or "autoRefresh" in content

    def test_profiler_dashboard_no_hex(self):
        p = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "ProfilerDashboard.svelte"
        import re
        content = p.read_text()
        # Check CSS section only
        style_start = content.find("<style>")
        if style_start >= 0:
            css = content[style_start:]
            hex_matches = re.findall(r':\s*#[0-9a-fA-F]{3,8}\b', css)
            assert len(hex_matches) == 0, f"Found hardcoded hex: {hex_matches}"

    def test_observability_panel_exists(self):
        p = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "ObservabilityPanel.svelte"
        assert p.exists()
        content = p.read_text()
        assert "TelemetryDashboard" in content
        assert "ProfilerDashboard" in content
        assert "PerformanceDashboard" in content
        assert "sub-tab" in content or "subTab" in content
        assert "status-card" in content

    def test_observability_panel_no_hex(self):
        p = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "ObservabilityPanel.svelte"
        import re
        content = p.read_text()
        style_start = content.find("<style>")
        if style_start >= 0:
            css = content[style_start:]
            hex_matches = re.findall(r':\s*#[0-9a-fA-F]{3,8}\b', css)
            assert len(hex_matches) == 0, f"Found hardcoded hex: {hex_matches}"

    def test_observability_cross_linking(self):
        p = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "ObservabilityPanel.svelte"
        content = p.read_text()
        assert "selectModel" in content
        assert "linkedModel" in content
        assert "clearLinkedModel" in content or "Clear filter" in content

    def test_telemetry_api_client_history(self):
        p = ROOT / "frontend" / "src" / "lib" / "api" / "telemetry.ts"
        content = p.read_text()
        assert "getTelemetryHistory" in content
        assert "getTelemetryTrends" in content
        assert "getHistoryModelBreakdown" in content
        assert "getHistoryStats" in content
        assert "purgeHistory" in content
        assert "HistoryEvent" in content
        assert "TrendBucket" in content

    def test_profiler_api_client(self):
        p = ROOT / "frontend" / "src" / "lib" / "api" / "profiler.ts"
        content = p.read_text()
        assert "getProfilerSummary" in content
        assert "getRecentProfiles" in content
        assert "InferenceProfile" in content

    def test_settings_page_uses_observability(self):
        p = ROOT / "frontend" / "src" / "routes" / "settings" / "+page.svelte"
        content = p.read_text()
        assert "ObservabilityPanel" in content
        assert "Observe" in content


# =========================================================================
# Test Group 8 — Schemas validation
# =========================================================================

class TestSchemas:
    """Verify new Pydantic schemas parse correctly."""

    def test_history_schemas_importable(self):
        _stub_opti()
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.api.schemas",
            str(ROOT / "opti_oignon" / "api" / "schemas.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Need fastapi/pydantic
        spec.loader.exec_module(mod)

        assert hasattr(mod, "HistoryEventSchema")
        assert hasattr(mod, "TelemetryHistoryResponse")
        assert hasattr(mod, "TrendBucketSchema")
        assert hasattr(mod, "TelemetryTrendsResponse")
        assert hasattr(mod, "ModelBreakdownSchema")
        assert hasattr(mod, "TelemetryHistoryPurgeResponse")
        assert hasattr(mod, "TelemetryHistoryStatsResponse")

    def test_history_event_schema_fields(self):
        _stub_opti()
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.api.schemas",
            str(ROOT / "opti_oignon" / "api" / "schemas.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        ev = mod.HistoryEventSchema(
            id=1, request_id="r1", model="m1",
            timestamp=time.time(), latency_ms=500,
            tokens_in=10, tokens_out=5, tok_per_sec=10.0,
        )
        d = ev.model_dump()
        assert d["request_id"] == "r1"
        assert d["latency_ms"] == 500

    def test_trend_bucket_schema_fields(self):
        _stub_opti()
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.api.schemas",
            str(ROOT / "opti_oignon" / "api" / "schemas.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        bucket = mod.TrendBucketSchema(
            bucket_start=time.time(),
            bucket_label="2025-01-01 12:00",
            event_count=5,
            avg_latency_ms=400.5,
            avg_tok_per_sec=25.3,
        )
        d = bucket.model_dump()
        assert d["event_count"] == 5


# =========================================================================
# Test Group 9 — Routes telemetry endpoint presence
# =========================================================================

class TestRouteEndpoints:
    """Verify new endpoints exist in routes_telemetry.py."""

    def test_history_endpoint_exists(self):
        source = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert '"/history"' in source
        assert "get_event_history" in source

    def test_trends_endpoint_exists(self):
        source = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert '"/trends"' in source
        assert "get_telemetry_trends" in source

    def test_history_models_endpoint_exists(self):
        source = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert '"/history/models"' in source
        assert "get_history_model_breakdown" in source

    def test_history_stats_endpoint_exists(self):
        source = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert '"/history/stats"' in source
        assert "get_history_stats" in source

    def test_delete_history_endpoint_exists(self):
        source = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert "purge_event_history" in source
        assert "older_than_days" in source

    def test_deps_has_history_store(self):
        source = (ROOT / "opti_oignon" / "api" / "deps.py").read_text()
        assert "TELEMETRY_HISTORY_AVAILABLE" in source
        assert "get_history_store" in source
