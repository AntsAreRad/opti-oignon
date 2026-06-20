#!/usr/bin/env python3
"""
Tests for S115 — Telemetry History UI + Event Timeline + Retention Settings.

Test groups:
1. TelemetryHistoryStore.update_settings — retention, auto-purge toggle
2. TelemetryHistoryStore.export_csv — full export, filtered, empty
3. Auto-purge timer lifecycle
4. Routes: PUT settings, GET export (schema validation)
5. Frontend: TelemetryHistoryPanel, EventTimeline, API client AST/content checks
6. ObservabilityPanel History tab integration
7. Schema validation for new request/response models
"""

import ast
import importlib.util
import os
import sys
import tempfile
import threading
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
        cfg.DATA_DIR = tempfile.mkdtemp(prefix="oo_test_s115_")
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
    return str(tmp_path / "test_history.db")


@pytest.fixture
def history_mod():
    return _load_module("telemetry_history", "telemetry_history.py")


@pytest.fixture
def store(history_mod, tmp_db):
    s = history_mod.TelemetryHistoryStore(db_path=tmp_db, retention_days=7)
    yield s
    s.shutdown()


@pytest.fixture
def populated_store(store):
    """Store with 10 events across 2 models."""
    events = []
    base_ts = time.time() - 3600  # 1 hour ago
    for i in range(10):
        model = "llama3" if i % 2 == 0 else "mistral"
        latency = 200 + i * 100
        events.append(FakeEvent(
            "inference_end",
            f"req_{i:03d}",
            model=model,
            data={"latency_ms": latency, "tokens_in": 50 + i * 10, "tokens_out": 100 + i * 20},
            timestamp=base_ts + i * 60,
        ))
    store.consume(events)
    return store


# ===========================================================================
# Group 1: update_settings
# ===========================================================================

class TestUpdateSettings:
    """Tests for TelemetryHistoryStore.update_settings (S115)."""

    def test_update_retention_days(self, store):
        result = store.update_settings(retention_days=14)
        assert result["retention_days"] == 14

    def test_update_retention_clamped_min(self, store):
        result = store.update_settings(retention_days=0)
        assert result["retention_days"] == 1

    def test_update_retention_clamped_max(self, store):
        result = store.update_settings(retention_days=999)
        assert result["retention_days"] == 365

    def test_update_auto_purge_enable(self, store):
        result = store.update_settings(auto_purge_enabled=True)
        assert result["auto_purge_enabled"] is True

    def test_update_auto_purge_disable(self, store):
        store.update_settings(auto_purge_enabled=True)
        result = store.update_settings(auto_purge_enabled=False)
        assert result["auto_purge_enabled"] is False

    def test_update_both(self, store):
        result = store.update_settings(retention_days=30, auto_purge_enabled=True)
        assert result["retention_days"] == 30
        assert result["auto_purge_enabled"] is True

    def test_update_none_keeps_current(self, store):
        store.update_settings(retention_days=21)
        result = store.update_settings(auto_purge_enabled=False)
        assert result["retention_days"] == 21

    def test_stats_includes_auto_purge(self, store):
        store.update_settings(auto_purge_enabled=True)
        stats = store.get_stats()
        assert stats["auto_purge_enabled"] is True

    def test_stats_auto_purge_default_false(self, store):
        stats = store.get_stats()
        assert stats["auto_purge_enabled"] is False

    def test_update_settings_no_db(self, history_mod):
        s = history_mod.TelemetryHistoryStore(db_path="")
        result = s.update_settings(retention_days=10)
        assert result["retention_days"] == 10


# ===========================================================================
# Group 2: export_csv
# ===========================================================================

class TestExportCsv:
    """Tests for TelemetryHistoryStore.export_csv (S115)."""

    def test_export_empty(self, store):
        csv = store.export_csv()
        lines = csv.strip().split("\n")
        assert len(lines) == 1
        assert lines[0].startswith("id,request_id,model")

    def test_export_header_columns(self, store):
        csv = store.export_csv()
        header = csv.strip().split("\n")[0]
        expected = "id,request_id,model,timestamp,latency_ms,tokens_in,tokens_out,tok_per_sec,prompt_eval_ms,token_gen_ms"
        assert header == expected

    def test_export_with_events(self, populated_store):
        csv = populated_store.export_csv()
        lines = csv.strip().split("\n")
        assert len(lines) == 11  # header + 10 events

    def test_export_filtered_by_model(self, populated_store):
        csv = populated_store.export_csv(model="llama3")
        lines = csv.strip().split("\n")
        assert len(lines) == 6  # header + 5 llama3 events

    def test_export_filtered_model_not_found(self, populated_store):
        csv = populated_store.export_csv(model="nonexistent")
        lines = csv.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_export_no_db(self, history_mod):
        s = history_mod.TelemetryHistoryStore(db_path="")
        csv = s.export_csv()
        assert csv.startswith("id,request_id,model")
        assert len(csv.strip().split("\n")) == 1

    def test_export_csv_ends_with_newline(self, populated_store):
        csv = populated_store.export_csv()
        assert csv.endswith("\n")

    def test_export_csv_contains_model_names(self, populated_store):
        csv = populated_store.export_csv()
        assert "llama3" in csv
        assert "mistral" in csv

    def test_export_csv_numeric_fields(self, populated_store):
        csv = populated_store.export_csv()
        lines = csv.strip().split("\n")
        # Check first data row has numeric latency field
        parts = lines[1].split(",")
        assert len(parts) == 10
        # latency_ms is the 5th field (index 4)
        latency_val = float(parts[4])
        assert latency_val > 0


# ===========================================================================
# Group 3: Auto-purge timer lifecycle
# ===========================================================================

class TestAutoPurge:
    """Tests for auto-purge timer management."""

    def test_auto_purge_timer_created(self, store):
        store.update_settings(auto_purge_enabled=True)
        assert store._auto_purge_timer is not None
        assert store._auto_purge_timer.is_alive()
        store._stop_auto_purge()

    def test_auto_purge_timer_cancelled(self, store):
        store.update_settings(auto_purge_enabled=True)
        assert store._auto_purge_timer is not None
        store.update_settings(auto_purge_enabled=False)
        assert store._auto_purge_timer is None

    def test_auto_purge_timer_replaced_on_re_enable(self, store):
        store.update_settings(auto_purge_enabled=True)
        timer1 = store._auto_purge_timer
        store.update_settings(auto_purge_enabled=False)
        store.update_settings(auto_purge_enabled=True)
        timer2 = store._auto_purge_timer
        assert timer2 is not timer1
        store._stop_auto_purge()

    def test_shutdown_stops_auto_purge(self, history_mod, tmp_db):
        s = history_mod.TelemetryHistoryStore(db_path=tmp_db)
        s.update_settings(auto_purge_enabled=True)
        assert s._auto_purge_timer is not None
        s.shutdown()
        assert s._auto_purge_timer is None

    def test_auto_purge_off_no_timer(self, store):
        store.update_settings(auto_purge_enabled=False)
        assert store._auto_purge_timer is None


# ===========================================================================
# Group 4: Route validation
# ===========================================================================

class TestRoutes:
    """Tests for new S115 API routes."""

    def test_routes_telemetry_ast(self):
        path = ROOT / "opti_oignon" / "api" / "routes_telemetry.py"
        tree = ast.parse(path.read_text())
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert "update_history_settings" in funcs
        assert "export_history_csv" in funcs

    def test_routes_telemetry_imports_plain_text_response(self):
        src = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert "PlainTextResponse" in src

    def test_settings_route_uses_put(self):
        src = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert '@router.put("/history/settings"' in src

    def test_export_route_uses_get(self):
        src = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert '@router.get("/history/export"' in src

    def test_export_route_csv_content_type(self):
        src = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert "text/csv" in src

    def test_export_route_content_disposition(self):
        src = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert "Content-Disposition" in src
        assert "telemetry_history.csv" in src

    def test_settings_route_model(self):
        src = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert "TelemetryHistorySettingsRequest" in src
        assert "TelemetryHistorySettingsResponse" in src

    def test_total_telemetry_route_count(self):
        """Verify total number of telemetry API endpoints."""
        src = (ROOT / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        import re
        decorators = re.findall(r"@router\.(get|post|put|delete|patch)\(", src)
        assert len(decorators) == 10  # 8 from S113/S114 + 2 new S115


# ===========================================================================
# Group 5: Schema validation
# ===========================================================================

class TestSchemas:
    """Tests for new S115 Pydantic schemas."""

    def test_settings_request_schema(self):
        _stub_opti()
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.api.schemas",
            str(ROOT / "opti_oignon" / "api" / "schemas.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        req = mod.TelemetryHistorySettingsRequest(retention_days=14, auto_purge_enabled=True)
        assert req.retention_days == 14
        assert req.auto_purge_enabled is True

    def test_settings_request_optional(self):
        _stub_opti()
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.api.schemas",
            str(ROOT / "opti_oignon" / "api" / "schemas.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        req = mod.TelemetryHistorySettingsRequest()
        assert req.retention_days is None
        assert req.auto_purge_enabled is None

    def test_settings_response_schema(self):
        _stub_opti()
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.api.schemas",
            str(ROOT / "opti_oignon" / "api" / "schemas.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        resp = mod.TelemetryHistorySettingsResponse(retention_days=30, auto_purge_enabled=True)
        assert resp.retention_days == 30
        d = resp.model_dump()
        assert "auto_purge_enabled" in d

    def test_stats_response_has_auto_purge(self):
        _stub_opti()
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.api.schemas",
            str(ROOT / "opti_oignon" / "api" / "schemas.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        resp = mod.TelemetryHistoryStatsResponse(auto_purge_enabled=True)
        assert resp.auto_purge_enabled is True


# ===========================================================================
# Group 6: Frontend — TelemetryHistoryPanel
# ===========================================================================

class TestTelemetryHistoryPanel:
    """Tests for TelemetryHistoryPanel.svelte (S115)."""

    @pytest.fixture(autouse=True)
    def _load_src(self):
        self.path = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "TelemetryHistoryPanel.svelte"
        self.src = self.path.read_text()

    def test_file_exists(self):
        assert self.path.exists()

    def test_has_script_block(self):
        assert "<script" in self.src
        assert "</script>" in self.src

    def test_has_style_block(self):
        assert "<style>" in self.src
        assert "</style>" in self.src

    def test_imports_telemetry_api(self):
        assert "getTelemetryHistory" in self.src
        assert "getTelemetryTrends" in self.src
        assert "getHistoryModelBreakdown" in self.src
        assert "getHistoryStats" in self.src
        assert "purgeHistory" in self.src

    def test_imports_settings_api(self):
        assert "updateHistorySettings" in self.src
        assert "getHistoryExportUrl" in self.src

    def test_imports_event_timeline(self):
        assert "EventTimeline" in self.src

    def test_has_paginated_table(self):
        assert "event-table" in self.src
        assert "pagination" in self.src
        assert "goToPage" in self.src

    def test_has_trend_chart(self):
        assert "trend-chart" in self.src
        assert "trend-bar" in self.src

    def test_has_model_filter(self):
        assert "model-select" in self.src
        assert "modelFilter" in self.src

    def test_has_model_breakdown(self):
        assert "model-card" in self.src
        assert "modelBreakdown" in self.src

    def test_has_stats_display(self):
        assert "stats-grid" in self.src
        assert "Stored Events" in self.src
        assert "Retention" in self.src

    def test_has_purge_controls(self):
        assert "purge-section" in self.src
        assert "Purge All Events" in self.src
        assert "purgeConfirmOpen" in self.src

    def test_has_retention_slider(self):
        assert "retention-slider" in self.src
        assert 'type="range"' in self.src

    def test_has_auto_purge_toggle(self):
        assert "autoPurgeEnabled" in self.src
        assert 'type="checkbox"' in self.src

    def test_has_save_settings(self):
        assert "saveSettings" in self.src
        assert "Save Settings" in self.src

    def test_has_storage_bar(self):
        assert "storage-bar" in self.src
        assert "storage-fill" in self.src
        assert "storageUsagePct" in self.src

    def test_has_export_csv(self):
        assert "handleExportCsv" in self.src
        assert "Export to CSV" in self.src

    def test_no_hardcoded_hex_in_template(self):
        import re
        style_start = self.src.index("<style>")
        template = self.src[:style_start]
        hexes = re.findall(r"#[0-9a-fA-F]{3,8}\b", template)
        # Filter out Svelte template syntax like {#each}
        real_hexes = [h for h in hexes if h.replace("#", "").isalnum() and len(h) >= 4]
        for h in real_hexes:
            # Must be inside var(--oo-*)
            idx = template.index(h)
            context = template[max(0, idx - 30):idx + len(h)]
            assert "var(--oo-" in context, f"Hardcoded hex {h} found outside var(): {context}"

    def test_uses_oo_css_variables(self):
        assert "--oo-text-primary" in self.src
        assert "--oo-bg-surface" in self.src
        assert "--oo-accent-primary" in self.src

    def test_export_initialModelFilter_prop(self):
        assert "export let initialModelFilter" in self.src


# ===========================================================================
# Group 7: Frontend — EventTimeline
# ===========================================================================

class TestEventTimeline:
    """Tests for EventTimeline.svelte (S115)."""

    @pytest.fixture(autouse=True)
    def _load_src(self):
        self.path = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "EventTimeline.svelte"
        self.src = self.path.read_text()

    def test_file_exists(self):
        assert self.path.exists()

    def test_has_script_and_style(self):
        assert "<script" in self.src
        assert "</script>" in self.src
        assert "<style>" in self.src
        assert "</style>" in self.src

    def test_imports_telemetry_api(self):
        assert "getTelemetryHistory" in self.src

    def test_has_zoom_controls(self):
        assert "zoomLevels" in self.src
        assert "1h" in self.src
        assert "6h" in self.src
        assert "24h" in self.src
        assert "7d" in self.src

    def test_has_percentile_computation(self):
        assert "computePercentiles" in self.src
        assert "p50" in self.src
        assert "p95" in self.src

    def test_has_dot_color_function(self):
        assert "dotColor" in self.src

    def test_has_tooltip(self):
        assert "tl-tooltip" in self.src
        assert "tooltipEvent" in self.src
        assert "tooltipVisible" in self.src

    def test_has_event_dots(self):
        assert "event-dot" in self.src

    def test_has_click_selection(self):
        assert "selectedEvent" in self.src
        assert "detail-card" in self.src

    def test_has_axis_labels(self):
        assert "tl-axis" in self.src
        assert "axisLabels" in self.src

    def test_has_legend(self):
        assert "tl-legend" in self.src
        assert "dot-green" in self.src
        assert "dot-yellow" in self.src
        assert "dot-red" in self.src

    def test_dispatches_select_event(self):
        assert "dispatch(" in self.src
        assert "selectEvent" in self.src

    def test_export_model_filter_prop(self):
        assert "export let modelFilter" in self.src

    def test_auto_refresh(self):
        assert "refreshTimer" in self.src
        assert "setInterval" in self.src

    def test_event_y_position(self):
        assert "eventY" in self.src

    def test_dot_size_varies(self):
        assert "dotSize" in self.src

    def test_y_axis_labels(self):
        assert "Slow" in self.src
        assert "Fast" in self.src

    def test_no_hardcoded_hex_in_style(self):
        import re
        style_match = re.search(r"<style>([\s\S]*?)</style>", self.src)
        if style_match:
            style = style_match.group(1)
            hexes = re.findall(r"#[0-9a-fA-F]{3,8}\b", style)
            for h in hexes:
                idx = style.index(h)
                context = style[max(0, idx - 30):idx + len(h)]
                assert "var(--oo-" in context, f"Hardcoded hex {h} outside var(): {context}"


# ===========================================================================
# Group 8: ObservabilityPanel History integration
# ===========================================================================

class TestObservabilityIntegration:
    """Tests for History tab in ObservabilityPanel.svelte."""

    @pytest.fixture(autouse=True)
    def _load_src(self):
        self.path = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "ObservabilityPanel.svelte"
        self.src = self.path.read_text()

    def test_imports_history_panel(self):
        assert "TelemetryHistoryPanel" in self.src

    def test_has_history_subtab(self):
        assert "'history'" in self.src
        assert "History" in self.src

    def test_history_tab_renders_panel(self):
        assert "<TelemetryHistoryPanel" in self.src

    def test_history_tab_passes_model_filter(self):
        assert "initialModelFilter={linkedModel}" in self.src

    def test_profiler_cross_link_targets_history(self):
        assert "activeSubTab = 'history'" in self.src

    def test_history_card_navigates_to_history(self):
        # The Event History overview card should navigate to 'history' tab
        assert "on:click={() => activeSubTab = 'history'}" in self.src

    def test_linked_model_banner_in_history(self):
        # linked-model-banner should be inside the history tab block
        idx_history = self.src.index("<!-- History tab -->")
        idx_banner = self.src.index("linked-model-banner")
        assert idx_banner > idx_history


# ===========================================================================
# Group 9: Frontend API client updates
# ===========================================================================

class TestApiClientUpdates:
    """Tests for telemetry.ts API client S115 additions."""

    @pytest.fixture(autouse=True)
    def _load_src(self):
        self.path = ROOT / "frontend" / "src" / "lib" / "api" / "telemetry.ts"
        self.src = self.path.read_text()

    def test_has_settings_request_interface(self):
        assert "HistorySettingsRequest" in self.src

    def test_has_settings_response_interface(self):
        assert "HistorySettingsResponse" in self.src

    def test_has_update_settings_function(self):
        assert "updateHistorySettings" in self.src

    def test_has_export_url_function(self):
        assert "getHistoryExportUrl" in self.src

    def test_settings_uses_put(self):
        assert "apiPut" in self.src

    def test_export_url_returns_string(self):
        assert "): string" in self.src or "string {" in self.src

    def test_history_stats_has_auto_purge(self):
        assert "auto_purge_enabled" in self.src

    def test_export_url_handles_model_param(self):
        assert "encodeURIComponent" in self.src


# ===========================================================================
# Group 10: Backend module AST integrity
# ===========================================================================

class TestBackendAst:
    """AST validation for modified backend modules."""

    def test_telemetry_history_ast(self):
        path = ROOT / "opti_oignon" / "telemetry_history.py"
        tree = ast.parse(path.read_text())
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "TelemetryHistoryStore" in classes

    def test_telemetry_history_methods(self):
        path = ROOT / "opti_oignon" / "telemetry_history.py"
        tree = ast.parse(path.read_text())
        methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TelemetryHistoryStore":
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        assert "update_settings" in methods
        assert "export_csv" in methods
        assert "_start_auto_purge" in methods
        assert "_stop_auto_purge" in methods

    def test_schemas_ast(self):
        path = ROOT / "opti_oignon" / "api" / "schemas.py"
        tree = ast.parse(path.read_text())
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "TelemetryHistorySettingsRequest" in classes
        assert "TelemetryHistorySettingsResponse" in classes

    def test_routes_telemetry_ast(self):
        path = ROOT / "opti_oignon" / "api" / "routes_telemetry.py"
        tree = ast.parse(path.read_text())
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert "update_history_settings" in funcs
        assert "export_history_csv" in funcs
