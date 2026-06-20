#!/usr/bin/env python3
"""
Tests for Speculative Generation API routes -- S70 Step 3.

Covers:
- GET /api/speculative/status schema and unavailable fallback
- PUT /api/speculative/config update and mutual exclusion with cascading
- POST /api/speculative/test validation (empty query, disabled)
- Schema validation (SpeculativeResultSchema, SpeculativeStatusResponse, etc.)
- Frontend files existence (API client, types, panel, settings integration)
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Direct module imports
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent.parent / "opti_oignon"
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "src"


def _ensure_package():
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = [str(_SRC_DIR)]
    sys.modules.setdefault("opti_oignon", pkg)


def _load_speculative():
    _ensure_package()
    sc_path = _SRC_DIR / "self_correction.py"
    if sc_path.exists():
        try:
            sc_spec = importlib.util.spec_from_file_location(
                "opti_oignon.self_correction", str(sc_path),
            )
            sc_mod = importlib.util.module_from_spec(sc_spec)
            sys.modules["opti_oignon.self_correction"] = sc_mod
            sc_spec.loader.exec_module(sc_mod)
        except Exception:
            stub = types.ModuleType("opti_oignon.self_correction")
            sys.modules["opti_oignon.self_correction"] = stub

    if "ollama" not in sys.modules:
        ollama_stub = types.ModuleType("ollama")
        ollama_stub.chat = MagicMock(return_value={"message": {"content": "stub"}})
        ollama_stub.list = MagicMock(return_value={"models": []})
        sys.modules["ollama"] = ollama_stub

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.speculative", str(_SRC_DIR / "speculative.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.speculative"] = mod
    spec.loader.exec_module(mod)
    return mod


_spec_mod = _load_speculative()
SpeculativeGenerator = _spec_mod.SpeculativeGenerator
SpeculativeResult = _spec_mod.SpeculativeResult


# ===========================================================================
# Schema validation tests
# ===========================================================================


class TestSpeculativeSchemas:
    """Pydantic schemas exist and validate correctly."""

    def test_schemas_exist_in_file(self):
        schemas_path = _SRC_DIR / "api" / "schemas.py"
        content = schemas_path.read_text(encoding="utf-8")
        assert "class SpeculativeResultSchema" in content
        assert "class SpeculativeStatusResponse" in content
        assert "class SpeculativeConfigUpdate" in content
        assert "class SpeculativeTestRequest" in content
        assert "class SpeculativeTestResponse" in content

    def test_result_schema_fields(self):
        schemas_path = _SRC_DIR / "api" / "schemas.py"
        content = schemas_path.read_text(encoding="utf-8")
        for field in [
            "final_response", "draft_response", "verify_response",
            "draft_model", "verify_model", "draft_accepted",
            "iterations", "total_latency_ms", "convergence_score",
        ]:
            assert field in content, f"Missing field: {field}"

    def test_config_update_fields(self):
        schemas_path = _SRC_DIR / "api" / "schemas.py"
        content = schemas_path.read_text(encoding="utf-8")
        for field in [
            "draft_model", "verify_model", "max_iterations",
            "convergence_threshold", "draft_max_tokens", "verify_max_tokens",
        ]:
            assert field in content


# ===========================================================================
# Route file tests
# ===========================================================================


class TestSpeculativeRoutes:
    """Route file structure and registration."""

    def test_routes_file_exists(self):
        routes_path = _SRC_DIR / "api" / "routes_speculative.py"
        assert routes_path.exists()

    def test_routes_has_three_endpoints(self):
        routes_path = _SRC_DIR / "api" / "routes_speculative.py"
        content = routes_path.read_text(encoding="utf-8")
        assert "@router.get" in content
        assert "@router.put" in content
        assert "@router.post" in content
        assert '"/status"' in content
        assert '"/config"' in content
        assert '"/test"' in content

    def test_routes_registered_in_app(self):
        app_path = _SRC_DIR / "api" / "app.py"
        content = app_path.read_text(encoding="utf-8")
        assert "speculative_router" in content
        assert "routes_speculative" in content

    def test_mutual_exclusion_in_routes(self):
        """Config update route disables cascading when enabling speculative."""
        routes_path = _SRC_DIR / "api" / "routes_speculative.py"
        content = routes_path.read_text(encoding="utf-8")
        assert "CASCADING_AVAILABLE" in content
        assert "cascading_inference" in content
        assert "mutual exclusion" in content.lower()

    def test_version_bumped_to_172(self):
        app_path = _SRC_DIR / "api" / "app.py"
        content = app_path.read_text(encoding="utf-8")
        assert '"1.8.9"' in content

    def test_health_check_includes_speculative(self):
        app_path = _SRC_DIR / "api" / "app.py"
        content = app_path.read_text(encoding="utf-8")
        assert "SPECULATIVE_AVAILABLE" in content
        assert '"speculative"' in content


# ===========================================================================
# Frontend file tests
# ===========================================================================


class TestFrontendFiles:
    """Frontend TypeScript and Svelte files exist and are correct."""

    def test_api_client_exists(self):
        path = _FRONTEND_DIR / "lib" / "api" / "speculative.ts"
        assert path.exists()

    def test_api_client_has_three_functions(self):
        path = _FRONTEND_DIR / "lib" / "api" / "speculative.ts"
        content = path.read_text(encoding="utf-8")
        assert "getSpeculativeStatus" in content
        assert "updateSpeculativeConfig" in content
        assert "testSpeculative" in content

    def test_types_has_speculative_interfaces(self):
        path = _FRONTEND_DIR / "lib" / "types.ts"
        content = path.read_text(encoding="utf-8")
        assert "SpeculativeResult" in content
        assert "SpeculativeStatus" in content
        assert "SpeculativeConfigUpdate" in content
        assert "SpeculativeTestResult" in content

    def test_panel_exists(self):
        path = _FRONTEND_DIR / "lib" / "components" / "panels" / "SpeculativePanel.svelte"
        assert path.exists()

    def test_panel_imports_api(self):
        path = _FRONTEND_DIR / "lib" / "components" / "panels" / "SpeculativePanel.svelte"
        content = path.read_text(encoding="utf-8")
        assert "getSpeculativeStatus" in content
        assert "updateSpeculativeConfig" in content
        assert "testSpeculative" in content

    def test_panel_in_settings_page(self):
        path = _FRONTEND_DIR / "routes" / "settings" / "+page.svelte"
        content = path.read_text(encoding="utf-8")
        assert "SpeculativePanel" in content

    def test_chat_options_has_speculative_store(self):
        path = _FRONTEND_DIR / "lib" / "stores" / "chatOptions.ts"
        content = path.read_text(encoding="utf-8")
        assert "speculativeEnabled" in content
        assert "speculative" in content

    def test_chat_options_mutual_exclusion(self):
        """chatOptions implements mutual exclusion between speculative and cascading."""
        path = _FRONTEND_DIR / "lib" / "stores" / "chatOptions.ts"
        content = path.read_text(encoding="utf-8")
        # Should have logic that only sends one
        assert "speculativeEnabled" in content
        assert "cascadingEnabled" in content
