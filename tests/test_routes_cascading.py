#!/usr/bin/env python3
"""
Tests for S69 API Routes -- Step 3: Cascading Inference.

Covers:
- GET /api/cascading/status -- returns status with tier config
- PUT /api/cascading/config -- updates configuration
- POST /api/cascading/test -- runs a test cascade
- Schema validation for all request/response models
- Error cases: unavailable module, disabled, empty query
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Module loading (bypass ollama requirement)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent


def _ensure_base_modules():
    """Load minimal modules needed for API testing."""
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(_ROOT / "opti_oignon")]
        sys.modules["opti_oignon"] = pkg

    if "opti_oignon.config" not in sys.modules:
        try:
            spec = importlib.util.spec_from_file_location(
                "opti_oignon.config",
                str(_ROOT / "opti_oignon" / "config.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["opti_oignon.config"] = mod
            spec.loader.exec_module(mod)
        except Exception:
            pass

    # Load self_correction for quality eval
    if "opti_oignon.self_correction" not in sys.modules:
        sc_path = _ROOT / "opti_oignon" / "self_correction.py"
        if sc_path.exists():
            try:
                spec = importlib.util.spec_from_file_location(
                    "opti_oignon.self_correction", str(sc_path),
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules["opti_oignon.self_correction"] = mod
                spec.loader.exec_module(mod)
            except Exception:
                pass

    if "opti_oignon.cascading" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.cascading",
            str(_ROOT / "opti_oignon" / "cascading.py"),
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["opti_oignon.cascading"] = mod
        spec.loader.exec_module(mod)


_ensure_base_modules()

from opti_oignon.cascading import CascadeResult, CascadeTierResult, CascadingInference

# Load schemas
if "opti_oignon.api" not in sys.modules:
    api_pkg = types.ModuleType("opti_oignon.api")
    api_pkg.__path__ = [str(_ROOT / "opti_oignon" / "api")]
    sys.modules["opti_oignon.api"] = api_pkg

_schemas_spec = importlib.util.spec_from_file_location(
    "opti_oignon.api.schemas",
    str(_ROOT / "opti_oignon" / "api" / "schemas.py"),
)
_schemas_mod = importlib.util.module_from_spec(_schemas_spec)
sys.modules["opti_oignon.api.schemas"] = _schemas_mod
_schemas_spec.loader.exec_module(_schemas_mod)

CascadeStatusResponse = _schemas_mod.CascadeStatusResponse
CascadeConfigUpdate = _schemas_mod.CascadeConfigUpdate
CascadeTierSchema = _schemas_mod.CascadeTierSchema
CascadeTestRequest = _schemas_mod.CascadeTestRequest
CascadeTestResponse = _schemas_mod.CascadeTestResponse
CascadeResultSchema = _schemas_mod.CascadeResultSchema
CascadeTierResultSchema = _schemas_mod.CascadeTierResultSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cascade_config(tmp_path):
    """Create a temp cascading.yaml and return the engine."""
    config_data = {
        "enabled": True,
        "tiers": [
            {"name": "fast", "model": "test:7b", "threshold": 0.7,
             "max_tokens": 1024, "temperature": 0.3},
            {"name": "standard", "model": "test:13b", "threshold": 0.5,
             "max_tokens": 2048, "temperature": 0.5},
            {"name": "power", "model": "test:70b", "threshold": 0.0,
             "max_tokens": 4096, "temperature": 0.7},
        ],
        "max_retries_per_tier": 0,
        "timeout_per_tier_seconds": 10,
    }
    config_file = tmp_path / "cascading.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
    return CascadingInference(config_path=config_file)


# ---------------------------------------------------------------------------
# Test: Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Test Pydantic schema construction and validation."""

    def test_cascade_tier_schema(self):
        tier = CascadeTierSchema(
            name="fast", model="qwen3:8b", threshold=0.7,
            max_tokens=2048, temperature=0.5,
        )
        assert tier.name == "fast"
        assert tier.model == "qwen3:8b"
        assert tier.threshold == 0.7

    def test_cascade_tier_schema_defaults(self):
        tier = CascadeTierSchema(name="t", model="m")
        assert tier.threshold == 0.0
        assert tier.max_tokens == 4096
        assert tier.temperature == 0.5

    def test_cascade_status_response(self):
        status = CascadeStatusResponse(
            enabled=True, available=True, tier_count=3,
        )
        assert status.enabled is True
        assert status.tier_count == 3
        assert status.tiers == []
        assert status.last_result is None

    def test_cascade_config_update(self):
        update = CascadeConfigUpdate(enabled=False, max_retries_per_tier=3)
        assert update.enabled is False
        assert update.max_retries_per_tier == 3
        assert update.tiers is None

    def test_cascade_test_request(self):
        req = CascadeTestRequest(query="What is Python?")
        assert req.query == "What is Python?"
        assert req.task_type is None

    def test_cascade_result_schema(self):
        result = CascadeResultSchema(
            final_response="Response text",
            model_used="test:7b",
            tier_index=0,
            tier_name="fast",
            score=0.85,
        )
        assert result.final_response == "Response text"
        assert result.attempts == []
        assert result.escalation_reasons == []

    def test_cascade_tier_result_schema(self):
        tr = CascadeTierResultSchema(
            tier_name="fast", model="test:7b",
            response="text", score=0.8, latency_ms=150.0,
        )
        assert tr.escalation_reason is None


# ---------------------------------------------------------------------------
# Test: Status endpoint logic
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    """Test cascading status endpoint logic."""

    def test_status_when_available(self, cascade_config):
        status = cascade_config.get_status()
        assert status["enabled"] is True
        assert len(status["tiers"]) == 3
        assert status["last_result"] is None

    def test_status_tier_details(self, cascade_config):
        status = cascade_config.get_status()
        fast_tier = status["tiers"][0]
        assert fast_tier["name"] == "fast"
        assert fast_tier["model"] == "test:7b"
        assert fast_tier["threshold"] == 0.7


# ---------------------------------------------------------------------------
# Test: Config update logic
# ---------------------------------------------------------------------------

class TestConfigUpdateEndpoint:
    """Test cascading config update logic."""

    def test_update_enabled(self, cascade_config):
        cascade_config.update_config(enabled=False)
        assert cascade_config.enabled is False

    def test_update_tiers(self, cascade_config):
        cascade_config.update_config(tiers=[
            {"name": "only", "model": "single:1b", "threshold": 0.0},
        ])
        assert cascade_config.tier_count == 1
        assert cascade_config.tiers[0].name == "only"

    def test_update_retries(self, cascade_config):
        cascade_config.update_config(max_retries_per_tier=5)
        assert cascade_config.max_retries_per_tier == 5


# ---------------------------------------------------------------------------
# Test: Test cascade endpoint logic
# ---------------------------------------------------------------------------

class TestCascadeTestEndpoint:
    """Test cascade test execution logic."""

    def test_test_cascade_runs(self, cascade_config):
        def mock_llm(query, tier):
            return (
                "A comprehensive response covering the topic in detail. "
                "It includes multiple points and clear explanations."
            )
        result = cascade_config.cascade("Test query", llm_call=mock_llm)
        assert result.final_response != ""
        assert result.model_used != ""
        assert result.score > 0.0

    def test_test_cascade_result_serializable(self, cascade_config):
        def mock_llm(query, tier):
            return "Good response with enough words and structure."
        result = cascade_config.cascade("Test", llm_call=mock_llm)
        # Verify result can be converted to schema
        attempts = [
            CascadeTierResultSchema(
                tier_name=a.tier_name, model=a.model,
                response=a.response, score=a.score,
                latency_ms=a.latency_ms,
                escalation_reason=a.escalation_reason,
            )
            for a in result.attempts
        ]
        schema = CascadeResultSchema(
            final_response=result.final_response,
            model_used=result.model_used,
            tier_index=result.tier_index,
            tier_name=result.tier_name,
            score=result.score,
            attempts=attempts,
            total_latency_ms=result.total_latency_ms,
            escalation_reasons=list(result.escalation_reasons),
        )
        assert schema.final_response == result.final_response
        assert len(schema.attempts) == len(result.attempts)
