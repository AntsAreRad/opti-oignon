#!/usr/bin/env python3
"""
Tests for CascadingInference -- S69 Step 1: Core functionality.

Covers:
- YAML config loading and defaults
- Tier configuration management
- Quality evaluation (with and without self_correction)
- Cascade logic: fast passes, fast fails -> standard passes,
  all fail -> power tier, all tiers exhausted
- Retry logic within tiers
- Timeout/error handling
- CascadeResult and CascadeTierResult field validation
- Config persistence
- Status reporting
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
# Direct module import (bypass __init__.py which requires ollama)
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).resolve().parent.parent / "opti_oignon" / "cascading.py"


def _load_module():
    """Load cascading.py directly from file path."""
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = [str(_MODULE_PATH.parent)]
    sys.modules.setdefault("opti_oignon", pkg)

    # Load self_correction stub (for compute_heuristic_quality)
    sc_path = _MODULE_PATH.parent / "self_correction.py"
    if sc_path.exists():
        try:
            sc_spec = importlib.util.spec_from_file_location(
                "opti_oignon.self_correction",
                str(sc_path),
            )
            sc_mod = importlib.util.module_from_spec(sc_spec)
            sys.modules["opti_oignon.self_correction"] = sc_mod
            sc_spec.loader.exec_module(sc_mod)
        except Exception:
            # Create minimal stub
            stub = types.ModuleType("opti_oignon.self_correction")
            sys.modules["opti_oignon.self_correction"] = stub

    # Load config.py
    config_path = _MODULE_PATH.parent / "config.py"
    if config_path.exists():
        try:
            config_spec = importlib.util.spec_from_file_location(
                "opti_oignon.config",
                str(config_path),
            )
            config_mod = importlib.util.module_from_spec(config_spec)
            sys.modules["opti_oignon.config"] = config_mod
            config_spec.loader.exec_module(config_mod)
        except Exception:
            pass

    # Load cascading.py
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.cascading",
        str(_MODULE_PATH),
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.cascading"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
CascadingInference = _mod.CascadingInference
CascadeResult = _mod.CascadeResult
CascadeTierResult = _mod.CascadeTierResult
CascadeTierConfig = _mod.CascadeTierConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary cascading.yaml for testing."""
    config_data = {
        "enabled": True,
        "tiers": [
            {"name": "fast", "model": "test-small:7b", "threshold": 0.7,
             "max_tokens": 1024, "temperature": 0.3},
            {"name": "standard", "model": "test-medium:13b", "threshold": 0.5,
             "max_tokens": 2048, "temperature": 0.5},
            {"name": "power", "model": "test-large:70b", "threshold": 0.0,
             "max_tokens": 4096, "temperature": 0.7},
        ],
        "max_retries_per_tier": 1,
        "timeout_per_tier_seconds": 10,
        "score_weights": {
            "completeness": 0.5,
            "coherence": 0.3,
            "hallucination_penalty": 0.2,
        },
    }
    config_file = tmp_path / "cascading.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)
    return config_file


@pytest.fixture
def engine(tmp_config):
    """Create a CascadingInference instance with test config."""
    return CascadingInference(config_path=tmp_config)


@pytest.fixture
def engine_defaults(tmp_path):
    """Create a CascadingInference with non-existent config (defaults)."""
    return CascadingInference(config_path=tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

def make_llm_fn(responses: dict[str, str]):
    """Create a mock LLM callable that returns fixed responses per model.

    Args:
        responses: Mapping of model name -> response text.
    """
    def llm_call(query, tier):
        resp = responses.get(tier.model, "")
        if resp == "__ERROR__":
            raise RuntimeError(f"Simulated error for {tier.model}")
        return resp
    return llm_call


def make_counting_llm(responses: dict[str, list[str]]):
    """Create a mock LLM that returns different responses on successive calls.

    Args:
        responses: Mapping of model name -> list of response strings.
    """
    counters = {model: 0 for model in responses}

    def llm_call(query, tier):
        idx = counters.get(tier.model, 0)
        resps = responses.get(tier.model, [""])
        result = resps[min(idx, len(resps) - 1)]
        counters[tier.model] = idx + 1
        if result == "__ERROR__":
            raise RuntimeError(f"Simulated error for {tier.model}")
        return result
    return llm_call


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Test YAML config loading and defaults."""

    def test_loads_from_yaml(self, engine, tmp_config):
        assert engine.enabled is True
        assert engine.tier_count == 3
        assert engine.tiers[0].name == "fast"
        assert engine.tiers[0].model == "test-small:7b"
        assert engine.tiers[0].threshold == 0.7
        assert engine.tiers[1].name == "standard"
        assert engine.tiers[2].name == "power"

    def test_max_retries_loaded(self, engine):
        assert engine.max_retries_per_tier == 1

    def test_timeout_loaded(self, engine):
        assert engine.timeout_per_tier == 10

    def test_score_weights_loaded(self, engine):
        w = engine.score_weights
        assert w["completeness"] == 0.5
        assert w["coherence"] == 0.3
        assert w["hallucination_penalty"] == 0.2

    def test_defaults_when_config_missing(self, engine_defaults):
        assert engine_defaults.enabled is False
        assert engine_defaults.tier_count == 3
        assert engine_defaults.tiers[0].name == "fast"
        assert engine_defaults.tiers[1].name == "standard"
        assert engine_defaults.tiers[2].name == "power"
        assert engine_defaults.max_retries_per_tier == 1
        assert engine_defaults.timeout_per_tier == 30

    def test_empty_yaml(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        eng = CascadingInference(config_path=config_file)
        assert eng.enabled is False
        assert eng.tier_count == 0


class TestTierManagement:
    """Test tier update and configuration."""

    def test_update_tiers(self, engine):
        new_tiers = [
            {"name": "tiny", "model": "phi3:mini", "threshold": 0.8},
            {"name": "big", "model": "llama3:70b", "threshold": 0.0},
        ]
        engine.update_tiers(new_tiers)
        assert engine.tier_count == 2
        assert engine.tiers[0].name == "tiny"
        assert engine.tiers[1].name == "big"

    def test_update_tiers_ignores_invalid(self, engine):
        original_count = engine.tier_count
        engine.update_tiers([{"invalid": "data"}, "not_a_dict"])
        # No valid tiers -- original kept (empty list not applied)
        assert engine.tier_count == original_count

    def test_update_config(self, engine, tmp_config):
        result = engine.update_config(
            enabled=False,
            max_retries_per_tier=3,
            timeout_per_tier_seconds=60,
        )
        assert engine.enabled is False
        assert engine.max_retries_per_tier == 3
        assert engine.timeout_per_tier == 60
        assert result["enabled"] is False

    def test_update_config_persists(self, engine, tmp_config):
        engine.update_config(enabled=False, timeout_per_tier_seconds=99)
        # Reload from same file
        reloaded = CascadingInference(config_path=tmp_config)
        assert reloaded.enabled is False
        assert reloaded.timeout_per_tier == 99


class TestQualityEvaluation:
    """Test quality scoring."""

    def test_empty_response_scores_zero(self, engine):
        assert engine.evaluate_quality("test query", "") == 0.0

    def test_whitespace_response_scores_zero(self, engine):
        assert engine.evaluate_quality("test query", "   ") == 0.0

    def test_short_response_low_score(self, engine):
        score = engine.evaluate_quality(
            "Explain the theory of relativity in detail",
            "It's about physics."
        )
        assert score < 0.6

    def test_substantive_response_higher_score(self, engine):
        score = engine.evaluate_quality(
            "What is Python?",
            "Python is a high-level programming language known for its "
            "readable syntax and versatility. It supports multiple paradigms "
            "including object-oriented, functional, and procedural programming. "
            "Python is widely used in web development, data science, and automation."
        )
        assert score > 0.5

    def test_basic_fallback_quality(self, engine):
        # Test the static fallback method directly
        score = CascadingInference._basic_quality_score(
            "What is x?",
            "X is something. It has properties. It is used widely."
        )
        assert 0.0 < score <= 1.0


class TestCascadeLogic:
    """Test cascade execution with mock LLM calls."""

    def test_fast_tier_passes(self, engine):
        llm = make_llm_fn({
            "test-small:7b": (
                "Python is a high-level programming language. "
                "It supports object-oriented and functional paradigms. "
                "It is widely used for web development and data science."
            ),
        })
        result = engine.cascade("What is Python?", llm_call=llm)
        assert result.tier_name == "fast"
        assert result.tier_index == 0
        assert result.model_used == "test-small:7b"
        assert result.score >= 0.7
        assert len(result.attempts) == 1
        assert len(result.escalation_reasons) == 0

    def test_fast_fails_standard_passes(self, engine):
        llm = make_llm_fn({
            "test-small:7b": "Yes.",  # Too short -> low score
            "test-medium:13b": (
                "Python is an interpreted programming language designed by "
                "Guido van Rossum. It emphasizes code readability and supports "
                "multiple programming paradigms. It has a comprehensive standard library."
            ),
        })
        result = engine.cascade("What is Python?", llm_call=llm)
        assert result.tier_name == "standard"
        assert result.tier_index == 1
        assert result.model_used == "test-medium:13b"
        assert len(result.escalation_reasons) >= 1

    def test_all_escalate_to_power(self, engine):
        # Raise standard threshold so short responses cannot pass
        engine.update_tiers([
            {"name": "fast", "model": "test-small:7b", "threshold": 0.7},
            {"name": "standard", "model": "test-medium:13b", "threshold": 0.7},
            {"name": "power", "model": "test-large:70b", "threshold": 0.0},
        ])
        llm = make_llm_fn({
            "test-small:7b": "No.",
            "test-medium:13b": "Nope.",
            "test-large:70b": "This is a longer and more detailed response. "
                              "It covers the topic thoroughly with examples.",
        })
        result = engine.cascade("Explain quantum physics", llm_call=llm)
        # Power tier has threshold 0.0, so even a mediocre response passes
        assert result.tier_name == "power"
        assert result.tier_index == 2
        assert result.score >= 0.0
        assert len(result.escalation_reasons) >= 2

    def test_all_tiers_fail_returns_best(self, engine):
        # Set all thresholds very high so nothing passes
        engine.update_tiers([
            {"name": "fast", "model": "test-small:7b", "threshold": 0.99},
            {"name": "power", "model": "test-large:70b", "threshold": 0.99},
        ])
        llm = make_llm_fn({
            "test-small:7b": "Short answer.",
            "test-large:70b": "A slightly better but still brief answer.",
        })
        result = engine.cascade("Complex question here", llm_call=llm)
        # Should return best scoring attempt
        assert result.final_response != ""
        assert result.score > 0.0
        assert len(result.attempts) >= 2

    def test_no_tiers_configured(self, tmp_path):
        config_file = tmp_path / "empty_tiers.yaml"
        config_file.write_text(yaml.safe_dump({"enabled": True, "tiers": []}))
        eng = CascadingInference(config_path=config_file)
        result = eng.cascade("test")
        assert "[ERR]" in result.final_response
        assert result.tier_index == -1

    def test_cascade_result_fields(self, engine):
        llm = make_llm_fn({
            "test-small:7b": (
                "A complete and well-structured response about the topic. "
                "It covers multiple aspects and provides clear explanations."
            ),
        })
        result = engine.cascade("What is X?", llm_call=llm)
        assert isinstance(result, CascadeResult)
        assert isinstance(result.final_response, str)
        assert isinstance(result.model_used, str)
        assert isinstance(result.tier_index, int)
        assert isinstance(result.tier_name, str)
        assert isinstance(result.score, float)
        assert isinstance(result.attempts, list)
        assert isinstance(result.total_latency_ms, float)
        assert isinstance(result.escalation_reasons, list)
        assert result.total_latency_ms >= 0

    def test_cascade_tier_result_fields(self, engine):
        llm = make_llm_fn({
            "test-small:7b": "Short.",
            "test-medium:13b": (
                "A good response with enough detail to pass the threshold."
            ),
        })
        result = engine.cascade("What is Y?", llm_call=llm)
        for attempt in result.attempts:
            assert isinstance(attempt, CascadeTierResult)
            assert isinstance(attempt.tier_name, str)
            assert isinstance(attempt.model, str)
            assert isinstance(attempt.response, str)
            assert isinstance(attempt.score, float)
            assert isinstance(attempt.latency_ms, float)

    def test_last_result_stored(self, engine):
        assert engine.last_result is None
        llm = make_llm_fn({
            "test-small:7b": (
                "A reasonable response. It covers the basics well enough."
            ),
        })
        engine.cascade("test", llm_call=llm)
        assert engine.last_result is not None
        assert isinstance(engine.last_result, CascadeResult)


class TestRetryLogic:
    """Test retry within tiers before escalation."""

    def test_retry_on_low_score(self, engine):
        responses = {
            # First call: bad, second call (retry): still bad -> escalate
            "test-small:7b": ["Bad.", "Also bad."],
            "test-medium:13b": [
                "A well-formed response with clear structure and detail."
            ],
        }
        llm = make_counting_llm(responses)
        result = engine.cascade("What is Z?", llm_call=llm)
        # Fast tier: 1 call + 1 retry = 2 attempts from fast
        fast_attempts = [a for a in result.attempts if a.tier_name == "fast"]
        assert len(fast_attempts) == 2  # 1 original + 1 retry

    def test_retry_on_error(self, engine):
        responses = {
            "test-small:7b": ["__ERROR__", "__ERROR__"],
            "test-medium:13b": [
                "A successful response from the standard tier."
            ],
        }
        llm = make_counting_llm(responses)
        result = engine.cascade("Test error handling", llm_call=llm)
        assert result.tier_name == "standard"
        error_attempts = [a for a in result.attempts if a.tier_name == "fast"]
        assert len(error_attempts) == 2

    def test_zero_retries(self, engine):
        engine.update_config(max_retries_per_tier=0)
        responses = {
            "test-small:7b": ["Bad."],
            "test-medium:13b": ["Ok response with enough words and detail."],
        }
        llm = make_counting_llm(responses)
        result = engine.cascade("Test", llm_call=llm)
        fast_attempts = [a for a in result.attempts if a.tier_name == "fast"]
        assert len(fast_attempts) == 1  # No retry


class TestErrorHandling:
    """Test error scenarios."""

    def test_all_tiers_error(self, engine):
        llm = make_llm_fn({
            "test-small:7b": "__ERROR__",
            "test-medium:13b": "__ERROR__",
            "test-large:70b": "__ERROR__",
        })
        result = engine.cascade("test", llm_call=llm)
        assert "[ERR]" in result.final_response
        assert result.tier_index == -1
        assert len(result.escalation_reasons) == 3

    def test_partial_errors_recovers(self, engine):
        llm = make_llm_fn({
            "test-small:7b": "__ERROR__",
            "test-medium:13b": "__ERROR__",
            "test-large:70b": "Power tier delivers a solid response."
        })
        result = engine.cascade("test", llm_call=llm)
        assert result.tier_name == "power"
        assert result.model_used == "test-large:70b"


class TestStatus:
    """Test status reporting."""

    def test_status_before_cascade(self, engine):
        status = engine.get_status()
        assert status["enabled"] is True
        assert status["tier_count"] == 3
        assert status["last_result"] is None
        assert len(status["tiers"]) == 3

    def test_status_after_cascade(self, engine):
        llm = make_llm_fn({
            "test-small:7b": (
                "A good response. Clear, detailed, and well-structured."
            ),
        })
        engine.cascade("test", llm_call=llm)
        status = engine.get_status()
        assert status["last_result"] is not None
        assert "model_used" in status["last_result"]
        assert "tier_name" in status["last_result"]
        assert "score" in status["last_result"]

    def test_get_config(self, engine):
        cfg = engine.get_config()
        assert "enabled" in cfg
        assert "tiers" in cfg
        assert "max_retries_per_tier" in cfg
        assert "timeout_per_tier_seconds" in cfg
        assert "score_weights" in cfg
        assert len(cfg["tiers"]) == 3
