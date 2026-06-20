#!/usr/bin/env python3
"""
Tests for SpeculativeGenerator -- S70 Step 1: Core functionality.

Covers:
- YAML config loading and defaults
- Config update and persistence
- Draft-verify loop: draft accepted (high convergence)
- Draft-verify loop: draft rejected (low convergence, verify generates)
- Convergence detection (compute_similarity)
- Iteration limit respected
- Draft failure fallback to verify
- SpeculativeResult field validation
- Status reporting
- Properties and singleton
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

_MODULE_PATH = Path(__file__).resolve().parent.parent / "opti_oignon" / "speculative.py"


def _load_module():
    """Load speculative.py directly from file path."""
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = [str(_MODULE_PATH.parent)]
    sys.modules.setdefault("opti_oignon", pkg)

    # Load self_correction stub
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
            stub = types.ModuleType("opti_oignon.self_correction")
            sys.modules["opti_oignon.self_correction"] = stub

    # Stub ollama
    if "ollama" not in sys.modules:
        ollama_stub = types.ModuleType("ollama")
        ollama_stub.chat = MagicMock(return_value={"message": {"content": "stub"}})
        ollama_stub.list = MagicMock(return_value={"models": []})
        sys.modules["ollama"] = ollama_stub

    spec = importlib.util.spec_from_file_location("opti_oignon.speculative", str(_MODULE_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.speculative"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
SpeculativeGenerator = _mod.SpeculativeGenerator
SpeculativeResult = _mod.SpeculativeResult
compute_similarity = _mod.compute_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(tmpdir: Path, data: dict) -> Path:
    """Write a YAML config to a temp file and return its path."""
    path = tmpdir / "speculative.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return path


def _make_generator(tmpdir: Path, **overrides) -> SpeculativeGenerator:
    """Create a SpeculativeGenerator with a temp config."""
    defaults = {
        "enabled": True,
        "draft_model": "test-draft:7b",
        "verify_model": "test-verify:32b",
        "max_iterations": 2,
        "convergence_threshold": 0.85,
        "draft_max_tokens": 1024,
        "verify_max_tokens": 2048,
        "draft_temperature": 0.5,
        "verify_temperature": 0.3,
    }
    defaults.update(overrides)
    path = _write_config(tmpdir, defaults)
    return SpeculativeGenerator(config_path=path)


# ===========================================================================
# Tests
# ===========================================================================


class TestSpeculativeConfig:
    """Config loading, defaults, updates, persistence."""

    def test_load_from_yaml(self, tmp_path):
        gen = _make_generator(tmp_path, draft_model="fast:3b", verify_model="big:70b")
        assert gen.draft_model == "fast:3b"
        assert gen.verify_model == "big:70b"
        assert gen.enabled is True

    def test_defaults_when_no_file(self, tmp_path):
        gen = SpeculativeGenerator(config_path=tmp_path / "nonexistent.yaml")
        assert gen.enabled is False
        assert gen.draft_model == "qwen3:8b"
        assert gen.verify_model == "qwen3:32b"
        assert gen.max_iterations == 2
        assert gen.convergence_threshold == 0.85

    def test_partial_config(self, tmp_path):
        path = _write_config(tmp_path, {"enabled": True, "draft_model": "tiny:1b"})
        gen = SpeculativeGenerator(config_path=path)
        assert gen.enabled is True
        assert gen.draft_model == "tiny:1b"
        assert gen.verify_model == "qwen3:32b"  # default

    def test_update_config_persists(self, tmp_path):
        gen = _make_generator(tmp_path)
        gen.update_config(draft_model="new-draft:14b", convergence_threshold=0.9)
        assert gen.draft_model == "new-draft:14b"
        assert gen.convergence_threshold == 0.9

        # Reload from file to verify persistence
        gen2 = SpeculativeGenerator(config_path=gen._config_path)
        assert gen2.draft_model == "new-draft:14b"
        assert gen2.convergence_threshold == 0.9

    def test_update_config_clamps_values(self, tmp_path):
        gen = _make_generator(tmp_path)
        gen.update_config(convergence_threshold=5.0, draft_temperature=-1.0, max_iterations=0)
        assert gen.convergence_threshold == 1.0
        assert gen._draft_temperature == 0.0
        assert gen.max_iterations == 1

    def test_get_config(self, tmp_path):
        gen = _make_generator(tmp_path)
        cfg = gen.get_config()
        assert cfg["enabled"] is True
        assert cfg["draft_model"] == "test-draft:7b"
        assert cfg["verify_model"] == "test-verify:32b"
        assert "max_iterations" in cfg
        assert "convergence_threshold" in cfg

    def test_enabled_property_setter(self, tmp_path):
        gen = _make_generator(tmp_path)
        gen.enabled = False
        assert gen.enabled is False
        gen.enabled = True
        assert gen.enabled is True


class TestComputeSimilarity:
    """Heuristic text similarity function."""

    def test_identical_texts(self):
        assert compute_similarity("hello world foo", "hello world foo") == pytest.approx(1.0)

    def test_empty_texts(self):
        assert compute_similarity("", "") == 0.0
        assert compute_similarity("hello", "") == 0.0
        assert compute_similarity("", "world") == 0.0

    def test_completely_different(self):
        sim = compute_similarity("alpha beta gamma", "one two three")
        assert sim < 0.3

    def test_partial_overlap(self):
        sim = compute_similarity(
            "the quick brown fox jumps",
            "the quick red fox runs",
        )
        assert 0.3 < sim < 0.9

    def test_high_similarity(self):
        sim = compute_similarity(
            "the answer is 42 because of the meaning of life",
            "the answer is 42 because of the meaning of everything",
        )
        assert sim > 0.6


class TestSpeculativeGenerate:
    """Draft-verify generation loop."""

    def test_draft_accepted_high_convergence(self, tmp_path):
        """When draft and verify are very similar, draft is accepted."""
        gen = _make_generator(tmp_path, convergence_threshold=0.5)

        def mock_draft(query):
            return "The answer to your question is 42."

        def mock_verify(query, draft):
            return "The answer to your question is 42."

        result = gen.generate("What is 42?", draft_call=mock_draft, verify_call=mock_verify)

        assert result.draft_accepted is True
        assert result.final_response == "The answer to your question is 42."
        assert result.iterations == 1
        assert result.convergence_score >= 0.5
        assert result.draft_model == "test-draft:7b"
        assert result.verify_model == "test-verify:32b"

    def test_draft_rejected_low_convergence(self, tmp_path):
        """When draft and verify differ significantly, verify response wins."""
        gen = _make_generator(tmp_path, convergence_threshold=0.99, max_iterations=1)

        def mock_draft(query):
            return "I have no idea about this topic."

        def mock_verify(query, draft):
            return "The Pythagorean theorem states that a squared plus b squared equals c squared."

        result = gen.generate("Explain Pythagorean theorem", draft_call=mock_draft, verify_call=mock_verify)

        assert result.draft_accepted is False
        assert result.final_response == result.verify_response
        assert result.draft_response == "I have no idea about this topic."
        assert result.iterations == 1

    def test_convergence_across_iterations(self, tmp_path):
        """Verify response becomes the new draft for subsequent iterations."""
        gen = _make_generator(tmp_path, convergence_threshold=0.99, max_iterations=3)
        call_count = [0]

        def mock_draft(query):
            return "draft response about alpha"

        def mock_verify(query, draft):
            call_count[0] += 1
            if call_count[0] == 1:
                return "completely different verify response about beta"
            elif call_count[0] == 2:
                return "still different response about gamma"
            else:
                return "still different response about gamma"  # same as iteration 2

        result = gen.generate("test", draft_call=mock_draft, verify_call=mock_verify)

        assert result.iterations >= 2
        assert call_count[0] >= 2

    def test_max_iterations_respected(self, tmp_path):
        """Generator stops after max_iterations even if not converged."""
        gen = _make_generator(tmp_path, convergence_threshold=0.99, max_iterations=2)
        verify_calls = [0]

        def mock_draft(query):
            return "draft"

        def mock_verify(query, draft):
            verify_calls[0] += 1
            return f"verify iteration {verify_calls[0]}"

        result = gen.generate("test", draft_call=mock_draft, verify_call=mock_verify)

        assert result.iterations == 2
        assert verify_calls[0] == 2

    def test_draft_failure_fallback(self, tmp_path):
        """When draft fails, fall back to verify model directly."""
        gen = _make_generator(tmp_path)

        def mock_draft(query):
            raise RuntimeError("Draft model unavailable")

        def mock_verify(query, draft):
            return "Fallback response from verify model"

        result = gen.generate("test", draft_call=mock_draft, verify_call=mock_verify)

        assert result.draft_accepted is False
        assert result.draft_response == ""
        assert result.final_response == "Fallback response from verify model"
        assert result.iterations == 1

    def test_verify_failure_keeps_draft(self, tmp_path):
        """When verify fails, keep the draft response."""
        gen = _make_generator(tmp_path)

        def mock_draft(query):
            return "Good draft response"

        def mock_verify(query, draft):
            raise RuntimeError("Verify model unavailable")

        result = gen.generate("test", draft_call=mock_draft, verify_call=mock_verify)

        # Verify failed, so the loop breaks with current_draft (which is the draft)
        assert result.draft_response == "Good draft response"

    def test_result_fields_complete(self, tmp_path):
        """SpeculativeResult has all required fields."""
        gen = _make_generator(tmp_path)

        result = gen.generate(
            "test",
            draft_call=lambda q: "draft text",
            verify_call=lambda q, d: "verify text",
        )

        assert isinstance(result, SpeculativeResult)
        assert isinstance(result.final_response, str)
        assert isinstance(result.draft_response, str)
        assert isinstance(result.verify_response, str)
        assert isinstance(result.draft_model, str)
        assert isinstance(result.verify_model, str)
        assert isinstance(result.draft_accepted, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.total_latency_ms, float)
        assert isinstance(result.draft_latency_ms, float)
        assert isinstance(result.verify_latency_ms, float)
        assert isinstance(result.convergence_score, float)

    def test_latency_tracked(self, tmp_path):
        """Latency fields are positive values."""
        gen = _make_generator(tmp_path)

        result = gen.generate(
            "test",
            draft_call=lambda q: "draft",
            verify_call=lambda q, d: "verify",
        )

        assert result.total_latency_ms >= 0
        assert result.draft_latency_ms >= 0
        assert result.verify_latency_ms >= 0


class TestSpeculativeStatus:
    """Status and last_result reporting."""

    def test_status_no_result(self, tmp_path):
        gen = _make_generator(tmp_path)
        status = gen.get_status()
        assert status["enabled"] is True
        assert status["draft_model"] == "test-draft:7b"
        assert status["last_result"] is None

    def test_status_after_generate(self, tmp_path):
        gen = _make_generator(tmp_path)
        gen.generate("test", draft_call=lambda q: "d", verify_call=lambda q, d: "d")
        status = gen.get_status()
        assert status["last_result"] is not None
        assert "draft_accepted" in status["last_result"]
        assert "total_latency_ms" in status["last_result"]

    def test_last_result_property(self, tmp_path):
        gen = _make_generator(tmp_path)
        assert gen.last_result is None
        gen.generate("test", draft_call=lambda q: "d", verify_call=lambda q, d: "d")
        assert gen.last_result is not None


class TestSpeculativeSingleton:
    """Module-level singleton initialization."""

    def test_singleton_exists(self):
        assert hasattr(_mod, "speculative_generator")
        assert hasattr(_mod, "SPECULATIVE_AVAILABLE")

    def test_singleton_is_speculative_generator(self):
        if _mod.speculative_generator is not None:
            assert isinstance(_mod.speculative_generator, SpeculativeGenerator)
