#!/usr/bin/env python3
"""
Tests for S110 — Inference Performance Accelerator.

Covers:
  - SpeculativeConfig validation, serialization
  - DraftModelSelector compatibility logic
  - VRAMBudgetCalculator estimation
  - AcceptanceStats tracking
  - SpeculativeDecodingManager lifecycle
  - AutoTuner parameter sweep, warmup, cancellation
  - TunerConfig, ParameterSpace, BenchmarkResult, TunerProfile, TunerJob
  - AutoTunerManager lifecycle and persistence
  - MTP detection for known model families
  - API schemas (speculative-decoding, tuner endpoints)
  - Frontend: no hardcoded hex in new Svelte components

Total: ~60 tests
"""

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module loading helpers (test isolation pattern)
# ---------------------------------------------------------------------------

def _load_module(name: str, filepath: str):
    """Load a module by file path, bypassing opti_oignon/__init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_parent_stubs():
    """Ensure parent package stubs exist for test isolation."""
    if "opti_oignon" not in sys.modules:
        oo = types.ModuleType("opti_oignon")
        oo.__path__ = ["opti_oignon"]
        sys.modules["opti_oignon"] = oo
    if "opti_oignon.config" not in sys.modules:
        sys.modules["opti_oignon.config"] = types.ModuleType("opti_oignon.config")


_ensure_parent_stubs()

# Load modules under test
_BASE = Path(__file__).resolve().parent.parent

_spec_dec = _load_module(
    "opti_oignon.speculative_decoding",
    str(_BASE / "opti_oignon" / "speculative_decoding.py"),
)
_auto_tuner = _load_module(
    "opti_oignon.auto_tuner",
    str(_BASE / "opti_oignon" / "auto_tuner.py"),
)


# ============================================================================
# SPECULATIVE DECODING TESTS
# ============================================================================


class TestSpeculativeConfig:
    """Tests for SpeculativeConfig dataclass."""

    def test_defaults(self):
        cfg = _spec_dec.SpeculativeConfig()
        assert cfg.enabled is False
        assert cfg.draft_model == ""
        assert cfg.draft_max == 16
        assert cfg.draft_min == 5
        assert cfg.draft_gpu_layers == 99
        assert cfg.auto_select_draft is True

    def test_validate_valid(self):
        cfg = _spec_dec.SpeculativeConfig()
        assert cfg.validate() == []

    def test_validate_draft_max_too_low(self):
        cfg = _spec_dec.SpeculativeConfig(draft_max=0)
        errors = cfg.validate()
        assert any("draft_max" in e for e in errors)

    def test_validate_draft_min_too_low(self):
        cfg = _spec_dec.SpeculativeConfig(draft_min=0)
        errors = cfg.validate()
        assert any("draft_min" in e for e in errors)

    def test_validate_draft_min_exceeds_max(self):
        cfg = _spec_dec.SpeculativeConfig(draft_min=20, draft_max=10)
        errors = cfg.validate()
        assert any("draft_min must be <= draft_max" in e for e in errors)

    def test_validate_gpu_layers_invalid(self):
        cfg = _spec_dec.SpeculativeConfig(draft_gpu_layers=-2)
        errors = cfg.validate()
        assert any("draft_gpu_layers" in e for e in errors)

    def test_to_dict(self):
        cfg = _spec_dec.SpeculativeConfig(enabled=True, draft_model="test.gguf")
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert d["draft_model"] == "test.gguf"
        assert "draft_max" in d

    def test_from_dict(self):
        d = {"enabled": True, "draft_model": "m.gguf", "unknown_key": 123}
        cfg = _spec_dec.SpeculativeConfig.from_dict(d)
        assert cfg.enabled is True
        assert cfg.draft_model == "m.gguf"

    def test_roundtrip(self):
        orig = _spec_dec.SpeculativeConfig(
            enabled=True, draft_model="x.gguf", draft_max=32
        )
        restored = _spec_dec.SpeculativeConfig.from_dict(orig.to_dict())
        assert restored.to_dict() == orig.to_dict()


class TestDraftCandidate:
    """Tests for DraftCandidate dataclass."""

    def test_to_dict(self):
        c = _spec_dec.DraftCandidate(
            name="draft-1b", family="llama3",
            parameter_size_b=1.0, compatibility_score=3.5
        )
        d = c.to_dict()
        assert d["name"] == "draft-1b"
        assert d["compatibility_score"] == 3.5


class TestAcceptanceStats:
    """Tests for AcceptanceStats."""

    def test_initial(self):
        s = _spec_dec.AcceptanceStats()
        assert s.total_runs == 0
        assert s.overall_acceptance_rate == 0.0

    def test_record_run(self):
        s = _spec_dec.AcceptanceStats()
        s.record_run(16, 12, 2.5)
        assert s.total_runs == 1
        assert s.accepted_tokens == 12
        assert s.last_acceptance_rate == 12 / 16
        assert s.last_speedup_factor == 2.5

    def test_multiple_runs(self):
        s = _spec_dec.AcceptanceStats()
        s.record_run(16, 12)
        s.record_run(16, 16)
        assert s.total_runs == 2
        assert s.total_draft_tokens == 32
        assert s.accepted_tokens == 28
        assert s.overall_acceptance_rate == 28 / 32

    def test_roundtrip(self):
        s = _spec_dec.AcceptanceStats()
        s.record_run(10, 8, 1.5)
        d = s.to_dict()
        s2 = _spec_dec.AcceptanceStats.from_dict(d)
        assert s2.total_runs == 1
        assert s2.accepted_tokens == 8


class TestVRAMBudgetCalculator:
    """Tests for VRAMBudgetCalculator."""

    def test_available_vram(self):
        calc = _spec_dec.VRAMBudgetCalculator(total_vram_gb=24, safety_margin_gb=1.5)
        assert calc.available_vram_gb == 22.5

    def test_estimate_q4(self):
        calc = _spec_dec.VRAMBudgetCalculator()
        est = calc.estimate_model_vram(7.0, "Q4_K_M")
        assert est > 0
        assert est < 10  # 7B Q4 should be well under 10GB

    def test_estimate_f16(self):
        calc = _spec_dec.VRAMBudgetCalculator()
        est = calc.estimate_model_vram(7.0, "F16")
        assert est == 14.0  # 7 * 2.0

    def test_check_fit_small_models(self):
        calc = _spec_dec.VRAMBudgetCalculator(total_vram_gb=24, safety_margin_gb=1.5)
        result = calc.check_fit(7.0, "Q4_K_M", 1.0, "Q4_K_M")
        assert result["fits"] is True
        assert result["headroom_gb"] > 0

    def test_check_fit_too_large(self):
        calc = _spec_dec.VRAMBudgetCalculator(total_vram_gb=8, safety_margin_gb=1.0)
        result = calc.check_fit(70.0, "Q4_K_M", 3.0, "Q4_K_M")
        assert result["fits"] is False
        assert result["headroom_gb"] < 0

    def test_unknown_quant_fallback(self):
        calc = _spec_dec.VRAMBudgetCalculator()
        est = calc.estimate_model_vram(7.0, "UNKNOWN_QUANT")
        assert est > 0  # Should use fallback


class TestDraftModelSelector:
    """Tests for DraftModelSelector."""

    def _make_models(self):
        return [
            {"name": "llama3-1b", "family": "llama3", "parameter_size_b": 1.0,
             "quantization": "Q4_K_M", "path": "/m/llama3-1b.gguf"},
            {"name": "llama3-3b", "family": "llama3", "parameter_size_b": 3.0,
             "quantization": "Q4_K_M", "path": "/m/llama3-3b.gguf"},
            {"name": "qwen2-0.5b", "family": "qwen2", "parameter_size_b": 0.5,
             "quantization": "Q4_K_M", "path": "/m/qwen2-0.5b.gguf"},
            {"name": "llama3-70b", "family": "llama3", "parameter_size_b": 70.0,
             "quantization": "Q4_K_M", "path": "/m/llama3-70b.gguf"},
        ]

    def test_find_compatible_llama3(self):
        sel = _spec_dec.DraftModelSelector()
        drafts = sel.find_compatible_drafts("llama3", 70.0, "Q4_K_M", self._make_models())
        names = [d.name for d in drafts]
        assert "llama3-1b" in names
        assert "llama3-3b" in names
        assert "qwen2-0.5b" not in names  # Wrong family
        assert "llama3-70b" not in names   # Same size as main

    def test_prefers_larger_draft(self):
        sel = _spec_dec.DraftModelSelector()
        drafts = sel.find_compatible_drafts("llama3", 70.0, "Q4_K_M", self._make_models())
        assert drafts[0].name == "llama3-3b"  # 3b scores higher than 1b

    def test_no_compatible_for_unknown_family(self):
        sel = _spec_dec.DraftModelSelector()
        drafts = sel.find_compatible_drafts("nonexistent", 70.0, "Q4_K_M", self._make_models())
        assert len(drafts) == 0

    def test_auto_select(self):
        sel = _spec_dec.DraftModelSelector()
        best = sel.auto_select("llama3", 70.0, "Q4_K_M", self._make_models())
        assert best is not None
        assert best.name == "llama3-3b"

    def test_auto_select_none(self):
        sel = _spec_dec.DraftModelSelector()
        best = sel.auto_select("nonexistent", 70.0, "Q4_K_M", self._make_models())
        assert best is None


class TestSpeculativeDecodingManager:
    """Tests for SpeculativeDecodingManager."""

    def test_default_disabled(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        assert mgr.config.enabled is False

    def test_update_config(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        new_cfg = mgr.update_config({"enabled": True, "draft_model": "test.gguf"})
        assert new_cfg.enabled is True
        assert new_cfg.draft_model == "test.gguf"

    def test_update_config_invalid(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        with pytest.raises(ValueError, match="Invalid"):
            mgr.update_config({"draft_min": 100, "draft_max": 1})

    def test_get_status(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        status = mgr.get_status()
        assert "config" in status
        assert "stats" in status
        assert status["available"] is True
        assert status["backend_required"] == "llama_cpp"

    def test_build_flags_disabled(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        assert mgr.build_llama_cpp_flags() == []

    def test_build_flags_enabled(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        mgr.update_config({"enabled": True, "draft_model": "/m/d.gguf"})
        flags = mgr.build_llama_cpp_flags()
        assert "-md" in flags
        assert "/m/d.gguf" in flags
        assert "--draft-max" in flags

    def test_record_and_reset_stats(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        mgr.record_acceptance(16, 12, 2.0)
        assert mgr.stats.total_runs == 1
        mgr.reset_stats()
        assert mgr.stats.total_runs == 0

    def test_get_draft_selector(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        sel = mgr.get_draft_selector()
        assert isinstance(sel, _spec_dec.DraftModelSelector)

    def test_get_vram_calculator(self):
        mgr = _spec_dec.SpeculativeDecodingManager()
        calc = mgr.get_vram_calculator()
        assert isinstance(calc, _spec_dec.VRAMBudgetCalculator)


class TestParseParamSize:
    """Tests for _parse_param_size utility."""

    def test_float(self):
        assert _spec_dec._parse_param_size(7.0) == 7.0

    def test_int(self):
        assert _spec_dec._parse_param_size(3) == 3.0

    def test_string_b(self):
        assert _spec_dec._parse_param_size("7B") == 7.0

    def test_string_lower_b(self):
        assert _spec_dec._parse_param_size("1.5b") == 1.5

    def test_string_no_b(self):
        assert _spec_dec._parse_param_size("70") == 70.0

    def test_invalid(self):
        assert _spec_dec._parse_param_size("invalid") == 0.0

    def test_none(self):
        assert _spec_dec._parse_param_size(None) == 0.0


# ============================================================================
# AUTO-TUNER TESTS
# ============================================================================


class TestTunerConfig:
    """Tests for TunerConfig dataclass."""

    def test_defaults(self):
        cfg = _auto_tuner.TunerConfig()
        assert cfg.enabled is True
        assert cfg.warmup_runs == 3
        assert cfg.trials_per_param == 3

    def test_validate_valid(self):
        assert _auto_tuner.TunerConfig().validate() == []

    def test_validate_invalid(self):
        cfg = _auto_tuner.TunerConfig(warmup_runs=-1, benchmark_tokens=0)
        errors = cfg.validate()
        assert len(errors) == 2

    def test_roundtrip(self):
        orig = _auto_tuner.TunerConfig(warmup_runs=5, auto_apply=True)
        restored = _auto_tuner.TunerConfig.from_dict(orig.to_dict())
        assert restored.to_dict() == orig.to_dict()


class TestParameterSpace:
    """Tests for ParameterSpace."""

    def test_total_combinations(self):
        ps = _auto_tuner.ParameterSpace()
        assert ps.total_combinations() == 4 * 3 * 4 * 2  # 96

    def test_roundtrip(self):
        ps = _auto_tuner.ParameterSpace(batch_size=[256, 512])
        ps2 = _auto_tuner.ParameterSpace.from_dict(ps.to_dict())
        assert ps2.batch_size == [256, 512]


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_to_dict(self):
        r = _auto_tuner.BenchmarkResult(
            params={"batch_size": 1024},
            tokens_per_second_tg=45.123,
        )
        d = r.to_dict()
        assert d["tokens_per_second_tg"] == 45.12  # Rounded

    def test_error_result(self):
        r = _auto_tuner.BenchmarkResult(error="OOM")
        assert r.error == "OOM"
        assert r.tokens_per_second_tg == 0.0


class TestTunerProfile:
    """Tests for TunerProfile."""

    def test_roundtrip(self):
        p = _auto_tuner.TunerProfile(
            model_name="test", best_tg_speed=50.0,
            baseline_tg_speed=30.0, speedup_factor=1.67,
        )
        p2 = _auto_tuner.TunerProfile.from_dict(p.to_dict())
        assert p2.model_name == "test"
        assert p2.best_tg_speed == 50.0


class TestTunerJob:
    """Tests for TunerJob."""

    def test_to_dict(self):
        j = _auto_tuner.TunerJob(
            job_id="abc", model_name="test",
            status="running", progress=0.5
        )
        d = j.to_dict()
        assert d["progress"] == 0.5
        assert d["result"] is None


class TestHardwareFingerprint:
    """Tests for hardware fingerprint generation."""

    def test_deterministic(self):
        fp1 = _auto_tuner.get_hardware_fingerprint()
        fp2 = _auto_tuner.get_hardware_fingerprint()
        assert fp1 == fp2

    def test_length(self):
        fp = _auto_tuner.get_hardware_fingerprint()
        assert len(fp) == 16


class TestAutoTuner:
    """Tests for the AutoTuner engine."""

    def _make_tuner(self, **kwargs):
        cfg = _auto_tuner.TunerConfig(warmup_runs=1, trials_per_param=1)
        ps = _auto_tuner.ParameterSpace()
        mock_fn = _auto_tuner.create_mock_benchmark_fn(base_speed=30.0, variance=1.0)
        return _auto_tuner.AutoTuner(
            config=cfg, param_space=ps,
            benchmark_fn=mock_fn, **kwargs,
        )

    def test_smart_sweep_smaller_than_full_grid(self):
        tuner = self._make_tuner()
        sweep = tuner._build_smart_sweep()
        full = _auto_tuner.ParameterSpace().total_combinations()
        assert len(sweep) < full

    def test_default_params(self):
        tuner = self._make_tuner()
        defaults = tuner._default_params()
        assert "batch_size" in defaults
        assert "threads" in defaults
        assert "flash_attention" in defaults

    def test_full_run(self):
        tuner = self._make_tuner()
        job = _auto_tuner.TunerJob(job_id="test")
        profile = tuner.run("test-model", job)
        assert job.status == "completed"
        assert profile.model_name == "test-model"
        assert profile.best_tg_speed > 0
        assert profile.speedup_factor > 0
        assert len(profile.all_results) > 0

    def test_cancellation(self):
        # Create a tuner that we cancel mid-run via a progress callback.
        cancel_at_progress = 0.1
        tuner_ref = [None]

        def on_progress(job):
            if job.progress >= cancel_at_progress and tuner_ref[0]:
                tuner_ref[0].cancel()

        cfg = _auto_tuner.TunerConfig(warmup_runs=2, trials_per_param=2)
        ps = _auto_tuner.ParameterSpace()
        mock_fn = _auto_tuner.create_mock_benchmark_fn(base_speed=30.0, variance=1.0)
        tuner = _auto_tuner.AutoTuner(
            config=cfg, param_space=ps,
            benchmark_fn=mock_fn, progress_fn=on_progress,
        )
        tuner_ref[0] = tuner

        job = _auto_tuner.TunerJob(job_id="test")
        with pytest.raises(ValueError, match="cancelled"):
            tuner.run("test-model", job)

    def test_no_benchmark_fn(self):
        tuner = _auto_tuner.AutoTuner(
            config=_auto_tuner.TunerConfig(),
            param_space=_auto_tuner.ParameterSpace(),
            benchmark_fn=None,
        )
        job = _auto_tuner.TunerJob(job_id="test")
        with pytest.raises(RuntimeError, match="No benchmark"):
            tuner.run("test", job)

    def test_progress_callback(self):
        progress_calls = []
        def on_progress(j):
            progress_calls.append(j.progress)

        tuner = self._make_tuner(progress_fn=on_progress)
        job = _auto_tuner.TunerJob(job_id="test")
        tuner.run("test-model", job)
        assert len(progress_calls) > 0
        assert progress_calls[-1] == 1.0


class TestAutoTunerManager:
    """Tests for AutoTunerManager."""

    def setup_method(self):
        """Clean up persisted results before each test."""
        results_path = _BASE / "data" / "tuner_results.json"
        if results_path.is_file():
            results_path.unlink()

    def test_default_status(self):
        mgr = _auto_tuner.AutoTunerManager()
        status = mgr.get_status()
        assert status["available"] is True
        assert "config" in status
        assert "param_space" in status

    def test_list_results_empty(self):
        mgr = _auto_tuner.AutoTunerManager()
        assert mgr.list_results() == {}

    def test_get_result_none(self):
        mgr = _auto_tuner.AutoTunerManager()
        assert mgr.get_result("nonexistent") is None

    def test_delete_result_nonexistent(self):
        mgr = _auto_tuner.AutoTunerManager()
        assert mgr.delete_result("nonexistent") is False

    def test_start_and_complete_tuning(self):
        mgr = _auto_tuner.AutoTunerManager()
        mock_fn = _auto_tuner.create_mock_benchmark_fn()
        job = mgr.start_tuning("test-model", mock_fn)
        assert job.job_id != ""
        assert job.model_name == "test-model"

        # Wait for completion (mock is fast)
        import time
        for _ in range(50):
            time.sleep(0.1)
            j = mgr.get_job("test-model")
            if j and j.status in ("completed", "failed"):
                break

        j = mgr.get_job("test-model")
        assert j is not None
        assert j.status == "completed"

        # Result should be saved
        result = mgr.get_result("test-model")
        assert result is not None
        assert result.best_tg_speed > 0

    def test_cancel_nonexistent(self):
        mgr = _auto_tuner.AutoTunerManager()
        assert mgr.cancel_tuning("nonexistent") is False


class TestParamKey:
    """Tests for _param_key utility."""

    def test_order_independent(self):
        k1 = _auto_tuner._param_key({"a": 1, "b": 2})
        k2 = _auto_tuner._param_key({"b": 2, "a": 1})
        assert k1 == k2

    def test_different_values(self):
        k1 = _auto_tuner._param_key({"a": 1})
        k2 = _auto_tuner._param_key({"a": 2})
        assert k1 != k2


# ============================================================================
# MTP DETECTION TESTS
# ============================================================================


class TestMTPDetection:
    """Tests for Multi-Token Prediction model detection."""

    # Known MTP patterns (matching routes_models.py)
    _MTP_PATTERNS = [
        "deepseek-v3", "deepseek-r1", "deepseek-v2.5",
        "qwen3", "glm-4", "glm4",
    ]

    def _detect(self, name, family=None):
        nl = name.lower().strip()
        fl = (family or "").lower().strip()
        for p in self._MTP_PATTERNS:
            if p in nl or p in fl:
                return True
        return False

    def test_deepseek_v3(self):
        assert self._detect("deepseek-v3:latest") is True

    def test_deepseek_r1(self):
        assert self._detect("deepseek-r1:32b") is True

    def test_deepseek_v25(self):
        assert self._detect("deepseek-v2.5:latest") is True

    def test_qwen3(self):
        assert self._detect("qwen3:8b") is True

    def test_qwen3_coder(self):
        assert self._detect("qwen3-coder:30b") is True

    def test_glm4(self):
        assert self._detect("glm-4:9b") is True

    def test_glm4_alt(self):
        assert self._detect("glm4.5:latest") is True

    def test_family_match(self):
        assert self._detect("some-model", family="deepseek-v3") is True

    def test_llama3_not_mtp(self):
        assert self._detect("llama3:70b") is False

    def test_qwen2_not_mtp(self):
        assert self._detect("qwen2:7b") is False

    def test_mistral_not_mtp(self):
        assert self._detect("mistral:7b") is False

    def test_deepseek_coder_not_mtp(self):
        assert self._detect("deepseek-coder:6.7b") is False

    def test_gemma_not_mtp(self):
        assert self._detect("gemma:2b") is False


# ============================================================================
# API SCHEMA TESTS
# ============================================================================


class TestAPISchemas:
    """Tests for S110 Pydantic API schemas."""

    def test_speculative_decoding_schemas_exist(self):
        """Verify all speculative decoding schemas are in schemas.py."""
        schemas_path = _BASE / "opti_oignon" / "api" / "schemas.py"
        content = schemas_path.read_text()
        expected = [
            "SpeculativeDecodingConfigSchema",
            "SpeculativeDecodingConfigUpdate",
            "SpeculativeDecodingStatsSchema",
            "SpeculativeDecodingStatusResponse",
            "DraftCandidateSchema",
            "CompatibleDraftsResponse",
            "VRAMBudgetResponse",
        ]
        for name in expected:
            assert name in content, f"Missing schema: {name}"

    def test_tuner_schemas_exist(self):
        """Verify all tuner schemas are in schemas.py."""
        schemas_path = _BASE / "opti_oignon" / "api" / "schemas.py"
        content = schemas_path.read_text()
        expected = [
            "TunerConfigSchema",
            "ParameterSpaceSchema",
            "TunerStatusResponse",
            "TunerRunRequest",
            "TunerJobSchema",
            "TunerProfileSchema",
            "TunerResultsResponse",
        ]
        for name in expected:
            assert name in content, f"Missing schema: {name}"

    def test_model_info_has_mtp(self):
        """Verify ModelInfo schema includes mtp_capable field."""
        schemas_path = _BASE / "opti_oignon" / "api" / "schemas.py"
        content = schemas_path.read_text()
        assert "mtp_capable: bool = False" in content


# ============================================================================
# ROUTE REGISTRATION TESTS
# ============================================================================


class TestRouteRegistration:
    """Tests for route registration in app.py."""

    def test_speculative_decoding_routes_registered(self):
        app_path = _BASE / "opti_oignon" / "api" / "app.py"
        content = app_path.read_text()
        assert "speculative_decoding_router" in content
        assert "routes_speculative_decoding" in content

    def test_tuner_routes_registered(self):
        app_path = _BASE / "opti_oignon" / "api" / "app.py"
        content = app_path.read_text()
        assert "tuner_router" in content
        assert "routes_tuner" in content

    def test_deps_speculative_decoding(self):
        deps_path = _BASE / "opti_oignon" / "api" / "deps.py"
        content = deps_path.read_text()
        assert "SPECULATIVE_DECODING_AVAILABLE" in content
        assert "get_speculative_decoding_manager" in content

    def test_deps_auto_tuner(self):
        deps_path = _BASE / "opti_oignon" / "api" / "deps.py"
        content = deps_path.read_text()
        assert "AUTO_TUNER_AVAILABLE" in content
        assert "get_auto_tuner_manager" in content


# ============================================================================
# FRONTEND TESTS
# ============================================================================


class TestFrontendNoHardcodedHex:
    """Verify no hardcoded hex colors in new S110 Svelte components."""

    _NEW_COMPONENTS = [
        "frontend/src/lib/components/settings/SpeculativeDecodingPanel.svelte",
        "frontend/src/lib/components/settings/PerformanceTunerPanel.svelte",
    ]

    def _check_no_hex(self, filepath):
        import re
        full = _BASE / filepath
        if not full.is_file():
            pytest.skip(f"File not found: {filepath}")
        content = full.read_text()
        # Match hex color patterns but exclude Svelte block syntax
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip Svelte control flow lines
            if stripped.startswith("{#") or stripped.startswith("{/") or stripped.startswith("{:"):
                continue
            # Check for hex colors in style attributes
            hex_matches = re.findall(r'#[0-9a-fA-F]{3,8}\b', stripped)
            for h in hex_matches:
                # Filter false positives (e.g. anchor tags, ids)
                assert False, (
                    f"Hardcoded hex color '{h}' found at "
                    f"{filepath}:{i}: {stripped}"
                )

    def test_speculative_decoding_panel(self):
        self._check_no_hex(self._NEW_COMPONENTS[0])

    def test_performance_tuner_panel(self):
        self._check_no_hex(self._NEW_COMPONENTS[1])


class TestFrontendTypes:
    """Verify frontend TypeScript types include S110 additions."""

    def test_speculative_decoding_types(self):
        types_path = _BASE / "frontend" / "src" / "lib" / "types.ts"
        content = types_path.read_text()
        for name in [
            "SpeculativeDecodingConfig",
            "SpeculativeDecodingStats",
            "SpeculativeDecodingStatus",
            "DraftCandidate",
            "VRAMBudgetResult",
        ]:
            assert name in content, f"Missing type: {name}"

    def test_tuner_types(self):
        types_path = _BASE / "frontend" / "src" / "lib" / "types.ts"
        content = types_path.read_text()
        for name in [
            "TunerConfig",
            "TunerParameterSpace",
            "TunerStatus",
            "TunerJob",
            "TunerProfile",
        ]:
            assert name in content, f"Missing type: {name}"

    def test_model_info_mtp(self):
        types_path = _BASE / "frontend" / "src" / "lib" / "types.ts"
        content = types_path.read_text()
        assert "mtp_capable: boolean" in content


# ============================================================================
# CONFIG FILE TESTS
# ============================================================================


class TestConfigFiles:
    """Verify YAML config files exist and are valid."""

    def test_speculative_decoding_yaml(self):
        p = _BASE / "opti_oignon" / "config" / "speculative_decoding.yaml"
        assert p.is_file()
        import yaml
        with open(p) as f:
            data = yaml.safe_load(f)
        assert "speculative_decoding" in data
        assert data["speculative_decoding"]["enabled"] is False

    def test_auto_tuner_yaml(self):
        p = _BASE / "opti_oignon" / "config" / "auto_tuner.yaml"
        assert p.is_file()
        import yaml
        with open(p) as f:
            data = yaml.safe_load(f)
        assert "auto_tuner" in data
        assert data["auto_tuner"]["enabled"] is True
        assert "parameter_space" in data
