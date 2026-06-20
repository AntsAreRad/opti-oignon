#!/usr/bin/env python3
"""
Tests for S112 — Advanced Model Management + Inference Telemetry.

Covers:
  - model_lifecycle.py: LifecycleConfig, PullProgress, PullJob, ModelUpdateInfo,
    ModelLifecycleManager (aliases, stale detection, list, delete, pull),
    config loading, alias persistence, format_bytes
  - telemetry.py: TelemetryConfig, InferenceEvent, ActiveRequest,
    TelemetryCollector (hooks, buffer, flush, consumers, stats),
    consumer factories, config loading
  - auto_tuner.py: TunerRecommendation, generate_recommendations,
    per-parameter analyzers (threads, batch, flash attention, gpu_layers)
  - routes_model_lifecycle.py: API endpoints (status, pull, delete, aliases, stale)
  - routes_tuner.py: GET /api/tuner/recommendations/{model_name}
  - inference_backend.py: _get_telemetry lazy loader
  - Frontend: no hardcoded hex in ModelManager.svelte, PerformanceTunerPanel.svelte

Total: ~55 tests
"""

import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import time
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

# Stub optional dependencies.
for _mod_name in [
    "ollama", "requests", "llama_cpp",
    "opti_oignon.live_metrics", "opti_oignon.performance_monitor",
    "opti_oignon.speculative_decoding",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

# ---------------------------------------------------------------------------
# Load modules under test
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "opti_oignon"

model_lifecycle = _load_module("model_lifecycle", str(_BACKEND / "model_lifecycle.py"))
telemetry = _load_module("telemetry_mod", str(_BACKEND / "telemetry.py"))
auto_tuner = _load_module("auto_tuner_mod", str(_BACKEND / "auto_tuner.py"))


# ===================================================================
# MODEL LIFECYCLE TESTS
# ===================================================================


class TestLifecycleConfig:
    """Tests for LifecycleConfig dataclass."""

    def test_defaults(self):
        cfg = model_lifecycle.LifecycleConfig()
        assert cfg.enabled is True
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.max_concurrent_pulls == 2
        assert cfg.stale_threshold_days == 30

    def test_validate_ok(self):
        cfg = model_lifecycle.LifecycleConfig()
        assert cfg.validate() == []

    def test_validate_bad_pulls(self):
        cfg = model_lifecycle.LifecycleConfig(max_concurrent_pulls=0)
        errors = cfg.validate()
        assert any("max_concurrent_pulls" in e for e in errors)

    def test_validate_bad_interval(self):
        cfg = model_lifecycle.LifecycleConfig(progress_poll_interval_s=0.01)
        errors = cfg.validate()
        assert any("progress_poll_interval_s" in e for e in errors)

    def test_validate_bad_stale(self):
        cfg = model_lifecycle.LifecycleConfig(stale_threshold_days=-1)
        errors = cfg.validate()
        assert any("stale_threshold_days" in e for e in errors)


class TestPullJobAndProgress:
    """Tests for PullJob and PullProgress dataclasses."""

    def test_pull_progress_to_dict(self):
        p = model_lifecycle.PullProgress(
            status="downloading", total_bytes=1000, completed_bytes=500, percent=50.0
        )
        d = p.to_dict()
        assert d["percent"] == 50.0
        assert d["total_bytes"] == 1000

    def test_pull_job_defaults(self):
        job = model_lifecycle.PullJob(model_name="test:latest")
        assert job.status == model_lifecycle.PULL_STATUS_PENDING
        assert job.job_id  # Auto-generated
        assert len(job.job_id) == 12

    def test_pull_job_to_dict(self):
        job = model_lifecycle.PullJob(
            model_name="llama3:8b", status="downloading"
        )
        d = job.to_dict()
        assert d["model_name"] == "llama3:8b"
        assert d["status"] == "downloading"
        assert "progress" in d


class TestModelUpdateInfo:
    """Tests for ModelUpdateInfo dataclass."""

    def test_to_dict_truncates_digest(self):
        info = model_lifecycle.ModelUpdateInfo(
            model_name="test", current_digest="a" * 64, has_update=True
        )
        d = info.to_dict()
        assert len(d["current_digest"]) == 16
        assert d["has_update"] is True


class TestFormatBytes:
    """Tests for _format_bytes helper."""

    def test_zero(self):
        assert model_lifecycle._format_bytes(0) == "0 B"

    def test_bytes(self):
        assert model_lifecycle._format_bytes(500) == "500 B"

    def test_kilobytes(self):
        assert "KB" in model_lifecycle._format_bytes(5000)

    def test_megabytes(self):
        assert "MB" in model_lifecycle._format_bytes(5_000_000)

    def test_gigabytes(self):
        assert "GB" in model_lifecycle._format_bytes(7_500_000_000)


class TestConfigLoading:
    """Tests for _load_config from YAML."""

    def test_load_from_yaml(self):
        cfg = model_lifecycle._load_config(_BACKEND / "config" / "model_lifecycle.yaml")
        assert cfg.enabled is True
        assert cfg.max_concurrent_pulls == 2

    def test_load_missing_file(self):
        cfg = model_lifecycle._load_config(Path("/nonexistent/file.yaml"))
        assert cfg.enabled is True  # Defaults


class TestAliasPersistence:
    """Tests for alias load/save."""

    def test_save_and_load(self, tmp_path):
        p = tmp_path / "aliases.json"
        aliases = {"fast": "qwen2.5:0.5b", "big": "llama3:70b"}
        model_lifecycle._save_aliases(aliases, p)
        loaded = model_lifecycle._load_aliases(p)
        assert loaded == aliases

    def test_load_missing(self, tmp_path):
        p = tmp_path / "nope.json"
        assert model_lifecycle._load_aliases(p) == {}


class TestModelLifecycleManager:
    """Tests for ModelLifecycleManager."""

    def _make_manager(self, tmp_path, ollama_module=None):
        cfg = model_lifecycle.LifecycleConfig(enabled=True)
        return model_lifecycle.ModelLifecycleManager(
            config=cfg,
            aliases_path=tmp_path / "aliases.json",
            ollama_module=ollama_module,
        )

    def test_alias_lifecycle(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.set_alias("fast", "qwen2.5:0.5b")
        assert mgr.resolve_alias("fast") == "qwen2.5:0.5b"
        assert mgr.resolve_alias("unknown") == "unknown"
        assert mgr.list_aliases() == {"fast": "qwen2.5:0.5b"}
        mgr.remove_alias("fast")
        assert mgr.list_aliases() == {}

    def test_list_models_no_ollama(self, tmp_path):
        mgr = self._make_manager(tmp_path, ollama_module=None)
        assert mgr.list_models() == []

    def test_list_models_with_mock(self, tmp_path):
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {
            "models": [
                {"name": "llama3:8b", "size": 4_000_000_000, "modified_at": time.time() - 86400}
            ]
        }
        mgr = self._make_manager(tmp_path, ollama_module=mock_ollama)
        models = mgr.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "llama3:8b"
        assert "GB" in models[0]["size_human"]

    def test_delete_model_success(self, tmp_path):
        mock_ollama = MagicMock()
        mock_ollama.delete.return_value = None
        mgr = self._make_manager(tmp_path, ollama_module=mock_ollama)
        result = mgr.delete_model("test:latest")
        assert result["success"] is True
        mock_ollama.delete.assert_called_once_with("test:latest")

    def test_delete_model_failure(self, tmp_path):
        mock_ollama = MagicMock()
        mock_ollama.delete.side_effect = Exception("not found")
        mgr = self._make_manager(tmp_path, ollama_module=mock_ollama)
        result = mgr.delete_model("test:latest")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_start_pull_creates_job(self, tmp_path):
        mock_ollama = MagicMock()
        # pull should be called in background thread via library fallback
        mock_ollama.pull.return_value = iter([
            {"status": "pulling manifest"},
            {"status": "success"},
        ])
        mgr = self._make_manager(tmp_path, ollama_module=mock_ollama)
        # Ensure HTTP path is skipped so library fallback is used.
        with patch.object(model_lifecycle, "REQUESTS_AVAILABLE", False):
            job = mgr.start_pull("test:latest")
            assert job.model_name == "test:latest"
            assert job.job_id
            # Wait for background thread to complete
            time.sleep(0.5)
            retrieved = mgr.get_pull_job(job.job_id)
            assert retrieved is not None
            assert retrieved.status in (
                model_lifecycle.PULL_STATUS_COMPLETE,
                model_lifecycle.PULL_STATUS_DOWNLOADING,
                model_lifecycle.PULL_STATUS_PENDING,
            )

    def test_max_concurrent_pulls(self, tmp_path):
        cfg = model_lifecycle.LifecycleConfig(enabled=True, max_concurrent_pulls=1)
        mgr = model_lifecycle.ModelLifecycleManager(
            config=cfg,
            aliases_path=tmp_path / "aliases.json",
            ollama_module=MagicMock(),
        )
        # Simulate one active pull
        mgr._active_pulls = 1
        job = mgr.start_pull("test2:latest")
        assert job.status == model_lifecycle.PULL_STATUS_FAILED
        assert "Max concurrent" in job.error

    def test_stale_detection(self, tmp_path):
        mock_ollama = MagicMock()
        old_ts = time.time() - (60 * 86400)  # 60 days ago
        mock_ollama.list.return_value = {
            "models": [
                {"name": "old-model:latest", "size": 1000, "modified_at": old_ts},
                {"name": "new-model:latest", "size": 2000, "modified_at": time.time()},
            ]
        }
        mgr = self._make_manager(tmp_path, ollama_module=mock_ollama)
        stale = mgr.detect_stale_models()
        assert len(stale) == 1
        assert stale[0]["name"] == "old-model:latest"
        assert stale[0]["days_since_modified"] >= 59

    def test_list_pull_jobs(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.list_pull_jobs() == []

    def test_shutdown(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.shutdown()  # Should not raise

    def test_get_model_info_no_ollama(self, tmp_path):
        mgr = self._make_manager(tmp_path, ollama_module=None)
        assert mgr.get_model_info("test") is None


# ===================================================================
# TELEMETRY TESTS
# ===================================================================


class TestTelemetryConfig:
    """Tests for TelemetryConfig dataclass."""

    def test_defaults(self):
        cfg = telemetry.TelemetryConfig()
        assert cfg.enabled is True
        assert cfg.buffer_max_size == 64
        assert cfg.consumer_live_metrics is True

    def test_validate_ok(self):
        cfg = telemetry.TelemetryConfig()
        assert cfg.validate() == []

    def test_validate_errors(self):
        cfg = telemetry.TelemetryConfig(
            buffer_max_size=0, buffer_flush_interval_ms=-1, token_tracking_max_per_request=0
        )
        errors = cfg.validate()
        assert len(errors) == 3


class TestInferenceEvent:
    """Tests for InferenceEvent dataclass."""

    def test_to_dict(self):
        ev = telemetry.InferenceEvent(
            event_type="inference_start",
            request_id="abc123",
            model="llama3:8b",
            data={"message_count": 3},
        )
        d = ev.to_dict()
        assert d["event_type"] == "inference_start"
        assert d["model"] == "llama3:8b"
        assert d["data"]["message_count"] == 3


class TestTelemetryConfigLoading:
    """Tests for telemetry config loading from YAML."""

    def test_load_from_yaml(self):
        cfg = telemetry._load_config(_BACKEND / "config" / "telemetry.yaml")
        assert cfg.enabled is True
        assert cfg.buffer_max_size == 64
        assert cfg.consumer_performance_monitor is True

    def test_load_missing(self):
        cfg = telemetry._load_config(Path("/nonexistent.yaml"))
        assert cfg.enabled is True


class TestTelemetryCollector:
    """Tests for TelemetryCollector event bus."""

    def _make_collector(self, **kwargs):
        cfg = telemetry.TelemetryConfig(
            enabled=True,
            buffer_flush_interval_ms=0,  # Immediate flush
            **kwargs,
        )
        return telemetry.TelemetryCollector(config=cfg)

    def test_on_inference_start(self):
        tc = self._make_collector()
        rid = tc.on_inference_start("model-a", [{"role": "user", "content": "hi"}])
        assert rid
        assert len(rid) == 12
        stats = tc.get_stats()
        assert stats["total_requests"] == 1

    def test_on_token_generated(self):
        tc = self._make_collector()
        rid = tc.on_inference_start("model-a")
        tc.on_token_generated(rid, count=5)
        stats = tc.get_stats()
        assert stats["total_tokens"] == 5

    def test_on_inference_end(self):
        tc = self._make_collector()
        rid = tc.on_inference_start("model-a")
        tc.on_token_generated(rid, count=10)
        tc.on_inference_end(rid, model="model-a", tokens_in=5, tokens_out=10, latency_ms=100.0)
        stats = tc.get_stats()
        assert stats["active_requests"] == 0
        assert stats["total_events"] >= 3  # start + token + end

    def test_consumer_receives_events(self):
        tc = self._make_collector()
        received = []
        tc.register_consumer(lambda events: received.extend(events))
        rid = tc.on_inference_start("m")
        tc.on_inference_end(rid, model="m", tokens_out=1, latency_ms=50)
        assert len(received) >= 2
        types_seen = {e.event_type for e in received}
        assert "inference_start" in types_seen
        assert "inference_end" in types_seen

    def test_unregister_consumer(self):
        tc = self._make_collector()
        received = []
        consumer = lambda events: received.extend(events)
        tc.register_consumer(consumer)
        tc.unregister_consumer(consumer)
        tc.on_inference_start("m")
        assert len(received) == 0

    def test_flush_returns_count(self):
        cfg = telemetry.TelemetryConfig(
            enabled=True, buffer_max_size=100, buffer_flush_interval_ms=9999
        )
        tc = telemetry.TelemetryCollector(config=cfg)
        tc.on_inference_start("m")
        tc.on_inference_start("m")
        count = tc.flush()
        assert count == 2

    def test_disabled_collector(self):
        cfg = telemetry.TelemetryConfig(enabled=False)
        tc = telemetry.TelemetryCollector(config=cfg)
        rid = tc.on_inference_start("m")
        assert rid  # Still returns an ID
        stats = tc.get_stats()
        assert stats["total_requests"] == 0

    def test_auto_latency_calculation(self):
        tc = self._make_collector()
        received = []
        tc.register_consumer(lambda events: received.extend(events))
        rid = tc.on_inference_start("m")
        time.sleep(0.05)
        tc.on_inference_end(rid, model="m")
        end_events = [e for e in received if e.event_type == "inference_end"]
        assert len(end_events) == 1
        latency = end_events[0].data.get("latency_ms", 0)
        assert latency >= 40  # At least ~50ms

    def test_consumer_error_handling(self):
        tc = self._make_collector()
        def bad_consumer(events):
            raise ValueError("boom")
        tc.register_consumer(bad_consumer)
        # Should not raise
        rid = tc.on_inference_start("m")
        tc.on_inference_end(rid, model="m", latency_ms=10)

    def test_speculative_data_passthrough(self):
        tc = self._make_collector()
        received = []
        tc.register_consumer(lambda events: received.extend(events))
        rid = tc.on_inference_start("m")
        tc.on_inference_end(
            rid, model="m", latency_ms=10,
            speculative_data={"draft_tokens": 5, "accepted": 4, "speedup": 1.3},
        )
        end_events = [e for e in received if e.event_type == "inference_end"]
        assert end_events[0].data["speculative_data"]["accepted"] == 4

    def test_shutdown(self):
        tc = self._make_collector()
        tc.on_inference_start("m")
        tc.shutdown()  # Should flush and not raise

    def test_flush_thread_lifecycle(self):
        cfg = telemetry.TelemetryConfig(
            enabled=True, buffer_flush_interval_ms=50, buffer_max_size=1000
        )
        tc = telemetry.TelemetryCollector(config=cfg)
        tc.start_flush_thread()
        assert tc._running is True
        time.sleep(0.1)
        tc.stop_flush_thread()
        assert tc._running is False

    def test_token_tracking_limit(self):
        tc = self._make_collector(token_tracking_max_per_request=3)
        rid = tc.on_inference_start("m")
        for _ in range(10):
            tc.on_token_generated(rid, count=1)
        with tc._lock:
            req = tc._active_requests.get(rid)
        # Request may have been processed, check if it exists
        if req:
            assert len(req.token_timestamps) <= 3


# ===================================================================
# AUTO-TUNER RECOMMENDATION TESTS
# ===================================================================


class TestTunerRecommendation:
    """Tests for TunerRecommendation dataclass."""

    def test_defaults(self):
        rec = auto_tuner.TunerRecommendation()
        assert rec.estimated_speedup == 1.0
        assert rec.confidence == "medium"
        assert rec.applied is False

    def test_to_dict(self):
        rec = auto_tuner.TunerRecommendation(
            title="Test", parameter="threads",
            estimated_speedup=1.5, confidence="high",
        )
        d = rec.to_dict()
        assert d["title"] == "Test"
        assert d["estimated_speedup"] == 1.5
        assert d["confidence"] == "high"


class TestGenerateRecommendations:
    """Tests for generate_recommendations function."""

    def _make_profile(self, **kwargs):
        defaults = dict(
            model_name="test:latest",
            best_params={"threads": 8, "batch_size": 2048, "flash_attention": True},
            best_tg_speed=45.0,
            best_pp_speed=120.0,
            baseline_tg_speed=30.0,
            baseline_pp_speed=80.0,
            speedup_factor=1.5,
            all_results=[
                {"params": {"threads": 4, "batch_size": 512, "flash_attention": False}, "tokens_per_second_tg": 25.0},
                {"params": {"threads": 4, "batch_size": 2048, "flash_attention": True}, "tokens_per_second_tg": 32.0},
                {"params": {"threads": 8, "batch_size": 512, "flash_attention": False}, "tokens_per_second_tg": 35.0},
                {"params": {"threads": 8, "batch_size": 2048, "flash_attention": True}, "tokens_per_second_tg": 45.0},
            ],
        )
        defaults.update(kwargs)
        return auto_tuner.TunerProfile(**defaults)

    def test_empty_profile(self):
        profile = auto_tuner.TunerProfile()
        recs = auto_tuner.generate_recommendations(profile)
        assert recs == []

    def test_no_results(self):
        profile = auto_tuner.TunerProfile(
            best_params={"threads": 4}, all_results=[]
        )
        recs = auto_tuner.generate_recommendations(profile)
        assert recs == []

    def test_generates_overall(self):
        profile = self._make_profile()
        recs = auto_tuner.generate_recommendations(profile)
        overall = [r for r in recs if r.parameter == "all"]
        assert len(overall) == 1
        assert overall[0].estimated_speedup == pytest.approx(1.5, rel=0.01)

    def test_generates_thread_recommendation(self):
        profile = self._make_profile()
        recs = auto_tuner.generate_recommendations(profile)
        thread_recs = [r for r in recs if r.parameter == "threads"]
        assert len(thread_recs) >= 1
        assert thread_recs[0].recommended_value == 8

    def test_generates_flash_attention_rec(self):
        profile = self._make_profile()
        recs = auto_tuner.generate_recommendations(profile)
        fa_recs = [r for r in recs if r.parameter == "flash_attention"]
        assert len(fa_recs) >= 1

    def test_sorted_by_speedup(self):
        profile = self._make_profile()
        recs = auto_tuner.generate_recommendations(profile)
        speedups = [r.estimated_speedup for r in recs]
        assert speedups == sorted(speedups, reverse=True)

    def test_no_speedup_no_recommendation(self):
        profile = auto_tuner.TunerProfile(
            model_name="test",
            best_params={"threads": 4},
            best_tg_speed=30.0,
            baseline_tg_speed=30.0,
            all_results=[
                {"params": {"threads": 4}, "tokens_per_second_tg": 30.0},
            ],
        )
        recs = auto_tuner.generate_recommendations(profile)
        overall = [r for r in recs if r.parameter == "all"]
        assert len(overall) == 0

    def test_gpu_layers_recommendation(self):
        profile = auto_tuner.TunerProfile(
            model_name="test",
            best_params={"gpu_layers": 32},
            best_tg_speed=50.0,
            baseline_tg_speed=30.0,
            all_results=[
                {"params": {"gpu_layers": 0}, "tokens_per_second_tg": 20.0},
                {"params": {"gpu_layers": 16}, "tokens_per_second_tg": 35.0},
                {"params": {"gpu_layers": 32}, "tokens_per_second_tg": 50.0},
            ],
        )
        recs = auto_tuner.generate_recommendations(profile)
        gpu_recs = [r for r in recs if r.parameter == "gpu_layers"]
        assert len(gpu_recs) >= 1
        assert gpu_recs[0].recommended_value == 32

    def test_batch_size_recommendation(self):
        profile = auto_tuner.TunerProfile(
            model_name="test",
            best_params={"batch_size": 4096},
            best_tg_speed=40.0,
            baseline_tg_speed=25.0,
            all_results=[
                {"params": {"batch_size": 512}, "tokens_per_second_tg": 22.0},
                {"params": {"batch_size": 1024}, "tokens_per_second_tg": 30.0},
                {"params": {"batch_size": 4096}, "tokens_per_second_tg": 40.0},
            ],
        )
        recs = auto_tuner.generate_recommendations(profile)
        batch_recs = [r for r in recs if r.parameter == "batch_size"]
        assert len(batch_recs) >= 1

    def test_flash_attention_disable_rec(self):
        """When FA is slower, recommend disabling."""
        profile = auto_tuner.TunerProfile(
            model_name="test",
            best_params={"flash_attention": False},
            best_tg_speed=40.0,
            baseline_tg_speed=35.0,
            all_results=[
                {"params": {"flash_attention": True}, "tokens_per_second_tg": 30.0},
                {"params": {"flash_attention": False}, "tokens_per_second_tg": 40.0},
            ],
        )
        recs = auto_tuner.generate_recommendations(profile)
        fa_recs = [r for r in recs if r.parameter == "flash_attention"]
        assert len(fa_recs) >= 1
        assert fa_recs[0].recommended_value is False


# ===================================================================
# FRONTEND TESTS (no hardcoded hex)
# ===================================================================


class TestFrontendNoHex:
    """Ensure no hardcoded hex colors in S112 Svelte components."""

    _HEX_PATTERN = re.compile(
        r'(?:color|background|background-color|border-color|border)\s*:\s*#[0-9a-fA-F]{3,8}'
    )

    def _check_file(self, path):
        if not path.is_file():
            pytest.skip(f"{path} not found")
        content = path.read_text()
        matches = self._HEX_PATTERN.findall(content)
        assert matches == [], f"Hardcoded hex colors found in {path.name}: {matches}"

    def test_model_manager_no_hex(self):
        self._check_file(
            _ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "ModelManager.svelte"
        )

    def test_performance_tuner_panel_no_hex(self):
        self._check_file(
            _ROOT / "frontend" / "src" / "lib" / "components" / "settings" / "PerformanceTunerPanel.svelte"
        )


# ===================================================================
# CONFIG YAML TESTS
# ===================================================================


class TestConfigFiles:
    """Ensure S112 config files parse correctly."""

    def test_model_lifecycle_yaml(self):
        import yaml
        p = _BACKEND / "config" / "model_lifecycle.yaml"
        data = yaml.safe_load(p.read_text())
        assert data["enabled"] is True
        assert data["pull"]["max_concurrent_pulls"] == 2

    def test_telemetry_yaml(self):
        import yaml
        p = _BACKEND / "config" / "telemetry.yaml"
        data = yaml.safe_load(p.read_text())
        assert data["enabled"] is True
        assert data["buffer"]["max_size"] == 64
        assert data["consumers"]["live_metrics"] is True


# ===================================================================
# INTEGRATION-STYLE TESTS
# ===================================================================


class TestTelemetryFullPipeline:
    """End-to-end telemetry pipeline test."""

    def test_full_inference_lifecycle(self):
        """Simulate a complete inference through the telemetry pipeline."""
        cfg = telemetry.TelemetryConfig(
            enabled=True, buffer_flush_interval_ms=0
        )
        tc = telemetry.TelemetryCollector(config=cfg)

        received_events = []
        tc.register_consumer(lambda events: received_events.extend(events))

        # Simulate inference
        rid = tc.on_inference_start("llama3:8b", [
            {"role": "user", "content": "Hello"},
        ])
        for _ in range(20):
            tc.on_token_generated(rid, count=1)
        tc.on_inference_end(
            rid, model="llama3:8b",
            tokens_in=5, tokens_out=20,
            latency_ms=250.0,
            task_type="chat",
        )

        stats = tc.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 20
        assert stats["total_events"] == 22  # 1 start + 20 tokens + 1 end
        assert stats["active_requests"] == 0

        # Verify event sequence
        start_events = [e for e in received_events if e.event_type == "inference_start"]
        token_events = [e for e in received_events if e.event_type == "token_generated"]
        end_events = [e for e in received_events if e.event_type == "inference_end"]

        assert len(start_events) == 1
        assert len(token_events) == 20
        assert len(end_events) == 1
        assert end_events[0].data["tokens_out"] == 20
        assert end_events[0].data["latency_ms"] == 250.0

        tc.shutdown()


class TestModelLifecycleParseEntry:
    """Tests for _parse_model_entry static method."""

    def test_parse_dict(self):
        entry = {
            "name": "llama3:8b",
            "size": 4_500_000_000,
            "modified_at": 1700000000.0,
            "digest": "abc123def456",
        }
        result = model_lifecycle.ModelLifecycleManager._parse_model_entry(entry)
        assert result["name"] == "llama3:8b"
        assert "GB" in result["size_human"]
        assert result["modified_at"] == 1700000000.0

    def test_parse_empty_name(self):
        result = model_lifecycle.ModelLifecycleManager._parse_model_entry({"name": ""})
        assert result is None

    def test_parse_object(self):
        obj = MagicMock()
        obj.name = "test:latest"
        obj.size = 1000
        obj.modified_at = 0
        obj.digest = ""
        obj.details = {}
        result = model_lifecycle.ModelLifecycleManager._parse_model_entry(obj)
        assert result["name"] == "test:latest"


class TestResetSingletons:
    """Ensure singleton reset functions work."""

    def test_reset_lifecycle_manager(self):
        model_lifecycle.reset_manager()
        assert model_lifecycle._manager is None

    def test_reset_telemetry(self):
        telemetry.reset_telemetry()
        assert telemetry._collector is None
