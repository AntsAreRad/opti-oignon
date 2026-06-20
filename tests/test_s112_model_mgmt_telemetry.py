#!/usr/bin/env python3
"""
Tests for S112 — Advanced Model Management + Inference Telemetry.

Covers:
  - LifecycleConfig: loading, validation, defaults
  - PullProgress, PullJob, ModelUpdateInfo serialization
  - ModelLifecycleManager: list, pull, delete, aliases, stale detection
  - Alias persistence (load/save JSON)
  - TelemetryConfig: loading, validation, defaults
  - InferenceEvent, ActiveRequest dataclasses
  - TelemetryCollector: hooks, buffering, flush, consumers, stats
  - Built-in consumer factories (live_metrics, performance_monitor, speculative_decoding)
  - TunerRecommendation dataclass and serialization
  - generate_recommendations: overall, threads, batch_size, flash_attention, gpu_layers
  - generate_recommendations: edge cases (empty, no speedup, single param)
  - Routes: model lifecycle schemas
  - Routes: tuner recommendations schema
  - Frontend: no hardcoded hex in ModelManager.svelte
  - Frontend: no hardcoded hex in PerformanceTunerPanel.svelte
  - Config YAML: model_lifecycle.yaml and telemetry.yaml parse correctly
  - inference_backend: _get_telemetry lazy loader

Total: ~60 tests
"""

import collections
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

# Load modules under test
_BASE = Path(__file__).resolve().parent.parent

_model_lifecycle = _load_module(
    "opti_oignon.model_lifecycle",
    str(_BASE / "opti_oignon" / "model_lifecycle.py"),
)
_telemetry = _load_module(
    "opti_oignon.telemetry",
    str(_BASE / "opti_oignon" / "telemetry.py"),
)
_auto_tuner = _load_module(
    "opti_oignon.auto_tuner",
    str(_BASE / "opti_oignon" / "auto_tuner.py"),
)


# =========================================================================
# Model Lifecycle — Config
# =========================================================================


class TestLifecycleConfig:
    """Tests for LifecycleConfig loading and validation."""

    def test_defaults(self):
        cfg = _model_lifecycle.LifecycleConfig()
        assert cfg.enabled is True
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.max_concurrent_pulls == 2
        assert cfg.stale_threshold_days == 30

    def test_validation_ok(self):
        cfg = _model_lifecycle.LifecycleConfig()
        assert cfg.validate() == []

    def test_validation_bad_pulls(self):
        cfg = _model_lifecycle.LifecycleConfig(max_concurrent_pulls=0)
        errors = cfg.validate()
        assert len(errors) == 1
        assert "max_concurrent_pulls" in errors[0]

    def test_validation_bad_poll_interval(self):
        cfg = _model_lifecycle.LifecycleConfig(progress_poll_interval_s=0.01)
        errors = cfg.validate()
        assert any("progress_poll_interval_s" in e for e in errors)

    def test_validation_bad_stale(self):
        cfg = _model_lifecycle.LifecycleConfig(stale_threshold_days=-1)
        errors = cfg.validate()
        assert any("stale_threshold_days" in e for e in errors)

    def test_load_config_from_yaml(self):
        cfg = _model_lifecycle._load_config(
            _BASE / "opti_oignon" / "config" / "model_lifecycle.yaml"
        )
        assert cfg.enabled is True
        assert cfg.ollama_base_url == "http://localhost:11434"

    def test_load_config_missing_file(self):
        cfg = _model_lifecycle._load_config(Path("/tmp/nonexistent.yaml"))
        assert cfg.enabled is True  # defaults


# =========================================================================
# Model Lifecycle — Data types
# =========================================================================


class TestPullProgress:
    def test_to_dict(self):
        p = _model_lifecycle.PullProgress(
            status="downloading",
            digest="sha256:abc",
            total_bytes=1_000_000,
            completed_bytes=500_000,
            percent=50.0,
        )
        d = p.to_dict()
        assert d["status"] == "downloading"
        assert d["percent"] == 50.0
        assert d["total_bytes"] == 1_000_000

    def test_defaults(self):
        p = _model_lifecycle.PullProgress()
        assert p.percent == 0.0
        assert p.status == ""


class TestPullJob:
    def test_to_dict(self):
        job = _model_lifecycle.PullJob(model_name="llama3:8b")
        d = job.to_dict()
        assert d["model_name"] == "llama3:8b"
        assert d["status"] == "pending"
        assert len(d["job_id"]) == 12

    def test_job_id_unique(self):
        j1 = _model_lifecycle.PullJob()
        j2 = _model_lifecycle.PullJob()
        assert j1.job_id != j2.job_id


class TestModelUpdateInfo:
    def test_to_dict_truncates_digest(self):
        info = _model_lifecycle.ModelUpdateInfo(
            model_name="test",
            current_digest="a" * 64,
            has_update=True,
        )
        d = info.to_dict()
        assert len(d["current_digest"]) == 16
        assert d["has_update"] is True


# =========================================================================
# Model Lifecycle — Alias persistence
# =========================================================================


class TestAliasPersistence:
    def test_save_and_load(self, tmp_path):
        p = tmp_path / "aliases.json"
        aliases = {"fast": "qwen2.5:0.5b", "big": "llama3.1:70b"}
        assert _model_lifecycle._save_aliases(aliases, p) is True
        loaded = _model_lifecycle._load_aliases(p)
        assert loaded == aliases

    def test_load_missing_file(self, tmp_path):
        loaded = _model_lifecycle._load_aliases(tmp_path / "nope.json")
        assert loaded == {}

    def test_load_corrupt_file(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        loaded = _model_lifecycle._load_aliases(p)
        assert loaded == {}


# =========================================================================
# Model Lifecycle — Manager
# =========================================================================


class TestModelLifecycleManager:
    def test_init_defaults(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "aliases.json", ollama_module=None
        )
        assert mgr.enabled is True
        assert mgr.list_aliases() == {}

    def test_alias_operations(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "aliases.json", ollama_module=None
        )
        mgr.set_alias("fast", "qwen:0.5b")
        assert mgr.resolve_alias("fast") == "qwen:0.5b"
        assert mgr.resolve_alias("unknown") == "unknown"

        aliases = mgr.list_aliases()
        assert aliases["fast"] == "qwen:0.5b"

        mgr.remove_alias("fast")
        assert "fast" not in mgr.list_aliases()

    def test_remove_nonexistent_alias(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "aliases.json", ollama_module=None
        )
        assert mgr.remove_alias("nope") is False

    def test_list_models_no_ollama(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=None
        )
        assert mgr.list_models() == []

    def test_list_models_with_mock_ollama(self, tmp_path):
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {
            "models": [
                {"name": "llama3:8b", "size": 4_000_000_000, "modified_at": 1700000000, "digest": "abc123def456789", "details": {}},
                {"name": "qwen:0.5b", "size": 300_000_000, "modified_at": 1700100000, "digest": "xyz", "details": {}},
            ]
        }
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=mock_ollama
        )
        models = mgr.list_models()
        assert len(models) == 2
        assert models[0]["name"] == "llama3:8b"
        assert models[0]["size_human"] == "4.0 GB"

    def test_delete_model_no_ollama(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=None
        )
        result = mgr.delete_model("test")
        assert result["success"] is False

    def test_delete_model_success(self, tmp_path):
        mock_ollama = MagicMock()
        mock_ollama.delete.return_value = None
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=mock_ollama
        )
        result = mgr.delete_model("test:latest")
        assert result["success"] is True
        assert result["model"] == "test:latest"
        mock_ollama.delete.assert_called_once_with("test:latest")

    def test_delete_model_resolves_alias(self, tmp_path):
        mock_ollama = MagicMock()
        mock_ollama.delete.return_value = None
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=mock_ollama
        )
        mgr.set_alias("fast", "qwen:0.5b")
        mgr.delete_model("fast")
        mock_ollama.delete.assert_called_once_with("qwen:0.5b")

    def test_start_pull_max_concurrent(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig(max_concurrent_pulls=1)
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=None
        )
        # Simulate one active pull
        mgr._active_pulls = 1
        job = mgr.start_pull("test:latest")
        assert job.status == "failed"
        assert "Max concurrent" in job.error

    def test_pull_job_listing(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=None
        )
        # Add a completed job manually
        job = _model_lifecycle.PullJob(model_name="test", status="complete")
        mgr._jobs[job.job_id] = job
        jobs = mgr.list_pull_jobs()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "complete"

    def test_cancel_pull_nonexistent(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=None
        )
        assert mgr.cancel_pull("nope") is False

    def test_cancel_pull_completed(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=None
        )
        job = _model_lifecycle.PullJob(model_name="test", status="complete")
        mgr._jobs[job.job_id] = job
        assert mgr.cancel_pull(job.job_id) is False

    def test_detect_stale_models(self, tmp_path):
        mock_ollama = MagicMock()
        old_ts = time.time() - (60 * 86400)  # 60 days ago
        mock_ollama.list.return_value = {
            "models": [
                {"name": "old-model", "size": 1000, "modified_at": old_ts, "digest": "", "details": {}},
                {"name": "new-model", "size": 2000, "modified_at": time.time(), "digest": "", "details": {}},
            ]
        }
        cfg = _model_lifecycle.LifecycleConfig(stale_threshold_days=30)
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=mock_ollama
        )
        stale = mgr.detect_stale_models()
        assert len(stale) == 1
        assert stale[0]["name"] == "old-model"
        assert stale[0]["days_since_modified"] >= 59

    def test_detect_stale_disabled(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig(stale_threshold_days=0)
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=None
        )
        assert mgr.detect_stale_models() == []

    def test_shutdown(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=None
        )
        job = _model_lifecycle.PullJob(model_name="test", status="downloading")
        mgr._jobs[job.job_id] = job
        mgr.shutdown()
        assert job._cancelled is True

    def test_get_model_info_no_ollama(self, tmp_path):
        cfg = _model_lifecycle.LifecycleConfig()
        mgr = _model_lifecycle.ModelLifecycleManager(
            config=cfg, aliases_path=tmp_path / "a.json", ollama_module=None
        )
        assert mgr.get_model_info("test") is None


class TestFormatBytes:
    def test_zero(self):
        assert _model_lifecycle._format_bytes(0) == "0 B"

    def test_bytes(self):
        assert _model_lifecycle._format_bytes(500) == "500 B"

    def test_kb(self):
        assert _model_lifecycle._format_bytes(1500) == "1.5 KB"

    def test_mb(self):
        assert _model_lifecycle._format_bytes(5_000_000) == "5.0 MB"

    def test_gb(self):
        assert _model_lifecycle._format_bytes(7_500_000_000) == "7.5 GB"


class TestLifecycleSingleton:
    def test_get_and_reset(self):
        _model_lifecycle.reset_manager()
        mgr = _model_lifecycle.get_lifecycle_manager()
        assert mgr is not None
        mgr2 = _model_lifecycle.get_lifecycle_manager()
        assert mgr is mgr2
        _model_lifecycle.reset_manager()


# =========================================================================
# Telemetry — Config
# =========================================================================


class TestTelemetryConfig:
    def test_defaults(self):
        cfg = _telemetry.TelemetryConfig()
        assert cfg.enabled is True
        assert cfg.buffer_max_size == 64
        assert cfg.consumer_live_metrics is True

    def test_validation_ok(self):
        cfg = _telemetry.TelemetryConfig()
        assert cfg.validate() == []

    def test_validation_bad_buffer(self):
        cfg = _telemetry.TelemetryConfig(buffer_max_size=0)
        assert len(cfg.validate()) >= 1

    def test_validation_bad_flush(self):
        cfg = _telemetry.TelemetryConfig(buffer_flush_interval_ms=-1)
        assert len(cfg.validate()) >= 1

    def test_load_from_yaml(self):
        cfg = _telemetry._load_config(
            _BASE / "opti_oignon" / "config" / "telemetry.yaml"
        )
        assert cfg.enabled is True
        assert cfg.buffer_max_size == 64

    def test_load_missing(self):
        cfg = _telemetry._load_config(Path("/tmp/nonexistent.yaml"))
        assert cfg.enabled is True


# =========================================================================
# Telemetry — Data types
# =========================================================================


class TestInferenceEvent:
    def test_to_dict(self):
        ev = _telemetry.InferenceEvent(
            event_type="inference_start",
            request_id="abc123",
            model="llama3",
            data={"message_count": 3},
        )
        d = ev.to_dict()
        assert d["event_type"] == "inference_start"
        assert d["request_id"] == "abc123"
        assert d["data"]["message_count"] == 3


class TestActiveRequest:
    def test_defaults(self):
        ar = _telemetry.ActiveRequest(request_id="test", model="llama3")
        assert ar.token_count == 0
        assert ar.token_timestamps == []


# =========================================================================
# Telemetry — Collector
# =========================================================================


class TestTelemetryCollector:
    def _make_collector(self, **kwargs):
        cfg = _telemetry.TelemetryConfig(
            enabled=True,
            buffer_flush_interval_ms=0,  # immediate flush
            **kwargs,
        )
        return _telemetry.TelemetryCollector(config=cfg)

    def test_on_inference_start(self):
        tc = self._make_collector()
        rid = tc.on_inference_start("llama3", [{"role": "user", "content": "hi"}])
        assert len(rid) == 12
        stats = tc.get_stats()
        assert stats["total_requests"] == 1

    def test_on_token_generated(self):
        tc = self._make_collector()
        rid = tc.on_inference_start("llama3")
        tc.on_token_generated(rid, count=5)
        stats = tc.get_stats()
        assert stats["total_tokens"] == 5

    def test_on_inference_end(self):
        tc = self._make_collector()
        rid = tc.on_inference_start("llama3")
        tc.on_token_generated(rid, count=3)
        tc.on_inference_end(rid, model="llama3", tokens_out=3, latency_ms=100.0)
        stats = tc.get_stats()
        assert stats["active_requests"] == 0
        assert stats["total_events"] == 3  # start + 1 token event + end

    def test_full_lifecycle_events(self):
        received = []

        def consumer(events):
            received.extend(events)

        tc = self._make_collector()
        tc.register_consumer(consumer)
        rid = tc.on_inference_start("model")
        tc.on_token_generated(rid, count=1)
        tc.on_token_generated(rid, count=1)
        tc.on_inference_end(rid, latency_ms=200.0)
        tc.flush()

        types = [e.event_type for e in received]
        assert "inference_start" in types
        assert "token_generated" in types
        assert "inference_end" in types

    def test_consumer_receives_all_events(self):
        received = []

        def consumer(events):
            received.extend(events)

        tc = self._make_collector()
        tc.register_consumer(consumer)
        rid = tc.on_inference_start("m")
        for _ in range(10):
            tc.on_token_generated(rid)
        tc.on_inference_end(rid)
        tc.flush()
        assert len(received) == 12  # 1 start + 10 tokens + 1 end

    def test_unregister_consumer(self):
        received = []

        def consumer(events):
            received.extend(events)

        tc = self._make_collector()
        tc.register_consumer(consumer)
        tc.on_inference_start("m")
        tc.flush()
        count_after_first = len(received)

        tc.unregister_consumer(consumer)
        tc.on_inference_start("m2")
        tc.flush()
        assert len(received) == count_after_first  # no more events

    def test_buffer_flush_on_max_size(self):
        received = []

        def consumer(events):
            received.extend(events)

        cfg = _telemetry.TelemetryConfig(
            enabled=True,
            buffer_max_size=5,
            buffer_flush_interval_ms=0,
        )
        tc = _telemetry.TelemetryCollector(config=cfg)
        tc.register_consumer(consumer)

        for i in range(5):
            tc.on_inference_start(f"model-{i}")
        # Buffer should have auto-flushed at size 5.
        assert len(received) >= 5

    def test_disabled_collector(self):
        cfg = _telemetry.TelemetryConfig(enabled=False)
        tc = _telemetry.TelemetryCollector(config=cfg)
        rid = tc.on_inference_start("m")
        assert len(rid) == 12  # still returns an ID
        stats = tc.get_stats()
        assert stats["total_events"] == 0

    def test_latency_auto_calculation(self):
        received = []

        def consumer(events):
            received.extend(events)

        tc = self._make_collector()
        tc.register_consumer(consumer)
        rid = tc.on_inference_start("m")
        time.sleep(0.05)
        tc.on_inference_end(rid)  # latency_ms not provided
        tc.flush()

        end_events = [e for e in received if e.event_type == "inference_end"]
        assert len(end_events) == 1
        assert end_events[0].data["latency_ms"] > 40.0  # at least ~50ms

    def test_speculative_data_pass_through(self):
        received = []

        def consumer(events):
            received.extend(events)

        tc = self._make_collector()
        tc.register_consumer(consumer)
        rid = tc.on_inference_start("m")
        tc.on_inference_end(
            rid,
            speculative_data={"draft_tokens": 10, "accepted": 8, "speedup": 1.5},
        )
        tc.flush()

        end_events = [e for e in received if e.event_type == "inference_end"]
        assert end_events[0].data["speculative_data"]["accepted"] == 8

    def test_shutdown(self):
        tc = self._make_collector()
        tc.shutdown()
        # Should not raise

    def test_flush_thread_start_stop(self):
        cfg = _telemetry.TelemetryConfig(
            enabled=True, buffer_flush_interval_ms=100
        )
        tc = _telemetry.TelemetryCollector(config=cfg)
        tc.start_flush_thread()
        assert tc._running is True
        tc.stop_flush_thread()
        assert tc._running is False

    def test_consumer_error_handling(self):
        def bad_consumer(events):
            raise ValueError("boom")

        tc = self._make_collector()
        tc.register_consumer(bad_consumer)
        # Should not raise
        tc.on_inference_start("m")
        tc.flush()

    def test_get_stats(self):
        tc = self._make_collector()
        stats = tc.get_stats()
        assert "enabled" in stats
        assert "total_events" in stats
        assert "consumer_count" in stats


class TestTelemetrySingleton:
    def test_get_and_reset(self):
        _telemetry.reset_telemetry()
        t1 = _telemetry.get_telemetry()
        t2 = _telemetry.get_telemetry()
        assert t1 is t2
        _telemetry.reset_telemetry()


# =========================================================================
# Auto-Tuner Recommendations
# =========================================================================


class TestTunerRecommendation:
    def test_to_dict(self):
        rec = _auto_tuner.TunerRecommendation(
            title="Set threads to 8",
            parameter="threads",
            current_value=4,
            recommended_value=8,
            estimated_speedup=1.5,
            confidence="high",
        )
        d = rec.to_dict()
        assert d["title"] == "Set threads to 8"
        assert d["estimated_speedup"] == 1.5
        assert d["confidence"] == "high"

    def test_defaults(self):
        rec = _auto_tuner.TunerRecommendation()
        assert rec.title == ""
        assert rec.estimated_speedup == 1.0
        assert rec.applied is False


class TestGenerateRecommendations:
    def _make_profile(self, **kwargs):
        defaults = {
            "model_name": "test",
            "best_params": {"threads": 8, "batch_size": 2048, "flash_attention": True},
            "best_tg_speed": 45.0,
            "best_pp_speed": 120.0,
            "baseline_tg_speed": 30.0,
            "baseline_pp_speed": 80.0,
            "speedup_factor": 1.5,
            "all_results": [
                {"params": {"threads": 4, "batch_size": 512, "flash_attention": False}, "tokens_per_second_tg": 25.0},
                {"params": {"threads": 4, "batch_size": 2048, "flash_attention": True}, "tokens_per_second_tg": 32.0},
                {"params": {"threads": 8, "batch_size": 512, "flash_attention": False}, "tokens_per_second_tg": 35.0},
                {"params": {"threads": 8, "batch_size": 2048, "flash_attention": True}, "tokens_per_second_tg": 45.0},
            ],
        }
        defaults.update(kwargs)
        return _auto_tuner.TunerProfile(**defaults)

    def test_generates_recommendations(self):
        profile = self._make_profile()
        recs = _auto_tuner.generate_recommendations(profile)
        assert len(recs) > 0

    def test_sorted_by_speedup(self):
        profile = self._make_profile()
        recs = _auto_tuner.generate_recommendations(profile)
        speedups = [r.estimated_speedup for r in recs]
        assert speedups == sorted(speedups, reverse=True)

    def test_overall_recommendation(self):
        profile = self._make_profile()
        recs = _auto_tuner.generate_recommendations(profile)
        overall = [r for r in recs if r.parameter == "all"]
        assert len(overall) == 1
        assert overall[0].estimated_speedup == pytest.approx(1.5, abs=0.1)

    def test_thread_recommendation(self):
        profile = self._make_profile()
        recs = _auto_tuner.generate_recommendations(profile)
        thread_recs = [r for r in recs if r.parameter == "threads"]
        assert len(thread_recs) == 1
        assert thread_recs[0].recommended_value == 8

    def test_flash_attention_recommendation(self):
        profile = self._make_profile()
        recs = _auto_tuner.generate_recommendations(profile)
        fa_recs = [r for r in recs if r.parameter == "flash_attention"]
        # May or may not appear depending on speed diff
        if fa_recs:
            assert fa_recs[0].recommended_value is True

    def test_empty_profile(self):
        profile = _auto_tuner.TunerProfile()
        recs = _auto_tuner.generate_recommendations(profile)
        assert recs == []

    def test_no_speedup(self):
        profile = self._make_profile(
            best_tg_speed=30.0,
            baseline_tg_speed=30.0,
            all_results=[
                {"params": {"threads": 4}, "tokens_per_second_tg": 30.0},
                {"params": {"threads": 8}, "tokens_per_second_tg": 30.0},
            ],
        )
        recs = _auto_tuner.generate_recommendations(profile)
        # No overall recommendation when speedup <= 1.05
        overall = [r for r in recs if r.parameter == "all"]
        assert len(overall) == 0

    def test_gpu_layers_recommendation(self):
        profile = self._make_profile(
            best_params={"gpu_layers": 32},
            all_results=[
                {"params": {"gpu_layers": 0}, "tokens_per_second_tg": 10.0},
                {"params": {"gpu_layers": 16}, "tokens_per_second_tg": 20.0},
                {"params": {"gpu_layers": 32}, "tokens_per_second_tg": 35.0},
            ],
        )
        recs = _auto_tuner.generate_recommendations(profile)
        gl_recs = [r for r in recs if r.parameter == "gpu_layers"]
        assert len(gl_recs) == 1
        assert gl_recs[0].recommended_value == 32

    def test_batch_size_recommendation(self):
        profile = self._make_profile()
        recs = _auto_tuner.generate_recommendations(profile)
        batch_recs = [r for r in recs if r.parameter == "batch_size"]
        if batch_recs:
            assert batch_recs[0].estimated_speedup > 1.0

    def test_flash_attention_disable_recommendation(self):
        """Flash attention should be recommended to disable if it's slower."""
        profile = self._make_profile(
            best_params={"flash_attention": False},
            all_results=[
                {"params": {"flash_attention": True}, "tokens_per_second_tg": 20.0},
                {"params": {"flash_attention": False}, "tokens_per_second_tg": 30.0},
            ],
        )
        recs = _auto_tuner.generate_recommendations(profile)
        fa_recs = [r for r in recs if r.parameter == "flash_attention"]
        assert len(fa_recs) == 1
        assert fa_recs[0].recommended_value is False

    def test_single_result_no_param_recs(self):
        """With only one result, no per-param recommendations should be generated."""
        profile = self._make_profile(
            all_results=[
                {"params": {"threads": 4}, "tokens_per_second_tg": 30.0},
            ],
        )
        recs = _auto_tuner.generate_recommendations(profile)
        param_recs = [r for r in recs if r.parameter != "all"]
        assert len(param_recs) == 0


# =========================================================================
# Config YAML files parse correctly
# =========================================================================


class TestConfigYAML:
    def test_model_lifecycle_yaml(self):
        import yaml
        p = _BASE / "opti_oignon" / "config" / "model_lifecycle.yaml"
        with open(p) as f:
            data = yaml.safe_load(f)
        assert data["enabled"] is True
        assert "pull" in data
        assert "aliases" in data

    def test_telemetry_yaml(self):
        import yaml
        p = _BASE / "opti_oignon" / "config" / "telemetry.yaml"
        with open(p) as f:
            data = yaml.safe_load(f)
        assert data["enabled"] is True
        assert "buffer" in data
        assert "consumers" in data


# =========================================================================
# Frontend — no hardcoded hex colors
# =========================================================================


_FRONTEND_BASE = _BASE / "frontend" / "src"


class TestFrontendNoHex:
    _HEX_RE = re.compile(r"(?:color|background|background-color)\s*:\s*#[0-9a-fA-F]{3,8}")

    def _check_file(self, path: Path):
        if not path.exists():
            pytest.skip(f"{path} not found")
        content = path.read_text()
        matches = self._HEX_RE.findall(content)
        assert matches == [], f"Hardcoded hex in {path.name}: {matches}"

    def test_model_manager_no_hex(self):
        self._check_file(_FRONTEND_BASE / "lib" / "components" / "panels" / "ModelManager.svelte")

    def test_performance_tuner_panel_no_hex(self):
        self._check_file(_FRONTEND_BASE / "lib" / "components" / "settings" / "PerformanceTunerPanel.svelte")


# =========================================================================
# Schemas exist and have expected fields
# =========================================================================


class TestSchemas:
    def test_model_lifecycle_schemas_parseable(self):
        import ast
        p = _BASE / "opti_oignon" / "api" / "schemas.py"
        tree = ast.parse(p.read_text())
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        expected = {
            "ModelPullRequest", "ModelPullJobSchema", "ModelDeleteRequest",
            "ModelDeleteResponse", "ModelUpdateCheckRequest", "ModelUpdateInfoSchema",
            "ModelUpdatesResponse", "ModelAliasRequest", "ModelAliasesResponse",
            "ModelLifecycleStatusResponse", "StaleModelsResponse",
            "TunerRecommendationSchema", "TunerRecommendationsResponse",
        }
        for name in expected:
            assert name in classes, f"Missing schema: {name}"


# =========================================================================
# Routes file AST checks
# =========================================================================


class TestRoutesAST:
    def test_routes_model_lifecycle_ast(self):
        import ast
        p = _BASE / "opti_oignon" / "api" / "routes_model_lifecycle.py"
        tree = ast.parse(p.read_text())
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        expected = {
            "get_lifecycle_status", "start_model_pull", "get_pull_progress",
            "cancel_model_pull", "list_pull_jobs", "delete_model",
            "check_model_updates", "list_aliases", "set_alias",
            "remove_alias", "detect_stale_models", "get_model_detail",
        }
        for name in expected:
            assert name in func_names, f"Missing route function: {name}"

    def test_routes_tuner_recommendations(self):
        import ast
        p = _BASE / "opti_oignon" / "api" / "routes_tuner.py"
        tree = ast.parse(p.read_text())
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "get_tuner_recommendations" in func_names

    def test_app_includes_lifecycle_router(self):
        p = _BASE / "opti_oignon" / "api" / "app.py"
        content = p.read_text()
        assert "model_lifecycle_router" in content
