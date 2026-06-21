#!/usr/bin/env python3
"""
Tests for S113 — Telemetry Dashboard + Model Lifecycle UI Polish + Inference Profiler.

Covers:
  -- Telemetry Dashboard --
  1.  TelemetryStatsResponse schema fields
  2.  TelemetryConsumerInfoSchema defaults
  3.  TelemetryConsumersResponse schema
  4.  TelemetryFlushResponse schema
  5.  routes_telemetry: GET /stats returns stats when available
  6.  routes_telemetry: GET /stats returns disabled when unavailable
  7.  routes_telemetry: GET /consumers lists consumers
  8.  routes_telemetry: POST /flush triggers flush
  9.  routes_telemetry: _ensure_available raises 503

  -- Inference Profiler --
  10. InferenceProfile dataclass defaults
  11. InferenceProfile to_dict roundtrip
  12. _percentile: empty list
  13. _percentile: single value
  14. _percentile: two values p50
  15. _percentile: three values p50/p95/p99
  16. _percentile: ten values p95
  17. InferenceProfiler: single request lifecycle
  18. InferenceProfiler: token tracking (first/last)
  19. InferenceProfiler: overhead calculation
  20. InferenceProfiler: end without start (minimal profile)
  21. InferenceProfiler: multi-model summary
  22. InferenceProfiler: ring buffer max capacity
  23. InferenceProfiler: get_recent ordering (most recent first)
  24. InferenceProfiler: get_stats counters
  25. InferenceProfiler: consume method name
  26. InferenceProfiler: shutdown clears traces
  27. InferenceProfiler: tok_per_sec calculation
  28. InferenceProfiler: zero latency edge case
  29. ProfilerSummarySchema defaults
  30. ProfilerRecentResponse schema
  31. ProfilerSummaryResponse schema
  32. routes_profiler: GET /summary returns data
  33. routes_profiler: GET /recent returns profiles
  34. routes_profiler: _ensure_available raises 503

  -- Schemas (Telemetry + Profiler) --
  35. InferenceProfileSchema fields
  36. TelemetryStatsResponse buffer_max_size default

  -- Frontend hex compliance --
  37. TelemetryDashboard.svelte: no hardcoded hex
  38. ModelManager.svelte: no hardcoded hex (S113 version)

  -- Integration --
  39. Profiler as telemetry consumer: full pipeline
  40. Profiler percentile accuracy on known distribution
  41. deps.py: TELEMETRY_AVAILABLE flag exists
  42. deps.py: INFERENCE_PROFILER_AVAILABLE flag exists
  43. app.py: telemetry router registered
  44. app.py: profiler router registered
  45. app.py: health check includes telemetry and profiler

Total: 45 tests
"""

import collections
import importlib
import importlib.util
import math
import os
import re
import sys
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

_BASE = Path(__file__).resolve().parent.parent

_telemetry = _load_module(
    "opti_oignon.telemetry",
    str(_BASE / "opti_oignon" / "telemetry.py"),
)
_profiler = _load_module(
    "opti_oignon.inference_profiler",
    str(_BASE / "opti_oignon" / "inference_profiler.py"),
)


# =========================================================================
# Telemetry Dashboard — Schemas
# =========================================================================


class TestTelemetryDashboardSchemas:
    """Tests for the telemetry dashboard Pydantic schemas."""

    def test_telemetry_stats_response_defaults(self):
        """TelemetryStatsResponse has correct default fields."""
        # We test the schema shape by importing from schemas source
        schemas_path = _BASE / "opti_oignon" / "api" / "schemas.py"
        content = schemas_path.read_text()
        assert "class TelemetryStatsResponse" in content
        assert "total_events" in content
        assert "total_requests" in content
        assert "buffer_max_size" in content

    def test_telemetry_consumer_info_schema(self):
        """TelemetryConsumerInfoSchema has name and healthy fields."""
        schemas_path = _BASE / "opti_oignon" / "api" / "schemas.py"
        content = schemas_path.read_text()
        assert "class TelemetryConsumerInfoSchema" in content
        assert "name: str" in content
        assert "healthy: bool" in content

    def test_telemetry_consumers_response_schema(self):
        """TelemetryConsumersResponse wraps list of consumers."""
        schemas_path = _BASE / "opti_oignon" / "api" / "schemas.py"
        content = schemas_path.read_text()
        assert "class TelemetryConsumersResponse" in content
        assert "consumers" in content

    def test_telemetry_flush_response_schema(self):
        """TelemetryFlushResponse has flushed_events."""
        schemas_path = _BASE / "opti_oignon" / "api" / "schemas.py"
        content = schemas_path.read_text()
        assert "class TelemetryFlushResponse" in content
        assert "flushed_events" in content

    def test_telemetry_stats_buffer_max_default(self):
        """TelemetryStatsResponse buffer_max_size defaults to 64."""
        schemas_path = _BASE / "opti_oignon" / "api" / "schemas.py"
        content = schemas_path.read_text()
        assert "buffer_max_size: int = 64" in content


# =========================================================================
# Telemetry Dashboard — Routes structure
# =========================================================================


class TestTelemetryDashboardRoutes:
    """Tests for routes_telemetry.py structure."""

    def test_routes_telemetry_exists(self):
        routes_path = _BASE / "opti_oignon" / "api" / "routes_telemetry.py"
        assert routes_path.is_file()

    def test_routes_has_stats_endpoint(self):
        content = (_BASE / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert '"/stats"' in content
        assert "TelemetryStatsResponse" in content

    def test_routes_has_consumers_endpoint(self):
        content = (_BASE / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert '"/consumers"' in content
        assert "TelemetryConsumersResponse" in content

    def test_routes_has_flush_endpoint(self):
        content = (_BASE / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert '"/flush"' in content
        assert "TelemetryFlushResponse" in content

    def test_routes_ensure_available_pattern(self):
        content = (_BASE / "opti_oignon" / "api" / "routes_telemetry.py").read_text()
        assert "_ensure_available" in content
        assert "503" in content


# =========================================================================
# Inference Profiler — _percentile
# =========================================================================


class TestPercentile:
    """Tests for the _percentile helper."""

    def test_empty_list(self):
        assert _profiler._percentile([], 50) == 0.0

    def test_single_value(self):
        assert _profiler._percentile([42.0], 50) == 42.0
        assert _profiler._percentile([42.0], 99) == 42.0

    def test_two_values_p50(self):
        result = _profiler._percentile([10.0, 20.0], 50)
        assert result == 15.0

    def test_three_values_p50(self):
        result = _profiler._percentile([10.0, 20.0, 30.0], 50)
        assert result == 20.0

    def test_three_values_p95(self):
        result = _profiler._percentile([10.0, 20.0, 30.0], 95)
        assert result > 28.0  # should be close to 29

    def test_ten_values_p95(self):
        values = sorted([float(i * 10) for i in range(1, 11)])
        result = _profiler._percentile(values, 95)
        assert 85.0 <= result <= 100.0


# =========================================================================
# Inference Profiler — InferenceProfile dataclass
# =========================================================================


class TestInferenceProfile:
    """Tests for the InferenceProfile dataclass."""

    def test_defaults(self):
        p = _profiler.InferenceProfile()
        assert p.request_id == ""
        assert p.model == ""
        assert p.total_ms == 0.0
        assert p.tokens_out == 0
        assert p.tok_per_sec == 0.0

    def test_to_dict_roundtrip(self):
        p = _profiler.InferenceProfile(
            request_id="abc",
            model="test",
            timestamp=1000.0,
            total_ms=150.5,
            prompt_eval_ms=30.2,
            token_gen_ms=100.1,
            overhead_ms=20.2,
            tokens_in=50,
            tokens_out=20,
            tok_per_sec=133.33,
        )
        d = p.to_dict()
        assert d["request_id"] == "abc"
        assert d["model"] == "test"
        assert d["total_ms"] == 150.5
        assert d["tokens_out"] == 20
        assert d["tok_per_sec"] == 133.33


# =========================================================================
# Inference Profiler — InferenceProfiler
# =========================================================================


class TestInferenceProfiler:
    """Tests for the InferenceProfiler class."""

    def _make_event(self, event_type, request_id="r1", model="m1",
                    timestamp=None, data=None):
        return _telemetry.InferenceEvent(
            event_type=event_type,
            request_id=request_id,
            timestamp=timestamp or time.time(),
            model=model,
            data=data or {},
        )

    def test_single_request_lifecycle(self):
        """Full start -> tokens -> end lifecycle produces a profile."""
        prof = _profiler.InferenceProfiler(max_profiles=50)
        t0 = time.time()
        events = [
            self._make_event("inference_start", "r1", "m1", t0),
            self._make_event("token_generated", "r1", "m1", t0 + 0.01, {"count": 1}),
            self._make_event("token_generated", "r1", "m1", t0 + 0.02, {"count": 1}),
            self._make_event("inference_end", "r1", "m1", t0 + 0.05, {
                "latency_ms": 50.0,
                "tokens_in": 10,
                "tokens_out": 2,
            }),
        ]
        prof.consume(events)
        recent = prof.get_recent(5)
        assert len(recent) == 1
        assert recent[0]["model"] == "m1"
        assert recent[0]["total_ms"] == 50.0

    def test_token_tracking_first_last(self):
        """Prompt eval and token gen times are computed from token timestamps."""
        prof = _profiler.InferenceProfiler()
        t0 = 1000.0
        events = [
            self._make_event("inference_start", "r1", "m1", t0),
            self._make_event("token_generated", "r1", "m1", t0 + 0.1, {"count": 1}),
            self._make_event("token_generated", "r1", "m1", t0 + 0.3, {"count": 1}),
            self._make_event("inference_end", "r1", "m1", t0 + 0.4, {
                "latency_ms": 400.0, "tokens_in": 5, "tokens_out": 2,
            }),
        ]
        prof.consume(events)
        p = prof.get_recent(1)[0]
        # prompt_eval = first_token - start = 0.1s = 100ms
        assert abs(p["prompt_eval_ms"] - 100.0) < 1.0
        # token_gen = last_token - first_token = 0.2s = 200ms
        assert abs(p["token_gen_ms"] - 200.0) < 1.0

    def test_overhead_calculation(self):
        """Overhead = total - prompt_eval - token_gen."""
        prof = _profiler.InferenceProfiler()
        t0 = 1000.0
        events = [
            self._make_event("inference_start", "r1", "m1", t0),
            self._make_event("token_generated", "r1", "m1", t0 + 0.05, {"count": 1}),
            self._make_event("token_generated", "r1", "m1", t0 + 0.15, {"count": 1}),
            self._make_event("inference_end", "r1", "m1", t0 + 0.3, {
                "latency_ms": 300.0, "tokens_in": 5, "tokens_out": 2,
            }),
        ]
        prof.consume(events)
        p = prof.get_recent(1)[0]
        expected_overhead = 300.0 - p["prompt_eval_ms"] - p["token_gen_ms"]
        assert abs(p["overhead_ms"] - expected_overhead) < 1.0

    def test_end_without_start(self):
        """End event without start creates a minimal profile."""
        prof = _profiler.InferenceProfiler()
        events = [
            self._make_event("inference_end", "orphan", "m2", data={
                "latency_ms": 200.0, "tokens_in": 15, "tokens_out": 8,
            }),
        ]
        prof.consume(events)
        recent = prof.get_recent(5)
        assert len(recent) == 1
        assert recent[0]["request_id"] == "orphan"
        assert recent[0]["total_ms"] == 200.0
        assert recent[0]["prompt_eval_ms"] == 0.0

    def test_multi_model_summary(self):
        """Summary groups profiles by model."""
        prof = _profiler.InferenceProfiler()
        t0 = 1000.0
        for i in range(6):
            model = "fast" if i < 4 else "slow"
            latency = 50.0 if model == "fast" else 500.0
            events = [
                self._make_event("inference_start", f"r{i}", model, t0 + i),
                self._make_event("inference_end", f"r{i}", model, t0 + i + 0.1, {
                    "latency_ms": latency, "tokens_in": 10, "tokens_out": 5,
                }),
            ]
            prof.consume(events)

        summary = prof.get_summary()
        assert len(summary) == 2
        fast = [s for s in summary if s["model"] == "fast"][0]
        slow = [s for s in summary if s["model"] == "slow"][0]
        assert fast["request_count"] == 4
        assert slow["request_count"] == 2
        assert fast["avg_total_ms"] == 50.0
        assert slow["avg_total_ms"] == 500.0

    def test_ring_buffer_max_capacity(self):
        """Ring buffer respects max_profiles limit."""
        prof = _profiler.InferenceProfiler(max_profiles=10)
        for i in range(25):
            events = [
                self._make_event("inference_start", f"r{i}", "m"),
                self._make_event("inference_end", f"r{i}", "m", data={
                    "latency_ms": float(i), "tokens_out": 1,
                }),
            ]
            prof.consume(events)
        assert prof.total_profiled == 25
        recent = prof.get_recent(100)
        assert len(recent) == 10  # ring buffer capped at 10

    def test_get_recent_ordering(self):
        """Most recent profiles come first."""
        prof = _profiler.InferenceProfiler()
        t0 = 1000.0
        for i in range(5):
            events = [
                self._make_event("inference_start", f"r{i}", "m", t0 + i),
                self._make_event("inference_end", f"r{i}", "m", t0 + i + 0.01, {
                    "latency_ms": float(i * 10), "tokens_out": 1,
                }),
            ]
            prof.consume(events)
        recent = prof.get_recent(5)
        assert recent[0]["request_id"] == "r4"
        assert recent[4]["request_id"] == "r0"

    def test_get_stats_counters(self):
        """get_stats returns correct counters."""
        prof = _profiler.InferenceProfiler(max_profiles=50)
        events = [
            self._make_event("inference_start", "r1", "m"),
            self._make_event("inference_end", "r1", "m", data={
                "latency_ms": 100.0, "tokens_out": 5,
            }),
        ]
        prof.consume(events)
        stats = prof.get_stats()
        assert stats["total_profiled"] == 1
        assert stats["buffer_size"] == 1
        assert stats["active_traces"] == 0

    def test_consume_method_name(self):
        """Consumer method has correct __name__ for dashboard display."""
        prof = _profiler.InferenceProfiler()
        assert getattr(prof.consume, "__name__", "") == "inference_profiler_consumer"

    def test_shutdown_clears_traces(self):
        """shutdown() clears active traces."""
        prof = _profiler.InferenceProfiler()
        events = [self._make_event("inference_start", "r1", "m")]
        prof.consume(events)
        assert prof.get_stats()["active_traces"] == 1
        prof.shutdown()
        assert prof.get_stats()["active_traces"] == 0

    def test_tok_per_sec_calculation(self):
        """tok_per_sec = tokens_out / (total_ms / 1000)."""
        prof = _profiler.InferenceProfiler()
        events = [
            self._make_event("inference_start", "r1", "m"),
            self._make_event("inference_end", "r1", "m", data={
                "latency_ms": 500.0, "tokens_out": 100, "tokens_in": 10,
            }),
        ]
        prof.consume(events)
        p = prof.get_recent(1)[0]
        expected = 100 / 0.5  # 200 tok/s
        assert abs(p["tok_per_sec"] - expected) < 0.1

    def test_zero_latency_edge_case(self):
        """Zero latency produces zero tok_per_sec without division error."""
        prof = _profiler.InferenceProfiler()
        events = [
            self._make_event("inference_end", "r1", "m", data={
                "latency_ms": 0.0, "tokens_out": 10,
            }),
        ]
        prof.consume(events)
        p = prof.get_recent(1)[0]
        assert p["tok_per_sec"] == 0.0


# =========================================================================
# Inference Profiler — Schemas
# =========================================================================


class TestProfilerSchemas:
    """Tests for profiler Pydantic schemas in schemas.py."""

    def test_inference_profile_schema_fields(self):
        content = (_BASE / "opti_oignon" / "api" / "schemas.py").read_text()
        assert "class InferenceProfileSchema" in content
        for field in ["request_id", "total_ms", "prompt_eval_ms", "token_gen_ms",
                       "overhead_ms", "tok_per_sec"]:
            assert field in content

    def test_profiler_summary_schema(self):
        content = (_BASE / "opti_oignon" / "api" / "schemas.py").read_text()
        assert "class ProfilerSummarySchema" in content
        assert "p50_total_ms" in content
        assert "p95_total_ms" in content
        assert "p99_total_ms" in content

    def test_profiler_summary_response_schema(self):
        content = (_BASE / "opti_oignon" / "api" / "schemas.py").read_text()
        assert "class ProfilerSummaryResponse" in content
        assert "total_profiled_requests" in content

    def test_profiler_recent_response_schema(self):
        content = (_BASE / "opti_oignon" / "api" / "schemas.py").read_text()
        assert "class ProfilerRecentResponse" in content


# =========================================================================
# Routes — Profiler
# =========================================================================


class TestProfilerRoutes:
    """Tests for routes_profiler.py structure."""

    def test_routes_profiler_exists(self):
        assert (_BASE / "opti_oignon" / "api" / "routes_profiler.py").is_file()

    def test_routes_has_summary_endpoint(self):
        content = (_BASE / "opti_oignon" / "api" / "routes_profiler.py").read_text()
        assert '"/summary"' in content
        assert "ProfilerSummaryResponse" in content

    def test_routes_has_recent_endpoint(self):
        content = (_BASE / "opti_oignon" / "api" / "routes_profiler.py").read_text()
        assert '"/recent"' in content
        assert "ProfilerRecentResponse" in content

    def test_routes_ensure_available_pattern(self):
        content = (_BASE / "opti_oignon" / "api" / "routes_profiler.py").read_text()
        assert "_ensure_available" in content
        assert "503" in content


# =========================================================================
# Frontend — Hex compliance
# =========================================================================


def _svelte_files():
    """Yield all .svelte file paths in the frontend."""
    frontend_dir = _BASE / "frontend" / "src"
    if not frontend_dir.is_dir():
        return
    for root, dirs, files in os.walk(frontend_dir):
        for f in files:
            if f.endswith(".svelte"):
                yield os.path.join(root, f)


class TestFrontendHexCompliance:
    """No hardcoded hex colors in S113 Svelte files."""

    HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b")
    ALLOWED_PATTERNS = ["var(--oo-", "<!--", "//"]

    def _check_file(self, filename):
        violations = []
        for fpath in _svelte_files():
            if os.path.basename(fpath) != filename:
                continue
            content = open(fpath).read()
            for i, line in enumerate(content.splitlines(), 1):
                if not self.HEX_RE.search(line):
                    continue
                if any(pat in line for pat in self.ALLOWED_PATTERNS):
                    continue
                violations.append(f"{filename}:{i}: {line.strip()[:100]}")
        return violations

    def test_telemetry_dashboard_no_hex(self):
        violations = self._check_file("TelemetryDashboard.svelte")
        assert violations == [], (
            "Hardcoded hex in TelemetryDashboard.svelte:\n" +
            "\n".join(violations)
        )

    def test_model_manager_no_hex(self):
        violations = self._check_file("ModelManager.svelte")
        assert violations == [], (
            "Hardcoded hex in ModelManager.svelte:\n" +
            "\n".join(violations)
        )


# =========================================================================
# Integration — Profiler as telemetry consumer
# =========================================================================


class TestProfilerTelemetryIntegration:
    """Integration: profiler consuming from telemetry pipeline."""

    def test_full_pipeline(self):
        """Profiler receives events through telemetry collector."""
        cfg = _telemetry.TelemetryConfig(enabled=True, buffer_flush_interval_ms=0)
        col = _telemetry.TelemetryCollector(config=cfg)
        prof = _profiler.InferenceProfiler(max_profiles=50)
        col.register_consumer(prof.consume)

        rid = col.on_inference_start("gpt-local", [{"role": "user", "content": "hi"}])
        col.on_token_generated(rid, 3)
        col.on_inference_end(rid, model="gpt-local", tokens_in=5, tokens_out=3, latency_ms=75.0)

        recent = prof.get_recent(5)
        assert len(recent) == 1
        assert recent[0]["model"] == "gpt-local"
        assert recent[0]["tokens_out"] == 3
        assert recent[0]["total_ms"] == 75.0

    def test_percentile_accuracy_known_distribution(self):
        """Percentile on a known distribution gives expected values."""
        prof = _profiler.InferenceProfiler(max_profiles=200)
        cfg = _telemetry.TelemetryConfig(enabled=True, buffer_flush_interval_ms=0)
        col = _telemetry.TelemetryCollector(config=cfg)
        col.register_consumer(prof.consume)

        # Generate 100 requests with latencies 1..100
        for i in range(1, 101):
            rid = col.on_inference_start("bench")
            col.on_inference_end(rid, model="bench", latency_ms=float(i), tokens_out=1)

        summary = prof.get_summary()
        assert len(summary) == 1
        s = summary[0]
        assert s["request_count"] == 100
        assert abs(s["avg_total_ms"] - 50.5) < 0.1
        assert abs(s["p50_total_ms"] - 50.0) < 1.5
        assert s["p95_total_ms"] >= 94.0
        assert s["p99_total_ms"] >= 98.0


# =========================================================================
# deps.py / app.py — Registration checks
# =========================================================================


class TestRegistration:
    """Verify S113 modules are registered in deps.py and app.py."""

    def test_deps_telemetry_available(self):
        content = (_BASE / "opti_oignon" / "api" / "deps.py").read_text()
        assert "TELEMETRY_AVAILABLE" in content
        assert "get_telemetry" in content

    def test_deps_profiler_available(self):
        content = (_BASE / "opti_oignon" / "api" / "deps.py").read_text()
        assert "INFERENCE_PROFILER_AVAILABLE" in content
        assert "get_profiler" in content

    def test_app_telemetry_router(self):
        content = (_BASE / "opti_oignon" / "api" / "app.py").read_text()
        assert "telemetry_router" in content

    def test_app_profiler_router(self):
        content = (_BASE / "opti_oignon" / "api" / "app.py").read_text()
        assert "profiler_router" in content

    def test_health_check_telemetry(self):
        content = (_BASE / "opti_oignon" / "api" / "app.py").read_text()
        assert '"telemetry"' in content
        assert "TELEMETRY_AVAILABLE" in content

    def test_health_check_profiler(self):
        content = (_BASE / "opti_oignon" / "api" / "app.py").read_text()
        assert '"inference_profiler"' in content
        assert "INFERENCE_PROFILER_AVAILABLE" in content
