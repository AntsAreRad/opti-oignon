#!/usr/bin/env python3
"""
S193 F6d — observability fixes (live_metrics / performance_monitor /
inference_profiler).

Covers:
  - LMT-01: GPU subprocess is gated to active generation (idle carries forward)
  - PRF-03: performance_metrics.db is purged opportunistically on record
  - PRF-04: the profiler tolerates a None telemetry data payload
"""

import importlib.util
import sys
import tempfile
import time
import types
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    full = _PROJECT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_lm_mod = _load_module("s193d_live_metrics", "opti_oignon/live_metrics.py")
_pm_mod = _load_module("s193d_performance_monitor", "opti_oignon/performance_monitor.py")
_ip_mod = _load_module("s193d_inference_profiler", "opti_oignon/inference_profiler.py")


# ---------------------------------------------------------------------------
# LMT-01 — GPU sampling gated to generation
# ---------------------------------------------------------------------------

class TestLMT01GpuGating:
    def _collector(self):
        c = _lm_mod.LiveMetricsCollector(config=_lm_mod.LiveMetricsConfig(gpu_monitoring=True))
        c._gpu_available = True  # force the GPU path on a CPU-only container
        return c

    def test_idle_does_not_query_gpu(self, monkeypatch):
        calls = {"n": 0}

        def fake_query():
            calls["n"] += 1
            return {"gpu_utilization_pct": 50.0, "gpu_memory_used_mb": 1.0,
                    "gpu_memory_total_mb": 2.0, "gpu_temperature_c": 40.0}

        monkeypatch.setattr(_lm_mod, "_query_gpu_metrics", fake_query)
        c = self._collector()
        # Not generating, no tokens -> no subprocess.
        c._take_sample()
        assert calls["n"] == 0

    def test_generating_queries_gpu(self, monkeypatch):
        calls = {"n": 0}

        def fake_query():
            calls["n"] += 1
            return {"gpu_utilization_pct": 50.0, "gpu_memory_used_mb": 1.0,
                    "gpu_memory_total_mb": 2.0, "gpu_temperature_c": 40.0}

        monkeypatch.setattr(_lm_mod, "_query_gpu_metrics", fake_query)
        c = self._collector()
        c.start_generation(model="m1")
        sample = c._take_sample()
        assert calls["n"] == 1
        assert sample.gpu_utilization_pct == 50.0

    def test_idle_carries_forward_last_gpu(self, monkeypatch):
        monkeypatch.setattr(
            _lm_mod, "_query_gpu_metrics",
            lambda: {"gpu_utilization_pct": 77.0, "gpu_memory_used_mb": 3.0,
                     "gpu_memory_total_mb": 4.0, "gpu_temperature_c": 55.0},
        )
        c = self._collector()
        # Seed history with a generating sample carrying GPU values.
        c.start_generation(model="m1")
        gen_sample = c._take_sample()
        c._history.append(gen_sample)
        c.end_generation()
        # Push the last token well outside the rolling window so idle holds.
        c._token_timestamps.clear()
        idle = c._take_sample()
        assert idle.gpu_utilization_pct == 77.0  # carried forward

    def test_source_has_gate(self):
        src = (_PROJECT / "opti_oignon/live_metrics.py").read_text()
        assert "_should_sample_gpu" in src and "S193 LMT-01" in src


# ---------------------------------------------------------------------------
# PRF-03 — opportunistic retention
# ---------------------------------------------------------------------------

class TestPRF03Retention:
    def test_record_purges_old_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "perf.db"
            mon = _pm_mod.PerformanceMonitor(db_path=db)
            mon._cleanup_interval_s = 0.0  # purge on every record
            mon._retention_days = 7

            now = time.time()
            old_ts = now - (30 * 86400)  # 30 days old, beyond 7d retention

            # Seed an old row directly.
            conn = mon._get_conn()
            conn.execute(
                "INSERT INTO metrics (model, task_type, latency_ms, tokens_in,"
                " tokens_out, quality_score, timestamp) VALUES (?,?,?,?,?,?,?)",
                ("old", "t", 10.0, 1, 1, 1.0, old_ts),
            )
            conn.commit()
            conn.close()

            # A fresh record triggers opportunistic cleanup.
            mon.record_execution("new", "t", 12.0, 1, 1, 1.0, timestamp=now)

            stats = mon.get_latency_stats(window_seconds=10_000_000)
            assert stats.count == 1  # only the fresh row survives

    def test_source_has_opportunistic_cleanup(self):
        src = (_PROJECT / "opti_oignon/performance_monitor.py").read_text()
        assert "_last_cleanup_ts" in src and "S193 PRF-03" in src


# ---------------------------------------------------------------------------
# PRF-04 — profiler tolerates None data payload
# ---------------------------------------------------------------------------

def _ev(event_type, request_id="r1", model="m1", data=None, ts=None):
    e = types.SimpleNamespace()
    e.event_type = event_type
    e.request_id = request_id
    e.model = model
    e.timestamp = ts if ts is not None else time.time()
    e.data = data
    return e


class TestPRF04NullData:
    def test_token_event_with_none_data(self):
        p = _ip_mod.InferenceProfiler()
        # None data on the token event must not raise.
        p.consume([_ev("inference_start")])
        p.consume([_ev("token_generated", data=None)])
        p.consume([_ev("token_generated", data={"count": 2})])
        p.consume([_ev("inference_end", data={})])
        recent = p.get_recent(5)
        assert len(recent) == 1
        # The point of PRF-04: a None data payload did not raise; the profile
        # was still built and is serialisable.
        assert "tokens_out" in recent[0]

    def test_source_has_guard(self):
        src = (_PROJECT / "opti_oignon/inference_profiler.py").read_text()
        assert "S193 PRF-04" in src
