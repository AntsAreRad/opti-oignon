#!/usr/bin/env python3
"""
Tests for PerformanceMonitor -- S72 Step 1.

Covers:
- MetricsRecord dataclass
- Config loading (YAML + defaults)
- record_execution (store and return)
- Disabled monitoring (no-op)
- get_throughput (rolling window)
- get_latency_stats (p50/p95/p99, per-model, all-models)
- get_model_utilization (distribution)
- detect_drift (latency and quality)
- detect_all_drift
- get_recommendations (rule evaluation)
- get_history (raw records)
- cleanup_old_records (retention)
- get_summary
- Singleton creation
"""

import importlib.util
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Direct module import (bypass __init__.py which requires ollama)
# ---------------------------------------------------------------------------

_mod_path = Path(__file__).resolve().parent.parent / "opti_oignon" / "performance_monitor.py"
_spec = importlib.util.spec_from_file_location("performance_monitor_mod", _mod_path)
_mod = importlib.util.module_from_spec(_spec)

# Provide mock ollama
sys.modules.setdefault("ollama", MagicMock())

_spec.loader.exec_module(_mod)

PerformanceMonitor = _mod.PerformanceMonitor
MetricsRecord = _mod.MetricsRecord
LatencyStats = _mod.LatencyStats
DriftResult = _mod.DriftResult
Recommendation = _mod.Recommendation
_load_config = _mod._load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Create a temporary performance.yaml config."""
    cfg = {
        "enabled": True,
        "retention_days": 7,
        "default_window_seconds": 300,
        "drift": {
            "window_seconds": 3600,
            "baseline_window_seconds": 86400,
            "threshold": 0.3,
        },
        "recommendation_rules": [],
    }
    if overrides:
        cfg.update(overrides)
    path = tmp_path / "performance.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def _make_monitor(tmp_path: Path, config_overrides: dict | None = None) -> PerformanceMonitor:
    """Create a PerformanceMonitor with temp DB and config."""
    cfg_path = _make_config(tmp_path, config_overrides)
    db_path = tmp_path / "test_performance.db"
    return PerformanceMonitor(db_path=db_path, config_path=cfg_path)


def _seed_records(monitor: PerformanceMonitor, records: list[dict]):
    """Insert multiple records into the monitor."""
    for rec in records:
        monitor.record_execution(**rec)


# ---------------------------------------------------------------------------
# Tests: Dataclass
# ---------------------------------------------------------------------------

class TestMetricsRecord:
    """Tests for MetricsRecord dataclass."""

    def test_create_record(self):
        rec = MetricsRecord(
            model="qwen3:32b",
            task_type="code_python",
            latency_ms=1200.5,
            tokens_in=100,
            tokens_out=500,
            quality_score=0.85,
        )
        assert rec.model == "qwen3:32b"
        assert rec.task_type == "code_python"
        assert rec.latency_ms == 1200.5
        assert rec.tokens_in == 100
        assert rec.tokens_out == 500
        assert rec.quality_score == 0.85
        assert rec.timestamp > 0

    def test_record_default_timestamp(self):
        before = time.time()
        rec = MetricsRecord(
            model="m", task_type="t", latency_ms=0,
            tokens_in=0, tokens_out=0, quality_score=0,
        )
        after = time.time()
        assert before <= rec.timestamp <= after

    def test_latency_stats_defaults(self):
        stats = LatencyStats()
        assert stats.p50 == 0.0
        assert stats.p95 == 0.0
        assert stats.p99 == 0.0
        assert stats.mean == 0.0
        assert stats.count == 0


# ---------------------------------------------------------------------------
# Tests: Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Tests for YAML config loading."""

    def test_load_defaults_when_no_file(self, tmp_path):
        cfg = _load_config(tmp_path / "nonexistent.yaml")
        assert cfg["enabled"] is True
        assert cfg["retention_days"] == 7
        assert cfg["drift"]["threshold"] == 0.3

    def test_load_custom_config(self, tmp_path):
        path = _make_config(tmp_path, {
            "retention_days": 14,
            "drift": {"threshold": 0.5, "window_seconds": 1800},
        })
        cfg = _load_config(path)
        assert cfg["retention_days"] == 14
        assert cfg["drift"]["threshold"] == 0.5
        assert cfg["drift"]["window_seconds"] == 1800

    def test_load_partial_config(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text("enabled: false\n", encoding="utf-8")
        cfg = _load_config(path)
        assert cfg["enabled"] is False
        # Defaults preserved for unspecified keys
        assert cfg["retention_days"] == 7


# ---------------------------------------------------------------------------
# Tests: record_execution
# ---------------------------------------------------------------------------

class TestRecordExecution:
    """Tests for recording metrics."""

    def test_record_returns_record(self, tmp_path):
        mon = _make_monitor(tmp_path)
        rec = mon.record_execution(
            model="qwen3:32b", task_type="code_python",
            latency_ms=1500, tokens_in=200, tokens_out=800,
            quality_score=0.9,
        )
        assert isinstance(rec, MetricsRecord)
        assert rec.model == "qwen3:32b"
        assert rec.latency_ms == 1500

    def test_record_with_custom_timestamp(self, tmp_path):
        mon = _make_monitor(tmp_path)
        ts = 1700000000.0
        rec = mon.record_execution(
            model="m", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=50,
            quality_score=0.8, timestamp=ts,
        )
        assert rec.timestamp == ts

    def test_disabled_monitor_returns_none(self, tmp_path):
        mon = _make_monitor(tmp_path, {"enabled": False})
        rec = mon.record_execution(
            model="m", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=50,
            quality_score=0.8,
        )
        assert rec is None

    def test_multiple_records_stored(self, tmp_path):
        mon = _make_monitor(tmp_path)
        for i in range(5):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=100 * (i + 1), tokens_in=10, tokens_out=50,
                quality_score=0.8,
            )
        history = mon.get_history(hours=1)
        assert len(history) == 5


# ---------------------------------------------------------------------------
# Tests: get_throughput
# ---------------------------------------------------------------------------

class TestThroughput:
    """Tests for throughput calculation."""

    def test_empty_throughput(self, tmp_path):
        mon = _make_monitor(tmp_path)
        tp = mon.get_throughput()
        assert tp["tokens_in_per_sec"] == 0.0
        assert tp["tokens_out_per_sec"] == 0.0
        assert tp["total_tokens"] == 0
        assert tp["request_count"] == 0

    def test_throughput_calculation(self, tmp_path):
        mon = _make_monitor(tmp_path)
        # Insert 3 records, each with 100 tokens in, 200 out
        now = time.time()
        for i in range(3):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=100, tokens_in=100, tokens_out=200,
                quality_score=0.9, timestamp=now - i,
            )
        tp = mon.get_throughput(window_seconds=300)
        assert tp["total_tokens"] == 900  # 3 * (100 + 200)
        assert tp["request_count"] == 3
        assert tp["tokens_in_per_sec"] == pytest.approx(300 / 300, abs=0.1)
        assert tp["tokens_out_per_sec"] == pytest.approx(600 / 300, abs=0.1)

    def test_throughput_window_filter(self, tmp_path):
        mon = _make_monitor(tmp_path)
        now = time.time()
        # Old record
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=100, tokens_in=1000, tokens_out=1000,
            quality_score=0.9, timestamp=now - 600,
        )
        # Recent record
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=100, tokens_in=50, tokens_out=50,
            quality_score=0.9, timestamp=now,
        )
        tp = mon.get_throughput(window_seconds=300)
        # Only recent should count
        assert tp["request_count"] == 1
        assert tp["total_tokens"] == 100


# ---------------------------------------------------------------------------
# Tests: get_latency_stats
# ---------------------------------------------------------------------------

class TestLatencyStats:
    """Tests for latency percentile calculation."""

    def test_empty_latency_stats(self, tmp_path):
        mon = _make_monitor(tmp_path)
        stats = mon.get_latency_stats()
        assert stats.count == 0
        assert stats.p50 == 0.0

    def test_single_record_latency(self, tmp_path):
        mon = _make_monitor(tmp_path)
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=500, tokens_in=10, tokens_out=50,
            quality_score=0.9,
        )
        stats = mon.get_latency_stats()
        assert stats.count == 1
        assert stats.p50 == 500.0
        assert stats.p95 == 500.0
        assert stats.mean == 500.0

    def test_multiple_latencies_percentiles(self, tmp_path):
        mon = _make_monitor(tmp_path)
        # Insert 100 records with latency 1..100
        now = time.time()
        for i in range(1, 101):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=float(i), tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now,
            )
        stats = mon.get_latency_stats(window_seconds=300)
        assert stats.count == 100
        assert stats.p50 == pytest.approx(50.5, abs=1.0)
        assert stats.p95 >= 94.0
        assert stats.p99 >= 98.0

    def test_latency_per_model_filter(self, tmp_path):
        mon = _make_monitor(tmp_path)
        now = time.time()
        mon.record_execution(
            model="fast", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=10,
            quality_score=0.9, timestamp=now,
        )
        mon.record_execution(
            model="slow", task_type="t",
            latency_ms=5000, tokens_in=10, tokens_out=10,
            quality_score=0.9, timestamp=now,
        )
        fast_stats = mon.get_latency_stats(model="fast")
        slow_stats = mon.get_latency_stats(model="slow")
        assert fast_stats.mean == 100.0
        assert slow_stats.mean == 5000.0


# ---------------------------------------------------------------------------
# Tests: get_model_utilization
# ---------------------------------------------------------------------------

class TestUtilization:
    """Tests for model utilization distribution."""

    def test_empty_utilization(self, tmp_path):
        mon = _make_monitor(tmp_path)
        util = mon.get_model_utilization()
        assert util == {}

    def test_single_model_utilization(self, tmp_path):
        mon = _make_monitor(tmp_path)
        for _ in range(10):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=100, tokens_in=10, tokens_out=10,
                quality_score=0.9,
            )
        util = mon.get_model_utilization()
        assert util["m"] == pytest.approx(1.0)

    def test_multi_model_utilization(self, tmp_path):
        mon = _make_monitor(tmp_path)
        now = time.time()
        # 3 requests to model A, 1 to model B
        for _ in range(3):
            mon.record_execution(
                model="A", task_type="t",
                latency_ms=100, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now,
            )
        mon.record_execution(
            model="B", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=10,
            quality_score=0.9, timestamp=now,
        )
        util = mon.get_model_utilization()
        assert util["A"] == pytest.approx(0.75)
        assert util["B"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Tests: detect_drift
# ---------------------------------------------------------------------------

class TestDriftDetection:
    """Tests for drift detection."""

    def test_no_data_returns_none(self, tmp_path):
        mon = _make_monitor(tmp_path)
        result = mon.detect_drift("m", "latency")
        assert result is None

    def test_no_baseline_returns_none(self, tmp_path):
        """Only recent data, no baseline => None."""
        mon = _make_monitor(tmp_path)
        now = time.time()
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=10,
            quality_score=0.9, timestamp=now,
        )
        result = mon.detect_drift("m", "latency")
        assert result is None

    def test_latency_drift_detected(self, tmp_path):
        cfg_path = _make_config(tmp_path, {
            "drift": {
                "window_seconds": 100,
                "baseline_window_seconds": 1000,
                "threshold": 0.3,
            },
        })
        db_path = tmp_path / "drift.db"
        mon = PerformanceMonitor(db_path=db_path, config_path=cfg_path)

        now = time.time()
        # Baseline: 5 records at latency=1000 (old, outside recent window)
        for i in range(5):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=1000, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now - 500 - i,
            )
        # Recent: 5 records at latency=2000 (within recent window)
        for i in range(5):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=2000, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now - i,
            )

        result = mon.detect_drift("m", "latency")
        assert result is not None
        assert result.is_drifted is True
        assert result.direction == "up"
        assert result.change_ratio == pytest.approx(1.0, abs=0.01)

    def test_quality_drift_detected(self, tmp_path):
        cfg_path = _make_config(tmp_path, {
            "drift": {
                "window_seconds": 100,
                "baseline_window_seconds": 1000,
                "threshold": 0.2,
            },
        })
        db_path = tmp_path / "drift_q.db"
        mon = PerformanceMonitor(db_path=db_path, config_path=cfg_path)

        now = time.time()
        # Baseline: quality 0.9
        for i in range(5):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=100, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now - 500 - i,
            )
        # Recent: quality 0.5 (big drop)
        for i in range(5):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=100, tokens_in=10, tokens_out=10,
                quality_score=0.5, timestamp=now - i,
            )

        result = mon.detect_drift("m", "quality")
        assert result is not None
        assert result.is_drifted is True
        assert result.direction == "down"

    def test_no_drift_when_stable(self, tmp_path):
        cfg_path = _make_config(tmp_path, {
            "drift": {
                "window_seconds": 100,
                "baseline_window_seconds": 1000,
                "threshold": 0.3,
            },
        })
        db_path = tmp_path / "stable.db"
        mon = PerformanceMonitor(db_path=db_path, config_path=cfg_path)

        now = time.time()
        # Both baseline and recent have similar latency
        for i in range(5):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=1000, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now - 500 - i,
            )
        for i in range(5):
            mon.record_execution(
                model="m", task_type="t",
                latency_ms=1050, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now - i,
            )

        result = mon.detect_drift("m", "latency")
        assert result is not None
        assert result.is_drifted is False

    def test_invalid_metric_returns_none(self, tmp_path):
        mon = _make_monitor(tmp_path)
        result = mon.detect_drift("m", "nonexistent")
        assert result is None

    def test_detect_all_drift(self, tmp_path):
        cfg_path = _make_config(tmp_path, {
            "drift": {
                "window_seconds": 100,
                "baseline_window_seconds": 1000,
                "threshold": 0.3,
            },
        })
        db_path = tmp_path / "all_drift.db"
        mon = PerformanceMonitor(db_path=db_path, config_path=cfg_path)

        now = time.time()
        # Create a drifted model
        for i in range(5):
            mon.record_execution(
                model="drifty", task_type="t",
                latency_ms=1000, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now - 500 - i,
            )
        for i in range(5):
            mon.record_execution(
                model="drifty", task_type="t",
                latency_ms=3000, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now - i,
            )

        results = mon.detect_all_drift()
        assert len(results) >= 1
        assert any(r.model == "drifty" for r in results)


# ---------------------------------------------------------------------------
# Tests: get_recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:
    """Tests for rule-based recommendations."""

    def test_no_rules_no_recommendations(self, tmp_path):
        mon = _make_monitor(tmp_path)
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=10,
            quality_score=0.9,
        )
        recs = mon.get_recommendations()
        assert recs == []

    def test_latency_p95_rule_triggered(self, tmp_path):
        rules = [
            {
                "metric": "latency_p95",
                "condition": "gt",
                "threshold": 1000,
                "message": "Model {model} p95 latency too high",
            }
        ]
        cfg_path = _make_config(tmp_path, {"recommendation_rules": rules})
        db_path = tmp_path / "recs.db"
        mon = PerformanceMonitor(db_path=db_path, config_path=cfg_path)

        # Insert high-latency records
        now = time.time()
        for i in range(10):
            mon.record_execution(
                model="slow_model", task_type="t",
                latency_ms=5000, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now,
            )

        recs = mon.get_recommendations()
        assert len(recs) >= 1
        assert any("slow_model" in r.message for r in recs)

    def test_utilization_rule_triggered(self, tmp_path):
        rules = [
            {
                "metric": "utilization",
                "condition": "gt",
                "threshold": 0.5,
                "message": "Model {model} over-utilized at {value:.0%}",
            }
        ]
        cfg_path = _make_config(tmp_path, {"recommendation_rules": rules})
        db_path = tmp_path / "recs_util.db"
        mon = PerformanceMonitor(db_path=db_path, config_path=cfg_path)

        now = time.time()
        for _ in range(10):
            mon.record_execution(
                model="popular", task_type="t",
                latency_ms=100, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now,
            )

        recs = mon.get_recommendations()
        assert len(recs) >= 1
        assert any(r.metric == "utilization" for r in recs)

    def test_rule_not_triggered(self, tmp_path):
        rules = [
            {
                "metric": "latency_p95",
                "condition": "gt",
                "threshold": 10000,
                "message": "Too slow",
            }
        ]
        cfg_path = _make_config(tmp_path, {"recommendation_rules": rules})
        db_path = tmp_path / "recs_ok.db"
        mon = PerformanceMonitor(db_path=db_path, config_path=cfg_path)

        now = time.time()
        for _ in range(5):
            mon.record_execution(
                model="fast", task_type="t",
                latency_ms=100, tokens_in=10, tokens_out=10,
                quality_score=0.9, timestamp=now,
            )

        recs = mon.get_recommendations()
        assert recs == []


# ---------------------------------------------------------------------------
# Tests: get_history
# ---------------------------------------------------------------------------

class TestHistory:
    """Tests for raw metric history retrieval."""

    def test_empty_history(self, tmp_path):
        mon = _make_monitor(tmp_path)
        history = mon.get_history()
        assert history == []

    def test_history_returns_records(self, tmp_path):
        mon = _make_monitor(tmp_path)
        mon.record_execution(
            model="m", task_type="code",
            latency_ms=500, tokens_in=100, tokens_out=200,
            quality_score=0.85,
        )
        history = mon.get_history(hours=1)
        assert len(history) == 1
        assert history[0]["model"] == "m"
        assert history[0]["latency_ms"] == 500

    def test_history_model_filter(self, tmp_path):
        mon = _make_monitor(tmp_path)
        mon.record_execution(
            model="A", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=10,
            quality_score=0.9,
        )
        mon.record_execution(
            model="B", task_type="t",
            latency_ms=200, tokens_in=10, tokens_out=10,
            quality_score=0.9,
        )
        history = mon.get_history(model="A", hours=1)
        assert len(history) == 1
        assert history[0]["model"] == "A"

    def test_history_hours_filter(self, tmp_path):
        mon = _make_monitor(tmp_path)
        now = time.time()
        # Old record
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=10,
            quality_score=0.9, timestamp=now - 7200,
        )
        # Recent record
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=200, tokens_in=10, tokens_out=10,
            quality_score=0.9, timestamp=now,
        )
        history = mon.get_history(hours=1)
        assert len(history) == 1
        assert history[0]["latency_ms"] == 200


# ---------------------------------------------------------------------------
# Tests: cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    """Tests for retention cleanup."""

    def test_cleanup_old_records(self, tmp_path):
        cfg_path = _make_config(tmp_path, {"retention_days": 1})
        db_path = tmp_path / "cleanup.db"
        mon = PerformanceMonitor(db_path=db_path, config_path=cfg_path)

        now = time.time()
        # Old record (2 days ago)
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=10,
            quality_score=0.9, timestamp=now - 172800,
        )
        # Recent record
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=200, tokens_in=10, tokens_out=10,
            quality_score=0.9, timestamp=now,
        )

        deleted = mon.cleanup_old_records()
        assert deleted == 1

        history = mon.get_history(hours=72)
        assert len(history) == 1


# ---------------------------------------------------------------------------
# Tests: get_summary
# ---------------------------------------------------------------------------

class TestSummary:
    """Tests for performance summary."""

    def test_summary_structure(self, tmp_path):
        mon = _make_monitor(tmp_path)
        summary = mon.get_summary()
        assert "throughput" in summary
        assert "latency" in summary
        assert "utilization" in summary
        assert "enabled" in summary
        assert summary["enabled"] is True

    def test_summary_with_data(self, tmp_path):
        mon = _make_monitor(tmp_path)
        mon.record_execution(
            model="m", task_type="t",
            latency_ms=1000, tokens_in=100, tokens_out=200,
            quality_score=0.9,
        )
        summary = mon.get_summary()
        assert summary["throughput"]["request_count"] == 1
        assert summary["latency"]["count"] == 1
        assert "m" in summary["utilization"]


# ---------------------------------------------------------------------------
# Tests: Properties and toggle
# ---------------------------------------------------------------------------

class TestProperties:
    """Tests for enabled property and config access."""

    def test_enabled_default(self, tmp_path):
        mon = _make_monitor(tmp_path)
        assert mon.enabled is True

    def test_disable_toggle(self, tmp_path):
        mon = _make_monitor(tmp_path)
        mon.enabled = False
        assert mon.enabled is False
        rec = mon.record_execution(
            model="m", task_type="t",
            latency_ms=100, tokens_in=10, tokens_out=10,
            quality_score=0.9,
        )
        assert rec is None

    def test_config_property(self, tmp_path):
        mon = _make_monitor(tmp_path)
        cfg = mon.config
        assert isinstance(cfg, dict)
        assert "enabled" in cfg
        assert "drift" in cfg


# ---------------------------------------------------------------------------
# Tests: Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    """Tests for module-level singleton."""

    def test_module_exports(self):
        assert hasattr(_mod, "performance_monitor")
        assert hasattr(_mod, "PERFORMANCE_MONITOR_AVAILABLE")
        assert _mod.PERFORMANCE_MONITOR_AVAILABLE is True
