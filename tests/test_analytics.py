#!/usr/bin/env python3
"""
TESTS -- Analytics Engine & Performance Tracker (S55)
=======================================================

Comprehensive tests for PerformanceRecord, PerformanceTracker,
AnalyticsEngine overview, trends, routing accuracy, and cleanup.

Target: 35+ tests, zero regressions.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from opti_oignon.analytics import (
    AnalyticsEngine,
    AnalyticsOverview,
    PerformanceRecord,
    PerformanceTracker,
    TrendPoint,
    _parse_window,
)

# =============================================================================
# HELPERS
# =============================================================================

def _temp_tracker() -> PerformanceTracker:
    """Create a PerformanceTracker with a temporary database."""
    tmp = tempfile.mktemp(suffix=".db")
    return PerformanceTracker(db_path=Path(tmp))


def _temp_engine() -> AnalyticsEngine:
    """Create an AnalyticsEngine with a temporary tracker."""
    tracker = _temp_tracker()
    return AnalyticsEngine(tracker=tracker, config_path=Path("/dev/null"))


def _make_record(**kwargs) -> PerformanceRecord:
    """Create a PerformanceRecord with sensible defaults."""
    defaults = dict(
        model_used="qwen3:32b",
        pipeline_used="direct",
        task_type="code",
        response_time_ms=500.0,
        prompt_tokens=100,
        completion_tokens=200,
        success=True,
    )
    defaults.update(kwargs)
    return PerformanceRecord(**defaults)


# =============================================================================
# TEST: _parse_window utility
# =============================================================================

class TestParseWindow(unittest.TestCase):
    """Tests for the _parse_window helper."""

    def test_seconds(self):
        self.assertEqual(_parse_window("30s"), 30)

    def test_minutes(self):
        self.assertEqual(_parse_window("5m"), 300)

    def test_hours(self):
        self.assertEqual(_parse_window("24h"), 86400)

    def test_days(self):
        self.assertEqual(_parse_window("7d"), 604800)

    def test_weeks(self):
        self.assertEqual(_parse_window("2w"), 1209600)

    def test_invalid_suffix(self):
        with self.assertRaises(ValueError):
            _parse_window("10x")

    def test_invalid_format_empty(self):
        with self.assertRaises(ValueError):
            _parse_window("")

    def test_invalid_format_single_char(self):
        with self.assertRaises(ValueError):
            _parse_window("h")

    def test_invalid_numeric(self):
        with self.assertRaises(ValueError):
            _parse_window("abch")


# =============================================================================
# TEST: PerformanceRecord dataclass
# =============================================================================

class TestPerformanceRecord(unittest.TestCase):
    """Tests for the PerformanceRecord dataclass."""

    def test_auto_generates_id(self):
        """Record auto-generates a record_id."""
        r = PerformanceRecord()
        self.assertTrue(len(r.record_id) > 0)

    def test_auto_generates_timestamp(self):
        """Record auto-generates a timestamp."""
        r = PerformanceRecord()
        self.assertGreater(r.timestamp, 0)

    def test_auto_computes_tokens_per_second(self):
        """tokens_per_second is auto-computed from completion_tokens and response_time_ms."""
        r = PerformanceRecord(
            completion_tokens=100,
            response_time_ms=2000.0,
        )
        self.assertAlmostEqual(r.tokens_per_second, 50.0, places=1)

    def test_auto_computes_total_tokens(self):
        """total_tokens is auto-computed from prompt + completion."""
        r = PerformanceRecord(prompt_tokens=50, completion_tokens=100)
        self.assertEqual(r.total_tokens, 150)

    def test_to_dict(self):
        """to_dict produces a serializable dict."""
        r = _make_record()
        d = r.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("record_id", d)
        json.dumps(d)

    def test_from_dict_roundtrip(self):
        """from_dict correctly reconstructs a record."""
        r = _make_record()
        d = r.to_dict()
        r2 = PerformanceRecord.from_dict(d)
        self.assertEqual(r.record_id, r2.record_id)
        self.assertEqual(r.model_used, r2.model_used)

    def test_from_dict_ignores_unknown(self):
        """from_dict ignores unknown keys."""
        d = {"model_used": "test", "unknown_field": 42}
        r = PerformanceRecord.from_dict(d)
        self.assertEqual(r.model_used, "test")


# =============================================================================
# TEST: PerformanceTracker CRUD
# =============================================================================

class TestPerformanceTracker(unittest.TestCase):
    """Tests for PerformanceTracker operations."""

    def setUp(self):
        self.tracker = _temp_tracker()

    def tearDown(self):
        try:
            os.unlink(self.tracker._db_path)
        except OSError:
            pass

    def test_record_and_count(self):
        """record stores entry and count reflects it."""
        self.assertEqual(self.tracker.count(), 0)
        self.tracker.record(_make_record())
        self.assertEqual(self.tracker.count(), 1)

    def test_record_multiple(self):
        """Multiple records are stored correctly."""
        for _ in range(10):
            self.tracker.record(_make_record())
        self.assertEqual(self.tracker.count(), 10)

    def test_get_records_default(self):
        """get_records returns all records."""
        for i in range(5):
            self.tracker.record(_make_record(response_time_ms=100 + i * 10))
        records = self.tracker.get_records()
        self.assertEqual(len(records), 5)

    def test_get_records_limit(self):
        """get_records respects limit."""
        for _ in range(10):
            self.tracker.record(_make_record())
        records = self.tracker.get_records(limit=3)
        self.assertEqual(len(records), 3)

    def test_get_records_filter_model(self):
        """get_records filters by model."""
        self.tracker.record(_make_record(model_used="model-a"))
        self.tracker.record(_make_record(model_used="model-b"))
        self.tracker.record(_make_record(model_used="model-a"))
        records = self.tracker.get_records(model="model-a")
        self.assertEqual(len(records), 2)

    def test_get_records_filter_pipeline(self):
        """get_records filters by pipeline."""
        self.tracker.record(_make_record(pipeline_used="direct"))
        self.tracker.record(_make_record(pipeline_used="think"))
        records = self.tracker.get_records(pipeline="think")
        self.assertEqual(len(records), 1)

    def test_get_records_filter_since(self):
        """get_records filters by timestamp."""
        now = time.time()
        for i in range(5):
            r = _make_record()
            r.timestamp = now - (4 - i) * 100
            self.tracker.record(r)
        records = self.tracker.get_records(since=now - 250)
        self.assertEqual(len(records), 3)

    def test_clear_all(self):
        """clear without argument removes all records."""
        for _ in range(5):
            self.tracker.record(_make_record())
        deleted = self.tracker.clear()
        self.assertEqual(deleted, 5)
        self.assertEqual(self.tracker.count(), 0)

    def test_clear_before(self):
        """clear with before removes only old records."""
        now = time.time()
        for i in range(4):
            r = _make_record()
            r.timestamp = now - (3 - i) * 100
            self.tracker.record(r)
        deleted = self.tracker.clear(before=now - 150)
        self.assertEqual(deleted, 2)
        self.assertEqual(self.tracker.count(), 2)

    def test_boolean_roundtrip(self):
        """was_routed and success booleans survive storage."""
        r = _make_record(was_routed=True, success=False, error_message="timeout")
        self.tracker.record(r)
        records = self.tracker.get_records()
        self.assertTrue(records[0].was_routed)
        self.assertFalse(records[0].success)

    def test_count_with_since(self):
        """count with since filter works."""
        now = time.time()
        for i in range(5):
            r = _make_record()
            r.timestamp = now - (4 - i) * 100
            self.tracker.record(r)
        self.assertEqual(self.tracker.count(since=now - 250), 3)


# =============================================================================
# TEST: AnalyticsEngine Overview
# =============================================================================

class TestAnalyticsEngineOverview(unittest.TestCase):
    """Tests for AnalyticsEngine.get_overview()."""

    def setUp(self):
        self.engine = _temp_engine()
        self.tracker = self.engine.tracker
        # Add diverse data
        self.tracker.record(_make_record(
            model_used="m1", pipeline_used="direct",
            response_time_ms=200, completion_tokens=100, success=True
        ))
        self.tracker.record(_make_record(
            model_used="m1", pipeline_used="think",
            response_time_ms=600, completion_tokens=300, success=True
        ))
        self.tracker.record(_make_record(
            model_used="m2", pipeline_used="direct",
            response_time_ms=150, completion_tokens=80, success=False,
            error_message="timeout"
        ))

    def tearDown(self):
        try:
            os.unlink(self.tracker._db_path)
        except OSError:
            pass

    def test_total_requests(self):
        ov = self.engine.get_overview()
        self.assertEqual(ov.total_requests, 3)

    def test_success_count(self):
        ov = self.engine.get_overview()
        self.assertEqual(ov.success_count, 2)
        self.assertEqual(ov.error_count, 1)

    def test_success_rate(self):
        ov = self.engine.get_overview()
        self.assertAlmostEqual(ov.success_rate, 2 / 3, places=3)

    def test_avg_response_time(self):
        ov = self.engine.get_overview()
        expected = (200 + 600 + 150) / 3
        self.assertAlmostEqual(ov.avg_response_time_ms, expected, places=1)

    def test_model_distribution(self):
        ov = self.engine.get_overview()
        self.assertEqual(ov.model_distribution.get("m1"), 2)
        self.assertEqual(ov.model_distribution.get("m2"), 1)

    def test_pipeline_distribution(self):
        ov = self.engine.get_overview()
        self.assertEqual(ov.pipeline_distribution.get("direct"), 2)
        self.assertEqual(ov.pipeline_distribution.get("think"), 1)

    def test_model_performance(self):
        ov = self.engine.get_overview()
        self.assertIn("m1", ov.model_performance)
        self.assertEqual(ov.model_performance["m1"]["count"], 2)

    def test_pipeline_performance(self):
        ov = self.engine.get_overview()
        self.assertIn("direct", ov.pipeline_performance)

    def test_to_dict(self):
        ov = self.engine.get_overview()
        d = ov.to_dict()
        self.assertIsInstance(d, dict)
        json.dumps(d)

    def test_empty_overview(self):
        """Overview on empty tracker returns zeros."""
        empty_engine = _temp_engine()
        ov = empty_engine.get_overview()
        self.assertEqual(ov.total_requests, 0)
        self.assertEqual(ov.success_rate, 0.0)
        try:
            os.unlink(empty_engine.tracker._db_path)
        except OSError:
            pass


# =============================================================================
# TEST: AnalyticsEngine Trends
# =============================================================================

class TestAnalyticsEngineTrends(unittest.TestCase):
    """Tests for AnalyticsEngine.get_trends()."""

    def setUp(self):
        self.engine = _temp_engine()
        self.tracker = self.engine.tracker
        now = time.time()
        # Spread records over the last hour
        for i in range(12):
            r = _make_record(response_time_ms=100 + i * 50)
            r.timestamp = now - (11 - i) * 300  # every 5 min
            self.tracker.record(r)

    def tearDown(self):
        try:
            os.unlink(self.tracker._db_path)
        except OSError:
            pass

    def test_trend_bucket_count(self):
        """get_trends returns the correct number of buckets."""
        trends = self.engine.get_trends("1h", buckets=6)
        self.assertEqual(len(trends), 6)

    def test_trend_total_count(self):
        """Total count across buckets equals total records."""
        trends = self.engine.get_trends("1h", buckets=12)
        total = sum(t.count for t in trends)
        self.assertEqual(total, 12)

    def test_trend_points_are_ordered(self):
        """Trend points are in chronological order."""
        trends = self.engine.get_trends("1h", buckets=6)
        for i in range(len(trends) - 1):
            self.assertLessEqual(trends[i].window_start, trends[i + 1].window_start)

    def test_trend_with_model_filter(self):
        """get_trends respects model filter."""
        self.tracker.record(_make_record(model_used="special-model"))
        trends = self.engine.get_trends("1h", buckets=4, model="special-model")
        total = sum(t.count for t in trends)
        self.assertEqual(total, 1)

    def test_trend_empty_buckets(self):
        """Empty buckets have count=0 and default values."""
        engine = _temp_engine()
        trends = engine.get_trends("1h", buckets=4)
        for t in trends:
            self.assertEqual(t.count, 0)
            self.assertEqual(t.avg_response_time_ms, 0.0)
        try:
            os.unlink(engine.tracker._db_path)
        except OSError:
            pass


# =============================================================================
# TEST: AnalyticsEngine Routing Accuracy
# =============================================================================

class TestAnalyticsEngineRouting(unittest.TestCase):
    """Tests for routing accuracy computation."""

    def setUp(self):
        self.engine = _temp_engine()
        self.tracker = self.engine.tracker
        # Routed requests
        self.tracker.record(_make_record(was_routed=True, success=True, response_time_ms=200))
        self.tracker.record(_make_record(was_routed=True, success=True, response_time_ms=300))
        # Unrouted requests
        self.tracker.record(_make_record(was_routed=False, success=True, response_time_ms=500))
        self.tracker.record(_make_record(was_routed=False, success=False, response_time_ms=800))

    def tearDown(self):
        try:
            os.unlink(self.tracker._db_path)
        except OSError:
            pass

    def test_routed_count(self):
        ra = self.engine.get_routing_accuracy()
        self.assertEqual(ra["routed"]["count"], 2)

    def test_unrouted_count(self):
        ra = self.engine.get_routing_accuracy()
        self.assertEqual(ra["unrouted"]["count"], 2)

    def test_routed_success_rate(self):
        ra = self.engine.get_routing_accuracy()
        self.assertAlmostEqual(ra["routed"]["success_rate"], 1.0)

    def test_unrouted_success_rate(self):
        ra = self.engine.get_routing_accuracy()
        self.assertAlmostEqual(ra["unrouted"]["success_rate"], 0.5)

    def test_routed_avg_response_time(self):
        ra = self.engine.get_routing_accuracy()
        self.assertAlmostEqual(ra["routed"]["avg_response_time_ms"], 250.0)


# =============================================================================
# TEST: AnalyticsEngine Cleanup
# =============================================================================

class TestAnalyticsEngineCleanup(unittest.TestCase):
    """Tests for analytics cleanup."""

    def setUp(self):
        self.engine = _temp_engine()
        self.tracker = self.engine.tracker

    def tearDown(self):
        try:
            os.unlink(self.tracker._db_path)
        except OSError:
            pass

    def test_cleanup_removes_old_records(self):
        """cleanup_old_records removes records beyond retention."""
        now = time.time()
        # Old record (beyond default retention)
        old = _make_record()
        old.timestamp = now - 9999999
        self.tracker.record(old)
        # Recent record
        self.tracker.record(_make_record())

        deleted = self.engine.cleanup_old_records()
        self.assertEqual(deleted, 1)
        self.assertEqual(self.tracker.count(), 1)

    def test_cleanup_no_old_records(self):
        """cleanup_old_records returns 0 when nothing to clean."""
        self.tracker.record(_make_record())
        deleted = self.engine.cleanup_old_records()
        self.assertEqual(deleted, 0)


# =============================================================================
# TEST: Config loading
# =============================================================================

class TestAnalyticsConfig(unittest.TestCase):
    """Tests for AnalyticsEngine configuration."""

    def test_default_enabled(self):
        """Engine is enabled by default."""
        engine = _temp_engine()
        self.assertTrue(engine.enabled)
        try:
            os.unlink(engine.tracker._db_path)
        except OSError:
            pass

    def test_default_retention(self):
        """Default retention is 30 days."""
        engine = _temp_engine()
        self.assertEqual(engine.retention_seconds, 2592000)
        try:
            os.unlink(engine.tracker._db_path)
        except OSError:
            pass

    def test_default_trend_windows(self):
        """Default trend windows include 1h, 24h, 7d, 30d."""
        engine = _temp_engine()
        windows = engine.trend_windows
        self.assertIn("1h", windows)
        self.assertIn("24h", windows)
        try:
            os.unlink(engine.tracker._db_path)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
