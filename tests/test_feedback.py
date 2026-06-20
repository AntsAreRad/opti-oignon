#!/usr/bin/env python3
"""
TESTS -- Feedback Store & API (S55)
=====================================

Comprehensive tests for FeedbackEntry, FeedbackStore CRUD,
aggregation, export, and API endpoint integration.

Target: 35+ tests, zero regressions.
"""

import csv
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from opti_oignon.feedback import (
    MAX_STARS,
    MIN_STARS,
    RATING_TYPE_STARS,
    RATING_TYPE_THUMBS,
    THUMBS_DOWN,
    THUMBS_UP,
    FeedbackEntry,
    FeedbackStats,
    FeedbackStore,
)

# =============================================================================
# HELPERS
# =============================================================================

def _temp_store() -> FeedbackStore:
    """Create a FeedbackStore with a temporary database."""
    tmp = tempfile.mktemp(suffix=".db")
    return FeedbackStore(db_path=Path(tmp), config_path=Path("/dev/null"))


def _make_entry(**kwargs) -> FeedbackEntry:
    """Create a FeedbackEntry with sensible defaults."""
    defaults = dict(
        conversation_id="conv-001",
        message_id="msg-001",
        rating_type=RATING_TYPE_THUMBS,
        rating_value=THUMBS_UP,
        model_used="qwen3:32b",
        pipeline_used="direct",
        task_type="code",
    )
    defaults.update(kwargs)
    return FeedbackEntry(**defaults)


# =============================================================================
# TEST: FeedbackEntry dataclass
# =============================================================================

class TestFeedbackEntry(unittest.TestCase):
    """Tests for the FeedbackEntry dataclass."""

    def test_auto_generates_id(self):
        """Entry auto-generates a feedback_id."""
        e = FeedbackEntry()
        self.assertTrue(len(e.feedback_id) > 0)

    def test_auto_generates_timestamp(self):
        """Entry auto-generates a timestamp."""
        e = FeedbackEntry()
        self.assertGreater(e.timestamp, 0)

    def test_preserves_explicit_id(self):
        """Explicit feedback_id is preserved."""
        e = FeedbackEntry(feedback_id="custom-id")
        self.assertEqual(e.feedback_id, "custom-id")

    def test_validate_thumbs_up(self):
        """Thumbs up (value=1) passes validation."""
        e = _make_entry(rating_type=RATING_TYPE_THUMBS, rating_value=THUMBS_UP)
        valid, msg = e.validate()
        self.assertTrue(valid)

    def test_validate_thumbs_down(self):
        """Thumbs down (value=0) passes validation."""
        e = _make_entry(rating_type=RATING_TYPE_THUMBS, rating_value=THUMBS_DOWN)
        valid, msg = e.validate()
        self.assertTrue(valid)

    def test_validate_thumbs_invalid_value(self):
        """Invalid thumbs value (e.g. 5) fails validation."""
        e = _make_entry(rating_type=RATING_TYPE_THUMBS, rating_value=5)
        valid, msg = e.validate()
        self.assertFalse(valid)
        self.assertIn("0 or 1", msg)

    def test_validate_stars_valid_range(self):
        """Star ratings 1-5 pass validation."""
        for v in range(MIN_STARS, MAX_STARS + 1):
            e = _make_entry(rating_type=RATING_TYPE_STARS, rating_value=v)
            valid, _ = e.validate()
            self.assertTrue(valid, f"Star rating {v} should be valid")

    def test_validate_stars_out_of_range(self):
        """Star rating 0 or 6 fails validation."""
        for v in [0, 6, -1, 10]:
            e = _make_entry(rating_type=RATING_TYPE_STARS, rating_value=v)
            valid, _ = e.validate()
            self.assertFalse(valid, f"Star rating {v} should be invalid")

    def test_validate_invalid_rating_type(self):
        """Invalid rating_type fails validation."""
        e = _make_entry(rating_type="emoji")
        valid, msg = e.validate()
        self.assertFalse(valid)
        self.assertIn("rating_type", msg)

    def test_is_positive_thumbs_up(self):
        """is_positive returns True for thumbs up."""
        e = _make_entry(rating_value=THUMBS_UP)
        self.assertTrue(e.is_positive)

    def test_is_positive_thumbs_down(self):
        """is_positive returns False for thumbs down."""
        e = _make_entry(rating_value=THUMBS_DOWN)
        self.assertFalse(e.is_positive)

    def test_is_positive_stars_4(self):
        """is_positive returns True for 4+ stars."""
        e = _make_entry(rating_type=RATING_TYPE_STARS, rating_value=4)
        self.assertTrue(e.is_positive)

    def test_is_positive_stars_2(self):
        """is_positive returns False for 2 stars."""
        e = _make_entry(rating_type=RATING_TYPE_STARS, rating_value=2)
        self.assertFalse(e.is_positive)

    def test_normalized_score_thumbs(self):
        """normalized_score is 0.0 or 1.0 for thumbs."""
        self.assertEqual(_make_entry(rating_value=THUMBS_UP).normalized_score, 1.0)
        self.assertEqual(_make_entry(rating_value=THUMBS_DOWN).normalized_score, 0.0)

    def test_normalized_score_stars(self):
        """normalized_score maps 1-5 stars to 0.0-1.0."""
        e1 = _make_entry(rating_type=RATING_TYPE_STARS, rating_value=1)
        e5 = _make_entry(rating_type=RATING_TYPE_STARS, rating_value=5)
        e3 = _make_entry(rating_type=RATING_TYPE_STARS, rating_value=3)
        self.assertAlmostEqual(e1.normalized_score, 0.0)
        self.assertAlmostEqual(e5.normalized_score, 1.0)
        self.assertAlmostEqual(e3.normalized_score, 0.5)

    def test_to_dict_roundtrip(self):
        """to_dict and from_dict produce equivalent entries."""
        e = _make_entry(feedback_text="Good answer")
        d = e.to_dict()
        e2 = FeedbackEntry.from_dict(d)
        self.assertEqual(e.feedback_id, e2.feedback_id)
        self.assertEqual(e.feedback_text, e2.feedback_text)
        self.assertEqual(e.model_used, e2.model_used)

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict ignores keys not in the dataclass."""
        d = {"rating_value": 1, "unknown_field": "ignored"}
        e = FeedbackEntry.from_dict(d)
        self.assertEqual(e.rating_value, 1)


# =============================================================================
# TEST: FeedbackStore CRUD
# =============================================================================

class TestFeedbackStoreCRUD(unittest.TestCase):
    """Tests for FeedbackStore CRUD operations."""

    def setUp(self):
        self.store = _temp_store()

    def tearDown(self):
        try:
            os.unlink(self.store._db_path)
        except OSError:
            pass

    def test_add_and_get(self):
        """add_feedback + get_feedback roundtrip."""
        e = _make_entry()
        stored = self.store.add_feedback(e)
        got = self.store.get_feedback(stored.feedback_id)
        self.assertIsNotNone(got)
        self.assertEqual(got.model_used, "qwen3:32b")

    def test_add_invalid_entry_raises(self):
        """add_feedback raises ValueError for invalid entries."""
        e = _make_entry(rating_type="bad")
        with self.assertRaises(ValueError):
            self.store.add_feedback(e)

    def test_get_nonexistent_returns_none(self):
        """get_feedback returns None for unknown ID."""
        self.assertIsNone(self.store.get_feedback("nonexistent"))

    def test_delete_feedback(self):
        """delete_feedback removes the entry."""
        e = _make_entry()
        self.store.add_feedback(e)
        self.assertTrue(self.store.delete_feedback(e.feedback_id))
        self.assertIsNone(self.store.get_feedback(e.feedback_id))

    def test_delete_nonexistent_returns_false(self):
        """delete_feedback returns False for unknown ID."""
        self.assertFalse(self.store.delete_feedback("nonexistent"))

    def test_count(self):
        """count returns the correct number of entries."""
        self.assertEqual(self.store.count(), 0)
        self.store.add_feedback(_make_entry())
        self.store.add_feedback(_make_entry())
        self.assertEqual(self.store.count(), 2)

    def test_clear(self):
        """clear removes all entries."""
        for _ in range(5):
            self.store.add_feedback(_make_entry())
        self.assertEqual(self.store.count(), 5)
        deleted = self.store.clear()
        self.assertEqual(deleted, 5)
        self.assertEqual(self.store.count(), 0)

    def test_list_feedback_default(self):
        """list_feedback returns entries in reverse chronological order."""
        for i in range(3):
            e = _make_entry(feedback_text=f"entry-{i}")
            e.timestamp = time.time() + i
            self.store.add_feedback(e)
        entries = self.store.list_feedback()
        self.assertEqual(len(entries), 3)
        # Most recent first
        self.assertGreaterEqual(entries[0].timestamp, entries[1].timestamp)

    def test_list_feedback_limit(self):
        """list_feedback respects limit parameter."""
        for _ in range(10):
            self.store.add_feedback(_make_entry())
        entries = self.store.list_feedback(limit=3)
        self.assertEqual(len(entries), 3)

    def test_list_feedback_time_filter(self):
        """list_feedback filters by since/until."""
        now = time.time()
        for i in range(5):
            e = _make_entry()
            e.timestamp = now - (4 - i) * 100
            self.store.add_feedback(e)
        # Only entries in the last 250 seconds
        entries = self.store.list_feedback(since=now - 250)
        self.assertEqual(len(entries), 3)

    def test_list_by_model(self):
        """list_by_model filters correctly."""
        self.store.add_feedback(_make_entry(model_used="model-a"))
        self.store.add_feedback(_make_entry(model_used="model-b"))
        self.store.add_feedback(_make_entry(model_used="model-a"))
        entries = self.store.list_by_model("model-a")
        self.assertEqual(len(entries), 2)
        for e in entries:
            self.assertEqual(e.model_used, "model-a")

    def test_list_by_pipeline(self):
        """list_by_pipeline filters correctly."""
        self.store.add_feedback(_make_entry(pipeline_used="direct"))
        self.store.add_feedback(_make_entry(pipeline_used="think"))
        entries = self.store.list_by_pipeline("think")
        self.assertEqual(len(entries), 1)

    def test_list_by_conversation(self):
        """list_by_conversation returns all entries for a conversation."""
        self.store.add_feedback(_make_entry(conversation_id="conv-A"))
        self.store.add_feedback(_make_entry(conversation_id="conv-B"))
        self.store.add_feedback(_make_entry(conversation_id="conv-A"))
        entries = self.store.list_by_conversation("conv-A")
        self.assertEqual(len(entries), 2)

    def test_text_truncation(self):
        """Feedback text is truncated to max_text_length."""
        long_text = "x" * 5000
        e = _make_entry(feedback_text=long_text)
        stored = self.store.add_feedback(e)
        got = self.store.get_feedback(stored.feedback_id)
        self.assertEqual(len(got.feedback_text), self.store.max_text_length)


# =============================================================================
# TEST: FeedbackStore Aggregation
# =============================================================================

class TestFeedbackStoreAggregation(unittest.TestCase):
    """Tests for FeedbackStore aggregation methods."""

    def setUp(self):
        self.store = _temp_store()
        # Populate with mixed feedback
        self.store.add_feedback(_make_entry(model_used="m1", pipeline_used="direct", rating_value=THUMBS_UP))
        self.store.add_feedback(_make_entry(model_used="m1", pipeline_used="think", rating_value=THUMBS_DOWN))
        self.store.add_feedback(_make_entry(model_used="m2", pipeline_used="direct", rating_value=THUMBS_UP))
        self.store.add_feedback(_make_entry(
            model_used="m1", pipeline_used="direct",
            rating_type=RATING_TYPE_STARS, rating_value=4, task_type="code"
        ))

    def tearDown(self):
        try:
            os.unlink(self.store._db_path)
        except OSError:
            pass

    def test_average_rating_by_model(self):
        """Aggregation by model produces correct groups."""
        result = self.store.average_rating_by_model()
        self.assertIn("m1", result)
        self.assertIn("m2", result)
        self.assertEqual(result["m1"]["total"], 3)
        self.assertEqual(result["m2"]["total"], 1)

    def test_average_rating_by_pipeline(self):
        """Aggregation by pipeline produces correct groups."""
        result = self.store.average_rating_by_pipeline()
        self.assertIn("direct", result)
        self.assertIn("think", result)
        self.assertEqual(result["direct"]["total"], 3)

    def test_average_rating_by_task_type(self):
        """Aggregation by task_type filters empty values."""
        result = self.store.average_rating_by_task_type()
        self.assertIn("code", result)

    def test_get_stats_total(self):
        """get_stats returns correct total count."""
        stats = self.store.get_stats()
        self.assertEqual(stats.total_count, 4)

    def test_get_stats_thumbs(self):
        """get_stats returns correct thumbs up/down counts."""
        stats = self.store.get_stats()
        self.assertEqual(stats.thumbs_up, 2)
        self.assertEqual(stats.thumbs_down, 1)

    def test_get_stats_positive_negative(self):
        """get_stats computes positive/negative including stars."""
        stats = self.store.get_stats()
        # 2 thumbs up + 1 star(4) = 3 positive
        self.assertEqual(stats.positive_count, 3)
        # 1 thumbs down = 1 negative
        self.assertEqual(stats.negative_count, 1)

    def test_get_stats_star_distribution(self):
        """get_stats includes star rating distribution."""
        stats = self.store.get_stats()
        self.assertIn(4, stats.star_distribution)
        self.assertEqual(stats.star_distribution[4], 1)

    def test_get_stats_by_model(self):
        """get_stats includes per-model breakdown."""
        stats = self.store.get_stats()
        self.assertIn("m1", stats.by_model)
        self.assertIn("m2", stats.by_model)

    def test_get_stats_to_dict(self):
        """FeedbackStats.to_dict produces a serializable dict."""
        stats = self.store.get_stats()
        d = stats.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("total_count", d)
        # JSON-serializable
        json.dumps(d)


# =============================================================================
# TEST: FeedbackStore Export
# =============================================================================

class TestFeedbackStoreExport(unittest.TestCase):
    """Tests for feedback export to JSON and CSV."""

    def setUp(self):
        self.store = _temp_store()
        self.store.add_feedback(_make_entry(feedback_text="Great"))
        self.store.add_feedback(_make_entry(feedback_text="Poor", rating_value=THUMBS_DOWN))

    def tearDown(self):
        try:
            os.unlink(self.store._db_path)
        except OSError:
            pass

    def test_export_json(self):
        """export_json produces valid JSON with all entries."""
        data = json.loads(self.store.export_json())
        self.assertEqual(len(data), 2)
        self.assertIn("feedback_id", data[0])

    def test_export_csv(self):
        """export_csv produces valid CSV with headers."""
        csv_str = self.store.export_csv()
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertIn("feedback_id", rows[0])

    def test_export_empty_csv(self):
        """export_csv returns empty string for empty store."""
        self.store.clear()
        self.assertEqual(self.store.export_csv(), "")

    def test_export_json_empty(self):
        """export_json returns empty array for empty store."""
        self.store.clear()
        data = json.loads(self.store.export_json())
        self.assertEqual(data, [])


# =============================================================================
# TEST: API Endpoints
# =============================================================================

class TestFeedbackAPI(unittest.TestCase):
    """Tests for feedback and analytics API endpoints via TestClient."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from opti_oignon.api.app import app
        cls.client = TestClient(app)

    def test_submit_thumbs_up(self):
        """POST /api/feedback with thumbs up returns 200."""
        r = self.client.post("/api/feedback", json={
            "rating_type": "thumbs",
            "rating_value": 1,
            "model_used": "test-model",
        })
        self.assertEqual(r.status_code, 200)
        self.assertIn("feedback_id", r.json())

    def test_submit_thumbs_down(self):
        """POST /api/feedback with thumbs down returns 200."""
        r = self.client.post("/api/feedback", json={
            "rating_type": "thumbs",
            "rating_value": 0,
            "feedback_text": "Could be better",
        })
        self.assertEqual(r.status_code, 200)

    def test_submit_stars(self):
        """POST /api/feedback with star rating returns 200."""
        r = self.client.post("/api/feedback", json={
            "rating_type": "stars",
            "rating_value": 4,
        })
        self.assertEqual(r.status_code, 200)

    def test_submit_invalid_rating(self):
        """POST /api/feedback with invalid rating returns 400."""
        r = self.client.post("/api/feedback", json={
            "rating_type": "thumbs",
            "rating_value": 99,
        })
        self.assertEqual(r.status_code, 400)

    def test_get_feedback_stats(self):
        """GET /api/feedback/stats returns valid stats."""
        r = self.client.get("/api/feedback/stats")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("total_count", data)
        self.assertIn("by_model", data)

    def test_get_feedback_by_model(self):
        """GET /api/feedback/by-model/{model} returns entries."""
        # Submit one first
        self.client.post("/api/feedback", json={
            "rating_type": "thumbs",
            "rating_value": 1,
            "model_used": "api-test-model",
        })
        r = self.client.get("/api/feedback/by-model/api-test-model")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.json()), 0)

    def test_list_feedback(self):
        """GET /api/feedback/list returns paginated entries."""
        r = self.client.get("/api/feedback/list?limit=10")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_get_and_delete_feedback(self):
        """GET + DELETE /api/feedback/{id} lifecycle."""
        # Create
        r = self.client.post("/api/feedback", json={
            "rating_type": "thumbs", "rating_value": 1
        })
        fid = r.json()["feedback_id"]
        # Get
        r = self.client.get(f"/api/feedback/{fid}")
        self.assertEqual(r.status_code, 200)
        # Delete
        r = self.client.delete(f"/api/feedback/{fid}")
        self.assertEqual(r.status_code, 200)
        # 404 after delete
        r = self.client.get(f"/api/feedback/{fid}")
        self.assertEqual(r.status_code, 404)

    def test_analytics_overview(self):
        """GET /api/analytics/overview returns valid overview."""
        r = self.client.get("/api/analytics/overview")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("total_requests", data)
        self.assertIn("success_rate", data)

    def test_analytics_trends(self):
        """GET /api/analytics/trends returns trend data."""
        r = self.client.get("/api/analytics/trends?window=1h&buckets=6")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["buckets"], 6)
        self.assertEqual(len(data["data"]), 6)

    def test_analytics_trends_invalid_window(self):
        """GET /api/analytics/trends with bad window returns 400."""
        r = self.client.get("/api/analytics/trends?window=bad")
        self.assertEqual(r.status_code, 400)

    def test_analytics_routing_accuracy(self):
        """GET /api/analytics/routing-accuracy returns data."""
        r = self.client.get("/api/analytics/routing-accuracy")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("routed", data)
        self.assertIn("unrouted", data)

    def test_health_includes_feedback_analytics(self):
        """GET /api/health includes feedback and analytics modules."""
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        modules = r.json()["modules"]
        self.assertIn("feedback", modules)
        self.assertIn("analytics", modules)


if __name__ == "__main__":
    unittest.main()
