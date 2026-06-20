#!/usr/bin/env python3
"""
Tests for Adaptive Routing -- Feedback-Driven Score Adjustments (S62)
======================================================================

Tests cover:
- FeedbackRoutingAdapter initialization and config loading
- Score adjustment computation with weighted moving average
- Temporal decay weighting
- Min sample threshold
- Max adjustment capping (±0.15)
- Cache behavior and TTL
- Integration with SmartRouter._compute_task_score
- API endpoint for feedback-adjustments
- Edge cases (empty feedback, missing data, graceful degradation)
"""

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import yaml

# =============================================================================
# MOCK FEEDBACK ENTRY
# =============================================================================

@dataclass
class MockFeedbackEntry:
    """Minimal mock matching FeedbackEntry interface."""
    model_used: str = ""
    task_type: str = ""
    rating_type: str = "thumbs"
    rating_value: int = 1
    timestamp: float = 0.0
    feedback_id: str = ""
    pipeline_used: str = ""


# =============================================================================
# MOCK FEEDBACK STORE
# =============================================================================

class MockFeedbackStore:
    """Mock FeedbackStore with configurable feedback entries."""

    def __init__(self, entries=None, auto_adjust=True):
        self._entries = list(entries or [])
        self._auto_adjust = auto_adjust

    @property
    def auto_adjust_routing(self) -> bool:
        return self._auto_adjust

    @property
    def enabled(self) -> bool:
        return True

    def list_feedback(self, limit=10000) -> list:
        return self._entries[:limit]

    def add(self, entry):
        self._entries.append(entry)


# =============================================================================
# HELPER: Create adapter with mock store
# =============================================================================

def make_adapter(
    entries=None,
    auto_adjust=True,
    min_samples=3,
    adjustment_factor=0.05,
    max_adjustment=0.15,
    cache_ttl=0.0,
    decay_half_life=7 * 24 * 3600,
):
    """Create a FeedbackRoutingAdapter with a mock feedback store."""
    from opti_oignon.adaptive_routing import FeedbackRoutingAdapter

    store = MockFeedbackStore(entries=entries, auto_adjust=auto_adjust)
    # Use a temp config path that doesn't exist to skip YAML loading
    adapter = FeedbackRoutingAdapter(
        feedback_store=store,
        config_path=Path("/tmp/nonexistent_feedback.yaml"),
        min_samples=min_samples,
        adjustment_factor=adjustment_factor,
        max_adjustment=max_adjustment,
        decay_half_life=decay_half_life,
        cache_ttl=cache_ttl,
    )
    return adapter, store


# =============================================================================
# TESTS: Initialization & Configuration
# =============================================================================

class TestAdaptiveRoutingInit:
    """Tests for adapter initialization and configuration."""

    def test_init_defaults(self):
        """Adapter initializes with default config values."""
        from opti_oignon.adaptive_routing import (
            DEFAULT_ADJUSTMENT_FACTOR,
            DEFAULT_MIN_SAMPLES,
            MAX_ADJUSTMENT,
            FeedbackRoutingAdapter,
        )
        adapter = FeedbackRoutingAdapter(
            feedback_store=MockFeedbackStore(),
            config_path=Path("/tmp/nonexistent.yaml"),
        )
        assert adapter.min_samples == DEFAULT_MIN_SAMPLES
        assert adapter.adjustment_factor == DEFAULT_ADJUSTMENT_FACTOR
        assert adapter.max_adjustment == MAX_ADJUSTMENT

    def test_init_custom_params(self):
        """Constructor parameters override defaults."""
        adapter, _ = make_adapter(min_samples=5, adjustment_factor=0.1, max_adjustment=0.2)
        assert adapter.min_samples == 5
        assert adapter.adjustment_factor == 0.1
        assert adapter.max_adjustment == 0.2

    def test_load_config_from_yaml(self):
        """Adapter loads config from YAML file."""
        from opti_oignon.adaptive_routing import FeedbackRoutingAdapter

        config_data = {
            "feedback": {
                "auto_adjust_routing": True,
                "min_samples_for_adjustment": 20,
                "adjustment_factor": 0.08,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            tmp_path = Path(f.name)

        try:
            adapter = FeedbackRoutingAdapter(
                feedback_store=MockFeedbackStore(),
                config_path=tmp_path,
            )
            assert adapter.min_samples == 20
            assert adapter.adjustment_factor == 0.08
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_constructor_overrides_yaml(self):
        """Explicit constructor params take priority over YAML."""
        from opti_oignon.adaptive_routing import FeedbackRoutingAdapter

        config_data = {
            "feedback": {
                "min_samples_for_adjustment": 20,
                "adjustment_factor": 0.08,
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            tmp_path = Path(f.name)

        try:
            adapter = FeedbackRoutingAdapter(
                feedback_store=MockFeedbackStore(),
                config_path=tmp_path,
                min_samples=7,
            )
            # Constructor value wins
            assert adapter.min_samples == 7
            # YAML value applied where constructor didn't set
            assert adapter.adjustment_factor == 0.08
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_missing_config_file(self):
        """Adapter handles missing config file gracefully."""
        from opti_oignon.adaptive_routing import FeedbackRoutingAdapter

        adapter = FeedbackRoutingAdapter(
            feedback_store=MockFeedbackStore(),
            config_path=Path("/tmp/does_not_exist_xyz.yaml"),
        )
        # Should use defaults
        assert adapter.min_samples > 0
        assert adapter.max_adjustment > 0


# =============================================================================
# TESTS: Enabled / Disabled
# =============================================================================

class TestAdaptiveRoutingEnabled:
    """Tests for enabled/disabled state."""

    def test_enabled_when_store_auto_adjust_true(self):
        """Adapter is enabled when store.auto_adjust_routing is True."""
        adapter, _ = make_adapter(auto_adjust=True)
        assert adapter.enabled is True

    def test_disabled_when_store_auto_adjust_false(self):
        """Adapter is disabled when store.auto_adjust_routing is False."""
        adapter, _ = make_adapter(auto_adjust=False)
        assert adapter.enabled is False

    def test_disabled_when_no_store(self):
        """Adapter is disabled when no feedback store is available."""
        from opti_oignon.adaptive_routing import FeedbackRoutingAdapter

        adapter = FeedbackRoutingAdapter(
            feedback_store=None,
            config_path=Path("/tmp/nonexistent.yaml"),
        )
        # _get_feedback_store returns None since we passed None and import may fail
        # In isolation, this should be False
        assert adapter.enabled is False

    def test_get_adjustment_returns_zero_when_disabled(self):
        """Adjustments return 0.0 when adapter is disabled."""
        adapter, _ = make_adapter(auto_adjust=False)
        assert adapter.get_adjustment("model", "task") == 0.0


# =============================================================================
# TESTS: Score Normalization
# =============================================================================

class TestScoreNormalization:
    """Tests for feedback entry score normalization."""

    def test_thumbs_up_normalized(self):
        """Thumbs up (1) normalizes to 1.0."""
        adapter, _ = make_adapter()
        entry = MockFeedbackEntry(rating_type="thumbs", rating_value=1)
        assert adapter._normalize_entry_score(entry) == 1.0

    def test_thumbs_down_normalized(self):
        """Thumbs down (0) normalizes to 0.0."""
        adapter, _ = make_adapter()
        entry = MockFeedbackEntry(rating_type="thumbs", rating_value=0)
        assert adapter._normalize_entry_score(entry) == 0.0

    def test_stars_1_normalized(self):
        """Stars 1 normalizes to 0.0."""
        adapter, _ = make_adapter()
        entry = MockFeedbackEntry(rating_type="stars", rating_value=1)
        assert adapter._normalize_entry_score(entry) == 0.0

    def test_stars_3_normalized(self):
        """Stars 3 normalizes to 0.5."""
        adapter, _ = make_adapter()
        entry = MockFeedbackEntry(rating_type="stars", rating_value=3)
        assert adapter._normalize_entry_score(entry) == 0.5

    def test_stars_5_normalized(self):
        """Stars 5 normalizes to 1.0."""
        adapter, _ = make_adapter()
        entry = MockFeedbackEntry(rating_type="stars", rating_value=5)
        assert adapter._normalize_entry_score(entry) == 1.0


# =============================================================================
# TESTS: Temporal Weighting
# =============================================================================

class TestTemporalWeight:
    """Tests for exponential decay temporal weighting."""

    def test_recent_entry_full_weight(self):
        """Entry at current time gets weight ~1.0."""
        adapter, _ = make_adapter()
        now = time.time()
        w = adapter._temporal_weight(now, now)
        assert abs(w - 1.0) < 0.001

    def test_half_life_decay(self):
        """Entry at half-life age gets weight ~0.5."""
        adapter, _ = make_adapter(decay_half_life=3600)
        now = time.time()
        w = adapter._temporal_weight(now - 3600, now)
        assert abs(w - 0.5) < 0.01

    def test_old_entry_low_weight(self):
        """Entry at 3x half-life gets weight ~0.125."""
        adapter, _ = make_adapter(decay_half_life=3600)
        now = time.time()
        w = adapter._temporal_weight(now - 3 * 3600, now)
        assert abs(w - 0.125) < 0.01

    def test_zero_timestamp_full_weight(self):
        """Entry with timestamp=0 gets weight 1.0."""
        adapter, _ = make_adapter()
        w = adapter._temporal_weight(0, time.time())
        assert w == 1.0

    def test_negative_age_clamps_to_zero(self):
        """Future timestamps are clamped to weight 1.0."""
        adapter, _ = make_adapter()
        now = time.time()
        w = adapter._temporal_weight(now + 1000, now)
        assert abs(w - 1.0) < 0.001


# =============================================================================
# TESTS: Score Adjustment Computation
# =============================================================================

class TestAdjustmentComputation:
    """Tests for the core adjustment computation logic."""

    def test_positive_feedback_positive_adjustment(self):
        """All thumbs-up feedback produces a positive adjustment."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now - i * 60,
            )
            for i in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        adj = adapter.get_adjustment("model-a", "code")
        assert adj > 0.0

    def test_negative_feedback_negative_adjustment(self):
        """All thumbs-down feedback produces a negative adjustment."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=0,
                timestamp=now - i * 60,
            )
            for i in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        adj = adapter.get_adjustment("model-a", "code")
        assert adj < 0.0

    def test_mixed_feedback_moderate_adjustment(self):
        """Mixed feedback produces near-zero adjustment."""
        now = time.time()
        entries = []
        for i in range(5):
            entries.append(MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now - i * 60,
            ))
            entries.append(MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=0,
                timestamp=now - i * 60,
            ))
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        adj = adapter.get_adjustment("model-a", "code")
        assert abs(adj) < 0.01

    def test_adjustment_capped_positive(self):
        """Positive adjustment is capped at max_adjustment."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(50)
        ]
        adapter, _ = make_adapter(
            entries=entries, min_samples=3,
            adjustment_factor=10.0,  # Very high factor to trigger capping
            max_adjustment=0.15,
        )
        adj = adapter.get_adjustment("model-a", "code")
        assert adj <= 0.15
        assert adj > 0.0

    def test_adjustment_capped_negative(self):
        """Negative adjustment is capped at -max_adjustment."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=0,
                timestamp=now,
            )
            for _ in range(50)
        ]
        adapter, _ = make_adapter(
            entries=entries, min_samples=3,
            adjustment_factor=10.0,
            max_adjustment=0.15,
        )
        adj = adapter.get_adjustment("model-a", "code")
        assert adj >= -0.15
        assert adj < 0.0

    def test_below_min_samples_no_adjustment(self):
        """Below min_samples threshold, adjustment is 0.0."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(2)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=5)
        adj = adapter.get_adjustment("model-a", "code")
        assert adj == 0.0

    def test_exactly_min_samples_activates(self):
        """At exactly min_samples, adjustment becomes active."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=5)
        adj = adapter.get_adjustment("model-a", "code")
        assert adj > 0.0

    def test_different_model_task_pairs_independent(self):
        """Adjustments for different model/task pairs are independent."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ] + [
            MockFeedbackEntry(
                model_used="model-b", task_type="code",
                rating_type="thumbs", rating_value=0,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        adj_a = adapter.get_adjustment("model-a", "code")
        adj_b = adapter.get_adjustment("model-b", "code")
        assert adj_a > 0.0
        assert adj_b < 0.0

    def test_nonexistent_pair_returns_zero(self):
        """Model/task pair not in feedback returns 0.0."""
        adapter, _ = make_adapter(entries=[], min_samples=3)
        assert adapter.get_adjustment("unknown", "unknown") == 0.0

    def test_stars_feedback_adjustment(self):
        """Star ratings produce correct adjustments."""
        now = time.time()
        # All 5-star ratings -> positive
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="general",
                rating_type="stars", rating_value=5,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        adj = adapter.get_adjustment("model-a", "general")
        assert adj > 0.0

    def test_stars_low_rating_negative_adjustment(self):
        """Low star ratings produce negative adjustments."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="general",
                rating_type="stars", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        adj = adapter.get_adjustment("model-a", "general")
        assert adj < 0.0


# =============================================================================
# TESTS: Cache Behavior
# =============================================================================

class TestCacheBehavior:
    """Tests for caching and TTL."""

    def test_cache_reuses_results(self):
        """Multiple calls within TTL use cached results."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, store = make_adapter(entries=entries, min_samples=3, cache_ttl=60.0)

        # First call computes
        adj1 = adapter.get_adjustment("model-a", "code")

        # Modify store (shouldn't be visible due to cache)
        store._entries.clear()

        adj2 = adapter.get_adjustment("model-a", "code")
        assert adj1 == adj2

    def test_cache_expires(self):
        """Results are recomputed after TTL expires."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, store = make_adapter(entries=entries, min_samples=3, cache_ttl=0.0)

        adj1 = adapter.get_adjustment("model-a", "code")
        assert adj1 > 0.0

        # Clear store and invalidate
        store._entries.clear()
        adapter.invalidate_cache()

        adj2 = adapter.get_adjustment("model-a", "code")
        assert adj2 == 0.0

    def test_invalidate_cache(self):
        """invalidate_cache forces recomputation."""
        adapter, _ = make_adapter(cache_ttl=3600.0)
        # Access to populate cache
        adapter.get_adjustment("x", "y")
        # Invalidate
        adapter.invalidate_cache()
        assert adapter._cache_timestamp == 0.0


# =============================================================================
# TESTS: Full State Retrieval
# =============================================================================

class TestFullState:
    """Tests for get_all_adjustments and related methods."""

    def test_get_all_adjustments_structure(self):
        """get_all_adjustments returns proper AdaptiveRoutingState."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        state = adapter.get_all_adjustments()

        assert state.enabled is True
        assert state.total_adjustments >= 1
        assert state.active_adjustments >= 1
        assert len(state.adjustments) >= 1
        assert state.min_samples == 3
        assert state.max_adjustment == 0.15

    def test_get_all_adjustments_serialization(self):
        """AdaptiveRoutingState.to_dict() produces valid JSON-serializable dict."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        state = adapter.get_all_adjustments()
        d = state.to_dict()

        assert "enabled" in d
        assert "adjustments" in d
        assert isinstance(d["adjustments"], list)
        if d["adjustments"]:
            adj = d["adjustments"][0]
            assert "model" in adj
            assert "task_type" in adj
            assert "adjustment" in adj
            assert "active" in adj

    def test_get_adjustments_for_model(self):
        """get_adjustments_for_model returns task->adjustment mapping."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ] + [
            MockFeedbackEntry(
                model_used="model-a", task_type="general",
                rating_type="thumbs", rating_value=0,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        adjs = adapter.get_adjustments_for_model("model-a")

        assert "code" in adjs
        assert "general" in adjs
        assert adjs["code"] > 0.0
        assert adjs["general"] < 0.0

    def test_has_active_adjustments_true(self):
        """has_active_adjustments returns True when adjustments are active."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        assert adapter.has_active_adjustments() is True

    def test_has_active_adjustments_false_no_data(self):
        """has_active_adjustments returns False when no feedback."""
        adapter, _ = make_adapter(entries=[], min_samples=3)
        assert adapter.has_active_adjustments() is False

    def test_has_active_adjustments_false_disabled(self):
        """has_active_adjustments returns False when adapter disabled."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3, auto_adjust=False)
        assert adapter.has_active_adjustments() is False


# =============================================================================
# TESTS: ScoreAdjustment Dataclass
# =============================================================================

class TestScoreAdjustment:
    """Tests for ScoreAdjustment dataclass."""

    def test_to_dict(self):
        """ScoreAdjustment.to_dict() serializes all fields."""
        from opti_oignon.adaptive_routing import ScoreAdjustment

        adj = ScoreAdjustment(
            model="qwen3:32b",
            task_type="code_python",
            adjustment=0.05,
            sample_count=15,
            weighted_avg_score=0.8,
            last_updated=1000.0,
            active=True,
        )
        d = adj.to_dict()
        assert d["model"] == "qwen3:32b"
        assert d["task_type"] == "code_python"
        assert d["adjustment"] == 0.05
        assert d["sample_count"] == 15
        assert d["active"] is True

    def test_default_inactive(self):
        """ScoreAdjustment defaults to inactive."""
        from opti_oignon.adaptive_routing import ScoreAdjustment

        adj = ScoreAdjustment(model="m", task_type="t")
        assert adj.active is False
        assert adj.adjustment == 0.0


# =============================================================================
# TESTS: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and graceful degradation."""

    def test_entries_missing_model(self):
        """Entries without model_used are skipped."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="", task_type="code",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        assert adapter.get_adjustment("", "code") == 0.0

    def test_entries_missing_task(self):
        """Entries without task_type are skipped."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="",
                rating_type="thumbs", rating_value=1,
                timestamp=now,
            )
            for _ in range(5)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        assert adapter.get_adjustment("model-a", "") == 0.0

    def test_store_query_exception(self):
        """Adapter handles store query exceptions gracefully."""
        store = MockFeedbackStore(auto_adjust=True)
        store.list_feedback = MagicMock(side_effect=RuntimeError("DB error"))

        from opti_oignon.adaptive_routing import FeedbackRoutingAdapter

        adapter = FeedbackRoutingAdapter(
            feedback_store=store,
            config_path=Path("/tmp/nonexistent.yaml"),
            cache_ttl=0.0,
        )
        # Should not raise, returns 0.0
        assert adapter.get_adjustment("model", "task") == 0.0

    def test_large_entry_count(self):
        """Adapter handles large numbers of entries."""
        now = time.time()
        entries = [
            MockFeedbackEntry(
                model_used="model-a", task_type="code",
                rating_type="thumbs", rating_value=1 if i % 3 != 0 else 0,
                timestamp=now - i * 10,
            )
            for i in range(1000)
        ]
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        adj = adapter.get_adjustment("model-a", "code")
        # Should produce some adjustment, not crash
        assert isinstance(adj, float)

    def test_multiple_models_multiple_tasks(self):
        """Adapter handles many model/task combinations."""
        now = time.time()
        entries = []
        for m in range(5):
            for t in range(5):
                for _ in range(5):
                    entries.append(MockFeedbackEntry(
                        model_used=f"model-{m}",
                        task_type=f"task-{t}",
                        rating_type="thumbs",
                        rating_value=1 if m > t else 0,
                        timestamp=now,
                    ))
        adapter, _ = make_adapter(entries=entries, min_samples=3)
        state = adapter.get_all_adjustments()
        assert state.total_adjustments == 25  # 5 models x 5 tasks

    def test_zero_decay_half_life(self):
        """Adapter handles zero decay half-life (no decay)."""
        adapter, _ = make_adapter(decay_half_life=0)
        w = adapter._temporal_weight(1000, time.time())
        assert w == 1.0


# =============================================================================
# TESTS: SmartRouter Integration
# =============================================================================

class TestSmartRouterIntegration:
    """Tests for integration with SmartRouter._compute_task_score."""

    def test_smart_routing_result_has_feedback_adjusted(self):
        """SmartRoutingResult includes feedback_adjusted field."""
        from opti_oignon.smart_router import SmartRoutingResult

        result = SmartRoutingResult(model="test", feedback_adjusted=True)
        d = result.to_dict()
        assert "feedback_adjusted" in d
        assert d["feedback_adjusted"] is True

    def test_smart_routing_result_default_not_adjusted(self):
        """SmartRoutingResult defaults to feedback_adjusted=False."""
        from opti_oignon.smart_router import SmartRoutingResult

        result = SmartRoutingResult(model="test")
        assert result.feedback_adjusted is False


# =============================================================================
# TESTS: API Endpoint
# =============================================================================

class TestFeedbackAdjustmentsAPI:
    """Tests for the GET /api/smart-routing/feedback-adjustments endpoint."""

    def test_endpoint_returns_state(self):
        """Endpoint returns adaptive routing state."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from opti_oignon.api.routes_smart_routing import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/api/smart-routing/feedback-adjustments")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "adjustments" in data
        assert "total_adjustments" in data
        assert "active_adjustments" in data

    def test_invalidate_endpoint(self):
        """POST /feedback-adjustments/invalidate returns ok."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from opti_oignon.api.routes_smart_routing import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.post("/api/smart-routing/feedback-adjustments/invalidate")
        # May return 200 or 503 depending on adapter availability
        assert resp.status_code in (200, 503)


# =============================================================================
# TESTS: Singleton
# =============================================================================

class TestSingleton:
    """Tests for module-level singleton."""

    def test_singleton_exists(self):
        """Module-level singleton is created on import."""
        from opti_oignon.adaptive_routing import feedback_routing_adapter
        assert feedback_routing_adapter is not None

    def test_singleton_is_adapter(self):
        """Singleton is a FeedbackRoutingAdapter instance."""
        from opti_oignon.adaptive_routing import (
            FeedbackRoutingAdapter,
            feedback_routing_adapter,
        )
        assert isinstance(feedback_routing_adapter, FeedbackRoutingAdapter)
