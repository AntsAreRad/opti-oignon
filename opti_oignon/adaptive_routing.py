#!/usr/bin/env python3
"""
ADAPTIVE ROUTING -- Feedback-Driven Score Adjustments
=============================================================

Connects feedback data to SmartRouter scoring. When
auto_adjust_routing is enabled in feedback.yaml, accumulated
user feedback adjusts model task_scores via a weighted moving
average. Recent feedback carries more weight, and adjustments
are capped to prevent runaway drift.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_FEEDBACK_CONFIG = _CONFIG_DIR / "feedback.yaml"

# Maximum score adjustment per model/task pair (prevents drift)
MAX_ADJUSTMENT = 0.15

# Default minimum samples before adjustments take effect
DEFAULT_MIN_SAMPLES = 10

# Default adjustment factor (multiplied by deviation from neutral)
DEFAULT_ADJUSTMENT_FACTOR = 0.05

# Decay half-life in seconds (7 days) for weighted moving average
DEFAULT_DECAY_HALF_LIFE = 7 * 24 * 3600

# Neutral score: 0.5 for thumbs (midpoint of 0-1), 0.6 for stars (midpoint of 1-5 normalized)
NEUTRAL_SCORE = 0.5


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScoreAdjustment:
    """Score adjustment for a specific model/task pair."""
    model: str
    task_type: str
    adjustment: float = 0.0
    sample_count: int = 0
    weighted_avg_score: float = 0.0
    last_updated: float = 0.0
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "model": self.model,
            "task_type": self.task_type,
            "adjustment": round(self.adjustment, 6),
            "sample_count": self.sample_count,
            "weighted_avg_score": round(self.weighted_avg_score, 4),
            "last_updated": self.last_updated,
            "active": self.active,
        }


@dataclass
class AdaptiveRoutingState:
    """Full state of adaptive routing adjustments."""
    enabled: bool = False
    total_adjustments: int = 0
    active_adjustments: int = 0
    adjustments: list[ScoreAdjustment] = field(default_factory=list)
    min_samples: int = DEFAULT_MIN_SAMPLES
    max_adjustment: float = MAX_ADJUSTMENT
    adjustment_factor: float = DEFAULT_ADJUSTMENT_FACTOR

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "enabled": self.enabled,
            "total_adjustments": self.total_adjustments,
            "active_adjustments": self.active_adjustments,
            "adjustments": [a.to_dict() for a in self.adjustments],
            "min_samples": self.min_samples,
            "max_adjustment": self.max_adjustment,
            "adjustment_factor": self.adjustment_factor,
        }


# =============================================================================
# FEEDBACK ROUTING ADAPTER
# =============================================================================

class FeedbackRoutingAdapter:
    """Adapts SmartRouter scores based on accumulated feedback.

    Reads feedback entries from the FeedbackStore, computes
    weighted moving averages per model/task pair, and produces
    score adjustments that the SmartRouter applies during
    model selection.

    The adapter is lazy: it computes adjustments on demand
    and caches them with a configurable TTL.

    Usage:
        adapter = FeedbackRoutingAdapter()
        adj = adapter.get_adjustment("qwen3:32b", "code_python")
        # adj is a float in [-0.15, +0.15]
    """

    def __init__(
        self,
        feedback_store=None,
        config_path: Path | None = None,
        min_samples: int | None = None,
        adjustment_factor: float | None = None,
        max_adjustment: float | None = None,
        decay_half_life: float | None = None,
        cache_ttl: float = 60.0,
    ):
        """Initialize the feedback routing adapter.

        Args:
            feedback_store: FeedbackStore instance (None = import singleton)
            config_path: Path to feedback.yaml config
            min_samples: Minimum feedback entries before adjustments activate
            adjustment_factor: Multiplier for score deviation
            max_adjustment: Maximum absolute adjustment value
            decay_half_life: Half-life in seconds for temporal weighting
            cache_ttl: Seconds to cache computed adjustments
        """
        self._feedback_store = feedback_store
        self._config_path = config_path or _DEFAULT_FEEDBACK_CONFIG

        # Configuration (loaded from YAML, overridable via constructor)
        self._min_samples = min_samples
        self._adjustment_factor = adjustment_factor
        self._max_adjustment = max_adjustment
        self._decay_half_life = decay_half_life

        # Cache
        self._cache_ttl = cache_ttl
        self._adjustments_cache: dict[str, ScoreAdjustment] = {}
        self._cache_timestamp: float = 0.0

        # Load config from YAML
        self._load_config()

    def _load_config(self):
        """Load configuration from feedback.yaml."""
        if not self._config_path.exists():
            logger.debug("Feedback config not found: %s", self._config_path)
            self._apply_defaults()
            return

        try:
            with open(self._config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Error reading feedback config: %s", e)
            self._apply_defaults()
            return

        fb_config = data.get("feedback", {})
        if not isinstance(fb_config, dict):
            self._apply_defaults()
            return

        # Apply YAML values where constructor didn't override
        if self._min_samples is None:
            self._min_samples = int(fb_config.get(
                "min_samples_for_adjustment", DEFAULT_MIN_SAMPLES
            ))
        if self._adjustment_factor is None:
            self._adjustment_factor = float(fb_config.get(
                "adjustment_factor", DEFAULT_ADJUSTMENT_FACTOR
            ))
        if self._max_adjustment is None:
            self._max_adjustment = MAX_ADJUSTMENT
        if self._decay_half_life is None:
            self._decay_half_life = DEFAULT_DECAY_HALF_LIFE

        logger.info(
            "AdaptiveRouting config loaded: min_samples=%d, "
            "factor=%.3f, max_adj=%.3f",
            self._min_samples, self._adjustment_factor, self._max_adjustment,
        )

    def _apply_defaults(self):
        """Apply default values for all config parameters."""
        if self._min_samples is None:
            self._min_samples = DEFAULT_MIN_SAMPLES
        if self._adjustment_factor is None:
            self._adjustment_factor = DEFAULT_ADJUSTMENT_FACTOR
        if self._max_adjustment is None:
            self._max_adjustment = MAX_ADJUSTMENT
        if self._decay_half_life is None:
            self._decay_half_life = DEFAULT_DECAY_HALF_LIFE

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether adaptive routing is enabled (checks feedback store config)."""
        store = self._get_feedback_store()
        if store is None:
            return False
        return getattr(store, "auto_adjust_routing", False)

    @property
    def min_samples(self) -> int:
        return self._min_samples

    @property
    def max_adjustment(self) -> float:
        return self._max_adjustment

    @property
    def adjustment_factor(self) -> float:
        return self._adjustment_factor

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------

    def get_adjustment(self, model: str, task_type: str) -> float:
        """Get the score adjustment for a model/task pair.

        Returns a value in [-max_adjustment, +max_adjustment].
        Returns 0.0 if adaptive routing is disabled, insufficient
        samples exist, or no feedback is found.

        Args:
            model: Model name (e.g. "qwen3:32b")
            task_type: Task type (e.g. "code_python")

        Returns:
            Score adjustment float
        """
        if not self.enabled:
            return 0.0

        # Refresh cache if stale
        self._refresh_cache_if_needed()

        key = f"{model}:{task_type}"
        adj = self._adjustments_cache.get(key)
        if adj is None or not adj.active:
            return 0.0

        return adj.adjustment

    def get_all_adjustments(self) -> AdaptiveRoutingState:
        """Get the full adaptive routing state.

        Returns:
            AdaptiveRoutingState with all current adjustments
        """
        self._refresh_cache_if_needed()

        adjustments = list(self._adjustments_cache.values())
        active = [a for a in adjustments if a.active]

        return AdaptiveRoutingState(
            enabled=self.enabled,
            total_adjustments=len(adjustments),
            active_adjustments=len(active),
            adjustments=adjustments,
            min_samples=self._min_samples,
            max_adjustment=self._max_adjustment,
            adjustment_factor=self._adjustment_factor,
        )

    def get_adjustments_for_model(self, model: str) -> dict[str, float]:
        """Get all task adjustments for a specific model.

        Args:
            model: Model name

        Returns:
            Dict mapping task_type to adjustment value
        """
        self._refresh_cache_if_needed()

        result = {}
        prefix = f"{model}:"
        for key, adj in self._adjustments_cache.items():
            if key.startswith(prefix) and adj.active:
                result[adj.task_type] = adj.adjustment
        return result

    def invalidate_cache(self):
        """Force cache invalidation on next access."""
        self._cache_timestamp = 0.0

    def has_active_adjustments(self) -> bool:
        """Check if any adjustments are currently active."""
        if not self.enabled:
            return False
        self._refresh_cache_if_needed()
        return any(a.active for a in self._adjustments_cache.values())

    # -------------------------------------------------------------------------
    # Internal computation
    # -------------------------------------------------------------------------

    def _get_feedback_store(self):
        """Get the feedback store (lazy import if needed)."""
        if self._feedback_store is not None:
            return self._feedback_store

        try:
            from .feedback import feedback_store
            self._feedback_store = feedback_store
            return feedback_store
        except ImportError:
            return None

    def _refresh_cache_if_needed(self):
        """Recompute adjustments if cache has expired."""
        now = time.time()
        if now - self._cache_timestamp < self._cache_ttl:
            return

        self._compute_adjustments()
        self._cache_timestamp = now

    def _compute_adjustments(self):
        """Compute all score adjustments from feedback data.

        Queries the feedback store for model/task aggregations,
        computes weighted moving averages, and produces capped
        score adjustments.
        """
        store = self._get_feedback_store()
        if store is None:
            self._adjustments_cache = {}
            return

        try:
            entries = self._get_feedback_entries(store)
        except Exception as e:
            logger.warning("Failed to query feedback for adaptive routing: %s", e)
            self._adjustments_cache = {}
            return

        # Group by model:task_type
        groups: dict[str, list[tuple[float, float]]] = {}
        for entry in entries:
            model = getattr(entry, "model_used", "") or ""
            task = getattr(entry, "task_type", "") or ""
            if not model or not task:
                continue

            score = self._normalize_entry_score(entry)
            if score is None:
                continue
            timestamp = getattr(entry, "timestamp", 0.0) or 0.0

            key = f"{model}:{task}"
            if key not in groups:
                groups[key] = []
            groups[key].append((score, timestamp))

        # Compute adjustments per group
        now = time.time()
        new_cache: dict[str, ScoreAdjustment] = {}

        for key, scores_and_times in groups.items():
            model, task = key.split(":", 1)
            count = len(scores_and_times)

            # Weighted moving average with temporal decay
            weighted_sum = 0.0
            weight_total = 0.0
            for score, ts in scores_and_times:
                w = self._temporal_weight(ts, now)
                weighted_sum += score * w
                weight_total += w

            if weight_total > 0:
                weighted_avg = weighted_sum / weight_total
            else:
                weighted_avg = NEUTRAL_SCORE

            # Compute deviation from neutral
            deviation = weighted_avg - NEUTRAL_SCORE

            # Apply adjustment factor
            raw_adjustment = deviation * self._adjustment_factor * 2.0

            # Clamp to max adjustment
            clamped = max(-self._max_adjustment, min(self._max_adjustment, raw_adjustment))

            # Active only if enough samples
            active = count >= self._min_samples

            new_cache[key] = ScoreAdjustment(
                model=model,
                task_type=task,
                adjustment=clamped if active else 0.0,
                sample_count=count,
                weighted_avg_score=weighted_avg,
                last_updated=now,
                active=active,
            )

        self._adjustments_cache = new_cache

    def _get_feedback_entries(self, store) -> list:
        """Retrieve feedback entries from the store.

        Returns all feedback entries. The store's list_feedback
        method handles pagination.
        """
        # Use list_feedback with a large limit to get all entries
        if hasattr(store, "list_feedback"):
            return store.list_feedback(limit=10000)
        return []

    def _normalize_entry_score(self, entry) -> float | None:
        """Normalize a feedback entry to a 0-1 score.

        Thumbs: 0 (down) or 1 (up)
        Stars: (value - 1) / 4 to map 1-5 to 0-1

        Returns None when the rating value cannot be read as a number, so a
        malformed or crafted entry is skipped rather than breaking the whole
        adjustment computation.
        """
        rating_type = getattr(entry, "rating_type", "thumbs")
        rating_value = getattr(entry, "rating_value", 1)

        try:
            numeric = float(rating_value)
        except (TypeError, ValueError):
            return None

        if rating_type == "stars":
            return (numeric - 1.0) / 4.0
        # Thumbs: already 0 or 1
        return numeric

    def _temporal_weight(self, timestamp: float, now: float) -> float:
        """Compute temporal weight using exponential decay.

        Recent feedback is weighted more heavily than older feedback.
        Uses a half-life decay model.

        Args:
            timestamp: Feedback entry timestamp
            now: Current time

        Returns:
            Weight in (0, 1]
        """
        if timestamp <= 0 or self._decay_half_life <= 0:
            return 1.0

        age = max(0.0, now - timestamp)
        # Exponential decay: w = 2^(-age / half_life)
        return math.pow(2.0, -age / self._decay_half_life)


# =============================================================================
# SINGLETON
# =============================================================================

feedback_routing_adapter = FeedbackRoutingAdapter()
