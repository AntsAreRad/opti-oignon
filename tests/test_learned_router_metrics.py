#!/usr/bin/env python3
"""
Tests for LearnedRouterMetrics A/B comparison (S67, Step 3).

Covers: empty state, learned vs yaml counts, confidence averages,
agreement rate, top disagreements, confidence histogram, and
window filtering.
"""

import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_router_and_metrics(tmp_path: Path):
    """Create an isolated LearnedRouter + LearnedRouterMetrics pair."""
    from opti_oignon.learned_router import LearnedRouter, LearnedRouterMetrics

    router = LearnedRouter(
        config_path=tmp_path / "lr.yaml",
        db_path=tmp_path / "lr.db",
        model_path=tmp_path / "lr.pkl",
    )
    metrics = LearnedRouterMetrics(router)
    return router, metrics


def _log_decision(router, ml_task, ml_conf, yaml_task, source):
    router.log_routing_decision(
        query_text="test query",
        ml_task_type=ml_task,
        ml_confidence=ml_conf,
        yaml_task_type=yaml_task,
        routing_source=source,
    )


# ---------------------------------------------------------------------------
# Tests: empty state
# ---------------------------------------------------------------------------

class TestEmptyState:
    def test_empty_returns_zero_total(self, tmp_path):
        _, metrics = _make_router_and_metrics(tmp_path)
        result = metrics.compute()
        assert result.total_decisions == 0

    def test_empty_returns_zero_counts(self, tmp_path):
        _, metrics = _make_router_and_metrics(tmp_path)
        result = metrics.compute()
        assert result.learned_count == 0
        assert result.yaml_count == 0

    def test_empty_returns_zero_ratios(self, tmp_path):
        _, metrics = _make_router_and_metrics(tmp_path)
        result = metrics.compute()
        assert result.learned_ratio == 0.0
        assert result.avg_ml_confidence == 0.0

    def test_empty_to_dict(self, tmp_path):
        _, metrics = _make_router_and_metrics(tmp_path)
        d = metrics.compute().to_dict()
        assert d["total_decisions"] == 0
        assert "learned_ratio" in d
        assert "top_disagreements" in d


# ---------------------------------------------------------------------------
# Tests: source counts
# ---------------------------------------------------------------------------

class TestSourceCounts:
    def test_learned_count(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        _log_decision(router, "code_python", 0.9, "general", "learned")
        _log_decision(router, "code_python", 0.85, "general", "learned")
        _log_decision(router, "debug", 0.3, "debug", "yaml")
        result = metrics.compute()
        assert result.learned_count == 2
        assert result.yaml_count == 1
        assert result.total_decisions == 3

    def test_learned_ratio_calculation(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        for _ in range(3):
            _log_decision(router, "code_python", 0.8, "general", "learned")
        for _ in range(1):
            _log_decision(router, "debug", 0.2, "debug", "yaml")
        result = metrics.compute()
        assert abs(result.learned_ratio - 0.75) < 0.01

    def test_decisions_by_source_dict(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        _log_decision(router, "planning", 0.7, "general", "learned")
        _log_decision(router, "general", 0.4, "general", "yaml")
        result = metrics.compute()
        assert result.decisions_by_source["learned"] == 1
        assert result.decisions_by_source["yaml"] == 1


# ---------------------------------------------------------------------------
# Tests: confidence averages
# ---------------------------------------------------------------------------

class TestConfidenceAverages:
    def test_avg_ml_confidence(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        _log_decision(router, "code_python", 0.8, "general", "learned")
        _log_decision(router, "debug", 0.6, "debug", "yaml")
        result = metrics.compute()
        assert abs(result.avg_ml_confidence - 0.7) < 0.01

    def test_avg_ml_confidence_learned_only(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        _log_decision(router, "code_python", 0.9, "general", "learned")
        _log_decision(router, "code_python", 0.7, "general", "learned")
        _log_decision(router, "debug", 0.3, "debug", "yaml")
        result = metrics.compute()
        # Only the 'learned' decisions: (0.9 + 0.7) / 2 = 0.8
        assert abs(result.avg_ml_confidence_learned - 0.8) < 0.01

    def test_avg_ml_confidence_yaml_only(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        _log_decision(router, "code_python", 0.9, "general", "learned")
        _log_decision(router, "debug", 0.2, "debug", "yaml")
        _log_decision(router, "debug", 0.4, "debug", "yaml")
        result = metrics.compute()
        # Only the 'yaml' decisions: (0.2 + 0.4) / 2 = 0.3
        assert abs(result.avg_ml_confidence_yaml - 0.3) < 0.01


# ---------------------------------------------------------------------------
# Tests: class agreement rate
# ---------------------------------------------------------------------------

class TestClassAgreementRate:
    def test_full_agreement(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        for _ in range(4):
            _log_decision(router, "debug", 0.8, "debug", "learned")
        result = metrics.compute()
        assert result.class_agreement_rate == 1.0

    def test_no_agreement(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        for _ in range(3):
            _log_decision(router, "code_python", 0.8, "debug", "learned")
        result = metrics.compute()
        assert result.class_agreement_rate == 0.0

    def test_partial_agreement(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        _log_decision(router, "debug", 0.8, "debug", "learned")    # agree
        _log_decision(router, "debug", 0.8, "debug", "learned")    # agree
        _log_decision(router, "code_python", 0.8, "debug", "learned")  # disagree
        _log_decision(router, "code_python", 0.8, "debug", "learned")  # disagree
        result = metrics.compute()
        assert abs(result.class_agreement_rate - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Tests: top disagreements
# ---------------------------------------------------------------------------

class TestTopDisagreements:
    def test_top_disagreements_empty_when_full_agreement(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        _log_decision(router, "debug", 0.9, "debug", "learned")
        result = metrics.compute()
        assert result.top_disagreements == []

    def test_top_disagreements_ranked_by_frequency(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        # Most common: ml=code_python vs yaml=debug (3 times)
        for _ in range(3):
            _log_decision(router, "code_python", 0.8, "debug", "learned")
        # Less common: ml=reasoning vs yaml=general (1 time)
        _log_decision(router, "reasoning", 0.8, "general", "yaml")
        result = metrics.compute()
        assert len(result.top_disagreements) >= 1
        assert result.top_disagreements[0]["ml_task_type"] == "code_python"
        assert result.top_disagreements[0]["yaml_task_type"] == "debug"
        assert result.top_disagreements[0]["count"] == 3

    def test_top_disagreements_capped_at_five(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        pairs = [
            ("a", "b"), ("c", "d"), ("e", "f"),
            ("g", "h"), ("i", "j"), ("k", "l"),
        ]
        for ml, yaml_ in pairs:
            for _ in range(2):
                _log_decision(router, ml, 0.5, yaml_, "learned")
        result = metrics.compute()
        assert len(result.top_disagreements) <= 5


# ---------------------------------------------------------------------------
# Tests: time window filtering
# ---------------------------------------------------------------------------

class TestWindowFiltering:
    def test_old_decisions_excluded(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        # Manually insert an old decision
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "lr.db"))
        conn.execute(
            "INSERT INTO routing_decisions "
            "(query_text, ml_task_type, ml_confidence, yaml_task_type, routing_source, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("old query", "debug", 0.8, "general", "learned", time.time() - 48 * 3600),
        )
        conn.commit()
        conn.close()
        # Recent decision
        _log_decision(router, "code_python", 0.9, "general", "learned")
        # Only 24h window: should see 1 decision, not 2
        result = metrics.compute(window_hours=24)
        assert result.total_decisions == 1

    def test_window_hours_in_result(self, tmp_path):
        _, metrics = _make_router_and_metrics(tmp_path)
        result = metrics.compute(window_hours=48)
        assert result.window_hours == 48.0


# ---------------------------------------------------------------------------
# Tests: confidence histogram
# ---------------------------------------------------------------------------

class TestConfidenceHistogram:
    def test_histogram_returns_list(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        hist = metrics.get_confidence_histogram()
        assert isinstance(hist, list)

    def test_histogram_empty_on_no_data(self, tmp_path):
        _, metrics = _make_router_and_metrics(tmp_path)
        hist = metrics.get_confidence_histogram()
        assert hist == []

    def test_histogram_bin_count(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        for conf in [0.1, 0.5, 0.9]:
            _log_decision(router, "debug", conf, "debug", "learned")
        hist = metrics.get_confidence_histogram(bins=5)
        assert len(hist) == 5

    def test_histogram_bucket_structure(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        _log_decision(router, "debug", 0.8, "debug", "learned")
        hist = metrics.get_confidence_histogram(bins=10)
        for bucket in hist:
            assert "bucket_min" in bucket
            assert "bucket_max" in bucket
            assert "count" in bucket

    def test_histogram_counts_sum_to_total(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        for conf in [0.1, 0.3, 0.6, 0.8, 0.95]:
            _log_decision(router, "debug", conf, "debug", "learned")
        hist = metrics.get_confidence_histogram(bins=10)
        total = sum(b["count"] for b in hist)
        assert total == 5


# ---------------------------------------------------------------------------
# Tests: to_dict completeness
# ---------------------------------------------------------------------------

class TestABMetricsResultToDict:
    def test_to_dict_all_keys(self, tmp_path):
        router, metrics = _make_router_and_metrics(tmp_path)
        _log_decision(router, "code_python", 0.8, "general", "learned")
        _log_decision(router, "debug", 0.3, "debug", "yaml")
        d = metrics.compute().to_dict()
        for key in (
            "total_decisions", "learned_count", "yaml_count", "learned_ratio",
            "avg_ml_confidence", "avg_ml_confidence_learned", "avg_ml_confidence_yaml",
            "class_agreement_rate", "top_disagreements", "decisions_by_source", "window_hours",
        ):
            assert key in d, f"Missing key: {key}"
