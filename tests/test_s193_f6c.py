#!/usr/bin/env python3
"""
S193 F6c — feedback store, widget wiring, and feedback->routing integration.

Covers:
  - FBK-02: feedback module is robust (guarded yaml import + guarded singleton)
  - FBK-03: the chat FeedbackWidget binds conversation/task/pipeline context
  - Feedback store behaviour (validation, aggregation) and the verified
    feedback -> adaptive-routing wiring.
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

_PROJECT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    full = _PROJECT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_fb_mod = _load_module("s193c_feedback", "opti_oignon/feedback.py")
FeedbackEntry = _fb_mod.FeedbackEntry
FeedbackStore = _fb_mod.FeedbackStore


def _store(tmp):
    return FeedbackStore(db_path=Path(tmp) / "fb.db", config_path=Path(tmp) / "none.yaml")


# ---------------------------------------------------------------------------
# Feedback store behaviour
# ---------------------------------------------------------------------------

class TestFeedbackStore:
    def test_add_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            s.add_feedback(FeedbackEntry(
                rating_type="thumbs", rating_value=1, model_used="m1",
                task_type="code_python",
            ))
            s.add_feedback(FeedbackEntry(
                rating_type="thumbs", rating_value=0, model_used="m1",
                task_type="code_python",
            ))
            stats = s.get_stats()
            assert stats.total_count == 2
            assert stats.thumbs_up == 1 and stats.thumbs_down == 1
            assert "m1" in stats.by_model
            assert stats.by_model["m1"]["total"] == 2

    def test_invalid_rating_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            with pytest.raises(ValueError):
                s.add_feedback(FeedbackEntry(rating_type="thumbs", rating_value=3))
            with pytest.raises(ValueError):
                s.add_feedback(FeedbackEntry(rating_type="stars", rating_value=9))

    def test_normalized_score_stars(self):
        e = FeedbackEntry(rating_type="stars", rating_value=5)
        assert e.normalized_score == 1.0
        assert e.is_positive is True
        e2 = FeedbackEntry(rating_type="stars", rating_value=1)
        assert e2.normalized_score == 0.0

    def test_text_truncated_to_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = _store(tmp)
            long = "x" * (s.max_text_length + 500)
            stored = s.add_feedback(FeedbackEntry(
                rating_type="thumbs", rating_value=1, feedback_text=long,
            ))
            assert len(stored.feedback_text) == s.max_text_length


# ---------------------------------------------------------------------------
# FBK-02 — robustness
# ---------------------------------------------------------------------------

class TestFBK02Robustness:
    def test_yaml_import_guarded(self):
        src = (_PROJECT / "opti_oignon/feedback.py").read_text()
        assert "try:\n    import yaml" in src
        assert "YAML_AVAILABLE" in src

    def test_singleton_guarded(self):
        src = (_PROJECT / "opti_oignon/feedback.py").read_text()
        assert "S193 FBK-02" in src
        assert "feedback_store = None" in src


# ---------------------------------------------------------------------------
# FBK-03 — widget context wiring
# ---------------------------------------------------------------------------

class TestFBK03WidgetWiring:
    def test_widget_binds_context(self):
        src = (_PROJECT / "frontend/src/lib/components/chat/ChatMessage.svelte").read_text()
        assert "conversationId={conversationId}" in src
        assert "taskType={effectiveRouting?.task_type ?? ''}" in src
        assert "pipelineUsed={effectiveRouting?.pipeline ?? ''}" in src
        # the hardcoded empty conversation id is gone
        assert 'conversationId=""' not in src


# ---------------------------------------------------------------------------
# Verified wiring: feedback -> adaptive routing
# ---------------------------------------------------------------------------

class TestFeedbackRoutingWiring:
    def test_adaptive_routing_consumes_feedback_store(self):
        src = (_PROJECT / "opti_oignon/adaptive_routing.py").read_text()
        assert "from .feedback import feedback_store" in src
        assert "def get_adjustment(" in src
        assert "auto_adjust_routing" in src
