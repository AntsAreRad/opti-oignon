#!/usr/bin/env python3
"""Contracts for the feedback-driven routing score bounds.

Accumulated user feedback nudges model/task scores, but a learning loop fed
by feedback must never let that feedback move a score without bound, activate
on a trickle of entries, act while disabled, or be broken by a single crafted
entry. These contracts pin the guard-rails that make the loop safe to feed
without pinning the learned magnitudes themselves.

  * A1 -- the adjustment saturates at the cap: an arbitrarily strong flood of
    feedback (positive or negative) can only move a score by the ceiling, no
    more. The clamp is what holds the line.
  * A2 -- the cap is not widenable from configuration: the ceiling comes from
    the module constant, never from the feedback config file, so nothing that
    can write that file can also lift the ceiling.
  * A3 -- adjustments require a sample floor: below the configured minimum the
    adjustment is inert and reported inactive; at the floor the same feedback
    activates.
  * A4 -- disabled means zero: with adaptive routing off, every adjustment is
    zero regardless of how favourable the accumulated feedback is.
  * A5 -- a crafted entry cannot break the loop: a non-numeric rating among
    valid entries must not raise, the result stays bounded, and the unusable
    entry does not even count toward the sample floor.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package, with an injected feedback store so no store
import or database is required.
"""

import importlib.util
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load():
    """Load the real adaptive_routing module under a stand-in package.

    Returns (module, restore). The feedback store is injected per test, so the
    lazy store import is never reached and no real store is touched.
    """
    keys = (
        "opti_oignon", "opti_oignon.feedback", "opti_oignon.adaptive_routing",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    fb = types.ModuleType("opti_oignon.feedback")
    fb.feedback_store = None
    sys.modules["opti_oignon.feedback"] = fb
    pkg.feedback = fb

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.adaptive_routing", _OO / "adaptive_routing.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.adaptive_routing"] = mod
    spec.loader.exec_module(mod)
    pkg.adaptive_routing = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


# ---------------------------------------------------------------------------
# Local material
# ---------------------------------------------------------------------------
class _Entry:
    """A feedback entry as the store would yield it."""

    def __init__(
        self, model_used, task_type,
        rating_type="thumbs", rating_value=1, timestamp=None,
    ):
        self.model_used = model_used
        self.task_type = task_type
        self.rating_type = rating_type
        self.rating_value = rating_value
        self.timestamp = time.time() if timestamp is None else timestamp


class _Store:
    """A stand-in feedback store exposing the surface the adapter reads."""

    def __init__(self, entries, auto_adjust=True):
        self._entries = list(entries)
        self.auto_adjust_routing = auto_adjust

    def list_feedback(self, limit=10000):
        return list(self._entries)


def _absent_config():
    """A config path that does not exist, forcing built-in defaults."""
    return Path(tempfile.mkdtemp()) / "absent.yaml"


MODEL = "some-model"
TASK = "code_python"


# ---------------------------------------------------------------------------
# A1 -- adjustment saturates at the cap
# ---------------------------------------------------------------------------
def test_a1_adjustment_is_clamped_to_the_cap():
    mod, restore = _load()
    try:
        # A large factor would drive the raw adjustment far past the cap; the
        # clamp must hold a maximally positive flood at exactly the ceiling.
        up = [_Entry(MODEL, TASK, "thumbs", 1) for _ in range(50)]
        adapter = mod.FeedbackRoutingAdapter(
            feedback_store=_Store(up, auto_adjust=True),
            config_path=_absent_config(), min_samples=1, adjustment_factor=1.0,
        )
        adj = adapter.get_adjustment(MODEL, TASK)
        assert abs(adj) <= adapter.max_adjustment
        assert adj == adapter.max_adjustment, (
            "a positive flood must saturate at the cap, not exceed it"
        )

        # A maximally negative flood saturates at the negative ceiling.
        down = [_Entry(MODEL, TASK, "thumbs", 0) for _ in range(50)]
        adapter_neg = mod.FeedbackRoutingAdapter(
            feedback_store=_Store(down, auto_adjust=True),
            config_path=_absent_config(), min_samples=1, adjustment_factor=1.0,
        )
        adj_neg = adapter_neg.get_adjustment(MODEL, TASK)
        assert abs(adj_neg) <= adapter_neg.max_adjustment
        assert adj_neg == -adapter_neg.max_adjustment, (
            "a negative flood must saturate at the negative cap"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# A2 -- cap cannot be widened from configuration
# ---------------------------------------------------------------------------
def test_a2_cap_cannot_be_widened_by_config():
    mod, restore = _load()
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "feedback.yaml"
    try:
        # Anything able to write this file tries to lift the ceiling far above
        # the built-in constant. The adapter must ignore that value.
        cfg.write_text(
            "feedback:\n"
            "  min_samples_for_adjustment: 1\n"
            "  adjustment_factor: 1.0\n"
            "  max_adjustment: 0.99\n",
            encoding="utf-8",
        )
        up = [_Entry(MODEL, TASK, "thumbs", 1) for _ in range(50)]
        adapter = mod.FeedbackRoutingAdapter(
            feedback_store=_Store(up, auto_adjust=True), config_path=cfg,
        )
        assert adapter.max_adjustment == mod.MAX_ADJUSTMENT, (
            "the cap must come from the module ceiling, never from config"
        )
        adj = adapter.get_adjustment(MODEL, TASK)
        assert abs(adj) <= mod.MAX_ADJUSTMENT, (
            "no config value may let the adjustment exceed the ceiling"
        )
        assert adj == mod.MAX_ADJUSTMENT
    finally:
        restore()


# ---------------------------------------------------------------------------
# A3 -- adjustments require a sample floor
# ---------------------------------------------------------------------------
def test_a3_adjustment_requires_the_sample_floor():
    mod, restore = _load()
    cfg = _absent_config()
    try:
        # Below the floor: favourable feedback must stay inert and inactive.
        few = [_Entry(MODEL, TASK, "thumbs", 1) for _ in range(3)]
        adapter = mod.FeedbackRoutingAdapter(
            feedback_store=_Store(few, auto_adjust=True),
            config_path=cfg, min_samples=10, adjustment_factor=1.0,
        )
        assert adapter.get_adjustment(MODEL, TASK) == 0.0, (
            "below the sample floor the adjustment must be inert"
        )
        state = adapter.get_all_adjustments()
        pair = [
            a for a in state.adjustments
            if a.model == MODEL and a.task_type == TASK
        ]
        assert pair and pair[0].active is False, (
            "an under-floor pair must be reported inactive"
        )

        # At the floor: the same feedback now activates and moves the score.
        enough = [_Entry(MODEL, TASK, "thumbs", 1) for _ in range(10)]
        adapter2 = mod.FeedbackRoutingAdapter(
            feedback_store=_Store(enough, auto_adjust=True),
            config_path=cfg, min_samples=10, adjustment_factor=1.0,
        )
        assert adapter2.get_adjustment(MODEL, TASK) > 0.0, (
            "at the sample floor a favourable trend must activate"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# A4 -- disabled means zero
# ---------------------------------------------------------------------------
def test_a4_disabled_yields_no_adjustment():
    mod, restore = _load()
    try:
        up = [_Entry(MODEL, TASK, "thumbs", 1) for _ in range(50)]
        adapter = mod.FeedbackRoutingAdapter(
            feedback_store=_Store(up, auto_adjust=False),
            config_path=_absent_config(), min_samples=1, adjustment_factor=1.0,
        )
        assert adapter.enabled is False
        assert adapter.get_adjustment(MODEL, TASK) == 0.0, (
            "with adaptive routing disabled every adjustment is zero"
        )
        assert adapter.has_active_adjustments() is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# A5 -- a crafted entry cannot break the loop
# ---------------------------------------------------------------------------
def test_a5_poisoned_rating_does_not_break_adjustment():
    mod, restore = _load()
    cfg = _absent_config()
    try:
        # A single entry with a non-numeric rating sits among valid ones.
        # Computing the adjustment must not raise and must stay bounded.
        entries = [_Entry(MODEL, TASK, "thumbs", 1) for _ in range(20)]
        entries.append(_Entry(MODEL, TASK, "thumbs", "poison"))
        adapter = mod.FeedbackRoutingAdapter(
            feedback_store=_Store(entries, auto_adjust=True),
            config_path=cfg, min_samples=1, adjustment_factor=1.0,
        )
        adj = adapter.get_adjustment(MODEL, TASK)
        assert isinstance(adj, float)
        assert abs(adj) <= adapter.max_adjustment

        # The unusable entry must not count toward the floor: with (floor - 1)
        # valid entries plus one crafted entry, the pair stays under the floor.
        floor = 10
        near = [_Entry(MODEL, TASK, "thumbs", 1) for _ in range(floor - 1)]
        near.append(_Entry(MODEL, TASK, "thumbs", "poison"))
        adapter2 = mod.FeedbackRoutingAdapter(
            feedback_store=_Store(near, auto_adjust=True),
            config_path=cfg, min_samples=floor, adjustment_factor=1.0,
        )
        assert adapter2.get_adjustment(MODEL, TASK) == 0.0, (
            "a crafted entry must not fill the sample floor"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("A1 adjustment clamped to cap", test_a1_adjustment_is_clamped_to_the_cap),
        ("A2 cap not widenable by config", test_a2_cap_cannot_be_widened_by_config),
        ("A3 adjustment requires floor", test_a3_adjustment_requires_the_sample_floor),
        ("A4 disabled yields zero", test_a4_disabled_yields_no_adjustment),
        ("A5 crafted rating cannot break loop", test_a5_poisoned_rating_does_not_break_adjustment),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
