#!/usr/bin/env python3
"""What the six-zone budget engine promises about its allocation.

This engine is the reference allocator: the executor sizes the prompt
with it, the orchestrator delegates to it, and the budget routes expose
it. Its arithmetic is pinned here at the digit on a fixed window so a
drift in any rule surfaces as a red instead of a silently reshaped
prompt.

Pinned rules: the balanced split on a standard window, with and without
an active project; the redistribution of a withheld project share into
history and user by their own proportions; the fingerprint zone carved
out of history when it is active; the overflow absorber that trims the
reserve down to its floor first and history second; the explicit
zero-project override that opts out of redistribution entirely; and the
floors on a tiny window, where the honest outcome is an allocation
larger than the window itself -- reported as such, never hidden. Two
override mechanisms coexist on this engine: the keyword argument it
accepts natively, and callers that rewrite the ratio attributes around a
call. Both are pinned equivalent on the same inputs, which is the ground
the unified caller stands on. Window resolution pins the override, the
prefix fallback and the ultimate default, plus the cache that keeps a
resolved window until it is cleared.

Loaded through the shared isolation window with the model runtime
declared unreachable, so window resolution is deterministic everywhere.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.prompt_optimization"

_CFG = {
    "allocation": {
        "system_ratio": 0.10, "project_ratio": 0.25, "history_ratio": 0.40,
        "user_ratio": 0.10, "reserve_ratio": 0.15, "fingerprint_ratio": 0.025,
    },
    "fallback_context_windows": {},
    "default_context_window": 8192,
    "minimum_budgets": {
        "system": 256, "project": 0, "history": 512, "user": 256, "reserve": 512,
    },
}

_RAG = {
    "system_ratio": 0.10, "project_ratio": 0.35, "history_ratio": 0.30,
    "user_ratio": 0.10, "reserve_ratio": 0.15,
}


def _load():
    """Load the real module with the model runtime unreachable."""
    loaded, restore = isolate(
        targets={_TARGET: source("prompt_optimization.py")},
        blocked=("ollama",),
    )
    return loaded[_TARGET], restore


def _manager(module):
    return module.PromptTokenBudgetManager(config=dict(_CFG))


def _patched_call(manager, **kwargs):
    """Replay the attribute-rewrite override mechanism around one call."""
    overrides = kwargs.pop("ratios")
    saved = (
        manager._system_ratio, manager._project_ratio, manager._history_ratio,
        manager._user_ratio, manager._reserve_ratio,
    )
    manager._system_ratio = overrides["system_ratio"]
    manager._project_ratio = overrides["project_ratio"]
    manager._history_ratio = overrides["history_ratio"]
    manager._user_ratio = overrides["user_ratio"]
    manager._reserve_ratio = overrides["reserve_ratio"]
    try:
        return manager.calculate_budget(**kwargs)
    finally:
        (manager._system_ratio, manager._project_ratio, manager._history_ratio,
         manager._user_ratio, manager._reserve_ratio) = saved


def _shape(budget):
    return (budget.system_tokens, budget.project_tokens, budget.history_tokens,
            budget.user_tokens, budget.reserve_tokens, budget.fingerprint_tokens)


# ---------------------------------------------------------------------------
# f1 -- the balanced split on a standard window with a project active
# ---------------------------------------------------------------------------

def test_f1_balanced_split_with_project_on_the_standard_window():
    module, restore = _load()
    try:
        budget = _manager(module).calculate_budget(
            model="m", project_active=True, context_window_override=8192,
        )
        assert _shape(budget) == (819, 2048, 3276, 819, 1228, 0)
        assert budget.total_allocated == 8190
        assert budget.total_window == 8192
        assert budget.as_dict()["utilization"] == 0.9998
    finally:
        restore()


# ---------------------------------------------------------------------------
# f2 -- with no project, the withheld share flows to history and user
# ---------------------------------------------------------------------------

def test_f2_withheld_project_share_flows_to_history_and_user():
    module, restore = _load()
    try:
        budget = _manager(module).calculate_budget(
            model="m", project_active=False, context_window_override=8192,
        )
        assert _shape(budget) == (819, 0, 4915, 1228, 1228, 0)
        assert budget.total_allocated == 8190
    finally:
        restore()


# ---------------------------------------------------------------------------
# f3 -- the fingerprint zone is carved out of history
# ---------------------------------------------------------------------------

def test_f3_fingerprint_zone_is_carved_out_of_history():
    module, restore = _load()
    try:
        budget = _manager(module).calculate_budget(
            model="m", project_active=True, context_window_override=8192,
            fingerprint_active=True,
        )
        assert _shape(budget) == (819, 2048, 3072, 819, 1228, 204)
        assert budget.total_allocated == 8190
    finally:
        restore()


# ---------------------------------------------------------------------------
# f4 -- overflow is absorbed by the reserve first, history second
# ---------------------------------------------------------------------------

def test_f4_overflow_is_absorbed_by_reserve_first_history_second():
    module, restore = _load()
    try:
        heavy = {
            "system_ratio": 0.10, "project_ratio": 0.40, "history_ratio": 0.40,
            "user_ratio": 0.10, "reserve_ratio": 0.15,
        }
        budget = _manager(module).calculate_budget(
            model="m", project_active=True, context_window_override=8192,
            priority_overrides=heavy,
        )
        # The reserve gave up to its floor, then history absorbed the rest,
        # and the result lands exactly on the window.
        assert _shape(budget) == (819, 3276, 2766, 819, 512, 0)
        assert budget.total_allocated == 8192
    finally:
        restore()


# ---------------------------------------------------------------------------
# f5 -- the native override keyword equals the attribute-rewrite mechanism
# ---------------------------------------------------------------------------

def test_f5_native_override_keyword_equals_attribute_rewrite():
    module, restore = _load()
    try:
        manager = _manager(module)
        for active in (True, False):
            native = manager.calculate_budget(
                model="m", project_active=active, context_window_override=8192,
                priority_overrides=dict(_RAG),
            )
            patched = _patched_call(
                manager, ratios=dict(_RAG),
                model="m", project_active=active, context_window_override=8192,
            )
            assert native == patched
        # The rewrite left no residue: a plain call is balanced again.
        after = manager.calculate_budget(
            model="m", project_active=True, context_window_override=8192,
        )
        assert _shape(after) == (819, 2048, 3276, 819, 1228, 0)
    finally:
        restore()


# ---------------------------------------------------------------------------
# f6 -- an explicit zero project override opts out of redistribution
# ---------------------------------------------------------------------------

def test_f6_explicit_zero_project_override_opts_out_of_redistribution():
    module, restore = _load()
    try:
        no_project = {
            "system_ratio": 0.10, "project_ratio": 0.0, "history_ratio": 0.40,
            "user_ratio": 0.10, "reserve_ratio": 0.15,
        }
        budget = _manager(module).calculate_budget(
            model="m", project_active=False, context_window_override=8192,
            priority_overrides=no_project,
        )
        # History and user keep exactly their stated shares: nothing flowed.
        assert _shape(budget) == (819, 0, 3276, 819, 1228, 0)
    finally:
        restore()


# ---------------------------------------------------------------------------
# f7 -- on a tiny window the floors win and the excess is reported honestly
# ---------------------------------------------------------------------------

def test_f7_floors_win_on_a_tiny_window_and_excess_is_reported():
    module, restore = _load()
    try:
        budget = _manager(module).calculate_budget(
            model="m", project_active=False, context_window_override=1024,
        )
        assert _shape(budget) == (256, 0, 512, 256, 512, 0)
        assert budget.total_allocated == 1536
        assert budget.total_window == 1024
        assert budget.utilization == 1.5
    finally:
        restore()


# ---------------------------------------------------------------------------
# f8 -- window resolution: override, prefix fallback, ultimate default
# ---------------------------------------------------------------------------

def test_f8_window_resolution_override_prefix_and_default():
    module, restore = _load()
    try:
        manager = _manager(module)
        forced = manager.calculate_budget(
            model="m", project_active=True, context_window_override=4096,
        )
        assert forced.total_window == 4096
        manager._fallbacks = {"aa:1b": 2048}
        assert manager.get_context_window("aa:1b-q4") == 2048
        assert manager.get_context_window("totally-x") == 8192
    finally:
        restore()


# ---------------------------------------------------------------------------
# f9 -- a resolved window stays cached until the cache is cleared
# ---------------------------------------------------------------------------

def test_f9_resolved_window_stays_cached_until_cleared():
    module, restore = _load()
    try:
        manager = _manager(module)
        assert manager.get_context_window("x") == 8192
        manager._fallbacks = {"x": 4096}
        assert manager.get_context_window("x") == 8192  # served from the cache
        assert manager.clear_cache() == 1
        assert manager.get_context_window("x") == 4096
    finally:
        restore()
