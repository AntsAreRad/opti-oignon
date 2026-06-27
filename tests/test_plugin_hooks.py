#!/usr/bin/env python3
"""Tests for the plugin hook system (plugin_hooks.HookManager + S124 redaction).

The hook manager is what makes plugin effects apply during inference, so its
contracts are load-bearing AND security-sensitive:

  * HK-01 -- each hook receives its OWN copy of the chain data; in-place
    mutation never propagates, only a RETURNED dict is merged downstream.
  * HK-02 -- a returned value equal to the redaction placeholder is never
    merged back, so a permission-limited plugin cannot overwrite the real
    prompt/response by echoing the placeholder it received.
  * Error isolation -- one hook raising must not stop the others.

Each test uses a fresh ``HookManager()`` (never the module singleton) so the
cases are fully isolated.
"""

from opti_oignon.plugin_hooks import (
    REDACTED_PLACEHOLDER,
    HookManager,
    redact_hook_data,
)
from opti_oignon.plugin_manifest import VALID_HOOKS

# A hook point that is guaranteed valid (kept explicit so the tests document
# which names the manager accepts).
HOOK = "pre_inference"
assert HOOK in VALID_HOOKS


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------

def test_register_valid_hook_returns_true():
    hm = HookManager()
    assert hm.register(HOOK, "p1", lambda ctx: None) is True
    assert hm.get_hook_count(HOOK) == 1
    assert hm.has_hooks(HOOK) is True


def test_register_invalid_hook_name_returns_false():
    hm = HookManager()
    assert hm.register("not_a_real_hook", "p1", lambda ctx: None) is False
    assert hm.get_hook_count() == 0


def test_register_non_callable_returns_false():
    hm = HookManager()
    assert hm.register(HOOK, "p1", "not callable") is False
    assert hm.get_hook_count() == 0


def test_get_hook_count_total_vs_per_hook():
    hm = HookManager()
    hm.register("pre_inference", "p1", lambda ctx: None)
    hm.register("post_inference", "p1", lambda ctx: None)
    assert hm.get_hook_count("pre_inference") == 1
    assert hm.get_hook_count() == 2


# ----------------------------------------------------------------------------
# Execution order / priority
# ----------------------------------------------------------------------------

def test_execute_runs_in_priority_order():
    hm = HookManager()
    order: list[str] = []
    hm.register(HOOK, "low_prio", lambda ctx: order.append("A") or None, priority=100)
    hm.register(HOOK, "high_prio", lambda ctx: order.append("B") or None, priority=10)
    hm.execute(HOOK)
    assert order == ["B", "A"]  # lower priority number runs first


def test_execute_same_priority_preserves_registration_order():
    hm = HookManager()
    order: list[str] = []
    hm.register(HOOK, "first", lambda ctx: order.append("1") or None, priority=50)
    hm.register(HOOK, "second", lambda ctx: order.append("2") or None, priority=50)
    hm.execute(HOOK)
    assert order == ["1", "2"]


# ----------------------------------------------------------------------------
# HK-01: isolation of the chain data
# ----------------------------------------------------------------------------

def test_hk01_in_place_mutation_does_not_propagate():
    hm = HookManager()
    seen: dict[str, object] = {}

    def first(ctx):
        ctx.set("injected", "x")   # mutates this hook's LOCAL copy only
        return None                # ... and does not return it

    def second(ctx):
        seen["value"] = ctx.get("injected")
        return None

    hm.register(HOOK, "first", first, priority=10)
    hm.register(HOOK, "second", second, priority=20)
    report = hm.execute(HOOK, data={})

    assert seen["value"] is None          # not visible to the next hook
    assert "injected" not in report.final_data


def test_returned_dict_merges_into_chain_and_final_data():
    hm = HookManager()
    seen: dict[str, object] = {}

    def first(ctx):
        return {"injected": "x"}           # returned -> propagates

    def second(ctx):
        seen["value"] = ctx.get("injected")
        return None

    hm.register(HOOK, "first", first, priority=10)
    hm.register(HOOK, "second", second, priority=20)
    report = hm.execute(HOOK, data={})

    assert seen["value"] == "x"
    assert report.final_data["injected"] == "x"


# ----------------------------------------------------------------------------
# HK-02: the redaction placeholder is never merged back (security)
# ----------------------------------------------------------------------------

def test_hk02_returned_placeholder_is_never_merged():
    hm = HookManager()

    def echoes_placeholder(ctx):
        # A permission-limited plugin received the placeholder and echoes it
        # while also returning a legitimate field.
        return {"message": REDACTED_PLACEHOLDER, "extra": "real"}

    hm.register(HOOK, "echo", echoes_placeholder)
    report = hm.execute(HOOK, data={"message": "the real user prompt"})

    # The real prompt is preserved; the placeholder did NOT overwrite it.
    assert report.final_data["message"] == "the real user prompt"
    # The legitimate field still propagates.
    assert report.final_data["extra"] == "real"


# ----------------------------------------------------------------------------
# Error isolation + return handling
# ----------------------------------------------------------------------------

def test_execute_isolates_hook_exception():
    hm = HookManager()
    ran_after: list[str] = []

    def boom(ctx):
        raise ValueError("kaboom")

    def after(ctx):
        ran_after.append("ok")
        return None

    hm.register(HOOK, "boom", boom, priority=10)
    hm.register(HOOK, "after", after, priority=20)
    report = hm.execute(HOOK)

    assert ran_after == ["ok"]            # the second hook still ran
    assert report.failed == 1
    assert report.successful == 1
    boom_result = next(r for r in report.results if r.plugin_name == "boom")
    assert boom_result.success is False
    assert boom_result.error is not None
    assert boom_result.error.startswith("ValueError")


def test_non_dict_return_is_not_merged_but_succeeds():
    hm = HookManager()
    hm.register(HOOK, "weird", lambda ctx: "a string, not a dict")
    report = hm.execute(HOOK, data={"k": "v"})
    assert report.successful == 1
    assert report.failed == 0
    assert report.final_data == {"k": "v"}   # unchanged


# ----------------------------------------------------------------------------
# Report shape + empty case
# ----------------------------------------------------------------------------

def test_execute_with_no_hooks_returns_empty_report():
    hm = HookManager()
    report = hm.execute(HOOK, data={"k": "v"})
    assert report.total_hooks == 0
    assert report.successful == 0
    assert report.failed == 0
    assert report.results == []
    assert report.final_data == {"k": "v"}


def test_report_counts_and_results_length():
    hm = HookManager()
    hm.register(HOOK, "p1", lambda ctx: None)
    hm.register(HOOK, "p2", lambda ctx: None)
    report = hm.execute(HOOK)
    assert report.total_hooks == 2
    assert report.successful == 2
    assert len(report.results) == 2


# ----------------------------------------------------------------------------
# list_hooks / set_hook_enabled
# ----------------------------------------------------------------------------

def test_list_hooks_filtering():
    hm = HookManager()
    hm.register("pre_inference", "p1", lambda ctx: None, priority=5)
    hm.register("post_inference", "p1", lambda ctx: None)
    hm.register("pre_inference", "p2", lambda ctx: None)

    by_hook = hm.list_hooks(hook_name="pre_inference")
    assert {h["plugin_name"] for h in by_hook} == {"p1", "p2"}

    by_plugin = hm.list_hooks(plugin_name="p1")
    assert {h["hook_name"] for h in by_plugin} == {"pre_inference", "post_inference"}

    entry = hm.list_hooks(hook_name="pre_inference", plugin_name="p1")[0]
    assert entry["priority"] == 5
    assert entry["enabled"] is True


def test_disabled_hook_is_skipped_in_execute():
    hm = HookManager()
    calls: list[str] = []
    hm.register(HOOK, "on", lambda ctx: calls.append("on") or None)
    hm.register(HOOK, "off", lambda ctx: calls.append("off") or None)
    assert hm.set_hook_enabled(HOOK, "off", False) is True

    report = hm.execute(HOOK)
    assert calls == ["on"]                 # disabled hook did not run
    assert report.total_hooks == 2         # still counted as registered
    assert report.successful == 1
    assert len(report.results) == 1


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------

def test_stats_accumulate_calls_and_errors():
    hm = HookManager()
    hm.register(HOOK, "counter", lambda ctx: None)

    def boom(ctx):
        raise RuntimeError("x")

    hm.register(HOOK, "boomer", boom)
    for _ in range(3):
        hm.execute(HOOK)

    stats = hm.get_stats()
    assert stats[f"counter:{HOOK}"]["calls"] == 3
    assert stats[f"counter:{HOOK}"]["errors"] == 0
    assert stats[f"boomer:{HOOK}"]["calls"] == 3
    assert stats[f"boomer:{HOOK}"]["errors"] == 3


def test_reset_stats_and_clear():
    hm = HookManager()
    hm.register(HOOK, "p1", lambda ctx: None)
    hm.execute(HOOK)
    assert hm.get_stats() != {}

    hm.reset_stats()
    assert hm.get_stats() == {}
    assert hm.get_hook_count() == 1        # reset_stats keeps the hooks

    hm.clear()
    assert hm.get_hook_count() == 0        # clear removes hooks too
    assert hm.get_stats() == {}


# ----------------------------------------------------------------------------
# unregister
# ----------------------------------------------------------------------------

def test_unregister_and_unregister_plugin_counts():
    hm = HookManager()
    hm.register("pre_inference", "p1", lambda ctx: None)
    hm.register("post_inference", "p1", lambda ctx: None)
    hm.register("pre_inference", "p2", lambda ctx: None)

    assert hm.unregister("pre_inference", "p1") == 1
    assert hm.get_hook_count("pre_inference") == 1   # p2 remains

    removed = hm.unregister_plugin("p1")             # the post_inference one
    assert removed == 1
    assert hm.get_hook_count() == 1                  # only p2 left

    assert hm.unregister("pre_inference", "nobody") == 0


# ----------------------------------------------------------------------------
# S124 redaction helper
# ----------------------------------------------------------------------------

def test_redact_hook_data_force_redacts_sensitive_fields():
    data = {
        "message": "secret prompt",
        "response": "secret reply",
        "arguments": "secret args",
        "result": "secret result",
        "model": "qwen",
        "tokens_in": 7,
    }
    out = redact_hook_data(data, "any_plugin", force_redact=True)

    assert out["message"] == REDACTED_PLACEHOLDER
    assert out["response"] == REDACTED_PLACEHOLDER
    assert out["arguments"] == REDACTED_PLACEHOLDER
    assert out["result"] == REDACTED_PLACEHOLDER
    # non-sensitive fields pass through untouched
    assert out["model"] == "qwen"
    assert out["tokens_in"] == 7
    # the original dict is not mutated
    assert data["message"] == "secret prompt"


def test_redact_sensitive_path_hides_prompt_from_unpermissioned_plugin():
    # An unknown plugin name has no registry record -> no inference_content
    # permission -> redact_sensitive must hand it the placeholder, not the
    # real prompt.
    hm = HookManager()
    seen: dict[str, object] = {}
    hm.register(
        HOOK,
        "unknown_unpermissioned_plugin",
        lambda ctx: seen.__setitem__("message", ctx.get("message")) or None,
    )
    hm.execute(HOOK, data={"message": "the real prompt"}, redact_sensitive=True)
    assert seen["message"] == REDACTED_PLACEHOLDER
