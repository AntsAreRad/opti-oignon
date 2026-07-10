#!/usr/bin/env python3
"""Contracts for the benchmark auto-trigger's opt-in, bounds, and fail-safes.

The auto-trigger is the only component in the benchmark family that acts on
its own: it watches the installed model list and starts benchmark runs on
changes. The safety-relevant properties are strict opt-in (disabled unless
the user enables it), anti-runaway bounds (poll floor, cooldown, resource
guard, skipped changes kept for re-detection), and fail-secure handling of
an indeterminate backend (an empty model list never triggers and never
wipes the baseline). These contracts pin those guards without pinning the
interval values, the profile names, or the diff heuristics. The poll cycle
is driven synchronously so no thread timing is involved.

  * TG1 -- disabled by default: with no configuration the trigger is off
    and not running. Consent is explicit, never assumed.
  * TG2 -- start is refused while disabled: no polling thread can exist
    without the enabled flag.
  * TG3 -- the poll interval has a floor: a zero or tiny configured value
    is raised to the floor, so the watcher can never hot-loop.
  * TG4 -- the cooldown has a floor of zero: a negative value cannot arm a
    time-travel window.
  * TG5 -- the first poll is a baseline: it records the installed set and
    triggers nothing; a later new model triggers exactly one run for
    exactly the added model.
  * TG6 -- a change skipped by the cooldown is re-detected: the skip is
    recorded with its reason and the old snapshot is kept, so the model is
    benchmarked once the cooldown clears instead of being lost.
  * TG7 -- the resource guard blocks under load: with a load ceiling set
    and exceeded, the trigger skips with a load reason and starts nothing.
  * TG8 -- a backend outage is inert: an empty model list neither triggers
    nor commits an empty baseline, so recovery produces no false wave of
    added models.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package, the model list and the run starter are
deterministic stand-ins, so no model backend is required.
"""

import importlib.util
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_KEYS = (
    "opti_oignon",
    "opti_oignon.benchmark_auto_trigger",
    "opti_oignon.benchmark_runner",
)


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load():
    """Load the real auto-trigger under a stand-in package.

    Returns (module, restore). Every instance built by these tests receives
    an injected model-list function and an injected runner, so nothing here
    can reach a real backend or start a real benchmark.
    """
    saved = {k: sys.modules.get(k) for k in _KEYS}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.benchmark_auto_trigger",
        _OO / "benchmark_auto_trigger.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.benchmark_auto_trigger"] = mod
    spec.loader.exec_module(mod)
    pkg.benchmark_auto_trigger = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


class _RunnerRecorder:
    """Stand-in benchmark runner that records start requests."""

    def __init__(self):
        self.calls = []
        self.is_busy = False

    def start_run(self, **kwargs):
        self.calls.append(dict(kwargs))
        return f"run-{len(self.calls)}"


class _ModelList:
    """Mutable deterministic stand-in for the installed model list."""

    def __init__(self, models=None):
        self.current = list(models or [])

    def __call__(self):
        return list(self.current)

    def set(self, models):
        self.current = list(models)


def _make_trigger(mod, runner=None, listing=None):
    """Build a trigger on a fresh temporary config path."""
    tmp = Path(tempfile.mkdtemp(prefix="oo-trig-"))
    return mod.AutoTrigger(
        config_path=tmp / "auto_trigger.yaml",
        benchmark_runner=runner,
        ollama_list_fn=listing,
    )


def _entry(name, digest):
    return {"model": name, "digest": digest}


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
def test_tg1_disabled_by_default():
    mod, restore = _load()
    try:
        trigger = _make_trigger(mod)
        try:
            assert trigger.enabled is False, (
                "with no configuration the trigger must be opt-in (disabled)"
            )
            assert trigger.running is False
        finally:
            trigger.stop()
    finally:
        restore()


def test_tg2_start_refused_while_disabled():
    mod, restore = _load()
    try:
        trigger = _make_trigger(mod)
        try:
            trigger.disable()
            assert trigger.start() is False, (
                "start must refuse while the trigger is disabled"
            )
            assert trigger.running is False
        finally:
            trigger.stop()
    finally:
        restore()


def test_tg3_poll_interval_has_a_floor():
    mod, restore = _load()
    try:
        trigger = _make_trigger(mod)
        try:
            trigger.disable()
            config = trigger.update_config({"poll_interval_seconds": 0})
            assert config["poll_interval_seconds"] >= 10.0, (
                f"poll interval accepted {config['poll_interval_seconds']};"
                " a zero interval would hot-loop the watcher"
            )
            config = trigger.update_config({"poll_interval_seconds": 3.5})
            assert config["poll_interval_seconds"] >= 10.0
        finally:
            trigger.stop()
    finally:
        restore()


def test_tg4_cooldown_has_a_zero_floor():
    mod, restore = _load()
    try:
        trigger = _make_trigger(mod)
        try:
            trigger.disable()
            config = trigger.update_config({"cooldown_seconds": -5})
            assert config["cooldown_seconds"] >= 0.0, (
                f"cooldown accepted {config['cooldown_seconds']};"
                " a negative window must be floored to zero"
            )
        finally:
            trigger.stop()
    finally:
        restore()


def test_tg5_baseline_never_triggers_then_change_triggers_once():
    mod, restore = _load()
    try:
        runner = _RunnerRecorder()
        listing = _ModelList([_entry("model-one", "d1")])
        trigger = _make_trigger(mod, runner=runner, listing=listing)
        try:
            trigger._poll_once()
            assert runner.calls == [], (
                "the first poll is a baseline and must trigger nothing"
            )

            listing.set([_entry("model-one", "d1"), _entry("model-two", "d2")])
            trigger._poll_once()
            assert len(runner.calls) == 1, (
                f"one new model must trigger exactly one run,"
                f" got {len(runner.calls)}"
            )
            assert runner.calls[0]["models"] == ["model-two"], (
                f"the run must target the added model,"
                f" got {runner.calls[0]['models']!r}"
            )
        finally:
            trigger.stop()
    finally:
        restore()


def test_tg6_cooldown_skip_is_recorded_and_redetected():
    mod, restore = _load()
    try:
        runner = _RunnerRecorder()
        listing = _ModelList([_entry("model-one", "d1")])
        trigger = _make_trigger(mod, runner=runner, listing=listing)
        try:
            trigger._poll_once()
            listing.set([_entry("model-one", "d1"), _entry("model-two", "d2")])
            trigger._poll_once()

            listing.set([
                _entry("model-one", "d1"),
                _entry("model-two", "d2"),
                _entry("model-three", "d3"),
            ])
            trigger._poll_once()
            assert len(runner.calls) == 1, (
                "a change inside the cooldown window must not start a run"
            )
            last = trigger.events[-1]
            assert last["skipped"] is True
            assert "cooldown" in last["skip_reason"].lower()

            trigger._last_trigger_time = (
                time.time() - (trigger.config["cooldown_seconds"] + 1.0)
            )
            trigger._poll_once()
            assert len(runner.calls) == 2, (
                "the skipped change must be re-detected once the cooldown"
                " clears; it was consumed instead"
            )
            assert "model-three" in runner.calls[1]["models"]
        finally:
            trigger.stop()
    finally:
        restore()


def test_tg7_resource_guard_blocks_under_load():
    mod, restore = _load()
    try:
        runner = _RunnerRecorder()
        listing = _ModelList([])
        trigger = _make_trigger(mod, runner=runner, listing=listing)
        try:
            trigger.update_config({"resource_guard_load_max": 0.5})

            with mock.patch("os.getloadavg", return_value=(99.0, 0.0, 0.0)):
                event = trigger._do_trigger(["model-nine"], "new_model")
            assert event.skipped is True, (
                "load above the ceiling must skip the trigger"
            )
            assert "load" in event.skip_reason.lower()
            assert runner.calls == []

            with mock.patch("os.getloadavg", return_value=(0.1, 0.0, 0.0)):
                event = trigger._do_trigger(["model-nine"], "new_model")
            assert event.skipped is False, (
                "load below the ceiling must let the trigger proceed"
            )
            assert len(runner.calls) == 1
        finally:
            trigger.stop()
    finally:
        restore()


def test_tg8_backend_outage_never_triggers_or_wipes_baseline():
    mod, restore = _load()
    try:
        runner = _RunnerRecorder()
        listing = _ModelList([_entry("model-one", "d1")])
        trigger = _make_trigger(mod, runner=runner, listing=listing)
        try:
            trigger._poll_once()
            runner.calls.clear()

            listing.set([])
            outcome = trigger._poll_once()
            assert outcome is None
            assert runner.calls == [], (
                "an empty model list must never trigger anything"
            )
            assert trigger.status["known_models"] == 1, (
                "an outage must not wipe the known baseline"
            )

            listing.set([_entry("model-one", "d1")])
            trigger._poll_once()
            assert runner.calls == [], (
                "recovery of the same models must not read as a false wave"
                " of added models"
            )
            assert trigger.status["known_models"] == 1
        finally:
            trigger.stop()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("TG1 disabled by default",
         test_tg1_disabled_by_default),
        ("TG2 start refused while disabled",
         test_tg2_start_refused_while_disabled),
        ("TG3 poll interval has a floor",
         test_tg3_poll_interval_has_a_floor),
        ("TG4 cooldown has a zero floor",
         test_tg4_cooldown_has_a_zero_floor),
        ("TG5 baseline never triggers then change triggers once",
         test_tg5_baseline_never_triggers_then_change_triggers_once),
        ("TG6 cooldown skip is recorded and redetected",
         test_tg6_cooldown_skip_is_recorded_and_redetected),
        ("TG7 resource guard blocks under load",
         test_tg7_resource_guard_blocks_under_load),
        ("TG8 backend outage never triggers or wipes baseline",
         test_tg8_backend_outage_never_triggers_or_wipes_baseline),
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
