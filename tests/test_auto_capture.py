#!/usr/bin/env python3
"""Tests for automatic memory capture (M2).

``auto_capture.maybe_capture`` is the trigger the new memory store was missing:
after an assistant turn it fires the (already-built) ``extract_and_store`` so
durable facts accumulate without the manual ``/extract`` route. It is gated
(``OPTI_AUTO_CAPTURE``, default on), throttled (fire only when the conversation
has grown by ``min_new`` messages since the last capture, tracked by an
in-memory watermark), and fire-and-forget (a daemon thread; never blocks or
breaks the turn). This suite loads ``auto_capture.py`` in isolation (the
extraction dispatch injected as a synchronous recorder) and proves:

  * it fires once when the growth threshold is met, with the messages passed;
  * it is throttled below the threshold, and fires again after enough new
    messages;
  * the gate (``OPTI_AUTO_CAPTURE=0``) disables it;
  * an empty conversation / missing id is a no-op;
  * a dispatch error is swallowed (the turn is never broken);
  * the default runner path is safe (returns without raising).

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load():
    keys = ("opti_oignon", "opti_oignon.memory", "opti_oignon.memory.auto_capture")
    saved = {k: sys.modules.get(k) for k in keys}
    for n in ("opti_oignon", "opti_oignon.memory"):
        pkg = types.ModuleType(n)
        pkg.__path__ = []
        sys.modules[n] = pkg
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.memory.auto_capture", _OO / "memory" / "auto_capture.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.memory.auto_capture"] = mod
    spec.loader.exec_module(mod)
    mod.reset_auto_capture()

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


class _Recorder:
    def __init__(self, *, raises=False):
        self.calls = []
        self._raises = raises

    def __call__(self, messages, *, user_id=None, model=None):
        self.calls.append({"messages": messages, "user_id": user_id, "model": model})
        if self._raises:
            raise RuntimeError("boom")


def _msgs(n):
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def _env(value):
    """Context-less env setter returning the prior value for restore."""
    prior = os.environ.get("OPTI_AUTO_CAPTURE")
    if value is None:
        os.environ.pop("OPTI_AUTO_CAPTURE", None)
    else:
        os.environ["OPTI_AUTO_CAPTURE"] = value
    return prior


def test_fires_when_threshold_met():
    mod, restore = _load()
    prior = _env("1")
    try:
        rec = _Recorder()
        msgs = _msgs(6)
        fired = mod.maybe_capture("conv1", msgs, min_new=6, runner=rec)
        assert fired is True
        assert len(rec.calls) == 1
        assert rec.calls[0]["messages"] == msgs
    finally:
        _env(prior)
        restore()


def test_throttled_below_threshold():
    mod, restore = _load()
    prior = _env("1")
    try:
        rec = _Recorder()
        assert mod.maybe_capture("conv1", _msgs(6), min_new=6, runner=rec) is True
        # only 3 more messages since last capture (< 6) -> no fire
        assert mod.maybe_capture("conv1", _msgs(9), min_new=6, runner=rec) is False
        assert len(rec.calls) == 1
    finally:
        _env(prior)
        restore()


def test_fires_again_after_enough_new():
    mod, restore = _load()
    prior = _env("1")
    try:
        rec = _Recorder()
        assert mod.maybe_capture("conv1", _msgs(6), min_new=6, runner=rec) is True
        assert mod.maybe_capture("conv1", _msgs(12), min_new=6, runner=rec) is True
        assert len(rec.calls) == 2
    finally:
        _env(prior)
        restore()


def test_gate_off_disables():
    mod, restore = _load()
    prior = _env("0")
    try:
        rec = _Recorder()
        assert mod.maybe_capture("conv1", _msgs(20), min_new=6, runner=rec) is False
        assert rec.calls == []
    finally:
        _env(prior)
        restore()


def test_empty_or_missing_is_noop():
    mod, restore = _load()
    prior = _env("1")
    try:
        rec = _Recorder()
        assert mod.maybe_capture("conv1", [], min_new=6, runner=rec) is False
        assert mod.maybe_capture("", _msgs(6), min_new=6, runner=rec) is False
        assert rec.calls == []
    finally:
        _env(prior)
        restore()


def test_dispatch_error_swallowed():
    mod, restore = _load()
    prior = _env("1")
    try:
        rec = _Recorder(raises=True)
        # the runner raises, but maybe_capture must not propagate
        assert mod.maybe_capture("conv1", _msgs(6), min_new=6, runner=rec) is True
        assert len(rec.calls) == 1
    finally:
        _env(prior)
        restore()


def test_default_runner_is_safe():
    mod, restore = _load()
    prior = _env("1")
    try:
        # runner=None -> the real daemon-thread runner; extraction is not
        # importable in isolation, but the job swallows it. Must not raise.
        assert mod.maybe_capture("conv1", _msgs(6), min_new=6) is True
    finally:
        _env(prior)
        restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
