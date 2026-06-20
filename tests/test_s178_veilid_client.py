#!/usr/bin/env python3
"""Tests for S178 Goal 3 -- the async Veilid client wrapper (Theme 4).

Covers opti_oignon/veilid/client.py, the bounded loop-bridge:

- The synchronous connector surface (connect / attach / detach / shutdown /
  attachment_state) drives an injected fake api on a dedicated loop thread.
- Timeouts: a slow operation raises VeilidTimeout rather than hanging.
- Fail-secure: an underlying error is wrapped as VeilidError (never the raw
  type); attach before connect is a typed VeilidError; the best-effort
  attachment_state read never raises and reflects reported updates.
- Without an injected factory and without the veilid framework, connect refuses
  with VeilidUnavailable; nothing is started.
- The dedicated loop runs on its own thread (off the main loop) and is torn down
  on shutdown; an idle shutdown is a no-op.
- The async surface (aconnect / aattach / ashutdown) is awaitable from another
  event loop and schedules onto the dedicated loop, never the caller's.

Loaded via spec_from_file_location with opti_oignon stubbed. A fake api factory
stands in for veilid.api_connector, so the loop-bridge, the timeouts, and the
fail-secure behaviour are exercised without the framework or a live server.
"""

import asyncio
import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"


def _ensure_stubs():
    for name, sub in (("opti_oignon", OO), ("opti_oignon.veilid", VEILID)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod


def _load(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load("guard")
client_mod = _load("client")
VeilidClient = client_mod.VeilidClient


def _update(name: str):
    return types.SimpleNamespace(
        attachment=types.SimpleNamespace(state=types.SimpleNamespace(name=name))
    )


class FakeAPI:
    """A stand-in for a connected VeilidAPI; records calls, can sleep or fail."""

    def __init__(self, callback, *, attach_sleep=0.0, fail=None, state_name="AttachedGood"):
        self.callback = callback
        self.calls: list[str] = []
        self._attach_sleep = attach_sleep
        self._fail = set(fail or ())
        self._state_name = state_name

    async def attach(self):
        self.calls.append("attach")
        if self._attach_sleep:
            await asyncio.sleep(self._attach_sleep)
        if "attach" in self._fail:
            raise ValueError("attach kaboom")
        self.callback(_update(self._state_name))

    async def detach(self):
        self.calls.append("detach")
        if "detach" in self._fail:
            raise ValueError("detach kaboom")
        self.callback(_update("Detached"))

    async def release(self):
        self.calls.append("release")
        if "release" in self._fail:
            raise ValueError("release kaboom")

    async def get_state(self):
        return _update(self._state_name)


def make_factory(**kwargs):
    """A sync factory returning a FakeAPI; exposes the built api via .holder."""
    holder: dict[str, FakeAPI] = {}

    def factory(callback):
        api = FakeAPI(callback, **kwargs)
        holder["api"] = api
        return api

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


# Sentinels


class TestSentinels:
    def test_flags(self):
        assert client_mod.checkpoint_before_apply is True
        assert client_mod.FEATURE_AVAILABLE is True


# Synchronous surface, happy path


class TestSyncSurface:
    def test_full_cycle(self):
        factory = make_factory()
        c = VeilidClient(api_factory=factory)
        c.connect()
        assert c.is_loop_running() is True
        c.attach()
        assert c.attachment_state() == "AttachedGood"
        c.detach()
        c.shutdown()
        assert c.is_loop_running() is False
        assert factory.holder["api"].calls == ["attach", "detach", "release"]

    def test_attachment_state_default_empty(self):
        c = VeilidClient(api_factory=make_factory())
        assert c.attachment_state() == ""

    def test_shutdown_idle_is_noop(self):
        factory = make_factory()
        c = VeilidClient(api_factory=factory)
        c.shutdown()
        assert c.is_loop_running() is False
        assert "api" not in factory.holder  # loop never spun, api never built

    def test_dedicated_loop_is_off_main_thread(self):
        c = VeilidClient(api_factory=make_factory())
        c.connect()
        try:
            assert c._thread is not None and c._thread.is_alive()
            assert c._thread is not threading.main_thread()
        finally:
            c.shutdown()


# Framework requirement


class TestFrameworkRequirement:
    def test_connect_without_factory_or_framework_refuses(self):
        # No injected factory and no veilid in the sandbox -> refuse, start nothing.
        c = VeilidClient()
        with pytest.raises(guard.VeilidUnavailable):
            c.connect()
        assert c.is_loop_running() is False


# Timeouts


class TestTimeouts:
    def test_slow_operation_times_out(self):
        factory = make_factory(attach_sleep=3.0)
        c = VeilidClient(api_factory=factory, timeout=0.3)
        c.connect()
        try:
            with pytest.raises(guard.VeilidTimeout):
                c.attach()
        finally:
            c.shutdown()

    def test_timeout_is_a_veilid_error(self):
        assert issubclass(guard.VeilidTimeout, guard.VeilidError)


# Fail-secure


class TestFailSecure:
    def test_underlying_error_is_wrapped(self):
        factory = make_factory(fail={"attach"})
        c = VeilidClient(api_factory=factory)
        c.connect()
        try:
            with pytest.raises(guard.VeilidError) as ei:
                c.attach()
            assert not isinstance(ei.value, ValueError)
        finally:
            c.shutdown()

    def test_attach_before_connect_is_typed(self):
        c = VeilidClient(api_factory=make_factory())
        try:
            with pytest.raises(guard.VeilidError):
                c.attach()
        finally:
            c.shutdown()

    def test_attachment_state_reflects_updates_and_never_raises(self):
        factory = make_factory(state_name="FullyAttached")
        c = VeilidClient(api_factory=factory)
        c.connect()
        try:
            assert c.attachment_state() == ""
            c.attach()
            assert c.attachment_state() == "FullyAttached"
            c.detach()
            assert c.attachment_state() == "Detached"
        finally:
            c.shutdown()


# Async surface (awaitable from another loop; runs on the dedicated loop)


class TestAsyncSurface:
    async def test_async_cycle_off_caller_loop(self):
        factory = make_factory()
        c = VeilidClient(api_factory=factory)
        await c.aconnect()
        try:
            await c.aattach()
            assert "attach" in factory.holder["api"].calls
            # The dedicated loop is not the caller's loop: work was offloaded.
            assert c._loop is not asyncio.get_running_loop()
        finally:
            await c.ashutdown()
        assert c.is_loop_running() is False

    async def test_async_timeout(self):
        factory = make_factory(attach_sleep=3.0)
        c = VeilidClient(api_factory=factory, timeout=0.3)
        await c.aconnect()
        try:
            with pytest.raises(guard.VeilidTimeout):
                await c.aattach()
        finally:
            await c.ashutdown()
