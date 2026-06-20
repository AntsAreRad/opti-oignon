#!/usr/bin/env python3
"""Integration tests for S178 -- node + client composed (Theme 4, Goal 5).

Two layers:

- Composition with fakes: the node lifecycle (Goal 1) driving the real async
  client wrapper (Goal 3) over an injected fake api. This proves the two halves
  fit -- start/attach/detach/stop flow through the loop-bridge, the Bulbe gate
  refuses before the client is ever touched, and a client timeout surfaces to
  the node as a controlled failure (state reverts, never a leaked exception).

- An ephemeral real-veilid integration test, guarded with importorskip and a
  best-effort connect: it skips cleanly when the veilid framework or a running
  veilid-server is absent (the sandbox and CI case), so it never adds baseline
  noise, but exercises a genuine node when both are present.

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode is
stubbed to Daily and the audit log to a no-op, so the suite leaves nothing on
disk.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"

_MODE = {"value": "daily"}


def set_mode(value: str = "daily") -> None:
    _MODE["value"] = value
    sys.modules["opti_oignon.security_mode"].get_current_mode = lambda: _MODE["value"]  # type: ignore[attr-defined]


def _ensure_stubs():
    for name, sub in (("opti_oignon", OO), ("opti_oignon.veilid", VEILID)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod
    if "opti_oignon.security_mode" not in sys.modules:
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.get_current_mode = lambda: _MODE["value"]  # type: ignore[attr-defined]
        sys.modules["opti_oignon.security_mode"] = sm
    if "opti_oignon.signed_audit_log" not in sys.modules:
        al = types.ModuleType("opti_oignon.signed_audit_log")
        al.chain_log = lambda **kwargs: None  # type: ignore[attr-defined]
        sys.modules["opti_oignon.signed_audit_log"] = al


def _load(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load("guard")
node_mod = _load("node")
client_mod = _load("client")
VeilidClient = client_mod.VeilidClient
VeilidNode = node_mod.VeilidNode
NodeState = node_mod.NodeState


@pytest.fixture(autouse=True)
def _daily():
    set_mode("daily")
    yield
    set_mode("daily")


def _update(name: str):
    return types.SimpleNamespace(
        attachment=types.SimpleNamespace(state=types.SimpleNamespace(name=name))
    )


class FakeAPI:
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
        self.callback(_update("Detached"))

    async def release(self):
        self.calls.append("release")


def make_factory(**kwargs):
    holder: dict[str, FakeAPI] = {}

    def factory(callback):
        api = FakeAPI(callback, **kwargs)
        holder["api"] = api
        return api

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


# Composition with fakes


class TestNodeOverClient:
    def test_lifecycle_through_the_bridge(self):
        factory = make_factory()
        client = VeilidClient(api_factory=factory)
        node = VeilidNode(connector=client)

        node.start()
        assert node.state() == NodeState.STARTED
        assert client.is_loop_running() is True

        node.attach()
        assert node.state() == NodeState.ATTACHED
        # The node's status surfaces the client's reported attachment state.
        assert node.status()["attachment"] == "AttachedGood"

        node.detach()
        assert node.state() == NodeState.STARTED

        node.stop()
        assert node.state() == NodeState.STOPPED
        assert client.is_loop_running() is False
        assert factory.holder["api"].calls == ["attach", "detach", "release"]

    def test_bulbe_refuses_before_client_is_touched(self):
        set_mode("bulbe")
        factory = make_factory()
        client = VeilidClient(api_factory=factory)
        node = VeilidNode(connector=client)
        with pytest.raises(guard.VeilidDisabledInBulbe):
            node.start()
        assert node.state() == NodeState.STOPPED
        # The binding-layer gate fired first: no loop, no api.
        assert client.is_loop_running() is False
        assert "api" not in factory.holder

    def test_client_timeout_surfaces_as_node_error(self):
        factory = make_factory(attach_sleep=3.0)
        client = VeilidClient(api_factory=factory, timeout=0.3)
        node = VeilidNode(connector=client)
        node.start()
        try:
            with pytest.raises(node_mod.VeilidError):
                node.attach()
            # A controlled failure: the node reverts to STARTED and records why.
            assert node.state() == NodeState.STARTED
            assert node.status()["last_error"]
        finally:
            node.stop()


# Ephemeral real-veilid integration (skips cleanly without framework / server)


class TestEphemeralReal:
    def test_ephemeral_node_when_available(self):
        pytest.importorskip("veilid")
        client = VeilidClient(timeout=3.0)
        node = VeilidNode(connector=client)
        try:
            node.start()
        except guard.VeilidError as exc:
            pytest.skip(f"no running veilid-server available: {exc}")
        try:
            assert node.is_running() is True
            node.attach()
            assert node.is_attached() is True
            node.detach()
        finally:
            node.stop()
            assert node.state() == NodeState.STOPPED
