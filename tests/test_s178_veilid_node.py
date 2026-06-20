#!/usr/bin/env python3
"""Tests for S178 Goal 1 -- the Veilid node lifecycle (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/guard.py and opti_oignon/veilid/node.py:

- The lifecycle state machine: start -> started -> attach -> attached -> detach
  -> started -> stop -> stopped, with idempotent no-ops and rejected transitions.
- The Bulbe-refusal invariant: start and attach are refused at the binding layer
  under Bulbe (and when the mode is undeterminable, fail-secure); detach and stop
  are never gated, so a node can always leave the network and shut down.
- Fail-secure transitions: an underlying connector failure settles the node in a
  truthful state (ERROR, or back to STARTED for a failed attach) and surfaces a
  typed VeilidError, never an arbitrary one; stop always ends STOPPED.
- The state surface, the module singleton, and the default-connector resolution
  (None when the veilid framework is absent, so start refuses cleanly).

Loaded via spec_from_file_location with opti_oignon stubbed. The security mode is
read from a stubbed opti_oignon.security_mode so Daily / Bulbe / undeterminable
are driven deterministically; the audit log is stubbed to a no-op so the suite
leaves nothing on disk and never touches the real chain.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"


# Stubs: the package, a settable security mode, and a no-op audit log.

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily", *, raises: bool = False) -> None:
    """Drive the stubbed security mode for the next guard call."""
    if raises:
        def _gm() -> str:
            raise RuntimeError("mode undeterminable")
    else:
        def _gm() -> str:
            return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def _ensure_stubs():
    for name, sub in (("opti_oignon", OO), ("opti_oignon.veilid", VEILID)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod
    if "opti_oignon.security_mode" not in sys.modules:
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.get_current_mode = _MODE["fn"]  # type: ignore[attr-defined]
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
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load("guard")
node = _load("node")
NodeState = node.NodeState


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    node.reset_node()
    yield
    node.reset_node()
    set_mode("daily")


class FakeConnector:
    """Records calls; can be told to fail a given operation, and reports a state."""

    def __init__(self, *, attachment: str = "AttachedGood") -> None:
        self.calls: list[str] = []
        self.fail_on: set[str] = set()
        self._attachment = attachment

    def _step(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail_on:
            raise RuntimeError(f"{name} boom")

    def connect(self) -> None:
        self._step("connect")

    def attach(self) -> None:
        self._step("attach")

    def detach(self) -> None:
        self._step("detach")

    def shutdown(self) -> None:
        self._step("shutdown")

    def attachment_state(self) -> str:
        return self._attachment


def _started_node():
    fake = FakeConnector()
    n = node.VeilidNode(connector=fake)
    n.start()
    return n, fake


def _attached_node():
    n, fake = _started_node()
    n.attach()
    return n, fake


# Sentinels and framework presence


class TestSentinels:
    def test_flags(self):
        assert guard.checkpoint_before_apply is True
        assert guard.FEATURE_AVAILABLE is True
        assert node.checkpoint_before_apply is True
        assert node.FEATURE_AVAILABLE is True

    def test_veilid_absent_in_sandbox(self):
        # The sandbox has no veilid framework; the lazy probe reports that.
        assert guard.veilid_available() is False


# Happy-path lifecycle


class TestLifecycleHappy:
    def test_full_cycle(self):
        fake = FakeConnector()
        n = node.VeilidNode(connector=fake)
        assert n.state() == NodeState.STOPPED

        n.start()
        assert n.state() == NodeState.STARTED
        assert n.is_running() is True
        assert n.is_attached() is False

        n.attach()
        assert n.state() == NodeState.ATTACHED
        assert n.is_attached() is True

        n.detach()
        assert n.state() == NodeState.STARTED
        assert n.is_attached() is False

        n.stop()
        assert n.state() == NodeState.STOPPED
        assert n.is_running() is False

        assert fake.calls == ["connect", "attach", "detach", "shutdown"]

    def test_start_returns_status(self):
        n, _ = _started_node()
        st = n.status()
        assert st["state"] == "started"
        assert st["running"] is True
        assert st["attached"] is False


# Idempotency


class TestIdempotency:
    def test_double_start_connects_once(self):
        n, fake = _started_node()
        n.start()
        assert fake.calls.count("connect") == 1
        assert n.state() == NodeState.STARTED

    def test_attach_when_attached_is_noop(self):
        n, fake = _attached_node()
        n.attach()
        assert fake.calls.count("attach") == 1

    def test_detach_when_started_is_noop(self):
        n, fake = _started_node()
        n.detach()
        assert "detach" not in fake.calls
        assert n.state() == NodeState.STARTED

    def test_stop_when_stopped_is_noop(self):
        fake = FakeConnector()
        n = node.VeilidNode(connector=fake)
        n.stop()
        assert fake.calls == []
        assert n.state() == NodeState.STOPPED


# Rejected transitions


class TestInvalidTransitions:
    def test_attach_from_stopped_raises(self):
        n = node.VeilidNode(connector=FakeConnector())
        with pytest.raises(node.VeilidStateError):
            n.attach()

    def test_detach_from_attached_only(self):
        # detach from stopped is a state error (not idempotent, since not started)
        n = node.VeilidNode(connector=FakeConnector())
        with pytest.raises(node.VeilidStateError):
            n.detach()

    def test_start_when_attached_is_idempotent(self):
        n, fake = _attached_node()
        n.start()
        assert n.state() == NodeState.ATTACHED
        assert fake.calls.count("connect") == 1


# The Bulbe-refusal invariant


class TestBulbeRefusal:
    def test_start_refused_under_bulbe(self):
        set_mode("bulbe")
        n = node.VeilidNode(connector=FakeConnector())
        with pytest.raises(guard.VeilidDisabledInBulbe):
            n.start()
        assert n.state() == NodeState.STOPPED  # never opened a connection

    def test_attach_refused_under_bulbe(self):
        n, _ = _started_node()  # started in Daily
        set_mode("bulbe")
        with pytest.raises(guard.VeilidDisabledInBulbe):
            n.attach()
        assert n.state() == NodeState.STARTED

    def test_undeterminable_mode_is_fail_secure(self):
        set_mode(raises=True)
        assert guard.bulbe_disabled() is True
        n = node.VeilidNode(connector=FakeConnector())
        with pytest.raises(guard.VeilidDisabledInBulbe):
            n.start()

    def test_detach_not_gated_under_bulbe(self):
        n, fake = _attached_node()  # attached in Daily
        set_mode("bulbe")
        n.detach()  # leaving the network must always be allowed
        assert n.state() == NodeState.STARTED
        assert "detach" in fake.calls

    def test_stop_not_gated_under_bulbe(self):
        n, fake = _attached_node()
        set_mode("bulbe")
        n.stop()  # shutting down must always be allowed
        assert n.state() == NodeState.STOPPED
        assert "shutdown" in fake.calls

    def test_disabled_is_a_veilid_error(self):
        assert issubclass(guard.VeilidDisabledInBulbe, guard.VeilidError)


# Fail-secure transitions


class TestFailSecure:
    def test_connect_failure_settles_in_error(self):
        fake = FakeConnector()
        fake.fail_on = {"connect"}
        n = node.VeilidNode(connector=fake)
        with pytest.raises(node.VeilidError):
            n.start()
        assert n.state() == NodeState.ERROR
        assert n.status()["last_error"]

    def test_retry_start_from_error(self):
        fake = FakeConnector()
        fake.fail_on = {"connect"}
        n = node.VeilidNode(connector=fake)
        with pytest.raises(node.VeilidError):
            n.start()
        fake.fail_on = set()
        n.start()  # ERROR -> STARTING -> STARTED on retry
        assert n.state() == NodeState.STARTED

    def test_attach_failure_reverts_to_started(self):
        fake = FakeConnector()
        n = node.VeilidNode(connector=fake)
        n.start()
        fake.fail_on = {"attach"}
        with pytest.raises(node.VeilidError):
            n.attach()
        assert n.state() == NodeState.STARTED  # node is up, just not attached

    def test_detach_failure_reverts_to_attached(self):
        n, fake = _attached_node()
        fake.fail_on = {"detach"}
        with pytest.raises(node.VeilidError):
            n.detach()
        assert n.state() == NodeState.ATTACHED

    def test_stop_failure_still_ends_stopped(self):
        n, fake = _attached_node()
        fake.fail_on = {"shutdown"}
        with pytest.raises(node.VeilidError):
            n.stop()
        assert n.state() == NodeState.STOPPED  # always ends down


# State surface


class TestStatusShape:
    def test_keys_and_types(self):
        n, _ = _attached_node()
        st = n.status()
        for key in (
            "state",
            "running",
            "attached",
            "attachment",
            "bulbe_disabled",
            "veilid_available",
            "last_error",
        ):
            assert key in st
        assert st["state"] == "attached"
        assert st["attached"] is True
        assert st["attachment"] == "AttachedGood"
        assert st["bulbe_disabled"] is False
        assert st["veilid_available"] is False


# Module singleton


class TestSingleton:
    def test_get_node_is_stable(self):
        a = node.get_node()
        b = node.get_node()
        assert a is b

    def test_set_and_reset(self):
        custom = node.VeilidNode(connector=FakeConnector())
        node.set_node(custom)
        assert node.get_node() is custom
        node.reset_node()
        assert node.get_node() is not custom


# Default connector resolution


class TestDefaultConnector:
    def test_start_without_connector_refuses_when_veilid_absent(self):
        # No connector and no veilid framework -> resolve to None -> VeilidUnavailable.
        n = node.VeilidNode()
        with pytest.raises(node.VeilidUnavailable):
            n.start()
        assert n.state() == NodeState.ERROR

    def test_connector_factory_is_used(self):
        fake = FakeConnector()
        n = node.VeilidNode(connector_factory=lambda: fake)
        n.start()
        assert fake.calls == ["connect"]
        assert n.state() == NodeState.STARTED


# Package facade


class TestPackageFacade:
    def test_init_reexports(self):
        # __init__ wires guard + node behind the package; load it after both.
        init = _load("__init__")
        assert init.FEATURE_AVAILABLE is True
        assert init.VeilidNode is node.VeilidNode
        assert init.NodeState is node.NodeState
        assert init.VeilidDisabledInBulbe is guard.VeilidDisabledInBulbe
        assert callable(init.assert_sync_allowed)
        assert "VeilidNode" in init.__all__
