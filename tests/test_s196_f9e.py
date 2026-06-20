#!/usr/bin/env python3
"""S196 F9e -- functional audit fixes for the client/node wrappers (code-checkable).

One tight test group per fix:

- VLD-01: node singleton creation is lock-guarded (the last unguarded singleton
  of the sub-package; the VL-02 class), so two racing first calls cannot build
  two node state machines.
- VLD-02: a reconnect releases the previous api connection (best-effort)
  instead of leaking it; a failing release never breaks the reconnect, and the
  newly installed api is the active one.

The live transport itself stays shakedown territory; these are the
code-checkable seams. Loader idiom matches the f9a..f9d suites.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily") -> None:
    def _gm() -> str:
        return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def _ensure_stubs() -> None:
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
client = _load("client")


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    node.reset_node()
    yield
    node.reset_node()
    set_mode("daily")


class FakeApi:
    """A minimal api double recording attach/detach/release calls."""

    def __init__(self, log, name):
        self._log = log
        self.name = name

    async def attach(self):
        self._log.append(("attach", self.name))

    async def detach(self):
        self._log.append(("detach", self.name))

    async def release(self):
        self._log.append(("release", self.name))


class ExplodingReleaseApi(FakeApi):
    async def release(self):
        self._log.append(("release_attempt", self.name))
        raise RuntimeError("release failed")


def _factory(log, cls=FakeApi):
    counter = {"n": 0}

    def factory(callback):
        counter["n"] += 1
        return cls(log, f"api{counter['n']}")

    return factory


# --- VLD-01: lock-guarded node singleton --------------------------------------


class TestVLD01NodeSingletonLock:
    def test_source_uses_lock(self):
        src = (VEILID / "node.py").read_text()
        assert "_NODE_LOCK = threading.Lock()" in src
        assert src.count("with _NODE_LOCK:") >= 3  # get / set / reset

    def test_concurrent_get_yields_single_instance(self):
        node.reset_node()
        barrier = threading.Barrier(8)
        seen = []

        def hammer():
            barrier.wait()
            seen.append(node.get_node())

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(n) for n in seen}) == 1


# --- VLD-02: reconnect releases the previous api -------------------------------


class TestVLD02ReconnectReleasesPrevious:
    def test_single_connect_releases_nothing(self):
        log = []
        c = client.VeilidClient(api_factory=_factory(log), timeout=5.0)
        try:
            c.connect()
            assert [e for e in log if e[0] == "release"] == []
        finally:
            c.shutdown()

    def test_reconnect_releases_previous_once(self):
        log = []
        c = client.VeilidClient(api_factory=_factory(log), timeout=5.0)
        try:
            c.connect()
            c.connect()
            assert [e for e in log if e[0] == "release"] == [("release", "api1")]
            # The second api is the active one.
            c.attach()
            assert ("attach", "api2") in log
        finally:
            c.shutdown()
        assert ("release", "api2") in log  # shutdown released the active api

    def test_failing_release_does_not_break_reconnect(self):
        log = []
        c = client.VeilidClient(api_factory=_factory(log, ExplodingReleaseApi), timeout=5.0)
        c.connect()
        c.connect()  # the previous release raises; the reconnect must survive
        assert ("release_attempt", "api1") in log
        c.attach()
        assert ("attach", "api2") in log
        # Shutdown's release also raises for this double; the loop still tears
        # down and the client reports the loop stopped.
        with pytest.raises(guard.VeilidError):
            c.shutdown()
        assert c.is_loop_running() is False
