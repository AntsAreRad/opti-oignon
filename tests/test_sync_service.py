#!/usr/bin/env python3
"""Tests for the Veilid sync auto-driver (SYN-01 orchestration).

The driver is the missing trigger: the producers journal locally on every save,
but nothing pulls a round, so nothing moves between paired devices. This suite
covers the loop logic of `sync_service.SyncService.run_once` and the
start/stop lifecycle, with every dependency injected so no heavy veilid module
(or live transport) is imported:

  * the conservative gate -- OPT-IN and HARD-STOPPED in Bulbe -- short-circuits
    to a no-op;
  * confirmed peers only (a pending peer is skipped, never contacted);
  * a transport-down peer (resolver returns None) records a failure and the
    pass continues;
  * a per-peer round error is recorded and the pass continues to the next peer;
  * start()/stop()/is_running() drive a daemon thread cleanly.

The live round itself is host-bound and is proven on the maintainer's machine;
here the resolver and engine are fakes. Local-only (the public distribution
ships no tests). Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Isolated loading: the bare sync_service module, no veilid heavy deps
# ---------------------------------------------------------------------------
def _load():
    """Return the sync_service module loaded in isolation.

    The module imports every veilid dependency lazily inside its resolvers and
    takes its gates as injectable callables, so exec'ing the file pulls only
    stdlib. A throwaway `opti_oignon` package stub keeps the absolute module
    name resolvable without triggering the real package __init__ (which needs
    ollama). sys.modules is saved/restored so sibling suites stay clean.
    """
    keys = ("opti_oignon", "opti_oignon.veilid", "opti_oignon.veilid.sync_service")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    vpkg = types.ModuleType("opti_oignon.veilid")
    vpkg.__path__ = []
    sys.modules["opti_oignon.veilid"] = vpkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.veilid.sync_service", _OO / "veilid" / "sync_service.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.veilid.sync_service"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


# ---------------------------------------------------------------------------
# Fakes (attribute-shaped, the only surface the driver touches)
# ---------------------------------------------------------------------------
class _FakePeer:
    def __init__(self, peer_id, *, pending=False):
        self.peer_id = peer_id
        self.pending = pending
        self.routing_key = "rk-" + peer_id


class _FakeStore:
    def __init__(self, peers):
        self._peers = list(peers)

    def list_peers(self):
        return list(self._peers)


class _FakeEngine:
    def __init__(self, *, raise_for=None):
        self.device = "device-A"
        self.rounds = []  # peer_ids run_round was called with
        self._raise_for = set(raise_for or ())

    def run_round(self, peer_id, peer, **kw):
        self.rounds.append(peer_id)
        if peer_id in self._raise_for:
            raise RuntimeError("round boom for " + peer_id)
        # A RoundResult-shaped dict; record_round reads by attr-or-key.
        return {"peer_id": peer_id, "applied": 1, "new_watermark": 5, "advanced": True}


class _FakeStatus:
    def __init__(self):
        self.rounds = []      # summaries passed to record_round
        self.failures = []    # (peer_id, error) passed to record_failure

    def record_round(self, summary, *, at=None):
        self.rounds.append(summary)
        return summary

    def record_failure(self, peer_id, error, *, at=None):
        self.failures.append((peer_id, error))
        return (peer_id, error)


def _service(mod, *, peers, enabled=True, bulbe=False, resolver=None, engine=None):
    """Build a SyncService wired entirely to fakes."""
    engine = engine if engine is not None else _FakeEngine()
    status = _FakeStatus()
    store = _FakeStore(peers)
    if resolver is None:
        def resolver(peer_id, **kw):  # default: transport up, returns a live peer
            return object()
    svc = mod.SyncService(
        interval_seconds=1.0,  # the module minimum; run_once never sleeps, stop() interrupts the wait
        engine=engine,
        store=store,
        status=status,
        node=object(),
        peer_resolver=resolver,
        enabled_fn=lambda: enabled,
        bulbe_fn=lambda: bulbe,
    )
    return svc, engine, status, store


# ---------------------------------------------------------------------------
# run_once: the gate
# ---------------------------------------------------------------------------
def test_run_once_noop_when_disabled():
    mod, restore = _load()
    try:
        svc, engine, status, _ = _service(
            mod, peers=[_FakePeer("B")], enabled=False
        )
        assert svc.run_once() == 0
        assert engine.rounds == []
        assert status.rounds == [] and status.failures == []
    finally:
        restore()


def test_run_once_noop_in_bulbe():
    mod, restore = _load()
    try:
        svc, engine, status, _ = _service(
            mod, peers=[_FakePeer("B")], enabled=True, bulbe=True
        )
        assert svc.run_once() == 0
        assert engine.rounds == []
        assert status.rounds == [] and status.failures == []
    finally:
        restore()


def test_run_once_noop_without_peers():
    mod, restore = _load()
    try:
        svc, engine, status, _ = _service(mod, peers=[])
        assert svc.run_once() == 0
        assert engine.rounds == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# run_once: the rounds
# ---------------------------------------------------------------------------
def test_run_once_rounds_confirmed_peers_only():
    mod, restore = _load()
    try:
        peers = [_FakePeer("B"), _FakePeer("C"), _FakePeer("P", pending=True)]
        svc, engine, status, _ = _service(mod, peers=peers)
        assert svc.run_once() == 2
        assert engine.rounds == ["B", "C"]            # the pending peer is skipped
        assert len(status.rounds) == 2
        assert status.failures == []
    finally:
        restore()


def test_run_once_records_failure_on_transport_down():
    mod, restore = _load()
    try:
        svc, engine, status, _ = _service(
            mod, peers=[_FakePeer("B")], resolver=lambda peer_id, **kw: None
        )
        assert svc.run_once() == 0
        assert engine.rounds == []                    # no live peer -> no round
        assert status.failures and status.failures[0][0] == "B"
    finally:
        restore()


def test_run_once_continues_past_round_error():
    mod, restore = _load()
    try:
        engine = _FakeEngine(raise_for={"B"})
        svc, engine, status, _ = _service(
            mod, peers=[_FakePeer("B"), _FakePeer("C")], engine=engine
        )
        assert svc.run_once() == 1                     # B failed, C succeeded
        assert engine.rounds == ["B", "C"]             # both attempted
        assert len(status.rounds) == 1                 # only C recorded a round
        assert status.failures and status.failures[0][0] == "B"
    finally:
        restore()


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
def test_start_stop_lifecycle():
    mod, restore = _load()
    try:
        # Disabled gate keeps every tick a cheap no-op; we only assert the
        # thread lifecycle, not peer contact.
        svc, _, _, _ = _service(mod, peers=[_FakePeer("B")], enabled=False)
        assert svc.is_running() is False
        assert svc.start() is True
        assert svc.is_running() is True
        assert svc.start() is False                    # idempotent while running
        svc.stop()
        assert svc.is_running() is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# arming seam (the lifespan calls these)
# ---------------------------------------------------------------------------
class _FakeService:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return True

    def stop(self, *, timeout=5.0):
        self.stopped = True


def test_arm_if_enabled_starts_when_on():
    mod, restore = _load()
    try:
        fake = _FakeService()
        out = mod.arm_if_enabled(factory=lambda: fake, enabled_fn=lambda: True)
        assert out is True
        assert fake.started is True
    finally:
        restore()


def test_arm_if_enabled_noop_when_off():
    mod, restore = _load()
    try:
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return _FakeService()

        out = mod.arm_if_enabled(factory=factory, enabled_fn=lambda: False)
        assert out is False
        assert calls["n"] == 0                         # no service even constructed
    finally:
        restore()


def test_arm_if_enabled_swallows_errors():
    mod, restore = _load()
    try:
        def boom():
            raise RuntimeError("factory boom")

        out = mod.arm_if_enabled(factory=boom, enabled_fn=lambda: True)
        assert out is False                            # never raises out of arming
    finally:
        restore()


def test_reset_stops_current_service():
    mod, restore = _load()
    try:
        fake = _FakeService()
        mod.set_sync_service(fake)
        mod.reset_sync_service()
        assert fake.stopped is True
        assert mod.get_sync_service() is not fake      # singleton cleared
    finally:
        restore()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'} - {failures} failure(s)")
    sys.exit(1 if failures else 0)
