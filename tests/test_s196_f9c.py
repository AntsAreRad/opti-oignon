#!/usr/bin/env python3
"""S196 F9c -- functional audit fixes for the sync engine, status, and route seams.

One tight test group per fix:

- SYN-02: a persistent per-install device identity (uuid4 hex, minted once in a
  one-row meta table of veilid_peers.db) replaces the universal "local" default
  for engines created without an explicit device -- every device naming itself
  "local" made pairing peer_ids collide and record provenance meaningless.
  Explicit devices and the guarded "local" fallback are preserved.
- SYN-03: a malformed peer answer is now distinguishable from an empty round:
  ``RoundResult.parsed`` is False on an unparseable batch, the route payload
  carries it, and the run handler records a status FAILURE ("malformed answer")
  instead of a clean round. Re-asserts the superseded strict-equality
  ``round_result_to_dict`` test with the new key (deselect-plus-reassert).
- SYN-04: sync-engine singleton creation is lock-guarded (the VL-02 unguarded-
  singleton class), matching the feed/store/status singletons.
- SYN-06: the sync router carries the per-router ``_auth_dep`` (S136 /
  MKT-01 parity); /api/sync stays off the auth middleware's public allowlist.
- SYN-07: ``advanced`` reports strictly-forward movement only; an unpaired-
  mid-round no-op advance (which returns 0) is no longer reported as True.

Loader idiom matches the s180 routes suite (veilid + api package stubs).
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
API = OO / "api"

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily") -> None:
    def _gm() -> str:
        return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def _ensure_stubs() -> None:
    for name, sub in (
        ("opti_oignon", OO),
        ("opti_oignon.veilid", VEILID),
        ("opti_oignon.api", API),
    ):
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


def _load_veilid(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


def _load_api(name: str):
    full = f"opti_oignon.api.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(API / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load_veilid("guard")
records = _load_veilid("records")
reconcile = _load_veilid("reconcile")
change_feed = _load_veilid("change_feed")
peers = _load_veilid("peers")
protocol = _load_veilid("protocol")
producers = _load_veilid("producers")
sync_engine = _load_veilid("sync_engine")
sync_status = _load_veilid("sync_status")
rs = _load_api("routes_sync")
RecordKind = records.RecordKind


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    sync_status.reset_sync_status_store()
    yield
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    sync_status.reset_sync_status_store()
    set_mode("daily")


def _rec(record_id, clock, *, device, payload=None, kind=None):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
    )


def _engine(tmp_path, *, device="A"):
    feed = change_feed.ChangeFeed(root=tmp_path / "local")
    store = peers.PeerStore(root=tmp_path / "store")
    return sync_engine.SyncEngine(device=device, feed=feed, store=store), feed, store


# --- SYN-02: persistent per-install device identity --------------------------


class TestSYN02DeviceIdentity:
    def test_minted_once_and_stable(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "a")
        first = store.local_device_id()
        assert isinstance(first, str) and len(first) == 32
        assert store.local_device_id() == first
        # A new store instance over the same root reads the same identity back.
        again = peers.PeerStore(root=tmp_path / "a")
        assert again.local_device_id() == first

    def test_distinct_installs_distinct_ids(self, tmp_path):
        a = peers.PeerStore(root=tmp_path / "a").local_device_id()
        b = peers.PeerStore(root=tmp_path / "b").local_device_id()
        assert a != b

    def test_engine_default_resolves_persistent_id(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "a")
        eng = sync_engine.get_sync_engine(store=store)
        assert eng.device == store.local_device_id()
        assert eng.device != "local"

    def test_explicit_device_wins(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "a")
        eng = sync_engine.get_sync_engine(device="X", store=store)
        assert eng.device == "X"

    def test_identity_failure_falls_back_local(self):
        class BrokenStore:
            def local_device_id(self):
                raise RuntimeError("no identity")

        eng = sync_engine.get_sync_engine(store=BrokenStore())
        assert eng.device == "local"


# --- SYN-03: malformed answer is distinguishable -----------------------------


class TestSYN03ParsedFlag:
    def test_unparseable_batch_sets_parsed_false(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        eng.register_peer("B", "rk-B")

        class GarbagePeer:
            def fetch(self, request):
                return 123  # not a batch in any shape

        res = eng.run_round("B", GarbagePeer())
        assert res.parsed is False
        assert res.applied == 0
        assert res.advanced is False

    def test_normal_round_parsed_true(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        eng.register_peer("B", "rk-B")
        feed_b = change_feed.ChangeFeed(root=tmp_path / "remote")
        feed_b.record(_rec("b1", 1, device="B"))

        class HonestPeer:
            def fetch(self, request):
                return protocol.respond_to_request(feed_b, request, device="B")

        res = eng.run_round("B", HonestPeer())
        assert res.parsed is True
        assert res.applied == 1

    def test_round_result_to_dict_includes_parsed(self):
        # Re-assertion of the superseded
        # test_s180_routes_sync::TestRoundShaping::test_round_result_to_dict
        # with the new key under strict equality.
        rr = sync_engine.RoundResult(
            peer_id="B",
            applied=2,
            deferred=1,
            conflicts=3,
            rejected=0,
            previous_watermark=1,
            new_watermark=5,
            advanced=True,
        )
        d = rs.round_result_to_dict(rr)
        assert d == {
            "peer_id": "B",
            "applied": 2,
            "deferred": 1,
            "conflicts": 3,
            "rejected": 0,
            "previous_watermark": 1,
            "new_watermark": 5,
            "advanced": True,
            "parsed": True,
        }

    def test_route_records_failure_on_malformed_answer(self, tmp_path):
        # Through the FastAPI handler: a garbage-answering peer yields a 200
        # payload with parsed false, and the status store records a FAILED
        # attempt ("malformed answer"), not a clean round.
        if rs.router is None:
            pytest.skip("fastapi unavailable")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        feed = change_feed.ChangeFeed(root=tmp_path / "local")
        store = peers.PeerStore(root=tmp_path / "store")
        eng = sync_engine.SyncEngine(device="A", feed=feed, store=store)
        change_feed.set_change_feed(feed)
        peers.set_peer_store(store)
        sync_engine.set_sync_engine(eng)
        eng.register_peer("B", "rk-B")

        class GarbagePeer:
            def fetch(self, request):
                return b"\xff\xfe not a batch"

        rs.set_peer_resolver(lambda peer_id, st: GarbagePeer())
        try:
            app = FastAPI()
            app.include_router(rs.router)
            client = TestClient(app)
            resp = client.post("/api/sync/peers/B/run")
            assert resp.status_code == 200
            body = resp.json()
            assert body["parsed"] is False
            assert body["applied"] == 0
            status = sync_status.get_sync_status_store().last_for("B")
            assert status is not None
            assert status.ok is False
            assert status.error == "malformed answer"
        finally:
            rs.reset_peer_resolver()


# --- SYN-04: lock-guarded singleton creation ----------------------------------


class TestSYN04SingletonLock:
    def test_source_uses_lock(self):
        src = (VEILID / "sync_engine.py").read_text()
        assert "_engine_lock = threading.Lock()" in src
        assert src.count("with _engine_lock:") >= 3  # get / set / reset

    def test_concurrent_get_yields_single_instance(self):
        sync_engine.reset_sync_engine()
        barrier = threading.Barrier(8)
        seen = []

        def hammer():
            barrier.wait()
            seen.append(sync_engine.get_sync_engine(device="X"))

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(e) for e in seen}) == 1


# --- SYN-06: per-router auth parity ------------------------------------------


class TestSYN06AuthParity:
    def test_router_carries_auth_dep(self):
        src = (API / "routes_sync.py").read_text()
        assert "from .routes_auth import _get_current_user" in src
        assert "_auth_dep = [Depends(_get_current_user)]" in src
        assert 'dependencies=_auth_dep' in src
        assert 'APIRouter(prefix="/api/sync", tags=["sync"], dependencies=_auth_dep)' in src

    def test_sync_paths_not_public(self):
        auth_mw = _load_api("auth_middleware")
        for path in (
            "/api/sync/status",
            "/api/sync/peers",
            "/api/sync/peers/B/run",
            "/api/sync/pairing/self",
            "/api/sync/pairing/accept",
        ):
            assert auth_mw._is_public_path(path) is False


# --- SYN-07: advanced means strictly forward ----------------------------------


class TestSYN07AdvancedFlag:
    def test_unpaired_mid_round_not_reported_advanced(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        eng.register_peer("B", "rk-B")
        store.advance_watermark("B", 3)
        feed_b = change_feed.ChangeFeed(root=tmp_path / "remote")
        for r in (_rec("b%d" % i, 1, device="B") for i in range(5)):
            feed_b.record(r)

        class VanishingPeer:
            def fetch(self, request):
                store.remove_peer("B")  # unpaired between the check and the advance
                return protocol.respond_to_request(feed_b, request, device="B")

        res = eng.run_round("B", VanishingPeer())
        assert res.advanced is False
        assert res.new_watermark == 0  # the no-op advance's truthful answer

    def test_normal_advance_still_true_then_idle_false(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        eng.register_peer("B", "rk-B")
        feed_b = change_feed.ChangeFeed(root=tmp_path / "remote")
        feed_b.record(_rec("b1", 1, device="B"))

        class HonestPeer:
            def fetch(self, request):
                return protocol.respond_to_request(feed_b, request, device="B")

        first = eng.run_round("B", HonestPeer())
        assert first.advanced is True
        second = eng.run_round("B", HonestPeer())
        assert second.advanced is False
        assert second.applied == 0
