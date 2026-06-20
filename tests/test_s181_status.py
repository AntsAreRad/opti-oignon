#!/usr/bin/env python3
"""Tests for S181 Goal 3 -- the sync-status surface (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/sync_status.py (the in-memory last-sync / outcome store)
and the route's status surface (opti_oignon/api/routes_sync.py):

- The store records a completed round (from a RoundResult or a dict) and a failed
  attempt; per-peer last-for and the single most-recent last_round; reset / isolation.
- status_payload shapes the surface: running / attached / the framework and Bulbe
  flags from the node snapshot, the last round across peers, and per-peer last_sync
  and last_round. A failed last attempt leaves last_sync empty.
- peer_status_payload is enriched with last_sync and last_round only when a status
  store is given (the S180 contract is unchanged without one).
- The FastAPI surface: GET /api/sync/status returns the shape; a run records an
  outcome that then shows on the peer's status and on /status; a paired Daily run
  with no transport records a failure and returns 503 (fastapi-guarded).

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode is
driven through a stubbed opti_oignon.security_mode and the audit log is a no-op.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"
API = OO / "api"

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily", *, raises: bool = False) -> None:
    if raises:
        def _gm() -> str:
            raise RuntimeError("mode undeterminable")
    else:
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
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_api(name: str):
    full = f"opti_oignon.api.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(API / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
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
transport = _load_veilid("transport")
sync_status = _load_veilid("sync_status")
rs = _load_api("routes_sync")

RecordKind = records.RecordKind
SyncStatusStore = sync_status.SyncStatusStore
RoundResult = sync_engine.RoundResult


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    sync_status.reset_sync_status_store()
    rs.reset_peer_resolver()
    yield
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    sync_status.reset_sync_status_store()
    rs.reset_peer_resolver()
    set_mode("daily")


def _rec(record_id, clock, *, device, payload=None, kind=None):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
    )


class FakePeer:
    def __init__(self, feed, device):
        self.feed = feed
        self.device = device

    def fetch(self, request):
        return protocol.respond_to_request(self.feed, request, device=self.device)


class FakeNode:
    def __init__(self, **status):
        self._status = status

    def status(self):
        return dict(self._status)


# --- The in-memory store ----------------------------------------------------


class TestStatusStore:
    def test_record_round_from_result(self):
        st = SyncStatusStore()
        rr = RoundResult(
            peer_id="B", applied=2, deferred=0, conflicts=1, rejected=0,
            previous_watermark=0, new_watermark=5, advanced=True,
        )
        out = st.record_round(rr)
        assert out.peer_id == "B"
        assert out.applied == 2 and out.conflicts == 1 and out.advanced is True
        assert out.ok is True and out.error == "" and out.at
        assert st.last_for("B") == out
        assert st.last_round() == out

    def test_record_round_from_dict(self):
        st = SyncStatusStore()
        out = st.record_round(
            {"peer_id": "B", "applied": 1, "advanced": True, "new_watermark": 3}
        )
        assert out.peer_id == "B" and out.applied == 1 and out.new_watermark == 3
        assert out.ok is True

    def test_record_failure(self):
        st = SyncStatusStore()
        out = st.record_failure("B", "timeout")
        assert out.ok is False and out.error == "timeout"
        assert out.advanced is False and out.new_watermark == 0
        assert st.last_for("B") == out

    def test_last_round_is_most_recent(self):
        st = SyncStatusStore()
        st.record_round({"peer_id": "B", "applied": 1})
        last = st.record_round({"peer_id": "C", "applied": 2})
        assert st.last_round() == last
        assert st.last_for("B").applied == 1
        assert st.last_for("C").applied == 2

    def test_last_for_unknown_is_none(self):
        st = SyncStatusStore()
        assert st.last_for("ghost") is None
        assert st.last_for("") is None
        assert st.last_round() is None

    def test_clear_and_count(self):
        st = SyncStatusStore()
        st.record_round({"peer_id": "B", "applied": 1})
        assert st.peer_count() == 1
        st.clear()
        assert st.peer_count() == 0
        assert st.last_round() is None

    def test_singleton_get_set_reset(self):
        st = SyncStatusStore()
        sync_status.set_sync_status_store(st)
        assert sync_status.get_sync_status_store() is st
        sync_status.reset_sync_status_store()
        assert sync_status.get_sync_status_store() is not st


# --- status_payload ---------------------------------------------------------


class TestStatusPayload:
    def test_shape_with_running_node(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "RKB", label="phone")
        ss = SyncStatusStore()
        node = FakeNode(
            running=True, attached=True, attachment="FullyAttached",
            bulbe_disabled=False, veilid_available=True,
        )
        out = rs.status_payload(node=node, store=store, status_store=ss)
        assert out["running"] is True
        assert out["attached"] is True
        assert out["attachment"] == "FullyAttached"
        assert out["bulbe_disabled"] is False
        assert out["veilid_available"] is True
        assert out["last_round"] is None
        assert len(out["peers"]) == 1
        p = out["peers"][0]
        assert p["peer_id"] == "B"
        assert p["last_sync"] == ""  # no round yet
        assert p["last_round"] is None

    def test_reflects_a_recorded_round(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "RKB")
        ss = SyncStatusStore()
        ss.record_round(
            {"peer_id": "B", "applied": 2, "advanced": True, "new_watermark": 4}
        )
        node = FakeNode(running=True, attached=True)
        out = rs.status_payload(node=node, store=store, status_store=ss)
        assert out["last_round"]["peer_id"] == "B"
        assert out["last_round"]["applied"] == 2
        p = out["peers"][0]
        assert p["last_sync"]  # a successful round sets last_sync
        assert p["last_round"]["new_watermark"] == 4

    def test_failed_attempt_leaves_last_sync_empty(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "RKB")
        ss = SyncStatusStore()
        ss.record_failure("B", "timeout")
        node = FakeNode(running=True)
        out = rs.status_payload(node=node, store=store, status_store=ss)
        p = out["peers"][0]
        assert p["last_sync"] == ""  # failure does not count as a sync
        assert p["last_round"]["ok"] is False
        assert p["last_round"]["error"] == "timeout"

    def test_no_node_defaults_to_not_running(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        ss = SyncStatusStore()
        out = rs.status_payload(node=None, store=store, status_store=ss)
        assert out["running"] is False
        assert out["attached"] is False
        assert out["peers"] == []


# --- peer_status_payload enrichment -----------------------------------------


class TestPeerStatusEnrichment:
    def test_unenriched_without_store(self, tmp_path):
        # The S180 contract: no status store -> no extra keys.
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "RKB")
        out = rs.peer_status_payload(store, "B")
        assert out["peer_id"] == "B" and out["routing_key"] == "RKB"
        assert "last_sync" not in out
        assert "last_round" not in out

    def test_enriched_with_store(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "RKB")
        ss = SyncStatusStore()
        ss.record_round({"peer_id": "B", "applied": 1, "advanced": True})
        out = rs.peer_status_payload(store, "B", ss)
        assert out["last_sync"]
        assert out["last_round"]["applied"] == 1

    def test_enriched_missing_peer_still_404s(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        ss = SyncStatusStore()
        with pytest.raises(rs.PeerNotFound):
            rs.peer_status_payload(store, "ghost", ss)


# --- The FastAPI status surface (fastapi-guarded) ---------------------------


class TestStatusRoute:
    @pytest.fixture
    def client(self, tmp_path):
        fastapi = pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        if rs.router is None:  # pragma: no cover - defensive
            pytest.skip("sync router unavailable")
        st = peers.PeerStore(root=tmp_path / "store")
        feed = change_feed.ChangeFeed(root=tmp_path / "local")
        peers.set_peer_store(st)
        sync_engine.set_sync_engine(
            sync_engine.SyncEngine(device="A", feed=feed, store=st)
        )
        sync_status.set_sync_status_store(SyncStatusStore())
        app = fastapi.FastAPI()
        assert rs.register(app) is True
        return TestClient(app), st, tmp_path

    def test_status_route_shape(self, client):
        tc, st, _ = client
        st.add_peer("B", "RKB", label="phone")
        r = tc.get("/api/sync/status")
        assert r.status_code == 200
        body = r.json()
        for key in ("running", "attached", "bulbe_disabled", "veilid_available", "last_round", "peers"):
            assert key in body
        assert len(body["peers"]) == 1
        assert body["peers"][0]["peer_id"] == "B"
        assert body["peers"][0]["last_sync"] == ""

    def test_run_then_status_reflects_outcome(self, client):
        tc, st, tmp_path = client
        st.add_peer("B", "RKB")
        rfeed = change_feed.ChangeFeed(root=tmp_path / "remote")
        rfeed.record(_rec("c1", 1, device="B"))
        rs.set_peer_resolver(lambda pid, store: FakePeer(rfeed, "B"))
        run = tc.post("/api/sync/peers/B/run")
        assert run.status_code == 200
        # the per-peer status is now enriched
        ps = tc.get("/api/sync/peers/B")
        assert ps.status_code == 200
        assert ps.json()["last_round"]["applied"] == 1
        assert ps.json()["last_sync"]
        # and /status carries the last round
        stt = tc.get("/api/sync/status")
        assert stt.json()["last_round"]["peer_id"] == "B"

    def test_failed_run_records_failure(self, client):
        # No resolver and no veilid -> 503 transport-unavailable, recorded as a failure.
        tc, st, _ = client
        st.add_peer("B", "RKB")
        run = tc.post("/api/sync/peers/B/run")
        assert run.status_code == 503
        ps = tc.get("/api/sync/peers/B")
        assert ps.json()["last_round"]["ok"] is False
        assert ps.json()["last_sync"] == ""
