#!/usr/bin/env python3
"""Tests for S180 Goal 3 -- the sync HTTP route (Theme 4 / Veilid Sync).

Covers opti_oignon/api/routes_sync.py: the read and run surfaces the eventual
sync panel consumes:

- ``GET  /api/sync/peers``                       list the paired peers
- ``GET  /api/sync/peers/{peer_id}``             one peer's status
- ``GET  /api/sync/peers/{peer_id}/watermark``   one peer's watermark
- ``POST /api/sync/peers/{peer_id}/run``         run a pull round

The route logic is web-free (it takes a resolved store or engine and returns plain
payloads), so the contract is exercised in isolation: the list / status / watermark
payload shapes, PeerNotFound on a miss, the pure round-summary shaping, and the run
helper driving a round against an injected fake peer (its PeerNotFound and its
Bulbe refusal propagate for HTTP mapping). A separate, fastapi-guarded class drives
the live surface end to end with a TestClient -- 200s, a 404 for a miss, the 403
Bulbe refusal, and the 503 while the live transport is unavailable (it lands in
S181) -- and skips cleanly where fastapi is absent (the sandbox case), keeping it
out of the regression baseline.

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode is
driven through a stubbed opti_oignon.security_mode and the audit log is a no-op.
The peer is a fake answering from its own local feed -- no live transport.
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
sync_engine = _load_veilid("sync_engine")
rs = _load_api("routes_sync")
RecordKind = records.RecordKind
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    yield
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
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


def _store(tmp_path):
    return peers.PeerStore(root=tmp_path / "store")


def _engine(tmp_path, *, device="A", store=None):
    feed = change_feed.ChangeFeed(root=tmp_path / "local")
    st = store if store is not None else _store(tmp_path)
    return sync_engine.SyncEngine(device=device, feed=feed, store=st), st


def _remote(tmp_path, device="B", seed=()):
    f = change_feed.ChangeFeed(root=tmp_path / "remote")
    for r in seed:
        f.record(r)
    return FakePeer(f, device)


# Module surface (importable without FastAPI)


class TestModuleSurface:
    def test_sentinels(self):
        assert rs.checkpoint_before_apply is True
        assert rs.FEATURE_AVAILABLE is True

    def test_veilid_resolved_in_isolation(self):
        assert rs._SYNC_OK is True

    def test_logic_functions_present(self):
        for name in (
            "list_peers_payload",
            "peer_status_payload",
            "peer_watermark_payload",
            "run_sync_payload",
            "round_result_to_dict",
            "register",
        ):
            assert hasattr(rs, name)

    def test_peer_not_found_is_exception(self):
        assert issubclass(rs.PeerNotFound, Exception)


# Web-free list / status / watermark payloads


class TestReadPayloads:
    def test_list_empty(self, tmp_path):
        st = _store(tmp_path)
        assert rs.list_peers_payload(st) == {"peers": []}

    def test_list_shapes(self, tmp_path):
        st = _store(tmp_path)
        st.add_peer("B", "RKB", label="phone")
        st.advance_watermark("B", 4)
        out = rs.list_peers_payload(st)
        assert len(out["peers"]) == 1
        p = out["peers"][0]
        assert p["peer_id"] == "B"
        assert p["routing_key"] == "RKB"
        assert p["label"] == "phone"
        assert p["watermark"] == 4
        assert "added_at" in p and "updated_at" in p

    def test_status_returns_peer(self, tmp_path):
        st = _store(tmp_path)
        st.add_peer("B", "RKB")
        out = rs.peer_status_payload(st, "B")
        assert out["peer_id"] == "B" and out["routing_key"] == "RKB"

    def test_status_missing_raises(self, tmp_path):
        st = _store(tmp_path)
        with pytest.raises(rs.PeerNotFound):
            rs.peer_status_payload(st, "ghost")

    def test_watermark_payload(self, tmp_path):
        st = _store(tmp_path)
        st.add_peer("B", "RKB")
        st.advance_watermark("B", 9)
        assert rs.peer_watermark_payload(st, "B") == {"peer_id": "B", "watermark": 9}

    def test_watermark_missing_raises(self, tmp_path):
        st = _store(tmp_path)
        with pytest.raises(rs.PeerNotFound):
            rs.peer_watermark_payload(st, "ghost")


# The pure round-summary shaping


class TestRoundShaping:
    def test_round_result_to_dict(self):
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
        }


# The web-free run helper (an injected fake peer; no transport)


class TestRunPayload:
    def test_run_applies_and_advances(self, tmp_path):
        eng, st = _engine(tmp_path)
        st.add_peer("B", "RKB")
        peer = _remote(tmp_path, seed=[_rec("c1", 1, device="B")])
        out = rs.run_sync_payload(eng, "B", peer)
        assert out["peer_id"] == "B"
        assert out["applied"] == 1
        assert out["advanced"] is True
        assert out["new_watermark"] >= 1

    def test_run_unpaired_raises_peer_not_found(self, tmp_path):
        eng, st = _engine(tmp_path)
        peer = _remote(tmp_path, seed=())
        with pytest.raises(rs.PeerNotFound):
            rs.run_sync_payload(eng, "ghost", peer)

    def test_run_refuses_in_bulbe(self, tmp_path):
        eng, st = _engine(tmp_path)
        st.add_peer("B", "RKB")
        peer = _remote(tmp_path, seed=[_rec("c1", 1, device="B")])
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            rs.run_sync_payload(eng, "B", peer)

    def test_run_passes_through_approval_fn(self, tmp_path):
        eng, st = _engine(tmp_path)
        st.add_peer("B", "RKB")
        peer = _remote(
            tmp_path, seed=[_rec("s1", 1, device="B", kind=RecordKind.SKILL)]
        )
        out = rs.run_sync_payload(eng, "B", peer, approval_fn=lambda c, t, a: False)
        assert out["deferred"] == 1
        assert out["advanced"] is False


# Live FastAPI surface (skips cleanly where fastapi is absent)


class TestFastApiWiring:
    """End-to-end through the real router with a TestClient; fastapi-guarded."""

    @pytest.fixture
    def client(self, tmp_path):
        fastapi = pytest.importorskip("fastapi")
        pytest.importorskip("httpx")  # TestClient transport
        from fastapi.testclient import TestClient

        if rs.router is None:  # pragma: no cover - defensive
            pytest.skip("sync router unavailable")
        st = peers.PeerStore(root=tmp_path / "store")
        feed = change_feed.ChangeFeed(root=tmp_path / "local")
        peers.set_peer_store(st)
        sync_engine.set_sync_engine(
            sync_engine.SyncEngine(device="A", feed=feed, store=st)
        )
        app = fastapi.FastAPI()
        assert rs.register(app) is True
        return TestClient(app), st

    def test_list_route(self, client):
        tc, st = client
        st.add_peer("B", "RKB", label="phone")
        r = tc.get("/api/sync/peers")
        assert r.status_code == 200
        ids = {p["peer_id"] for p in r.json()["peers"]}
        assert ids == {"B"}

    def test_status_route(self, client):
        tc, st = client
        st.add_peer("B", "RKB")
        r = tc.get("/api/sync/peers/B")
        assert r.status_code == 200
        assert r.json()["routing_key"] == "RKB"

    def test_status_missing_is_404(self, client):
        tc, st = client
        r = tc.get("/api/sync/peers/ghost")
        assert r.status_code == 404

    def test_watermark_route(self, client):
        tc, st = client
        st.add_peer("B", "RKB")
        st.advance_watermark("B", 6)
        r = tc.get("/api/sync/peers/B/watermark")
        assert r.status_code == 200
        assert r.json() == {"peer_id": "B", "watermark": 6}

    def test_watermark_missing_is_404(self, client):
        tc, st = client
        r = tc.get("/api/sync/peers/ghost/watermark")
        assert r.status_code == 404

    def test_run_unpaired_is_404(self, client):
        tc, st = client
        r = tc.post("/api/sync/peers/ghost/run")
        assert r.status_code == 404

    def test_run_paired_daily_is_503_transport_pending(self, client):
        # The live transport lands in S181; a paired peer in Daily reports 503.
        tc, st = client
        st.add_peer("B", "RKB")
        r = tc.post("/api/sync/peers/B/run")
        assert r.status_code == 503

    def test_run_refuses_in_bulbe_403(self, client):
        tc, st = client
        st.add_peer("B", "RKB")
        set_mode("bulbe")
        r = tc.post("/api/sync/peers/B/run")
        assert r.status_code == 403
