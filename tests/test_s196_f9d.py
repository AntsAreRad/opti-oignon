#!/usr/bin/env python3
"""S196 F9d -- functional audit fix for pairing (peers + pairing + guard sub-lot).

One tight test group for the single fix:

- PAIR-01: ``accept_pairing_payload`` refuses self-pairing -- a payload whose
  peer_id equals the accepting engine's own device identity is rejected (None /
  HTTP 400) instead of registering the device as its own peer. Other devices
  still register, tamper rejection is untouched, and a duck-typed engine
  without a ``device`` attribute skips the check (the documented guarded
  posture).

peers.py and guard.py carried no fixes in this sub-lot (PEER-01 is recorded and
security-routed; guard verified clean); the loader idiom matches the s182/f9c
suites.
"""

from __future__ import annotations

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
pairing = _load_veilid("pairing")
rs = _load_api("routes_sync")


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


def _engine(tmp_path, device="A"):
    feed = change_feed.ChangeFeed(root=tmp_path / "local")
    store = peers.PeerStore(root=tmp_path / "store")
    return sync_engine.SyncEngine(device=device, feed=feed, store=store), store


class TestPAIR01SelfPairingRefused:
    def test_own_payload_refused(self, tmp_path):
        eng, store = _engine(tmp_path, device="A")
        own = pairing.build_pairing_payload("A", "route-A")
        assert pairing.accept_pairing_payload(eng, own) is None
        assert store.has_peer("A") is False

    def test_other_device_still_accepted(self, tmp_path):
        eng, store = _engine(tmp_path, device="A")
        rec = pairing.accept_pairing_payload(
            eng, pairing.build_pairing_payload("B", "route-B"), label="phone"
        )
        assert rec is not None and rec.peer_id == "B"
        assert store.has_peer("B")

    def test_tamper_rejection_untouched(self, tmp_path):
        eng, store = _engine(tmp_path, device="A")
        bad = pairing.build_pairing_payload("B", "route-B")
        bad["routing_key"] = "tampered"
        assert pairing.accept_pairing_payload(eng, bad) is None
        assert store.has_peer("B") is False

    def test_duck_typed_engine_without_device_skips_check(self, tmp_path):
        seen = {}

        class BareEngine:
            def register_peer(self, peer_id, routing_key, *, label=""):
                seen["peer"] = (peer_id, routing_key, label)
                return ("rec", peer_id)

        out = pairing.accept_pairing_payload(
            BareEngine(), pairing.build_pairing_payload("A", "route-A")
        )
        assert out == ("rec", "A")
        assert seen["peer"][0] == "A"

    def test_route_maps_self_pairing_to_400(self, tmp_path):
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

        app = FastAPI()
        app.include_router(rs.router)
        client = TestClient(app)

        own = pairing.build_pairing_payload("A", "route-A")
        resp = client.post("/api/sync/pairing/accept", json={"payload": own})
        assert resp.status_code == 400
        assert store.has_peer("A") is False

        other = pairing.build_pairing_payload("B", "route-B")
        ok = client.post(
            "/api/sync/pairing/accept", json={"payload": other, "label": "phone"}
        )
        assert ok.status_code == 200
        assert ok.json()["peer_id"] == "B"
        assert store.has_peer("B")
