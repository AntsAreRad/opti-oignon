#!/usr/bin/env python3
"""Tests for S182 Goal 1 -- the pairing key exchange (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/pairing.py and the pairing surface added to
opti_oignon/api/routes_sync.py:

- The payload round-trips: build -> encode JSON -> decode JSON -> parse, and a
  built payload carries the public material (identity, routing key) plus an
  integrity check over it. The encode side is pure and validating (an empty
  identity or key raises); the decode side is defensive (it never raises).
- Defensive rejection: a tampered routing key or identity no longer matches the
  integrity check and is rejected; a wrong version or type, a missing or mistyped
  field, a non-mapping, and malformed JSON all yield None, never an exception.
- Store population: accepting a valid payload registers the peer through the
  engine's register_peer, and a re-pair (a rotated routing key) preserves the
  watermark and the original pairing time (the store's upsert).
- The Bulbe-mode permission: pairing management -- building this device's payload,
  parsing and accepting a peer's payload, registering / labelling / removing a
  peer -- runs in any mode, never gated; only a round or a served answer is
  Daily-only, and that gate lives elsewhere.
- The audit: accepting a payload records the registration in the hash-chain audit
  log (through register_peer).
- The route helpers (web-free): self_pairing_payload, accept_pairing (400 path via
  InvalidPairing), relabel_peer_payload (PeerNotFound on a miss), and the
  injectable self-routing-key resolver (a fixed key in tests; the transport stub
  returns None without an attached node).
- A fastapi-guarded class drives the live pairing surface end to end and skips
  where fastapi is absent (the sandbox), keeping it out of the regression baseline.

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode is
driven through a stubbed opti_oignon.security_mode and the audit log is a no-op or
a recorder. No live transport: the routing key is injected.
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
_AUDIT: list = []


def set_mode(value: str = "daily", *, raises: bool = False) -> None:
    if raises:
        def _gm() -> str:
            raise RuntimeError("mode undeterminable")
    else:
        def _gm() -> str:
            return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def _record_audit(**kwargs) -> None:
    _AUDIT.append(kwargs)


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
node = _load_veilid("node")
transport = _load_veilid("transport")
pairing = _load_veilid("pairing")
rs = _load_api("routes_sync")

VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    _AUDIT.clear()
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    sys.modules["opti_oignon.signed_audit_log"].chain_log = lambda **k: None  # type: ignore[attr-defined]
    yield
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    rs.reset_self_routing_resolver()
    set_mode("daily")


def _engine(tmp_path, device="A"):
    feed = change_feed.ChangeFeed(root=tmp_path / "local")
    store = peers.PeerStore(root=tmp_path / "store")
    return sync_engine.SyncEngine(device=device, feed=feed, store=store), store


# The pure encode / decode round-trip


class TestPayloadRoundTrip:
    def test_build_carries_public_material(self):
        p = pairing.build_pairing_payload("B", "route-key-B")
        assert p["v"] == pairing.PAIRING_FORMAT_VERSION
        assert p["type"] == pairing.PAIRING_TYPE
        assert p["peer_id"] == "B"
        assert p["routing_key"] == "route-key-B"
        assert isinstance(p["integrity"], str) and len(p["integrity"]) == 64

    def test_integrity_is_deterministic(self):
        a = pairing.pairing_integrity("B", "route-key-B")
        b = pairing.pairing_integrity("B", "route-key-B")
        assert a == b

    def test_integrity_changes_with_key(self):
        a = pairing.pairing_integrity("B", "route-key-B")
        b = pairing.pairing_integrity("B", "route-key-C")
        assert a != b

    def test_integrity_changes_with_identity(self):
        a = pairing.pairing_integrity("B", "route-key-B")
        b = pairing.pairing_integrity("C", "route-key-B")
        assert a != b

    def test_parse_round_trips(self):
        p = pairing.build_pairing_payload("B", "route-key-B")
        parsed = pairing.parse_pairing_payload(p)
        assert parsed is not None
        assert parsed.peer_id == "B"
        assert parsed.routing_key == "route-key-B"

    def test_json_round_trips(self):
        p = pairing.build_pairing_payload("B", "route-key-B")
        text = pairing.encode_pairing_json(p)
        assert isinstance(text, str)
        parsed = pairing.decode_pairing_json(text)
        assert parsed is not None
        assert parsed.peer_id == "B"
        assert parsed.routing_key == "route-key-B"

    def test_verify_true_for_valid(self):
        p = pairing.build_pairing_payload("B", "k")
        assert pairing.verify_pairing_payload(p) is True


# The encode side is pure and validating


class TestEncodeValidates:
    def test_empty_peer_id_raises(self):
        with pytest.raises(ValueError):
            pairing.build_pairing_payload("", "k")

    def test_empty_routing_key_raises(self):
        with pytest.raises(ValueError):
            pairing.build_pairing_payload("B", "")

    def test_non_string_peer_id_raises(self):
        with pytest.raises(ValueError):
            pairing.build_pairing_payload(123, "k")  # type: ignore[arg-type]


# The decode side is defensive: it never raises, and a tampered payload is rejected


class TestDecodeDefensive:
    def test_tampered_routing_key_rejected(self):
        p = pairing.build_pairing_payload("B", "route-key-B")
        p["routing_key"] = "route-key-EVIL"  # integrity no longer matches
        assert pairing.parse_pairing_payload(p) is None

    def test_tampered_identity_rejected(self):
        p = pairing.build_pairing_payload("B", "route-key-B")
        p["peer_id"] = "C"
        assert pairing.parse_pairing_payload(p) is None

    def test_tampered_integrity_rejected(self):
        p = pairing.build_pairing_payload("B", "route-key-B")
        p["integrity"] = "0" * 64
        assert pairing.parse_pairing_payload(p) is None

    def test_wrong_version_rejected(self):
        p = pairing.build_pairing_payload("B", "k")
        p["v"] = 999
        assert pairing.parse_pairing_payload(p) is None

    def test_wrong_type_rejected(self):
        p = pairing.build_pairing_payload("B", "k")
        p["type"] = "not_a_pairing"
        assert pairing.parse_pairing_payload(p) is None

    def test_missing_field_rejected(self):
        p = pairing.build_pairing_payload("B", "k")
        del p["routing_key"]
        assert pairing.parse_pairing_payload(p) is None

    def test_non_mapping_rejected(self):
        for bad in (None, 42, "string", ["list"], object()):
            assert pairing.parse_pairing_payload(bad) is None

    def test_bad_json_rejected(self):
        assert pairing.decode_pairing_json("{not json") is None
        assert pairing.decode_pairing_json("[1,2,3]") is None
        assert pairing.decode_pairing_json("null") is None

    def test_empty_integrity_rejected(self):
        p = pairing.build_pairing_payload("B", "k")
        p["integrity"] = ""
        assert pairing.parse_pairing_payload(p) is None


# Accepting a payload populates the peer store through the engine


class TestStorePopulation:
    def test_accept_registers_peer(self, tmp_path):
        eng, store = _engine(tmp_path)
        p = pairing.build_pairing_payload("B", "route-key-B")
        rec = pairing.accept_pairing_payload(eng, p, label="laptop")
        assert rec is not None
        assert store.has_peer("B")
        stored = store.get_peer("B")
        assert stored.routing_key == "route-key-B"
        assert stored.label == "laptop"
        assert stored.watermark == 0

    def test_accept_invalid_registers_nothing(self, tmp_path):
        eng, store = _engine(tmp_path)
        p = pairing.build_pairing_payload("B", "route-key-B")
        p["routing_key"] = "tampered"
        assert pairing.accept_pairing_payload(eng, p) is None
        assert not store.has_peer("B")

    def test_repair_preserves_watermark(self, tmp_path):
        eng, store = _engine(tmp_path)
        pairing.accept_pairing_payload(
            eng, pairing.build_pairing_payload("B", "key-1"), label="old"
        )
        store.advance_watermark("B", 7)
        # Re-pair with a rotated routing key and a new label.
        pairing.accept_pairing_payload(
            eng, pairing.build_pairing_payload("B", "key-2"), label="new"
        )
        stored = store.get_peer("B")
        assert stored.routing_key == "key-2"
        assert stored.label == "new"
        assert stored.watermark == 7  # preserved across the re-pair

    def test_repair_preserves_pairing_time(self, tmp_path):
        eng, store = _engine(tmp_path)
        pairing.accept_pairing_payload(eng, pairing.build_pairing_payload("B", "k1"))
        first_added = store.get_peer("B").added_at
        pairing.accept_pairing_payload(eng, pairing.build_pairing_payload("B", "k2"))
        assert store.get_peer("B").added_at == first_added

    def test_non_string_label_coerced(self, tmp_path):
        eng, store = _engine(tmp_path)
        rec = pairing.accept_pairing_payload(
            eng, pairing.build_pairing_payload("B", "k"), label=None  # type: ignore[arg-type]
        )
        assert rec is not None
        assert store.get_peer("B").label == ""


# Pairing management is permitted in any mode (it is never gated)


class TestBulbePermission:
    def test_build_under_bulbe(self):
        set_mode("bulbe")
        p = pairing.build_pairing_payload("B", "k")  # no raise
        assert p["peer_id"] == "B"

    def test_parse_under_bulbe(self):
        set_mode("bulbe")
        p = pairing.build_pairing_payload("B", "k")
        assert pairing.parse_pairing_payload(p) is not None

    def test_accept_registers_under_bulbe(self, tmp_path):
        set_mode("bulbe")
        eng, store = _engine(tmp_path)
        rec = pairing.accept_pairing_payload(eng, pairing.build_pairing_payload("B", "k"))
        assert rec is not None
        assert store.has_peer("B")

    def test_unpair_under_bulbe(self, tmp_path):
        set_mode("bulbe")
        eng, store = _engine(tmp_path)
        pairing.accept_pairing_payload(eng, pairing.build_pairing_payload("B", "k"))
        assert eng.unregister_peer("B") is True
        assert not store.has_peer("B")

    def test_indeterminable_mode_still_permits_management(self, tmp_path):
        set_mode(raises=True)  # fail-secure to bulbe for the wire, still local here
        eng, store = _engine(tmp_path)
        rec = pairing.accept_pairing_payload(eng, pairing.build_pairing_payload("B", "k"))
        assert rec is not None and store.has_peer("B")


# Accepting a payload is audited (through register_peer)


class TestAudit:
    def test_accept_is_audited(self, tmp_path):
        sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
        eng, _store = _engine(tmp_path)
        pairing.accept_pairing_payload(eng, pairing.build_pairing_payload("B", "k"))
        actions = [a.get("action") for a in _AUDIT]
        assert "peer_add" in actions

    def test_rejected_payload_not_audited(self, tmp_path):
        sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
        eng, _store = _engine(tmp_path)
        bad = pairing.build_pairing_payload("B", "k")
        bad["integrity"] = "0" * 64
        assert pairing.accept_pairing_payload(eng, bad) is None
        assert [a for a in _AUDIT if a.get("action") == "peer_add"] == []


# The route helpers (web-free)


class TestRouteHelpers:
    def test_self_pairing_payload_shape(self):
        out = rs.self_pairing_payload("A", "route-A")
        assert out["peer_id"] == "A"
        assert out["routing_key"] == "route-A"
        assert out["payload"]["integrity"]
        assert pairing.decode_pairing_json(out["text"]) is not None

    def test_accept_pairing_ok(self, tmp_path):
        eng, store = _engine(tmp_path)
        out = rs.accept_pairing(eng, pairing.build_pairing_payload("B", "k"), label="phone")
        assert out["peer_id"] == "B"
        assert out["label"] == "phone"
        assert store.has_peer("B")

    def test_accept_pairing_invalid_raises(self, tmp_path):
        eng, _store = _engine(tmp_path)
        bad = pairing.build_pairing_payload("B", "k")
        bad["routing_key"] = "tampered"
        with pytest.raises(rs.InvalidPairing):
            rs.accept_pairing(eng, bad)

    def test_relabel_ok(self, tmp_path):
        eng, store = _engine(tmp_path)
        pairing.accept_pairing_payload(eng, pairing.build_pairing_payload("B", "k"), label="old")
        store.advance_watermark("B", 5)
        out = rs.relabel_peer_payload(store, eng, "B", "renamed")
        assert out["label"] == "renamed"
        assert out["watermark"] == 5  # preserved by the upsert

    def test_relabel_missing_raises(self, tmp_path):
        eng, store = _engine(tmp_path)
        with pytest.raises(rs.PeerNotFound):
            rs.relabel_peer_payload(store, eng, "ghost", "x")

    def test_self_routing_resolver_injected(self, tmp_path):
        eng, _store = _engine(tmp_path)
        rs.set_self_routing_resolver(lambda e: "fixed-key")
        try:
            assert rs.resolve_self_routing_key_for_route(eng) == "fixed-key"
        finally:
            rs.reset_self_routing_resolver()

    def test_self_routing_resolver_failure_is_none(self, tmp_path):
        eng, _store = _engine(tmp_path)

        def _boom(_e):
            raise RuntimeError("nope")

        rs.set_self_routing_resolver(_boom)
        try:
            assert rs.resolve_self_routing_key_for_route(eng) is None
        finally:
            rs.reset_self_routing_resolver()


# The transport self-routing-key resolver returns None without an attached node


class TestTransportSelfRouting:
    def test_returns_none_without_framework(self):
        # veilid is absent in the sandbox, so the resolver returns None.
        assert transport.resolve_self_routing_key() is None

    def test_returns_none_for_detached_node(self):
        class _Detached:
            def is_attached(self):
                return False

        # Even with a (fake) node, a detached node supplies no route; and without
        # the framework veilid_available() is False, so this is None either way.
        assert transport.resolve_self_routing_key(node=_Detached()) is None


# The live FastAPI pairing surface (skips where fastapi is absent)


class TestLivePairingRoute:
    def setup_method(self):
        pytest.importorskip("fastapi")

    def _client(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        feed = change_feed.ChangeFeed(root=tmp_path / "local")
        store = peers.PeerStore(root=tmp_path / "store")
        eng = sync_engine.SyncEngine(device="A", feed=feed, store=store)
        peers.set_peer_store(store)
        sync_engine.set_sync_engine(eng)
        app = FastAPI()
        rs.register(app)
        return TestClient(app), store

    def test_self_503_without_routing_key(self, tmp_path):
        client, _store = self._client(tmp_path)
        rs.reset_self_routing_resolver()
        r = client.get("/api/sync/pairing/self")
        assert r.status_code == 503

    def test_self_200_with_resolver(self, tmp_path):
        client, _store = self._client(tmp_path)
        rs.set_self_routing_resolver(lambda e: "route-A")
        try:
            r = client.get("/api/sync/pairing/self")
            assert r.status_code == 200
            body = r.json()
            assert body["peer_id"] == "A"
            assert body["routing_key"] == "route-A"
            assert body["payload"]["integrity"]
        finally:
            rs.reset_self_routing_resolver()

    def test_accept_200(self, tmp_path):
        client, store = self._client(tmp_path)
        payload = pairing.build_pairing_payload("B", "route-B")
        r = client.post("/api/sync/pairing/accept", json={"payload": payload, "label": "laptop"})
        assert r.status_code == 200
        assert store.has_peer("B")
        assert r.json()["label"] == "laptop"

    def test_accept_400_on_tampered(self, tmp_path):
        client, _store = self._client(tmp_path)
        payload = pairing.build_pairing_payload("B", "route-B")
        payload["routing_key"] = "tampered"
        r = client.post("/api/sync/pairing/accept", json=payload)
        assert r.status_code == 400

    def test_unpair_200_then_404(self, tmp_path):
        client, store = self._client(tmp_path)
        client.post(
            "/api/sync/pairing/accept",
            json=pairing.build_pairing_payload("B", "route-B"),
        )
        assert store.has_peer("B")
        r1 = client.delete("/api/sync/peers/B")
        assert r1.status_code == 200 and r1.json()["removed"] is True
        r2 = client.delete("/api/sync/peers/B")
        assert r2.status_code == 404

    def test_relabel_200_and_404(self, tmp_path):
        client, _store = self._client(tmp_path)
        client.post(
            "/api/sync/pairing/accept",
            json=pairing.build_pairing_payload("B", "route-B"),
        )
        r1 = client.post("/api/sync/peers/B/label", json={"label": "phone"})
        assert r1.status_code == 200 and r1.json()["label"] == "phone"
        r2 = client.post("/api/sync/peers/ghost/label", json={"label": "x"})
        assert r2.status_code == 404
