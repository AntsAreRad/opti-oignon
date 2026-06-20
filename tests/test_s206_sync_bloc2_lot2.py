#!/usr/bin/env python3
"""S206 per-fix tests -- sync cycle Bloc 2 lot 2 (PAIR-02 mutual confirmation).

The lot: the pairing ceremony completed by a human comparison. A short
confirmation code derived from BOTH devices' canonical public material
(order-normalized, length-prefix framed, scrypt-hardened, 8 decimal digits);
the ceremony registers a peer PENDING in an additive nullable column (NULL/0
confirmed -- the pre-PAIR-02 grandfather by construction -- 1 fresh pending, 2
demoted on a signing-key change); a pending entry gates everything (rounds,
serving with an identity, and the verification key lookup, which refuses
rather than grace-admits); only an explicit confirm activates; rejection
removes only a pending row. The production self-payload surface now threads
this device's signing public key (closing the S205 gap where no real pairing
ever registered one) and pins the generated material so the code recomputes
from local disk in any mode.

liboqs is absent in the container, so signing runs through the injectable
signer seam with the deterministic HMAC fake; the real ML-DSA-65 path is
host-verified by the shakedown's crypto item.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import importlib
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

# The established sync-suite isolation harness (S180/S204/S205 idiom): stub the
# opti_oignon package roots with __path__ so submodules load from disk without
# the heavy package __init__ (the ollama-requiring chain), and drive the
# security mode through a stubbed opti_oignon.security_mode.

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"
API = OO / "api"

_MODE = {"fn": (lambda: "daily")}
_AUDIT = {"events": []}


def set_mode(value: str = "daily", *, raises: bool = False) -> None:
    if raises:
        def _gm() -> str:
            raise RuntimeError("mode undeterminable")
    else:
        def _gm() -> str:
            return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def _record_audit(**kwargs):
    _AUDIT["events"].append(kwargs)


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
        al.chain_log = _record_audit  # type: ignore[attr-defined]
        sys.modules["opti_oignon.signed_audit_log"] = al


_ensure_stubs()

signing_mod = importlib.import_module("opti_oignon.veilid.signing")
guard = importlib.import_module("opti_oignon.veilid.guard")
_change_feed_mod = importlib.import_module("opti_oignon.veilid.change_feed")
_peers_mod = importlib.import_module("opti_oignon.veilid.peers")
_pairing_mod = importlib.import_module("opti_oignon.veilid.pairing")
_records_mod = importlib.import_module("opti_oignon.veilid.records")
_engine_mod = importlib.import_module("opti_oignon.veilid.sync_engine")
_status_mod = importlib.import_module("opti_oignon.veilid.sync_status")
rs = importlib.import_module("opti_oignon.api.routes_sync")

ChangeFeed = _change_feed_mod.ChangeFeed
PeerStore = _peers_mod.PeerStore
accept_pairing_payload = _pairing_mod.accept_pairing_payload
build_pairing_payload = _pairing_mod.build_pairing_payload
confirmation_code = _pairing_mod.confirmation_code
pairing_canonical_material = _pairing_mod.pairing_canonical_material
pairing_integrity = _pairing_mod.pairing_integrity
parse_pairing_payload = _pairing_mod.parse_pairing_payload
RecordKind = _records_mod.RecordKind
new_record = _records_mod.new_record
attach_signature = signing_mod.attach_signature
decode_public_key = signing_mod.decode_public_key
encode_public_key = signing_mod.encode_public_key
SigningUnavailable = signing_mod.SigningUnavailable
SyncEngine = _engine_mod.SyncEngine
PeerNotConfirmed = _engine_mod.PeerNotConfirmed
PeerNotFound = _engine_mod.PeerNotFound
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
    _AUDIT["events"].clear()
    _change_feed_mod.reset_change_feed()
    _peers_mod.reset_peer_store()
    _engine_mod.reset_sync_engine()
    _status_mod.reset_sync_status_store()
    signing_mod.reset_record_signer()
    rs.reset_peer_resolver()
    rs.reset_self_routing_resolver()
    yield
    _change_feed_mod.reset_change_feed()
    _peers_mod.reset_peer_store()
    _engine_mod.reset_sync_engine()
    _status_mod.reset_sync_status_store()
    signing_mod.reset_record_signer()
    rs.reset_peer_resolver()
    rs.reset_self_routing_resolver()
    set_mode("daily")


# ---------------------------------------------------------------------------
# The deterministic fake signer (the S205 injectable seam)
# ---------------------------------------------------------------------------


class FakeSigner:
    """A deterministic HMAC-SHA256 'signature' scheme keyed per device."""

    def __init__(self, secret: bytes) -> None:
        self._secret = secret

    def public_key(self) -> bytes:
        return hmac_mod.new(self._secret, b"pub", hashlib.sha256).digest()

    def sign(self, data: bytes) -> bytes:
        return hmac_mod.new(
            self._secret + self.public_key(), data, hashlib.sha256
        ).digest()

    def verify(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        expected_like = hmac_mod.new(
            self._mac_key_for(public_key), data, hashlib.sha256
        ).digest()
        return hmac_mod.compare_digest(expected_like, signature)

    def _mac_key_for(self, public_key: bytes) -> bytes:
        secret = _PUB_REGISTRY.get(public_key)
        return (secret or b"\x00") + public_key


_PUB_REGISTRY: dict[bytes, bytes] = {}


def make_signer(seed: str) -> FakeSigner:
    secret = hashlib.sha256(seed.encode()).digest()
    s = FakeSigner(secret)
    _PUB_REGISTRY[s.public_key()] = secret
    return s


class UnavailableSigner:
    """A signer whose backend is absent: sign raises, verify_available False."""

    def public_key(self) -> bytes:
        raise SigningUnavailable("no backend")

    def sign(self, data: bytes) -> bytes:
        raise SigningUnavailable("no backend")

    def verify(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        return False

    def verify_available(self) -> bool:
        return False


def b64(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii")


def rec(device: str, *, clock: int = 1, rid: str = "c1", payload=None, deleted=False):
    return new_record(
        RecordKind.CONVERSATION,
        rid,
        payload if payload is not None else {"title": "hello"},
        device=device,
        clock=clock,
        deleted=deleted,
        updated_at="2026-01-01T00:00:00+00:00",
    )


class FakeServingPeer:
    """A peer answering from its own feed through the real protocol."""

    def __init__(self, feed, device: str) -> None:
        self._feed = feed
        self._device = device

    def fetch(self, request):
        from opti_oignon.veilid.protocol import respond_to_request

        return respond_to_request(self._feed, request, device=self._device)


def mat(peer_id: str, routing_key: str, signing_pub=None) -> str:
    return pairing_canonical_material(peer_id, routing_key, signing_pub)


# ---------------------------------------------------------------------------
# Decision 1 -- the confirmation-code derivation
# ---------------------------------------------------------------------------


class TestConfirmationCodeDerivation:
    def test_order_normalized_identical_on_both_devices(self):
        a = mat("dev-a", "rk-a", "PUBA")
        b = mat("dev-b", "rk-b", "PUBB")
        assert confirmation_code(a, b) == confirmation_code(b, a)

    def test_deterministic_and_human_comparable_format(self):
        a = mat("dev-a", "rk-a", "PUBA")
        b = mat("dev-b", "rk-b", "PUBB")
        code = confirmation_code(a, b)
        assert re.fullmatch(r"\d{4} \d{4}", code)
        assert confirmation_code(a, b) == code  # deterministic

    def test_sensitive_to_every_public_field_of_the_peer(self):
        a = mat("dev-a", "rk-a", "PUBA")
        base = confirmation_code(a, mat("dev-b", "rk-b", "PUBB"))
        assert confirmation_code(a, mat("dev-X", "rk-b", "PUBB")) != base
        assert confirmation_code(a, mat("dev-b", "rk-X", "PUBB")) != base
        assert confirmation_code(a, mat("dev-b", "rk-b", "PUBX")) != base
        # A stripped signing key changes the code too: the trust root is
        # inside the derivation, the whole point of PAIR-02.
        assert confirmation_code(a, mat("dev-b", "rk-b")) != base

    def test_sensitive_to_this_devices_own_material(self):
        b = mat("dev-b", "rk-b", "PUBB")
        base = confirmation_code(mat("dev-a", "rk-a", "PUBA"), b)
        assert confirmation_code(mat("dev-a", "rk-OTHER", "PUBA"), b) != base

    def test_validates_inputs(self):
        good = mat("dev-a", "rk-a")
        with pytest.raises(ValueError):
            confirmation_code("", good)
        with pytest.raises(ValueError):
            confirmation_code(good, "")
        with pytest.raises(ValueError):
            confirmation_code(good, None)  # type: ignore[arg-type]

    def test_material_is_the_integrity_serialisation(self):
        # ONE canonical serialisation: the integrity is its SHA-256, so the
        # code and the integrity cover exactly the same public fields.
        m = mat("dev-a", "rk-a", "PUBA")
        assert (
            hashlib.sha256(m.encode("utf-8")).hexdigest()
            == pairing_integrity("dev-a", "rk-a", "PUBA")
        )

    def test_material_validates_like_the_builder(self):
        with pytest.raises(ValueError):
            mat("", "rk")
        with pytest.raises(ValueError):
            mat("dev", "")
        with pytest.raises(ValueError):
            mat("dev", "rk", "")

    def test_recipe_constants_are_the_documented_construction(self):
        # The derivation is ONE documented construction (Kerckhoffs: open
        # parameters); pin them so a silent parameter drift cannot weaken the
        # stated grind bound.
        assert _pairing_mod.CONFIRM_CODE_SALT == b"oo-pairing-confirm-v1"
        assert _pairing_mod.CONFIRM_SCRYPT_N == 2**14
        assert _pairing_mod.CONFIRM_SCRYPT_R == 8
        assert _pairing_mod.CONFIRM_SCRYPT_P == 1
        assert _pairing_mod.CONFIRM_CODE_DIGITS == 8

    def test_exact_construction(self):
        # The full recipe, recomputed independently: sorted materials,
        # byte-length-prefix framing, scrypt, first 8 bytes mod 10**8.
        a = mat("dev-a", "rk-a", "PUBA")
        b = mat("dev-b", "rk-b", "PUBB")
        parts = sorted((a, b))
        framed = "".join(
            "{}:{}".format(len(p.encode("utf-8")), p) for p in parts
        ).encode("utf-8")
        digest = hashlib.scrypt(
            framed,
            salt=b"oo-pairing-confirm-v1",
            n=2**14,
            r=8,
            p=1,
            maxmem=64 * 1024 * 1024,
            dklen=32,
        )
        value = int.from_bytes(digest[:8], "big") % (10**8)
        expected = "{:04d} {:04d}".format(value // 10000, value % 10000)
        assert confirmation_code(a, b) == expected


# ---------------------------------------------------------------------------
# Decisions 2 and 3 -- the pending state machine at the store and engine seams
# ---------------------------------------------------------------------------


def _engine(tmp_path, *, device="dev-b", seed="asker-b"):
    feed = ChangeFeed(root=tmp_path / device)
    store = PeerStore(root=tmp_path / device)
    eng = SyncEngine(
        device=device, feed=feed, store=store, signer=make_signer(seed)
    )
    return eng, store, feed


class TestPendingStateMachine:
    def test_ceremony_registers_pending(self, tmp_path):
        eng, store, _ = _engine(tmp_path)
        payload = build_pairing_payload("dev-a", "rk-a", b64(b"PUBA"))
        out = accept_pairing_payload(eng, payload, label="laptop")
        assert out is not None and out.pending and not out.key_changed
        stored = store.get_peer("dev-a")
        assert stored.pending and stored.signing_pub == b64(b"PUBA")

    def test_programmatic_registration_defaults_confirmed(self, tmp_path):
        eng, store, _ = _engine(tmp_path)
        eng.register_peer("dev-a", "rk-a", signing_pub="K")
        assert not store.get_peer("dev-a").pending

    def test_run_round_refuses_pending(self, tmp_path):
        eng, store, _ = _engine(tmp_path)
        store.add_peer("dev-a", "rk-a", pending=True)
        with pytest.raises(PeerNotConfirmed):
            eng.run_round("dev-a", object())

    def test_run_round_still_404s_unknown_first(self, tmp_path):
        eng, _, _ = _engine(tmp_path)
        with pytest.raises(PeerNotFound):
            eng.run_round("ghost", object())

    def test_serve_refuses_pending_when_identity_supplied(self, tmp_path):
        eng, store, _ = _engine(tmp_path)
        store.add_peer("dev-a", "rk-a", pending=True)
        with pytest.raises(PeerNotConfirmed):
            eng.serve_request({"v": 1}, peer_id="dev-a")

    def test_serve_without_identity_is_the_documented_posture(self, tmp_path):
        # Stated honestly: with no asker identity the gate has nothing to
        # check (the private route is the implicit authenticator); the serve
        # itself stays the defensive PRT-01 responder.
        eng, store, _ = _engine(tmp_path)
        store.add_peer("dev-a", "rk-a", pending=True)
        batch = eng.serve_request({"v": 1})
        assert isinstance(batch, dict)

    def test_serve_allows_confirmed_identity(self, tmp_path):
        eng, store, _ = _engine(tmp_path)
        store.add_peer("dev-a", "rk-a", pending=True)
        store.confirm_peer("dev-a")
        batch = eng.serve_request({"v": 1}, peer_id="dev-a")
        assert isinstance(batch, dict)

    def test_confirm_activates_the_round(self, tmp_path):
        eng, store, _ = _engine(tmp_path)
        feed_a = ChangeFeed(root=tmp_path / "a")
        signer_a = make_signer("origin-a")
        engine_a = SyncEngine(device="dev-a", feed=feed_a, signer=signer_a)
        engine_a.publish(rec("dev-a", clock=1))
        store.add_peer(
            "dev-a", "rk-a", signing_pub=b64(signer_a.public_key()), pending=True
        )
        with pytest.raises(PeerNotConfirmed):
            eng.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert eng.confirm_peer("dev-a") is True
        result = eng.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.applied == 1 and result.refused == 0

    def test_store_confirm_unknown_false_and_idempotent(self, tmp_path):
        _, store, _ = _engine(tmp_path)
        assert store.confirm_peer("ghost") is False
        store.add_peer("dev-a", "rk", pending=True)
        assert store.confirm_peer("dev-a") is True
        assert store.confirm_peer("dev-a") is True  # idempotent
        assert not store.get_peer("dev-a").pending

    def test_upsert_never_lowers_and_caller_flag_ignored(self, tmp_path):
        _, store, _ = _engine(tmp_path)
        # Pending stays pending across a re-pair, whatever the caller passes.
        store.add_peer("dev-a", "rk1", pending=True)
        assert store.add_peer("dev-a", "rk2", pending=False).pending
        # Confirmed stays confirmed across a ceremony re-pair with the same
        # key: the trust root did not change, no re-confirmation theater.
        store.add_peer("dev-c", "rk1", signing_pub="K")
        store.confirm_peer("dev-a")
        assert not store.add_peer("dev-a", "rk3", pending=True).pending
        assert not store.add_peer(
            "dev-c", "rk2", signing_pub="K", pending=True
        ).pending

    def test_grandfather_pre_pair02_rows_confirmed(self, tmp_path):
        # A registry written by the S205 schema (no pending column, no
        # self_pairing_material meta column): the migration adds the columns
        # and every pre-existing row reads CONFIRMED -- never a retroactive
        # lockout -- while the new surfaces work after the migration.
        db = tmp_path / _peers_mod.DB_FILENAME
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE veilid_peers ("
            "peer_id TEXT PRIMARY KEY, routing_key TEXT NOT NULL, "
            "label TEXT NOT NULL DEFAULT '', "
            "watermark INTEGER NOT NULL DEFAULT 0, "
            "added_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "last_epoch TEXT, signing_pub TEXT)"
        )
        conn.execute(
            "CREATE TABLE veilid_local_identity ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), "
            "device_id TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO veilid_peers VALUES "
            "('old-peer', 'rk', 'L', 5, 't0', 't0', NULL, 'K')"
        )
        conn.execute(
            "INSERT INTO veilid_local_identity VALUES (1, 'dev-old', 't0')"
        )
        conn.commit()
        conn.close()
        store = PeerStore(root=tmp_path)
        old = store.get_peer("old-peer")
        assert old is not None
        assert not old.pending and not old.key_changed
        assert old.watermark == 5 and old.signing_pub == "K"
        # The meta migration landed too: pinning works on the old registry.
        assert store.get_self_pairing_material() is None
        store.pin_self_pairing_material("MAT")
        assert store.get_self_pairing_material() == "MAT"
        assert store.local_device_id() == "dev-old"
        store.close()

    def test_verify_refuses_pending_origin_never_grace(self, tmp_path):
        # The relay case: B pulls from CONFIRMED peer C, whose feed carries a
        # record ORIGINATED by dev-a -- correctly signed by dev-a's key, which
        # B holds for a PENDING dev-a entry. The record must be REFUSED
        # (counted), not verified-trusted and not grace-admitted as
        # unverified, even while the migration grace is open; and the refusal
        # never pins the watermark.
        assert signing_mod.ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS is True
        signer_a = make_signer("origin-a")
        signer_c = make_signer("relay-c")
        feed_c = ChangeFeed(root=tmp_path / "c")
        engine_c = SyncEngine(device="dev-c", feed=feed_c, signer=signer_c)
        signed_a = attach_signature(rec("dev-a", clock=1), signer_a)
        engine_c.publish(signed_a)  # foreign provenance, journalled verbatim
        engine_c.publish(rec("dev-c", clock=1, rid="c2"))

        eng, store, _ = _engine(tmp_path)
        store.add_peer("dev-c", "rk-c", signing_pub=b64(signer_c.public_key()))
        store.add_peer(
            "dev-a", "rk-a", signing_pub=b64(signer_a.public_key()), pending=True
        )
        result = eng.run_round("dev-c", FakeServingPeer(feed_c, "dev-c"))
        assert result.refused == 1  # the pending-origin record
        assert result.unverified == 0  # never the grace path
        assert result.applied == 1  # dev-c's own record applies
        assert result.advanced is True  # a refusal never pins convergence

    def test_pending_origin_verifies_after_confirmation(self, tmp_path):
        signer_a = make_signer("origin-a")
        signer_c = make_signer("relay-c")
        feed_c = ChangeFeed(root=tmp_path / "c")
        engine_c = SyncEngine(device="dev-c", feed=feed_c, signer=signer_c)
        engine_c.publish(attach_signature(rec("dev-a", clock=1), signer_a))
        eng, store, _ = _engine(tmp_path)
        store.add_peer("dev-c", "rk-c", signing_pub=b64(signer_c.public_key()))
        store.add_peer(
            "dev-a", "rk-a", signing_pub=b64(signer_a.public_key()), pending=True
        )
        store.confirm_peer("dev-a")
        result = eng.run_round("dev-c", FakeServingPeer(feed_c, "dev-c"))
        assert result.applied == 1 and result.refused == 0


# ---------------------------------------------------------------------------
# Decision 4 -- re-pair semantics: the demotion
# ---------------------------------------------------------------------------


class TestRepairDemotion:
    def test_same_key_repair_stays_confirmed(self, tmp_path):
        _, store, _ = _engine(tmp_path)
        store.add_peer("p", "rk1", signing_pub="K1")
        out = store.add_peer("p", "rk2", signing_pub="K1")
        assert not out.pending and not out.key_changed
        assert out.routing_key == "rk2"

    def test_key_change_demotes_with_key_changed(self, tmp_path):
        _, store, _ = _engine(tmp_path)
        store.add_peer("p", "rk1", signing_pub="K1")
        store.advance_watermark("p", 9)
        out = store.add_peer("p", "rk2", signing_pub="K2")
        assert out.pending and out.key_changed
        assert out.signing_pub == "K2"
        assert out.watermark == 9  # the demotion never resets sync progress

    def test_first_key_over_unkeyed_row_demotes(self, tmp_path):
        # A NEW trust root over a previously unkeyed row is a new trust
        # decision: the post-upgrade fleet re-pair includes exactly one
        # confirmation per peer, which is the point of PAIR-02 before the
        # Bloc 4 grace flip.
        _, store, _ = _engine(tmp_path)
        store.add_peer("p", "rk1")
        out = store.add_peer("p", "rk1", signing_pub="K1")
        assert out.pending and out.key_changed

    def test_keyless_repair_preserves_key_and_never_demotes(self, tmp_path):
        _, store, _ = _engine(tmp_path)
        store.add_peer("p", "rk1", signing_pub="K1")
        out = store.add_peer("p", "rk2")
        assert not out.pending and out.signing_pub == "K1"

    def test_confirm_clears_the_demotion(self, tmp_path):
        _, store, _ = _engine(tmp_path)
        store.add_peer("p", "rk1", signing_pub="K1")
        store.add_peer("p", "rk1", signing_pub="K2")
        assert store.get_peer("p").key_changed
        store.confirm_peer("p")
        refreshed = store.get_peer("p")
        assert not refreshed.pending and not refreshed.key_changed

    def test_demoted_peer_refuses_the_wire_until_reconfirmed(self, tmp_path):
        eng, store, _ = _engine(tmp_path)
        signer_a = make_signer("origin-a")
        store.add_peer("dev-a", "rk", signing_pub=b64(signer_a.public_key()))
        # A re-pair with a DIFFERENT key demotes; the wire refuses again.
        store.add_peer("dev-a", "rk", signing_pub=b64(b"ATTACKER"))
        with pytest.raises(PeerNotConfirmed):
            eng.run_round("dev-a", object())


# ---------------------------------------------------------------------------
# The self key custody surface (engine + signing helpers)
# ---------------------------------------------------------------------------


class TestSelfSigningPub:
    def test_fake_signer_yields_base64(self, tmp_path):
        signer = make_signer("me")
        eng = SyncEngine(
            device="me",
            feed=ChangeFeed(root=tmp_path),
            store=PeerStore(root=tmp_path),
            signer=signer,
        )
        out = eng.self_signing_pub()
        assert out == b64(signer.public_key())
        assert decode_public_key(out) == signer.public_key()

    def test_unavailable_backend_degrades_to_none(self, tmp_path):
        eng = SyncEngine(
            device="me",
            feed=ChangeFeed(root=tmp_path),
            store=PeerStore(root=tmp_path),
            signer=UnavailableSigner(),
        )
        assert eng.self_signing_pub() is None

    def test_encode_public_key_validates_and_round_trips(self):
        raw = b"\x01\x02\xffkey"
        assert decode_public_key(encode_public_key(raw)) == raw
        with pytest.raises(ValueError):
            encode_public_key(b"")
        with pytest.raises(ValueError):
            encode_public_key("not-bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The production self-payload gap, closed (routes web-free helpers)
# ---------------------------------------------------------------------------


class TestSelfPayloadKeyThreading:
    def test_self_pairing_payload_threads_the_key(self):
        out = rs.self_pairing_payload("me", "rk-me", "PUBME")
        payload = out["payload"]
        assert payload["signing_pub"] == "PUBME"
        assert payload["integrity"] == pairing_integrity("me", "rk-me", "PUBME")
        parsed = parse_pairing_payload(payload)
        assert parsed is not None and parsed.signing_pub == "PUBME"

    def test_self_pairing_payload_without_key_keeps_pre_vl01_shape(self):
        out = rs.self_pairing_payload("me", "rk-me")
        assert "signing_pub" not in out["payload"]
        assert out["payload"]["integrity"] == pairing_integrity("me", "rk-me")


# ---------------------------------------------------------------------------
# The pending surfaces (routes web-free helpers)
# ---------------------------------------------------------------------------


class TestPendingSurfaces:
    def test_accept_returns_pending_and_the_symmetric_code(self, tmp_path):
        eng, store, _ = _engine(tmp_path, device="me", seed="me")
        store.pin_self_pairing_material(mat("me", "rk-me", "PUBME"))
        payload = build_pairing_payload("dev-b", "rk-b", "PUBB")
        out = rs.accept_pairing(eng, payload, label="phone", store=store)
        assert out["pending"] is True and out["key_changed"] is False
        code = out["confirmation_code"]
        assert re.fullmatch(r"\d{4} \d{4}", code)
        # The OTHER device derives the identical code from its own halves.
        other_side = confirmation_code(
            mat("dev-b", "rk-b", "PUBB"), mat("me", "rk-me", "PUBME")
        )
        assert code == other_side

    def test_accept_code_null_without_self_material(self, tmp_path):
        eng, store, _ = _engine(tmp_path, device="me", seed="me")
        payload = build_pairing_payload("dev-b", "rk-b", "PUBB")
        out = rs.accept_pairing(eng, payload, store=store)
        assert out["pending"] is True
        assert out["confirmation_code"] is None
        listed = rs.pending_pairings_payload(store)
        assert listed["self_ready"] is False
        assert listed["pending"][0]["confirmation_code"] is None

    def test_pending_payload_lists_codes_and_skips_confirmed(self, tmp_path):
        eng, store, _ = _engine(tmp_path, device="me", seed="me")
        store.pin_self_pairing_material(mat("me", "rk-me"))
        store.add_peer("confirmed-peer", "rk")
        store.add_peer("pend-1", "rk1", signing_pub="P1", pending=True)
        store.add_peer("pend-2", "rk2", pending=True)
        listed = rs.pending_pairings_payload(store)
        assert listed["self_ready"] is True
        codes = {p["peer_id"]: p["confirmation_code"] for p in listed["pending"]}
        assert set(codes) == {"pend-1", "pend-2"}
        for code in codes.values():
            assert re.fullmatch(r"\d{4} \d{4}", code)
        # The codes recompute deterministically from the stored columns,
        # the signing key included when the peer registered one.
        assert codes["pend-1"] == confirmation_code(
            mat("me", "rk-me"), mat("pend-1", "rk1", "P1")
        )
        assert codes["pend-2"] == confirmation_code(
            mat("me", "rk-me"), mat("pend-2", "rk2")
        )

    def test_peer_dict_defensive_for_pre_pair02_records(self):
        class OldRecord:
            peer_id = "p"
            routing_key = "rk"
            label = ""
            watermark = 0
            added_at = ""
            updated_at = ""

        d = rs._peer_to_dict(OldRecord())
        assert d["pending"] is False and d["key_changed"] is False

    def test_key_changed_surfaces_on_the_wire_dict(self, tmp_path):
        _, store, _ = _engine(tmp_path)
        store.add_peer("p", "rk", signing_pub="K1")
        store.add_peer("p", "rk", signing_pub="K2")
        d = rs._peer_to_dict(store.get_peer("p"))
        assert d["pending"] is True and d["key_changed"] is True


# ---------------------------------------------------------------------------
# The live FastAPI surface (skips where fastapi is absent; s182 idiom)
# ---------------------------------------------------------------------------


class TestLiveConfirmationRoutes:
    def setup_method(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    def _client(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        feed = ChangeFeed(root=tmp_path / "local")
        store = PeerStore(root=tmp_path / "store")
        eng = SyncEngine(
            device="A", feed=feed, store=store, signer=make_signer("router-a")
        )
        _peers_mod.set_peer_store(store)
        _engine_mod.set_sync_engine(eng)
        app = FastAPI()
        rs.register(app)
        return TestClient(app), store, eng

    def test_full_ceremony_over_the_routes(self, tmp_path):
        client, store, eng = self._client(tmp_path)
        rs.set_self_routing_resolver(lambda e: "route-A")
        # Generate: the payload carries the signing key and pins the material.
        r = client.get("/api/sync/pairing/self")
        assert r.status_code == 200
        body = r.json()
        own_key = eng.self_signing_pub()
        assert body["payload"]["signing_pub"] == own_key
        assert store.get_self_pairing_material() == mat("A", "route-A", own_key)
        # Accept: pending plus the code.
        peer_payload = build_pairing_payload("B", "route-B", b64(b"PUBB"))
        r = client.post(
            "/api/sync/pairing/accept",
            json={"payload": peer_payload, "label": "laptop"},
        )
        assert r.status_code == 200
        accepted = r.json()
        assert accepted["pending"] is True
        code = accepted["confirmation_code"]
        assert re.fullmatch(r"\d{4} \d{4}", code)
        # The pending list shows the same code.
        r = client.get("/api/sync/pairing/pending")
        assert r.status_code == 200
        listed = r.json()
        assert listed["self_ready"] is True
        assert listed["pending"][0]["confirmation_code"] == code
        # A round refuses with the actionable 409.
        rs.set_peer_resolver(lambda peer_id, store: object())
        r = client.post("/api/sync/peers/B/run")
        assert r.status_code == 409
        assert "confirm" in r.json()["detail"].lower()
        # Confirm activates; the pending list empties.
        r = client.post("/api/sync/pairing/pending/B/confirm")
        assert r.status_code == 200 and r.json()["pending"] is False
        assert client.get("/api/sync/pairing/pending").json()["pending"] == []
        actions = [a.get("action") for a in _AUDIT["events"]]
        assert "pairing_confirm" in actions

    def test_confirm_404_unknown_and_idempotent_200(self, tmp_path):
        client, store, _ = self._client(tmp_path)
        assert client.post("/api/sync/pairing/pending/ghost/confirm").status_code == 404
        store.add_peer("B", "rk", pending=True)
        assert client.post("/api/sync/pairing/pending/B/confirm").status_code == 200
        r = client.post("/api/sync/pairing/pending/B/confirm")
        assert r.status_code == 200 and r.json()["pending"] is False

    def test_reject_removes_only_pending(self, tmp_path):
        client, store, _ = self._client(tmp_path)
        assert client.post("/api/sync/pairing/pending/ghost/reject").status_code == 404
        store.add_peer("B", "rk", pending=True)
        r = client.post("/api/sync/pairing/pending/B/reject")
        assert r.status_code == 200 and r.json()["rejected"] is True
        assert store.get_peer("B") is None
        actions = [a.get("action") for a in _AUDIT["events"]]
        assert "pairing_reject" in actions

    def test_reject_409_on_confirmed_peer(self, tmp_path):
        client, store, _ = self._client(tmp_path)
        store.add_peer("B", "rk")  # confirmed
        r = client.post("/api/sync/pairing/pending/B/reject")
        assert r.status_code == 409
        assert store.get_peer("B") is not None  # never removed

    def test_self_unavailable_signer_still_pairs_pre_vl01(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        feed = ChangeFeed(root=tmp_path / "local")
        store = PeerStore(root=tmp_path / "store")
        eng = SyncEngine(
            device="A", feed=feed, store=store, signer=UnavailableSigner()
        )
        _peers_mod.set_peer_store(store)
        _engine_mod.set_sync_engine(eng)
        app = FastAPI()
        rs.register(app)
        client = TestClient(app)
        rs.set_self_routing_resolver(lambda e: "route-A")
        r = client.get("/api/sync/pairing/self")
        assert r.status_code == 200
        assert "signing_pub" not in r.json()["payload"]
        # The material still pins, so the code exists for the keyless half.
        assert store.get_self_pairing_material() == mat("A", "route-A")


# ---------------------------------------------------------------------------
# Audit events
# ---------------------------------------------------------------------------


class TestAuditEvents:
    def test_peer_add_carries_the_pending_flag(self, tmp_path):
        eng, _, _ = _engine(tmp_path)
        eng.register_peer("p1", "rk")
        eng.register_peer("p2", "rk", pending=True)
        adds = {
            e["peer_id"]: e
            for e in _AUDIT["events"]
            if e.get("action") == "peer_add"
        }
        assert adds["p1"]["pending"] is False
        assert adds["p2"]["pending"] is True

    def test_pairing_confirm_audited_only_on_activation(self, tmp_path):
        eng, store, _ = _engine(tmp_path)
        store.add_peer("p", "rk", pending=True)
        eng.confirm_peer("p")
        eng.confirm_peer("p")  # idempotent: no second event
        confirms = [
            e for e in _AUDIT["events"] if e.get("action") == "pairing_confirm"
        ]
        assert len(confirms) == 1 and confirms[0]["peer_id"] == "p"

    def test_ceremony_accept_audits_pending_registration(self, tmp_path):
        eng, _, _ = _engine(tmp_path)
        accept_pairing_payload(eng, build_pairing_payload("B", "rk"))
        adds = [e for e in _AUDIT["events"] if e.get("action") == "peer_add"]
        assert adds and adds[-1]["pending"] is True


# ---------------------------------------------------------------------------
# Mode posture: pending management is local-disk, mode-free; the wire is not
# ---------------------------------------------------------------------------


class TestModePosture:
    def test_pending_management_mode_free_under_bulbe(self, tmp_path):
        set_mode("bulbe")
        eng, store, _ = _engine(tmp_path, device="me", seed="me")
        store.pin_self_pairing_material(mat("me", "rk-me"))
        out = rs.accept_pairing(
            eng, build_pairing_payload("B", "rk-b"), store=store
        )
        assert out["pending"] is True
        assert re.fullmatch(r"\d{4} \d{4}", out["confirmation_code"])
        listed = rs.pending_pairings_payload(store)
        assert listed["pending"][0]["peer_id"] == "B"
        assert eng.confirm_peer("B") is True
        assert not store.get_peer("B").pending
        # The wire refuses via the REAL gate, before any pending check.
        store.add_peer("C", "rk", pending=True)
        with pytest.raises(VeilidDisabledInBulbe):
            eng.run_round("C", object())
        with pytest.raises(VeilidDisabledInBulbe):
            eng.serve_request({"v": 1}, peer_id="C")

    def test_indeterminable_mode_still_permits_confirmation(self, tmp_path):
        set_mode(raises=True)  # fail-secure to bulbe for the wire only
        _, store, _ = _engine(tmp_path)
        store.add_peer("B", "rk", pending=True)
        assert store.confirm_peer("B") is True
