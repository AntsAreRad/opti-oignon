#!/usr/bin/env python3
"""S205 per-fix tests -- sync cycle Bloc 2 lot 1 (VL-01 per-record signing).

The lot: a per-device ML-DSA-65 signing keypair with encrypted-at-rest custody
(veilid/signing.py); a `signature` wire field over canonical record bytes that
bind clock and device (records.canonical_record_bytes); sign-at-publish into an
additive journal column; verification at the engine's apply seam against the
ORIGIN device's registered key, refusals counted and surfaced
(RoundResult/sync_status/API payload); the pairing payload extended with the
signing public key under the present-fields integrity; the peer registry's
additive signing_pub column with refresh-with-route / preserve-on-absence
re-pair semantics; the bounded migration grace for unkeyed origins; and the
mode-free posture of key custody and local signing.

liboqs is absent in the container, so every signing test runs through the
injectable signer seam with a deterministic HMAC-backed fake; the real
ML-DSA-65 path is host-verified by the shakedown's crypto item.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac as hmac_mod
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

# The established sync-suite isolation harness (S180/S204 idiom): stub the
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
_protocol_mod = importlib.import_module("opti_oignon.veilid.protocol")
_records_mod = importlib.import_module("opti_oignon.veilid.records")
_engine_mod = importlib.import_module("opti_oignon.veilid.sync_engine")
_status_mod = importlib.import_module("opti_oignon.veilid.sync_status")

ChangeFeed = _change_feed_mod.ChangeFeed
PeerStore = _peers_mod.PeerStore
accept_pairing_payload = _pairing_mod.accept_pairing_payload
build_pairing_payload = _pairing_mod.build_pairing_payload
pairing_integrity = _pairing_mod.pairing_integrity
parse_pairing_payload = _pairing_mod.parse_pairing_payload
SENDER_MAX_BYTES = _protocol_mod.SENDER_MAX_BYTES
SENDER_MAX_RECORDS = _protocol_mod.SENDER_MAX_RECORDS
build_record_batch = _protocol_mod.build_record_batch
parse_record_batch = _protocol_mod.parse_record_batch
RecordKind = _records_mod.RecordKind
canonical_record_bytes = _records_mod.canonical_record_bytes
decode_record = _records_mod.decode_record
encode_record = _records_mod.encode_record
new_record = _records_mod.new_record
verify_record_hash = _records_mod.verify_record_hash
PqcRecordSigner = signing_mod.PqcRecordSigner
SigningUnavailable = signing_mod.SigningUnavailable
attach_signature = signing_mod.attach_signature
decode_public_key = signing_mod.decode_public_key
verify_record_signature = signing_mod.verify_record_signature
SyncEngine = _engine_mod.SyncEngine
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
    yield
    _change_feed_mod.reset_change_feed()
    _peers_mod.reset_peer_store()
    _engine_mod.reset_sync_engine()
    _status_mod.reset_sync_status_store()
    signing_mod.reset_record_signer()
    set_mode("daily")


# ---------------------------------------------------------------------------
# The deterministic fake signer (the injectable seam under test)
# ---------------------------------------------------------------------------


class FakeSigner:
    """A deterministic HMAC-SHA256 'signature' scheme keyed per device.

    sign(data) = HMAC(secret, data); verify recomputes against the public key,
    which is defined as HMAC(secret, b"pub") -- so only the matching secret's
    public key verifies, giving real wrong-key semantics without liboqs.
    """

    def __init__(self, secret: bytes) -> None:
        self._secret = secret

    def public_key(self) -> bytes:
        return hmac_mod.new(self._secret, b"pub", hashlib.sha256).digest()

    def sign(self, data: bytes) -> bytes:
        return hmac_mod.new(
            self._secret + self.public_key(), data, hashlib.sha256
        ).digest()

    def verify(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        # Recompute under every secret is impossible for a fake; emulate
        # asymmetry by binding the public key into the MAC input. A signature
        # made with secret S only verifies when public_key == pub(S).
        expected_like = hmac_mod.new(
            self._mac_key_for(public_key), data, hashlib.sha256
        ).digest()
        return hmac_mod.compare_digest(expected_like, signature)

    def _mac_key_for(self, public_key: bytes) -> bytes:
        # The verifier must be able to check signatures from ANY device given
        # its public key. The fake keeps a class-level registry mapping
        # public keys back to secrets, mimicking asymmetric verification.
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


# ---------------------------------------------------------------------------
# Decision 2 -- the canonical byte recipe binds clock and device
# ---------------------------------------------------------------------------


class TestCanonicalRecipe:
    def test_recipe_is_sorted_compact_json_without_signature(self):
        r = rec("dev-a")
        signed = dataclasses.replace(r, signature="c2ln")
        wire = encode_record(signed)
        wire.pop("signature")
        expected = json.dumps(
            wire, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        assert canonical_record_bytes(signed) == expected

    def test_recipe_identical_signed_or_not(self):
        r = rec("dev-a")
        assert canonical_record_bytes(r) == canonical_record_bytes(
            dataclasses.replace(r, signature="c2ln")
        )

    def test_recipe_stable_and_deterministic(self):
        assert canonical_record_bytes(rec("dev-a")) == canonical_record_bytes(
            rec("dev-a")
        )

    @pytest.mark.parametrize(
        "mutation",
        [
            {"clock": 9},
            {"device": "dev-evil"},
            {"deleted": True},
            {"updated_at": "2030-01-01T00:00:00+00:00"},
            {"record_id": "c2"},
        ],
    )
    def test_recipe_sensitive_to_each_bound_field(self, mutation):
        r = rec("dev-a")
        mutated = dataclasses.replace(r, **mutation)
        assert canonical_record_bytes(mutated) != canonical_record_bytes(r)

    def test_recipe_sensitive_to_payload(self):
        a = rec("dev-a", payload={"title": "x"})
        b = rec("dev-a", payload={"title": "y"})
        assert canonical_record_bytes(a) != canonical_record_bytes(b)


class TestTamperBreaksSignature:
    @pytest.mark.parametrize(
        "mutation",
        [
            {"clock": 9},
            {"device": "dev-evil"},
            {"deleted": True},
            {"updated_at": "2030-01-01T00:00:00+00:00"},
        ],
    )
    def test_tampering_each_field_breaks_verification(self, mutation):
        signer = make_signer("a")
        signed = attach_signature(rec("dev-a"), signer)
        assert verify_record_signature(signed, signer.public_key(), signer)
        tampered = dataclasses.replace(signed, **mutation)
        assert not verify_record_signature(tampered, signer.public_key(), signer)

    def test_tampering_payload_breaks_verification(self):
        signer = make_signer("a")
        signed = attach_signature(rec("dev-a"), signer)
        tampered = dataclasses.replace(
            signed, payload={"title": "forged"}
        )
        assert not verify_record_signature(tampered, signer.public_key(), signer)

    def test_content_hash_keeps_its_role(self):
        # The hash still covers content only: a re-clocked record keeps a
        # valid hash (storage integrity unchanged) while its signature breaks
        # (authenticity layer above it).
        signer = make_signer("a")
        signed = attach_signature(rec("dev-a"), signer)
        reclocked = dataclasses.replace(signed, clock=99)
        assert verify_record_hash(reclocked)
        assert not verify_record_signature(reclocked, signer.public_key(), signer)


# ---------------------------------------------------------------------------
# Wire round-trip and both compat directions
# ---------------------------------------------------------------------------


class TestWireCompat:
    def test_signed_record_round_trips_and_verifies(self):
        signer = make_signer("a")
        signed = attach_signature(rec("dev-a"), signer)
        decoded = decode_record(encode_record(signed))
        assert decoded == signed
        assert verify_record_signature(decoded, signer.public_key(), signer)

    def test_unsigned_wire_shape_is_pre_vl01_shape(self):
        wire = encode_record(rec("dev-a"))
        assert "signature" not in wire

    def test_old_reader_ignores_the_signature_field(self):
        # The pre-S205 decoder read fields by name and ignored unknowns; the
        # current decoder consuming a signed wire object yields the record.
        signer = make_signer("a")
        wire = encode_record(attach_signature(rec("dev-a"), signer))
        assert "signature" in wire
        decoded = decode_record(wire)
        assert decoded is not None
        assert decoded.signature == wire["signature"]

    def test_mistyped_signature_degrades_to_unsigned_never_rejects(self):
        wire = encode_record(rec("dev-a"))
        wire["signature"] = 12345
        decoded = decode_record(wire)
        assert decoded is not None
        assert decoded.signature == ""

    def test_pre_vl01_sender_absent_field_parses_as_unsigned(self):
        decoded = decode_record(encode_record(rec("dev-a")))
        assert decoded is not None
        assert decoded.signature == ""


# ---------------------------------------------------------------------------
# Decision 1 / 6 -- custody seam and mint stability (the injectable signer)
# ---------------------------------------------------------------------------


class TestCustody:
    @staticmethod
    def _stub_crypto(monkeypatch):
        """Deterministic encryption/secure_bytes stubs owned by these tests.

        Earlier suites in a full sweep may leave bare opti_oignon.encryption
        stubs in sys.modules; the custody tests must not depend on collection
        order, so they install their own consistent pair (restored by
        monkeypatch) and exercise the custody DISCIPLINE (wrap, no plaintext
        at rest, unwrap, wipe); the real AES-256-GCM path is encryption.py's
        own suite and the shakedown's crypto item.
        """
        enc = types.ModuleType("opti_oignon.encryption")

        def encrypt_bytes(key: bytes, pt: bytes) -> bytes:
            return b"ENC1" + hashlib.sha256(key).digest()[:8] + pt

        def decrypt_bytes(key: bytes, data: bytes) -> bytes:
            tag = b"ENC1" + hashlib.sha256(key).digest()[:8]
            if not data.startswith(tag):
                raise ValueError("bad key or corrupt data")
            return data[len(tag):]

        enc.encrypt_bytes = encrypt_bytes  # type: ignore[attr-defined]
        enc.decrypt_bytes = decrypt_bytes  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "opti_oignon.encryption", enc)
        sb = types.ModuleType("opti_oignon.secure_bytes")

        class _SB:
            def __init__(self, d: bytes) -> None:
                self._d = bytes(d)

            def as_bytes(self) -> bytes:
                return self._d

            def wipe(self) -> None:
                self._d = b""

        sb.secure_key_from_bytes = lambda d: _SB(d)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "opti_oignon.secure_bytes", sb)

    def test_keypair_minted_once_and_stable(self, tmp_path, monkeypatch):
        # The PqcRecordSigner custody path without liboqs: stub the mint
        # primitives so the envelope/discipline is exercised; the real
        # ML-DSA-65 path is the shakedown's crypto item.
        import opti_oignon.pqc_signatures as pqc

        self._stub_crypto(monkeypatch)
        monkeypatch.setattr(pqc, "PQC_AVAILABLE", True)
        monkeypatch.setattr(
            pqc, "generate_pqc_keypair", lambda: (b"PUBKEY", b"PRIVKEY")
        )
        master = b"M" * 32
        monkeypatch.setattr(
            signing_mod,
            "_wrap_subkey",
            lambda: hashlib.sha256(master).digest(),
        )
        s = PqcRecordSigner(path=tmp_path / ".veilid_signing_key")
        first = s.public_key()
        second = s.public_key()
        assert first == second == b"PUBKEY"
        envelope = json.loads((tmp_path / ".veilid_signing_key").read_text())
        assert envelope["format"] == "veilid-signing-v1"
        assert "private_key_enc" in envelope
        # The private key is NOT plaintext-equivalent at rest.
        import base64

        stored = base64.urlsafe_b64decode(envelope["private_key_enc"])
        assert stored != b"PRIVKEY"
        assert "PRIVKEY" not in json.dumps(envelope)

    def test_private_key_encrypted_at_rest_and_round_trips(self, tmp_path, monkeypatch):
        import opti_oignon.pqc_signatures as pqc

        self._stub_crypto(monkeypatch)
        monkeypatch.setattr(pqc, "PQC_AVAILABLE", True)
        monkeypatch.setattr(
            pqc, "generate_pqc_keypair", lambda: (b"PUBKEY", b"PRIVKEY")
        )
        captured: dict = {}

        def fake_sign(data: bytes, private_key: bytes) -> bytes:
            captured["priv"] = private_key
            return b"SIG"

        monkeypatch.setattr(pqc, "sign_bytes", fake_sign)
        monkeypatch.setattr(
            signing_mod, "_wrap_subkey", lambda: hashlib.sha256(b"M" * 32).digest()
        )
        s = PqcRecordSigner(path=tmp_path / ".veilid_signing_key")
        assert s.sign(b"data") == b"SIG"
        # The unwrapped private key fed to the primitive is the minted one.
        assert captured["priv"] == b"PRIVKEY"

    def test_mint_refuses_without_master_key(self, tmp_path, monkeypatch):
        import opti_oignon.pqc_signatures as pqc

        monkeypatch.setattr(pqc, "PQC_AVAILABLE", True)
        monkeypatch.setattr(signing_mod, "_wrap_subkey", lambda: None)
        s = PqcRecordSigner(path=tmp_path / ".veilid_signing_key")
        with pytest.raises(SigningUnavailable):
            s.public_key()
        assert not (tmp_path / ".veilid_signing_key").exists()

    def test_mint_refuses_without_backend(self, tmp_path, monkeypatch):
        import opti_oignon.pqc_signatures as pqc

        monkeypatch.setattr(pqc, "PQC_AVAILABLE", False)
        monkeypatch.setattr(
            signing_mod, "_wrap_subkey", lambda: hashlib.sha256(b"M").digest()
        )
        s = PqcRecordSigner(path=tmp_path / ".veilid_signing_key")
        with pytest.raises(SigningUnavailable):
            s.public_key()
        assert not (tmp_path / ".veilid_signing_key").exists()

    def test_singleton_get_set_reset(self):
        try:
            fake = make_signer("singleton")
            signing_mod.set_record_signer(fake)
            assert signing_mod.get_record_signer() is fake
            signing_mod.reset_record_signer()
            fresh = signing_mod.get_record_signer()
            assert fresh is not fake
            assert isinstance(fresh, PqcRecordSigner)
        finally:
            signing_mod.reset_record_signer()

    def test_decode_public_key_defensive(self):
        assert decode_public_key(None) is None
        assert decode_public_key("") is None
        assert decode_public_key(123) is None
        assert decode_public_key("!!!not-base64!!!") is None
        assert decode_public_key(b64(b"key")) == b"key"


# ---------------------------------------------------------------------------
# Decision 3 -- sign-at-publish journals the signature; the wire stays bounded
# ---------------------------------------------------------------------------


class TestSignAtPublish:
    def test_publish_signs_own_records_and_journal_round_trips(self, tmp_path):
        signer = make_signer("a")
        feed = ChangeFeed(root=tmp_path)
        engine = SyncEngine(device="dev-a", feed=feed, signer=signer)
        engine.publish_conversation("c1", {"title": "hello"}, clock=1)
        stored = feed.current_records()
        assert len(stored) == 1
        assert stored[0].signature
        assert verify_record_signature(stored[0], signer.public_key(), signer)

    def test_sign_once_per_local_edit_not_at_serve(self, tmp_path):
        calls = []
        base = make_signer("a")

        class CountingSigner:
            def public_key(self):
                return base.public_key()

            def sign(self, data):
                calls.append(1)
                return base.sign(data)

            def verify(self, data, sig, pub):
                return base.verify(data, sig, pub)

        feed = ChangeFeed(root=tmp_path)
        engine = SyncEngine(device="dev-a", feed=feed, signer=CountingSigner())
        engine.publish_conversation("c1", {"title": "hello"}, clock=1)
        assert len(calls) == 1
        # Serving the record (twice) never re-signs.
        for _ in range(2):
            build_record_batch(feed, device="dev-a", watermark=0)
        assert len(calls) == 1

    def test_foreign_provenance_journalled_verbatim(self, tmp_path):
        # A record whose device is NOT this engine's is never signed here:
        # its signature (or absence) is the originator's, preserved verbatim.
        signer_a = make_signer("a")
        feed = ChangeFeed(root=tmp_path)
        engine = SyncEngine(device="dev-b", feed=feed, signer=make_signer("b"))
        foreign = attach_signature(rec("dev-a"), signer_a)
        engine.publish(foreign)
        stored = feed.current_records()[0]
        assert stored.signature == foreign.signature

    def test_publish_degrades_unsigned_when_signing_unavailable(self, tmp_path):
        feed = ChangeFeed(root=tmp_path)
        engine = SyncEngine(device="dev-a", feed=feed, signer=UnavailableSigner())
        seq = engine.publish_conversation("c1", {"title": "hello"}, clock=1)
        assert seq >= 1
        assert feed.current_records()[0].signature == ""

    def test_since_page_bounds_the_signed_wire_size(self, tmp_path):
        signer = make_signer("a")
        feed = ChangeFeed(root=tmp_path)
        engine = SyncEngine(device="dev-a", feed=feed, signer=signer)
        for i in range(6):
            engine.publish_conversation(f"c{i}", {"title": "x" * 50}, clock=1)
        one = feed.current_records()[0]
        one_size = len(
            json.dumps(
                encode_record(one), separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        # A budget of ~2.5 records: the page must stop within it (progress
        # guarantee keeps at least one), measured on the SIGNED encoding.
        budget = int(one_size * 2.5)
        page = feed.since_page(0, max_count=100, max_bytes=budget)
        assert 1 <= len(page.records) <= 2
        shipped = sum(
            len(
                json.dumps(
                    encode_record(r), separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
            )
            for r in page.records
        )
        assert shipped <= budget

    def test_signature_column_additive_migration(self, tmp_path):
        # A pre-S205 journal (no signature column) opens, migrates, and both
        # reads back its old rows (unsigned) and accepts signed appends.
        import sqlite3

        db = tmp_path / "veilid_change_feed.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE veilid_change_feed ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, "
            "record_id TEXT NOT NULL, clock INTEGER NOT NULL, "
            "device TEXT NOT NULL, content_hash TEXT NOT NULL, "
            "deleted INTEGER NOT NULL DEFAULT 0, "
            "updated_at TEXT NOT NULL DEFAULT '', "
            "payload TEXT NOT NULL DEFAULT '{}', journaled_at TEXT NOT NULL)"
        )
        old = rec("dev-old")
        conn.execute(
            "INSERT INTO veilid_change_feed (kind, record_id, clock, device, "
            "content_hash, deleted, updated_at, payload, journaled_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                old.kind.value,
                old.record_id,
                old.clock,
                old.device,
                old.content_hash,
                0,
                old.updated_at,
                json.dumps(dict(old.payload), separators=(",", ":")),
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
        conn.close()
        feed = ChangeFeed(root=tmp_path)
        got = feed.current_records()
        assert len(got) == 1 and got[0].signature == ""
        signer = make_signer("a")
        feed.record(attach_signature(rec("dev-a", rid="c2"), signer))
        by_id = {r.record_id: r for r in feed.current_records()}
        assert by_id["c1"].signature == ""
        assert by_id["c2"].signature


# ---------------------------------------------------------------------------
# Decision 4 -- the refusal taxonomy at the apply seam, counted, never applied
# ---------------------------------------------------------------------------


class FakeServingPeer:
    """A peer answering from its own feed through the real protocol."""

    def __init__(self, feed, device: str) -> None:
        self._feed = feed
        self._device = device

    def fetch(self, request):
        from opti_oignon.veilid.protocol import respond_to_request

        return respond_to_request(self._feed, request, device=self._device)


def make_pair(tmp_path, *, register_key=True):
    """Two engines: dev-a (origin/server) and dev-b (asker), b paired to a."""
    signer_a = make_signer("origin-a")
    signer_b = make_signer("asker-b")
    feed_a = ChangeFeed(root=tmp_path / "a")
    feed_b = ChangeFeed(root=tmp_path / "b")
    store_b = PeerStore(root=tmp_path / "b")
    engine_a = SyncEngine(device="dev-a", feed=feed_a, signer=signer_a)
    engine_b = SyncEngine(
        device="dev-b", feed=feed_b, store=store_b, signer=signer_b
    )
    engine_b.register_peer(
        "dev-a",
        "rk-a",
        label="A",
        signing_pub=b64(signer_a.public_key()) if register_key else None,
    )
    return engine_a, engine_b, feed_a, feed_b, store_b, signer_a, signer_b


class TestRefusalTaxonomy:
    def test_valid_signed_record_applies_through_a_full_round(self, tmp_path):
        engine_a, engine_b, feed_a, feed_b, *_ = make_pair(tmp_path)
        engine_a.publish_conversation("c1", {"title": "hello"}, clock=1)
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.applied == 1
        assert result.refused == 0
        assert result.unverified == 0
        assert {r.record_id for r in feed_b.current_records()} == {"c1"}
        # Provenance preserved end to end: the applied row keeps A's signature.
        stored = feed_b.current_records()[0]
        assert stored.device == "dev-a" and stored.signature

    def test_unsigned_from_keyed_origin_refused_and_counted(self, tmp_path):
        engine_a, engine_b, feed_a, feed_b, *_ = make_pair(tmp_path)
        # Bypass publish signing: journal an unsigned record as dev-a.
        feed_a.record(rec("dev-a"))
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.refused == 1
        assert result.applied == 0
        assert feed_b.current_records() == []
        # A refusal never pins the watermark: the round advanced past it.
        assert result.advanced

    def test_invalid_signature_refused(self, tmp_path):
        engine_a, engine_b, feed_a, feed_b, *_ = make_pair(tmp_path)
        signer_a = make_signer("origin-a")
        signed = attach_signature(rec("dev-a"), signer_a)
        corrupted = dataclasses.replace(signed, signature=b64(b"garbage-sig"))
        feed_a.record(corrupted)
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.refused == 1 and result.applied == 0
        assert feed_b.current_records() == []

    def test_wrong_key_signature_refused(self, tmp_path):
        # Signed by SOME valid key (the compromised peer's own) but attributed
        # to dev-a: fails against dev-a's registered key. The forgery VL-01
        # names -- merge steering by re-attribution -- refuses here.
        engine_a, engine_b, feed_a, feed_b, *_ = make_pair(tmp_path)
        evil = make_signer("evil")
        forged = attach_signature(rec("dev-a", payload={"title": "forged"}), evil)
        feed_a.record(forged)
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.refused == 1 and result.applied == 0

    def test_device_key_mismatch_is_the_same_refusal(self, tmp_path):
        # dev-b knows BOTH origins' keys; a record re-attributed from dev-c to
        # dev-a (the signature is dev-c's) fails dev-a's key: the lookup by
        # origin IS the device<->key binding.
        engine_a, engine_b, feed_a, feed_b, store_b, signer_a, _ = make_pair(tmp_path)
        signer_c = make_signer("origin-c")
        engine_b.register_peer(
            "dev-c", "rk-c", label="C", signing_pub=b64(signer_c.public_key())
        )
        legit_c = attach_signature(rec("dev-c", rid="c9"), signer_c)
        reattributed = dataclasses.replace(legit_c, device="dev-a")
        # The signature was over device=dev-c; re-encoding under dev-a keeps
        # the old signature bytes, which now fail dev-a's key AND the recipe.
        feed_a.record(reattributed)
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.refused == 1 and result.applied == 0

    def test_refused_never_prompts_approval(self, tmp_path):
        # A forged SKILL record (sensitive kind) must be refused at the
        # signature seam WITHOUT consulting the approval gate.
        engine_a, engine_b, feed_a, feed_b, *_ = make_pair(tmp_path)
        forged_skill = new_record(
            RecordKind.SKILL,
            "s1",
            {"body": "evil"},
            device="dev-a",
            clock=1,
        )
        feed_a.record(forged_skill)  # unsigned, keyed origin -> refusal
        prompts = []

        def approval_fn(conv_id, label, args):
            prompts.append(label)
            return True

        result = engine_b.run_round(
            "dev-a", FakeServingPeer(feed_a, "dev-a"), approval_fn=approval_fn
        )
        assert result.refused == 1
        assert prompts == []

    def test_own_records_coming_back_verify_against_own_key(self, tmp_path):
        # The backstop/resync case: dev-b's own signed records served back by
        # a peer verify against dev-b's own public key and apply idempotently.
        engine_a, engine_b, feed_a, feed_b, store_b, signer_a, signer_b = make_pair(
            tmp_path
        )
        engine_b.publish_conversation("mine", {"title": "b"}, clock=1)
        own = next(r for r in feed_b.current_records() if r.record_id == "mine")
        feed_a.record(own)  # A relays B's record back
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.refused == 0
        assert result.applied == 0  # idempotent: nothing new

    def test_verification_unavailable_accepts_unverified_with_count(self, tmp_path):
        engine_a, _, feed_a, _, _, signer_a, _ = make_pair(tmp_path)
        engine_a.publish_conversation("c1", {"title": "hello"}, clock=1)
        feed_b = ChangeFeed(root=tmp_path / "b2")
        store_b = PeerStore(root=tmp_path / "b2")
        engine_b = SyncEngine(
            device="dev-b2",
            feed=feed_b,
            store=store_b,
            signer=UnavailableSigner(),
        )
        engine_b.register_peer(
            "dev-a", "rk-a", label="A", signing_pub=b64(signer_a.public_key())
        )
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.unverified == 1
        assert result.refused == 0
        assert result.applied == 1


# ---------------------------------------------------------------------------
# Decision 6 -- the bounded grace for unkeyed (pre-VL-01) origins
# ---------------------------------------------------------------------------


class TestGraceWindow:
    def test_unkeyed_origin_accepted_and_counted_under_grace(self, tmp_path):
        engine_a, engine_b, feed_a, feed_b, *_ = make_pair(
            tmp_path, register_key=False
        )
        feed_a.record(rec("dev-a"))  # unsigned, origin has NO registered key
        assert signing_mod.ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS is True
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.unverified == 1
        assert result.refused == 0
        assert result.applied == 1

    def test_unkeyed_origin_refused_once_the_window_closes(
        self, tmp_path, monkeypatch
    ):
        engine_a, engine_b, feed_a, feed_b, *_ = make_pair(
            tmp_path, register_key=False
        )
        feed_a.record(rec("dev-a"))
        monkeypatch.setattr(
            signing_mod, "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS", False
        )
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.refused == 1
        assert result.unverified == 0
        assert result.applied == 0

    def test_grace_never_admits_unsigned_from_keyed_origin(self, tmp_path):
        # The window is for UNKEYED origins only; a keyed origin's unsigned
        # record refuses, window or no window.
        engine_a, engine_b, feed_a, feed_b, *_ = make_pair(tmp_path)
        feed_a.record(rec("dev-a"))
        assert signing_mod.ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS is True
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        assert result.refused == 1 and result.unverified == 0

    def test_republish_signed_signs_the_local_set_at_same_clock(self, tmp_path):
        signer = make_signer("a")
        feed = ChangeFeed(root=tmp_path)
        unsigned_engine = SyncEngine(
            device="dev-a", feed=feed, signer=UnavailableSigner()
        )
        unsigned_engine.publish_conversation("c1", {"title": "x"}, clock=1)
        unsigned_engine.publish_conversation("gone", {}, clock=1, deleted=True)
        foreign = attach_signature(rec("dev-z", rid="z1"), make_signer("z"))
        feed.record(foreign)
        engine = SyncEngine(device="dev-a", feed=feed, signer=signer)
        count = engine.republish_signed()
        assert count == 2  # c1 + the tombstone; the foreign record untouched
        by_id = {r.record_id: r for r in feed.current_records()}
        assert by_id["c1"].signature and by_id["c1"].clock == 1
        assert by_id["gone"].signature and by_id["gone"].deleted
        assert by_id["z1"].signature == foreign.signature
        assert feed.current_clock(RecordKind.CONVERSATION, "c1") == 1
        # Idempotent: a second run republishes nothing.
        assert engine.republish_signed() == 0


# ---------------------------------------------------------------------------
# Surfacing: RoundResult -> sync_status -> API payload
# ---------------------------------------------------------------------------


class TestSurfacing:
    def test_round_result_fields_flow_to_status_and_payload(self, tmp_path):
        from opti_oignon.api.routes_sync import (
            _outcome_to_dict,
            round_result_to_dict,
        )
        from opti_oignon.veilid.sync_status import SyncStatusStore

        engine_a, engine_b, feed_a, feed_b, *_ = make_pair(tmp_path)
        feed_a.record(rec("dev-a"))  # one refusal
        result = engine_b.run_round("dev-a", FakeServingPeer(feed_a, "dev-a"))
        payload = round_result_to_dict(result)
        assert payload["refused"] == 1
        assert payload["unverified"] == 0
        store = SyncStatusStore()
        outcome = store.record_round(result)
        assert outcome.refused == 1
        assert outcome.unverified == 0
        out_payload = _outcome_to_dict(outcome)
        assert out_payload["refused"] == 1
        assert out_payload["unverified"] == 0

    def test_payloads_read_pre_vl01_results_defensively(self):
        from opti_oignon.api.routes_sync import round_result_to_dict

        class OldResult:
            peer_id = "p"
            applied = 1
            deferred = 0
            conflicts = 0
            rejected = 0
            previous_watermark = 0
            new_watermark = 1
            advanced = True

        payload = round_result_to_dict(OldResult())
        assert payload["refused"] == 0
        assert payload["unverified"] == 0

    def test_round_payload_exact_shape(self):
        # Re-assertion of the superseded f9c
        # TestSYN03ParsedFlag::test_round_result_to_dict_includes_parsed
        # (itself the re-assertion of the superseded s180 shape test): the
        # payload under strict equality, now carrying refused/unverified.
        from opti_oignon.api.routes_sync import round_result_to_dict

        rr = _engine_mod.RoundResult(
            peer_id="B",
            applied=2,
            deferred=1,
            conflicts=3,
            rejected=0,
            previous_watermark=1,
            new_watermark=5,
            advanced=True,
        )
        d = round_result_to_dict(rr)
        assert d == {
            "peer_id": "B",
            "applied": 2,
            "deferred": 1,
            "conflicts": 3,
            "rejected": 0,
            "refused": 0,
            "unverified": 0,
            "previous_watermark": 1,
            "new_watermark": 5,
            "advanced": True,
            "parsed": True,
        }


# ---------------------------------------------------------------------------
# Decision 5 -- the pairing payload extension and the peer column
# ---------------------------------------------------------------------------


class TestPairingExtension:
    def test_new_payload_carries_key_under_integrity(self):
        pub = b64(b"PUB")
        payload = build_pairing_payload("dev-a", "rk-a", pub)
        assert payload["signing_pub"] == pub
        assert payload["integrity"] == pairing_integrity("dev-a", "rk-a", pub)
        parsed = parse_pairing_payload(payload)
        assert parsed is not None
        assert parsed.signing_pub == pub

    def test_tampered_key_fails_integrity(self):
        payload = build_pairing_payload("dev-a", "rk-a", b64(b"PUB"))
        payload["signing_pub"] = b64(b"EVIL")
        assert parse_pairing_payload(payload) is None

    def test_stripped_key_fails_integrity(self):
        # Removing the key in transit breaks the five-field digest: tampering
        # never degrades silently into "no key".
        payload = build_pairing_payload("dev-a", "rk-a", b64(b"PUB"))
        del payload["signing_pub"]
        assert parse_pairing_payload(payload) is None

    def test_old_payload_parses_as_pre_vl01(self):
        payload = build_pairing_payload("dev-a", "rk-a")
        assert "signing_pub" not in payload
        assert payload["integrity"] == pairing_integrity("dev-a", "rk-a")
        parsed = parse_pairing_payload(payload)
        assert parsed is not None
        assert parsed.signing_pub is None

    def test_old_reader_rejects_new_payload_closed(self):
        # The honest compat matrix: an old reader's fixed four-field
        # recomputation cannot match the five-field digest; the ceremony
        # fails CLOSED (upgrade, re-pair), never accepts a key it cannot
        # protect. Emulated by recomputing the old recipe.
        payload = build_pairing_payload("dev-a", "rk-a", b64(b"PUB"))
        old_expected = pairing_integrity("dev-a", "rk-a")
        assert old_expected != payload["integrity"]

    def test_accept_threads_key_to_engine(self):
        registered = {}

        class FakeEngine:
            device = "me"

            def register_peer(self, peer_id, routing_key, *, label="", signing_pub=None):
                registered.update(
                    peer_id=peer_id, routing_key=routing_key, signing_pub=signing_pub
                )
                return "ok"

        payload = build_pairing_payload("dev-a", "rk-a", b64(b"PUB"))
        assert accept_pairing_payload(FakeEngine(), payload) == "ok"
        assert registered["signing_pub"] == b64(b"PUB")

    def test_accept_falls_back_for_pre_vl01_engine(self):
        calls = []

        class OldEngine:
            device = "me"

            def register_peer(self, peer_id, routing_key, *, label=""):
                calls.append((peer_id, routing_key, label))
                return "ok"

        payload = build_pairing_payload("dev-a", "rk-a", b64(b"PUB"))
        assert accept_pairing_payload(OldEngine(), payload) == "ok"
        assert calls == [("dev-a", "rk-a", "")]


class TestPeerColumn:
    def test_column_round_trips_with_additive_migration(self, tmp_path):
        import sqlite3

        db = tmp_path / "veilid_peers.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE veilid_peers ("
            "peer_id TEXT PRIMARY KEY, routing_key TEXT NOT NULL, "
            "label TEXT NOT NULL DEFAULT '', "
            "watermark INTEGER NOT NULL DEFAULT 0, "
            "added_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO veilid_peers VALUES ('old-peer', 'rk', '', 7, 't', 't')"
        )
        conn.commit()
        conn.close()
        store = PeerStore(root=tmp_path)
        old = store.get_peer("old-peer")
        assert old is not None
        assert old.signing_pub is None
        assert old.watermark == 7
        store.add_peer("new-peer", "rk2", signing_pub="KEY")
        got = store.get_peer("new-peer")
        assert got.signing_pub == "KEY"

    def test_repair_refreshes_key_with_route(self, tmp_path):
        store = PeerStore(root=tmp_path)
        store.add_peer("p", "rk1", signing_pub="K1")
        store.advance_watermark("p", 5)
        rec2 = store.add_peer("p", "rk2", signing_pub="K2")
        assert rec2.routing_key == "rk2"
        assert rec2.signing_pub == "K2"
        assert rec2.watermark == 5  # preserved, as ever

    def test_repair_without_key_preserves_stored_key(self, tmp_path):
        # A pre-VL-01 payload can never strip a registered key (the silent
        # downgrade into the grace path).
        store = PeerStore(root=tmp_path)
        store.add_peer("p", "rk1", signing_pub="K1")
        rec2 = store.add_peer("p", "rk2")
        assert rec2.signing_pub == "K1"
        assert rec2.routing_key == "rk2"

    def test_invalid_signing_pub_rejected(self, tmp_path):
        store = PeerStore(root=tmp_path)
        with pytest.raises(ValueError):
            store.add_peer("p", "rk", signing_pub="")
        with pytest.raises(ValueError):
            store.add_peer("p", "rk", signing_pub=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Mode posture: Bulbe unchanged at the wire; custody and signing mode-free
# ---------------------------------------------------------------------------


class TestModePosture:
    def test_local_signing_and_custody_mode_free(self, tmp_path):
        # Under Bulbe (the REAL gate, driven through the stubbed mode):
        # publishing -- and therefore signing -- the republish, and peer
        # registration stay permitted; only the wire refuses, unchanged.
        set_mode("bulbe")
        signer = make_signer("a")
        feed = ChangeFeed(root=tmp_path)
        store = PeerStore(root=tmp_path)
        engine = SyncEngine(device="dev-a", feed=feed, store=store, signer=signer)
        seq = engine.publish_conversation("c1", {"title": "x"}, clock=1)
        assert seq >= 1
        assert feed.current_records()[0].signature
        assert engine.republish_signed() == 0  # already signed; still permitted
        engine.register_peer("dev-z", "rk", signing_pub="K")  # registry mode-free
        # The wire refuses, unchanged.
        with pytest.raises(VeilidDisabledInBulbe):
            engine.run_round("dev-z", FakeServingPeer(feed, "dev-z"))
        with pytest.raises(VeilidDisabledInBulbe):
            engine.serve_request({"v": 1})


# ---------------------------------------------------------------------------
# The signed batch through the real envelope (parse caps, decode, epoch intact)
# ---------------------------------------------------------------------------


class TestEnvelopeIntegration:
    def test_signed_batch_builds_parses_and_carries_signatures(self, tmp_path):
        signer = make_signer("a")
        feed = ChangeFeed(root=tmp_path)
        engine = SyncEngine(device="dev-a", feed=feed, signer=signer)
        engine.publish_conversation("c1", {"title": "hello"}, clock=1)
        batch = build_record_batch(feed, device="dev-a", watermark=0)
        assert batch["records"] and "signature" in batch["records"][0]
        parsed = parse_record_batch(batch)
        assert parsed is not None
        assert parsed.records[0].signature
        assert parsed.rejected == 0

    def test_sender_bounds_unchanged_constants(self):
        # The S203 caps stand; signatures ride INSIDE the measured records.
        assert SENDER_MAX_RECORDS == 256
        assert SENDER_MAX_BYTES == 1_048_576
