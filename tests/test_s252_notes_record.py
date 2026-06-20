#!/usr/bin/env python3
"""S252 -- the Notes feature's N.8 Veilid sync, container-provable half: notes as
a FIFTH user record type on the sync engine.

The Notes backend is complete through the two media post-processing blocs (S250
voice transcription, S251 picture caption / OCR). N.8 makes a note a syncable
record so the desktop fleet (and later the phone) converges note state over the
Veilid DHT, E2E. The desktop sync engine already moves four record kinds
(CONVERSATION, MEMORY_CANONICAL, MEMORY_ARCHIVE, SKILL) through one convergent
reconcile, one VL-01 signing floor, and one PAIR-02 trust boundary; this bloc
adds the fifth kind by exact symmetry with the conversation kind, the user-content
sibling (notes, like conversations, are the user's own content and apply without
the human gate; only SKILL, the executable kind, is gated).

This suite pins the CONTAINER-PROVABLE half:

 1. RecordKind.NOTE -- the fifth member, additive against the test_s241 presence
    pin; the four prior members intact; RECORD_KINDS (the decoder allowlist,
    derived from the enum) grows to admit "note", so the decoder accepts a
    note-kind record rather than rejecting an unknown kind.
 2. note_record -- the producer, a sibling of conversation_record: a NOTE record
    over an opaque, JSON-safe payload, the defensive None / non-mapping contract,
    the body CRDT carried verbatim (the backend never interprets note structure).
 3. publish_note -- the engine convenience, a sibling of publish_conversation:
    journals a NOTE change, returns its sequence, local-disk and mode-free (it is
    not gated by Bulbe; only moving a delta over the wire is Daily-only), and a
    deleted=True publish journals a tombstone.
 4. NOTE is user content, not sensitive: NOTE is NOT in SENSITIVE_KINDS, so a note
    received in a round APPLIES without the human gate (unlike a skill, which
    defers); SENSITIVE_KINDS stays SKILL-only.
 5. VL-01 on a note: the kind-agnostic signer signs a note over its canonical
    bytes and verifies it; a tampered note fails verification (re-clocking or
    re-attribution is refused); an unsigned note from an unkeyed origin REFUSES at
    a round (the ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS = False floor covers notes
    too); the constant stays False.
 6. Tombstone + opaque body convergence: a note tombstone wins reconciliation
    (tombstone-wins), OR-Set-shaped tags ride in the payload carried verbatim (the
    backend is CRDT-agnostic), and an opaque body_crdt blob survives a
    sign -> round -> apply round-trip unchanged.
 7. Seams + import additivity: the conversation producer and publish method, and
    the RecordKind presence-pin shape, are pinned (a later edit that removes a
    premise turns this suite red); the sync_engine import block and the producer /
    publish sources gain the note symbols.
 8. AST: the touched sources parse; the suite self-parses.

What is host-assured and NOT in this suite (NOTES_SYNC_E2E_S252.md): the live sync
round over a real Veilid DHT with the real ML-DSA-65 signer (liboqs / oqs are
absent in-container); the in-browser capture / gallery / canvas UIs (the N.5 /
N.6 / N.7 UI halves); and the phone-app note sync (N.9).

Red-before discipline: on the pristine S251 tree (no RecordKind.NOTE, no
note_record, no publish_note) every new-surface pin FAILS -- the new symbols are
bound INSIDE the test bodies through hasattr / getattr so absence is a clean
failure, never a collection error -- while the premise / seam guards pass by
design (they pin pre-existing invariants this bloc relies on). The harness is the
S208 idiom: opti_oignon stubbed, the security mode driven, the audit log a
recorder, a deterministic FakeSigner per device, the real protocol responder as
the fake peer. Nothing here imports the package's ollama chain.

``checkpoint_before_apply`` discipline and the auth core (auth.py, auth_2fa.py,
emergency_stop.py) are untouched by this bloc.
"""

from __future__ import annotations

import ast
import hashlib
import hmac as hmac_mod
import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"

RECORDS_SRC = VEILID / "records.py"
PRODUCERS_SRC = VEILID / "producers.py"
ENGINE_SRC = VEILID / "sync_engine.py"

_MODE = {"fn": lambda: "daily"}
_AUDIT: dict = {"events": []}


def set_mode(value: str = "daily") -> None:
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
_change_feed_mod = importlib.import_module("opti_oignon.veilid.change_feed")
_peers_mod = importlib.import_module("opti_oignon.veilid.peers")
_records_mod = importlib.import_module("opti_oignon.veilid.records")
_producers_mod = importlib.import_module("opti_oignon.veilid.producers")
_ledger_mod = importlib.import_module("opti_oignon.veilid.deferred_ledger")
_engine_mod = importlib.import_module("opti_oignon.veilid.sync_engine")
_reconcile_mod = importlib.import_module("opti_oignon.veilid.reconcile")

ChangeFeed = _change_feed_mod.ChangeFeed
PeerStore = _peers_mod.PeerStore
DeferredLedger = _ledger_mod.DeferredLedger
SyncEngine = _engine_mod.SyncEngine
new_record = _records_mod.new_record
attach_signature = signing_mod.attach_signature
verify_record_signature = signing_mod.verify_record_signature

UTC0 = "2026-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
    _AUDIT["events"].clear()
    _change_feed_mod.reset_change_feed()
    _peers_mod.reset_peer_store()
    _ledger_mod.reset_deferred_ledger()
    _engine_mod.reset_sync_engine()
    signing_mod.reset_record_signer()
    yield
    _change_feed_mod.reset_change_feed()
    _peers_mod.reset_peer_store()
    _ledger_mod.reset_deferred_ledger()
    _engine_mod.reset_sync_engine()
    signing_mod.reset_record_signer()
    set_mode("daily")


# ---------------------------------------------------------------------------
# The deterministic fake signer (the S205 / S208 seam) and peers / engines
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


def b64(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii")


class CountingPeer:
    """A peer answering from its own feed through the real responder."""

    def __init__(self, feed, device: str) -> None:
        self._feed = feed
        self._device = device
        self.fetch_calls = 0

    def fetch(self, request):
        from opti_oignon.veilid.protocol import respond_to_request

        self.fetch_calls += 1
        return respond_to_request(self._feed, request, device=self._device)


def asker(tmp_path, *, device="dev-b", seed="asker-b", signer=None):
    """An asking engine: its own feed, store, ledger, and signer."""
    feed = ChangeFeed(root=tmp_path / device)
    store = PeerStore(root=tmp_path / device)
    ledger = DeferredLedger(root=tmp_path / device)
    eng = SyncEngine(
        device=device,
        feed=feed,
        store=store,
        signer=signer if signer is not None else make_signer(seed),
        ledger=ledger,
    )
    return eng, feed, store, ledger


# ---------------------------------------------------------------------------
# New-surface helpers: bind the new symbols INSIDE the call so absence on the
# pristine tree is a clean FAILURE (assert), never a collection error.
# ---------------------------------------------------------------------------


def _note_kind():
    RecordKind = _records_mod.RecordKind
    assert hasattr(RecordKind, "NOTE"), "RecordKind.NOTE missing (N.8 not landed)"
    return RecordKind.NOTE


def make_note(rid: str, clock: int = 1, *, device: str = "dev-a", payload=None):
    """A NOTE SyncRecord; asserts the kind exists so pristine fails cleanly."""
    if payload is None:
        payload = {"body_crdt": "opaque-blob-" + rid, "tags": [], "pinned": False}
    return new_record(
        _note_kind(),
        rid,
        payload,
        device=device,
        clock=clock,
        updated_at=UTC0,
    )


def _note_record():
    nr = getattr(_producers_mod, "note_record", None)
    assert nr is not None, "producers.note_record missing (N.8 not landed)"
    return nr


def skill(rid: str = "s1", clock: int = 1, *, device: str = "dev-a"):
    return new_record(
        _records_mod.RecordKind.SKILL,
        rid,
        {"body": "code"},
        device=device,
        clock=clock,
        updated_at=UTC0,
    )


# ---------------------------------------------------------------------------
# Family 1 -- RecordKind.NOTE (the fifth kind, additive)
# ---------------------------------------------------------------------------


class TestRecordKindNote:
    def test_note_member_present(self):
        RecordKind = _records_mod.RecordKind
        assert hasattr(RecordKind, "NOTE"), "RecordKind.NOTE missing"
        assert RecordKind.NOTE.value == "note"

    def test_prior_four_members_intact(self):
        # Premise guard: green before and after. The fifth kind never disturbs
        # the four it joins.
        RecordKind = _records_mod.RecordKind
        assert RecordKind.CONVERSATION.value == "conversation"
        assert RecordKind.MEMORY_CANONICAL.value == "memory_canonical"
        assert RecordKind.MEMORY_ARCHIVE.value == "memory_archive"
        assert RecordKind.SKILL.value == "skill"

    def test_record_kinds_allowlist_includes_note(self):
        assert "note" in _records_mod.RECORD_KINDS

    def test_record_kinds_still_includes_prior(self):
        # Premise guard.
        for v in ("conversation", "memory_canonical", "memory_archive", "skill"):
            assert v in _records_mod.RECORD_KINDS, v

    def test_decoder_accepts_note_kind(self):
        # The decoder allowlist is derived from the enum, so a fifth kind is
        # admitted (an unknown kind would be rejected).
        RecordKind = _records_mod.RecordKind
        assert hasattr(RecordKind, "NOTE"), "RecordKind.NOTE missing"
        rec = make_note("n1")
        wire = _records_mod.encode_record(rec)
        back = _records_mod.decode_record(wire)
        assert back is not None, "note-kind record rejected by the decoder allowlist"
        assert back.kind == RecordKind.NOTE
        assert back.record_id == "n1"


# ---------------------------------------------------------------------------
# Family 2 -- note_record (a sibling of conversation_record)
# ---------------------------------------------------------------------------


class TestNoteProducer:
    def test_note_record_exists(self):
        assert getattr(_producers_mod, "note_record", None) is not None

    def test_note_record_produces_note_kind(self):
        nr = _note_record()
        rec = nr("n1", {"body_crdt": "x"}, device="dev-a", clock=3)
        assert rec.kind == _note_kind()
        assert rec.record_id == "n1"
        assert rec.clock == 3
        assert rec.device == "dev-a"
        assert rec.deleted is False

    def test_note_record_carries_opaque_body_verbatim(self):
        # The backend never interprets note structure: an opaque body_crdt blob
        # rides in the payload untouched.
        nr = _note_record()
        body = {"y_update": "AAECAwQF", "schema": "yjs-v1"}
        rec = nr("n1", {"body_crdt": body, "title": "t"}, device="dev-a", clock=1)
        assert rec.payload["body_crdt"] == body
        assert rec.payload["title"] == "t"

    def test_note_record_normalises_none_payload(self):
        # None -> empty mapping (a contentless record / pure tombstone), the
        # conversation producer's defensive contract.
        nr = _note_record()
        rec = nr("n1", None, device="dev-a", clock=1)
        assert rec.payload == {}

    def test_note_record_rejects_non_mapping(self):
        # The producer-side contract raises on a non-mapping (never untrusted
        # wire data), like the conversation producer.
        nr = _note_record()
        with pytest.raises(ValueError):
            nr("n1", ["not", "a", "mapping"], device="dev-a", clock=1)

    def test_note_record_keyword_shape_mirrors_conversation(self):
        # Same call shape as conversation_record: deleted / updated_at keywords.
        nr = _note_record()
        rec = nr("n1", {"a": 1}, device="dev-a", clock=2, deleted=True, updated_at=UTC0)
        assert rec.deleted is True
        assert rec.updated_at == UTC0


# ---------------------------------------------------------------------------
# Family 3 -- publish_note (a sibling of publish_conversation)
# ---------------------------------------------------------------------------


class TestPublishNote:
    def test_publish_note_exists_on_engine(self, tmp_path):
        eng, _f, _s, _l = asker(tmp_path)
        assert getattr(eng, "publish_note", None) is not None

    def test_publish_note_journals_and_returns_sequence(self, tmp_path):
        eng, feed, _s, _l = asker(tmp_path)
        pn = getattr(eng, "publish_note", None)
        assert pn is not None, "engine.publish_note missing (N.8 not landed)"
        seq = pn("n1", {"body_crdt": "blob", "title": "hi"}, clock=1)
        assert isinstance(seq, int)
        ids = [r.record_id for r in feed.current_records()]
        assert "n1" in ids
        rec = next(r for r in feed.current_records() if r.record_id == "n1")
        assert rec.kind == _note_kind()

    def test_publish_note_mode_free_local(self, tmp_path):
        # Journalling is local-disk: permitted in Bulbe (only moving a delta over
        # the wire is Daily-only). publish_conversation has this property; the
        # note sibling inherits it.
        set_mode("bulbe")
        eng, feed, _s, _l = asker(tmp_path)
        pn = getattr(eng, "publish_note", None)
        assert pn is not None, "engine.publish_note missing (N.8 not landed)"
        pn("n1", {"title": "in bulbe"}, clock=1)
        assert any(r.record_id == "n1" for r in feed.current_records())

    def test_publish_note_tombstone(self, tmp_path):
        eng, feed, _s, _l = asker(tmp_path)
        pn = getattr(eng, "publish_note", None)
        assert pn is not None, "engine.publish_note missing (N.8 not landed)"
        pn("n1", None, clock=2, deleted=True)
        rec = next(r for r in feed.current_records() if r.record_id == "n1")
        assert rec.deleted is True
        assert rec.payload == {}


# ---------------------------------------------------------------------------
# Family 4 -- NOTE is user content, not sensitive (applies without the gate)
# ---------------------------------------------------------------------------


class TestNoteNotSensitive:
    def test_note_not_in_sensitive_kinds(self):
        RecordKind = _records_mod.RecordKind
        assert hasattr(RecordKind, "NOTE"), "RecordKind.NOTE missing"
        assert RecordKind.NOTE.value not in _engine_mod.SENSITIVE_KINDS

    def test_skill_stays_sensitive(self):
        # Premise guard: the executable kind stays gated.
        assert _records_mod.RecordKind.SKILL.value in _engine_mod.SENSITIVE_KINDS

    def test_sensitive_kinds_is_skill_only(self):
        # Premise guard, green before and after: NOTE must not sneak in as
        # sensitive; only SKILL is gated.
        assert _engine_mod.SENSITIVE_KINDS == frozenset(
            {_records_mod.RecordKind.SKILL.value}
        )

    def test_note_applies_without_gate_in_a_round(self, tmp_path):
        # A signed note received in a round is APPLIED, not deferred to the human
        # gate (the SENSITIVE_KINDS gate lets a non-sensitive kind straight
        # through). Contrast: a skill defers.
        signer_a = make_signer("origin-a")
        feed_a = ChangeFeed(root=tmp_path / "raw-a")
        feed_a.record(attach_signature(make_note("n1", device="dev-a"), signer_a))

        eng, feed, store, _l = asker(tmp_path)
        store.add_peer("dev-a", "rk", signing_pub=b64(signer_a.public_key()))
        result = eng.run_round("dev-a", CountingPeer(feed_a, "dev-a"))
        assert result.applied == 1
        assert result.deferred == 0
        assert result.refused == 0
        assert any(r.record_id == "n1" for r in feed.current_records())

    def test_skill_defers_for_contrast(self, tmp_path):
        # The contrast that proves the note path is the non-gated one: a signed
        # skill defers to the ledger rather than applying.
        signer_a = make_signer("origin-a")
        feed_a = ChangeFeed(root=tmp_path / "raw-a")
        feed_a.record(attach_signature(skill("s1", device="dev-a"), signer_a))

        eng, feed, store, _l = asker(tmp_path)
        store.add_peer("dev-a", "rk", signing_pub=b64(signer_a.public_key()))
        result = eng.run_round("dev-a", CountingPeer(feed_a, "dev-a"))
        assert result.deferred == 1
        assert result.applied == 0


# ---------------------------------------------------------------------------
# Family 5 -- VL-01 on a note (the kind-agnostic signer)
# ---------------------------------------------------------------------------


class TestNoteSigning:
    def test_note_signs_and_verifies(self):
        signer = make_signer("dev-a")
        signed = attach_signature(make_note("n1", device="dev-a"), signer)
        assert signed.signature != ""
        assert verify_record_signature(signed, signer.public_key(), signer) is True

    def test_tampered_note_fails_verification(self):
        # Re-clocking / re-attribution is refused: the signature binds the
        # canonical bytes, so a mutated note no longer verifies.
        signer = make_signer("dev-a")
        signed = attach_signature(make_note("n1", clock=1, device="dev-a"), signer)
        from dataclasses import replace

        tampered = replace(signed, clock=999)
        assert (
            verify_record_signature(tampered, signer.public_key(), signer) is False
        )

    def test_unsigned_note_from_unkeyed_origin_refuses_at_round(self, tmp_path):
        # The VL-01 floor covers notes: an unsigned note from an origin with no
        # registered signing key REFUSES (counted, never applied), the closed
        # migration window.
        eng, feed, store, _l = asker(tmp_path)
        feed_a = ChangeFeed(root=tmp_path / "raw-a")
        feed_a.record(make_note("n1", device="dev-a"))  # unsigned
        store.add_peer("dev-a", "rk")  # no signing key
        result = eng.run_round("dev-a", CountingPeer(feed_a, "dev-a"))
        assert result.refused == 1
        assert result.applied == 0
        assert all(r.record_id != "n1" for r in feed.current_records())

    def test_accept_unsigned_constant_still_false(self):
        # Premise guard: the closed-window hard constant.
        assert signing_mod.ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS is False


# ---------------------------------------------------------------------------
# Family 6 -- tombstone + opaque-body convergence (kind-agnostic reconcile)
# ---------------------------------------------------------------------------


class TestNoteConvergence:
    def test_note_tombstone_converges(self):
        # A note tombstone (higher clock) wins reconciliation; the deletion is
        # not silently resurrected.
        reconcile = _reconcile_mod.reconcile
        tombstone_record = _producers_mod.tombstone_record
        live = make_note("n1", clock=1, device="dev-a")
        dead = tombstone_record(_note_kind(), "n1", device="dev-a", clock=2)
        result = reconcile([live], [dead])
        winners = {(r.kind.value, r.record_id): r for r in result.records}
        w = winners[("note", "n1")]
        assert w.deleted is True
        assert w.clock == 2

    def test_orset_tags_carried_verbatim(self):
        # The tags OR-Set rides in the payload; the backend stores it without
        # interpreting it (CRDT-agnostic). The producer carries the shape as-is.
        nr = _note_record()
        tags = {"add": ["work", "idea"], "remove": ["draft"]}
        rec = nr("n1", {"tags": tags, "body_crdt": "blob"}, device="dev-a", clock=1)
        assert rec.payload["tags"] == tags

    def test_opaque_body_survives_round_trip(self, tmp_path):
        # An opaque body_crdt blob survives sign -> round -> apply unchanged: the
        # backend relays the blob without interpreting note structure.
        body = {"y_update": "QkFTRTY0LWJsb2I=", "v": 7}
        signer_a = make_signer("origin-a")
        feed_a = ChangeFeed(root=tmp_path / "raw-a")
        feed_a.record(
            attach_signature(
                make_note("n1", device="dev-a", payload={"body_crdt": body}),
                signer_a,
            )
        )
        eng, feed, store, _l = asker(tmp_path)
        store.add_peer("dev-a", "rk", signing_pub=b64(signer_a.public_key()))
        eng.run_round("dev-a", CountingPeer(feed_a, "dev-a"))
        rec = next(r for r in feed.current_records() if r.record_id == "n1")
        assert rec.payload["body_crdt"] == body


# ---------------------------------------------------------------------------
# Family 7 -- seams (green before and after) + import / source additivity
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


class TestSeams:
    def test_conversation_producer_seam_present(self):
        # Premise guard: the sibling this bloc mirrors.
        assert getattr(_producers_mod, "conversation_record", None) is not None

    def test_publish_conversation_seam_present(self, tmp_path):
        eng, _f, _s, _l = asker(tmp_path)
        assert getattr(eng, "publish_conversation", None) is not None

    def test_record_kind_presence_pin_holds(self):
        # Premise guard: the only canonical RecordKind pin (test_s241 shape).
        assert "class RecordKind" in _read(RECORDS_SRC)

    def test_producers_note_record_source_present(self):
        assert "def note_record(" in _read(PRODUCERS_SRC)

    def test_engine_publish_note_source_present(self):
        assert "def publish_note(" in _read(ENGINE_SRC)

    def test_sync_engine_imports_note_record(self):
        # The producers import block gains note_record (additive; no pin fixes the
        # exact import set).
        src = _read(ENGINE_SRC)
        assert "note_record" in src


# ---------------------------------------------------------------------------
# Family 8 -- AST
# ---------------------------------------------------------------------------


class TestAST:
    def test_touched_sources_parse(self):
        for path in (RECORDS_SRC, PRODUCERS_SRC, ENGINE_SRC):
            src = _read(path)
            assert src != "", str(path)
            ast.parse(src, filename=str(path))

    def test_this_suite_parses(self):
        src = _read(Path(__file__))
        assert src != ""
        ast.parse(src, filename=__file__)
