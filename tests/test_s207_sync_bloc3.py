#!/usr/bin/env python3
"""S207 -- Sync cycle Bloc 3: SYN-05 per-record deferred ledger.

The lot under test: a denied sensitive record no longer pins the peer's
watermark. The gate persists it to the new deferred ledger
(veilid/deferred_ledger.py) with its full wire envelope, the round advances
past every consumed chunk, a still-pending record arriving again dedups into
its entry silently (no re-prompt, no re-fetch beyond the normal delta), and
the human approves or refuses from the panel surface -- an approval
re-entering the engine's verify -> gate -> apply seam against the CURRENT
trust state, a refusal removing the entry without applying.

Designed supersessions (deselect-plus-reassert; originals never edited):
- test_s180_sync_engine.py::TestApprovalGate::test_skill_deferred_when_denied_and_watermark_held
- test_s180_sync_engine.py::TestApprovalGate::test_mixed_batch_defers_only_sensitive
- test_s180_sync_engine.py::TestApprovalGate::test_deferred_skill_reoffered_then_applied
- test_s203_sync_bloc1_lot1.py::...::test_single_chunk_defer_holds_at_previous
- test_s203_sync_bloc1_lot1.py::...::test_deferred_in_chunk_k_holds_at_boundary
- test_s204_sync_bloc1_lot2.py::...::test_deferred_hold_is_reoffered_from_zero_after_a_reset
Each is re-asserted here under the S207 semantics (the watermark ADVANCES,
the record persists, the re-offer comes from the ledger).

Harness: the s205/s206 idiom -- opti_oignon stubbed, the security mode driven,
the audit log a recorder, FakeSigner per device, the real protocol responder
as the fake peer, routes driven through TestClient with injected resolvers.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import importlib
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"
API = OO / "api"

_MODE = {"fn": lambda: "daily"}
_AUDIT: dict = {"events": []}


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
_records_mod = importlib.import_module("opti_oignon.veilid.records")
_ledger_mod = importlib.import_module("opti_oignon.veilid.deferred_ledger")
_engine_mod = importlib.import_module("opti_oignon.veilid.sync_engine")
_status_mod = importlib.import_module("opti_oignon.veilid.sync_status")
rs = importlib.import_module("opti_oignon.api.routes_sync")

ChangeFeed = _change_feed_mod.ChangeFeed
PeerStore = _peers_mod.PeerStore
DeferredLedger = _ledger_mod.DeferredLedger
RecordKind = _records_mod.RecordKind
new_record = _records_mod.new_record
decode_record = _records_mod.decode_record
attach_signature = signing_mod.attach_signature
SyncEngine = _engine_mod.SyncEngine
DeferredNotFound = _engine_mod.DeferredNotFound
PeerNotFound = _engine_mod.PeerNotFound
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe

OFFER_INSERTED = _ledger_mod.OFFER_INSERTED
OFFER_REPLACED = _ledger_mod.OFFER_REPLACED
OFFER_DUPLICATE = _ledger_mod.OFFER_DUPLICATE
OFFER_STALE = _ledger_mod.OFFER_STALE


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
    _AUDIT["events"].clear()
    _change_feed_mod.reset_change_feed()
    _peers_mod.reset_peer_store()
    _ledger_mod.reset_deferred_ledger()
    _engine_mod.reset_sync_engine()
    _status_mod.reset_sync_status_store()
    signing_mod.reset_record_signer()
    rs.reset_peer_resolver()
    rs.reset_self_routing_resolver()
    yield
    _change_feed_mod.reset_change_feed()
    _peers_mod.reset_peer_store()
    _ledger_mod.reset_deferred_ledger()
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


def b64(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Records, peers, engines
# ---------------------------------------------------------------------------


def conv(rid: str, clock: int = 1, *, device: str = "dev-a") -> object:
    return new_record(
        RecordKind.CONVERSATION,
        rid,
        {"title": "hello " + rid},
        device=device,
        clock=clock,
        updated_at="2026-01-01T00:00:00+00:00",
    )


def skill(rid: str = "s1", clock: int = 1, *, device: str = "dev-a", body: str = "code") -> object:
    return new_record(
        RecordKind.SKILL,
        rid,
        {"body": body},
        device=device,
        clock=clock,
        updated_at="2026-01-01T00:00:00+00:00",
    )


class CountingPeer:
    """A peer answering from its own feed through the real responder, counting fetches."""

    def __init__(self, feed, device: str, *, max_count: int = 0) -> None:
        self._feed = feed
        self._device = device
        self._max = max_count
        self.fetch_calls = 0

    def fetch(self, request):
        from opti_oignon.veilid.protocol import respond_to_request

        self.fetch_calls += 1
        kwargs = {"device": self._device}
        if self._max:
            kwargs["max_count"] = self._max
        return respond_to_request(self._feed, request, **kwargs)


class CountingApproval:
    """An approval function recording every prompt; answers a fixed verdict."""

    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.calls: list = []

    def __call__(self, conversation_id, label, args):
        self.calls.append((conversation_id, label, dict(args)))
        return self.verdict


def asker(tmp_path, *, device="dev-b", seed="asker-b"):
    """An asking engine: its own feed, store, ledger, and signer."""
    feed = ChangeFeed(root=tmp_path / device)
    store = PeerStore(root=tmp_path / device)
    ledger = DeferredLedger(root=tmp_path / device)
    eng = SyncEngine(
        device=device, feed=feed, store=store, signer=make_signer(seed), ledger=ledger
    )
    return eng, feed, store, ledger


def origin(tmp_path, *, device="dev-a", seed="origin-a", name=None):
    """An originating engine that signs at publish (the S205 seam)."""
    feed = ChangeFeed(root=tmp_path / (name or device))
    signer = make_signer(seed)
    eng = SyncEngine(device=device, feed=feed, signer=signer)
    return eng, feed, signer


def audit_actions() -> list:
    return [a.get("action") for a in _AUDIT["events"]]


# ---------------------------------------------------------------------------
# A. The ledger module: CRUD, dedup, arbitration, idempotence, hygiene
# ---------------------------------------------------------------------------


class TestLedgerStore:
    def test_insert_get_list_count_remove_roundtrip(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        r = skill("s1", 1)
        assert led.offer(r, peer_id="dev-p") == OFFER_INSERTED
        assert led.has("skill", "s1") is True
        e = led.get("skill", "s1")
        assert e.kind == "skill" and e.record_id == "s1"
        assert e.origin_device == "dev-a" and e.peer_id == "dev-p"
        assert e.clock == 1 and e.content_hash == r.content_hash
        assert e.deferred_at and e.last_offered_at
        assert led.count() == 1 and len(led.list_entries()) == 1
        assert led.remove("skill", "s1") is True
        assert led.remove("skill", "s1") is False
        assert led.count() == 0

    def test_envelope_roundtrips_with_signature(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        signer = make_signer("env")
        r = attach_signature(skill("s1", 2), signer)
        assert r.signature
        led.offer(r, peer_id="dev-p")
        back = decode_record(led.get("skill", "s1").envelope)
        assert back is not None
        assert back.signature == r.signature
        assert back.clock == 2 and back.content_hash == r.content_hash

    def test_duplicate_refreshes_silently(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        r = skill("s1", 1)
        led.offer(r, peer_id="dev-p")
        first = led.get("skill", "s1")
        assert led.offer(r, peer_id="dev-p") == OFFER_DUPLICATE
        again = led.get("skill", "s1")
        assert again.deferred_at == first.deferred_at  # not a new decision
        assert again.clock == first.clock
        assert led.count() == 1  # idempotent

    def test_newer_clock_replaces(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        led.offer(skill("s1", 1, body="v1"), peer_id="dev-p")
        assert led.offer(skill("s1", 3, body="v3"), peer_id="dev-q") == OFFER_REPLACED
        e = led.get("skill", "s1")
        assert e.clock == 3 and e.peer_id == "dev-q"
        assert led.count() == 1

    def test_older_clock_is_stale(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        led.offer(skill("s1", 3, body="v3"), peer_id="dev-p")
        assert led.offer(skill("s1", 1, body="v1"), peer_id="dev-p") == OFFER_STALE
        assert led.get("skill", "s1").clock == 3

    def test_equal_clock_arbitrated_by_the_reconciler_recipe(self, tmp_path):
        # Two concurrent versions at the same clock: the stored entry ends up
        # holding whichever choose_winner picks, regardless of arrival order.
        from opti_oignon.veilid.reconcile import choose_winner

        a = skill("s1", 2, device="dev-a", body="alpha")
        b = skill("s1", 2, device="dev-b", body="beta")
        winner = choose_winner([a, b])
        for first, second in ((a, b), (b, a)):
            led = DeferredLedger(root=tmp_path / (first.device + second.device))
            led.offer(first, peer_id="p")
            led.offer(second, peer_id="p")
            assert led.get("skill", "s1").content_hash == winner.content_hash

    def test_corrupt_envelope_is_replaced_and_decodes_empty(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        led.offer(skill("s1", 5), peer_id="dev-p")
        with led._lock:
            led._conn().execute(
                f"UPDATE {_ledger_mod.TABLE_NAME} SET envelope = ?",
                ("{not json",),
            )
            led._conn().commit()
        assert led.get("skill", "s1").envelope == {}
        # Fresh provenance beats a corrupt row even at a lower clock.
        assert led.offer(skill("s1", 1), peer_id="dev-p") == OFFER_REPLACED
        assert led.get("skill", "s1").clock == 1

    def test_purge_below_is_strictly_below(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        led.offer(skill("s1", 2), peer_id="dev-p")
        assert led.purge_below("skill", "s1", 2) is False  # equal stands
        assert led.has("skill", "s1")
        assert led.purge_below("skill", "s1", 3) is True
        assert led.count() == 0

    def test_remove_for_peer_cascade(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        led.offer(skill("s1", 1), peer_id="dev-p")
        led.offer(skill("s2", 1), peer_id="dev-p")
        led.offer(skill("s3", 1), peer_id="dev-q")
        assert led.remove_for_peer("dev-p") == 2
        assert led.count() == 1 and led.has("skill", "s3")

    def test_wal_and_identifier_allowlist(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        led.offer(skill("s1", 1), peer_id="p")
        assert led.journal_mode() == "wal"
        with pytest.raises(ValueError):
            _ledger_mod._safe_table("evil_table")

    def test_singleton_lifecycle(self, tmp_path):
        _ledger_mod.set_deferred_ledger(DeferredLedger(root=tmp_path))
        led = _ledger_mod.get_deferred_ledger()
        led.offer(skill("s1", 1), peer_id="p")
        assert _ledger_mod.get_deferred_ledger().count() == 1
        _ledger_mod.reset_deferred_ledger()
        _ledger_mod.set_deferred_ledger(DeferredLedger(root=tmp_path / "fresh"))
        assert _ledger_mod.get_deferred_ledger().count() == 0

    def test_validation(self, tmp_path):
        led = DeferredLedger(root=tmp_path)
        with pytest.raises(ValueError):
            led.offer("not-a-record", peer_id="p")
        with pytest.raises(ValueError):
            led.offer(skill("s1", 1), peer_id="")
        assert led.get(None, "x") is None
        assert led.remove(None, "x") is False
        assert led.remove_for_peer("") == 0
        assert led.purge_below("skill", "s1", True) is False


# ---------------------------------------------------------------------------
# B. The watermark rule: advancing past deferrals (the designed supersessions)
# ---------------------------------------------------------------------------


class TestWatermarkAdvancesPastDeferrals:
    def test_single_chunk_defer_advances_and_persists(self, tmp_path):
        # Supersedes s180 test_skill_deferred_when_denied_and_watermark_held
        # and s203 test_single_chunk_defer_holds_at_previous: the watermark now
        # ADVANCES and the record persists to the ledger.
        eng, feed, store, ledger = asker(tmp_path)
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(make_signer("origin-a").public_key()))
        gate = CountingApproval(False)
        peer = CountingPeer(o_feed, "dev-a")
        res = eng.run_round("dev-a", peer, approval_fn=gate)
        assert res.deferred == 1 and res.applied == 0
        assert res.advanced is True
        assert res.new_watermark == o_feed.high_water()
        assert store.get_watermark("dev-a") == o_feed.high_water()
        assert all(r.record_id != "s1" for r in feed.current_records())  # NOT applied
        assert ledger.has("skill", "s1")
        assert len(gate.calls) == 1  # prompted exactly once
        assert "sync_deferred" in audit_actions()

    def test_mixed_batch_applies_nonsensitive_and_advances(self, tmp_path):
        # Supersedes s180 test_mixed_batch_defers_only_sensitive: advanced is
        # now True; the conversation applies, the skill quarantines.
        eng, feed, store, ledger = asker(tmp_path)
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(conv("c1", 1, device="dev-a"))
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(make_signer("origin-a").public_key()))
        res = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        assert res.applied == 1 and res.deferred == 1
        assert res.advanced is True
        ids = {r.record_id for r in feed.current_records()}
        assert "c1" in ids and "s1" not in ids
        assert ledger.has("skill", "s1")

    def test_deferred_in_chunk_k_consumes_every_chunk(self, tmp_path):
        # Supersedes s203 test_deferred_in_chunk_k_holds_at_boundary: the
        # deferring chunk no longer holds; the round consumes to the end.
        eng, feed, store, ledger = asker(tmp_path)
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        for r in (conv("c1", 1, device="dev-a"), conv("c2", 1, device="dev-a"),
                  skill("s1", 1, device="dev-a"), conv("c3", 1, device="dev-a")):
            o_eng.publish(r)
        store.add_peer("dev-a", "rk", signing_pub=b64(make_signer("origin-a").public_key()))
        res = eng.run_round(
            "dev-a", CountingPeer(o_feed, "dev-a", max_count=2),
            approval_fn=lambda c, l, a: False,
        )
        assert res.deferred == 1
        assert res.legs >= 2
        assert res.new_watermark == o_feed.high_water()  # past the deferring chunk
        ids = {r.record_id for r in feed.current_records()}
        assert {"c1", "c2", "c3"} <= ids and "s1" not in ids
        assert ledger.has("skill", "s1")

    def test_reoffer_comes_from_the_ledger_not_the_wire(self, tmp_path):
        # Supersedes s180 test_deferred_skill_reoffered_then_applied: round 2
        # fetches only the caught-up confirming leg (no growing-delta
        # re-fetch), never re-prompts, and the apply comes from the panel
        # approval, not a wire re-offer.
        eng, feed, store, ledger = asker(tmp_path)
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(make_signer("origin-a").public_key()))
        deny = CountingApproval(False)
        p1 = CountingPeer(o_feed, "dev-a")
        first = eng.run_round("dev-a", p1, approval_fn=deny)
        assert first.deferred == 1 and len(deny.calls) == 1

        deny2 = CountingApproval(False)
        p2 = CountingPeer(o_feed, "dev-a")
        second = eng.run_round("dev-a", p2, approval_fn=deny2)
        assert second.deferred == 0 and second.applied == 0
        assert p2.fetch_calls == 1  # one caught-up leg, nothing re-fetched
        assert len(deny2.calls) == 0  # dedup-and-silence: never re-prompted
        assert ledger.count() == 1

        out = eng.approve_deferred("skill", "s1")
        assert out["approved"] is True and out["applied"] == 1
        assert ledger.count() == 0
        assert any(r.record_id == "s1" for r in feed.current_records())

    def test_rearrival_via_backstop_dedups_silently(self, tmp_path):
        # CHF-01: an impossible watermark serves the full set; the pending
        # skill re-arrives and dedups, never re-prompting.
        eng, feed, store, ledger = asker(tmp_path)
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(make_signer("origin-a").public_key()))
        eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        store.advance_watermark("dev-a", o_feed.high_water() + 50)  # force the backstop
        gate = CountingApproval(False)
        res = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=gate)
        assert len(gate.calls) == 0
        assert res.deferred == 0
        assert ledger.count() == 1
        e = ledger.get("skill", "s1")
        assert e.last_offered_at >= e.deferred_at  # refreshed by the dedup


# ---------------------------------------------------------------------------
# C. The approval path: the same seam, the CURRENT trust state
# ---------------------------------------------------------------------------


class TestApprovalPath:
    def _defer_signed_skill(self, tmp_path):
        eng, feed, store, ledger = asker(tmp_path)
        o_eng, o_feed, o_signer = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        res = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        assert res.deferred == 1 and res.refused == 0
        return eng, feed, store, ledger

    def test_approve_applies_through_the_seam_and_audits(self, tmp_path):
        eng, feed, store, ledger = self._defer_signed_skill(tmp_path)
        out = eng.approve_deferred("skill", "s1")
        assert out["approved"] is True and out["refused"] is False
        assert out["applied"] == 1 and out["unverified"] == 0  # verified, not grace
        assert ledger.count() == 0
        assert any(r.record_id == "s1" for r in feed.current_records())
        assert "sync_deferred_approved" in audit_actions()

    def test_key_change_demotion_since_deferral_refuses(self, tmp_path):
        # The trust root changed after the record was quarantined: a re-pair
        # carried a DIFFERENT signing key, demoting the origin to pending
        # (S206). The approval re-verifies against the CURRENT state and
        # refuses; the entry is removed and nothing applies.
        eng, feed, store, ledger = self._defer_signed_skill(tmp_path)
        eng.register_peer(
            "dev-a", "rk2", signing_pub=b64(make_signer("attacker").public_key())
        )
        assert store.get_peer("dev-a").pending  # demoted
        out = eng.approve_deferred("skill", "s1")
        assert out["approved"] is False and out["refused"] is True
        assert out["applied"] == 0
        assert ledger.count() == 0  # removed, never lingering
        assert all(r.record_id != "s1" for r in feed.current_records())
        assert "sync_deferred_approve_refused" in audit_actions()

    def test_grace_closed_unkeyed_origin_refuses_at_approval(self, tmp_path, monkeypatch):
        # An UNSIGNED record from an unkeyed origin entered under the grace;
        # by approval time the grace is flipped off (the Bloc 4 posture): the
        # re-verification refuses instead of applying.
        eng, feed, store, ledger = asker(tmp_path)
        o_feed = ChangeFeed(root=tmp_path / "raw-a")
        o_feed.record(skill("s1", 1, device="dev-a"))  # journalled unsigned
        store.add_peer("dev-a", "rk")  # no signing key registered
        res = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        assert res.deferred == 1 and res.unverified == 1
        monkeypatch.setattr(
            signing_mod, "ACCEPT_UNSIGNED_FROM_UNKEYED_ORIGINS", False
        )
        out = eng.approve_deferred("skill", "s1")
        assert out["refused"] is True and out["applied"] == 0
        assert ledger.count() == 0

    def test_corrupt_envelope_refuses_failsecure(self, tmp_path):
        eng, feed, store, ledger = self._defer_signed_skill(tmp_path)
        with ledger._lock:
            ledger._conn().execute(
                f"UPDATE {_ledger_mod.TABLE_NAME} SET envelope = ?",
                ("][", ),
            )
            ledger._conn().commit()
        out = eng.approve_deferred("skill", "s1")
        assert out["refused"] is True and out["reason"] == "undecodable"
        assert ledger.count() == 0
        assert all(r.record_id != "s1" for r in feed.current_records())

    def test_approved_record_older_than_local_applies_zero_honestly(self, tmp_path):
        # A local edit out-clocked the pending entry with no wire traffic to
        # sweep it: the approval is honest -- the seam runs, LWW makes it a
        # no-op (applied 0), the entry is gone, the local version stands.
        eng, feed, store, ledger = self._defer_signed_skill(tmp_path)
        feed.record(skill("s1", 9, device="dev-b", body="local-newer"))
        out = eng.approve_deferred("skill", "s1")
        assert out["approved"] is True and out["applied"] == 0
        assert ledger.count() == 0
        current = {r.record_id: r for r in feed.current_records()}
        assert current["s1"].clock == 9

    def test_refuse_removes_applies_nothing_audits(self, tmp_path):
        eng, feed, store, ledger = self._defer_signed_skill(tmp_path)
        entry = eng.refuse_deferred("skill", "s1")
        assert entry.record_id == "s1"
        assert ledger.count() == 0
        assert all(r.record_id != "s1" for r in feed.current_records())
        assert "sync_deferred_refused" in audit_actions()

    def test_unknown_key_raises_deferred_not_found(self, tmp_path):
        eng, _, _, _ = asker(tmp_path)
        with pytest.raises(DeferredNotFound):
            eng.approve_deferred("skill", "ghost")
        with pytest.raises(DeferredNotFound):
            eng.refuse_deferred("skill", "ghost")


# ---------------------------------------------------------------------------
# D. Interactions: epoch reset, unpair cascade, LWW staleness
# ---------------------------------------------------------------------------


class TestInteractions:
    def test_epoch_reset_resync_dedups_the_standing_entry(self, tmp_path):
        # Supersedes s204 test_deferred_hold_is_reoffered_from_zero_after_a_reset:
        # round 1 now ADVANCES; the recreated journal resyncs from 0 and the
        # pending skill dedups into its standing entry -- no re-prompt, the
        # decision survives the reset.
        eng, feed, store, ledger = asker(tmp_path)
        o_signer = make_signer("origin-a")
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a", name="a1")
        o_eng.publish(conv("c1", 1, device="dev-a"))
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        first = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        assert first.deferred == 1
        assert first.new_watermark == o_feed.high_water()  # the flipped premise
        assert ledger.count() == 1

        # The peer's journal is recreated (a new root mints a new epoch).
        o_eng2, o_feed2, _ = origin(tmp_path, device="dev-a", seed="origin-a", name="a2")
        o_eng2.publish(conv("c1", 2, device="dev-a"))
        o_eng2.publish(skill("s1", 1, device="dev-a"))  # the same pending version
        gate = CountingApproval(False)
        res = eng.run_round("dev-a", CountingPeer(o_feed2, "dev-a"), approval_fn=gate)
        assert res.epoch_reset is True
        assert len(gate.calls) == 0  # dedup-and-silence across the reset
        assert res.deferred == 0
        assert ledger.count() == 1  # the entry STANDS
        ids = {r.record_id: r.clock for r in feed.current_records()}
        assert ids.get("c1") == 2 and "s1" not in ids

    def test_unpair_cascades_the_ledger_and_audits(self, tmp_path):
        eng, feed, store, ledger = asker(tmp_path)
        o_signer = make_signer("origin-a")
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        assert ledger.count() == 1
        assert eng.unregister_peer("dev-a") is True
        assert ledger.count() == 0
        actions = audit_actions()
        assert "peer_remove" in actions and "sync_deferred_unpair_cascade" in actions

    def test_stale_incoming_is_skipped_not_prompted_not_ledgered(self, tmp_path):
        # The local set already holds a newer version: the deferred candidate
        # is a dead decision -- skipped and audited, the gate never prompts.
        eng, feed, store, ledger = asker(tmp_path)
        o_signer = make_signer("origin-a")
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        feed.record(skill("s1", 5, device="dev-b", body="local-v5"))
        gate = CountingApproval(False)
        res = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=gate)
        assert res.deferred == 0 and len(gate.calls) == 0
        assert ledger.count() == 0
        assert "sync_deferred_stale" in audit_actions()

    def test_wire_observation_sweeps_a_dead_entry(self, tmp_path):
        # An entry quarantined at clock 1; a local edit out-clocks it; the
        # next wire arrival of the same stale version proves it superseded and
        # sweeps the entry (audited), instead of leaving a dead decision.
        eng, feed, store, ledger = asker(tmp_path)
        o_signer = make_signer("origin-a")
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        assert ledger.count() == 1
        feed.record(skill("s1", 5, device="dev-b", body="local-v5"))
        store.advance_watermark("dev-a", o_feed.high_water() + 50)  # backstop re-serve
        res = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        assert res.deferred == 0
        assert ledger.count() == 0
        assert "sync_deferred_superseded" in audit_actions()

    def test_newer_wire_version_replaces_the_entry_without_prompting(self, tmp_path):
        eng, feed, store, ledger = asker(tmp_path)
        o_signer = make_signer("origin-a")
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a", body="v1"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        o_eng.publish(skill("s1", 2, device="dev-a", body="v2"))
        gate = CountingApproval(False)
        res = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=gate)
        assert len(gate.calls) == 0  # the panel decides, not a modal storm
        assert res.deferred == 1  # a replacement is a fresh pending decision
        assert ledger.get("skill", "s1").clock == 2
        assert ledger.count() == 1

    def test_refused_records_never_enter_the_ledger(self, tmp_path):
        # Decision 3: a forgery is refused-and-counted, never one click from
        # application. A record signed with the WRONG key for its origin.
        eng, feed, store, ledger = asker(tmp_path)
        o_signer = make_signer("origin-a")
        forger = make_signer("mallory")
        o_feed = ChangeFeed(root=tmp_path / "forged")
        o_feed.record(attach_signature(skill("s1", 1, device="dev-a"), forger))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        gate = CountingApproval(True)
        res = eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=gate)
        assert res.refused == 1 and res.deferred == 0
        assert len(gate.calls) == 0  # never prompted for a refused record
        assert ledger.count() == 0
        assert res.advanced is True  # a forgery does not pin convergence

    def test_pending_origin_relay_refused_not_ledgered(self, tmp_path):
        # The S206 inheritance: a record relayed from a PENDING origin refuses
        # outright and stays out of the ledger.
        eng, feed, store, ledger = asker(tmp_path)
        a_signer = make_signer("origin-a")
        b_signer = make_signer("server-b")
        b_feed = ChangeFeed(root=tmp_path / "b")
        b_feed.record(attach_signature(skill("s1", 1, device="dev-a"), a_signer))
        store.add_peer("dev-a", "rk-a", signing_pub=b64(a_signer.public_key()), pending=True)
        store.add_peer("dev-b", "rk-b", signing_pub=b64(b_signer.public_key()))
        res = eng.run_round("dev-b", CountingPeer(b_feed, "dev-b"), approval_fn=lambda c, l, a: True)
        assert res.refused == 1 and res.deferred == 0
        assert ledger.count() == 0


# ---------------------------------------------------------------------------
# E. Mode posture: the ledger is local-disk, mode-free; the wire stays gated
# ---------------------------------------------------------------------------


class TestModePosture:
    def test_apply_seam_split_wire_gated_local_not(self, tmp_path):
        # The protocol split behind decision 4's posture: apply_record_batch
        # (the WIRE apply) refuses under Bulbe; apply_local_batch (the
        # approval seam's local-disk apply) is the same merge core, ungated.
        from opti_oignon.veilid import protocol as proto

        feed = ChangeFeed(root=tmp_path / "feed")
        r = conv("c1", 1, device="dev-a")
        batch = proto.RecordBatch(
            device="dev-a", high_water=0, records=[r], rejected=0, epoch=None
        )
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            proto.apply_record_batch(feed, batch)
        out = proto.apply_local_batch(feed, batch)
        assert out.applied == 1
        assert any(x.record_id == "c1" for x in feed.current_records())

    def test_local_apply_shares_the_merge_core(self, tmp_path):
        # Idempotence and LWW are the same core: a second local apply adopts
        # nothing, and an older version loses.
        from opti_oignon.veilid import protocol as proto

        feed = ChangeFeed(root=tmp_path / "feed")
        newer = skill("s1", 3, body="v3")
        older = skill("s1", 1, body="v1")
        mk = lambda rec_: proto.RecordBatch(
            device=rec_.device, high_water=0, records=[rec_], rejected=0, epoch=None
        )
        assert proto.apply_local_batch(feed, mk(newer)).applied == 1
        assert proto.apply_local_batch(feed, mk(newer)).applied == 0  # idempotent
        assert proto.apply_local_batch(feed, mk(older)).applied == 0  # LWW lose
        current = {r.record_id: r for r in feed.current_records()}
        assert current["s1"].clock == 3

    def test_ledger_surface_works_in_bulbe(self, tmp_path):
        eng, feed, store, ledger = asker(tmp_path)
        o_signer = make_signer("origin-a")
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        o_eng.publish(skill("s2", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        assert ledger.count() == 2

        set_mode("bulbe")
        assert len(eng.list_deferred()) == 2  # read: any mode
        out = eng.approve_deferred("skill", "s1")  # local decision: any mode
        assert out["approved"] is True and out["applied"] == 1
        entry = eng.refuse_deferred("skill", "s2")
        assert entry.record_id == "s2"
        assert ledger.count() == 0
        with pytest.raises(VeilidDisabledInBulbe):
            eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"))  # the wire stays gated


# ---------------------------------------------------------------------------
# F. The route surface: payload helpers and the live FastAPI contract
# ---------------------------------------------------------------------------


class TestRoutePayloads:
    def _engine_with_pending(self, tmp_path):
        eng, feed, store, ledger = asker(tmp_path)
        o_signer = make_signer("origin-a")
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)
        return eng, ledger

    def test_list_payload_is_provenance_only(self, tmp_path):
        eng, _ = self._engine_with_pending(tmp_path)
        payload = rs.deferred_list_payload(eng)
        assert payload["count"] == 1
        entry = payload["deferred"][0]
        assert entry["kind"] == "skill" and entry["record_id"] == "s1"
        assert entry["origin_device"] == "dev-a" and entry["peer_id"] == "dev-a"
        assert entry["clock"] == 1
        assert entry["deferred_at"] and entry["last_offered_at"]
        # The record BODY never reaches the panel.
        assert "payload" not in entry and "envelope" not in entry
        assert "body" not in json.dumps(payload)

    def test_approve_and_refuse_payloads_propagate(self, tmp_path):
        eng, ledger = self._engine_with_pending(tmp_path)
        out = rs.approve_deferred_payload(eng, "skill", "s1")
        assert out["approved"] is True
        with pytest.raises(DeferredNotFound):
            rs.refuse_deferred_payload(eng, "skill", "s1")

    def test_refuse_payload_shape(self, tmp_path):
        eng, ledger = self._engine_with_pending(tmp_path)
        out = rs.refuse_deferred_payload(eng, "skill", "s1")
        assert out["removed"] is True and out["record_id"] == "s1"
        assert ledger.count() == 0

    def test_run_payload_reports_advance_on_deferral(self, tmp_path):
        # Supersedes s180_routes_sync TestRunPayload::
        # test_run_passes_through_approval_fn ('advanced is False' after a
        # deferral): the run payload now reports the persisted advance, the
        # deferred count, and the record sits in the ledger.
        eng, feed, store, ledger = asker(tmp_path)
        o_signer = make_signer("origin-a")
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        out = rs.run_sync_payload(
            eng, "dev-a", CountingPeer(o_feed, "dev-a"),
            approval_fn=lambda c, t, a: False,
        )
        assert out["deferred"] == 1
        assert out["advanced"] is True
        assert out["new_watermark"] == o_feed.high_water()
        assert ledger.has("skill", "s1")


class TestLiveDeferredRoutes:
    def setup_method(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")

    def _client(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        feed = ChangeFeed(root=tmp_path / "local")
        store = PeerStore(root=tmp_path / "store")
        ledger = DeferredLedger(root=tmp_path / "ledger")
        eng = SyncEngine(
            device="dev-b", feed=feed, store=store,
            signer=make_signer("router-b"), ledger=ledger,
        )
        _peers_mod.set_peer_store(store)
        _engine_mod.set_sync_engine(eng)
        _ledger_mod.set_deferred_ledger(ledger)
        app = FastAPI()
        rs.register(app)
        return TestClient(app), eng, feed, store, ledger

    def _seed_pending(self, tmp_path, eng, store):
        o_signer = make_signer("origin-a")
        o_eng, o_feed, _ = origin(tmp_path, device="dev-a", seed="origin-a")
        o_eng.publish(skill("s1", 1, device="dev-a"))
        store.add_peer("dev-a", "rk", signing_pub=b64(o_signer.public_key()))
        eng.run_round("dev-a", CountingPeer(o_feed, "dev-a"), approval_fn=lambda c, l, a: False)

    def test_list_approve_refuse_over_the_routes(self, tmp_path):
        client, eng, feed, store, ledger = self._client(tmp_path)
        self._seed_pending(tmp_path, eng, store)
        r = client.get("/api/sync/deferred")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1 and body["deferred"][0]["record_id"] == "s1"
        r = client.post(
            "/api/sync/deferred/approve",
            json={"kind": "skill", "record_id": "s1"},
        )
        assert r.status_code == 200
        out = r.json()
        assert out["approved"] is True and out["applied"] == 1
        assert client.get("/api/sync/deferred").json()["count"] == 0
        assert any(rec_.record_id == "s1" for rec_ in feed.current_records())

    def test_refuse_route_and_404_and_400(self, tmp_path):
        client, eng, feed, store, ledger = self._client(tmp_path)
        self._seed_pending(tmp_path, eng, store)
        r = client.post(
            "/api/sync/deferred/refuse",
            json={"kind": "skill", "record_id": "s1"},
        )
        assert r.status_code == 200 and r.json()["removed"] is True
        assert ledger.count() == 0
        assert all(rec_.record_id != "s1" for rec_ in feed.current_records())
        # Unknown key -> 404; missing fields -> 400.
        assert client.post(
            "/api/sync/deferred/refuse", json={"kind": "skill", "record_id": "s1"}
        ).status_code == 404
        assert client.post(
            "/api/sync/deferred/approve", json={"kind": "skill", "record_id": "ghost"}
        ).status_code == 404
        assert client.post("/api/sync/deferred/approve", json={}).status_code == 400
        assert client.post(
            "/api/sync/deferred/refuse", json={"kind": "skill"}
        ).status_code == 400

    def test_approve_refused_outcome_is_a_200_with_honest_flags(self, tmp_path):
        # A trust change since deferral is not an HTTP error: the route
        # answers 200 with refused true and applied 0, the panel's honest
        # signal; the entry is gone.
        client, eng, feed, store, ledger = self._client(tmp_path)
        self._seed_pending(tmp_path, eng, store)
        eng.register_peer(
            "dev-a", "rk2", signing_pub=b64(make_signer("attacker").public_key())
        )
        r = client.post(
            "/api/sync/deferred/approve",
            json={"kind": "skill", "record_id": "s1"},
        )
        assert r.status_code == 200
        out = r.json()
        assert out["refused"] is True and out["applied"] == 0
        assert ledger.count() == 0

    def test_routes_work_in_bulbe(self, tmp_path):
        client, eng, feed, store, ledger = self._client(tmp_path)
        self._seed_pending(tmp_path, eng, store)
        set_mode("bulbe")
        assert client.get("/api/sync/deferred").status_code == 200
        r = client.post(
            "/api/sync/deferred/approve",
            json={"kind": "skill", "record_id": "s1"},
        )
        assert r.status_code == 200 and r.json()["approved"] is True


# ---------------------------------------------------------------------------
# G. Source assertions: the panel, the API mirror, the spec registry
# ---------------------------------------------------------------------------


PANEL = ROOT / "frontend" / "src" / "lib" / "components" / "panels" / "SyncPanel.svelte"
SYNC_TS = ROOT / "frontend" / "src" / "lib" / "api" / "sync.ts"
SPEC = ROOT / "VEILID_SPEC.md"


class TestSources:
    def test_panel_has_the_two_distinct_waiting_lists(self):
        src = PANEL.read_text()
        assert "Awaiting confirmation" in src  # device trust (PAIR-02)
        assert "Pending record approvals" in src  # content approval (SYN-05)
        for needle in (
            "listDeferredRecords",
            "approveDeferredRecord",
            "refuseDeferredRecord",
            "deferredRecords",
            "origin_device",
        ):
            assert needle in src
        # Provenance is shown; the record body never is.
        assert "d.payload" not in src and "envelope" not in src

    def test_panel_token_hygiene_and_balance(self):
        src = PANEL.read_text()
        style = src[src.find("<style"):]
        for h in re.findall(r"#[0-9a-fA-F]{3,8}\b", style):
            assert re.search(r"var\(--oo-[^)]*" + re.escape(h), style), h
        for kind in ("if", "each"):
            assert len(re.findall(r"\{#" + kind + r"\b", src)) == len(
                re.findall(r"\{/" + kind + r"\}", src)
            )

    def test_sync_ts_mirrors_the_contract(self):
        src = SYNC_TS.read_text()
        for needle in (
            "interface DeferredRecord",
            "interface DeferredApproveResult",
            "listDeferredRecords",
            "approveDeferredRecord",
            "refuseDeferredRecord",
            "/deferred",
            "record_id",
        ):
            assert needle in src

    def test_spec_registers_the_module_and_the_semantics(self):
        spec = SPEC.read_text()
        assert "opti_oignon/veilid/deferred_ledger.py" in spec
        assert "SYN-05" in spec
        assert "deferred ledger" in spec.lower()
        # The superseded watermark rule is annotated, not erased.
        assert "superseded by S207" in spec

    def test_ledger_module_sql_hygiene_source(self):
        src = (VEILID / "deferred_ledger.py").read_text()
        assert 'f"' not in src and "f'" not in src  # no f-strings near SQL
        assert "str.format" in src or ".format(" in src
        assert "checkpoint_before_apply = True" in src
        assert "PRAGMA journal_mode=WAL" in src
