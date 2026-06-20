#!/usr/bin/env python3
"""Tests for S203 -- sync cycle Bloc 1 lot 1 (PRT-04 batch chunking + envelope caps).

Covers the wire-format bounds added this lot:

- change_feed.since_page: a bounded page read (count + wire-byte bounds, always one
  row of progress), the page's high-water is the chunk's max sequence, the CHF-01
  backstop served in bounded pages, the empty-page contract, bound validation.
- protocol.build_record_batch / respond_to_request: each answer is one bounded
  chunk advertising the chunk's max sequence; the PRT-01 benign batch still
  advertises 0.
- protocol.parse_record_batch: a defensive envelope cap that REJECTS (never
  truncates) past the count or byte bound; a compliant sender is never rejected by
  the default caps; forward-compatibility with unknown envelope fields.
- sync_engine.run_round: the per-round leg loop -- the watermark advances chunk by
  chunk, a full first sync converges across chunks to the same set as the unchunked
  path, the loop terminates on a static feed and on an empty delta, the
  deferred-in-chunk-k watermark semantics (chunks before k stick, chunk k re-offered),
  idempotent re-application across a chunk-spanning key, chunking composes with a
  compacted journal, and the wire refuses under Bulbe.

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode is
driven through a stubbed opti_oignon.security_mode and the audit log is a recording
stub, the S181/S202 idiom.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"

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
    for name, sub in (("opti_oignon", OO), ("opti_oignon.veilid", VEILID)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod
    if "opti_oignon.security_mode" not in sys.modules:
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.get_current_mode = _MODE["fn"]  # type: ignore[attr-defined]
        sys.modules["opti_oignon.security_mode"] = sm
    if "opti_oignon.signed_audit_log" not in sys.modules:
        sys.modules["opti_oignon.signed_audit_log"] = types.ModuleType(
            "opti_oignon.signed_audit_log"
        )
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]


def _load(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load("guard")
records = _load("records")
reconcile = _load("reconcile")
change_feed = _load("change_feed")
peers = _load("peers")
protocol = _load("protocol")
producers = _load("producers")
sync_engine = _load("sync_engine")

RecordKind = records.RecordKind
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
    _AUDIT["events"].clear()
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    yield
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    set_mode("daily")


def _rec(record_id, clock, *, device="B", payload=None, kind=None):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
    )


def _feed(tmp_path, name="feed", seed=()):
    f = change_feed.ChangeFeed(root=tmp_path / name)
    for r in seed:
        f.record(r)
    return f


def _engine(tmp_path, device="A", seed=()):
    feed = change_feed.ChangeFeed(root=tmp_path / "local_feed")
    for r in seed:
        feed.record(r)
    store = peers.PeerStore(root=tmp_path / "store")
    return sync_engine.SyncEngine(device=device, feed=feed, store=store), feed, store


class FakePeer:
    """A peer that answers from its own feed in bounded chunks (server side)."""

    def __init__(self, feed, device, *, max_count=None, max_bytes=None):
        self.feed = feed
        self.device = device
        self.max_count = (
            max_count if max_count is not None else protocol.SENDER_MAX_RECORDS
        )
        self.max_bytes = (
            max_bytes if max_bytes is not None else protocol.SENDER_MAX_BYTES
        )

    def fetch(self, request):
        return protocol.respond_to_request(
            self.feed,
            request,
            device=self.device,
            max_count=self.max_count,
            max_bytes=self.max_bytes,
        )


# --- since_page: the bounded read seam -------------------------------------


class TestSincePage:
    def test_count_bound_and_chunk_max_seq(self, tmp_path):
        feed = _feed(
            tmp_path,
            seed=[_rec(f"k{i}", 1) for i in range(5)],
        )  # seqs 1..5, distinct keys
        page = feed.since_page(0, max_count=2, max_bytes=1_000_000)
        assert len(page.records) == 2
        # The page's high-water is the chunk's max sequence (the 2nd row), not the
        # feed's overall high-water (5).
        assert page.high_water == 2
        assert feed.high_water() == 5

    def test_byte_bound_keeps_at_least_one(self, tmp_path):
        big = {"blob": "x" * 4000}
        feed = _feed(
            tmp_path,
            seed=[_rec("a", 1, payload=big), _rec("b", 1, payload=big)],
        )
        # A byte budget below one record still yields one row (progress guarantee).
        page = feed.since_page(0, max_count=10, max_bytes=10)
        assert len(page.records) == 1
        assert page.high_water == 1

    def test_byte_bound_stops_mid_page(self, tmp_path):
        big = {"blob": "x" * 4000}
        feed = _feed(
            tmp_path,
            seed=[_rec(f"k{i}", 1, payload=big) for i in range(4)],
        )
        one = feed.since_page(0, max_count=10, max_bytes=10)
        size = change_feed.ChangeFeed._wire_size(one.records[0])
        # Budget for ~2 records.
        page = feed.since_page(0, max_count=10, max_bytes=size * 2 + 10)
        assert len(page.records) == 2

    def test_empty_page_reports_high_water(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        hw = feed.high_water()
        page = feed.since_page(hw, max_count=5, max_bytes=1_000_000)
        assert page.records == []
        assert page.high_water == hw  # the caught-up empty-delta contract

    def test_empty_feed_page(self, tmp_path):
        feed = _feed(tmp_path)
        page = feed.since_page(0, max_count=5, max_bytes=1_000_000)
        assert page.records == []
        assert page.high_water == 0

    def test_backstop_served_in_bounded_pages(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec(f"k{i}", 1) for i in range(5)])
        high = feed.high_water()
        # An impossible watermark reads from the start, bounded.
        page = feed.since_page(high + 100, max_count=2, max_bytes=1_000_000)
        assert len(page.records) == 2
        assert page.high_water == 2  # the chunk's max seq, below the impossible w

    def test_since_is_unchanged_by_paging(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec(f"k{i}", 1) for i in range(3)])
        delta = feed.since(0)
        assert {r.record_id for r in delta.records} == {"k0", "k1", "k2"}
        assert delta.high_water == feed.high_water()  # whole delta, feed high-water

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2", None])
    def test_validates_max_count(self, tmp_path, bad):
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        with pytest.raises(ValueError):
            feed.since_page(0, max_count=bad, max_bytes=1000)

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2", None])
    def test_validates_max_bytes(self, tmp_path, bad):
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        with pytest.raises(ValueError):
            feed.since_page(0, max_count=5, max_bytes=bad)

    def test_corrupt_row_advances_cursor(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("a", 1), _rec("b", 1)])
        # Corrupt the first row's payload so it fails the hash check on read.
        conn = feed._conn()
        conn.execute(
            "UPDATE veilid_change_feed SET payload = ? WHERE seq = 1",
            ('{"tampered":true}',),
        )
        conn.commit()
        page = feed.since_page(0, max_count=1, max_bytes=1_000_000)
        # The corrupt row ships nothing but the page still steps past its sequence.
        assert page.records == []
        assert page.high_water == 1


# --- build_record_batch / respond_to_request: bounded answers --------------


class TestBuildBounded:
    def test_chunk_high_water_is_chunk_max(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec(f"k{i}", 1) for i in range(5)])
        batch = protocol.build_record_batch(
            feed, device="B", watermark=0, max_count=2, max_bytes=1_000_000
        )
        assert batch["high_water"] == 2
        assert len(batch["records"]) == 2

    def test_single_chunk_equals_feed_high_water(self, tmp_path):
        # The degenerate single-chunk case: default bounds cover a tiny feed, so
        # the chunk's max seq equals the feed's high-water (the pre-PRT-04 shape).
        feed = _feed(tmp_path, seed=[_rec("a", 1), _rec("b", 1)])
        batch = protocol.build_record_batch(feed, device="B", watermark=0)
        assert batch["high_water"] == feed.high_water()

    def test_benign_batch_still_zero(self, tmp_path):
        # PRT-01 reasserted under chunking: a malformed request answers 0.
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        batch = protocol.respond_to_request(feed, {"garbage": True}, device="B")
        assert batch["records"] == []
        assert batch["high_water"] == 0


# --- parse_record_batch: the envelope cap ----------------------------------


class TestReceiverCap:
    def _batch(self, recs, high_water=1, device="B", extra=None):
        obj = {
            "v": protocol.PROTOCOL_VERSION,
            "type": protocol.MSG_RECORD_BATCH,
            "device": device,
            "high_water": high_water,
            "records": records.encode_records(recs),
        }
        if extra:
            obj.update(extra)
        return obj

    def test_rejects_oversized_count_without_truncating(self, tmp_path):
        recs = [_rec(f"k{i}", 1) for i in range(5)]
        obj = self._batch(recs, high_water=5)
        # A cap below the count REJECTS (None), never returns a truncated batch.
        assert protocol.parse_record_batch(obj, max_count=3, max_bytes=10_000_000) is None

    def test_rejects_oversized_bytes_without_truncating(self, tmp_path):
        recs = [_rec(f"k{i}", 1, payload={"blob": "x" * 2000}) for i in range(4)]
        obj = self._batch(recs, high_water=4)
        wire = len(
            json.dumps(obj["records"], separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        assert protocol.parse_record_batch(obj, max_count=100, max_bytes=wire - 1) is None

    def test_under_cap_parses(self, tmp_path):
        recs = [_rec(f"k{i}", 1) for i in range(5)]
        obj = self._batch(recs, high_water=5)
        parsed = protocol.parse_record_batch(obj, max_count=10, max_bytes=10_000_000)
        assert parsed is not None
        assert len(parsed.records) == 5

    def test_compliant_sender_never_rejected_by_defaults(self, tmp_path):
        # A full default-bounded chunk parses cleanly under the default receiver caps.
        feed = _feed(tmp_path, seed=[_rec(f"k{i}", 1) for i in range(300)])
        batch = protocol.build_record_batch(feed, device="B", watermark=0)
        assert len(batch["records"]) == protocol.SENDER_MAX_RECORDS
        parsed = protocol.parse_record_batch(batch)  # default receiver caps
        assert parsed is not None
        assert len(parsed.records) == protocol.SENDER_MAX_RECORDS

    def test_receiver_caps_exceed_sender_bound(self):
        assert protocol.RECEIVER_MAX_RECORDS >= protocol.SENDER_MAX_RECORDS
        assert protocol.RECEIVER_MAX_BYTES >= protocol.SENDER_MAX_BYTES

    def test_forward_compat_unknown_fields_both_ways(self, tmp_path):
        # Unknown envelope fields are ignored on both the request and the batch.
        req = protocol.build_delta_request(device="A", watermark=0)
        req["future_field"] = {"unknown": True}
        parsed_req = protocol.parse_delta_request(req)
        assert parsed_req is not None and parsed_req.watermark == 0

        recs = [_rec("a", 1)]
        obj = self._batch(recs, high_water=1, extra={"future_field": 42})
        obj["records"][0]["future_record_field"] = "ignored"
        parsed = protocol.parse_record_batch(obj)
        assert parsed is not None
        assert {r.record_id for r in parsed.records} == {"a"}


# --- run_round: the per-round leg loop -------------------------------------


class TestLegLoop:
    def test_watermark_advances_chunk_by_chunk(self, tmp_path):
        eng, local, store = _engine(tmp_path, device="A")
        remote = _feed(tmp_path, "remote", seed=[_rec(f"k{i}", 1) for i in range(5)])
        store.add_peer("B", "rk-B")
        peer = FakePeer(remote, "B", max_count=2)
        res = eng.run_round("B", peer)
        assert res.applied == 5
        assert res.legs > 1  # several chunks
        assert store.get_watermark("B") == remote.high_water()
        assert {r.record_id for r in local.current_records()} == {
            "k0", "k1", "k2", "k3", "k4"
        }

    def test_full_first_sync_equals_unchunked(self, tmp_path):
        seed = [_rec(f"k{i}", 1, payload={"v": i}) for i in range(7)]
        # Chunked path.
        eng_c, local_c, store_c = _engine(tmp_path, device="A")
        remote_c = _feed(tmp_path, "remote_c", seed=seed)
        store_c.add_peer("B", "rk-B")
        eng_c.run_round("B", FakePeer(remote_c, "B", max_count=2))
        chunked = {r.record_id: r.content_hash for r in local_c.current_records()}
        # Unchunked path (one big chunk).
        eng_u, local_u, store_u = _engine(tmp_path, device="A2")
        remote_u = _feed(tmp_path, "remote_u", seed=seed)
        store_u.add_peer("B", "rk-B")
        eng_u.run_round("B", FakePeer(remote_u, "B", max_count=1000))
        unchunked = {r.record_id: r.content_hash for r in local_u.current_records()}
        assert chunked == unchunked
        assert len(chunked) == 7

    def test_terminates_on_static_feed(self, tmp_path):
        eng, local, store = _engine(tmp_path, device="A")
        remote = _feed(tmp_path, "remote", seed=[_rec(f"k{i}", 1) for i in range(4)])
        store.add_peer("B", "rk-B")
        res = eng.run_round("B", FakePeer(remote, "B", max_count=2))
        # A data round of 4 records over 2-record chunks: two data legs plus one
        # confirming empty leg. Bounded and small, never the MAX_LEGS guard.
        assert res.legs <= 4
        assert res.advanced is True

    def test_terminates_on_empty_delta(self, tmp_path):
        eng, local, store = _engine(tmp_path, device="A")
        remote = _feed(tmp_path, "remote")  # empty
        store.add_peer("B", "rk-B")
        res = eng.run_round("B", FakePeer(remote, "B", max_count=2))
        assert res.applied == 0
        assert res.advanced is False
        assert res.legs == 1

    def test_caught_up_round_is_single_leg(self, tmp_path):
        eng, local, store = _engine(tmp_path, device="A")
        remote = _feed(tmp_path, "remote", seed=[_rec("a", 1)])
        store.add_peer("B", "rk-B")
        eng.run_round("B", FakePeer(remote, "B", max_count=2))  # converge
        res2 = eng.run_round("B", FakePeer(remote, "B", max_count=2))  # already caught up
        assert res2.applied == 0
        assert res2.advanced is False
        assert res2.legs == 1

    def test_idempotent_chunk_spanning_key(self, tmp_path):
        eng, local, store = _engine(tmp_path, device="A")
        # "shared" appears twice, in different 2-record chunks; clock 2 must win.
        seed = [_rec("k1", 1), _rec("shared", 1), _rec("k3", 1), _rec("shared", 2)]
        remote = _feed(tmp_path, "remote", seed=seed)
        store.add_peer("B", "rk-B")
        eng.run_round("B", FakePeer(remote, "B", max_count=2))
        by_id = {r.record_id: r for r in local.current_records()}
        assert by_id["shared"].clock == 2
        # A second round applies nothing (idempotent).
        res2 = eng.run_round("B", FakePeer(remote, "B", max_count=2))
        assert res2.applied == 0

    def test_single_chunk_defer_holds_at_previous(self, tmp_path):
        # The pre-PRT-04 contract under the new loop: a single-chunk round with a
        # deferred sensitive record holds the watermark at previous, advanced False.
        eng, local, store = _engine(tmp_path, device="A")
        remote = _feed(
            tmp_path, "remote", seed=[_rec("s1", 1, kind=RecordKind.SKILL)]
        )
        store.add_peer("B", "rk-B")
        res = eng.run_round(
            "B", FakePeer(remote, "B", max_count=10), approval_fn=lambda c, t, a: False
        )
        assert res.deferred == 1
        assert res.advanced is False
        assert store.get_watermark("B") == 0
        assert local.current_records() == []  # the skill was not applied

    def test_deferred_in_chunk_k_holds_at_boundary(self, tmp_path):
        # A skill sits in chunk 2 (seq 3-4); chunks before it must stick while the
        # deferring chunk is held for re-offer. max_count=2 -> chunk1 = seq1,2;
        # chunk2 = seq3,4 (the skill at seq3).
        eng, local, store = _engine(tmp_path, device="A")
        seed = [
            _rec("c1", 1),
            _rec("c2", 1),
            _rec("s1", 1, kind=RecordKind.SKILL),
            _rec("c3", 1),
        ]
        remote = _feed(tmp_path, "remote", seed=seed)
        store.add_peer("B", "rk-B")
        res = eng.run_round(
            "B", FakePeer(remote, "B", max_count=2), approval_fn=lambda c, t, a: False
        )
        # Chunk 1 (c1, c2) applied and committed; chunk 2 defers only the skill.
        # The non-sensitive record in the deferring chunk (c3) applies immediately;
        # the watermark is held at the pre-chunk boundary so the whole chunk is
        # re-offered next round (c3 re-applies idempotently, the skill is re-prompted).
        assert res.deferred == 1
        assert res.advanced is True  # the persisted watermark moved past chunk 1
        assert store.get_watermark("B") == 2  # held at the chunk-1 boundary
        ids = {r.record_id for r in local.current_records()}
        assert ids == {"c1", "c2", "c3"}  # only the sensitive skill is withheld
        assert "s1" not in ids
        # A second round, now approving, delivers the rest including the skill.
        res2 = eng.run_round(
            "B", FakePeer(remote, "B", max_count=2), approval_fn=lambda c, t, a: True
        )
        assert store.get_watermark("B") == remote.high_water()
        ids2 = {r.record_id for r in local.current_records()}
        assert ids2 == {"c1", "c2", "c3", "s1"}

    def test_backstop_converges_data_watermark_held(self, tmp_path):
        # CHF-01 under chunking: an impossible (corrupt) watermark converges the
        # full set across bounded pages in one round; the persisted watermark stays
        # put (the real repair is CHF-05), so advanced is False.
        eng, local, store = _engine(tmp_path, device="A")
        remote = _feed(tmp_path, "remote", seed=[_rec(f"k{i}", 1) for i in range(5)])
        store.add_peer("B", "rk-B")
        store.advance_watermark("B", 10_000)  # corrupt / beyond the remote high-water
        res = eng.run_round("B", FakePeer(remote, "B", max_count=2))
        assert res.applied == 5
        assert res.legs > 1
        assert store.get_watermark("B") == 10_000  # monotonic max keeps the corrupt w
        assert res.advanced is False
        assert {r.record_id for r in local.current_records()} == {
            "k0", "k1", "k2", "k3", "k4"
        }

    def test_malformed_answer_holds_and_marks_unparsed(self, tmp_path):
        eng, local, store = _engine(tmp_path, device="A")
        store.add_peer("B", "rk-B")

        class BadPeer:
            def fetch(self, request):
                return {"garbage": True}

        res = eng.run_round("B", BadPeer())
        assert res.parsed is False
        assert res.advanced is False
        assert store.get_watermark("B") == 0

    def test_chunking_composes_with_compaction(self, tmp_path):
        seed = [
            _rec("k1", 1),
            _rec("k1", 2),  # supersedes k1@1
            _rec("k2", 1),
            _rec("k3", 1),
        ]
        # Pre-compaction chunked sync.
        eng1, local1, store1 = _engine(tmp_path, device="A")
        remote1 = _feed(tmp_path, "remote1", seed=seed)
        store1.add_peer("B", "rk-B")
        eng1.run_round("B", FakePeer(remote1, "B", max_count=2))
        pre = {r.record_id: r.clock for r in local1.current_records()}
        # Post-compaction chunked sync over the same logical content.
        eng2, local2, store2 = _engine(tmp_path, device="A2")
        remote2 = _feed(tmp_path, "remote2", seed=seed)
        removed = remote2.compact()
        assert removed >= 1  # the superseded k1@1 row is gone
        assert remote2.high_water() == remote1.high_water()  # high-water preserved
        store2.add_peer("B", "rk-B")
        eng2.run_round("B", FakePeer(remote2, "B", max_count=2))
        post = {r.record_id: r.clock for r in local2.current_records()}
        assert pre == post
        assert post["k1"] == 2


# --- Mode posture ----------------------------------------------------------


class TestModePosture:
    def test_wire_refuses_in_bulbe(self, tmp_path):
        eng, local, store = _engine(tmp_path, device="A")
        remote = _feed(tmp_path, "remote", seed=[_rec("a", 1)])
        store.add_peer("B", "rk-B")
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            eng.run_round("B", FakePeer(remote, "B"))
        with pytest.raises(VeilidDisabledInBulbe):
            protocol.build_record_batch(remote, device="B", watermark=0)
        with pytest.raises(VeilidDisabledInBulbe):
            protocol.respond_to_request(remote, {"x": 1}, device="B")

    def test_feed_page_and_parse_are_mode_free(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        obj = {
            "v": protocol.PROTOCOL_VERSION,
            "type": protocol.MSG_RECORD_BATCH,
            "device": "B",
            "high_water": 1,
            "records": records.encode_records([_rec("a", 1)]),
        }
        set_mode("bulbe")
        # since_page is a local-disk read; parse is reading data: both ungated.
        page = feed.since_page(0, max_count=5, max_bytes=1_000_000)
        assert len(page.records) == 1
        assert protocol.parse_record_batch(obj) is not None

    def test_module_has_no_gate_in_change_feed(self):
        # since_page must not reach for the gate: the feed stays mode-free.
        src = (VEILID / "change_feed.py").read_text()
        assert "assert_sync_allowed" not in src
