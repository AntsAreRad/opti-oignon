#!/usr/bin/env python3
"""S196 F9a -- functional audit fixes for records + protocol (serialization/envelope).

One tight test group per fix:

- PRT-01: ``respond_to_request`` answers an unparseable request with a benign
  empty batch advertising high-water 0 (never the feed's real high-water), so a
  defensive answer can never advance the asker's watermark past unseen deltas.
  Re-asserts the superseded s179/s181 malformed-request tests under the new
  contract (deselect-plus-reassert).
- PRT-02: ``apply_record_batch`` adopts a winner whose clock advanced past the
  local one even when the content is identical, so the local clock never lags
  and a later local edit is never silently superseded by older content.
- PRT-03: garbage input to ``decode_records`` / ``from_wire_json`` (non-iterable,
  invalid JSON, non-array top level) counts as one rejection instead of zero, so
  garbage-in stays distinguishable from a legitimately empty batch. Re-asserts
  the superseded s179 zero-count tests under the new contract.

Loader idiom: spec_from_file_location with sys.modules registration BEFORE
exec_module (3.12+ dataclass processing), package stubs for the relative-free
absolute imports, no config/db_utils stubs (change_feed's db_utils import is
guarded and every feed here gets an injected tmp root).
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

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily") -> None:
    def _gm() -> str:
        return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


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
        al = types.ModuleType("opti_oignon.signed_audit_log")
        al.chain_log = lambda **kwargs: None  # type: ignore[attr-defined]
        sys.modules["opti_oignon.signed_audit_log"] = al


def _load(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load("guard")
records = _load("records")
reconcile = _load("reconcile")
change_feed = _load("change_feed")
protocol = _load("protocol")
peers = _load("peers")
producers = _load("producers")
sync_engine = _load("sync_engine")
transport = _load("transport")
RecordKind = records.RecordKind


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


def _rec(record_id, clock, *, device, payload=None, kind=None, deleted=False):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
        deleted=deleted,
    )


def _feed(tmp_path, name, seed=()):
    f = change_feed.ChangeFeed(root=tmp_path / name)
    for r in seed:
        f.record(r)
    return f


def _batch(device, high_water, recs):
    return {
        "v": protocol.PROTOCOL_VERSION,
        "type": "record_batch",
        "device": device,
        "high_water": high_water,
        "records": records.encode_records(recs),
    }


# --- PRT-01: benign empty batch advertises high-water 0 ---------------------


class TestPRT01BenignBatchNeverAdvances:
    def test_malformed_request_gets_zero_high_water(self, tmp_path):
        # Re-assertion of the superseded
        # test_s179_protocol::TestResponder::test_malformed_request_gets_empty_batch:
        # the batch is still benign and empty, but it now advertises 0, never the
        # feed's real high-water.
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        assert feed.high_water() > 0
        batch = protocol.respond_to_request(feed, {"garbage": True}, device="A")
        assert batch["records"] == []
        assert batch["high_water"] == 0

    def test_valid_request_keeps_real_high_water(self, tmp_path):
        # The honest path is untouched: a parseable request gets the feed's real
        # high-water with the delta.
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        req = protocol.build_delta_request(device="B", watermark=0)
        batch = protocol.respond_to_request(feed, req, device="A")
        assert batch["high_water"] == feed.high_water()
        assert {r["id"] for r in batch["records"]} == {"a"}

    def test_bridge_malformed_request_zero_high_water(self, tmp_path):
        # Re-assertion of the superseded
        # test_s181_responder::TestServeAppCall::test_bridge_malformed_request_is_empty_batch
        # at the wire bridge.
        feed = _feed(tmp_path, "A", [_rec("c1", 1, device="A")])
        eng = sync_engine.SyncEngine(device="A", feed=feed, store=peers.PeerStore(root=tmp_path / "pa"))
        reply = transport.serve_app_call(eng, b"garbage")
        batch = transport.decode_answer(reply)
        assert batch["records"] == []
        assert batch["high_water"] == 0

    def test_engine_serve_request_malformed_zero_high_water(self, tmp_path):
        # Re-assertion of the superseded
        # test_s181_responder::TestServeRequest::test_unparseable_request_yields_empty_batch
        # at the engine seam: still a benign empty batch, now advertising 0.
        feed = _feed(tmp_path, "A", [_rec("c1", 1, device="A")])
        eng = sync_engine.SyncEngine(device="A", feed=feed, store=peers.PeerStore(root=tmp_path / "pa"))
        batch = eng.serve_request({"not": "a request"})
        assert batch["type"] == protocol.MSG_RECORD_BATCH
        assert batch["records"] == []
        assert batch["high_water"] == 0

    def test_benign_answer_holds_watermark_and_skips_nothing(self, tmp_path):
        # End to end: a round whose answer is the benign batch (the peer could not
        # parse the request) holds the watermark; the next honest round still
        # delivers every record -- nothing was skipped.
        feed_a = _feed(tmp_path, "A")
        store_a = peers.PeerStore(root=tmp_path / "pa")
        eng = sync_engine.SyncEngine(device="A", feed=feed_a, store=store_a)
        eng.register_peer("B", "rk-B")
        feed_b = _feed(tmp_path, "B", [_rec("b1", 1, device="B"), _rec("b2", 2, device="B")])

        class GarblingPeer:
            def fetch(self, request):
                # The request arrives garbled; the responder answers benignly.
                return protocol.respond_to_request(feed_b, {"garbage": 1}, device="B")

        res = eng.run_round("B", GarblingPeer())
        assert res.applied == 0
        assert res.advanced is False
        assert res.new_watermark == 0
        assert store_a.get_watermark("B") == 0

        class HonestPeer:
            def fetch(self, request):
                return protocol.respond_to_request(feed_b, request, device="B")

        res2 = eng.run_round("B", HonestPeer())
        assert res2.applied == 2
        assert {r.record_id for r in feed_a.current_records()} == {"b1", "b2"}
        assert store_a.get_watermark("B") == feed_b.high_water()


# --- PRT-02: a same-content winner at a higher clock is adopted -------------


class TestPRT02ClockOnlyAdoption:
    def test_same_content_higher_clock_is_adopted(self, tmp_path):
        feed = _feed(tmp_path, "B", [_rec("c1", 5, device="B", payload={"body": "X"})])
        remote = _rec("c1", 7, device="A", payload={"body": "X"})  # same content
        res = protocol.apply_record_batch(feed, _batch("A", 1, [remote]))
        assert res.applied == 1
        (latest,) = [r for r in feed.current_records() if r.record_id == "c1"]
        assert latest.clock == 7

    def test_reapply_is_idempotent(self, tmp_path):
        feed = _feed(tmp_path, "B", [_rec("c1", 5, device="B", payload={"body": "X"})])
        batch = _batch("A", 1, [_rec("c1", 7, device="A", payload={"body": "X"})])
        assert protocol.apply_record_batch(feed, batch).applied == 1
        assert protocol.apply_record_batch(feed, batch).applied == 0

    def test_lower_clock_same_content_not_adopted(self, tmp_path):
        feed = _feed(tmp_path, "B", [_rec("c1", 7, device="B", payload={"body": "X"})])
        res = protocol.apply_record_batch(
            feed, _batch("A", 1, [_rec("c1", 5, device="A", payload={"body": "X"})])
        )
        assert res.applied == 0
        (latest,) = [r for r in feed.current_records() if r.record_id == "c1"]
        assert latest.clock == 7

    def test_no_silent_loss_after_revert_chain(self, tmp_path):
        # The regression scenario: A edited X -> Y -> X (clock 5 -> 6 -> 7); B holds
        # (5, X). B adopts A's (7, X), so B's next local edit bumps from 7, not 5,
        # and wins everywhere instead of being silently superseded.
        feed_b = _feed(tmp_path, "B", [_rec("c1", 5, device="B", payload={"body": "X"})])
        protocol.apply_record_batch(
            feed_b, _batch("A", 3, [_rec("c1", 7, device="A", payload={"body": "X"})])
        )
        (latest,) = [r for r in feed_b.current_records() if r.record_id == "c1"]
        edited = _rec("c1", latest.clock + 1, device="B", payload={"body": "Z"})
        feed_b.record(edited)

        merged = reconcile.reconcile(
            [_rec("c1", 7, device="A", payload={"body": "X"})],
            [r for r in feed_b.current_records() if r.record_id == "c1"],
        )
        (winner,) = merged.records
        assert winner.payload == {"body": "Z"}
        assert winner.clock == 8
        assert merged.conflicts == []


# --- PRT-03: garbage input counts as one rejection ---------------------------


class TestPRT03GarbageRejectedCount:
    def test_non_iterable_counts_one_rejection(self):
        # Re-assertion of the superseded
        # test_s179_records::TestBatchDecode::test_non_iterable_yields_empty.
        result = records.decode_records(123)
        assert result.records == []
        assert result.rejected == 1

    def test_bad_json_counts_one_rejection(self):
        # Re-assertion of the superseded
        # test_s179_records::TestBatchDecode::test_bad_json_yields_empty.
        result = records.from_wire_json("{ not json")
        assert result.records == []
        assert result.rejected == 1

    def test_non_array_counts_one_rejection(self):
        result = records.from_wire_json('{"v": 1}')
        assert result.records == []
        assert result.rejected == 1

    def test_empty_list_is_zero_rejections(self):
        # A legitimately empty batch stays distinguishable from garbage.
        result = records.from_wire_json("[]")
        assert result.records == []
        assert result.rejected == 0

    def test_partial_batch_counts_items(self):
        # The per-item path is unchanged: bad items are counted individually.
        good = _rec("a", 1, device="A")
        result = records.decode_records([records.encode_record(good), {"junk": 1}])
        assert result.rejected == 1
        assert [r.record_id for r in result.records] == ["a"]
