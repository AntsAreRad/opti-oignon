#!/usr/bin/env python3
"""Tests for S179 Goal 4 -- the protocol envelope (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/protocol.py:

- Building and parsing the two messages: a delta request (a watermark) and a record
  batch (records since a watermark plus a high-water). The parsers are defensive --
  a non-dict, a wrong version or type, a bad device, or a bad watermark / high-water
  is rejected as None and never raises; a batch with a bad wire record keeps the
  good ones and counts the rejected.
- The responder: a valid request is answered from the feed since its watermark; a
  malformed request gets a benign empty batch.
- Applying a batch: it reconciles into the local set, journals only what changed,
  surfaces the conflict log, passes through the reject count, and is idempotent (a
  second apply of the same batch adopts nothing).
- A full pull round with an injected fake peer, and convergence: after B pulls from
  A and A pulls from B, both feeds hold the same content per key; the watermark
  advances and never re-applies.
- The Bulbe seam: every wire-acting function refuses under Bulbe (and when the mode
  is undeterminable, fail-secure); the pure parsers and the reconciler run anyway.

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
RecordKind = records.RecordKind
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    change_feed.reset_change_feed()
    yield
    change_feed.reset_change_feed()
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


class FakePeer:
    """A peer that answers a request from its own local feed -- no transport."""

    def __init__(self, feed, device):
        self.feed = feed
        self.device = device
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)
        return protocol.respond_to_request(self.feed, request, device=self.device)


# Delta request: build and parse


class TestDeltaRequest:
    def test_build_shape(self):
        req = protocol.build_delta_request(device="B", watermark=7)
        assert req["v"] == protocol.PROTOCOL_VERSION
        assert req["type"] == protocol.MSG_DELTA_REQUEST
        assert req["device"] == "B"
        assert req["watermark"] == 7

    def test_round_trip(self):
        req = protocol.build_delta_request(device="B", watermark=3)
        parsed = protocol.parse_delta_request(req)
        assert parsed is not None
        assert parsed.device == "B"
        assert parsed.watermark == 3

    @pytest.mark.parametrize("over", [{"device": ""}, {"watermark": -1}, {"watermark": True}])
    def test_build_rejects_bad_input(self, over):
        kwargs = {"device": "B", "watermark": 0}
        kwargs.update(over)
        with pytest.raises(ValueError):
            protocol.build_delta_request(**kwargs)

    @pytest.mark.parametrize(
        "obj",
        [
            None,
            "x",
            {"v": 999, "type": "delta_request", "device": "B", "watermark": 0},
            {"v": 1, "type": "record_batch", "device": "B", "watermark": 0},
            {"v": 1, "type": "delta_request", "device": "", "watermark": 0},
            {"v": 1, "type": "delta_request", "device": "B", "watermark": -1},
            {"v": 1, "type": "delta_request", "device": "B", "watermark": True},
        ],
    )
    def test_parse_defensive(self, obj):
        assert protocol.parse_delta_request(obj) is None


# Record batch: build and parse


class TestRecordBatch:
    def test_build_from_feed(self, tmp_path):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A"), _rec("b", 1, device="A")])
        batch = protocol.build_record_batch(feed, device="A", watermark=0)
        assert batch["type"] == protocol.MSG_RECORD_BATCH
        assert batch["high_water"] == feed.high_water()
        ids = {r["id"] for r in batch["records"]}
        assert ids == {"a", "b"}

    def test_round_trip(self, tmp_path):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        batch = protocol.build_record_batch(feed, device="A", watermark=0)
        parsed = protocol.parse_record_batch(batch)
        assert parsed is not None
        assert parsed.device == "A"
        assert parsed.rejected == 0
        assert {r.record_id for r in parsed.records} == {"a"}

    def test_bad_record_kept_separate(self, tmp_path):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        batch = protocol.build_record_batch(feed, device="A", watermark=0)
        batch["records"].append({"v": 999})  # one unparseable wire record
        parsed = protocol.parse_record_batch(batch)
        assert parsed is not None
        assert parsed.rejected == 1
        assert {r.record_id for r in parsed.records} == {"a"}

    @pytest.mark.parametrize(
        "obj",
        [
            None,
            {"v": 999, "type": "record_batch", "device": "A", "high_water": 0, "records": []},
            {"v": 1, "type": "delta_request", "device": "A", "high_water": 0, "records": []},
            {"v": 1, "type": "record_batch", "device": "", "high_water": 0, "records": []},
            {"v": 1, "type": "record_batch", "device": "A", "high_water": -1, "records": []},
            {"v": 1, "type": "record_batch", "device": "A", "high_water": 0, "records": "x"},
        ],
    )
    def test_parse_defensive(self, obj):
        assert protocol.parse_record_batch(obj) is None


# Responder


class TestResponder:
    def test_answers_valid_request(self, tmp_path):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        req = protocol.build_delta_request(device="B", watermark=0)
        batch = protocol.respond_to_request(feed, req, device="A")
        assert {r["id"] for r in batch["records"]} == {"a"}

    def test_malformed_request_gets_empty_batch(self, tmp_path):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        batch = protocol.respond_to_request(feed, {"garbage": True}, device="A")
        assert batch["records"] == []
        assert batch["high_water"] == feed.high_water()


# Applying a batch


class TestApply:
    def test_applies_and_journals(self, tmp_path):
        local = _feed(tmp_path, "B", [_rec("b", 1, device="B")])
        incoming = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        batch = protocol.build_record_batch(incoming, device="A", watermark=0)
        result = protocol.apply_record_batch(local, batch)
        assert result.applied == 1
        assert result.new_watermark == incoming.high_water()
        assert {r.record_id for r in local.current_records()} == {"a", "b"}

    def test_idempotent(self, tmp_path):
        local = _feed(tmp_path, "B")
        incoming = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        batch = protocol.build_record_batch(incoming, device="A", watermark=0)
        first = protocol.apply_record_batch(local, batch)
        second = protocol.apply_record_batch(local, batch)
        assert first.applied == 1
        assert second.applied == 0

    def test_surfaces_conflict(self, tmp_path):
        # Concurrent divergence: local and incoming at the same clock, different content.
        local = _feed(tmp_path, "B", [_rec("k", 3, device="B", payload={"t": "B"})])
        incoming = _feed(tmp_path, "A", [_rec("k", 3, device="A", payload={"t": "A"})])
        batch = protocol.build_record_batch(incoming, device="A", watermark=0)
        result = protocol.apply_record_batch(local, batch)
        assert len(result.conflicts) == 1
        assert result.conflicts[0].key == ("conversation", "k")

    def test_passes_through_rejects(self, tmp_path):
        local = _feed(tmp_path, "B")
        incoming = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        batch = protocol.build_record_batch(incoming, device="A", watermark=0)
        batch["records"].append({"v": 999})
        result = protocol.apply_record_batch(local, batch)
        assert result.rejected == 1
        assert result.applied == 1

    def test_unparseable_batch_is_empty_result(self, tmp_path):
        local = _feed(tmp_path, "B")
        result = protocol.apply_record_batch(local, {"garbage": True})
        assert result.converged == []
        assert result.applied == 0
        assert result.new_watermark == 0


# Full round with a fake peer, and convergence


class TestSyncRound:
    def test_pull_round_converges(self, tmp_path):
        feed_a = _feed(
            tmp_path,
            "A",
            [
                _rec("a", 1, device="A", payload={"t": "a-A"}),
                _rec("shared", 2, device="A", payload={"t": "shared-A"}),
            ],
        )
        feed_b = _feed(
            tmp_path,
            "B",
            [
                _rec("b", 1, device="B", payload={"t": "b-B"}),
                _rec("shared", 1, device="B", payload={"t": "shared-B"}),
            ],
        )
        peer_a = FakePeer(feed_a, "A")
        peer_b = FakePeer(feed_b, "B")

        # B pulls from A.
        rb = protocol.sync_with_peer(feed_b, peer_a, device="B", watermark=0)
        assert rb.new_watermark == feed_a.high_water()
        # A pulls from B.
        protocol.sync_with_peer(feed_a, peer_b, device="A", watermark=0)

        def snapshot(feed):
            return {r.record_id: r.content_hash for r in feed.current_records()}

        a_snap, b_snap = snapshot(feed_a), snapshot(feed_b)
        assert set(a_snap) == {"a", "b", "shared"}
        assert a_snap == b_snap  # converged content per key

    def test_repeated_pull_applies_nothing_new(self, tmp_path):
        feed_a = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        feed_b = _feed(tmp_path, "B")
        peer_a = FakePeer(feed_a, "A")
        first = protocol.sync_with_peer(feed_b, peer_a, device="B", watermark=0)
        second = protocol.sync_with_peer(
            feed_b, peer_a, device="B", watermark=first.new_watermark
        )
        assert first.applied == 1
        assert second.applied == 0
        assert second.new_watermark == first.new_watermark


# Bulbe seam: wire-acting functions refuse; pure helpers run


class TestBulbeSeam:
    def _feed_with_one(self, tmp_path):
        return _feed(tmp_path, "A", [_rec("a", 1, device="A")])

    def test_build_delta_request_refused(self, tmp_path):
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            protocol.build_delta_request(device="B", watermark=0)

    def test_build_record_batch_refused(self, tmp_path):
        feed = self._feed_with_one(tmp_path)
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            protocol.build_record_batch(feed, device="A", watermark=0)

    def test_respond_refused(self, tmp_path):
        feed = self._feed_with_one(tmp_path)
        req = protocol.build_delta_request(device="B", watermark=0)
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            protocol.respond_to_request(feed, req, device="A")

    def test_apply_refused(self, tmp_path):
        feed = self._feed_with_one(tmp_path)
        batch = protocol.build_record_batch(feed, device="A", watermark=0)
        local = _feed(tmp_path, "B")
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            protocol.apply_record_batch(local, batch)

    def test_sync_with_peer_refused(self, tmp_path):
        feed_a = self._feed_with_one(tmp_path)
        feed_b = _feed(tmp_path, "B")
        peer = FakePeer(feed_a, "A")
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            protocol.sync_with_peer(feed_b, peer, device="B", watermark=0)

    def test_undeterminable_mode_is_fail_secure(self, tmp_path):
        set_mode(raises=True)
        with pytest.raises(VeilidDisabledInBulbe):
            protocol.build_delta_request(device="B", watermark=0)

    def test_parsers_and_reconcile_run_under_bulbe(self, tmp_path):
        # Build artefacts under Daily, then switch to Bulbe and parse / reconcile.
        feed = self._feed_with_one(tmp_path)
        req = protocol.build_delta_request(device="B", watermark=0)
        batch = protocol.build_record_batch(feed, device="A", watermark=0)
        set_mode("bulbe")
        assert protocol.parse_delta_request(req) is not None
        parsed = protocol.parse_record_batch(batch)
        assert parsed is not None
        merged = reconcile.reconcile([], parsed.records)
        assert {r.record_id for r in merged.records} == {"a"}
