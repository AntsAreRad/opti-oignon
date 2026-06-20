#!/usr/bin/env python3
"""Tests for S181 Goal 2 -- the responder (Theme 4 / Veilid Sync).

Covers the serve half of the exchange: the engine's responder
(opti_oignon/veilid/sync_engine.py: serve_request) and the transport responder
bridge (opti_oignon/veilid/transport.py: serve_app_call):

- serve_request answers an inbound delta request with a batch drawn from the local
  feed via the protocol's respond_to_request: the right records, the feed's
  high-water, and the protocol envelope shape.
- An unparseable request gets a benign empty batch (high-water, no records), never
  an over-send or a crash.
- It refuses under Bulbe at the binding-layer gate (and fail-secure when the mode
  is undeterminable); a sensitive served record is still just data on the encode
  side -- a skill in the feed is served like any record (applying it is the gated
  action on the receiving side, not serving it).
- The served answer is recorded in the hash-chain audit log.
- serve_app_call bridges bytes in / bytes out around serve_request: the reply
  decodes to the same batch; a malformed request byte-string still yields a benign
  empty batch; it refuses under Bulbe.
- A bidirectional exchange: device A's live peer fetch is answered by device B's
  responder over the same feed, and A converges on B's data -- both pull and serve.

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode is
driven through a stubbed opti_oignon.security_mode and the audit log is captured
through a recording stub.
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
    # Ensure the audit module exists, then always bind the recording chain_log:
    # another test file may have registered a no-op chain_log first, and the engine
    # re-imports chain_log per call, so the current attribute is what gets used.
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
transport = _load("transport")

RecordKind = records.RecordKind
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    # Re-bind the recorder each test: another file's fixture never touches it, but
    # this keeps the audit assertion independent of test-file import order.
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


def _rec(record_id, clock, *, device, payload=None, kind=None):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
    )


def _engine(tmp_path, device="B", seed=()):
    feed = change_feed.ChangeFeed(root=tmp_path / "feed")
    for r in seed:
        feed.record(r)
    store = peers.PeerStore(root=tmp_path / "store")
    return sync_engine.SyncEngine(device=device, feed=feed, store=store), feed


def _request(device="A", watermark=0):
    return {
        "v": protocol.PROTOCOL_VERSION,
        "type": protocol.MSG_DELTA_REQUEST,
        "device": device,
        "watermark": watermark,
    }


# --- The engine responder ---------------------------------------------------


class TestServeRequest:
    def test_serves_records_since_watermark(self, tmp_path):
        eng, feed = _engine(
            tmp_path, seed=[_rec("c1", 1, device="B"), _rec("c2", 1, device="B")]
        )
        batch = eng.serve_request(_request(watermark=0))
        assert batch["type"] == protocol.MSG_RECORD_BATCH
        assert batch["device"] == "B"
        assert batch["high_water"] == feed.high_water()
        ids = {r["id"] for r in batch["records"]}
        assert ids == {"c1", "c2"}

    def test_serves_only_the_delta(self, tmp_path):
        eng, feed = _engine(tmp_path, seed=[_rec("c1", 1, device="B")])
        hw = feed.high_water()
        eng.publish(_rec("c2", 1, device="B"))
        # ask since the first high-water -> only c2 comes back
        batch = eng.serve_request(_request(watermark=hw))
        ids = {r["id"] for r in batch["records"]}
        assert ids == {"c2"}

    def test_unparseable_request_yields_empty_batch(self, tmp_path):
        eng, feed = _engine(tmp_path, seed=[_rec("c1", 1, device="B")])
        batch = eng.serve_request({"not": "a request"})
        assert batch["type"] == protocol.MSG_RECORD_BATCH
        assert batch["records"] == []
        assert batch["high_water"] == feed.high_water()

    def test_serves_a_skill_record_as_data(self, tmp_path):
        # Serving a skill is just encoding data; applying it is the gated action.
        eng, feed = _engine(
            tmp_path, seed=[_rec("s1", 1, device="B", kind=RecordKind.SKILL)]
        )
        batch = eng.serve_request(_request(watermark=0))
        kinds = {r["kind"] for r in batch["records"]}
        assert "skill" in kinds

    def test_refuses_in_bulbe(self, tmp_path):
        eng, feed = _engine(tmp_path, seed=[_rec("c1", 1, device="B")])
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            eng.serve_request(_request())

    def test_refuses_when_mode_undeterminable(self, tmp_path):
        eng, feed = _engine(tmp_path, seed=[_rec("c1", 1, device="B")])
        set_mode(raises=True)
        with pytest.raises(VeilidDisabledInBulbe):
            eng.serve_request(_request())

    def test_served_answer_is_audited(self, tmp_path):
        eng, feed = _engine(tmp_path, seed=[_rec("c1", 1, device="B")])
        eng.serve_request(_request(), peer_id="A")
        served = [e for e in _AUDIT["events"] if e.get("action") == "sync_serve"]
        assert served, "the served answer must be audited"
        assert served[-1]["peer_id"] == "A"
        assert served[-1]["records"] == 1


# --- The transport responder bridge ----------------------------------------


class TestServeAppCall:
    def test_bridge_round_trips_bytes(self, tmp_path):
        eng, feed = _engine(tmp_path, seed=[_rec("c1", 1, device="B")])
        req_bytes = transport._encode_message(_request())
        reply = transport.serve_app_call(eng, req_bytes)
        assert isinstance(reply, bytes)
        batch = transport.decode_answer(reply)
        assert batch["type"] == protocol.MSG_RECORD_BATCH
        assert {r["id"] for r in batch["records"]} == {"c1"}

    def test_bridge_accepts_a_dict_request(self, tmp_path):
        eng, feed = _engine(tmp_path, seed=[_rec("c1", 1, device="B")])
        reply = transport.serve_app_call(eng, _request())
        batch = transport.decode_answer(reply)
        assert {r["id"] for r in batch["records"]} == {"c1"}

    def test_bridge_malformed_request_is_empty_batch(self, tmp_path):
        eng, feed = _engine(tmp_path, seed=[_rec("c1", 1, device="B")])
        reply = transport.serve_app_call(eng, b"garbage")
        batch = transport.decode_answer(reply)
        assert batch["records"] == []
        assert batch["high_water"] == feed.high_water()

    def test_bridge_refuses_in_bulbe(self, tmp_path):
        eng, feed = _engine(tmp_path, seed=[_rec("c1", 1, device="B")])
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            transport.serve_app_call(eng, transport._encode_message(_request()))


# --- A bidirectional exchange (pull answered by serve) ----------------------


class TestBidirectional:
    def test_a_pulls_what_b_serves(self, tmp_path):
        # Device B holds data and serves; device A pulls over a live peer whose
        # messenger is wired to B's responder bridge -- both sides use the real code.
        eng_b, feed_b = _engine(
            tmp_path / "b",
            device="B",
            seed=[_rec("c1", 1, device="B"), _rec("s1", 1, device="B", kind=RecordKind.SKILL)],
        )

        class ResponderMessenger:
            def call(self, routing_key, payload, *, timeout=None):
                return transport.serve_app_call(eng_b, payload, peer_id="A")

        feed_a = change_feed.ChangeFeed(root=tmp_path / "a-feed")
        store_a = peers.PeerStore(root=tmp_path / "a-store")
        eng_a = sync_engine.SyncEngine(device="A", feed=feed_a, store=store_a)
        store_a.add_peer("B", "RKB")
        peer = transport.VeilidPeer(ResponderMessenger(), "RKB", device="A")

        res = eng_a.run_round("B", peer, approval_fn=lambda c, t, a: True)
        assert res.applied == 2
        kinds = {(r.kind.value, r.record_id) for r in feed_a.current_records()}
        assert kinds == {("conversation", "c1"), ("skill", "s1")}

    def test_exchange_refuses_when_b_is_in_bulbe(self, tmp_path):
        # If the serving side is under Bulbe, its responder refuses; the error
        # surfaces through the messenger to the pulling side.
        eng_b, feed_b = _engine(
            tmp_path / "b", device="B", seed=[_rec("c1", 1, device="B")]
        )

        class ResponderMessenger:
            def call(self, routing_key, payload, *, timeout=None):
                return transport.serve_app_call(eng_b, payload)

        feed_a = change_feed.ChangeFeed(root=tmp_path / "a-feed")
        store_a = peers.PeerStore(root=tmp_path / "a-store")
        eng_a = sync_engine.SyncEngine(device="A", feed=feed_a, store=store_a)
        store_a.add_peer("B", "RKB")
        peer = transport.VeilidPeer(ResponderMessenger(), "RKB", device="A")

        # both sides read the same (stubbed) mode; under Bulbe the pull gate trips
        # first on A's side, and the responder would refuse on B's side too.
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            eng_a.run_round("B", peer)
