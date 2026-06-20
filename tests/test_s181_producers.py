#!/usr/bin/env python3
"""Tests for S181 Goal 2 -- the producers (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/producers.py and the engine's publish_* convenience
methods (opti_oignon/veilid/sync_engine.py):

- Each producer builds a valid record of the right kind with a verified content
  hash; the two memory tiers are distinct kinds, so a canonical fact and an archive
  entry with the same identity never collide.
- The encode side is pure and defensive: a None payload becomes an empty mapping, a
  non-mapping raises ValueError (the producer-side contract), and a tombstone is an
  empty payload with deleted=True.
- A produced record round-trips through the wire encoding (encode -> decode equals
  the original) -- the property the protocol relies on.
- The engine's publish_* methods journal a real record locally (in any mode) and a
  pull round carries it to a peer and back: a conversation, a canonical fact, an
  archive entry, and a skill each round-trip from one device's feed to another's.

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
peers = _load("peers")
protocol = _load("protocol")
producers = _load("producers")
sync_engine = _load("sync_engine")

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


class FakePeer:
    def __init__(self, feed, device):
        self.feed = feed
        self.device = device

    def fetch(self, request):
        return protocol.respond_to_request(self.feed, request, device=self.device)


def _engine(tmp_path, device="A"):
    feed = change_feed.ChangeFeed(root=tmp_path / "local")
    store = peers.PeerStore(root=tmp_path / "store")
    return sync_engine.SyncEngine(device=device, feed=feed, store=store), feed, store


def _remote(tmp_path, device="B", seed=()):
    f = change_feed.ChangeFeed(root=tmp_path / "remote")
    for r in seed:
        f.record(r)
    return FakePeer(f, device), f


# --- Each producer builds the right kind -----------------------------------


class TestProducerKinds:
    def test_conversation(self):
        r = producers.conversation_record("c1", {"text": "hi"}, device="A", clock=1)
        assert r.kind == RecordKind.CONVERSATION
        assert r.record_id == "c1"
        assert records.verify_record_hash(r) is True

    def test_memory_canonical(self):
        r = producers.memory_canonical_record("f1", {"fact": "x"}, device="A", clock=2)
        assert r.kind == RecordKind.MEMORY_CANONICAL
        assert records.verify_record_hash(r) is True

    def test_memory_archive(self):
        r = producers.memory_archive_record("a1", {"entry": "y"}, device="A", clock=3)
        assert r.kind == RecordKind.MEMORY_ARCHIVE
        assert records.verify_record_hash(r) is True

    def test_skill(self):
        r = producers.skill_record("s1", {"name": "lint"}, device="A", clock=4)
        assert r.kind == RecordKind.SKILL
        assert records.verify_record_hash(r) is True

    def test_two_memory_tiers_do_not_collide(self):
        canonical = producers.memory_canonical_record("m1", {"k": 1}, device="A", clock=1)
        archive = producers.memory_archive_record("m1", {"k": 1}, device="A", clock=1)
        assert records.key_of(canonical) != records.key_of(archive)


# --- The encode side is pure and defensive ---------------------------------


class TestEncodeDefensive:
    def test_none_payload_is_empty_mapping(self):
        r = producers.conversation_record("c1", None, device="A", clock=1)
        assert dict(r.payload) == {}
        assert records.verify_record_hash(r) is True

    def test_non_mapping_payload_raises(self):
        with pytest.raises(ValueError):
            producers.conversation_record("c1", ["not", "a", "map"], device="A", clock=1)

    def test_tombstone(self):
        r = producers.tombstone_record(RecordKind.SKILL, "s1", device="A", clock=5)
        assert r.deleted is True
        assert dict(r.payload) == {}
        assert r.kind == RecordKind.SKILL
        assert records.verify_record_hash(r) is True

    def test_tombstone_accepts_kind_value(self):
        r = producers.tombstone_record("conversation", "c1", device="A", clock=1)
        assert r.kind == RecordKind.CONVERSATION
        assert r.deleted is True

    def test_producer_validates_clock(self):
        with pytest.raises(ValueError):
            producers.skill_record("s1", {}, device="A", clock=-1)

    def test_producer_validates_device(self):
        with pytest.raises(ValueError):
            producers.skill_record("s1", {}, device="", clock=1)


# --- A produced record round-trips through the wire encoding ----------------


class TestWireRoundTrip:
    @pytest.mark.parametrize(
        "producer,rid,payload",
        [
            ("conversation_record", "c1", {"text": "hello"}),
            ("memory_canonical_record", "f1", {"fact": "the sky"}),
            ("memory_archive_record", "a1", {"entry": "a long note"}),
            ("skill_record", "s1", {"name": "format"}),
        ],
    )
    def test_encode_decode_equals_original(self, producer, rid, payload):
        make = getattr(producers, producer)
        rec = make(rid, payload, device="A", clock=7)
        wire = records.encode_record(rec)
        back = records.decode_record(wire)
        assert back == rec


# --- publish_* journals locally (any mode) ----------------------------------


class TestPublishLocal:
    def test_publish_conversation_journals(self, tmp_path):
        eng, feed, _ = _engine(tmp_path)
        seq = eng.publish_conversation("c1", {"text": "hi"}, clock=1)
        assert isinstance(seq, int) and seq > 0
        snap = {(r.kind.value, r.record_id) for r in feed.current_records()}
        assert ("conversation", "c1") in snap

    def test_publish_all_kinds(self, tmp_path):
        eng, feed, _ = _engine(tmp_path)
        eng.publish_conversation("c1", {"a": 1}, clock=1)
        eng.publish_memory_canonical("f1", {"b": 2}, clock=1)
        eng.publish_memory_archive("a1", {"c": 3}, clock=1)
        eng.publish_skill("s1", {"d": 4}, clock=1)
        snap = {(r.kind.value, r.record_id) for r in feed.current_records()}
        assert snap == {
            ("conversation", "c1"),
            ("memory_canonical", "f1"),
            ("memory_archive", "a1"),
            ("skill", "s1"),
        }

    def test_publish_runs_in_bulbe(self, tmp_path):
        eng, feed, _ = _engine(tmp_path)
        set_mode("bulbe")
        eng.publish_conversation("c1", {"a": 1}, clock=1)
        assert len(eng.local_records()) == 1

    def test_publish_uses_engine_device(self, tmp_path):
        eng, feed, _ = _engine(tmp_path, device="device-A")
        eng.publish_conversation("c1", {"a": 1}, clock=1)
        rec = feed.current_records()[0]
        assert rec.device == "device-A"


# --- A round carries a produced record peer-to-peer (real data) -------------


class TestRoundTripsRealData:
    def test_conversation_round_trips_to_a_peer(self, tmp_path):
        # The remote device produces a conversation; the local device pulls it.
        remote_eng, _, _ = _engine(tmp_path / "r", device="B")
        # build the remote feed via its producer, then expose it as a fake peer
        rfeed = change_feed.ChangeFeed(root=tmp_path / "rfeed")
        rfeed.record(producers.conversation_record("c1", {"text": "remote"}, device="B", clock=1))
        peer = FakePeer(rfeed, "B")

        eng, feed, store = _engine(tmp_path / "l", device="A")
        store.add_peer("B", "RKB")
        res = eng.run_round("B", peer)
        assert res.applied == 1
        got = {r.record_id: r for r in feed.current_records()}["c1"]
        assert got.kind == RecordKind.CONVERSATION
        assert dict(got.payload) == {"text": "remote"}

    def test_memory_and_skill_round_trip(self, tmp_path):
        rfeed = change_feed.ChangeFeed(root=tmp_path / "rfeed")
        rfeed.record(producers.memory_canonical_record("f1", {"fact": "x"}, device="B", clock=1))
        rfeed.record(producers.memory_archive_record("a1", {"entry": "y"}, device="B", clock=1))
        rfeed.record(producers.skill_record("s1", {"name": "lint"}, device="B", clock=1))
        peer = FakePeer(rfeed, "B")

        eng, feed, store = _engine(tmp_path / "l", device="A")
        store.add_peer("B", "RKB")
        # the skill is sensitive; approve it so the whole batch applies
        res = eng.run_round("B", peer, approval_fn=lambda c, t, a: True)
        assert res.applied == 3
        kinds = {(r.kind.value, r.record_id) for r in feed.current_records()}
        assert kinds == {
            ("memory_canonical", "f1"),
            ("memory_archive", "a1"),
            ("skill", "s1"),
        }
