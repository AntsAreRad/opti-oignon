#!/usr/bin/env python3
"""Tests for S180 Goal 2 -- the web-free sync engine (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/sync_engine.py:

- A round: a pull against a paired peer applies the peer's records into the local
  feed, advances the per-peer watermark to the peer's high-water, and reports the
  applied count; a second identical round adopts nothing (idempotent) and does not
  re-advance; an empty or unparseable answer leaves the watermark untouched.
- The conflict pass-through: a concurrent divergence between the local set and the
  incoming batch is retained and surfaced in the round summary, and the converged
  set holds the tie-break winner.
- The Bulbe seam: a round refuses under Bulbe (and fail-secure when the mode is
  undeterminable) at the binding-layer gate, before it touches the wire and before
  the peer-not-found check; peer management and local production run in any mode.
- Peer-not-found: a round against an unpaired peer raises PeerNotFound.
- The approval discipline: applying a skill is sensitive and passes the gate; a
  denied skill is deferred (not applied) and the watermark is held so it is
  re-offered; a non-sensitive record applies without a gate; the default gate is
  the manager-backed allowlists.request_approval, fail-secure.
- The producer and local production; the pure result-shaping helper; the singleton.

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode is
driven through a stubbed opti_oignon.security_mode and the audit log is a no-op.
The default approval gate (opti_oignon.agent.allowlists) is installed per-test via
monkeypatch so it never pollutes the shared sys.modules. The peer is a fake
answering from its own local feed -- no live transport.
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
# A controllable default approval gate (opti_oignon.agent.allowlists stub).
_APPROVAL = {"value": True, "calls": []}


def set_mode(value: str = "daily", *, raises: bool = False) -> None:
    if raises:
        def _gm() -> str:
            raise RuntimeError("mode undeterminable")
    else:
        def _gm() -> str:
            return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def set_default_approval(value: bool) -> None:
    _APPROVAL["value"] = value


def _default_request_approval(conversation_id, tool_name, arguments=None, *, manager=None, timeout=None):
    _APPROVAL["calls"].append((conversation_id, tool_name, dict(arguments or {}), manager))
    return bool(_APPROVAL["value"])


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


def _allowlists_stub() -> types.ModuleType:
    """A controllable opti_oignon.agent.allowlists, for the default-gate tests.

    Installed per-test via monkeypatch.setitem so it never pollutes the shared
    sys.modules for the real agent suite; the engine binds request_approval from
    the fully-qualified submodule name, so this entry is authoritative.
    """
    mod = types.ModuleType("opti_oignon.agent.allowlists")
    mod.request_approval = _default_request_approval  # type: ignore[attr-defined]
    return mod


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
sync_engine = _load("sync_engine")
RecordKind = records.RecordKind
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe
PeerNotFound = sync_engine.PeerNotFound


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    set_default_approval(True)
    _APPROVAL["calls"].clear()
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


class FakePeer:
    """A peer that answers a request from its own local feed -- no transport."""

    def __init__(self, feed, device):
        self.feed = feed
        self.device = device
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)
        return protocol.respond_to_request(self.feed, request, device=self.device)


class RawPeer:
    """A peer that returns a fixed raw object (for the unparseable-answer path)."""

    def __init__(self, value):
        self.value = value
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)
        return self.value


def _engine(tmp_path, *, device="A"):
    feed = change_feed.ChangeFeed(root=tmp_path / "local")
    store = peers.PeerStore(root=tmp_path / "store")
    eng = sync_engine.SyncEngine(device=device, feed=feed, store=store)
    return eng, feed, store


def _remote(tmp_path, device="B", seed=()):
    f = change_feed.ChangeFeed(root=tmp_path / "remote")
    for r in seed:
        f.record(r)
    return FakePeer(f, device), f


# A round: apply, advance, idempotence


class TestRound:
    def test_applies_and_advances(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, rfeed = _remote(
            tmp_path, seed=[_rec("c1", 1, device="B"), _rec("c2", 1, device="B")]
        )
        res = eng.run_round("B", peer)
        assert res.applied == 2
        assert res.deferred == 0
        assert res.advanced is True
        assert res.new_watermark == rfeed.high_water()
        keys = {(r.kind.value, r.record_id) for r in feed.current_records()}
        assert keys == {("conversation", "c1"), ("conversation", "c2")}

    def test_second_round_is_idempotent(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, rfeed = _remote(tmp_path, seed=[_rec("c1", 1, device="B")])
        first = eng.run_round("B", peer)
        assert first.applied == 1 and first.advanced is True
        second = eng.run_round("B", peer)
        assert second.applied == 0
        assert second.advanced is False
        assert second.new_watermark == first.new_watermark
        assert store.get_watermark("B") == first.new_watermark

    def test_empty_remote_does_not_advance(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, _ = _remote(tmp_path, seed=())
        res = eng.run_round("B", peer)
        assert res.applied == 0
        assert res.advanced is False
        assert res.new_watermark == 0

    def test_unparseable_answer_holds_watermark(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        store.advance_watermark("B", 4)
        peer = RawPeer({"not": "a batch"})
        res = eng.run_round("B", peer)
        assert res.applied == 0
        assert res.advanced is False
        assert res.previous_watermark == 4
        assert res.new_watermark == 4

    def test_newer_local_then_peer_pull_converges(self, tmp_path):
        eng, feed, store = _engine(tmp_path, device="A")
        store.add_peer("B", "RKB")
        # local has c1@2; remote has c1@1 (older) -> local wins, applied 0
        eng.publish(_rec("c1", 2, device="A", payload={"v": "newer"}))
        peer, _ = _remote(tmp_path, seed=[_rec("c1", 1, device="B", payload={"v": "older"})])
        res = eng.run_round("B", peer)
        assert res.applied == 0
        snap = {r.record_id: r for r in feed.current_records()}
        assert snap["c1"].clock == 2


# Conflict pass-through


class TestConflict:
    def test_concurrent_divergence_is_retained(self, tmp_path):
        eng, feed, store = _engine(tmp_path, device="A")
        store.add_peer("B", "RKB")
        # same key and clock, different content from two devices -> a conflict
        eng.publish(_rec("c1", 5, device="A", payload={"side": "A"}))
        peer, _ = _remote(tmp_path, seed=[_rec("c1", 5, device="B", payload={"side": "B"})])
        res = eng.run_round("B", peer)
        assert res.conflicts == 1
        # the converged set holds exactly one winner for the key
        winners = [r for r in feed.current_records() if r.record_id == "c1"]
        assert len(winners) == 1


# The Bulbe seam


class TestBulbeSeam:
    def test_round_refuses_in_bulbe(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, _ = _remote(tmp_path, seed=[_rec("c1", 1, device="B")])
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            eng.run_round("B", peer)

    def test_round_refuses_when_mode_undeterminable(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, _ = _remote(tmp_path, seed=[_rec("c1", 1, device="B")])
        set_mode(raises=True)
        with pytest.raises(VeilidDisabledInBulbe):
            eng.run_round("B", peer)

    def test_gate_precedes_peer_check(self, tmp_path):
        # Under Bulbe, an unpaired peer still trips the gate first, not PeerNotFound.
        eng, feed, store = _engine(tmp_path)
        peer, _ = _remote(tmp_path, seed=())
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            eng.run_round("unpaired", peer)

    def test_peer_management_runs_in_bulbe(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        set_mode("bulbe")
        rec = eng.register_peer("B", "RKB", label="phone")
        assert rec.peer_id == "B"
        assert [p.peer_id for p in eng.list_peers()] == ["B"]
        assert eng.unregister_peer("B") is True

    def test_publish_runs_in_bulbe(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        set_mode("bulbe")
        eng.publish(_rec("c1", 1, device="A"))
        assert len(eng.local_records()) == 1


# Peer-not-found


class TestPeerNotFound:
    def test_round_on_unpaired_peer_raises(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        peer, _ = _remote(tmp_path, seed=())
        with pytest.raises(PeerNotFound):
            eng.run_round("ghost", peer)


# Approval discipline for a sensitive apply


class TestApproval:
    def test_skill_applied_when_approved(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, _ = _remote(
            tmp_path, seed=[_rec("s1", 1, device="B", kind=RecordKind.SKILL)]
        )
        res = eng.run_round("B", peer, approval_fn=lambda c, t, a: True)
        assert res.applied == 1
        assert res.deferred == 0
        assert res.advanced is True

    def test_skill_deferred_when_denied_and_watermark_held(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, _ = _remote(
            tmp_path, seed=[_rec("s1", 1, device="B", kind=RecordKind.SKILL)]
        )
        res = eng.run_round("B", peer, approval_fn=lambda c, t, a: False)
        assert res.applied == 0
        assert res.deferred == 1
        assert res.advanced is False
        assert res.new_watermark == 0  # held
        # the skill is not in the local set
        assert all(r.record_id != "s1" for r in feed.current_records())

    def test_mixed_batch_defers_only_sensitive(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, _ = _remote(
            tmp_path,
            seed=[
                _rec("c1", 1, device="B"),
                _rec("s1", 1, device="B", kind=RecordKind.SKILL),
            ],
        )
        res = eng.run_round("B", peer, approval_fn=lambda c, t, a: False)
        assert res.applied == 1  # the conversation
        assert res.deferred == 1  # the skill
        assert res.advanced is False  # held because something deferred
        ids = {r.record_id for r in feed.current_records()}
        assert "c1" in ids and "s1" not in ids

    def test_deferred_skill_reoffered_then_applied(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, _ = _remote(
            tmp_path, seed=[_rec("s1", 1, device="B", kind=RecordKind.SKILL)]
        )
        denied = eng.run_round("B", peer, approval_fn=lambda c, t, a: False)
        assert denied.deferred == 1 and denied.advanced is False
        granted = eng.run_round("B", peer, approval_fn=lambda c, t, a: True)
        assert granted.applied == 1
        assert granted.advanced is True

    def test_default_gate_is_allowlists_and_fail_secure(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "opti_oignon.agent.allowlists", _allowlists_stub())
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, _ = _remote(
            tmp_path, seed=[_rec("s1", 1, device="B", kind=RecordKind.SKILL)]
        )
        set_default_approval(False)
        res = eng.run_round("B", peer)  # no approval_fn -> default gate
        assert res.deferred == 1
        assert _APPROVAL["calls"], "the default gate must be consulted"
        conv, label, args, manager = _APPROVAL["calls"][-1]
        assert label == "sync_apply:skill"
        assert args["id"] == "s1" and args["kind"] == "skill"

    def test_default_gate_applies_when_granted(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "opti_oignon.agent.allowlists", _allowlists_stub())
        eng, feed, store = _engine(tmp_path)
        store.add_peer("B", "RKB")
        peer, _ = _remote(
            tmp_path, seed=[_rec("s1", 1, device="B", kind=RecordKind.SKILL)]
        )
        set_default_approval(True)
        res = eng.run_round("B", peer)
        assert res.applied == 1

    def test_sensitive_kinds_is_skill_only(self):
        assert sync_engine.SENSITIVE_KINDS == frozenset({"skill"})


# Producer and local production


class TestProducer:
    def test_record_from_payload_builds_valid_record(self):
        rec = sync_engine.record_from_payload(
            RecordKind.CONVERSATION, "c1", {"k": "v"}, device="A", clock=3
        )
        assert rec.record_id == "c1"
        assert rec.clock == 3
        assert rec.device == "A"
        assert records.verify_record_hash(rec) is True

    def test_publish_journals_and_lists(self, tmp_path):
        eng, feed, store = _engine(tmp_path)
        seq = eng.publish(_rec("c1", 1, device="A"))
        assert isinstance(seq, int) and seq > 0
        assert len(eng.local_records()) == 1
        assert feed.count() == 1


# The pure result-shaping helper


class TestResultShaping:
    def test_round_result_is_pure_and_correct(self):
        class _AR:
            applied = 2
            rejected = 1
            conflicts = [object(), object()]

        a = sync_engine._round_result(
            "B", _AR(), previous_watermark=1, new_watermark=5, deferred=3, advanced=True
        )
        b = sync_engine._round_result(
            "B", _AR(), previous_watermark=1, new_watermark=5, deferred=3, advanced=True
        )
        assert a == b
        assert a.applied == 2 and a.rejected == 1 and a.conflicts == 2
        assert a.deferred == 3 and a.previous_watermark == 1 and a.new_watermark == 5
        assert a.advanced is True


# The singleton


class TestSingleton:
    def test_get_set_reset(self, tmp_path):
        eng = sync_engine.SyncEngine(device="A")
        sync_engine.set_sync_engine(eng)
        assert sync_engine.get_sync_engine() is eng
        sync_engine.reset_sync_engine()
        assert sync_engine.get_sync_engine() is not eng

    def test_engine_requires_device(self):
        with pytest.raises(ValueError):
            sync_engine.SyncEngine(device="")
