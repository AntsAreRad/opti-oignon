#!/usr/bin/env python3
"""Tests for S182 Goal 3 -- real-network conditions (Theme 4 / Veilid Sync).

Exercises a round and the responder under adverse conditions over the injectable
transport, with no live framework: the round is driven through the engine against
a live VeilidPeer whose messenger answers from a remote change feed (the same code
the live route uses), and the failure modes are induced at the messenger seam.

- Simulated devices: three devices publish and pull pairwise until the full set
  converges on every device; a re-run is idempotent (applies nothing).
- A stalled peer (the timeout bound): a messenger that raises VeilidTimeout
  propagates through VeilidPeer.fetch and run_round (the route maps it to 504), and
  the watermark is left unchanged.
- An abrupt disconnect and a late re-join: while a peer is unreachable its reply is
  unusable, so the round holds the watermark and applies nothing; once the peer
  re-joins the round resumes from the held watermark, advances it, and applies the
  delta -- nothing is skipped.
- Conflicting edits: two devices make concurrent divergent edits to the same key at
  the same clock; the reconciler keeps one deterministically and retains the loser
  in the conflict log, and both devices converge on the same set.

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode is
driven Daily, the audit log is a no-op. The transport is exercised with a fake
messenger answering from a feed -- no veilid framework and no live server.
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
transport = _load("transport")

RecordKind = records.RecordKind
VeilidTimeout = guard.VeilidTimeout


@pytest.fixture(autouse=True)
def _daily():
    set_mode("daily")
    yield
    set_mode("daily")


# --- Fakes: a messenger that answers from a remote feed, and failure injectors ---


class FeedMessenger:
    """Answers a request from a remote change feed over the wire encoding (no socket)."""

    def __init__(self, feed, device):
        self.feed = feed
        self.device = device

    def call(self, routing_key, payload, *, timeout=None):
        request = transport.decode_answer(payload)
        batch = protocol.respond_to_request(self.feed, request, device=self.device)
        return transport._encode_message(batch)


class TimeoutMessenger:
    """A stalled or hostile peer: the send never returns within the budget."""

    def call(self, routing_key, payload, *, timeout=None):
        raise VeilidTimeout("peer stalled")


class DisconnectedMessenger:
    """An unreachable peer: the reply is unusable, so the round must hold."""

    def call(self, routing_key, payload, *, timeout=None):
        return b""  # decode_answer -> None -> empty round, watermark held


def _device(tmp_path, name):
    feed = change_feed.ChangeFeed(root=tmp_path / f"feed-{name}")
    store = peers.PeerStore(root=tmp_path / f"store-{name}")
    eng = sync_engine.SyncEngine(device=name, feed=feed, store=store)
    return types.SimpleNamespace(name=name, feed=feed, store=store, engine=eng)


def _live_peer(remote, *, routing_key="rk"):
    return transport.VeilidPeer(FeedMessenger(remote.feed, remote.name), routing_key)


def _pull(local, remote, *, approve=True):
    """Register the remote (if needed) and run one live pull round from it.

    A sensitive record (a skill) passes the human gate; the approval discipline is
    tested elsewhere, so here an injected approval_fn approves so the round is
    deterministic and instant rather than blocking on the manager-backed gate.
    """
    peer_id = remote.name
    if not local.store.has_peer(peer_id):
        local.engine.register_peer(peer_id, "rk-" + peer_id, label=peer_id)
    approval_fn = (lambda cid, label, args: True) if approve else None
    return local.engine.run_round(peer_id, _live_peer(remote), approval_fn=approval_fn)


def _keys(device):
    return {records.key_of(r): r.content_hash for r in device.feed.current_records()}


# Three simulated devices converge


class TestSimulatedDevices:
    def test_three_devices_converge(self, tmp_path):
        a = _device(tmp_path, "A")
        b = _device(tmp_path, "B")
        c = _device(tmp_path, "C")
        a.engine.publish_conversation("conv-a", {"t": "from A"}, clock=1)
        b.engine.publish_memory_canonical("fact-b", {"t": "from B"}, clock=1)
        c.engine.publish_skill("skill-c", {"t": "from C"}, clock=1)

        # Gossip: each device pulls from the other two, twice, to propagate fully.
        for _ in range(2):
            for local, remotes in ((a, (b, c)), (b, (a, c)), (c, (a, b))):
                for remote in remotes:
                    _pull(local, remote)

        ka, kb, kc = _keys(a), _keys(b), _keys(c)
        assert ka == kb == kc
        assert len(ka) == 3  # all three records on every device

    def test_round_is_idempotent(self, tmp_path):
        a = _device(tmp_path, "A")
        b = _device(tmp_path, "B")
        b.engine.publish_conversation("conv-b", {"t": "x"}, clock=1)
        first = _pull(a, b)
        assert first.applied == 1 and first.advanced is True
        second = _pull(a, b)
        assert second.applied == 0 and second.advanced is False
        assert second.new_watermark == first.new_watermark


# A stalled peer: the timeout bound


class TestStalledPeer:
    def test_timeout_propagates(self, tmp_path):
        a = _device(tmp_path, "A")
        a.engine.register_peer("B", "rk-B", label="B")
        peer = transport.VeilidPeer(TimeoutMessenger(), "rk-B")
        with pytest.raises(VeilidTimeout):
            a.engine.run_round("B", peer)

    def test_watermark_unchanged_after_timeout(self, tmp_path):
        a = _device(tmp_path, "A")
        a.engine.register_peer("B", "rk-B", label="B")
        a.store.advance_watermark("B", 5)
        peer = transport.VeilidPeer(TimeoutMessenger(), "rk-B")
        with pytest.raises(VeilidTimeout):
            a.engine.run_round("B", peer)
        assert a.store.get_watermark("B") == 5  # untouched by the failed round


# An abrupt disconnect and a late re-join


class TestDisconnectAndRejoin:
    def test_disconnect_holds_watermark(self, tmp_path):
        a = _device(tmp_path, "A")
        b = _device(tmp_path, "B")
        b.engine.publish_conversation("conv-b", {"t": "x"}, clock=1)
        a.engine.register_peer("B", "rk-B", label="B")
        # While B is unreachable, the reply is unusable: the round holds.
        down = transport.VeilidPeer(DisconnectedMessenger(), "rk-B")
        result = a.engine.run_round("B", down)
        assert result.applied == 0
        assert result.advanced is False
        assert a.store.get_watermark("B") == 0

    def test_rejoin_resumes_from_held_watermark(self, tmp_path):
        a = _device(tmp_path, "A")
        b = _device(tmp_path, "B")
        b.engine.publish_conversation("conv-b1", {"t": "1"}, clock=1)
        a.engine.register_peer("B", "rk-B", label="B")

        # First contact succeeds and advances.
        r1 = a.engine.run_round("B", _live_peer(b))
        assert r1.applied == 1 and r1.advanced is True
        held = a.store.get_watermark("B")

        # B keeps writing while A is disconnected; A's round holds.
        b.engine.publish_conversation("conv-b2", {"t": "2"}, clock=1)
        down = transport.VeilidPeer(DisconnectedMessenger(), "rk-B")
        r2 = a.engine.run_round("B", down)
        assert r2.applied == 0 and r2.advanced is False
        assert a.store.get_watermark("B") == held  # nothing lost

        # B re-joins: the round resumes from the held watermark and catches up.
        r3 = a.engine.run_round("B", _live_peer(b))
        assert r3.applied == 1 and r3.advanced is True
        assert a.store.get_watermark("B") > held
        keys = {records.key_of(r) for r in a.feed.current_records()}
        assert ("conversation", "conv-b1") in keys
        assert ("conversation", "conv-b2") in keys  # the late write was not skipped


# Conflicting edits: the conflict log is retained and the set converges


class TestConflictingEdits:
    def _diverge(self, tmp_path):
        a = _device(tmp_path, "A")
        b = _device(tmp_path, "B")
        # Concurrent divergence: same key, same clock, different content.
        a.engine.publish_conversation("shared", {"edit": "from A"}, clock=1)
        b.engine.publish_conversation("shared", {"edit": "from B"}, clock=1)
        return a, b

    def test_conflict_retained(self, tmp_path):
        a, b = self._diverge(tmp_path)
        result = _pull(a, b)
        assert result.conflicts >= 1  # the loser is retained, not dropped

    def test_devices_converge_on_same_winner(self, tmp_path):
        a, b = self._diverge(tmp_path)
        # Pull both ways, twice, so each adopts the deterministic winner.
        for _ in range(2):
            _pull(a, b)
            _pull(b, a)
        wa = _keys(a)[("conversation", "shared")]
        wb = _keys(b)[("conversation", "shared")]
        assert wa == wb  # deterministic tie-break -> same winner on both

    def test_winner_is_deterministic_tiebreak(self, tmp_path):
        a, b = self._diverge(tmp_path)
        # The reconciler's tie-break (highest clock, then content hash, then device)
        # is order-independent: reconciling A+B equals reconciling B+A.
        ra = reconcile.reconcile(
            a.feed.current_records(), b.feed.current_records()
        )
        rb = reconcile.reconcile(
            b.feed.current_records(), a.feed.current_records()
        )
        wa = {records.key_of(r): r.content_hash for r in ra.records}
        wb = {records.key_of(r): r.content_hash for r in rb.records}
        assert wa == wb
        assert len(ra.conflicts) >= 1 and len(rb.conflicts) >= 1
