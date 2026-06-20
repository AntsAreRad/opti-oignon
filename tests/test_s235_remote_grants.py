"""S235 -- cas 7 Lot 2: the per-device capability scoping and revocation.

Container-provable proof of the per-device grant store (REMOTE_INFERENCE_SPEC
sections 3, 12, 14 / D1), the RAG read-only sub-grant turning on, and the
revocation wiring (RA-01-class grant revoke plus the emergency-stop detach),
all without inventing a new revocation primitive.

What is proven here:

  - the peer store grows two nullable, additive grant columns by the existing
    ``table_info`` + ``ALTER TABLE`` idiom (the SYN-02/PAIR-02 migration shape):
    ``remote_chat_grant`` (NULL/1 = enabled, the grandfathered tier-1 default;
    0 = disabled) and ``rag_subgrant`` (NULL/0 = off, the conservative default;
    1 = on);
  - a fresh peer reads remote chat ENABLED and the RAG sub-grant OFF (the spec's
    default tier 1, RAG by separate sub-grant); a re-pair PRESERVES the grants
    (a local trust decision, like the watermark);
  - the handler's grant check is now store-backed: a device whose remote chat is
    disabled is refused (``remote_chat_disabled``); the Lot 1 default-tier-1
    stance is unchanged for a record without the columns (read defensively);
  - the RAG sub-grant gates the ``rag`` field: a granted device's ``rag`` scope
    is ACCEPTED (the surface gate passes; the funnel is entered), an ungranted
    device's ``rag`` is REFUSED (``rag_not_granted``) -- the Lot 1 default;
  - the revocation kills a LIVE session: revoking a device's grant drops its
    in-flight streaming buffers, so a subsequent continuation is refused -- the
    durable state is the grant column, the live kill is the buffer drop, neither
    a new primitive;
  - the emergency-stop detach (the existing unpair) is wired to the same buffer
    kill.

Red-before on the pristine tree: the grant columns, the set methods, and the
store-backed grant/sub-grant checks do not exist, so every assertion that
exercises them is RED.
"""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

guard = importlib.import_module("opti_oignon.veilid.guard")
protocol = importlib.import_module("opti_oignon.veilid.protocol")
peers = importlib.import_module("opti_oignon.veilid.peers")


def _streaming():
    try:
        return importlib.import_module("opti_oignon.veilid.remote_streaming")
    except Exception:
        return None


def _remote_inference():
    try:
        return importlib.import_module("opti_oignon.veilid.remote_inference")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fakes for the handler-level grant checks
# ---------------------------------------------------------------------------


class FakeRecord:
    def __init__(
        self,
        *,
        pending=False,
        signing_pub="PUBKEY",
        remote_chat_enabled=True,
        rag_subgrant=False,
    ):
        self.pending = pending
        self.signing_pub = signing_pub
        self.remote_chat_enabled = remote_chat_enabled
        self.rag_subgrant = rag_subgrant


class FakePeerStore:
    def __init__(self, peers=None):
        self._peers = dict(peers or {})

    def get_peer(self, peer_id):
        if not isinstance(peer_id, str) or not peer_id:
            return None
        return self._peers.get(peer_id)


class FakeExecutor:
    def __init__(self, reply="canned", boom=False):
        self.calls = []
        self._reply = reply
        self._boom = boom

    def execute(self, question, routing, **kwargs):
        self.calls.append({"question": question})
        if self._boom:
            raise AssertionError("the funnel must not be entered")
        reply = self._reply

        def _gen():
            yield reply
            return (reply, "chat")

        return _gen()


def _fake_router(prompt):
    import types

    return types.SimpleNamespace(model="fake-model")


class _AuditSpy:
    def __init__(self):
        self.events = []

    def __call__(self, action, **details):
        self.events.append((action, details))


def _request(**over):
    base = {
        "v": protocol.PROTOCOL_VERSION,
        "type": "remote_infer",
        "device": "phone-A",
        "request_id": "req-1",
        "prompt": "hello",
    }
    base.update(over)
    return base


def _serve(request, store, **kw):
    mod = _remote_inference()
    assert mod is not None
    kw.setdefault("peer_id", "phone-A")
    kw.setdefault("router", _fake_router)
    kw.setdefault("audit", _AuditSpy())
    kw.setdefault("executor", FakeExecutor())
    return mod.serve_remote_inference(request, peer_store=store, **kw)


@pytest.fixture(autouse=True)
def _reset_streaming():
    mod = _streaming()
    if mod is not None and hasattr(mod, "reset_for_tests"):
        mod.reset_for_tests()
    yield
    if mod is not None and hasattr(mod, "reset_for_tests"):
        mod.reset_for_tests()


def _fresh_store():
    d = tempfile.mkdtemp(prefix="oo_s235_peers_")
    return peers.PeerStore(root=d)


# ---------------------------------------------------------------------------
# Family 1 -- the peer store grant columns and their defaults
# ---------------------------------------------------------------------------


class TestPeerStoreGrantColumns:
    def test_record_exposes_grant_fields(self):
        rec = peers.PeerRecord(peer_id="p", routing_key="rk")
        assert hasattr(rec, "remote_chat_enabled")
        assert hasattr(rec, "rag_subgrant")

    def test_fresh_peer_defaults_remote_chat_on_rag_off(self):
        store = _fresh_store()
        store.add_peer("phone-A", "routekey-A", signing_pub="PUB")
        rec = store.get_peer("phone-A")
        assert rec is not None
        assert rec.remote_chat_enabled is True
        assert rec.rag_subgrant is False

    def test_set_remote_chat_grant_disables_and_enables(self):
        store = _fresh_store()
        store.add_peer("phone-A", "routekey-A")
        assert store.set_remote_chat_grant("phone-A", False) is True
        assert store.get_peer("phone-A").remote_chat_enabled is False
        assert store.set_remote_chat_grant("phone-A", True) is True
        assert store.get_peer("phone-A").remote_chat_enabled is True

    def test_set_rag_subgrant_on_and_off(self):
        store = _fresh_store()
        store.add_peer("phone-A", "routekey-A")
        assert store.set_rag_subgrant("phone-A", True) is True
        assert store.get_peer("phone-A").rag_subgrant is True
        assert store.set_rag_subgrant("phone-A", False) is True
        assert store.get_peer("phone-A").rag_subgrant is False

    def test_set_grant_on_unknown_peer_returns_false(self):
        store = _fresh_store()
        assert store.set_remote_chat_grant("ghost", False) is False
        assert store.set_rag_subgrant("ghost", True) is False

    def test_repair_preserves_grants(self):
        store = _fresh_store()
        store.add_peer("phone-A", "routekey-A", signing_pub="PUB")
        store.set_remote_chat_grant("phone-A", False)
        store.set_rag_subgrant("phone-A", True)
        # a re-pair with a rotated route (same signing key) must not reset grants
        store.add_peer("phone-A", "routekey-A-rotated", signing_pub="PUB")
        rec = store.get_peer("phone-A")
        assert rec.remote_chat_enabled is False
        assert rec.rag_subgrant is True
        assert rec.routing_key == "routekey-A-rotated"

    def test_migration_is_additive_table_info_idiom(self):
        src = (_REPO / "opti_oignon/veilid/peers.py").read_text(encoding="utf-8")
        assert "remote_chat_grant" in src
        assert "rag_subgrant" in src
        assert "ADD COLUMN remote_chat_grant" in src
        assert "ADD COLUMN rag_subgrant" in src


# ---------------------------------------------------------------------------
# Family 2 -- the handler's store-backed grant check
# ---------------------------------------------------------------------------


class TestHandlerGrantCheck:
    def test_enabled_confirmed_peer_is_served(self):
        store = FakePeerStore({"phone-A": FakeRecord(remote_chat_enabled=True)})
        out = _serve(_request(), store)
        assert out.get("ok") is True

    def test_disabled_peer_is_refused(self):
        store = FakePeerStore({"phone-A": FakeRecord(remote_chat_enabled=False)})
        out = _serve(_request(), store, executor=FakeExecutor(boom=True))
        assert out.get("ok") is False
        assert out.get("reason") == "remote_chat_disabled"

    def test_lot1_record_without_columns_still_granted(self):
        # a record stand-in lacking the new fields reads as the Lot 1 default
        class Lot1Record:
            pending = False
            signing_pub = "PUB"

        store = FakePeerStore({"phone-A": Lot1Record()})
        out = _serve(_request(), store)
        assert out.get("ok") is True


# ---------------------------------------------------------------------------
# Family 3 -- the RAG read-only sub-grant gate
# ---------------------------------------------------------------------------


class TestRagSubgrant:
    def test_rag_refused_when_subgrant_off(self):
        store = FakePeerStore({"phone-A": FakeRecord(rag_subgrant=False)})
        out = _serve(
            _request(rag={"collections": ["c1"], "query": "x"}),
            store,
            executor=FakeExecutor(boom=True),
        )
        assert out.get("ok") is False
        assert out.get("reason") == "rag_not_granted"

    def test_rag_accepted_when_subgrant_on(self):
        store = FakePeerStore({"phone-A": FakeRecord(rag_subgrant=True)})
        ex = FakeExecutor(reply="answer with rag")
        out = _serve(_request(rag={"collections": ["c1"], "query": "x"}), store, executor=ex)
        # the surface gate passes: the request is NOT refused and the funnel runs
        assert out.get("ok") is True
        assert len(ex.calls) == 1

    def test_rag_off_is_the_default_for_lot1_record(self):
        class Lot1Record:
            pending = False
            signing_pub = "PUB"

        store = FakePeerStore({"phone-A": Lot1Record()})
        out = _serve(
            _request(rag={"collections": ["c1"]}),
            store,
            executor=FakeExecutor(boom=True),
        )
        assert out.get("reason") == "rag_not_granted"


# ---------------------------------------------------------------------------
# Family 4 -- revocation kills a live session (no new primitive)
# ---------------------------------------------------------------------------


class TestRevocationKillsLiveSession:
    def test_revoke_drops_inflight_buffer(self):
        stream = _streaming()
        ri = _remote_inference()
        assert stream is not None and ri is not None
        store = FakePeerStore({"phone-A": FakeRecord()})
        # start a multi-chunk stream so a buffer is in flight
        ri.serve_remote_inference(
            _request(),
            peer_id="phone-A",
            peer_store=store,
            router=_fake_router,
            audit=_AuditSpy(),
            executor=_multi_chunk_executor(["a", "b", "c"]),
        )
        assert stream.active_session_count() == 1
        # revoke: the durable grant flips AND the live buffer is killed
        killed = stream.kill_sessions_for_device("phone-A")
        assert killed == 1
        out = ri.serve_remote_inference_continuation(
            {
                "v": protocol.PROTOCOL_VERSION,
                "type": "remote_infer_cont",
                "device": "phone-A",
                "request_id": "req-1",
                "cursor": 1,
            },
            peer_id="phone-A",
            peer_store=store,
            audit=_AuditSpy(),
        )
        assert out.get("ok") is False
        assert out.get("reason") == "buffer_mismatch"

    def test_unpair_kill_wired_in_routes_source(self):
        src = (_REPO / "opti_oignon/api/routes_sync.py").read_text(encoding="utf-8")
        # the detach-in-one-gesture (unpair) drops the device's live sessions
        assert "kill_sessions_for_device" in src


def _multi_chunk_executor(chunks):
    class _Ex:
        def __init__(self):
            self.calls = []

        def execute(self, question, routing, **kwargs):
            self.calls.append(question)

            def _gen():
                for c in chunks:
                    yield c
                return ("".join(chunks), "chat")

            return _gen()

    return _Ex()
