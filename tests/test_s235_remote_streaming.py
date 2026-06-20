"""S235 -- cas 7 Lot 2: the streaming shape over app_call (Option A, pull).

Container-provable proof of the chunked-app_call streaming design
(REMOTE_INFERENCE_SPEC section 5, Option A) and the section-14 buffer-mismatch
risk. The desktop buffers the response chunks keyed by the
``(route-authenticated peer_id, request_id)`` pair; the phone issues the
initial ``remote_infer`` request and then successive ``remote_infer_cont``
app_calls, each carrying the request id and a cursor, until a terminal done
marker. The buffer is bound to the device AND the request id, so a malformed or
hostile continuation can never read another request id's buffer: a lookup keyed
by the route peer_id (which the transport supplies, never the payload) for a
request id it does not own simply misses, and the miss is a structured refusal.

What is proven here, all over fakes (no ollama, no live route):

  - the new ``remote_infer_cont`` wire kind and its dispatch from inside
    ``serve_app_call`` (the sync delta and the inference kinds never collide);
  - the streaming session registry: open a session, pull chunk by chunk with a
    monotonic cursor, terminate on the done marker, the session dropped when
    drained;
  - the buffer binding: a continuation for an unknown ``(peer, request_id)``, or
    for a request id owned by a DIFFERENT device, is a ``buffer_mismatch``
    refusal -- never a cross-read (the section-14 risk);
  - the bound is the registry's compound key, so the proof is a literal lookup;
  - the registry is bounded (a cap on live sessions; never unbounded growth);
  - end to end through the handler: the initial reply carries the first chunk
    plus a cursor and a done flag, and the continuation handler serves the rest,
    each audit-chained, each refused under Bulbe.

The real-route streamed latency (the incremental buffer fill on a live private
route) is host-assured and named in the spec (section 12, Lot 2 / Lot 3); it is
never simulated here. The chunk/cursor/done/mismatch CONTRACT is what the
container proves.

Red-before on the pristine tree: every assertion that depends on the streaming
module or the continuation handler is RED (neither exists, and ``serve_app_call``
carries no continuation dispatch); the dispatch and protocol pins fail too until
the kind and the route land.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

guard = importlib.import_module("opti_oignon.veilid.guard")
protocol = importlib.import_module("opti_oignon.veilid.protocol")
transport = importlib.import_module("opti_oignon.veilid.transport")


def _streaming():
    """Lazy import of the Lot 2 streaming module; absence is a clean failure."""
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
# Fakes (the established idiom: no live transport, no backend, no ollama)
# ---------------------------------------------------------------------------


class FakeRecord:
    """A peer-store record stand-in carrying the grant fields the handler reads."""

    def __init__(
        self,
        *,
        pending: bool = False,
        signing_pub: str = "PUBKEY",
        remote_chat_enabled: bool = True,
        rag_subgrant: bool = False,
    ) -> None:
        self.pending = pending
        self.signing_pub = signing_pub
        self.remote_chat_enabled = remote_chat_enabled
        self.rag_subgrant = rag_subgrant


class FakePeerStore:
    def __init__(self, peers=None) -> None:
        self._peers = dict(peers or {})

    def get_peer(self, peer_id):
        if not isinstance(peer_id, str) or not peer_id:
            return None
        return self._peers.get(peer_id)


class FakeChunkExecutor:
    """A chat-funnel stand-in that yields a fixed list of chunks (the stream)."""

    def __init__(self, chunks) -> None:
        self.calls = []
        self._chunks = list(chunks)

    def execute(self, question, routing, **kwargs):
        self.calls.append({"question": question, "routing": routing})
        chunks = self._chunks

        def _gen():
            for c in chunks:
                yield c
            return ("".join(chunks), "chat")

        return _gen()


def _fake_router(prompt):
    import types

    return types.SimpleNamespace(model="fake-model", task_type="chat")


class _AuditSpy:
    def __init__(self) -> None:
        self.events = []

    def __call__(self, action, **details):
        self.events.append((action, details))


def _granted_store(peer_id="phone-A"):
    return FakePeerStore({peer_id: FakeRecord()})


def _init_request(**over):
    base = {
        "v": protocol.PROTOCOL_VERSION,
        "type": "remote_infer",
        "device": "phone-A",
        "request_id": "req-1",
        "prompt": "stream me a long answer please",
    }
    base.update(over)
    return base


def _cont_request(**over):
    base = {
        "v": protocol.PROTOCOL_VERSION,
        "type": "remote_infer_cont",
        "device": "phone-A",
        "request_id": "req-1",
        "cursor": 1,
    }
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def _reset_streaming():
    mod = _streaming()
    if mod is not None and hasattr(mod, "reset_for_tests"):
        mod.reset_for_tests()
    yield
    if mod is not None and hasattr(mod, "reset_for_tests"):
        mod.reset_for_tests()


# ---------------------------------------------------------------------------
# Family 1 -- module presence, the sentinel, the new wire kind, the dispatch
# ---------------------------------------------------------------------------


class TestModuleAndWiring:
    def test_streaming_module_present(self):
        mod = _streaming()
        assert mod is not None, "the remote_streaming module must exist"

    def test_checkpoint_sentinel_hardcoded(self):
        mod = _streaming()
        assert mod is not None
        assert getattr(mod, "checkpoint_before_apply", None) is True

    def test_protocol_carries_continuation_kind(self):
        assert getattr(protocol, "MSG_REMOTE_INFER_CONT", None) == "remote_infer_cont"
        # the continuation kind is distinct from the inference and sync kinds.
        kinds = {
            protocol.MSG_DELTA_REQUEST,
            protocol.MSG_RECORD_BATCH,
            protocol.MSG_REMOTE_INFER,
            protocol.MSG_REMOTE_INFER_CONT,
        }
        assert len(kinds) == 4

    def test_continuation_handler_present(self):
        mod = _remote_inference()
        assert mod is not None
        assert hasattr(mod, "serve_remote_inference_continuation")


# ---------------------------------------------------------------------------
# Family 2 -- the streaming session registry: open, pull, terminate, drop
# ---------------------------------------------------------------------------


class TestSessionRegistry:
    def test_single_chunk_session_is_done_at_first_pull(self):
        mod = _streaming()
        assert mod is not None
        mod.open_session("phone-A", "req-1", ["the whole answer"])
        out = mod.pull("phone-A", "req-1", 0)
        assert out is not None
        assert out["content"] == "the whole answer"
        assert out["cursor"] == 1
        assert out["done"] is True

    def test_single_chunk_session_dropped_after_terminal_pull(self):
        mod = _streaming()
        assert mod is not None
        mod.open_session("phone-A", "req-1", ["only chunk"])
        assert mod.pull("phone-A", "req-1", 0) is not None
        # drained and dropped: a later pull is a miss (the stream is consumed).
        assert mod.pull("phone-A", "req-1", 1) is None

    def test_multi_chunk_session_streams_in_order(self):
        mod = _streaming()
        assert mod is not None
        mod.open_session("phone-A", "req-1", ["alpha", "beta", "gamma"])
        first = mod.pull("phone-A", "req-1", 0)
        assert first["content"] == "alpha" and first["cursor"] == 1 and first["done"] is False
        second = mod.pull("phone-A", "req-1", 1)
        assert second["content"] == "beta" and second["cursor"] == 2 and second["done"] is False
        third = mod.pull("phone-A", "req-1", 2)
        assert third["content"] == "gamma" and third["cursor"] == 3 and third["done"] is True

    def test_active_session_count_tracks_open_and_drop(self):
        mod = _streaming()
        assert mod is not None
        assert mod.active_session_count() == 0
        mod.open_session("phone-A", "req-1", ["a", "b"])
        assert mod.active_session_count() == 1
        mod.pull("phone-A", "req-1", 0)  # not done
        assert mod.active_session_count() == 1
        mod.pull("phone-A", "req-1", 1)  # done -> dropped
        assert mod.active_session_count() == 0


# ---------------------------------------------------------------------------
# Family 3 -- the buffer binding (section-14): a mismatch is a refusal, never
#             a cross-read
# ---------------------------------------------------------------------------


class TestBufferMismatch:
    def test_pull_unknown_request_is_a_miss(self):
        mod = _streaming()
        assert mod is not None
        assert mod.pull("phone-A", "no-such-request", 0) is None

    def test_pull_for_another_devices_request_is_a_miss(self):
        mod = _streaming()
        assert mod is not None
        # phone-A owns req-1; phone-B must NOT be able to read it.
        mod.open_session("phone-A", "req-1", ["secret", "answer"])
        assert mod.pull("phone-B", "req-1", 0) is None
        # phone-A's own session is untouched by the hostile pull.
        assert mod.pull("phone-A", "req-1", 0) is not None

    def test_key_is_compound_device_and_request(self):
        mod = _streaming()
        assert mod is not None
        mod.open_session("phone-A", "req-1", ["x", "y"])
        mod.open_session("phone-B", "req-1", ["p", "q"])
        # same request id, different devices -> two independent sessions.
        a = mod.pull("phone-A", "req-1", 0)
        b = mod.pull("phone-B", "req-1", 0)
        assert a["content"] == "x"
        assert b["content"] == "p"


# ---------------------------------------------------------------------------
# Family 4 -- kill levers and bounds
# ---------------------------------------------------------------------------


class TestKillAndBounds:
    def test_kill_sessions_for_device_drops_only_that_device(self):
        mod = _streaming()
        assert mod is not None
        mod.open_session("phone-A", "req-1", ["a", "b"])
        mod.open_session("phone-B", "req-2", ["c", "d"])
        killed = mod.kill_sessions_for_device("phone-A")
        assert killed == 1
        assert mod.pull("phone-A", "req-1", 0) is None
        assert mod.pull("phone-B", "req-2", 0) is not None

    def test_kill_all_sessions_clears_everything(self):
        mod = _streaming()
        assert mod is not None
        mod.open_session("phone-A", "req-1", ["a", "b"])
        mod.open_session("phone-B", "req-2", ["c", "d"])
        mod.kill_all_sessions()
        assert mod.active_session_count() == 0

    def test_registry_is_bounded(self):
        mod = _streaming()
        assert mod is not None
        cap = getattr(mod, "MAX_SESSIONS", None)
        assert isinstance(cap, int) and cap > 0
        for i in range(cap + 50):
            mod.open_session("dev-%d" % i, "req", ["x", "y"])
        assert mod.active_session_count() <= cap


# ---------------------------------------------------------------------------
# Family 5 -- end to end through the handler (init first chunk + continuation)
# ---------------------------------------------------------------------------


def _serve_init(request, **kw):
    mod = _remote_inference()
    assert mod is not None
    kw.setdefault("peer_id", "phone-A")
    kw.setdefault("peer_store", _granted_store())
    kw.setdefault("router", _fake_router)
    kw.setdefault("audit", _AuditSpy())
    return mod.serve_remote_inference(request, **kw)


def _serve_cont(request, **kw):
    mod = _remote_inference()
    assert mod is not None
    kw.setdefault("peer_id", "phone-A")
    kw.setdefault("peer_store", _granted_store())
    kw.setdefault("audit", _AuditSpy())
    return mod.serve_remote_inference_continuation(request, **kw)


class TestHandlerStreaming:
    def test_init_reply_carries_first_chunk_and_not_done(self):
        ex = FakeChunkExecutor(["one ", "two ", "three"])
        out = _serve_init(_init_request(), executor=ex)
        assert out.get("ok") is True
        assert out.get("content") == "one "
        assert out.get("cursor") == 1
        assert out.get("done") is False
        assert out.get("type") == "remote_infer"

    def test_continuation_serves_the_rest_in_order(self):
        ex = FakeChunkExecutor(["one ", "two ", "three"])
        _serve_init(_init_request(), executor=ex)
        c1 = _serve_cont(_cont_request(cursor=1))
        assert c1.get("content") == "two " and c1.get("done") is False
        c2 = _serve_cont(_cont_request(cursor=2))
        assert c2.get("content") == "three" and c2.get("done") is True
        assert c2.get("type") == "remote_infer_cont"

    def test_single_chunk_init_is_done_immediately(self):
        ex = FakeChunkExecutor(["all of it"])
        out = _serve_init(_init_request(), executor=ex)
        assert out.get("content") == "all of it"
        assert out.get("done") is True

    def test_continuation_for_another_device_refused_buffer_mismatch(self):
        ex = FakeChunkExecutor(["one ", "two "])
        # phone-A starts the stream
        _serve_init(_init_request(), executor=ex, peer_id="phone-A")
        # phone-B (a different route-authenticated peer, present in its own
        # store) tries to read phone-A's request id
        out = _serve_cont(
            _cont_request(device="phone-B", cursor=1),
            peer_id="phone-B",
            peer_store=_granted_store("phone-B"),
        )
        assert out.get("ok") is False
        assert out.get("reason") == "buffer_mismatch"

    def test_continuation_unknown_request_refused(self):
        out = _serve_cont(_cont_request(request_id="never-opened", cursor=1))
        assert out.get("ok") is False
        assert out.get("reason") == "buffer_mismatch"

    def test_continuation_reply_is_json_safe(self):
        ex = FakeChunkExecutor(["a", "b"])
        _serve_init(_init_request(), executor=ex)
        out = _serve_cont(_cont_request(cursor=1))
        json.dumps(out)  # raises if not JSON-safe

    def test_continuation_refused_under_bulbe(self, monkeypatch):
        ex = FakeChunkExecutor(["a", "b"])
        _serve_init(_init_request(), executor=ex)
        monkeypatch.setattr(guard, "bulbe_disabled", lambda: True)
        spy = _AuditSpy()
        with pytest.raises(guard.VeilidDisabledInBulbe):
            _serve_cont(_cont_request(cursor=1), audit=spy)
        assert any(a == "remote_infer_refused" for a, _ in spy.events)


# ---------------------------------------------------------------------------
# Family 6 -- the dispatch wiring on the transport
# ---------------------------------------------------------------------------


class TestDispatchWiring:
    def test_continuation_routed_to_handler_not_to_sync(self):
        class FakeEngine:
            def __init__(self):
                self.served = []

            def serve_request(self, request, *, peer_id=""):
                self.served.append((request, peer_id))
                return {"v": protocol.PROTOCOL_VERSION, "type": protocol.MSG_RECORD_BATCH,
                        "device": "desk", "high_water": 0, "records": []}

        eng = FakeEngine()
        msg = transport._encode_message(_cont_request(cursor=1))
        reply = transport.serve_app_call(eng, msg, peer_id="phone-A")
        out = transport.decode_answer(reply)
        assert eng.served == [], "the continuation must not reach the sync responder"
        assert out is not None and out.get("type") == "remote_infer_cont"

    def test_continuation_dispatch_present_in_source(self):
        src = (_REPO / "opti_oignon/veilid/transport.py").read_text(encoding="utf-8")
        assert "MSG_REMOTE_INFER_CONT" in src
        assert "serve_remote_inference_continuation" in src
