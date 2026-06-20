"""S234 -- cas 7 Lot 1: the served remote-inference handler.

Container-provable proof of the served inference handler on the desktop
responder side, in the request-then-single-reply form (no streaming yet;
streaming is Lot 2). The suite holds the Lot 1 invariants from
REMOTE_INFERENCE_SPEC sections 3, 4, 6, 7, 11, 12 and D5:

  - the request-kind discrimination on the wire (a new ``remote_infer``
    envelope type, dispatched from inside ``serve_app_call`` alongside the
    sync delta);
  - the tier 1 bounded surface enforced as a single gate before any chat
    request is built (inference only in Lot 1; any out-of-surface field is
    REFUSED with a structured refusal, never silently dropped);
  - RAG-read treated as a SEPARATE SUB-GRANT, off by default in Lot 1
    (the conservative default; the per-device grant store is deferred to
    Lot 2), so a request carrying a RAG scope is refused;
  - the default-tier-1 grant stance: a route-authenticated peer that is
    known and not pending (PAIR-02 confirmed) holds the remote-chat grant;
    an unknown or pending peer, or a provenance mismatch, is refused;
  - the admission funnel: the handler submits to the executor's chat funnel
    (which traverses ``admit()``) and NEVER calls the backend directly;
  - the request-then-single-reply form keyed by the request id;
  - the Bulbe refusal: the binding-layer gate is re-asserted at the handler
    seam, the refusal is audit-chained, and ``VeilidDisabledInBulbe``
    propagates (Bulbe means nothing remotely -- physical, not a policy flag);
  - every served request and every refusal is audit-chained on the same
    hash-chain trail ``serve_request`` writes to;
  - a stalled or hostile peer surfaces as ``VeilidTimeout`` and never wedges
    the caller.

The real route between two devices is host-assured and named in the spec
(section 12, Lot 1 / Lot 3), never simulated here. The handler is exercised
with the established fake-messenger / fake-client / fake-executor idiom and
an injected audit seam; no ollama import chain is pulled.

Red-before on the pristine tree: every module-dependent assertion is RED
(the module is absent and ``serve_app_call`` carries no dispatch); the seam
pins on the surfaces the handler builds on and the VeilidTimeout pin on the
existing transport are GREEN by design.
"""

from __future__ import annotations

import importlib
import inspect
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

# Import-safe veilid modules (no ollama chain): the handler builds on these.
guard = importlib.import_module("opti_oignon.veilid.guard")
protocol = importlib.import_module("opti_oignon.veilid.protocol")
transport = importlib.import_module("opti_oignon.veilid.transport")


def _load_remote_inference():
    """Lazy import of the Lot 1 module.

    Absence is a clean assertion FAILURE (returns ``None``), never a
    collection error, so the pristine tree reports red, not broken.
    """
    try:
        return importlib.import_module("opti_oignon.veilid.remote_inference")
    except Exception:
        return None


def _read(rel: str) -> str:
    """Defensive source read: a missing file yields '' so a pin fails, not errors."""
    path = _REPO / rel
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ---------------------------------------------------------------------------
# Fakes (the established idiom: no live transport, no backend, no ollama)
# ---------------------------------------------------------------------------


class FakeRecord:
    """A peer-store record stand-in carrying the two fields Lot 1 reads."""

    def __init__(self, *, pending: bool = False, signing_pub: str = "PUBKEY") -> None:
        self.pending = pending
        self.signing_pub = signing_pub


class FakePeerStore:
    """A peer store stand-in: ``get_peer`` only, keyed by peer id."""

    def __init__(self, peers=None) -> None:
        self._peers = dict(peers or {})

    def get_peer(self, peer_id):
        if not isinstance(peer_id, str) or not peer_id:
            return None
        return self._peers.get(peer_id)


class FakeExecutor:
    """The chat-funnel boundary stand-in.

    The handler must submit through ``execute`` (the funnel that admits) and
    never around it; the canned reply is exactly what the handler must return,
    which proves the funnel was traversed and the backend was never reached
    directly. ``boom`` asserts the funnel is never entered on a refusal path.
    """

    def __init__(self, reply: str = "canned remote reply", boom: bool = False) -> None:
        self.calls = []
        self._reply = reply
        self._boom = boom

    def execute(self, question, routing, **kwargs):
        self.calls.append({"question": question, "routing": routing, "kwargs": kwargs})
        if self._boom:
            raise AssertionError("the funnel must not be entered on a refusal path")
        reply = self._reply

        def _gen():
            yield reply
            return (reply, "chat")

        return _gen()


def _fake_router(prompt):
    """Stand in for analyze + route: returns an opaque routing, no ollama."""
    return types.SimpleNamespace(model="fake-model", task_type="chat", temperature=0.5)


class _AuditSpy:
    """Capture the handler's audit-chain events."""

    def __init__(self) -> None:
        self.events = []

    def __call__(self, action, **details):
        self.events.append((action, details))


def _granted_store(peer_id: str = "phone-A") -> FakePeerStore:
    return FakePeerStore({peer_id: FakeRecord(pending=False, signing_pub="PUBKEY")})


def _request(**over) -> dict:
    base = {
        "v": protocol.PROTOCOL_VERSION,
        "type": "remote_infer",
        "device": "phone-A",
        "request_id": "req-1",
        "prompt": "What is the capital of France?",
    }
    base.update(over)
    return base


def _serve(request, **kw):
    """Call the handler with sane defaults; tests override per case."""
    mod = _load_remote_inference()
    assert mod is not None, "the remote_inference module must exist"
    kw.setdefault("peer_id", "phone-A")
    kw.setdefault("peer_store", _granted_store())
    kw.setdefault("executor", FakeExecutor())
    kw.setdefault("router", _fake_router)
    kw.setdefault("audit", _AuditSpy())
    return mod.serve_remote_inference(request, **kw)


# ---------------------------------------------------------------------------
# Family 1 -- module presence, the sentinel, the protocol kind, the signature
# ---------------------------------------------------------------------------


class TestModulePresence:
    def test_module_imports_and_handler_present(self):
        mod = _load_remote_inference()
        assert mod is not None, "remote_inference module must exist"
        assert hasattr(mod, "serve_remote_inference")

    def test_checkpoint_sentinel_hardcoded(self):
        mod = _load_remote_inference()
        assert mod is not None
        assert getattr(mod, "checkpoint_before_apply", None) is True

    def test_protocol_carries_remote_infer_kind(self):
        assert getattr(protocol, "MSG_REMOTE_INFER", None) == "remote_infer"

    def test_handler_signature_exposes_injectable_seams(self):
        mod = _load_remote_inference()
        assert mod is not None
        params = inspect.signature(mod.serve_remote_inference).parameters
        for name in ("peer_id", "executor", "router", "peer_store", "audit"):
            assert name in params, "missing injectable seam: " + name


# ---------------------------------------------------------------------------
# Family 2 -- the tier 1 bounded surface as a single gate
# ---------------------------------------------------------------------------


class TestBoundedSurface:
    def test_inference_only_request_is_admitted(self):
        out = _serve(_request())
        assert out.get("ok") is True
        assert out.get("refused") is not True
        assert isinstance(out.get("content"), str) and out["content"]

    def test_state_mutation_field_refused_not_dropped(self):
        out = _serve(_request(manage_memory={"set": "x"}))
        assert out.get("ok") is False
        assert out.get("refused") is True
        assert out.get("reason") == "out_of_surface"

    def test_sandbox_field_refused(self):
        out = _serve(_request(sandbox={"cmd": "ls"}))
        assert out.get("refused") is True
        assert out.get("reason") == "out_of_surface"

    def test_config_field_refused(self):
        out = _serve(_request(config={"x": 1}))
        assert out.get("refused") is True
        assert out.get("reason") == "out_of_surface"

    def test_unknown_field_refused(self):
        out = _serve(_request(surprise=1))
        assert out.get("refused") is True
        assert out.get("reason") == "out_of_surface"

    def test_rag_scope_refused_as_separate_subgrant_off_in_lot1(self):
        out = _serve(_request(rag={"collections": ["c1"], "query": "x"}))
        assert out.get("refused") is True
        assert out.get("reason") == "rag_not_granted"

    def test_refusal_happens_before_any_chat_is_built(self):
        ex = FakeExecutor(boom=True)
        out = _serve(_request(config={"x": 1}), executor=ex)
        assert out.get("refused") is True
        assert ex.calls == [], "the chat funnel must not be entered on a refusal"


# ---------------------------------------------------------------------------
# Family 3 -- authentication and the default-tier-1 grant stance
# ---------------------------------------------------------------------------


class TestAuthAndGrant:
    def test_confirmed_peer_is_granted(self):
        out = _serve(_request())
        assert out.get("ok") is True

    def test_pending_peer_refused(self):
        store = FakePeerStore({"phone-A": FakeRecord(pending=True)})
        out = _serve(_request(), peer_store=store)
        assert out.get("refused") is True
        assert out.get("reason") == "peer_not_confirmed"

    def test_unknown_peer_refused(self):
        out = _serve(_request(), peer_store=FakePeerStore({}))
        assert out.get("refused") is True
        assert out.get("reason") == "unknown_device"

    def test_provenance_mismatch_refused(self):
        # the route authenticated phone-A; the request claims to be phone-B.
        out = _serve(
            _request(device="phone-B"),
            peer_id="phone-A",
            peer_store=_granted_store("phone-A"),
        )
        assert out.get("refused") is True
        assert out.get("reason") == "provenance_mismatch"

    def test_no_authenticated_identity_refused(self):
        req = _request()
        req.pop("device")
        out = _serve(req, peer_id="")
        assert out.get("refused") is True


# ---------------------------------------------------------------------------
# Family 4 -- the admission funnel and the single-reply form
# ---------------------------------------------------------------------------


class TestAdmissionFunnelAndSingleReply:
    def test_submits_through_executor_funnel_exactly_once(self):
        ex = FakeExecutor(reply="hello from desktop")
        out = _serve(_request(), executor=ex)
        assert out.get("ok") is True
        assert len(ex.calls) == 1
        assert ex.calls[0]["question"] == "What is the capital of France?"

    def test_single_reply_carries_request_id_and_content(self):
        out = _serve(_request(request_id="req-42"), executor=FakeExecutor(reply="R"))
        assert out.get("request_id") == "req-42"
        assert out.get("content") == "R"

    def test_handler_never_calls_backend_directly(self):
        # the fake executor IS the funnel boundary; the reply is exactly the
        # fake's, so the handler went through the funnel, never around it.
        out = _serve(_request(), executor=FakeExecutor(reply="from-funnel-only"))
        assert out.get("content") == "from-funnel-only"

    def test_reply_is_a_json_safe_wire_dict(self):
        import json

        out = _serve(_request())
        json.dumps(out)  # raises if not JSON-safe
        assert out.get("type") == "remote_infer"


# ---------------------------------------------------------------------------
# Family 5 -- the Bulbe refusal (physical, audit-chained, propagates)
# ---------------------------------------------------------------------------


class TestBulbeRefusal:
    def test_bulbe_refuses_physically(self, monkeypatch):
        mod = _load_remote_inference()
        assert mod is not None
        monkeypatch.setattr(guard, "bulbe_disabled", lambda: True)
        with pytest.raises(guard.VeilidDisabledInBulbe):
            mod.serve_remote_inference(
                _request(),
                peer_id="phone-A",
                peer_store=_granted_store(),
                executor=FakeExecutor(),
                router=_fake_router,
                audit=_AuditSpy(),
            )

    def test_bulbe_refusal_is_audit_chained(self, monkeypatch):
        mod = _load_remote_inference()
        assert mod is not None
        monkeypatch.setattr(guard, "bulbe_disabled", lambda: True)
        spy = _AuditSpy()
        with pytest.raises(guard.VeilidDisabledInBulbe):
            mod.serve_remote_inference(
                _request(),
                peer_id="phone-A",
                peer_store=_granted_store(),
                executor=FakeExecutor(),
                router=_fake_router,
                audit=spy,
            )
        actions = [a for a, _ in spy.events]
        assert any("refus" in a for a in actions), "Bulbe refusal must be audit-chained"


# ---------------------------------------------------------------------------
# Family 6 -- the audit chain (serve and refusal ride the same trail)
# ---------------------------------------------------------------------------


class TestAuditChain:
    def test_served_request_is_audit_chained(self):
        spy = _AuditSpy()
        _serve(_request(), audit=spy)
        actions = [a for a, _ in spy.events]
        assert any("serve" in a for a in actions)

    def test_refusal_is_audit_chained(self):
        spy = _AuditSpy()
        _serve(_request(config={"x": 1}), audit=spy)
        actions = [a for a, _ in spy.events]
        assert any("refus" in a for a in actions)

    def test_default_audit_rides_the_hash_chain(self):
        # the module's default audit path uses the same chain_log trail
        # serve_request writes to.
        src = _read("opti_oignon/veilid/remote_inference.py")
        assert "chain_log" in src


# ---------------------------------------------------------------------------
# Family 7 -- the dispatch wiring in serve_app_call
# ---------------------------------------------------------------------------


class FakeEngine:
    def __init__(self) -> None:
        self.served = []

    def serve_request(self, request, *, peer_id=""):
        self.served.append((request, peer_id))
        return {
            "v": protocol.PROTOCOL_VERSION,
            "type": protocol.MSG_RECORD_BATCH,
            "device": "desk",
            "high_water": 0,
            "records": [],
        }


class TestDispatchWiring:
    def test_remote_infer_routed_to_handler_not_to_sync(self):
        eng = FakeEngine()
        msg = transport._encode_message(_request())
        reply = transport.serve_app_call(eng, msg, peer_id="phone-A")
        out = transport.decode_answer(reply)
        # the inference kind must not be served as a sync delta.
        assert eng.served == [], "remote_infer must not reach the sync responder"
        assert out is not None and out.get("type") == "remote_infer"

    def test_delta_request_still_routed_to_sync(self):
        eng = FakeEngine()
        delta = {
            "v": protocol.PROTOCOL_VERSION,
            "type": protocol.MSG_DELTA_REQUEST,
            "device": "phone-A",
            "watermark": 0,
        }
        msg = transport._encode_message(delta)
        reply = transport.serve_app_call(eng, msg, peer_id="phone-A")
        out = transport.decode_answer(reply)
        assert len(eng.served) == 1, "the sync delta must still reach serve_request"
        assert out is not None and out.get("type") == protocol.MSG_RECORD_BATCH


# ---------------------------------------------------------------------------
# Family 8 -- VeilidTimeout propagation (the inference request rides the
#             same fail-secure transport); green on the pristine tree
# ---------------------------------------------------------------------------


class TestVeilidTimeoutPropagation:
    def test_stalled_peer_surfaces_veilid_timeout(self, monkeypatch):
        monkeypatch.setattr(guard, "bulbe_disabled", lambda: False)

        class StallMessenger:
            def call(self, routing_key, payload, *, timeout=None):
                raise guard.VeilidTimeout("peer stalled")

        peer = transport.VeilidPeer(StallMessenger(), "routekey-1", device="desk")
        with pytest.raises(guard.VeilidTimeout):
            peer.fetch(_request())


# ---------------------------------------------------------------------------
# Family 9 -- seam pins on the surfaces the handler builds on; green by design
# ---------------------------------------------------------------------------


class TestSeamPins:
    def test_guard_bulbe_gate_present(self):
        assert hasattr(guard, "assert_sync_allowed")
        assert issubclass(guard.VeilidDisabledInBulbe, guard.VeilidError)
        assert issubclass(guard.VeilidTimeout, guard.VeilidError)

    def test_transport_responder_present(self):
        assert hasattr(transport, "serve_app_call")
        assert hasattr(transport, "decode_answer")
        assert hasattr(transport, "VeilidPeer")

    def test_executor_funnel_present_in_source(self):
        src = _read("opti_oignon/executor.py")
        assert "def execute_simple" in src
        assert "_governor_admit" in src

    def test_peer_store_grant_lookup_present(self):
        peers = importlib.import_module("opti_oignon.veilid.peers")
        assert hasattr(peers.PeerStore, "get_peer")
        assert hasattr(peers, "get_peer_store")

    def test_admission_contract_present_in_source(self):
        src = _read("opti_oignon/resource_governor.py")
        assert "class AdmissionDecision" in src
        assert "def admit" in src
