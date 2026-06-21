"""Served remote-inference handler for cas 7 Lot 1 (S234, REMOTE_INFERENCE_SPEC).

cas 7 lets a paired, lower-trust device -- the phone -- borrow the desktop's
local models without the model, the prompt, or the response leaving the user's
own two machines. This module is the desktop responder side, in the
request-then-single-reply form (no streaming yet; streaming is Lot 2).

The handler does exactly five things and nothing more, in order:

  1. Re-assert the Bulbe binding-layer gate at this seam (defense in depth, the
     same posture ``sync_engine.serve_request`` takes). Bulbe means nothing
     remotely: the node does not even bind in Bulbe, so a remote request cannot
     arrive; if one is processed here under Bulbe anyway, the refusal is
     audit-chained and ``VeilidDisabledInBulbe`` propagates, so the responder
     sends no reply. This is physical, not a policy flag.
  2. Authenticate the request against the route-authenticated peer. The peer
     must be known in the peer store and not pending (PAIR-02 confirmed), and
     the request's claimed origin device must match the route-authenticated
     peer id (a device cannot impersonate another). This is the VL-01 trust
     inheritance for Lot 1: a route-authenticated, confirmed peer whose signing
     public key is registered. Per-record signature verification
     (``verify_record_signature``) is the records/RAG path and lands in Lot 2.
  3. Enforce the tier 1 bounded surface as a single gate, BEFORE any chat
     request is built: inference only in Lot 1. Any out-of-surface field is
     REFUSED with a structured refusal, never silently dropped. RAG-read is a
     SEPARATE SUB-GRANT, off by default in Lot 1 (the conservative default; the
     per-device grant store is deferred to Lot 2), so a request carrying a RAG
     scope is refused.
  4. Build an ordinary chat request -- the same object a local UI request
     builds (analyze + route -> the executor's chat funnel) -- and submit it to
     the executor funnel, which traverses the resource governor's ``admit()``.
     The handler NEVER calls the backend directly and adds NO admission logic of
     its own.
  5. Return the result in the request-then-single-reply form, keyed by the
     request id, as a JSON-safe wire dict; audit-chain the served request.

Every served request and every refusal is audit-chained on the same hash-chain
trail ``serve_request`` writes to (``chain_log``). A stalled or hostile peer
surfaces as ``VeilidTimeout`` on the transport and never wedges the caller.

The grant stance in Lot 1 is the conservative default-tier-1: the per-device
grant store and the differentiated enable/disable + revocation wiring are Lot 2
(REMOTE_INFERENCE_SPEC section 12). Here a confirmed, route-authenticated peer
holds the tier 1 remote-chat grant; the RAG sub-grant is uniformly off.

All seams (the executor funnel, the analyze/route resolver, the peer store, the
audit sink) are injectable so the path is container-provable with fakes and
pulls no ollama import chain; the live route between two devices is host-assured
and named in the spec, never simulated.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from opti_oignon.veilid import remote_streaming
from opti_oignon.veilid.guard import VeilidDisabledInBulbe, assert_sync_allowed
from opti_oignon.veilid.protocol import (
    MSG_REMOTE_INFER,
    MSG_REMOTE_INFER_CONT,
    PROTOCOL_VERSION,
)

logger = logging.getLogger(__name__)

# S73/S74: every new module hardcodes the checkpoint sentinel; never overridable.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True

# The tier 1 bounded surface (REMOTE_INFERENCE_SPEC section 3). A remote request
# may carry ONLY these fields. The allow-set is the gate: any other field is out
# of surface and refused. ``rag`` is within the tier 1 ceiling but gated by a
# per-device sub-grant (cas 7 Lot 2, S235): a request carrying ``rag`` is refused
# unless the asking device's RAG read-only sub-grant is on.
_ALLOWED_FIELDS = frozenset({"v", "type", "device", "request_id", "prompt", "rag"})

# cas 7 Lot 2 (S235): a streaming continuation may carry ONLY these fields. It
# builds no chat -- it reads a chunk from the buffer -- so its surface is the
# request id and the cursor, bound to the route-authenticated device.
_CONT_ALLOWED_FIELDS = frozenset({"v", "type", "device", "request_id", "cursor"})

# Fields that, if present, name a capability the remote surface must never reach
# at tier 1 (state-mutation, sandbox, filesystem, shell, config, mutating
# pipelines). The allow-set above already refuses anything not allowed; this set
# only sharpens the refusal so it names the offending capability class rather
# than reporting a bare unknown field.
_FORBIDDEN_FIELDS = frozenset(
    {
        "tool",
        "tools",
        "manage_memory",
        "manage_skills",
        "sandbox",
        "shell",
        "exec",
        "command",
        "file",
        "fs",
        "path",
        "write",
        "config",
        "settings",
        "pipeline",
    }
)


def _ok(request_id: str, content: str) -> dict:
    """A request-then-single-reply success envelope (a JSON-safe wire dict)."""
    return {
        "v": PROTOCOL_VERSION,
        "type": MSG_REMOTE_INFER,
        "ok": True,
        "request_id": request_id,
        "content": content,
    }


def _stream_reply(
    kind: str, request_id: str, content: str, cursor: int, done: bool
) -> dict:
    """A streaming reply envelope: a chunk, the advanced cursor, the done marker.

    A superset of the Lot 1 single-reply shape: it carries ``ok``, ``request_id``
    and ``content`` (so a single-chunk reply IS the request-then-single-reply
    shape, ``done`` true at once), plus the streaming cursor and the terminal
    done marker (cas 7 Lot 2, Option A). The ``type`` is the request's kind --
    the initial ``remote_infer`` or a ``remote_infer_cont``.
    """
    return {
        "v": PROTOCOL_VERSION,
        "type": kind,
        "ok": True,
        "request_id": request_id,
        "content": content,
        "cursor": cursor,
        "done": done,
    }


def _refusal(request_id: str, reason: str, detail: str, *, kind: str = MSG_REMOTE_INFER) -> dict:
    """A structured refusal envelope -- never a silent failure (section 4).

    ``kind`` is the reply ``type``: the initial ``remote_infer`` by default, or
    ``remote_infer_cont`` for a streaming continuation, so a refusal is routed by
    the same parser as the corresponding success reply.
    """
    return {
        "v": PROTOCOL_VERSION,
        "type": kind,
        "ok": False,
        "refused": True,
        "request_id": request_id,
        "reason": reason,
        "detail": detail,
    }


def _audit_event(audit: Callable[..., Any] | None, action: str, **details: Any) -> None:
    """Record an event on the hash-chain audit log, best-effort.

    Rides the same trail ``serve_request`` writes to. An injected ``audit`` sink
    is used when provided (tests pass a spy); otherwise the default lazily logs
    via ``chain_log`` -- lazy and guarded, the same idiom as the sync engine, so
    it never raises into the serve path.
    """
    if audit is not None:
        try:
            audit(action, **details)
        except Exception:  # pragma: no cover - audit is best-effort
            logger.debug("remote inference audit sink failed", exc_info=True)
        return
    try:
        from opti_oignon.signed_audit_log import chain_log

        chain_log(
            event_type="veilid_remote_infer",
            source="veilid.remote_inference",
            action=action,
            severity="INFO",
            **details,
        )
    except Exception:  # pragma: no cover - audit is best-effort
        logger.debug("remote inference audit log unavailable", exc_info=True)


def _enforce_bounded_surface(
    request: dict, request_id: str, *, rag_granted: bool = False
) -> dict | None:
    """The single tier 1 bounded-surface gate (section 3).

    Returns a structured refusal when the request is out of surface, else
    ``None``. Refuses BEFORE any chat request is built. Any field outside the
    allow-set is refused (the offending capability named when recognised). A RAG
    scope is refused UNLESS the asking device's RAG read-only sub-grant is on
    (cas 7 Lot 2, S235): a device can have remote chat without remote RAG.
    """
    for key in request.keys():
        if key in _ALLOWED_FIELDS:
            continue
        if key in _FORBIDDEN_FIELDS:
            return _refusal(
                request_id,
                "out_of_surface",
                "field '" + str(key) + "' names a capability the tier 1 remote "
                "surface does not reach (state-mutation/sandbox/filesystem/"
                "shell/config); refused, not dropped",
            )
        return _refusal(
            request_id,
            "out_of_surface",
            "field '" + str(key) + "' is not part of the tier 1 bounded surface; "
            "refused, not dropped",
        )
    if "rag" in request and not rag_granted:
        # RAG-read is a separate per-device sub-grant (D-iii). It is off for this
        # device, so the scope is refused, not silently dropped.
        return _refusal(
            request_id,
            "rag_not_granted",
            "remote RAG read is a separate per-device sub-grant and is off for "
            "this device",
        )
    return None


def _resolve_peer_store(peer_store: Any) -> Any:
    """Return the injected peer store, else the process default (fail-secure).

    A resolution failure returns ``None`` so the caller refuses (a grant that
    cannot be confirmed is no grant).
    """
    if peer_store is not None:
        return peer_store
    try:
        from opti_oignon.veilid.peers import get_peer_store

        return get_peer_store()
    except Exception:  # pragma: no cover - defensive; no store -> refuse
        logger.debug("remote inference peer store unavailable", exc_info=True)
        return None


def _resolve_router(router: Callable[[str], Any] | None) -> Callable[[str], Any]:
    """Return the injected router, else a default analyze + route resolver.

    The default is resolved lazily and only on the success path, so a refusal
    never pulls the analyzer/router (and the ollama chain) into the import.
    """
    if router is not None:
        return router

    def _default_router(prompt: str) -> Any:
        from opti_oignon.analyzer import analyze
        from opti_oignon.router import router as model_router

        return model_router.route(analyze(prompt))

    return _default_router


def _resolve_executor(executor: Any) -> Any:
    """Return the injected executor, else the process-default chat funnel.

    Resolved lazily and only on the success path, so a refusal never pulls the
    executor (and the ollama chain) into the import.
    """
    if executor is not None:
        return executor
    from opti_oignon.executor import executor as default_executor

    return default_executor


def _drain_chunks(gen: Any) -> list:
    """Collect a streaming chat generator's YIELDED chunks into a list.

    Each yield is one stream chunk (cas 7 Lot 2, Option A); the end of the list
    is the terminal done marker. A plain string is a single chunk. If the
    generator yields nothing but returns a ``(response, task)`` tuple, the
    response is the single chunk -- so a single-reply funnel streams as one
    chunk (the Lot 1 shape: the first chunk IS the whole reply, done at once).
    """
    if isinstance(gen, str):
        return [gen] if gen else []
    chunks: list = []
    final: Any = None
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        final = stop.value
    except TypeError:
        # not a generator; nothing to drain
        return []
    string_chunks = [c for c in chunks if isinstance(c, str)]
    if string_chunks:
        return string_chunks
    if isinstance(final, tuple) and final and isinstance(final[0], str) and final[0]:
        return [final[0]]
    return []


def _resolve_origin(req: dict, peer_id: str, *, kind: str = MSG_REMOTE_INFER) -> tuple:
    """Resolve the authenticated origin device, or a ``(None, refusal)`` pair.

    The origin is the route-authenticated peer when present; the request's
    claimed device must match it (a device cannot impersonate another). With an
    empty peer id the route is the implicit authenticator and the claimed device
    is the asserted origin; with neither there is no identity to grant. Returns
    ``(origin, None)`` on success, or ``(None, refusal_dict)`` on a
    provenance/identity refusal (typed by ``kind``). Shared by the initial and
    continuation handlers so they authenticate identically.
    """
    request_id = req.get("request_id")
    if not isinstance(request_id, str):
        request_id = ""
    claimed_device = req.get("device")
    if not isinstance(claimed_device, str):
        claimed_device = ""
    if peer_id:
        if claimed_device and claimed_device != peer_id:
            return None, _refusal(
                request_id,
                "provenance_mismatch",
                "the request's claimed device does not match the route-"
                "authenticated peer",
                kind=kind,
            )
        return peer_id, None
    if not claimed_device:
        return None, _refusal(
            request_id,
            "no_authenticated_identity",
            "no route-authenticated peer and no asserted origin device",
            kind=kind,
        )
    return claimed_device, None


def serve_remote_inference(
    request: Any,
    *,
    peer_id: str = "",
    executor: Any = None,
    router: Callable[[str], Any] | None = None,
    peer_store: Any = None,
    audit: Callable[..., Any] | None = None,
) -> dict:
    """Serve a remote chat request from a paired phone over the private route.

    Decodes and validates the request, re-asserts the Bulbe gate, authenticates
    the route-authenticated peer, enforces the tier 1 bounded surface, submits
    an ordinary chat request to the executor funnel (which admits), and returns
    the single reply -- all audit-chained. Never calls the backend directly and
    adds no admission logic. Raises ``VeilidDisabledInBulbe`` under Bulbe (after
    audit-chaining the refusal); every application-layer refusal is returned as a
    structured refusal dict, never a silent failure.
    """
    req = request if isinstance(request, dict) else {}
    request_id = req.get("request_id")
    if not isinstance(request_id, str):
        request_id = ""

    # 1. Bulbe: physical, re-asserted at the seam. Audit then propagate.
    try:
        assert_sync_allowed()
    except VeilidDisabledInBulbe:
        _audit_event(audit, "remote_infer_refused", reason="bulbe", request_id=request_id)
        raise

    # The request must be a well-formed remote-inference envelope.
    if req.get("v") != PROTOCOL_VERSION or req.get("type") != MSG_REMOTE_INFER:
        out = _refusal(request_id, "malformed", "not a remote inference request")
        _audit_event(audit, "remote_infer_refused", reason="malformed", request_id=request_id)
        return out

    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        out = _refusal(request_id, "malformed", "missing or empty prompt")
        _audit_event(audit, "remote_infer_refused", reason="malformed", request_id=request_id)
        return out

    # 2. Authenticate against the route-authenticated peer.
    claimed_device = req.get("device")
    if not isinstance(claimed_device, str):
        claimed_device = ""
    # The origin is the route-authenticated peer when present; the request's
    # claimed device must match it. With an empty peer id the route is the
    # implicit authenticator (the serve_request posture) and the claimed device
    # is the asserted origin; with neither there is no identity to grant.
    if peer_id:
        if claimed_device and claimed_device != peer_id:
            out = _refusal(
                request_id,
                "provenance_mismatch",
                "the request's claimed device does not match the route-"
                "authenticated peer",
            )
            _audit_event(
                audit,
                "remote_infer_refused",
                reason="provenance_mismatch",
                request_id=request_id,
                peer_id=peer_id,
            )
            return out
        origin = peer_id
    else:
        origin = claimed_device

    if not origin:
        out = _refusal(
            request_id,
            "no_authenticated_identity",
            "no route-authenticated peer and no asserted origin device",
        )
        _audit_event(
            audit, "remote_infer_refused", reason="no_authenticated_identity",
            request_id=request_id,
        )
        return out

    store = _resolve_peer_store(peer_store)
    record = store.get_peer(origin) if store is not None else None
    if record is None:
        out = _refusal(
            request_id,
            "unknown_device",
            "the asking device is not a registered peer",
        )
        _audit_event(
            audit, "remote_infer_refused", reason="unknown_device",
            request_id=request_id, peer_id=origin,
        )
        return out
    if getattr(record, "pending", False):
        out = _refusal(
            request_id,
            "peer_not_confirmed",
            "the peer's pairing awaits PAIR-02 mutual confirmation; it grants "
            "nothing, serving included",
        )
        _audit_event(
            audit, "remote_infer_refused", reason="peer_not_confirmed",
            request_id=request_id, peer_id=origin,
        )
        return out

    # 3a. The store-backed per-device remote-chat grant (cas 7 Lot 2). A
    #     disabled device is refused; a record without the column reads as the
    #     grandfathered tier-1 default (enabled), so Lot 1's stance is unchanged.
    if not getattr(record, "remote_chat_enabled", True):
        out = _refusal(
            request_id,
            "remote_chat_disabled",
            "this device's remote chat grant is disabled at the desktop",
        )
        _audit_event(
            audit, "remote_infer_refused", reason="remote_chat_disabled",
            request_id=request_id, peer_id=origin,
        )
        return out

    # 3b. The tier 1 bounded surface, as a single gate, before any chat is built.
    #     The RAG scope is gated by the device's RAG read-only sub-grant.
    rag_granted = bool(getattr(record, "rag_subgrant", False))
    refusal = _enforce_bounded_surface(req, request_id, rag_granted=rag_granted)
    if refusal is not None:
        _audit_event(
            audit,
            "remote_infer_refused",
            reason=refusal.get("reason", "out_of_surface"),
            request_id=request_id,
            peer_id=origin,
        )
        return refusal

    # 3c. The per-device rate gate, just before generation: a device over its
    #     fixed window is refused (an alert recorded in the channel telemetry),
    #     never absorbed work. Called via the module so it stays patchable;
    #     continuations are cheap buffer reads and do not consume the budget.
    if not remote_streaming.check_rate(origin):
        out = _refusal(
            request_id,
            "rate_limited",
            "this device has exceeded its remote inference rate; try again later",
        )
        _audit_event(
            audit, "remote_infer_refused", reason="rate_limited",
            request_id=request_id, peer_id=origin,
        )
        return out

    # 4. Build the ordinary chat request and submit it to the executor funnel
    #    (which traverses admit()); never call the backend directly.
    try:
        resolved_router = _resolve_router(router)
        routing = resolved_router(prompt)
        resolved_executor = _resolve_executor(executor)
        gen = resolved_executor.execute(
            question=prompt,
            routing=routing,
            refine=False,
            web_search=False,
        )
        chunks = _drain_chunks(gen)
    except Exception as exc:
        logger.warning("remote inference funnel failed: %s", exc)
        out = _refusal(
            request_id,
            "execution_error",
            "the desktop could not complete the request",
        )
        _audit_event(
            audit, "remote_infer_refused", reason="execution_error",
            request_id=request_id, peer_id=origin,
        )
        return out

    # 5. Buffer the chunks keyed by (peer, request id) and return the first one
    #    (cas 7 Lot 2, Option A). The phone pulls the rest with remote_infer_cont
    #    requests. A single-chunk reply is done at once -- the first chunk IS the
    #    whole reply -- which is the Lot 1 request-then-single-reply shape.
    remote_streaming.open_session(origin, request_id, chunks)
    first = remote_streaming.pull(origin, request_id, 0)
    if first is None:
        # an empty generation (no chunks): an honest empty, done reply
        content, cursor, done = "", 0, True
    else:
        content, cursor, done = first["content"], first["cursor"], first["done"]
    out = _stream_reply(MSG_REMOTE_INFER, request_id, content, cursor, done)
    _audit_event(
        audit, "remote_infer_serve", request_id=request_id, peer_id=origin,
        chars=len(content), chunks=len(chunks),
    )
    return out


def serve_remote_inference_continuation(
    request: Any,
    *,
    peer_id: str = "",
    peer_store: Any = None,
    audit: Callable[..., Any] | None = None,
) -> dict:
    """Serve a streaming continuation: return the next buffered chunk (Lot 2).

    The phone pulls the remainder of a streamed reply with ``remote_infer_cont``
    requests, each carrying the request id and a cursor. The handler re-asserts
    the Bulbe gate, re-authenticates the route peer (an unknown, pending, or
    revoked device is refused), and reads the chunk at the cursor from the buffer
    keyed by the ``(route-authenticated peer, request id)`` pair. A miss -- an
    unknown request id, or a request id owned by a DIFFERENT device -- is a
    ``buffer_mismatch`` refusal, never a cross-read (the section-14 risk). All
    audit-chained; raises ``VeilidDisabledInBulbe`` under Bulbe.
    """
    req = request if isinstance(request, dict) else {}
    request_id = req.get("request_id")
    if not isinstance(request_id, str):
        request_id = ""

    def _ref(reason: str, detail: str) -> dict:
        # a continuation reply, success or refusal, carries the continuation type
        return _refusal(request_id, reason, detail, kind=MSG_REMOTE_INFER_CONT)

    # 1. Bulbe: physical, re-asserted at the seam. Audit then propagate.
    try:
        assert_sync_allowed()
    except VeilidDisabledInBulbe:
        _audit_event(audit, "remote_infer_refused", reason="bulbe", request_id=request_id)
        raise

    # 2. A well-formed continuation envelope, the continuation surface only.
    if req.get("v") != PROTOCOL_VERSION or req.get("type") != MSG_REMOTE_INFER_CONT:
        _audit_event(audit, "remote_infer_refused", reason="malformed", request_id=request_id)
        return _ref("malformed", "not a remote inference continuation")
    if not request_id:
        _audit_event(audit, "remote_infer_refused", reason="malformed", request_id=request_id)
        return _ref("malformed", "missing request id")
    cursor = req.get("cursor")
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        _audit_event(audit, "remote_infer_refused", reason="malformed", request_id=request_id)
        return _ref("malformed", "missing or invalid cursor")
    for key in req.keys():
        if key not in _CONT_ALLOWED_FIELDS:
            _audit_event(
                audit, "remote_infer_refused", reason="out_of_surface",
                request_id=request_id,
            )
            return _ref(
                "out_of_surface",
                "field '" + str(key) + "' is not part of the continuation "
                "surface; refused, not dropped",
            )

    # 3. Re-authenticate the route peer (identical to the initial handler).
    origin, refusal = _resolve_origin(req, peer_id, kind=MSG_REMOTE_INFER_CONT)
    if refusal is not None:
        _audit_event(
            audit, "remote_infer_refused",
            reason=refusal.get("reason", "provenance_mismatch"),
            request_id=request_id, peer_id=peer_id,
        )
        return refusal

    store = _resolve_peer_store(peer_store)
    record = store.get_peer(origin) if store is not None else None
    if record is None:
        _audit_event(
            audit, "remote_infer_refused", reason="unknown_device",
            request_id=request_id, peer_id=origin,
        )
        return _ref("unknown_device", "the asking device is not a registered peer")
    if getattr(record, "pending", False):
        _audit_event(
            audit, "remote_infer_refused", reason="peer_not_confirmed",
            request_id=request_id, peer_id=origin,
        )
        return _ref(
            "peer_not_confirmed",
            "the peer's pairing awaits PAIR-02 mutual confirmation",
        )
    if not getattr(record, "remote_chat_enabled", True):
        # a grant revoked mid-stream: the durable refusal (the live buffer was
        # also dropped, so the pull below would miss regardless).
        _audit_event(
            audit, "remote_infer_refused", reason="remote_chat_disabled",
            request_id=request_id, peer_id=origin,
        )
        return _ref(
            "remote_chat_disabled",
            "this device's remote chat grant is disabled at the desktop",
        )

    # 4. Pull the chunk at the cursor; a miss is a bounded-buffer mismatch, never
    #    a cross-read. The lookup is bound to the route peer AND the request id.
    chunk = remote_streaming.pull(origin, request_id, cursor)
    if chunk is None:
        _audit_event(
            audit, "remote_infer_refused", reason="buffer_mismatch",
            request_id=request_id, peer_id=origin,
        )
        return _ref(
            "buffer_mismatch",
            "no in-flight stream for this device and request id at this cursor",
        )
    out = _stream_reply(
        MSG_REMOTE_INFER_CONT, request_id, chunk["content"], chunk["cursor"], chunk["done"]
    )
    _audit_event(
        audit, "remote_infer_serve", request_id=request_id, peer_id=origin,
        chars=len(chunk["content"]), cursor=chunk["cursor"], done=chunk["done"],
    )
    return out
