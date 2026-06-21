#!/usr/bin/env python3
"""Live Veilid transport for sync (S181 Goal 1, Theme 4 / Veilid Sync).

The piece that turns the transport-agnostic protocol envelope into a real
exchange between a user's own devices. S179 made a round a pull against a
duck-typed peer (``protocol.Peer.fetch(request) -> dict``); S180 ran that round
behind an engine and a route with the peer still injected as a fake. This module
supplies the live peer: a ``VeilidPeer`` that satisfies the same ``fetch``
contract by driving the node and client across a private route to a paired peer's
public routing key, and the production resolver the route uses to build one.

A live ``fetch`` is one request/response over a Veilid private route. The request
dict is serialised to bytes, sent to the peer's routing key through a route
messenger (the async client's ``app_call`` in production), and the reply bytes are
parsed back into a dict for the protocol layer to apply. The peer answers from its
own change feed via the responder half (see ``sync_engine.serve_request`` and
``serve_app_call`` below), so a paired device both pulls and serves.

Three disciplines hold here, the same ones the envelope and the engine enforce:

- Bulbe: ``fetch`` calls the binding-layer gate before it acts, so a live round
  refuses under Bulbe at the binding layer -- and the node will not bind under
  Bulbe anyway, so there is no live route to open. The responder refuses to answer
  a peer under Bulbe at the same gate (it delegates to ``serve_request``, which
  gates).
- The DoS bound: every send is timeout-bounded by the client's timeouts. A stalled
  or hostile peer surfaces as ``VeilidTimeout`` rather than wedging the caller; the
  error propagates so the route can map it to a clear status. Only the parse of the
  peer's *answer* is defensive (a malformed reply yields ``None``, which the engine
  treats as an empty round and holds the watermark), never the transport error.
- Incoming answers are data: the reply is parsed defensively and then handed to the
  protocol decoder, whose integrity check still applies, so a poisoned record is
  dropped before it is reconciled.

The messenger is a thin seam (a ``call`` that takes a routing key and request bytes
and returns reply bytes), so the live peer is exercised with a fake messenger -- and
the production messenger with a fake client -- without the veilid framework or a
live server. Kerckhoffs: a peer is addressed by a public routing key the user
holds; nothing about a round depends on the secrecy of the mechanism.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from opti_oignon.veilid.guard import assert_sync_allowed, veilid_available
from opti_oignon.veilid.protocol import MSG_REMOTE_INFER, MSG_REMOTE_INFER_CONT

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True


def _encode_message(obj: Any) -> bytes:
    """Serialise a request or batch dict to compact UTF-8 JSON bytes.

    Producer side; raises if the object is not JSON-safe (a programmer error,
    never untrusted wire data).
    """
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def decode_answer(raw: Any) -> dict | None:
    """Parse a peer's reply (bytes, str, or an already-decoded dict) defensively.

    Returns the decoded dict, or ``None`` on anything malformed -- never raises.
    The protocol layer treats ``None`` as an empty answer and holds the watermark,
    so a garbled reply degrades to a no-op round rather than an error. A dict is
    passed through (a fake messenger may answer with one directly); bytes and str
    are JSON-decoded.
    """
    try:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw).decode("utf-8")
        if isinstance(raw, str):
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        return None
    except Exception:
        logger.debug("Rejected an unparseable peer reply", exc_info=True)
        return None


class RouteMessenger(Protocol):
    """The minimal seam a live peer drives: one request/response over a route.

    ``call`` takes the peer's public routing key and the request bytes and returns
    the reply (bytes, str, or a decoded dict). It is timeout-bounded and
    fail-secure: a stall must surface as ``VeilidTimeout``. A fake messenger stands
    in for the client in tests.
    """

    def call(
        self, routing_key: str, payload: bytes, *, timeout: float | None = None
    ) -> Any:
        ...


class ClientRouteMessenger:
    """A route messenger backed by the async client's ``app_call``.

    The production messenger: it forwards a request/response to the loop-bridge
    client, which submits it to the dedicated Veilid loop, bounded by its timeout.
    The client owns the fail-secure behaviour (a stall is a ``VeilidTimeout``, an
    underlying error a ``VeilidError``), so this wrapper is a thin adapter.
    """

    def __init__(self, client: Any, *, timeout: float | None = None) -> None:
        if client is None:
            raise ValueError("client must not be None")
        self._client = client
        self._timeout = timeout

    def call(
        self, routing_key: str, payload: bytes, *, timeout: float | None = None
    ) -> Any:
        budget = timeout if timeout is not None else self._timeout
        return self._client.app_call(routing_key, payload, timeout=budget)


class VeilidPeer:
    """A live peer reached over a Veilid private route; implements ``Peer.fetch``.

    Holds the route messenger and the peer's public routing key. ``fetch`` is the
    only method the protocol needs: it gates under Bulbe, serialises the request,
    sends it over the route, and parses the reply. A transport error (timeout,
    unavailable) propagates for the route to map; only a malformed reply is
    swallowed (to ``None``), which the engine treats as an empty round.
    """

    def __init__(
        self,
        messenger: RouteMessenger,
        routing_key: str,
        *,
        device: str = "",
        timeout: float | None = None,
    ) -> None:
        if messenger is None:
            raise ValueError("messenger must not be None")
        if not isinstance(routing_key, str) or not routing_key:
            raise ValueError("routing_key must be a non-empty string")
        self._messenger = messenger
        self._routing_key = routing_key
        self._device = device
        self._timeout = timeout

    @property
    def routing_key(self) -> str:
        return self._routing_key

    def fetch(self, request: dict) -> dict | None:
        """Send a delta request to the peer over the route and return its batch.

        Refuses under Bulbe at the binding-layer gate. A ``VeilidTimeout`` or other
        transport error propagates (the route maps it to a status); the reply is
        parsed defensively, so a malformed answer is ``None`` and the round holds.
        """
        assert_sync_allowed()
        payload = _encode_message(request)
        raw = self._messenger.call(self._routing_key, payload, timeout=self._timeout)
        return decode_answer(raw)


def serve_app_call(engine: Any, message: Any, *, peer_id: str = "") -> bytes:
    """Answer an inbound app_call from a peer, as reply bytes.

    The responder bridge over the wire: decode the incoming request defensively,
    discriminate the request kind, and serialise the reply back to bytes.

    Three kinds ride this transport (cas 7, S234/S235). A sync delta (the
    default) is delegated to the engine's gated, audited responder (which draws
    the batch from the local feed via ``respond_to_request``). A remote-inference
    request (``MSG_REMOTE_INFER``) is dispatched to the served inference handler,
    which re-asserts the Bulbe gate, authenticates the route-authenticated peer,
    enforces the tier 1 bounded surface, submits to the executor funnel, and
    returns the first streamed chunk -- all audit-chained. A streaming
    continuation (``MSG_REMOTE_INFER_CONT``, S235) is dispatched to the
    continuation handler, which returns the next buffered chunk for the peer's
    request id. The kind is discriminated by the envelope ``type``; the sync
    responder rejects an unknown type on parse, so the kinds never collide.

    Under Bulbe both paths refuse at the binding-layer gate, so this refuses to
    answer a peer under Bulbe; the error propagates and the live node callback
    simply sends no reply. An unparseable request becomes a benign empty sync
    batch (high-water 0, no records, so it can never advance the asker; PRT-01),
    so a malformed peer cannot crash the responder.
    """
    request = decode_answer(message)
    if isinstance(request, dict):
        kind = request.get("type")
        if kind == MSG_REMOTE_INFER:
            from opti_oignon.veilid.remote_inference import serve_remote_inference

            return _encode_message(serve_remote_inference(request, peer_id=peer_id))
        if kind == MSG_REMOTE_INFER_CONT:
            from opti_oignon.veilid.remote_inference import (
                serve_remote_inference_continuation,
            )

            return _encode_message(
                serve_remote_inference_continuation(request, peer_id=peer_id)
            )
    batch = engine.serve_request(request, peer_id=peer_id)
    return _encode_message(batch)


def resolve_live_peer(
    peer_id: str,
    *,
    store: Any,
    node: Any = None,
    client: Any = None,
    device: str = "",
    timeout: float | None = None,
) -> VeilidPeer | None:
    """Build a live peer for a paired peer, or ``None`` when the transport is down.

    The production resolver the route uses. Returns ``None`` -- which the route maps
    to a transport-unavailable status -- when the veilid framework is absent, the
    peer is not paired, or no attached node and client can supply a route. When a
    client is given (or resolved from an attached node's connector) and the peer has
    a routing key, returns a ``VeilidPeer`` over a ``ClientRouteMessenger``. It opens
    no socket itself: it only assembles the adapter from a live, attached transport.
    """
    if not veilid_available():
        return None
    rec = store.get_peer(peer_id) if store is not None else None
    if rec is None or not getattr(rec, "routing_key", ""):
        return None
    cl = client
    if cl is None:
        nd = node
        if nd is None:
            try:
                from opti_oignon.veilid.node import get_node

                nd = get_node()
            except Exception:  # pragma: no cover - node resolution is defensive
                return None
        try:
            if not nd.is_attached():
                return None
            cl = nd.connector()
        except Exception:  # pragma: no cover - node query is defensive
            return None
    if cl is None:
        return None
    messenger = ClientRouteMessenger(cl, timeout=timeout)
    return VeilidPeer(messenger, rec.routing_key, device=device, timeout=timeout)


def resolve_self_routing_key(*, node: Any = None, client: Any = None) -> str | None:
    """This device's own public routing key for pairing, or ``None`` when unavailable.

    The production source the pairing route uses to mint the routing key it puts in
    this device's pairing payload. Returns ``None`` -- which the route maps to a
    transport-unavailable status -- when the veilid framework is absent or no
    attached node can supply a route. Minting a live self-route over the attached
    node and client lands with the live veilid-server; until then this returns
    ``None`` rather than fabricating a key, and the route's resolver is injectable so
    the pairing surface is exercised with a fixed key in isolation, the same seam as
    ``resolve_live_peer``. It opens no socket and never raises into the caller.

    Under Bulbe the node never attaches, so this returns ``None`` there as well; the
    pairing route reports the key is not yet available rather than refusing, because
    pairing management itself is permitted in any mode -- it simply cannot read a
    live self-route from a node that is not attached.
    """
    if not veilid_available():
        return None
    nd = node
    if nd is None:
        try:
            from opti_oignon.veilid.node import get_node

            nd = get_node()
        except Exception:  # pragma: no cover - node resolution is defensive
            return None
    try:
        if not nd.is_attached():
            return None
    except Exception:  # pragma: no cover - node query is defensive
        return None
    # Live self-route minting over the attached node and client lands with the
    # live veilid-server; until then the route's injectable resolver supplies the
    # key, and production reports the key is not yet available.
    return None
