#!/usr/bin/env python3
"""Live API route for Veilid sync (Goal 3, Theme 4 / Veilid Sync).

Wires the sync engine behind a thin, guarded FastAPI surface, the same shape as
the agent route: a web-free core that takes resolved dependencies and returns
plain payloads, and a thin FastAPI wrapper over it that maps faults to HTTP
codes. The contract the eventual sync panel consumes:

- ``GET  /api/sync/peers``                       -> ``{peers: [...]}``
- ``GET  /api/sync/peers/{peer_id}``             -> one peer's status
- ``GET  /api/sync/peers/{peer_id}/watermark``   -> ``{peer_id, watermark}``
- ``POST /api/sync/peers/{peer_id}/run``         -> run a pull round; the round summary
- ``GET  /api/sync/pairing/pending``             -> pairings awaiting confirmation (PAIR-02)
- ``POST /api/sync/pairing/pending/{peer_id}/confirm`` -> activate a pending pairing
- ``POST /api/sync/pairing/pending/{peer_id}/reject``  -> remove a pending pairing
- ``GET  /api/sync/deferred``                    -> pending content approvals (SYN-05)
- ``POST /api/sync/deferred/approve``            -> apply a deferred record through the seam
- ``POST /api/sync/deferred/refuse``             -> remove a deferred record; nothing applies

The list, status, and watermark surfaces read the per-peer store and work now.
The run surface drives a full pull round through the engine; the live Veilid
transport that supplies the peer over a private route comes later, so the
handler resolves no live peer yet and reports that the transport is unavailable.
The web-free ``run_sync_payload`` takes an injected peer, so the round contract is
exercised in isolation with a fake peer without the framework or a live node.

Two disciplines hold at the route seam. Bulbe: sync is a Daily-only capability,
so the run handler refuses under Bulbe through the binding-layer gate, mapped to a
403 the same way the security-mode middleware refuses a Bulbe-blocked capability;
the engine re-asserts the gate at its own seam, so the refusal is enforced, not a
handler policy. Approval: a sensitive apply over sync (a skill record) goes through
the same human gate as the agent's ``manage_skills`` and ``manage_memory`` writes,
reused verbatim -- the run handler passes no override, so the engine consults the
existing manager-backed ``allowlists.request_approval``. Pairing peers (add /
remove) belong to the panel; this module exposes the read and run surfaces.

The web-free helpers are defined at module level, outside the FastAPI block, so the
contract is importable and testable where fastapi is absent (the sandbox). The
FastAPI wrapper is guarded so the module loads without the framework; the veilid
imports are guarded too, so a partial build cannot block app startup.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Emergency-stop admission guard (a stopped system refuses honestly)
try:
    from opti_oignon import emergency_stop as _emergency_stop
except Exception:  # pragma: no cover - constrained environments only
    _emergency_stop = None  # type: ignore[assignment]

# Guarded veilid imports (the sub-package is lightweight and collects without the
# fastapi or veilid frameworks; guarded anyway so a partial build degrades to 503).
try:
    from opti_oignon.veilid import guard as _guard
    from opti_oignon.veilid import pairing as _pairing
    from opti_oignon.veilid import remote_streaming as _remote_streaming
    from opti_oignon.veilid import transport as _transport
    from opti_oignon.veilid.node import get_node
    from opti_oignon.veilid.peers import (
        DEVICE_CLASS_DESKTOP,
        PeerStore,
        get_peer_store,
    )
    from opti_oignon.veilid.sync_engine import (
        DeferredNotFound,
        PeerNotConfirmed,
        PeerNotFound,
        SyncEngine,
        get_sync_engine,
    )
    from opti_oignon.veilid.sync_status import get_sync_status_store

    _SYNC_OK = True
except Exception:  # pragma: no cover - constrained environments only
    _guard = None  # type: ignore[assignment]
    _pairing = None  # type: ignore[assignment]
    _remote_streaming = None  # type: ignore[assignment]
    _transport = None  # type: ignore[assignment]
    get_node = None  # type: ignore[assignment]
    DEVICE_CLASS_DESKTOP = None  # type: ignore[assignment]
    PeerStore = None  # type: ignore[assignment]
    get_peer_store = None  # type: ignore[assignment]
    SyncEngine = None  # type: ignore[assignment]
    get_sync_engine = None  # type: ignore[assignment]
    get_sync_status_store = None  # type: ignore[assignment]
    _SYNC_OK = False

    class PeerNotFound(Exception):  # type: ignore[no-redef]
        """Fallback so the symbol exists where the veilid package is absent."""

    class PeerNotConfirmed(Exception):  # type: ignore[no-redef]
        """Fallback so the symbol exists where the veilid package is absent."""

    class DeferredNotFound(Exception):  # type: ignore[no-redef]
        """Fallback so the symbol exists where the veilid package is absent."""


# Web-free payload helpers (module level; thin handlers wrap these)
#
# Each takes a resolved store or engine and returns a plain payload, so the
# contract is exercised in isolation without the web stack. A missing peer raises
# PeerNotFound, which the FastAPI layer maps to 404; the run helper propagates the
# engine's PeerNotFound and its Bulbe refusal for the same mapping.


def _peer_to_dict(rec: Any) -> dict[str, Any]:
    """Serialise a stored peer for the wire (the public routing key included).

    PAIR-02: ``pending`` flags an entry awaiting the mutual
    confirmation (it gates nothing until confirmed) and ``key_changed``
    distinguishes the demotion case -- a re-pair carried a different signing
    key -- from a fresh pairing, so the panel can say why a confirmation is
    being asked again. PAIR-03: ``device_class`` rides too (``None``
    for the grandfathered NULL), so the pending panel shows the RECORDED
    class next to the confirmation code -- the human-visible compensating
    control for the code's deliberate class exclusion -- and the later UI
    lot has its read surface. getattr-defensive, so a pre-PAIR-02 record
    still shapes.
    """
    return {
        "peer_id": rec.peer_id,
        "routing_key": rec.routing_key,
        "label": rec.label,
        "watermark": int(rec.watermark),
        "added_at": rec.added_at,
        "updated_at": rec.updated_at,
        "pending": bool(getattr(rec, "pending", False)),
        "key_changed": bool(getattr(rec, "key_changed", False)),
        "device_class": getattr(rec, "device_class", None),
    }


def list_peers_payload(store: Any) -> dict[str, Any]:
    """The paired-peers index payload."""
    return {"peers": [_peer_to_dict(p) for p in store.list_peers()]}


def peer_status_payload(
    store: Any, peer_id: str, status_store: Any = None
) -> dict[str, Any]:
    """One peer's status; raises PeerNotFound when it is not paired.

    When a status store is supplied, the payload is enriched with the peer's
    last-sync time (the timestamp of its last successful round, empty when none or
    when the last attempt failed) and the full outcome of its last round. With no
    status store the payload is exactly the stored peer record, so the
    contract is unchanged.
    """
    rec = store.get_peer(peer_id)
    if rec is None:
        raise PeerNotFound(peer_id)
    out = _peer_to_dict(rec)
    if status_store is not None:
        last = status_store.last_for(peer_id)
        out["last_sync"] = last.at if (last is not None and last.ok) else ""
        out["last_round"] = _outcome_to_dict(last)
    return out


def _outcome_to_dict(outcome: Any) -> dict[str, Any] | None:
    """Shape a RoundOutcome into a JSON-safe payload, or None when there is none."""
    if outcome is None:
        return None
    return {
        "peer_id": outcome.peer_id,
        "applied": int(outcome.applied),
        "deferred": int(outcome.deferred),
        "conflicts": int(outcome.conflicts),
        "rejected": int(outcome.rejected),
        "refused": int(getattr(outcome, "refused", 0)),
        "unverified": int(getattr(outcome, "unverified", 0)),
        "previous_watermark": int(outcome.previous_watermark),
        "new_watermark": int(outcome.new_watermark),
        "advanced": bool(outcome.advanced),
        "at": outcome.at,
        "ok": bool(outcome.ok),
        "error": outcome.error,
    }


def status_payload(*, node: Any, store: Any, status_store: Any) -> dict[str, Any]:
    """The sync-status surface: whether sync is running, the last round, per peer.

    The transport block (running, attached, the attachment state, the Bulbe and
    framework flags) comes from the node's status snapshot, the live source of
    truth for whether sync can run now; querying it is best-effort and never raises.
    The last round across all peers and each peer's last-sync time and last round
    come from the in-memory status store. Web-free: it takes resolved dependencies,
    so it is exercised in isolation with a fake node and an injected status store.
    """
    node_status: dict[str, Any] = {}
    if node is not None:
        try:
            node_status = dict(node.status() or {})
        except Exception:  # pragma: no cover - node query is best-effort
            node_status = {}
    peers_out: list[dict[str, Any]] = []
    for p in store.list_peers():
        d = _peer_to_dict(p)
        last = status_store.last_for(p.peer_id) if status_store is not None else None
        d["last_sync"] = last.at if (last is not None and last.ok) else ""
        d["last_round"] = _outcome_to_dict(last)
        peers_out.append(d)
    last_round = status_store.last_round() if status_store is not None else None
    return {
        "running": bool(node_status.get("running", False)),
        "attached": bool(node_status.get("attached", False)),
        "attachment": node_status.get("attachment", ""),
        "bulbe_disabled": bool(node_status.get("bulbe_disabled", False)),
        "veilid_available": bool(node_status.get("veilid_available", False)),
        "last_round": _outcome_to_dict(last_round),
        "peers": peers_out,
    }


def peer_watermark_payload(store: Any, peer_id: str) -> dict[str, Any]:
    """One peer's watermark; raises PeerNotFound when it is not paired."""
    rec = store.get_peer(peer_id)
    if rec is None:
        raise PeerNotFound(peer_id)
    return {"peer_id": peer_id, "watermark": int(rec.watermark)}


def round_result_to_dict(result: Any) -> dict[str, Any]:
    """Shape a RoundResult into a JSON-safe payload. Pure.

    VL-01: ``refused`` (signature-seam refusals) and ``unverified``
    (records accepted without verification -- the verify-incapable posture;
    the historical unkeyed-origin grace closed) join the payload
    additively, the
    PRT-03 reject-surfacing idiom extended, so SyncPanel can show them; read
    defensively so a pre-VL-01 result still shapes.
    """
    return {
        "peer_id": result.peer_id,
        "applied": int(result.applied),
        "deferred": int(result.deferred),
        "conflicts": int(result.conflicts),
        "rejected": int(result.rejected),
        "refused": int(getattr(result, "refused", 0)),
        "unverified": int(getattr(result, "unverified", 0)),
        "previous_watermark": int(result.previous_watermark),
        "new_watermark": int(result.new_watermark),
        "advanced": bool(result.advanced),
        "parsed": bool(getattr(result, "parsed", True)),
    }


def run_sync_payload(
    engine: Any,
    peer_id: str,
    peer: Any,
    *,
    approval_fn: Any = None,
    conversation_id: str = "",
    approval_manager: Any = None,
) -> dict[str, Any]:
    """Run one pull round through the engine and shape the summary.

    Propagates the engine's PeerNotFound (an unpaired peer) and its Bulbe refusal
    (VeilidDisabledInBulbe) to the caller for HTTP mapping. The peer is injected,
    so this is the full round contract without a live transport.
    """
    result = engine.run_round(
        peer_id,
        peer,
        approval_fn=approval_fn,
        conversation_id=conversation_id,
        approval_manager=approval_manager,
    )
    return round_result_to_dict(result)


# Live-peer resolution (web-free; injectable so the run surface is testable)
#
# The run handler resolves a live peer over the Veilid transport. In production
# that is transport.resolve_live_peer, which returns None -- a transport-unavailable
# status -- when the framework is absent or no attached node and client can supply a
# route. A test injects a resolver via set_peer_resolver so the run surface is driven
# end to end with a fake peer; reset_peer_resolver restores the production path.

_PEER_RESOLVER: Any = None


def set_peer_resolver(resolver: Any) -> None:
    """Install a live-peer resolver (``peer_id, store -> Peer | None``) for tests."""
    global _PEER_RESOLVER
    _PEER_RESOLVER = resolver


def reset_peer_resolver() -> None:
    """Restore the production live-peer resolver."""
    global _PEER_RESOLVER
    _PEER_RESOLVER = None


def resolve_peer_for_route(peer_id: str, store: Any) -> Any:
    """Resolve a live peer for the run handler; ``None`` when the transport is down.

    Uses an injected resolver when one is set, else the production transport
    resolver. Defensive: a resolver failure degrades to ``None`` (a clean
    transport-unavailable status) rather than a 500.
    """
    if _PEER_RESOLVER is not None:
        return _PEER_RESOLVER(peer_id, store)
    if not _SYNC_OK or _transport is None:
        return None
    try:
        return _transport.resolve_live_peer(peer_id, store=store)
    except Exception:  # pragma: no cover - resolution is defensive
        logger.exception("live peer resolution failed")
        return None


# Pairing surface (web-free; the ceremony that populates the peer store)
#
# Pairing management -- generating this device's payload, accepting a peer's
# payload, listing / labelling / removing peers -- is local-disk and permitted in
# any mode, like the peer store. Only running a round or serving a peer is
# Daily-only, and that gate lives at the binding layer in the engine and the
# transport, not here. The helpers are web-free, so the contract is exercised in
# isolation with a fake engine and a fixed routing key, without the framework.


class InvalidPairing(Exception):
    """A pairing payload was malformed or its integrity check did not match."""


def _audit_pairing(action: str, **details: Any) -> None:
    """Record a pairing event in the hash-chain audit log, best-effort.

    Generating this device's payload is audited here; accepting a peer's payload is
    audited by the engine's ``register_peer``. Lazy and guarded so it never raises.
    """
    try:
        from opti_oignon.signed_audit_log import chain_log

        chain_log(
            event_type="veilid_sync",
            source="veilid.pairing",
            action=action,
            severity="INFO",
            **details,
        )
    except Exception:  # pragma: no cover - audit is best-effort
        logger.debug("pairing audit log unavailable", exc_info=True)


def self_pairing_payload(
    peer_id: str,
    routing_key: str,
    signing_pub: str | None = None,
    device_class: str | None = None,
) -> dict[str, Any]:
    """Build this device's pairing payload plus the JSON text for a QR / transcription.

    Pure: it delegates to the pairing module's pure builder, which computes the
    integrity check over the public material, and returns both the structured
    payload and its compact JSON form (what a QR encodes). The device's
    signing PUBLIC key (VL-01) is folded in when given -- closing the
    gap where the module-level extension existed but the production surface
    never threaded it, leaving every real pairing unkeyed -- and ``None``
    keeps the honest pre-VL-01 payload (a device that cannot sign still
    pairs). PAIR-03: the device class is threaded the same way, in
    the SAME session as the module half (the lesson applied forward);
    ``None`` keeps the class-less payload and its historical digest. Raises
    ``ValueError`` on an empty identity or key or an out-of-vocabulary class
    (a programmer error), never on untrusted input.
    """
    if _pairing is None:  # pragma: no cover - constrained environments only
        raise InvalidPairing("pairing module not available")
    payload = _pairing.build_pairing_payload(
        peer_id, routing_key, signing_pub, device_class=device_class
    )
    return {
        "peer_id": peer_id,
        "routing_key": routing_key,
        "payload": payload,
        "text": _pairing.encode_pairing_json(payload),
    }


def pairing_confirmation_code_for(store: Any, rec: Any) -> str | None:
    """The PAIR-02 confirmation code for a pending peer, or ``None``.

    Recomputed from local disk: the peer's stored public material (its
    identity, routing key, and signing key, exactly what the registry holds)
    and this device's own material pinned when its payload was generated. With
    no pinned self material the code cannot exist yet -- the human has not
    shown this device's payload -- so the surface answers ``None`` and the
    panel says to generate the pairing code first. Defensive: any derivation
    failure degrades to ``None``, never a 500 into the pending list.
    """
    if _pairing is None:  # pragma: no cover - constrained environments only
        return None
    try:
        getter = getattr(store, "get_self_pairing_material", None)
        self_material = getter() if callable(getter) else None
        if not self_material:
            return None
        peer_material = _pairing.pairing_canonical_material(
            rec.peer_id,
            rec.routing_key,
            getattr(rec, "signing_pub", None),
        )
        return _pairing.confirmation_code(self_material, peer_material)
    except Exception:  # pragma: no cover - derivation is defensive here
        logger.debug("confirmation code derivation failed", exc_info=True)
        return None


def pending_pairings_payload(store: Any) -> dict[str, Any]:
    """The pending-pairings surface (PAIR-02): entries awaiting confirmation.

    Each entry is the peer dict plus its ``confirmation_code`` (or ``None``
    when this device has not generated its own payload yet); ``self_ready``
    says whether the pinned self material exists, so the panel can hint at the
    missing half instead of showing an inexplicable absent code. Local-disk,
    permitted in any mode, like the rest of pairing management.
    """
    getter = getattr(store, "get_self_pairing_material", None)
    self_ready = bool(getter()) if callable(getter) else False
    pending_out: list[dict[str, Any]] = []
    for p in store.list_peers():
        if not bool(getattr(p, "pending", False)):
            continue
        d = _peer_to_dict(p)
        d["confirmation_code"] = pairing_confirmation_code_for(store, p)
        pending_out.append(d)
    return {"pending": pending_out, "self_ready": self_ready}


def accept_pairing(
    engine: Any, obj: Any, *, label: str = "", store: Any = None
) -> dict[str, Any]:
    """Accept a peer's pairing payload through the engine; raise InvalidPairing on a miss.

    Parses the payload defensively and, on a valid one, registers the peer via the
    engine's ``register_peer`` (the upsert that preserves the watermark on a re-pair
    and audits the registration). PAIR-02: the ceremony registers the
    peer PENDING; the returned dict carries the ``pending`` flag and, when a
    ``store`` is supplied, the ``confirmation_code`` both humans must compare
    (``None`` when this device's own payload was never generated -- the panel
    hints at the missing half). PAIR-03: the resolved ``store`` is
    threaded into the seam so it can read the PRIOR row state and record the
    payload's declared device class under the monotone rule -- best-effort,
    never voiding the registration. Raises :class:`InvalidPairing` on anything
    malformed or tampered, which the FastAPI layer maps to a 400; the integrity
    check rejects a garbled or altered payload before it is ever stored.
    """
    if _pairing is None:  # pragma: no cover - constrained environments only
        raise InvalidPairing("pairing module not available")
    rec = _pairing.accept_pairing_payload(engine, obj, label=label, store=store)
    if rec is None:
        raise InvalidPairing("invalid or tampered pairing payload")
    out = _peer_to_dict(rec)
    if store is not None and out.get("pending"):
        out["confirmation_code"] = pairing_confirmation_code_for(store, rec)
    return out


def relabel_peer_payload(store: Any, engine: Any, peer_id: str, label: str) -> dict[str, Any]:
    """Relabel a paired peer through the engine's watermark-preserving upsert.

    Raises :class:`PeerNotFound` when the peer is not paired. Reuses the stored
    routing key and re-registers with the new label, so the watermark and the
    original pairing time are preserved (the upsert in the peer store). Local-disk,
    permitted in any mode.
    """
    rec = store.get_peer(peer_id)
    if rec is None:
        raise PeerNotFound(peer_id)
    updated = engine.register_peer(peer_id, rec.routing_key, label=str(label or ""))
    return _peer_to_dict(updated)


def _normalise_device_class(value: Any) -> str | None:
    """Normalise a control-surface device-class value to the allowlist.

    The strict pure half of the setter leg: ``None`` and a blank
    string are an explicit CLEAR (the store's documented ``None``, the
    grandfathered desktop class); ``"phone"`` / ``"desktop"`` pass
    case-insensitively after stripping; anything else -- free text, an
    unknown vocabulary, a non-string -- raises ``ValueError`` BEFORE the
    value can reach the engine or the store (the route maps it to a 400).
    Deliberately mirrors the registry-side ``peers.DEVICE_CLASSES``
    allowlist without importing it: the store still validates on write
    (defense in depth, two independent gates), and this wire-side gate
    stays importable where the veilid package is absent.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("device_class must be a string or null")
    candidate = value.strip().lower()
    if not candidate:
        return None
    if candidate in {"phone", "desktop"}:
        return candidate
    raise ValueError("device_class must be one of ['desktop', 'phone'] or null")


def set_device_class_payload(
    store: Any, engine: Any, peer_id: str, device_class: Any
) -> dict[str, Any]:
    """Set or clear a paired peer's device class through the audited seam.

    The control-surface half the design anticipated (the
    engine docstring's "the control surface"; phone -> desktop stays
    human-only, and this is that human path). Lookup BEFORE write: an
    unknown peer raises :class:`PeerNotFound` (mapped to a 404) without
    probing the engine. Strict normalisation next: free text raises
    ``ValueError`` (mapped to a 400) before anything reaches the store.
    Then the engine's audited ``set_device_class``, so the flip lands in
    the hash-chain audit log like every trust-state change; a ``False``
    from the setter (the row vanished under the write) also raises
    :class:`PeerNotFound`. Returns the FRESH peer dict re-read from the
    store -- the confirmed posture: the panel only ever renders what the
    registry now holds. Local-disk, permitted in any mode (a human trust
    decision, like relabel).
    """
    rec = store.get_peer(peer_id)
    if rec is None:
        raise PeerNotFound(peer_id)
    value = _normalise_device_class(device_class)
    ok = engine.set_device_class(peer_id, value)
    if not ok:
        raise PeerNotFound(peer_id)
    fresh = store.get_peer(peer_id)
    return _peer_to_dict(fresh if fresh is not None else rec)


# The deferred-ledger surface (SYN-05; web-free, thin handlers wrap these)
#
# Pending content approvals: sensitive records the round quarantined instead of
# applying. The payload carries PROVENANCE only (kind, id, origin device, the
# serving peer, clock, timestamps) -- never the record body: an unapproved
# record's content is untrusted and stays out of the panel; the human decides
# on provenance, and the body only enters the local set through the engine's
# verify -> gate -> apply seam on approval. Local-disk decisions, permitted in
# any mode (only the wire round that fills the ledger is Daily-gated).


def deferred_entry_to_dict(entry: Any) -> dict[str, Any]:
    """Serialise a deferred entry's provenance for the wire (no record body)."""
    return {
        "kind": str(getattr(entry, "kind", "")),
        "record_id": str(getattr(entry, "record_id", "")),
        "origin_device": str(getattr(entry, "origin_device", "")),
        "peer_id": str(getattr(entry, "peer_id", "")),
        "clock": int(getattr(entry, "clock", 0)),
        "deferred_at": str(getattr(entry, "deferred_at", "")),
        "last_offered_at": str(getattr(entry, "last_offered_at", "")),
    }


def deferred_list_payload(engine: Any) -> dict[str, Any]:
    """The pending-approval list, oldest deferred first, with its count."""
    entries = engine.list_deferred()
    return {
        "deferred": [deferred_entry_to_dict(e) for e in entries],
        "count": len(entries),
    }


def approve_deferred_payload(engine: Any, kind: str, record_id: str) -> dict[str, Any]:
    """Approve one pending record through the engine's full seam.

    Propagates the engine's DeferredNotFound for the 404 mapping. The summary
    is the engine's: ``approved`` true when the record entered the apply;
    ``refused`` true when re-verification against the CURRENT trust state said
    no (a changed key, a demoted origin, a closed grace) -- the entry is gone
    either way, and nothing applied on a refusal.
    """
    return engine.approve_deferred(str(kind), str(record_id))


def refuse_deferred_payload(engine: Any, kind: str, record_id: str) -> dict[str, Any]:
    """Refuse one pending record: remove its entry; nothing applies.

    Propagates the engine's DeferredNotFound for the 404 mapping.
    """
    entry = engine.refuse_deferred(str(kind), str(record_id))
    payload = deferred_entry_to_dict(entry)
    payload["removed"] = True
    return payload


def republish_payload(engine: Any) -> dict[str, Any]:
    """Re-sign and re-journal this device's own unsigned records.

    The operator surface for the one-time VL-01 fleet-order step (spec
    section 8): every record this device ORIGINATED that carries no signature
    is re-journalled with one, at the SAME clock, so peers adopt the signed
    bytes idempotently on their next pull. Local-disk signing under the
    documented producers posture -- permitted in any mode, never gated; the
    engine audits the act (``republish_signed``). Foreign-origin records stay
    untouched (their signatures are their originators' to mint). Propagates
    the engine's SigningUnavailable for honest HTTP mapping: the caller asked
    for a signed set and must know, never a silent unsigned republish.
    """
    return {"republished": int(engine.republish_signed())}


# This device's own routing key for its pairing payload (injectable; testable)
#
# Production resolution falls through to the live transport, which returns None --
# a transport-unavailable status -- when the framework is absent or no attached
# node can supply a route. A test injects a resolver via set_self_routing_resolver
# so the pairing-self surface is driven with a fixed key in isolation.

_SELF_ROUTING_RESOLVER: Any = None


def set_self_routing_resolver(resolver: Any) -> None:
    """Install a self-routing-key resolver (``engine -> str | None``) for tests."""
    global _SELF_ROUTING_RESOLVER
    _SELF_ROUTING_RESOLVER = resolver


def reset_self_routing_resolver() -> None:
    """Restore the production self-routing-key resolver."""
    global _SELF_ROUTING_RESOLVER
    _SELF_ROUTING_RESOLVER = None


def resolve_self_routing_key_for_route(engine: Any) -> str | None:
    """This device's routing key, or ``None`` when the transport cannot supply one.

    Uses an injected resolver when set, else the live transport. Defensive: a
    resolver failure degrades to ``None`` (a clean transport-unavailable status),
    which the route maps to a 503, rather than a 500.
    """
    if _SELF_ROUTING_RESOLVER is not None:
        try:
            return _SELF_ROUTING_RESOLVER(engine)
        except Exception:  # pragma: no cover - resolution is defensive
            return None
    if not _SYNC_OK or _transport is None:
        return None
    try:
        return _transport.resolve_self_routing_key()
    except Exception:  # pragma: no cover - resolution is defensive
        logger.exception("self routing key resolution failed")
        return None


# cas 7 Lot 2: the remote-channel control surface helpers. Web-free, so
# the grant/revoke/telemetry contract is exercised without the web stack. The
# durable grant lives on the peer store; the live-session kill is the streaming
# module's buffer drop -- no new revocation primitive. A missing peer raises
# PeerNotFound (mapped to 404 by the thin wrapper).


def _remote_chat_state(store: Any, peer_id: str) -> dict[str, Any]:
    rec = store.get_peer(peer_id) if store is not None else None
    if rec is None:
        raise PeerNotFound(peer_id)
    return {
        "peer_id": peer_id,
        "remote_chat_enabled": bool(getattr(rec, "remote_chat_enabled", True)),
        "rag_subgrant": bool(getattr(rec, "rag_subgrant", False)),
        "pending": bool(getattr(rec, "pending", False)),
    }


def remote_chat_grant_payload(store: Any, peer_id: str) -> dict[str, Any]:
    """A device's remote-inference grant state (enabled, RAG sub-grant, pending)."""
    return _remote_chat_state(store, peer_id)


def set_remote_chat_grant_payload(
    store: Any,
    peer_id: str,
    *,
    enabled: bool | None = None,
    rag: bool | None = None,
) -> dict[str, Any]:
    """Enable/disable remote chat and/or set the RAG sub-grant for a device.

    A missing peer raises PeerNotFound. Disabling remote chat also drops the
    device's in-flight streaming sessions (a disable is a revoke of live work).
    Returns the resulting grant state.
    """
    if store is None or store.get_peer(peer_id) is None:
        raise PeerNotFound(peer_id)
    if enabled is not None:
        store.set_remote_chat_grant(peer_id, bool(enabled))
        if not enabled and _remote_streaming is not None:
            _remote_streaming.kill_sessions_for_device(peer_id)
    if rag is not None:
        store.set_rag_subgrant(peer_id, bool(rag))
    return _remote_chat_state(store, peer_id)


def revoke_remote_chat_payload(store: Any, peer_id: str) -> dict[str, Any]:
    """Revoke a device's remote-chat grant: disable it AND kill live sessions.

    The durable half (the grant column flip) plus the live half (dropping the
    device's in-flight streaming buffers), the instantly-revocable gesture --
    no new revocation primitive. A missing peer raises PeerNotFound.
    """
    if store is None or store.get_peer(peer_id) is None:
        raise PeerNotFound(peer_id)
    store.set_remote_chat_grant(peer_id, False)
    killed = 0
    if _remote_streaming is not None:
        killed = _remote_streaming.kill_sessions_for_device(peer_id)
    return {
        "peer_id": peer_id,
        "revoked": True,
        "killed_sessions": int(killed),
        "remote_chat_enabled": False,
    }


def remote_chat_telemetry_payload() -> dict[str, Any]:
    """The channel rate/telemetry state for the panel (per-device, live sessions)."""
    if _remote_streaming is None:
        return {"devices": {}, "active_sessions": 0}
    return _remote_streaming.telemetry()


# FastAPI surface (guarded; thin wrappers over the helpers)

try:
    from fastapi import APIRouter, Body, Depends, HTTPException

    # Parity (SYN-06): require authentication on every endpoint, the same
    # defense-in-depth per-router dependency routes_plugins carries. The global
    # deny-by-default AuthMiddleware already covers /api/sync (it is not on the
    # public allowlist), so this is parity, not a gap closure.
    try:
        from .routes_auth import _get_current_user

        _auth_dep = [Depends(_get_current_user)]
    except ImportError:
        _auth_dep = []

    router = APIRouter(prefix="/api/sync", tags=["sync"], dependencies=_auth_dep)

    def _resolve_store() -> Any:
        if not _SYNC_OK or get_peer_store is None:
            raise HTTPException(status_code=503, detail="Sync store not available")
        try:
            return get_peer_store()
        except Exception:  # pragma: no cover - store resolution is defensive
            logger.exception("peer store resolution failed")
            raise HTTPException(status_code=503, detail="Sync store not available")

    def _resolve_engine() -> Any:
        if not _SYNC_OK or get_sync_engine is None:
            raise HTTPException(status_code=503, detail="Sync engine not available")
        try:
            return get_sync_engine()
        except Exception:  # pragma: no cover - engine resolution is defensive
            logger.exception("sync engine resolution failed")
            raise HTTPException(status_code=503, detail="Sync engine not available")

    def _resolve_status_store() -> Any:
        """The status store, or None (status is best-effort, never a hard failure)."""
        if not _SYNC_OK or get_sync_status_store is None:
            return None
        try:
            return get_sync_status_store()
        except Exception:  # pragma: no cover - status resolution is defensive
            logger.exception("status store resolution failed")
            return None

    def _resolve_node() -> Any:
        """The node singleton for the status snapshot, or None when unavailable."""
        if not _SYNC_OK or get_node is None:
            return None
        try:
            return get_node()
        except Exception:  # pragma: no cover - node resolution is defensive
            logger.exception("node resolution failed")
            return None

    @router.get("/status")
    def sync_status() -> dict[str, Any]:
        """The sync-status surface: running, the last round, per-peer last-sync."""
        store = _resolve_store()
        status_store = _resolve_status_store()
        node = _resolve_node()
        try:
            return status_payload(node=node, store=store, status_store=status_store)
        except Exception:  # pragma: no cover - status read is defensive
            logger.exception("sync status failed")
            raise HTTPException(status_code=500, detail="Failed to read sync status")

    @router.get("/peers")
    def sync_list_peers() -> dict[str, Any]:
        """List the paired peers."""
        store = _resolve_store()
        try:
            return list_peers_payload(store)
        except Exception:  # pragma: no cover - store read is defensive
            logger.exception("peer list failed")
            raise HTTPException(status_code=500, detail="Failed to list peers")

    @router.get("/peers/{peer_id}")
    def sync_peer_status(peer_id: str) -> dict[str, Any]:
        """One paired peer's status, enriched with its last-sync and last round."""
        store = _resolve_store()
        status_store = _resolve_status_store()
        try:
            return peer_status_payload(store, peer_id, status_store)
        except PeerNotFound:
            raise HTTPException(status_code=404, detail="Peer not paired")
        except Exception:  # pragma: no cover - store read is defensive
            logger.exception("peer status failed")
            raise HTTPException(status_code=500, detail="Failed to read peer")

    @router.get("/peers/{peer_id}/watermark")
    def sync_peer_watermark(peer_id: str) -> dict[str, Any]:
        """One paired peer's current watermark."""
        store = _resolve_store()
        try:
            return peer_watermark_payload(store, peer_id)
        except PeerNotFound:
            raise HTTPException(status_code=404, detail="Peer not paired")
        except Exception:  # pragma: no cover - store read is defensive
            logger.exception("peer watermark failed")
            raise HTTPException(status_code=500, detail="Failed to read watermark")

    @router.post("/peers/{peer_id}/run")
    def sync_run(peer_id: str, conversation_id: str = "") -> dict[str, Any]:
        """Run a pull round against a paired peer over the live transport.

        Refuses under Bulbe (403, via the binding-layer gate), 404 for an unpaired
        peer, 503 when the live transport is unavailable (the framework is absent or
        no attached node and client can supply a route), and 504 when the peer
        stalls past the client's timeout. A sensitive apply reuses the existing
        approval surface: no override is passed, so the engine consults the
        manager-backed approval gate.
        """
        if _emergency_stop is not None:
            _emergency_stop.guard_http()  # Refused, not hung
        store = _resolve_store()
        if not store.has_peer(peer_id):
            raise HTTPException(status_code=404, detail="Peer not paired")
        if _guard is not None and _guard.bulbe_disabled():
            raise HTTPException(
                status_code=403, detail="Sync is disabled in Bulbe mode"
            )
        status_store = _resolve_status_store()
        peer = resolve_peer_for_route(peer_id, store)
        if peer is None:
            if status_store is not None:
                status_store.record_failure(peer_id, "transport unavailable")
            raise HTTPException(
                status_code=503, detail="Live sync transport not available"
            )
        engine = _resolve_engine()
        try:
            result = run_sync_payload(
                engine, peer_id, peer, conversation_id=conversation_id
            )
            if status_store is not None:
                if result.get("parsed", True):
                    status_store.record_round(result)
                else:
                    # SYN-03: a malformed peer answer is a failed attempt, not a
                    # clean empty round; the payload still returns (parsed: false).
                    status_store.record_failure(peer_id, "malformed answer")
            return result
        except PeerNotFound:
            raise HTTPException(status_code=404, detail="Peer not paired")
        except PeerNotConfirmed:
            # PAIR-02: the pairing exists but awaits the mutual
            # confirmation; 409 with an explicit detail so the panel can say
            # what to do rather than show an opaque failure.
            raise HTTPException(
                status_code=409,
                detail="Peer pairing not confirmed; compare and confirm "
                "the pairing code on both devices first",
            )
        except Exception as exc:  # noqa: BLE001 - mapped below
            if _guard is not None and isinstance(exc, _guard.VeilidDisabledInBulbe):
                raise HTTPException(
                    status_code=403, detail="Sync is disabled in Bulbe mode"
                )
            if _guard is not None and isinstance(exc, _guard.VeilidTimeout):
                if status_store is not None:
                    status_store.record_failure(peer_id, "timeout")
                raise HTTPException(
                    status_code=504, detail="Sync peer timed out"
                )
            if _guard is not None and isinstance(exc, _guard.VeilidUnavailable):
                if status_store is not None:
                    status_store.record_failure(peer_id, "transport unavailable")
                raise HTTPException(
                    status_code=503, detail="Live sync transport not available"
                )
            if status_store is not None:
                status_store.record_failure(peer_id, "round failed")
            logger.exception("sync round failed")
            raise HTTPException(status_code=500, detail="Sync round failed")

    @router.get("/pairing/self")
    def sync_pairing_self() -> dict[str, Any]:
        """This device's pairing payload (its identity, routing key, integrity check).

        Local-disk and permitted in any mode, but the routing key is read from the
        live transport: 503 when no attached node can supply one (the framework is
        absent, the node is not attached, or the mode is Bulbe, under which the node
        never attaches). The payload itself carries only public material.
        The device's signing PUBLIC key joins the payload when custody can
        supply one (a device that cannot sign pairs as an honest
        pre-VL-01 peer), and the generated canonical material is PINNED in the
        peer store's meta row so the PAIR-02 confirmation code recomputes from
        local disk in any mode -- a pin failure degrades to a payload without
        a later code, logged, never a 500.
        """
        engine = _resolve_engine()
        peer_id = str(getattr(engine, "device", "") or "local")
        routing_key = resolve_self_routing_key_for_route(engine)
        if not routing_key:
            raise HTTPException(
                status_code=503,
                detail="Routing key not available (node not attached)",
            )
        signing_pub: str | None = None
        pub_getter = getattr(engine, "self_signing_pub", None)
        if callable(pub_getter):
            try:
                signing_pub = pub_getter()
            except Exception:  # pragma: no cover - accessor is defensive
                signing_pub = None
        try:
            # PAIR-03: this codebase IS the desktop node, so its
            # payload declares the desktop class (the Android client declares
            # phone). Guarded: a degraded import leaves the constant None and
            # the payload honestly class-less (the legacy digest).
            payload = self_pairing_payload(
                peer_id,
                routing_key,
                signing_pub,
                device_class=DEVICE_CLASS_DESKTOP
                if isinstance(DEVICE_CLASS_DESKTOP, str)
                else None,
            )
        except Exception:  # pragma: no cover - builder is defensive
            logger.exception("pairing payload build failed")
            raise HTTPException(status_code=500, detail="Failed to build pairing payload")
        try:
            store = _resolve_store()
            pinner = getattr(store, "pin_self_pairing_material", None)
            if callable(pinner) and _pairing is not None:
                pinner(
                    _pairing.pairing_canonical_material(
                        peer_id, routing_key, signing_pub
                    )
                )
        except Exception:  # pragma: no cover - pinning is best-effort
            logger.warning(
                "failed to pin self pairing material; the confirmation code "
                "will be unavailable until the payload is regenerated",
                exc_info=True,
            )
        _audit_pairing("pairing_self", peer_id=peer_id)
        return payload

    @router.post("/pairing/accept")
    def sync_pairing_accept(body: dict = Body(default={})) -> dict[str, Any]:
        """Accept a peer's pairing payload and register it; 400 on a bad payload.

        The body is either the pairing payload itself, or an envelope
        ``{"payload": {...}, "label": "..."}``. Local-disk and permitted in any
        mode; the integrity check rejects a tampered payload before it is stored.
        The peer registers PENDING; the response carries the
        confirmation code to compare on both devices (``null`` until this
        device generates its own pairing code).
        """
        engine = _resolve_engine()
        store = _resolve_store()
        data = body if isinstance(body, dict) else {}
        label = str(data.get("label", "") or "")
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else data
        try:
            return accept_pairing(engine, payload, label=label, store=store)
        except InvalidPairing:
            raise HTTPException(status_code=400, detail="Invalid pairing payload")
        except Exception:  # pragma: no cover - acceptance is defensive
            logger.exception("pairing accept failed")
            raise HTTPException(status_code=500, detail="Failed to accept pairing")

    @router.get("/pairing/pending")
    def sync_pairing_pending() -> dict[str, Any]:
        """The pairings awaiting mutual confirmation (PAIR-02), with their codes.

        Local-disk and permitted in any mode. ``self_ready`` is false until
        this device has generated its own pairing code, the missing half of
        every confirmation code.
        """
        store = _resolve_store()
        try:
            return pending_pairings_payload(store)
        except Exception:  # pragma: no cover - store read is defensive
            logger.exception("pending pairings read failed")
            raise HTTPException(
                status_code=500, detail="Failed to read pending pairings"
            )

    @router.post("/pairing/pending/{peer_id}/confirm")
    def sync_pairing_confirm(peer_id: str) -> dict[str, Any]:
        """Confirm a pending pairing (PAIR-02): the human compared the codes.

        404 when the peer is unknown. Idempotent on an already-confirmed peer.
        Local-disk and permitted in any mode; the activation is audited.
        """
        store = _resolve_store()
        engine = _resolve_engine()
        if store.get_peer(peer_id) is None:
            raise HTTPException(status_code=404, detail="Peer not paired")
        try:
            engine.confirm_peer(peer_id)
            refreshed = store.get_peer(peer_id)
        except Exception:  # pragma: no cover - store write is defensive
            logger.exception("pairing confirm failed")
            raise HTTPException(status_code=500, detail="Failed to confirm pairing")
        if refreshed is None:  # pragma: no cover - removed concurrently
            raise HTTPException(status_code=404, detail="Peer not paired")
        return _peer_to_dict(refreshed)

    @router.post("/pairing/pending/{peer_id}/reject")
    def sync_pairing_reject(peer_id: str) -> dict[str, Any]:
        """Reject a pending pairing (PAIR-02): remove the entry, trust nothing.

        404 when the peer is unknown; 409 when it is already confirmed (an
        active peer is removed through the explicit unpair surface, not the
        confirmation UI -- no fat-finger removal of a trusted device). The
        removal is the existing audited unregister; the rejection itself is
        additionally recorded, and nothing is ever audited as paired.
        """
        store = _resolve_store()
        engine = _resolve_engine()
        rec = store.get_peer(peer_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Peer not paired")
        if not bool(getattr(rec, "pending", False)):
            raise HTTPException(
                status_code=409,
                detail="Peer already confirmed; remove it via unpair instead",
            )
        try:
            removed = engine.unregister_peer(peer_id)
        except Exception:  # pragma: no cover - store write is defensive
            logger.exception("pairing reject failed")
            raise HTTPException(status_code=500, detail="Failed to reject pairing")
        if not removed:  # pragma: no cover - removed concurrently
            raise HTTPException(status_code=404, detail="Peer not paired")
        _audit_pairing("pairing_reject", peer_id=peer_id)
        return {"peer_id": peer_id, "rejected": True}

    @router.delete("/peers/{peer_id}")
    def sync_unpair(peer_id: str) -> dict[str, Any]:
        """Unpair a peer; 404 when it is not paired. Permitted in any mode."""
        engine = _resolve_engine()
        try:
            removed = engine.unregister_peer(peer_id)
        except Exception:  # pragma: no cover - store write is defensive
            logger.exception("unpair failed")
            raise HTTPException(status_code=500, detail="Failed to unpair")
        if not removed:
            raise HTTPException(status_code=404, detail="Peer not paired")
        # cas 7 Lot 2: the detach-in-one-gesture also drops the device's
        # in-flight remote-inference streaming sessions -- the emergency-stop
        # detach kills live work, reusing the streaming buffer drop, not a new
        # revocation primitive.
        if _remote_streaming is not None:
            try:
                _remote_streaming.kill_sessions_for_device(peer_id)
            except Exception:  # pragma: no cover - best-effort kill
                logger.debug("remote streaming kill on unpair failed", exc_info=True)
        return {"peer_id": peer_id, "removed": True}

    @router.get("/peers/{peer_id}/remote-chat")
    def sync_remote_chat_grant(peer_id: str) -> dict[str, Any]:
        """A device's remote-inference grant state (cas 7 Lot 2). 404 if unpaired."""
        store = _resolve_store()
        try:
            return remote_chat_grant_payload(store, peer_id)
        except PeerNotFound:
            raise HTTPException(status_code=404, detail="Peer not paired")

    @router.post("/peers/{peer_id}/remote-chat")
    def sync_set_remote_chat_grant(
        peer_id: str, body: dict = Body(default={})
    ) -> dict[str, Any]:
        """Enable/disable remote chat and/or set the RAG sub-grant (cas 7 Lot 2).

        Body: ``{enabled?: bool, rag?: bool}``. Disabling remote chat also drops
        the device's in-flight streaming sessions. 404 when the peer is unpaired.
        """
        store = _resolve_store()
        data = body if isinstance(body, dict) else {}
        raw_enabled = data.get("enabled")
        raw_rag = data.get("rag")
        enabled = bool(raw_enabled) if isinstance(raw_enabled, bool) else None
        rag = bool(raw_rag) if isinstance(raw_rag, bool) else None
        try:
            return set_remote_chat_grant_payload(store, peer_id, enabled=enabled, rag=rag)
        except PeerNotFound:
            raise HTTPException(status_code=404, detail="Peer not paired")

    @router.post("/peers/{peer_id}/remote-chat/revoke")
    def sync_revoke_remote_chat(peer_id: str) -> dict[str, Any]:
        """Revoke a device's remote-chat grant and kill its live sessions (cas 7).

        Instantly revocable: the durable grant flip plus the live-session kill,
        in one gesture. 404 when the peer is unpaired.
        """
        store = _resolve_store()
        try:
            return revoke_remote_chat_payload(store, peer_id)
        except PeerNotFound:
            raise HTTPException(status_code=404, detail="Peer not paired")

    @router.get("/remote-chat/telemetry")
    def sync_remote_chat_telemetry() -> dict[str, Any]:
        """The remote channel rate/telemetry state for the panel (cas 7 Lot 2)."""
        return remote_chat_telemetry_payload()

    @router.post("/peers/{peer_id}/label")
    def sync_relabel(peer_id: str, body: dict = Body(default={})) -> dict[str, Any]:
        """Relabel a paired peer; 404 when it is not paired. Permitted in any mode."""
        store = _resolve_store()
        engine = _resolve_engine()
        data = body if isinstance(body, dict) else {}
        label = str(data.get("label", "") or "")
        try:
            return relabel_peer_payload(store, engine, peer_id, label)
        except PeerNotFound:
            raise HTTPException(status_code=404, detail="Peer not paired")
        except Exception:  # pragma: no cover - store write is defensive
            logger.exception("relabel failed")
            raise HTTPException(status_code=500, detail="Failed to relabel peer")

    @router.post("/peers/{peer_id}/device-class")
    def sync_set_device_class(
        peer_id: str, body: dict = Body(default={})
    ) -> dict[str, Any]:
        """Set or clear a paired peer's device class; the human-confirmed path.

        The control surface the earlier accept seam deferred to
        (phone -> desktop never happens at the ceremony; it happens HERE,
        by a human). The body carries ``{"device_class": "phone" |
        "desktop" | null}``; free text is refused with a 400 before the
        store is touched, an unpaired peer is a 404, and the write goes
        through the engine's audited setter. Permitted in any mode (local
        trust state, like relabel).
        """
        store = _resolve_store()
        engine = _resolve_engine()
        data = body if isinstance(body, dict) else {}
        value = data.get("device_class")
        try:
            return set_device_class_payload(store, engine, peer_id, value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except PeerNotFound:
            raise HTTPException(status_code=404, detail="Peer not paired")
        except Exception:  # pragma: no cover - store write is defensive
            logger.exception("device-class set failed")
            raise HTTPException(
                status_code=500, detail="Failed to set device class"
            )

    @router.post("/republish")
    def sync_republish() -> dict[str, Any]:
        """Re-sign and re-journal this device's own records (VL-01 fleet order).

        The one-time fleet ceremony step the grace flip's upgrade order
        requires (spec section 8). Local-disk and mode-free under the
        documented producers posture: signing a local set is a local edit, so
        this works under Bulbe too -- only the wire is Daily-gated. Returns
        the republished count (0 is honest idempotence: nothing unsigned
        remained); audited by the engine. 503 when this device cannot sign
        (no PQC backend or no master key) -- stated honestly, never a silent
        unsigned republish.
        """
        engine = _resolve_engine()
        try:
            return republish_payload(engine)
        except Exception as exc:
            logger.warning("republish refused: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Signing unavailable on this device; nothing republished",
            )

    @router.get("/deferred")
    def sync_deferred_list() -> dict[str, Any]:
        """The pending content approvals (SYN-05): provenance only, no bodies.

        Sensitive records the round quarantined instead of applying, awaiting
        the human's approve/refuse. Local-disk read, permitted in any mode.
        """
        engine = _resolve_engine()
        try:
            return deferred_list_payload(engine)
        except Exception:  # pragma: no cover - ledger read is defensive
            logger.exception("deferred list failed")
            raise HTTPException(
                status_code=500, detail="Failed to list deferred records"
            )

    @router.post("/deferred/approve")
    def sync_deferred_approve(body: dict = Body(default={})) -> dict[str, Any]:
        """Approve a pending record; it re-enters the full apply seam.

        Body: ``{"kind": ..., "record_id": ...}``; 400 when either is missing,
        404 for an unknown key. The approval re-verifies against the CURRENT
        trust state -- ``refused: true`` in the response means a changed key, a
        demoted origin, or a closed grace said no, the entry is removed, and
        nothing applied. Local-disk decision, permitted in any mode.
        """
        engine = _resolve_engine()
        data = body if isinstance(body, dict) else {}
        kind = str(data.get("kind", "") or "")
        record_id = str(data.get("record_id", "") or "")
        if not kind or not record_id:
            raise HTTPException(
                status_code=400, detail="kind and record_id are required"
            )
        try:
            return approve_deferred_payload(engine, kind, record_id)
        except DeferredNotFound:
            raise HTTPException(
                status_code=404, detail="No pending record for that key"
            )
        except Exception:  # pragma: no cover - the seam is defensive
            logger.exception("deferred approve failed")
            raise HTTPException(
                status_code=500, detail="Failed to approve deferred record"
            )

    @router.post("/deferred/refuse")
    def sync_deferred_refuse(body: dict = Body(default={})) -> dict[str, Any]:
        """Refuse a pending record: remove its entry; nothing applies.

        Body: ``{"kind": ..., "record_id": ...}``; 400 when either is missing,
        404 for an unknown key. Local-disk decision, permitted in any mode.
        """
        engine = _resolve_engine()
        data = body if isinstance(body, dict) else {}
        kind = str(data.get("kind", "") or "")
        record_id = str(data.get("record_id", "") or "")
        if not kind or not record_id:
            raise HTTPException(
                status_code=400, detail="kind and record_id are required"
            )
        try:
            return refuse_deferred_payload(engine, kind, record_id)
        except DeferredNotFound:
            raise HTTPException(
                status_code=404, detail="No pending record for that key"
            )
        except Exception:  # pragma: no cover - the removal is defensive
            logger.exception("deferred refuse failed")
            raise HTTPException(
                status_code=500, detail="Failed to refuse deferred record"
            )

except Exception:  # pragma: no cover - FastAPI absent (e.g. isolated tests)
    router = None  # type: ignore[assignment]


def register(app: Any) -> bool:
    """Register the sync router on a FastAPI app. Returns False when unavailable."""
    if router is None:
        return False
    try:
        app.include_router(router)
        return True
    except Exception:  # pragma: no cover - defensive
        logger.exception("failed to register sync router")
        return False
