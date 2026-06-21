#!/usr/bin/env python3
"""Server-side streaming channel state for cas 7 Lot 2 (S235).

The desktop-side state behind the chunked-app_call streaming shape
(REMOTE_INFERENCE_SPEC section 5, Option A) and the channel abuse control
(section 7). It holds three things, all in process memory, all behind one lock:

  - the streaming session registry: per in-flight remote request, the produced
    response chunks, keyed by the ``(route-authenticated peer_id, request_id)``
    PAIR. The compound key IS the section-14 binding: a continuation lookup is a
    literal ``(peer_id, request_id)`` match, and the ``peer_id`` is the route's,
    never the request payload's, so a malformed or hostile continuation can only
    ever read a buffer the asking device itself owns. A miss is a refusal at the
    handler, never a cross-read.
  - the per-device rate limiter: a fixed window per device. A device over its
    window gets a structured refusal at the handler and an alert is recorded in
    the channel telemetry for the panel to surface. The clock is injectable so
    the gate is deterministic in tests.
  - the channel telemetry: the per-device window counters and alert counts, plus
    the live session count, read by the desktop control surface.

Revocation reuses no new primitive: the durable half of a revoke is the peer
store's grant column; the LIVE half is :func:`kill_sessions_for_device`, which
drops a device's in-flight buffers so its next continuation misses. The
emergency-stop detach (the existing unpair) is wired to the same kill, and the
global emergency stop already stops the node (the channel dies physically) and
sets the admission stop flag.

The registry is bounded: at most :data:`MAX_SESSIONS` concurrent sessions, the
oldest evicted past the cap, so a flood of half-consumed streams can never grow
memory without bound. The chunks are the desktop's own model output (not wire
data), but a single session is still capped defensively by count and bytes.

This module is import-light (stdlib only), so the refusal paths in
``remote_inference`` that touch it pull no ollama chain.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

# S73/S74: every new module hardcodes the checkpoint sentinel; never overridable.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True

# Registry bounds. MAX_SESSIONS caps concurrent in-flight streams (the oldest is
# evicted past the cap). The per-session caps guard memory against a pathological
# generation; the desktop's own output should never approach them.
MAX_SESSIONS = 256
MAX_CHUNKS_PER_SESSION = 8192
MAX_SESSION_BYTES = 8_388_608  # 8 MiB

# Rate-limit defaults (a fixed window per device). The same order of magnitude as
# the HTTPS-path RemoteSessionGuard's per-minute request bound; this is the
# Veilid-channel analogue, keyed by peer id rather than a cert fingerprint, and
# does not share that guard's configuration.
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60.0

# Module state, all guarded by _LOCK (re-entrant so telemetry may read the
# session count without deadlocking).
_LOCK = threading.RLock()
# key: (peer_id, request_id) -> list[str] of response chunks.
_SESSIONS: OrderedDict[tuple, list] = OrderedDict()
# peer_id -> [window_start_epoch, count_in_window]
_RATE: dict = {}
# peer_id -> {"count": int, "last": float}
_ALERTS: dict = {}


def _key(peer_id: Any, request_id: Any) -> tuple | None:
    if not isinstance(peer_id, str) or not peer_id:
        return None
    if not isinstance(request_id, str) or not request_id:
        return None
    return (peer_id, request_id)


def _bounded_chunks(chunks: Any) -> list:
    """Coerce to a bounded list of string chunks (defensive memory guard)."""
    out: list = []
    total = 0
    for chunk in list(chunks)[:MAX_CHUNKS_PER_SESSION]:
        s = chunk if isinstance(chunk, str) else str(chunk)
        total += len(s.encode("utf-8"))
        if total > MAX_SESSION_BYTES:
            logger.debug("remote streaming session truncated at the byte cap")
            break
        out.append(s)
    return out


def open_session(peer_id: str, request_id: str, chunks: Any) -> None:
    """Register an in-flight stream's chunks, keyed by (peer_id, request_id).

    Replaces any existing session under the same key (a re-issued request id
    starts fresh). Past :data:`MAX_SESSIONS` the oldest session is evicted, so
    the registry is bounded. A bad key is a silent no-op (the handler validated
    the identity already).
    """
    key = _key(peer_id, request_id)
    if key is None:
        return
    bounded = _bounded_chunks(chunks)
    with _LOCK:
        if key in _SESSIONS:
            del _SESSIONS[key]
        _SESSIONS[key] = bounded
        while len(_SESSIONS) > MAX_SESSIONS:
            _SESSIONS.popitem(last=False)  # evict the oldest


def pull(peer_id: str, request_id: str, cursor: Any) -> dict | None:
    """Return the chunk at ``cursor`` for a session, or ``None`` on a mismatch.

    The lookup is a literal ``(peer_id, request_id)`` match -- the binding. A
    missing session, a request id owned by a different device, or a cursor out
    of range all return ``None``; the handler maps that to a ``buffer_mismatch``
    refusal (the section-14 risk -- never a cross-read). On success returns
    ``{"request_id", "content", "cursor", "done"}`` with ``cursor`` advanced;
    when the terminal chunk is served the session is dropped (the stream is
    consumed).
    """
    key = _key(peer_id, request_id)
    if key is None:
        return None
    if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
        return None
    with _LOCK:
        chunks = _SESSIONS.get(key)
        if chunks is None:
            return None
        n = len(chunks)
        if cursor >= n:
            # Anomalous on a live session (the terminal pull drops it); treat a
            # past-the-end read as a mismatch and clear the stale session.
            _SESSIONS.pop(key, None)
            return None
        content = chunks[cursor]
        next_cursor = cursor + 1
        done = next_cursor >= n
        if done:
            _SESSIONS.pop(key, None)
    return {
        "request_id": request_id,
        "content": content,
        "cursor": next_cursor,
        "done": done,
    }


def kill_sessions_for_device(peer_id: str) -> int:
    """Drop all in-flight streaming sessions for a device (the live revoke half).

    Returns the number dropped. Used by the grant-revoke control surface and by
    the unpair (emergency-stop detach), so revoking or detaching a device kills
    its live remote sessions instantly. No new revocation primitive: this is a
    buffer drop on the channel's own state.
    """
    if not isinstance(peer_id, str) or not peer_id:
        return 0
    with _LOCK:
        keys = [k for k in _SESSIONS if k[0] == peer_id]
        for k in keys:
            del _SESSIONS[k]
    return len(keys)


def kill_all_sessions() -> int:
    """Drop every in-flight streaming session. Returns the number dropped."""
    with _LOCK:
        count = len(_SESSIONS)
        _SESSIONS.clear()
    return count


def active_session_count() -> int:
    """The number of in-flight streaming sessions."""
    with _LOCK:
        return len(_SESSIONS)


def _record_alert(peer_id: str, now: float) -> None:
    entry = _ALERTS.get(peer_id)
    if entry is None:
        _ALERTS[peer_id] = {"count": 1, "last": now}
    else:
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last"] = now


def check_rate(
    peer_id: str,
    *,
    now: float | None = None,
    limit: int | None = None,
    window: float | None = None,
) -> bool:
    """Per-device fixed-window rate gate. ``True`` allows, ``False`` is a breach.

    Counts requests per device in a fixed window; the window resets once the
    elapsed time reaches ``window``. On a breach the request is NOT counted and
    an alert is recorded in the telemetry (so the desktop surfaces it); the
    handler returns a structured refusal. The clock is injectable for
    deterministic tests; the limit and window default to the module constants.
    A bad peer id allows (the handler authenticated the peer already).
    """
    if not isinstance(peer_id, str) or not peer_id:
        return True
    now = time.time() if now is None else float(now)
    limit = RATE_LIMIT_REQUESTS if limit is None else int(limit)
    window = RATE_LIMIT_WINDOW_SECONDS if window is None else float(window)
    with _LOCK:
        window_start, count = _RATE.get(peer_id, (now, 0))
        if now - window_start >= window:
            window_start, count = now, 0
        if count >= limit:
            _RATE[peer_id] = (window_start, count)
            _record_alert(peer_id, now)
            return False
        _RATE[peer_id] = (window_start, count + 1)
        return True


def telemetry(peer_id: str | None = None) -> dict:
    """The channel rate/telemetry state for the desktop control surface.

    For a single device, its window counter, the window start, and its alert
    count. For the whole channel (``peer_id`` omitted), the per-device map plus
    the live session count. Read-only; never on the remote surface.
    """
    with _LOCK:
        if isinstance(peer_id, str) and peer_id:
            window_start, count = _RATE.get(peer_id, (0.0, 0))
            alerts = int(_ALERTS.get(peer_id, {}).get("count", 0))
            return {
                "peer_id": peer_id,
                "requests_in_window": count,
                "window_started_at": window_start,
                "alerts": alerts,
            }
        devices = {}
        for pid in set(_RATE) | set(_ALERTS):
            window_start, count = _RATE.get(pid, (0.0, 0))
            alerts = int(_ALERTS.get(pid, {}).get("count", 0))
            devices[pid] = {
                "requests_in_window": count,
                "window_started_at": window_start,
                "alerts": alerts,
            }
        return {"devices": devices, "active_sessions": len(_SESSIONS)}


def reset_for_tests() -> None:
    """Clear the session registry, the rate windows, and the alert counts."""
    with _LOCK:
        _SESSIONS.clear()
        _RATE.clear()
        _ALERTS.clear()
