#!/usr/bin/env python3
"""In-memory last-sync / status store for Veilid sync (S181 Goal 3, Theme 4).

Sync status is runtime telemetry, not durable state: how the last round went, when
a peer last synced, whether a round succeeded or failed. The durable fact -- how far
a device has synced a peer -- is the watermark, which lives in the peer store
(``peers.py``). This store holds the per-run outcome a status surface shows, so it
is in-memory and resets on restart, which is correct: a fresh process has not run a
round this session.

It keeps, per peer, the outcome of the most recent round (applied, deferred,
conflicts retained, rejected, the watermark before and after, whether it advanced,
a timestamp, an ``ok`` flag, and an error string), and the single most recent round
across all peers. A round that succeeds and a round that fails -- a peer timeout, an
unavailable transport -- are both recorded, so the surface reflects the last
attempt, not only the last success.

A process singleton with a reset hook, thread-safe under a lock. It is web-free and
domain-free: ``record_round`` reads a round summary by attribute or key (so it does
not import the engine), and the route's status payload helpers consume plain
outcomes, so they are testable in isolation without the web stack or the framework.
There is no SQL here, so the SQL hygiene rules do not apply; only the durable
watermark is persisted, and that is the peer store's job.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(summary: Any, name: str, default: Any = 0) -> Any:
    """Read a field from a round summary that may be an object or a dict."""
    if isinstance(summary, dict):
        return summary.get(name, default)
    return getattr(summary, name, default)


@dataclass(frozen=True)
class RoundOutcome:
    """The recorded outcome of one sync round (or one failed attempt).

    Attributes:
        peer_id: The peer the round ran against.
        applied: How many records were newly adopted.
        deferred: How many sensitive records were held for approval.
        conflicts: How many concurrent divergences were retained.
        rejected: How many wire records were dropped on decode.
        refused: How many records were refused at the signature seam (S205,
            VL-01): unsigned or invalid against the origin device's
            registered key. Counted so SyncPanel can show them.
        unverified: How many records were accepted without verification:
            no verification backend on this device (pre-VL-01 posture), or
            the historical S205..S207 unkeyed-origin grace (closed at S208;
            tests may re-open it by monkeypatch).
        previous_watermark: The peer's watermark before the round.
        new_watermark: The peer's watermark after the round.
        advanced: Whether the watermark moved this round.
        at: ISO-8601 timestamp of when the outcome was recorded.
        ok: True for a completed round, False for a failed attempt.
        error: A short reason when ``ok`` is False; empty otherwise.
    """

    peer_id: str
    applied: int = 0
    deferred: int = 0
    conflicts: int = 0
    rejected: int = 0
    refused: int = 0
    unverified: int = 0
    previous_watermark: int = 0
    new_watermark: int = 0
    advanced: bool = False
    at: str = ""
    ok: bool = True
    error: str = ""


class SyncStatusStore:
    """The per-peer last-round outcomes and the most recent round, in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_peer: dict[str, RoundOutcome] = {}
        self._last: Optional[RoundOutcome] = None

    def record_round(self, summary: Any, *, at: Optional[str] = None) -> RoundOutcome:
        """Record a completed round from its summary (a RoundResult or a dict).

        Returns the stored outcome. The summary is read by attribute or key, so the
        store does not depend on the engine's concrete type.
        """
        peer_id = _get(summary, "peer_id", "")
        outcome = RoundOutcome(
            peer_id=str(peer_id),
            applied=int(_get(summary, "applied", 0)),
            deferred=int(_get(summary, "deferred", 0)),
            conflicts=int(_get(summary, "conflicts", 0)),
            rejected=int(_get(summary, "rejected", 0)),
            refused=int(_get(summary, "refused", 0)),
            unverified=int(_get(summary, "unverified", 0)),
            previous_watermark=int(_get(summary, "previous_watermark", 0)),
            new_watermark=int(_get(summary, "new_watermark", 0)),
            advanced=bool(_get(summary, "advanced", False)),
            at=at or _now_iso(),
            ok=True,
            error="",
        )
        with self._lock:
            if outcome.peer_id:
                self._by_peer[outcome.peer_id] = outcome
            self._last = outcome
        return outcome

    def record_failure(
        self, peer_id: str, error: str, *, at: Optional[str] = None
    ) -> RoundOutcome:
        """Record a failed attempt (a timeout, an unavailable transport, a refusal).

        The watermark is unchanged on a failure, so the outcome carries no movement;
        ``ok`` is False and ``error`` carries the reason.
        """
        outcome = RoundOutcome(
            peer_id=str(peer_id),
            at=at or _now_iso(),
            ok=False,
            error=str(error),
        )
        with self._lock:
            if outcome.peer_id:
                self._by_peer[outcome.peer_id] = outcome
            self._last = outcome
        return outcome

    def last_for(self, peer_id: str) -> Optional[RoundOutcome]:
        """The most recent outcome for a peer, or None when it has none yet."""
        if not isinstance(peer_id, str) or not peer_id:
            return None
        with self._lock:
            return self._by_peer.get(peer_id)

    def last_round(self) -> Optional[RoundOutcome]:
        """The single most recent outcome across all peers, or None."""
        with self._lock:
            return self._last

    def peer_count(self) -> int:
        """How many peers have a recorded outcome."""
        with self._lock:
            return len(self._by_peer)

    def clear(self) -> None:
        """Forget all recorded outcomes."""
        with self._lock:
            self._by_peer.clear()
            self._last = None


# Module-level singleton with a reset hook (one status store per process, testable).

_status: Optional[SyncStatusStore] = None
_status_lock = threading.Lock()


def get_sync_status_store() -> SyncStatusStore:
    """Return the process status store, creating it once."""
    global _status
    with _status_lock:
        if _status is None:
            _status = SyncStatusStore()
        return _status


def set_sync_status_store(store: Optional[SyncStatusStore]) -> None:
    """Install a specific status store as the process singleton (used by tests)."""
    global _status
    with _status_lock:
        _status = store


def reset_sync_status_store() -> None:
    """Clear the process singleton so the next get creates a fresh one."""
    global _status
    with _status_lock:
        _status = None
