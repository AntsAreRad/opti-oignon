#!/usr/bin/env python3
"""The user's review of pending writes: list them, accept or decline them in a batch.

A memory or notes write of the agent that the user's typed words did not
endorse waits in the review queue (``opti_oignon.pending_writes``) until the
user decides; so does a fact the manual extraction drew from anything but
the user's typed words. This router is where the user decides:

- ``GET /api/pending-writes`` lists the proposals still waiting, oldest
  first, each with its store, its action, its arguments exactly as they
  would be written, the provenance of each argument and, for an update or a
  delete, what it would change as the store holds it now;
- ``POST /api/pending-writes/accept`` applies the given proposals in the
  order given, each once and exactly as proposed, through the store's own
  write; an unknown or already decided id is reported, never fatal;
- ``POST /api/pending-writes/decline`` declines the given proposals;
  nothing is written.

Design notes:

- This is the user's own surface, never a model-reachable tool, so it
  carries no route-level mode gate; the security-mode middleware still
  applies its global posture to every path.
- It rides a router of its own (``pending_writes_router``), authed exactly
  like the notes router: the current-user dependency on the router.
- The route issues no SQL and applies nothing itself: the queue and its
  ``accept`` / ``decline`` do.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Hardcoded checkpoint discipline for every new module; never overridable.
checkpoint_before_apply = True

# The most ids one decision may carry.
_MAX_IDS = 500

try:
    from opti_oignon import pending_writes as _pending_writes

    _QUEUE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    _pending_writes = None  # type: ignore[assignment]
    _QUEUE_AVAILABLE = False

try:
    from .routes_auth import _get_current_user

    _auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional

    _auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


pending_writes_router = APIRouter(
    prefix="/api/pending-writes", tags=["pending-writes"], dependencies=_auth_dep
)


class PendingWriteDecision(BaseModel):
    """The proposals one decision applies to, by id, in the order to apply them."""

    ids: list[str] = Field(default_factory=list, max_length=_MAX_IDS)


def _pending_dep() -> Any:
    """The review queue; a 503 when it cannot be opened."""
    if not _QUEUE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Review queue not available")
    try:
        return _pending_writes.get_pending_store()
    except Exception as exc:
        logger.warning("review queue unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Review queue not available")


def _memory_dep() -> Any:
    """The coordinated memory store, or None when it cannot be reached."""
    try:
        from opti_oignon.memory import get_memory_store

        return get_memory_store()
    except Exception:  # pragma: no cover - backend optional
        return None


def _notes_dep() -> Any:
    """The coordinated notes store, or None when it cannot be reached."""
    try:
        from opti_oignon.notes import get_notes_store

        return get_notes_store()
    except Exception:  # pragma: no cover - backend optional
        return None


def _target(record: Any, memory: Any, notes: Any, user_id: str | None) -> dict[str, Any] | None:
    """What an update or a delete would change, as the store holds it now; None for any other write."""
    if record.action not in ("update", "delete"):
        return None
    try:
        if record.store == "memory" and memory is not None:
            fact = memory.get(str(record.arguments.get("fact_id", "")), user_id=user_id)
            return {"text": getattr(fact, "text", "")} if fact is not None else None
        if record.store == "notes" and notes is not None:
            note = notes.get_note(str(record.arguments.get("note_id", "")), user_id=user_id)
            return {"title": getattr(note, "title", "")} if note is not None else None
    except Exception:  # a target that cannot be read is shown as unknown
        logger.debug("pending write target unreadable", exc_info=True)
    return None


@pending_writes_router.get("")
def list_pending_writes(
    store: str = "",
    pending: Any = Depends(_pending_dep),
    memory: Any = Depends(_memory_dep),
    notes: Any = Depends(_notes_dep),
    current_user: dict = Depends(_get_current_user),
) -> list[dict[str, Any]]:
    """List the writes waiting for the user's review, oldest first; ``store`` keeps one store's.

    An acceptance an earlier process left unfinished is completed first.
    """
    if store not in ("", "memory", "notes"):
        raise HTTPException(status_code=422, detail="store must be memory or notes")
    user_id = current_user.get("sub")
    try:
        _pending_writes.recover(pending=pending, memory_store=memory, notes_store=notes, user_id=user_id)
    except Exception as exc:  # the listing never fails on it
        logger.warning("unfinished acceptances not completed: %s", exc)
    records = pending.list(status="pending", store=store or None, user_id=user_id)
    return [dict(record.to_dict(), target=_target(record, memory, notes, user_id)) for record in records]


@pending_writes_router.post("/accept")
def accept_pending_writes(
    request: PendingWriteDecision,
    pending: Any = Depends(_pending_dep),
    memory: Any = Depends(_memory_dep),
    notes: Any = Depends(_notes_dep),
    current_user: dict = Depends(_get_current_user),
) -> dict[str, Any]:
    """Apply the given proposals in order, each once and exactly as proposed; one result per id."""
    results = _pending_writes.accept(request.ids, pending=pending, memory_store=memory, notes_store=notes,
                                     user_id=current_user.get("sub"))
    return {"results": results}


@pending_writes_router.post("/decline")
def decline_pending_writes(
    request: PendingWriteDecision,
    pending: Any = Depends(_pending_dep),
    current_user: dict = Depends(_get_current_user),
) -> dict[str, Any]:
    """Decline the given proposals; nothing is written. One result per id."""
    return {"results": _pending_writes.decline(request.ids, pending=pending, user_id=current_user.get("sub"))}
