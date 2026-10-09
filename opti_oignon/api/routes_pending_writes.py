#!/usr/bin/env python3
"""The user's review of pending writes: list them, accept or decline them in a batch.

A memory or notes write of the agent that the user's typed words did not
endorse waits in the review queue (``opti_oignon.pending_writes``) until the
user decides; so does a fact the manual extraction drew from anything but
the user's typed words, and so does every skill the agent or its teacher
writes. This router is where the user decides:

- ``GET /api/pending-writes`` lists the proposals still waiting, oldest
  first, each with its store, its action, its arguments exactly as they
  would be written, the provenance of each argument and, for an update or a
  delete, what it would change as the store holds it now; beside them, each
  value whole as the approval drawer shows it, every character a screen
  hides written as its escape, the digest an acceptance names, and for a
  skill its high risk;
- ``POST /api/pending-writes/accept`` applies the given proposals in the
  order given, each once and exactly as proposed, through the store's own
  write; a skill only when the request names the digest of the text shown,
  any other proposal only when a digest given for it names it; an unknown or
  already decided id is reported, never fatal;
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
    """The proposals one decision applies to, by id, in the order to apply them, with the digests shown."""

    ids: list[str] = Field(default_factory=list, max_length=_MAX_IDS)
    digests: dict[str, str] = Field(default_factory=dict, max_length=_MAX_IDS)


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


def _skills_dep() -> Any:
    """The skills registry, or None when it cannot be reached."""
    try:
        from opti_oignon.agent.skills import get_skill_registry

        return get_skill_registry()
    except Exception:  # pragma: no cover - backend optional
        return None


def _registry(skills: Any) -> Any:
    """``skills`` when it is a registry; a dependency left unresolved by a direct call is none."""
    return skills if callable(getattr(skills, "get", None)) else None


def _visible(value: Any) -> Any:
    """A value as the approval drawer shows it, whole: text with every hidden character written as its escape."""
    if isinstance(value, (list, tuple)):
        return [_visible(item) for item in value]
    if not isinstance(value, str):
        return value
    try:
        from opti_oignon.tool_call_approval import _visible as drawn
    except Exception:  # no drawer to agree with: escape everything past ASCII, hide nothing
        return ascii(value)[1:-1]
    return drawn(value)


def _skill_target(record: Any, skills: Any) -> dict[str, Any] | None:
    """The skill text a proposal would replace or delete, and its digest, as the registry holds it now.

    For a new skill, the one published under its name since it was proposed, if any: accepting it would be refused.
    """
    registry = _registry(skills)
    if registry is None or record.action not in ("add", "edit", "delete"):
        return None
    arguments = record.arguments
    skill = registry.get(str(arguments.get("name", "")), str(arguments.get("category", "")),
                         draft=bool(arguments.get("draft")))
    return {"text": skill.canonical(), "sha256": skill.digest()} if skill is not None else None


def _target(record: Any, memory: Any, notes: Any, user_id: str | None, skills: Any = None) -> dict[str, Any] | None:
    """What an update or a delete would change, as the store holds it now; None for any other write."""
    try:
        if record.store == "skills":
            return _skill_target(record, skills)
    except Exception:  # a target that cannot be read is shown as unknown
        logger.debug("pending skill target unreadable", exc_info=True)
        return None
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


def _item(record: Any, memory: Any, notes: Any, user_id: str | None, skills: Any) -> dict[str, Any]:
    """One proposal for the review: as stored, with each value and its target as shown, its digest and risk."""
    target = _target(record, memory, notes, user_id, skills)
    shown_target = {key: _visible(value) for key, value in target.items() if key != "sha256"} if target else None
    item = dict(record.to_dict(), target=target,
                shown={"arguments": {key: _visible(value) for key, value in record.arguments.items()},
                       "target": shown_target},
                digest=_pending_writes.shown_digest(record))
    if record.store == "skills":
        # A skill's text reaches a system prompt: its write is high risk, whoever proposed it.
        item["risk"] = "high"
    return item


@pending_writes_router.get("")
def list_pending_writes(
    store: str = "",
    pending: Any = Depends(_pending_dep),
    memory: Any = Depends(_memory_dep),
    notes: Any = Depends(_notes_dep),
    skills: Any = Depends(_skills_dep),
    current_user: dict = Depends(_get_current_user),
) -> list[dict[str, Any]]:
    """List the writes waiting for the user's review, oldest first; ``store`` keeps one store's.

    Each value comes whole as the approval drawer shows it, beside the
    digest an acceptance names. An acceptance an earlier process left
    unfinished is completed first.
    """
    if store not in ("", "memory", "notes", "skills"):
        raise HTTPException(status_code=422, detail="store must be memory, notes or skills")
    user_id = current_user.get("sub")
    try:
        _pending_writes.recover(pending=pending, memory_store=memory, notes_store=notes, user_id=user_id,
                                skills_registry=_registry(skills))
    except Exception as exc:  # the listing never fails on it
        logger.warning("unfinished acceptances not completed: %s", exc)
    records = pending.list(status="pending", store=store or None, user_id=user_id)
    return [_item(record, memory, notes, user_id, skills) for record in records]


@pending_writes_router.post("/accept")
def accept_pending_writes(
    request: PendingWriteDecision,
    pending: Any = Depends(_pending_dep),
    memory: Any = Depends(_memory_dep),
    notes: Any = Depends(_notes_dep),
    skills: Any = Depends(_skills_dep),
    current_user: dict = Depends(_get_current_user),
) -> dict[str, Any]:
    """Apply the given proposals in order, each once and exactly as proposed; one result per id.

    A skill is applied only when the request names the digest of the text
    shown; any other proposal only when a digest given for it names it.
    """
    results = _pending_writes.accept(request.ids, pending=pending, memory_store=memory, notes_store=notes,
                                     user_id=current_user.get("sub"), digests=request.digests,
                                     skills_registry=_registry(skills))
    return {"results": results}


@pending_writes_router.post("/decline")
def decline_pending_writes(
    request: PendingWriteDecision,
    pending: Any = Depends(_pending_dep),
    current_user: dict = Depends(_get_current_user),
) -> dict[str, Any]:
    """Decline the given proposals; nothing is written. One result per id."""
    return {"results": _pending_writes.decline(request.ids, pending=pending, user_id=current_user.get("sub"))}
