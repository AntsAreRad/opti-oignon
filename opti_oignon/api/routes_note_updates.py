#!/usr/bin/env python3
"""FastAPI notes-update legs (N.8 editor seam): bind the update store.

The Notes body collaboration model (NOTES_CRDT_SPEC.md) keeps
the CRDT in the client and moves opaque Yjs update blobs through the platform.
The at-rest append-only ``note_update`` store landed first,
and the transport (the ``note_update`` record kind on the seam, the
serve floor) followed. This module is the HTTP surface the SvelteKit editor
rides for the update log, alongside the five whole-note routes in
``routes_notes``:

- append: ``POST /api/notes/{note_id}/updates`` over ``append_update`` -- the
  local editor's incremental update. A store refusal (a dead or unknown
  parent, a duplicate seq, a missing blob -- NOTES_CRDT_SPEC.md section 5)
  maps to a 409, never a silent success; an undecodable blob is a 422.
- tail read: ``GET /api/notes/{note_id}/updates?after_seq=N`` over
  ``list_updates`` -- the section-4 replay tail a fresh or behind device
  replays after bootstrapping from the checkpoint body.

Design notes (the routes_notes house rules, mirrored):

- This is the user's own manual editor surface, NOT a model-reachable tool, so
  it carries no route-level mode gate (it mirrors ``routes_memory`` /
  ``routes_notes``); the security-mode middleware still applies its global
  posture to every path.
- It rides a SEPARATE router object (``note_updates_router``), never the
  ``notes_router``, so the five-routes-exact pins on ``routes_notes`` are
  untouched. It is authed exactly like ``notes_router`` (the current-user
  dependency on the router).
- The update blob is opaque: it crosses the wire base64-encoded and is stored
  as opaque bytes; the route never interprets it. No author or device identity
  rides the request payload (decision N9-D3): the per-(user, note) ``seq`` is
  minted by the store, and the local author's signature is attached by the
  sync engine at publish.
- The route delegates all persistence to the store; it issues no SQL itself.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .schemas import NoteUpdateAppendRequest, NoteUpdateRecordSchema

logger = logging.getLogger(__name__)

# Hardcoded checkpoint discipline for every new module; never overridable.
checkpoint_before_apply = True

_DEFAULT_TAIL_LIMIT = 1000

try:
    from opti_oignon.notes.note_updates_store import (
        NoteUpdateRefused,
        get_note_updates_store,
    )

    FEATURE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    FEATURE_AVAILABLE = False
    get_note_updates_store = None  # type: ignore[assignment]

    class NoteUpdateRefused(Exception):  # type: ignore[no-redef]
        """Fallback so the except clause below has a type to catch."""

        def __init__(self, reason: str, note_id: str = "") -> None:
            super().__init__(reason)
            self.reason = reason
            self.note_id = note_id


try:
    from .routes_auth import _get_current_user

    _updates_auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional

    _updates_auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


note_updates_router = APIRouter(
    prefix="/api/notes", tags=["notes"], dependencies=_updates_auth_dep
)


def _check_store() -> None:
    if not FEATURE_AVAILABLE or get_note_updates_store is None:
        raise HTTPException(
            status_code=503, detail="Note updates store not available"
        )


def _updates_store_dep() -> Any:
    """Resolve the coordinated ``NoteUpdatesStore`` singleton.

    A FastAPI dependency seam so tests inject a store through the singleton
    without touching the request path.
    """
    _check_store()
    return get_note_updates_store()


def _decode_blob(b64: str) -> bytes:
    """Decode the base64 opaque update; an invalid value is a 422, not a 500."""
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422, detail="update_blob_b64 is not valid base64"
        )


def _record_to_schema(record: Any) -> NoteUpdateRecordSchema:
    raw_blob = bytes(getattr(record, "update_blob", b"") or b"")
    return NoteUpdateRecordSchema(
        id=int(record.id),
        note_id=record.note_id,
        seq=int(record.seq),
        update_blob_b64=base64.b64encode(raw_blob).decode("ascii"),
        author_device=getattr(record, "author_device", None),
        created_at=record.created_at,
    )


@note_updates_router.post("/{note_id}/updates")
def append_note_update(
    note_id: str,
    request: NoteUpdateAppendRequest,
    store: Any = Depends(_updates_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> NoteUpdateRecordSchema:
    """Append one opaque update to ``note_id``; refuse fail-secure.

    The store mints the per-(user, note) ``seq`` and refuses an
    indeterminable, unknown, or dead parent (NOTES_CRDT_SPEC.md section 5);
    a refusal maps to a 409, never a silent 200. No device identity rides the
    payload (decision N9-D3): the engine attaches the local author's signature
    at publish through the store's best-effort sync glue.
    """
    user_id: str | None = current_user.get("sub")
    blob = _decode_blob(request.update_blob_b64)
    try:
        record = store.append_update(note_id, blob, user_id=user_id)
    except NoteUpdateRefused as exc:
        raise HTTPException(
            status_code=409,
            detail="Update refused: {0}".format(getattr(exc, "reason", "")),
        )
    return _record_to_schema(record)


@note_updates_router.get("/{note_id}/updates")
def list_note_updates(
    note_id: str,
    after_seq: int = 0,
    limit: int = _DEFAULT_TAIL_LIMIT,
    store: Any = Depends(_updates_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> list[NoteUpdateRecordSchema]:
    """Replay the surviving update tail of ``note_id`` (section 4).

    Returns the per-user-scoped updates with ``seq`` greater than
    ``after_seq``, in ascending ``seq`` order; a fresh or behind device
    bootstraps from the checkpoint body and then replays this tail.
    """
    user_id: str | None = current_user.get("sub")
    records = store.list_updates(
        note_id, user_id=user_id, after_seq=after_seq, limit=limit
    )
    return [_record_to_schema(r) for r in records]
