#!/usr/bin/env python3
"""FastAPI notes route (N.2 backend half): bind the N.1 ``NotesStore`` to HTTP.

The Notes data layer (``opti_oignon/notes/``) and the
LLM-from-chat write surface (the gated ``manage_notes`` tool) came first. This
module is the HTTP surface the SvelteKit notes client (N.2 proper) rides: list /
get / create / update / delete over the per-user :class:`NotesStore`, registered
on the app exactly like the per-user ``memories_router``.

Design notes:

- Per-user isolation is the store's (``effective_user_id``). The route resolves
  the active user through the auth dependency exactly as ``routes_memory`` does,
  and passes ``user_id`` to every store call.
- Mode posture: this is the user's own manual surface, NOT a model-reachable
  tool, so it is NOT route-level mode-gated -- it mirrors ``routes_memory``,
  whose mutations carry no mode gate even though ``manage_memory`` is a
  Bulbe-forbidden tool. The Bulbe restriction on notes lives at the
  ``manage_notes`` tool layer; the user creates/edits notes manually in both
  modes (NOTES_FEATURE_ROADMAP). The security-mode middleware still applies its
  global posture (locality, no Bearer auth, search/plugin gates) to every path.
- The note body is an opaque, client-owned CRDT. The route never interprets it:
  it crosses the wire base64-encoded and is stored as opaque bytes. ``tags``
  cross as a JSON array (``list[str]``) and are stored as the opaque JSON-array
  string the store and the ``manage_notes`` tool already use. Cross-device CRDT
  relay is N.8; this route persists whole note state (the editor's save), it
  does not merge.
- The route delegates all persistence to the store; it issues no SQL itself.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .schemas import NoteCreateRequest, NoteSchema, NoteUpdateRequest

logger = logging.getLogger(__name__)

# Hardcoded checkpoint discipline for every new module; never overridable.
checkpoint_before_apply = True

_DEFAULT_LIST_LIMIT = 200

try:
    from opti_oignon.notes.notes_store import NotesStore, get_notes_store

    FEATURE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    FEATURE_AVAILABLE = False
    NotesStore = None  # type: ignore[assignment,misc]
    get_notes_store = None  # type: ignore[assignment]

try:
    # N.8: the section-4 compaction watermark is recorded through the
    # same append-only update store the update legs (routes_note_updates) use.
    from opti_oignon.notes.note_updates_store import get_note_updates_store

    _UPDATES_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    _UPDATES_AVAILABLE = False
    get_note_updates_store = None  # type: ignore[assignment]

try:
    from .routes_auth import _get_current_user

    _notes_auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional

    _notes_auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


notes_router = APIRouter(
    prefix="/api/notes", tags=["notes"], dependencies=_notes_auth_dep
)


def _check_store() -> None:
    if not FEATURE_AVAILABLE or get_notes_store is None:
        raise HTTPException(status_code=503, detail="Notes store not available")


def _notes_store_dep():
    """Resolve the coordinated ``NotesStore`` singleton.

    A FastAPI dependency seam so tests inject a store through
    ``app.dependency_overrides`` without touching the process singleton.
    """
    _check_store()
    return get_notes_store()


def _note_updates_store_dep() -> Any:
    """Resolve the ``NoteUpdatesStore`` singleton for the compaction watermark.

    A dependency seam (the s256 idiom) so the PATCH leg records the section-4
    checkpoint watermark through the same store the update legs use, while
    tests inject through the singleton. Returns ``None`` when the update store
    is unavailable, so the watermark recording is best-effort and never breaks
    a plain note update.
    """
    if not _UPDATES_AVAILABLE or get_note_updates_store is None:
        return None
    return get_note_updates_store()


def _tags_to_list(raw: Any) -> list[str]:
    """Decode the store's opaque tag string (a JSON array) to a list.

    Tolerant: a non-JSON or non-list value yields an empty list, never raises.
    """
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(t) for t in raw]
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if isinstance(parsed, list):
        return [str(t) for t in parsed]
    return []


def _tags_to_opaque(tags: list[str] | None) -> str | None:
    """Encode a tag list as the opaque JSON-array string the store stores.

    Mirrors the ``manage_notes`` tool's ``_notes_tags_value`` so the route and
    the tool write the same shape. ``None`` means "leave unchanged" (the update
    path); an empty list is a deliberate clear.
    """
    if tags is None:
        return None
    items = [str(t).strip() for t in tags if str(t).strip()]
    return json.dumps(items)


def _decode_body(b64: str | None) -> bytes:
    """Decode the base64 opaque body; an invalid value is a 422, not a 500."""
    if not b64:
        return b""
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422, detail="body_crdt_b64 is not valid base64"
        )


def _record_to_schema(record: Any) -> NoteSchema:
    raw_body = bytes(getattr(record, "body_crdt", b"") or b"")
    return NoteSchema(
        id=record.id,
        title=record.title,
        body_crdt_b64=base64.b64encode(raw_body).decode("ascii"),
        tags=_tags_to_list(getattr(record, "tags", "")),
        pinned=bool(record.pinned),
        created_at=record.created_at,
        updated_at=record.updated_at,
        deleted=bool(record.deleted),
        # N.9: getattr-defensive so a pre-N.9 record shape still maps
        # (fail-secure: absent reads False).
        mobile_allowed=bool(getattr(record, "mobile_allowed", False)),
    )


@notes_router.get("", response_model=list[NoteSchema])
def list_notes(
    pinned_only: bool = False,
    include_deleted: bool = False,
    limit: int = _DEFAULT_LIST_LIMIT,
    store: Any = Depends(_notes_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> list:
    """List the user's notes (most-recently-updated first)."""
    user_id = current_user.get("sub")
    records = store.list_notes(
        user_id=user_id,
        pinned_only=pinned_only,
        include_deleted=include_deleted,
        limit=limit,
    )
    return [_record_to_schema(r) for r in records]


@notes_router.post("", response_model=NoteSchema)
def create_note(
    request: NoteCreateRequest,
    store: Any = Depends(_notes_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> NoteSchema:
    """Create a note. The body is stored opaque; tags are an OR-Set JSON array."""
    if not request.title.strip():
        raise HTTPException(status_code=422, detail="Note title cannot be empty")
    user_id = current_user.get("sub")
    record = store.add_note(
        request.title,
        body_crdt=_decode_body(request.body_crdt_b64),
        tags=_tags_to_opaque(request.tags),
        pinned=bool(request.pinned),
        user_id=user_id,
    )
    return _record_to_schema(record)


@notes_router.get("/{note_id}", response_model=NoteSchema)
def get_note(
    note_id: str,
    store: Any = Depends(_notes_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> NoteSchema:
    """Fetch one note by id; a missing or tombstoned note is a 404."""
    user_id = current_user.get("sub")
    record = store.get_note(note_id, user_id=user_id)
    if record is None or record.deleted:
        raise HTTPException(status_code=404, detail="Note not found")
    return _record_to_schema(record)


@notes_router.patch("/{note_id}", response_model=NoteSchema)
def update_note(
    note_id: str,
    request: NoteUpdateRequest,
    store: Any = Depends(_notes_store_dep),
    updates_store: Any = Depends(_note_updates_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> NoteSchema:
    """Update title / body / tags / pinned. Omitted fields are left unchanged."""
    user_id = current_user.get("sub")
    fields: dict[str, Any] = {}
    if request.title is not None:
        if not request.title.strip():
            raise HTTPException(
                status_code=422, detail="Note title cannot be empty"
            )
        fields["title"] = request.title
    if request.body_crdt_b64 is not None:
        fields["body_crdt"] = _decode_body(request.body_crdt_b64)
    tags_value = _tags_to_opaque(request.tags)
    if tags_value is not None:
        fields["tags"] = tags_value
    if request.pinned is not None:
        fields["pinned"] = bool(request.pinned)
    # A missing or tombstoned note is a 404 rather than a silent no-op success.
    existing = store.get_note(note_id, user_id=user_id)
    if existing is None or existing.deleted:
        raise HTTPException(status_code=404, detail="Note not found")
    # N.9: the per-item phone-sync opt-in rides this existing PATCH
    # leg (no new route) and ONLY through the dedicated setter -- the flag
    # is deliberately not an updatable column, so the generic path below and
    # the gated manage_notes tool can never flip it (decision N9-D3). A
    # human trust decision made at the desktop.
    if request.mobile_allowed is not None:
        store.set_mobile_allowed(
            note_id, bool(request.mobile_allowed), user_id=user_id
        )
    record = store.update_note(note_id, user_id=user_id, **fields)
    if record is None:  # pragma: no cover - race
        raise HTTPException(status_code=404, detail="Note not found")
    # N.8: the section-4 compaction trigger. When the client folds the
    # update log into this whole-blob PATCH it carries the highest local seq
    # folded -- the checkpoint watermark. Record it through the update store
    # AFTER the body commit (the placement precedent), then prune the
    # folded tail lazily (prune_below_watermark never over-prunes: rows above
    # the watermark survive, and serving never depends on pruned history).
    # Omitted records nothing (fail-secure); a missing update store is a
    # best-effort no-op that never breaks the note update.
    if request.checkpoint_watermark is not None and updates_store is not None:
        updates_store.set_checkpoint_watermark(
            note_id, int(request.checkpoint_watermark), user_id=user_id
        )
        updates_store.prune_below_watermark(note_id, user_id=user_id)
    return _record_to_schema(record)


@notes_router.delete("/{note_id}")
def delete_note(
    note_id: str,
    store: Any = Depends(_notes_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Soft-delete (tombstone) so the deletion syncs (CRDT-safe)."""
    user_id = current_user.get("sub")
    ok = store.delete_note(note_id, user_id=user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"deleted": True, "id": note_id}
