#!/usr/bin/env python3
"""FastAPI notes-attachment route (the media blocs' shared backend prerequisite):
expose the N.1 ``attachment`` manifest and the two-layer ``NotesBlobStore`` over
HTTP.

The Notes data layer landed at S243 (``opti_oignon/notes/``) and already carries
the full media data layer for all three kinds: the ``attachment`` manifest table
(with ``transcript_text`` / ``caption_text`` / ``ocr_text``), the
``{audio, image, drawing}`` kind allowlist, and the per-attachment AES-256-GCM
``NotesBlobStore`` (the second independent at-rest layer beside the SQLCipher
metadata store). The piece missing to open the N.5 voice / N.6 picture vault /
N.7 drawing front is the shared HTTP surface that moves the encrypted blobs and
the manifest rows; this module is that surface, the client the capture / gallery
/ canvas UIs (later blocs) ride.

Design notes:

- A SEPARATE router. The attachment endpoints live on ``notes_attachments_router``
  (prefix ``/api/notes/attachments``), NOT on the S245 ``notes_router`` -- the
  ``routes_note_actions`` precedent. Folding them into ``notes_router`` would grow
  its route set and break the S245 five-routes pin; keeping them separate makes
  this a pure chain addition. Registered on the app exactly like ``notes_router``.
- Per-user isolation is the store's (``effective_user_id``). The route resolves
  the active user through the auth dependency exactly as ``routes_notes`` does and
  passes ``user_id`` to every store and blob call, so one user cannot read,
  download, or delete another's attachment (a cross-user id is a 404, never a
  served blob).
- The two-layer at-rest guarantee is the blob store's. The route hands plaintext
  to ``NotesBlobStore.seal`` (AES-256-GCM under a per-attachment HKDF-domain
  subkey, a fresh nonce per blob, ciphertext written atomically, no plaintext
  temp file) and reads it back with ``open`` (decrypted in memory only). The
  manifest carries the kind, mime, byte size, and the on-disk nonce; the bytes
  never touch the database. With no master key the seal refuses
  (``NotesBlobUnavailable``) and the route returns a 503 -- it never persists a
  plaintext blob.
- The route issues no SQL. It delegates persistence to the store
  (``add_attachment`` / ``get_attachment`` / ``list_attachments`` /
  ``delete_attachment``) and the bytes to the blob store. The ``kind`` is
  validated against the store's allowlist BEFORE sealing, so a rejected kind
  leaves no orphan blob and is a 422, not a 500.
- Not a model-reachable tool. This is the user's own manual media surface, like
  ``routes_notes``; it defines no ``ToolSchema`` and registers nothing in the
  agent tool registry. The opt-in, sandboxed whisper.cpp transcription (N.5) and
  vision caption / OCR (N.6) that fill ``transcript_text`` / ``caption_text`` /
  ``ocr_text`` are later blocs and run in the disposable bubblewrap; this route
  does no post-processing.
- The one HTTP error code beyond 404 / 422 is the availability guard (503),
  mirroring ``routes_notes._check_store``.

``checkpoint_before_apply`` is hardcoded True and never overridable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile

from .schemas import AttachmentDeleteResponse, AttachmentSchema

logger = logging.getLogger(__name__)

# Hardcoded checkpoint discipline for every new module; never overridable.
checkpoint_before_apply = True

try:
    from opti_oignon.notes.blob_store import (
        NotesBlobUnavailable,
        get_notes_blob_store,
    )
    from opti_oignon.notes.notes_store import (
        ATTACHMENT_KINDS,
        NotesStore,
        get_notes_store,
    )

    FEATURE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    FEATURE_AVAILABLE = False
    NotesStore = None  # type: ignore[assignment,misc]
    get_notes_store = None  # type: ignore[assignment]
    get_notes_blob_store = None  # type: ignore[assignment]
    ATTACHMENT_KINDS = frozenset()  # type: ignore[assignment]

    class NotesBlobUnavailable(RuntimeError):  # type: ignore[no-redef]
        pass


try:
    from .routes_auth import _get_current_user

    _att_auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional

    _att_auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


notes_attachments_router = APIRouter(
    prefix="/api/notes/attachments", tags=["notes"], dependencies=_att_auth_dep
)


def _check() -> None:
    if not FEATURE_AVAILABLE or get_notes_store is None or get_notes_blob_store is None:
        raise HTTPException(status_code=503, detail="Notes attachment surface not available")


def _notes_store_dep():
    """Resolve the coordinated ``NotesStore`` singleton (a seam for tests)."""
    _check()
    return get_notes_store()


def _blob_store_dep():
    """Resolve the coordinated ``NotesBlobStore`` singleton (a seam for tests)."""
    _check()
    return get_notes_blob_store()


def _attachment_to_schema(record: Any) -> AttachmentSchema:
    return AttachmentSchema(
        id=record.id,
        note_id=record.note_id,
        kind=record.kind,
        mime=record.mime,
        byte_size=int(record.byte_size),
        nonce=record.nonce,
        created_at=record.created_at,
        transcript_text=record.transcript_text,
        caption_text=record.caption_text,
        ocr_text=record.ocr_text,
    )


@notes_attachments_router.post("/note/{note_id}", response_model=AttachmentSchema)
async def upload_attachment(
    note_id: str,
    kind: str = Form(...),
    file: UploadFile = File(...),
    store: Any = Depends(_notes_store_dep),
    blobs: Any = Depends(_blob_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> AttachmentSchema:
    """Seal an uploaded media blob and add its manifest row.

    The note must exist and not be tombstoned; ``kind`` is validated against the
    store's allowlist BEFORE sealing (a rejected kind is a 422 with no orphan
    blob). The bytes are sealed by the blob store under a per-attachment subkey;
    with no master key the seal refuses and this is a 503, never a plaintext
    write.
    """
    user_id = current_user.get("sub")
    note = store.get_note(note_id, user_id=user_id)
    if note is None or note.deleted:
        raise HTTPException(status_code=404, detail="Note not found")
    if kind not in ATTACHMENT_KINDS:
        raise HTTPException(status_code=422, detail="Invalid attachment kind: " + repr(kind))

    payload = await file.read()
    import uuid as _uuid

    aid = _uuid.uuid4().hex
    try:
        blobs.seal(aid, payload)
    except NotesBlobUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        record = store.add_attachment(
            note_id,
            kind,
            blob_ref=aid,
            mime=file.content_type or "",
            byte_size=len(payload),
            nonce=blobs.nonce_of(aid).hex(),
            user_id=user_id,
            attachment_id=aid,
        )
    except ValueError as exc:  # pragma: no cover - kind pre-validated above
        blobs.delete(aid)
        raise HTTPException(status_code=422, detail=str(exc))
    return _attachment_to_schema(record)


@notes_attachments_router.get("/note/{note_id}", response_model=list[AttachmentSchema])
def list_attachments(
    note_id: str,
    store: Any = Depends(_notes_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> list:
    """List one note's attachment manifests (oldest first)."""
    user_id = current_user.get("sub")
    records = store.list_attachments(note_id, user_id=user_id)
    return [_attachment_to_schema(r) for r in records]


@notes_attachments_router.get("/{attachment_id}", response_model=AttachmentSchema)
def get_attachment_meta(
    attachment_id: str,
    store: Any = Depends(_notes_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> AttachmentSchema:
    """Fetch one attachment's manifest; a missing or cross-user id is a 404."""
    user_id = current_user.get("sub")
    record = store.get_attachment(attachment_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return _attachment_to_schema(record)


@notes_attachments_router.get("/{attachment_id}/blob")
def download_attachment(
    attachment_id: str,
    store: Any = Depends(_notes_store_dep),
    blobs: Any = Depends(_blob_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> Response:
    """Stream the decrypted blob bytes (in memory only).

    Per-user: the manifest is fetched first, so a missing or cross-user id is a
    404 before any blob is opened. The bytes are decrypted in memory; no
    plaintext temp file is produced.
    """
    user_id = current_user.get("sub")
    record = store.get_attachment(attachment_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    try:
        raw = blobs.open(attachment_id)
    except NotesBlobUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Attachment blob not found")
    return Response(content=raw, media_type=record.mime or "application/octet-stream")


@notes_attachments_router.delete(
    "/{attachment_id}", response_model=AttachmentDeleteResponse
)
def delete_attachment(
    attachment_id: str,
    store: Any = Depends(_notes_store_dep),
    blobs: Any = Depends(_blob_store_dep),
    current_user: dict = Depends(_get_current_user),
) -> AttachmentDeleteResponse:
    """Delete an attachment: the encrypted blob first, then the manifest row.

    Per-user: a missing or cross-user id is a 404. The blob is removed before the
    manifest so a missing-key blob store cannot leave a dangling manifest.
    """
    user_id = current_user.get("sub")
    record = store.get_attachment(attachment_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    blobs.delete(attachment_id)
    store.delete_attachment(attachment_id, user_id=user_id)
    return AttachmentDeleteResponse(deleted=True, id=attachment_id)
