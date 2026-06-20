#!/usr/bin/env python3
"""The notes voice-transcription trigger route (N.5): a thin HTTP surface over the
opt-in, sandboxed whisper.cpp transcription orchestration.

Design notes:

- A SEPARATE router (``notes_transcription_router`` at ``/api/notes/transcription``),
  NOT folded into the S249 ``notes_attachments_router`` -- so that router's
  ``test_five_routes_exact`` pin stays green (the ``routes_note_actions`` /
  ``routes_notes_attachments`` precedent) and this is a pure chain addition. The
  attachment surface moves blobs (CRUD); this surface triggers compute (post
  -processing), a distinct concern.
- ONE ``POST /{attachment_id}``. It resolves the live disposable SandboxManager
  and the live whisper.cpp transcriber (both host-assured; whisper.cpp and bwrap
  are absent in-container), hands them plus the per-user stores to
  ``transcribe_attachment``, and returns the structured result. The orchestration
  never raises, so refusals (the fail-secure sandbox gate, a missing / non-audio
  attachment, an unavailable blob) cross the wire as a 200 with ``refused`` True;
  the only HTTP error is the 503 availability guard (the notes surface did not
  import). Per-user via the existing auth dependency: a cross-user attachment id
  resolves to a structured ``not_found``, never a served transcript.
- The durable write-back requires ``approve=True`` in the body. The default
  (no body, or ``approve=False``) is a preview: the transcript is returned for
  review but NOT persisted. The plaintext audio never leaves the sandbox.

This route is not a model tool and runs no SQL; it delegates entirely to the
orchestration and the store. ``checkpoint_before_apply`` is hardcoded True and
never overridable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from .schemas import TranscriptionRequest, TranscriptionResultSchema

logger = logging.getLogger(__name__)

# Hardcoded checkpoint discipline for every new module; never overridable.
checkpoint_before_apply = True

try:
    from opti_oignon.notes.transcription import (
        build_live_transcriber,
        transcribe_attachment,
    )
    from opti_oignon.notes.notes_store import get_notes_store
    from opti_oignon.notes.blob_store import get_notes_blob_store

    FEATURE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    FEATURE_AVAILABLE = False
    transcribe_attachment = None  # type: ignore[assignment]
    build_live_transcriber = None  # type: ignore[assignment]
    get_notes_store = None  # type: ignore[assignment]
    get_notes_blob_store = None  # type: ignore[assignment]


try:
    from .routes_auth import _get_current_user

    _tr_auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional

    _tr_auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


notes_transcription_router = APIRouter(
    prefix="/api/notes/transcription", tags=["notes"], dependencies=_tr_auth_dep
)


def _check() -> None:
    if not FEATURE_AVAILABLE or get_notes_store is None or get_notes_blob_store is None:
        raise HTTPException(
            status_code=503, detail="Notes transcription surface not available"
        )


def _notes_store_dep():
    """Resolve the coordinated ``NotesStore`` singleton (a seam for tests)."""
    _check()
    return get_notes_store()


def _blob_store_dep():
    """Resolve the coordinated ``NotesBlobStore`` singleton (a seam for tests)."""
    _check()
    return get_notes_blob_store()


def _sandbox_dep():
    """Resolve the live disposable SandboxManager (a seam for tests).

    Imported lazily so the route module never forces the sandbox import chain at
    load. In-container the manager reports no bwrap, so the orchestration's fail
    -secure gate refuses -- the live disposable run is host-assured.
    """
    _check()
    from opti_oignon.sandbox_manager import SandboxManager

    return SandboxManager()


def _transcriber_dep():
    """Resolve the live whisper.cpp transcriber (a seam for tests).

    Returns None when the opt-in ``transcribe`` extra is absent (the default,
    in-container), which the orchestration turns into a structured refusal.
    """
    _check()
    return build_live_transcriber()


@notes_transcription_router.post(
    "/{attachment_id}", response_model=TranscriptionResultSchema
)
def transcribe(
    attachment_id: str,
    request: Optional[TranscriptionRequest] = None,
    store: Any = Depends(_notes_store_dep),
    blobs: Any = Depends(_blob_store_dep),
    sandbox: Any = Depends(_sandbox_dep),
    transcriber: Any = Depends(_transcriber_dep),
    current_user: dict = Depends(_get_current_user),
) -> TranscriptionResultSchema:
    """Transcribe one audio attachment inside a disposable sandbox.

    ``approve`` (body) gates the durable write-back; its default is the safe one
    (preview, no persist). The structured result crosses the wire (200) even for
    refusals; the only HTTP error here is the 503 availability guard.
    """
    approve = bool(request.approve) if request is not None else False
    user_id = current_user.get("sub")
    result = transcribe_attachment(
        attachment_id,
        user_id=user_id,
        store=store,
        blobs=blobs,
        sandbox=sandbox,
        transcriber=transcriber,
        approve=approve,
    )
    return TranscriptionResultSchema(**result.to_dict())
