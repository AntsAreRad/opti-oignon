"""Opti-Oignon Notes data layer (N.1).

The container-provable data layer for the Notes feature: a SQLCipher
metadata/text store with per-user isolation (:class:`~opti_oignon.notes.notes_store.NotesStore`)
and a two-layer per-attachment AES-256-GCM blob store
(:class:`~opti_oignon.notes.blob_store.NotesBlobStore`). This package is the data
layer ONLY -- the notes UI (N.2+), the LLM-from-note / LLM-from-chat surfaces
(N.3 / N.4, the gated ``manage_notes`` STATE_MUTATION tool), and the Veilid
record type (N.8) are later blocs.
"""

from __future__ import annotations

from .blob_store import (
    NotesBlobStore,
    NotesBlobUnavailable,
    get_notes_blob_store,
    reset_notes_blob_store,
)
from .notes_store import (
    AttachmentRecord,
    NoteRecord,
    NotesStore,
    get_notes_store,
    reset_notes_store,
)

__all__ = [
    "NotesStore",
    "NoteRecord",
    "AttachmentRecord",
    "get_notes_store",
    "reset_notes_store",
    "NotesBlobStore",
    "NotesBlobUnavailable",
    "get_notes_blob_store",
    "reset_notes_blob_store",
]
