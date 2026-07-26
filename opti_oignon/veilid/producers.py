#!/usr/bin/env python3
"""Real producers for Veilid sync.

The encode side of sync: the functions that turn a domain object -- a
conversation, a canonical memory fact, an archive memory entry, or a skill --
into a ``SyncRecord`` ready for the change feed. The first cut carried a single minimal
producer (``sync_engine.record_from_payload`` over an opaque payload); this
module gives each syncable domain its own producer, so a round moves real data
rather than a hand-built record. The engine's ``publish_*`` convenience methods
journal the result locally.

Pure and defensive on the encode side. A producer reaches into no store and opens
no socket: it takes a stable identity within its kind, a domain payload (any
JSON-safe mapping), a source device, and a logical clock; it normalises the
payload defensively (``None`` becomes an empty mapping, a non-mapping is rejected
with a clear ``ValueError`` -- the producer-side contract, never untrusted wire
data), and delegates to :func:`records.new_record`, which computes the content
hash. Because the payload is opaque content, the producers are domain-free at the
wire level: the two memory tiers are two record kinds, so a canonical fact and an
archive entry never collide, and a deletion is a tombstone (an empty payload with
``deleted=True``) that converges like any other change.

Daily-only is not enforced here. Producing a record and journalling it are
local-disk operations, permitted in any mode (see ``change_feed``); only moving a
delta over the wire is Daily-only, and that gate lives in the protocol envelope
and the sync engine. These producers are the pure step before the gate.

Kerckhoffs: the encoding is open. There is no secret in a record; security for
sync lives in the keys and routes of the transport, not in the shape of a record.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from opti_oignon.veilid.records import RecordKind, SyncRecord, new_record

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True


def _normalise_payload(payload: Any) -> dict:
    """Coerce a domain payload to a plain dict, defensively.

    ``None`` becomes an empty mapping (a pure tombstone or a contentless record);
    a mapping is copied to a dict; anything else is a programmer error and raises
    ``ValueError``. This is the producer-side contract -- the defensive path for
    untrusted wire data is the record decoder, which never raises.
    """
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError("payload must be a mapping or None")


def _produce(
    kind: RecordKind,
    record_id: str,
    payload: Any,
    *,
    device: str,
    clock: int,
    deleted: bool,
    updated_at: str,
) -> SyncRecord:
    """Build a record of a fixed kind from a normalised payload. Pure."""
    return new_record(
        kind,
        record_id,
        _normalise_payload(payload),
        device=device,
        clock=clock,
        deleted=deleted,
        updated_at=updated_at,
    )


def conversation_record(
    conversation_id: str,
    payload: Mapping[str, Any] | None = None,
    *,
    device: str,
    clock: int,
    deleted: bool = False,
    updated_at: str = "",
) -> SyncRecord:
    """A conversation as a sync record (``RecordKind.CONVERSATION``)."""
    return _produce(
        RecordKind.CONVERSATION,
        conversation_id,
        payload,
        device=device,
        clock=clock,
        deleted=deleted,
        updated_at=updated_at,
    )


def note_record(
    note_id: str,
    payload: Mapping[str, Any] | None = None,
    *,
    device: str,
    clock: int,
    deleted: bool = False,
    updated_at: str = "",
) -> SyncRecord:
    """A note as a sync record (``RecordKind.NOTE``), the conversation sibling.

    A note is the user's own content (N.8): it applies on a peer without the human
    gate, exactly like a conversation, and unlike a skill. The payload is opaque
    domain content -- the note body rides as an E2E-encrypted CRDT blob and the
    tags / attachment set ride as their OR-Set shape -- so this producer, like the
    others, interprets nothing: it normalises the payload defensively and computes
    the content hash, leaving the CRDT merge to the client and the record-level
    convergence to the reconciler.
    """
    return _produce(
        RecordKind.NOTE,
        note_id,
        payload,
        device=device,
        clock=clock,
        deleted=deleted,
        updated_at=updated_at,
    )


def note_update_record(
    note_id: str,
    seq: int,
    payload: Mapping[str, Any] | None = None,
    *,
    device: str,
    clock: int,
    updated_at: str = "",
) -> SyncRecord:
    """One opaque CRDT update as a sync record (``RecordKind.NOTE_UPDATE``).

    The note kind's sibling (NOTES_CRDT_SPEC.md section 3): one Yjs increment
    of a note body, riding the existing sync envelope. The identity is
    ``note_id:seq`` -- ``seq`` is the author's per-``(user, note)`` append
    order, the platform's only ordering duty (section 4), so each update is
    its own immutable record key and the record-level reconciliation never
    rewrites history. The payload is opaque domain content (the update blob
    rides base64-encoded alongside its coordinates); this producer, like the
    others, interprets nothing. Deliberately STRUCTURAL: there is no
    ``deleted`` parameter and the record is never a tombstone -- an update
    leaves the world only by the local pruning rules of section 4, and a
    note's deletion syncs as the ``note`` kind already does.
    """
    if not isinstance(note_id, str) or not note_id:
        raise ValueError("note_id must be a non-empty string")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
        raise ValueError("seq must be a positive integer")
    return _produce(
        RecordKind.NOTE_UPDATE,
        f"{note_id}:{seq}",
        payload,
        device=device,
        clock=clock,
        deleted=False,
        updated_at=updated_at,
    )


def memory_canonical_record(
    fact_id: str,
    payload: Mapping[str, Any] | None = None,
    *,
    device: str,
    clock: int,
    deleted: bool = False,
    updated_at: str = "",
) -> SyncRecord:
    """A canonical memory fact as a sync record (``RecordKind.MEMORY_CANONICAL``).

    The structured fact store tier; distinct in kind from the archive tier, so the
    two never collide on identity.
    """
    return _produce(
        RecordKind.MEMORY_CANONICAL,
        fact_id,
        payload,
        device=device,
        clock=clock,
        deleted=deleted,
        updated_at=updated_at,
    )


def memory_archive_record(
    entry_id: str,
    payload: Mapping[str, Any] | None = None,
    *,
    device: str,
    clock: int,
    deleted: bool = False,
    updated_at: str = "",
) -> SyncRecord:
    """An archive memory entry as a sync record (``RecordKind.MEMORY_ARCHIVE``).

    The full, searchable archive tier; distinct in kind from the canonical tier.
    """
    return _produce(
        RecordKind.MEMORY_ARCHIVE,
        entry_id,
        payload,
        device=device,
        clock=clock,
        deleted=deleted,
        updated_at=updated_at,
    )


def skill_record(
    skill_id: str,
    payload: Mapping[str, Any] | None = None,
    *,
    device: str,
    clock: int,
    deleted: bool = False,
    updated_at: str = "",
) -> SyncRecord:
    """A skill as a sync record (``RecordKind.SKILL``).

    A skill carries executable surface, so applying one received over sync is a
    sensitive action gated by the engine; producing one locally is not.
    """
    return _produce(
        RecordKind.SKILL,
        skill_id,
        payload,
        device=device,
        clock=clock,
        deleted=deleted,
        updated_at=updated_at,
    )


def tombstone_record(
    kind: Any,
    record_id: str,
    *,
    device: str,
    clock: int,
    updated_at: str = "",
) -> SyncRecord:
    """A tombstone for any kind: an empty payload with ``deleted=True``.

    A converged deletion. ``kind`` accepts a :class:`RecordKind` or its value; an
    unknown kind raises (producer-side), the same as :func:`records.new_record`.
    """
    return new_record(
        kind,
        record_id,
        {},
        device=device,
        clock=clock,
        deleted=True,
        updated_at=updated_at,
    )
