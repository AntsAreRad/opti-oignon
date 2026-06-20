#!/usr/bin/env python3
"""Versioned record encoding for Veilid sync (S179 Goal 1, Theme 4).

The transport-ready encoding for the syncable record types: conversations, the
two-tier memory (a canonical tier and an archive tier, carried as two kinds), and
the skills registry. A record is the unit the sync protocol moves between a user's
own devices; this module turns a record into a stable, self-describing wire object
and back, and computes the content hash the reconciler uses.

Every record carries five things the rest of the bloc depends on: a stable
identity within its kind, a logical clock (a scalar version that the reconciler
orders by, last-writer-wins), a content hash over the record's content, the source
device that produced the version, and a kind tag that namespaces the identity so
two domains never collide. A record may also be a tombstone (``deleted``), so a
deletion converges like any other change rather than silently dropping data.

Two properties matter for the protocol built on top. The encoding round-trips:
``decode_record(encode_record(r))`` returns an equal record. And decoding is
defensive: it never raises into a caller. An object that is not a well-formed
record -- wrong format version, missing or mistyped fields, or a content hash that
does not match its content -- is rejected (returned as ``None`` / counted), not
crashed on. Incoming sync data is data, not trusted input; the self-consistency
check on the hash is the integrity gate at the edge.

Kerckhoffs: the encoding is open. There is no secret in the format; the content
hash is a plain SHA-256 over the canonical JSON of the content, and security for
sync lives in the keys and routes of the transport, not in the shape of a record.

This module is pure and domain-free: the payload is opaque content (any JSON-safe
mapping), so the encoder never reaches into a store and never opens a socket. It
imports only the standard library, so the package collects with it in any
environment, and it is safe to run anywhere -- the Daily-only boundary is enforced
by the protocol envelope at the transport seam, not by these helpers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The wire format version. Bumped only on an incompatible change to the encoding;
# a record that does not carry exactly this version is rejected on decode.
RECORD_FORMAT_VERSION = 1


class RecordKind(str, Enum):
    """The kind of a syncable record; namespaces the record identity.

    Memory is two-tier, so it contributes two kinds: ``memory_canonical`` for the
    structured fact store and ``memory_archive`` for the full, searchable archive.
    Conversations, the skills registry, and notes are one kind each. A note is the
    user's own content (N.8), the conversation kind's sibling: it applies without
    the human gate (only the executable ``skill`` kind is gated), and its body is
    carried as an opaque, E2E-encrypted CRDT blob the engine never interprets.
    A note update (``note_update``, NOTES_CRDT_SPEC.md section 3) is the note
    kind's sibling: one opaque Yjs increment of a note body, identified as
    ``note_id:seq``, applied without the human gate like the note itself, and
    never a tombstone -- an update leaves the world only by the local pruning
    rules of the spec's section 4, never by a converged deletion.
    """

    CONVERSATION = "conversation"
    MEMORY_CANONICAL = "memory_canonical"
    MEMORY_ARCHIVE = "memory_archive"
    SKILL = "skill"
    NOTE = "note"
    NOTE_UPDATE = "note_update"


# A frozenset of the kind values; doubles as the allowlist the decoder checks
# membership against, so an unknown kind is rejected rather than constructed.
RECORD_KINDS: frozenset[str] = frozenset(k.value for k in RecordKind)


@dataclass(frozen=True)
class SyncRecord:
    """One version of one syncable item, ready for transport.

    Attributes:
        kind: The record kind; namespaces the identity.
        record_id: Stable identity within the kind.
        clock: Logical clock / version; the reconciler orders by this (higher
            wins), so a producer bumps it when it changes the record.
        device: The source device that produced this version.
        content_hash: SHA-256 hex digest over the canonical content
            (kind, id, payload, deleted); the reconciler's tie-break and the
            decoder's integrity check.
        payload: Opaque, JSON-safe domain content. Empty for a pure tombstone.
        deleted: True for a tombstone (a converged deletion).
        updated_at: Informational ISO-8601 timestamp; not used for ordering.
        signature: Base64url ML-DSA-65 signature over the record's canonical
            bytes (S205, VL-01), or "" for an unsigned (pre-VL-01) record. The
            authenticity layer above the content hash: it binds kind, id,
            clock, device, hash, payload, deleted, and updated_at to the
            origin device's signing key, so re-clocking or re-attributing a
            signed record breaks it. Attached at publish, carried verbatim on
            relay (provenance preserved end to end), verified at the engine's
            apply seam against the origin device's registered key -- never
            here in the pure layer, which stays key-free.
    """

    kind: RecordKind
    record_id: str
    clock: int
    device: str
    content_hash: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    deleted: bool = False
    updated_at: str = ""
    signature: str = ""


@dataclass(frozen=True)
class DecodeResult:
    """The outcome of decoding a batch: the parseable records and a reject count."""

    records: list[SyncRecord]
    rejected: int


def _coerce_kind(kind: Any) -> RecordKind:
    """Return a :class:`RecordKind` for a kind or its value, else raise.

    Used on the producer side (``new_record`` / ``content_hash_for``); the
    defensive decoder validates membership itself and never relies on this raising.
    """
    if isinstance(kind, RecordKind):
        return kind
    if isinstance(kind, str) and kind in RECORD_KINDS:
        return RecordKind(kind)
    raise ValueError(f"unknown record kind: {kind!r}")


def content_hash_for(
    kind: Any, record_id: str, payload: Mapping[str, Any], deleted: bool
) -> str:
    """The content hash for a record: SHA-256 over the canonical content JSON.

    The hash covers the kind, the identity, the payload, and the tombstone flag --
    the content, not the metadata (clock, device, timestamp). It is computed over a
    canonical JSON serialisation (sorted keys, tight separators), so it is
    independent of key insertion order and stable across a JSON round-trip for
    JSON-safe payloads. The clock and device are deliberately excluded so that the
    same content from two devices hashes identically, which is what the reconciler
    tie-break relies on.
    """
    canonical = {
        "kind": _coerce_kind(kind).value,
        "id": record_id,
        "payload": payload,
        "deleted": bool(deleted),
    }
    blob = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def new_record(
    kind: Any,
    record_id: str,
    payload: Mapping[str, Any],
    *,
    device: str,
    clock: int,
    deleted: bool = False,
    updated_at: str = "",
) -> SyncRecord:
    """Build a :class:`SyncRecord`, computing its content hash.

    The producer-side constructor. It validates its inputs and raises ``ValueError``
    on a programmer error (an unknown kind, an empty identity, a non-integer clock,
    and so on); the defensive path for untrusted wire data is :func:`decode_record`,
    which never raises. ``bool`` is rejected for the clock even though it is an
    ``int`` subclass, so ``True`` is never mistaken for a version.
    """
    k = _coerce_kind(kind)
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a non-empty string")
    if isinstance(clock, bool) or not isinstance(clock, int) or clock < 0:
        raise ValueError("clock must be a non-negative integer")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    if not isinstance(deleted, bool):
        raise ValueError("deleted must be a bool")
    if not isinstance(updated_at, str):
        raise ValueError("updated_at must be a string")
    payload_d = dict(payload)
    digest = content_hash_for(k, record_id, payload_d, deleted)
    return SyncRecord(
        kind=k,
        record_id=record_id,
        clock=clock,
        device=device,
        content_hash=digest,
        payload=payload_d,
        deleted=deleted,
        updated_at=updated_at,
    )


def key_of(record: SyncRecord) -> tuple[str, str]:
    """The reconciliation key for a record: (kind value, identity)."""
    return (record.kind.value, record.record_id)


def verify_record_hash(record: SyncRecord) -> bool:
    """True when a record's content hash matches its content; never raises.

    This is an integrity check, not authenticity: the content hash detects
    corruption, but any peer can compute the correct SHA-256 for a forged
    payload. The authenticity layer is the per-record signature (S205, VL-01):
    :func:`canonical_record_bytes` defines what it covers, ``veilid/signing.py``
    holds the keys, and the sync engine verifies on receive against the origin
    device's registered key. The content hash keeps its role unchanged
    (storage/decoder integrity gate, reconciler tie-break).
    """
    try:
        expected = content_hash_for(
            record.kind, record.record_id, record.payload, record.deleted
        )
        return expected == record.content_hash
    except Exception:  # pragma: no cover - defensive
        return False


def canonical_record_bytes(record: SyncRecord) -> bytes:
    """The canonical bytes a record signature covers (S205, VL-01).

    THE one signing recipe, documented here and used by signer and verifier
    alike: the sorted-key compact JSON (UTF-8) of the encoded wire record
    minus the ``signature`` field itself -- that is, v, kind, id, clock,
    device, hash, payload, deleted, and updated_at. The clock and device are
    deliberately INSIDE (unlike the content hash): re-clocking or
    re-attributing a signed record must break its signature, or a
    paired-but-compromised peer could steer LWW merges with forged provenance
    (the VL-01 threat). Stable across versions by construction: the encoder
    emits only the known fields, so unknown wire fields never enter the
    recipe, and sorted keys make it independent of insertion order. Pure and
    key-free, like the rest of this module.
    """
    wire = encode_record(record)
    wire.pop("signature", None)
    blob = json.dumps(wire, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return blob.encode("utf-8")


def encode_record(record: SyncRecord) -> dict[str, Any]:
    """Encode a record to its self-describing wire object (a plain dict).

    The ``signature`` field is emitted only when non-empty (S205, the epoch
    omit idiom): an unsigned record's wire shape is byte-identical to the
    pre-VL-01 shape, so old readers see exactly what they always saw and a
    pre-VL-01 sender is indistinguishable by construction.
    """
    wire: dict[str, Any] = {
        "v": RECORD_FORMAT_VERSION,
        "kind": record.kind.value,
        "id": record.record_id,
        "clock": record.clock,
        "device": record.device,
        "hash": record.content_hash,
        "payload": dict(record.payload),
        "deleted": record.deleted,
        "updated_at": record.updated_at,
    }
    if record.signature:
        wire["signature"] = record.signature
    return wire


def decode_record(obj: Any) -> Optional[SyncRecord]:
    """Decode one wire object to a record, or return ``None`` on any problem.

    Defensive throughout: a non-mapping, an unknown or mismatched format version, a
    missing or mistyped field, an unknown kind, or a content hash that does not
    match the content all yield ``None``. It never raises into the caller. The
    integrity check (recomputing the hash and comparing) is what keeps a poisoned
    record from being trusted: a tampered payload no longer matches its hash and is
    rejected here, before the reconciler ever sees it.
    """
    try:
        if not isinstance(obj, Mapping):
            return None
        if obj.get("v") != RECORD_FORMAT_VERSION:
            return None
        kind_raw = obj.get("kind")
        if not isinstance(kind_raw, str) or kind_raw not in RECORD_KINDS:
            return None
        kind = RecordKind(kind_raw)
        record_id = obj.get("id")
        if not isinstance(record_id, str) or not record_id:
            return None
        clock = obj.get("clock")
        if isinstance(clock, bool) or not isinstance(clock, int) or clock < 0:
            return None
        device = obj.get("device")
        if not isinstance(device, str) or not device:
            return None
        content_hash = obj.get("hash")
        if not isinstance(content_hash, str) or not content_hash:
            return None
        payload = obj.get("payload")
        if not isinstance(payload, Mapping):
            return None
        deleted = obj.get("deleted", False)
        if not isinstance(deleted, bool):
            return None
        updated_at = obj.get("updated_at", "")
        if not isinstance(updated_at, str):
            return None
        # S205 (VL-01): the signature is read defensively. A missing or
        # mistyped field means an unsigned record ("") -- NEVER a parse
        # reject. Parsing stays the ungated integrity gate it always was;
        # whether an unsigned or invalid signature is acceptable depends on
        # the origin device's registered key, which only the engine's apply
        # seam knows (peer context lives there, not here).
        signature_raw = obj.get("signature", "")
        signature = signature_raw if isinstance(signature_raw, str) else ""
        payload_d = dict(payload)
        expected = content_hash_for(kind, record_id, payload_d, deleted)
        if expected != content_hash:
            return None
        return SyncRecord(
            kind=kind,
            record_id=record_id,
            clock=clock,
            device=device,
            content_hash=content_hash,
            payload=payload_d,
            deleted=deleted,
            updated_at=updated_at,
            signature=signature,
        )
    except Exception:
        logger.debug("Rejected an unparseable sync record", exc_info=True)
        return None


def encode_records(records: Iterable[SyncRecord]) -> list[dict[str, Any]]:
    """Encode a sequence of records to a list of wire objects."""
    return [encode_record(r) for r in records]


def decode_records(objs: Any) -> DecodeResult:
    """Decode a sequence of wire objects, separating the parseable from the rejected.

    Never raises: each item that fails to decode is counted in ``rejected``
    rather than aborting the batch, and a non-iterable input yields an empty
    result counted as one rejection (PRT-03: garbage-in must stay
    distinguishable from a legitimately empty batch).
    """
    records: list[SyncRecord] = []
    rejected = 0
    try:
        items = list(objs)
    except Exception:
        return DecodeResult(records=[], rejected=1)
    for obj in items:
        rec = decode_record(obj)
        if rec is None:
            rejected += 1
        else:
            records.append(rec)
    return DecodeResult(records=records, rejected=rejected)


def to_wire_json(records: Iterable[SyncRecord]) -> str:
    """Serialise records to a compact JSON array of wire objects.

    Producer side; may raise if a payload is not JSON-safe.
    """
    return json.dumps(
        encode_records(records), separators=(",", ":"), ensure_ascii=False
    )


def from_wire_json(text: Any) -> DecodeResult:
    """Parse a JSON array of wire objects defensively into a :class:`DecodeResult`.

    Never raises: invalid JSON, or a top-level value that is not an array, yields
    an empty result counted as one rejection (PRT-03: garbage-in must stay
    distinguishable from a legitimately empty batch).
    """
    try:
        data = json.loads(text)
    except Exception:
        return DecodeResult(records=[], rejected=1)
    if not isinstance(data, list):
        return DecodeResult(records=[], rejected=1)
    return decode_records(data)
