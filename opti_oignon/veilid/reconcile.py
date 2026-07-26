#!/usr/bin/env python3
"""Convergent reconciliation for Veilid sync.

Sync is convergent: every device holds the full set of records and reconciles
incoming changes, with no primary. This module is the merge. Given any number of
record sides (two devices' sets, or a delta against a local set), it produces one
converged set -- a single winning version per record key -- plus a conflict log
that retains the losing versions where an automatic merge would be unsafe.

The policy is last-writer-wins by logical clock with a deterministic tie-break.
For each key (a kind and an identity), the winner is the version with the highest
logical clock; ties on the clock are broken by content hash, then by device id
(then by timestamp as a final stable disambiguator), so the choice never depends
on the order the sides arrive in. A version superseded by a strictly higher clock
is a clean update and is discarded; that is safe. The unsafe case is concurrent
divergence -- two versions that tie on the winning clock but carry different
content, where neither happened-before the other. There the tie-break picks one
arbitrarily, so the loser is not thrown away: it is retained in the conflict log,
the "richer merge is unsafe" path, available for a later merge or human review.

Two structural properties hold and are tested. The merge is order-independent on
its inputs: the result is a function of the union of the sides, the per-key winner
is a maximum over that union, and both the converged set and the conflict log are
emitted in a deterministic key order, so swapping the sides yields an identical
result. And the merge is idempotent: reconciling a converged set with itself
returns the same set with an empty conflict log, because each key then has a single
candidate. Tombstones win like any other version, so a deletion converges across
devices rather than being silently resurrected.

This module is pure: it consumes already-decoded records (the defensive parsing is
records.decode_*), performs no I/O, opens no socket, and imports only the record
encoding, so it collects and runs anywhere. The Daily-only boundary lives at the
transport seam in the protocol envelope, not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from opti_oignon.veilid.records import SyncRecord, key_of

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

RecordKey = tuple[str, str]


@dataclass(frozen=True)
class ConflictEntry:
    """A retained concurrent divergence for one record key.

    Attributes:
        key: The reconciliation key (kind value, identity).
        winner: The version selected by the tie-break.
        retained: The losing versions at the winning clock, one per distinct losing
            content, deterministically ordered. These are kept because an automatic
            merge of concurrent edits is unsafe; nothing is discarded silently.
    """

    key: RecordKey
    winner: SyncRecord
    retained: list[SyncRecord] = field(default_factory=list)


@dataclass(frozen=True)
class MergeResult:
    """The outcome of a reconciliation.

    Attributes:
        records: The converged set, one winning version per key, ordered by key.
        conflicts: The conflict log, ordered by key; empty when no concurrent
            divergence occurred.
    """

    records: list[SyncRecord] = field(default_factory=list)
    conflicts: list[ConflictEntry] = field(default_factory=list)


def _selection_key(record: SyncRecord) -> tuple[int, str, str, str]:
    """The ordering key for last-writer-wins.

    Highest clock wins; ties break on content hash, then device id, then timestamp.
    The trailing fields make the choice fully deterministic and independent of the
    order the candidates were collected in, even for byte-identical duplicates.
    """
    return (record.clock, record.content_hash, record.device, record.updated_at)


def choose_winner(candidates: Iterable[SyncRecord]) -> SyncRecord:
    """Return the winning version among candidates for a single key.

    Raises ``ValueError`` on an empty candidate set; the reconciler only ever calls
    this with at least one candidate.
    """
    ordered = list(candidates)
    if not ordered:
        raise ValueError("choose_winner requires at least one candidate")
    return max(ordered, key=_selection_key)


def _retained_losers(
    candidates: list[SyncRecord], winner: SyncRecord
) -> list[SyncRecord]:
    """The retained losing versions for a key, or an empty list when none apply.

    Retained history is exactly the concurrent divergence: versions that tie on the
    winning clock but carry different content from the winner. One representative
    per distinct losing content is kept, deterministically ordered. Versions at a
    strictly lower clock are a clean supersession and are not retained.
    """
    at_winning_clock = [c for c in candidates if c.clock == winner.clock]
    distinct_contents = {c.content_hash for c in at_winning_clock}
    if len(distinct_contents) <= 1:
        return []
    losers: list[SyncRecord] = []
    seen: set[str] = set()
    for c in sorted(at_winning_clock, key=lambda r: (r.content_hash, r.device)):
        if c.content_hash == winner.content_hash or c.content_hash in seen:
            continue
        seen.add(c.content_hash)
        losers.append(c)
    return losers


def _resolve(by_key: dict[RecordKey, list[SyncRecord]]) -> MergeResult:
    """Resolve the grouped candidates into a converged set and a conflict log."""
    converged: list[SyncRecord] = []
    conflicts: list[ConflictEntry] = []
    for key in sorted(by_key.keys()):
        candidates = by_key[key]
        winner = choose_winner(candidates)
        converged.append(winner)
        losers = _retained_losers(candidates, winner)
        if losers:
            conflicts.append(
                ConflictEntry(key=key, winner=winner, retained=losers)
            )
    return MergeResult(records=converged, conflicts=conflicts)


def reconcile_many(sides: Iterable[Iterable[SyncRecord]]) -> MergeResult:
    """Reconcile any number of record sides into one converged set and a log.

    The result is a function of the union of all sides: candidates are grouped by
    key across every side, so the merge is order-independent both across sides and
    within a side.
    """
    by_key: dict[RecordKey, list[SyncRecord]] = {}
    for side in sides:
        for record in side:
            by_key.setdefault(key_of(record), []).append(record)
    return _resolve(by_key)


def reconcile(
    left: Iterable[SyncRecord], right: Iterable[SyncRecord]
) -> MergeResult:
    """Reconcile two record sides; a convenience over :func:`reconcile_many`."""
    return reconcile_many([left, right])
