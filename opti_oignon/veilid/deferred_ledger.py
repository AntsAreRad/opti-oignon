#!/usr/bin/env python3
"""Per-record deferred ledger for sensitive sync records.

Before this module, a sensitive record (a skill) denied at the approval gate
held the peer's watermark: the round stopped at the boundary before the
deferring chunk, so every later round re-fetched the whole growing delta and
re-prompted the human for the same record. A permanently-unapproved record
pinned convergence forever. The ledger is the fix: the round now APPLIES the
non-deferred records, ADVANCES the watermark past every consumed chunk, and
PERSISTS each deferred record here -- id plus the full wire envelope, so a
later re-offer needs no re-fetch (the protocol deliberately has no
fetch-by-id) and an approval can re-verify. The human acts from the SyncPanel
pending-approval list instead of a per-round modal storm.

Fail-secure semantics, non-negotiable: a deferred record is NOT applied. The
ledger never applies anything; it only holds. An approval re-enters the sync
engine's apply seam (verify -> gate -> apply) against the CURRENT trust state
-- a signing key that changed, or an origin demoted to pending, since the
record was deferred refuses at that seam, never applies. A refusal removes
the entry and applies nothing.

One pending decision per logical record: the primary key is (kind, record_id).
A candidate arriving for an already-pending key is arbitrated by the
reconciler's own public selection (``reconcile.choose_winner``), THE single
LWW recipe -- a strictly better version replaces the entry, a byte-identical
re-arrival refreshes ``last_offered_at`` silently (dedup-and-silence: no
re-prompt is the fix working), and a stale candidate is ignored. The engine
performs the local-clock staleness checks (skip-at-insert against
``current_clock``; purge-on-apply when a newer version lands through the
normal seam); the ledger stays feed-free pure storage.

At-rest posture, stated honestly: an UNAPPROVED sensitive record persists on
local disk in quarantine, in plain SQLite -- the same plaintext-without-
SQLCipher posture as the change feed and the peer registry (CHF-04 / PEER-01
family), documented and routed to the at-rest consistency lot (RS-01). The
envelope is stored verbatim, signature included, precisely so that approval
re-verification has the provenance it needs; nothing here is ever applied
without passing that seam.

The ledger is a local-disk structure, not a network one, so it is not gated
by the Bulbe boundary: listing, approving, and refusing entries are local
decisions permitted in any mode, like pairing management. Only the wire round
that fills the ledger is Daily-gated, at the engine and guard.

SQL hygiene, the project idiom (peers.py is the reference): SQLite in WAL
mode under the data directory; WAL set outside any transaction right after
connecting; every query parameterised; the single table identifier assembled
with ``str.format()`` under a frozenset allowlist, never an f-string.

Kerckhoffs: the ledger is open. There is no secret in the schema; what it
holds is quarantined content whose security lives in the signature it
carries and the registered keys it is re-verified against, not in this table.

The store is a process singleton with a reset hook (the SYN-04 idiom), and
its root is injectable so tests run against a temporary directory; the
data-directory import is lazy and guarded so the module collects without the
backend.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opti_oignon.veilid.reconcile import choose_winner
from opti_oignon.veilid.records import SyncRecord, decode_record, encode_record

# Encrypted DB connections, the same pattern as the
# veilid change feed and sync_queue. The deferred ledger quarantines
# sensitive synced record payloads awaiting human approval, so the quarantine
# joins safe_connect and is encrypted at rest rather than stored in a plain
# sqlite3 file. The in-container plaintext fallback below is the documented
# db_encryption degradation (SQLCipher absent), identical to every other
# safe_connect store.
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    logging.getLogger(__name__).warning(
        "db_utils unavailable: veilid deferred ledger falling back to PLAINTEXT "
        "sqlite3. The quarantined record payloads are NOT encrypted at rest."
    )
    _safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The one physical table. Referenced through the constant and validated against
# the allowlist below wherever it is interpolated into a statement.
TABLE_NAME = "veilid_deferred_records"
DB_FILENAME = "veilid_deferred.db"

# The allowlist for the only identifier ever formatted into SQL (the table name).
# This is the sanctioned alternative to an f-string: str.format() under a frozenset.
_TABLES: frozenset[str] = frozenset({TABLE_NAME})


def _safe_table(name: str) -> str:
    """Return the table name only if it is in the allowlist, else raise."""
    if name not in _TABLES:
        raise ValueError(f"table identifier not in allowlist: {name!r}")
    return name


# Offer outcomes (the gate counts inserted + replaced as deferred-this-round;
# a duplicate is the silent dedup; stale is an older candidate, ignored).
OFFER_INSERTED = "inserted"
OFFER_REPLACED = "replaced"
OFFER_DUPLICATE = "duplicate"
OFFER_STALE = "stale"

_CREATE_TABLE = (
    f"CREATE TABLE IF NOT EXISTS {_safe_table(TABLE_NAME)} ("
    "kind TEXT NOT NULL, "
    "record_id TEXT NOT NULL, "
    "origin_device TEXT NOT NULL, "
    "peer_id TEXT NOT NULL, "
    "clock INTEGER NOT NULL, "
    "content_hash TEXT NOT NULL, "
    "envelope TEXT NOT NULL, "
    "deferred_at TEXT NOT NULL, "
    "last_offered_at TEXT NOT NULL, "
    "PRIMARY KEY (kind, record_id)"
    ")"
)

_INSERT = (
    f"INSERT INTO {_safe_table(TABLE_NAME)} (kind, record_id, origin_device, peer_id, clock, "
    "content_hash, envelope, deferred_at, last_offered_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_REPLACE = (
    f"UPDATE {_safe_table(TABLE_NAME)} SET origin_device = ?, peer_id = ?, clock = ?, "
    "content_hash = ?, envelope = ?, deferred_at = ?, last_offered_at = ? "
    "WHERE kind = ? AND record_id = ?"
)

_TOUCH = (
    f"UPDATE {_safe_table(TABLE_NAME)} SET last_offered_at = ? WHERE kind = ? AND record_id = ?"
)

_SELECT_ONE = (
    "SELECT kind, record_id, origin_device, peer_id, clock, content_hash, "
    f"envelope, deferred_at, last_offered_at FROM {_safe_table(TABLE_NAME)} "
    "WHERE kind = ? AND record_id = ?"
)

_SELECT_ALL = (
    "SELECT kind, record_id, origin_device, peer_id, clock, content_hash, "
    f"envelope, deferred_at, last_offered_at FROM {_safe_table(TABLE_NAME)} "
    "ORDER BY deferred_at ASC, kind ASC, record_id ASC"
)

_SELECT_COUNT = f"SELECT COUNT(*) FROM {_safe_table(TABLE_NAME)}"

_DELETE_ONE = f"DELETE FROM {_safe_table(TABLE_NAME)} WHERE kind = ? AND record_id = ?"

_DELETE_FOR_PEER = f"DELETE FROM {_safe_table(TABLE_NAME)} WHERE peer_id = ?"

_DELETE_BELOW = (
    f"DELETE FROM {_safe_table(TABLE_NAME)} WHERE kind = ? AND record_id = ? AND clock < ?"
)

_DELETE_ALL = f"DELETE FROM {_safe_table(TABLE_NAME)}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DeferredEntry:
    """One pending approval: a quarantined sensitive record and its provenance.

    Attributes:
        kind: The record kind value (namespaces the identity).
        record_id: Stable identity within the kind.
        origin_device: The device that produced the version (``record.device``)
            -- the provenance approval re-verifies against.
        peer_id: The paired peer the round that deferred this ran against (the
            serving peer; a relayed record's origin may differ). Unpairing that
            peer cascades its entries away.
        clock: The version's logical clock, denormalised for staleness checks.
        content_hash: The version's content hash, denormalised for dedup.
        envelope: The full wire envelope (``encode_record`` output, signature
            included) -- enough to re-offer without a re-fetch and to re-verify
            at approval. ``{}`` when the stored JSON failed to parse (a corrupt
            row the approval path removes fail-secure).
        deferred_at: When this version entered (or replaced into) the ledger.
        last_offered_at: When this version last arrived on the wire (refreshed
            by the silent dedup).
    """

    kind: str
    record_id: str
    origin_device: str
    peer_id: str
    clock: int
    content_hash: str
    envelope: dict[str, Any]
    deferred_at: str
    last_offered_at: str


def _row_to_entry(row: Any) -> DeferredEntry:
    try:
        envelope = json.loads(row[6])
        if not isinstance(envelope, dict):
            envelope = {}
    except Exception:
        envelope = {}
    return DeferredEntry(
        kind=str(row[0]),
        record_id=str(row[1]),
        origin_device=str(row[2]),
        peer_id=str(row[3]),
        clock=int(row[4]),
        content_hash=str(row[5]),
        envelope=envelope,
        deferred_at=str(row[7]),
        last_offered_at=str(row[8]),
    )


class DeferredLedger:
    """Pure storage for deferred sensitive records, keyed (kind, record_id).

    The root is injectable for tests; with no root it resolves under the data
    directory. The connection is created lazily and guarded by a lock, so the
    ledger is safe to share across threads. The ledger never applies anything
    and never consults the change feed: arbitration between an incoming
    candidate and the stored entry reuses the reconciler's public
    ``choose_winner`` (THE LWW recipe); staleness against the LOCAL set is the
    engine's job.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root: Path | None = Path(root) if root is not None else None
        self._db_path: Path | None = None
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _resolve_db_path(self) -> Path:
        if self._root is not None:
            base = self._root
        else:
            from opti_oignon.config import DATA_DIR

            base = Path(DATA_DIR)
        return base / DB_FILENAME

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            self._db_path = self._resolve_db_path()
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = _safe_connect(
                str(self._db_path), check_same_thread=False, timeout=5.0
            )
            # WAL is set outside any transaction, right after connect.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.commit()
            self._connection = conn
        return self._connection

    @property
    def db_path(self) -> Path | None:
        return self._db_path

    def journal_mode(self) -> str:
        """The active journal mode; expected to be 'wal'."""
        with self._lock:
            conn = self._conn()
            row = conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]) if row else ""

    # Writes

    def offer(self, record: SyncRecord, *, peer_id: str) -> str:
        """Quarantine a deferred record; returns the offer outcome.

        ``inserted``: no entry existed for (kind, record_id); one was created.
        ``replaced``: the candidate beats the stored version under the
        reconciler's selection (``choose_winner``) and took its place --
        ``deferred_at`` resets, it is a new pending decision. ``duplicate``:
        the candidate is the stored version arriving again; only
        ``last_offered_at`` is refreshed (the silent dedup -- the caller must
        not re-prompt). ``stale``: the stored version beats the candidate;
        nothing is written. A stored envelope that no longer decodes is
        treated as beaten (replaced): fresh verified provenance over a
        corrupt row.
        """
        if not isinstance(record, SyncRecord):
            raise ValueError("record must be a SyncRecord")
        if not isinstance(peer_id, str) or not peer_id:
            raise ValueError("peer_id must be a non-empty string")
        envelope_json = json.dumps(
            encode_record(record), sort_keys=True, separators=(",", ":")
        )
        now = _now_iso()
        kind = record.kind.value
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_ONE, (kind, record.record_id)).fetchone()
            if row is None:
                conn.execute(
                    _INSERT,
                    (
                        kind,
                        record.record_id,
                        record.device,
                        peer_id,
                        int(record.clock),
                        record.content_hash,
                        envelope_json,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return OFFER_INSERTED
            if str(row[6]) == envelope_json:
                conn.execute(_TOUCH, (now, kind, record.record_id))
                conn.commit()
                return OFFER_DUPLICATE
            stored = decode_record(_row_to_entry(row).envelope)
            replace = stored is None or choose_winner([stored, record]) is record
            if not replace:
                return OFFER_STALE
            conn.execute(
                _REPLACE,
                (
                    record.device,
                    peer_id,
                    int(record.clock),
                    record.content_hash,
                    envelope_json,
                    now,
                    now,
                    kind,
                    record.record_id,
                ),
            )
            conn.commit()
            return OFFER_REPLACED

    def remove(self, kind: str, record_id: str) -> bool:
        """Remove an entry; returns True when a row was deleted."""
        if not isinstance(kind, str) or not isinstance(record_id, str):
            return False
        with self._lock:
            conn = self._conn()
            cur = conn.execute(_DELETE_ONE, (kind, record_id))
            conn.commit()
            return cur.rowcount > 0

    def remove_for_peer(self, peer_id: str) -> int:
        """Remove every entry deferred from a peer; returns how many.

        The unpair cascade: unpairing severs the trust the quarantine rode in
        on, so its pending decisions go with it -- fail-secure, a record from
        a removed peer must not remain one click from application. A record
        whose ORIGIN was unpaired (a relayed record) is covered by approval
        re-verification refusing against the missing key instead.
        """
        if not isinstance(peer_id, str) or not peer_id:
            return 0
        with self._lock:
            conn = self._conn()
            cur = conn.execute(_DELETE_FOR_PEER, (peer_id,))
            conn.commit()
            return int(cur.rowcount)

    def purge_below(self, kind: str, record_id: str, clock: int) -> bool:
        """Remove the entry for a key when its clock is strictly below ``clock``.

        The purge-on-apply hook: when a NEWER version of a key lands through
        the normal seam, the older quarantined version is obsolete by
        reconciliation (applying it would lose LWW, a no-op); keeping it would
        offer the human a dead decision. Strictly below: an equal-clock entry
        stands -- it may be the concurrent divergence the human should see.
        """
        if not isinstance(kind, str) or not isinstance(record_id, str):
            return False
        if isinstance(clock, bool) or not isinstance(clock, int):
            return False
        with self._lock:
            conn = self._conn()
            cur = conn.execute(_DELETE_BELOW, (kind, record_id, int(clock)))
            conn.commit()
            return cur.rowcount > 0

    def clear(self) -> None:
        """Remove every entry (the file and connection are kept)."""
        with self._lock:
            conn = self._conn()
            conn.execute(_DELETE_ALL)
            conn.commit()

    # Reads (local-disk; never gated)

    def get(self, kind: str, record_id: str) -> DeferredEntry | None:
        """The entry for a key, or None."""
        if not isinstance(kind, str) or not isinstance(record_id, str):
            return None
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_ONE, (kind, record_id)).fetchone()
        return _row_to_entry(row) if row is not None else None

    def has(self, kind: str, record_id: str) -> bool:
        """True when an entry exists for the key."""
        return self.get(kind, record_id) is not None

    def list_entries(self) -> list[DeferredEntry]:
        """Every pending entry, oldest deferred first."""
        with self._lock:
            conn = self._conn()
            rows = conn.execute(_SELECT_ALL).fetchall()
        return [_row_to_entry(r) for r in rows]

    def count(self) -> int:
        """The number of pending entries."""
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_COUNT).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the underlying connection, if open."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                finally:
                    self._connection = None


# Module-level singleton with a reset hook (one ledger per process, testable).
# SYN-04: creation is guarded by a lock, the same idiom as the change feed, the
# peer store, the status store, and the engine singletons.

_ledger: DeferredLedger | None = None
_ledger_lock = threading.Lock()


def get_deferred_ledger(root: Path | str | None = None) -> DeferredLedger:
    """Return the process deferred ledger, creating it once (with ``root`` if given)."""
    global _ledger
    with _ledger_lock:
        if _ledger is None:
            _ledger = DeferredLedger(root=root)
        return _ledger


def set_deferred_ledger(ledger: DeferredLedger | None) -> None:
    """Install a specific ledger as the process singleton (used by tests)."""
    global _ledger
    with _ledger_lock:
        _ledger = ledger


def reset_deferred_ledger() -> None:
    """Close and clear the process singleton so the next get creates a fresh one."""
    global _ledger
    with _ledger_lock:
        if _ledger is not None:
            _ledger.close()
        _ledger = None
