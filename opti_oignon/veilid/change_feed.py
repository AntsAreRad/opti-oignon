#!/usr/bin/env python3
"""Per-device change journal for Veilid sync.

A device journals every change it makes to a syncable record, so a peer can ask
"what have you changed since this point?" and receive only the delta. The journal
is an append-only log of record versions, each at a monotonic sequence number; the
sequence is the watermark a peer holds. A delta request carries the peer's last
seen sequence and gets back the versions written after it, collapsed to the latest
per key, plus the journal's current high-water mark to advance the peer's watermark.

The journal is backed by SQLite in WAL mode under the data directory, with the
project's SQL hygiene: WAL is set outside any transaction right after connecting;
queries are parameterised; and the single table identifier is assembled with
``str.format()`` under a frozenset allowlist, never an f-string, so no
caller-controlled string ever reaches a statement. Each row stores the record's
fields and a JSON payload; a row is read back through the record decoder, so the
hash integrity check applies to the journal too and a corrupt row is skipped rather
than trusted.

Journalling is a local-disk operation, not a network one, so it is not gated by
the Bulbe boundary: a device may record its own changes in any mode. Only moving a
delta over the wire is Daily-only, and that gate lives in the protocol envelope at
the transport seam, not here.

The journal compacts transparently: a row superseded by a later
sequence for the same (kind, record_id) can be deleted without changing any
watermark's delta, the high-water, or any key's current clock, because ``since()``
collapses to the latest per key, the global MAX(seq) row is by construction the
latest of its key, and every writer appends a key's rows at non-decreasing clocks.
Every append applies that rule at once to the keys it appends: a record's earlier
versions are erased in the append's own transaction, so the journal keeps one
version of each record, and a tombstone -- a deletion -- erases every version the
record had; after one, the write-ahead log is emptied. A version whose clock is
higher than an ordinary version appended after it stays: a writer appending out of
order never sends a record's clock back; a tombstone erases it all the same. What
is erased is overwritten (``secure_delete``), never left in the file's free pages.
A conversation's deletion also erases every version of it before its commit
(``forget_record``), so its words leave this device even when the tombstone
cannot be appended after. A note's deletion erases its versions and every update
of its body (``forget_note``), which no tombstone ever takes out; the sweep of the
dead notes erases what older builds left of them (``forget_dead_notes``). ``compact()`` erases what older builds left superseded,
by the same rule (``tend``, at boot), plus an optional every-N-appends trigger
(``compact_every``, off by default); like journalling it is local-disk maintenance,
permitted in any mode, never gated.

The journal also serves bounded pages: ``since_page`` reads at most
a caller-given count of rows after a watermark, stops at a caller-given wire-byte
budget (always keeping one row so a page never stalls), and reports the page's max
sequence as its high-water, so a peer walks the journal chunk by chunk over the
existing monotonic-advance semantics. ``since`` (the whole delta in one read) is
unchanged and still serves the CHF-01 backstop for an impossible watermark; the
paged read serves the same backstop in bounded pages.

The journal carries an identity: a random feed epoch, minted once
per journal file in a one-row meta table (the SYN-02 identity-row idiom) and read
through ``feed_epoch``. The epoch travels in the batch envelope so an asker can
detect that a peer's journal was recreated (the file remade, sequences restarted)
and repair its watermark with a single full resync, instead of riding the CHF-01
backstop forever. ``clear()`` keeps the epoch on purpose: AUTOINCREMENT preserves
the sequence counter across a delete-all, so clearing is not a reset in the CHF-01
sense; only recreating the file mints a new epoch, by construction, because the
meta table lives in the same file. Compaction names the feed table only and never
touches the meta table.

The journal is a process singleton with a reset hook, and its root is injectable so
tests run against a temporary directory; the data-directory import is lazy and
guarded so the module collects without the backend.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from opti_oignon.veilid.records import (
    RECORD_FORMAT_VERSION,
    RecordKind,
    SyncRecord,
    decode_record,
    encode_record,
    key_of,
    verify_record_hash,
)

# Audit fix: use encrypted DB connections (same pattern as sync_queue.py).
# The change-feed journal holds the synced record payloads (conversations, memory,
# etc.), so it must be encrypted at rest like the rest of the data layer rather than
# stored in a plain sqlite3 file.
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    logging.getLogger(__name__).warning(
        "db_utils unavailable: veilid change_feed falling back to PLAINTEXT "
        "sqlite3. Synced record payloads are NOT encrypted at rest."
    )
    _safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)

logger = logging.getLogger(__name__)

checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# The single physical table. Referenced through this constant and validated
# against the allowlist below wherever it is interpolated into a statement.
TABLE_NAME = "veilid_change_feed"
DB_FILENAME = "veilid_change_feed.db"
# How long a write waits for another connection's lock.
_BUSY_TIMEOUT_MS = 5000

# The one-row feed-identity meta table: the journal's random epoch,
# minted once per journal file. A second physical identifier in this database; it
# joins the allowlist below (the peers.py two-table precedent), while the
# compaction statement keeps naming the feed table only, so compaction never
# touches the meta table by construction.
META_TABLE_NAME = "veilid_feed_meta"

# The clock floors: for a record whose versions a deletion erased before its tombstone landed, the highest clock it
# erased -- ids and numbers, never a word. ``current_clock`` honours it, so the tombstone, minted above it, out-clocks
# every version a peer may hold; an incoming version at or below it is the deleted one coming back
# (``protocol._apply_batch``). The version that lands at or above it, the tombstone first, drops it.
FLOOR_TABLE_NAME = "veilid_feed_floors"

# The allowlist for the only identifiers ever formatted into SQL (table names).
# This is the sanctioned alternative to an f-string: str.format() under a frozenset.
_TABLES: frozenset[str] = frozenset({TABLE_NAME, META_TABLE_NAME, FLOOR_TABLE_NAME})


def _safe_table(name: str) -> str:
    """Return the table name only if it is in the allowlist, else raise."""
    if name not in _TABLES:
        raise ValueError(f"table identifier not in allowlist: {name!r}")
    return name


_CREATE_TABLE = (
    f"CREATE TABLE IF NOT EXISTS {_safe_table(TABLE_NAME)} ("
    "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
    "kind TEXT NOT NULL, "
    "record_id TEXT NOT NULL, "
    "clock INTEGER NOT NULL, "
    "device TEXT NOT NULL, "
    "content_hash TEXT NOT NULL, "
    "deleted INTEGER NOT NULL DEFAULT 0, "
    "updated_at TEXT NOT NULL DEFAULT '', "
    "payload TEXT NOT NULL DEFAULT '{}', "
    "journaled_at TEXT NOT NULL, "
    "signature TEXT"
    ")"
)

_CREATE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_veilid_change_feed_key "
    f"ON {_safe_table(TABLE_NAME)}(kind, record_id)"
)

_INSERT = (
    f"INSERT INTO {_safe_table(TABLE_NAME)} "
    "(kind, record_id, clock, device, content_hash, deleted, updated_at, "
    "payload, journaled_at, signature) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_SELECT_COLUMNS = (
    "seq, kind, record_id, clock, device, content_hash, deleted, updated_at, "
    "payload, signature"
)

_SELECT_SINCE = (
    f"SELECT {_SELECT_COLUMNS} FROM {_safe_table(TABLE_NAME)} WHERE seq > ? ORDER BY seq ASC"
)

# The bounded page read. Same WHERE/ORDER as _SELECT_SINCE with a
# row cap; the byte cap is applied in Python over the fetched rows so the page's
# wire size is bounded by the same encoding the envelope ships. Names the feed
# table only, through the allowlist.
_SELECT_SINCE_LIMIT = (
    f"SELECT {_SELECT_COLUMNS} FROM {_safe_table(TABLE_NAME)} WHERE seq > ? ORDER BY seq ASC LIMIT ?"
)

_SELECT_ALL = (
    f"SELECT {_SELECT_COLUMNS} FROM {_safe_table(TABLE_NAME)} ORDER BY seq ASC"
)

_SELECT_MAX_SEQ = f"SELECT MAX(seq) FROM {_safe_table(TABLE_NAME)}"
# A record's versions before the one just appended: the supersession rule below, applied at once to that key. A
# version whose clock is higher stays, so a record's clock never goes back even if a writer appends out of order.
_DELETE_EARLIER = (
    f"DELETE FROM {_safe_table(TABLE_NAME)} WHERE kind = ? AND record_id = ? AND seq < ? AND clock <= ?"
)
# A tombstone's: every version before it, whatever its clock -- a deletion leaves none of the record's words.
_DELETE_ALL_EARLIER = (
    f"DELETE FROM {_safe_table(TABLE_NAME)} WHERE kind = ? AND record_id = ? AND seq < ?"
)
# Every version of one record: a deletion's erasure before its commit.
_DELETE_RECORD = (
    f"DELETE FROM {_safe_table(TABLE_NAME)} WHERE kind = ? AND record_id = ?"
)
_CREATE_FLOORS = (
    f"CREATE TABLE IF NOT EXISTS {_safe_table(FLOOR_TABLE_NAME)} ("
    "kind TEXT NOT NULL, record_id TEXT NOT NULL, clock INTEGER NOT NULL, PRIMARY KEY (kind, record_id))"
)
_SELECT_FLOOR = f"SELECT clock FROM {_safe_table(FLOOR_TABLE_NAME)} WHERE kind = ? AND record_id = ?"
_RAISE_FLOOR = (
    f"INSERT INTO {_safe_table(FLOOR_TABLE_NAME)} (kind, record_id, clock) VALUES (?, ?, ?) "
    "ON CONFLICT(kind, record_id) DO UPDATE SET clock = MAX(clock, excluded.clock)"
)
# A version appended at or above a floor carries the clock itself: the floor goes.
_DROP_FLOOR = f"DELETE FROM {_safe_table(FLOOR_TABLE_NAME)} WHERE kind = ? AND record_id = ? AND clock <= ?"
# The latest version of one record.
_SELECT_ONE_LATEST = (
    f"SELECT {_SELECT_COLUMNS} FROM {_safe_table(TABLE_NAME)} WHERE kind = ? AND record_id = ? "
    "ORDER BY seq DESC LIMIT 1"
)
# The keys of a kind a deletion left a floor for and whose tombstone was never journalled.
_SELECT_OWED = (
    f"SELECT record_id FROM {_safe_table(FLOOR_TABLE_NAME)} WHERE kind = ? AND clock > 0 "
    f"AND record_id NOT IN (SELECT record_id FROM {_safe_table(TABLE_NAME)} WHERE kind = ?)"
)
_SELECT_COUNT = f"SELECT COUNT(*) FROM {_safe_table(TABLE_NAME)}"
_DELETE_ALL = f"DELETE FROM {_safe_table(TABLE_NAME)}"

# The read side of per-key clocks. Uses the
# existing (kind, record_id) index; a read-only query, no schema change.
_SELECT_MAX_CLOCK = (
    f"SELECT MAX(clock) FROM {_safe_table(TABLE_NAME)} WHERE kind = ? AND record_id = ?"
)

# A note's updates are the records ``note_id:seq``: the keys from ``note_id:`` up to, not including, ``note_id;``
# (the character after the colon), a range the (kind, record_id) index serves.
_DELETE_RANGE = (
    f"DELETE FROM {_safe_table(TABLE_NAME)} WHERE kind = ? AND record_id >= ? AND record_id < ?"
)
_SELECT_IDS = f"SELECT DISTINCT record_id FROM {_safe_table(TABLE_NAME)} WHERE kind = ?"
_SELECT_LATEST = (
    f"SELECT deleted FROM {_safe_table(TABLE_NAME)} WHERE kind = ? AND record_id = ? ORDER BY seq DESC LIMIT 1"
)

# The transparent supersession rule -- delete any row superseded
# by a later sequence for the same (kind, record_id) whose clock is as high, or
# by a later tombstone: the rule each append applies to its own key. With every
# writer appending at non-decreasing clocks, only the MAX(seq) row of a key stays;
# a row of a higher clock that an out-of-order append followed stays too, so no
# key's clock goes back. One bounded statement, served by the existing
# (kind, record_id) index (``seq`` is the rowid alias every secondary index
# carries). Names the feed table only, through the allowlist, so a future meta
# table (CHF-05) is untouched by construction.
_DELETE_SUPERSEDED = (
    f"DELETE FROM {_safe_table(TABLE_NAME)} WHERE EXISTS (SELECT 1 FROM {_safe_table(TABLE_NAME)} AS later "
    f"WHERE later.kind = {_safe_table(TABLE_NAME)}.kind AND later.record_id = {_safe_table(TABLE_NAME)}.record_id "
    f"AND later.seq > {_safe_table(TABLE_NAME)}.seq "
    f"AND (later.clock >= {_safe_table(TABLE_NAME)}.clock OR later.deleted = 1))"
)

# The one-row meta table holding the feed epoch, minted once per
# journal file. INSERT OR IGNORE under the CHECK(id = 1) constraint keeps the
# first minted value under concurrency -- the SYN-02 identity-row idiom
# (peers.py, veilid_local_identity). These are the only statements that name the
# meta table; the compaction statement above names the feed table only.
_CREATE_META = (
    f"CREATE TABLE IF NOT EXISTS {_safe_table(META_TABLE_NAME)} ("
    "id INTEGER PRIMARY KEY CHECK (id = 1), "
    "epoch TEXT NOT NULL, "
    "created_at TEXT NOT NULL"
    ")"
)

_SELECT_EPOCH = (
    f"SELECT epoch FROM {_safe_table(META_TABLE_NAME)} WHERE id = 1"
)

_INSERT_EPOCH = (
    f"INSERT OR IGNORE INTO {_safe_table(META_TABLE_NAME)} (id, epoch, created_at) VALUES (1, ?, ?)"
)


@dataclass(frozen=True)
class Delta:
    """A delta served to a peer: the changed records and the new watermark.

    Attributes:
        records: The latest version per key written after the requested watermark.
        high_water: The journal's current maximum sequence; the peer advances its
            watermark to this once it has consumed the delta.
    """

    records: list[SyncRecord] = field(default_factory=list)
    high_water: int = 0


class ChangeFeed:
    """An append-only, per-device journal of syncable record versions.

    The root is injectable for tests; with no root it resolves under the data
    directory. The connection is created lazily and guarded by a lock, so the
    journal is safe to share across threads.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        compact_every: int | None = None,
    ) -> None:
        # The optional every-N-appends compaction trigger. OFF by
        # default (None) -- a convenience, not the guarantee; ``compact()`` is the
        # on-demand entry point. The counter is in-process on purpose: a trigger
        # missed across a restart costs nothing.
        if compact_every is not None:
            if (
                isinstance(compact_every, bool)
                or not isinstance(compact_every, int)
                or compact_every < 1
            ):
                raise ValueError("compact_every must be a positive integer or None")
        self._compact_every: int | None = compact_every
        self._appends_since_compact: int = 0
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
                str(self._db_path), check_same_thread=False, timeout=_BUSY_TIMEOUT_MS / 1000
            )
            # WAL is set outside any transaction, right after connect.
            conn.execute("PRAGMA journal_mode=WAL")
            # A version erased is overwritten, never left in the file's free pages.
            conn.execute("PRAGMA secure_delete = ON")
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)
            # Add the signature column to journals that predate
            # it. SQLite has no ADD COLUMN IF NOT EXISTS, so guard with
            # table_info. Nullable on purpose: NULL or
            # '' means an unsigned (pre-VL-01) row.
            cols = {
                row[1]
                for row in conn.execute(
                    f"PRAGMA table_info({_safe_table(TABLE_NAME)})"
                ).fetchall()
            }
            if "signature" not in cols:
                conn.execute(
                    f"ALTER TABLE {_safe_table(TABLE_NAME)} ADD COLUMN signature TEXT"
                )
            conn.execute(_CREATE_META)
            conn.execute(_CREATE_FLOORS)
            conn.commit()
            self._connection = conn
        return self._connection

    @property
    def db_path(self) -> Path:
        if self._db_path is None:
            self._db_path = self._resolve_db_path()
        return self._db_path

    def journal_mode(self) -> str:
        """The active journal mode; expected to be 'wal'."""
        with self._lock:
            conn = self._conn()
            row = conn.execute("PRAGMA journal_mode").fetchone()
        return (row[0] if row else "").lower()

    @staticmethod
    def _row_params(record: SyncRecord, journaled_at: str) -> tuple:
        """The INSERT parameter tuple for a record (shared by record/record_many)."""
        payload_json = json.dumps(
            dict(record.payload), separators=(",", ":"), ensure_ascii=False
        )
        return (
            record.kind.value,
            record.record_id,
            record.clock,
            record.device,
            record.content_hash,
            1 if record.deleted else 0,
            record.updated_at,
            payload_json,
            journaled_at,
            # The signature is journalled with the record
            # (sign-at-publish): signed once per local edit, and a received,
            # verified winner keeps its originator's signature -- provenance
            # preserved end to end. NULL for an unsigned (pre-VL-01) row.
            record.signature or None,
        )

    def record(self, record: SyncRecord) -> int:
        """Append a record version to the journal and return its sequence.

        Refuses a record whose content hash does not match its content, so the
        journal never stores an inconsistent row. The record's earlier versions
        are erased in the same transaction; after a tombstone the log is emptied
        (``_empty_log``). Local-disk only; never gated.
        """
        if not isinstance(record, SyncRecord):
            raise TypeError("record() requires a SyncRecord")
        if not verify_record_hash(record):
            raise ValueError("refusing to journal a record with a mismatched hash")
        now = datetime.now(timezone.utc).isoformat()
        params = self._row_params(record, now)
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute(_INSERT, params)
                seq = int(cur.lastrowid)
                if record.deleted:
                    conn.execute(_DELETE_ALL_EARLIER, (record.kind.value, record.record_id, seq))
                else:
                    conn.execute(_DELETE_EARLIER, (record.kind.value, record.record_id, seq, record.clock))
                conn.execute(_DROP_FLOOR, (record.kind.value, record.record_id, record.clock))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            if record.deleted:
                self._empty_log(conn)
            # The trigger ticks after the commit, under the lock;
            # the sequence is captured first and returned whatever the trigger does.
            self._maybe_autocompact(conn, 1)
        _tombstones_journalled([record])
        return seq

    def record_many(self, records: Iterable[SyncRecord]) -> list[int]:
        """Append several record versions in one transaction; sequences in order.

        All-or-nothing (the per-record-commit note folded): every
        record is type-, hash- and serialisation-verified before anything is
        inserted, the batch lands under a single commit, and a mid-batch
        failure rolls the whole batch back and raises -- nothing is journalled,
        so the caller's at-least-once retry re-journals the full batch. Each
        record's earlier versions are erased in the same transaction, the
        batch's own included; after a tombstone the log is emptied. The
        every-N trigger counts the batch once, after the commit.
        """
        recs = list(records)
        for r in recs:
            if not isinstance(r, SyncRecord):
                raise TypeError("record_many() requires SyncRecords")
            if not verify_record_hash(r):
                raise ValueError(
                    "refusing to journal a record with a mismatched hash"
                )
        if not recs:
            return []
        now = datetime.now(timezone.utc).isoformat()
        params = [self._row_params(r, now) for r in recs]
        with self._lock:
            conn = self._conn()
            seqs: list[int] = []
            try:
                for r, p in zip(recs, params):
                    cur = conn.execute(_INSERT, p)
                    seqs.append(int(cur.lastrowid))
                    if r.deleted:
                        conn.execute(_DELETE_ALL_EARLIER, (r.kind.value, r.record_id, seqs[-1]))
                    else:
                        conn.execute(_DELETE_EARLIER, (r.kind.value, r.record_id, seqs[-1], r.clock))
                    conn.execute(_DROP_FLOOR, (r.kind.value, r.record_id, r.clock))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            if any(r.deleted for r in recs):
                self._empty_log(conn)
            self._maybe_autocompact(conn, len(recs))
        _tombstones_journalled(recs)
        return seqs

    def _empty_log(self, conn: sqlite3.Connection) -> None:
        """Empty the write-ahead log after a deletion: its older frames still hold the versions the tombstone
        erased. Assumes ``self._lock`` is held. Never waits: a read in progress keeps the log, said, until the next
        deletion or the connection's close; the connection's own wait is given back after."""
        try:
            wait = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
            conn.execute("PRAGMA busy_timeout = 0")
            try:
                busy = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
            finally:
                conn.execute(f"PRAGMA busy_timeout = {wait}")
        except Exception as exc:  # noqa: BLE001 - the cipher driver's errors are not sqlite3's, and this only says
            logger.warning("change feed: a record deleted; the write-ahead log could not be emptied (%s: %s)",
                           type(exc).__name__, exc)
            return
        if busy:
            logger.warning("change feed: a record deleted; a read in progress keeps the write-ahead log, whose older "
                           "frames hold what the deletion erased, until the next deletion")

    def _row_to_record(self, rest: tuple) -> SyncRecord | None:
        (
            kind,
            record_id,
            clock,
            device,
            content_hash,
            deleted,
            updated_at,
            payload,
            signature,
        ) = rest
        try:
            payload_obj = json.loads(payload) if payload else {}
        except Exception:
            logger.debug("Skipping a journal row with an unparseable payload")
            return None
        wire = {
            "v": RECORD_FORMAT_VERSION,
            "kind": kind,
            "id": record_id,
            "clock": clock,
            "device": device,
            "hash": content_hash,
            "payload": payload_obj,
            "deleted": bool(deleted),
            "updated_at": updated_at,
        }
        # A stored signature rides back through the decoder; a
        # NULL/'' column stays an unsigned record (the field is omitted, the
        # pre-VL-01 wire shape). The decoder's hash check still gates the row.
        if isinstance(signature, str) and signature:
            wire["signature"] = signature
        return decode_record(wire)

    def _collapse_latest(self, rows: Iterable[tuple]) -> list[SyncRecord]:
        # rows are ordered by seq ascending, so a later seq overwrites an earlier
        # one for the same key; the dict ends holding the latest per key.
        latest: dict[tuple[str, str], SyncRecord] = {}
        skipped = 0
        for row in rows:
            rec = self._row_to_record(row[1:])
            if rec is None:
                skipped += 1
                continue
            latest[key_of(rec)] = rec
        if skipped:
            # CHF-03: a corrupt journal must not shrink the served set silently.
            # The per-row detail stays at debug (_row_to_record); the aggregate
            # count is surfaced once per read.
            logger.warning(
                "change feed: skipped %d corrupt journal row(s) on read "
                "(payload unparseable or content-hash mismatch)",
                skipped,
            )
        return list(latest.values())

    def high_water(self) -> int:
        """The current maximum sequence in the journal, or 0 when empty."""
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_MAX_SEQ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def count(self) -> int:
        """The number of journal rows."""
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_COUNT).fetchone()
        return int(row[0]) if row else 0

    def current_clock(self, kind: RecordKind | str, record_id: str) -> int:
        """The highest clock journalled for a record key, or 0 when unseen.

        The read side of clock discipline: a domain hook computes
        the clock for a local edit as ``current_clock(kind, key) + 1``. The
        journal is the merged latest view (``apply_record_batch`` journals
        winners, including clock-only adoptions since PRT-02), so the local
        journal is the correct basis for a local edit; an unseen key yields 0,
        so the first clock is 1. MAX(clock) rather than the latest row's clock
        is the defensive choice: under the current writers the two coincide
        (a local mint is current+1; an applied winner's clock is >= the local
        one), but MAX guarantees a minted clock never collides with anything
        ever journalled for the key, tombstones included -- a re-create after a
        delete out-clocks the tombstone and wins the LWW merge. A key whose
        versions a deletion erased keeps its highest erased clock as a floor
        (``forget_record``), honoured here: the tombstone minted after it
        out-clocks every version a peer may hold. Local-disk read; never gated.
        """
        k = kind.value if isinstance(kind, RecordKind) else str(kind)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_MAX_CLOCK, (k, record_id)).fetchone()
            floor = conn.execute(_SELECT_FLOOR, (k, record_id)).fetchone()
        journalled = int(row[0]) if row and row[0] is not None else 0
        return max(journalled, int(floor[0]) if floor else 0)

    def clock_floor(self, kind: RecordKind | str, record_id: str) -> int:
        """The clock floor a deletion left a record whose versions it erased, 0 when there is none: an incoming
        version at or below it is the deleted one coming back."""
        k = kind.value if isinstance(kind, RecordKind) else str(kind)
        with self._lock:
            conn = self._conn()
            floor = conn.execute(_SELECT_FLOOR, (k, record_id)).fetchone()
        return int(floor[0]) if floor else 0

    def feed_epoch(self) -> str:
        """This journal's stable identity, minted once per journal file (CHF-05).

        A random epoch id (uuid4 hex) created lazily in the one-row meta table
        next to the feed -- the SYN-02 identity-row idiom, INSERT OR IGNORE
        keeping the first minted value under concurrency. The epoch travels in
        the batch envelope so an asker can detect that this journal was
        recreated (the file remade, sequences restarted at 1) and reset its
        watermark for a single full resync, instead of riding the CHF-01
        backstop forever. The value is public material (Kerckhoffs): a fresh
        random label, not a secret. ``clear()`` keeps the epoch on purpose --
        AUTOINCREMENT preserves the sequence counter across a delete-all, so
        clearing is not a reset in the CHF-01 sense; only recreating the file
        mints a new epoch, by construction, because the meta table lives in
        the same file. Compaction never touches this row (it names the feed
        table only). Local-disk read; never gated.
        """
        candidate = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._conn()
            conn.execute(_INSERT_EPOCH, (candidate, now))
            conn.commit()
            row = conn.execute(_SELECT_EPOCH).fetchone()
        return str(row[0]) if row is not None and row[0] else candidate

    def since(self, watermark: int) -> Delta:
        """The delta written after ``watermark``: latest per key, plus high-water.

        ``watermark`` is a sequence; ``since(0)`` returns the whole current set and
        ``since(high_water())`` returns nothing. The high-water in the result is the
        journal's overall maximum, so the peer advances past every consumed
        sequence, including versions collapsed away.

        A watermark beyond the journal's high-water is impossible against this
        journal's history: it means this journal was reset (the file recreated, so
        sequences restarted at 1) or the asker's stored watermark is corrupt.
        Serving nothing there would let the asker silently skip everything until
        the sequence caught up -- a permanent divergence, and a re-pair does not
        repair it (the upsert deliberately preserves the watermark). The backstop
        (CHF-01) serves the full current set instead: applying is idempotent, so
        the repetition costs bandwidth, never correctness, and the devices
        converge. The asker's watermark stays put (its monotonic advance ignores
        the lower high-water) until the sequence passes it; the real repair -- a
        feed epoch in the envelope with a per-peer epoch reset -- is a wire-format
        change recorded for the VL-01 cycle.
        """
        w = int(watermark)
        with self._lock:
            conn = self._conn()
            hw_row = conn.execute(_SELECT_MAX_SEQ).fetchone()
            high = int(hw_row[0]) if hw_row and hw_row[0] is not None else 0
            if w > high:
                rows = conn.execute(_SELECT_ALL).fetchall()
            else:
                rows = conn.execute(_SELECT_SINCE, (w,)).fetchall()
        if w > high:
            logger.warning(
                "change feed: asker watermark %d exceeds journal high-water %d "
                "(journal reset or corrupt watermark); serving the full current "
                "set as a convergence backstop (CHF-01)",
                w,
                high,
            )
        return Delta(records=self._collapse_latest(rows), high_water=high)

    @staticmethod
    def _wire_size(record: SyncRecord) -> int:
        """The serialised wire-byte size of one record, the same encoding the
        envelope ships (``encode_record`` -> compact JSON, UTF-8)."""
        blob = json.dumps(
            encode_record(record), separators=(",", ":"), ensure_ascii=False
        )
        return len(blob.encode("utf-8"))

    def since_page(
        self, watermark: int, *, max_count: int, max_bytes: int
    ) -> Delta:
        """One bounded page of the delta after ``watermark``.

        Like :meth:`since` but bounded: it reads at most ``max_count`` rows after
        the watermark (in ascending sequence) and stops accumulating once the
        serialised wire size would exceed ``max_bytes``, always keeping at least
        one row when any exist so a page never stalls (the progress guarantee).
        The page's ``high_water`` is the sequence of the last row it consumed (the
        chunk's max sequence), NOT the journal's overall maximum -- so an asker
        threading the received high-water into its next request walks the journal
        chunk by chunk over the existing monotonic-advance semantics, with no new
        token. A key whose versions span two pages is served in both and applies
        twice, idempotent by the LWW merge.

        Collapse is within the page: the rows kept are reduced to the latest per
        key, exactly as :meth:`since` collapses, so the shipped record set is a
        subset of the bounded rows and never exceeds ``max_bytes``. A corrupt row
        is skipped from the record set (CHF-03, surfaced once) but still advances
        the page's high-water past its sequence, so a corrupt row can never stall
        the cursor.

        CHF-01 backstop: a watermark beyond the journal's high-water reads from
        the start (effective watermark 0) and serves the full current set in
        bounded pages, with the same warning :meth:`since` emits. The asker's
        persisted watermark stays put under its monotonic advance until the real
        repair (the CHF-05 feed epoch), so the full set is re-served each round --
        bandwidth, never correctness.

        An empty page (the asker is caught up, or the journal is empty) reports
        ``high_water`` equal to the journal's maximum, matching the empty-delta
        contract of :meth:`since` so a caught-up round is a no-op under the
        monotonic advance. Local-disk read; never gated.
        """
        if isinstance(max_count, bool) or not isinstance(max_count, int) or max_count < 1:
            raise ValueError("max_count must be a positive integer")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        w = int(watermark)
        with self._lock:
            conn = self._conn()
            hw_row = conn.execute(_SELECT_MAX_SEQ).fetchone()
            high = int(hw_row[0]) if hw_row and hw_row[0] is not None else 0
            effective_w = 0 if w > high else w
            rows = conn.execute(_SELECT_SINCE_LIMIT, (effective_w, max_count)).fetchall()
        if w > high:
            logger.warning(
                "change feed: asker watermark %d exceeds journal high-water %d "
                "(journal reset or corrupt watermark); serving the full current "
                "set in bounded pages as a convergence backstop (CHF-01)",
                w,
                high,
            )
        latest: dict[tuple[str, str], SyncRecord] = {}
        skipped = 0
        acc = 0
        page_max = high
        took_any = False
        for row in rows:
            seq = int(row[0])
            rec = self._row_to_record(row[1:])
            if rec is None:
                # A corrupt row ships nothing but still occupies a sequence; step
                # the page past it so the cursor never re-reads it forever.
                skipped += 1
                page_max = seq
                took_any = True
                continue
            size = self._wire_size(rec)
            if took_any and acc + size > max_bytes:
                break
            latest[key_of(rec)] = rec
            acc += size
            page_max = seq
            took_any = True
        if skipped:
            logger.warning(
                "change feed: skipped %d corrupt journal row(s) on a paged read "
                "(payload unparseable or content-hash mismatch)",
                skipped,
            )
        return Delta(
            records=list(latest.values()),
            high_water=page_max if took_any else high,
        )

    def current_records(self) -> list[SyncRecord]:
        """The current latest-per-key snapshot across the whole journal."""
        with self._lock:
            conn = self._conn()
            rows = conn.execute(_SELECT_ALL).fetchall()
        return self._collapse_latest(rows)

    def _maybe_autocompact(self, conn: sqlite3.Connection, appended: int) -> None:
        """Tick the every-N-appends trigger; fire a bounded compaction at N.

        Called with ``self._lock`` held, after a successful commit. Disabled
        (the default) pays a single None check. A trigger-fired compaction
        failure is swallowed and logged: the append already committed and its
        sequence is returned regardless. The counter resets on fire whatever
        the outcome, so a persistently failing compaction retries every N
        appends, never on every append.
        """
        if self._compact_every is None:
            return
        self._appends_since_compact += appended
        if self._appends_since_compact < self._compact_every:
            return
        try:
            deleted = self._compact_locked(conn)
            logger.debug(
                "change feed: auto-compaction removed %d superseded row(s)",
                deleted,
            )
        except Exception:
            logger.warning(
                "change feed: auto-compaction failed; the append is unaffected",
                exc_info=True,
            )
        finally:
            self._appends_since_compact = 0

    def _compact_locked(self, conn: sqlite3.Connection) -> int:
        """Delete every row superseded by a later sequence for its key.

        Assumes ``self._lock`` is held. One bounded statement: the subquery
        keeps the MAX(seq) row per (kind, record_id), served by the existing
        key index. Touches feed rows only -- no other table is read or written,
        so a future meta table (CHF-05) is unaffected by construction.
        """
        cur = conn.execute(_DELETE_SUPERSEDED)
        conn.commit()
        n = cur.rowcount
        return int(n) if isinstance(n, int) and n > 0 else 0

    def compact(self, *, vacuum: bool = False) -> int:
        """Delete superseded journal rows; returns how many were removed (CHF-02).

        The transparent supersession rule proven in the F9b register: a row is
        superseded when a later sequence exists for the same (kind, record_id),
        of a clock as high or a tombstone (an out-of-order append never sends a
        key's clock back).
        ``since()`` collapses to the latest per key and the global MAX(seq) row
        is by construction the latest of its key, so no watermark's delta, no
        ``high_water``, no ``current_records`` and no per-key ``current_clock``
        changes -- every journal writer (a local mint at current+1 under its
        domain hook lock; ``apply_record_batch`` journalling winners only,
        including PRT-02 clock-only adoptions) appends a key's rows at
        non-decreasing clocks, so the MAX(seq) row carries the key's
        MAX(clock). The latest tombstone of a key is a row like any other and
        survives. Idempotent: a second run deletes 0.

        Blind to row health by design: rows were hash-verified at write time
        (``record``); post-write on-disk corruption is CHF-03 read territory.
        The journal is not the source of truth -- the domain stores re-journal
        at-least-once on their next write.

        On-demand maintenance: errors propagate (the caller asked and must
        know), unlike the every-N trigger which swallows and logs. Runs under
        the feed lock, serialised against record/record_many/since/
        current_clock/high_water/count/current_records/clear by construction.
        ``vacuum=True`` additionally rewrites the database file to reclaim
        space -- an unbounded extra cost, off by default. Local-disk
        maintenance, permitted in any mode, never gated.
        """
        with self._lock:
            conn = self._conn()
            deleted = self._compact_locked(conn)
            if vacuum:
                # Outside any transaction (the compaction just committed).
                conn.execute("VACUUM")
        return deleted

    def forget_record(self, kind: RecordKind | str, record_id: str, *, below: int | None = None) -> int:
        """Erase every version of one record, overwritten, and empty the log; how many rows went.

        A deletion asks it before its commit, so the record's words leave this device even when its tombstone
        cannot be appended after (the peers are then not told). The highest clock erased stays as the key's floor,
        in the same transaction: ``current_clock`` honours it, so the tombstone minted after out-clocks every version
        a peer may hold, and an incoming version at or below it loses (``protocol._apply_batch``). A peer's deletion
        asks with its tombstone's clock as ``below``: the floor stays under it, so the tombstone the round journals
        next passes it, even one that won at the clock of the version it erased. A journal whose file is not there
        holds none, and is not created to be asked; one that cannot be written raises, and the deletion refuses.
        """
        if self._connection is None and not self._resolve_db_path().exists():
            return 0
        k = kind.value if isinstance(kind, RecordKind) else str(kind)
        with self._lock:
            conn = self._conn()
            try:
                top = conn.execute(_SELECT_MAX_CLOCK, (k, record_id)).fetchone()
                gone = int(conn.execute(_DELETE_RECORD, (k, record_id)).rowcount or 0)
                floor = int(top[0]) if top and top[0] is not None else 0
                if below is not None:
                    floor = min(floor, int(below) - 1)
                if floor > 0:
                    conn.execute(_RAISE_FLOOR, (k, record_id, floor))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            if gone:
                self._empty_log(conn)
        return gone

    def forget_note_updates(self, note_id: str) -> int:
        """Erase every update of one note (the records ``note_id:seq``), overwritten, and empty the log; how many rows
        went. A note's deletion asks it, its own and a peer's: an update is never a tombstone, so nothing else takes
        a dead note's increments of its body out of the journal, which serves them on. No floor is kept: an update
        of a dead note is refused where it lands, and never journalled again. A journal whose file is not there
        holds none, and is not created to be asked; one that cannot be written raises."""
        if self._connection is None and not self._resolve_db_path().exists():
            return 0
        with self._lock:
            conn = self._conn()
            try:
                gone = int(conn.execute(_DELETE_RANGE, (RecordKind.NOTE_UPDATE.value, note_id + ":",
                                                        note_id + ";")).rowcount or 0)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            if gone:
                self._empty_log(conn)
        return gone

    def forget_dead_notes(self, note_ids: Iterable[str]) -> int:
        """Of these notes, all dead here and none of them held alive by this journal (``note_states``):
        erase every update the journal holds, overwritten, and empty the log; how many rows went. A note's own
        version is never erased here: a tombstone tells the peers, and a version alive is the journal's verdict
        over a store that took a losing tombstone. The sweep of the dead notes asks it, for what a deletion made
        before these rules, or interrupted, left. A journal whose file is not there holds none, and is not created
        to be asked; one that cannot be written raises."""
        dead = set(note_ids)
        if not dead or (self._connection is None and not self._resolve_db_path().exists()):
            return 0
        update = RecordKind.NOTE_UPDATE.value
        with self._lock:
            conn = self._conn()
            try:
                updated = {rid.rpartition(":")[0] for (rid,) in conn.execute(_SELECT_IDS, (update,)).fetchall()}
                gone = 0
                for note_id in sorted(updated & dead):
                    gone += int(conn.execute(_DELETE_RANGE, (update, note_id + ":", note_id + ";")).rowcount or 0)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            if gone:
                self._empty_log(conn)
        return gone

    def note_states(self, note_ids: Iterable[str]) -> dict[str, str]:
        """What this journal holds of each of these notes: ``tombstone`` (its deletion, told to the peers), ``live``
        (a version that is not a tombstone), ``owed`` (nothing but the floor a deletion here left: its versions
        erased, its tombstone never journalled -- the peers were not told), ``none``; ``absent`` for every note when
        the journal's file is not there (sync never used here), which is not created to be asked."""
        wanted = list(dict.fromkeys(note_ids))
        if not wanted:
            return {}
        if self._connection is None and not self._resolve_db_path().exists():
            return {note_id: "absent" for note_id in wanted}
        note = RecordKind.NOTE.value
        states: dict[str, str] = {}
        with self._lock:
            conn = self._conn()
            for note_id in wanted:
                row = conn.execute(_SELECT_LATEST, (note, note_id)).fetchone()
                if row is not None:
                    states[note_id] = "tombstone" if row[0] else "live"
                    continue
                floor = conn.execute(_SELECT_FLOOR, (note, note_id)).fetchone()
                states[note_id] = "owed" if floor and floor[0] else "none"
        return states

    def record_states(self, kind: RecordKind | str, record_ids: Iterable[str]) -> dict[str, str]:
        """``note_states`` for the records of any kind: ``tombstone``, ``live``, ``owed`` (nothing but the floor a
        deletion here left), ``none``; ``absent`` for every record when the journal's file is not there, which is not
        created to be asked."""
        wanted = list(dict.fromkeys(record_ids))
        if not wanted:
            return {}
        if self._connection is None and not self._resolve_db_path().exists():
            return {record_id: "absent" for record_id in wanted}
        k = kind.value if isinstance(kind, RecordKind) else str(kind)
        states: dict[str, str] = {}
        with self._lock:
            conn = self._conn()
            for record_id in wanted:
                row = conn.execute(_SELECT_LATEST, (k, record_id)).fetchone()
                if row is not None:
                    states[record_id] = "tombstone" if row[0] else "live"
                    continue
                floor = conn.execute(_SELECT_FLOOR, (k, record_id)).fetchone()
                states[record_id] = "owed" if floor and floor[0] else "none"
        return states

    def latest(self, kind: RecordKind | str, record_id: str) -> SyncRecord | None:
        """The latest version of one record, or None: none journalled, or one that no longer reads. A journal whose
        file is not there holds none, and is not created to be asked."""
        if self._connection is None and not self._resolve_db_path().exists():
            return None
        k = kind.value if isinstance(kind, RecordKind) else str(kind)
        with self._lock:
            conn = self._conn()
            row = conn.execute(_SELECT_ONE_LATEST, (k, record_id)).fetchone()
        return self._row_to_record(row[1:]) if row is not None else None

    def owed(self, kind: RecordKind | str) -> list[str]:
        """The records of a kind whose deletion here erased their versions and whose tombstone the journal never
        took -- the peers were not told --, sorted. A journal whose file is not there owes none, and is not created
        to be asked."""
        if self._connection is None and not self._resolve_db_path().exists():
            return []
        k = kind.value if isinstance(kind, RecordKind) else str(kind)
        with self._lock:
            conn = self._conn()
            rows = conn.execute(_SELECT_OWED, (k, k)).fetchall()
        return sorted(str(row[0]) for row in rows)

    def tend(self) -> int:
        """Erase the versions older builds left superseded, overwritten, and empty the log; how many rows went. A
        journal whose file is not there holds none, and is not created to be asked."""
        if self._connection is None and not self._resolve_db_path().exists():
            return 0
        with self._lock:
            conn = self._conn()
            erased = self._compact_locked(conn)
            if erased:
                self._empty_log(conn)
        return erased

    def clear(self) -> None:
        """Delete every journal row (the file and connection are kept)."""
        with self._lock:
            conn = self._conn()
            conn.execute(_DELETE_ALL)
            conn.commit()

    def close(self) -> None:
        """Close the underlying connection, if open."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                finally:
                    self._connection = None


# Module-level singleton with a reset hook (one journal per process, testable).

_feed: ChangeFeed | None = None
_feed_lock = threading.Lock()


def get_change_feed(root: Path | str | None = None) -> ChangeFeed:
    """Return the process change feed, creating it once (with ``root`` if given)."""
    global _feed
    with _feed_lock:
        if _feed is None:
            _feed = ChangeFeed(root=root)
        return _feed


def set_change_feed(feed: ChangeFeed | None) -> None:
    """Install a specific feed as the process singleton (used by tests)."""
    global _feed
    with _feed_lock:
        _feed = feed


def forget_conversation(conversation_id: str, *, below: int | None = None) -> int:
    """A deletion's erasure of every version of the conversation in the process journal, before its commit
    (``ChangeFeed.forget_record``; a peer's, with its tombstone's clock as ``below``); how many rows. Raises when the
    journal cannot be written."""
    return get_change_feed().forget_record(RecordKind.CONVERSATION, conversation_id, below=below)


def forget_record(kind: RecordKind | str, record_id: str) -> int:
    """An erasure of every version of one record in the process journal (``ChangeFeed.forget_record``)."""
    return get_change_feed().forget_record(kind, record_id)


def forget_note(note_id: str, *, versions: bool = True) -> int:
    """A note's deletion's erasure in the process journal: every update of the note, and with ``versions`` every
    version of the note itself, its clock kept as the floor the tombstone minted after out-clocks (``forget_record``;
    a peer's tombstone, about to be journalled, asks without); how many rows. Raises when the journal cannot be
    written."""
    feed = get_change_feed()
    gone = feed.forget_note_updates(note_id)
    if versions:
        gone += feed.forget_record(RecordKind.NOTE, note_id)
    return gone


def record_states(kind: RecordKind | str, record_ids: Iterable[str]) -> dict[str, str]:
    """What the process journal holds of each of these records (``ChangeFeed.record_states``)."""
    return get_change_feed().record_states(kind, record_ids)


def owed(kind: RecordKind | str) -> list[str]:
    """The records of a kind whose tombstone the process journal owes its peers (``ChangeFeed.owed``)."""
    return get_change_feed().owed(kind)


def forget_dead_notes(note_ids: Iterable[str]) -> int:
    """The sweep of the dead notes in the process journal (``ChangeFeed.forget_dead_notes``)."""
    return get_change_feed().forget_dead_notes(note_ids)


def note_states(note_ids: Iterable[str]) -> dict[str, str]:
    """What the process journal holds of each of these notes (``ChangeFeed.note_states``)."""
    return get_change_feed().note_states(note_ids)


def _tombstones_journalled(records: Iterable[SyncRecord]) -> None:
    """After an append: for each note tombstone just journalled -- this device's own, or a peer's that won here --
    the notes store of this process, if this process loaded it, forgets what it still holds of the note
    (``notes_store.forget_tombstoned``). Asked once the journal's lock is let go; it never raises into the append,
    which stands: what could not be forgotten is said, and the store's next sweep finishes it."""
    dead = [r.record_id for r in records if r.deleted and r.kind == RecordKind.NOTE]
    if not dead:
        return
    held = sys.modules.get("opti_oignon.notes.notes_store")
    forget = getattr(held, "forget_tombstoned", None) if held is not None else None
    if not callable(forget):
        return
    for note_id in dead:
        try:
            left = forget(note_id)
        except Exception as exc:  # noqa: BLE001 - the append stands; what stays is said
            logger.warning("change feed: the tombstone of the note %s journalled; the notes store could not forget "
                           "it (%s: %s)", note_id, type(exc).__name__, exc)
            continue
        if left:
            logger.warning("change feed: the tombstone of the note %s journalled; not forgotten yet: %s", note_id,
                           "; ".join(left))


def tend() -> int:
    """At boot: the process journal's superseded versions erased (``ChangeFeed.tend``); how many. Never raises: a
    journal that cannot be tended is said, and the next boot tries again."""
    try:
        return get_change_feed().tend()
    except Exception as exc:  # noqa: BLE001 - the boot never breaks on the journal
        logger.warning("change feed: superseded versions could not be erased (%s: %s); the next boot tries again",
                       type(exc).__name__, exc)
        return 0


def reset_change_feed() -> None:
    """Close and clear the process singleton so the next get creates a fresh one."""
    global _feed
    with _feed_lock:
        if _feed is not None:
            _feed.close()
        _feed = None
