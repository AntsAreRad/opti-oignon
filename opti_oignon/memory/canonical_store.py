"""Canonical memory store for Opti-Oignon (S173, Theme 3 / Odysseus Core).

The source of truth for personal memory facts. SQLite in WAL mode, parameterized
queries only, frozenset allowlists for any dynamic column or table name (no
f-string SQL, per the project SQL-hygiene standard). Per-user isolation via
``user_isolation.py``; encrypted at rest via ``db_encryption.py`` (SQLCipher)
when available, plain SQLite otherwise (Daily mode).

The legacy ``MemoryManager`` / ``MemoryFact`` (formerly ``opti_oignon/memory.py``)
were folded into this package in S173 and now live in
``opti_oignon/memory/legacy.py``; the package ``__init__`` re-exports them for
backward compatibility. This module is intentionally importable in isolation
(``spec_from_file_location`` + ``sys.modules`` stubs) without ollama or fastapi:
the ``db_encryption`` / ``user_isolation`` imports are guarded with a plain
SQLite fallback so the runtime tests collect without the backend.

S200 (sync cycle Bloc 0 lot 2 / SYN-01): the write methods publish to the
Veilid change feed after their commit via :func:`_sync_publish_memory_fact` --
full state for add/update and for the soft-delete/restore flag, a tombstone
for ``hard_delete``, per-fact tombstones for ``clear``. Touches are local
telemetry and publish nothing.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# Module sentinels (project convention).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True


# Guarded backend integration. In the full backend these resolve to the real
# modules. Loaded in isolation, the relative imports fall back to a plain
# SQLite path so the runtime tests collect without fastapi / ollama / sqlcipher.
try:
    from ..db_encryption import SQLCIPHER_AVAILABLE, get_encrypted_connection

    _HAS_DB_ENCRYPTION = True
except Exception:  # ImportError, or relative-import-beyond-top-level in isolation
    _HAS_DB_ENCRYPTION = False
    SQLCIPHER_AVAILABLE = False

    def get_encrypted_connection(  # type: ignore[misc]
        db_path: Any,
        *,
        check_same_thread: bool = True,
        timeout: float = 5.0,
        enforce_encryption: bool | None = None,
    ) -> sqlite3.Connection:
        return sqlite3.connect(
            str(db_path), check_same_thread=check_same_thread, timeout=timeout
        )


try:
    from ..user_isolation import DEFAULT_LOCAL_USER, effective_user_id

    _HAS_USER_ISOLATION = True
except Exception:
    _HAS_USER_ISOLATION = False
    DEFAULT_LOCAL_USER = "local"

    def effective_user_id(  # type: ignore[misc]
        user_id: str | None, single_user_mode: bool = True
    ) -> str:
        if single_user_mode or user_id is None:
            return DEFAULT_LOCAL_USER
        return user_id


# The six canonical categories. A frozenset so it doubles as the allowlist for
# any category-keyed query path.
CATEGORIES: frozenset[str] = frozenset(
    {"identity", "preference", "fact", "contact", "project", "goal"}
)
DEFAULT_CATEGORY = "fact"

# The physical table. Static, but referenced through this single constant so
# the schema and every query stay in lockstep.
TABLE_NAME = "memory_facts"

# Allowlists for the only places a column identifier is ever assembled into a
# statement (the dynamic UPDATE SET clause and ORDER BY). Used with str.format()
# under these frozensets so no caller-controlled string is interpolated into
# SQL. This is the sanctioned alternative to f-string SQL.
_UPDATABLE_COLUMNS: frozenset[str] = frozenset(
    {"text", "category", "source", "active", "use_count", "updated_at"}
)
_ORDERABLE_COLUMNS: frozenset[str] = frozenset(
    {"created_at", "updated_at", "use_count", "category"}
)


@dataclass
class MemoryRecord:
    """A single canonical memory fact.

    Attributes:
        id: Unique identifier (hex uuid4).
        text: The fact, one short sentence.
        category: One of CATEGORIES.
        source: Free-form provenance (e.g. a conversation id or "manual").
        user_id: Owning user; "local" in single-user mode.
        created_at: ISO-8601 creation timestamp.
        updated_at: ISO-8601 last-update timestamp.
        active: False once soft-deleted.
        use_count: Number of times the fact was surfaced/used.
    """

    id: str
    text: str
    category: str
    source: str = ""
    user_id: str = DEFAULT_LOCAL_USER
    created_at: str = ""
    updated_at: str = ""
    active: bool = True
    use_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    """ISO-8601 UTC timestamp with microseconds; lexically sortable."""
    return datetime.now(timezone.utc).isoformat()


def _default_db_path() -> Path:
    """Default on-disk location, distinct from the legacy memories.db."""
    try:
        from ..config import DATA_DIR

        base = Path(DATA_DIR)
    except Exception:
        base = Path("data")
    return base / "memory_facts.db"


def _coerce(column: str, value: Any) -> Any:
    """Coerce a Python value to its stored representation for a known column."""
    if column == "active":
        return 1 if value else 0
    if column == "use_count":
        return int(value)
    return value


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        text=row["text"],
        category=row["category"],
        source=row["source"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        active=bool(row["active"]),
        use_count=int(row["use_count"]),
    )


# Veilid sync producer (SYN-01, S200 / sync cycle Bloc 0 lot 2). Mirrors the
# lot-1 conversation template: domain commit first, payload built inside the
# hook's guard, a failure never breaks the write, mode-free (only the wire is
# Daily-gated downstream). Touches are deliberately NOT published: the
# retrieval hot path bumps use counters on every surfaced fact, and a counter
# does not merge under LWW -- device-local telemetry, excluded from the
# payload as well.


def _fact_payload(record: MemoryRecord) -> dict[str, Any]:
    """Full-state payload for a canonical fact (state-based LWW).

    ``user_id`` is hoisted to the top level (the lot-1 scoping rule: the bare
    fact id is the per-kind key, ownership rides in the payload); the nested
    fact is ``to_dict`` minus ``use_count`` (device-local) and minus the
    hoisted ``user_id``.
    """
    fact = record.to_dict()
    fact.pop("use_count", None)
    uid = fact.pop("user_id", DEFAULT_LOCAL_USER)
    return {"user_id": uid, "fact": fact}


def _sync_wanted() -> bool:
    """Cheap availability probe for ``clear``'s pre-delete id read.

    ``clear`` must read the doomed ids BEFORE the DELETE to tombstone them,
    so the payload-inside-the-guard deferral cannot apply there; this probe
    keeps the absent-sync cost at zero extra reads instead.
    """
    try:
        from opti_oignon.veilid.guard import veilid_available

        return bool(veilid_available())
    except Exception:
        return False


def _sync_publish_memory_fact(
    fact_id: str,
    payload_fn: Callable[[], dict[str, Any] | None] | None = None,
    *,
    deleted: bool = False,
    updated_at: str = "",
) -> None:
    """Journal a canonical-fact change for Veilid sync, best-effort (SYN-01).

    Called by the store's write methods AFTER the domain commit, while the
    store lock is still held. ``payload_fn`` is a zero-arg callable building
    the full-state payload; it runs INSIDE this hook's protection, and only
    after the availability probe passes, so when sync is absent the write
    pays nothing (no extra reads, no journal append). The contract
    (ROADMAP_SYNC_CYCLE, Bloc 0, the lot-1 precedents):

    - A payload or journalling failure must never break the write: any error
      is logged and swallowed (at-least-once on the next write).
    - No-op when the optional veilid framework is absent
      (``guard.veilid_available`` is the cheap probe).
    - Mode-free: producing and journalling are local-disk operations
      permitted in ANY mode (the documented ``producers.py`` posture); only
      the wire is Daily-gated, downstream at the engine/guard.
    - A soft delete is STATE (the ``active`` flag in the payload; restore
      round-trips); only a hard delete is a tombstone.

    Clock discipline: next = the highest clock journalled for the key, plus
    one (an unseen key yields 0, so the first clock is 1). Running under the
    store lock serialises mint + append per process, keeping same-key clocks
    strictly monotonic. Lock order is store lock -> feed lock; the feed never
    calls back into domain code, so the order is acyclic.
    """
    try:
        from opti_oignon.veilid.guard import veilid_available

        if not veilid_available():
            return
        payload: dict[str, Any] | None = None
        if not deleted:
            payload = payload_fn() if payload_fn is not None else None
            if payload is None:
                # The state could not be built (row gone mid-write).
                # Publishing an empty non-tombstone payload would wipe the
                # fact on peers under LWW -- skip instead.
                logger.debug(
                    "sync publish skipped for fact %s: no state available",
                    fact_id,
                )
                return
        from opti_oignon.veilid.records import RecordKind
        from opti_oignon.veilid.sync_engine import get_sync_engine

        engine = get_sync_engine()
        clock = engine.current_clock(RecordKind.MEMORY_CANONICAL, fact_id) + 1
        engine.publish_memory_canonical(
            fact_id,
            payload,
            clock=clock,
            deleted=deleted,
            updated_at=updated_at,
        )
    except Exception:
        logger.warning(
            "veilid sync publish failed for memory fact %s (write unaffected)",
            fact_id,
            exc_info=True,
        )


class CanonicalMemoryStore:
    """SQLite-backed canonical store for memory facts (the source of truth)."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        single_user_mode: bool = True,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else _default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._single_user_mode = single_user_mode
        self._lock = threading.RLock()
        self._init_db()

    # Connection handling

    def _connect(self) -> sqlite3.Connection:
        conn = get_encrypted_connection(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL is set outside any transaction (right after connect).
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact',
                    source TEXT NOT NULL DEFAULT '',
                    user_id TEXT NOT NULL DEFAULT 'local',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    use_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_memory_facts_category
                    ON memory_facts(category);
                CREATE INDEX IF NOT EXISTS idx_memory_facts_active
                    ON memory_facts(active);
                CREATE INDEX IF NOT EXISTS idx_memory_facts_user
                    ON memory_facts(user_id);
                """
            )

    @property
    def db_path(self) -> Path:
        return self._db_path

    def journal_mode(self) -> str:
        """Return the active SQLite journal mode (expected: 'wal')."""
        with self._lock, self._conn() as conn:
            row = conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    def resolve_user(self, user_id: str | None = None) -> str:
        """Resolve the effective user id under this store's isolation mode.

        Exposed so a coordinating layer can pass one consistent id to both the
        canonical store and the vector layer.
        """
        return effective_user_id(user_id, self._single_user_mode)

    # Create

    def add(
        self,
        text: str,
        category: str = DEFAULT_CATEGORY,
        *,
        source: str = "",
        user_id: str | None = None,
        fact_id: str | None = None,
    ) -> MemoryRecord:
        """Insert a fact and return its record. Unknown categories are coerced."""
        category = category if category in CATEGORIES else DEFAULT_CATEGORY
        uid = effective_user_id(user_id, self._single_user_mode)
        rid = fact_id or uuid.uuid4().hex
        ts = _now()
        record = MemoryRecord(
            id=rid,
            text=text,
            category=category,
            source=source,
            user_id=uid,
            created_at=ts,
            updated_at=ts,
            active=True,
            use_count=0,
        )
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO memory_facts "
                    "(id, text, category, source, user_id, created_at, updated_at, active, use_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0)",
                    (rid, text, category, source, uid, ts, ts),
                )
            # S200 SYN-01: domain commit first (the inner ``with`` commits on
            # exit), then the sync publish under the store lock. The payload
            # closes over the just-built record -- zero extra reads.
            _sync_publish_memory_fact(
                rid, lambda: _fact_payload(record), updated_at=ts
            )
        return record

    # Read

    def get(self, fact_id: str, *, user_id: str | None = None) -> MemoryRecord | None:
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memory_facts WHERE id = ? AND user_id = ?",
                (fact_id, uid),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list(
        self,
        *,
        category: str | None = None,
        active_only: bool = True,
        user_id: str | None = None,
        order_by: str = "created_at",
        descending: bool = True,
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        if order_by not in _ORDERABLE_COLUMNS:
            order_by = "created_at"
        uid = effective_user_id(user_id, self._single_user_mode)

        # Every fragment below is a constant string or a "?" placeholder; the
        # only identifiers are user_id / active / category (literals) and the
        # allowlisted order_by. No caller value reaches the SQL text.
        clauses = ["user_id = ?"]
        params: list[Any] = [uid]
        if active_only:
            clauses.append("active = 1")
        if category is not None:
            if category not in CATEGORIES:
                return []
            clauses.append("category = ?")
            params.append(category)
        where = " AND ".join(clauses)
        direction = "DESC" if descending else "ASC"
        sql = f"SELECT * FROM memory_facts WHERE {where} ORDER BY {order_by} {direction}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_record(r) for r in rows]

    def count(self, *, active_only: bool = True, user_id: str | None = None) -> int:
        uid = effective_user_id(user_id, self._single_user_mode)
        sql = "SELECT COUNT(*) AS n FROM memory_facts WHERE user_id = ?"
        if active_only:
            sql += " AND active = 1"
        with self._lock, self._conn() as conn:
            row = conn.execute(sql, (uid,)).fetchone()
        return int(row["n"])

    # Update

    def update(
        self,
        fact_id: str,
        *,
        user_id: str | None = None,
        **fields: Any,
    ) -> MemoryRecord | None:
        """Update one or more allowlisted columns; refreshes updated_at."""
        if "category" in fields and fields["category"] not in CATEGORIES:
            raise ValueError("Invalid category: " + repr(fields["category"]))

        columns: list[str] = []
        values: list[Any] = []
        for key, val in fields.items():
            if key not in _UPDATABLE_COLUMNS:
                raise ValueError("Column not updatable: " + repr(key))
            columns.append(key)
            values.append(_coerce(key, val))

        if not columns:
            return self.get(fact_id, user_id=user_id)
        if "updated_at" not in columns:
            columns.append("updated_at")
            values.append(_now())

        # columns are drawn from _UPDATABLE_COLUMNS only; the assembled clause
        # therefore contains allowlisted identifiers plus "?" placeholders.
        set_clause = ", ".join(f"{col} = ?" for col in columns)
        uid = effective_user_id(user_id, self._single_user_mode)
        sql = f"UPDATE memory_facts SET {set_clause} WHERE id = ? AND user_id = ?"
        values.extend([fact_id, uid])
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(sql, values)
                changed = cur.rowcount
            if changed == 0:
                return None
            record = self.get(fact_id, user_id=user_id)
            # S200 SYN-01: publish the post-update state; the payload closes
            # over the record just fetched for the return value -- zero extra
            # reads. ``get`` re-enters the RLock from the same thread.
            if record is not None:
                _sync_publish_memory_fact(
                    record.id,
                    lambda: _fact_payload(record),
                    updated_at=record.updated_at,
                )
            return record

    def touch(self, fact_id: str, *, user_id: str | None = None) -> bool:
        """Increment the use counter and refresh updated_at."""
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE memory_facts SET use_count = use_count + 1, updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (_now(), fact_id, uid),
            )
            return cur.rowcount > 0

    def _fact_state(self, fact_id: str, uid: str) -> dict[str, Any] | None:
        """Re-read a fact and shape its sync payload; None if the row is gone.

        Runs inside the publish hook's guard (only when sync is available),
        re-entering the store RLock from the same thread.
        """
        record = self.get(fact_id, user_id=uid)
        return _fact_payload(record) if record is not None else None

    # Delete: soft (clear active) and hard (row removal)

    def soft_delete(self, fact_id: str, *, user_id: str | None = None) -> bool:
        uid = effective_user_id(user_id, self._single_user_mode)
        ts = _now()
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "UPDATE memory_facts SET active = 0, updated_at = ? "
                    "WHERE id = ? AND user_id = ? AND active = 1",
                    (ts, fact_id, uid),
                )
                changed = cur.rowcount > 0
            if changed:
                # S200 SYN-01: a soft delete is STATE, not a tombstone -- the
                # active flag rides the payload so restore round-trips. The
                # re-read runs inside the hook's guard, paid only when sync
                # is available.
                _sync_publish_memory_fact(
                    fact_id,
                    lambda: self._fact_state(fact_id, uid),
                    updated_at=ts,
                )
            return changed

    def restore(self, fact_id: str, *, user_id: str | None = None) -> bool:
        uid = effective_user_id(user_id, self._single_user_mode)
        ts = _now()
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "UPDATE memory_facts SET active = 1, updated_at = ? "
                    "WHERE id = ? AND user_id = ? AND active = 0",
                    (ts, fact_id, uid),
                )
                changed = cur.rowcount > 0
            if changed:
                # S200 SYN-01: the restore leg of the soft-delete round-trip,
                # published as state (active back to True).
                _sync_publish_memory_fact(
                    fact_id,
                    lambda: self._fact_state(fact_id, uid),
                    updated_at=ts,
                )
            return changed

    def hard_delete(self, fact_id: str, *, user_id: str | None = None) -> bool:
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "DELETE FROM memory_facts WHERE id = ? AND user_id = ?",
                    (fact_id, uid),
                )
                deleted = cur.rowcount > 0
            if deleted:
                # S200 SYN-01: only a hard delete is the converged deletion --
                # a tombstone (empty payload, deleted=True).
                _sync_publish_memory_fact(
                    fact_id, deleted=True, updated_at=_now()
                )
            return deleted

    def clear(self, *, user_id: str | None = None) -> int:
        """Remove all of the user's rows (tests, resets, the UD-03 user wipe).

        S200 SYN-01: the wipe propagates as per-fact tombstones so peers
        converge on the deletion. The doomed ids must be read BEFORE the
        DELETE, so the availability probe runs first (``_sync_wanted``) and
        the absent-sync cost stays at zero extra reads. Bounded by the user's
        fact count; tombstones are published after the commit, under the
        store lock.
        """
        uid = effective_user_id(user_id, self._single_user_mode)
        ts = _now()
        with self._lock:
            sync_ids: list[str] = []
            with self._conn() as conn:
                if _sync_wanted():
                    rows = conn.execute(
                        "SELECT id FROM memory_facts WHERE user_id = ?", (uid,)
                    ).fetchall()
                    sync_ids = [str(r["id"]) for r in rows]
                cur = conn.execute(
                    "DELETE FROM memory_facts WHERE user_id = ?", (uid,)
                )
                count = int(cur.rowcount)
            for fid in sync_ids:
                _sync_publish_memory_fact(fid, deleted=True, updated_at=ts)
            return count

    def apply_synced_memory_canonical(
        self,
        record_id: str,
        payload: dict[str, Any],
        *,
        deleted: bool = False,
        updated_at: str = "",
    ) -> bool:
        """Materialise a synced canonical memory fact (SYN-01 apply, receive half).

        The RECEIVING half of a sync round for ``MEMORY_CANONICAL``: a winning
        record, already reconciled and signature-verified upstream, is written
        into the store so the fact surfaces on this device. Deliberately
        HOOK-FREE -- it never calls ``_sync_publish_memory_fact`` -- so applying
        a received record cannot re-publish it and start an
        apply -> write -> publish echo (which would inflate the clock and
        ping-pong between devices forever; the same posture as the
        conversation/note landings).

        Full-state and idempotent: a non-tombstone UPSERTs the wire-carried
        columns by id, so applying the same winner twice converges to the same
        row. ``use_count`` is device-local (the producer strips it from the
        payload) and is PRESERVED across an apply -- an existing row keeps its
        count, a new row starts at 0 -- so a remote LWW win on a fact's content
        never zeroes this device's usage telemetry. (The ``ON CONFLICT DO
        UPDATE`` touches only the wire columns; an ``INSERT OR REPLACE`` would
        wipe ``use_count``.) The ``active`` flag is STATE (soft-delete/restore
        round-trips), written verbatim. Fact text is stored as-is into the
        SQLCipher store (no per-field layer, unlike conversation messages). A
        ``deleted`` record HARD-deletes the fact by id (the converged deletion;
        a soft delete is the ``active`` flag, never a tombstone).

        Fail-secure: a malformed payload (not a dict, no nested ``fact``, no id,
        a non-string text, or a nested id that does not match the record key)
        or any write error returns False and never raises into the round -- the
        caller drops a False-returning record rather than journalling unapplied
        state.
        """
        try:
            if deleted:
                with self._lock, self._conn() as conn:
                    conn.execute(
                        "DELETE FROM memory_facts WHERE id = ?", (record_id,)
                    )
                return True
            if not isinstance(payload, dict):
                return False
            fact = payload.get("fact")
            if not isinstance(fact, dict):
                return False
            fid = fact.get("id")
            if not isinstance(fid, str) or not fid or fid != record_id:
                return False
            text = fact.get("text")
            if not isinstance(text, str):
                return False
            uid = payload.get("user_id") or DEFAULT_LOCAL_USER
            category = fact.get("category") or DEFAULT_CATEGORY
            if category not in CATEGORIES:
                category = DEFAULT_CATEGORY
            source = fact.get("source") or ""
            created_at = fact.get("created_at") or ""
            upd = fact.get("updated_at") or updated_at or ""
            active = 1 if fact.get("active", True) else 0
            with self._lock, self._conn() as conn:
                conn.execute(
                    "INSERT INTO memory_facts "
                    "(id, text, category, source, user_id, created_at, "
                    "updated_at, active, use_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "text=excluded.text, category=excluded.category, "
                    "source=excluded.source, user_id=excluded.user_id, "
                    "created_at=excluded.created_at, "
                    "updated_at=excluded.updated_at, active=excluded.active",
                    (fid, text, category, source, uid, created_at, upd, active),
                )
            return True
        except Exception:
            logger.debug(
                "memory-canonical apply failed for %s", record_id, exc_info=True
            )
            return False


# Module-level singleton with a reset for test isolation (S171 lesson: never
# leak shared state across pytest invocations).
_store: CanonicalMemoryStore | None = None


def get_canonical_store() -> CanonicalMemoryStore:
    global _store
    if _store is None:
        _store = CanonicalMemoryStore()
    return _store


def reset_canonical_store() -> None:
    global _store
    _store = None
