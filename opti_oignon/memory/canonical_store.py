"""Canonical memory store for Opti-Oignon.

The source of truth for personal memory facts. SQLite in WAL mode, parameterized
queries only: every caller value travels as a "?" placeholder and never reaches
the SQL text. The two statements that need a dynamic column name assemble it
from a frozenset allowlist, so the interpolated fragments are drawn from a
closed set rather than from input. Per-user isolation via
``user_isolation.py``; encrypted at rest via ``db_encryption.py`` (SQLCipher)
when available, plain SQLite otherwise (Daily mode).

The legacy ``MemoryManager`` / ``MemoryFact`` (formerly ``opti_oignon/memory.py``)
were folded into this package and now live in
``opti_oignon/memory/legacy.py``; the package ``__init__`` re-exports them for
backward compatibility. This module is intentionally importable in isolation
(``spec_from_file_location`` + ``sys.modules`` stubs) without ollama or fastapi:
the ``db_encryption`` / ``user_isolation`` imports are guarded with a plain
SQLite fallback so the runtime tests collect without the backend.

The write methods publish to the
Veilid change feed after their commit via :func:`_sync_publish_memory_fact` --
full state for add/update and for the soft-delete/restore flag, a tombstone
for ``hard_delete``, per-fact tombstones for ``clear``. Touches are local
telemetry and publish nothing.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import sys
import threading
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

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
# statement (the dynamic UPDATE SET clause and ORDER BY). Membership is checked
# against these frozensets before the name reaches the query, so the only
# strings interpolated into SQL come from this closed set; every caller value
# travels as a "?" placeholder. What makes those two statements safe is the
# allowlist, not the formatting mechanism -- see the comment at each site.
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
        endorsed: The digest of the text the user endorsed -- by typing it,
            writing it by hand, accepting a proposal of it, or adopting it
            -- or None. The fact is endorsed while its text is still those
            bytes (see ``is_endorsed``). Device-local: never published.
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
    endorsed: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fact_digest(text: str) -> str:
    """The digest an endorsement binds a fact's text by."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def is_endorsed(record: Any) -> bool:
    """Whether the user endorsed this fact's text as it reads now: a text changed since is no longer theirs."""
    endorsed = getattr(record, "endorsed", None)
    return bool(endorsed) and endorsed == fact_digest(getattr(record, "text", ""))


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
        endorsed=row["endorsed"] if "endorsed" in row.keys() else None,
    )


# Veilid sync producer. Mirrors the
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
    # An endorsement is this device's user's act: no peer receives it.
    fact.pop("endorsed", None)
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
    (ROADMAP_SYNC_CYCLE, the lot-1 precedents):

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


def open_vector_layer() -> Any:
    """This process's vector layer when it is open, else None. It is never opened only to drop a vector: a store
    reached by a sync round, a deletion in another process, or a test, is not made to open the knowledge base's file
    and its key. What the layer holds of a fact is then owed (``owed_vectors``), and the trash's pass, which opens it,
    settles it."""
    module = sys.modules.get("opti_oignon.memory.vector_store")
    return getattr(module, "_vector_store", None) if module is not None else None


def _drop_vector(fact_id: str) -> bool:
    """Drop this device's vector of a fact a peer put out of service or erased: the vector layer keeps its text.
    False when the layer is not open in this process, or fails: the drop is then owed."""
    layer = open_vector_layer()
    if layer is None:
        return False
    try:
        layer.delete(fact_id)
    except Exception:  # noqa: BLE001 - said, and owed to the trash's next pass
        logger.warning("fact %s: this device's vector of it could not be dropped; owed to the trash's next pass",
                       fact_id, exc_info=True)
        return False
    return True


def _write_lock() -> Any:
    """The facts' write lock (``dedup.write_lock``); a process without the memory store has none to take."""
    try:
        from . import dedup
    except ImportError:
        return nullcontext()
    return dedup.write_lock()


def _forgotten_here(source: str) -> bool:
    """Whether ``source`` names a conversation this process forgot (the facts' marks, ``dedup.forgotten``)."""
    try:
        from . import dedup
    except ImportError:
        return False
    drawn = dedup.drawn_from(source)
    return drawn is not None and dedup.forgotten(drawn)


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
        # What any write frees -- an erased fact, the text an edit replaced -- is overwritten, never left in the
        # file's free pages: the cipher's driver does not do it by default.
        conn.execute("PRAGMA secure_delete = ON")
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
                    use_count INTEGER NOT NULL DEFAULT 0,
                    endorsed TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memory_facts_category
                    ON memory_facts(category);
                CREATE INDEX IF NOT EXISTS idx_memory_facts_active
                    ON memory_facts(active);
                CREATE INDEX IF NOT EXISTS idx_memory_facts_user
                    ON memory_facts(user_id);
                CREATE TABLE IF NOT EXISTS vectors_owed (
                    fact_id TEXT PRIMARY KEY,
                    act TEXT NOT NULL,
                    since TEXT NOT NULL
                );
                """
            )
            # Endorsements arrived after the table: an older file gains the
            # column, and every fact it already holds waits for the user to
            # adopt it by the digest of the text shown.
            present = {row[1] for row in conn.execute("PRAGMA table_info(memory_facts)")}
            if "endorsed" not in present:
                conn.execute("ALTER TABLE memory_facts ADD COLUMN endorsed TEXT")

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
        """Insert a fact and return its record. Unknown categories are coerced.

        The fact is written unendorsed: a writer that is the user's own act --
        their typing, their hand, a proposal they accepted -- adopts it next,
        by the digest of the very text it wrote (see ``adopt``).
        """
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
            # Domain commit first (the inner ``with`` commits on
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

    def held_under(self, sources: Iterable[str]) -> list[tuple[str, str]]:
        """The id and the owner of every fact written under one of ``sources``, in service or in the trash, every
        user's: what a conversation's deletion erases (``MemoryStore.forget_conversation``)."""
        wanted = sorted({str(source) for source in sources})
        if not wanted:
            return []
        # Only "?" placeholders are assembled; every source travels as a parameter.
        marks = ", ".join("?" for _ in wanted)
        with self._lock, self._conn() as conn:
            rows = conn.execute(f"SELECT id, user_id FROM memory_facts WHERE source IN ({marks})", wanted).fetchall()
        return [(str(row["id"]), str(row["user_id"])) for row in rows]

    def trashed_before(self, cutoff: str) -> list[tuple[str, str]]:
        """The id and the owner of every fact in the trash whose last change is ``cutoff`` (ISO-8601 UTC, as
        ``updated_at``) or older, every user's: what the trash erases (``memory.trash``)."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, user_id FROM memory_facts WHERE active = 0 AND updated_at <= ?", (cutoff,)
            ).fetchall()
        return [(str(row["id"]), str(row["user_id"])) for row in rows]

    def empty_log(self, what: str) -> None:
        """Empty the write-ahead log after an erasure: while another connection keeps the log, its older frames
        still hold the erased facts. Never waits: a read in progress keeps the log, said, until the next erasure or
        the last connection's close removes it. ``what`` names the erasure in what is said."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("PRAGMA busy_timeout = 0")
                busy = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
            except Exception as exc:  # noqa: BLE001 - the cipher driver's errors are not sqlite3's, and this only says
                logger.warning("%s; the write-ahead log of the facts could not be emptied (%s: %s)",
                               what, type(exc).__name__, exc)
                return
            finally:
                conn.close()
        if busy:
            logger.warning("%s; a read in progress keeps the write-ahead log of the facts, whose older frames hold "
                           "them until the next erasure or the last connection closes", what)

    # Update

    def update(
        self,
        fact_id: str,
        *,
        user_id: str | None = None,
        **fields: Any,
    ) -> MemoryRecord | None:
        """Update one or more allowlisted columns; refreshes updated_at.

        An endorsement names the text in service: a change of the text, or the
        fact leaving service, clears it, and the fact is no longer the user's
        until a writer that is their act adopts it again (see ``adopt``).
        """
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
        # The endorsement is kept only while the text stays the same and the
        # fact stays in service (each expression reads the row as it was).
        keep: list[str] = []
        if "text" in fields:
            keep.append("text = ?")
            values.append(_coerce("text", fields["text"]))
        if "active" in fields:
            # In service before this update and after it.
            keep.append("active = 1 AND ? = 1")
            values.append(1 if fields["active"] else 0)
        if keep:
            set_clause += f", endorsed = CASE WHEN {' AND '.join(keep)} THEN endorsed ELSE NULL END"
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
            # Publish the post-update state; the payload closes
            # over the record just fetched for the return value -- zero extra
            # reads. ``get`` re-enters the RLock from the same thread.
            if record is not None:
                _sync_publish_memory_fact(
                    record.id,
                    lambda: _fact_payload(record),
                    updated_at=record.updated_at,
                )
            return record

    def adopt(self, fact_id: str, digest: str, *, user_id: str | None = None) -> bool:
        """The user's endorsement of a fact's text, named by its digest; whether it took.

        The one way an endorsement is written: the user adopting the text they
        were shown, or a writer that is their act -- their typing, their hand,
        a proposal they accepted -- adopting the very text it just wrote. It
        takes only while the fact is active and its text is still the bytes
        that digest names: a text changed since is refused, never endorsed in
        its place. It writes the endorsement alone -- no text, no timestamp --
        and publishes nothing: an endorsement is this device's.
        """
        if not isinstance(digest, str) or not digest:
            return False
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock:
            record = self.get(fact_id, user_id=uid)
            if record is None or not record.active or fact_digest(record.text) != digest:
                return False
            with self._conn() as conn:
                cur = conn.execute(
                    "UPDATE memory_facts SET endorsed = ? WHERE id = ? AND user_id = ? AND text = ?",
                    (digest, fact_id, uid, record.text),
                )
                return cur.rowcount == 1

    def unendorsed(self, *, user_id: str | None = None) -> list[MemoryRecord]:
        """The active facts the user has not endorsed as they read now, oldest first."""
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_facts WHERE user_id = ? AND active = 1 ORDER BY created_at, id", (uid,)
            ).fetchall()
        return [record for record in map(_row_to_record, rows) if not is_endorsed(record)]

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
                # A fact out of service loses its endorsement: brought back, it
                # waits for an adoption again.
                cur = conn.execute(
                    "UPDATE memory_facts SET active = 0, updated_at = ?, endorsed = NULL "
                    "WHERE id = ? AND user_id = ? AND active = 1",
                    (ts, fact_id, uid),
                )
                changed = cur.rowcount > 0
            if changed:
                # A soft delete is STATE, not a tombstone -- the
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
                # Back in service, a fact waits for an adoption: no
                # endorsement comes back with it.
                cur = conn.execute(
                    "UPDATE memory_facts SET active = 1, updated_at = ?, endorsed = NULL "
                    "WHERE id = ? AND user_id = ? AND active = 0",
                    (ts, fact_id, uid),
                )
                changed = cur.rowcount > 0
            if changed:
                # The restore leg of the soft-delete round-trip,
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
                # Only a hard delete is the converged deletion --
                # a tombstone (empty payload, deleted=True).
                _sync_publish_memory_fact(
                    fact_id, deleted=True, updated_at=_now()
                )
            return deleted

    def erase(self, fact_id: str, *, trashed_only: bool = False) -> bool:
        """Delete a fact by its id whoever owns it, and tell the peers (a tombstone); whether it went.

        The erasures the store makes on its own account -- a deleted conversation's facts, the trash past its days
        -- act for no user: a fact a peer brought keeps the owner it came with, which this device's mode may not
        resolve to. ``trashed_only`` erases the fact only while it is in the trash: one restored meanwhile stays.
        """
        # Two constant statements; the id travels as a parameter.
        sql = "DELETE FROM memory_facts WHERE id = ? AND active = 0" if trashed_only else (
            "DELETE FROM memory_facts WHERE id = ?")
        with self._lock:
            with self._conn() as conn:
                deleted = conn.execute(sql, (fact_id,)).rowcount > 0
            if deleted:
                _sync_publish_memory_fact(fact_id, deleted=True, updated_at=_now())
            return deleted

    def clear(self, *, user_id: str | None = None) -> int:
        """Remove all of the user's rows (tests, resets, the UD-03 user wipe).

        The wipe propagates as per-fact tombstones so peers
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
        a soft delete is the ``active`` flag, never a tombstone). This
        device's vector of a fact out of service or erased goes with it, or is
        owed (``owed_vectors``); a fact drawn from a conversation this process
        forgot is refused. All of it under the facts' write lock.

        Fail-secure: a malformed payload (not a dict, no nested ``fact``, no id,
        a non-string text, or a nested id that does not match the record key)
        or any write error returns False and never raises into the round -- the
        caller drops a False-returning record rather than journalling unapplied
        state.
        """
        try:
            # Under the facts' write lock, as a write of this device's own: a
            # forgetting that comes erases what lands, or refuses it after.
            with _write_lock():
                return self._apply_synced(record_id, payload, deleted=deleted, updated_at=updated_at)
        except Exception:
            logger.debug(
                "memory-canonical apply failed for %s", record_id, exc_info=True
            )
            return False

    def _apply_synced(self, record_id: str, payload: Any, *, deleted: bool, updated_at: str) -> bool:
        """The apply, under the facts' write lock (``apply_synced_memory_canonical``).

        The fact's row changes first. This device's vector of a fact out of
        service or erased is dropped after it, and one back in service is
        given back later (``owed_vectors``, settled by the trash's pass): a
        record refused here would never come again, the round having moved
        past it, so the vector layer never refuses one.
        """
        if deleted:
            with self._lock, self._conn() as conn:
                conn.execute(
                    "DELETE FROM memory_facts WHERE id = ?", (record_id,)
                )
            if not _drop_vector(record_id):
                self.owe_vector(record_id, "drop")
            self.empty_log(f"a peer's erasure of fact {record_id} applied")
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
        if _forgotten_here(source):
            # Drawn from a conversation this process forgot: refused, as
            # a write of this device's own would be.
            return False
        created_at = fact.get("created_at") or ""
        upd = fact.get("updated_at") or updated_at or ""
        active = 1 if fact.get("active", True) else 0
        with self._lock, self._conn() as conn:
            prior = conn.execute("SELECT active FROM memory_facts WHERE id = ?", (fid,)).fetchone()
            conn.execute(
                "INSERT INTO memory_facts "
                "(id, text, category, source, user_id, created_at, "
                "updated_at, active, use_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(id) DO UPDATE SET "
                "text=excluded.text, category=excluded.category, "
                "source=excluded.source, user_id=excluded.user_id, "
                "created_at=excluded.created_at, "
                "updated_at=excluded.updated_at, active=excluded.active, "
                # A peer never brings an endorsement: this device's stays
                # only while the fact stays in service, its text and its
                # owner unchanged.
                "endorsed=CASE WHEN memory_facts.active = 1 AND excluded.active = 1 "
                "AND memory_facts.text = excluded.text AND memory_facts.user_id = excluded.user_id "
                "THEN memory_facts.endorsed ELSE NULL END",
                (fid, text, category, source, uid, created_at, upd, active),
            )
        if not active:
            # Out of service on the peer: this device's vector of it goes,
            # as this device's own deletion drops it, or it is owed.
            if not _drop_vector(fid):
                self.owe_vector(fid, "drop")
        elif prior is not None and not prior["active"]:
            # Back in service on the peer: its vector comes back here.
            self.owe_vector(fid, "add")
        return True

    def by_id(self, fact_id: str) -> MemoryRecord | None:
        """A fact by its id, whoever owns it: what the vector layer owes is settled on the fact as it is now."""
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM memory_facts WHERE id = ?", (fact_id,)).fetchone()
        return _row_to_record(row) if row is not None else None

    def owed_vectors(self) -> list[tuple[str, str]]:
        """What the vector layer owes the facts a peer changed: (fact id, ``drop`` or ``add``), oldest first."""
        with self._lock, self._conn() as conn:
            rows = conn.execute("SELECT fact_id, act FROM vectors_owed ORDER BY since").fetchall()
        return [(str(row["fact_id"]), str(row["act"])) for row in rows]

    def settle_vector(self, fact_id: str) -> None:
        """What the vector layer owed a fact is done."""
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM vectors_owed WHERE fact_id = ?", (fact_id,))

    def owe_vector(self, fact_id: str, act: str) -> None:
        """Owe a fact's vector a ``drop`` or an ``add``, for the trash's pass to settle
        (``MemoryStore.settle_vectors``)."""
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO vectors_owed (fact_id, act, since) VALUES (?, ?, ?) "
                "ON CONFLICT(fact_id) DO UPDATE SET act = excluded.act, since = excluded.since",
                (fact_id, act, _now()),
            )


# Module-level singleton with a reset for test isolation (lesson: never
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
