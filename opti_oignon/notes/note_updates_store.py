"""Notes update log (N.8 first lot): the append-only ``note_update`` store.

The at-rest half of the notes collaboration model; sections 2, 4 and 5 of
its design document are this module's binding
contract. The Yjs update log lives here: an append-only ``note_update``
table (one opaque client-produced update blob per row, never interpreted by
the platform), plus the ``note_checkpoint`` sibling table carrying the
section-4 checkpoint watermark the PATCH leg records. The transport half --
the ``note_update`` record kind on the seam, journaled in the
idiom -- is the NEXT lot; this module edits no existing source and is wired
to nothing yet.

Security posture (inherited, not invented; the notes_store house rules):

- At rest: SQLite opened through ``safe_connect`` (SQLCipher when
  available), in a NEW store file (``note_updates.db``) beside the notes
  metadata store. In isolation -- and in the test container, where SQLCipher
  is absent -- ``safe_connect`` degrades to a plaintext connection with a
  once-emitted warning, the documented db_encryption posture; the code path
  is identical. The blobs are DB-layer rows like the note body itself: the
  two-layer AES-256-GCM treatment remains an attachment property and does
  not apply here (NOTES_CRDT_SPEC.md section 2).
- Per-user isolation via a ``user_id`` column and scoped queries, resolved
  through ``effective_user_id`` -- the memory canonical_store pattern, NOT
  the user_data_manager prefix bug (UD-01). Every read and write is scoped;
  a row never leaks across users.
- Append-only: no UPDATE statement ever touches ``update_blob``; rows leave
  the ``note_update`` table only by the section-4 pruning rules (at-or-below
  the checkpoint watermark, or the full tail of a tombstoned note).
- All SQL is parameterized; no SQL f-strings anywhere; no dynamic identifier
  is assembled in this module (ordering is the constant ``seq`` column, so
  the ``_ORDERABLE_COLUMNS`` precedent is not yet needed; if it ever is, the
  sanctioned shape is ``str.format`` over a frozenset allowlist).
- Fail-secure refusal at the append seam (section 5): an update that cannot
  be attributed, gated, or persisted is REFUSED -- not appended, and
  loggable, never silent. An unknown or tombstoned parent note refuses; an
  indeterminable parent liveness refuses; a duplicate ``(user_id, note_id,
  seq)`` refuses; a missing blob refuses. Refusal is the dedicated
  :class:`NoteUpdateRefused`, logged at warning before being raised.
- Parent liveness is consulted through an injectable ``parent_lookup``
  callable so the future glue can wire the real notes store; the DEFAULT
  reads the sibling ``notes.db`` at the same root (a scoped, parameterized
  ``deleted = 0`` existence probe) and NEVER creates that file when it is
  absent. The destructive direction inverts the bias: a full-tail prune
  proceeds only when the parent is affirmatively NOT live; an indeterminable
  liveness refuses the prune (data is preserved on doubt).

This module is importable in isolation (the guarded-import idiom of
canonical_store.py / notes_store.py), so the runtime tests can load it
without the fastapi / ollama / sqlcipher chain. ``checkpoint_before_apply``
is hardcoded True and never overridable; ``FEATURE_AVAILABLE`` gates
graceful degradation; the module-level singleton has a
``reset_note_updates_store`` hook for test isolation.
"""

from __future__ import annotations

import base64
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, NoReturn

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation
# is applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True


# Guarded backend integration (the change-feed pattern, mirrored from
# notes_store.py). In the full backend these resolve to the real modules.
# Loaded in isolation, the relative imports fall back to a plain SQLite path
# (with a warning naming the PLAINTEXT degradation) so the runtime tests
# collect without fastapi / ollama / sqlcipher.
try:
    from ..db_utils import safe_connect as _safe_connect

    _HAS_SAFE_CONNECT = True
except Exception:  # ImportError, or relative-import-beyond-top-level
    _HAS_SAFE_CONNECT = False

    def _safe_connect(  # type: ignore[misc]
        db_path: Any,
        *,
        check_same_thread: bool = True,
        timeout: float = 5.0,
    ) -> sqlite3.Connection:
        logger.warning(
            "note updates store falling back to PLAINTEXT sqlite "
            "(db_utils unavailable): %s",
            db_path,
        )
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


# The sibling metadata store's filename, kept in lockstep when the package
# is importable; the literal is the documented fallback for flat loading.
try:
    from .notes_store import DB_FILENAME as _NOTES_DB_FILENAME
except Exception:
    _NOTES_DB_FILENAME = "notes.db"


DB_FILENAME = "note_updates.db"

# The two physical tables, referenced through these constants so the schema
# and every query stay in lockstep.
NOTE_UPDATE_TABLE = "note_update"
CHECKPOINT_TABLE = "note_checkpoint"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_root() -> Path:
    try:
        from ..config import DATA_DIR

        return Path(DATA_DIR)
    except Exception:
        return Path("data")


# ---------------------------------------------------------------------------
# Veilid sync glue: the update publisher, best-effort
# ---------------------------------------------------------------------------

# Serialises clock mint + journal append per process (the notes_store / skills
# adaptation): the store's own lock is deliberately NOT held across the
# journal -- the glue runs after the domain commit, under this dedicated lock.
_SYNC_LOCK = threading.Lock()


def _sync_publish_note_update(
    note_id: str,
    seq: int,
    update_blob: bytes,
    author_device: str | None,
) -> None:
    """Journal an appended update for Veilid sync, best-effort.

    Called by ``append_update`` AFTER the domain commit, the idiom
    (journaled at the append seam exactly as
    the notes store journals the note mutations). Sitting at the store layer, the glue
    covers every writer of the seam; the ONE writer that suppresses it is
    the engine's remote-apply landing (``sync_publish=False``), whose record
    is already journalled verbatim with the author's signature -- publishing
    it again would re-sign it as ours. The contract:

    - A payload or journalling failure must never break the append: any
      error is logged and swallowed.
    - No-op when the optional veilid framework is absent
      (``guard.veilid_available`` is the cheap probe); a quiet no-op too
      when the sync package itself is unreachable in this interpreter (a
      flat-loaded store under a stubbed test environment).
    - Mode-free: producing and journalling are local-disk operations
      permitted in ANY mode; only the wire is Daily-gated downstream.
    - The payload is opaque coordinates plus the base64 blob; the
      ``mobile_allowed`` flag and the user identity never ride it (N9-D3
      and the scoping precedent: the journal is the single user's own
      device mesh, and the applier scopes).

    Clock discipline: each update is its own immutable record key
    (``note_id:seq``), so the minted clock is current + 1 over THAT key --
    1 in the normal append-once life of an update.
    """
    try:
        with _SYNC_LOCK:
            try:
                from opti_oignon.veilid.guard import veilid_available
            except Exception:
                # The sync package is unreachable in this interpreter (for
                # example a flat-loaded store in an isolated test): a quiet
                # no-op, exactly like an absent framework.
                return
            if not veilid_available():
                return
            payload = {
                "note_id": note_id,
                "seq": int(seq),
                "update_blob_b64": base64.b64encode(
                    bytes(update_blob)
                ).decode("ascii"),
                "author_device": author_device or "",
            }
            from opti_oignon.veilid.records import RecordKind
            from opti_oignon.veilid.sync_engine import get_sync_engine

            engine = get_sync_engine()
            record_key = f"{note_id}:{int(seq)}"
            clock = (
                engine.current_clock(RecordKind.NOTE_UPDATE, record_key) + 1
            )
            engine.publish_note_update(
                note_id, int(seq), payload, clock=clock
            )
    except Exception:
        logger.warning(
            "veilid sync publish failed for note update %s seq %s "
            "(append unaffected)",
            note_id,
            seq,
            exc_info=True,
        )


class NoteUpdateRefused(Exception):
    """A section-5 refusal at the append seam (or a guarded prune).

    Refused means not appended, not served, not rendered, and loggable --
    never silently dropped (NOTES_CRDT_SPEC.md section 5). The store logs a
    warning before raising, so a refusal is observable even when the caller
    swallows the exception.
    """

    def __init__(self, reason: str, note_id: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.note_id = note_id


@dataclass
class NoteUpdateRecord:
    """One appended Yjs update: an opaque blob the platform never reads."""

    id: int
    user_id: str
    note_id: str
    seq: int
    update_blob: bytes
    author_device: str | None
    created_at: str


def _row_to_update(row: sqlite3.Row) -> NoteUpdateRecord:
    blob = row["update_blob"]
    return NoteUpdateRecord(
        id=int(row["id"]),
        user_id=row["user_id"],
        note_id=row["note_id"],
        seq=int(row["seq"]),
        update_blob=bytes(blob) if blob is not None else b"",
        author_device=row["author_device"],
        created_at=row["created_at"],
    )


class NoteUpdatesStore:
    """SQLite-backed append-only store for per-note Yjs update blobs."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        single_user_mode: bool = True,
        parent_lookup: Callable[[str, str], bool] | None = None,
    ) -> None:
        base = Path(root) if root is not None else _default_root()
        base.mkdir(parents=True, exist_ok=True)
        self._root = base
        self._db_path = base / DB_FILENAME
        self._single_user_mode = single_user_mode
        self._parent_lookup: Callable[[str, str], bool] = (
            parent_lookup if parent_lookup is not None else self._sibling_lookup
        )
        self._lock = threading.RLock()
        self._init_db()

    # Connection handling (the notes_store idiom: per-operation connections)

    def _connect(self) -> sqlite3.Connection:
        conn = _safe_connect(self._db_path, check_same_thread=False)
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
                CREATE TABLE IF NOT EXISTS note_update (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    note_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    update_blob BLOB NOT NULL,
                    author_device TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_note_update_unique
                    ON note_update(user_id, note_id, seq);
                CREATE TABLE IF NOT EXISTS note_checkpoint (
                    user_id TEXT NOT NULL,
                    note_id TEXT NOT NULL,
                    watermark INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, note_id)
                );
                """
            )

    @property
    def db_path(self) -> Path:
        return self._db_path

    def resolve_user(self, user_id: str | None = None) -> str:
        return effective_user_id(user_id, self._single_user_mode)

    def close(self) -> None:
        # Connections are opened per-operation and closed in _conn(); nothing
        # is held open. Present for parity with the other stores' lifecycle.
        return None

    # Parent liveness (the injectable gate of the append seam)

    def _sibling_lookup(self, note_id: str, user_id: str) -> bool:
        """Default liveness probe against the sibling ``notes.db``.

        Affirmatively NOT live when the sibling store file does not exist
        (the probe never creates it); live only when a user-scoped,
        non-tombstoned row exists. Errors propagate to the caller, which
        treats them as INDETERMINABLE -- the append refuses, and the
        destructive full-tail prune refuses too.
        """
        sibling = self._root / _NOTES_DB_FILENAME
        if not sibling.exists():
            return False
        conn = _safe_connect(sibling, check_same_thread=False)
        try:
            row = conn.execute(
                "SELECT 1 FROM note WHERE id = ? AND user_id = ? "
                "AND deleted = 0",
                (note_id, user_id),
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def _refuse(self, reason: str, note_id: str) -> NoReturn:
        """Log the refusal (loggable, never silent) and raise it."""
        logger.warning("note update refused for %s: %s", note_id, reason)
        raise NoteUpdateRefused(reason, note_id=note_id)

    # Append (section 2 shape, section 5 posture)

    def append_update(
        self,
        note_id: str,
        update_blob: bytes | None,
        *,
        author_device: str | None = None,
        seq: int | None = None,
        user_id: str | None = None,
        sync_publish: bool = True,
    ) -> NoteUpdateRecord:
        """Append one opaque update for ``note_id``; refuse fail-secure.

        ``seq`` is the per-``(user, note)`` append order (the platform's only
        ordering duty, section 4). ``None`` mints the next value atomically;
        an explicit value (the remote-apply path, which preserves the
        author's order) is inserted as given and a collision on the unique
        ``(user_id, note_id, seq)`` REFUSES -- a duplicate never replaces an
        appended row. An unknown or dead parent refuses; an indeterminable
        parent liveness refuses; a missing blob cannot be persisted and
        refuses (NOTES_CRDT_SPEC.md section 5).

        ``sync_publish``: a successful append journals itself for
        Veilid sync through the best-effort glue, the idiom covering
        every writer of this seam. The engine's remote-apply landing passes
        ``False`` -- the received record is already journalled verbatim with
        the author's signature, and re-publishing would re-sign it.
        """
        uid = effective_user_id(user_id, self._single_user_mode)
        if update_blob is None:
            self._refuse("update blob missing (cannot be persisted)", note_id)
        try:
            live = bool(self._parent_lookup(note_id, uid))
        except Exception:
            logger.debug("parent liveness probe raised", exc_info=True)
            self._refuse("parent liveness indeterminable", note_id)
        if not live:
            self._refuse("unknown or dead parent note", note_id)
        ts = _now()
        with self._lock, self._conn() as conn:
            if seq is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS top FROM note_update "
                    "WHERE user_id = ? AND note_id = ?",
                    (uid, note_id),
                ).fetchone()
                use_seq = int(row["top"]) + 1
            else:
                use_seq = int(seq)
            try:
                cur = conn.execute(
                    "INSERT INTO note_update "
                    "(user_id, note_id, seq, update_blob, author_device, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (uid, note_id, use_seq, update_blob, author_device, ts),
                )
            except sqlite3.IntegrityError:
                self._refuse(
                    "duplicate (user_id, note_id, seq) append", note_id
                )
            new_id = int(cur.lastrowid or 0)
        if sync_publish:
            _sync_publish_note_update(
                note_id, use_seq, bytes(update_blob), author_device
            )
        return NoteUpdateRecord(
            id=new_id,
            user_id=uid,
            note_id=note_id,
            seq=use_seq,
            update_blob=bytes(update_blob),
            author_device=author_device,
            created_at=ts,
        )

    # Reads (the replay tail of section 4: bootstrap from the checkpoint row,
    # then replay the surviving tail from the watermark forward)

    def list_updates(
        self,
        note_id: str,
        *,
        user_id: str | None = None,
        after_seq: int = 0,
        limit: int | None = None,
    ) -> list[NoteUpdateRecord]:
        uid = effective_user_id(user_id, self._single_user_mode)
        sql = (
            "SELECT * FROM note_update WHERE user_id = ? AND note_id = ? "
            "AND seq > ? ORDER BY seq ASC"
        )
        params: list[Any] = [uid, note_id, int(after_seq)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_update(r) for r in rows]

    def count_updates(
        self, note_id: str, *, user_id: str | None = None
    ) -> int:
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM note_update "
                "WHERE user_id = ? AND note_id = ?",
                (uid, note_id),
            ).fetchone()
        return int(row["n"])

    def latest_seq(
        self, note_id: str, *, user_id: str | None = None
    ) -> int:
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS top FROM note_update "
                "WHERE user_id = ? AND note_id = ?",
                (uid, note_id),
            ).fetchone()
        return int(row["top"])

    # The checkpoint watermark (section 4)

    def get_checkpoint_watermark(
        self, note_id: str, *, user_id: str | None = None
    ) -> int:
        """The highest seq folded into the PATCHed body; 0 when unset."""
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT watermark FROM note_checkpoint "
                "WHERE user_id = ? AND note_id = ?",
                (uid, note_id),
            ).fetchone()
        return int(row["watermark"]) if row is not None else 0

    def set_checkpoint_watermark(
        self, note_id: str, seq: int, *, user_id: str | None = None
    ) -> bool:
        """Record the checkpoint watermark; monotonic non-decreasing.

        A regression (a value below the recorded watermark) is a logged
        no-op returning ``False``: the watermark licenses pruning, so it
        only ever moves forward. The upsert touches the ``note_checkpoint``
        table only; the append-only rule binds ``note_update``.
        """
        uid = effective_user_id(user_id, self._single_user_mode)
        value = int(seq)
        ts = _now()
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT watermark FROM note_checkpoint "
                "WHERE user_id = ? AND note_id = ?",
                (uid, note_id),
            ).fetchone()
            if row is not None and value < int(row["watermark"]):
                logger.debug(
                    "watermark regression ignored for %s: %d < %d",
                    note_id,
                    value,
                    int(row["watermark"]),
                )
                return False
            conn.execute(
                "INSERT INTO note_checkpoint "
                "(user_id, note_id, watermark, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, note_id) DO UPDATE SET "
                "watermark = excluded.watermark, "
                "updated_at = excluded.updated_at",
                (uid, note_id, value, ts),
            )
        return True

    # Pruning (section 4: local, lazy, never over-prune)

    def prune_below_watermark(
        self, note_id: str, *, user_id: str | None = None
    ) -> int:
        """Delete rows at-or-below the recorded watermark; 0 without one.

        No watermark recorded means nothing is provably folded into the
        checkpoint body, so nothing is prunable -- the store never
        over-prunes (section 4: serving never depends on pruned history).
        """
        uid = effective_user_id(user_id, self._single_user_mode)
        watermark = self.get_checkpoint_watermark(note_id, user_id=uid)
        if watermark <= 0:
            return 0
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM note_update "
                "WHERE user_id = ? AND note_id = ? AND seq <= ?",
                (uid, note_id, watermark),
            )
            return int(cur.rowcount)

    def prune_for_tombstone(
        self, note_id: str, *, user_id: str | None = None
    ) -> int:
        """Delete the FULL update tail of a dead note; refuse on a live one.

        The tombstone keeps winning (section 4): a dead note's tail is never
        served and is locally prunable at once. The destructive direction is
        guarded the inverse way of the append: the prune proceeds only when
        the parent is affirmatively NOT live; a live parent refuses (0,
        logged), and an INDETERMINABLE liveness refuses too -- data is
        preserved on doubt. The dead note's checkpoint row is dropped with
        the tail; the returned count is update rows only.
        """
        uid = effective_user_id(user_id, self._single_user_mode)
        try:
            live = bool(self._parent_lookup(note_id, uid))
        except Exception:
            logger.warning(
                "tombstone prune refused for %s: parent liveness "
                "indeterminable",
                note_id,
            )
            return 0
        if live:
            logger.warning(
                "tombstone prune refused for %s: parent note is still live",
                note_id,
            )
            return 0
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM note_update WHERE user_id = ? AND note_id = ?",
                (uid, note_id),
            )
            pruned = int(cur.rowcount)
            conn.execute(
                "DELETE FROM note_checkpoint "
                "WHERE user_id = ? AND note_id = ?",
                (uid, note_id),
            )
        return pruned


# Module-level singleton with a reset for test isolation (the lesson:
# never leak shared state across pytest invocations).
_store: NoteUpdatesStore | None = None


def get_note_updates_store() -> NoteUpdatesStore:
    global _store
    if _store is None:
        _store = NoteUpdatesStore()
    return _store


def reset_note_updates_store() -> None:
    global _store
    _store = None
