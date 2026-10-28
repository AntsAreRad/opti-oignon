"""Notes data layer (N.1): the SQLCipher metadata/text store.

The per-user source of truth for note metadata and text: the ``note`` table (the
title, the opaque body CRDT state, the opaque OR-Set tags, the pinned flag, and a
tombstone ``deleted`` flag) and the ``attachment`` manifest (one row per media
blob: its kind, the blob reference, the mime, the byte size, the nonce, and the
opt-in derived text -- transcript / caption / OCR). The encrypted blob bytes
themselves live in :mod:`opti_oignon.notes.blob_store`; this store holds only the
manifest and the searchable text.

Security posture (inherited, not invented):

- At rest: SQLite opened through ``safe_connect`` (SQLCipher when available),
  exactly like auth.db / memory / coding_history. In isolation -- and in this
  test container, where SQLCipher is absent -- ``safe_connect`` degrades to a
  plaintext connection with a once-emitted warning, the documented db_encryption
  posture (and Bulbe refuses to open plaintext); the code path is identical.
- Per-user isolation via a ``user_id`` column and scoped queries, resolved through
  ``effective_user_id`` -- the memory canonical_store pattern, NOT the
  user_data_manager prefix bug (UD-01). Every read and write is scoped by the
  effective user id; a row never leaks across users.
- No f-string SQL anywhere. The one place a column identifier is assembled into a
  statement (the ORDER BY clause) uses ``str.format`` under a frozenset allowlist
  (:data:`_ORDERABLE_COLUMNS` / :data:`NotesStore._UPDATABLE_COLUMNS`), the
  sanctioned alternative; everything else is a constant string or a ``?``
  placeholder.
- The body CRDT and the OR-Set tags are stored OPAQUE (a BLOB and a JSON-ish TEXT
  the store never interprets), so the backend stays CRDT-agnostic and note
  structure stays end-to-end private (the CRDT model).
- Deletions are tombstones (``deleted = 1``), so a delete on one device syncs
  safely once N.8 makes notes a Veilid record type -- and a tombstone keeps no
  words: a deletion, this device's or a peer's, blanks the row to its id, user,
  flag and times, removes its attachments with their derived text and blobs,
  its update tail, and its versions and updates in the sync journal
  (``forget_note``); the sweep (``sweep_dead_notes``) finishes what an
  interrupted or older deletion left.

This module is importable in isolation (the guarded-import idiom of
canonical_store.py), so the runtime tests can load it without the fastapi /
ollama / sqlcipher chain. ``checkpoint_before_apply`` is hardcoded True and never
overridable; ``FEATURE_AVAILABLE`` gates graceful degradation; the module-level
singleton has a ``reset_notes_store`` hook for test isolation.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation is
# applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True


# Guarded backend integration. In the full backend these resolve to the real
# modules. Loaded in isolation, the relative imports fall back to a plain SQLite
# path (with a warning naming the PLAINTEXT degradation) so the runtime tests
# collect without fastapi / ollama / sqlcipher. The change-feed pattern.
try:
    from ..db_utils import safe_connect as _safe_connect

    _HAS_SAFE_CONNECT = True
except Exception:  # ImportError, or relative-import-beyond-top-level in isolation
    _HAS_SAFE_CONNECT = False

    def _safe_connect(  # type: ignore[misc]
        db_path: Any,
        *,
        check_same_thread: bool = True,
        timeout: float = 5.0,
    ) -> sqlite3.Connection:
        logger.warning(
            "notes store falling back to PLAINTEXT sqlite (db_utils unavailable): %s",
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


DB_FILENAME = "notes.db"

# The two physical tables, referenced through these constants so the schema and
# every query stay in lockstep.
NOTE_TABLE = "note"
ATTACHMENT_TABLE = "attachment"

# The attachment-kind allowlist. A frozenset so it doubles as the validation set
# for the only enumerated identifier the API accepts.
ATTACHMENT_KINDS: frozenset[str] = frozenset({"audio", "image", "drawing"})

# Allowlist for the one place a column identifier is assembled into a statement
# (the ORDER BY clause). Used with str.format() under this frozenset so no
# caller-controlled string is interpolated into SQL -- the sanctioned
# alternative to f-string SQL (the canonical_store idiom).
_ORDERABLE_COLUMNS: frozenset[str] = frozenset(
    {"created_at", "updated_at", "title", "pinned"}
)

# Sentinel for partial-update kwargs: distinguishes "field omitted" (leave as is)
# from "field set to None" (write a NULL). Used by update_attachment so a
# write-back that touches one column never blanks the others.
_UNSET: Any = object()

# A deletion's statements, constant strings. The tombstone it leaves holds the id, the user, the flag and its times:
# nothing the user wrote. A tombstone an older deletion left with its words is blanked the same, its time kept.
_KILL_NOTE = (
    "UPDATE note SET title = '', body_crdt = x'', tags = '[]', pinned = 0, "
    "deleted = 1, updated_at = ? WHERE id = ? AND user_id = ? AND deleted = 0"
)
# A peer's tombstone that won the sync journal's verdict, landed before the journal takes it: the flag alone. What
# the note held is forgotten once the tombstone is journalled (``forget_tombstoned``), and kept while it is not.
_FLAG_SYNCED_NOTE = "UPDATE note SET deleted = 1, updated_at = ? WHERE id = ? AND user_id = ?"
_BLANK_DEAD_NOTE = (
    "UPDATE note SET title = '', body_crdt = x'', tags = '[]', pinned = 0 "
    "WHERE id = ? AND user_id = ? AND deleted = 1"
)
# An attachment's derived text and thumbnail go with its note's death; its row goes once its blob is gone, so a blob
# that could not be removed stays named by a row the sweep finds.
_BLANK_NOTE_ATTACHMENTS = (
    "UPDATE attachment SET transcript_text = NULL, caption_text = NULL, ocr_text = NULL, thumbnail = x'' "
    "WHERE note_id = ? AND user_id = ?"
)
_SELECT_NOTE_ATTACHMENTS = "SELECT id FROM attachment WHERE note_id = ? AND user_id = ? ORDER BY id"
_DELETE_ATTACHMENT_ROW = "DELETE FROM attachment WHERE id = ? AND user_id = ?"
_SELECT_DEAD_IDS = "SELECT id FROM note WHERE user_id = ? AND deleted = 1 ORDER BY id"
_SELECT_DELETED = "SELECT deleted FROM note WHERE id = ? AND user_id = ?"

# The update store's file, beside this one (``note_updates_store.DB_FILENAME``; spelled here so that loading this
# store loads nothing else).
_NOTE_UPDATES_FILENAME = "note_updates.db"
# The blob store's folder and suffix beside this one (``blob_store.BLOB_DIRNAME``, ``BLOB_SUFFIX``).
_BLOB_DIRNAME = "notes_blobs"
_BLOB_SUFFIX = ".blob"


def _absent(exc: ModuleNotFoundError, module: str) -> bool:
    """Whether an import failed because the module, or a package it would be in, is not in this tree -- then it
    holds nothing. A module that is there and fails to load is broken, and said."""
    return bool(exc.name) and (exc.name == module or module.startswith(exc.name + "."))


def _holding_appends() -> Any:
    """The update store's process-wide append lock (``note_updates_store.holding_appends``), held while a deletion
    commits its tombstone: every update appended before it is in the tail the deletion then prunes, and every one
    after it finds the note dead. Nothing to hold in a tree without the update store."""
    try:
        from opti_oignon.notes.note_updates_store import holding_appends
    except ModuleNotFoundError as exc:
        if _absent(exc, "opti_oignon.notes.note_updates_store"):
            return nullcontext()
        raise
    return holding_appends()


# The attachments a peer's tombstone took off this device's manifest are kept, by note, in ``awaited_blob`` until the
# sync journal's verdict -- in the file, so a restart does not lose them: their blobs are forgotten once the tombstone
# is journalled (it won here) or the note is deleted here, kept while neither -- it lost, and the version that won
# may name them again.
_SELECT_AWAITED = "SELECT attachment_id FROM awaited_blob WHERE note_id = ? AND user_id = ? ORDER BY attachment_id"
_DELETE_AWAITED = "DELETE FROM awaited_blob WHERE note_id = ? AND user_id = ?"
_INSERT_AWAITED = "INSERT OR IGNORE INTO awaited_blob (note_id, attachment_id, user_id) VALUES (?, ?, ?)"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_root() -> Path:
    try:
        from ..config import DATA_DIR

        return Path(DATA_DIR)
    except Exception:
        return Path("data")


# ---------------------------------------------------------------------------
# Veilid sync glue: the notes publisher, best-effort
# ---------------------------------------------------------------------------

# Serialises clock mint + journal append per process, the skills adaptation
# (``agent/skills.py``): the store's own RLock is deliberately NOT held across
# the journal -- the glue runs after the domain commit, outside the DB lock,
# under this dedicated lock instead, keeping same-key clocks monotonic.
_SYNC_LOCK = threading.Lock()


def _note_sync_payload(
    record: NoteRecord, attachments: list[AttachmentRecord]
) -> dict[str, Any]:
    """The full-state, JSON-safe payload a note journals (opaque downstream).

    The opaque CRDT body rides base64-encoded; the tags OR-Set rides as the
    stored opaque string verbatim. ``mobile_allowed`` is deliberately NOT
    carried: the flag is local desktop trust state whose only writer is the
    route's dedicated setter (decision N9-D3) -- carrying it would let a
    future apply path become a writer of it on a receiving device. The
    serve-time filter's LIVE lookup (N9-D1) keys on the serving store, so
    the wire never needs the flag. ``user_id`` is likewise not carried: the
    journal is the single user's own device mesh, and the applier scopes.

    ``attachments`` is the eager vault manifest (SYN-01): the AUTHORITATIVE set
    of the note's media, each entry carrying its id, kind, mime, plaintext
    ``content_hash``, ``blob_ref``, and a small ``thumbnail`` -- but NOT the
    full bytes, which a peer fetches on demand. A peer reconciles its attachment
    rows to exactly this set (added, removed), so the caller MUST pass the
    note's CURRENT full attachment list (an empty list legitimately clears it);
    the argument is required so no publisher can silently wipe a peer's media.
    """
    manifest: list[dict[str, Any]] = []
    for a in attachments:
        manifest.append(
            {
                "id": a.id,
                "kind": a.kind,
                "mime": a.mime,
                "byte_size": int(a.byte_size),
                "blob_ref": a.blob_ref,
                "content_hash": a.content_hash,
                "thumbnail_b64": base64.b64encode(
                    bytes(a.thumbnail or b"")
                ).decode("ascii"),
            }
        )
    return {
        "title": record.title,
        "body_crdt_b64": base64.b64encode(
            bytes(record.body_crdt or b"")
        ).decode("ascii"),
        "tags": record.tags,
        "pinned": bool(record.pinned),
        "created_at": record.created_at,
        "attachments": manifest,
    }


def _sync_publish_note(
    note_id: str,
    payload_fn: Any = None,
    *,
    deleted: bool = False,
    updated_at: str = "",
) -> None:
    """Journal a note change for Veilid sync, best-effort (SYN-01).

    Called by the store's mutation seams AFTER the domain commit --
    ``add_note``, ``update_note`` (when a column was actually applied),
    ``delete_note`` (the tombstone), and ``set_mobile_allowed`` (the
    republish-on-opt-in delivery contract:
    a flag flipped to allowed journals a fresh record, or a phone whose
    watermark has advanced past the filtered entries never sees the newly
    allowed note; the flip journals in BOTH directions -- republish is
    delivery, not security, the serve filter's live lookup staying the
    authority). Sitting at the store layer, the glue covers every caller:
    the routes, the gated ``manage_notes`` tool, and any future seam.
    ``payload_fn`` is a zero-arg callable building the full-state payload;
    it runs INSIDE this hook's protection, and only after the availability
    probe passes, so when sync is absent the save pays nothing (no payload
    build, no journal append). The contract (the conversation / skills
    precedents):

    - A payload or journalling failure must never break the save: any error
      is logged and swallowed (at-least-once on the next mutation).
    - No-op when the optional veilid framework is absent
      (``guard.veilid_available`` is the cheap probe); a quiet no-op too
      when the sync package itself is unreachable in this interpreter (a
      flat-loaded store under a stubbed test environment).
    - Mode-free: producing and journalling are local-disk operations
      permitted in ANY mode (the documented ``producers.py`` posture); only
      the wire is Daily-gated, downstream at the engine/guard.
    - The flag never rides the payload (see ``_note_sync_payload``).

    Clock discipline: next = the highest clock journalled for the key, plus
    one (an unseen key yields 0, so the first clock is 1). ``_SYNC_LOCK``
    serialises mint + append per process.
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
            payload: dict[str, Any] | None = None
            if not deleted:
                payload = payload_fn() if payload_fn is not None else None
                if payload is None:
                    # The state could not be built. Publishing an empty
                    # non-tombstone payload would wipe the note on peers
                    # under LWW -- skip instead.
                    logger.debug(
                        "sync publish skipped for note %s: no state available",
                        note_id,
                    )
                    return
            from opti_oignon.veilid.records import RecordKind
            from opti_oignon.veilid.sync_engine import get_sync_engine

            engine = get_sync_engine()
            clock = engine.current_clock(RecordKind.NOTE, note_id) + 1
            engine.publish_note(
                note_id,
                payload,
                clock=clock,
                deleted=deleted,
                updated_at=updated_at,
            )
    except Exception:
        logger.warning(
            "veilid sync publish failed for note %s (save unaffected)",
            note_id,
            exc_info=True,
        )


@dataclass
class NoteRecord:
    """A single note's metadata and opaque CRDT body.

    ``mobile_allowed`` is the per-item phone-sync opt-in:
    ``False`` is the secure default and
    the only creation-time value; flipping it is a deliberate second gesture
    through the dedicated setter, never the generic update path.
    """

    id: str
    user_id: str
    title: str
    body_crdt: bytes
    tags: str
    pinned: bool
    created_at: str
    updated_at: str
    deleted: bool
    mobile_allowed: bool = False


@dataclass
class AttachmentRecord:
    """One media blob's manifest row (the bytes live in the blob store)."""

    id: str
    note_id: str
    user_id: str
    kind: str
    blob_ref: str
    mime: str
    byte_size: int
    nonce: str
    created_at: str
    transcript_text: str | None = None
    caption_text: str | None = None
    ocr_text: str | None = None
    # Vault sync (SYN-01): the eager manifest fields that ride the NOTE record
    # so a peer sees the item before any bytes transfer. ``content_hash`` is the
    # PLAINTEXT SHA-256 of the blob (cross-device integrity, set at upload);
    # ``thumbnail`` is a small downscaled preview (generated frontend-side at
    # upload, dep-free; empty for a video until a poster frame exists). The full
    # bytes are fetched on demand (the chunked transfer is a separate lot).
    content_hash: str = ""
    thumbnail: bytes = b""


def _row_to_note(row: sqlite3.Row) -> NoteRecord:
    body = row["body_crdt"]
    # Defensive read (N9-D2 direction): a row from a connection opened before
    # the migration ran simply reads not-allowed.
    allowed_raw = row["mobile_allowed"] if "mobile_allowed" in row.keys() else 0
    return NoteRecord(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        body_crdt=bytes(body) if body is not None else b"",
        tags=row["tags"],
        pinned=bool(row["pinned"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        deleted=bool(row["deleted"]),
        mobile_allowed=bool(allowed_raw),
    )


def _row_to_attachment(row: sqlite3.Row) -> AttachmentRecord:
    return AttachmentRecord(
        id=row["id"],
        note_id=row["note_id"],
        user_id=row["user_id"],
        kind=row["kind"],
        blob_ref=row["blob_ref"],
        mime=row["mime"],
        byte_size=int(row["bytes"]),
        nonce=row["nonce"],
        created_at=row["created_at"],
        transcript_text=row["transcript_text"],
        caption_text=row["caption_text"],
        ocr_text=row["ocr_text"],
        content_hash=row["content_hash"] if "content_hash" in row.keys() else "",
        thumbnail=(
            bytes(row["thumbnail"])
            if "thumbnail" in row.keys() and row["thumbnail"] is not None
            else b""
        ),
    )


class NotesStore:
    """SQLite-backed notes metadata/text store (the per-user source of truth)."""

    # Allowlist for the dynamic UPDATE SET clause; mirrors _ORDERABLE_COLUMNS in
    # purpose (no caller-controlled identifier reaches the SQL text).
    _UPDATABLE_COLUMNS: frozenset[str] = frozenset(
        {"title", "body_crdt", "tags", "pinned"}
    )

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        single_user_mode: bool = True,
    ) -> None:
        base = Path(root) if root is not None else _default_root()
        base.mkdir(parents=True, exist_ok=True)
        self._db_path = base / DB_FILENAME
        self._single_user_mode = single_user_mode
        self._lock = threading.RLock()
        self._init_db()

    # Connection handling

    def _connect(self) -> sqlite3.Connection:
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL is set outside any transaction (right after connect).
        conn.execute("PRAGMA journal_mode=WAL")
        # What a deletion frees is overwritten, never left in the file's free pages.
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
                CREATE TABLE IF NOT EXISTS note (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    title TEXT NOT NULL DEFAULT '',
                    body_crdt BLOB NOT NULL DEFAULT x'',
                    tags TEXT NOT NULL DEFAULT '[]',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    mobile_allowed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS attachment (
                    id TEXT PRIMARY KEY,
                    note_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    kind TEXT NOT NULL,
                    blob_ref TEXT NOT NULL,
                    mime TEXT NOT NULL DEFAULT '',
                    bytes INTEGER NOT NULL DEFAULT 0,
                    nonce TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    transcript_text TEXT,
                    caption_text TEXT,
                    ocr_text TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_note_user ON note(user_id);
                CREATE INDEX IF NOT EXISTS idx_note_deleted ON note(deleted);
                CREATE INDEX IF NOT EXISTS idx_attachment_user
                    ON attachment(user_id);
                CREATE INDEX IF NOT EXISTS idx_attachment_note
                    ON attachment(note_id);
                CREATE TABLE IF NOT EXISTS awaited_blob (
                    note_id TEXT NOT NULL,
                    attachment_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'local',
                    PRIMARY KEY (note_id, attachment_id)
                );
                """
            )
            # The per-item mobile-allowed flag for phone-class
            # sync. SQLite has no ADD COLUMN IF NOT EXISTS, so guard with
            # table_info (the AU-06 idiom, the peers-registry shape). NOT
            # NULL DEFAULT 0 is the secure default: nothing crosses to a
            # phone until the user opts the item in;
            # every pre-N.9 row reads 0 = not allowed after the
            # migration, by construction.
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(note)").fetchall()
            }
            if "mobile_allowed" not in cols:
                conn.execute(
                    "ALTER TABLE note ADD COLUMN "
                    "mobile_allowed INTEGER NOT NULL DEFAULT 0"
                )
            # Vault sync (SYN-01): the eager manifest columns on attachment,
            # same table_info guard (the AU-06 idiom). ``content_hash`` is the
            # PLAINTEXT blob digest (cross-device integrity); ``thumbnail`` is a
            # small preview that rides the NOTE record so a peer sees the item
            # before the bytes transfer. Both take empty defaults, so every
            # pre-existing row reads empty after the migration.
            acols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(attachment)"
                ).fetchall()
            }
            if "content_hash" not in acols:
                conn.execute(
                    "ALTER TABLE attachment ADD COLUMN "
                    "content_hash TEXT NOT NULL DEFAULT ''"
                )
            if "thumbnail" not in acols:
                conn.execute(
                    "ALTER TABLE attachment ADD COLUMN "
                    "thumbnail BLOB NOT NULL DEFAULT x''"
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
        return effective_user_id(user_id, self._single_user_mode)

    def close(self) -> None:
        # Connections are opened per-operation and closed in _conn(); nothing is
        # held open. Present for parity with the other stores' lifecycle.
        return None

    # Notes -- create

    def add_note(
        self,
        title: str,
        *,
        body_crdt: bytes = b"",
        tags: str | None = None,
        pinned: bool = False,
        user_id: str | None = None,
        note_id: str | None = None,
    ) -> NoteRecord:
        """Insert a note and return its record."""
        uid = effective_user_id(user_id, self._single_user_mode)
        rid = note_id or uuid.uuid4().hex
        if ":" in rid:
            # A note's updates are journalled as ``note_id:seq``: an id with a colon would share another's.
            raise ValueError(f"a note id cannot hold ':': {rid!r}")
        ts = _now()
        tags_value = tags if tags is not None else "[]"
        record = NoteRecord(
            id=rid,
            user_id=uid,
            title=title,
            body_crdt=body_crdt,
            tags=tags_value,
            pinned=pinned,
            created_at=ts,
            updated_at=ts,
            deleted=False,
            # N.9: creation never opts a note onto the phone (the secure
            # default); the flag is a deliberate second gesture.
            mobile_allowed=False,
        )
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO note "
                "(id, user_id, title, body_crdt, tags, pinned, created_at, "
                "updated_at, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (rid, uid, title, body_crdt, tags_value, int(pinned), ts, ts),
            )
        # Journal the creation, best-effort, after the commit.
        _sync_publish_note(
            rid, lambda: _note_sync_payload(record, []), updated_at=ts
        )
        return record

    # Notes -- read

    def get_note(
        self, note_id: str, *, user_id: str | None = None
    ) -> NoteRecord | None:
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM note WHERE id = ? AND user_id = ?",
                (note_id, uid),
            ).fetchone()
        return _row_to_note(row) if row is not None else None

    def list_notes(
        self,
        *,
        user_id: str | None = None,
        include_deleted: bool = False,
        pinned_only: bool = False,
        order_by: str = "updated_at",
        descending: bool = True,
        limit: int | None = None,
    ) -> list[NoteRecord]:
        if order_by not in _ORDERABLE_COLUMNS:
            order_by = "updated_at"
        uid = effective_user_id(user_id, self._single_user_mode)

        # Every fragment below is a constant string or a "?" placeholder; the
        # only identifiers are user_id / deleted / pinned (literals) and the
        # allowlisted order_by. No caller value reaches the SQL text.
        clauses = ["user_id = ?"]
        params: list[Any] = [uid]
        if not include_deleted:
            clauses.append("deleted = 0")
        if pinned_only:
            clauses.append("pinned = 1")
        where = " AND ".join(clauses)
        direction = "DESC" if descending else "ASC"
        sql = f"SELECT * FROM note WHERE {where} ORDER BY {order_by} {direction}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_note(r) for r in rows]

    def count_notes(
        self, *, user_id: str | None = None, include_deleted: bool = False
    ) -> int:
        uid = effective_user_id(user_id, self._single_user_mode)
        sql = "SELECT COUNT(*) AS n FROM note WHERE user_id = ?"
        if not include_deleted:
            sql += " AND deleted = 0"
        with self._lock, self._conn() as conn:
            row = conn.execute(sql, (uid,)).fetchone()
        return int(row["n"])

    # Notes -- update / delete

    def update_note(
        self,
        note_id: str,
        *,
        user_id: str | None = None,
        **fields: Any,
    ) -> NoteRecord | None:
        """Update one or more allowlisted columns; refreshes updated_at."""
        uid = effective_user_id(user_id, self._single_user_mode)
        columns: list[str] = []
        params: list[Any] = []
        for key, value in fields.items():
            if key not in self._UPDATABLE_COLUMNS:
                raise ValueError("Not an updatable column: " + repr(key))
            columns.append(f"{key} = ?")
            if key == "pinned":
                value = int(bool(value))
            params.append(value)
        if not columns:
            return self.get_note(note_id, user_id=uid)
        ts = _now()
        columns.append("updated_at = ?")
        params.append(ts)
        params.extend([note_id, uid])
        sql = "UPDATE note SET {} WHERE id = ? AND user_id = ?".format(
            ", ".join(columns)
        )
        with self._lock, self._conn() as conn:
            conn.execute(sql, params)
        record = self.get_note(note_id, user_id=uid)
        # Journal the fresh state (with the current attachment manifest),
        # best-effort, only when the row exists.
        self._republish_note_full_state(note_id, record, user_id=uid)
        return record

    def delete_note(self, note_id: str, *, user_id: str | None = None) -> bool:
        """Tombstone delete that keeps no words (``forget_note``); whether a live note died. What could not be
        forgotten is logged, and the next sweep finishes it."""
        died, left = self.forget_note(note_id, user_id=user_id)
        if left:
            logger.warning("the note %s deleted; not forgotten yet: %s", note_id, "; ".join(left))
        return died

    def forget_note(
        self, note_id: str, *, user_id: str | None = None, sweep: bool = True
    ) -> tuple[bool, list[str]]:
        """Delete a note so that nothing it held stays: whether a live note died, and what could not be forgotten.

        One transaction, no update appended while it commits (``_holding_appends``), leaves the tombstone sync
        needs -- the id, the user, ``deleted = 1`` and its times, the title, body, tags and pin emptied (a dead note
        is blanked the same) -- and empties the derived text and thumbnails of the note's attachments. After the
        commit: the sync journal's versions of the note and every update of its body erased, the update tail and
        checkpoint pruned, each attachment's blob overwritten and removed before its row, the log emptied, then the
        sweep of every dead note's residue; then the tombstone published, as before, when a live note died -- or
        when the note was dead here while the journal still holds a version of it alive (a peer's tombstone that
        lost): this deletion is the user's own, and its tombstone out-clocks that version. A step after the commit
        that fails is named in the list and logged, the deletion standing: the next sweep finishes it.
        ``sweep=False`` leaves the sweep to a caller deleting many (the wipe).
        """
        uid = effective_user_id(user_id, self._single_user_mode)
        ts = _now()
        with _holding_appends(), self._lock, self._conn() as conn:
            died = int(conn.execute(_KILL_NOTE, (ts, note_id, uid)).rowcount) > 0
            if not died:
                conn.execute(_BLANK_DEAD_NOTE, (note_id, uid))
            conn.execute(_BLANK_NOTE_ATTACHMENTS, (note_id, uid))
        left: list[str] = []
        states = self._journal_states([note_id], left)
        state = states.get(note_id) if states is not None else None
        disputed = not died and state == "live"
        # A deletion here whose tombstone the journal never took: the peers are told now.
        owed = not died and state == "owed"
        # The journal's versions go only when this deletion's tombstone follows them: a note already dead here keeps
        # the tombstone the journal holds -- erased, no peer that had not pulled it would ever learn the deletion.
        left += self._forget_after(note_id, uid, versions=died or disputed)
        left += self._forget_awaited(note_id, uid)
        if sweep:
            left += self.sweep_dead_notes(user_id=uid)
        if died or disputed or owed:
            left += self._tell_peers(note_id, ts)
        return died, left

    def forget_tombstoned(self, note_id: str, *, user_id: str | None = None) -> list[str]:
        """A note whose tombstone the sync journal now holds -- this device's own, or a peer's that won here -- if it
        is dead in this store: its row blanked, its attachments' text emptied, then its updates in the journal, its
        update tail, its attachments' blobs and rows, and the blobs a peer's tombstone took off its manifest
        (``_forget_after``; the journal's tombstone stays); what could not be forgotten, named. A live note -- a
        version that won after the tombstone -- is not touched; a store whose file is not there holds none, and is
        not created. Hook-free (no publish): the journal asks it once its append is done."""
        if not self._db_path.exists():
            return []
        uid = effective_user_id(user_id, self._single_user_mode)
        # No update appended while the note's death is read and its row blanked: one appended before is in the
        # tail forgotten below, one after finds the note dead (``_holding_appends``).
        with _holding_appends(), self._lock, self._conn() as conn:
            row = conn.execute(_SELECT_DELETED, (note_id, uid)).fetchone()
            if row is None or not row["deleted"]:
                return []
            conn.execute(_BLANK_DEAD_NOTE, (note_id, uid))
            conn.execute(_BLANK_NOTE_ATTACHMENTS, (note_id, uid))
        return self._forget_after(note_id, uid, versions=False) + self._forget_awaited(note_id, uid)

    @staticmethod
    def _journal_states(note_ids: list[str], left: list[str]) -> dict[str, str] | None:
        """What the sync journal holds of each of these notes (``change_feed.note_states``: ``tombstone``, ``live``,
        ``owed``, ``none``, ``absent``); None when it cannot be read, said in ``left``. A tree without the journal
        holds nothing: ``absent``."""
        try:
            from opti_oignon.veilid.change_feed import note_states

            return note_states(note_ids)
        except ModuleNotFoundError as exc:
            if _absent(exc, "opti_oignon.veilid.change_feed"):
                return {note_id: "absent" for note_id in note_ids}
            left.append(f"sync journal (not read: {type(exc).__name__})")
        except Exception as exc:  # noqa: BLE001 - what stays is said
            logger.warning("the sync journal could not be read for the dead notes: %s", exc)
            left.append(f"sync journal (not read: {type(exc).__name__})")
        return None

    def _tell_peers(self, note_id: str, ts: str) -> list[str]:
        """Publish the note's tombstone; when the journal, which exists, does not hold it after -- the framework
        absent, or the publication failed --, the peers were not told: said, and owed (``note_states``), so the next
        deletion or the sweep publishes it."""
        _sync_publish_note(note_id, None, deleted=True, updated_at=ts)
        after = self._journal_states([note_id], [])
        if after is None or after.get(note_id) in ("tombstone", "absent"):
            return []
        return [f"paired devices (not told of the note {note_id}'s deletion: the sync journal did not take its "
                "tombstone; the next deletion or wipe tells them)"]

    def _forget_after(self, note_id: str, uid: str, *, versions: bool) -> list[str]:
        """What a note's death leaves outside its row, forgotten: its versions (``versions``: this device's own
        deletion, whose tombstone is minted after) and its updates in the sync journal, its update tail, its
        attachments' blobs then their rows, the log; each step that fails named. A store whose module this tree
        does not have holds nothing of the note; one whose file is not there holds none, and is not created."""
        left: list[str] = []
        try:
            from opti_oignon.veilid.change_feed import forget_note as _forget_in_feed

            _forget_in_feed(note_id, versions=versions)
        except ModuleNotFoundError as exc:
            if not _absent(exc, "opti_oignon.veilid.change_feed"):
                left.append(f"sync journal (not forgotten: {type(exc).__name__})")
        except Exception as exc:  # noqa: BLE001 - what stays is said
            logger.warning("note %s deleted; the sync journal could not forget it: %s", note_id, exc)
            left.append(f"sync journal (not forgotten: {type(exc).__name__})")
        left += self._forget_updates([note_id], uid)
        with self._lock, self._conn() as conn:
            attachments = [str(r[0]) for r in conn.execute(_SELECT_NOTE_ATTACHMENTS, (note_id, uid)).fetchall()]
        left += self._forget_attachments(attachments, uid)
        self._empty_log()
        return left

    def _forget_updates(self, note_ids: list[str], uid: str) -> list[str]:
        """The update tails and checkpoints of these dead notes pruned, through the update store beside this one;
        each that fails named."""
        if not note_ids or not (self._db_path.parent / _NOTE_UPDATES_FILENAME).exists():
            return []
        try:
            from opti_oignon.notes.note_updates_store import NoteUpdatesStore

            updates = NoteUpdatesStore(root=self._db_path.parent, single_user_mode=self._single_user_mode)
        except ModuleNotFoundError as exc:
            if _absent(exc, "opti_oignon.notes.note_updates_store"):
                return []
            return [f"note updates (not pruned: {type(exc).__name__})"]
        except Exception as exc:  # noqa: BLE001 - what stays is said
            logger.warning("the note updates store could not be opened to prune dead notes: %s", exc)
            return [f"note updates (not pruned: {type(exc).__name__})"]
        left = []
        for note_id in note_ids:
            try:
                updates.forget_dead_note(note_id, user_id=uid)
            except Exception as exc:  # noqa: BLE001 - what stays is said
                logger.warning("the update tail of the dead note %s could not be pruned: %s", note_id, exc)
                left.append(f"note updates of {note_id} (not pruned: {type(exc).__name__})")
        return left

    def _forget_awaited(self, note_id: str, uid: str) -> list[str]:
        """The blobs a peer's tombstone took off the note's manifest here (``awaited_blob``), overwritten and removed
        now that the note is deleted for good, and their ids let go; what could not be, named and kept."""
        with self._lock, self._conn() as conn:
            awaited = [str(r[0]) for r in conn.execute(_SELECT_AWAITED, (note_id, uid)).fetchall()]
        if not awaited:
            return []
        left = self._forget_blobs(awaited)
        if not left:
            with self._lock, self._conn() as conn:
                conn.execute(_DELETE_AWAITED, (note_id, uid))
        return left

    def _forget_blobs(self, attachment_ids: list[str]) -> list[str]:
        """The blobs of attachments whose rows are already gone (a peer's tombstone dropped them), overwritten and
        removed, unless a row names one again; each that could not be, named."""
        if not attachment_ids:
            return []
        try:
            from opti_oignon.notes.blob_store import forget_blob
        except ModuleNotFoundError as exc:
            if _absent(exc, "opti_oignon.notes.blob_store"):
                return []
            return [f"note attachments' blobs (not removed: {type(exc).__name__})"]
        with self._lock, self._conn() as conn:
            named = {str(r[0]) for r in conn.execute("SELECT id FROM attachment").fetchall()}
        left = []
        for aid in attachment_ids:
            if aid in named:
                continue
            try:
                forget_blob(self._db_path.parent, aid)
            except Exception as exc:  # noqa: BLE001 - what stays is said
                logger.warning("the blob %s of a dead note could not be removed: %s", aid, exc)
                left.append(f"note attachment {aid} (blob not removed: {type(exc).__name__})")
        return left

    def _forget_attachments(self, attachment_ids: list[str], uid: str) -> list[str]:
        """Each attachment's blob overwritten and removed, then its row; one whose blob could not be keeps its row,
        named, for the next sweep."""
        if not attachment_ids:
            return []
        try:
            from opti_oignon.notes.blob_store import forget_blob
        except ModuleNotFoundError as exc:
            if not _absent(exc, "opti_oignon.notes.blob_store"):
                return [f"note attachments (not removed: {type(exc).__name__})"]
            forget_blob = None
        left = []
        for aid in attachment_ids:
            try:
                if forget_blob is not None:
                    forget_blob(self._db_path.parent, aid)
                with self._lock, self._conn() as conn:
                    conn.execute(_DELETE_ATTACHMENT_ROW, (aid, uid))
            except Exception as exc:  # noqa: BLE001 - what stays is said
                logger.warning("the attachment %s of a dead note could not be removed: %s", aid, exc)
                left.append(f"note attachment {aid} (not removed: {type(exc).__name__})")
        return left

    def sweep_dead_notes(self, *, user_id: str | None = None) -> list[str]:
        """Forget what every dead note still holds -- a tombstone with words (a deletion made before these rules),
        attachments, an update tail, updates in the sync journal -- unless the journal still holds a version of it
        alive: a peer's tombstone that landed here and then lost, the note alive everywhere else, is never taken for
        a deletion (it waits for the user's own, or for a version that brings it back). What could not be forgotten,
        named; a journal that cannot be read leaves every dead note as it is, said. A live note is never touched.
        Idempotent and cheap when nothing is left; each deletion runs it, so what an older or an interrupted
        deletion left goes at the next one."""
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            dead = [str(r[0]) for r in conn.execute(_SELECT_DEAD_IDS, (uid,)).fetchall()]
        if not dead:
            return []
        left: list[str] = []
        states = self._journal_states(dead, left)
        if states is None:
            return left
        settled = [note_id for note_id in dead if states.get(note_id) != "live"]
        # What an earlier deletion could not tell the peers, told now.
        for note_id in settled:
            if states.get(note_id) == "owed":
                left += self._tell_peers(note_id, _now())
        if not settled:
            return left
        attachments: list[str] = []
        with self._lock, self._conn() as conn:
            for note_id in settled:
                conn.execute(_BLANK_DEAD_NOTE, (note_id, uid))
                conn.execute(_BLANK_NOTE_ATTACHMENTS, (note_id, uid))
                attachments += [str(r[0]) for r in conn.execute(_SELECT_NOTE_ATTACHMENTS, (note_id, uid)).fetchall()]
        try:
            from opti_oignon.veilid.change_feed import forget_dead_notes

            forget_dead_notes(settled)
        except ModuleNotFoundError as exc:
            if not _absent(exc, "opti_oignon.veilid.change_feed"):
                left.append(f"sync journal (dead notes not forgotten: {type(exc).__name__})")
        except Exception as exc:  # noqa: BLE001 - what stays is said
            logger.warning("the sync journal could not forget the dead notes: %s", exc)
            left.append(f"sync journal (dead notes not forgotten: {type(exc).__name__})")
        held = self._updated_notes(uid)
        if held is None:
            left.append("note updates (dead notes not pruned: the store could not be read)")
        else:
            left += self._forget_updates(sorted(held & set(settled)), uid)
        left += self._forget_attachments(attachments, uid)
        for note_id in settled:
            left += self._forget_awaited(note_id, uid)
        self._empty_log()
        return left

    def live_ids(self, *, user_id: str | None = None) -> list[str]:
        """Every live note's id, oldest first."""
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM note WHERE user_id = ? AND deleted = 0 ORDER BY created_at, id",
                (uid,),
            ).fetchall()
        return [str(r[0]) for r in rows]

    def counts(self, *, user_id: str | None = None) -> dict[str, int]:
        """What the wipe's preview counts: the live notes, their attachments, the notes dead here that the sync
        journal holds alive (``disputed``: hidden here, alive on the paired devices -- the wipe deletes them too; None
        when the journal cannot be read), and the dead notes (whose tombstones stay, and whose residue the wipe
        forgets)."""
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            live = conn.execute(
                "SELECT COUNT(*) FROM note WHERE user_id = ? AND deleted = 0", (uid,)
            ).fetchone()[0]
            dead = conn.execute(
                "SELECT COUNT(*) FROM note WHERE user_id = ? AND deleted = 1", (uid,)
            ).fetchone()[0]
            attachments = conn.execute(
                "SELECT COUNT(*) FROM attachment WHERE user_id = ? AND note_id IN "
                "(SELECT id FROM note WHERE user_id = ? AND deleted = 0)",
                (uid, uid),
            ).fetchone()[0]
        disputed = self.disputed_ids(user_id=uid)
        return {"notes": int(live), "attachments": int(attachments),
                "disputed": len(disputed) if disputed is not None else None, "deleted_notes": int(dead)}

    def fingerprint(self, *, user_id: str | None = None) -> str:
        """Four hexadecimal characters drawn from every live note's id and its attachments' ids, and from the
        disputed notes' ids: the wipe's phrase carries them, so a note or an attachment that came or went since its
        preview -- the count unchanged -- refuses the wipe."""
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT note.id, attachment.id FROM note LEFT JOIN attachment "
                "ON attachment.note_id = note.id AND attachment.user_id = note.user_id "
                "WHERE note.user_id = ? AND note.deleted = 0 ORDER BY note.id, attachment.id",
                (uid,),
            ).fetchall()
        disputed = self.disputed_ids(user_id=uid)
        seen = "\n".join([f"{r[0]}:{r[1] or ''}" for r in rows]
                         + ([f"{note_id}:disputed" for note_id in disputed] if disputed is not None else ["unread"]))
        return hashlib.sha256(seen.encode("utf-8")).hexdigest()[:4]

    def disputed_ids(self, *, user_id: str | None = None) -> list[str] | None:
        """The notes dead here that the sync journal still holds alive -- a peer's tombstone that landed and then
        lost: the wipe deletes them as the user's own deletion does, publishing a tombstone that out-clocks the
        version alive. None when the journal cannot be read: never taken for none."""
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            dead = [str(r[0]) for r in conn.execute(_SELECT_DEAD_IDS, (uid,)).fetchall()]
        if not dead:
            return []
        states = self._journal_states(dead, [])
        if states is None:
            return None
        return [note_id for note_id in dead if states.get(note_id) == "live"]

    def forget_orphan_blobs(self, *, before: float | None = None) -> list[str]:
        """Every blob in the folder beside this store that no attachment row names, overwritten and removed -- what a
        deletion made before these rules, or a peer's tombstone, left -- each that could not be, named. The wipe runs
        it once every note is dead, with ``before`` its own start: a blob written since -- an upload in flight, its
        row not yet written -- is not one."""
        folder = self._db_path.parent / _BLOB_DIRNAME
        if not folder.is_dir():
            return []
        with self._lock, self._conn() as conn:
            named = {str(r[0]) for r in conn.execute("SELECT id FROM attachment").fetchall()}
        try:
            from opti_oignon.notes.blob_store import forget_blob
        except ModuleNotFoundError as exc:
            if _absent(exc, "opti_oignon.notes.blob_store"):
                return []
            return [f"note attachments' blobs (not removed: {type(exc).__name__})"]
        left = []
        stems: dict[str, list[Path]] = {}
        for path in sorted(folder.iterdir()):
            name = path.name
            stem = name[:-len(_BLOB_SUFFIX)] if name.endswith(_BLOB_SUFFIX) else (
                name[:-len(_BLOB_SUFFIX + ".tmp")] if name.endswith(_BLOB_SUFFIX + ".tmp") else None)
            if stem and stem not in named:
                stems.setdefault(stem, []).append(path)
        for stem, paths in stems.items():
            try:
                # A blob or its partial file written since the wipe began is an upload in flight.
                if before is not None and any(p.stat().st_mtime >= before for p in paths if p.exists()):
                    continue
                forget_blob(self._db_path.parent, stem)
            except Exception as exc:  # noqa: BLE001 - what stays is said
                logger.warning("the blob %s no note names could not be removed: %s", stem, exc)
                left.append(f"note attachment {stem} (blob not removed: {type(exc).__name__})")
        return left

    def _updated_notes(self, uid: str) -> set[str] | None:
        """The notes the update store beside this one holds anything of; None when it cannot be read."""
        if not (self._db_path.parent / _NOTE_UPDATES_FILENAME).exists():
            return set()
        try:
            from opti_oignon.notes.note_updates_store import NoteUpdatesStore

            return NoteUpdatesStore(root=self._db_path.parent,
                                    single_user_mode=self._single_user_mode).notes_with_updates(user_id=uid)
        except ModuleNotFoundError as exc:
            return set() if _absent(exc, "opti_oignon.notes.note_updates_store") else None
        except Exception as exc:  # noqa: BLE001 - what stays is said
            logger.warning("the note updates store could not be read for the dead notes: %s", exc)
            return None

    def _empty_log(self) -> None:
        """Empty the write-ahead log after a deletion: its older frames still hold what the deletion blanked. Never
        waits: a read in progress keeps the log, said, until the next deletion."""
        try:
            conn = _safe_connect(self._db_path, check_same_thread=False)
            try:
                conn.execute("PRAGMA busy_timeout = 0")
                busy = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - the cipher driver's errors are not sqlite3's, and this only says
            logger.warning("notes: a note deleted; the write-ahead log could not be emptied (%s: %s)",
                           type(exc).__name__, exc)
            return
        if busy:
            logger.warning("notes: a note deleted; a read in progress keeps the write-ahead log, whose older frames "
                           "hold what the deletion blanked, until the next deletion")

    def apply_synced_note(
        self,
        note_id: str,
        payload: dict[str, Any] | None,
        *,
        deleted: bool = False,
        updated_at: str = "",
    ) -> bool:
        """Materialise a synced note record into the local store (SYN-01 apply).

        The receiving half for notes. Deliberately HOOK-FREE -- it never calls
        ``_sync_publish_note`` -- so applying a received record cannot
        re-publish it and start an apply -> write -> publish echo. Two
        invariants the apply path MUST hold, both load-bearing:

          * It NEVER writes ``mobile_allowed``. The wire never carries the flag
            (decision N9-D3); a received note must not be able to opt ITSELF
            onto the phone. On create the row takes the schema default (0, the
            secure OFF); on update the column is untouched -- exactly what
            ``create_note`` and ``update_note`` already guarantee.
          * It NEVER overwrites ``body_crdt`` of an EXISTING note. The text body
            converges through NOTE_UPDATE (the CRDT relay's append seam), so the
            full-state record owns only existence + metadata (title, tags,
            pinned). The body is seeded ONLY when the note is first created
            here; thereafter NOTE_UPDATE owns it and a stale snapshot can never
            regress it.

        A ``deleted`` record tombstones (``deleted = 1``), CRDT-safe, never a
        hard delete, so the deletion converges and tombstone-wins downstream --
        and it forgets as this device's own deletion does (``forget_note``),
        hook-free: the row blanked, the attachments, blobs, update tail and the
        journal's updates of the note gone; the peer's tombstone itself is
        journalled after, by the round. A version that brings a dead note back
        (an edit that won under last-writer-wins) lands as a creation: its body
        seeded from the payload, the one here being gone.
        Idempotent (INSERT/UPDATE by id) and fail-secure (False on a malformed
        payload or any write error, never raising into the round). Scoped to the
        local user via the ``effective_user_id`` pattern; the note id is the
        record id (the note payload does not carry it).
        """
        try:
            if not isinstance(note_id, str) or not note_id or ":" in note_id:
                # A note's updates are journalled as ``note_id:seq``: an id with a colon would share another's.
                return False
            if not deleted and not isinstance(payload, dict):
                return False
            uid = effective_user_id(None, self._single_user_mode)
            ts = updated_at if isinstance(updated_at, str) and updated_at else _now()
            if deleted:
                return self._apply_synced_tombstone(note_id, uid, ts)
            with self._lock, self._conn() as conn:
                held = conn.execute(
                    "SELECT deleted FROM note WHERE id = ? AND user_id = ?",
                    (note_id, uid),
                ).fetchone()
                exists = held is not None and not held["deleted"]
                forgotten = held is not None and bool(held["deleted"])

                title = payload.get("title")
                if not isinstance(title, str):
                    title = ""
                tags = payload.get("tags")
                if not isinstance(tags, str):
                    tags = "[]"
                pinned = 1 if payload.get("pinned") else 0

                if exists:
                    # Metadata only: body_crdt (CRDT, owned by NOTE_UPDATE) and
                    # mobile_allowed (N9-D3) are deliberately untouched. A dead
                    # note comes back through the creation branch below.
                    conn.execute(
                        "UPDATE note SET title = ?, tags = ?, pinned = ?, "
                        "deleted = 0, updated_at = ? WHERE id = ? AND user_id = ?",
                        (title, tags, pinned, ts, note_id, uid),
                    )
                else:
                    created_at = payload.get("created_at")
                    if not isinstance(created_at, str) or not created_at:
                        created_at = ts
                    body_b64 = payload.get("body_crdt_b64")
                    body = b""
                    if isinstance(body_b64, str):
                        try:
                            body = base64.b64decode(body_b64)
                        except Exception:
                            body = b""
                    if forgotten:
                        # A dead note brought back by a version that won: as a creation, its body seeded -- the
                        # deletion left none here; mobile_allowed untouched (N9-D3).
                        conn.execute(
                            "UPDATE note SET title = ?, body_crdt = ?, tags = ?, pinned = ?, "
                            "created_at = ?, deleted = 0, updated_at = ? WHERE id = ? AND user_id = ?",
                            (title, body, tags, pinned, created_at, ts, note_id, uid),
                        )
                    else:
                        # mobile_allowed omitted -> schema default 0 (secure OFF).
                        conn.execute(
                            "INSERT INTO note "
                            "(id, user_id, title, body_crdt, tags, pinned, "
                            "created_at, updated_at, deleted) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                            (note_id, uid, title, body, tags, pinned, created_at, ts),
                        )

                # Reconcile the eager vault manifest to the authoritative set:
                # UPSERT each entry (no blob bytes -- fetched on demand -- and
                # ON CONFLICT preserves an already-downloaded blob's local nonce
                # and derived text), then prune rows no longer in the set.
                manifest = payload.get("attachments")
                if not isinstance(manifest, list):
                    manifest = []
                seen: list[str] = []
                for a in manifest:
                    if not isinstance(a, dict):
                        continue
                    aid = a.get("id")
                    kind = a.get("kind")
                    if not isinstance(aid, str) or not aid or "/" in aid or "\\" in aid or aid in (".", ".."):
                        # An attachment's id names its blob's file: one that is no plain file name is refused.
                        continue
                    if kind not in ATTACHMENT_KINDS:
                        continue
                    seen.append(aid)
                    blob_ref = a.get("blob_ref")
                    blob_ref = blob_ref if isinstance(blob_ref, str) else ""
                    mime = a.get("mime")
                    mime = mime if isinstance(mime, str) else ""
                    content_hash = a.get("content_hash")
                    content_hash = content_hash if isinstance(content_hash, str) else ""
                    byte_size = a.get("byte_size")
                    if not isinstance(byte_size, int) or isinstance(byte_size, bool):
                        byte_size = 0
                    thumb_b64 = a.get("thumbnail_b64")
                    thumb = b""
                    if isinstance(thumb_b64, str):
                        try:
                            thumb = base64.b64decode(thumb_b64)
                        except Exception:
                            thumb = b""
                    conn.execute(
                        "INSERT INTO attachment "
                        "(id, note_id, user_id, kind, blob_ref, mime, bytes, "
                        "created_at, content_hash, thumbnail) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET "
                        "kind=excluded.kind, blob_ref=excluded.blob_ref, "
                        "mime=excluded.mime, bytes=excluded.bytes, "
                        "content_hash=excluded.content_hash, "
                        "thumbnail=excluded.thumbnail",
                        (
                            aid,
                            note_id,
                            uid,
                            kind,
                            blob_ref,
                            mime,
                            int(byte_size),
                            ts,
                            content_hash,
                            thumb,
                        ),
                    )
                released: list[str] = []
                for r in conn.execute(
                    "SELECT id FROM attachment WHERE note_id = ? AND user_id = ?",
                    (note_id, uid),
                ).fetchall():
                    if r[0] not in seen:
                        conn.execute(
                            "DELETE FROM attachment WHERE id = ? AND user_id = ?",
                            (r[0], uid),
                        )
                        released.append(str(r[0]))
                # The attachments a losing tombstone took off the manifest that this version names no more.
                released += [str(r[0]) for r in conn.execute(_SELECT_AWAITED, (note_id, uid)).fetchall()
                             if r[0] not in seen]
                conn.execute(_DELETE_AWAITED, (note_id, uid))
            # An attachment the version no longer names keeps no words here: its blob goes with its row.
            left = self._forget_blobs(released)
            if left:
                logger.warning("a note's version landed; not forgotten yet: %s", "; ".join(left))
            return True
        except Exception:
            logger.warning(
                "veilid sync apply failed for a note (no domain change)",
                exc_info=True,
            )
            return False

    def _republish_note_full_state(
        self,
        note_id: str,
        record: NoteRecord | None,
        *,
        user_id: str | None = None,
    ) -> None:
        """Journal a note's fresh full state WITH its current attachment manifest.

        The single place the SYN-01 note publish is assembled, so every
        publisher (create, update, mobile-allowed flip, attachment add/remove)
        carries the authoritative manifest and none can silently wipe a peer's
        media. Best-effort: a ``None`` record (the row is gone) journals
        nothing, and the hook itself no-ops when sync is unavailable.
        """
        if record is None:
            return
        attachments = self.list_attachments(note_id, user_id=user_id)
        _sync_publish_note(
            note_id,
            lambda: _note_sync_payload(record, attachments),
            updated_at=record.updated_at,
        )

    # Notes -- the per-item mobile-allowed flag

    def _apply_synced_tombstone(self, note_id: str, uid: str, ts: str) -> bool:
        """A peer's tombstone that won the sync journal's verdict, landed before the journal takes it: the flag, or a
        tombstone recorded for a note never seen here, so a later out-of-order non-tombstone cannot silently
        resurrect it; the attachment manifest dropped, its ids kept (``awaited_blob``) until the journal holds it.
        Nothing else: the journal may not take it (unreadable, its write failing), and a note dead here before 192
        may be one whose tombstone lost. Once the journal holds it ``forget_tombstoned`` forgets the note; while it
        does not, the sweep leaves the note as it is. Hook-free (no publish). mobile_allowed keeps the secure
        default."""
        with self._lock, self._conn() as conn:
            dropped = [str(r[0]) for r in conn.execute(_SELECT_NOTE_ATTACHMENTS, (note_id, uid)).fetchall()]
            killed = conn.execute(_FLAG_SYNCED_NOTE, (ts, note_id, uid)).rowcount
            if not killed:
                conn.execute(
                    "INSERT INTO note "
                    "(id, user_id, title, body_crdt, tags, pinned, "
                    "created_at, updated_at, deleted) "
                    "VALUES (?, ?, '', x'', '[]', 0, ?, ?, 1)",
                    (note_id, uid, ts, ts),
                )
            # A tombstoned note drops its attachment manifest too; the ids are kept for the verdict.
            conn.execute(
                "DELETE FROM attachment WHERE note_id = ? AND user_id = ?",
                (note_id, uid),
            )
            for aid in dropped:
                conn.execute(_INSERT_AWAITED, (note_id, aid, uid))
        return True

    def set_mobile_allowed(
        self, note_id: str, allowed: bool, *, user_id: str | None = None
    ) -> bool:
        """Set the per-item phone-sync opt-in.

        A human trust decision made at the desktop: the route is the only
        caller, and the flag is deliberately NOT in ``_UPDATABLE_COLUMNS``,
        so neither the generic update path nor the gated ``manage_notes``
        tool can ever flip it (decision N9-D3). Scoped per user; a
        tombstoned or unknown note is never updated (returns ``False``).
        """
        uid = effective_user_id(user_id, self._single_user_mode)
        ts = _now()
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE note SET mobile_allowed = ?, updated_at = ? "
                "WHERE id = ? AND user_id = ? AND deleted = 0",
                (1 if allowed else 0, ts, note_id, uid),
            )
            changed = int(cur.rowcount)
        # The republish delivery contract:
        # an effective flip journals a fresh full-state record, best-effort,
        # in BOTH directions -- the phone past its watermark sees a newly
        # allowed note; security stays with the serve filter's live lookup.
        if changed > 0:
            record = self.get_note(note_id, user_id=uid)
            self._republish_note_full_state(note_id, record, user_id=uid)
        return changed > 0

    def is_mobile_allowed(
        self, note_id: str, *, user_id: str | None = None
    ) -> bool:
        """Fail-secure read of the phone-sync opt-in (decision N9-D2).

        ``True`` only for an existing, non-tombstoned, user-scoped note
        whose flag is affirmatively 1. Anything indeterminable -- an unknown
        note, a tombstone, an unreadable store, any error -- reads
        ``False``: an absent or unreadable flag means NOT allowed. This is
        the live lookup the sync responder's phone-class filter consults
        (filter-at-serve, decision N9-D1).
        """
        try:
            uid = effective_user_id(user_id, self._single_user_mode)
            with self._lock, self._conn() as conn:
                row = conn.execute(
                    "SELECT mobile_allowed FROM note "
                    "WHERE id = ? AND user_id = ? AND deleted = 0",
                    (note_id, uid),
                ).fetchone()
            if row is None:
                return False
            return int(row["mobile_allowed"]) == 1
        except Exception:
            return False

    # Attachments

    def add_attachment(
        self,
        note_id: str,
        kind: str,
        *,
        blob_ref: str,
        mime: str = "",
        byte_size: int = 0,
        nonce: str = "",
        user_id: str | None = None,
        attachment_id: str | None = None,
        transcript_text: str | None = None,
        caption_text: str | None = None,
        ocr_text: str | None = None,
        content_hash: str = "",
        thumbnail: bytes = b"",
    ) -> AttachmentRecord:
        """Insert an attachment manifest row. ``kind`` is allowlisted."""
        if kind not in ATTACHMENT_KINDS:
            raise ValueError("Invalid attachment kind: " + repr(kind))
        uid = effective_user_id(user_id, self._single_user_mode)
        aid = attachment_id or uuid.uuid4().hex
        ts = _now()
        record = AttachmentRecord(
            id=aid,
            note_id=note_id,
            user_id=uid,
            kind=kind,
            blob_ref=blob_ref,
            mime=mime,
            byte_size=byte_size,
            nonce=nonce,
            created_at=ts,
            transcript_text=transcript_text,
            caption_text=caption_text,
            ocr_text=ocr_text,
            content_hash=content_hash,
            thumbnail=thumbnail,
        )
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO attachment "
                "(id, note_id, user_id, kind, blob_ref, mime, bytes, nonce, "
                "created_at, transcript_text, caption_text, ocr_text, "
                "content_hash, thumbnail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    aid,
                    note_id,
                    uid,
                    kind,
                    blob_ref,
                    mime,
                    int(byte_size),
                    nonce,
                    ts,
                    transcript_text,
                    caption_text,
                    ocr_text,
                    content_hash,
                    thumbnail,
                ),
            )
        # Vault sync (SYN-01): the note's manifest gained an item, so republish
        # its full state with the new manifest (best-effort) so peers reconcile
        # the item in.
        self._republish_note_full_state(
            note_id, self.get_note(note_id, user_id=uid), user_id=uid
        )
        return record

    def get_attachment(
        self, attachment_id: str, *, user_id: str | None = None
    ) -> AttachmentRecord | None:
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM attachment WHERE id = ? AND user_id = ?",
                (attachment_id, uid),
            ).fetchone()
        return _row_to_attachment(row) if row is not None else None

    def list_attachments(
        self, note_id: str, *, user_id: str | None = None
    ) -> list[AttachmentRecord]:
        uid = effective_user_id(user_id, self._single_user_mode)
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM attachment WHERE note_id = ? AND user_id = ? "
                "ORDER BY created_at ASC",
                (note_id, uid),
            ).fetchall()
        return [_row_to_attachment(r) for r in rows]

    def delete_attachment(
        self, attachment_id: str, *, user_id: str | None = None
    ) -> bool:
        """Hard-delete one attachment's manifest row; return whether a row went.

        The encrypted blob in the blob store is the caller's to remove (the route
        deletes the blob then the manifest, so a missing-key blob store cannot
        leave a dangling manifest). Per-user scoped and parameterized -- no
        f-string SQL, no caller-controlled identifier in the statement.
        """
        uid = effective_user_id(user_id, self._single_user_mode)
        existing = self.get_attachment(attachment_id, user_id=uid)
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM attachment WHERE id = ? AND user_id = ?",
                (attachment_id, uid),
            )
            changed = cur.rowcount > 0
        if changed and existing is not None:
            # Vault sync (SYN-01): the note's manifest lost an item, so
            # republish so peers reconcile it out (best-effort).
            self._republish_note_full_state(
                existing.note_id,
                self.get_note(existing.note_id, user_id=uid),
                user_id=uid,
            )
        return changed

    def update_attachment(
        self,
        attachment_id: str,
        *,
        transcript_text: Any = _UNSET,
        caption_text: Any = _UNSET,
        ocr_text: Any = _UNSET,
        user_id: str | None = None,
    ) -> bool:
        """Write derived post-processing text back onto an existing manifest row.

        The opt-in, sandboxed post-processing fills the derived-text columns
        AFTER the human approves the result: transcript_text (audio, N.5),
        caption_text / ocr_text (image, N.6). Only the fields explicitly passed
        are written, so a call that omits a field never blanks an existing value,
        and a call with no fields is a no-op that merely reports whether the row
        exists. Per-user scoped and parameterized: the column names are fixed
        literals chosen by the if-branches (never caller-controlled), and no SQL
        is assembled by f-string.

        Returns whether a matching row was found (and updated, when fields were
        given).
        """
        uid = effective_user_id(user_id, self._single_user_mode)
        assignments: list[str] = []
        params: list[Any] = []
        if transcript_text is not _UNSET:
            assignments.append("transcript_text = ?")
            params.append(transcript_text)
        if caption_text is not _UNSET:
            assignments.append("caption_text = ?")
            params.append(caption_text)
        if ocr_text is not _UNSET:
            assignments.append("ocr_text = ?")
            params.append(ocr_text)
        with self._lock, self._conn() as conn:
            if not assignments:
                row = conn.execute(
                    "SELECT 1 FROM attachment WHERE id = ? AND user_id = ?",
                    (attachment_id, uid),
                ).fetchone()
                return row is not None
            params.extend([attachment_id, uid])
            cur = conn.execute(
                "UPDATE attachment SET "
                + ", ".join(assignments)
                + " WHERE id = ? AND user_id = ?",
                params,
            )
            return cur.rowcount > 0


# Module-level singleton with a reset for test isolation (the lesson: never
# leak shared state across pytest invocations).
_store: NotesStore | None = None


def get_notes_store() -> NotesStore:
    global _store
    if _store is None:
        _store = NotesStore()
    return _store


def reset_notes_store() -> None:
    global _store
    _store = None


def forget_tombstoned(note_id: str) -> list[str]:
    """The sync journal now holds the note's tombstone: the process's notes store, if it opened one, forgets what it
    still holds of the note when it is dead there (``NotesStore.forget_tombstoned``); a store not opened in this
    process is not asked -- its next sweep does it."""
    store = _store
    if store is None:
        return []
    return store.forget_tombstoned(note_id)
