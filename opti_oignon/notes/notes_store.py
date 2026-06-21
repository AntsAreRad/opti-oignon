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
  structure stays end-to-end private (NOTES_FEATURE_ROADMAP.md, the CRDT model).
- Deletions are tombstones (``deleted = 1``), so a delete on one device syncs
  safely once N.8 makes notes a Veilid record type.

This module is importable in isolation (the guarded-import idiom of
canonical_store.py), so the runtime tests can load it without the fastapi /
ollama / sqlcipher chain. ``checkpoint_before_apply`` is hardcoded True and never
overridable; ``FEATURE_AVAILABLE`` gates graceful degradation; the module-level
singleton has a ``reset_notes_store`` hook for test isolation.
"""

from __future__ import annotations

import base64
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
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
# collect without fastapi / ollama / sqlcipher. The S136 change-feed pattern.
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_root() -> Path:
    try:
        from ..config import DATA_DIR

        return Path(DATA_DIR)
    except Exception:
        return Path("data")


# ---------------------------------------------------------------------------
# Veilid sync glue (S257): the notes publisher, best-effort (SYN-01)
# ---------------------------------------------------------------------------

# Serialises clock mint + journal append per process, the skills adaptation
# (``agent/skills.py``): the store's own RLock is deliberately NOT held across
# the journal -- the glue runs after the domain commit, outside the DB lock,
# under this dedicated lock instead, keeping same-key clocks monotonic.
_SYNC_LOCK = threading.Lock()


def _note_sync_payload(record: NoteRecord) -> dict[str, Any]:
    """The full-state, JSON-safe payload a note journals (opaque downstream).

    The opaque CRDT body rides base64-encoded; the tags OR-Set rides as the
    stored opaque string verbatim. ``mobile_allowed`` is deliberately NOT
    carried: the flag is local desktop trust state whose only writer is the
    route's dedicated setter (decision N9-D3) -- carrying it would let a
    future apply path become a writer of it on a receiving device. The
    serve-time filter's LIVE lookup (N9-D1) keys on the serving store, so
    the wire never needs the flag. ``user_id`` is likewise not carried: the
    journal is the single user's own device mesh, and the applier scopes.
    """
    return {
        "title": record.title,
        "body_crdt_b64": base64.b64encode(
            bytes(record.body_crdt or b"")
        ).decode("ascii"),
        "tags": record.tags,
        "pinned": bool(record.pinned),
        "created_at": record.created_at,
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
    republish-on-opt-in delivery contract of NOTES_MOBILE_SYNC_N9_S256.md:
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

    ``mobile_allowed`` (N.9, S256) is the per-item phone-sync opt-in
    (MOBILE_THREAT_MODEL.md section 3): ``False`` is the secure default and
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
                """
            )
            # N.9 (S256): the per-item mobile-allowed flag for phone-class
            # sync. SQLite has no ADD COLUMN IF NOT EXISTS, so guard with
            # table_info (the AU-06 idiom, the peers-registry shape). NOT
            # NULL DEFAULT 0 is the secure default: nothing crosses to a
            # phone until the user opts the item in (MOBILE_THREAT_MODEL.md
            # section 3); every pre-N.9 row reads 0 = not allowed after the
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
        # S257: journal the creation, best-effort, after the commit.
        _sync_publish_note(
            rid, lambda: _note_sync_payload(record), updated_at=ts
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
        # S257: journal the fresh state, best-effort, only when the row
        # exists (an unknown id matched nothing and must not journal).
        if record is not None:
            _sync_publish_note(
                note_id,
                lambda: _note_sync_payload(record),
                updated_at=record.updated_at,
            )
        return record

    def delete_note(self, note_id: str, *, user_id: str | None = None) -> bool:
        """Tombstone delete: set deleted=1 so the deletion syncs (CRDT-safe)."""
        uid = effective_user_id(user_id, self._single_user_mode)
        ts = _now()
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE note SET deleted = 1, updated_at = ? "
                "WHERE id = ? AND user_id = ? AND deleted = 0",
                (ts, note_id, uid),
            )
            changed = int(cur.rowcount)
        # S257: journal the tombstone, best-effort, only when a live row
        # actually flipped (tombstone-wins downstream; a repeat is a no-op).
        if changed > 0:
            _sync_publish_note(note_id, None, deleted=True, updated_at=ts)
        return changed > 0

    # Notes -- the per-item mobile-allowed flag (N.9, S256)

    def set_mobile_allowed(
        self, note_id: str, allowed: bool, *, user_id: str | None = None
    ) -> bool:
        """Set the per-item phone-sync opt-in (MOBILE_THREAT_MODEL.md s.3).

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
        # S257: the republish delivery contract (NOTES_MOBILE_SYNC_N9_S256.md):
        # an effective flip journals a fresh full-state record, best-effort,
        # in BOTH directions -- the phone past its watermark sees a newly
        # allowed note; security stays with the serve filter's live lookup.
        if changed > 0:
            record = self.get_note(note_id, user_id=uid)
            if record is not None:
                _sync_publish_note(
                    note_id,
                    lambda: _note_sync_payload(record),
                    updated_at=record.updated_at,
                )
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
        )
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO attachment "
                "(id, note_id, user_id, kind, blob_ref, mime, bytes, nonce, "
                "created_at, transcript_text, caption_text, ocr_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ),
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
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM attachment WHERE id = ? AND user_id = ?",
                (attachment_id, uid),
            )
            return cur.rowcount > 0

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

        The opt-in, sandboxed post-processing blocs fill the derived-text columns
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


# Module-level singleton with a reset for test isolation (the S171 lesson: never
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
