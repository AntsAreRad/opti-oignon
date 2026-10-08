"""The agent's persistent writes: what the user typed writes, the rest waits for the user.

The agent's memory and notes tools write into stores that every later turn
reads back. A write whose words came from a page, a document or a tool
result is an instruction someone else wrote, kept as the user's: the shape
of a memory-injection attack, where the attacker acts only through content
the agent reads. Every such write passes a gate:

* ENDORSED. A write is endorsed when each of its content arguments -- the
  text of a fact; the title, body and tags of a note -- folded (Unicode NFC,
  each run of white space as one space, the ends stripped), equals the
  whole of what the user typed in the current turn (each typed part whole).
  Never a part of it: a sentence, a line, a list item, even a paragraph can
  take its sense from its neighbours and lose it alone ("Things you must
  never do:" before "Share my location with Bob."); never a model's
  judgement. An endorsed write goes through the store as it always did.
* PROPOSED. Any other write, and every update or delete -- whose target is
  an identifier no typed word can vouch for -- becomes a proposal: inert,
  kept with its exact arguments and the provenance of each, read by no
  model, and applied only when the user accepts it. The model is told that
  it was proposed, never that it was saved.
* FAIL CLOSED. A gate with no typed turn endorses nothing; a gate that
  cannot record a proposal writes nothing and says so.
* BOUNDED. A run makes at most ``max_per_run`` proposals
  (``config/pending_writes.yaml``); a proposal already waiting is never
  queued twice, and one the user declined in a conversation is not queued
  again in that conversation.

Accepting a proposal claims it first, pending to accepted, so it is applied
once; then its arguments are written exactly as proposed, through the same
store calls a direct write makes, a fact with the source ``accepted:<id>``
and a note under an id drawn from the proposal's, so a write that landed can
always be found. A write that fails is put back, unless it landed; one whose
target is gone is decided and said not applied; an acceptance the process
never finished is completed by the next review, never reverted. Declining
writes nothing. The queue is one SQLite file beside the other stores,
encrypted at rest the same way (SQLCipher through ``db_encryption`` when
available), with the same rule for users as theirs: one local user in
single-user mode, the default; each user's own proposals in multi-user mode.

What this cannot see: a fact the model paraphrases from the user's own
words is not their words, so it is proposed, not written; the user accepts
it in one gesture. The cost of the gate is that gesture, never a silent
write. Words pasted into a turn count as typed until pasted text is told
apart.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

# Every new module hardcodes this; it is never overridable.
checkpoint_before_apply = True

# Guarded backend integration, as the canonical memory store does it: loaded
# alone, the queue falls back to a plain SQLite connection and to the local
# user, so its contracts run without the backend.
try:
    from .db_encryption import get_encrypted_connection
except Exception:  # loaded alone, or a backend without the encryption layer

    def get_encrypted_connection(  # type: ignore[misc]
        db_path: Any,
        *,
        check_same_thread: bool = True,
        timeout: float = 5.0,
        enforce_encryption: bool | None = None,
    ) -> sqlite3.Connection:
        return sqlite3.connect(str(db_path), check_same_thread=check_same_thread, timeout=timeout)


try:
    from .user_isolation import effective_user_id
except Exception:

    def effective_user_id(user_id: str | None, single_user_mode: bool = True) -> str:  # type: ignore[misc]
        if single_user_mode or user_id is None:
            return "local"
        return user_id


STORES = ("memory", "notes")
STATUSES = ("pending", "accepted", "declined")

# The arguments that carry words a later turn reads back, per write. For a
# new fact or note each must be endorsed for the write to go through; for a
# change, which is always proposed, their provenance is still recorded, so
# the user sees whether the new words are theirs.
_CONTENT: dict[tuple[str, str], tuple[str, ...]] = {
    ("memory", "add"): ("text",),
    ("notes", "make"): ("title", "body", "tags"),
}
_WORDS: dict[tuple[str, str], tuple[str, ...]] = {
    **_CONTENT,
    ("memory", "update"): ("text",),
    ("notes", "update"): ("title", "tags"),
}

# The identifier an update or a delete aims at. No typed word vouches for an
# identifier, so these writes are always proposed.
_TARGET: dict[tuple[str, str], str] = {
    ("memory", "update"): "fact_id",
    ("memory", "delete"): "fact_id",
    ("notes", "update"): "note_id",
    ("notes", "delete"): "note_id",
}

# A fact's category is a label from a closed set, not words: it is brought
# into the set before it is proposed, so a proposal shows what accepting it
# writes.
_CATEGORIES_FALLBACK = frozenset({"identity", "preference", "fact", "contact", "project", "goal"})
_DEFAULT_CATEGORY = "fact"

_CONFIG = Path(__file__).resolve().parent / "config" / "pending_writes.yaml"
_DEFAULTS: dict[str, Any] = {"max_per_run": 20, "stale_claim_seconds": 300}
# The whole numbers each setting may take; anything else keeps the default.
_BOUNDS: dict[str, tuple[int, int]] = {"max_per_run": (1, 1000), "stale_claim_seconds": (60, 86400)}

_FILE_NAME = "pending_writes.db"
_LIST_LIMIT = 500

_SPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# What the user typed
# ---------------------------------------------------------------------------
def fold(text: Any) -> str:
    """``text`` as the gate compares it: Unicode NFC, each run of white space one space, the ends stripped."""
    return _SPACE.sub(" ", unicodedata.normalize("NFC", str(text))).strip()


def typed_units(content: str, origin: str = "legacy", segments: Iterable = ()) -> tuple[str, ...]:
    """The units that may endorse a write: each part of the turn the user typed, whole.

    Nothing smaller is a unit. A sentence, a line, a list item, a line of
    code and even a paragraph can each take their sense from what stands
    next to them ("Things you must never do:" before "Share my location
    with Bob."), and every cut into a typed part -- at sentence ends, at
    lines, at blocks -- is a reading of the text, which an attacker who
    knows the reader can aim at. A typed part whole is the only unit no
    neighbour can turn around. A turn with segments gives its typed
    segments; a turn with none gives its whole content when its origin is
    typed, and nothing otherwise; a declaration outside the turn-origin
    grammar is held legacy and gives nothing.
    """
    from .memory import probes

    text = str(content or "")
    turn = {"turn_id": "turn", "role": "user", "text": text, "origin": origin,
            "segments": [list(s) for s in segments]}
    held_origin, held_segments, _defect = probes.read_origin(turn)
    if held_segments:
        pieces = [text[start:stop] for start, stop, label in held_segments if label == "typed"]
    else:
        pieces = [text] if held_origin == "typed" else []
    return tuple(dict.fromkeys(piece.strip() for piece in pieces if piece.strip()))


@dataclass(frozen=True)
class Endorsers:
    """The whole units the user typed in the current turn: the only words that write directly."""

    units: tuple[str, ...] = ()
    _folded: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for unit in self.units:
            self._folded.setdefault(fold(unit), unit)

    @classmethod
    def for_turn(cls, content: str, origin: str = "legacy", segments: Iterable = ()) -> Endorsers:
        """The endorsers of a turn whose words, of ``origin``, are ``content`` with ``segments``."""
        if not content:
            return cls()
        return cls(typed_units(content, origin, segments))

    @property
    def vouched(self) -> bool:
        """Whether the turn holds any typed unit at all."""
        return bool(self.units)

    def endorse(self, value: Any) -> str | None:
        """The typed unit ``value`` equals once both are folded, or None."""
        if not isinstance(value, str):
            return None
        folded = fold(value)
        return self._folded.get(folded) if folded else None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """The queue's settings from ``config/pending_writes.yaml``.

    A missing file or library keeps the defaults; a value that is not a
    whole number within its bounds is refused by name and the default kept,
    so a mistyped bound never lifts the bound.
    """
    config = dict(_DEFAULTS)
    try:
        import yaml

        raw = yaml.safe_load(Path(path or _CONFIG).read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("pending writes: configuration unreadable (%s); defaults kept", exc)
        return config
    for key, (low, high) in _BOUNDS.items():
        value = raw.get(key) if isinstance(raw, dict) else None
        if type(value) is int and low <= value <= high:
            config[key] = value
        elif value is not None:
            logger.warning("pending writes: %s %r is not a whole number from %d to %d; %d kept",
                           key, value, low, high, config[key])
    return config


# ---------------------------------------------------------------------------
# The review queue
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_db_path() -> Path:
    """The queue's file in the data directory: never a name resolved against the caller's directory."""
    from .config import DATA_DIR

    return Path(DATA_DIR).resolve() / _FILE_NAME


@dataclass(frozen=True)
class PendingWrite:
    """One proposal: a write as it would be applied, the provenance of each argument, and its decision."""

    id: str
    store: str
    action: str
    arguments: dict
    provenance: dict
    conversation_id: str = ""
    run_id: str = ""
    status: str = "pending"
    created_at: str = ""
    decided_at: str | None = None
    outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_COLUMNS = ("id", "store", "action", "arguments", "provenance", "conversation_id", "run_id", "status",
            "created_at", "decided_at", "outcome")

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS pending_writes (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        store TEXT NOT NULL CHECK (store IN ('memory', 'notes')),
        action TEXT NOT NULL,
        arguments TEXT NOT NULL,
        provenance TEXT NOT NULL,
        digest TEXT NOT NULL,
        conversation_id TEXT NOT NULL DEFAULT '',
        run_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'declined')),
        created_at TEXT NOT NULL,
        decided_at TEXT,
        outcome TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_pending_writes_status ON pending_writes (user_id, status, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_pending_writes_digest ON pending_writes (user_id, digest, status)",
)


def _digest(store: str, action: str, arguments: dict) -> str:
    payload = json.dumps([store, action, arguments], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record(row: Iterable) -> PendingWrite:
    values = dict(zip(_COLUMNS, row))
    values["arguments"] = json.loads(values["arguments"])
    values["provenance"] = json.loads(values["provenance"])
    return PendingWrite(**values)


class PendingWriteStore:
    """The review queue: proposals waiting for the user, in one SQLite file encrypted like the other stores.

    Users are told apart as the memory and notes stores tell them apart: in
    single-user mode (the default, as theirs) every caller is the one local
    user; in multi-user mode a user lists, accepts and declines only their
    own proposals.
    """

    def __init__(self, db_path: str | Path | None = None, *, single_user_mode: bool = True) -> None:
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self._single_user_mode = bool(single_user_mode)
        self._lock = threading.RLock()
        with self._connect() as conn:
            for statement in _SCHEMA:
                conn.execute(statement)

    def _uid(self, user_id: str | None) -> str:
        return effective_user_id(user_id, self._single_user_mode)

    @contextmanager
    def _connect(self):
        conn = get_encrypted_connection(self.db_path, check_same_thread=False, timeout=5.0)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _select(self, conn: Any, where: str, params: tuple) -> list[PendingWrite]:
        # ``where`` is one of the fixed clauses below; every value travels as a placeholder.
        rows = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM pending_writes WHERE {where}", params).fetchall()
        return [_record(row) for row in rows]

    def propose(self, store: str, action: str, arguments: dict, provenance: dict, *, conversation_id: str = "",
                run_id: str = "", user_id: str | None = None) -> tuple[PendingWrite, bool]:
        """Queue a proposal, unless the same write is already waiting, or was declined in this conversation.

        Returns the proposal and whether it is new: the one already waiting,
        or the one the user declined in the same conversation (its status
        says so), is returned unqueued, so a page read again or an
        extraction run again does not put a refused write back before them.
        Another conversation may propose it again: a later request of the
        user's own is never barred by an earlier refusal. A run with no
        conversation has nothing to scope a refusal to, so another run may
        propose the same write again, and the user declines it again.
        """
        if (store, action) not in _CONTENT and (store, action) not in _TARGET:
            raise ValueError(f"no write '{action}' in store '{store}'")
        uid = self._uid(user_id)
        digest = _digest(store, action, arguments)
        conversation = conversation_id or ""
        with self._lock, self._connect() as conn:
            known = self._select(
                conn, "user_id = ? AND digest = ? AND (status = 'pending' OR (status = 'declined' "
                "AND conversation_id = ? AND conversation_id != '')) "
                "ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END LIMIT 1", (uid, digest, conversation))
            if known:
                return known[0], False
            record = PendingWrite(id=uuid.uuid4().hex[:16], store=store, action=action, arguments=dict(arguments),
                                  provenance=dict(provenance), conversation_id=conversation_id or "",
                                  run_id=run_id or "", status="pending", created_at=_now())
            conn.execute(
                """INSERT INTO pending_writes (id, user_id, store, action, arguments, provenance, digest,
                   conversation_id, run_id, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (record.id, uid, store, action, json.dumps(record.arguments, ensure_ascii=False),
                 json.dumps(record.provenance, ensure_ascii=False), digest, record.conversation_id, record.run_id,
                 record.created_at),
            )
            return record, True

    def get(self, pid: str, *, user_id: str | None = None) -> PendingWrite | None:
        with self._lock, self._connect() as conn:
            found = self._select(conn, "id = ? AND user_id = ?", (str(pid), self._uid(user_id)))
        return found[0] if found else None

    def list(self, *, status: str | None = "pending", store: str | None = None, user_id: str | None = None,
             limit: int | None = _LIST_LIMIT) -> list[PendingWrite]:
        """The user's proposals, oldest first; ``status`` None gives every decision, ``limit`` None every row."""
        where, params = "user_id = ?", [self._uid(user_id)]
        if status is not None:
            where, params = where + " AND status = ?", params + [status]
        if store is not None:
            where, params = where + " AND store = ?", params + [store]
        where += " ORDER BY created_at ASC, rowid ASC"
        if limit is not None:
            where, params = where + " LIMIT ?", params + [max(1, int(limit))]
        with self._lock, self._connect() as conn:
            return self._select(conn, where, tuple(params))

    def delete_user(self, user_id: str | None) -> int:
        """Delete every proposal of a user, whatever its decision (the per-user wipe); how many."""
        with self._lock, self._connect() as conn:
            return conn.execute("DELETE FROM pending_writes WHERE user_id = ?", (self._uid(user_id),)).rowcount

    def claim(self, pid: str, *, user_id: str | None = None) -> PendingWrite | None:
        """Mark a pending proposal accepted, once; the proposal, or None when it was not pending."""
        uid = self._uid(user_id)
        with self._lock, self._connect() as conn:
            moved = conn.execute(
                "UPDATE pending_writes SET status = 'accepted', decided_at = ? "
                "WHERE id = ? AND user_id = ? AND status = 'pending'", (_now(), str(pid), uid)).rowcount
            if moved != 1:
                return None
            return self._select(conn, "id = ? AND user_id = ?", (str(pid), uid))[0]

    def settle(self, pid: str, outcome: str, *, user_id: str | None = None) -> bool:
        """Record what applying an accepted proposal did, once; whether a claimed proposal took it."""
        with self._lock, self._connect() as conn:
            return conn.execute(
                "UPDATE pending_writes SET outcome = ? "
                "WHERE id = ? AND user_id = ? AND status = 'accepted' AND outcome IS NULL",
                (str(outcome), str(pid), self._uid(user_id))).rowcount == 1

    def retake(self, pid: str, claimed_at: str, *, user_id: str | None = None) -> bool:
        """Take up an unfinished acceptance claimed at ``claimed_at``, once; whether this caller has it.

        The claim is stamped anew only while it is still the one that was
        read, so two reviews that read it together never both complete it.
        """
        with self._lock, self._connect() as conn:
            return conn.execute(
                "UPDATE pending_writes SET decided_at = ? WHERE id = ? AND user_id = ? AND status = 'accepted' "
                "AND outcome IS NULL AND decided_at = ?",
                (_now(), str(pid), self._uid(user_id), str(claimed_at))).rowcount == 1

    def unfinished(self, older_than: str, *, user_id: str | None = None) -> list[PendingWrite]:
        """The acceptances claimed before ``older_than`` (ISO time) that never recorded an outcome."""
        with self._lock, self._connect() as conn:
            return self._select(conn, "user_id = ? AND status = 'accepted' AND outcome IS NULL AND decided_at < ? "
                                "ORDER BY decided_at ASC, rowid ASC", (self._uid(user_id), older_than))

    def release(self, pid: str, *, user_id: str | None = None) -> None:
        """Put back a claimed proposal whose write failed, so it can be accepted again."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE pending_writes SET status = 'pending', decided_at = NULL "
                "WHERE id = ? AND user_id = ? AND status = 'accepted' AND outcome IS NULL",
                (str(pid), self._uid(user_id)))

    def decline(self, pid: str, *, user_id: str | None = None) -> bool:
        """Mark a pending proposal declined; whether it was pending."""
        with self._lock, self._connect() as conn:
            return conn.execute(
                "UPDATE pending_writes SET status = 'declined', decided_at = ? "
                "WHERE id = ? AND user_id = ? AND status = 'pending'",
                (_now(), str(pid), self._uid(user_id))).rowcount == 1


_QUEUE: PendingWriteStore | None = None
_QUEUE_LOCK = threading.Lock()


def get_pending_store() -> PendingWriteStore:
    """The process's review queue, opened on first use in the data directory."""
    global _QUEUE
    with _QUEUE_LOCK:
        if _QUEUE is None:
            _QUEUE = PendingWriteStore()
        return _QUEUE


def set_pending_store(store: Any) -> None:
    """Use ``store`` as the process's review queue (an embedding host, or a test's temporary file)."""
    global _QUEUE
    with _QUEUE_LOCK:
        _QUEUE = store


def reset_pending_store() -> None:
    """Forget the process's review queue; the next use opens it again."""
    set_pending_store(None)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def memory_category(value: Any, otherwise: str | None = _DEFAULT_CATEGORY) -> str | None:
    """A fact's category brought into the closed set, ``otherwise`` when it lies outside; None stays None.

    A new fact outside the set takes the default, as the store files it; an
    update outside the set keeps the fact's category (``otherwise`` None),
    where the store would refuse it.
    """
    if value is None:
        return None
    try:
        from .memory.canonical_store import CATEGORIES
    except Exception:
        CATEGORIES = _CATEGORIES_FALLBACK
    category = str(value).strip().lower()
    return category if category in CATEGORIES else otherwise


def _default_memory_store() -> Any:
    from .memory import get_memory_store

    return get_memory_store()


def _default_notes_store() -> Any:
    from .notes import get_notes_store

    return get_notes_store()


def _fact_line(record: Any) -> str:
    return f"[{getattr(record, 'id', '')}] ({getattr(record, 'category', '')}) {getattr(record, 'text', '')}"


def _note_line(record: Any) -> str:
    pin = "*" if getattr(record, "pinned", False) else " "
    return (f"[{getattr(record, 'id', '')}] {pin}{getattr(record, 'title', '')}  tags={getattr(record, 'tags', '')}"
            f"  updated={getattr(record, 'updated_at', '')}")


class TargetMissing(LookupError):
    """An update or a delete whose fact or note no longer exists: nothing to apply."""


def accepted_note_id(pid: str) -> str:
    """The id a note created by accepting proposal ``pid`` takes: the same for every attempt, so it can be found."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"opti-oignon:pending-write:{pid}").hex


def apply_write(store: str, action: str, arguments: dict, *, source: str, memory_store: Any = None,
                notes_store: Any = None, new_note_id: str | None = None) -> str:
    """Write ``arguments`` through the store's own calls; what was done, as the tools always said it.

    The one write path: a direct write of the agent and an accepted proposal
    both come through here. Raises when the store is unavailable or refuses,
    and ``TargetMissing`` when an update or a delete finds nothing to change.
    """
    if store == "memory":
        target = memory_store if memory_store is not None else _default_memory_store()
        if target is None:
            raise RuntimeError("memory store unavailable")
        fact_id = str(arguments.get("fact_id", ""))
        if action == "add":
            record, decision = target.add(arguments["text"], arguments.get("category") or _DEFAULT_CATEGORY,
                                          source=source)
            if getattr(decision, "action", "") == "merge":
                return f"Memory merged into existing fact {_fact_line(record)}."
            return f"Memory added {_fact_line(record)}."
        if action == "update":
            record = target.update(fact_id, text=arguments.get("text"), category=arguments.get("category"))
            if record is None:
                raise TargetMissing(f"No memory with id '{fact_id}' to update.")
            return f"Memory updated {_fact_line(record)}."
        if action == "delete":
            # Soft delete only: the row is retained for restore via the panel.
            if not target.soft_delete(fact_id):
                raise TargetMissing(f"No memory with id '{fact_id}'.")
            return f"Memory '{fact_id}' archived."
    elif store == "notes":
        target = notes_store if notes_store is not None else _default_notes_store()
        if target is None:
            raise RuntimeError("notes store unavailable")
        note_id = str(arguments.get("note_id", ""))
        if action == "make":
            extra = {"note_id": new_note_id} if new_note_id else {}
            record = target.add_note(arguments["title"], body_crdt=str(arguments.get("body") or "").encode("utf-8"),
                                     tags=arguments.get("tags"), pinned=bool(arguments.get("pinned", False)), **extra)
            return f"Note created [{record.id}] {record.title}."
        if action == "update":
            fields = {key: value for key, value in arguments.items() if key != "note_id"}
            record = target.update_note(note_id, **fields)
            if record is None:
                raise TargetMissing(f"No note with id '{note_id}' to update.")
            return f"Note updated {_note_line(record)}."
        if action == "delete":
            # Soft delete only: a tombstone, so the deletion syncs.
            if not target.delete_note(note_id):
                raise TargetMissing(f"No note with id '{note_id}'.")
            return f"Note '{note_id}' deleted."
    raise ValueError(f"no write '{action}' in store '{store}'")


def _landed(record: PendingWrite, memory_store: Any = None, notes_store: Any = None) -> bool | None:
    """Whether an accepted proposal's write is in the store: True, False, or None when it cannot be told.

    A new fact carries the source ``accepted:<id>`` and a new note the id
    drawn from the proposal's, both unique to it; a change is there when the
    store shows it. A lookup that fails says nothing (None), never "absent".
    """
    arguments = record.arguments
    try:
        if record.store == "memory":
            target = memory_store if memory_store is not None else _default_memory_store()
            if record.action == "add":
                source = f"accepted:{record.id}"
                return any(getattr(fact, "source", "") == source for fact in target.list(active_only=False))
            fact = target.get(str(arguments.get("fact_id", "")))
            if record.action == "delete":
                # A fact absent altogether may never have existed: that cannot be told from here.
                return None if fact is None else not getattr(fact, "active", True)
            if fact is None:
                return False
            wanted = {k: arguments[k] for k in ("text", "category") if arguments.get(k) is not None}
            return all(getattr(fact, key, None) == value for key, value in wanted.items())
        target = notes_store if notes_store is not None else _default_notes_store()
        if record.action == "make":
            return target.get_note(accepted_note_id(record.id)) is not None
        note = target.get_note(str(arguments.get("note_id", "")))
        if record.action == "delete":
            return None if note is None else bool(getattr(note, "deleted", False))
        if note is None:
            return False
        wanted = {k: v for k, v in arguments.items() if k != "note_id" and v is not None}
        return all(getattr(note, key, None) == value for key, value in wanted.items())
    except Exception:
        return None


def _content_values(name: str, value: Any) -> list[str]:
    """The words an argument carries: a note's tags one by one, an empty body none."""
    if value is None:
        return []
    if name == "tags":
        try:
            items = json.loads(value) if isinstance(value, str) else list(value)
        except (TypeError, ValueError):
            return [str(value)]
        return [str(item) for item in items]
    return [value] if str(value) else []


class WriteGate:
    """Decides, for one turn, which persistent writes go through and which wait for the user."""

    def __init__(self, endorsers: Endorsers | None = None, *, pending: Any = None, conversation_id: str = "",
                 run_id: str = "", read: Callable[[], Iterable[str]] | None = None, max_per_run: int | None = None,
                 user_id: str | None = None) -> None:
        self.endorsers = endorsers if endorsers is not None else Endorsers()
        self.conversation_id = conversation_id or ""
        self.run_id = run_id or ""
        self.user_id = user_id
        self.max_per_run = int(max_per_run) if max_per_run is not None else load_config()["max_per_run"]
        self._pending = pending
        self._read = read
        self._proposed = 0
        self._count_lock = threading.Lock()

    def _provenance(self, key: tuple[str, str], typed: dict, untyped: list) -> dict[str, Any]:
        try:
            read = list(dict.fromkeys(str(name) for name in (self._read() if self._read else ())))
        except Exception:
            read = []
        return {"source": "agent", "turn": "typed" if self.endorsers.vouched else "none", "typed": typed,
                "untyped": untyped, "target": _TARGET.get(key), "read": read}

    def write(self, store: str, action: str, arguments: dict, *, target: Any = None) -> str:
        """Write, or propose, ``arguments``; what the model is told."""
        key = (store, action)
        arguments = dict(arguments)
        if store == "memory" and "category" in arguments:
            arguments["category"] = memory_category(arguments["category"],
                                                    None if action == "update" else _DEFAULT_CATEGORY)
        typed: dict[str, Any] = {}
        untyped: list[str] = []
        for name in _WORDS.get(key, ()):
            values = _content_values(name, arguments.get(name))
            units = [self.endorsers.endorse(value) for value in values]
            if any(unit is None for unit in units):
                untyped.append(name)
            elif units:
                typed[name] = units if name == "tags" else units[0]
        if key in _CONTENT and not untyped:
            return apply_write(store, action, arguments, source="agent",
                               **{"memory_store" if store == "memory" else "notes_store": target})
        with self._count_lock:
            if self._proposed >= self.max_per_run:
                return (f"Not proposed: this run already made {self.max_per_run} proposals, the most one run may "
                        "make; nothing was saved.")
            try:
                queue = self._pending if self._pending is not None else get_pending_store()
                record, new = queue.propose(store, action, arguments, self._provenance(key, typed, untyped),
                                            conversation_id=self.conversation_id, run_id=self.run_id,
                                            user_id=self.user_id)
            except Exception as exc:
                logger.warning("pending write not recorded: %s", exc)
                return "Not saved: the review queue is unavailable, so nothing was written or proposed."
            if new:
                self._proposed += 1
        if record.status == "declined":
            return "Not proposed: an identical proposal was declined earlier in this conversation; nothing was saved."
        return f"Proposed to the user for review ({record.id}): not saved unless the user accepts it."


def propose_facts(facts: Iterable[Any], *, conversation_id: str = "", known: Iterable[str] = (),
                  pending: Any = None, user_id: str | None = None) -> int:
    """Propose extracted facts no typed word endorsed, skipping any already written; how many were queued."""
    queue = pending if pending is not None else get_pending_store()
    seen = {fold(text) for text in known}
    count = 0
    for fact in facts:
        text = str(getattr(fact, "text", "") or "").strip()
        if not text or fold(text) in seen:
            continue
        seen.add(fold(text))
        provenance = {"source": "extraction", "turn": "none", "typed": {}, "untyped": ["text"], "target": None,
                      "read": []}
        _record, new = queue.propose("memory", "add",
                                     {"text": text, "category": memory_category(getattr(fact, "category", None))
                                      or _DEFAULT_CATEGORY},
                                     provenance, conversation_id=conversation_id, user_id=user_id)
        count += 1 if new else 0
    return count


def rest_turns(messages: Iterable[Any] | None) -> list[dict[str, str]]:
    """What the user did not type in ``messages``, as turns for an extraction whose facts are only proposed.

    The complement of the capture's typed words: a document's segment, a
    turn the model reworded, a turn of no known origin, an assistant reply,
    and the whole of a turn whose origin or segments cannot be read -- held
    for review, never written. The words the executor writes between parts
    belong to no one and are left out.
    """
    turns = []
    for message in messages or ():
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if message.get("role") != "user":
            turns.append({"role": "assistant", "content": content})
            continue
        text = "\n".join(part.strip() for part in _untyped_parts(content, message.get("origin"),
                                                                 message.get("segments")) if part.strip())
        if text:
            turns.append({"role": "user", "content": text})
    return turns


def _untyped_parts(content: str, origin: Any, segments: Any) -> list[str]:
    if not isinstance(origin, str) or not isinstance(segments, (list, tuple)):
        return [content]
    if not segments:
        return [] if origin == "typed" else [content]
    parts, end = [], 0
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 3:
            return [content]
        start, stop, label = segment
        if type(start) is not int or type(stop) is not int or not end <= start < stop <= len(content):
            return [content]
        end = stop
        if label != "typed":
            parts.append(content[start:stop])
    return parts


def _apply(record: PendingWrite, memory_store: Any, notes_store: Any) -> str:
    """Apply an accepted proposal's write, under its source and its note id."""
    return apply_write(record.store, record.action, record.arguments, source=f"accepted:{record.id}",
                       memory_store=memory_store, notes_store=notes_store, new_note_id=accepted_note_id(record.id))


def _settle(queue: Any, pid: str, outcome: str, user_id: str | None) -> None:
    """Record an outcome; a failure to record never undoes the write, the next review completes it."""
    try:
        if not queue.settle(pid, outcome, user_id=user_id):
            logger.warning("pending write %s: outcome not recorded (no longer claimed)", pid)
    except Exception as exc:
        logger.warning("pending write %s: outcome not recorded (%s); the next review completes it", pid, exc)


def recover(*, pending: Any = None, memory_store: Any = None, notes_store: Any = None,
            user_id: str | None = None) -> list[dict[str, Any]]:
    """Complete the acceptances a process never finished; one result per proposal it took up.

    An acceptance claimed longer than ``stale_claim_seconds`` ago without an
    outcome was cut short. It is completed, never put back: the user
    accepted it. A write that landed is settled as it stands; one that did
    not is applied now, under the same source and note id, so a write that
    was in fact still in flight elsewhere is found or refused by the store,
    never doubled. One that cannot be told or fails is left for the next
    review.
    """
    queue = pending if pending is not None else get_pending_store()
    window = load_config()["stale_claim_seconds"]
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=window)).isoformat()
    results = []
    for record in queue.unfinished(cutoff, user_id=user_id):
        if not queue.retake(record.id, record.decided_at, user_id=user_id):
            continue  # another review took it up first
        landed = _landed(record, memory_store, notes_store)
        applied = True
        try:
            outcome = ("Completed after an interruption: the write had landed." if landed
                       else f"Completed after an interruption: {_apply(record, memory_store, notes_store)}")
        except TargetMissing as gone:
            outcome, applied = str(gone), False
        except Exception as exc:
            logger.warning("pending write %s: interrupted acceptance not completed yet (%s)", record.id, exc)
            continue
        _settle(queue, record.id, outcome, user_id)
        result = {"id": record.id, "applied": applied, "outcome": outcome}
        results.append(result if applied else dict(result, reason="target not found"))
    return results


def accept(ids: Iterable[str], *, pending: Any = None, memory_store: Any = None, notes_store: Any = None,
           user_id: str | None = None) -> list[dict[str, Any]]:
    """Apply each proposal of ``ids``, in order, once and exactly as proposed; one result per id.

    Acceptances an earlier process left unfinished are completed first.
    """
    queue = pending if pending is not None else get_pending_store()
    try:
        recover(pending=queue, memory_store=memory_store, notes_store=notes_store, user_id=user_id)
    except Exception as exc:
        logger.warning("pending writes: completing unfinished acceptances failed (%s)", exc)
    results = []
    for pid in ids:
        pid = str(pid)
        if queue.get(pid, user_id=user_id) is None:
            results.append({"id": pid, "applied": False, "reason": "not found"})
            continue
        claimed = queue.claim(pid, user_id=user_id)
        if claimed is None:
            results.append({"id": pid, "applied": False, "reason": "not pending"})
            continue
        try:
            outcome = _apply(claimed, memory_store, notes_store)
        except TargetMissing as gone:
            # Nothing left to change: the proposal is decided, and said not applied.
            _settle(queue, pid, str(gone), user_id)
            results.append({"id": pid, "applied": False, "reason": "target not found", "outcome": str(gone)})
            continue
        except Exception as exc:
            landed = _landed(claimed, memory_store, notes_store)
            if landed:
                # The write is in the store; what failed came after. Decided,
                # so it can no longer be declined while it stays written.
                outcome = f"Saved, but the store then failed: {exc}"
                _settle(queue, pid, outcome, user_id)
                results.append({"id": pid, "applied": True, "outcome": outcome})
            elif landed is False:
                queue.release(pid, user_id=user_id)
                results.append({"id": pid, "applied": False, "reason": f"failed: {exc}"})
            else:
                # Whether it landed cannot be told: left claimed, the next
                # review completes it rather than letting it be declined.
                results.append({"id": pid, "applied": False, "reason": f"failed: {exc}; the next review completes it"})
            continue
        _settle(queue, pid, outcome, user_id)
        results.append({"id": pid, "applied": True, "outcome": outcome})
    return results


def decline(ids: Iterable[str], *, pending: Any = None, user_id: str | None = None) -> list[dict[str, Any]]:
    """Decline each proposal of ``ids``; nothing is written. One result per id."""
    queue = pending if pending is not None else get_pending_store()
    results = []
    for pid in ids:
        pid = str(pid)
        if queue.get(pid, user_id=user_id) is None:
            results.append({"id": pid, "declined": False, "reason": "not found"})
        elif queue.decline(pid, user_id=user_id):
            results.append({"id": pid, "declined": True})
        else:
            results.append({"id": pid, "declined": False, "reason": "not pending"})
    return results
