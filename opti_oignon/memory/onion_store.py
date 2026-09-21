#!/usr/bin/env python3
"""The onion's table: one conversation's Core, Cellar, receipts, Peels and
Flesh, on disk, surviving the process that grew them.

The four stores prove themselves in memory by re-hashing: an entry answers
to its id, a span to its key, a peel to its text and sources. Persistence
keeps that property rather than trusting the file: a load rebuilds every
store through the same surface the librarian uses, re-hashes every row
against the id it was saved under, recomputes the root the whole state was
saved under, and refuses by name the first thing that no longer answers.
A conversation the file does not know is ``None``, never an empty state:
an empty state would be indistinguishable from a new conversation, and a
lost memory must not read as a fresh one.

Connections go through the repository's ``safe_connect``, and the store
asks the connection whether it is encrypted before it writes a byte. The
rest of the tree opens plaintext with a warning outside Bulbe mode; this
table holds conversation text, so here plaintext is a refusal by name
unless ``onion.yaml`` says otherwise. A store built where the seam is
unreachable refuses too, and creates no file.

The root over the four stores is a pure function of their canonical text,
recomputed at every save and checked at every load. The drift ledger has
its own table and its own root is a decision for another block.
"""

import hashlib
import json
import logging
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

checkpoint_before_apply = True

logger = logging.getLogger(__name__)

TABLES = ("onion_core", "onion_cellar", "onion_receipts", "onion_peels", "onion_flesh", "onion_cursor")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS onion_core (
    conversation TEXT NOT NULL,
    seq INTEGER NOT NULL,
    id TEXT NOT NULL,
    text TEXT NOT NULL,
    superseded_by TEXT,
    PRIMARY KEY (conversation, id)
);
CREATE TABLE IF NOT EXISTS onion_cellar (
    conversation TEXT NOT NULL,
    seq INTEGER NOT NULL,
    key TEXT NOT NULL,
    span TEXT NOT NULL,
    PRIMARY KEY (conversation, key)
);
CREATE TABLE IF NOT EXISTS onion_receipts (
    conversation TEXT NOT NULL,
    seq INTEGER NOT NULL,
    key TEXT NOT NULL,
    stub TEXT NOT NULL,
    turn_ids TEXT NOT NULL,
    resolved INTEGER NOT NULL,
    PRIMARY KEY (conversation, seq)
);
CREATE TABLE IF NOT EXISTS onion_peels (
    conversation TEXT NOT NULL,
    seq INTEGER NOT NULL,
    id TEXT NOT NULL,
    text TEXT NOT NULL,
    level INTEGER NOT NULL,
    sources TEXT NOT NULL,
    children TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    probes_passed INTEGER NOT NULL,
    probes_total INTEGER NOT NULL,
    PRIMARY KEY (conversation, id)
);
CREATE TABLE IF NOT EXISTS onion_flesh (
    conversation TEXT NOT NULL,
    seq INTEGER NOT NULL,
    turn TEXT NOT NULL,
    PRIMARY KEY (conversation, seq)
);
CREATE TABLE IF NOT EXISTS onion_cursor (
    conversation TEXT PRIMARY KEY,
    seen INTEGER NOT NULL,
    root TEXT NOT NULL,
    saved_at TEXT NOT NULL
);
"""


class OnionStoreError(ValueError):
    """The store cannot answer for what it holds."""


class PlaintextRefused(OnionStoreError):
    """The connection is not encrypted and the configuration did not allow that."""


class OnionIntegrityError(OnionStoreError):
    """A saved row no longer answers to the id it was saved under."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Snapshot:
    """One conversation's onion as rows: what is saved and what a load rebuilds."""

    core: tuple      # (id, text, superseded_by) in insertion order
    cellar: tuple    # (key, span) in insertion order
    receipts: tuple  # (key, stub, turn_ids, resolved) in ledger order
    peels: tuple     # (id, text, level, sources, children, source_digest, passed, total) in order
    flesh: tuple     # turns, oldest first
    seen: int


def onion_root(snapshot):
    """The root of the four stores: SHA-256 over their canonical rows, in order.

    A pure function of what is saved, so two processes holding the same
    state compute the same root and one holding a moved byte does not.
    The Flesh and the cursor are not in it: they are the part that moves
    every turn, and the root anchors what the model is told was kept.
    """
    payload = _canonical({
        "core": [list(row) for row in snapshot.core],
        "cellar": [[key, span] for key, span in snapshot.cellar],
        "receipts": [[key, stub, list(ids), bool(resolved)] for key, stub, ids, resolved in snapshot.receipts],
        "peels": [list(row) for row in snapshot.peels],
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_of(state):
    """The rows of a librarian state, read through each store's public surface."""
    core = tuple((e.id, e.text, e.superseded_by) for e in state.core.all())
    cellar = tuple((key, state.cellar.get(key)) for key in state.cellar.keys())
    receipts = tuple((r.key, r.stub, tuple(r.turn_ids), bool(r.resolved)) for r in state.ledger.all())
    peels = tuple(
        (p.id, p.text, int(p.level), tuple(p.sources), tuple(p.children), p.source_digest,
         int(p.probes_passed), int(p.probes_total))
        for p in state.tree.all()
    )
    return Snapshot(core=core, cellar=cellar, receipts=receipts, peels=peels,
                    flesh=tuple(state.flesh.turns()), seen=int(state.seen))


def _is_encrypted(conn):
    """True when the connection answers to a cipher; a plain client answers nothing."""
    try:
        return bool(conn.execute("PRAGMA cipher_version").fetchall())
    except Exception:  # noqa: BLE001 - a client without the pragma is a plain one
        return False


class OnionStore:
    """The onion's table for every conversation, on one SQLite file."""

    def __init__(self, path, *, connect=None, require_encryption=True):
        self._path = Path(path)
        if connect is None:
            from ..db_utils import safe_connect

            connect = safe_connect
        self._connect = connect
        self._require_encryption = bool(require_encryption)
        self._lock = threading.Lock()
        self._init_db()

    @property
    def path(self):
        return self._path

    def _conn(self):
        conn = self._connect(self._path)
        if self._require_encryption and not _is_encrypted(conn):
            conn.close()
            try:
                if self._path.exists() and self._path.stat().st_size == 0:
                    self._path.unlink()
            except OSError:
                pass
            raise PlaintextRefused(
                f"the onion store at {self._path} would be written in plaintext; refused "
                f"(set persistence.require_encryption to false in onion.yaml to allow it)"
            )
        return closing(conn)

    def _init_db(self):
        with self._lock, self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # -- save --------------------------------------------------------------

    def save(self, conversation_id, state):
        """Replace the conversation's rows with the state's, under one transaction."""
        cid = str(conversation_id)
        if not cid:
            raise OnionStoreError("a conversation id is required to save an onion state")
        snapshot = snapshot_of(state)
        root = onion_root(snapshot)
        with self._lock, self._conn() as conn:
            for table in TABLES:
                conn.execute(f"DELETE FROM {table} WHERE conversation = ?", (cid,))
            conn.executemany(
                "INSERT INTO onion_core (conversation, seq, id, text, superseded_by) VALUES (?, ?, ?, ?, ?)",
                [(cid, i, e_id, text, sup) for i, (e_id, text, sup) in enumerate(snapshot.core)],
            )
            conn.executemany(
                "INSERT INTO onion_cellar (conversation, seq, key, span) VALUES (?, ?, ?, ?)",
                [(cid, i, key, _canonical(span)) for i, (key, span) in enumerate(snapshot.cellar)],
            )
            conn.executemany(
                "INSERT INTO onion_receipts (conversation, seq, key, stub, turn_ids, resolved) VALUES (?, ?, ?, ?, ?, ?)",
                [(cid, i, key, stub, _canonical(list(ids)), 1 if resolved else 0)
                 for i, (key, stub, ids, resolved) in enumerate(snapshot.receipts)],
            )
            conn.executemany(
                "INSERT INTO onion_peels (conversation, seq, id, text, level, sources, children, source_digest, "
                "probes_passed, probes_total) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(cid, i, p_id, text, level, _canonical(list(sources)), _canonical(list(children)), digest, passed, total)
                 for i, (p_id, text, level, sources, children, digest, passed, total) in enumerate(snapshot.peels)],
            )
            conn.executemany(
                "INSERT INTO onion_flesh (conversation, seq, turn) VALUES (?, ?, ?)",
                [(cid, i, _canonical(turn)) for i, turn in enumerate(snapshot.flesh)],
            )
            conn.execute(
                "INSERT INTO onion_cursor (conversation, seen, root, saved_at) VALUES (?, ?, ?, ?)",
                (cid, snapshot.seen, root, _now()),
            )
            conn.commit()
        return root

    # -- load --------------------------------------------------------------

    def load_snapshot(self, conversation_id):
        """The saved rows of a conversation, or None when the file does not know it."""
        cid = str(conversation_id)
        with self._lock, self._conn() as conn:
            cursor = conn.execute("SELECT seen, root FROM onion_cursor WHERE conversation = ?", (cid,)).fetchone()
            if cursor is None:
                return None, None
            core = conn.execute(
                "SELECT id, text, superseded_by FROM onion_core WHERE conversation = ? ORDER BY seq", (cid,)
            ).fetchall()
            cellar = conn.execute(
                "SELECT key, span FROM onion_cellar WHERE conversation = ? ORDER BY seq", (cid,)
            ).fetchall()
            receipts = conn.execute(
                "SELECT key, stub, turn_ids, resolved FROM onion_receipts WHERE conversation = ? ORDER BY seq", (cid,)
            ).fetchall()
            peels = conn.execute(
                "SELECT id, text, level, sources, children, source_digest, probes_passed, probes_total "
                "FROM onion_peels WHERE conversation = ? ORDER BY seq", (cid,)
            ).fetchall()
            flesh = conn.execute(
                "SELECT turn FROM onion_flesh WHERE conversation = ? ORDER BY seq", (cid,)
            ).fetchall()
        snapshot = Snapshot(
            core=tuple((r[0], r[1], r[2]) for r in core),
            cellar=tuple((r[0], json.loads(r[1])) for r in cellar),
            receipts=tuple((r[0], r[1], tuple(json.loads(r[2])), bool(r[3])) for r in receipts),
            peels=tuple((r[0], r[1], int(r[2]), tuple(json.loads(r[3])), tuple(json.loads(r[4])), r[5], int(r[6]), int(r[7]))
                        for r in peels),
            flesh=tuple(json.loads(r[0]) for r in flesh),
            seen=int(cursor[0]),
        )
        return snapshot, cursor[1]

    def load(self, conversation_id, state):
        """Rebuild ``state`` (an empty librarian state) from the file, proving every row.

        Returns the state, or None when the file does not know the
        conversation. Every Core entry, Cellar span and Peel is re-hashed
        against its saved id, and the root recomputed against the saved
        one; the first that does not answer is refused by name.
        """
        cid = str(conversation_id)
        snapshot, saved_root = self.load_snapshot(cid)
        if snapshot is None:
            return None
        root = onion_root(snapshot)
        if root != saved_root:
            raise OnionIntegrityError(
                f"onion state {cid}: the saved root {saved_root} does not answer to its rows ({root}); refused, not repaired"
            )
        _rebuild(state, snapshot)
        return state

    def conversations(self):
        with self._lock, self._conn() as conn:
            return [r[0] for r in conn.execute("SELECT conversation FROM onion_cursor ORDER BY conversation").fetchall()]


def _rebuild(state, snapshot):
    """Fill an empty librarian state from a snapshot through each store's surface, re-hashing as it goes."""
    from .core_store import USER, entry_hash
    from .peels import Peel
    from .receipts import Receipt, span_key

    for entry_id, text, _sup in snapshot.core:
        if entry_hash(text) != entry_id:
            raise OnionIntegrityError(f"Core entry {entry_id} no longer answers to its bytes: refused, not repaired")
        state.core.add(text, actor=USER)
    by_id = {entry_id: text for entry_id, text, _sup in snapshot.core}
    for entry_id, _text, sup in snapshot.core:
        if sup:
            if sup not in by_id:
                raise OnionIntegrityError(f"Core entry {entry_id} is superseded by {sup}, which the file does not hold")
            state.core.supersede(entry_id, by_id[sup], actor=USER)
    state.core.verify()

    for key, span in snapshot.cellar:
        if span_key(span) != key:
            raise OnionIntegrityError(f"Cellar span {key} no longer answers to its bytes: refused, not repaired")
        state.cellar.store(span)

    for key, stub, ids, resolved in snapshot.receipts:
        state.ledger.append(Receipt(key=key, stub=stub, turn_ids=tuple(ids), resolved=bool(resolved)))

    for p_id, text, level, sources, children, digest, passed, total in snapshot.peels:
        state.tree.add(Peel(id=p_id, text=text, level=int(level), sources=tuple(sources), children=tuple(children),
                            source_digest=digest, probes_passed=int(passed), probes_total=int(total)))
    state.tree.verify(state.cellar)
    state.ledger.digest(state.cellar)

    for turn in snapshot.flesh:
        state.flesh.append(turn)
    state.seen = int(snapshot.seen)
    return state
