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

What the queue adds to a receipt -- its kind when no accepted peel stands
for its span, the anchors it keeps in the Cellar -- lives in a table of
marks beside the receipts, and enters the root only when a mark exists: a
state with none has the root it always had, so a file written before marks
existed still answers to its rows. A peel's references to the user's words
and the sentences its repair dropped live in its mark the same way, and
enter the root only for a peel that has some; each reference is read again
in the Cellar when the file is loaded, and a peel whose bytes no longer
answer to it is refused by name. The lineage of each mirrored turn lives in
a table of its own, in the root once a state holds one: a row moved in the
file is refused by name, and one read outside the grammar reads cut, never
empty.
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

TABLES = (
    "onion_core", "onion_cellar", "onion_receipts", "onion_peels", "onion_flesh", "onion_cursor",
    "onion_receipt_marks", "onion_peel_marks", "onion_proposals", "onion_refusals", "onion_lineage",
)
_PROPOSAL_STATUSES = ("open", "accepted", "declined", "deferred", "superseded")
# The rungs a peel can be made on; a mark naming another is refused on load.
_PEEL_RUNGS = ("accepted", "reasked", "repaired")

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
CREATE TABLE IF NOT EXISTS onion_receipt_marks (
    conversation TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    anchors TEXT NOT NULL,
    PRIMARY KEY (conversation, seq)
);
CREATE TABLE IF NOT EXISTS onion_peel_marks (
    conversation TEXT NOT NULL,
    id TEXT NOT NULL,
    rung TEXT NOT NULL,
    stitched TEXT NOT NULL,
    residual TEXT NOT NULL,
    refs TEXT NOT NULL DEFAULT '[]',
    dropped TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (conversation, id)
);
CREATE TABLE IF NOT EXISTS onion_lineage (
    conversation TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    lineage TEXT NOT NULL,
    PRIMARY KEY (conversation, turn_id)
);
CREATE TABLE IF NOT EXISTS onion_proposals (
    conversation TEXT NOT NULL,
    seq INTEGER NOT NULL,
    id TEXT NOT NULL,
    span_key TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    start INTEGER NOT NULL,
    stop INTEGER NOT NULL,
    origin TEXT NOT NULL,
    made_on TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (conversation, seq)
);
CREATE TABLE IF NOT EXISTS onion_refusals (
    conversation TEXT NOT NULL,
    span_key TEXT NOT NULL,
    mark TEXT NOT NULL,
    PRIMARY KEY (conversation, span_key)
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
    # (seq, kind, anchors) for each receipt that is not a plain accepted
    # one, by its place in the ledger; anchors as (turn_id, start, stop).
    receipt_marks: tuple = ()
    # (id, rung, stitched, residual, refs, dropped) for each peel the queue
    # marked: one made on another rung than the summary's, with runs copied
    # in (before references), with probes it still fails, with references
    # to the user's words or with sentences its repair dropped.
    peel_marks: tuple = ()
    # (id, span_key, turn_id, start, stop, origin, made_on, status) for each
    # proposal to the Core, in the order made. Not in the root: a proposal
    # tells the model nothing, and its words are shown before it is taken.
    proposals: tuple = ()
    # (span_key, mark) for each span whose second summary was refused, by
    # key. Not in the root: a mark saves a call, it tells the model nothing.
    refusals: tuple = ()
    # (turn_id, lineage) for each turn of the Cellar or the Flesh whose
    # lineage was recorded. In the root when there is one, so a lineage moved
    # in the file -- in the grammar or not -- is refused by name: a withdrawal
    # finds what it reached by these rows, and none of them may change unseen.
    lineage: tuple = ()


def onion_root(snapshot):
    """The root of the four stores: SHA-256 over their canonical rows, in order.

    A pure function of what is saved, so two processes holding the same
    state compute the same root and one holding a moved byte does not.
    The Flesh and the cursor are not in it: they are the part that moves
    every turn, and the root anchors what the model is told was kept. The
    receipts' marks are in it when there is one, and absent otherwise, so
    a state without marks keeps the root it had before marks existed.
    """
    rows = {
        "core": [list(row) for row in snapshot.core],
        "cellar": [[key, span] for key, span in snapshot.cellar],
        "receipts": [[key, stub, list(ids), bool(resolved)] for key, stub, ids, resolved in snapshot.receipts],
        "peels": [list(row) for row in snapshot.peels],
    }
    if snapshot.receipt_marks:
        rows["receipt_marks"] = [
            [int(seq), kind, [list(anchor) for anchor in anchors]] for seq, kind, anchors in snapshot.receipt_marks
        ]
    if snapshot.peel_marks:
        rows["peel_marks"] = [_mark_row(mark) for mark in snapshot.peel_marks]
    if snapshot.lineage:
        rows["lineage"] = [[turn_id, list(entries)] for turn_id, entries in snapshot.lineage]
    return hashlib.sha256(_canonical(rows).encode("utf-8")).hexdigest()


def _mark_parts(mark):
    """A peel mark's six parts; a mark of four, written before references, has none and drops none."""
    p_id, rung, stitched, residual, *more = mark
    return p_id, rung, stitched, residual, (more[0] if more else ()), (more[1] if len(more) > 1 else ())


def _mark_row(mark):
    """A peel mark as the root reads it: its references and dropped sentences only when it has some, so a state
    with none keeps the root it had before they existed."""
    p_id, rung, stitched, residual, refs, dropped = _mark_parts(mark)
    row = [p_id, rung, [list(unit) for unit in stitched], [list(probe) for probe in residual]]
    if refs or dropped:
        row += [[list(ref) for ref in refs], [list(entry) for entry in dropped]]
    return row


def snapshot_of(state):
    """The rows of a librarian state, read through each store's public surface."""
    core = tuple((e.id, e.text, e.superseded_by) for e in state.core.all())
    cellar = tuple((key, state.cellar.get(key)) for key in state.cellar.keys())
    receipts = tuple((r.key, r.stub, tuple(r.turn_ids), bool(r.resolved)) for r in state.ledger.all())
    marks = tuple(
        (seq, r.kind, tuple(tuple(anchor) for anchor in r.anchors))
        for seq, r in enumerate(state.ledger.all())
        if r.kind != "accepted" or r.anchors
    )
    peels = tuple(
        (p.id, p.text, int(p.level), tuple(p.sources), tuple(p.children), p.source_digest,
         int(p.probes_passed), int(p.probes_total))
        for p in state.tree.all()
    )
    peel_marks = tuple(
        (p.id, p.rung, tuple(tuple(unit) for unit in p.stitched), tuple(tuple(probe) for probe in p.residual),
         tuple(tuple(ref) for ref in getattr(p, "refs", ())), tuple(tuple(entry) for entry in getattr(p, "dropped", ())))
        for p in state.tree.all()
        if p.rung != "accepted" or p.stitched or p.residual or getattr(p, "refs", ()) or getattr(p, "dropped", ())
    )
    proposals = tuple(
        (q.id, q.span_key, q.turn_id, int(q.start), int(q.stop), q.origin, q.made_on, q.status)
        for q in getattr(state, "proposals", ())
    )
    refusals = tuple(sorted((str(k), str(v)) for k, v in getattr(state, "refusals", {}).items()))
    flesh = tuple(state.flesh.turns())
    present = {str(t.get("turn_id", "")) for _key, span in cellar for t in span} | {
        str(t.get("turn_id", "")) for t in flesh}
    lineage = tuple(sorted((str(turn_id), tuple(entries)) for turn_id, entries in getattr(state, "lineage", {}).items()
                           if str(turn_id) in present))
    return Snapshot(core=core, cellar=cellar, receipts=receipts, peels=peels,
                    flesh=flesh, seen=int(state.seen), receipt_marks=marks,
                    peel_marks=peel_marks, proposals=proposals, refusals=refusals, lineage=lineage)


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
        """Create the tables of a new file, or bring an older one to ``SCHEMA_VERSION`` through its migrations.

        One write transaction from the first read of the version to the last
        write: a second process opening the same file waits, then finds it
        brought up, and migrates nothing twice.
        """
        with self._lock, self._conn() as conn:
            # The transaction is this method's, named in SQL, whichever
            # driver the connection comes from: none of them opens or closes
            # one behind it.
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version > SCHEMA_VERSION:
                    raise OnionStoreError(
                        f"the onion store at {self._path} is of schema {version}, newer than this build's "
                        f"{SCHEMA_VERSION} ({_MIGRATIONS[-1][1]}): refused, left as it was"
                    )
                written = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'onion_cursor'"
                ).fetchone() is not None
                for statement in _SCHEMA_STATEMENTS:
                    conn.execute(statement)
                if written:
                    for target, _name, step in _MIGRATIONS:
                        if version < target:
                            step(conn)
                if version != SCHEMA_VERSION:
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except Exception:  # noqa: BLE001 - the engine may have rolled back already: the first error is told
                    pass
                raise

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
            conn.executemany(
                "INSERT INTO onion_receipt_marks (conversation, seq, kind, anchors) VALUES (?, ?, ?, ?)",
                [(cid, int(seq), kind, _canonical([list(anchor) for anchor in anchors]))
                 for seq, kind, anchors in snapshot.receipt_marks],
            )
            conn.executemany(
                "INSERT INTO onion_peel_marks (conversation, id, rung, stitched, residual, refs, dropped) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(cid, p_id, rung, _canonical([list(unit) for unit in stitched]),
                  _canonical([list(probe) for probe in residual]), _canonical([list(ref) for ref in refs]),
                  _canonical([list(entry) for entry in dropped]))
                 for p_id, rung, stitched, residual, refs, dropped in map(_mark_parts, snapshot.peel_marks)],
            )
            conn.executemany(
                "INSERT INTO onion_lineage (conversation, turn_id, lineage) VALUES (?, ?, ?)",
                [(cid, turn_id, _canonical(list(entries))) for turn_id, entries in snapshot.lineage],
            )
            conn.executemany(
                "INSERT INTO onion_proposals (conversation, seq, id, span_key, turn_id, start, stop, origin, made_on, "
                "status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(cid, i, *row) for i, row in enumerate(snapshot.proposals)],
            )
            conn.executemany(
                "INSERT INTO onion_refusals (conversation, span_key, mark) VALUES (?, ?, ?)",
                [(cid, key, mark) for key, mark in snapshot.refusals],
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
        with self._lock, self._conn() as conn:
            return _read_snapshot(conn, str(conversation_id))

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


def _read_snapshot(conn, cid):
    """The rows of conversation ``cid`` on an open connection, and its saved root; ``(None, None)`` when unknown."""
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
    marks = conn.execute(
        "SELECT seq, kind, anchors FROM onion_receipt_marks WHERE conversation = ? ORDER BY seq", (cid,)
    ).fetchall()
    # A file a migration is still bringing up may not hold the columns of
    # references yet: its marks have none.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(onion_peel_marks)").fetchall()}
    more = "m.refs, m.dropped" if {"refs", "dropped"} <= columns else "'[]', '[]'"
    peel_marks = conn.execute(
        f"SELECT m.id, m.rung, m.stitched, m.residual, {more} FROM onion_peel_marks m "
        "LEFT JOIN onion_peels p ON p.conversation = m.conversation AND p.id = m.id "
        "WHERE m.conversation = ? ORDER BY p.seq, m.id", (cid,)
    ).fetchall()
    lineage = conn.execute(
        "SELECT turn_id, lineage FROM onion_lineage WHERE conversation = ? ORDER BY turn_id", (cid,)
    ).fetchall()
    proposals = conn.execute(
        "SELECT id, span_key, turn_id, start, stop, origin, made_on, status FROM onion_proposals "
        "WHERE conversation = ? ORDER BY seq", (cid,)
    ).fetchall()
    refusals = conn.execute(
        "SELECT span_key, mark FROM onion_refusals WHERE conversation = ? ORDER BY span_key", (cid,)
    ).fetchall()
    try:
        snapshot = Snapshot(
            core=tuple((r[0], r[1], r[2]) for r in core),
            cellar=tuple((r[0], json.loads(r[1])) for r in cellar),
            receipts=tuple((r[0], r[1], tuple(json.loads(r[2])), bool(r[3])) for r in receipts),
            peels=tuple((r[0], r[1], int(r[2]), tuple(json.loads(r[3])), tuple(json.loads(r[4])), r[5], int(r[6]),
                         int(r[7])) for r in peels),
            flesh=tuple(json.loads(r[0]) for r in flesh),
            seen=int(cursor[0]),
            receipt_marks=tuple(
                (int(r[0]), r[1], tuple(tuple(anchor) for anchor in json.loads(r[2]))) for r in marks
            ),
            peel_marks=tuple(
                (r[0], r[1], tuple(tuple(unit) for unit in json.loads(r[2])),
                 tuple(tuple(probe) for probe in json.loads(r[3])), tuple(tuple(ref) for ref in json.loads(r[4])),
                 tuple(tuple(entry) for entry in json.loads(r[5])))
                for r in peel_marks
            ),
            proposals=tuple((r[0], r[1], r[2], int(r[3]), int(r[4]), r[5], r[6], r[7]) for r in proposals),
            refusals=tuple((r[0], r[1]) for r in refusals),
            lineage=tuple((r[0], _read_lineage(r[1])) for r in lineage),
        )
    except _CONTENT_ERRORS as exc:
        raise OnionIntegrityError(
            f"onion state {cid}: a row does not decode ({type(exc).__name__}); refused, not repaired"
        ) from exc
    return snapshot, cursor[1]


# What a row that does not read as the schema says raises while it is decoded
# or written again: the content's failure, told as the conversation's
# refusal. An error of the database itself is none of these, and is let
# through as it was.
_CONTENT_ERRORS = (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError, RecursionError)


def _read_lineage(text):
    """A stored lineage, read through the grammar: one that lies outside it reads cut, never empty."""
    from .peels import CUT
    from .probes import _context_defect

    try:
        entries = json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return (CUT,)
    if not isinstance(entries, list) or _context_defect("assistant", "assistant", [], entries) is not None:
        return (CUT,)
    return tuple(entries)


def _typed_receipts(conn):
    """Schema 1, "typed-receipts": each receipt's line says its key, turns, kind and origins, no word of its span.

    A file written before it keeps, in each receipt, the first words of its
    span. Each conversation's saved root is proved first, then every line is
    written again from the Cellar and the root recomputed over the rows as
    they now stand, under a savepoint of its own. A conversation that does
    not prove -- its root does not answer to its rows, a receipt names a
    span the file does not hold, a row does not read as the schema says --
    is left as it was, never laundered, and refused by name when it is
    loaded; the others migrate. An error of the database itself is no
    conversation's: it rolls the whole migration back, and the file is
    brought up at a later opening.
    """
    for (cid,) in conn.execute("SELECT conversation FROM onion_cursor ORDER BY conversation").fetchall():
        conn.execute("SAVEPOINT conversation")
        try:
            _type_receipts_of(conn, cid)
        except OnionStoreError:
            conn.execute("ROLLBACK TO conversation")
        conn.execute("RELEASE conversation")


def _type_receipts_of(conn, cid):
    """Write one proved conversation's receipt lines again; refused by name when its content does not prove."""
    from dataclasses import replace

    from .receipts import make_receipt

    snapshot, saved_root = _read_snapshot(conn, cid)
    spans = dict(snapshot.cellar)
    if onion_root(snapshot) != saved_root or any(key not in spans for key, *_rest in snapshot.receipts):
        raise OnionIntegrityError(f"onion state {cid}: does not prove; not migrated")
    kinds = {int(seq): kind for seq, kind, _anchors in snapshot.receipt_marks}
    receipts = []
    for seq, (key, _stub, ids, resolved) in enumerate(snapshot.receipts):
        try:
            line = make_receipt(spans[key], key, kinds.get(seq, "accepted")).stub
        except _CONTENT_ERRORS as exc:
            raise OnionIntegrityError(
                f"onion state {cid}: a span does not read as a span ({type(exc).__name__}); not migrated"
            ) from exc
        receipts.append((key, line, ids, resolved))
        conn.execute("UPDATE onion_receipts SET stub = ? WHERE conversation = ? AND seq = ?", (line, cid, seq))
    root = onion_root(replace(snapshot, receipts=tuple(receipts)))
    conn.execute("UPDATE onion_cursor SET root = ? WHERE conversation = ?", (root, cid))


def _references(conn):
    """Schema 2, "references": a peel's mark holds its references and its dropped sentences; turns have a lineage table.

    Both columns are added to the marks of a file written before them,
    empty, and the table of lineages comes with the schema: no peel of such
    a file references or dropped anything, so every root it holds still
    answers to its rows and nothing is written again. Its peels are read as
    they were saved, their copied runs with them.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(onion_peel_marks)").fetchall()}
    for name in ("refs", "dropped"):
        if name not in columns:
            conn.execute(f"ALTER TABLE onion_peel_marks ADD COLUMN {name} TEXT NOT NULL DEFAULT '[]'")


# The schema a file was written with, in SQLite's ``user_version``: 0 for
# every file written before versions existed. Each migration brings a file
# to its version, in order and once, inside one transaction with the tables
# it adds; a file of a newer version than the last one here is refused by
# name, and left as it was.
_MIGRATIONS = (
    (1, "typed-receipts", _typed_receipts),
    (2, "references", _references),
)
SCHEMA_VERSION = _MIGRATIONS[-1][0]
_SCHEMA_STATEMENTS = tuple(statement.strip() for statement in _SCHEMA.split(";") if statement.strip())


def _rebuild(state, snapshot):
    """Fill an empty librarian state from a snapshot through each store's surface, re-hashing as it goes."""
    from .core_store import USER, entry_hash
    from .peels import DROP_MOTIVES, Peel, _well_formed
    from .receipts import RECEIPT_KINDS, Receipt, span_key

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

    marks = {int(seq): (kind, anchors) for seq, kind, anchors in snapshot.receipt_marks}
    if any(seq < 0 or seq >= len(snapshot.receipts) for seq in marks):
        raise OnionIntegrityError("a receipt mark names a receipt the file does not hold: refused, not repaired")
    if any(kind not in RECEIPT_KINDS for kind, _anchors in marks.values()):
        raise OnionIntegrityError("a receipt mark names a kind the ledger does not know: refused, not repaired")
    for seq, (key, stub, ids, resolved) in enumerate(snapshot.receipts):
        kind, anchors = marks.get(seq, ("accepted", ()))
        state.ledger.append(Receipt(key=key, stub=stub, turn_ids=tuple(ids), resolved=bool(resolved), kind=kind,
                                    anchors=tuple(tuple(anchor) for anchor in anchors)))

    peel_marks = {mark[0]: _mark_parts(mark)[1:] for mark in snapshot.peel_marks}
    if set(peel_marks) - {row[0] for row in snapshot.peels}:
        raise OnionIntegrityError("a peel mark names a peel the file does not hold: refused, not repaired")
    if any(mark[0] not in _PEEL_RUNGS for mark in peel_marks.values()):
        raise OnionIntegrityError("a peel mark names a rung no peel is made on: refused, not repaired")
    if any(not _well_formed(ref) for mark in peel_marks.values() for ref in mark[3]):
        raise OnionIntegrityError("a peel mark holds a reference that is no place with its digest: refused, not repaired")
    if any(len(entry) != 2 or entry[0] not in DROP_MOTIVES or not isinstance(entry[1], str) or len(entry[1]) != 64
           or not all(ch in "0123456789abcdef" for ch in entry[1])
           for mark in peel_marks.values() for entry in mark[4]):
        raise OnionIntegrityError("a peel mark holds a dropped sentence of no known motive or digest: refused, "
                                  "not repaired")
    for p_id, text, level, sources, children, digest, passed, total in snapshot.peels:
        rung, stitched, residual, refs, dropped = peel_marks.get(p_id, ("accepted", (), (), (), ()))
        state.tree.add(Peel(id=p_id, text=text, level=int(level), sources=tuple(sources), children=tuple(children),
                            source_digest=digest, probes_passed=int(passed), probes_total=int(total),
                            rung=rung, stitched=tuple(stitched), residual=tuple(residual),
                            refs=tuple(tuple(ref) for ref in refs), dropped=tuple(tuple(entry) for entry in dropped)))
    state.tree.verify(state.cellar)
    state.ledger.digest(state.cellar)

    for turn in snapshot.flesh:
        state.flesh.append(turn)
    state.seen = int(snapshot.seen)
    state.refusals = {key: mark for key, mark in snapshot.refusals}
    if hasattr(state, "lineage"):
        state.lineage = {str(turn_id): tuple(entries) for turn_id, entries in snapshot.lineage}

    if snapshot.proposals:
        from .core_store import Proposal, proposal_id

        for p_id, span_key, turn_id, start, stop, origin, made_on, status in snapshot.proposals:
            if status not in _PROPOSAL_STATUSES or not state.cellar.has(span_key):
                raise OnionIntegrityError(f"proposal {p_id[:12]} names a status or a span the file does not hold")
            if p_id != proposal_id(span_key, turn_id, start, stop):
                raise OnionIntegrityError(f"proposal {p_id[:12]} no longer answers to its place: refused, not repaired")
            text = {str(t.get("turn_id", "")): str(t.get("text", "")) for t in state.cellar.get(span_key)}.get(turn_id)
            if text is None or not 0 <= int(start) < int(stop) <= len(text):
                raise OnionIntegrityError(f"proposal {p_id[:12]} points outside its span: refused, not repaired")
            state.proposals.append(Proposal(p_id, span_key, turn_id, int(start), int(stop), origin, made_on, status))
    return state
