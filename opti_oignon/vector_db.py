#!/usr/bin/env python3
"""The knowledge base's store: collections of texts, their metadata and their vectors, in one SQLite file the
platform opens as it opens every other store.

ChromaDB kept the knowledge base -- the documents' chunks, the memory's facts, the projects' files -- in files of
its own, in plaintext, with no encryption at rest: the texts, their metadata, and the vectors, which give the texts
back (vec2text, Morris et al. 2023, recovers 92 per cent of 32-token texts word for word from their embeddings).
This store keeps all three in the pages of a database opened through ``safe_connect``: encrypted by SQLCipher under
the platform's key where one loads, in plaintext where none does, as every store of the platform. Its journals are
encrypted with it, its temporary tables are kept in memory, and what a deletion frees is overwritten.

THE CONTRACT. The modules that kept their collections in ChromaDB speak to this store through ChromaDB's contract,
reduced to what they use. A ``Client`` opens one namespace of the file -- one for each directory ChromaDB kept, so
two collections of one name in two of them stay apart -- and gets, creates, lists and deletes its collections. A
``Collection`` adds, upserts, gets, queries, deletes and counts its items, and is found by its name at each call;
an add skips an id the collection holds already, as ChromaDB does.
A ``where`` filter compares one metadata field (``$eq``, ``$ne``, ``$in``, ``$nin``, ``$gt``, ``$gte``, ``$lt``,
``$lte``, or a bare value for ``$eq``), or joins two filters or more (``$and``, ``$or``); any other form is refused
by name, never ignored. A field a filter compares with ``$ne`` or ``$nin`` and an item does not have counts as
different.

WHAT IT NEVER DOES. It computes no vector. ChromaDB embedded a text itself when it was handed no vector, with a model
it downloaded on first use; here a query by text, and an item without its vector, are refused by name.

A MIGRATION THAT WAITS. A namespace whose ChromaDB folder still holds ChromaDB's store is closed: nothing of it is
read or written (``MigrationPending``, naming the folder and the command that moves it) until
``python3 -m opti_oignon.db_encryption migrate`` has moved that store into this one. What the platform would write or
delete meanwhile could not be told apart from what the migration brings, so it does neither. What waits is said by
one test, ``chroma_state``, which the migration reads too: a folder that holds ChromaDB's database as a regular file
that begins as one; a folder, or that file, that is a symbolic link, which the migration never follows; a folder
that cannot be looked at.

THE SEARCH. Exact: the query is compared with every vector the filter keeps, in ChromaDB's distances -- the squared
Euclidean distance (``l2``), one minus the cosine similarity (``cosine``), one minus the inner product (``ip``) --
and the nearest come first, ties in the order the items were added. A collection's vectors are read once per change
of it -- each write counts one, its generation, read again before every query, so a write by another process is
seen -- and kept in memory. The distances are computed in pure Python, each sum exactly rounded (``math.fsum``), so
a distance does not depend on the order of its terms: about 25 microseconds a vector of 1024 dimensions.

THE ENCODER. Each vector is kept with the encoder that made it, and a collection reads only its active encoder's
vectors: a query never mixes two encoders. Nothing here changes a collection's encoder yet; the platform's vectors
are kept under the empty encoder.
"""

from __future__ import annotations

import heapq
import ipaddress
import json
import logging
import math
import os
import re
import stat
import sys
import threading
import time
from array import array
from contextlib import contextmanager
from operator import mul
from pathlib import Path
from typing import Any

from opti_oignon.db_utils import safe_connect

logger = logging.getLogger(__name__)

# The file of the knowledge base, in the store folder ``rag`` of the package's place of data.
FILENAME = "vectors.db"
# The version of the schema below, kept in the file's user_version.
SCHEMA_VERSION = 1
# ChromaDB's distances, under its names, and the one a collection takes when its metadata names none: squared L2,
# ChromaDB's own default (1.3.7 included).
SPACES = ("cosine", "l2", "ip")
DEFAULT_SPACE = "l2"
SPACE_KEY = "hnsw:space"
# The command that moves ChromaDB's stores into this one, the platform stopped; ``db_encryption`` says the same.
MIGRATE_COMMAND = "python3 -m opti_oignon.db_encryption migrate"
# ChromaDB's database in each of its folders, and how every SQLite file no cipher wrote begins.
CHROMA_FILE = "chroma.sqlite3"
_PLAIN_HEADER = b"SQLite format 3\x00"
# The comparisons a filter makes on one field, and the joins of filters.
_COMPARISONS = ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte")
_JOINS = ("$and", "$or")
# What a read may include, and what a query adds to it.
_FIELDS = ("documents", "metadatas", "embeddings")
_QUERY_FIELDS = _FIELDS + ("distances",)
# How many ids one statement names.
_BATCH = 500
# ChromaDB's rule for a collection's name: 3 to 512 characters, letters, digits, dots, dashes and underscores,
# beginning and ending with a letter or a digit, no two dots in a row, no IPv4 address.
_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")

_TABLES = (
    """CREATE TABLE IF NOT EXISTS collections (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace   TEXT NOT NULL,
        name        TEXT NOT NULL,
        space       TEXT NOT NULL CHECK (space IN ('cosine', 'l2', 'ip')),
        metadata    TEXT,
        dimension   INTEGER,
        encoder     TEXT NOT NULL DEFAULT '',
        generation  INTEGER NOT NULL DEFAULT 0,
        created_at  REAL NOT NULL,
        UNIQUE (namespace, name)
    )""",
    """CREATE TABLE IF NOT EXISTS items (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        collection  INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        item_id     TEXT NOT NULL,
        document    TEXT,
        metadata    TEXT,
        UNIQUE (collection, item_id)
    )""",
    """CREATE TABLE IF NOT EXISTS vectors (
        item        INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
        encoder     TEXT NOT NULL,
        dimension   INTEGER NOT NULL,
        vector      BLOB NOT NULL,
        PRIMARY KEY (item, encoder)
    )""",
    """CREATE TABLE IF NOT EXISTS migrations (
        id          INTEGER PRIMARY KEY,
        source      TEXT NOT NULL,
        namespace   TEXT NOT NULL,
        collection  TEXT NOT NULL,
        copied      INTEGER NOT NULL,
        kept        INTEGER NOT NULL,
        removed     INTEGER NOT NULL DEFAULT 0,
        at          REAL NOT NULL
    )""",
)


class CollectionNotFound(ValueError):
    """A collection the namespace does not hold."""


class CollectionExists(ValueError):
    """A collection created under a name the namespace holds already."""


class MigrationPending(RuntimeError):
    """A namespace whose ChromaDB store waits for its migration: nothing of it is read or written."""


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

def chroma_state(folder: Any) -> tuple:
    """(state, detail) of a folder ChromaDB kept a namespace's collections in: ``"store"`` when it holds ChromaDB's
    database, a regular file that begins as a SQLite file; ``"link"`` when the folder or that file is a symbolic
    link, never followed; ``"unreadable"`` when either cannot be looked at, with why; ``"none"`` otherwise --
    absent, empty, or a file that is no such database. The migration moves what is a ``"store"``; every state but
    ``"none"`` closes the namespace."""
    folder = Path(folder)
    for path in (folder, folder / CHROMA_FILE):
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            return "none", ""
        except OSError as exc:
            return "unreadable", f"{type(exc).__name__}: {exc.strerror or exc}"
        if stat.S_ISLNK(mode):
            return "link", str(path)
        if path == folder and not stat.S_ISDIR(mode):
            return "none", ""
    if not stat.S_ISREG(mode):
        return "none", ""
    try:
        fd = os.open(folder / CHROMA_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            head = os.read(fd, len(_PLAIN_HEADER))
        finally:
            os.close(fd)
    except OSError as exc:
        return "unreadable", f"{type(exc).__name__}: {exc.strerror or exc}"
    return ("store", "") if head == _PLAIN_HEADER else ("none", "")


@contextmanager
def _connection(path: Path, *, write: bool, journal: bool = True) -> Any:
    """A connection to the store at ``path`` inside one transaction, committed when the block ends and rolled back
    when it raises; immediate for a write, its tables made first, and the store put back on its write-ahead log
    unless ``journal`` is False. A read of a file that does not exist, or holds no table yet, yields None: nothing
    is created to be read."""
    if not write and not path.exists():
        yield None
        return
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = safe_connect(path, check_same_thread=False)
    try:
        conn.isolation_level = None
        for pragma in ("foreign_keys = ON", "temp_store = MEMORY", "secure_delete = ON"):
            conn.execute(f"PRAGMA {pragma}")
        if write and journal:
            _write_ahead(conn)
        if not _schema(conn, path, write=write):
            yield None
            return
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield conn
        except BaseException:
            _roll_back(conn)
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


def _write_ahead(conn: Any) -> None:
    """Put the store on a write-ahead log, which lets a reader read while a writer writes; a store the encryption
    copied comes back on a rollback journal. Another connection may hold the lock the change needs: the store then
    keeps its journal, and the next write tries again."""
    try:
        if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
            conn.execute("PRAGMA journal_mode = WAL")
    except Exception as exc:  # noqa: BLE001 - a store on its rollback journal works the same, only less at once
        logger.debug("the knowledge base's store stays on its rollback journal for now: %s", exc)


def _roll_back(conn: Any) -> None:
    """Roll the transaction back; one SQLite rolled back already, or a failed rollback, leaves the raise to the error
    that came first -- closing the connection rolls back what is left."""
    try:
        conn.execute("ROLLBACK")
    except Exception:  # noqa: BLE001 - the first error is the one to raise
        pass


def _schema(conn: Any, path: Path, *, write: bool) -> bool:
    """Whether the store's tables are there, made first for a write. A file of a later schema is refused."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"{path} was written by a later version of the platform (schema {version}; this one reads "
                           f"{SCHEMA_VERSION}): nothing was read or written")
    if version == SCHEMA_VERSION or not write:
        return version == SCHEMA_VERSION
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Another process may have made them meanwhile: the version is read again under the write lock.
        if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            for statement in _TABLES:
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    except BaseException:
        _roll_back(conn)
        raise
    conn.execute("COMMIT")
    return True


def _row(conn: Any, namespace: str, name: str) -> Any:
    """The collection ``name`` of ``namespace``: (id, space, metadata, dimension, encoder, generation), or None."""
    if conn is None:
        return None
    return conn.execute("SELECT id, space, metadata, dimension, encoder, generation FROM collections "
                        "WHERE namespace = ? AND name = ?", (namespace, name)).fetchone()


def _dumps(value: Any) -> Any:
    return None if not value else json.dumps(value, allow_nan=False, sort_keys=True)


def _loads(text: Any) -> Any:
    return None if text is None else json.loads(text)


def _blob(vector: array) -> bytes:
    """The bytes of a float32 vector, little-endian on every machine."""
    if sys.byteorder == "big":
        vector = array("f", vector)
        vector.byteswap()
    return vector.tobytes()


def _unblob(blob: bytes) -> array:
    vector = array("f")
    vector.frombytes(blob)
    if sys.byteorder == "big":
        vector.byteswap()
    return vector


# ---------------------------------------------------------------------------
# What a caller hands in
# ---------------------------------------------------------------------------

def _check_name(name: Any) -> None:
    if not (isinstance(name, str) and 3 <= len(name) <= 512 and _NAME.fullmatch(name) and ".." not in name):
        raise ValueError(f"{name!r} is no collection name: 3 to 512 letters, digits, dots, dashes or underscores, "
                         "beginning and ending with a letter or a digit, no two dots in a row")
    try:
        ipaddress.IPv4Address(name)
    except ValueError:
        return
    raise ValueError(f"{name!r} is no collection name: it is an IPv4 address")


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, bool, int)) or (isinstance(value, float) and math.isfinite(value))


def _collection_metadata(metadata: Any) -> tuple:
    """(the space, the metadata kept) of a collection's metadata."""
    if metadata is None:
        return DEFAULT_SPACE, None
    if not isinstance(metadata, dict) or not all(isinstance(k, str) and _scalar(v) for k, v in metadata.items()):
        raise ValueError("a collection's metadata maps names to strings, numbers or booleans")
    space = metadata.get(SPACE_KEY, DEFAULT_SPACE)
    if space not in SPACES:
        raise ValueError(f"{SPACE_KEY} is {space!r}: this store measures {', '.join(SPACES)}")
    return space, dict(metadata) or None


def _ids(ids: Any) -> list:
    if not isinstance(ids, (list, tuple)) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("ids are a list of non-empty strings")
    seen, twice = set(), []
    for item in ids:
        if item in seen:
            twice.append(item)
        seen.add(item)
    if twice:
        raise ValueError(f"ids given twice in one call: {sorted(set(twice))[:10]}")
    return list(ids)


def _column(values: Any, count: int, what: str) -> list:
    if values is None:
        return [None] * count
    values = list(values)
    if len(values) != count:
        raise ValueError(f"{len(values)} {what} for {count} ids")
    return values


def _vector(values: Any) -> array:
    try:
        numbers = [float(x) for x in values if not isinstance(x, (str, bytes, bool))]
        whole = len(numbers) == len(values)
    except (TypeError, ValueError):
        whole = False
    if not whole or not numbers or not all(math.isfinite(x) for x in numbers):
        raise ValueError("an embedding is a non-empty sequence of finite numbers")
    return array("f", numbers)


def _metadata(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(isinstance(k, str) and (v is None or _scalar(v))
                                              for k, v in value.items()):
        raise ValueError("an item's metadata maps names to strings, finite numbers, booleans, or None")
    return value


def _document(value: Any) -> Any:
    if value is not None and not isinstance(value, str):
        raise ValueError("a document is a string")
    return value


def _records(ids: Any, embeddings: Any, metadatas: Any, documents: Any) -> list:
    """(id, vector, metadata, document) of each item a write hands in, each checked; refused by name otherwise."""
    ids = _ids(ids)
    if embeddings is None:
        raise ValueError("no embeddings: this store computes no vector; hand one in for each id")
    vectors = [_vector(v) for v in _column(embeddings, len(ids), "embeddings")]
    if len({len(v) for v in vectors}) > 1:
        raise ValueError("embeddings of different dimensions in one call")
    metas = [_metadata(m) for m in _column(metadatas, len(ids), "metadatas")]
    docs = [_document(d) for d in _column(documents, len(ids), "documents")]
    return list(zip(ids, vectors, metas, docs))


def _include(include: Any, allowed: tuple) -> list:
    if not isinstance(include, (list, tuple)) or not all(field in allowed for field in include):
        raise ValueError(f"include names some of {', '.join(allowed)}: {include!r}")
    return list(include)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _kind(value: Any) -> str:
    return "bool" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "str"


def _equal(have: Any, want: Any) -> bool:
    return _kind(have) == _kind(want) and have == want


def _comparison(field: str, operator: str, operand: Any) -> Any:
    if operator in ("$eq", "$ne"):
        if not _scalar(operand):
            raise ValueError(f"{field}: {operator} compares with a string, a number or a boolean")
        if operator == "$eq":
            return lambda meta: field in meta and _equal(meta[field], operand)
        return lambda meta: field not in meta or not _equal(meta[field], operand)
    if operator in ("$in", "$nin"):
        if not isinstance(operand, (list, tuple)) or not operand or not all(_scalar(v) for v in operand):
            raise ValueError(f"{field}: {operator} takes a non-empty list of strings, numbers or booleans")
        values = list(operand)
        if operator == "$in":
            return lambda meta: field in meta and any(_equal(meta[field], v) for v in values)
        return lambda meta: field not in meta or not any(_equal(meta[field], v) for v in values)
    if not isinstance(operand, (int, float)) or isinstance(operand, bool) or not math.isfinite(operand):
        raise ValueError(f"{field}: {operator} compares with a finite number")
    test = {"$gt": lambda a: a > operand, "$gte": lambda a: a >= operand,
            "$lt": lambda a: a < operand, "$lte": lambda a: a <= operand}[operator]
    return lambda meta: field in meta and _kind(meta[field]) == "number" and test(meta[field])


def _compile(where: Any) -> Any:
    """The test a ``where`` filter makes of an item's metadata; refused by name when it is no filter this store
    reads."""
    if not isinstance(where, dict) or len(where) != 1:
        raise ValueError(f"a where filter is a dictionary of exactly one field or join: {where!r}")
    (key, value), = where.items()
    if key in _JOINS:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError(f"{key} joins a list of two filters or more: {value!r}")
        parts = [_compile(part) for part in value]
        if key == "$and":
            return lambda meta: all(part(meta) for part in parts)
        return lambda meta: any(part(meta) for part in parts)
    if not isinstance(key, str) or key.startswith("$"):
        raise ValueError(f"{key!r} is no field and no join this store reads")
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"{key}: one comparison per field: {value!r}")
        (operator, operand), = value.items()
        if operator not in _COMPARISONS:
            raise ValueError(f"{key}: {operator!r} is no comparison this store reads ({', '.join(_COMPARISONS)})")
        return _comparison(key, operator, operand)
    return _comparison(key, "$eq", value)


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

class _Snapshot:
    """A collection's vectors of one encoder, as of one generation: in the order the items were added."""

    def __init__(self, generation: int, rows: list, ids: list, metadatas: list, vectors: list) -> None:
        self.generation = generation
        self.rows = rows
        self.ids = ids
        self.metadatas = metadatas
        self.vectors = vectors
        self.norms = [math.sqrt(math.fsum(map(mul, v, v))) for v in vectors]

    def vector(self, index: int) -> list:
        return list(self.vectors[index])

    def nearest(self, space: str, query: list, candidates: list, n: int) -> list:
        """The ``n`` candidates nearest ``query``, nearest first, ties in the order they were added: (index,
        distance)."""
        if not candidates:
            return []
        qnorm = math.sqrt(math.fsum(map(mul, query, query)))
        distances = []
        for index in candidates:
            vector = self.vectors[index]
            if space == "l2":
                diff = [a - b for a, b in zip(vector, query)]
                distances.append(math.fsum(map(mul, diff, diff)))
            elif space == "ip":
                distances.append(1.0 - math.fsum(map(mul, vector, query)))
            else:
                denominator = self.norms[index] * qnorm
                distances.append(1.0 - math.fsum(map(mul, vector, query)) / denominator if denominator > 0 else 1.0)
        best = heapq.nsmallest(n, range(len(candidates)), key=lambda j: (distances[j], self.rows[candidates[j]]))
        return [(candidates[j], distances[j]) for j in best]


def exact_searcher(space: str, vectors: list) -> Any:
    """A search of ``vectors`` in ``space``, read once: ``nearest(query, n)`` gives the ``n`` nearest ``query``,
    nearest first, ties in their order, as (index, distance) measured as a query of this store measures them."""
    if space not in SPACES:
        raise ValueError(f"{space!r} is no space this store measures")
    held = [_vector(v) for v in vectors]
    snapshot = _Snapshot(0, list(range(len(held))), [str(i) for i in range(len(held))], [None] * len(held), held)
    every = list(range(len(held)))
    return lambda query, n: snapshot.nearest(space, list(_vector(query)), every, n)


def exact_nearest(space: str, vectors: list, query: Any, n: int) -> list:
    """The ``n`` of ``vectors`` nearest ``query`` in ``space``: ``exact_searcher`` for one query."""
    return exact_searcher(space, vectors)(query, n)


def similarity(distance: float, space: str) -> float:
    """The similarity a distance of ``space`` stands for: one minus a cosine or inner-product distance; one minus
    half a squared Euclidean distance, which is the cosine similarity of two vectors of unit length."""
    if space not in SPACES:
        raise ValueError(f"{space!r} is no space this store measures")
    return 1.0 - float(distance) / 2.0 if space == "l2" else 1.0 - float(distance)


# ---------------------------------------------------------------------------
# The facade
# ---------------------------------------------------------------------------

class Client:
    """One namespace of the knowledge base's store at ``path``: its collections, got, created, listed and deleted.

    ``legacy`` is the folder ChromaDB kept this namespace's collections in:
    while it holds ChromaDB's store, every call raises ``MigrationPending``.
    Once seen empty of it, it is not looked at again.
    """

    def __init__(self, path: Any, namespace: str, *, legacy: Any = None) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("a namespace is a non-empty string")
        self.path = Path(path)
        self.namespace = namespace
        self.legacy = None if legacy is None else Path(legacy)
        self._moved = legacy is None
        self._lock = threading.Lock()
        self._snapshots: dict = {}
        # Per collection, the lowest generation a snapshot may be kept at: a deletion raises it past what it erased,
        # so a query whose read began before the deletion never keeps its view of it -- a collection deleted, forever.
        self._floor: dict = {}

    def pending(self) -> Any:
        """(folder, state, detail) of the ChromaDB folder this namespace waits for (``chroma_state``), or None."""
        if not self._moved:
            state, detail = chroma_state(self.legacy)
            if state != "none":
                return self.legacy, state, detail
            self._moved = True
        return None

    def ready(self) -> None:
        """Raise ``MigrationPending`` while this namespace waits for its migration."""
        waiting = self.pending()
        if waiting is None:
            return
        folder, state, detail = waiting
        what = "file" if Path(detail).name == CHROMA_FILE else "folder"
        why = {
            "store": "holds a ChromaDB store this version no longer reads",
            "link": f"leads through a symbolic link ({detail}), which the migration never follows: put the {what} "
                    "it leads to in its place",
            "unreadable": f"cannot be looked at ({detail}): make it readable",
        }[state]
        raise MigrationPending(
            f"{folder} {why}: the knowledge base's {self.namespace} waits for `{MIGRATE_COMMAND}`, run with the "
            "platform stopped, before anything of it is read or written")

    def _open(self, *, write: bool) -> Any:
        self.ready()
        return _connection(self.path, write=write)

    def get_collection(self, name: str) -> Collection:
        with self._open(write=False) as conn:
            if _row(conn, self.namespace, name) is None:
                raise CollectionNotFound(f"no collection {name!r} in {self.namespace}")
        return Collection(self, name)

    def get_or_create_collection(self, name: str, metadata: Any = None) -> Collection:
        """The collection ``name``, made with ``metadata`` when the namespace does not hold it; the metadata of one
        it holds is kept as it is."""
        _check_name(name)
        space, kept = _collection_metadata(metadata)
        with self._open(write=False) as conn:
            if _row(conn, self.namespace, name) is not None:
                return Collection(self, name)
        with self._open(write=True) as conn:
            if _row(conn, self.namespace, name) is None:
                conn.execute("INSERT INTO collections (namespace, name, space, metadata, created_at) "
                             "VALUES (?, ?, ?, ?, ?)", (self.namespace, name, space, _dumps(kept), time.time()))
        return Collection(self, name)

    def create_collection(self, name: str, metadata: Any = None) -> Collection:
        _check_name(name)
        space, kept = _collection_metadata(metadata)
        with self._open(write=True) as conn:
            if _row(conn, self.namespace, name) is not None:
                raise CollectionExists(f"collection {name!r} exists in {self.namespace}")
            conn.execute("INSERT INTO collections (namespace, name, space, metadata, created_at) "
                         "VALUES (?, ?, ?, ?, ?)", (self.namespace, name, space, _dumps(kept), time.time()))
        return Collection(self, name)

    def delete_collection(self, name: str) -> None:
        """Delete the collection ``name``, its items and their vectors."""
        with self._open(write=True) as conn:
            row = _row(conn, self.namespace, name)
            if row is None:
                raise CollectionNotFound(f"no collection {name!r} in {self.namespace}")
            conn.execute("DELETE FROM collections WHERE id = ?", (row[0],))
        self._forget_snapshots(row[0], math.inf)

    def list_collections(self) -> list:
        with self._open(write=False) as conn:
            if conn is None:
                return []
            names = [name for (name,) in conn.execute(
                "SELECT name FROM collections WHERE namespace = ? ORDER BY id", (self.namespace,))]
        return [Collection(self, name) for name in names]

    def empty_log(self) -> bool:
        """Empty the store's write-ahead log after a deletion: its older frames still hold what the deletion erased
        (the freed pages themselves are overwritten). Never waits: whether it was emptied -- a read in progress keeps
        it until the next deletion. A store that is not there holds no log, and is not created."""
        if not self.path.exists():
            return True
        conn = safe_connect(self.path, check_same_thread=False)
        try:
            conn.execute("PRAGMA busy_timeout = 0")
            return not conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
        finally:
            conn.close()

    def _forget_snapshots(self, collection: int, floor: float) -> None:
        """Drop every snapshot of ``collection``, and keep none below ``floor`` from now on."""
        with self._lock:
            self._floor[collection] = max(self._floor.get(collection, 0), floor)
            for key in [key for key in self._snapshots if key[0] == collection]:
                del self._snapshots[key]

    def _snapshot(self, conn: Any, row: Any) -> _Snapshot:
        """The vectors of the collection at ``row`` under its active encoder, read again when its generation moved;
        kept for the next query only when no deletion came since the read began."""
        key = (row[0], row[4])
        with self._lock:
            held = self._snapshots.get(key)
            if held is not None and held.generation == row[5]:
                return held
            rows, ids, metadatas, vectors = [], [], [], []
            for item, item_id, metadata, blob in conn.execute(
                    "SELECT i.id, i.item_id, i.metadata, v.vector FROM items i JOIN vectors v "
                    "ON v.item = i.id AND v.encoder = ? WHERE i.collection = ? ORDER BY i.id", (row[4], row[0])):
                rows.append(item)
                ids.append(item_id)
                metadatas.append(_loads(metadata))
                vectors.append(_unblob(blob))
            snapshot = _Snapshot(row[5], rows, ids, metadatas, vectors)
            if row[5] >= self._floor.get(row[0], 0):
                self._snapshots[key] = snapshot
            return snapshot


class Collection:
    """A collection of one namespace, found by its name at each call: one deleted and made again under the same
    name is the one a later call reaches."""

    def __init__(self, client: Client, name: str) -> None:
        self._client = client
        self.name = name
        self._space: Any = None

    def _read(self, conn: Any) -> Any:
        row = _row(conn, self._client.namespace, self.name)
        if row is None:
            raise CollectionNotFound(f"no collection {self.name!r} in {self._client.namespace}")
        self._space = row[1]
        return row

    @property
    def metadata(self) -> Any:
        with self._client._open(write=False) as conn:
            return _loads(self._read(conn)[2])

    @property
    def space(self) -> str:
        with self._client._open(write=False) as conn:
            return self._read(conn)[1]

    def similarity(self, distance: float) -> float:
        """The similarity ``distance`` stands for in this collection's space: the one its last query measured in."""
        return similarity(distance, self._space if self._space is not None else self.space)

    def count(self) -> int:
        with self._client._open(write=False) as conn:
            row = self._read(conn)
            return conn.execute("SELECT count(*) FROM items WHERE collection = ?", (row[0],)).fetchone()[0]

    # -- writes ------------------------------------------------------------

    def add(self, ids: Any, embeddings: Any = None, metadatas: Any = None, documents: Any = None) -> None:
        """Add items: an id the collection holds already is skipped, the item it names kept as it is, as ChromaDB
        does."""
        records = _records(ids, embeddings, metadatas, documents)
        with self._client._open(write=True) as conn:
            row = self._read(conn)
            self._fits(row, records)
            held = self._held(conn, row[0], [record[0] for record in records])
            fresh = [record for record in records if record[0] not in held]
            for item_id, vector, metadata, document in fresh:
                self._insert(conn, row, item_id, vector, metadata, document)
            self._written(conn, row, fresh)

    def upsert(self, ids: Any, embeddings: Any = None, metadatas: Any = None, documents: Any = None) -> None:
        """Add items, or update the ones the collection holds: their vector replaced; their document replaced when
        one is handed in; their metadata merged with what is handed in, a field handed in as None removed."""
        records = _records(ids, embeddings, metadatas, documents)
        with self._client._open(write=True) as conn:
            row = self._read(conn)
            self._fits(row, records)
            held = self._held(conn, row[0], [record[0] for record in records])
            for item_id, vector, metadata, document in records:
                if item_id not in held:
                    self._insert(conn, row, item_id, vector, metadata, document)
                    continue
                item, old_document, old_metadata = held[item_id]
                merged = dict(_loads(old_metadata) or {})
                for field, value in (metadata or {}).items():
                    if value is None:
                        merged.pop(field, None)
                    else:
                        merged[field] = value
                conn.execute("UPDATE items SET document = ?, metadata = ? WHERE id = ?",
                             (old_document if document is None else document, _dumps(merged), item))
                conn.execute("INSERT OR REPLACE INTO vectors (item, encoder, dimension, vector) VALUES (?, ?, ?, ?)",
                             (item, row[4], len(vector), _blob(vector)))
            self._written(conn, row, records)

    def delete(self, ids: Any = None, where: Any = None) -> None:
        """Delete the items ``ids`` names that ``where`` keeps; a deletion that names neither is refused."""
        if ids is None and where is None:
            raise ValueError("a deletion names its ids, its filter, or both: never the whole collection")
        test = None if where is None else _compile(where)
        wanted = None if ids is None else _ids(ids)
        with self._client._open(write=True) as conn:
            row = self._read(conn)
            doomed = [item for item, _item_id, metadata in self._items(conn, row[0], wanted)
                      if test is None or test(_loads(metadata) or {})]
            for at in range(0, len(doomed), _BATCH):
                part = doomed[at:at + _BATCH]
                conn.execute(f"DELETE FROM items WHERE id IN ({','.join('?' * len(part))})", part)
            if doomed:
                conn.execute("UPDATE collections SET generation = generation + 1 WHERE id = ?", (row[0],))
        if doomed:
            # What the deleted items held leaves memory with them.
            self._client._forget_snapshots(row[0], row[5] + 1)

    def _fits(self, row: Any, records: list) -> None:
        dimension = len(records[0][1]) if records else None
        if records and row[3] is not None and dimension != row[3]:
            raise ValueError(f"embedding dimension {dimension} does not match the dimension of {self.name}, "
                             f"{row[3]}: nothing was written")

    @staticmethod
    def _held(conn: Any, collection: int, ids: list) -> dict:
        """{id: (item, document, metadata)} of the ``ids`` the collection holds."""
        held = {}
        for at in range(0, len(ids), _BATCH):
            part = ids[at:at + _BATCH]
            for item, item_id, document, metadata in conn.execute(
                    f"SELECT id, item_id, document, metadata FROM items WHERE collection = ? "
                    f"AND item_id IN ({','.join('?' * len(part))})", (collection, *part)):
                held[item_id] = (item, document, metadata)
        return held

    @staticmethod
    def _insert(conn: Any, row: Any, item_id: str, vector: array, metadata: Any, document: Any) -> None:
        kept = None if metadata is None else {k: v for k, v in metadata.items() if v is not None}
        cursor = conn.execute("INSERT INTO items (collection, item_id, document, metadata) VALUES (?, ?, ?, ?)",
                              (row[0], item_id, document, _dumps(kept)))
        conn.execute("INSERT INTO vectors (item, encoder, dimension, vector) VALUES (?, ?, ?, ?)",
                     (cursor.lastrowid, row[4], len(vector), _blob(vector)))

    @staticmethod
    def _written(conn: Any, row: Any, records: list) -> None:
        if records:
            conn.execute("UPDATE collections SET dimension = COALESCE(dimension, ?), generation = generation + 1 "
                         "WHERE id = ?", (len(records[0][1]), row[0]))

    # -- reads -------------------------------------------------------------

    @staticmethod
    def _items(conn: Any, collection: int, ids: Any) -> list:
        """(item, id, metadata) of the collection's items, in the order they were added, or of ``ids`` in their
        order, those it holds."""
        if ids is None:
            return conn.execute("SELECT id, item_id, metadata FROM items WHERE collection = ? ORDER BY id",
                                (collection,)).fetchall()
        found = {}
        for at in range(0, len(ids), _BATCH):
            part = ids[at:at + _BATCH]
            for item, item_id, metadata in conn.execute(
                    f"SELECT id, item_id, metadata FROM items WHERE collection = ? "
                    f"AND item_id IN ({','.join('?' * len(part))})", (collection, *part)):
                found[item_id] = (item, item_id, metadata)
        return [found[item_id] for item_id in ids if item_id in found]

    def get(self, ids: Any = None, where: Any = None, limit: Any = None, offset: Any = None,
            include: Any = ("metadatas", "documents")) -> dict:
        """The items ``ids`` names that ``where`` keeps -- all of them when neither is given --, in the order the
        ids name them or, without ids, in the order they were added; ``offset`` skipped and ``limit`` kept."""
        fields = _include(include, _FIELDS)
        test = None if where is None else _compile(where)
        wanted = None if ids is None else _ids(ids)
        for name, value in (("limit", limit), ("offset", offset)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} is a whole number, 0 or more")
        with self._client._open(write=False) as conn:
            row = self._read(conn)
            kept = [(item, item_id, _loads(metadata)) for item, item_id, metadata in self._items(conn, row[0], wanted)]
            if test is not None:
                kept = [entry for entry in kept if test(entry[2] or {})]
            start = offset or 0
            kept = kept[start:] if limit is None else kept[start:start + limit]
            documents = self._documents(conn, row[0], [entry[0] for entry in kept]) if "documents" in fields else {}
            vectors = self._vectors(conn, row, [entry[0] for entry in kept]) if "embeddings" in fields else {}
        return {
            "ids": [entry[1] for entry in kept],
            "embeddings": [vectors.get(entry[0]) for entry in kept] if "embeddings" in fields else None,
            "documents": [documents.get(entry[0]) for entry in kept] if "documents" in fields else None,
            "metadatas": [entry[2] for entry in kept] if "metadatas" in fields else None,
            "included": fields,
        }

    def query(self, query_embeddings: Any = None, n_results: int = 10, where: Any = None,
              include: Any = ("metadatas", "documents", "distances"), query_texts: Any = None) -> dict:
        """The ``n_results`` items nearest each of ``query_embeddings`` among those ``where`` keeps, nearest first:
        an exact search under the collection's active encoder."""
        if query_texts is not None:
            raise ValueError("a query by text: this store computes no vector; hand in query_embeddings")
        if query_embeddings is None:
            raise ValueError("a query needs query_embeddings")
        if not isinstance(n_results, int) or isinstance(n_results, bool) or n_results < 1:
            raise ValueError("n_results is a whole number, 1 or more")
        fields = _include(include, _QUERY_FIELDS)
        test = None if where is None else _compile(where)
        listed = list(query_embeddings)
        if listed and not isinstance(listed[0], (list, tuple)) and not hasattr(listed[0], "__len__"):
            listed = [listed]
        if not listed:
            raise ValueError("a query needs query_embeddings")
        queries = [list(_vector(q)) for q in listed]
        found = []
        with self._client._open(write=False) as conn:
            row = self._read(conn)
            snapshot = self._client._snapshot(conn, row)
            for query in queries:
                if row[3] is not None and len(query) != row[3]:
                    raise ValueError(f"query embedding dimension {len(query)} does not match the dimension of "
                                     f"{self.name}, {row[3]}")
            candidates = [i for i in range(len(snapshot.ids))
                          if test is None or test(snapshot.metadatas[i] or {})]
            for query in queries:
                found.append(snapshot.nearest(row[1], query, candidates, n_results))
            picked = [snapshot.rows[index] for hits in found for index, _distance in hits]
            documents = self._documents(conn, row[0], picked) if "documents" in fields else {}
        return {
            "ids": [[snapshot.ids[i] for i, _d in hits] for hits in found],
            "distances": [[d for _i, d in hits] for hits in found] if "distances" in fields else None,
            "documents": ([[documents.get(snapshot.rows[i]) for i, _d in hits] for hits in found]
                          if "documents" in fields else None),
            "metadatas": ([[snapshot.metadatas[i] for i, _d in hits] for hits in found]
                          if "metadatas" in fields else None),
            "embeddings": ([[snapshot.vector(i) for i, _d in hits] for hits in found]
                           if "embeddings" in fields else None),
            "included": fields,
        }

    @staticmethod
    def _documents(conn: Any, collection: int, items: list) -> dict:
        """{item: document} of ``items`` that the collection holds: never another collection's."""
        found = {}
        for at in range(0, len(items), _BATCH):
            part = items[at:at + _BATCH]
            found.update(conn.execute(f"SELECT id, document FROM items WHERE collection = ? "
                                      f"AND id IN ({','.join('?' * len(part))})", (collection, *part)).fetchall())
        return found

    @staticmethod
    def _vectors(conn: Any, row: Any, items: list) -> dict:
        """{item: vector} of ``items`` the collection at ``row`` holds, under its active encoder."""
        found = {}
        for at in range(0, len(items), _BATCH):
            part = items[at:at + _BATCH]
            for item, blob in conn.execute(
                    f"SELECT v.item, v.vector FROM vectors v JOIN items i ON i.id = v.item WHERE i.collection = ? "
                    f"AND v.encoder = ? AND v.item IN ({','.join('?' * len(part))})", (row[0], row[4], *part)):
                found[item] = list(_unblob(blob))
        return found


# ---------------------------------------------------------------------------
# A migration's write
# ---------------------------------------------------------------------------

def load(path: Any, batches: list, *, before_commit: Any = None) -> list:
    """Write ``batches`` into the store at ``path`` in one transaction, and prove them before it commits.

    Each batch is a dictionary: ``namespace``, ``name``, ``metadata``,
    ``source`` -- where it was read -- and ``items``, each (id, document,
    metadata, vector), and, for items whose copy must read a given text,
    ``drop``: {id: that text, or None when no copy may stay}. Its
    collection is made when the namespace does not hold it, and refused
    when it holds it measured in another space or with vectors of another
    dimension. An item of ``drop`` the collection holds with another text,
    or with any text when None, is deleted first -- a copy an earlier run
    made, from whichever source, of what has changed since --; then an item
    whose id the collection holds is kept as it is, every other item is
    written, and the batch is recorded in ``migrations``. Then every item
    written is read back and compared with what was handed in -- document,
    metadata, vector bit for bit -- and ``before_commit()`` is called: a
    difference, or a raise, rolls everything back, and nothing is written.

    Every batch is checked before the store is opened, and nothing is
    written before the transaction. A collection the store holds empty,
    measured in another space or sized for another dimension, takes the
    batch's space and metadata, its dimension unset: it holds nothing
    measured or sized the other way.

    Returns (written, kept, deleted) for each batch, in their order.
    """
    path = Path(path)
    checked = []
    for batch in batches:
        _check_name(batch["name"])
        items = batch["items"]
        checked.append((batch, *_collection_metadata(batch["metadata"]),
                        _records([i[0] for i in items], [i[3] for i in items], [i[2] for i in items],
                                 [i[1] for i in items])))
    counts, written = [], []
    # A migration writes nothing before its transaction: the store keeps its journal as it found it.
    with _connection(path, write=True, journal=False) as conn:
        for batch, space, kept_metadata, records in checked:
            row = _row(conn, batch["namespace"], batch["name"])
            if row is None:
                conn.execute("INSERT INTO collections (namespace, name, space, metadata, created_at) "
                             "VALUES (?, ?, ?, ?, ?)",
                             (batch["namespace"], batch["name"], space, _dumps(kept_metadata), time.time()))
                row = _row(conn, batch["namespace"], batch["name"])
            elif ((row[1] != space or (records and row[3] is not None and len(records[0][1]) != row[3]))
                  and not conn.execute("SELECT 1 FROM items WHERE collection = ? LIMIT 1", (row[0],)).fetchone()):
                conn.execute("UPDATE collections SET space = ?, metadata = ?, dimension = NULL, "
                             "generation = generation + 1 WHERE id = ?", (space, _dumps(kept_metadata), row[0]))
                row = _row(conn, batch["namespace"], batch["name"])
            elif row[1] != space:
                raise ValueError(f"{batch['namespace']}/{batch['name']} is measured in {row[1]} in {path} and in "
                                 f"{space} in {batch['source']}: nothing was written")
            if records and row[3] is not None and len(records[0][1]) != row[3]:
                raise ValueError(f"{batch['namespace']}/{batch['name']} holds vectors of dimension {row[3]} in {path} "
                                 f"and of dimension {len(records[0][1])} in {batch['source']}: nothing was written")
            drop = dict(batch.get("drop") or {})
            dropped = [found[0] for item_id, found in Collection._held(conn, row[0], list(drop)).items()
                       if drop[item_id] is None or found[1] != drop[item_id]]
            for at in range(0, len(dropped), _BATCH):
                part = dropped[at:at + _BATCH]
                conn.execute(f"DELETE FROM items WHERE id IN ({','.join('?' * len(part))})", part)
            if dropped:
                conn.execute("UPDATE collections SET generation = generation + 1 WHERE id = ?", (row[0],))
            held = Collection._held(conn, row[0], [record[0] for record in records])
            fresh = [record for record in records if record[0] not in held]
            for record in fresh:
                Collection._insert(conn, row, *record)
            Collection._written(conn, row, fresh)
            conn.execute("INSERT INTO migrations (source, namespace, collection, copied, kept, removed, at) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (str(batch["source"]), batch["namespace"], batch["name"], len(fresh), len(held),
                          len(dropped), time.time()))
            written.append((row, fresh))
            counts.append((len(fresh), len(held), len(dropped)))
        for row, fresh in written:
            back = Collection._held(conn, row[0], [record[0] for record in fresh])
            vectors = Collection._vectors(conn, row, [back[record[0]][0] for record in fresh if record[0] in back])
            for item_id, vector, metadata, document in fresh:
                expected = None if metadata is None else {k: v for k, v in metadata.items() if v is not None} or None
                if (item_id not in back or back[item_id][1] != document or _loads(back[item_id][2]) != expected
                        or vectors.get(back[item_id][0]) != list(vector)):
                    raise ValueError(f"{item_id} does not read back from {path} as it was handed in: nothing was "
                                     "written")
        if before_commit is not None:
            before_commit()
    return counts
