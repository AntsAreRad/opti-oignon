#!/usr/bin/env python3
"""Contracts for the knowledge base's store: texts, metadata and vectors in one SQLite file the platform encrypts.

ChromaDB kept the knowledge base in files of its own, in plaintext. The store
in ``opti_oignon.vector_db`` keeps it in a database opened like every store,
through ``safe_connect``, and speaks ChromaDB's contract to the five modules
that used it.

  * VD1 -- ENCRYPTED AT REST: under the platform's key the file never begins
    as a SQLite file, and none of its files -- the database, its write-ahead
    log -- holds a text, a metadata value or a vector as written; another key
    does not open it; each connection keeps its temporary data in memory and
    overwrites what a deletion frees, whatever its SQLite's defaults; the
    journal is a write-ahead log, and a store copied back to a rollback
    journal -- as the encryption copies it -- returns to it at its next
    write.
  * VD2 -- CHROMADB'S CONTRACT, as the five modules use it: collections got,
    made, listed and deleted in their namespace, two namespaces apart, squared
    L2 when none is named, as ChromaDB 1.3.7; items added, read, upserted,
    deleted and counted; what a read includes; an add that names an id held
    skips it and keeps the item, as ChromaDB 1.3.7 does; the names a
    collection may have.
  * VD3 -- THE FILTERS: each comparison, the joins, a field an item does not
    have; every other form refused by name, never ignored.
  * VD4 -- NO VECTOR COMPUTED: a query by text, a query with no vector, an add
    or an upsert with no vector, refused by name, nothing written.
  * VD5 -- AN EXACT SEARCH: each space's distances; nearest first; ties in the
    order the items were added; the filter applied before the nearest are
    taken; a vector of zeros; a dimension that differs refused.
  * VD6 -- A WRITE ELSEWHERE IS SEEN: a write by another client is found by
    the next query; with no write between two queries, the vectors are read
    once.
  * VD7 -- ONE ENCODER AT A TIME: a query reads only the vectors of the
    collection's active encoder; an item with none of them is counted and
    never returned.
  * VD8 -- THE REAL WIRING: the RAG store, the memory's vector layer, the
    project index and its context builder, the documents' indexer and
    retriever, each loaded from its file, write and read the same file, each
    in its namespace.
  * VD9 -- A SCORE IS A SIMILARITY in the collection's own space: the
    retriever and the project context builder score a cosine collection
    ``1 - d`` and a squared-L2 one ``1 - d/2``.
  * VD10 -- ONE FILE, WHERE EVERY MODULE OPENS IT: in the store folder
    ``rag`` of the package's place of data, where the migration writes it.
  * VD11 -- CHROMADB IS READ, NEVER RUN: one module imports it, the
    migration, inside a function; its client is built with its telemetry
    off, and nothing asks it for an embedding.
  * VD12 -- THE RAG STORE NEVER STORES WHAT IT CANNOT SEARCH: an ingestion
    with no embedding at all is refused by name, nothing stored; a query with
    no embedding searches nothing; a score is never below 0. It supersedes
    a15, a19 and a20 of the RAG store's suite, deselected by name.
  * VD13 -- THE MEMORY SAYS WHAT IT LACKS: without the store, its health says
    the knowledge base's store could not be opened, and every write is a
    no-op. It supersedes the health contract of the memory's vector suite
    that named chromadb, deselected by name.
  * VD14 -- NOTHING MADE TO BE READ, NOTHING READ OF A LATER SCHEMA: a read of
    a store that does not exist creates no file; a store of a later schema is
    refused, nothing read or written.
  * VD15 -- NEVER ANOTHER COLLECTION'S TEXT: a collection deleted and made
    again, another namespace's items written between, a client that searched
    before reads only the new collection's items, never a text of another;
    no id of a collection or an item is given twice.
  * VD16 -- A MIGRATION THAT WAITS CLOSES ITS NAMESPACE: while ChromaDB's
    folder of a namespace holds its store, nothing of the namespace is read
    or written, each call refused by name with the command that moves it; the
    RAG store deletes no document and no collection, searches nothing and
    says so, stores nothing; the memory degrades and says why; the documents'
    indexer and retriever, and the project index, open nothing; once the
    folder has moved, all of them work. An ingestion is refused before its
    first write, a document with no chunk too, and a page before it is
    fetched. What waits is one test, the one the migration reads
    (``chroma_state``).
  * VD17 -- THE ROUTES SAY IT: while the knowledge base waits, creating a
    collection, deleting one and deleting a document answer 503 with the
    reason the store gave, nothing created or deleted.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import ast
import math
import sqlite3
import sys
import types
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _allium_census import modules as package_modules  # noqa: E402
from _isolation import isolate, source  # noqa: E402

_KEY = bytes(range(1, 33))
_OTHER_KEY = bytes(range(33, 65))
_PLAIN = b"SQLite format 3\x00"
_BASE = {
    "opti_oignon.encryption": source("encryption.py"),
    "opti_oignon.db_encryption": source("db_encryption.py"),
    "opti_oignon.db_utils": source("db_utils.py"),
    "opti_oignon.vector_db": source("vector_db.py"),
}


def _window(tmp_path, monkeypatch, *, key=_KEY, targets=None, seeded=None, packages=()):
    """The store's modules in a window, under ``key`` (None: no key loads), the key file redirected."""
    pytest.importorskip("sqlcipher3")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    wanted = dict(_BASE)
    wanted.update(targets or {})
    row = sqlite3.Row
    loaded, restore = isolate(targets=wanted, blocked=("opti_oignon.secure_bytes",), seeded=seeded,
                              packages=packages)

    def closed():
        try:
            restore()
        finally:
            sqlite3.Row = row

    try:
        loaded["opti_oignon.encryption"]._DEFAULT_KEYFILE = tmp_path / "keys" / ".keyfile"
        dbe = loaded["opti_oignon.db_encryption"]
        dbe._get_db_encryption_key = lambda: key
        world = SimpleNamespace(vdb=loaded["opti_oignon.vector_db"], dbe=dbe, loaded=loaded, restore=closed,
                                path=tmp_path / "rag" / "vectors.db")
    except BaseException:
        closed()
        raise
    return world


def _items(collection, **kwargs):
    got = collection.get(include=["documents", "metadatas", "embeddings"], **kwargs)
    return list(zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]))


def _fsum_distance(space, a, b):
    a = [float(x) for x in array("f", a)]
    b = [float(x) for x in array("f", b)]
    if space == "l2":
        return math.fsum((x - y) ** 2 for x, y in zip(a, b))
    dot = math.fsum(x * y for x, y in zip(a, b))
    if space == "ip":
        return 1.0 - dot
    norms = math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b))
    return 1.0 - dot / norms if norms > 0 else 1.0


# ---------------------------------------------------------------------------
# VD1 -- encrypted at rest
# ---------------------------------------------------------------------------
def test_vd1_the_store_is_encrypted_at_rest_under_the_platforms_key(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        vdb, path = world.vdb, world.path
        collection = vdb.Client(path, "knowledge").get_or_create_collection("papers")
        text, field, vector = "quokka meridian 4471 on the shore", "kestrel-ledger.pdf", [0.25, -1.5, 3.0, 7.75]
        needles = (text.encode(), field.encode(), array("f", vector).tobytes())
        with vdb._connection(path, write=False) as reader:
            assert reader is not None, "control: a reader holds the log open"
            collection.add(ids=["a"], embeddings=[vector], documents=[text], metadatas=[{"source_file": field}])
            log = path.with_name(path.name + "-wal")
            assert log.stat().st_size > 0, "control: the write is in the log"
            assert not any(needle in log.read_bytes() for needle in needles), "the log holds it as written"
        files = sorted(p for p in path.parent.iterdir() if p.name.startswith(path.name))
        assert path.read_bytes()[:16] != _PLAIN, "the store never begins as a SQLite file under a key"
        raw = b"".join(p.read_bytes() for p in files)
        for needle in needles:
            assert needle not in raw, ("written as it is into the store's files", needle)
        assert _items(collection) == [("a", text, {"source_file": field}, vector)], "control: it reads back"
        with vdb._connection(path, write=False) as conn:
            assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2, "temporary data kept in memory"
            assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1, "what a deletion frees is overwritten"
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        world.dbe._get_db_encryption_key = lambda: _OTHER_KEY
        with pytest.raises(RuntimeError, match="key verification failed"):
            vdb.Client(path, "knowledge").get_collection("papers")
        assert raw == b"".join(p.read_bytes() for p in files), "another key's attempt wrote nothing"
        # Where no key loads the store is in plaintext, as every store is, and still keeps its temporary data in
        # memory and overwrites what a deletion frees, whatever the defaults its SQLite was built with: here a
        # build that keeps both off.
        world.dbe._get_db_encryption_key = lambda: None
        plain = tmp_path / "plain" / "vectors.db"
        vdb.Client(plain, "knowledge").get_or_create_collection("papers")
        assert plain.read_bytes()[:16] == _PLAIN, "control: no key, no cipher"
        sqlcipher3 = pytest.importorskip("sqlcipher3")
        bare = sqlcipher3.connect(":memory:")
        assert bare.execute("PRAGMA secure_delete").fetchone()[0] == 0, "control: a build that keeps it off"
        assert bare.execute("PRAGMA temp_store").fetchone()[0] == 0, "control: a build that leaves it to itself"
        bare.close()
        safe_connect = vdb.safe_connect
        vdb.safe_connect = lambda db_path, **options: sqlcipher3.connect(str(db_path), **options)
        try:
            with vdb._connection(plain, write=False) as conn:
                assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2
                assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
        finally:
            vdb.safe_connect = safe_connect
        # A store on a rollback journal, as the offline encryption's copy leaves it, returns to its log.
        world.dbe._get_db_encryption_key = lambda: _KEY
        conn = world.loaded["opti_oignon.db_utils"].safe_connect(path)
        try:
            assert conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0] == "delete"
        finally:
            conn.close()
        with vdb._connection(path, write=False) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete", "control: on its rollback journal"
        collection.upsert(ids=["a"], embeddings=[vector])
        with vdb._connection(path, write=False) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal", "back on its write-ahead log"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD2 -- ChromaDB's contract
# ---------------------------------------------------------------------------
def test_vd2_chromadbs_contract_as_the_five_modules_use_it(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        vdb, path = world.vdb, world.path
        knowledge, documents = vdb.Client(path, "knowledge"), vdb.Client(path, "documents")
        papers = knowledge.get_or_create_collection("papers", metadata={"hnsw:space": "l2", "description": "d"})
        twin = documents.get_or_create_collection("papers")
        assert papers.space == "l2" and twin.space == "l2", "the space named, or squared L2, ChromaDB's default"
        assert knowledge.get_or_create_collection("cosine-said", metadata={"hnsw:space": "cosine"}).space == "cosine"
        knowledge.delete_collection("cosine-said")
        assert knowledge.get_or_create_collection("papers", metadata={"hnsw:space": "ip"}).space == "l2", (
            "a collection held keeps its metadata")
        assert papers.metadata == {"hnsw:space": "l2", "description": "d"} and twin.metadata is None
        with pytest.raises(vdb.CollectionExists):
            knowledge.create_collection("papers")
        with pytest.raises(vdb.CollectionNotFound):
            knowledge.get_collection("absent")
        with pytest.raises(vdb.CollectionNotFound):
            knowledge.delete_collection("absent")
        assert issubclass(vdb.CollectionNotFound, ValueError) and issubclass(vdb.CollectionExists, ValueError)
        knowledge.create_collection("notes")
        assert [c.name for c in knowledge.list_collections()] == ["papers", "notes"]
        assert [c.name for c in documents.list_collections()] == ["papers"], "a namespace lists its own"
        papers.add(ids=["a", "b", "c"], embeddings=[[1, 0], [0, 1], [1, 1]], documents=["A", "B", None],
                   metadatas=[{"k": 1, "drop": None}, None, {"s": "x"}])
        assert papers.count() == 3 and twin.count() == 0, "two namespaces apart"
        got = papers.get()
        assert got["ids"] == ["a", "b", "c"] and got["documents"] == ["A", "B", None]
        assert got["metadatas"] == [{"k": 1}, None, {"s": "x"}] and got["embeddings"] is None, (
            "metadatas and documents by default, a None value never kept", got)
        assert papers.get(include=[]) == {"ids": ["a", "b", "c"], "embeddings": None, "documents": None,
                                          "metadatas": None, "included": []}
        assert papers.get(ids=["c", "zz", "a"], include=["embeddings"])["embeddings"] == [[1.0, 1.0], [1.0, 0.0]], (
            "in the order the ids name them, an id not held skipped")
        assert papers.get(limit=1, offset=1)["ids"] == ["b"]
        papers.add(ids=["d", "a"], embeddings=[[5, 5], [9, 9]], documents=["D", "not A"])
        assert _items(papers, ids=["a", "d"]) == [("a", "A", {"k": 1}, [1.0, 0.0]), ("d", "D", None, [5.0, 5.0])], (
            "an id held is skipped, the item kept as it is; the others added")
        papers.delete(ids=["d"])
        assert papers.count() == 3
        papers.upsert(ids=["a", "e"], embeddings=[[2, 2], [3, 3]], documents=[None, "E"],
                      metadatas=[{"k": None, "n": 2}, {"t": True}])
        assert _items(papers, ids=["a", "e"]) == [("a", "A", {"n": 2}, [2.0, 2.0]), ("e", "E", {"t": True}, [3.0, 3.0])], (
            "an upsert replaces the vector, keeps a document not handed in, merges the metadata, a None removing")
        assert papers.get()["ids"] == ["a", "b", "c", "e"], "an item upserted keeps its place"
        papers.delete(ids=["b"])
        papers.delete(where={"s": "x"})
        assert papers.get()["ids"] == ["a", "e"]
        with pytest.raises(ValueError, match="never the whole collection"):
            papers.delete()
        hits = papers.query(query_embeddings=[[2, 2]], n_results=5)
        assert set(hits) == {"ids", "distances", "documents", "metadatas", "embeddings", "included"}
        assert hits["ids"] == [["a", "e"]] and hits["documents"] == [["A", "E"]] and hits["embeddings"] is None
        assert hits["metadatas"] == [[{"n": 2}, {"t": True}]] and hits["distances"][0][0] == 0.0
        with pytest.raises(ValueError, match="include names"):
            papers.get(include=["uris"])
        for name in ("ab", "a" * 513, "-ab", "ab-", "a..b", "a b", "10.0.0.1", 5):
            with pytest.raises(ValueError, match="no collection name"):
                knowledge.get_or_create_collection(name)
        for metadata in ({"hnsw:space": "manhattan"}, {"x": [1]}, {"x": float("nan")}, ["x"]):
            with pytest.raises(ValueError):
                knowledge.get_or_create_collection("other", metadata=metadata)
        for bad in ({"x": [1]}, {"x": {"y": 1}}, {"x": float("inf")}, {1: "x"}):
            with pytest.raises(ValueError):
                papers.add(ids=["z"], embeddings=[[0, 0]], metadatas=[bad])
        with pytest.raises(ValueError):
            papers.add(ids=["z"], embeddings=[[0, 0]], documents=[3])
        with pytest.raises(ValueError, match="twice"):
            papers.upsert(ids=["z", "z"], embeddings=[[0, 0], [0, 0]])
        assert papers.count() == 2
        knowledge.delete_collection("papers")
        with pytest.raises(vdb.CollectionNotFound):
            papers.count()
        again = knowledge.create_collection("papers")
        assert papers.count() == 0 and again.count() == 0, "a collection is found by its name at each call"
        assert twin.count() == 0 and [c.name for c in documents.list_collections()] == ["papers"]
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD3 -- the filters
# ---------------------------------------------------------------------------
def test_vd3_each_filter_the_store_reads_and_every_other_refused_by_name(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        vdb = world.vdb
        collection = vdb.Client(world.path, "documents").get_or_create_collection("documents")
        rows = {"a": {"source_file": "/x.py", "n": 1, "f": 1.5, "b": True},
                "b": {"source_file": "/y.py", "n": 2, "b": False},
                "c": {"source_file": "/x.py", "n": 3.0},
                "d": None,
                "e": {"source_file": "1", "n": "1"}}
        collection.add(ids=list(rows), embeddings=[[float(i), 1.0] for i in range(len(rows))],
                       metadatas=list(rows.values()))

        def kept(where):
            return collection.get(where=where, include=[])["ids"]

        assert kept({"source_file": "/x.py"}) == kept({"source_file": {"$eq": "/x.py"}}) == ["a", "c"]
        assert kept({"source_file": {"$ne": "/x.py"}}) == ["b", "d", "e"], "a field an item lacks is different"
        assert kept({"source_file": {"$in": ["/y.py", "1"]}}) == ["b", "e"]
        assert kept({"source_file": {"$nin": ["/y.py", "1"]}}) == ["a", "c", "d"]
        assert kept({"n": {"$gt": 1}}) == ["b", "c"] and kept({"n": {"$gte": 1}}) == ["a", "b", "c"]
        assert kept({"n": {"$lt": 3}}) == ["a", "b"] and kept({"n": {"$lte": 3}}) == ["a", "b", "c"]
        assert kept({"n": 1}) == ["a"], "a number is never a string, whatever it reads as"
        assert kept({"n": 3}) == ["c"], "an integer and a float of one value are equal"
        assert kept({"b": True}) == ["a"] and kept({"n": True}) == [], "a boolean is never a number"
        assert kept({"$and": [{"source_file": "/x.py"}, {"n": {"$gt": 2}}]}) == ["c"]
        assert kept({"$or": [{"b": False}, {"f": {"$lt": 2}}, {"source_file": "1"}]}) == ["a", "b", "e"]
        assert kept({"$and": [{"$or": [{"n": 1}, {"n": 2}]}, {"b": {"$ne": True}}]}) == ["b"]
        assert collection.query(query_embeddings=[[0, 1]], n_results=5, where={"n": {"$gte": 2}},
                                include=[])["ids"] == [["b", "c"]], "a query is filtered the same way"
        refused = (({"source_file": {"$contains": "x"}}, r"source_file: '\$contains' is no comparison this store reads"),
                   ({"source_file": {"$regex": "x"}}, r"'\$regex' is no comparison this store reads"),
                   ({"$not": [{"n": 1}]}, r"'\$not' is no field and no join this store reads"),
                   ({"source_file": "/x.py", "n": 1}, "exactly one field or join"),
                   ({"$and": [{"n": 1}]}, r"\$and joins a list of two filters or more"),
                   ({"$or": {"n": 1}}, r"\$or joins a list of two filters or more"),
                   ({}, "exactly one field or join"), ([{"n": 1}], "exactly one field or join"),
                   ({"n": {"$gt": 1, "$lt": 3}}, "n: one comparison per field"),
                   ({"n": {"$gt": "1"}}, r"n: \$gt compares with a finite number"),
                   ({"n": {"$gt": True}}, r"n: \$gt compares with a finite number"),
                   ({"n": {"$in": []}}, r"n: \$in takes a non-empty list"),
                   ({"n": {"$in": "1"}}, r"n: \$in takes a non-empty list"),
                   ({"n": [1]}, r"n: \$eq compares with a string, a number or a boolean"),
                   ({"n": {"$eq": [1]}}, r"n: \$eq compares with a string, a number or a boolean"),
                   ({"n": None}, r"n: \$eq compares with a string, a number or a boolean"))
        for where, said in refused:
            with pytest.raises(ValueError, match=said):
                kept(where)
            with pytest.raises(ValueError, match=said):
                collection.delete(where=where)
        assert collection.count() == 5, "a filter refused deletes nothing"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD4 -- no vector computed
# ---------------------------------------------------------------------------
def test_vd4_the_store_computes_no_vector_and_says_so(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        collection = world.vdb.Client(world.path, "knowledge").get_or_create_collection("papers")
        collection.add(ids=["a"], embeddings=[[1.0, 0.0]], documents=["kept"])
        for write in (collection.add, collection.upsert):
            with pytest.raises(ValueError, match="no embeddings: this store computes no vector"):
                write(ids=["b"], documents=["no vector"])
            with pytest.raises(ValueError, match="no embeddings"):
                write(ids=["a"], documents=["changed"], embeddings=None)
            with pytest.raises(ValueError, match="an embedding is"):
                write(ids=["b"], embeddings=[[]])
            with pytest.raises(ValueError, match="an embedding is"):
                write(ids=["b"], embeddings=[[1.0, float("nan")]])
            with pytest.raises(ValueError, match="an embedding is"):
                write(ids=["b"], embeddings=[["1.0", "2.0"]])
            with pytest.raises(ValueError, match="1 embeddings for 2 ids"):
                write(ids=["b", "c"], embeddings=[[1.0, 0.0]])
        with pytest.raises(ValueError, match="a query by text"):
            collection.query(query_texts=["a question"], n_results=1)
        with pytest.raises(ValueError, match="needs query_embeddings"):
            collection.query(n_results=1)
        with pytest.raises(ValueError, match="needs query_embeddings"):
            collection.query(query_embeddings=[], n_results=1)
        assert _items(collection) == [("a", "kept", None, [1.0, 0.0])], "nothing written by a refused call"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD5 -- an exact search
# ---------------------------------------------------------------------------
def test_vd5_the_search_is_exact_nearest_first_ties_in_order(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        vdb = world.vdb
        client = vdb.Client(world.path, "knowledge")
        vectors = {"p": [0.5, 2.0, -1.0], "q": [3.0, -0.25, 0.75], "r": [-1.5, 1.0, 2.5], "s": [0.0, 0.0, 0.0],
                   "t": [1.0, 4.0, -2.0]}
        query = [1.0, 0.5, -0.5]
        for space in vdb.SPACES:
            collection = client.get_or_create_collection(f"space-{space}", metadata={"hnsw:space": space})
            collection.add(ids=list(vectors), embeddings=list(vectors.values()),
                           metadatas=[{"i": i} for i in range(len(vectors))])
            hits = collection.query(query_embeddings=[query], n_results=len(vectors), include=["distances"])
            want = sorted((_fsum_distance(space, v, query), list(vectors).index(k), k) for k, v in vectors.items())
            assert hits["ids"][0] == [k for _d, _i, k in want], (space, hits)
            assert all(abs(got - d) <= 1e-12 for got, (d, _i, _k) in zip(hits["distances"][0], want)), (space, hits)
            if space == "cosine":
                assert dict(zip(hits["ids"][0], hits["distances"][0]))["s"] == 1.0, "a vector of zeros is at 1"
                assert dict(zip(hits["ids"][0], hits["distances"][0]))["t"] == pytest.approx(
                    dict(zip(hits["ids"][0], hits["distances"][0]))["p"], abs=1e-12), "control: t and p point alike"
            nearest = collection.query(query_embeddings=[query], n_results=2, where={"i": {"$gte": 2}}, include=[])
            assert nearest["ids"] == [[k for _d, i, k in want if i >= 2][:2]], (
                "the filter is applied before the nearest are taken", space)
            assert collection.query(query_embeddings=[query, vectors["r"]], n_results=1, include=[])["ids"] == [
                [want[0][2]], ["r"] if space != "ip" else [collection.query(
                    query_embeddings=[vectors["r"]], n_results=1, include=[])["ids"][0][0]]], "one list per query"
        ties = client.get_or_create_collection("ties")
        ties.add(ids=["second", "first"], embeddings=[[1.0, 1.0], [1.0, 1.0]])
        ties.add(ids=["third"], embeddings=[[2.0, 2.0]])
        assert ties.query(query_embeddings=[[1.0, 1.0]], n_results=3, include=[])["ids"] == [["second", "first",
                                                                                             "third"]], (
            "equal distances in the order the items were added")
        with pytest.raises(ValueError, match="dimension 2 does not match"):
            client.get_collection("space-l2").add(ids=["x"], embeddings=[[1.0, 2.0]])
        with pytest.raises(ValueError, match="query embedding dimension 4 does not match"):
            client.get_collection("space-l2").query(query_embeddings=[[1.0, 2.0, 3.0, 4.0]], n_results=1)
        with pytest.raises(ValueError, match="different dimensions"):
            client.get_or_create_collection("fresh").add(ids=["x", "y"], embeddings=[[1.0], [1.0, 2.0]])
        for n in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match="n_results"):
                ties.query(query_embeddings=[[1.0, 1.0]], n_results=n)
        assert vdb.similarity(0.25, "cosine") == 0.75 and vdb.similarity(0.5, "l2") == 0.75
        assert vdb.similarity(-0.5, "ip") == 1.5
        with pytest.raises(ValueError):
            vdb.similarity(0.1, "manhattan")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD6 -- a write elsewhere is seen
# ---------------------------------------------------------------------------
def test_vd6_a_write_by_another_client_is_seen_and_vectors_are_read_once_per_change(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        vdb = world.vdb
        built = []
        real = vdb._Snapshot

        class Counted(real):
            def __init__(self, *args, **kwargs):
                built.append(args[0])
                super().__init__(*args, **kwargs)

        vdb._Snapshot = Counted
        mine = vdb.Client(world.path, "memory").get_or_create_collection("oo_memories",
                                                                         metadata={"hnsw:space": "cosine"})
        theirs = vdb.Client(world.path, "memory").get_collection("oo_memories")
        mine.add(ids=["a"], embeddings=[[1.0, 0.0]])
        assert mine.query(query_embeddings=[[0.0, 1.0]], n_results=5, include=[])["ids"] == [["a"]]
        assert mine.query(query_embeddings=[[0.0, 1.0]], n_results=5, include=[])["ids"] == [["a"]]
        assert len(built) == 1, ("no write between two queries: the vectors read once", built)
        theirs.add(ids=["b"], embeddings=[[0.0, 1.0]])
        assert mine.query(query_embeddings=[[0.0, 1.0]], n_results=5, include=[])["ids"] == [["b", "a"]], (
            "another client's write is found by the next query")
        theirs.delete(ids=["b"])
        assert mine.query(query_embeddings=[[0.0, 1.0]], n_results=5, include=[])["ids"] == [["a"]], (
            "another client's deletion is seen by the next query")
        theirs.upsert(ids=["a"], embeddings=[[-1.0, 0.0]])
        assert mine.query(query_embeddings=[[1.0, 0.0]], n_results=5)["distances"] == [[2.0]]
        assert len(built) == 4 and built == sorted(built) and len(set(built)) == 4, built
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD7 -- one encoder at a time
# ---------------------------------------------------------------------------
def test_vd7_a_query_reads_only_the_active_encoders_vectors(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        vdb = world.vdb
        collection = vdb.Client(world.path, "knowledge").get_or_create_collection("papers")
        collection.add(ids=["far", "near"], embeddings=[[0.0, 1.0], [1.0, 0.1]])
        collection.add(ids=["other"], embeddings=[[1.0, 0.0]])
        with vdb._connection(world.path, write=True) as conn:
            far, other = (conn.execute("SELECT id FROM items WHERE item_id = ?", (name,)).fetchone()[0]
                          for name in ("far", "other"))
            blob = array("f", [1.0, 0.0]).tobytes()
            conn.execute("INSERT INTO vectors (item, encoder, dimension, vector) VALUES (?, 'next', 2, ?)", (far, blob))
            conn.execute("UPDATE vectors SET encoder = 'next' WHERE item = ? AND encoder = ''", (other,))
            conn.execute("UPDATE collections SET generation = generation + 1")
        assert collection.count() == 3, "an item with no vector of the active encoder is counted"
        hits = collection.query(query_embeddings=[[1.0, 0.0]], n_results=5, include=["embeddings", "distances"])
        assert hits["ids"] == [["near", "far"]], ("the other encoder's vectors are never read", hits)
        assert hits["embeddings"][0][1] == [0.0, 1.0], "the active encoder's vector, never the other"
        assert collection.get(ids=["far", "other"], include=["embeddings"])["embeddings"] == [[0.0, 1.0], None]
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD8 -- the real wiring of the five modules
# ---------------------------------------------------------------------------
class _Embedder:
    """Embeds by a table of texts, every other text to the same vector."""

    def __init__(self, table, fallback=(0.0, 0.0, 1.0)):
        self.table = table
        self.fallback = list(fallback)

    def embed(self, texts, show_progress=False):
        return [list(self.table.get(text, self.fallback)) for text in texts]

    def embed_single(self, text):
        return list(self.table.get(text, self.fallback))


def _chunk(chunk_id, content, **metadata):
    return SimpleNamespace(chunk_id=chunk_id, content=content, metadata=metadata)


def test_vd8_the_five_modules_write_and_read_one_store_each_in_its_namespace(tmp_path, monkeypatch):
    table = {"cats purr": [1.0, 0.0, 0.0], "dogs bark": [0.0, 1.0, 0.0], "about cats": [0.9, 0.1, 0.0],
             "dogs and wolves": [0.1, 0.9, 0.0]}
    embeddings = types.ModuleType("opti_oignon.rag.embeddings")
    embeddings.OllamaEmbeddings = lambda *args, **kwargs: _Embedder(table)
    chunkers = types.ModuleType("opti_oignon.rag.chunkers")

    class _Chunker:
        def chunk(self, content, filepath, file_type):
            return [_chunk(f"{Path(filepath).name}-{i}", line, source_file=filepath, file_type=file_type,
                           chunk_index=i) for i, line in enumerate(content.splitlines()) if line.strip()]

    chunkers.get_chunker = lambda file_type: _Chunker()
    world = _window(tmp_path, monkeypatch, targets={
        "opti_oignon.rag_store": source("rag_store.py"),
        "opti_oignon.memory.vector_store": source("memory", "vector_store.py"),
        "opti_oignon.project_context": source("project_context.py"),
        "opti_oignon.rag.config": source("rag", "config.py"),
        "opti_oignon.rag.indexer": source("rag", "indexer.py"),
        "opti_oignon.rag.retriever": source("rag", "retriever.py"),
    }, seeded={"opti_oignon.rag.embeddings": embeddings, "opti_oignon.rag.chunkers": chunkers},
        packages=("opti_oignon.memory", "opti_oignon.rag"))
    try:
        path, loaded = world.path, world.loaded
        # The RAG store, its chunks in the namespace "knowledge".
        rag = loaded["opti_oignon.rag_store"]
        store = rag.RAGVectorStore(data_dir=str(tmp_path / "rag"))
        assert store.vectors_path == path and not store.chroma_dir.exists(), "ChromaDB's folder is never made"
        store._embedder = _Embedder(table)
        store._chunker = SimpleNamespace(chunk_text=lambda text, source=None, file_type=None, doc_id=None: (
            SimpleNamespace(chunks=[_chunk(f"{doc_id}-0", "cats purr", parent_doc_id=doc_id, chunk_index=0),
                                    _chunk(f"{doc_id}-1", "dogs bark", parent_doc_id=doc_id, chunk_index=1)],
                            chunk_count=2, raw_text_length=18, source_file=source, file_type=file_type,
                            doc_id=doc_id)))
        store.ingest_text("cats purr. dogs bark.", collection="papers", doc_id="d1")
        answer = store.query("about cats", collection="papers", n_results=1, min_score=-1.0, rerank=False,
                             track_citations=False)
        assert [r.content for r in answer.results] == ["cats purr"], answer
        # The memory's vector layer, in "memory".
        memory = loaded["opti_oignon.memory.vector_store"]
        layer = memory.MemoryVectorStore(vectors_db=path, embedder=_Embedder(table))
        assert layer.collection is not None and layer.health()["status"] == "ok"
        layer.add("f1", "dogs bark", user_id="local")
        layer.add("f2", "cats purr", user_id="other")
        assert [m.id for m in layer.find_similar([0.0, 1.0, 0.0], user_id="local")] == ["f1"]
        assert layer.update("f1", text="dogs and wolves") and layer.get("f1")["embedding"] == pytest.approx(
            [0.1, 0.9, 0.0], abs=1e-6)
        assert layer.count() == 2 and layer.count(user_id="other") == 1 and layer.clear(user_id="other") == 1
        # The project index and its context builder, in "projects".
        project = loaded["opti_oignon.project_context"]
        monkeypatch.setattr(project, "EMBEDDINGS_AVAILABLE", True)
        indexer = project.ProjectIndexer(vectors_db=path)
        indexer._get_collection("p1").add(ids=["p1::a::0", "p1::a::1"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                                          documents=["cats purr", "dogs bark"],
                                          metadatas=[{"source_file": "a", "chunk_index": 0},
                                                     {"source_file": "a", "chunk_index": 1}])
        builder = project.ProjectContextBuilder(vectors_db=path)
        builder._embedder, builder._embedder_checked = _Embedder(table), True
        chunks = builder.retrieve_chunks("p1", "about cats", max_chunks=1, min_score=-1.0)
        assert [c.content for c in chunks] == ["cats purr"], chunks
        assert indexer.delete_project_index("p1") and builder.retrieve_chunks("p1", "about cats") == []
        # The documents' indexer and retriever, in "documents".
        config = loaded["opti_oignon.rag.config"].RAGConfig(
            rag_home=tmp_path / "home", chroma_dir=tmp_path / "home" / "chroma_db", vectors_db=path,
            cache_dir=tmp_path / "home" / "cache", logs_dir=tmp_path / "home" / "logs")
        assert not config.chroma_dir.exists(), "ChromaDB's folder is never made"
        doc = tmp_path / "notes.txt"
        doc.write_text("cats purr\ndogs bark\n", encoding="utf-8")
        indexer = loaded["opti_oignon.rag.indexer"].DocumentIndexer(config=config)
        assert indexer.index_file(doc) == (2, 0)
        retriever = loaded["opti_oignon.rag.retriever"].DocumentRetriever(config=config)
        found = retriever.search("about cats", n_results=1, min_score=-1.0)
        assert [r.content for r in found] == ["cats purr"], found
        assert indexer.index_file(doc, force=True) == (2, 0) and retriever.count() == 2, "re-indexed, not doubled"
        # One file, four namespaces apart.
        vdb = world.vdb
        with vdb._connection(path, write=False) as conn:
            held = sorted(conn.execute("SELECT namespace, name FROM collections").fetchall())
        assert held == [("documents", "documents"), ("knowledge", "papers"), ("memory", "oo_memories")], held
        beside = {p.name.removesuffix("-wal").removesuffix("-shm") for p in path.parent.iterdir()}
        assert beside == {"rag_documents.db", "vectors.db"}, ("no other file beside them", beside)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD9 -- a score is a similarity in the collection's own space
# ---------------------------------------------------------------------------
def test_vd9_the_retriever_and_the_project_builder_score_in_their_collections_space(tmp_path, monkeypatch):
    embeddings = types.ModuleType("opti_oignon.rag.embeddings")
    embeddings.OllamaEmbeddings = lambda *args, **kwargs: _Embedder({"q": [1.0, 0.0]})
    chunkers = types.ModuleType("opti_oignon.rag.chunkers")
    chunkers.get_chunker = lambda file_type: None
    world = _window(tmp_path, monkeypatch, targets={
        "opti_oignon.project_context": source("project_context.py"),
        "opti_oignon.rag.config": source("rag", "config.py"),
        "opti_oignon.rag.retriever": source("rag", "retriever.py"),
    }, seeded={"opti_oignon.rag.embeddings": embeddings, "opti_oignon.rag.chunkers": chunkers},
        packages=("opti_oignon.rag",))
    try:
        vdb, path, loaded = world.vdb, world.path, world.loaded
        vector = [0.6, 0.8]
        for space, score in (("cosine", 0.6), ("l2", 0.6), ("ip", 0.6)):
            documents = vdb.Client(path, "documents")
            if space != "cosine":
                documents.delete_collection("documents")
            documents.get_or_create_collection("documents", metadata={"hnsw:space": space}).add(
                ids=["x"], embeddings=[vector], documents=["text"], metadatas=[{"source_file": "/f"}])
            config = loaded["opti_oignon.rag.config"].RAGConfig(
                rag_home=tmp_path / "home", chroma_dir=tmp_path / "home" / "chroma_db", vectors_db=path,
                cache_dir=tmp_path / "home" / "cache", logs_dir=tmp_path / "home" / "logs")
            found = loaded["opti_oignon.rag.retriever"].DocumentRetriever(config=config).search("q", min_score=-1.0)
            assert [r.score for r in found] == [pytest.approx(score, abs=1e-6)], (space, found)
            project = loaded["opti_oignon.project_context"]
            monkeypatch.setattr(project, "EMBEDDINGS_AVAILABLE", True)
            projects = vdb.Client(path, "projects")
            if space != "cosine":
                projects.delete_collection("project_p1")
            projects.get_or_create_collection("project_p1", metadata={"hnsw:space": space}).add(
                ids=["y"], embeddings=[vector], documents=["text"], metadatas=[{"source_file": "/f"}])
            builder = project.ProjectContextBuilder(vectors_db=path)
            builder._embedder, builder._embedder_checked = _Embedder({"q": [1.0, 0.0]}), True
            assert [c.score for c in builder.retrieve_chunks("p1", "q", min_score=-1.0)] == [
                pytest.approx(score, abs=1e-6)], space
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD10 -- one file, where every module opens it
# ---------------------------------------------------------------------------
def test_vd10_every_module_opens_the_store_where_the_migration_writes_it(tmp_path, monkeypatch):
    package_data = tmp_path / "package" / "data"
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = package_data
    rag_config = types.ModuleType("opti_oignon.rag.config")
    rag_config.get_config = lambda: SimpleNamespace(vectors_db=Path("/elsewhere/vectors.db"))
    world = _window(tmp_path, monkeypatch, targets={
        "opti_oignon.chroma_migration": source("chroma_migration.py"),
        "opti_oignon.rag_store": source("rag_store.py"),
    }, seeded={"opti_oignon.config": config})
    try:
        vdb, dbe, loaded = world.vdb, world.dbe, world.loaded
        package = source().resolve()
        want = package / "data" / "rag" / vdb.FILENAME
        dbe._STORE_PLACES = None
        assert loaded["opti_oignon.chroma_migration"].target() == want == dbe._places()[1] / "rag" / vdb.FILENAME
        rag_module = ast.parse(source("rag", "config.py").read_text(encoding="utf-8"))
        bound = {node.targets[0].id: ast.unparse(node.value) for node in rag_module.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        assert bound["STORE_DIR"] == "_MODULE_DIR.parent / 'data' / 'rag'" and bound[
            "VECTORS_DB"] == "STORE_DIR / 'vectors.db'", "the documents' index: the package's place, no variable"
        assert source("rag", "config.py").parent.parent / "data" / "rag" / "vectors.db" == want
        project = ast.parse(source("project_context.py").read_text(encoding="utf-8"))
        bound = {node.targets[0].id: ast.unparse(node.value) for node in project.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        assert bound["_DATA_DIR"] == "Path(__file__).parent / 'data'"
        assert bound["_DEFAULT_VECTORS_DB"] == "_DATA_DIR / 'rag' / 'vectors.db'", "the projects' index"
        config_module = ast.parse(source("config.py").read_text(encoding="utf-8"))
        bound = {node.targets[0].id: ast.unparse(node.value) for node in config_module.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        assert bound["PROJECT_ROOT"] == "Path(__file__).parent" and bound["DATA_DIR"] == "PROJECT_ROOT / 'data'", (
            "the platform's DATA_DIR is the package's place of data", bound.get("DATA_DIR"))
        rag_module = ast.parse(source("rag_store.py").read_text(encoding="utf-8"))
        init = next(node for node in ast.walk(rag_module) if isinstance(node, ast.ClassDef)
                    and node.name == "RAGVectorStore").body
        init = ast.unparse(next(node for node in init if isinstance(node, ast.FunctionDef) and node.name == "__init__"))
        assert "data_dir = Path(DATA_DIR) / 'rag'" in init and "self.data_dir / vector_db.FILENAME" in init, (
            "the RAG store's default folder: DATA_DIR's rag folder")
        store = loaded["opti_oignon.rag_store"].RAGVectorStore()
        assert store.vectors_path == package_data / "rag" / vdb.FILENAME, "the RAG store: DATA_DIR's rag folder"
        assert store._chroma.path == store.vectors_path and store._chroma.namespace == "knowledge"
        memory = ast.parse(source("memory", "vector_store.py").read_text(encoding="utf-8"))
        defaults = [node for node in ast.walk(memory) if isinstance(node, ast.FunctionDef)
                    and node.name == "_default_vectors_db"]
        assert len(defaults) == 1 and "config = get_config()" in ast.unparse(defaults[0]) and (
            "return (Path(config.vectors_db), Path(config.chroma_dir))" in ast.unparse(defaults[0])), (
            "the memory: the index's store, and the index's folder of ChromaDB as the one it waits for")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD11 -- chromadb is read, never run
# ---------------------------------------------------------------------------
def test_vd11_chromadb_is_imported_by_the_migration_alone_its_telemetry_off_asked_no_embedding():
    package = source()
    importers = {}
    for path in package_modules().values():
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name == "chromadb" or name.startswith("chromadb.") for name in names):
                importers.setdefault(Path(path).relative_to(package).as_posix(), []).append(node)
    assert sorted(importers) == ["chroma_migration.py"], sorted(importers)
    tree = ast.parse(source("chroma_migration.py").read_text(encoding="utf-8"))
    top = {id(node) for node in tree.body}
    assert all(id(node) not in top for node in importers["chroma_migration.py"]), "imported inside a function"
    clients = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ast.unparse(node.func).endswith("PersistentClient")]
    assert len(clients) == 1 and "Settings(anonymized_telemetry=False)" in ast.unparse(clients[0]), (
        "its one client built with its telemetry off")
    asked = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                "add", "upsert", "update", "query", "get"):
            keywords = {k.arg for k in node.keywords}
            assert None not in keywords and "query_texts" not in keywords and "documents" not in keywords, (
                ast.unparse(node))
            if node.func.attr == "query":
                assert "query_embeddings" in keywords, ast.unparse(node)
            if node.func.attr in ("add", "upsert", "update"):
                asked.add(ast.unparse(node))
    assert not asked, ("nothing written into ChromaDB", asked)
    assert all("query_texts" not in getattr(node, "value", "") for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str))


# ---------------------------------------------------------------------------
# VD12 -- the RAG store never stores what it cannot search
# ---------------------------------------------------------------------------
def test_vd12_the_rag_store_refuses_what_it_cannot_embed_and_searches_nothing_without_a_vector(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch, targets={"opti_oignon.rag_store": source("rag_store.py")})
    try:
        rag = world.loaded["opti_oignon.rag_store"]
        store = rag.RAGVectorStore(data_dir=str(tmp_path / "rag"))
        chunks = [_chunk("i0", "c0", parent_doc_id="d"), _chunk("i1", "c1", parent_doc_id="d")]
        for embedder, said in ((None, "no embedder for the 2 chunk"), (SimpleNamespace(
                embed=lambda texts, show_progress=False: [None for _ in texts]), "every embedding failed")):
            store._get_embedder = lambda embedder=embedder: embedder
            with pytest.raises(rag.EmbeddingsUnavailable, match=said):
                store._store_chunks(chunks, "papers")
        assert issubclass(rag.EmbeddingsUnavailable, RuntimeError)
        assert store._chroma.get_collection("papers").count() == 0, "nothing stored"
        store._get_embedder = lambda: SimpleNamespace(
            embed=lambda texts, show_progress=False: [[1.0, 0.0], None], embed_single=lambda text: [1.0, 0.0])
        store._store_chunks(chunks, "papers")
        assert store._chroma.get_collection("papers").get()["ids"] == ["i0"], "a chunk with no embedding dropped"
        asked = []
        query = world.vdb.Collection.query
        monkeypatch.setattr(world.vdb.Collection, "query", lambda self, *a, **k: asked.append(k) or query(self, *a, **k))
        store._get_embedder = lambda: SimpleNamespace(embed_single=lambda text: None)
        assert store.query("anything", collection="papers", min_score=-1.0).results == [], "no vector, no search"
        assert asked == [], "the store is never asked to search without a vector"
        store._get_embedder = lambda: SimpleNamespace(embed_single=lambda text: [-1.0, 0.0])
        far = store.query("anything", collection="papers", min_score=-1.0, rerank=False, track_citations=False)
        assert [r.score for r in far.results] == [0.0], "a distance above 1 scores 0, never below"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD13 -- the memory says what it lacks
# ---------------------------------------------------------------------------
def test_vd13_without_the_store_the_memorys_health_names_it_and_writes_are_no_ops(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch, targets={
        "opti_oignon.memory.vector_store": source("memory", "vector_store.py")}, packages=("opti_oignon.memory",))
    try:
        memory = world.loaded["opti_oignon.memory.vector_store"]
        monkeypatch.setattr(memory, "_HAS_VECTOR_STORE", False)
        layer = memory.MemoryVectorStore(embedder=_Embedder({}))
        health = layer.health()
        assert layer.collection is None and health["status"] == "unavailable" and health["available"] is False
        assert "the knowledge base's store could not be opened" in health["detail"] and "chromadb" not in health[
            "detail"], health
        assert layer.add("f1", "x") == "f1" and layer.get("f1") is None and layer.count() == 0
        assert layer.update("f1", text="y") is False and layer.delete("f1") is False and layer.clear() == 0
        assert layer.find_similar([1.0, 0.0]) == [] and not world.path.exists()
        monkeypatch.setattr(memory, "_HAS_VECTOR_STORE", True)
        (tmp_path / "a-file").write_text("not a directory", encoding="utf-8")
        unopened = memory.MemoryVectorStore(vectors_db=tmp_path / "a-file" / "vectors.db", embedder=_Embedder({}))
        assert unopened.collection is None and "could not be opened" in unopened.health()["detail"], (
            "a store that cannot be opened degrades the memory, never crashes it")
        layer = memory.MemoryVectorStore(vectors_db=world.path)
        assert layer._embedder is None, "control: no embedder in the window"
        layer.add("f1", "x", embedding=[1.0, 0.0])
        with world.vdb._connection(world.path, write=True) as conn:
            conn.execute("UPDATE vectors SET encoder = 'next'")
            conn.execute("UPDATE collections SET generation = generation + 1")
        assert layer.get("f1")["embedding"] is None, "control: no vector of the active encoder"
        with pytest.raises(RuntimeError, match="no embedding available for the update"):
            layer.update("f1", text="z")
        assert layer.get("f1")["document"] == "x", "an update with no vector writes nothing"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD14 -- nothing made to be read, nothing read of a later schema
# ---------------------------------------------------------------------------
def test_vd14_a_read_makes_no_file_and_a_later_schema_is_refused(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        vdb, path = world.vdb, world.path
        client = vdb.Client(path, "knowledge")
        assert client.list_collections() == []
        with pytest.raises(vdb.CollectionNotFound):
            client.get_collection("papers")
        assert not path.exists() and not path.parent.exists(), "a read creates nothing"
        client.get_or_create_collection("papers").add(ids=["a"], embeddings=[[1.0]])
        with vdb._connection(path, write=True) as conn:
            conn.execute("COMMIT")
            conn.execute(f"PRAGMA user_version = {vdb.SCHEMA_VERSION + 1}")
            conn.execute("BEGIN")
        before = path.read_bytes()
        for call in (client.list_collections, lambda: client.get_collection("papers"),
                     lambda: client.get_or_create_collection("other"),
                     lambda: vdb.Collection(client, "papers").add(ids=["b"], embeddings=[[2.0]])):
            with pytest.raises(RuntimeError, match="later version of the platform"):
                call()
        assert path.read_bytes() == before, "nothing written"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD15 -- never another collection's text
# ---------------------------------------------------------------------------
def test_vd15_a_search_never_reads_another_collections_text_and_no_id_is_given_twice(tmp_path, monkeypatch):
    world = _window(tmp_path, monkeypatch)
    try:
        vdb, path = world.vdb, world.path
        reader, writer, memory = (vdb.Client(path, namespace) for namespace in ("documents", "documents", "memory"))
        writer.get_or_create_collection("documents").add(ids=["old-0", "old-1"], embeddings=[[1.0, 0.0], [0.0, 1.0]],
                                                         documents=["OLD CHUNK 0", "OLD CHUNK 1"])
        found = reader.get_collection("documents")
        assert found.query(query_embeddings=[[1.0, 0.0]], n_results=2)["documents"] == [["OLD CHUNK 0", "OLD CHUNK 1"]]
        with vdb._connection(path, write=False) as conn:
            first = conn.execute("SELECT id FROM collections WHERE name = 'documents'").fetchone()[0]
            items = [row[0] for row in conn.execute("SELECT id FROM items")]
        writer.delete_collection("documents")
        writer.create_collection("documents")
        memory.get_or_create_collection("oo_memories").add(ids=["f1", "f2"], embeddings=[[1.0, 0.0], [0.0, 1.0]],
                                                           documents=["SECRET MEMORY FACT of user B", "another fact"])
        writer.get_collection("documents").add(ids=["new-0"], embeddings=[[1.0, 0.0]], documents=["NEW CHUNK 0"])
        hits = found.query(query_embeddings=[[1.0, 0.0]], n_results=5)
        assert hits["ids"] == [["new-0"]] and hits["documents"] == [["NEW CHUNK 0"]], (
            "a search reads the collection made again, never what a cache kept of the one deleted", hits)
        assert found.get(include=["documents"])["documents"] == ["NEW CHUNK 0"]
        with vdb._connection(path, write=False) as conn:
            second = conn.execute("SELECT id FROM collections WHERE name = 'documents'").fetchone()[0]
            later = [row[0] for row in conn.execute("SELECT id FROM items")]
            assert second > first and min(later) > max(items), ("no id given twice", first, second, items, later)
            secret = conn.execute("SELECT id FROM items WHERE item_id = 'f1'").fetchone()[0]
            assert vdb.Collection._documents(conn, second, [secret]) == {}, "a document read only from its collection"
            assert vdb.Collection._vectors(conn, (second, "l2", None, 2, "", 0), [secret]) == {}, "nor a vector"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD16 -- a migration that waits closes its namespace
# ---------------------------------------------------------------------------
def _chroma_store(folder):
    folder.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(folder / "chroma.sqlite3"))
    try:
        conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()


def test_vd16_a_waiting_migration_closes_its_namespace_and_every_module_says_so(tmp_path, monkeypatch, caplog):
    table = {"cats purr": [1.0, 0.0, 0.0], "about cats": [0.9, 0.1, 0.0]}
    embeddings = types.ModuleType("opti_oignon.rag.embeddings")
    embeddings.OllamaEmbeddings = lambda *args, **kwargs: _Embedder(table)
    chunkers = types.ModuleType("opti_oignon.rag.chunkers")
    chunkers.get_chunker = lambda file_type: None
    home = tmp_path / "rag"
    rag_config = types.ModuleType("opti_oignon.rag.config")
    rag_config.get_config = lambda: SimpleNamespace(vectors_db=home / "vectors.db", chroma_dir=home / "chroma_db")
    real_config = source("rag", "config.py")
    world = _window(tmp_path, monkeypatch, targets={
        "opti_oignon.rag_store": source("rag_store.py"),
        "opti_oignon.memory.vector_store": source("memory", "vector_store.py"),
        "opti_oignon.project_context": source("project_context.py"),
    }, seeded={"opti_oignon.rag.embeddings": embeddings, "opti_oignon.rag.chunkers": chunkers,
               "opti_oignon.rag.config": rag_config}, packages=("opti_oignon.memory", "opti_oignon.rag"))
    try:
        vdb, path, loaded = world.vdb, world.path, world.loaded
        assert vdb.MIGRATE_COMMAND == world.dbe.MIGRATE_COMMAND == "python3 -m opti_oignon.db_encryption migrate"
        for folder in ("chroma_v2", "chroma_db"):
            _chroma_store(home / folder)
        _chroma_store(tmp_path / "project_chroma")
        client = vdb.Client(path, "knowledge", legacy=home / "chroma_v2")
        said = (f"{home / 'chroma_v2'} holds a ChromaDB store this version no longer reads: the knowledge base's "
                "knowledge waits for `python3 -m opti_oignon.db_encryption migrate`, run with the platform stopped")
        for call in (client.list_collections, lambda: client.get_collection("papers"),
                     lambda: client.get_or_create_collection("papers"), lambda: client.create_collection("papers"),
                     lambda: client.delete_collection("papers"), lambda: vdb.Collection(client, "papers").count(),
                     lambda: vdb.Collection(client, "papers").add(ids=["a"], embeddings=[[1.0]]),
                     lambda: vdb.Collection(client, "papers").query(query_embeddings=[[1.0]]), client.ready):
            with pytest.raises(vdb.MigrationPending) as refused:
                call()
            assert str(refused.value).startswith(said), str(refused.value)
        assert issubclass(vdb.MigrationPending, RuntimeError) and not path.exists(), "nothing written"
        # The RAG store: no deletion, no ingestion, no search, each said.
        rag = loaded["opti_oignon.rag_store"]
        store = rag.RAGVectorStore(data_dir=str(home))
        store.db.create_collection("papers")
        store.db.insert_document(rag.IngestedDocument(doc_id="d1", collection_name="papers", source_file="f",
                                                      file_type="text", chunk_count=1, raw_text_length=9,
                                                      ingested_at=1.0))
        with pytest.raises(vdb.MigrationPending):
            store.delete_document("d1")
        with pytest.raises(vdb.MigrationPending):
            store.delete_collection("papers")
        assert store.db.get_document("d1") is not None and [c["name"] for c in store.db.list_collections()] == [
            "papers"], "nothing deleted while the chunks wait in ChromaDB's store"
        store._embedder = _Embedder(table)
        store._chunker = SimpleNamespace(chunk_text=lambda text, source=None, file_type=None, doc_id=None: (
            SimpleNamespace(chunks=[_chunk(f"{doc_id}-0", "cats purr", parent_doc_id=doc_id)], chunk_count=1,
                            raw_text_length=9, source_file=source, file_type=file_type, doc_id=doc_id)))
        with pytest.raises(vdb.MigrationPending):
            store.ingest_text("cats purr", collection="fresh", doc_id="d2")
        with pytest.raises(vdb.MigrationPending):
            store.store_chunked(SimpleNamespace(chunks=[], chunk_count=0, raw_text_length=0, source_file="empty.txt",
                                               file_type="text", doc_id="d3"), collection="fresh")
        fetched = []
        monkeypatch.setattr(store, "_load_web_config", lambda: {"enabled": True})
        monkeypatch.setattr(store, "_fetch_page", lambda url, config: fetched.append(url))
        with pytest.raises(vdb.MigrationPending):
            store.ingest_url("https://example.org/a-page", collection="fresh", doc_id="d4")
        assert fetched == [], "nothing fetched while the knowledge base waits"
        assert store.db.get_document("d2") is None and store.db.get_document("d3") is None
        assert store.db.get_document("d4") is None
        assert [c["name"] for c in store.db.list_collections()] == ["papers"], (
            "no collection recorded either: an ingestion is refused before its first write")
        with caplog.at_level("WARNING"):
            assert store.query("about cats", collection="papers", min_score=-1.0).results == []
        assert any("nothing searched" in r.getMessage() and "waits for" in r.getMessage() for r in caplog.records)
        # The memory degrades, and says why.
        memory = loaded["opti_oignon.memory.vector_store"]
        layer = memory.MemoryVectorStore(embedder=_Embedder(table))
        assert layer.collection is None and "waits for `python3 -m opti_oignon.db_encryption migrate`" in (
            layer.health()["detail"]), layer.health()
        # The project index opens nothing.
        project = loaded["opti_oignon.project_context"]
        monkeypatch.setattr(project, "_DEFAULT_VECTORS_DB", path)
        monkeypatch.setattr(project, "_DEFAULT_CHROMA_BASE", tmp_path / "project_chroma")
        monkeypatch.setattr(project, "EMBEDDINGS_AVAILABLE", True)
        indexer, builder = project.ProjectIndexer(), project.ProjectContextBuilder()
        builder._embedder, builder._embedder_checked = _Embedder(table), True
        assert indexer._get_collection("p1") is None and builder.retrieve_chunks("p1", "about cats") == []
        assert not path.exists(), "nothing written by any of them"
        # The documents' indexer and retriever, in a window of their own over the real configuration.
        documents = _window(tmp_path / "second", monkeypatch, targets={
            "opti_oignon.rag.config": real_config,
            "opti_oignon.rag.indexer": source("rag", "indexer.py"),
            "opti_oignon.rag.retriever": source("rag", "retriever.py"),
        }, seeded={"opti_oignon.rag.embeddings": embeddings, "opti_oignon.rag.chunkers": chunkers},
            packages=("opti_oignon.rag",))
        try:
            config = documents.loaded["opti_oignon.rag.config"].RAGConfig(
                rag_home=home, chroma_dir=home / "chroma_db", vectors_db=path, cache_dir=tmp_path / "cache",
                logs_dir=tmp_path / "logs")
            for build in (documents.loaded["opti_oignon.rag.indexer"].DocumentIndexer,
                          documents.loaded["opti_oignon.rag.retriever"].DocumentRetriever):
                with pytest.raises(documents.vdb.MigrationPending, match="chroma_db holds a ChromaDB store"):
                    build(config=config)
        finally:
            documents.restore()
        # Once the folders have moved, every one of them works.
        for folder in (home / "chroma_v2", home / "chroma_db", tmp_path / "project_chroma"):
            folder.rename(folder.with_name(folder.name + "-moved"))
        assert client.list_collections() == [] and client.get_or_create_collection("papers").count() == 0
        store.ingest_text("cats purr", collection="papers", doc_id="d2")
        assert [r.content for r in store.query("about cats", collection="papers", min_score=-1.0,
                                               rerank=False, track_citations=False).results] == ["cats purr"]
        assert memory.MemoryVectorStore(embedder=_Embedder(table)).collection is not None
        assert project.ProjectIndexer()._get_collection("p1") is not None
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# VD17 -- the routes say it
# ---------------------------------------------------------------------------
def test_vd17_the_rag_routes_answer_a_waiting_migration_with_503_and_its_reason(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    world = _window(tmp_path, monkeypatch, targets={
        "opti_oignon.rag_store": source("rag_store.py"),
        "opti_oignon.api.routes_rag": source("api", "routes_rag.py"),
    }, packages=("opti_oignon.api",))
    try:
        home = tmp_path / "rag"
        _chroma_store(home / "chroma_v2")
        rag, routes = world.loaded["opti_oignon.rag_store"], world.loaded["opti_oignon.api.routes_rag"]
        store = rag.RAGVectorStore(data_dir=str(home))
        store.db.create_collection("papers")
        store.db.insert_document(rag.IngestedDocument(doc_id="d1", collection_name="papers", source_file="f",
                                                      file_type="text", chunk_count=1, raw_text_length=9,
                                                      ingested_at=1.0))
        monkeypatch.setattr(routes, "_get_store", lambda: store)
        for call in (lambda: routes.create_collection(routes.CollectionCreateRequest(name="fresh", description="")),
                     lambda: routes.delete_collection("papers"), lambda: routes.delete_document("d1")):
            with pytest.raises(fastapi.HTTPException) as answer:
                call()
            assert answer.value.status_code == 503, answer.value
            assert f"{home / 'chroma_v2'} holds a ChromaDB store" in answer.value.detail, answer.value.detail
            assert "waits for `python3 -m opti_oignon.db_encryption migrate`" in answer.value.detail
        assert [c["name"] for c in store.db.list_collections()] == ["papers"] and store.db.get_document("d1"), (
            "nothing created, nothing deleted")
        assert not (home / "vectors.db").exists()
    finally:
        world.restore()
