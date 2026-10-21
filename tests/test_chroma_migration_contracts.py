#!/usr/bin/env python3
"""Contracts for the move of ChromaDB's stores into the knowledge base's store: offline, all of them or none.

``python3 -m opti_oignon.db_encryption migrate`` (``chroma_migration``)
reads every ChromaDB store below the places through chromadb, writes its
collections into the knowledge base's store in one transaction, proves the
copy before it commits, and moves each ChromaDB folder into a folder of
plaintext copies beside it. chromadb is a stand-in here: it serves the
collections each test writes for it, from folders that hold a real
``chroma.sqlite3`` in plaintext, where ``check`` finds them.

  * CM1 -- WHERE EACH COLLECTION GOES: the folders the command knows, each
    to its namespace, the memory's collection to its own; a ChromaDB store in
    another folder said and left as it is; a folder of plaintext copies
    passed over, in a store folder or anywhere else.
  * CM2 -- EVERYTHING, PROVED, THEN MOVED: every item of every collection --
    its document, its metadata, its vector bit for bit -- in the store,
    encrypted, each collection in its space; each folder kept in a folder of
    plaintext copies, said with the store; the command says it.
  * CM3 -- REFUSED, NOTHING WRITTEN AND NOTHING MOVED: chromadb that does
    not import; the platform running, or a process holding a file; the store
    in plaintext while a key loads; a space or a dimension that differs from
    the one of a collection the store holds with items in it -- an empty one
    takes the space it is given; an item with no vector; a collection that
    does not read whole; a vector float32 does not hold; a space the store
    does not measure; a value the store does not keep. The tree is as it was,
    byte for byte (chromadb is a stand-in here, which writes nothing; the
    real one rewrites some of its own files as it reads them).
  * CM4 -- THE PROOF REFUSES: ChromaDB's distances not found again, a copy
    that does not read back: nothing written.
  * CM5 -- WHAT THE PLATFORM WROTE SINCE IS KEPT: an id the store holds is
    kept as it is, and counted, the memory's facts, read first, never judging
    another namespace's items; a folder that cannot be moved once the store
    has committed stops the run, said after what the store took -- in that
    order in one log, and a stdout closed early costs neither the line nor
    the 3 --, and the next run moves it, every item kept.
  * CM6 -- A SIGNAL: before the commit it stops the run, nothing changed;
    after it, the run finishes and says so, or, stopped by a folder that
    does not move, says it went on.
  * CM7 -- CHROMADB'S VERSIONS: a listing of names (0.6) or of collections
    (1.x); the space from the configuration, the metadata, or ChromaDB's
    default, squared L2.
  * CM8 -- SAID BY CHECK, ONE RUN AT A TIME: ``check`` names the command for
    each ChromaDB store the command moves, never for another folder or a
    folder of plaintext copies; another run of ``encrypt`` or ``migrate``
    under way refuses; a relative ``RAG_HOME`` refuses the run.
  * CM9 -- GIT TAKES NO PLAINTEXT COPY: every file the command keeps in a
    folder of plaintext copies, in either place of data, is one git ignores.
  * CM10 -- WHAT THE MEMORY STILL HOLDS: a vector is copied only when an
    active fact of the memory holds it with the text the vector was made of;
    a fact deleted or rewritten since leaves its vector out, counted, and the
    active facts left with no vector are counted and said; with no store of
    facts, every vector of the memory is left out; facts that do not read
    refuse the run, and facts in plaintext while a key loads say to encrypt
    first.
  * CM11 -- NO HALF RUN: a process that starts holding a file while the run
    reads is seen before the commit, nothing written, the store and the
    folder the run made removed, a store found on its rollback journal left
    on it; a store chromadb does not read, an item it returns unlisted, a
    store that does not open under the key: each refused by name, never a
    trace.
  * CM12 -- THE REAL CHROMADB: in a process of its own, network refused and
    its home in a temporary folder, the installed chromadb makes ChromaDB's
    stores as the five modules made them, and the command moves every item of
    them, encrypted, each collection in the space ChromaDB measured it in.
    Skipped where chromadb does not import.
  * CM13 -- ONE TEST OF WHAT WAITS: a folder of the platform's modules that is
    a symbolic link, or cannot be looked at, closes the platform's namespace
    and refuses the run, both naming it; so does ChromaDB's store the walk of
    the places does not reach; a ``chroma.sqlite3`` that is no database
    closes nothing and leaves the command nothing to move; a repository below
    a folder named like a folder of plaintext copies moves as any other.
  * CM14 -- A RESUMED RUN UNDOES WHAT IT MUST NOT KEEP: a run that commits
    and cannot move its folder says what the store took before it stopped,
    and leaves no empty folder of plaintext copies; after it, a fact of the memory rewritten or deleted has its vector,
    copied by that run, removed by the next one, said and recorded, every
    reader's view of the collection moved on; a vector the store holds with
    the text the fact reads now is kept, whoever wrote it; the active facts
    with no vector of their text are counted from what the store holds, and
    said not counted when the store, committed, does not read back -- the
    folders move all the same; a copy with no text, whose fact is gone, goes
    too.
  * CM15 -- TWO SOURCES OF ONE COLLECTION: two ChromaDB folders that both
    hold the memory's collection, the current copy read first or last, move
    together, the current text kept; and when a run stops between them, the
    next keeps only the copy of the text the fact reads then, from whichever
    folder holds it.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import fcntl
import importlib.util
import io
import json
import math
import os
import signal
import site
import sqlite3
import subprocess
import sys
import types
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_KEY = bytes(range(1, 33))
_OTHER_KEY = bytes(range(33, 65))
_PLAIN = b"SQLite format 3\x00"
_REPO = Path(__file__).resolve().parent.parent


class _Collection:
    """A ChromaDB collection as chromadb hands it out: its items read by page, its nearest found in its space."""

    def __init__(self, name, items, *, metadata=None, configuration=None, space="cosine"):
        self.name = name
        self.metadata = metadata
        self.configuration_json = {"hnsw": {"space": space}} if configuration is None else configuration
        self.items = items
        self.space = space
        self.count_offset = 0
        self.skew = 0.0
        self.on_get = None
        self.reads = []

    def count(self):
        return len(self.items) + self.count_offset

    def get(self, include=None, limit=None, offset=None):
        self.reads.append((tuple(include), limit, offset))
        if self.on_get is not None:
            self.on_get()
        page = self.items[offset:offset + limit]
        return {"ids": [i[0] for i in page], "documents": [i[1] for i in page], "metadatas": [i[2] for i in page],
                "embeddings": [None if i[3] is None else list(i[3]) for i in page], "uris": None}

    def _distance(self, a, b):
        if self.space == "l2":
            return sum((x - y) ** 2 for x, y in zip(a, b))
        dot = sum(x * y for x, y in zip(a, b))
        if self.space == "ip":
            return 1.0 - dot
        norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return 1.0 - dot / norms if norms else 1.0

    def query(self, query_embeddings=None, n_results=10, include=None):
        query = query_embeddings[0]
        scored = sorted((self._distance(item[3], query), at, item[0]) for at, item in enumerate(self.items))[:n_results]
        return {"ids": [[i for _d, _at, i in scored]], "distances": [[d + self.skew for d, _at, _id in scored]]}


class _Chroma:
    """A stand-in for chromadb: a client per folder, built from what each test registered for that folder."""

    def __init__(self, version="1.3.7"):
        self.module = types.ModuleType("chromadb")
        self.module.__version__ = version
        self.config = types.ModuleType("chromadb.config")
        self.config.Settings = lambda **kwargs: SimpleNamespace(**kwargs)
        self.module.config = self.config
        self.folders = {}
        self.settings = []
        self.names = False
        self.module.PersistentClient = self._client

    def _client(self, path=None, settings=None):
        self.settings.append(settings)
        collections = self.folders[str(path)]
        lister = (lambda: [c.name for c in collections]) if self.names else (lambda: list(collections))
        return SimpleNamespace(list_collections=lister,
                               get_collection=lambda name: next(c for c in collections if c.name == name))


def _vec(*values):
    return [float(x) for x in array("f", values)]


def _plaintext_sqlite(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()


def _world(tmp_path, monkeypatch, *, key=_KEY, chroma=True):
    """The command in a window: the places, the process table and the key redirected, chromadb a stand-in."""
    pytest.importorskip("sqlcipher3")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    row = sqlite3.Row
    loaded, restore = isolate(targets={
        "opti_oignon.encryption": source("encryption.py"),
        "opti_oignon.db_encryption": source("db_encryption.py"),
        "opti_oignon.db_utils": source("db_utils.py"),
        "opti_oignon.vector_db": source("vector_db.py"),
        "opti_oignon.chroma_migration": source("chroma_migration.py"),
    }, blocked=("opti_oignon.secure_bytes",))

    def closed():
        try:
            restore()
        finally:
            sqlite3.Row = row

    try:
        dbe = loaded["opti_oignon.db_encryption"]
        places = (tmp_path / "data", tmp_path / "package" / "data")
        for place in places:
            place.mkdir(parents=True)
        proc = tmp_path / "proc"
        proc.mkdir()
        loaded["opti_oignon.encryption"]._DEFAULT_KEYFILE = tmp_path / "keys" / ".keyfile"
        dbe._STORE_PLACES = places
        dbe._LONE_STORES = []
        dbe._PROC = proc
        dbe._get_db_encryption_key = lambda: key
        dbe._loaded_key = lambda: key
        fake = _Chroma()
        if chroma:
            monkeypatch.setitem(sys.modules, "chromadb", fake.module)
            monkeypatch.setitem(sys.modules, "chromadb.config", fake.config)
        world = SimpleNamespace(dbe=dbe, cm=loaded["opti_oignon.chroma_migration"], vdb=loaded["opti_oignon.vector_db"],
                                dbu=loaded["opti_oignon.db_utils"], places=places, proc=proc, chroma=fake,
                                restore=closed, store=places[1] / "rag" / "vectors.db")
    except BaseException:
        closed()
        raise
    return world


def _folder(world, place, relative, collections):
    folder = place / relative
    _plaintext_sqlite(folder / "chroma.sqlite3")
    (folder / "0a1b-segment").mkdir()
    (folder / "0a1b-segment" / "data_level0.bin").write_bytes(b"\x00" * 64)
    world.chroma.folders[str(folder)] = collections
    return folder


def _facts(world, *, active=(), inactive=()):
    """The memory's store of facts in the package's place, under the window's key: (id, text), active or not."""
    conn = world.dbu.safe_connect(world.places[1] / "memory_facts.db")
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS memory_facts (id TEXT PRIMARY KEY, text TEXT NOT NULL DEFAULT '', "
                     "active INTEGER NOT NULL DEFAULT 1)")
        conn.executemany("INSERT INTO memory_facts (id, text, active) VALUES (?, ?, ?)",
                         [(fact, text, 1) for fact, text in active] + [(fact, text, 0) for fact, text in inactive])
        conn.commit()
    finally:
        conn.close()


def _tree(*roots):
    """Every file below ``roots``, with its bytes, and every folder."""
    found = {}
    for root in roots:
        for top, dirs, files in os.walk(root):
            for name in dirs:
                found[str(Path(top) / name) + "/"] = None
            for name in files:
                path = Path(top) / name
                found[str(path)] = path.read_bytes()
    return found


def _papers():
    return _Collection("papers", [("p1", "cats purr", {"source_file": "/a.pdf", "page": 2}, _vec(1, 0, 0)),
                                  ("p2", "dogs bark", {"source_file": "/b.pdf", "flag": True}, _vec(0, 1, 0)),
                                  ("p3", None, None, _vec(0.6, 0.8, 0)),
                                  ("p4", "birds sing", {"score": 0.25}, _vec(0, 0, 1))],
                       metadata={"hnsw:space": "cosine"})


def _held(world, namespace, name):
    collection = world.vdb.Client(world.store, namespace).get_collection(name)
    got = collection.get(include=["documents", "metadatas", "embeddings"])
    return collection.space, list(zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]))


# ---------------------------------------------------------------------------
# CM1 -- where each collection goes
# ---------------------------------------------------------------------------
def test_cm1_each_folder_the_command_knows_goes_to_its_namespace_and_another_is_left(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        repo, package = world.places
        knowledge = _folder(world, package, "rag/chroma_v2", [_papers()])
        index = _folder(world, package, "rag/chroma_db", [
            _Collection("documents", [("d1", "doc", {"source_file": "/x"}, _vec(1, 1))]),
            _Collection("oo_memories", [("m1", "fact", {"user_id": "local"}, _vec(0, 1))])])
        projects = _folder(world, package, "project_chroma", [
            _Collection("project_p1", [("p1::a::0", "line", {"source_file": "a"}, _vec(1, 0))])])
        elsewhere = _folder(world, repo, "rag/chroma_db", [_Collection("documents", [("d2", "more", None, _vec(2, 2))])])
        unknown = _folder(world, package, "someone_elses_chroma", [_Collection("theirs", [])])
        _facts(world, active=[("m1", "fact")])
        kept = _folder(world, package, "rag/plaintext-20260101T000000Z/chroma_v2", [_Collection("old", [])])
        copied = _folder(world, package, "exports/plaintext-20260101T000000Z/chroma_v2", [_Collection("old", [])])
        assert copied / "chroma.sqlite3" in world.dbe.plaintext_others(), "control: a copy outside a store folder"
        found, other = world.cm.sources()
        assert sorted((str(f), n) for f, n in found) == sorted([
            (str(knowledge), "knowledge"), (str(index), "documents"), (str(projects), "projects"),
            (str(elsewhere), "documents")]), found
        assert other == [unknown], "a folder the command does not know is left; a folder of copies passed over"
        before = _tree(unknown, kept.parent, copied.parent)
        done = world.cm.migrate_all()
        assert sorted(entry["folder"] for entry in done["moved"]) == sorted(str(f) for f, _n in found)
        assert done["unknown"] == [str(unknown)] and _tree(unknown, kept.parent, copied.parent) == before, (
            "left as they were")
        with world.vdb._connection(world.store, write=False) as conn:
            held = sorted(conn.execute("SELECT namespace, name FROM collections").fetchall())
        assert held == [("documents", "documents"), ("knowledge", "papers"), ("memory", "oo_memories"),
                        ("projects", "project_p1")], held
        assert sorted(i[0] for i in _held(world, "documents", "documents")[1]) == ["d1", "d2"], (
            "two folders, one collection")
        assert [i[0] for i in _held(world, "memory", "oo_memories")[1]] == ["m1"], "the memory's collection, its own"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM2 -- everything, proved, then moved
# ---------------------------------------------------------------------------
def test_cm2_every_item_moves_bit_for_bit_encrypted_and_each_folder_is_kept_beside(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        papers = _papers()
        notes = _Collection("notes", [(f"n{i}", f"note {i}", {"i": i}, _vec(i, 1 - i / 3, 0.5))
                                      for i in range(2500)], metadata={"hnsw:space": "l2", "about": "notes"},
                            configuration={}, space="l2")
        empty = _Collection("empty", [], metadata=None)
        folder = _folder(world, package, "rag/chroma_v2", [papers, notes, empty])
        assert world.dbe.main(["migrate"]) == 0
        said = capsys.readouterr().out
        assert not folder.exists(), "the folder moved"
        kept = [p for p in (package / "rag").iterdir() if p.name.startswith("plaintext-")]
        assert len(kept) == 1 and (kept[0] / "chroma_v2" / "chroma.sqlite3").is_file(), kept
        assert (kept[0] / "chroma_v2" / "0a1b-segment" / "data_level0.bin").is_file(), "whole, its index with it"
        assert kept[0] in world.dbe.plaintext_folders(), "a folder of plaintext copies, as encrypt keeps them"
        assert oct(kept[0].stat().st_mode & 0o777) == "0o700"
        assert world.store.read_bytes()[:16] != _PLAIN, "the store is encrypted"
        assert _held(world, "knowledge", "papers") == ("cosine", [(i[0], i[1], i[2], i[3]) for i in papers.items])
        space, items = _held(world, "knowledge", "notes")
        assert space == "l2" and items == [(i[0], i[1], i[2], i[3]) for i in notes.items], "every page read"
        assert [read[1:] for read in notes.reads] == [(1000, 0), (1000, 1000), (1000, 2000)]
        assert all(read[0] == ("documents", "metadatas", "embeddings") for read in notes.reads)
        assert world.vdb.Client(world.store, "knowledge").get_collection("notes").metadata == {
            "hnsw:space": "l2", "about": "notes"}
        assert _held(world, "knowledge", "empty") == ("cosine", []), "an empty collection moves too"
        assert all(settings.anonymized_telemetry is False for settings in world.chroma.settings) and world.chroma.settings
        assert f"{folder}: 3 collection(s) moved into the knowledge base's store\n" in said, said
        assert "  papers, into the namespace knowledge: 4 item(s) written, 0 already there kept\n" in said
        assert "  notes, into the namespace knowledge: 2500 item(s) written, 0 already there kept\n" in said
        assert f"its folder is kept in {kept[0]}: delete it once the platform reads its knowledge base again" in said
        assert f"the knowledge base's store: {world.store}, encrypted under the key that loads" in said
        with world.vdb._connection(world.store, write=False) as conn:
            records = conn.execute("SELECT source, namespace, collection, copied, kept FROM migrations").fetchall()
        assert records == [(str(folder), "knowledge", "papers", 4, 0), (str(folder), "knowledge", "notes", 2500, 0),
                           (str(folder), "knowledge", "empty", 0, 0)], records
        assert world.dbe.main(["migrate"]) == 0 and "no ChromaDB store to move" in capsys.readouterr().out
    finally:
        world.restore()


def test_cm2b_with_no_key_the_store_is_written_as_every_store_is_and_said(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch, key=None)
    try:
        _folder(world, world.places[1], "rag/chroma_v2", [_papers()])
        assert world.dbe.main(["migrate"]) == 0
        assert world.store.read_bytes()[:16] == _PLAIN, "no key loads: in plaintext, as every store of the platform"
        assert "in plaintext: no key loads, as every store of the platform" in capsys.readouterr().out
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM3 -- refused, nothing changed
# ---------------------------------------------------------------------------
def _refused(world, said):
    before = _tree(*world.places)
    with pytest.raises(world.dbe.EncryptionRefused, match=said):
        world.cm.migrate_all()
    assert _tree(*world.places) == before, ("the tree as it was", said)


def test_cm3_each_refusal_changes_nothing(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        papers = _papers()
        _folder(world, package, "rag/chroma_v2", [papers])
        # chromadb that does not import
        monkeypatch.setitem(sys.modules, "chromadb", None)
        _refused(world, "chromadb does not import here")
        monkeypatch.setitem(sys.modules, "chromadb", world.chroma.module)
        # the platform running, and a process holding a file
        entry = world.proc / "4242"
        (entry / "fd").mkdir(parents=True)
        (entry / "cmdline").write_bytes(b"python3\0-m\0uvicorn\0opti_oignon.api.app:app\0")
        _refused(world, "stop the platform first \\(process 4242 runs the platform")
        (entry / "cmdline").write_bytes(b"sqlite3\0")
        os.symlink(package / "rag" / "chroma_v2" / "chroma.sqlite3", entry / "fd" / "3")
        _refused(world, "process 4242 holds chroma.sqlite3 open")
        os.unlink(entry / "fd" / "3")
        # the store in plaintext while a key loads
        _plaintext_sqlite(world.store)
        _refused(world, "is in plaintext while a key loads: run `python3 -m opti_oignon.db_encryption encrypt`")
        for name in os.listdir(world.store.parent):
            if name.startswith("vectors.db"):
                os.unlink(world.store.parent / name)
        # a space, then a dimension, that differs from the store's
        held = world.vdb.Client(world.store, "knowledge")
        held.create_collection("papers", metadata={"hnsw:space": "l2"}).add(ids=["other"], embeddings=[[1.0, 0.0, 0.0]])
        _refused(world, "knowledge/papers is measured in l2 in .* and in cosine in")
        held.delete_collection("papers")
        emptied = held.create_collection("papers", metadata={"hnsw:space": "cosine"})
        emptied.add(ids=["other"], embeddings=[[1.0, 2.0]])
        _refused(world, "holds vectors of dimension 2 in .* and of dimension 3 in")
        emptied.delete(ids=["other"])
        assert world.cm.migrate_all()["moved"], "an emptied collection takes the dimension it is given"
        assert [i[0] for i in _held(world, "knowledge", "papers")[1]] == ["p1", "p2", "p3", "p4"]
        moved = next(p for p in (world.places[1] / "rag").iterdir() if p.name.startswith("plaintext-"))
        (moved / "chroma_v2").rename(world.places[1] / "rag" / "chroma_v2")
        moved.rmdir()
        held.delete_collection("papers")
        # what chromadb hands back that the store cannot take whole
        cases = (("p5", "x", None, None, "p5 of papers in .* has no vector"),
                 ("p5", "x", None, [0.1, 0.2, 0.3], "p5 of papers in .* has a vector float32 does not hold exactly"),
                 ("p5", "x", {"tags": ["a", "b"]}, _vec(1, 1, 1), "an item's metadata maps names to"))
        for item_id, document, metadata, vector, said in cases:
            papers.items.append((item_id, document, metadata, vector))
            _refused(world, said)
            papers.items.pop()
        papers.count_offset = 1
        _refused(world, "papers of .* does not read whole \\(4 items read, 5 counted\\)")
        papers.count_offset = 0
        papers.configuration_json = {"hnsw": {"space": "manhattan"}}
        _refused(world, "is measured in manhattan, which the knowledge base's store does not measure")
        papers.configuration_json = {"hnsw": {"space": "cosine"}}
        held.create_collection("papers", metadata={"hnsw:space": "l2"})
        assert world.cm.migrate_all()["moved"], "control: the same tree moves once each cause is gone"
        assert _held(world, "knowledge", "papers")[0] == "cosine", "an empty collection takes the space it is given"
        assert world.vdb.Client(world.store, "knowledge").get_collection("papers").metadata["hnsw:space"] == "cosine"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM4 -- the proof refuses
# ---------------------------------------------------------------------------
def test_cm4_a_copy_that_does_not_prove_writes_nothing(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        papers = _papers()
        _folder(world, world.places[1], "rag/chroma_v2", [papers])
        papers.skew = 0.01
        _refused(world, "ChromaDB measures .* where the knowledge base's store measures .* in cosine")
        papers.skew = 0.0
        papers.space = "l2"
        _refused(world, "ChromaDB measures .* where the knowledge base's store measures .* in cosine")
        papers.space = "cosine"
        insert = world.vdb.Collection._insert

        def altered(conn, row, item_id, vector, metadata, document):
            if item_id == "p3":
                vector = array("f", [0.6, 0.8, 1e-7])
            return insert(conn, row, item_id, vector, metadata, document)

        world.vdb.Collection._insert = staticmethod(altered)
        _refused(world, "p3 does not read back from .* as it was handed in")
        world.vdb.Collection._insert = staticmethod(insert)
        papers.skew = 0.00001
        assert world.cm.migrate_all()["moved"], "a measure within float32's reach of the store's proves"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM5 -- what the platform wrote since is kept
# ---------------------------------------------------------------------------
def test_cm5_an_id_the_store_holds_is_kept_and_a_run_stuck_after_its_commit_is_finished_by_the_next(
        tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        papers = _papers()
        folder = _folder(world, package, "rag/chroma_v2", [papers])
        # The memory's folder, read first: its facts never judge another namespace's items.
        _folder(world, package, "rag/chroma_db", [_Collection("oo_memories", [
            ("f1", "Lives in Lyon", {"user_id": "local"}, _vec(1, 0))], metadata={"hnsw:space": "cosine"})])
        _facts(world, active=[("f1", "Lives in Lyon")])
        since = world.vdb.Client(world.store, "knowledge").create_collection("papers",
                                                                           metadata={"hnsw:space": "cosine"})
        since.add(ids=["p2"], embeddings=[[0.0, 0.0, 1.0]], documents=["written since"])
        rename = os.rename

        def stuck(src, dst):
            if Path(src) == folder:
                raise OSError(5, "Input/output error")
            return rename(src, dst)

        monkeypatch.setattr(world.cm.os, "rename", stuck)
        assert world.dbe.main(["migrate"]) == 3
        said = capsys.readouterr()
        err = said.err
        assert f"the knowledge base's store holds every collection, but {folder} could not be moved" in err, err
        assert f"{folder}: 1 collection(s) written into the knowledge base's store, the folder not moved: the next " \
               "run of migrate moves it\n  papers, into the namespace knowledge: 3 item(s) written, 1 already there " \
               "kept\n" in said.out, "what the store took, said before where the run stopped"
        # One log of both streams reads in that order; a stdout closed early costs neither the line nor the 3.
        order = []

        class Buffered(io.StringIO):
            def flush(self):
                if self.getvalue():
                    order.append("summary")
                super().flush()

        class Errors(io.StringIO):
            def write(self, text):
                order.append("stopped" if text.startswith("stopped part way") else "other")
                return super().write(text)

        class Closed(io.TextIOBase):
            def write(self, text):
                raise BrokenPipeError(32, "Broken pipe")

            def fileno(self):
                raise io.UnsupportedOperation("fileno")

        with monkeypatch.context() as patch:
            patch.setattr(sys, "stdout", Buffered())
            patch.setattr(sys, "stderr", Errors())
            assert world.dbe.main(["migrate"]) == 3
        assert "summary" in order and order.index("summary") < order.index("stopped"), order
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stdout", Closed())
            assert world.dbe.main(["migrate"]) == 3
        err = capsys.readouterr().err
        assert "what this run did is said only up to here (BrokenPipeError: [Errno 32] Broken pipe)\nstopped part " \
               "way: " in err, err
        # An interrupt while it is said: the same.
        with monkeypatch.context() as patch:
            patch.setattr(world.dbe, "_migration_summary", lambda done: (_ for _ in ()).throw(KeyboardInterrupt()))
            try:
                code = world.dbe.main(["migrate"])
            except KeyboardInterrupt:
                code = "the interrupt escaped"
            assert code == 3
        err = capsys.readouterr().err
        assert "what this run did is said only up to here (KeyboardInterrupt)\nstopped part way: " in err, err
        assert "the next run of migrate moves it" in err and folder.is_dir(), "stopped part way, said"
        space, items = _held(world, "knowledge", "papers")
        assert [i[0] for i in items] == ["p2", "p1", "p3", "p4"] and items[0][1] == "written since", (
            "what the platform wrote since is kept as it is", items)
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        assert "  papers, into the namespace knowledge: 0 item(s) written, 4 already there kept" in (
            capsys.readouterr().out)
        assert not folder.exists() and _held(world, "knowledge", "papers") == (space, items)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM6 -- a signal
# ---------------------------------------------------------------------------
def test_cm6_a_signal_before_the_commit_changes_nothing_and_one_after_lets_the_run_finish(
        tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        first, second = _papers(), _Collection("notes", [("n1", "note", None, _vec(1, 1, 1))])
        folder = _folder(world, world.places[1], "rag/chroma_v2", [first, second])
        first.on_get = lambda: os.kill(os.getpid(), signal.SIGTERM)
        before, handler = _tree(*world.places), signal.getsignal(signal.SIGTERM)
        assert world.dbe.main(["migrate"]) == 1
        err = capsys.readouterr().err
        assert "signal 15 noted before notes of" in err and "nothing was written" in err, err
        assert _tree(*world.places) == before and not world.store.exists()
        assert signal.getsignal(signal.SIGTERM) == handler, "the handler put back"
        first.on_get = None
        fresh = world.dbe._fresh_folder

        def late(place, name):
            os.kill(os.getpid(), signal.SIGTERM)
            return fresh(place, name)

        second.on_get = lambda: os.kill(os.getpid(), signal.SIGTERM)
        assert world.dbe.main(["migrate"]) == 1
        err = capsys.readouterr().err
        assert "signal 15 noted before the knowledge base's store committed: nothing was written" in err, err
        assert _tree(*world.places) == before and not world.store.exists()
        second.on_get = None
        monkeypatch.setattr(world.dbe, "_fresh_folder", late)
        rename = os.rename

        def stuck(src, dst):
            if Path(src) == folder:
                raise OSError(5, "Input/output error")
            return rename(src, dst)

        monkeypatch.setattr(world.cm.os, "rename", stuck)
        assert world.dbe.main(["migrate"]) == 3
        err = capsys.readouterr().err
        assert "signal 15 came after the commit: the run went on\nstopped part way: " in err, err
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        captured = capsys.readouterr()
        assert "signal 15 came after the commit: the run finished" in captured.err, captured.err
        assert not folder.exists() and [i[0] for i in _held(world, "knowledge", "notes")[1]] == ["n1"]
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM7 -- chromadb's versions
# ---------------------------------------------------------------------------
def test_cm7_both_listings_and_the_space_each_version_gave(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        world.chroma.names = True
        world.chroma.module.__version__ = "0.4.24"
        legacy = [_Collection("named", [("a", "x", None, _vec(1, 0))], metadata=None, configuration={}, space="l2"),
                  _Collection("said", [("b", "y", None, _vec(0, 1))], metadata={"hnsw:space": "ip"},
                              configuration={}, space="ip"),
                  _Collection("spann", [("c", "z", None, _vec(1, 1))], metadata=None,
                              configuration={"spann": {"space": "cosine"}}, space="cosine")]
        _folder(world, package, "rag/chroma_v2", legacy)
        assert world.cm.migrate_all()["moved"]
        assert [_held(world, "knowledge", name)[0] for name in ("named", "said", "spann")] == ["l2", "ip", "cosine"]
        world.chroma.names = False
        world.chroma.module.__version__ = "1.3.7"
        _folder(world, package, "project_chroma", [_Collection("project_p", [("d", "w", None, _vec(1, 2))],
                                                               metadata={"project_id": "p"}, configuration={},
                                                               space="l2"),
                                                   _Collection("project_q", [("e", "v", None, _vec(2, 1))],
                                                               metadata=None, space="ip")])
        assert world.cm.migrate_all()["moved"]
        assert _held(world, "projects", "project_p")[0] == "l2", "no space said: ChromaDB's default"
        assert _held(world, "projects", "project_q")[0] == "ip", "the space its configuration says"
        assert world.vdb.Client(world.store, "projects").get_collection("project_p").metadata == {
            "project_id": "p", "hnsw:space": "l2"}, "the space it was measured in, kept with its metadata"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM8 -- said by check, one run at a time
# ---------------------------------------------------------------------------
def test_cm8_check_names_the_command_for_the_stores_it_moves_and_one_run_at_a_time(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        repo, package = world.places
        known = _folder(world, package, "rag/chroma_db", [_papers()])
        unknown = _folder(world, package, "other_chroma", [])
        _folder(world, package, "rag/plaintext-20260101T000000Z/chroma_v2", [])
        assert world.dbe.main(["check"]) == 0
        out = capsys.readouterr().out
        hint = ("`python3 -m opti_oignon.db_encryption migrate`, the platform stopped, moves its collections into the "
                "knowledge base's store, encrypted with the other stores")
        assert f"{known}: {hint}" in out, out
        assert f"{unknown}/chroma.sqlite3: a ChromaDB vector store" in out and f"{unknown}: {hint}" not in out
        assert out.count(hint) == 1, "never for a folder of plaintext copies"
        assert world.dbe.MIGRATE_COMMAND == "python3 -m opti_oignon.db_encryption migrate"
        fd = os.open(repo, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = _tree(*world.places)
            assert world.dbe.main(["migrate"]) == 1
            assert "another run of encrypt or migrate is under way: nothing was written" in capsys.readouterr().err
            assert _tree(*world.places) == before
        finally:
            os.close(fd)
        assert world.dbe.main(["migrate"]) == 0 and not known.exists()
    finally:
        world.restore()


def test_cm8b_a_relative_rag_home_refuses_the_run_by_name(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        before = _tree(*world.places)
        for value in ("relative/rag", ""):
            monkeypatch.setenv("RAG_HOME", value)
            assert world.dbe.main(["check"]) == 0
            shown = value or "''"
            assert f"RAG_HOME is {shown}, below whatever directory the platform starts in: not checked" in (
                capsys.readouterr().out), "check says it too"
            assert world.dbe.main(["migrate"]) == 1
            err = capsys.readouterr().err
            assert "this command cannot move what the platform waits on: the documents' index and the memory's " \
                   f"folder below RAG_HOME, which is relative ({value!r}): this command cannot look below it; set it " \
                   "to the absolute path the platform reads" in err, err
            assert _tree(*world.places) == before
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM9 -- git takes no plaintext copy
# ---------------------------------------------------------------------------
def test_cm9_every_file_the_command_keeps_in_plaintext_is_one_git_ignores(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        repo, package = world.places
        for place, relative in ((repo, "rag/chroma_db"), (repo, "project_chroma"), (package, "rag/chroma_v2"),
                                (package, "rag/chroma_db"), (package, "project_chroma")):
            _folder(world, place, relative, [_Collection("documents", [("d", "x", None, _vec(1, 0))])])
        _facts(world)
        done = world.cm.migrate_all()
        assert len(done["moved"]) == 5, done
        world.dbe._STORE_PLACES = None
        real = [Path(place).relative_to(_REPO) for place in world.dbe._places()]
        rooms = {repo: real[0], package: real[1]}
        paths = []
        for entry in done["moved"]:
            kept = Path(entry["kept_in"]) / Path(entry["folder"]).name
            place = repo if repo in kept.parents else package
            paths += [str(rooms[place] / path.relative_to(place)) for path in sorted(kept.rglob("*")) if path.is_file()]
        assert len(paths) == 10, ("control: each folder's database and index", paths)
        probe = "opti_oignon/chroma_migration.py"
        answer = subprocess.run(["git", "-C", str(_REPO), "check-ignore", "--no-index", "-v", "-n", *paths, probe],
                                capture_output=True, text=True, timeout=60)
        rules = dict(reversed(line.split("\t", 1)) for line in answer.stdout.splitlines())

        def ignored(path):
            source_, _, rest = rules.get(path, "::").partition(":")
            return bool(source_) and not rest.partition(":")[2].startswith("!")

        assert not ignored(probe) and probe in rules, ("control: a source file is not ignored", answer.stdout,
                                                       answer.stderr)
        assert [path for path in paths if not ignored(path)] == [], ("git would take these", answer.stdout)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM10 -- what the memory still holds
# ---------------------------------------------------------------------------
def test_cm10_a_memory_vector_moves_only_when_an_active_fact_holds_it_as_it_reads(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]

        def memories():
            return _Collection("oo_memories", [("f-active", "Allergic to penicillin", {"user_id": "local"}, _vec(1, 0)),
                                               ("f-soft", "b", {"user_id": "local"}, _vec(0, 1)),
                                               ("f-gone", "c", {"user_id": "local"}, _vec(1, 1)),
                                               ("f-edited", "Allergic to amoxicillin", {"user_id": "local"},
                                                _vec(1, 2))],
                               metadata={"hnsw:space": "cosine"})

        _folder(world, package, "rag/chroma_db", [memories(), _Collection("documents", [("d1", "doc", None,
                                                                                         _vec(2, 1))])])
        _facts(world, active=[("f-active", "Allergic to penicillin"), ("f-edited", "Allergic to aspirin"),
                              ("f-added", "Lives in Lyon")], inactive=[("f-soft", "b")])
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert [i[0] for i in _held(world, "memory", "oo_memories")[1]] == ["f-active"], (
            "a deleted fact's vector stays out, and a rewritten one's, which holds the old text")
        assert "  oo_memories, into the namespace memory: 1 item(s) written, 0 already there kept, 3 left out: no " \
               "active fact of the memory holds them as they read now" in out, out
        assert "  documents, into the namespace documents: 1 item(s) written" in out, "each collection where it went"
        assert "2 active fact(s) of the memory hold no vector of their text as it reads now: recall finds them by " \
               "their words" in out, out
        assert [i[0] for i in _held(world, "documents", "documents")[1]] == ["d1"], "the documents have no such owner"
        _folder(world, package, "rag/chroma_db", [memories()])
        facts = package / "memory_facts.db"
        kept = facts.read_bytes()
        facts.write_bytes(b"no database of facts, only these words")
        _refused(world, "the memory's facts at .* do not read here")
        conn = sqlite3.connect(str(facts.with_name("plain.db")))
        try:
            conn.execute("CREATE TABLE memory_facts (id TEXT PRIMARY KEY, text TEXT, active INTEGER)")
            conn.commit()
        finally:
            conn.close()
        facts.unlink()
        facts.with_name("plain.db").rename(facts)
        _refused(world, "the memory's facts at .* do not read here .*it is in plaintext while a key loads: run "
                        "`python3 -m opti_oignon.db_encryption encrypt` first")
        facts.unlink()
        done = world.cm.migrate_all()
        assert done["moved"][0]["left_out"] == 4 and done["moved"][0]["written"] == 0, (
            "no store of facts: no fact holds a vector")
        facts.write_bytes(kept)
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM11 -- no half run
# ---------------------------------------------------------------------------
def test_cm11_a_holder_seen_before_the_commit_and_every_failure_a_refusal_by_name(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    try:
        holders, calls = world.dbe.store_holders, []

        def late_holder(paths, proc=None):
            calls.append(list(paths))
            return {4242: "runs the platform (uvicorn)"} if len(calls) == 2 else holders(paths, proc=proc)

        # Only the projects' folder: the run makes the store's folder too, and removes both.
        _folder(world, world.places[1], "project_chroma", [_Collection("project_p", [("p", "x", None, _vec(1, 0))])])
        monkeypatch.setattr(world.dbe, "store_holders", late_holder)
        _refused(world, "stop the platform first \\(process 4242 runs the platform \\(uvicorn\\)\\)")
        assert len(calls) == 3 and calls[2] == [world.store], ("read again before the commit, then the store it "
                                                                "made removed once no one holds it", calls)
        assert not world.store.parent.exists(), "the folder the run made, removed"
        # A store on its rollback journal, as the encryption leaves it, is left on it by a refused run.
        world.vdb.Client(world.store, "knowledge").create_collection("other")
        conn = world.dbu.safe_connect(world.store)
        try:
            assert conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0] == "delete"
        finally:
            conn.close()
        calls.clear()
        _refused(world, "stop the platform first")
        monkeypatch.setattr(world.dbe, "store_holders", holders)
        papers = _papers()
        _folder(world, world.places[1], "rag/chroma_v2", [papers])
        papers.on_get = lambda: (_ for _ in ()).throw(RuntimeError("segment file truncated"))
        _refused(world, "does not read through chromadb \\(RuntimeError: segment file truncated\\)")
        papers.on_get = None
        query = papers.query
        papers.query = lambda **kwargs: {"ids": [["ghost"]], "distances": [[0.0]]}
        _refused(world, "ChromaDB returns ghost, which it did not list")
        papers.query = query
        for name in os.listdir(world.store.parent):
            if name.startswith("vectors.db"):
                os.unlink(world.store.parent / name)
        world.dbe._get_db_encryption_key = lambda: _OTHER_KEY
        world.vdb.Client(world.store, "knowledge").create_collection("other")
        world.dbe._get_db_encryption_key = lambda: _KEY
        _refused(world, "the knowledge base's store at .* took nothing \\(RuntimeError: SQLCipher key verification")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM12 -- the real chromadb
# ---------------------------------------------------------------------------
_REAL = r'''
import json, os, socket, sys
from pathlib import Path
repo, root = sys.argv[1], Path(sys.argv[2])
for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
    os.environ.pop(name, None)


def refuse(*args, **kwargs):
    raise OSError("the network is refused in this contract")


socket.socket.connect = refuse
socket.socket.connect_ex = refuse
socket.create_connection = refuse
socket.getaddrinfo = refuse
import chromadb
from chromadb.config import Settings
sys.meta_path[:] = [finder for finder in sys.meta_path if "editable" not in repr(finder).lower()]
sys.path.insert(0, repo)
import types
# The security mode is Daily, as the shared window has it: its lock file lies in a data place this contract never
# reads.
mode = types.ModuleType("opti_oignon.security_mode")
mode.is_bulbe = lambda: False
sys.modules["opti_oignon.security_mode"] = mode
from opti_oignon import chroma_migration as cm, db_encryption as dbe, encryption as enc, vector_db as vdb
assert all(module.__file__.startswith(repo) for module in (cm, dbe, enc, vdb)), [m.__file__ for m in (cm, dbe, vdb)]
places = (root / "data", root / "package" / "data")
for place in places:
    place.mkdir(parents=True, exist_ok=True)
(root / "proc").mkdir(exist_ok=True)
dbe._STORE_PLACES, dbe._LONE_STORES, dbe._PROC = places, [], root / "proc"
enc._DEFAULT_KEYFILE = root / "keys" / ".keyfile"
key = bytes(range(1, 33))
dbe._get_db_encryption_key = lambda: key
dbe._loaded_key = lambda: key
made = json.loads(sys.argv[3])
for relative, collections in made.items():
    client = chromadb.PersistentClient(path=str(places[1] / relative), settings=Settings(anonymized_telemetry=False))
    for name, metadata, items in collections:
        collection = client.get_or_create_collection(name=name, metadata=metadata)
        collection.add(ids=[i[0] for i in items], documents=[i[1] for i in items], metadatas=[i[2] for i in items],
                       embeddings=[i[3] for i in items])
conn = vdb.safe_connect(places[1] / "memory_facts.db")
conn.execute("CREATE TABLE memory_facts (id TEXT PRIMARY KEY, text TEXT NOT NULL, active INTEGER NOT NULL)")
conn.executemany("INSERT INTO memory_facts VALUES (?, ?, ?)", [("m1", "a fact", 1), ("m2", "a deleted fact", 0)])
conn.commit()
conn.close()
done = cm.migrate_all()
held = {}
for namespace, name in (("knowledge", "papers"), ("documents", "documents"), ("memory", "oo_memories"),
                        ("projects", "project_p1")):
    collection = vdb.Client(done["target"], namespace).get_collection(name)
    got = collection.get(include=["documents", "metadatas", "embeddings"])
    held[f"{namespace}/{name}"] = [collection.space, list(zip(got["ids"], got["documents"], got["metadatas"],
                                                              got["embeddings"]))]
print("RESULT " + json.dumps({"moved": done["moved"], "held": held, "encrypted": done["encrypted"],
                              "head": Path(done["target"]).read_bytes()[:16].hex()}))
'''


def test_cm12_the_real_chromadb_makes_stores_the_command_moves_whole(tmp_path):
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb does not import here")
    papers = [["p1", "cats purr", {"source_file": "/a.pdf", "page": 2, "parent_doc_id": "d1"}, [0.5, 0.25, 1.0]],
              ["p2", "dogs bark", {"source_file": "/b.pdf", "flag": True, "parent_doc_id": "d2"}, [0.0, 1.0, 0.5]],
              ["p3", "birds sing", {"score": 0.75, "parent_doc_id": "d3"}, [2.0, 0.5, 0.25]]]
    made = {
        "rag/chroma_v2": [["papers", {"hnsw:space": "cosine"}, papers]],
        "rag/chroma_db": [["documents", {"description": "Documents RAG personnels"},
                           [["/n.txt-0", "a line", {"source_file": "/n.txt", "chunk_index": 0}, [1.0, 0.0]],
                            ["/n.txt-1", "another", {"source_file": "/n.txt", "chunk_index": 1}, [0.0, 1.0]]]],
                          ["oo_memories", {"hnsw:space": "cosine", "description": "memory"},
                           [["m1", "a fact", {"user_id": "local", "category": "fact"}, [0.5, 0.5]],
                            ["m2", "a deleted fact", {"user_id": "local", "category": "fact"}, [0.25, 1.0]]]]],
        "project_chroma": [["project_p1", {"project_id": "p1"},
                            [["p1::a::0", "x = 1", {"source_file": "a", "chunk_index": 0}, [0.6000000238418579, 0.800000011920929]]]]],
    }
    # Its home in a temporary folder, its user site kept: where this interpreter finds chromadb's numpy.
    env = dict(os.environ, HOME=str(tmp_path / "home"), PYTHONUSERBASE=site.getuserbase(), PYTHONDONTWRITEBYTECODE="1")
    (tmp_path / "home").mkdir()
    child = subprocess.run([sys.executable, "-c", _REAL, str(_REPO), str(tmp_path / "root"), json.dumps(made)],
                           capture_output=True, text=True, timeout=300, env=env)
    lines = [line for line in child.stdout.splitlines() if line.startswith("RESULT ")]
    assert child.returncode == 0 and lines, (child.returncode, child.stdout[-2000:], child.stderr[-3000:])
    result = json.loads(lines[-1][len("RESULT "):])
    assert sorted(Path(entry["folder"]).name for entry in result["moved"]) == ["chroma_db", "chroma_v2",
                                                                               "project_chroma"]
    assert result["encrypted"] is True and bytes.fromhex(result["head"]) != _PLAIN, "the store is encrypted"
    held = result["held"]

    def same(got, want):
        """Ids, documents and metadata as written; each vector as chromadb hands it back, within float32's reach
        of the one written (1.3.7 hands back some of a cosine collection's a unit in the last place apart)."""
        return len(got) == len(want) and all(
            g[:3] == w[:3] and len(g[3]) == len(w[3]) and all(abs(a - b) <= 1e-6 for a, b in zip(g[3], w[3]))
            for g, w in zip(got, want))

    assert held["knowledge/papers"][0] == "cosine" and same(held["knowledge/papers"][1], papers), held
    assert held["documents/documents"][0] == "l2", "no space said: ChromaDB's own default, kept"
    assert same(held["documents/documents"][1], made["rag/chroma_db"][0][2])
    assert held["memory/oo_memories"][0] == "cosine" and same(held["memory/oo_memories"][1],
                                                              [made["rag/chroma_db"][1][2][0]]), "an active fact's"
    assert held["projects/project_p1"][0] == "l2" and same(held["projects/project_p1"][1], made["project_chroma"][0][2])
    memory = next(entry for entry in result["moved"] if entry["folder"].endswith("chroma_db"))
    assert memory["left_out"] == 1 and memory["written"] == 3, memory


# ---------------------------------------------------------------------------
# CM13 -- one test of what waits
# ---------------------------------------------------------------------------
def test_cm13_what_the_platform_waits_on_the_command_moves_or_refuses_by_name(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        elsewhere = _folder(world, tmp_path / "other-disk", "chroma_v2", [_papers()])
        (package / "rag").mkdir()
        (package / "rag" / "chroma_v2").symlink_to(elsewhere, target_is_directory=True)
        client = world.vdb.Client(world.store, "knowledge", legacy=package / "rag" / "chroma_v2")
        with pytest.raises(world.vdb.MigrationPending, match="leads through a symbolic link .* which the migration "
                                                             "never follows"):
            client.list_collections()
        assert world.vdb.chroma_state(package / "rag" / "chroma_v2")[0] == "link"
        before = _tree(*world.places)
        assert world.dbe.main(["migrate"]) == 1
        err = capsys.readouterr().err
        assert f"this command cannot move what the platform waits on: {package / 'rag' / 'chroma_v2'} (a symbolic " \
               "link" in err and "never followed: put the folder it leads to in its place)" in err, err
        assert _tree(*world.places) == before, "nothing written, nothing moved"
        (package / "rag" / "chroma_v2").unlink()
        # ChromaDB's database itself a link: the file is named as such on both sides.
        (package / "rag" / "chroma_v2").mkdir()
        (package / "rag" / "chroma_v2" / "chroma.sqlite3").symlink_to(elsewhere / "chroma.sqlite3")
        with pytest.raises(world.vdb.MigrationPending, match="never follows: put the file it leads to in its place"):
            client.list_collections()
        assert world.dbe.main(["migrate"]) == 1
        assert "never followed: put the file it leads to in its place)" in capsys.readouterr().err
        (package / "rag" / "chroma_v2" / "chroma.sqlite3").unlink()
        (package / "rag" / "chroma_v2").rmdir()
        # A folder that cannot be looked at: the same on both sides.
        locked = _folder(world, package, "project_chroma", [_Collection("project_p", [])])
        locked.chmod(0)
        try:
            assert world.vdb.chroma_state(locked)[0] == "unreadable"
            with pytest.raises(world.vdb.MigrationPending, match="cannot be looked at"):
                world.vdb.Client(world.store, "projects", legacy=locked).list_collections()
            assert world.dbe.main(["migrate"]) == 1
            assert f"{locked} (it cannot be looked at" in capsys.readouterr().err
        finally:
            locked.chmod(0o700)
        (locked / "chroma.sqlite3").unlink()
        (locked / "chroma.sqlite3").write_bytes(b"")
        # A chroma.sqlite3 that is no database closes nothing and gives the command nothing to move.
        assert world.vdb.chroma_state(locked) == ("none", "")
        assert world.vdb.Client(world.store, "projects", legacy=locked).list_collections() == []
        assert world.dbe.main(["migrate"]) == 0 and "no ChromaDB store to move" in capsys.readouterr().out
        # ChromaDB's store in a folder the walk cannot list: the platform waits on it, the run refuses.
        hidden = _folder(world, package, "rag/chroma_v2", [_papers()])
        (package / "rag").chmod(0o311)
        try:
            assert world.vdb.chroma_state(hidden)[0] == "store", "control: the platform's test sees it"
            assert world.dbe.main(["migrate"]) == 1
            assert f"{hidden} (ChromaDB's store, which the walk of the places did not reach" in (
                capsys.readouterr().err)
        finally:
            (package / "rag").chmod(0o700)
    finally:
        world.restore()


def test_cm13b_a_repository_below_a_folder_named_like_a_copy_moves_as_any_other(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path / "plaintext-archive", monkeypatch)
    try:
        folder = _folder(world, world.places[1], "rag/chroma_v2", [_papers()])
        assert world.cm.sources()[0] == [(folder, "knowledge")]
        assert world.dbe._migration_hint(folder / "chroma.sqlite3") is not None, "check names the command for it"
        assert world.dbe.main(["migrate"]) == 0 and not folder.exists()
        assert [i[0] for i in _held(world, "knowledge", "papers")[1]] == ["p1", "p2", "p3", "p4"]
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM14 -- a resumed run undoes what it must not keep
# ---------------------------------------------------------------------------
def test_cm14_a_resumed_run_removes_what_a_rewritten_or_deleted_fact_no_longer_holds(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        folder = _folder(world, package, "rag/chroma_db", [
            _Collection("oo_memories", [("f1", "Allergic to penicillin", {"user_id": "local"}, _vec(1, 0)),
                                        ("f2", "Lives in Lyon", {"user_id": "local"}, _vec(0, 1)),
                                        ("f3", "Plays the cello", {"user_id": "local"}, _vec(1, 1))],
                        metadata={"hnsw:space": "cosine"})])
        _facts(world, active=[("f1", "Allergic to penicillin"), ("f2", "Lives in Lyon"), ("f3", "Plays the cello")])
        rename = os.rename

        def stuck(src, dst):
            if Path(src) == folder:
                raise OSError(5, "Input/output error")
            return rename(src, dst)

        monkeypatch.setattr(world.cm.os, "rename", stuck)
        assert world.dbe.main(["migrate"]) == 3
        said = capsys.readouterr()
        assert [i[0] for i in _held(world, "memory", "oo_memories")[1]] == ["f1", "f2", "f3"], "control: committed"
        assert "  oo_memories, into the namespace memory: 3 item(s) written, 0 already there kept\n" in said.out
        assert "no ChromaDB store to move" not in said.out and "the knowledge base's store: " in said.out, said.out
        assert not [kept.name for kept in (package / "rag").iterdir() if kept.name.startswith("plaintext-")], (
            "the folder of plaintext copies made for it, gone again")
        # The platform rewrites f1 and deletes f2 while the folder still waits.
        conn = world.dbu.safe_connect(package / "memory_facts.db")
        try:
            conn.execute("UPDATE memory_facts SET text = 'Allergic to aspirin' WHERE id = 'f1'")
            conn.execute("UPDATE memory_facts SET active = 0 WHERE id = 'f2'")
            conn.commit()
        finally:
            conn.close()
        # A fact added meanwhile whose vector something else wrote.
        conn = world.dbu.safe_connect(package / "memory_facts.db")
        try:
            conn.execute("INSERT INTO memory_facts (id, text, active) VALUES ('f4', 'Reads in Spanish', 1)")
            conn.commit()
        finally:
            conn.close()
        held = world.vdb.Client(world.store, "memory").get_collection("oo_memories")
        held.add(ids=["f4"], embeddings=[[0.5, 0.5]], documents=["Reads in Spanish"])
        with world.vdb._connection(world.store, write=False) as reader:
            generation = reader.execute("SELECT generation FROM collections WHERE name = 'oo_memories'").fetchone()[0]
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert [i[0] for i in _held(world, "memory", "oo_memories")[1]] == ["f3", "f4"], (
            "the rewritten fact's old vector and the deleted fact's are gone; what something else wrote stays")
        assert "  oo_memories, into the namespace memory: 0 item(s) written, 1 already there kept, 2 left out: no " \
               "active fact of the memory holds them as they read now, 2 removed: the store held them with a text " \
               "no active fact of the memory holds now" in out
        assert "1 active fact(s) of the memory hold no vector of their text as it reads now: recall finds them by " \
               "their words" in out, out
        assert not folder.exists()
        with world.vdb._connection(world.store, write=False) as reader:
            assert reader.execute("SELECT generation FROM collections WHERE name = 'oo_memories'").fetchone()[0] > (
                generation), "every reader's view of the collection moves on"
            assert reader.execute("SELECT removed FROM migrations ORDER BY id DESC LIMIT 1").fetchone()[0] == 2
    finally:
        world.restore()


def test_cm14b_a_vector_the_store_holds_with_the_text_its_fact_reads_is_kept(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        _folder(world, package, "rag/chroma_db", [
            _Collection("oo_memories", [("f1", "Allergic to penicillin", {"user_id": "local"}, _vec(1, 0))],
                        metadata={"hnsw:space": "cosine"})])
        _facts(world, active=[("f1", "Allergic to aspirin"), ("f5", "Speaks German")])
        held = world.vdb.Client(world.store, "memory").get_or_create_collection("oo_memories",
                                                                                metadata={"hnsw:space": "cosine"})
        held.add(ids=["f1", "f5"], embeddings=[[0.0, 1.0], [1.0, 1.0]],
                 documents=["Allergic to aspirin", "Speaks Dutch"])
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert _held(world, "memory", "oo_memories")[1] == [("f1", "Allergic to aspirin", None, [0.0, 1.0]),
                                                            ("f5", "Speaks Dutch", None, [1.0, 1.0])], (
            "the platform's vector for the fact as it reads now, kept; one no folder names, left as it is")
        assert "1 left out" in out and "removed" not in out, out
        assert "1 active fact(s) of the memory hold no vector of their text as it reads now" in out, (
            "f5's vector is of another text: counted with those that hold none")
        # A store that does not read back once it has committed: the count unknown, said; the folder moves.
        folder = _folder(world, package, "rag/chroma_db", [
            _Collection("oo_memories", [("f1", "Allergic to penicillin", {"user_id": "local"}, _vec(1, 0))],
                        metadata={"hnsw:space": "cosine"})])
        found = world.cm.vector_db.Client.get_collection

        def gone(self, name):
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(world.cm.vector_db.Client, "get_collection", gone)
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        monkeypatch.setattr(world.cm.vector_db.Client, "get_collection", found)
        assert "the active facts of the memory that hold no vector of their text: not counted, the store did not " \
               "read back (OSError: [Errno 5] Input/output error); recall finds such facts by their words" in out, out
        assert not folder.exists()
    finally:
        world.restore()


def test_cm14c_a_copy_with_no_text_whose_fact_is_gone_is_removed(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        _folder(world, package, "rag/chroma_db", [
            _Collection("oo_memories", [("f1", "Lives in Lyon", {"user_id": "local"}, _vec(1, 0))],
                        metadata={"hnsw:space": "cosine"})])
        _facts(world, inactive=[("f1", "Lives in Lyon")])
        held = world.vdb.Client(world.store, "memory").get_or_create_collection("oo_memories",
                                                                                metadata={"hnsw:space": "cosine"})
        held.add(ids=["f1"], embeddings=[[1.0, 0.0]])
        assert _held(world, "memory", "oo_memories")[1][0][1] is None, "control: a copy with no text"
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert _held(world, "memory", "oo_memories")[1] == [], "no active fact holds it: removed"
        assert ", 1 removed: the store held them with a text no active fact of the memory holds now" in out, out
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# CM15 -- two sources of one collection
# ---------------------------------------------------------------------------
def test_cm15_two_folders_holding_the_memory_move_together_the_current_text_kept(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        repo, package = world.places
        current = [("f1", "Allergic to aspirin", {"user_id": "local"}, _vec(0, 1))]
        stale = [("f1", "Allergic to penicillin", {"user_id": "local"}, _vec(1, 0))]
        for first, second in ((current, stale), (stale, current)):
            _folder(world, repo, "rag/chroma_db", [_Collection("oo_memories", list(first),
                                                               metadata={"hnsw:space": "cosine"})])
            _folder(world, package, "rag/chroma_db", [_Collection("oo_memories", list(second),
                                                                  metadata={"hnsw:space": "cosine"})])
            if not (package / "memory_facts.db").exists():
                _facts(world, active=[("f1", "Allergic to aspirin")])
            assert world.dbe.main(["migrate"]) == 0, capsys.readouterr().err
            capsys.readouterr()
            assert _held(world, "memory", "oo_memories")[1] == [("f1", "Allergic to aspirin", {"user_id": "local"},
                                                                 _vec(0, 1))], "the current copy, whichever came first"
            world.vdb.Client(world.store, "memory").delete_collection("oo_memories")
    finally:
        world.restore()


def test_cm15b_a_run_stopped_between_two_folders_leaves_only_the_text_the_fact_reads(tmp_path, monkeypatch, capsys):
    rename = os.rename

    def memories(text, vector):
        return [_Collection("oo_memories", [("f1", text, {"user_id": "local"}, vector)],
                            metadata={"hnsw:space": "cosine"})]

    def rewrite(world, sql):
        conn = world.dbu.safe_connect(world.places[1] / "memory_facts.db")
        try:
            conn.execute(sql)
            conn.commit()
        finally:
            conn.close()

    lyon = "  oo_memories, into the namespace memory: "
    left = "1 left out: no active fact of the memory holds them as they read now"
    gone = "1 removed: the store held them with a text no active fact of the memory holds now"
    # The first folder moves, the second stays; the fact is deleted: the next run removes the first folder's copy.
    world = _world(tmp_path / "deleted", monkeypatch)
    try:
        repo, package = world.places
        first = _folder(world, repo, "rag/chroma_db", memories("Lives in Lyon", _vec(1, 0)))
        second = _folder(world, package, "rag/chroma_db", memories("Lives in Paris", _vec(0, 1)))
        _facts(world, active=[("f1", "Lives in Lyon")])
        monkeypatch.setattr(world.cm.os, "rename",
                            lambda src, dst: (_ for _ in ()).throw(OSError(5, "Input/output error"))
                            if Path(src) == second else rename(src, dst))
        assert world.dbe.main(["migrate"]) == 3
        said = capsys.readouterr()
        assert not first.exists() and second.is_dir(), "control: the first folder moved, the second stayed"
        assert f"{first}: 1 collection(s) moved into the knowledge base's store\n{lyon}1 item(s) written, 0 already " \
               "there kept\n" in said.out, said.out
        assert f"{second}: 1 collection(s) written into the knowledge base's store, the folder not moved: the next " \
               f"run of migrate moves it\n{lyon}0 item(s) written, 0 already there kept, {left}\n" in said.out
        rewrite(world, "UPDATE memory_facts SET active = 0 WHERE id = 'f1'")
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert _held(world, "memory", "oo_memories")[1] == [], "the deleted fact's copy, made from the other folder"
        assert f"{lyon}0 item(s) written, 0 already there kept, {left}, {gone}\n" in out, out
        assert not second.exists()
    finally:
        world.restore()
    # The run stops at the first folder; the fact goes back to that folder's text: the next run keeps that copy.
    world = _world(tmp_path / "reverted", monkeypatch)
    try:
        repo, package = world.places
        first = _folder(world, repo, "rag/chroma_db", memories("Lives in Lyon", _vec(1, 0)))
        second = _folder(world, package, "rag/chroma_db", memories("Lives in Paris", _vec(0, 1)))
        _facts(world, active=[("f1", "Lives in Paris")])
        monkeypatch.setattr(world.cm.os, "rename",
                            lambda src, dst: (_ for _ in ()).throw(OSError(5, "Input/output error"))
                            if Path(src) == first else rename(src, dst))
        assert world.dbe.main(["migrate"]) == 3
        said = capsys.readouterr()
        assert first.is_dir() and second.is_dir(), "control: stopped at the first folder"
        assert f"{second}: 1 collection(s) written into the knowledge base's store, the folder not moved" in (
            said.out), "every folder left after the one that stopped, said"
        rewrite(world, "UPDATE memory_facts SET text = 'Lives in Lyon' WHERE id = 'f1'")
        monkeypatch.setattr(world.cm.os, "rename", rename)
        assert world.dbe.main(["migrate"]) == 0
        out = capsys.readouterr().out
        assert _held(world, "memory", "oo_memories")[1] == [("f1", "Lives in Lyon", {"user_id": "local"},
                                                             _vec(1, 0))], "the copy of the text the fact reads now"
        assert f"{lyon}1 item(s) written, 0 already there kept, {gone}\n" in out, out
        assert f"{lyon}0 item(s) written, 0 already there kept, {left}\n" in out, out
        assert "hold no vector" not in out and not first.exists() and not second.exists()
    finally:
        world.restore()


def test_cm13c_a_rag_home_spelled_another_way_moves_as_the_place_spells_it(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    try:
        package = world.places[1]
        (tmp_path / "alias").symlink_to(package, target_is_directory=True)
        for spelling in (package / "rag" / ".." / "rag", tmp_path / "alias" / "rag"):
            folder = _folder(world, package, "rag/chroma_db", [_Collection("documents", [("d", "x", None,
                                                                                          _vec(1, 0))])])
            monkeypatch.setenv("RAG_HOME", str(spelling))
            assert world.dbe.main(["migrate"]) == 0, (spelling, capsys.readouterr().err)
            capsys.readouterr()
            assert not folder.exists(), spelling
            assert world.vdb.chroma_state(spelling / "chroma_db") == ("none", ""), "the platform's folder moved"
    finally:
        world.restore()
    # rag/ itself a link, RAG_HOME naming where it leads.
    world = _world(tmp_path / "linked", monkeypatch)
    try:
        package = world.places[1]
        real = tmp_path / "linked" / "big-disk" / "rag"
        _folder(world, real.parent, "rag/chroma_db", [_Collection("documents", [("d", "x", None, _vec(1, 0))])])
        world.chroma.folders[str(package / "rag" / "chroma_db")] = world.chroma.folders[str(real / "chroma_db")]
        (package / "rag").symlink_to(real, target_is_directory=True)
        monkeypatch.setenv("RAG_HOME", str(real))
        assert world.dbe.main(["migrate"]) == 0, capsys.readouterr().err
        capsys.readouterr()
        assert world.vdb.chroma_state(real / "chroma_db") == ("none", ""), "the platform's folder moved"
        assert [i[0] for i in _held(world, "documents", "documents")[1]] == ["d"]
    finally:
        world.restore()
