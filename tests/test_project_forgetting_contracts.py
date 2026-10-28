#!/usr/bin/env python3
"""Contracts for a project's deletion that keeps no words, and the wipe that removes the projects when named.

A deleted project kept its index: ``delete_project`` removed the folder (nothing overwritten) and the rows, and never
touched the knowledge base's store, where every chunk of every file -- its text, its metadata, its vector -- stayed.
Removing a file left its chunks too, and the project went on retrieving the removed file into every linked turn.
The server's log named the projects and their files. Now a deletion forgets: the index first, every file and output
overwritten before it goes, the rows, the folder, both logs emptied, the log naming ids only; a step that fails is
named and the next deletion's sweep finishes it, with what deletions made before this left. The account wipe
removes every project -- single-user mode, the install's one user -- only when the request names them and repeats
the phrase its preview gave; the export carries every project and what was drawn from its files.

  * PF1 -- A DELETED PROJECT KEEPS NO WORDS: read back in the bytes of every file under the data folder, the logs
    included, a reader holding them open; the bytes overwritten before they go; the builder's snapshot gone; the
    server's log never names them; another project keeps all of its own.
  * PF2 -- A REMOVED FILE IS NO LONGER RETRIEVED: its chunks, bytes and row gone, the project and its other file
    kept; a removed output overwritten.
  * PF3 -- WHAT DELETIONS BEFORE 191 LEFT IS SWEPT by the next deletion -- an orphan collection, a removed file's
    chunks, an orphan folder, a file no row names -- and the first start removes nothing.
  * PF4 -- A DELETION RACING AN INDEXING OR AN ADDITION KEEPS NOTHING: chunks embedded meanwhile never land, bytes
    written meanwhile never stay, a landing in flight is waited for.
  * PF5 -- A STEP THAT FAILS IS NAMED, AND THE NEXT DELETION FINISHES IT: the index unreachable, a file that cannot
    be overwritten; through the route's ``not_forgotten`` too.
  * PF6 -- WHILE THE LEGACY FOLDER WAITS FOR ITS MIGRATION the index is named as left, nothing is made or touched,
    and the first deletion after the migration removes it.
  * PF7 -- THE WIPE WITH THE PROJECTS NAMED FORGETS EVERY PROJECT and what earlier deletions left, counted, its
    labels leaving what stays; a project made meanwhile goes, or is named; without the word, nothing moves.
  * PF8 -- EVERY REFUSAL BEFORE ANYTHING IS REMOVED, through the routes: another mode or name, a missing phrase,
    moved counts or other ids under the same counts, a store that cannot be read.
  * PF9 -- THE EXPORT CARRIES EVERY PROJECT, its instructions, links, files' and outputs' manifests and what was
    drawn from them; what could not be read is named; in multi-user mode none, named; nothing is made to be read.
  * PF10 -- A FILE'S AND AN OUTPUT'S BYTES ARE DOWNLOADED from their project, under their own name; another
    project's id answers 404; the deletions answer ``not_forgotten``.
  * PF11 -- THE PANEL NAMES THE PROJECTS APART: its rules run under Node; the downloads and the warnings are wired.
  * PF12 -- WHAT A DELETION NAMES STARTS WITH THE INVENTORY'S LABEL, and the request's word is the inventory's.
  * PF13 -- NO DELETION OVERWRITES A FILE ANOTHER LINK NAMES, and none leaves a file it may not write: every copy
    of the package's overwrite, the same text, removes a shared file without its overwrite and says so, makes its
    own read-only file writable, overwrites and removes it.
  * PF14 -- A READ IN PROGRESS IS NAMED, THE DELETION NEVER WAITS FOR IT, and the next deletion empties what it
    kept.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_pf1_a_deleted_project_keeps_no_words": 5.0,
    "test_pf2_a_removed_file_is_no_longer_retrieved": 5.0,
    "test_pf3_what_deletions_before_191_left_is_swept_by_the_next": 5.0,
    "test_pf4_a_deletion_racing_an_indexing_or_an_addition_keeps_nothing": 10.0,
    "test_pf5_a_step_that_fails_is_named_and_the_next_deletion_finishes_it": 5.0,
    "test_pf6_while_the_legacy_folder_waits_the_index_is_named_and_swept_after": 5.0,
    "test_pf7_the_wipe_with_the_projects_named_forgets_every_project": 10.0,
    "test_pf8_every_refusal_before_anything_is_removed": 10.0,
    "test_pf9_the_export_carries_every_project": 5.0,
    "test_pf10_bytes_are_downloaded_from_their_project": 5.0,
    "test_pf11_the_panel_names_the_projects_apart": 30.0,
    "test_pf12_what_a_deletion_names_starts_with_the_inventorys_label": 3.0,
    "test_pf13_no_deletion_overwrites_a_file_another_link_names": 5.0,
    "test_pf14_a_read_in_progress_is_named_and_never_waited_for": 10.0,
}

# The words planted, each one token no file holds by chance, checked as written and lowercased.
NAME, DESCRIPTION, INSTRUCTIONS = "Zorbprojname", "Zorbprojdesc", "Zorbinstructs"
FILENAME, CONTENT = "Zorbfilename.txt", "Zorbcontentword"
OUTPUT, OUTPUT_BODY, OUTPUT_NOTE = "Zorboutname.py", "Zorboutbody", "Zorboutnote"
KEEP = "Zorbkeepword"
_WORDS = (NAME, DESCRIPTION, INSTRUCTIONS, "Zorbfilename", CONTENT, "Zorboutname", OUTPUT_BODY, OUTPUT_NOTE)
_PROJECT_LABELS = ("projects store (single-user)", "project files and saved outputs",
                   "project indexes (chunks of the project files)")


# How long the seam's connections wait for a lock, unless a statement says otherwise (PF14 raises it).
_TIMEOUT = 0.3


def _plain(path, *, check_same_thread=False, **_options):
    """A plain connection that, like many builds, does not overwrite deleted pages unless asked to."""
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=_TIMEOUT)
    conn.execute("PRAGMA secure_delete = OFF")
    return conn


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _held(folder, *words):
    """Which of ``words`` any file under ``folder`` still holds, as written or lowercased."""
    found = set()
    for path in Path(folder).rglob("*"):
        if path.is_file() and not path.is_symlink():
            raw = path.read_bytes()
            found.update(w for w in words if w.encode() in raw or w.lower().encode() in raw)
    return found


def _witness(path):
    """A descriptor open on a file's bytes, which reads them after the file is gone: a hard link would make it a
    file another link names, which no deletion overwrites."""
    return os.open(path, os.O_RDONLY)


def _zeroed(fd):
    """Whether the file ``fd`` reads holds nothing but zeros now, closing it."""
    try:
        return set(os.pread(fd, os.fstat(fd).st_size, 0)) == {0}
    finally:
        os.close(fd)


def _reader(path):
    """A connection that has read the database and stays open: the last connection's close no longer empties the
    write-ahead log, which only the deletion's own emptying then does."""
    conn = sqlite3.connect(str(path))
    conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    return conn


class _Embedder:
    """Three numbers per text, drawn from it; ``during`` runs inside the embedding of a file."""

    def __init__(self):
        self.during = None

    @staticmethod
    def _vector(text):
        return [float(len(text) % 7 + 1), 1.0, float(sum(map(ord, text)) % 5 + 1)]

    def embed(self, texts, show_progress=False):
        if self.during is not None:
            during, self.during = self.during, None
            during()
        return [self._vector(t) for t in texts]

    def embed_single(self, text):
        return self._vector(text)


class _Chunker:
    """One chunk per line, carrying the path it was drawn from, as the knowledge base's chunkers do."""

    def chunk(self, content, filepath, file_type):
        return [SimpleNamespace(content=line, chunk_index=i,
                                metadata={"source_file": filepath, "chunk_index": i, "file_type": file_type})
                for i, line in enumerate(content.splitlines()) if line.strip()]


def _world(tmp_path, *, wipe=False, routes=False, single=True, auth=True):
    """The projects store, its index and the knowledge base's store over ``tmp_path/data``, from their sources,
    wired as the package wires its defaults (the store's index the indexer's default); with ``wipe``, the account
    wipe, single-user or not (``auth=False``: not readable), one account ``u-1``; with ``routes``, the projects' and
    the users' routes."""
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    seeded = {
        "opti_oignon.db_utils": _module("opti_oignon.db_utils", safe_connect=_plain),
        "opti_oignon.config": _module("opti_oignon.config", DATA_DIR=data),
        "opti_oignon.rag.chunkers": _module("opti_oignon.rag.chunkers", Chunk=SimpleNamespace,
                                            get_chunker=lambda file_type: _Chunker()),
        "opti_oignon.rag.config": _module("opti_oignon.rag.config",
                                          get_config=lambda: SimpleNamespace(embedding=None)),
        "opti_oignon.rag.embeddings": _module("opti_oignon.rag.embeddings",
                                              OllamaEmbeddings=lambda config=None: _Embedder()),
    }
    targets = {
        "opti_oignon.vector_db": source("vector_db.py"),
        "opti_oignon.projects": source("projects.py"),
        "opti_oignon.project_context": source("project_context.py"),
    }
    if wipe or routes:
        seeded["opti_oignon.run_journal"] = _module("opti_oignon.run_journal", wipe=lambda: {"wiped": True})
        if auth:
            seeded["opti_oignon.auth"] = _module("opti_oignon.auth", auth_manager=SimpleNamespace(
                single_user_mode=single, count_users=lambda: 1,
                get_user=lambda uid: SimpleNamespace(id=uid) if uid == "u-1" else None))
        targets["opti_oignon.user_data_manager"] = source("user_data_manager.py")
    if routes:
        targets["opti_oignon.api.routes_users"] = source("api", "routes_users.py")
        targets["opti_oignon.api.routes_projects"] = source("api", "routes_projects.py")
    loaded, restore = isolate(targets=targets, seeded=seeded, packages=("opti_oignon.rag", "opti_oignon.api"))
    try:
        pj, pc = loaded["opti_oignon.projects"], loaded["opti_oignon.project_context"]
        pj._DEFAULT_DB_PATH = data / "projects.db"
        pj._DEFAULT_STORAGE_BASE = data / "projects"
        pc._DEFAULT_VECTORS_DB = data / "rag" / "vectors.db"
        pc._DEFAULT_CHROMA_BASE = data / "project_chroma"
        store = pj.ProjectStore(config_path=tmp_path / "absent.yaml")
        assert store._vectors_db is None, "the default store's index is the indexer's default"
        indexer, builder = pc.ProjectIndexer(store=store), pc.ProjectContextBuilder(store=store)
        builder._embedder, builder._embedder_checked = _Embedder(), True
        pj.project_store = pc.project_store = store
        pc.project_indexer, pc.project_context_builder = indexer, builder
        if routes:
            for name in ("opti_oignon.api.routes_projects",):
                loaded[name].project_store, loaded[name].project_indexer = store, indexer
    except BaseException:
        restore()
        raise
    return SimpleNamespace(data=data, loaded=loaded, pj=pj, pc=pc, vdb=loaded["opti_oignon.vector_db"],
                           store=store, indexer=indexer, builder=builder, restore=restore)


def _plant(world, *, name=NAME, description=DESCRIPTION, instructions=INSTRUCTIONS, filename=FILENAME,
           content=CONTENT, output=OUTPUT, body=OUTPUT_BODY, note=OUTPUT_NOTE):
    """A project holding words everywhere a project keeps them: its row, a file indexed (its chunks, their metadata
    and ids, its summary and key terms), an output and its description, a linked conversation."""
    project = world.store.create_project(name, description=description, system_instructions=instructions)
    text = f"{content} first line of the planted file.\n{content} second line, {content} again.\n"
    pf = world.store.add_file(project.id, filename, text.encode())
    result = world.indexer.index_file(project.id, pf.id)
    assert result.success, result.error
    po = world.store.add_output(project.id, output, f"print('{body}')\n".encode(), description=note)
    world.store.link_conversation(project.id, "conv-1")
    return SimpleNamespace(project=project, file=world.store.get_file(pf.id), output=po)


def _collections(world):
    """The names of the projects' namespace's collections in the knowledge base's store."""
    return sorted(c.name for c in world.pc._vector_client(None).list_collections())


def _retrieved(world, project_id, query=CONTENT):
    return [c.content for c in world.builder.retrieve_chunks(project_id, query, max_chunks=20, min_score=-1.0)]


def _refused(call, status):
    with pytest.raises(Exception) as caught:
        call()
    assert getattr(caught.value, "status", None) == status, caught.value


# ---------------------------------------------------------------------------
# PF1 -- a deleted project keeps no words
# ---------------------------------------------------------------------------
def test_pf1_a_deleted_project_keeps_no_words(tmp_path, caplog, monkeypatch):
    world = _world(tmp_path)
    readers = []
    try:
        # Both stores written once, then held open by readers that read them on their log: no close empties a log
        # after that, so every frame the planting appends stays in it until the deletion's own emptying.
        opener = world.store.create_project("opener")
        first = world.store.add_file(opener.id, "opener.txt", b"an opening line of neutral text\n")
        assert world.indexer.index_file(opener.id, first.id).success
        readers = [_reader(world.data / "projects.db"), _reader(world.data / "rag" / "vectors.db")]
        with caplog.at_level(logging.DEBUG):
            planted = _plant(world)
            kept = _plant(world, name=KEEP, description=KEEP, instructions=KEEP, filename=f"{KEEP}.txt",
                          content=KEEP, output=f"{KEEP}.py", body=KEEP, note=KEEP)
            assert _retrieved(world, planted.project.id), "the planted file was not retrieved before"
            client = world.pc._vector_client(None)
            assert client._snapshots, "the builder read no snapshot"
            # Both stores overwrite what they free, whatever their connection's default.
            conn = world.store._get_conn()
            try:
                assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1, "the projects store keeps pages"
            finally:
                conn.close()
            witness = _witness(planted.file.file_path)
            out_witness = _witness(planted.output.file_path)
            for log in (world.data / "projects.db-wal", world.data / "rag" / "vectors.db-wal"):
                assert CONTENT.encode() in log.read_bytes() or NAME.encode() in log.read_bytes(), f"{log}: no frame"
            synced = []
            real_fsync = os.fsync
            monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
            # The deletion alone: no sweep after it redoes what it should have done.
            assert world.store.forget_project(planted.project.id, sweep=False) == (True, [])
            monkeypatch.setattr(os, "fsync", real_fsync)
            assert len(synced) >= 2, "the zeros were written, never synced before the removal"
            assert world.store.delete_project(planted.project.id) is False, "a project is not deleted twice"

        assert _held(world.data, *_WORDS) == set(), f"a deleted project's words stay: {_held(world.data, *_WORDS)}"
        assert _zeroed(witness), "the file was removed, not overwritten first"
        assert _zeroed(out_witness), "the output was removed, not overwritten first"
        assert not (world.data / "projects" / planted.project.id).exists(), "the folder stays"
        assert f"project_{planted.project.id}" not in _collections(world), "the collection stays"
        assert not [key for key in client._snapshots
                    if key[0] not in {row[0] for row in _rows(world, "SELECT id FROM collections")}], (
            "the builder keeps a snapshot of the deleted collection")
        assert _retrieved(world, planted.project.id) == [], "the deleted project is still retrieved"
        # A query whose read began before a deletion never keeps its view of what the deletion erased.
        racing = _plant(world, name="Zorbracer", filename="Zorbracerfile.txt", content="Zorbracerchunk")
        real_snapshot = client._snapshot

        def snapshot(conn, row):
            world.store.forget_project(racing.project.id, sweep=False)
            return real_snapshot(conn, row)

        client._snapshot = snapshot
        try:
            _retrieved(world, racing.project.id, "Zorbracerchunk")
        finally:
            client._snapshot = real_snapshot
        held = [m for snap in client._snapshots.values() for m in snap.metadatas]
        assert not [m for m in held if "Zorbracer" in str(m)], "a query in flight kept the deleted project's view"
        logged = caplog.text
        assert not [w for w in (*_WORDS, KEEP) if w in logged or w.lower() in logged], (
            "the server's log names a project's words")
        # Another project keeps all of its own.
        assert _held(world.data, KEEP) == {KEEP}
        assert world.store.get_project(kept.project.id).name == KEEP and _retrieved(world, kept.project.id, KEEP)
        assert Path(kept.file.file_path).read_bytes().startswith(KEEP.encode())
        assert world.store.list_conversations(kept.project.id) == [
            {"conversation_id": "conv-1", "linked_at": world.store.list_conversations(kept.project.id)[0]["linked_at"]}]
    finally:
        for conn in readers:
            conn.close()
        world.restore()


def _rows(world, sql):
    conn = sqlite3.connect(str(world.data / "rag" / "vectors.db"))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# PF2 -- a removed file is no longer retrieved
# ---------------------------------------------------------------------------
def test_pf2_a_removed_file_is_no_longer_retrieved(tmp_path):
    world = _world(tmp_path)
    try:
        planted = _plant(world)
        other = world.store.add_file(planted.project.id, f"{KEEP}.txt", f"{KEEP} stays in the project.\n".encode())
        assert world.indexer.index_file(planted.project.id, other.id).success
        assert {CONTENT, KEEP} <= {w for c in _retrieved(world, planted.project.id, "line") for w in c.split()} | {
            w.strip(",.") for c in _retrieved(world, planted.project.id, "line") for w in c.split()}
        witness = _witness(planted.file.file_path)
        # The removal alone: no sweep after it redoes what it should have done.
        assert world.store.forget_file(planted.file.id, sweep=False) == (True, [])
        assert world.store.remove_file(planted.file.id) is False, "a file is not removed twice"
        assert _zeroed(witness), "the file was removed, not overwritten first"
        held = [m for snap in world.pc._vector_client(None)._snapshots.values() for m in snap.metadatas]
        assert not [m for m in held if "Zorbfilename" in str(m)], "the removed file's chunks stay in memory"
        # A query whose read began before a file's removal never keeps its view of the removed chunks.
        racing = world.store.add_file(planted.project.id, "Zorbracingfile.txt", b"Zorbracingchunk line of text.\n")
        assert world.indexer.index_file(planted.project.id, racing.id).success
        client = world.pc._vector_client(None)
        real_snapshot = client._snapshot

        def snapshot(conn, row):
            client._snapshot = real_snapshot
            world.store.forget_file(racing.id, sweep=False)
            return real_snapshot(conn, row)

        client._snapshot = snapshot
        try:
            _retrieved(world, planted.project.id, "Zorbracingchunk")
        finally:
            client._snapshot = real_snapshot
        held = [m for snap in client._snapshots.values() for m in snap.metadatas]
        assert not [m for m in held if "Zorbracingfile" in str(m)], "a query in flight kept the removed file's view"
        retrieved = _retrieved(world, planted.project.id, CONTENT)
        assert retrieved and not [c for c in retrieved if CONTENT in c], f"the removed file is retrieved: {retrieved}"
        file_words = ("Zorbfilename", CONTENT)
        assert _held(world.data, *file_words) == set(), _held(world.data, *file_words)
        assert world.store.get_project(planted.project.id).name == NAME, "the project went with its file"
        assert [f.id for f in world.store.list_files(planted.project.id)] == [other.id]
        # Through the public removal, as the route and the tools call it: chunks, bytes and row gone.
        third = world.store.add_file(planted.project.id, "Zorbthirdfile.txt", b"Zorbthirdchunk line of text.\n")
        assert world.indexer.index_file(planted.project.id, third.id).success
        assert world.store.remove_file(third.id) is True
        assert _held(world.data, "Zorbthirdfile", "Zorbthirdchunk") == set()
        # A removed output is overwritten, its description gone; the project keeps its own words.
        out_witness = _witness(planted.output.file_path)
        assert world.store.remove_output(planted.output.id) is True
        assert world.store.forget_output(planted.output.id) == (False, [])
        assert _zeroed(out_witness)
        assert _held(world.data, "Zorboutname", OUTPUT_BODY, OUTPUT_NOTE) == set()
        assert _held(world.data, NAME, DESCRIPTION, INSTRUCTIONS, KEEP) == {NAME, DESCRIPTION, INSTRUCTIONS, KEEP}
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF3 -- what deletions before 191 left is swept by the next one
# ---------------------------------------------------------------------------
def _old_delete_project(world, project_id):
    """A project's deletion as it was before 191: the folder removed with ``rmtree``, the rows by cascade, the
    index never touched."""
    import shutil

    shutil.rmtree(world.data / "projects" / project_id)
    conn = world.store._get_conn()
    try:
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
    finally:
        conn.close()


def _old_remove_file(world, file_id):
    """A file's removal as it was before 191: the bytes unlinked, the row deleted, the chunks left."""
    pf = world.store.get_file(file_id)
    Path(pf.file_path).unlink()
    conn = world.store._get_conn()
    try:
        conn.execute("DELETE FROM project_files WHERE id = ?", (file_id,))
        conn.commit()
    finally:
        conn.close()


def test_pf3_what_deletions_before_191_left_is_swept_by_the_next(tmp_path):
    world = _world(tmp_path)
    try:
        gone = _plant(world)
        _old_delete_project(world, gone.project.id)
        live = _plant(world, name=KEEP, description=KEEP, instructions=KEEP, filename="Zorbstalefile.txt",
                      content="Zorbstalechunk", output=f"{KEEP}.py", body=KEEP, note=KEEP)
        _old_remove_file(world, live.file.id)
        kept_file = world.store.add_file(live.project.id, f"{KEEP}.txt", f"{KEEP} stays here.\n".encode())
        assert world.indexer.index_file(live.project.id, kept_file.id).success
        # A folder of a project gone whose removal failed, and a file in a live folder that no row names.
        orphan = world.data / "projects" / "abcdef12-345"
        (orphan / "files").mkdir(parents=True)
        (orphan / "files" / "abcdef12-999_Zorborphanfolder.txt").write_bytes(b"Zorborphanfolder")
        stray = world.data / "projects" / live.project.id / "files" / "fedcba98-765_Zorbstraybytes.txt"
        stray.write_bytes(b"Zorbstraybytes")
        residue = ("Zorbcontentword", "Zorbstalechunk", "Zorborphanfolder", "Zorbstraybytes")
        assert _held(world.data, *residue) == set(residue), "the residue was not built"
        # The first start removes nothing.
        restarted = world.pj.ProjectStore(config_path=tmp_path / "absent.yaml")
        assert [p.id for p in restarted.list_projects()] == [live.project.id]
        assert _held(world.data, *residue) == set(residue), "a start removed something"
        # The next deletion sweeps it: here a project made and deleted.
        doomed = world.store.create_project("Zorbdoomed")
        assert world.store.delete_project(doomed.id) is True
        assert _held(world.data, *residue, "Zorbdoomed") == set(), _held(world.data, *residue, "Zorbdoomed")
        assert _collections(world) == [f"project_{live.project.id}"]
        assert _retrieved(world, live.project.id, KEEP) and _held(world.data, KEEP) == {KEEP}
        assert not orphan.exists() and Path(kept_file.file_path).exists() and Path(live.output.file_path).exists()
        # A folder not named like a project, and a file not named like the store's, are not the store's: kept.
        foreign = world.data / "projects" / "not-a-project"
        foreign.mkdir()
        (world.data / "projects" / live.project.id / "files" / "README").write_bytes(b"x")
        assert world.store.sweep_orphans() == []
        assert foreign.exists() and (world.data / "projects" / live.project.id / "files" / "README").exists()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF4 -- a deletion racing an indexing or an addition keeps nothing
# ---------------------------------------------------------------------------
def test_pf4_a_deletion_racing_an_indexing_or_an_addition_keeps_nothing(tmp_path):
    world = _world(tmp_path)
    try:
        # A project deleted while one of its files is embedded: the chunks never land, no collection is made.
        planted = _plant(world)
        assert world.store.forget_project(planted.project.id)[0]
        project = world.store.create_project("Zorbracing")
        pf = world.store.add_file(project.id, "Zorbracefile.txt", b"Zorbracechunk is the racing file's text.\n")
        embedder = world.indexer._get_embedder()
        embedder.during = lambda: world.store.forget_project(project.id)
        result = world.indexer.index_file(project.id, pf.id)
        assert not result.success and "removed while it was indexed" in result.error, result
        assert _collections(world) == [] and _held(world.data, "Zorbracechunk", "Zorbracefile") == set()
        # A file removed while it is embedded: the same, the project kept.
        project = world.store.create_project("Zorbracing2")
        pf = world.store.add_file(project.id, "Zorbracefile.txt", b"Zorbracechunk is the racing file's text.\n")
        embedder.during = lambda: world.store.forget_file(pf.id)
        assert not world.indexer.index_file(project.id, pf.id).success
        assert _held(world.data, "Zorbracechunk", "Zorbracefile") == set() and world.store.get_project(project.id)
        # A deletion that comes while the chunks land, the row read again, waits for them, then removes them.
        pf = world.store.add_file(project.id, "Zorbracefile.txt", b"Zorbracechunk is the racing file's text.\n")
        real_collection = world.indexer._get_collection
        deleter = threading.Thread(target=lambda: world.store.forget_project(project.id, sweep=False))

        def collection(project_id):
            deleter.start()
            deleter.join(timeout=0.5)
            return real_collection(project_id)

        world.indexer._get_collection = collection
        try:
            world.indexer.index_file(project.id, pf.id)
        finally:
            deleter.join(timeout=5.0)
            world.indexer._get_collection = real_collection
        assert not deleter.is_alive()
        assert _collections(world) == [] and _held(world.data, "Zorbracechunk", "Zorbracefile") == set(), (
            _collections(world))
        # A file, or an output, added while its project is deleted leaves no bytes behind.
        for add in (lambda pid: world.store.add_file(pid, "Zorbaddedfile.txt", b"Zorbaddedbytes"),
                    lambda pid: world.store.add_output(pid, "Zorbaddedfile.py", b"Zorbaddedbytes")):
            project = world.store.create_project("Zorbadding")
            real_list = world.store.list_files
            world.store.list_files = lambda pid, _real=real_list: (world.store.forget_project(pid), _real(pid))[1]
            real_project = world.store.get_project
            first = {"done": False}

            def get_project(pid, _real=real_project, _first=first):
                found = _real(pid)
                if not _first["done"]:
                    _first["done"] = True
                    world.store.forget_project(pid)
                return found

            world.store.get_project = get_project
            try:
                with pytest.raises(ValueError, match="Project not found"):
                    add(project.id)
            finally:
                world.store.list_files, world.store.get_project = real_list, real_project
            assert not (world.data / "projects" / project.id).exists(), "a folder was made again"
            assert _held(world.data, "Zorbaddedbytes", "Zorbaddedfile") == set()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF5 -- a step that fails is named, and the next deletion finishes it
# ---------------------------------------------------------------------------
def test_pf5_a_step_that_fails_is_named_and_the_next_deletion_finishes_it(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root removes from a read-only folder")
    world = _world(tmp_path, routes=True)
    try:
        client = world.pc._vector_client(None)
        planted = _plant(world)
        real_delete = client.delete_collection

        def locked(name):
            raise sqlite3.OperationalError("database is locked")

        client.delete_collection = locked
        try:
            deleted, left = world.store.forget_project(planted.project.id)
        finally:
            client.delete_collection = real_delete
        assert deleted and left == [f"project indexes (chunks of the project files) (project {planted.project.id}'s, "
                                    "not removed: OperationalError)"], left
        assert world.store.get_project(planted.project.id) is None and not Path(planted.file.file_path).exists()
        assert CONTENT in str(_held(world.data, CONTENT)), "the index was not left, as said"
        # A folder that refuses its files' removal: each overwritten all the same, named, the project's rows gone.
        second = _plant(world, name="Zorbsecond", filename="Zorbsecondfile.txt", content="Zorbsecondchunk")
        files = Path(second.file.file_path).parent
        os.chmod(files, 0o500)
        try:
            routes = world.loaded["opti_oignon.api.routes_projects"]
            answer = routes.delete_project(second.project.id)
        finally:
            os.chmod(files, 0o700)
        assert answer["status"] == "ok" and answer["not_forgotten"] == [
            f"project files and saved outputs (project {second.project.id}'s, not all removed: PermissionError)"], (
            answer)
        # The index left by the first is swept by the second; the file left holds zeros.
        assert _collections(world) == [] and _held(world.data, CONTENT, "Zorbsecondchunk") == set()
        assert Path(second.file.file_path).exists() and set(Path(second.file.file_path).read_bytes()) == {0}
        # The next deletion removes the file left.
        third = world.store.create_project("Zorbthird")
        assert world.store.forget_project(third.id) == (True, [])
        assert not (world.data / "projects" / second.project.id).exists()
        assert _held(world.data, *_WORDS, "Zorbsecondfile", "Zorbsecond", "Zorbthird") == set()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF6 -- while the legacy folder waits for its migration
# ---------------------------------------------------------------------------
def test_pf6_while_the_legacy_folder_waits_the_index_is_named_and_swept_after(tmp_path):
    world = _world(tmp_path)
    try:
        legacy = world.data / "project_chroma"
        legacy.mkdir()
        (legacy / "chroma.sqlite3").write_bytes(b"SQLite format 3\x00" + b"Zorblegacychunk" * 4)
        project = world.store.create_project(NAME, description=DESCRIPTION)
        pf = world.store.add_file(project.id, FILENAME, f"{CONTENT} waits.\n".encode())
        world.indexer._get_embedder().during = lambda: pytest.fail("a file embedded while the index waits")
        assert not world.indexer.index_file(project.id, pf.id).success, "the index took a file while waiting"
        deleted, left = world.store.forget_project(project.id)
        assert deleted and len(left) == 1 and left[0].startswith(
            "project indexes (chunks of the project files) (in the legacy folder, which waits for `") and (
            "the first deletion after it removes them" in left[0]), left
        assert world.store.get_project(project.id) is None and _held(world.data / "projects", *_WORDS) == set()
        assert not (world.data / "rag" / "vectors.db").exists(), "the knowledge base's store was made"
        assert (legacy / "chroma.sqlite3").read_bytes().count(b"Zorblegacychunk") == 4, "the legacy folder touched"
        # The migration moves the collection over: the first deletion after it removes what it carried.
        legacy.rename(legacy.with_name("project_chroma-moved"))
        migrated = world.vdb.Client(world.data / "rag" / "vectors.db", "projects")
        migrated.get_or_create_collection(f"project_{project.id}").add(
            ids=[f"{project.id}::x::0"], embeddings=[[1.0, 1.0, 1.0]], documents=["Zorblegacychunk moved"],
            metadatas=[{"source_file": "x"}])
        other = world.store.create_project("Zorbafter")
        assert world.store.forget_project(other.id) == (True, [])
        assert _collections(world) == [] and _held(world.data / "rag", "Zorblegacychunk") == set()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF7 -- the wipe with the projects named
# ---------------------------------------------------------------------------
def test_pf7_the_wipe_with_the_projects_named_forgets_every_project(tmp_path):
    world = _world(tmp_path, wipe=True)
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        first, second = _plant(world), _plant(world, name="Zorbsecond", filename="Zorbsecondfile.txt",
                                              content="Zorbsecondchunk")
        gone = _plant(world, name="Zorbresidue", filename="Zorbresiduefile.txt", content="Zorbresiduechunk")
        _old_delete_project(world, gone.project.id)
        # Without the word, the projects are left and named.
        plain = udm.UserDataDeleter().delete_all("local")
        assert set(_PROJECT_LABELS) <= set(plain["not_covered"]) and "projects" not in plain["content_removed"]
        assert len(world.store.project_ids()) == 2
        preview = udm.wipe_preview("local", ["projects"])
        counted = preview["counts"]["projects"]
        assert {k: counted[k] for k in ("projects", "files", "outputs")} == {"projects": 2, "files": 2, "outputs": 2}
        assert len(counted["code"]) == 4 and counted["code"] == world.store.fingerprint()
        assert preview["phrase"] == f"remove 2 projects {counted['code']}"
        assert preview["notice"] == udm.PROJECTS_NOTICE
        assert set(_PROJECT_LABELS) <= set(preview["removed"]) and not set(_PROJECT_LABELS) & set(preview["left"])
        assert preview["on_request"]["projects"] == list(_PROJECT_LABELS)
        # Both words: one phrase for both, in the words' order; a tree without notes counts none.
        both = udm.wipe_preview("local", ["projects", "notes"])
        assert both["phrase"] == f"remove 0 notes and 2 projects {counted['code']}", both["phrase"]
        assert udm._phrase({"projects": {"projects": 1, "code": ""}}) == "remove 1 project"
        # A project made while the wipe runs goes with the others.
        real_forget = world.store.forget_project
        made = {"n": 0}

        def forget(project_id, **kwargs):
            if made["n"] == 0:
                made["n"] += 1
                world.store.create_project("Zorblatecomer")
            return real_forget(project_id, **kwargs)

        world.store.forget_project = forget
        try:
            result = udm.UserDataDeleter().delete_all("local", remove=["projects"], confirm=preview["phrase"])
        finally:
            world.store.forget_project = real_forget
        assert result["content_removed"] == {"projects": 3}, result["content_removed"]
        assert not set(_PROJECT_LABELS) & set(result["not_covered"]), result["not_covered"]
        words = (*_WORDS, "Zorbsecond", "Zorbsecondfile", "Zorbsecondchunk", "Zorbresidue", "Zorbresiduechunk",
                 "Zorblatecomer")
        assert _held(world.data, *words) == set(), _held(world.data, *words)
        assert world.store.project_ids() == [] and _collections(world) == []
        assert not [p for p in (world.data / "projects").iterdir()], "a project's folder stays"
        # One that keeps arriving is named, never taken for removed.
        world.store.create_project("Zorbstayer")
        preview = udm.wipe_preview("local", ["projects"])

        def always(project_id, **kwargs):
            world.store.create_project("Zorbstayer")
            return real_forget(project_id, **kwargs)

        world.store.forget_project = always
        try:
            result = udm.UserDataDeleter().delete_all("local", remove=["projects"], confirm=preview["phrase"])
        finally:
            world.store.forget_project = real_forget
        assert "projects (1 made while the wipe ran, left)" in result["not_covered"], result
        world.store.forget_project = real_forget
        for project_id in world.store.project_ids():
            real_forget(project_id)
        # One project that cannot be deleted: named, the others deleted and counted, the sweep run.
        a, b = _plant(world, name="Zorbfirst"), _plant(world, name="Zorbstuck")
        gone = _plant(world, name="Zorbresidue2", filename="Zorbresidue2file.txt", content="Zorbresidue2chunk")
        _old_delete_project(world, gone.project.id)

        def stuck(project_id, **kwargs):
            if project_id == b.project.id:
                raise sqlite3.OperationalError("database is locked")
            return real_forget(project_id, **kwargs)

        world.store.forget_project = stuck
        try:
            result = udm.UserDataDeleter().delete_all("local", remove=["projects"],
                                                      confirm=udm.wipe_preview("local", ["projects"])["phrase"])
        finally:
            world.store.forget_project = real_forget
        assert result["content_removed"] == {"projects": 1}, result
        assert f"projects (project {b.project.id} not deleted: OperationalError)" in result["not_covered"]
        assert "projects (0 made while the wipe ran, left)" not in result["not_covered"]
        assert not [line for line in result["not_covered"] if "made while the wipe ran" in line], result
        assert world.store.project_ids() == [b.project.id] and world.store.get_project(a.project.id) is None
        assert _held(world.data, "Zorbresidue2chunk", "Zorbfirst") == set(), "the sweep did not run"
        # The one that fails, beside one that keeps arriving through the passes: only the arrivals are counted.

        def stuck_and_arriving(project_id, **kwargs):
            world.store.create_project("Zorbarriving")
            return stuck(project_id, **kwargs)

        world.store.forget_project = stuck_and_arriving
        try:
            result = udm.UserDataDeleter().delete_all("local", remove=["projects"],
                                                      confirm=udm.wipe_preview("local", ["projects"])["phrase"])
        finally:
            world.store.forget_project = real_forget
        # Each of the three passes tries the stuck one again: three arrivals, the last pass's, are left.
        assert "projects (3 made while the wipe ran, left)" in result["not_covered"], result["not_covered"]
        # An index the first deletion could not reach and the final sweep then removed is not said.
        for project_id in world.store.project_ids():
            real_forget(project_id)
        c = _plant(world, name="Zorbonce")
        client = world.pc._vector_client(None)
        real_delete, calls = client.delete_collection, {"n": 0}

        def once(name):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return real_delete(name)

        client.delete_collection = once
        try:
            result = udm.UserDataDeleter().delete_all("local", remove=["projects"],
                                                      confirm=udm.wipe_preview("local", ["projects"])["phrase"])
        finally:
            client.delete_collection = real_delete
        assert calls["n"] >= 2 and _collections(world) == [], _collections(world)
        assert not [line for line in result["not_covered"] if line.startswith(_PROJECT_LABELS[2])], result
        assert result["content_removed"] == {"projects": 1} and world.store.get_project(c.project.id) is None
        # A deletion that fails once is tried again at the next pass: deleted, counted, not named.
        d, e = _plant(world, name="Zorbretried"), _plant(world, name="Zorbsteady")
        tries = {"n": 0}

        def flaky(project_id, **kwargs):
            if project_id == d.project.id and tries["n"] == 0:
                tries["n"] += 1
                raise sqlite3.OperationalError("database is locked")
            return real_forget(project_id, **kwargs)

        world.store.forget_project = flaky
        try:
            result = udm.UserDataDeleter().delete_all("local", remove=["projects"],
                                                      confirm=udm.wipe_preview("local", ["projects"])["phrase"])
        finally:
            world.store.forget_project = real_forget
        assert result["content_removed"] == {"projects": 2} and world.store.project_ids() == [], result
        assert not [line for line in result["not_covered"] if "not deleted" in line], result["not_covered"]
        # A listing that fails between passes, then a sweep that fails: each named, what went still counted.
        f = _plant(world, name="Zorblisted")
        real_ids, real_sweep, listed = world.store.project_ids, world.store.sweep_orphans, {"n": 0}

        def ids():
            listed["n"] += 1
            if listed["n"] > 2:  # the preview's and the wipe's first listings pass
                raise sqlite3.OperationalError("database is locked")
            return real_ids()

        world.store.project_ids = ids
        world.store.sweep_orphans = lambda **kw: (_ for _ in ()).throw(OSError("unreadable"))
        try:
            phrase = udm.wipe_preview("local", ["projects"])["phrase"]
            listed["n"] = 1
            result = udm.UserDataDeleter().delete_all("local", remove=["projects"], confirm=phrase)
        finally:
            world.store.project_ids, world.store.sweep_orphans = real_ids, real_sweep
        assert result["content_removed"] == {"projects": 1} and world.store.get_project(f.project.id) is None, result
        assert "projects (not listed again: OperationalError; any project left is not counted)" in result[
            "not_covered"] and "projects (what earlier deletions left, not swept: OSError)" in result["not_covered"]
        # The logs are emptied once, at the end, under a reader that holds the index open: the residue the final
        # sweep removes leaves the store's file; a read that ended before the end is not said.
        g = _plant(world, name="Zorbender")
        late = _plant(world, name="Zorbresidue3", filename="Zorbresidue3file.txt", content="Zorbresidue3chunk")
        _old_delete_project(world, late.project.id)
        holder = _reader(world.data / "rag" / "vectors.db")
        busy = sqlite3.connect(str(world.data / "projects.db"), isolation_level=None)
        busy.execute("BEGIN")
        busy.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()

        def ending(project_id, **kwargs):
            gone = real_forget(project_id, **kwargs)
            if busy.in_transaction:
                busy.execute("ROLLBACK")
            return gone

        world.store.forget_project = ending
        try:
            result = udm.UserDataDeleter().delete_all("local", remove=["projects"],
                                                      confirm=udm.wipe_preview("local", ["projects"])["phrase"])
        finally:
            world.store.forget_project = real_forget
            busy.close()
        try:
            assert result["content_removed"] == {"projects": 1} and world.store.get_project(g.project.id) is None
            assert not [line for line in result["not_covered"] if "in progress" in line], result["not_covered"]
            assert _held(world.data / "rag", "Zorbresidue3chunk", "Zorbresidue3file") == set(), (
                "the final sweep's residue stays in the store's file")
        finally:
            holder.close()
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF8 -- every refusal before anything is removed
# ---------------------------------------------------------------------------
def test_pf8_every_refusal_before_anything_is_removed(tmp_path):
    world = _world(tmp_path, wipe=True, routes=True)
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        routes = world.loaded["opti_oignon.api.routes_users"]
        planted = _plant(world)
        deleter = udm.UserDataDeleter()
        phrase = udm.wipe_preview("local", ["projects"])["phrase"]
        _refused(lambda: deleter.delete_all("local", remove=["projects"]), 422)
        _refused(lambda: deleter.delete_all("u-2", remove=["projects"], confirm=phrase), 409)
        _refused(lambda: deleter.delete_all("local", remove=["projects"], confirm="remove 1 project"), 409)
        # The same counts over other ids: a file replaced by another.
        assert world.store.forget_file(planted.file.id)[0]
        world.store.add_file(planted.project.id, "Zorbswapped.txt", b"swapped")
        assert world.store.counts() == {"projects": 1, "files": 1, "outputs": 1}
        _refused(lambda: deleter.delete_all("local", remove=["projects"], confirm=phrase), 409)
        # An output come since the preview.
        phrase = udm.wipe_preview("local", ["projects"])["phrase"]
        world.store.add_output(planted.project.id, "late.py", b"late")
        _refused(lambda: deleter.delete_all("local", remove=["projects"], confirm=phrase), 409)
        assert world.store.project_ids() == [planted.project.id], "a refused wipe removed a project"
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={})
        for call, status in (
            (routes.delete_user_data("local", request, remove=["projects"], confirm=None,
                                     current_user={"sub": "local"}), 422),
            (routes.delete_user_data("local", request, remove=["projects"], confirm=phrase,
                                     current_user={"sub": "local"}), 409),
        ):
            with pytest.raises(Exception) as caught:
                asyncio.run(call)
            assert getattr(caught.value, "status_code", None) == status, caught.value
        shown = asyncio.run(routes.preview_user_data_wipe("local", request, remove=["projects"],
                                                          current_user={"sub": "local"}))
        assert shown["phrase"] == f"remove 1 project {world.store.fingerprint()}"
        # A store that cannot be read: 503, nothing removed.
        real_counts = world.store.counts
        world.store.counts = lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("file is not a database"))
        try:
            _refused(lambda: udm.wipe_preview("local", ["projects"]), 503)
            _refused(lambda: deleter.delete_all("local", remove=["projects"], confirm=shown["phrase"]), 503)
        finally:
            world.store.counts = real_counts
        assert world.store.project_ids() == [planted.project.id] and _held(world.data, NAME) == {NAME}
    finally:
        world.restore()
    # Outside single-user mode, and when the mode cannot be read: refused, nothing read.
    for single, auth in ((False, True), (True, False)):
        world = _world(tmp_path / f"m{int(single)}{int(auth)}", wipe=True, single=single, auth=auth)
        try:
            udm = world.loaded["opti_oignon.user_data_manager"]
            _plant(world)
            _refused(lambda: udm.wipe_preview("local", ["projects"]), 409)
            _refused(lambda: udm.UserDataDeleter().delete_all("local", remove=["projects"],
                                                              confirm="remove 1 project"), 409)
            assert len(world.store.project_ids()) == 1
        finally:
            world.restore()


# ---------------------------------------------------------------------------
# PF9 -- the export carries every project
# ---------------------------------------------------------------------------
def test_pf9_the_export_carries_every_project(tmp_path):
    world = _world(tmp_path, wipe=True)
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        # A store that is not there: none exported, nothing made.
        data = udm.UserDataExporter().export("local")
        assert data["projects"] == [] and not (world.data / "projects.db").exists(), "the export made the store"
        planted, other = _plant(world), _plant(world, name=KEEP)
        data = udm.UserDataExporter().export("local")
        meta = data["export_metadata"]
        assert meta["format_version"] == "1.4" and meta["failed"] == []
        exported = {p["id"]: p for p in data["projects"]}
        entry = exported[planted.project.id]
        assert (entry["name"], entry["description"], entry["system_instructions"]) == (
            NAME, DESCRIPTION, INSTRUCTIONS)
        assert entry["conversations"] == ["conv-1"]
        (f,) = entry["files"]
        assert f["filename"] == FILENAME and f["indexed"] and f["chunk_count"] == 2 and CONTENT in f["summary"]
        assert CONTENT.lower() in [t.lower() for t in f["key_terms"]], f["key_terms"]
        (o,) = entry["outputs"]
        assert (o["filename"], o["description"]) == (OUTPUT, OUTPUT_NOTE)
        assert set(exported) == {planted.project.id, other.project.id}
        assert not set(_PROJECT_LABELS[:2]) & set(meta["not_covered"]), meta["not_covered"]
        assert _PROJECT_LABELS[2] in meta["not_covered"] and udm.EXPORT_NOT_COVERED[2] in meta["not_covered"]
        # A project that cannot be read is named, the others kept; a store that cannot, never taken for empty.
        real_get = world.store.get_project
        world.store.get_project = lambda pid: (_ for _ in ()).throw(OSError("unreadable")) if (
            pid == planted.project.id) else real_get(pid)
        try:
            data = udm.UserDataExporter().export("local")
        finally:
            world.store.get_project = real_get
        assert [p["id"] for p in data["projects"]] == [other.project.id]
        assert data["export_metadata"]["failed"] == [f"project {planted.project.id} (not exported: OSError)"]
        world.store.project_ids = lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("file is not a database"))
        data = udm.UserDataExporter().export("local")
        assert data["projects"] == [] and data["export_metadata"]["failed"] == [
            "projects (not exported: DatabaseError)"]
    finally:
        world.restore()
    world = _world(tmp_path / "multi", wipe=True, single=False)
    try:
        udm = world.loaded["opti_oignon.user_data_manager"]
        _plant(world)
        data = udm.UserDataExporter().export("local")
        assert data["projects"] == [] and set(_PROJECT_LABELS) <= set(data["export_metadata"]["not_covered"])
        assert udm.EXPORT_NOT_COVERED[2] not in data["export_metadata"]["not_covered"]
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF10 -- bytes are downloaded from their project
# ---------------------------------------------------------------------------
def test_pf10_bytes_are_downloaded_from_their_project(tmp_path):
    world = _world(tmp_path, routes=True)
    try:
        routes = world.loaded["opti_oignon.api.routes_projects"]
        project = world.store.create_project("p")
        name = "my file \"quoted\" \u00e9t\u00e9.txt"
        pf = world.store.add_file(project.id, name, b"the bytes, as uploaded")
        po = world.store.add_output(project.id, "out put.py", b"print(1)")
        answer = routes.download_project_file(project.id, pf.id)
        assert answer.body == b"the bytes, as uploaded" and answer.media_type == "application/octet-stream"
        assert answer.headers["content-disposition"] == (
            "attachment; filename*=UTF-8''my%20file%20%22quoted%22%20%C3%A9t%C3%A9.txt"), answer.headers
        answer = routes.download_project_output(project.id, po.id)
        assert answer.body == b"print(1)" and "out%20put.py" in answer.headers["content-disposition"]
        other = world.store.create_project("q")
        for call in (lambda: routes.download_project_file(other.id, pf.id),
                     lambda: routes.download_project_output(other.id, po.id),
                     lambda: routes.download_project_file(project.id, "missing")):
            with pytest.raises(Exception) as caught:
                call()
            assert getattr(caught.value, "status_code", None) == 404, caught.value
        # The row names another place (an install moved): the bytes come from the project's folder, the deletion's.
        elsewhere = tmp_path / "elsewhere.txt"
        elsewhere.write_bytes(b"ELSEWHERE")
        conn = world.store._get_conn()
        try:
            conn.execute("UPDATE project_files SET file_path = ? WHERE id = ?", (str(elsewhere), pf.id))
            conn.commit()
        finally:
            conn.close()
        assert routes.download_project_file(project.id, pf.id).body == b"the bytes, as uploaded"
        # A link in the folder is never followed.
        stored = next((world.data / "projects" / project.id / "files").iterdir())
        stored.unlink()
        stored.symlink_to(elsewhere)
        with pytest.raises(Exception) as caught:
            routes.download_project_file(project.id, pf.id)
        assert getattr(caught.value, "status_code", None) == 404
        stored.unlink()
        with pytest.raises(Exception) as caught:
            routes.download_project_file(project.id, pf.id)
        assert getattr(caught.value, "status_code", None) == 404 and elsewhere.read_bytes() == b"ELSEWHERE"
        # A project's folder reached through a link: never read through, and its deletion says what stays.
        moved = world.store.create_project("moved")
        kept = world.store.add_file(moved.id, "kept.txt", b"Zorblinkedbytes line of text.\n")
        target = tmp_path / "other-disk"
        (world.data / "projects" / moved.id).rename(target)
        (world.data / "projects" / moved.id).symlink_to(target)
        with pytest.raises(Exception) as caught:
            routes.download_project_file(moved.id, kept.id)
        assert getattr(caught.value, "status_code", None) == 404
        answer = routes.delete_project(moved.id)
        assert answer["not_forgotten"] == [f"project files and saved outputs (project {moved.id}'s: removed, not "
                                           "overwritten -- a link: what it led to is left)"], answer
        assert not (world.data / "projects" / moved.id).exists() and (target / "files").is_dir()
        assert routes.delete_project_output(project.id, po.id) == {
            "status": "ok", "deleted": po.id, "not_forgotten": []}
        assert routes.delete_project_file(project.id, pf.id) == {
            "status": "ok", "deleted": pf.id, "not_forgotten": []}
        assert routes.delete_project(project.id) == {
            "status": "ok", "deleted": project.id, "not_forgotten": []}
        # Every route that takes the store's write lock runs off the event loop: a plain function, or its write
        # handed to the thread pool.
        import ast

        tree = ast.parse(source("api", "routes_projects.py").read_text(encoding="utf-8"))
        handlers = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for name in ("create_project", "delete_project", "download_project_file", "delete_project_file",
                     "download_project_output", "delete_project_output", "index_project_file", "reindex_project"):
            assert isinstance(handlers[name], ast.FunctionDef), f"{name} holds the event loop"
        for name, write in (("upload_project_file", "add_file"), ("upload_project_output", "add_output")):
            body = ast.unparse(handlers[name])
            assert f"await run_in_threadpool(project_store.{write}," in body, f"{name} writes on the event loop"
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF11 -- the panel names the projects apart
# ---------------------------------------------------------------------------
_RULES = "frontend/src/lib/settings/yourData.ts"
_DRIVER = r"""
const m = await import(process.env.OO_YOUR_DATA);
const ok = (cond, what) => { if (!cond) { console.log('FAIL ' + what); process.exit(1); } };
const gate = { exportOffered: true, typedWipe: 'delete my data', named: ['projects'], phrase: 'remove 2 projects b2c1',
               typedPhrase: '', busy: false };
ok(m.wipeQuery(['projects', 'notes'], 'remove 1 note and 2 projects b2c1') ===
   '?remove=notes&remove=projects&confirm=remove%201%20note%20and%202%20projects%20b2c1', 'both words, sorted');
ok(!m.canWipe(gate), 'the phrase asked for the projects');
ok(m.canWipe({ ...gate, typedPhrase: 'remove 2 projects b2c1' }), 'the phrase typed');
ok(!m.canWipe({ ...gate, phrase: undefined, typedPhrase: 'remove 2 projects b2c1' }), 'no phrase, no wipe');
ok(m.counted(1, 'project') === '1 project' && m.counted(0, 'file') === '0 files' && m.counted(3, 'note') === '3 notes',
   'counted');
console.log('PASS rules');
"""


def test_pf11_the_panel_names_the_projects_apart():
    from _frontend import read, run_ts

    run_ts({"OO_YOUR_DATA": _RULES}, _DRIVER, "rules")
    panel = read("frontend/src/lib/components/settings/YourDataPanel.svelte")
    assert 'label="Also delete my projects"' in panel and "on:change={toggleProjects}" in panel
    assert "[...(notes ? ['notes'] : []), ...(projects ? ['projects'] : [])]" in panel, "the words named"
    assert "{#if named.length > 0 && preview?.notice}" in panel, "the notice shown for any word named"
    assert "to delete your ${named.join(' and ')}" in panel
    client = read("frontend/src/lib/api/projects.ts")
    assert "/files/${fileId}/content" in client and "/outputs/${outputId}/content" in client
    assert "URL.createObjectURL(await apiGetBlob(path))" in client, "the download not through the client"
    assert "setTimeout(() => URL.revokeObjectURL(url), 1000);" in client, "the download revoked before it starts"
    assert "counted(projectsCounted.projects, 'project')" in panel and "counted(notesCounted.notes, 'note')" in panel
    blob = read("frontend/src/lib/api/client.ts")
    assert "export async function apiGetBlob(path: string): Promise<Blob> {" in blob
    assert "headers: { 'Accept': 'application/octet-stream', ...authHeaders() }," in blob, "no credentials"
    manager = read("frontend/src/lib/components/panels/FileManager.svelte")
    assert "downloadProjectFile(projectId, fileId, filename)" in manager
    assert "on:click={() => handleDownload(f.id, f.filename)}" in manager, "the file's download wired to nothing"
    detail = read("frontend/src/lib/components/panels/ProjectDetail.svelte")
    assert "downloadProjectOutput(detail.id, outputId, filename)" in detail
    assert "on:click={() => downloadOutput(output.id, output.filename)}" in detail, "the output's download unwired"
    stores = read("frontend/src/lib/stores/projects.ts")
    assert stores.count("return answer.not_forgotten ?? [];") == 2, "a deletion's leftovers are dropped"
    for panel_name in ("FileManager", "ProjectList"):
        assert "not all of it went: ${left.join('; ')}" in read(
            f"frontend/src/lib/components/panels/{panel_name}.svelte")


# ---------------------------------------------------------------------------
# PF12 -- what a deletion names starts with the inventory's label
# ---------------------------------------------------------------------------
def test_pf12_what_a_deletion_names_starts_with_the_inventorys_label(tmp_path):
    loaded, restore = isolate(targets={"opti_oignon.store_inventory": source("store_inventory.py")})
    try:
        inventory = loaded["opti_oignon.store_inventory"]
        assert inventory.REQUESTS == ("notes", "projects")
        named = [e.label for e in inventory.STORES if e.request == inventory.PROJECTS]
        assert tuple(named) == _PROJECT_LABELS, named
    finally:
        restore()
    world = _world(tmp_path)
    try:
        assert world.pc._INDEX_LABEL == _PROJECT_LABELS[2] and world.pj._FILES_LABEL == _PROJECT_LABELS[1]
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF13 -- no deletion overwrites a file another link names
# ---------------------------------------------------------------------------
def _raised(call, *args):
    """The class an OSError ``call`` raised is, or None."""
    try:
        call(*args)
    except OSError:
        return OSError
    return None


def _copies():
    """Every module of the package that spells the overwrite, with the text of its two functions."""
    import ast

    found = {}
    paths = []
    for top, dirs, files in os.walk(source()):
        # The package's data folder is never walked into: in the maintainer's tree it holds real data.
        dirs[:] = sorted(d for d in dirs if d not in ("data", "__pycache__"))
        paths += [Path(top) / name for name in files if name.endswith(".py")]
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {node.name: ast.get_source_segment(path.read_text(encoding="utf-8"), node)
                     for node in tree.body if isinstance(node, ast.FunctionDef)}
        if "_overwrite_and_remove" in functions:
            found[str(path.relative_to(source()))] = (functions["_overwrite_and_remove"],
                                                     functions.get("_made_writable"))
    return found


def test_pf13_no_deletion_overwrites_a_file_another_link_names(tmp_path):
    copies = _copies()
    assert sorted(copies) == ["history.py", "learned_router.py", "notes/blob_store.py", "projects.py"], sorted(copies)
    assert len(set(copies.values())) == 1, "the copies of the overwrite differ"
    ((helper, writable),) = set(copies.values())
    logged = []
    space = {"os": os, "stat": __import__("stat"), "Path": Path,
             "logger": SimpleNamespace(warning=lambda *args: logged.append(args))}
    exec(compile(writable + "\n\n" + helper, "overwrite", "exec"), space)
    forget = space["_overwrite_and_remove"]
    # A file another link names: removed, never overwritten, said.
    kept, stored = tmp_path / "backup.txt", tmp_path / "stored.txt"
    kept.write_bytes(b"the backup's own copy")
    os.link(kept, stored)
    said = []
    assert forget(stored, said) is True and not stored.exists() and kept.read_bytes() == b"the backup's own copy"
    assert said == ["other links hold its bytes"]
    os.link(kept, stored)
    assert forget(stored) is True and logged and kept.read_bytes() == b"the backup's own copy", "not said"
    # A read-only file of this process: made writable, overwritten, removed.
    own = tmp_path / "own.txt"
    own.write_bytes(b"read-only words")
    os.chmod(own, 0o400)
    held, said = _witness(own), []
    assert forget(own, said) is True and not own.exists() and said == [] and _zeroed(held)
    # A file its owner may not even read: made readable and writable, overwritten, removed.
    sealed = tmp_path / "sealed.txt"
    sealed.write_bytes(b"sealed words")
    held = _witness(sealed)
    os.chmod(sealed, 0)
    said = []
    assert forget(sealed, said) is True and not sealed.exists() and said == [] and _zeroed(held)
    # A read-only file another link names: neither its bytes nor its mode change.
    backup, stored = tmp_path / "ro-backup.txt", tmp_path / "ro-stored.txt"
    backup.write_bytes(b"read-only backup")
    os.chmod(backup, 0o444)
    os.link(backup, stored)
    said = []
    assert forget(stored, said) is True and not stored.exists() and said == ["other links hold its bytes"]
    assert backup.read_bytes() == b"read-only backup" and (backup.stat().st_mode & 0o777) == 0o444
    # A FIFO with no reader refuses the open, never waits for one (a daemon thread, so a wait fails, never hangs).
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    outcome = []
    attempt = threading.Thread(target=lambda: outcome.append(_raised(forget, fifo)), daemon=True)
    attempt.start()
    attempt.join(timeout=2.0)
    assert not attempt.is_alive() and outcome == [OSError], f"a FIFO waited for its reader: {outcome}"
    # A link is refused, never followed; a file that is not there is said absent.
    target = tmp_path / "target.txt"
    target.write_bytes(b"target")
    (tmp_path / "link.txt").symlink_to(target)
    with pytest.raises(OSError):
        forget(tmp_path / "link.txt")
    assert target.read_bytes() == b"target" and forget(tmp_path / "absent.txt") is False
    # Through a project's deletion: the outside copy intact, the answer says it.
    world = _world(tmp_path / "w")
    try:
        planted = _plant(world)
        backup = tmp_path / "w" / "backup-of-file.txt"
        os.link(planted.file.file_path, backup)
        deleted, left = world.store.forget_project(planted.project.id, sweep=False)
        assert deleted and left == [f"project files and saved outputs (project {planted.project.id}'s: removed, "
                                    "not overwritten -- other links hold its bytes)"], left
        assert backup.read_bytes().startswith(CONTENT.encode()), "the copy another link names was overwritten"
        # A file another link names, removed alone: the answer says it, and the sweep still runs on disk.
        live = _plant(world, name="Zorblive", filename="Zorblivefile.txt", content="Zorblivechunk")
        orphan = world.data / "projects" / "abcdef12-777"
        (orphan / "files").mkdir(parents=True)
        (orphan / "files" / "abcdef12-778_Zorborphan.txt").write_bytes(b"Zorborphanbytes")
        (world.data / "projects" / live.project.id / "files" / "abcdef12-779").mkdir()
        os.link(live.file.file_path, tmp_path / "w" / "backup-of-live.txt")
        deleted, left = world.store.forget_file(live.file.id)
        assert deleted and left == [f"project files and saved outputs (a removed file of project {live.project.id}: "
                                    "removed, not overwritten -- other links hold its bytes)"], left
        assert not orphan.exists(), "a linked file's removal skipped the sweep on disk"
        assert not (world.data / "projects" / live.project.id / "files" / "abcdef12-779").exists(), (
            "a folder where a file was is left at every sweep")
    finally:
        world.restore()


# ---------------------------------------------------------------------------
# PF14 -- a read in progress is named, and never waited for
# ---------------------------------------------------------------------------
_BUSY = [
    "project indexes (chunks of the project files) (a read or a write in progress keeps the removed chunks in the "
    "store until it ends; the next deletion empties it)",
    "projects store (single-user) (a read or a write in progress keeps what the deletion erased in the database "
    "until it ends; the next deletion empties it)"]


def test_pf14_a_read_in_progress_is_named_and_never_waited_for(tmp_path, monkeypatch):
    import time

    monkeypatch.setitem(globals(), "_TIMEOUT", 2.0)
    world = _world(tmp_path)
    readers = []
    try:
        planted = _plant(world)
        for name in ("projects.db", "rag/vectors.db"):
            conn = sqlite3.connect(str(world.data / name), isolation_level=None)
            conn.execute("BEGIN")
            conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            readers.append(conn)
        started = time.monotonic()
        deleted, left = world.store.forget_project(planted.project.id, sweep=False)
        assert time.monotonic() - started < 1.0, "the deletion waited for a read in progress"
        assert deleted and left == _BUSY, left
        # A file's removal, and an output's, under the same reads: each says them.
        other = _plant(world, name="Zorbother", description="kept", instructions="kept", filename="Zorbotherfile.txt",
                       content="Zorbotherchunk")
        assert world.store.forget_file(other.file.id, sweep=False) == (True, _BUSY)
        assert world.store.forget_output(other.output.id) == (True, [_BUSY[1]])
        for conn in readers:
            conn.execute("ROLLBACK")
            conn.close()
        readers = []
        last = world.store.create_project("Zorbnext")
        assert world.store.forget_project(last.id) == (True, [])
        words = (*_WORDS, "Zorbnext", "Zorbotherfile", "Zorbotherchunk")
        assert _held(world.data, *words) == set(), _held(world.data, *words)
    finally:
        for conn in readers:
            conn.close()
        world.restore()
