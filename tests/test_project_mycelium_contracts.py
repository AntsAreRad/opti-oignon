"""Contracts of a project's files in the Mycelium (M4a): each project's own store, ``project_<id>``, held to the files
its rows name -- indexed with the project's files, left by a removed file, removed with a forgotten project, swept
when its project is gone -- through the projects' index (``project_context.ProjectIndexer``), never raised into it.

  * PM1 -- THE FILES A PROJECT HOLDS: indexing a project's file passes the files its rows name into its store,
    private, whatever the knowledge base's index made of it; a binary file, a file left on the disk with no row, and
    a file named as a secret's (its name read without the id the projects put before it) are not in it; the rows are
    read and the pass is run under the projects' write lock, the vector pass never, whatever path the knowledge
    base's index took (review 253-1).
  * PM2 -- A REMOVED FILE LEAVES THE STORE: its parts gone and its words out of the store's bytes, the project's other
    file kept; the last file removed, the store is gone; a rewrite that failed is run by the next sweep.
  * PM3 -- A FORGOTTEN PROJECT LEAVES NO STORE: its store and the files beside it overwritten (a reader holding them
    reads no word) and removed; another project's store and an owner's corpus untouched, byte for byte.
  * PM4 -- THE SWEEP: the store of a project that is gone removed, from a live project's store the files its rows no
    longer name; an owner's corpus named like a project's but with no project's id never touched; the owner's route
    refuses a project's store by name, to index or to forget.
  * PM5 -- NEVER RAISED INTO THE PROJECTS: a store that cannot be opened, or removed, is named in what is answered,
    under the inventory's label of the project indexes and without a file's words; a tree with no Mycelium (an
    isolation window that leaves it out) answers nothing.
  * PM6 -- THE VECTOR PASS: after the index pass, the store's texts get their vectors; another vector pass holding the
    install's stores leaves them to a later pass, without waiting for it; a deletion meeting the vector pass's opening
    of the store leaves no store behind.
  * PM7 -- A PROJECT'S ID: the Mycelium reads a project's id by the projects' own rule, and every id the projects'
    store makes is one.
The suite runs the real package, its data folder a test folder and its stores' opener a plain one, the projects' store
a real one over that folder; no contract reaches the engine (the vector pass's encoder is a stand-in).
"""
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_mycelium_agent_contracts as my  # noqa: E402
from _isolation import isolate, source  # noqa: E402
from test_mycelium_place_contracts import _folder  # noqa: E402

# The Mycelium agent tool's fixtures: no engine reached, a test data folder and a quick opener.
no_engine = my.no_engine
owned = my.owned

BUDGET_S = {
    "test_pm1_the_files_a_project_holds": 6.0,
    "test_pm2_a_removed_file_leaves_the_store": 6.0,
    "test_pm3_a_forgotten_project_leaves_no_store": 6.0,
    "test_pm4_the_sweep": 6.0,
    "test_pm5_never_raised_into_the_projects": 6.0,
    "test_pm6_the_vector_pass": 6.0,
    "test_pm7_a_projects_id": 3.0,
}

LABEL = "project indexes (chunks of the project files)"
BUDGET = '"""The budget of a turn."""\n\n\ndef turn_budget(tokens):\n    return min(tokens, 4096)\n'
NOTES = "# Notes\n\nThe zanzibar_quartz plan keeps every draft.\n"


class _Embedder:
    """The vector pass's encoder, a stand-in: one fixed vector for each text, the engine never reached."""

    calls = []

    def __init__(self, encoder, timeout_s=None):
        self.encoder = encoder

    def texts(self, texts):
        _Embedder.calls.append(len(texts))
        return [[0.25, 0.5, 0.75] for _ in texts], set()


@pytest.fixture
def world(owned, tmp_path, monkeypatch):
    """The real projects' store and index over the test data folder, its opener a plain one; the vector pass's
    encoder a stand-in; the knowledge base's embeddings absent (its index refuses, the Mycelium's runs)."""
    from opti_oignon import project_context as pc
    from opti_oignon import projects as pj
    from opti_oignon.mycelium import vectors

    monkeypatch.setattr(pj, "_safe_connect", lambda path, **kw: sqlite3.connect(str(path), **kw))
    monkeypatch.setattr(vectors, "Embedder", _Embedder)
    monkeypatch.setattr(pc, "EMBEDDINGS_AVAILABLE", False)
    _Embedder.calls = []
    data = owned.data
    store = pj.ProjectStore(db_path=data / "projects.db", config_path=tmp_path / "absent.yaml",
                            storage_base=data / "projects", vectors_db=data / "rag" / "vectors.db")
    indexer = pc.ProjectIndexer(vectors_db=data / "rag" / "vectors.db", store=store)
    return types.SimpleNamespace(pj=pj, pc=pc, store=store, indexer=indexer, data=data, place=owned.place,
                                 vectors_db=data / "rag" / "vectors.db")


def _project(world, files, name="p"):
    """A project and its files, added through the projects' store: ``(project, {original name: ProjectFile})``."""
    project = world.store.create_project(name)
    added = {original: world.store.add_file(project.id, original, content) for original, content in files.items()}
    return project, added


def _held(world, project_id):
    """The names the project's store holds, or None when it has no store."""
    from opti_oignon.mycelium import place

    try:
        store = place.open_to_read(f"project_{project_id}")
    except place.NoStore:
        return None
    try:
        return {row[0] for row in store.db.execute("SELECT path FROM file")}
    finally:
        store.close()


def _owed(world, project_id):
    """Whether the project's store still owes the rewrite a dropped file marks in it."""
    store = world.place.open_to_read(f"project_{project_id}")
    try:
        return store.meta("rewrite")
    finally:
        store.close()


def _store_bytes(world, corpus):
    """Every byte of a store and of the files SQLite keeps beside it."""
    folder = world.place.folder()
    return b"".join(path.read_bytes() for path in sorted(folder.glob(f"{corpus}.db*")))


def test_pm1_the_files_a_project_holds(world, monkeypatch):
    """PM1 -- a project's .py and .md files, a binary file, and a file left on its folder with no row: indexing one of
    its files passes the ones its rows name and the indexer reads into ``project_<id>``, source ``files``, private,
    though the knowledge base's index refused; the binary and the stray file are not in it; the pass ran under the
    projects' write lock. A file never indexed, one with no row indexed, a public store, or a pass outside the lock
    -> RED."""
    project, added = _project(world, {"budget.py": BUDGET.encode(), "notes.md": NOTES.encode(),
                                      "scan.bin": bytes(range(256)) * 8, "credentials.json": b'{"token": "s3cr3t"}\n',
                                      ".env": b"API_KEY=s3cr3t\n", "id_rsa": b"-----BEGIN KEY-----\ns3cr3t\n"})
    folder = Path(added["budget.py"].file_path).parent
    (folder / "0123abcd-ab1_stray.py").write_text("stray_value = 1\n", encoding="utf-8")
    from opti_oignon.mycelium import projects as stores

    real, locked = stores.index, []
    real_vectors, vector_locked = stores.vectors, []

    def index(project_id, paths, vectors=True):
        locked.append(world.pj._WRITES._is_owned())
        return real(project_id, paths, vectors=vectors)

    def vectors(project_id, *args, **kwargs):
        vector_locked.append(world.pj._WRITES._is_owned())
        return real_vectors(project_id, *args, **kwargs)

    monkeypatch.setattr(stores, "index", index)
    monkeypatch.setattr(stores, "vectors", vectors)
    result = world.indexer.index_file(project.id, added["budget.py"].id)
    assert result.success is False and "Missing dependencies" in result.error, result
    assert result.mycelium == [], result.mycelium
    assert locked == [True] and vector_locked == [False], (locked, vector_locked)
    assert _held(world, project.id) == {Path(added["budget.py"].file_path).name,
                                        Path(added["notes.md"].file_path).name}
    assert b"s3cr3t" not in _store_bytes(world, f"project_{project.id}")

    # The knowledge base's index taking its path under the projects' lock (its collection unavailable there), its
    # chunker and embeddings stand-ins: the package's own settings are never read.
    class Embeddings:
        def embed(self, texts, show_progress=False):
            return [[0.5, 0.5] for _ in texts]

    class Chunker:
        def chunk(self, content, filepath, file_type):
            return [types.SimpleNamespace(content=line, chunk_index=i, metadata={"source_file": filepath})
                    for i, line in enumerate(content.splitlines()) if line.strip()]

    with monkeypatch.context() as patched:
        patched.setattr(world.pc, "EMBEDDINGS_AVAILABLE", True)
        patched.setattr(world.pc, "get_chunker", lambda rag_type: Chunker())
        patched.setattr(world.indexer, "_embedder", Embeddings())
        patched.setattr(world.indexer, "_embedder_checked", True)
        patched.setattr(world.indexer, "_get_collection", lambda project_id: None)
        result = world.indexer.index_file(project.id, added["notes.md"].id)
    assert result.error == "Project collection unavailable", result.error
    assert vector_locked == [False, False], vector_locked
    store = world.place.open_to_read(f"project_{project.id}")
    try:
        assert [(s["name"], s["sensitivity"]) for s in store.sources()] == [("files", "private")]
    finally:
        store.close()


def test_pm2_a_removed_file_leaves_the_store(world, monkeypatch):
    """PM2 -- a project of two files indexed; one removed through the index's erasure (``forget_file_chunks``, the
    call a file's removal makes): it leaves the store, its words leave the store's bytes, the rewrite it owed has run,
    the other stays; the last one removed, the store is gone. A removed file still held, its words left in the pages,
    a rewrite left owed, or an empty store kept -> RED."""
    project, added = _project(world, {"budget.py": BUDGET.encode(), "notes.md": NOTES.encode()})
    assert world.indexer.index_mycelium(project.id) == []
    # The full-text index keeps its terms prefix-compressed: "zanzibar" is in the bytes whole, "zanzibar_quartz" after it.
    assert b"zanzibar" in _store_bytes(world, f"project_{project.id}")
    left = world.pc.ProjectIndexer.forget_file_chunks(project.id, {added["notes.md"].file_path}, world.vectors_db)
    assert left == [], left
    assert _held(world, project.id) == {Path(added["budget.py"].file_path).name}
    assert b"zanzibar" not in _store_bytes(world, f"project_{project.id}")
    assert _owed(world, project.id) is None, "the rewrite the drop owed did not run"
    left = world.pc.ProjectIndexer.forget_file_chunks(project.id, {added["budget.py"].file_path}, world.vectors_db)
    assert left == [] and _held(world, project.id) is None, left
    assert list(world.place.folder().glob(f"project_{project.id}.db*")) == []

    # A rewrite that fails is named, and the next deletion's sweep runs it.
    again, kept = _project(world, {"budget.py": BUDGET.encode(), "notes.md": NOTES.encode()}, name="again")
    assert world.indexer.index_mycelium(again.id) == []
    from opti_oignon.mycelium.store import Store

    with monkeypatch.context() as patched:
        patched.setattr(Store, "rewrite", lambda self: False)
        left = world.pc.ProjectIndexer.forget_file_chunks(again.id, {kept["notes.md"].file_path}, world.vectors_db)
    assert len(left) == 1 and "not rewritten" in left[0] and _owed(world, again.id), left
    live = {again.id: {kept["budget.py"].file_path}}
    assert world.pc.ProjectIndexer.sweep_index(live, world.vectors_db) == []
    assert _owed(world, again.id) is None and _held(world, again.id) == {Path(kept["budget.py"].file_path).name}


def test_pm3_a_forgotten_project_leaves_no_store(world, owned, tmp_path):
    """PM3 -- two projects indexed and an owner's corpus: the first forgotten (``forget_project_index``, the call a
    project's deletion makes) leaves no ``project_<id>.db`` nor any file beside it; the second project's store and the
    owner's corpus are byte for byte as they were. A store left, or another one touched -> RED."""
    first, _ = _project(world, {"notes.md": NOTES.encode()}, name="first")
    second, _ = _project(world, {"budget.py": BUDGET.encode()}, name="second")
    assert world.indexer.index_mycelium(first.id) == [] and world.indexer.index_mycelium(second.id) == []
    my._index(owned, "notes", _folder(tmp_path / "owner", {"a.md": "owner words\n"}))
    before = {corpus: _store_bytes(world, corpus) for corpus in (f"project_{second.id}", "notes")}
    store = world.place.folder() / f"project_{first.id}.db"
    journal = store.parent / (store.name + "-journal")
    journal.write_bytes(b"zanzibar left in a journal\n" * 64)
    readers = [open(store, "rb"), open(journal, "rb")]      # a reader holding the files the deletion removes
    try:
        assert world.pc.ProjectIndexer.forget_project_index(first.id, world.vectors_db) == []
        assert list(world.place.folder().glob(f"project_{first.id}.db*")) == []
        for reader in readers:
            reader.seek(0)
            assert b"zanzibar" not in reader.read(), reader.name
    finally:
        for reader in readers:
            reader.close()
    assert {corpus: _store_bytes(world, corpus) for corpus in before} == before


def test_pm4_the_sweep(world, owned, tmp_path, capsys):
    """PM4 -- a live project whose store holds a file its rows no longer name, a store whose project is gone, and an
    owner's corpus named ``project_notes``: the sweep (``sweep_index``, the call every deletion's sweep makes) removes
    the gone project's store and the unnamed file, and leaves the owner's corpus as it was; the owner's route refuses
    to index or forget a project's store. An orphan store kept, a live store emptied, an owner's corpus touched, or a
    project's store written by the route -> RED."""
    live, added = _project(world, {"budget.py": BUDGET.encode(), "notes.md": NOTES.encode()}, name="live")
    gone, _ = _project(world, {"notes.md": NOTES.encode()}, name="gone")
    assert world.indexer.index_mycelium(live.id) == [] and world.indexer.index_mycelium(gone.id) == []
    owner = _folder(tmp_path / "owner", {"a.md": "owner words\n"})
    my._index(owned, "project_notes", owner)
    before = _store_bytes(world, "project_notes")
    left = world.pc.ProjectIndexer.sweep_index({live.id: {added["budget.py"].file_path}}, world.vectors_db)
    assert left == [], left
    assert _held(world, gone.id) is None
    assert _held(world, live.id) == {Path(added["budget.py"].file_path).name}
    assert _store_bytes(world, "project_notes") == before

    from opti_oignon.mycelium import route

    capsys.readouterr()
    assert route.index(f"project_{live.id}", str(owner)) == 1
    assert route.forget(f"project_{live.id}", "files") == 1
    said = capsys.readouterr().out
    assert said.count("is a project's store") == 2, said
    assert _held(world, live.id) == {Path(added["budget.py"].file_path).name}


def test_pm5_never_raised_into_the_projects(world, monkeypatch):
    """PM5 -- a store that cannot be opened: indexing answers a line under the inventory's label, no file's words, and
    raises nothing; a store that cannot be removed: the project's erasure names it; the Mycelium's own module failing:
    the file's erasure names it, raising nothing; a tree with no Mycelium (the
    projects' index alone in an isolation window): its erasure answers nothing. A failure raised into the projects, a
    line without the label or with a file's words, or a window's absence named as a failure -> RED."""
    project, _ = _project(world, {"notes.md": NOTES.encode()})
    from opti_oignon.mycelium import place

    def refused(*args, **kwargs):
        raise RuntimeError("the store's key is sealed")

    with monkeypatch.context() as patched:
        patched.setattr(place, "open_to_write", refused)
        left = world.indexer.index_mycelium(project.id)
    assert left and all(line.startswith(LABEL) for line in left) and "RuntimeError" in left[0], left
    assert all("zanzibar" not in line and "Notes" not in line for line in left)
    assert world.indexer.index_mycelium(project.id) == []

    def stuck(corpus):
        raise PermissionError("read-only file system")

    with monkeypatch.context() as patched:
        patched.setattr(place, "remove_store", stuck)
        left = world.pc.ProjectIndexer.forget_project_index(project.id, world.vectors_db)
    assert len(left) == 1 and left[0].startswith(LABEL) and "not removed" in left[0], left
    assert all("zanzibar" not in line and "Notes" not in line for line in left)

    from opti_oignon.mycelium import projects as stores

    def broken(*args, **kwargs):
        raise RuntimeError("the Mycelium failed")

    with monkeypatch.context() as patched:
        patched.setattr(stores, "drop", broken)
        left = world.pc.ProjectIndexer.forget_file_chunks(project.id, {"/x/files/a.md"}, world.vectors_db)
    assert len(left) == 1 and left[0].startswith(LABEL) and "not reached: RuntimeError" in left[0], left

    loaded, restore = isolate(targets={"opti_oignon.project_context": source("project_context.py")})
    try:
        index = loaded["opti_oignon.project_context"].ProjectIndexer
        assert index.forget_project_index(project.id) == []
        assert index.forget_file_chunks(project.id, {"/x/files/a.md"}) == []
        assert index.sweep_index({}) == []
    finally:
        restore()


def test_pm6_the_vector_pass(world):
    """PM6 -- after the index pass, the vector pass gives the store's texts their vectors; while another vector pass
    holds the install's stores, a file's index pass ends without waiting and leaves its vectors to the next pass,
    which gives them. Texts left without a vector, or a pass that waits for another -> RED."""
    project, added = _project(world, {"budget.py": BUDGET.encode()})

    def vectors(project_id):
        store = world.place.open_to_read(f"project_{project_id}")
        try:
            return store.db.execute("SELECT COUNT(*) FROM vector").fetchone()[0]
        finally:
            store.close()

    assert world.indexer.index_mycelium(project.id) == []
    assert vectors(project.id) > 0 and _Embedder.calls, _Embedder.calls
    world.store.add_file(project.id, "notes.md", NOTES.encode())
    count, calls = vectors(project.id), len(_Embedder.calls)
    started = time.perf_counter()
    with world.place.one_vector_pass():      # another pass holds the install's stores
        assert world.indexer.index_mycelium(project.id) == []
    assert time.perf_counter() - started < 3.0
    assert len(_Embedder.calls) == calls and vectors(project.id) == count
    assert world.indexer.index_mycelium(project.id) == []
    assert len(_Embedder.calls) > calls and vectors(project.id) > count

    # A deletion meets the vector pass where it opens the store: the store, deleted, is never made again.
    import threading

    from opti_oignon.mycelium import projects as stores
    from opti_oignon.mycelium.store import Store

    real_at, met, deleted = Store.at.__func__, threading.Event(), threading.Event()

    def delete():
        met.wait(5)
        with world.pj.holding():        # as a project's deletion holds the projects' write lock
            world.pc.ProjectIndexer.forget_project_index(project.id, world.vectors_db)
        deleted.set()

    def at(cls, path, read_only=False):
        if not read_only and Path(path).name == f"project_{project.id}.db" and not met.is_set():
            met.set()
            deleted.wait(0.5)           # the deletion lands here unless the opening holds it off
        return real_at(cls, path, read_only=read_only)

    deleter = threading.Thread(target=delete)
    deleter.start()
    try:
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(Store, "at", classmethod(at))
            stores.vectors(project.id, world.pj.holding)
    finally:
        deleter.join(10)
    assert deleted.is_set() and list(world.place.folder().glob(f"project_{project.id}.db*")) == []


def test_pm7_a_projects_id(world):
    """PM7 -- the ids the projects' store makes are projects' ids to the Mycelium, and its corpus names map back to
    them; a name that is not an id (too short, upper case, a path) is no project's. A rule apart from the projects',
    or an owner's name read as a project's -> RED."""
    from opti_oignon.mycelium import projects as stores

    made = [world.store.create_project(f"p{n}").id for n in range(20)]
    for project_id in made:
        assert world.pj._ID.fullmatch(project_id) and stores.project_of(stores.corpus_of(project_id)) == project_id
    for name in ("project_notes", "project_0123abcd-ab", "project_0123ABCD-ab1", "project_0123abcd-ab1/x", "notes",
                 "project_0123abcd-ab1-2"):
        assert stores.project_of(name) is None, name
    with pytest.raises(Exception):
        stores.corpus_of("../notes")
