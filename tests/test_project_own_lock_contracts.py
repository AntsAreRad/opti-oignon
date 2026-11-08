"""Contracts of each project's own lock (265, BACKLOG 253 b): a project's Mycelium pass held the lock of every
project's writes for its whole length (ten files of 1 MB: about 9 s), so adding a file to ANOTHER project, or
deleting one there, waited for it. Now the pass holds its own project's lock alone; that project's writes and
deletions still wait for it (a deletion is never undone by the pass), the other projects' do not.

  * PK1 -- ANOTHER PROJECT GOES ON: while project A's pass is held inside its indexing, a file is added to project B,
    then deleted there, both before A's pass ends.
  * PK2 -- ITS OWN PROJECT WAITS: while A's pass is held, deleting one of A's files, and deleting A, wait for the
    pass; once it ends they go, and A's store holds nothing of the deleted file afterwards (nothing undone).
  * PK3 -- THE ORDER OF THE LOCKS: a deletion takes its project's lock before the common one, never the other way
    round: a pass of A held, a deletion of A's file -- then of A -- waiting on A's lock, and a file added to B, all end
    once the pass is let go, B's write first -- none waits on another for ever; an id that is no project's makes no
    lock.
  * PK4 -- THE TEXTS: the CHANGELOG says it.
  * PK6 -- THE SWEEP LEAVES A PASS ALONE (review 265-1): while A's pass stands inside its store's transaction, a
    deletion in B runs the sweep: it ends at once, says nothing of A, and drops nothing A's pass writes (a file of A
    whose row was gone, beside a file newly added, made the sweep index A's store again from its stale reading).
  * PK5 -- PM1 of tests/test_project_mycelium_contracts.py, superseded (it held the pass under the projects' common
    write lock): a project's .py and .md files, a binary file, a secret's name and a file left on its folder with no
    row: indexing one of its files passes the ones its rows name and the indexer reads into ``project_<id>``, source
    ``files``, private, though the knowledge base's index refused; the pass ran under the project's own lock and not
    the common one, its vectors embedded outside both.
The suite runs the real projects' store and indexer with the project store suite's world (the data folder a test
folder, a plain opener, the vector pass's encoder a stand-in); the Mycelium's index is the real one, held at a
rendezvous for the length of a contract.
"""
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_mycelium_agent_contracts as my  # noqa: E402
import test_project_mycelium_contracts as pmc  # noqa: E402
from test_project_mycelium_contracts import BUDGET, NOTES  # noqa: E402

BUDGET_S = {
    "test_pk1_another_project_goes_on": 6.0,
    "test_pk2_its_own_project_waits": 6.0,
    "test_pk3_the_order_of_the_locks": 6.0,
    "test_pk4_the_texts": 2.0,
    "test_pk5_the_files_a_project_holds_under_its_own_lock": 4.0,
    "test_pk6_the_sweep_leaves_a_pass_alone": 6.0,
}

# The project store suite's fixtures.
no_engine = my.no_engine
owned = my.owned
world = pmc.world


class _Held:
    """The Mycelium's index of one project held at a rendezvous: ``inside`` set once the pass stands there, the pass
    going on once ``release`` is set; every other project's pass runs as it is."""

    def __init__(self, world, monkeypatch, project_id):
        from opti_oignon.mycelium import projects as stores

        self.inside, self.release, self.common_free = threading.Event(), threading.Event(), []
        real = stores.index

        def index(pid, paths, vectors=True):
            if pid == project_id:
                self.inside.set()
                # The common lock is free while the pass runs: another thread takes it at once.
                taken = []
                other = threading.Thread(target=lambda: taken.append(world.pj._WRITES.acquire(timeout=0.5)
                                                                     and (world.pj._WRITES.release() or True)))
                other.start()
                other.join(2)
                self.common_free.append(taken == [True])
                assert self.release.wait(10), "the contract never let the pass go"
            return real(pid, paths, vectors=vectors)

        monkeypatch.setattr(stores, "index", index)

    def run(self, world, project_id):
        self.done = threading.Event()

        def go():
            try:
                world.indexer.index_mycelium(project_id, vectors=False)
            finally:
                self.done.set()

        threading.Thread(target=go, name="pk-pass", daemon=True).start()
        assert self.inside.wait(5), "the pass never reached its indexing"


def _both(world):
    a, a_files = pmc._project(world, {"notes.md": NOTES.encode(), "budget.py": BUDGET.encode()}, name="a")
    b, _b_files = pmc._project(world, {"other.md": b"# Other\n\nA note of project B.\n"}, name="b")
    return a, a_files, b


def _after(event, seconds=5.0):
    return event.wait(seconds)


def test_pk1_another_project_goes_on(world, monkeypatch):
    a, _a_files, b = _both(world)
    held = _Held(world, monkeypatch, a.id)
    held.run(world, a.id)
    started = time.monotonic()
    added = world.store.add_file(b.id, "late.md", b"# Late\n\nAdded while A's pass ran.\n")
    removed, _left = world.store.forget_file(added.id)
    took = time.monotonic() - started
    assert removed and not held.done.is_set(), "B's writes waited for A's pass"
    assert took < 3.0, took
    held.release.set()
    assert _after(held.done), "A's pass never ended"
    assert held.common_free == [True], "the pass held the common lock"


def test_pk2_its_own_project_waits(world, monkeypatch):
    a, a_files, _b = _both(world)
    held = _Held(world, monkeypatch, a.id)
    held.run(world, a.id)
    gone = threading.Event()

    def delete_notes():
        world.store.forget_file(a_files["notes.md"].id)
        gone.set()

    threading.Thread(target=delete_notes, daemon=True).start()
    assert not gone.wait(0.5), "a deletion of A's file went through A's pass"
    held.release.set()
    assert _after(held.done) and _after(gone), "the pass or the deletion never ended"
    assert pmc._held(world, a.id) is None or "notes.md" not in pmc._held(world, a.id), (
        "the pass undid the deletion", pmc._held(world, a.id))
    # Deleting the whole project waits too, then goes.
    second = _Held(world, monkeypatch, a.id)
    second.run(world, a.id)
    project_gone = threading.Event()
    threading.Thread(target=lambda: (world.store.forget_project(a.id), project_gone.set()), daemon=True).start()
    assert not project_gone.wait(0.5), "deleting A went through A's pass"
    second.release.set()
    assert _after(second.done) and _after(project_gone)
    assert world.store.get_project(a.id) is None


def test_pk3_the_order_of_the_locks(world, monkeypatch):
    a, a_files, b = _both(world)
    held = _Held(world, monkeypatch, a.id)
    held.run(world, a.id)
    ended = {name: threading.Event() for name in ("delete_a", "add_b")}

    def delete_a():
        world.store.forget_file(a_files["budget.py"].id)
        ended["delete_a"].set()

    def add_b():
        world.store.add_file(b.id, "more.md", b"# More\n\nB again.\n")
        ended["add_b"].set()

    threading.Thread(target=delete_a, daemon=True).start()
    time.sleep(0.2)     # the deletion of A stands at A's lock
    threading.Thread(target=add_b, daemon=True).start()
    assert _after(ended["add_b"], 3.0), "a write of B waited behind a deletion of A waiting on A's pass"
    assert not ended["delete_a"].is_set()
    held.release.set()
    assert _after(held.done) and _after(ended["delete_a"]), "a writer waited for ever"
    assert world.pj._PROJECT_LOCKS.get(a.id) is not None and world.pj.project_lock(a.id) is world.pj.project_lock(a.id)
    # The project's deletion takes the locks in the same order: a write of B goes on while it waits on A's pass.
    second = _Held(world, monkeypatch, a.id)
    second.run(world, a.id)
    ended = {name: threading.Event() for name in ("forget_a", "add_b")}
    threading.Thread(target=lambda: (world.store.forget_project(a.id), ended["forget_a"].set()), daemon=True).start()
    time.sleep(0.2)     # the deletion of A stands at A's lock
    threading.Thread(target=lambda: (world.store.add_file(b.id, "last.md", b"# Last\n"), ended["add_b"].set()),
                     daemon=True).start()
    assert _after(ended["add_b"], 3.0), "a write of B waited behind A's deletion waiting on A's pass"
    second.release.set()
    assert _after(second.done) and _after(ended["forget_a"]), "a writer waited for ever"
    # An id that is no project's makes no lock.
    assert world.store.forget_project("0123abcd-fff") == (False, []) and "0123abcd-fff" not in world.pj._PROJECT_LOCKS


def test_pk4_the_texts():
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "PK1-PK6" in e), "").split())
    assert "own lock" in entry and "another project" in entry, entry[:300]


def test_pk5_the_files_a_project_holds_under_its_own_lock(world, monkeypatch):

    project, added = pmc._project(world, {"budget.py": BUDGET.encode(), "notes.md": NOTES.encode(),
                                      "scan.bin": bytes(range(256)) * 8, "credentials.json": b'{"token": "s3cr3t"}\n',
                                      ".env": b"API_KEY=s3cr3t\n", "id_rsa": b"-----BEGIN KEY-----\ns3cr3t\n"})
    folder = Path(added["budget.py"].file_path).parent
    (folder / "0123abcd-ab1_stray.py").write_text("stray_value = 1\n", encoding="utf-8")
    from opti_oignon.mycelium import projects as stores

    real, locked = stores.index, []
    real_vectors, vector_locked = stores.vectors, []
    from opti_oignon.mycelium import place

    real_open, opened = place.open_to_write, []

    def open_to_write(corpus, make=True):
        if corpus.startswith("project_"):
            opened.append((make, world.pj.project_lock(corpus[len("project_"):])._is_owned()))
        return real_open(corpus, make=make)

    monkeypatch.setattr(place, "open_to_write", open_to_write)

    def index(project_id, paths, vectors=True):
        locked.append((world.pj._WRITES._is_owned(), world.pj.project_lock(project_id)._is_owned()))
        return real(project_id, paths, vectors=vectors)

    def vectors(project_id, *args, **kwargs):
        vector_locked.append((world.pj._WRITES._is_owned(), world.pj.project_lock(project_id)._is_owned()))
        return real_vectors(project_id, *args, **kwargs)

    monkeypatch.setattr(stores, "index", index)
    monkeypatch.setattr(stores, "vectors", vectors)
    result = world.indexer.index_file(project.id, added["budget.py"].id)
    assert result.success is False and "Missing dependencies" in result.error, result
    assert result.mycelium == [], result.mycelium
    assert locked == [(False, True)] and vector_locked == [(False, False)], (locked, vector_locked)
    assert opened and all(owned for _make, owned in opened) and (False, True) in opened, (
        "a store of the project opened outside its lock", opened)
    assert pmc._held(world, project.id) == {Path(added["budget.py"].file_path).name,
                                        Path(added["notes.md"].file_path).name}
    assert b"s3cr3t" not in pmc._store_bytes(world, f"project_{project.id}")

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
    assert vector_locked == [(False, False), (False, False)], vector_locked
    store = world.place.open_to_read(f"project_{project.id}")
    try:
        assert [(s["name"], s["sensitivity"]) for s in store.sources()] == [("files", "private")]
    finally:
        store.close()


def test_pk6_the_sweep_leaves_a_pass_alone(world, monkeypatch):
    from opti_oignon.mycelium import store as store_mod

    a, a_files = pmc._project(world, {"notes.md": NOTES.encode(), "budget.py": BUDGET.encode()}, name="a")
    b, b_files = pmc._project(world, {"other.md": b"# Other\n\nA note of project B.\n"}, name="b")
    world.indexer.index_mycelium(a.id, vectors=False)
    # A file of A's store whose row is gone (a drop that failed), and a file added since, by its row alone.
    conn = sqlite3.connect(world.data / "projects.db")
    conn.execute("DELETE FROM project_files WHERE id = ?", (a_files["budget.py"].id,))
    conn.commit()
    conn.close()
    late = world.store.add_file(a.id, "late.md", b"# Late\n\nThe zanzibar_quartz plan, added since.\n")
    inside, release, done = threading.Event(), threading.Event(), threading.Event()
    real_episode = store_mod.Store.episode

    def episode(self, *args, **kwargs):
        if str(self.path).endswith(f"project_{a.id}.db") and not release.is_set():
            inside.set()
            release.wait(10)        # A's pass, inside its store's transaction
        return real_episode(self, *args, **kwargs)

    monkeypatch.setattr(store_mod.Store, "episode", episode)
    threading.Thread(target=lambda: (world.indexer.index_mycelium(a.id, vectors=False), done.set()),
                     daemon=True).start()
    assert inside.wait(5), "A's pass never reached its transaction"
    started = time.monotonic()
    removed, left = world.store.forget_file(b_files["other.md"].id)
    took = time.monotonic() - started
    release.set()
    assert done.wait(10), "A's pass never ended"
    assert removed and took < 2.0, ("B's deletion waited on A's pass", took)
    assert not any(a.id in line for line in left), left
    assert pmc._held(world, a.id) == {Path(a_files["notes.md"].file_path).name, Path(late.file_path).name}, (
        "the sweep undid what A's pass wrote", pmc._held(world, a.id))
