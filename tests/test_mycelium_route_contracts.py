"""Contracts of the owner's route to the Mycelium, ``python -m opti_oignon mycelium index|forget|status``, and of
the store's two writers.

The route indexes a folder into a corpus's store, forgets a source, and says what the stores hold without writing a
byte. A pass, a documents' pass and a forgetting take the store's write lock at their start; one vector pass runs at
a time on the install's stores; a pass of the route that removed a file rewrites the store; the documents' reader
asks the store; a store is written in the rollback journal.

  * MO1 -- THE ROUTE INDEXES A FOLDER INTO ITS CORPUS: through ``main``, a folder indexed into ``<corpus>.db``, the
    source named as the corpus unless named, its sensitivity as asked; a corpus or a source named outside the rule, a
    folder that is missing, not a folder, the package's data folder or inside it (through a link too), a source
    rooted elsewhere: refused by name, exit 1, nothing made; a file the pass could not read: said, exit 1.
  * MO2 -- THE ROUTE FORGETS A SOURCE: its counts said and the store rewritten; an unknown source or corpus refused
    by name, nothing made; a reader holding the store at the forgetting's commit: refused by name, nothing
    forgotten (review 1); a rewrite that fails is said, exit 1, and owed.
  * MO3 -- THE STATUS WRITES NOTHING: each store's sources, files, parts, last pass, what a pass would do now and its
    vectors, said; nothing written, nothing made, no store opened to write; a store that cannot be read named, the
    others said; a missing corpus refused by name.
  * MO4 -- A PASS TAKES THE WRITE LOCK AT ITS START: a pass, a documents' pass and a forgetting on a store another
    connection writes are refused by name, nothing written, no transaction left; the generation and the files are
    read under the lock (another pass at that very moment is refused); opening a store to write takes no lock.
  * MO5 -- ONE VECTOR PASS AT A TIME: the stores' folder itself held, no file made; another holder, in this process
    or another, refuses the second by name; a folder that cannot be locked refuses it; through the route, the
    index pass stands and the vectors are left, said.
  * MO6 -- A PASS OF THE ROUTE THAT DROPS PARTS REWRITES THE STORE: a removed file's names and words, and a changed
    file's old ones, leave the store's bytes; the rewrite is owed by the pass's own transaction, so a rewrite that
    fails, one another connection holds off, or a process that ends before it runs to its end leaves it owed (review
    1): said, exit 1, said by the status, and done by the next pass of the route.
  * MO7 -- THE DOCUMENTS' READER ASKS THE STORE (MP10 of tests/test_mycelium_place_contracts.py, superseded): a
    document's text is served only while the store holds it at the same sha; forgotten or changed by another
    connection, it is not; a partial pass keeps nothing the store no longer holds.
  * MO8 -- A STORE IS WRITTEN IN THE ROLLBACK JOURNAL: a store left in WAL is back in DELETE once opened to write, no
    -wal or -shm left; one another connection holds in WAL is refused by name.
  * MO9 -- THE ROUTE'S VECTOR PASS: under the real settings, admitted by the governor, its counts said and the
    encoder read once it covers the store; held, said, exit 0; the encoder missing, said, exit 1, the index pass
    standing; settings refused: nothing indexed, no store recut under another header (review 1), exit 1; a batch that
    finds the store written by another pass stops the pass as held, what it wrote kept.
  * MO10 -- MXBAI'S CONTEXT IS ITS MODEL'S: 512 tokens, as the engine shows it (BACKLOG 235 a).
  * MO11 -- THE REAL PACKAGE SHAPE: ``python -m opti_oignon mycelium`` through the package's ``__main__``, the
    platform's opener asked to write by the index pass and to read by the status.
"""
import errno
import fcntl
import importlib
import runpy
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_mycelium_place_contracts import GATE, _folder, _many  # noqa: E402
from test_mycelium_vector_contracts import Engine, Governor, Registry  # noqa: E402

BUDGET_S = {
    "test_mo1_the_route_indexes_a_folder_into_its_corpus": 4.0,
    "test_mo2_the_route_forgets_a_source": 3.0,
    "test_mo3_the_status_writes_nothing": 3.0,
    "test_mo4_a_pass_takes_the_write_lock_at_its_start": 4.0,
    "test_mo5_one_vector_pass_at_a_time": 5.0,
    "test_mo6_a_pass_of_the_route_that_drops_parts_rewrites_the_store": 6.0,
    "test_mo7_the_documents_reader_asks_the_store": 3.0,
    "test_mo8_a_store_is_written_in_the_rollback_journal": 3.0,
    "test_mo9_the_routes_vector_pass": 5.0,
    "test_mo10_mxbais_context_is_its_models": 2.0,
    "test_mo11_the_real_package_shape": 4.0,
}

# How long a test's connection waits for a lock another holds: short, the waits are the contracts' own.
WAIT = 0.2
NOTES = "# Notes\n\nThe zanzibar inventory of the gate.\n"


def _quick(path, read_only=False):
    """The platform's opener as a plain SQLite one that waits ``WAIT`` for a lock: read-only through a URI."""
    if read_only:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False, timeout=WAIT)
    return sqlite3.connect(path, check_same_thread=False, timeout=WAIT)


@pytest.fixture(autouse=True)
def no_engine(monkeypatch):
    """No contract reaches the machine's engine or governor until it installs its stand-ins."""
    from opti_oignon.mycelium import vectors

    def refused():
        raise RuntimeError("a contract asked the real inference registry")

    monkeypatch.setattr(vectors, "_registry", refused)
    monkeypatch.setattr(vectors, "_governor", lambda: None)


@pytest.fixture
def owned(tmp_path, monkeypatch, capsys):
    """The real configuration, place, store and main, the data folder a test folder, the store's opener a quick plain
    one; ``run`` runs ``opti-oignon ARGS`` through ``main`` and gives (exit code, what it printed)."""
    config = importlib.import_module("opti_oignon.config")
    main_mod = importlib.import_module("opti_oignon.main")
    from opti_oignon.mycelium import place as place_mod
    from opti_oignon.mycelium import store as store_mod

    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(store_mod, "_safe_connect", _quick)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["opti-oignon", *argv])
        try:
            main_mod.main()
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        captured = capsys.readouterr()
        return code, captured.out + captured.err

    return types.SimpleNamespace(mod=place_mod, store=store_mod, data=data, folder=data / "mycelium", run=run)


def _snapshot(folder):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(folder.iterdir())}


def test_mo1_the_route_indexes_a_folder_into_its_corpus(owned, tmp_path):
    """MO1 -- ``opti-oignon mycelium index oo FOLDER`` makes ``data/mycelium/oo.db`` holding the folder's files under
    a private source named oo; ``--source`` and ``--sensitivity`` name another source of the corpus and its
    sensitivity; a second pass adds nothing. Refused by name, exit 1, no store or folder made: a corpus or a source
    outside the names' rule, a folder missing or not a folder, the package's data folder or a folder inside it, or a
    link to one, a source already rooted elsewhere, a folder this process cannot list (review 2). The action missing:
    its help, exit 1. A file or a folder the pass cannot read: said, exit 1 (not done in full). A route that indexes
    what it should refuse, or refuses silently -> RED."""
    src = _folder(tmp_path / "src", {"gate.py": GATE, "notes.md": NOTES})
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    assert code == 0, out
    assert "2 added" in out and "oo" in out, out
    assert owned.mod.corpora() == ["oo"] and (owned.folder / "oo.db").is_file()
    store = owned.mod.open_to_read("oo")
    try:
        [source] = store.sources()
        assert (source["name"], source["root"], source["kind"], source["sensitivity"]) == (
            "oo", str(src), "folder", "private"), source
        assert store.stats()["files"] == 2
    finally:
        store.close()
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    assert code == 0 and "0 added" in out, out
    other = _folder(tmp_path / "other", {"more.md": NOTES.replace("zanzibar", "quokka")})
    code, out = owned.run("mycelium", "index", "oo", str(other), "--source", "notes", "--sensitivity", "secret",
                          "--no-vectors")
    assert code == 0, out
    store = owned.mod.open_to_read("oo")
    try:
        assert {s["name"]: s["sensitivity"] for s in store.sources()} == {"oo": "private", "notes": "secret"}
    finally:
        store.close()

    inside = owned.data / "history"
    inside.mkdir()
    (inside / "c.json").write_text('{"said": "private words"}', encoding="utf-8")
    link = tmp_path / "looks_elsewhere"
    link.symlink_to(inside, target_is_directory=True)
    runnable = src / "run.sh"
    runnable.write_text("#!/bin/sh\n", encoding="utf-8")
    runnable.chmod(0o700)
    before = _snapshot(owned.folder)
    refusals = [
        (("Bad/Name", str(src)), "Bad/Name"),
        (("UPPER", str(src)), "UPPER"),
        (("x", str(tmp_path / "missing")), "missing"),
        (("x", str(src / "gate.py")), "gate.py"),
        (("x", str(runnable)), "run.sh"),
        (("x", str(owned.data)), str(owned.data)),
        (("x", str(inside)), str(inside)),
        (("x", str(link)), str(link)),
        (("x", str(src), "--source", "../up"), "../up"),
        (("oo", str(other)), "rooted"),
    ]
    for argv, said in refusals:
        code, out = owned.run("mycelium", "index", *argv, "--no-vectors")
        assert code == 1 and said in out and "[ERR]" in out, (argv, code, out)
    assert _snapshot(owned.folder) == before, "a refused pass wrote or made a store"
    code, out = owned.run("mycelium")
    assert code == 1 and "index" in out and "forget" in out and "status" in out, out
    code, _out = owned.run("mycelium", "index", "oo")
    assert code == 2
    (src / "closed.md").write_text(NOTES, encoding="utf-8")
    (src / "closed.md").chmod(0)
    try:
        code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    finally:
        (src / "closed.md").chmod(0o600)
    assert code == 1 and "could not be read" in out, (code, out)
    shut = src / "shut"
    shut.mkdir()
    (shut / "fresh.md").write_text(NOTES, encoding="utf-8")
    shut.chmod(0)
    try:
        code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    finally:
        shut.chmod(0o700)
    assert code == 1 and "1 folder could not be listed" in out, (code, out)
    closed_root = _folder(tmp_path / "closed_root", {"a.md": NOTES})
    closed_root.chmod(0)
    try:
        code, out = owned.run("mycelium", "index", "newc", str(closed_root), "--no-vectors")
    finally:
        closed_root.chmod(0o700)
    assert code == 1 and "cannot be listed" in out and "newc" not in owned.mod.corpora(), (code, out)


def test_mo2_the_route_forgets_a_source(owned, tmp_path, monkeypatch):
    """MO2 -- ``mycelium forget oo a`` forgets the source a of oo's store: its counts said, the store merged and
    rewritten, said; b stays. An unknown source or corpus, or a name outside the rule: refused by name, exit 1, no
    store made. A reader holding the store when the forgetting commits (a search, a status in another terminal):
    refused by name, exit 1, nothing forgotten. A rewrite that fails: the source forgotten, the words said to stay,
    exit 1, the rewrite owed in the store. A forgetting that makes a store, ends in a traceback, or keeps quiet about
    words that stay -> RED."""
    a = _folder(tmp_path / "a", {"gate.py": GATE.replace("entity", "alphaentity")})
    b = _folder(tmp_path / "b", {"gate.py": GATE.replace("entity", "betaentity")})
    for name, root in (("a", a), ("b", b)):
        code, out = owned.run("mycelium", "index", "oo", str(root), "--source", name, "--no-vectors")
        assert code == 0, out
    code, out = owned.run("mycelium", "forget", "oo", "a")
    assert code == 0 and "forgot" in out and "1 file" in out and "rewritten" in out, out
    store = owned.mod.open_to_read("oo")
    try:
        assert [s["name"] for s in store.sources()] == ["b"]
        assert store.meta("rewrite") is None
    finally:
        store.close()
    for argv, said in ((("oo", "zz"), "zz"), (("nope", "a"), "nope"), (("Bad", "a"), "Bad")):
        code, out = owned.run("mycelium", "forget", *argv)
        assert code == 1 and said in out and "[ERR]" in out, (argv, out)
    assert owned.mod.corpora() == ["oo"], "a forgetting made a store"
    reader = sqlite3.connect(owned.folder / "oo.db", timeout=WAIT)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM part").fetchall()
    try:
        code, out = owned.run("mycelium", "forget", "oo", "b")
    finally:
        reader.rollback()
        reader.close()
    assert code == 1 and "[ERR]" in out and "b" in out, (code, out)
    store = owned.mod.open_to_read("oo")
    try:
        assert [s["name"] for s in store.sources()] == ["b"], "a forgetting held off forgot"
    finally:
        store.close()
    from opti_oignon.mycelium.store import Store

    real = Store.purge

    def full(self, pages=None):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(Store, "purge", full)
    code, out = owned.run("mycelium", "forget", "oo", "b")
    assert code == 1 and "forgot" in out and "stay" in out, out
    monkeypatch.setattr(Store, "purge", real)
    store = owned.mod.open_to_read("oo")
    try:
        assert store.sources() == [] and (store.meta("rewrite") or "").startswith("owed")
    finally:
        store.close()


def test_mo3_the_status_writes_nothing(owned, tmp_path, monkeypatch):
    """MO3 -- with no store, the status says no corpus is indexed and makes no folder. Over two corpora, one holding
    a folder's source and a documents' source: each store, its sources (kind, root, sensitivity), files, parts and
    last pass, what a pass would do now (a new file said), and its vectors, said -- the folder's bytes and times as
    they were, no file made, the stores never opened to write. One corpus named: only it. A missing corpus: refused,
    exit 1. A store at a newer schema: named as refused, exit 1, the other still said. A status that writes, makes a
    file or hides a store -> RED."""
    code, out = owned.run("mycelium", "status")
    assert code == 0 and "no corpus" in out, out
    assert not owned.data.exists(), "the status made the data folder"
    src = _folder(tmp_path / "src", {"gate.py": GATE})
    for corpus in ("oo", "zz"):
        code, out = owned.run("mycelium", "index", corpus, str(src), "--no-vectors")
        assert code == 0, out
    from opti_oignon.mycelium.index import Indexer

    store = owned.mod.open_to_write("oo")
    Indexer(store).index_documents("cellar", {"n.md": NOTES})
    store.close()
    (src / "new.md").write_text(NOTES, encoding="utf-8")
    before = _snapshot(owned.folder)

    def no_writer(*_a, **_kw):
        raise AssertionError("the status opened a store to write")

    monkeypatch.setattr(owned.mod, "open_to_write", no_writer)
    code, out = owned.run("mycelium", "status")
    assert code == 0, out
    for said in ("oo", "zz", "cellar", "documents", "folder", "private", str(src), "1 file", "1 new"):
        assert said in out, (said, out)
    assert "vector" in out, out
    assert _snapshot(owned.folder) == before, "the status wrote or made a file"
    code, out = owned.run("mycelium", "status", "zz")
    assert code == 0 and "zz" in out and "cellar" not in out, out
    code, out = owned.run("mycelium", "status", "nope")
    assert code == 1 and "nope" in out, out
    db = sqlite3.connect(owned.folder / "zz.db")
    db.execute("UPDATE meta SET value = '9' WHERE key = 'schema'")
    db.commit()
    db.close()
    code, out = owned.run("mycelium", "status")
    assert code == 1 and "schema 9" in out and "cellar" in out, out


class Hooked:
    """A connection that runs ``hook`` once, at the first statement that reads a source's generation."""

    def __init__(self, db, hook):
        self.db, self.hook, self.done = db, hook, False

    def execute(self, sql, *args):
        if not self.done and sql.startswith("SELECT generation FROM source"):
            self.done = True
            self.hook()
        return self.db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.db, name)


def test_mo4_a_pass_takes_the_write_lock_at_its_start(owned, tmp_path):
    """MO4 -- another connection holds the store's write lock: the store still opens to write (no lock taken at the
    opening); a pass, a documents' pass and a forgetting are refused by name (another pass writes the store), the
    store as it was, no transaction left. And the generation is read under the lock: another pass that comes at that
    very moment is refused, never interleaved. A pass that reads before it locks -> RED."""
    from opti_oignon.mycelium.index import Indexer
    from opti_oignon.mycelium.store import Store, StoreError

    src = _folder(tmp_path / "src", {"gate.py": GATE})
    first = owned.mod.open_to_write("oo")
    Indexer(first).index("oo", src)
    first.close()
    path = owned.folder / "oo.db"
    other = sqlite3.connect(path, timeout=WAIT)
    other.execute("BEGIN IMMEDIATE")
    try:
        store = owned.mod.open_to_write("oo")
        held = store.generation()
        for act in (lambda: Indexer(store).index("oo", src),
                    lambda: Indexer(store).index_documents("cellar", {"n.md": NOTES}),
                    lambda: store.forget("oo")):
            with pytest.raises(StoreError, match="another pass writes the store"):
                act()
            assert not store.db.in_transaction
        assert store.generation() == held and [s["name"] for s in store.sources()] == ["oo"]
    finally:
        other.rollback()
        other.close()
    (src / "new.md").write_text(NOTES, encoding="utf-8")
    met = []

    def another_pass():
        second = Store(path, _quick)
        try:
            Indexer(second).index("oo", src)
            met.append("interleaved")
        except StoreError as exc:
            met.append(str(exc))
        finally:
            second.close()

    store = Store(path, lambda p: Hooked(_quick(p), another_pass))
    counts = Indexer(store).index("oo", src)
    assert len(met) == 1 and "another pass writes the store" in met[0], met
    assert counts["added"] == 1
    store.close()


def _child_holding(folder):
    """A process of its own holding the stores' folder as a vector pass does; returned once it holds it."""
    code = ("import fcntl, os, sys, time\n"
            "fd = os.open(sys.argv[1], os.O_RDONLY)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "print('held', flush=True)\n"
            "time.sleep(30)\n")
    child = subprocess.Popen([sys.executable, "-I", "-c", code, str(folder)], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "held"
    return child


def test_mo5_one_vector_pass_at_a_time(owned, tmp_path, monkeypatch):
    """MO5 -- ``place.one_vector_pass()`` holds the stores' folder itself, no file made: a second holder in this
    process is refused by name (another vector pass is under way), and so is one while another process holds it;
    released, it is taken again. A folder that cannot be locked refuses the pass by name. Through the route, while
    another holds it: the index pass stands, no vector asked, the vectors said to be left, exit 0. Two vector passes
    at once -> RED."""
    from opti_oignon.mycelium.store import StoreError

    src = _folder(tmp_path / "src", {"gate.py": GATE})
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    assert code == 0, out
    names = sorted(p.name for p in owned.folder.iterdir())
    with owned.mod.one_vector_pass():
        with pytest.raises(StoreError, match="another vector pass is under way"):
            with owned.mod.one_vector_pass():
                pass
        assert sorted(p.name for p in owned.folder.iterdir()) == names, "the lock made a file"
        engine = Engine(dim=768)
        from opti_oignon.mycelium import vectors

        monkeypatch.setattr(vectors, "_registry", lambda: Registry({"embeddinggemma-2": engine}))
        monkeypatch.setattr(vectors, "_governor", lambda: Governor())
        (src / "new.md").write_text(NOTES, encoding="utf-8")
        code, out = owned.run("mycelium", "index", "oo", str(src))
        assert code == 0 and "1 added" in out and "another vector pass is under way" in out, out
        assert engine.calls == []
    with owned.mod.one_vector_pass():
        pass
    child = _child_holding(owned.folder)
    try:
        with pytest.raises(StoreError, match="another vector pass is under way"):
            with owned.mod.one_vector_pass():
                pass
    finally:
        child.kill()
        child.wait()

    def refusing(fd, op):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(fcntl, "flock", refusing)
    with pytest.raises(StoreError, match="cannot be held"):
        with owned.mod.one_vector_pass():
            pass


def _markers(raw, tag, n):
    kinds = ("sentence", "path", "func", "word", "callee")
    return [f"{tag}{k}{i:03d}" for k in kinds for i in range(n) if f"{tag}{k}{i:03d}".encode() in raw]


def _two_rounds(owned, src, n, corpus="oo"):
    """kq and zx modules indexed through the route, then two rounds of changes, each indexed: the shape that leaves
    copies of moved rows in the pages' free space once the index is merged whole (probe
    m245/sondes/test_stale_shape_probe.py: two of the removed names)."""
    _many(src, "kq", n)
    _many(src, "zx", n)
    code, out = owned.run("mycelium", "index", corpus, str(src), "--no-vectors")
    assert code == 0, out
    for rounds in (0, 1):
        for i in range(rounds, n, 3):
            for tag in ("kq", "zx"):
                path = src / f"{tag}path{i:03d}.py"
                path.write_text(path.read_text(encoding="utf-8") + f"\n# more {rounds}\n", encoding="utf-8")
        code, out = owned.run("mycelium", "index", corpus, str(src), "--no-vectors")
        assert code == 0, out


class AfterVacuum:
    """The store's connection as it is, but ``hook`` runs once right after VACUUM, before anything else."""

    def __init__(self, db, hook):
        self.db, self.hook, self.done = db, hook, False

    def execute(self, sql, *args):
        out = self.db.execute(sql, *args)
        if not self.done and sql.strip().upper() == "VACUUM":
            self.done = True
            self.hook()
        return out

    def __getattr__(self, name):
        return getattr(self.db, name)


class EndsAtVacuum:
    """The store's connection as it is, until VACUUM: there the process ends (a signal, a closed terminal)."""

    def __init__(self, db):
        self.db = db

    def execute(self, sql, *args):
        if sql.strip().upper() == "VACUUM":
            raise KeyboardInterrupt("the process ends here")
        return self.db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.db, name)


def _owed(owned):
    store = owned.mod.open_to_read("oo")
    try:
        return store.meta("rewrite")
    finally:
        store.close()


def test_mo6_a_pass_of_the_route_that_drops_parts_rewrites_the_store(owned, tmp_path, monkeypatch):
    """MO6 -- kept modules and modules to go, indexed through the route in passes that changed some, then the modules
    to go removed and the route run again: it says it rewrote the store, and once closed none of the removed modules'
    names, paths, sentences or words is in the store's bytes, the kept ones are. A pass that only changed files (their
    names renamed inside): rewritten too, the old names gone. The rewrite is owed by the pass's own transaction
    (review 1): a rewrite that fails -- a full disk; another connection holding the store right after the pass -- is
    said, exit 1, owed, said by the status; a process that ends during the rewrite leaves it owed; and the next pass
    of the route, dropping nothing, rewrites it, exit 0, owed no more. A pass of another connection that drops words
    right after the rewrite ran keeps its own debt (review 2); a recut owes none. Dropped words left in the pages, or
    a rewrite owed nowhere -> RED."""
    from opti_oignon.mycelium.store import Store

    n = 150
    src = tmp_path / "src"
    _two_rounds(owned, src, n)
    for path in src.glob("zxpath*.py"):
        path.unlink()
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    assert code == 0 and f"{n} removed" in out and "rewritten" in out, out
    raw = (owned.folder / "oo.db").read_bytes()
    assert _markers(raw, "zx", n) == [], _markers(raw, "zx", n)[:10]
    assert len(_markers(raw, "kq", n)) > n

    changed = tmp_path / "changed"
    _two_rounds(owned, changed, n, corpus="o2")
    for path in changed.glob("zxpath*.py"):
        path.write_text(path.read_text(encoding="utf-8").replace("zx", "yy"), encoding="utf-8")
    code, out = owned.run("mycelium", "index", "o2", str(changed), "--no-vectors")
    assert code == 0 and f"{n} changed" in out and "rewritten" in out, out
    raw = (owned.folder / "o2.db").read_bytes()
    assert [m for m in _markers(raw, "zx", n) if "path" not in m] == [], "a changed file's old names stayed"

    real = Store.purge

    def full(self, pages=None):
        raise sqlite3.OperationalError("database or disk is full")

    (src / "kqpath000.py").unlink()
    monkeypatch.setattr(Store, "purge", full)
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    assert code == 1 and "1 removed" in out and "stay" in out, out
    monkeypatch.setattr(Store, "purge", real)
    code, out = owned.run("mycelium", "status", "oo")
    assert code == 0 and "rewrite is owed" in out, out
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    assert code == 0 and "0 removed" in out and "rewritten" in out, out
    code, out = owned.run("mycelium", "status", "oo")
    assert code == 0 and "owed" not in out, out
    raw = (owned.folder / "oo.db").read_bytes()
    assert b"kqfunc000" not in raw and b"kqfunc001" in raw

    (src / "kqpath001.py").unlink()
    writer = sqlite3.connect(owned.folder / "oo.db", timeout=WAIT)
    real_rewrite = Store.rewrite

    def writer_first(self):
        writer.execute("BEGIN IMMEDIATE")
        return real_rewrite(self)

    monkeypatch.setattr(Store, "rewrite", writer_first)
    try:
        code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    finally:
        writer.rollback()
        writer.close()
    monkeypatch.setattr(Store, "rewrite", real_rewrite)
    assert code == 1 and "1 removed" in out and "stay" in out and (_owed(owned) or "").startswith("owed"), out

    (src / "kqpath002.py").unlink()
    real_open = owned.mod.open_to_write

    def ends_at_vacuum(corpus, make=True):
        store = real_open(corpus, make)
        store.db = EndsAtVacuum(store.db)
        return store

    monkeypatch.setattr(owned.mod, "open_to_write", ends_at_vacuum)
    with pytest.raises(KeyboardInterrupt):
        owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    monkeypatch.setattr(owned.mod, "open_to_write", real_open)
    assert (_owed(owned) or "").startswith("owed"), "a rewrite cut short by the process's end is owed nowhere"
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    assert code == 0 and "0 removed" in out and "rewritten" in out and _owed(owned) is None, out
    raw = (owned.folder / "oo.db").read_bytes()
    assert b"kqfunc001" not in raw and b"kqfunc002" not in raw and b"kqfunc003" in raw

    # Review 2: another pass that drops words between the rewrite's end and the clearing of the debt keeps its own.
    (src / "kqpath003.py").unlink()
    path = owned.folder / "oo.db"
    meanwhile = []

    def another_pass():
        from opti_oignon.mycelium.index import Indexer

        (src / "kqpath004.py").write_text("def changed():\n    return 4\n", encoding="utf-8")
        second = Store(path, _quick)
        try:
            meanwhile.append(Indexer(second).index("oo", src)["changed"])
        finally:
            second.close()

    def after_vacuum(corpus, make=True):
        store = real_open(corpus, make)
        store.db = AfterVacuum(store.db, another_pass)
        return store

    monkeypatch.setattr(owned.mod, "open_to_write", after_vacuum)
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    monkeypatch.setattr(owned.mod, "open_to_write", real_open)
    assert code == 0 and "rewritten" in out and meanwhile == [1], (meanwhile, out)
    assert (_owed(owned) or "").startswith("owed"), "the rewrite cleared the debt of a pass made after it ran"
    code, out = owned.run("mycelium", "status", "oo")
    assert "rewrite is owed" in out, out
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    assert code == 0 and "rewritten" in out and _owed(owned) is None, out

    # A recut drops no word the store does not hold again: it owes no rewrite.
    from opti_oignon.mycelium import index as index_mod

    monkeypatch.setattr(index_mod, "CHUNKER_VERSION", "another")
    code, out = owned.run("mycelium", "index", "oo", str(src), "--no-vectors")
    assert code == 0 and "rewritten" not in out and _owed(owned) is None, out


def test_mo7_the_documents_reader_asks_the_store(tmp_path):
    """MO7 -- MP10, superseded: forgetting a source of documents through the indexer that indexed it, an old citation
    is not said fresh and is not read, by a searcher made before or after; a later partial pass does not find the
    forgotten document, which the store holds no more (None, where MP10 said UNKNOWN). And the reader asks the store:
    a document forgotten by another connection is no longer served, nor kept by a later partial pass; one changed by
    another indexer is UNKNOWN to this one; one the store holds and this indexer was never given is UNKNOWN; one
    neither holds is None. A text served after the store let it go -> RED."""
    from opti_oignon.mycelium.index import UNKNOWN, Indexer
    from opti_oignon.mycelium.search import Searcher, verify
    from opti_oignon.mycelium.store import Store

    store = Store(tmp_path / "s.db", sqlite3.connect)
    indexer = Indexer(store)
    secret = "# Secret\n\nquizzacious plans of the user.\n"
    indexer.index_documents("d", {"x.md": secret, "y.md": "# Other\n\nplain words.\n"})
    indexer.index("f", _folder(tmp_path / "src", {"gate.py": GATE}))
    before = Searcher(store, readers=indexer.readers)
    cited = before.search("quizzacious plans", 3, stages=("lexical",))[0]
    assert cited["path"] == "x.md" and verify(cited, indexer.readers) == "fresh" and before.read(cited)
    counts = indexer.forget("d")
    assert counts["merged"] is True
    assert [s["name"] for s in store.sources()] == ["f"]
    assert verify(cited, indexer.readers) != "fresh"
    assert before.read(cited) is None and Searcher(store, readers=indexer.readers).read(cited) is None
    indexer.index_documents("d", {"y.md": "# Other\n\nplain words.\n"}, complete=False)
    assert indexer.readers["d"]("x.md") is None, "a partial pass found the forgotten document"
    assert Searcher(store, readers=indexer.readers).search("check_entity", 3, stages=("symbols",))
    store.close()

    path = tmp_path / "t.db"
    server = Store(path, sqlite3.connect)
    indexer = Indexer(server)
    indexer.index_documents("d", {"x.md": secret, "y.md": "# Other\n\nplain words.\n"})
    cited = Searcher(server, readers=indexer.readers).search("quizzacious plans", 3, stages=("lexical",))[0]
    other = Store(path, sqlite3.connect)
    other.forget("d")
    other.close()
    assert indexer.readers["d"]("x.md") is None, "the indexer serves a document another connection forgot"
    assert verify(cited, indexer.readers) != "fresh"
    indexer.index_documents("d", {"y.md": "# Other\n\nplain words.\n"}, complete=False)
    assert indexer.readers["d"]("x.md") is None, "a partial pass kept a document the store let go"
    assert "x.md" not in indexer._documents["d"], "the indexer still holds the text of a document the store let go"
    elsewhere = Indexer(Store(path, sqlite3.connect))
    elsewhere.index_documents("d", {"y.md": "# Other\n\nchanged words.\n", "z.md": "# Z\n\nzed.\n"}, complete=False)
    assert indexer.readers["d"]("y.md") is UNKNOWN, "a document changed by another indexer served as it was"
    assert indexer.readers["d"]("z.md") is UNKNOWN
    assert indexer.readers["d"]("never.md") is None
    assert elsewhere.readers["d"]("z.md") == "# Z\n\nzed.\n"
    elsewhere.store.close()
    server.close()


def _journal(path):
    db = sqlite3.connect(path)
    try:
        return db.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        db.close()


def test_mo8_a_store_is_written_in_the_rollback_journal(owned, tmp_path):
    """MO8 -- a store set to WAL by another program is back in the rollback journal (DELETE) once opened to write,
    and no -wal or -shm stays beside it once closed, nor once read; one that another connection holds open in WAL is
    refused by name, left in WAL. A store written in WAL -> RED."""
    from opti_oignon.mycelium.index import Indexer
    from opti_oignon.mycelium.store import StoreError

    src = _folder(tmp_path / "src", {"gate.py": GATE})
    store = owned.mod.open_to_write("oo")
    Indexer(store).index("oo", src)
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    store.close()
    path = owned.folder / "oo.db"
    db = sqlite3.connect(path)
    assert db.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    db.close()
    assert _journal(path) == "wal"
    store = owned.mod.open_to_write("oo")
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    store.close()
    beside = sorted(p.name for p in owned.folder.iterdir())
    assert beside == ["oo.db"], beside
    owned.mod.open_to_read("oo").close()
    assert sorted(p.name for p in owned.folder.iterdir()) == ["oo.db"]
    holder = sqlite3.connect(path, timeout=WAIT)
    holder.execute("PRAGMA journal_mode = WAL")
    holder.execute("SELECT COUNT(*) FROM file").fetchone()
    try:
        with pytest.raises(StoreError, match="WAL"):
            owned.mod.open_to_write("oo")
    finally:
        holder.close()


def test_mo9_the_routes_vector_pass(owned, tmp_path, monkeypatch):
    """MO9 -- under the package's own settings (``mycelium.yaml``: embeddinggemma-2, 768), the route's pass is
    followed by the vector pass, every request admitted by the governor as background work: its counts said, the
    encoder read. A governor holding: said, exit 0, the index pass standing. An encoder the engine does not serve:
    said by name, exit 1, the index pass standing. Settings refused (review 1): nothing indexed, the store as it was --
    the header they name decides the parts' embedded texts, and another would cut every file again and drop every
    vector --, said, exit 1, with or without the vectors. A batch that finds the store written by
    another pass: the vector pass stops as held, said, what it wrote kept, exit 0; the next pass takes up the rest.
    Vectors asked outside the governor, or a store written by another pass breaking the route -> RED."""
    from opti_oignon.mycelium import vectors

    def install(engine, governor):
        monkeypatch.setattr(vectors, "_registry", lambda: Registry({"embeddinggemma-2": engine}))
        monkeypatch.setattr(vectors, "_governor", lambda: governor)

    src = _folder(tmp_path / "src", {"gate.py": GATE, "notes.md": NOTES})
    engine, governor = Engine(dim=768), Governor()
    install(engine, governor)
    code, out = owned.run("mycelium", "index", "oo", str(src))
    assert code == 0 and "embedded" in out, out
    assert engine.calls and governor.asked and {c for _m, c in governor.asked} == {"index"}
    store = owned.mod.open_to_read("oo")
    try:
        assert store.active_encoder()["key"].startswith("embeddinggemma-2")
    finally:
        store.close()

    (src / "held.md").write_text("# Held\n\nwords held back.\n", encoding="utf-8")
    install(Engine(dim=768), Governor(hold_callers={"index"}))
    code, out = owned.run("mycelium", "index", "oo", str(src))
    assert code == 0 and "1 added" in out and "held" in out, out

    (src / "refused.md").write_text("# Refused\n\nsettings refused.\n", encoding="utf-8")
    real = vectors.load_settings

    def malformed(path=None):
        raise vectors.VectorError("mycelium.yaml: vectors.batch is not a positive number")

    monkeypatch.setattr(vectors, "load_settings", malformed)
    before = _snapshot(owned.folder)
    for vectors_too in ((), ("--no-vectors",)):
        code, out = owned.run("mycelium", "index", "oo", str(src), *vectors_too)
        assert code == 1 and "vectors.batch" in out and "added" not in out, out
    assert _snapshot(owned.folder) == before, "a pass ran under settings that were refused"
    monkeypatch.setattr(vectors, "load_settings", real)

    (src / "missing.md").write_text("# Missing\n\nno encoder.\n", encoding="utf-8")
    install(Engine(dim=768, endpoint=False), Governor())
    code, out = owned.run("mycelium", "index", "oo", str(src))
    assert code == 1 and "2 added" in out and "stopped" in out, out

    other = sqlite3.connect(owned.folder / "oo.db", timeout=WAIT, check_same_thread=False)

    class Writer(Engine):
        def embed_many(self, model, texts, timeout=None):
            if not self.calls:
                other.execute("BEGIN IMMEDIATE")
            return super().embed_many(model, texts, timeout)

    install(Writer(dim=768), Governor())
    try:
        code, out = owned.run("mycelium", "index", "oo", str(src))
    finally:
        other.rollback()
        other.close()
    assert code == 0 and "held" in out and "another pass writes the store" in out, out
    install(Engine(dim=768), Governor())
    code, out = owned.run("mycelium", "index", "oo", str(src))
    assert code == 0 and "embedded" in out, out
    store = owned.mod.open_to_read("oo")
    try:
        lacking = store.db.execute("SELECT COUNT(*) FROM part WHERE embed_sha NOT IN (SELECT embed_sha FROM vector)")
        assert lacking.fetchone()[0] == 0
    finally:
        store.close()


def test_mo10_mxbais_context_is_its_models():
    """MO10 -- ``config/models.yaml`` gives mxbai-embed-large the 512 tokens of context its engine shows, not 8192
    (BACKLOG 235 a). A context the model does not have -> RED."""
    import yaml

    config = importlib.import_module("opti_oignon.config")
    raw = yaml.safe_load(Path(config.MODELS_CONFIG_FILE).read_text(encoding="utf-8"))
    assert raw["context_windows"]["mxbai-embed-large"] == 512


def test_mo11_the_real_package_shape(tmp_path, monkeypatch, capsys):
    """MO11 -- ``python -m opti_oignon mycelium index oo FOLDER --no-vectors`` then ``... status`` through the
    package's own ``__main__`` and the store's own opener, the data folder a test folder: the platform's opener asked
    to write by the pass and to read by the status, the corpus said. A route that reads through a writable connection
    -> RED."""
    from opti_oignon import db_utils
    from opti_oignon.mycelium import place

    config = importlib.import_module("opti_oignon.config")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    asked = []

    def platform(path, **kw):
        asked.append(kw.get("read_only", False))
        return _quick(path, read_only=kw.get("read_only", False))

    monkeypatch.setattr(db_utils, "safe_connect", platform)
    src = _folder(tmp_path / "src", {"gate.py": GATE})
    for argv in (("mycelium", "index", "oo", str(src), "--no-vectors"), ("mycelium", "status")):
        monkeypatch.setattr(sys, "argv", ["opti-oignon", *argv])
        try:
            runpy.run_module("opti_oignon", run_name="__main__", alter_sys=False)
        except SystemExit as exc:
            assert not exc.code, exc
    assert asked == [False, True], asked
    assert place.corpora() == ["oo"]
    assert "oo" in capsys.readouterr().out
