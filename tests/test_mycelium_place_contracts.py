"""Contracts of the Mycelium's stores in their place: where they lie, how they are read, how a source is forgotten,
and the account wipe.

A store lies at ``<data folder>/mycelium/<corpus>.db``, the data folder read at each call, a corpus named by a closed
rule. Reading a corpus creates nothing and writes nothing; forgetting a source leaves nothing of it, its words out of
the store's bytes, and moves the store's generation to a value never read before; the wipe, for a single-user
install's one user, overwrites and removes every store.

  * MP1 -- THE PLACE AND THE NAMES: a corpus's store under the data folder as it is at the call; every name outside
    the rule refused by name; the corpora are the stores the folder holds, an absent folder none, and not made.
  * MP2 -- READING CREATES NOTHING: a corpus with no store is refused by its name, and neither the folder nor a file
    is made; a missing file opened read-only is refused too.
  * MP3 -- READING WRITES NOTHING: a store read, searched and closed keeps its bytes and its time; a write through a
    reader is refused; a newer schema, a schema-1 store, another kind of database and an empty file are refused by
    name and left as they were.
  * MP4 -- FORGETTING A SOURCE: its files, parts, tokens, names and references go, its own vectors, settings and
    episodes too; another source's, and a vector it shares, stay; an episode says it with numbers; the generation
    moves, and never back to a value read before, even when the same name is indexed again under the same id; an
    unknown name is refused and changes nothing.
  * MP5 -- THE FORGOTTEN WORDS LEAVE THE BYTES: a rare word of a forgotten source is nowhere in the store's file once
    it is closed; another source's word still is.
  * MP6 -- THE WIPE: every store of the folder and its journal files overwritten and removed, other files kept; an
    absent folder not made; a store that stays raised by name; through the account wipe, for one user, emptied, and
    named by its label when one stays; the inventory says so.
  * MP7 -- XO6 of tests/test_code_outputs_forgetting_contracts.py, which counted six copies of the overwrite: the
    seventh, the Mycelium's place's, the same text; XO6's assertions re-run unchanged on the six.
  * MP8 -- THE REAL PACKAGE SHAPE: the real configuration's data folder moved to a test folder, a corpus written,
    indexed, read and searched through the real modules; the store's own opener asks the platform's for a read-only
    connection when it reads.
  * MP9 -- THE ENCRYPTION COMMAND TAKES THEM (review 1): a store is <corpus>.db in the Mycelium's folder, a store
    folder of the encryption command: one written in plaintext is among its stores and said in plaintext, never left
    as another database.
  * MP10 -- AN INDEXER FORGETS WHAT IT HELD (review 1): forgetting a source of documents through the indexer that
    indexed it drops the texts it kept to check citations: an old citation is no longer read nor said fresh, by a
    searcher made before or after, and a later partial pass does not find the forgotten documents.
  * MP11 -- A FORGETTING SAYS WHETHER ITS WORDS LEFT THE BYTES (review 1): it returns its counts and whether the
    index was merged; a merge that fails leaves the source forgotten, says so, and is logged.
  * MP12 -- WHAT CANNOT BE OPENED TO READ IS REFUSED AS A STORE, AND A LINK IS NEVER FOLLOWED (review 1): an error of
    the platform's opener is a StoreError naming the store; a store's place that is a symbolic link is refused for
    reading and writing, its target untouched.
"""
import ast
import importlib
import os
import sqlite3
import stat
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_deletion_leaves_nothing_contracts as dl  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_mp1_the_place_and_the_names": 2.0,
    "test_mp2_reading_creates_nothing": 2.0,
    "test_mp3_reading_writes_nothing": 3.0,
    "test_mp4_forgetting_a_source": 3.0,
    "test_mp5_the_forgotten_words_leave_the_bytes": 3.0,
    "test_mp6_the_wipe": 5.0,
    "test_mp7_the_overwrite_is_one_text_in_its_seven_copies": 5.0,
    "test_mp8_the_real_package_shape": 4.0,
    "test_mp9_the_encryption_command_takes_the_myceliums_stores": 4.0,
    "test_mp10_an_indexer_forgets_what_it_held_of_a_forgotten_source": 3.0,
    "test_mp11_a_forgetting_says_whether_its_words_left_the_bytes": 3.0,
    "test_mp12_what_cannot_be_opened_to_read_is_refused_as_a_store_and_a_link_never_followed": 3.0,
}

_PLACE = "opti_oignon.mycelium.place"
_STORE = "opti_oignon.mycelium.store"

GATE = '''"""The gate checks entities."""


def check_entity(name):
    """Check one entity."""
    return name.strip()
'''


def _folder(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _plain(path, read_only=False):
    """The platform's opener as a plain SQLite one: read-only through a URI, never creating the file."""
    if read_only:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    return sqlite3.connect(path, check_same_thread=False)


@pytest.fixture
def place(tmp_path, monkeypatch):
    """The real place and store modules, the data folder a test folder, the store's opener a plain one."""
    config = importlib.import_module("opti_oignon.config")
    from opti_oignon.mycelium import place as place_mod
    from opti_oignon.mycelium import store as store_mod

    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(store_mod, "_safe_connect", _plain)
    return types.SimpleNamespace(mod=place_mod, store=store_mod, data=data, folder=data / "mycelium")


def _snapshot(folder):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(folder.iterdir())}


def test_mp1_the_place_and_the_names(place, tmp_path, monkeypatch):
    """MP1 -- a corpus's store is <data folder>/mycelium/<corpus>.db, the data folder as it is at the call; a name
    with a dot, a bar, a capital, a space, a line end, a leading '_' or '-', none, or longer than 100 characters is
    refused by name; the corpora are the stores the folder holds, a file not named like one is not one, and an absent
    folder holds none and is not made. A store placed elsewhere, a name that escapes the folder, or a folder made by
    a listing -> RED."""
    config = importlib.import_module("opti_oignon.config")

    mod = place.mod
    assert mod.path_for("oo") == place.folder / "oo.db"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "moved")
    assert mod.path_for("project_1a2b3c4d-5e6") == tmp_path / "moved" / "mycelium" / "project_1a2b3c4d-5e6.db"
    assert mod.path_for("a" * 100).name == "a" * 100 + ".db"
    for bad in ("", ".", "..", "../x", "a/b", "a\\b", "A", "x.db", "a b", "x\n", "\u00e9t\u00e9", "-x", "_x",
                "a" * 101, None, 3):
        with pytest.raises(place.store.StoreError) as refused:
            mod.path_for(bad)
        assert repr(bad) in str(refused.value), (bad, str(refused.value))
    assert mod.corpora() == [] and not (tmp_path / "moved").exists(), "a listing made the folder"
    monkeypatch.setattr(config, "DATA_DIR", place.data)
    place.folder.mkdir(parents=True)
    for name in ("oo.db", "b.db", "Bad.db", "x.txt", "c.db-journal", ".db"):
        (place.folder / name).write_bytes(b"")
    (place.folder / "d.db").mkdir()
    assert mod.corpora() == ["b", "oo"]


def test_mp2_reading_creates_nothing(place):
    """MP2 -- reading a corpus with no store: refused by its name (NoStore, a StoreError), the folder not made; the
    folder there and the file not, nothing made in it; a missing file opened read-only by the store itself refused,
    nothing made, even through an opener that would make it. A store or a folder created by a reader -> RED."""
    mod = place.mod
    with pytest.raises(mod.NoStore) as refused:
        mod.open_to_read("oo")
    assert "'oo'" in str(refused.value) and isinstance(refused.value, place.store.StoreError)
    assert not place.data.exists(), "a reader made the data folder"
    place.folder.mkdir(parents=True)
    with pytest.raises(mod.NoStore):
        mod.open_to_read("oo")
    assert list(place.folder.iterdir()) == [], "a reader made a file"
    for connect in (None, sqlite3.connect):    # the platform's read-only opener; one that would make the file
        with pytest.raises(place.store.StoreError):
            place.store.Store(place.folder / "gone.db", connect, read_only=True)
        assert list(place.folder.iterdir()) == [], connect


def test_mp3_reading_writes_nothing(place, tmp_path):
    """MP3 -- a store written and closed, then read: its sources, generation and a search, then closed: the folder's
    files keep their bytes and their times, no journal left; a write through the reader is refused, the bytes kept;
    a store at a newer schema, one at schema 1, a database of another kind and an empty file: refused by name, each
    left as it was. A reader that stamps, migrates, checkpoints or creates -> RED."""
    from opti_oignon.mycelium.index import Indexer
    from opti_oignon.mycelium.search import Searcher

    mod = place.mod
    root = _folder(tmp_path / "src", {"gate.py": GATE})
    writer = mod.open_to_write("oo")
    Indexer(writer).index("oo", root)
    writer.close()
    before = _snapshot(place.folder)
    reader = mod.open_to_read("oo")
    assert [s["name"] for s in reader.sources()] == ["oo"]
    reader.generation()
    assert Searcher(reader).search("check_entity", 3, stages=("symbols",))
    with pytest.raises(Exception):
        reader.meta("vectors", 7)
        reader.db.commit()
    reader.close()
    assert _snapshot(place.folder) == before, "reading wrote to the store"

    def planted(name, statements):
        db = sqlite3.connect(place.folder / f"{name}.db")
        for statement in statements:
            db.execute(statement)
        db.commit()
        db.close()
        return (place.folder / f"{name}.db").read_bytes()

    cases = {
        "newer": (["CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                   "INSERT INTO meta VALUES ('schema', '3')"], "schema 3"),
        "old": (["CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                 "INSERT INTO meta VALUES ('schema', '1')"], "schema 1"),
        "foreign": (["CREATE TABLE notes(body TEXT)", "INSERT INTO notes VALUES ('mine')"], "not a Mycelium store"),
    }
    for name, (statements, said) in cases.items():
        held = planted(name, statements)
        with pytest.raises(place.store.StoreError) as refused:
            mod.open_to_read(name)
        assert said in str(refused.value), (name, str(refused.value))
        assert (place.folder / f"{name}.db").read_bytes() == held, f"{name}: the refused file was changed"
    (place.folder / "empty.db").write_bytes(b"")
    with pytest.raises(place.store.StoreError) as refused:
        mod.open_to_read("empty")
    assert "not a Mycelium store" in str(refused.value)
    assert (place.folder / "empty.db").read_bytes() == b""
    assert sorted(p.name for p in place.folder.iterdir()) == ["empty.db", "foreign.db", "newer.db", "old.db",
                                                              "oo.db"], "a reader left a file behind"


def _two_sources(tmp_path, store, first="b", second="a"):
    """Two sources, ``first`` indexed first: each holds shared.md (the same text) and a file of its own."""
    from opti_oignon.mycelium.index import Indexer

    roots = {
        "a": _folder(tmp_path / "ra", {"shared.md": "# Shared\n\nThe same words in both.\n",
                                       "own_a.py": GATE.replace("entity", "alphaentity")}),
        "b": _folder(tmp_path / "rb", {"shared.md": "# Shared\n\nThe same words in both.\n",
                                       "own_b.py": GATE.replace("entity", "betaentity")}),
    }
    indexer = Indexer(store)
    for name in (first, second):
        indexer.index(name, roots[name])
    return roots, indexer


def test_mp4_forgetting_a_source(tmp_path):
    """MP4 -- two sources, each with a file of its own and one of the same text, a vector for every text: forgetting
    one removes its source row, its files, parts, tokens, names, references, its vectors but the shared one, its
    cutter, header and pass settings, its episodes; the other's all stay and its words are found; an episode
    'forget' with no source and numbers only; the generation moves. Then, with no vector at all: the source forgotten
    and indexed again under the same name, files and id: the generation is not the one read before the forgetting.
    An unknown name is refused by name and moves nothing. Anything of the forgotten source left, the other's touched,
    or a generation that comes back -> RED."""
    from opti_oignon.mycelium.index import Indexer
    from opti_oignon.mycelium.search import Searcher
    from opti_oignon.mycelium.store import Store, StoreError

    store = Store(tmp_path / "s.db", sqlite3.connect)
    roots, _indexer = _two_sources(tmp_path, store)
    db = store.db
    ids = {s["name"]: s["id"] for s in store.sources()}
    for (embed_sha,) in db.execute("SELECT DISTINCT embed_sha FROM part").fetchall():
        db.execute("INSERT INTO vector(embed_sha, encoder, dim, data) VALUES (?, 'enc', 2, ?)", (embed_sha, b"\0" * 4))
    store.vectors_changed()
    db.commit()
    shared = {r[0] for r in db.execute("SELECT p.embed_sha FROM part p JOIN file f ON f.id = p.file "
                                       "WHERE f.source = ? AND f.path = 'shared.md'", (ids["a"],))}
    own_a = {r[0] for r in db.execute("SELECT p.embed_sha FROM part p JOIN file f ON f.id = p.file "
                                      "WHERE f.source = ? AND f.path = 'own_a.py'", (ids["a"],))}
    assert shared and own_a and not shared & own_a
    b_before = sorted(db.execute("SELECT f.path, p.sha256 FROM part p JOIN file f ON f.id = p.file "
                                 "WHERE f.source = ?", (ids["b"],)))
    generation = store.generation()
    store.forget("a")
    assert [s["name"] for s in store.sources()] == ["b"]
    assert db.execute("SELECT COUNT(*) FROM file WHERE source = ?", (ids["a"],)).fetchone()[0] == 0
    alive = {r[0] for r in db.execute("SELECT id FROM part")}
    assert db.execute("SELECT COUNT(*) FROM part WHERE file NOT IN (SELECT id FROM file)").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM fts").fetchone()[0] == len(alive), "tokens of the forgotten parts stay"
    for table in ("symbol", "ref"):
        assert db.execute(f"SELECT COUNT(*) FROM {table} WHERE part NOT IN (SELECT id FROM part)").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM symbol WHERE name = 'check_alphaentity'").fetchone()[0] == 0
    vectors = {r[0] for r in db.execute("SELECT embed_sha FROM vector")}
    assert shared <= vectors and not own_a & vectors, "the shared vector went, or the forgotten one stayed"
    keys = {r[0] for r in db.execute("SELECT key FROM meta")}
    assert not {f"cutter:{ids['a']}", f"header:{ids['a']}", f"pass:{ids['a']}"} & keys, keys
    assert {f"cutter:{ids['b']}", f"header:{ids['b']}", f"pass:{ids['b']}"} <= keys, keys
    assert store.episodes(ids["a"]) == [], "the forgotten source's episodes stay"
    last = store.episodes(None, 1)[0]
    assert last["action"] == "forget" and last["source"] is None
    assert all(type(v) in (int, float, bool) for v in last["detail"].values()) and last["detail"]
    assert sorted(db.execute("SELECT f.path, p.sha256 FROM part p JOIN file f ON f.id = p.file "
                             "WHERE f.source = ?", (ids["b"],))) == b_before
    assert Searcher(store).search("check_betaentity", 3, stages=("symbols",))
    assert not Searcher(store).search("check_alphaentity", 3, stages=("symbols",))
    assert store.generation() != generation
    moved = store.generation()
    with pytest.raises(StoreError) as refused:
        store.forget("zz")
    assert "'zz'" in str(refused.value) and store.generation() == moved
    store.close()

    again = Store(tmp_path / "again.db", sqlite3.connect)
    _two_sources(tmp_path, again)
    held = {s["name"]: s["id"] for s in again.sources()}
    read = again.generation()
    again.forget("a")
    Indexer(again).index("a", roots["a"])
    assert {s["name"]: s["id"] for s in again.sources()} == held, "control: the same id came back"
    assert again.generation() != read, "the generation came back to a value read before the forgetting"
    again.close()


def _many(root, tag, n):
    """``n`` modules whose words are ``tag`` + a kind + a number: a sentence, a function, a word, a callee."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"{tag}path{i:03d}.py").write_text(
            f'"""The {tag}sentence{i:03d} opens this module."""\n\n\n'
            f"def {tag}func{i:03d}(x):\n"
            f'    """Body {tag}word{i:03d} here."""\n'
            f"    return {tag}callee{i:03d}(x)\n", encoding="utf-8")
    return root


def test_mp5_the_forgotten_words_leave_the_bytes(tmp_path):
    """MP5 -- two sources of many modules, each indexed in several passes so the index holds several segments: once
    one is forgotten and the store closed, none of its words, names, first sentences or paths is in the store's file;
    the other's paths and names all still are. Then a small source forgotten from a large store whose index is one
    merged segment (review 2: the shape where the index's own merge of deleted rows does not rewrite it): none of its
    words is left either. Forgotten words left in the index's pages, a merge bounded to a few pages, or the other
    source's paths or names lost -> RED."""
    from opti_oignon.mycelium.index import Indexer
    from opti_oignon.mycelium.store import Store

    n = 150
    path = tmp_path / "s.db"
    store = Store(path, sqlite3.connect)
    indexer = Indexer(store)
    keep, gone = _many(tmp_path / "keep", "kq", n), _many(tmp_path / "gone", "zx", n)
    indexer.index("keep", keep)
    indexer.index("gone", gone)
    for i in range(0, n, 7):
        for root, tag in ((keep, "kq"), (gone, "zx")):
            module = root / f"{tag}path{i:03d}.py"
            module.write_text(module.read_text(encoding="utf-8") + "\n# more\n", encoding="utf-8")
    indexer.index("keep", keep)
    indexer.index("gone", gone)
    store.db.commit()
    kinds = ("word", "sentence", "path", "func", "callee")

    def left(raw, tag):
        return [f"{tag}{kind}{i:03d}" for kind in kinds for i in range(n) if f"{tag}{kind}{i:03d}".encode() in raw]

    assert len(left(path.read_bytes(), "zx")) > n, "control: the words were indexed"
    store.forget("gone")
    store.close()
    raw = path.read_bytes()
    assert left(raw, "zx") == [], "the forgotten source's words stay in the store's bytes"
    # The full-text index keeps words by their shared heads; the paths and names the tables keep are whole.
    kept = [f"kq{kind}{i:03d}" for kind in ("path", "func") for i in range(n)]
    assert [word for word in kept if word.encode() not in raw] == [], "the other source lost its paths or names"

    path = tmp_path / "big.db"
    store = Store(path, sqlite3.connect)
    indexer = Indexer(store)
    indexer.index("keep", _many(tmp_path / "keep", "kq", 1500))
    indexer.index("small", _many(tmp_path / "small", "zy", 4))
    store.purge(None)
    store.db.commit()
    assert len([1 for kind in kinds for i in range(4) if f"zy{kind}{i:03d}".encode() in path.read_bytes()]) > 4
    store.forget("small")
    store.close()
    raw = path.read_bytes()
    assert [f"zy{kind}{i:03d}" for kind in kinds for i in range(4) if f"zy{kind}{i:03d}".encode() in raw] == [], (
        "a small forgotten source's words stay in a large store's merged index")


def _place_alone(data):
    """place.py alone, the data folder ``data``; the closer restores."""
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = data
    loaded, restore = isolate(targets={_STORE: source("mycelium", "store.py"), _PLACE: source("mycelium", "place.py")},
                              seeded={"opti_oignon.config": config}, packages=("opti_oignon.mycelium",))
    return loaded[_PLACE], restore


def test_mp6_the_wipe(tmp_path):
    """MP6 -- remove_every_store: each <corpus>.db and its -journal, -wal and -shm files overwritten (a handle held
    on one reads zeros) and removed, a file not of a store kept; an absent folder holds none and is not made; a store
    that cannot be removed is raised by its name. Through the account wipe, for a single-user install's one user:
    emptied and nothing said; one that stays named 'the Mycelium's indexes (not emptied: ...)'; the label among the
    stores removed for one user only, named otherwise. The inventory's entry: drawn words, one user. A store left
    by the wipe, its bytes left on the disk, or a failure unsaid -> RED."""
    data = tmp_path / "data"
    place, close = _place_alone(data)
    try:
        place.remove_every_store()
        assert not data.exists(), "the wipe made the folder"
        folder = data / "mycelium"
        folder.mkdir(parents=True)
        for name in ("oo.db", "oo.db-journal", "b.db", "b.db-wal", "b.db-shm", "b.db.encrypting", "c.db.restoring"):
            (folder / name).write_bytes(b"secret words " * 8)
        (folder / "notes.txt").write_bytes(b"mine")
        handle = os.open(folder / "oo.db", os.O_RDONLY)
        try:
            place.remove_every_store()
            assert os.pread(handle, 104, 0) == b"\0" * 104, "the store was removed without its overwrite"
        finally:
            os.close(handle)
        assert sorted(p.name for p in folder.iterdir()) == ["notes.txt"]
        if os.geteuid() != 0:
            (folder / "stuck.db").write_bytes(b"x")
            folder.chmod(0o500)
            try:
                with pytest.raises(OSError) as stayed:
                    place.remove_every_store()
            finally:
                folder.chmod(0o700)
            assert "stuck.db" in str(stayed.value)
            (folder / "stuck.db").unlink()
    finally:
        close()

    place, close = _place_alone(tmp_path / "world" / "data")
    try:
        world = dl._world(tmp_path / "world", stores={_PLACE: place}, wipe=True, packages=("opti_oignon.mycelium",))
        try:
            udm = world.mod("user_data_manager")
            folder = world.data / "mycelium"
            folder.mkdir()
            (folder / "oo.db").write_bytes(b"words")
            assert udm.UserDataDeleter._empty_copies() == []
            assert list(folder.iterdir()) == [], "the wipe left a store"
            if os.geteuid() != 0:
                (folder / "stuck.db").write_bytes(b"x")
                folder.chmod(0o500)
                try:
                    failed = udm.UserDataDeleter._empty_copies()
                finally:
                    folder.chmod(0o700)
                assert [line for line in failed if line.startswith("the Mycelium's indexes (not emptied")], failed
            label = next(label for label in udm.WIPE_SINGLE_USER if label.startswith("the Mycelium's indexes"))
            assert label in udm.WIPE_NOT_COVERED, "the stores are removed for one user only, named otherwise"
        finally:
            world.restore()
    finally:
        close()
    loaded, restore = isolate(targets={"opti_oignon.store_inventory": source("store_inventory.py")})
    try:
        inventory = loaded["opti_oignon.store_inventory"]
        entry = next(e for e in inventory.STORES if e.names == ("mycelium",))
        assert (entry.label, entry.holds, entry.wipe, entry.failures) == (
            label, inventory.DRAWN, inventory.ONE_USER, ("the Mycelium's indexes",)), entry
        assert "mycelium/place.py" in entry.modules
    finally:
        restore()


def test_mp7_the_overwrite_is_one_text_in_its_seven_copies(tmp_path, monkeypatch):
    # XO6 of tests/test_code_outputs_forgetting_contracts.py, which counted six copies, re-run unchanged on them; the
    # seventh, the Mycelium's place's, is the same text.
    import test_code_outputs_forgetting_contracts as xo
    import test_project_forgetting_contracts as pf

    copies = pf._copies()
    assert sorted(copies) == ["agent/skills.py", "code_executor.py", "history.py", "learned_router.py",
                              "mycelium/place.py", "notes/blob_store.py", "projects.py"], sorted(copies)
    assert len(set(copies.values())) == 1, "the copies of the overwrite differ"
    six = {path: text for path, text in copies.items() if path != "mycelium/place.py"}
    monkeypatch.setattr(pf, "_copies", lambda: dict(six))
    xo.test_xo6_the_overwrite_is_one_text_in_its_six_copies(tmp_path, monkeypatch)
    tree = ast.parse(source("mycelium", "place.py").read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"remove_every_store", "_overwrite_and_remove", "_made_writable"} <= defined


def test_mp8_the_real_package_shape(place, tmp_path, monkeypatch):
    """MP8 -- through the real configuration, place, store, indexer and searcher, the data folder a test folder: a
    corpus written, indexed, closed, listed, then read and searched; the folder is the owner's alone (0o700). The
    store's own opener, reading, asks the platform's opener for a read-only connection, and writing for a writable
    one. A reader opened writable by the platform -> RED."""
    from opti_oignon.mycelium.index import Indexer
    from opti_oignon.mycelium.search import Searcher

    mod = place.mod
    writer = mod.open_to_write("oo")
    Indexer(writer).index("oo", _folder(tmp_path / "src", {"gate.py": GATE}))
    writer.close()
    assert stat.S_IMODE(place.folder.stat().st_mode) == 0o700
    assert mod.corpora() == ["oo"]
    reader = mod.open_to_read("oo")
    try:
        hits = Searcher(reader).search("check_entity", 3, stages=("symbols",))
        assert [hit["path"] for hit in hits] == ["gate.py"], hits
    finally:
        reader.close()

    from opti_oignon import db_utils

    asked = []

    def platform(path, **kw):
        asked.append(kw.get("read_only", False))
        return _plain(path, read_only=kw.get("read_only", False))

    monkeypatch.undo()
    config = importlib.import_module("opti_oignon.config")
    from opti_oignon.mycelium import store as store_mod

    monkeypatch.setattr(config, "DATA_DIR", place.data)
    monkeypatch.setattr(db_utils, "safe_connect", platform)
    assert store_mod._safe_connect is not _plain
    mod.open_to_read("oo").close()
    mod.open_to_write("oo").close()
    assert asked == [True, False], asked


def test_mp9_the_encryption_command_takes_the_myceliums_stores(place, tmp_path):
    """MP9 -- a corpus's store is <corpus>.db in the Mycelium's folder, one of the encryption command's store folders:
    a store written there in plaintext, as a mode with no key writes it, is among the command's stores, said in
    plaintext, and never one of the other databases it leaves as they are. A store the command passes over -- left in
    plaintext once the key is written, then unreadable -> RED."""
    from opti_oignon.mycelium.index import Indexer

    dbe = importlib.import_module("opti_oignon.db_encryption")
    mod = place.mod
    assert mod.path_for("oo") == place.folder / "oo.db"
    writer = mod.open_to_write("oo")
    Indexer(writer).index("oo", _folder(tmp_path / "src", {"gate.py": GATE}))
    writer.close()
    store = place.folder / "oo.db"
    assert "mycelium" in dbe._STORE_FOLDERS
    assert store in dbe.store_files([place.data]), dbe.store_files([place.data])
    assert store in dbe.plaintext_stores([place.data])
    assert store not in dbe.other_databases([place.data])


def test_mp10_an_indexer_forgets_what_it_held_of_a_forgotten_source(tmp_path):
    """MP10 -- a source of documents and a folder indexed by one indexer, a searcher made over its readers: the
    indexer forgets the documents' source -- the store forgets it, and the indexer drops the texts it kept to check
    its citations: an old citation is not said fresh and is not read, by the searcher made before or one made after;
    a later partial pass under the same name does not find the forgotten document; the folder's source stays. A
    forgotten document's text still served from the indexer's memory -> RED."""
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
    assert indexer.readers["d"]("x.md") is UNKNOWN, "a partial pass found the forgotten document"
    assert Searcher(store, readers=indexer.readers).search("check_entity", 3, stages=("symbols",))
    store.close()


def test_mp11_a_forgetting_says_whether_its_words_left_the_bytes(tmp_path, monkeypatch, caplog):
    """MP11 -- forgetting a source returns its counts -- files, parts, vectors -- and merged True once the index is
    merged; a merge that fails (a full disk) leaves the source forgotten, returns merged False and is logged as words
    that stay; so does a rewrite that fails after the merge (a statement of the store's own connection still running,
    review 2), no transaction left open. A forgetting that cannot say whether the words left -> RED."""
    from opti_oignon.mycelium.store import Store

    store = Store(tmp_path / "s.db", sqlite3.connect)
    _two_sources(tmp_path, store)
    counts = store.forget("a")
    assert counts["merged"] is True and counts["files"] == 2 and counts["parts"] >= 2, counts
    store.close()
    store = Store(tmp_path / "t.db", sqlite3.connect)
    _two_sources(tmp_path, store)

    def full(self, pages=None):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(Store, "purge", full)
    with caplog.at_level("WARNING", logger="opti_oignon.mycelium.store"):
        counts = store.forget("a")
    assert counts["merged"] is False and [s["name"] for s in store.sources()] == ["b"], counts
    assert any("stay" in record.getMessage() for record in caplog.records), [r.getMessage() for r in caplog.records]
    store.close()
    monkeypatch.undo()
    store = Store(tmp_path / "u.db", sqlite3.connect)
    _two_sources(tmp_path, store)
    running = store.db.execute("SELECT id FROM part")
    running.fetchone()
    counts = store.forget("a")
    running.fetchall()
    assert counts["merged"] is False and [s["name"] for s in store.sources()] == ["b"], counts
    assert not store.db.in_transaction
    store.close()


def test_mp12_what_cannot_be_opened_to_read_is_refused_as_a_store_and_a_link_never_followed(place, tmp_path,
                                                                                            monkeypatch):
    """MP12 -- the platform's opener failing on a store (a wrong key, a hot journal it will not roll back, a database
    in plaintext where a sealed one is read): a StoreError naming the store and why. A store's place that is a
    symbolic link: refused to read and to write, and the file it leads to unchanged. An opener's error let through as
    another kind, or a store reached through a link -> RED."""
    from opti_oignon.mycelium.index import Indexer

    mod = place.mod
    writer = mod.open_to_write("oo")
    Indexer(writer).index("oo", _folder(tmp_path / "src", {"gate.py": GATE}))
    writer.close()

    def failing(path, read_only=False):
        raise RuntimeError("SQLCipher key verification failed")

    monkeypatch.setattr(place.store, "_safe_connect", failing)
    with pytest.raises(place.store.StoreError) as refused:
        mod.open_to_read("oo")
    assert "oo.db" in str(refused.value) and "key verification failed" in str(refused.value), str(refused.value)
    monkeypatch.setattr(place.store, "_safe_connect", _plain)
    target = tmp_path / "elsewhere.db"
    target.write_bytes((place.folder / "oo.db").read_bytes())
    held = target.read_bytes()
    (place.folder / "ln.db").symlink_to(target)
    for opening in (mod.open_to_read, mod.open_to_write):
        with pytest.raises(place.store.StoreError) as refused:
            opening("ln")
        assert "link" in str(refused.value), str(refused.value)
    assert target.read_bytes() == held
