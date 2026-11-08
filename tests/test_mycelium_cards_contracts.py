"""Contracts of the map's file cards (264, M9, P20 phase 2 stage 2): two sentences a local model writes of a file,
kept only CHECKED -- each identifier and number they give a whole token of the file --, of the public and private
sources only, shown by the map as a model's and never as something to cite, stale once the file has changed; and the
store's schema 3 that holds them.

  * CA1 -- THE SCHEMA: a new store is at 3 with its ``card`` table; a store of schema 2 (its tables as 263 made them,
    files and parts in them) is read as it is -- searched, mapped, no card --, and opened to write moves to 3 in one
    transaction, the table added and nothing else touched (no part cut again); a store at 4 is refused by name.
  * CA2 -- THE PASS: one card per file of the folders of the public and private sources, the secret source never
    read nor asked of, a source of documents never carded; ``limit`` bounds it; the store's lock is free while the
    model writes; a source marked secret while the model writes is passed over, and one marked secret after loses
    its cards at once, a rewrite owed (review 26y-1); the map shows a card as a model's, its names checked, not a
    citation.
  * CA3 -- THE CHECK: a sentence naming what the file does not hold -- a name, a dotted name, a decimal, a path, an
    address -- is dropped and counted (a stop inside a name or a number ends no sentence, review 26y-1); a card whose
    sentences all fail is kept empty -- the map shows nothing -- and the file is not asked again until it changes; a
    model's reasoning block is never kept.
  * CA4 -- A CHANGED FILE LOSES ITS CARD (review 26y-1: a stale card kept the old file's names past the rewrite):
    the index pass that changes a file removes its card, a rewrite owed, so its words leave the store's bytes; the
    next pass writes it again; a card of another text of the file is never shown, and replaced owes a rewrite; a file
    removed takes its card; a file changed on disk since its index, or while the model wrote, is passed over.
  * CA5 -- THE STOP, THE COMMAND, THE TEXTS: the model's first refusal ends the pass, what it wrote kept, said; the
    route's ``cards`` command writes and says its counts, refuses malformed settings and a project's store by name;
    the CHANGELOG says it.
  * CA6 -- MV8 of tests/test_mycelium_vector_contracts.py, superseded (it held the schema at 2): a store of schema 1
    opens at schema 3 and is cut again.
  * CA7 -- MV14 there, superseded: each move from schema 1 (to 2, then to 3) is one transaction of its own, a failed
    one leaving the store as it was; opened again, the store is at 3.
  * CA8 -- MP3 of tests/test_mycelium_place_contracts.py, superseded (its newer schema was 3): reading writes nothing,
    and a store at a newer schema (4), one of schema 1, a database of another kind and an empty file are refused by
    name, each left as it was.
  * CA9 -- MV19 of tests/test_mycelium_vector_contracts.py, superseded (it held the schema at 2): an opener reads
    schema 1, another moves the store first: the first opens it at 3, its column and its card table there once.
CA1-CA7 run the real store, index, cards and map modules on stores in the test's folder through a plain SQLite
opener; no engine is reached (the vector suite's guard): the model is a stand-in asker. CA5's command and CA8 run the
real place with the data folder a test folder, as the place suite does.
"""
import sqlite3
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_mycelium_place_contracts as pl  # noqa: E402
import test_mycelium_vector_contracts as mv  # noqa: E402

from opti_oignon.mycelium import cards  # noqa: E402
from opti_oignon.mycelium import map as mp  # noqa: E402
from opti_oignon.mycelium import store as store_mod  # noqa: E402
from opti_oignon.mycelium.bridge import AskError  # noqa: E402
from opti_oignon.mycelium.index import Indexer  # noqa: E402
from opti_oignon.mycelium.search import Searcher  # noqa: E402
from opti_oignon.mycelium.store import Store, StoreError  # noqa: E402

BUDGET_S = {
    "test_ca1_the_schema": 3.0,
    "test_ca2_the_pass": 3.0,
    "test_ca3_the_check": 3.0,
    "test_ca4_a_changed_file_loses_its_card": 3.0,
    "test_ca5_the_stop_the_command_the_texts": 4.0,
    "test_ca6_a_schema_1_store_opens_at_schema_3_and_is_cut_again": 1.5,
    "test_ca7_each_move_from_schema_1_is_whole": 1.5,
    "test_ca8_reading_writes_nothing": 3.0,
    "test_ca9_two_openers_of_a_schema_1_store_move_it_once": 1.5,
}

# The vector suite's guard: no contract reaches the machine's engine or governor.
no_engine = mv.no_engine
seams = mv.seams
place = pl.place

SETTINGS = cards.CardSettings("small-model", 50, 30.0)
GOOD = "The gate checks a summary with `probe_summary`. It is used when a summary replaces its sources."


class Model:
    """The model, a stand-in: each prompt recorded, its answer by file (``GOOD`` by default), and whether the
    store's lock was free while it answered."""

    def __init__(self, store=None, answers=None, refuse_after=None, during=None):
        self.store, self.answers, self.refuse_after, self.during = store, answers or {}, refuse_after, during
        self.prompts, self.lock_free, self.tries = [], [], 0

    def ask(self, prompt, num_predict, system=None, timeout_s=None):
        self.tries += 1
        if self.during is not None:
            self.during()
        if self.refuse_after is not None and len(self.prompts) >= self.refuse_after:
            raise AskError("small-model held by the governor: busy")
        self.prompts.append(prompt)
        if self.store is not None:
            free = []
            other = threading.Thread(target=lambda: free.append(self.store.lock.acquire(timeout=0) and
                                                                (self.store.lock.release() or True)))
            other.start()
            other.join(5)
            self.lock_free.append(free == [True])
        path = prompt.split("PATH: ", 1)[1].split("\n", 1)[0]
        return self.answers.get(path, GOOD)


def _store(tmp_path, files, *, secret=None, name="s.myc"):
    root = mv.folder(tmp_path, "src", files)
    store = Store(tmp_path / name, lambda p: sqlite3.connect(p, check_same_thread=False))
    Indexer(store).index("oo", root)
    if secret:
        hidden = mv.folder(tmp_path, "vault", secret)
        Indexer(store).index("vault", hidden, sensitivity="secret")
    return store, root


def _cards(store):
    return {path: (text, dropped) for path, text, dropped in store.db.execute(
        "SELECT f.path, c.text, c.dropped FROM card c JOIN file f ON f.id = c.file ORDER BY f.path")}


def test_ca1_the_schema(tmp_path):
    store, root = _store(tmp_path, {"gate.py": mv.GATE, "peels.py": mv.PEELS})
    assert store.meta("schema") == "3" and store_mod.SCHEMA_VERSION == 3
    assert store.db.execute("SELECT name FROM sqlite_master WHERE name = 'card'").fetchone() == ("card",)
    # A store of schema 2, as 263 made it: no card table, its stamp 2; files, parts and their texts in it.
    store.db.execute("DROP TABLE card")
    store.db.execute("UPDATE meta SET value = '2' WHERE key = 'schema'")
    store.db.commit()
    parts = store.db.execute("SELECT id, embed_sha, sha256 FROM part ORDER BY id").fetchall()
    store.close()
    path = tmp_path / "s.myc"
    reader = Store(path, lambda p: sqlite3.connect(f"file:{p}?mode=ro", uri=True), read_only=True)
    try:
        assert Searcher(reader).search("probe_summary", 3, stages=("symbols",)), "a store of schema 2 not searched"
        card = mp.file_card(reader, "oo", "gate.py")
        assert card is not None and card["card"] is None and "card (" not in mp.branch(reader, "oo", "gate.py")
    finally:
        reader.close()
    assert sqlite3.connect(path).execute("SELECT value FROM meta WHERE key = 'schema'").fetchone() == ("2",), (
        "reading moved the store")
    writer = Store(path, sqlite3.connect)
    assert writer.meta("schema") == "3"
    assert writer.db.execute("SELECT name FROM sqlite_master WHERE name = 'card'").fetchone() == ("card",)
    assert writer.db.execute("SELECT id, embed_sha, sha256 FROM part ORDER BY id").fetchall() == parts
    assert Indexer(writer).index("oo", root)["recut"] == 0, "the move to 3 cut the parts again"
    writer.close()

    class FailingStamp:
        def __init__(self, path):
            self.db = sqlite3.connect(path)

        def execute(self, sql, *args):
            if sql.startswith("UPDATE meta SET value = '3'"):
                raise sqlite3.OperationalError("disk I/O error")
            return self.db.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self.db, name)

    again = tmp_path / "again.myc"
    first = Store(again, sqlite3.connect)
    Indexer(first).index("oo", root)
    first.db.execute("DROP TABLE card")
    first.db.execute("UPDATE meta SET value = '2' WHERE key = 'schema'")
    first.db.commit()
    first.close()
    failing = FailingStamp(again)
    with pytest.raises(sqlite3.OperationalError):
        Store(again, lambda _p: failing)
    failing.db.close()
    held = sqlite3.connect(again)
    assert held.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone() == ("2",)
    assert held.execute("SELECT name FROM sqlite_master WHERE name = 'card'").fetchone() is None, (
        "a failed move left its table behind")
    held.close()
    moved = Store(again, sqlite3.connect)
    assert moved.meta("schema") == "3"
    moved.db.execute("UPDATE meta SET value = '4' WHERE key = 'schema'")
    moved.db.commit()
    moved.close()
    for read_only in (False, True):
        with pytest.raises(StoreError) as refused:
            Store(again, sqlite3.connect, read_only=read_only)
        assert "schema 4" in str(refused.value), refused.value


def test_ca2_the_pass(tmp_path):
    store, _root = _store(tmp_path, {"gate.py": mv.GATE, "peels.py": mv.PEELS},
                          secret={"keys.py": "def vault_key():\n    return 'k'\n"})
    model = Model(store)
    counts = cards.write(store, SETTINGS, limit=1, asker=model)
    assert counts["written"] == 1 and len(model.prompts) == 1, counts
    counts = cards.write(store, SETTINGS, asker=model)
    assert counts["written"] == 1 and counts["stopped"] == "" and counts["passed"] == 0, counts
    held = _cards(store)
    assert sorted(held) == ["gate.py", "peels.py"], held
    assert not any("keys.py" in prompt or "vault_key" in prompt for prompt in model.prompts), "a secret file asked"
    assert model.lock_free and all(model.lock_free), "the store's lock was held while the model wrote"
    assert "DEFINES: probe_summary" in model.prompts[0] and "PATH: gate.py" in model.prompts[0], model.prompts[0][:300]
    said = mp.branch(store, "oo", "gate.py")
    assert ("card (a model's, small-model, its names checked against the file; not a citation): The gate checks a "
            "summary with `probe_summary`.") in said, said
    assert cards.write(store, SETTINGS, asker=model)["written"] == 0 and len(model.prompts) == 2, "asked again"
    # Marked secret after: its cards go at once, a rewrite owed.
    store.db.execute("DELETE FROM meta WHERE key = 'rewrite'")
    store.db.commit()
    store.source("oo", store.db.execute("SELECT root FROM source WHERE name = 'oo'").fetchone()[0],
                 sensitivity="secret")
    assert _cards(store) == {} and store.meta("rewrite"), "a secret source kept its cards"
    store.close()
    # Marked secret while the model writes: the files not read yet are passed over, no card kept.
    (tmp_path / "b").mkdir()
    again, root = _store(tmp_path / "b", {"gate.py": mv.GATE, "peels.py": mv.PEELS})
    marked = []

    def secret():
        if not marked:
            marked.append(1)
            again.source("oo", str(root), sensitivity="secret")

    late = Model(during=secret)
    counts = cards.write(again, SETTINGS, asker=late)
    assert counts["written"] == 0 and counts["passed"] == 2 and len(late.prompts) == 1, counts
    assert _cards(again) == {}, _cards(again)
    # A source of documents is never carded, however many it holds.
    Indexer(again).index_documents("docs", {f"d{n}.md": f"# D{n}\n\nText {n}.\n" for n in range(3)},
                                   sensitivity="public")
    again.source("oo", str(root), sensitivity="public")
    counts = cards.write(again, SETTINGS, limit=1, asker=Model())
    assert counts["written"] == 1 and sorted(_cards(again)) == ["gate.py"], (counts, _cards(again))
    again.close()


def test_ca3_the_check(tmp_path):
    store, _root = _store(tmp_path, {"gate.py": mv.GATE, "peels.py": mv.PEELS})
    answers = {"gate.py": ("<think>It defines `probe_summary`. </think>The gate calls `frobnicate_all` first. It "
                           "checks a summary with `probe_summary`."),
               "peels.py": "It wraps `missing_name` around 4096 summaries."}
    model = Model(answers=answers)
    counts = cards.write(store, SETTINGS, asker=model)
    held = _cards(store)
    assert held["gate.py"] == ("It checks a summary with `probe_summary`.", 1), held
    assert held["peels.py"] == ("", 1) and counts["empty"] == 1 and counts["dropped"] == 2, (held, counts)
    assert "card (" not in mp.branch(store, "oo", "peels.py") and mp.file_card(store, "oo", "peels.py")["card"] is None
    assert cards.write(store, SETTINGS, asker=model)["written"] == 0 and len(model.prompts) == 2, (
        "a file whose card was left empty was asked again")
    assert "frobnicate_all" not in str(_cards(store)) and "defines" not in str(_cards(store)), "a reasoning block kept"
    text, dropped = cards.checked("The gate keeps 4096 summaries in `probe_summary`. Its `probe_summary` runs.",
                                  mv.GATE, "gate.py")
    assert dropped == 1 and "4096" not in text, (text, dropped)
    for sentence in ("It calls os.system on each summary.", "It posts with requests.post first.",
                     "It waits 2.5 seconds per probe.", "It reads /etc/passwd before the probes.",
                     "It sends them to https://collector.example/x right away."):
        text, dropped = cards.checked(sentence + " Its `probe_summary` runs.", mv.GATE, "gate.py")
        assert (text, dropped) == ("Its `probe_summary` runs.", 1), (sentence, text, dropped)
    assert cards.checked("It returns True from summary.", mv.GATE, "gate.py") == ("It returns True from summary.", 0)
    text, dropped = cards.checked("One `probe_summary`. Two `probe_summary`. Three `probe_summary`.", mv.GATE, "gate.py")
    assert text == "One `probe_summary`. Two `probe_summary`." and dropped == 0, (text, dropped)
    store.close()


def test_ca4_a_changed_file_loses_its_card(tmp_path):
    store, root = _store(tmp_path, {"gate.py": mv.GATE, "peels.py": mv.PEELS})
    model = Model(answers={"gate.py": "It keeps `probe_summary` zebrafinch."})
    assert cards.write(store, SETTINGS, asker=model)["written"] == 2
    (root / "gate.py").write_text(mv.GATE + "\n\ndef check_more():\n    return 2\n", encoding="utf-8")
    Indexer(store).index("oo", root)
    assert sorted(_cards(store)) == ["peels.py"], "a changed file kept the card of its old text"
    assert "card (" not in mp.branch(store, "oo", "gate.py") and store.meta("rewrite"), "no rewrite owed"
    assert store.rewrite() and b"zebrafinch" not in (tmp_path / "s.myc").read_bytes(), "the old card's words stayed"
    counts = cards.write(store, SETTINGS, asker=model)
    assert counts["written"] == 1 and counts["replaced"] == 0 and len(model.prompts) == 3, counts
    assert "its names checked against the file" in mp.branch(store, "oo", "gate.py")
    # A card of another text of the file (as no pass of this code leaves one): never shown; written again, replaced,
    # a rewrite owed.
    store.db.execute("UPDATE card SET sha256 = 'another' WHERE file = (SELECT id FROM file WHERE path = 'gate.py')")
    store.db.execute("DELETE FROM meta WHERE key = 'rewrite'")
    store.db.commit()
    assert "card (" not in mp.branch(store, "oo", "gate.py") and mp.file_card(store, "oo", "gate.py")["card"] is None
    counts = cards.write(store, SETTINGS, asker=model)
    assert counts["written"] == 1 and counts["replaced"] == 1 and store.meta("rewrite"), counts
    (root / "peels.py").unlink()
    Indexer(store).index("oo", root)
    assert sorted(_cards(store)) == ["gate.py"], "a removed file's card stayed"
    # A file changed on disk since its index: passed over, nothing asked.
    (root / "gate.py").write_text(mv.GATE + "\n\ndef check_again():\n    return 3\n", encoding="utf-8")
    store.db.execute("DELETE FROM card")
    store.db.commit()
    asked = len(model.prompts)
    counts = cards.write(store, SETTINGS, asker=model)
    assert counts["passed"] == 1 and counts["written"] == 0 and len(model.prompts) == asked, counts
    # A file changed and indexed again while the model wrote its card: that card is not kept.
    Indexer(store).index("oo", root)

    def changed():
        (root / "gate.py").write_text(mv.GATE + "\n\ndef check_late():\n    return 4\n", encoding="utf-8")
        Indexer(store).index("oo", root)

    late = Model(during=changed)
    counts = cards.write(store, SETTINGS, asker=late)
    assert counts["passed"] == 1 and counts["written"] == 0 and _cards(store) == {}, (counts, _cards(store))
    store.close()


def test_ca5_the_stop_the_command_the_texts(place, tmp_path, monkeypatch, capsys):
    from opti_oignon.mycelium import route

    # The settings: the package's own, and a malformed section refused by name.
    assert cards.load_settings() == cards.CardSettings("qwen3.5:4b", 200, 60.0)
    config = tmp_path / "mycelium.yaml"
    config.write_text("cards:\n  per_pass: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cards.per_pass"):
        cards.load_settings(config)
    config.write_text("cards:\n  model: small\n  temperature: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        cards.load_settings(config)
    store, _root = _store(tmp_path, {"gate.py": mv.GATE, "peels.py": mv.PEELS, "more.py": mv.HELPER})
    model = Model(refuse_after=1)
    counts = cards.write(store, SETTINGS, asker=model)
    assert counts["written"] == 1 and "held by the governor" in counts["stopped"], counts
    assert model.tries == 2, ("the pass asked on after the model's refusal", model.tries)
    assert len(_cards(store)) == 1, "the card written before the refusal was not kept"
    store.close()
    # The route's command, on the place's own store, the model a stand-in.
    root = mv.folder(tmp_path, "proj", {"gate.py": mv.GATE})
    assert route.index("oo", str(root), vectors=False) == 0
    monkeypatch.setattr(cards, "Asker", lambda model, timeout_s, caller=None: Model())
    monkeypatch.setattr(cards, "load_settings", lambda path=None: SETTINGS)
    capsys.readouterr()
    assert route.run(types.SimpleNamespace(mycelium_action="cards", corpus="oo", source=None, limit=None)) == 0
    out = capsys.readouterr().out
    assert "1 card written into oo by small-model" in out, out
    held = place.mod.open_to_write("oo")
    held.db.execute("UPDATE card SET sha256 = 'another'")      # a card of another text, replaced by the next pass
    held.db.commit()
    held.close()
    assert route.cards("oo") == 0
    out = capsys.readouterr().out
    assert "1 card written" in out and "(1 replaced)" in out and "the replaced cards left its pages" in out, out
    written = place.mod.open_to_read("oo")
    try:
        assert written.meta("rewrite") is None, "the rewrite the replaced card owed was not done"
    finally:
        written.close()
    monkeypatch.setattr(cards, "write", lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")))
    assert route.cards("oo") == 1 and "OperationalError: disk full" in capsys.readouterr().out
    written = place.mod.open_to_read("oo")
    try:
        assert "card (a model's, small-model" in mp.branch(written, "oo", "gate.py")
    finally:
        written.close()
    assert route.cards("project_0123abcd-012") == 1 and "its project keeps it" in capsys.readouterr().out
    assert route.cards("nothing_here") == 1 and "no store for the corpus" in capsys.readouterr().out
    monkeypatch.setattr(cards, "load_settings", lambda path=None: (_ for _ in ()).throw(ValueError("cards.model")))
    assert route.cards("oo") == 1 and "the cards' settings are refused" in capsys.readouterr().out
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "CA1-CA9" in e), "").split())
    assert "mycelium cards" in entry and "never a secret one" in entry and "schema 3" in entry, entry[:300]


def test_ca6_a_schema_1_store_opens_at_schema_3_and_is_cut_again(tmp_path, seams):
    root, path = mv.schema_1_store(tmp_path)
    store = Store(path, sqlite3.connect)
    assert store.meta("schema") == str(store_mod.SCHEMA_VERSION) == "3"
    assert store.db.execute("SELECT name FROM sqlite_master WHERE name = 'card'").fetchone() == ("card",)
    assert store.db.execute("SELECT embed_sha FROM part WHERE id = 1").fetchone() == ("",)
    counts = Indexer(store).index("oo", root)
    assert counts["recut"] == 1
    assert store.db.execute("SELECT COUNT(*) FROM part WHERE embed_sha = ''").fetchone()[0] == 0
    seams(mv.Registry({"enc-a": mv.Engine()}), mv.Governor())
    assert Indexer(store).embed(mv.Embedder(mv.A))["embedded"] > 0 and mv.texts_without_vector(store, mv.A) == 0
    store.close()


def test_ca7_each_move_from_schema_1_is_whole(tmp_path):
    root, path = mv.schema_1_store(tmp_path)
    failing = mv.FailingMove(path)
    with pytest.raises(sqlite3.OperationalError):
        Store(path, lambda _p: failing)
    failing.db.close()
    store = Store(path, sqlite3.connect)
    assert store.meta("schema") == "3" and store.meta("cutter:1") is None
    assert Indexer(store).index("oo", root)["recut"] == 1
    store.close()


def test_ca8_reading_writes_nothing(place, tmp_path):
    mod = place.mod
    root = pl._folder(tmp_path / "src", {"gate.py": pl.GATE})
    writer = mod.open_to_write("oo")
    Indexer(writer).index("oo", root)
    writer.close()
    before = pl._snapshot(place.folder)
    reader = mod.open_to_read("oo")
    assert [s["name"] for s in reader.sources()] == ["oo"]
    reader.generation()
    assert Searcher(reader).search("check_entity", 3, stages=("symbols",))
    with pytest.raises(Exception):
        reader.meta("vectors", 7)
        reader.db.commit()
    reader.close()
    assert pl._snapshot(place.folder) == before, "reading wrote to the store"

    def planted(name, statements):
        db = sqlite3.connect(place.folder / f"{name}.db")
        for statement in statements:
            db.execute(statement)
        db.commit()
        db.close()
        return (place.folder / f"{name}.db").read_bytes()

    cases = {
        "newer": (["CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
                   "INSERT INTO meta VALUES ('schema', '4')"], "schema 4"),
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


def test_ca9_two_openers_of_a_schema_1_store_move_it_once(tmp_path):
    root, path = mv.schema_1_store(tmp_path)
    second = mv.SecondOpener(path)
    store = Store(path, lambda _p: second)
    assert second.done and store.meta("schema") == "3"
    columns = [r[1] for r in store.db.execute("PRAGMA table_info(part)")]
    assert columns.count("embed_sha") == 1
    assert store.db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'card'").fetchone() == (1,)
    assert Indexer(store).index("oo", root)["recut"] == 1
    store.close()
