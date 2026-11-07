"""Contracts of the Mycelium's search as a tool and a citation gate will ask it: a citation read through the store,
the silences each search gives back, the vectors' matrix one store holds for every searcher, and the bridge asked
cold within its own deadline.

  * MQ1 -- A CITATION IS READ THROUGH THE STORE: the root is the one the store holds for the source the citation
    names, the path one of that source's files, the source one the searcher may read, the lines whole numbers in
    order; a citation naming another root, a file the store does not hold (a secret file the walk skipped), a path
    out of the root, an unknown or secret source, lines out of order, no sha256 or a name no store can hold is read
    as nothing and verified as unknown; lines past the file's end are no text of it.
  * MQ2 -- EACH SEARCH KEEPS ITS OWN SILENCES: two searches on one searcher, held inside each other, each give back
    the silences of its own stages, and so does a search run inside another on one thread; ``search`` still gives
    the passages and keeps the last silences.
  * MQ3 -- ONE MATRIX PER STORE: two searchers of one store read its vectors once, whatever the order of their
    sensitivities; a pass that changes the vectors makes the next search read them again; a searcher of other
    sensitivities reads a matrix of its own, never one holding parts it may not read; a pass of the process holding
    the store keeps a vector search waiting no longer than its timeout.
  * MQ4 -- THE BRIDGE ASKED COLD ANSWERS WITHIN ITS DEADLINE: the engine's module is loaded before the call's deadline
    is set, so a slow first import does not spend the call's own time.
"""
import hashlib
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_mycelium_place_contracts import GATE, _folder  # noqa: E402
from test_mycelium_vector_contracts import A, Engine, Governor, Registry  # noqa: E402

BUDGET_S = {
    "test_mq1_a_citation_is_read_through_the_store": 2.0,
    "test_mq2_each_search_keeps_its_own_silences": 2.0,
    "test_mq3_one_matrix_per_store": 2.0,
    "test_mq4_the_bridge_asked_cold_answers_within_its_deadline": 3.0,
}

NOTES = "# Notes\n\nThe zanzibar inventory of the gate.\n"


@pytest.fixture(autouse=True)
def no_engine(monkeypatch):
    """No contract reaches the machine's engine or governor until it installs its stand-ins."""
    from opti_oignon.mycelium import vectors

    def refused():
        raise RuntimeError("a contract asked the real inference registry")

    monkeypatch.setattr(vectors, "_registry", refused)
    monkeypatch.setattr(vectors, "_governor", lambda: None)


def _sha(lines):
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def test_mq1_a_citation_is_read_through_the_store(tmp_path):
    """MQ1 -- a folder (holding a ``.env`` the walk skips), a secret source and a source of documents in one store;
    a searcher of public and private sources. A citation the search gave is read and fresh. The same citation naming
    another root, where a file of that path says something else, is read from the store's root. A citation of the
    skipped ``.env``, of a path out of the root, of an unknown source, of the secret source, or with lines that are
    no whole numbers in order, is read as nothing and verified as unknown -- though the file is there to read. A
    document is read through its reader, an unknown one is unknown. A citation's own root or path read -> RED."""
    from opti_oignon.mycelium.index import Indexer
    from opti_oignon.mycelium.search import Searcher
    from opti_oignon.mycelium.store import Store

    root = _folder(tmp_path / "src", {"gate.py": GATE, "notes.md": NOTES, ".env": "TOKEN=hunter2\n"})
    vault = _folder(tmp_path / "vault", {"keys.md": "# Keys\n\nThe quokka plan.\n"})
    store = Store(tmp_path / "s.db", sqlite3.connect)
    indexer = Indexer(store)
    indexer.index("oo", root)
    indexer.index("vault", vault, sensitivity="secret")
    indexer.index_documents("cellar", {"n.md": "# Cellar\n\nThe wombat ledger.\n"})
    searcher = Searcher(store, readers=indexer.readers)
    [hit] = searcher.search("check_entity", 1, stages=("symbols",))
    assert hit["path"] == "gate.py" and searcher.read(hit) and searcher.verify(hit) == "fresh"

    elsewhere = _folder(tmp_path / "elsewhere", {"gate.py": GATE.replace("Check", "Forge")})
    assert searcher.read(dict(hit, root=str(elsewhere))) == searcher.read(hit), "the citation's own root was read"
    assert searcher.verify(dict(hit, root=str(elsewhere))) == "fresh"

    env = ["TOKEN=hunter2"]
    secret = Searcher(store, allow=("public", "private", "secret")).search("quokka plan", 1, stages=("lexical",))[0]
    assert secret["source"] == "vault"
    forged = {
        "a file the walk skipped": dict(hit, path=".env", first=1, last=1, sha256=_sha(env)),
        "a path out of the root": dict(hit, path="../vault/keys.md", first=1, last=3),
        "an unknown source": dict(hit, source="ghost"),
        "the secret source": secret,
        "line zero": dict(hit, first=0),
        "lines out of order": dict(hit, first=hit["last"] + 1),
        "lines that are text": dict(hit, first="1"),
        "lines that are a flag": dict(hit, first=True, last=True),
        "no sha256 (review 1)": {key: value for key, value in hit.items() if key != "sha256"},
        "a lone surrogate in the path (review 1)": dict(hit, path="gate" + chr(0xD800) + ".py"),
        "a lone surrogate in the source (review 1)": dict(hit, source="o" + chr(0xD800)),
    }
    for what, citation in forged.items():
        assert searcher.read(citation) is None, what
        assert searcher.verify(citation) == "unknown", what
    assert (root / ".env").read_text(encoding="utf-8") == "TOKEN=hunter2\n"
    past = dict(hit, first=1_000_000, last=1_000_000, sha256=_sha([""]))
    assert searcher.read(past) is None and searcher.verify(past) == "modified", "lines the file does not hold"

    [doc] = searcher.search("wombat ledger", 1, stages=("lexical",))
    assert doc["source"] == "cellar" and searcher.read(doc) and searcher.verify(doc) == "fresh"
    assert searcher.verify(dict(doc, path="other.md")) == "unknown"
    store.close()


def test_mq2_each_search_keeps_its_own_silences(tmp_path):
    """MQ2 -- one searcher, no encoder active, no reranker: a search of the symbols and the vectors and a search of
    the full text that asks for the reranker, held inside each other at the question's words (both stand there
    before either stage runs): each gives back its own silences, the vectors' for the first, the reranker's for the
    second. ``search`` still gives the passages and keeps its silences on the searcher. Silences of one search given
    to the other -> RED."""
    from opti_oignon.mycelium.index import Indexer
    from opti_oignon.mycelium.search import Searcher
    from opti_oignon.mycelium.store import Store

    # Shared between threads, as the platform's opener makes it (check_same_thread off).
    store = Store(tmp_path / "s.db", lambda path: sqlite3.connect(path, check_same_thread=False))
    Indexer(store).index("oo", _folder(tmp_path / "src", {"gate.py": GATE}))
    both = threading.Barrier(2, timeout=5)

    def words(query):
        both.wait()
        return query

    searcher = Searcher(store, lexical_query=words)
    found = {}

    def first():
        found["first"] = searcher.find("check_entity", 3, stages=("symbols", "dense"))

    def second():
        found["second"] = searcher.find("check entity", 3, stages=("lexical",), rerank=True)

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert set(found) == {"first", "second"}, found
    assert set(found["first"].silent) == {"dense"}, found["first"].silent
    assert set(found["second"].silent) == {"rerank"}, found["second"].silent
    assert found["first"].hits and found["first"].hits[0]["path"] == "gate.py"
    alone = Searcher(store)
    hits = alone.search("check_entity", 3, stages=("symbols", "dense"))
    assert isinstance(hits, list) and hits and set(alone.silent) == {"dense"}
    inner = {}

    def nested_words(query):
        if not inner:
            inner["asked"] = True       # once: the inner search asks for the words too
            inner["found"] = nested.find("check entity", 3, stages=("lexical",), rerank=True)
        return query

    nested = Searcher(store, lexical_query=nested_words)
    outer = nested.find("check_entity", 3, stages=("symbols", "dense"))
    assert set(inner["found"].silent) == {"rerank"}, inner["found"].silent
    assert set(outer.silent) == {"dense"}, ("a search run inside another lost the outer one's silences", outer.silent)
    store.close()


class Counting:
    """A connection that counts the statements reading the vectors' matrix."""

    def __init__(self, db):
        self.db, self.reads = db, 0

    def execute(self, sql, *args):
        if "JOIN vector v" in sql and sql.lstrip().startswith("SELECT p.id, v.data"):
            self.reads += 1
        return self.db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.db, name)


def test_mq3_one_matrix_per_store(tmp_path, monkeypatch):
    """MQ3 -- a store whose vectors are read under an encoder (a stand-in engine on this machine): two searchers of
    the store, each searching the vectors, read the matrix once between them; a pass that adds a file and its vector
    makes the next search read it again, once; a searcher of public sources only reads its own -- which holds none of
    the private parts, and says so. A matrix read again by every searcher, or served across sensitivities -> RED."""
    from opti_oignon.mycelium import vectors
    from opti_oignon.mycelium.index import Indexer
    from opti_oignon.mycelium.search import Searcher
    from opti_oignon.mycelium.store import Store
    from opti_oignon.mycelium.vectors import Embedder

    engine = Engine()
    monkeypatch.setattr(vectors, "_registry", lambda: Registry({A.model: engine}))
    monkeypatch.setattr(vectors, "_governor", lambda: Governor())
    held = {}

    def connect(path):
        held["db"] = Counting(sqlite3.connect(path, check_same_thread=False))
        return held["db"]

    root = _folder(tmp_path / "src", {"gate.py": GATE, "notes.md": NOTES})
    store = Store(tmp_path / "s.db", connect)
    indexer = Indexer(store)
    indexer.index("oo", root)
    assert indexer.embed(Embedder(A)).get("activated"), "the stand-in encoder was not made the one read"
    db = held["db"]
    first, second = Searcher(store), Searcher(store)
    assert first.find("the gate checks entities", 3, stages=("dense",)).hits
    assert second.find("zanzibar inventory", 3, stages=("dense",)).hits
    assert db.reads == 1, f"the matrix was read {db.reads} times for two searchers of one store"
    (root / "more.md").write_text("# More\n\nA new note.\n", encoding="utf-8")
    indexer.index("oo", root)
    indexer.embed(Embedder(A))
    assert second.find("a new note", 3, stages=("dense",)).hits
    assert first.find("a new note", 3, stages=("dense",)).hits
    assert db.reads == 2, f"after the vectors changed, the matrix was read {db.reads - 1} more times, not once"
    for same in (("private", "public"), ("public", "private", "public")):
        assert Searcher(store, allow=same).find("the gate checks entities", 3, stages=("dense",)).hits
    assert db.reads == 2, f"the same sensitivities in another order read the matrix again ({db.reads})"
    public = Searcher(store, allow=("public",))
    found = public.find("the gate checks entities", 3, stages=("dense",))
    assert found.hits == [] and "dense" in found.silent, found
    assert db.reads == 3
    # Review 1: a pass of this process holds the store; a vector search waits no longer than its own timeout.
    holding, release = threading.Event(), threading.Event()

    def a_pass():
        with store.lock:
            holding.set()
            release.wait(5)

    passing = threading.Thread(target=a_pass)
    passing.start()
    holding.wait(5)
    started = time.monotonic()
    try:
        found = Searcher(store, timeout_s=0.2).find("the gate checks entities", 3, stages=("dense",))
    finally:
        took = time.monotonic() - started
        release.set()
        passing.join(5)
    assert took < 1.0 and found.hits == [] and "pass holds the store" in found.silent.get("dense", ""), (took, found)
    store.close()


class Talker:
    """A stand-in engine on this machine that answers at once."""

    name = "ollama"
    forwards_requests = False

    def endpoint(self):
        return "http://127.0.0.1:11434"

    def generate(self, model, messages, options=None, think=None, **_kw):
        return types.SimpleNamespace(content="the gate checks entities")


def test_mq4_the_bridge_asked_cold_answers_within_its_deadline(monkeypatch):
    """MQ4 -- the first import of the engine's module takes 0.6 s (a cold process, a slow disk); the bridge's model,
    a stand-in on this machine that answers at once, is asked with 0.4 s for the call: the answer comes, the import
    paid before the call's deadline was set. A first call that pays the import on its own time ('no time left') ->
    RED."""
    from opti_oignon.mycelium import bridge, vectors

    real = vectors.importlib.import_module
    slow = {"paid": False}

    def import_module(name, *args):
        if name == "opti_oignon.inference_backend" and not slow["paid"]:
            slow["paid"] = True
            time.sleep(0.6)
        return real(name, *args)

    monkeypatch.setattr(vectors, "importlib", types.SimpleNamespace(import_module=import_module))
    monkeypatch.setattr(vectors, "_registry", lambda: Registry({"qwen-stand-in": Talker()}))
    asker = bridge.Asker("qwen-stand-in", 0.4)
    assert asker.ask("Translate: la porte verifie les entites", 32) == "the gate checks entities"
    assert slow["paid"]
