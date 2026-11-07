"""Contracts of the chat's turn reading a project's Mycelium store (M4b): the project context of a turn comes from the
project's own store (``mycelium.turn``), the knowledge base serving only a project the store does not cover, and
nothing in silence in place of a store that cannot be read.

  * TM1 -- THE TURN READS THE PROJECT'S STORE: a project indexed into its store gets its turn's context from it --
    the passages under the names the files were added under, their lines and their state (a file changed since, said
    modified; a cut passage keeping its lines), the knowledge base never asked --, and the optimizer's project zone is
    labelled with that source; a project with no store gets the knowledge base, as before.
  * TM2 -- A PROJECT THE STORE DOES NOT COVER: one that holds a file its store does not (a binary, a PDF) is served
    by the knowledge base, and the turn says why; a secret's name does not count.
  * TM3 -- A STORE THAT CANNOT BE READ: no passage, the knowledge base never asked in its place; said in the turn and
    the optimizer's zone, and once in the log for the project and the reason.
  * TM4 -- THE SENSITIVITIES: a source marked secret is never read; public and private are.
  * TM5 -- THE DEADLINES: the question's vector waits VECTOR_S, 5 s, at most (the searcher's own timeout); a whole
    search that passes DEADLINE_S, 10 s, is answered by the lexical stages alone, without waiting for it, and the turn
    says so.
  * TM6 -- THE BRIDGE KEEPS NO QUESTION: each turn's search makes its own bridge, so a question asked again is
    translated again, and nothing of a conversation's questions stays in the process once its turn is answered.
  * TM7 -- THE UNIFIED LAYER: its project source reads the project's store under the same rule.
  * TM8 -- A STORE WHOSE SEARCH FAILS (review 254-1): the full search failing, or the lexical search past the
    deadline failing, and the unified layer over such a store: no passage, the knowledge base never asked, said.
The suite runs the real package (the data folder a test folder, the stores' opener a plain one, the projects' store a
real one over that folder); the bridge's model, the encoder and the judge are stand-ins: no contract reaches the
engine.
"""
import logging
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_mycelium_agent_contracts as my  # noqa: E402
import test_project_mycelium_contracts as pmc  # noqa: E402
from test_project_mycelium_contracts import BUDGET, NOTES, _project  # noqa: E402

# The fixtures of the project store suite: no engine reached, a test data folder, a real projects' store.
no_engine = my.no_engine
owned = my.owned
world = pmc.world

BUDGET_S = {
    "test_tm1_the_turn_reads_the_projects_store": 6.0,
    "test_tm2_a_project_the_store_does_not_cover": 6.0,
    "test_tm3_a_store_that_cannot_be_read": 6.0,
    "test_tm8_a_store_whose_search_fails": 8.0,
    "test_tm4_the_sensitivities": 6.0,
    "test_tm5_the_deadlines": 6.0,
    "test_tm6_the_bridge_keeps_no_question": 6.0,
    "test_tm7_the_unified_layer": 6.0,
}


class _Asker:
    """The bridge's model, a stand-in: each prompt counted, a fixed English answer."""

    calls = []

    def __init__(self, model, timeout_s, caller=None):
        pass

    def ask(self, prompt, num_predict, system=None, timeout_s=None):
        _Asker.calls.append(prompt[:40])
        return "where is the zanzibar plan kept"


class _Query:
    """The question's encoder, a stand-in: one unit vector, the engine never reached."""

    def __init__(self, encoder, timeout_s=None):
        self.encoder, self.timeout_s = encoder, timeout_s

    def query(self, text):
        return [0.25 / 0.9354, 0.5 / 0.9354, 0.75 / 0.9354]


@pytest.fixture
def chat(world, monkeypatch):
    """The world of the project store suite, with the chat's stand-ins: the bridge's model, the question's encoder,
    the judge off, and a context builder whose knowledge base is a spy."""
    from opti_oignon.mycelium import bridge, search, turn

    monkeypatch.setattr(bridge, "Asker", _Asker)
    monkeypatch.setattr(search, "Embedder", _Query)
    monkeypatch.setattr(turn, "_reranker", lambda: None)
    _Asker.calls = []
    turn._said.clear()
    builder = world.pc.ProjectContextBuilder(store=world.store)
    asked = []

    def knowledge_base(project_id, query, *args, **kwargs):
        asked.append(project_id)
        return [world.pc.RetrievedChunk(content="from the knowledge base", source_file="kb.md")]

    monkeypatch.setattr(builder, "retrieve_chunks", knowledge_base)
    world.builder, world.asked, world.turn = builder, asked, turn
    yield world
    turn._said.clear()


def _indexed(world, files, name="p"):
    project, added = _project(world, files, name=name)
    assert world.indexer.index_mycelium(project.id) == []
    return project, added


def test_tm1_the_turn_reads_the_projects_store(chat):
    """TM1 -- a project indexed: its turn's context holds the store's passages, under the files' own names and their
    lines, and the knowledge base is never asked; a project with no store is served by the knowledge base. A turn
    still served by the knowledge base, or a passage under a stored name -> RED."""
    project, added = _indexed(chat, {"budget.py": BUDGET.encode(), "notes.md": NOTES.encode()})
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    assert ctx.source == "mycelium" and chat.asked == [], (ctx.source, chat.asked)
    assert "[notes.md (lines " in ctx.context_text and "zanzibar_quartz" in ctx.context_text, ctx.context_text
    stored = Path(added["notes.md"].file_path).name
    assert stored not in ctx.context_text and ctx.chunks_used >= 1
    from opti_oignon.context_optimizer import ContextOptimizer

    holder = types.SimpleNamespace(_project_builder=chat.builder, _estimate_tokens=lambda text, model: len(text) // 4)
    _text, zone = ContextOptimizer._inject_project_context(holder, project_id=project.id, query="zanzibar_quartz plan",
                                                           budget_tokens=4096, model="m")
    assert zone.strategy == "mycelium", (zone.strategy, zone.detail)
    # A file changed since it was indexed: its passage is said modified; a cut passage keeps its lines and state.
    path = Path(added["notes.md"].file_path)
    path.write_text(NOTES + "A line written after the index.\n", encoding="utf-8")
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    assert "[notes.md (lines " in ctx.context_text and ", modified)]" in ctx.context_text, ctx.context_text
    long_text = "".join(f"Line {n} of the zanzibar_quartz plan keeps its draft number {n}.\n" for n in range(1, 81))
    longer, _ = _indexed(chat, {"plan.md": long_text.encode()}, name="long")
    cut = chat.builder.build_context(longer.id, "zanzibar_quartz plan", budget_tokens=200)
    assert "[plan.md (lines " in cut.context_text and ", cut)]" in cut.context_text, cut.context_text[:300]
    bare = chat.store.create_project("bare")
    ctx = chat.builder.build_context(bare.id, "anything")
    assert ctx.source == "rag" and chat.asked == [bare.id] and "from the knowledge base" in ctx.context_text


def test_tm2_a_project_the_store_does_not_cover(chat):
    """TM2 -- a project whose rows name a binary file its store does not hold: the knowledge base serves its turn,
    and the turn says why; a file named as a secret's does not count against the store. A project half served by its
    store, or one sent to the knowledge base by a secret's name -> RED."""
    project, _added = _indexed(chat, {"notes.md": NOTES.encode(), "scan.pdf": b"%PDF-1.4\x00\x01\x02" * 64})
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    assert ctx.source == "rag" and chat.asked == [project.id] and "knowledge base serves it" in ctx.said, ctx.said
    secret, _ = _indexed(chat, {"notes.md": NOTES.encode(), ".env": b"API_KEY=s3cr3t\n"}, name="secret")
    ctx = chat.builder.build_context(secret.id, "zanzibar_quartz plan")
    assert ctx.source == "mycelium" and chat.asked == [project.id], (ctx.source, chat.asked)


def test_tm3_a_store_that_cannot_be_read(chat, caplog):
    """TM3 -- a project's store whose bytes are no store: two turns get no passage and the knowledge base is never
    asked; each turn says it, the log once. A turn served by the knowledge base in silence, or a log line at every
    turn -> RED."""
    project, _added = _indexed(chat, {"notes.md": NOTES.encode()})
    path = chat.place.folder() / f"project_{project.id}.db"
    path.write_bytes(b"not a store at all " * 512)
    with caplog.at_level(logging.WARNING, logger="opti_oignon.mycelium.turn"):
        first = chat.builder.build_context(project.id, "zanzibar_quartz plan")
        second = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    for ctx in (first, second):
        assert ctx.chunks_used == 0 and "cannot be read" in ctx.said and ctx.source == "", (ctx.source, ctx.said)
    assert chat.asked == [], chat.asked
    said = [r for r in caplog.records if "cannot be read" in r.getMessage()]
    assert len(said) == 1, [r.getMessage() for r in caplog.records]
    from opti_oignon.context_optimizer import ContextOptimizer

    holder = types.SimpleNamespace(_project_builder=chat.builder, _estimate_tokens=lambda text, model: len(text) // 4)
    _text, zone = ContextOptimizer._inject_project_context(holder, project_id=project.id, query="zanzibar_quartz plan",
                                                           budget_tokens=4096, model="m")
    assert zone.strategy == "unreadable" and "cannot be read" in zone.detail, (zone.strategy, zone.detail)


def _failing_search(chat, monkeypatch, *, full, lexical, slow=0.0):
    """The searcher's find failing (or slow) as asked: the full search, the lexical one."""
    from opti_oignon.mycelium import search

    real = search.Searcher.find

    def find(self, query, k=10, stages=("symbols", "lexical", "dense"), rerank=False):
        if "dense" in stages:
            time.sleep(slow)
            if full:
                raise sqlite3.DatabaseError("database disk image is malformed")
        elif lexical:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return real(self, query, k, stages, rerank)

    monkeypatch.setattr(search.Searcher, "find", find)


def test_tm8_a_store_whose_search_fails(chat, monkeypatch, caplog):
    """TM8 (review 254-1) -- a store that opens but whose search fails, in the full search, or in the lexical search
    past the deadline, and the unified layer over such a store: no passage, the knowledge base never asked, said;
    each reason logged once. A failing search answered by the knowledge base -> RED."""
    from opti_oignon import unified_retrieval

    project, _added = _indexed(chat, {"notes.md": NOTES.encode()})
    with monkeypatch.context() as patched:
        _failing_search(chat, patched, full=True, lexical=False)
        ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
        assert ctx.chunks_used == 0 and "cannot be read" in ctx.said and chat.asked == [], (ctx.said, chat.asked)
        patched.setattr(chat.pj, "project_store", chat.store)
        engine = []
        import opti_oignon.rag_hybrid_search as hybrid

        patched.setattr(hybrid, "get_hybrid_engine", lambda: engine.append(1))
        assert unified_retrieval._default_sources("c1", project.id)["rag"]("zanzibar_quartz plan", 5) == []
        assert engine == [], engine
    with monkeypatch.context() as patched:
        _failing_search(chat, patched, full=False, lexical=True, slow=1.0)
        patched.setattr(chat.turn, "DEADLINE_S", 0.2)
        with caplog.at_level(logging.WARNING, logger="opti_oignon.mycelium.turn"):
            ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
        assert ctx.chunks_used == 0 and "cannot be read" in ctx.said and chat.asked == [], (ctx.said, chat.asked)
        for thread in threading.enumerate():
            if thread.name == "mycelium-turn":
                thread.join(10)


def test_tm4_the_sensitivities(chat):
    """TM4 -- the project's source marked secret: the turn reads none of it; marked public, then private, it is read.
    A secret source read, or a private one left out -> RED."""
    project, _added = _indexed(chat, {"notes.md": NOTES.encode()})
    for sensitivity, read in (("secret", False), ("public", True), ("private", True)):
        store = chat.place.open_to_write(f"project_{project.id}", make=False)
        try:
            source = next(s for s in store.sources() if s["name"] == "files")
            store.source("files", source["root"], source["kind"], sensitivity)
        finally:
            store.close()
        ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
        assert ("zanzibar_quartz" in ctx.context_text) is read, (sensitivity, ctx.context_text)


def test_tm5_the_deadlines(chat, monkeypatch):
    """TM5 -- every searcher of a turn waits VECTOR_S for the question's vector; a whole search slower than
    DEADLINE_S (here cut to 0.3 s) is answered by the lexical stages, well before the slow one ends, and the turn says
    so. A vector waited past its bound, or a turn that waits for the slow search -> RED."""
    from opti_oignon.mycelium import search

    assert (chat.turn.VECTOR_S, chat.turn.DEADLINE_S) == (5.0, 10.0), (chat.turn.VECTOR_S, chat.turn.DEADLINE_S)
    project, _added = _indexed(chat, {"notes.md": NOTES.encode()})
    real_init, real_find, timeouts = search.Searcher.__init__, search.Searcher.find, []

    def init(self, *args, **kwargs):
        timeouts.append(kwargs.get("timeout_s"))
        real_init(self, *args, **kwargs)

    def find(self, query, k=10, stages=("symbols", "lexical", "dense"), rerank=False):
        if "dense" in stages:
            time.sleep(2.0)         # the whole search, slow
        return real_find(self, query, k, stages, rerank)

    monkeypatch.setattr(search.Searcher, "__init__", init)
    monkeypatch.setattr(search.Searcher, "find", find)
    monkeypatch.setattr(chat.turn, "DEADLINE_S", 0.3)
    started = time.perf_counter()
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    took = time.perf_counter() - started
    assert took < 1.5, took
    assert ctx.source == "mycelium" and "zanzibar_quartz" in ctx.context_text, ctx.context_text
    assert "lexical stages answered" in ctx.said, ctx.said
    assert timeouts and all(t == chat.turn.VECTOR_S for t in timeouts), timeouts
    for thread in threading.enumerate():       # the slow search ends here, never in a later contract
        if thread.name == "mycelium-turn":
            thread.join(10)


def test_tm6_the_bridge_keeps_no_question(chat):
    """TM6 -- a French question asked in three turns: the bridge's model asked anew at each (translated and rewritten),
    no bridge kept by the turn's module between them. A question kept from one turn to the next -> RED."""
    from opti_oignon.mycelium import bridge

    project, _added = _indexed(chat, {"notes.md": NOTES.encode()})
    question = "o\u00f9 est le plan zanzibar et qui le garde"
    chat.builder.build_context(project.id, question)
    once = len(_Asker.calls)
    assert once >= 1, _Asker.calls
    chat.builder.build_context(project.id, question)
    chat.builder.build_context(project.id, question)
    assert len(_Asker.calls) == 3 * once, _Asker.calls
    kept = [name for name, value in vars(chat.turn).items() if isinstance(value, bridge.Bridge)
            or (isinstance(value, dict) and any(isinstance(v, bridge.Bridge) for v in value.values()))]
    assert kept == [], kept


def test_tm7_the_unified_layer(chat, monkeypatch):
    """TM7 -- the unified layer's project source, for a project indexed into its store: the store's passages, named
    by the files' own names and lines; for a project with no store, the knowledge base's engine. A unified layer that
    still reads the knowledge base for an indexed project -> RED."""
    from opti_oignon import unified_retrieval

    project, _added = _indexed(chat, {"notes.md": NOTES.encode()})
    monkeypatch.setattr(chat.pj, "project_store", chat.store)
    engine = []

    class Engine:
        def search(self, query, **kwargs):
            engine.append(kwargs.get("collection"))
            return type("Response", (), {"results": []})()

    import opti_oignon.rag_hybrid_search as hybrid

    monkeypatch.setattr(hybrid, "get_hybrid_engine", lambda: Engine())
    items = unified_retrieval._default_sources("c1", project.id)["rag"]("zanzibar_quartz plan", 5)
    assert items and all(i.metadata.get("store") == "mycelium" for i in items), items
    assert any(i.origin.startswith("notes.md (lines ") for i in items) and engine == [], (items, engine)
    Path(_added["notes.md"].file_path).write_text(NOTES + "Changed.\n", encoding="utf-8")
    items = unified_retrieval._default_sources("c1", project.id)["rag"]("zanzibar_quartz plan", 5)
    assert any(i.origin.endswith(", modified)") for i in items), [i.origin for i in items]
    bare = chat.store.create_project("bare")
    assert unified_retrieval._default_sources("c1", bare.id)["rag"]("anything", 5) == []
    assert engine == [f"project_{bare.id}"], engine
