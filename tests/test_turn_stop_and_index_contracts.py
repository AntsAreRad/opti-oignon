"""Contracts of a chat turn's search stopped at its deadline, and of the Mycelium's index result given by the project
routes (263, BACKLOG 254 b and 253 a). Past a turn's deadline the whole search ran on to its end on its own thread:
its bridge's and its judge's questions still left for the shared model, the question's vector for the encoder, their
answers unused. And what a project's Mycelium store could not be given was said in the log alone.

  * SI1 -- THE DEADLINE STOPS THE CALLS: a turn whose bridge is slow is answered by the lexical stages at its
    deadline; from then on the full search sends nothing more -- the bridge's next question refused, the vector and
    the judging stages left out, each said in its silences --; the question already sent runs to its end, its answer
    unused.
  * SI2 -- WITHIN ITS DEADLINE, UNCHANGED: a turn answered in time asks the bridge both its questions, the encoder its
    vector and the judge its verdicts; nothing of its silences speaks of a deadline; a searcher made without a stop
    (the agent's tool) is never stopped.
  * SI3 -- THE INDEX RESULT GIVEN: ``POST .../files/{id}/index`` gives what the project's Mycelium store could not be
    given (``mycelium``, empty when it took everything), a failed indexation names it in its detail, and
    ``POST .../reindex`` gives it for the whole pass.
  * SI4 -- THE TEXTS: the CHANGELOG says it.
  * SI5 -- THE LAST CHECK (review 26y-1): the bridge's own asker, its question admitted when the deadline passes,
    sends nothing and hands the admission back; admitted in time, it sends.
  * SI6 -- THE VECTOR AFTER THE VECTORS ARE READ (review 26y-1): the deadline passing while the store's vectors are
    read, the question's vector is never asked, said.
The suite runs the real package with the project turn suite's world: the data folder a test folder, the stores'
opener a plain one, a real projects' store; the bridge's model, the encoder and the judge stand-ins, so no contract
reaches the engine.
"""
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_project_turn_contracts as tm  # noqa: E402
from test_project_mycelium_contracts import NOTES  # noqa: E402

BUDGET_S = {
    "test_si1_the_deadline_stops_the_calls": 6.0,
    "test_si2_within_its_deadline_unchanged": 6.0,
    "test_si3_the_index_result_given": 6.0,
    "test_si4_the_texts": 2.0,
    "test_si5_the_last_check": 3.0,
    "test_si6_the_vector_after_the_vectors_are_read": 5.0,
}

# The project turn suite's fixtures.
no_engine = tm.no_engine
owned = tm.owned
world = tm.world
chat = tm.chat

QUESTION = "dans quel fichier est le plan zanzibar_quartz"      # French: the bridge asks its model


class _Judge:
    """The judge, a stand-in shaped as the real one: its asker asked once per judging, a refusal said, not raised."""

    def __init__(self, asked):
        self.asked = asked
        self.settings = types.SimpleNamespace(depth=5)
        self.asker = types.SimpleNamespace(ask=lambda *a, **k: asked.append(("judge", time.monotonic())) or "yes")

    def __call__(self, question, hits, read, silent=None):
        from opti_oignon.mycelium.bridge import AskError

        self.asked.append(("judging", time.monotonic()))
        try:
            self.asker.ask("judge this", 1)
        except AskError as exc:
            silent["rerank"] = str(exc)
        return list(hits)


def _spied(chat, monkeypatch, *, slow=0.0):
    """The bridge's model (slow on its first question when ``slow``), the encoder and the judge, each call recorded
    with its time; the full search's silences kept."""
    from opti_oignon.mycelium import bridge, search

    asked, kept = [], {}

    class Asker:
        def __init__(self, model, timeout_s, caller=None):
            pass

        def ask(self, prompt, num_predict, system=None, timeout_s=None):
            asked.append(("bridge", time.monotonic()))
            if slow and len([a for a in asked if a[0] == "bridge"]) == 1:
                time.sleep(slow)
                asked.append(("bridge answered", time.monotonic()))
            return "where is the zanzibar plan kept"

    class Query(tm._Query):
        def query(self, text):
            asked.append(("encoder", time.monotonic()))
            return super().query(text)

    real = chat.turn._search

    def searched(corpus, query, k, lexical, stop=None):
        found = real(corpus, query, k, lexical, stop=stop)
        if not lexical:
            kept.update(found[1])
        return found

    monkeypatch.setattr(bridge, "Asker", Asker)
    monkeypatch.setattr(search, "Embedder", Query)
    monkeypatch.setattr(chat.turn, "_reranker", lambda: _Judge(asked))
    monkeypatch.setattr(chat.turn, "_search", searched)
    return asked, kept


def _join_turn_threads():
    for thread in threading.enumerate():       # the full search ends here, never in a later contract
        if thread.name == "mycelium-turn":
            thread.join(10)


def test_si1_the_deadline_stops_the_calls(chat, monkeypatch):
    project, _added = tm._indexed(chat, {"notes.md": NOTES.encode()})
    asked, kept = _spied(chat, monkeypatch, slow=0.8)
    monkeypatch.setattr(chat.turn, "DEADLINE_S", 0.3)
    started = time.monotonic()
    ctx = chat.builder.build_context(project.id, QUESTION)
    answered = time.monotonic()
    _join_turn_threads()
    assert answered - started < 0.7 and "lexical stages answered" in ctx.said, (answered - started, ctx.said)
    kinds = [kind for kind, _at in asked]
    assert kinds == ["bridge", "bridge answered"], ("a call left after the deadline", kinds)
    assert asked[1][1] > answered, "the question already sent did not run to its end"
    for stage in ("rewrite", "dense", "rerank"):
        assert "deadline passed" in kept.get(stage, ""), (stage, kept)


def test_si2_within_its_deadline_unchanged(chat, monkeypatch):
    from opti_oignon.mycelium import search

    project, _added = tm._indexed(chat, {"notes.md": NOTES.encode()})
    asked, kept = _spied(chat, monkeypatch)
    ctx = chat.builder.build_context(project.id, QUESTION)
    _join_turn_threads()
    assert ctx.source == "mycelium" and "lexical stages answered" not in ctx.said, ctx.said
    kinds = [kind for kind, _at in asked]
    assert kinds.count("bridge") == 2 and kinds.count("encoder") == 1 and kinds.count("judge") == 1, kinds
    assert kinds.count("judging") == 1, kinds
    assert not any("deadline" in why for why in kept.values()), kept
    store = chat.place.open_to_read(f"project_{project.id}")
    try:
        assert search.Searcher(store).stop is None
    finally:
        store.close()


def test_si3_the_index_result_given(chat, monkeypatch):
    from opti_oignon.api import routes_projects as rp
    from opti_oignon.project_context import IndexResult

    project, added = tm._project(chat, {"notes.md": NOTES.encode(), "scan.pdf": b"%PDF-1.4\x00\x01\x02" * 64})
    monkeypatch.setattr(rp, "project_store", chat.store)
    monkeypatch.setattr(rp, "project_indexer", chat.indexer)
    monkeypatch.setattr(rp, "PROJECTS_AVAILABLE", True)
    monkeypatch.setattr(rp, "PROJECT_CONTEXT_AVAILABLE", True)
    left = ["scan.pdf: not given"]
    monkeypatch.setattr(chat.indexer, "index_mycelium", lambda project_id, vectors=True: list(left))
    ok = {"value": True}

    def knowledge_base(project_id, file_id):
        return IndexResult(file_id=file_id, filename="notes.md", success=ok["value"], chunk_count=2,
                           error="" if ok["value"] else "no embeddings")

    monkeypatch.setattr(chat.indexer, "_index_file", knowledge_base)
    answer = rp.index_project_file(project.id, added["notes.md"].id)
    assert answer["mycelium"] == left and answer["chunk_count"] == 2, answer
    left.clear()
    assert rp.index_project_file(project.id, added["notes.md"].id)["mycelium"] == [], "a store that took everything"
    left.append("scan.pdf: not given")
    ok["value"] = False
    with pytest.raises(rp.HTTPException) as refused:
        rp.index_project_file(project.id, added["notes.md"].id)
    assert refused.value.status_code == 422 and "scan.pdf: not given" in refused.value.detail, refused.value.detail
    ok["value"] = True
    whole = rp.reindex_project(project.id)
    assert whole["mycelium"] == left and whole["total_files"] == 2, whole
    bare = chat.store.create_project("bare")
    nothing = rp.reindex_project(bare.id)
    assert nothing["total_files"] == 0 and nothing["mycelium"] == left, ("a project with no file lost it", nothing)


def test_si4_the_texts():
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "SI1-SI6" in e), "").split())
    assert "deadline" in entry and "mycelium" in entry and "in flight" in entry, entry[:300]


def test_si5_the_last_check(monkeypatch):
    import test_mycelium_bridge_contracts as mb

    from opti_oignon.mycelium import bridge, turn, vectors
    from opti_oignon.mycelium.bridge import AskError

    stop = threading.Event()

    class Late(mb.Governor):
        """A governor whose admission takes until the turn's deadline has passed."""

        def __init__(self):
            super().__init__()
            self.handed = []

        def admit(self, model, requested_ctx=None, caller="chat", **kw):
            stop.set()
            decision = super().admit(model, requested_ctx, caller, **kw)
            decision.ticket_id = "t-1"
            return decision

        def end_pending_load(self, ticket):
            self.handed.append(ticket)

    model, governor = mb.Model(), Late()
    monkeypatch.setattr(vectors, "_registry", lambda: mb.Registry({"m-4b": model}))
    monkeypatch.setattr(vectors, "_governor", lambda: governor)
    stopping = turn._Stopping(bridge.Asker("m-4b", 5.0), stop)
    assert stopping.passes_stop
    with pytest.raises(AskError, match="deadline"):
        stopping.ask("You are a professional French translator", 8)
    assert model.calls == [] and governor.handed == ["t-1"], (model.calls, governor.handed)
    stop.clear()
    monkeypatch.setattr(vectors, "_governor", lambda: mb.Governor())
    assert stopping.ask("You are a professional French translator", 8)
    assert len(model.calls) == 1, model.calls
    mb.bridge_module._hung.clear()      # as the bridge suite does after each contract


def test_si6_the_vector_after_the_vectors_are_read(chat, monkeypatch):
    from opti_oignon.mycelium import search

    project, _added = tm._indexed(chat, {"notes.md": NOTES.encode()})
    asked, kept = _spied(chat, monkeypatch)
    real_matrix = search.Searcher._matrix

    def slow_matrix(self):
        time.sleep(0.6)
        return real_matrix(self)

    monkeypatch.setattr(search.Searcher, "_matrix", slow_matrix)
    monkeypatch.setattr(chat.turn, "DEADLINE_S", 0.3)
    ctx = chat.builder.build_context(project.id, QUESTION)
    _join_turn_threads()
    assert "lexical stages answered" in ctx.said, ctx.said
    assert "encoder" not in [kind for kind, _at in asked], ("the vector was asked after the deadline", asked)
    assert "deadline passed" in kept.get("dense", ""), kept
