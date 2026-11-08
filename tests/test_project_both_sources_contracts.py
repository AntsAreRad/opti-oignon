"""Contracts of a project served by its Mycelium store and the knowledge base together (262, BACKLOG 254 c): one file
of a project its store does not hold (a PDF, a file not UTF-8 or larger than the index reads) sent the whole project to
the knowledge base, for good. The store now serves the files it holds and the knowledge base the others, said.

  * BS1 -- BOTH: a project whose store holds some of its files: the store's passages for those, the knowledge base's
    chunks of the others and of those alone (never a second copy of a file the store serves), the source said
    ``mycelium+rag`` with the files the knowledge base serves named, in the turn and in the optimizer's zone.
  * BS2 -- TM2 of tests/test_project_turn_contracts.py, superseded (it held such a project wholly to the knowledge
    base): a store that holds none of the files the rows name gives the knowledge base the whole project, said; a
    file named as a secret's counts on neither side.
  * BS3 -- THE UNIFIED LAYER: its project source reads the same way: the store's passages, then the knowledge base's
    results for the files the store does not hold, and for those alone, the knowledge base asked wider so that the
    chunks of held files ranked first do not crowd them out.
  * BS4 -- THE OTHER RULES HOLD IN BOTH: a store that cannot be read gives nothing, never the knowledge base in its
    place; a search past the deadline answers by the lexical stages, and the files the knowledge base serves are
    still named.
  * BS5 -- THE TEXTS: the CHANGELOG says it.
The suite runs the real package with the project turn suite's world: the data folder a test folder, the stores'
opener a plain one, a real projects' store; the bridge, the encoder and the knowledge base stand-ins.
"""
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_project_turn_contracts as tm  # noqa: E402
from test_project_mycelium_contracts import NOTES  # noqa: E402

BUDGET_S = {
    "test_bs1_both": 6.0,
    "test_bs2_none_held_and_the_secrets": 6.0,
    "test_bs3_the_unified_layer": 6.0,
    "test_bs4_the_other_rules_hold_in_both": 6.0,
    "test_bs5_the_texts": 2.0,
}

# The project turn suite's fixtures.
no_engine = tm.no_engine
owned = tm.owned
world = tm.world
chat = tm.chat

_PDF = b"%PDF-1.4\x00\x01\x02" * 64


def _both(chat):
    """A project of a note the store holds and a PDF it does not; the knowledge base answering for both files."""
    project, added = tm._indexed(chat, {"notes.md": NOTES.encode(), "scan.pdf": _PDF})
    chunks = [chat.pc.RetrievedChunk(content="from the scan", source_file=added["scan.pdf"].file_path),
              chat.pc.RetrievedChunk(content="from the note, a second copy", source_file=added["notes.md"].file_path)]

    def knowledge_base(project_id, query, *args, **kwargs):
        chat.asked.append(project_id)
        return list(chunks)

    chat.builder.retrieve_chunks = knowledge_base
    return project, added


def test_bs1_both(chat):
    project, added = _both(chat)
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    assert ctx.source == "mycelium+rag" and chat.asked == [project.id], (ctx.source, chat.asked)
    assert "[notes.md (lines " in ctx.context_text and "zanzibar_quartz" in ctx.context_text, ctx.context_text
    assert "from the scan" in ctx.context_text, ctx.context_text
    assert "a second copy" not in ctx.context_text, "the knowledge base served a file the store serves"
    assert "1 file(s) its Mycelium store does not (scan.pdf)" in ctx.said, ctx.said
    from opti_oignon.context_optimizer import ContextOptimizer

    holder = types.SimpleNamespace(_project_builder=chat.builder, _estimate_tokens=lambda text, model: len(text) // 4)
    _text, zone = ContextOptimizer._inject_project_context(holder, project_id=project.id, query="zanzibar_quartz plan",
                                                           budget_tokens=4096, model="m")
    assert zone.strategy == "mycelium+rag" and "scan.pdf" in zone.detail, (zone.strategy, zone.detail)
    turn = chat.turn.passages(project.id, [added["notes.md"].file_path, added["scan.pdf"].file_path], "zanzibar")
    assert turn.state == "both" and turn.uncovered == (added["scan.pdf"].file_path,), turn


def test_bs2_none_held_and_the_secrets(chat):
    project, added = _both(chat)
    # The store holds none of the files the rows name: the knowledge base serves the whole project.
    turn = chat.turn.passages(project.id, [added["scan.pdf"].file_path], "zanzibar_quartz plan")
    assert turn.state == "rag" and "knowledge base serves it" in turn.said, turn
    # A secret's name counts on neither side: never sent to the knowledge base by this rule, never against the store.
    secret, secret_added = tm._indexed(chat, {"notes.md": NOTES.encode(), ".env": b"API_KEY=s3cr3t\n"}, name="secret")
    ctx = chat.builder.build_context(secret.id, "zanzibar_quartz plan")
    assert ctx.source == "mycelium" and secret.id not in chat.asked, (ctx.source, chat.asked)
    turn = chat.turn.passages(secret.id, [p.file_path for p in secret_added.values()], "zanzibar_quartz plan")
    assert turn.state == "ok" and turn.uncovered == (), turn


def test_bs3_the_unified_layer(chat, monkeypatch):
    from opti_oignon import rag_hybrid_search, unified_retrieval

    project, added = _both(chat)
    monkeypatch.setattr(unified_retrieval, "_project_turn", lambda project_id, query: chat.turn.passages(
        project_id, [added["notes.md"].file_path, added["scan.pdf"].file_path], query))

    def result(path, content):
        return types.SimpleNamespace(source_file=path, section="", total_chunks=1, chunk_index=0, content=content,
                                     fused_score=0.5, chunk_id=content)

    # The knowledge base ranks six chunks of the note the store holds before the scan's, and answers as many as asked.
    ranked = [result(added["notes.md"].file_path, f"from the note, a second copy {n}") for n in range(6)]
    ranked.append(result(added["scan.pdf"].file_path, "from the scan"))
    asked = []

    def search(query, collection=None, n_results=10, **kwargs):
        asked.append(n_results)
        return types.SimpleNamespace(results=ranked[:n_results])

    monkeypatch.setattr(rag_hybrid_search, "get_hybrid_engine", lambda: types.SimpleNamespace(search=search))
    items = unified_retrieval._default_sources(None, project.id)["rag"]("zanzibar_quartz plan", 6)
    contents = [item.content for item in items]
    assert any("zanzibar_quartz" in c for c in contents) and "from the scan" in contents, (asked, contents)
    assert not any(c.startswith("from the note, a second copy") for c in contents), contents
    assert items[0].metadata.get("store") == "mycelium", items[0]
    # A project its store does not cover at all: the knowledge base asked for the limit, as before.
    monkeypatch.setattr(unified_retrieval, "_project_turn", lambda project_id, query: None)
    asked.clear()
    unified_retrieval._default_sources(None, project.id)["rag"]("zanzibar_quartz plan", 6)
    assert asked == [6], asked


def test_bs4_the_other_rules_hold_in_both(chat):
    from opti_oignon.mycelium import search

    project, _added = _both(chat)
    real_find = search.Searcher.find

    def find(self, query, k=10, stages=("symbols", "lexical", "dense"), rerank=False):
        if "dense" in stages:
            time.sleep(2.0)
        return real_find(self, query, k, stages, rerank)

    deadline = chat.turn.DEADLINE_S
    search.Searcher.find, chat.turn.DEADLINE_S = find, 0.3
    try:
        ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    finally:
        search.Searcher.find, chat.turn.DEADLINE_S = real_find, deadline
    assert ctx.source == "mycelium+rag" and "lexical stages answered" in ctx.said and "scan.pdf" in ctx.said, ctx.said
    for thread in threading.enumerate():
        if thread.name == "mycelium-turn":
            thread.join(10)
    chat.asked.clear()
    path = chat.place.folder() / f"project_{project.id}.db"
    path.write_bytes(b"not a store at all " * 512)
    ctx = chat.builder.build_context(project.id, "zanzibar_quartz plan")
    assert ctx.chunks_used == 0 and ctx.source == "" and "cannot be read" in ctx.said, (ctx.source, ctx.said)
    assert chat.asked == [], "the knowledge base served a store that cannot be read"


def test_bs5_the_texts():
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    added = changelog.split("### Added\n", 1)[1].split("\n### ", 1)[0]
    entry = " ".join(next((e for e in added.split("\n- ") if "BS1-BS5" in e), "").split())
    assert "mycelium+rag" in entry and "knowledge base" in entry, entry[:300]
