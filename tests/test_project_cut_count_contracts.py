#!/usr/bin/env python3
"""Contracts for a project text the window's fit cuts: the turn says the passages that were sent.

On the onion journal's path the project's text is fitted to its share of the window (``Executor._onion_fit``): whole
passages leave first, from the last, then characters. The turn's line and the facts its answer keeps said the
builder's numbers, the passages and tokens of the text before the fit ("40 chunks, cut" when 4 were sent).

  * PCC1 -- THE BUILDER NOTES WHERE EACH PASSAGE'S WORDS BEGIN: the real builder (the real package, the projects
    suite's world) notes where the words of each passage it keeps begin in its text, the one its own budget cuts
    included; ``sent`` of a cut of the text from its end counts the passages whose words it reaches and its tokens as
    the builder estimates them; a text that is not such a cut is refused; instructions alone note none.
  * PCC2 -- THE COMPOSER SAYS THE PASSAGES SENT (supersedes pq1 of tests/test_project_facts_contracts.py and ld31 of
    tests/test_project_context_words_contracts.py, their other assertions kept): for a text the fit cut, the line
    and the facts say what the real context's ``sent`` counts; a whole text keeps the builder's numbers.
  * PCC3 -- THE FIT CUTS FROM THE END: ``_onion_fit`` gives a beginning of its text for every share tried, whole
    passages leaving first (a share between two sends them whole).
  * PCC4 -- THE TEXTS: the CHANGELOG says it.
  * PCC5 -- A CUT THAT CANNOT BE COUNTED KEEPS ITS TEXT (review 272-1): a fit whose text is not a cut of the
    context's text from its end (none does today) loses neither the text nor what the turn said: the line says the
    builder's numbers, as before 272, and the log says the count could not be made.

Local-only (the public distribution ships no tests). PCC1 runs the real package over the projects suite's test data
folder; the executor runs in the shared isolation window through the project context words suite's world, the
context's class and the facts' module loaded from their files there: no contract reaches an engine or a store on disk.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_window_world as w  # noqa: E402
import test_mycelium_agent_contracts as my  # noqa: E402
import test_project_context_words_contracts as ldw  # noqa: E402
import test_project_facts_contracts as pq  # noqa: E402
import test_project_mycelium_contracts as pmc  # noqa: E402
from _isolation import REPO  # noqa: E402

no_engine = my.no_engine
owned = my.owned
world = pmc.world

BUDGET_S = {
    "test_pcc1_the_builder_notes_where_each_passages_words_begin": 6.0,
    "test_pcc2_the_composer_says_the_passages_sent": 6.0,
    "test_pcc3_the_fit_cuts_from_the_end": 4.0,
    "test_pcc4_the_texts": 2.0,
    "test_pcc5_a_cut_that_cannot_be_counted_keeps_its_text": 6.0,
}

PF = pq.PF


def _real(cls, parts, instructions="", **fields):
    """A context as ``build_context`` assembles it, from ``(label, words)`` parts: its text and where each passage's
    words begin (PCC1 holds the real builder to this shape)."""
    header, footer = "--- Project Context ---", "--- End Project Context ---"
    pieces = ([instructions] if instructions else []) + ([header] if parts else [])
    starts, at = [], sum(len(p) + 2 for p in pieces)
    for label, words in parts:
        pieces.append(f"{label}{words}")
        starts.append(at + len(label))
        at += len(label) + len(words) + 2
    if parts:
        pieces.append(footer)
    text = "\n\n".join(pieces)
    return cls(context_text=text, chunks_used=len(parts), total_tokens_estimate=len(text) // 4,
               passage_starts=starts, **fields)


# --- PCC1 ---------------------------------------------------------------------------------------------------------


def test_pcc1_the_builder_notes_where_each_passages_words_begin(world, monkeypatch):
    pc = world.pc
    builder = pc.ProjectContextBuilder(store=world.store)
    chunks = [pc.RetrievedChunk(content=f"passage {n} " + "zanzibar " * 40, source_file=f"kb{n}.md",
                                section_name=f"part {n}" if n else "") for n in range(3)]
    monkeypatch.setattr(builder, "retrieve_chunks", lambda project_id, query, *a, **k: [c for c in chunks])
    project = world.store.create_project("p", system_instructions="Answer in English.")
    whole = builder.build_context(project.id, "where is the plan", budget_tokens=4096)
    text, starts = whole.context_text, whole.passage_starts
    assert whole.source == "rag" and whole.chunks_used == 3, ("control: three passages kept", whole)
    assert len(starts) == 3 and starts == sorted(starts), starts
    for n, at in enumerate(starts):
        assert text[at:].startswith(f"passage {n} "), (n, text[at - 20:at + 20])
        assert text[:at].endswith(f"[kb{n}.md{f' (part {n})' if n else ''}]\n"), (n, text[at - 30:at])
    assert whole.sent(text) == (3, whole.total_tokens_estimate), "the whole text sends every passage"
    for n, at in enumerate(starts):
        assert whole.sent(text[:at]) == (n, len(text[:at]) // 4), (n, "a cut before a passage's words")
        assert whole.sent(text[:at + 1]) == (n + 1, len(text[:at + 1]) // 4), (n, "a cut into them")
    assert whole.sent("") == (0, 0)
    with pytest.raises(ValueError):
        whole.sent("x" + text[:starts[1]])
    with pytest.raises(ValueError):
        whole.sent(text[:starts[0]] + "elsewhere")
    small = builder.build_context(project.id, "where is the plan", budget_tokens=200)
    assert small.chunks_used == 2 and "(part 1, cut)]" in small.context_text, ("control: the budget cut one", small)
    last = small.passage_starts[-1]
    assert small.context_text[:last].endswith("[kb1.md (part 1, cut)] "), small.context_text[last - 30:last]
    assert small.context_text[last:].startswith("passage 1 "), small.context_text[last:last + 20]
    alone = builder.build_system_instructions_only(project.id)
    assert alone.context_text == "Answer in English." and alone.passage_starts == [], alone
    assert alone.sent(alone.context_text) == (0, len("Answer in English.") // 4)
    bare = world.store.create_project("q")
    plain = builder.build_context(bare.id, "where is the plan", budget_tokens=4096)
    assert not plain.system_instructions and plain.chunks_used == 3, ("control: no instructions", plain)
    assert plain.context_text.startswith("--- Project Context ---"), plain.context_text[:40]
    for n, at in enumerate(plain.passage_starts):
        assert plain.context_text[at:].startswith(f"passage {n} "), ("no instructions (review 272-1)", n, at)


# --- PCC2 ---------------------------------------------------------------------------------------------------------


def test_pcc2_the_composer_says_the_passages_sent():
    # pq1's cases and assertions, the cut's count the passages sent.
    ctx = ldw._context
    cases = {
        "mycelium": (ctx(context_text="[a (lines 1-4)] plan", chunks_used=3, total_tokens_estimate=40,
                         source="mycelium"), {"source": "mycelium", "passages": 3, "said": None}),
        "both": (ctx(context_text="[a (lines 1-4)] plan", chunks_used=4, total_tokens_estimate=50,
                     source="mycelium+rag", said=pq._LEFT),
                 {"source": "mycelium+rag", "passages": 4, "said": {"form": "left", "files": 2}}),
        "rag": (ctx(context_text="from the knowledge base", chunks_used=2, total_tokens_estimate=30, source="rag",
                    said="the project holds files its Mycelium store does not: the knowledge base serves it"),
                {"source": "rag", "passages": 2, "said": {"form": "rag_serves"}}),
        "unreadable": (ctx(said="the project's Mycelium store cannot be read (DatabaseError): no project passage"),
                       {"source": "none", "passages": 0, "said": {"form": "unreadable"}}),
        "instructions": (ctx(context_text="Answer in English.", chunks_used=0, total_tokens_estimate=5,
                             source="mycelium"), {"source": "none", "passages": 0, "said": None}),
        "lexical": (ctx(context_text="[a (lines 1-4)] plan", chunks_used=1, total_tokens_estimate=9,
                        source="mycelium", said="the search took longer than 2.5 s: the lexical stages answered"),
                    {"source": "mycelium", "passages": 1, "said": {"form": "lexical", "s": 2.5}}),
        "unknown": (ctx(context_text="[a (lines 1-4)] plan", chunks_used=1, total_tokens_estimate=9,
                        source="mycelium", said="a new form that names notes.md"),
                    {"source": "mycelium", "passages": 1, "said": {"form": "unknown"}}),
    }
    said_both = ("the project holds 1 file(s) its Mycelium store does not (scan.pdf): those are left to the "
                 "knowledge base")
    executor, restore = ldw._load_executor()
    try:
        cls = w.project_context_class()
        four = _real(cls, [("[a (lines 1-4)]\n", "plan"), ("[b (lines 1-9)]\n", "draft"), ("[c]\n", "minutes"),
                           ("[scan.pdf]\n", "from the scan")], source="mycelium+rag", said=pq._LEFT)
        two = _real(cls, [("[notes.md (lines 1-4)]\n", "zanzibar plan"), ("[scan.pdf]\n", "from the scan")],
                    source="mycelium+rag", said=said_both)
        executor._project_facts = PF
        got = {name: pq._composed(executor, c) for name, (c, _want) in cases.items()}
        whole = pq._composed(executor, four, fit=lambda text: text)
        one = four.passage_starts[0] + 1
        cut = pq._composed(executor, four, fit=lambda text: text[:one])
        nothing = pq._composed(executor, four, fit=lambda text: text[:four.passage_starts[0]])
        floor = pq._composed(executor, cases["mycelium"][0], budget_tokens=executor._RAG_FLOOR - 1)
        lines = {name: ldw._executor_lines(executor, two, budget_tokens=4096, fit=fit) for name, fit in (
            ("whole", lambda text: text), ("cut", lambda text: text.split("\n\n[scan.pdf]")[0]),
            ("gone", lambda text: ""))}
    finally:
        restore()
    for name, (_c, want) in cases.items():
        assert got[name][0] == want, f"{name}: {got[name][0]} (said: {got[name][1]})"
    assert any(line.endswith("from the project's Mycelium store") for line in got["mycelium"][1]), (
        "control: the passages' line names the source the facts keep")
    assert whole[0] == {"source": "mycelium+rag", "passages": 4, "said": {"form": "left", "files": 2}}, whole
    assert cut[0] == {"source": "cut", "passages": 1, "said": {"form": "left", "files": 2}}, cut
    tokens = len(four.context_text[:one]) // 4
    assert f"[OK] Project context injected: 1 chunks, ~{tokens} tokens, cut to its share of the window" in cut[1], cut
    assert nothing[0] == {"source": "cut", "passages": 0, "said": {"form": "left", "files": 2}}, nothing
    assert floor[0] == {"source": "left_out", "passages": 0, "said": None}, floor
    every = json.dumps([g[0] for g in got.values()] + [whole[0], cut[0], nothing[0], floor[0]])
    assert not [n for n in pq._NAMES if n in every], f"no name is kept: {every}"
    # ld31's, the cut's line the passage sent.
    full, kept, gone = lines["whole"], lines["cut"], lines["gone"]
    assert full[0] == two.context_text and full[1][1] == (
        f"[OK] Project context injected: 2 chunks, ~{two.total_tokens_estimate} tokens, from the project's Mycelium "
        "store and the knowledge base"), full
    assert kept[0] == two.context_text.split("\n\n[scan.pdf]")[0], ("the text handed on is the fitted one", kept[0])
    assert kept[1][1] == (f"[OK] Project context injected: 1 chunks, ~{len(kept[0]) // 4} tokens, cut to its share "
                          "of the window"), kept[1]
    assert gone[0] == "" and not [line for line in gone[1] if line.startswith("[OK]")], gone[1]
    for _text, said in (full, kept, gone):
        assert said[-1] == f"[!] Project context: {said_both}", ("what the turn says, last, whatever the fit", said)
    out = ldw._words([kept[1][1]])
    assert out["statuses"] == ["Project context: 1 passage, cut to fit the window"], out["statuses"]
    written = (REPO / "opti_oignon" / "executor.py").read_text(encoding="utf-8")
    journal = written.split("_rag_fit = (", 1)[1].split("_onion_tail.append", 1)[0]
    composed, after = journal.split("_rag_text = self._compose_project_context(", 1)
    assert "self._onion_fit(" in composed and "fit=_rag_fit" in after, "the journal's path hands its fit to the composer"
    assert "_onion_fit(" not in after, "the text is fitted once, before its lines are said"


# --- PCC3 ---------------------------------------------------------------------------------------------------------


def test_pcc3_the_fit_cuts_from_the_end():
    world, close = w.open_real(gov={}, projects=True)
    try:
        cls = w.project_context_class()
        ctx = _real(cls, [(f"[f{n}.md (lines 1-9)]\n", f"passage {n} " + "plan " * 30) for n in range(5)],
                    instructions="Answer in English.")
        seen = {}
        executor = world.executor_module.Executor
        fitter = executor.__new__(executor)
        for share in range(0, 4 * ctx.total_tokens_estimate, 7):  # the window counts heavier than the builder
            seen[share] = executor._onion_fit(fitter, ctx.context_text, share, "test-model:1b")
    finally:
        close()
    text = ctx.context_text
    assert all(text.startswith(out) for out in seen.values()), [s for s, out in seen.items() if not text.startswith(out)]
    sent = {share: ctx.sent(out)[0] for share, out in seen.items()}
    assert {0, 1, 5} <= set(sent.values()) and len(set(sent.values())) >= 4, ("control: the shares tried cut "
                                                                              "passages away one by one", sent)
    assert sorted(sent.values()) == [sent[s] for s in sorted(sent)], ("a larger share sends no fewer passages", sent)
    whole = {ctx.sent(out)[0] for out in seen.values() if text[len(out):].startswith("\n\n[")}
    assert {1, 2, 3, 4} <= whole, ("whole passages leave first: a share between two sends them whole (review 272-1)",
                                   whole)


# --- PCC4 ---------------------------------------------------------------------------------------------------------


def test_pcc4_the_texts():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "PCC1-PCC5" in e), "").split())
    assert "cut" in entry and "passages" in entry and "sent" in entry, entry[:300]


# --- PCC5 ---------------------------------------------------------------------------------------------------------


def test_pcc5_a_cut_that_cannot_be_counted_keeps_its_text(caplog):
    said = "the search took longer than 2.5 s: the lexical stages answered"
    executor, restore = ldw._load_executor()
    try:
        cls = w.project_context_class()
        two = _real(cls, [("[notes.md (lines 1-4)]\n", "zanzibar plan"), ("[scan.pdf]\n", "from the scan")],
                    source="mycelium", said=said)
        executor._project_facts = PF
        with caplog.at_level("WARNING"):
            text, lines = ldw._executor_lines(executor, two, budget_tokens=4096, fit=lambda text: text[1:])
            facts, _lines = pq._composed(executor, two, fit=lambda text: text[1:])
    finally:
        restore()
    with pytest.raises(ValueError):
        two.sent(two.context_text[1:])          # control: the count refuses a text that is not a cut from its end
    assert text == two.context_text[1:], ("the text is handed on", text)
    assert (f"[OK] Project context injected: 2 chunks, ~{two.total_tokens_estimate} tokens, cut to its share of the "
            "window") in lines, lines
    assert lines[-1] == f"[!] Project context: {said}", ("what the turn says, last", lines)
    assert facts == {"source": "cut", "passages": 2, "said": {"form": "lexical", "s": 2.5}}, facts
    assert any("cannot be counted" in record.getMessage() for record in caplog.records), caplog.text
