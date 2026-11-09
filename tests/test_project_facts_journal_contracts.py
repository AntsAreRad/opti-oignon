#!/usr/bin/env python3
"""Contracts for the onion journal's turns: they keep the facts of their project's context too.

Every other path of a project turn writes the facts of its project's context (``project_facts``: the source of the
passages kept, their number, the form of what the turn said; never a name) in the turn's results and saves the answer
with them. The onion journal's path composed the project's text with its share of the window and said its lines, and
kept nothing.

  * PJR1 -- THE JOURNAL'S TURN KEEPS ITS FACTS: a turn of the real onion (its librarian, its window) over a stand-in
    project builder: the turn's results carry the facts the composer's lines say, and the answer is saved with them.
  * PJR2 -- A RETRIEVAL THE WINDOW LETS GO IS KEPT LEFT OUT: the journal's window cuts the project's retrieval: the
    turn says it was left out, and its facts say so (no passage), the answer saved with them; a window that keeps it,
    or cuts another kind only (review 271-1), keeps the composer's facts; a share of the window that holds no chunk
    keeps the composer's own left out (review 271-1).
  * PJR3 -- THE TEXTS: the CHANGELOG says it; and the facts' module carries the platform's module flag
    (``checkpoint_before_apply``), as every module the platform adds.

Local-only (the public distribution ships no tests). The onion's journal runs in the shared isolation window through
the onion window suite's own world (tests/_onion_window_world.py), its trace keeping what every save was handed; the
facts' module is loaded from its file where the window does not carry it: no contract reaches an engine or a store.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_off_golden as golden  # noqa: E402
import _onion_window_world as w  # noqa: E402
from _isolation import REPO  # noqa: E402

BUDGET_S = {
    "test_pjr1_the_journals_turn_keeps_its_facts": 10.0,
    "test_pjr2_a_retrieval_the_window_lets_go_is_kept_left_out": 10.0,
    "test_pjr3_the_texts": 2.0,
}

_SAID = "the search took longer than 10 s: the lexical stages answered"
_TEXT = "[notes.md (lines 1-4)] the venue is Hall B"


def _facts_module():
    spec = importlib.util.spec_from_file_location("project_facts_under_contract",
                                                  REPO / "opti_oignon" / "project_facts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PF = _facts_module()


def _builder(world):
    world.projects.build_context = lambda *a, **k: SimpleNamespace(
        context_text=_TEXT, chunks_used=2, total_tokens_estimate=12, source="mycelium", said=_SAID)


def _saved_answers(record):
    """What each answer the turn saved was handed beside its text, as the trace kept it."""
    return [fields.get("metadata") for event in record["trace"] if event[0] == "store.add"
            for fields in [event[1] if len(event) > 1 else {}] if fields.get("role") == "assistant"]


def test_pjr1_the_journals_turn_keeps_its_facts():
    world, close = w.open_real(gov={}, projects=True)
    try:
        world.executor_module._project_facts = PF
        _builder(world)
        (record,) = w.run_turns(world, (("conv-1", w.FIRST, {}),))
    finally:
        close()
    want = PF.facts("mycelium", 2, _SAID)
    assert any(line.startswith("[OK] Project context injected: 2 chunks") for line in record["statuses"]), (
        "control: the journal's turn composed the project's passages", record["statuses"])
    assert record["results"].get("project_context") == want, record["results"]
    assert _saved_answers(record) == [golden.shape({"m": {"project_context": want}})["m"]], _saved_answers(record)


def test_pjr2_a_retrieval_the_window_lets_go_is_kept_left_out():
    kept = {}
    for name, cut in (("cut", (("rag", 1), ("web", 0))), ("whole", ()), ("memory", (("memory", 2), ("rag", 0)))):
        world, close = w.open_world(gov={}, projects=True, journal={"cut": cut})
        try:
            world.executor_module._project_facts = PF
            _builder(world)
            (record,) = w.run_turns(world, (("conv-1", w.FIRST, {}),))
        finally:
            close()
        kept[name] = record
    cut, whole = kept["cut"], kept["whole"]
    assert "[!] Project context left out: over its share of the window" in cut["statuses"], cut["statuses"]
    want = PF.facts("left_out", 0, _SAID)
    assert cut["results"].get("project_context") == want, cut["results"]
    assert _saved_answers(cut) == [golden.shape({"m": {"project_context": want}})["m"]], _saved_answers(cut)
    assert whole["results"].get("project_context") == PF.facts("mycelium", 2, _SAID), whole["results"]
    other = kept["memory"]
    assert other["results"].get("project_context") == PF.facts("mycelium", 2, _SAID), (
        "a window that cuts another kind keeps the project's facts", other["results"])
    world, close = w.open_real(gov={}, projects=True, limits=(8192, 4096))
    try:
        world.executor_module._project_facts = PF
        _builder(world)
        (floor,) = w.run_turns(world, (("conv-1", w.FIRST, {}),))
    finally:
        close()
    assert "[!] Project context left out: its share of the window holds no chunk" in floor["statuses"], (
        "control: the share holds no chunk", floor["statuses"])
    assert floor["results"].get("project_context") == PF.facts("left_out", 0), floor["results"]


def test_pjr3_the_texts():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    flat = " ".join(changelog.split())
    entry = " ".join(next((e for e in changelog.split("\n- ") if "PJR1-PJR3" in e), "").split())
    assert "onion journal" in entry and "left out" in entry, entry[:300]
    assert "not yet on the onion journal" not in flat, "the CHANGELOG no longer says the journal keeps none"
    module = (REPO / "opti_oignon" / "project_facts.py").read_text(encoding="utf-8")
    assert "\ncheckpoint_before_apply = True\n" in module, "the platform's module flag"
