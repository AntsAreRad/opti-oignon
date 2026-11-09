#!/usr/bin/env python3
"""Contracts for a turn that does not leave: it keeps no fact of its project's context.

A turn composes its project's context early; several of its ends come after, with nothing sent: the onion window
refused by name (its plan, its composition), the governor's refusal, the engine offline, a stop before the stream.
Each left in the turn's results the facts of the passages kept (``project_facts``), and the ``done`` frame said them
under the refusal. Every end of a turn writes one ledger row: the row of a turn whose request never left (no stream
opened to the engine, no answer from the cache) hands the run back the facts it found.

  * PUS1 -- THE JOURNAL'S TURN THAT DOES NOT LEAVE KEEPS NO FACT: a plan refused, a composition refused, the
    governor's refusal, the engine offline, a stop at the plan, at the admission, at the cache's lookup, in the stream's
    thread before the engine is called: the composer said its line, the ledger says the end, nothing was sent, and the
    turn's results carry no fact; nothing is saved with any.
  * PUS2 -- NOR DOES THE MANUAL PATH'S: the onion off, the governor's refusal, the engine offline, a stop at the
    admission, at the cache's lookup, in the stream's thread.
  * PUS3 -- A TURN THAT LEAVES KEEPS ITS FACTS: completed, or answered from the cache, the onion on or off: the
    composer's facts, the answer saved with them; stopped while the engine answers: the composer's facts (review
    273-1).
  * PUS4 -- THE TEXTS: the CHANGELOG says it.
  * PUS5 -- A CALL THAT DOES NOT LEAVE HANDS THE RUN BACK ITS FACTS (review 273-1): two calls on one run, as a
    pipeline's phases make them (think and tools): the first completes, the second is refused, the onion on or off;
    the run keeps the first call's facts, those the answer it saves was composed with.

Local-only (the public distribution ships no tests). The turns run in the shared isolation window through the onion
window suite's own world (tests/_onion_window_world.py), its trace keeping what every save was handed; the facts'
module is loaded from its file where the window does not carry it: no contract reaches an engine or a store.
"""

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_off_golden as golden  # noqa: E402
import _onion_window_world as w  # noqa: E402
from _isolation import REPO  # noqa: E402

BUDGET_S = {
    "test_pus1_the_journals_turn_that_does_not_leave_keeps_no_fact": 20.0,
    "test_pus2_nor_does_the_manual_paths": 20.0,
    "test_pus3_a_turn_that_leaves_keeps_its_facts": 20.0,
    "test_pus4_the_texts": 2.0,
    "test_pus5_a_call_that_does_not_leave_hands_the_run_back_its_facts": 20.0,
}

_SAID = "the search took longer than 10 s: the lexical stages answered"
_TEXT = "[notes.md (lines 1-4)] the venue is Hall B"
_COMPOSED = "[OK] Project context injected: 2 chunks"


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


def _turn(onion, *, again=False, stop_when_asked=False, **options):
    """A project turn in a world opened with ``options``, the onion on or off, and its ledger's rows; ``again``: the
    turn is the second, its question the first's in another conversation (a cache that keeps the first answers it);
    ``stop_when_asked``: the turn's stop is raised once the engine was asked for the stream.
    """
    journal = dict(options.pop("journal", {}), enabled=onion)
    world, close = w.open_world(journal=journal, projects=True, **options)
    try:
        world.executor_module._project_facts = PF
        _builder(world)
        if stop_when_asked:
            streaming = world.backend.stream

            def stream(*args, **kwargs):
                answer = streaming(*args, **kwargs)
                world.run.stop.set()
                return answer

            world.backend.stream = stream
        turns = (("conv-1", w.FIRST, {}), ("conv-2", w.FIRST, {})) if again else (("conv-1", w.FIRST, {}),)
        records = w.run_turns(world, turns)
        return records[-1], list(world.rows)
    finally:
        close()


def _kept_none(case, record, rows, outcome):
    assert any(line.startswith(_COMPOSED) for line in record["statuses"]), (
        "control: the turn composed its project's passages", case, record["statuses"])
    assert [row.get("outcome") for row in rows][-1:] == [outcome], ("control: the turn's end", case, rows)
    assert record["requests"] == [], ("control: nothing was sent", case, record["requests"])
    assert "project_context" not in record["results"], (case, record["results"])
    assert not any((saved or {}).get("project_context") for saved in _saved_answers(record)), (
        case, _saved_answers(record))


def test_pus1_the_journals_turn_that_does_not_leave_keeps_no_fact():
    for case, options, outcome in (
        ("a plan refused", {"gov": {}, "journal": {"plan_refusal": ("turn", (900, 512))}}, "window_refused"),
        ("a composition refused", {"gov": {}, "journal": {"compose_refusal": ("window", (17000, 16384))}},
         "window_refused"),
        ("the governor's refusal", {"gov": {"admitted": False}}, "governor_refused"),
        ("the engine offline", {"gov": {}, "offline": True}, "offline_queued"),
        ("a stop at the plan", {"gov": {}, "stop_at": {"plan": {0}}}, "cancelled"),
        ("a stop at the admission", {"gov": {}, "stop_at": {"admission": {0}}}, "cancelled"),
        ("a stop at the cache's lookup", {"gov": {}, "stop_at": {"lookup": {0}}}, "cancelled"),
        ("a stop in the stream's thread", {"gov": {}, "stop_at": {"thread": {0}}}, "cancelled"),
    ):
        record, rows = _turn(True, **options)
        assert "plan" in w.kinds(record), ("control: the onion's journal laid the turn", case)
        _kept_none(case, record, rows, outcome)


def test_pus2_nor_does_the_manual_paths():
    for case, options, outcome in (
        ("the governor's refusal", {"gov": {"admitted": False}}, "governor_refused"),
        ("the engine offline", {"gov": {}, "offline": True}, "offline_queued"),
        ("a stop at the admission", {"gov": {}, "stop_at": {"admission": {0}}}, "cancelled"),
        ("a stop at the cache's lookup", {"gov": {}, "stop_at": {"lookup": {0}}}, "cancelled"),
        ("a stop in the stream's thread", {"gov": {}, "stop_at": {"thread": {0}}}, "cancelled"),
    ):
        record, rows = _turn(False, **options)
        assert "plan" not in w.kinds(record), ("control: the onion off lays no journal", case)
        _kept_none(case, record, rows, outcome)


def test_pus3_a_turn_that_leaves_keeps_its_facts():
    want = PF.facts("mycelium", 2, _SAID)
    for onion in (True, False):
        for case, options, outcome in (("completed", {"gov": {}}, "completed"),
                                       ("answered from the cache", {"gov": {}, "hit": True, "again": True,
                                                                    "engine_extra": {"eval_count": 3}}, "cache_hit")):
            record, rows = _turn(onion, **options)
            assert [row.get("outcome") for row in rows][-1:] == [outcome], ("control", onion, case, rows)
            assert record["results"].get("project_context") == want, (onion, case, record["results"])
            assert _saved_answers(record) == [golden.shape({"m": {"project_context": want}})["m"]], (
                onion, case, _saved_answers(record))
        record, rows = _turn(onion, gov={}, stop_when_asked=True)
        assert [row.get("outcome") for row in rows][-1:] == ["cancelled"], ("control: stopped", onion, rows)
        assert record["requests"], ("control: the engine was asked before the stop", onion)
        assert record["results"].get("project_context") == want, ("stopped while answering", onion,
                                                                   record["results"])


def test_pus4_the_texts():
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = " ".join(next((e for e in changelog.split("\n- ") if "Contracts PUS1-PUS" in e), "").split())
    assert "refused" in entry and "offline" in entry and "facts" in entry, entry[:300]
    assert "PUS1-PUS5" in entry, entry[-200:]


def _two_calls(onion):
    """Two calls on one run, the first completed and the second refused (the engine offline), as a pipeline's phases
    make them: the run's results, and the ledger's rows."""
    world, close = w.open_world(journal={"enabled": onion}, gov={}, projects=True)
    try:
        world.executor_module._project_facts = PF
        _builder(world)
        executor = world.executor_module.Executor()
        run = SimpleNamespace(stop=threading.Event(), results={})
        world.run = run
        for index in range(2):
            if index:
                world.trace.next_turn()
                world.requests.append([])
                world.executor_module._network_manager._online = False
            world.turn = index
            run.user_turn = world.executor_module.user_turn(w.FIRST, w.FIRST, (), ())
            for _chunk in executor.execute(w.FIRST, w.routing(), refine=False, conversation_id="conv-1", run=run,
                                           on_status=lambda line: None):
                pass
        return dict(run.results), list(world.rows)
    finally:
        close()


def test_pus5_a_call_that_does_not_leave_hands_the_run_back_its_facts():
    want = PF.facts("mycelium", 2, _SAID)
    for onion in (True, False):
        results, rows = _two_calls(onion)
        assert [row.get("outcome") for row in rows] == ["completed", "offline_queued"], ("control", onion, rows)
        assert results.get("project_context") == want, (onion, results)
