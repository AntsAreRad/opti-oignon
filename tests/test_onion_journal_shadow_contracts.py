#!/usr/bin/env python3
"""Contracts for the journal's Flesh when it does not fit: caught up at the plan, or laid over a shadow.

A turn's plan finds the Flesh over its cap at the conversation's own rung --
a conversation the process had not seen, one older than the onion, one
grown past its cap -- and catches it up before the turn is admitted, with
no model, while no burst holds the conversation's slot: the turns past the
journal's reach leave bare, the queue runs its rungs that need no model,
then bare spans until it fits. With the slot held, nothing is evicted, and
the window lays the overflow over a shadow: its oldest turns leave into a
state of the shadow's own, for one composition; one line names them, first
among the tail's memory and charged to the Flesh room they freed, and the
peels the shadow made of them follow the memory the plan brings. A descent
-- a context under the plan's rung -- is composed the same way, under the
largest rung it holds, and the conversation keeps its own rung. Once a
journal laid the conversation, a burst fires at the journal's mark of the
Flesh's cap and evicts down to it.

  * JW6a -- a descent is composed over a shadow: under the rung the context
    holds, with that rung's digest, the answer keeping its floor; one line
    in the turn's message names the turns set aside, the newest included,
    beside the memory the plan brings, even at the smallest rung, whose
    memory share cannot hold its envelope; the conversation's state
    untouched, the epoch begun by the rung, each window counted once; the
    next plan is at the conversation's own rung again. A line even the
    whole Flesh room cannot hold asks the next rung up, by name.
  * JW9 -- a conversation the process had not seen, 300 turns long, every
    model seam raising, is caught up at its plan, each turn costed once:
    the Flesh within its cap at the conversation's own rung, the receipts'
    spans and the Flesh together the turns, each once; the composition
    evicts nothing more. With no step of the queue allowed, bare spans
    still bring it within its cap.
  * JW12b -- the Flesh is counted as the journal lays it wherever its fit
    is read: the window's Flesh layer is what the burst's cost reads, the
    plan's catch-up runs the queue at that cost, and the queue given that
    cost evicts a Flesh whose own count fits its cap but whose laid cost
    does not.
  * JW17 -- the plan shares the writer: with a burst holding the slot while
    its model writes, a plan neither waits on the slot nor evicts, the
    window lays the overflow over a shadow, and the burst then commits its
    span, not stale; a plan that caught up returns holding neither the
    slot nor the state's lock, and a burst between it and its composition
    runs to its end.
  * JW20 -- once a journal laid the conversation, a burst fires at the
    journal's mark of the Flesh's cap, however few turns the Flesh grew by,
    and evicts down to that mark; under the mark, one turn grown fires
    nothing, and a conversation no journal laid keeps the growth rule alone.
  * JW24 -- a Flesh over its cap, or over the room the answer's floor or a
    turn's images leave it, is laid over a shadow, the state untouched: the
    line naming the turns set aside first among the memory, the memory the
    plan brings next, the shadow's peels after it. Such a window begins its
    epoch by eviction, a take-back under it by take_back; a line even the
    whole Flesh room cannot hold is refused flesh by name, and a turn with
    no conversation whose images pass the whole Flesh cap of a descent is
    refused document. A Flesh that, laid, passes the sum of its turns -- a
    header neutralized where two user turns meet -- gives again over a
    shadow, at its own rung and at a descent, and is not refused. One over
    its cap at the plan is caught up when the slot is free. With JW6a and
    JW9, it takes JW23's place.
  * JW25 -- the room each kind of source may take at the conversation's own
    rung (``tail_room``) is the cap a window cuts it to; none when no rung
    is within the model's window, and it never raises.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import re
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_journal as oj  # noqa: E402

BUDGET_S = {
    "test_jw6a_a_descent_is_composed_over_a_shadow_and_the_conversation_keeps_its_rung": 8.0,
    "test_jw9_a_conversation_unseen_is_caught_up_at_its_plan_with_no_model": 5.0,
    "test_jw12b_the_flesh_is_counted_as_laid_wherever_its_fit_is_read": 5.0,
    "test_jw17_a_plan_shares_the_writer_with_the_burst": 10.0,
    "test_jw20_a_burst_fires_at_the_journals_mark_and_evicts_down_to_it": 5.0,
    "test_jw24_a_flesh_that_does_not_fit_is_laid_over_a_shadow_or_caught_up": 8.0,
    "test_jw25_the_tail_room_of_each_kind_is_the_cap_a_window_cuts_it_to": 5.0,
}

_FOLD = re.compile(r"(\d+) earlier spans folded: turns (\S+)\.\.(\S+)")


def _texts(messages):
    return [message["content"] for message in messages]


def _digests(state):
    """What the conversation's state holds, by name: the Flesh's turn ids, the receipts, the peels and the Core."""
    with state.lock:
        return (tuple(t.get("turn_id") for t in state.flesh.turns()),
                tuple((r.key, r.kind) for r in state.ledger.all()), tuple(p.id for p in state.tree.all()),
                state.core.root())


def _held_texts(state):
    """The texts the state holds, in order: each open receipt's span from the Cellar, then the Flesh."""
    with state.lock:
        gone = {r.key for r in state.ledger.all() if r.kind == "superseded"}
        spans = [t.get("text") for r in state.ledger.all() if r.key not in gone for t in state.cellar.get(r.key)]
        return spans + [t.get("text") for t in state.flesh.turns()]


def _flesh_cost(lib, state, wrap, journal):
    cost = lib._turn_cost(oj.count, wrap, journal.message_overhead)
    with state.lock:
        return sum(cost(turn) for turn in state.flesh.turns())


def _last(win):
    return win.messages[-1][1]


def _fold(win):
    """The line a window lays for the turns its shadow set aside: ``(spans, first turn, last turn)``; empty when
    none."""
    found = _FOLD.search(_last(win)) if win.refusal is None else None
    return found.groups() if found else ()


def _set_aside(lib, state, wrap, journal, win):
    """The ids of the state's Flesh turns a window did not lay: the oldest, read from the Flesh layer it laid."""
    cost = lib._turn_cost(oj.count, wrap, journal.message_overhead)
    with state.lock:
        turns = state.flesh.turns()
    costs = [cost(turn) for turn in turns]
    sums = [sum(costs[start:]) for start in range(len(costs) + 1)]
    laid = dict(win.layers)["flesh"]
    assert laid in sums, ("control: the window laid the newest turns of the Flesh", laid)
    return tuple(turn.get("turn_id") for turn in turns[:sums.index(laid)])


def _plan_aside(lib, state, cid, question, **kw):
    """The plan on a thread of its own, bounded: one that waits on the slot, or returns holding the state's lock,
    fails by name instead of hanging the run."""
    box = []
    worker = threading.Thread(target=lambda: box.append(oj.plan(lib, cid, question, **kw)), daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "the plan waited on the slot under the state's lock"
    assert state.lock.acquire(timeout=5), "the plan returned holding the state's lock"
    state.lock.release()
    return box[0]


def _no_model(*_args, **_kwargs):
    raise AssertionError("a model was asked")


# ---------------------------------------------------------------------------
# JW6a -- a descent over a shadow
# ---------------------------------------------------------------------------
def test_jw6a_a_descent_is_composed_over_a_shadow_and_the_conversation_keeps_its_rung():
    loaded, restore = oj.open_window()
    try:
        lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
        wrap = loaded[oj.WRAPPER].wrap
        cfg, journal, budgets = oj.config(lib), composer.load_journal(), composer.load_budgets()
        stored = oj.conversation(120)
        ask = "What did Alice review?"
        home = oj.window(loaded, "c1", ask, messages=stored, config=cfg)
        assert home.refusal is None and home.rung == budgets.home, ("control: laid at home", home.refusal)
        laid = dict(home.layers)["flesh"]
        assert composer.layer_cap(budgets.at(8192), "flesh") < laid <= composer.layer_cap(budgets.at(home.rung),
                                                                                           "flesh"), (
            "control: a Flesh within its cap at home, over it at 8192", laid)
        state = lib.state_for("c1", cfg)
        before = _digests(state)
        plan = oj.plan(lib, "c1", ask, wrap=wrap, messages=stored, config=cfg)
        assert plan.refusal is None and plan.rung == budgets.home, plan
        down = lib.compose_window(plan, num_ctx=8192)
        assert down.refusal is None, down.refusal
        assert (down.rung, down.num_ctx) == (8192, 8192), (down.rung, down.num_ctx)
        assert down.accounted + down.num_predict == 8192 and down.num_predict >= journal.answer_floor, (
            down.accounted, down.num_predict)
        assert dict(down.layers)["flesh"] <= composer.layer_cap(budgets.at(8192), "flesh"), down.layers
        assert down.motive == "rung", ("a descent begins its epoch by the rung", down.motive)
        aside = _set_aside(lib, state, wrap, journal, down)
        assert aside and _fold(down)[1:] == (aside[0], aside[-1]), (
            "one line in the turn's message names the turns set aside, the newest included", _fold(down), aside[-1:])
        assert "provenance=shadow:" not in "".join(content for _role, content, _label in down.messages[:-1]), (
            "nothing of the shadow in the prefix")
        assert _digests(state) == before, "the descent evicts nothing of the conversation's state"
        assert lib.counters()["journal"]["descent"] == 1 and lib.counters()["shadow"] == {"8192": 1}, (
            "one window over a shadow, counted once", lib.counters())
        # At the smallest rung the memory's share cannot hold even its envelope: the line is charged to the Flesh
        # room the turns it names freed.
        low = lib.compose_window(oj.plan(lib, "c1", ask, wrap=wrap, messages=stored, config=cfg), num_ctx=4096)
        assert low.refusal is None and low.rung == 4096, low.refusal
        aside = _set_aside(lib, state, wrap, journal, low)
        assert aside and _fold(low)[1:] == (aside[0], aside[-1]), (
            "at the smallest rung too, the line names the newest turn set aside", _fold(low), aside[-1:])
        line = _FOLD.search(_last(low)).group(0)
        charge = composer.fold_cost(("memory", "fold", line, ("receipts", "shadow:receipts"), None), wrap, oj.count)
        assert int(journal.shares()["memory"] * budgets.at(4096).tail) < charge, (
            "control: the memory's share at the smallest rung cannot hold the line", charge)
        assert dict(low.layers)["flesh"] + charge <= composer.layer_cap(budgets.at(4096), "flesh"), (
            "the line charged to the Flesh room its turns freed", low.layers, charge)
        again = oj.plan(lib, "c1", ask, wrap=wrap, messages=stored, config=cfg)
        assert again.rung == budgets.home, ("the conversation keeps its own rung", again.rung)
        back = lib.compose_window(again)
        assert back.refusal is None and back.rung == budgets.home and back.motive == "rung", (back.rung, back.motive)
        assert "provenance=shadow:" not in _last(back), "control: at its own rung, nothing over a shadow"
        under = lib.compose_window(again, num_ctx=1000)
        assert under.refusal == lib.Refusal("rung", (1000, budgets.rungs[0])), (
            "a context under every rung is refused by name", under.refusal)
        # The line beside the memory the plan brings: a span the user recalls is laid with it, after it.
        recalls = oj.conversation(122)
        kept = lib.state_for("c1r", cfg)
        kept.mirror(recalls)
        recalled = oj.bare(kept, 2)
        beside = lib.compose_window(oj.plan(lib, "c1r", ask, wrap=wrap, messages=recalls, config=cfg,
                                            recalled=(recalled.key,)), num_ctx=8192)
        last = _last(beside) if beside.refusal is None else ""
        assert _fold(beside) and "layer=recalled" in last, ("the recall laid beside the line", beside.refusal,
                                                            beside.cut)
        assert last.index("provenance=shadow:receipts") < last.index("layer=recalled"), (
            "the line first among the memory")
        # Each rung its own digest: thirty receipts render over the 8192 rung's receipts cap at home, within it
        # below.
        older = oj.conversation(240)
        held = lib.state_for("c1d", cfg)
        held.mirror(older)
        for _ in range(30):
            oj.bare(held, 4)
        top = oj.window(loaded, "c1d", ask, messages=older, config=cfg)
        assert top.refusal is None and dict(top.layers)["receipts"] > budgets.at(8192).receipts, (
            "control: the digest at home is over the 8192 rung's cap", top.refusal, top.layers)
        below = lib.compose_window(oj.plan(lib, "c1d", ask, wrap=wrap, messages=older, config=cfg), num_ctx=8192)
        assert below.refusal is None and dict(below.layers)["receipts"] <= budgets.at(8192).receipts, (
            "the descent lays the digest of the rung it is composed under", below.refusal, below.layers)
        aside = _set_aside(lib, held, wrap, journal, below)
        assert aside and _fold(below)[1:] == (aside[0], aside[-1]), (_fold(below), aside[-1:])
        # A window that builds its shadow more than once -- its Flesh giving to the room the answer's floor
        # leaves -- is counted once.
        wide = composer.Budget(window=8192, reserve=0, system=256, core=256, receipts=128, peels=128, flesh=6200,
                               tail=1000, turn=224)
        counted = lib.counters()["shadow"]["8192"]
        question = ("Which services did Alice review, and which of them stay on the new cluster after the move, "
                    "given what the team decided in March about the old racks and the nightly backups?")
        twice = oj.window(loaded, "c1t", question, messages=oj.conversation(220), config=cfg,
                          budgets=composer.Budgets(reserve=0, home=8192, tables=((8192, wide),)), model_window=8192)
        assert twice.refusal is None and _fold(twice), ("control: laid over a shadow", twice.refusal)
        assert lib.counters()["shadow"]["8192"] == counted + 1, ("one window, counted once", lib.counters()["shadow"])
        # A line even the whole Flesh room cannot hold: a descent asks the next rung up, by name.
        narrow = composer.Budget(window=4096, reserve=0, system=256, core=256, receipts=64, peels=64, flesh=150,
                                 tail=320, turn=332)
        assert narrow.flesh < charge, ("control: that Flesh room cannot hold a line", narrow.flesh, charge)
        pair = composer.Budgets(reserve=0, home=8192, tables=((4096, narrow), (8192, budgets.at(8192))))
        asked = oj.plan(lib, "c1s", ask, wrap=wrap, messages=oj.conversation(60), config=cfg, budgets=pair,
                        model_window=8192)
        assert asked.refusal is None and asked.rung == 8192, asked
        refused = lib.compose_window(asked, num_ctx=4096)
        assert refused.refusal == lib.Refusal("rung", (4096, 8192)), (
            "no turn set aside without its line: the next rung up, asked by name", refused.refusal)
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW9 -- a conversation unseen, caught up at its plan
# ---------------------------------------------------------------------------
def test_jw9_a_conversation_unseen_is_caught_up_at_its_plan_with_no_model():
    loaded, restore = oj.open_window()
    try:
        lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
        wrap = loaded[oj.WRAPPER].wrap
        cfg, journal, budgets = oj.config(lib), composer.load_journal(), composer.load_budgets()
        lib._ask = lib.registry_summarizer = lib.registry_reasker = lib._resolve_through_registry = _no_model
        stored = oj.conversation(300)
        ask = "What did Alice review?"
        cap = composer.layer_cap(budgets.at(budgets.home), "flesh")
        made, read = lib._turn_cost, {}

        def costing(counted, wrap_, overhead):
            cost = made(counted, wrap_, overhead)

            def counting(turn):
                read[turn.get("turn_id")] = read.get(turn.get("turn_id"), 0) + 1
                return cost(turn)

            return counting

        lib._turn_cost = costing
        try:
            plan = oj.plan(lib, "c9", ask, wrap=wrap, messages=stored, config=cfg, model_window=32768)
        finally:
            lib._turn_cost = made
        assert plan.refusal is None and plan.rung == budgets.home, (plan.refusal, plan.rung)
        assert len(read) == len(stored) and set(read.values()) == {1}, (
            "the catch-up costs each turn once", len(read), max(read.values()))
        state = lib.peek_state("c9", cfg)
        assert state is not None, "the plan made the conversation's state"
        held = _held_texts(state)
        assert held == _texts(stored), ("the receipts' spans and the Flesh are the turns, each once", len(held))
        cost = _flesh_cost(lib, state, wrap, journal)
        assert 0 < cost <= cap, ("the Flesh within its cap at the conversation's own rung", cost, cap)
        assert len(state.ledger.all()) > 0, "control: spans left the Flesh"
        caught = lib.counters()["catch_up"]
        assert caught.get("bare", 0) + caught.get("steps", 0) > 0 and "shadowed" not in caught, caught
        assert "turn_time" not in caught, ("a conversation unseen is no catch-up between two windows", caught)
        before = _digests(state)
        win = lib.compose_window(plan)
        assert win.refusal is None and dict(win.layers)["flesh"] == cost, (win.refusal, win.layers)
        assert _digests(state) == before, "the composition evicts nothing more"
        assert "provenance=shadow:" not in _last(win), "control: a Flesh caught up needs no shadow"
        # With no step of the queue allowed, bare spans still bring the Flesh within its cap.
        long = [oj.typed(f"{ask} " + " ".join(["service"] * 400)) if i % 2 == 0 else oj.answer("ok") for i in range(40)]
        assert sum(oj.count(message["content"]) + journal.message_overhead for message in long) > cap, (
            "control: the Flesh is over its cap")
        stepless = oj.plan(lib, "c9b", ask, wrap=wrap, messages=long, config=cfg,
                           journal=replace(journal, catch_up_steps=0))
        state = lib.peek_state("c9b", cfg)
        assert stepless.refusal is None and _flesh_cost(lib, state, wrap, journal) <= cap, (
            "bare spans bring it within its cap", stepless.refusal, _flesh_cost(lib, state, wrap, journal), cap)
        assert _held_texts(state) == _texts(long), "every turn held once"
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW12b -- the Flesh counted as laid wherever its fit is read
# ---------------------------------------------------------------------------
def test_jw12b_the_flesh_is_counted_as_laid_wherever_its_fit_is_read():
    loaded, restore = oj.open_window()
    try:
        lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
        wrap = loaded[oj.WRAPPER].wrap
        cfg, journal = oj.config(lib), composer.load_journal()
        gate, ladder = loaded[oj.PEELS].load_gate(), loaded[oj.PEELS].load_ladder()
        stored = oj.conversation(20)
        raw = sum(oj.count(message["content"]) for message in stored)
        laid = raw + len(stored) * journal.message_overhead
        tight = composer.Budget(window=4096, reserve=0, system=256, core=256, receipts=64, peels=64,
                                flesh=(raw + laid) // 2, tail=320, turn=332)
        budgets = composer.Budgets(reserve=0, home=4096, tables=((4096, tight),))
        assert raw <= composer.layer_cap(tight, "flesh") < laid, ("control: its own count fits, its laid cost not",
                                                                  raw, laid)
        win = oj.window(loaded, "c12", "What did Alice review?", messages=stored, config=cfg, budgets=budgets,
                        model_window=4096)
        assert win.refusal is None, win.refusal
        assert lib.counters()["catch_up"].get("steps", 0) >= 1, (
            "the plan's catch-up runs the queue at the laid cost", lib.counters()["catch_up"])
        state = lib.state_for("c12", cfg)
        cost = state.journal.burst[2]
        with state.lock:
            read = sum(cost(turn) for turn in state.flesh.turns())
        assert dict(win.layers)["flesh"] == read <= composer.layer_cap(tight, "flesh"), (win.layers, read)
        fresh, bare = lib.state_for("c12-fresh", cfg), lib.state_for("c12-raw", cfg)
        fresh.mirror(stored)
        bare.mirror(stored)
        moved = lib.curate(fresh, None, gate=gate, budget=tight, ladder=ladder, cost=cost)
        kept = lib.curate(bare, None, gate=gate, budget=tight, ladder=ladder, estimate=oj.count)
        assert moved.evicted, ("the queue given the laid cost evicts", moved.reason)
        assert not kept.evicted, ("control: its own count fits its cap", kept.reason)
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW17 -- a plan shares the writer with the burst
# ---------------------------------------------------------------------------
def test_jw17_a_plan_shares_the_writer_with_the_burst():
    loaded, restore = oj.open_window()
    try:
        lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
        wrap = loaded[oj.WRAPPER].wrap
        cfg, journal, budgets = oj.config(lib, max_steps_per_burst=2), composer.load_journal(), composer.load_budgets()
        gate, ladder = loaded[oj.PEELS].load_gate(), loaded[oj.PEELS].load_ladder()
        stored = oj.conversation(300)
        state = lib.state_for("c17", cfg)
        state.mirror(stored)
        started, release, steps = threading.Event(), threading.Event(), []

        def writing(turns):
            started.set()
            release.wait(10)
            return " ".join(str(turn.get("text", "")) for turn in turns)

        burst = threading.Thread(target=lambda: steps.append(lib._curation_burst(
            "c17", config=cfg, summarize=writing, gate=gate, ladder=ladder, budget=budgets.at(8192))), daemon=True)
        burst.start()
        try:
            assert started.wait(10), "control: the burst asked its model"
            before = _digests(state)
            plan = _plan_aside(lib, state, "c17", "What did Alice review?", wrap=wrap, messages=stored, config=cfg)
            assert plan.refusal is None, plan.refusal
            assert _digests(state) == before, "with the slot held, the plan evicts nothing"
            assert lib.counters()["catch_up"].get("shadowed", 0) >= 1, lib.counters()
            win = lib.compose_window(plan)
            assert win.refusal is None and _fold(win), "the overflow laid over a shadow"
            assert dict(win.layers)["flesh"] <= composer.layer_cap(budgets.at(plan.rung), "flesh"), win.layers
            assert _digests(state) == before, "nor does the composition"
        finally:
            release.set()
        burst.join(10)
        assert not burst.is_alive() and steps and steps[0] >= 1, ("the burst commits its span, not stale", steps)
        assert _held_texts(state) == _texts(stored), "no turn under two receipts, none lost"
        # A plan that caught up returns holding neither the slot nor the state's lock: between it and its
        # composition -- where the turn is admitted -- a burst runs to its end.
        other = lib.state_for("c17b", cfg)
        assert lib.counters()["catch_up"].get("bare", 0) == 0, "control: no plan caught up yet"
        planned = oj.plan(lib, "c17b", "What did Alice review?", wrap=wrap, messages=oj.conversation(300), config=cfg)
        assert planned.refusal is None, planned.refusal
        assert lib.counters()["catch_up"].get("bare", 0) > 0, "control: the plan caught up, under the slot"
        assert other.slot.acquire(blocking=False), "a plan that caught up returns without its slot"
        other.slot.release()
        caught = len(other.ledger.all())
        between = threading.Thread(target=lambda: lib._curation_burst(
            "c17b", config=cfg, summarize=writing, gate=gate, ladder=ladder, budget=budgets.at(4096)), daemon=True)
        between.start()
        between.join(5)
        assert not between.is_alive() and len(other.ledger.all()) > caught, (
            "a plan that caught up returns without the state's lock: a burst runs to its end", caught)
        composed = lib.compose_window(planned)
        assert composed.refusal is None and other.journal is not None, composed.refusal
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW20 -- a burst fires at the journal's mark
# ---------------------------------------------------------------------------
def test_jw20_a_burst_fires_at_the_journals_mark_and_evicts_down_to_it():
    loaded, restore = oj.open_window()
    try:
        lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
        wrap = loaded[oj.WRAPPER].wrap
        cfg, journal, budgets = oj.config(lib), composer.load_journal(), composer.load_budgets()
        sticky = budgets.at(budgets.home)
        mark = composer.layer_cap(composer.at_mark(sticky, journal.burst_mark), "flesh")
        stored = oj.conversation(160)
        fired = []

        def runner(conversation_id):
            fired.append(conversation_id)

        def refusing(_turns):
            raise RuntimeError("no model in this test")

        for cid in ("c20", "c20-none"):
            assert lib.maybe_curate(cid, stored, config=cfg, runner=runner), "control: the first turns fire by growth"
        win = oj.window(loaded, "c20", "What did Alice review?", messages=stored, config=cfg)
        assert win.refusal is None and win.rung == budgets.home, win.refusal
        state = lib.state_for("c20", cfg)
        grown = stored + [oj.typed("Turn 161: did Alice review service 161?")]
        assert mark < _flesh_cost(lib, state, wrap, journal) + oj.count(grown[-1]["content"]) + \
            journal.message_overhead <= composer.layer_cap(sticky, "flesh"), "control: within its cap, past the mark"
        fired.clear()
        assert lib.maybe_curate("c20", grown, config=cfg, runner=runner) and fired == ["c20"], (
            "one turn grown, past the mark: a burst fires", fired)
        assert lib.counters()["burst"]["marked"] == 1, lib.counters()["burst"]
        lib._curation_burst("c20", config=cfg, summarize=refusing)
        assert _flesh_cost(lib, state, wrap, journal) <= mark, ("the burst evicts down to the mark",
                                                                _flesh_cost(lib, state, wrap, journal), mark)
        assert _held_texts(state) == _texts(grown), "control: every turn held once"
        fired.clear()
        assert not lib.maybe_curate("c20-none", grown, config=cfg, runner=runner) and fired == [], (
            "no journal laid the conversation: one turn grown fires nothing", fired)
        # Under the mark, a conversation a journal laid keeps the growth rule: one turn grown fires nothing.
        low = oj.conversation(60)
        assert lib.maybe_curate("c20-low", low, config=cfg, runner=runner), "control: the first turns fire by growth"
        laid = oj.window(loaded, "c20-low", "What did Alice review?", messages=low, config=cfg)
        quiet = lib.state_for("c20-low", cfg)
        assert laid.refusal is None and quiet.journal is not None, ("control: a journal laid it", laid.refusal)
        more = low + [oj.typed("Turn 61: did Alice review service 61?")]
        assert _flesh_cost(lib, quiet, wrap, journal) + oj.count(more[-1]["content"]) + \
            journal.message_overhead <= mark, "control: under the mark"
        fired.clear()
        assert not lib.maybe_curate("c20-low", more, config=cfg, runner=runner) and fired == [], (
            "under the mark, one turn grown fires nothing", fired)
        assert lib.counters()["burst"]["marked"] == 1, "control: counted once"
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW24 -- a Flesh that does not fit, over a shadow or caught up (with JW6a and JW9, in JW23's place)
# ---------------------------------------------------------------------------
def test_jw24_a_flesh_that_does_not_fit_is_laid_over_a_shadow_or_caught_up():
    loaded, restore = oj.open_window()
    try:
        lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
        wrap = loaded[oj.WRAPPER].wrap
        cfg, journal, budgets = oj.config(lib), composer.load_journal(), composer.load_budgets()
        ladder = loaded[oj.PEELS].load_ladder()
        ask = "What did Alice review?"
        cap = composer.layer_cap(budgets.at(budgets.home), "flesh")
        # Over its cap with the slot held: laid over a shadow, the state untouched; the line naming the turns set
        # aside comes first among the memory, before the memory the plan brings (here a span the user recalls).
        stored = oj.conversation(300)
        state = lib.state_for("c24a", cfg)
        state.mirror(stored)
        recalled = oj.bare(state, 2)
        state.slot.acquire()
        try:
            plan = _plan_aside(lib, state, "c24a", ask, wrap=wrap, messages=stored, config=cfg,
                               recalled=(recalled.key,))
            before = _digests(state)
            held = lib.compose_window(plan)
        finally:
            state.slot.release()
        assert held.refusal is None and dict(held.layers)["flesh"] <= cap, (held.refusal, held.layers)
        last = _last(held)
        assert _fold(held) and "layer=recalled" in last, "the line and the recall in the turn's message"
        assert last.index("provenance=shadow:receipts") < last.index("layer=recalled"), (
            "the line first among the memory, before the plan's", last.index("provenance=shadow:receipts"),
            last.index("layer=recalled"))
        assert _digests(state) == before and len(before[0]) == 298, "the state untouched, 298 turns in its Flesh"
        # Turns the shadow's queue makes peels of: its peels follow the memory the plan brings.
        long = [oj.typed(f"{ask} " + " ".join(["service"] * 400)) if i % 2 == 0 else oj.answer("ok") for i in range(40)]
        opened = oj.conversation(2) + long[:38]
        state = lib.state_for("c24p", cfg)
        state.mirror(opened)
        recalled = oj.bare(state, 2)
        state.slot.acquire()
        try:
            plan = _plan_aside(lib, state, "c24p", ask, wrap=wrap, messages=opened, config=cfg,
                               recalled=(recalled.key,))
            peeled = lib.compose_window(plan)
        finally:
            state.slot.release()
        last = _last(peeled) if peeled.refusal is None else ""
        assert "provenance=shadow:peel" in last and "layer=recalled" in last, (
            "control: the shadow made peels, the recall laid", peeled.refusal, peeled.cut)
        assert (last.index("provenance=shadow:receipts") < last.index("layer=recalled")
                < last.index("provenance=shadow:peel")), "the line, the memory the plan brings, then the shadow's peels"
        # Over its cap with the slot free: caught up at the plan.
        cost = sum(oj.count(message["content"]) + journal.message_overhead for message in long)
        assert cost > cap, ("control: the Flesh is over its cap", cost, cap)
        caught = oj.window(loaded, "c24b", ask, messages=long, config=cfg)
        state = lib.state_for("c24b", cfg)
        assert caught.refusal is None and _flesh_cost(lib, state, wrap, journal) <= cap, caught.refusal
        assert _held_texts(state) == _texts(long), "every turn held once, the oldest in the Cellar"
        # Within its cap, over the room the answer's floor leaves it: laid over a shadow.
        tight = composer.Budget(window=4096, reserve=0, system=256, core=256, receipts=64, peels=64, flesh=2800,
                                tail=320, turn=332)
        packed = [oj.typed(" ".join(["service"] * 300)) if i % 2 == 0 else oj.answer("ok") for i in range(8)]
        whole = sum(oj.count(message["content"]) + journal.message_overhead for message in packed)
        assert whole <= composer.layer_cap(tight, "flesh"), ("control: within its cap", whole)
        floor = oj.window(loaded, "c24c", "Which services stay?", messages=packed, config=cfg,
                          budgets=composer.Budgets(reserve=0, home=4096, tables=((4096, tight),)), model_window=4096)
        assert floor.refusal is None and floor.accounted + floor.num_predict == 4096, floor.refusal
        assert floor.num_predict >= journal.answer_floor and dict(floor.layers)["flesh"] < whole, floor.layers
        state = lib.state_for("c24c", cfg)
        assert len(state.flesh.turns()) == 8 and not state.ledger.all(), "the floor's room taken over a shadow only"
        # Over what a turn's image leaves of its cap: laid at once over a shadow.
        reduced = composer.layer_cap(budgets.at(8192), "flesh", journal.image_tokens)
        asked = dict(messages=oj.conversation(60), config=cfg, budgets=oj.budgets_at_home(loaded, 8192),
                     journal=journal, ladder=ladder)
        plain = oj.window(loaded, "c24d", "Look.", **asked)
        history = dict(plain.layers)["flesh"]
        assert plain.refusal is None and reduced < history <= composer.layer_cap(budgets.at(8192), "flesh"), (
            "control: a history within the Flesh's cap, over what an image leaves of it", history, reduced)
        seen = oj.window(loaded, "c24d", oj.typed("Look.", images=1), **asked)
        assert seen.refusal is None and seen.rung == 8192, seen.refusal
        assert dict(seen.layers)["flesh"] <= reduced and dict(seen.layers)["images"] == journal.image_tokens, (
            seen.layers)
        state = lib.state_for("c24d", cfg)
        assert len(state.flesh.turns()) == 60 and not state.ledger.all(), "the image's room taken over a shadow only"
        # A turn with no conversation whose images pass the whole Flesh cap of a descent: refused document.
        smallest = composer.layer_cap(budgets.at(4096), "flesh")
        assert journal.image_tokens > smallest, ("control: an image over the smallest rung's Flesh cap", smallest)
        alone = oj.plan(lib, None, oj.typed("Look.", images=1), wrap=wrap, config=cfg)
        assert alone.refusal is None and alone.rung > 4096, (alone.refusal, alone.rung)
        looked = lib.compose_window(alone, num_ctx=4096)
        assert looked.refusal == lib.Refusal("document", (journal.image_tokens, smallest)), (
            "images the Flesh cannot hold are refused by name", looked.refusal)
        # The epoch a shadow begins: a window laid whole, then one over a shadow with the slot held, reads
        # eviction; a take-back under the held shadow reads take_back.
        whole = oj.window(loaded, "c24e", ask, messages=oj.conversation(160), config=cfg)
        assert whole.refusal is None and not _fold(whole), ("control: laid whole", whole.refusal)
        state = lib.state_for("c24e", cfg)
        state.slot.acquire()
        try:
            over = lib.compose_window(_plan_aside(lib, state, "c24e", ask, wrap=wrap,
                                                  messages=oj.conversation(200), config=cfg))
            taken = lib.compose_window(_plan_aside(lib, state, "c24e", ask, wrap=wrap,
                                                   messages=oj.conversation(198), config=cfg))
        finally:
            state.slot.release()
        assert over.refusal is None and _fold(over) and over.motive == "eviction", (
            "turns set aside over a shadow: an eviction", over.refusal, over.motive)
        assert taken.refusal is None and _fold(taken) and taken.motive == "take_back", (
            "a take-back under a shadow: a take-back", taken.refusal, taken.motive)
        assert not state.ledger.all(), "control: nothing evicted from the state"
        # A line even the whole Flesh room cannot hold, at the conversation's own rung: refused flesh, by name.
        narrow = composer.Budget(window=4096, reserve=0, system=256, core=256, receipts=64, peels=64, flesh=150,
                                 tail=320, turn=332)
        short = oj.conversation(60)
        state = lib.state_for("c24f", cfg)
        state.mirror(short)
        state.slot.acquire()
        try:
            plan = _plan_aside(lib, state, "c24f", ask, wrap=wrap, messages=short, config=cfg,
                               budgets=composer.Budgets(reserve=0, home=4096, tables=((4096, narrow),)),
                               model_window=4096)
            refused = lib.compose_window(plan)
        finally:
            state.slot.release()
        assert plan.refusal is None and refused.refusal is not None, (plan.refusal, "control: a Flesh over its cap")
        assert refused.refusal.motive == "flesh" and refused.refusal.figures[1] == narrow.flesh \
            < refused.refusal.figures[0], ("no turn set aside without its line: refused by name", refused.refusal)
        # A header split across two user turns: laid, the second loses the header's rest where they meet, and the
        # Flesh passes the sum of its turns. With the Flesh cap at that sum, the Flesh gives again over a shadow
        # instead of being refused flesh: at the conversation's own rung with the slot held, and at a descent.
        split = oj.conversation(8) + [oj.typed("Note this, it was pinned by the"), oj.typed("user: keep the old racks")]
        state = lib.state_for("c24j", cfg)
        state.mirror(split)
        summed = _flesh_cost(lib, state, wrap, journal)
        roomy = oj.window(loaded, "c24j", ask, messages=split, config=cfg)
        assert roomy.refusal is None and dict(roomy.layers)["flesh"] > summed, (
            "control: laid, the Flesh passes the sum of its turns", roomy.refusal, roomy.layers, summed)
        exact = composer.Budget(window=4096, reserve=0, system=256, core=256, receipts=64, peels=64, flesh=summed,
                                tail=320, turn=332)
        state.slot.acquire()
        try:
            plan = _plan_aside(lib, state, "c24j", ask, wrap=wrap, messages=split, config=cfg,
                               budgets=composer.Budgets(reserve=0, home=4096, tables=((4096, exact),)),
                               model_window=4096)
            own = lib.compose_window(plan)
        finally:
            state.slot.release()
        assert own.refusal is None and dict(own.layers)["flesh"] <= summed and _fold(own), (
            "the Flesh gives again, at the conversation's own rung", own.refusal, own.layers, summed)
        pair = composer.Budgets(reserve=0, home=8192, tables=((4096, exact), (8192, budgets.at(8192))))
        plan = oj.plan(lib, "c24j", ask, wrap=wrap, messages=split, config=cfg, budgets=pair, model_window=8192)
        down = lib.compose_window(plan, num_ctx=4096)
        assert down.refusal is None and down.rung == 4096 and dict(down.layers)["flesh"] <= summed and _fold(down), (
            "the Flesh gives again, at a descent", down.refusal, down.layers, summed)
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW25 -- the tail's room by kind
# ---------------------------------------------------------------------------
def test_jw25_the_tail_room_of_each_kind_is_the_cap_a_window_cuts_it_to():
    loaded, restore = oj.open_window()
    try:
        lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
        journal, budgets = composer.load_journal(), composer.load_budgets()

        def caps(rung):
            tail = composer.layer_cap(budgets.at(rung), "tail")
            return {kind: int(share * tail) for kind, share in journal.shares().items()}

        assert lib.tail_room(32768) == caps(budgets.home), "the conversation's own rung, within the model's window"
        assert lib.tail_room() == caps(budgets.home) and lib.tail_room(8192) == caps(8192), lib.tail_room(8192)
        assert caps(8192) != caps(budgets.home), "control: the rungs give different rooms"
        assert lib.tail_room(1000) == {} and lib.tail_room("a window") == {}, "no rung within it: no room, no raise"
        room = lib.tail_room(32768)["rag"]
        item = " ".join(["cluster"] * 40)
        few = [item] * max(1, room // (2 * oj.count(item)))
        many = [item] * (4 * room // oj.count(item) + 4)
        within = oj.window(loaded, "c25", "Which cluster?", tail=(oj.rag(*few),), model_window=32768)
        over = oj.window(loaded, "c25", "Which cluster?", tail=(oj.rag(*many),), model_window=32768)
        assert within.refusal is None and dict(within.cut)["rag"] == 0, ("control: within the room, whole",
                                                                          within.cut)
        assert over.refusal is None and dict(over.cut)["rag"] > 0, ("past the room, cut", over.cut)
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
