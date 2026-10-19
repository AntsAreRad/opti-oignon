#!/usr/bin/env python3
"""Contracts for the peels the region lays that a question reaches, laid again in the tail.

The tail brought the peels the region of accepted peels left out, never one
it laid. A fact and the later one that updates it could then stand apart --
one summarised in a peel the region laid, at the head of the window, the
other the anchor of a held span, nearest the question -- and the memory was
laid in the order of time within the tail only. A peel the
region lays is now among the peels the question brings into the tail,
chosen by rank with the rest and laid in its place in time; the region
keeps it.

  * RA1 -- THE UPDATE NOT FARTHER THAN THE FACT: a fact held as an anchor,
    its update in a peel the region lays, both reached by the question: the
    tail lays the update after the fact, at 4096 and at 8192, and the region
    still lays it.
  * RA2 -- THE REGION AS IT WAS: every message before the turn's is the same
    whether the question reaches the peel the region lays or not; the region
    gives up nothing the tail lays again.
  * RA3 -- WHAT THE QUESTION DOES NOT REACH IS NOT LAID AGAIN: of two peels
    the region lays, only the one the question reaches comes into the tail.
  * RA4 -- CHOSEN BY RANK, WHEREVER IT IS SHOWN: peels the region left out
    and one it lays, all the same size, under a memory cap that leaves some
    of them out: the one the region lays is laid again when it holds more of
    the question's words, and left out when it holds fewer -- no priority
    either way for being shown.
  * RA5 -- THE FACT THE REGION LAYS BESIDE ITS UPDATE: a fact summarised in a
    peel the region lays, its update held as an anchor, both reached: the
    tail lays the fact before the update.
  * RA6 -- A COPY GIVES WAY BEFORE THE FLESH: a tool turn at its prompt's
    cap, where the copy of the peel the region lays does not fit beside the
    whole Flesh: the question that reaches it keeps the Flesh whole and
    every message before the turn's, the copy left out and not counted as
    cut -- the region still shows it.
  * RA7 -- A COPY LEFT OUT IS NOT CUT: the copy of a peel the region lays,
    kept back by the plan where the memory's cap holds its words but not
    its envelope, or left out by a window that cuts further at a descent,
    counts nothing as cut; the window's cut counts what it shows nowhere.

The onion's modules and the wrapper come from source through the shared
isolation window (``_onion_journal``); the counter is the journal suite's.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _onion_journal as oj  # noqa: E402

BUDGET_S = {
    "test_ra1_the_update_not_farther_than_the_fact": 5.0,
    "test_ra2_the_region_as_it_was": 5.0,
    "test_ra3_what_the_question_does_not_reach_is_not_laid_again": 5.0,
    "test_ra4_chosen_by_rank_wherever_it_is_shown": 10.0,
    "test_ra5_the_fact_the_region_lays_beside_its_update": 5.0,
    "test_ra6_a_copy_gives_way_before_the_flesh": 10.0,
    "test_ra7_a_copy_left_out_is_not_cut": 5.0,
}

_FACT = "Alice lives in Berlin."
_UPDATE = "News: the move is done, Alice lives in Oslo."
_PEEL = "Alice moved: she lives in Oslo now."
_QUESTION = "Where does Alice live now?"


def _fact_then_update(loaded, cid, cfg, peels=(_PEEL,), after=6):
    """A fact typed, then its update: the fact's span held, the fact its anchor; then the update's span and one
    more for each other summary, each accepted under the next of ``peels``."""
    lib = loaded[oj.LIBRARIAN]
    messages = [oj.typed(_FACT), oj.answer("Noted.")]
    for n, _peel in enumerate(peels):
        messages += [oj.typed(_UPDATE if n == 0 else f"Turn {n}: the rota for the depot is out."),
                     oj.answer("Noted.")]
    messages += oj.conversation(after, start=len(messages) + 1)
    state = lib.state_for(cid, cfg)
    state.mirror(messages)
    oj.hold(state, 2, anchors=((state.flesh.turns()[0]["turn_id"], 0, len(_FACT)),))
    marks = []
    for peel in peels:
        oj.accept(loaded, state, 2, peel)
        marks.append(f"provenance=peel:{_peel_id(state, peel)[:12]}")
    return state, messages, marks


def _peel_id(state, text):
    return next(peel.id for peel in state.tree.all() if peel.text == text)


# ---------------------------------------------------------------------------
# RA1 -- the update not farther than the fact
# ---------------------------------------------------------------------------
def test_ra1_the_update_not_farther_than_the_fact():
    """RA1: at 4096 and 8192, the fact held, its update in a peel the region lays: the tail lays the update after."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        _state, messages, (mark,) = _fact_then_update(loaded, "c1", cfg)
        for window in (4096, 8192):
            win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=window)
            assert win.refusal is None and dict(win.cut)["memory"] == 0, (window, win.refusal, win.cut)
            assert mark in win.messages[1][1], ("control: the region lays the update's peel", window)
            last = win.messages[-1][1]
            assert _FACT in last and mark in last, ("the fact and the update's peel, both in the tail", window)
            assert last.find(_FACT) < last.find(_PEEL), ("the update after the fact, nearest the question", window)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RA2 -- the region as it was
# ---------------------------------------------------------------------------
def test_ra2_the_region_as_it_was():
    """RA2: a question that reaches the region's peel and one that reaches none: every message before the turn's
    the same."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        _state, messages, (mark,) = _fact_then_update(loaded, "c1", cfg)
        for window in (4096, 8192):
            reached = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=window)
            elsewhere = oj.window(loaded, "c1", "Which train leaves for Rome tonight?", messages=messages,
                                  config=cfg, model_window=window)
            assert reached.refusal is None and elsewhere.refusal is None, (reached.refusal, elsewhere.refusal)
            assert mark in reached.messages[-1][1] and mark not in elsewhere.messages[-1][1], (
                "control: laid again where the question reaches it", window)
            assert reached.messages[:-1] == elsewhere.messages[:-1], ("every message before the turn's the same",
                                                                     window)
            assert mark in reached.messages[1][1], ("the region keeps it", window)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RA3 -- what the question does not reach is not laid again
# ---------------------------------------------------------------------------
def test_ra3_what_the_question_does_not_reach_is_not_laid_again():
    """RA3: two peels the region lays, one the question reaches: only that one in the tail."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        other = "The depot rota changes every week."
        _state, messages, (mark, unreached) = _fact_then_update(loaded, "c1", cfg, peels=(_PEEL, other))
        win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=8192)
        assert win.refusal is None, win.refusal
        assert mark in win.messages[1][1] and unreached in win.messages[1][1], "control: the region lays both"
        last = win.messages[-1][1]
        assert mark in last, "the peel the question reaches, laid again"
        assert unreached not in last and other not in last, "the one it does not reach, not"
    finally:
        restore()


# ---------------------------------------------------------------------------
# RA4 -- chosen by rank, wherever it is shown
# ---------------------------------------------------------------------------
def _crowded(loaded, cid, cfg, newest, others):
    """Same-size peels accepted in turn, ``newest`` last, under a peels' cap that lays the newest alone; the window
    at 8192 for a question every one of them answers in part: ``(region, tail, marks)``."""
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    assert {oj.count(text) for text in others} == {oj.count(newest)}, "control: all the same size"
    messages = []
    for n in range(len(others) + 1):
        messages += [oj.typed(f"Turn {n}: notes for the trip."), oj.answer("Noted.")]
    messages += oj.conversation(6, start=len(messages) + 1)
    state = lib.state_for(cid, cfg)
    state.mirror(messages)
    for text in others + [newest]:
        oj.accept(loaded, state, 2, text)
    marks = {text: f"provenance=peel:{_peel_id(state, text)[:12]}" for text in others + [newest]}
    shipped = composer.load_budgets()
    at = shipped.at(8192)
    room = oj.count(newest) + oj.count(newest) // 2
    narrow = composer.Budgets(reserve=shipped.reserve, home=8192, tables=(
        (8192, replace(at, peels=room, flesh=at.flesh + at.peels - room)),))
    win = oj.window(loaded, cid, "Where does Alice live now in Oslo?", messages=messages, config=cfg,
                    model_window=8192, budgets=narrow)
    assert win.refusal is None, win.refusal
    region, last = win.messages[1][1], win.messages[-1][1]
    assert marks[newest] in region and not any(marks[text] in region for text in others), (
        "control: the region lays the newest alone")
    laid = [text for text in others if marks[text] in last]
    assert 0 < len(laid) < len(others), ("control: the cap leaves some of the others out", len(laid))
    return region, last, marks


def test_ra4_chosen_by_rank_wherever_it_is_shown():
    """RA4: same-size peels, a cap that leaves some of them out: the region's laid again when it holds more of the
    question's words, left out when it holds fewer."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        others = [f"Alice was in Oslo on day {n:02d} of the trip." for n in range(1, 41)]
        more = "Alice lives in Oslo now, as she told us all."
        _region, last, marks = _crowded(loaded, "c1", cfg, more, others)
        assert marks[more] in last, "the region's peel, ranking first, laid again"
        fewer = "Alice sent us a card from the trip, as promised."
        _region, last, marks = _crowded(loaded, "c2", cfg, fewer, others)
        assert marks[fewer] not in last, "the region's peel, ranking last, left out: no priority for being shown"
    finally:
        restore()


# ---------------------------------------------------------------------------
# RA5 -- the fact the region lays beside its update
# ---------------------------------------------------------------------------
def test_ra5_the_fact_the_region_lays_beside_its_update():
    """RA5: at 4096 and 8192, the fact in a peel the region lays, its update held: the tail lays the fact first."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        summary = "The user said Alice lives in Berlin."
        messages = [oj.typed(_FACT), oj.answer("Noted."), oj.typed(_UPDATE), oj.answer("Noted.")]
        messages += oj.conversation(6, start=5)
        state = lib.state_for("c1", cfg)
        state.mirror(messages)
        oj.accept(loaded, state, 2, summary)
        oj.hold(state, 2, anchors=((state.flesh.turns()[0]["turn_id"], 0, len(_UPDATE)),))
        mark = f"provenance=peel:{_peel_id(state, summary)[:12]}"
        for window in (4096, 8192):
            win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=window)
            assert win.refusal is None and dict(win.cut)["memory"] == 0, (window, win.refusal, win.cut)
            assert mark in win.messages[1][1], ("control: the region lays the fact's peel", window)
            last = win.messages[-1][1]
            assert mark in last and _UPDATE in last, ("the fact's peel and the update, both in the tail", window)
            assert last.find(summary) < last.find(_UPDATE), ("the fact before its update, in the order typed", window)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RA6 -- a copy gives way before the Flesh
# ---------------------------------------------------------------------------
def test_ra6_a_copy_gives_way_before_the_flesh():
    """RA6: a tool turn at 8192 at its prompt's cap: the question that reaches the region's peel keeps the Flesh
    whole and the prefix, its copy left out, nothing counted as cut."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        key = " ".join(["Alice keeps the spare key of the depot in the blue box."] * 5)
        messages = oj.conversation(4) + oj.conversation(79, start=5)
        state = lib.state_for("c1", cfg)
        state.mirror(messages)
        oj.accept(loaded, state, 2, key)
        oj.accept(loaded, state, 2, " ".join(["The rota for the night shift changes on Mondays."] * 45))
        lib.pin("c1", " ".join(["Answers cite."] * 40), actor="user", config=cfg)
        mark = f"provenance=peel:{_peel_id(state, key)[:12]}"
        tool = dict(config=cfg, model_window=8192, system=(" ".join(["rule"] * 345),),
                    tools=('{"name": "lookup", "description": "look a word up"}',))
        journal, budgets = composer.load_journal(), composer.load_budgets()
        cap = composer.prompt_cap(budgets.at(8192), 8192 - composer.at_rung(journal.tool_room, 8192, budgets.home),
                                  journal.margin, journal.answer_floor)
        roomy = [oj.window(loaded, "c1", q, messages=messages, **tool)
                 for q in ("What is the weather like?", "Where is the spare key of the depot?")]
        assert all(w.refusal is None for w in roomy) and mark in roomy[1].messages[-1][1], (
            "control: with room, the copy is laid", [w.refusal for w in roomy])
        copy = roomy[1].tokens - roomy[0].tokens
        padding = " " + " ".join(["please"] * 150)
        elsewhere = oj.window(loaded, "c1", "What is the weather like?" + padding, messages=messages, **tool)
        assert elsewhere.refusal is None and "shadow:receipts" not in elsewhere.messages[-1][1], (
            "control: the question that reaches nothing keeps the Flesh whole", elsewhere.refusal)
        assert elsewhere.tokens + copy > cap >= elsewhere.tokens, (
            "control: at the cap, the copy does not fit beside the whole Flesh", elsewhere.tokens, copy, cap)
        reached = oj.window(loaded, "c1", "Where is the spare key of the depot?" + padding, messages=messages, **tool)
        assert reached.refusal is None and reached.motive is None, ("the stable prefix kept", reached.motive)
        assert reached.messages[:-1] == elsewhere.messages[:-1], "every message before the turn's the same"
        last = reached.messages[-1][1]
        assert "shadow:receipts" not in last and mark not in last, "the Flesh whole, the copy left out"
        assert dict(reached.cut)["memory"] == 0, ("a copy that gives way is not cut", reached.cut)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RA7 -- a copy left out is not cut
# ---------------------------------------------------------------------------
def test_ra7_a_copy_left_out_is_not_cut():
    """RA7: the copy kept back by the plan, or left out at a descent, while the region shows it: nothing cut."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        peel = " ".join(["Alice keeps the spare key of the depot in the blue box."] * 7)
        question = "Where does Alice keep the spare key of the depot?"
        messages = oj.conversation(2) + oj.conversation(6, start=3)
        state = lib.state_for("c1", cfg)
        state.mirror(messages)
        oj.accept(loaded, state, 2, peel)
        mark = f"provenance=peel:{_peel_id(state, peel)[:12]}"
        # The plan: a peels' cap the peel fills, a tail whose memory cap holds its words but not its envelope.
        shipped = composer.load_budgets()
        at = shipped.at(8192)
        size = oj.count(peel)
        tight = composer.Budgets(reserve=shipped.reserve, home=8192, tables=(
            (8192, replace(at, peels=size, tail=size + 100, flesh=at.flesh + at.peels - size + at.tail - size - 100)),))
        _own, whole = composer.memory_cap(tight.at(8192), composer.load_journal().shares(), set(), size)
        assert size <= whole, ("control: the memory's cap holds the copy's words, the plan chooses it", size, whole)
        win = oj.window(loaded, "c1", question, messages=messages, config=cfg, model_window=8192, budgets=tight)
        assert win.refusal is None and mark in win.messages[1][1], ("control: the region lays the peel", win.refusal)
        assert mark not in win.messages[-1][1], "control: the plan keeps the copy back"
        assert dict(win.cut)["memory"] == 0, ("the copy the plan kept back, not cut", win.cut)
        # A descent: planned at 8192 with the copy, sent at 4096, where the memory's room no longer holds it.
        planned = oj.plan(lib, "c1", question, wrap=loaded[oj.WRAPPER].wrap, messages=messages, config=cfg,
                          model_window=8192)
        assert planned.rung == 8192 and any(f"provenance={item[1]}".startswith(mark) for item in planned._held.tail), (
            "control: the copy handed at 8192")
        sent = lib.compose_window(planned, num_ctx=4096)
        assert sent.refusal is None and mark in sent.messages[1][1], ("control: the region at 4096 lays it",
                                                                     sent.refusal)
        assert mark not in sent.messages[-1][1], "control: the window at 4096 leaves the copy out"
        assert dict(sent.cut)["memory"] == 0, ("the copy left out at the descent, not cut", sent.cut)
    finally:
        restore()
