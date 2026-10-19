#!/usr/bin/env python3
"""Contracts for the order in which the memory a question brings is laid.

The plan chose the memory by rank -- the share of the question's terms an
item holds, then the newest receipt first -- and the window laid it in that
order. A fact and the later one that updates it could then stand the
stale one last, nearest the question: on the drift A/B a 4B model read
"Alice lives in Berlin." after "Alice lives in Oslo." and answered Berlin.
The memory is now chosen by rank and laid in the order of time, the newest
nearest the question.

  * MT1 -- A FACT AND THE ONE THAT UPDATES IT, IN THE ORDER TYPED: both
    reached, the later ranking first, the window lays the earlier first.
  * MT2 -- CHOSEN BY RANK, LAID BY TIME: under a cap that binds, the items
    laid are those the rank keeps -- the newest, where all rank alike --
    and they are laid oldest first; ordering by time never changes which.
  * MT3 -- THE COMPOSER LAYS ALL THE PLAN HANDS, IN THE ORDER OF TIME: what
    the plan kept back is the window's cut, nothing more; the plan hands the
    memory in the order it chose it, each item saying where it stands in
    time, and the window lays it in that order.
  * MT4 -- THE MEMORY BLOCK IN THE ORDER OF TIME TOO: the block the
    executor places lays a fact before the later one that updates it, and
    a held fact before a later peel.
  * MT5 -- A PEEL STANDS AT ITS RECEIPT'S PLACE: a peel and an anchor are
    laid in the order of their receipts in the ledger, whichever is older;
    a peel that stands for several, at the newest.
  * MT6 -- A RECALLED SPAN STAYS LAST: after the memory laid in time.
  * MT7 -- A WINDOW THAT CUTS FURTHER LEAVES OUT THE LAST CHOSEN: sent under
    a smaller context than its plan's rung, the window keeps the fact that
    ranks first and is the newest, and leaves out the stale one that ranks
    last -- never the newest for being the newest.
  * MT8 -- CHOSEN BY RANK, WHATEVER ITS AGE: under a cap that binds, the
    fact that ranks first is laid though it is the oldest.
  * MT9 -- WITHIN A SPAN, IN THE ORDER TYPED: two anchors of one held span
    are laid in their order in it, the later ranking first.
  * MT10 -- THE ORDER OF TIME NEVER LAYS LESS: under a count that reads the
    whole, where laying the memory in the order of time would pass the cap
    with all of it, it is laid in the order chosen, none left out; what it
    wants is read from the tries that choose; an item whose time is not
    known comes first.

The onion's modules and the wrapper come from source through the shared
isolation window (``_onion_journal``); the counter is the journal suite's.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _onion_journal as oj  # noqa: E402

BUDGET_S = {
    "test_mt1_a_fact_and_the_one_that_updates_it_in_the_order_typed": 5.0,
    "test_mt2_chosen_by_rank_laid_by_time": 5.0,
    "test_mt3_the_composer_lays_all_the_plan_hands_in_the_order_of_time": 5.0,
    "test_mt4_the_memory_block_in_the_order_of_time_too": 5.0,
    "test_mt5_a_peel_stands_at_its_receipts_place": 5.0,
    "test_mt6_a_recalled_span_stays_last": 5.0,
    "test_mt7_a_window_that_cuts_further_leaves_out_the_last_chosen": 5.0,
    "test_mt8_chosen_by_rank_whatever_its_age": 5.0,
    "test_mt9_within_a_span_in_the_order_typed": 5.0,
    "test_mt10_the_order_of_time_never_lays_less": 5.0,
}

_OLD = "Alice lives in Berlin."
_NEW = "Alice lives in Oslo now."
_PEEL = "Alice said she lives near the harbour in Oslo now."
_QUESTION = "Where does Alice live now?"


def _held_facts(lib, cid, facts, cfg, after=6):
    """A conversation whose first spans, each a typed fact and an answer, were held, each fact its anchor."""
    messages = []
    for fact in facts:
        messages += [oj.typed(fact), oj.answer("Noted.")]
    messages += oj.conversation(after, start=len(messages) + 1)
    state = lib.state_for(cid, cfg)
    state.mirror(messages)
    for fact in facts:
        oj.hold(state, 2, anchors=((state.flesh.turns()[0]["turn_id"], 0, len(fact)),))
    return state, messages


def _order(text, parts):
    """Where each of ``parts`` stands in ``text``, all of them found."""
    places = [text.find(part) for part in parts]
    assert all(place >= 0 for place in places), ("each laid", parts, places)
    return places


# ---------------------------------------------------------------------------
# MT1 -- a fact and the one that updates it, in the order typed
# ---------------------------------------------------------------------------
def test_mt1_a_fact_and_the_one_that_updates_it_in_the_order_typed():
    """MT1: the later fact ranks first; the window lays the earlier first, at 4096 and at 8192."""
    loaded, restore = oj.open_window()
    lib, peels = loaded[oj.LIBRARIAN], loaded[oj.PEELS]
    try:
        cfg = oj.config(lib)
        state, messages = _held_facts(lib, "c1", [_OLD, _NEW], cfg)
        ranked = [item.text for item in peels.select_anchors(state.ledger, state.cellar, _QUESTION, 768,
                                                             estimate=oj.count)]
        assert ranked == [_NEW, _OLD], ("control: the later fact ranks first", ranked)
        for window in (4096, 8192):
            win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=window)
            assert win.refusal is None and dict(win.cut)["memory"] == 0, (window, win.refusal, win.cut)
            old, new = _order(win.messages[-1][1], [_OLD, _NEW])
            assert old < new, ("the later fact nearest the question", window)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MT2 -- chosen by rank, laid by time
# ---------------------------------------------------------------------------
def test_mt2_chosen_by_rank_laid_by_time():
    """MT2: facts that rank alike, a cap that binds: the newest laid, oldest first."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        facts = [f"Alice moved build {n} to Lisbon on 2026-03-{n:02d}." for n in range(1, 31)]
        _state, messages = _held_facts(lib, "c1", facts, cfg)
        # The parts laid whole fill their caps (``_onion_journal.snug``): the cap binds as written.
        question = "Where did Alice move the builds to Lisbon?"
        win = oj.window(loaded, "c1", question, messages=messages, config=cfg, model_window=4096,
                        budgets=oj.snug(loaded, question))
        assert win.refusal is None, win.refusal
        last = win.messages[-1][1]
        laid = [fact for fact in facts if fact in last]
        assert 0 < len(laid) < len(facts), ("control: the cap binds", len(laid))
        assert laid == facts[-len(laid):], ("the rank keeps the newest where all rank alike", laid)
        places = _order(last, laid)
        assert places == sorted(places), ("laid oldest first", laid)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MT3 -- the composer lays all the plan hands, in the order of time
# ---------------------------------------------------------------------------
def test_mt3_the_composer_lays_all_the_plan_hands_in_the_order_of_time():
    """MT3: at 4096 and at 8192, every memory item handed is laid, in the order of time; the cut is what was kept
    back."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        wrap = loaded[oj.WRAPPER].wrap
        facts = [f"Alice moved build {n} to Lisbon on 2026-03-{n % 28 + 1:02d}." for n in range(1, 61)]
        _state, messages = _held_facts(lib, "c1", facts + [_OLD, _NEW], cfg)
        for window in (4096, 8192):
            planned = oj.plan(lib, "c1", "Where does Alice live, and where did she move the builds?", wrap=wrap,
                              messages=messages, config=cfg, model_window=window)
            win = lib.compose_window(planned)
            assert win.refusal is None, (window, win.refusal)
            handed = [item for item in planned._held.tail if item[0] == "memory"]
            assert all(len(item) > 5 for item in handed), ("control: each says where it stands in time", window)
            places = _order(win.messages[-1][1], [item[2] for item in handed])
            in_time = sorted(range(len(handed)), key=lambda index: handed[index][5])
            assert [places[index] for index in in_time] == sorted(places), ("laid in the order of time", window)
            assert in_time != list(range(len(handed))), ("control: handed in another order, the one chosen", window)
            assert dict(win.cut)["memory"] == planned._held.trimmed, ("the cut is what was kept back", window)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MT4 -- the memory block in the order of time too
# ---------------------------------------------------------------------------
def test_mt4_the_memory_block_in_the_order_of_time_too():
    """MT4: the executor's memory block lays the earlier fact first, the later ranking first."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        _held_facts(lib, "c1", [_OLD, _NEW], cfg)
        block = str(lib.memory_block("c1", _QUESTION, config=cfg))
        old, new = _order(block, [_OLD, _NEW])
        assert old < new, ("the later fact last", block[-300:])
        messages = [oj.typed(_OLD), oj.answer("Noted.")] + oj.conversation(8, start=3)
        state = lib.state_for("c2", cfg)
        state.mirror(messages)
        oj.hold(state, 2, anchors=((state.flesh.turns()[0]["turn_id"], 0, len(_OLD)),))
        oj.accept(loaded, state, 2, _PEEL)
        block = str(lib.memory_block("c2", _QUESTION, config=cfg))
        fact, peel = _order(block, [_OLD, _PEEL])
        assert fact < peel, ("a held fact before a later peel", block[-300:])
    finally:
        restore()


# ---------------------------------------------------------------------------
# MT5 -- a peel stands at its receipt's place
# ---------------------------------------------------------------------------
def test_mt5_a_peel_stands_at_its_receipts_place():
    """MT5: a held fact, then a peel the region does not lay: the fact first; and the other way round."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        peel_text = _PEEL
        peels = loaded[oj.PEELS]
        assert peels.peel_time(SimpleNamespace(sources=("k1", "k2", "k3")), {"k1": 4, "k2": 9, "k3": 2}) == (9, 0), (
            "a peel that stands for several receipts, at the newest")
        filler = [" ".join([f"Bob checked queue {n} and it stays green on the staging mirror."] * 6) for n in range(2)]
        for cid, peel_first in (("c1", False), ("c2", True)):
            fact, other = [oj.typed(_OLD), oj.answer("Noted.")], oj.conversation(2, start=101)
            messages = (other + fact if peel_first else fact + other) + oj.conversation(14, start=3)
            state = lib.state_for(cid, cfg)
            state.mirror(messages)
            for step in ("peel", "fact") if peel_first else ("fact", "peel"):
                if step == "fact":
                    first = state.flesh.turns()[0]
                    assert first["text"] == _OLD, ("control: the fact's span is the oldest left", first["text"])
                    oj.hold(state, 2, anchors=((first["turn_id"], 0, len(_OLD)),))
                else:
                    oj.accept(loaded, state, 2, peel_text)
            for words in filler:
                oj.accept(loaded, state, 2, words)
            win = oj.window(loaded, cid, _QUESTION, messages=messages, config=cfg, model_window=4096)
            assert win.refusal is None, (cid, win.refusal)
            fact_at, peel_at = _order(win.messages[-1][1], [_OLD, peel_text])
            assert (fact_at > peel_at) == peel_first, ("the older first, peel or anchor", cid)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MT6 -- a recalled span stays last
# ---------------------------------------------------------------------------
def test_mt6_a_recalled_span_stays_last():
    """MT6: at 8192 a recalled span is laid after the memory laid in time."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        state, messages = _held_facts(lib, "c1", [_OLD, _NEW], cfg)
        recalled = oj.bare(state, 2)
        win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=8192,
                        recalled=(recalled.key,))
        assert win.refusal is None, win.refusal
        last = win.messages[-1][1]
        old, new, recall = _order(last, [_OLD, _NEW, "layer=recalled"])
        assert old < new < recall, "the recalled span after the memory laid in time"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MT7 -- a window that cuts further leaves out the last chosen
# ---------------------------------------------------------------------------
def test_mt7_a_window_that_cuts_further_leaves_out_the_last_chosen():
    """MT7: planned at 8192, sent at 4096: the newest fact, ranking first, laid; the stale one, ranking last, cut."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        old = "Alice lives in Berlin, in the flat on the long canal street near the old harbour market."
        new = "Alice lives in Oslo now, in the flat on the long fjord street near the new harbour market."
        filler = [f"Alice met Bob on day {n}." for n in range(1, 13)]
        _state, messages = _held_facts(lib, "c1", [old] + filler + [new], cfg, after=12)
        # The parts laid whole fill their caps (``_onion_journal.snug``): the cap binds as written.
        planned = oj.plan(lib, "c1", _QUESTION, wrap=loaded[oj.WRAPPER].wrap, messages=messages, config=cfg,
                          model_window=8192, budgets=oj.snug(loaded, _QUESTION))
        handed = [item[2] for item in planned._held.tail if item[0] == "memory"]
        assert planned.rung == 8192 and old in handed and new in handed, ("control: both handed at 8192", handed)
        win = lib.compose_window(planned, num_ctx=4096)
        assert win.refusal is None and win.num_ctx == 4096, win.refusal
        last = win.messages[-1][1]
        assert dict(win.cut)["memory"] > planned._held.trimmed, ("control: the window cuts further", win.cut)
        assert new in last, "the fact that ranks first and is the newest, laid"
        assert old not in last, "the stale fact, ranking last, left out first"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MT8 -- chosen by rank, whatever its age
# ---------------------------------------------------------------------------
def test_mt8_chosen_by_rank_whatever_its_age():
    """MT8: at 4096 the oldest fact, ranking first among thirty newer ones, laid, and laid first."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        filler = [f"Alice met Bob on day {n}." for n in range(1, 31)]
        _state, messages = _held_facts(lib, "c1", [_NEW] + filler, cfg)
        # The parts laid whole fill their caps (``_onion_journal.snug``): the cap binds as written.
        win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=4096,
                        budgets=oj.snug(loaded, _QUESTION))
        assert win.refusal is None, win.refusal
        last = win.messages[-1][1]
        laid = [fact for fact in filler if fact in last]
        assert 0 < len(laid) < len(filler), ("control: the cap binds", len(laid))
        assert _NEW in last, "the fact that ranks first, laid though the oldest"
        assert last.find(_NEW) < min(last.find(fact) for fact in laid), "and laid first, as the oldest"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MT9 -- within a span, in the order typed
# ---------------------------------------------------------------------------
def test_mt9_within_a_span_in_the_order_typed():
    """MT9: one typed turn holds a fact, then the one that updates it; both anchors laid, in that order."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        first, second = "Alice lives in Paris.", "Alice lives in Rome now."
        messages = [oj.typed(f"{first} {second}"), oj.answer("Noted.")] + oj.conversation(6, start=3)
        state = lib.state_for("c1", cfg)
        state.mirror(messages)
        turn = state.flesh.turns()[0]["turn_id"]
        oj.hold(state, 2, anchors=((turn, 0, len(first)), (turn, len(first) + 1, len(first) + 1 + len(second))))
        win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=8192)
        assert win.refusal is None, win.refusal
        earlier, later = _order(win.messages[-1][1], [first, second])
        assert earlier < later, "the anchors of one span in the order typed"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MT10 -- the order of time never lays less
# ---------------------------------------------------------------------------
def test_mt10_the_order_of_time_never_lays_less():
    """MT10: a count that reads the whole makes the order of time pass the cap: the order chosen, all laid; what
    it wants read from the tries that choose; an item whose time is not known first."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        wrap = loaded[oj.WRAPPER].wrap
        held = oj.plan(lib, None, _QUESTION, wrap=wrap)._held
        budget = replace(composer.load_budgets().at(8192), peels=0)

        def lead(text):
            """A count that reads the whole: a fifth more where the first code word it holds names rust."""
            found = [(text.find(word), word) for word in ("fn main", "def main") if word in text]
            base = oj.count(text)
            return base + base // 5 if found and min(found)[1] == "fn main" else base

        label = (("typed",), ())
        newest = ("memory", "a1", "Alice lives in Oslo now, where her def main runs the harbour loader for the whole "
                  "fleet of the new port every night of the week.", ("peels", "anchor:a1"), label, (2, 0))
        older = ("memory", "a0", "Alice lived in Berlin; fn main was hers.", ("peels", "anchor:a0"), label, (1, 0))
        chosen = (newest, older)

        def measured(items):
            return lead(composer.fit_kind("memory", items, 10 ** 6, wrap, lead)[2])

        rank = measured([item[:5] for item in chosen])
        assert measured([older[:5], newest[:5]]) > rank >= measured([older[:5]]), (
            "control: in the order of time the whole costs more, the newest tried last passing the cap")
        taken, _kept, placed, dropped = composer.fit_kind("memory", chosen, rank, wrap, lead)
        assert taken == [0, 1] and dropped == 0 and lead(placed) <= rank, (
            "all laid, in the order chosen, within the cap", taken, dropped)
        shares = dict(held.shares, memory=(rank - 30) / budget.tail)
        own = int(shares["memory"] * budget.tail)
        wants = composer.memory_wants(chosen, own, wrap, lead)
        assert wants == rank - own, ("what it wants, read from the tries that choose", wants, rank, own)

        def laid(lend):
            journal = composer.compose_journal(system=oj.SYSTEM, pinned=held.pinned, region=(), digest="", flesh=(),
                                               tail=chosen, turn=held.turn, budget=budget, count=lead,
                                               overhead=held.overhead, wrap=wrap, shares=shares, lend=lend)
            return dict(journal.cut)["memory"]

        assert laid(wants) == 0 < laid(wants - 1), "from what it wants on, all laid; below, not"
        unknown = ("memory", "u", "Alice keeps a boat.", ("peels", "anchor:u"), label, ())
        taken = composer.fit_kind("memory", (older, unknown), 10 ** 6, wrap, oj.count)[0]
        assert taken == [1, 0], ("an item whose time is not known first", taken)
    finally:
        restore()
