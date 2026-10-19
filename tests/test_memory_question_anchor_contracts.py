#!/usr/bin/env python3
"""Contracts for the place of a question the user typed among the memory a question brings.

A held span keeps, as anchors, the fewest units the user typed that answer
its probes, and a question the user typed answers an entity's probe by
naming it. Ranked by the share of the question's terms their words hold,
the earlier questions that a later one repeats ("Quick check: where does Bob
live?") outranked the fact that answers it ("Bob lives in Lyon.": "lives"
is not "live"), filled the memory's cap, and left the fact out. On the
drift A/B at 4096, the one probe a 27B model still missed was answered from
four earlier questions, two of their instructions and no fact.

  * QA1 -- THE FACT BEFORE THE QUESTIONS THAT ECHO IT: under a cap the
    earlier questions fill, the fact that answers them is laid.
  * QA2 -- A QUESTION STILL LAID WHERE ROOM REMAINS: it ranks after what
    states, it is not left out for it.
  * QA3 -- THE MEMORY BLOCK TOO: the block the executor places ranks a
    unit that only asks after the one that states.
  * QA4 -- WHAT ASKS IS READ AS THE PROBES READ IT: a unit that ends on a
    question mark, an exclamation mark or a bracket after it included,
    ranks after one that states, whatever its words hold.
  * QA5 -- A QUESTION THAT ORDERS IS AN ORDER: an order or a prohibition
    the user typed as a question keeps its rank, laid where the facts that
    name what it names would crowd a question out.
  * QA6 -- A WHOLE SEGMENT NEVER ONLY ASKS: a segment kept whole that states
    and then asks ranks as what it states, and the fact within it is shown
    once.

The onion's modules and the wrapper come from source through the shared
isolation window (``_onion_journal``); the counter is the journal suite's.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _onion_journal as oj  # noqa: E402

BUDGET_S = {
    "test_qa1_the_fact_before_the_questions_that_echo_it": 5.0,
    "test_qa2_a_question_still_laid_where_room_remains": 5.0,
    "test_qa3_the_memory_block_too": 5.0,
    "test_qa4_what_asks_is_read_as_the_probes_read_it": 5.0,
    "test_qa5_a_question_that_orders_is_an_order": 5.0,
    "test_qa6_a_whole_segment_never_only_asks": 5.0,
}

_FACT = "Bob lives in Lyon."
_ASKED = "Quick check: where does Bob live?"
_QUESTION = "Where does Bob live?"


def _held(lib, cid, units, cfg, after=6):
    """A conversation whose first spans, each a typed unit and an answer, were held, each unit its anchor."""
    messages = []
    for unit in units:
        messages += [oj.typed(unit), oj.answer("Noted.")]
    messages += oj.conversation(after, start=len(messages) + 1)
    state = lib.state_for(cid, cfg)
    state.mirror(messages)
    for unit in units:
        oj.hold(state, 2, anchors=((state.flesh.turns()[0]["turn_id"], 0, len(unit)),))
    return state, messages


# ---------------------------------------------------------------------------
# QA1 -- the fact before the questions that echo it
# ---------------------------------------------------------------------------
def test_qa1_the_fact_before_the_questions_that_echo_it():
    """QA1: at 4096, a fact and forty earlier questions echoing the one asked: the fact laid."""
    loaded, restore = oj.open_window()
    lib, peels = loaded[oj.LIBRARIAN], loaded[oj.PEELS]
    try:
        cfg = oj.config(lib)
        state, messages = _held(lib, "c1", [_FACT] + [_ASKED] * 40, cfg)
        ranked = peels.select_anchors(state.ledger, state.cellar, _QUESTION, 10 ** 6, estimate=oj.count)
        hits = {item.text: item.score for item in ranked}
        assert hits[_ASKED] > hits[_FACT], ("control: the question's words hold more of the one asked", hits)
        # The parts laid whole fill their caps (``_onion_journal.snug``): the cap binds as written.
        win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=4096,
                        budgets=oj.snug(loaded, _QUESTION))
        assert win.refusal is None, win.refusal
        last = win.messages[-1][1]
        assert last.count(_ASKED) < 40, ("control: the earlier questions fill the cap", last.count(_ASKED))
        assert _FACT in last, "the fact that answers them, laid"
    finally:
        restore()


# ---------------------------------------------------------------------------
# QA2 -- a question still laid where room remains
# ---------------------------------------------------------------------------
def test_qa2_a_question_still_laid_where_room_remains():
    """QA2: at 8192, the fact and the earlier question both laid."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        _state, messages = _held(lib, "c1", [_FACT, _ASKED], cfg)
        win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=8192)
        assert win.refusal is None and dict(win.cut)["memory"] == 0, (win.refusal, win.cut)
        assert _FACT in win.messages[-1][1] and _ASKED in win.messages[-1][1], "both laid"
    finally:
        restore()


# ---------------------------------------------------------------------------
# QA3 -- the memory block too
# ---------------------------------------------------------------------------
def test_qa3_the_memory_block_too():
    """QA3: the executor's block, its peels' cap filled by the earlier questions, lays the fact."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        _held(lib, "c1", [_FACT] + [_ASKED] * 40, cfg)
        block = str(lib.memory_block("c1", _QUESTION, config=cfg, budget=composer.load_budgets().at(4096)))
        anchored = block.count(f"{_ASKED}\n[/data]")
        assert 0 < anchored < 40, ("control: the questions fill the block's cap", anchored)
        assert f"{_FACT}\n[/data]" in block, "the fact that answers them, in the block"
    finally:
        restore()


# ---------------------------------------------------------------------------
# QA4 -- what asks is read as the probes read it
# ---------------------------------------------------------------------------
def test_qa4_what_asks_is_read_as_the_probes_read_it():
    """QA4: a question closed by "?!" or inside brackets ranks after a fact holding fewer of the words asked."""
    loaded, restore = oj.open_window()
    lib, peels = loaded[oj.LIBRARIAN], loaded[oj.PEELS]
    try:
        cfg = oj.config(lib)
        for cid, asked in (("c1", "Does Bob really live in Lyon?!"), ("c2", "(Does Bob live in Lyon now?)")):
            state, _messages = _held(lib, cid, [_FACT, asked], cfg)
            ranked = [item.text for item in peels.select_anchors(state.ledger, state.cellar,
                                                                 "Does Bob live in Lyon now?", 10 ** 6,
                                                                 estimate=oj.count)]
            assert ranked == [_FACT, asked], ("the fact first, the question after it", cid, ranked)
            only = [item.text for item in peels.select_anchors(state.ledger, state.cellar,
                                                               "Does Bob live in Lyon now?", oj.count(_FACT),
                                                               estimate=oj.count)]
            assert only == [_FACT], ("a cap that holds one holds the fact", cid, only)
    finally:
        restore()



# ---------------------------------------------------------------------------
# QA5 -- a question that orders is an order
# ---------------------------------------------------------------------------
def test_qa5_a_question_that_orders_is_an_order():
    """QA5: at 4096, a prohibition typed with a tag question among forty facts naming Atlas: laid."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        order = "Never push to main on Atlas, ok?"
        facts = [f"Atlas runs build {n} on the cluster." for n in range(1, 41)]
        _state, messages = _held(lib, "c1", [order] + facts, cfg)
        # The parts laid whole fill their caps (``_onion_journal.snug``): the cap binds as written.
        question = "Can I push to main on Atlas?"
        win = oj.window(loaded, "c1", question, messages=messages, config=cfg, model_window=4096,
                        budgets=oj.snug(loaded, question))
        assert win.refusal is None, win.refusal
        last = win.messages[-1][1]
        assert sum(fact in last for fact in facts) < len(facts), "control: the facts crowd the cap"
        assert order in last, "the prohibition the user typed as a question, laid at its rank"
    finally:
        restore()


# ---------------------------------------------------------------------------
# QA6 -- a whole segment never only asks
# ---------------------------------------------------------------------------
def test_qa6_a_whole_segment_never_only_asks():
    """QA6: a segment kept whole that states then asks, and the fact within it: the segment ranks as stating,
    the fact shown once."""
    loaded, restore = oj.open_window()
    lib, peels = loaded[oj.LIBRARIAN], loaded[oj.PEELS]
    try:
        cfg = oj.config(lib)
        whole = f"{_FACT} Is that still so?"
        messages = [oj.typed(whole), oj.answer("Noted.")] + oj.conversation(6, start=3)
        state = lib.state_for("c1", cfg)
        state.mirror(messages)
        turn = state.flesh.turns()[0]["turn_id"]
        # In the order a held span keeps them: a segment before the units it holds.
        oj.hold(state, 2, anchors=((turn, 0, len(whole)), (turn, 0, len(_FACT))))
        ranked = [item.text for item in peels.select_anchors(state.ledger, state.cellar, _QUESTION, 10 ** 6,
                                                             estimate=oj.count)]
        assert ranked == [whole], ("the segment as what it states, the fact within it not again", ranked)
        win = oj.window(loaded, "c1", _QUESTION, messages=messages, config=cfg, model_window=8192)
        assert win.refusal is None and win.messages[-1][1].count(_FACT) == 1, "the fact shown once"
    finally:
        restore()
