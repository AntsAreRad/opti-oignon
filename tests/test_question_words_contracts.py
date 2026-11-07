"""Contracts for the question words the onion's anchors are never reached by (237 a).

A question reaches the anchors by its terms, its function words and the words it asks with set aside. "where",
"when", "why", "whom", "whose", and in French "quand", "pourquoi", "combien", "lequel", "laquelle", "lesquels" and
"lesquelles" were terms: "Where does Jo live?" was reached through "where", and an anchor holding that word ranked
with the one that answers (AR13 had rephrased its question around the hole). A replay registered before it ran
(LoCoMo, ten conversations, 1 530 questions with an answer) kept the answer's words in the window as well: +0.001
[-0.002, +0.004].
  * QW1 -- THE WORDS A QUESTION ASKS WITH: each of the twelve, in a question, leaves only the question's content as
    its terms; "ou" (the folded "o\u00f9") stays aside as a function word.
  * QW2 -- AR13'S QUESTION AS IT WAS ASKED: "Where does Jo live?" lays the anchor that answers, not the newer one
    that holds "where".
Each contract opens the onion's shared isolation window (``tests/_onion_journal.py``), as the anchor rank contracts
do, and lays anchors through their helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_journal as oj  # noqa: E402
from test_anchor_rank_contracts import _RECEIPTS, _laid, _receipt, _turn, _whole  # noqa: E402

BUDGET_S = {
    "test_qw1_the_words_a_question_asks_with": 3.0,
    "test_qw2_ar13s_question_as_it_was_asked": 4.0,
}

ADDED = ("where", "when", "why", "whom", "whose", "quand", "pourquoi", "combien", "lequel", "laquelle", "lesquels",
         "lesquelles")


def test_qw1_the_words_a_question_asks_with():
    """QW1 -- "<word> Jo lives in Lyon": the terms are Jo's, Lyon's and the verb's, never the word asked with, for
    each of the twelve; "ou" and its accented form stay out. A question word counted as a term -> RED."""
    loaded, restore = oj.open_window()
    peels = loaded[oj.PEELS]
    try:
        content = peels._terms("Jo lives in Lyon")
        assert {"jo", "lyon"} <= content, content
        for word in ADDED + ("ou", "o\u00f9"):
            for text in (f"{word} Jo lives in Lyon", f"{word.capitalize()} Jo lives in Lyon?"):
                assert peels._terms(text) == content, (word, peels._terms(text))
    finally:
        restore()


def test_qw2_ar13s_question_as_it_was_asked():
    """QW2 -- AR13's question in the words first asked: the anchor that answers is laid where the cap keeps one.
    The anchor holding the question word laid first -> RED."""
    loaded, restore = oj.open_window()
    peels, receipts = loaded[oj.PEELS], loaded[_RECEIPTS]
    try:
        # The decoy holds the question word and as many of its other terms as the answer, and is the newer
        # anchor: the word asked with is what put it first. (Other questions built so were laid right before the
        # change too: the twelve words are each proven by QW1.)
        cases = (("Where does Jo live?", ["Jo lives in Lyon now.", "Where I live does not matter."]),)
        for query, texts in cases:
            span = [_turn(f"u{n}", t) for n, t in enumerate(texts, 1)]
            cellar, ledger, _key = _receipt(receipts, span, _whole(span))
            assert _laid(peels, ledger, cellar, query, texts[:2]) == [texts[0]], (query, texts[0])
    finally:
        restore()
