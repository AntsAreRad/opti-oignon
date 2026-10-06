#!/usr/bin/env python3
"""Contracts for the selection of peels at query time: terms are words in
any script, read as the probes read them.

Selection is keyword-based: a peel is scored by the share of the query's
terms its text holds. A term is a word as the reader of the probes splits
one -- a run of letters or figures in any script -- in lower case with its
accents taken off; the function words of both languages and the words a
question is asked with are no terms. An ASCII-only reading broke every
French word at its accents and matched the pieces.

  * US1 -- a French query is read as whole words: the peel that holds them is
    selected alone, at the full score, and a peel that shares only the
    pieces of broken words is not.
  * US2 -- a term is folded: a query without accents or in capitals selects
    the peel written with them, and the reverse.
  * US3 -- the function words and question words of both languages are no
    terms: a query made of them selects nothing, and one more word selects by
    that word alone.
  * US4 -- a word of any script is a term: a Cyrillic name selects the peel
    that holds it.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_E = chr(0xE9)
_A_GRAVE = chr(0xE0)
_E_CAP = chr(0xC9)
_MOSCOW = "".join(chr(c) for c in (0x41C, 0x43E, 0x441, 0x43A, 0x432, 0x430))

_PREFERENCE = "pr" + _E + "f" + _E + "rence"
_DECIDED = "d" + _E + "cid" + _E + "e"
_FR = "La " + _PREFERENCE + " " + _DECIDED + " " + _A_GRAVE + " " + _E_CAP + "vreux tient."
_PIECES = "Une r" + _E + "f" + _E + "rence de fichier."
_RU = "The office moves to " + _MOSCOW + " in spring."
_PLAIN = "The Evreux depot keeps the spare drives."
_QUERY = _PREFERENCE + " " + _DECIDED + " " + _A_GRAVE + " " + _E_CAP + "vreux"


def _open():
    loaded, restore = isolate(
        targets={
            "opti_oignon.memory.probes": source("memory", "probes.py"),
            "opti_oignon.memory.receipts": source("memory", "receipts.py"),
            "opti_oignon.memory.peels": source("memory", "peels.py"),
        },
        packages=("opti_oignon.memory",),
    )
    loaded["opti_oignon.memory.probes"]._native = lambda: None
    return loaded["opti_oignon.memory.peels"], restore


def _tree(peels, *texts):
    tree = peels.PeelTree()
    for index, text in enumerate(texts):
        tree.add(peels.Peel(id=f"p{index}", text=text, level=0, sources=(), children=(), source_digest="",
                            probes_passed=0, probes_total=0))
    return tree


def _chosen(peels, tree, query):
    return [(c.text, c.score) for c in peels.select_peels(tree, query, cap=500)]


# ---------------------------------------------------------------------------
# US1 -- a French query is read as whole words
# ---------------------------------------------------------------------------
def test_us1_a_french_query_is_read_as_whole_words_and_selects_the_peel_that_holds_them_alone():
    peels, restore = _open()
    try:
        tree = _tree(peels, _FR, _PIECES, _RU)
        assert _chosen(peels, tree, _QUERY) == [(_FR, 1.0)]
    finally:
        restore()


# ---------------------------------------------------------------------------
# US2 -- a term is folded
# ---------------------------------------------------------------------------
def test_us2_a_term_is_folded_its_accents_and_its_case_taken_off():
    peels, restore = _open()
    try:
        assert _chosen(peels, _tree(peels, _FR, _PIECES, _RU), "PREFERENCE DECIDEE A EVREUX") == [(_FR, 1.0)]
        tree = _tree(peels, _FR, _PIECES, _RU, _PLAIN)
        assert _chosen(peels, tree, _E_CAP + "vreux depot") == [(_PLAIN, 1.0), (_FR, 0.5)]
    finally:
        restore()


# ---------------------------------------------------------------------------
# US3 -- function words and question words are no terms
# ---------------------------------------------------------------------------
def test_us3_the_function_and_question_words_of_both_languages_are_no_terms():
    peels, restore = _open()
    try:
        tree = _tree(peels, _FR, _PIECES, _RU, _PLAIN)
        for query in ("quelle est la", "what is the", "which did they", "de la une", "comment"):
            assert _chosen(peels, tree, query) == [], query
        assert _chosen(peels, tree, "quelle est la " + _PREFERENCE + " ?") == [(_FR, 1.0)]
    finally:
        restore()


# ---------------------------------------------------------------------------
# US4 -- a word of any script is a term
# ---------------------------------------------------------------------------
def test_us4_a_word_of_any_script_is_a_term():
    peels, restore = _open()
    try:
        tree = _tree(peels, _FR, _PIECES, _RU, _PLAIN)
        assert _chosen(peels, tree, _MOSCOW) == [(_RU, 1.0)]
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
