#!/usr/bin/env python3
"""Contracts for the recall-probe generator and scorer.

A compressed memory may replace verbatim text only once it has been shown to
still answer for it. The instrument that shows it is a set of grounded probes
generated from the source span -- entities, numbers, dates, decisions, each
carrying the turn it came from -- and a scorer that says whether a candidate
text answers them. Nothing here summarises anything: this is the measuring
instrument the memory block builds first, and the block's own rule is that no
pipeline is written before the instrument is proven capable.

Proven capable means both directions. A generator that returns nothing on a
rich span is a defect of the silent-zero family, and a scorer that reports a
rate of 0.0 when there were no probes to score has invented a measurement.

  * RP1 -- a rich span yields probes, and every kind is represented.
  * RP2 -- a blank span yields none, and the rate is then unknown, not zero.
  * RP3 -- every probe names the turn it came from, and that turn answers it.
  * RP4 -- a span answers all of its own probes.
  * RP5 -- inverting a decision fails at least one probe (blade 1).
  * RP6 -- deleting the decision sentence fails at least one (blade 2).
  * RP7 -- swapping an entity fails at least one (blade 3).
  * RP8 -- shifting a date fails at least one (blade 4).
  * RP9 -- a decision written in French is drawn, and inverting it fails
    its probe, whether the negation is formal (ne ... pas) or familiar.
  * RP10 -- a typographic apostrophe negates as the ASCII one does.
  * RP11 -- French function words are neither entities nor decision words.
  * RP12 -- a French name is drawn whatever its accents, at its head or
    inside it, and a summary that swaps it fails its probe.
  * RP13 -- a French decision keeps its words whole in its key.
  * RP14 -- on ASCII text the expressions that read French draw and score
    exactly what the ASCII expressions did, over a deterministic corpus.
  * RP15 to RP22 -- RP1, RP5, RP6, RP9, RP10, RP11, RP13 and RP14 held on
    typed turns, with names off the head of a sentence: only typed text
    decides, and a head names nothing by its capital alone.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the module is pure and reaches nothing.
"""

import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_SPAN = [
    {"turn_id": "t1", "text": "Alice moved to Berlin on 2024-03-15 and started at Contoso."},
    {"turn_id": "t2", "text": "We decided to ship the release on 2024-05-01 with 3 reviewers."},
    {"turn_id": "t3", "text": "Bob prefers tea and the budget is 1200 euros."},
    {"turn_id": "t4", "text": "The team agreed not to use Docker for the demo."},
]


def _open():
    loaded, restore = isolate(
        targets={"opti_oignon.memory.probes": source("memory", "probes.py")},
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.probes"], restore


def _text(span):
    return " ".join(t["text"] for t in span)


def _with(span, turn_id, text):
    return [dict(t, text=text) if t["turn_id"] == turn_id else t for t in span]


# ---------------------------------------------------------------------------
# RP1 -- a rich span yields probes of every kind
# ---------------------------------------------------------------------------
def test_rp1_a_rich_span_yields_probes_of_every_kind():
    mod, restore = _open()
    try:
        probes = mod.generate_probes(_SPAN)
        assert len(probes) >= 4, "a span this rich yields several probes"
        kinds = {p.kind for p in probes}
        assert {"entity", "number", "date", "decision"} <= kinds, (
            f"every probe kind is represented on this span, got {sorted(kinds)}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP2 -- a blank span yields none, and the rate is unknown
# ---------------------------------------------------------------------------
def test_rp2_a_blank_span_yields_no_probes_and_an_unknown_rate():
    mod, restore = _open()
    try:
        probes = mod.generate_probes([{"turn_id": "t1", "text": "   "}])
        assert probes == [], "nothing to probe in a blank span"
        result = mod.score(probes, "anything")
        assert result.rate is None, (
            "no probes means no rate: reporting 0.0 here would invent a "
            "measurement of total failure out of an absence of questions"
        )
        assert result.passed == 0 and result.failed == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP3 -- provenance: every probe names its turn, and that turn answers it
# ---------------------------------------------------------------------------
def test_rp3_every_probe_carries_a_turn_that_answers_it():
    mod, restore = _open()
    try:
        by_turn = {t["turn_id"]: t["text"] for t in _SPAN}
        probes = mod.generate_probes(_SPAN)
        assert probes, "control: there are probes to check"
        for p in probes:
            assert p.turn_id in by_turn, f"probe names an unknown turn {p.turn_id!r}"
            assert mod.answers(p, by_turn[p.turn_id]), (
                f"the turn a probe came from answers it: {p.question!r} in "
                f"{p.turn_id}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP4 -- a span answers all of its own probes
# ---------------------------------------------------------------------------
def test_rp4_a_span_answers_all_of_its_own_probes():
    mod, restore = _open()
    try:
        probes = mod.generate_probes(_SPAN)
        result = mod.score(probes, _text(_SPAN))
        assert result.failed == 0 and result.rate == 1.0, (
            f"verbatim text answers every probe drawn from it; failed: "
            f"{[p.question for p in result.failures]}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP5 -- blade 1: inverting a decision fails a probe
# ---------------------------------------------------------------------------
def test_rp5_inverting_a_decision_fails_a_probe():
    mod, restore = _open()
    try:
        probes = mod.generate_probes(_SPAN)
        inverted = _with(_SPAN, "t4", "The team agreed to use Docker for the demo.")
        result = mod.score(probes, _text(inverted))
        assert result.failed >= 1, "a decision turned into its opposite is caught"
        assert any(p.kind == "decision" for p in result.failures), (
            "and it is a decision probe that catches it"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP6 -- blade 2: deleting the decision sentence fails a probe
# ---------------------------------------------------------------------------
def test_rp6_deleting_a_decision_fails_a_probe():
    mod, restore = _open()
    try:
        probes = mod.generate_probes(_SPAN)
        without = [t for t in _SPAN if t["turn_id"] != "t2"]
        result = mod.score(probes, _text(without))
        assert result.failed >= 1, "a dropped decision is caught"
        assert any(p.kind == "decision" and p.turn_id == "t2" for p in result.failures), (
            "by the decision probe drawn from the dropped turn"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP7 -- blade 3: swapping an entity fails a probe
# ---------------------------------------------------------------------------
def test_rp7_swapping_an_entity_fails_a_probe():
    mod, restore = _open()
    try:
        probes = mod.generate_probes(_SPAN)
        swapped = _with(_SPAN, "t1", "Alice moved to Paris on 2024-03-15 and started at Contoso.")
        result = mod.score(probes, _text(swapped))
        assert result.failed >= 1, "an entity swapped for another is caught"
        assert any(p.kind == "entity" and p.answer == "Berlin" for p in result.failures), (
            "by the entity probe whose answer was swapped away"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP8 -- blade 4: shifting a date fails a probe
# ---------------------------------------------------------------------------
def test_rp8_shifting_a_date_fails_a_probe():
    mod, restore = _open()
    try:
        probes = mod.generate_probes(_SPAN)
        shifted = _with(_SPAN, "t2", "We decided to ship the release on 2024-06-01 with 3 reviewers.")
        result = mod.score(probes, _text(shifted))
        assert result.failed >= 1, "a shifted date is caught"
        assert any(p.kind == "date" and p.answer == "2024-05-01" for p in result.failures), (
            "by the date probe whose answer moved"
        )
    finally:
        restore()



# ---------------------------------------------------------------------------
# RP9 -- a French decision is drawn and its inversion caught
# ---------------------------------------------------------------------------
def test_rp9_a_french_decision_is_drawn_and_inverting_it_fails_its_probe():
    mod, restore = _open()
    try:
        cases = (
            ("Nous avons d\u00e9cid\u00e9 de ne pas utiliser Docker pour la d\u00e9mo.",
             "Nous avons d\u00e9cid\u00e9 d'utiliser Docker pour la d\u00e9mo."),
            ("On a d\u00e9cid\u00e9 : on utilisera pas Docker pour la d\u00e9mo.",
             "On a d\u00e9cid\u00e9 : on utilisera Docker pour la d\u00e9mo."),
            ("Il faut qu'on n'utilise plus Docker pour la d\u00e9mo.",
             "Il faut qu'on utilise Docker pour la d\u00e9mo."),
        )
        for source_text, inverted in cases:
            span = [{"turn_id": "t1", "text": source_text}]
            drawn = mod.generate_probes(span)
            decisions = [p for p in drawn if p.kind == "decision"]
            assert decisions, f"a French decision is drawn: {source_text!r}"
            assert decisions[0].negations >= 1, "its negation is counted"
            assert mod.score(drawn, source_text).failed == 0, "the source answers its own probes"
            failures = mod.score(drawn, inverted).failures
            assert any(p.kind == "decision" for p in failures), f"the inversion is caught: {inverted!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP10 -- the typographic apostrophe negates
# ---------------------------------------------------------------------------
def test_rp10_a_typographic_apostrophe_negates_as_the_ascii_one():
    mod, restore = _open()
    try:
        typographic = "We decided we don\u2019t ship the demo on Friday."
        span = [{"turn_id": "t1", "text": typographic}]
        drawn = mod.generate_probes(span)
        decision = next(p for p in drawn if p.kind == "decision")
        assert decision.negations == 1, "don\u2019t is a negation"
        assert mod.score(drawn, "We decided we don't ship the demo on Friday.").failed == 0, "either apostrophe answers"
        assert mod.score(drawn, "We decided we do ship the demo on Friday.").failed >= 1, "the inversion is caught"
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP11 -- French function words
# ---------------------------------------------------------------------------
def test_rp11_french_function_words_are_neither_entities_nor_decision_words():
    mod, restore = _open()
    try:
        span = [{"turn_id": "t1", "text": "Nous avons d\u00e9cid\u00e9 que Carol m\u00e8ne la revue avec Bob."}]
        drawn = mod.generate_probes(span)
        entities = {p.answer for p in drawn if p.kind == "entity"}
        assert entities == {"Carol", "Bob"}, entities
        decision = next(p for p in drawn if p.kind == "decision")
        assert not decision.key & {"nous", "avons", "que", "la", "avec"}, decision.key
        assert {"carol", "bob", "revue"} <= decision.key
    finally:
        restore()



# ---------------------------------------------------------------------------
# RP12 -- French names, whatever their accents
# ---------------------------------------------------------------------------
def test_rp12_a_french_name_is_drawn_whatever_its_accents_and_swapping_it_fails():
    mod, restore = _open()
    try:
        text = "Nous avons invit\u00e9 \u00c9lodie, H\u00e9l\u00e8ne et Chlo\u00e9 \u00e0 la revue avec Bob."
        drawn = mod.generate_probes([{"turn_id": "t1", "text": text}])
        names = {p.answer for p in drawn if p.kind == "entity"}
        assert names == {"\u00c9lodie", "H\u00e9l\u00e8ne", "Chlo\u00e9", "Bob"}, names
        assert mod.score(drawn, text).failed == 0, "the source answers its own probes"
        for name, other in (("\u00c9lodie", "\u00c9mile"), ("H\u00e9l\u00e8ne", "H\u00e9lo\u00efse"), ("Chlo\u00e9", "Chlo\u00eb")):
            failures = mod.score(drawn, text.replace(name, other)).failures
            assert [p.answer for p in failures if p.kind == "entity"] == [name], (name, failures)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP13 -- French words stay whole in a decision key
# ---------------------------------------------------------------------------
def test_rp13_a_french_decision_keeps_its_words_whole_in_its_key():
    mod, restore = _open()
    try:
        text = "Chlo\u00e9 et Andr\u00e9 ont d\u00e9cid\u00e9 que Z\u00f6e m\u00e8ne la d\u00e9mo \u00e0 Lyon."
        drawn = mod.generate_probes([{"turn_id": "t1", "text": text}])
        decision = next(p for p in drawn if p.kind == "decision")
        assert {"chlo\u00e9", "andr\u00e9", "d\u00e9cid\u00e9", "z\u00f6e", "m\u00e8ne", "d\u00e9mo", "lyon"} <= decision.key, decision.key
        assert not decision.key & {"\u00e0", "e", "m", "mo", "cid", "chlo", "andr", "z"}, decision.key
        assert {p.answer for p in drawn if p.kind == "entity"} == {"Chlo\u00e9", "Andr\u00e9", "Z\u00f6e", "Lyon"}
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP14 -- ASCII text draws and scores as it did
# ---------------------------------------------------------------------------
# The expressions the module held for names and words before it read French.
_ASCII_NAME = r"\b[A-Z][a-zA-Z]+\b"
_ASCII_WORD = r"[a-z0-9]+"
_ASCII_PIECES = (
    "Alice", "McDonald", "USA", "iPhone", "OpenAI", "A", "x", "b2", "C_d", "Docker2", "e-F", "Gh",
    "We decided", " will ", "not", "no", "agreed", "Bob", "the", "The", " ", "  ", ".", "!", "?", "-",
    "_", "1", "42", "2024-03-15", "'", "can't", "\t", "\n",
)


def test_rp14_on_ascii_text_the_expressions_that_read_french_draw_and_score_as_before():
    mod, restore = _open()
    try:
        rng = random.Random(14)
        texts = [_text(_SPAN)] + [
            "".join(rng.choice(_ASCII_PIECES) for _ in range(rng.randint(1, 30))) for _ in range(2000)
        ]
        spans = [[{"turn_id": f"t{i}", "text": text}] for i, text in enumerate(texts)]
        candidates = [(text, rng.choice(texts), text.lower()) for text in texts]

        def run():
            drawn = [mod.generate_probes(span) for span in spans]
            scored = [[mod.score(probes, c) for c in cands] for probes, cands in zip(drawn, candidates)]
            return drawn, scored

        now = run()
        saved = mod._WORD, mod._CAPITALISED
        mod._WORD, mod._CAPITALISED = re.compile(_ASCII_WORD), re.compile(_ASCII_NAME)
        try:
            before = run()
        finally:
            mod._WORD, mod._CAPITALISED = saved
        assert now[0] == before[0], "the same probes, in the same order"
        assert now[1] == before[1], "the same verdicts on every candidate"
        entities = sum(p.kind == "entity" for probes in now[0] for p in probes)
        decisions = sum(p.kind == "decision" for probes in now[0] for p in probes)
        assert entities >= 1500 and decisions >= 1200, (entities, decisions)
    finally:
        restore()


# ---------------------------------------------------------------------------
# RP15-RP22 -- the contracts above, on typed turns and names off the head
# ---------------------------------------------------------------------------
# Only typed text decides, and a capital at the head of a sentence names
# nothing unless the span capitalises the word elsewhere. The contracts above
# drew their decisions from turns of no origin and some names from the head
# of a sentence; each one below holds the same property on typed turns, with
# its names where a capital still says something.
_TYPED_SPAN = [dict(t, role="user", origin="typed") for t in _SPAN]
_E, _EG, _A, _OU = chr(0xE9), chr(0xE8), chr(0xE0), chr(0xF6)


def _typed_turn(text, turn_id="t1"):
    return {"turn_id": turn_id, "role": "user", "origin": "typed", "text": text}


def test_rp15_a_rich_typed_span_yields_probes_of_every_kind():
    mod, restore = _open()
    try:
        probes = mod.generate_probes(_TYPED_SPAN)
        assert len(probes) >= 4, "a span this rich yields several probes"
        kinds = {p.kind for p in probes}
        assert {"entity", "number", "date", "decision"} <= kinds, (
            f"every probe kind is represented on this span, got {sorted(kinds)}"
        )
    finally:
        restore()


def test_rp16_inverting_a_typed_decision_fails_a_probe():
    mod, restore = _open()
    try:
        probes = mod.generate_probes(_TYPED_SPAN)
        inverted = _with(_TYPED_SPAN, "t4", "The team agreed to use Docker for the demo.")
        result = mod.score(probes, _text(inverted))
        assert result.failed >= 1, "a decision turned into its opposite is caught"
        assert any(p.kind == "decision" for p in result.failures), "and it is a decision probe that catches it"
    finally:
        restore()


def test_rp17_deleting_a_typed_decision_fails_a_probe():
    mod, restore = _open()
    try:
        probes = mod.generate_probes(_TYPED_SPAN)
        without = [t for t in _TYPED_SPAN if t["turn_id"] != "t2"]
        result = mod.score(probes, _text(without))
        assert result.failed >= 1, "a dropped decision is caught"
        assert any(p.kind == "decision" and p.turn_id == "t2" for p in result.failures), (
            "by the decision probe drawn from the dropped turn"
        )
    finally:
        restore()


def test_rp18_a_typed_french_decision_is_drawn_and_inverting_it_fails_its_probe():
    mod, restore = _open()
    try:
        decide, demo = f"d{_E}cid{_E}", f"d{_E}mo"
        cases = (
            (f"Nous avons {decide} de ne pas utiliser Docker pour la {demo}.",
             f"Nous avons {decide} d'utiliser Docker pour la {demo}."),
            (f"On a {decide} : on utilisera pas Docker pour la {demo}.",
             f"On a {decide} : on utilisera Docker pour la {demo}."),
            (f"Il faut qu'on n'utilise plus Docker pour la {demo}.",
             f"Il faut qu'on utilise Docker pour la {demo}."),
        )
        for source_text, inverted in cases:
            drawn = mod.generate_probes([_typed_turn(source_text)])
            decisions = [p for p in drawn if p.kind == "decision"]
            assert decisions, f"a French decision is drawn: {source_text!r}"
            assert decisions[0].negations >= 1, "its negation is counted"
            assert mod.score(drawn, source_text).failed == 0, "the source answers its own probes"
            failures = mod.score(drawn, inverted).failures
            assert any(p.kind == "decision" for p in failures), f"the inversion is caught: {inverted!r}"
    finally:
        restore()


def test_rp19_a_typographic_apostrophe_negates_a_typed_decision_as_the_ascii_one():
    mod, restore = _open()
    try:
        typographic = "We decided we don" + chr(0x2019) + "t ship the demo on Friday."
        drawn = mod.generate_probes([_typed_turn(typographic)])
        decision = next(p for p in drawn if p.kind == "decision")
        assert decision.negations == 1, "the typographic n't is a negation"
        assert mod.score(drawn, "We decided we don't ship the demo on Friday.").failed == 0, "either apostrophe answers"
        assert mod.score(drawn, "We decided we do ship the demo on Friday.").failed >= 1, "the inversion is caught"
    finally:
        restore()


def test_rp20_french_function_words_are_neither_entities_nor_words_of_a_typed_decision():
    mod, restore = _open()
    try:
        drawn = mod.generate_probes([_typed_turn(f"Nous avons d{_E}cid{_E} que Carol m{_EG}ne la revue avec Bob.")])
        entities = {p.answer for p in drawn if p.kind == "entity"}
        assert entities == {"Carol", "Bob"}, entities
        decision = next(p for p in drawn if p.kind == "decision")
        assert not decision.key & {"nous", "avons", "que", "la", "avec"}, decision.key
        assert {"carol", "bob", "revue"} <= decision.key
    finally:
        restore()


def test_rp21_a_typed_french_decision_keeps_its_words_whole_in_its_key():
    mod, restore = _open()
    try:
        chloe, andre, zoe = f"Chlo{_E}", f"Andr{_E}", f"Z{_OU}e"
        text = f"Hier, {chloe} et {andre} ont d{_E}cid{_E} que {zoe} m{_EG}ne la d{_E}mo {_A} Lyon."
        drawn = mod.generate_probes([_typed_turn(text)])
        decision = next(p for p in drawn if p.kind == "decision")
        whole = {chloe.lower(), andre.lower(), f"d{_E}cid{_E}", zoe.lower(), f"m{_EG}ne", f"d{_E}mo", "lyon"}
        assert whole <= decision.key, decision.key
        assert not decision.key & {_A, "e", "m", "mo", "cid", "chlo", "andr", "z"}, decision.key
        assert {p.answer for p in drawn if p.kind == "entity"} == {chloe, andre, zoe, "Lyon"}
    finally:
        restore()


def test_rp22_on_ascii_typed_text_the_expressions_that_read_french_draw_and_score_as_before():
    mod, restore = _open()
    try:
        rng = random.Random(14)
        texts = [_text(_SPAN)] + [
            "".join(rng.choice(_ASCII_PIECES) for _ in range(rng.randint(1, 30))) for _ in range(2000)
        ]
        spans = [[_typed_turn(text, f"t{i}")] for i, text in enumerate(texts)]
        candidates = [(text, rng.choice(texts), text.lower()) for text in texts]

        def run():
            drawn = [mod.generate_probes(span) for span in spans]
            scored = [[mod.score(probes, c) for c in cands] for probes, cands in zip(drawn, candidates)]
            return drawn, scored

        now = run()
        saved = mod._WORD, mod._CAPITALISED
        mod._WORD, mod._CAPITALISED = re.compile(_ASCII_WORD), re.compile(_ASCII_NAME)
        try:
            before = run()
        finally:
            mod._WORD, mod._CAPITALISED = saved
        assert now[0] == before[0], "the same probes, in the same order"
        assert now[1] == before[1], "the same verdicts on every candidate"
        entities = sum(p.kind == "entity" for probes in now[0] for p in probes)
        decisions = sum(p.kind == "decision" for probes in now[0] for p in probes)
        assert entities >= 1200 and decisions >= 1200, (entities, decisions)
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
