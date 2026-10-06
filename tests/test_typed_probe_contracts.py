#!/usr/bin/env python3
"""Contracts for typed probes: dates, numbers, units and names read to one canonical form.

A summary may write a date, a number or a quantity otherwise than its source
did and still keep it: "5 octobre 2026" is "October 5, 2026" is
"2026-10-05", "1,5 Go" is "1.5 GB". It may also change it while keeping its
digits: "16 Go" is not "16 Gio". The generator reads each date and each
quantity to a canonical form and asks for that form; the scorer reads the
candidate the same way. A writing that reads two ways -- "05/10/2026",
"1,500" -- is never guessed at: its probe asks for the same writing.

A name is read by the span it stands in. Every word at the head of a
sentence takes a capital, so there the capital says nothing unless the span
capitalises the same word elsewhere; a day of the week is a relative date,
never a name; and a full name is one entity, whose aliases are only the
parts the span itself goes on to use alone.

  * TY1 -- a date written in French, in English or in ISO form is one
    answer, and each writing answers the probe drawn from the others.
  * TY2 -- an abbreviated or ordinal date reads whole, and the period of an
    abbreviated month ends no sentence.
  * TY3 -- a date keeps its granularity: a day, a month of a year, a day of
    a month without its year; it answers only at its own.
  * TY4 -- a relative date is never probed.
  * TY5 -- a numeric date that reads two ways is probed as written, never
    guessed; one that reads one way is read.
  * TY6 -- an impossible calendar day is no date, and no part of it is read
    as a coarser one.
  * TY7 -- an English month is read only with its capital; a French one in
    lower case or with a capital; a French abbreviation takes its period.
  * TY8 -- a decimal comma and a decimal point are one value, and the
    separators of thousands are not part of it.
  * TY9 -- Go is GB and never Gio: a unit reads to its class, and only its
    class answers.
  * TY10 -- a number that reads two ways is probed as written.
  * TY11 -- the written precision is kept: 3.10 is not 3.1, and a version
    or an address is probed as written.
  * TY12 -- a sign is part of the value; a hyphen between figures is none.
  * TY13 -- a symbol with more than one meaning is a class of its own.
  * TY14 -- the words of a date or a quantity are not entities, so a
    translated summary answers every probe.
  * TY15 -- the generator's version is raised, and what the typed readers
    draw on a fixed corpus is pinned with it.
  * TY16 -- a capitalised word at the head of a sentence or an item is a
    name only where the span capitalises the same word off a head; a capital
    inside inline code is no evidence.
  * TY17 -- a day of the week, English or French, is never a name.
  * TY18 -- names that follow one another are one entity; a part of it that
    the span uses alone after it is its alias, draws nothing of its own and
    answers for it; a part used before it, or never alone, is no alias.

Local-only (the public distribution ships no tests).
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_E = chr(0xE9)
_EG = chr(0xE8)
_OC = chr(0xF4)
_NBSP = chr(0xA0)
_NNBSP = chr(0x202F)
_MINUS = chr(0x2212)
_DEG = chr(0xB0)

_FEVRIER = f"f{_E}vrier"

# The corpus whose typed probes TY15 pins, and the digest of what they are,
# by version. A new version adds its line; a line is never rewritten.
_TYPED_CORPUS = (
    "Nous livrons le 5 octobre 2026.",
    "We ship on October 5, 2026.",
    "Livraison le 1er oct. 2026, budget 1 234,56 euros.",
    "Released Oct. 1st, 2026 with 16 GB and 45 ms of latency.",
    "Le budget couvre octobre 2026 et le 5 octobre.",
    "Livraison le 05/10/2026 ou le 13/10/2026.",
    f"Il a fait {_MINUS}3 {_DEG}C, 1,5 Go et 1,500 euros.",
    "Python 3.10 et 3.12.1, routeur 10.0.0.1, 16 Gio, 20 %, $16.",
)
_TYPED_PINS = {
    3: "dd3d3d6805dbf6492cb674515bf8dca2a126a01fbbb7b01b6631f96a447521a4",
    4: "dd3d3d6805dbf6492cb674515bf8dca2a126a01fbbb7b01b6631f96a447521a4",
    5: "dd3d3d6805dbf6492cb674515bf8dca2a126a01fbbb7b01b6631f96a447521a4",
    6: "dd3d3d6805dbf6492cb674515bf8dca2a126a01fbbb7b01b6631f96a447521a4",
    7: "dd3d3d6805dbf6492cb674515bf8dca2a126a01fbbb7b01b6631f96a447521a4",
}


def _probes():
    loaded, restore = isolate(
        targets={"opti_oignon.memory.probes": source("memory", "probes.py")},
        packages=("opti_oignon.memory",),
    )
    mod = loaded["opti_oignon.memory.probes"]
    mod._native = lambda: None
    return mod, restore


def _span(*texts):
    return [{"turn_id": f"t{i}", "role": "user", "origin": "typed", "text": t} for i, t in enumerate(texts, 1)]


def _typed(drawn, kind):
    return [(p.answer, p.canonical) for p in drawn if p.kind == kind]


def _of(drawn, kind):
    return [p for p in drawn if p.kind == kind]


# ---------------------------------------------------------------------------
# TY1 -- one date, three writings
# ---------------------------------------------------------------------------
def test_ty1_a_date_written_in_french_english_or_iso_is_one_answer_and_each_writing_answers_the_others():
    mod, restore = _probes()
    try:
        writings = ("Nous livrons le 5 octobre 2026.", "We ship on October 5, 2026.", "Ship date: 2026-10-05.")
        for text in writings:
            drawn = mod.generate_probes(_span(text))
            assert _typed(drawn, "date") == [("2026-10-05", True)], f"{text!r} draws {_typed(drawn, 'date')}"
            for other in writings:
                assert mod.score(_of(drawn, "date"), other).failed == 0, f"{other!r} answers the date of {text!r}"
            assert mod.score(_of(drawn, "date"), "We ship on October 6, 2026.").failed == 1, "a shifted day fails"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY2 -- abbreviations and ordinals read whole
# ---------------------------------------------------------------------------
def test_ty2_an_abbreviated_or_ordinal_date_reads_whole_and_its_period_ends_no_sentence():
    mod, restore = _probes()
    try:
        written = {
            "Livraison le 1er oct. 2026 au plus tard.": "2026-10-01",
            f"Le d{_E}p{_OC}t ouvre le 3 f{_E}vr. 2027.": "2027-02-03",
            "Released Oct. 1st, 2026 to everyone.": "2026-10-01",
            "Released on the 22nd of Sept 2026.": "2026-09-22",
            "Due Sep 3 2026.": "2026-09-03",
        }
        for text, day in written.items():
            drawn = mod.generate_probes(_span(text))
            assert _typed(drawn, "date") == [(day, True)], f"{text!r} draws {_typed(drawn, 'date')}"
        sentence = f"Nous avons d{_E}cid{_E} le 5 oct. 2026 de livrer la d{_E}mo."
        drawn = mod.generate_probes(_span(sentence))
        assert [p.answer for p in _of(drawn, "decision")] == [sentence], "the abbreviation's period ends no sentence"
        assert _typed(drawn, "number") == [], f"the year is the date's, not a number: {_typed(drawn, 'number')}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY3 -- a date keeps its granularity
# ---------------------------------------------------------------------------
def test_ty3_a_date_keeps_its_granularity_and_answers_only_at_its_own():
    mod, restore = _probes()
    try:
        day = mod.generate_probes(_span("Le budget couvre le 5 octobre 2026."))
        month = mod.generate_probes(_span("Le budget couvre octobre 2026."))
        yearless = mod.generate_probes(_span("Rendez-vous le 5 octobre."))
        assert _typed(day, "date") == [("2026-10-05", True)]
        assert _typed(month, "date") == [("2026-10", True)]
        assert _typed(yearless, "date") == [("--10-05", True)]
        assert mod.score(_of(month, "date"), "Budget covers October 2026.").failed == 0, "a month in English"
        assert mod.score(_of(yearless, "date"), "Meet on October 5.").failed == 0, "a day of a month in English"
        assert mod.score(_of(day, "date"), "Le budget couvre octobre 2026.").failed == 1, "a month is not the day"
        assert mod.score(_of(month, "date"), "Le budget couvre le 5 octobre 2026.").failed == 1, "a day is not the month"
        assert mod.score(_of(yearless, "date"), "Meet on October 5, 2026.").failed == 1, "a year added is another date"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY4 -- a relative date is never probed
# ---------------------------------------------------------------------------
def test_ty4_a_relative_date_is_never_probed():
    mod, restore = _probes()
    try:
        relative = (
            "Je l'ai vu hier et je reviens demain.",
            "On se voit la semaine prochaine, ou lundi prochain.",
            "See you tomorrow, or next Monday; it shipped yesterday.",
            "On livre le 5.",
            "Dans trois jours, on verra.",
        )
        for text in relative:
            drawn = mod.generate_probes(_span(text))
            assert _typed(drawn, "date") == [], f"{text!r} draws {_typed(drawn, 'date')}"
        control = mod.generate_probes(_span("Rendez-vous le 5 octobre 2026."))
        assert _typed(control, "date") == [("2026-10-05", True)], "the reader draws a date that is written"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY5 -- a numeric date that reads two ways is probed as written
# ---------------------------------------------------------------------------
def test_ty5_a_numeric_date_that_reads_two_ways_is_probed_as_written_and_one_that_reads_one_way_is_read():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span("Livraison le 05/10/2026."))
        assert _typed(drawn, "date") == [("05/10/2026", False)], _typed(drawn, "date")
        assert _typed(drawn, "number") == [], "its figures are the date's"
        assert mod.score(_of(drawn, "date"), "Livraison le 05/10/2026.").failed == 0, "the same writing answers"
        for guess in ("le 5 octobre 2026", "le 10 mai 2026", "le 2026-10-05", "le 5/10/2026"):
            assert mod.score(_of(drawn, "date"), f"Livraison {guess}.").failed == 1, f"{guess!r} is a guess"
        one_way = {
            "Livraison le 13/10/2026.": "2026-10-13",
            "Shipped 10/13/2026.": "2026-10-13",
            "Livraison le 2026/10/05.": "2026-10-05",
            "Livraison le 07/07/2026.": "2026-07-07",
        }
        for text, read in one_way.items():
            assert _typed(mod.generate_probes(_span(text)), "date") == [(read, True)], text
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY6 -- an impossible day is no date
# ---------------------------------------------------------------------------
def test_ty6_an_impossible_calendar_day_is_no_date_and_no_part_of_it_is_read_as_a_coarser_one():
    mod, restore = _probes()
    try:
        for text in (f"Le 31 {_FEVRIER} 2026.", "Le 2026-02-30.", f"Le 29 {_FEVRIER} 2025.", "On 2026-13-01."):
            drawn = mod.generate_probes(_span(text))
            assert _typed(drawn, "date") == [], f"{text!r} draws {_typed(drawn, 'date')}"
        leap = mod.generate_probes(_span(f"Le 29 {_FEVRIER} 2028."))
        assert _typed(leap, "date") == [("2028-02-29", True)], "a leap day is a day"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY7 -- how a month is written
# ---------------------------------------------------------------------------
def test_ty7_an_english_month_needs_its_capital_and_a_french_abbreviation_its_period():
    mod, restore = _probes()
    try:
        for text in ("You may 5 times retry.", "They march 3 miles.", "We meet in august 2026.", "Le 5 oct 2026."):
            drawn = mod.generate_probes(_span(text))
            assert _typed(drawn, "date") == [], f"{text!r} draws {_typed(drawn, 'date')}"
        for text in ("May 5, 2026.", "Le 5 mai 2026.", "Le 5 Mai 2026.", "Le 5 oct. 2026."):
            drawn = mod.generate_probes(_span(text))
            assert len(_typed(drawn, "date")) == 1, f"{text!r} draws {_typed(drawn, 'date')}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY8 -- one value, whatever its separators
# ---------------------------------------------------------------------------
def test_ty8_a_decimal_comma_and_point_are_one_value_and_thousands_separators_are_not_part_of_it():
    mod, restore = _probes()
    try:
        families = (
            (("La carte a 1,5 Go.", "The card has 1.5 GB."), "1.5 GB"),
            (
                ("1 500 000 lignes.", f"1{_NNBSP}500{_NNBSP}000 lignes.", f"1{_NBSP}500{_NBSP}000 lignes.",
                 "1,500,000 rows.", "1500000 rows."),
                "1500000",
            ),
            (("1 234,56 euros.", "1,234.56 EUR.", "1.234,56 EUR."), "1234.56 EUR"),
        )
        for writings, answer in families:
            for text in writings:
                drawn = mod.generate_probes(_span(text))
                assert _typed(drawn, "number") == [(answer, True)], f"{text!r} draws {_typed(drawn, 'number')}"
                for other in writings:
                    assert mod.score(_of(drawn, "number"), other).failed == 0, f"{other!r} answers {text!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY9 -- Go is GB, never Gio
# ---------------------------------------------------------------------------
def test_ty9_go_is_gb_and_never_gio():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span("La machine a 16 Go de RAM."))
        assert _typed(drawn, "number") == [("16 GB", True)], _typed(drawn, "number")
        probes = _of(drawn, "number")
        for same in ("16 GB of RAM", "16GB of RAM", "16 gigaoctets", f"16{_NBSP}Go", "16 gigabytes"):
            assert mod.score(probes, same).failed == 0, f"{same!r} is the same quantity"
        for other in ("16 Gio", "16 GiB", "16 de RAM", "16 Mo", "16 Gb", "16 To"):
            assert mod.score(probes, other).failed == 1, f"{other!r} is another quantity"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY10 -- a number that reads two ways is probed as written
# ---------------------------------------------------------------------------
def test_ty10_a_number_that_reads_two_ways_is_probed_as_written():
    mod, restore = _probes()
    try:
        for text, written in (("Budget: 1,500 euros.", "1,500 euros"), ("Budget : 1.500 euros.", "1.500 euros")):
            drawn = mod.generate_probes(_span(text))
            assert _typed(drawn, "number") == [(written, False)], f"{text!r} draws {_typed(drawn, 'number')}"
            probes = _of(drawn, "number")
            assert mod.score(probes, text).failed == 0, "the same writing answers"
            for guess in ("Budget: 1500 euros.", "Budget: 1.5 euros.", "Budget: 1 500 euros."):
                assert mod.score(probes, guess).failed == 1, f"{guess!r} is a guess"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY11 -- the written precision is kept
# ---------------------------------------------------------------------------
def test_ty11_the_written_precision_is_kept_and_a_version_or_an_address_is_probed_as_written():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span("Python 3.10 tourne, pas 3.1."))
        assert _typed(drawn, "number") == [("3.10", True), ("3.1", True)], _typed(drawn, "number")
        assert mod.score(_of(drawn, "number")[:1], "Python 3.1 tourne.").failed == 1, "3.1 is not 3.10"
        drawn = mod.generate_probes(_span("Python 3.12.1 et le routeur 10.0.0.1."))
        assert _typed(drawn, "number") == [("3.12.1", False), ("10.0.0.1", False)], _typed(drawn, "number")
        assert mod.score(_of(drawn, "number"), "Python 3.12 et le routeur 10.0.0.1.").failed == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY12 -- a sign is part of the value
# ---------------------------------------------------------------------------
def test_ty12_a_sign_is_part_of_the_value_and_a_hyphen_between_figures_is_none():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span(f"Il a fait -3 {_DEG}C cette nuit."))
        assert _typed(drawn, "number") == [("-3 degC", True)], _typed(drawn, "number")
        probes = _of(drawn, "number")
        assert mod.score(probes, f"Il a fait 3 {_DEG}C.").failed == 1, "the sign dropped is another value"
        for same in (f"Il a fait {_MINUS}3 {_DEG}C.", f"Il a fait -3{_DEG}C."):
            assert mod.score(probes, same).failed == 0, f"{same!r} is the same value"
        assert _typed(mod.generate_probes(_span("Les versions 3-4 tournent.")), "number") == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY13 -- a symbol with more than one meaning is a class of its own
# ---------------------------------------------------------------------------
def test_ty13_a_symbol_with_more_than_one_meaning_is_a_class_of_its_own():
    mod, restore = _probes()
    try:
        metres = _of(mod.generate_probes(_span("La parcelle fait 16 m de long.")), "number")
        assert [p.answer for p in metres] == ["16 m"]
        assert mod.score(metres, f"La parcelle fait 16 m{_EG}tres.").failed == 1, "m may be a million"
        kilometres = _of(mod.generate_probes(_span("Le chemin fait 16 km.")), "number")
        assert mod.score(kilometres, f"Le chemin fait 16 kilom{_EG}tres.").failed == 0, "km has one meaning"
        dollars = _of(mod.generate_probes(_span("It costs $16.")), "number")
        assert [p.answer for p in dollars] == ["16 $"]
        assert mod.score(dollars, "It costs 16 dollars.").failed == 0, "a dollar is a dollar"
        for other in ("It costs 16 USD.", "It costs 16 EUR."):
            assert mod.score(dollars, other).failed == 1, f"{other!r} names a currency the source did not"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY14 -- the words of a date or a quantity are not entities
# ---------------------------------------------------------------------------
def test_ty14_the_words_of_a_date_or_a_quantity_are_not_entities_and_a_translated_summary_answers_all():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span("We ship on October 5, 2026 with 16 GB and 45 EUR."))
        assert _typed(drawn, "entity") == [], f"no month or unit is a name: {_typed(drawn, 'entity')}"
        assert len(_typed(drawn, "date")) == 1 and len(_typed(drawn, "number")) == 2, drawn
        translated = "Nous livrons le 5 octobre 2026 avec 16 Go et 45 euros."
        assert mod.score(drawn, translated).failed == 0, mod.score(drawn, translated).failures
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY15 -- the version is raised and the typed forms are pinned with it
# ---------------------------------------------------------------------------
def test_ty15_the_version_is_raised_and_what_the_typed_readers_draw_is_pinned_with_it():
    mod, restore = _probes()
    try:
        assert type(mod.GENERATOR_VERSION) is int and mod.GENERATOR_VERSION >= 3
        drawn = mod.generate_probes(_span(*_TYPED_CORPUS))
        rows = [[p.kind, p.answer, p.canonical, p.turn_id] for p in drawn if p.kind in ("date", "number")]
        forms = {(kind, canonical) for kind, _answer, canonical, _turn in rows}
        assert forms == {("date", True), ("date", False), ("number", True), ("number", False)}, forms
        assert len(rows) >= 20, f"the corpus draws typed probes: {len(rows)}"
        digest = hashlib.sha256(json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest()
        assert _TYPED_PINS and mod.GENERATOR_VERSION == max(_TYPED_PINS), "the version is the last one pinned"
        assert _TYPED_PINS[mod.GENERATOR_VERSION] == digest, (
            f"the typed readers draw something else than their version pinned: {digest}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY16 -- the head of a sentence names nothing by its capital alone
# ---------------------------------------------------------------------------
def test_ty16_a_capitalised_head_is_a_name_only_where_the_span_capitalises_it_elsewhere():
    mod, restore = _probes()
    try:
        misfires = ("Ok, I will check.", "Voici la liste.", "Ouvrez le fichier.", "Lancez la commande.",
                    "Let me check the logs.", "- Release notes are ready.\n- **Note**: nothing else.")
        drawn = mod.generate_probes(_span(*misfires))
        assert _typed(drawn, "entity") == [], f"a head takes its capital from its place: {_typed(drawn, 'entity')}"
        drawn = mod.generate_probes(_span("Docker is too heavy.", "We tried Docker yesterday."))
        assert [(p.answer, p.turn_id) for p in _of(drawn, "entity")] == [("Docker", "t1"), ("Docker", "t2")], (
            "a head the span capitalises elsewhere, in another turn, is a name in both places"
        )
        for elsewhere in ("we tried docker yesterday.", "We tried `Docker` yesterday.", "Docker is gone."):
            drawn = mod.generate_probes(_span("Docker is too heavy.", elsewhere))
            assert _of(drawn, "entity") == [], f"no capital outside a head or inline code: {elsewhere!r}"
        drawn = mod.generate_probes(_span("We tried Docker.", "On 5 May 2026, Alice left."))
        assert [p.answer for p in _of(drawn, "entity")] == ["Docker", "Alice"], "a name off the head stays a name"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY17 -- a day of the week is never a name
# ---------------------------------------------------------------------------
def test_ty17_a_day_of_the_week_is_never_a_name():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span(
            "We ship on Friday with Alice.", "La revue est Lundi avec Carol.", "Friday works for Dave, not Sunday.",
        ))
        assert [p.answer for p in _of(drawn, "entity")] == ["Alice", "Carol", "Dave"], _typed(drawn, "entity")
        drawn = mod.generate_probes(_span("We ship on Fridays with Fred."))
        assert [p.answer for p in _of(drawn, "entity")] == ["Fridays", "Fred"], "only the days themselves"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TY18 -- a full name is one entity; its aliases are the span's own
# ---------------------------------------------------------------------------
def test_ty18_a_full_name_is_one_entity_and_a_part_the_span_uses_alone_after_it_is_its_alias():
    mod, restore = _probes()
    try:
        drawn = mod.generate_probes(_span("We hired Alice Martin for the audit.", "The report from Martin is due."))
        entities = _of(drawn, "entity")
        assert [(p.answer, p.turn_id) for p in entities] == [("Alice Martin", "t1")], _typed(drawn, "entity")
        assert entities[0].key == frozenset({"martin"}), "the part used alone after the full name is its alias"
        for kept in ("Alice Martin ran the audit.", "Martin, Alice ran it.", "Martin ran the audit.", "Martin's audit."):
            assert mod.score(entities, kept).failed == 0, f"the name or its alias answers: {kept!r}"
        for lost in ("Alice ran the audit.", "Carol ran the audit.", "The audit is due."):
            assert mod.score(entities, lost).failed == 1, f"a part the span never used alone does not: {lost!r}"
        drawn = mod.generate_probes(_span("We called Martin first.", "We hired Alice Martin."))
        assert [(p.answer, p.turn_id, p.key) for p in _of(drawn, "entity")] == [
            ("Martin", "t1", frozenset()), ("Alice Martin", "t2", frozenset()),
        ], "a part used before the full name is a name of its own, and no alias"
        drawn = mod.generate_probes(_span("We hired Alice, Martin and Carol."))
        assert [p.answer for p in _of(drawn, "entity")] == ["Alice", "Martin", "Carol"], "a comma parts names"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
