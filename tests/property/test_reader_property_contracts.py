#!/usr/bin/env python3
"""Property contracts for the readers of the probes: the segmenter and the
normalizers of dates, numbers and words, under seeded generated input.

A contract here states one property and holds it over inputs drawn from a
generator of the standard library, seeded: the same inputs on every run, so
a red names the seed and the case that broke it. The forms generated are
the ones the readers are written for; the noise is anything a text may hold.
The oracle of a valid day is the calendar module, never the reader.

  * PY1 -- the segmenter's blocks are in order and disjoint, each within its
    text, and a paragraph or an item is exactly the slice it bounds.
  * PY2 -- nothing is lost or doubled: every line that is not blank lies in
    exactly one block, and a blank line outside a fence in none.
  * PY3 -- a fence opens a code block that runs to the next fence of its
    character at least as long, or to the end of the text, its body the lines
    between, word for word; a backtick fence whose info string holds a
    backtick is no fence.
  * PY4 -- no text makes the segmenter raise, and every block of any text lies
    within it.
  * PY5 -- every day of the calendar, written in each form the reader knows,
    reads to its canonical form at the bounds of its writing.
  * PY6 -- a day the calendar does not hold reads as no date, and nothing
    inside it as a coarser one.
  * PY7 -- a number written with separators of thousands and a decimal mark,
    in either language, reads to its plain value, its written precision kept;
    one separator before three figures keeps its writing.
  * PY8 -- a unit of the closed table, in any of its writings, reads to its
    class with the number, and the same number in two classes is two answers.
  * PY9 -- folding is idempotent, leaves no capital, no accent and no
    combining mark, and on ASCII text gives the lower-case runs of letters
    and figures.
  * PY10 -- no text makes the readers of dates and numbers raise; the dates
    they read lie within the text, in order and disjoint, every answer in its
    shape: a canonical form, or the writing itself.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import calendar
import random
import re
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _isolation import isolate, source  # noqa: E402

_SEEDS = (11, 23, 47)
_NBSP, _NNBSP, _MINUS = chr(0xA0), chr(0x202F), chr(0x2212)
_E, _U_CIRC = chr(0xE9), chr(0xFB)
_FILLER = ("the", "review", "budget", "team", "planning", "server", "notes", "draft", "projet", "revue", "equipe",
           "avance", "with", "and", "et", "set", "is", "for")
_TAIL = ("rows", "items", "points", "lines", "copies")
_FR_MONTHS = ("janvier", "f" + _E + "vrier", "mars", "avril", "mai", "juin", "juillet", "ao" + _U_CIRC + "t",
              "septembre", "octobre", "novembre", "d" + _E + "cembre")
_EN_MONTHS = tuple(calendar.month_name[m] for m in range(1, 13))
_FR_ABBR = {1: "janv.", 2: "f" + _E + "vr.", 4: "avr.", 7: "juil.", 9: "sept.", 10: "oct.", 11: "nov.",
            12: "d" + _E + "c."}
_EN_ABBR = {m: calendar.month_abbr[m] for m in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12)}


def _probes():
    loaded, restore = isolate(
        targets={"opti_oignon.memory.probes": source("memory", "probes.py")},
        packages=("opti_oignon.memory",),
    )
    probes = loaded["opti_oignon.memory.probes"]
    probes._native = lambda: None
    return probes, restore


def _filler(rng, n=None):
    return " ".join(rng.choice(_FILLER) for _ in range(n or rng.randint(1, 4)))


# ---------------------------------------------------------------------------
# The segmenter
# ---------------------------------------------------------------------------
def _line(rng):
    kind = rng.random()
    if kind < 0.35:
        return _filler(rng)
    if kind < 0.5:
        return ""
    if kind < 0.6:
        return " " * rng.randint(1, 3)
    if kind < 0.75:
        bullet = rng.choice(("-", "*", "+", chr(0x2022), f"{rng.randint(1, 99)}.", f"{rng.randint(1, 99)})"))
        return " " * rng.randint(0, 2) + bullet + " " + _filler(rng)
    if kind < 0.85:
        return " " * rng.randint(0, 3) + rng.choice("`~") * rng.randint(3, 5) + rng.choice(("", "python", " yaml"))
    if kind < 0.92:
        return _filler(rng) + "\r"
    return "  " + _filler(rng)


def _document(rng):
    return "\n".join(_line(rng) for _ in range(rng.randint(1, 14)))


def _cases(n):
    for seed in _SEEDS:
        rng = random.Random(seed)
        for case in range(n):
            yield seed, case, rng


def test_py1_the_blocks_are_in_order_disjoint_and_within_their_text_and_prose_is_its_own_slice():
    probes, restore = _probes()
    try:
        for seed, case, rng in _cases(300):
            text = _document(rng)
            end = 0
            for block in probes.segment(text):
                assert end <= block.start <= block.end <= len(text), (seed, case, text, block)
                if block.kind != "code":
                    assert text[block.start:block.end] == block.text, (seed, case, text, block)
                end = block.end
    finally:
        restore()


def test_py2_every_line_that_is_not_blank_lies_in_exactly_one_block_and_a_blank_one_in_none():
    probes, restore = _probes()
    try:
        for seed, case, rng in _cases(300):
            text = _document(rng)
            blocks = probes.segment(text)
            fenced = [(b.start, b.end) for b in blocks if b.kind == "code"]
            position = 0
            for line in text.split("\n"):
                inside = sum(1 for start, stop in fenced if start <= position < stop or start < position + len(line)
                             <= stop)
                if line.strip():
                    last = position + len(line.rstrip()) - 1
                    holders = [b for b in blocks if b.start <= last < b.end]
                    assert len(holders) == 1, (seed, case, text, line, holders)
                elif not inside:
                    holders = [b for b in blocks if b.kind != "code" and b.start <= position < b.end]
                    assert holders == [], (seed, case, text, line, holders)
                position += len(line) + 1
    finally:
        restore()


def test_py3_a_fence_runs_to_the_next_fence_of_its_character_at_least_as_long_or_to_the_end():
    probes, restore = _probes()
    try:
        for seed, case, rng in _cases(200):
            mark = rng.choice("`~") * rng.randint(3, 5)
            other = ("~" if mark[0] == "`" else "`") * rng.randint(3, 6)
            body = [rng.choice((_filler(rng), mark[:-1], other, "", "  " + _filler(rng))) for _ in range(rng.randint(0, 5))]
            closed = rng.random() < 0.7
            closing = [mark[0] * (len(mark) + rng.randint(0, 2))] if closed else []
            after = [_filler(rng)] if closed else []
            text = "\n".join([_filler(rng), "", mark + rng.choice(("", "python")), *body, *closing, *after])
            code = [b for b in probes.segment(text) if b.kind == "code"]
            assert len(code) == 1 and code[0].text == "\n".join(body), (seed, case, text, code)
            assert (code[0].end == len(text)) is (not closed), (seed, case, text, code)
        for info in ("py`thon", "`x`"):
            text = "```" + info + "\nprint(1)\n```"
            assert [b.kind for b in probes.segment(text)][0] != "code", info
    finally:
        restore()


_NOISE = ("a", "Z", " ", "\n", "\n\n", "\r", "\t", "`", "```", "~~~", "-", "* ", "1. ", "9) ", chr(0x2022), _NBSP,
          _E, chr(0x301), chr(0x41C), chr(0x4E00), "0", "7", ".", ",", ":", "[code:", "]")


def test_py4_no_text_makes_the_segmenter_raise_and_every_block_lies_within_its_text():
    probes, restore = _probes()
    try:
        for seed, case, rng in _cases(400):
            text = "".join(rng.choice(_NOISE) for _ in range(rng.randint(0, 60)))
            for block in probes.segment(text):
                assert 0 <= block.start <= block.end <= len(text), (seed, case, text, block)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
def _ordinal(day):
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _writings(rng, y, m, d):
    """Each form the reader knows, written for one day: ``(writing, canonical)``."""
    fr = _FR_MONTHS[m - 1]
    en = _EN_MONTHS[m - 1]
    iso = f"{y:04d}-{m:02d}-{d:02d}"
    day = rng.choice((str(d), f"{d:02d}")) if d < 10 else str(d)
    out = [
        (iso, iso),
        (f"{'1er' if d == 1 and rng.random() < 0.5 else day} {rng.choice((fr, fr.capitalize()))} {y}", iso),
        (f"{en} {d}{rng.choice(('', _ordinal(d)))}{rng.choice(('', ','))} {y}", iso),
        (f"{d}{rng.choice(('', _ordinal(d)))} {rng.choice(('', 'of '))}{en} {y}", iso),
        (f"{y}/{m}/{d}", iso),
    ]
    if m in _FR_ABBR:
        out.append((f"{d} {_FR_ABBR[m]} {y}", iso))
    if m in _EN_ABBR:
        out.append((f"{_EN_ABBR[m]}{rng.choice(('', '.'))} {d}, {y}", iso))
    if d > 12:
        separator = rng.choice("/.-")
        out.append((f"{d:02d}{separator}{m:02d}{separator}{y}", iso))
    return out


def test_py5_every_day_written_in_each_known_form_reads_to_its_canonical_form_at_its_bounds():
    probes, restore = _probes()
    try:
        for seed, case, rng in _cases(150):
            y, m = rng.randint(1000, 9999), rng.randint(1, 12)
            d = rng.randint(1, calendar.monthrange(y, m)[1])
            for writing, answer in _writings(rng, y, m, d):
                before, after = _filler(rng), _filler(rng)
                text = f"{before} {writing} {after}."
                read = [(t.answer, t.canonical, text[t.start:t.end]) for t in probes.read_dates(text)]
                assert read == [(answer, True, writing)], (seed, case, text, read)
    finally:
        restore()


def test_py6_a_day_the_calendar_does_not_hold_reads_as_no_date_and_nothing_inside_it_as_a_coarser_one():
    probes, restore = _probes()
    try:
        for seed, case, rng in _cases(150):
            y, m = rng.randint(1000, 9999), rng.randint(1, 12)
            last = calendar.monthrange(y, m)[1]
            if last == 31:
                continue
            d = rng.randint(last + 1, 31)
            fr, en = _FR_MONTHS[m - 1], _EN_MONTHS[m - 1]
            for writing in (f"{y:04d}-{m:02d}-{d:02d}", f"{d} {fr} {y}", f"{en} {d}, {y}", f"{d}/{m:02d}/{y}"):
                text = f"{_filler(rng)} {writing} {_filler(rng)}."
                assert probes.read_dates(text) == [], (seed, case, text)
        for writing in ("2026-13-05", "2026-00-10", "2026-04-00", "31 avril 2026", "April 31, 2026", "29 " + _FR_MONTHS[1]
                        + " 2027"):
            assert probes.read_dates(f"The review is on {writing} with the team.") == [], writing
    finally:
        restore()


# ---------------------------------------------------------------------------
# Numbers and units
# ---------------------------------------------------------------------------
def _grouped(integer, separator):
    digits = str(integer)
    head = len(digits) % 3 or 3
    return separator.join([digits[:head]] + [digits[i:i + 3] for i in range(head, len(digits), 3)])


def test_py7_a_number_reads_to_its_plain_value_with_its_written_precision_and_an_ambiguous_one_keeps_its_writing():
    probes, restore = _probes()
    try:
        for seed, case, rng in _cases(250):
            integer = rng.choice((rng.randint(0, 999), rng.randint(1000, 999999999)))
            fraction = rng.choice(("", "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 3)))))
            plain = str(integer) + ("." + fraction if fraction else "")
            # One separator before three figures reads two ways: 1.500, 1,500.
            lone = 1 <= integer <= 999 and len(fraction) == 3
            writings = (
                (plain, lone),
                (_grouped(integer, rng.choice((" ", _NBSP, _NNBSP))) + ("," + fraction if fraction else ""), lone),
                (_grouped(integer, ",") + ("." + fraction if fraction else ""),
                 lone or (1000 <= integer <= 999999 and not fraction)),
            )
            for writing, two_ways in writings:
                sign = rng.choice(("", "", "-", _MINUS))
                text = f"{_filler(rng)} {sign}{writing} {rng.choice(_TAIL)} {_filler(rng)}."
                read = probes.read_quantities(text)
                assert len(read) == 1, (seed, case, text, read)
                got = read[0]
                assert text[got.start:got.end] == sign + writing, (seed, case, text, got)
                if two_ways:
                    assert not got.canonical and got.answer == sign + writing, (seed, case, text, got)
                    continue
                value = plain if not sign or plain.strip("0.") == "" else "-" + plain
                assert got.canonical and got.answer == value, (seed, case, text, got, value)
    finally:
        restore()


def test_py8_a_unit_of_the_closed_table_reads_to_its_class_and_two_classes_are_two_answers():
    probes, restore = _probes()
    try:
        symbols = [(unit, written, False) for unit, *writings in probes._UNIT_SYMBOLS for written in writings]
        words = [(unit, written, True) for unit, *writings in probes._UNIT_WORDS for written in writings]
        rows = symbols + words
        for seed, case, rng in _cases(200):
            unit, written, word = rng.choice(rows)
            if word:  # a word is read in any case; a symbol only as written
                written = rng.choice((written, written.upper(), written.title()))
            number = rng.randint(1, 9999)
            text = f"{_filler(rng)} {number} {written} {_filler(rng)}."
            read = probes.read_quantities(text)
            assert [(t.answer, t.canonical) for t in read] == [(f"{number} {unit}", True)], (seed, case, text, read)
            other, other_written, _word = rng.choice(rows)
            if other != unit:
                second = probes.read_quantities(f"{_filler(rng)} {number} {other_written} {_filler(rng)}.")
                assert second and second[0].answer != read[0].answer, (seed, case, other_written, second)
        for prefix, unit in (("$", "$"), (chr(0x20AC), "EUR")):
            for gap in ("", " "):
                read = probes.read_quantities(f"It costs {prefix}{gap}120 in total.")
                assert [(t.answer, t.canonical) for t in read] == [(f"120 {unit}", True)], (prefix, gap, read)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------
_POOL = ("abcXYZ019 _-.,'" + "".join(chr(c) for c in range(0xC0, 0x180)) + "".join(chr(c) for c in range(0x391, 0x3CA))
         + "".join(chr(c) for c in range(0x410, 0x450)) + "".join(chr(c) for c in range(0x300, 0x370))
         + chr(0x4E00) + chr(0x4E8C) + chr(0x130) + chr(0x1E9E) + chr(0xDF) + chr(0x660) + chr(0x669))


def test_py9_folding_is_idempotent_leaves_no_capital_and_no_mark_and_on_ascii_gives_lower_case_runs():
    probes, restore = _probes()
    try:
        for seed, case, rng in _cases(400):
            text = "".join(rng.choice(_POOL) for _ in range(rng.randint(0, 40)))
            words = probes.folded_words(text)
            assert probes.folded_words(" ".join(words)) == words, (seed, case, text, words)
            assert all(w == w.lower() and not any(unicodedata.combining(c) for c in w) for w in words), (seed, case, text)
            assert all(unicodedata.normalize("NFD", w) == w for w in words), (seed, case, text, "an accent stays")
            ascii_text = "".join(c for c in text if c.isascii())
            assert probes.folded_words(ascii_text) == re.findall("[a-z0-9]+", ascii_text.lower()), (seed, case, ascii_text)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Robustness of the readers of dates and numbers
# ---------------------------------------------------------------------------
_DIGITS = ("0", "1", "2", "3", "9", "12", "31", "2026", "1999", ".", ",", "/", "-", _MINUS, " ", _NBSP, _NNBSP, "er",
           "st", "th", "of", "mars", "May", "oct.", "Oct", "f" + _E + "vrier", "Go", "GB", "Gio", "%", chr(0x20AC), "$",
           "km/h", "m" + chr(0xB2), "h", "min", "x", "v", "(", "\n")
_ISO = re.compile(r"\d{4}-\d{2}(-\d{2})?|--\d{2}-\d{2}")
_VALUE = re.compile(r"-?\d+(\.\d+)?( \S.*)?")


def test_py10_no_text_makes_the_date_and_number_readers_raise_and_what_they_read_keeps_its_shape():
    probes, restore = _probes()
    try:
        for seed, case, rng in _cases(500):
            text = "".join(rng.choice(_DIGITS) for _ in range(rng.randint(0, 30)))
            dates = probes.read_dates(text)
            end = 0
            for t in dates:
                assert end <= t.start < t.end <= len(text), (seed, case, text, dates)
                assert (_ISO.fullmatch(t.answer) is not None) if t.canonical else t.answer == text[t.start:t.end], (
                    seed, case, text, t)
                end = t.end
            for t in probes.read_quantities(text):
                assert 0 <= t.start < t.end <= len(text), (seed, case, text, t)
                assert (_VALUE.fullmatch(t.answer) is not None) if t.canonical else t.answer == text[t.start:t.end], (
                    seed, case, text, t)
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
