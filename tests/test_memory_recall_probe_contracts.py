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

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the module is pure and reaches nothing.
"""

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
