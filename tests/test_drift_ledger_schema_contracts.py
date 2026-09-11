#!/usr/bin/env python3
"""Contracts for the drift ledger's record and its supersession rule.

A memory fact carries today an id, a sentence, one of six categories, a
free-form source string and a use count. It carries no provenance by turn,
no confidence, no validity window, no link to the fact that replaced it. The
drift ledger is the record that does, and its one rule is that a fact is
never edited in place: it is superseded, the original stays, and a reader
who follows the chain arrives at the fact that currently holds.

This is the schema and its semantics, held in memory. Moving the physical
table to it is the last session of the block, not this one.

  * DR1 -- a complete record validates clean; a missing provenance, an
    out-of-range confidence and an unknown kind are each refused by name.
  * DR2 -- the ledger has no edit; superseding keeps the original intact,
    marks it, and links it to its successor.
  * DR3 -- a chain resolves to its active head, and the active set never
    contains a superseded fact.
  * DR4 -- a validity window that ends before it starts is refused, and an
    id already recorded is refused rather than overwritten.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the module is pure and reaches nothing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _open():
    loaded, restore = isolate(
        targets={"opti_oignon.memory.drift": source("memory", "drift.py")},
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.drift"], restore


def _fact(mod, **over):
    fields = dict(
        id="f1", statement="Alice lives in Berlin", kind="fact",
        provenance=["c1:t1"], confidence=0.9, valid_from="2024-03-15",
    )
    fields.update(over)
    return mod.LedgerFact(**fields)


# ---------------------------------------------------------------------------
# DR1 -- validation names what it refuses
# ---------------------------------------------------------------------------
def test_dr1_a_complete_record_validates_and_defects_are_named():
    mod, restore = _open()
    try:
        assert mod.validate_fact(_fact(mod)) == [], "a complete record is clean"
        for over, word in (
            ({"provenance": []}, "provenance"),
            ({"confidence": 1.5}, "confidence"),
            ({"kind": "rumour"}, "kind"),
        ):
            errors = mod.validate_fact(_fact(mod, **over))
            assert errors and any(word in e for e in errors), (
                f"a record with {over} is refused, naming {word!r}: {errors}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# DR2 -- no edit; supersession keeps the original
# ---------------------------------------------------------------------------
def test_dr2_supersession_keeps_the_original_and_there_is_no_edit():
    mod, restore = _open()
    try:
        ledger = mod.DriftLedger()
        assert not hasattr(ledger, "edit") and not hasattr(ledger, "update"), (
            "the ledger offers no way to change a fact in place"
        )
        ledger.add(_fact(mod))
        new_id = ledger.supersede("f1", _fact(mod, id="f2", statement="Alice lives in Paris", provenance=["c2:t7"]))
        assert new_id == "f2"
        old = ledger.get("f1")
        assert old.statement == "Alice lives in Berlin", "the original text is untouched"
        assert old.superseded_by == "f2", "and it points at its successor"
        assert old.status == "superseded"
        assert ledger.get("f2").status == "active"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DR3 -- a chain resolves to its head; active excludes the superseded
# ---------------------------------------------------------------------------
def test_dr3_a_chain_resolves_to_its_active_head():
    mod, restore = _open()
    try:
        ledger = mod.DriftLedger()
        ledger.add(_fact(mod))
        ledger.supersede("f1", _fact(mod, id="f2", statement="Alice lives in Paris"))
        ledger.supersede("f2", _fact(mod, id="f3", statement="Alice lives in Rome"))
        assert ledger.head("f1").id == "f3", "following the chain from the oldest lands on the head"
        assert ledger.head("f3").id == "f3", "the head is its own head"
        active = {f.id for f in ledger.active()}
        assert active == {"f3"}, f"only the head is active, got {active}"
        assert len(ledger.all()) == 3, "and nothing was ever removed"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DR4 -- refusals: an inverted window, a duplicate id
# ---------------------------------------------------------------------------
def test_dr4_an_inverted_window_and_a_duplicate_id_are_refused():
    mod, restore = _open()
    try:
        errors = mod.validate_fact(_fact(mod, valid_from="2024-05-01", valid_until="2024-03-01"))
        assert errors and any("valid" in e for e in errors), (
            f"a window ending before it starts is refused by name: {errors}"
        )
        ledger = mod.DriftLedger()
        ledger.add(_fact(mod))
        with pytest.raises(ValueError):
            ledger.add(_fact(mod, statement="Alice lives in Paris"))
        assert ledger.get("f1").statement == "Alice lives in Berlin", (
            "a second add under the same id is refused, never an overwrite"
        )
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
