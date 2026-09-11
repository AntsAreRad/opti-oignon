#!/usr/bin/env python3
"""Contracts for the deterministic contradiction templates and the injection
harness that proves them capable.

Drift is the slow accumulation of statements that cannot all be true. Today
nothing in the memory package detects it: the only mention of contradiction is
a line in a prompt handed to an optional curator model. These templates are
the deterministic half the design asks for -- negation, mutual exclusion on a
closed set of exclusive predicates, and a number or date that disagrees for
the same subject -- so that a contradiction becomes a count in CI, while the
model judge stays a host runbook.

A detector that finds nothing on a consistent set proves nothing on its own;
the harness injects known contradictions and requires each one back by name.

  * CT1 -- a consistent set yields no contradiction.
  * CT2 -- a negation of a held fact is detected.
  * CT3 -- a different value for an exclusive attribute is detected.
  * CT4 -- a disagreeing number or date for the same subject is detected.
  * CT5 -- a different, non-exclusive predicate is not a contradiction.
  * CT6 -- the injection harness gets exactly its injections back, each
    naming the fact it contradicts, and injecting nothing yields nothing.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the module is pure and reaches nothing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_HELD = [
    ("f1", "Alice lives in Berlin"),
    ("f2", "Bob works at Contoso"),
    ("f3", "The release is on 2024-05-01"),
    ("f4", "The budget is 1200 euros"),
    ("f5", "Alice prefers tea"),
]


def _open():
    loaded, restore = isolate(
        targets={"opti_oignon.memory.drift": source("memory", "drift.py")},
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.drift"], restore


def _ids(found):
    return sorted(c.fact_id for c in found)


# ---------------------------------------------------------------------------
# CT1 -- a consistent set is quiet
# ---------------------------------------------------------------------------
def test_ct1_a_consistent_set_yields_no_contradiction():
    mod, restore = _open()
    try:
        for _fid, statement in _HELD:
            assert mod.find_contradictions(_HELD, statement) == [], (
                f"a statement already held contradicts nothing: {statement!r}"
            )
        assert mod.find_contradictions(_HELD, "Carol lives in Oslo") == [], (
            "a new subject contradicts nothing"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CT2 -- negation
# ---------------------------------------------------------------------------
def test_ct2_a_negation_is_detected():
    mod, restore = _open()
    try:
        found = mod.find_contradictions(_HELD, "Alice does not live in Berlin")
        assert _ids(found) == ["f1"], f"the negated fact is named, got {found}"
        assert found[0].template == "negation"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CT3 -- exclusive attribute
# ---------------------------------------------------------------------------
def test_ct3_a_different_exclusive_attribute_is_detected():
    mod, restore = _open()
    try:
        found = mod.find_contradictions(_HELD, "Alice lives in Paris")
        assert _ids(found) == ["f1"], f"one place of residence at a time: {found}"
        assert found[0].template == "exclusive"
        found = mod.find_contradictions(_HELD, "Bob works at Fabrikam")
        assert _ids(found) == ["f2"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# CT4 -- number or date disagreement
# ---------------------------------------------------------------------------
def test_ct4_a_disagreeing_number_or_date_is_detected():
    mod, restore = _open()
    try:
        found = mod.find_contradictions(_HELD, "The release is on 2024-06-01")
        assert _ids(found) == ["f3"], f"the date moved: {found}"
        assert found[0].template == "value"
        found = mod.find_contradictions(_HELD, "The budget is 1300 euros")
        assert _ids(found) == ["f4"], f"the number moved: {found}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CT5 -- a non-exclusive predicate is not a contradiction
# ---------------------------------------------------------------------------
def test_ct5_a_different_non_exclusive_predicate_is_not_a_contradiction():
    mod, restore = _open()
    try:
        assert mod.find_contradictions(_HELD, "Alice likes Berlin") == [], (
            "liking a city and living in it are not exclusive of each other"
        )
        assert mod.find_contradictions(_HELD, "Alice prefers coffee in the morning") == [], (
            "a preference with a qualifier is not the same claim, so no template fires"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CT6 -- the injection harness
# ---------------------------------------------------------------------------
def test_ct6_injected_contradictions_come_back_by_name():
    mod, restore = _open()
    try:
        statements, expected = mod.inject_contradictions(_HELD, count=3)
        assert len(statements) == 3 and len(expected) == 3, (
            "the harness produced exactly what was asked"
        )
        for statement, fact_id in zip(statements, expected):
            found = mod.find_contradictions(_HELD, statement)
            assert fact_id in _ids(found), (
                f"the injected contradiction {statement!r} is detected against "
                f"{fact_id}; found {found}"
            )
        assert mod.inject_contradictions(_HELD, count=0) == ([], []), (
            "injecting nothing yields nothing, so the count above is real"
        )
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
