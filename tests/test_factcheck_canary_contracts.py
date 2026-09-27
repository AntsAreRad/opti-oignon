#!/usr/bin/env python3
"""A canary of planted errors runs at every construction, through the check that answers.

The canary is a fixed set of claims with their evidence and the verdict each
must receive: planted errors that must never come out supported, and positive
controls that must. It proves the verifier able to refuse on the classes it
holds; it does not measure its power.

  * PE1 -- every canary item comes out as planted, through the production
    path: every item gets its expected verdict, no planted error is
    supported, every positive control is; the canary runs the checker's own
    check function (the same object that answers the owner, never a copy);
    every class of the canary has an item in English and one in French.
  * PE2 -- the canary runs at every construction, and a failing canary
    refuses: two constructions are two runs; a checker built over a check
    that supports everything is refused, and every call then returns a
    refusal naming the failing items, never a verdict and never a record;
    every record carries the canary's digest and outcome, and changing one
    item changes the digest.

Loaded through the shared isolation window (``tests/_factcheck.py``), with the
native core unreachable. Nothing reaches a model, the network or the
maintainer's data.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _factcheck as F  # noqa: E402

BUDGET_S = {
    "test_pe1_every_canary_item_comes_out_as_planted_through_the_production_path": 2.0,
    "test_pe2_the_canary_runs_at_every_construction_and_a_failing_canary_refuses": 2.0,
}

CLASSES = ("passage", "restatement", "provenance", "time", "scope", "context", "person", "claim_only")


@pytest.fixture
def fc():
    ns, restore = F.load()
    try:
        yield ns
    finally:
        restore()


def test_pe1_every_canary_item_comes_out_as_planted_through_the_production_path(fc):
    cfg = F.config(fc)
    rows = fc.canary.rows()

    # c1 -- every item gets its expected verdict, through decide.check.
    result = fc.canary.run(fc.decide.check, config=cfg)
    assert result.n == len(rows) >= 40 and result.failing == (), result.failing
    assert result.outcome == "pass"
    planted = controls = 0
    for row, verdict in result.results:
        assert verdict.value == row.expected_value, (row.name, verdict.value, verdict.reasons)
        if row.expected_reason:
            assert row.expected_reason in verdict.reasons, (row.name, verdict.reasons)
        if row.role == "planted":
            planted += 1
            assert verdict.value != "supported", row.name
        else:
            controls += 1
            assert verdict.value == "supported", (row.name, verdict.reasons)
    assert planted >= 30 and controls >= 8, (planted, controls)

    # c2 -- the canary runs the checker's own check function, not a copy.
    seen = []
    real_run = fc.canary.run
    answered = []

    def check(*args, **kwargs):
        answered.append(args[0])
        return fc.decide.check(*args, **kwargs)

    def spy(function, *args, **kwargs):
        seen.append(function)
        return real_run(function, *args, **kwargs)

    fc.canary.run = spy
    try:
        checker = fc.checker.FactChecker(check=check)
    finally:
        fc.canary.run = real_run
    assert len(seen) == 1 and seen[0] is check and checker.check_function is check
    assert not checker.refused and len(answered) == len(rows)
    verdict = checker.check("The Moon is 384,400 km away.", [], as_of=F.AS_OF, read_on=F.READ_ON)
    assert verdict.value == "not_enough_evidence" and len(answered) == len(rows) + 1

    # c3 -- every class has an item in English and in French.
    for cls in CLASSES:
        langs = {row.lang for row in rows if row.cls == cls}
        assert {"en", "fr"} <= langs, (cls, langs)
    assert {row.cls for row in rows} == set(CLASSES)


def _supports_everything(fc):
    def check(claim, items, **kwargs):
        return fc.vocabulary.Verdict("supported", "deterministic", ("verbatim_sentence",),
                                     text="Stated.", record={"canary": kwargs.get("canary")})
    return check


def test_pe2_the_canary_runs_at_every_construction_and_a_failing_canary_refuses(fc):
    # c1 -- two constructions, two runs.
    runs = []
    real_run = fc.canary.run

    def counting(function, *args, **kwargs):
        runs.append(function)
        return real_run(function, *args, **kwargs)

    fc.canary.run = counting
    try:
        first = fc.checker.FactChecker()
        second = fc.checker.FactChecker()
    finally:
        fc.canary.run = real_run
    assert len(runs) == 2 and not first.refused and not second.refused

    # c2 -- a checker over a check that supports everything is refused.
    broken = fc.checker.FactChecker(check=_supports_everything(fc))
    assert broken.refused and broken.canary.outcome == "fail" and broken.canary.failing
    answer = broken.check("The Moon is 384,400 km away.", [], as_of=F.AS_OF, read_on=F.READ_ON)
    assert isinstance(answer, fc.vocabulary.Refusal) and not isinstance(answer, fc.vocabulary.Verdict)
    assert not hasattr(answer, "record") and not hasattr(answer, "value")
    assert answer.failing == broken.canary.failing and answer.failing[0] in answer.text
    listed = broken.claims_from_text("The Moon is 384,400 km away.", lang="en", origin={})
    assert isinstance(listed, fc.vocabulary.Refusal)

    # c3 -- every record carries the canary's digest and outcome; one item
    # changed changes the digest.
    moon = "The Moon is 384,400 km away."
    verdict = first.check(moon, [F.item(fc, "library:moon", moon)], as_of=F.AS_OF, read_on=F.READ_ON)
    assert verdict.record["canary"] == {"digest": first.canary.digest, "n": first.canary.n,
                                        "outcome": "pass"}
    assert first.canary.digest == fc.canary.digest()
    rows = list(fc.canary.rows())
    claim = rows[0].claim
    changed = claim + " " if isinstance(claim, str) else dataclasses.replace(claim, text=claim.text + " ")
    rows[0] = dataclasses.replace(rows[0], claim=changed)
    assert fc.canary.digest(rows) != fc.canary.digest()
