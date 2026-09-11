#!/usr/bin/env python3
"""Contracts that the fuzzy accuracy tolerance is a gate and not a decoration.

Fourteen of the shipped benchmark questions are scored by fuzzy similarity and
each declares a tolerance between 0.70 and 0.85. The scorer read that
tolerance, compared the best similarity against it, and then returned the same
tuple from both branches -- so the threshold had no effect and a wrong answer
kept whatever incidental string overlap it happened to have. Measured over all
fourteen with the real scorer, a deliberately wrong answer scored a mean of
0.211 where the declared tolerances call for 0.000, and eleven of the fourteen
credited a wrong answer below their own threshold. That credit entered
accuracy_avg, and through it the composite, on every run.

These contracts pin the gate itself rather than any particular question's
number, so the question catalogue stays free to change:

  * FZ1 -- with no threshold to clear, a partial match keeps its similarity.
    This is also the non-zero half of the probe: the scorer is shown able to
    return a positive score on input that should produce one, so FZ2's zero
    means the gate fired rather than the scorer being inert.
  * FZ2 -- the same input, under a threshold it cannot clear, scores exactly
    zero rather than partial credit.
  * FZ3 -- a true match is not swallowed by a demanding threshold, and a gated
    result still names the expectation it was measured against, so a zero is
    explainable instead of silent.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window with the sandbox manager declared unreachable: the evaluator
imports it at module scope inside a try, and reaching the real one would pull
in the whole package.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

# One expectation and one answer that overlaps it in part without being it.
# Chosen so the similarity is strictly between 0 and 1; the contracts below
# never assert what it equals, only which side of a threshold it falls.
_EXPECTED = ["the powerhouse of the cell"]
_PARTIAL = "the power supply of the city"


def _open():
    """Open the shared window on the evaluator alone."""
    loaded, restore = isolate(
        targets={
            "opti_oignon.benchmark_evaluator": source("benchmark_evaluator.py"),
        },
        blocked=("opti_oignon.sandbox_manager",),
    )
    return loaded["opti_oignon.benchmark_evaluator"], restore


# ---------------------------------------------------------------------------
# FZ1 -- with nothing to clear, a partial match keeps its similarity
# ---------------------------------------------------------------------------
def test_fz1_a_partial_match_scores_above_zero_when_nothing_gates_it():
    mod, restore = _open()
    try:
        score, match = mod.score_fuzzy(_PARTIAL, _EXPECTED, 0.0)
        assert score > 0.0, (
            "the scorer returns a positive similarity on input that should "
            "produce one, so a zero elsewhere is the gate and not inertia"
        )
        assert score < 1.0, (
            "the chosen answer really is a partial match, not the expectation"
        )
        assert match == _EXPECTED[0], "the expectation it matched is named"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FZ2 -- below the threshold, the score is zero and not partial credit
# ---------------------------------------------------------------------------
def test_fz2_a_similarity_below_the_tolerance_scores_zero():
    mod, restore = _open()
    try:
        ungated, _ = mod.score_fuzzy(_PARTIAL, _EXPECTED, 0.0)
        gated, _ = mod.score_fuzzy(_PARTIAL, _EXPECTED, 0.99)
        assert ungated > 0.0, (
            "the same input scores above zero when nothing gates it, so the "
            "comparison below is between two verdicts and not two inputs"
        )
        assert gated == 0.0, (
            "a similarity that cannot clear its declared tolerance earns no "
            "accuracy credit at all"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# FZ3 -- a true match survives, and a gated zero still names its expectation
# ---------------------------------------------------------------------------
def test_fz3_the_gate_spares_a_true_match_and_explains_a_zero():
    mod, restore = _open()
    try:
        score, match = mod.score_fuzzy(_EXPECTED[0], _EXPECTED, 0.99)
        assert score == 1.0, (
            "the answer that is the expectation is not swallowed by a "
            "demanding threshold"
        )
        assert match == _EXPECTED[0], "the match is named"

        gated_score, gated_match = mod.score_fuzzy(_PARTIAL, _EXPECTED, 0.99)
        assert gated_score == 0.0, "the partial answer is gated"
        assert gated_match == _EXPECTED[0], (
            "a gated zero still names what it was measured against, so the "
            "refusal can be read rather than guessed at"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("FZ1 partial match scores above zero ungated", test_fz1_a_partial_match_scores_above_zero_when_nothing_gates_it),
        ("FZ2 below tolerance scores zero", test_fz2_a_similarity_below_the_tolerance_scores_zero),
        ("FZ3 gate spares a true match", test_fz3_the_gate_spares_a_true_match_and_explains_a_zero),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
