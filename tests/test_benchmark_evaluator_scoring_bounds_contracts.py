#!/usr/bin/env python3
"""Contracts for the benchmark evaluator's scoring bounds and fail-safes.

The evaluator scores model outputs deterministically (no judge model): exact,
fuzzy, and keyword accuracy against ground truth, plus performance and a
weighted composite. The safety-relevant properties are that degenerate
answers can never score as matches, that a dead generation is never scored as
fast, and that the composite stays a bounded weighted average that fails to
zero when nothing was evaluated. These contracts pin those guards without
pinning tolerances, weight values, or the question catalog.

  * BE1 -- degenerate answers never match: an empty extracted answer and a
    single-character answer both score 0.0 under exact scoring, so reverse
    containment cannot be gamed by trivial output.
  * BE2 -- legitimate reverse containment still works: a short valid answer
    contained in the expected string scores 1.0, so the degenerate guard is
    a minimum length, not a blanket rejection.
  * BE3 -- a dead generation scores zero performance: no token and no first
    token means failure, never an instant answer.
  * BE4 -- zero evaluated weight yields a zero composite: when no axis was
    evaluated the composite is 0.0, never spurious credit.
  * BE5 -- the composite renormalizes over evaluated axes: with a single
    evaluated axis the composite equals that axis score.
  * BE6 -- an unknown scoring method scores 0.0 and is labeled unknown, so
    a bad catalog entry degrades instead of inflating accuracy.
  * BE7 -- the keyword score is the found fraction, bounded to [0, 1].

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package, so no sandbox backend and no sibling module
are required.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load():
    """Load the real evaluator under a stand-in package.

    Returns (module, restore). The stand-in package has an empty search
    path, so the optional sandbox import degrades and code execution is
    never attempted by anything these tests touch.
    """
    keys = ("opti_oignon", "opti_oignon.benchmark_evaluator")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.benchmark_evaluator", _OO / "benchmark_evaluator.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.benchmark_evaluator"] = mod
    spec.loader.exec_module(mod)
    pkg.benchmark_evaluator = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
def test_be1_degenerate_answers_never_match_exact():
    mod, restore = _load()
    try:
        score, matched = mod.score_exact("", ["4", "four"])
        assert score == 0.0, f"empty response scored {score}, expected 0.0"
        assert matched == ""

        score, matched = mod.score_exact(
            "x", ["extremely long expected answer x"],
        )
        assert score == 0.0, (
            f"single-character answer scored {score} via reverse containment"
        )
    finally:
        restore()


def test_be2_short_answer_reverse_containment_matches():
    mod, restore = _load()
    try:
        score, matched = mod.score_exact("Paris", ["the city of paris"])
        assert score == 1.0, (
            f"valid contained answer scored {score}, expected 1.0"
        )
        assert matched == "the city of paris"
    finally:
        restore()


def test_be3_dead_generation_scores_zero_performance():
    mod, restore = _load()
    try:
        dead = mod.evaluate_performance(
            ttft_ms=0.0, tokens_per_second=0.0, total_time_ms=0.0,
        )
        assert dead.score == 0.0, (
            f"generation with no token scored {dead.score}, expected 0.0"
        )

        alive = mod.evaluate_performance(
            ttft_ms=100.0, tokens_per_second=50.0, total_time_ms=1000.0,
        )
        assert alive.score == 1.0, "a fast healthy generation should score 1.0"
    finally:
        restore()


def test_be4_zero_evaluated_weight_yields_zero_composite():
    mod, restore = _load()
    try:
        composite = mod.compute_composite_score(
            1.0, 1.0, 1.0, 1.0, evaluated=set(),
        )
        assert composite == 0.0, (
            f"composite with no evaluated axis was {composite}, expected 0.0"
        )
    finally:
        restore()


def test_be5_single_axis_composite_renormalizes():
    mod, restore = _load()
    try:
        composite = mod.compute_composite_score(
            0.8, 0.0, 0.0, 0.0, evaluated={"accuracy"},
        )
        assert abs(composite - 0.8) < 1e-9, (
            f"single-axis composite was {composite}, expected 0.8"
        )

        composite = mod.compute_composite_score(
            0.0, 0.0, 0.0, 0.4, evaluated={"speed"},
        )
        assert abs(composite - 0.4) < 1e-9, (
            f"single-axis composite was {composite}, expected 0.4"
        )
    finally:
        restore()


def test_be6_unknown_scoring_method_scores_zero():
    mod, restore = _load()
    try:
        class _OddQuestion:
            id = "odd-1"
            scoring = "__not_a_method__"
            expected = ["42"]
            keywords = []
            tolerance = 0.0

        result = mod.evaluate_accuracy(_OddQuestion(), "42")
        assert result.score == 0.0, (
            f"unknown scoring method scored {result.score}, expected 0.0"
        )
        assert result.method == "unknown"
    finally:
        restore()


def test_be7_keyword_score_is_bounded_fraction():
    mod, restore = _load()
    try:
        score, found = mod.score_keyword(
            "alpha beta gamma", ["alpha", "beta", "delta"],
        )
        assert abs(score - 2.0 / 3.0) < 1e-9, (
            f"keyword score was {score}, expected 2/3"
        )
        assert 0.0 <= score <= 1.0
        assert "alpha" in found and "beta" in found

        score, found = mod.score_keyword("anything", [])
        assert score == 0.0 and found == ""
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("BE1 degenerate answers never match exact",
         test_be1_degenerate_answers_never_match_exact),
        ("BE2 short answer reverse containment matches",
         test_be2_short_answer_reverse_containment_matches),
        ("BE3 dead generation scores zero performance",
         test_be3_dead_generation_scores_zero_performance),
        ("BE4 zero evaluated weight yields zero composite",
         test_be4_zero_evaluated_weight_yields_zero_composite),
        ("BE5 single axis composite renormalizes",
         test_be5_single_axis_composite_renormalizes),
        ("BE6 unknown scoring method scores zero",
         test_be6_unknown_scoring_method_scores_zero),
        ("BE7 keyword score is bounded fraction",
         test_be7_keyword_score_is_bounded_fraction),
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
