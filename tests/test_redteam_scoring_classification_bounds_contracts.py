#!/usr/bin/env python3
"""Contracts for classification bounds in the red team scoring module.

Two soundness properties keep campaign metrics trustworthy. First, every
scored attack must land in exactly one bucket: a score whose
classification is none of the three recognized values would otherwise
count toward totals but toward no bucket, silently deflating the reported
detection rate. Second, a target result that reports no defense score
(the attribute present but ``None``) must not crash the classifier; an
indeterminate defense score is treated as no detection (bypass), the
fail-secure reading for an audit tool.

  * Contract 1 -- bucket exhaustiveness: for a set of validly classified
    scores, the per-category, per-target, and per-strategy bucket counts
    each sum to that group's total, and the global buckets sum to the
    campaign total.
  * Contract 2 -- an unrecognized classification is rejected: aggregating
    a score whose classification is outside the recognized set raises
    rather than absorbing it silently.
  * Contract 3 -- the guard is not over-broad: each of the three
    recognized classifications aggregates without error.
  * Contract 4 -- a ``None`` defense score does not crash: it is read as
    0.0 and classified as a bypass (blocked being False).
  * Contract 5 -- a missing score attribute is read as 0.0 (bypass),
    matching the ``None`` case.
  * Contract 6 -- an explicit block wins even with a ``None`` score: the
    block short-circuit is unaffected by the None normalization.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The scoring module has no backend dependency and is
loaded in isolation under a stub package.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_MOD_NAME = "opti_oignon.redteam.scoring"


def _load_scoring():
    """Load the scoring module in isolation under a stub package."""
    saved = {
        name: sys.modules.get(name)
        for name in ("opti_oignon", "opti_oignon.redteam", _MOD_NAME)
    }
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    rt = types.ModuleType("opti_oignon.redteam")
    rt.__path__ = []
    sys.modules["opti_oignon.redteam"] = rt
    pkg.redteam = rt

    spec = importlib.util.spec_from_file_location(
        _MOD_NAME, _OO / "redteam" / "scoring.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod, saved


def _restore(saved):
    sys.modules.pop(_MOD_NAME, None)
    for name, value in saved.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


class _Result:
    """Minimal duck-typed stand-in for a TargetResult."""

    def __init__(self, score=0.0, blocked=False, target_name="t",
                 metadata=None, has_score=True):
        if has_score:
            self.score = score
        self.blocked = blocked
        self.target_name = target_name
        self.metadata = metadata if metadata is not None else {}


def _mk(scoring, classification, category="c", strategy="s", target="t"):
    return scoring.AttackScore(
        category=category,
        strategy=strategy,
        target=target,
        classification=classification,
        defense_score=0.0,
        blocked=False,
    )


def test_c1_buckets_are_exhaustive_for_valid_scores():
    scoring, saved = _load_scoring()
    try:
        scores = [
            _mk(scoring, scoring.CLASSIFICATION_BYPASS, category="a", target="x"),
            _mk(scoring, scoring.CLASSIFICATION_FLAG, category="a", target="x"),
            _mk(scoring, scoring.CLASSIFICATION_BLOCK, category="b", target="y"),
            _mk(scoring, scoring.CLASSIFICATION_BLOCK, category="b", target="y"),
        ]
        camp = scoring.aggregate_scores(scores)
        assert (
            camp.total_bypasses + camp.total_flags + camp.total_blocks
            == camp.total
        ), "global buckets must sum to the campaign total"
        for group in (camp.by_category, camp.by_target, camp.by_strategy):
            for bd in group.values():
                assert bd.bypasses + bd.flags + bd.blocks == bd.total, (
                    "each group's buckets must sum to its total"
                )
    finally:
        _restore(saved)


def test_c2_unrecognized_classification_is_rejected():
    scoring, saved = _load_scoring()
    try:
        rogue = _mk(scoring, "MYSTERY")
        good = _mk(scoring, scoring.CLASSIFICATION_BYPASS)
        raised = False
        try:
            scoring.aggregate_scores([good, rogue])
        except ValueError:
            raised = True
        assert raised, (
            "aggregating a score with an unrecognized classification must "
            "raise rather than absorb it silently"
        )
    finally:
        _restore(saved)


def test_c3_all_recognized_classifications_aggregate():
    scoring, saved = _load_scoring()
    try:
        for cls in (
            scoring.CLASSIFICATION_BYPASS,
            scoring.CLASSIFICATION_FLAG,
            scoring.CLASSIFICATION_BLOCK,
        ):
            camp = scoring.aggregate_scores([_mk(scoring, cls)])
            assert camp.total == 1, (
                f"a recognized classification {cls!r} must aggregate cleanly"
            )
    finally:
        _restore(saved)


def test_c4_none_defense_score_does_not_crash():
    scoring, saved = _load_scoring()
    try:
        result = _Result(score=None, blocked=False)
        score = scoring.score_result(result, category="c", strategy="s")
        assert score.defense_score == 0.0, (
            "a None defense score must be read as 0.0"
        )
        assert score.classification == scoring.CLASSIFICATION_BYPASS, (
            "an indeterminate (0.0) defense score with no block is a bypass"
        )
    finally:
        _restore(saved)


def test_c5_missing_score_attribute_is_zero():
    scoring, saved = _load_scoring()
    try:
        result = _Result(has_score=False, blocked=False)
        score = scoring.score_result(result, category="c", strategy="s")
        assert score.defense_score == 0.0, (
            "a missing score attribute must default to 0.0"
        )
        assert score.classification == scoring.CLASSIFICATION_BYPASS
    finally:
        _restore(saved)


def test_c6_block_wins_even_with_none_score():
    scoring, saved = _load_scoring()
    try:
        result = _Result(score=None, blocked=True)
        score = scoring.score_result(result, category="c", strategy="s")
        assert score.classification == scoring.CLASSIFICATION_BLOCK, (
            "an explicit block must classify as block regardless of score"
        )
    finally:
        _restore(saved)


_TESTS = [
    test_c1_buckets_are_exhaustive_for_valid_scores,
    test_c2_unrecognized_classification_is_rejected,
    test_c3_all_recognized_classifications_aggregate,
    test_c4_none_defense_score_does_not_crash,
    test_c5_missing_score_attribute_is_zero,
    test_c6_block_wins_even_with_none_score,
]


def _main():
    passed = 0
    for test in _TESTS:
        try:
            test()
        except Exception:
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}")
            passed += 1
    print(f"{passed}/{len(_TESTS)} passed")
    return 0 if passed == len(_TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
