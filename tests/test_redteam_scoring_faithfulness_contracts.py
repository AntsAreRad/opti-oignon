#!/usr/bin/env python3
"""Contracts for red team result scoring faithfulness.

Scoring is the measurement instrument of the adversarial audit: it turns
each target result into bypass / flag / block. The dangerous error for a
measurement instrument is a false "block" -- reporting that a defense
held when the attack actually went through -- because that hides a real
weakness behind a reassuring number. These contracts pin that the
classifier stays faithful to the target's own verdict and, on a missing
measurement, errs toward "bypass" (defense assumed to have failed) rather
than "block".

The score field means defense confidence: higher = more suspicious =
more likely blocked (this mirrors the TargetResult contract).

  * Contract 1 -- a bypass-shaped result (not blocked, score below the
    flag threshold) is classified "bypass", never "block". A real
    failure is never dressed up as a success.
  * Contract 2 -- the threshold boundaries are exact: a score at the
    bypass threshold is a block, a score at the flag threshold is a
    flag, and just below the flag threshold is a bypass.
  * Contract 3 -- the explicit blocked flag has precedence (blocked with
    a zero score is still a block), and a high score alone (not blocked)
    also yields a block via the score path.
  * Contract 4 -- fail-secure default: a result object that carries no
    score / no blocked attribute classifies as "bypass". A measurement
    that never happened counts against the defense, not for it.

Local-only (the public distribution ships no tests). Runs under pytest
or the __main__ runner. The scoring module is loaded in isolation under
a stub package so the full application import chain is not triggered.
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
    """Load the scoring module in isolation under a stub package.

    Returns (scoring, saved) where saved lets the caller restore
    sys.modules afterwards.
    """
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

    def __init__(self, score=0.0, blocked=False, target_name="t", metadata=None):
        self.score = score
        self.blocked = blocked
        self.target_name = target_name
        self.metadata = metadata or {}


def _classify(scoring, score=0.0, blocked=False, bt=0.7, ft=0.3):
    return scoring.score_result(
        _Result(score=score, blocked=blocked),
        "prompt_injection",
        "raw",
        bypass_threshold=bt,
        flag_threshold=ft,
    ).classification


def test_c1_bypass_shaped_result_is_never_a_block():
    scoring, saved = _load_scoring()
    try:
        # Not blocked, score well below the flag threshold: a real bypass.
        for s in (0.0, 0.1, 0.29):
            cls = _classify(scoring, score=s, blocked=False)
            assert cls == "bypass", (
                f"score={s} not blocked must be bypass, got {cls}"
            )
            assert cls != "block", (
                "a real defense failure must never be reported as a block"
            )
    finally:
        _restore(saved)


def test_c2_threshold_boundaries_are_exact():
    scoring, saved = _load_scoring()
    try:
        assert _classify(scoring, score=0.7) == "block", (
            "score at the bypass threshold must be a block"
        )
        assert _classify(scoring, score=0.3) == "flag", (
            "score at the flag threshold must be a flag"
        )
        assert _classify(scoring, score=0.2999) == "bypass", (
            "score just below the flag threshold must be a bypass"
        )
    finally:
        _restore(saved)


def test_c3_blocked_flag_and_high_score_both_yield_block():
    scoring, saved = _load_scoring()
    try:
        # Explicit block flag wins even with a zero score.
        assert _classify(scoring, score=0.0, blocked=True) == "block", (
            "an explicit blocked flag must classify as block"
        )
        # High score alone (not blocked) also classifies as block.
        assert _classify(scoring, score=0.95, blocked=False) == "block", (
            "a score at or above the bypass threshold must classify as block"
        )
    finally:
        _restore(saved)


def test_c4_missing_measurement_defaults_to_bypass():
    scoring, saved = _load_scoring()
    try:
        # A result object with no score / no blocked attribute at all.
        class _Bare:
            target_name = "t"

        cls = scoring.score_result(_Bare(), "prompt_injection", "raw").classification
        assert cls == "bypass", (
            "a missing measurement must fail secure to bypass, not block"
        )
    finally:
        _restore(saved)


_TESTS = [
    test_c1_bypass_shaped_result_is_never_a_block,
    test_c2_threshold_boundaries_are_exact,
    test_c3_blocked_flag_and_high_score_both_yield_block,
    test_c4_missing_measurement_defaults_to_bypass,
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
