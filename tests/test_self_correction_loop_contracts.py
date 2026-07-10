#!/usr/bin/env python3
"""Contracts for the self-correction critique/revision loop.

The engine re-scores a model's own answer and, when scores fall below the
configured thresholds, asks the model for an improved version -- a loop fed by
a critique that may itself be adversarial. These contracts pin the guard-rails
that make the loop safe to run against a hostile critique without pinning the
learned content of the critique or the heuristic scoring magnitudes.

  * C1 -- the loop is bounded: a critique that never satisfies the thresholds
    and always yields a fresh, distinct rewrite can still only drive as many
    correction attempts as the configured iteration bound, never more. This is
    the divergence guard.
  * C2 -- best-of, never worse: a correction that re-scores strictly worse than
    the original is never returned; the original stands and the result is
    marked uncorrected.
  * C3 -- preserve on no improvement: when the generator returns nothing or an
    unchanged answer, the loop stops before it counts an iteration and the
    original is returned untouched.
  * C4 -- no correction without the model: with the model path off, no
    correction is produced and the generator is never invoked, however far the
    scores sit below the thresholds.
  * C5 -- adversarial flags are capped: no matter how many factual flags the
    critique emits, at most three enter the correction prompt.
  * C6 -- scores are clamped: any model-returned score is bounded into the unit
    range and a non-numeric score falls to the caller default, so an
    out-of-range score can never wrongly satisfy or fail a threshold.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package, with the model backend faked per test so no
inference backend is required.
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
    """Load the real self_correction module under a stand-in package.

    Returns (module, restore). The module is a leaf (stdlib plus optional
    yaml/ollama), so nothing else is stubbed; the backend gate is opened and
    the generator faked per test.
    """
    keys = ("opti_oignon", "opti_oignon.self_correction")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.self_correction", _OO / "self_correction.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.self_correction"] = mod
    spec.loader.exec_module(mod)
    pkg.self_correction = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


# ---------------------------------------------------------------------------
# Local material
# ---------------------------------------------------------------------------
def _engine(mod, max_iter, check_facts=False):
    """Build an engine with an explicit, deterministic configuration."""
    return mod.SelfCorrectionEngine(config=mod.SelfCorrectionConfig(
        enable_auto=True,
        max_iterations=max_iter,
        compliance_threshold=0.7,
        quality_threshold=0.6,
        check_instructions=True,
        check_facts=check_facts,
        check_quality=True,
    ))


def _force_scores(mod, engine, value):
    """Pin every compliance/quality score the loop reads to a fixed value.

    Overrides both the engine's check methods (baseline scoring) and the
    module heuristics (loop re-scoring and the reported before-scores), so the
    loop's control flow is fully determined by ``value`` against the
    thresholds.
    """
    engine.check_compliance = lambda *a, **k: mod.ComplianceResult(score=value)
    engine.check_quality = lambda *a, **k: mod.QualityResult(overall_score=value)
    mod.compute_heuristic_compliance = (
        lambda *a, **k: mod.ComplianceResult(score=value)
    )
    mod.compute_heuristic_quality = (
        lambda *a, **k: mod.QualityResult(overall_score=value)
    )


# ---------------------------------------------------------------------------
# C1 -- the loop is bounded (divergence guard)
# ---------------------------------------------------------------------------
def test_c1_loop_is_bounded_by_max_iterations():
    for max_iter in (1, 2, 3, 5):
        mod, restore = _load()
        try:
            mod.OLLAMA_AVAILABLE = True
            engine = _engine(mod, max_iter)
            # Scores always below the thresholds so the loop is never satisfied.
            _force_scores(mod, engine, 0.30)
            # A critique that always yields a fresh, distinct, never-good-enough
            # rewrite: no unchanged-break, no threshold-break can fire.
            calls = {"n": 0}

            def generate(*_a, **_k):
                calls["n"] += 1
                return f"never satisfactory rewrite number {calls['n']} here"

            engine._generate_correction = generate
            engine.correct("answer in a list", "seed answer", use_llm=True)
            assert calls["n"] == max_iter, (
                "an adversarial critique that never satisfies must still stop "
                "at the configured iteration bound, not beyond"
            )
        finally:
            restore()


# ---------------------------------------------------------------------------
# C2 -- best-of, never worse
# ---------------------------------------------------------------------------
def test_c2_worse_correction_is_never_returned():
    mod, restore = _load()
    try:
        mod.OLLAMA_AVAILABLE = True
        engine = _engine(mod, 2)
        # Baseline needs correction (below thresholds), best score is 0.50.
        engine.check_compliance = lambda *a, **k: mod.ComplianceResult(score=0.50)
        engine.check_quality = lambda *a, **k: mod.QualityResult(overall_score=0.50)
        # One strictly-worse rewrite, then the generator stops.
        state = {"n": 0}

        def generate(*_a, **_k):
            state["n"] += 1
            return "strictly worse rewrite" if state["n"] == 1 else None

        engine._generate_correction = generate
        # The loop re-scores that rewrite strictly worse than the baseline.
        mod.compute_heuristic_compliance = (
            lambda *a, **k: mod.ComplianceResult(score=0.20)
        )
        mod.compute_heuristic_quality = (
            lambda *a, **k: mod.QualityResult(overall_score=0.20)
        )
        result = engine.correct("a question", "ORIGINAL ANSWER", use_llm=True)
        assert result.corrected_response == "ORIGINAL ANSWER", (
            "a correction that scores strictly worse must not replace the "
            "original answer"
        )
        assert result.was_corrected is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# C3 -- preserve on no improvement
# ---------------------------------------------------------------------------
def test_c3_unchanged_or_empty_correction_preserves_original():
    # Facet 1: an unchanged rewrite breaks the loop before it counts.
    mod, restore = _load()
    try:
        mod.OLLAMA_AVAILABLE = True
        engine = _engine(mod, 3)
        _force_scores(mod, engine, 0.40)
        engine._generate_correction = lambda _user, current, *a, **k: current
        result = engine.correct("a question", "KEEP ME", use_llm=True)
        assert result.corrected_response == "KEEP ME"
        assert result.was_corrected is False
        assert result.iterations_performed == 0, (
            "an unchanged rewrite must break the loop before it counts an "
            "iteration"
        )
    finally:
        restore()

    # Facet 2: a generator that cannot produce leaves the original untouched.
    mod, restore = _load()
    try:
        mod.OLLAMA_AVAILABLE = True
        engine = _engine(mod, 3)
        _force_scores(mod, engine, 0.40)
        engine._generate_correction = lambda *a, **k: None
        result = engine.correct("a question", "KEEP ME TOO", use_llm=True)
        assert result.corrected_response == "KEEP ME TOO"
        assert result.was_corrected is False
        assert result.iterations_performed == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# C4 -- no correction without the model
# ---------------------------------------------------------------------------
def test_c4_no_model_path_makes_no_correction():
    mod, restore = _load()
    try:
        # The backend is present, but a caller opting out of the model path
        # must still get no correction.
        mod.OLLAMA_AVAILABLE = True
        engine = _engine(mod, 2)
        # Scores far below the thresholds: a correction WOULD be attempted if
        # the model path were taken.
        _force_scores(mod, engine, 0.10)
        # A generator that would change the text, present only to prove it is
        # never called.
        called = {"n": 0}

        def generate(*_a, **_k):
            called["n"] += 1
            return "a rewrite that must never be produced without the model"

        engine._generate_correction = generate
        result = engine.correct("a question", "ORIGINAL", use_llm=False)
        assert called["n"] == 0, (
            "the correction generator must never run when the model path is off"
        )
        assert result.corrected_response == "ORIGINAL", (
            "with the model path off no correction may be produced"
        )
        assert result.was_corrected is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# C5 -- adversarial flags are capped in the prompt
# ---------------------------------------------------------------------------
def test_c5_adversarial_fact_flags_are_capped_in_the_prompt():
    mod, restore = _load()
    try:
        mod.OLLAMA_AVAILABLE = True
        captured = {}

        class _Backend:
            def generate(self, model, prompt, options=None):
                captured["prompt"] = prompt
                return {"response": "a corrected answer long enough to pass"}

        mod.ollama = _Backend()
        engine = _engine(mod, 2)
        flags = [
            mod.FactualFlag(claim=f"CLAIMTOKEN{i}", concern="c", severity="high")
            for i in range(10)
        ]
        facts = mod.FactCheckResult(flags=flags, flag_count=10)
        out = engine._generate_correction("q", "resp", None, None, facts, "m")
        assert out is not None, "the faked backend must yield a correction"
        seen = captured["prompt"].count("CLAIMTOKEN")
        assert seen == 3, (
            "no matter how many flags an adversarial critique emits, at most "
            "three may enter the correction prompt"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C6 -- scores are clamped into the unit range
# ---------------------------------------------------------------------------
def test_c6_score_clamp_bounds_model_scores():
    mod, restore = _load()
    try:
        clamp = mod._clamp01
        assert clamp(5.0, 0.5) == 1.0, "an over-range score must clamp to 1.0"
        assert clamp(-2.0, 0.5) == 0.0, "a negative score must clamp to 0.0"
        assert clamp(0.42, 0.9) == 0.42, "an in-range score passes through"
        assert clamp("not-a-number", 0.5) == 0.5, (
            "a non-numeric score must fall to the caller default"
        )
        assert clamp(None, 0.7) == 0.7
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("C1 loop bounded by max iterations", test_c1_loop_is_bounded_by_max_iterations),
        ("C2 worse correction never returned", test_c2_worse_correction_is_never_returned),
        ("C3 no-improvement preserves original", test_c3_unchanged_or_empty_correction_preserves_original),
        ("C4 no model path makes no correction", test_c4_no_model_path_makes_no_correction),
        ("C5 adversarial flags capped in prompt", test_c5_adversarial_fact_flags_are_capped_in_the_prompt),
        ("C6 score clamp bounds model scores", test_c6_score_clamp_bounds_model_scores),
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
