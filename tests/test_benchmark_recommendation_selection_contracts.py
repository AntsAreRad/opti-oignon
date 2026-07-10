#!/usr/bin/env python3
"""Contracts for the benchmark recommender's selection and apply path.

The recommender turns per-model benchmark scores into role recommendations
(quality, fast, code, value) and exposes the single place where a
recommendation can change routing: an explicit apply that pushes the quality
model into the smart router. The safety-relevant properties are that the
selection is a structural argmax that can never prefer a strictly worse
model for its role, that the fast role respects a quality floor, and that
the apply path is fail-safe and pushes exactly the quality model. These
contracts pin those guards without pinning score values or the threshold
constant.

  * RC1 -- empty input yields an empty snapshot: no scores means no
    recommendations and no crash, never a fabricated pick.
  * RC2 -- the quality role is the composite argmax, regardless of input
    order.
  * RC3 -- the code role is the code-score argmax, not a speed pick.
  * RC4 -- the fast role respects the quality floor: a fast model below the
    acceptable-quality threshold is excluded even if it has the top speed.
  * RC5 -- the value role never rewards zero quality: a zero-composite model
    cannot win value on speed alone.
  * RC6 -- apply without any snapshot is refused: it reports not applied and
    raises nothing, so routing is never touched on empty state.
  * RC7 -- apply pushes exactly the quality model as the router default and
    marks the snapshot applied in the store.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package with an empty search path, storage goes to a
temporary database, and the router is a recording stand-in, so no application
stack is required.
"""

import importlib.util
import sys
import tempfile
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_KEYS = (
    "opti_oignon",
    "opti_oignon.benchmark_recommendations",
    "opti_oignon.smart_router",
)


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load():
    """Load the real recommender under a stand-in package.

    Returns (module, restore). The stand-in package has an empty search
    path, so the encrypted-connection helper degrades to plain sqlite for
    the temporary test databases and the router import resolves only to
    whatever stand-in a test installs.
    """
    saved = {k: sys.modules.get(k) for k in _KEYS}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.benchmark_recommendations",
        _OO / "benchmark_recommendations.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.benchmark_recommendations"] = mod
    spec.loader.exec_module(mod)
    pkg.benchmark_recommendations = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


def _make_recommender(mod):
    """Build a recommender backed by a fresh temporary database."""
    tmp = Path(tempfile.mkdtemp(prefix="oo-rec-"))
    db_path = tmp / "recommendations.db"
    store = mod.RecommendationStore(db_path=db_path)
    return mod.BenchmarkRecommender(store=store, db_path=db_path), store


def _score(model, composite, speed, accuracy=0.5, code=0.5, structure=0.5):
    """A per-model score row in the run-level key style."""
    return {
        "model": model,
        "composite": composite,
        "speed_avg": speed,
        "accuracy_avg": accuracy,
        "code_avg": code,
        "structure_avg": structure,
    }


class _RouterRecorder:
    """Stand-in smart router that records configure calls."""

    def __init__(self):
        self.configured = []

    def configure(self, **kwargs):
        self.configured.append(dict(kwargs))


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
def test_rc1_empty_scores_yield_empty_snapshot():
    mod, restore = _load()
    try:
        recommender, store = _make_recommender(mod)
        snapshot = recommender.generate_from_scores([])
        assert snapshot.recommendations == [], (
            "no input scores must produce no recommendations"
        )
        assert snapshot.snapshot_id
        assert store.get_latest() is None, (
            "an empty snapshot must not be persisted as latest"
        )
    finally:
        restore()


def test_rc2_quality_role_is_composite_argmax():
    mod, restore = _load()
    try:
        recommender, _ = _make_recommender(mod)
        snapshot = recommender.generate_from_scores([
            _score("second-best", composite=0.4, speed=0.9),
            _score("best", composite=0.9, speed=0.1),
        ])
        quality = snapshot.get_recommendation(mod.ROLE_QUALITY)
        assert quality is not None
        assert quality.model == "best", (
            f"quality picked {quality.model!r}, expected the composite argmax"
        )
    finally:
        restore()


def test_rc3_code_role_is_code_argmax():
    mod, restore = _load()
    try:
        recommender, _ = _make_recommender(mod)
        snapshot = recommender.generate_from_scores([
            _score("fast-weak-code", composite=0.6, speed=0.9, code=0.2),
            _score("strong-code", composite=0.6, speed=0.1, code=0.9),
        ])
        code = snapshot.get_recommendation(mod.ROLE_CODE)
        assert code is not None
        assert code.model == "strong-code", (
            f"code role picked {code.model!r}, expected the code argmax"
        )
    finally:
        restore()


def test_rc4_fast_role_respects_quality_floor():
    mod, restore = _load()
    try:
        recommender, _ = _make_recommender(mod)
        snapshot = recommender.generate_from_scores([
            _score("solid", composite=1.0, speed=0.2),
            _score("fast-junk", composite=0.2, speed=1.0),
        ])
        fast = snapshot.get_recommendation(mod.ROLE_FAST)
        assert fast is not None
        assert fast.model == "solid", (
            f"fast role picked {fast.model!r}; a model below the quality"
            " floor must be excluded even with the top speed"
        )
    finally:
        restore()


def test_rc5_value_role_never_rewards_zero_quality():
    mod, restore = _load()
    try:
        recommender, _ = _make_recommender(mod)
        snapshot = recommender.generate_from_scores([
            _score("balanced", composite=0.5, speed=0.5),
            _score("zero-quality", composite=0.0, speed=1.0),
        ])
        value = snapshot.get_recommendation(mod.ROLE_VALUE)
        assert value is not None
        assert value.model == "balanced", (
            f"value role picked {value.model!r}; zero quality must not win"
            " on speed alone"
        )
    finally:
        restore()


def test_rc6_apply_without_snapshot_is_refused():
    mod, restore = _load()
    try:
        recommender, _ = _make_recommender(mod)
        result = recommender.apply_to_smart_router()
        assert result.get("applied") is False, (
            f"apply on empty state returned {result!r}, expected a refusal"
        )
    finally:
        restore()


def test_rc7_apply_pushes_quality_model_to_router():
    mod, restore = _load()
    try:
        recommender, store = _make_recommender(mod)
        snapshot = recommender.generate_from_scores([
            _score("second-best", composite=0.4, speed=0.9),
            _score("best", composite=0.9, speed=0.1),
        ])

        quality = snapshot.get_recommendation(mod.ROLE_QUALITY)
        assert quality is not None

        router = _RouterRecorder()
        fake = types.ModuleType("opti_oignon.smart_router")
        fake.smart_router = router
        sys.modules["opti_oignon.smart_router"] = fake

        result = recommender.apply_to_smart_router(snapshot=snapshot)
        assert result.get("applied") is True
        assert result["changes"].get("default_model") == quality.model
        assert router.configured == [{"default_model": quality.model}], (
            f"router received {router.configured!r}, expected exactly the"
            " snapshot's quality model as default"
        )

        reloaded = store.get_by_id(snapshot.snapshot_id)
        assert reloaded is not None and reloaded.applied is True, (
            "the applied flag must be persisted in the store"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("RC1 empty scores yield empty snapshot",
         test_rc1_empty_scores_yield_empty_snapshot),
        ("RC2 quality role is composite argmax",
         test_rc2_quality_role_is_composite_argmax),
        ("RC3 code role is code argmax",
         test_rc3_code_role_is_code_argmax),
        ("RC4 fast role respects quality floor",
         test_rc4_fast_role_respects_quality_floor),
        ("RC5 value role never rewards zero quality",
         test_rc5_value_role_never_rewards_zero_quality),
        ("RC6 apply without snapshot is refused",
         test_rc6_apply_without_snapshot_is_refused),
        ("RC7 apply pushes quality model to router",
         test_rc7_apply_pushes_quality_model_to_router),
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
