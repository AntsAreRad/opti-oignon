#!/usr/bin/env python3
"""Contracts for the auto-tuner's sweep bound, best-of, and load robustness.

The tuner benchmarks inference parameters and keeps the fastest per model. It
uses a fixed benchmark prompt and persists results to a local file. The
safety-relevant properties are that the search stays a bounded per-axis sweep
rather than an exhaustive product, that the recorded best can never be slower
than the measured baseline, and that loading a malformed results file does not
crash. These contracts pin those guards without pinning the search grid, the
hill-climb thresholds, or the recommendation heuristics.

  * AT1 -- the sweep is a deduplicated per-axis search: it contains no
    duplicate parameter combinations and stays within the sum of the axis
    lengths, which is strictly smaller than the full grid. The search cannot
    silently balloon into an exhaustive product.
  * AT2 -- best-of never regresses: the recorded best generation speed is at
    least the measured baseline, so tuning can only keep an equal-or-faster
    configuration.
  * AT3 -- loading tolerates junk: rehydrating a profile from a malformed
    entry does not raise, so a corrupt results file degrades instead of
    breaking the tuner.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package and driven with a deterministic in-process
benchmark, so no inference backend and no sibling module are required.
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
    """Load the real auto-tuner under a stand-in package.

    Returns (module, restore). The benchmark function is injected per test, so
    no inference backend is touched and no results file is written by the
    engine under test.
    """
    keys = ("opti_oignon", "opti_oignon.auto_tuner")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.auto_tuner", _OO / "auto_tuner.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.auto_tuner"] = mod
    spec.loader.exec_module(mod)
    pkg.auto_tuner = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


def _deterministic_benchmark(mod):
    """A no-variance benchmark: speed rises with batch size, threads, flash.

    Deterministic so the sweep's fastest configuration and the baseline are
    stable regardless of the host. Higher batch size and flash attention
    always help, so a faster-than-baseline configuration always exists.
    """
    def _bench(params):
        speed = 30.0
        speed += min(params.get("batch_size", 1024) / 1024.0, 4.0) * 2.0
        speed += params.get("threads", 4) * 0.1
        if params.get("flash_attention", True):
            speed += 3.0
        return mod.BenchmarkResult(
            params=params,
            tokens_per_second_tg=speed,
            tokens_per_second_pp=speed * 1.5,
            total_time_ms=1.0,
        )

    return _bench


# ---------------------------------------------------------------------------
# AT1 -- the sweep is a deduplicated, bounded per-axis search
# ---------------------------------------------------------------------------
def test_at1_sweep_is_bounded_and_deduplicated():
    mod, restore = _load()
    try:
        space = mod.ParameterSpace()
        tuner = mod.AutoTuner(
            config=mod.TunerConfig(warmup_runs=0, trials_per_param=1),
            param_space=space,
            benchmark_fn=_deterministic_benchmark(mod),
        )
        sweep = tuner._build_smart_sweep()
        keys = [mod._param_key(p) for p in sweep]
        axis_sum = (
            len(space.batch_size) + len(space.ubatch_size)
            + len(space.threads) + len(space.flash_attention)
        )
        cartesian = space.total_combinations()
        assert len(keys) == len(set(keys)), (
            "the sweep carries no duplicate parameter combinations"
        )
        assert len(sweep) <= axis_sum, (
            "the sweep stays within the sum of the axis lengths"
        )
        assert axis_sum < cartesian, (
            "a per-axis sweep is strictly smaller than the full grid"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AT2 -- best-of never regresses below the baseline
# ---------------------------------------------------------------------------
def test_at2_best_of_never_regresses():
    mod, restore = _load()
    try:
        tuner = mod.AutoTuner(
            config=mod.TunerConfig(warmup_runs=0, trials_per_param=1),
            param_space=mod.ParameterSpace(),
            benchmark_fn=_deterministic_benchmark(mod),
        )
        profile = tuner.run("mock", mod.TunerJob())
        assert profile.best_tg_speed >= profile.baseline_tg_speed, (
            "the recorded best is at least as fast as the measured baseline"
        )
        assert profile.speedup_factor >= 1.0, (
            "tuning can only keep an equal-or-faster configuration"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# AT3 -- loading a malformed profile does not crash
# ---------------------------------------------------------------------------
def test_at3_profile_load_tolerates_junk():
    mod, restore = _load()
    try:
        # A malformed entry: unknown keys, and a non-mapping where a mapping
        # would be expected. Rehydration must not raise.
        profile = mod.TunerProfile.from_dict({
            "unknown_field": 1,
            "best_params": "not-a-dict",
        })
        assert profile is not None, "a malformed profile still rehydrates"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("AT1 sweep bounded and deduplicated", test_at1_sweep_is_bounded_and_deduplicated),
        ("AT2 best-of never regresses", test_at2_best_of_never_regresses),
        ("AT3 profile load tolerates junk", test_at3_profile_load_tolerates_junk),
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
