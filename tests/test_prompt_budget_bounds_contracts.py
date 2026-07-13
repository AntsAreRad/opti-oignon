#!/usr/bin/env python3
"""Contracts for the prompt budget floors and template interpolation.

The budget manager splits a model's context window across zones, and the
template engine substitutes variables into task-specific system prompts. The
safety-relevant properties are that each zone keeps a minimum floor regardless
of the configured ratios, and that variable substitution is a single pass that
neither raises on unknown placeholders nor re-expands a hostile substituted
value. These contracts pin those guards without pinning the allocation ratios,
fallback window sizes, or template wording.

  * PO1 -- minimum zone budgets are honored even when every ratio is zero: no
    zone collapses below its configured floor.
  * PO2 -- interpolation is a single pass that leaves unknown placeholders as
    written and does not re-expand a value that itself looks like a
    placeholder, so a hostile context value cannot smuggle in another
    substitution.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package and driven with in-memory config, so no
inference backend and no sibling module are required.
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
    """Load the real prompt optimization module under a stand-in package.

    Returns (module, restore). Config is passed in memory per test, so no
    on-disk config is read and the model-window lookup is never reached.
    """
    keys = ("opti_oignon", "opti_oignon.prompt_optimization")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.prompt_optimization", _OO / "prompt_optimization.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.prompt_optimization"] = mod
    spec.loader.exec_module(mod)
    pkg.prompt_optimization = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


# ---------------------------------------------------------------------------
# PO1 -- minimum zone budgets survive zero ratios
# ---------------------------------------------------------------------------
def test_po1_minimum_budgets_survive_zero_ratios():
    mod, restore = _load()
    try:
        manager = mod.PromptTokenBudgetManager(config={
            "allocation": {
                "system_ratio": 0.0,
                "project_ratio": 0.0,
                "history_ratio": 0.0,
                "user_ratio": 0.0,
                "reserve_ratio": 0.0,
            },
            "minimum_budgets": {
                "system": 256,
                "project": 0,
                "history": 512,
                "user": 256,
                "reserve": 512,
            },
            "default_context_window": 100000,
        })
        budget = manager.calculate_budget(model="m", context_window_override=100000)
        assert budget.system_tokens >= 256, "the system floor is honored"
        assert budget.history_tokens >= 512, "the history floor is honored"
        assert budget.user_tokens >= 256, "the user floor is honored"
        assert budget.reserve_tokens >= 512, "the reserve floor is honored"
    finally:
        restore()


# ---------------------------------------------------------------------------
# PO2 -- interpolation is single-pass and does not re-expand a hostile value
# ---------------------------------------------------------------------------
def test_po2_interpolation_is_single_pass_and_unknown_safe():
    mod, restore = _load()
    try:
        engine = mod.PromptTemplateEngine(config={
            "language_rule": "LANG",
            "templates": {},
        })
        template = mod.PromptTemplate(
            task_type="t",
            system_prompt="A {language_rule} B {unknown} C {inject}",
        )
        # The injected value itself looks like a placeholder. A single pass in
        # a fixed order must not re-expand it.
        out = engine.interpolate(template, context={"inject": "{language_rule}"})
        assert "LANG" in out, "the built-in language rule is substituted once"
        assert "{unknown}" in out, "an unknown placeholder is left as written"
        assert out.count("LANG") == 1, (
            "the hostile value that looks like a placeholder is not re-expanded"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("PO1 minimum budgets survive zero ratios", test_po1_minimum_budgets_survive_zero_ratios),
        ("PO2 interpolation single-pass unknown-safe", test_po2_interpolation_is_single_pass_and_unknown_safe),
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
