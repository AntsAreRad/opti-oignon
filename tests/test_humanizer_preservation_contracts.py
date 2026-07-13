#!/usr/bin/env python3
"""Contracts for the humanizer's preservation and code-integrity guards.

The humanizer rewrites model output to read less mechanically, optionally via
a further model pass. Because it post-processes text that may contain code and
may run with the model backend unavailable, the safety-relevant properties are
that it never enlarges or corrupts its input beyond the configured bound, that
it always returns a defined result with the original intact, and that its
rule-based transforms never touch code. These contracts pin those guards
without pinning the learned word lists, rewrite wording, or intensity levels.

  * HZ1 -- input above the configured maximum length is returned unchanged:
    over-long text is never partially processed.
  * HZ2 -- with the model backend unavailable in rewrite mode, the engine
    falls back to the rule pass; the result is always a defined, non-empty
    string and the original is preserved on the result.
  * HZ3 -- rule-based transforms leave code segments byte-for-byte intact:
    fenced and inline code are masked before any transform and restored after.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module is
loaded under a stand-in package, the model backend gate is forced closed, and
the engine is built with an in-memory-style temporary feedback store so no
inference backend and no sibling module are required.
"""

import importlib.util
import sys
import tempfile
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load():
    """Load the real humanizer under a stand-in package.

    Returns (module, restore). The encrypted-connection helper import is
    guarded in the module and falls back to plain sqlite here; the model
    backend gate is forced closed per test.
    """
    keys = ("opti_oignon", "opti_oignon.humanizer")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.humanizer", _OO / "humanizer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.humanizer"] = mod
    spec.loader.exec_module(mod)
    pkg.humanizer = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


def _engine(mod, **overrides):
    """Build an engine with a deterministic config and a temporary store.

    The model backend gate is forced closed so the rule path and its
    fallbacks are what run. Built via __new__ to avoid touching on-disk
    config; the feedback store is redirected to a temporary directory.
    """
    mod.OLLAMA_AVAILABLE = False
    mod._ollama = None
    eng = mod.HumanizerEngine.__new__(mod.HumanizerEngine)
    cfg = mod.HumanizerConfig(
        enabled=True,
        mode="rewrite",
        intensity="moderate",
        formality="neutral",
        banned_phrases=["it is important to note that"],
        vocabulary_replacements={"utilize": "use"},
        contractions={"it is": "it's"},
        hedging_excess=["arguably"],
        max_input_length=8000,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    eng._config = cfg
    tmp = Path(tempfile.mkdtemp())
    eng._feedback_db = mod.HumanizerFeedbackDB(tmp / "feedback.db")
    return eng


# ---------------------------------------------------------------------------
# HZ1 -- over-long input is returned unchanged
# ---------------------------------------------------------------------------
def test_hz1_over_max_length_input_returned_unchanged():
    mod, restore = _load()
    try:
        eng = _engine(mod, max_input_length=100)
        # Over 100 chars and full of a token the rule pass would otherwise
        # rewrite, so an unchanged result can only mean the guard fired.
        text = "utilize " * 60
        result = eng.humanize(text)
        assert result.humanized == text, (
            "input longer than the configured maximum is returned unchanged"
        )
        assert result.original == text
    finally:
        restore()


# ---------------------------------------------------------------------------
# HZ2 -- model unavailable in rewrite mode falls back to the rule pass
# ---------------------------------------------------------------------------
def test_hz2_model_unavailable_falls_back_to_rules():
    mod, restore = _load()
    try:
        eng = _engine(mod)
        text = "You should utilize this approach. It is important to note that it works."
        result = eng.humanize(text, mode="rewrite")
        assert isinstance(result.humanized, str) and result.humanized, (
            "the result is always a defined, non-empty string, never None"
        )
        assert result.original == text, "the original is preserved on the result"
        assert "use" in result.humanized.lower(), (
            "with no model backend, the rule pass runs as the fallback"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# HZ3 -- rule transforms leave code segments byte-for-byte intact
# ---------------------------------------------------------------------------
def test_hz3_rule_transforms_preserve_code_segments():
    mod, restore = _load()
    try:
        text = (
            "Here is code: `x = utilize(y)` and a block:\n"
            "```\nutilize()\n```\n"
            "You should utilize this in prose."
        )
        masked, segments = mod._protect_code_segments(text)
        transformed, _ = mod._apply_vocabulary_replacements(
            masked, {"utilize": "use"}
        )
        restored = mod._restore_code_segments(transformed, segments)
        assert "`x = utilize(y)`" in restored, "inline code is untouched"
        assert "utilize()" in restored, "fenced code is untouched"
        assert "use this in prose" in restored, "prose outside code is rewritten"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("HZ1 over-length input unchanged", test_hz1_over_max_length_input_returned_unchanged),
        ("HZ2 model down falls back to rules", test_hz2_model_unavailable_falls_back_to_rules),
        ("HZ3 rule transforms preserve code", test_hz3_rule_transforms_preserve_code_segments),
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
