#!/usr/bin/env python3
"""Contracts for malformed-shape handling when writing back a pattern.

Applying an accepted suggestion navigates ``config -> rag -> sanitization
-> custom_patterns`` in rag.yaml. The list at the leaf was already guarded
against a non-list value, but the two container levels were not: a
hand-edited file whose ``rag`` or ``sanitization`` key holds a non-mapping
value raised a raw ``AttributeError`` mid-navigation. The container levels
now degrade the same way the list does -- an unexpected non-mapping is
replaced with an empty mapping so the write proceeds cleanly -- while a
well-formed file is written exactly as before.

  * Contract 1 -- a non-mapping ``sanitization`` degrades cleanly: the
    accepted pattern is still written and the call returns True.
  * Contract 2 -- a non-mapping ``rag`` degrades cleanly: the accepted
    pattern is still written and the call returns True.
  * Contract 3 -- a well-formed file preserves its other ``rag`` keys and
    appends the new pattern (the normalization does not clobber a valid
    mapping).

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. The module is loaded in isolation under a stub
package and every write targets a temporary file, never the real
rag.yaml.
"""

import importlib.util
import sys
import tempfile
import traceback
import types
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"

_MOD_NAME = "opti_oignon.redteam.feedback"


def _load_feedback():
    """Load the feedback module in isolation under a stub package."""
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
        _MOD_NAME, _OO / "redteam" / "feedback.py",
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


def _write_config(text):
    fd = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    fd.write(text)
    fd.close()
    return Path(fd.name)


def _accepted(feedback, name="rt_candidate"):
    return feedback.Suggestion(
        id="sg-x",
        pattern_name=name,
        regex=r"ignore\s+previous",
        source_category="prompt_injection",
        source_strategy="raw",
        status="accepted",
    )


def test_c1_non_mapping_sanitization_degrades_cleanly():
    feedback, saved = _load_feedback()
    # rag is a mapping, but sanitization is a scalar string.
    path = _write_config("rag:\n  sanitization: just_a_string\n")
    try:
        s = _accepted(feedback)
        applied = feedback.apply_suggestion_to_config(s, path)
        assert applied is True, (
            "a non-mapping sanitization must degrade cleanly, not crash"
        )
        assert s.pattern_name in path.read_text(), (
            "the accepted pattern must still reach the config file"
        )
    finally:
        path.unlink(missing_ok=True)
        _restore(saved)


def test_c2_non_mapping_rag_degrades_cleanly():
    feedback, saved = _load_feedback()
    # The rag key itself is a scalar string.
    path = _write_config("rag: just_a_string\n")
    try:
        s = _accepted(feedback)
        applied = feedback.apply_suggestion_to_config(s, path)
        assert applied is True, (
            "a non-mapping rag section must degrade cleanly, not crash"
        )
        assert s.pattern_name in path.read_text(), (
            "the accepted pattern must still reach the config file"
        )
    finally:
        path.unlink(missing_ok=True)
        _restore(saved)


def test_c3_wellformed_file_preserves_other_keys():
    feedback, saved = _load_feedback()
    path = _write_config(
        "rag:\n"
        "  chunk_size: 500\n"
        "  sanitization:\n"
        "    custom_patterns: []\n"
    )
    try:
        s = _accepted(feedback, name="rt_keep")
        applied = feedback.apply_suggestion_to_config(s, path)
        assert applied is True
        data = yaml.safe_load(path.read_text())
        assert data["rag"]["chunk_size"] == 500, (
            "a valid rag mapping must keep its other keys"
        )
        names = {
            p.get("name")
            for p in data["rag"]["sanitization"]["custom_patterns"]
        }
        assert "rt_keep" in names, "the new pattern must be appended"
    finally:
        path.unlink(missing_ok=True)
        _restore(saved)


_TESTS = [
    test_c1_non_mapping_sanitization_degrades_cleanly,
    test_c2_non_mapping_rag_degrades_cleanly,
    test_c3_wellformed_file_preserves_other_keys,
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
