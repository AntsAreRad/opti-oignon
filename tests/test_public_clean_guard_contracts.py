#!/usr/bin/env python3
"""Contracts for the public-clean CI guard.

The guard scans the added lines of a diff over the published trees and fails
when a line introduces internal session nomenclature. These contracts pin
the pure detection helper (find_violations) so its behaviour is verified
independently of git:

  * G1 -- a session code (the letter S then two-to-four digits, standalone)
    on an added line is flagged.
  * G2 -- an internal document reference (a known prefix followed by a
    session code or the tracking marker) is flagged.
  * G3 -- legitimate lines are NOT flagged: an uppercase constant that
    merely starts with one of the prefixes, ordinary lowercase identifiers,
    and the exempt public product terms.
  * G4 -- an internal process word is flagged.
  * G5 -- the scan perimeter covers every published tree that ships, not
    the Python trees alone. The detector is a regex over added lines and is
    agnostic to language, so a tree of TypeScript, shell or Kotlin is
    guarded on exactly the same terms as a tree of Python.

Every input that must be flagged is assembled from fragments at runtime, so
the literal nomenclature never appears in this file's source and the guard
does not trip on a scan of the test itself.

Local-only. Runs under pytest or via the __main__ runner. The guard script
lives under .github/, outside the importable package, and is loaded through
the shared isolation window.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = "_public_clean_guard_under_contract"
_GUARD_PATH = REPO / ".github" / "scripts" / "public_clean_guard.py"

# Fragments assembled at runtime; no clear nomenclature in the source text.
_S = "S"
_DIGITS = "312"
_UNDERSCORE = "_"


def _load():
    """Load the guard through the shared window; returns (module, restore)."""
    loaded, restore = isolate(targets={_GUARD: _GUARD_PATH})
    return loaded[_GUARD], restore


def _kinds(violations):
    return {kind for _idx, kind, _snippet in violations}


# ---------------------------------------------------------------------------
# G1 -- a standalone session code is flagged
# ---------------------------------------------------------------------------
def test_g1_session_code_is_flagged():
    guard, restore = _load()
    try:
        code = _S + _DIGITS  # assembled session code
        line = "    # delivered in " + code + " with contracts"
        violations = guard.find_violations([line])
        assert violations, "a standalone session code on an added line must flag"
        assert "session_code" in _kinds(violations)
    finally:
        restore()


# ---------------------------------------------------------------------------
# G2 -- an internal document reference is flagged
# ---------------------------------------------------------------------------
def test_g2_document_reference_is_flagged():
    guard, restore = _load()
    try:
        prefix = "PROMP" + "T"  # assembled prefix
        ref = prefix + _UNDERSCORE + _S + "313"  # doc reference form
        line = 'DOC = "' + ref + '.md"'
        violations = guard.find_violations([line])
        assert violations, "an internal document reference must flag"
        assert "doc_reference" in _kinds(violations)
    finally:
        restore()


# ---------------------------------------------------------------------------
# G3 -- legitimate lines are not flagged (incl. exempt product terms)
# ---------------------------------------------------------------------------
def test_g3_legitimate_lines_are_not_flagged():
    guard, restore = _load()
    try:
        # An uppercase constant that starts with a prefix but is NOT a doc ref.
        const_line = "SESSION" + _UNDERSCORE + "STATE_TOOLS = frozenset()"
        lowercase_line = "prompt_block = build_block(tools)"
        # Exempt public product terms must never account for a violation.
        exempt_line = 'permissions = ["' + "network" + _UNDERSCORE + 'outbound"]'
        injection_line = 'label = "' + "prompt" + _UNDERSCORE + 'injection"'
        for line in (const_line, lowercase_line, exempt_line, injection_line):
            assert guard.find_violations([line]) == [], (
                f"legitimate line must not be flagged: {line!r}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# G4 -- an internal process word is flagged
# ---------------------------------------------------------------------------
def test_g4_process_word_is_flagged():
    guard, restore = _load()
    try:
        word = "back" + "fill"  # assembled process word
        line = "    # " + word + " the legacy rows"
        violations = guard.find_violations([line])
        assert violations, "an internal process word must flag"
        assert "process_word" in _kinds(violations)
    finally:
        restore()


# ---------------------------------------------------------------------------
# G5 -- the perimeter covers every published tree, not the Python ones only
# ---------------------------------------------------------------------------
def test_g5_perimeter_covers_every_published_tree():
    guard, restore = _load()
    try:
        paths = set(guard._SCAN_PATHS)
        for tree in ("opti_oignon/", "tests/", "frontend/", "scripts/",
                     "android/"):
            assert tree in paths, (
                f"the published tree {tree!r} must be inside the perimeter; "
                "the detector is language-agnostic, so leaving a shipped tree "
                "out is a choice, never a limitation"
            )
        # Every entry must name a tree that exists: a perimeter that lists a
        # path the repository does not carry guards nothing and reads clean.
        for tree in guard._SCAN_PATHS:
            assert (REPO / tree.rstrip("/")).is_dir(), (
                f"perimeter entry {tree!r} names no directory in the tree"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("G1 session code flagged", test_g1_session_code_is_flagged),
        ("G2 document reference flagged", test_g2_document_reference_is_flagged),
        ("G3 legitimate lines pass", test_g3_legitimate_lines_are_not_flagged),
        ("G4 process word flagged", test_g4_process_word_is_flagged),
        ("G5 perimeter covers published trees",
         test_g5_perimeter_covers_every_published_tree),
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
