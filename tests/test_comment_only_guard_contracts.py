#!/usr/bin/env python3
"""Contracts for the comment-only CI guard.

The guard proves that when internal nomenclature leaves a file, nothing that
executes leaves with it. These contracts pin the pure helpers so the
behaviour is verified independently of git:

  * C1 -- a file that sheds nomenclature from its comments alone is
    accepted: comments never reach the parser, so the shape cannot move.
  * C2 -- a file that sheds nomenclature AND edits a runtime string is
    refused. That string is a log line, a banner or a schema description: a
    surface no suite in this repository pins.
  * C3 -- a file that sheds nomenclature AND renames an identifier is
    refused.
  * C4 -- a file whose nomenclature count is unchanged is never examined,
    whatever else moved in it. Ordinary work must not be blocked by a guard
    aimed at one dull mechanical edit.
  * C5 -- a file that GAINS nomenclature is not this guard's business; the
    public-clean guard owns that direction.
  * C6 -- the docstring of a web route handler is NOT free. The framework
    publishes it verbatim as the endpoint description of the generated API
    schema, so editing one while shedding nomenclature moves the shape and
    is refused. An internal docstring in the very same file is free.
  * C7 -- outside Python the proof is comment stripping, quote-aware: a
    comment-only edit is accepted, and an edit inside a string literal is
    refused even though it looks like a comment marker.
  * C8 -- a file whose shape cannot be established is REFUSED, never waved
    through. A prover that stays silent on what it failed to understand
    proves nothing.

Every input carrying nomenclature is assembled from fragments at runtime, so
the literal form never appears in this file's source and neither guard trips
on a scan of the test itself.

Local-only. Runs under pytest or via the __main__ runner. The guard script
lives under .github/, outside the importable package, and is loaded through
the shared isolation window.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = "_comment_only_guard_under_contract"
_GUARD_PATH = REPO / ".github" / "scripts" / "comment_only_guard.py"

# Fragments assembled at runtime; no clear nomenclature in the source text.
_S = "S"
_CODE = _S + "347"
_OTHER = _S + "348"


def _load():
    """Load the guard through the shared window; returns (module, restore)."""
    loaded, restore = isolate(targets={_GUARD: _GUARD_PATH})
    return loaded[_GUARD], restore


# A small Python file carrying nomenclature in a comment, in an internal
# docstring, in a runtime string and in a published route docstring.
def _python_before():
    return (
        '"""Module note ' + _CODE + '."""\n'
        'import os\n'
        '\n'
        'router = os\n'
        'LABEL = "cache ' + _CODE + ' hit"\n'
        '\n'
        '\n'
        '@router.get("/x")\n'
        'def endpoint(a):\n'
        '    """Return the thing (' + _CODE + ')."""\n'
        '    # helper for ' + _CODE + ' routing\n'
        '    return os.path.join(a, "x")\n'
    )


def _typescript_before():
    return (
        '// helper for ' + _CODE + ' routing\n'
        'const url = "https://example.invalid/' + _CODE + '";\n'
        'export function g() { return url; }\n'
    )


# ---------------------------------------------------------------------------
# C1 -- shedding nomenclature from comments alone is accepted
# ---------------------------------------------------------------------------
def test_c1_comment_only_removal_is_accepted():
    guard, restore = _load()
    try:
        before = _python_before()
        after = before.replace("# helper for " + _CODE + " routing",
                               "# helper for routing")
        assert guard.debt_count(after) < guard.debt_count(before), (
            "the fixture must actually shed nomenclature"
        )
        assert guard.verdict("m.py", before, after) is None, (
            "a comment-only removal must be accepted"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C2 -- shedding nomenclature while editing a runtime string is refused
# ---------------------------------------------------------------------------
def test_c2_runtime_string_edit_is_refused():
    guard, restore = _load()
    try:
        before = _python_before()
        after = before.replace("# helper for " + _CODE + " routing",
                               "# helper for routing")
        after = after.replace('"cache ' + _CODE + ' hit"', '"cache hit"')
        assert guard.debt_count(after) < guard.debt_count(before)
        reason = guard.verdict("m.py", before, after)
        assert reason is not None, (
            "a runtime string is a shipped surface; it must not ride along"
        )
        assert "shape" in reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# C3 -- shedding nomenclature while renaming an identifier is refused
# ---------------------------------------------------------------------------
def test_c3_identifier_rename_is_refused():
    guard, restore = _load()
    try:
        before = _python_before()
        after = before.replace("# helper for " + _CODE + " routing",
                               "# helper for routing")
        after = after.replace("LABEL", "TAG")
        assert guard.verdict("m.py", before, after) is not None, (
            "a rename must not ride along with a comment purge"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C4 -- an unchanged count means the file is never examined
# ---------------------------------------------------------------------------
def test_c4_unchanged_count_is_not_examined():
    guard, restore = _load()
    try:
        before = _python_before()
        # A real code change that keeps every nomenclature line intact.
        after = before.replace('return os.path.join(a, "x")',
                               'return os.path.join(a, "y", "z")')
        assert guard.debt_count(after) == guard.debt_count(before)
        assert guard.verdict("m.py", before, after) is None, (
            "ordinary work must never be blocked by this guard"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C5 -- a file that gains nomenclature belongs to the other guard
# ---------------------------------------------------------------------------
def test_c5_added_nomenclature_is_not_this_guard():
    guard, restore = _load()
    try:
        before = _python_before()
        after = before.replace("import os",
                               "import os  # added in " + _OTHER)
        after = after.replace("LABEL", "TAG")
        assert guard.debt_count(after) > guard.debt_count(before)
        assert guard.verdict("m.py", before, after) is None, (
            "the rising direction is the public-clean guard's business"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C6 -- a published route description is shape, an internal docstring is not
# ---------------------------------------------------------------------------
def test_c6_published_route_description_is_not_free():
    guard, restore = _load()
    try:
        before = _python_before()
        # Internal module docstring: free.
        internal = before.replace('"""Module note ' + _CODE + '."""',
                                  '"""Module note."""')
        assert guard.debt_count(internal) < guard.debt_count(before)
        assert guard.verdict("m.py", before, internal) is None, (
            "an internal docstring is not a shipped surface"
        )
        # Route handler docstring: published in the API schema, so it counts.
        published = before.replace('"""Return the thing (' + _CODE + ')."""',
                                   '"""Return the thing."""')
        assert guard.debt_count(published) < guard.debt_count(before)
        assert guard.verdict("m.py", before, published) is not None, (
            "an endpoint description ships; it must be declared, not absorbed"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C7 -- outside Python, the proof is quote-aware comment stripping
# ---------------------------------------------------------------------------
def test_c7_non_python_proof_is_quote_aware():
    guard, restore = _load()
    try:
        before = _typescript_before()
        comment_only = before.replace("// helper for " + _CODE + " routing",
                                      "// helper for routing")
        assert guard.debt_count(comment_only) < guard.debt_count(before)
        assert guard.verdict("a.ts", before, comment_only) is None, (
            "a comment-only edit outside Python must be accepted"
        )
        # The same token inside a string literal is NOT a comment.
        in_string = before.replace("// helper for " + _CODE + " routing",
                                   "// helper for routing")
        in_string = in_string.replace("example.invalid/" + _CODE,
                                      "example.invalid/x")
        assert guard.verdict("a.ts", before, in_string) is not None, (
            "an edit inside a string literal must not pass as a comment"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C8 -- a shape that cannot be established is refused, never assumed
# ---------------------------------------------------------------------------
def test_c8_unestablished_shape_is_refused():
    guard, restore = _load()
    try:
        before = "# note " + _CODE + "\nvalue = (1,\n"  # unbalanced on purpose
        after = "# note\nvalue = (1,\n"
        assert guard.debt_count(after) < guard.debt_count(before)
        reason = guard.verdict("broken.py", before, after)
        assert reason is not None, (
            "a file the prover cannot parse must be refused, not waved on"
        )
        assert "could not be established" in reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# C9 -- a hash glued to the token before it does not open a comment
# ---------------------------------------------------------------------------
def test_c9_glued_hash_does_not_open_a_comment():
    guard, restore = _load()
    try:
        before = (
            "while [[ $# -gt 0 ]]; do  # argument loop " + _CODE + "\n"
            "  shift\n"
            "done\n"
        )
        # Nomenclature leaves the trailing comment AND the loop bound moves.
        after = (
            "while [[ $# -gt 1 ]]; do  # argument loop\n"
            "  shift\n"
            "done\n"
        )
        assert guard.debt_count(after) < guard.debt_count(before)
        assert guard.verdict("run.sh", before, after) is not None, (
            "a hash with no whitespace before it is not a comment opener; "
            "the code after it must stay visible to the prover"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# C10 -- the two honest hash positions still open a comment
# ---------------------------------------------------------------------------
def test_c10_line_start_and_spaced_hash_still_comment():
    guard, restore = _load()
    try:
        at_start = "# header " + _CODE + "\nvalue: 1\n"
        at_start_after = "# header\nvalue: 1\n"
        assert guard.debt_count(at_start_after) < guard.debt_count(at_start)
        assert guard.verdict("a.yaml", at_start, at_start_after) is None, (
            "a hash at the start of a line opens a comment"
        )

        spaced = "value: 1  # note " + _CODE + "\n"
        spaced_after = "value: 1  # note\n"
        assert guard.debt_count(spaced_after) < guard.debt_count(spaced)
        assert guard.verdict("a.yaml", spaced, spaced_after) is None, (
            "a hash preceded by whitespace opens a comment"
        )

        # And the value itself is still shape, not comment.
        moved = "value: 2  # note\n"
        assert guard.verdict("a.yaml", spaced, moved) is not None, (
            "an edit to the value must not pass as a comment edit"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("C1 comment-only removal accepted",
         test_c1_comment_only_removal_is_accepted),
        ("C2 runtime string edit refused",
         test_c2_runtime_string_edit_is_refused),
        ("C3 identifier rename refused", test_c3_identifier_rename_is_refused),
        ("C4 unchanged count not examined",
         test_c4_unchanged_count_is_not_examined),
        ("C5 added nomenclature not this guard",
         test_c5_added_nomenclature_is_not_this_guard),
        ("C6 published route description is shape",
         test_c6_published_route_description_is_not_free),
        ("C7 non-Python proof is quote-aware",
         test_c7_non_python_proof_is_quote_aware),
        ("C8 unestablished shape refused",
         test_c8_unestablished_shape_is_refused),
        ("C9 glued hash is not a comment opener",
         test_c9_glued_hash_does_not_open_a_comment),
        ("C10 honest hash positions still comment",
         test_c10_line_start_and_spaced_hash_still_comment),
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
