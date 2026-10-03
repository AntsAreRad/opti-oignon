#!/usr/bin/env python3
"""Public-clean guard: reject internal session nomenclature in added lines.

No published tree may carry internal session nomenclature -- not the Python
ones alone, but the frontend, the operator scripts, the mobile tree and the
native crates as well, so that each of them is born clean rather than cleaned
later. This guard scans the ADDED lines of a diff over those trees and fails
when a line introduces:

  * a session code -- the letter S followed by two-to-four digits as a
    standalone token; or
  * an internal document reference -- one of a small set of uppercase
    prefixes immediately followed by a session code or the tracking marker
    (the bare prefixes are legitimate elsewhere, e.g. an uppercase constant,
    so only the document-reference form is a violation); or
  * an internal process word.

A short list of public product terms is exempt and can never account for a
violation; so are the rule codes of a lint pragma, which name rules of the
linter rather than anything internal.

Diff-only by design over those trees: it guards against NEW nomenclature on
added lines without failing on pre-existing debt, so it can be adopted
before that debt is paid down.

A second pass reads the whole tracked tree, root and hidden files included,
where that debt is already zero: no tracked file may carry a session code or
the name of the tool used to write the tree in its path, and no tracked line
may name that tool. A name ships as surely as a line does, and a file at the
root sits outside every scan tree of the diff pass. Outside the scan trees
-- the root, the documentation, the CI tree, any other directory -- the
debt is zero under the whole rule as well, so every line of every tracked
text file there is held to it, standing lines included; inside them the
standing debt is left to the diff pass until it is paid. A tree git cannot
read fails the guard: nothing read is never a pass.

The forbidden patterns are assembled from fragments at import
time, so this published script carries no clear instance of the
nomenclature it rejects and does not trip on a scan of itself.

The pure helper ``find_violations`` is import-safe and unit-tested;
``main`` runs both passes and exits non-zero on any violation.
Usage: ``public_clean_guard.py [BASE_REF]`` (default base ref: origin/main).
"""

import re
import subprocess
import sys

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

# Public product terms exempt from every pattern, assembled from fragments.
_ALLOWED_TERMS = ("network" + "_outbound", "prompt" + "_injection")

# A lint pragma's rule codes are exempt: the ``noqa:`` marker followed by
# letter-digit codes names rules of the linter, not internal history, and
# charging them would force authors to strip real suppressions to read
# clean. Only the pragma form is stripped; the bare token elsewhere stays
# charged.
_LINT_PRAGMA = re.compile(
    r"\bnoqa\s*:\s*[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*"
)

# Session code: the letter S (either case) then two-to-four digits. The
# token may not be entered from a letter or digit, and the digits may not
# run on into more digits or a lowercase letter. A capitalized continuation
# stays charged -- the camel-case identifier shape is a disguise, not a
# boundary -- while a platform name like a hardware architecture triple,
# whose digits run straight into a lowercase letter, is never charged.
_SESSION_CODE = re.compile(r"(?<![A-Za-z0-9])[sS][0-9]{2,4}(?![0-9a-z])")

# Internal document reference: a known uppercase prefix immediately followed
# by a session code or the tracking marker. Prefixes and marker are built
# from fragments; the bare words appear legitimately elsewhere, so only the
# document-reference form is matched.
_DOC_PREFIXES = ("PROMP" + "T", "ROADMA" + "P", "SESSIO" + "N")
_DOC_TAIL = "(?:S[0-9]|" + "TRACK" + "ING)"
_DOC_REFERENCE = re.compile(
    r"\b(?:" + "|".join(_DOC_PREFIXES) + r")_" + _DOC_TAIL
)

# Internal process words (case-insensitive), assembled from fragments.
_PROCESS_WORDS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        "back" + "fill",
        "mutation" + "-proven",
        "read" + "[ -]" + "gate",
        "contract" + "-id",
    )
)

# Trees the diff pass scans. Outside these, the tree-wide pass reads every
# line instead.
#
# Every tree that ships is here, not the Python ones alone. The detector is a
# regex over added lines and knows nothing about syntax, so a tree of
# TypeScript, shell, Kotlin or Rust is guarded on exactly the same terms as a
# tree of Python: leaving a shipped tree out would be a choice, never a
# technical limit. Diff-only, so the standing debt in these trees is not
# charged -- what is charged is any new instance arriving on an added line.
# The comment-only guard covers these same trees, and the release recipes
# read their perimeter from this list.
_SCAN_PATHS = (
    "opti_oignon/", "tests/", "frontend/", "scripts/", "android/", "rust/",
)

_DEFAULT_BASE_REF = "origin/main"

# The tool used to write the tree (case-insensitive, from fragments). It is
# not part of the work: no tracked path or line may name it.
_TOOL_NAME = re.compile("cla" + "ude", re.IGNORECASE)


def _strip_allowed(line):
    """Remove exempt terms and pragmas so they cannot account for a match."""
    for term in _ALLOWED_TERMS:
        line = line.replace(term, " ")
    return _LINT_PRAGMA.sub(" ", line)


def find_violations(lines):
    """Return ``[(index, kind, snippet), ...]`` for offending lines.

    ``lines`` is any iterable of strings -- added lines, without the diff
    ``+`` marker. ``index`` is the position within ``lines``. Exempt product
    terms are removed before matching. At most one violation is reported per
    line; the most specific kind wins, so a document reference outranks the
    session code it necessarily contains.
    """
    violations = []
    for index, raw in enumerate(lines):
        line = _strip_allowed(raw)
        if _DOC_REFERENCE.search(line):
            violations.append((index, "doc_reference", raw.strip()))
            continue
        if _SESSION_CODE.search(line):
            violations.append((index, "session_code", raw.strip()))
            continue
        for pattern in _PROCESS_WORDS:
            if pattern.search(line):
                violations.append((index, "process_word", raw.strip()))
                break
    return violations


def find_name_violations(paths):
    """Return ``[(path, kind), ...]`` for tracked paths that must not ship.

    A path is charged when any of its components carries a session code
    (``session_code_in_name``) or names the tool (``tool_in_name``).
    """
    violations = []
    for path in paths:
        if _TOOL_NAME.search(path):
            violations.append((path, "tool_in_name"))
        elif _SESSION_CODE.search(path):
            violations.append((path, "session_code_in_name"))
    return violations


def find_tool_mentions(lines):
    """Return ``[(index, snippet), ...]`` for lines that name the tool."""
    return [
        (index, raw.strip())
        for index, raw in enumerate(lines)
        if _TOOL_NAME.search(raw)
    ]


def _grep_pairs(stdout):
    """Split the output of ``git grep -z`` into ``[(path, line), ...]``."""
    pairs = []
    for line in stdout.decode("utf-8", "replace").split("\n"):
        path, sep, text = line.partition("\0")
        if sep:
            pairs.append((path, text))
    return pairs


def _tracked_tree():
    """Return ``(paths, [(path, line), ...])`` for the whole tracked tree.

    Binary files are skipped by content; their names are still read.
    Returns ``None`` when git fails either read: half a tree read is not a
    tree read.
    """
    # Fixed argv, no shell: safe.
    listed = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, check=False,
    )
    if listed.returncode != 0:
        return None
    paths = [p for p in listed.stdout.decode("utf-8").split("\0") if p]
    grep = subprocess.run(
        ["git", "grep", "-I", "-i", "-z", "--no-color", "-e", "cla" + "ude"],
        capture_output=True, check=False,
    )
    # git grep exits 1 when no line matches; anything else is an error.
    if grep.returncode not in (0, 1):
        return None
    return paths, _grep_pairs(grep.stdout)


def _lines_outside_scan_trees():
    """Return ``[(path, line), ...]`` for every tracked line outside the scan trees.

    Every line of every tracked text file, binary files skipped by content.
    Returns ``None`` when git cannot read the tree, so that a failed read
    fails the guard instead of passing it on nothing read.
    """
    excluded = [":(exclude)" + path for path in _SCAN_PATHS]
    # Fixed argv, no shell: safe. The empty pattern matches every line.
    grep = subprocess.run(
        ["git", "grep", "-I", "-z", "--no-color", "-e", "", "--", ".", *excluded],
        capture_output=True, check=False,
    )
    # git grep exits 1 when it read no line at all; anything else is an error.
    if grep.returncode not in (0, 1):
        return None
    return _grep_pairs(grep.stdout)


def _added_lines_with_paths(base_ref):
    """Return ``[(path, added_line), ...]`` for the diff over the scan trees.

    Uses a zero-context unified diff so only genuinely added content is
    considered. The path is the post-image file each added line belongs to.

    The diff is read as bytes and cut at newlines alone. Text mode and
    ``splitlines`` also break a line at a lone CR, a vertical tab or a form
    feed, and the rest of that added line came back without its ``+`` and
    was never read: a code placed after one of them passed. And a line is a
    file header only between a ``diff --git`` line and the first hunk: inside
    a hunk, an added line whose own text opens with ``++ `` reads ``+++ ``
    and was taken for a header, its code never read.

    Returns ``None`` when git cannot produce the diff, a base it cannot
    resolve first of all: an empty diff and a failed one are not the same
    answer, and only the first is a pass.
    """
    cmd = [
        "git", "diff", "--unified=0", "--no-color", base_ref,
        "--", *_SCAN_PATHS,
    ]
    # Fixed argv, no shell: safe.
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        return None
    pairs = []
    current_path = None
    in_hunk = False
    for line in result.stdout.decode("utf-8").split("\n"):
        if line.startswith("diff --git "):
            in_hunk = False
            continue
        if not in_hunk and line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current_path = None if path == "/dev/null" else path
            continue
        if line.startswith("@@"):
            in_hunk = True
            continue
        if in_hunk and line.startswith("+"):
            pairs.append((current_path, line[1:]))
    return pairs


def main(argv=None):
    """Run the diff pass and the tree-wide pass; exit non-zero on any violation."""
    argv = list(sys.argv[1:] if argv is None else argv)
    base_ref = argv[0] if argv else _DEFAULT_BASE_REF

    pairs = _added_lines_with_paths(base_ref)
    if pairs is None:
        print(f"public-clean guard: FAILED -- git could not diff against base {base_ref}")
        return 1
    lines = [added for _path, added in pairs]
    violations = find_violations(lines)

    tracked = _tracked_tree()
    outside = _lines_outside_scan_trees()
    if tracked is None or outside is None:
        print("public-clean guard: FAILED -- git could not read the tracked tree")
        return 1
    paths, tree_lines = tracked
    names = find_name_violations(paths)
    mentions = find_tool_mentions([text for _path, text in tree_lines])
    standing = find_violations([text for _path, text in outside])

    if not violations and not standing and not names and not mentions:
        print(
            f"public-clean guard: no session nomenclature in {len(pairs)} "
            f"added line(s) (base {base_ref}) nor in {len(outside)} lines "
            f"outside the scan trees, none in {len(paths)} tracked names, "
            "no tool mention"
        )
        return 0

    print("public-clean guard: FAILED")
    for index, kind, snippet in violations:
        path = pairs[index][0] or "?"
        print(f"  {path} [{kind}]: {snippet}")
    for index, kind, snippet in standing:
        print(f"  {outside[index][0]} [{kind}]: {snippet}")
    for path, kind in names:
        print(f"  {path} [{kind}]")
    for index, snippet in mentions:
        print(f"  {tree_lines[index][0]} [tool_mention]: {snippet[:120]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
