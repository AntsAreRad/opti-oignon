#!/usr/bin/env python3
"""Public-clean guard: reject internal session nomenclature in added lines.

No published tree may carry internal session nomenclature -- not the Python
ones alone, but the frontend, the operator scripts and the mobile tree as
well, so that each of them is born clean rather than cleaned later. This
guard scans the ADDED lines of a diff over those trees and fails when a line
introduces:

  * a session code -- the letter S followed by two-to-four digits as a
    standalone token; or
  * an internal document reference -- one of a small set of uppercase
    prefixes immediately followed by a session code or the tracking marker
    (the bare prefixes are legitimate elsewhere, e.g. an uppercase constant,
    so only the document-reference form is a violation); or
  * an internal process word.

A short list of public product terms is exempt and can never account for a
violation.

Diff-only by design: it guards against NEW nomenclature on added lines
without failing on pre-existing debt, so it can be adopted before that debt
is paid down. The forbidden patterns are assembled from fragments at import
time, so this published script carries no clear instance of the
nomenclature it rejects and does not trip on a scan of itself.

The pure helper ``find_violations`` is import-safe and unit-tested;
``main`` performs the git diff scan and exits non-zero on any violation.
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

# Session code: the letter S then two-to-four digits, as a standalone token.
_SESSION_CODE = re.compile(r"\bS[0-9]{2,4}\b")

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

# Trees the guard scans. Nothing outside these is considered.
#
# Every tree that ships is here, not the Python ones alone. The detector is a
# regex over added lines and knows nothing about syntax, so a tree of
# TypeScript, shell or Kotlin is guarded on exactly the same terms as a tree
# of Python: leaving a shipped tree out would be a choice, never a technical
# limit. Diff-only, so the standing debt in these trees is not charged --
# what is charged is any new instance arriving on an added line.
_SCAN_PATHS = (
    "opti_oignon/", "tests/", "frontend/", "scripts/", "android/",
)

_DEFAULT_BASE_REF = "origin/main"


def _strip_allowed(line):
    """Remove exempt product terms so they cannot account for a match."""
    for term in _ALLOWED_TERMS:
        line = line.replace(term, " ")
    return line


def find_violations(lines):
    """Return ``[(index, kind, snippet), ...]`` for offending lines.

    ``lines`` is any iterable of strings -- added lines, without the diff
    ``+`` marker. ``index`` is the position within ``lines``. Exempt product
    terms are removed before matching. At most one violation is reported per
    line; the earliest-listed kind wins.
    """
    violations = []
    for index, raw in enumerate(lines):
        line = _strip_allowed(raw)
        if _SESSION_CODE.search(line):
            violations.append((index, "session_code", raw.strip()))
            continue
        if _DOC_REFERENCE.search(line):
            violations.append((index, "doc_reference", raw.strip()))
            continue
        for pattern in _PROCESS_WORDS:
            if pattern.search(line):
                violations.append((index, "process_word", raw.strip()))
                break
    return violations


def _added_lines_with_paths(base_ref):
    """Return ``[(path, added_line), ...]`` for the diff over the scan trees.

    Uses a zero-context unified diff so only genuinely added content is
    considered. The path is the post-image file each added line belongs to.
    """
    cmd = [
        "git", "diff", "--unified=0", "--no-color", base_ref,
        "--", *_SCAN_PATHS,
    ]
    # Fixed argv, no shell: safe.
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
    )
    pairs = []
    current_path = None
    for line in result.stdout.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current_path = None if path == "/dev/null" else path
            continue
        if line.startswith("+") and not line.startswith("+++"):
            pairs.append((current_path, line[1:]))
    return pairs


def main(argv=None):
    """Scan the diff's added lines and exit non-zero on any violation."""
    argv = list(sys.argv[1:] if argv is None else argv)
    base_ref = argv[0] if argv else _DEFAULT_BASE_REF

    pairs = _added_lines_with_paths(base_ref)
    lines = [added for _path, added in pairs]
    violations = find_violations(lines)

    if not violations:
        print(
            "public-clean guard: no session nomenclature in added lines "
            f"(base {base_ref})"
        )
        return 0

    print(
        "public-clean guard: FAILED -- session nomenclature in added lines:"
    )
    for index, kind, snippet in violations:
        path = pairs[index][0] or "?"
        print(f"  {path} [{kind}]: {snippet}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
