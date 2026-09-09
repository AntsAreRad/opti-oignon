#!/usr/bin/env python3
"""Recipes for the measurements this estate anchors its releases on.

An anchor is a number, and a number moves every time the tree moves. What
must not move is the route taken to arrive at it: two readers who compute
the same anchor two different ways have measured two different things and
will disagree about which tree they are holding, with nothing in either
result to say so. The four routes below are written down once, here, so
that every later reading takes the same one.

Three of the four share a single correction. A census that walks the disk
reports whatever happens to be lying in the tree -- installed
dependencies, caches written by the run in progress, private working
documents the repository explicitly refuses -- and calls the total a
property of the product. It is not: it is a property of the machine. So
the perimeter every census runs over is the checksum manifest that ships
beside the release, never a walk of the filesystem. A path the manifest
does not list is not part of what shipped, and cannot be charged to it.

The fourth, the suite fingerprint, has the same shape of defect available
to it and is closed the same way: it reads the machine-readable report
the run wrote, never the terminal summary a human happened to see.

Usage: ``surface_recipes.py MANIFEST [JUNIT]``
"""

import hashlib
import importlib
import json
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The guard that owns the nomenclature detector also owns the list of trees
# that ship. Restating that list here would create a second register: it
# would agree on the day it was written and disagree silently ever after,
# which is the defect the release checks were written against. It is read
# from the guard instead.
_GUARD = ROOT / ".github" / "scripts" / "public_clean_guard.py"


def load_clean_guard(guard_path=None):
    """Import the shipped guard as a plain module.

    The guard is a standalone program rather than a project module, so
    putting its directory on the path for the duration of the import is
    enough; the entry is removed again so nothing else in the run sees
    it. Importing is safe -- the guard only acts when run as a program.
    """
    path = Path(guard_path or _GUARD)
    directory = str(path.parent)
    sys.path.insert(0, directory)
    try:
        return importlib.import_module(path.stem)
    finally:
        if directory in sys.path:
            sys.path.remove(directory)


def scanned_trees(guard=None):
    """The trees that ship, as the guard declares them."""
    return tuple((guard or load_clean_guard())._SCAN_PATHS)


# ---------------------------------------------------------------------------
# The perimeter: what shipped, as the manifest lists it
# ---------------------------------------------------------------------------
def read_perimeter(manifest_path):
    """Return the relative paths the checksum manifest lists.

    An empty perimeter is refused rather than returned. A census over an
    empty perimeter reports zero and reads exactly like a clean tree,
    which is the most expensive way for a mistyped path to stay quiet.
    """
    perimeter = set()
    text = Path(manifest_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"manifest line is not a checksum pair: {line!r}")
        perimeter.add(parts[1].strip().removeprefix("./"))
    if not perimeter:
        raise ValueError(f"manifest lists no path at all: {manifest_path}")
    return frozenset(perimeter)


def within(root, path, perimeter):
    """True when ``path`` is a file the manifest lists."""
    return str(Path(path).relative_to(root)) in perimeter


# ---------------------------------------------------------------------------
# Recipe 1 -- the suite fingerprint
# ---------------------------------------------------------------------------
def suite_fingerprint(junit_path):
    """Digest the roster of cases the run reported.

    The roster is every ``classname::name`` in the report, sorted, joined
    by single newlines, with no trailing newline. Reading the report
    rather than the terminal summary matters for the same reason the
    perimeter matters: the summary is what a reader saw, the report is
    what the run did.
    """
    return hashlib.md5(
        "\n".join(suite_roster(junit_path)).encode("utf-8")
    ).hexdigest()


def suite_roster(junit_path):
    """The sorted case identifiers the report carries."""
    root = ElementTree.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    ids = [
        f"{case.get('classname')}::{case.get('name')}"
        for suite in suites
        for case in suite.findall("testcase")
    ]
    if not ids:
        raise ValueError(f"the report names no case at all: {junit_path}")
    return sorted(ids)


# ---------------------------------------------------------------------------
# Recipe 2 -- the published interface digest
# ---------------------------------------------------------------------------
def openapi_digest(spec):
    """Digest the published interface.

    Keys are sorted so that a framework free to emit them in any order
    cannot move the anchor without the interface moving, and non-ASCII
    characters are left as themselves so that prose in the published
    descriptions is digested as written rather than as escape sequences.
    """
    return hashlib.md5(openapi_text(spec).encode("utf-8")).hexdigest()


def openapi_text(spec):
    """The canonical serialisation the digest is taken over."""
    return json.dumps(spec, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Recipe 3 -- the non-ASCII census
# ---------------------------------------------------------------------------
def non_ascii_census(root, perimeter, trees=None):
    """Count characters above ASCII in the trees that ship.

    Returns ``(characters, files)``. Only paths the manifest lists are
    charged: a runtime file written beside the product, or a dependency
    installed into it, is not part of what shipped.
    """
    root = Path(root)
    total = 0
    charged = set()
    for path in _candidates(root, perimeter, trees):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        count = sum(1 for character in text if ord(character) > 127)
        if count:
            total += count
            charged.add(str(path.relative_to(root)))
    return total, len(charged)


# ---------------------------------------------------------------------------
# Recipe 4 -- the standing nomenclature census
# ---------------------------------------------------------------------------
def nomenclature_census(root, perimeter, detector=None, trees=None):
    """Count standing internal-nomenclature spans in the trees that ship.

    Returns ``(spans, files)``. The guard itself charges added lines on a
    diff and leaves the standing text alone; this is the same detector
    run over whole files, so the debt can be reported as a figure that
    falls. The perimeter is what keeps that figure a property of the
    product: an installed dependency tree sits under the scanned trees
    and would otherwise be charged to the release.
    """
    root = Path(root)
    guard = None
    if detector is None:
        guard = load_clean_guard()
        detector = guard.find_violations
    spans = 0
    charged = set()
    for path in _candidates(root, perimeter, trees or scanned_trees(guard)):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        found = detector(lines)
        if found:
            spans += len(found)
            charged.add(str(path.relative_to(root)))
    return spans, len(charged)


def _candidates(root, perimeter, trees=None):
    """Files under the scanned trees that the manifest lists, sorted."""
    root = Path(root)
    for tree in trees if trees is not None else scanned_trees():
        for path in sorted((root / tree.rstrip("/")).rglob("*")):
            if not path.is_file():
                continue
            if within(root, path, perimeter):
                yield path


def _main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    perimeter = read_perimeter(argv[0])
    non_ascii = non_ascii_census(ROOT, perimeter)
    nomenclature = nomenclature_census(ROOT, perimeter)
    print(f"perimeter        {len(perimeter)} path(s) listed by the manifest")
    print(f"non-ascii        {non_ascii[0]} character(s) / {non_ascii[1]} file(s)")
    print(f"nomenclature     {nomenclature[0]} span(s) / {nomenclature[1]} file(s)")
    if len(argv) > 1:
        roster = suite_roster(argv[1])
        print(f"suite            {len(roster)} case(s)")
        print(f"suite digest     {suite_fingerprint(argv[1])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
