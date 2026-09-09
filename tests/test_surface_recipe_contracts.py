#!/usr/bin/env python3
"""The measurement recipes are pinned by method, never by value.

An anchor is a number and every number here moves when the tree moves, so
pinning a value would pin the tree rather than the measurement and would
have to be rewritten at every change -- which is a register that agrees on
the day it is written and disagrees silently ever after.

What is pinned instead is the route. Each clause below removes one choice
from a recipe and shows the result move: sorting, the joining rule, the
report the roster is read from, the escaping of published prose, and the
perimeter the two censuses run over. A recipe with a choice nobody can
observe is a recipe two readers can take differently, and the whole point
of writing them down is that they cannot.
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_RECIPES = REPO / "scripts" / "surface_recipes.py"


def _load():
    """Import the recipe module as a plain module, not through a window.

    It is a standalone program rather than a project module, so putting
    its directory on the path for the duration of the import is enough;
    the entry is removed again so nothing else in the run sees it. No
    module is seeded and nothing is stubbed, so this suite needs no
    isolation window and must not manufacture one.
    """
    directory = str(_RECIPES.parent)
    sys.path.insert(0, directory)
    try:
        return importlib.import_module(_RECIPES.stem)
    finally:
        if directory in sys.path:
            sys.path.remove(directory)


recipes = _load()


def _report(tmp_path, cases, wrapped=False):
    """Write a report naming ``cases`` as ``(classname, name)`` pairs."""
    body = "".join(
        f'<testcase classname="{classname}" name="{name}"/>'
        for classname, name in cases
    )
    suite = f'<testsuite tests="{len(cases)}">{body}</testsuite>'
    document = f"<testsuites>{suite}</testsuites>" if wrapped else suite
    path = tmp_path / "report.xml"
    path.write_text(f'<?xml version="1.0"?>{document}', encoding="utf-8")
    return path


def _manifest(tmp_path, paths):
    path = tmp_path / "manifest.md5"
    path.write_text(
        "".join(f"{'0' * 32}  ./{name}\n" for name in paths), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# sr1 -- the roster is sorted, newline-joined, and carries no trailing newline
# ---------------------------------------------------------------------------
def test_sr1_the_roster_is_sorted_and_joined_without_a_trailing_newline(tmp_path):
    """Three choices, each one observable.

    Order, separator and the absence of a final separator are the whole
    of the recipe. If any of them were free, two honest readings of one
    report would disagree and neither would be wrong.
    """
    cases = [("b_suite", "second"), ("a_suite", "first")]
    report = _report(tmp_path, cases)

    ordered = ["a_suite::first", "b_suite::second"]
    assert recipes.suite_roster(report) == ordered, (
        "the roster must be sorted, not left in the order the run emitted"
    )

    expected = hashlib.md5("\n".join(ordered).encode("utf-8")).hexdigest()
    assert recipes.suite_fingerprint(report) == expected

    unsorted = hashlib.md5(
        "\n".join(reversed(ordered)).encode("utf-8")
    ).hexdigest()
    trailing = hashlib.md5(
        ("\n".join(ordered) + "\n").encode("utf-8")
    ).hexdigest()
    assert expected not in (unsorted, trailing), (
        "order and the trailing newline must both change the digest, or "
        "neither is being pinned by this clause"
    )


# ---------------------------------------------------------------------------
# sr2 -- the roster is read from the report, and an empty report is refused
# ---------------------------------------------------------------------------
def test_sr2_the_roster_reads_the_report_and_refuses_an_empty_one(tmp_path):
    """A report naming nothing must raise, never digest the empty string.

    The digest of no cases at all is a perfectly ordinary-looking hex
    string. A run that collected nothing would publish it and read as a
    tree that simply changed, which is the failure this refuses.
    """
    flat = _report(tmp_path, [("suite", "one")])
    (tmp_path / "nested").mkdir()
    nested = _report(tmp_path / "nested", [("suite", "one")], wrapped=True)
    assert recipes.suite_fingerprint(flat) == recipes.suite_fingerprint(nested), (
        "a report wrapped in a suite list names the same cases and must "
        "digest the same"
    )

    empty = tmp_path / "empty.xml"
    empty.write_text('<?xml version="1.0"?><testsuite tests="0"/>', encoding="utf-8")
    with pytest.raises(ValueError):
        recipes.suite_fingerprint(empty)


# ---------------------------------------------------------------------------
# sr3 -- the interface digest sorts keys and leaves prose as written
# ---------------------------------------------------------------------------
def test_sr3_the_interface_digest_is_order_free_and_keeps_prose_as_written():
    """Two choices, both observable on a two-key document."""
    spec = {"beta": "resume", "alpha": "cafe\u0301 de\u0301tail"}
    reordered = {"alpha": spec["alpha"], "beta": spec["beta"]}
    assert recipes.openapi_digest(spec) == recipes.openapi_digest(reordered), (
        "key order must not reach the digest"
    )

    escaped = hashlib.md5(
        json.dumps(spec, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    assert recipes.openapi_digest(spec) != escaped, (
        "escaping published prose must change the digest, or the recipe "
        "is not pinning how prose is serialised"
    )

    assert recipes.openapi_text(spec) == recipes.openapi_text(reordered), (
        "the serialisation the digest is taken over must itself be "
        "order-free, or the agreement above is an accident of hashing"
    )


# ---------------------------------------------------------------------------
# sr4 -- the perimeter is the manifest, and an empty one is refused
# ---------------------------------------------------------------------------
def test_sr4_the_perimeter_is_read_from_the_manifest_and_never_empty(tmp_path):
    """A perimeter listing nothing reads exactly like a clean tree.

    Both censuses report zero over an empty perimeter, and zero is the
    answer everyone wants, so a mistyped manifest path would be believed.
    It raises instead.
    """
    manifest = _manifest(tmp_path, ["opti_oignon/a.py", "tests/b.py"])
    perimeter = recipes.read_perimeter(manifest)
    assert perimeter == frozenset({"opti_oignon/a.py", "tests/b.py"}), (
        "the leading ./ of a manifest entry is not part of the path"
    )

    empty = tmp_path / "empty.md5"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        recipes.read_perimeter(empty)

    malformed = tmp_path / "malformed.md5"
    malformed.write_text("no-checksum-pair-here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        recipes.read_perimeter(malformed)


# ---------------------------------------------------------------------------
# sr5 -- the non-ASCII census charges only what the manifest lists
# ---------------------------------------------------------------------------
def test_sr5_the_non_ascii_census_charges_only_the_listed_paths(tmp_path):
    """An unlisted file under a scanned tree is not part of what shipped."""
    tree = tmp_path / "opti_oignon"
    (tree / "vendor").mkdir(parents=True)
    (tree / "shipped.py").write_text("caf\u00e9\n", encoding="utf-8")
    (tree / "vendor" / "installed.py").write_text(
        "\u00e9\u00e8\u00ea\n", encoding="utf-8"
    )

    listed = recipes.read_perimeter(_manifest(tmp_path, ["opti_oignon/shipped.py"]))
    assert recipes.non_ascii_census(tmp_path, listed, ("opti_oignon/",)) == (1, 1)

    everything = recipes.read_perimeter(
        _manifest(
            tmp_path,
            ["opti_oignon/shipped.py", "opti_oignon/vendor/installed.py"],
        )
    )
    assert recipes.non_ascii_census(tmp_path, everything, ("opti_oignon/",)) == (4, 2), (
        "the unlisted file must be the whole of the difference; if it is "
        "not, the perimeter is doing nothing here"
    )


# ---------------------------------------------------------------------------
# sr6 -- the nomenclature census charges only what the manifest lists
# ---------------------------------------------------------------------------
def test_sr6_the_nomenclature_census_charges_only_the_listed_paths(tmp_path):
    """Same perimeter, same reason, and the shipped detector.

    The detector is the guard's own, loaded from the tree rather than
    restated here: a census that carried its own copy of the rules would
    report a debt against rules nobody else applies.
    """
    guard = recipes.load_clean_guard()
    offending = "# " + "S" + "912 note\n"

    tree = tmp_path / "tests"
    (tree / "vendor").mkdir(parents=True)
    (tree / "shipped.py").write_text(offending, encoding="utf-8")
    (tree / "vendor" / "installed.py").write_text(offending, encoding="utf-8")

    listed = recipes.read_perimeter(_manifest(tmp_path, ["tests/shipped.py"]))
    assert recipes.nomenclature_census(
        tmp_path, listed, guard.find_violations, ("tests/",)
    ) == (1, 1)

    everything = recipes.read_perimeter(
        _manifest(tmp_path, ["tests/shipped.py", "tests/vendor/installed.py"])
    )
    assert recipes.nomenclature_census(
        tmp_path, everything, guard.find_violations, ("tests/",)
    ) == (2, 2)


# ---------------------------------------------------------------------------
# sr7 -- the scanned trees are read from the guard, not restated
# ---------------------------------------------------------------------------
def test_sr7_the_scanned_trees_come_from_the_guard():
    """One register for the trees that ship, and the guard owns it."""
    guard = recipes.load_clean_guard()
    assert recipes.scanned_trees(guard) == tuple(guard._SCAN_PATHS)
    assert recipes.scanned_trees() == tuple(guard._SCAN_PATHS)
    assert str(REPO / ".github" / "scripts") not in sys.path, (
        "the guard directory must not be left on the path after loading"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
