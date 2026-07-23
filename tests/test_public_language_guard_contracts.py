#!/usr/bin/env python3
"""Contracts for the public-language CI guard.

The published source and test trees are English-only. That rule is the one
the project states most strictly and the one nothing enforced: the
public-clean guard reads added lines for internal nomenclature and has no
notion of language at all. This guard supplies the missing half.

What it reads, and what it deliberately does not. Prose written for a human
reader -- comments and docstrings -- must be English. Text inside string
literals must not be touched: a classifier that recognises a French question
carries French patterns because its input is French, and a guard that
flagged those would be wrong about the code it polices. The distinction is
structural, not statistical, so it is drawn by parsing the post-image rather
than by reading diff lines in isolation.

Density, never a single word. An author's name, a borrowed noun, a product
term: none of those makes a sentence French. A span is charged only when it
carries at least two French function words and more French than English
ones, over a span long enough to have a grammar at all.

Two questions, one detector. Added lines answer "is new debt arriving"; the
whole file answers "has the standing debt grown". The first lets the guard
be adopted while the debt is still owed, exactly as its sibling was. The
second is the ratchet: a per-file seal that may fall and may not rise.

Every French input below lives in a string literal, so this suite is immune
to the guard it exercises -- by the same rule the guard states, not by an
exemption written for it.

Local-only, stdlib-only. The guard script lives under .github/, outside the
importable package, and is loaded through the shared isolation window.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate  # noqa: E402

_GUARD = "_public_language_guard_under_contract"
_GUARD_PATH = REPO / ".github" / "scripts" / "public_language_guard.py"


def _load():
    """Load the guard through the shared window; returns (module, restore)."""
    loaded, restore = isolate(targets={_GUARD: _GUARD_PATH})
    return loaded[_GUARD], restore


def _kinds(violations):
    return [kind for _line, kind, _text in violations]


def _lines(violations):
    return [line for line, _kind, _text in violations]


# ---------------------------------------------------------------------------
# l1 -- a French comment on an added line is flagged, nominatively
# ---------------------------------------------------------------------------
def test_l1_french_comment_on_an_added_line_is_flagged():
    guard, restore = _load()
    try:
        src = (
            "value = 1\n"
            "# Nombre maximum de recherches par tour de generation\n"
            "other = 2\n"
        )
        found = guard.find_violations(src, added_lines={2})
        assert len(found) == 1, "the French comment must be charged"
        assert _kinds(found) == ["comment"]
        assert _lines(found) == [2], "the report must name the line"
        assert "Nombre" in found[0][2], "the report must carry the offending text"
    finally:
        restore()


# ---------------------------------------------------------------------------
# l2 -- an English comment is never flagged
# ---------------------------------------------------------------------------
def test_l2_english_comment_is_not_flagged():
    guard, restore = _load()
    try:
        src = (
            "# Remove the stale chunks that belong to this file before "
            "reindexing it\n"
            "value = 1\n"
        )
        assert guard.find_violations(src, added_lines={1}) == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# l3 -- a French docstring on an added line is flagged
# ---------------------------------------------------------------------------
def test_l3_french_docstring_is_flagged():
    guard, restore = _load()
    try:
        src = (
            "def f():\n"
            '    """Estimation rapide du nombre de tokens dans la chaine."""\n'
            "    return 0\n"
        )
        found = guard.find_violations(src, added_lines={2})
        assert _kinds(found) == ["docstring"], "a docstring is prose too"
        assert _lines(found) == [2]
    finally:
        restore()


# ---------------------------------------------------------------------------
# l4 -- French inside a string literal is data, and survives untouched
# ---------------------------------------------------------------------------
def test_l4_french_inside_a_string_literal_is_never_flagged():
    guard, restore = _load()
    try:
        src = (
            "PATTERNS = [\n"
            '    r"\\bproblème\\b", r"\\baide\\b.*\\brésoudre\\b",\n'
            '    r"\\banalyse\\b.*\\bdonnées\\b", r"\\bécologie\\b",\n'
            "]\n"
            'MESSAGE = "Le fichier est introuvable dans le repertoire"\n'
        )
        found = guard.find_violations(src, added_lines={1, 2, 3, 4, 5})
        assert found == [], (
            "a classifier that recognises French input must be able to carry "
            "French patterns"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# l5 -- a half-translated comment is flagged: the debt's real shape
# ---------------------------------------------------------------------------
def test_l5_half_translated_comment_is_flagged():
    guard, restore = _load()
    try:
        src = (
            "# Check que la reponse is not trop courte pour le nb de "
            "questions\n"
            "value = 1\n"
        )
        found = guard.find_violations(src, added_lines={1})
        assert len(found) == 1, (
            "English tokens dropped into French grammar is still not English"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# l6 -- a name or a lone borrowed word is below the density floor
# ---------------------------------------------------------------------------
def test_l6_a_name_or_a_single_word_is_below_the_floor():
    guard, restore = _load()
    try:
        # One input per floor, each sitting exactly on its boundary, so no
        # part of the density rule can be removed without a red.
        for text in (
            "# Author: Léon",
            "# le fichier",
            "# Chaine normalisation happens once per run",
            "# The archive trigger fires on a phrase like ne pas",
        ):
            src = text + "\nvalue = 1\n"
            assert guard.find_violations(src, added_lines={1}) == [], (
                f"must not charge: {text}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# l7 -- diff-only: French outside the added lines is passed over
# ---------------------------------------------------------------------------
def test_l7_french_outside_the_added_lines_is_passed_over():
    guard, restore = _load()
    try:
        src = (
            "# Nombre maximum de recherches par tour de generation\n"
            "value = 1\n"
            "# Retirer les anciens chunks de ce fichier avant de le relire\n"
        )
        assert guard.find_violations(src, added_lines={2}) == [], (
            "standing debt must not block adoption"
        )
        assert len(guard.find_violations(src, added_lines={1, 3})) == 2
    finally:
        restore()


# ---------------------------------------------------------------------------
# l8 -- a source that does not parse is reported, never silently skipped
# ---------------------------------------------------------------------------
def test_l8_unparsable_source_is_reported_not_skipped():
    guard, restore = _load()
    try:
        found = guard.find_violations("def (:\n", added_lines={1})
        assert _kinds(found) == ["unparsable"], (
            "a file the guard cannot read must not read as a clean file"
        )
        assert guard.census("def (:\n") == 0, (
            "an unreadable file owes no language debt: it owes a fix"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# l9 -- the census counts every French span, added or not
# ---------------------------------------------------------------------------
def test_l9_census_counts_the_whole_source():
    guard, restore = _load()
    try:
        src = (
            "# Nombre maximum de recherches par tour de generation\n"
            "def f():\n"
            '    """Estimation rapide du nombre de tokens dans la chaine."""\n'
            "    # Remove the stale chunks before reindexing this file\n"
            "    return 0\n"
        )
        assert guard.census(src) == 2, "two French spans, one English"
    finally:
        restore()


# ---------------------------------------------------------------------------
# l10 -- a count above its seal is a regression
# ---------------------------------------------------------------------------
def test_l10_a_count_above_its_seal_is_a_regression():
    guard, restore = _load()
    try:
        found = guard.find_ledger_regressions(
            {"a.py": 4, "b.py": 2}, ledger={"a.py": 3, "b.py": 2},
        )
        assert found == [("a.py", 3, 4)], (
            "the report must name the file, its seal and what it now carries"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# l11 -- paying the debt down is never a regression
# ---------------------------------------------------------------------------
def test_l11_paying_down_is_not_a_regression():
    guard, restore = _load()
    try:
        assert guard.find_ledger_regressions(
            {"a.py": 0, "b.py": 1}, ledger={"a.py": 3, "b.py": 2},
        ) == [], "the ratchet must let the debt fall"
        assert guard.find_ledger_regressions(
            {"c.py": 1}, ledger={},
        ) == [("c.py", 0, 1)], "an unsealed file owes nothing, so it may carry nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# l12 -- the sealed ledger matches the tree as it stands
# ---------------------------------------------------------------------------
def test_l12_the_sealed_ledger_matches_the_tree():
    guard, restore = _load()
    try:
        counts = guard.census_tree(REPO)
        regressions = guard.find_ledger_regressions(counts)
        assert regressions == [], (
            "the seal must describe the tree it was taken from"
        )
        stale = sorted(set(guard.LEDGER) - set(counts))
        assert stale == [], f"the ledger names files that carry nothing: {stale}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# l13 -- a docstring English at the head and French below is charged, and
#        the report quotes the French line rather than the innocent head
# ---------------------------------------------------------------------------
def test_l13_report_quotes_the_offending_line_not_the_english_head():
    guard, restore = _load()
    try:
        src = (
            "def f():\n"
            '    """Tool-call results of the last run.\n'
            "\n"
            "    Returns:\n"
            "        Liste de ToolCallResult, vide si pas d'appels d'outils.\n"
            '    """\n'
            "    return 0\n"
        )
        found = guard.find_violations(src, added_lines={2})
        assert len(found) == 1, "a docstring French below its head is prose too"
        assert "Liste" in found[0][2], (
            "the report must quote the French line, not the English head"
        )
        assert "Tool-call" not in found[0][2]
    finally:
        restore()


# ---------------------------------------------------------------------------
# l14 -- the perimeter names the script tree, and every entry in it is a
#        tree this guard can actually read
# ---------------------------------------------------------------------------
def test_l14_the_perimeter_names_only_trees_the_guard_can_read():
    guard, restore = _load()
    try:
        assert "scripts/" in set(guard._SCAN_PATHS), (
            "the script tree carries Python and must be guarded like the rest"
        )
        assert guard.unreadable_scan_paths(REPO) == (), (
            "a perimeter entry this guard cannot read must never be left in "
            "place: it would turn the final report into an all-clear over a "
            "tree that was never opened"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# l15 -- a tree the guard cannot read is NAMED, never signed clean
# ---------------------------------------------------------------------------
def test_l15_a_tree_the_guard_cannot_read_is_named():
    guard, restore = _load()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "readable").mkdir()
            (root / "readable" / "mod.py").write_text(
                "x = 1\n", encoding="utf-8",
            )
            (root / "opaque").mkdir()
            (root / "opaque" / "view.ts").write_text(
                "// Liste des valeurs pour chaque entree\n", encoding="utf-8",
            )
            found = guard.unreadable_scan_paths(
                root, scan_paths=("readable/", "opaque/"),
            )
            assert found == ("opaque/",), (
                "the tree carrying no readable file is the one to name"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# l16 -- the census reads Python and nothing else, and that is deliberate:
#        French prose in a neighbouring file of another language is NOT seen
# ---------------------------------------------------------------------------
def test_l16_the_census_reads_python_and_nothing_else():
    guard, restore = _load()
    try:
        assert guard._SCANNED_SUFFIX == ".py", (
            "the suffix the census reads must be stated, not implied by a glob"
        )
        prose = "Liste des valeurs pour chaque entree du fichier\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tree").mkdir()
            (root / "tree" / "mod.py").write_text(
                "# " + prose + "x = 1\n", encoding="utf-8",
            )
            # The neighbour carries prose the detector CAN read, on purpose:
            # were it written in a syntax the parser rejects, it would be
            # skipped as unreadable and this clause would pass under any glob,
            # pinning nothing. Readable content leaves the suffix as the only
            # thing that can keep it out of the count.
            (root / "tree" / "view.ts").write_text(
                "# " + prose + "value = 2\n", encoding="utf-8",
            )
            counts = guard.census_tree(root, scan_paths=("tree/",))
            assert counts == {"tree/mod.py": 1}, (
                "the neighbouring file of another language is invisible here; "
                "that blindness is why a perimeter must not name its tree"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# l17 -- the entry point ACTS on the refusal; computing it is not enough
# ---------------------------------------------------------------------------
def test_l17_the_entry_point_fails_on_an_unreadable_perimeter():
    guard, restore = _load()
    try:
        original = guard._SCAN_PATHS
        # docs/ is a real tree of the repository and carries no Python at all.
        guard._SCAN_PATHS = ("opti_oignon/", "docs/")
        guard._added_lines_by_path = lambda _base_ref: {}
        try:
            assert guard.main(["HEAD"]) == 1, (
                "a perimeter naming a tree the guard cannot read must fail the "
                "run, not merely be computed and discarded"
            )
        finally:
            guard._SCAN_PATHS = original
    finally:
        restore()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
