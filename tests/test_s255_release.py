"""S255 release suite: the Notes cycle close as v3.12.0.

A. Version sites at 3.12.0: the version file (exact and bare, no rc),
   pyproject hardcoded and matching the version file.
B. CHANGELOG: top entry v3.12.0 dated S255, the cycle told bloc by bloc
   (the N.1 data layer through the N.7 drawing canvas and the N.8 sync
   record kind), the at-rest posture stated, honest host-assured
   limitations, no marketing prose, the previous entries retained.
C. README: sits-between refreshed to 3.12.0, the intro carries the notes
   cycle sentence, the v3.12.0 features section ABOVE the retained
   v3.11.0 one, the superseded README truths re-asserted, and the Notes
   section telling the feature with substance.
D. Addopts lineage: the eight S255 supersessions carried, the prior
   families untouched, the count grown by exactly eight to 228.
E. Structure: the suite parses and is pure ASCII.

Red-before: on the pristine S254 tree the 3.12.0 assertions fail (the
version still 3.11.0, no v3.12.0 changelog entry, no README refresh, no
new deselects) while the design-green seam pins (the pyproject-matches
pin, the retained previous entries, the prior deselect families, the
structure family) hold green by construction. After the release edits
the whole suite is green.

Supersessions this suite re-asserts (deselect-plus-reassert; originals
never edited):
- tests/test_s236_release.py::TestVersionRelease::test_version_file_is_3110,
  ::test_version_bare_no_rc,
  ::test_pyproject_version_is_3110_and_hardcoded;
  ::TestChangelogRelease::test_top_entry_is_3110;
  ::TestReadmeRelease::test_sits_between_refreshed_to_3110,
  ::test_superseded_readme_truths_reasserted (the version-bearing
  truths, re-asserted here at 3.12.0; the retained-sections truths are
  re-asserted verbatim).
- tests/test_s244_manage_notes.py::TestAddoptsLineage::
  test_count_grew_by_exactly_twelve (the live addopts-count node,
  re-asserted at the new total 228).
- tests/test_s254_drawing_ui.py::TestStructure::test_version_held (the
  dunder literal, re-asserted at 3.12.0).
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FINAL = "3.12.0"
PREVIOUS = "3.11.0"

S255_DESELECTS = (
    "test_s236_release.py::TestVersionRelease::test_version_file_is_3110",
    "test_s236_release.py::TestVersionRelease::test_version_bare_no_rc",
    "test_s236_release.py::TestVersionRelease::test_pyproject_version_is_3110_and_hardcoded",
    "test_s236_release.py::TestChangelogRelease::test_top_entry_is_3110",
    "test_s236_release.py::TestReadmeRelease::test_sits_between_refreshed_to_3110",
    "test_s236_release.py::TestReadmeRelease::test_superseded_readme_truths_reasserted",
    "test_s244_manage_notes.py::TestAddoptsLineage::test_count_grew_by_exactly_twelve",
    "test_s254_drawing_ui.py::TestStructure::test_version_held",
)

S244_SPOT_DESELECTS = (
    "test_s243_notes_data_layer.py::TestPremiseGuards::test_manage_notes_not_yet_a_state_mutation_tool",
    "test_s242_atrest_consistency.py::TestAddoptsLineageS242::test_count_grew_by_exactly_two_to_208",
)

S236_SPOT_DESELECTS = (
    "test_s232_release.py::TestVersionRelease::test_version_file_is_3100",
    "test_s232_release.py::TestReadmeRelease::test_superseded_readme_truths_reasserted",
    "test_s236_release.py::TestAddoptsLineage::test_count_grew_by_exactly_eight",
)


def _read(*parts: str) -> str:
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


def _norm(text: str) -> str:
    """Whitespace-flattening helper (the S221 lesson) for document pins."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# A. Version sites at 3.12.0
# ---------------------------------------------------------------------------


class TestVersionRelease:
    def test_version_file_is_3120(self):
        src = _read("opti_oignon", "__version__.py")
        assert f'"{FINAL}"' in src
        assert f'"{PREVIOUS}"' not in src

    def test_version_bare_no_rc(self):
        src = _read("opti_oignon", "__version__.py")
        m = re.search(r'__version__\s*=\s*"([^"]+)"', src)
        assert m is not None
        assert m.group(1) == FINAL

    def test_pyproject_version_is_3120_and_hardcoded(self):
        src = _read("pyproject.toml")
        assert f'version = "{FINAL}"' in src
        data = tomllib.loads(src)
        assert data["project"]["version"] == FINAL

    def test_pyproject_matches_version_file(self):
        m = re.search(
            r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py")
        )
        data = tomllib.loads(_read("pyproject.toml"))
        assert m is not None
        assert data["project"]["version"] == m.group(1)


# ---------------------------------------------------------------------------
# B. CHANGELOG
# ---------------------------------------------------------------------------


class TestChangelogRelease:
    def setup_method(self):
        self.c = _read("CHANGELOG.md")

    def _entry(self) -> str:
        return self.c.split(f"## v{FINAL}")[1].split(f"## v{PREVIOUS}")[0]

    def test_top_entry_is_3120(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == FINAL

    def test_entry_header_dated_s255(self):
        assert f"## v{FINAL} -- 2026-06-10 (S255)" in self.c

    def test_entry_tells_the_cycle_bloc_by_bloc(self):
        entry = _norm(self._entry())
        for term in (
            "data layer",
            "manage_notes",
            "attachment",
            "transcription",
            "caption",
            "drawing",
            "RecordKind.NOTE",
            "Yjs",
        ):
            assert term in entry, term

    def test_entry_states_the_at_rest_posture(self):
        entry = _norm(self._entry())
        for term in (
            "two independent",
            "AES-256-GCM",
            "HKDF",
            "SQLCipher",
            "bubblewrap",
        ):
            assert term in entry, term

    def test_entry_gating_posture(self):
        entry = _norm(self._entry())
        assert "STATE_MUTATION_TOOLS" in entry
        assert "Bulbe" in entry
        assert "untrusted" in entry

    def test_entry_honest_limitations(self):
        entry = _norm(self._entry())
        assert "host-assured" in entry
        assert "opt-in" in entry

    def test_entry_no_marketing_prose(self):
        entry = _norm(self._entry()).lower()
        for word in (
            "revolutionary",
            "blazing",
            "world-class",
            "game-changing",
            "seamless",
            "effortless",
        ):
            assert word not in entry, word

    def test_previous_entries_retained(self):
        assert f"## v{PREVIOUS} -- 2026-06-08 (S236)" in self.c
        assert "## v3.10.0 -- 2026-06-08 (S232)" in self.c
        assert "## v3.9.0 -- 2026-06-07 (S227)" in self.c
        assert "## v3.8.0 -- 2026-06-06 (S214)" in self.c
        assert "## v3.7.0 -- 2026-06-05 (S208)" in self.c


# ---------------------------------------------------------------------------
# C. README
# ---------------------------------------------------------------------------


class TestReadmeRelease:
    def setup_method(self):
        self.src = _read("README.md")

    def test_sits_between_refreshed_to_3120(self):
        assert f"Opti-Oignon v{FINAL} sits between" in self.src
        assert f"Opti-Oignon v{PREVIOUS} sits between" not in self.src

    def test_intro_carries_the_cycle_sentence(self):
        assert "the notes cycle" in self.src

    def test_features_section_above_retained_previous(self):
        new = f"## Features Added in v{FINAL} (Notes Cycle)"
        old = f"## Features Added in v{PREVIOUS} (Remote Inference Cycle)"
        assert new in self.src
        assert old in self.src
        assert self.src.index(new) < self.src.index(old)

    def test_superseded_readme_truths_reasserted(self):
        # The truths of the deselected s236 TestReadmeRelease nodes: the
        # retained v3.11.0 features section, the sits-between line (now at
        # 3.12.0), the retained remote-inference cycle phrase.
        assert (
            f"## Features Added in v{PREVIOUS} (Remote Inference Cycle)" in self.src
        )
        assert f"Opti-Oignon v{FINAL} sits between" in self.src
        assert "the remote inference cycle" in self.src

    def test_notes_section_substance(self):
        section = self.src.split(f"## Features Added in v{FINAL}")[1].split(
            "## Features Added in"
        )[0]
        flat = _norm(section)
        for term in (
            "Notes",
            "voice",
            "picture",
            "drawing",
            "Veilid",
            "encrypted",
            "manage_notes",
        ):
            assert term in flat, term


# ---------------------------------------------------------------------------
# D. Addopts lineage (deselect-plus-reassert; originals never edited)
# ---------------------------------------------------------------------------


class TestAddoptsLineage:
    def setup_method(self):
        self.src = _read("pyproject.toml")

    def test_carries_the_eight_s255_supersessions(self):
        for node in S255_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_prior_families_untouched(self):
        for node in S244_SPOT_DESELECTS + S236_SPOT_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_count_grew_by_exactly_eight(self):
        # 220 deselects at the S254 close; the eight S255 supersessions
        # join (the six live s236 release pins, the s244 addopts-count pin
        # this lot's own growth supersedes, and the s254 version-held
        # dunder pin); nothing is ever removed from the lineage.
        assert self.src.count("--deselect=") == 228


# ---------------------------------------------------------------------------
# E. Structure
# ---------------------------------------------------------------------------


class TestStructure:
    def test_suite_parses(self):
        ast.parse(Path(__file__).read_text(encoding="utf-8"))

    def test_suite_ascii(self):
        assert Path(__file__).read_text(encoding="utf-8").isascii()

    def test_version_reasserted_from_dunder(self):
        # The re-assertion of the superseded s254 dunder pin, at 3.12.0.
        m = re.search(
            r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py")
        )
        assert m is not None
        assert m.group(1) == FINAL
