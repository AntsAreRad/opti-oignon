"""S266 release suite: the Notes transport/editor cycle close as v3.13.0.

A. Version sites at 3.13.0: the version file (exact and bare, no rc),
   pyproject hardcoded and matching the version file.
B. CHANGELOG: top entry v3.13.0 dated S266, the Notes collaboration cycle
   told bloc by bloc (the N.1 data layer through the confirmed-posture
   editor and the compaction watermark), the at-rest posture stated, the
   gating posture stated, honest host-assured limitations, no marketing
   prose, the previous entries retained verbatim.
C. README: sits-between refreshed to 3.13.0, the intro carries the notes
   collaboration sentence, the v3.13.0 features section ABOVE the retained
   v3.12.0 one, the superseded README truths re-asserted, and the new
   section telling the feature with substance.
D. Addopts lineage: the fourteen S266 supersessions carried, the prior
   families untouched, the count grown by exactly fourteen to 242.
E. Structure: the suite parses, is pure ASCII, avoids the selection
   literal, and re-asserts the held-version dunder pins at 3.13.0.

Red-before: on the pristine S265 tree the 3.13.0 assertions fail (the
version still 3.12.0, no v3.13.0 changelog entry, no README refresh, no
new deselects) while the design-green seam pins (the pyproject-matches
pin, the retained previous entries, the prior deselect families, the
structure family) hold green by construction. After the release edits
the whole suite is green.

Supersessions this suite re-asserts (deselect-plus-reassert; originals
never edited):
- The eight live s255 version-bearing release pins: the three version
  sites, the changelog top entry, the readme sits-between and the
  superseded-readme-truths node, the addopts-count node (its own growth
  supersedes the 228 assertion), and the structure dunder re-assertion;
  re-asserted here at 3.13.0 (the retained-sections truths are
  re-asserted verbatim).
- The six version-held structure pins held at 3.12.0 across the cycle
  (s256, s257, s258, s260, s261, s265), re-asserted at 3.13.0.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

FINAL = "3.13.0"
PREVIOUS = "3.12.0"
DATE = "2026-06-13"

S266_DESELECTS = (
    "test_s255_release.py::TestVersionRelease::test_version_file_is_3120",
    "test_s255_release.py::TestVersionRelease::test_version_bare_no_rc",
    "test_s255_release.py::TestVersionRelease::test_pyproject_version_is_3120_and_hardcoded",
    "test_s255_release.py::TestChangelogRelease::test_top_entry_is_3120",
    "test_s255_release.py::TestReadmeRelease::test_sits_between_refreshed_to_3120",
    "test_s255_release.py::TestReadmeRelease::test_superseded_readme_truths_reasserted",
    "test_s255_release.py::TestAddoptsLineage::test_count_grew_by_exactly_eight",
    "test_s255_release.py::TestStructure::test_version_reasserted_from_dunder",
    "test_s256_mobile_allowed.py::TestStructure::test_version_held",
    "test_s257_notes_publish_glue.py::TestReassertions::test_version_held_3_12_0",
    "test_s258_pairing_device_class.py::TestStructure::test_version_held",
    "test_s260_ui_toggles.py::TestStructure::test_version_held",
    "test_s261_debt_lot.py::TestStructure::test_version_held",
    "test_s265_note_editor.py::TestStructure::test_version_held_3_12_0",
)

# The lineage that existed before S266, never removed (append-only): the
# eight s255 supersessions and the s244 / s236 spot deselects s255 carried.
S255_LINEAGE = (
    "test_s236_release.py::TestVersionRelease::test_version_file_is_3110",
    "test_s236_release.py::TestVersionRelease::test_version_bare_no_rc",
    "test_s236_release.py::TestVersionRelease::test_pyproject_version_is_3110_and_hardcoded",
    "test_s236_release.py::TestChangelogRelease::test_top_entry_is_3110",
    "test_s236_release.py::TestReadmeRelease::test_sits_between_refreshed_to_3110",
    "test_s236_release.py::TestReadmeRelease::test_superseded_readme_truths_reasserted",
    "test_s244_manage_notes.py::TestAddoptsLineage::test_count_grew_by_exactly_twelve",
    "test_s254_drawing_ui.py::TestStructure::test_version_held",
)

PRIOR_SPOT_LINEAGE = (
    "test_s243_notes_data_layer.py::TestPremiseGuards::test_manage_notes_not_yet_a_state_mutation_tool",
    "test_s242_atrest_consistency.py::TestAddoptsLineageS242::test_count_grew_by_exactly_two_to_208",
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
# A. Version sites at 3.13.0
# ---------------------------------------------------------------------------


class TestVersionRelease:
    def test_version_file_is_3130(self):
        src = _read("opti_oignon", "__version__.py")
        assert f'"{FINAL}"' in src
        assert f'"{PREVIOUS}"' not in src

    def test_version_bare_no_rc(self):
        src = _read("opti_oignon", "__version__.py")
        m = re.search(r'__version__\s*=\s*"([^"]+)"', src)
        assert m is not None
        assert m.group(1) == FINAL

    def test_pyproject_version_is_3130_and_hardcoded(self):
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

    def test_top_entry_is_3130(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == FINAL

    def test_entry_header_dated_s266(self):
        assert f"## v{FINAL} -- {DATE} (S266)" in self.c

    def test_entry_tells_the_cycle_bloc_by_bloc(self):
        assert f"## v{FINAL}" in self.c
        entry = _norm(self._entry())
        for term in (
            "data layer",
            "manage_notes",
            "attachment",
            "transcription",
            "caption",
            "drawing",
            "editor",
            "watermark",
            "Yjs",
        ):
            assert term in entry, term

    def test_entry_states_the_at_rest_posture(self):
        assert f"## v{FINAL}" in self.c
        entry = _norm(self._entry())
        for term in (
            "append-only",
            "note_update",
            "safe_connect",
            "SQLCipher",
        ):
            assert term in entry, term

    def test_entry_gating_posture(self):
        assert f"## v{FINAL}" in self.c
        entry = _norm(self._entry())
        assert "serve floor" in entry
        assert "republish" in entry
        assert "N9-D" in entry

    def test_entry_honest_limitations(self):
        assert f"## v{FINAL}" in self.c
        entry = _norm(self._entry())
        assert "host-assured" in entry
        assert "paired devices" in entry
        assert "server-side merge" in entry

    def test_entry_no_marketing_prose(self):
        assert f"## v{FINAL}" in self.c
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
        assert f"## v{PREVIOUS} -- 2026-06-10 (S255)" in self.c
        assert "## v3.11.0 -- 2026-06-08 (S236)" in self.c
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

    def test_sits_between_refreshed_to_3130(self):
        assert f"Opti-Oignon v{FINAL} sits between" in self.src
        assert f"Opti-Oignon v{PREVIOUS} sits between" not in self.src

    def test_intro_carries_the_cycle_sentence(self):
        assert "the notes collaboration cycle" in self.src

    def test_features_section_above_retained_previous(self):
        new = f"## Features Added in v{FINAL} (Notes Collaboration Cycle)"
        old = f"## Features Added in v{PREVIOUS} (Notes Cycle)"
        assert new in self.src
        assert old in self.src
        assert self.src.index(new) < self.src.index(old)

    def test_superseded_readme_truths_reasserted(self):
        # The truths of the deselected s255 TestReadmeRelease nodes: the
        # retained v3.12.0 features section, the sits-between line (now at
        # 3.13.0), the retained notes-cycle and remote-inference phrases.
        assert f"## Features Added in v{PREVIOUS} (Notes Cycle)" in self.src
        assert f"Opti-Oignon v{FINAL} sits between" in self.src
        assert "the notes cycle" in self.src
        assert "the remote inference cycle" in self.src

    def test_new_section_substance(self):
        assert f"## Features Added in v{FINAL}" in self.src
        section = self.src.split(f"## Features Added in v{FINAL}")[1].split(
            "## Features Added in"
        )[0]
        flat = _norm(section)
        for term in (
            "collaborative",
            "editor",
            "watermark",
            "Veilid",
            "paired",
            "encrypted",
        ):
            assert term in flat, term


# ---------------------------------------------------------------------------
# D. Addopts lineage (deselect-plus-reassert; originals never edited)
# ---------------------------------------------------------------------------


class TestAddoptsLineage:
    def setup_method(self):
        self.src = _read("pyproject.toml")

    def test_carries_the_fourteen_s266_supersessions(self):
        for node in S266_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_prior_families_untouched(self):
        for node in S255_LINEAGE + PRIOR_SPOT_LINEAGE:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_count_grew_by_exactly_fourteen(self):
        # 228 deselects at the S255 close; the fourteen S266 supersessions
        # join (the eight live s255 release pins and the six version-held
        # dunder pins held at 3.12.0 across the cycle); nothing is ever
        # removed from the lineage.
        assert self.src.count("--deselect=") == 242


# ---------------------------------------------------------------------------
# E. Structure
# ---------------------------------------------------------------------------


class TestStructure:
    def test_suite_parses(self):
        ast.parse(Path(__file__).read_text(encoding="utf-8"))

    def test_suite_ascii(self):
        assert Path(__file__).read_text(encoding="utf-8").isascii()

    def test_this_suite_avoids_the_selection_literal(self):
        here = Path(__file__).read_text(encoding="utf-8")
        token = "sandbox" + "_manager"
        assert token not in here

    def test_version_reasserted_from_dunder(self):
        # The re-assertion of the superseded s255 dunder pin, at 3.13.0.
        m = re.search(
            r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py")
        )
        assert m is not None
        assert m.group(1) == FINAL
