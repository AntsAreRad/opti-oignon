"""S232 release suite: the AGT cycle close as v3.10.0.

Families:
A. Version sites at 3.10.0: the version file (exact and bare, no rc),
   pyproject hardcoded (no dynamic), pyproject/version-file consistency.
B. CHANGELOG: top entry v3.10.0 dated S232, the cycle told lot by lot
   (S228 tools and feedback, S229 loop hardening, S230 the harness), the
   inherited dispatch posture stated, the honest host-assured limitations
   naming HOST_SHAKEDOWN_S231.md and the summarize flag, no marketing
   prose, prior entries retained intact.
C. README: sits-between refreshed to 3.10.0, the intro sentence, the
   v3.10.0 features section ABOVE the retained v3.9.0 one, the
   /api/agent-eval table row, and the re-assertions of the three truths
   the superseded s227 TestReadme node carried.
D. AGT_SPEC close-out: the header Status note, the Section 11 delivery
   record (lots in order, why 3.10.0), the Section 13 answered-questions
   block, the Route A closed-pending-gap status.
E. ROADMAP_POST_AUDIT: the AGT entry rolled to LANDED and RELEASED at
   S232 (v3.10.0) while the s222 pins and the historical s227 governor
   pin stay untouched.
F. Supersessions and the addopts lineage, id by id: the six new S232
   deselects present; the five s214 and the fourteen s228 deselects
   untouched; the total deselect count grew by exactly six (192 -> 198,
   nothing removed).
G. Docs: the new agent-performance page covers the cycle, the mkdocs nav
   entry exists, sandboxed-agent.md carries the cross-reference.

Red-before discipline: on the pristine S231 tree the S232 mechanics fail
(version still 3.9.0, no v3.10.0 changelog entry, no README section, no
Status note, no roll, no new deselects, no docs page) while the
retained-truth guards (prior changelog entries, the s222 roadmap pins,
the historical governor pin, the s214/s228 deselect lineage) hold green
by construction. After the release edits the whole suite is green.

Supersessions this suite re-asserts (deselect-plus-reassert; originals
never edited):
- tests/test_s227_governor_bloc4.py::TestVersionRelease::
  test_version_file_is_390, ::test_version_bare_no_rc,
  ::test_pyproject_version_is_390_and_hardcoded;
  ::TestChangelogRelease::test_top_entry_is_390;
  ::TestReadme::test_feature_section_and_intro (all three of its truths
  re-asserted here, sits-between at 3.10.0).
- tests/test_s230_agt_lot3.py::TestDeliverablePins::test_version_holds
  (the source-read pin, re-asserted at 3.10.0).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FINAL = "3.10.0"
PREVIOUS = "3.9.0"


def _read(*parts: str) -> str:
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


def _norm(text: str) -> str:
    """Whitespace-flattening helper (the S221 lesson) for document pins."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# A. Version sites at 3.10.0
# ---------------------------------------------------------------------------


class TestVersionRelease:
    def test_version_file_is_3100(self):
        src = _read("opti_oignon", "__version__.py")
        assert f'"{FINAL}"' in src
        assert f'"{PREVIOUS}"' not in src

    def test_version_bare_no_rc(self):
        m = re.search(
            r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py")
        )
        assert m is not None
        assert m.group(1) == FINAL

    def test_pyproject_version_is_3100_and_hardcoded(self):
        src = _read("pyproject.toml")
        assert f'version = "{FINAL}"' in src
        data = tomllib.loads(src)
        assert "dynamic" not in data["project"]
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
        return self.c.split("## v3.10.0")[1].split("## v3.9.0")[0]

    def test_top_entry_is_3100(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == FINAL

    def test_entry_header_dated_s232(self):
        assert "## v3.10.0 -- 2026-06-08 (S232)" in self.c

    def test_entry_tells_the_cycle_lot_by_lot(self):
        entry = _norm(self._entry())
        for term in (
            "S228",
            "grep, glob and ls",
            "diagnostics-after-write",
            "todo",
            "depth-1 task",
            "recovery chain",
            "S229",
            ".agent/spill/",
            "doom-loop",
            "two-rounds reminder",
            "governor feed",
            "S230",
            "TaskSpec",
            "micro.yaml",
            "EvalRunner",
            "refuse or skip",
            "agent_eval_results.db",
            "/api/agent-eval",
            "run_agent_eval.sh",
            "opencode",
            "reference-only",
            "[SECURITY]",
        ):
            assert term in entry, term

    def test_entry_states_the_inherited_posture(self):
        entry = _norm(self._entry())
        assert "dispatch posture" in entry
        assert "nothing weakened" in entry

    def test_entry_honest_limitations(self):
        entry = _norm(self._entry())
        assert "HOST_SHAKEDOWN_S231.md" in entry
        assert "summarize" in entry
        assert "stays off" in entry
        assert "release before the walk" in entry

    def test_entry_no_marketing_prose(self):
        entry = self._entry().lower()
        for fluff in (
            "exciting",
            "amazing",
            "revolutionary",
            "game-chang",
            "seamless",
        ):
            assert fluff not in entry, fluff

    def test_previous_entries_retained(self):
        assert "## v3.9.0 -- 2026-06-07 (S227)" in self.c
        assert "## v3.8.0 -- 2026-06-06 (S214)" in self.c
        assert "## v3.7.0 -- 2026-06-05 (S208)" in self.c


# ---------------------------------------------------------------------------
# C. README
# ---------------------------------------------------------------------------


class TestReadmeRelease:
    def setup_method(self):
        self.src = _read("README.md")

    def test_sits_between_refreshed_to_3100(self):
        assert f"Opti-Oignon v{FINAL} sits between" in self.src
        assert f"Opti-Oignon v{PREVIOUS} sits between" not in self.src

    def test_intro_carries_the_cycle_sentence(self):
        assert "the agent performance cycle" in self.src

    def test_features_section_above_retained_previous(self):
        new = "## Features Added in v3.10.0 (Agent Performance Cycle)"
        old = "## Features Added in v3.9.0 (Resource Governor Cycle)"
        assert new in self.src
        assert old in self.src
        assert self.src.index(new) < self.src.index(old)

    def test_superseded_readme_truths_reasserted(self):
        # The three truths of the deselected s227 TestReadme node:
        # the retained v3.9.0 features section, the sits-between line
        # (now at 3.10.0), the retained governor intro phrase.
        assert "## Features Added in v3.9.0 (Resource Governor Cycle)" in self.src
        assert f"Opti-Oignon v{FINAL} sits between" in self.src
        assert "the resource governor cycle" in self.src

    def test_api_row(self):
        assert "| `/api/agent-eval` |" in self.src


# ---------------------------------------------------------------------------
# D. AGT_SPEC close-out
# ---------------------------------------------------------------------------


class TestSpecCloseout:
    def setup_method(self):
        self.src = _norm(_read("AGT_SPEC.md"))

    def test_header_status_note(self):
        assert "Status: RELEASED at S232 as v3.10.0" in self.src
        assert "HOST_SHAKEDOWN_S231.md" in self.src

    def test_delivery_record(self):
        assert "Delivery record: RELEASED" in self.src
        assert "Lot 1 S228" in self.src
        assert "Lot 2 S229" in self.src
        assert "Lot 3 S230" in self.src
        assert "feature-cycle-minor precedent" in self.src
        assert "release before the walk" in self.src

    def test_answered_questions(self):
        assert "Answered at the release (S232), with what shipped:" in self.src
        assert "todo_updated" in self.src
        assert "{todos, total, completed}" in self.src
        assert "mtime descending then name" in self.src
        assert "fed form" in self.src

    def test_route_a_closed_pending_gap(self):
        assert "Route A stays CLOSED pending" in self.src
        assert "a data point, not a verdict" in self.src


# ---------------------------------------------------------------------------
# E. ROADMAP_POST_AUDIT roll
# ---------------------------------------------------------------------------


class TestRoadmapRoll:
    def setup_method(self):
        self.src = _norm(_read("ROADMAP_POST_AUDIT.md"))

    def test_cycle_rolled(self):
        assert "LANDED and RELEASED at S232 (v3.10.0)" in self.src

    def test_s222_pins_still_hold(self):
        assert "Agent Performance cycle (AGT) -- spec WRITTEN at S222" in self.src
        assert "AGT_SPEC.md is the design contract" in self.src
        assert "sst/opencode 4519a1da, v1.16.2, MIT" in self.src
        assert "explicitly-arbitrated fallback spike" in self.src

    def test_historical_governor_pin_untouched(self):
        assert "LANDED and RELEASED at S227 (v3.9.0)" in self.src


# ---------------------------------------------------------------------------
# F. Supersessions and the addopts lineage
# ---------------------------------------------------------------------------

S232_DESELECTS = (
    "test_s227_governor_bloc4.py::TestVersionRelease::test_version_file_is_390",
    "test_s227_governor_bloc4.py::TestVersionRelease::test_version_bare_no_rc",
    "test_s227_governor_bloc4.py::TestVersionRelease::"
    "test_pyproject_version_is_390_and_hardcoded",
    "test_s227_governor_bloc4.py::TestChangelogRelease::test_top_entry_is_390",
    "test_s227_governor_bloc4.py::TestReadme::test_feature_section_and_intro",
    "test_s230_agt_lot3.py::TestDeliverablePins::test_version_holds",
)

S214_DESELECTS = (
    "test_s214_release.py::TestVersionRelease::test_version_file_is_380",
    "test_s214_release.py::TestVersionRelease::test_version_bare_no_rc",
    "test_s214_release.py::TestVersionRelease::"
    "test_pyproject_version_is_380_and_hardcoded",
    "test_s214_release.py::TestChangelogRelease::test_top_entry_is_380",
    "test_s214_release.py::TestReadmeRelease::test_sits_between_refreshed_to_380",
)

S228_DESELECTS = (
    "test_s176_tools.py::TestSchemas::test_six_schemas",
    "test_s176_tools.py::TestSchemas::test_handler_two_not_sandboxed",
    "test_s176_tools.py::TestRegistryPerMode::test_daily_handlers_are_non_sandbox_two",
    "test_s176_tools.py::TestRegistryPerMode::test_bulbe_has_no_handlers",
    "test_s176_tools.py::TestRegistryPerMode::test_bulbe_exposes_sandbox_only",
    "test_s176_tools.py::TestRegistryPerMode::test_unknown_mode_is_fail_secure_bulbe",
    "test_s176_tools.py::TestSchemas::"
    "test_sandbox_argument_names_cover_dispatch_lambdas",
    "test_s177_manage_skills.py::TestSchemaSupersede::test_seven_schemas",
    "test_s177_manage_skills.py::TestSchemaSupersede::"
    "test_manage_skills_is_third_non_sandbox",
    "test_s177_manage_skills.py::TestSchemaSupersede::test_daily_includes_manage_skills",
    "test_s177_manage_skills.py::TestSchemaSupersede::test_bulbe_excludes_manage_skills",
    "test_s222_agt_spec.py::TestSeamToolsSchemas::test_all_schemas_is_seven_today",
    "test_s222_agt_spec.py::TestSeamToolsSchemas::test_handler_names_are_the_three",
    "test_s222_agt_spec.py::TestSeamAllowlists::test_frozensets_exact",
)


class TestAddoptsLineage:
    def setup_method(self):
        self.src = _read("pyproject.toml")

    def test_carries_the_six_s232_supersessions(self):
        for node in S232_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_s214_five_untouched(self):
        for node in S214_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_s228_fourteen_untouched(self):
        for node in S228_DESELECTS:
            assert f"--deselect=tests/{node}" in self.src, node

    def test_count_grew_by_exactly_six(self):
        # 192 deselects at the S231 close; the six S232 supersessions
        # join; nothing is ever removed from the lineage.
        assert self.src.count("--deselect=") == 198


# ---------------------------------------------------------------------------
# G. Docs: the page, the nav, the cross-reference
# ---------------------------------------------------------------------------


class TestDocsRelease:
    def test_page_exists_and_covers_the_cycle(self):
        src = _norm(_read("docs", "agent", "agent-performance.md"))
        for term in (
            "grep",
            "glob",
            "ls",
            "todo",
            "task",
            "doom-loop",
            ".agent/spill/",
            "micro",
            "/api/agent-eval",
            "run_agent_eval.sh",
            "HOST_SHAKEDOWN_S231.md",
            "host-assured",
            "approval-gated",
        ):
            assert term in src, term

    def test_nav_entry(self):
        nav = _read("mkdocs.yml")
        assert "Agent Performance: agent/agent-performance.md" in nav

    def test_cross_reference_in_sandboxed_agent(self):
        src = _read("docs", "agent", "sandboxed-agent.md")
        assert "agent-performance.md" in src
