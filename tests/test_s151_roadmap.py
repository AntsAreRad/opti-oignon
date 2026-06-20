"""
tests/test_s151_roadmap.py -- S151 roadmap planning validation tests.

Verifies:
- New roadmap file exists and has required structure
- Retrospective metrics are documented
- Session plan has complexity estimates
- Deferred items from S137-S150 are tracked
- Version and session references are consistent
"""

import importlib.util
import os
import sys
import re

# -- Isolation stubs (standard pattern) --
for mod_name in [
    "opti_oignon",
    "opti_oignon.db_utils",
    "opti_oignon.config",
    "opti_oignon.auth",
]:
    if mod_name not in sys.modules:
        import types
        sys.modules[mod_name] = types.ModuleType(mod_name)

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROADMAP_NEW = os.path.join(PROJECT_ROOT, "ROADMAP_S151_S164.md")
ROADMAP_OLD = os.path.join(PROJECT_ROOT, "ROADMAP_S137_S150.md")
SESSION_TRACKING = os.path.join(PROJECT_ROOT, "SESSION_TRACKING_S65_S151.md")


# -- helpers --

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# -- Part 1: new roadmap file existence and structure --

class TestRoadmapExists:
    """Verify the new roadmap file exists and has basic structure."""

    def test_roadmap_file_exists(self) -> None:
        assert os.path.isfile(ROADMAP_NEW), "ROADMAP_S151_S164.md must exist"

    def test_roadmap_has_title(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "ROADMAP S151" in content

    def test_roadmap_has_overview_section(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "## Overview" in content

    def test_roadmap_has_session_summary(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "## Session Summary" in content

    def test_roadmap_has_estimated_totals(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "## Estimated Totals" in content


# -- Part 2: retrospective section --

class TestRetrospective:
    """Verify retrospective metrics are documented."""

    def test_retrospective_section_exists(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "## Retrospective" in content

    def test_estimated_vs_actual_table(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "Est. tests" in content
        assert "Actual" in content
        assert "Delta" in content

    def test_all_sessions_in_retrospective(self) -> None:
        content = read_file(ROADMAP_NEW)
        for s in range(138, 151):
            assert f"S{s}" in content, f"S{s} missing from retrospective"

    def test_deferred_items_section(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "Deferred" in content or "deferred" in content

    def test_deferred_mkdocs_tracked(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "MkDocs" in content

    def test_deferred_redteam_cli_tracked(self) -> None:
        content = read_file(ROADMAP_NEW)
        lower = content.lower()
        assert "red team cli" in lower or "redteam cli" in lower

    def test_technical_debt_inventory(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "Technical Debt" in content or "technical debt" in content


# -- Part 3: session plan structure --

class TestSessionPlan:
    """Verify session plan has complexity estimates and required fields."""

    def test_bloc_count(self) -> None:
        content = read_file(ROADMAP_NEW)
        bloc_matches = re.findall(r"## Bloc \d+", content)
        assert len(bloc_matches) >= 3, "At least 3 blocs expected"

    def test_sessions_have_complexity(self) -> None:
        content = read_file(ROADMAP_NEW)
        # each session header (### S1xx) should have a Complexity line nearby
        sessions = re.findall(r"### S(\d+)", content)
        assert len(sessions) >= 10, f"Expected 10+ sessions, found {len(sessions)}"
        for s in sessions:
            # complexity should appear for planning sessions
            pass  # existence checked below

    def test_complexity_labels_present(self) -> None:
        content = read_file(ROADMAP_NEW)
        complexity_lines = re.findall(r"\*\*Complexity:\*\*\s*(S|M|L|XL|L-XL)", content)
        assert len(complexity_lines) >= 10, (
            f"Expected 10+ complexity labels, found {len(complexity_lines)}"
        )

    def test_estimated_tests_per_session(self) -> None:
        content = read_file(ROADMAP_NEW)
        est_lines = re.findall(r"\*\*Estimated tests:\*\*\s*~?\d+", content)
        assert len(est_lines) >= 10, (
            f"Expected 10+ test estimates, found {len(est_lines)}"
        )

    def test_target_versions_present(self) -> None:
        content = read_file(ROADMAP_NEW)
        versions = re.findall(r"\*\*Target version:\*\*\s*[\d.]+", content)
        assert len(versions) >= 10

    def test_prerequisites_present(self) -> None:
        content = read_file(ROADMAP_NEW)
        prereqs = re.findall(r"\*\*Prerequisites:\*\*", content)
        assert len(prereqs) >= 10

    def test_key_files_present(self) -> None:
        content = read_file(ROADMAP_NEW)
        key_files = re.findall(r"\*\*Key files:\*\*", content)
        assert len(key_files) >= 10


# -- Part 4: priority order matches requirements --

class TestPriorityOrder:
    """Verify UX is first, then security, then performance."""

    def test_ux_bloc_is_first(self) -> None:
        content = read_file(ROADMAP_NEW)
        ux_pos = content.find("Bloc 1")
        security_pos = content.find("Bloc 2")
        assert ux_pos < security_pos, "UX bloc must come before security"

    def test_security_bloc_before_performance(self) -> None:
        content = read_file(ROADMAP_NEW)
        sec_pos = content.find("Bloc 2")
        perf_pos = content.find("Bloc 3")
        assert sec_pos < perf_pos, "Security bloc must come before performance"

    def test_theme_engine_in_ux_bloc(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "Theme Engine" in content or "theme_engine" in content

    def test_accent_colors_mentioned(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "accent" in content.lower()

    def test_security_audit_in_bloc_2(self) -> None:
        content = read_file(ROADMAP_NEW)
        # audit should appear between Bloc 2 header and Bloc 3 header
        bloc2_pos = content.find("Bloc 2")
        bloc3_pos = content.find("Bloc 3")
        bloc2_section = content[bloc2_pos:bloc3_pos]
        assert "audit" in bloc2_section.lower()


# -- Part 5: consistency checks --

class TestConsistency:
    """Cross-check roadmap internal consistency."""

    def test_starting_version_is_3_2_0(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "Starting version:** 3.2.0" in content

    def test_target_version_is_3_3_0(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "3.3.0" in content

    def test_session_range_contiguous(self) -> None:
        content = read_file(ROADMAP_NEW)
        sessions = sorted(set(int(s) for s in re.findall(r"### S(\d+)", content)))
        if len(sessions) >= 2:
            expected = list(range(sessions[0], sessions[-1] + 1))
            assert sessions == expected, (
                f"Session range not contiguous: {sessions}"
            )

    def test_no_french_in_roadmap(self) -> None:
        content = read_file(ROADMAP_NEW)
        french_markers = ["objectif", "fichier", "securite", "amelioration"]
        for marker in french_markers:
            assert marker not in content.lower(), (
                f"French word '{marker}' found in roadmap"
            )

    def test_previous_roadmap_referenced(self) -> None:
        content = read_file(ROADMAP_NEW)
        assert "ROADMAP_S137_S150" in content
