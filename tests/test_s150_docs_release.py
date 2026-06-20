#!/usr/bin/env python3
"""
Tests for S150 -- Documentation Pass and v3.2.0 Release
========================================================

Covers:
- Part 1:  README.md contains required sections
- Part 2:  README.md updated counts and version
- Part 3:  SECURITY.md version and new sections
- Part 4:  SECURITY.md Known Limitations resolved items
- Part 5:  SECURITY.md version history completeness
- Part 6:  CHANGELOG.md has v3.2.0 entry
- Part 7:  CHANGELOG.md v3.2.0 categories present
- Part 8:  docs/API_REFERENCE.md exists with endpoint sections
- Part 9:  API_REFERENCE.md endpoint completeness
- Part 10: Version is 3.2.0 (no rc suffix)
- Part 11: Version consistency across test files
- Part 12: E2E mock version updated
"""

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_PROJECT = Path(__file__).resolve().parent.parent
_DOCS = _PROJECT / "docs"


def _load_version():
    """Load __version__ via importlib to avoid import chain."""
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.__version__",
        _PROJECT / "opti_oignon" / "__version__.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.__version__


def _read(path):
    """Read file content as string."""
    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Part 1: README.md contains required sections
# ---------------------------------------------------------------------------

class TestPart01ReadmeSections:
    """README.md must contain all required sections."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read(_PROJECT / "README.md")

    def test_security_architecture_section(self):
        assert "## Security Architecture" in self.content

    def test_features_added_section(self):
        assert "## Features Added in v3.2.0" in self.content

    def test_e2e_tests_subsection(self):
        assert "### E2E Tests" in self.content

    def test_type_checking_subsection(self):
        assert "### Type Checking" in self.content

    def test_api_endpoints_section(self):
        assert "## API Endpoints" in self.content

    def test_project_structure_section(self):
        assert "## Project Structure" in self.content

    def test_configuration_reference_section(self):
        assert "## Configuration Reference" in self.content

    def test_contributing_md_link(self):
        assert "CONTRIBUTING.md" in self.content

    def test_api_reference_link(self):
        assert "docs/API_REFERENCE.md" in self.content


# ---------------------------------------------------------------------------
# Part 2: README.md updated counts and version
# ---------------------------------------------------------------------------

class TestPart02ReadmeCounts:
    """README.md reflects current project state."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read(_PROJECT / "README.md")

    def test_version_320(self):
        assert "v3.2.0" in self.content

    def test_endpoint_count_updated(self):
        assert "494+" in self.content

    def test_test_count_updated(self):
        assert "8172" in self.content

    def test_test_file_count_updated(self):
        assert "137 test files" in self.content

    def test_module_count_updated(self):
        assert "~240" in self.content

    def test_svelte_count(self):
        assert "132 Svelte" in self.content

    def test_redteam_in_structure(self):
        assert "redteam/" in self.content

    def test_e2e_in_structure(self):
        assert "tests/e2e/" in self.content

    def test_security_layers(self):
        assert "Layer 1" in self.content
        assert "Layer 6" in self.content


# ---------------------------------------------------------------------------
# Part 3: SECURITY.md version and new sections
# ---------------------------------------------------------------------------

class TestPart03SecurityVersion:
    """SECURITY.md reflects v3.2.0 with new sections."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read(_PROJECT / "SECURITY.md")

    def test_title_version(self):
        assert "v3.2.0" in self.content.split("\n")[0]

    def test_no_rc_in_title(self):
        first_line = self.content.split("\n")[0]
        assert "rc" not in first_line

    def test_multi_user_section(self):
        assert "## Multi-User Data Isolation" in self.content

    def test_rag_injection_section(self):
        assert "## RAG Prompt Injection Defense" in self.content

    def test_red_team_section(self):
        assert "## Red Team Engine" in self.content

    def test_rbac_in_auth_section(self):
        assert "RBAC" in self.content

    def test_plugin_subprocess_in_plugin_section(self):
        assert "subprocess" in self.content.lower()


# ---------------------------------------------------------------------------
# Part 4: SECURITY.md Known Limitations resolved
# ---------------------------------------------------------------------------

class TestPart04SecurityLimitations:
    """Known Limitations marks resolved items."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read(_PROJECT / "SECURITY.md")

    def _limitation_line(self, tag):
        for line in self.content.split("\n"):
            if line.startswith(f"**{tag}"):
                return line
        return ""

    def test_l5_resolved(self):
        line = self._limitation_line("L5")
        assert "resolved" in line.lower()

    def test_l8_resolved(self):
        line = self._limitation_line("L8")
        assert "resolved" in line.lower()

    def test_l9_resolved(self):
        line = self._limitation_line("L9")
        assert "resolved" in line.lower()

    def test_l12_resolved(self):
        line = self._limitation_line("L12")
        assert "resolved" in line.lower()


# ---------------------------------------------------------------------------
# Part 5: SECURITY.md version history completeness
# ---------------------------------------------------------------------------

class TestPart05SecurityHistory:
    """Version history includes all S138-S150 entries."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read(_PROJECT / "SECURITY.md")

    def test_s138_entry(self):
        assert "S138" in self.content

    def test_s142_entry(self):
        assert "S142" in self.content

    def test_s143_entry(self):
        assert "S143" in self.content

    def test_s144_entry(self):
        assert "S144" in self.content

    def test_s145_entry(self):
        assert "S145" in self.content

    def test_s146_entry(self):
        assert "S146" in self.content

    def test_s147_entry(self):
        assert "S147" in self.content

    def test_s148_entry(self):
        assert "S148" in self.content

    def test_s150_entry(self):
        assert "S150" in self.content


# ---------------------------------------------------------------------------
# Part 6: CHANGELOG.md has v3.2.0 entry
# ---------------------------------------------------------------------------

class TestPart06ChangelogEntry:
    """CHANGELOG.md has a v3.2.0 entry."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = _read(_PROJECT / "CHANGELOG.md")

    def test_v320_header(self):
        assert "## v3.2.0" in self.content

    def test_v320_date(self):
        assert "2026-03-20" in self.content

    def test_v320_session_range(self):
        assert "S138-S150" in self.content

    def test_v320_before_v311(self):
        idx_320 = self.content.index("## v3.2.0")
        idx_311 = self.content.index("## v3.1.1")
        assert idx_320 < idx_311


# ---------------------------------------------------------------------------
# Part 7: CHANGELOG.md v3.2.0 categories present
# ---------------------------------------------------------------------------

class TestPart07ChangelogCategories:
    """CHANGELOG.md v3.2.0 entry has required category subsections."""

    @pytest.fixture(autouse=True)
    def _load(self):
        content = _read(_PROJECT / "CHANGELOG.md")
        idx_start = content.index("## v3.2.0")
        idx_end = content.index("## v3.1.1")
        self.section = content[idx_start:idx_end]

    def test_security_subsection(self):
        assert "### Security" in self.section

    def test_red_team_subsection(self):
        assert "### Red Team Engine" in self.section

    def test_quality_subsection(self):
        assert "### Quality Infrastructure" in self.section

    def test_testing_subsection(self):
        assert "### Testing" in self.section

    def test_documentation_subsection(self):
        assert "### Documentation" in self.section

    def test_new_modules_subsection(self):
        assert "### New Modules" in self.section

    def test_new_api_endpoints_subsection(self):
        assert "### New API Endpoints" in self.section

    def test_version_progression_subsection(self):
        assert "### Version Progression" in self.section

    def test_security_tag_count(self):
        count = self.section.count("[SECURITY]")
        assert count >= 15, f"Expected 15+ [SECURITY] tags, got {count}"


# ---------------------------------------------------------------------------
# Part 8: docs/API_REFERENCE.md exists with endpoint sections
# ---------------------------------------------------------------------------

class TestPart08ApiReference:
    """docs/API_REFERENCE.md exists and has required sections."""

    def test_file_exists(self):
        assert (_DOCS / "API_REFERENCE.md").is_file()

    @pytest.fixture(autouse=True)
    def _load(self):
        path = _DOCS / "API_REFERENCE.md"
        if path.is_file():
            self.content = _read(path)
        else:
            self.content = ""

    def test_red_team_section(self):
        assert "## Red Team Engine" in self.content

    def test_audit_chain_section(self):
        assert "## Audit Chain External Anchor" in self.content

    def test_startup_checks_section(self):
        assert "## Startup Security Checklist" in self.content

    def test_rag_injection_section(self):
        assert "## RAG Prompt Injection Defense" in self.content

    def test_user_management_section(self):
        assert "## User Management and RBAC" in self.content


# ---------------------------------------------------------------------------
# Part 9: API_REFERENCE.md endpoint completeness
# ---------------------------------------------------------------------------

class TestPart09ApiEndpoints:
    """API_REFERENCE.md documents all required endpoints."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = _DOCS / "API_REFERENCE.md"
        self.content = _read(path) if path.is_file() else ""

    def test_redteam_run(self):
        assert "/api/security/redteam/run" in self.content

    def test_redteam_status(self):
        assert "/api/security/redteam/status" in self.content

    def test_redteam_results(self):
        assert "/api/security/redteam/results" in self.content

    def test_redteam_report(self):
        assert "/api/security/redteam/report" in self.content

    def test_audit_export_qr(self):
        assert "/api/security/audit/export-qr" in self.content

    def test_audit_export_anchor(self):
        assert "/api/security/audit/export-anchor" in self.content

    def test_audit_anchor_text(self):
        assert "/api/security/audit/anchor-text" in self.content

    def test_audit_verify_anchor(self):
        assert "/api/security/audit/verify-anchor" in self.content

    def test_startup_checks(self):
        assert "/api/security/startup-checks" in self.content

    def test_sanitize_preview(self):
        assert "/api/rag/injection-defense/sanitize-preview" in self.content

    def test_approve_chunks(self):
        assert "/api/rag/injection-defense/approve" in self.content

    def test_injection_audit(self):
        assert "/api/rag/injection-defense/audit" in self.content

    def test_injection_config(self):
        assert "/api/rag/injection-defense/config" in self.content

    def test_user_export(self):
        assert "/api/users/{user_id}/export" in self.content

    def test_user_delete(self):
        assert "/api/users/{user_id}/data" in self.content

    def test_admin_audit(self):
        assert "/api/admin/audit" in self.content


# ---------------------------------------------------------------------------
# Part 10: Version is 3.2.0 (no rc suffix)
# ---------------------------------------------------------------------------

class TestPart10Version:
    """Version is exactly 3.2.0 with no rc suffix."""

    def test_version_value(self):
        assert _load_version() == "3.2.0"

    def test_version_file_content(self):
        content = _read(_PROJECT / "opti_oignon" / "__version__.py")
        assert '"3.2.0"' in content

    def test_no_rc_suffix(self):
        version = _load_version()
        assert "rc" not in version

    def test_no_dev_suffix(self):
        version = _load_version()
        assert "dev" not in version


# ---------------------------------------------------------------------------
# Part 11: Version consistency across test files
# ---------------------------------------------------------------------------

class TestPart11VersionConsistency:
    """Updated test files reference 3.2.0, not rc5."""

    def test_s147_test_version(self):
        content = _read(_PROJECT / "tests" / "test_s147_redteam_generation.py")
        assert '"3.2.0"' in content
        assert '"3.2.0-rc5"' not in content

    def test_s148_test_version(self):
        content = _read(_PROJECT / "tests" / "test_s148_redteam_runner.py")
        assert '"3.2.0"' in content
        assert '"3.2.0-rc5"' not in content

    def test_s149_test_version(self):
        content = _read(_PROJECT / "tests" / "test_s149_e2e_setup.py")
        assert '"3.2.0"' in content
        assert '"3.2.0-rc5"' not in content


# ---------------------------------------------------------------------------
# Part 12: E2E mock version updated
# ---------------------------------------------------------------------------

class TestPart12E2EMockVersion:
    """E2E mock data reflects 3.2.0."""

    def test_mock_data_version(self):
        mock_path = _PROJECT / "tests" / "e2e" / "mocks" / "data.ts"
        content = _read(mock_path)
        assert "'3.2.0'" in content
        assert "'3.2.0-rc5'" not in content
