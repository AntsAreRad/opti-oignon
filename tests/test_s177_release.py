#!/usr/bin/env python3
"""Tests for S177 -- the 3.5.0 version bump, CHANGELOG, README, and docs (Goal 6).

These re-assert the version surface after the 3.4.0 -> 3.5.0 bump that closes
Theme 3. The superseded 3.4.0 assertions in test_s170 and test_s171 stay
deselected (never edited or deleted); this file re-asserts the new state. The
S164 README v3.3.0 tests keep passing because the historical v3.3.0 references
are retained alongside the new v3.5.0 headline.

- __version__.py and pyproject.toml are the bare final 3.5.0.
- The CHANGELOG top entry is the v3.5.0 (S177) product-surface entry; the v3.4.0,
  v3.4.0-rc, and v3.3.0 entries are retained.
- The README headline names v3.5.0 and gains a Theme 3 features section, while
  the historical v3.3.0 references (description body, features section) are kept.
- MkDocs pages exist for the agent, memory, and skills surfaces and are in nav.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(*parts) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def _version_from_file() -> str:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py"))
    assert m, "no __version__ assignment found"
    return m.group(1)


# Version surface


class TestVersionBump:
    def test_version_file_is_350(self):
        assert _version_from_file() == "3.5.0"

    def test_version_file_contains_350(self):
        assert '"3.5.0"' in _read("opti_oignon", "__version__.py")

    def test_version_is_bare_form(self):
        assert re.match(r"^\d+\.\d+\.\d+$", _version_from_file())

    def test_no_rc_suffix(self):
        assert "-rc" not in _read("opti_oignon", "__version__.py")

    def test_pyproject_version_is_350(self):
        assert 'version = "3.5.0"' in _read("pyproject.toml")

    def test_pyproject_consistent_with_version_file(self):
        assert f'version = "{_version_from_file()}"' in _read("pyproject.toml")


# CHANGELOG


class TestChangelog:
    def setup_method(self):
        self.c = _read("CHANGELOG.md")

    def test_v350_entry_present(self):
        assert "## v3.5.0 -- 2026-06-02 (S177)" in self.c

    def test_top_entry_is_350(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == "3.5.0"

    def test_entry_is_product_surface_for_theme3(self):
        entry = self.c.split("## v3.5.0")[1].split("## v3.4.0")[0]
        for term in ("agent", "memory", "skill"):
            assert term.lower() in entry.lower()

    def test_security_section_present(self):
        entry = self.c.split("## v3.5.0")[1].split("## v3.4.0")[0]
        assert "[SECURITY]" in entry

    def test_prior_entries_retained(self):
        assert "## v3.4.0 -- 2026-06-01 (S171)" in self.c
        assert "## v3.4.0-rc -- 2026-06-01 (S170)" in self.c
        assert "## v3.3.0" in self.c


# README


class TestReadme:
    def setup_method(self):
        self.r = _read("README.md")

    def test_headline_names_350(self):
        line = [l for l in self.r.split("\n") if "Opti-Oignon v" in l][0]
        assert "v3.5.0" in line

    def test_describes_theme3(self):
        line = [l for l in self.r.split("\n") if "Opti-Oignon v3.5.0" in l][0]
        assert "sandbox" in line.lower()
        assert "skill" in line.lower()

    def test_features_v350_section(self):
        assert "## Features Added in v3.5.0" in self.r

    def test_v330_references_retained(self):
        # Historical references kept so the S164 v3.3.0 README tests stay green.
        assert "v3.3.0" in self.r
        assert "## Features Added in v3.3.0" in self.r


# MkDocs pages


class TestDocs:
    PAGES = (
        "docs/agent/sandboxed-agent.md",
        "docs/agent/evolving-memory.md",
        "docs/agent/evolving-skills.md",
    )

    def test_pages_exist(self):
        for page in self.PAGES:
            assert ROOT.joinpath(page).exists(), f"missing doc page: {page}"

    def test_pages_in_nav(self):
        nav = _read("mkdocs.yml")
        for page in ("agent/sandboxed-agent.md", "agent/evolving-memory.md", "agent/evolving-skills.md"):
            assert page in nav, f"{page} not registered in mkdocs nav"

    def test_agent_page_covers_sandbox_and_approval(self):
        text = _read("docs", "agent", "sandboxed-agent.md").lower()
        assert "sandbox" in text and "approval" in text and "untrusted" in text

    def test_skills_page_covers_approval_gate(self):
        text = _read("docs", "agent", "evolving-skills.md").lower()
        assert "manage_skills" in text and "approv" in text and "draft" in text

    def test_memory_page_covers_two_tier(self):
        text = _read("docs", "agent", "evolving-memory.md").lower()
        assert "working" in text and "archive" in text
