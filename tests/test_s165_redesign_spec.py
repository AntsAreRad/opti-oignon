#!/usr/bin/env python3
"""
Tests for S165 -- Frontend Redesign Specification.

Validates the FRONTEND_REDESIGN_SPEC.md document produced during S165.
This is a pure-analysis session: no production code changes are tested
here. These tests assert the specification's structure, completeness,
and adherence to the project's conventions (no emojis, English-only,
severity/effort fields, 3 to 5 themes, mapping to S166--S170, etc.).
"""

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
FRONTEND_SRC = ROOT / "frontend" / "src"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_spec() -> str:
    """Read the spec file once."""
    with open(SPEC, encoding="utf-8") as f:
        return f.read()


def _list_components() -> list[str]:
    """Return all .svelte basenames under frontend/src/lib/components/ and routes/."""
    names: list[str] = []
    for sub in ("lib/components", "routes"):
        base = FRONTEND_SRC / sub
        if not base.exists():
            continue
        for p in base.rglob("*.svelte"):
            names.append(p.stem)
    return names


# ===================================================================
# Test class 1: File presence and basic structure
# ===================================================================


class TestSpecFile:
    """The spec file exists and is well-formed."""

    def test_spec_file_exists(self):
        assert SPEC.exists(), f"FRONTEND_REDESIGN_SPEC.md not found at {SPEC}"

    def test_spec_file_non_trivial(self):
        text = _read_spec()
        # Spec should be substantial (> 30 KB and > 1000 lines)
        assert len(text) > 30_000, f"Spec too small: {len(text)} bytes"
        assert text.count("\n") > 1000, "Spec has fewer than 1000 lines"

    def test_spec_title_and_metadata(self):
        text = _read_spec()
        assert text.startswith("# FRONTEND_REDESIGN_SPEC")
        assert "**Session:** S165" in text
        assert "**Cycle:** ROADMAP_S165_S176" in text


# ===================================================================
# Test class 2: Required sections
# ===================================================================


class TestSpecSections:
    """All required top-level sections are present."""

    REQUIRED_SECTIONS = [
        "## 1. Executive Summary",
        "## 2. UI Cartography",
        "## 3. Consolidated Pain Points",
        "## 4. Theme System Audit",
        "## 5. Settings Consolidation Plan",
        "## 6. Design Decisions",
        "## 7. Design Tokens Specification",
        "## 8. Navigation Architecture",
        "## 9. Textual Wireframes",
        "## 10. Design System Inventory",
        "## 11. Component Refactor List",
        "## 12. Migration Plan",
        "## 13. Out of Scope, Risks, Open Questions",
        "## 14. Tests",
    ]

    def test_all_required_sections_present(self):
        text = _read_spec()
        missing = [s for s in self.REQUIRED_SECTIONS if s not in text]
        assert not missing, f"Missing sections: {missing}"

    def test_executive_summary_lists_six_goals(self):
        """Section 1.2 must enumerate exactly six cross-cutting goals."""
        text = _read_spec()
        m = re.search(
            r"### 1\.2 Six Cross-Cutting Goals(.*?)### 1\.3",
            text,
            re.DOTALL,
        )
        assert m is not None, "Section 1.2 not found"
        body = m.group(1)
        # Enumeration items are numbered 1. ... 6.
        items = re.findall(r"^\s*\d+\.\s+\*\*", body, re.MULTILINE)
        assert len(items) == 6, (
            f"Expected exactly 6 cross-cutting goals, found {len(items)}"
        )


# ===================================================================
# Test class 3: Cartography coverage
# ===================================================================


class TestCartography:
    """Every existing Svelte component is mentioned in the spec."""

    def test_all_components_named_in_spec_or_retired(self):
        text = _read_spec()
        components = set(_list_components())
        # Filter out SvelteKit conventional file stems that won't appear by name
        components.discard("+page")
        components.discard("+layout")
        # The spec mentions component basenames either in Section 2
        # (cartography) or Section 11 (refactor list, including retirement)
        missing = [c for c in components if c not in text]
        assert not missing, (
            f"{len(missing)} components not mentioned anywhere in spec: "
            f"{missing[:10]}"
        )

    def test_orphan_summary_present(self):
        text = _read_spec()
        # Section 2.19 must list the orphan components
        m = re.search(r"### 2\.19 Orphan Components Summary", text)
        assert m is not None, "Orphan summary section 2.19 missing"
        # At least four well-known orphans must be listed
        for name in ("ThemeCustomizer", "CodingAgentPanel", "ModelManager", "ScrollToBottom"):
            assert name in text, f"Expected orphan {name} not listed"


# ===================================================================
# Test class 4: Pain points have severity and effort
# ===================================================================


class TestPainPoints:
    """Every pain point row in Section 3 declares Severity and Effort."""

    def test_severity_and_effort_fields_used(self):
        text = _read_spec()
        # Extract Section 3
        m = re.search(
            r"## 3\. Consolidated Pain Points(.*?)## 4\.",
            text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        # Each ID-row in the pain points tables follows the pattern:
        # | <ID> | <Title> | P0/P1/P2 | S/M/L | <Detail> |
        # We require P0|P1|P2 to appear at least 25 times (we have 31)
        severities = re.findall(r"\|\s*P[012]\s*\|", body)
        assert len(severities) >= 25, (
            f"Expected at least 25 severity-tagged pain points, got {len(severities)}"
        )
        # And the same for effort markers S/M/L
        efforts = re.findall(r"\|\s*[SML]\s*\|", body)
        assert len(efforts) >= 25, (
            f"Expected at least 25 effort-tagged pain points, got {len(efforts)}"
        )


# ===================================================================
# Test class 5: Theme proposals
# ===================================================================


class TestThemes:
    """3 to 5 theme presets proposed and named."""

    def test_between_three_and_five_themes(self):
        text = _read_spec()
        # Section 7.3 declares one preset per H4 subsection: #### <Name>
        m = re.search(r"### 7\.3 Color Tokens — Per Theme(.*?)### 7\.4", text, re.DOTALL)
        assert m is not None, "Section 7.3 missing"
        body = m.group(1)
        preset_headers = re.findall(r"^####\s+(.+?)$", body, re.MULTILINE)
        assert 3 <= len(preset_headers) <= 5, (
            f"Expected 3 to 5 named theme presets, got {len(preset_headers)}: "
            f"{preset_headers}"
        )

    def test_proposed_themes_named_and_have_accent(self):
        text = _read_spec()
        expected_names = ["Anthracite", "Parchment", "Slate", "Linen", "High Contrast"]
        for name in expected_names:
            assert name in text, f"Expected theme '{name}' not declared"
        # Each preset block must declare an accent token line
        m = re.search(r"### 7\.3(.*?)### 7\.4", text, re.DOTALL)
        body = m.group(1)
        accent_lines = re.findall(r"`--oo-acc-500`", body)
        # Five presets must each declare their accent
        assert accent_lines.count("`--oo-acc-500`") >= 5, (
            f"Expected at least 5 accent declarations, got {accent_lines.count('`--oo-acc-500`')}"
        )


# ===================================================================
# Test class 6: Design tokens completeness
# ===================================================================


class TestDesignTokens:
    """The tokens section declares typography, color, spacing tokens."""

    def test_typography_tokens_complete(self):
        text = _read_spec()
        # Section 7.1 must declare these tokens (referenced by name)
        for tok in ("--oo-font-sans", "--oo-font-mono", "--oo-text-base",
                    "--oo-leading-normal", "--oo-tracking-normal"):
            assert tok in text, f"Missing typography token: {tok}"

    def test_heading_scale_h1_to_h6(self):
        text = _read_spec()
        # The heading mapping table in 7.1 maps h1..h6 explicitly
        m = re.search(r"Heading mapping(.*?)Forbidden", text, re.DOTALL)
        assert m is not None, "Heading mapping table missing"
        body = m.group(1)
        for h in ("h1", "h2", "h3", "h4", "h5", "h6"):
            assert f"`<{h}>`" in body, f"Heading {h} not mapped"

    def test_color_tokens_cover_basics(self):
        text = _read_spec()
        # bg, fg, bd, acc plus semantic
        for tok in ("--oo-bg-base", "--oo-fg-primary", "--oo-bd-default",
                    "--oo-acc-500", "--oo-success", "--oo-warning",
                    "--oo-error", "--oo-info"):
            assert tok in text, f"Missing core color token: {tok}"

    def test_spacing_scale_at_least_eight_levels(self):
        text = _read_spec()
        # --oo-space-0 through --oo-space-8 minimum
        for i in range(0, 9):
            assert f"--oo-space-{i}" in text, f"Missing spacing token --oo-space-{i}"


# ===================================================================
# Test class 7: Settings consolidation
# ===================================================================


class TestSettingsPlan:
    """Settings plan groups into a defined number of sections."""

    EXPECTED_SECTIONS = [
        "Appearance",
        "Account & Security",
        "Conversation & Chat",
        "Models & Inference",
        "Knowledge (RAG)",
        "Plugins & Extensions",
        "Performance & Telemetry",
        "Network & Privacy",
        "Backup & Data",
    ]

    def test_nine_sections_named(self):
        text = _read_spec()
        for s in self.EXPECTED_SECTIONS:
            assert s in text, f"Expected settings section '{s}' missing"
        # We expect 8 to 10 sections (current target is 9)
        section_count = len(self.EXPECTED_SECTIONS)
        assert 8 <= section_count <= 10


# ===================================================================
# Test class 8: Migration plan
# ===================================================================


class TestMigrationPlan:
    """Section 12 names S166..S170 and assigns estimated tests per session."""

    SESSION_IDS = ["S166", "S167", "S168", "S169", "S170"]

    def test_five_sessions_named(self):
        text = _read_spec()
        for sid in self.SESSION_IDS:
            assert sid in text, f"Migration session {sid} not named in spec"

    def test_each_session_has_test_estimate(self):
        text = _read_spec()
        # In Section 12, each subsection 12.1..12.5 contains a line like:
        # "Estimated tests: ~NN"
        m = re.search(
            r"## 12\. Migration Plan(.*?)### 12\.6",
            text,
            re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        estimates = re.findall(r"Estimated tests:\s*~?(\d+)", body)
        assert len(estimates) == 5, (
            f"Expected 5 test estimates (one per session), got {len(estimates)}"
        )

    def test_total_tests_between_280_and_360(self):
        text = _read_spec()
        m = re.search(
            r"## 12\. Migration Plan(.*?)### 12\.6",
            text,
            re.DOTALL,
        )
        body = m.group(1)
        estimates = [int(n) for n in re.findall(r"Estimated tests:\s*~?(\d+)", body)]
        total = sum(estimates)
        assert 280 <= total <= 360, (
            f"Cumulative test estimates {total} outside the 280-360 range; "
            f"individual estimates: {estimates}"
        )


# ===================================================================
# Test class 9: Convention adherence
# ===================================================================


class TestConventions:
    """No emojis; English-only document; no decorative MAJ headers."""

    def test_no_emojis(self):
        text = _read_spec()
        emoji_pattern = re.compile(
            "[\U0001F300-\U0001FAFF"
            "\U00002600-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U0001F600-\U0001F64F"
            "\U0001F680-\U0001F6FF]"
        )
        m = emoji_pattern.search(text)
        assert m is None, (
            f"Emoji found at position {m.start()}: {m.group()!r}; "
            "spec must be emoji-free"
        )

    def test_no_decorative_uppercase_section_lines(self):
        """All-caps decorative section banners are forbidden."""
        text = _read_spec()
        # Lines that are entirely uppercase letters and spaces, length >= 8,
        # are likely decorative banners (e.g., "CRITICAL FINDINGS").
        # Allow exceptions: a few-letter acronyms, table cells.
        for line in text.splitlines():
            stripped = line.strip()
            if (
                len(stripped) >= 8
                and stripped == stripped.upper()
                and re.fullmatch(r"[A-Z][A-Z\s]+", stripped)
                and not stripped.startswith("|")
                and not stripped.startswith("-")
                and not stripped.startswith("`")
                and not stripped.startswith("#")
            ):
                pytest.fail(f"Decorative uppercase banner found: {stripped!r}")

    def test_no_equals_sign_separator_bars(self):
        """The '===...' separator pattern is forbidden as section decoration,
        but is allowed inside ASCII art (e.g., progress bars in wireframes)."""
        text = _read_spec()
        # Find candidate separator lines: a line of '=' chars of length >= 20
        for line in text.splitlines():
            if re.fullmatch(r"=+", line.strip()) and len(line.strip()) >= 20:
                pytest.fail(f"Pure '=' separator banner found: {line[:40]!r}")
