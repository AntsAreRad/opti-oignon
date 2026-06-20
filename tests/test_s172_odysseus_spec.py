#!/usr/bin/env python3
"""
Tests for S172 -- Odysseus Analysis and Specification.

Validates the ODYSSEUS_SPEC.md document produced during S172, the
cartography invariant for Theme 3 (S172-S177). This is a pure-analysis
session: no production code is tested here. These tests assert the
specification's structure, the borrow-versus-reject decisions, the security
deltas, the cartography invariant, and the project's document conventions
(no emojis, no decorative uppercase banners, no equals-sign separator bars).

Mirrors tests/test_s165_redesign_spec.py. File-content only, so the suite
collects without the backend runtime.
"""

import re
from pathlib import Path

import pytest

# Paths

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "ODYSSEUS_SPEC.md"
AGENT_PKG = ROOT / "opti_oignon" / "agent"
MEMORY_PKG = ROOT / "opti_oignon" / "memory"

# The forward-declared module registry from Section 10 of the spec. The
# cartography invariant requires each of these to be named in the spec.
PLANNED_AGENT_MODULES = [
    "opti_oignon/agent/__init__.py",
    "opti_oignon/agent/loop.py",
    "opti_oignon/agent/dispatch.py",
    "opti_oignon/agent/tool_parsing.py",
    "opti_oignon/agent/tools.py",
    "opti_oignon/agent/skills.py",
    "opti_oignon/agent/allowlists.py",
    "opti_oignon/agent/untrusted_context.py",
    "opti_oignon/agent/teacher.py",
    "opti_oignon/agent/config.yaml",
]
PLANNED_MEMORY_MODULES = [
    "opti_oignon/memory/__init__.py",
    "opti_oignon/memory/canonical_store.py",
    "opti_oignon/memory/vector_store.py",
    "opti_oignon/memory/dedup.py",
    "opti_oignon/memory/retrieval.py",
    "opti_oignon/memory/extraction.py",
    "opti_oignon/memory/curation.py",
]

# Helpers


def _read_spec() -> str:
    with open(SPEC, encoding="utf-8") as f:
        return f.read()


def _norm() -> str:
    """Lowercased, whitespace-collapsed spec text for prose phrase checks.

    Markdown prose wraps across lines, so a multi-word phrase may straddle a
    line break in the raw file. Collapse runs of whitespace to single spaces
    before substring matching.
    """
    return re.sub(r"\s+", " ", _read_spec()).lower()


def _list_package_files() -> list[str]:
    """Files on disk under the planned agent/ and memory/ packages.

    Returned as POSIX paths relative to ROOT. At S172 both packages are
    absent, so this is empty and the on-disk cartography check is vacuous;
    it becomes meaningful from S173 onward.
    """
    names: list[str] = []
    for pkg in (AGENT_PKG, MEMORY_PKG):
        if pkg.exists() and pkg.is_dir():
            for pattern in ("*.py", "*.yaml", "*.yml"):
                for p in pkg.rglob(pattern):
                    names.append(p.relative_to(ROOT).as_posix())
    return names


# Test class 1: file presence and basic structure


class TestSpecFile:
    """The spec exists, is substantial, and is correctly titled."""

    def test_spec_file_exists(self):
        assert SPEC.exists(), f"ODYSSEUS_SPEC.md not found at {SPEC}"

    def test_spec_file_non_trivial(self):
        text = _read_spec()
        assert len(text) > 8000, "Spec is unexpectedly short for an analysis session"
        assert len(text.splitlines()) > 200, "Spec has too few lines"

    def test_spec_title(self):
        text = _read_spec()
        assert text.startswith("# ODYSSEUS_SPEC"), "Spec must start with the title heading"

    def test_odysseus_is_reference_not_subsystem(self):
        """Native module names only; no opti_oignon/odysseus package branding."""
        text = _read_spec()
        assert "opti_oignon/agent/" in text
        assert "opti_oignon/memory/" in text
        assert "opti_oignon/odysseus" not in text


# Test class 2: required sections


class TestSpecSections:
    """All twelve numbered sections are present."""

    REQUIRED = [f"## {i}. " for i in range(1, 13)]

    def test_all_required_sections_present(self):
        text = _read_spec()
        missing = [h for h in self.REQUIRED if h not in text]
        assert not missing, f"Missing sections: {missing}"


# Test class 3: executive summary


class TestExecutiveSummary:
    """The constraint and the six-session plan are stated."""

    def test_borrow_not_execution_model_constraint(self):
        text = _norm()
        assert "borrow the patterns" in text
        assert "never the execution model" in text
        assert "admin console" in text

    def test_six_sessions_named_with_bump(self):
        text = _read_spec()
        for s in ("S172", "S173", "S174", "S175", "S176", "S177"):
            assert s in text, f"Session {s} not named"
        assert "3.5.0" in text, "Version bump to 3.5.0 must be stated"

    def test_session_test_estimates_present(self):
        """The implementation-plan table carries a per-session test estimate."""
        text = _read_spec()
        for estimate in ("70", "65", "60", "15"):
            assert estimate in text, f"Expected test estimate {estimate} missing"


# Test class 4: pattern mapping, taken vs rejected


class TestPatternMapping:
    """The borrow-versus-reject decisions are explicit."""

    def test_rejected_patterns(self):
        text = _norm()
        # Host execution of agent tools is rejected.
        assert "host execution" in text
        assert "rejected" in text
        # Webhooks are rejected.
        assert "webhook" in text
        # The RBAC-only gate is rejected in favour of sandbox + Daily/Bulbe.
        assert "non_admin_blocked_tools" in text or "rbac" in text

    def test_taken_patterns(self):
        text = _norm()
        assert "taken" in text
        # The loop, the dual dispatch, the wrapping, the memory mechanics,
        # and the skills format are all carried over.
        assert "dual tool dispatch" in text or "dual dispatch" in text
        assert "untrusted-context wrapping" in text or "untrusted context" in text


# Test class 5: memory architecture


class TestMemoryArchitecture:
    """Two-tier store, dedup thresholds, categories, curation, S66."""

    def test_two_tier_and_collections(self):
        text = _read_spec()
        assert "SQLite WAL" in text
        assert "oo_memories" in text
        assert "distinct from the rag" in _norm()

    def test_categories_listed(self):
        low = _read_spec().lower()
        for cat in ("identity", "preference", "fact", "contact", "project", "goal"):
            assert cat in low, f"Memory category {cat} missing"

    def test_double_dedup_thresholds(self):
        text = _read_spec()
        assert "0.6" in text, "Jaccard threshold 0.6 missing"
        assert "0.92" in text, "Cosine near-duplicate threshold 0.92 missing"
        assert "Jaccard" in text

    def test_curation_sidecar_and_s66(self):
        text = _read_spec()
        assert "memory_tidy_state.json" in text, "Curation fingerprint sidecar missing"
        assert "S66" in text, "S66 dual-layer alignment must be referenced"


# Test class 6: agent loop


class TestAgentLoop:
    """Sandbox dispatch invariant and loop mechanics."""

    def test_sandbox_dispatch_invariant(self):
        text = _read_spec()
        assert "S73/S74" in text
        low = _norm()
        assert "never the host" in low
        assert "copy-out only after explicit human approval" in low or "copy-out after" in low

    def test_loop_mechanics(self):
        text = _read_spec()
        assert "MAX_AGENT_ROUNDS" in text or "round cap" in _norm()
        assert "frozenset" in text
        low = _norm()
        assert "daily" in low and "bulbe" in low
        assert "teacher" in low


# Test class 7: skills


class TestSkills:
    """On-disk SKILL.md registry, approval-gated manage_skills, usage sidecar."""

    def test_skills_registry_path(self):
        text = _read_spec()
        assert "data/skills/" in text
        assert "SKILL.md" in text
        assert "_usage.json" in text

    def test_manage_skills_approval_gated(self):
        text = _read_spec()
        assert "manage_skills" in text
        assert "approval" in text.lower()


# Test class 8: threat-model delta


class TestThreatModel:
    """The five agent-specific threats and Kerckhoffs."""

    def test_threat_items_present(self):
        low = _norm()
        assert "prompt injection" in low or "prompt-injection" in low
        assert "poisoning" in low
        assert "autonomy" in low

    def test_kerckhoffs(self):
        text = _read_spec()
        assert "Kerckhoffs" in text
        assert "obscurity" in text.lower()


# Test class 9: cartography invariant


class TestCartography:
    """The registry is complete and matches what is on disk."""

    def test_agent_registry_listed(self):
        text = _read_spec()
        missing = [m for m in PLANNED_AGENT_MODULES if m not in text]
        assert not missing, f"Agent modules not registered in spec: {missing}"

    def test_memory_registry_listed(self):
        text = _read_spec()
        missing = [m for m in PLANNED_MEMORY_MODULES if m not in text]
        assert not missing, f"Memory modules not registered in spec: {missing}"

    def test_no_unregistered_module_on_disk(self):
        """Every file under agent/ or memory/ must be named in the spec.

        Vacuous at S172 (packages absent); meaningful from S173.
        """
        text = _read_spec()
        on_disk = _list_package_files()
        unregistered = [p for p in on_disk if p not in text and Path(p).name not in text]
        assert not unregistered, (
            f"{len(unregistered)} on-disk modules not registered in spec: "
            f"{unregistered[:10]}"
        )


# Test class 10: document conventions


class TestConventions:
    """No emojis; no decorative MAJ headers; no equals-sign separator bars."""

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
            f"Emoji found at position {m.start()}: {m.group()!r}; spec must be emoji-free"
        )

    def test_no_decorative_uppercase_section_lines(self):
        text = _read_spec()
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
        text = _read_spec()
        for line in text.splitlines():
            if re.fullmatch(r"=+", line.strip()) and len(line.strip()) >= 20:
                pytest.fail(f"Pure '=' separator banner found: {line[:40]!r}")
