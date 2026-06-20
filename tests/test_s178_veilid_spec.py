#!/usr/bin/env python3
"""Tests for S178 Goal 6 -- VEILID_SPEC.md and the Veilid docs.

Validates the Theme 4 cartography document and the MkDocs page:

- The spec exists, is substantial, and is correctly titled, with all twelve
  numbered sections.
- The cartography invariant holds both ways: every module on disk under
  opti_oignon/veilid/ is registered in the spec, and every registry entry is a
  real module.
- The security model is stated: the Bulbe boundary, Kerckhoffs, private routing
  and end-to-end encryption, the Daily-only nature, and the no-outbound boundary
  that retired webhooks.
- ODYSSEUS_SPEC and FRONTEND_REDESIGN_SPEC stay green: they still exist, and the
  veilid package is registered in VEILID_SPEC, not folded into ODYSSEUS_SPEC.
- The MkDocs page exists, has an H1, and is registered in the nav.
- Document conventions: no emojis, no decorative uppercase banners, no
  equals-sign separator bars.

File-content only, so the suite collects without the backend.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "VEILID_SPEC.md"
VEILID_PKG = ROOT / "opti_oignon" / "veilid"
ODYSSEUS_SPEC = ROOT / "ODYSSEUS_SPEC.md"
FRONTEND_SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
MKDOCS = ROOT / "mkdocs.yml"
DOCS_PAGE = ROOT / "docs" / "sync" / "veilid-sync.md"

# The veilid package registry the cartography invariant requires named in the spec.
PLANNED_VEILID_MODULES = [
    "opti_oignon/veilid/__init__.py",
    "opti_oignon/veilid/guard.py",
    "opti_oignon/veilid/node.py",
    "opti_oignon/veilid/client.py",
]


def _read_spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def _norm() -> str:
    return re.sub(r"\s+", " ", _read_spec()).lower()


def _list_package_files() -> list[str]:
    names: list[str] = []
    if VEILID_PKG.is_dir():
        for p in VEILID_PKG.rglob("*.py"):
            names.append(p.relative_to(ROOT).as_posix())
    return names


# Spec file and structure


class TestSpecFile:
    def test_exists(self):
        assert SPEC.exists(), f"VEILID_SPEC.md not found at {SPEC}"

    def test_non_trivial(self):
        text = _read_spec()
        assert len(text) > 8000, "Spec is unexpectedly short for a theme spec"
        assert len(text.splitlines()) > 200, "Spec has too few lines"

    def test_title(self):
        assert _read_spec().startswith("# VEILID_SPEC")

    def test_native_package_name(self):
        text = _read_spec()
        assert "opti_oignon/veilid/" in text
        assert "opti_oignon/veilid_sync" not in text


class TestSpecSections:
    REQUIRED = [f"## {i}. " for i in range(1, 13)]

    def test_all_sections_present(self):
        text = _read_spec()
        missing = [h for h in self.REQUIRED if h not in text]
        assert not missing, f"Missing sections: {missing}"


# Cartography invariant


class TestCartography:
    def test_registry_listed(self):
        text = _read_spec()
        missing = [m for m in PLANNED_VEILID_MODULES if m not in text]
        assert not missing, f"Veilid modules not registered in spec: {missing}"

    def test_no_unregistered_module_on_disk(self):
        text = _read_spec()
        on_disk = _list_package_files()
        unregistered = [p for p in on_disk if p not in text and Path(p).name not in text]
        assert not unregistered, f"On-disk modules not registered: {unregistered}"

    def test_packaging_script_registered(self):
        assert "scripts/fetch_veilid_server.py" in _read_spec()


# Security model


class TestSecurityModel:
    def test_bulbe_boundary(self):
        text = _norm()
        assert "bulbe" in text
        assert "binding layer" in text
        assert "fail-secure" in text

    def test_kerckhoffs(self):
        assert "kerckhoffs" in _norm()

    def test_private_routing_and_e2e(self):
        text = _norm()
        assert "private rout" in text
        assert "end-to-end" in text

    def test_daily_only(self):
        text = _norm()
        assert "daily-only" in text or "daily-mode capability" in text

    def test_no_outbound_channel_like_webhooks(self):
        text = _norm()
        assert "webhook" in text


# Cross-specs stay green


class TestCrossSpecsGreen:
    def test_other_specs_exist(self):
        assert ODYSSEUS_SPEC.exists()
        assert FRONTEND_SPEC.exists()

    def test_veilid_not_folded_into_odysseus(self):
        # The veilid package is registered in its own spec, not ODYSSEUS.
        odysseus = ODYSSEUS_SPEC.read_text(encoding="utf-8")
        assert "opti_oignon/veilid/" not in odysseus

    def test_odysseus_registry_still_named(self):
        # A light guard that the ODYSSEUS cartography list is intact.
        odysseus = ODYSSEUS_SPEC.read_text(encoding="utf-8")
        assert "opti_oignon/agent/skills.py" in odysseus
        assert "opti_oignon/memory/" in odysseus


# MkDocs page


class TestDocsPage:
    def test_page_exists(self):
        assert DOCS_PAGE.exists(), f"docs page not found at {DOCS_PAGE}"

    def test_page_has_h1(self):
        first = DOCS_PAGE.read_text(encoding="utf-8").lstrip().splitlines()[0]
        assert first.startswith("# "), "docs page must start with an H1"

    def test_page_in_nav(self):
        nav = MKDOCS.read_text(encoding="utf-8")
        assert "sync/veilid-sync.md" in nav, "page not registered in mkdocs nav"

    def test_nav_section_named(self):
        assert "Sync (Veilid)" in MKDOCS.read_text(encoding="utf-8")

    def test_page_mentions_bulbe_and_install(self):
        text = DOCS_PAGE.read_text(encoding="utf-8").lower()
        assert "bulbe" in text
        assert "opti-oignon[veilid]" in text


# Document conventions (mirrors tests/test_s172_odysseus_spec.py)


class TestConventions:
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
        assert m is None, f"Emoji found at {m.start()}: {m.group()!r}" if m else ""

    def test_no_decorative_uppercase_section_lines(self):
        for line in _read_spec().splitlines():
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
        for line in _read_spec().splitlines():
            if re.fullmatch(r"=+", line.strip()) and len(line.strip()) >= 20:
                pytest.fail(f"Pure '=' separator banner found: {line[:40]!r}")
