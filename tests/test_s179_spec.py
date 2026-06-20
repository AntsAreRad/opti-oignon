#!/usr/bin/env python3
"""Tests for S179 Goal 6 -- the cartography and docs (Theme 4 / Veilid Sync).

Validates that the S179 protocol is recorded where the bloc's cartography requires:

- The four S179 modules (records, reconcile, change_feed, protocol) are registered
  in VEILID_SPEC.md, and every module on disk under opti_oignon/veilid/ is
  registered (the on-disk invariant, re-asserted for S179).
- VEILID_SPEC section 6 has been promoted from scoped to implemented: it names the
  protocol and the four modules, and the old "S178 does not implement" wording is
  gone; the twelve-section structure is intact.
- The MkDocs page describes the protocol model (convergent, last-writer-wins,
  watermark, tombstone, the pull exchange) while still mentioning the Bulbe boundary
  and the optional install (keep-green from S178).
- ODYSSEUS_SPEC and FRONTEND_REDESIGN_SPEC stay green: they still exist and the
  veilid package is not folded into ODYSSEUS.
- The edits introduced no emojis, no equals-sign separator bars, and no decorative
  uppercase banners.

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
DOCS_PAGE = ROOT / "docs" / "sync" / "veilid-sync.md"

S179_MODULES = [
    "opti_oignon/veilid/records.py",
    "opti_oignon/veilid/reconcile.py",
    "opti_oignon/veilid/change_feed.py",
    "opti_oignon/veilid/protocol.py",
]


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def _docs() -> str:
    return DOCS_PAGE.read_text(encoding="utf-8")


def _section6() -> str:
    text = _spec()
    start = text.index("## 6. ")
    end = text.index("## 7. ", start)
    return text[start:end]


def _list_package_files() -> list[str]:
    names: list[str] = []
    if VEILID_PKG.is_dir():
        for p in VEILID_PKG.rglob("*.py"):
            names.append(p.relative_to(ROOT).as_posix())
    return names


# Cartography registration


class TestCartography:
    def test_s179_modules_registered(self):
        text = _spec()
        missing = [m for m in S179_MODULES if m not in text]
        assert not missing, f"S179 modules not registered in spec: {missing}"

    def test_all_on_disk_modules_registered(self):
        text = _spec()
        on_disk = _list_package_files()
        unregistered = [
            p for p in on_disk if p not in text and Path(p).name not in text
        ]
        assert not unregistered, f"On-disk modules not registered: {unregistered}"

    def test_new_modules_present_on_disk(self):
        for m in S179_MODULES:
            assert (ROOT / m).exists(), f"expected module missing: {m}"


# Section 6 promoted to implemented


class TestSection6Implemented:
    def test_section6_heading_present(self):
        assert "## 6. " in _spec()

    def test_section6_says_implemented(self):
        s6 = _section6().lower()
        assert "implemented" in s6
        assert "protocol" in s6

    def test_old_scoped_wording_gone(self):
        assert "does not implement a sync protocol" not in _spec().lower()

    def test_section6_names_the_modules(self):
        s6 = _section6()
        for name in ("records.py", "reconcile.py", "change_feed.py", "protocol.py"):
            assert name in s6, f"section 6 should name {name}"

    def test_twelve_sections_intact(self):
        text = _spec()
        missing = [f"## {i}. " for i in range(1, 13) if f"## {i}. " not in text]
        assert not missing, f"missing sections: {missing}"


# Docs page describes the protocol model


class TestDocsProtocolModel:
    def test_page_exists_and_has_h1(self):
        first = _docs().lstrip().splitlines()[0]
        assert first.startswith("# ")

    def test_mentions_protocol_model(self):
        text = _docs().lower()
        for token in ("convergent", "last-writer-wins", "watermark", "tombstone", "pull"):
            assert token in text, f"docs should mention {token!r}"

    def test_keeps_bulbe_and_install(self):
        text = _docs().lower()
        assert "bulbe" in text
        assert "opti-oignon[veilid]" in text


# Cross-specs stay green


class TestCrossSpecsGreen:
    def test_other_specs_exist(self):
        assert ODYSSEUS_SPEC.exists()
        assert FRONTEND_SPEC.exists()

    def test_veilid_not_folded_into_odysseus(self):
        odysseus = ODYSSEUS_SPEC.read_text(encoding="utf-8")
        assert "opti_oignon/veilid/" not in odysseus

    def test_odysseus_registry_still_named(self):
        odysseus = ODYSSEUS_SPEC.read_text(encoding="utf-8")
        assert "opti_oignon/agent/skills.py" in odysseus
        assert "opti_oignon/memory/" in odysseus


# Conventions hold after the edits


class TestConventions:
    EMOJI = re.compile(
        "[\U0001F300-\U0001FAFF"
        "\U00002600-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF]"
    )

    def test_no_emojis(self):
        for text, label in ((_spec(), "spec"), (_docs(), "docs")):
            m = self.EMOJI.search(text)
            assert m is None, f"emoji in {label} at {m.start()}" if m else ""

    def test_no_equals_bars(self):
        for text in (_spec(), _docs()):
            for line in text.splitlines():
                if re.fullmatch(r"=+", line.strip()) and len(line.strip()) >= 20:
                    pytest.fail(f"'=' banner found: {line[:40]!r}")

    def test_no_uppercase_banners_in_spec(self):
        for line in _spec().splitlines():
            stripped = line.strip()
            if (
                len(stripped) >= 8
                and stripped == stripped.upper()
                and re.fullmatch(r"[A-Z][A-Z\s]+", stripped)
                and not stripped.startswith(("|", "-", "`", "#"))
            ):
                pytest.fail(f"uppercase banner found: {stripped!r}")
