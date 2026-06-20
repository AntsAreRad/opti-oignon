#!/usr/bin/env python3
"""Tests for S182 Goal 5 -- the cartography, spec, and docs (Theme 4 / Veilid Sync).

Validates that the S182 pairing key exchange and the sharing-control panel are
recorded where the bloc's cartography requires, closing Theme 4:

- pairing.py is registered in VEILID_SPEC.md, and every module on disk under
  opti_oignon/veilid/ is registered (the on-disk invariant, re-asserted).
- VEILID_SPEC section 9 has been promoted: it names the pairing module and the
  panel (SyncPanel.svelte), says they are built (S182), and the old "deferred to
  S182" wording for the panel is gone; section 6 lists S182 as built, not deferred;
  the keep-green S180-S181 names survive, and the twelve-section structure is intact.
- The package __init__ exports the pairing surface.
- The sharing-control panel is registered in FRONTEND_REDESIGN_SPEC.md as NEW | S182
  (the frontend cartography invariant).
- The MkDocs page describes the pairing ceremony and the panel while keeping the
  S178-S181 model tokens, the Bulbe boundary, and the optional install.
- ODYSSEUS_SPEC and FRONTEND_REDESIGN_SPEC stay green.
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
INIT = VEILID_PKG / "__init__.py"
ODYSSEUS_SPEC = ROOT / "ODYSSEUS_SPEC.md"
FRONTEND_SPEC = ROOT / "FRONTEND_REDESIGN_SPEC.md"
DOCS_PAGE = ROOT / "docs" / "sync" / "veilid-sync.md"

S182_MODULE = "opti_oignon/veilid/pairing.py"


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def _docs() -> str:
    return DOCS_PAGE.read_text(encoding="utf-8")


def _init() -> str:
    return INIT.read_text(encoding="utf-8")


def _frontend() -> str:
    return FRONTEND_SPEC.read_text(encoding="utf-8")


def _section(n: int) -> str:
    text = _spec()
    start = text.index(f"## {n}. ")
    end = text.index(f"## {n + 1}. ", start)
    return text[start:end]


def _list_package_files() -> list[str]:
    names: list[str] = []
    if VEILID_PKG.is_dir():
        for p in VEILID_PKG.rglob("*.py"):
            names.append(p.relative_to(ROOT).as_posix())
    return names


# Cartography registration


class TestCartography:
    def test_pairing_module_registered(self):
        assert S182_MODULE in _spec()

    def test_pairing_module_present_on_disk(self):
        assert (ROOT / S182_MODULE).exists()

    def test_all_on_disk_modules_registered(self):
        text = _spec()
        on_disk = _list_package_files()
        unregistered = [
            p for p in on_disk if p not in text and Path(p).name not in text
        ]
        assert not unregistered, f"On-disk modules not registered: {unregistered}"

    def test_s178_s181_modules_still_registered(self):
        text = _spec()
        for m in (
            "opti_oignon/veilid/peers.py",
            "opti_oignon/veilid/sync_engine.py",
            "opti_oignon/veilid/transport.py",
            "opti_oignon/veilid/producers.py",
            "opti_oignon/veilid/sync_status.py",
            "opti_oignon/api/routes_sync.py",
        ):
            assert m in text, f"prior registration lost: {m}"


# Section 9 promoted to name the S182 work as built


class TestSection9Promoted:
    def test_section9_heading_present(self):
        assert "## 9. " in _spec()

    def test_section9_names_pairing_module(self):
        assert "pairing.py" in _section(9)

    def test_section9_names_panel(self):
        assert "SyncPanel.svelte" in _section(9)

    def test_section9_says_pairing_and_panel_built_s182(self):
        s9 = _section(9).lower()
        assert "s182" in s9
        assert "built" in s9
        assert "pairing" in s9 and "panel" in s9

    def test_section9_old_panel_deferred_wording_gone(self):
        s9 = _section(9).lower()
        assert "deferred: the pairing and sharing-control panel (s182)" not in s9

    def test_section9_keeps_prior_artifacts(self):
        s9 = _section(9)
        for name in (
            "peers.py",
            "sync_engine.py",
            "routes_sync.py",
            "transport.py",
            "producers.py",
            "sync_status.py",
        ):
            assert name in s9, f"section 9 should keep naming {name}"

    def test_section9_names_pairing_routes(self):
        s9 = _section(9)
        assert "/api/sync/pairing/self" in s9
        assert "/api/sync/pairing/accept" in s9

    def test_section9_keeps_bulbe_and_kerckhoffs(self):
        s9 = _section(9).lower()
        assert "bulbe" in s9
        assert "kerckhoffs" in s9

    def test_twelve_sections_intact(self):
        text = _spec()
        missing = [f"## {i}. " for i in range(1, 13) if f"## {i}. " not in text]
        assert not missing, f"missing sections: {missing}"


# Section 6 lists S182 as built, not deferred


class TestSection6Updated:
    def test_section6_lists_s182_as_built(self):
        s6 = _section(6).lower()
        assert "s182" in s6
        assert "built" in s6

    def test_section6_panel_no_longer_deferred(self):
        s6 = _section(6).lower()
        assert "deferred to the named session" not in s6


# Package exports the pairing surface


class TestPackageExports:
    def test_init_exports_pairing_symbols(self):
        text = _init()
        for name in (
            "build_pairing_payload",
            "parse_pairing_payload",
            "accept_pairing_payload",
            "ParsedPairing",
            "resolve_self_routing_key",
        ):
            assert name in text, f"__init__ should export {name}"


# Frontend cartography: the panel is registered


class TestFrontendCartography:
    def test_panel_registered(self):
        assert "SyncPanel" in _frontend()

    def test_panel_marked_new_s182(self):
        assert re.search(r"SyncPanel\.svelte`?\s*\|\s*NEW\s*\|\s*S182", _frontend()) is not None


# Docs page describes the pairing ceremony and the panel


class TestDocsModel:
    def test_page_has_h1(self):
        assert _docs().lstrip().splitlines()[0].startswith("# ")

    def test_mentions_pairing(self):
        text = _docs().lower()
        assert "pairing" in text
        assert "integrity check" in text

    def test_mentions_panel_and_control(self):
        text = _docs().lower()
        assert "panel" in text
        assert "control what is shared" in text or "controls what is shared" in text

    def test_mentions_pairing_routes(self):
        text = _docs()
        assert "/api/sync/pairing/accept" in text

    def test_keeps_s181_model_tokens(self):
        text = _docs().lower()
        for token in ("live transport", "private route", "sync status", "watermark"):
            assert token in text, f"docs should keep {token!r}"

    def test_keeps_protocol_model_tokens(self):
        text = _docs().lower()
        for token in ("convergent", "last-writer-wins", "tombstone", "pull"):
            assert token in text, f"docs should keep {token!r}"

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
