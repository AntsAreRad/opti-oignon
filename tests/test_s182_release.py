#!/usr/bin/env python3
"""Tests for S182 Goal 4 -- the 3.5.0 -> 3.6.0 version bump and release (Theme 4).

Re-asserts the version surface after the bump that closes Theme 4 (Veilid Sync).
The superseded 3.5.0 assertions in test_s177_release and test_s178_veilid_pyproject
stay deselected (never edited or deleted); this file re-asserts the new 3.6.0
state, the CHANGELOG entry, and that the optional dependency groups stay isolated
from [all] with the version still hardcoded.

- __version__.py and pyproject.toml are the bare final 3.6.0, and consistent with
  each other; app.py reads the version from __version__ (no hardcoded literal).
- The CHANGELOG top entry is the v3.6.0 (S182) Theme 4 product-surface entry, with
  a [SECURITY] section; the v3.5.0 and prior entries are retained.
- The optional groups (llama, sqlcipher, veilid) are present but excluded from
  [all]; the version is hardcoded (no dynamic version).

File-content / manifest only, so the suite collects without the backend.
"""

import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent

FINAL_VERSION = "3.6.0"


def _read(*parts) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def _version_from_file() -> str:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', _read("opti_oignon", "__version__.py"))
    assert m, "no __version__ assignment found"
    return m.group(1)


def _pyproject() -> dict:
    return tomllib.loads(_read("pyproject.toml"))


# Version surface


class TestVersionBump:
    def test_version_file_is_final(self):
        assert _version_from_file() == FINAL_VERSION

    def test_version_file_contains_360(self):
        assert '"3.6.0"' in _read("opti_oignon", "__version__.py")

    def test_version_is_bare_form(self):
        assert re.match(r"^\d+\.\d+\.\d+$", _version_from_file())

    def test_no_rc_suffix(self):
        assert "-rc" not in _read("opti_oignon", "__version__.py")

    def test_pyproject_version_is_final(self):
        assert f'version = "{FINAL_VERSION}"' in _read("pyproject.toml")

    def test_pyproject_consistent_with_version_file(self):
        assert _pyproject()["project"]["version"] == _version_from_file()

    def test_no_stale_350_in_version_sites(self):
        assert "3.5.0" not in _read("opti_oignon", "__version__.py")
        assert 'version = "3.5.0"' not in _read("pyproject.toml")

    def test_app_py_reads_version_not_literal(self):
        app = _read("opti_oignon", "api", "app.py")
        assert "__version__" in app
        assert "3.6.0" not in app and "3.5.0" not in app


# CHANGELOG


class TestChangelog:
    def setup_method(self):
        self.c = _read("CHANGELOG.md")

    def test_v360_entry_present(self):
        assert "## v3.6.0 -- 2026-06-02 (S182)" in self.c

    def test_top_entry_is_360(self):
        entries = re.findall(r"## v(\d+\.\d+\.\d+)", self.c)
        assert entries and entries[0] == FINAL_VERSION

    def test_entry_is_product_surface_for_theme4(self):
        entry = self.c.split("## v3.6.0")[1].split("## v3.5.0")[0]
        for term in ("pairing", "sync", "panel"):
            assert term.lower() in entry.lower()

    def test_security_section_present(self):
        entry = self.c.split("## v3.6.0")[1].split("## v3.5.0")[0]
        assert "[SECURITY]" in entry

    def test_entry_states_bulbe_and_kerckhoffs(self):
        entry = self.c.split("## v3.6.0")[1].split("## v3.5.0")[0].lower()
        assert "bulbe" in entry
        assert "kerckhoffs" in entry

    def test_prior_entries_retained(self):
        assert "## v3.5.0 -- 2026-06-02 (S177)" in self.c
        assert "## v3.4.0 -- 2026-06-01 (S171)" in self.c
        assert "## v3.3.0" in self.c


# Optional dependency groups stay isolated from [all], version hardcoded


class TestOptionalGroups:
    def setup_method(self):
        self.data = _pyproject()
        self.optional = self.data["project"].get("optional-dependencies", {})

    def test_groups_present(self):
        for group in ("llama", "sqlcipher", "veilid"):
            assert group in self.optional, f"optional group missing: {group}"

    def test_all_group_exists(self):
        assert "all" in self.optional

    def test_heavy_groups_excluded_from_all(self):
        all_text = " ".join(self.optional.get("all", [])).lower()
        for token in ("llama-cpp-python", "sqlcipher", "veilid"):
            assert token not in all_text, f"{token} should not be in [all]"

    def test_version_is_hardcoded(self):
        assert "dynamic" not in self.data["project"]
        assert self.data["project"]["version"] == FINAL_VERSION
