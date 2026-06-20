#!/usr/bin/env python3
"""Tests for S178 Goal 4 -- the optional [veilid] dependency group.

Covers pyproject.toml:

- A ``[project.optional-dependencies].veilid`` group exists and pins the veilid
  Python bindings.
- veilid is isolated from ``[all]`` exactly as llama and sqlcipher are: the
  [all] meta-group pulls only the pure-Python extras (auth, dev, docs), so a
  stock ``opti-oignon[all]`` install never drags in platform-specific bindings.
- The version stays hardcoded (3.5.0, matching __version__.py) with no dynamic
  version, so importing the package is never triggered to compute it.
- The group is documented as a Daily-only, explicitly-installed capability.

Pure file/TOML parsing, so the suite collects without the backend.
"""

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _data() -> dict:
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def _content() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _optional() -> dict:
    return _data()["project"]["optional-dependencies"]


def _version_from_file() -> str:
    import re

    text = (ROOT / "opti_oignon" / "__version__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    assert m, "no __version__ assignment found"
    return m.group(1)


# The [veilid] group


class TestVeilidGroup:
    def test_group_exists(self):
        assert "veilid" in _optional()

    def test_group_pins_veilid_bindings(self):
        reqs = _optional()["veilid"]
        assert any(r.lower().startswith("veilid") for r in reqs), reqs

    def test_group_uses_a_floor_pin(self):
        reqs = _optional()["veilid"]
        assert any(">=" in r for r in reqs), reqs

    def test_group_is_small(self):
        # The bindings only; the server is staged by a script, not pip.
        assert len(_optional()["veilid"]) <= 3


# Isolation from [all]


class TestIsolationFromAll:
    def test_all_group_exists(self):
        assert "all" in _optional()

    def test_veilid_not_in_all(self):
        joined = " ".join(_optional()["all"])
        assert "veilid" not in joined

    def test_llama_and_sqlcipher_also_excluded_from_all(self):
        joined = " ".join(_optional()["all"])
        assert "llama" not in joined
        assert "sqlcipher" not in joined

    def test_all_pulls_only_pure_python_extras(self):
        joined = " ".join(_optional()["all"])
        for extra in ("auth", "dev", "docs"):
            assert extra in joined

    def test_platform_groups_present_but_separate(self):
        opt = _optional()
        for group in ("llama", "sqlcipher", "veilid"):
            assert group in opt


# Version stays hardcoded


class TestVersionPinning:
    def test_version_is_hardcoded_350(self):
        assert _data()["project"]["version"] == "3.5.0"

    def test_version_matches_version_file(self):
        assert _data()["project"]["version"] == _version_from_file()

    def test_no_dynamic_version(self):
        assert "dynamic" not in _data()["project"]

    def test_version_literal_in_text(self):
        assert 'version = "3.5.0"' in _content()


# Documentation in the manifest


class TestDocumented:
    def test_explicit_install_documented(self):
        assert "opti-oignon[veilid]" in _content()

    def test_daily_only_documented(self):
        text = _content().lower()
        assert "daily" in text and "bulbe" in text
