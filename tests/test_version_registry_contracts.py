#!/usr/bin/env python3
"""Contracts for the single-source version registry.

The package register (``opti_oignon/__version__.py``) is the only place a
version may be declared. Every other site either derives from it at runtime
or is pinned equal to it here, so that a bump is one edit and any site that
lags turns the tree red instead of shipping silently stale:

  * RV1 -- the packaging manifest declares the register's version.
  * RV2 -- the frontend package declares the register's version.
  * RV3 -- the health dashboard's declaration default IS the register:
    equal in value, and derived in form (no version-shaped literal in the
    class body, so the two cannot drift apart between releases).
  * RV4 -- the newest versioned changelog heading is the register's
    version: a release without its changelog entry does not read green.

Local-only. Runs under pytest. No isolation window is needed: everything
under contract is read as data or imported as a leaf module.
"""

import json
import re
from pathlib import Path

import tomllib

from opti_oignon.__version__ import __version__ as REGISTER

REPO = Path(__file__).resolve().parent.parent

_SEMVER_LITERAL = re.compile(r'"\d+\.\d+\.\d+"')


def test_rv1_packaging_manifest_declares_the_register_version():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == REGISTER


def test_rv2_frontend_package_declares_the_register_version():
    data = json.loads(
        (REPO / "frontend" / "package.json").read_text(encoding="utf-8")
    )
    assert data["version"] == REGISTER


def test_rv3_health_dashboard_default_is_the_register():
    """Equal in value, and derived in form.

    Value equality alone would stay green for a hardcoded copy of today's
    version, which is exactly the drift this contract exists to prevent.
    The class body must carry no version-shaped literal at all: the default
    has to reference the register, so the next bump moves both or neither.
    """
    from opti_oignon.api import schemas

    assert schemas.HealthDashboard().version == REGISTER

    source = (REPO / "opti_oignon" / "api" / "schemas.py").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"class HealthDashboard\b.*?(?=\nclass |\Z)", source, re.S
    )
    assert match, "HealthDashboard class not found in schemas source"
    assert not _SEMVER_LITERAL.search(match.group(0)), (
        "the health dashboard default must derive from the register, "
        "never restate it as a literal"
    )


def test_rv4_newest_changelog_heading_is_the_register():
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## (\d+\.\d+\.\d+)", text, re.M)
    assert match, "no versioned changelog heading found"
    assert match.group(1) == REGISTER, (
        "a release ships with its changelog entry; the newest heading "
        "must be the register's version"
    )
