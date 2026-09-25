#!/usr/bin/env python3
"""Contracts for the suite-wide check on what a contract leaves behind.

``tests/conftest.py`` reads the module cache and the transport before any
fixture of a contract is set up, and fails the contract at its teardown when
it left a stand-in project module, a neutralised project entry or a replaced
``urllib.request.urlopen``. These contracts run a real pytest session in a
temporary directory -- a copy of that conftest beside canary tests -- and
read its outcome from the session's own junit file.

  * PH1 -- each of the three leaves fails the contract that left it, at its
    teardown and by name, while the body of that contract passed.
  * PH2 -- what is not a leak passes: a stand-in a suite's own fixture puts
    back, a real module imported for the first time, and a contract that
    touches nothing.

Local-only (the public distribution ships no tests).
"""

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_CONFTEST = Path(__file__).resolve().parent / "conftest.py"

_LEAVES = '''import sys
import types
import urllib.request


def test_leaves_a_stand_in():
    sys.modules["opti_oignon.ph_stand_in"] = types.ModuleType("opti_oignon.ph_stand_in")


def test_leaves_a_neutralised_entry():
    sys.modules["opti_oignon.ph_neutralised"] = None


def test_leaves_the_transport_replaced():
    urllib.request.urlopen = lambda *args, **kwargs: None
'''

_NOT_LEAVES = '''import sys
import types

import pytest


@pytest.fixture
def window():
    sys.modules["opti_oignon.ph_window"] = types.ModuleType("opti_oignon.ph_window")
    yield
    del sys.modules["opti_oignon.ph_window"]


def test_a_suites_own_fixture_puts_it_back(window):
    assert "opti_oignon.ph_window" in sys.modules


def test_a_real_module_imported_for_the_first_time():
    module = types.ModuleType("opti_oignon.ph_real")
    module.__file__ = __file__
    sys.modules["opti_oignon.ph_real"] = module


def test_nothing_touched():
    assert True
'''


def _session(tmp_path, canaries):
    """Run pytest on ``canaries`` beside a copy of the conftest; name -> [(phase, message)]."""
    shutil.copy2(_CONFTEST, tmp_path / "conftest.py")
    shutil.copy2(_CONFTEST.with_name("_data_firewall.py"), tmp_path / "_data_firewall.py")
    (tmp_path / "test_canaries.py").write_text(canaries, encoding="utf-8")
    junit = tmp_path / "junit.xml"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTEST_ADDOPTS", None)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
         f"--junitxml={junit}", str(tmp_path / "test_canaries.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    outcomes = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        reds = outcomes.setdefault(case.get("name"), [])
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                reds.append((kind, node.get("message") or ""))
    return outcomes


# ---------------------------------------------------------------------------
# PH1 -- each leave fails its contract, at teardown, by name
# ---------------------------------------------------------------------------
def test_ph1_each_leave_fails_the_contract_that_left_it_at_teardown_and_by_name(tmp_path):
    outcomes = _session(tmp_path, _LEAVES)
    expected = {
        "test_leaves_a_stand_in": "opti_oignon.ph_stand_in",
        "test_leaves_a_neutralised_entry": "opti_oignon.ph_neutralised",
        "test_leaves_the_transport_replaced": "urlopen",
    }
    assert set(outcomes) == set(expected), outcomes
    for name, named in expected.items():
        reds = outcomes[name]
        assert reds, f"{name} is failed"
        assert all(kind == "error" and "teardown" in message for kind, message in reds), (name, reds)
        assert any(named in message for _kind, message in reds), (name, reds)


# ---------------------------------------------------------------------------
# PH2 -- what is not a leak passes
# ---------------------------------------------------------------------------
def test_ph2_a_fixture_that_puts_back_a_real_module_and_an_untouched_contract_pass(tmp_path):
    outcomes = _session(tmp_path, _NOT_LEAVES)
    assert set(outcomes) == {
        "test_a_suites_own_fixture_puts_it_back",
        "test_a_real_module_imported_for_the_first_time",
        "test_nothing_touched",
    }, outcomes
    assert not any(outcomes.values()), outcomes


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
