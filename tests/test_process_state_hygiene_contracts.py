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
  * PH3 -- the heap the collection built is frozen for the run, so a
    collector pass does not rescan it inside whichever contract happens to
    trigger it, and it is thawed when the session ends. A session whose
    conftest lacks the freeze shows the probe failing.

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


def _session(tmp_path, canaries, conftest_text=None):
    """Run pytest on ``canaries`` beside a copy of the conftest; name -> [(phase, message)].

    ``conftest_text`` replaces the copy's text, for a session run against a
    conftest with one line taken out.
    """
    shutil.copy2(_CONFTEST, tmp_path / "conftest.py")
    if conftest_text is not None:
        (tmp_path / "conftest.py").write_text(conftest_text, encoding="utf-8")
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


# ---------------------------------------------------------------------------
# PH3 -- the heap the collection built is frozen, and thawed at the end
# ---------------------------------------------------------------------------
_FROZEN = '''import gc


def test_the_heap_collection_built_is_frozen():
    assert gc.get_freeze_count() > 0, gc.get_freeze_count()
'''


def _conftest_without(tmp_path, needle):
    """A copy of the conftest with the one line holding ``needle`` taken out; its text."""
    lines = _CONFTEST.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if needle not in line]
    assert len(kept) == len(lines) - 1, f"exactly one line holds {needle!r}"
    return "".join(kept)


def test_ph3_the_heap_collection_built_is_frozen_for_the_run_and_thawed_at_the_end(tmp_path):
    import ast

    # The real conftest freezes: a contract sees a frozen heap.
    (tmp_path / "real").mkdir()
    outcomes = _session(tmp_path / "real", _FROZEN)
    assert outcomes == {"test_the_heap_collection_built_is_frozen": []}, outcomes

    # The probe can fail: without the freeze, the same contract sees nothing frozen.
    (tmp_path / "bare").mkdir()
    stripped = _session(tmp_path / "bare", _FROZEN, conftest_text=_conftest_without(tmp_path, "gc.freeze()"))
    assert [kind for kind, _message in stripped["test_the_heap_collection_built_is_frozen"]] == ["failure"], stripped

    # The heap is thawed when the session ends, so nothing frozen outlives it.
    tree = ast.parse(_CONFTEST.read_text(encoding="utf-8"))
    unconfigure = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "pytest_unconfigure")
    calls = {f"{node.func.value.id}.{node.func.attr}" for node in ast.walk(unconfigure)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)}
    assert "gc.unfreeze" in calls, sorted(calls)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
