#!/usr/bin/env python3
"""Contracts for the ladder's blade census: every contract a change adds has its blade.

The directed-mutation tier counted the unchecked lines of the blade register
and nothing else. A change that added a contract and never wrote its line
passed the tier: the count found nothing to count. The tier now takes a
census of the contracts in the tree, compares it with HEAD, and holds every
contract the change adds to a checked line that names it.

  * BK1 -- a contract added against HEAD with no checked line fails the tier,
    and the tier names it.
  * BK2 -- a checked line naming the added contract covers it.
  * BK3 -- an unchecked line covers nothing.
  * BK4 -- a new suite git does not track yet is counted whole.
  * BK5 -- a contract moved inside its file is not added.
  * BK6 -- a contract the selection rule deselects, alone or with its file,
    is exempt and named.
  * BK7 -- a contract under an ignored path is exempt and named.
  * BK8 -- deselecting one parameter leaves the function a contract.
  * BK9 -- a bare name two contracts share covers neither; the
    path-qualified form covers its own.
  * BK10 -- a line naming one parameter names its function.
  * BK11 -- a method of a Test class is a contract; one of a plain class is
    not.
  * BK12 -- a file pytest does not collect adds no contract.
  * BK13 -- a directory pytest does not enter adds no contract.
  * BK14 -- an added Rust test needs its line.
  * BK15 -- an ignored Rust test is exempt, whichever side of the test
    attribute its ignore sits.
  * BK16 -- an added browser test is owed to the machine until a checked
    line names it.
  * BK17 -- a census that finds no Python contract fails rather than passes.
  * BK18 -- a Rust test attribute the census cannot pair with a function
    fails rather than hides.
  * BK19 -- a checked line naming nothing in the tree is counted.
  * BK20 -- a test file the census cannot read fails the tier by name.
  * BK21 -- an unreadable selection rule exempts nothing.
  * BK22 -- a staged new suite is counted, as it is at a block's closure.
  * BK23 -- the census follows the collection names the rule sets.
  * BK24 -- the data places are never opened.
  * BK25 -- a function whose every parameter is deselected is exempt.
  * BK26 -- a contract pytest is told to skip, by its own mark, its class's
    or its module's, is exempt and named; one skipped on a condition is not.
  * BK27 -- a contract defined under a module-level block is counted.
  * BK28 -- a unittest case is counted whatever its name.

Local-only (the public distribution ships no tests). Each contract runs the
real tier on a committed fake tree carrying a copy of the ladder; git runs
without the maintainer's global or system configuration, so no commit there
is signed and no hook runs.
"""

import os
import shutil
import subprocess
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_REPO = _TESTS.parent
_IDENTITY = ["-c", "user.name=contract", "-c", "user.email=contract@example.invalid",
             "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
_ENV.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", PYTHONDONTWRITEBYTECODE="1")

_KEPT = "def test_kept():\n    assert True\n"
_PARAM = '\n\n@pytest.mark.parametrize("n", [1, 2])\ndef {name}(n):\n    assert n\n'
_RUST = "#[cfg(test)]\nmod tests {{\n{body}}}\n"


def _fn(name):
    return f"\n\ndef {name}():\n    assert True\n"


def _pyproject(*opts, extra=""):
    lines = "".join(f"    {o}\n" for o in ("-p no:randomly", *opts))
    return f'[tool.pytest.ini_options]\n{extra}addopts = """\n{lines}"""\n'


_BASE = {"pyproject.toml": _pyproject(), "tests/test_base.py": _KEPT}


def _place(tree, files):
    for rel, text in files.items():
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *_IDENTITY, *args], env=_ENV, check=True,
                          capture_output=True, text=True)


def _tree(tmp_path, files=_BASE, register=""):
    """A fake tree whose ``files`` and copy of the ladder are HEAD, with its register."""
    tree = tmp_path / "tree"
    _place(tree, files)
    (tree / "scripts").mkdir()
    shutil.copy2(_REPO / "scripts" / "ladder.sh", tree / "scripts" / "ladder.sh")
    _git(tree, "init", "-q")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "--no-verify", "-m", "base")
    register_path = tmp_path / "register.md"
    register_path.write_text(register, encoding="utf-8")
    _git(tree, "config", "oo.bladeRegister", str(register_path))
    return tree


def _t3(tree):
    return subprocess.run(["bash", str(tree / "scripts" / "ladder.sh"), "t3"], cwd=tree, env=_ENV,
                          capture_output=True, text=True, timeout=120)


# ---------------------------------------------------------------------------
# BK1-BK5 -- what a change adds, and what covers it
# ---------------------------------------------------------------------------
def test_bk1_an_added_contract_without_a_checked_line_fails_the_tier_by_name(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/test_base.py": _KEPT + _fn("test_new")})
    run = _t3(tree)
    assert run.returncode == 1, run.stdout
    assert "FAIL  1 contract(s) added against HEAD have no checked blade line" in run.stdout, run.stdout
    assert "uncovered: tests/test_base.py::test_new" in run.stdout, run.stdout


def test_bk2_a_checked_line_naming_the_added_contract_covers_it(tmp_path):
    tree = _tree(tmp_path, register="- [x] test_new - defect: its body is emptied\n")
    _place(tree, {"tests/test_base.py": _KEPT + _fn("test_new")})
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert ("PASS  every contract added against HEAD is accounted for "
            "(1 added: 1 covered, 0 exempt, 0 owed)") in run.stdout, run.stdout


def test_bk3_an_unchecked_line_covers_nothing(tmp_path):
    tree = _tree(tmp_path, register="- [ ] test_new - defect: its body is emptied\n")
    _place(tree, {"tests/test_base.py": _KEPT + _fn("test_new")})
    run = _t3(tree)
    assert "uncovered: tests/test_base.py::test_new" in run.stdout, run.stdout


def test_bk4_a_new_suite_git_does_not_track_yet_is_counted_whole(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/test_fresh.py": _fn("test_one") + _fn("test_two")})
    run = _t3(tree)
    assert "FAIL  2 contract(s) added against HEAD have no checked blade line" in run.stdout, run.stdout
    for name in ("test_one", "test_two"):
        assert f"uncovered: tests/test_fresh.py::{name}" in run.stdout, run.stdout


def test_bk5_a_contract_moved_inside_its_file_is_not_added(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/test_base.py": "def helper():\n    return 1\n\n\ndef test_kept():\n    assert helper() == 1\n"})
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "PASS  no contract added against HEAD" in run.stdout, run.stdout


# ---------------------------------------------------------------------------
# BK6-BK10 -- the selection rule, and the names a line may use
# ---------------------------------------------------------------------------
def test_bk6_a_contract_the_selection_rule_deselects_alone_or_with_its_file_is_exempt_and_named(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {
        "pyproject.toml": _pyproject("--deselect=tests/test_base.py::test_gone", "--deselect=tests/test_other.py"),
        "tests/test_base.py": _KEPT + _fn("test_gone"),
        "tests/test_other.py": _fn("test_elsewhere"),
    })
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "exempt (--deselect): tests/test_base.py::test_gone" in run.stdout, run.stdout
    assert "exempt (--deselect): tests/test_other.py::test_elsewhere" in run.stdout, run.stdout
    assert "(2 added: 0 covered, 2 exempt, 0 owed)" in run.stdout, run.stdout


def test_bk7_a_contract_under_an_ignored_path_is_exempt_and_named(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"pyproject.toml": _pyproject("--ignore=tests/life"),
                  "tests/life/test_life.py": _fn("test_alive")})
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "exempt (--ignore): tests/life/test_life.py::test_alive" in run.stdout, run.stdout


def test_bk8_deselecting_one_parameter_leaves_the_function_a_contract(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {
        "pyproject.toml": _pyproject("--deselect=tests/test_base.py::test_param[1]"),
        "tests/test_base.py": "import pytest\n\n\n" + _KEPT + _PARAM.format(name="test_param"),
    })
    run = _t3(tree)
    assert "uncovered: tests/test_base.py::test_param" in run.stdout, run.stdout


def test_bk9_a_bare_name_two_contracts_share_covers_neither_until_path_qualified(tmp_path):
    register = tmp_path / "register.md"
    tree = _tree(tmp_path, register="- [x] test_kept - defect: its assertion is inverted\n")
    _place(tree, {"tests/test_twin.py": _fn("test_kept")})
    run = _t3(tree)
    assert "uncovered: tests/test_twin.py::test_kept" in run.stdout, run.stdout
    register.write_text(register.read_text(encoding="utf-8")
                        + "- [x] tests/test_twin.py::test_kept - defect: its assertion is inverted\n",
                        encoding="utf-8")
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "(1 added: 1 covered, 0 exempt, 0 owed)" in run.stdout, run.stdout


def test_bk10_a_line_naming_one_parameter_names_its_function(tmp_path):
    tree = _tree(tmp_path, register="- [x] test_param[case one] - defect: the case is dropped\n")
    _place(tree, {"tests/test_base.py": "import pytest\n\n\n" + _KEPT + _PARAM.format(name="test_param")})
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "(1 added: 1 covered, 0 exempt, 0 owed)" in run.stdout, run.stdout


# ---------------------------------------------------------------------------
# BK11-BK13 -- what pytest collects is what the census counts
# ---------------------------------------------------------------------------
def test_bk11_a_test_class_method_is_a_contract_and_a_plain_class_method_is_not(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/test_base.py": _KEPT
                  + "\n\nclass TestThing:\n    def test_inside(self):\n        assert True\n"
                  + "\n\nclass Helper:\n    def test_helper(self):\n        assert True\n"})
    run = _t3(tree)
    assert "FAIL  1 contract(s) added against HEAD have no checked blade line" in run.stdout, run.stdout
    assert "uncovered: tests/test_base.py::TestThing::test_inside" in run.stdout, run.stdout


def test_bk12_a_file_pytest_does_not_collect_adds_no_contract(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/helpers.py": _fn("test_helper"), "tests/conftest.py": _fn("test_fixture_like")})
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "PASS  no contract added against HEAD" in run.stdout, run.stdout


def test_bk13_a_directory_pytest_does_not_enter_adds_no_contract(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/.cache/test_hidden.py": _fn("test_hidden"),
                  "tests/node_modules/test_vendored.py": _fn("test_vendored"),
                  "build/test_built.py": _fn("test_built")})
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "PASS  no contract added against HEAD" in run.stdout, run.stdout


# ---------------------------------------------------------------------------
# BK14-BK16 -- Rust tests, and browser tests owed to the machine
# ---------------------------------------------------------------------------
def test_bk14_an_added_rust_test_needs_its_line(tmp_path):
    tree = _tree(tmp_path)
    body = "    #[test]\n    fn adds_one() {}\n\n    #[tokio::test]\n    async fn waits_for_one() {}\n"
    _place(tree, {"rust/core/src/lib.rs": _RUST.format(body=body)})
    run = _t3(tree)
    assert run.returncode == 1, run.stdout
    for name in ("adds_one", "waits_for_one"):
        assert f"uncovered: rust/core/src/lib.rs::{name}" in run.stdout, run.stdout


def test_bk15_an_ignored_rust_test_is_exempt_whichever_side_of_the_test_attribute_its_ignore_sits(tmp_path):
    tree = _tree(tmp_path)
    body = ("    #[ignore]\n    #[test]\n    fn slow_before() {}\n\n"
            '    #[test]\n    #[ignore = "machine only"]\n    fn slow_after() {}\n')
    _place(tree, {"rust/core/src/lib.rs": _RUST.format(body=body)})
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "exempt (#[ignore]): rust/core/src/lib.rs::slow_before" in run.stdout, run.stdout
    assert "exempt (#[ignore]): rust/core/src/lib.rs::slow_after" in run.stdout, run.stdout


def test_bk16_an_added_browser_test_is_owed_until_a_checked_line_names_it(tmp_path):
    register = tmp_path / "register.md"
    tree = _tree(tmp_path)
    _place(tree, {"frontend/tests/e2e/home.spec.ts":
                  "import { test } from '@playwright/test';\n\ntest('home renders', async () => {});\n"})
    run = _t3(tree)
    assert run.returncode == 3, run.stdout
    assert "owed: frontend/tests/e2e/home.spec.ts::home renders" in run.stdout, run.stdout
    register.write_text("- [x] frontend/tests/e2e/home.spec.ts::home renders - defect: the page is left blank\n",
                        encoding="utf-8")
    run = _t3(tree)
    assert run.returncode == 0, run.stdout


# ---------------------------------------------------------------------------
# BK17-BK21 -- a census that cannot see says so
# ---------------------------------------------------------------------------
def test_bk17_a_census_that_finds_no_python_contract_fails_rather_than_passes(tmp_path):
    tree = _tree(tmp_path, files={"pyproject.toml": _pyproject(), "tests/helpers.py": "def helper():\n    return 1\n"})
    run = _t3(tree)
    assert run.returncode == 1, run.stdout
    assert "FAIL  the census is blind: no Python contract found in the tree" in run.stdout, run.stdout


def test_bk18_a_rust_test_attribute_the_census_cannot_pair_with_a_function_fails_rather_than_hides(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"rust/core/src/lib.rs": "macro_rules! case {\n    ($name:ident) => {\n        #[test]\n"
                                          "        fn $name() {}\n    };\n}\n\ncase!(one);\n"})
    run = _t3(tree)
    assert run.returncode == 1, run.stdout
    assert ("FAIL  the census is blind to Rust tests: 1 test attribute(s), 0 test function(s) found"
            in run.stdout), run.stdout


def test_bk19_a_checked_line_naming_nothing_in_the_tree_is_counted(tmp_path):
    tree = _tree(tmp_path, register="- [x] test_kept - defect: its assertion is inverted\n"
                                    "- [x] test_long_gone - defect: renamed since\n")
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "1 checked line(s) name nothing in the tree" in run.stdout, run.stdout


def test_bk20_a_test_file_the_census_cannot_read_fails_the_tier_by_name(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/test_broken.py": "def test_x(:\n    pass\n"})
    run = _t3(tree)
    assert run.returncode == 1, run.stdout
    assert "unreadable: tests/test_broken.py: SyntaxError" in run.stdout, run.stdout


def test_bk21_an_unreadable_selection_rule_exempts_nothing(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"pyproject.toml": '[tool.pytest.ini_options\naddopts = "--deselect=tests/test_base.py::test_gone"\n',
                  "tests/test_base.py": _KEPT + _fn("test_gone")})
    run = _t3(tree)
    assert "uncovered: tests/test_base.py::test_gone" in run.stdout, run.stdout


# ---------------------------------------------------------------------------
# BK22-BK24 -- the closure's index, the rule's names, the data places
# ---------------------------------------------------------------------------
def test_bk22_a_staged_new_suite_is_counted_as_it_is_at_a_closure(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/test_staged.py": _fn("test_staged")})
    _git(tree, "add", "tests/test_staged.py")
    run = _t3(tree)
    assert "uncovered: tests/test_staged.py::test_staged" in run.stdout, run.stdout


def test_bk23_the_census_follows_the_collection_names_the_rule_sets(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"pyproject.toml": _pyproject(extra='python_files = ["check_*.py"]\npython_functions = ["check"]\n'),
                  "tests/check_rules.py": _fn("check_one"), "tests/test_plain.py": _fn("test_two")})
    run = _t3(tree)
    assert "FAIL  1 contract(s) added against HEAD have no checked blade line" in run.stdout, run.stdout
    assert "uncovered: tests/check_rules.py::check_one" in run.stdout, run.stdout


def test_bk24_the_data_places_are_never_opened(tmp_path):
    tree = _tree(tmp_path)
    trap = "def test_x(:\n    pass\n"
    _place(tree, {"data/test_trap.py": trap, "opti_oignon/data/test_trap.py": trap})
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "PASS  no contract added against HEAD" in run.stdout, run.stdout


# ---------------------------------------------------------------------------
# BK25-BK28 -- what pytest runs, skips or finds where a parser might not look
# ---------------------------------------------------------------------------
def test_bk25_a_function_whose_every_parameter_is_deselected_is_exempt(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {
        "pyproject.toml": _pyproject("--deselect=tests/test_base.py::test_halves[node]",
                                     "--deselect=tests/test_base.py::test_halves[wiring]"),
        "tests/test_base.py": "import pytest\n\n\n" + _KEPT
        + '\n\n@pytest.mark.parametrize("half", ("node", "wiring"))\ndef test_halves(half):\n    assert half\n',
    })
    run = _t3(tree)
    assert run.returncode == 0, run.stdout
    assert "exempt (--deselect): tests/test_base.py::test_halves" in run.stdout, run.stdout


def test_bk26_a_contract_pytest_is_told_to_skip_is_exempt_and_one_skipped_on_a_condition_is_not(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {
        "tests/test_base.py": "import pytest\n\n\n" + _KEPT
        + '\n\n@pytest.mark.skip(reason="superseded")\ndef test_skipped():\n    assert True\n'
        + "\n\n@pytest.mark.skip\nclass TestShelved:\n    def test_shelved(self):\n        assert True\n"
        + '\n\n@pytest.mark.skipif(True, reason="on a condition")\ndef test_conditional():\n    assert True\n',
        "tests/test_parked.py": 'import pytest\n\npytestmark = [pytest.mark.skip(reason="parked")]\n' + _fn("test_parked"),
    })
    run = _t3(tree)
    for node in ("tests/test_base.py::test_skipped", "tests/test_base.py::TestShelved::test_shelved",
                 "tests/test_parked.py::test_parked"):
        assert f"exempt (skip): {node}" in run.stdout, run.stdout
    assert "uncovered: tests/test_base.py::test_conditional" in run.stdout, run.stdout


def test_bk27_a_contract_defined_under_a_module_level_block_is_counted(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/test_base.py": "import sys\n\n\n" + _KEPT
                  + "\n\nif sys.version_info >= (3,):\n    def test_under_if():\n        assert True\n"
                  + "\n\ntry:\n    import json\nexcept ImportError:\n    json = None\nelse:\n"
                  + "    def test_under_try():\n        assert json\n"})
    run = _t3(tree)
    assert "FAIL  2 contract(s) added against HEAD have no checked blade line" in run.stdout, run.stdout
    for name in ("test_under_if", "test_under_try"):
        assert f"uncovered: tests/test_base.py::{name}" in run.stdout, run.stdout


def test_bk28_a_unittest_case_is_counted_whatever_its_name(tmp_path):
    tree = _tree(tmp_path)
    _place(tree, {"tests/test_base.py": "import unittest\n\n\n" + _KEPT
                  + "\n\nclass Checks(unittest.TestCase):\n    def test_one(self):\n        self.assertTrue(True)\n\n"
                  + "    def helper(self):\n        return 1\n"})
    run = _t3(tree)
    assert "FAIL  1 contract(s) added against HEAD have no checked blade line" in run.stdout, run.stdout
    assert "uncovered: tests/test_base.py::Checks::test_one" in run.stdout, run.stdout


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
