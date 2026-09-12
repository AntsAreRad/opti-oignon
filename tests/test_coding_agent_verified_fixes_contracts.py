#!/usr/bin/env python3
"""Contracts for the coding agent's verified fix loop.

The agent's test verdict was read from substrings: a run whose output
mentioned the word error failed, a run with no tests passed. The verdict
now comes from the pytest summary, and the one tolerance that stays -- no
tests ran counts as passed, with the count at zero -- is explicit. The fix
loop can spend several candidates per attempt: each is applied, tested,
and undone by its inverse when it fails, so the next candidate starts from
the same workspace; the first that passes stays. One candidate, the
default, is the loop as it was.

  * CA1 -- the verdict is the summary's: counts are read, a passing run
    that mentions error still passes, no tests ran is passed with count
    zero, a failing runner is a failure with its error.
  * CA2 -- with several candidates a failing fix is undone by its inverse
    before the next one is tried, and the first passing fix stays.
  * CA3 -- with one candidate the loop is unchanged: one call per attempt,
    no undo.
  * CA4 -- a created file is undone by removal, an overwritten one by its
    original content, and the candidate count is the YAML's.

Local-only (the public distribution ships no tests). The coding module is
loaded through the shared isolation window with a recording sandbox
session and a scripted model call; nothing else in the package is reached.
"""

import json
import shlex
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_YAML = REPO / "opti_oignon" / "config" / "coding_agent.yaml"
_TEST_CMD = "python3 -m pytest -x --tb=short"


class _Session:
    """A sandbox stand-in with a small virtual filesystem and scripted tests."""

    def __init__(self, files, passes_when, test_output=None):
        self.fs = dict(files)
        self.passes_when = passes_when
        self.test_output = test_output
        self.ops = []

    def bash(self, command, timeout=30):
        self.ops.append(("bash", command))
        if command.startswith("cat "):
            path = shlex.split(command)[1]
            if path not in self.fs:
                raise RuntimeError(f"cat: {path}: No such file")
            return self.fs[path]
        if command.startswith("rm -f "):
            self.fs.pop(shlex.split(command)[2], None)
            return ""
        if command == _TEST_CMD:
            if self.test_output is not None:
                out = self.test_output
                if isinstance(out, Exception):
                    raise out
                return out
            return "1 passed in 0.1s" if self.passes_when(self.fs) else "1 failed in 0.1s"
        return ""

    def create_file(self, path, content):
        self.ops.append(("create_file", path, content))
        self.fs[path] = content
        return f"created {path}"

    def str_replace(self, path, old, new):
        self.ops.append(("str_replace", path, old, new))
        if old not in self.fs.get(path, ""):
            raise ValueError(f"{old!r} not found in {path}")
        self.fs[path] = self.fs[path].replace(old, new, 1)
        return f"replaced in {path}"

    def extract_files(self):
        return [{"path": p} for p in self.fs]


def _open():
    loaded, restore = isolate(
        targets={
            "opti_oignon.inference_compute": source("inference_compute.py"),
            "opti_oignon.coding_agent": source("coding_agent.py"),
        },
        packages=("opti_oignon",),
    )
    return loaded["opti_oignon.coding_agent"], restore


def _agent(mod, session, llm_call=None, **cfg):
    fields = dict(max_fix_retries=1, auto_test_command=_TEST_CMD, enable_cascading=False, fix_candidates=1)
    fields.update(cfg)
    config = mod.CodingAgentConfig(**fields)
    return mod.CodingAgent(sandbox_session=session, llm_call=llm_call, config=config, model="m")


def _fix(old, new, path="app.py"):
    return json.dumps({"analysis": "a", "fix_type": "str_replace", "file_path": path, "old_str": old, "new_str": new})


def _scripted(*replies):
    calls = []
    replies = list(replies)

    def llm_call(prompt, system="", model=None):
        calls.append({"prompt": prompt, "system": system, "model": model})
        return replies.pop(0) if replies else replies_exhausted()

    def replies_exhausted():
        raise AssertionError("the model was asked more often than scripted")
    llm_call.calls = calls
    return llm_call


# ---------------------------------------------------------------------------
# CA1 -- the verdict is the summary's
# ---------------------------------------------------------------------------
def test_ca1_the_test_verdict_is_read_from_the_summary_not_from_substrings():
    mod, restore = _open()
    try:
        def run(output):
            agent = _agent(mod, _Session({}, lambda fs: True, test_output=output))
            return agent.run_tests()
        good = run("collected 3 items\n\n3 passed in 0.12s")
        assert good.passed is True and good.test_count == 3 and good.failures == 0 and good.return_code == 0
        bad = run("2 passed, 1 failed in 0.20s")
        assert bad.passed is False and bad.test_count == 3 and bad.failures == 1 and bad.return_code == 1
        errored = run("1 passed, 1 error in 0.1s")
        assert errored.passed is False and errored.failures == 1 and errored.test_count == 2, "an error is a failure"
        trap = run("test_error_handling PASSED\n1 passed in 0.1s")
        assert trap.passed is True, "a passing run that mentions error is a passing run"
        none = run("no tests ran in 0.01s")
        assert none.passed is True and none.test_count == 0, "the one tolerance, kept and explicit"
        broken = run(RuntimeError("bash exited 2: FAILED collecting"))
        assert broken.passed is False and "FAILED" in broken.output and broken.error
    finally:
        restore()


# ---------------------------------------------------------------------------
# CA2 -- several candidates, undone by their inverse
# ---------------------------------------------------------------------------
def test_ca2_a_failing_candidate_is_undone_before_the_next_and_the_first_pass_stays():
    mod, restore = _open()
    try:
        session = _Session({"app.py": "x = 1\n"}, lambda fs: fs["app.py"] == "x = 2\n")
        llm = _scripted(_fix("x = 1", "x = 3"), _fix("x = 1", "x = 2"), _fix("x = 1", "x = 9"))
        agent = _agent(mod, session, llm, fix_candidates=3)
        assert agent._fix_loop(agent.run_tests()) is True
        assert session.fs["app.py"] == "x = 2\n"
        assert len(llm.calls) == 2, "the third candidate was never asked for"
        edits = [op for op in session.ops if op[0] == "str_replace"]
        assert edits == [
            ("str_replace", "app.py", "x = 1", "x = 3"),
            ("str_replace", "app.py", "x = 3", "x = 1"),
            ("str_replace", "app.py", "x = 1", "x = 2"),
        ], "apply, undo by the inverse, apply the next"
        tests = [op for op in session.ops if op == ("bash", _TEST_CMD)]
        assert len(tests) == 3, "the initial run, then one per candidate"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CA3 -- one candidate is the loop as it was
# ---------------------------------------------------------------------------
def test_ca3_with_one_candidate_the_loop_is_unchanged():
    mod, restore = _open()
    try:
        session = _Session({"app.py": "x = 1\n"}, lambda fs: fs["app.py"] == "x = 2\n")
        llm = _scripted(_fix("x = 1", "x = 3"), _fix("x = 3", "x = 2"))
        agent = _agent(mod, session, llm, fix_candidates=1, max_fix_retries=2)
        assert agent._fix_loop(agent.run_tests()) is True
        assert len(llm.calls) == 2
        edits = [op for op in session.ops if op[0] == "str_replace"]
        assert edits == [
            ("str_replace", "app.py", "x = 1", "x = 3"),
            ("str_replace", "app.py", "x = 3", "x = 2"),
        ], "a failed fix stays in place and the next attempt builds on it, as before"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CA4 -- undo of a created file, and the YAML count
# ---------------------------------------------------------------------------
def test_ca4_a_created_file_is_removed_on_undo_and_the_count_is_the_yamls():
    import yaml

    mod, restore = _open()
    try:
        raw = yaml.safe_load(_YAML.read_text(encoding="utf-8"))
        assert int(raw["fix_candidates"]) == 1, "the shipped default is the loop as it was"
        assert mod._load_config().fix_candidates == int(raw["fix_candidates"])
        create_new = json.dumps({"fix_type": "create_file", "file_path": "conftest.py", "content": "bad\n"})
        overwrite = json.dumps({"fix_type": "create_file", "file_path": "app.py", "content": "x = 5\n"})
        good = _fix("x = 1", "x = 2")
        session = _Session({"app.py": "x = 1\n"}, lambda fs: fs["app.py"] == "x = 2\n" and "conftest.py" not in fs)
        agent = _agent(mod, session, _scripted(create_new, overwrite, good), fix_candidates=3)
        assert agent._fix_loop(agent.run_tests()) is True
        assert "conftest.py" not in session.fs, "a created file that did not help is removed"
        assert session.fs["app.py"] == "x = 2\n"
        assert ("bash", "rm -f conftest.py") in session.ops
        assert ("create_file", "app.py", "x = 1\n") in session.ops, "an overwritten file gets its original back"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
