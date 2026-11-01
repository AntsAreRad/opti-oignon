"""Contracts for the images the check of an answer's code writes: it keeps none.

When the user turns automatic checking on, the code blocks of an answer run in a sandbox made for each run
(``verification.py``, from the chat turn and the agentic one). A run whose sandbox goes with it copies its new images
to the code output directory and names the copies in its result, which the code panel lists; a check's result names no
file, and nothing else reads the directory but the forgetting and the wipe: the checks' copies were kept for nobody, by
no conversation, at the top, until the next deletion of any conversation. Now a check keeps none: the executor, told
so, neither copies its images out nor names them, and they go with the check's sandbox.

  * VO1 -- THROUGH THE REAL CHECK: the real verification engine over the real executor and sandbox runs an answer's
    block that writes an image, and one fixed after a failure: nothing reaches the output directory and no sandbox
    stays; a panel's run of the same code, on the same executor, still copies its image out.
  * VO2 -- THE EXECUTOR KEEPS NONE WHEN TOLD: a run told to keep none names no output and copies none out, of no
    conversation or of one, and its sandbox goes; in persistent mode the image stays in the conversation's sandbox,
    named by no result.
  * VO3 -- ce8 of tests/test_code_execution_in_the_sandbox_contracts.py, whose stand-in executor took no word on its
    outputs: its cases and assertion unchanged, the stand-in taking the word a check passes, and every check's run
    asked to keep none.
  * VO4 -- ce9 of the same suite, for the same reason: its assertions unchanged, the stand-in taking the word, and
    every run of the check asked to keep none.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import os
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_vo1_through_the_real_check": 8.0,
    "test_vo2_the_executor_keeps_none_when_told": 8.0,
    "test_vo3_the_blocks_of_an_answer_run_only_when_automatic_checking_is_on_and_keep_no_output": 2.0,
    "test_vo4_code_that_never_ran_is_never_handed_to_the_model_to_fix_and_no_run_keeps_an_output": 2.0,
}

_CHECKING = {"code_execution": True, "code_auto_verify": True}


def _image(words):
    """A run that writes ``words`` to an image file of its workspace, then says so."""
    return f"open('chart.png', 'wb').write({words!r})\nprint('drawn')\n"


def _check_window(tmp_path, prefs):
    """ce's window -- code_executor and file_tools over a byte-exact copy of sandbox_manager whose import-time singleton
    is off -- with the real verification engine over them; the structured output engine is not reached."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *a, **k: sqlite3.connect(path)
    loaded, restore = isolate(
        targets={
            "opti_oignon.process_secrets": source("process_secrets.py"),
            "opti_oignon.sandbox_manager": copy,
            "opti_oignon.file_tools": source("file_tools.py"),
            "opti_oignon.code_executor": source("code_executor.py"),
            "opti_oignon.verification": source("verification.py"),
        },
        blocked=("opti_oignon.structured_output",),
        seeded={"opti_oignon.db_utils": db_utils, "opti_oignon.config": ce_t._config(tmp_path, prefs)},
    )
    sm = loaded["opti_oignon.sandbox_manager"]
    if sm.sandbox_manager is not None:
        restore()
        raise AssertionError("the import-time singleton must stay off in this window")
    return sm, loaded["opti_oignon.code_executor"], loaded["opti_oignon.verification"], restore


def _kept(tmp_path):
    """Every file under the code output directory, by its path below it."""
    root = tmp_path / "data" / "exec_outputs"
    return sorted(
        os.path.relpath(os.path.join(folder, name), root)
        for folder, _dirs, names in os.walk(root) for name in names
    )


def test_vo1_through_the_real_check(tmp_path):
    answer = f"Here it is:\n```python\n{_image(b'vo1-checked')}```\n"
    sm, ce, vf, restore = _check_window(tmp_path, _CHECKING)
    try:
        executor = ce.CodeExecutor(sandbox_mgr=ce_t._manager(sm, tmp_path))
        engine = vf.VerificationEngine(structured_engine=None, code_exec=executor, max_iterations=2)
        assert engine.available, "the window does not let an answer's code run"
        checked = engine.verify_response_code_blocks(answer, original_question="vo1")
        fixes = []
        engine._attempt_fix = lambda **kw: fixes.append(kw["error"]) or _image(b"vo1-fixed")
        fixed = engine.verify_and_fix(_image(b"vo1-failed") + "vo1_undefined\n", "python", original_question="vo1",
                                      model="m")
        after_checks, sandboxes = _kept(tmp_path), ce_t._workspaces_left(tmp_path)
        panel = executor.execute(_image(b"vo1-panel"), "python")
    finally:
        restore()
    assert [r.status for r in checked] == ["passed"], [(r.status, r.errors_encountered) for r in checked]
    assert (fixed.status, fixed.iterations, len(fixes)) == ("fixed", 2, 1), (fixed.status, fixed.errors_encountered)
    assert after_checks == [], f"a check of an answer's code kept its images: {after_checks}"
    assert sandboxes == [], f"a check's sandbox stayed: {sandboxes}"
    assert panel.success and len(panel.output_files) == 1, (panel.output_files, panel.error_message)
    assert Path(panel.output_files[0]).read_bytes() == b"vo1-panel", "a panel's run no longer copies its image out"


def test_vo2_the_executor_keeps_none_when_told(tmp_path):
    sm, ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        executor = ce.CodeExecutor(sandbox_mgr=ce_t._manager(sm, tmp_path))
        runs = {conv: executor.execute(_image(b"vo2"), "python", conv_id=conv, keep_outputs=False)
                for conv in (None, "conv-vo2")}
        kept, sandboxes = _kept(tmp_path), ce_t._workspaces_left(tmp_path)
        executor.persistent_mode = True
        persistent = executor.execute(_image(b"vo2-persistent"), "python", conv_id="conv-vo2", keep_outputs=False)
        in_sandbox = executor.list_persistent_files("conv-vo2")
        kept_persistent = _kept(tmp_path)
        executor.reset_persistent_dir("conv-vo2")
    finally:
        restore()
    for conv, result in runs.items():
        assert result.success, (conv, result.error_message, result.stderr[-300:])
        assert result.stdout.strip() == "drawn", (conv, result.stdout)
        assert result.output_files == [], (conv, result.output_files)
    assert kept == [], f"a run told to keep none copied its image out: {kept}"
    assert sandboxes == [], f"a run's sandbox stayed: {sandboxes}"
    assert persistent.success and persistent.output_files == [], (persistent.output_files, persistent.error_message)
    assert "chart.png" in in_sandbox, f"the image left the conversation's sandbox: {in_sandbox}"
    assert kept_persistent == [], f"a persistent run told to keep none copied out: {kept_persistent}"


class _Executor(ce_t._Executor):
    """ce's stand-in, taking the word a check passes on its outputs, and noting it."""

    def __init__(self, enabled=True, result=None):
        super().__init__(enabled=enabled, result=result)
        self.kept = []

    def execute(self, code, language="python", timeout=None, conv_id=None, keep_outputs=True):
        self.kept.append(keep_outputs)
        return super().execute(code, language, timeout, conv_id)


def test_vo3_the_blocks_of_an_answer_run_only_when_automatic_checking_is_on_and_keep_no_output(tmp_path):
    cases = {
        "unset": ({}, True),
        "checking on, execution off": ({"code_auto_verify": True}, False),
        "a string": ({"code_auto_verify": "true"}, True),
        "both on": ({"code_auto_verify": True}, True),
    }
    seen = {}
    kept = []
    for name, (prefs, executing) in cases.items():
        mod, restore = ce_t._verification(tmp_path, prefs)
        try:
            executor = _Executor(enabled=executing)
            engine = mod.VerificationEngine(structured_engine=None, code_exec=executor, max_iterations=1)
            results = engine.verify_response_code_blocks(ce_t._ANSWER, original_question="ce8")
            seen[name] = (engine.available, executor.runs, len(results))
            kept += executor.kept
        finally:
            restore()
    assert seen == {
        "unset": (False, 0, 0),
        "checking on, execution off": (False, 0, 0),
        "a string": (False, 0, 0),
        "both on": (True, 1, 1),
    }, seen
    assert kept == [False], f"a check's run was not asked to keep no output: {kept}"


def test_vo4_code_that_never_ran_is_never_handed_to_the_model_to_fix_and_no_run_keeps_an_output(tmp_path):
    never_ran = SimpleNamespace(
        ran=False, success=False, stdout="", stderr="", return_code=-1,
        error_message="Code execution is disabled. Enable it in Settings.",
    )
    ran_and_failed = SimpleNamespace(
        ran=True, success=False, stdout="", stderr="NameError: name 'ce9' is not defined",
        return_code=1, error_message="",
    )
    seen = {}
    kept = []
    for name, result in (("never ran", never_ran), ("ran and failed", ran_and_failed)):
        mod, restore = ce_t._verification(tmp_path, {"code_auto_verify": True})
        try:
            executor = _Executor(result=result)
            engine = mod.VerificationEngine(structured_engine=None, code_exec=executor, max_iterations=3)
            fixes = []
            engine._attempt_fix = lambda **kw: fixes.append(kw["error"]) or "print('ce9 fixed')\n"
            verdict = engine.verify_and_fix("print(ce9)\n", "python", original_question="ce9", model="m")
            seen[name] = (verdict.status, len(fixes))
            kept += executor.kept
        finally:
            restore()
    assert seen["never ran"] == ("failed", 0), seen
    assert seen["ran and failed"][1] >= 1, f"a real failure must still be offered for a fix: {seen}"
    assert kept and set(kept) == {False}, f"a check's run was not asked to keep no output: {kept}"
