#!/usr/bin/env python3
"""Code execution runs in the sandbox, and only when the user turned it on.

``CodeExecutor`` serves the code panel (``POST /api/code/execute``) and the
automatic check of the code blocks in an answer. It used to start its own
process on this machine, with the server's environment. It is a client of the
sandbox manager now, like every other runner, and each switch it obeys is the
user's own:

  * CE1 -- the module starts no process of its own;
  * CE2 -- the code runs in a workspace of the sandbox manager;
  * CE3 -- the server's environment does not reach the code;
  * CE4 -- without a usable sandbox nothing runs, and the result says the code
    never ran: no sandbox, no session, or a command the sandbox refuses;
  * CE5 -- a sandbox made for one run is gone after it, whatever the outcome;
  * CE6 -- execution follows the ``code_execution`` setting: off when unset,
    when not a boolean, or when the configuration cannot be read;
  * CE7 -- the settings screen offers both switches, wired to the keys the
    server reads, off by default like the server;
  * CE8 -- the blocks of an answer run only when ``code_auto_verify`` is on as
    well;
  * CE9 -- code that never ran is never handed to the model to fix;
  * CE10 -- in persistent mode a conversation keeps one sandbox, and a reset
    ends it;
  * CE11 -- an output image is copied out only as a regular file, never
    through a link.

The sandbox manager is loaded from a byte-exact copy of its file next to a
configuration that switches its import-time singleton off, and the manager
the runs use is built on the tempdir backend over ``tmp_path``: the runs are
real, without the namespace isolation, which only a machine with bubblewrap
can show. Local-only.
"""

import ast
import os
import re
import sqlite3
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_SETTINGS_SCREEN = REPO / "frontend" / "src" / "lib" / "components" / "settings" / "sections" / "ConversationDefaults.svelte"

_ON = {"code_execution": True}


class _Prefs:
    """The configuration seam: the user's preferences, or a failure to read them."""

    def __init__(self, prefs, broken=False):
        self._prefs = dict(prefs or {})
        self._broken = broken

    def get_user_preference(self, key, default=None):
        if self._broken:
            raise RuntimeError("ce: the configuration cannot be read")
        return self._prefs.get(key, default)


def _config(tmp_path, prefs=None, broken=False):
    module = types.ModuleType("opti_oignon.config")
    module.DATA_DIR = tmp_path / "data"
    module.config = _Prefs(prefs, broken)
    return module


def _window(tmp_path, prefs=None, broken=False):
    """code_executor and file_tools over a byte-exact copy of sandbox_manager
    whose configuration switches the import-time singleton off (it would
    reconcile the configured workspace base)."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *a, **k: sqlite3.connect(path)
    loaded, restore = isolate(
        targets={
            "opti_oignon.sandbox_manager": copy,
            "opti_oignon.file_tools": source("file_tools.py"),
            "opti_oignon.code_executor": source("code_executor.py"),
        },
        seeded={
            "opti_oignon.db_utils": db_utils,
            "opti_oignon.config": _config(tmp_path, prefs, broken),
        },
    )
    sm = loaded["opti_oignon.sandbox_manager"]
    if sm.sandbox_manager is not None:
        restore()
        raise AssertionError("the import-time singleton must stay off in this window")
    return sm, loaded["opti_oignon.code_executor"], restore


def _manager(sm, tmp_path, **overrides):
    """A manager on the tempdir backend whose workspaces live under tmp_path."""
    sm._detect_bwrap = lambda: (False, "stand-in: no bwrap")
    settings = dict(
        isolation_backend="auto",
        strict_mode=False,
        require_degraded_confirmation=False,
        workspace_base=str(tmp_path / "workspaces"),
        reconcile_on_start=False,
    )
    settings.update(overrides)
    return sm.SandboxManager(sm.SandboxConfig(**settings))


def _workspaces_left(tmp_path):
    base = tmp_path / "workspaces"
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


# ---------------------------------------------------------------------------
# CE1 -- no process of its own
# ---------------------------------------------------------------------------
_SPAWN_MODULES = ("subprocess", "multiprocessing", "pty", "pexpect")
_OS_SPAWN = re.compile(r"system|popen|fork|forkpty|posix_spawnp?|exec[lv]p?e?|spawn[lv]p?e?")


def test_ce1_the_module_starts_no_process_of_its_own():
    tree = ast.parse(source("code_executor.py").read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in _SPAWN_MODULES]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in _SPAWN_MODULES:
                found.append(module)
            if module == "os":
                found += [f"os.{a.name}" for a in node.names if _OS_SPAWN.fullmatch(a.name)]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and _OS_SPAWN.fullmatch(node.attr):
                found.append(f"os.{node.attr}")
    assert found == [], f"code_executor.py reaches for a process of its own: {found}"


# ---------------------------------------------------------------------------
# CE2, CE3 -- where the code runs, and with what
# ---------------------------------------------------------------------------
def test_ce2_the_code_runs_in_a_workspace_of_the_sandbox_manager(tmp_path):
    sm, ce, restore = _window(tmp_path, _ON)
    try:
        manager = _manager(sm, tmp_path)
        result = ce.CodeExecutor(sandbox_mgr=manager).execute(
            "import os\nprint('ce2', 6 * 7)\nprint(os.getcwd())\n", "python",
        )
    finally:
        restore()
    assert result.success, (result.error_message, result.stderr[-300:])
    lines = result.stdout.splitlines()
    assert lines[0] == "ce2 42", lines
    base = os.path.realpath(tmp_path / "workspaces")
    assert os.path.realpath(lines[1]).startswith(base + os.sep), f"ran in {lines[1]}, not under {base}"


def test_ce3_the_servers_environment_does_not_reach_the_code(tmp_path, monkeypatch):
    monkeypatch.setenv("OO_CE3_SERVER_SECRET", "ce3-server-secret")
    sm, ce, restore = _window(tmp_path, _ON)
    try:
        manager = _manager(sm, tmp_path)
        result = ce.CodeExecutor(sandbox_mgr=manager).execute(
            "import os\nprint(os.environ.get('OO_CE3_SERVER_SECRET', 'ce3-absent'))\n", "python",
        )
    finally:
        restore()
    assert result.success, (result.error_message, result.stderr[-300:])
    assert "ce3-server-secret" not in result.stdout + result.stderr, "the server's environment reached the code"
    assert result.stdout.strip() == "ce3-absent", result.stdout


# ---------------------------------------------------------------------------
# CE4, CE5 -- no sandbox, no run; one sandbox, one run
# ---------------------------------------------------------------------------
class _NoSession:
    """A sandbox manager that has no session to give."""

    def create_sandbox(self, *args, **kwargs):
        raise RuntimeError("ce4: the sandbox is at capacity")

    def execute_command(self, *args, **kwargs):
        raise AssertionError("ce4: asked to run without a session")


def _refuse_every_spawn(monkeypatch, seen):
    def refuse(name):
        def _refuse(*args, **kwargs):
            seen.append(name)
            raise AssertionError(f"ce4: {name} was asked to start a process")
        return _refuse
    monkeypatch.setattr(subprocess.Popen, "__init__", refuse("Popen"))
    monkeypatch.setattr(os, "fork", refuse("fork"))
    monkeypatch.setattr(os, "system", refuse("system"))
    if hasattr(os, "posix_spawn"):
        monkeypatch.setattr(os, "posix_spawn", refuse("posix_spawn"))


def test_ce4_without_a_usable_sandbox_nothing_runs(tmp_path, monkeypatch):
    sm, ce, restore = _window(tmp_path, _ON)
    try:
        refusing = _manager(sm, tmp_path)
        executors = {
            "no sandbox": ce.CodeExecutor(),
            "no session": ce.CodeExecutor(sandbox_mgr=_NoSession()),
            "refused command": ce.CodeExecutor(sandbox_mgr=refusing),
        }
        spawns = []
        _refuse_every_spawn(monkeypatch, spawns)
        results = {
            name: executor.execute("import socket\nprint('ce4 ran')\n", "python")
            for name, executor in executors.items()
        }
        monkeypatch.undo()
        left = refusing.active_session_count
    finally:
        restore()
    assert spawns == [], f"a process was asked for: {spawns}"
    for name, result in results.items():
        assert "ce4 ran" not in result.stdout, f"{name}: the code ran"
        assert result.success is False and result.ran is False, f"{name}: {result}"
        assert result.error_message, f"{name}: a refusal must say why"
    assert "sandbox" in results["no sandbox"].error_message.lower(), results["no sandbox"].error_message
    assert "capacity" in results["no session"].error_message, results["no session"].error_message
    assert "script.py" in results["refused command"].error_message, results["refused command"].error_message
    assert left == 0, f"{left} sandbox(es) left after a refused run"


def test_ce5_a_sandbox_made_for_one_run_is_gone_after_it(tmp_path):
    sm, ce, restore = _window(tmp_path, _ON)
    try:
        manager = _manager(sm, tmp_path)
        executor = ce.CodeExecutor(sandbox_mgr=manager)
        outcomes, left = {}, {}
        for name, code, timeout in (
            ("passes", "print('ce5')\n", None),
            ("fails", "raise SystemExit(3)\n", None),
            ("times out", "import time\ntime.sleep(30)\n", 1),
        ):
            outcomes[name] = executor.execute(code, "python", timeout=timeout)
            left[name] = (manager.active_session_count, _workspaces_left(tmp_path))

        def broken(*args, **kwargs):
            raise RuntimeError("ce5: the run broke")

        manager.execute_command = broken
        outcomes["breaks"] = executor.execute("print('ce5')\n", "python")
        left["breaks"] = (manager.active_session_count, _workspaces_left(tmp_path))
    finally:
        restore()
    assert outcomes["passes"].success and outcomes["passes"].stdout.strip() == "ce5", outcomes["passes"]
    assert outcomes["fails"].return_code == 3 and outcomes["fails"].ran is True, outcomes["fails"]
    assert outcomes["times out"].error_message.startswith("Timeout"), outcomes["times out"]
    assert outcomes["breaks"].ran is False, outcomes["breaks"]
    assert left == {name: (0, []) for name in outcomes}, left


# ---------------------------------------------------------------------------
# CE6, CE7 -- the user's switch, and the screen that shows it
# ---------------------------------------------------------------------------
def test_ce6_execution_follows_the_users_setting_and_is_off_by_default(tmp_path):
    cases = {
        "unset": ({}, False),
        "on": ({"code_execution": True}, False),
        "a_string": ({"code_execution": "true"}, False),
        "off": ({"code_execution": False}, False),
        "unreadable": ({"code_execution": True}, True),
    }
    seen = {}
    for name, (prefs, broken) in cases.items():
        sm, ce, restore = _window(tmp_path / name, prefs, broken)
        try:
            manager = _manager(sm, tmp_path / name)
            asked = []
            create = manager.create_sandbox
            manager.create_sandbox = lambda *a, **k: asked.append(1) or create(*a, **k)
            executor = ce.CodeExecutor(sandbox_mgr=manager)
            result = executor.execute("print('ce6')\n", "python")
            seen[name] = (executor.enabled, result.stdout.strip(), len(asked))
        finally:
            restore()
    assert seen == {
        "unset": (False, "", 0),
        "on": (True, "ce6", 1),
        "a_string": (False, "", 0),
        "off": (False, "", 0),
        "unreadable": (False, "", 0),
    }, seen


def _assigned(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} assigns no {name}")


def test_ce7_the_settings_screen_offers_both_switches_off_like_the_server():
    executor, verification = source("code_executor.py"), source("verification.py")
    server = {
        _assigned(executor, "CODE_EXECUTION_SETTING"): _assigned(executor, "CODE_EXECUTION_DEFAULT"),
        _assigned(verification, "AUTO_VERIFY_SETTING"): _assigned(verification, "AUTO_VERIFY_DEFAULT"),
    }
    assert server == {"code_execution": False, "code_auto_verify": False}, server
    text = _SETTINGS_SCREEN.read_text(encoding="utf-8")
    for key, var in (("code_execution", "codeExecutionEnabled"), ("code_auto_verify", "codeAutoVerifyEnabled")):
        assert re.search(rf"let\s+{var}\s*=\s*false\s*;", text), f"{var} starts on"
        assert re.search(rf"{var}\s*=\s*\(data\.user\?\.{key}\s+as\s+boolean\)\s*\?\?\s*false\s*;", text), (
            f"an unset {key} reads on"
        )
        assert re.search(rf"updateSetting\('{key}',\s*false\)", text), f"a reset does not turn {key} off"
        assert re.search(rf"bind:checked=\{{{var}\}}", text), f"no switch shows {key}"
        assert re.search(rf"saveSetting\('{key}',\s*{var}\b", text), f"the switch does not save {key}"
        assert not re.search(rf"\b{var}\s*=\s*true\b", text), f"{var} is set on somewhere"
        assert not re.search(rf"updateSetting\('{key}',\s*true\)", text), f"something saves {key} on"


# ---------------------------------------------------------------------------
# CE8, CE9 -- the automatic check of an answer's code
# ---------------------------------------------------------------------------
def _verification(tmp_path, prefs):
    loaded, restore = isolate(
        targets={"opti_oignon.verification": source("verification.py")},
        blocked=("opti_oignon.structured_output", "opti_oignon.code_executor"),
        seeded={"opti_oignon.config": _config(tmp_path, prefs)},
    )
    return loaded["opti_oignon.verification"], restore


class _Executor:
    """A code executor stand-in that counts its runs."""

    def __init__(self, enabled=True, result=None):
        self.enabled = enabled
        self.runs = 0
        self._result = result or SimpleNamespace(
            ran=True, success=True, stdout="ce", stderr="", return_code=0, error_message="",
        )

    def execute(self, code, language="python", timeout=None, conv_id=None):
        self.runs += 1
        return self._result


_ANSWER = "Here it is:\n```python\nprint('ce8 block')\n```\n"


def test_ce8_the_blocks_of_an_answer_run_only_when_automatic_checking_is_on(tmp_path):
    cases = {
        "unset": ({}, True),
        "checking on, execution off": ({"code_auto_verify": True}, False),
        "a string": ({"code_auto_verify": "true"}, True),
        "both on": ({"code_auto_verify": True}, True),
    }
    seen = {}
    for name, (prefs, executing) in cases.items():
        mod, restore = _verification(tmp_path, prefs)
        try:
            executor = _Executor(enabled=executing)
            engine = mod.VerificationEngine(structured_engine=None, code_exec=executor, max_iterations=1)
            results = engine.verify_response_code_blocks(_ANSWER, original_question="ce8")
            seen[name] = (engine.available, executor.runs, len(results))
        finally:
            restore()
    assert seen == {
        "unset": (False, 0, 0),
        "checking on, execution off": (False, 0, 0),
        "a string": (False, 0, 0),
        "both on": (True, 1, 1),
    }, seen


def test_ce9_code_that_never_ran_is_never_handed_to_the_model_to_fix(tmp_path):
    never_ran = SimpleNamespace(
        ran=False, success=False, stdout="", stderr="", return_code=-1,
        error_message="Code execution is disabled. Enable it in Settings.",
    )
    ran_and_failed = SimpleNamespace(
        ran=True, success=False, stdout="", stderr="NameError: name 'ce9' is not defined",
        return_code=1, error_message="",
    )
    seen = {}
    for name, result in (("never ran", never_ran), ("ran and failed", ran_and_failed)):
        mod, restore = _verification(tmp_path, {"code_auto_verify": True})
        try:
            engine = mod.VerificationEngine(
                structured_engine=None, code_exec=_Executor(result=result), max_iterations=3,
            )
            fixes = []
            engine._attempt_fix = lambda **kw: fixes.append(kw["error"]) or "print('ce9 fixed')\n"
            verdict = engine.verify_and_fix("print(ce9)\n", "python", original_question="ce9", model="m")
            seen[name] = (verdict.status, len(fixes))
        finally:
            restore()
    assert seen["never ran"] == ("failed", 0), seen
    assert seen["ran and failed"][1] >= 1, f"a real failure must still be offered for a fix: {seen}"


# ---------------------------------------------------------------------------
# CE10, CE11 -- what stays, and what leaves
# ---------------------------------------------------------------------------
def test_ce10_a_conversation_keeps_one_sandbox_and_a_reset_ends_it(tmp_path):
    probe = "import os\nprint(os.path.exists('ce10.txt'))\nopen('ce10.txt', 'w').write('ce10')\n"
    sm, ce, restore = _window(tmp_path, _ON)
    try:
        manager = _manager(sm, tmp_path)
        executor = ce.CodeExecutor(sandbox_mgr=manager)
        executor.persistent_mode = True
        runs = [
            executor.execute(probe, "python", conv_id=conv).stdout.strip()
            for conv in ("ce10-a", "ce10-a", "ce10-b")
        ]
        listed = executor.list_persistent_files("ce10-a")
        live = manager.active_session_count
        reset = executor.reset_persistent_dir("ce10-a")
        after_reset = manager.active_session_count
        reset_twice = executor.reset_persistent_dir("ce10-a")
        fresh = executor.execute(probe, "python", conv_id="ce10-a").stdout.strip()
        executor.persistent_mode = False
        after_off = manager.active_session_count
    finally:
        restore()
    assert runs == ["False", "True", "False"], runs
    assert listed == ["ce10.txt"], listed
    assert (live, reset, after_reset, reset_twice) == (2, True, 1, False), (live, reset, after_reset, reset_twice)
    assert fresh == "False", "a reset conversation found the files of its previous sandbox"
    assert after_off == 0, f"{after_off} conversation sandbox(es) outlived persistent mode"


def test_ce11_an_output_image_is_copied_out_only_as_a_regular_file(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    secret = host / "ce11-host-secret.txt"
    secret.write_text("ce11-host-bytes", encoding="utf-8")
    code = (
        "import os\n"
        f"os.symlink({str(secret)!r}, 'ce11-link.png')\n"
        "open('ce11-chart.png', 'wb').write(b'ce11-chart-bytes')\n"
    )
    sm, ce, restore = _window(tmp_path, _ON)
    try:
        manager = _manager(sm, tmp_path)
        result = ce.CodeExecutor(sandbox_mgr=manager).execute(code, "python")
    finally:
        restore()
    assert result.success, (result.error_message, result.stderr[-300:])
    copied = {Path(p).name.split("_", 1)[-1]: Path(p).read_bytes() for p in result.output_files}
    assert all(b"ce11-host-bytes" not in data for data in copied.values()), "a host file left through a link"
    assert copied == {"ce11-chart.png": b"ce11-chart-bytes"}, sorted(copied)
    outputs = tmp_path / "data" / "exec_outputs"
    assert all(Path(p).parent == outputs for p in result.output_files), result.output_files
