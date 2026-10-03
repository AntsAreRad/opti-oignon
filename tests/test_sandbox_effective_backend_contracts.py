#!/usr/bin/env python3
"""Effective-backend contracts for the sandbox's isolation checks.

Whether a command runs isolated depends on the backend the manager resolved,
not on whether bubblewrap is installed: with ``isolation_backend: tempdir``
the manager runs commands in a plain temporary directory even when bwrap is
present. Every check that decides on isolation reads the backend in use:

  * EB1 -- strict mode blocks a configured tempdir backend with bwrap
    present, and no runner is reached;
  * EB2 -- ``execution_blocked`` reports that configuration as blocked;
  * EB3 -- the health level names the backend in use: blocked under strict
    mode, tempdir without it, never bwrap;
  * EB4 -- ``bwrap_in_use`` is false for a tempdir backend and true for a
    bwrap backend, bwrap present in both;
  * EB5 -- the agent dispatch's ``sandbox_ready`` refuses a manager that has
    bwrap installed but does not run it;
  * EB6 -- the skills tool's sandbox check refuses it too;
  * EB7 -- note transcription refuses it;
  * EB8 -- note captioning refuses it;
  * EB9 -- the security score credits the sandbox only when bwrap is in use.

The manager module is loaded from a copy of its file next to a configuration
that switches the import-time singleton off, so no test reaches the
configured workspace base. Bubblewrap detection is stood in for; no command
reaches a process. Local-only.
"""

import sqlite3
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _manager_module(tmp_path):
    """sandbox_manager.py loaded from a byte-exact copy whose configuration
    disables the singleton (it would reconcile the configured workspace base)."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = _module("opti_oignon.db_utils", safe_connect=lambda path, *a, **k: sqlite3.connect(path))
    loaded, restore = isolate(
        targets={"opti_oignon.sandbox_manager": copy},
        seeded={"opti_oignon.db_utils": db_utils},
    )
    module = loaded["opti_oignon.sandbox_manager"]
    if module.sandbox_manager is not None:
        restore()
        raise AssertionError("the import-time singleton must stay off in this window")
    return module, restore


def _manager(sm, root, *, backend="tempdir", strict=True):
    sm._detect_bwrap = lambda: (True, "stand-in bwrap")
    config = sm.SandboxConfig(
        isolation_backend=backend,
        strict_mode=strict,
        require_degraded_confirmation=False,
        workspace_base=str(root / "workspaces"),
        reconcile_on_start=False,
    )
    return sm.SandboxManager(config)


def _seam(*, in_use):
    """A manager seam with bwrap installed; ``in_use`` says whether it runs it."""
    return types.SimpleNamespace(bwrap_available=True, bwrap_in_use=in_use)


def test_eb1_strict_mode_blocks_a_configured_tempdir_backend_with_bwrap_present(tmp_path):
    sm, restore = _manager_module(tmp_path)
    try:
        manager = _manager(sm, tmp_path)
        manager.create_sandbox("eb1")
        reached = []
        manager._run_tempdir = lambda *a, **k: reached.append("tempdir") or sm.CommandResult()
        manager._run_bwrap = lambda *a, **k: reached.append("bwrap") or sm.CommandResult()
        result = manager.execute_command("eb1", "echo eb1")
    finally:
        restore()
    assert result.blocked, "a tempdir backend under strict mode ran a command"
    assert reached == [], f"a runner was reached: {reached}"


def test_eb2_execution_blocked_reports_a_configured_tempdir_under_strict_mode(tmp_path):
    sm, restore = _manager_module(tmp_path)
    try:
        blocked = _manager(sm, tmp_path).execution_blocked
    finally:
        restore()
    assert blocked is True, "execution_blocked hid a tempdir backend under strict mode"


def test_eb3_health_level_names_the_backend_in_use(tmp_path):
    sm, restore = _manager_module(tmp_path)
    try:
        strict = _manager(sm, tmp_path / "strict").get_isolation_status()["isolation_level"]
        lenient = _manager(sm, tmp_path / "lenient", strict=False).get_isolation_status()["isolation_level"]
    finally:
        restore()
    assert (strict, lenient) == ("blocked", "tempdir"), (strict, lenient)


def test_eb4_bwrap_in_use_follows_the_resolved_backend(tmp_path):
    sm, restore = _manager_module(tmp_path)
    try:
        tempdir = _manager(sm, tmp_path / "tempdir").bwrap_in_use
        bwrap = _manager(sm, tmp_path / "bwrap", backend="bwrap").bwrap_in_use
    finally:
        restore()
    assert (tempdir, bwrap) == (False, True), (tempdir, bwrap)


def test_eb5_dispatch_sandbox_ready_refuses_bwrap_installed_but_not_in_use():
    seeded = {
        "opti_oignon.agent.allowlists": _module("opti_oignon.agent.allowlists"),
        "opti_oignon.agent.tool_parsing": _module(
            "opti_oignon.agent.tool_parsing",
            ParsedToolCall=object,
            parse_tool_blocks=lambda *a, **k: [],
        ),
    }
    loaded, restore = isolate(
        targets={"opti_oignon.agent.dispatch": source("agent", "dispatch.py")},
        seeded=seeded,
        packages=("opti_oignon.agent",),
    )
    try:
        ready = loaded["opti_oignon.agent.dispatch"].sandbox_ready
        verdicts = (
            ready(types.SimpleNamespace(sandbox_manager=_seam(in_use=False))),
            ready(types.SimpleNamespace(sandbox_manager=_seam(in_use=True))),
        )
    finally:
        restore()
    assert verdicts == (False, True), verdicts


def test_eb6_skills_sandbox_check_refuses_bwrap_installed_but_not_in_use():
    loaded, restore = isolate(
        targets={"opti_oignon.agent.skills": source("agent", "skills.py")},
        packages=("opti_oignon.agent",),
    )
    try:
        ready = loaded["opti_oignon.agent.skills"]._sandbox_ready
        verdicts = (
            ready(types.SimpleNamespace(sandbox_manager=_seam(in_use=False))),
            ready(types.SimpleNamespace(sandbox_manager=_seam(in_use=True))),
        )
    finally:
        restore()
    assert verdicts == (False, True), verdicts


def test_eb7_transcription_refuses_bwrap_installed_but_not_in_use():
    loaded, restore = isolate(
        targets={"opti_oignon.notes.transcription": source("notes", "transcription.py")},
        packages=("opti_oignon.notes",),
    )
    try:
        notes = loaded["opti_oignon.notes.transcription"]
        result = notes.transcribe_attachment(
            "eb7",
            user_id="local",
            store=None,
            blobs=None,
            sandbox=_seam(in_use=False),
            transcriber=lambda *a, **k: "",
        )
        expected = notes.REASON_SANDBOX_UNAVAILABLE
    finally:
        restore()
    assert result.refused and result.reason == expected, (result.refused, result.reason)


def test_eb8_caption_refuses_bwrap_installed_but_not_in_use():
    loaded, restore = isolate(
        targets={"opti_oignon.notes.caption": source("notes", "caption.py")},
        packages=("opti_oignon.notes",),
    )
    try:
        notes = loaded["opti_oignon.notes.caption"]
        result = notes.caption_attachment(
            "eb8",
            user_id="local",
            store=None,
            blobs=None,
            sandbox=_seam(in_use=False),
            captioner=lambda *a, **k: "",
        )
        expected = notes.REASON_SANDBOX_UNAVAILABLE
    finally:
        restore()
    assert result.refused and result.reason == expected, (result.refused, result.reason)


def test_eb9_security_score_credits_the_sandbox_only_when_bwrap_is_in_use():
    def credited(manager):
        seeded = {"opti_oignon.sandbox_manager": _module("opti_oignon.sandbox_manager", sandbox_manager=manager)}
        loaded, restore = isolate(
            targets={"opti_oignon.api.routes_security": source("api", "routes_security.py")},
            seeded=seeded,
            packages=("opti_oignon.api",),
        )
        try:
            _, _, checks = loaded["opti_oignon.api.routes_security"]._compute_security_score()
        finally:
            restore()
        return next(check["passed"] for check in checks if check["name"] == "sandbox_bwrap")

    verdicts = (credited(_seam(in_use=False)), credited(_seam(in_use=True)))
    assert verdicts == (False, True), verdicts
