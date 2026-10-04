#!/usr/bin/env python3
"""Under Bulbe the sandbox's strict settings are pinned; Daily keeps them configurable.

Every switch that lowers the bubblewrap path is configuration, and Bulbe is
the mode that promises the strict one. So while Bulbe is on, the sandbox
reads each switch at its strict value, whatever sandbox.yaml or
security.yaml say:

  * BU1 -- strict_mode reads on: without bubblewrap, execution is blocked,
    and the isolation status says strict;
  * BU2 -- a configured ``isolation_backend: tempdir`` gives way to
    bubblewrap wherever bubblewrap runs;
  * BU3 -- the seccomp filter is on and required: one that cannot be built
    refuses the launch instead of running unfiltered;
  * BU4 -- the resource limits are installed on every launch;
  * BU5 -- a degraded sandbox needs the user's confirmation;
  * BU6 -- the mode is read at each use: a switch to Bulbe pins the settings
    without a restart, a mode that cannot be read pins them, and without the
    security-mode module there is no Bulbe to pin for;
  * BU7 -- plugins follow: without bubblewrap they do not start under Bulbe,
    even with strict_mode off in the configuration.

In Daily each contract sees the configured value stand. The sandbox manager
is loaded from a byte-exact copy of its file next to a configuration that
switches its import-time singleton off; the security mode, the seccomp
program and the process launch are stood in at their seams. Local-only.
"""

import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

# Every switch at its lowest: what Bulbe has to raise.
LOWERED = {
    "strict_mode": False,
    "isolation_backend": "tempdir",
    "seccomp_enabled": False,
    "seccomp_required": False,
    "limits_enabled": False,
    "require_degraded_confirmation": False,
}


class _Mode:
    """The security-mode seam: Bulbe or not, or a mode that cannot be read."""

    def __init__(self, bulbe, readable=True):
        self.bulbe = bulbe
        self.readable = readable

    def module(self):
        mode = types.ModuleType("opti_oignon.security_mode")

        def is_bulbe():
            if not self.readable:
                raise RuntimeError("the security mode cannot be read")
            return self.bulbe

        mode.is_bulbe = is_bulbe
        return mode


def _seccomp(builds):
    module = types.ModuleType("opti_oignon.sandbox_seccomp")

    def build_filter_program(arch=None):
        if not builds:
            raise RuntimeError("no syscall table in this contract")
        return b"\x06\x00\x00\x00\x00\x00\xff\x7f"

    module.build_filter_program = build_filter_program
    return module


def _window(tmp_path, mode=None, seccomp_builds=True):
    """The sandbox manager over a copy whose import-time singleton stays off."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *a, **k: sqlite3.connect(path)
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = tmp_path / "data"
    config.CONFIG_DIR = tmp_path / "config"
    config.PROJECT_ROOT = tmp_path / "project"
    seeded = {
        "opti_oignon.db_utils": db_utils,
        "opti_oignon.config": config,
        "opti_oignon.sandbox_seccomp": _seccomp(seccomp_builds),
    }
    if mode is not None:
        seeded["opti_oignon.security_mode"] = mode.module()
    loaded, restore = isolate(targets={"opti_oignon.sandbox_manager": copy}, seeded=seeded)
    sm = loaded["opti_oignon.sandbox_manager"]
    if sm.sandbox_manager is not None:
        restore()
        raise AssertionError("the import-time singleton must stay off in this window")
    return sm, restore


def _manager(sm, tmp_path, bwrap, **overrides):
    sm._detect_bwrap = lambda: (bwrap, "bubblewrap of the contract" if bwrap else "no bubblewrap here")
    settings = dict(LOWERED, **overrides)
    config = sm.SandboxConfig(
        workspace_base=str(tmp_path / "workspaces"),
        audit_db_path=str(tmp_path / "audit.db"),
        reconcile_on_start=False,
        **settings,
    )
    return sm.SandboxManager(config)


def _launches(manager):
    """Stand in for the launch: record each one, run nothing."""
    calls = []

    def spawn(argv, **kwargs):
        calls.append(dict(kwargs, argv=list(argv)))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    manager._spawn_tracked = spawn
    return calls


@pytest.mark.parametrize("bulbe", [True, False])
def test_bu1_strict_mode_reads_on_under_bulbe(tmp_path, bulbe):
    sm, restore = _window(tmp_path, _Mode(bulbe))
    try:
        manager = _manager(sm, tmp_path, bwrap=False)
        strict, blocked = manager.strict_mode, manager.execution_blocked
        status = manager.get_isolation_status()
    finally:
        restore()
    assert (strict, blocked, status["strict_mode"]) == (bulbe, bulbe, bulbe)


@pytest.mark.parametrize("bulbe", [True, False])
def test_bu2_a_configured_tempdir_gives_way_to_bubblewrap_under_bulbe(tmp_path, bulbe):
    sm, restore = _window(tmp_path, _Mode(bulbe))
    try:
        manager = _manager(sm, tmp_path, bwrap=True)
        in_use = manager.bwrap_in_use
        backend = manager.isolation_backend
    finally:
        restore()
    assert in_use is bulbe
    assert backend == (sm.IsolationBackend.BWRAP if bulbe else sm.IsolationBackend.TEMPDIR)


@pytest.mark.parametrize("bulbe", [True, False])
def test_bu3_a_seccomp_filter_that_cannot_be_built_refuses_the_launch_under_bulbe(tmp_path, bulbe):
    sm, restore = _window(tmp_path, _Mode(bulbe), seccomp_builds=False)
    try:
        manager = _manager(sm, tmp_path, bwrap=True, isolation_backend="auto")
        calls = _launches(manager)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        result = manager._run_bwrap("true", str(workspace), 5)
    finally:
        restore()
    assert result.blocked is bulbe
    assert len(calls) == (0 if bulbe else 1)


@pytest.mark.parametrize("bulbe", [True, False])
def test_bu4_the_resource_limits_are_installed_under_bulbe(tmp_path, bulbe):
    sm, restore = _window(tmp_path, _Mode(bulbe))
    try:
        manager = _manager(sm, tmp_path, bwrap=True, isolation_backend="auto")
        calls = _launches(manager)
        workspace = tmp_path / "ws"
        workspace.mkdir()
        manager._run_bwrap("true", str(workspace), 5)
    finally:
        restore()
    assert len(calls) >= 1
    assert (calls[-1]["preexec_fn"] is not None) is bulbe


@pytest.mark.parametrize("bulbe", [True, False])
def test_bu5_a_degraded_sandbox_needs_the_users_confirmation_under_bulbe(tmp_path, bulbe):
    sm, restore = _window(tmp_path, _Mode(bulbe))
    try:
        manager = _manager(sm, tmp_path, bwrap=False)
        try:
            manager.create_sandbox("bu5")
            refused = False
        except RuntimeError as exc:
            refused = "DEGRADED" in str(exc)
    finally:
        restore()
    assert refused is bulbe


def test_bu6_the_mode_is_read_at_each_use(tmp_path):
    mode = _Mode(bulbe=False)
    sm, restore = _window(tmp_path / "live", mode)
    try:
        manager = _manager(sm, tmp_path / "live", bwrap=False)
        before = manager.strict_mode
        mode.bulbe = True
        switched = manager.strict_mode
        mode.readable = False
        unreadable = manager.strict_mode
    finally:
        restore()
    assert (before, switched, unreadable) == (False, True, True)
    sm, restore = _window(tmp_path / "absent", mode=None)
    try:
        absent = _manager(sm, tmp_path / "absent", bwrap=False).strict_mode
    finally:
        restore()
    assert absent is False


@pytest.mark.parametrize("bulbe", [True, False])
def test_bu7_plugins_do_not_start_without_bubblewrap_under_bulbe(tmp_path, bulbe):
    from importlib import import_module

    # Resolved before the window opens: inside it, only seeded modules resolve.
    isolation = import_module("opti_oignon.plugin_isolation")
    sm, restore = _window(tmp_path, _Mode(bulbe))
    try:
        manager = _manager(sm, tmp_path, bwrap=False)
        posture = isolation.resolve_posture(
            types.SimpleNamespace(
                sandbox_manager=manager,
                _HARDCODED_NEVER_BIND=sm._HARDCODED_NEVER_BIND,
            ))
    finally:
        restore()
    assert posture.mode == ("blocked" if bulbe else "direct")
