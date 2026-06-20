#!/usr/bin/env python3
"""
S183 / S-01: the bubblewrap command must clear the inherited environment and
set only a minimal, explicit set of variables.

Regression guard for the env leak (S-01) and the elevated key exposure (C-01):
without --clearenv, the sandboxed ``bash -c`` inherited the server's full
environment, which could include an env-provided OPTI_ENCRYPTION_KEY or other
secrets. The fix adds --clearenv plus explicit --setenv for PATH, HOME, TMPDIR,
PWD, LANG, LC_ALL. The command is a pure builder, so this verifies the argv
without requiring bwrap to be installed.
"""

import os
import sys
import types

# Guarded stub: in CI ollama is installed and this is a no-op; locally it lets
# the isolated module load resolve opti_oignon.db_utils without the heavy chain.
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

import importlib.util

_mod_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "sandbox_manager.py",
)
_spec = importlib.util.spec_from_file_location("sandbox_manager_s183", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_build_bwrap_command = _mod._build_bwrap_command
SandboxConfig = _mod.SandboxConfig

# Expected minimal environment set after --clearenv.
EXPECTED_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/workspace",
    "TMPDIR": "/tmp",
    "PWD": "/workspace",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}


def _build(command="echo hi", workspace="/tmp/ws", ro_binds=("/usr",)):
    cfg = SandboxConfig(bwrap_ro_binds=list(ro_binds))
    return _build_bwrap_command(command, workspace, cfg)


def _setenv_pairs(cmd):
    pairs = {}
    for i, tok in enumerate(cmd):
        if tok == "--setenv":
            pairs[cmd[i + 1]] = cmd[i + 2]
    return pairs


class TestBwrapCleanEnvironment:
    def test_clearenv_present_exactly_once(self):
        cmd = _build()
        assert cmd.count("--clearenv") == 1

    def test_clearenv_precedes_every_setenv(self):
        cmd = _build()
        ce = cmd.index("--clearenv")
        setenv_indices = [i for i, t in enumerate(cmd) if t == "--setenv"]
        assert setenv_indices, "no --setenv emitted"
        assert ce < min(setenv_indices)

    def test_clearenv_precedes_workspace_bind(self):
        # The clear must happen before mounts/binds so nothing leaks via order.
        cmd = _build()
        assert cmd.index("--clearenv") < cmd.index("--bind")

    def test_minimal_env_values(self):
        pairs = _setenv_pairs(_build())
        for var, value in EXPECTED_ENV.items():
            assert pairs.get(var) == value, f"{var}={pairs.get(var)!r}"

    def test_env_is_a_closed_whitelist(self):
        # No variable beyond the documented minimal set is injected.
        pairs = _setenv_pairs(_build())
        assert set(pairs) == set(EXPECTED_ENV)

    def test_no_secret_variable_passed(self):
        pairs = _setenv_pairs(_build())
        assert "OPTI_ENCRYPTION_KEY" not in pairs
        assert not any(
            "KEY" in v or "SECRET" in v or "TOKEN" in v
            for v in pairs
        )

    def test_command_still_terminates_with_bash_c(self):
        cmd = _build(command="ls -la")
        assert cmd[-3:] == ["bash", "-c", "ls -la"]

    def test_namespace_flags_still_present(self):
        # The env fix must not disturb the existing isolation flags.
        cmd = _build()
        for flag in (
            "--unshare-net", "--unshare-pid", "--new-session",
            "--die-with-parent",
        ):
            assert flag in cmd
