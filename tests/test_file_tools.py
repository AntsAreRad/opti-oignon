#!/usr/bin/env python3
"""
Tests for Sandboxed File Tools (S73 Step 2).

Covers: sandbox_bash, sandbox_view, sandbox_create_file, sandbox_str_replace.
Each tool is tested for normal operation, error handling, path traversal
rejection, symlink escape prevention, workspace boundary enforcement,
and integration with sandbox_manager.
"""

import os
import tempfile

import pytest

# Direct imports to avoid needing full opti_oignon package resolution
import importlib.util

_sm_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "sandbox_manager.py",
)
_sm_spec = importlib.util.spec_from_file_location("sandbox_manager", _sm_path)
_sm_mod = importlib.util.module_from_spec(_sm_spec)
_sm_spec.loader.exec_module(_sm_mod)

SandboxConfig = _sm_mod.SandboxConfig
SandboxManager = _sm_mod.SandboxManager

_ft_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "file_tools.py",
)
_ft_spec = importlib.util.spec_from_file_location("file_tools", _ft_path)
_ft_mod = importlib.util.module_from_spec(_ft_spec)

# Patch module references before exec so imports resolve
import sys
sys.modules["opti_oignon"] = type(sys)("opti_oignon")
sys.modules["opti_oignon.sandbox_manager"] = _sm_mod
# We need tool_registry types - load them too
_tr_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "tool_registry.py",
)
_tr_spec = importlib.util.spec_from_file_location("tool_registry", _tr_path)
_tr_mod = importlib.util.module_from_spec(_tr_spec)
_tr_spec.loader.exec_module(_tr_mod)
sys.modules["opti_oignon.tool_registry"] = _tr_mod

_ft_spec.loader.exec_module(_ft_mod)

_handle_sandbox_bash = _ft_mod._handle_sandbox_bash
_handle_sandbox_view = _ft_mod._handle_sandbox_view
_handle_sandbox_create_file = _ft_mod._handle_sandbox_create_file
_handle_sandbox_str_replace = _ft_mod._handle_sandbox_str_replace
get_sandbox_bash_definition = _ft_mod.get_sandbox_bash_definition
get_sandbox_view_definition = _ft_mod.get_sandbox_view_definition
get_sandbox_create_file_definition = _ft_mod.get_sandbox_create_file_definition
get_sandbox_str_replace_definition = _ft_mod.get_sandbox_str_replace_definition
get_all_sandbox_tool_definitions = _ft_mod.get_all_sandbox_tool_definitions
ToolDefinition = _tr_mod.ToolDefinition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox_env(tmp_path):
    """Create a SandboxManager + active session for testing."""
    cfg = SandboxConfig(
        enabled=True,
        isolation_backend="tempdir",
        require_degraded_confirmation=False,
        workspace_base=str(tmp_path / "sandboxes"),
        command_timeout=10,
        max_output_bytes=4096,
        max_stderr_bytes=2048,
        max_concurrent_sessions=3,
        audit_db_path=str(tmp_path / "ft_audit.db"),
        blocked_commands=["sudo", "curl", "wget"],
        blocked_patterns=[
            r"\.\./\.\.",
            r"/proc/",
        ],
    )
    mgr = SandboxManager(cfg)
    sess = mgr.create_sandbox("ft-test")
    yield mgr, sess
    try:
        mgr.destroy_sandbox("ft-test")
    except Exception:
        pass


@pytest.fixture
def mgr(sandbox_env):
    """Just the manager."""
    return sandbox_env[0]


@pytest.fixture
def sid(sandbox_env):
    """Just the session ID."""
    return sandbox_env[1].session_id


@pytest.fixture
def workspace(sandbox_env):
    """Workspace path."""
    return sandbox_env[1].workspace_path


# ---------------------------------------------------------------------------
# Helper: create a file in workspace
# ---------------------------------------------------------------------------

def _make_file(workspace, name, content):
    """Create a file in the workspace for test setup."""
    path = os.path.join(workspace, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# Test: sandbox_bash tool
# ---------------------------------------------------------------------------

class TestSandboxBash:
    """Tests for the sandbox_bash tool handler."""

    def test_echo_command(self, mgr, sid):
        """Simple echo returns output."""
        result = _handle_sandbox_bash(sid, "echo hello", _sandbox_manager=mgr)
        assert "hello" in result

    def test_blocked_command_reports_blocked(self, mgr, sid):
        """Blocked commands return BLOCKED message."""
        result = _handle_sandbox_bash(sid, "sudo ls", _sandbox_manager=mgr)
        assert "BLOCKED" in result
        assert "security" in result.lower()

    def test_timeout_reports_timeout(self, mgr, sid):
        """Timed-out commands report timeout."""
        result = _handle_sandbox_bash(
            sid, "sleep 60", timeout=1, _sandbox_manager=mgr
        )
        assert "TIMEOUT" in result

    def test_nonexistent_session(self, mgr):
        """Nonexistent session returns error."""
        result = _handle_sandbox_bash(
            "ghost", "echo hi", _sandbox_manager=mgr
        )
        assert "Error" in result

    def test_no_manager(self):
        """Passing a non-functional manager returns error."""
        # Save and clear the module-level default
        saved = _ft_mod._default_sandbox_manager
        _ft_mod._default_sandbox_manager = None
        try:
            result = _handle_sandbox_bash(
                "x", "echo", _sandbox_manager=None
            )
            assert "not available" in result
        finally:
            _ft_mod._default_sandbox_manager = saved

    def test_stderr_included(self, mgr, sid):
        """stderr is included in result."""
        result = _handle_sandbox_bash(
            sid, "echo err >&2", _sandbox_manager=mgr
        )
        assert "STDERR" in result
        assert "err" in result

    def test_return_code_on_failure(self, mgr, sid):
        """Non-zero exit code is reported."""
        result = _handle_sandbox_bash(
            sid, "exit 42", _sandbox_manager=mgr
        )
        assert "42" in result


# ---------------------------------------------------------------------------
# Test: sandbox_view tool
# ---------------------------------------------------------------------------

class TestSandboxView:
    """Tests for the sandbox_view tool handler."""

    def test_view_file(self, mgr, sid, workspace):
        """View a file returns content with line numbers."""
        _make_file(workspace, "hello.py", "print('hello')\nprint('world')\n")
        result = _handle_sandbox_view(sid, "hello.py", _sandbox_manager=mgr)
        assert "hello.py" in result
        assert "2 lines" in result
        assert "print('hello')" in result
        assert "print('world')" in result

    def test_view_with_line_range(self, mgr, sid, workspace):
        """View with line range returns only specified lines."""
        content = "\n".join(f"line {i}" for i in range(1, 21))
        _make_file(workspace, "lines.txt", content)
        result = _handle_sandbox_view(
            sid, "lines.txt", start_line=5, end_line=8,
            _sandbox_manager=mgr,
        )
        assert "line 5" in result
        assert "line 8" in result
        assert "line 4" not in result
        assert "line 9" not in result
        assert "showing lines 5-8" in result

    def test_view_directory(self, mgr, sid, workspace):
        """View a directory returns listing."""
        _make_file(workspace, "src/main.py", "# main")
        _make_file(workspace, "src/utils.py", "# utils")
        result = _handle_sandbox_view(sid, "src", _sandbox_manager=mgr)
        assert "[DIR]" in result or "[FILE]" in result
        assert "main.py" in result
        assert "utils.py" in result

    def test_view_nonexistent_file(self, mgr, sid):
        """Viewing nonexistent file returns error."""
        result = _handle_sandbox_view(
            sid, "nope.txt", _sandbox_manager=mgr
        )
        assert "Error" in result or "not found" in result

    def test_view_path_traversal_rejected(self, mgr, sid):
        """Path traversal is rejected."""
        result = _handle_sandbox_view(
            sid, "../../../etc/passwd", _sandbox_manager=mgr
        )
        assert "Error" in result

    def test_view_absolute_outside_rejected(self, mgr, sid):
        """Absolute path outside sandbox is rejected."""
        result = _handle_sandbox_view(
            sid, "/etc/passwd", _sandbox_manager=mgr
        )
        assert "Error" in result

    def test_view_workspace_prefix(self, mgr, sid, workspace):
        """Paths with /workspace/ prefix work."""
        _make_file(workspace, "prefixed.txt", "content here")
        result = _handle_sandbox_view(
            sid, "/workspace/prefixed.txt", _sandbox_manager=mgr
        )
        assert "content here" in result

    def test_view_symlink_escape(self, mgr, sid, workspace):
        """Symlinks pointing outside workspace are rejected."""
        link_path = os.path.join(workspace, "escape")
        os.symlink("/etc", link_path)
        result = _handle_sandbox_view(
            sid, "escape", _sandbox_manager=mgr
        )
        assert "Error" in result

    def test_view_empty_directory(self, mgr, sid, workspace):
        """Viewing an empty directory reports it."""
        os.makedirs(os.path.join(workspace, "empty_dir"))
        result = _handle_sandbox_view(
            sid, "empty_dir", _sandbox_manager=mgr
        )
        assert "Empty" in result or "0 entries" in result.lower()

    def test_no_manager(self):
        """Passing a non-functional manager returns error."""
        saved = _ft_mod._default_sandbox_manager
        _ft_mod._default_sandbox_manager = None
        try:
            result = _handle_sandbox_view("x", "f.txt", _sandbox_manager=None)
            assert "not available" in result
        finally:
            _ft_mod._default_sandbox_manager = saved


# ---------------------------------------------------------------------------
# Test: sandbox_create_file tool
# ---------------------------------------------------------------------------

class TestSandboxCreateFile:
    """Tests for the sandbox_create_file tool handler."""

    def test_create_simple_file(self, mgr, sid, workspace):
        """Creating a file writes it to the workspace."""
        result = _handle_sandbox_create_file(
            sid, "new.py", "print('new')", _sandbox_manager=mgr
        )
        assert "created" in result.lower()
        assert os.path.isfile(os.path.join(workspace, "new.py"))
        with open(os.path.join(workspace, "new.py")) as fh:
            assert fh.read() == "print('new')"

    def test_create_in_subdirectory(self, mgr, sid, workspace):
        """Creating a file in a subdirectory creates parent dirs."""
        result = _handle_sandbox_create_file(
            sid, "src/lib/module.py", "# module",
            _sandbox_manager=mgr,
        )
        assert "created" in result.lower()
        full = os.path.join(workspace, "src", "lib", "module.py")
        assert os.path.isfile(full)

    def test_overwrite_existing(self, mgr, sid, workspace):
        """Creating a file that exists overwrites it."""
        _make_file(workspace, "exist.txt", "old")
        result = _handle_sandbox_create_file(
            sid, "exist.txt", "new", _sandbox_manager=mgr
        )
        assert "created" in result.lower()
        with open(os.path.join(workspace, "exist.txt")) as fh:
            assert fh.read() == "new"

    def test_create_path_traversal_rejected(self, mgr, sid):
        """Path traversal is rejected."""
        result = _handle_sandbox_create_file(
            sid, "../../../tmp/evil.py", "bad",
            _sandbox_manager=mgr,
        )
        assert "Error" in result

    def test_create_absolute_outside_rejected(self, mgr, sid):
        """Absolute path outside sandbox is rejected."""
        result = _handle_sandbox_create_file(
            sid, "/tmp/evil.py", "bad", _sandbox_manager=mgr,
        )
        assert "Error" in result

    def test_create_workspace_prefix(self, mgr, sid, workspace):
        """Paths with /workspace/ prefix work."""
        result = _handle_sandbox_create_file(
            sid, "/workspace/prefixed.py", "ok",
            _sandbox_manager=mgr,
        )
        assert "created" in result.lower()
        assert os.path.isfile(os.path.join(workspace, "prefixed.py"))

    def test_create_reports_size(self, mgr, sid):
        """Result includes byte count."""
        content = "x" * 100
        result = _handle_sandbox_create_file(
            sid, "sized.txt", content, _sandbox_manager=mgr
        )
        assert "100" in result

    def test_no_manager(self):
        """Passing a non-functional manager returns error."""
        saved = _ft_mod._default_sandbox_manager
        _ft_mod._default_sandbox_manager = None
        try:
            result = _handle_sandbox_create_file(
                "x", "f.py", "c", _sandbox_manager=None
            )
            assert "not available" in result
        finally:
            _ft_mod._default_sandbox_manager = saved


# ---------------------------------------------------------------------------
# Test: sandbox_str_replace tool
# ---------------------------------------------------------------------------

class TestSandboxStrReplace:
    """Tests for the sandbox_str_replace tool handler."""

    def test_simple_replace(self, mgr, sid, workspace):
        """Simple replacement works."""
        _make_file(workspace, "code.py", "x = 1\ny = 2\nz = 3\n")
        result = _handle_sandbox_str_replace(
            sid, "code.py", "y = 2", "y = 42",
            _sandbox_manager=mgr,
        )
        assert "successful" in result.lower()
        with open(os.path.join(workspace, "code.py")) as fh:
            content = fh.read()
        assert "y = 42" in content
        assert "y = 2" not in content

    def test_delete_string(self, mgr, sid, workspace):
        """Replacement with empty string deletes the match."""
        _make_file(workspace, "del.py", "keep\nremove_this\nkeep\n")
        result = _handle_sandbox_str_replace(
            sid, "del.py", "remove_this\n", "",
            _sandbox_manager=mgr,
        )
        assert "deletion" in result.lower() or "successful" in result.lower()
        with open(os.path.join(workspace, "del.py")) as fh:
            assert "remove_this" not in fh.read()

    def test_string_not_found(self, mgr, sid, workspace):
        """Non-existent string returns error."""
        _make_file(workspace, "nf.py", "hello world")
        result = _handle_sandbox_str_replace(
            sid, "nf.py", "nonexistent", "x",
            _sandbox_manager=mgr,
        )
        assert "not found" in result.lower()

    def test_string_found_multiple_times(self, mgr, sid, workspace):
        """String found multiple times returns error."""
        _make_file(workspace, "dup.py", "foo\nbar\nfoo\n")
        result = _handle_sandbox_str_replace(
            sid, "dup.py", "foo", "baz",
            _sandbox_manager=mgr,
        )
        assert "2 times" in result

    def test_empty_old_str(self, mgr, sid, workspace):
        """Empty old_str returns error."""
        _make_file(workspace, "empty.py", "content")
        result = _handle_sandbox_str_replace(
            sid, "empty.py", "", "new",
            _sandbox_manager=mgr,
        )
        assert "Error" in result

    def test_file_not_found(self, mgr, sid):
        """File not found returns error."""
        result = _handle_sandbox_str_replace(
            sid, "ghost.py", "a", "b",
            _sandbox_manager=mgr,
        )
        assert "not found" in result.lower()

    def test_path_traversal_rejected(self, mgr, sid):
        """Path traversal is rejected."""
        result = _handle_sandbox_str_replace(
            sid, "../../../etc/passwd", "root", "hacked",
            _sandbox_manager=mgr,
        )
        assert "Error" in result

    def test_multiline_replace(self, mgr, sid, workspace):
        """Multi-line replacement works."""
        _make_file(workspace, "ml.py", "def foo():\n    pass\n\ndef bar():\n    pass\n")
        result = _handle_sandbox_str_replace(
            sid, "ml.py",
            "def foo():\n    pass",
            "def foo():\n    return 42",
            _sandbox_manager=mgr,
        )
        assert "successful" in result.lower()
        with open(os.path.join(workspace, "ml.py")) as fh:
            content = fh.read()
        assert "return 42" in content

    def test_no_manager(self):
        """Passing a non-functional manager returns error."""
        saved = _ft_mod._default_sandbox_manager
        _ft_mod._default_sandbox_manager = None
        try:
            result = _handle_sandbox_str_replace(
                "x", "f.py", "a", "b", _sandbox_manager=None
            )
            assert "not available" in result
        finally:
            _ft_mod._default_sandbox_manager = saved


# ---------------------------------------------------------------------------
# Test: ToolDefinition factories
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    """Tests for ToolDefinition factory functions."""

    def test_bash_definition_structure(self, mgr):
        """sandbox_bash definition has correct structure."""
        td = get_sandbox_bash_definition(mgr)
        assert td.name == "sandbox_bash"
        assert "session_id" in td.parameters
        assert "command" in td.parameters
        assert "timeout" in td.parameters
        assert td.parameters["session_id"].required is True
        assert td.parameters["timeout"].required is False
        assert td.handler is not None

    def test_view_definition_structure(self, mgr):
        """sandbox_view definition has correct structure."""
        td = get_sandbox_view_definition(mgr)
        assert td.name == "sandbox_view"
        assert "session_id" in td.parameters
        assert "path" in td.parameters
        assert "start_line" in td.parameters
        assert "end_line" in td.parameters

    def test_create_file_definition_structure(self, mgr):
        """sandbox_create_file definition has correct structure."""
        td = get_sandbox_create_file_definition(mgr)
        assert td.name == "sandbox_create_file"
        assert "session_id" in td.parameters
        assert "path" in td.parameters
        assert "content" in td.parameters

    def test_str_replace_definition_structure(self, mgr):
        """sandbox_str_replace definition has correct structure."""
        td = get_sandbox_str_replace_definition(mgr)
        assert td.name == "sandbox_str_replace"
        assert "session_id" in td.parameters
        assert "path" in td.parameters
        assert "old_str" in td.parameters
        assert "new_str" in td.parameters
        assert td.parameters["new_str"].required is False

    def test_get_all_definitions(self, mgr):
        """get_all_sandbox_tool_definitions returns 4 tools."""
        defs = get_all_sandbox_tool_definitions(mgr)
        assert len(defs) == 4
        names = {d.name for d in defs}
        assert names == {
            "sandbox_bash",
            "sandbox_view",
            "sandbox_create_file",
            "sandbox_str_replace",
        }

    def test_definitions_are_callable(self, mgr, sid, workspace):
        """Tool handlers from definitions are callable end-to-end."""
        _make_file(workspace, "callable.txt", "test content")
        td = get_sandbox_view_definition(mgr)
        result = td.handler(session_id=sid, path="callable.txt")
        assert "test content" in result

    def test_bash_definition_handler_works(self, mgr, sid):
        """bash definition handler executes commands."""
        td = get_sandbox_bash_definition(mgr)
        result = td.handler(session_id=sid, command="echo works")
        assert "works" in result

    def test_create_then_view_integration(self, mgr, sid, workspace):
        """Create a file then view it via definitions."""
        create_td = get_sandbox_create_file_definition(mgr)
        view_td = get_sandbox_view_definition(mgr)

        create_td.handler(
            session_id=sid,
            path="integration.py",
            content="# integration test\npass\n",
        )
        result = view_td.handler(session_id=sid, path="integration.py")
        assert "integration test" in result
        assert "2 lines" in result

    def test_create_then_str_replace_integration(self, mgr, sid, workspace):
        """Create a file then replace content via definitions."""
        create_td = get_sandbox_create_file_definition(mgr)
        replace_td = get_sandbox_str_replace_definition(mgr)
        view_td = get_sandbox_view_definition(mgr)

        create_td.handler(
            session_id=sid,
            path="replace_test.py",
            content="value = 'old'\n",
        )
        replace_td.handler(
            session_id=sid,
            path="replace_test.py",
            old_str="value = 'old'",
            new_str="value = 'new'",
        )
        result = view_td.handler(session_id=sid, path="replace_test.py")
        assert "value = 'new'" in result
        assert "value = 'old'" not in result
