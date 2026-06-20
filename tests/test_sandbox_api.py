#!/usr/bin/env python3
"""
Tests for Sandbox API, Tool Registration, and SandboxToolSession (S73 Step 3).

Covers: API endpoints, tool registration in tool_registry,
SandboxToolSession lifecycle and tool generation, frontend file existence.
"""

import os
import sys
import importlib.util

import pytest

# ---------------------------------------------------------------------------
# Module loading (test isolation)
# ---------------------------------------------------------------------------

_base = os.path.join(os.path.dirname(__file__), os.pardir, "opti_oignon")

# sandbox_manager
_sm_path = os.path.join(_base, "sandbox_manager.py")
_sm_spec = importlib.util.spec_from_file_location("sandbox_manager", _sm_path)
_sm_mod = importlib.util.module_from_spec(_sm_spec)
_sm_spec.loader.exec_module(_sm_mod)

SandboxConfig = _sm_mod.SandboxConfig
SandboxManager = _sm_mod.SandboxManager

# tool_registry
_tr_path = os.path.join(_base, "tool_registry.py")
_tr_spec = importlib.util.spec_from_file_location("tool_registry", _tr_path)
_tr_mod = importlib.util.module_from_spec(_tr_spec)
_tr_spec.loader.exec_module(_tr_mod)

ToolRegistry = _tr_mod.ToolRegistry
ToolDefinition = _tr_mod.ToolDefinition

# Ensure opti_oignon sub-modules are findable
sys.modules["opti_oignon"] = type(sys)("opti_oignon")
sys.modules["opti_oignon.sandbox_manager"] = _sm_mod
sys.modules["opti_oignon.tool_registry"] = _tr_mod

# file_tools
_ft_path = os.path.join(_base, "file_tools.py")
_ft_spec = importlib.util.spec_from_file_location("file_tools", _ft_path)
_ft_mod = importlib.util.module_from_spec(_ft_spec)
_ft_spec.loader.exec_module(_ft_mod)
sys.modules["opti_oignon.file_tools"] = _ft_mod

# sandbox_tools
_st_path = os.path.join(_base, "sandbox_tools.py")
_st_spec = importlib.util.spec_from_file_location("sandbox_tools", _st_path)
_st_mod = importlib.util.module_from_spec(_st_spec)
_st_spec.loader.exec_module(_st_mod)

SandboxToolSession = _st_mod.SandboxToolSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    """SandboxConfig for testing."""
    return SandboxConfig(
        enabled=True,
        isolation_backend="tempdir",
        require_degraded_confirmation=False,
        workspace_base=str(tmp_path / "sandboxes"),
        command_timeout=10,
        max_output_bytes=4096,
        max_stderr_bytes=2048,
        max_concurrent_sessions=3,
        audit_db_path=str(tmp_path / "s3_audit.db"),
        blocked_commands=["sudo", "curl", "wget"],
        blocked_patterns=[r"\.\./\.\.", r"/proc/"],
    )


@pytest.fixture
def mgr(config):
    return SandboxManager(config)


# ---------------------------------------------------------------------------
# Test: SandboxToolSession
# ---------------------------------------------------------------------------

class TestSandboxToolSession:
    """Tests for the session-aware tool wrapper."""

    def test_start_creates_session(self, mgr):
        """Starting a session creates a sandbox."""
        sts = SandboxToolSession(mgr)
        sid = sts.start("s3-test")
        assert sid == "s3-test"
        assert sts.active is True
        assert sts.session_id == "s3-test"
        sts.stop()

    def test_stop_destroys_session(self, mgr):
        """Stopping destroys the sandbox."""
        sts = SandboxToolSession(mgr)
        sts.start("s3-stop")
        result = sts.stop()
        assert result is True
        assert sts.active is False
        assert sts.session_id is None

    def test_auto_session_id(self, mgr):
        """Session ID is auto-generated when None."""
        sts = SandboxToolSession(mgr)
        sid = sts.start()
        assert sid.startswith("tool-session-")
        assert len(sid) > 15
        sts.stop()

    def test_double_start_raises(self, mgr):
        """Starting twice without stop raises."""
        sts = SandboxToolSession(mgr)
        sts.start("double")
        with pytest.raises(RuntimeError, match="already active"):
            sts.start("double2")
        sts.stop()

    def test_bash_works(self, mgr):
        """bash() executes commands."""
        sts = SandboxToolSession(mgr)
        sts.start("bash-test")
        result = sts.bash("echo hello")
        assert "hello" in result
        sts.stop()

    def test_create_and_view(self, mgr):
        """create_file() then view() round-trips."""
        sts = SandboxToolSession(mgr)
        sts.start("cv-test")
        sts.create_file("test.py", "print('hi')\n")
        result = sts.view("test.py")
        assert "print('hi')" in result
        sts.stop()

    def test_str_replace(self, mgr):
        """str_replace() modifies files."""
        sts = SandboxToolSession(mgr)
        sts.start("sr-test")
        sts.create_file("edit.py", "x = 1\n")
        result = sts.str_replace("edit.py", "x = 1", "x = 99")
        assert "successful" in result.lower()
        view_result = sts.view("edit.py")
        assert "x = 99" in view_result
        sts.stop()

    def test_inactive_raises(self, mgr):
        """Using tools without active session raises."""
        sts = SandboxToolSession(mgr)
        with pytest.raises(RuntimeError, match="No active"):
            sts.bash("echo fail")

    def test_inject_files(self, mgr, tmp_path):
        """inject_files works through session wrapper."""
        src = tmp_path / "inject_me.txt"
        src.write_text("injected content")
        sts = SandboxToolSession(mgr)
        sts.start("inject-test")
        paths = sts.inject_files([str(src)])
        assert len(paths) == 1
        result = sts.view("inject_me.txt")
        assert "injected content" in result
        sts.stop()

    def test_extract_files(self, mgr):
        """extract_files lists workspace contents."""
        sts = SandboxToolSession(mgr)
        sts.start("extract-test")
        sts.create_file("a.txt", "a")
        sts.create_file("b.txt", "b")
        files = sts.extract_files()
        names = [f["path"] for f in files]
        assert "a.txt" in names
        assert "b.txt" in names
        sts.stop()

    def test_get_tool_definitions(self, mgr):
        """get_tool_definitions returns 4 simplified tools."""
        sts = SandboxToolSession(mgr)
        sts.start("defs-test")
        defs = sts.get_tool_definitions()
        assert len(defs) == 4
        names = {d.name for d in defs}
        assert names == {"bash", "view", "create_file", "str_replace"}
        # No session_id in parameters
        for d in defs:
            assert "session_id" not in d.parameters
        sts.stop()

    def test_tool_definitions_callable(self, mgr):
        """Tool definitions from get_tool_definitions are callable."""
        sts = SandboxToolSession(mgr)
        sts.start("callable-test")
        defs = sts.get_tool_definitions()
        bash_def = next(d for d in defs if d.name == "bash")
        result = bash_def.handler(command="echo works")
        assert "works" in result
        sts.stop()

    def test_full_coding_cycle(self, mgr):
        """Simulate a full create-run-test cycle."""
        sts = SandboxToolSession(mgr)
        sts.start("cycle-test")

        # Create a Python file
        sts.create_file("add.py", "def add(a, b):\n    return a + b\n")

        # Create a test file
        sts.create_file(
            "test_add.py",
            "from add import add\n"
            "assert add(2, 3) == 5\n"
            "assert add(-1, 1) == 0\n"
            "print('All tests passed')\n",
        )

        # Run the test
        result = sts.bash("python3 test_add.py")
        assert "All tests passed" in result

        # Edit the code
        sts.str_replace("add.py", "return a + b", "return a + b  # addition")

        # Verify the edit
        view_result = sts.view("add.py")
        assert "# addition" in view_result

        # Re-run to confirm nothing broke
        result2 = sts.bash("python3 test_add.py")
        assert "All tests passed" in result2

        sts.stop()


# ---------------------------------------------------------------------------
# Test: Tool registration in tool_registry
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Tests for sandbox tool registration in tool_registry."""

    def test_registry_has_sandbox_tools(self):
        """After import, tool_registry contains sandbox tools."""
        registry = ToolRegistry()
        _tr_mod._register_builtin_tools(registry)
        from opti_oignon.file_tools import get_all_sandbox_tool_definitions
        for td in get_all_sandbox_tool_definitions():
            registry.register(td)

        assert registry.get("sandbox_bash") is not None
        assert registry.get("sandbox_view") is not None
        assert registry.get("sandbox_create_file") is not None
        assert registry.get("sandbox_str_replace") is not None

    def test_sandbox_tools_have_handlers(self):
        """Sandbox tools have non-None handlers."""
        from opti_oignon.file_tools import get_all_sandbox_tool_definitions
        for td in get_all_sandbox_tool_definitions():
            assert td.handler is not None, f"{td.name} has no handler"

    def test_sandbox_tools_have_correct_params(self):
        """Sandbox tools have session_id as required param."""
        from opti_oignon.file_tools import get_all_sandbox_tool_definitions
        for td in get_all_sandbox_tool_definitions():
            assert "session_id" in td.parameters, (
                f"{td.name} missing session_id"
            )
            assert td.parameters["session_id"].required is True

    def test_tools_prompt_includes_sandbox(self):
        """get_tools_prompt includes sandbox tool descriptions."""
        registry = ToolRegistry()
        from opti_oignon.file_tools import get_all_sandbox_tool_definitions
        for td in get_all_sandbox_tool_definitions():
            registry.register(td)
        prompt = registry.get_tools_prompt()
        assert "sandbox_bash" in prompt
        assert "sandbox_view" in prompt
        assert "sandbox_create_file" in prompt
        assert "sandbox_str_replace" in prompt

    def test_simplified_tools_no_session_id(self, mgr):
        """Session-bound tools do NOT have session_id param."""
        sts = SandboxToolSession(mgr, tool_registry=None)
        sts.start("reg-test", allow_degraded=True)
        defs = sts.get_tool_definitions()
        for d in defs:
            assert "session_id" not in d.parameters
        sts.stop()


# ---------------------------------------------------------------------------
# Test: Sandbox mode — unsafe tool lockout (CRITICAL SECURITY)
# ---------------------------------------------------------------------------

class TestSandboxModeSecurity:
    """Tests for sandbox mode disabling unsafe tools.

    This is a CRITICAL security test class. If these tests fail,
    the LLM can bypass the sandbox entirely by calling unsandboxed
    tools (execute_code, read_file, write_file, list_files).
    """

    def _make_registry_with_builtins(self):
        """Create a registry with builtin tools registered."""
        registry = ToolRegistry()
        _tr_mod._register_builtin_tools(registry)
        return registry

    def test_unsafe_tools_defined(self):
        """ToolRegistry.UNSAFE_TOOLS contains the 4 dangerous tools."""
        assert "execute_code" in ToolRegistry.UNSAFE_TOOLS
        assert "read_file" in ToolRegistry.UNSAFE_TOOLS
        assert "write_file" in ToolRegistry.UNSAFE_TOOLS
        assert "list_files" in ToolRegistry.UNSAFE_TOOLS

    def test_unsafe_tools_is_frozen(self):
        """UNSAFE_TOOLS is immutable."""
        assert isinstance(ToolRegistry.UNSAFE_TOOLS, frozenset)

    def test_sandbox_mode_disables_unsafe_tools(self):
        """Entering sandbox mode disables all unsafe tools."""
        registry = self._make_registry_with_builtins()
        # Force-enable unsafe tools for this test (they may be disabled
        # due to missing dependencies in the test environment)
        for name in ToolRegistry.UNSAFE_TOOLS:
            tool = registry.get(name)
            if tool is not None:
                tool.enabled = True

        # Enter sandbox mode
        disabled = registry.set_sandbox_mode(True)
        assert len(disabled) > 0

        # After: unsafe tools should be disabled
        for name in ToolRegistry.UNSAFE_TOOLS:
            tool = registry.get(name)
            if tool is not None:
                assert tool.enabled is False, (
                    f"SECURITY VIOLATION: {name} is still enabled "
                    f"in sandbox mode!"
                )

    def test_sandbox_mode_keeps_safe_tools(self):
        """Sandbox mode does NOT disable web_search or sandbox tools."""
        registry = self._make_registry_with_builtins()
        from opti_oignon.file_tools import get_all_sandbox_tool_definitions
        for td in get_all_sandbox_tool_definitions():
            registry.register(td)

        registry.set_sandbox_mode(True)

        # web_search should still work
        ws = registry.get("web_search")
        if ws is not None:
            # web_search may be disabled due to missing dependency,
            # but NOT due to sandbox mode
            assert "web_search" not in registry._disabled_by_sandbox

        # Sandbox tools should be enabled
        assert registry.is_available("sandbox_bash")
        assert registry.is_available("sandbox_view")
        assert registry.is_available("sandbox_create_file")
        assert registry.is_available("sandbox_str_replace")

    def test_sandbox_mode_exit_restores_tools(self):
        """Exiting sandbox mode re-enables previously disabled tools."""
        registry = self._make_registry_with_builtins()
        # Force-enable so sandbox mode has something to disable
        for name in ToolRegistry.UNSAFE_TOOLS:
            tool = registry.get(name)
            if tool is not None:
                tool.enabled = True

        registry.set_sandbox_mode(True)
        restored = registry.set_sandbox_mode(False)
        assert len(restored) > 0

        # Tools that were disabled by sandbox mode should be re-enabled
        for name in restored:
            tool = registry.get(name)
            assert tool is not None
            assert tool.enabled is True, (
                f"{name} not restored after sandbox mode exit"
            )

    def test_tools_prompt_excludes_unsafe_in_sandbox_mode(self):
        """get_tools_prompt does NOT include unsafe tools in sandbox mode."""
        registry = self._make_registry_with_builtins()
        # Force-enable unsafe tools first
        for name in ToolRegistry.UNSAFE_TOOLS:
            tool = registry.get(name)
            if tool is not None:
                tool.enabled = True

        registry.set_sandbox_mode(True)
        prompt = registry.get_tools_prompt()
        assert "execute_code" not in prompt
        assert "read_file" not in prompt
        assert "write_file" not in prompt
        assert "list_files" not in prompt

    def test_session_activates_sandbox_mode(self, mgr):
        """SandboxToolSession.start() activates sandbox mode."""
        registry = self._make_registry_with_builtins()
        sts = SandboxToolSession(mgr, tool_registry=registry)
        sts.start("sec-test", allow_degraded=True)

        assert registry.sandbox_mode is True

        # Unsafe tools should be disabled
        for name in ToolRegistry.UNSAFE_TOOLS:
            tool = registry.get(name)
            if tool is not None:
                assert tool.enabled is False, (
                    f"SECURITY VIOLATION: {name} enabled during session"
                )

        sts.stop()

    def test_session_stop_deactivates_sandbox_mode(self, mgr):
        """SandboxToolSession.stop() deactivates sandbox mode."""
        registry = self._make_registry_with_builtins()
        sts = SandboxToolSession(mgr, tool_registry=registry)
        sts.start("sec-stop", allow_degraded=True)
        sts.stop()

        assert registry.sandbox_mode is False

    def test_double_enable_idempotent(self):
        """Calling set_sandbox_mode(True) twice is safe."""
        registry = self._make_registry_with_builtins()
        registry.set_sandbox_mode(True)
        disabled2 = registry.set_sandbox_mode(True)
        assert len(disabled2) == 0  # Already in sandbox mode

    def test_double_disable_idempotent(self):
        """Calling set_sandbox_mode(False) twice is safe."""
        registry = ToolRegistry()
        restored = registry.set_sandbox_mode(False)
        assert len(restored) == 0  # Not in sandbox mode

    def test_llm_cannot_reach_host_during_session(self, mgr):
        """Full integration: LLM tools cannot access host filesystem."""
        registry = self._make_registry_with_builtins()
        from opti_oignon.file_tools import get_all_sandbox_tool_definitions
        for td in get_all_sandbox_tool_definitions():
            registry.register(td)

        sts = SandboxToolSession(mgr, tool_registry=registry)
        sts.start("host-check", allow_degraded=True)

        # The ONLY available tools should be sandbox tools + web_search
        available = registry.list_available()
        available_names = {t.name for t in available}

        # CRITICAL: no unsandboxed filesystem/code tools
        assert "execute_code" not in available_names, \
            "SECURITY: execute_code available during sandbox session!"
        assert "read_file" not in available_names, \
            "SECURITY: read_file available during sandbox session!"
        assert "write_file" not in available_names, \
            "SECURITY: write_file available during sandbox session!"
        assert "list_files" not in available_names, \
            "SECURITY: list_files available during sandbox session!"

        # Sandbox tools should be available
        assert "sandbox_bash" in available_names
        assert "sandbox_view" in available_names
        assert "sandbox_create_file" in available_names
        assert "sandbox_str_replace" in available_names

        sts.stop()

    def test_code_verify_blocked_in_sandbox_mode(self):
        """Code verification pipeline is blocked in sandbox mode.

        This prevents the LLM from including malicious code in its
        response that would be auto-executed on the host via
        verification.py -> code_executor.execute().
        """
        registry = self._make_registry_with_builtins()
        registry.set_sandbox_mode(True)

        # The agentic_executor checks registry.sandbox_mode and returns
        # verification_available=False. Verify the flag is set.
        assert registry.sandbox_mode is True

    def test_web_search_disable_config(self, config):
        """disable_web_search_in_sandbox config is supported."""
        config_with_ws = SandboxConfig(
            enabled=True,
            isolation_backend="tempdir",
            require_degraded_confirmation=False,
            workspace_base=config.workspace_base,
            audit_db_path=config.audit_db_path,
            disable_web_search_in_sandbox=True,
        )
        assert config_with_ws.disable_web_search_in_sandbox is True


# ---------------------------------------------------------------------------
# Test: Frontend file existence
# ---------------------------------------------------------------------------

class TestFrontendFiles:
    """Verify frontend files were created."""

    _frontend = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "frontend", "src", "lib",
    )

    def test_sandbox_api_client_exists(self):
        """frontend/src/lib/api/sandbox.ts exists."""
        path = os.path.join(self._frontend, "api", "sandbox.ts")
        assert os.path.isfile(path)

    def test_sandbox_api_client_has_functions(self):
        """sandbox.ts contains expected exports."""
        path = os.path.join(self._frontend, "api", "sandbox.ts")
        with open(path) as fh:
            content = fh.read()
        assert "getSandboxStatus" in content
        assert "createSandbox" in content
        assert "executeSandboxTool" in content
        assert "destroySandbox" in content
        assert "getAuditLog" in content
        assert "confirmDegradedMode" in content

    def test_types_has_sandbox_interfaces(self):
        """types.ts contains sandbox interfaces."""
        path = os.path.join(self._frontend, "types.ts")
        with open(path) as fh:
            content = fh.read()
        assert "SandboxStatusResponse" in content
        assert "SandboxCreateRequest" in content
        assert "SandboxExecuteRequest" in content
        assert "SandboxAuditEntry" in content
        assert "SandboxSessionInfo" in content
        assert "SandboxConfirmDegradedResponse" in content


# ---------------------------------------------------------------------------
# Test: Config file existence
# ---------------------------------------------------------------------------

class TestConfigFiles:
    """Verify config files exist."""

    _config = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "opti_oignon", "config",
    )

    def test_sandbox_yaml_exists(self):
        """config/sandbox.yaml exists."""
        assert os.path.isfile(os.path.join(self._config, "sandbox.yaml"))

    def test_tools_yaml_exists(self):
        """config/tools.yaml exists."""
        assert os.path.isfile(os.path.join(self._config, "tools.yaml"))

    def test_sandbox_yaml_has_bwrap_config(self):
        """sandbox.yaml includes bwrap configuration."""
        with open(os.path.join(self._config, "sandbox.yaml")) as fh:
            content = fh.read()
        assert "bwrap_ro_binds" in content
        assert "bwrap_never_bind" in content
        assert "isolation_backend" in content
        assert "require_degraded_confirmation" in content


# ---------------------------------------------------------------------------
# Test: API routes file structure
# ---------------------------------------------------------------------------

class TestAPIRoutes:
    """Verify API route file and app registration."""

    _api = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "opti_oignon", "api",
    )

    def test_routes_sandbox_exists(self):
        """routes_sandbox.py exists."""
        assert os.path.isfile(
            os.path.join(self._api, "routes_sandbox.py")
        )

    def test_routes_sandbox_has_endpoints(self):
        """routes_sandbox.py has all expected endpoints."""
        with open(os.path.join(self._api, "routes_sandbox.py")) as fh:
            content = fh.read()
        assert "/status" in content
        assert "/create" in content
        assert "/inject" in content
        assert "/files/" in content
        assert "/execute" in content
        assert "/sessions" in content
        assert "/audit" in content
        assert "confirm-degraded" in content

    def test_app_registers_sandbox_router(self):
        """app.py includes the sandbox router."""
        with open(os.path.join(self._api, "app.py")) as fh:
            content = fh.read()
        assert "sandbox_router" in content
        assert "routes_sandbox" in content

    def test_app_uses_version_import(self):
        """app.py imports version from __version__.py."""
        with open(os.path.join(self._api, "app.py")) as fh:
            content = fh.read()
        assert "__version__" in content

    def test_deps_has_sandbox(self):
        """deps.py has sandbox imports."""
        with open(os.path.join(self._api, "deps.py")) as fh:
            content = fh.read()
        assert "SANDBOX_AVAILABLE" in content
        assert "FILE_TOOLS_AVAILABLE" in content
        assert "SANDBOX_TOOLS_AVAILABLE" in content

    def test_schemas_has_sandbox(self):
        """schemas.py has sandbox schemas."""
        with open(os.path.join(self._api, "schemas.py")) as fh:
            content = fh.read()
        assert "SandboxCreateRequest" in content
        assert "SandboxExecuteRequest" in content
        assert "SandboxStatusResponse" in content
        assert "SandboxAuditEntry" in content


# ---------------------------------------------------------------------------
# S116: Copy-Out Schemas + Routes + Frontend Tests
# ---------------------------------------------------------------------------

class TestS116SchemasExist:
    """Verify all S116 Pydantic schemas are defined."""

    _schemas_path = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "opti_oignon", "api", "schemas.py",
    )

    def _content(self):
        with open(self._schemas_path) as fh:
            return fh.read()

    def test_preview_response(self):
        assert "SandboxPreviewResponse" in self._content()

    def test_approve_request(self):
        assert "SandboxApproveRequest" in self._content()

    def test_approve_response(self):
        assert "SandboxApproveResponse" in self._content()

    def test_copy_out_entry(self):
        assert "SandboxCopyOutEntry" in self._content()

    def test_copy_out_response(self):
        assert "SandboxCopyOutResponse" in self._content()

    def test_reject_response(self):
        assert "SandboxRejectResponse" in self._content()

    def test_approval_info_response(self):
        assert "SandboxApprovalInfoResponse" in self._content()

    def test_approval_audit_entry(self):
        assert "SandboxApprovalAuditEntry" in self._content()

    def test_approval_audit_response(self):
        assert "SandboxApprovalAuditResponse" in self._content()

    def test_file_entry_has_approved_field(self):
        """SandboxFileEntry now has 'approved' field."""
        assert "approved: bool" in self._content()

    def test_files_response_has_approval_state(self):
        """SandboxFilesResponse has 'approval_state' field."""
        assert "approval_state: str" in self._content()


class TestS116RoutesExist:
    """Verify S116 route endpoints are defined in routes_sandbox.py."""

    _routes_path = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "opti_oignon", "api", "routes_sandbox.py",
    )

    def _content(self):
        with open(self._routes_path) as fh:
            return fh.read()

    def test_preview_route(self):
        assert "/preview/{session_id}/{path:path}" in self._content()

    def test_download_route(self):
        assert "/download/{session_id}/{path:path}" in self._content()

    def test_approve_route(self):
        assert "/{session_id}/approve" in self._content()

    def test_copy_out_route(self):
        assert "/{session_id}/copy-out" in self._content()

    def test_reject_route(self):
        assert "/{session_id}/reject" in self._content()

    def test_approval_info_route(self):
        assert "/{session_id}/approval" in self._content()

    def test_approval_audit_route(self):
        assert "/{session_id}/approval-audit" in self._content()

    def test_file_response_import(self):
        assert "FileResponse" in self._content()

    def test_s116_docstring(self):
        assert "S116" in self._content()

    def test_default_export_dir(self):
        assert "sandbox_exports" in self._content()

    def test_require_sandbox_helper(self):
        assert "_require_sandbox" in self._content()

    def test_destroy_is_last(self):
        """DELETE /{session_id} is the last route (catch-all ordering)."""
        content = self._content()
        delete_pos = content.rfind('@router.delete("/{session_id}"')
        last_route_pos = content.rfind("@router.")
        assert delete_pos == last_route_pos


class TestS116FrontendFilesExist:
    """Verify S116 frontend files and types exist."""

    _frontend = os.path.join(
        os.path.dirname(__file__), os.pardir, "frontend", "src", "lib",
    )

    def test_sandbox_file_manager_exists(self):
        path = os.path.join(
            self._frontend, "components", "panels",
            "SandboxFileManager.svelte",
        )
        assert os.path.isfile(path)

    def test_sandbox_file_manager_has_approval_ui(self):
        path = os.path.join(
            self._frontend, "components", "panels",
            "SandboxFileManager.svelte",
        )
        with open(path) as fh:
            content = fh.read()
        assert "Approve" in content
        assert "Reject" in content
        assert "preview" in content.lower()

    def test_sandbox_file_manager_no_hex_colors(self):
        """SandboxFileManager.svelte uses only CSS variables for colors."""
        import re
        path = os.path.join(
            self._frontend, "components", "panels",
            "SandboxFileManager.svelte",
        )
        with open(path) as fh:
            content = fh.read()
        # Extract <style> block
        style_match = re.search(r"<style>(.*?)</style>", content, re.DOTALL)
        if style_match:
            style = style_match.group(1)
            # Find color/background properties with raw hex
            hex_props = re.findall(
                r"(?:color|background):\s*#[0-9a-fA-F]{3,8}", style
            )
            assert hex_props == [], f"Hardcoded hex colors found: {hex_props}"

    def test_types_has_s116_interfaces(self):
        path = os.path.join(self._frontend, "types.ts")
        with open(path) as fh:
            content = fh.read()
        for iface in [
            "SandboxPreviewResponse",
            "SandboxApproveRequest",
            "SandboxApproveResponse",
            "SandboxCopyOutResponse",
            "SandboxRejectResponse",
            "SandboxApprovalInfoResponse",
        ]:
            assert iface in content, f"Missing type: {iface}"

    def test_sandbox_api_has_s116_functions(self):
        path = os.path.join(self._frontend, "api", "sandbox.ts")
        with open(path) as fh:
            content = fh.read()
        for func in [
            "previewSandboxFile",
            "getSandboxDownloadUrl",
            "approveSandboxFiles",
            "copyOutSandboxFiles",
            "rejectSandboxFiles",
            "getApprovalInfo",
            "getApprovalAudit",
        ]:
            assert func in content, f"Missing API function: {func}"

    def test_coding_agent_panel_imports_file_manager(self):
        path = os.path.join(
            self._frontend, "components", "panels",
            "CodingAgentPanel.svelte",
        )
        with open(path) as fh:
            content = fh.read()
        assert "SandboxFileManager" in content
