#!/usr/bin/env python3
"""
Tests for the Quick Sandbox module (S117).

Covers: QuickSandboxSession lifecycle, tool handler wrapping,
QuickSandboxManager pool, session expiry, config loading,
status reporting, and edge cases.
"""

import importlib.util
import os
import sys
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Isolated import of quick_sandbox.py (bypass __init__.py chain)
# ---------------------------------------------------------------------------

_mod_path = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "quick_sandbox.py",
)
_spec = importlib.util.spec_from_file_location("quick_sandbox", _mod_path)
_mod = importlib.util.module_from_spec(_spec)

# Patch heavy dependencies before loading
sys.modules.setdefault("yaml", MagicMock())
sys.modules.setdefault("opti_oignon", MagicMock())
sys.modules.setdefault("opti_oignon.sandbox_manager", MagicMock())
sys.modules.setdefault("opti_oignon.file_tools", MagicMock())

_spec.loader.exec_module(_mod)

QuickSandboxSession = _mod.QuickSandboxSession
QuickSandboxManager = _mod.QuickSandboxManager
QuickSandboxConfig = _mod.QuickSandboxConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeSandboxSession:
    """Minimal fake for SandboxSession."""
    def __init__(self, active=True):
        self.active = active


class FakeSandboxManager:
    """Minimal fake for SandboxManager."""
    def __init__(self):
        self._sessions = {}
        self._counter = 0

    def create_sandbox(self, session_id, allow_degraded=False):
        s = FakeSandboxSession(active=True)
        self._sessions[session_id] = s
        self._counter += 1
        return s

    def destroy_sandbox(self, session_id):
        s = self._sessions.pop(session_id, None)
        if s:
            s.active = False
            return True
        return False

    def extract_files(self, session_id):
        return [{"path": "test.py", "size": 42, "modified": 1.0}]


@pytest.fixture
def fake_mgr():
    return FakeSandboxManager()


@pytest.fixture
def qs_config():
    return QuickSandboxConfig(
        enabled=True,
        auto_destroy_minutes=1,
        max_concurrent_quick_sessions=3,
    )


# ---------------------------------------------------------------------------
# QuickSandboxConfig tests
# ---------------------------------------------------------------------------

class TestQuickSandboxConfig:
    def test_defaults(self):
        cfg = QuickSandboxConfig()
        assert cfg.enabled is False
        assert cfg.auto_destroy_minutes == 30
        assert cfg.max_concurrent_quick_sessions == 3

    def test_custom_values(self):
        cfg = QuickSandboxConfig(
            enabled=True,
            auto_destroy_minutes=15,
            max_concurrent_quick_sessions=5,
        )
        assert cfg.enabled is True
        assert cfg.auto_destroy_minutes == 15
        assert cfg.max_concurrent_quick_sessions == 5


# ---------------------------------------------------------------------------
# QuickSandboxSession tests
# ---------------------------------------------------------------------------

class TestQuickSandboxSession:
    def test_creation(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-1",
            sandbox_mgr=fake_mgr,
            auto_destroy_minutes=30,
        )
        assert session.session_id == "test-1"
        assert session.active is False  # Lazy creation
        assert session.files_created == []
        assert session.expired is False

    def test_lazy_sandbox_creation(self, fake_mgr):
        """Sandbox is created on first _ensure_sandbox call."""
        session = QuickSandboxSession(
            session_id="test-lazy",
            sandbox_mgr=fake_mgr,
        )
        assert fake_mgr._counter == 0
        session._ensure_sandbox()
        assert fake_mgr._counter == 1
        assert session.active is True

    def test_ensure_sandbox_idempotent(self, fake_mgr):
        """Multiple _ensure_sandbox calls don't create multiple sandboxes."""
        session = QuickSandboxSession(
            session_id="test-idem",
            sandbox_mgr=fake_mgr,
        )
        session._ensure_sandbox()
        session._ensure_sandbox()
        session._ensure_sandbox()
        assert fake_mgr._counter == 1

    def test_expired_after_timeout(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-expiry",
            sandbox_mgr=fake_mgr,
            auto_destroy_minutes=0,  # 0 minutes = instant expiry
        )
        # Force _last_activity to the past
        session._last_activity = time.time() - 10
        assert session.expired is True

    def test_not_expired_when_active(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-active",
            sandbox_mgr=fake_mgr,
            auto_destroy_minutes=60,
        )
        assert session.expired is False

    def test_handle_execute_code_creates_sandbox(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-exec",
            sandbox_mgr=fake_mgr,
        )
        # Mock _handle_sandbox_bash
        with patch.object(_mod, '_handle_sandbox_bash', return_value="output") as mock_bash:
            result = session.handle_execute_code("print('hello')")
            assert result == "output"
            mock_bash.assert_called_once()
            assert fake_mgr._counter == 1

    def test_handle_execute_code_r_language(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-r",
            sandbox_mgr=fake_mgr,
        )
        with patch.object(_mod, '_handle_sandbox_bash', return_value="R output") as mock_bash:
            result = session.handle_execute_code("cat('hi')", language="r")
            assert result == "R output"
            call_args = mock_bash.call_args
            assert "Rscript" in call_args[0][1]

    def test_handle_execute_code_bash(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-bash",
            sandbox_mgr=fake_mgr,
        )
        with patch.object(_mod, '_handle_sandbox_bash', return_value="bash out") as mock_bash:
            result = session.handle_execute_code("ls -la", language="bash")
            assert result == "bash out"
            assert mock_bash.call_args[0][1] == "ls -la"

    def test_handle_execute_code_unsupported(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-unsup",
            sandbox_mgr=fake_mgr,
        )
        result = session.handle_execute_code("code", language="ruby")
        assert "Unsupported language" in result

    def test_handle_write_file(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-write",
            sandbox_mgr=fake_mgr,
        )
        with patch.object(_mod, '_handle_sandbox_create_file', return_value="Created: test.py"):
            result = session.handle_write_file("test.py", "print('x')")
            assert "Created" in result
            assert "test.py" in session.files_created

    def test_handle_write_file_tracks_unique(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-write-dup",
            sandbox_mgr=fake_mgr,
        )
        with patch.object(_mod, '_handle_sandbox_create_file', return_value="ok"):
            session.handle_write_file("a.py", "x")
            session.handle_write_file("a.py", "y")  # Same file
            session.handle_write_file("b.py", "z")
            assert session.files_created == ["a.py", "b.py"]

    def test_handle_read_file(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-read",
            sandbox_mgr=fake_mgr,
        )
        with patch.object(_mod, '_handle_sandbox_view', return_value="file content"):
            result = session.handle_read_file("test.py")
            assert result == "file content"

    def test_handle_list_files(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-list",
            sandbox_mgr=fake_mgr,
        )
        with patch.object(_mod, '_handle_sandbox_view', return_value="dir listing"):
            result = session.handle_list_files(".")
            assert result == "dir listing"

    def test_get_sandbox_files(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-files",
            sandbox_mgr=fake_mgr,
        )
        session._ensure_sandbox()
        files = session.get_sandbox_files()
        assert len(files) == 1
        assert files[0]["path"] == "test.py"

    def test_get_sandbox_files_inactive(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-inactive",
            sandbox_mgr=fake_mgr,
        )
        files = session.get_sandbox_files()
        assert files == []

    def test_destroy(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-destroy",
            sandbox_mgr=fake_mgr,
        )
        session._ensure_sandbox()
        assert session.active is True
        result = session.destroy()
        assert result is True
        assert session.active is False

    def test_destroy_without_sandbox(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-nodestroy",
            sandbox_mgr=fake_mgr,
        )
        result = session.destroy()
        assert result is False

    def test_no_sandbox_manager(self):
        session = QuickSandboxSession(
            session_id="test-nomgr",
            sandbox_mgr=None,
        )
        # Ensure the default fallback is also None
        with patch.object(_mod, '_default_sandbox_manager', None):
            session._mgr = None
            with pytest.raises(RuntimeError, match="not available"):
                session._ensure_sandbox()

    def test_handle_execute_code_error(self, fake_mgr):
        session = QuickSandboxSession(
            session_id="test-err",
            sandbox_mgr=fake_mgr,
        )
        with patch.object(_mod, '_handle_sandbox_bash', side_effect=Exception("boom")):
            result = session.handle_execute_code("print('x')")
            assert "error" in result.lower()


# ---------------------------------------------------------------------------
# QuickSandboxManager tests
# ---------------------------------------------------------------------------

class TestQuickSandboxManager:
    def test_creation_defaults(self):
        mgr = QuickSandboxManager(
            sandbox_mgr=MagicMock(),
            config=QuickSandboxConfig(),
        )
        assert mgr.enabled is False

    def test_enabled_toggle(self):
        cfg = QuickSandboxConfig(enabled=False)
        mgr = QuickSandboxManager(sandbox_mgr=MagicMock(), config=cfg)
        assert mgr.enabled is False
        mgr.enabled = True
        assert mgr.enabled is True

    def test_available_with_deps(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            assert mgr.available is True

    def test_not_available_without_deps(self, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', False):
            mgr = QuickSandboxManager(
                sandbox_mgr=None, config=qs_config
            )
            assert mgr.available is False

    def test_get_or_create_session(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            session = mgr.get_or_create_session("conv-1")
            assert session.session_id == "conv-1"

    def test_get_or_create_returns_existing(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            s1 = mgr.get_or_create_session("conv-2")
            s2 = mgr.get_or_create_session("conv-2")
            assert s1 is s2

    def test_max_concurrent_sessions(self, fake_mgr):
        cfg = QuickSandboxConfig(
            enabled=True,
            max_concurrent_quick_sessions=2,
            auto_destroy_minutes=60,
        )
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=cfg
            )
            mgr.get_or_create_session("s1")
            mgr.get_or_create_session("s2")
            with pytest.raises(RuntimeError, match="Maximum"):
                mgr.get_or_create_session("s3")

    def test_expired_session_replaced(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            s1 = mgr.get_or_create_session("conv-exp")
            # Force expiry
            s1._last_activity = time.time() - 9999
            s1._auto_destroy_seconds = 1
            s2 = mgr.get_or_create_session("conv-exp")
            assert s2 is not s1

    def test_get_session(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            mgr.get_or_create_session("conv-get")
            assert mgr.get_session("conv-get") is not None
            assert mgr.get_session("nonexistent") is None

    def test_destroy_session(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            s = mgr.get_or_create_session("conv-del")
            s._ensure_sandbox()
            assert mgr.destroy_session("conv-del") is True
            assert mgr.get_session("conv-del") is None

    def test_destroy_nonexistent(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            assert mgr.destroy_session("nope") is False

    def test_cleanup_expired(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            s1 = mgr.get_or_create_session("exp-1")
            s2 = mgr.get_or_create_session("exp-2")
            # Force s1 to be expired
            s1._last_activity = 0
            s1._auto_destroy_seconds = 1
            s1._ensure_sandbox()
            count = mgr.cleanup_expired()
            assert count == 1
            assert mgr.get_session("exp-1") is None
            assert mgr.get_session("exp-2") is not None

    def test_list_sessions(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            mgr.get_or_create_session("list-1")
            mgr.get_or_create_session("list-2")
            sessions = mgr.list_sessions()
            assert len(sessions) == 2
            sids = {s["session_id"] for s in sessions}
            assert "list-1" in sids
            assert "list-2" in sids

    def test_active_session_count(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            assert mgr.active_session_count() == 0
            mgr.get_or_create_session("cnt-1")
            assert mgr.active_session_count() == 1

    def test_get_status(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            status = mgr.get_status()
            assert status["enabled"] is True
            assert status["available"] is True
            assert status["auto_destroy_minutes"] == 1
            assert status["active_sessions"] == 0

    def test_unavailable_raises_on_create(self):
        cfg = QuickSandboxConfig(enabled=True)
        with patch.object(_mod, 'SANDBOX_AVAILABLE', False):
            mgr = QuickSandboxManager(sandbox_mgr=None, config=cfg)
            with pytest.raises(RuntimeError, match="not available"):
                mgr.get_or_create_session("fail")

    def test_auto_generate_id(self, fake_mgr, qs_config):
        with patch.object(_mod, 'SANDBOX_AVAILABLE', True), \
             patch.object(_mod, 'FILE_TOOLS_AVAILABLE', True):
            mgr = QuickSandboxManager(
                sandbox_mgr=fake_mgr, config=qs_config
            )
            session = mgr.get_or_create_session(None)
            assert session.session_id.startswith("qs-")


# ---------------------------------------------------------------------------
# Tool registry integration tests
# ---------------------------------------------------------------------------

class TestToolRegistryIntegration:
    """Tests for set_quick_sandbox_mode on the ToolRegistry."""

    def _load_registry(self):
        """Load tool_registry module in isolation."""
        reg_path = os.path.join(
            os.path.dirname(__file__), os.pardir,
            "opti_oignon", "tool_registry.py",
        )
        spec = importlib.util.spec_from_file_location(
            "tool_registry_test", reg_path
        )
        mod = importlib.util.module_from_spec(spec)
        # Patch deps
        sys.modules.setdefault("opti_oignon.web_search", MagicMock())
        sys.modules.setdefault("opti_oignon.code_executor", MagicMock())
        sys.modules.setdefault("opti_oignon.file_tools", MagicMock(
            FILE_TOOLS_AVAILABLE=False,
            get_all_sandbox_tool_definitions=lambda: [],
        ))
        spec.loader.exec_module(mod)
        return mod

    def test_quick_sandbox_mode_replaces_handlers(self):
        mod = self._load_registry()
        registry = mod.ToolRegistry()
        # Register a fake execute_code tool
        registry.register(mod.ToolDefinition(
            name="execute_code",
            description="test",
            handler=lambda code, language="python", timeout=30: "original",
            enabled=True,
        ))
        original_handler = registry.get("execute_code").handler

        # Create a mock session
        mock_session = MagicMock()
        mock_session.handle_execute_code = MagicMock(return_value="sandboxed")

        affected = registry.set_quick_sandbox_mode(True, session=mock_session)
        assert "execute_code" in affected
        assert registry.quick_sandbox_mode is True

        # Handler should now be different
        new_handler = registry.get("execute_code").handler
        assert new_handler is not original_handler

    def test_quick_sandbox_mode_restores_handlers(self):
        mod = self._load_registry()
        registry = mod.ToolRegistry()
        original = lambda code, language="python", timeout=30: "original"
        registry.register(mod.ToolDefinition(
            name="execute_code",
            description="test",
            handler=original,
            enabled=True,
        ))
        mock_session = MagicMock()
        registry.set_quick_sandbox_mode(True, session=mock_session)
        registry.set_quick_sandbox_mode(False)
        assert registry.quick_sandbox_mode is False
        assert registry.get("execute_code").handler is original

    def test_quick_sandbox_requires_session(self):
        mod = self._load_registry()
        registry = mod.ToolRegistry()
        with pytest.raises(ValueError, match="required"):
            registry.set_quick_sandbox_mode(True, session=None)

    def test_quick_sandbox_idempotent_enable(self):
        mod = self._load_registry()
        registry = mod.ToolRegistry()
        registry.register(mod.ToolDefinition(
            name="execute_code",
            description="test",
            handler=lambda: "x",
            enabled=True,
        ))
        mock_session = MagicMock()
        registry.set_quick_sandbox_mode(True, session=mock_session)
        # Second enable should return empty (already enabled)
        affected = registry.set_quick_sandbox_mode(True, session=mock_session)
        assert affected == []

    def test_quick_sandbox_idempotent_disable(self):
        mod = self._load_registry()
        registry = mod.ToolRegistry()
        # Disable without enable should be no-op
        affected = registry.set_quick_sandbox_mode(False)
        assert affected == []

    def test_quick_sandbox_all_unsafe_tools(self):
        mod = self._load_registry()
        registry = mod.ToolRegistry()
        for name in ["execute_code", "write_file", "read_file", "list_files"]:
            registry.register(mod.ToolDefinition(
                name=name,
                description=f"test {name}",
                handler=lambda: "original",
                enabled=True,
            ))
        mock_session = MagicMock()
        affected = registry.set_quick_sandbox_mode(True, session=mock_session)
        assert set(affected) == {"execute_code", "write_file", "read_file", "list_files"}

    def test_quick_sandbox_session_property(self):
        mod = self._load_registry()
        registry = mod.ToolRegistry()
        assert registry.quick_sandbox_session is None
        mock_session = MagicMock()
        registry.register(mod.ToolDefinition(
            name="execute_code", description="t",
            handler=lambda: "x", enabled=True,
        ))
        registry.set_quick_sandbox_mode(True, session=mock_session)
        assert registry.quick_sandbox_session is mock_session
        registry.set_quick_sandbox_mode(False)
        assert registry.quick_sandbox_session is None


# ---------------------------------------------------------------------------
# Schema & API tests
# ---------------------------------------------------------------------------

class TestSchemas:
    """Test that S117 schemas load and validate correctly."""

    def _load_schemas(self):
        schema_path = os.path.join(
            os.path.dirname(__file__), os.pardir,
            "opti_oignon", "api", "schemas.py",
        )
        spec = importlib.util.spec_from_file_location(
            "schemas_test", schema_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_chat_request_has_quick_sandbox(self):
        mod = self._load_schemas()
        req = mod.ChatRequest(message="hello")
        assert req.quick_sandbox is None
        req2 = mod.ChatRequest(message="hello", quick_sandbox=True)
        assert req2.quick_sandbox is True

    def test_quick_sandbox_status_response(self):
        mod = self._load_schemas()
        resp = mod.QuickSandboxStatusResponse(
            enabled=True, available=True,
            auto_destroy_minutes=30,
            max_concurrent_sessions=3,
            active_sessions=1,
        )
        assert resp.enabled is True
        assert resp.active_sessions == 1

    def test_quick_sandbox_toggle_request(self):
        mod = self._load_schemas()
        req = mod.QuickSandboxToggleRequest(enabled=True)
        assert req.enabled is True

    def test_quick_sandbox_session_info(self):
        mod = self._load_schemas()
        info = mod.QuickSandboxSessionInfo(
            session_id="qs-abc",
            active=True,
            expired=False,
            created_at=1.0,
            files_created=["test.py"],
        )
        assert info.session_id == "qs-abc"
        assert "test.py" in info.files_created


# ---------------------------------------------------------------------------
# Frontend file existence tests
# ---------------------------------------------------------------------------

class TestFrontendFiles:
    """Verify S117 frontend changes exist."""

    _fe_root = os.path.join(
        os.path.dirname(__file__), os.pardir, "frontend", "src"
    )

    def test_chat_options_has_quick_sandbox_store(self):
        path = os.path.join(self._fe_root, "lib", "stores", "chatOptions.ts")
        content = open(path).read()
        assert "quickSandboxEnabled" in content
        assert "quick_sandbox" in content

    def test_chat_store_has_sandbox_meta(self):
        path = os.path.join(self._fe_root, "lib", "stores", "chat.ts")
        content = open(path).read()
        assert "lastSandboxMeta" in content
        assert "sandbox_active" in content
        assert "sandbox_session_id" in content

    def test_chat_control_bar_has_toggle(self):
        path = os.path.join(
            self._fe_root, "lib", "components", "chat",
            "ChatControlBar.svelte",
        )
        content = open(path).read()
        assert "quickSandboxEnabled" in content
        assert "toggleQuickSandbox" in content
        assert "/api/sandbox/quick/toggle" in content

    def test_chat_message_has_inline_sandbox(self):
        path = os.path.join(
            self._fe_root, "lib", "components", "chat",
            "ChatMessage.svelte",
        )
        content = open(path).read()
        assert "SandboxFileManager" in content
        assert "sandboxMeta" in content
        assert "hasSandbox" in content

    def test_types_has_sandbox_meta(self):
        path = os.path.join(self._fe_root, "lib", "types.ts")
        content = open(path).read()
        assert "SandboxMeta" in content
        assert "sandbox_meta" in content


# ---------------------------------------------------------------------------
# Config file tests
# ---------------------------------------------------------------------------

class TestConfigFile:
    """Verify sandbox.yaml has quick_sandbox section."""

    def test_config_has_quick_sandbox(self):
        cfg_path = os.path.join(
            os.path.dirname(__file__), os.pardir,
            "opti_oignon", "config", "sandbox.yaml",
        )
        content = open(cfg_path).read()
        assert "quick_sandbox:" in content
        assert "enabled: false" in content
        assert "auto_destroy_minutes:" in content
        assert "max_concurrent_quick_sessions:" in content


# ---------------------------------------------------------------------------
# Routes integration tests (route existence)
# ---------------------------------------------------------------------------

class TestRoutesSandbox:
    """Verify quick sandbox endpoints are registered."""

    def test_routes_sandbox_has_quick_endpoints(self):
        path = os.path.join(
            os.path.dirname(__file__), os.pardir,
            "opti_oignon", "api", "routes_sandbox.py",
        )
        content = open(path).read()
        assert "/quick/status" in content
        assert "/quick/toggle" in content
        assert "/quick/sessions" in content
        assert "/quick/cleanup" in content
        assert "QuickSandboxStatusResponse" in content

    def test_routes_chat_has_quick_sandbox_integration(self):
        path = os.path.join(
            os.path.dirname(__file__), os.pardir,
            "opti_oignon", "api", "routes_chat.py",
        )
        content = open(path).read()
        assert "quick_sandbox_manager" in content
        assert "set_quick_sandbox_mode" in content
        assert "sandbox_active" in content
        assert "sandbox_session_id" in content
        assert "sandbox_files" in content

    def test_quick_routes_before_catchall(self):
        """Verify /quick/* routes appear before DELETE /{session_id} catch-all."""
        path = os.path.join(
            os.path.dirname(__file__), os.pardir,
            "opti_oignon", "api", "routes_sandbox.py",
        )
        content = open(path).read()
        quick_pos = content.find("/quick/status")
        # The actual catch-all is DELETE /{session_id} (not /files/{session_id})
        catchall_pos = content.find('@router.delete("/{session_id}"')
        assert quick_pos > 0 and catchall_pos > 0
        assert quick_pos < catchall_pos, (
            "/quick/* routes must appear before DELETE /{session_id} catch-all"
        )
