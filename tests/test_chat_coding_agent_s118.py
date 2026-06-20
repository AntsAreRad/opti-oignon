#!/usr/bin/env python3
"""
Tests for S118 — Chat Coding Agent in Chat.

Tests cover:
  - ChatCodingSession lifecycle (create, resume, destroy)
  - Multi-turn conversation (modify existing code in same sandbox)
  - Directive parsing (--no-test, --plan-only, natural language variants)
  - Adaptive pipeline (skip test, skip plan on follow-up)
  - Context accumulation across turns
  - Streaming events (plan, step, test, fix)
  - Rich LLM callback types (LLMCallContext, LLMCallResult)
  - Session timeout and cleanup
  - ChatCodingManager pool management
  - Config loading from YAML
  - SandboxState context block formatting
  - Frontend file existence and structure
  - Schema additions (ChatRequest.chat_coding, ChatCodingStatusResponse, etc.)
  - Route existence (coding/status, coding/toggle, coding/sessions, etc.)
"""

import importlib.util
import os
import re
import time

import pytest

# ---------------------------------------------------------------------------
# Load modules via importlib to bypass __init__.py chain
# ---------------------------------------------------------------------------

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MOD_PATH = os.path.join(_BASE, "opti_oignon", "chat_coding_agent.py")
_SCHEMA_PATH = os.path.join(_BASE, "opti_oignon", "api", "schemas.py")
_ROUTES_PATH = os.path.join(_BASE, "opti_oignon", "api", "routes_chat.py")
_FRONTEND = os.path.join(_BASE, "frontend", "src", "lib")
_CONFIG_PATH = os.path.join(
    _BASE, "opti_oignon", "config", "coding_agent.yaml"
)
_STORE_PATH = os.path.join(_FRONTEND, "stores", "chatOptions.ts")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cca():
    """Load chat_coding_agent module."""
    return _load_module("chat_coding_agent", _MOD_PATH)


@pytest.fixture(scope="module")
def schemas():
    """Load schemas module."""
    return _load_module("schemas", _SCHEMA_PATH)


# ===================================================================
# Directive Parser
# ===================================================================

class TestDirectiveParser:
    """Test directive parsing from user messages."""

    def test_no_directives(self, cca):
        d = cca.parse_directives("build a Flask API with JWT auth")
        assert d.skip_test is False
        assert d.skip_plan is False
        assert d.skip_fix is False
        assert d.plan_only is False
        assert d.max_fix_retries is None

    def test_flag_no_test(self, cca):
        d = cca.parse_directives("build it --no-test")
        assert d.skip_test is True
        assert "build it" in d.cleaned_message

    def test_flag_notest(self, cca):
        d = cca.parse_directives("implement --notest feature X")
        assert d.skip_test is True

    def test_flag_no_fix(self, cca):
        d = cca.parse_directives("write code --no-fix")
        assert d.skip_fix is True

    def test_flag_plan_only(self, cca):
        d = cca.parse_directives("design an API --plan-only")
        assert d.plan_only is True

    def test_flag_skip_plan(self, cca):
        d = cca.parse_directives("change the port --skip-plan")
        assert d.skip_plan is True

    def test_flag_max_retries(self, cca):
        d = cca.parse_directives("fix this --max-retries 7")
        assert d.max_fix_retries == 7

    def test_natural_skip_test(self, cca):
        d = cca.parse_directives("don't test this, just implement")
        assert d.skip_test is True

    def test_natural_just_write_it(self, cca):
        d = cca.parse_directives("just write it, add a logger")
        assert d.skip_test is True

    def test_natural_skip_testing(self, cca):
        d = cca.parse_directives("skip testing for now")
        assert d.skip_test is True

    def test_natural_no_fix(self, cca):
        d = cca.parse_directives("don't auto-fix anything")
        assert d.skip_fix is True

    def test_natural_plan_only(self, cca):
        d = cca.parse_directives("only plan the architecture")
        assert d.plan_only is True

    def test_natural_try_harder(self, cca):
        d = cca.parse_directives("try harder to fix the bug")
        assert d.max_fix_retries == 5

    def test_natural_skip_planning(self, cca):
        d = cca.parse_directives("skip planning, change port to 8080")
        assert d.skip_plan is True

    def test_multiple_directives(self, cca):
        d = cca.parse_directives("build it --no-test --plan-only")
        assert d.skip_test is True
        assert d.plan_only is True

    def test_cleaned_message_preserves_content(self, cca):
        d = cca.parse_directives("implement a REST API --no-test")
        assert "REST API" in d.cleaned_message
        assert "--no-test" not in d.cleaned_message


# ===================================================================
# SandboxState
# ===================================================================

class TestSandboxState:
    """Test sandbox state context block generation."""

    def test_empty_state(self, cca):
        state = cca.SandboxState()
        assert state.as_context_block() == ""

    def test_files_only(self, cca):
        state = cca.SandboxState(files=["main.py", "test_main.py"])
        block = state.as_context_block()
        assert "[SANDBOX STATE]" in block
        assert "main.py" in block
        assert "test_main.py" in block

    def test_test_passed(self, cca):
        state = cca.SandboxState(
            files=["app.py"], last_test_passed=True,
            last_test_output="2 passed",
        )
        block = state.as_context_block()
        assert "PASSED" in block
        assert "2 passed" in block

    def test_test_failed(self, cca):
        state = cca.SandboxState(
            files=["app.py"], last_test_passed=False,
            last_test_output="1 failed",
        )
        block = state.as_context_block()
        assert "FAILED" in block

    def test_cumulative_summary(self, cca):
        state = cca.SandboxState(
            cumulative_summary="Turn 1: Created Flask API\nTurn 2: Added auth",
        )
        block = state.as_context_block()
        assert "Turn 1" in block
        assert "Turn 2" in block

    def test_last_error(self, cca):
        state = cca.SandboxState(
            files=["x.py"], last_error="ImportError: no module named foo",
        )
        block = state.as_context_block()
        assert "ImportError" in block

    def test_truncated_test_output(self, cca):
        long_output = "x" * 2000
        state = cca.SandboxState(
            files=["a.py"], last_test_passed=False,
            last_test_output=long_output,
        )
        block = state.as_context_block()
        assert "truncated" in block.lower()

    def test_many_files_truncated(self, cca):
        files = [f"file_{i}.py" for i in range(50)]
        state = cca.SandboxState(files=files)
        block = state.as_context_block()
        assert "and 20 more" in block


# ===================================================================
# LLMCallContext and LLMCallResult
# ===================================================================

class TestRichLLMTypes:
    """Test rich LLM callback types."""

    def test_llm_call_context_defaults(self, cca):
        ctx = cca.LLMCallContext()
        assert ctx.images is None
        assert ctx.web_search is False
        assert ctx.think is False
        assert ctx.tools_enabled is True
        assert ctx.conversation_id is None

    def test_llm_call_context_with_values(self, cca):
        ctx = cca.LLMCallContext(
            images=["base64data"],
            web_search=True,
            think=True,
            conversation_id="conv-123",
        )
        assert ctx.images == ["base64data"]
        assert ctx.web_search is True
        assert ctx.think is True
        assert ctx.conversation_id == "conv-123"

    def test_llm_call_result_defaults(self, cca):
        r = cca.LLMCallResult()
        assert r.text == ""
        assert r.tool_calls == []
        assert r.vision_meta == {}
        assert r.plugin_annotations == []
        assert r.thinking == ""
        assert r.error == ""

    def test_llm_call_result_with_data(self, cca):
        r = cca.LLMCallResult(
            text="Hello world",
            tool_calls=[{"tool_name": "write_file"}],
            vision_meta={"delegated": True},
            plugin_annotations=[{"plugin": "fact-checker"}],
            thinking="Let me think...",
        )
        assert r.text == "Hello world"
        assert len(r.tool_calls) == 1
        assert r.vision_meta["delegated"] is True
        assert len(r.plugin_annotations) == 1
        assert "think" in r.thinking.lower()


# ===================================================================
# Configuration
# ===================================================================

class TestConfig:
    """Test configuration loading."""

    def test_default_config(self, cca):
        cfg = cca.ChatCodingConfig()
        assert cfg.enabled is False
        assert cfg.session_timeout_minutes == 60
        assert cfg.max_concurrent_sessions == 3
        assert cfg.max_fix_retries == 3
        assert cfg.auto_test is True
        assert cfg.command_timeout == 30

    def test_load_config_from_yaml(self, cca):
        cfg = cca._load_config()
        # From the YAML we added: chat_coding.enabled = false
        assert cfg.enabled is False
        assert cfg.session_timeout_minutes == 60

    def test_yaml_has_chat_coding_section(self):
        import yaml
        with open(_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "chat_coding" in data
        cc = data["chat_coding"]
        assert cc["enabled"] is False
        assert cc["session_timeout_minutes"] == 60
        assert cc["max_fix_retries"] == 3
        assert cc["auto_test"] is True


# ===================================================================
# ChatCodingSession
# ===================================================================

class TestChatCodingSession:
    """Test ChatCodingSession lifecycle and properties."""

    def test_session_creation(self, cca):
        session = cca.ChatCodingSession(
            conversation_id="test-conv-001",
            sandbox_mgr=None,
            config=cca.ChatCodingConfig(),
        )
        assert session.conversation_id == "test-conv-001"
        assert session.session_id.startswith("cc-test-conv-0")
        assert session.active is False
        assert session.expired is False
        assert session.turn_count == 0

    def test_session_expiry(self, cca):
        cfg = cca.ChatCodingConfig(session_timeout_minutes=0)
        session = cca.ChatCodingSession(
            conversation_id="test-exp",
            sandbox_mgr=None,
            config=cfg,
        )
        # timeout_minutes=0 means 0 seconds => immediately expired
        # Need to set _last_activity in the past
        session._last_activity = time.time() - 1
        assert session.expired is True

    def test_session_not_expired(self, cca):
        session = cca.ChatCodingSession(
            conversation_id="test-alive",
            sandbox_mgr=None,
            config=cca.ChatCodingConfig(session_timeout_minutes=60),
        )
        assert session.expired is False

    def test_session_status(self, cca):
        session = cca.ChatCodingSession(
            conversation_id="test-status",
            sandbox_mgr=None,
            config=cca.ChatCodingConfig(),
        )
        status = session.get_status()
        assert status["conversation_id"] == "test-status"
        assert status["active"] is False
        assert status["turn_count"] == 0
        assert isinstance(status["files"], list)

    def test_cumulative_summary_updates(self, cca):
        session = cca.ChatCodingSession(
            conversation_id="test-summary",
            sandbox_mgr=None,
            config=cca.ChatCodingConfig(),
        )
        session._sandbox_state.turn_count = 1
        session._update_cumulative_summary("build API", "Created 3 files")
        assert "Turn 1" in session._sandbox_state.cumulative_summary
        assert "build API" in session._sandbox_state.cumulative_summary

    def test_cumulative_summary_caps_at_5(self, cca):
        session = cca.ChatCodingSession(
            conversation_id="test-cap",
            sandbox_mgr=None,
            config=cca.ChatCodingConfig(),
        )
        for i in range(8):
            session._sandbox_state.turn_count = i + 1
            session._update_cumulative_summary(f"task {i}", f"done {i}")
        lines = session._sandbox_state.cumulative_summary.strip().split("\n")
        assert len(lines) <= 5

    def test_per_turn_feature_state_init(self, cca):
        session = cca.ChatCodingSession(
            conversation_id="test-features",
            sandbox_mgr=None,
            config=cca.ChatCodingConfig(),
        )
        assert session._turn_images is None
        assert session._turn_web_search is False
        assert session._turn_think is False
        assert session._last_tool_calls == []
        assert session._last_vision_meta == {}
        assert session._last_plugin_annotations == []

    def test_rich_callback_detection_simple(self, cca):
        def simple_llm(prompt, system, model):
            return "response"

        session = cca.ChatCodingSession(
            conversation_id="test-detect-simple",
            sandbox_mgr=None,
            config=cca.ChatCodingConfig(),
            llm_call=simple_llm,
        )
        assert session._detect_rich_callback() is False

    def test_rich_callback_detection_rich(self, cca):
        def rich_llm(messages, model, context):
            return cca.LLMCallResult(text="rich")

        session = cca.ChatCodingSession(
            conversation_id="test-detect-rich",
            sandbox_mgr=None,
            config=cca.ChatCodingConfig(),
            llm_call=rich_llm,
        )
        assert session._detect_rich_callback() is True

    def test_call_llm_no_callback(self, cca):
        session = cca.ChatCodingSession(
            conversation_id="test-no-cb",
            sandbox_mgr=None,
            config=cca.ChatCodingConfig(),
            llm_call=None,
        )
        result = session._call_llm("prompt", "system", "model")
        assert result.error != ""


# ===================================================================
# ChatCodingManager
# ===================================================================

class TestChatCodingManager:
    """Test ChatCodingManager pool management."""

    def test_manager_defaults(self, cca):
        mgr = cca.ChatCodingManager(sandbox_mgr=None)
        assert mgr.enabled is False
        assert mgr.active_session_count() == 0

    def test_manager_status(self, cca):
        mgr = cca.ChatCodingManager(sandbox_mgr=None)
        status = mgr.get_status()
        assert "enabled" in status
        assert "available" in status
        assert "session_timeout_minutes" in status
        assert "max_concurrent_sessions" in status
        assert "active_sessions" in status
        assert "auto_test" in status
        assert "max_fix_retries" in status

    def test_manager_list_sessions_empty(self, cca):
        mgr = cca.ChatCodingManager(sandbox_mgr=None)
        assert mgr.list_sessions() == []

    def test_manager_enable_disable(self, cca):
        mgr = cca.ChatCodingManager(sandbox_mgr=None)
        assert mgr.enabled is False
        mgr.enabled = True
        assert mgr.enabled is True
        mgr.enabled = False
        assert mgr.enabled is False

    def test_manager_cleanup_empty(self, cca):
        mgr = cca.ChatCodingManager(sandbox_mgr=None)
        assert mgr.cleanup_expired() == 0

    def test_manager_destroy_nonexistent(self, cca):
        mgr = cca.ChatCodingManager(sandbox_mgr=None)
        assert mgr.destroy_session("nonexistent") is False

    def test_manager_get_session_nonexistent(self, cca):
        mgr = cca.ChatCodingManager(sandbox_mgr=None)
        assert mgr.get_session("nonexistent") is None


# ===================================================================
# CodingEvent
# ===================================================================

class TestCodingEvent:
    """Test CodingEvent data class."""

    def test_event_creation(self, cca):
        event = cca.CodingEvent(
            event_type="coding_plan",
            data={"steps": ["step1"]},
            content="Planning...",
        )
        assert event.event_type == "coding_plan"
        assert event.data["steps"] == ["step1"]
        assert event.content == "Planning..."

    def test_event_defaults(self, cca):
        event = cca.CodingEvent(event_type="coding_status")
        assert event.data == {}
        assert event.content == ""


# ===================================================================
# Schemas
# ===================================================================

class TestSchemas:
    """Test schema additions for S118."""

    def test_chat_request_has_chat_coding(self, schemas):
        req = schemas.ChatRequest(message="hello")
        assert hasattr(req, "chat_coding")
        assert req.chat_coding is None

    def test_chat_request_chat_coding_true(self, schemas):
        req = schemas.ChatRequest(message="test", chat_coding=True)
        assert req.chat_coding is True

    def test_chat_coding_status_response(self, schemas):
        resp = schemas.ChatCodingStatusResponse(
            enabled=True, available=True, active_sessions=2
        )
        assert resp.enabled is True
        assert resp.active_sessions == 2
        assert resp.max_fix_retries == 3  # default

    def test_chat_coding_toggle_request(self, schemas):
        req = schemas.ChatCodingToggleRequest(enabled=True)
        assert req.enabled is True

    def test_chat_coding_session_info(self, schemas):
        info = schemas.ChatCodingSessionInfo(
            session_id="cc-abc-123",
            conversation_id="conv-abc",
            turn_count=3,
            files=["main.py", "test.py"],
        )
        assert info.turn_count == 3
        assert len(info.files) == 2
        assert info.compression_active is False


# ===================================================================
# Routes (source code analysis)
# ===================================================================

class TestRoutesSourceCode:
    """Test that routes_chat.py has the expected S118 code."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        with open(_ROUTES_PATH) as f:
            self.source = f.read()

    def test_chat_coding_import(self):
        assert "from opti_oignon.chat_coding_agent import" in self.source

    def test_chat_coding_manager_import(self):
        assert "chat_coding_manager" in self.source

    def test_coding_status_endpoint(self):
        assert '"/coding/status"' in self.source

    def test_coding_toggle_endpoint(self):
        assert '"/coding/toggle"' in self.source

    def test_coding_sessions_endpoint(self):
        assert '"/coding/sessions"' in self.source

    def test_coding_destroy_endpoint(self):
        assert '"/coding/{conversation_id}"' in self.source

    def test_coding_session_status_endpoint(self):
        assert '"/coding/{conversation_id}/status"' in self.source

    def test_coding_cleanup_endpoint(self):
        assert '"/coding/cleanup"' in self.source

    def test_stream_chat_coding_function(self):
        assert "async def _stream_chat_coding" in self.source

    def test_build_rich_llm_callback(self):
        assert "def _build_rich_llm_callback" in self.source

    def test_code_prefix_detection(self):
        assert '/code ' in self.source

    def test_mutual_exclusion_qs_disabled(self):
        # When Code Agent is ON, Quick Sandbox should be disabled
        assert "set_quick_sandbox_mode(False)" in self.source

    def test_chat_coding_metadata_in_ws(self):
        assert '"chat_coding"' in self.source

    def test_vision_in_rich_callback(self):
        assert "vision_pipeline" in self.source

    def test_web_search_in_rich_callback(self):
        assert "search_and_augment" in self.source

    def test_plugin_hooks_in_rich_callback(self):
        assert "pre_inference" in self.source
        assert "post_inference" in self.source


# ===================================================================
# Frontend file existence and structure
# ===================================================================

class TestFrontendFiles:
    """Test frontend component files exist and have correct structure."""

    def test_coding_agent_inline_exists(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "CodingAgentInline.svelte"
        )
        assert os.path.isfile(path)

    def test_coding_agent_inline_imports_sandbox(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "CodingAgentInline.svelte"
        )
        content = open(path).read()
        assert "SandboxFileManager" in content

    def test_coding_agent_inline_has_plan(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "CodingAgentInline.svelte"
        )
        content = open(path).read()
        assert "planSteps" in content

    def test_coding_agent_inline_has_test_badge(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "CodingAgentInline.svelte"
        )
        content = open(path).read()
        assert "testStatus" in content
        assert "Tests passed" in content
        assert "Tests failed" in content

    def test_coding_agent_inline_has_vision_badge(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "CodingAgentInline.svelte"
        )
        content = open(path).read()
        assert "hasVision" in content
        assert "Vision" in content

    def test_coding_agent_inline_no_hex_colors(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "CodingAgentInline.svelte"
        )
        content = open(path).read()
        # No inline hex colors like #fff, #000, etc.
        matches = re.findall(
            r'(?:color|background|border)(?:-color)?:\s*#[0-9a-fA-F]',
            content,
        )
        assert len(matches) == 0, f"Hex colors found: {matches}"

    def test_chat_control_bar_has_code_toggle(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "ChatControlBar.svelte"
        )
        content = open(path).read()
        assert "chatCodingEnabled" in content
        assert "toggleChatCoding" in content
        assert "Code" in content

    def test_chat_control_bar_mutual_exclusion(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "ChatControlBar.svelte"
        )
        content = open(path).read()
        # Check that toggling one disables the other
        assert "chatCodingEnabled.set(false)" in content
        assert "quickSandboxEnabled.set(false)" in content

    def test_chat_message_has_coding_inline(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "ChatMessage.svelte"
        )
        content = open(path).read()
        assert "CodingAgentInline" in content
        assert "hasCoding" in content
        assert "codingMeta" in content

    def test_chat_input_has_code_detection(self):
        path = os.path.join(
            _FRONTEND, "components", "chat", "ChatInput.svelte"
        )
        content = open(path).read()
        assert "isCodeCommand" in content
        assert "/code" in content
        assert "Code Agent" in content

    def test_chat_options_store_has_chat_coding(self):
        content = open(_STORE_PATH).read()
        assert "chatCodingEnabled" in content
        assert "chat_coding" in content

    def test_chat_options_no_duplicate_quick_sandbox(self):
        content = open(_STORE_PATH).read()
        count = content.count(
            "export const quickSandboxEnabled = writable"
        )
        assert count == 1, f"Found {count} quickSandboxEnabled declarations"

    def test_chat_options_reset_includes_chat_coding(self):
        content = open(_STORE_PATH).read()
        assert "chatCodingEnabled.set(false)" in content


# ===================================================================
# Feature flag
# ===================================================================

class TestFeatureFlag:
    """Test CHAT_CODING_AVAILABLE flag."""

    def test_flag_exists(self, cca):
        assert hasattr(cca, "CHAT_CODING_AVAILABLE")

    def test_flag_is_bool(self, cca):
        assert isinstance(cca.CHAT_CODING_AVAILABLE, bool)


# ===================================================================
# Module-level singleton
# ===================================================================

class TestSingleton:
    """Test module-level chat_coding_manager singleton."""

    def test_singleton_exists(self, cca):
        assert hasattr(cca, "chat_coding_manager")
        assert isinstance(cca.chat_coding_manager, cca.ChatCodingManager)

    def test_singleton_default_disabled(self, cca):
        assert cca.chat_coding_manager.enabled is False


# ===================================================================
# Fix verification tests
# ===================================================================

class TestFixVerification:
    """Verify the hotfix patches applied after initial delivery."""

    def test_routes_no_system_prompt_override(self):
        """system_prompt_override was removed (not a valid param)."""
        with open(_ROUTES_PATH) as f:
            source = f.read()
        assert "system_prompt_override" not in source

    def test_routes_uses_system_prompt_suffix(self):
        """Rich callback uses system_prompt_suffix instead."""
        with open(_ROUTES_PATH) as f:
            source = f.read()
        assert "system_prompt_suffix=coding_system" in source

    def test_types_has_coding_events(self):
        """ChatToken type includes coding_plan, coding_step, etc."""
        path = os.path.join(_FRONTEND, "..", "..", "types.ts")
        # Normalize path
        types_path = os.path.join(
            _BASE, "frontend", "src", "lib", "types.ts"
        )
        content = open(types_path).read()
        assert "coding_plan" in content
        assert "coding_step" in content
        assert "coding_test" in content
        assert "coding_fix" in content
        assert "coding_done" in content

    def test_types_chat_response_has_coding(self):
        """ChatResponse has chat_coding, coding_result, turn_count."""
        types_path = os.path.join(
            _BASE, "frontend", "src", "lib", "types.ts"
        )
        content = open(types_path).read()
        assert "chat_coding?: boolean" in content
        assert "coding_result?" in content
        assert "turn_count?" in content

    def test_types_callbacks_has_coding_event(self):
        """ChatStreamCallbacks has onCodingEvent."""
        types_path = os.path.join(
            _BASE, "frontend", "src", "lib", "types.ts"
        )
        content = open(types_path).read()
        assert "onCodingEvent?" in content

    def test_api_chat_handles_coding_events(self):
        """api/chat.ts switch handles coding_plan etc."""
        api_path = os.path.join(
            _BASE, "frontend", "src", "lib", "api", "chat.ts"
        )
        content = open(api_path).read()
        assert "case 'coding_plan'" in content
        assert "case 'coding_step'" in content
        assert "case 'coding_done'" in content
        assert "onCodingEvent" in content

    def test_api_chat_done_has_coding_metadata(self):
        """api/chat.ts done handler extracts chat_coding."""
        api_path = os.path.join(
            _BASE, "frontend", "src", "lib", "api", "chat.ts"
        )
        content = open(api_path).read()
        assert "data.metadata?.chat_coding" in content
        assert "data.metadata?.coding_result" in content

    def test_store_has_coding_meta(self):
        """stores/chat.ts has lastCodingMeta store."""
        store_path = os.path.join(
            _BASE, "frontend", "src", "lib", "stores", "chat.ts"
        )
        content = open(store_path).read()
        assert "lastCodingMeta" in content
        assert "coding_meta" in content

    def test_store_send_has_chat_coding_option(self):
        """stores/chat.ts sendMessage accepts chat_coding option."""
        store_path = os.path.join(
            _BASE, "frontend", "src", "lib", "stores", "chat.ts"
        )
        content = open(store_path).read()
        assert "chat_coding?: boolean" in content
        assert "chat_coding: options?.chat_coding" in content

    def test_store_has_streaming_coding_events(self):
        """stores/chat.ts has streamingCodingEvents store."""
        store_path = os.path.join(
            _BASE, "frontend", "src", "lib", "stores", "chat.ts"
        )
        content = open(store_path).read()
        assert "streamingCodingEvents" in content
        assert "CodingEventEntry" in content

    def test_store_has_is_coding_stream(self):
        """stores/chat.ts has isCodingStream store."""
        store_path = os.path.join(
            _BASE, "frontend", "src", "lib", "stores", "chat.ts"
        )
        content = open(store_path).read()
        assert "isCodingStream" in content

    def test_store_on_coding_event_callback(self):
        """stores/chat.ts wires onCodingEvent callback."""
        store_path = os.path.join(
            _BASE, "frontend", "src", "lib", "stores", "chat.ts"
        )
        content = open(store_path).read()
        assert "onCodingEvent:" in content
        assert "streamingCodingEvents.update" in content

    def test_coding_agent_progress_exists(self):
        """CodingAgentProgress.svelte exists."""
        path = os.path.join(
            _FRONTEND, "components", "chat", "CodingAgentProgress.svelte"
        )
        assert os.path.isfile(path)

    def test_coding_agent_progress_has_live_indicators(self):
        """CodingAgentProgress shows plan steps, test results, files."""
        path = os.path.join(
            _FRONTEND, "components", "chat", "CodingAgentProgress.svelte"
        )
        content = open(path).read()
        assert "planSteps" in content
        assert "implementedFiles" in content
        assert "lastTest" in content
        assert "fixAttempts" in content
        assert "animate-pulse" in content

    def test_chat_page_imports_coding_progress(self):
        """Chat page imports CodingAgentProgress."""
        page_path = os.path.join(
            _BASE, "frontend", "src", "routes", "chat", "[id]", "+page.svelte"
        )
        content = open(page_path).read()
        assert "CodingAgentProgress" in content
        assert "isCodingStream" in content
