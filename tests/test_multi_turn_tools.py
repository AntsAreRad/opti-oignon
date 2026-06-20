#!/usr/bin/env python3
"""
Tests for Multi-Turn Tool Use (S62)
=====================================

Tests cover:
- Tool call history tracking per conversation
- History passed to ToolExecutor across turns
- History clearing and management
- Max history trimming
- API endpoint for tool history CRUD
- Backward compatibility: single-turn tool calling unaffected
- Edge cases (no conversation_id, empty history, etc.)
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# =============================================================================
# MOCK OBJECTS
# =============================================================================

@dataclass
class MockToolCallResult:
    """Mock ToolCallResult for testing."""
    tool_name: str = "web_search"
    arguments: dict[str, Any] = None
    result: str = "Mock result"
    success: bool = True
    execution_time: float = 0.5
    reasoning: str = "Test reasoning"

    def __post_init__(self):
        if self.arguments is None:
            self.arguments = {"query": "test"}


class MockToolExecutionResult:
    """Mock ToolExecutionResult."""
    def __init__(self, response="", tool_calls=None, model="test", total_time=0.0):
        self.response = response
        self.tool_calls = tool_calls or []
        self.model = model
        self.total_time = total_time


class MockToolExecutor:
    """Mock ToolExecutor that records calls."""
    def __init__(self, result=None):
        self._result = result or MockToolExecutionResult(
            response="Test response",
            tool_calls=[MockToolCallResult()],
        )
        self.last_call_kwargs = {}

    @property
    def available(self):
        return True

    def should_use_tools(self, message, model=None):
        return True

    def execute_with_tools(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._result

    def get_tools_prompt(self):
        return "test tools"


class MockRouting:
    """Mock RoutingResult."""
    def __init__(self, model="test-model"):
        self.model = model


# =============================================================================
# TESTS: AgenticExecutor Tool History Tracking
# =============================================================================

class TestToolHistoryTracking:
    """Tests for per-conversation tool call history management."""

    def _make_executor(self, tool_executor=None):
        """Create AgenticExecutor with mocked dependencies."""
        from opti_oignon.agentic_executor import AgenticExecutor

        mock_exec = MagicMock()
        mock_exec.execute = MagicMock(return_value=iter(["response"]))

        ae = AgenticExecutor(
            executor=mock_exec,
            tool_executor=tool_executor or MockToolExecutor(),
            default_model="test-model",
        )
        return ae

    def test_get_tool_history_empty(self):
        """Empty history for new conversation."""
        ae = self._make_executor()
        assert ae.get_tool_history("conv-1") == []

    def test_get_tool_history_no_id(self):
        """Empty history when no conversation_id."""
        ae = self._make_executor()
        assert ae.get_tool_history("") == []
        assert ae.get_tool_history(None) == []

    def test_record_tool_calls(self):
        """Tool calls are recorded in history."""
        ae = self._make_executor()
        calls = [MockToolCallResult(tool_name="web_search")]
        ae._record_tool_calls("conv-1", calls)
        history = ae.get_tool_history("conv-1")
        assert len(history) == 1
        assert history[0].tool_name == "web_search"

    def test_record_multiple_turns(self):
        """Tool calls accumulate across multiple turns."""
        ae = self._make_executor()
        ae._record_tool_calls("conv-1", [MockToolCallResult(tool_name="web_search")])
        ae._record_tool_calls("conv-1", [MockToolCallResult(tool_name="execute_code")])
        history = ae.get_tool_history("conv-1")
        assert len(history) == 2
        assert history[0].tool_name == "web_search"
        assert history[1].tool_name == "execute_code"

    def test_record_no_conversation_id(self):
        """No recording when conversation_id is None/empty."""
        ae = self._make_executor()
        ae._record_tool_calls(None, [MockToolCallResult()])
        ae._record_tool_calls("", [MockToolCallResult()])
        # No history should exist
        assert ae.get_tool_history("") == []

    def test_record_empty_calls(self):
        """No recording when tool_calls list is empty."""
        ae = self._make_executor()
        ae._record_tool_calls("conv-1", [])
        assert ae.get_tool_history("conv-1") == []

    def test_separate_conversations(self):
        """Tool history is independent per conversation."""
        ae = self._make_executor()
        ae._record_tool_calls("conv-1", [MockToolCallResult(tool_name="tool-a")])
        ae._record_tool_calls("conv-2", [MockToolCallResult(tool_name="tool-b")])

        h1 = ae.get_tool_history("conv-1")
        h2 = ae.get_tool_history("conv-2")
        assert len(h1) == 1
        assert h1[0].tool_name == "tool-a"
        assert len(h2) == 1
        assert h2[0].tool_name == "tool-b"

    def test_max_history_trimming(self):
        """History is trimmed when exceeding max size."""
        ae = self._make_executor()
        ae._max_history_per_conversation = 5

        for i in range(10):
            ae._record_tool_calls("conv-1", [
                MockToolCallResult(tool_name=f"tool-{i}")
            ])

        history = ae.get_tool_history("conv-1")
        assert len(history) == 5
        # Should keep the most recent
        assert history[0].tool_name == "tool-5"
        assert history[4].tool_name == "tool-9"

    def test_clear_tool_history(self):
        """Clear history for a specific conversation."""
        ae = self._make_executor()
        ae._record_tool_calls("conv-1", [MockToolCallResult()])
        ae._record_tool_calls("conv-2", [MockToolCallResult()])

        cleared = ae.clear_tool_history("conv-1")
        assert cleared == 1
        assert ae.get_tool_history("conv-1") == []
        # conv-2 unaffected
        assert len(ae.get_tool_history("conv-2")) == 1

    def test_clear_nonexistent_conversation(self):
        """Clearing nonexistent conversation returns 0."""
        ae = self._make_executor()
        assert ae.clear_tool_history("nonexistent") == 0

    def test_clear_all_tool_history(self):
        """Clear all tool history across all conversations."""
        ae = self._make_executor()
        ae._record_tool_calls("conv-1", [MockToolCallResult()])
        ae._record_tool_calls("conv-2", [MockToolCallResult()])

        cleared = ae.clear_all_tool_history()
        assert cleared == 2
        assert ae.get_tool_history("conv-1") == []
        assert ae.get_tool_history("conv-2") == []

    def test_history_returns_copy(self):
        """get_tool_history returns a copy, not a reference."""
        ae = self._make_executor()
        ae._record_tool_calls("conv-1", [MockToolCallResult()])

        h1 = ae.get_tool_history("conv-1")
        h1.clear()
        # Original should be unaffected
        assert len(ae.get_tool_history("conv-1")) == 1


# =============================================================================
# TESTS: ToolExecutor tool_history Parameter
# =============================================================================

class TestToolExecutorHistory:
    """Tests for tool_history parameter in ToolExecutor."""

    def test_execute_with_tools_accepts_history(self):
        """execute_with_tools accepts tool_history parameter."""
        from opti_oignon.tool_executor import ToolExecutor

        te = ToolExecutor(registry=None)
        result = te.execute_with_tools(
            message="test",
            model="test",
            tool_history=[MockToolCallResult()],
        )
        # Should return some result without crashing
        assert result is not None
        assert isinstance(result.response, str)

    def test_execute_without_history_backward_compat(self):
        """execute_with_tools works without tool_history (backward compat)."""
        from opti_oignon.tool_executor import ToolExecutor

        te = ToolExecutor(registry=None)
        result = te.execute_with_tools(
            message="test",
            model="test",
        )
        # Should return some result without crashing
        assert result is not None
        assert isinstance(result.response, str)

    def test_history_none_is_valid(self):
        """tool_history=None is handled correctly."""
        from opti_oignon.tool_executor import ToolExecutor

        te = ToolExecutor(registry=None)
        result = te.execute_with_tools(
            message="test",
            model="test",
            tool_history=None,
        )
        assert result is not None


# =============================================================================
# TESTS: Integration - History Passed Through Pipeline
# =============================================================================

class TestToolsPipelineHistoryIntegration:
    """Tests for tool history being passed through _execute_tools_pipeline."""

    def _make_executor_with_tracking(self):
        """Create executor with a tracking tool executor."""
        from opti_oignon.agentic_executor import AgenticExecutor

        mock_tool_exec = MockToolExecutor()
        mock_base_exec = MagicMock()
        mock_base_exec.execute = MagicMock(return_value=iter(["ok"]))

        ae = AgenticExecutor(
            executor=mock_base_exec,
            tool_executor=mock_tool_exec,
            default_model="test-model",
        )
        return ae, mock_tool_exec

    def test_first_turn_no_history(self):
        """First turn passes no tool history."""
        ae, mock_te = self._make_executor_with_tracking()

        # Force tools pipeline
        routing = MockRouting()
        list(ae._execute_tools_pipeline(
            "test message", routing, "conv-1", None,
        ))

        kwargs = mock_te.last_call_kwargs
        # First turn: tool_history should be None or empty
        assert kwargs.get("tool_history") is None

    def test_second_turn_receives_history(self):
        """Second turn receives tool history from first turn."""
        ae, mock_te = self._make_executor_with_tracking()
        routing = MockRouting()

        # First turn
        list(ae._execute_tools_pipeline(
            "first message", routing, "conv-1", None,
        ))

        # After first turn, history should be recorded
        history = ae.get_tool_history("conv-1")
        assert len(history) >= 1

        # Second turn
        list(ae._execute_tools_pipeline(
            "second message", routing, "conv-1", None,
        ))

        kwargs = mock_te.last_call_kwargs
        # Should include prior history
        assert kwargs.get("tool_history") is not None
        assert len(kwargs["tool_history"]) >= 1

    def test_tool_calls_accumulate(self):
        """Tool calls from multiple turns accumulate in history."""
        ae, mock_te = self._make_executor_with_tracking()
        routing = MockRouting()

        # Three turns
        for i in range(3):
            list(ae._execute_tools_pipeline(
                f"message {i}", routing, "conv-1", None,
            ))

        history = ae.get_tool_history("conv-1")
        assert len(history) == 3  # One tool call per turn

    def test_no_conversation_id_still_works(self):
        """Pipeline works without conversation_id (no history tracking)."""
        ae, mock_te = self._make_executor_with_tracking()
        routing = MockRouting()

        list(ae._execute_tools_pipeline(
            "test", routing, None, None,
        ))

        kwargs = mock_te.last_call_kwargs
        # No history passed when no conv_id
        assert kwargs.get("tool_history") is None


# =============================================================================
# TESTS: API Endpoint
# =============================================================================

class TestToolHistoryAPI:
    """Tests for conversation tool history API endpoints."""

    def test_get_tool_history_endpoint(self):
        """GET /api/conversations/{id}/tool-history returns history."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from opti_oignon.api.routes_conversations import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/api/conversations/test-conv/tool-history")
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert "tool_calls" in data
        assert "count" in data
        assert data["conversation_id"] == "test-conv"

    def test_delete_tool_history_endpoint(self):
        """DELETE /api/conversations/{id}/tool-history clears history."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from opti_oignon.api.routes_conversations import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        resp = client.delete("/api/conversations/test-conv/tool-history")
        assert resp.status_code == 200
        data = resp.json()
        assert "cleared" in data


# =============================================================================
# TESTS: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Ensure multi-turn tool use doesn't break single-turn."""

    def test_single_turn_still_works(self):
        """Single-turn tool calling works as before."""
        from opti_oignon.tool_executor import ToolExecutionResult, ToolExecutor

        te = ToolExecutor(registry=None)
        # Without tool_history param - should still work
        result = te.execute_with_tools(message="test", model="test")
        assert isinstance(result, ToolExecutionResult)

    def test_tool_execution_result_unchanged(self):
        """ToolExecutionResult structure is unchanged."""
        from opti_oignon.tool_executor import ToolExecutionResult

        result = ToolExecutionResult(
            response="test",
            tool_calls=[],
            model="test",
            total_time=0.1,
        )
        assert result.response == "test"
        assert result.tool_calls == []

    def test_tool_call_result_unchanged(self):
        """ToolCallResult structure is unchanged."""
        from opti_oignon.tool_executor import ToolCallResult

        tc = ToolCallResult(
            tool_name="web_search",
            arguments={"query": "test"},
            result="result",
            success=True,
            execution_time=0.5,
            reasoning="test",
        )
        assert tc.tool_name == "web_search"
        assert tc.success is True


# =============================================================================
# TESTS: Additional Edge Cases
# =============================================================================

class TestMultiTurnEdgeCases:
    """Additional edge case tests for multi-turn tool use."""

    def _make_executor(self, tool_executor=None):
        """Create AgenticExecutor with mocked dependencies."""
        from opti_oignon.agentic_executor import AgenticExecutor

        mock_exec = MagicMock()
        mock_exec.execute = MagicMock(return_value=iter(["response"]))

        return AgenticExecutor(
            executor=mock_exec,
            tool_executor=tool_executor or MockToolExecutor(),
            default_model="test-model",
        )

    def test_history_with_failed_tool_calls(self):
        """Failed tool calls are also recorded in history."""
        ae = self._make_executor()
        failed_call = MockToolCallResult(
            tool_name="web_search", success=False, result="Error: timeout"
        )
        ae._record_tool_calls("conv-1", [failed_call])
        history = ae.get_tool_history("conv-1")
        assert len(history) == 1
        assert history[0].success is False

    def test_mixed_success_failure_in_history(self):
        """History correctly stores mix of successful and failed calls."""
        ae = self._make_executor()
        ae._record_tool_calls("conv-1", [
            MockToolCallResult(tool_name="search", success=True),
            MockToolCallResult(tool_name="code", success=False),
        ])
        history = ae.get_tool_history("conv-1")
        assert len(history) == 2
        assert history[0].success is True
        assert history[1].success is False

    def test_clear_all_then_record(self):
        """Can record new history after clearing all."""
        ae = self._make_executor()
        ae._record_tool_calls("conv-1", [MockToolCallResult()])
        ae.clear_all_tool_history()
        ae._record_tool_calls("conv-1", [MockToolCallResult(tool_name="new")])
        assert len(ae.get_tool_history("conv-1")) == 1
        assert ae.get_tool_history("conv-1")[0].tool_name == "new"

    def test_many_conversations(self):
        """Tool history works across many conversations."""
        ae = self._make_executor()
        for i in range(50):
            ae._record_tool_calls(f"conv-{i}", [
                MockToolCallResult(tool_name=f"tool-{i}")
            ])
        assert len(ae.get_tool_history("conv-0")) == 1
        assert len(ae.get_tool_history("conv-49")) == 1
        assert ae.get_tool_history("conv-25")[0].tool_name == "tool-25"
