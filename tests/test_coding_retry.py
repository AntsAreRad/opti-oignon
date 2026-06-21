#!/usr/bin/env python3
"""
Tests for coding agent auto-retry and error classification -- Opti-Oignon S79

Covers:
- is_transient_error: pattern classification for retryable vs permanent errors
- CodingAgentConfig: max_auto_retries and retry_backoff_seconds fields
- _execute_step_with_retry: retry logic, backoff, event emission
- Integration with execute_next_step
"""

import importlib.util
import os
import time
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Direct module loading (bypass __init__.py chain)
# ---------------------------------------------------------------------------

_MOD_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "coding_agent.py",
)
_spec = importlib.util.spec_from_file_location("coding_agent", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CodingAgent = _mod.CodingAgent
CodingAgentConfig = _mod.CodingAgentConfig
CodingPhase = _mod.CodingPhase
PlanStep = _mod.PlanStep
PlanStepType = _mod.PlanStepType
CodingPlan = _mod.CodingPlan


# ===================================================================
# Error Classification
# ===================================================================

class TestIsTransientError(unittest.TestCase):
    """Tests for CodingAgent.is_transient_error static method."""

    def test_empty_string_not_transient(self):
        self.assertFalse(CodingAgent.is_transient_error(""))

    def test_none_like_not_transient(self):
        self.assertFalse(CodingAgent.is_transient_error(""))

    def test_timeout_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("Command timed out after 30s"))

    def test_timed_out_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("Process timed out"))

    def test_bwrap_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("bwrap: Can't create namespace"))

    def test_bubblewrap_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("bubblewrap failed to initialize"))

    def test_connection_refused_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("Connection refused"))

    def test_broken_pipe_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("BrokenPipeError: Broken pipe"))

    def test_sandbox_startup_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("Failed to start sandbox session"))

    def test_resource_unavailable_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("Resource temporarily unavailable"))

    def test_no_space_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("No space left on device"))

    def test_errno_11_is_transient(self):
        self.assertTrue(CodingAgent.is_transient_error("OSError: [Errno 11] Resource unavailable"))

    def test_syntax_error_is_permanent(self):
        self.assertFalse(CodingAgent.is_transient_error("SyntaxError: invalid syntax"))

    def test_import_error_is_permanent(self):
        self.assertFalse(CodingAgent.is_transient_error("ImportError: No module named 'foo'"))

    def test_file_not_found_is_permanent(self):
        self.assertFalse(CodingAgent.is_transient_error("FileNotFoundError: /path/to/file"))

    def test_permission_denied_is_permanent(self):
        self.assertFalse(CodingAgent.is_transient_error("PermissionError: [Errno 13] Permission denied"))

    def test_case_insensitive(self):
        self.assertTrue(CodingAgent.is_transient_error("TIMEOUT: exceeded limit"))
        self.assertTrue(CodingAgent.is_transient_error("BWRAP failed"))

    def test_value_error_is_permanent(self):
        self.assertFalse(CodingAgent.is_transient_error("ValueError: invalid literal"))


# ===================================================================
# Config defaults
# ===================================================================

class TestRetryConfig(unittest.TestCase):
    """Tests for retry-related config fields."""

    def test_default_max_auto_retries(self):
        cfg = CodingAgentConfig()
        self.assertEqual(cfg.max_auto_retries, 2)

    def test_default_backoff(self):
        cfg = CodingAgentConfig()
        self.assertEqual(cfg.retry_backoff_seconds, [1.0, 2.0])

    def test_custom_max_retries(self):
        cfg = CodingAgentConfig(max_auto_retries=5)
        self.assertEqual(cfg.max_auto_retries, 5)

    def test_custom_backoff(self):
        cfg = CodingAgentConfig(retry_backoff_seconds=[0.5, 1.0, 2.0])
        self.assertEqual(cfg.retry_backoff_seconds, [0.5, 1.0, 2.0])

    def test_zero_retries(self):
        cfg = CodingAgentConfig(max_auto_retries=0)
        self.assertEqual(cfg.max_auto_retries, 0)

    def test_empty_backoff(self):
        cfg = CodingAgentConfig(retry_backoff_seconds=[])
        self.assertEqual(cfg.retry_backoff_seconds, [])


# ===================================================================
# _execute_step_with_retry logic
# ===================================================================

class TestExecuteStepWithRetry(unittest.TestCase):
    """Tests for _execute_step_with_retry method behavior."""

    def _make_agent(self, max_retries=2, backoff=None):
        """Create a minimal agent mock for retry testing."""
        if backoff is None:
            backoff = [0.0, 0.0]  # No real delay in tests
        agent = object.__new__(CodingAgent)
        agent._config = CodingAgentConfig(
            max_auto_retries=max_retries,
            retry_backoff_seconds=backoff,
        )
        agent._progress_callbacks = []
        agent._history = []
        agent._history_store = None
        agent._fingerprint = None
        agent._phase = CodingPhase.IMPLEMENTING
        agent._task_id = "test-task"
        return agent

    def _make_step(self, step_number=1):
        return PlanStep(
            step_number=step_number,
            step_type=PlanStepType.BASH,
            description="test step",
        )

    def test_success_on_first_try(self):
        agent = self._make_agent()
        step = self._make_step()
        agent._execute_step = MagicMock(return_value="ok")
        result = agent._execute_step_with_retry(step)
        self.assertEqual(result, "ok")
        self.assertEqual(agent._execute_step.call_count, 1)

    def test_permanent_error_no_retry(self):
        agent = self._make_agent()
        step = self._make_step()
        agent._execute_step = MagicMock(
            side_effect=Exception("SyntaxError: invalid syntax")
        )
        with self.assertRaises(Exception) as ctx:
            agent._execute_step_with_retry(step)
        self.assertIn("SyntaxError", str(ctx.exception))
        self.assertEqual(agent._execute_step.call_count, 1)

    def test_transient_error_retries(self):
        agent = self._make_agent(max_retries=2, backoff=[0.0, 0.0])
        step = self._make_step()
        agent._execute_step = MagicMock(
            side_effect=[
                Exception("Connection refused"),
                Exception("Connection refused"),
                "success",
            ]
        )
        result = agent._execute_step_with_retry(step)
        self.assertEqual(result, "success")
        self.assertEqual(agent._execute_step.call_count, 3)

    def test_transient_error_exhausts_retries(self):
        agent = self._make_agent(max_retries=1, backoff=[0.0])
        step = self._make_step()
        agent._execute_step = MagicMock(
            side_effect=Exception("bwrap startup failed")
        )
        with self.assertRaises(Exception) as ctx:
            agent._execute_step_with_retry(step)
        self.assertIn("bwrap", str(ctx.exception))
        # 1 initial + 1 retry = 2 calls
        self.assertEqual(agent._execute_step.call_count, 2)

    def test_zero_retries_no_retry_on_transient(self):
        agent = self._make_agent(max_retries=0, backoff=[])
        step = self._make_step()
        agent._execute_step = MagicMock(
            side_effect=Exception("timeout during execution")
        )
        with self.assertRaises(Exception):
            agent._execute_step_with_retry(step)
        self.assertEqual(agent._execute_step.call_count, 1)

    def test_retry_emits_event(self):
        agent = self._make_agent(max_retries=1, backoff=[0.0])
        step = self._make_step()
        events = []
        agent._progress_callbacks = [lambda e: events.append(e)]
        agent._execute_step = MagicMock(
            side_effect=[
                Exception("timeout"),
                "recovered",
            ]
        )
        # Need _emit method
        def _emit(event_type, data=None):
            event = {"type": event_type}
            if data:
                event.update(data)
            for cb in agent._progress_callbacks:
                try:
                    cb(event)
                except Exception:
                    pass
        agent._emit = _emit

        result = agent._execute_step_with_retry(step)
        self.assertEqual(result, "recovered")
        retry_events = [e for e in events if e.get("type") == "retry"]
        self.assertEqual(len(retry_events), 1)
        self.assertEqual(retry_events[0]["attempt"], 1)
        self.assertIn("timeout", retry_events[0]["error"])

    def test_success_after_first_transient_failure(self):
        agent = self._make_agent(max_retries=3, backoff=[0.0, 0.0, 0.0])
        step = self._make_step()
        agent._execute_step = MagicMock(
            side_effect=[
                Exception("bwrap: namespace creation failed"),
                "result",
            ]
        )
        result = agent._execute_step_with_retry(step)
        self.assertEqual(result, "result")
        self.assertEqual(agent._execute_step.call_count, 2)

    def test_backoff_index_clamped(self):
        """When more retries than backoff entries, use last entry."""
        agent = self._make_agent(max_retries=5, backoff=[0.0])
        step = self._make_step()
        call_count = [0]

        def _fake_execute(s):
            call_count[0] += 1
            if call_count[0] <= 5:
                raise Exception("timeout again")
            return "finally"

        agent._execute_step = _fake_execute
        # Need _emit
        agent._emit = lambda *a, **kw: None

        result = agent._execute_step_with_retry(step)
        self.assertEqual(result, "finally")
        self.assertEqual(call_count[0], 6)


# ===================================================================
# Transient patterns comprehensive
# ===================================================================

class TestTransientPatterns(unittest.TestCase):
    """Ensure all documented transient patterns are matched."""

    TRANSIENT_MESSAGES = [
        "Command timeout after 60 seconds",
        "Process timed out waiting for sandbox",
        "bwrap: cannot create child process",
        "bubblewrap initialization error",
        "Connection refused by sandbox host",
        "Resource temporarily unavailable: fork()",
        "No space left on device in sandbox",
        "Broken pipe while writing to sandbox",
        "OSError: [Errno 11] temporarily unavailable",
        "OSError: [Errno 110] connection timed out",
        "Sandbox startup failed: permission in namespace",
        "Failed to start sandbox: cgroup error",
    ]

    PERMANENT_MESSAGES = [
        "SyntaxError: unexpected indent",
        "NameError: name 'x' is not defined",
        "TypeError: unsupported operand type",
        "IndexError: list index out of range",
        "KeyError: 'missing_key'",
        "AssertionError",
        "AttributeError: module has no attribute",
        "ZeroDivisionError: division by zero",
    ]

    def test_all_transient_patterns_detected(self):
        for msg in self.TRANSIENT_MESSAGES:
            with self.subTest(msg=msg):
                self.assertTrue(
                    CodingAgent.is_transient_error(msg),
                    f"Should be transient: {msg!r}"
                )

    def test_all_permanent_patterns_rejected(self):
        for msg in self.PERMANENT_MESSAGES:
            with self.subTest(msg=msg):
                self.assertFalse(
                    CodingAgent.is_transient_error(msg),
                    f"Should be permanent: {msg!r}"
                )


if __name__ == "__main__":
    unittest.main()
