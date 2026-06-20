#!/usr/bin/env python3
"""
Tests for coding agent cascading model escalation -- Opti-Oignon S81

Covers:
- CodingAgentConfig: enable_cascading, escalate_after_failures, per_step_routing
- _fix_loop: model escalation after N consecutive failures
- _maybe_escalate: tier resolution, event emission, counter reset
- _get_model_for_step: per-step routing based on step type
- get_status: cascading info in status dict
- Properties: cascading_engine, escalated_model
"""

import importlib.util
import os
import time
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch, call

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
TestResult = _mod.TestResult
WorkingMemory = _mod.WorkingMemory

# Load cascading module for tier config
_CASC_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "cascading.py",
)
try:
    _casc_spec = importlib.util.spec_from_file_location("cascading", _CASC_PATH)
    _casc_mod = importlib.util.module_from_spec(_casc_spec)
    _casc_spec.loader.exec_module(_casc_mod)
    CascadeTierConfig = _casc_mod.CascadeTierConfig
    CascadingInference = _casc_mod.CascadingInference
    CASCADING_LOADED = True
except Exception:
    CASCADING_LOADED = False
    CascadeTierConfig = None
    CascadingInference = None


def _make_agent(
    config_overrides=None,
    cascading_engine=None,
    smart_router=None,
):
    """Helper: create a CodingAgent with mocked session and LLM."""
    cfg = CodingAgentConfig()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)

    mock_session = MagicMock()
    mock_session.active = True
    mock_session.start.return_value = "test-session"
    mock_session.bash.return_value = ""
    mock_session.view.return_value = ""
    mock_session.create_file.return_value = "File created"
    mock_session.str_replace.return_value = "Replaced"
    mock_session.extract_files.return_value = []
    mock_session.inject_directory.return_value = 0

    mock_llm = MagicMock(return_value='{"fix_type":"str_replace","file_path":"f.py","old_str":"a","new_str":"b","analysis":"fix"}')

    agent = CodingAgent(
        sandbox_session=mock_session,
        model="base-model",
        llm_call=mock_llm,
        config=cfg,
        fingerprint_manager=None,
        cascading_engine=cascading_engine,
        smart_router=smart_router,
    )
    return agent, mock_session, mock_llm


def _make_mock_cascading(tiers=None):
    """Create a mock cascading engine with configurable tiers."""
    mock = MagicMock()
    if tiers is None:
        tiers = [
            MagicMock(name="fast", model="qwen3:8b", threshold=0.6),
            MagicMock(name="standard", model="qwen3:32b", threshold=0.5),
            MagicMock(name="power", model="deepseek-r1:32b", threshold=0.3),
        ]
        # Set .name attribute properly (MagicMock name kwarg is special)
        tiers[0].name = "fast"
        tiers[1].name = "standard"
        tiers[2].name = "power"
    mock.tiers = tiers
    mock.enabled = True
    return mock


# ===================================================================
# Config fields
# ===================================================================

class TestCodingAgentConfigCascading(unittest.TestCase):
    """Tests for cascading-related config fields."""

    def test_default_enable_cascading_true(self):
        cfg = CodingAgentConfig()
        self.assertTrue(cfg.enable_cascading)

    def test_default_escalate_after_failures_2(self):
        cfg = CodingAgentConfig()
        self.assertEqual(cfg.escalate_after_failures, 2)

    def test_default_per_step_routing_false(self):
        cfg = CodingAgentConfig()
        self.assertFalse(cfg.per_step_routing)

    def test_custom_escalate_after_failures(self):
        cfg = CodingAgentConfig(escalate_after_failures=5)
        self.assertEqual(cfg.escalate_after_failures, 5)

    def test_custom_per_step_routing_enabled(self):
        cfg = CodingAgentConfig(per_step_routing=True)
        self.assertTrue(cfg.per_step_routing)

    def test_cascading_disabled(self):
        cfg = CodingAgentConfig(enable_cascading=False)
        self.assertFalse(cfg.enable_cascading)


# ===================================================================
# Agent initialization
# ===================================================================

class TestCodingAgentCascadingInit(unittest.TestCase):
    """Tests for cascading initialization in CodingAgent."""

    def test_cascading_engine_stored(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(cascading_engine=mock_casc)
        self.assertIs(agent.cascading_engine, mock_casc)

    def test_no_cascading_when_disabled(self):
        agent, _, _ = _make_agent(
            config_overrides={"enable_cascading": False},
            cascading_engine=None,
        )
        self.assertIsNone(agent.cascading_engine)

    def test_escalated_model_initially_none(self):
        agent, _, _ = _make_agent()
        self.assertIsNone(agent.escalated_model)

    def test_consecutive_failures_initially_zero(self):
        agent, _, _ = _make_agent()
        self.assertEqual(agent._consecutive_fix_failures, 0)


# ===================================================================
# _maybe_escalate
# ===================================================================

class TestMaybeEscalate(unittest.TestCase):
    """Tests for _maybe_escalate model escalation logic."""

    def test_no_escalation_when_disabled(self):
        agent, _, _ = _make_agent(
            config_overrides={"enable_cascading": False}
        )
        agent._consecutive_fix_failures = 10
        agent._maybe_escalate(0)
        self.assertIsNone(agent._escalated_model)

    def test_no_escalation_when_no_engine(self):
        agent, _, _ = _make_agent(cascading_engine=None)
        agent._consecutive_fix_failures = 10
        agent._maybe_escalate(0)
        self.assertIsNone(agent._escalated_model)

    def test_no_escalation_below_threshold(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"escalate_after_failures": 3},
            cascading_engine=mock_casc,
        )
        agent._consecutive_fix_failures = 2
        agent._maybe_escalate(0)
        self.assertIsNone(agent._escalated_model)

    def test_escalation_at_threshold(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"escalate_after_failures": 2},
            cascading_engine=mock_casc,
        )
        # Agent model is "base-model", not in tiers, so current_tier_idx = -1
        # Next tier is index 0 (fast)
        agent._consecutive_fix_failures = 2
        agent._maybe_escalate(0)
        self.assertEqual(agent._escalated_model, "qwen3:8b")

    def test_escalation_resets_counter(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"escalate_after_failures": 2},
            cascading_engine=mock_casc,
        )
        agent._consecutive_fix_failures = 2
        agent._maybe_escalate(0)
        self.assertEqual(agent._consecutive_fix_failures, 0)

    def test_escalation_emits_event(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"escalate_after_failures": 2},
            cascading_engine=mock_casc,
        )
        events = []
        agent.add_progress_callback(lambda e: events.append(e))
        agent._phase = CodingPhase.FIXING
        agent._task_id = "test-task"

        agent._consecutive_fix_failures = 2
        agent._maybe_escalate(0)

        escalated_events = [e for e in events if e["type"] == "escalated"]
        self.assertEqual(len(escalated_events), 1)
        self.assertEqual(escalated_events[0]["from_model"], "base-model")
        self.assertEqual(escalated_events[0]["to_model"], "qwen3:8b")
        self.assertEqual(escalated_events[0]["tier_name"], "fast")

    def test_progressive_escalation(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"escalate_after_failures": 1},
            cascading_engine=mock_casc,
        )
        # First escalation: base-model (not in tiers) -> tier 0 (fast)
        agent._consecutive_fix_failures = 1
        agent._maybe_escalate(0)
        self.assertEqual(agent._escalated_model, "qwen3:8b")

        # Second escalation: tier 0 (fast) -> tier 1 (standard)
        agent._consecutive_fix_failures = 1
        agent._maybe_escalate(1)
        self.assertEqual(agent._escalated_model, "qwen3:32b")

        # Third escalation: tier 1 (standard) -> tier 2 (power)
        agent._consecutive_fix_failures = 1
        agent._maybe_escalate(2)
        self.assertEqual(agent._escalated_model, "deepseek-r1:32b")

    def test_no_escalation_at_highest_tier(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"escalate_after_failures": 1},
            cascading_engine=mock_casc,
        )
        # Already at the last tier
        agent._escalated_model = "deepseek-r1:32b"
        agent._consecutive_fix_failures = 5
        agent._maybe_escalate(0)
        # Should stay at deepseek-r1:32b (no tier after index 2)
        self.assertEqual(agent._escalated_model, "deepseek-r1:32b")

    def test_no_escalation_with_empty_tiers(self):
        mock_casc = MagicMock()
        mock_casc.tiers = []
        agent, _, _ = _make_agent(
            config_overrides={"escalate_after_failures": 1},
            cascading_engine=mock_casc,
        )
        agent._consecutive_fix_failures = 5
        agent._maybe_escalate(0)
        self.assertIsNone(agent._escalated_model)


# ===================================================================
# _fix_loop with cascading
# ===================================================================

class TestFixLoopCascading(unittest.TestCase):
    """Tests for _fix_loop with cascading model escalation."""

    def _setup_agent_for_fix_loop(self, escalate_after=2, max_fix_retries=5):
        mock_casc = _make_mock_cascading()
        agent, mock_session, mock_llm = _make_agent(
            config_overrides={
                "escalate_after_failures": escalate_after,
                "max_fix_retries": max_fix_retries,
                "enable_cascading": True,
            },
            cascading_engine=mock_casc,
        )
        agent._task_id = "test-fix"
        agent._phase = CodingPhase.FIXING
        agent._plan = CodingPlan(task="test")
        agent._working_memory = WorkingMemory(task_id="test-fix")
        return agent, mock_session, mock_llm

    def test_fix_loop_uses_escalated_model(self):
        agent, mock_session, mock_llm = self._setup_agent_for_fix_loop(
            escalate_after=1, max_fix_retries=3
        )
        # LLM always returns valid fix JSON, but tests never pass
        mock_llm.return_value = '{"fix_type":"str_replace","file_path":"f.py","old_str":"a","new_str":"b","analysis":"fix"}'

        # Mock run_tests to always fail
        agent.run_tests = MagicMock(return_value=TestResult(
            passed=False, output="FAILED", return_code=1
        ))

        fail_result = TestResult(passed=False, output="FAILED", return_code=1)
        agent._fix_loop(fail_result)

        # Verify LLM was called with different models as escalation occurred
        # _llm_call is called as: self._llm_call(prompt, system=..., model=...)
        models_used = []
        for c in mock_llm.call_args_list:
            m = c.kwargs.get("model")
            if m is None and len(c.args) >= 3:
                m = c.args[2]
            models_used.append(m)
        # After 1 failure, should escalate from base-model to qwen3:8b (tier 0)
        self.assertTrue(
            any(m == "qwen3:8b" for m in models_used),
            f"Expected qwen3:8b in models_used, got: {models_used}"
        )

    def test_fix_loop_success_resets_failures(self):
        agent, mock_session, mock_llm = self._setup_agent_for_fix_loop()
        mock_llm.return_value = '{"fix_type":"str_replace","file_path":"f.py","old_str":"a","new_str":"b","analysis":"fix"}'

        # First test fails, second passes
        call_count = [0]
        def mock_run_tests(command=None):
            call_count[0] += 1
            if call_count[0] >= 2:
                return TestResult(passed=True, output="passed")
            return TestResult(passed=False, output="FAILED")

        agent.run_tests = mock_run_tests

        fail_result = TestResult(passed=False, output="FAILED")
        result = agent._fix_loop(fail_result)
        self.assertTrue(result)
        self.assertEqual(agent._consecutive_fix_failures, 0)

    def test_fix_loop_returns_false_when_exhausted(self):
        agent, _, mock_llm = self._setup_agent_for_fix_loop(max_fix_retries=2)
        mock_llm.return_value = '{"fix_type":"str_replace","file_path":"f.py","old_str":"a","new_str":"b","analysis":"fix"}'
        agent.run_tests = MagicMock(return_value=TestResult(
            passed=False, output="FAILED"
        ))

        fail_result = TestResult(passed=False, output="FAILED")
        result = agent._fix_loop(fail_result)
        self.assertFalse(result)


# ===================================================================
# _get_model_for_step (per-step routing)
# ===================================================================

class TestGetModelForStep(unittest.TestCase):
    """Tests for per-step routing based on step type."""

    def test_returns_base_model_when_routing_disabled(self):
        agent, _, _ = _make_agent(
            config_overrides={"per_step_routing": False}
        )
        step = PlanStep(step_number=1, step_type=PlanStepType.BASH, description="ls")
        result = agent._get_model_for_step(step)
        self.assertEqual(result, "base-model")

    def test_returns_escalated_model_over_routing(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"per_step_routing": True},
            cascading_engine=mock_casc,
        )
        agent._escalated_model = "deepseek-r1:32b"
        step = PlanStep(step_number=1, step_type=PlanStepType.BASH, description="ls")
        result = agent._get_model_for_step(step)
        self.assertEqual(result, "deepseek-r1:32b")

    def test_simple_step_routes_to_fast_tier(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"per_step_routing": True},
            cascading_engine=mock_casc,
        )
        step = PlanStep(step_number=1, step_type=PlanStepType.BASH, description="ls")
        result = agent._get_model_for_step(step)
        self.assertEqual(result, "qwen3:8b")

    def test_test_step_routes_to_fast_tier(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"per_step_routing": True},
            cascading_engine=mock_casc,
        )
        step = PlanStep(step_number=1, step_type=PlanStepType.TEST, description="pytest")
        result = agent._get_model_for_step(step)
        self.assertEqual(result, "qwen3:8b")

    def test_create_step_routes_to_standard_tier(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"per_step_routing": True},
            cascading_engine=mock_casc,
        )
        step = PlanStep(step_number=1, step_type=PlanStepType.CREATE, description="write file")
        result = agent._get_model_for_step(step)
        self.assertEqual(result, "qwen3:32b")

    def test_edit_step_routes_to_standard_tier(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(
            config_overrides={"per_step_routing": True},
            cascading_engine=mock_casc,
        )
        step = PlanStep(step_number=1, step_type=PlanStepType.EDIT, description="fix bug")
        result = agent._get_model_for_step(step)
        self.assertEqual(result, "qwen3:32b")

    def test_routing_with_no_cascading_engine(self):
        agent, _, _ = _make_agent(
            config_overrides={"per_step_routing": True},
            cascading_engine=None,
        )
        step = PlanStep(step_number=1, step_type=PlanStepType.BASH, description="ls")
        result = agent._get_model_for_step(step)
        self.assertEqual(result, "base-model")


# ===================================================================
# get_status cascading block
# ===================================================================

class TestGetStatusCascading(unittest.TestCase):
    """Tests for cascading info in get_status()."""

    def test_status_contains_cascading_key(self):
        agent, _, _ = _make_agent()
        st = agent.get_status()
        self.assertIn("cascading", st)

    def test_status_cascading_enabled(self):
        agent, _, _ = _make_agent(
            config_overrides={"enable_cascading": True}
        )
        st = agent.get_status()
        self.assertTrue(st["cascading"]["enabled"])

    def test_status_cascading_disabled(self):
        agent, _, _ = _make_agent(
            config_overrides={"enable_cascading": False}
        )
        st = agent.get_status()
        self.assertFalse(st["cascading"]["enabled"])

    def test_status_cascading_available(self):
        mock_casc = _make_mock_cascading()
        agent, _, _ = _make_agent(cascading_engine=mock_casc)
        st = agent.get_status()
        self.assertTrue(st["cascading"]["available"])

    def test_status_cascading_not_available(self):
        agent, _, _ = _make_agent(cascading_engine=None)
        st = agent.get_status()
        self.assertFalse(st["cascading"]["available"])

    def test_status_escalated_model_none(self):
        agent, _, _ = _make_agent()
        st = agent.get_status()
        self.assertIsNone(st["cascading"]["escalated_model"])

    def test_status_escalated_model_set(self):
        agent, _, _ = _make_agent()
        agent._escalated_model = "big-model"
        st = agent.get_status()
        self.assertEqual(st["cascading"]["escalated_model"], "big-model")

    def test_status_per_step_routing(self):
        agent, _, _ = _make_agent(
            config_overrides={"per_step_routing": True}
        )
        st = agent.get_status()
        self.assertTrue(st["cascading"]["per_step_routing"])


# ===================================================================
# start_task resets escalation state
# ===================================================================

class TestStartTaskResetsEscalation(unittest.TestCase):
    """Tests that start_task resets cascading state."""

    def test_start_task_resets_escalated_model(self):
        agent, _, _ = _make_agent()
        agent._escalated_model = "some-model"
        agent._consecutive_fix_failures = 5
        agent._phase = CodingPhase.COMPLETED
        agent.start_task("new task", allow_degraded=True)
        self.assertIsNone(agent._escalated_model)

    def test_start_task_resets_consecutive_failures(self):
        agent, _, _ = _make_agent()
        agent._consecutive_fix_failures = 5
        agent._phase = CodingPhase.COMPLETED
        agent.start_task("new task", allow_degraded=True)
        self.assertEqual(agent._consecutive_fix_failures, 0)


if __name__ == "__main__":
    unittest.main()
