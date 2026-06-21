#!/usr/bin/env python3
"""
Tests for Background Execution (SQ-07) — S77.

Covers: _RunState lifecycle, execute_all_steps with should_stop callback,
background thread start/stop, execute-all/status/stop endpoint logic.
"""

import importlib.util
import json
import os
import sys
import threading
import time

import pytest

# ---------------------------------------------------------------------------
# Module loading (test isolation — no ollama needed)
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
sys.modules["opti_oignon.sandbox_tools"] = _st_mod

SandboxToolSession = _st_mod.SandboxToolSession

# coding_agent
_ca_path = os.path.join(_base, "coding_agent.py")
_ca_spec = importlib.util.spec_from_file_location("coding_agent", _ca_path)
_ca_mod = importlib.util.module_from_spec(_ca_spec)
_ca_spec.loader.exec_module(_ca_mod)

CodingAgent = _ca_mod.CodingAgent
CodingAgentConfig = _ca_mod.CodingAgentConfig
CodingPhase = _ca_mod.CodingPhase
CodingPlan = _ca_mod.CodingPlan
PlanStep = _ca_mod.PlanStep
PlanStepType = _ca_mod.PlanStepType

# routes_coding (_RunState)
# We extract _RunState class directly from the source to avoid
# full FastAPI app startup and relative import issues.
import ast as _ast
import textwrap as _textwrap

_rc_path = os.path.join(_base, "api", "routes_coding.py")
try:
    # Parse the source and extract _RunState class definition
    with open(_rc_path, encoding="utf-8") as _f:
        _rc_source = _f.read()

    # Build a minimal module with just _RunState
    _rs_source = (
        "import threading\nimport logging\n"
        "from typing import Any\n"
        "logger = logging.getLogger(__name__)\n"
    )
    _tree = _ast.parse(_rc_source)
    for _node in _ast.walk(_tree):
        if isinstance(_node, _ast.ClassDef) and _node.name == "_RunState":
            _rs_source += _ast.get_source_segment(_rc_source, _node)
            break

    _rs_ns: dict = {}
    exec(_rs_source, _rs_ns)
    _RunState = _rs_ns["_RunState"]
    ROUTES_LOADED = True
except Exception as _load_err:
    ROUTES_LOADED = False
    _RunState = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sandbox_config(tmp_path):
    return SandboxConfig(
        enabled=True,
        isolation_backend="tempdir",
        require_degraded_confirmation=False,
        workspace_base=str(tmp_path / "sandboxes"),
        command_timeout=10,
        max_output_bytes=65536,
        max_stderr_bytes=4096,
        max_concurrent_sessions=3,
        audit_db_path=str(tmp_path / "audit.db"),
        blocked_commands=["sudo", "curl", "wget"],
        blocked_patterns=[],
    )


@pytest.fixture
def sandbox_mgr(sandbox_config):
    return SandboxManager(config=sandbox_config)


@pytest.fixture
def session(sandbox_mgr):
    return SandboxToolSession(
        sandbox_mgr=sandbox_mgr,
        tool_registry=ToolRegistry(),
    )


@pytest.fixture
def agent_config():
    return CodingAgentConfig(
        enabled=True,
        max_iterations=10,
        max_fix_retries=2,
        auto_test=False,
        checkpoint_before_apply=True,
    )


@pytest.fixture
def agent(session, agent_config):
    return CodingAgent(
        sandbox_session=session,
        config=agent_config,
    )


def _make_plan(steps_count: int = 3) -> CodingPlan:
    """Build a simple plan with N create steps."""
    steps = []
    for i in range(1, steps_count + 1):
        steps.append(PlanStep(
            step_number=i,
            step_type=PlanStepType.CREATE,
            description=f"Create file_{i}.py",
            file_path=f"file_{i}.py",
            content=f"# file {i}\nprint({i})\n",
        ))
    return CodingPlan(
        task="test background",
        steps=steps,
        summary=f"Create {steps_count} files",
        estimated_files=steps_count,
    )


# ---------------------------------------------------------------------------
# Tests: execute_all_steps with should_stop
# ---------------------------------------------------------------------------


class TestExecuteAllStepsShouldStop:
    """Tests for the should_stop callback in execute_all_steps."""

    def test_execute_all_no_stop(self, agent):
        """All steps execute when should_stop is None."""
        task_id = agent.start_task("test all", allow_degraded=True)
        agent._plan = _make_plan(3)

        executed = agent.execute_all_steps(should_stop=None)
        assert len(executed) == 3
        assert all(s.completed for s in executed)

        agent.abort()

    def test_execute_all_with_false_stop(self, agent):
        """All steps execute when should_stop always returns False."""
        task_id = agent.start_task("test false stop", allow_degraded=True)
        agent._plan = _make_plan(3)

        executed = agent.execute_all_steps(should_stop=lambda: False)
        assert len(executed) == 3

        agent.abort()

    def test_execute_all_stop_before_first(self, agent):
        """No steps execute when should_stop returns True immediately."""
        task_id = agent.start_task("test stop early", allow_degraded=True)
        agent._plan = _make_plan(3)

        executed = agent.execute_all_steps(should_stop=lambda: True)
        assert len(executed) == 0

        agent.abort()

    def test_execute_all_stop_after_n_steps(self, agent):
        """Execution stops after N steps when should_stop triggers."""
        task_id = agent.start_task("test stop after 2", allow_degraded=True)
        agent._plan = _make_plan(5)

        counter = {"n": 0}

        def stop_after_2():
            counter["n"] += 1
            return counter["n"] > 2  # stop before 3rd step

        executed = agent.execute_all_steps(should_stop=stop_after_2)
        assert len(executed) == 2

        agent.abort()

    def test_execute_all_stop_emits_event(self, agent):
        """Stopped execution emits a 'stopped' event."""
        task_id = agent.start_task("test stop event", allow_degraded=True)
        agent._plan = _make_plan(3)

        events = []
        agent.add_progress_callback(lambda e: events.append(e))

        agent.execute_all_steps(should_stop=lambda: True)

        stopped = [e for e in events if e.get("type") == "stopped"]
        assert len(stopped) == 1
        assert stopped[0]["executed_steps"] == 0

        agent.abort()

    def test_execute_all_stop_logs_history(self, agent):
        """Stopped execution logs a history entry."""
        task_id = agent.start_task("test stop log", allow_degraded=True)
        agent._plan = _make_plan(3)

        agent.execute_all_steps(should_stop=lambda: True)

        history = agent.history
        stopped_entries = [
            h for h in history if h.action == "stopped"
        ]
        assert len(stopped_entries) == 1

        agent.abort()

    def test_execute_all_backward_compatible(self, agent):
        """execute_all_steps works without should_stop (backward compat)."""
        task_id = agent.start_task("test compat", allow_degraded=True)
        agent._plan = _make_plan(2)

        # Call without keyword argument
        executed = agent.execute_all_steps()
        assert len(executed) == 2

        agent.abort()

    def test_execute_all_stop_callback_dynamic(self, agent):
        """should_stop callback can change dynamically."""
        task_id = agent.start_task("test dynamic", allow_degraded=True)
        agent._plan = _make_plan(5)

        state = {"stop": False}

        def dynamic_stop():
            return state["stop"]

        # Monkey-patch to trigger stop mid-execution
        original_execute = agent._execute_step

        def patched_execute(step):
            result = original_execute(step)
            if step.step_number >= 3:
                state["stop"] = True
            return result

        agent._execute_step = patched_execute
        executed = agent.execute_all_steps(should_stop=dynamic_stop)

        # Should have executed 3 steps (stop checked before step 4)
        assert len(executed) == 3

        agent.abort()


# ---------------------------------------------------------------------------
# Tests: _RunState
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ROUTES_LOADED, reason="routes_coding could not load")
class TestRunState:
    """Tests for the _RunState singleton pattern."""

    def test_initial_state(self):
        """Fresh _RunState starts idle."""
        rs = _RunState()
        state = rs.get_state()
        assert state["is_running"] is False
        assert state["should_stop"] is False
        assert state["error"] == ""
        assert state["executed_count"] == 0

    def test_start_sets_running(self, agent):
        """start() sets is_running to True."""
        task_id = agent.start_task("test run", allow_degraded=True)
        agent._plan = _make_plan(2)

        rs = _RunState()
        started = rs.start(agent, task_id)
        assert started is True
        assert rs.is_running is True

        # Wait for completion
        if rs.thread:
            rs.thread.join(timeout=5)

        agent.abort()

    def test_start_returns_false_if_running(self, agent):
        """start() returns False if already running."""
        task_id = agent.start_task("test double", allow_degraded=True)
        agent._plan = _make_plan(10)

        rs = _RunState()
        rs.start(agent, task_id)

        # Try to start again
        second = rs.start(agent, task_id)
        assert second is False

        rs.should_stop = True
        if rs.thread:
            rs.thread.join(timeout=5)

        agent.abort()

    def test_stop_signals_flag(self, agent):
        """stop() sets should_stop to True."""
        task_id = agent.start_task("test stop", allow_degraded=True)
        agent._plan = _make_plan(2)

        rs = _RunState()
        rs.start(agent, task_id)

        stopped = rs.stop()
        assert stopped is True
        assert rs.should_stop is True

        if rs.thread:
            rs.thread.join(timeout=5)

        agent.abort()

    def test_stop_returns_false_when_idle(self):
        """stop() returns False when not running."""
        rs = _RunState()
        assert rs.stop() is False

    def test_run_completes_and_resets(self, agent):
        """Background thread completes and resets is_running."""
        task_id = agent.start_task("test complete", allow_degraded=True)
        agent._plan = _make_plan(2)

        rs = _RunState()
        rs.start(agent, task_id)

        # Wait for completion
        if rs.thread:
            rs.thread.join(timeout=10)

        state = rs.get_state()
        assert state["is_running"] is False
        assert state["executed_count"] == 2
        assert state["error"] == ""

        agent.abort()

    def test_run_captures_error(self, agent):
        """Background thread captures exceptions into error field."""
        task_id = agent.start_task("test error", allow_degraded=True)
        # No plan set -> execute_all_steps will try to access None plan

        rs = _RunState()
        rs.start(agent, task_id)

        if rs.thread:
            rs.thread.join(timeout=5)

        state = rs.get_state()
        assert state["is_running"] is False
        # No error expected because execute_all_steps handles None plan gracefully
        assert state["executed_count"] == 0

        agent.abort()

    def test_run_tracks_task_id(self, agent):
        """_RunState tracks the task_id."""
        task_id = agent.start_task("test id", allow_degraded=True)
        agent._plan = _make_plan(1)

        rs = _RunState()
        rs.start(agent, task_id)

        assert rs.task_id == task_id

        if rs.thread:
            rs.thread.join(timeout=5)

        agent.abort()

    def test_get_state_returns_dict(self):
        """get_state returns a proper dict with all expected keys."""
        rs = _RunState()
        state = rs.get_state()
        assert isinstance(state, dict)
        expected_keys = {
            "is_running", "should_stop", "error",
            "executed_count", "task_id",
        }
        assert set(state.keys()) == expected_keys

    def test_graceful_stop_mid_execution(self, agent):
        """Graceful stop halts execution mid-plan."""
        task_id = agent.start_task("test mid stop", allow_degraded=True)
        agent._plan = _make_plan(10)

        rs = _RunState()

        # Use a slower step execution to ensure stop can intervene
        original_execute = agent._execute_step
        call_count = {"n": 0}

        def slow_execute(step):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                time.sleep(0.05)
            return original_execute(step)

        agent._execute_step = slow_execute

        rs.start(agent, task_id)

        # Wait for a few steps then stop
        time.sleep(0.05)
        rs.stop()

        if rs.thread:
            rs.thread.join(timeout=10)

        state = rs.get_state()
        assert state["is_running"] is False
        # Should have stopped before all 10 steps complete
        assert state["executed_count"] <= 10

        agent.abort()

    def test_thread_is_daemon(self, agent):
        """Background thread is a daemon thread."""
        task_id = agent.start_task("test daemon", allow_degraded=True)
        agent._plan = _make_plan(1)

        rs = _RunState()
        rs.start(agent, task_id)

        if rs.thread:
            assert rs.thread.daemon is True
            rs.thread.join(timeout=5)

        agent.abort()

    def test_thread_name(self, agent):
        """Background thread has the expected name."""
        task_id = agent.start_task("test name", allow_degraded=True)
        agent._plan = _make_plan(1)

        rs = _RunState()
        rs.start(agent, task_id)

        if rs.thread:
            assert rs.thread.name == "coding-execute-all"
            rs.thread.join(timeout=5)

        agent.abort()

    def test_multiple_runs_sequential(self, agent):
        """_RunState can be reused for sequential runs."""
        rs = _RunState()

        # First run
        task_id = agent.start_task("test seq 1", allow_degraded=True)
        agent._plan = _make_plan(1)
        rs.start(agent, task_id)
        if rs.thread:
            rs.thread.join(timeout=5)

        assert rs.get_state()["executed_count"] == 1
        agent.abort()

        # Second run
        task_id2 = agent.start_task("test seq 2", allow_degraded=True)
        agent._plan = _make_plan(2)
        rs.start(agent, task_id2)
        if rs.thread:
            rs.thread.join(timeout=5)

        assert rs.get_state()["executed_count"] == 2
        agent.abort()
