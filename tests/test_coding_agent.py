#!/usr/bin/env python3
"""
Tests for the Coding Agent core (S74).

Covers: CodingAgent lifecycle, planning, execution, testing, fixing,
diffing, applying, abort, checkpoints, configuration, security.
"""

import json
import os
import sys
import importlib.util
import tempfile
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
CheckpointResult = _ca_mod.CheckpointResult
CodingPlan = _ca_mod.CodingPlan
PlanStep = _ca_mod.PlanStep
PlanStepType = _ca_mod.PlanStepType
CodingHistoryEntry = _ca_mod.CodingHistoryEntry
FileDiff = _ca_mod.FileDiff
TestResult = _ca_mod.TestResult
_parse_json_response = _ca_mod._parse_json_response
_build_plan_from_response = _ca_mod._build_plan_from_response
_build_fix_from_response = _ca_mod._build_fix_from_response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp(tmp_path):
    return tmp_path


@pytest.fixture
def sandbox_config(tmp_path):
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
        audit_db_path=str(tmp_path / "audit.db"),
        blocked_commands=["sudo", "curl", "wget"],
        blocked_patterns=[],
    )


@pytest.fixture
def sandbox_mgr(sandbox_config):
    """SandboxManager instance for testing."""
    return SandboxManager(config=sandbox_config)


@pytest.fixture
def tool_registry():
    """Fresh ToolRegistry for testing."""
    return ToolRegistry()


@pytest.fixture
def session(sandbox_mgr, tool_registry):
    """SandboxToolSession for testing."""
    return SandboxToolSession(
        sandbox_mgr=sandbox_mgr,
        tool_registry=tool_registry,
    )


@pytest.fixture
def agent_config():
    """CodingAgentConfig with test defaults."""
    return CodingAgentConfig(
        enabled=True,
        max_iterations=5,
        max_fix_retries=2,
        auto_test=False,
        auto_test_command="echo 'tests passed'",
        checkpoint_before_apply=True,
    )


@pytest.fixture
def agent(session, agent_config):
    """CodingAgent with sandbox session."""
    return CodingAgent(
        sandbox_session=session,
        config=agent_config,
    )


def _mock_llm_plan(prompt, system="", model=None):
    """Mock LLM that returns a valid plan JSON."""
    return json.dumps({
        "summary": "Create hello.py and test it",
        "estimated_files": 1,
        "steps": [
            {
                "step_type": "create",
                "description": "Create hello.py",
                "file_path": "hello.py",
                "content": "print('hello world')\n",
            },
            {
                "step_type": "bash",
                "description": "Run hello.py",
                "command": "python3 /workspace/hello.py",
            },
        ],
    })


def _mock_llm_fix(prompt, system="", model=None):
    """Mock LLM that returns a valid fix JSON."""
    return json.dumps({
        "analysis": "Missing newline",
        "fix_type": "str_replace",
        "file_path": "hello.py",
        "old_str": "print('hello world')",
        "new_str": "print('hello world')\n# fixed",
    })


# ---------------------------------------------------------------------------
# TestCodingAgentConfig
# ---------------------------------------------------------------------------

class TestCodingAgentConfig:
    """Tests for CodingAgentConfig and config loading."""

    def test_default_config(self):
        cfg = CodingAgentConfig()
        assert cfg.enabled is True
        assert cfg.max_iterations == 10
        assert cfg.max_fix_retries == 3
        assert cfg.auto_test is True
        assert cfg.checkpoint_before_apply is True

    def test_config_checkpoint_always_true(self):
        cfg = CodingAgentConfig(checkpoint_before_apply=False)
        # The dataclass allows False but CodingAgent constructor enforces True
        agent = CodingAgent(config=cfg)
        assert agent.config.checkpoint_before_apply is True

    def test_config_custom_values(self):
        cfg = CodingAgentConfig(
            max_iterations=20,
            max_fix_retries=5,
            auto_test=False,
            default_model="qwen3:32b",
            planning_model="deepseek-r1:32b",
        )
        assert cfg.max_iterations == 20
        assert cfg.max_fix_retries == 5
        assert cfg.auto_test is False
        assert cfg.default_model == "qwen3:32b"
        assert cfg.planning_model == "deepseek-r1:32b"

    def test_config_context_window_reserve(self):
        cfg = CodingAgentConfig(context_window_reserve=4096)
        assert cfg.context_window_reserve == 4096


# ---------------------------------------------------------------------------
# TestEnumsAndDataclasses
# ---------------------------------------------------------------------------

class TestEnumsAndDataclasses:
    """Tests for enums and data classes."""

    def test_coding_phase_values(self):
        assert CodingPhase.IDLE.value == "idle"
        assert CodingPhase.PLANNING.value == "planning"
        assert CodingPhase.IMPLEMENTING.value == "implementing"
        assert CodingPhase.TESTING.value == "testing"
        assert CodingPhase.FIXING.value == "fixing"
        assert CodingPhase.REVIEWING.value == "reviewing"
        assert CodingPhase.APPLYING.value == "applying"
        assert CodingPhase.COMPLETED.value == "completed"
        assert CodingPhase.ABORTED.value == "aborted"
        assert CodingPhase.FAILED.value == "failed"

    def test_checkpoint_result_values(self):
        assert CheckpointResult.APPROVE.value == "approve"
        assert CheckpointResult.MODIFY.value == "modify"
        assert CheckpointResult.ABORT.value == "abort"

    def test_plan_step_type_values(self):
        assert PlanStepType.CREATE.value == "create"
        assert PlanStepType.EDIT.value == "edit"
        assert PlanStepType.TEST.value == "test"
        assert PlanStepType.BASH.value == "bash"

    def test_plan_step_defaults(self):
        step = PlanStep(step_number=1, step_type=PlanStepType.CREATE, description="test")
        assert step.file_path == ""
        assert step.completed is False
        assert step.result == ""
        assert step.error == ""

    def test_coding_plan_properties(self):
        plan = CodingPlan(
            task="test task",
            steps=[
                PlanStep(1, PlanStepType.CREATE, "step 1", completed=True),
                PlanStep(2, PlanStepType.BASH, "step 2", completed=False),
            ],
        )
        assert plan.total_steps == 2
        assert plan.completed_steps == 1

    def test_coding_plan_to_dict(self):
        plan = CodingPlan(task="test", summary="summary", estimated_files=3)
        d = plan.to_dict()
        assert d["task"] == "test"
        assert d["summary"] == "summary"
        assert d["estimated_files"] == 3
        assert d["total_steps"] == 0
        assert isinstance(d["steps"], list)

    def test_coding_history_entry(self):
        entry = CodingHistoryEntry(
            phase="testing", action="run_tests", detail="passed", success=True
        )
        assert entry.phase == "testing"
        assert entry.success is True
        assert entry.timestamp > 0

    def test_file_diff_to_dict(self):
        diff = FileDiff(
            path="test.py", is_new=True,
            diff_lines=["+hello"], modified_content="hello"
        )
        d = diff.to_dict()
        assert d["path"] == "test.py"
        assert d["is_new"] is True
        assert "+hello" in d["diff"]

    def test_test_result_defaults(self):
        tr = TestResult()
        assert tr.passed is False
        assert tr.output == ""
        assert tr.return_code == -1


# ---------------------------------------------------------------------------
# TestJSONParsing
# ---------------------------------------------------------------------------

class TestJSONParsing:
    """Tests for LLM response parsing helpers."""

    def test_parse_clean_json(self):
        data = _parse_json_response('{"key": "value"}')
        assert data["key"] == "value"

    def test_parse_json_with_markdown_fences(self):
        text = '```json\n{"key": "value"}\n```'
        data = _parse_json_response(text)
        assert data["key"] == "value"

    def test_parse_json_with_plain_fences(self):
        text = '```\n{"key": 42}\n```'
        data = _parse_json_response(text)
        assert data["key"] == 42

    def test_parse_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            _parse_json_response("not json at all")

    def test_build_plan_from_response(self):
        data = {
            "summary": "Test plan",
            "estimated_files": 2,
            "steps": [
                {"step_type": "create", "description": "Create file", "file_path": "a.py"},
                {"step_type": "bash", "description": "Run", "command": "echo hi"},
            ],
        }
        plan = _build_plan_from_response("task", data)
        assert plan.task == "task"
        assert plan.summary == "Test plan"
        assert plan.total_steps == 2
        assert plan.steps[0].step_type == PlanStepType.CREATE
        assert plan.steps[1].step_type == PlanStepType.BASH

    def test_build_plan_unknown_step_type(self):
        data = {"steps": [{"step_type": "unknown", "description": "x"}]}
        plan = _build_plan_from_response("t", data)
        assert plan.steps[0].step_type == PlanStepType.BASH

    def test_build_fix_from_response(self):
        data = {
            "analysis": "bug found",
            "fix_type": "str_replace",
            "file_path": "a.py",
            "old_str": "old",
            "new_str": "new",
        }
        fix = _build_fix_from_response(data)
        assert fix["analysis"] == "bug found"
        assert fix["fix_type"] == "str_replace"
        assert fix["file_path"] == "a.py"


# ---------------------------------------------------------------------------
# TestCodingAgentLifecycle
# ---------------------------------------------------------------------------

class TestCodingAgentLifecycle:
    """Tests for agent creation and task lifecycle."""

    def test_agent_initial_state(self, agent):
        assert agent.phase == CodingPhase.IDLE
        assert agent.plan is None
        assert agent.history == []
        assert agent.task_id == ""
        assert agent.session_active is False

    def test_start_task(self, agent):
        task_id = agent.start_task("Create hello.py", allow_degraded=True)
        assert task_id.startswith("coding-")
        assert agent.task_id == task_id
        assert agent.session_active is True
        assert len(agent.history) >= 1

    def test_start_task_without_session_raises(self):
        agent = CodingAgent(
            sandbox_session=None,
            sandbox_manager=None,
            config=CodingAgentConfig(),
        )
        # Force session to None (bypassing fallback to module-level singleton)
        agent._session = None
        with pytest.raises(RuntimeError, match="not available"):
            agent.start_task("test")

    def test_start_task_while_busy_raises(self, agent):
        agent.start_task("task 1", allow_degraded=True)
        # Manually set phase to implementing
        agent._phase = CodingPhase.IMPLEMENTING
        with pytest.raises(RuntimeError, match="busy"):
            agent.start_task("task 2", allow_degraded=True)

    def test_start_task_after_completed(self, agent):
        agent.start_task("task 1", allow_degraded=True)
        agent._phase = CodingPhase.COMPLETED
        # Should not raise - can restart after completion
        agent._session.stop()
        task_id = agent.start_task("task 2", allow_degraded=True)
        assert task_id.startswith("coding-")

    def test_start_task_with_project_injection(self, agent, tmp):
        # Create a temp project dir
        proj = tmp / "myproject"
        proj.mkdir()
        (proj / "main.py").write_text("print('hi')\n")
        (proj / "lib.py").write_text("x = 1\n")

        agent.start_task(
            "Fix main.py",
            project_path=str(proj),
            allow_degraded=True,
        )
        assert agent.session_active is True
        # Check that originals were snapshotted
        assert len(agent._original_files) >= 1

    def test_get_status_idle(self, agent):
        status = agent.get_status()
        assert status["phase"] == "idle"
        assert status["task_id"] == ""
        assert status["plan"] is None

    def test_get_status_after_start(self, agent):
        agent.start_task("test task", allow_degraded=True)
        status = agent.get_status()
        assert status["task_id"].startswith("coding-")
        assert status["task"] == "test task"
        assert status["session_active"] is True


# ---------------------------------------------------------------------------
# TestPlanningPhase
# ---------------------------------------------------------------------------

class TestPlanningPhase:
    """Tests for plan generation and modification."""

    def test_generate_plan_without_llm(self, agent):
        agent.start_task("test", allow_degraded=True)
        plan = agent.generate_plan()
        assert plan.task == "test"
        assert "No LLM" in plan.summary or plan.total_steps == 0

    def test_generate_plan_with_mock_llm(self, agent):
        agent._llm_call = _mock_llm_plan
        agent.start_task("Create hello", allow_degraded=True)
        plan = agent.generate_plan()
        assert plan.total_steps == 2
        assert plan.steps[0].step_type == PlanStepType.CREATE
        assert plan.steps[1].step_type == PlanStepType.BASH
        assert plan.summary == "Create hello.py and test it"

    def test_generate_plan_without_task_raises(self, agent):
        with pytest.raises(RuntimeError, match="No task"):
            agent.generate_plan()

    def test_set_plan(self, agent):
        agent.start_task("test", allow_degraded=True)
        custom_plan = CodingPlan(
            task="custom",
            steps=[PlanStep(1, PlanStepType.BASH, "echo hi", command="echo hi")],
        )
        agent.set_plan(custom_plan)
        assert agent.plan is custom_plan
        assert agent.plan.total_steps == 1

    def test_generate_plan_llm_failure_fallback(self, agent):
        def bad_llm(prompt, system="", model=None):
            raise Exception("LLM timeout")

        agent._llm_call = bad_llm
        agent.start_task("test", allow_degraded=True)
        plan = agent.generate_plan()
        assert "failed" in plan.summary.lower() or plan.total_steps == 0

    def test_plan_phase_change(self, agent):
        agent._llm_call = _mock_llm_plan
        agent.start_task("test", allow_degraded=True)
        agent.generate_plan()
        assert agent.phase == CodingPhase.PLANNING


# ---------------------------------------------------------------------------
# TestExecutionLoop
# ---------------------------------------------------------------------------

class TestExecutionLoop:
    """Tests for step execution."""

    def test_execute_create_step(self, agent):
        agent.start_task("test", allow_degraded=True)
        plan = CodingPlan(
            task="test",
            steps=[PlanStep(
                1, PlanStepType.CREATE, "Create file",
                file_path="test.txt", content="hello\n",
            )],
        )
        agent.set_plan(plan)
        step = agent.execute_next_step()
        assert step is not None
        assert step.completed is True
        assert step.error == ""

    def test_execute_bash_step(self, agent):
        agent.start_task("test", allow_degraded=True)
        plan = CodingPlan(
            task="test",
            steps=[PlanStep(
                1, PlanStepType.BASH, "Echo",
                command="echo 'hello'",
            )],
        )
        agent.set_plan(plan)
        step = agent.execute_next_step()
        assert step is not None
        assert step.completed is True
        assert "hello" in step.result

    def test_execute_edit_step(self, agent):
        agent.start_task("test", allow_degraded=True)
        # First create a file
        agent._session.create_file("edit_me.txt", "old content here\n")
        plan = CodingPlan(
            task="test",
            steps=[PlanStep(
                1, PlanStepType.EDIT, "Edit file",
                file_path="edit_me.txt",
                old_str="old content",
                new_str="new content",
            )],
        )
        agent.set_plan(plan)
        step = agent.execute_next_step()
        assert step is not None
        assert step.completed is True

    def test_execute_returns_none_when_done(self, agent):
        agent.start_task("test", allow_degraded=True)
        plan = CodingPlan(task="test", steps=[
            PlanStep(1, PlanStepType.BASH, "echo", command="echo done"),
        ])
        agent.set_plan(plan)
        agent.execute_next_step()
        result = agent.execute_next_step()
        assert result is None

    def test_execute_no_plan_raises(self, agent):
        agent.start_task("test", allow_degraded=True)
        with pytest.raises(RuntimeError, match="No plan"):
            agent.execute_next_step()

    def test_execute_all_steps(self, agent):
        agent.start_task("test", allow_degraded=True)
        plan = CodingPlan(
            task="test",
            steps=[
                PlanStep(1, PlanStepType.BASH, "step1", command="echo 1"),
                PlanStep(2, PlanStepType.BASH, "step2", command="echo 2"),
                PlanStep(3, PlanStepType.BASH, "step3", command="echo 3"),
            ],
        )
        agent.set_plan(plan)
        executed = agent.execute_all_steps()
        assert len(executed) == 3
        assert all(s.completed for s in executed)

    def test_max_iterations_limit(self, agent):
        agent._config.max_iterations = 2
        agent.start_task("test", allow_degraded=True)
        plan = CodingPlan(
            task="test",
            steps=[
                PlanStep(i, PlanStepType.BASH, f"step{i}", command="echo x")
                for i in range(1, 6)
            ],
        )
        agent.set_plan(plan)
        executed = agent.execute_all_steps()
        # Should stop after max_iterations
        assert len(executed) <= 3  # 2 iterations + possible extra
        assert agent.phase == CodingPhase.FAILED

    def test_phase_changes_to_implementing(self, agent):
        agent.start_task("test", allow_degraded=True)
        plan = CodingPlan(task="test", steps=[
            PlanStep(1, PlanStepType.BASH, "echo", command="echo x"),
        ])
        agent.set_plan(plan)
        agent.execute_next_step()
        # Phase should have been IMPLEMENTING during execution
        assert agent._iteration > 0


# ---------------------------------------------------------------------------
# TestTestingPhase
# ---------------------------------------------------------------------------

class TestTestingPhase:
    """Tests for test execution in sandbox."""

    def test_run_tests_passing(self, agent):
        agent.start_task("test", allow_degraded=True)
        result = agent.run_tests(command="echo '1 passed'")
        assert result.passed is True
        assert "passed" in result.output.lower()

    def test_run_tests_failing(self, agent):
        agent.start_task("test", allow_degraded=True)
        result = agent.run_tests(command="echo 'FAILED 1 test'")
        assert result.passed is False

    def test_run_tests_error(self, agent):
        agent.start_task("test", allow_degraded=True)
        result = agent.run_tests(command="echo 'Error: module not found'")
        assert result.passed is False

    def test_test_results_accumulated(self, agent):
        agent.start_task("test", allow_degraded=True)
        agent.run_tests(command="echo 'passed'")
        agent.run_tests(command="echo 'FAILED'")
        assert len(agent.test_results) == 2
        assert agent.test_results[0].passed is True
        assert agent.test_results[1].passed is False

    def test_run_tests_phase(self, agent):
        agent.start_task("test", allow_degraded=True)
        agent.run_tests(command="echo ok")
        assert agent.phase == CodingPhase.TESTING


# ---------------------------------------------------------------------------
# TestFixLoop
# ---------------------------------------------------------------------------

class TestFixLoop:
    """Tests for the fix loop."""

    def test_fix_loop_no_llm(self, agent):
        agent.start_task("test", allow_degraded=True)
        result = TestResult(passed=False, output="FAILED")
        fixed = agent._fix_loop(result)
        assert fixed is False

    def test_fix_loop_with_mock(self, agent):
        agent._llm_call = _mock_llm_fix
        agent.start_task("test", allow_degraded=True)
        # Create the file that the fix will edit
        agent._session.create_file("hello.py", "print('hello world')\n")
        # Set plan so fix prompt can find files
        agent._plan = CodingPlan(task="test", steps=[
            PlanStep(1, PlanStepType.CREATE, "create", file_path="hello.py", completed=True)
        ])
        # The fix replaces content, then re-runs tests.
        # With echo test command, tests will "pass".
        agent._config.auto_test_command = "echo '1 passed'"
        result = TestResult(passed=False, output="FAILED: assertion error")
        fixed = agent._fix_loop(result)
        assert fixed is True
        assert agent._fix_count >= 1

    def test_fix_loop_exhausts_retries(self, agent):
        call_count = [0]

        def always_fail_fix(prompt, system="", model=None):
            call_count[0] += 1
            return json.dumps({
                "analysis": "unknown bug",
                "fix_type": "create_file",
                "file_path": "hello.py",
                "content": "still broken\n",
            })

        agent._llm_call = always_fail_fix
        agent._config.max_fix_retries = 2
        # Use a test command that always reports failure
        agent._config.auto_test_command = "echo 'FAILED'"
        agent.start_task("test", allow_degraded=True)
        agent._plan = CodingPlan(task="test", steps=[])

        result = TestResult(passed=False, output="FAILED")
        fixed = agent._fix_loop(result)
        assert fixed is False
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# TestDiffGeneration
# ---------------------------------------------------------------------------

class TestDiffGeneration:
    """Tests for diff generation between original and modified files."""

    def test_generate_diffs_no_changes(self, agent):
        agent.start_task("test", allow_degraded=True)
        diffs = agent.generate_diffs()
        assert len(diffs) == 0

    def test_generate_diffs_new_file(self, agent):
        agent.start_task("test", allow_degraded=True)
        agent._session.create_file("new_file.py", "print('new')\n")
        diffs = agent.generate_diffs()
        assert len(diffs) == 1
        assert diffs[0].is_new is True
        assert diffs[0].path == "new_file.py"

    def test_generate_diffs_modified_file(self, agent, tmp):
        agent.start_task("test", allow_degraded=True)
        # Create file directly in sandbox, snapshot it manually
        agent._session.create_file("main.py", "old code\n")
        agent._original_files["main.py"] = "old code\n"
        # Modify the file in sandbox
        agent._session.str_replace("main.py", "old code", "new code")
        diffs = agent.generate_diffs()
        assert len(diffs) == 1
        assert diffs[0].is_new is False
        assert diffs[0].is_deleted is False
        assert len(diffs[0].diff_lines) > 0

    def test_generate_diffs_phase(self, agent):
        agent.start_task("test", allow_degraded=True)
        agent.generate_diffs()
        assert agent.phase == CodingPhase.REVIEWING


# ---------------------------------------------------------------------------
# TestApplyPhase
# ---------------------------------------------------------------------------

class TestApplyPhase:
    """Tests for the apply phase (human-gated)."""

    def test_apply_no_target_raises(self, agent):
        agent.start_task("test", allow_degraded=True)
        with pytest.raises(RuntimeError, match="No target path"):
            agent.apply_changes()

    def test_apply_no_changes(self, agent, tmp):
        target = tmp / "target"
        target.mkdir()
        agent.start_task("test", allow_degraded=True)
        result = agent.apply_changes(target_path=str(target))
        assert result["applied"] == 0
        assert agent.phase == CodingPhase.COMPLETED

    def test_apply_new_file(self, agent, tmp):
        target = tmp / "target"
        target.mkdir()

        agent.start_task("test", allow_degraded=True)
        agent._session.create_file("output.py", "print('applied')\n")
        agent.generate_diffs()
        result = agent.apply_changes(target_path=str(target))
        assert result["applied"] == 1
        assert (target / "output.py").exists()
        assert (target / "output.py").read_text() == "print('applied')\n"

    def test_apply_modified_file(self, agent, tmp):
        target = tmp / "target"
        target.mkdir()
        (target / "app.py").write_text("version = 1\n")

        agent.start_task("test", allow_degraded=True)
        # Create and snapshot original
        agent._session.create_file("app.py", "version = 1\n")
        agent._original_files["app.py"] = "version = 1\n"
        # Modify
        agent._session.str_replace("app.py", "version = 1", "version = 2")
        agent.generate_diffs()
        result = agent.apply_changes(target_path=str(target))
        assert result["applied"] == 1
        assert "version = 2" in (target / "app.py").read_text()

    def test_apply_uses_project_path(self, agent, tmp):
        proj = tmp / "proj"
        proj.mkdir()
        (proj / "test.txt").write_text("original\n")

        agent.start_task("test", project_path=str(proj), allow_degraded=True)
        agent._session.create_file("new.txt", "new file\n")
        agent.generate_diffs()
        result = agent.apply_changes()  # no target_path — uses project_path
        assert result["applied"] >= 1
        assert (proj / "new.txt").exists()

    def test_apply_creates_subdirectories(self, agent, tmp):
        target = tmp / "target"
        target.mkdir()

        agent.start_task("test", allow_degraded=True)
        agent._session.bash("mkdir -p /workspace/sub/dir")
        agent._session.create_file("sub/dir/deep.py", "deep = True\n")
        agent.generate_diffs()
        result = agent.apply_changes(target_path=str(target))
        assert (target / "sub" / "dir" / "deep.py").exists()


# ---------------------------------------------------------------------------
# TestAbort
# ---------------------------------------------------------------------------

class TestAbort:
    """Tests for task abortion."""

    def test_abort_sets_phase(self, agent):
        agent.start_task("test", allow_degraded=True)
        agent.abort()
        assert agent.phase == CodingPhase.ABORTED

    def test_abort_cleans_up_session(self, agent):
        agent.start_task("test", allow_degraded=True)
        assert agent.session_active is True
        # Agent does not own session in fixture (owns_session=False)
        agent._owns_session = True
        agent.abort()
        assert agent.session_active is False

    def test_abort_returns_true(self, agent):
        agent.start_task("test", allow_degraded=True)
        agent._owns_session = True
        result = agent.abort()
        assert result is True


# ---------------------------------------------------------------------------
# TestProgressCallbacks
# ---------------------------------------------------------------------------

class TestProgressCallbacks:
    """Tests for progress event emission."""

    def test_callback_on_start(self, agent):
        events = []
        agent.add_progress_callback(lambda e: events.append(e))
        agent.start_task("test", allow_degraded=True)
        assert any(e["type"] == "task_started" for e in events)

    def test_callback_on_step(self, agent):
        events = []
        agent.add_progress_callback(lambda e: events.append(e))
        agent.start_task("test", allow_degraded=True)
        agent.set_plan(CodingPlan(task="t", steps=[
            PlanStep(1, PlanStepType.BASH, "echo", command="echo x"),
        ]))
        agent.execute_next_step()
        types = [e["type"] for e in events]
        assert "step_start" in types
        assert "step_complete" in types

    def test_callback_on_test(self, agent):
        events = []
        agent.add_progress_callback(lambda e: events.append(e))
        agent.start_task("test", allow_degraded=True)
        agent.run_tests(command="echo passed")
        assert any(e["type"] == "test_result" for e in events)

    def test_callback_error_does_not_crash(self, agent):
        def bad_callback(e):
            raise ValueError("callback error")

        agent.add_progress_callback(bad_callback)
        # Should not raise
        agent.start_task("test", allow_degraded=True)

    def test_callback_on_diffs(self, agent):
        events = []
        agent.add_progress_callback(lambda e: events.append(e))
        agent.start_task("test", allow_degraded=True)
        agent._session.create_file("f.txt", "data\n")
        agent.generate_diffs()
        assert any(e["type"] == "diffs_ready" for e in events)


# ---------------------------------------------------------------------------
# TestSecurity
# ---------------------------------------------------------------------------

class TestSecurity:
    """Tests for security constraints."""

    def test_checkpoint_before_apply_always_true(self):
        cfg = CodingAgentConfig(checkpoint_before_apply=False)
        agent = CodingAgent(config=cfg)
        assert agent.config.checkpoint_before_apply is True

    def test_checkpoint_before_apply_in_loaded_config(self):
        cfg = CodingAgentConfig()
        cfg.checkpoint_before_apply = False
        agent = CodingAgent(config=cfg)
        # Constructor always forces True
        assert agent.config.checkpoint_before_apply is True

    def test_sandbox_tools_used(self, agent):
        """Agent must use SandboxToolSession, not raw filesystem."""
        assert agent._session is not None
        assert isinstance(agent._session, SandboxToolSession)

    def test_read_raw_uses_shlex_quote(self):
        """Verify _read_raw uses shlex.quote (check source code)."""
        import inspect
        source = inspect.getsource(CodingAgent._read_raw)
        assert "shlex.quote" in source

    def test_apply_rejects_root(self, agent, tmp):
        agent.start_task("test", allow_degraded=True)
        agent._session.create_file("x.py", "x=1\n")
        agent.generate_diffs()
        with pytest.raises(ValueError, match="system directory"):
            agent.apply_changes(target_path="/")

    def test_apply_rejects_etc(self, agent, tmp):
        agent.start_task("test", allow_degraded=True)
        agent._session.create_file("x.py", "x=1\n")
        agent.generate_diffs()
        with pytest.raises(ValueError, match="system directory|protected|top-level"):
            agent.apply_changes(target_path="/etc")

    def test_apply_rejects_home(self, agent, tmp):
        agent.start_task("test", allow_degraded=True)
        agent._session.create_file("x.py", "x=1\n")
        agent.generate_diffs()
        with pytest.raises(ValueError, match="system directory|protected|top-level"):
            agent.apply_changes(target_path="/home")

    def test_apply_accepts_deep_project_path(self, agent, tmp):
        target = tmp / "myproject"
        target.mkdir()
        agent.start_task("test", allow_degraded=True)
        agent._session.create_file("x.py", "x=1\n")
        agent.generate_diffs()
        # Should NOT raise for a deep user path
        result = agent.apply_changes(target_path=str(target))
        assert result["applied"] >= 1

    def test_diff_integrity_hash_set(self, agent):
        agent.start_task("test", allow_degraded=True)
        agent._session.create_file("f.py", "a=1\n")
        agent.generate_diffs()
        assert agent._diffs_hash != ""
        assert len(agent._diffs_hash) == 64  # SHA-256 hex

    def test_diff_integrity_blocks_tampered_apply(self, agent, tmp):
        target = tmp / "out"
        target.mkdir()
        agent.start_task("test", allow_degraded=True)
        agent._session.create_file("f.py", "a=1\n")
        agent.generate_diffs()
        # Tamper: change diffs content without regenerating hash
        agent._diffs[0].modified_content = "TAMPERED"
        with pytest.raises(RuntimeError, match="integrity"):
            agent.apply_changes(target_path=str(target))


# ---------------------------------------------------------------------------
# TestStatus
# ---------------------------------------------------------------------------

class TestStatus:
    """Tests for status reporting."""

    def test_status_includes_all_fields(self, agent):
        agent.start_task("test task", allow_degraded=True)
        status = agent.get_status()
        required_keys = [
            "task_id", "task", "phase", "session_active", "plan",
            "current_step", "total_steps", "iteration", "max_iterations",
            "fix_count", "max_fix_retries", "test_results", "diffs",
            "history_count", "history",
        ]
        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_status_history_limit(self, agent):
        agent.start_task("test", allow_degraded=True)
        # Add many history entries
        for i in range(100):
            agent._log("test", f"action_{i}", f"detail_{i}")
        status = agent.get_status()
        # History is capped at 50 in status
        assert len(status["history"]) <= 50
        assert status["history_count"] >= 100

    def test_status_with_plan(self, agent):
        agent._llm_call = _mock_llm_plan
        agent.start_task("test", allow_degraded=True)
        agent.generate_plan()
        status = agent.get_status()
        assert status["plan"] is not None
        assert status["total_steps"] == 2
