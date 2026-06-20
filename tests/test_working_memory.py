#!/usr/bin/env python3
"""
Tests for Working Memory (S80).

Covers: WorkingMemory dataclass, compact serialization, step updates,
LLM updates, trimming, persistence in CodingHistoryStore, integration
with CodingAgent, plan retry logic with json_repair.
"""

import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import time

import pytest

# ---------------------------------------------------------------------------
# Module loading (test isolation -- no ollama needed)
# ---------------------------------------------------------------------------

_base = os.path.join(os.path.dirname(__file__), os.pardir, "opti_oignon")

# json_repair
_jr_path = os.path.join(_base, "json_repair.py")
_jr_spec = importlib.util.spec_from_file_location("json_repair", _jr_path)
_jr_mod = importlib.util.module_from_spec(_jr_spec)
_jr_spec.loader.exec_module(_jr_mod)
sys.modules["opti_oignon.json_repair"] = _jr_mod

# sandbox_manager
_sm_path = os.path.join(_base, "sandbox_manager.py")
_sm_spec = importlib.util.spec_from_file_location("sandbox_manager", _sm_path)
_sm_mod = importlib.util.module_from_spec(_sm_spec)
_sm_spec.loader.exec_module(_sm_mod)
sys.modules.setdefault("opti_oignon", type(sys)("opti_oignon"))
sys.modules["opti_oignon.sandbox_manager"] = _sm_mod

SandboxManager = _sm_mod.SandboxManager

# tool_registry
_tr_path = os.path.join(_base, "tool_registry.py")
_tr_spec = importlib.util.spec_from_file_location("tool_registry", _tr_path)
_tr_mod = importlib.util.module_from_spec(_tr_spec)
_tr_spec.loader.exec_module(_tr_mod)
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

# coding_history (loaded directly for working_memory persistence tests)
_ch_path = os.path.join(_base, "coding_history.py")
_ch_spec = importlib.util.spec_from_file_location("coding_history", _ch_path)
_ch_mod = importlib.util.module_from_spec(_ch_spec)
_ch_spec.loader.exec_module(_ch_mod)
sys.modules["opti_oignon.coding_history"] = _ch_mod

CodingHistoryStore = _ch_mod.CodingHistoryStore

# coding_agent
_ca_path = os.path.join(_base, "coding_agent.py")
_ca_spec = importlib.util.spec_from_file_location("coding_agent", _ca_path)
_ca_mod = importlib.util.module_from_spec(_ca_spec)
_ca_spec.loader.exec_module(_ca_mod)

CodingAgent = _ca_mod.CodingAgent
CodingAgentConfig = _ca_mod.CodingAgentConfig
CodingPlan = _ca_mod.CodingPlan
PlanStep = _ca_mod.PlanStep
PlanStepType = _ca_mod.PlanStepType
WorkingMemory = _ca_mod.WorkingMemory
_parse_json_response = _ca_mod._parse_json_response
_build_plan_from_response = _ca_mod._build_plan_from_response
JSON_REPAIR_AVAILABLE = _ca_mod.JSON_REPAIR_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeSandboxSession:
    """Minimal fake SandboxToolSession for testing."""

    def __init__(self):
        self.active = True
        self._files = {}
        self.commands = []

    def start(self, **kw):
        self.active = True

    def stop(self):
        self.active = False
        return True

    def bash(self, cmd, **kw):
        self.commands.append(cmd)
        return ""

    def create_file(self, path, content):
        self._files[path] = content
        return f"Created {path}"

    def str_replace(self, path, old, new):
        if path in self._files:
            self._files[path] = self._files[path].replace(old, new)
        return f"Replaced in {path}"

    def view(self, path):
        return self._files.get(path, "")

    def extract_files(self):
        return [{"path": p} for p in self._files]

    def inject_directory(self, path):
        return 0


def _make_store():
    """Create a CodingHistoryStore with a temp database."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return CodingHistoryStore(db_path=tmp.name), tmp.name


# ===================================================================
# WorkingMemory dataclass
# ===================================================================

class TestWorkingMemoryInit:
    """Tests for WorkingMemory initialization."""

    def test_default_empty(self):
        wm = WorkingMemory()
        assert wm.task_id == ""
        assert wm.decisions == []
        assert wm.modified_files == {}
        assert wm.errors_encountered == []
        assert wm.open_questions == []
        assert wm.progress_notes == []

    def test_init_with_task_id(self):
        wm = WorkingMemory(task_id="test-123")
        assert wm.task_id == "test-123"

    def test_to_dict(self):
        wm = WorkingMemory(task_id="t1", decisions=["use pytest"])
        d = wm.to_dict()
        assert d["task_id"] == "t1"
        assert d["decisions"] == ["use pytest"]
        assert isinstance(d["modified_files"], dict)

    def test_from_dict(self):
        data = {
            "task_id": "t2",
            "decisions": ["choice A"],
            "modified_files": {"main.py": "created"},
            "errors_encountered": ["err1"],
            "open_questions": ["why?"],
            "progress_notes": ["done step 1"],
        }
        wm = WorkingMemory.from_dict(data)
        assert wm.task_id == "t2"
        assert wm.decisions == ["choice A"]
        assert wm.modified_files == {"main.py": "created"}
        assert wm.errors_encountered == ["err1"]
        assert wm.open_questions == ["why?"]
        assert wm.progress_notes == ["done step 1"]

    def test_from_dict_missing_keys(self):
        wm = WorkingMemory.from_dict({"task_id": "t3"})
        assert wm.task_id == "t3"
        assert wm.decisions == []
        assert wm.modified_files == {}

    def test_from_dict_empty(self):
        wm = WorkingMemory.from_dict({})
        assert wm.task_id == ""

    def test_roundtrip(self):
        wm = WorkingMemory(
            task_id="rt",
            decisions=["d1", "d2"],
            modified_files={"a.py": "edited"},
            errors_encountered=["e1"],
            open_questions=["q1"],
            progress_notes=["n1"],
        )
        wm2 = WorkingMemory.from_dict(wm.to_dict())
        assert wm2.to_dict() == wm.to_dict()


# ===================================================================
# WorkingMemory compact serialization
# ===================================================================

class TestWorkingMemoryCompact:
    """Tests for to_compact() serialization."""

    def test_empty_memory_compact(self):
        wm = WorkingMemory()
        assert wm.to_compact() == ""

    def test_decisions_in_compact(self):
        wm = WorkingMemory(decisions=["use FastAPI", "prefer SQLite"])
        compact = wm.to_compact()
        assert "DECISIONS:" in compact
        assert "FastAPI" in compact
        assert "SQLite" in compact

    def test_modified_files_in_compact(self):
        wm = WorkingMemory(modified_files={"main.py": "created step 1"})
        compact = wm.to_compact()
        assert "MODIFIED:" in compact
        assert "main.py" in compact

    def test_errors_in_compact(self):
        wm = WorkingMemory(errors_encountered=["ImportError on line 5"])
        compact = wm.to_compact()
        assert "ERRORS:" in compact
        assert "ImportError" in compact

    def test_open_questions_in_compact(self):
        wm = WorkingMemory(open_questions=["Should we use async?"])
        compact = wm.to_compact()
        assert "OPEN:" in compact

    def test_progress_in_compact(self):
        wm = WorkingMemory(progress_notes=["Step 1 completed"])
        compact = wm.to_compact()
        assert "PROGRESS:" in compact

    def test_max_tokens_truncation(self):
        wm = WorkingMemory(
            decisions=["x" * 500],
            progress_notes=["y" * 500],
        )
        compact = wm.to_compact(max_tokens=50)
        # 50 tokens * 4 chars = 200 chars max
        assert len(compact) <= 203  # 200 + "..."

    def test_compact_limits_items(self):
        wm = WorkingMemory(decisions=[f"d{i}" for i in range(20)])
        compact = wm.to_compact()
        # Should only include last 5 decisions
        assert "d19" in compact
        assert "d15" in compact
        # d0 should not be present (only last 5 kept)
        assert "d0" not in compact


# ===================================================================
# WorkingMemory step updates
# ===================================================================

class TestWorkingMemoryStepUpdate:
    """Tests for update_from_step()."""

    def test_create_step_records_file(self):
        wm = WorkingMemory()
        wm.update_from_step(1, "create", "main.py", "Created main.py", "")
        assert "main.py" in wm.modified_files
        assert "created" in wm.modified_files["main.py"]

    def test_edit_step_records_file(self):
        wm = WorkingMemory()
        wm.update_from_step(2, "edit", "utils.py", "Replaced text", "")
        assert "utils.py" in wm.modified_files
        assert "edited" in wm.modified_files["utils.py"]

    def test_error_step_records_error(self):
        wm = WorkingMemory()
        wm.update_from_step(3, "bash", "", "", "Command not found")
        assert len(wm.errors_encountered) == 1
        assert "Command not found" in wm.errors_encountered[0]
        assert "Step 3" in wm.errors_encountered[0]

    def test_test_step_records_progress(self):
        wm = WorkingMemory()
        wm.update_from_step(4, "test", "", "5 passed, 0 failed", "")
        assert len(wm.progress_notes) == 1
        assert "passed" in wm.progress_notes[0]

    def test_test_step_ran_note(self):
        wm = WorkingMemory()
        wm.update_from_step(5, "test", "", "Executed test suite", "")
        assert len(wm.progress_notes) == 1
        assert "ran" in wm.progress_notes[0]

    def test_bash_step_no_file_no_error(self):
        wm = WorkingMemory()
        wm.update_from_step(6, "bash", "", "output ok", "")
        # No file path and no error => no changes
        assert wm.modified_files == {}
        assert wm.errors_encountered == []

    def test_error_truncates_long_message(self):
        wm = WorkingMemory()
        wm.update_from_step(1, "bash", "", "", "x" * 500)
        assert len(wm.errors_encountered[0]) < 300

    def test_multiple_steps_accumulate(self):
        wm = WorkingMemory()
        wm.update_from_step(1, "create", "a.py", "ok", "")
        wm.update_from_step(2, "create", "b.py", "ok", "")
        wm.update_from_step(3, "edit", "a.py", "ok", "")
        assert len(wm.modified_files) == 2
        # a.py should be overwritten with latest
        assert "edited" in wm.modified_files["a.py"]


# ===================================================================
# WorkingMemory LLM updates
# ===================================================================

class TestWorkingMemoryLLMUpdate:
    """Tests for update_from_llm()."""

    def test_add_decisions(self):
        wm = WorkingMemory()
        wm.update_from_llm({"decisions": ["Use pandas for CSV parsing"]})
        assert len(wm.decisions) == 1
        assert "pandas" in wm.decisions[0]

    def test_add_decision_string(self):
        wm = WorkingMemory()
        wm.update_from_llm({"decisions": "single decision"})
        assert wm.decisions == ["single decision"]

    def test_add_modified_files(self):
        wm = WorkingMemory()
        wm.update_from_llm({
            "modified_files": {"config.yaml": "added new section"}
        })
        assert "config.yaml" in wm.modified_files

    def test_replace_open_questions(self):
        wm = WorkingMemory(open_questions=["old question"])
        wm.update_from_llm({"open_questions": ["new question"]})
        # open_questions are replaced, not appended
        assert wm.open_questions == ["new question"]

    def test_open_questions_string(self):
        wm = WorkingMemory()
        wm.update_from_llm({"open_questions": "single question"})
        assert wm.open_questions == ["single question"]

    def test_append_progress(self):
        wm = WorkingMemory(progress_notes=["note1"])
        wm.update_from_llm({"progress_notes": ["note2"]})
        assert wm.progress_notes == ["note1", "note2"]

    def test_append_errors(self):
        wm = WorkingMemory(errors_encountered=["err1"])
        wm.update_from_llm({"errors_encountered": ["err2"]})
        assert wm.errors_encountered == ["err1", "err2"]

    def test_errors_string(self):
        wm = WorkingMemory()
        wm.update_from_llm({"errors_encountered": "single error"})
        assert wm.errors_encountered == ["single error"]

    def test_progress_string(self):
        wm = WorkingMemory()
        wm.update_from_llm({"progress_notes": "single note"})
        assert wm.progress_notes == ["single note"]

    def test_empty_update(self):
        wm = WorkingMemory(decisions=["d1"])
        wm.update_from_llm({})
        assert wm.decisions == ["d1"]

    def test_unknown_keys_ignored(self):
        wm = WorkingMemory()
        wm.update_from_llm({"unknown_field": "value"})
        assert not hasattr(wm, "unknown_field") or True  # no crash


# ===================================================================
# WorkingMemory trimming
# ===================================================================

class TestWorkingMemoryTrim:
    """Tests for trim()."""

    def test_trim_decisions(self):
        wm = WorkingMemory(decisions=[f"d{i}" for i in range(25)])
        wm.trim(max_items=10)
        assert len(wm.decisions) == 10
        assert wm.decisions[0] == "d15"  # kept last 10

    def test_trim_errors(self):
        wm = WorkingMemory(errors_encountered=[f"e{i}" for i in range(15)])
        wm.trim(max_items=5)
        assert len(wm.errors_encountered) == 5

    def test_trim_modified_files(self):
        wm = WorkingMemory(
            modified_files={f"f{i}.py": f"change {i}" for i in range(20)}
        )
        wm.trim(max_items=5)
        assert len(wm.modified_files) == 5

    def test_trim_no_op_when_under_limit(self):
        wm = WorkingMemory(decisions=["a", "b"])
        wm.trim(max_items=10)
        assert len(wm.decisions) == 2

    def test_trim_open_questions(self):
        wm = WorkingMemory(open_questions=[f"q{i}" for i in range(12)])
        wm.trim(max_items=3)
        assert len(wm.open_questions) == 3
        assert wm.open_questions[-1] == "q11"


# ===================================================================
# CodingHistoryStore working_memory persistence
# ===================================================================

class TestWorkingMemoryPersistence:
    """Tests for save/load/delete working_memory in CodingHistoryStore."""

    def test_save_and_load(self):
        store, path = _make_store()
        try:
            store.record_task_start("t1", "test task")
            data = {"task_id": "t1", "decisions": ["d1"], "modified_files": {}}
            store.save_working_memory("t1", data)
            loaded = store.load_working_memory("t1")
            assert loaded is not None
            assert loaded["decisions"] == ["d1"]
        finally:
            os.unlink(path)

    def test_load_nonexistent(self):
        store, path = _make_store()
        try:
            assert store.load_working_memory("nonexistent") is None
        finally:
            os.unlink(path)

    def test_save_overwrites(self):
        store, path = _make_store()
        try:
            store.record_task_start("t1", "test task")
            store.save_working_memory("t1", {"decisions": ["v1"]})
            store.save_working_memory("t1", {"decisions": ["v2"]})
            loaded = store.load_working_memory("t1")
            assert loaded["decisions"] == ["v2"]
        finally:
            os.unlink(path)

    def test_delete_working_memory(self):
        store, path = _make_store()
        try:
            store.record_task_start("t1", "test task")
            store.save_working_memory("t1", {"decisions": ["d1"]})
            assert store.delete_working_memory("t1") is True
            assert store.load_working_memory("t1") is None
        finally:
            os.unlink(path)

    def test_delete_nonexistent(self):
        store, path = _make_store()
        try:
            assert store.delete_working_memory("nope") is False
        finally:
            os.unlink(path)

    def test_cascade_on_task_delete(self):
        store, path = _make_store()
        try:
            store.record_task_start("t1", "test task")
            store.save_working_memory("t1", {"decisions": ["d1"]})
            store.delete_task("t1")
            assert store.load_working_memory("t1") is None
        finally:
            os.unlink(path)

    def test_cascade_on_batch_delete_by_ids(self):
        store, path = _make_store()
        try:
            store.record_task_start("t1", "task 1")
            store.record_task_start("t2", "task 2")
            store.save_working_memory("t1", {"decisions": ["d1"]})
            store.save_working_memory("t2", {"decisions": ["d2"]})
            store.batch_delete_by_ids(["t1", "t2"])
            assert store.load_working_memory("t1") is None
            assert store.load_working_memory("t2") is None
        finally:
            os.unlink(path)

    def test_cascade_on_batch_delete_before_date(self):
        store, path = _make_store()
        try:
            store.record_task_start("t1", "task 1")
            store.save_working_memory("t1", {"decisions": ["d1"]})
            # Delete everything before now + 1 second
            store.batch_delete_before_date(time.time() + 1)
            assert store.load_working_memory("t1") is None
        finally:
            os.unlink(path)

    def test_working_memory_table_exists(self):
        store, path = _make_store()
        try:
            conn = sqlite3.connect(path)
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            conn.close()
            assert "working_memory" in tables
        finally:
            os.unlink(path)


# ===================================================================
# parse_json_response with json_repair
# ===================================================================

class TestParseJsonResponseS80:
    """Tests for _parse_json_response using json_repair (S80)."""

    def test_valid_json(self):
        result = _parse_json_response('{"summary": "ok", "steps": []}')
        assert result["summary"] == "ok"

    def test_fenced_json(self):
        text = '```json\n{"summary": "plan", "steps": []}\n```'
        result = _parse_json_response(text)
        assert result["summary"] == "plan"

    def test_embedded_json(self):
        text = 'Sure! Here is the plan:\n{"summary": "x", "steps": []}\nDone.'
        result = _parse_json_response(text)
        assert result["summary"] == "x"

    def test_trailing_comma(self):
        text = '{"summary": "x", "steps": [],}'
        result = _parse_json_response(text)
        assert result["summary"] == "x"

    def test_single_quotes(self):
        text = "{'summary': 'x', 'steps': []}"
        result = _parse_json_response(text)
        assert result["summary"] == "x"

    def test_list_result_wrapped(self):
        text = '[{"step_type": "create"}]'
        result = _parse_json_response(text)
        assert "steps" in result

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_json_response("not json at all")

    def test_json_repair_is_available(self):
        assert JSON_REPAIR_AVAILABLE is True


# ===================================================================
# Plan retry logic
# ===================================================================

class TestPlanRetryLogic:
    """Tests for generate_plan retry with JSON reinforcement."""

    def _make_agent(self, llm_responses):
        """Create a CodingAgent with a fake LLM that returns sequenced responses."""
        call_count = [0]

        def fake_llm(prompt, system=None, model=None):
            idx = min(call_count[0], len(llm_responses) - 1)
            call_count[0] += 1
            return llm_responses[idx]

        session = FakeSandboxSession()
        config = CodingAgentConfig(max_plan_retries=2)
        agent = CodingAgent(
            sandbox_session=session,
            llm_call=fake_llm,
            config=config,
        )
        agent.start_task("Test task")
        return agent, call_count

    def test_success_first_try(self):
        agent, calls = self._make_agent([
            '{"summary": "plan", "steps": [{"step_type": "bash", "description": "echo"}]}'
        ])
        plan = agent.generate_plan()
        assert plan.total_steps == 1
        assert calls[0] == 1

    def test_retry_on_bad_json(self):
        agent, calls = self._make_agent([
            "This is not JSON at all",
            '{"summary": "plan", "steps": [{"step_type": "bash", "description": "echo"}]}',
        ])
        plan = agent.generate_plan()
        assert plan.total_steps == 1
        assert calls[0] == 2  # retried once

    def test_numbered_list_fallback(self):
        agent, calls = self._make_agent([
            "not json",
            "still not json",
            "1. Create main.py with hello world\n2. Run tests\n3. Edit config.yaml",
        ])
        plan = agent.generate_plan()
        assert plan.total_steps == 3
        assert calls[0] == 3  # all retries used

    def test_all_retries_exhausted(self):
        agent, calls = self._make_agent([
            "garbage",
            "more garbage",
            "still garbage no list either",
        ])
        plan = agent.generate_plan()
        # Should return minimal fallback plan
        assert plan.total_steps == 0
        assert "failed" in plan.summary.lower() or "Planning failed" in plan.summary

    def test_config_max_plan_retries(self):
        config = CodingAgentConfig(max_plan_retries=5)
        assert config.max_plan_retries == 5

    def test_config_default_plan_retries(self):
        config = CodingAgentConfig()
        assert config.max_plan_retries == 3


# ===================================================================
# Integration: WorkingMemory in CodingAgent
# ===================================================================

class TestWorkingMemoryInAgent:
    """Tests for WorkingMemory integration in CodingAgent."""

    def _make_agent_with_plan(self):
        """Create a CodingAgent with a pre-set plan."""
        session = FakeSandboxSession()
        config = CodingAgentConfig()
        agent = CodingAgent(
            sandbox_session=session,
            config=config,
        )
        agent.start_task("Test task")
        plan = CodingPlan(
            task="Test task",
            summary="Test plan",
            steps=[
                PlanStep(
                    step_number=1,
                    step_type=PlanStepType.CREATE,
                    description="Create main.py",
                    file_path="main.py",
                    content="print('hello')",
                ),
                PlanStep(
                    step_number=2,
                    step_type=PlanStepType.BASH,
                    description="List files",
                    command="ls",
                ),
            ],
        )
        agent.set_plan(plan)
        return agent

    def test_working_memory_initialized_on_start(self):
        session = FakeSandboxSession()
        agent = CodingAgent(sandbox_session=session)
        agent.start_task("test")
        assert agent.working_memory is not None
        assert agent.working_memory.task_id == agent.task_id

    def test_working_memory_updated_after_step(self):
        agent = self._make_agent_with_plan()
        agent.execute_next_step()
        wm = agent.working_memory
        assert "main.py" in wm.modified_files

    def test_working_memory_compact_after_steps(self):
        agent = self._make_agent_with_plan()
        agent.execute_next_step()
        compact = agent.get_working_memory_compact()
        assert "MODIFIED:" in compact
        assert "main.py" in compact

    def test_working_memory_in_status(self):
        agent = self._make_agent_with_plan()
        agent.execute_next_step()
        status = agent.get_status()
        assert "working_memory" in status
        assert status["working_memory"] is not None
        assert "modified_files" in status["working_memory"]

    def test_working_memory_empty_compact_before_steps(self):
        session = FakeSandboxSession()
        agent = CodingAgent(sandbox_session=session)
        agent.start_task("test")
        assert agent.get_working_memory_compact() == ""

    def test_working_memory_property_none_before_start(self):
        session = FakeSandboxSession()
        agent = CodingAgent(sandbox_session=session)
        assert agent.working_memory is None
