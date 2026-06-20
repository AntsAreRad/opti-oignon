#!/usr/bin/env python3
"""
Tests for coding_history.py — Opti-Oignon S76

Covers CodingHistoryStore: tables, task lifecycle, steps, tests,
checkpoints, resume, queries, pagination, stats, pruning, deletion,
edge cases, config, and thread safety.
"""

import importlib.util
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest

# ---------------------------------------------------------------------------
# Direct module loading (bypass __init__.py chain)
# ---------------------------------------------------------------------------

_MOD_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "coding_history.py",
)
_spec = importlib.util.spec_from_file_location("coding_history", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CodingHistoryStore = _mod.CodingHistoryStore
TaskSummary = _mod.TaskSummary
TaskDetail = _mod.TaskDetail
CheckpointState = _mod.CheckpointState
CODING_HISTORY_AVAILABLE = _mod.CODING_HISTORY_AVAILABLE
_load_config = _mod._load_config


def _tmp_store(**kwargs) -> CodingHistoryStore:
    """Create a store with a temp DB."""
    db = os.path.join(tempfile.mkdtemp(), "test_history.db")
    return CodingHistoryStore(db_path=db, **kwargs)


class TestModuleAvailability(unittest.TestCase):
    """Module-level flag should be True."""

    def test_available(self):
        self.assertTrue(CODING_HISTORY_AVAILABLE)


class TestConfigDefaults(unittest.TestCase):
    """Config loader returns sane defaults."""

    def test_default_enabled(self):
        cfg = _load_config("/nonexistent/path.yaml")
        self.assertTrue(cfg["enabled"])

    def test_default_max_tasks(self):
        cfg = _load_config("/nonexistent/path.yaml")
        self.assertEqual(cfg["max_tasks"], 200)

    def test_default_retention_days(self):
        cfg = _load_config("/nonexistent/path.yaml")
        self.assertEqual(cfg["retention_days"], 30)

    def test_default_max_output_length(self):
        cfg = _load_config("/nonexistent/path.yaml")
        self.assertEqual(cfg["max_output_length"], 10000)


class TestConfigFromYAML(unittest.TestCase):
    """Config loader reads YAML when available."""

    def test_loads_yaml(self):
        config_path = os.path.join(
            os.path.dirname(__file__), os.pardir,
            "opti_oignon", "config", "coding_history.yaml",
        )
        if os.path.isfile(config_path):
            cfg = _load_config(config_path)
            self.assertIn("enabled", cfg)
            self.assertIn("max_tasks", cfg)


class TestDatabaseInit(unittest.TestCase):
    """Store creates DB with correct tables and indexes."""

    def setUp(self):
        self.db_path = os.path.join(tempfile.mkdtemp(), "init.db")
        self.store = CodingHistoryStore(db_path=self.db_path)

    def test_db_file_created(self):
        self.assertTrue(os.path.isfile(self.db_path))

    def test_tasks_table_exists(self):
        conn = sqlite3.connect(self.db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        self.assertIn("tasks", tables)

    def test_steps_table_exists(self):
        conn = sqlite3.connect(self.db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        self.assertIn("steps", tables)

    def test_tests_table_exists(self):
        conn = sqlite3.connect(self.db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        self.assertIn("tests", tables)

    def test_checkpoints_table_exists(self):
        conn = sqlite3.connect(self.db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        self.assertIn("checkpoints", tables)

    def test_wal_mode(self):
        conn = sqlite3.connect(self.db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(mode, "wal")


class TestTaskLifecycle(unittest.TestCase):
    """Basic task create, update, complete flow."""

    def setUp(self):
        self.store = _tmp_store()

    def test_record_task_start(self):
        self.store.record_task_start("t1", "Fix bug", "/proj", "qwen3")
        tasks = self.store.list_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "t1")
        self.assertEqual(tasks[0].task_text, "Fix bug")
        self.assertEqual(tasks[0].status, "started")

    def test_update_status(self):
        self.store.record_task_start("t1", "Test", "", "")
        self.store.update_task_status("t1", "planning")
        tasks = self.store.list_tasks()
        self.assertEqual(tasks[0].status, "planning")

    def test_update_with_plan(self):
        self.store.record_task_start("t1", "Test", "", "")
        plan = {"task": "Test", "steps": [{"n": 1}]}
        self.store.update_task_status("t1", "planning", plan_json=plan)
        detail = self.store.get_task_detail("t1")
        self.assertIsNotNone(detail.plan_json)
        self.assertEqual(detail.plan_json["task"], "Test")

    def test_completed_sets_completed_at(self):
        self.store.record_task_start("t1", "Test", "", "")
        self.store.update_task_status("t1", "completed")
        detail = self.store.get_task_detail("t1")
        self.assertIsNotNone(detail.completed_at)

    def test_in_progress_no_completed_at(self):
        self.store.record_task_start("t1", "Test", "", "")
        self.store.update_task_status("t1", "implementing")
        detail = self.store.get_task_detail("t1")
        self.assertIsNone(detail.completed_at)

    def test_aborted_sets_completed_at(self):
        self.store.record_task_start("t1", "Test", "", "")
        self.store.update_task_status("t1", "aborted")
        detail = self.store.get_task_detail("t1")
        self.assertIsNotNone(detail.completed_at)

    def test_failed_sets_completed_at(self):
        self.store.record_task_start("t1", "Test", "", "")
        self.store.update_task_status("t1", "failed")
        detail = self.store.get_task_detail("t1")
        self.assertIsNotNone(detail.completed_at)


class TestStepRecording(unittest.TestCase):
    """Steps are recorded and retrieved correctly."""

    def setUp(self):
        self.store = _tmp_store()
        self.store.record_task_start("t1", "Task", "", "")

    def test_record_step(self):
        self.store.record_step("t1", 1, "create", "/file.py", "completed", "ok")
        detail = self.store.get_task_detail("t1")
        self.assertEqual(len(detail.steps), 1)
        self.assertEqual(detail.steps[0]["step_number"], 1)
        self.assertEqual(detail.steps[0]["step_type"], "create")

    def test_multiple_steps(self):
        for i in range(5):
            self.store.record_step("t1", i + 1, "edit", f"/f{i}.py")
        detail = self.store.get_task_detail("t1")
        self.assertEqual(len(detail.steps), 5)

    def test_step_order(self):
        self.store.record_step("t1", 2, "edit", "/b.py")
        self.store.record_step("t1", 1, "create", "/a.py")
        detail = self.store.get_task_detail("t1")
        self.assertEqual(detail.steps[0]["step_number"], 1)
        self.assertEqual(detail.steps[1]["step_number"], 2)

    def test_step_result_truncated(self):
        long_result = "x" * 20000
        self.store.record_step("t1", 1, "bash", "", "completed", long_result)
        detail = self.store.get_task_detail("t1")
        self.assertLessEqual(len(detail.steps[0]["result"]), 10001)

    def test_step_count_in_summary(self):
        self.store.record_step("t1", 1, "create", "/a.py", "completed")
        self.store.record_step("t1", 2, "edit", "/b.py", "failed")
        tasks = self.store.list_tasks()
        self.assertEqual(tasks[0].step_count, 2)
        self.assertEqual(tasks[0].completed_steps, 1)


class TestTestRecording(unittest.TestCase):
    """Test runs are recorded and retrieved correctly."""

    def setUp(self):
        self.store = _tmp_store()
        self.store.record_task_start("t1", "Task", "", "")

    def test_record_test_passed(self):
        self.store.record_test("t1", 1, True, "3 passed")
        detail = self.store.get_task_detail("t1")
        self.assertEqual(len(detail.tests), 1)
        self.assertTrue(detail.tests[0]["passed"])

    def test_record_test_failed(self):
        self.store.record_test("t1", 1, False, "1 failed")
        detail = self.store.get_task_detail("t1")
        self.assertFalse(detail.tests[0]["passed"])

    def test_multiple_test_runs(self):
        self.store.record_test("t1", 1, False, "fail")
        self.store.record_test("t1", 2, True, "pass")
        detail = self.store.get_task_detail("t1")
        self.assertEqual(len(detail.tests), 2)

    def test_last_passed_in_summary(self):
        self.store.record_test("t1", 1, False, "fail")
        self.store.record_test("t1", 2, True, "pass")
        tasks = self.store.list_tasks()
        self.assertTrue(tasks[0].last_passed)

    def test_last_failed_in_summary(self):
        self.store.record_test("t1", 1, True, "pass")
        self.store.record_test("t1", 2, False, "fail")
        tasks = self.store.list_tasks()
        self.assertFalse(tasks[0].last_passed)

    def test_no_tests_last_passed_none(self):
        tasks = self.store.list_tasks()
        self.assertIsNone(tasks[0].last_passed)

    def test_test_runs_count_in_summary(self):
        for i in range(3):
            self.store.record_test("t1", i + 1, True, "ok")
        tasks = self.store.list_tasks()
        self.assertEqual(tasks[0].test_runs, 3)


class TestCheckpointRecording(unittest.TestCase):
    """Checkpoints are recorded with state snapshots."""

    def setUp(self):
        self.store = _tmp_store()
        self.store.record_task_start("t1", "Task", "/proj", "model1")

    def test_record_checkpoint(self):
        self.store.record_checkpoint(
            "t1", "planning", "approve", 0, "hash123",
            plan_snapshot={"task": "Task", "steps": []},
        )
        detail = self.store.get_task_detail("t1")
        self.assertEqual(len(detail.checkpoints), 1)
        self.assertEqual(detail.checkpoints[0]["phase"], "planning")
        self.assertEqual(detail.checkpoints[0]["decision"], "approve")

    def test_multiple_checkpoints(self):
        self.store.record_checkpoint("t1", "planning", "approve", 0, "h1")
        self.store.record_checkpoint("t1", "implementing", "approve", 3, "h2")
        self.store.record_checkpoint("t1", "applying", "apply", 5, "h3")
        detail = self.store.get_task_detail("t1")
        self.assertEqual(len(detail.checkpoints), 3)

    def test_checkpoint_order(self):
        self.store.record_checkpoint("t1", "planning", "approve", 0, "")
        time.sleep(0.01)
        self.store.record_checkpoint("t1", "applying", "apply", 5, "")
        detail = self.store.get_task_detail("t1")
        self.assertEqual(detail.checkpoints[0]["phase"], "planning")
        self.assertEqual(detail.checkpoints[1]["phase"], "applying")


class TestGetLastCheckpoint(unittest.TestCase):
    """get_last_checkpoint returns most recent checkpoint state."""

    def setUp(self):
        self.store = _tmp_store()
        self.store.record_task_start("t1", "Fix bugs", "/proj", "qwen3")

    def test_no_checkpoints_returns_none(self):
        result = self.store.get_last_checkpoint("t1")
        self.assertIsNone(result)

    def test_returns_latest(self):
        self.store.record_checkpoint("t1", "planning", "approve", 0, "h1")
        time.sleep(0.01)
        self.store.record_checkpoint(
            "t1", "implementing", "approve", 3, "h2",
            plan_snapshot={"task": "Fix bugs", "steps": [1, 2, 3]},
        )
        cp = self.store.get_last_checkpoint("t1")
        self.assertIsNotNone(cp)
        self.assertEqual(cp.phase, "implementing")
        self.assertEqual(cp.current_step, 3)
        self.assertEqual(cp.originals_hash, "h2")
        self.assertEqual(cp.task_text, "Fix bugs")
        self.assertEqual(cp.model, "qwen3")

    def test_checkpoint_has_plan_json(self):
        plan = {"task": "Fix bugs", "steps": [{"n": 1}]}
        self.store.record_checkpoint("t1", "planning", "approve", 0, "", plan)
        cp = self.store.get_last_checkpoint("t1")
        self.assertIsNotNone(cp.plan_json)
        self.assertEqual(cp.plan_json["task"], "Fix bugs")

    def test_nonexistent_task_returns_none(self):
        result = self.store.get_last_checkpoint("nonexistent")
        self.assertIsNone(result)

    def test_returns_checkpoint_state_type(self):
        self.store.record_checkpoint("t1", "planning", "approve", 0, "")
        cp = self.store.get_last_checkpoint("t1")
        self.assertIsInstance(cp, CheckpointState)


class TestResumableTasks(unittest.TestCase):
    """get_resumable_tasks returns only in-progress tasks."""

    def setUp(self):
        self.store = _tmp_store()

    def test_started_is_resumable(self):
        self.store.record_task_start("t1", "Task1", "", "")
        resumable = self.store.get_resumable_tasks()
        self.assertEqual(len(resumable), 1)

    def test_completed_not_resumable(self):
        self.store.record_task_start("t1", "Task1", "", "")
        self.store.update_task_status("t1", "completed")
        resumable = self.store.get_resumable_tasks()
        self.assertEqual(len(resumable), 0)

    def test_aborted_not_resumable(self):
        self.store.record_task_start("t1", "Task1", "", "")
        self.store.update_task_status("t1", "aborted")
        resumable = self.store.get_resumable_tasks()
        self.assertEqual(len(resumable), 0)

    def test_failed_not_resumable(self):
        self.store.record_task_start("t1", "Task1", "", "")
        self.store.update_task_status("t1", "failed")
        resumable = self.store.get_resumable_tasks()
        self.assertEqual(len(resumable), 0)

    def test_mixed_tasks(self):
        self.store.record_task_start("t1", "Completed", "", "")
        self.store.update_task_status("t1", "completed")
        self.store.record_task_start("t2", "In progress", "", "")
        self.store.update_task_status("t2", "implementing")
        self.store.record_task_start("t3", "Aborted", "", "")
        self.store.update_task_status("t3", "aborted")
        resumable = self.store.get_resumable_tasks()
        self.assertEqual(len(resumable), 1)
        self.assertEqual(resumable[0].task_id, "t2")


class TestListTasks(unittest.TestCase):
    """list_tasks with pagination and status filter."""

    def setUp(self):
        self.store = _tmp_store()
        for i in range(10):
            self.store.record_task_start(f"t{i}", f"Task {i}", "", "")
        self.store.update_task_status("t0", "completed")
        self.store.update_task_status("t1", "completed")
        self.store.update_task_status("t2", "aborted")

    def test_list_all(self):
        tasks = self.store.list_tasks()
        self.assertEqual(len(tasks), 10)

    def test_limit(self):
        tasks = self.store.list_tasks(limit=3)
        self.assertEqual(len(tasks), 3)

    def test_offset(self):
        all_tasks = self.store.list_tasks()
        offset_tasks = self.store.list_tasks(limit=3, offset=3)
        self.assertEqual(len(offset_tasks), 3)
        self.assertNotEqual(all_tasks[0].task_id, offset_tasks[0].task_id)

    def test_filter_completed(self):
        tasks = self.store.list_tasks(status="completed")
        self.assertEqual(len(tasks), 2)
        for t in tasks:
            self.assertEqual(t.status, "completed")

    def test_filter_aborted(self):
        tasks = self.store.list_tasks(status="aborted")
        self.assertEqual(len(tasks), 1)

    def test_filter_started(self):
        tasks = self.store.list_tasks(status="started")
        self.assertEqual(len(tasks), 7)

    def test_newest_first(self):
        tasks = self.store.list_tasks()
        for i in range(len(tasks) - 1):
            self.assertGreaterEqual(
                tasks[i].created_at, tasks[i + 1].created_at
            )

    def test_returns_task_summary_type(self):
        tasks = self.store.list_tasks(limit=1)
        self.assertIsInstance(tasks[0], TaskSummary)


class TestGetTaskDetail(unittest.TestCase):
    """get_task_detail returns full task with steps, tests, checkpoints."""

    def setUp(self):
        self.store = _tmp_store()
        self.store.record_task_start("t1", "Complex task", "/proj", "model1")
        self.store.update_task_status("t1", "planning", {"steps": []})
        self.store.record_step("t1", 1, "create", "/a.py", "completed", "ok")
        self.store.record_step("t1", 2, "edit", "/b.py", "completed", "ok")
        self.store.record_test("t1", 1, True, "2 passed")
        self.store.record_checkpoint("t1", "planning", "approve", 0, "h1")

    def test_returns_task_detail(self):
        detail = self.store.get_task_detail("t1")
        self.assertIsInstance(detail, TaskDetail)

    def test_includes_steps(self):
        detail = self.store.get_task_detail("t1")
        self.assertEqual(len(detail.steps), 2)

    def test_includes_tests(self):
        detail = self.store.get_task_detail("t1")
        self.assertEqual(len(detail.tests), 1)

    def test_includes_checkpoints(self):
        detail = self.store.get_task_detail("t1")
        self.assertEqual(len(detail.checkpoints), 1)

    def test_includes_plan(self):
        detail = self.store.get_task_detail("t1")
        self.assertIsNotNone(detail.plan_json)

    def test_nonexistent_returns_none(self):
        detail = self.store.get_task_detail("nonexistent")
        self.assertIsNone(detail)

    def test_to_dict(self):
        detail = self.store.get_task_detail("t1")
        d = detail.to_dict()
        self.assertIn("task_id", d)
        self.assertIn("steps", d)
        self.assertIn("tests", d)
        self.assertIn("checkpoints", d)


class TestCountTasks(unittest.TestCase):
    """count_tasks returns correct counts."""

    def setUp(self):
        self.store = _tmp_store()
        for i in range(5):
            self.store.record_task_start(f"t{i}", f"Task {i}", "", "")
        self.store.update_task_status("t0", "completed")
        self.store.update_task_status("t1", "completed")

    def test_count_all(self):
        self.assertEqual(self.store.count_tasks(), 5)

    def test_count_completed(self):
        self.assertEqual(self.store.count_tasks(status="completed"), 2)

    def test_count_started(self):
        self.assertEqual(self.store.count_tasks(status="started"), 3)

    def test_count_nonexistent_status(self):
        self.assertEqual(self.store.count_tasks(status="xxxx"), 0)


class TestDeleteTask(unittest.TestCase):
    """delete_task cascades to steps, tests, checkpoints."""

    def setUp(self):
        self.store = _tmp_store()
        self.store.record_task_start("t1", "Task", "", "")
        self.store.record_step("t1", 1, "create", "/a.py")
        self.store.record_test("t1", 1, True, "ok")
        self.store.record_checkpoint("t1", "planning", "approve", 0, "")

    def test_delete_returns_true(self):
        self.assertTrue(self.store.delete_task("t1"))

    def test_delete_removes_task(self):
        self.store.delete_task("t1")
        self.assertEqual(self.store.count_tasks(), 0)

    def test_delete_removes_steps(self):
        self.store.delete_task("t1")
        detail = self.store.get_task_detail("t1")
        self.assertIsNone(detail)

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(self.store.delete_task("nonexistent"))

    def test_delete_only_target(self):
        self.store.record_task_start("t2", "Other task", "", "")
        self.store.delete_task("t1")
        self.assertEqual(self.store.count_tasks(), 1)
        tasks = self.store.list_tasks()
        self.assertEqual(tasks[0].task_id, "t2")


class TestPrune(unittest.TestCase):
    """Prune removes old tasks and enforces max_tasks."""

    def test_prune_old_tasks(self):
        db_path = os.path.join(tempfile.mkdtemp(), "prune.db")
        store = CodingHistoryStore(db_path=db_path)

        # Insert task with old timestamp manually
        conn = sqlite3.connect(db_path)
        old_ts = time.time() - (31 * 86400)  # 31 days ago
        conn.execute(
            "INSERT INTO tasks (task_id, task_text, project_path, model, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("old1", "Old task", "", "", "completed", old_ts),
        )
        conn.commit()
        conn.close()

        pruned = store.prune()
        self.assertEqual(pruned, 1)
        self.assertEqual(store.count_tasks(), 0)

    def test_prune_keeps_recent(self):
        store = _tmp_store()
        store.record_task_start("t1", "Recent task", "", "")
        pruned = store.prune()
        self.assertEqual(pruned, 0)
        self.assertEqual(store.count_tasks(), 1)

    def test_prune_enforces_max_tasks(self):
        db_path = os.path.join(tempfile.mkdtemp(), "maxprune.db")
        # Store with max_tasks=3
        cfg_path = os.path.join(tempfile.mkdtemp(), "cfg.yaml")
        with open(cfg_path, "w") as f:
            f.write("enabled: true\nmax_tasks: 3\nretention_days: 365\n")
        store = CodingHistoryStore(db_path=db_path, config_path=cfg_path)

        for i in range(6):
            store.record_task_start(f"t{i}", f"Task {i}", "", "")
            time.sleep(0.01)  # Ensure different timestamps

        pruned = store.prune()
        self.assertEqual(pruned, 3)
        self.assertEqual(store.count_tasks(), 3)


class TestStats(unittest.TestCase):
    """get_stats returns correct aggregated statistics."""

    def setUp(self):
        self.store = _tmp_store()
        self.store.record_task_start("t1", "Task 1", "", "")
        self.store.update_task_status("t1", "completed")
        self.store.record_task_start("t2", "Task 2", "", "")
        self.store.record_step("t1", 1, "create", "/a.py")
        self.store.record_step("t2", 1, "edit", "/b.py")
        self.store.record_test("t1", 1, True, "ok")
        self.store.record_test("t2", 1, False, "fail")
        self.store.record_checkpoint("t1", "applying", "apply", 1, "")

    def test_total_tasks(self):
        stats = self.store.get_stats()
        self.assertEqual(stats["total_tasks"], 2)

    def test_by_status(self):
        stats = self.store.get_stats()
        self.assertEqual(stats["by_status"]["completed"], 1)
        self.assertEqual(stats["by_status"]["started"], 1)

    def test_total_steps(self):
        stats = self.store.get_stats()
        self.assertEqual(stats["total_steps"], 2)

    def test_total_tests(self):
        stats = self.store.get_stats()
        self.assertEqual(stats["total_tests"], 2)

    def test_passed_tests(self):
        stats = self.store.get_stats()
        self.assertEqual(stats["passed_tests"], 1)

    def test_total_checkpoints(self):
        stats = self.store.get_stats()
        self.assertEqual(stats["total_checkpoints"], 1)


class TestTaskSummaryToDict(unittest.TestCase):
    """TaskSummary.to_dict serializes all fields."""

    def test_to_dict_fields(self):
        s = TaskSummary(
            task_id="t1", task_text="Test", project_path="/p",
            model="m", status="completed", step_count=3,
            completed_steps=2, test_runs=1, last_passed=True,
            created_at=1000.0, completed_at=2000.0,
        )
        d = s.to_dict()
        self.assertEqual(d["task_id"], "t1")
        self.assertEqual(d["step_count"], 3)
        self.assertEqual(d["completed_at"], 2000.0)
        self.assertTrue(d["last_passed"])


class TestTaskDetailToDict(unittest.TestCase):
    """TaskDetail.to_dict serializes all fields including sublists."""

    def test_to_dict_includes_lists(self):
        d = TaskDetail(
            task_id="t1", task_text="T", project_path="",
            model="", status="completed", plan_json=None,
            created_at=0.0, completed_at=1.0,
            steps=[{"n": 1}], tests=[{"p": True}],
            checkpoints=[{"phase": "apply"}],
        )
        out = d.to_dict()
        self.assertEqual(len(out["steps"]), 1)
        self.assertEqual(len(out["tests"]), 1)
        self.assertEqual(len(out["checkpoints"]), 1)


class TestDisabledStore(unittest.TestCase):
    """When enabled=False, all write operations are no-ops."""

    def setUp(self):
        cfg_path = os.path.join(tempfile.mkdtemp(), "disabled.yaml")
        with open(cfg_path, "w") as f:
            f.write("enabled: false\n")
        self.db_path = os.path.join(tempfile.mkdtemp(), "disabled.db")
        self.store = CodingHistoryStore(
            db_path=self.db_path, config_path=cfg_path
        )

    def test_record_task_noop(self):
        self.store.record_task_start("t1", "Task", "", "")
        self.assertEqual(self.store.count_tasks(), 0)

    def test_record_step_noop(self):
        self.store.record_step("t1", 1, "create", "/a.py")
        # No error even though task doesn't exist

    def test_record_test_noop(self):
        self.store.record_test("t1", 1, True, "ok")

    def test_record_checkpoint_noop(self):
        self.store.record_checkpoint("t1", "planning", "approve", 0, "")

    def test_update_status_noop(self):
        self.store.update_task_status("t1", "completed")


class TestThreadSafety(unittest.TestCase):
    """Concurrent writes do not corrupt the database."""

    def test_concurrent_inserts(self):
        store = _tmp_store()
        errors = []

        def worker(tid):
            try:
                store.record_task_start(tid, f"Task {tid}", "", "")
                store.record_step(tid, 1, "create", "/f.py")
                store.record_test(tid, 1, True, "ok")
                store.record_checkpoint(tid, "planning", "approve", 0, "")
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=worker, args=(f"t{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertEqual(store.count_tasks(), 10)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def setUp(self):
        self.store = _tmp_store()

    def test_empty_task_text(self):
        self.store.record_task_start("t1", "", "", "")
        tasks = self.store.list_tasks()
        self.assertEqual(tasks[0].task_text, "")

    def test_very_long_task_text_truncated(self):
        long_text = "a" * 5000
        self.store.record_task_start("t1", long_text, "", "")
        tasks = self.store.list_tasks()
        self.assertLessEqual(len(tasks[0].task_text), 2001)

    def test_replace_task(self):
        # INSERT OR REPLACE on same task_id
        self.store.record_task_start("t1", "First", "", "")
        self.store.record_task_start("t1", "Second", "", "")
        self.assertEqual(self.store.count_tasks(), 1)
        tasks = self.store.list_tasks()
        self.assertEqual(tasks[0].task_text, "Second")

    def test_plan_json_invalid_on_read(self):
        # Manually insert invalid JSON
        db_path = os.path.join(tempfile.mkdtemp(), "bad.db")
        store = CodingHistoryStore(db_path=db_path)
        store.record_task_start("t1", "Test", "", "")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE tasks SET plan_json = 'not-json' WHERE task_id = 't1'"
        )
        conn.commit()
        conn.close()
        detail = store.get_task_detail("t1")
        self.assertIsNone(detail.plan_json)

    def test_checkpoint_without_plan_snapshot(self):
        self.store.record_task_start("t1", "Test", "", "")
        self.store.record_checkpoint("t1", "planning", "approve", 0, "")
        cp = self.store.get_last_checkpoint("t1")
        self.assertIsNone(cp.plan_json)

    def test_stats_empty_db(self):
        stats = self.store.get_stats()
        self.assertEqual(stats["total_tasks"], 0)
        self.assertEqual(stats["total_steps"], 0)


class TestRoutesSchemas(unittest.TestCase):
    """Verify route and schema files have the expected S76 additions."""

    _ROUTES_PATH = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "opti_oignon", "api", "routes_coding.py",
    )
    _SCHEMAS_PATH = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "opti_oignon", "api", "schemas.py",
    )
    _DEPS_PATH = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "opti_oignon", "api", "deps.py",
    )

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_routes_has_history_endpoint(self):
        src = self._read(self._ROUTES_PATH)
        self.assertIn("/history", src)

    def test_routes_has_resume_endpoint(self):
        src = self._read(self._ROUTES_PATH)
        self.assertIn("/resume/", src)

    def test_routes_has_prune_endpoint(self):
        src = self._read(self._ROUTES_PATH)
        self.assertIn("/history/prune", src)

    def test_routes_has_stats_endpoint(self):
        src = self._read(self._ROUTES_PATH)
        self.assertIn("/history/stats", src)

    def test_schemas_has_task_summary(self):
        src = self._read(self._SCHEMAS_PATH)
        self.assertIn("CodingTaskSummaryResponse", src)

    def test_schemas_has_task_detail(self):
        src = self._read(self._SCHEMAS_PATH)
        self.assertIn("CodingTaskDetailResponse", src)

    def test_schemas_has_history_list(self):
        src = self._read(self._SCHEMAS_PATH)
        self.assertIn("CodingHistoryListResponse", src)

    def test_schemas_has_stats_response(self):
        src = self._read(self._SCHEMAS_PATH)
        self.assertIn("CodingHistoryStatsResponse", src)

    def test_schemas_has_resume_request(self):
        src = self._read(self._SCHEMAS_PATH)
        self.assertIn("CodingResumeRequest", src)

    def test_deps_has_coding_history(self):
        src = self._read(self._DEPS_PATH)
        self.assertIn("CODING_HISTORY_AVAILABLE", src)


if __name__ == "__main__":
    unittest.main()
