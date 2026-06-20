#!/usr/bin/env python3
"""
Tests for coding history export and batch delete -- Opti-Oignon S79

Covers:
- export_tasks_json: full export with steps/tests/computed fields
- export_tasks_csv_rows: flat SQL-aggregated rows
- batch_delete_by_ids: bulk delete by list of IDs
- batch_delete_before_date: delete by timestamp cutoff
- Edge cases: empty DB, nonexistent IDs, cascade integrity
"""

import csv
import importlib.util
import io
import json
import os
import tempfile
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


def _tmp_store() -> CodingHistoryStore:
    """Create a store with a temp DB."""
    db = os.path.join(tempfile.mkdtemp(), "test_export.db")
    return CodingHistoryStore(db_path=db)


def _populate_store(store, count=3):
    """Insert count tasks with steps and tests."""
    task_ids = []
    for i in range(count):
        tid = f"export-task-{i}"
        task_ids.append(tid)
        store.record_task_start(tid, f"Task {i}", "/project", f"model-{i % 2}")
        for s in range(1, i + 2):
            store.record_step(tid, s, step_type="bash", status="completed", result=f"ok-{s}")
        for r in range(1, i + 2):
            store.record_test(tid, r, passed=(r % 2 == 1), output=f"test-{r}")
        if i % 2 == 0:
            store.update_task_status(tid, "completed")
        else:
            store.update_task_status(tid, "failed")
    return task_ids


# ===================================================================
# Export JSON tests
# ===================================================================

class TestExportJSON(unittest.TestCase):
    """Tests for export_tasks_json method."""

    def test_empty_db_returns_empty_list(self):
        store = _tmp_store()
        result = store.export_tasks_json()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_export_returns_all_tasks(self):
        store = _tmp_store()
        _populate_store(store, 3)
        result = store.export_tasks_json()
        self.assertEqual(len(result), 3)

    def test_export_task_fields(self):
        store = _tmp_store()
        _populate_store(store, 1)
        result = store.export_tasks_json()
        task = result[0]
        expected_keys = {
            "task_id", "task_text", "project_path", "model", "status",
            "step_count", "test_runs", "pass_rate", "created_at",
            "completed_at", "duration_seconds", "steps", "tests",
        }
        self.assertTrue(expected_keys.issubset(set(task.keys())))

    def test_export_includes_steps(self):
        store = _tmp_store()
        _populate_store(store, 2)
        result = store.export_tasks_json()
        # Task 0 has 1 step, task 1 has 2 steps
        steps_counts = sorted([t["step_count"] for t in result])
        self.assertIn(1, steps_counts)
        self.assertIn(2, steps_counts)

    def test_export_includes_tests(self):
        store = _tmp_store()
        _populate_store(store, 2)
        result = store.export_tasks_json()
        for task in result:
            self.assertIsInstance(task["tests"], list)
            self.assertGreater(len(task["tests"]), 0)

    def test_export_test_passed_is_bool(self):
        store = _tmp_store()
        _populate_store(store, 1)
        result = store.export_tasks_json()
        for test in result[0]["tests"]:
            self.assertIsInstance(test["passed"], bool)

    def test_export_pass_rate_computed(self):
        store = _tmp_store()
        store.record_task_start("pr-1", "rate test", "/p", "m")
        store.record_test("pr-1", 1, passed=True)
        store.record_test("pr-1", 2, passed=False)
        result = store.export_tasks_json()
        self.assertAlmostEqual(result[0]["pass_rate"], 50.0, places=1)

    def test_export_duration_computed(self):
        store = _tmp_store()
        store.record_task_start("dur-1", "dur test", "/p", "m")
        time.sleep(0.05)
        store.update_task_status("dur-1", "completed")
        result = store.export_tasks_json()
        dur = result[0]["duration_seconds"]
        self.assertIsNotNone(dur)
        self.assertGreater(dur, 0)

    def test_export_duration_none_when_not_completed(self):
        store = _tmp_store()
        store.record_task_start("dur-2", "not done", "/p", "m")
        result = store.export_tasks_json()
        self.assertIsNone(result[0]["duration_seconds"])

    def test_export_is_json_serializable(self):
        store = _tmp_store()
        _populate_store(store, 2)
        result = store.export_tasks_json()
        serialized = json.dumps(result, default=str)
        self.assertIsInstance(serialized, str)
        parsed = json.loads(serialized)
        self.assertEqual(len(parsed), 2)


# ===================================================================
# Export CSV tests
# ===================================================================

class TestExportCSV(unittest.TestCase):
    """Tests for export_tasks_csv_rows method."""

    def test_empty_db_returns_empty_list(self):
        store = _tmp_store()
        result = store.export_tasks_csv_rows()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_csv_returns_all_tasks(self):
        store = _tmp_store()
        _populate_store(store, 3)
        result = store.export_tasks_csv_rows()
        self.assertEqual(len(result), 3)

    def test_csv_row_is_flat(self):
        store = _tmp_store()
        _populate_store(store, 1)
        result = store.export_tasks_csv_rows()
        row = result[0]
        # No nested structures
        for key, value in row.items():
            self.assertNotIsInstance(value, (list, dict))

    def test_csv_row_fields(self):
        store = _tmp_store()
        _populate_store(store, 1)
        result = store.export_tasks_csv_rows()
        row = result[0]
        expected_keys = {
            "task_id", "task_text", "model", "status",
            "step_count", "test_runs", "pass_rate",
            "created_at", "completed_at", "duration_seconds",
        }
        self.assertTrue(expected_keys.issubset(set(row.keys())))

    def test_csv_step_count_from_sql(self):
        store = _tmp_store()
        store.record_task_start("csv-1", "test", "/p", "m")
        store.record_step("csv-1", 1, step_type="bash")
        store.record_step("csv-1", 2, step_type="create")
        store.record_step("csv-1", 3, step_type="bash")
        result = store.export_tasks_csv_rows()
        self.assertEqual(result[0]["step_count"], 3)

    def test_csv_pass_rate_from_sql(self):
        store = _tmp_store()
        store.record_task_start("csv-2", "test", "/p", "m")
        store.record_test("csv-2", 1, passed=True)
        store.record_test("csv-2", 2, passed=True)
        store.record_test("csv-2", 3, passed=False)
        result = store.export_tasks_csv_rows()
        self.assertAlmostEqual(result[0]["pass_rate"], 66.7, places=1)

    def test_csv_writable(self):
        store = _tmp_store()
        _populate_store(store, 2)
        rows = store.export_tasks_csv_rows()
        output = io.StringIO()
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        output.seek(0)
        content = output.read()
        self.assertIn("task_id", content)
        self.assertIn("export-task-", content)


# ===================================================================
# Batch Delete by IDs tests
# ===================================================================

class TestBatchDeleteByIds(unittest.TestCase):
    """Tests for batch_delete_by_ids method."""

    def test_empty_list_returns_zero(self):
        store = _tmp_store()
        _populate_store(store, 2)
        result = store.batch_delete_by_ids([])
        self.assertEqual(result, 0)

    def test_delete_single_task(self):
        store = _tmp_store()
        ids = _populate_store(store, 3)
        result = store.batch_delete_by_ids([ids[0]])
        self.assertEqual(result, 1)
        remaining = store.list_tasks()
        self.assertEqual(len(remaining), 2)

    def test_delete_multiple_tasks(self):
        store = _tmp_store()
        ids = _populate_store(store, 4)
        result = store.batch_delete_by_ids([ids[0], ids[2], ids[3]])
        self.assertEqual(result, 3)
        remaining = store.list_tasks()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].task_id, ids[1])

    def test_delete_nonexistent_ids(self):
        store = _tmp_store()
        _populate_store(store, 2)
        result = store.batch_delete_by_ids(["nonexistent-1", "nonexistent-2"])
        self.assertEqual(result, 0)

    def test_delete_mixed_existing_nonexistent(self):
        store = _tmp_store()
        ids = _populate_store(store, 2)
        result = store.batch_delete_by_ids([ids[0], "nonexistent"])
        self.assertEqual(result, 1)

    def test_cascade_deletes_steps(self):
        store = _tmp_store()
        ids = _populate_store(store, 1)
        # Verify steps exist before delete
        detail = store.get_task_detail(ids[0])
        self.assertGreater(len(detail.steps), 0)
        store.batch_delete_by_ids([ids[0]])
        # Task gone
        detail = store.get_task_detail(ids[0])
        self.assertIsNone(detail)

    def test_cascade_deletes_tests(self):
        store = _tmp_store()
        ids = _populate_store(store, 1)
        detail = store.get_task_detail(ids[0])
        self.assertGreater(len(detail.tests), 0)
        store.batch_delete_by_ids([ids[0]])
        self.assertIsNone(store.get_task_detail(ids[0]))

    def test_all_tasks_deleted(self):
        store = _tmp_store()
        ids = _populate_store(store, 3)
        result = store.batch_delete_by_ids(ids)
        self.assertEqual(result, 3)
        self.assertEqual(store.count_tasks(), 0)


# ===================================================================
# Batch Delete before date tests
# ===================================================================

class TestBatchDeleteBeforeDate(unittest.TestCase):
    """Tests for batch_delete_before_date method."""

    def test_no_tasks_before_date(self):
        store = _tmp_store()
        _populate_store(store, 2)
        # All tasks created just now, cutoff is in the past
        result = store.batch_delete_before_date(1000.0)
        self.assertEqual(result, 0)

    def test_all_tasks_before_date(self):
        store = _tmp_store()
        _populate_store(store, 3)
        # All tasks created just now, cutoff is in the future
        result = store.batch_delete_before_date(time.time() + 3600)
        self.assertEqual(result, 3)
        self.assertEqual(store.count_tasks(), 0)

    def test_partial_delete(self):
        store = _tmp_store()
        # Create old task
        store.record_task_start("old-1", "old task", "/p", "m")
        # Manually backdate it via raw SQL
        import sqlite3
        conn = sqlite3.connect(store._db_path)
        conn.execute(
            "UPDATE tasks SET created_at = ? WHERE task_id = ?",
            (1000.0, "old-1"),
        )
        conn.commit()
        conn.close()
        # Create recent task
        store.record_task_start("new-1", "new task", "/p", "m")
        # Delete tasks before epoch 2000
        result = store.batch_delete_before_date(2000.0)
        self.assertEqual(result, 1)
        remaining = store.list_tasks()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].task_id, "new-1")

    def test_cascade_on_date_delete(self):
        store = _tmp_store()
        store.record_task_start("cas-1", "cascade", "/p", "m")
        store.record_step("cas-1", 1, step_type="bash")
        store.record_test("cas-1", 1, passed=True)
        result = store.batch_delete_before_date(time.time() + 3600)
        self.assertEqual(result, 1)
        self.assertIsNone(store.get_task_detail("cas-1"))

    def test_empty_db(self):
        store = _tmp_store()
        result = store.batch_delete_before_date(time.time() + 3600)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
