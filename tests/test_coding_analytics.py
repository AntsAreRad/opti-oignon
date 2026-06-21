#!/usr/bin/env python3
"""
Tests for Coding History Analytics (S78 SQ-08).

Covers:
- Success rate by model (SQL aggregation)
- Average steps by model and overall
- Failure reasons distribution
- Time-to-completion trends
- Test pass rate per task
- Steps distribution histogram
- Full analytics payload
- Analytics API endpoint schema
"""

# ---------------------------------------------------------------------------
# Direct module loading to avoid import chain issues in test env
# ---------------------------------------------------------------------------
import importlib.util
import os
import sqlite3
import tempfile
import time
import unittest

_base = os.path.join(
    os.path.dirname(__file__), os.pardir, "opti_oignon"
)
_base = os.path.normpath(_base)


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_base, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_history_mod = _load_module("coding_history", "coding_history.py")
CodingHistoryStore = _history_mod.CodingHistoryStore


def _make_store(db_path: str) -> CodingHistoryStore:
    """Create a CodingHistoryStore pointing at a temp DB."""
    return CodingHistoryStore(db_path=db_path)


def _seed_tasks(store: CodingHistoryStore, tasks: list[dict]) -> None:
    """Seed the store with tasks, steps, tests, and checkpoints."""
    for t in tasks:
        store.record_task_start(
            task_id=t["task_id"],
            task_text=t.get("task_text", "test task"),
            project_path=t.get("project_path", "/tmp/p"),
            model=t.get("model", "qwen3:32b"),
        )
        if "status" in t:
            store.update_task_status(t["task_id"], t["status"])
        if "steps" in t:
            for i, s in enumerate(t["steps"], start=1):
                store.record_step(
                    task_id=t["task_id"],
                    step_number=i,
                    step_type=s.get("type", "bash"),
                    status=s.get("status", "completed"),
                    result=s.get("result", "ok"),
                )
        if "tests" in t:
            for i, tr in enumerate(t["tests"], start=1):
                store.record_test(
                    task_id=t["task_id"],
                    run_number=i,
                    passed=tr.get("passed", True),
                    output=tr.get("output", ""),
                )
        if "checkpoints" in t:
            for cp in t["checkpoints"]:
                store.record_checkpoint(
                    task_id=t["task_id"],
                    phase=cp.get("phase", "implementing"),
                    decision=cp.get("decision", "approve"),
                    current_step=cp.get("current_step", 0),
                )


# ============================================================================
# Tests: Success Rate by Model
# ============================================================================


class TestSuccessRateByModel(unittest.TestCase):
    """get_success_rate_by_model SQL aggregation."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_empty_db_returns_empty_list(self):
        result = self.store.get_success_rate_by_model()
        self.assertEqual(result, [])

    def test_single_model_all_completed(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "qwen3:32b", "status": "completed"},
            {"task_id": "t2", "model": "qwen3:32b", "status": "completed"},
        ])
        result = self.store.get_success_rate_by_model()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model"], "qwen3:32b")
        self.assertEqual(result[0]["total"], 2)
        self.assertEqual(result[0]["completed"], 2)
        self.assertAlmostEqual(result[0]["success_rate"], 100.0, places=1)

    def test_mixed_statuses(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "modelA", "status": "completed"},
            {"task_id": "t2", "model": "modelA", "status": "failed"},
            {"task_id": "t3", "model": "modelA", "status": "aborted"},
        ])
        result = self.store.get_success_rate_by_model()
        self.assertEqual(result[0]["total"], 3)
        self.assertEqual(result[0]["completed"], 1)
        self.assertAlmostEqual(result[0]["success_rate"], 33.3, places=1)

    def test_multiple_models_sorted_by_rate(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "good", "status": "completed"},
            {"task_id": "t2", "model": "good", "status": "completed"},
            {"task_id": "t3", "model": "bad", "status": "failed"},
            {"task_id": "t4", "model": "bad", "status": "failed"},
        ])
        result = self.store.get_success_rate_by_model()
        self.assertEqual(len(result), 2)
        # Sorted DESC by success_rate
        self.assertEqual(result[0]["model"], "good")
        self.assertEqual(result[1]["model"], "bad")

    def test_empty_model_excluded(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "", "status": "completed"},
            {"task_id": "t2", "model": "qwen3:32b", "status": "completed"},
        ])
        result = self.store.get_success_rate_by_model()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model"], "qwen3:32b")


# ============================================================================
# Tests: Average Steps by Model
# ============================================================================


class TestAvgStepsByModel(unittest.TestCase):
    """get_avg_steps_by_model SQL aggregation."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_empty_db(self):
        result = self.store.get_avg_steps_by_model()
        self.assertEqual(result, [])

    def test_single_task_single_model(self):
        _seed_tasks(self.store, [
            {
                "task_id": "t1", "model": "qwen3:32b",
                "steps": [{"type": "bash"}, {"type": "bash"}, {"type": "bash"}],
            },
        ])
        result = self.store.get_avg_steps_by_model()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["avg_steps"], 3.0, places=1)
        self.assertEqual(result[0]["min_steps"], 3)
        self.assertEqual(result[0]["max_steps"], 3)

    def test_multiple_tasks_averages(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1", "steps": [{}] * 2},
            {"task_id": "t2", "model": "m1", "steps": [{}] * 6},
        ])
        result = self.store.get_avg_steps_by_model()
        self.assertAlmostEqual(result[0]["avg_steps"], 4.0, places=1)
        self.assertEqual(result[0]["min_steps"], 2)
        self.assertEqual(result[0]["max_steps"], 6)

    def test_no_steps_not_included(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1"},
        ])
        result = self.store.get_avg_steps_by_model()
        self.assertEqual(result, [])


# ============================================================================
# Tests: Average Steps Overall
# ============================================================================


class TestAvgStepsOverall(unittest.TestCase):
    """get_avg_steps_overall SQL aggregation."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_empty_db(self):
        result = self.store.get_avg_steps_overall()
        self.assertEqual(result["task_count"], 0)

    def test_overall_average(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1", "steps": [{}] * 4},
            {"task_id": "t2", "model": "m2", "steps": [{}] * 8},
        ])
        result = self.store.get_avg_steps_overall()
        self.assertAlmostEqual(result["avg_steps"], 6.0, places=1)
        self.assertEqual(result["min_steps"], 4)
        self.assertEqual(result["max_steps"], 8)
        self.assertEqual(result["task_count"], 2)


# ============================================================================
# Tests: Failure Reasons
# ============================================================================


class TestFailureReasons(unittest.TestCase):
    """get_failure_reasons SQL aggregation."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_empty_db(self):
        result = self.store.get_failure_reasons()
        self.assertEqual(result, [])

    def test_no_failed_tasks(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1", "status": "completed"},
        ])
        result = self.store.get_failure_reasons()
        self.assertEqual(result, [])

    def test_failed_with_checkpoint(self):
        _seed_tasks(self.store, [
            {
                "task_id": "t1", "model": "m1", "status": "failed",
                "checkpoints": [{"phase": "testing"}],
            },
        ])
        result = self.store.get_failure_reasons()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["failure_phase"], "testing")
        self.assertEqual(result[0]["count"], 1)

    def test_failed_without_checkpoint_shows_unknown(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1", "status": "failed"},
        ])
        result = self.store.get_failure_reasons()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["failure_phase"], "unknown")

    def test_multiple_failure_phases(self):
        _seed_tasks(self.store, [
            {
                "task_id": "t1", "model": "m1", "status": "failed",
                "checkpoints": [{"phase": "testing"}],
            },
            {
                "task_id": "t2", "model": "m1", "status": "failed",
                "checkpoints": [{"phase": "implementing"}],
            },
            {
                "task_id": "t3", "model": "m1", "status": "failed",
                "checkpoints": [{"phase": "testing"}],
            },
        ])
        result = self.store.get_failure_reasons()
        # Sorted by count DESC
        self.assertEqual(result[0]["failure_phase"], "testing")
        self.assertEqual(result[0]["count"], 2)
        self.assertEqual(result[1]["failure_phase"], "implementing")
        self.assertEqual(result[1]["count"], 1)


# ============================================================================
# Tests: Time Trends
# ============================================================================


class TestTimeTrends(unittest.TestCase):
    """get_time_trends SQL query."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_empty_db(self):
        result = self.store.get_time_trends()
        self.assertEqual(result, [])

    def test_completed_task_has_duration(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1", "status": "started"},
        ])
        # Ensure measurable time gap between created_at and completed_at
        time.sleep(0.05)
        self.store.update_task_status("t1", "completed")
        result = self.store.get_time_trends()
        self.assertEqual(len(result), 1)
        self.assertIn("duration_seconds", result[0])
        self.assertGreater(result[0]["duration_seconds"], 0)

    def test_incomplete_task_excluded(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1", "status": "started"},
        ])
        result = self.store.get_time_trends()
        self.assertEqual(result, [])

    def test_limit_parameter(self):
        for i in range(5):
            _seed_tasks(self.store, [
                {"task_id": f"t{i}", "model": "m1", "status": "completed"},
            ])
        result = self.store.get_time_trends(limit=3)
        self.assertEqual(len(result), 3)


# ============================================================================
# Tests: Test Pass Rate Per Task
# ============================================================================


class TestPassRatePerTask(unittest.TestCase):
    """get_test_pass_rate_per_task SQL aggregation."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_empty_db(self):
        result = self.store.get_test_pass_rate_per_task()
        self.assertEqual(result, [])

    def test_all_tests_pass(self):
        _seed_tasks(self.store, [
            {
                "task_id": "t1", "model": "m1",
                "tests": [{"passed": True}, {"passed": True}],
            },
        ])
        result = self.store.get_test_pass_rate_per_task()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["total_runs"], 2)
        self.assertEqual(result[0]["passed_runs"], 2)
        self.assertAlmostEqual(result[0]["pass_rate"], 100.0, places=1)

    def test_mixed_pass_fail(self):
        _seed_tasks(self.store, [
            {
                "task_id": "t1", "model": "m1",
                "tests": [
                    {"passed": True},
                    {"passed": False},
                    {"passed": False},
                    {"passed": True},
                ],
            },
        ])
        result = self.store.get_test_pass_rate_per_task()
        self.assertEqual(result[0]["total_runs"], 4)
        self.assertEqual(result[0]["passed_runs"], 2)
        self.assertAlmostEqual(result[0]["pass_rate"], 50.0, places=1)

    def test_task_without_tests_excluded(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1"},
        ])
        result = self.store.get_test_pass_rate_per_task()
        self.assertEqual(result, [])


# ============================================================================
# Tests: Steps Distribution
# ============================================================================


class TestStepsDistribution(unittest.TestCase):
    """get_steps_distribution SQL aggregation."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_empty_db(self):
        result = self.store.get_steps_distribution()
        self.assertEqual(result, [])

    def test_distribution_counts(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1", "steps": [{}] * 3},
            {"task_id": "t2", "model": "m1", "steps": [{}] * 3},
            {"task_id": "t3", "model": "m1", "steps": [{}] * 5},
        ])
        result = self.store.get_steps_distribution()
        # Should have 2 buckets: 3 steps (2 tasks), 5 steps (1 task)
        self.assertEqual(len(result), 2)
        dist = {r["step_count"]: r["task_count"] for r in result}
        self.assertEqual(dist[3], 2)
        self.assertEqual(dist[5], 1)


# ============================================================================
# Tests: Full Analytics Payload
# ============================================================================


class TestGetAnalytics(unittest.TestCase):
    """get_analytics aggregation method."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = _make_store(self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_empty_db_returns_zeros(self):
        result = self.store.get_analytics()
        self.assertEqual(result["total_tasks"], 0)
        self.assertEqual(result["completed_tasks"], 0)
        self.assertAlmostEqual(result["overall_success_rate"], 0.0)
        self.assertEqual(result["success_rate_by_model"], [])
        self.assertEqual(result["failure_reasons"], [])

    def test_full_payload_keys(self):
        _seed_tasks(self.store, [
            {
                "task_id": "t1", "model": "m1", "status": "completed",
                "steps": [{}] * 3,
                "tests": [{"passed": True}],
            },
        ])
        result = self.store.get_analytics()
        expected_keys = {
            "total_tasks", "completed_tasks", "overall_success_rate",
            "success_rate_by_model", "avg_steps_by_model",
            "avg_steps_overall", "failure_reasons", "time_trends",
            "test_pass_rate_per_task", "steps_distribution",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_overall_success_rate_calculation(self):
        _seed_tasks(self.store, [
            {"task_id": "t1", "model": "m1", "status": "completed"},
            {"task_id": "t2", "model": "m1", "status": "completed"},
            {"task_id": "t3", "model": "m1", "status": "failed"},
            {"task_id": "t4", "model": "m1", "status": "aborted"},
        ])
        result = self.store.get_analytics()
        self.assertEqual(result["total_tasks"], 4)
        self.assertEqual(result["completed_tasks"], 2)
        self.assertAlmostEqual(result["overall_success_rate"], 50.0, places=1)

    def test_analytics_includes_all_subqueries(self):
        _seed_tasks(self.store, [
            {
                "task_id": "t1", "model": "m1", "status": "completed",
                "steps": [{}] * 5,
                "tests": [{"passed": True}, {"passed": False}],
                "checkpoints": [{"phase": "testing"}],
            },
            {
                "task_id": "t2", "model": "m2", "status": "failed",
                "steps": [{}] * 3,
                "tests": [{"passed": False}],
                "checkpoints": [{"phase": "implementing"}],
            },
        ])
        result = self.store.get_analytics()
        self.assertEqual(len(result["success_rate_by_model"]), 2)
        self.assertEqual(len(result["avg_steps_by_model"]), 2)
        self.assertGreater(result["avg_steps_overall"]["task_count"], 0)
        self.assertEqual(len(result["failure_reasons"]), 1)
        self.assertGreater(len(result["time_trends"]), 0)
        self.assertEqual(len(result["test_pass_rate_per_task"]), 2)
        self.assertEqual(len(result["steps_distribution"]), 2)


# ============================================================================
# Tests: API Schema Validation
# ============================================================================


class TestAnalyticsSchema(unittest.TestCase):
    """Verify Pydantic schemas exist and are structured correctly."""

    def test_schemas_module_has_analytics_response(self):
        schemas_path = os.path.join(_base, "api", "schemas.py")
        with open(schemas_path) as fh:
            content = fh.read()
        self.assertIn("class CodingAnalyticsResponse", content)
        self.assertIn("class CodingModelSuccessRate", content)
        self.assertIn("class CodingModelAvgSteps", content)
        self.assertIn("class CodingAvgStepsOverall", content)
        self.assertIn("class CodingFailureReason", content)
        self.assertIn("class CodingTimeTrend", content)
        self.assertIn("class CodingTestPassRate", content)
        self.assertIn("class CodingStepsDistribution", content)

    def test_routes_has_analytics_endpoint(self):
        routes_path = os.path.join(_base, "api", "routes_coding.py")
        with open(routes_path) as fh:
            content = fh.read()
        self.assertIn("/history/analytics", content)
        self.assertIn("CodingAnalyticsResponse", content)

    def test_analytics_endpoint_before_task_id_route(self):
        """analytics route must be registered before {task_id} to avoid conflict."""
        routes_path = os.path.join(_base, "api", "routes_coding.py")
        with open(routes_path) as fh:
            content = fh.read()
        analytics_pos = content.index("/history/analytics")
        task_id_pos = content.index("/history/{task_id}")
        self.assertLess(analytics_pos, task_id_pos)


# ============================================================================
# Tests: Version and Integration
# ============================================================================


class TestVersionAndIntegration(unittest.TestCase):
    """Version bump and module integration checks."""

    def test_app_version_is_180(self):
        app_path = os.path.join(_base, "api", "app.py")
        with open(app_path) as fh:
            content = fh.read()
        self.assertIn('"1.8.9"', content)

    def test_coding_history_has_analytics_methods(self):
        history_path = os.path.join(_base, "coding_history.py")
        with open(history_path) as fh:
            content = fh.read()
        methods = [
            "get_success_rate_by_model",
            "get_avg_steps_by_model",
            "get_avg_steps_overall",
            "get_failure_reasons",
            "get_time_trends",
            "get_test_pass_rate_per_task",
            "get_steps_distribution",
            "get_analytics",
        ]
        for m in methods:
            self.assertIn(f"def {m}(", content, f"Missing method: {m}")

    def test_types_ts_has_analytics_interfaces(self):
        types_path = os.path.join(
            os.path.dirname(__file__), os.pardir,
            "frontend", "src", "lib", "types.ts"
        )
        types_path = os.path.normpath(types_path)
        with open(types_path) as fh:
            content = fh.read()
        interfaces = [
            "CodingAnalyticsResponse",
            "CodingModelSuccessRate",
            "CodingExecuteAllStatus",
            "CodingStepsDistribution",
        ]
        for iface in interfaces:
            self.assertIn(iface, content, f"Missing TS interface: {iface}")


if __name__ == "__main__":
    unittest.main()
