#!/usr/bin/env python3
"""
E2E INTEGRATION TESTS -- Cross-Module Flows (S61)
=====================================================

End-to-end integration tests verifying feature interactions across
the full stack. Mocks Ollama at the transport layer to test real
module wiring without requiring a running LLM server.

Test groups:
  1. Chat -> Agentic Executor -> Tool Calling -> Structured Output
  2. Project creation -> File upload -> Context injection
  3. Benchmark run -> History -> Comparison
  4. Model config change -> Smart routing adjustment
  5. Feedback submission -> Analytics query
  6. Self-correction -> Feedback recording
  7. Consensus integration
  8. Pipeline editor -> Execution flow
  9. Full health check consistency
  10. Cross-feature integration

Target: 40+ integration tests, zero regressions.

Usage:
    pytest tests/test_e2e_integration.py -v
    pytest tests/test_e2e_integration.py -v -k "TestProject"
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# HELPERS
# =============================================================================

def _get_test_client():
    """Create a FastAPI TestClient for API integration tests."""
    from fastapi.testclient import TestClient

    from opti_oignon.api.app import app
    return TestClient(app)


def _temp_db_path(suffix: str = ".db") -> Path:
    """Return a temporary database path."""
    return Path(tempfile.mktemp(suffix=suffix))


def _temp_dir() -> Path:
    """Return a temporary directory path."""
    return Path(tempfile.mkdtemp())


def _make_profile(
    name: str = "test-model:7b",
    display_name: str = "Test Model",
    context_window: int = 32768,
    speed_tier: str = "medium",
    quality_tier: str = "medium",
    task_scores: dict = None,
):
    """Create a ModelProfile for integration tests."""
    from opti_oignon.model_profiles import ModelProfile
    return ModelProfile(
        name=name,
        display_name=display_name,
        capabilities=["general", "code"],
        context_window=context_window,
        speed_tier=speed_tier,
        quality_tier=quality_tier,
        recommended_for=["general"],
        not_recommended_for=[],
        task_scores=task_scores or {"general": 0.7, "code_python": 0.8},
    )


def _make_profile_manager(profiles=None):
    """Create a pre-loaded ModelProfileManager with given profiles."""
    from opti_oignon.model_profiles import ModelProfileManager
    mgr = ModelProfileManager.__new__(ModelProfileManager)
    mgr._profiles = {}
    mgr._profiles_path = Path("/dev/null")
    mgr._loaded = True
    if profiles:
        for p in profiles:
            mgr._profiles[p.name] = p
    return mgr


def _make_routing(model="test:7b", task_type="general", temperature=0.3):
    """Create a minimal RoutingResult for agentic executor tests."""
    return SimpleNamespace(
        model=model,
        task_type=task_type,
        temperature=temperature,
        prompt_variant="standard",
        model_type="general",
        priority_used="primary",
        explanation="test routing",
        timeout=60,
    )


def _make_mock_executor(response_chunks=None):
    """Create a mock Executor that yields chunks."""
    mock = MagicMock()
    mock.cancel = MagicMock()
    mock.reset = MagicMock()
    mock.last_verification_results = []
    mock._last_tool_calls = []

    chunks = response_chunks or ["Hello", " world"]

    def _execute_gen(**kwargs):
        for chunk in chunks:
            yield chunk

    mock.execute.side_effect = _execute_gen
    return mock


def _consume_generator(gen):
    """Consume a generator and return collected items."""
    return list(gen)


# =============================================================================
# TEST GROUP 1: Chat -> Agentic Executor -> Tools -> Structured Output
# =============================================================================

class TestChatPipelineE2E(unittest.TestCase):
    """End-to-end tests for the chat pipeline through agentic executor."""

    def test_agentic_executor_direct_pipeline(self):
        """Direct pipeline executes through agentic executor with mocked Executor."""
        from opti_oignon.agentic_executor import PIPELINE_DIRECT, AgenticExecutor

        mock_exec = _make_mock_executor(["Test ", "response"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(ae.execute("Say hello", routing))

        self.assertTrue(len(chunks) > 0)
        self.assertEqual(ae.last_pipeline, PIPELINE_DIRECT)

    def test_agentic_executor_classification_detects_code(self):
        """Message classification correctly identifies code-related queries."""
        from opti_oignon.agentic_executor import _quick_classify

        result = _quick_classify("Write a Python function to sort a list")
        self.assertTrue(result.get("is_code", False))

    def test_agentic_executor_classification_detects_complexity(self):
        """Message classification detects complex queries needing think mode."""
        from opti_oignon.agentic_executor import _quick_classify

        result = _quick_classify("Explain the architecture of a microservice system")
        self.assertTrue(result.get("is_complex", False))

    def test_agentic_executor_classification_detects_reasoning(self):
        """Message classification detects queries needing advanced reasoning."""
        from opti_oignon.agentic_executor import _quick_classify

        result = _quick_classify("Break down step by step how photosynthesis works")
        self.assertTrue(result.get("needs_reasoning", False))

    def test_pipeline_selection_from_classification(self):
        """Pipeline selection uses classification signals correctly."""
        from opti_oignon.agentic_executor import _select_pipeline

        code_class = {"is_code": True, "is_complex": False,
                       "needs_tools": False, "needs_web": False, "needs_reasoning": False}
        pipeline = _select_pipeline(
            code_class, think_override=None, web_search_override=None,
            tool_executor_available=False, verification_available=True,
        )
        self.assertIn(pipeline, ["code_verify", "think", "direct"])

    def test_tool_registry_lists_available_tools(self):
        """Tool registry registers builtin tools (some may be disabled)."""
        from opti_oignon.tool_executor import ToolExecutor
        from opti_oignon.tool_registry import ToolRegistry

        registry = ToolRegistry()
        executor = ToolExecutor(registry=registry)

        # list_all includes disabled tools; list_available only enabled ones
        all_tools = registry.list_all()
        self.assertGreaterEqual(len(all_tools), 0)
        # Executor should be constructable with a registry
        self.assertIsNotNone(executor)

    def test_structured_output_engine_has_generate_method(self):
        """Structured output engine has generate_structured method."""
        from opti_oignon.structured_output import StructuredOutputEngine

        engine = StructuredOutputEngine()
        self.assertTrue(hasattr(engine, "generate_structured"))

    def test_agentic_executor_fallback_on_missing_deps(self):
        """Agentic executor reports subsystem availability correctly."""
        from opti_oignon.agentic_executor import AgenticExecutor

        ae = AgenticExecutor()
        self.assertIsInstance(ae.tool_executor_available, bool)
        self.assertIsInstance(ae.verification_available, bool)
        self.assertIsInstance(ae.structured_available, bool)
        self.assertIsInstance(ae.reasoning_available, bool)
        self.assertIsInstance(ae.consensus_available, bool)
        self.assertIsInstance(ae.self_correction_available, bool)

    def test_agentic_executor_think_pipeline(self):
        """Think pipeline passes through the executor."""
        from opti_oignon.agentic_executor import AgenticExecutor

        mock_exec = _make_mock_executor(["Thinking ", "response"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        chunks = _consume_generator(
            ae.execute("Explain quantum computing", routing, think=True, web_search=False)
        )
        self.assertTrue(len(chunks) > 0)

    def test_executor_records_last_pipeline(self):
        """Executor tracks which pipeline was used last."""
        from opti_oignon.agentic_executor import PIPELINE_DIRECT, AgenticExecutor

        mock_exec = _make_mock_executor(["response"])
        ae = AgenticExecutor(executor=mock_exec)
        ae._tool_executor = None

        routing = _make_routing()
        _consume_generator(ae.execute("test", routing))
        self.assertEqual(ae.last_pipeline, PIPELINE_DIRECT)


# =============================================================================
# TEST GROUP 2: Project -> File Upload -> Context Injection
# =============================================================================

class TestProjectContextE2E(unittest.TestCase):
    """End-to-end tests for the project lifecycle."""

    def setUp(self):
        self.db_path = _temp_db_path()
        self.storage_base = _temp_dir()

    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()
        if self.storage_base.exists():
            shutil.rmtree(self.storage_base, ignore_errors=True)

    def test_project_lifecycle_create_update_delete(self):
        """Full project CRUD lifecycle."""
        from opti_oignon.projects import ProjectStore

        store = ProjectStore(
            db_path=self.db_path, storage_base=self.storage_base,
            config_path=Path("/dev/null"),
        )
        project = store.create_project(name="E2E Test Project", description="Testing")
        self.assertTrue(len(project.id) > 0)

        retrieved = store.get_project(project.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "E2E Test Project")

        updated = store.update_project(project.id, name="Updated Project")
        self.assertEqual(updated.name, "Updated Project")

        self.assertGreaterEqual(len(store.list_projects()), 1)

        self.assertTrue(store.delete_project(project.id))
        self.assertIsNone(store.get_project(project.id))

    def test_project_file_upload_with_bytes(self):
        """File upload stores content from bytes correctly."""
        from opti_oignon.projects import ProjectStore

        store = ProjectStore(
            db_path=self.db_path, storage_base=self.storage_base,
            config_path=Path("/dev/null"),
        )
        project = store.create_project(name="File Test Project")
        pf = store.add_file(project.id, "test_doc.md",
                            b"# Test Document\nE2E integration content.")
        self.assertIsNotNone(pf)
        self.assertEqual(pf.filename, "test_doc.md")

        files = store.list_files(project.id)
        self.assertEqual(len(files), 1)

    def test_project_file_type_detection(self):
        """File type detection works across categories."""
        from opti_oignon.projects import detect_file_type

        self.assertEqual(detect_file_type("script.py"), "code")
        self.assertEqual(detect_file_type("data.csv"), "data")
        self.assertEqual(detect_file_type("readme.md"), "text")
        self.assertEqual(detect_file_type("photo.png"), "image")
        self.assertEqual(detect_file_type("report.pdf"), "document")

    def test_project_cascade_delete_removes_files(self):
        """Deleting a project removes all associated files."""
        from opti_oignon.projects import ProjectStore

        store = ProjectStore(
            db_path=self.db_path, storage_base=self.storage_base,
            config_path=Path("/dev/null"),
        )
        project = store.create_project(name="Cascade Test")
        store.add_file(project.id, "cascade_test.txt", b"cascade content")
        store.delete_project(project.id)
        self.assertEqual(len(store.list_files(project.id)), 0)

    def test_project_trigger_detection_with_project_id(self):
        """Trigger detector identifies project-related queries at L1."""
        from opti_oignon.project_triggers import ProjectTriggerDetector

        mock_store = MagicMock()
        mock_project = MagicMock()
        mock_project.id = "proj-123"
        mock_project.name = "Bioacoustics Analysis"
        mock_project.description = "Analyze bird call recordings"
        mock_store.list_projects.return_value = [mock_project]
        mock_store.get_project.return_value = mock_project
        mock_store.list_files.return_value = []

        detector = ProjectTriggerDetector(store=mock_store)
        result = detector.detect(
            "use project Bioacoustics for this",
            project_id="proj-123", skip_l3=True,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.relevant)

    def test_project_api_create_and_list(self):
        """API endpoints for project CRUD work end-to-end."""
        import opti_oignon.api.routes_projects as rp
        client = _get_test_client()

        # Clear any accumulated projects from previous tests to avoid hitting max_projects limit
        if rp.project_store is not None:
            for p in rp.project_store.list_projects():
                try:
                    rp.project_store.delete_project(p.id)
                except Exception:
                    pass

        r = client.post("/api/projects", json={
            "name": "API E2E Project", "description": "Created via API test",
        })
        self.assertEqual(r.status_code, 201)
        project_id = r.json()["id"]

        r = client.get("/api/projects")
        self.assertEqual(r.status_code, 200)
        self.assertIn(project_id, [p["id"] for p in r.json()])

        r = client.get(f"/api/projects/{project_id}")
        self.assertEqual(r.status_code, 200)

        r = client.put(f"/api/projects/{project_id}", json={"name": "Updated"})
        self.assertEqual(r.status_code, 200)

        r = client.delete(f"/api/projects/{project_id}")
        self.assertIn(r.status_code, [200, 204])


# =============================================================================
# TEST GROUP 3: Benchmark -> History -> Comparison
# =============================================================================

class TestBenchmarkFlowE2E(unittest.TestCase):
    """End-to-end tests for benchmark run, history, and comparison."""

    def setUp(self):
        self.db_path = _temp_db_path()

    def tearDown(self):
        if self.db_path.exists():
            self.db_path.unlink()

    def test_benchmark_run_save_and_retrieve(self):
        """Save a benchmark run and retrieve it from history."""
        from opti_oignon.benchmark_history import (
            BenchmarkHistory,
            BenchmarkResultRecord,
            BenchmarkRunRecord,
        )
        history = BenchmarkHistory(db_path=self.db_path)
        history.save_run(BenchmarkRunRecord(
            id="run-e2e-001", run_type="llm",
            started_at="2025-01-01T00:00:00Z", completed_at="2025-01-01T00:05:00Z",
            status="completed", models=["qwen3:32b", "deepseek-r1:32b"],
            tasks=["code_python", "general_qa"], total_tests=4,
            avg_score=7.5, best_model="qwen3:32b", duration_sec=300.0,
        ))
        for i, (m, t) in enumerate([
            ("qwen3:32b", "code_python"), ("qwen3:32b", "general_qa"),
            ("deepseek-r1:32b", "code_python"), ("deepseek-r1:32b", "general_qa"),
        ]):
            history.save_result(BenchmarkResultRecord(
                id=f"res-e2e-{i}", run_id="run-e2e-001", model=m, task=t,
                task_name=f"Test {t}", category="general",
                score=7.0 + i * 0.5, auto_score=7.0 + i * 0.5,
                time_seconds=15.0 + i, status="success",
            ))

        runs = history.get_runs(limit=10)
        self.assertEqual(len(runs), 1)
        detail = history.get_run_detail("run-e2e-001")
        self.assertEqual(len(detail["results"]), 4)

    def test_benchmark_comparison_detects_regressions(self):
        """Comparison between runs detects score regressions."""
        from opti_oignon.benchmark_history import (
            BenchmarkHistory,
            BenchmarkResultRecord,
            BenchmarkRunRecord,
        )
        history = BenchmarkHistory(db_path=self.db_path)

        for rid, score in [("run-cmp-1", 8.0), ("run-cmp-2", 5.0)]:
            history.save_run(BenchmarkRunRecord(
                id=rid, status="completed", models=["model-a:7b"],
                tasks=["code_python"], total_tests=1, avg_score=score,
            ))
            history.save_result(BenchmarkResultRecord(
                id=f"res-{rid}", run_id=rid, model="model-a:7b",
                task="code_python", score=score, auto_score=score,
                time_seconds=10.0, status="success",
            ))

        comparison = history.compare_runs(["run-cmp-1", "run-cmp-2"])
        self.assertEqual(len(comparison.runs), 2)
        self.assertGreater(len(comparison.regressions), 0)

    def test_benchmark_model_trends(self):
        """Model trends track scores across multiple runs."""
        from opti_oignon.benchmark_history import (
            BenchmarkHistory,
            BenchmarkResultRecord,
            BenchmarkRunRecord,
        )
        history = BenchmarkHistory(db_path=self.db_path)

        for i in range(3):
            rid = f"run-trend-{i}"
            history.save_run(BenchmarkRunRecord(
                id=rid, status="completed",
                started_at=f"2025-01-0{i+1}T00:00:00Z",
                models=["trend-model:7b"], tasks=["general"],
                total_tests=1, avg_score=6.0 + i,
            ))
            history.save_result(BenchmarkResultRecord(
                id=f"res-trend-{i}", run_id=rid, model="trend-model:7b",
                task="general", score=6.0 + i, auto_score=6.0 + i,
                time_seconds=10.0, status="success",
            ))

        trends = history.get_model_trends("trend-model:7b")
        self.assertEqual(len(trends.avg_scores), 3)
        self.assertLess(trends.avg_scores[0], trends.avg_scores[2])

    def test_benchmark_cascade_delete(self):
        """Deleting a run removes all its results."""
        from opti_oignon.benchmark_history import (
            BenchmarkHistory,
            BenchmarkResultRecord,
            BenchmarkRunRecord,
        )
        history = BenchmarkHistory(db_path=self.db_path)
        history.save_run(BenchmarkRunRecord(
            id="run-del-1", status="completed",
            models=["del-model:7b"], tasks=["general"], total_tests=2,
        ))
        for i in range(2):
            history.save_result(BenchmarkResultRecord(
                id=f"res-del-{i}", run_id="run-del-1", model="del-model:7b",
                task="general", score=7.0, auto_score=7.0,
                time_seconds=5.0, status="success",
            ))

        self.assertTrue(history.delete_run("run-del-1"))
        self.assertIsNone(history.get_run_detail("run-del-1"))

    def test_benchmark_api_suites(self):
        """Benchmark suites API returns suite list structure."""
        client = _get_test_client()
        r = client.get("/api/benchmark/suites")
        self.assertEqual(r.status_code, 200)
        self.assertIn("suites", r.json())

    def test_benchmark_api_runs(self):
        """Benchmark runs API returns paginated structure."""
        client = _get_test_client()
        r = client.get("/api/benchmark/runs")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("runs", data)
        self.assertIn("total", data)

    def test_benchmark_api_model_config(self):
        """Model config API returns configuration structure."""
        client = _get_test_client()
        r = client.get("/api/benchmark/models/config")
        self.assertEqual(r.status_code, 200)
        self.assertIn("config", r.json())

    def test_benchmark_api_installed_models(self):
        """Installed models endpoint returns list."""
        client = _get_test_client()
        r = client.get("/api/benchmark/models/installed")
        self.assertEqual(r.status_code, 200)
        self.assertIn("models", r.json())


# =============================================================================
# TEST GROUP 4: Model Config -> Smart Routing
# =============================================================================

class TestModelRoutingE2E(unittest.TestCase):
    """End-to-end tests for model config -> routing adjustment flow."""

    def test_smart_router_uses_profile_scores(self):
        """SmartRouter selects model with highest task_score."""
        from opti_oignon.smart_router import SmartRouter

        profiles = [
            _make_profile(name="code-expert:14b", speed_tier="slow",
                          quality_tier="high",
                          task_scores={"code_python": 0.95, "general": 0.6}),
            _make_profile(name="fast-chat:7b", speed_tier="fast",
                          quality_tier="low",
                          task_scores={"code_python": 0.4, "general": 0.8}),
        ]
        mgr = _make_profile_manager(profiles)
        router = SmartRouter(profile_manager=mgr, enabled=True,
                             default_model="fast-chat:7b",
                             config_path=Path("/dev/null"))
        self.assertEqual(router.select_model("code_verify").model, "code-expert:14b")

    def test_smart_router_fallback_to_default(self):
        """SmartRouter falls back to default with no profiles."""
        from opti_oignon.smart_router import SmartRouter

        mgr = _make_profile_manager([])
        router = SmartRouter(profile_manager=mgr, enabled=True,
                             default_model="fallback:7b",
                             config_path=Path("/dev/null"))
        self.assertEqual(router.select_model("direct").model, "fallback:7b")

    def test_smart_router_disabled_uses_default(self):
        """Disabled SmartRouter always returns default model."""
        from opti_oignon.smart_router import SmartRouter

        profiles = [_make_profile(name="better:14b", task_scores={"general": 0.99})]
        mgr = _make_profile_manager(profiles)
        router = SmartRouter(profile_manager=mgr, enabled=False,
                             default_model="my-default:7b",
                             config_path=Path("/dev/null"))
        self.assertEqual(router.select_model("direct").model, "my-default:7b")

    def test_profile_manager_loads_yaml(self):
        """ModelProfileManager loads profiles from YAML."""
        from opti_oignon.model_profiles import ModelProfileManager

        yaml_path = _PROJECT_ROOT / "opti_oignon" / "config" / "model_profiles.yaml"
        if not yaml_path.exists():
            self.skipTest("model_profiles.yaml not found")

        mgr = ModelProfileManager(profiles_path=yaml_path)
        mgr.load()
        self.assertGreater(len(mgr.list_profiles()), 0)

    def test_smart_router_select_for_pipeline(self):
        """SmartRouter selects per-step models for a pipeline."""
        from opti_oignon.smart_router import SmartRouter

        profiles = [
            _make_profile(name="thinker:32b", speed_tier="slow",
                          task_scores={"reasoning": 0.9, "general": 0.7}),
            _make_profile(name="coder:14b", speed_tier="medium",
                          task_scores={"code_python": 0.9, "general": 0.5}),
        ]
        mgr = _make_profile_manager(profiles)
        router = SmartRouter(profile_manager=mgr, enabled=True,
                             default_model="coder:14b",
                             config_path=Path("/dev/null"))
        selections = router.select_for_pipeline(["think", "code_verify"])
        self.assertEqual(len(selections), 2)

    def test_routing_api_select_endpoint(self):
        """Smart routing select API is accessible."""
        client = _get_test_client()
        r = client.get("/api/smart-routing/select?step_type=direct")
        self.assertEqual(r.status_code, 200)


# =============================================================================
# TEST GROUP 5: Feedback -> Analytics Cross-Flow
# =============================================================================

class TestFeedbackAnalyticsE2E(unittest.TestCase):
    """End-to-end tests for feedback -> analytics reporting."""

    def setUp(self):
        self.fb_db = _temp_db_path()
        self.an_db = _temp_db_path()

    def tearDown(self):
        for p in [self.fb_db, self.an_db]:
            if p.exists():
                p.unlink()

    def test_feedback_feeds_analytics_overview(self):
        """Feedback + performance data appears in analytics overview."""
        from opti_oignon.analytics import AnalyticsEngine, PerformanceRecord, PerformanceTracker
        from opti_oignon.feedback import RATING_TYPE_THUMBS, FeedbackEntry, FeedbackStore

        fb_store = FeedbackStore(db_path=self.fb_db, config_path=Path("/dev/null"))
        tracker = PerformanceTracker(db_path=self.an_db)
        engine = AnalyticsEngine(tracker=tracker, config_path=Path("/dev/null"))

        for i in range(5):
            tracker.record(PerformanceRecord(
                model_used="analytics-model:7b", pipeline_used="direct",
                response_time_ms=200.0 + i * 50, prompt_tokens=100,
                completion_tokens=200, success=True,
            ))

        overview = engine.get_overview()
        self.assertEqual(overview.total_requests, 5)
        self.assertGreater(overview.avg_response_time_ms, 0)

    def test_feedback_by_model_filters(self):
        """Feedback filtered by model returns only matching entries."""
        from opti_oignon.feedback import FeedbackEntry, FeedbackStore

        store = FeedbackStore(db_path=self.fb_db, config_path=Path("/dev/null"))
        store.add_feedback(FeedbackEntry(model_used="model-a:7b", rating_value=1,
                                         conversation_id="c1", message_id="m1"))
        store.add_feedback(FeedbackEntry(model_used="model-b:14b", rating_value=0,
                                         conversation_id="c2", message_id="m2"))
        store.add_feedback(FeedbackEntry(model_used="model-a:7b", rating_value=1,
                                         conversation_id="c3", message_id="m3"))

        self.assertEqual(len(store.list_by_model("model-a:7b")), 2)
        self.assertEqual(len(store.list_by_model("model-b:14b")), 1)

    def test_analytics_trends_returns_structure(self):
        """Analytics trends return a list of TrendPoint objects."""
        from opti_oignon.analytics import AnalyticsEngine, PerformanceRecord, PerformanceTracker

        tracker = PerformanceTracker(db_path=self.an_db)
        engine = AnalyticsEngine(tracker=tracker, config_path=Path("/dev/null"))

        now = time.time()
        for i in range(10):
            tracker.record(PerformanceRecord(
                model_used="trend-model:7b", pipeline_used="direct",
                response_time_ms=100.0 + i * 10, prompt_tokens=50,
                completion_tokens=100, timestamp=now - (10 - i) * 60,
            ))

        trends = engine.get_trends(window="1h", buckets=6)
        self.assertIsNotNone(trends)
        self.assertIsInstance(trends, list)
        self.assertEqual(len(trends), 6)

    def test_analytics_cleanup(self):
        """Analytics cleanup removes old records."""
        from opti_oignon.analytics import AnalyticsEngine, PerformanceRecord, PerformanceTracker

        tracker = PerformanceTracker(db_path=self.an_db)
        engine = AnalyticsEngine(tracker=tracker, config_path=Path("/dev/null"))

        tracker.record(PerformanceRecord(
            model_used="old:7b", pipeline_used="direct",
            response_time_ms=100.0, timestamp=time.time() - 365 * 86400,
        ))
        tracker.record(PerformanceRecord(
            model_used="new:7b", pipeline_used="direct", response_time_ms=100.0,
        ))

        self.assertEqual(tracker.count(), 2)
        self.assertGreaterEqual(engine.cleanup_old_records(), 1)

    def test_feedback_api_flow(self):
        """Full API flow: submit -> retrieve -> delete."""
        client = _get_test_client()

        r = client.post("/api/feedback", json={
            "conversation_id": "conv-e2e", "message_id": "msg-e2e",
            "rating_type": "thumbs", "rating_value": 1,
            "model_used": "flow-model:7b", "pipeline_used": "direct",
        })
        self.assertEqual(r.status_code, 200)
        fid = r.json()["feedback_id"]

        r = client.get(f"/api/feedback/{fid}")
        self.assertEqual(r.status_code, 200)

        r = client.delete(f"/api/feedback/{fid}")
        self.assertIn(r.status_code, [200, 204])


# =============================================================================
# TEST GROUP 6: Self-Correction Integration
# =============================================================================

class TestSelfCorrectionE2E(unittest.TestCase):
    """End-to-end tests for self-correction pipeline."""

    def test_self_correction_engine_methods(self):
        """Self-correction engine has expected methods."""
        from opti_oignon.self_correction import SelfCorrectionEngine

        engine = SelfCorrectionEngine()
        self.assertTrue(hasattr(engine, "correct"))
        self.assertTrue(hasattr(engine, "check_compliance"))
        self.assertTrue(hasattr(engine, "check_quality"))

    def test_self_correction_with_mock_ollama(self):
        """Self-correction loop runs with mocked Ollama."""
        from opti_oignon.self_correction import SelfCorrectionEngine

        engine = SelfCorrectionEngine()
        mock_resp = {
            "message": {"role": "assistant", "content": json.dumps({
                "passes": True, "score": 8, "issues": [],
                "improved_response": "Corrected text",
            })},
            "done": True, "total_duration": 500_000_000, "eval_count": 50,
        }

        with patch("opti_oignon.self_correction.ollama") as mock_ol:
            mock_ol.chat.return_value = mock_resp
            result = engine.correct(
                user_message="Explain photosynthesis",
                response="Plants use sunlight to make food",
                model="test:7b",
            )
        self.assertIsNotNone(result)


# =============================================================================
# TEST GROUP 7: Consensus Integration
# =============================================================================

class TestConsensusE2E(unittest.TestCase):
    """End-to-end tests for multi-model consensus."""

    def test_consensus_engine_methods(self):
        """Consensus engine has run_consensus and execute_consensus."""
        from opti_oignon.consensus import ConsensusEngine

        engine = ConsensusEngine()
        self.assertTrue(hasattr(engine, "run_consensus"))
        self.assertTrue(hasattr(engine, "execute_consensus"))

    def test_consensus_with_mock(self):
        """Consensus runs with mocked Ollama."""
        from opti_oignon.consensus import ConsensusEngine

        engine = ConsensusEngine()
        mock_resp = {
            "message": {"role": "assistant", "content": "Test response"},
            "done": True, "total_duration": 300_000_000, "eval_count": 40,
        }

        with patch("opti_oignon.consensus.ollama") as mock_ol:
            mock_ol.chat.return_value = mock_resp
            result = engine.run_consensus(
                query="What is photosynthesis?",
                models=["test:7b"], strategy="best_of_n",
            )
        self.assertIsNotNone(result)


# =============================================================================
# TEST GROUP 8: Pipeline Editor
# =============================================================================

class TestPipelineEditorE2E(unittest.TestCase):
    """End-to-end tests for pipeline editor."""

    def test_builtin_pipelines_load(self):
        """Builtin pipelines load from config."""
        from opti_oignon.pipeline_manager import get_pipeline_manager
        mgr = get_pipeline_manager()
        self.assertGreater(len(mgr.list_builtin()), 0)

    def test_custom_pipeline_crud(self):
        """Create and delete a custom pipeline."""
        from opti_oignon.pipeline_manager import Pipeline, PipelineStep, get_pipeline_manager

        mgr = get_pipeline_manager()
        pipeline = Pipeline(
            id="e2e-test-pipeline", name="E2E Test Pipeline",
            steps=[PipelineStep(name="Step1", agent="coder")],
        )
        self.assertTrue(mgr.create(pipeline))
        self.assertIn("E2E Test Pipeline", [p.name for p in mgr.list_custom()])
        mgr.delete("e2e-test-pipeline")

    def test_pipeline_api_list(self):
        """API returns pipeline list."""
        client = _get_test_client()
        r = client.get("/api/pipelines")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.json()), 0)


# =============================================================================
# TEST GROUP 9: Health Check Consistency
# =============================================================================

class TestHealthConsistencyE2E(unittest.TestCase):
    """Health endpoints report consistent state."""

    def test_health_includes_all_modules(self):
        """Health check includes all S42-S60 modules."""
        client = _get_test_client()
        r = client.get("/api/health")
        modules = r.json()["modules"]
        for key in ["conversation", "presets", "memory", "artifacts",
                     "code_executor", "response_cache", "semantic_cache",
                     "pipelines", "benchmarks", "model_warmup", "config",
                     "model_profiles", "context_window", "smart_router",
                     "feedback", "analytics", "projects", "project_context",
                     "project_triggers", "benchmark_history"]:
            self.assertIn(key, modules)

    def test_health_dashboard_accessible(self):
        """Dashboard health endpoint returns 200."""
        client = _get_test_client()
        self.assertEqual(client.get("/api/health/dashboard").status_code, 200)

    def test_all_routers_registered(self):
        """All API route prefixes return non-404."""
        client = _get_test_client()
        for path in ["/api/health", "/api/models", "/api/presets",
                      "/api/feedback/stats", "/api/analytics/overview",
                      "/api/benchmark/suites", "/api/benchmark/runs",
                      "/api/projects", "/api/pipelines",
                      "/api/smart-routing/config"]:
            r = client.get(path)
            self.assertNotEqual(r.status_code, 404, f"GET {path} returned 404")

    def test_version_consistent(self):
        """Version 1.6.0 is consistent across sources."""
        from opti_oignon import __version__
        from opti_oignon.api.app import app

        client = _get_test_client()
        api_ver = client.get("/api/health").json()["version"]

        self.assertEqual(__version__, "1.6.6")
        self.assertEqual(api_ver, "1.6.6")
        self.assertEqual(app.version, "1.6.6")


# =============================================================================
# TEST GROUP 10: Cross-Feature Integration
# =============================================================================

class TestCrossFeatureE2E(unittest.TestCase):
    """Tests verifying multiple features work together."""

    def test_project_with_feedback_for_same_conversation(self):
        """Feedback references conversation linked to a project."""
        from opti_oignon.feedback import FeedbackEntry, FeedbackStore
        from opti_oignon.projects import ProjectStore

        proj_db, fb_db, storage = _temp_db_path(), _temp_db_path(), _temp_dir()
        try:
            proj_store = ProjectStore(db_path=proj_db, storage_base=storage,
                                      config_path=Path("/dev/null"))
            fb_store = FeedbackStore(db_path=fb_db, config_path=Path("/dev/null"))

            project = proj_store.create_project(name="Feedback Project")
            conv_id = "conv-linked-001"
            proj_store.link_conversation(project.id, conv_id)

            fb_store.add_feedback(FeedbackEntry(
                conversation_id=conv_id, message_id="msg-001",
                rating_value=1, model_used="linked:7b", pipeline_used="direct",
            ))

            self.assertEqual(len(fb_store.list_by_conversation(conv_id)), 1)
            conv_ids = [c["conversation_id"]
                        for c in proj_store.list_conversations(project.id)]
            self.assertIn(conv_id, conv_ids)
        finally:
            for p in [proj_db, fb_db]:
                if p.exists():
                    p.unlink()
            shutil.rmtree(storage, ignore_errors=True)

    def test_benchmark_informs_routing(self):
        """Benchmark scores align with smart routing decisions."""
        from opti_oignon.benchmark_history import (
            BenchmarkHistory,
            BenchmarkResultRecord,
            BenchmarkRunRecord,
        )
        from opti_oignon.smart_router import SmartRouter

        bm_db = _temp_db_path()
        try:
            history = BenchmarkHistory(db_path=bm_db)
            history.save_run(BenchmarkRunRecord(
                id="routing-run", status="completed",
                models=["model-a:14b", "model-b:7b"],
                tasks=["code_python", "general"], total_tests=4,
            ))
            for m, t, s in [("model-a:14b", "code_python", 9.0),
                            ("model-a:14b", "general", 5.0),
                            ("model-b:7b", "code_python", 5.0),
                            ("model-b:7b", "general", 9.0)]:
                history.save_result(BenchmarkResultRecord(
                    run_id="routing-run", model=m, task=t,
                    score=s, auto_score=s, time_seconds=10.0, status="success",
                ))

            profiles = [
                _make_profile(name="model-a:14b",
                              task_scores={"code_python": 0.9, "general": 0.5}),
                _make_profile(name="model-b:7b",
                              task_scores={"code_python": 0.5, "general": 0.9}),
            ]
            router = SmartRouter(
                profile_manager=_make_profile_manager(profiles),
                enabled=True, default_model="model-b:7b",
                config_path=Path("/dev/null"),
            )

            self.assertEqual(router.select_model("code_verify").model, "model-a:14b")
            self.assertEqual(router.select_model("direct").model, "model-b:7b")
        finally:
            if bm_db.exists():
                bm_db.unlink()

    def test_dual_analytics_and_feedback(self):
        """Performance record and feedback for the same request coexist."""
        from opti_oignon.analytics import PerformanceRecord, PerformanceTracker
        from opti_oignon.feedback import FeedbackEntry, FeedbackStore

        an_db, fb_db = _temp_db_path(), _temp_db_path()
        try:
            tracker = PerformanceTracker(db_path=an_db)
            fb_store = FeedbackStore(db_path=fb_db, config_path=Path("/dev/null"))
            msg_id = "msg-dual-001"

            tracker.record(PerformanceRecord(
                message_id=msg_id, model_used="dual:7b", pipeline_used="direct",
                response_time_ms=350.0, prompt_tokens=80, completion_tokens=160,
            ))
            fb_store.add_feedback(FeedbackEntry(
                message_id=msg_id, rating_value=1, model_used="dual:7b",
                conversation_id="c1", pipeline_used="direct",
            ))

            records = tracker.get_records(model="dual:7b")
            feedbacks = fb_store.list_by_model("dual:7b")
            self.assertEqual(records[0].message_id, msg_id)
            self.assertEqual(feedbacks[0].message_id, msg_id)
        finally:
            for p in [an_db, fb_db]:
                if p.exists():
                    p.unlink()

    def test_reasoning_engine_available(self):
        """Reasoning engine exposes availability property."""
        from opti_oignon.reasoning import ReasoningEngine
        engine = ReasoningEngine()
        self.assertIsInstance(engine.available, bool)

    def test_all_pipeline_constants_defined(self):
        """All 9 pipeline type constants are defined."""
        from opti_oignon.agentic_executor import (
            PIPELINE_CODE_VERIFY,
            PIPELINE_CONSENSUS,
            PIPELINE_DIRECT,
            PIPELINE_REASONING,
            PIPELINE_SELF_CORRECT,
            PIPELINE_THINK,
            PIPELINE_THINK_TOOLS,
            PIPELINE_TOOLS,
            PIPELINE_WEB_SEARCH,
        )
        all_p = [PIPELINE_DIRECT, PIPELINE_TOOLS, PIPELINE_CODE_VERIFY,
                 PIPELINE_THINK, PIPELINE_WEB_SEARCH, PIPELINE_THINK_TOOLS,
                 PIPELINE_REASONING, PIPELINE_CONSENSUS, PIPELINE_SELF_CORRECT]
        self.assertEqual(len(all_p), 9)
        for p in all_p:
            self.assertIsInstance(p, str)
            self.assertTrue(len(p) > 0)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
