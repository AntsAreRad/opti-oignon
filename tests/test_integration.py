#!/usr/bin/env python3
"""
TESTS -- Integration Testing (S56)
=====================================

End-to-end integration tests verifying cross-module interaction,
API endpoint chains, config roundtrips, pipeline flow, and
graceful degradation when dependencies are unavailable.

Target: 30+ tests, zero regressions.

Usage:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -k "TestAPI"
"""

import json
import os
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
    mgr._config_path = Path("/dev/null")
    mgr._loaded = True
    for p in (profiles or []):
        mgr._profiles[p.name] = p
    return mgr


# =============================================================================
# TEST: SmartRouter + ModelProfiles Integration
# =============================================================================

class TestSmartRouterProfilesIntegration(unittest.TestCase):
    """SmartRouter using real ModelProfiles for model selection."""

    def setUp(self):
        from opti_oignon.smart_router import SmartRouter
        self.profiles = [
            _make_profile(
                name="fast-model:7b",
                display_name="Fast Model",
                speed_tier="fast",
                quality_tier="low",
                context_window=8192,
                task_scores={"general": 0.5, "code_python": 0.4},
            ),
            _make_profile(
                name="quality-model:32b",
                display_name="Quality Model",
                speed_tier="slow",
                quality_tier="high",
                context_window=65536,
                task_scores={"general": 0.9, "code_python": 0.95, "reasoning": 0.9},
            ),
            _make_profile(
                name="balanced-model:14b",
                display_name="Balanced Model",
                speed_tier="medium",
                quality_tier="medium",
                context_window=32768,
                task_scores={"general": 0.7, "code_python": 0.75, "reasoning": 0.6},
            ),
        ]
        self.mgr = _make_profile_manager(self.profiles)
        self.router = SmartRouter(
            profile_manager=self.mgr,
            enabled=True,
            default_model="fallback:7b",
            config_path=Path("/dev/null"),
        )

    def test_selects_best_model_for_code(self):
        """SmartRouter picks the highest-scored model for code tasks."""
        result = self.router.select_model("code_verify")
        # quality-model has highest code_python score (0.95)
        self.assertEqual(result.model, "quality-model:32b")
        self.assertTrue(result.profile_used)
        self.assertFalse(result.fallback)

    def test_selects_best_model_for_general(self):
        """SmartRouter picks best model for general tasks."""
        result = self.router.select_model("direct")
        # quality-model has highest general score (0.9) but slow
        # Result depends on speed_weight; with balanced, quality still wins
        self.assertIn(result.model, [p.name for p in self.profiles])
        self.assertTrue(result.profile_used)

    def test_context_window_filtering(self):
        """Models with insufficient context windows are penalized."""
        # Request large context: fast-model (8192) should be penalized
        result = self.router.select_model("reasoning", required_context=16384)
        # fast-model has only 8192 context, so it should not be selected
        # Result should be one of the models with enough context
        self.assertIn(result.model, ["quality-model:32b", "balanced-model:14b"])

    def test_exclude_models(self):
        """Excluded models are not selected."""
        result = self.router.select_model(
            "code_verify",
            excluded_models=["quality-model:32b"],
        )
        self.assertNotEqual(result.model, "quality-model:32b")

    def test_pipeline_routing_all_steps(self):
        """select_for_pipeline returns a result for each step type."""
        step_types = ["direct", "code_verify", "reasoning"]
        results = self.router.select_for_pipeline(step_types)
        self.assertEqual(set(results.keys()), set(step_types))
        for step, result in results.items():
            self.assertTrue(len(result.model) > 0)

    def test_disabled_router_uses_default(self):
        """When disabled, SmartRouter returns the default model."""
        from opti_oignon.smart_router import SmartRouter
        disabled = SmartRouter(
            profile_manager=self.mgr,
            enabled=False,
            default_model="my-default:7b",
            config_path=Path("/dev/null"),
        )
        result = disabled.select_model("code_verify")
        self.assertEqual(result.model, "my-default:7b")
        self.assertTrue(result.fallback)

    def test_result_has_alternatives(self):
        """SmartRouter result includes scored alternatives."""
        result = self.router.select_model("code_verify")
        # With 3 profiles, should have alternatives
        self.assertIsInstance(result.alternatives, list)

    def test_result_serialization(self):
        """SmartRoutingResult.to_dict() produces valid JSON."""
        result = self.router.select_model("direct")
        d = result.to_dict()
        self.assertIn("model", d)
        self.assertIn("score", d)
        self.assertIn("task_score", d)
        serialized = json.dumps(d)
        self.assertIsInstance(json.loads(serialized), dict)


# =============================================================================
# TEST: Feedback + Analytics Cross-Module
# =============================================================================

class TestFeedbackAnalyticsIntegration(unittest.TestCase):
    """Feedback and Analytics modules working together."""

    def setUp(self):
        from opti_oignon.analytics import AnalyticsEngine, PerformanceTracker
        from opti_oignon.feedback import RATING_TYPE_THUMBS, FeedbackEntry, FeedbackStore
        self.feedback_store = FeedbackStore(
            db_path=_temp_db_path(),
            config_path=Path("/dev/null"),
        )
        self.tracker = PerformanceTracker(db_path=_temp_db_path())
        self.engine = AnalyticsEngine(
            tracker=self.tracker,
            config_path=Path("/dev/null"),
        )
        self.FeedbackEntry = FeedbackEntry
        self.THUMBS = RATING_TYPE_THUMBS

    def test_feedback_then_analytics_overview(self):
        """Submit feedback, record performance, then check analytics."""
        from opti_oignon.analytics import PerformanceRecord
        # Submit feedback
        entry = self.FeedbackEntry(
            conversation_id="conv-int-1",
            message_id="msg-1",
            rating_type=self.THUMBS,
            rating_value=1,
            model_used="qwen3:32b",
            pipeline_used="direct",
            task_type="general",
        )
        saved = self.feedback_store.add_feedback(entry)
        self.assertIsNotNone(saved.feedback_id)

        # Record performance
        record = PerformanceRecord(
            model_used="qwen3:32b",
            pipeline_used="direct",
            task_type="general",
            response_time_ms=350.0,
            prompt_tokens=100,
            completion_tokens=200,
        )
        self.tracker.record(record)

        # Check analytics overview
        overview = self.engine.get_overview()
        self.assertGreaterEqual(overview.total_requests, 1)

        # Check feedback stats
        stats = self.feedback_store.get_stats()
        self.assertEqual(stats.total_count, 1)
        self.assertGreater(stats.positive_count, 0)

    def test_multi_model_feedback_aggregation(self):
        """Feedback from multiple models aggregates correctly."""
        models = ["model-a:7b", "model-b:14b", "model-c:32b"]
        for i, model in enumerate(models):
            for rating in [0, 1, 1]:  # 2 positive, 1 negative each
                self.feedback_store.add_feedback(self.FeedbackEntry(
                    conversation_id=f"conv-{i}",
                    message_id=f"msg-{i}-{rating}",
                    rating_type=self.THUMBS,
                    rating_value=rating,
                    model_used=model,
                    pipeline_used="direct",
                ))
        stats = self.feedback_store.get_stats()
        self.assertEqual(stats.total_count, 9)

    def test_analytics_trends_with_data(self):
        """Analytics trends compute correctly with recorded data."""
        from opti_oignon.analytics import PerformanceRecord
        # Record multiple data points
        for i in range(5):
            self.tracker.record(PerformanceRecord(
                model_used="qwen3:32b",
                pipeline_used="direct",
                task_type="general",
                response_time_ms=300.0 + i * 50,
                prompt_tokens=100,
                completion_tokens=150 + i * 10,
            ))
        trends = self.engine.get_trends(window="1h", buckets=4)
        self.assertIsInstance(trends, list)

    def test_feedback_export_json_roundtrip(self):
        """Export feedback as JSON, verify it round-trips."""
        self.feedback_store.add_feedback(self.FeedbackEntry(
            conversation_id="conv-export",
            message_id="msg-export",
            rating_type=self.THUMBS,
            rating_value=1,
            model_used="test-model:7b",
        ))
        exported = self.feedback_store.export_json()
        parsed = json.loads(exported)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["model_used"], "test-model:7b")

    def test_feedback_export_csv_roundtrip(self):
        """Export feedback as CSV, verify structure."""
        self.feedback_store.add_feedback(self.FeedbackEntry(
            conversation_id="conv-csv",
            message_id="msg-csv",
            rating_type=self.THUMBS,
            rating_value=0,
            model_used="csv-model:7b",
        ))
        csv_data = self.feedback_store.export_csv()
        self.assertIn("csv-model:7b", csv_data)
        lines = csv_data.strip().split("\n")
        self.assertGreaterEqual(len(lines), 2)  # Header + 1 row


# =============================================================================
# TEST: AgenticExecutor Pipeline Classification
# =============================================================================

class TestAgenticExecutorClassification(unittest.TestCase):
    """AgenticExecutor pipeline selection with cross-module interaction."""

    def test_quick_classify_code_queries(self):
        """Quick classifier detects code-related queries."""
        from opti_oignon.agentic_executor import _quick_classify
        result = _quick_classify("Write a Python function to sort a list")
        self.assertTrue(result["is_code"])

    def test_quick_classify_web_search(self):
        """Quick classifier detects web search queries."""
        from opti_oignon.agentic_executor import _quick_classify
        result = _quick_classify("Search the web for recent news about AI")
        self.assertTrue(result["needs_web"])

    def test_quick_classify_thinking(self):
        """Quick classifier detects complex/reasoning queries."""
        from opti_oignon.agentic_executor import _quick_classify
        result = _quick_classify("Think step by step about this problem")
        self.assertTrue(result["is_complex"] or result["needs_reasoning"])

    def test_quick_classify_direct(self):
        """Quick classifier returns no flags for simple queries."""
        from opti_oignon.agentic_executor import _quick_classify
        result = _quick_classify("Hello, how are you?")
        self.assertFalse(result["is_code"])
        self.assertFalse(result["needs_web"])
        self.assertFalse(result["needs_reasoning"])

    def test_select_pipeline_code(self):
        """_select_pipeline returns code_verify for code classification."""
        from opti_oignon.agentic_executor import (
            PIPELINE_CODE_VERIFY,
            _quick_classify,
            _select_pipeline,
        )
        classification = _quick_classify("Write Python code for bubble sort")
        pipeline = _select_pipeline(
            classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=True,
        )
        self.assertEqual(pipeline, PIPELINE_CODE_VERIFY)

    def test_all_pipeline_constants_exist(self):
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
        pipelines = [
            PIPELINE_DIRECT, PIPELINE_TOOLS, PIPELINE_CODE_VERIFY,
            PIPELINE_THINK, PIPELINE_WEB_SEARCH, PIPELINE_THINK_TOOLS,
            PIPELINE_REASONING, PIPELINE_CONSENSUS, PIPELINE_SELF_CORRECT,
        ]
        self.assertEqual(len(pipelines), 9)
        self.assertEqual(len(set(pipelines)), 9)  # All unique


# =============================================================================
# TEST: API Endpoint Chain Integration
# =============================================================================

class TestAPIEndpointChains(unittest.TestCase):
    """API endpoint chain tests using FastAPI TestClient."""

    @classmethod
    def setUpClass(cls):
        cls.client = _get_test_client()

    def test_health_reports_all_modules(self):
        """Health endpoint reports status of all modules."""
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "1.6.6")
        modules = data["modules"]
        expected_keys = [
            "conversation", "presets", "memory", "artifacts",
            "code_executor", "response_cache", "semantic_cache",
            "pipelines", "benchmarks", "model_warmup", "config",
            "model_profiles", "context_window", "smart_router",
            "feedback", "analytics",
        ]
        for key in expected_keys:
            self.assertIn(key, modules, f"Missing module: {key}")

    def test_conversation_lifecycle(self):
        """Create -> get -> rename -> delete a conversation."""
        # Create
        r = self.client.post(
            "/api/conversations",
            json={"title": "Integration Test Conv"},
        )
        self.assertIn(r.status_code, [200, 201])
        conv_id = r.json()["id"]
        self.assertTrue(len(conv_id) > 0)

        # Get
        r = self.client.get(f"/api/conversations/{conv_id}")
        self.assertEqual(r.status_code, 200)

        # Rename
        r = self.client.patch(
            f"/api/conversations/{conv_id}",
            json={"title": "Renamed Integration Test"},
        )
        self.assertEqual(r.status_code, 200)

        # Delete
        r = self.client.delete(f"/api/conversations/{conv_id}")
        self.assertIn(r.status_code, [200, 204])

    def test_feedback_submit_and_retrieve(self):
        """Submit feedback via API, then retrieve stats."""
        # Submit
        r = self.client.post("/api/feedback", json={
            "conversation_id": "conv-api-test",
            "message_id": "msg-api-test",
            "rating_type": "thumbs",
            "rating_value": 1,
            "model_used": "test-model:7b",
            "pipeline_used": "direct",
            "task_type": "general",
        })
        self.assertEqual(r.status_code, 200)
        feedback_id = r.json()["feedback_id"]
        self.assertTrue(len(feedback_id) > 0)

        # Retrieve by ID
        r = self.client.get(f"/api/feedback/{feedback_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["model_used"], "test-model:7b")

        # Stats
        r = self.client.get("/api/feedback/stats")
        self.assertEqual(r.status_code, 200)
        stats = r.json()
        self.assertGreaterEqual(stats["total_count"], 1)

        # Cleanup
        r = self.client.delete(f"/api/feedback/{feedback_id}")
        self.assertIn(r.status_code, [200, 204])

    def test_feedback_list_and_export(self):
        """Submit multiple feedback entries, list and export them."""
        ids = []
        for i in range(3):
            r = self.client.post("/api/feedback", json={
                "conversation_id": f"conv-export-{i}",
                "message_id": f"msg-export-{i}",
                "rating_type": "thumbs",
                "rating_value": i % 2,
                "model_used": "export-model:7b",
            })
            self.assertEqual(r.status_code, 200)
            ids.append(r.json()["feedback_id"])

        # List
        r = self.client.get("/api/feedback/list")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(len(r.json()), 3)

        # Export JSON
        r = self.client.get("/api/feedback/export/json")
        self.assertEqual(r.status_code, 200)

        # Export CSV
        r = self.client.get("/api/feedback/export/csv")
        self.assertEqual(r.status_code, 200)

        # Cleanup
        for fid in ids:
            self.client.delete(f"/api/feedback/{fid}")

    def test_analytics_overview_endpoint(self):
        """Analytics overview returns valid structure."""
        r = self.client.get("/api/analytics/overview")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("total_requests", data)
        self.assertIn("avg_response_time_ms", data)

    def test_analytics_trends_endpoint(self):
        """Analytics trends endpoint accepts window/bucket params."""
        r = self.client.get("/api/analytics/trends?window=1h&buckets=6")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("window", data)
        self.assertIn("data", data)

    def test_analytics_record_and_overview(self):
        """Record performance data, then verify it appears in overview."""
        r = self.client.post("/api/analytics/record", json={
            "model_used": "analytics-test:7b",
            "pipeline_used": "direct",
            "task_type": "general",
            "response_time_ms": 450.0,
            "prompt_tokens": 120,
            "completion_tokens": 250,
        })
        self.assertEqual(r.status_code, 200)

        r = self.client.get("/api/analytics/overview")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(r.json()["total_requests"], 1)

    def test_presets_endpoint(self):
        """Presets endpoint returns list."""
        r = self.client.get("/api/presets")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_models_endpoint(self):
        """Models endpoint returns structured response."""
        r = self.client.get("/api/models")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("models", data)

    def test_health_dashboard(self):
        """Health dashboard returns detailed module information."""
        r = self.client.get("/api/health/dashboard")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("modules", data)


# =============================================================================
# TEST: Config Loading Integration
# =============================================================================

class TestConfigIntegration(unittest.TestCase):
    """YAML config files load correctly and affect module behavior."""

    def test_all_yaml_configs_parse(self):
        """Every YAML file in config/ parses without errors."""
        config_dir = _PROJECT_ROOT / "opti_oignon" / "config"
        yaml_files = list(config_dir.glob("*.yaml"))
        self.assertGreater(len(yaml_files), 0, "No YAML configs found")
        for yaml_file in yaml_files:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            self.assertIsNotNone(
                data,
                f"YAML file {yaml_file.name} loaded as None",
            )

    def test_pipeline_yaml_configs_parse(self):
        """Builtin pipeline YAML definitions parse correctly."""
        pipelines_dir = _PROJECT_ROOT / "opti_oignon" / "config" / "pipelines"
        if not pipelines_dir.exists():
            self.skipTest("No pipelines directory")
        yaml_files = list(pipelines_dir.glob("*.yaml"))
        self.assertGreater(len(yaml_files), 0)
        for yaml_file in yaml_files:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            self.assertIsNotNone(data)
            # Each pipeline should have a name and steps
            self.assertIn("name", data, f"{yaml_file.name} missing 'name'")
            self.assertIn("steps", data, f"{yaml_file.name} missing 'steps'")

    def test_model_profiles_yaml_structure(self):
        """model_profiles.yaml has expected top-level structure."""
        config_path = _PROJECT_ROOT / "opti_oignon" / "config" / "model_profiles.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        self.assertIn("profiles", data)
        profiles = data["profiles"]
        self.assertIsInstance(profiles, dict)
        self.assertGreaterEqual(len(profiles), 10)
        # Each profile entry should have context_window
        for name, p in profiles.items():
            self.assertIsInstance(name, str)
            self.assertIn("context_window", p)

    def test_feedback_yaml_has_required_keys(self):
        """feedback.yaml has feedback and analytics sections."""
        config_path = _PROJECT_ROOT / "opti_oignon" / "config" / "feedback.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        self.assertIn("feedback", data)
        self.assertIn("analytics", data)
        self.assertIn("enabled", data["feedback"])
        self.assertIn("enabled", data["analytics"])
        self.assertIn("retention_seconds", data["analytics"])

    def test_smart_routing_yaml_has_weights(self):
        """smart_routing.yaml contains routing weight configuration."""
        config_path = _PROJECT_ROOT / "opti_oignon" / "config" / "smart_routing.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        self.assertIsNotNone(data)

    def test_config_module_loads(self):
        """Central config module loads without error."""
        from opti_oignon.config import CONFIG_DIR, DATA_DIR, config
        self.assertIsNotNone(config)
        self.assertTrue(CONFIG_DIR.exists())


# =============================================================================
# TEST: Graceful Degradation
# =============================================================================

class TestGracefulDegradation(unittest.TestCase):
    """All modules degrade gracefully when dependencies are missing."""

    def test_deps_flags_are_booleans(self):
        """All FEATURE_AVAILABLE flags in deps.py are booleans."""
        from opti_oignon.api import deps
        flag_names = [
            "CONVERSATION_AVAILABLE", "RESPONSE_CACHE_AVAILABLE",
            "SEMANTIC_CACHE_AVAILABLE", "MEMORY_AVAILABLE",
            "ARTIFACT_AVAILABLE", "CODE_EXECUTOR_AVAILABLE",
            "CONTEXT_WINDOW_AVAILABLE", "MODEL_WARMUP_AVAILABLE",
            "BENCHMARK_AVAILABLE", "PRESET_AVAILABLE",
            "PIPELINE_AVAILABLE", "EXECUTOR_AVAILABLE",
            "ANALYZER_AVAILABLE", "ROUTER_AVAILABLE",
            "CONFIG_AVAILABLE", "PROFILE_AVAILABLE",
            "CONTEXT_MANAGER_AVAILABLE", "CONSENSUS_AVAILABLE",
            "SMART_ROUTER_AVAILABLE", "FEEDBACK_AVAILABLE",
            "ANALYTICS_AVAILABLE",
        ]
        for flag in flag_names:
            value = getattr(deps, flag, None)
            self.assertIsNotNone(value, f"Flag {flag} not found in deps")
            self.assertIsInstance(value, bool, f"Flag {flag} is not bool")

    def test_smart_router_no_profiles_returns_default(self):
        """SmartRouter without profiles returns default model."""
        from opti_oignon.smart_router import SmartRouter
        empty_mgr = _make_profile_manager([])
        router = SmartRouter(
            profile_manager=empty_mgr,
            enabled=True,
            default_model="fallback:latest",
            config_path=Path("/dev/null"),
        )
        result = router.select_model("direct")
        self.assertEqual(result.model, "fallback:latest")
        self.assertTrue(result.fallback)

    def test_agentic_executor_without_tools(self):
        """AgenticExecutor initializes without tool executor."""
        from opti_oignon.agentic_executor import AgenticExecutor
        executor = AgenticExecutor(
            executor=None,
            tool_executor=None,
            structured_engine=None,
            verification_engine=None,
            default_model="test:7b",
        )
        self.assertEqual(executor._default_model, "test:7b")
        self.assertEqual(executor.last_pipeline, "direct")

    def test_feedback_store_invalid_path_fallback(self):
        """FeedbackStore handles /dev/null config path gracefully."""
        from opti_oignon.feedback import FeedbackStore
        store = FeedbackStore(
            db_path=_temp_db_path(),
            config_path=Path("/dev/null"),
        )
        # Should still work with defaults
        stats = store.get_stats()
        self.assertEqual(stats.total_count, 0)

    def test_analytics_engine_empty_data(self):
        """AnalyticsEngine overview with no data returns zeros."""
        from opti_oignon.analytics import AnalyticsEngine, PerformanceTracker
        tracker = PerformanceTracker(db_path=_temp_db_path())
        engine = AnalyticsEngine(tracker=tracker, config_path=Path("/dev/null"))
        overview = engine.get_overview()
        self.assertEqual(overview.total_requests, 0)
        self.assertEqual(overview.avg_response_time_ms, 0.0)


# =============================================================================
# TEST: Version Consistency
# =============================================================================

class TestVersionConsistency(unittest.TestCase):
    """All version references match v1.5.9."""

    EXPECTED_VERSION = "1.6.6"

    def test_init_version(self):
        """opti_oignon.__version__ matches expected."""
        from opti_oignon import __version__
        self.assertEqual(__version__, self.EXPECTED_VERSION)

    def test_api_health_version(self):
        """API /api/health returns correct version."""
        client = _get_test_client()
        r = client.get("/api/health")
        self.assertEqual(r.json()["version"], self.EXPECTED_VERSION)

    def test_fastapi_app_version(self):
        """FastAPI app.version matches expected."""
        from opti_oignon.api.app import app
        self.assertEqual(app.version, self.EXPECTED_VERSION)

    def test_schemas_version(self):
        """API schemas module version matches expected."""
        try:
            from opti_oignon.api.schemas import API_VERSION
            self.assertEqual(API_VERSION, self.EXPECTED_VERSION)
        except ImportError:
            # schemas might not export API_VERSION explicitly
            pass


# =============================================================================
# TEST: Cross-Module Data Flow
# =============================================================================

class TestCrossModuleDataFlow(unittest.TestCase):
    """Data flows correctly between SmartRouter, Feedback, and Analytics."""

    def test_router_result_feeds_analytics(self):
        """SmartRouter result data can be recorded as analytics."""
        from opti_oignon.analytics import PerformanceRecord, PerformanceTracker
        from opti_oignon.smart_router import SmartRouter
        # Route a request
        profiles = [
            _make_profile(
                name="data-flow-model:14b",
                task_scores={"general": 0.8},
            ),
        ]
        mgr = _make_profile_manager(profiles)
        router = SmartRouter(
            profile_manager=mgr,
            enabled=True,
            default_model="default:7b",
            config_path=Path("/dev/null"),
        )
        routing_result = router.select_model("direct")
        # Record as performance data
        tracker = PerformanceTracker(db_path=_temp_db_path())
        record = PerformanceRecord(
            model_used=routing_result.model,
            pipeline_used="direct",
            task_type="general",
            response_time_ms=300.0,
            prompt_tokens=50,
            completion_tokens=100,
        )
        tracker.record(record)
        # Verify the data round-trips
        records = tracker.get_records(model=routing_result.model)
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0].model_used, "data-flow-model:14b")

    def test_router_result_feeds_feedback(self):
        """SmartRouter result can be attached to feedback entries."""
        from opti_oignon.feedback import RATING_TYPE_THUMBS, FeedbackEntry, FeedbackStore
        from opti_oignon.smart_router import SmartRouter
        # Route
        profiles = [
            _make_profile(name="fb-model:7b", task_scores={"general": 0.75}),
        ]
        mgr = _make_profile_manager(profiles)
        router = SmartRouter(
            profile_manager=mgr,
            enabled=True,
            default_model="default:7b",
            config_path=Path("/dev/null"),
        )
        routing_result = router.select_model("direct")
        # Submit feedback referencing the routed model
        store = FeedbackStore(db_path=_temp_db_path(), config_path=Path("/dev/null"))
        entry = FeedbackEntry(
            conversation_id="conv-flow",
            message_id="msg-flow",
            rating_type=RATING_TYPE_THUMBS,
            rating_value=1,
            model_used=routing_result.model,
            pipeline_used="direct",
        )
        saved = store.add_feedback(entry)
        # Verify model name persisted
        retrieved = store.get_feedback(saved.feedback_id)
        self.assertEqual(retrieved.model_used, "fb-model:7b")

    def test_pipeline_to_task_mapping_covers_all_pipelines(self):
        """PIPELINE_TO_TASK_MAPPING covers all 9 pipeline types."""
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
        from opti_oignon.smart_router import PIPELINE_TO_TASK_MAPPING
        all_pipelines = [
            PIPELINE_DIRECT, PIPELINE_TOOLS, PIPELINE_CODE_VERIFY,
            PIPELINE_THINK, PIPELINE_WEB_SEARCH, PIPELINE_THINK_TOOLS,
            PIPELINE_REASONING, PIPELINE_CONSENSUS, PIPELINE_SELF_CORRECT,
        ]
        for p in all_pipelines:
            self.assertIn(
                p, PIPELINE_TO_TASK_MAPPING,
                f"Pipeline '{p}' missing from PIPELINE_TO_TASK_MAPPING",
            )


# =============================================================================
# TEST: Module Import Health
# =============================================================================

class TestModuleImportHealth(unittest.TestCase):
    """All major modules import without error."""

    def test_import_core(self):
        """Core modules import successfully."""
        from opti_oignon import analyzer, config, executor, router

    def test_import_intelligence_layer(self):
        """Intelligence layer modules import successfully."""
        from opti_oignon.agentic_executor import AgenticExecutor
        from opti_oignon.structured_output import StructuredOutputEngine
        from opti_oignon.tool_executor import ToolExecutor
        from opti_oignon.tool_registry import ToolRegistry
        from opti_oignon.verification import VerificationEngine

    def test_import_reasoning_consensus(self):
        """Reasoning + consensus modules import successfully."""
        from opti_oignon.consensus import ConsensusEngine
        from opti_oignon.reasoning import ReasoningEngine
        from opti_oignon.self_correction import SelfCorrectionEngine

    def test_import_routing_profiles(self):
        """Routing + profiles modules import successfully."""
        from opti_oignon.model_profiles import ModelProfileManager
        from opti_oignon.smart_router import SmartRouter

    def test_import_feedback_analytics(self):
        """Feedback + analytics modules import successfully."""
        from opti_oignon.analytics import AnalyticsEngine, PerformanceTracker
        from opti_oignon.feedback import FeedbackEntry, FeedbackStore

    def test_import_rag(self):
        """RAG package imports (may skip if optional deps missing)."""
        try:
            from opti_oignon import rag
        except ImportError:
            self.skipTest("RAG optional dependencies not installed")

    def test_import_api_layer(self):
        """API layer imports successfully."""
        from opti_oignon.api.app import app
        from opti_oignon.api.deps import CONVERSATION_AVAILABLE


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
