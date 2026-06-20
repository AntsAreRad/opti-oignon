#!/usr/bin/env python3
"""
Tests for session_fingerprint.py — Opti-Oignon S75

Covers all 10 dimensions, FingerprintManager hooks,
serialization, persistence, and edge cases.
"""

import importlib.util
import json
import os
import sqlite3
import tempfile
import time
import unittest

# ---------------------------------------------------------------------------
# Direct module loading (bypass __init__.py chain)
# ---------------------------------------------------------------------------

_MOD_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "opti_oignon", "session_fingerprint.py",
)
_spec = importlib.util.spec_from_file_location("session_fingerprint", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Import symbols
classify_task = _mod.classify_task
TaskType = _mod.TaskType
detect_stack = _mod.detect_stack
HotFilesTracker = _mod.HotFilesTracker
classify_bug = _mod.classify_bug
BugTracker = _mod.BugTracker
SuiteHealthTracker = _mod.SuiteHealthTracker
MomentumTracker = _mod.MomentumTracker
extract_terms = _mod.extract_terms
compute_tfidf = _mod.compute_tfidf
DomainTermsTracker = _mod.DomainTermsTracker
build_import_graph = _mod.build_import_graph
find_clusters = _mod.find_clusters
DepClustersTracker = _mod.DepClustersTracker
UserPreferencesStore = _mod.UserPreferencesStore
ContextAnchorsTracker = _mod.ContextAnchorsTracker
FingerprintManager = _mod.FingerprintManager
FingerprintConfig = _mod.FingerprintConfig
FINGERPRINT_AVAILABLE = _mod.FINGERPRINT_AVAILABLE


def _temp_db() -> str:
    """Return a temporary database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


# ===================================================================
# D1: Task Type Classification
# ===================================================================

class TestTaskTypeClassification(unittest.TestCase):
    """Tests for D1 task type classification."""

    def test_bug_fix_detected(self):
        result = classify_task("Fix the import error in utils.py")
        self.assertEqual(result["type"], TaskType.BUG_FIX)

    def test_refactor_detected(self):
        result = classify_task("Refactor the database module to simplify queries")
        self.assertEqual(result["type"], TaskType.REFACTOR)

    def test_test_detected(self):
        result = classify_task("Write pytest unit tests for the parser coverage")
        self.assertEqual(result["type"], TaskType.TEST)

    def test_docs_detected(self):
        result = classify_task("Update the README with new configuration docs")
        self.assertEqual(result["type"], TaskType.DOCS)

    def test_create_detected(self):
        result = classify_task("Create a new authentication module")
        self.assertEqual(result["type"], TaskType.CREATE)

    def test_feature_detected(self):
        result = classify_task("Add support for WebSocket streaming and enhance the UI")
        self.assertEqual(result["type"], TaskType.FEATURE)

    def test_unknown_for_empty(self):
        result = classify_task("")
        self.assertEqual(result["type"], TaskType.UNKNOWN)
        self.assertEqual(result["confidence"], 0.0)

    def test_unknown_for_gibberish(self):
        result = classify_task("lorem ipsum dolor sit amet")
        self.assertEqual(result["type"], TaskType.UNKNOWN)

    def test_complexity_simple(self):
        result = classify_task("Fix bug")
        self.assertEqual(result["complexity"], "simple")

    def test_complexity_moderate(self):
        text = " ".join(["word"] * 40) + " fix the bug"
        result = classify_task(text)
        self.assertEqual(result["complexity"], "moderate")

    def test_complexity_complex(self):
        text = " ".join(["word"] * 100) + " fix the bug"
        result = classify_task(text)
        self.assertEqual(result["complexity"], "complex")

    def test_confidence_between_0_and_1(self):
        result = classify_task("Fix the crash and add new tests for coverage")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


# ===================================================================
# D2: Stack Detection
# ===================================================================

class TestStackDetection(unittest.TestCase):
    """Tests for D2 stack detection."""

    def test_python_from_extensions(self):
        result = detect_stack(["main.py", "utils.py", "test_main.py"])
        self.assertEqual(result["primary"], "python")

    def test_typescript_from_extensions(self):
        result = detect_stack(["app.ts", "types.ts", "index.ts"])
        self.assertEqual(result["primary"], "typescript")

    def test_mixed_stack(self):
        result = detect_stack(["main.py", "app.svelte", "config.yaml"])
        self.assertIn("python", result["languages"])
        self.assertIn("svelte", result["languages"])

    def test_framework_detection_fastapi(self):
        contents = {"app.py": "from fastapi import FastAPI\napp = FastAPI()"}
        result = detect_stack(["app.py"], file_contents=contents)
        self.assertIn("fastapi", result["frameworks"])

    def test_framework_detection_pytest(self):
        contents = {"test.py": "import pytest\ndef test_foo(): pass"}
        result = detect_stack(["test.py"], file_contents=contents)
        self.assertIn("pytest", result["frameworks"])

    def test_unknown_for_empty(self):
        result = detect_stack([])
        self.assertEqual(result["primary"], "unknown")

    def test_unknown_extension_ignored(self):
        result = detect_stack(["data.xyz", "config.abc"])
        self.assertEqual(result["primary"], "unknown")


# ===================================================================
# D3: Hot Files
# ===================================================================

class TestHotFilesTracker(unittest.TestCase):
    """Tests for D3 hot files tracking."""

    def test_touch_increments(self):
        tracker = HotFilesTracker()
        tracker.touch("a.py")
        tracker.touch("a.py")
        tracker.touch("b.py")
        top = tracker.top(5)
        self.assertEqual(top[0]["path"], "a.py")
        self.assertEqual(top[0]["touches"], 2)

    def test_file_count(self):
        tracker = HotFilesTracker()
        tracker.touch("a.py")
        tracker.touch("b.py")
        self.assertEqual(tracker.file_count, 2)

    def test_avg_file_size(self):
        tracker = HotFilesTracker()
        tracker.touch("a.py", size=100)
        tracker.touch("b.py", size=200)
        self.assertEqual(tracker.avg_file_size, 150)

    def test_serialize_top_n(self):
        tracker = HotFilesTracker()
        for i in range(10):
            tracker.touch(f"file_{i}.py")
        result = tracker.serialize(max_files=3)
        self.assertEqual(len(result["top"]), 3)


# ===================================================================
# D4: Recent Bugs
# ===================================================================

class TestBugClassification(unittest.TestCase):
    """Tests for D4 bug classification."""

    def test_assertion_error(self):
        self.assertEqual(classify_bug("AssertionError: 5 != 3"), "assertion")

    def test_import_error(self):
        self.assertEqual(classify_bug("ImportError: No module named xyz"), "import")

    def test_type_error(self):
        self.assertEqual(classify_bug("TypeError: expected int got str"), "type")

    def test_syntax_error(self):
        self.assertEqual(classify_bug("SyntaxError: invalid syntax"), "syntax")

    def test_runtime_error(self):
        self.assertEqual(classify_bug("RuntimeError: recursion limit"), "runtime")

    def test_unknown_for_empty(self):
        self.assertEqual(classify_bug(""), "unknown")

    def test_unknown_for_novel_error(self):
        self.assertEqual(classify_bug("Something went terribly wrong"), "unknown")

    def test_bug_tracker_records(self):
        tracker = BugTracker()
        cat = tracker.record("ImportError: no module xyz", step=1)
        self.assertEqual(cat, "import")
        self.assertEqual(tracker.category_counts["import"], 1)

    def test_bug_tracker_max_history(self):
        tracker = BugTracker()
        tracker._max_history = 5
        for i in range(10):
            tracker.record(f"AssertionError #{i}")
        self.assertEqual(len(tracker._bugs), 5)


# ===================================================================
# D5: Test Health
# ===================================================================

class TestSuiteHealthTracker(unittest.TestCase):
    """Tests for D5 test health tracking."""

    def test_pass_rate_all_pass(self):
        tracker = SuiteHealthTracker()
        tracker.record(True)
        tracker.record(True)
        self.assertEqual(tracker.pass_rate, 1.0)

    def test_pass_rate_mixed(self):
        tracker = SuiteHealthTracker()
        tracker.record(True)
        tracker.record(False, "assertion")
        self.assertAlmostEqual(tracker.pass_rate, 0.5)

    def test_pass_rate_empty(self):
        tracker = SuiteHealthTracker()
        self.assertEqual(tracker.pass_rate, 1.0)

    def test_last_result(self):
        tracker = SuiteHealthTracker()
        tracker.record(True)
        tracker.record(False)
        self.assertFalse(tracker.last_result)

    def test_common_failure_types(self):
        tracker = SuiteHealthTracker()
        tracker.record(False, "assertion")
        tracker.record(False, "assertion")
        tracker.record(False, "import")
        failures = tracker.common_failure_types
        self.assertEqual(failures[0][0], "assertion")
        self.assertEqual(failures[0][1], 2)


# ===================================================================
# D6: Session Momentum
# ===================================================================

class TestMomentumTracker(unittest.TestCase):
    """Tests for D6 session momentum."""

    def test_complete_step(self):
        tracker = MomentumTracker()
        tracker.set_total_steps(5)
        tracker.complete_step()
        self.assertEqual(tracker.steps_completed, 1)
        self.assertEqual(tracker.steps_remaining, 4)

    def test_progress_ratio(self):
        tracker = MomentumTracker()
        tracker.set_total_steps(4)
        tracker.complete_step()
        tracker.complete_step()
        self.assertAlmostEqual(tracker.progress_ratio, 0.5)

    def test_stuck_count(self):
        tracker = MomentumTracker()
        tracker.record_stuck()
        tracker.record_stuck()
        self.assertEqual(tracker.stuck_count, 2)

    def test_velocity_zero_initially(self):
        tracker = MomentumTracker()
        self.assertEqual(tracker.velocity, 0.0)

    def test_serialize_includes_all_fields(self):
        tracker = MomentumTracker()
        tracker.set_total_steps(3)
        tracker.complete_step()
        s = tracker.serialize()
        self.assertIn("completed", s)
        self.assertIn("remaining", s)
        self.assertIn("velocity", s)
        self.assertIn("progress", s)


# ===================================================================
# D7: Domain Terms (TF-IDF)
# ===================================================================

class TestDomainTerms(unittest.TestCase):
    """Tests for D7 domain term extraction and TF-IDF."""

    def test_extract_terms_python(self):
        code = "def calculate_total_price(items):\n    pass\nclass OrderManager:\n    pass"
        terms = extract_terms(code)
        self.assertIn("calculate", terms)
        self.assertIn("total", terms)
        self.assertIn("price", terms)

    def test_extract_terms_filters_stop_words(self):
        code = "def init(self):\n    return None"
        terms = extract_terms(code)
        self.assertNotIn("self", terms)
        self.assertNotIn("none", terms)

    def test_extract_terms_empty(self):
        self.assertEqual(extract_terms(""), [])

    def test_compute_tfidf_returns_sorted(self):
        docs = [["alpha", "beta", "alpha"], ["beta", "gamma"]]
        result = compute_tfidf(docs, max_terms=5)
        self.assertIsInstance(result, list)
        if len(result) >= 2:
            self.assertGreaterEqual(result[0][1], result[1][1])

    def test_domain_terms_tracker_refresh(self):
        tracker = DomainTermsTracker()
        tracker._refresh_interval = 2
        tracker.update_file("a.py", "def calculate_price(): pass")
        tracker.update_file("b.py", "def compute_total(): pass")
        self.assertTrue(tracker.should_refresh())
        terms = tracker.refresh()
        self.assertIsInstance(terms, list)


# ===================================================================
# D8: Dependency Clusters
# ===================================================================

class TestDepClusters(unittest.TestCase):
    """Tests for D8 dependency cluster detection."""

    def test_build_import_graph(self):
        contents = {
            "app.py": "from utils import helper\nimport config",
            "utils.py": "import os\nimport config",
        }
        graph = build_import_graph(contents)
        self.assertIn("utils", graph.get("app", set()))
        self.assertIn("config", graph.get("app", set()))

    def test_find_clusters_connected(self):
        graph = {"a": {"b"}, "b": {"c"}, "d": {"e"}}
        clusters = find_clusters(graph)
        self.assertTrue(len(clusters) >= 2)
        # a-b-c should be in one cluster
        for c in clusters:
            if "a" in c:
                self.assertIn("b", c)
                self.assertIn("c", c)

    def test_find_clusters_empty(self):
        clusters = find_clusters({})
        self.assertEqual(clusters, [])

    def test_dep_clusters_tracker_compute(self):
        tracker = DepClustersTracker()
        contents = {
            "main.py": "import utils\nimport config",
            "utils.py": "import config",
        }
        tracker.compute(contents)
        self.assertTrue(tracker._computed)
        self.assertTrue(len(tracker.clusters) > 0)


# ===================================================================
# D9: User Preferences (SQLite persistence)
# ===================================================================

class TestUserPreferencesStore(unittest.TestCase):
    """Tests for D9 persistent user preferences."""

    def setUp(self):
        self._db_path = _temp_db()
        self.store = UserPreferencesStore(db_path=self._db_path)

    def tearDown(self):
        self.store.close()
        if os.path.exists(self._db_path):
            os.unlink(self._db_path)

    def test_record_and_ratios(self):
        self.store.record("approve", "planning")
        self.store.record("approve", "planning")
        self.store.record("modify", "review")
        ratios = self.store.get_ratios()
        self.assertAlmostEqual(ratios["approve"], 2 / 3, places=2)
        self.assertAlmostEqual(ratios["modify"], 1 / 3, places=2)
        self.assertAlmostEqual(ratios["abort"], 0.0)

    def test_total_decisions(self):
        self.store.record("approve")
        self.store.record("abort")
        self.assertEqual(self.store.total_decisions, 2)

    def test_phase_preferences(self):
        self.store.record("approve", "planning")
        self.store.record("modify", "planning")
        self.store.record("approve", "review")
        prefs = self.store.get_phase_preferences()
        self.assertEqual(prefs["planning"]["approve"], 1)
        self.assertEqual(prefs["planning"]["modify"], 1)
        self.assertEqual(prefs["review"]["approve"], 1)

    def test_empty_ratios(self):
        ratios = self.store.get_ratios()
        self.assertEqual(ratios["approve"], 0.0)
        self.assertEqual(ratios["modify"], 0.0)
        self.assertEqual(ratios["abort"], 0.0)

    def test_persistence_across_instances(self):
        self.store.record("approve", "planning")
        self.store.close()
        store2 = UserPreferencesStore(db_path=self._db_path)
        self.assertEqual(store2.total_decisions, 1)
        store2.close()

    def test_serialize_structure(self):
        self.store.record("approve")
        result = self.store.serialize()
        self.assertIn("ratios", result)
        self.assertIn("total_decisions", result)
        self.assertIn("phase_prefs", result)


# ===================================================================
# D10: Context Anchors
# ===================================================================

class TestContextAnchors(unittest.TestCase):
    """Tests for D10 context anchors."""

    def test_add_anchor(self):
        tracker = ContextAnchorsTracker()
        tracker.add("Must preserve backward compat")
        self.assertEqual(len(tracker.anchors), 1)

    def test_max_anchors_eviction(self):
        tracker = ContextAnchorsTracker()
        tracker._max_anchors = 3
        for i in range(5):
            tracker.add(f"Anchor number {i}")
        self.assertEqual(len(tracker.anchors), 3)

    def test_dedup_by_content(self):
        tracker = ContextAnchorsTracker()
        tracker.add("Same anchor text")
        tracker.add("Same anchor text")
        self.assertEqual(len(tracker.anchors), 1)

    def test_empty_ignored(self):
        tracker = ContextAnchorsTracker()
        tracker.add("")
        self.assertEqual(len(tracker.anchors), 0)

    def test_truncation_at_200_chars(self):
        tracker = ContextAnchorsTracker()
        long_text = "x" * 300
        tracker.add(long_text)
        self.assertLessEqual(len(tracker.anchors[0]), 200)


# ===================================================================
# FingerprintManager — Integration
# ===================================================================

class TestFingerprintManagerInit(unittest.TestCase):
    """Tests for FingerprintManager initialization."""

    def _make_manager(self):
        db = _temp_db()
        self._db_paths = getattr(self, "_db_paths", [])
        self._db_paths.append(db)
        return FingerprintManager(
            preferences_store=UserPreferencesStore(db_path=db)
        )

    def tearDown(self):
        for p in getattr(self, "_db_paths", []):
            if os.path.exists(p):
                os.unlink(p)

    def test_initial_state(self):
        mgr = self._make_manager()
        self.assertEqual(mgr.step_count, 0)
        self.assertEqual(mgr.task_type, TaskType.UNKNOWN)

    def test_set_task(self):
        mgr = self._make_manager()
        mgr.set_task("Fix the broken test suite", total_steps=8)
        self.assertEqual(mgr.task_type, TaskType.BUG_FIX)

    def test_config_loaded(self):
        config = FingerprintConfig(max_anchors=7)
        db = _temp_db()
        self._db_paths = getattr(self, "_db_paths", [])
        self._db_paths.append(db)
        mgr = FingerprintManager(
            config=config,
            preferences_store=UserPreferencesStore(db_path=db),
        )
        self.assertEqual(mgr.config.max_anchors, 7)


class TestFingerprintManagerHooks(unittest.TestCase):
    """Tests for FingerprintManager on_step/on_test/on_checkpoint."""

    def setUp(self):
        self._db_path = _temp_db()
        self.mgr = FingerprintManager(
            preferences_store=UserPreferencesStore(db_path=self._db_path)
        )
        self.mgr.set_task("Create a new REST API endpoint", total_steps=5)

    def tearDown(self):
        if os.path.exists(self._db_path):
            os.unlink(self._db_path)

    def test_on_step_updates_hot_files(self):
        self.mgr.on_step({"file_path": "routes.py", "completed": True})
        state = self.mgr.get_full_state()
        self.assertEqual(state["d3_hot_files"]["file_count"], 1)

    def test_on_step_updates_momentum(self):
        self.mgr.on_step({"file_path": "a.py", "completed": True})
        state = self.mgr.get_full_state()
        self.assertEqual(state["d6_momentum"]["completed"], 1)

    def test_on_step_updates_stack(self):
        self.mgr.on_step({"file_path": "app.ts", "completed": True})
        state = self.mgr.get_full_state()
        self.assertEqual(state["d2_stack"]["primary"], "typescript")

    def test_on_step_increments_step_count(self):
        self.mgr.on_step({"file_path": "a.py", "completed": True})
        self.mgr.on_step({"file_path": "b.py", "completed": True})
        self.assertEqual(self.mgr.step_count, 2)

    def test_on_test_pass(self):
        self.mgr.on_test({"passed": True, "output": "5 passed"})
        state = self.mgr.get_full_state()
        self.assertEqual(state["d5_test_health"]["pass_rate"], 1.0)

    def test_on_test_fail_records_bug(self):
        self.mgr.on_test({
            "passed": False,
            "output": "TypeError: expected int got str",
        })
        state = self.mgr.get_full_state()
        self.assertIn("type", state["d4_recent_bugs"]["categories"])

    def test_on_test_fail_records_stuck(self):
        self.mgr.on_test({"passed": False, "error": "AssertionError"})
        state = self.mgr.get_full_state()
        self.assertEqual(state["d6_momentum"]["stuck_count"], 1)

    def test_on_checkpoint_approve(self):
        self.mgr.on_checkpoint({
            "action": "approve",
            "phase": "planning",
        })
        state = self.mgr.get_full_state()
        ratios = state["d9_user_preferences"]["ratios"]
        self.assertEqual(ratios["approve"], 1.0)

    def test_on_checkpoint_with_anchor(self):
        self.mgr.on_checkpoint({
            "action": "modify",
            "phase": "review",
            "anchor": "Keep backward compat with v1 API",
        })
        state = self.mgr.get_full_state()
        self.assertEqual(state["d10_context_anchors"]["count"], 1)

    def test_add_anchor_direct(self):
        self.mgr.add_anchor("Never delete user data")
        state = self.mgr.get_full_state()
        self.assertIn("Never delete user data", state["d10_context_anchors"]["anchors"])


class TestFingerprintManagerSerialization(unittest.TestCase):
    """Tests for FingerprintManager serialization."""

    def setUp(self):
        self._db_path = _temp_db()
        self.mgr = FingerprintManager(
            preferences_store=UserPreferencesStore(db_path=self._db_path)
        )

    def tearDown(self):
        if os.path.exists(self._db_path):
            os.unlink(self._db_path)

    def test_serialize_empty_session(self):
        result = self.mgr.serialize()
        self.assertIsInstance(result, dict)

    def test_serialize_after_activity(self):
        self.mgr.set_task("Fix import errors", total_steps=3)
        self.mgr.on_step({"file_path": "utils.py", "content": "def foo(): pass", "completed": True})
        self.mgr.on_test({"passed": False, "output": "ImportError: no module"})
        result = self.mgr.serialize()
        self.assertIn("task", result)
        self.assertIn("bugs", result)

    def test_serialize_compact_yaml(self):
        self.mgr._config.serialization_format = "yaml"
        self.mgr.set_task("Add feature", total_steps=2)
        self.mgr.on_step({"file_path": "feat.py", "completed": True})
        compact = self.mgr.serialize_compact()
        self.assertIsInstance(compact, str)
        self.assertTrue(len(compact) > 0)

    def test_serialize_compact_json(self):
        self.mgr._config.serialization_format = "json"
        self.mgr.set_task("Add feature", total_steps=2)
        self.mgr.on_step({"file_path": "feat.py", "completed": True})
        compact = self.mgr.serialize_compact()
        parsed = json.loads(compact)
        self.assertIsInstance(parsed, dict)

    def test_serialize_respects_zero_weight(self):
        self.mgr._config.dimension_weights = {
            "task_type": 0.0,
            "stack": 0.0,
            "hot_files": 0.0,
            "recent_bugs": 0.0,
            "test_health": 0.0,
            "momentum": 0.0,
            "domain_terms": 0.0,
            "dep_clusters": 0.0,
            "user_preferences": 0.0,
            "context_anchors": 0.0,
        }
        self.mgr.set_task("Fix bug", total_steps=1)
        self.mgr.on_step({"file_path": "a.py", "completed": True})
        result = self.mgr.serialize()
        self.assertEqual(result, {})

    def test_get_full_state_all_dimensions(self):
        state = self.mgr.get_full_state()
        self.assertIn("d1_task_type", state)
        self.assertIn("d2_stack", state)
        self.assertIn("d3_hot_files", state)
        self.assertIn("d4_recent_bugs", state)
        self.assertIn("d5_test_health", state)
        self.assertIn("d6_momentum", state)
        self.assertIn("d7_domain_terms", state)
        self.assertIn("d8_dep_clusters", state)
        self.assertIn("d9_user_preferences", state)
        self.assertIn("d10_context_anchors", state)
        self.assertIn("step_count", state)


class TestFingerprintManagerBatchOps(unittest.TestCase):
    """Tests for batch operations (D7 refresh, D8 clusters)."""

    def setUp(self):
        self._db_path = _temp_db()
        self.mgr = FingerprintManager(
            preferences_store=UserPreferencesStore(db_path=self._db_path)
        )

    def tearDown(self):
        if os.path.exists(self._db_path):
            os.unlink(self._db_path)

    def test_compute_dep_clusters(self):
        contents = {
            "app.py": "import utils\nfrom config import settings",
            "utils.py": "import config",
            "standalone.py": "import os",
        }
        self.mgr.compute_dep_clusters(contents)
        state = self.mgr.get_full_state()
        self.assertTrue(len(state["d8_dep_clusters"]["clusters"]) > 0)

    def test_refresh_domain_terms(self):
        self.mgr.on_step({
            "file_path": "parser.py",
            "content": "def parse_json_response(data):\n    pass\nclass ResponseParser:\n    pass",
            "completed": True,
        })
        self.mgr.refresh_domain_terms()
        state = self.mgr.get_full_state()
        self.assertIsInstance(state["d7_domain_terms"]["terms"], dict)

    def test_tfidf_auto_refresh_after_interval(self):
        self.mgr._config.tfidf_refresh_interval = 2
        self.mgr._domain_terms._refresh_interval = 2
        self.mgr.on_step({
            "file_path": "a.py",
            "content": "def calculate_price(): pass",
            "completed": True,
        })
        self.mgr.on_step({
            "file_path": "b.py",
            "content": "def compute_total(): pass",
            "completed": True,
        })
        # After 2 steps, refresh should have happened
        terms = self.mgr._domain_terms.terms
        self.assertIsInstance(terms, list)


class TestFingerprintAvailability(unittest.TestCase):
    """Tests for module-level availability flag."""

    def test_flag_is_true(self):
        self.assertTrue(FINGERPRINT_AVAILABLE)

    def test_module_singleton_exists(self):
        self.assertIsNotNone(_mod.fingerprint_manager)


if __name__ == "__main__":
    unittest.main()
