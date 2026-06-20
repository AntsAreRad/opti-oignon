"""
Tests for S96 -- Fine-Tuning Data Export & Management.

Validates:
- Part 1: FineTuneExporter (formats, filters, quality scoring)
- Part 2: FineTuneTracker (variant CRUD, A/B comparison, stats)
- Part 3: API routes (endpoints, schemas, error handling)
- Part 4: Frontend (types, API client, FineTunePanel, settings page)
- Part 5: Config persistence (fine_tune.yaml)
- Part 6: Integration wiring (deps.py, app.py, version bump)
- Zero regressions

Target: ~50 tests
"""

import importlib.util
import json
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
API_DIR = os.path.join(BACKEND_DIR, "api")
CONFIG_DIR = os.path.join(BACKEND_DIR, "config")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components")
SETTINGS_DIR = os.path.join(COMPONENTS_DIR, "settings")
ROUTES_DIR = os.path.join(FRONTEND_SRC, "routes")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    """Read file contents as string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Load modules in isolation
# ---------------------------------------------------------------------------

fine_tune_export_mod = _load_module(
    "opti_oignon.fine_tune_export",
    os.path.join(BACKEND_DIR, "fine_tune_export.py"),
)
FineTuneExporter = fine_tune_export_mod.FineTuneExporter
ExportFilter = fine_tune_export_mod.ExportFilter
QualityScorer = fine_tune_export_mod.QualityScorer
QualityScore = fine_tune_export_mod.QualityScore
ExportResult = fine_tune_export_mod.ExportResult

fine_tune_tracker_mod = _load_module(
    "opti_oignon.fine_tune_tracker",
    os.path.join(BACKEND_DIR, "fine_tune_tracker.py"),
)
FineTuneTracker = fine_tune_tracker_mod.FineTuneTracker
FineTuneVariant = fine_tune_tracker_mod.FineTuneVariant
ComparisonPrompt = fine_tune_tracker_mod.ComparisonPrompt
ComparisonResult = fine_tune_tracker_mod.ComparisonResult


# ===========================================================================
# PART 1: FineTuneExporter -- Formats
# ===========================================================================

class TestExportFormats(unittest.TestCase):
    """Test ShareGPT, Alpaca, and JSONL export formats."""

    def setUp(self):
        self.exporter = FineTuneExporter.__new__(FineTuneExporter)
        self.exporter._config = {"export": {"include_system_messages": True, "strip_whitespace": True}}
        self.exporter._scorer = QualityScorer()
        self.exporter._conversation_manager = None
        self.exporter._feedback_store = None

    def _make_conv(self, conv_id="conv-1", messages=None):
        if messages is None:
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        return {"id": conv_id, "messages": messages}

    def test_sharegpt_structure(self):
        """ShareGPT format uses 'from' and 'value' keys with role mapping."""
        convs = [self._make_conv()]
        data, count = self.exporter._format_sharegpt(convs)
        parsed = json.loads(data)
        self.assertEqual(len(parsed), 1)
        self.assertIn("conversations", parsed[0])
        turns = parsed[0]["conversations"]
        self.assertEqual(turns[0]["from"], "human")
        self.assertEqual(turns[1]["from"], "gpt")
        self.assertEqual(count, 2)

    def test_alpaca_structure(self):
        """Alpaca format produces instruction/input/output entries."""
        convs = [self._make_conv()]
        data, count = self.exporter._format_alpaca(convs)
        parsed = json.loads(data)
        self.assertEqual(len(parsed), 1)
        self.assertIn("instruction", parsed[0])
        self.assertIn("input", parsed[0])
        self.assertIn("output", parsed[0])
        self.assertEqual(parsed[0]["instruction"], "Hello")
        self.assertEqual(parsed[0]["output"], "Hi there!")
        self.assertEqual(count, 2)

    def test_alpaca_with_system_message(self):
        """Alpaca injects system message into the input field."""
        convs = [self._make_conv(messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ])]
        data, _ = self.exporter._format_alpaca(convs)
        parsed = json.loads(data)
        self.assertEqual(parsed[0]["input"], "You are helpful")

    def test_jsonl_structure(self):
        """JSONL format produces one JSON object per line."""
        convs = [self._make_conv(), self._make_conv(conv_id="conv-2")]
        data, count = self.exporter._format_jsonl(convs)
        lines = data.strip().split("\n")
        self.assertEqual(len(lines), 2)
        for line in lines:
            parsed = json.loads(line)
            self.assertIn("messages", parsed)
        self.assertEqual(count, 4)

    def test_sharegpt_strips_whitespace(self):
        """ShareGPT strips leading/trailing whitespace from content."""
        convs = [self._make_conv(messages=[
            {"role": "user", "content": "  Hello  "},
            {"role": "assistant", "content": "\nWorld\n"},
        ])]
        data, _ = self.exporter._format_sharegpt(convs)
        parsed = json.loads(data)
        self.assertEqual(parsed[0]["conversations"][0]["value"], "Hello")
        self.assertEqual(parsed[0]["conversations"][1]["value"], "World")

    def test_sharegpt_skips_empty_content(self):
        """ShareGPT skips messages with empty content after strip."""
        convs = [self._make_conv(messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "   "},
        ])]
        data, count = self.exporter._format_sharegpt(convs)
        parsed = json.loads(data)
        turns = parsed[0]["conversations"]
        self.assertEqual(len(turns), 1)
        self.assertEqual(count, 1)

    def test_sharegpt_excludes_system_when_configured(self):
        """ShareGPT respects include_system_messages=False."""
        self.exporter._config["export"]["include_system_messages"] = False
        convs = [self._make_conv(messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])]
        data, count = self.exporter._format_sharegpt(convs)
        parsed = json.loads(data)
        roles = [t["from"] for t in parsed[0]["conversations"]]
        self.assertNotIn("system", roles)
        self.assertEqual(count, 2)

    def test_export_invalid_format_raises(self):
        """export() raises ValueError for unknown format."""
        with self.assertRaises(ValueError) as ctx:
            self.exporter.export(fmt="xml")
        self.assertIn("xml", str(ctx.exception))

    def test_export_empty_returns_empty(self):
        """export() returns empty data when no conversations match."""
        result = self.exporter.export(fmt="sharegpt")
        self.assertEqual(result.conversation_count, 0)
        self.assertEqual(result.data, "[]")

    def test_export_jsonl_empty(self):
        """JSONL export returns empty string for no conversations."""
        result = self.exporter.export(fmt="jsonl")
        self.assertEqual(result.data, "")

    def test_alpaca_multi_turn_pairs(self):
        """Alpaca creates one entry per user-assistant pair."""
        convs = [self._make_conv(messages=[
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ])]
        data, count = self.exporter._format_alpaca(convs)
        parsed = json.loads(data)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(count, 4)


# ===========================================================================
# PART 1b: Quality Scoring
# ===========================================================================

class TestQualityScorer(unittest.TestCase):
    """Test conversation quality scoring logic."""

    def setUp(self):
        self.scorer = QualityScorer(
            feedback_weight=0.6,
            benchmark_weight=0.4,
            default_score=0.5,
            min_feedback_count=1,
        )

    def test_no_data_returns_default(self):
        """Score with no feedback or benchmarks returns default."""
        qs = self.scorer.score_conversation("conv-1")
        self.assertAlmostEqual(qs.combined_score, 0.5)
        self.assertFalse(qs.has_feedback)
        self.assertFalse(qs.has_benchmarks)

    def test_thumbs_up_only(self):
        """All thumbs up gives score of 1.0."""
        entries = [
            {"rating_type": "thumbs", "rating_value": 1},
            {"rating_type": "thumbs", "rating_value": 1},
        ]
        qs = self.scorer.score_conversation("conv-1", feedback_entries=entries)
        self.assertAlmostEqual(qs.feedback_score, 1.0)
        self.assertTrue(qs.has_feedback)
        self.assertEqual(qs.feedback_count, 2)

    def test_thumbs_mixed(self):
        """Mixed thumbs give 0.5 score."""
        entries = [
            {"rating_type": "thumbs", "rating_value": 1},
            {"rating_type": "thumbs", "rating_value": 0},
        ]
        qs = self.scorer.score_conversation("conv-1", feedback_entries=entries)
        self.assertAlmostEqual(qs.feedback_score, 0.5)

    def test_stars_normalized(self):
        """Star ratings are normalized to 0.0-1.0."""
        entries = [{"rating_type": "stars", "rating_value": 5}]
        qs = self.scorer.score_conversation("conv-1", feedback_entries=entries)
        self.assertAlmostEqual(qs.feedback_score, 1.0)

        entries2 = [{"rating_type": "stars", "rating_value": 1}]
        qs2 = self.scorer.score_conversation("conv-2", feedback_entries=entries2)
        self.assertAlmostEqual(qs2.feedback_score, 0.0)

    def test_combined_weights(self):
        """Combined score uses configured weights when both sources present."""
        entries = [{"rating_type": "thumbs", "rating_value": 1}]
        benchmarks = [0.8]
        qs = self.scorer.score_conversation(
            "conv-1", feedback_entries=entries, benchmark_scores=benchmarks
        )
        expected = 0.6 * 1.0 + 0.4 * 0.8
        self.assertAlmostEqual(qs.combined_score, expected, places=4)

    def test_benchmark_only(self):
        """Benchmark-only score uses benchmark directly."""
        qs = self.scorer.score_conversation("conv-1", benchmark_scores=[0.7, 0.9])
        self.assertAlmostEqual(qs.combined_score, 0.8)
        self.assertTrue(qs.has_benchmarks)

    def test_min_feedback_count(self):
        """Below min_feedback_count, feedback is ignored."""
        scorer = QualityScorer(min_feedback_count=5)
        entries = [{"rating_type": "thumbs", "rating_value": 1}]
        qs = scorer.score_conversation("conv-1", feedback_entries=entries)
        self.assertFalse(qs.has_feedback)

    def test_quality_score_to_dict(self):
        """QualityScore.to_dict serializes all fields."""
        qs = QualityScore(conversation_id="c1", combined_score=0.75)
        d = qs.to_dict()
        self.assertEqual(d["conversation_id"], "c1")
        self.assertIn("combined_score", d)


# ===========================================================================
# PART 2: FineTuneTracker -- Variant CRUD
# ===========================================================================

class TestFineTuneTrackerCRUD(unittest.TestCase):
    """Test variant registration, listing, update, and deletion."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.tracker = FineTuneTracker(
            db_path=Path(self.tmp.name),
            config_path=Path(os.path.join(CONFIG_DIR, "fine_tune.yaml")),
        )

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _make_variant(self, name="test-v1", base="qwen3:32b", variant="qwen3:32b-ft"):
        return FineTuneVariant(
            name=name, base_model=base, variant_model=variant,
            dataset_size=1000, epochs=3, learning_rate=1e-5,
        )

    def test_register_and_get(self):
        """Register a variant and retrieve it."""
        v = self._make_variant()
        registered = self.tracker.register_variant(v)
        self.assertEqual(registered.name, "test-v1")
        fetched = self.tracker.get_variant(registered.variant_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.base_model, "qwen3:32b")

    def test_register_duplicate_model_raises(self):
        """Registering same variant_model twice raises ValueError."""
        v1 = self._make_variant()
        self.tracker.register_variant(v1)
        v2 = self._make_variant(name="another")
        with self.assertRaises(ValueError):
            self.tracker.register_variant(v2)

    def test_register_missing_name_raises(self):
        """Variant without name raises ValueError."""
        v = FineTuneVariant(base_model="m", variant_model="m-ft")
        with self.assertRaises(ValueError):
            self.tracker.register_variant(v)

    def test_list_variants(self):
        """list_variants returns registered variants."""
        self.tracker.register_variant(self._make_variant("v1", "base1", "ft1"))
        self.tracker.register_variant(self._make_variant("v2", "base2", "ft2"))
        variants = self.tracker.list_variants()
        self.assertEqual(len(variants), 2)

    def test_list_filter_base_model(self):
        """list_variants filters by base_model."""
        self.tracker.register_variant(self._make_variant("v1", "base1", "ft1"))
        self.tracker.register_variant(self._make_variant("v2", "base2", "ft2"))
        variants = self.tracker.list_variants(base_model="base1")
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].base_model, "base1")

    def test_update_variant(self):
        """update_variant modifies allowed fields."""
        v = self._make_variant()
        reg = self.tracker.register_variant(v)
        updated = self.tracker.update_variant(reg.variant_id, {"epochs": 10, "status": "inactive"})
        self.assertIsNotNone(updated)
        self.assertEqual(updated.epochs, 10)
        self.assertEqual(updated.status, "inactive")

    def test_update_nonexistent_returns_none(self):
        """update_variant for unknown ID returns None."""
        result = self.tracker.update_variant("nope", {"epochs": 5})
        self.assertIsNone(result)

    def test_unregister_variant(self):
        """unregister_variant removes the variant."""
        v = self._make_variant()
        reg = self.tracker.register_variant(v)
        self.assertTrue(self.tracker.unregister_variant(reg.variant_id))
        self.assertIsNone(self.tracker.get_variant(reg.variant_id))

    def test_unregister_nonexistent(self):
        """Unregistering unknown ID returns False."""
        self.assertFalse(self.tracker.unregister_variant("nope"))

    def test_variant_to_dict(self):
        """FineTuneVariant.to_dict includes all fields."""
        v = self._make_variant()
        d = v.to_dict()
        self.assertIn("variant_id", d)
        self.assertIn("base_model", d)
        self.assertIn("dataset_size", d)

    def test_variant_from_dict(self):
        """FineTuneVariant.from_dict ignores unknown keys."""
        d = {"name": "x", "base_model": "b", "variant_model": "v", "unknown_field": 42}
        v = FineTuneVariant.from_dict(d)
        self.assertEqual(v.name, "x")


# ===========================================================================
# PART 2b: FineTuneTracker -- A/B Comparison
# ===========================================================================

class TestFineTuneTrackerComparison(unittest.TestCase):
    """Test A/B comparison creation, execution, and stats."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.tracker = FineTuneTracker(
            db_path=Path(self.tmp.name),
            config_path=Path(os.path.join(CONFIG_DIR, "fine_tune.yaml")),
        )
        v = FineTuneVariant(
            name="test-ft", base_model="base:7b", variant_model="ft:7b",
        )
        self.variant = self.tracker.register_variant(v)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_create_comparison(self):
        """create_comparison stores prompts and returns pending status."""
        comp = self.tracker.create_comparison(
            self.variant.variant_id, ["Hello", "Test"]
        )
        self.assertEqual(comp.status, "pending")
        self.assertEqual(len(comp.prompts), 2)
        self.assertEqual(comp.base_model, "base:7b")

    def test_create_comparison_unknown_variant(self):
        """create_comparison raises for unknown variant."""
        with self.assertRaises(ValueError):
            self.tracker.create_comparison("nope", ["Hello"])

    def test_create_comparison_empty_prompts(self):
        """create_comparison raises for empty prompts."""
        with self.assertRaises(ValueError):
            self.tracker.create_comparison(self.variant.variant_id, [])

    def test_run_comparison_no_inference(self):
        """run_comparison without inference_fn marks as failed."""
        comp = self.tracker.create_comparison(
            self.variant.variant_id, ["Test"]
        )
        result = self.tracker.run_comparison(comp.comparison_id, inference_fn=None)
        self.assertEqual(result.status, "failed")

    def test_run_comparison_with_mock_inference(self):
        """run_comparison with mock inference completes successfully."""
        comp = self.tracker.create_comparison(
            self.variant.variant_id, ["What is AI?"]
        )

        def mock_infer(model, prompt):
            if "ft" in model:
                return "Fine-tuned answer with more detail and explanation"
            return "Base answer"

        result = self.tracker.run_comparison(comp.comparison_id, inference_fn=mock_infer)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.base_wins + result.variant_wins + result.ties, 1)
        self.assertTrue(result.summary)
        self.assertTrue(result.completed_at)

    def test_get_comparison(self):
        """get_comparison retrieves persisted result."""
        comp = self.tracker.create_comparison(
            self.variant.variant_id, ["Hi"]
        )
        fetched = self.tracker.get_comparison(comp.comparison_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.variant_id, self.variant.variant_id)

    def test_list_comparisons(self):
        """list_comparisons returns all comparisons for a variant."""
        self.tracker.create_comparison(self.variant.variant_id, ["A"])
        self.tracker.create_comparison(self.variant.variant_id, ["B"])
        comps = self.tracker.list_comparisons(variant_id=self.variant.variant_id)
        self.assertEqual(len(comps), 2)

    def test_get_variant_stats_empty(self):
        """get_variant_stats returns zeros when no completed comparisons."""
        stats = self.tracker.get_variant_stats(self.variant.variant_id)
        self.assertEqual(stats["comparison_count"], 0)
        self.assertEqual(stats["total_prompts"], 0)

    def test_cascade_delete(self):
        """Unregistering variant cascades to delete comparisons."""
        self.tracker.create_comparison(self.variant.variant_id, ["Test"])
        self.tracker.unregister_variant(self.variant.variant_id)
        comps = self.tracker.list_comparisons(variant_id=self.variant.variant_id)
        self.assertEqual(len(comps), 0)

    def test_comparison_result_to_dict(self):
        """ComparisonResult.to_dict serializes prompts."""
        comp = self.tracker.create_comparison(
            self.variant.variant_id, ["Hi"]
        )
        d = comp.to_dict()
        self.assertIn("prompts", d)
        self.assertIsInstance(d["prompts"], list)


# ===========================================================================
# PART 3: ExportFilter and DataClasses
# ===========================================================================

class TestDataClasses(unittest.TestCase):
    """Test dataclass serialization and construction."""

    def test_export_filter_to_dict(self):
        """ExportFilter serializes all fields."""
        f = ExportFilter(model="qwen3:32b", min_quality=0.7)
        d = f.to_dict()
        self.assertEqual(d["model"], "qwen3:32b")
        self.assertEqual(d["min_quality"], 0.7)

    def test_export_result_to_dict(self):
        """ExportResult.to_dict excludes raw data, includes size."""
        r = ExportResult(format="sharegpt", data='[{"test": true}]')
        d = r.to_dict()
        self.assertNotIn("data", d)
        self.assertIn("data_size_bytes", d)
        self.assertGreater(d["data_size_bytes"], 0)

    def test_comparison_prompt_to_dict(self):
        """ComparisonPrompt serializes correctly."""
        cp = ComparisonPrompt(prompt="Hi", winner="tie")
        d = cp.to_dict()
        self.assertEqual(d["prompt"], "Hi")
        self.assertEqual(d["winner"], "tie")


# ===========================================================================
# PART 4: API Routes File
# ===========================================================================

class TestRoutesFineTune(unittest.TestCase):
    """Test routes_fine_tune.py file structure and endpoints."""

    def setUp(self):
        self.source = _read(os.path.join(API_DIR, "routes_fine_tune.py"))

    def test_router_prefix(self):
        """Router uses /api/fine-tune prefix."""
        self.assertIn('prefix="/api/fine-tune"', self.source)

    def test_export_endpoint(self):
        """POST /export endpoint exists."""
        self.assertIn('@router.post("/export")', self.source)

    def test_preview_endpoint(self):
        """GET /export/preview endpoint exists."""
        self.assertIn('@router.get("/export/preview")', self.source)

    def test_quality_endpoint(self):
        """GET /quality endpoint exists."""
        self.assertIn('@router.get("/quality")', self.source)

    def test_variants_list_endpoint(self):
        """GET /variants endpoint exists."""
        self.assertIn('@router.get("/variants")', self.source)

    def test_variants_create_endpoint(self):
        """POST /variants endpoint with 201 status."""
        self.assertIn('@router.post("/variants", status_code=201)', self.source)

    def test_variants_delete_endpoint(self):
        """DELETE /variants/{variant_id} endpoint exists."""
        self.assertIn('@router.delete("/variants/{variant_id}")', self.source)

    def test_compare_endpoint(self):
        """POST /compare endpoint exists."""
        self.assertIn('@router.post("/compare")', self.source)

    def test_compare_get_endpoint(self):
        """GET /compare/{comparison_id} endpoint exists."""
        self.assertIn('@router.get("/compare/{comparison_id}")', self.source)

    def test_no_hardcoded_hex(self):
        """No hardcoded hex color values in routes."""
        hex_matches = re.findall(r'#[0-9a-fA-F]{3,6}\b', self.source)
        self.assertEqual(len(hex_matches), 0)


# ===========================================================================
# PART 5: Frontend Files
# ===========================================================================

class TestFrontendTypes(unittest.TestCase):
    """Test TypeScript type definitions for fine-tune."""

    def setUp(self):
        self.source = _read(os.path.join(FRONTEND_SRC, "lib", "types.ts"))

    def test_export_request_type(self):
        self.assertIn("FineTuneExportRequest", self.source)

    def test_export_response_type(self):
        self.assertIn("FineTuneExportResponse", self.source)

    def test_variant_type(self):
        self.assertIn("FineTuneVariant", self.source)

    def test_compare_response_type(self):
        self.assertIn("FineTuneCompareResponse", self.source)

    def test_quality_score_type(self):
        self.assertIn("FineTuneQualityScore", self.source)


class TestFrontendApiClient(unittest.TestCase):
    """Test fineTune.ts API client."""

    def setUp(self):
        self.source = _read(os.path.join(API_TS_DIR, "fineTune.ts"))

    def test_export_function(self):
        self.assertIn("exportTrainingData", self.source)

    def test_preview_function(self):
        self.assertIn("previewExport", self.source)

    def test_quality_function(self):
        self.assertIn("getQualityScores", self.source)

    def test_list_variants_function(self):
        self.assertIn("listVariants", self.source)

    def test_register_variant_function(self):
        self.assertIn("registerVariant", self.source)

    def test_unregister_variant_function(self):
        self.assertIn("unregisterVariant", self.source)

    def test_run_comparison_function(self):
        self.assertIn("runComparison", self.source)

    def test_get_comparison_function(self):
        self.assertIn("getComparison", self.source)

    def test_imports_types(self):
        self.assertIn("FineTuneExportRequest", self.source)
        self.assertIn("FineTuneVariant", self.source)


class TestFrontendPanel(unittest.TestCase):
    """Test FineTunePanel.svelte component."""

    def setUp(self):
        self.source = _read(os.path.join(SETTINGS_DIR, "FineTunePanel.svelte"))

    def test_component_exists(self):
        self.assertIn("<script", self.source)

    def test_three_sub_tabs(self):
        """Panel has export, variants, and quality sub-tabs."""
        self.assertIn("'export'", self.source)
        self.assertIn("'variants'", self.source)
        self.assertIn("'quality'", self.source)

    def test_format_selector(self):
        """Export tab has format selector with all three formats."""
        self.assertIn("sharegpt", self.source)
        self.assertIn("alpaca", self.source)
        self.assertIn("jsonl", self.source)

    def test_no_hardcoded_hex(self):
        """No hardcoded hex color values in Svelte."""
        # Filter out Svelte template syntax like {#each, {#if
        lines = self.source.split("\n")
        for i, line in enumerate(lines, 1):
            if re.search(r'#[0-9a-fA-F]{3,8}\b', line):
                if not re.search(r'{#(each|if|key|await)', line):
                    self.fail(f"Hardcoded hex at line {i}: {line.strip()}")

    def test_uses_css_vars(self):
        """Component uses CSS custom properties."""
        self.assertIn("var(--oo-", self.source)

    def test_imports_api_functions(self):
        """Component imports from fineTune API client."""
        self.assertIn("from '$lib/api/fineTune'", self.source)

    def test_register_form(self):
        """Component has variant registration form."""
        self.assertIn("Register New Variant", self.source)

    def test_comparison_section(self):
        """Component has A/B comparison UI."""
        self.assertIn("A/B Comparison", self.source)

    def test_download_button(self):
        """Export tab has download functionality."""
        self.assertIn("downloadExport", self.source)


class TestSettingsPage(unittest.TestCase):
    """Test settings page integration of FineTunePanel."""

    def setUp(self):
        self.source = _read(os.path.join(ROUTES_DIR, "settings", "+page.svelte"))

    def test_imports_panel(self):
        self.assertIn("FineTunePanel", self.source)

    def test_tab_type_includes_fine_tune(self):
        self.assertIn("'fine-tune'", self.source)

    def test_tab_entry_exists(self):
        self.assertIn("Fine-Tune", self.source)

    def test_panel_rendered(self):
        self.assertIn("<FineTunePanel", self.source)


# ===========================================================================
# PART 6: Config & Integration Wiring
# ===========================================================================

class TestConfig(unittest.TestCase):
    """Test fine_tune.yaml configuration."""

    def setUp(self):
        with open(os.path.join(CONFIG_DIR, "fine_tune.yaml"), "r") as f:
            self.config = yaml.safe_load(f)

    def test_export_section(self):
        self.assertIn("export", self.config)
        self.assertIn("default_format", self.config["export"])

    def test_quality_section(self):
        self.assertIn("quality", self.config)
        self.assertIn("feedback_weight", self.config["quality"])
        self.assertIn("benchmark_weight", self.config["quality"])

    def test_tracking_section(self):
        self.assertIn("tracking", self.config)
        self.assertIn("enabled", self.config["tracking"])

    def test_weights_sum_to_one(self):
        w = self.config["quality"]["feedback_weight"] + self.config["quality"]["benchmark_weight"]
        self.assertAlmostEqual(w, 1.0)


class TestDepsWiring(unittest.TestCase):
    """Test deps.py includes fine-tune singletons."""

    def setUp(self):
        self.source = _read(os.path.join(API_DIR, "deps.py"))

    def test_export_import(self):
        self.assertIn("fine_tune_exporter", self.source)
        self.assertIn("FINE_TUNE_EXPORT_AVAILABLE", self.source)

    def test_tracker_import(self):
        self.assertIn("fine_tune_tracker", self.source)
        self.assertIn("FINE_TUNE_TRACKER_AVAILABLE", self.source)


class TestAppWiring(unittest.TestCase):
    """Test app.py includes fine-tune router and version bump."""

    def setUp(self):
        self.source = _read(os.path.join(API_DIR, "app.py"))

    def test_router_import(self):
        self.assertIn("routes_fine_tune", self.source)
        self.assertIn("fine_tune_router", self.source)

    def test_router_registered(self):
        self.assertIn("app.include_router(fine_tune_router)", self.source)

    def test_version_bump(self):
        """Version bumped to 1.9.8."""
        self.assertIn('"1.10.0"', self.source)
        self.assertNotIn('"1.9.7"', self.source)

    def test_health_modules_include_fine_tune(self):
        self.assertIn("fine_tune_export", self.source)
        self.assertIn("fine_tune_tracker", self.source)


# ===========================================================================
# PART 7: Code Quality
# ===========================================================================

class TestCodeQuality(unittest.TestCase):
    """Verify code quality constraints."""

    def test_no_emoji_in_backend(self):
        """No emoji in Python backend files."""
        import re
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
            flags=re.UNICODE,
        )
        for fname in ["fine_tune_export.py", "fine_tune_tracker.py"]:
            content = _read(os.path.join(BACKEND_DIR, fname))
            matches = emoji_pattern.findall(content)
            self.assertEqual(
                len(matches), 0, f"Emoji found in {fname}: {matches}"
            )

    def test_no_french_in_new_code(self):
        """No French comments in new S96 modules."""
        french_markers = ["Recupere", "Parametr", "Gestionnaire", "supprime"]
        for fname in ["fine_tune_export.py", "fine_tune_tracker.py"]:
            content = _read(os.path.join(BACKEND_DIR, fname))
            for marker in french_markers:
                self.assertNotIn(
                    marker, content,
                    f"French text '{marker}' found in {fname}"
                )

    def test_routes_english_only(self):
        """Routes file uses English only."""
        content = _read(os.path.join(API_DIR, "routes_fine_tune.py"))
        self.assertNotIn("Recupere", content)
        self.assertNotIn("parametr", content)


if __name__ == "__main__":
    unittest.main()
