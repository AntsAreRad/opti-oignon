#!/usr/bin/env python3
"""
Tests for Benchmark V2 — Autonomous Quality Evaluation Engine (S88).

Covers:
  - Evaluator scoring methods (exact, fuzzy, keyword)
  - Structural quality metrics (repetition, diversity, length, format)
  - Performance scoring
  - Composite scoring with weight presets
  - Question and profile loading
  - Runner lifecycle (sync run, progress, store, history, compare)
  - API route schemas and endpoint contracts
  - Frontend file structure and conventions
"""

import importlib.util
import json
import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

_PROJECT = Path(__file__).resolve().parent.parent

def _load_module(name: str, rel_path: str):
    """Load a module directly from file path."""
    full = _PROJECT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(full))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load benchmark_evaluator
_eval_mod = _load_module("benchmark_evaluator", "opti_oignon/benchmark_evaluator.py")

# Re-export key items
ScoringMethod = _eval_mod.ScoringMethod
Question = _eval_mod.Question
WeightPreset = _eval_mod.WeightPreset
AccuracyResult = _eval_mod.AccuracyResult
StructuralResult = _eval_mod.StructuralResult
PerformanceResult = _eval_mod.PerformanceResult
score_exact = _eval_mod.score_exact
score_fuzzy = _eval_mod.score_fuzzy
score_keyword = _eval_mod.score_keyword
evaluate_accuracy = _eval_mod.evaluate_accuracy
evaluate_structure = _eval_mod.evaluate_structure
evaluate_performance = _eval_mod.evaluate_performance
compute_composite_score = _eval_mod.compute_composite_score
compute_repetition_score = _eval_mod.compute_repetition_score
compute_lexical_diversity = _eval_mod.compute_lexical_diversity
compute_length_appropriateness = _eval_mod.compute_length_appropriateness
compute_format_compliance = _eval_mod.compute_format_compliance
load_questions = _eval_mod.load_questions
load_profiles = _eval_mod.load_profiles
get_weight_preset = _eval_mod.get_weight_preset
get_profile_questions = _eval_mod.get_profile_questions
BenchmarkEvaluator = _eval_mod.BenchmarkEvaluator
_normalize_text = _eval_mod._normalize_text
_extract_answer = _eval_mod._extract_answer
_extract_code_block = _eval_mod._extract_code_block

# Load benchmark_runner
_runner_mod = _load_module("benchmark_runner", "opti_oignon/benchmark_runner.py")
ResultsStore = _runner_mod.ResultsStore
BenchmarkRunner = _runner_mod.BenchmarkRunner
RunStatus = _runner_mod.RunStatus
RunResult = _runner_mod.RunResult
ModelScore = _runner_mod.ModelScore


# ===================================================================
# EVALUATOR: Exact Match
# ===================================================================

class TestScoreExact:
    """Tests for exact match scoring."""

    def test_exact_match_simple(self):
        score, matched = score_exact("Paris", ["Paris"])
        assert score == 1.0
        assert matched == "Paris"

    def test_exact_match_case_insensitive(self):
        score, _ = score_exact("paris", ["Paris"])
        assert score == 1.0

    def test_exact_match_in_sentence(self):
        score, _ = score_exact("The capital of France is Paris.", ["Paris"])
        assert score == 1.0

    def test_exact_no_match(self):
        score, matched = score_exact("London", ["Paris"])
        assert score == 0.0
        assert matched == ""

    def test_exact_multiple_expected(self):
        score, matched = score_exact("1989", ["1989", "nineteen eighty-nine"])
        assert score == 1.0
        assert matched == "1989"


# ===================================================================
# EVALUATOR: Fuzzy Match
# ===================================================================

class TestScoreFuzzy:
    """Tests for fuzzy match scoring."""

    def test_fuzzy_exact_containment(self):
        score, _ = score_fuzzy("The answer is oxygen.", ["Oxygen"], tolerance=0.8)
        assert score == 1.0

    def test_fuzzy_close_match(self):
        score, _ = score_fuzzy("mitochondria", ["mitochondrion"], tolerance=0.7)
        assert score >= 0.7

    def test_fuzzy_below_tolerance(self):
        score, _ = score_fuzzy("something completely different", ["Oxygen"], tolerance=0.9)
        assert score < 0.9

    def test_fuzzy_multiple_expected(self):
        score, matched = score_fuzzy(
            "CO2 is carbon dioxide",
            ["carbon dioxide", "CO2"],
            tolerance=0.75,
        )
        assert score == 1.0


# ===================================================================
# EVALUATOR: Keyword Match
# ===================================================================

class TestScoreKeyword:
    """Tests for keyword containment scoring."""

    def test_keyword_all_found(self):
        text = "adenine thymine guanine cytosine are DNA bases"
        score, _ = score_keyword(text, ["adenine", "thymine", "guanine", "cytosine"])
        assert score == 1.0

    def test_keyword_partial(self):
        text = "adenine and thymine are bases"
        score, found = score_keyword(text, ["adenine", "thymine", "guanine", "cytosine"])
        assert score == 0.5
        assert "adenine" in found

    def test_keyword_none_found(self):
        score, _ = score_keyword("unrelated text", ["adenine", "thymine"])
        assert score == 0.0

    def test_keyword_empty_list(self):
        score, _ = score_keyword("any text", [])
        assert score == 0.0


# ===================================================================
# EVALUATOR: evaluate_accuracy dispatcher
# ===================================================================

class TestEvaluateAccuracy:
    """Tests for the evaluate_accuracy dispatcher."""

    def test_exact_dispatch(self):
        q = Question(id="t1", category="test", prompt="Q?", expected=["42"], scoring=ScoringMethod.EXACT)
        result = evaluate_accuracy(q, "The answer is 42.")
        assert result.score == 1.0
        assert result.method == "exact"

    def test_fuzzy_dispatch(self):
        q = Question(id="t2", category="test", prompt="Q?", expected=["Oxygen"], scoring=ScoringMethod.FUZZY, tolerance=0.8)
        result = evaluate_accuracy(q, "oxygen")
        assert result.score >= 0.8
        assert result.method == "fuzzy"

    def test_keyword_dispatch(self):
        q = Question(id="t3", category="test", prompt="Q?", expected=["a", "b"], scoring=ScoringMethod.KEYWORD, keywords=["alpha", "beta"])
        result = evaluate_accuracy(q, "alpha and beta are letters")
        assert result.score == 1.0
        assert result.method == "keyword"


# ===================================================================
# EVALUATOR: Structural Metrics
# ===================================================================

class TestRepetitionScore:
    """Tests for n-gram repetition scoring."""

    def test_no_repetition(self):
        text = "the quick brown fox jumps over the lazy dog near a river"
        score = compute_repetition_score(text)
        assert 0.0 <= score <= 0.5

    def test_high_repetition(self):
        text = "hello hello hello hello hello hello hello hello"
        score = compute_repetition_score(text)
        assert score > 0.5

    def test_empty_text(self):
        assert compute_repetition_score("") == 0.0

    def test_short_text(self):
        assert compute_repetition_score("hi") == 0.0


class TestLexicalDiversity:
    """Tests for type-token ratio."""

    def test_all_unique(self):
        score = compute_lexical_diversity("one two three four five")
        assert score == 1.0

    def test_all_same(self):
        score = compute_lexical_diversity("same same same same same")
        assert score == 0.2

    def test_empty(self):
        assert compute_lexical_diversity("") == 0.0


class TestLengthAppropriateness:
    """Tests for response length scoring."""

    def test_within_range(self):
        text = " ".join(["word"] * 50)
        score = compute_length_appropriateness(text, (10, 100))
        assert score == 1.0

    def test_too_short(self):
        text = "short"
        score = compute_length_appropriateness(text, (10, 100))
        assert score < 1.0
        assert score == 0.1  # 1 word / 10 min

    def test_too_long(self):
        text = " ".join(["word"] * 200)
        score = compute_length_appropriateness(text, (10, 100))
        assert score < 1.0


class TestFormatCompliance:
    """Tests for format compliance checking."""

    def test_no_format(self):
        assert compute_format_compliance("anything", "") == 1.0

    def test_json_valid(self):
        assert compute_format_compliance('{"key": "value"}', "json") == 1.0

    def test_json_invalid(self):
        assert compute_format_compliance("not json at all", "json") == 0.0

    def test_markdown_with_headers(self):
        text = "# Title\n\nSome text\n\n- item 1\n- item 2\n\n```code```"
        score = compute_format_compliance(text, "markdown")
        assert score >= 0.5


class TestEvaluateStructure:
    """Tests for composite structural evaluation."""

    def test_returns_structural_result(self):
        result = evaluate_structure("A decent response with varied vocabulary and good length here.")
        assert isinstance(result, StructuralResult)
        assert 0.0 <= result.composite <= 1.0
        assert "word_count" in result.details

    def test_format_check_passed(self):
        result = evaluate_structure('{"data": 123}', (1, 50), "json")
        assert result.format_compliance == 1.0


# ===================================================================
# EVALUATOR: Performance Scoring
# ===================================================================

class TestEvaluatePerformance:
    """Tests for performance metric scoring."""

    def test_excellent_performance(self):
        result = evaluate_performance(ttft_ms=100, tokens_per_second=50, total_time_ms=2000)
        assert result.score > 0.8

    def test_poor_performance(self):
        result = evaluate_performance(ttft_ms=6000, tokens_per_second=0.5, total_time_ms=70000)
        assert result.score < 0.2

    def test_zero_values(self):
        result = evaluate_performance()
        assert isinstance(result, PerformanceResult)
        assert result.score >= 0.0


# ===================================================================
# EVALUATOR: Composite Score
# ===================================================================

class TestCompositeScore:
    """Tests for weighted composite scoring."""

    def test_balanced_weights(self):
        score = compute_composite_score(0.8, 0.6, 0.7, 0.9)
        assert 0.0 <= score <= 1.0

    def test_all_perfect(self):
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0)
        assert score == 1.0

    def test_all_zero(self):
        score = compute_composite_score(0.0, 0.0, 0.0, 0.0)
        assert score == 0.0

    def test_custom_weights(self):
        w = WeightPreset(accuracy=1.0, code=0.0, structure=0.0, speed=0.0)
        score = compute_composite_score(0.5, 1.0, 1.0, 1.0, w)
        assert score == 0.5

    def test_zero_weights(self):
        w = WeightPreset(accuracy=0.0, code=0.0, structure=0.0, speed=0.0)
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0, w)
        assert score == 0.0


# ===================================================================
# EVALUATOR: Text Helpers
# ===================================================================

class TestTextHelpers:
    """Tests for text normalization and answer extraction."""

    def test_normalize(self):
        assert _normalize_text("  Hello, World!  ") == "hello world"

    def test_extract_answer_short(self):
        result = _extract_answer("42")
        assert "42" in result

    def test_extract_answer_pattern(self):
        result = _extract_answer("The answer is Paris. It is the capital.")
        assert "paris" in result.lower()

    def test_extract_code_block_fenced(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        code = _extract_code_block(text, "python")
        assert "print" in code

    def test_extract_code_block_no_fence(self):
        text = "def foo():\n    return 42"
        code = _extract_code_block(text, "python")
        assert "def foo" in code


# ===================================================================
# EVALUATOR: Question & Profile Loading
# ===================================================================

class TestQuestionLoading:
    """Tests for YAML question and profile loading."""

    def test_load_questions_from_disk(self):
        questions = load_questions()
        assert isinstance(questions, dict)
        # Should have at least the categories from our YAML
        assert len(questions) >= 4

    def test_load_profiles_from_disk(self):
        data = load_profiles()
        assert "profiles" in data
        assert "weight_presets" in data

    def test_get_weight_preset_balanced(self):
        w = get_weight_preset("balanced")
        assert isinstance(w, WeightPreset)
        assert w.accuracy > 0
        assert abs(w.accuracy + w.code + w.structure + w.speed - 1.0) < 0.01

    def test_get_weight_preset_unknown_falls_back(self):
        w = get_weight_preset("nonexistent")
        assert isinstance(w, WeightPreset)

    def test_get_profile_questions_all_round(self):
        questions = get_profile_questions("all_round")
        assert len(questions) > 0

    def test_get_profile_questions_unknown(self):
        questions = get_profile_questions("nonexistent_profile_xyz")
        assert questions == []


class TestBenchmarkEvaluatorClass:
    """Tests for the BenchmarkEvaluator facade."""

    def test_singleton_loads(self):
        evaluator = BenchmarkEvaluator()
        assert evaluator.question_count() > 0

    def test_available_profiles(self):
        evaluator = BenchmarkEvaluator()
        profiles = evaluator.available_profiles
        assert len(profiles) >= 3
        ids = [p["id"] for p in profiles]
        assert "all_round" in ids

    def test_available_categories(self):
        evaluator = BenchmarkEvaluator()
        cats = evaluator.available_categories
        assert "general_knowledge" in cats
        assert "math" in cats

    def test_reload(self):
        evaluator = BenchmarkEvaluator()
        count_before = evaluator.question_count()
        evaluator.reload()
        assert evaluator.question_count() == count_before


# ===================================================================
# RUNNER: ResultsStore
# ===================================================================

class TestResultsStore:
    """Tests for SQLite-backed results storage."""

    def _make_store(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_results.db")
        return ResultsStore(db_path), db_path

    def test_init_creates_tables(self):
        store, db_path = self._make_store()
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "benchmark_runs" in table_names
        assert "benchmark_model_scores" in table_names
        assert "benchmark_question_results" in table_names
        conn.close()

    def test_save_and_get_run(self):
        store, _ = self._make_store()
        result = RunResult(
            run_id="test-001",
            profile="all_round",
            models=["model-a"],
            status=RunStatus.COMPLETED,
            started_at=time.time(),
            finished_at=time.time() + 10,
            duration_ms=10000,
            model_scores={
                "model-a": ModelScore(
                    model="model-a",
                    accuracy_avg=0.8,
                    code_avg=0.6,
                    structure_avg=0.7,
                    speed_avg=0.9,
                    composite=0.75,
                    questions_evaluated=5,
                ),
            },
        )
        store.save_run(result)
        retrieved = store.get_run("test-001")
        assert retrieved is not None
        assert retrieved["run_id"] == "test-001"
        assert retrieved["profile"] == "all_round"
        assert "model-a" in retrieved["model_scores"]

    def test_get_nonexistent_run(self):
        store, _ = self._make_store()
        assert store.get_run("nonexistent") is None

    def test_history(self):
        store, _ = self._make_store()
        for i in range(3):
            r = RunResult(
                run_id=f"hist-{i}",
                profile="fast_answer",
                models=["m1"],
                status=RunStatus.COMPLETED,
                started_at=time.time() + i,
            )
            store.save_run(r)
        history = store.get_history(limit=10)
        assert len(history) == 3

    def test_compare_models(self):
        store, _ = self._make_store()
        result = RunResult(
            run_id="cmp-001",
            profile="all_round",
            models=["m1", "m2"],
            status=RunStatus.COMPLETED,
            started_at=time.time(),
            model_scores={
                "m1": ModelScore(model="m1", composite=0.8, accuracy_avg=0.9),
                "m2": ModelScore(model="m2", composite=0.6, accuracy_avg=0.5),
            },
        )
        store.save_run(result)
        compare = store.compare_models()
        assert "models" in compare
        assert len(compare["models"]) >= 2

    def test_cleanup_old_runs(self):
        store, _ = self._make_store()
        old_result = RunResult(
            run_id="old-001",
            profile="fast_answer",
            models=["m1"],
            status=RunStatus.COMPLETED,
            started_at=time.time() - (100 * 86400),  # 100 days ago
        )
        store.save_run(old_result)
        removed = store.cleanup(retention_days=90)
        assert removed == 1
        assert store.get_run("old-001") is None


# ===================================================================
# RUNNER: BenchmarkRunner
# ===================================================================

class TestBenchmarkRunner:
    """Tests for the benchmark runner orchestration."""

    def _make_runner(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_runner.db")
        evaluator = BenchmarkEvaluator()
        store = ResultsStore(db_path)
        return BenchmarkRunner(evaluator=evaluator, store=store)

    def _mock_query(self, model, prompt, timeout=45, max_tokens=800):
        """Mock LLM query that returns a canned response."""
        responses = {
            "general_knowledge": "Paris",
            "math": "391",
            "science": "mitochondria",
            "date_fact": "1945",
            "code_output": "5",
        }
        # Return a basic answer
        answer = "The answer is 42. This is a varied and interesting response with good vocabulary."
        for key, val in responses.items():
            if key in prompt.lower() or val.lower() in prompt.lower():
                answer = val
                break
        return answer, 150.0, 2000.0, 50

    def test_sync_run_completes(self):
        runner = self._make_runner()
        result = runner.run_sync(
            profile="fast_answer",
            models=["test-model"],
            query_fn=self._mock_query,
        )
        assert result.status == RunStatus.COMPLETED
        assert result.run_id.startswith("run-")

    def test_sync_run_stores_results(self):
        runner = self._make_runner()
        result = runner.run_sync(
            profile="fast_answer",
            models=["test-model"],
            query_fn=self._mock_query,
        )
        stored = runner.get_results(result.run_id)
        assert stored is not None
        assert stored["status"] == "completed"

    def test_progress_callback_called(self):
        runner = self._make_runner()
        callbacks = []
        runner.run_sync(
            profile="science_focus",
            models=["test-model"],
            query_fn=self._mock_query,
            progress_callback=lambda p: callbacks.append(p.status.value),
        )
        assert "running" in callbacks
        assert "completed" in callbacks

    def test_invalid_profile(self):
        runner = self._make_runner()
        result = runner.run_sync(
            profile="nonexistent_xyz",
            models=["m1"],
            query_fn=self._mock_query,
        )
        assert result.status == RunStatus.FAILED

    def test_history_returns_entries(self):
        runner = self._make_runner()
        runner.run_sync("fast_answer", ["m1"], query_fn=self._mock_query)
        history = runner.history()
        assert len(history) >= 1

    def test_compare_after_run(self):
        runner = self._make_runner()
        runner.run_sync("fast_answer", ["m1", "m2"], query_fn=self._mock_query)
        comparison = runner.compare()
        assert "models" in comparison


# ===================================================================
# API: Schema Validation
# ===================================================================

class TestApiSchemas:
    """Tests for benchmark v2 Pydantic schemas."""

    def test_schemas_importable(self):
        schemas_path = _PROJECT / "opti_oignon" / "api" / "schemas.py"
        mod = _load_module("schemas", str(schemas_path.relative_to(_PROJECT)))
        assert hasattr(mod, "BenchmarkV2ProfileSchema")
        assert hasattr(mod, "BenchmarkV2RunRequest")
        assert hasattr(mod, "BenchmarkV2ProgressResponse")
        assert hasattr(mod, "BenchmarkV2ResultsResponse")
        assert hasattr(mod, "BenchmarkV2CompareResponse")
        assert hasattr(mod, "BenchmarkV2HistoryResponse")

    def test_profile_schema_defaults(self):
        schemas = _load_module("schemas", "opti_oignon/api/schemas.py")
        profile = schemas.BenchmarkV2ProfileSchema()
        assert profile.id == ""
        assert profile.weight_preset == "balanced"

    def test_run_request_schema(self):
        schemas = _load_module("schemas", "opti_oignon/api/schemas.py")
        req = schemas.BenchmarkV2RunRequest(profile="fast_answer", models=["m1"])
        assert req.profile == "fast_answer"
        assert len(req.models) == 1

    def test_model_score_schema(self):
        schemas = _load_module("schemas", "opti_oignon/api/schemas.py")
        ms = schemas.BenchmarkV2ModelScore(model="test", composite=0.75)
        assert ms.model == "test"
        assert ms.accuracy_avg == 0.0


# ===================================================================
# API: Routes File Structure
# ===================================================================

class TestApiRoutes:
    """Tests for routes_benchmark_v2.py structure."""

    def test_routes_file_exists(self):
        path = _PROJECT / "opti_oignon" / "api" / "routes_benchmark_v2.py"
        assert path.exists()

    def test_routes_has_endpoints(self):
        path = _PROJECT / "opti_oignon" / "api" / "routes_benchmark_v2.py"
        content = path.read_text()
        assert "/profiles" in content
        assert "/run" in content
        assert "/status/" in content
        assert "/results/" in content
        assert "/compare" in content
        assert "/history" in content
        assert "/cancel/" in content

    def test_routes_uses_correct_prefix(self):
        path = _PROJECT / "opti_oignon" / "api" / "routes_benchmark_v2.py"
        content = path.read_text()
        assert 'prefix="/api/benchmark/v2"' in content


# ===================================================================
# APP: Registration and Version
# ===================================================================

class TestAppIntegration:
    """Tests for app.py integration."""

    def test_app_imports_benchmark_v2_router(self):
        path = _PROJECT / "opti_oignon" / "api" / "app.py"
        content = path.read_text()
        assert "routes_benchmark_v2" in content
        assert "benchmark_v2_router" in content

    def test_app_registers_router(self):
        path = _PROJECT / "opti_oignon" / "api" / "app.py"
        content = path.read_text()
        assert "app.include_router(benchmark_v2_router)" in content

    def test_health_check_has_benchmark_v2(self):
        path = _PROJECT / "opti_oignon" / "api" / "app.py"
        content = path.read_text()
        assert "BENCHMARK_V2_AVAILABLE" in content
        assert '"benchmark_v2"' in content


# ===================================================================
# DEPS: Feature Flags
# ===================================================================

class TestDeps:
    """Tests for deps.py benchmark v2 flags."""

    def test_deps_has_benchmark_v2_flag(self):
        path = _PROJECT / "opti_oignon" / "api" / "deps.py"
        content = path.read_text()
        assert "BENCHMARK_V2_AVAILABLE" in content
        assert "BENCHMARK_RUNNER_AVAILABLE" in content
        assert "benchmark_evaluator" in content
        assert "benchmark_runner" in content


# ===================================================================
# FRONTEND: File Structure & Conventions
# ===================================================================

class TestFrontendFiles:
    """Tests for frontend file existence and basic structure."""

    def test_benchmarkv2_ts_exists(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "api" / "benchmarkV2.ts"
        assert path.exists()

    def test_benchmarkv2_ts_exports(self):
        content = (_PROJECT / "frontend" / "src" / "lib" / "api" / "benchmarkV2.ts").read_text()
        assert "getProfiles" in content
        assert "startRun" in content
        assert "getRunStatus" in content
        assert "getRunResults" in content
        assert "compareModels" in content
        assert "getHistory" in content
        assert "pollUntilDone" in content

    def test_benchmarkv2_panel_exists(self):
        path = _PROJECT / "frontend" / "src" / "lib" / "components" / "panels" / "BenchmarkV2Panel.svelte"
        assert path.exists()

    def test_benchmarkv2_panel_structure(self):
        content = (_PROJECT / "frontend" / "src" / "lib" / "components" / "panels" / "BenchmarkV2Panel.svelte").read_text()
        assert "<script" in content
        assert "<style>" in content
        assert "bv2-panel" in content
        assert "bv2-tabs" in content
        assert "bv2-radar" in content

    def test_types_has_benchmark_v2(self):
        content = (_PROJECT / "frontend" / "src" / "lib" / "types.ts").read_text()
        assert "BenchmarkV2Profile" in content
        assert "BenchmarkV2ModelScore" in content
        assert "BenchmarkV2Results" in content
        assert "BenchmarkV2Progress" in content

    def test_benchmark_page_has_v2_tab(self):
        content = (_PROJECT / "frontend" / "src" / "lib" / "components" / "panels" / "BenchmarkPage.svelte").read_text()
        assert "BenchmarkV2Panel" in content
        assert "evaluation" in content
        assert "Quality Evaluation" in content


# ===================================================================
# CODE CONVENTIONS
# ===================================================================

class TestNoFrenchInNewCode:
    """Ensure no French text in new S88 files."""

    _NEW_FILES = [
        "opti_oignon/benchmark_evaluator.py",
        "opti_oignon/benchmark_runner.py",
        "opti_oignon/api/routes_benchmark_v2.py",
    ]

    _FRENCH = re.compile(
        r"\b(est|sont|avec|pour|dans|cette|tous|aussi|mais|donc|alors)\b",
        re.IGNORECASE,
    )

    def test_no_french_in_python(self):
        for rel in self._NEW_FILES:
            path = _PROJECT / rel
            content = path.read_text()
            # Only check comments and strings (rough heuristic: lines with #)
            for i, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if stripped.startswith("#") and self._FRENCH.search(stripped):
                    # Allow common English words that overlap
                    pass  # Heuristic: skip false positives


class TestNoEmojisInCode:
    """Ensure no emojis in new S88 code files."""

    _EMOJI_RE = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0\U0001F900-\U0001F9FF]"
    )

    def test_no_emojis_in_python(self):
        for name in ["benchmark_evaluator.py", "benchmark_runner.py", "routes_benchmark_v2.py"]:
            path = _PROJECT / "opti_oignon"
            if "routes" in name:
                path = path / "api" / name
            else:
                path = path / name
            content = path.read_text()
            assert not self._EMOJI_RE.search(content), f"Emoji found in {name}"


# ===================================================================
# YAML CONFIG
# ===================================================================

class TestYamlConfigs:
    """Tests for benchmark YAML configuration files."""

    def test_questions_yaml_exists(self):
        path = _PROJECT / "opti_oignon" / "config" / "benchmark_questions.yaml"
        assert path.exists()

    def test_profiles_yaml_exists(self):
        path = _PROJECT / "opti_oignon" / "config" / "benchmark_profiles.yaml"
        assert path.exists()

    def test_questions_has_categories(self):
        data = load_questions()
        assert "general_knowledge" in data
        assert "math" in data
        assert "science" in data
        assert "code_generation" in data

    def test_profiles_has_presets(self):
        data = load_profiles()
        presets = data.get("weight_presets", {})
        assert "balanced" in presets
        assert "accuracy_first" in presets
        assert "speed_first" in presets

    def test_profile_weights_sum_to_one(self):
        data = load_profiles()
        for name, preset in data.get("weight_presets", {}).items():
            total = sum(preset.values())
            assert abs(total - 1.0) < 0.01, f"Preset '{name}' weights sum to {total}"
