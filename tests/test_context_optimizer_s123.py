#!/usr/bin/env python3
"""
Tests for the Context Optimizer module — Opti-Oignon S123.

Covers:
- ContextOptimizer: optimize(), zone budget enforcement, RAG passthrough,
  fallback chain, emergency truncation (~15 tests)
- Token estimation: per-family calibration, code detection, batch helper (~8 tests)
- Priority overrides: presets, custom ratios, budget recalculation (~6 tests)
- API endpoints: config GET/PUT, report, presets, budget with presets (~6 tests)
- Executor integration: optimizer enabled vs disabled, report storage (~5 tests)
- Frontend: panel structure, imports, CSS compliance, HTML balance (~8 tests)
- Deprecation marker on S16 TokenBudgetManager (~1 test)

Uses importlib.util for direct module loading to avoid __init__.py
triggering hard ollama imports in this test environment.
"""

import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ============================================================================
# Module loading via importlib (bypasses __init__.py)
# ============================================================================

def _load_module(name: str, path: str):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_cm = _load_module("context_manager_mod", "opti_oignon/context_manager.py")
_co = _load_module("context_optimizer_mod", "opti_oignon/context_optimizer.py")

ContextOptimizer = _co.ContextOptimizer
OptimizedContext = _co.OptimizedContext
OptimizationReport = _co.OptimizationReport
ZoneReport = _co.ZoneReport
_load_config = _co._load_config
_estimate_tokens = _co._estimate_tokens
_estimate_messages_tokens = _co._estimate_messages_tokens
get_optimizer = _co.get_optimizer
init_optimizer = _co.init_optimizer

detect_model_family = _cm.detect_model_family
get_family_chars_per_token = _cm.get_family_chars_per_token
estimate_tokens_calibrated = _cm.estimate_tokens_calibrated
estimate_messages_tokens = _cm.estimate_messages_tokens
_has_code_content = _cm._has_code_content
_detect_code_language = _cm._detect_code_language
_FAMILY_CHARS_PER_TOKEN = _cm._FAMILY_CHARS_PER_TOKEN
_DEFAULT_CHARS_PER_TOKEN = _cm._DEFAULT_CHARS_PER_TOKEN


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_budget():
    """Create a mock PromptTokenBudget."""
    budget = MagicMock()
    budget.system_tokens = 800
    budget.project_tokens = 2000
    budget.history_tokens = 3200
    budget.user_tokens = 800
    budget.reserve_tokens = 1200
    budget.total_window = 8192
    budget.model = "test-model"
    budget.fingerprint_tokens = 0
    return budget


@pytest.fixture
def mock_budget_manager(mock_budget):
    """Create a mock PromptTokenBudgetManager."""
    mgr = MagicMock()
    mgr.calculate_budget.return_value = mock_budget
    mgr._system_ratio = 0.10
    mgr._project_ratio = 0.25
    mgr._history_ratio = 0.40
    mgr._user_ratio = 0.10
    mgr._reserve_ratio = 0.15
    return mgr


@pytest.fixture
def mock_project_builder():
    """Create a mock ProjectContextBuilder."""
    builder = MagicMock()
    ctx = MagicMock()
    ctx.context_text = "Project context: relevant chunks here"
    ctx.chunks_used = 3
    ctx.total_tokens_estimate = 150
    builder.build_context.return_value = ctx
    return builder


@pytest.fixture
def mock_compressor():
    """Create a mock ConversationCompressor."""
    comp = MagicMock()
    result = MagicMock()
    result.summary = "Summary of older messages"
    result.recent_messages = [
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    result.original_count = 10
    result.compressed_count = 8
    result.strategy_used = "hybrid"
    result.tokens_saved = 500
    result.compression_ratio = 0.6
    comp.compress.return_value = result
    return comp


@pytest.fixture
def mock_sliding_window():
    """Create a mock SlidingWindowManager."""
    sw = MagicMock()
    sw.prepare_messages.return_value = (
        [{"role": "user", "content": "kept message"}],
        {"strategy": "importance", "kept": 1, "dropped": 3, "total_tokens": 50},
    )
    return sw


@pytest.fixture
def optimizer(mock_budget_manager, mock_project_builder, mock_compressor, mock_sliding_window):
    """Create a fully wired ContextOptimizer."""
    return ContextOptimizer(
        config={"enabled": True, "active_preset": "balanced",
                "priority_presets": {
                    "balanced": {"system_ratio": 0.10, "project_ratio": 0.25,
                                 "history_ratio": 0.40, "user_ratio": 0.10,
                                 "reserve_ratio": 0.15},
                    "rag_heavy": {"system_ratio": 0.10, "project_ratio": 0.35,
                                  "history_ratio": 0.30, "user_ratio": 0.10,
                                  "reserve_ratio": 0.15},
                    "history_heavy": {"system_ratio": 0.10, "project_ratio": 0.15,
                                      "history_ratio": 0.50, "user_ratio": 0.10,
                                      "reserve_ratio": 0.15},
                },
                "emergency": {"enabled": True, "min_recent_messages": 2,
                              "max_block_chars": 2000},
                "compression": {"strategy": "auto"},
                "report": {"max_retained": 10}},
        budget_manager=mock_budget_manager,
        project_context_builder=mock_project_builder,
        conversation_compressor=mock_compressor,
        sliding_window_manager=mock_sliding_window,
    )


@pytest.fixture
def sample_history():
    """Sample conversation history."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Tell me about photosynthesis"},
        {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight..."},
    ]


# ============================================================================
# ContextOptimizer core tests
# ============================================================================

class TestContextOptimizerBasic:
    """Basic ContextOptimizer tests."""

    def test_creation_with_defaults(self):
        """Optimizer can be created with no arguments."""
        opt = ContextOptimizer()
        assert not opt.enabled
        assert opt.active_preset == "balanced"

    def test_creation_with_config(self):
        """Optimizer respects provided config."""
        opt = ContextOptimizer(config={"enabled": True, "active_preset": "rag_heavy",
                                       "priority_presets": {"rag_heavy": {"system_ratio": 0.10}}})
        assert opt.enabled
        assert opt.active_preset == "rag_heavy"

    def test_enabled_toggle(self, optimizer):
        """Enabled state can be toggled."""
        assert optimizer.enabled
        optimizer.enabled = False
        assert not optimizer.enabled
        optimizer.enabled = True
        assert optimizer.enabled

    def test_active_preset_setter(self, optimizer):
        """Active preset can be changed."""
        optimizer.active_preset = "rag_heavy"
        assert optimizer.active_preset == "rag_heavy"

    def test_active_preset_invalid(self, optimizer):
        """Setting invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            optimizer.active_preset = "nonexistent"

    def test_priority_presets_property(self, optimizer):
        """Priority presets are accessible."""
        presets = optimizer.priority_presets
        assert "balanced" in presets
        assert "rag_heavy" in presets
        assert "history_heavy" in presets

    def test_config_property(self, optimizer):
        """Config is accessible as dict."""
        cfg = optimizer.config
        assert "enabled" in cfg
        assert "priority_presets" in cfg

    def test_update_config(self, optimizer):
        """Config can be updated at runtime."""
        result = optimizer.update_config({"enabled": False})
        assert not optimizer.enabled
        assert "enabled" in result

    def test_update_config_custom_ratios(self, optimizer):
        """Custom ratios are stored via update_config."""
        optimizer.update_config({"custom_ratios": {"system_ratio": 0.20}})
        presets = optimizer.priority_presets
        assert "custom" in presets
        assert presets["custom"]["system_ratio"] == 0.20


class TestContextOptimizerOptimize:
    """Tests for the optimize() pipeline."""

    def test_optimize_basic(self, optimizer, sample_history):
        """optimize() returns OptimizedContext with messages."""
        result = optimizer.optimize(
            model="qwen3:32b",
            system_prompt="You are helpful.",
            user_message="What is AI?",
            conversation_history=sample_history,
        )
        assert isinstance(result, OptimizedContext)
        assert len(result.messages) >= 2  # system + user at minimum
        assert result.messages[0]["role"] == "system"
        assert result.messages[-1]["role"] == "user"
        assert result.total_tokens > 0

    def test_optimize_report_generated(self, optimizer, sample_history):
        """optimize() generates an OptimizationReport."""
        result = optimizer.optimize(
            model="qwen3:32b",
            system_prompt="You are helpful.",
            user_message="What is AI?",
            conversation_history=sample_history,
        )
        assert isinstance(result.report, OptimizationReport)
        assert result.report.model == "qwen3:32b"
        assert result.report.total_window > 0
        assert len(result.report.zones) > 0

    def test_optimize_report_stored(self, optimizer, sample_history):
        """Reports are stored in the optimizer's history."""
        optimizer.optimize(
            model="qwen3:32b",
            system_prompt="Test.",
            user_message="Hello",
            conversation_history=sample_history,
        )
        assert optimizer.last_report is not None
        assert len(optimizer.reports) == 1

    def test_optimize_multiple_reports(self, optimizer, sample_history):
        """Multiple reports are retained."""
        for _ in range(3):
            optimizer.optimize(
                model="qwen3:32b",
                system_prompt="Test.",
                user_message="Hello",
                conversation_history=sample_history,
            )
        assert len(optimizer.reports) == 3

    def test_optimize_rag_budget_passthrough(self, optimizer, mock_project_builder):
        """optimize() passes project_tokens to build_context()."""
        optimizer.optimize(
            model="qwen3:32b",
            system_prompt="Test.",
            user_message="Hello",
            project_id="proj-123",
            rag_query="search query",
        )
        mock_project_builder.build_context.assert_called_once()
        call_kwargs = mock_project_builder.build_context.call_args
        assert call_kwargs[1]["budget_tokens"] == 2000  # project_tokens from mock

    def test_optimize_no_project(self, optimizer, mock_project_builder):
        """optimize() without project_id skips RAG injection."""
        result = optimizer.optimize(
            model="qwen3:32b",
            system_prompt="Test.",
            user_message="Hello",
        )
        mock_project_builder.build_context.assert_not_called()

    def test_optimize_compression_triggered(self, optimizer, mock_compressor):
        """Compression is called when history exceeds budget."""
        # Create history that exceeds budget
        big_history = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
        ] * 5
        optimizer.optimize(
            model="qwen3:32b",
            system_prompt="Test.",
            user_message="Hello",
            conversation_history=big_history,
        )
        # Compressor should have been called
        mock_compressor.compress.assert_called()

    def test_optimize_empty_history(self, optimizer):
        """optimize() works with no history."""
        result = optimizer.optimize(
            model="qwen3:32b",
            system_prompt="System.",
            user_message="Question?",
            conversation_history=[],
        )
        assert len(result.messages) == 2  # system + user
        assert result.messages[0]["content"] == "System."
        assert result.messages[1]["content"] == "Question?"

    def test_optimize_no_budget_manager(self):
        """optimize() works without budget manager (fallback)."""
        opt = ContextOptimizer(config={"enabled": True})
        result = opt.optimize(
            model="test",
            system_prompt="Test.",
            user_message="Hello",
        )
        assert isinstance(result, OptimizedContext)
        assert len(result.messages) >= 2


class TestContextOptimizerEmergencyTruncation:
    """Tests for emergency truncation."""

    def test_emergency_truncate_drops_oldest(self, optimizer):
        """Emergency truncation drops oldest messages."""
        history = [
            {"role": "user", "content": "x" * 500 + str(i)} for i in range(20)
        ]
        truncated, removed = optimizer._emergency_truncate(
            history=history, target_tokens=50, model="test", min_recent=2
        )
        assert len(truncated) >= 2
        assert len(truncated) < 20
        assert removed > 0

    def test_emergency_truncate_keeps_min_recent(self, optimizer):
        """Emergency truncation keeps at least min_recent messages."""
        history = [
            {"role": "user", "content": "x" * 1000} for _ in range(10)
        ]
        truncated, _ = optimizer._emergency_truncate(
            history=history, target_tokens=1, model="test", min_recent=3
        )
        assert len(truncated) >= 3

    def test_emergency_truncate_empty_history(self, optimizer):
        """Emergency truncation on empty history is a no-op."""
        truncated, removed = optimizer._emergency_truncate(
            history=[], target_tokens=100, model="test"
        )
        assert truncated == []
        assert removed == 0


# ============================================================================
# Token estimation tests (S123 calibration)
# ============================================================================

class TestModelFamilyDetection:
    """Tests for detect_model_family()."""

    def test_qwen_family(self):
        assert detect_model_family("qwen3:32b") == "qwen"
        assert detect_model_family("qwen3-coder:30b") == "qwen"
        assert detect_model_family("qwen3.5:9b") == "qwen"

    def test_llama_family(self):
        assert detect_model_family("llama3:8b") == "llama"
        assert detect_model_family("llama3.2") == "llama"

    def test_codellama_before_llama(self):
        """codellama is detected as codellama, not llama."""
        assert detect_model_family("codellama:13b") == "codellama"

    def test_mistral_family(self):
        assert detect_model_family("mistral-small3.2") == "mistral"
        assert detect_model_family("mixtral:8x7b") == "mixtral"

    def test_deepseek_family(self):
        assert detect_model_family("deepseek-r1:32b") == "deepseek"

    def test_gemma_family(self):
        assert detect_model_family("gemma3:27b") == "gemma"

    def test_codegemma_before_gemma(self):
        assert detect_model_family("codegemma:7b") == "codegemma"

    def test_phi_family(self):
        assert detect_model_family("phi3:mini") == "phi"

    def test_unknown_family(self):
        assert detect_model_family("some-random-model") == "unknown"
        assert detect_model_family("") == "unknown"

    def test_starcoder(self):
        assert detect_model_family("starcoder2:15b") == "starcoder"


class TestFamilyCharsPerToken:
    """Tests for get_family_chars_per_token()."""

    def test_known_families(self):
        assert get_family_chars_per_token("qwen3:32b") == 3.2
        assert get_family_chars_per_token("llama3:8b") == 3.5
        assert get_family_chars_per_token("codellama:13b") == 3.0
        assert get_family_chars_per_token("mistral-small3.2") == 3.8

    def test_unknown_returns_default(self):
        result = get_family_chars_per_token("unknown-model")
        assert result == _DEFAULT_CHARS_PER_TOKEN


class TestEstimateTokensCalibrated:
    """Tests for estimate_tokens_calibrated()."""

    def test_empty_text(self):
        assert estimate_tokens_calibrated("") == 0

    def test_basic_estimation(self):
        text = "Hello world, this is a test."
        tokens = estimate_tokens_calibrated(text, "qwen3:32b")
        assert tokens > 0
        assert tokens == int(len(text) / 3.2)

    def test_code_detection_increases_tokens(self):
        code = "def hello():\n    print('world')\n    return True"
        plain = "Hello world this is a plain text message"
        code_tokens = estimate_tokens_calibrated(code, "qwen3:32b")
        plain_tokens = estimate_tokens_calibrated(plain, "qwen3:32b")
        # Code should produce more tokens per char
        code_ratio = code_tokens / len(code)
        plain_ratio = plain_tokens / len(plain)
        assert code_ratio > plain_ratio

    def test_override_chars_per_token(self):
        text = "x" * 100
        tokens = estimate_tokens_calibrated(text, chars_per_token_override=5.0)
        assert tokens == 20


class TestEstimateMessagesTokens:
    """Tests for estimate_messages_tokens() batch helper."""

    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        tokens = estimate_messages_tokens(msgs, "qwen3:32b")
        assert tokens > 0

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        tokens = estimate_messages_tokens(msgs, "llama3:8b")
        assert tokens > 0

    def test_batch_consistent_with_individual(self):
        """Batch estimation should match sum of individual estimations."""
        msgs = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi, how can I help?"},
        ]
        batch = estimate_messages_tokens(msgs, "qwen3:32b")
        individual = sum(
            estimate_tokens_calibrated(m["content"], "qwen3:32b") for m in msgs
        )
        # Should be equal or very close (batch doesn't apply code detection differently)
        assert abs(batch - individual) <= 2


class TestCodeDetection:
    """Tests for code language detection."""

    def test_python_detected(self):
        code = "def main():\n    import sys\n    print(sys.argv)"
        assert _has_code_content(code)
        assert _detect_code_language(code) == "python"

    def test_javascript_detected(self):
        code = "function hello() { const x = 42; console.log(x); }"
        assert _has_code_content(code)
        assert _detect_code_language(code) == "javascript"

    def test_rust_detected(self):
        code = "fn main() { let mut x = 5; pub fn helper(&self) {} }"
        assert _has_code_content(code)
        assert _detect_code_language(code) == "rust"

    def test_plain_text_no_code(self):
        text = "This is a normal sentence about cooking dinner."
        assert not _has_code_content(text)

    def test_r_detected(self):
        code = "library(tidyverse)\ndf <- data.frame(x=1:10)"
        assert _has_code_content(code)
        assert _detect_code_language(code) == "r"


# ============================================================================
# Priority override tests
# ============================================================================

class TestPriorityOverrides:
    """Tests for priority preset and custom ratio overrides."""

    def test_balanced_returns_none(self, optimizer):
        """Balanced preset returns None (use defaults)."""
        result = optimizer._resolve_priority_overrides(preset="balanced")
        assert result is None

    def test_rag_heavy_returns_ratios(self, optimizer):
        """rag_heavy preset returns its ratio dict."""
        result = optimizer._resolve_priority_overrides(preset="rag_heavy")
        assert result is not None
        assert result["project_ratio"] == 0.35

    def test_custom_ratios_override_preset(self, optimizer):
        """Custom ratios take priority over named preset."""
        custom = {"system_ratio": 0.20, "history_ratio": 0.60}
        result = optimizer._resolve_priority_overrides(
            preset="rag_heavy", custom_ratios=custom
        )
        assert result == custom

    def test_optimize_with_preset(self, optimizer, mock_budget_manager):
        """optimize() applies preset ratios to budget manager."""
        optimizer.optimize(
            model="qwen3:32b",
            system_prompt="Test.",
            user_message="Hello",
            preset="rag_heavy",
        )
        # Budget manager should have been called
        mock_budget_manager.calculate_budget.assert_called()

    def test_optimize_with_custom_ratios(self, optimizer, mock_budget_manager):
        """optimize() applies custom ratios to budget manager."""
        optimizer.optimize(
            model="qwen3:32b",
            system_prompt="Test.",
            user_message="Hello",
            custom_ratios={"system_ratio": 0.20},
        )
        mock_budget_manager.calculate_budget.assert_called()

    def test_budget_manager_ratios_restored(self, optimizer, mock_budget_manager):
        """Budget manager ratios are restored after override."""
        original_sys = mock_budget_manager._system_ratio
        optimizer.optimize(
            model="qwen3:32b",
            system_prompt="Test.",
            user_message="Hello",
            preset="rag_heavy",
        )
        assert mock_budget_manager._system_ratio == original_sys


# ============================================================================
# Data class tests
# ============================================================================

class TestDataClasses:
    """Tests for ZoneReport, OptimizationReport, OptimizedContext."""

    def test_zone_report_within_budget(self):
        z = ZoneReport(zone="test", budgeted_tokens=100, actual_tokens=80)
        assert z.within_budget

    def test_zone_report_over_budget(self):
        z = ZoneReport(zone="test", budgeted_tokens=100, actual_tokens=120)
        assert not z.within_budget

    def test_zone_report_as_dict(self):
        z = ZoneReport(zone="system", budgeted_tokens=100, actual_tokens=80,
                       trimmed_tokens=20, strategy="fixed")
        d = z.as_dict()
        assert d["zone"] == "system"
        assert d["within_budget"] is True

    def test_optimization_report_as_dict(self):
        r = OptimizationReport(model="test", total_window=8192)
        d = r.as_dict()
        assert d["model"] == "test"
        assert d["total_window"] == 8192
        assert "zones" in d
        assert "timestamp" in d

    def test_optimized_context_as_dict(self):
        ctx = OptimizedContext(
            system_prompt="Test",
            messages=[{"role": "system", "content": "Test"}],
            total_tokens=100,
        )
        d = ctx.as_dict()
        assert d["messages_count"] == 1
        assert d["total_tokens"] == 100


# ============================================================================
# Config loading tests
# ============================================================================

class TestConfigLoading:
    """Tests for YAML config loading."""

    def test_load_default_config(self):
        """Default config has expected keys."""
        cfg = _load_config(Path("/nonexistent"))
        assert "enabled" in cfg
        assert cfg["enabled"] is False
        assert "priority_presets" in cfg

    def test_load_from_yaml(self, tmp_path):
        """Config loads from a YAML file."""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"enabled": True, "active_preset": "rag_heavy"}))
        cfg = _load_config(cfg_file)
        assert cfg["enabled"] is True
        assert cfg["active_preset"] == "rag_heavy"

    def test_load_merges_with_defaults(self, tmp_path):
        """Loaded config merges with defaults."""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"enabled": True}))
        cfg = _load_config(cfg_file)
        assert cfg["enabled"] is True
        assert "priority_presets" in cfg  # from defaults


# ============================================================================
# Singleton tests
# ============================================================================

class TestSingleton:
    """Tests for module-level singleton."""

    def test_init_and_get(self):
        opt = init_optimizer(config={"enabled": False})
        assert get_optimizer() is opt

    def test_get_before_init(self):
        """get_optimizer returns None before init."""
        # Reset
        _co._optimizer = None
        assert get_optimizer() is None


# ============================================================================
# API endpoint tests (mock FastAPI)
# ============================================================================

class TestAPIEndpoints:
    """Tests for routes_context_optimizer endpoints."""

    def test_routes_file_exists(self):
        """Routes file exists."""
        assert Path("opti_oignon/api/routes_context_optimizer.py").exists()

    def test_routes_has_expected_endpoints(self):
        """Routes file defines expected endpoint functions."""
        content = Path("opti_oignon/api/routes_context_optimizer.py").read_text()
        assert "def get_optimizer_config" in content
        assert "def update_optimizer_config" in content
        assert "def get_optimization_report" in content
        assert "def list_presets" in content
        assert "def calculate_budget_with_preset" in content

    def test_routes_prefix(self):
        """Routes use correct API prefix."""
        content = Path("opti_oignon/api/routes_context_optimizer.py").read_text()
        assert "/api/context/optimizer" in content

    def test_routes_registered_in_app(self):
        """Routes are registered in app.py."""
        content = Path("opti_oignon/api/app.py").read_text()
        assert "context_optimizer_router" in content
        assert "include_router(context_optimizer_router)" in content

    def test_health_check_includes_optimizer(self):
        """Health check includes context_optimizer availability."""
        content = Path("opti_oignon/api/app.py").read_text()
        assert "CONTEXT_OPTIMIZER_AVAILABLE" in content
        assert '"context_optimizer"' in content

    def test_context_stats_includes_report(self):
        """context_stats endpoint includes optimization_report."""
        content = Path("opti_oignon/api/routes_context.py").read_text()
        assert "optimization_report" in content
        assert "last_optimization_report" in content


# ============================================================================
# Executor integration tests
# ============================================================================

class TestExecutorIntegration:
    """Tests for executor.py integration."""

    def test_executor_imports_optimizer(self):
        """executor.py imports context optimizer."""
        content = Path("opti_oignon/executor.py").read_text()
        assert "CONTEXT_OPTIMIZER_AVAILABLE" in content
        assert "_get_context_optimizer" in content

    def test_executor_has_last_optimization_report(self):
        """executor.py stores last optimization report."""
        content = Path("opti_oignon/executor.py").read_text()
        assert "_last_optimization_report" in content
        assert "last_optimization_report" in content

    def test_executor_s123_flag(self):
        """executor.py computes _s123_optimizer_active flag."""
        content = Path("opti_oignon/executor.py").read_text()
        assert "_s123_optimizer_active" in content

    def test_executor_fallback_on_failure(self):
        """executor.py falls back to manual pipeline on optimizer failure."""
        content = Path("opti_oignon/executor.py").read_text()
        assert "falling back to manual pipeline" in content

    def test_executor_skips_project_injection_when_active(self):
        """executor.py skips manual project injection when optimizer active."""
        content = Path("opti_oignon/executor.py").read_text()
        assert "not _s123_optimizer_active" in content


# ============================================================================
# Frontend tests
# ============================================================================

class TestFrontend:
    """Tests for frontend components."""

    def test_api_client_exists(self):
        """contextOptimizer.ts API client exists."""
        assert Path("frontend/src/lib/api/contextOptimizer.ts").exists()

    def test_api_client_exports(self):
        """API client exports expected functions."""
        content = Path("frontend/src/lib/api/contextOptimizer.ts").read_text()
        assert "getOptimizerConfig" in content
        assert "updateOptimizerConfig" in content
        assert "getOptimizerReports" in content
        assert "getOptimizerPresets" in content
        assert "calculateBudgetWithPreset" in content

    def test_panel_exists(self):
        """ContextOptimizerPanel.svelte exists."""
        assert Path("frontend/src/lib/components/settings/ContextOptimizerPanel.svelte").exists()

    def test_panel_imports_api(self):
        """Panel imports from contextOptimizer API client."""
        content = Path("frontend/src/lib/components/settings/ContextOptimizerPanel.svelte").read_text()
        assert "contextOptimizer" in content
        assert "getOptimizerConfig" in content

    def test_panel_no_hardcoded_hex(self):
        """Panel uses only --oo-* CSS variables, no hardcoded hex."""
        content = Path("frontend/src/lib/components/settings/ContextOptimizerPanel.svelte").read_text()
        # Find hex color patterns in style attributes (not in JS/comments)
        hex_matches = re.findall(r'style="[^"]*(?:color|background):\s*#[0-9a-fA-F]{3,8}', content)
        assert len(hex_matches) == 0, f"Hardcoded hex found: {hex_matches}"

    def test_panel_html_balance(self):
        """Panel has balanced HTML tags."""
        content = Path("frontend/src/lib/components/settings/ContextOptimizerPanel.svelte").read_text()
        # Only check the template part (after </script>)
        parts = content.split("</script>")
        template = parts[-1] if len(parts) > 1 else content
        div_opens = len(re.findall(r'<div[\s>]|<div$', template, re.MULTILINE))
        div_closes = template.count('</div>')
        assert div_opens == div_closes, f"div: {div_opens} opens, {div_closes} closes"

    def test_panel_registered_in_settings(self):
        """Panel is imported and used in settings page."""
        content = Path("frontend/src/routes/settings/+page.svelte").read_text()
        assert "ContextOptimizerPanel" in content
        assert "import ContextOptimizerPanel" in content

    def test_panel_no_french(self):
        """Panel contains no French text."""
        content = Path("frontend/src/lib/components/settings/ContextOptimizerPanel.svelte").read_text()
        french_words = ["disponible", "parametre", "retourne", "utilise", "fenetre"]
        for word in french_words:
            assert word not in content.lower(), f"French word found: {word}"


# ============================================================================
# Deprecation marker test
# ============================================================================

class TestDeprecation:
    """Tests for deprecation markers."""

    def test_s16_token_budget_manager_deprecated(self):
        """S16 TokenBudgetManager has deprecation notice."""
        content = Path("opti_oignon/context_window.py").read_text()
        # Find the TokenBudgetManager class docstring
        idx = content.find("class TokenBudgetManager")
        assert idx >= 0
        # Check deprecation in the docstring (next ~300 chars)
        docstring_area = content[idx:idx + 400]
        assert "deprecated" in docstring_area.lower()
        assert "S123" in docstring_area or "ContextOptimizer" in docstring_area


# ============================================================================
# Version bump test
# ============================================================================

class TestVersionBump:
    """Tests for version consistency."""

    def test_version_is_2_4_0(self):
        """Version should be 2.4.0 after S123."""
        content = Path("opti_oignon/__version__.py").read_text()
        assert '"2.4.0"' in content


# ============================================================================
# prompt_optimization priority_overrides test
# ============================================================================

class TestPromptOptimizationOverrides:
    """Tests for priority_overrides parameter in calculate_budget."""

    def test_calculate_budget_signature(self):
        """calculate_budget accepts priority_overrides parameter."""
        content = Path("opti_oignon/prompt_optimization.py").read_text()
        assert "priority_overrides" in content

    def test_calculate_budget_with_overrides_in_docstring(self):
        """Docstring mentions priority_overrides."""
        content = Path("opti_oignon/prompt_optimization.py").read_text()
        idx = content.find("def calculate_budget")
        area = content[idx:idx + 1500]
        assert "priority_overrides" in area
        assert "S123" in area
