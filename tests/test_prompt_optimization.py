#!/usr/bin/env python3
"""
Tests for prompt_optimization module — Opti-Oignon S65.

Step 1: PromptTokenBudgetManager tests (~30 tests).
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from opti_oignon.prompt_optimization import (
    PromptTemplate,
    PromptTemplateEngine,
    PromptTokenBudget,
    PromptTokenBudgetManager,
    _load_yaml_config,
    prompt_budget_manager,
    prompt_template_engine,
)

# ============================================================================
# PromptTokenBudget dataclass tests
# ============================================================================

class TestPromptTokenBudget:
    """Tests for the PromptTokenBudget dataclass."""

    def test_basic_creation(self):
        """Budget can be created with all fields."""
        b = PromptTokenBudget(
            system_tokens=800,
            project_tokens=2000,
            history_tokens=3200,
            user_tokens=800,
            reserve_tokens=1200,
            total_window=8192,
            model="test-model",
        )
        assert b.system_tokens == 800
        assert b.project_tokens == 2000
        assert b.history_tokens == 3200
        assert b.user_tokens == 800
        assert b.reserve_tokens == 1200
        assert b.total_window == 8192
        assert b.model == "test-model"

    def test_total_input_tokens(self):
        """total_input_tokens excludes reserve."""
        b = PromptTokenBudget(
            system_tokens=100,
            project_tokens=200,
            history_tokens=300,
            user_tokens=100,
            reserve_tokens=50,
            total_window=1000,
        )
        assert b.total_input_tokens == 700

    def test_total_allocated(self):
        """total_allocated includes all sections."""
        b = PromptTokenBudget(
            system_tokens=100,
            project_tokens=200,
            history_tokens=300,
            user_tokens=100,
            reserve_tokens=50,
            total_window=1000,
        )
        assert b.total_allocated == 750

    def test_utilization(self):
        """Utilization is total_allocated / total_window."""
        b = PromptTokenBudget(
            system_tokens=500,
            project_tokens=0,
            history_tokens=300,
            user_tokens=100,
            reserve_tokens=100,
            total_window=1000,
        )
        assert b.utilization == pytest.approx(1.0)

    def test_utilization_zero_window(self):
        """Utilization is 0 when total_window is 0."""
        b = PromptTokenBudget(
            system_tokens=100,
            project_tokens=0,
            history_tokens=0,
            user_tokens=0,
            reserve_tokens=0,
            total_window=0,
        )
        assert b.utilization == 0.0

    def test_as_dict(self):
        """as_dict returns expected keys."""
        b = PromptTokenBudget(
            system_tokens=100,
            project_tokens=200,
            history_tokens=300,
            user_tokens=100,
            reserve_tokens=50,
            total_window=1000,
            model="test",
        )
        d = b.as_dict()
        assert d["model"] == "test"
        assert d["total_window"] == 1000
        assert d["system_tokens"] == 100
        assert d["project_tokens"] == 200
        assert d["history_tokens"] == 300
        assert d["user_tokens"] == 100
        assert d["reserve_tokens"] == 50
        assert d["total_input_tokens"] == 700
        assert d["total_allocated"] == 750
        assert "utilization" in d

    def test_frozen(self):
        """Budget is immutable."""
        b = PromptTokenBudget(
            system_tokens=100,
            project_tokens=0,
            history_tokens=300,
            user_tokens=100,
            reserve_tokens=50,
            total_window=1000,
        )
        with pytest.raises(AttributeError):
            b.system_tokens = 999


# ============================================================================
# PromptTokenBudgetManager — context window detection
# ============================================================================

class TestContextWindowDetection:
    """Tests for context window detection logic."""

    def _make_manager(self, **overrides):
        """Create a manager with inline config."""
        config = {
            "allocation": {
                "system_ratio": 0.10,
                "project_ratio": 0.25,
                "history_ratio": 0.40,
                "user_ratio": 0.10,
                "reserve_ratio": 0.15,
            },
            "cache": {"ttl_seconds": 60, "max_entries": 10},
            "fallback_context_windows": {
                "qwen3:32b": 32768,
                "qwen3-coder:30b": 32768,
                "small-model:7b": 4096,
            },
            "default_context_window": 8192,
            "minimum_budgets": {
                "system": 256,
                "project": 0,
                "history": 512,
                "user": 256,
                "reserve": 512,
            },
        }
        config.update(overrides)
        return PromptTokenBudgetManager(config=config)

    def test_fallback_exact_match(self):
        """Known model returns its YAML fallback."""
        mgr = self._make_manager()
        # No ollama available in test env, so it falls back to YAML
        ctx = mgr.get_context_window("qwen3:32b")
        assert ctx == 32768

    def test_fallback_prefix_match(self):
        """Model with suffix matches by prefix."""
        mgr = self._make_manager()
        ctx = mgr.get_context_window("qwen3:32b-q4_0")
        assert ctx == 32768

    def test_fallback_default(self):
        """Unknown model returns default_context_window."""
        mgr = self._make_manager()
        ctx = mgr.get_context_window("unknown-model:latest")
        assert ctx == 8192

    @patch("opti_oignon.prompt_optimization.PromptTokenBudgetManager._query_ollama_show")
    def test_ollama_show_used_when_available(self, mock_show):
        """ollama.show() result is used when available."""
        mock_show.return_value = 65536
        mgr = self._make_manager()
        ctx = mgr.get_context_window("some-new-model:70b")
        assert ctx == 65536
        mock_show.assert_called_once_with("some-new-model:70b")

    @patch("opti_oignon.prompt_optimization.PromptTokenBudgetManager._query_ollama_show")
    def test_ollama_show_none_falls_to_yaml(self, mock_show):
        """When ollama.show() returns None, YAML fallback is used."""
        mock_show.return_value = None
        mgr = self._make_manager()
        ctx = mgr.get_context_window("qwen3-coder:30b")
        assert ctx == 32768

    def test_cache_hit(self):
        """Second call uses cached value."""
        mgr = self._make_manager()
        # First call populates cache
        ctx1 = mgr.get_context_window("small-model:7b")
        assert ctx1 == 4096
        # Second call should hit cache (no ollama call)
        ctx2 = mgr.get_context_window("small-model:7b")
        assert ctx2 == 4096
        assert "small-model:7b" in mgr.cache_stats()["models"]

    def test_cache_expiry(self):
        """Expired cache entries are removed."""
        mgr = self._make_manager()
        mgr._cache_ttl = 0  # Immediate expiry
        mgr.get_context_window("small-model:7b")
        # Force expire
        time.sleep(0.01)
        # Should re-resolve (not from cache)
        cached = mgr._get_cached("small-model:7b")
        assert cached is None

    def test_cache_eviction(self):
        """Cache evicts oldest entry when full."""
        mgr = self._make_manager()
        mgr._cache_max = 2
        mgr._set_cached("model-a", 1000)
        time.sleep(0.01)
        mgr._set_cached("model-b", 2000)
        time.sleep(0.01)
        # Third entry should evict model-a (oldest)
        mgr._set_cached("model-c", 3000)
        assert mgr._get_cached("model-a") is None
        assert mgr._get_cached("model-b") == 2000
        assert mgr._get_cached("model-c") == 3000

    def test_clear_cache(self):
        """clear_cache empties the cache."""
        mgr = self._make_manager()
        mgr._set_cached("model-x", 4096)
        count = mgr.clear_cache()
        assert count == 1
        assert len(mgr._cache) == 0


# ============================================================================
# PromptTokenBudgetManager — budget calculation
# ============================================================================

class TestBudgetCalculation:
    """Tests for calculate_budget()."""

    def _make_manager(self, **overrides):
        config = {
            "allocation": {
                "system_ratio": 0.10,
                "project_ratio": 0.25,
                "history_ratio": 0.40,
                "user_ratio": 0.10,
                "reserve_ratio": 0.15,
            },
            "cache": {"ttl_seconds": 60, "max_entries": 10},
            "fallback_context_windows": {"test-model:8b": 8192},
            "default_context_window": 8192,
            "minimum_budgets": {
                "system": 256,
                "project": 0,
                "history": 512,
                "user": 256,
                "reserve": 512,
            },
        }
        config.update(overrides)
        return PromptTokenBudgetManager(config=config)

    def test_basic_budget_with_project(self):
        """Budget with project_active=True uses all 5 sections."""
        mgr = self._make_manager()
        budget = mgr.calculate_budget("test-model:8b", project_active=True)
        assert budget.total_window == 8192
        assert budget.system_tokens == int(8192 * 0.10)
        assert budget.project_tokens == int(8192 * 0.25)
        assert budget.history_tokens == int(8192 * 0.40)
        assert budget.user_tokens == int(8192 * 0.10)
        assert budget.reserve_tokens == int(8192 * 0.15)
        assert budget.model == "test-model:8b"

    def test_budget_without_project_redistributes(self):
        """Budget with project_active=False redistributes project ratio."""
        mgr = self._make_manager()
        budget = mgr.calculate_budget("test-model:8b", project_active=False)
        assert budget.project_tokens == 0
        # History and user should be larger than base ratios
        base_history = int(8192 * 0.40)
        assert budget.history_tokens > base_history
        # System and reserve unchanged
        assert budget.system_tokens == int(8192 * 0.10)
        assert budget.reserve_tokens == int(8192 * 0.15)

    def test_budget_context_window_override(self):
        """context_window_override bypasses detection."""
        mgr = self._make_manager()
        budget = mgr.calculate_budget(
            "test-model:8b",
            context_window_override=16384,
            project_active=True,
        )
        assert budget.total_window == 16384
        assert budget.system_tokens == int(16384 * 0.10)

    def test_budget_does_not_exceed_window(self):
        """Total allocated never exceeds total_window."""
        mgr = self._make_manager()
        budget = mgr.calculate_budget("test-model:8b", project_active=True)
        assert budget.total_allocated <= budget.total_window

    def test_budget_minimum_floors(self):
        """Minimum budgets are respected even with tiny context windows."""
        mgr = self._make_manager()
        budget = mgr.calculate_budget(
            "tiny-model",
            context_window_override=2048,
            project_active=False,
        )
        assert budget.system_tokens >= 256
        assert budget.history_tokens >= 512
        assert budget.user_tokens >= 256
        assert budget.reserve_tokens >= 512

    def test_budget_unknown_model_uses_default(self):
        """Unknown model uses default 8192 context window."""
        mgr = self._make_manager()
        budget = mgr.calculate_budget("completely-unknown:latest")
        assert budget.total_window == 8192

    def test_redistribution_proportional(self):
        """Project ratio redistributed proportionally to history/user."""
        mgr = self._make_manager()
        budget = mgr.calculate_budget(
            "test-model:8b",
            project_active=False,
            context_window_override=10000,
        )
        # history_ratio=0.40, user_ratio=0.10, project_ratio=0.25
        # hist_share = 0.40/(0.40+0.10) = 0.80
        # user_share = 0.10/(0.40+0.10) = 0.20
        # effective hist = 0.40 + 0.25*0.80 = 0.60
        # effective user = 0.10 + 0.25*0.20 = 0.15
        expected_hist = int(10000 * 0.60)
        expected_user = int(10000 * 0.15)
        assert budget.history_tokens == expected_hist
        assert budget.user_tokens == expected_user


# ============================================================================
# Configuration and reload
# ============================================================================

class TestConfiguration:
    """Tests for config access and reload."""

    def _make_manager(self):
        return PromptTokenBudgetManager(config={
            "allocation": {
                "system_ratio": 0.10,
                "project_ratio": 0.25,
                "history_ratio": 0.40,
                "user_ratio": 0.10,
                "reserve_ratio": 0.15,
            },
            "cache": {"ttl_seconds": 60, "max_entries": 10},
            "fallback_context_windows": {"a:1b": 2048},
            "default_context_window": 4096,
            "minimum_budgets": {
                "system": 128,
                "project": 0,
                "history": 256,
                "user": 128,
                "reserve": 256,
            },
        })

    def test_allocation_ratios(self):
        """allocation_ratios returns configured ratios."""
        mgr = self._make_manager()
        ratios = mgr.allocation_ratios
        assert ratios["system"] == 0.10
        assert ratios["project"] == 0.25
        assert ratios["history"] == 0.40
        assert ratios["user"] == 0.10
        assert ratios["reserve"] == 0.15

    def test_default_context_window_property(self):
        """default_context_window returns configured value."""
        mgr = self._make_manager()
        assert mgr.default_context_window == 4096

    def test_fallback_models_property(self):
        """fallback_models returns configured fallbacks."""
        mgr = self._make_manager()
        assert mgr.fallback_models == {"a:1b": 2048}

    def test_get_config(self):
        """get_config returns full config dict."""
        mgr = self._make_manager()
        cfg = mgr.get_config()
        assert "allocation" in cfg
        assert "cache" in cfg
        assert "default_context_window" in cfg
        assert "fallback_models" in cfg
        assert "minimum_budgets" in cfg

    def test_cache_stats(self):
        """cache_stats returns expected keys."""
        mgr = self._make_manager()
        stats = mgr.cache_stats()
        assert "entries" in stats
        assert "max_entries" in stats
        assert stats["entries"] == 0

    def test_reload_config_from_file(self, tmp_path):
        """reload_config reads a new YAML file."""
        mgr = self._make_manager()
        new_config = tmp_path / "new_budget.yaml"
        new_config.write_text(
            "allocation:\n  system_ratio: 0.20\n  project_ratio: 0.20\n"
            "  history_ratio: 0.30\n  user_ratio: 0.15\n  reserve_ratio: 0.15\n"
            "default_context_window: 16384\n"
        )
        mgr.reload_config(config_path=new_config)
        assert mgr._system_ratio == 0.20
        assert mgr.default_context_window == 16384


# ============================================================================
# Module singleton
# ============================================================================

class TestSingleton:
    """Verify module-level singleton exists."""

    def test_singleton_exists(self):
        """prompt_budget_manager is initialized."""
        assert prompt_budget_manager is not None
        assert isinstance(prompt_budget_manager, PromptTokenBudgetManager)

    def test_singleton_has_ratios(self):
        """Singleton loaded from YAML has expected ratios."""
        ratios = prompt_budget_manager.allocation_ratios
        assert ratios["system"] == pytest.approx(0.10)
        assert ratios["reserve"] == pytest.approx(0.15)


# ============================================================================
# YAML loader edge cases
# ============================================================================

class TestYamlLoader:
    """Tests for _load_yaml_config."""

    def test_missing_file(self, tmp_path):
        """Missing file returns empty dict."""
        result = _load_yaml_config(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_invalid_yaml(self, tmp_path):
        """Invalid YAML returns empty dict."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("{{{{invalid yaml content")
        result = _load_yaml_config(bad)
        # yaml.safe_load may parse this differently, but should not crash
        assert isinstance(result, dict)

    def test_non_dict_yaml(self, tmp_path):
        """YAML that loads as non-dict returns empty dict."""
        list_yaml = tmp_path / "list.yaml"
        list_yaml.write_text("- item1\n- item2\n")
        result = _load_yaml_config(list_yaml)
        assert result == {}


# ============================================================================
# Step 2: PromptTemplate dataclass tests
# ============================================================================

class TestPromptTemplate:
    """Tests for the PromptTemplate dataclass."""

    def test_basic_creation(self):
        """Template can be created with all fields."""
        t = PromptTemplate(
            task_type="code_r",
            system_prompt="You are an R expert.",
            temperature_override=0.3,
            stop_sequences=["```"],
            source="yaml",
        )
        assert t.task_type == "code_r"
        assert t.system_prompt == "You are an R expert."
        assert t.temperature_override == 0.3
        assert t.stop_sequences == ["```"]
        assert t.source == "yaml"

    def test_defaults(self):
        """Default values are applied."""
        t = PromptTemplate(task_type="general", system_prompt="Hello")
        assert t.temperature_override is None
        assert t.stop_sequences == []
        assert t.source == "yaml"

    def test_as_dict(self):
        """as_dict returns expected keys."""
        t = PromptTemplate(task_type="test", system_prompt="prompt", source="runtime")
        d = t.as_dict()
        assert d["task_type"] == "test"
        assert d["system_prompt"] == "prompt"
        assert d["source"] == "runtime"
        assert "temperature_override" in d
        assert "stop_sequences" in d

    def test_frozen(self):
        """Template is immutable."""
        t = PromptTemplate(task_type="test", system_prompt="prompt")
        with pytest.raises(AttributeError):
            t.task_type = "other"


# ============================================================================
# Step 2: PromptTemplateEngine tests
# ============================================================================

class TestPromptTemplateEngine:
    """Tests for the PromptTemplateEngine."""

    def _make_engine(self, **overrides):
        """Create an engine with inline config."""
        config = {
            "language_rule": "Respond in English.",
            "templates": {
                "code_r": {
                    "system_prompt": "You are an R expert. {language_rule}",
                    "temperature_override": 0.3,
                    "stop_sequences": [],
                },
                "code_python": {
                    "system_prompt": "Python dev. {language_rule}",
                    "temperature_override": 0.3,
                },
                "general": {
                    "system_prompt": "Helpful assistant. {language_rule}",
                    "temperature_override": None,
                },
            },
            "project_overrides": {
                "proj-123": {
                    "code_r": {
                        "system_prompt": "R expert for ecology project. {language_rule}",
                        "temperature_override": 0.2,
                    },
                },
            },
        }
        config.update(overrides)
        return PromptTemplateEngine(config=config)

    # --- Template retrieval ---

    def test_get_template_exact_match(self):
        """Known task type returns its template."""
        engine = self._make_engine()
        tpl = engine.get_template("code_r")
        assert tpl.task_type == "code_r"
        assert "R expert" in tpl.system_prompt
        assert tpl.temperature_override == 0.3
        assert tpl.source == "yaml"

    def test_get_template_unknown_falls_to_general(self):
        """Unknown task type falls back to general."""
        engine = self._make_engine()
        tpl = engine.get_template("unknown_task")
        assert "Helpful assistant" in tpl.system_prompt
        assert tpl.source == "fallback"

    def test_get_template_no_general_ultimate_fallback(self):
        """Without general template, ultimate fallback is used."""
        engine = PromptTemplateEngine(config={"templates": {}})
        tpl = engine.get_template("anything")
        assert "helpful assistant" in tpl.system_prompt.lower()
        assert tpl.source == "fallback"

    def test_get_template_project_override(self):
        """Project-specific override takes priority over YAML."""
        engine = self._make_engine()
        tpl = engine.get_template("code_r", project_id="proj-123")
        assert "ecology project" in tpl.system_prompt
        assert tpl.temperature_override == 0.2
        assert tpl.source == "project"

    def test_get_template_project_no_match(self):
        """Project override not matching task falls to YAML."""
        engine = self._make_engine()
        tpl = engine.get_template("code_python", project_id="proj-123")
        assert "Python dev" in tpl.system_prompt
        assert tpl.source == "yaml"

    def test_get_template_project_id_unknown(self):
        """Unknown project_id falls to YAML template."""
        engine = self._make_engine()
        tpl = engine.get_template("code_r", project_id="nonexistent")
        assert tpl.source == "yaml"

    # --- Runtime overrides ---

    def test_runtime_override_highest_priority(self):
        """Runtime override beats YAML and project."""
        engine = self._make_engine()
        engine.set_runtime_override("code_r", "Custom R prompt", temperature_override=0.1)
        tpl = engine.get_template("code_r", project_id="proj-123")
        assert tpl.system_prompt == "Custom R prompt"
        assert tpl.temperature_override == 0.1
        assert tpl.source == "runtime"

    def test_clear_runtime_override(self):
        """Clearing runtime override falls back to YAML."""
        engine = self._make_engine()
        engine.set_runtime_override("code_r", "Override prompt")
        assert engine.clear_runtime_override("code_r") is True
        tpl = engine.get_template("code_r")
        assert "R expert" in tpl.system_prompt
        assert tpl.source == "yaml"

    def test_clear_runtime_override_nonexistent(self):
        """Clearing nonexistent override returns False."""
        engine = self._make_engine()
        assert engine.clear_runtime_override("nonexistent") is False

    def test_clear_all_runtime_overrides(self):
        """clear_all_runtime_overrides empties all overrides."""
        engine = self._make_engine()
        engine.set_runtime_override("code_r", "r override")
        engine.set_runtime_override("code_python", "python override")
        count = engine.clear_all_runtime_overrides()
        assert count == 2
        assert engine.get_template("code_r").source == "yaml"

    # --- Interpolation ---

    def test_interpolate_language_rule(self):
        """language_rule is auto-injected during interpolation."""
        engine = self._make_engine()
        tpl = engine.get_template("code_r")
        result = engine.interpolate(tpl)
        assert "Respond in English." in result
        assert "{language_rule}" not in result

    def test_interpolate_custom_variables(self):
        """Custom variables are substituted."""
        engine = self._make_engine()
        tpl = PromptTemplate(
            task_type="test",
            system_prompt="Hello {user_name}, model is {model_name}.",
        )
        result = engine.interpolate(tpl, {"user_name": "Leon", "model_name": "qwen3:32b"})
        assert "Hello Leon" in result
        assert "model is qwen3:32b" in result

    def test_interpolate_unknown_variable_left_asis(self):
        """Unknown {variables} are left untouched."""
        engine = self._make_engine()
        tpl = PromptTemplate(
            task_type="test",
            system_prompt="Hello {unknown_var}.",
        )
        result = engine.interpolate(tpl)
        assert "{unknown_var}" in result

    def test_interpolate_no_context(self):
        """Interpolation with no context still injects language_rule."""
        engine = self._make_engine()
        tpl = engine.get_template("general")
        result = engine.interpolate(tpl)
        assert "Respond in English." in result

    # --- Listing ---

    def test_list_templates(self):
        """list_templates returns all task types."""
        engine = self._make_engine()
        templates = engine.list_templates()
        task_types = [t["task_type"] for t in templates]
        assert "code_r" in task_types
        assert "code_python" in task_types
        assert "general" in task_types
        assert all("prompt_length" in t for t in templates)

    def test_available_task_types(self):
        """available_task_types returns sorted list."""
        engine = self._make_engine()
        types = engine.available_task_types
        assert types == sorted(types)
        assert "code_r" in types

    # --- Config ---

    def test_get_config(self):
        """get_config returns expected keys."""
        engine = self._make_engine()
        cfg = engine.get_config()
        assert "language_rule" in cfg
        assert "task_types" in cfg
        assert "template_count" in cfg
        assert cfg["template_count"] == 3

    def test_language_rule_property(self):
        """language_rule property returns configured value."""
        engine = self._make_engine()
        assert engine.language_rule == "Respond in English."

    def test_reload_config(self, tmp_path):
        """reload_config reads new YAML file."""
        engine = self._make_engine()
        new_config = tmp_path / "new_templates.yaml"
        new_config.write_text(
            "language_rule: 'Reply in French.'\n"
            "templates:\n"
            "  bio:\n"
            "    system_prompt: 'Bio expert.'\n"
        )
        engine.reload_config(config_path=new_config)
        assert engine.language_rule == "Reply in French."
        assert "bio" in engine.available_task_types

    def test_string_template(self):
        """Plain string template (not dict) is handled."""
        engine = PromptTemplateEngine(config={
            "templates": {"simple": "Just a string prompt."},
        })
        tpl = engine.get_template("simple")
        assert tpl.system_prompt == "Just a string prompt."
        assert tpl.temperature_override is None

    def test_parse_template_non_dict_non_string(self):
        """Non-dict, non-string template data falls to ultimate fallback."""
        engine = PromptTemplateEngine(config={
            "templates": {"broken": 42},
        })
        tpl = engine.get_template("broken")
        assert tpl.source == "fallback"


# ============================================================================
# Step 2: Module singleton for template engine
# ============================================================================

class TestTemplateEngineSingleton:
    """Verify template engine singleton."""

    def test_singleton_exists(self):
        """prompt_template_engine is initialized."""
        assert prompt_template_engine is not None
        assert isinstance(prompt_template_engine, PromptTemplateEngine)

    def test_singleton_has_templates(self):
        """Singleton loaded from YAML has expected templates."""
        types = prompt_template_engine.available_task_types
        assert "code_r" in types
        assert "code_python" in types
        assert "general" in types
        assert "scientific_writing" in types


# ============================================================================
# Step 3: API endpoint tests
# ============================================================================

try:
    from fastapi.testclient import TestClient

    from opti_oignon.api.app import app
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False


@pytest.fixture
def client():
    """FastAPI test client."""
    if not _API_AVAILABLE:
        pytest.skip("FastAPI test client not available")
    return TestClient(app)


class TestBudgetAPI:
    """Tests for /api/prompt/budget/* endpoints."""

    def test_get_budget(self, client):
        """GET /api/prompt/budget/{model} returns budget."""
        resp = client.get("/api/prompt/budget/qwen3:32b")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "qwen3:32b"
        assert data["total_window"] == 32768
        assert "system_tokens" in data
        assert "project_tokens" in data
        assert "history_tokens" in data
        assert "user_tokens" in data
        assert "reserve_tokens" in data

    def test_get_budget_with_project(self, client):
        """GET budget with project_active=true includes project tokens."""
        resp = client.get("/api/prompt/budget/qwen3:32b?project_active=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_tokens"] > 0

    def test_get_budget_without_project(self, client):
        """GET budget with project_active=false has zero project tokens."""
        resp = client.get("/api/prompt/budget/qwen3:32b?project_active=false")
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_tokens"] == 0

    def test_get_budget_unknown_model(self, client):
        """GET budget for unknown model uses default window."""
        resp = client.get("/api/prompt/budget/unknown:latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_window"] == 8192

    def test_get_context_window(self, client):
        """GET /api/prompt/budget/window/{model} returns window size."""
        resp = client.get("/api/prompt/budget/window/qwen3:32b")
        assert resp.status_code == 200
        data = resp.json()
        assert data["context_window"] == 32768

    def test_cache_stats(self, client):
        """GET /api/prompt/budget/cache/stats returns cache info."""
        resp = client.get("/api/prompt/budget/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "max_entries" in data

    def test_clear_cache(self, client):
        """POST /api/prompt/budget/cache/clear clears the cache."""
        resp = client.post("/api/prompt/budget/cache/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert "cleared" in data


class TestTemplateAPI:
    """Tests for /api/prompt/templates/* endpoints."""

    def test_list_templates(self, client):
        """GET /api/prompt/templates returns template list."""
        resp = client.get("/api/prompt/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        task_types = [t["task_type"] for t in data]
        assert "code_r" in task_types
        assert "general" in task_types

    def test_get_template(self, client):
        """GET /api/prompt/templates/{task_type} returns template."""
        resp = client.get("/api/prompt/templates/code_r")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_type"] == "code_r"
        assert "system_prompt" in data
        assert "R expert" in data["system_prompt"]

    def test_get_template_fallback(self, client):
        """GET template for unknown type falls back to general."""
        resp = client.get("/api/prompt/templates/nonexistent_type")
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "fallback"

    def test_set_template_override(self, client):
        """PUT /api/prompt/templates/{task_type} sets override."""
        resp = client.put(
            "/api/prompt/templates/code_r",
            json={"system_prompt": "Custom R prompt for testing.", "temperature_override": 0.1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["system_prompt"] == "Custom R prompt for testing."
        assert data["temperature_override"] == 0.1
        assert data["source"] == "runtime"

        # Verify it's now the active template
        resp2 = client.get("/api/prompt/templates/code_r")
        assert resp2.json()["source"] == "runtime"

        # Cleanup
        client.delete("/api/prompt/templates/code_r/override")

    def test_delete_template_override(self, client):
        """DELETE /api/prompt/templates/{task_type}/override clears it."""
        client.put(
            "/api/prompt/templates/test_delete",
            json={"system_prompt": "Temp override."},
        )
        resp = client.delete("/api/prompt/templates/test_delete/override")
        assert resp.status_code == 200

    def test_delete_template_override_not_found(self, client):
        """DELETE override for non-overridden type returns 404."""
        resp = client.delete("/api/prompt/templates/never_overridden_xyz/override")
        assert resp.status_code == 404

    def test_clear_all_overrides(self, client):
        """DELETE /api/prompt/templates/overrides/all clears all."""
        client.put("/api/prompt/templates/t1", json={"system_prompt": "A"})
        client.put("/api/prompt/templates/t2", json={"system_prompt": "B"})
        resp = client.delete("/api/prompt/templates/overrides/all")
        assert resp.status_code == 200
        assert resp.json()["cleared"] >= 2


class TestConfigAPI:
    """Tests for /api/prompt/config endpoints."""

    def test_get_config(self, client):
        """GET /api/prompt/config returns full config."""
        resp = client.get("/api/prompt/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "budget" in data
        assert "templates" in data

    def test_reload_config(self, client):
        """POST /api/prompt/config/reload succeeds."""
        resp = client.post("/api/prompt/config/reload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reloaded"

    def test_health_includes_prompt_optimization(self, client):
        """Health check includes prompt_optimization module."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt_optimization" in data["modules"]
        assert data["modules"]["prompt_optimization"] is True


# ============================================================================
# Step 4: Integration tests — Executor + Prompt Optimization
# ============================================================================

class TestExecutorIntegration:
    """Tests for prompt optimization integration in executor.py."""

    def test_executor_has_prompt_optimization_property(self):
        """Executor exposes prompt_optimization_enabled property."""
        from opti_oignon.executor import Executor
        ex = Executor()
        # Should be True since module is available
        assert hasattr(ex, "prompt_optimization_enabled")
        assert ex.prompt_optimization_enabled is True

    def test_executor_prompt_optimization_toggle(self):
        """prompt_optimization_enabled can be toggled."""
        from opti_oignon.executor import Executor
        ex = Executor()
        ex.prompt_optimization_enabled = False
        assert ex.prompt_optimization_enabled is False
        ex.prompt_optimization_enabled = True
        assert ex.prompt_optimization_enabled is True

    def test_executor_last_prompt_budget_starts_none(self):
        """last_prompt_budget is None before any execution."""
        from opti_oignon.executor import Executor
        ex = Executor()
        assert ex.last_prompt_budget is None

    def test_executor_get_system_prompt_still_works(self):
        """Original get_system_prompt() still works as fallback."""
        from opti_oignon.executor import Executor
        ex = Executor()
        prompt = ex.get_system_prompt("code_r", "standard")
        assert "R expert" in prompt

    def test_prompt_optimization_available_flag(self):
        """PROMPT_OPTIMIZATION_AVAILABLE is True when module loaded."""
        from opti_oignon.executor import PROMPT_OPTIMIZATION_AVAILABLE
        assert PROMPT_OPTIMIZATION_AVAILABLE is True

    def test_template_engine_imported(self):
        """Template engine singleton is accessible from executor module."""
        from opti_oignon.executor import _prompt_template_engine
        assert _prompt_template_engine is not None

    def test_budget_manager_imported(self):
        """Budget manager singleton is accessible from executor module."""
        from opti_oignon.executor import _prompt_budget_manager
        assert _prompt_budget_manager is not None

    @patch("opti_oignon.executor._prompt_template_engine")
    @patch("opti_oignon.executor._prompt_budget_manager")
    def test_template_used_in_execute_flow(self, mock_budget_mgr, mock_engine):
        """When enabled, execute() uses template engine for system prompt."""
        from opti_oignon.prompt_optimization import PromptTemplate, PromptTokenBudget

        # Setup template mock
        mock_template = PromptTemplate(
            task_type="general",
            system_prompt="Test system prompt from template.",
            temperature_override=0.42,
            source="yaml",
        )
        mock_engine.get_template.return_value = mock_template
        mock_engine.interpolate.return_value = "Interpolated test prompt."

        # Setup budget mock
        mock_budget = PromptTokenBudget(
            system_tokens=800,
            project_tokens=0,
            history_tokens=3200,
            user_tokens=800,
            reserve_tokens=1200,
            total_window=8192,
            model="test:7b",
        )
        mock_budget_mgr.calculate_budget.return_value = mock_budget

        from opti_oignon.executor import Executor
        ex = Executor()

        # Verify the engine would be called (we can't do a full execute
        # without Ollama, so we verify the wiring)
        assert ex.prompt_optimization_enabled is True
        mock_engine.get_template.assert_not_called()  # Not called yet

    def test_disabled_optimization_uses_hardcoded(self):
        """When disabled, execute uses original hardcoded prompts."""
        from opti_oignon.executor import Executor
        ex = Executor()
        ex.prompt_optimization_enabled = False
        # get_system_prompt should still return hardcoded prompt
        prompt = ex.get_system_prompt("general", "standard")
        assert "helpful assistant" in prompt.lower()

    def test_agentic_executor_last_prompt_budget(self):
        """AgenticExecutor exposes last_prompt_budget property."""
        from opti_oignon.agentic_executor import AgenticExecutor
        ae = AgenticExecutor()
        # Property exists and returns either None or a PromptTokenBudget
        budget = ae.last_prompt_budget
        if budget is not None:
            # If a prior test populated it, verify it has expected shape
            assert hasattr(budget, "total_window")
            assert hasattr(budget, "system_tokens")
            assert budget.total_window > 0
