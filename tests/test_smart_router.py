#!/usr/bin/env python3
"""
TESTS -- Smart Router (S54)
============================

Comprehensive tests for the SmartRouter, SmartRoutingResult,
YAML config loading, model selection scoring, pipeline routing,
fallback behavior, caching, and API endpoint integration.

Target: 65+ tests, zero regressions.
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from opti_oignon.model_profiles import (
    VALID_QUALITY_TIERS,
    VALID_SPEED_TIERS,
    ModelProfile,
    ModelProfileManager,
    RoutingReason,
)
from opti_oignon.smart_router import (
    DEFAULT_CONTEXT_REQUIREMENTS,
    DEFAULT_SPEED_WEIGHTS,
    PIPELINE_TO_TASK_MAPPING,
    SmartRouter,
    SmartRoutingResult,
)

# =============================================================================
# HELPERS
# =============================================================================

def _make_profile(
    name: str = "test-model:7b",
    display_name: str = "Test Model",
    capabilities: list[str] | None = None,
    context_window: int = 32768,
    speed_tier: str = "medium",
    quality_tier: str = "medium",
    recommended_for: list[str] | None = None,
    not_recommended_for: list[str] | None = None,
    task_scores: dict[str, float] | None = None,
) -> ModelProfile:
    """Create a ModelProfile for testing."""
    return ModelProfile(
        name=name,
        display_name=display_name,
        capabilities=capabilities or ["general"],
        context_window=context_window,
        speed_tier=speed_tier,
        quality_tier=quality_tier,
        recommended_for=recommended_for or ["general"],
        not_recommended_for=not_recommended_for or [],
        task_scores=task_scores or {},
    )


def _make_manager_with_profiles(profiles: list[ModelProfile]) -> ModelProfileManager:
    """Create a pre-loaded ModelProfileManager with given profiles."""
    mgr = ModelProfileManager(profiles_path=Path("/dev/null"))
    mgr._loaded = True
    for p in profiles:
        mgr._profiles[p.name] = p
    return mgr


def _make_yaml_config(config_dict: dict) -> Path:
    """Write a YAML config to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(config_dict, tmp, default_flow_style=False)
    tmp.close()
    return Path(tmp.name)


@dataclass
class FakeRouting:
    """Fake RoutingResult for testing override_routing."""
    model: str = "default-model"
    explanation: str = ""
    routing_reason: dict | None = None


# =============================================================================
# TEST: SmartRoutingResult
# =============================================================================

class TestSmartRoutingResult(unittest.TestCase):
    """Tests for the SmartRoutingResult dataclass."""

    def test_default_values(self):
        r = SmartRoutingResult(model="test:7b")
        self.assertEqual(r.model, "test:7b")
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.task_score, 0.0)
        self.assertEqual(r.speed_weight, 1.0)
        self.assertEqual(r.context_fit, 1.0)
        self.assertFalse(r.profile_used)
        self.assertFalse(r.fallback)
        self.assertEqual(r.alternatives, [])

    def test_to_dict(self):
        r = SmartRoutingResult(
            model="qwen3:32b",
            score=0.8765,
            task_score=0.9,
            speed_weight=1.2,
            context_fit=0.95,
            reason="Best for code",
            profile_used=True,
        )
        d = r.to_dict()
        self.assertEqual(d["model"], "qwen3:32b")
        self.assertEqual(d["score"], 0.8765)
        self.assertEqual(d["task_score"], 0.9)
        self.assertEqual(d["speed_weight"], 1.2)
        self.assertTrue(d["profile_used"])
        self.assertFalse(d["fallback"])

    def test_to_dict_rounding(self):
        r = SmartRoutingResult(model="x", score=0.123456789, task_score=0.999999)
        d = r.to_dict()
        self.assertEqual(d["score"], 0.1235)
        self.assertEqual(d["task_score"], 1.0)

    def test_alternatives_list(self):
        alts = [{"model": "a", "score": 0.5}, {"model": "b", "score": 0.3}]
        r = SmartRoutingResult(model="x", alternatives=alts)
        self.assertEqual(len(r.to_dict()["alternatives"]), 2)


# =============================================================================
# TEST: SmartRouter -- Initialization
# =============================================================================

class TestSmartRouterInit(unittest.TestCase):
    """Tests for SmartRouter initialization and config loading."""

    def test_default_init(self):
        mgr = _make_manager_with_profiles([])
        sr = SmartRouter(
            profile_manager=mgr,
            config_path=Path("/nonexistent/config.yaml"),
        )
        self.assertEqual(sr.default_model, "qwen3:32b")
        self.assertEqual(sr.speed_preference, "balanced")

    def test_custom_init(self):
        mgr = _make_manager_with_profiles([])
        sr = SmartRouter(
            profile_manager=mgr,
            enabled=False,
            default_model="custom:7b",
            speed_preference="fast",
            config_path=Path("/nonexistent/config.yaml"),
        )
        self.assertFalse(sr._enabled)
        self.assertEqual(sr.default_model, "custom:7b")
        self.assertEqual(sr.speed_preference, "fast")

    def test_yaml_config_loading(self):
        config = {
            "smart_routing": {
                "enabled": False,
                "default_model": "yaml-model:14b",
                "speed_preference": "quality",
                "speed_weights": {"fast": 1.5, "medium": 1.0, "slow": 0.5},
            }
        }
        config_path = _make_yaml_config(config)
        try:
            mgr = _make_manager_with_profiles([])
            sr = SmartRouter(profile_manager=mgr, config_path=config_path)
            self.assertFalse(sr._enabled)
            self.assertEqual(sr.default_model, "yaml-model:14b")
            self.assertEqual(sr.speed_preference, "quality")
            self.assertEqual(sr._speed_weights["fast"], 1.5)
        finally:
            os.unlink(config_path)

    def test_yaml_config_invalid_speed_preference(self):
        config = {"smart_routing": {"speed_preference": "invalid_value"}}
        config_path = _make_yaml_config(config)
        try:
            mgr = _make_manager_with_profiles([])
            sr = SmartRouter(profile_manager=mgr, config_path=config_path)
            # Should keep default 'balanced' when YAML value is invalid
            self.assertEqual(sr.speed_preference, "balanced")
        finally:
            os.unlink(config_path)

    def test_yaml_config_missing_file(self):
        mgr = _make_manager_with_profiles([])
        sr = SmartRouter(
            profile_manager=mgr,
            config_path=Path("/nonexistent/file.yaml"),
        )
        # Should not raise, just use defaults
        self.assertTrue(sr._enabled)

    def test_yaml_config_malformed(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        tmp.write(": : : invalid yaml {{{\n")
        tmp.close()
        try:
            mgr = _make_manager_with_profiles([])
            sr = SmartRouter(profile_manager=mgr, config_path=Path(tmp.name))
            # Should not raise
            self.assertTrue(sr._enabled)
        finally:
            os.unlink(tmp.name)

    def test_yaml_config_empty(self):
        config_path = _make_yaml_config({})
        try:
            mgr = _make_manager_with_profiles([])
            sr = SmartRouter(profile_manager=mgr, config_path=config_path)
            self.assertTrue(sr._enabled)
        finally:
            os.unlink(config_path)


# =============================================================================
# TEST: SmartRouter -- Enabled property
# =============================================================================

class TestSmartRouterEnabled(unittest.TestCase):
    """Tests for the enabled property and its dependencies."""

    def test_enabled_with_profiles(self):
        mgr = _make_manager_with_profiles([_make_profile()])
        sr = SmartRouter(
            profile_manager=mgr,
            config_path=Path("/nonexistent"),
        )
        self.assertTrue(sr.enabled)

    def test_disabled_explicitly(self):
        mgr = _make_manager_with_profiles([_make_profile()])
        sr = SmartRouter(
            profile_manager=mgr,
            enabled=False,
            config_path=Path("/nonexistent"),
        )
        self.assertFalse(sr.enabled)

    def test_disabled_no_profile_manager(self):
        # When profile_manager=None, constructor uses the default singleton
        # To truly disable, we need to mock PROFILES_AVAILABLE
        sr = SmartRouter(
            profile_manager=None,
            config_path=Path("/nonexistent"),
        )
        # Force _profile_manager to None to simulate unavailable profiles
        sr._profile_manager = None
        self.assertFalse(sr.enabled)

    def test_enabled_setter(self):
        mgr = _make_manager_with_profiles([_make_profile()])
        sr = SmartRouter(
            profile_manager=mgr,
            config_path=Path("/nonexistent"),
        )
        sr.enabled = False
        self.assertFalse(sr.enabled)
        sr.enabled = True
        self.assertTrue(sr.enabled)


# =============================================================================
# TEST: SmartRouter -- Model Selection
# =============================================================================

class TestSmartRouterSelection(unittest.TestCase):
    """Tests for select_model core functionality."""

    def setUp(self):
        self.code_model = _make_profile(
            name="coder:30b",
            display_name="Coder 30B",
            capabilities=["code", "reasoning"],
            speed_tier="medium",
            quality_tier="high",
            recommended_for=["code_python", "debug"],
            task_scores={
                "code_python": 0.95,
                "debug": 0.90,
                "general": 0.50,
            },
        )
        self.fast_model = _make_profile(
            name="fast:7b",
            display_name="Fast 7B",
            capabilities=["general", "fast"],
            speed_tier="fast",
            quality_tier="low",
            recommended_for=["quick_answer", "chat"],
            task_scores={
                "quick_answer": 0.88,
                "chat": 0.85,
                "general": 0.70,
            },
        )
        self.reasoning_model = _make_profile(
            name="thinker:32b",
            display_name="Thinker 32B",
            capabilities=["reasoning"],
            speed_tier="slow",
            quality_tier="high",
            recommended_for=["reasoning", "mathematical"],
            task_scores={
                "reasoning": 0.95,
                "mathematical": 0.93,
                "planning_deep": 0.88,
                "general": 0.40,
            },
        )
        self.embed_model = _make_profile(
            name="embed:large",
            capabilities=["embeddings"],
            recommended_for=["embeddings"],
        )
        self.mgr = _make_manager_with_profiles([
            self.code_model, self.fast_model,
            self.reasoning_model, self.embed_model,
        ])
        self.router = SmartRouter(
            profile_manager=self.mgr,
            config_path=Path("/nonexistent"),
        )

    def test_select_code_verify(self):
        r = self.router.select_model("code_verify")
        self.assertEqual(r.model, "coder:30b")
        self.assertTrue(r.profile_used)
        self.assertFalse(r.fallback)

    def test_select_reasoning(self):
        r = self.router.select_model("reasoning")
        self.assertEqual(r.model, "thinker:32b")
        self.assertGreater(r.task_score, 0.9)

    def test_select_direct_prefers_fast(self):
        r = self.router.select_model("direct")
        # fast:7b has high general/quick_answer + fast speed bonus
        self.assertEqual(r.model, "fast:7b")

    def test_select_with_prefer_speed_true(self):
        r = self.router.select_model("direct", prefer_speed=True)
        self.assertEqual(r.model, "fast:7b")
        # Score should be higher than balanced
        r2 = self.router.select_model("direct", prefer_speed=None)
        self.assertGreaterEqual(r.score, r2.score)

    def test_select_with_prefer_speed_false_quality(self):
        r = self.router.select_model("reasoning", prefer_speed=False)
        # Quality preference should boost slow models
        self.assertEqual(r.model, "thinker:32b")

    def test_select_excludes_embedding_models(self):
        r = self.router.select_model("direct")
        self.assertNotEqual(r.model, "embed:large")
        # Check alternatives too
        alt_models = [a["model"] for a in r.alternatives]
        self.assertNotIn("embed:large", alt_models)

    def test_select_with_excluded_models(self):
        r = self.router.select_model(
            "code_verify",
            excluded_models=["coder:30b"],
        )
        self.assertNotEqual(r.model, "coder:30b")

    def test_select_unknown_step_type_uses_direct_scoring(self):
        r = self.router.select_model("unknown_pipeline")
        # Should not crash, just use whatever scores best
        self.assertTrue(r.model)

    def test_select_result_has_alternatives(self):
        r = self.router.select_model("code_verify")
        # Should have alternatives (3 non-embed models - 1 best = up to 2)
        self.assertIsInstance(r.alternatives, list)

    def test_select_result_reason_not_empty(self):
        r = self.router.select_model("reasoning")
        self.assertTrue(r.reason)
        self.assertIn("reasoning", r.reason.lower())

    def test_select_context_fit_penalty(self):
        # Model with small context window for reasoning step (needs 32768)
        small_model = _make_profile(
            name="small:1b",
            context_window=4096,
            speed_tier="fast",
            task_scores={"reasoning": 0.99},
            recommended_for=["reasoning"],
        )
        mgr = _make_manager_with_profiles([small_model, self.reasoning_model])
        router = SmartRouter(
            profile_manager=mgr,
            config_path=Path("/nonexistent"),
        )
        r = router.select_model("reasoning")
        # thinker:32b should win despite lower raw task score because of context fit
        # small model gets penalized: 0.99 * ctx_fit(4096/32768=0.125) = ~0.12
        self.assertEqual(r.model, "thinker:32b")

    def test_select_explicit_required_context(self):
        r = self.router.select_model("direct", required_context=65536)
        # All test models have 32768 context, so all get penalized
        self.assertLess(r.context_fit, 1.0)

    def test_score_formula(self):
        r = self.router.select_model("code_verify")
        # Verify: score = task_score * speed_weight * context_fit
        expected = r.task_score * r.speed_weight * r.context_fit
        self.assertAlmostEqual(r.score, expected, places=3)


# =============================================================================
# TEST: SmartRouter -- Fallback behavior
# =============================================================================

class TestSmartRouterFallback(unittest.TestCase):
    """Tests for fallback when routing cannot select a model."""

    def test_fallback_when_disabled(self):
        mgr = _make_manager_with_profiles([_make_profile()])
        sr = SmartRouter(
            profile_manager=mgr,
            enabled=False,
            config_path=Path("/nonexistent"),
        )
        r = sr.select_model("code_verify")
        self.assertTrue(r.fallback)
        self.assertEqual(r.model, "qwen3:32b")
        self.assertIn("disabled", r.reason.lower())

    def test_fallback_no_profiles(self):
        mgr = _make_manager_with_profiles([])
        sr = SmartRouter(
            profile_manager=mgr,
            config_path=Path("/nonexistent"),
        )
        r = sr.select_model("code_verify")
        self.assertTrue(r.fallback)
        self.assertIn("no model profiles", r.reason.lower())

    def test_fallback_only_embeddings(self):
        embed = _make_profile(
            name="embed:x",
            capabilities=["embeddings"],
            recommended_for=["embeddings"],
        )
        mgr = _make_manager_with_profiles([embed])
        sr = SmartRouter(
            profile_manager=mgr,
            config_path=Path("/nonexistent"),
        )
        r = sr.select_model("code_verify")
        self.assertTrue(r.fallback)

    def test_fallback_all_excluded(self):
        p = _make_profile(name="only:7b", task_scores={"general": 0.5})
        mgr = _make_manager_with_profiles([p])
        sr = SmartRouter(
            profile_manager=mgr,
            config_path=Path("/nonexistent"),
        )
        r = sr.select_model("direct", excluded_models=["only:7b"])
        self.assertTrue(r.fallback)

    def test_fallback_custom_default_model(self):
        mgr = _make_manager_with_profiles([])
        sr = SmartRouter(
            profile_manager=mgr,
            default_model="my-fallback:latest",
            config_path=Path("/nonexistent"),
        )
        r = sr.select_model("think")
        self.assertEqual(r.model, "my-fallback:latest")
        self.assertTrue(r.fallback)


# =============================================================================
# TEST: SmartRouter -- Pipeline selection
# =============================================================================

class TestSmartRouterPipeline(unittest.TestCase):
    """Tests for select_for_pipeline."""

    def setUp(self):
        profiles = [
            _make_profile(
                name="coder:30b", speed_tier="medium", quality_tier="high",
                task_scores={"code_python": 0.95, "debug": 0.90, "general": 0.50},
                recommended_for=["code_python", "debug"],
            ),
            _make_profile(
                name="general:32b", speed_tier="medium", quality_tier="high",
                task_scores={"general": 0.90, "quick_answer": 0.80, "reasoning": 0.75},
                recommended_for=["general", "quick_answer"],
            ),
            _make_profile(
                name="fast:7b", speed_tier="fast", quality_tier="low",
                task_scores={"quick_answer": 0.88, "general": 0.70},
                recommended_for=["quick_answer", "chat"],
            ),
        ]
        self.mgr = _make_manager_with_profiles(profiles)
        self.router = SmartRouter(
            profile_manager=self.mgr,
            config_path=Path("/nonexistent"),
        )

    def test_pipeline_returns_all_steps(self):
        steps = ["direct", "code_verify", "think"]
        results = self.router.select_for_pipeline(steps)
        self.assertEqual(set(results.keys()), set(steps))

    def test_pipeline_different_models_per_step(self):
        steps = ["direct", "code_verify"]
        results = self.router.select_for_pipeline(steps)
        # code_verify should use coder, direct should use fast/general
        self.assertEqual(results["code_verify"].model, "coder:30b")
        self.assertNotEqual(results["direct"].model, "coder:30b")

    def test_pipeline_with_custom_context(self):
        steps = ["direct", "reasoning"]
        ctx = {"reasoning": 65536}
        results = self.router.select_for_pipeline(steps, required_contexts=ctx)
        self.assertIn("reasoning", results)

    def test_pipeline_empty_list(self):
        results = self.router.select_for_pipeline([])
        self.assertEqual(results, {})

    def test_pipeline_single_step(self):
        results = self.router.select_for_pipeline(["code_verify"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results["code_verify"].model, "coder:30b")


# =============================================================================
# TEST: SmartRouter -- Caching
# =============================================================================

class TestSmartRouterCaching(unittest.TestCase):
    """Tests for the internal routing cache."""

    def setUp(self):
        profiles = [
            _make_profile(
                name="model-a:7b", task_scores={"general": 0.90},
                speed_tier="fast", recommended_for=["general"],
            ),
        ]
        self.mgr = _make_manager_with_profiles(profiles)
        self.router = SmartRouter(
            profile_manager=self.mgr,
            config_path=Path("/nonexistent"),
        )

    def test_cache_hit(self):
        r1 = self.router.select_model("direct")
        r2 = self.router.select_model("direct")
        # Same object from cache
        self.assertIs(r1, r2)

    def test_cache_miss_different_step(self):
        r1 = self.router.select_model("direct")
        r2 = self.router.select_model("think")
        self.assertIsNot(r1, r2)

    def test_cache_miss_with_excluded(self):
        r1 = self.router.select_model("direct")
        r2 = self.router.select_model("direct", excluded_models=["nonexistent"])
        # Excluded models bypass cache
        self.assertIsNot(r1, r2)

    def test_clear_cache(self):
        self.router.select_model("direct")
        self.assertTrue(len(self.router._cache) > 0)
        self.router.clear_cache()
        self.assertEqual(len(self.router._cache), 0)

    def test_speed_preference_change_clears_cache(self):
        self.router.select_model("direct")
        self.assertTrue(len(self.router._cache) > 0)
        self.router.speed_preference = "fast"
        self.assertEqual(len(self.router._cache), 0)


# =============================================================================
# TEST: SmartRouter -- Configuration
# =============================================================================

class TestSmartRouterConfiguration(unittest.TestCase):
    """Tests for configure() and get_config()."""

    def setUp(self):
        self.mgr = _make_manager_with_profiles([_make_profile()])
        self.router = SmartRouter(
            profile_manager=self.mgr,
            config_path=Path("/nonexistent"),
        )

    def test_configure_enabled(self):
        self.router.configure(enabled=False)
        self.assertFalse(self.router._enabled)

    def test_configure_default_model(self):
        self.router.configure(default_model="new-model:latest")
        self.assertEqual(self.router.default_model, "new-model:latest")

    def test_configure_speed_preference(self):
        self.router.configure(speed_preference="quality")
        self.assertEqual(self.router.speed_preference, "quality")

    def test_configure_speed_weights(self):
        self.router.configure(speed_weights={"fast": 2.0})
        self.assertEqual(self.router._speed_weights["fast"], 2.0)

    def test_configure_clears_cache(self):
        self.router.select_model("direct")
        self.assertTrue(len(self.router._cache) > 0)
        self.router.configure(enabled=True)
        self.assertEqual(len(self.router._cache), 0)

    def test_get_config_structure(self):
        config = self.router.get_config()
        self.assertIn("enabled", config)
        self.assertIn("profiles_available", config)
        self.assertIn("operational", config)
        self.assertIn("default_model", config)
        self.assertIn("speed_preference", config)
        self.assertIn("speed_weights", config)
        self.assertIn("context_requirements", config)
        self.assertIn("profile_count", config)

    def test_get_config_values(self):
        config = self.router.get_config()
        self.assertEqual(config["default_model"], "qwen3:32b")
        self.assertEqual(config["speed_preference"], "balanced")

    def test_save_config(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", delete=False
        )
        tmp.close()
        try:
            self.router.configure(
                default_model="saved:7b",
                speed_preference="fast",
            )
            result = self.router.save_config(path=Path(tmp.name))
            self.assertTrue(result)
            # Verify file content
            with open(tmp.name) as f:
                data = yaml.safe_load(f)
            self.assertEqual(data["smart_routing"]["default_model"], "saved:7b")
            self.assertEqual(data["smart_routing"]["speed_preference"], "fast")
        finally:
            os.unlink(tmp.name)

    def test_to_dict(self):
        d = self.router.to_dict()
        self.assertIn("cache_size", d)
        self.assertIn("pipeline_task_mapping", d)
        self.assertIn("enabled", d)


# =============================================================================
# TEST: SmartRouter -- Speed adjustments
# =============================================================================

class TestSmartRouterSpeedAdjustment(unittest.TestCase):
    """Tests for speed preference impact on scoring."""

    def setUp(self):
        self.fast_model = _make_profile(
            name="speedy:7b", speed_tier="fast",
            task_scores={"general": 0.75, "quick_answer": 0.80},
            recommended_for=["quick_answer"],
        )
        self.slow_model = _make_profile(
            name="deep:32b", speed_tier="slow", quality_tier="high",
            task_scores={"general": 0.92, "quick_answer": 0.88},
            recommended_for=["general"],
        )
        self.mgr = _make_manager_with_profiles([self.fast_model, self.slow_model])

    def test_balanced_preference(self):
        router = SmartRouter(
            profile_manager=self.mgr,
            speed_preference="balanced",
            config_path=Path("/nonexistent"),
        )
        r = router.select_model("direct")
        # With balanced, speed weights are even
        self.assertTrue(r.model)

    def test_fast_preference_boosts_fast_model(self):
        router = SmartRouter(
            profile_manager=self.mgr,
            speed_preference="fast",
            config_path=Path("/nonexistent"),
        )
        r = router.select_model("direct")
        self.assertEqual(r.model, "speedy:7b")

    def test_quality_preference_boosts_slow_model(self):
        router = SmartRouter(
            profile_manager=self.mgr,
            speed_preference="quality",
            config_path=Path("/nonexistent"),
        )
        r = router.select_model("direct")
        self.assertEqual(r.model, "deep:32b")

    def test_prefer_speed_override(self):
        router = SmartRouter(
            profile_manager=self.mgr,
            speed_preference="balanced",
            config_path=Path("/nonexistent"),
        )
        r = router.select_model("direct", prefer_speed=True)
        self.assertEqual(r.model, "speedy:7b")


# =============================================================================
# TEST: SmartRouter -- override_routing integration
# =============================================================================

class TestSmartRouterOverrideRouting(unittest.TestCase):
    """Tests for the override_routing method used by PipelineRunner."""

    def setUp(self):
        profiles = [
            _make_profile(
                name="coder:30b", speed_tier="medium", quality_tier="high",
                task_scores={"code_python": 0.95},
                recommended_for=["code_python"],
            ),
        ]
        self.mgr = _make_manager_with_profiles(profiles)
        self.router = SmartRouter(
            profile_manager=self.mgr,
            config_path=Path("/nonexistent"),
        )

    def test_override_changes_model(self):
        routing = FakeRouting(model="default-model")
        new_routing = self.router.override_routing(routing, "code_verify")
        self.assertEqual(new_routing.model, "coder:30b")
        self.assertIn("Smart routed", new_routing.explanation)

    def test_override_when_disabled(self):
        self.router.enabled = False
        routing = FakeRouting(model="default-model")
        result = self.router.override_routing(routing, "code_verify")
        # Should return original routing unchanged
        self.assertEqual(result.model, "default-model")

    def test_override_fallback_returns_original(self):
        empty_mgr = _make_manager_with_profiles([])
        router = SmartRouter(
            profile_manager=empty_mgr,
            config_path=Path("/nonexistent"),
        )
        routing = FakeRouting(model="original")
        result = router.override_routing(routing, "code_verify")
        self.assertEqual(result.model, "original")


# =============================================================================
# TEST: SmartRouter -- Task score computation
# =============================================================================

class TestSmartRouterTaskScoreComputation(unittest.TestCase):
    """Tests for _compute_task_score internal method."""

    def setUp(self):
        self.profile = _make_profile(
            name="test:7b",
            task_scores={
                "code_python": 0.95,
                "code": 0.80,
                "general": 0.50,
            },
            recommended_for=["code_python", "general"],
        )
        self.mgr = _make_manager_with_profiles([self.profile])
        self.router = SmartRouter(
            profile_manager=self.mgr,
            config_path=Path("/nonexistent"),
        )

    def test_exact_match(self):
        score = self.router._compute_task_score(self.profile, "code_python")
        self.assertEqual(score, 0.95)

    def test_prefix_match(self):
        score = self.router._compute_task_score(self.profile, "code_r")
        # "code" prefix matches "code_r"
        self.assertEqual(score, 0.80)

    def test_no_match_fallback(self):
        score = self.router._compute_task_score(self.profile, "vision")
        # No task_score match, falls back to score_for_task (profile-based)
        self.assertGreaterEqual(score, 0.0)

    def test_empty_task_scores(self):
        profile = _make_profile(
            name="bare:7b", task_scores={},
            recommended_for=["general"],
        )
        score = self.router._compute_task_score(profile, "general")
        # Falls back to score_for_task
        self.assertGreater(score, 0.0)


# =============================================================================
# TEST: PIPELINE_TO_TASK_MAPPING constants
# =============================================================================

class TestPipelineToTaskMapping(unittest.TestCase):
    """Validate the mapping constants."""

    def test_all_standard_step_types_mapped(self):
        expected = [
            "direct", "tools", "code_verify", "think",
            "web_search", "think_tools", "reasoning",
            "consensus", "self_correct",
        ]
        for st in expected:
            self.assertIn(st, PIPELINE_TO_TASK_MAPPING, f"Missing mapping for {st}")

    def test_mapping_values_are_lists(self):
        for st, tasks in PIPELINE_TO_TASK_MAPPING.items():
            self.assertIsInstance(tasks, list, f"Mapping for {st} is not a list")
            self.assertGreater(len(tasks), 0, f"Mapping for {st} is empty")


# =============================================================================
# TEST: Default constants
# =============================================================================

class TestDefaultConstants(unittest.TestCase):
    """Validate default constant values."""

    def test_speed_weights_all_tiers(self):
        for tier in VALID_SPEED_TIERS:
            self.assertIn(tier, DEFAULT_SPEED_WEIGHTS)

    def test_context_requirements_all_steps(self):
        for st in PIPELINE_TO_TASK_MAPPING:
            self.assertIn(st, DEFAULT_CONTEXT_REQUIREMENTS)

    def test_context_requirements_positive(self):
        for st, ctx in DEFAULT_CONTEXT_REQUIREMENTS.items():
            self.assertGreater(ctx, 0, f"Context for {st} should be positive")


# =============================================================================
# TEST: Integration with real YAML profiles
# =============================================================================

class TestSmartRouterWithRealProfiles(unittest.TestCase):
    """Integration tests loading actual project YAML profiles."""

    @classmethod
    def setUpClass(cls):
        yaml_path = _PROJECT_ROOT / "opti_oignon" / "config" / "model_profiles.yaml"
        if not yaml_path.exists():
            raise unittest.SkipTest("model_profiles.yaml not found")
        cls.mgr = ModelProfileManager(profiles_path=yaml_path)
        cls.mgr.load()
        cls.router = SmartRouter(
            profile_manager=cls.mgr,
            config_path=Path("/nonexistent"),
        )

    def test_profiles_loaded(self):
        self.assertGreater(self.mgr.count, 10)

    def test_code_verify_selects_coder(self):
        r = self.router.select_model("code_verify")
        self.assertIn("coder", r.model.lower())
        self.assertGreater(r.task_score, 0.85)

    def test_reasoning_selects_strong_reasoner(self):
        r = self.router.select_model("reasoning")
        self.assertTrue(r.profile_used)
        self.assertGreater(r.score, 0.5)

    def test_direct_selects_fast_model(self):
        r = self.router.select_model("direct")
        # Should prefer fast or general model
        self.assertTrue(r.profile_used)

    def test_all_step_types_produce_result(self):
        for step_type in PIPELINE_TO_TASK_MAPPING:
            r = self.router.select_model(step_type)
            self.assertTrue(r.model, f"No model for {step_type}")

    def test_pipeline_all_steps(self):
        steps = list(PIPELINE_TO_TASK_MAPPING.keys())
        results = self.router.select_for_pipeline(steps)
        self.assertEqual(len(results), len(steps))
        for st, r in results.items():
            self.assertTrue(r.model, f"No model for pipeline step {st}")

    def test_embedding_model_never_selected(self):
        for step_type in PIPELINE_TO_TASK_MAPPING:
            r = self.router.select_model(step_type)
            self.assertNotIn("embed", r.model.lower())


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    unittest.main()
