"""
Tests for S86 -- Humanizer Mode (SQ-03).

Validates:
- HumanizerConfig dataclass and YAML loading
- Rule-based strategies: vocabulary, filler, contractions, hedging
- HumanizerEngine: init, config access, update, humanize, modes
- HumanizerFeedbackDB: store comparison, store rating, get stats
- API routes: schemas, route definitions, endpoint signatures
- Frontend: types, API client, stores, HumanizerPanel, ChatControlBar
- app.py version bump to 1.8.8, deps.py HUMANIZER_AVAILABLE flag
- No regressions on existing test conventions
"""

import ast
import copy
import glob
import os
import re
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'opti_oignon')
API_DIR = os.path.join(BACKEND_DIR, 'api')
FRONTEND_SRC = os.path.join(PROJECT_ROOT, 'frontend', 'src')
CONFIG_DIR = os.path.join(BACKEND_DIR, 'config')


def _read(path):
    """Read file content safely."""
    with open(path, encoding='utf-8') as f:
        return f.read()


def _load_module_from_file(name, filepath):
    """Load a Python module directly from file path, bypassing __init__.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    return mod, spec


# ---------------------------------------------------------------------------
# Mock setup for humanizer module
# ---------------------------------------------------------------------------

def _setup_mocks():
    """Set up minimal mocks so humanizer can be imported standalone."""
    import yaml
    # Mock ollama
    if 'ollama' not in sys.modules:
        mock_ollama = types.ModuleType('ollama')
        mock_ollama.generate = MagicMock(return_value={'response': 'mocked rewrite'})
        mock_ollama.list = MagicMock(return_value={'models': []})
        sys.modules['ollama'] = mock_ollama


def _get_humanizer_module():
    """Import humanizer module with mocks."""
    _setup_mocks()
    mod, spec = _load_module_from_file(
        'humanizer', os.path.join(BACKEND_DIR, 'humanizer.py')
    )
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# 1. YAML CONFIG
# ===========================================================================

class TestHumanizerYAML(unittest.TestCase):
    """Validate humanizer.yaml structure and content."""

    def setUp(self):
        import yaml
        self.config_path = os.path.join(CONFIG_DIR, 'humanizer.yaml')
        self.assertTrue(os.path.exists(self.config_path), "humanizer.yaml must exist")
        with open(self.config_path, encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

    def test_yaml_has_required_keys(self):
        required = [
            'enabled', 'mode', 'intensity', 'formality', 'rewrite_model',
            'max_input_length', 'banned_phrases', 'vocabulary_replacements',
            'contractions', 'hedging_excess', 'feedback_db',
        ]
        for key in required:
            self.assertIn(key, self.cfg, f"Missing key: {key}")

    def test_enabled_is_false_by_default(self):
        self.assertFalse(self.cfg['enabled'])

    def test_mode_is_rewrite(self):
        self.assertEqual(self.cfg['mode'], 'rewrite')

    def test_intensity_is_moderate(self):
        self.assertEqual(self.cfg['intensity'], 'moderate')

    def test_formality_is_neutral(self):
        self.assertEqual(self.cfg['formality'], 'neutral')

    def test_banned_phrases_is_list(self):
        self.assertIsInstance(self.cfg['banned_phrases'], list)
        self.assertGreater(len(self.cfg['banned_phrases']), 5)

    def test_vocabulary_replacements_is_dict(self):
        self.assertIsInstance(self.cfg['vocabulary_replacements'], dict)
        self.assertIn('crucial', self.cfg['vocabulary_replacements'])
        self.assertIn('delve', self.cfg['vocabulary_replacements'])
        self.assertIn('leverage', self.cfg['vocabulary_replacements'])

    def test_contractions_is_dict(self):
        self.assertIsInstance(self.cfg['contractions'], dict)
        self.assertIn('it is', self.cfg['contractions'])
        self.assertIn('do not', self.cfg['contractions'])

    def test_hedging_excess_is_list(self):
        self.assertIsInstance(self.cfg['hedging_excess'], list)
        self.assertGreater(len(self.cfg['hedging_excess']), 2)


# ===========================================================================
# 2. RULE-BASED STRATEGIES
# ===========================================================================

class TestVocabularyReplacements(unittest.TestCase):
    """Test vocabulary diversity strategy."""

    def setUp(self):
        self.mod = _get_humanizer_module()

    def test_replaces_overused_words(self):
        text = "This is a crucial step to leverage modern tools."
        result, count = self.mod._apply_vocabulary_replacements(
            text, {"crucial": "important", "leverage": "use"}
        )
        self.assertIn("important", result)
        self.assertIn("use", result)
        self.assertNotIn("crucial", result.lower())
        self.assertEqual(count, 2)

    def test_preserves_capitalization(self):
        text = "Crucial decisions require Leverage of resources."
        result, count = self.mod._apply_vocabulary_replacements(
            text, {"crucial": "important", "leverage": "use"}
        )
        self.assertTrue(result.startswith("Important"))
        self.assertEqual(count, 2)

    def test_no_partial_replacement(self):
        text = "The cruciality of this is undeniable."
        result, count = self.mod._apply_vocabulary_replacements(
            text, {"crucial": "important"}
        )
        # "cruciality" should not be affected by word-boundary matching
        self.assertIn("cruciality", result)
        self.assertEqual(count, 0)

    def test_empty_replacements(self):
        text = "Nothing to replace here."
        result, count = self.mod._apply_vocabulary_replacements(text, {})
        self.assertEqual(result, text)
        self.assertEqual(count, 0)


class TestFillerReduction(unittest.TestCase):
    """Test banned phrase stripping."""

    def setUp(self):
        self.mod = _get_humanizer_module()

    def test_strips_banned_phrases(self):
        text = "It's worth noting that the sky is blue. In conclusion, water is wet."
        result, count = self.mod._strip_banned_phrases(
            text, ["It's worth noting", "In conclusion"]
        )
        self.assertNotIn("It's worth noting", result)
        self.assertNotIn("In conclusion", result)
        self.assertIn("the sky is blue", result)
        self.assertEqual(count, 2)

    def test_case_insensitive(self):
        text = "IT'S WORTH NOTING that tests matter."
        result, count = self.mod._strip_banned_phrases(
            text, ["It's worth noting"]
        )
        self.assertNotIn("IT'S WORTH NOTING", result)
        self.assertEqual(count, 1)

    def test_no_double_spaces_after_removal(self):
        text = "It's worth noting the sky is blue."
        result, _ = self.mod._strip_banned_phrases(
            text, ["It's worth noting"]
        )
        self.assertNotIn("  ", result)


class TestContractionInjection(unittest.TestCase):
    """Test contraction injection with formality levels."""

    def setUp(self):
        self.mod = _get_humanizer_module()
        self.contractions = {"it is": "it's", "do not": "don't"}

    def test_casual_contracts_all(self):
        text = "It is important. Do not forget."
        result, count = self.mod._apply_contractions(
            text, self.contractions, "casual"
        )
        self.assertIn("It's", result)
        self.assertIn("Don't", result)
        self.assertEqual(count, 2)

    def test_neutral_skips_first_sentence(self):
        text = "It is important. Do not forget that it is key."
        result, count = self.mod._apply_contractions(
            text, self.contractions, "neutral"
        )
        # First sentence unchanged
        self.assertTrue(result.startswith("It is important."))
        # Second sentence contracted
        self.assertIn("Don't", result)

    def test_formal_no_contractions(self):
        text = "It is important. Do not forget."
        result, count = self.mod._apply_contractions(
            text, self.contractions, "formal"
        )
        self.assertEqual(result, text)
        self.assertEqual(count, 0)


class TestHedgingReduction(unittest.TestCase):
    """Test hedging calibration."""

    def setUp(self):
        self.mod = _get_humanizer_module()

    def test_reduces_hedging(self):
        text = "I think that perhaps the answer is 42."
        result, count = self.mod._reduce_hedging(
            text, ["I think that perhaps"]
        )
        self.assertNotIn("I think that perhaps", result)
        self.assertEqual(count, 1)

    def test_empty_hedging_list(self):
        text = "Some text here."
        result, count = self.mod._reduce_hedging(text, [])
        self.assertEqual(result, text)
        self.assertEqual(count, 0)


# ===========================================================================
# 3. HUMANIZER ENGINE
# ===========================================================================

class TestHumanizerEngine(unittest.TestCase):
    """Test HumanizerEngine class."""

    def setUp(self):
        self.mod = _get_humanizer_module()

    def test_engine_initializes(self):
        engine = self.mod.HumanizerEngine()
        self.assertIsNotNone(engine)

    def test_engine_default_config(self):
        engine = self.mod.HumanizerEngine()
        cfg = engine.get_config()
        self.assertIn('enabled', cfg)
        self.assertIn('mode', cfg)
        self.assertIn('intensity', cfg)
        self.assertIn('formality', cfg)

    def test_engine_enabled_property(self):
        engine = self.mod.HumanizerEngine()
        self.assertIsInstance(engine.enabled, bool)

    def test_update_config_valid(self):
        engine = self.mod.HumanizerEngine()
        cfg = engine.update_config(enabled=True, mode="hybrid", intensity="heavy")
        self.assertTrue(cfg['enabled'])
        self.assertEqual(cfg['mode'], 'hybrid')
        self.assertEqual(cfg['intensity'], 'heavy')

    def test_update_config_invalid_mode_ignored(self):
        engine = self.mod.HumanizerEngine()
        original_mode = engine.get_config()['mode']
        engine.update_config(mode="invalid_mode")
        self.assertEqual(engine.get_config()['mode'], original_mode)

    def test_update_config_invalid_intensity_ignored(self):
        engine = self.mod.HumanizerEngine()
        original = engine.get_config()['intensity']
        engine.update_config(intensity="extreme")
        self.assertEqual(engine.get_config()['intensity'], original)

    def test_update_config_invalid_formality_ignored(self):
        engine = self.mod.HumanizerEngine()
        original = engine.get_config()['formality']
        engine.update_config(formality="ultra_casual")
        self.assertEqual(engine.get_config()['formality'], original)

    def test_humanize_short_text_passthrough(self):
        engine = self.mod.HumanizerEngine()
        result = engine.humanize("Hi")
        self.assertEqual(result.original, "Hi")
        self.assertEqual(result.humanized, "Hi")
        self.assertEqual(result.strategies_applied, [])

    def test_humanize_too_long_text_passthrough(self):
        engine = self.mod.HumanizerEngine()
        engine.update_config(max_input_length=100)
        long_text = "A " * 200
        result = engine.humanize(long_text)
        self.assertEqual(result.humanized, long_text)

    def test_humanize_logprobs_mode_applies_rules(self):
        engine = self.mod.HumanizerEngine()
        engine.update_config(enabled=True)
        text = "It is crucial to leverage these tools. It's worth noting that they work."
        result = engine.humanize(text, mode="logprobs")
        self.assertEqual(result.mode, "logprobs")
        # Should have applied at least vocabulary or filler rules
        self.assertNotEqual(result.humanized, text)

    def test_humanize_returns_comparison_id(self):
        engine = self.mod.HumanizerEngine()
        result = engine.humanize("This is a test with enough text to process properly for testing purposes.")
        self.assertTrue(len(result.comparison_id) > 0)

    def test_humanize_result_to_dict(self):
        engine = self.mod.HumanizerEngine()
        result = engine.humanize("Some longer text with crucial information to delve into for testing.")
        d = result.to_dict()
        self.assertIn('original', d)
        self.assertIn('humanized', d)
        self.assertIn('strategies_applied', d)
        self.assertIn('comparison_id', d)
        self.assertIn('latency_ms', d)


# ===========================================================================
# 4. FEEDBACK DATABASE
# ===========================================================================

class TestHumanizerFeedbackDB(unittest.TestCase):
    """Test SQLite feedback storage."""

    def setUp(self):
        self.mod = _get_humanizer_module()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.tmp.close()
        self.db = self.mod.HumanizerFeedbackDB(Path(self.tmp.name))

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_db_creates_tables(self):
        conn = sqlite3.connect(self.tmp.name)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn('comparisons', table_names)
        self.assertIn('ratings', table_names)
        conn.close()

    def test_store_comparison(self):
        ok = self.db.store_comparison(
            comparison_id="test-1",
            original="original text",
            humanized="humanized text",
            strategies=["vocabulary_diversity"],
            model="test-model",
            intensity="moderate",
            mode="rewrite",
        )
        self.assertTrue(ok)

    def test_store_rating_valid(self):
        self.db.store_comparison(
            "test-2", "orig", "human", ["filler"], "model", "light", "rewrite",
        )
        ok = self.db.store_rating("test-2", "humanized")
        self.assertTrue(ok)

    def test_store_rating_invalid_winner(self):
        self.db.store_comparison(
            "test-3", "orig", "human", [], "model", "light", "rewrite",
        )
        ok = self.db.store_rating("test-3", "invalid_winner")
        self.assertFalse(ok)

    def test_store_rating_nonexistent_comparison(self):
        ok = self.db.store_rating("nonexistent-id", "humanized")
        self.assertFalse(ok)

    def test_get_stats_empty(self):
        stats = self.db.get_stats()
        self.assertEqual(stats.total_ratings, 0)
        self.assertEqual(stats.win_rate, 0.0)

    def test_get_stats_with_data(self):
        self.db.store_comparison(
            "s1", "orig", "human", ["vocab"], "model-a", "moderate", "rewrite",
        )
        self.db.store_comparison(
            "s2", "orig2", "human2", ["filler"], "model-b", "light", "hybrid",
        )
        self.db.store_rating("s1", "humanized")
        self.db.store_rating("s2", "original")
        self.db.store_rating("s1", "tie")

        stats = self.db.get_stats()
        self.assertEqual(stats.total_ratings, 3)
        self.assertEqual(stats.humanized_wins, 1)
        self.assertEqual(stats.original_wins, 1)
        self.assertEqual(stats.ties, 1)
        self.assertAlmostEqual(stats.win_rate, 1 / 3, places=2)

    def test_stats_by_strategy(self):
        self.db.store_comparison(
            "s1", "o", "h", ["vocabulary_diversity"], "m", "moderate", "rewrite",
        )
        self.db.store_rating("s1", "humanized")

        stats = self.db.get_stats()
        self.assertIn("vocabulary_diversity", stats.by_strategy)
        self.assertEqual(stats.by_strategy["vocabulary_diversity"]["humanized"], 1)

    def test_stats_by_model(self):
        self.db.store_comparison(
            "s1", "o", "h", ["vocab"], "qwen3:32b", "moderate", "rewrite",
        )
        self.db.store_rating("s1", "original")

        stats = self.db.get_stats()
        self.assertIn("qwen3:32b", stats.by_model)

    def test_stats_by_intensity(self):
        self.db.store_comparison(
            "s1", "o", "h", ["vocab"], "m", "heavy", "rewrite",
        )
        self.db.store_rating("s1", "humanized")

        stats = self.db.get_stats()
        self.assertIn("heavy", stats.by_intensity)


# ===========================================================================
# 5. DATA CLASSES
# ===========================================================================

class TestDataClasses(unittest.TestCase):
    """Test humanizer data classes."""

    def setUp(self):
        self.mod = _get_humanizer_module()

    def test_humanizer_config_to_dict(self):
        cfg = self.mod.HumanizerConfig()
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn('enabled', d)
        self.assertIn('mode', d)

    def test_humanizer_result_to_dict(self):
        result = self.mod.HumanizerResult(original="a", humanized="b")
        d = result.to_dict()
        self.assertEqual(d['original'], "a")
        self.assertEqual(d['humanized'], "b")

    def test_feedback_stats_to_dict(self):
        stats = self.mod.FeedbackStats(total_ratings=5, humanized_wins=3)
        d = stats.to_dict()
        self.assertEqual(d['total_ratings'], 5)
        self.assertEqual(d['humanized_wins'], 3)


# ===========================================================================
# 6. API SCHEMAS
# ===========================================================================

class TestHumanizerSchemas(unittest.TestCase):
    """Validate humanizer Pydantic schemas exist in schemas.py."""

    def setUp(self):
        self.src = _read(os.path.join(API_DIR, 'schemas.py'))

    def test_rewrite_request_schema(self):
        self.assertIn('class HumanizerRewriteRequest', self.src)

    def test_rewrite_response_schema(self):
        self.assertIn('class HumanizerRewriteResponse', self.src)

    def test_config_response_schema(self):
        self.assertIn('class HumanizerConfigResponse', self.src)

    def test_config_update_schema(self):
        self.assertIn('class HumanizerConfigUpdate', self.src)

    def test_feedback_request_schema(self):
        self.assertIn('class HumanizerFeedbackRequest', self.src)

    def test_feedback_response_schema(self):
        self.assertIn('class HumanizerFeedbackResponse', self.src)

    def test_strategy_stats_schema(self):
        self.assertIn('class HumanizerStrategyStats', self.src)

    def test_stats_response_schema(self):
        self.assertIn('class HumanizerStatsResponse', self.src)


# ===========================================================================
# 7. API ROUTES
# ===========================================================================

class TestHumanizerRoutes(unittest.TestCase):
    """Validate humanizer route definitions."""

    def setUp(self):
        self.src = _read(os.path.join(API_DIR, 'routes_humanizer.py'))

    def test_router_prefix(self):
        self.assertIn('prefix="/api/humanizer"', self.src)

    def test_rewrite_endpoint(self):
        self.assertIn('"/rewrite"', self.src)
        self.assertIn('def rewrite_text', self.src)

    def test_get_config_endpoint(self):
        self.assertIn('"/config"', self.src)
        self.assertIn('def get_humanizer_config', self.src)

    def test_update_config_endpoint(self):
        self.assertIn('def update_humanizer_config', self.src)

    def test_feedback_endpoint(self):
        self.assertIn('"/feedback"', self.src)
        self.assertIn('def submit_feedback', self.src)

    def test_stats_endpoint(self):
        self.assertIn('"/stats"', self.src)
        self.assertIn('def get_humanizer_stats', self.src)


# ===========================================================================
# 8. APP.PY INTEGRATION
# ===========================================================================

class TestAppIntegration(unittest.TestCase):
    """Validate app.py changes for S86."""

    def setUp(self):
        self.src = _read(os.path.join(API_DIR, 'app.py'))

    def test_version_bump(self):
        self.assertIn('version="1.8.9"', self.src)

    def test_health_version(self):
        self.assertIn('"version": "1.8.9"', self.src)

    def test_humanizer_router_import(self):
        self.assertIn('from .routes_humanizer import router as humanizer_router', self.src)

    def test_humanizer_router_registered(self):
        self.assertIn('app.include_router(humanizer_router)', self.src)

    def test_humanizer_in_health_check(self):
        self.assertIn('"humanizer": HUMANIZER_AVAILABLE', self.src)

    def test_humanizer_available_imported(self):
        self.assertIn('HUMANIZER_AVAILABLE', self.src)


# ===========================================================================
# 9. DEPS.PY
# ===========================================================================

class TestDepsIntegration(unittest.TestCase):
    """Validate deps.py changes for S86."""

    def setUp(self):
        self.src = _read(os.path.join(API_DIR, 'deps.py'))

    def test_humanizer_import_block(self):
        self.assertIn('from opti_oignon.humanizer import humanizer_engine', self.src)
        self.assertIn('HUMANIZER_AVAILABLE', self.src)

    def test_humanizer_fallback(self):
        self.assertIn('humanizer_engine = None', self.src)


# ===========================================================================
# 10. FRONTEND TYPES
# ===========================================================================

class TestFrontendTypes(unittest.TestCase):
    """Validate TypeScript interfaces for humanizer."""

    def setUp(self):
        self.src = _read(os.path.join(FRONTEND_SRC, 'lib', 'types.ts'))

    def test_rewrite_request_interface(self):
        self.assertIn('interface HumanizerRewriteRequest', self.src)

    def test_rewrite_response_interface(self):
        self.assertIn('interface HumanizerRewriteResponse', self.src)

    def test_config_response_interface(self):
        self.assertIn('interface HumanizerConfigResponse', self.src)

    def test_config_update_interface(self):
        self.assertIn('interface HumanizerConfigUpdate', self.src)

    def test_feedback_request_interface(self):
        self.assertIn('interface HumanizerFeedbackRequest', self.src)

    def test_stats_response_interface(self):
        self.assertIn('interface HumanizerStatsResponse', self.src)

    def test_strategy_stats_interface(self):
        self.assertIn('interface HumanizerStrategyStats', self.src)


# ===========================================================================
# 11. FRONTEND API CLIENT
# ===========================================================================

class TestFrontendAPIClient(unittest.TestCase):
    """Validate humanizer.ts API client."""

    def setUp(self):
        self.src = _read(os.path.join(FRONTEND_SRC, 'lib', 'api', 'humanizer.ts'))

    def test_rewrite_function(self):
        self.assertIn('export async function rewriteText', self.src)
        self.assertIn('/api/humanizer/rewrite', self.src)

    def test_get_config_function(self):
        self.assertIn('export async function getHumanizerConfig', self.src)
        self.assertIn('/api/humanizer/config', self.src)

    def test_update_config_function(self):
        self.assertIn('export async function updateHumanizerConfig', self.src)

    def test_feedback_function(self):
        self.assertIn('export async function submitHumanizerFeedback', self.src)
        self.assertIn('/api/humanizer/feedback', self.src)

    def test_stats_function(self):
        self.assertIn('export async function getHumanizerStats', self.src)
        self.assertIn('/api/humanizer/stats', self.src)

    def test_uses_typed_client(self):
        self.assertIn("from './client'", self.src)
        self.assertIn('apiGet', self.src)
        self.assertIn('apiPost', self.src)


# ===========================================================================
# 12. FRONTEND STORES
# ===========================================================================

class TestFrontendStores(unittest.TestCase):
    """Validate chatOptions store changes for S86."""

    def setUp(self):
        self.src = _read(os.path.join(FRONTEND_SRC, 'lib', 'stores', 'chatOptions.ts'))

    def test_humanize_enabled_store(self):
        self.assertIn('humanizeEnabled', self.src)
        self.assertIn("writable<boolean>(false)", self.src)

    def test_humanize_in_reset(self):
        self.assertIn('humanizeEnabled.set(false)', self.src)

    def test_humanize_in_get_options(self):
        self.assertIn('humanize?: boolean', self.src)
        self.assertIn("get(humanizeEnabled)", self.src)


# ===========================================================================
# 13. FRONTEND COMPONENTS
# ===========================================================================

class TestHumanizerPanel(unittest.TestCase):
    """Validate HumanizerPanel.svelte."""

    def setUp(self):
        self.src = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'panels', 'HumanizerPanel.svelte'
        ))

    def test_imports_api(self):
        self.assertIn('getHumanizerConfig', self.src)
        self.assertIn('updateHumanizerConfig', self.src)
        self.assertIn('getHumanizerStats', self.src)

    def test_mode_selector(self):
        self.assertIn('LLM Rewrite', self.src)
        self.assertIn('Rule-based', self.src)
        self.assertIn('Hybrid', self.src)

    def test_intensity_selector(self):
        self.assertIn('Light', self.src)
        self.assertIn('Moderate', self.src)
        self.assertIn('Heavy', self.src)

    def test_formality_selector(self):
        self.assertIn('Casual', self.src)
        self.assertIn('Neutral', self.src)
        self.assertIn('Formal', self.src)

    def test_enable_toggle(self):
        self.assertIn('Enable Humanizer', self.src)
        self.assertIn('toggleEnabled', self.src)

    def test_feedback_stats_section(self):
        self.assertIn('Feedback Statistics', self.src)
        self.assertIn('win_rate', self.src)

    def test_banned_phrases_editor(self):
        self.assertIn('Banned Phrases', self.src)
        self.assertIn('localBannedPhrases', self.src)


class TestChatControlBarHumanizer(unittest.TestCase):
    """Validate ChatControlBar.svelte humanizer additions."""

    def setUp(self):
        self.src = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'
        ))

    def test_humanize_enabled_import(self):
        self.assertIn('humanizeEnabled', self.src)

    def test_toggle_humanize_function(self):
        self.assertIn('toggleHumanize', self.src)

    def test_humanize_button(self):
        self.assertIn('Toggle humanizer', self.src)
        self.assertIn('Human', self.src)

    def test_humanizer_config_load(self):
        self.assertIn('/api/humanizer/config', self.src)


class TestSettingsPageHumanizer(unittest.TestCase):
    """Validate settings page humanizer integration."""

    def setUp(self):
        self.src = _read(os.path.join(
            FRONTEND_SRC, 'routes', 'settings', '+page.svelte'
        ))

    def test_humanizer_panel_import(self):
        self.assertIn("import HumanizerPanel from", self.src)

    def test_humanizer_collapsible(self):
        self.assertIn('advHumanizerOpen', self.src)
        self.assertIn('<HumanizerPanel', self.src)

    def test_humanizer_section_label(self):
        self.assertIn('Humanizer', self.src)
        self.assertIn('Post-process LLM output', self.src)


# ===========================================================================
# 14. CODE QUALITY
# ===========================================================================

class TestCodeQuality(unittest.TestCase):
    """Validate code conventions for S86 files."""

    def test_no_french_in_humanizer_py(self):
        src = _read(os.path.join(BACKEND_DIR, 'humanizer.py'))
        french_markers = ['fonction', 'parametre', 'resultat', 'reponse', 'echec']
        for marker in french_markers:
            self.assertNotIn(marker, src.lower(),
                             f"French word '{marker}' found in humanizer.py")

    def test_no_emojis_in_humanizer_py(self):
        src = _read(os.path.join(BACKEND_DIR, 'humanizer.py'))
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
        )
        self.assertIsNone(emoji_pattern.search(src))

    def test_no_french_in_routes(self):
        src = _read(os.path.join(API_DIR, 'routes_humanizer.py'))
        french_markers = ['fonction', 'parametre', 'resultat']
        for marker in french_markers:
            self.assertNotIn(marker, src.lower())

    def test_humanizer_yaml_no_emojis(self):
        src = _read(os.path.join(CONFIG_DIR, 'humanizer.yaml'))
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
        )
        self.assertIsNone(emoji_pattern.search(src))

    def test_version_consistency(self):
        app_src = _read(os.path.join(API_DIR, 'app.py'))
        self.assertEqual(app_src.count('"1.8.9"'), 2,
                         "Version 1.8.9 should appear twice in app.py (FastAPI + health)")


# ===========================================================================
# 15. MODULE SINGLETON
# ===========================================================================

class TestModuleSingleton(unittest.TestCase):
    """Test module-level singleton initialization."""

    def test_humanizer_engine_singleton(self):
        mod = _get_humanizer_module()
        self.assertIsNotNone(mod.humanizer_engine)
        self.assertTrue(mod.HUMANIZER_AVAILABLE)

    def test_valid_modes_constant(self):
        mod = _get_humanizer_module()
        self.assertIn('rewrite', mod.VALID_MODES)
        self.assertIn('logprobs', mod.VALID_MODES)
        self.assertIn('hybrid', mod.VALID_MODES)

    def test_valid_intensities_constant(self):
        mod = _get_humanizer_module()
        self.assertIn('light', mod.VALID_INTENSITIES)
        self.assertIn('moderate', mod.VALID_INTENSITIES)
        self.assertIn('heavy', mod.VALID_INTENSITIES)

    def test_valid_formalities_constant(self):
        mod = _get_humanizer_module()
        self.assertIn('casual', mod.VALID_FORMALITIES)
        self.assertIn('neutral', mod.VALID_FORMALITIES)
        self.assertIn('formal', mod.VALID_FORMALITIES)


if __name__ == '__main__':
    unittest.main()
