#!/usr/bin/env python3
"""
Tests for S109 -- Bug Fix Pass 2 + Embed Auto-detect + Status Streaming + Plugin UX + Auth Toggle + ddgs.

Covers:
- ISSUE-A: system_presets.yaml existence and structure
- ISSUE-B: Embedding auto-detection and legacy fallback
- ISSUE-C: WS status event pipeline
- ISSUE-D: Auth mode toggle endpoint
- ISSUE-E: Plugin usage hints in frontend
- ISSUE-F: StreamingIndicator status display
- ISSUE-G: ddgs dependency rename

~42 tests total.
"""

import ast
import os
import re
import sys
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
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CONFIG_DIR = os.path.join(BACKEND_DIR, 'config')
RAG_DIR = os.path.join(BACKEND_DIR, 'rag')


def _read(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        return f.read()


# ===========================================================================
# ISSUE-A: System presets YAML (data/system_presets.yaml)
# ===========================================================================

class TestSystemPresetsYAMLExists(unittest.TestCase):
    """Verify data/system_presets.yaml is present and valid."""

    @classmethod
    def setUpClass(cls):
        import yaml
        cls.yaml_path = os.path.join(DATA_DIR, 'system_presets.yaml')
        with open(cls.yaml_path) as f:
            cls.data = yaml.safe_load(f)

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(self.yaml_path))

    def test_has_three_presets(self):
        self.assertEqual(set(self.data['system_presets'].keys()), {'minimal', 'balanced', 'power'})

    def test_each_preset_has_pipelines(self):
        for pid, pdata in self.data['system_presets'].items():
            self.assertIn('pipelines', pdata, f"Preset '{pid}' missing 'pipelines'")
            self.assertIsInstance(pdata['pipelines'], list)

    def test_minimal_cache_disabled(self):
        overrides = self.data['system_presets']['minimal']['config_overrides']
        self.assertFalse(overrides['cache']['enabled'])

    def test_balanced_cache_enabled(self):
        overrides = self.data['system_presets']['balanced']['config_overrides']
        self.assertTrue(overrides['cache']['enabled'])

    def test_power_speculative_enabled(self):
        overrides = self.data['system_presets']['power']['config_overrides']
        self.assertTrue(overrides['speculative']['enabled'])

    def test_onboarding_default_false(self):
        self.assertFalse(self.data['onboarding']['user_initialized'])

    def test_power_pipelines_at_least_five(self):
        pipelines = self.data['system_presets']['power']['pipelines']
        self.assertGreaterEqual(len(pipelines), 5)

    def test_model_strategies(self):
        presets = self.data['system_presets']
        self.assertEqual(presets['minimal']['model_strategy'], 'smallest')
        self.assertEqual(presets['balanced']['model_strategy'], 'medium')
        self.assertEqual(presets['power']['model_strategy'], 'largest')


# ===========================================================================
# ISSUE-B: Embedding auto-detection (rag/embeddings.py)
# ===========================================================================

class TestEmbeddingsModuleAST(unittest.TestCase):
    """Verify rag/embeddings.py parses and has required structures."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(RAG_DIR, 'embeddings.py'))

    def test_ast_valid(self):
        ast.parse(self.src)

    def test_no_french_comments(self):
        french = re.findall(r'[àéèùîôêïçÉÀÎ]', self.src)
        self.assertEqual(len(french), 0, f"French chars found: {french[:10]}")

    def test_verify_model_method_exists(self):
        self.assertIn('def _verify_model(self)', self.src)

    def test_auto_detect_embed_keywords(self):
        self.assertIn('embed_keywords', self.src)

    def test_legacy_endpoint_fallback(self):
        self.assertIn('_embed_single_legacy', self.src)
        self.assertIn('/api/embeddings', self.src)

    def test_use_legacy_flag(self):
        self.assertIn('self._use_legacy', self.src)

    def test_400_handling_in_embed_single(self):
        self.assertIn('status_code == 400', self.src)

    def test_400_handling_in_embed_batch(self):
        # Ensure batch also handles 400
        batch_section = self.src[self.src.index('def embed_batch'):]
        self.assertIn('status_code == 400', batch_section)

    def test_full_name_resolution(self):
        # Should use full Ollama name (with :tag)
        self.assertIn('full_names', self.src)


# ===========================================================================
# ISSUE-C/F: WebSocket status event + StreamingIndicator
# ===========================================================================

class TestStatusEventBackend(unittest.TestCase):
    """Verify routes_chat.py emits status events."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(API_DIR, 'routes_chat.py'))

    def test_ast_valid(self):
        ast.parse(self.src)

    def test_status_event_in_on_status(self):
        self.assertIn('"status"', self.src)
        self.assertIn('chunks.append(("status"', self.src)

    def test_status_handler_in_loop(self):
        self.assertIn('event_type == "status"', self.src)

    def test_agentic_receives_on_status(self):
        # on_status should be passed to agentic executor
        self.assertIn('on_status=_on_status,', self.src)
        # Should appear in the agentic block
        agentic_block = self.src[self.src.index('if use_agentic'):]
        self.assertIn('on_status=_on_status', agentic_block)


class TestStatusEventFrontendTypes(unittest.TestCase):
    """Verify types.ts includes status type."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(FRONTEND_SRC, 'lib', 'types.ts'))

    def test_chat_token_has_status(self):
        self.assertIn("'status'", self.src)

    def test_callbacks_has_on_status(self):
        self.assertIn('onStatus?', self.src)


class TestStatusEventFrontendChat(unittest.TestCase):
    """Verify chat.ts handles status events."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(FRONTEND_SRC, 'lib', 'api', 'chat.ts'))

    def test_status_case_in_switch(self):
        self.assertIn("case 'status':", self.src)

    def test_calls_on_status(self):
        self.assertIn('callbacks.onStatus', self.src)


class TestStreamingStatusStore(unittest.TestCase):
    """Verify chat store has streamingStatus."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(FRONTEND_SRC, 'lib', 'stores', 'chat.ts'))

    def test_streaming_status_writable(self):
        self.assertIn('streamingStatus', self.src)
        self.assertIn("writable<string | null>(null)", self.src)

    def test_status_reset_on_done(self):
        self.assertIn('streamingStatus.set(null)', self.src)


class TestStreamingIndicator(unittest.TestCase):
    """Verify StreamingIndicator shows status text."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'StreamingIndicator.svelte'))

    def test_imports_streaming_status(self):
        self.assertIn('streamingStatus', self.src)

    def test_displays_status_text(self):
        self.assertIn('$streamingStatus', self.src)

    def test_no_hardcoded_hex(self):
        hex_matches = re.findall(r'#[0-9a-fA-F]{3,8}\b', self.src)
        self.assertEqual(len(hex_matches), 0, f"Hardcoded hex: {hex_matches}")

    def test_uses_css_variables(self):
        self.assertIn('var(--oo-', self.src)


# ===========================================================================
# ISSUE-D: Auth mode toggle
# ===========================================================================

class TestAuthModeEndpoint(unittest.TestCase):
    """Verify routes_auth.py has the mode toggle endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(API_DIR, 'routes_auth.py'))

    def test_ast_valid(self):
        ast.parse(self.src)

    def test_auth_mode_request_schema(self):
        self.assertIn('class AuthModeRequest', self.src)
        self.assertIn('single_user_mode', self.src)

    def test_put_mode_endpoint(self):
        self.assertIn('@router.put("/mode")', self.src)
        self.assertIn('def set_auth_mode', self.src)

    def test_persists_to_yaml(self):
        self.assertIn('yaml.dump', self.src)
        self.assertIn('auth.yaml', self.src)


class TestAuthToggleFrontend(unittest.TestCase):
    """Verify settings page has auth toggle UI."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(
            FRONTEND_SRC, 'routes', 'settings', '+page.svelte'))

    def test_auth_section_exists(self):
        self.assertIn('Authentication', self.src)
        self.assertIn('advAuthOpen', self.src)

    def test_toggle_auth_mode_function(self):
        self.assertIn('toggleAuthMode', self.src)
        self.assertIn('/api/auth/mode', self.src)

    def test_fetches_auth_status_on_mount(self):
        self.assertIn('/api/auth/status', self.src)
        self.assertIn('singleUserMode', self.src)

    def test_no_hardcoded_hex(self):
        hex_matches = re.findall(r'#[0-9a-fA-F]{3,8}\b', self.src)
        self.assertEqual(len(hex_matches), 0, f"Hardcoded hex: {hex_matches}")


# ===========================================================================
# ISSUE-E: Plugin usage hints
# ===========================================================================

class TestPluginsQuickPanelHints(unittest.TestCase):
    """Verify PluginsQuickPanel has usage hints."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'panels', 'PluginsQuickPanel.svelte'))

    def test_slash_commands_map(self):
        self.assertIn('SLASH_COMMANDS', self.src)
        self.assertIn('/note', self.src)
        self.assertIn('/tasks', self.src)

    def test_usage_hint_function(self):
        self.assertIn('function usageHint', self.src)

    def test_runs_automatically_text(self):
        self.assertIn('Runs automatically', self.src)

    def test_no_hardcoded_hex(self):
        hex_matches = re.findall(r'#[0-9a-fA-F]{3,8}\b', self.src)
        self.assertEqual(len(hex_matches), 0, f"Hardcoded hex: {hex_matches}")


class TestPluginsPanelHints(unittest.TestCase):
    """Verify PluginsPanel has detailed usage hints."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'settings', 'PluginsPanel.svelte'))

    def test_slash_commands_map_detailed(self):
        self.assertIn('SLASH_COMMANDS', self.src)
        self.assertIn('/note <text>', self.src)
        self.assertIn('/gh issues', self.src)
        self.assertIn('/summary', self.src)

    def test_how_to_use_section(self):
        self.assertIn('How to use', self.src)

    def test_hint_types(self):
        self.assertIn("type: 'commands'", self.src)
        self.assertIn("type: 'auto'", self.src)

    def test_no_hardcoded_hex(self):
        hex_matches = re.findall(r'#[0-9a-fA-F]{3,8}\b', self.src)
        self.assertEqual(len(hex_matches), 0, f"Hardcoded hex: {hex_matches}")


# ===========================================================================
# ISSUE-G: ddgs dependency rename
# ===========================================================================

class TestDdgsDependencyRename(unittest.TestCase):
    """Verify ddgs package is used instead of duckduckgo_search."""

    def test_pyproject_uses_ddgs(self):
        src = _read(os.path.join(PROJECT_ROOT, 'pyproject.toml'))
        self.assertIn('ddgs>=7.0.0', src)
        self.assertNotIn('duckduckgo_search>=6.0.0', src)

    def test_web_search_tries_ddgs_first(self):
        src = _read(os.path.join(BACKEND_DIR, 'web_search.py'))
        ddgs_idx = src.index('from ddgs import DDGS')
        # Legacy import should come after
        legacy_idx = src.index('from duckduckgo_search import DDGS')
        self.assertLess(ddgs_idx, legacy_idx)

    def test_web_search_no_stray_timeout(self):
        src = _read(os.path.join(BACKEND_DIR, 'web_search.py'))
        # The old stray "TimeoutException = Exception" after except block should be gone
        lines = src.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == 'TimeoutException = Exception':
                # Should only appear inside try/except blocks (indented)
                self.assertTrue(line.startswith('    ') or line.startswith('\t'),
                                f"Stray TimeoutException at line {i+1}: '{line}'")

    def test_web_search_ast_valid(self):
        src = _read(os.path.join(BACKEND_DIR, 'web_search.py'))
        ast.parse(src)

    def test_error_messages_reference_ddgs(self):
        src = _read(os.path.join(BACKEND_DIR, 'web_search.py'))
        self.assertIn('pip install ddgs', src)


# ===========================================================================
# General regression checks
# ===========================================================================

class TestNoEmojiInCode(unittest.TestCase):
    """No emojis in modified Python files."""

    def _check_no_emoji(self, path):
        src = _read(path)
        emoji_re = re.compile(
            r'[\U0001F300-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
            r'\U00002702-\U000027B0\U0000FE00-\U0000FE0F\U0001F000-\U0001F02F]'
        )
        matches = emoji_re.findall(src)
        self.assertEqual(len(matches), 0, f"Emoji in {path}: {matches[:5]}")

    def test_embeddings_no_emoji(self):
        self._check_no_emoji(os.path.join(RAG_DIR, 'embeddings.py'))

    def test_routes_chat_no_emoji(self):
        self._check_no_emoji(os.path.join(API_DIR, 'routes_chat.py'))

    def test_routes_auth_no_emoji(self):
        self._check_no_emoji(os.path.join(API_DIR, 'routes_auth.py'))

    def test_web_search_no_emoji(self):
        self._check_no_emoji(os.path.join(BACKEND_DIR, 'web_search.py'))


if __name__ == '__main__':
    unittest.main()
