"""
Tests for S84 — Presets, Simple Mode & Onboarding UX.

Validates:
- System presets YAML loading (3 presets: minimal, balanced, power)
- SystemPresetsManager: list, get, detect, recommend, apply, onboarding state
- Deep merge helper
- Model detection helpers (parse parameter count, quantization, family)
- API routes: schemas, route definitions
- Frontend: types, API client, stores, OnboardingOverlay, settings page
- ChatControlBar: Onion optimization button
- app.py version bump to 1.8.6
- deps.py: SYSTEM_PRESETS_AVAILABLE flag
- No regressions on existing test conventions
"""

import ast
import copy
import glob
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
# Mock setup for system_presets module
# ---------------------------------------------------------------------------

def _setup_mocks():
    """Set up minimal mocks so system_presets can be imported."""
    import yaml

    config_mod = types.ModuleType('opti_oignon.config')
    config_mod.CONFIG_DIR = Path(CONFIG_DIR)
    config_mod.DATA_DIR = Path(DATA_DIR)

    def load_yaml(filepath):
        if not filepath.exists():
            return {}
        with open(filepath) as f:
            return yaml.safe_load(f) or {}

    def save_yaml(filepath, data):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True

    config_mod.load_yaml = load_yaml
    config_mod.save_yaml = save_yaml

    if 'opti_oignon' not in sys.modules:
        sys.modules['opti_oignon'] = types.ModuleType('opti_oignon')
    sys.modules['opti_oignon.config'] = config_mod

    return config_mod


def _load_system_presets_module():
    """Load system_presets module with mocked dependencies."""
    _setup_mocks()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'opti_oignon.system_presets',
        os.path.join(BACKEND_DIR, 'system_presets.py'),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# TEST CLASSES
# ===========================================================================

class TestSystemPresetsYAML(unittest.TestCase):
    """Validate data/system_presets.yaml structure and content."""

    @classmethod
    def setUpClass(cls):
        import yaml
        yaml_path = os.path.join(DATA_DIR, 'system_presets.yaml')
        with open(yaml_path) as f:
            cls.data = yaml.safe_load(f)

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(DATA_DIR, 'system_presets.yaml')))

    def test_has_system_presets_key(self):
        self.assertIn('system_presets', self.data)

    def test_three_presets_defined(self):
        presets = self.data['system_presets']
        self.assertEqual(set(presets.keys()), {'minimal', 'balanced', 'power'})

    def test_each_preset_has_required_fields(self):
        required = ['name', 'description', 'icon', 'recommended_vram_gb',
                     'recommended_ram_gb', 'config_overrides', 'model_strategy', 'pipelines']
        for pid, pdata in self.data['system_presets'].items():
            for field in required:
                self.assertIn(field, pdata, f"Preset '{pid}' missing field '{field}'")

    def test_minimal_has_cache_disabled(self):
        overrides = self.data['system_presets']['minimal']['config_overrides']
        self.assertFalse(overrides['cache']['enabled'])

    def test_balanced_has_cache_enabled(self):
        overrides = self.data['system_presets']['balanced']['config_overrides']
        self.assertTrue(overrides['cache']['enabled'])

    def test_power_has_speculative_enabled(self):
        overrides = self.data['system_presets']['power']['config_overrides']
        self.assertTrue(overrides['speculative']['enabled'])

    def test_model_strategies(self):
        presets = self.data['system_presets']
        self.assertEqual(presets['minimal']['model_strategy'], 'smallest')
        self.assertEqual(presets['balanced']['model_strategy'], 'medium')
        self.assertEqual(presets['power']['model_strategy'], 'largest')

    def test_onboarding_section_exists(self):
        self.assertIn('onboarding', self.data)
        self.assertIn('user_initialized', self.data['onboarding'])

    def test_onboarding_default_false(self):
        self.assertFalse(self.data['onboarding']['user_initialized'])

    def test_minimal_has_only_direct_pipeline(self):
        self.assertEqual(self.data['system_presets']['minimal']['pipelines'], ['direct'])

    def test_power_has_all_pipelines(self):
        pipelines = self.data['system_presets']['power']['pipelines']
        self.assertGreaterEqual(len(pipelines), 5)
        self.assertIn('direct', pipelines)
        self.assertIn('web_search', pipelines)


class TestSystemPresetsModule(unittest.TestCase):
    """Test SystemPresetsManager class functionality."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_system_presets_module()

    def test_module_available(self):
        self.assertTrue(self.mod.SYSTEM_PRESETS_AVAILABLE)

    def test_manager_instance_exists(self):
        self.assertIsNotNone(self.mod.system_presets_manager)

    def test_list_presets_returns_three(self):
        presets = self.mod.system_presets_manager.list_presets()
        self.assertEqual(len(presets), 3)

    def test_list_presets_order(self):
        presets = self.mod.system_presets_manager.list_presets()
        ids = [p.id for p in presets]
        self.assertEqual(ids, ['minimal', 'balanced', 'power'])

    def test_get_preset_by_id(self):
        p = self.mod.system_presets_manager.get_preset('balanced')
        self.assertIsNotNone(p)
        self.assertEqual(p.name, 'Balanced')

    def test_get_preset_unknown_returns_none(self):
        p = self.mod.system_presets_manager.get_preset('nonexistent')
        self.assertIsNone(p)

    def test_preset_to_dict(self):
        p = self.mod.system_presets_manager.get_preset('minimal')
        d = p.to_dict()
        self.assertEqual(d['id'], 'minimal')
        self.assertIn('config_overrides', d)
        self.assertIn('pipelines', d)

    def test_is_initialized_default_false(self):
        self.assertFalse(self.mod.system_presets_manager.is_initialized())

    def test_onboarding_state_dict(self):
        state = self.mod.system_presets_manager.get_onboarding_state()
        self.assertIn('user_initialized', state)
        self.assertIn('applied_preset', state)
        self.assertIn('applied_at', state)


class TestDeepMerge(unittest.TestCase):
    """Test _deep_merge helper."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_system_presets_module()

    def test_simple_merge(self):
        result = self.mod._deep_merge({'a': 1}, {'b': 2})
        self.assertEqual(result, {'a': 1, 'b': 2})

    def test_override_value(self):
        result = self.mod._deep_merge({'a': 1}, {'a': 99})
        self.assertEqual(result, {'a': 99})

    def test_nested_merge(self):
        base = {'a': {'b': 1, 'c': 2}}
        over = {'a': {'c': 99, 'd': 3}}
        result = self.mod._deep_merge(base, over)
        self.assertEqual(result, {'a': {'b': 1, 'c': 99, 'd': 3}})

    def test_does_not_mutate_original(self):
        base = {'a': {'b': 1}}
        over = {'a': {'b': 2}}
        original_base = copy.deepcopy(base)
        self.mod._deep_merge(base, over)
        self.assertEqual(base, original_base)

    def test_empty_override(self):
        result = self.mod._deep_merge({'a': 1}, {})
        self.assertEqual(result, {'a': 1})


class TestModelDetectionHelpers(unittest.TestCase):
    """Test model parsing helper functions."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_system_presets_module()

    def test_parse_parameter_count_standard(self):
        self.assertAlmostEqual(self.mod._parse_parameter_count('qwen3:32b'), 32.0)

    def test_parse_parameter_count_with_dash(self):
        self.assertAlmostEqual(self.mod._parse_parameter_count('qwen3-coder:30b'), 30.0)

    def test_parse_parameter_count_small(self):
        self.assertAlmostEqual(self.mod._parse_parameter_count('phi3:3.8b'), 3.8)

    def test_parse_parameter_count_no_match(self):
        result = self.mod._parse_parameter_count('mxbai-embed-large')
        self.assertEqual(result, 0.0)

    def test_parse_quantization(self):
        self.assertIn('Q4', self.mod._parse_quantization('model-Q4_K_M'))

    def test_parse_quantization_unknown(self):
        self.assertEqual(self.mod._parse_quantization('plain-model'), 'unknown')

    def test_parse_family_qwen(self):
        self.assertEqual(self.mod._parse_family('qwen3:32b'), 'qwen')

    def test_parse_family_deepseek(self):
        self.assertEqual(self.mod._parse_family('deepseek-r1:32b'), 'deepseek')

    def test_parse_family_llava(self):
        self.assertEqual(self.mod._parse_family('llava:13b'), 'llava')

    def test_model_info_size_category_small(self):
        m = self.mod.ModelInfo(name='test:7b', parameter_count_b=7.0)
        self.assertEqual(m.size_category, 'small')

    def test_model_info_size_category_medium(self):
        m = self.mod.ModelInfo(name='test:32b', parameter_count_b=32.0)
        self.assertEqual(m.size_category, 'medium')

    def test_model_info_size_category_large(self):
        m = self.mod.ModelInfo(name='test:70b', parameter_count_b=70.0)
        self.assertEqual(m.size_category, 'large')

    def test_select_model_smallest(self):
        models = [
            self.mod.ModelInfo(name='small:7b', parameter_count_b=7.0),
            self.mod.ModelInfo(name='big:32b', parameter_count_b=32.0),
        ]
        result = self.mod._select_model_by_strategy(models, 'smallest')
        self.assertEqual(result, 'small:7b')

    def test_select_model_largest(self):
        models = [
            self.mod.ModelInfo(name='small:7b', parameter_count_b=7.0),
            self.mod.ModelInfo(name='big:32b', parameter_count_b=32.0),
        ]
        result = self.mod._select_model_by_strategy(models, 'largest')
        self.assertEqual(result, 'big:32b')

    def test_select_model_medium(self):
        models = [
            self.mod.ModelInfo(name='small:7b', parameter_count_b=7.0),
            self.mod.ModelInfo(name='mid:14b', parameter_count_b=14.0),
            self.mod.ModelInfo(name='big:32b', parameter_count_b=32.0),
        ]
        result = self.mod._select_model_by_strategy(models, 'medium')
        self.assertEqual(result, 'mid:14b')

    def test_select_model_empty_list(self):
        result = self.mod._select_model_by_strategy([], 'smallest')
        self.assertIsNone(result)

    def test_select_model_skips_embedding(self):
        models = [
            self.mod.ModelInfo(name='mxbai-embed-large', parameter_count_b=0.3),
            self.mod.ModelInfo(name='qwen:7b', parameter_count_b=7.0),
        ]
        result = self.mod._select_model_by_strategy(models, 'smallest')
        self.assertEqual(result, 'qwen:7b')


class TestDetectAndRecommend(unittest.TestCase):
    """Test detect_and_recommend with mocked ollama."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_system_presets_module()

    def test_no_ollama_recommends_minimal(self):
        result = self.mod.system_presets_manager.detect_and_recommend()
        self.assertEqual(result.recommended_preset, 'minimal')
        self.assertIn('No Ollama models detected', result.reason)

    def test_detection_result_to_dict(self):
        result = self.mod.system_presets_manager.detect_and_recommend()
        d = result.to_dict()
        self.assertIn('models', d)
        self.assertIn('recommended_preset', d)
        self.assertIn('reason', d)
        self.assertIn('model_counts', d)


class TestAppPy(unittest.TestCase):
    """Validate app.py changes."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(API_DIR, 'app.py'))

    def test_version_1_8_7(self):
        self.assertIn('version="1.8.9"', self.content)

    def test_system_presets_router_import(self):
        self.assertIn('from .routes_system_presets import router as system_presets_router', self.content)

    def test_system_presets_router_registered(self):
        self.assertIn('app.include_router(system_presets_router)', self.content)

    def test_health_check_system_presets(self):
        self.assertIn('"system_presets": SYSTEM_PRESETS_AVAILABLE', self.content)

    def test_health_version_consistent(self):
        # Both FastAPI version and health endpoint version should be 1.8.9
        # One in version="1.8.9" and one in "version": "1.8.9"
        self.assertIn('version="1.8.9"', self.content)
        self.assertIn('"version": "1.8.9"', self.content)


class TestDepsPy(unittest.TestCase):
    """Validate deps.py changes."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(API_DIR, 'deps.py'))

    def test_system_presets_import_block(self):
        self.assertIn('from opti_oignon.system_presets import', self.content)

    def test_system_presets_available_flag(self):
        self.assertIn('SYSTEM_PRESETS_AVAILABLE', self.content)

    def test_system_presets_manager_imported(self):
        self.assertIn('system_presets_manager', self.content)


class TestSchemas(unittest.TestCase):
    """Validate new Pydantic schemas."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(API_DIR, 'schemas.py'))

    def test_system_preset_model_info_schema(self):
        self.assertIn('class SystemPresetModelInfo(BaseModel)', self.content)

    def test_system_preset_info_schema(self):
        self.assertIn('class SystemPresetInfo(BaseModel)', self.content)

    def test_system_preset_list_response_schema(self):
        self.assertIn('class SystemPresetListResponse(BaseModel)', self.content)

    def test_system_preset_detect_response_schema(self):
        self.assertIn('class SystemPresetDetectResponse(BaseModel)', self.content)

    def test_system_preset_apply_response_schema(self):
        self.assertIn('class SystemPresetApplyResponse(BaseModel)', self.content)

    def test_onboarding_state_response_schema(self):
        self.assertIn('class OnboardingStateResponse(BaseModel)', self.content)


class TestRoutesSystemPresets(unittest.TestCase):
    """Validate routes_system_presets.py structure."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(API_DIR, 'routes_system_presets.py'))

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(API_DIR, 'routes_system_presets.py')))

    def test_syntax_valid(self):
        ast.parse(self.content)

    def test_list_endpoint(self):
        self.assertIn('/list', self.content)
        self.assertIn('def list_system_presets', self.content)

    def test_detect_endpoint(self):
        self.assertIn('/detect', self.content)
        self.assertIn('def detect_and_recommend', self.content)

    def test_apply_endpoint(self):
        self.assertIn('/apply/{preset_id}', self.content)
        self.assertIn('def apply_system_preset', self.content)

    def test_onboarding_endpoint(self):
        self.assertIn('/onboarding', self.content)
        self.assertIn('def get_onboarding_state', self.content)

    def test_reset_onboarding_endpoint(self):
        self.assertIn('/onboarding/reset', self.content)
        self.assertIn('def reset_onboarding', self.content)

    def test_all_comments_english(self):
        # No French accented chars in comments
        for i, line in enumerate(self.content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""'):
                self.assertNotRegex(
                    stripped, r'[àâéèêëïîôùûüÿçæœ]',
                    f"Line {i} has non-English chars: {stripped[:80]}"
                )


class TestFrontendTypes(unittest.TestCase):
    """Validate frontend type additions."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(FRONTEND_SRC, 'lib', 'types.ts'))

    def test_system_preset_model_info(self):
        self.assertIn('export interface SystemPresetModelInfo', self.content)

    def test_system_preset_info(self):
        self.assertIn('export interface SystemPresetInfo', self.content)

    def test_system_preset_list_response(self):
        self.assertIn('export interface SystemPresetListResponse', self.content)

    def test_system_preset_detect_response(self):
        self.assertIn('export interface SystemPresetDetectResponse', self.content)

    def test_system_preset_apply_response(self):
        self.assertIn('export interface SystemPresetApplyResponse', self.content)

    def test_onboarding_state_response(self):
        self.assertIn('export interface OnboardingStateResponse', self.content)

    def test_chat_request_prompt_enhance(self):
        self.assertIn('prompt_enhance?: boolean', self.content)


class TestFrontendAPIClient(unittest.TestCase):
    """Validate systemPresets.ts API client."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(FRONTEND_SRC, 'lib', 'api', 'systemPresets.ts'))

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(FRONTEND_SRC, 'lib', 'api', 'systemPresets.ts')))

    def test_list_system_presets_function(self):
        self.assertIn('export async function listSystemPresets', self.content)

    def test_detect_and_recommend_function(self):
        self.assertIn('export async function detectAndRecommend', self.content)

    def test_apply_system_preset_function(self):
        self.assertIn('export async function applySystemPreset', self.content)

    def test_get_onboarding_state_function(self):
        self.assertIn('export async function getOnboardingState', self.content)

    def test_reset_onboarding_function(self):
        self.assertIn('export async function resetOnboarding', self.content)


class TestOnboardingOverlay(unittest.TestCase):
    """Validate OnboardingOverlay.svelte component."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'ui', 'OnboardingOverlay.svelte'))

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'ui', 'OnboardingOverlay.svelte')))

    def test_imports_api_functions(self):
        self.assertIn('getOnboardingState', self.content)
        self.assertIn('detectAndRecommend', self.content)
        self.assertIn('applySystemPreset', self.content)

    def test_welcome_text(self):
        self.assertIn('Welcome to Opti-Oignon', self.content)

    def test_apply_handler(self):
        self.assertIn('handleApply', self.content)

    def test_skip_handler(self):
        self.assertIn('handleSkip', self.content)

    def test_uses_oo_css_variables(self):
        self.assertIn('var(--oo-', self.content)

    def test_no_hardcoded_hex_colors(self):
        # Allow hex in SVG gradients and rgba but not standalone color assignments
        lines = self.content.splitlines()
        for i, line in enumerate(lines, 1):
            if 'style=' in line and '#' in line:
                # Should only be in gradient/radial contexts
                if 'radial-gradient' not in line and 'rgba' not in line:
                    # Allow fallback patterns
                    self.assertNotRegex(
                        line, r'color:\s*#[0-9a-fA-F]{3,8}',
                        f"Line {i} has hardcoded hex color: {line.strip()[:80]}"
                    )

    def test_onion_svg_in_header(self):
        self.assertIn('ellipse cx="12" cy="14"', self.content)

    def test_registered_in_layout(self):
        layout = _read(os.path.join(FRONTEND_SRC, 'routes', '+layout.svelte'))
        self.assertIn('OnboardingOverlay', layout)
        self.assertIn('<OnboardingOverlay />', layout)


class TestSettingsReorganization(unittest.TestCase):
    """Validate settings page reorganization."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'routes', 'settings', '+page.svelte'))

    def test_quick_tab_exists(self):
        self.assertIn("id: 'quick'", self.content)

    def test_system_preset_section(self):
        self.assertIn('System Preset', self.content)

    def test_apply_system_preset_handler(self):
        self.assertIn('handleApplySystemPreset', self.content)

    def test_collapsible_advanced_sections(self):
        for section in ['advCacheOpen', 'advProxyOpen', 'advStorageOpen', 'advConfigOpen', 'advOnboardingOpen']:
            self.assertIn(section, self.content, f"Missing collapsible: {section}")

    def test_onboarding_reset_in_advanced(self):
        self.assertIn('handleResetOnboarding', self.content)

    def test_human_readable_descriptions(self):
        # Each tab should have a description tooltip
        self.assertIn('tab.desc', self.content)

    def test_imports_system_presets_api(self):
        self.assertIn('listSystemPresets', self.content)
        self.assertIn('detectAndRecommend', self.content)
        self.assertIn('applySystemPreset', self.content)

    def test_active_badge(self):
        self.assertIn('Active', self.content)

    def test_recommended_badge(self):
        self.assertIn('Recommended', self.content)


class TestOnionButton(unittest.TestCase):
    """Validate Onion optimization button in ChatControlBar."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(
            FRONTEND_SRC, 'lib', 'components', 'chat', 'ChatControlBar.svelte'))

    def test_prompt_enhance_import(self):
        self.assertIn('promptEnhanceEnabled', self.content)

    def test_toggle_function(self):
        self.assertIn('function togglePromptEnhance()', self.content)

    def test_onion_svg_three_layers(self):
        # Outer, middle, inner ellipses
        ellipse_count = self.content.count('ellipse cx="12" cy="14"')
        self.assertEqual(ellipse_count, 3, "Onion SVG should have 3 concentric ellipses")

    def test_onion_stem(self):
        self.assertIn('M12 7V3', self.content)

    def test_onion_sprout(self):
        self.assertIn('M10 4.5c1-1 3-1 4 0', self.content)

    def test_glow_when_active(self):
        self.assertIn('box-shadow: 0 0 10px var(--oo-acc-400)', self.content)

    def test_opti_label(self):
        self.assertIn('Opti', self.content)

    def test_aria_pressed(self):
        self.assertIn('aria-pressed={$promptEnhanceEnabled}', self.content)

    def test_uses_copper_accent(self):
        self.assertIn('var(--oo-acc-', self.content)


class TestChatOptionsStore(unittest.TestCase):
    """Validate chatOptions store changes."""

    @classmethod
    def setUpClass(cls):
        cls.content = _read(os.path.join(FRONTEND_SRC, 'lib', 'stores', 'chatOptions.ts'))

    def test_prompt_enhance_store_defined(self):
        self.assertIn('export const promptEnhanceEnabled', self.content)

    def test_prompt_enhance_in_get_chat_options(self):
        self.assertIn('prompt_enhance?: boolean', self.content)

    def test_prompt_enhance_get_logic(self):
        self.assertIn('get(promptEnhanceEnabled)', self.content)

    def test_prompt_enhance_in_reset(self):
        self.assertIn('promptEnhanceEnabled.set(false)', self.content)


class TestNoRegressions(unittest.TestCase):
    """Verify no regressions on conventions."""

    def test_all_python_files_valid_syntax(self):
        new_files = [
            os.path.join(BACKEND_DIR, 'system_presets.py'),
            os.path.join(API_DIR, 'routes_system_presets.py'),
        ]
        for fpath in new_files:
            with open(fpath) as f:
                ast.parse(f.read(), filename=fpath)

    def test_no_emoji_in_python_code(self):
        new_files = [
            os.path.join(BACKEND_DIR, 'system_presets.py'),
            os.path.join(API_DIR, 'routes_system_presets.py'),
        ]
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
            r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
            r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]'
        )
        for fpath in new_files:
            content = _read(fpath)
            self.assertIsNone(
                emoji_pattern.search(content),
                f"Emoji found in {os.path.basename(fpath)}"
            )

    def test_checkpoint_before_apply_always_true(self):
        content = _read(os.path.join(BACKEND_DIR, 'system_presets.py'))
        # The apply_preset method should hardcode checkpoint_before_apply = True
        self.assertIn('checkpoint_before_apply = True', content)


if __name__ == '__main__':
    unittest.main()
