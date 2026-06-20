"""
Tests for S94 -- UX Cleanup: Loader, Routing Indicator, Vision Selector.

Validates:
- Part 1: Streaming cleanup (model name hidden during streaming, tobacco color)
- Part 2: Vision config (detection strategies, capability probing, patterns,
  known models, YAML persistence, API routes, frontend API client)
- Part 3: Minor fixes (WebSocket reconnection, image error messages, thumbnails)
- Version bump to 1.9.6
- No hardcoded hex in new/modified Svelte files
- Zero regressions

Target: ~25 tests
"""

import importlib.util
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
BACKEND_DIR = os.path.join(PROJECT_ROOT, "opti_oignon")
API_DIR = os.path.join(BACKEND_DIR, "api")
CONFIG_DIR = os.path.join(BACKEND_DIR, "config")
FRONTEND_SRC = os.path.join(PROJECT_ROOT, "frontend", "src")
COMPONENTS_DIR = os.path.join(FRONTEND_SRC, "lib", "components")
CHAT_DIR = os.path.join(COMPONENTS_DIR, "chat")
SETTINGS_DIR = os.path.join(COMPONENTS_DIR, "settings")
ROUTES_DIR = os.path.join(FRONTEND_SRC, "routes")


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
# Load vision_config module (isolated from opti_oignon __init__)
# ---------------------------------------------------------------------------

vision_config_mod = _load_module(
    "opti_oignon.vision_config",
    os.path.join(BACKEND_DIR, "vision_config.py"),
)
VisionConfig = vision_config_mod.VisionConfig


# ===========================================================================
# PART 1: Streaming Cleanup
# ===========================================================================


class TestStreamingCleanup(unittest.TestCase):
    """Validate streaming indicator and chat message changes."""

    def test_streaming_indicator_no_model_prop(self):
        """StreamingIndicator should not accept a model prop."""
        content = _read(os.path.join(CHAT_DIR, "StreamingIndicator.svelte"))
        self.assertNotIn("export let model", content)

    def test_streaming_indicator_uses_tobacco_color(self):
        """StreamingIndicator should pass tobacco color to OnionLoader."""
        content = _read(os.path.join(CHAT_DIR, "StreamingIndicator.svelte"))
        self.assertIn("var(--oo-tobacco)", content)
        self.assertNotIn("var(--oo-acc-400)", content)

    def test_streaming_indicator_no_model_display(self):
        """StreamingIndicator should not display model name at all."""
        content = _read(os.path.join(CHAT_DIR, "StreamingIndicator.svelte"))
        self.assertNotIn("{model}", content)
        self.assertNotIn("font-mono", content)

    def test_chat_message_hides_model_during_streaming(self):
        """ChatMessage should hide model name when isStreaming is true."""
        content = _read(os.path.join(CHAT_DIR, "ChatMessage.svelte"))
        self.assertIn("!isStreaming", content)
        # The model display line should have the isStreaming guard
        model_lines = [l for l in content.split("\n") if "message.model" in l and "isStreaming" in l]
        self.assertTrue(len(model_lines) > 0, "Model display should be guarded by !isStreaming")

    def test_onion_loader_default_tobacco_color(self):
        """OnionLoader default color should be tobacco."""
        content = _read(os.path.join(COMPONENTS_DIR, "OnionLoader.svelte"))
        self.assertIn("var(--oo-tobacco)", content)
        # Should not have old acc-500 default
        self.assertNotIn("'var(--oo-acc-500)'", content)

    def test_onion_loader_no_road_track(self):
        """OnionLoader should not have road track elements."""
        content = _read(os.path.join(COMPONENTS_DIR, "OnionLoader.svelte"))
        self.assertNotIn("road-track", content)
        self.assertNotIn("road-line", content)
        self.assertNotIn("road-scroll", content)

    def test_onion_loader_keeps_spin_and_bounce(self):
        """OnionLoader should still have spin and bounce animations."""
        content = _read(os.path.join(COMPONENTS_DIR, "OnionLoader.svelte"))
        self.assertIn("onion-rotate", content)
        self.assertIn("onion-bump", content)
        self.assertIn("onion-spin", content)
        self.assertIn("onion-bounce", content)

    def test_chat_page_no_model_prop_on_streaming_indicator(self):
        """Chat page should not pass model prop to StreamingIndicator."""
        content = _read(os.path.join(ROUTES_DIR, "chat", "[id]", "+page.svelte"))
        # Should just be <StreamingIndicator /> with no props
        self.assertIn("<StreamingIndicator />", content)
        self.assertNotIn("StreamingIndicator model=", content)


# ===========================================================================
# PART 2: Vision Config
# ===========================================================================


class TestVisionConfigModule(unittest.TestCase):
    """Validate VisionConfig class logic."""

    def _make_config(self, yaml_content=None):
        """Create a VisionConfig with a temp YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            if yaml_content:
                f.write(yaml_content)
            tmp_path = Path(f.name)
        try:
            vc = VisionConfig(config_path=tmp_path)
            return vc, tmp_path
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def test_default_values(self):
        """VisionConfig should have sane defaults."""
        vc, path = self._make_config("")
        try:
            self.assertEqual(vc.vision_model, "auto")
            self.assertEqual(vc.detection_strategy, "both")
            self.assertIn("vl", vc.auto_detect_patterns)
            self.assertIn("clip", vc.vision_families)
            self.assertEqual(vc.known_vision_models, [])
        finally:
            path.unlink(missing_ok=True)

    def test_pattern_detection_vl(self):
        """Pattern detection should match 'vl' in model names."""
        vc, path = self._make_config("")
        try:
            self.assertTrue(vc._detect_by_patterns("qwen3-vl:32b"))
            self.assertTrue(vc._detect_by_patterns("llava:13b"))
            self.assertFalse(vc._detect_by_patterns("qwen3:32b"))
            self.assertFalse(vc._detect_by_patterns("deepseek-r1:32b"))
        finally:
            path.unlink(missing_ok=True)

    def test_known_models_override(self):
        """Manually declared models should always be detected as vision."""
        vc, path = self._make_config("")
        try:
            vc._known_vision_models = ["qwen3.5:32b"]
            self.assertTrue(vc.is_vision_model("qwen3.5:32b"))
            # Even with patterns strategy only
            vc._detection_strategy = "patterns"
            self.assertTrue(vc.is_vision_model("qwen3.5:32b"))
        finally:
            path.unlink(missing_ok=True)

    def test_detect_vision_models_list(self):
        """detect_vision_models should return all matching models."""
        vc, path = self._make_config("")
        try:
            vc._known_vision_models = ["qwen3.5:32b"]
            vc._detection_strategy = "patterns"
            models = ["qwen3:32b", "qwen3-vl:32b", "deepseek-r1:32b", "qwen3.5:32b"]
            result = vc.detect_vision_models(models)
            self.assertIn("qwen3-vl:32b", result)
            self.assertIn("qwen3.5:32b", result)
            self.assertNotIn("qwen3:32b", result)
            self.assertNotIn("deepseek-r1:32b", result)
        finally:
            path.unlink(missing_ok=True)

    def test_detect_vision_models_dedup(self):
        """detect_vision_models should deduplicate results."""
        vc, path = self._make_config("")
        try:
            models = ["llava:13b", "llava:13b", "qwen3-vl:32b"]
            result = vc.detect_vision_models(models)
            self.assertEqual(len(set(result)), len(result))
        finally:
            path.unlink(missing_ok=True)

    def test_effective_model_auto(self):
        """Auto mode should return first detected vision model."""
        vc, path = self._make_config("")
        try:
            vc._detection_strategy = "patterns"
            models = ["qwen3:32b", "llava:13b", "qwen3-vl:32b"]
            eff = vc.get_effective_model(models)
            self.assertEqual(eff, "llava:13b")
        finally:
            path.unlink(missing_ok=True)

    def test_effective_model_explicit(self):
        """Explicit selection should return that model if available."""
        vc, path = self._make_config("")
        try:
            vc._vision_model = "qwen3-vl:32b"
            models = ["qwen3:32b", "qwen3-vl:32b"]
            self.assertEqual(vc.get_effective_model(models), "qwen3-vl:32b")
        finally:
            path.unlink(missing_ok=True)

    def test_effective_model_explicit_missing(self):
        """Explicit model not in available list should return None."""
        vc, path = self._make_config("")
        try:
            vc._vision_model = "nonexistent:latest"
            models = ["qwen3:32b"]
            self.assertIsNone(vc.get_effective_model(models))
        finally:
            path.unlink(missing_ok=True)

    def test_effective_model_no_vision_available(self):
        """Auto mode with no vision models should return None."""
        vc, path = self._make_config("")
        try:
            vc._detection_strategy = "patterns"
            models = ["qwen3:32b", "deepseek-r1:32b"]
            self.assertIsNone(vc.get_effective_model(models))
        finally:
            path.unlink(missing_ok=True)

    def test_capability_cache(self):
        """Capability probe results should be cached."""
        vc, path = self._make_config("")
        try:
            # Manually fill cache
            import time
            vc._capability_cache["test-model"] = (True, time.monotonic())
            self.assertTrue(vc._probe_model_capabilities("test-model"))
            # Clear cache
            vc.clear_cache()
            self.assertEqual(len(vc._capability_cache), 0)
        finally:
            path.unlink(missing_ok=True)

    def test_yaml_persistence(self):
        """Setting vision_model should persist to YAML."""
        vc, path = self._make_config("")
        try:
            vc.vision_model = "qwen3-vl:32b"
            # Reload from same file
            vc2 = VisionConfig(config_path=path)
            self.assertEqual(vc2.vision_model, "qwen3-vl:32b")
        finally:
            path.unlink(missing_ok=True)

    def test_to_dict(self):
        """to_dict should include all config fields."""
        vc, path = self._make_config("")
        try:
            d = vc.to_dict()
            required_keys = {
                "vision_model", "detection_strategy", "auto_detect_patterns",
                "vision_families", "known_vision_models", "describe_prompt",
            }
            self.assertTrue(required_keys.issubset(set(d.keys())))
        finally:
            path.unlink(missing_ok=True)

    def test_is_vision_model_patterns_strategy(self):
        """Patterns-only strategy should not probe capabilities."""
        vc, path = self._make_config("")
        try:
            vc._detection_strategy = "patterns"
            # This model would need capability probe to detect
            self.assertFalse(vc.is_vision_model("qwen3.5:32b"))
            # But pattern models still work
            self.assertTrue(vc.is_vision_model("llava:7b"))
        finally:
            path.unlink(missing_ok=True)

    def test_describe_prompt_setter(self):
        """Setting describe_prompt should persist."""
        vc, path = self._make_config("")
        try:
            vc.describe_prompt = "Custom prompt here."
            self.assertEqual(vc.describe_prompt, "Custom prompt here.")
            # Empty should reset to default
            vc.describe_prompt = ""
            self.assertIn("Describe this image", vc.describe_prompt)
        finally:
            path.unlink(missing_ok=True)


# ===========================================================================
# PART 2b: Vision Files Existence & Structure
# ===========================================================================


class TestVisionFiles(unittest.TestCase):
    """Validate vision-related files exist and have correct structure."""

    def test_vision_config_yaml_exists(self):
        """vision.yaml should exist in config dir."""
        self.assertTrue(os.path.isfile(os.path.join(CONFIG_DIR, "vision.yaml")))

    def test_vision_config_py_exists(self):
        """vision_config.py should exist."""
        self.assertTrue(os.path.isfile(os.path.join(BACKEND_DIR, "vision_config.py")))

    def test_routes_vision_py_exists(self):
        """routes_vision.py should exist in api dir."""
        self.assertTrue(os.path.isfile(os.path.join(API_DIR, "routes_vision.py")))

    def test_vision_api_client_exists(self):
        """Frontend vision.ts API client should exist."""
        self.assertTrue(
            os.path.isfile(os.path.join(FRONTEND_SRC, "lib", "api", "vision.ts"))
        )

    def test_vision_model_selector_exists(self):
        """VisionModelSelector.svelte should exist in settings components."""
        self.assertTrue(
            os.path.isfile(os.path.join(SETTINGS_DIR, "VisionModelSelector.svelte"))
        )

    def test_routes_vision_has_endpoints(self):
        """routes_vision.py should define GET/PUT config and GET models."""
        content = _read(os.path.join(API_DIR, "routes_vision.py"))
        self.assertIn('@router.get("/config"', content)
        self.assertIn('@router.put("/config"', content)
        self.assertIn('@router.get("/models"', content)
        self.assertIn('@router.post("/clear-cache"', content)

    def test_vision_api_client_exports(self):
        """vision.ts should export get/update/list/clear functions."""
        content = _read(os.path.join(FRONTEND_SRC, "lib", "api", "vision.ts"))
        self.assertIn("getVisionConfig", content)
        self.assertIn("updateVisionConfig", content)
        self.assertIn("listVisionModels", content)
        self.assertIn("clearVisionCache", content)

    def test_deps_has_vision_config(self):
        """deps.py should import vision_config singleton."""
        content = _read(os.path.join(API_DIR, "deps.py"))
        self.assertIn("vision_config", content)
        self.assertIn("VISION_CONFIG_AVAILABLE", content)

    def test_app_registers_vision_router(self):
        """app.py should register the vision router."""
        content = _read(os.path.join(API_DIR, "app.py"))
        self.assertIn("vision_router", content)
        self.assertIn("include_router(vision_router)", content)

    def test_settings_page_imports_vision_selector(self):
        """Settings page should import VisionModelSelector."""
        content = _read(os.path.join(ROUTES_DIR, "settings", "+page.svelte"))
        self.assertIn("VisionModelSelector", content)


# ===========================================================================
# PART 3: Minor Fixes
# ===========================================================================


class TestMinorFixes(unittest.TestCase):
    """Validate WebSocket reconnection, error messages, thumbnails."""

    def test_websocket_reconnection_constants(self):
        """chat.ts should define reconnection constants."""
        content = _read(os.path.join(FRONTEND_SRC, "lib", "api", "chat.ts"))
        self.assertIn("WS_MAX_RETRIES", content)
        self.assertIn("WS_BASE_DELAY_MS", content)
        self.assertIn("WS_MAX_DELAY_MS", content)

    def test_websocket_backoff_function(self):
        """chat.ts should have a backoffDelay function."""
        content = _read(os.path.join(FRONTEND_SRC, "lib", "api", "chat.ts"))
        self.assertIn("function backoffDelay", content)
        self.assertIn("Math.pow", content)
        # Should have jitter
        self.assertIn("Math.random", content)

    def test_websocket_reconnect_logic(self):
        """chat.ts should attempt reconnection on unexpected close."""
        content = _read(os.path.join(FRONTEND_SRC, "lib", "api", "chat.ts"))
        self.assertIn("retryCount", content)
        self.assertIn("hasReceivedData", content)
        self.assertIn("reconnecting", content)

    def test_websocket_reconnect_metadata_event(self):
        """Reconnection attempts should emit metadata with attempt count."""
        content = _read(os.path.join(FRONTEND_SRC, "lib", "api", "chat.ts"))
        self.assertIn("reconnecting: true", content)
        self.assertIn("attempt:", content)

    def test_image_validation_better_messages(self):
        """validateImageFile should have descriptive error messages."""
        content = _read(os.path.join(FRONTEND_SRC, "lib", "api", "files.ts"))
        # Empty file check
        self.assertIn("Empty or invalid file", content)
        # No extension
        self.assertIn("No file extension detected", content)
        # Unsupported format with supported list
        self.assertIn("Supported:", content)
        # Size with compression advice
        self.assertIn("compressing or resizing", content)

    def test_image_upload_skipped_files_notification(self):
        """ChatInput should notify about skipped non-image files."""
        content = _read(os.path.join(CHAT_DIR, "ChatInput.svelte"))
        self.assertIn("skippedNonImage", content)
        self.assertIn("non-image files", content)

    def test_thumbnail_size_increased(self):
        """Image thumbnails should be larger (w-20 h-20)."""
        content = _read(os.path.join(CHAT_DIR, "ChatInput.svelte"))
        self.assertIn("w-20 h-20", content)
        self.assertNotIn("w-16 h-16", content)

    def test_thumbnail_uses_css_variable_border(self):
        """Image thumbnails should use CSS variable for border."""
        content = _read(os.path.join(CHAT_DIR, "ChatInput.svelte"))
        # The img tag should use inline style with var
        self.assertIn("var(--oo-bd-default)", content)


# ===========================================================================
# PART 4: Version & No-Regressions
# ===========================================================================


class TestVersionAndIntegrity(unittest.TestCase):
    """Validate version bump and code quality."""

    def test_version_app_py(self):
        """app.py should have version 1.9.6."""
        content = _read(os.path.join(API_DIR, "app.py"))
        self.assertIn('version="1.10.0"', content)

    def test_version_pyproject(self):
        """pyproject.toml should have version 1.9.6."""
        content = _read(os.path.join(PROJECT_ROOT, "pyproject.toml"))
        self.assertIn('version = "1.10.0"', content)

    def test_version_setup_py(self):
        """setup.py should have version 1.9.6."""
        content = _read(os.path.join(PROJECT_ROOT, "setup.py"))
        self.assertIn('version="1.10.0"', content)

    def test_no_hardcoded_hex_in_new_svelte(self):
        """New/modified Svelte files should not have hardcoded hex colors."""
        files_to_check = [
            os.path.join(COMPONENTS_DIR, "OnionLoader.svelte"),
            os.path.join(CHAT_DIR, "StreamingIndicator.svelte"),
            os.path.join(SETTINGS_DIR, "VisionModelSelector.svelte"),
        ]
        hex_pattern = re.compile(
            r'(?:color|background|border|fill|stroke)\s*[:=]\s*["\']?#[0-9a-fA-F]{3,8}\b'
        )
        for filepath in files_to_check:
            content = _read(filepath)
            matches = hex_pattern.findall(content)
            self.assertEqual(
                len(matches), 0,
                f"Hardcoded hex found in {os.path.basename(filepath)}: {matches}",
            )

    def test_no_emoji_in_python_code(self):
        """New Python files should not contain emoji."""
        files_to_check = [
            os.path.join(BACKEND_DIR, "vision_config.py"),
            os.path.join(API_DIR, "routes_vision.py"),
        ]
        emoji_pattern = re.compile(
            r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA6F]"
        )
        for filepath in files_to_check:
            content = _read(filepath)
            matches = emoji_pattern.findall(content)
            self.assertEqual(
                len(matches), 0,
                f"Emoji found in {os.path.basename(filepath)}: {matches}",
            )

    def test_vision_yaml_has_all_fields(self):
        """vision.yaml should contain all expected configuration keys."""
        content = _read(os.path.join(CONFIG_DIR, "vision.yaml"))
        for key in [
            "vision_model",
            "detection_strategy",
            "auto_detect_patterns",
            "vision_families",
            "known_vision_models",
            "describe_prompt",
        ]:
            self.assertIn(key, content, f"Missing key '{key}' in vision.yaml")


if __name__ == "__main__":
    unittest.main()
