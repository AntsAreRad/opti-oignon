"""
Tests for S95 -- Transparent Vision Pipeline.

Validates:
- Part 1: VisionPipeline backend (detection, description, augmentation, process)
- Part 2: Config persistence (delegation fields in vision.yaml)
- Part 3: Executor integration (vision pipeline hook in execute)
- Part 4: Frontend (types, chat.ts, ChatMessage badge, indicator, page)
- Part 5: Routes (vision_delegation WS events in routes_chat.py)
- Part 6: Version bump to 1.9.7
- Zero regressions

Target: ~40 tests
"""

import importlib.util
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

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
CHAT_DIR = os.path.join(COMPONENTS_DIR, "chat")
ROUTES_DIR = os.path.join(FRONTEND_SRC, "routes")
API_TS_DIR = os.path.join(FRONTEND_SRC, "lib", "api")
STORES_DIR = os.path.join(FRONTEND_SRC, "lib", "stores")


def _load_module(name, path):
    """Load a Python module from file path without triggering __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    """Read file contents as string."""
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Load modules in isolation
# ---------------------------------------------------------------------------

# Ensure vision_config is loaded first (dependency of vision_pipeline)
vision_config_mod = _load_module(
    "opti_oignon.vision_config",
    os.path.join(BACKEND_DIR, "vision_config.py"),
)
VisionConfig = vision_config_mod.VisionConfig

# Load vision_pipeline module
vision_pipeline_mod = _load_module(
    "opti_oignon.vision_pipeline",
    os.path.join(BACKEND_DIR, "vision_pipeline.py"),
)
VisionPipeline = vision_pipeline_mod.VisionPipeline


# ===========================================================================
# PART 1: VisionPipeline Backend
# ===========================================================================


class TestVisionPipelineDetection(unittest.TestCase):
    """Test detect_needs_delegation logic."""

    def _make_pipeline(self, delegation_enabled=True, vision_models=None):
        """Create a VisionPipeline with mocked dependencies."""
        vc = MagicMock()
        vc.is_vision_model = MagicMock(side_effect=lambda m: "vl" in m.lower())
        vc.get_effective_model = MagicMock(
            return_value=vision_models[0] if vision_models else None
        )
        vc.describe_prompt = "Describe this image."

        mock_ollama = MagicMock()
        mock_ollama.list = MagicMock(return_value={
            "models": [{"model": m} for m in (vision_models or [])]
        })

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.safe_dump({
                "delegation_enabled": delegation_enabled,
                "inject_format": "[Image analysis: {description}]\nUser question: {message}",
                "max_description_tokens": 500,
            }, f)
            config_path = Path(f.name)

        vp = VisionPipeline(
            vision_config=vc,
            ollama_module=mock_ollama,
            config_path=config_path,
        )
        config_path.unlink(missing_ok=True)
        return vp

    def test_delegation_needed_non_vision_model_with_images(self):
        """Should delegate when current model is non-vision and images present."""
        vp = self._make_pipeline(vision_models=["qwen3-vl:32b"])
        result = vp.detect_needs_delegation("describe this", ["base64img"], "qwen3:32b")
        self.assertTrue(result)

    def test_no_delegation_when_disabled(self):
        """Should not delegate when delegation_enabled is False."""
        vp = self._make_pipeline(delegation_enabled=False, vision_models=["qwen3-vl:32b"])
        result = vp.detect_needs_delegation("describe this", ["base64img"], "qwen3:32b")
        self.assertFalse(result)

    def test_no_delegation_without_images(self):
        """Should not delegate when no images are present."""
        vp = self._make_pipeline(vision_models=["qwen3-vl:32b"])
        self.assertFalse(vp.detect_needs_delegation("hello", None, "qwen3:32b"))
        self.assertFalse(vp.detect_needs_delegation("hello", [], "qwen3:32b"))

    def test_no_delegation_when_model_is_vision(self):
        """Should not delegate when current model has vision capabilities."""
        vp = self._make_pipeline(vision_models=["qwen3-vl:32b"])
        result = vp.detect_needs_delegation("describe", ["img"], "qwen3-vl:32b")
        self.assertFalse(result)

    def test_no_delegation_when_no_vision_model_available(self):
        """Should not delegate when no vision model is available."""
        vp = self._make_pipeline(vision_models=[])
        result = vp.detect_needs_delegation("describe", ["img"], "qwen3:32b")
        self.assertFalse(result)


class TestVisionPipelineDescribe(unittest.TestCase):
    """Test describe_image method."""

    def test_describe_image_calls_ollama(self):
        """Should call ollama.chat with vision model and images."""
        mock_ollama = MagicMock()
        mock_ollama.chat = MagicMock(return_value={
            "message": {"content": "A bar chart showing Q3 revenue growth."}
        })
        mock_ollama.list = MagicMock(return_value={"models": []})

        vc = MagicMock()
        vc.describe_prompt = "Describe this image."
        vc.get_effective_model = MagicMock(return_value="qwen3-vl:32b")
        vc.is_vision_model = MagicMock(return_value=False)

        vp = VisionPipeline(vision_config=vc, ollama_module=mock_ollama)
        desc = vp.describe_image(["b64data"], "What is this?", "qwen3-vl:32b")

        self.assertEqual(desc, "A bar chart showing Q3 revenue growth.")
        mock_ollama.chat.assert_called_once()
        call_kwargs = mock_ollama.chat.call_args
        self.assertEqual(call_kwargs[1]["model"], "qwen3-vl:32b")
        self.assertFalse(call_kwargs[1]["stream"])
        # Images should be in the message
        msgs = call_kwargs[1]["messages"]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["images"], ["b64data"])

    def test_describe_image_empty_on_failure(self):
        """Should return empty string on ollama error."""
        mock_ollama = MagicMock()
        mock_ollama.chat = MagicMock(side_effect=RuntimeError("Connection refused"))
        mock_ollama.list = MagicMock(return_value={"models": []})

        vc = MagicMock()
        vc.describe_prompt = "Describe."
        vc.get_effective_model = MagicMock(return_value="llava:13b")

        vp = VisionPipeline(vision_config=vc, ollama_module=mock_ollama)
        desc = vp.describe_image(["img"], "hello", "llava:13b")
        self.assertEqual(desc, "")

    def test_describe_image_no_model_returns_empty(self):
        """Should return empty if no vision model resolved."""
        mock_ollama = MagicMock()
        mock_ollama.list = MagicMock(return_value={"models": []})

        vc = MagicMock()
        vc.describe_prompt = "Describe."
        vc.get_effective_model = MagicMock(return_value=None)

        vp = VisionPipeline(vision_config=vc, ollama_module=mock_ollama)
        desc = vp.describe_image(["img"], "hello")
        self.assertEqual(desc, "")
        mock_ollama.chat.assert_not_called()

    def test_describe_image_combines_prompt_with_user_context(self):
        """Should append user question to base describe prompt."""
        mock_ollama = MagicMock()
        mock_ollama.chat = MagicMock(return_value={
            "message": {"content": "desc"}
        })
        mock_ollama.list = MagicMock(return_value={"models": []})

        vc = MagicMock()
        vc.describe_prompt = "Base prompt."
        vc.get_effective_model = MagicMock(return_value="vl:7b")

        vp = VisionPipeline(vision_config=vc, ollama_module=mock_ollama)
        vp.describe_image(["img"], "What colors?", "vl:7b")

        sent_content = mock_ollama.chat.call_args[1]["messages"][0]["content"]
        self.assertIn("Base prompt.", sent_content)
        self.assertIn("What colors?", sent_content)

    def test_describe_image_respects_max_tokens(self):
        """Should pass max_description_tokens as num_predict."""
        mock_ollama = MagicMock()
        mock_ollama.chat = MagicMock(return_value={"message": {"content": "ok"}})
        mock_ollama.list = MagicMock(return_value={"models": []})

        vc = MagicMock()
        vc.describe_prompt = "Describe."
        vc.get_effective_model = MagicMock(return_value="vl:7b")

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.safe_dump({"max_description_tokens": 200}, f)
            config_path = Path(f.name)

        vp = VisionPipeline(vision_config=vc, ollama_module=mock_ollama, config_path=config_path)
        vp.describe_image(["img"], "test", "vl:7b")
        config_path.unlink(missing_ok=True)

        options = mock_ollama.chat.call_args[1]["options"]
        self.assertEqual(options["num_predict"], 200)


class TestVisionPipelineAugment(unittest.TestCase):
    """Test build_augmented_message."""

    def test_augment_default_format(self):
        """Should inject description with default format."""
        vp = VisionPipeline(vision_config=MagicMock(), ollama_module=MagicMock())
        result = vp.build_augmented_message("What is this?", "A cat sitting on a table.")
        self.assertIn("[Image analysis: A cat sitting on a table.]", result)
        self.assertIn("User question: What is this?", result)

    def test_augment_empty_description_returns_original(self):
        """Should return original message if description is empty."""
        vp = VisionPipeline(vision_config=MagicMock(), ollama_module=MagicMock())
        result = vp.build_augmented_message("Hello", "")
        self.assertEqual(result, "Hello")

    def test_augment_custom_format(self):
        """Should use custom inject_format from config."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.safe_dump({
                "inject_format": "IMAGE: {description}\nQUESTION: {message}",
            }, f)
            config_path = Path(f.name)

        vp = VisionPipeline(
            vision_config=MagicMock(), ollama_module=MagicMock(),
            config_path=config_path,
        )
        config_path.unlink(missing_ok=True)

        result = vp.build_augmented_message("Describe", "A chart")
        self.assertIn("IMAGE: A chart", result)
        self.assertIn("QUESTION: Describe", result)

    def test_augment_malformed_format_fallback(self):
        """Should fall back to default format on malformed inject_format."""
        vp = VisionPipeline(vision_config=MagicMock(), ollama_module=MagicMock())
        vp._inject_format = "{unknown_key}"
        result = vp.build_augmented_message("Test", "Desc")
        self.assertIn("[Image analysis: Desc]", result)
        self.assertIn("User question: Test", result)


class TestVisionPipelineProcess(unittest.TestCase):
    """Test the full process() pipeline method."""

    def _make_pipeline(self, description="A beautiful landscape.", vision_model="qwen3-vl:32b"):
        """Create a pipeline that will delegate successfully."""
        mock_ollama = MagicMock()
        mock_ollama.chat = MagicMock(return_value={
            "message": {"content": description}
        })
        mock_ollama.list = MagicMock(return_value={
            "models": [{"model": vision_model}]
        })

        vc = MagicMock()
        vc.is_vision_model = MagicMock(side_effect=lambda m: m == vision_model)
        vc.get_effective_model = MagicMock(return_value=vision_model)
        vc.describe_prompt = "Describe this image."

        return VisionPipeline(vision_config=vc, ollama_module=mock_ollama)

    def test_process_delegation_full_flow(self):
        """Full pipeline: non-vision model + images -> augmented message, no images."""
        vp = self._make_pipeline()
        msg, imgs, meta = vp.process("What is this?", ["b64data"], "qwen3:32b")

        self.assertIn("A beautiful landscape.", msg)
        self.assertIn("What is this?", msg)
        self.assertIsNone(imgs)  # Images consumed
        self.assertTrue(meta.get("delegated"))
        self.assertEqual(meta["vision_model"], "qwen3-vl:32b")
        self.assertGreater(meta["description_length"], 0)
        self.assertIn("duration_ms", meta)

    def test_process_no_delegation_vision_model(self):
        """Should pass through when current model is vision-capable."""
        vp = self._make_pipeline(vision_model="qwen3-vl:32b")
        msg, imgs, meta = vp.process("Describe", ["img"], "qwen3-vl:32b")

        self.assertEqual(msg, "Describe")
        self.assertEqual(imgs, ["img"])
        self.assertEqual(meta, {})

    def test_process_no_delegation_no_images(self):
        """Should pass through when no images."""
        vp = self._make_pipeline()
        msg, imgs, meta = vp.process("Hello", None, "qwen3:32b")

        self.assertEqual(msg, "Hello")
        self.assertIsNone(imgs)
        self.assertEqual(meta, {})

    def test_process_calls_on_status(self):
        """Should call on_status callback during delegation."""
        vp = self._make_pipeline()
        statuses = []
        vp.process("test", ["img"], "qwen3:32b", on_status=statuses.append)
        self.assertTrue(any("Analyzing image" in s for s in statuses))

    def test_process_empty_description_returns_original(self):
        """Should return original message if vision model produces empty description."""
        vp = self._make_pipeline(description="")
        msg, imgs, meta = vp.process("What?", ["img"], "qwen3:32b")

        # Empty description -> no augmentation, images remain
        self.assertEqual(msg, "What?")
        self.assertEqual(imgs, ["img"])
        self.assertEqual(meta, {})

    def test_process_to_dict(self):
        """to_dict should return delegation config."""
        vp = self._make_pipeline()
        d = vp.to_dict()
        self.assertIn("delegation_enabled", d)
        self.assertIn("inject_format", d)
        self.assertIn("max_description_tokens", d)
        self.assertIn("effective_vision_model", d)


# ===========================================================================
# PART 2: Config Persistence
# ===========================================================================


class TestVisionYamlConfig(unittest.TestCase):
    """Validate vision.yaml delegation fields."""

    def setUp(self):
        self.config_path = os.path.join(CONFIG_DIR, "vision.yaml")
        with open(self.config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def test_delegation_enabled_field_exists(self):
        """vision.yaml should have delegation_enabled field."""
        self.assertIn("delegation_enabled", self.config)
        self.assertIsInstance(self.config["delegation_enabled"], bool)

    def test_inject_format_field_exists(self):
        """vision.yaml should have inject_format field."""
        self.assertIn("inject_format", self.config)
        fmt = self.config["inject_format"]
        self.assertIn("{description}", fmt)
        self.assertIn("{message}", fmt)

    def test_max_description_tokens_field_exists(self):
        """vision.yaml should have max_description_tokens field."""
        self.assertIn("max_description_tokens", self.config)
        self.assertIsInstance(self.config["max_description_tokens"], int)
        self.assertGreater(self.config["max_description_tokens"], 0)

    def test_delegation_enabled_default_true(self):
        """Default delegation_enabled should be True."""
        self.assertTrue(self.config["delegation_enabled"])

    def test_max_description_tokens_default_500(self):
        """Default max_description_tokens should be 500."""
        self.assertEqual(self.config["max_description_tokens"], 500)

    def test_s94_fields_preserved(self):
        """S94 fields should still be present."""
        for key in ["vision_model", "detection_strategy", "auto_detect_patterns",
                     "vision_families", "known_vision_models", "describe_prompt"]:
            self.assertIn(key, self.config, f"Missing S94 field: {key}")


# ===========================================================================
# PART 3: Executor Integration
# ===========================================================================


class TestExecutorVisionIntegration(unittest.TestCase):
    """Validate vision pipeline integration in executor.py."""

    def setUp(self):
        self.executor_src = _read(os.path.join(BACKEND_DIR, "executor.py"))

    def test_vision_pipeline_import(self):
        """executor.py should import vision_pipeline with FEATURE_AVAILABLE flag."""
        self.assertIn("VISION_PIPELINE_AVAILABLE", self.executor_src)
        self.assertIn("from .vision_pipeline import", self.executor_src)

    def test_vision_delegation_step(self):
        """executor.py should have Step 0b: Vision delegation."""
        self.assertIn("Step 0b: Vision delegation", self.executor_src)

    def test_last_vision_meta_property(self):
        """executor.py should expose last_vision_meta property."""
        self.assertIn("last_vision_meta", self.executor_src)
        self.assertIn("_last_vision_meta", self.executor_src)

    def test_vision_meta_reset_per_call(self):
        """_last_vision_meta should be reset at start of execute()."""
        # Find the reset block
        self.assertIn("self._last_vision_meta: dict = {}  # S95: reset per-call",
                       self.executor_src)

    def test_vision_pipeline_process_called(self):
        """executor.py should call _vision_pipeline.process()."""
        self.assertIn("_vision_pipeline.process(", self.executor_src)

    def test_vision_delegation_fail_safe(self):
        """Vision delegation failure should be caught gracefully."""
        self.assertIn("Vision delegation failed", self.executor_src)


# ===========================================================================
# PART 4: Frontend
# ===========================================================================


class TestFrontendTypes(unittest.TestCase):
    """Validate TypeScript type additions."""

    def setUp(self):
        self.types_src = _read(os.path.join(FRONTEND_SRC, "lib", "types.ts"))

    def test_vision_delegation_info_interface(self):
        """types.ts should have VisionDelegationInfo interface."""
        self.assertIn("interface VisionDelegationInfo", self.types_src)
        self.assertIn("vision_model: string", self.types_src)
        self.assertIn("description_length: number", self.types_src)
        self.assertIn("duration_ms: number", self.types_src)

    def test_message_item_has_vision_delegation(self):
        """MessageItem should have optional vision_delegation field."""
        self.assertIn("vision_delegation?: VisionDelegationInfo", self.types_src)

    def test_chat_token_has_vision_delegation_type(self):
        """ChatToken type union should include vision_delegation."""
        self.assertIn("'vision_delegation'", self.types_src)

    def test_callbacks_has_on_vision_delegation(self):
        """ChatStreamCallbacks should have onVisionDelegation."""
        self.assertIn("onVisionDelegation", self.types_src)


class TestFrontendChatApi(unittest.TestCase):
    """Validate chat.ts API updates."""

    def setUp(self):
        self.chat_api_src = _read(os.path.join(API_TS_DIR, "chat.ts"))

    def test_vision_delegation_case_in_stream(self):
        """chat.ts streamChat should handle vision_delegation messages."""
        self.assertIn("case 'vision_delegation':", self.chat_api_src)
        self.assertIn("onVisionDelegation", self.chat_api_src)


class TestFrontendChatStore(unittest.TestCase):
    """Validate chat store updates."""

    def setUp(self):
        self.store_src = _read(os.path.join(STORES_DIR, "chat.ts"))

    def test_streaming_vision_delegation_store(self):
        """Chat store should export streamingVisionDelegation."""
        self.assertIn("streamingVisionDelegation", self.store_src)

    def test_vision_delegation_reset_on_send(self):
        """streamingVisionDelegation should be reset in sendMessage."""
        self.assertIn("streamingVisionDelegation.set(null)", self.store_src)

    def test_on_vision_delegation_callback(self):
        """Chat store should have onVisionDelegation callback."""
        self.assertIn("onVisionDelegation:", self.store_src)


class TestFrontendComponents(unittest.TestCase):
    """Validate Svelte components."""

    def test_vision_delegation_indicator_exists(self):
        """VisionDelegationIndicator.svelte should exist."""
        path = os.path.join(CHAT_DIR, "VisionDelegationIndicator.svelte")
        self.assertTrue(os.path.isfile(path))

    def test_indicator_uses_onion_loader(self):
        """VisionDelegationIndicator should use OnionLoader."""
        src = _read(os.path.join(CHAT_DIR, "VisionDelegationIndicator.svelte"))
        self.assertIn("OnionLoader", src)

    def test_indicator_shows_analyzing_text(self):
        """VisionDelegationIndicator should display analyzing message."""
        src = _read(os.path.join(CHAT_DIR, "VisionDelegationIndicator.svelte"))
        self.assertIn("Analyzing image", src)

    def test_indicator_no_hardcoded_hex(self):
        """VisionDelegationIndicator should have no hardcoded hex colors."""
        src = _read(os.path.join(CHAT_DIR, "VisionDelegationIndicator.svelte"))
        # Match hex colors like #fff, #1a2b3c but not inside var() or comments
        hex_matches = re.findall(r'(?<!var\()#[0-9a-fA-F]{3,8}\b', src)
        self.assertEqual(hex_matches, [], f"Hardcoded hex found: {hex_matches}")

    def test_chat_message_vision_badge(self):
        """ChatMessage should display vision delegation badge."""
        src = _read(os.path.join(CHAT_DIR, "ChatMessage.svelte"))
        self.assertIn("vision_delegation", src)
        self.assertIn("Image analyzed by", src)

    def test_chat_page_imports_indicator(self):
        """Chat page should import VisionDelegationIndicator."""
        src = _read(os.path.join(ROUTES_DIR, "chat", "[id]", "+page.svelte"))
        self.assertIn("VisionDelegationIndicator", src)

    def test_chat_page_imports_vision_store(self):
        """Chat page should import streamingVisionDelegation."""
        src = _read(os.path.join(ROUTES_DIR, "chat", "[id]", "+page.svelte"))
        self.assertIn("streamingVisionDelegation", src)

    def test_chat_page_shows_indicator_during_analyzing(self):
        """Chat page should conditionally render indicator when analyzing."""
        src = _read(os.path.join(ROUTES_DIR, "chat", "[id]", "+page.svelte"))
        self.assertIn("status === 'analyzing'", src)


# ===========================================================================
# PART 5: Routes
# ===========================================================================


class TestRoutesChatVision(unittest.TestCase):
    """Validate routes_chat.py vision delegation events."""

    def setUp(self):
        self.routes_src = _read(os.path.join(API_DIR, "routes_chat.py"))

    def test_vision_delegation_ws_event(self):
        """routes_chat.py should emit vision_delegation WS events."""
        self.assertIn('"vision_delegation"', self.routes_src)

    def test_vision_meta_in_done_metadata(self):
        """done_metadata should include vision_delegation info."""
        self.assertIn('done_metadata["vision_delegation"]', self.routes_src)

    def test_on_status_callback_captures_vision(self):
        """_generate should define _on_status capturing vision events."""
        self.assertIn("_on_status", self.routes_src)
        self.assertIn("Analyzing image", self.routes_src)

    def test_last_vision_meta_accessed(self):
        """routes_chat.py should access executor.last_vision_meta."""
        self.assertIn("last_vision_meta", self.routes_src)


# ===========================================================================
# PART 6: Version & File Existence
# ===========================================================================


class TestVersionAndFiles(unittest.TestCase):
    """Validate version bump and file existence."""

    def test_vision_pipeline_py_exists(self):
        """vision_pipeline.py should exist."""
        self.assertTrue(os.path.isfile(os.path.join(BACKEND_DIR, "vision_pipeline.py")))

    def test_vision_pipeline_has_feature_flag(self):
        """Module should export VISION_PIPELINE_AVAILABLE."""
        src = _read(os.path.join(BACKEND_DIR, "vision_pipeline.py"))
        self.assertIn("VISION_PIPELINE_AVAILABLE = True", src)

    def test_version_bump(self):
        """app.py should have version 1.9.7."""
        src = _read(os.path.join(API_DIR, "app.py"))
        self.assertIn('version="1.10.0"', src)

    def test_no_french_in_vision_pipeline(self):
        """vision_pipeline.py should have no French comments."""
        src = _read(os.path.join(BACKEND_DIR, "vision_pipeline.py"))
        french_chars = re.findall(r'#.*[àéèêëùûüôîïçÀÉÈÊËÙÛÜÔÎÏÇ]', src)
        self.assertEqual(french_chars, [], f"French found: {french_chars}")

    def test_no_emoji_in_vision_pipeline(self):
        """vision_pipeline.py should have no emojis."""
        src = _read(os.path.join(BACKEND_DIR, "vision_pipeline.py"))
        emoji_pattern = re.compile(
            "[\U0001F300-\U0001F9FF\U00002702-\U000027B0\U0001FA00-\U0001FA6F]"
        )
        self.assertFalse(emoji_pattern.search(src), "Emoji found in vision_pipeline.py")


if __name__ == "__main__":
    unittest.main()
