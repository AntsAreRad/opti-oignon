#!/usr/bin/env python3
"""
Transparent vision pipeline (S95).

When the user sends an image to a non-vision model, this pipeline:
1. Detects the delegation need (image present + model lacks vision).
2. Calls the user's preferred vision model to describe the image.
3. Injects the description into the user message.
4. Returns the augmented message so the original model can respond
   with full image context.

The vision model selection respects the VisionConfig from S94:
- User can pick a specific model in Settings > Vision Model.
- "auto" mode detects the first available vision-capable model.
- Manual known_vision_models list is honored.

Configuration lives in config/vision.yaml (delegation_* keys).
"""

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / "config"
_CONFIG_FILE = _CONFIG_DIR / "vision.yaml"

# Default delegation settings
_DEFAULT_INJECT_FORMAT = (
    "[Image analysis: {description}]\nUser question: {message}"
)
_DEFAULT_MAX_DESCRIPTION_TOKENS = 500
_DEFAULT_DELEGATION_ENABLED = True

try:
    import ollama as _ollama_module
    OLLAMA_AVAILABLE = True
except ImportError:
    _ollama_module = None  # type: ignore[assignment]
    OLLAMA_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

try:
    from opti_oignon.vision_config import vision_config as _default_vision_config
    VISION_CONFIG_AVAILABLE = True
except ImportError:
    _default_vision_config = None  # type: ignore[assignment]
    VISION_CONFIG_AVAILABLE = False


class VisionPipeline:
    """Transparent image delegation pipeline.

    Delegates image analysis to a vision-capable model when the current
    conversation model does not support images.  The resulting description
    is injected into the user message so the original model can reason
    about image content.

    Args:
        vision_config: VisionConfig instance for model selection.
            Falls back to the module-level singleton when None.
        ollama_module: Ollama client module (for testing injection).
        config_path: Path to vision.yaml for delegation settings.
    """

    def __init__(
        self,
        vision_config=None,
        ollama_module=None,
        config_path: Optional[Path] = None,
    ) -> None:
        self._vision_config = vision_config or _default_vision_config
        self._ollama = ollama_module or _ollama_module
        self._config_path = config_path or _CONFIG_FILE

        # Delegation settings (loaded from YAML)
        self._delegation_enabled: bool = _DEFAULT_DELEGATION_ENABLED
        self._inject_format: str = _DEFAULT_INJECT_FORMAT
        self._max_description_tokens: int = _DEFAULT_MAX_DESCRIPTION_TOKENS

        self._load_delegation_config()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _load_delegation_config(self) -> None:
        """Load delegation-specific settings from vision.yaml."""
        if not YAML_AVAILABLE or yaml is None:
            return
        if not self._config_path.exists():
            return
        try:
            with open(self._config_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

            delegation = data.get("delegation_enabled")
            if isinstance(delegation, bool):
                self._delegation_enabled = delegation

            fmt = data.get("inject_format")
            if isinstance(fmt, str) and fmt.strip():
                self._inject_format = fmt.strip()

            max_tok = data.get("max_description_tokens")
            if isinstance(max_tok, int) and max_tok > 0:
                self._max_description_tokens = max_tok

        except Exception as exc:
            logger.warning("Failed to load delegation config: %s", exc)

    @property
    def delegation_enabled(self) -> bool:
        """Whether automatic vision delegation is active."""
        return self._delegation_enabled

    @delegation_enabled.setter
    def delegation_enabled(self, value: bool) -> None:
        self._delegation_enabled = value

    @property
    def inject_format(self) -> str:
        """Format string for injecting image description into user message."""
        return self._inject_format

    @property
    def max_description_tokens(self) -> int:
        """Maximum tokens for the vision model description."""
        return self._max_description_tokens

    # -----------------------------------------------------------------
    # Model resolution helpers
    # -----------------------------------------------------------------

    def _list_available_models(self) -> list[str]:
        """Retrieve available model names from Ollama."""
        if self._ollama is None:
            return []
        try:
            response = self._ollama.list()
            models_list = []
            if isinstance(response, dict):
                raw = response.get("models", [])
            elif hasattr(response, "models"):
                raw = response.models or []
            else:
                raw = []
            for m in raw:
                name = ""
                if isinstance(m, dict):
                    name = m.get("model", "") or m.get("name", "")
                elif hasattr(m, "model"):
                    name = m.model or getattr(m, "name", "") or ""
                if name:
                    models_list.append(str(name))
            return models_list
        except Exception as exc:
            logger.warning("Failed to list models: %s", exc)
            return []

    def _resolve_vision_model(self) -> Optional[str]:
        """Resolve the effective vision model using VisionConfig.

        Respects the user's preferred selection from Settings > Vision Model.
        Falls back to auto-detect if set to 'auto'.
        """
        if self._vision_config is None:
            return None
        available = self._list_available_models()
        return self._vision_config.get_effective_model(available)

    def _is_vision_capable(self, model_name: str) -> bool:
        """Check whether a model has vision capabilities."""
        if self._vision_config is None:
            return False
        return self._vision_config.is_vision_model(model_name)

    # -----------------------------------------------------------------
    # Core pipeline methods
    # -----------------------------------------------------------------

    def detect_needs_delegation(
        self,
        message: str,
        images: list[str] | None,
        current_model: str,
    ) -> bool:
        """Determine if vision delegation is needed.

        Delegation is needed when ALL of:
        - delegation is enabled
        - images are present and non-empty
        - the current model is NOT vision-capable
        - a vision model is available

        Args:
            message: User message text.
            images: List of base64-encoded image strings.
            current_model: Name of the model currently selected for conversation.

        Returns:
            True if delegation should occur.
        """
        if not self._delegation_enabled:
            return False
        if not images:
            return False
        if self._is_vision_capable(current_model):
            # Current model handles images natively, no delegation needed
            return False
        vision_model = self._resolve_vision_model()
        if not vision_model:
            logger.warning(
                "Vision delegation needed but no vision model available"
            )
            return False
        return True

    def describe_image(
        self,
        image_data: list[str],
        user_prompt: str,
        vision_model: Optional[str] = None,
    ) -> str:
        """Call the vision model to describe the image(s).

        Sends a non-streaming request to the vision model with the image(s)
        and a description prompt.  The prompt is built from the user's
        describe_prompt template (from VisionConfig) combined with the
        user's actual question for focused descriptions.

        Args:
            image_data: List of base64-encoded image strings.
            user_prompt: The user's original question/message.
            vision_model: Vision model to use. If None, resolves automatically
                from VisionConfig (respects user preference).

        Returns:
            Text description of the image(s). Empty string on failure.
        """
        if self._ollama is None:
            logger.error("Ollama not available for vision delegation")
            return ""

        model = vision_model or self._resolve_vision_model()
        if not model:
            logger.error("No vision model available for description")
            return ""

        # Build the description prompt
        base_prompt = ""
        if self._vision_config is not None:
            base_prompt = self._vision_config.describe_prompt
        if not base_prompt:
            base_prompt = (
                "Describe this image in detail, focusing on its content, "
                "structure, and any relevant text or data visible."
            )

        # Combine base prompt with user context for focused description
        if user_prompt.strip():
            prompt = f"{base_prompt}\n\nUser context: {user_prompt}"
        else:
            prompt = base_prompt

        try:
            start = time.monotonic()
            response = self._ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": image_data,
                    }
                ],
                options={
                    "num_predict": self._max_description_tokens,
                },
                stream=False,
            )
            elapsed = time.monotonic() - start

            # Extract content from response
            content = ""
            if isinstance(response, dict):
                msg = response.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                elif hasattr(msg, "content"):
                    content = msg.content or ""
            elif hasattr(response, "message"):
                msg = response.message
                if hasattr(msg, "content"):
                    content = msg.content or ""
                elif isinstance(msg, dict):
                    content = msg.get("content", "")

            description = str(content).strip()
            logger.info(
                "Vision delegation: %s described %d image(s) in %.1fs (%d chars)",
                model, len(image_data), elapsed, len(description),
            )
            return description

        except Exception as exc:
            logger.error("Vision description failed with %s: %s", model, exc)
            return ""

    def build_augmented_message(
        self,
        original_message: str,
        image_description: str,
    ) -> str:
        """Build the augmented user message with injected image description.

        Uses the configurable inject_format template. The template supports
        {description} and {message} placeholders.

        Args:
            original_message: The user's original message text.
            image_description: Text description from the vision model.

        Returns:
            Augmented message string with image context.
        """
        if not image_description:
            return original_message

        try:
            return self._inject_format.format(
                description=image_description,
                message=original_message,
            )
        except (KeyError, IndexError, ValueError):
            # VIS-02: fall back on any format error. A malformed inject_format
            # template (unknown placeholder -> KeyError, positional ref ->
            # IndexError, stray "{" or bad spec -> ValueError) must not let an
            # image description be dropped or crash the pipeline.
            return (
                f"[Image analysis: {image_description}]\n"
                f"User question: {original_message}"
            )

    def process(
        self,
        message: str,
        images: list[str] | None,
        current_model: str,
        on_status: Optional[callable] = None,
    ) -> tuple[str, list[str] | None, dict]:
        """Run the full vision delegation pipeline.

        This is the main entry point called from the executor. It checks
        whether delegation is needed, describes the image(s) via the
        user's preferred vision model, and returns an augmented message.

        BUG-09 S108: When images are present but no vision model is available
        and the current model is not vision-capable, strip images and inject
        a user-friendly warning instead of passing images through to crash.

        Args:
            message: User message text.
            images: List of base64-encoded image strings (may be None).
            current_model: Currently selected conversation model.
            on_status: Optional callback for status updates.

        Returns:
            Tuple of:
            - augmented_message: Original or augmented message string.
            - remaining_images: None if delegation consumed the images,
              original images list if no delegation occurred.
            - metadata: Dict with delegation details (empty if no delegation).
                Keys: delegated (bool), vision_model (str), description_length (int),
                duration_ms (float).
        """
        empty_meta: dict = {}

        if not self.detect_needs_delegation(message, images, current_model):
            # BUG-09 S108: If images are present but model is not vision-capable
            # and delegation didn't trigger (no vision model / disabled),
            # strip images to prevent 500 from non-vision model.
            if images and not self._is_vision_capable(current_model):
                vision_model = self._resolve_vision_model()
                if not vision_model:
                    logger.warning(
                        "Images provided but no vision-capable model available. "
                        "Stripping images to avoid model error."
                    )
                    if on_status:
                        on_status(
                            "No vision-capable model found. "
                            "Install llava, llama3.2-vision, or similar to analyze images."
                        )
                    warning_meta = {
                        "delegated": False,
                        "vision_error": "no_vision_model",
                        "vision_warning": (
                            "No vision-capable model available. "
                            "Install llava, llama3.2-vision, or similar."
                        ),
                    }
                    return message, None, warning_meta
            return message, images, empty_meta

        vision_model = self._resolve_vision_model()
        if not vision_model:
            return message, images, empty_meta

        if on_status:
            on_status(f"Analyzing image with {vision_model}...")

        start = time.monotonic()
        description = self.describe_image(images, message, vision_model)
        elapsed_ms = (time.monotonic() - start) * 1000

        if not description:
            logger.warning("Vision delegation produced empty description")
            return message, images, empty_meta

        augmented = self.build_augmented_message(message, description)

        metadata = {
            "delegated": True,
            "vision_model": vision_model,
            "description_length": len(description),
            "duration_ms": round(elapsed_ms, 1),
        }

        logger.info(
            "Vision delegation complete: %s -> augmented message (%d chars)",
            vision_model, len(augmented),
        )

        # Images are consumed by the vision model; the text model
        # receives only the augmented text description.
        return augmented, None, metadata

    def to_dict(self) -> dict:
        """Serialize pipeline state for API responses."""
        vision_model = self._resolve_vision_model()
        return {
            "delegation_enabled": self._delegation_enabled,
            "inject_format": self._inject_format,
            "max_description_tokens": self._max_description_tokens,
            "effective_vision_model": vision_model,
        }


# Module-level singleton
vision_pipeline = VisionPipeline()

VISION_PIPELINE_AVAILABLE = True
