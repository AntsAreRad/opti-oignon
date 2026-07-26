#!/usr/bin/env python3
"""
Vision model configuration manager.

Handles selection and persistence of the vision model used for image
analysis. Three detection strategies:

1. **capabilities** -- Probes ollama.show() for each model and checks
   if details.families contains a vision family (e.g. "clip", "mllama").
   This catches models like qwen3.5 that handle images without having
   "vl" or "vision" in their name.

2. **patterns** -- Matches model names against known substrings
   (vl, vision, llava, etc.). Fast but incomplete.

3. **both** (default) -- Tries capabilities first, then patterns as
   fallback. Merges results and deduplicates.

Additionally, a manual known_vision_models list lets the user
force-declare any model as vision-capable.

Config persisted in opti_oignon/config/vision.yaml.
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent / "config"
_CONFIG_FILE = _CONFIG_DIR / "vision.yaml"

_DEFAULT_PATTERNS = ["vl", "vision", "llava", "bakllava", "moondream"]
_DEFAULT_VISION_FAMILIES = ["clip", "mllama"]
_DEFAULT_DESCRIBE_PROMPT = (
    "Describe this image in detail, focusing on its content, "
    "structure, and any relevant text or data visible."
)

# TTL for cached capability probes (seconds)
_CAPABILITY_CACHE_TTL = 300

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore[assignment]


class VisionConfig:
    """Manages vision model selection with multi-strategy detection."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or _CONFIG_FILE
        self._vision_model: str = "auto"
        self._detection_strategy: str = "both"
        self._auto_detect_patterns: list[str] = list(_DEFAULT_PATTERNS)
        self._vision_families: list[str] = list(_DEFAULT_VISION_FAMILIES)
        self._known_vision_models: list[str] = []
        self._describe_prompt: str = _DEFAULT_DESCRIBE_PROMPT
        # Cache: model_name -> (is_vision: bool, timestamp: float)
        self._capability_cache: dict[str, tuple[bool, float]] = {}
        self._load()

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _load(self) -> None:
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            logger.debug("PyYAML not available, using defaults")
            return
        if not self._config_path.exists():
            logger.debug("Vision config not found at %s, using defaults", self._config_path)
            return
        try:
            with open(self._config_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            self._vision_model = str(data.get("vision_model", "auto"))
            strategy = data.get("detection_strategy", "both")
            if strategy in ("capabilities", "patterns", "both"):
                self._detection_strategy = strategy
            patterns = data.get("auto_detect_patterns")
            if isinstance(patterns, list):
                self._auto_detect_patterns = [str(p) for p in patterns]
            families = data.get("vision_families")
            if isinstance(families, list):
                self._vision_families = [str(f) for f in families]
            known = data.get("known_vision_models")
            if isinstance(known, list):
                self._known_vision_models = [str(m) for m in known if m]
            prompt = data.get("describe_prompt")
            if isinstance(prompt, str) and prompt.strip():
                self._describe_prompt = prompt.strip()
        except Exception as exc:
            logger.warning("Failed to load vision config: %s", exc)

    def _save(self) -> None:
        """Persist current configuration to YAML file."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, cannot save vision config")
            return
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "vision_model": self._vision_model,
                "detection_strategy": self._detection_strategy,
                "auto_detect_patterns": self._auto_detect_patterns,
                "vision_families": self._vision_families,
                "known_vision_models": self._known_vision_models,
                "describe_prompt": self._describe_prompt,
            }
            with open(self._config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, default_flow_style=False, allow_unicode=True)
        except Exception as exc:
            logger.warning("Failed to save vision config: %s", exc)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def vision_model(self) -> str:
        """Return the configured vision model name or 'auto'."""
        return self._vision_model

    @vision_model.setter
    def vision_model(self, value: str) -> None:
        self._vision_model = value.strip() if value else "auto"
        self._save()

    @property
    def detection_strategy(self) -> str:
        """Return the detection strategy: capabilities, patterns, or both."""
        return self._detection_strategy

    @property
    def auto_detect_patterns(self) -> list[str]:
        """Return patterns used for name-based detection."""
        return list(self._auto_detect_patterns)

    @property
    def vision_families(self) -> list[str]:
        """Return capability families indicating vision support."""
        return list(self._vision_families)

    @property
    def known_vision_models(self) -> list[str]:
        """Return manually declared vision-capable models."""
        return list(self._known_vision_models)

    @known_vision_models.setter
    def known_vision_models(self, value: list[str]) -> None:
        self._known_vision_models = [m.strip() for m in value if m and m.strip()]
        self._save()

    @property
    def describe_prompt(self) -> str:
        """Return the default prompt template for image description."""
        return self._describe_prompt

    @describe_prompt.setter
    def describe_prompt(self, value: str) -> None:
        self._describe_prompt = value.strip() if value else _DEFAULT_DESCRIBE_PROMPT
        self._save()

    # -----------------------------------------------------------------
    # Capability probing via ollama.show()
    # -----------------------------------------------------------------

    def _probe_model_capabilities(self, model_name: str) -> bool:
        """Check if a model has vision capabilities via ollama.show().

        Looks for vision-related families (e.g. 'clip', 'mllama') in the
        model details returned by Ollama.

        Args:
            model_name: Ollama model name to probe.

        Returns:
            True if the model has vision capabilities.
        """
        now = time.monotonic()
        cached = self._capability_cache.get(model_name)
        if cached is not None:
            is_vision, ts = cached
            if now - ts < _CAPABILITY_CACHE_TTL:
                return is_vision

        is_vision = False
        try:
            import ollama
            info = ollama.show(model_name)

            # Extract families from details
            families: list[str] = []
            details = None
            if hasattr(info, "details"):
                details = info.details
            elif isinstance(info, dict):
                details = info.get("details")

            if details is not None:
                if isinstance(details, dict):
                    fam = details.get("families", [])
                    if isinstance(fam, list):
                        families = [str(f).lower() for f in fam]
                    single = details.get("family", "")
                    if single and str(single).lower() not in families:
                        families.append(str(single).lower())
                else:
                    fam = getattr(details, "families", None)
                    if isinstance(fam, list):
                        families = [str(f).lower() for f in fam]
                    single = getattr(details, "family", "")
                    if single and str(single).lower() not in families:
                        families.append(str(single).lower())

            # Check against known vision families
            vision_fam_lower = [f.lower() for f in self._vision_families]
            is_vision = any(f in vision_fam_lower for f in families)

            if is_vision:
                logger.debug(
                    "Model %s has vision capabilities (families: %s)",
                    model_name, families,
                )

        except ImportError:
            logger.debug("ollama not available, cannot probe %s", model_name)
        except Exception as exc:
            logger.debug("Failed to probe %s: %s", model_name, exc)

        self._capability_cache[model_name] = (is_vision, now)
        return is_vision

    # -----------------------------------------------------------------
    # Detection methods
    # -----------------------------------------------------------------

    def _detect_by_patterns(self, model_name: str) -> bool:
        """Check if model name matches any vision pattern."""
        name_lower = model_name.lower()
        return any(p.lower() in name_lower for p in self._auto_detect_patterns)

    def _is_known_vision(self, model_name: str) -> bool:
        """Check if model is in the manual known vision models list."""
        return model_name in self._known_vision_models

    def is_vision_model(self, model_name: str) -> bool:
        """Check if a model has vision capabilities.

        Uses the configured detection strategy plus the manual known list.

        Args:
            model_name: Model name to check.

        Returns:
            True if the model is vision-capable.
        """
        if self._is_known_vision(model_name):
            return True

        strategy = self._detection_strategy
        if strategy == "capabilities":
            return self._probe_model_capabilities(model_name)
        elif strategy == "patterns":
            return self._detect_by_patterns(model_name)
        else:  # "both"
            if self._probe_model_capabilities(model_name):
                return True
            return self._detect_by_patterns(model_name)

    def detect_vision_models(self, available_models: list[str]) -> list[str]:
        """Find all vision-capable models from a list.

        Args:
            available_models: List of model name strings from Ollama.

        Returns:
            List of vision-capable model names (deduplicated, order preserved).
        """
        seen: set[str] = set()
        result: list[str] = []
        for model_name in available_models:
            if model_name in seen:
                continue
            if self.is_vision_model(model_name):
                result.append(model_name)
                seen.add(model_name)
        return result

    def get_effective_model(self, available_models: list[str]) -> str | None:
        """Return the effective vision model to use.

        If configured to 'auto', runs detection. Otherwise returns
        the explicitly configured model (validated against available list).

        Args:
            available_models: List of model name strings from Ollama.

        Returns:
            The model name to use, or None if unavailable.
        """
        if self._vision_model == "auto":
            vision_models = self.detect_vision_models(available_models)
            return vision_models[0] if vision_models else None
        if self._vision_model in available_models:
            return self._vision_model
        logger.warning(
            "Configured vision model '%s' not found in available models",
            self._vision_model,
        )
        return None

    def clear_cache(self) -> None:
        """Clear the capability probe cache."""
        self._capability_cache.clear()

    def to_dict(self) -> dict:
        """Serialize config to dictionary for API responses."""
        return {
            "vision_model": self._vision_model,
            "detection_strategy": self._detection_strategy,
            "auto_detect_patterns": list(self._auto_detect_patterns),
            "vision_families": list(self._vision_families),
            "known_vision_models": list(self._known_vision_models),
            "describe_prompt": self._describe_prompt,
        }


# Module-level singleton
vision_config = VisionConfig()

VISION_CONFIG_AVAILABLE = True
