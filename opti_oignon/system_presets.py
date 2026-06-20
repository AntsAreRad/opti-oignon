#!/usr/bin/env python3
"""
SYSTEM PRESETS - OPTI-OIGNON S84
=================================

Infrastructure-level presets that configure multiple YAML files at once.
Handles detection of installed Ollama models, preset recommendation based
on available hardware/models, and applying preset config overrides.

Separate from task presets (presets.py) which handle task routing.

Author: Leon
"""

import copy
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CONFIG_DIR, DATA_DIR, load_yaml, save_yaml

logger = logging.getLogger(__name__)

# Path to system presets definition
# Primary: opti_oignon/data/ (DATA_DIR), fallback: project root data/
SYSTEM_PRESETS_FILE = DATA_DIR / "system_presets.yaml"
_PROJECT_ROOT_DATA = Path(__file__).parent.parent / "data" / "system_presets.yaml"
if not SYSTEM_PRESETS_FILE.exists() and _PROJECT_ROOT_DATA.exists():
    SYSTEM_PRESETS_FILE = _PROJECT_ROOT_DATA


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelInfo:
    """Detected Ollama model with size metadata."""
    name: str
    size_bytes: int = 0
    parameter_count_b: float = 0.0
    quantization: str = ""
    family: str = ""

    @property
    def size_category(self) -> str:
        """Classify model as small/medium/large based on parameter count."""
        if self.parameter_count_b <= 10:
            return "small"
        elif self.parameter_count_b <= 35:
            return "medium"
        else:
            return "large"


@dataclass
class SystemPreset:
    """Infrastructure preset definition."""
    id: str
    name: str
    description: str
    icon: str
    recommended_vram_gb: int
    recommended_ram_gb: int
    config_overrides: dict[str, Any]
    model_strategy: str  # smallest, medium, largest
    pipelines: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "recommended_vram_gb": self.recommended_vram_gb,
            "recommended_ram_gb": self.recommended_ram_gb,
            "config_overrides": self.config_overrides,
            "model_strategy": self.model_strategy,
            "pipelines": self.pipelines,
        }


@dataclass
class DetectionResult:
    """Result of model detection and preset recommendation."""
    models: list[ModelInfo]
    recommended_preset: str
    reason: str
    model_counts: dict[str, int] = field(default_factory=dict)
    total_estimated_vram_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "models": [
                {
                    "name": m.name,
                    "size_bytes": m.size_bytes,
                    "parameter_count_b": m.parameter_count_b,
                    "quantization": m.quantization,
                    "family": m.family,
                    "size_category": m.size_category,
                }
                for m in self.models
            ],
            "recommended_preset": self.recommended_preset,
            "reason": self.reason,
            "model_counts": self.model_counts,
            "total_estimated_vram_gb": self.total_estimated_vram_gb,
        }


# =============================================================================
# MODEL DETECTION HELPERS
# =============================================================================

def _parse_parameter_count(model_name: str, model_details: Any = None) -> float:
    """
    Extract approximate parameter count in billions from model name or details.

    Parses patterns like '7b', '13b', '30b', '32b', '70b' from the model name.
    Falls back to size-based estimation if no pattern found.
    """
    # Try to extract from model name
    match = re.search(r'(\d+(?:\.\d+)?)\s*[bB]', model_name)
    if match:
        return float(match.group(1))

    # Try details if available (ollama show response)
    if model_details and hasattr(model_details, 'parameter_size'):
        size_str = str(model_details.parameter_size)
        match = re.search(r'(\d+(?:\.\d+)?)', size_str)
        if match:
            return float(match.group(1))

    # Fallback: estimate from file size (very rough)
    return 0.0


def _parse_quantization(model_name: str) -> str:
    """Extract quantization level from model name."""
    quant_patterns = [
        r'[qQ](\d+)_([kK]\w*)',  # Q4_K_M, q5_K_S
        r'[qQ](\d+)',             # Q4, q8
        r'[fF](\d+)',             # f16, f32
    ]
    for pattern in quant_patterns:
        match = re.search(pattern, model_name)
        if match:
            return match.group(0)
    return "unknown"


def _parse_family(model_name: str) -> str:
    """Extract model family from name."""
    name_lower = model_name.lower().split(":")[0]
    families = [
        "qwen", "llama", "deepseek", "phi", "mistral", "gemma",
        "codellama", "starcoder", "llava", "mxbai", "nomic",
    ]
    for family in families:
        if family in name_lower:
            return family
    return name_lower.split("/")[-1].split("-")[0]


def detect_ollama_models() -> list[ModelInfo]:
    """
    Detect installed Ollama models and extract metadata.

    Returns:
        List of ModelInfo with size and parameter data.
    """
    try:
        import ollama
        response = ollama.list()

        # Handle different ollama-python versions
        if hasattr(response, "models"):
            raw_models = response.models or []
        elif isinstance(response, dict):
            raw_models = response.get("models", [])
        else:
            raw_models = list(response) if response else []

        models = []
        for m in raw_models:
            # ollama-python < 0.4 uses .name, >= 0.4 uses .model
            name = ""
            if hasattr(m, "model") and m.model:
                name = m.model
            elif hasattr(m, "name") and m.name:
                name = m.name
            elif isinstance(m, dict):
                name = m.get("model", "") or m.get("name", "")
            if not name:
                continue

            size = getattr(m, "size", 0) or (m.get("size", 0) if isinstance(m, dict) else 0)

            info = ModelInfo(
                name=name,
                size_bytes=size,
                parameter_count_b=_parse_parameter_count(name, m),
                quantization=_parse_quantization(name),
                family=_parse_family(name),
            )
            models.append(info)

        return models

    except ImportError:
        logger.warning("ollama package not available for model detection")
        return []
    except Exception as e:
        logger.warning("Failed to detect Ollama models: %s", e)
        return []


def _select_model_by_strategy(models: list[ModelInfo], strategy: str) -> str | None:
    """
    Select a model name from detected models based on strategy.

    Args:
        models: List of detected models.
        strategy: One of 'smallest', 'medium', 'largest'.

    Returns:
        Model name string or None.
    """
    # Filter out embedding models and vision-only models for main selection
    chat_models = [
        m for m in models
        if "embed" not in m.name.lower()
        and m.parameter_count_b > 0
    ]

    if not chat_models:
        # Fallback: use all models
        chat_models = [m for m in models if "embed" not in m.name.lower()]

    if not chat_models:
        return None

    sorted_models = sorted(chat_models, key=lambda m: m.parameter_count_b)

    if strategy == "smallest":
        return sorted_models[0].name
    elif strategy == "largest":
        return sorted_models[-1].name
    elif strategy == "medium":
        # Pick middle or closest to median
        mid_idx = len(sorted_models) // 2
        return sorted_models[mid_idx].name
    else:
        return sorted_models[0].name


# =============================================================================
# SYSTEM PRESETS MANAGER
# =============================================================================

class SystemPresetsManager:
    """
    Manages infrastructure-level system presets.

    Loads preset definitions from data/system_presets.yaml,
    detects models, recommends presets, and applies config overrides.
    """

    def __init__(self, presets_file: Path | None = None):
        self._file = presets_file or SYSTEM_PRESETS_FILE
        self._presets: dict[str, SystemPreset] = {}
        self._onboarding: dict[str, Any] = {
            "user_initialized": False,
            "applied_preset": None,
            "applied_at": None,
        }
        self._load()

    def _load(self) -> None:
        """Load system presets from YAML file."""
        if not self._file.exists():
            logger.warning("System presets file not found: %s", self._file)
            return

        data = load_yaml(self._file)
        if not data:
            return

        # Load presets
        presets_data = data.get("system_presets", {})
        for preset_id, pdata in presets_data.items():
            if not isinstance(pdata, dict):
                continue
            self._presets[preset_id] = SystemPreset(
                id=preset_id,
                name=pdata.get("name", preset_id),
                description=pdata.get("description", ""),
                icon=pdata.get("icon", ""),
                recommended_vram_gb=pdata.get("recommended_vram_gb", 0),
                recommended_ram_gb=pdata.get("recommended_ram_gb", 0),
                config_overrides=pdata.get("config_overrides", {}),
                model_strategy=pdata.get("model_strategy", "smallest"),
                pipelines=pdata.get("pipelines", ["direct"]),
            )

        # Load onboarding state
        onboarding = data.get("onboarding", {})
        if isinstance(onboarding, dict):
            self._onboarding.update(onboarding)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def list_presets(self) -> list[SystemPreset]:
        """Return all system presets in order: minimal, balanced, power."""
        order = ["minimal", "balanced", "power"]
        result = []
        for pid in order:
            if pid in self._presets:
                result.append(self._presets[pid])
        # Add any remaining presets not in standard order
        for pid, preset in self._presets.items():
            if pid not in order:
                result.append(preset)
        return result

    def get_preset(self, preset_id: str) -> SystemPreset | None:
        """Get a preset by ID."""
        return self._presets.get(preset_id)

    def detect_and_recommend(self) -> DetectionResult:
        """
        Detect installed Ollama models and recommend a system preset.

        Returns:
            DetectionResult with models, recommendation, and reason.
        """
        models = detect_ollama_models()

        # Count by size category
        counts = {"small": 0, "medium": 0, "large": 0}
        for m in models:
            cat = m.size_category
            if cat in counts:
                counts[cat] += 1

        # Estimate total VRAM needed (rough: 1B params ~ 0.5-1 GB in Q4)
        total_vram = sum(m.parameter_count_b * 0.6 for m in models if m.parameter_count_b > 0)

        # Recommendation logic
        if not models:
            recommended = "minimal"
            reason = "No Ollama models detected. Minimal preset recommended until models are installed."
        elif counts["large"] >= 2:
            recommended = "power"
            reason = (
                f"Found {counts['large']} large model(s) (30B+ params). "
                "Power preset recommended for full-featured experience."
            )
        elif counts["medium"] >= 1:
            recommended = "balanced"
            reason = (
                f"Found {counts['medium']} medium model(s) (10-35B params). "
                "Balanced preset recommended for optimal quality/performance ratio."
            )
        elif counts["small"] >= 1:
            recommended = "minimal"
            reason = (
                f"Found {counts['small']} small model(s) (<10B params). "
                "Minimal preset recommended to conserve resources."
            )
        elif counts["large"] == 1:
            # Single large model, no medium/small. Every detected model is
            # categorized (a 0-param/unparseable model classifies as "small"),
            # so this is the only state reachable with models present and no
            # small/medium and large < 2 -- the else reason below ("could not
            # determine sizes") never actually applied to it. Tier kept at the
            # prior default (balanced); see PRS-01/PRS-02 for a hardware-aware
            # recommendation that could prefer "power" here.
            recommended = "balanced"
            reason = (
                "Found 1 large model (30B+ params). "
                "Balanced preset recommended as a safe default for a single large model."
            )
        else:
            recommended = "balanced"
            reason = "Could not determine model sizes. Balanced preset recommended as safe default."

        return DetectionResult(
            models=models,
            recommended_preset=recommended,
            reason=reason,
            model_counts=counts,
            total_estimated_vram_gb=round(total_vram, 1),
        )

    def apply_preset(
        self,
        preset_id: str,
        detected_models: list[ModelInfo] | None = None,
        checkpoint_before_apply: bool = True,
    ) -> dict[str, Any]:
        """
        Apply a system preset by writing config overrides to YAML files.

        Args:
            preset_id: ID of the system preset to apply.
            detected_models: Optional pre-detected models list.
            checkpoint_before_apply: Always True (hardcoded safety).

        Returns:
            Dict with applied config details and any warnings.
        """
        # checkpoint_before_apply is ALWAYS True (hardcoded, cannot be overridden)
        checkpoint_before_apply = True

        preset = self._presets.get(preset_id)
        if not preset:
            return {"error": f"Unknown preset: {preset_id}", "applied": False}

        # Detect models if not provided
        if detected_models is None:
            detected_models = detect_ollama_models()

        # Select default model based on strategy
        selected_model = _select_model_by_strategy(detected_models, preset.model_strategy)
        warnings = []

        if not selected_model:
            warnings.append("No suitable models found. Config files updated but no model assigned.")

        # Checkpoint existing configs (backup)
        applied_configs = {}
        backup_dir = DATA_DIR / "config_backups"
        if checkpoint_before_apply:
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            for config_name in preset.config_overrides:
                config_file = CONFIG_DIR / f"{config_name}.yaml"
                if config_file.exists():
                    backup_file = backup_dir / f"{config_name}_{timestamp}.yaml"
                    try:
                        import shutil
                        shutil.copy2(config_file, backup_file)
                    except Exception as e:
                        warnings.append(f"Failed to backup {config_name}.yaml: {e}")

        # Apply config overrides
        for config_name, overrides in preset.config_overrides.items():
            config_file = CONFIG_DIR / f"{config_name}.yaml"

            # Load existing config
            existing = {}
            if config_file.exists():
                existing = load_yaml(config_file) or {}

            # Deep merge overrides into existing
            merged = _deep_merge(existing, overrides)
            if save_yaml(config_file, merged):
                applied_configs[config_name] = list(overrides.keys())
            else:
                warnings.append(f"Failed to write {config_name}.yaml")

        # Update models.yaml default model if we have one
        if selected_model:
            _update_default_model(selected_model)

        # Mark onboarding as complete
        self._onboarding["user_initialized"] = True
        self._onboarding["applied_preset"] = preset_id
        self._onboarding["applied_at"] = datetime.now(timezone.utc).isoformat()
        self._save_onboarding()

        return {
            "applied": True,
            "preset_id": preset_id,
            "preset_name": preset.name,
            "selected_model": selected_model,
            "applied_configs": applied_configs,
            "pipelines": preset.pipelines,
            "warnings": warnings,
        }

    def is_initialized(self) -> bool:
        """Check if user has completed onboarding (applied a system preset)."""
        return bool(self._onboarding.get("user_initialized", False))

    def get_onboarding_state(self) -> dict[str, Any]:
        """Return current onboarding state."""
        return dict(self._onboarding)

    def reset_onboarding(self) -> None:
        """Reset onboarding state (for testing or re-onboarding)."""
        self._onboarding = {
            "user_initialized": False,
            "applied_preset": None,
            "applied_at": None,
        }
        self._save_onboarding()

    def _save_onboarding(self) -> None:
        """Persist onboarding state back to the system presets YAML."""
        if not self._file.exists():
            return

        data = load_yaml(self._file) or {}
        data["onboarding"] = self._onboarding
        save_yaml(self._file, data)


# =============================================================================
# HELPERS
# =============================================================================

def _deep_merge(base: dict, overrides: dict) -> dict:
    """
    Deep merge overrides into base dict.

    Nested dicts are merged recursively; other values are replaced.

    S171: defensive validation. Both arguments must be dicts, and a key that
    is a mapping on one side but a non-mapping on the other is a structural
    type conflict that would silently corrupt nested configuration -- such a
    conflict raises TypeError with the offending key instead of being applied
    blindly. Scalar-to-scalar replacement (including a changed scalar type) is
    still permitted.
    """
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        raise TypeError(
            "_deep_merge expects two dicts, got "
            f"{type(base).__name__} and {type(overrides).__name__}"
        )
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result:
            existing = result[key]
            existing_is_map = isinstance(existing, dict)
            value_is_map = isinstance(value, dict)
            if existing_is_map != value_is_map:
                raise TypeError(
                    f"_deep_merge type conflict at key '{key}': cannot merge "
                    f"{type(value).__name__} into {type(existing).__name__}"
                )
            if existing_is_map and value_is_map:
                result[key] = _deep_merge(existing, value)
            else:
                result[key] = copy.deepcopy(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _update_default_model(model_name: str) -> None:
    """Update the default model in models.yaml routing config."""
    models_file = CONFIG_DIR / "models.yaml"
    if not models_file.exists():
        return

    data = load_yaml(models_file) or {}
    routing = data.get("routing", {})

    # Update primary model for all task types
    for task_type in routing:
        if isinstance(routing[task_type], dict):
            routing[task_type]["primary"] = model_name

    # Update default_model if it exists
    if "default_model" in data:
        data["default_model"] = model_name

    save_yaml(models_file, data)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

try:
    system_presets_manager = SystemPresetsManager()
    SYSTEM_PRESETS_AVAILABLE = True
except Exception as e:
    logger.warning("Failed to initialize SystemPresetsManager: %s", e)
    system_presets_manager = None
    SYSTEM_PRESETS_AVAILABLE = False
