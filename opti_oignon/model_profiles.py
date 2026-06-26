#!/usr/bin/env python3
"""
MODEL PROFILES -- Smart Auto-Routing (S46/S54)
===============================================

Defines model capability profiles in a YAML-driven system.
Each model declares capabilities, strengths, weaknesses,
speed/quality tiers, task recommendations, and numeric
task_scores for smart routing.

The ModelProfileManager loads profiles from YAML, provides
task-based model lookup, and integrates with the SmartRouter
for transparent model selection.

Author: Leon
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Profile config file path
_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_PROFILES_PATH = _CONFIG_DIR / "model_profiles.yaml"

# Valid tiers
VALID_SPEED_TIERS = ("fast", "medium", "slow")
VALID_QUALITY_TIERS = ("high", "medium", "low")

# Known capabilities (non-restrictive, documentation purpose)
KNOWN_CAPABILITIES = (
    "code", "reasoning", "general", "multilingual", "vision",
    "fast", "creative", "long_context", "tool_use", "thinking",
)


# =============================================================================
# MODEL PROFILE
# =============================================================================

@dataclass
class ModelProfile:
    """Capability profile for an Ollama model."""
    name: str                                   # "qwen3:32b"
    display_name: str = ""                      # "Qwen3 32B"
    capabilities: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    context_window: int = 32768
    speed_tier: str = "medium"                  # fast, medium, slow
    quality_tier: str = "medium"                # high, medium, low
    recommended_for: list[str] = field(default_factory=list)
    not_recommended_for: list[str] = field(default_factory=list)
    # S54: Numeric task scores for smart routing (task_type -> 0.0-1.0)
    task_scores: dict[str, float] = field(default_factory=dict)
    # S54: Auto-detected model metadata
    parameter_count: str | None = None       # e.g. "32B"
    quantization: str | None = None          # e.g. "Q4_K_M"
    family: str | None = None                # e.g. "qwen3"
    auto_detected: bool = False                 # True if context_window was auto-detected

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.display_name:
            self.display_name = self.name
        # Normalize speed tier
        if self.speed_tier not in VALID_SPEED_TIERS:
            logger.warning(f"Profile {self.name}: invalid speed_tier '{self.speed_tier}', fallback 'medium'")
            self.speed_tier = "medium"
        if self.quality_tier not in VALID_QUALITY_TIERS:
            logger.warning(f"Profile {self.name}: invalid quality_tier '{self.quality_tier}', fallback 'medium'")
            self.quality_tier = "medium"
        # S54: Validate task_scores range
        if self.task_scores:
            for k, v in list(self.task_scores.items()):
                self.task_scores[k] = max(0.0, min(1.0, float(v)))

    def matches_task(self, task_type: str) -> bool:
        """Check if the model is recommended for a task type.

        Args:
            task_type: Task type (e.g. "code_python", "debug", "explanation")

        Returns:
            True if the model is recommended for this task
        """
        # Check exclusion first
        if task_type in self.not_recommended_for:
            return False
        # Check direct recommendation
        if task_type in self.recommended_for:
            return True
        # Check by prefix (e.g. "code" matches "code_python")
        for rec in self.recommended_for:
            if task_type.startswith(rec) or rec.startswith(task_type):
                return True
        return False

    def has_capability(self, capability: str) -> bool:
        """Check if the model has a given capability."""
        return capability in self.capabilities

    def score_for_task(self, task_type: str, requirements: list[str] | None = None) -> float:
        """Compute a relevance score for a task.

        The score combines task matching, required capabilities,
        and quality/speed tiers.

        Args:
            task_type: Task type
            requirements: Required capabilities (optional)

        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0

        # Explicit exclusion = zero score
        if task_type in self.not_recommended_for:
            return 0.0

        # Direct recommendation = major bonus
        if task_type in self.recommended_for:
            score += 0.5
        else:
            # Partial prefix match
            for rec in self.recommended_for:
                if task_type.startswith(rec) or rec.startswith(task_type):
                    score += 0.3
                    break

        # Quality bonus
        quality_bonus = {"high": 0.3, "medium": 0.15, "low": 0.05}
        score += quality_bonus.get(self.quality_tier, 0.1)

        # Required capabilities bonus
        if requirements:
            matched = sum(1 for req in requirements if req in self.capabilities)
            if requirements:
                score += 0.2 * (matched / len(requirements))

        return min(score, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert the profile to a dictionary."""
        d = {
            "name": self.name,
            "display_name": self.display_name,
            "capabilities": self.capabilities,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "context_window": self.context_window,
            "speed_tier": self.speed_tier,
            "quality_tier": self.quality_tier,
            "recommended_for": self.recommended_for,
            "not_recommended_for": self.not_recommended_for,
        }
        # S54: Include task_scores and metadata if present
        if self.task_scores:
            d["task_scores"] = dict(self.task_scores)
        if self.parameter_count:
            d["parameter_count"] = self.parameter_count
        if self.quantization:
            d["quantization"] = self.quantization
        if self.family:
            d["family"] = self.family
        if self.auto_detected:
            d["auto_detected"] = True
        return d


# =============================================================================
# ROUTING REASON
# =============================================================================

@dataclass
class RoutingReason:
    """Transparent explanation of a routing decision.

    Used by the RoutingIndicator in the frontend
    to show why a model was selected.
    """
    model: str                        # Selected model
    display_name: str = ""            # Human-readable name
    task_type: str = ""               # Detected task type
    pipeline: str = ""                # Agentic pipeline used
    reason: str = ""                  # Main reason (e.g. "Code task detected")
    score: float = 0.0                # Relevance score
    alternatives: list[str] = field(default_factory=list)
    profile_used: bool = False        # True if a profile was used

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for the API."""
        return {
            "model": self.model,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "pipeline": self.pipeline,
            "reason": self.reason,
            "score": round(self.score, 3),
            "alternatives": self.alternatives,
            "profile_used": self.profile_used,
        }


# =============================================================================
# MODEL PROFILE MANAGER
# =============================================================================

class ModelProfileManager:
    """Model profile manager.

    Loads profiles from a YAML file, provides task-based and
    capability-based search, and integrates with the existing
    Router for transparent model selection.

    Usage:
        manager = ModelProfileManager()
        manager.load()
        profiles = manager.find_best_for_task("code_python")
        for p in profiles:
            print(p.name, p.score_for_task("code_python"))
    """

    def __init__(self, profiles_path: Path | None = None):
        """Initialize the manager.

        Args:
            profiles_path: Path to the YAML file (default: config/model_profiles.yaml)
        """
        self._profiles_path = profiles_path or _DEFAULT_PROFILES_PATH
        self._profiles: dict[str, ModelProfile] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        """Whether profiles have been loaded."""
        return self._loaded

    @property
    def count(self) -> int:
        """Number of loaded profiles."""
        return len(self._profiles)

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load(self, force_reload: bool = False) -> int:
        """Load profiles from the YAML file.

        Args:
            force_reload: Reload even if already loaded

        Returns:
            Number of profiles loaded
        """
        if self._loaded and not force_reload:
            return len(self._profiles)

        self._profiles.clear()

        if not self._profiles_path.exists():
            logger.warning(f"Profiles file not found: {self._profiles_path}")
            self._loaded = True
            return 0

        try:
            with open(self._profiles_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"YAML error in {self._profiles_path}: {e}")
            self._loaded = True
            return 0
        except Exception as e:
            logger.error(f"Read error {self._profiles_path}: {e}")
            self._loaded = True
            return 0

        profiles_data = data.get("profiles", {})
        if not isinstance(profiles_data, dict):
            logger.warning("Invalid 'profiles' section in YAML")
            self._loaded = True
            return 0

        for model_name, profile_data in profiles_data.items():
            if not isinstance(profile_data, dict):
                logger.warning(f"Invalid profile for {model_name}, skipping")
                continue

            try:
                profile = ModelProfile(
                    name=model_name,
                    display_name=profile_data.get("display_name", model_name),
                    capabilities=profile_data.get("capabilities", []),
                    strengths=profile_data.get("strengths", []),
                    weaknesses=profile_data.get("weaknesses", []),
                    context_window=profile_data.get("context_window", 32768),
                    speed_tier=profile_data.get("speed_tier", "medium"),
                    quality_tier=profile_data.get("quality_tier", "medium"),
                    recommended_for=profile_data.get("recommended_for", []),
                    not_recommended_for=profile_data.get("not_recommended_for", []),
                    task_scores=profile_data.get("task_scores", {}),
                    parameter_count=profile_data.get("parameter_count"),
                    quantization=profile_data.get("quantization"),
                    family=profile_data.get("family"),
                )
                self._profiles[model_name] = profile
            except Exception as e:
                logger.error(f"Error creating profile {model_name}: {e}")

        self._loaded = True
        logger.info(f"{len(self._profiles)} model profiles loaded from {self._profiles_path}")
        return len(self._profiles)

    def _ensure_loaded(self):
        """Load profiles if not already done."""
        if not self._loaded:
            self.load()

    # -------------------------------------------------------------------------
    # Profile access
    # -------------------------------------------------------------------------

    def get_profile(self, model_name: str) -> ModelProfile | None:
        """Return the profile for a model.

        Args:
            model_name: Ollama model name (e.g. "qwen3:32b")

        Returns:
            The profile or None if not found
        """
        self._ensure_loaded()
        return self._profiles.get(model_name)

    def list_profiles(self) -> list[ModelProfile]:
        """Return all loaded profiles.

        Returns:
            List of profiles
        """
        self._ensure_loaded()
        return list(self._profiles.values())

    def list_profile_names(self) -> list[str]:
        """Return names of all models with a profile.

        Returns:
            List of model names
        """
        self._ensure_loaded()
        return list(self._profiles.keys())

    # -------------------------------------------------------------------------
    # Task-based search
    # -------------------------------------------------------------------------

    def find_best_for_task(
        self,
        task_type: str,
        requirements: list[str] | None = None,
        speed_tier: str | None = None,
        quality_tier: str | None = None,
        limit: int = 5,
    ) -> list[ModelProfile]:
        """Find the best models for a task type.

        Sorts profiles by descending relevance score,
        with optional speed/quality tier filters.

        Args:
            task_type: Task type (e.g. "code_python", "debug", "explanation")
            requirements: Required capabilities (e.g. ["code", "reasoning"])
            speed_tier: Filter by speed tier (optional)
            quality_tier: Filter by quality tier (optional)
            limit: Max number of results

        Returns:
            List of profiles sorted by relevance
        """
        self._ensure_loaded()

        candidates = []
        for profile in self._profiles.values():
            # Tier filters
            if speed_tier and profile.speed_tier != speed_tier:
                continue
            if quality_tier and profile.quality_tier != quality_tier:
                continue

            score = profile.score_for_task(task_type, requirements)
            if score > 0:
                candidates.append((score, profile))

        # Sort by descending score, then by quality as tiebreaker
        quality_order = {"high": 3, "medium": 2, "low": 1}
        candidates.sort(
            key=lambda x: (x[0], quality_order.get(x[1].quality_tier, 0)),
            reverse=True,
        )

        return [profile for _, profile in candidates[:limit]]

    def find_by_capability(self, capability: str) -> list[ModelProfile]:
        """Find models having a given capability.

        Args:
            capability: Capability to search for (e.g. "vision", "code")

        Returns:
            List of profiles with this capability
        """
        self._ensure_loaded()
        return [p for p in self._profiles.values() if p.has_capability(capability)]

    # -------------------------------------------------------------------------
    # Routing helpers
    # -------------------------------------------------------------------------

    def build_routing_reason(
        self,
        selected_model: str,
        task_type: str,
        pipeline: str = "",
        alternatives: list[str] | None = None,
    ) -> RoutingReason:
        """Build the routing explanation for the frontend.

        Args:
            selected_model: Selected model
            task_type: Detected task type
            pipeline: Agentic pipeline used
            alternatives: Alternative models considered

        Returns:
            RoutingReason with transparent explanation
        """
        self._ensure_loaded()
        profile = self._profiles.get(selected_model)

        if profile:
            # Build reason based on the profile
            if task_type in profile.recommended_for:
                reason = f"Recommended for {task_type}"
            elif any(task_type.startswith(r) for r in profile.recommended_for):
                matching = [r for r in profile.recommended_for if task_type.startswith(r)]
                reason = f"Matches profile category: {matching[0]}"
            else:
                reason = f"Best available ({profile.quality_tier} quality)"

            return RoutingReason(
                model=selected_model,
                display_name=profile.display_name,
                task_type=task_type,
                pipeline=pipeline,
                reason=reason,
                score=profile.score_for_task(task_type),
                alternatives=alternatives or [],
                profile_used=True,
            )
        else:
            # No profile: generic reason
            return RoutingReason(
                model=selected_model,
                display_name=selected_model,
                task_type=task_type,
                pipeline=pipeline,
                reason="Config-based routing (no profile)",
                score=0.0,
                alternatives=alternatives or [],
                profile_used=False,
            )

    # -------------------------------------------------------------------------
    # S54: CRUD operations
    # -------------------------------------------------------------------------

    def add_profile(self, profile: "ModelProfile") -> bool:
        """Add or update a model profile.

        Args:
            profile: ModelProfile instance to add

        Returns:
            True if added successfully
        """
        self._ensure_loaded()
        self._profiles[profile.name] = profile
        logger.info(f"Profile added/updated: {profile.name}")
        return True

    def remove_profile(self, model_name: str) -> bool:
        """Remove a model profile.

        Args:
            model_name: Name of the model to remove

        Returns:
            True if removed, False if not found
        """
        self._ensure_loaded()
        if model_name in self._profiles:
            del self._profiles[model_name]
            logger.info(f"Profile removed: {model_name}")
            return True
        return False

    def update_task_scores(self, model_name: str, task_scores: dict[str, float]) -> bool:
        """Update task scores for a specific model.

        Args:
            model_name: Name of the model
            task_scores: Dict of task_type -> score (0.0-1.0)

        Returns:
            True if updated, False if model not found
        """
        self._ensure_loaded()
        profile = self._profiles.get(model_name)
        if profile is None:
            return False
        for k, v in task_scores.items():
            profile.task_scores[k] = max(0.0, min(1.0, float(v)))
        return True

    def save(self, path: Path | None = None) -> bool:
        """Save profiles back to YAML file.

        Args:
            path: Output path (default: original profiles path)

        Returns:
            True if saved successfully
        """
        target = path or self._profiles_path
        try:
            data = {"profiles": {}}
            for name, profile in self._profiles.items():
                pdata = {
                    "display_name": profile.display_name,
                    "capabilities": profile.capabilities,
                    "strengths": profile.strengths,
                    "weaknesses": profile.weaknesses,
                    "context_window": profile.context_window,
                    "speed_tier": profile.speed_tier,
                    "quality_tier": profile.quality_tier,
                    "recommended_for": profile.recommended_for,
                    "not_recommended_for": profile.not_recommended_for,
                }
                if profile.task_scores:
                    pdata["task_scores"] = dict(profile.task_scores)
                if profile.parameter_count:
                    pdata["parameter_count"] = profile.parameter_count
                if profile.quantization:
                    pdata["quantization"] = profile.quantization
                if profile.family:
                    pdata["family"] = profile.family
                data["profiles"][name] = pdata
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            logger.info(f"Profiles saved to {target}")
            return True
        except Exception as e:
            logger.error(f"Error saving profiles to {target}: {e}")
            return False

    # -------------------------------------------------------------------------
    # S54: Auto-detection via ollama.show()
    # -------------------------------------------------------------------------

    def auto_detect(self, model_name: str) -> Optional["ModelProfile"]:
        """Auto-detect model capabilities via ollama.show().

        Queries Ollama for model metadata and creates/updates
        a profile with detected context window, parameter count,
        quantization, and family.

        Args:
            model_name: Ollama model name (e.g. "qwen3:32b")

        Returns:
            Updated or new ModelProfile, or None on failure
        """
        try:
            import ollama
            info = ollama.show(model_name)
        except ImportError:
            logger.debug("Ollama not available for auto-detection")
            return None
        except Exception as e:
            logger.warning(f"Auto-detection failed for {model_name}: {e}")
            return None

        model_info = info if isinstance(info, dict) else {}
        if hasattr(info, "modelinfo"):
            model_info = info.modelinfo if isinstance(info.modelinfo, dict) else {}
        elif hasattr(info, "model_info"):
            model_info = info.model_info if isinstance(info.model_info, dict) else {}

        details = {}
        if hasattr(info, "details"):
            details = info.details if isinstance(info.details, dict) else {}
        elif isinstance(info, dict):
            details = info.get("details", {})

        # Context window detection
        context_window = 32768
        for key in model_info:
            if "context" in key.lower() and "length" in key.lower():
                try:
                    context_window = int(model_info[key])
                except (ValueError, TypeError):
                    pass

        param_count = details.get("parameter_size") if isinstance(details, dict) else None
        if param_count is None and hasattr(details, "parameter_size"):
            param_count = getattr(details, "parameter_size", None)

        quant = details.get("quantization_level") if isinstance(details, dict) else None
        if quant is None and hasattr(details, "quantization_level"):
            quant = getattr(details, "quantization_level", None)

        family_val = details.get("family") if isinstance(details, dict) else None
        if family_val is None and hasattr(details, "family"):
            family_val = getattr(details, "family", None)

        self._ensure_loaded()
        existing = self._profiles.get(model_name)
        if existing:
            existing.context_window = context_window
            existing.parameter_count = str(param_count) if param_count else existing.parameter_count
            existing.quantization = str(quant) if quant else existing.quantization
            existing.family = str(family_val) if family_val else existing.family
            existing.auto_detected = True
            logger.info(f"Auto-detected metadata for {model_name}: ctx={context_window}")
            return existing
        else:
            profile = ModelProfile(
                name=model_name,
                display_name=model_name,
                context_window=context_window,
                parameter_count=str(param_count) if param_count else None,
                quantization=str(quant) if quant else None,
                family=str(family_val) if family_val else None,
                auto_detected=True,
                capabilities=["general"],
                recommended_for=["general"],
            )
            self._profiles[model_name] = profile
            logger.info(f"Auto-created profile for {model_name}: ctx={context_window}")
            return profile

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export all profiles as a dictionary.

        Returns:
            Dictionary with all profiles
        """
        self._ensure_loaded()
        return {
            "profiles": {name: p.to_dict() for name, p in self._profiles.items()},
            "count": len(self._profiles),
        }


# =============================================================================
# SINGLETON
# =============================================================================

profile_manager = ModelProfileManager()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_profile(model_name: str) -> ModelProfile | None:
    """Shortcut to get a model profile."""
    return profile_manager.get_profile(model_name)


def find_best_for_task(task_type: str, **kwargs) -> list[ModelProfile]:
    """Shortcut to find the best models for a task."""
    return profile_manager.find_best_for_task(task_type, **kwargs)
