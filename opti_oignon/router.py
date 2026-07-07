#!/usr/bin/env python3
"""
ROUTER - CONTEXTEUR 2.0 (with Smart Routing)
=============================================

Selects the optimal model based on task analysis.

This module bridges Opti-Oignon's Analyzer and the SmartRouter
from the routing/ module for advanced benchmark-based routing.

The Router takes Analyzer results and determines:
- Which model to use (via SmartRouter)
- Which temperature to apply
- Which system prompt to load
- Additional parameters

Author: Léon
"""

import logging
import re
from dataclasses import dataclass, field

import ollama

from .analyzer import AnalysisResult, TaskType
from .config import config

# S46: Conditional import of model profiles
try:
    from .model_profiles import ModelProfile, RoutingReason, profile_manager  # noqa: F401
    MODEL_PROFILES_AVAILABLE = True
except ImportError:
    MODEL_PROFILES_AVAILABLE = False
    profile_manager = None

logger = logging.getLogger(__name__)


class ToolCapableModelUnavailable(RuntimeError):
    """Raised when a tool-calling-capable model is required but none exists.

    A tool-bound turn must never silently run on a model whose profile
    carries an explicit negative tool-calling verdict. When the profile
    filter matches no tool-capable model and the config fallback is
    explicitly non-capable too, selection fails secure with this error
    instead of returning a model that cannot call tools.
    """


def _tool_calling_verdict_or_open(model: str) -> bool:
    """Tool-calling verdict for the fail-secure guard, via the manifest.

    Delegates to the capability manifest's public predicate -- the single
    source of truth -- so the router never reimplements the capability
    rule. A model with no profile stays capable there (the historical
    fallback), so only a model with an explicit negative verdict reads as
    False. If the predicate cannot be imported the capability subsystem is
    absent; the guard then declines to refuse (returns True) rather than
    blocking every tool-bound turn on an infrastructure gap.
    """
    try:
        from .capability_manifest import model_tool_capable
        return bool(model_tool_capable(model))
    except Exception:
        return True


# =============================================================================
# ROUTING RESULT
# =============================================================================

@dataclass
class RoutingResult:
    """Model routing result."""
    model: str                    # Selected Ollama model
    temperature: float            # Temperature to use
    task_type: str               # Task type
    prompt_variant: str          # Prompt variant (standard, reasoning, fast)
    model_type: str              # Model type (code, reasoning, general)
    priority_used: str           # Which priority was used (primary, fast, fallback)
    explanation: str             # Why this model was chosen
    timeout: int                 # Timeout in seconds
    # S46: Transparent routing data
    routing_reason: dict | None = None  # Detailed reason for the frontend
    # S48: Vision routing
    vision_routed: bool = False            # True if auto-routed to vision
    images: list[str] = field(default_factory=list)  # Base64 images for Ollama

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "model": self.model,
            "temperature": self.temperature,
            "task_type": self.task_type,
            "prompt_variant": self.prompt_variant,
            "model_type": self.model_type,
            "priority_used": self.priority_used,
            "explanation": self.explanation,
            "timeout": self.timeout,
            "vision_routed": self.vision_routed,
        }
        if self.routing_reason:
            result["routing_reason"] = self.routing_reason
        return result


# =============================================================================
# MAIN CLASS
# =============================================================================

class ModelRouter:
    """
    Intelligent model router.

    Uses SmartRouter from routing/ module if available,
    otherwise uses basic logic with config.py.

    Usage:
        router = ModelRouter()
        result = router.route(analysis_result)
        print(result.model)  # qwen3-coder:30b
    """

    def __init__(self):
        """Initialize the router."""
        self._available_models: list[str] = []
        self._last_check: float = 0
        self._cache_duration: float = 60.0
        self._config = config

    # -------------------------------------------------------------------------
    # Available Models Check
    # -------------------------------------------------------------------------

    def get_available_models(self, force_refresh: bool = False) -> list[str]:
        """
        Get the list of available Ollama models.

        Args:
            force_refresh: Force cache refresh

        Returns:
            List of model names
        """
        import time

        if not force_refresh and self._available_models:
            if time.time() - self._last_check < self._cache_duration:
                return self._available_models

        try:
            response = ollama.list()
            models = []
            if hasattr(response, 'models'):
                for m in response.models:
                    name = getattr(m, 'model', None) or getattr(m, 'name', None)
                    if name:
                        models.append(name)
            elif isinstance(response, dict):
                for m in response.get("models", []):
                    name = m.get("model") or m.get("name", "")
                    if name:
                        models.append(name)

            self._available_models = models
            self._last_check = time.time()
            logger.debug(f"Models detected: {models}")
            return models

        except Exception as e:
            logger.error(f"Ollama model listing error: {e}")
            return self._available_models

    def is_model_available(self, model: str) -> bool:
        """Check if a model is available."""
        return model in self.get_available_models()

    def find_best_available(self, preferred: str, alternatives: list[str]) -> tuple[str, str]:
        """
        Find the best available model among options.

        Args:
            preferred: Preferred model
            alternatives: Alternative list in preference order

        Returns:
            (selected_model, reason)
        """
        available = self.get_available_models()

        if preferred in available:
            return preferred, "primary"

        for alt in alternatives:
            if alt in available:
                logger.info(f"Model {preferred} not available, using {alt}")
                return alt, "fallback"

        if available:
            logger.warning(f"No preferred model available, using {available[0]}")
            return available[0], "emergency"

        logger.error("No Ollama models available!")
        return preferred, "unavailable"

    # -------------------------------------------------------------------------
    # S48: Vision Detection
    # -------------------------------------------------------------------------

    # Regex to detect base64 image data in a message
    _BASE64_IMAGE_PATTERN = re.compile(
        r"data:image/(png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=]{50,}",
    )
    # Extensions d'image connues
    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}

    def detect_images_in_message(
        self,
        message: str,
        images: list[str] | None = None,
    ) -> bool:
        """Detect if the message contains image content.

        Checks several signals:
        - Liste d'images explicites (base64) passees via ``images``
        - Inline base64 data in the message text
        - References to image files (path with image extension)

        Args:
            message: User message text
            images: Liste optionnelle d'images base64

        Returns:
            True if image content is detected
        """
        # Explicit signal: images provided in the request
        if images and len(images) > 0:
            return True

        # Inline base64 data in the message
        if self._BASE64_IMAGE_PATTERN.search(message):
            return True

        # Image-file references in the text
        # Ex: "Analyse image.png", "/path/to/photo.jpg"
        lower_msg = message.lower()
        for ext in self._IMAGE_EXTENSIONS:
            if ext in lower_msg:
                return True

        return False

    def find_best_vision_model(
        self,
        priority: str = "balanced",
    ) -> tuple[str, str, list[str]] | None:
        """Find the best available vision-capable model.

        Uses model profiles to find models
        with the 'vision' capability, filtered by Ollama availability.

        Args:
            priority: Selection priority (fast, balanced, quality)

        Returns:
            Tuple (model_name, reason, alternatives) or None if none available
        """
        if not MODEL_PROFILES_AVAILABLE or profile_manager is None:
            return None

        profile_manager._ensure_loaded()
        if profile_manager.count == 0:
            return None

        # Look for vision models
        speed_filter = "fast" if priority == "fast" else None
        quality_filter = "high" if priority == "quality" else None

        vision_profiles = profile_manager.find_best_for_task(
            "vision",
            speed_tier=speed_filter,
            quality_tier=quality_filter,
            limit=5,
        )

        # Relax the filters if no result
        if not vision_profiles and (speed_filter or quality_filter):
            vision_profiles = profile_manager.find_best_for_task(
                "vision",
                limit=5,
            )

        if not vision_profiles:
            return None

        # S48: Strictly filter models with vision capability
        vision_only = [p for p in vision_profiles if p.has_capability("vision")]
        if not vision_only:
            return None

        available = self.get_available_models()
        alternatives = [p.name for p in vision_only]  # noqa: F841

        for profile in vision_only:
            if profile.name in available:
                alt_names = [p.name for p in vision_only if p.name != profile.name]
                return profile.name, "vision_auto", alt_names

        return None

    # -------------------------------------------------------------------------
    # Routing Logic
    # -------------------------------------------------------------------------

    def route(
        self,
        analysis: AnalysisResult,
        priority: str = "balanced",
        force_model: str | None = None,
        force_variant: str | None = None,
        images: list[str] | None = None,
        message: str | None = None,
        require_tool_calling: bool = False,
    ) -> RoutingResult:
        """
        Route to optimal model based on analysis.

        Uses model profiles when available for smarter selection,
        falls back to config-based routing otherwise.

        S48: When images are detected in the message or provided
        explicitly, automatically routes to the best vision-capable model.

        Args:
            analysis: Task analysis result
            priority: "fast" (speed), "balanced" (default), "quality" (max quality)
            force_model: Force a specific model (ignores auto-selection)
            force_variant: Force a prompt variant
            images: Optional list of base64-encoded images (S48)
            message: Optional raw message text for vision detection (S48)

        Returns:
            RoutingResult with complete configuration
        """
        model_type = analysis.suggested_model_type
        task_type = analysis.task_type.value

        prompt_variant = self._determine_prompt_variant(analysis, force_variant)

        alternatives = []
        profile_used = False
        vision_routed = False
        image_list = images or []

        # S48: Image detection and vision routing
        has_images = self.detect_images_in_message(
            message or "", images=images,
        )

        if force_model:
            model, priority_used = self._validate_forced_model(force_model)
        elif has_images:
            # Auto-route to a vision model
            vision_result = self.find_best_vision_model(priority)
            if vision_result:
                model, priority_used, alternatives = vision_result
                profile_used = True
                vision_routed = True
                task_type = "vision"
                logger.info(
                    f"Vision auto-routing: images detected, "
                    f"routing to {model}"
                )
            else:
                # No vision model available, normal fallback
                logger.warning(
                    "Images detected but no vision model available, "
                    "falling back to standard routing"
                )
                model, priority_used, alternatives, profile_used = (
                    self._select_model_with_profiles(
                        model_type, task_type, priority,
                        require_tool_calling=require_tool_calling,
                    )
                )
        else:
            # S46: Try profile-based routing first
            model, priority_used, alternatives, profile_used = self._select_model_with_profiles(
                model_type, task_type, priority,
                require_tool_calling=require_tool_calling,
            )

        temperature = self._determine_temperature(task_type, analysis.complexity.value)
        timeout = self._determine_timeout(priority, analysis.complexity.value)

        explanation = self._build_explanation(
            analysis, model, model_type, priority, priority_used
        )

        # S46: Build the routing reason for the frontend
        routing_reason = None
        if MODEL_PROFILES_AVAILABLE and profile_manager is not None:
            reason = profile_manager.build_routing_reason(
                selected_model=model,
                task_type=task_type,
                alternatives=alternatives[:3],
            )
            routing_reason = reason.to_dict()

        return RoutingResult(
            model=model,
            temperature=temperature,
            task_type=task_type,
            prompt_variant=prompt_variant,
            model_type=model_type,
            priority_used=priority_used,
            explanation=explanation,
            timeout=timeout,
            routing_reason=routing_reason,
            vision_routed=vision_routed,
            images=image_list,
        )

    def _validate_forced_model(self, model: str) -> tuple[str, str]:
        """Validate forced model and return alternative if unavailable."""
        if self.is_model_available(model):
            return model, "forced"

        logger.warning(f"Forced model {model} not available")
        fallbacks = self._config.get_fallback_models()
        return self.find_best_available(model, fallbacks)

    def _select_model(self, model_type: str, priority: str) -> tuple[str, str]:
        """
        Select model based on type and priority (config-based).

        Args:
            model_type: Model type (code, reasoning, general, quick)
            priority: Priority (fast, balanced, quality)

        Returns:
            (model, reason)
        """
        # Map priority to config model type
        priority_map = {
            "fast": "fast",
            "balanced": "primary",
            "quality": "quality",
        }
        config_priority = priority_map.get(priority, "primary")

        # Get preferred model
        preferred = self._config.get_model(model_type, config_priority)

        # Get alternatives
        alternatives = []
        for alt_priority in ["primary", "fast", "quality"]:
            if alt_priority != config_priority:
                alt_model = self._config.get_model(model_type, alt_priority)
                if alt_model and alt_model not in alternatives:
                    alternatives.append(alt_model)

        # Add global fallbacks
        alternatives.extend(self._config.get_fallback_models())

        return self.find_best_available(preferred, alternatives)

    def _select_model_with_profiles(
        self,
        model_type: str,
        task_type: str,
        priority: str,
        require_tool_calling: bool = False,
    ) -> tuple[str, str, list[str], bool]:
        """
        Select model using profiles when available, fallback to config.

        S46: Enrich the selection with model profiles.
        First looks for recommended models for the task_type,
        filtered by tier if priority requires, then verifies
        availability via Ollama.

        Args:
            model_type: Model type from analyzer
            task_type: Specific task type (ex: "code_python")
            priority: Priority (fast, balanced, quality)
            require_tool_calling: When True, profiled models whose
                tool-calling verdict is negative are skipped, so the pick
                is a tool-capable model. Off by default, so ordinary
                routing is unchanged. Fail-secure: when no profiled model
                is both available and tool-capable and the config fallback
                is itself explicitly non-capable, this raises
                ToolCapableModelUnavailable rather than returning a model
                that cannot call tools. A model with no profile stays
                capable (the historical fallback), so a refusal happens
                only when the selected model is known to be incapable; if
                the capability predicate cannot be imported at all the
                guard declines to refuse and the config fallback stands.

        Returns:
            (model, reason, alternatives, profile_used)
        """
        # Try the profiles
        if MODEL_PROFILES_AVAILABLE and profile_manager is not None:
            profile_manager._ensure_loaded()
            if profile_manager.count > 0:
                # Determine the speed tier from the priority
                speed_filter = None
                quality_filter = None
                if priority == "fast":
                    speed_filter = "fast"
                elif priority == "quality":
                    quality_filter = "high"

                # Find the best models for the task
                best_profiles = profile_manager.find_best_for_task(
                    task_type,
                    speed_tier=speed_filter,
                    quality_tier=quality_filter,
                    limit=5,
                )

                # If no result with strict filters, relax them
                if not best_profiles and (speed_filter or quality_filter):
                    best_profiles = profile_manager.find_best_for_task(
                        task_type,
                        limit=5,
                    )

                if best_profiles:
                    available = self.get_available_models()
                    alternatives = [p.name for p in best_profiles]  # noqa: F841

                    # Optionally require a tool-calling-capable model.
                    # The verdict is the capability manifest's own; a
                    # defensive lazy import keeps the router decoupled and
                    # fails open (no filtering) if the predicate is absent.
                    tool_verdict = None
                    if require_tool_calling:
                        try:
                            from .capability_manifest import model_tool_capable
                            tool_verdict = model_tool_capable
                        except Exception:
                            tool_verdict = None

                    for profile in best_profiles:
                        if profile.name not in available:
                            continue
                        if (
                            require_tool_calling
                            and tool_verdict is not None
                            and not tool_verdict(profile.name)
                        ):
                            continue
                        alt_names = [p.name for p in best_profiles if p.name != profile.name]
                        return profile.name, "profile", alt_names, True

                    # No profiled model available (or none tool-capable when
                    # required), config fallback.
                    logger.debug(
                        f"No profiled model available for {task_type}, config fallback"
                    )

        # Fallback: selection config classique
        model, reason = self._select_model(model_type, priority)
        # Fail-secure for a tool-bound selection: never hand back a model
        # that is known to be unable to call tools. A model with no profile
        # stays capable (the historical fallback), so this refuses only when
        # the fallback carries an explicit negative verdict; an unavailable
        # predicate declines to refuse and the fallback stands.
        if require_tool_calling and not _tool_calling_verdict_or_open(model):
            raise ToolCapableModelUnavailable(
                "No tool-calling-capable model is available for task "
                f"'{task_type}': the profile filter matched none and the "
                f"config fallback '{model}' is explicitly not tool-capable."
            )
        return model, reason, [], False

    def select_tool_capable_model(
        self,
        *,
        model_type: str,
        task_type: str,
        priority: str = "balanced",
    ) -> str:
        """Select a tool-calling-capable model for a tool-bound turn.

        Runs the profile filter with the tool-calling requirement engaged,
        so a model whose profile carries an explicit negative tool-calling
        verdict is excluded from the selection. This is the real entry
        point that turns the requirement on; ordinary routing keeps its
        default (off). Fails secure: raises ToolCapableModelUnavailable
        when no tool-calling-capable model can be found, instead of handing
        back a model that cannot call tools.

        Args:
            model_type: Model type from the analyzer.
            task_type: Specific task type (e.g. "code_python").
            priority: Selection priority (fast, balanced, quality).

        Returns:
            The name of a tool-calling-capable model.

        Raises:
            ToolCapableModelUnavailable: When no tool-capable model exists.
        """
        model, _reason, _alternatives, _profile_used = (
            self._select_model_with_profiles(
                model_type, task_type, priority, require_tool_calling=True,
            )
        )
        return model

    def _determine_prompt_variant(
        self,
        analysis: AnalysisResult,
        force_variant: str | None
    ) -> str:
        """Determine which prompt variant to use."""
        if force_variant:
            return force_variant

        if analysis.complexity.value == "complex":
            return "reasoning"

        if analysis.task_type == TaskType.PLANNING_DEEP:
            return "reasoning"

        if analysis.task_type == TaskType.SIMPLE_QUESTION:
            return "fast"

        if analysis.complexity.value == "simple":
            return "fast"

        return "standard"

    def _determine_temperature(self, task_type: str, complexity: str) -> float:
        """Determine optimal temperature."""
        base_temp = self._config.get_temperature(task_type.split("_")[0])

        if complexity == "complex":
            return min(base_temp + 0.1, 0.9)
        elif complexity == "simple":
            return max(base_temp - 0.1, 0.1)

        return base_temp

    def _determine_timeout(self, priority: str, complexity: str) -> int:
        """Determine appropriate timeout."""
        if priority == "fast":
            return self._config.get_timeout("fast")
        elif complexity == "complex":
            return self._config.get_timeout("deep")
        else:
            return self._config.get_timeout("default")

    def _build_explanation(
        self,
        analysis: AnalysisResult,
        model: str,
        model_type: str,
        priority: str,
        priority_used: str,
    ) -> str:
        """Build readable routing explanation."""
        parts = [f"Task: {analysis.task_type.value}"]

        if analysis.confidence < 0.5:
            parts.append(f"(low confidence: {analysis.confidence:.0%})")

        parts.append(f"-> {model}")

        if priority_used == "profile":
            parts.append("(profile-matched)")
        elif priority_used == "vision_auto":
            parts.append("(vision auto-routed)")
        elif priority_used == "primary":
            parts.append(f"(optimal for {model_type})")
        elif priority_used == "fallback":
            parts.append("(fallback - preferred model unavailable)")
        elif priority_used == "forced":
            parts.append("(forced by user)")

        if priority != "balanced":
            parts.append(f"[priority: {priority}]")

        return " ".join(parts)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

router = ModelRouter()


def route(
    analysis: AnalysisResult,
    priority: str = "balanced",
    force_model: str | None = None,
    images: list[str] | None = None,
    message: str | None = None,
) -> RoutingResult:
    """Convenience function to route a task."""
    return router.route(
        analysis, priority, force_model,
        images=images, message=message,
    )


# =============================================================================
# TEST CLI
# =============================================================================

if __name__ == "__main__":
    from .analyzer import analyze

    print("=== Router Test ===\n")

    test_questions = [
        "How to calculate Shannon in R?",
        "Debug my Python code",
        "Write an abstract about biodiversity",
    ]

    for q in test_questions:
        print(f"Question: {q}")
        analysis = analyze(q)
        routing = router.route(analysis)

        print(f"  -> Model: {routing.model}")
        print(f"  -> Task: {routing.task_type}")
        print(f"  -> Variant: {routing.prompt_variant}")
        print(f"  -> Temperature: {routing.temperature}")
        print(f"  -> Explanation: {routing.explanation}")
        print()
