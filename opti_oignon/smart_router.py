#!/usr/bin/env python3
"""
SMART ROUTER -- Model Selection per Pipeline Step (S54)
========================================================

Selects the optimal model for each task/pipeline step type
based on model capability profiles. Integrates with the
PipelineRunner for automatic per-step model selection.

Scoring formula:
    final_score = task_score * speed_weight * context_fit

Configuration is loaded from config/smart_routing.yaml and
can be overridden via the API or configure() method.

Author: Leon
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Conditional import of model profiles
try:
    from .model_profiles import (
        ModelProfile,
        ModelProfileManager,
        RoutingReason,
    )
    from .model_profiles import (
        profile_manager as _default_profile_manager,
    )
    PROFILES_AVAILABLE = True
except ImportError:
    PROFILES_AVAILABLE = False
    ModelProfileManager = None
    ModelProfile = None
    RoutingReason = None
    _default_profile_manager = None

# Conditional import of adaptive routing (S62)
try:
    from .adaptive_routing import (
        FeedbackRoutingAdapter,
    )
    from .adaptive_routing import (
        feedback_routing_adapter as _default_feedback_adapter,
    )
    ADAPTIVE_ROUTING_AVAILABLE = True
except ImportError:
    ADAPTIVE_ROUTING_AVAILABLE = False
    FeedbackRoutingAdapter = None
    _default_feedback_adapter = None

# Conditional import of model health monitor (S63)
try:
    from .model_health import (
        ModelHealthMonitor,
        ModelStatus,
    )
    from .model_health import (
        model_health_monitor as _default_health_monitor,
    )
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False
    ModelHealthMonitor = None
    ModelStatus = None
    _default_health_monitor = None

# Conditional import of learned router (S67)
try:
    from .learned_router import (
        LearnedRouter,
        RoutingPrediction,
    )
    from .learned_router import (
        learned_router as _default_learned_router,
        LEARNED_ROUTER_AVAILABLE as _LR_AVAIL,
    )
    LEARNED_ROUTER_IN_SMART = _LR_AVAIL
except ImportError:
    LEARNED_ROUTER_IN_SMART = False
    LearnedRouter = None
    RoutingPrediction = None
    _default_learned_router = None


# Sentinel for distinguishing "not provided" from explicit None
_UNSET = object()


# =============================================================================
# CONSTANTS
# =============================================================================

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "smart_routing.yaml"

DEFAULT_SPEED_WEIGHTS = {
    "fast": 1.2,
    "medium": 1.0,
    "slow": 0.8,
}

DEFAULT_CONTEXT_REQUIREMENTS = {
    "direct": 4096,
    "tools": 8192,
    "code_verify": 8192,
    "think": 16384,
    "web_search": 8192,
    "think_tools": 16384,
    "reasoning": 32768,
    "consensus": 8192,
    "self_correct": 16384,
}

PIPELINE_TO_TASK_MAPPING = {
    "direct": ["general", "quick_answer", "simple_question"],
    "tools": ["tool_use", "general"],
    "code_verify": ["code_python", "code_r", "debug", "refactor"],
    "think": ["planning_deep", "complex_analysis", "reasoning"],
    "web_search": ["general", "quick_answer"],
    "think_tools": ["tool_use", "planning_deep", "complex_analysis"],
    "reasoning": ["reasoning", "mathematical", "planning_deep", "complex_analysis"],
    "consensus": ["general", "analysis"],
    "self_correct": ["code_python", "debug", "analysis"],
}

# S171: RAM pre-flight tuning. Rough resident-memory estimate per billion
# parameters for a typical quantized (q4/q5) GGUF weight set plus KV cache
# headroom. Deliberately conservative -- the goal is to avoid selecting a model
# the host plainly cannot hold, not to model VRAM precisely. A safety margin is
# subtracted from available RAM so selection leaves room for the OS and runtime.
_RAM_MB_PER_BILLION_PARAMS = 750.0
_RAM_SAFETY_MARGIN_MB = 1024.0


def _get_available_ram_mb() -> float:
    """Return available system RAM in MB, or 0.0 when it cannot be determined.

    Reads /proc/meminfo (MemAvailable) on Linux, falls back to psutil, and
    returns 0.0 if neither is usable. A 0.0 result disables the pre-flight
    (fail-open: no model is excluded when memory state is unknown).
    """
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
        except Exception:
            pass
    try:
        import psutil
        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _estimate_model_ram_mb(parameter_count: str | None) -> float:
    """Estimate resident RAM (MB) for a model from its parameter-count label.

    Parses labels such as "32B", "7b", "1.5B", "70 B". Returns 0.0 when the
    label is missing or unparseable, which disables the check for that model
    (fail-open: an unknown size is never treated as too large).
    """
    if not parameter_count:
        return 0.0
    text = str(parameter_count).strip().lower().replace(" ", "")
    multiplier = 1.0
    if text.endswith("b"):
        text = text[:-1]
    elif text.endswith("m"):
        text = text[:-1]
        multiplier = 0.001  # millions -> billions
    try:
        billions = float(text) * multiplier
    except ValueError:
        return 0.0
    if billions <= 0.0:
        return 0.0
    return billions * _RAM_MB_PER_BILLION_PARAMS


# =============================================================================
# SMART ROUTING RESULT
# =============================================================================

@dataclass
class SmartRoutingResult:
    """Result of smart model selection for a pipeline step."""
    model: str
    score: float = 0.0
    task_score: float = 0.0
    speed_weight: float = 1.0
    context_fit: float = 1.0
    reason: str = ""
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    profile_used: bool = False
    fallback: bool = False
    feedback_adjusted: bool = False  # S62: Whether feedback adjustments were applied
    failover: bool = False  # S63: Whether model substitution occurred due to health
    original_model: str = ""  # S63: Original model before failover
    routing_source: str = "yaml"  # S67: 'learned' or 'yaml'
    learned_confidence: float = 0.0  # S67: ML confidence score (0 when yaml used)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "model": self.model,
            "score": round(self.score, 4),
            "task_score": round(self.task_score, 4),
            "speed_weight": round(self.speed_weight, 2),
            "context_fit": round(self.context_fit, 4),
            "reason": self.reason,
            "alternatives": self.alternatives,
            "profile_used": self.profile_used,
            "fallback": self.fallback,
            "feedback_adjusted": self.feedback_adjusted,
            "failover": self.failover,
            "original_model": self.original_model,
            "routing_source": self.routing_source,
            "learned_confidence": round(self.learned_confidence, 4),
        }


# =============================================================================
# SMART ROUTER
# =============================================================================

class SmartRouter:
    """Selects the optimal model for each pipeline step type.

    Uses model capability profiles to compute a composite score
    combining task fitness, speed preference, and context window
    compatibility.

    Usage:
        router = SmartRouter()
        result = router.select_model("code_verify")
        print(result.model)  # qwen3-coder:30b
    """

    def __init__(
        self,
        profile_manager=None,
        enabled: bool = True,
        default_model: str = "qwen3:32b",
        speed_weights: dict[str, float] | None = None,
        context_requirements: dict[str, int] | None = None,
        speed_preference: str = "balanced",
        config_path: Path | None = None,
        feedback_adapter=None,
        health_monitor=_UNSET,
    ):
        """Initialize the smart router.

        Args:
            profile_manager: ModelProfileManager instance (None = singleton)
            enabled: Whether smart routing is active
            default_model: Fallback model when no profile matches
            speed_weights: Custom speed tier multipliers
            context_requirements: Custom context requirements per step type
            speed_preference: 'fast', 'balanced', or 'quality'
            config_path: Path to YAML config (None = default)
            feedback_adapter: FeedbackRoutingAdapter instance (None = singleton, S62)
            health_monitor: ModelHealthMonitor instance (None = disable, _UNSET = singleton, S63)
        """
        self._profile_manager = profile_manager or _default_profile_manager
        self._enabled = enabled
        self._default_model = default_model
        self._speed_weights = speed_weights or dict(DEFAULT_SPEED_WEIGHTS)
        self._context_requirements = context_requirements or dict(DEFAULT_CONTEXT_REQUIREMENTS)
        self._speed_preference = speed_preference
        self._cache: dict[str, SmartRoutingResult] = {}
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        # S62: Feedback-based adaptive routing
        self._feedback_adapter = feedback_adapter or _default_feedback_adapter
        # S63: Model health monitor for failover
        self._health_monitor = _default_health_monitor if health_monitor is _UNSET else health_monitor
        # S67: Learned router for ML-based task classification
        self._learned_router = _default_learned_router
        # S171: pre-flight RAM check toggle (skip models that plainly will not
        # fit in available system RAM). Fail-open when memory state is unknown.
        self._ram_preflight = True
        # Load YAML config (overrides constructor defaults)
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file if available.

        YAML values override constructor defaults, but explicit
        constructor arguments take priority over YAML when both
        the constructor and YAML provide values. This is handled
        by the singleton creation pattern.
        """
        if not self._config_path.exists():
            logger.debug(f"Smart routing config not found: {self._config_path}")
            return

        try:
            with open(self._config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Error reading smart routing config: {e}")
            return

        config = data.get("smart_routing", {})
        if not isinstance(config, dict):
            return

        # Apply YAML values
        if "enabled" in config:
            self._enabled = bool(config["enabled"])
        if "default_model" in config:
            self._default_model = str(config["default_model"])
        if "speed_preference" in config:
            pref = str(config["speed_preference"])
            if pref in ("fast", "balanced", "quality"):
                self._speed_preference = pref
        if "speed_weights" in config and isinstance(config["speed_weights"], dict):
            self._speed_weights.update(config["speed_weights"])
        if "context_requirements" in config and isinstance(config["context_requirements"], dict):
            self._context_requirements.update(config["context_requirements"])
        if "ram_preflight" in config:
            self._ram_preflight = bool(config["ram_preflight"])

        logger.info(
            f"Smart routing config loaded: enabled={self._enabled}, "
            f"default={self._default_model}, speed={self._speed_preference}"
        )

    @property
    def enabled(self) -> bool:
        """Whether smart routing is active."""
        return self._enabled and PROFILES_AVAILABLE and self._profile_manager is not None

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def default_model(self) -> str:
        return self._default_model

    @default_model.setter
    def default_model(self, value: str):
        self._default_model = value

    @property
    def speed_preference(self) -> str:
        return self._speed_preference

    @speed_preference.setter
    def speed_preference(self, value: str):
        if value in ("fast", "balanced", "quality"):
            self._speed_preference = value
            self._cache.clear()

    # -------------------------------------------------------------------------
    # Core selection
    # -------------------------------------------------------------------------

    def select_model(
        self,
        step_type: str,
        required_context: int | None = None,
        excluded_models: list[str] | None = None,
        prefer_speed: bool | None = None,
    ) -> SmartRoutingResult:
        """Select the best model for a pipeline step type.

        Computes: final_score = task_score * speed_weight * context_fit

        Args:
            step_type: Pipeline step type (direct, tools, code_verify, etc.)
            required_context: Minimum context window needed (tokens)
            excluded_models: Models to exclude from selection
            prefer_speed: Override speed preference for this call

        Returns:
            SmartRoutingResult with selected model and scoring details
        """
        if not self.enabled:
            return SmartRoutingResult(
                model=self._default_model,
                reason="Smart routing disabled, using default model",
                fallback=True,
            )

        # Cache lookup
        cache_key = f"{step_type}:{required_context}:{prefer_speed}"
        if cache_key in self._cache and not excluded_models:
            return self._cache[cache_key]

        # Ensure profiles are loaded
        if not self._profile_manager.loaded:
            self._profile_manager.load()

        profiles = self._profile_manager.list_profiles()
        if not profiles:
            return SmartRoutingResult(
                model=self._default_model,
                reason="No model profiles loaded, using default model",
                fallback=True,
            )

        # Determine context requirement and speed adjustments
        ctx_required = required_context or self._context_requirements.get(step_type, 4096)
        speed_adj = self._get_speed_adjustment(prefer_speed)

        # Score all candidate models
        candidates: list[tuple[float, Any, dict[str, float]]] = []
        excluded = set(excluded_models or [])
        task_types = PIPELINE_TO_TASK_MAPPING.get(step_type, [step_type])

        # S171: pre-flight RAM budget. Computed once per selection. A value of
        # 0.0 (memory state unknown) disables the check entirely (fail-open).
        ram_budget_mb = 0.0
        if self._ram_preflight:
            available = _get_available_ram_mb()
            if available > 0.0:
                ram_budget_mb = available - _RAM_SAFETY_MARGIN_MB
        ram_excluded: list[str] = []

        for profile in profiles:
            if profile.name in excluded:
                continue
            # Skip embedding-only models
            if profile.capabilities == ["embeddings"]:
                continue

            # S171: skip models that plainly will not fit in available RAM.
            # Only applies when both the budget and the model's parameter count
            # are known; an unknown size is never treated as too large.
            if ram_budget_mb > 0.0:
                est_mb = _estimate_model_ram_mb(getattr(profile, "parameter_count", None))
                if est_mb > 0.0 and est_mb > ram_budget_mb:
                    ram_excluded.append(profile.name)
                    logger.debug(
                        "RAM pre-flight excluded %s (~%.0f MB > %.0f MB budget)",
                        profile.name, est_mb, ram_budget_mb,
                    )
                    continue

            # S63: Skip unavailable models entirely when health monitor active
            health_penalty = 1.0
            if self._health_monitor is not None and HEALTH_MONITOR_AVAILABLE:
                try:
                    if self._health_monitor.auto_failover:
                        status = self._health_monitor.get_status(profile.name)
                        if hasattr(status, 'value'):
                            if status.value == "unavailable":
                                continue
                            elif status.value == "degraded":
                                health_penalty = 0.5
                except Exception as e:
                    logger.debug("Health check failed for %s: %s", profile.name, e)

            # Best task score across mapped types
            task_score = 0.0
            for tt in task_types:
                s = self._compute_task_score(profile, tt)
                task_score = max(task_score, s)
            # Also try step_type directly
            direct_score = self._compute_task_score(profile, step_type)
            task_score = max(task_score, direct_score)

            if task_score <= 0.0:
                continue

            # Speed weight
            base_speed = self._speed_weights.get(profile.speed_tier, 1.0)
            speed_weight = base_speed * speed_adj.get(profile.speed_tier, 1.0)

            # Context fit
            if profile.context_window >= ctx_required:
                context_fit = 1.0
            elif profile.context_window > 0:
                context_fit = profile.context_window / ctx_required
            else:
                context_fit = 0.5

            # S63: Apply health penalty to context_fit
            context_fit *= health_penalty

            final_score = task_score * speed_weight * context_fit
            breakdown = {
                "task_score": task_score,
                "speed_weight": speed_weight,
                "context_fit": context_fit,
                "health_penalty": health_penalty,
            }
            candidates.append((final_score, profile, breakdown))

        if not candidates:
            if ram_excluded:
                return SmartRoutingResult(
                    model=self._default_model,
                    reason=(
                        f"No model fits available RAM for step '{step_type}' "
                        f"(excluded by pre-flight: {', '.join(ram_excluded)}), "
                        "using default"
                    ),
                    fallback=True,
                )
            return SmartRoutingResult(
                model=self._default_model,
                reason=f"No suitable model for step '{step_type}', using default",
                fallback=True,
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_profile, best_breakdown = candidates[0]

        # Alternatives (top 3)
        alternatives = []
        for score, prof, brk in candidates[1:4]:
            alternatives.append({
                "model": prof.name,
                "display_name": prof.display_name,
                "score": round(score, 4),
            })

        reason = self._build_reason(best_profile, step_type, task_types, best_breakdown)

        # S62: Check if feedback adjustments are active
        fb_adjusted = False
        if self._feedback_adapter is not None and ADAPTIVE_ROUTING_AVAILABLE:
            try:
                fb_adjusted = self._feedback_adapter.has_active_adjustments()
            except Exception:
                pass

        # S63: Detect failover (health caused model substitution)
        is_failover = False
        original_model = ""

        if self._health_monitor is not None and HEALTH_MONITOR_AVAILABLE:
            try:
                if self._health_monitor.auto_failover:
                    # Check if any unavailable model was excluded
                    unavailable = self._health_monitor.get_unavailable_models()
                    for umodel in unavailable:
                        uprofile = self._profile_manager.get_profile(umodel) if self._profile_manager else None
                        if uprofile is not None and umodel not in excluded:
                            is_failover = True
                            if not original_model:
                                original_model = umodel
                            break

                    # Check if any degraded candidate would have won without penalty
                    if not is_failover:
                        for score, prof, brk in candidates:
                            hp = brk.get("health_penalty", 1.0)
                            if hp < 1.0 and prof.name != best_profile.name:
                                # Recompute score without health penalty
                                original_score = score / hp if hp > 0 else score
                                if original_score > best_score:
                                    is_failover = True
                                    original_model = prof.name
                                    break

                    if is_failover and original_model:
                        status = self._health_monitor.get_status(original_model)
                        status_str = status.value if hasattr(status, 'value') else str(status)
                        reason += f" [failover from {original_model}: {status_str}]"
            except Exception as e:
                logger.debug("Failover detection error: %s", e)

        result = SmartRoutingResult(
            model=best_profile.name,
            score=best_score,
            task_score=best_breakdown["task_score"],
            speed_weight=best_breakdown["speed_weight"],
            context_fit=best_breakdown["context_fit"],
            reason=reason,
            alternatives=alternatives,
            profile_used=True,
            fallback=False,
            feedback_adjusted=fb_adjusted,
            failover=is_failover,
            original_model=original_model,
        )

        if not excluded_models:
            self._cache[cache_key] = result
        return result

    def select_for_pipeline(
        self,
        step_types: list[str],
        required_contexts: dict[str, int] | None = None,
    ) -> dict[str, SmartRoutingResult]:
        """Select optimal models for each step in a pipeline.

        Args:
            step_types: List of pipeline step types
            required_contexts: Optional per-step context requirements

        Returns:
            Dict mapping step_type to SmartRoutingResult
        """
        results = {}
        for st in step_types:
            ctx = (required_contexts or {}).get(st)
            results[st] = self.select_model(st, required_context=ctx)
        return results

    def clear_cache(self):
        """Clear the internal routing cache."""
        self._cache.clear()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def configure(
        self,
        enabled: bool | None = None,
        default_model: str | None = None,
        speed_preference: str | None = None,
        speed_weights: dict[str, float] | None = None,
    ):
        """Update router configuration."""
        if enabled is not None:
            self._enabled = enabled
        if default_model is not None:
            self._default_model = default_model
        if speed_preference is not None:
            self.speed_preference = speed_preference
        if speed_weights is not None:
            self._speed_weights.update(speed_weights)
        self._cache.clear()

    def get_config(self) -> dict[str, Any]:
        """Return current configuration as a dictionary."""
        config = {
            "enabled": self._enabled,
            "profiles_available": PROFILES_AVAILABLE,
            "operational": self.enabled,
            "default_model": self._default_model,
            "speed_preference": self._speed_preference,
            "speed_weights": dict(self._speed_weights),
            "context_requirements": dict(self._context_requirements),
            "profile_count": self._profile_manager.count if self._profile_manager else 0,
        }
        # S62: Include feedback adapter status
        if self._feedback_adapter is not None and ADAPTIVE_ROUTING_AVAILABLE:
            try:
                config["feedback_routing_enabled"] = self._feedback_adapter.enabled
                config["feedback_routing_active"] = self._feedback_adapter.has_active_adjustments()
            except Exception:
                config["feedback_routing_enabled"] = False
                config["feedback_routing_active"] = False
        else:
            config["feedback_routing_enabled"] = False
            config["feedback_routing_active"] = False
        # S63: Include health monitor status
        if self._health_monitor is not None and HEALTH_MONITOR_AVAILABLE:
            try:
                config["health_monitor_enabled"] = self._health_monitor.enabled
                config["health_monitor_running"] = self._health_monitor.running
                config["auto_failover"] = self._health_monitor.auto_failover
            except Exception:
                config["health_monitor_enabled"] = False
                config["health_monitor_running"] = False
                config["auto_failover"] = False
        else:
            config["health_monitor_enabled"] = False
            config["health_monitor_running"] = False
            config["auto_failover"] = False
        # S67: Include learned router status
        config["learned_router_available"] = LEARNED_ROUTER_IN_SMART
        if LEARNED_ROUTER_IN_SMART and self._learned_router is not None:
            try:
                lr_status = self._learned_router.get_status()
                config["learned_router_enabled"] = lr_status.get("enabled", False)
                config["learned_router_trained"] = lr_status.get("trained", False)
                config["learned_router_samples"] = lr_status.get("sample_count", 0)
            except Exception:
                config["learned_router_enabled"] = False
                config["learned_router_trained"] = False
                config["learned_router_samples"] = 0
        else:
            config["learned_router_enabled"] = False
            config["learned_router_trained"] = False
            config["learned_router_samples"] = 0
        return config

    def save_config(self, path: Path | None = None) -> bool:
        """Save current configuration to YAML file.

        Args:
            path: Output path (default: config/smart_routing.yaml)

        Returns:
            True if saved successfully
        """
        target = path or self._config_path
        try:
            data = {
                "smart_routing": {
                    "enabled": self._enabled,
                    "default_model": self._default_model,
                    "speed_preference": self._speed_preference,
                    "speed_weights": dict(self._speed_weights),
                    "context_requirements": dict(self._context_requirements),
                }
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Smart routing config saved to {target}")
            return True
        except Exception as e:
            logger.error(f"Error saving smart routing config: {e}")
            return False

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _compute_task_score(self, profile, task_type: str) -> float:
        """Compute task score using task_scores dict or fallback method.

        Applies feedback-based adjustments (S62) when the adaptive
        routing adapter is available and active.
        """
        # Use numeric task_scores if available (S54 enhancement)
        task_scores = getattr(profile, "task_scores", None)
        if task_scores and isinstance(task_scores, dict):
            if task_type in task_scores:
                base_score = float(task_scores[task_type])
            else:
                base_score = None
                for key, score in task_scores.items():
                    if task_type.startswith(key) or key.startswith(task_type):
                        base_score = float(score)
                        break
                if base_score is None:
                    base_score = profile.score_for_task(task_type)
        else:
            # Fallback to existing scoring method
            base_score = profile.score_for_task(task_type)

        # S62: Apply feedback-based adjustment
        if self._feedback_adapter is not None and ADAPTIVE_ROUTING_AVAILABLE:
            try:
                adj = self._feedback_adapter.get_adjustment(profile.name, task_type)
                if adj != 0.0:
                    base_score = max(0.0, min(1.0, base_score + adj))
            except Exception as e:
                logger.debug("Feedback adjustment failed for %s/%s: %s",
                             profile.name, task_type, e)

        return base_score

    def _get_speed_adjustment(self, prefer_speed: bool | None = None) -> dict[str, float]:
        """Get speed tier adjustment multipliers based on preference."""
        if prefer_speed is True:
            pref = "fast"
        elif prefer_speed is False:
            pref = "quality"
        else:
            pref = self._speed_preference

        if pref == "fast":
            return {"fast": 1.3, "medium": 1.0, "slow": 0.7}
        elif pref == "quality":
            return {"fast": 0.8, "medium": 1.0, "slow": 1.2}
        return {"fast": 1.0, "medium": 1.0, "slow": 1.0}

    def _build_reason(self, profile, step_type, task_types, breakdown):
        """Build human-readable reason for model selection."""
        best_match = step_type
        for tt in task_types:
            if tt in getattr(profile, "recommended_for", []):
                best_match = tt
                break

        parts = [f"Best for '{step_type}'"]
        if best_match != step_type:
            parts.append(f"(matches '{best_match}')")
        parts.append(f"[{profile.quality_tier} quality, {profile.speed_tier} speed]")
        if breakdown["context_fit"] < 1.0:
            fit_pct = int(breakdown["context_fit"] * 100)
            parts.append(f"(context fit: {fit_pct}%)")
        return " ".join(parts)

    # -------------------------------------------------------------------------
    # Integration helpers
    # -------------------------------------------------------------------------

    def override_routing(self, routing, step_type: str):
        """Create a modified routing with smart-selected model.

        If smart routing finds a better model, returns modified routing.
        Otherwise returns original routing unchanged.

        Args:
            routing: Original RoutingResult from the router
            step_type: Pipeline step type

        Returns:
            Modified or original RoutingResult
        """
        if not self.enabled:
            return routing

        result = self.select_model(step_type)
        if result.fallback:
            return routing

        try:
            from dataclasses import replace
            new_routing = replace(
                routing,
                model=result.model,
                explanation=f"Smart routed: {result.reason}",
            )
            if hasattr(new_routing, "routing_reason"):
                new_routing.routing_reason = result.to_dict()
            return new_routing
        except Exception as e:
            logger.warning(f"SmartRouter: failed to override routing: {e}")
            return routing

    def to_dict(self) -> dict[str, Any]:
        """Export full state for debugging/API."""
        config = self.get_config()
        config["cache_size"] = len(self._cache)
        config["pipeline_task_mapping"] = dict(PIPELINE_TO_TASK_MAPPING)
        return config

    # -------------------------------------------------------------------------
    # S67: Learned router integration
    # -------------------------------------------------------------------------

    def classify_task_type(
        self,
        query: str,
        yaml_task_type: str = "general",
    ) -> tuple[str, str, float]:
        """Classify query task type, optionally using the ML learned router.

        When the learned router is available, trained, and enabled, it may
        override the YAML-derived task_type. Otherwise returns yaml_task_type
        unchanged. Also logs the routing decision for A/B metrics.

        Args:
            query: Raw query text.
            yaml_task_type: Task type already determined by the YAML heuristic.

        Returns:
            Tuple of (task_type, routing_source, confidence) where
            routing_source is 'learned' or 'yaml'.
        """
        if not LEARNED_ROUTER_IN_SMART or self._learned_router is None:
            return yaml_task_type, "yaml", 0.0

        try:
            prediction = self._learned_router.classify_with_fallback(
                query, yaml_task_type
            )
            source = "yaml" if prediction.fallback_used else "learned"
            # Log the decision for A/B metrics
            ml_pred = self._learned_router.classify(query) if LEARNED_ROUTER_IN_SMART else None
            ml_task = ml_pred.task_type if ml_pred else ""
            ml_conf = ml_pred.confidence if ml_pred else 0.0
            self._learned_router.log_routing_decision(
                query_text=query,
                ml_task_type=ml_task,
                ml_confidence=ml_conf,
                yaml_task_type=yaml_task_type,
                routing_source=source,
            )
            return prediction.task_type, source, prediction.confidence
        except Exception as exc:
            logger.debug("classify_task_type error: %s", exc)
            return yaml_task_type, "yaml", 0.0

    def log_routing_sample(self, query: str, task_type: str) -> None:
        """Log a confirmed routing sample to train the learned router.

        Should be called after each successful routing decision so the
        classifier accumulates real usage patterns.

        Args:
            query: Raw query text.
            task_type: Final task type label used for the request.
        """
        if not LEARNED_ROUTER_IN_SMART or self._learned_router is None:
            return
        try:
            self._learned_router.log_sample(query, task_type, source="smart_router")
            self._learned_router.auto_retrain_if_needed()
        except Exception as exc:
            logger.debug("log_routing_sample error: %s", exc)


# =============================================================================
# SINGLETON
# =============================================================================

smart_router = SmartRouter()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select_model(step_type: str, **kwargs) -> SmartRoutingResult:
    """Shortcut to select a model for a pipeline step type."""
    return smart_router.select_model(step_type, **kwargs)


def select_for_pipeline(step_types: list[str], **kwargs) -> dict[str, SmartRoutingResult]:
    """Shortcut to select models for a full pipeline."""
    return smart_router.select_for_pipeline(step_types, **kwargs)
