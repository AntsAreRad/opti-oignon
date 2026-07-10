#!/usr/bin/env python3
"""
Prompt Optimization -- Opti-Oignon
================================================

Foundation layer for prompt intelligence: dynamic token budget
allocation based on actual model context windows, and automatic
task-specific system prompt injection from a YAML template library.

Components:
    - PromptTokenBudget: 5-way budget dataclass
    - PromptTokenBudgetManager: dynamic context window detection + allocation
    - PromptTemplateEngine: YAML-driven task-specific prompt templates (Step 2)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Guard the yaml import so a missing PyYAML degrades the
# module instead of breaking its import (sibling-consistency class).
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover - PyYAML is a core dependency
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Path to config files
_CONFIG_DIR = Path(__file__).parent / "config"
_TOKEN_BUDGET_CONFIG = _CONFIG_DIR / "token_budget.yaml"
_PROMPT_TEMPLATES_CONFIG = _CONFIG_DIR / "prompt_templates.yaml"


# ============================================================================
# Configuration loader
# ============================================================================

def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file with fallback to empty dict.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed dict, or empty dict on failure.
    """
    try:
        if YAML_AVAILABLE and path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load config {path}: {e}")
    return {}


# ============================================================================
# Token Budget dataclass
# ============================================================================

@dataclass(frozen=True)
class PromptTokenBudget:
    """6-way token budget allocation for prompt assembly.

    Attributes:
        system_tokens: Tokens for system prompt (template + instructions).
        project_tokens: Tokens for RAG-injected project context.
        history_tokens: Tokens for conversation history.
        user_tokens: Tokens for current user message + document.
        reserve_tokens: Tokens reserved for generation headroom.
        fingerprint_tokens: Tokens for session fingerprint (carved from history).
        total_window: Total context window size of the model.
        model: Model name this budget was calculated for.
    """

    system_tokens: int
    project_tokens: int
    history_tokens: int
    user_tokens: int
    reserve_tokens: int
    total_window: int
    model: str = ""
    fingerprint_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """Total tokens available for input (everything except reserve)."""
        return (
            self.system_tokens + self.project_tokens
            + self.history_tokens + self.user_tokens
            + self.fingerprint_tokens
        )

    @property
    def total_allocated(self) -> int:
        """Total tokens allocated across all sections."""
        return self.total_input_tokens + self.reserve_tokens

    @property
    def utilization(self) -> float:
        """Fraction of the total window allocated (should be ~1.0)."""
        if self.total_window <= 0:
            return 0.0
        return self.total_allocated / self.total_window

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a dict for API responses."""
        return {
            "model": self.model,
            "total_window": self.total_window,
            "system_tokens": self.system_tokens,
            "project_tokens": self.project_tokens,
            "history_tokens": self.history_tokens,
            "user_tokens": self.user_tokens,
            "reserve_tokens": self.reserve_tokens,
            "fingerprint_tokens": self.fingerprint_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_allocated": self.total_allocated,
            "utilization": round(self.utilization, 4),
        }


# ============================================================================
# Cache entry
# ============================================================================

@dataclass
class _CacheEntry:
    """Cached context window size for a model."""
    context_window: int
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# PromptTokenBudgetManager
# ============================================================================

class PromptTokenBudgetManager:
    """Dynamic token budget allocation based on actual model context windows.

    Queries ollama.show() for real context window sizes, caches results,
    and falls back to YAML-configured or hardcoded defaults.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: Path | None = None,
    ):
        """Initialize with optional config override.

        Args:
            config: Direct config dict (overrides file loading).
            config_path: Path to YAML config (default: token_budget.yaml).
        """
        if config is not None:
            self._config = config
        else:
            path = config_path or _TOKEN_BUDGET_CONFIG
            self._config = _load_yaml_config(path)

        # Allocation ratios
        alloc = self._config.get("allocation", {})
        self._system_ratio = float(alloc.get("system_ratio", 0.10))
        self._project_ratio = float(alloc.get("project_ratio", 0.25))
        self._history_ratio = float(alloc.get("history_ratio", 0.40))
        self._user_ratio = float(alloc.get("user_ratio", 0.10))
        self._reserve_ratio = float(alloc.get("reserve_ratio", 0.15))
        self._fingerprint_ratio = float(alloc.get("fingerprint_ratio", 0.0))

        # Cache settings
        cache_cfg = self._config.get("cache", {})
        self._cache_ttl = int(cache_cfg.get("ttl_seconds", 3600))
        self._cache_max = int(cache_cfg.get("max_entries", 50))

        # Fallback context windows
        self._fallbacks: dict[str, int] = {}
        raw_fallbacks = self._config.get("fallback_context_windows", {})
        if isinstance(raw_fallbacks, dict):
            for model_name, size in raw_fallbacks.items():
                self._fallbacks[str(model_name)] = int(size)

        # Ultimate fallback
        self._default_context_window = int(
            self._config.get("default_context_window", 8192)
        )

        # Minimum budgets
        mins = self._config.get("minimum_budgets", {})
        self._min_system = int(mins.get("system", 256))
        self._min_project = int(mins.get("project", 0))
        self._min_history = int(mins.get("history", 512))
        self._min_user = int(mins.get("user", 256))
        self._min_reserve = int(mins.get("reserve", 512))

        # Context window cache: model_name -> _CacheEntry
        self._cache: dict[str, _CacheEntry] = {}

        logger.info(
            f"PromptTokenBudgetManager initialized: "
            f"ratios=({self._system_ratio}/{self._project_ratio}/"
            f"{self._history_ratio}/{self._user_ratio}/{self._reserve_ratio}), "
            f"fallbacks={len(self._fallbacks)}, "
            f"default_window={self._default_context_window}"
        )

    # ------------------------------------------------------------------
    # Context window detection
    # ------------------------------------------------------------------

    def get_context_window(self, model: str) -> int:
        """Get the context window size for a model.

        Resolution order:
        1. Cached value (if TTL not expired)
        2. ollama.show(model) live query
        3. YAML fallback_context_windows (exact or prefix match)
        4. default_context_window (8192)

        Args:
            model: Ollama model name (e.g. 'qwen3:32b').

        Returns:
            Context window size in tokens.
        """
        # Check cache first
        cached = self._get_cached(model)
        if cached is not None:
            return cached

        # Try ollama.show()
        live_value = self._query_ollama_show(model)
        if live_value is not None:
            self._set_cached(model, live_value)
            return live_value

        # Fallback to YAML config
        fallback = self._match_fallback(model)
        if fallback is not None:
            self._set_cached(model, fallback)
            return fallback

        # Ultimate fallback
        logger.debug(
            f"No context window info for '{model}', "
            f"using default {self._default_context_window}"
        )
        self._set_cached(model, self._default_context_window)
        return self._default_context_window

    def _query_ollama_show(self, model: str) -> int | None:
        """Query ollama.show() for context window size.

        Handles both the legacy dict response and the
        typed ShowResponse of modern ollama-python clients. The object
        branch was previously dead code nested inside the dict branch,
        so live detection silently fell back to YAML/defaults on
        current clients.

        Args:
            model: Model name.

        Returns:
            Context window size, or None if unavailable.
        """
        try:
            import ollama
            info = ollama.show(model)

            # Normalize the relevant fields across response shapes
            if isinstance(info, dict):
                model_info = info.get("model_info", info.get("modelinfo", {}))
                params_str = info.get("parameters", "")
            else:
                model_info = getattr(info, "model_info", None)
                if model_info is None:
                    model_info = getattr(info, "modelinfo", None)
                if model_info is None:
                    model_info = {}
                params_str = getattr(info, "parameters", "") or ""

            # Mapping-like model info: look for a context_length key
            if model_info is not None and hasattr(model_info, "items"):
                for key, val in model_info.items():
                    if "context_length" in str(key).lower():
                        if isinstance(val, (int, float)) and val > 0:
                            logger.debug(
                                f"ollama.show({model}): "
                                f"context_window={int(val)} via {key}"
                            )
                            return int(val)

            # Parameters string: num_ctx line
            if isinstance(params_str, str) and "num_ctx" in params_str:
                for line in params_str.splitlines():
                    line = line.strip()
                    if line.startswith("num_ctx"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                val = int(parts[-1])
                                if val > 0:
                                    logger.debug(
                                        f"ollama.show({model}): "
                                        f"context_window={val} via parameters"
                                    )
                                    return val
                            except ValueError:
                                pass

            logger.debug(f"ollama.show({model}): no context_length found")
            return None

        except ImportError:
            logger.debug("ollama package not installed")
            return None
        except Exception as e:
            logger.debug(f"ollama.show({model}) failed: {e}")
            return None

    def _match_fallback(self, model: str) -> int | None:
        """Match model against YAML fallback context windows.

        Tries exact match first, then prefix match.

        Args:
            model: Model name.

        Returns:
            Context window size or None.
        """
        # Exact match
        if model in self._fallbacks:
            return self._fallbacks[model]

        # Prefix match (e.g. 'qwen3:32b-q4_0' matches 'qwen3:32b')
        for prefix, size in self._fallbacks.items():
            if model.startswith(prefix):
                return size

        return None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _get_cached(self, model: str) -> int | None:
        """Get cached context window if TTL is valid."""
        entry = self._cache.get(model)
        if entry is None:
            return None
        age = time.time() - entry.timestamp
        if age > self._cache_ttl:
            del self._cache[model]
            return None
        return entry.context_window

    def _set_cached(self, model: str, context_window: int) -> None:
        """Store context window in cache, evicting oldest if full."""
        if len(self._cache) >= self._cache_max and model not in self._cache:
            # Evict oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]
        self._cache[model] = _CacheEntry(context_window=context_window)

    def clear_cache(self) -> int:
        """Clear the context window cache.

        Returns:
            Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def cache_stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with entries, max_entries, ttl_seconds, models.
        """
        return {
            "entries": len(self._cache),
            "max_entries": self._cache_max,
            "ttl_seconds": self._cache_ttl,
            "models": list(self._cache.keys()),
        }

    # ------------------------------------------------------------------
    # Budget calculation
    # ------------------------------------------------------------------

    def calculate_budget(
        self,
        model: str,
        conversation_length: int = 0,
        project_active: bool = False,
        context_window_override: int = 0,
        fingerprint_active: bool = False,
        priority_overrides: dict[str, float] | None = None,
    ) -> PromptTokenBudget:
        """Calculate token budget allocation for a model.

        When no project is active, the project_ratio is redistributed
        proportionally across history and user sections.

        The fingerprint budget is carved from the history section when
        fingerprint_active is True (session fingerprinting).

        priority_overrides allows per-call ratio adjustments. Keys
        are 'system_ratio', 'project_ratio', 'history_ratio', 'user_ratio',
        'reserve_ratio'. Missing keys use the configured defaults.

        Args:
            model: Ollama model name.
            conversation_length: Number of messages in conversation (informational).
            project_active: Whether a project context is active.
            context_window_override: Override context window size (0 = auto-detect).
            fingerprint_active: Whether to allocate a fingerprint budget zone.
            priority_overrides: Optional per-call ratio overrides.

        Returns:
            PromptTokenBudget with per-section allocation.
        """
        # Determine context window
        if context_window_override > 0:
            total_window = context_window_override
        else:
            total_window = self.get_context_window(model)

        # Apply priority overrides or use configured ratios
        if priority_overrides:
            sys_ratio = float(priority_overrides.get("system_ratio", self._system_ratio))
            proj_ratio = float(priority_overrides.get("project_ratio", self._project_ratio)) if project_active else 0.0
            hist_ratio = float(priority_overrides.get("history_ratio", self._history_ratio))
            user_ratio = float(priority_overrides.get("user_ratio", self._user_ratio))
            reserve_ratio = float(priority_overrides.get("reserve_ratio", self._reserve_ratio))
        else:
            sys_ratio = self._system_ratio
            proj_ratio = self._project_ratio if project_active else 0.0
            hist_ratio = self._history_ratio
            user_ratio = self._user_ratio
            reserve_ratio = self._reserve_ratio

        fp_ratio = self._fingerprint_ratio if fingerprint_active else 0.0

        # Redistribute the project ratio when no project is active.
        # Use the EFFECTIVE ratios -- the withheld amount
        # comes from the override when provided (so an explicit
        # project_ratio: 0.0 opts out of redistribution entirely), and
        # the shares from the effective history/user ratios rather than
        # the configured defaults.
        if priority_overrides:
            withheld_project = float(
                priority_overrides.get("project_ratio", self._project_ratio)
            )
        else:
            withheld_project = self._project_ratio
        if not project_active and withheld_project > 0:
            denom = hist_ratio + user_ratio
            hist_share = hist_ratio / denom if denom > 0 else 0.5
            user_share = 1.0 - hist_share
            hist_ratio += withheld_project * hist_share
            user_ratio += withheld_project * user_share

        # Fingerprint carved from history
        if fp_ratio > 0:
            hist_ratio = max(0.0, hist_ratio - fp_ratio)

        # Convert ratios to token counts
        system_tokens = max(self._min_system, int(total_window * sys_ratio))
        project_tokens = max(self._min_project, int(total_window * proj_ratio))
        history_tokens = max(self._min_history, int(total_window * hist_ratio))
        user_tokens = max(self._min_user, int(total_window * user_ratio))
        reserve_tokens = max(self._min_reserve, int(total_window * reserve_ratio))
        fingerprint_tokens = int(total_window * fp_ratio) if fp_ratio > 0 else 0

        # Ensure total does not exceed window (adjust reserve if minimums pushed over)
        total = (
            system_tokens + project_tokens + history_tokens
            + user_tokens + reserve_tokens + fingerprint_tokens
        )
        if total > total_window:
            overflow = total - total_window
            # Reduce reserve first, then history
            reserve_reduction = min(overflow, reserve_tokens - self._min_reserve)
            reserve_tokens -= reserve_reduction
            overflow -= reserve_reduction
            if overflow > 0:
                history_reduction = min(overflow, history_tokens - self._min_history)
                history_tokens -= history_reduction

        return PromptTokenBudget(
            system_tokens=system_tokens,
            project_tokens=project_tokens,
            history_tokens=history_tokens,
            user_tokens=user_tokens,
            reserve_tokens=reserve_tokens,
            total_window=total_window,
            model=model,
            fingerprint_tokens=fingerprint_tokens,
        )

    # ------------------------------------------------------------------
    # Configuration access
    # ------------------------------------------------------------------

    @property
    def allocation_ratios(self) -> dict[str, float]:
        """Current allocation ratios."""
        return {
            "system": self._system_ratio,
            "project": self._project_ratio,
            "history": self._history_ratio,
            "user": self._user_ratio,
            "reserve": self._reserve_ratio,
            "fingerprint": self._fingerprint_ratio,
        }

    @property
    def default_context_window(self) -> int:
        """Ultimate fallback context window size."""
        return self._default_context_window

    @property
    def fallback_models(self) -> dict[str, int]:
        """YAML-configured fallback context windows."""
        return dict(self._fallbacks)

    def get_config(self) -> dict[str, Any]:
        """Full configuration as dict for API responses."""
        return {
            "allocation": self.allocation_ratios,
            "cache": self.cache_stats(),
            "default_context_window": self._default_context_window,
            "fallback_models": self._fallbacks,
            "minimum_budgets": {
                "system": self._min_system,
                "project": self._min_project,
                "history": self._min_history,
                "user": self._min_user,
                "reserve": self._min_reserve,
            },
        }

    def reload_config(self, config_path: Path | None = None) -> None:
        """Reload configuration from YAML file.

        Args:
            config_path: Path to config file (default: token_budget.yaml).
        """
        path = config_path or _TOKEN_BUDGET_CONFIG
        self._config = _load_yaml_config(path)

        alloc = self._config.get("allocation", {})
        self._system_ratio = float(alloc.get("system_ratio", 0.10))
        self._project_ratio = float(alloc.get("project_ratio", 0.25))
        self._history_ratio = float(alloc.get("history_ratio", 0.40))
        self._user_ratio = float(alloc.get("user_ratio", 0.10))
        self._reserve_ratio = float(alloc.get("reserve_ratio", 0.15))
        self._fingerprint_ratio = float(alloc.get("fingerprint_ratio", 0.0))

        cache_cfg = self._config.get("cache", {})
        self._cache_ttl = int(cache_cfg.get("ttl_seconds", 3600))
        self._cache_max = int(cache_cfg.get("max_entries", 50))

        raw_fallbacks = self._config.get("fallback_context_windows", {})
        self._fallbacks = {}
        if isinstance(raw_fallbacks, dict):
            for model_name, size in raw_fallbacks.items():
                self._fallbacks[str(model_name)] = int(size)

        self._default_context_window = int(
            self._config.get("default_context_window", 8192)
        )

        mins = self._config.get("minimum_budgets", {})
        self._min_system = int(mins.get("system", 256))
        self._min_project = int(mins.get("project", 0))
        self._min_history = int(mins.get("history", 512))
        self._min_user = int(mins.get("user", 256))
        self._min_reserve = int(mins.get("reserve", 512))

        self.clear_cache()
        logger.info("PromptTokenBudgetManager: configuration reloaded")


# ============================================================================
# Prompt Template dataclass
# ============================================================================

@dataclass(frozen=True)
class PromptTemplate:
    """Task-specific prompt template.

    Attributes:
        task_type: Task type this template targets.
        system_prompt: The system prompt text (may contain {variables}).
        temperature_override: Optional temperature override for LLM calls.
        stop_sequences: Optional stop sequences for generation.
        source: Where this template came from (yaml/project/runtime/fallback).
    """

    task_type: str
    system_prompt: str
    temperature_override: float | None = None
    stop_sequences: list[str] = field(default_factory=list)
    source: str = "yaml"

    def as_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "task_type": self.task_type,
            "system_prompt": self.system_prompt,
            "temperature_override": self.temperature_override,
            "stop_sequences": self.stop_sequences,
            "source": self.source,
        }


# ============================================================================
# PromptTemplateEngine
# ============================================================================

class PromptTemplateEngine:
    """YAML-driven task-specific prompt template engine.

    Resolution order for get_template():
    1. Runtime override (set via API)
    2. Project-specific override (if project_id provided)
    3. Task-type template from YAML
    4. General fallback template
    """

    # Fallback system prompt when nothing is configured
    _ULTIMATE_FALLBACK = "You are a helpful assistant. Respond in the user's language."

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: Path | None = None,
    ):
        """Initialize with optional config override.

        Args:
            config: Direct config dict (overrides file loading).
            config_path: Path to YAML config (default: prompt_templates.yaml).
        """
        if config is not None:
            self._config = config
        else:
            path = config_path or _PROMPT_TEMPLATES_CONFIG
            self._config = _load_yaml_config(path)

        self._language_rule = str(
            self._config.get("language_rule", "Respond in the user's language.")
        )

        # Parse templates
        self._templates: dict[str, dict[str, Any]] = {}
        raw_templates = self._config.get("templates", {})
        if isinstance(raw_templates, dict):
            self._templates = dict(raw_templates)

        # Parse project overrides
        self._project_overrides: dict[str, dict[str, dict[str, Any]]] = {}
        raw_overrides = self._config.get("project_overrides", {})
        if isinstance(raw_overrides, dict):
            self._project_overrides = dict(raw_overrides)

        # Runtime overrides (set via API, ephemeral)
        self._runtime_overrides: dict[str, dict[str, Any]] = {}

        logger.info(
            f"PromptTemplateEngine initialized: "
            f"{len(self._templates)} templates, "
            f"{len(self._project_overrides)} project overrides"
        )

    # ------------------------------------------------------------------
    # Template retrieval
    # ------------------------------------------------------------------

    def get_template(
        self,
        task_type: str,
        project_id: str | None = None,
    ) -> PromptTemplate:
        """Get the best template for a task type.

        Resolution order:
        1. Runtime override for task_type
        2. Project-specific override (if project_id provided)
        3. Task-type template from YAML
        4. General fallback template

        Args:
            task_type: Task type (e.g. 'code_r', 'scientific_writing').
            project_id: Optional project ID for project-specific overrides.

        Returns:
            PromptTemplate with resolved system prompt.
        """
        # 1. Runtime override
        if task_type in self._runtime_overrides:
            raw = self._runtime_overrides[task_type]
            return self._parse_template(task_type, raw, source="runtime")

        # 2. Project-specific override
        if project_id and project_id in self._project_overrides:
            proj_templates = self._project_overrides[project_id]
            if isinstance(proj_templates, dict) and task_type in proj_templates:
                raw = proj_templates[task_type]
                return self._parse_template(task_type, raw, source="project")

        # 3. Task-type template
        if task_type in self._templates:
            raw = self._templates[task_type]
            return self._parse_template(task_type, raw, source="yaml")

        # 4. General fallback
        if "general" in self._templates:
            raw = self._templates["general"]
            return self._parse_template(task_type, raw, source="fallback")

        # 5. Ultimate fallback
        logger.debug(f"No template for task_type='{task_type}', using ultimate fallback")
        return PromptTemplate(
            task_type=task_type,
            system_prompt=self._ULTIMATE_FALLBACK,
            source="fallback",
        )

    def _parse_template(
        self,
        task_type: str,
        raw: dict[str, Any] | str,
        source: str = "yaml",
    ) -> PromptTemplate:
        """Parse a raw template dict or string into PromptTemplate.

        Args:
            task_type: Task type label.
            raw: Template data (dict with system_prompt key, or plain string).
            source: Source label.

        Returns:
            PromptTemplate instance.
        """
        if isinstance(raw, str):
            return PromptTemplate(
                task_type=task_type,
                system_prompt=raw,
                source=source,
            )

        if not isinstance(raw, dict):
            return PromptTemplate(
                task_type=task_type,
                system_prompt=self._ULTIMATE_FALLBACK,
                source="fallback",
            )

        system_prompt = str(raw.get("system_prompt", self._ULTIMATE_FALLBACK))
        temp = raw.get("temperature_override")
        temperature_override = float(temp) if temp is not None else None
        stop_seqs = raw.get("stop_sequences", [])
        if not isinstance(stop_seqs, list):
            stop_seqs = []

        return PromptTemplate(
            task_type=task_type,
            system_prompt=system_prompt,
            temperature_override=temperature_override,
            stop_sequences=list(stop_seqs),
            source=source,
        )

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self,
        template: PromptTemplate,
        context: dict[str, str] | None = None,
    ) -> str:
        """Interpolate variables in the template's system prompt.

        Substitutes {variable_name} placeholders with values from context.
        Always injects {language_rule} automatically.
        Unknown variables are left as-is (no KeyError).

        Args:
            template: PromptTemplate to interpolate.
            context: Dict of variable_name -> value for substitution.

        Returns:
            Interpolated system prompt string.
        """
        subs: dict[str, str] = {
            "language_rule": self._language_rule,
        }
        if context:
            subs.update(context)

        result = template.system_prompt
        for key, value in subs.items():
            placeholder = "{" + key + "}"
            result = result.replace(placeholder, str(value))

        return result

    # ------------------------------------------------------------------
    # Template listing and runtime overrides
    # ------------------------------------------------------------------

    def list_templates(self) -> list[dict[str, Any]]:
        """List all available task-type templates.

        Returns:
            List of template summary dicts.
        """
        result = []
        for task_type in sorted(self._templates.keys()):
            tpl = self.get_template(task_type)
            result.append({
                "task_type": task_type,
                "has_temperature_override": tpl.temperature_override is not None,
                "temperature_override": tpl.temperature_override,
                "source": tpl.source,
                "prompt_length": len(tpl.system_prompt),
            })
        return result

    def set_runtime_override(
        self,
        task_type: str,
        system_prompt: str,
        temperature_override: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> PromptTemplate:
        """Set a runtime override for a task type.

        Runtime overrides take highest priority and persist until cleared
        or the process restarts.

        Args:
            task_type: Task type to override.
            system_prompt: New system prompt text.
            temperature_override: Optional temperature override.
            stop_sequences: Optional stop sequences.

        Returns:
            The created PromptTemplate.
        """
        raw: dict[str, Any] = {"system_prompt": system_prompt}
        if temperature_override is not None:
            raw["temperature_override"] = temperature_override
        if stop_sequences is not None:
            raw["stop_sequences"] = stop_sequences

        self._runtime_overrides[task_type] = raw
        logger.info(f"Runtime override set for task_type='{task_type}'")
        return self._parse_template(task_type, raw, source="runtime")

    def clear_runtime_override(self, task_type: str) -> bool:
        """Clear a runtime override for a task type.

        Args:
            task_type: Task type to clear.

        Returns:
            True if an override was removed, False if none existed.
        """
        if task_type in self._runtime_overrides:
            del self._runtime_overrides[task_type]
            logger.info(f"Runtime override cleared for task_type='{task_type}'")
            return True
        return False

    def clear_all_runtime_overrides(self) -> int:
        """Clear all runtime overrides.

        Returns:
            Number of overrides cleared.
        """
        count = len(self._runtime_overrides)
        self._runtime_overrides.clear()
        return count

    @property
    def available_task_types(self) -> list[str]:
        """List of all configured task types."""
        return sorted(self._templates.keys())

    @property
    def language_rule(self) -> str:
        """Current language rule text."""
        return self._language_rule

    def get_config(self) -> dict[str, Any]:
        """Full engine configuration for API responses."""
        return {
            "language_rule": self._language_rule,
            "task_types": self.available_task_types,
            "template_count": len(self._templates),
            "project_override_count": len(self._project_overrides),
            "runtime_override_count": len(self._runtime_overrides),
            "runtime_overrides": list(self._runtime_overrides.keys()),
        }

    def reload_config(self, config_path: Path | None = None) -> None:
        """Reload configuration from YAML file.

        Args:
            config_path: Path to config file (default: prompt_templates.yaml).
        """
        path = config_path or _PROMPT_TEMPLATES_CONFIG
        self._config = _load_yaml_config(path)

        self._language_rule = str(
            self._config.get("language_rule", "Respond in the user's language.")
        )

        self._templates = {}
        raw_templates = self._config.get("templates", {})
        if isinstance(raw_templates, dict):
            self._templates = dict(raw_templates)

        self._project_overrides = {}
        raw_overrides = self._config.get("project_overrides", {})
        if isinstance(raw_overrides, dict):
            self._project_overrides = dict(raw_overrides)

        # Runtime overrides are NOT cleared on reload (ephemeral by design)
        logger.info("PromptTemplateEngine: configuration reloaded")


# ============================================================================
# Module-level singletons
# ============================================================================

# Guarded like every sibling store, for consistency;
# consumers (executor, deps, routes) already None-check both names.
try:
    prompt_budget_manager = PromptTokenBudgetManager()
    prompt_template_engine = PromptTemplateEngine()
    PROMPT_OPTIMIZATION_MODULE_AVAILABLE = True
except Exception as _init_exc:  # pragma: no cover - defensive init guard
    logger.warning(
        "Prompt optimization singletons failed to initialize: %s", _init_exc
    )
    prompt_budget_manager = None  # type: ignore[assignment]
    prompt_template_engine = None  # type: ignore[assignment]
    PROMPT_OPTIMIZATION_MODULE_AVAILABLE = False
