#!/usr/bin/env python3
"""
CONTEXT OPTIMIZER -- Opti-Oignon
=====================================

Unified context window orchestrator that replaces the manual 5-step
pipeline in executor.py with a single ``optimize()`` entry point.

Pipeline steps (executed in order):
1. Calculate 6-zone budget via PromptTokenBudgetManager
2. Pass ``project_tokens`` budget to ProjectContextBuilder.build_context()
   -- fixes the disconnected RAG budget gap
3. Apply ConversationCompressor within ``history_tokens`` budget
4. Fall back to SlidingWindowManager if still over budget
5. Validate total fits in context window; emergency truncation if needed

All existing modules are wrapped, never modified. When the optimizer
is disabled (config key ``enabled``), executor keeps its manual pipeline.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config" / "context_optimizer.yaml"


def _load_config(path: Path | None = None) -> dict[str, Any]:
    """Load optimizer config from YAML.

    Args:
        path: Optional override path. Defaults to config/context_optimizer.yaml.

    Returns:
        Parsed config dict with defaults for missing keys.
    """
    target = path or _CONFIG_PATH
    defaults: dict[str, Any] = {
        "enabled": False,
        "active_preset": "balanced",
        "priority_presets": {
            "balanced": {
                "system_ratio": 0.10,
                "project_ratio": 0.25,
                "history_ratio": 0.40,
                "user_ratio": 0.10,
                "reserve_ratio": 0.15,
            },
            "rag_heavy": {
                "system_ratio": 0.10,
                "project_ratio": 0.35,
                "history_ratio": 0.30,
                "user_ratio": 0.10,
                "reserve_ratio": 0.15,
            },
            "history_heavy": {
                "system_ratio": 0.10,
                "project_ratio": 0.15,
                "history_ratio": 0.50,
                "user_ratio": 0.10,
                "reserve_ratio": 0.15,
            },
        },
        "emergency": {
            "enabled": True,
            "min_recent_messages": 2,
            "max_block_chars": 2000,
        },
        "compression": {"strategy": "auto"},
        "report": {"max_retained": 10},
    }
    if target.exists():
        try:
            with open(target, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            # Shallow merge -- top-level keys
            for key, val in loaded.items():
                if isinstance(val, dict) and isinstance(defaults.get(key), dict):
                    defaults[key].update(val)
                else:
                    defaults[key] = val
        except Exception as exc:
            logger.warning("Failed to load context optimizer config: %s", exc)
    return defaults


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ZoneReport:
    """Per-zone budget vs actual metrics.

    Attributes:
        zone: Zone name (system, project, history, user, reserve).
        budgeted_tokens: Tokens allocated by the budget engine.
        actual_tokens: Tokens actually used after optimization.
        trimmed_tokens: Tokens removed by trimming/compression.
        strategy: Strategy applied to this zone (none, compressed,
            sliding_window, truncated).
        detail: Human-readable detail about what happened.
    """
    zone: str
    budgeted_tokens: int = 0
    actual_tokens: int = 0
    trimmed_tokens: int = 0
    strategy: str = "none"
    detail: str = ""

    @property
    def within_budget(self) -> bool:
        """Whether actual usage fits within budget."""
        return self.actual_tokens <= self.budgeted_tokens

    def as_dict(self) -> dict[str, Any]:
        """Serialize for API responses."""
        return {
            "zone": self.zone,
            "budgeted_tokens": self.budgeted_tokens,
            "actual_tokens": self.actual_tokens,
            "trimmed_tokens": self.trimmed_tokens,
            "strategy": self.strategy,
            "detail": self.detail,
            "within_budget": self.within_budget,
        }


@dataclass
class OptimizationReport:
    """Full report of a context optimization pass.

    Attributes:
        model: Model name the optimization targeted.
        total_window: Total context window size (tokens).
        zones: Per-zone reports.
        total_budgeted: Sum of all zone budgets.
        total_actual: Sum of all zone actual usage.
        total_trimmed: Total tokens trimmed across all zones.
        overflow: Whether emergency truncation was needed.
        preset_used: Priority preset that was active.
        duration_ms: Time taken for the optimization pass.
        timestamp: Unix timestamp of the report.
    """
    model: str = ""
    total_window: int = 0
    zones: list[ZoneReport] = field(default_factory=list)
    total_budgeted: int = 0
    total_actual: int = 0
    total_trimmed: int = 0
    overflow: bool = False
    preset_used: str = "balanced"
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for API responses."""
        return {
            "model": self.model,
            "total_window": self.total_window,
            "zones": [z.as_dict() for z in self.zones],
            "total_budgeted": self.total_budgeted,
            "total_actual": self.total_actual,
            "total_trimmed": self.total_trimmed,
            "overflow": self.overflow,
            "preset_used": self.preset_used,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class OptimizedContext:
    """Result of a full optimization pass.

    Attributes:
        system_prompt: Final system prompt (with project context injected).
        messages: Final Ollama-ready messages list.
        total_tokens: Estimated total token count.
        report: Detailed optimization report.
    """
    system_prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    total_tokens: int = 0
    report: OptimizationReport = field(default_factory=OptimizationReport)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for API responses."""
        return {
            "system_prompt_length": len(self.system_prompt),
            "messages_count": len(self.messages),
            "total_tokens": self.total_tokens,
            "report": self.report.as_dict(),
        }


# ---------------------------------------------------------------------------
# Token estimation helper (local, avoids circular imports)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str, chars_per_token: float = 3.7) -> int:
    """Quick token estimation.

    Uses calibrated model-family estimation when available, otherwise
    falls back to a fixed chars_per_token ratio.

    Args:
        text: Text to estimate.
        chars_per_token: Chars-per-token ratio.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def _estimate_messages_tokens(
    messages: list[dict[str, str]],
    chars_per_token: float = 3.7,
) -> int:
    """Estimate total tokens for a list of messages.

    Args:
        messages: Ollama-format messages.
        chars_per_token: Chars-per-token ratio.

    Returns:
        Total estimated tokens.
    """
    total = 0
    for msg in messages:
        total += _estimate_tokens(msg.get("content", ""), chars_per_token)
    return total


# ---------------------------------------------------------------------------
# ContextOptimizer
# ---------------------------------------------------------------------------

class ContextOptimizer:
    """Unified context window orchestrator.

    Replaces the manual 5-step pipeline in executor.py with a single
    ``optimize()`` call that enforces all zone budgets and handles
    overflow/redistribution.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: Path | None = None,
        budget_manager: Any | None = None,
        project_context_builder: Any | None = None,
        conversation_compressor: Any | None = None,
        sliding_window_manager: Any | None = None,
        context_manager: Any | None = None,
    ):
        """Initialize the optimizer.

        Args:
            config: Direct config dict (overrides file loading).
            config_path: Path to YAML config.
            budget_manager: PromptTokenBudgetManager instance.
            project_context_builder: ProjectContextBuilder instance.
            conversation_compressor: ConversationCompressor instance.
            sliding_window_manager: SlidingWindowManager instance.
            context_manager: ContextManager instance (S1) for token estimation.
        """
        self._config = config if config is not None else _load_config(config_path)
        self._budget_manager = budget_manager
        self._project_builder = project_context_builder
        self._compressor = conversation_compressor
        self._sliding_window = sliding_window_manager
        self._context_manager = context_manager

        # Report history (bounded deque)
        max_reports = self._config.get("report", {}).get("max_retained", 10)
        self._reports: collections.deque[OptimizationReport] = collections.deque(
            maxlen=max(1, max_reports)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the optimizer is enabled."""
        return bool(self._config.get("enabled", False))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the optimizer at runtime."""
        self._config["enabled"] = bool(value)

    @property
    def active_preset(self) -> str:
        """Currently active priority preset name."""
        return str(self._config.get("active_preset", "balanced"))

    @active_preset.setter
    def active_preset(self, value: str) -> None:
        """Set the active preset."""
        presets = self._config.get("priority_presets", {})
        if value not in presets and value != "custom":
            raise ValueError(
                f"Unknown preset '{value}'. Available: {list(presets.keys())} + 'custom'"
            )
        self._config["active_preset"] = value

    @property
    def priority_presets(self) -> dict[str, dict[str, float]]:
        """Available priority presets."""
        return dict(self._config.get("priority_presets", {}))

    @property
    def last_report(self) -> OptimizationReport | None:
        """Most recent optimization report."""
        return self._reports[-1] if self._reports else None

    @property
    def reports(self) -> list[OptimizationReport]:
        """All retained optimization reports (most recent last)."""
        return list(self._reports)

    @property
    def config(self) -> dict[str, Any]:
        """Current configuration (read-only copy)."""
        return dict(self._config)

    # ------------------------------------------------------------------
    # Configuration update
    # ------------------------------------------------------------------

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Update configuration at runtime.

        Supports updating enabled, active_preset, and priority_presets.
        Returns the updated config.

        Args:
            updates: Dict of keys to update.

        Returns:
            Updated config dict.
        """
        for key in ("enabled", "active_preset"):
            if key in updates:
                if key == "active_preset":
                    self.active_preset = updates[key]
                else:
                    self._config[key] = updates[key]

        if "priority_presets" in updates and isinstance(updates["priority_presets"], dict):
            existing = self._config.setdefault("priority_presets", {})
            existing.update(updates["priority_presets"])

        if "custom_ratios" in updates and isinstance(updates["custom_ratios"], dict):
            self._config.setdefault("priority_presets", {})["custom"] = updates["custom_ratios"]

        return dict(self._config)

    # ------------------------------------------------------------------
    # Priority override resolution
    # ------------------------------------------------------------------

    def _resolve_priority_overrides(
        self,
        preset: str | None = None,
        custom_ratios: dict[str, float] | None = None,
    ) -> dict[str, float] | None:
        """Resolve priority overrides to ratio dict.

        Args:
            preset: Named preset (overrides active_preset).
            custom_ratios: Direct ratio overrides (highest priority).

        Returns:
            Ratio dict or None if using default budget ratios.
        """
        if custom_ratios:
            return custom_ratios

        target_preset = preset or self.active_preset
        if target_preset == "balanced":
            # balanced = default ratios from token_budget.yaml
            return None

        presets = self._config.get("priority_presets", {})
        if target_preset in presets:
            return presets[target_preset]

        return None

    # ------------------------------------------------------------------
    # Token estimation (uses context_manager when available)
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str, model: str = "") -> int:
        """Estimate token count using best available method.

        Priority: context_manager > calibrated estimation > fallback.

        Args:
            text: Text to estimate.
            model: Model name for model-aware estimation.

        Returns:
            Estimated token count.
        """
        if self._context_manager is not None and model:
            try:
                return self._context_manager.estimate_tokens(text, model)
            except Exception:
                pass
        # Use calibrated module-level estimation when model is known
        if model:
            try:
                from opti_oignon.context_manager import estimate_tokens_calibrated
                return estimate_tokens_calibrated(text, model)
            except ImportError:
                pass
        return _estimate_tokens(text)

    def _estimate_messages_tokens(
        self, messages: list[dict[str, str]], model: str = ""
    ) -> int:
        """Estimate total tokens for messages.

        Args:
            messages: List of messages.
            model: Model name.

        Returns:
            Total estimated tokens.
        """
        return sum(
            self._estimate_tokens(m.get("content", ""), model)
            for m in messages
        )

    # ------------------------------------------------------------------
    # Core optimization pipeline
    # ------------------------------------------------------------------

    def optimize(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        conversation_id: str | None = None,
        project_id: str | None = None,
        rag_query: str | None = None,
        preset: str | None = None,
        custom_ratios: dict[str, float] | None = None,
        project_active: bool = False,
        fingerprint_active: bool = False,
        context_window_override: int = 0,
        manifest_block: str | None = None,
    ) -> OptimizedContext:
        """Run the full optimization pipeline.

        Single entry point that replaces the manual 5-step pipeline.

        Args:
            model: Ollama model name.
            system_prompt: Base system prompt (before project context).
            user_message: Current user message.
            conversation_history: Existing conversation messages (oldest first).
                If None, an empty list is used.
            conversation_id: Conversation UUID (for compression context).
            project_id: Active project ID (for RAG injection).
            rag_query: Query for RAG retrieval (defaults to user_message).
            preset: Priority preset override.
            custom_ratios: Custom zone ratios override.
            project_active: Whether a project is active.
            fingerprint_active: Whether session fingerprinting is active.
            context_window_override: Override context window size.
            manifest_block: Optional capability block pinned above the
                compressed history. Its cost is carved from the history
                budget and it survives every trim, including emergency
                truncation. None or an empty string is the exact
                pre-pin behavior (no zone, no extra message).

        Returns:
            OptimizedContext with final messages and report.
        """
        t_start = time.monotonic()
        history = list(conversation_history or [])
        zones: list[ZoneReport] = []

        # -- Step 1: Calculate 6-zone budget --
        budget = self._calculate_budget(
            model=model,
            project_active=project_active or project_id is not None,
            fingerprint_active=fingerprint_active,
            context_window_override=context_window_override,
            preset=preset,
            custom_ratios=custom_ratios,
        )

        # Capability block: a pinned segment sitting above the
        # compressed history. Measured once with the same estimator the
        # zones use; its cost is carved from the history budget so the
        # summary cedes room to it, and it survives every trim below.
        manifest_block = manifest_block or ""
        manifest_tokens = (
            self._estimate_tokens(manifest_block, model) if manifest_block else 0
        )

        # -- Step 2: Inject project context with budget passthrough --
        project_text = ""
        project_zone = ZoneReport(zone="project", budgeted_tokens=budget.project_tokens)

        if project_id and self._project_builder is not None:
            project_text, project_zone = self._inject_project_context(
                project_id=project_id,
                query=rag_query or user_message,
                budget_tokens=budget.project_tokens,
                model=model,
            )

        # Augment system prompt with project context
        final_system = system_prompt
        if project_text:
            final_system = system_prompt + "\n\n" + project_text

        zones.append(project_zone)

        # -- System zone report --
        system_actual = self._estimate_tokens(final_system, model)
        zones.append(ZoneReport(
            zone="system",
            budgeted_tokens=budget.system_tokens,
            actual_tokens=system_actual,
            trimmed_tokens=max(0, system_actual - budget.system_tokens),
            strategy="fixed",
            detail="System prompt is always included in full",
        ))

        # -- User zone report --
        user_actual = self._estimate_tokens(user_message, model)
        zones.append(ZoneReport(
            zone="user",
            budgeted_tokens=budget.user_tokens,
            actual_tokens=user_actual,
            trimmed_tokens=0,
            strategy="fixed",
            detail="User message is always included in full",
        ))

        # -- Reserve zone --
        zones.append(ZoneReport(
            zone="reserve",
            budgeted_tokens=budget.reserve_tokens,
            actual_tokens=0,
            strategy="reserved",
            detail="Reserved for generation headroom",
        ))

        # -- Pinned capability zone --
        # Recorded only when a block is present, so the no-block path keeps
        # its exact historical zone set. It never trims.
        if manifest_block:
            zones.append(ZoneReport(
                zone="manifest",
                budgeted_tokens=manifest_tokens,
                actual_tokens=manifest_tokens,
                trimmed_tokens=0,
                strategy="pinned",
                detail="Capability block pinned above the compressed history",
            ))

        # -- Step 3: Compress history within budget --
        # The pinned block's measure is carved from the history budget: the
        # summary is what cedes room to the capability block, nothing else.
        history_budget = max(0, budget.history_tokens - manifest_tokens)
        history_before_tokens = self._estimate_messages_tokens(history, model)
        history, history_zone = self._compress_history(
            history=history,
            budget_tokens=history_budget,
            model=model,
        )

        zones.append(history_zone)

        # -- Step 4: Sliding window fallback --
        history_after_compress = self._estimate_messages_tokens(history, model)
        if history_after_compress > history_budget and self._sliding_window is not None:
            history, sw_zone = self._apply_sliding_window(
                history=history,
                model=model,
                system_tokens=system_actual + user_actual,
            )
            # Update history zone with sliding window info
            history_zone.actual_tokens = self._estimate_messages_tokens(history, model)
            history_zone.trimmed_tokens = history_before_tokens - history_zone.actual_tokens
            if sw_zone.strategy != "none":
                history_zone.strategy = f"{history_zone.strategy}+sliding_window"
                history_zone.detail += f" | Sliding window: {sw_zone.detail}"

        # -- Step 5: Emergency truncation --
        # The pinned block occupies space too, so it counts toward the
        # overflow trigger and is subtracted from the history target: the
        # block is never what gets cut.
        total_used = (
            system_actual + user_actual + manifest_tokens
            + self._estimate_messages_tokens(history, model)
        )
        overflow = False
        if total_used > (budget.total_window - budget.reserve_tokens):
            emergency_cfg = self._config.get("emergency", {})
            if emergency_cfg.get("enabled", True):
                history, trimmed = self._emergency_truncate(
                    history=history,
                    target_tokens=(
                        budget.total_window - budget.reserve_tokens
                        - system_actual - user_actual - manifest_tokens
                    ),
                    model=model,
                    min_recent=emergency_cfg.get("min_recent_messages", 2),
                )
                overflow = True
                history_zone.actual_tokens = self._estimate_messages_tokens(history, model)
                history_zone.trimmed_tokens = history_before_tokens - history_zone.actual_tokens
                history_zone.strategy += "+emergency"
                history_zone.detail += f" | Emergency truncation: {trimmed}t removed"

        # -- Build final messages --
        # Order: system prompt, then the pinned capability block (when
        # present), then the compressed history, then the current turn.
        messages: list[dict[str, str]] = [{"role": "system", "content": final_system}]
        if manifest_block:
            messages.append({"role": "system", "content": manifest_block})
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        total_tokens = self._estimate_messages_tokens(messages, model)

        # -- Build report --
        total_budgeted = sum(z.budgeted_tokens for z in zones)
        total_actual = sum(
            z.actual_tokens for z in zones if z.zone != "reserve"
        )
        total_trimmed = sum(z.trimmed_tokens for z in zones)

        report = OptimizationReport(
            model=model,
            total_window=budget.total_window,
            zones=zones,
            total_budgeted=total_budgeted,
            total_actual=total_actual,
            total_trimmed=total_trimmed,
            overflow=overflow,
            preset_used=preset or self.active_preset,
            duration_ms=(time.monotonic() - t_start) * 1000,
        )
        self._reports.append(report)

        return OptimizedContext(
            system_prompt=final_system,
            messages=messages,
            total_tokens=total_tokens,
            report=report,
        )

    # ------------------------------------------------------------------
    # Pipeline step helpers
    # ------------------------------------------------------------------

    def _calculate_budget(
        self,
        model: str,
        project_active: bool,
        fingerprint_active: bool,
        context_window_override: int,
        preset: str | None,
        custom_ratios: dict[str, float] | None,
    ) -> Any:
        """Calculate the 6-zone budget, applying priority overrides.

        Args:
            model: Model name.
            project_active: Whether project context is active.
            fingerprint_active: Whether fingerprint zone is active.
            context_window_override: Context window override.
            preset: Named preset override.
            custom_ratios: Direct ratio overrides.

        Returns:
            PromptTokenBudget instance.
        """
        if self._budget_manager is None:
            # Fallback: create a minimal mock budget
            from dataclasses import dataclass as _dc

            @_dc
            class _FallbackBudget:
                system_tokens: int = 800
                project_tokens: int = 2000
                history_tokens: int = 3200
                user_tokens: int = 800
                reserve_tokens: int = 1200
                total_window: int = 8192
                model: str = ""
                fingerprint_tokens: int = 0

            return _FallbackBudget(model=model)

        # Resolve priority overrides
        overrides = self._resolve_priority_overrides(preset, custom_ratios)

        # The budget engine accepts per-call ratio overrides natively, so
        # hand them over as the keyword it defines instead of rewriting its
        # ratio attributes around the call. The keyword is only supplied
        # when the overrides resolved to something: the plain call keeps
        # its exact historical shape, and an injected engine that only
        # implements that shape keeps working untouched.
        if overrides:
            return self._budget_manager.calculate_budget(
                model=model,
                project_active=project_active,
                context_window_override=context_window_override,
                fingerprint_active=fingerprint_active,
                priority_overrides=overrides,
            )

        return self._budget_manager.calculate_budget(
            model=model,
            project_active=project_active,
            context_window_override=context_window_override,
            fingerprint_active=fingerprint_active,
        )

    def _inject_project_context(
        self,
        project_id: str,
        query: str,
        budget_tokens: int,
        model: str,
    ) -> tuple[str, ZoneReport]:
        """Inject project context with budget passthrough.

        This is the key fix: passes the 6-zone project_tokens budget
        to ProjectContextBuilder.build_context() instead of letting
        it use a fixed per-project default.

        Args:
            project_id: Project UUID.
            query: Query for semantic retrieval.
            budget_tokens: Token budget for the project zone.
            model: Model name.

        Returns:
            Tuple of (context_text, ZoneReport).
        """
        zone = ZoneReport(zone="project", budgeted_tokens=budget_tokens)

        if self._project_builder is None:
            zone.strategy = "unavailable"
            zone.detail = "ProjectContextBuilder not available"
            return "", zone

        try:
            ctx = self._project_builder.build_context(
                project_id, query, budget_tokens=budget_tokens
            )
            text = ctx.context_text or ""
            actual = self._estimate_tokens(text, model)
            zone.actual_tokens = actual
            zone.strategy = "rag"
            zone.detail = (
                f"{ctx.chunks_used} chunks, ~{ctx.total_tokens_estimate}t"
            )
            return text, zone
        except Exception as exc:
            logger.warning("Project context injection failed: %s", exc)
            zone.strategy = "error"
            zone.detail = str(exc)
            return "", zone

    def _compress_history(
        self,
        history: list[dict[str, str]],
        budget_tokens: int,
        model: str,
    ) -> tuple[list[dict[str, str]], ZoneReport]:
        """Compress conversation history within budget.

        Args:
            history: Conversation messages.
            budget_tokens: Token budget for history zone.
            model: Model name.

        Returns:
            Tuple of (compressed_history, ZoneReport).
        """
        before_tokens = self._estimate_messages_tokens(history, model)
        zone = ZoneReport(
            zone="history",
            budgeted_tokens=budget_tokens,
            actual_tokens=before_tokens,
        )

        if not history:
            zone.strategy = "empty"
            zone.detail = "No history to compress"
            return history, zone

        if before_tokens <= budget_tokens:
            zone.strategy = "none"
            zone.detail = "History fits within budget"
            return history, zone

        if self._compressor is None:
            zone.strategy = "no_compressor"
            zone.detail = "ConversationCompressor not available"
            return history, zone

        strategy = self._config.get("compression", {}).get("strategy", "auto")

        try:
            result = self._compressor.compress(
                messages=history,
                budget_tokens=budget_tokens,
                model=model,
                strategy=strategy,
            )
            if result.compressed_count > 0 and result.summary:
                compressed_history: list[dict[str, str]] = []
                if result.summary:
                    compressed_history.append({
                        "role": "system",
                        "content": result.summary,
                    })
                compressed_history.extend(result.recent_messages)

                after_tokens = self._estimate_messages_tokens(
                    compressed_history, model
                )
                zone.actual_tokens = after_tokens
                zone.trimmed_tokens = before_tokens - after_tokens
                zone.strategy = f"compressed:{result.strategy_used}"
                zone.detail = (
                    f"{result.original_count} -> {result.compressed_count} "
                    f"compressed + {len(result.recent_messages)} kept, "
                    f"{result.tokens_saved}t saved"
                )
                return compressed_history, zone
            else:
                zone.strategy = "none"
                zone.detail = "Compression not needed or no effect"
                return history, zone
        except Exception as exc:
            logger.warning("History compression failed: %s", exc)
            zone.strategy = "error"
            zone.detail = f"Compression failed: {exc}"
            return history, zone

    def _apply_sliding_window(
        self,
        history: list[dict[str, str]],
        model: str,
        system_tokens: int,
    ) -> tuple[list[dict[str, str]], ZoneReport]:
        """Apply sliding window as fallback.

        Args:
            history: Current history messages.
            model: Model name.
            system_tokens: Tokens used by system + user.

        Returns:
            Tuple of (trimmed_history, ZoneReport).
        """
        zone = ZoneReport(zone="sliding_window")

        if self._sliding_window is None:
            zone.strategy = "unavailable"
            return history, zone

        try:
            trimmed, stats = self._sliding_window.prepare_messages(
                history, model, system_tokens=system_tokens
            )
            dropped = stats.get("dropped", 0)
            zone.strategy = stats.get("strategy", "unknown")
            zone.detail = (
                f"{stats.get('kept', len(trimmed))} kept, {dropped} dropped"
            )
            return trimmed, zone
        except Exception as exc:
            logger.warning("Sliding window failed: %s", exc)
            zone.strategy = "error"
            zone.detail = str(exc)
            return history, zone

    def _emergency_truncate(
        self,
        history: list[dict[str, str]],
        target_tokens: int,
        model: str,
        min_recent: int = 2,
    ) -> tuple[list[dict[str, str]], int]:
        """Emergency truncation when all other strategies fail.

        Drops oldest messages until within target, keeping at least
        min_recent messages.

        Args:
            history: Current messages.
            target_tokens: Target total tokens for history.
            model: Model name.
            min_recent: Minimum recent messages to keep.

        Returns:
            Tuple of (truncated_history, tokens_removed).
        """
        if not history:
            return history, 0

        current_tokens = self._estimate_messages_tokens(history, model)  # noqa: F841
        tokens_removed = 0

        truncated = list(history)
        while (
            self._estimate_messages_tokens(truncated, model) > target_tokens
            and len(truncated) > min_recent
        ):
            removed = truncated.pop(0)
            removed_tokens = self._estimate_tokens(
                removed.get("content", ""), model
            )
            tokens_removed += removed_tokens

        logger.warning(
            "Emergency truncation: removed %d tokens, %d messages remain",
            tokens_removed,
            len(truncated),
        )
        return truncated, tokens_removed


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_optimizer: ContextOptimizer | None = None


def _build_default_collaborators() -> dict[str, Any]:
    """Resolve the live collaborator singletons for the lazy build.

    Each collaborator sits behind its own guarded import: one that is
    unreachable degrades that seam to None and the orchestrator's own
    per-seam fallbacks take over, instead of the whole accessor failing.

    Returns:
        Keyword arguments for the ContextOptimizer constructor.
    """
    collaborators: dict[str, Any] = {}
    try:
        from opti_oignon.prompt_optimization import prompt_budget_manager
        collaborators["budget_manager"] = prompt_budget_manager
    except ImportError:
        collaborators["budget_manager"] = None
    try:
        from opti_oignon.project_context import project_context_builder
        collaborators["project_context_builder"] = project_context_builder
    except ImportError:
        collaborators["project_context_builder"] = None
    try:
        from opti_oignon.conversation_compressor import conversation_compressor
        collaborators["conversation_compressor"] = conversation_compressor
    except ImportError:
        collaborators["conversation_compressor"] = None
    try:
        from opti_oignon.context_window import sliding_window_manager
        collaborators["sliding_window_manager"] = sliding_window_manager
    except ImportError:
        collaborators["sliding_window_manager"] = None
    try:
        from opti_oignon.context_manager import get_context_manager
        collaborators["context_manager"] = get_context_manager()
    except ImportError:
        collaborators["context_manager"] = None
    return collaborators


def get_optimizer() -> ContextOptimizer:
    """Get the module-level optimizer, building it on first use.

    The first call constructs the orchestrator from the shipped
    configuration and wires the live collaborators; later calls return
    the same instance. An instance installed through init_optimizer
    keeps priority: the lazy build only fills an empty slot.

    Returns:
        The module-level ContextOptimizer.
    """
    global _optimizer
    if _optimizer is None:
        _optimizer = ContextOptimizer(**_build_default_collaborators())
    return _optimizer


def reset_optimizer() -> None:
    """Drop the module-level optimizer so the next access rebuilds it."""
    global _optimizer
    _optimizer = None


def init_optimizer(**kwargs: Any) -> ContextOptimizer:
    """Initialize the module-level optimizer singleton.

    Args:
        **kwargs: Forwarded to ContextOptimizer constructor.

    Returns:
        The initialized ContextOptimizer.
    """
    global _optimizer
    _optimizer = ContextOptimizer(**kwargs)
    return _optimizer
