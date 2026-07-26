#!/usr/bin/env python3
"""
Plugin hook system for Opti-Oignon.

HookManager: register, unregister, and execute hooks at defined points
in the inference pipeline. Each hook runs with error isolation so one
plugin failure never crashes others or the main pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default priority (lower = runs first)
DEFAULT_PRIORITY = 100


@dataclass
class HookContext:
    """Context object passed to every hook invocation.

    Provides access to conversation data, model config, and arbitrary
    metadata.  HK-01: each hook receives its OWN copy of the chain data;
    in-place mutation (including ``set()``) affects only that local view.
    To propagate changes downstream, a hook must RETURN a dict -- it is
    merged into the chain data for subsequent hooks and ``final_data``.
    """

    hook_name: str
    plugin_name: str
    conversation_id: str | None = None
    model: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from data dict."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in this hook's LOCAL data view.

        HK-01: this does NOT propagate downstream; return a dict from
        the hook callback to merge changes into the chain data.
        """
        self.data[key] = value


@dataclass
class HookResult:
    """Result from a single hook execution."""

    plugin_name: str
    hook_name: str
    success: bool
    duration_ms: float = 0.0
    error: str | None = None
    modified_data: dict[str, Any] | None = None


@dataclass
class HookExecutionReport:
    """Aggregated report from executing all hooks at a given point."""

    hook_name: str
    total_hooks: int
    successful: int
    failed: int
    total_duration_ms: float
    results: list[HookResult] = field(default_factory=list)
    final_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class _HookRegistration:
    """Internal: a registered hook callable with metadata."""

    plugin_name: str
    hook_name: str
    callback: Callable[[HookContext], dict[str, Any] | None]
    priority: int = DEFAULT_PRIORITY
    enabled: bool = True


class HookManager:
    """Manages hook registration and execution across all plugins.

    Hook points (defined in plugin_manifest.VALID_HOOKS):
        pre_prompt      — Before prompt construction
        post_prompt     — After prompt construction, before inference
        pre_inference   — Just before sending to the LLM
        post_inference  — After receiving LLM response
        tool_call       — When a tool is invoked
        pipeline_step   — During pipeline execution
        ui_panel        — For UI panel registration

    Hooks are executed in priority order (lower number = higher priority).
    If two hooks have the same priority, they execute in registration order.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[_HookRegistration]] = {}
        self._execution_stats: dict[str, dict[str, float]] = {}

    def register(
        self,
        hook_name: str,
        plugin_name: str,
        callback: Callable[[HookContext], dict[str, Any] | None],
        *,
        priority: int = DEFAULT_PRIORITY,
    ) -> bool:
        """Register a hook callback for a specific hook point.

        Parameters
        ----------
        hook_name : str
            The hook point name (e.g. 'pre_prompt').
        plugin_name : str
            Name of the plugin registering the hook.
        callback : callable
            Function that receives a HookContext and optionally returns
            a dict of modified data.
        priority : int
            Execution priority (lower = runs first). Default 100.

        Returns
        -------
        bool
            True if registered successfully.
        """
        from opti_oignon.plugin_manifest import VALID_HOOKS

        if hook_name not in VALID_HOOKS:
            logger.warning(
                "Cannot register hook '%s' for plugin '%s': invalid hook name",
                hook_name, plugin_name,
            )
            return False

        if not callable(callback):
            logger.warning(
                "Cannot register hook '%s' for plugin '%s': callback not callable",
                hook_name, plugin_name,
            )
            return False

        reg = _HookRegistration(
            plugin_name=plugin_name,
            hook_name=hook_name,
            callback=callback,
            priority=priority,
        )

        if hook_name not in self._hooks:
            self._hooks[hook_name] = []

        self._hooks[hook_name].append(reg)
        # Sort by priority (stable sort preserves registration order for ties)
        self._hooks[hook_name].sort(key=lambda r: r.priority)

        logger.debug(
            "Registered hook '%s' for plugin '%s' (priority=%d)",
            hook_name, plugin_name, priority,
        )
        return True

    def unregister(self, hook_name: str, plugin_name: str) -> int:
        """Remove all hooks for a plugin at a specific hook point.

        Returns the number of hooks removed.
        """
        if hook_name not in self._hooks:
            return 0

        before = len(self._hooks[hook_name])
        self._hooks[hook_name] = [
            r for r in self._hooks[hook_name]
            if r.plugin_name != plugin_name
        ]
        removed = before - len(self._hooks[hook_name])

        if not self._hooks[hook_name]:
            del self._hooks[hook_name]

        if removed > 0:
            logger.debug(
                "Unregistered %d hook(s) '%s' for plugin '%s'",
                removed, hook_name, plugin_name,
            )
        return removed

    def unregister_plugin(self, plugin_name: str) -> int:
        """Remove ALL hooks registered by a specific plugin.

        Returns the total number of hooks removed.
        """
        total_removed = 0
        for hook_name in list(self._hooks.keys()):
            total_removed += self.unregister(hook_name, plugin_name)
        return total_removed

    def execute(
        self,
        hook_name: str,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
        data: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        redact_sensitive: bool = False,
    ) -> HookExecutionReport:
        """Execute all registered hooks for a hook point.

        Hooks are called in priority order. Each hook receives a
        HookContext with the current data. If a hook returns a dict,
        it is merged into the data for subsequent hooks.

        Error isolation: if a hook raises an exception, it is caught
        and logged, and execution continues with the next hook.

        Parameters
        ----------
        hook_name : str
            The hook point to execute.
        conversation_id : str, optional
            Current conversation ID.
        model : str, optional
            Current model name.
        data : dict, optional
            Data to pass through the hook chain.
        config : dict, optional
            Configuration data available to hooks.
        metadata : dict, optional
            Additional metadata for hooks.
        redact_sensitive : bool
            If True, redact sensitive fields (message, response,
            arguments, result) for plugins lacking inference_content
            permission. Default False for backward compatibility.

        Returns
        -------
        HookExecutionReport
            Aggregated results from all hook executions.
        """
        current_data = dict(data or {})
        registrations = self._hooks.get(hook_name, [])

        report = HookExecutionReport(
            hook_name=hook_name,
            total_hooks=len(registrations),
            successful=0,
            failed=0,
            total_duration_ms=0.0,
        )

        start_total = time.perf_counter()

        for reg in registrations:
            if not reg.enabled:
                continue

            # Per-plugin data redaction
            if redact_sensitive:
                plugin_data = redact_hook_data(
                    current_data, reg.plugin_name,
                )
            else:
                plugin_data = dict(current_data)

            ctx = HookContext(
                hook_name=hook_name,
                plugin_name=reg.plugin_name,
                conversation_id=conversation_id,
                model=model,
                data=plugin_data,
                config=dict(config or {}),
                metadata=dict(metadata or {}),
            )

            start = time.perf_counter()
            result = HookResult(
                plugin_name=reg.plugin_name,
                hook_name=hook_name,
                success=False,
            )

            try:
                returned = reg.callback(ctx)
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.success = True
                result.duration_ms = elapsed_ms

                # Merge returned data if any
                if isinstance(returned, dict):
                    # HK-02: never merge the redaction placeholder back
                    # into the shared chain data.  A plugin that received
                    # redacted fields and echoes one (e.g. returns its
                    # ctx.data, or rewrites "message" from the redacted
                    # view) would otherwise overwrite the REAL value for
                    # every downstream hook and for final_data -- at the
                    # pre_inference seam that silently replaces the user's
                    # prompt with the placeholder string.  No legitimate
                    # hook ever writes this literal.
                    merged = {
                        k: v for k, v in returned.items()
                        if v != REDACTED_PLACEHOLDER
                    }
                    current_data.update(merged)
                    result.modified_data = returned
                elif returned is not None:
                    # Hook returned something unexpected but not an error
                    logger.debug(
                        "Hook '%s' from '%s' returned non-dict: %s",
                        hook_name, reg.plugin_name, type(returned).__name__,
                    )

                report.successful += 1

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                result.success = False
                result.duration_ms = elapsed_ms
                result.error = f"{type(exc).__name__}: {exc}"
                report.failed += 1
                logger.warning(
                    "Hook '%s' from plugin '%s' failed: %s",
                    hook_name, reg.plugin_name, exc,
                )

            report.results.append(result)

            # Track execution stats
            stat_key = f"{reg.plugin_name}:{hook_name}"
            if stat_key not in self._execution_stats:
                self._execution_stats[stat_key] = {
                    "calls": 0, "total_ms": 0.0, "errors": 0,
                }
            self._execution_stats[stat_key]["calls"] += 1
            self._execution_stats[stat_key]["total_ms"] += elapsed_ms
            if not result.success:
                self._execution_stats[stat_key]["errors"] += 1

        report.total_duration_ms = (time.perf_counter() - start_total) * 1000
        report.final_data = current_data
        return report

    def has_hooks(self, hook_name: str) -> bool:
        """Check if any hooks are registered for a hook point."""
        return bool(self._hooks.get(hook_name))

    def get_hook_count(self, hook_name: str | None = None) -> int:
        """Get the number of registered hooks.

        If hook_name is provided, count for that hook only.
        Otherwise count all hooks across all points.
        """
        if hook_name:
            return len(self._hooks.get(hook_name, []))
        return sum(len(regs) for regs in self._hooks.values())

    def list_hooks(
        self,
        *,
        hook_name: str | None = None,
        plugin_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List registered hooks with optional filtering.

        Returns list of dicts with plugin_name, hook_name, priority, enabled.
        """
        results: list[dict[str, Any]] = []
        for hname, regs in self._hooks.items():
            if hook_name and hname != hook_name:
                continue
            for reg in regs:
                if plugin_name and reg.plugin_name != plugin_name:
                    continue
                results.append({
                    "plugin_name": reg.plugin_name,
                    "hook_name": reg.hook_name,
                    "priority": reg.priority,
                    "enabled": reg.enabled,
                })
        return results

    def set_hook_enabled(
        self,
        hook_name: str,
        plugin_name: str,
        enabled: bool,
    ) -> bool:
        """Enable or disable a specific plugin's hooks at a hook point.

        Returns True if at least one hook was updated.
        """
        updated = False
        for reg in self._hooks.get(hook_name, []):
            if reg.plugin_name == plugin_name:
                reg.enabled = enabled
                updated = True
        return updated

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get execution statistics for all hooks.

        Returns dict of {plugin:hook -> {calls, total_ms, errors}}.
        """
        return dict(self._execution_stats)

    def reset_stats(self) -> None:
        """Clear all execution statistics."""
        self._execution_stats.clear()

    def clear(self) -> None:
        """Remove all registered hooks and stats."""
        self._hooks.clear()
        self._execution_stats.clear()


# =========================================================================
# Hook data redaction for plugin permission model
# =========================================================================

REDACTED_PLACEHOLDER = "[REDACTED -- requires inference_content permission]"

# Fields in hook data that contain sensitive inference content
_SENSITIVE_FIELDS = frozenset({
    "message",      # User prompt
    "response",     # LLM response
    "arguments",    # Tool call arguments
    "result",       # Tool call result
})

# Fields that are always safe to pass through (non-sensitive metadata)
_SAFE_FIELDS = frozenset({
    "model",
    "duration_ms",
    "tokens_in",
    "tokens_out",
    "conversation_id",
    "tool_name",
    "success",
})


def get_plugin_permissions(plugin_name: str) -> list[str]:
    """Look up a plugin's declared permissions from the registry.

    Returns an empty list if the plugin or registry is unavailable.
    """
    try:
        from opti_oignon.plugin_manifest import plugin_registry
        if plugin_registry is None:
            return []
        record = plugin_registry.get(plugin_name)
        if record is None:
            return []
        return list(record.manifest.permissions)
    except Exception:
        return []


def has_inference_content_permission(plugin_name: str) -> bool:
    """Check if a plugin has the inference_content permission."""
    return "inference_content" in get_plugin_permissions(plugin_name)


def redact_hook_data(
    data: dict[str, Any],
    plugin_name: str,
    *,
    force_redact: bool = False,
) -> dict[str, Any]:
    """Redact sensitive fields from hook data if the plugin lacks permission.

    Parameters
    ----------
    data : dict
        The hook data dict (message, response, model, etc.).
    plugin_name : str
        Name of the plugin that will receive this data.
    force_redact : bool
        If True, always redact regardless of permission (for testing).

    Returns
    -------
    dict
        A copy of data with sensitive fields redacted if needed.
    """
    if not force_redact and has_inference_content_permission(plugin_name):
        return dict(data)  # Full access, return a copy

    redacted = {}
    for key, value in data.items():
        if key in _SENSITIVE_FIELDS:
            redacted[key] = REDACTED_PLACEHOLDER
        else:
            redacted[key] = value

    logger.debug(
        "Redacted hook data for plugin '%s' (no inference_content permission)",
        plugin_name,
    )
    return redacted


# =========================================================================
# Module-level singleton
# =========================================================================

PLUGIN_HOOKS_AVAILABLE = True

hook_manager = HookManager()
