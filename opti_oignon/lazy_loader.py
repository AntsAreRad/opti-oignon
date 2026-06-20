#!/usr/bin/env python3
"""
Lazy module loader for Opti-Oignon.

Defers heavy module imports until first attribute access, reducing
startup time. Modules like RAG (chromadb), coding agent, telemetry,
benchmark, fine-tune tracker, and plugin marketplace are loaded on
demand rather than at application boot.

Usage:
    from opti_oignon.lazy_loader import lazy_import, get_lazy_module

    # The module is not imported yet
    rag = lazy_import("opti_oignon.rag_store")

    # First attribute access triggers the real import
    rag.get_rag_store()
"""

import importlib
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class LazyModule:
    """Proxy for a lazily loaded module.

    The actual import is deferred until the first attribute access.
    Thread-safe via an internal lock.

    Attributes:
        _module_name: Fully qualified module name to import
        _module: Reference to the real module (None if not yet loaded)
        _lock: Lock for thread-safe import
        _load_time: Loading time in seconds
        _error: Exception if loading failed
    """

    def __init__(self, module_name: str):
        """Initialize the lazy module proxy.

        Args:
            module_name: Full dotted module name (e.g. "opti_oignon.rag")
        """
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_module", None)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_load_time", 0.0)
        object.__setattr__(self, "_error", None)

    def _load(self) -> Any:
        """Load the real module if not already done.

        Returns:
            The imported module.

        Raises:
            ImportError: If the module cannot be imported.
        """
        if self._module is not None:
            return self._module

        with self._lock:
            # Double-check after acquiring the lock
            if self._module is not None:
                return self._module

            module_name = self._module_name
            start = time.time()
            try:
                mod = importlib.import_module(module_name)
                elapsed = time.time() - start
                object.__setattr__(self, "_module", mod)
                object.__setattr__(self, "_load_time", elapsed)
                logger.info(
                    f"LazyModule loaded: {module_name} ({elapsed:.3f}s)"
                )
                return mod
            except Exception as e:
                elapsed = time.time() - start
                object.__setattr__(self, "_error", e)
                object.__setattr__(self, "_load_time", elapsed)
                logger.warning(
                    f"LazyModule failed: {module_name} ({elapsed:.3f}s): {e}"
                )
                raise ImportError(
                    f"Lazy import failed for {module_name}: {e}"
                ) from e

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access and load the module if needed."""
        mod = self._load()
        return getattr(mod, name)

    def __repr__(self) -> str:
        if self._module is not None:
            return f"<LazyModule({self._module_name}, loaded in {self._load_time:.3f}s)>"
        if self._error is not None:
            return f"<LazyModule({self._module_name}, FAILED: {self._error})>"
        return f"<LazyModule({self._module_name}, not loaded)>"

    @property
    def is_loaded(self) -> bool:
        """True if the module has been loaded."""
        return self._module is not None

    @property
    def load_time(self) -> float:
        """Time in seconds taken to load the module (0 if not loaded)."""
        return self._load_time

    @property
    def load_error(self) -> Exception | None:
        """Exception that occurred during loading, or None."""
        return self._error


# Global cache of lazy module proxies
_lazy_cache: dict[str, LazyModule] = {}
_cache_lock = threading.Lock()


def lazy_import(module_name: str) -> LazyModule:
    """Get or create a lazy-loaded module proxy.

    Thread-safe: multiple calls with the same name return the same proxy.

    Args:
        module_name: Full dotted module name

    Returns:
        LazyModule proxy that loads on first attribute access
    """
    with _cache_lock:
        if module_name not in _lazy_cache:
            _lazy_cache[module_name] = LazyModule(module_name)
        return _lazy_cache[module_name]


def get_lazy_stats() -> dict[str, dict[str, Any]]:
    """Get loading statistics for all lazy modules.

    Returns:
        Dict mapping module names to their status info
    """
    stats = {}
    with _cache_lock:
        for name, mod in _lazy_cache.items():
            stats[name] = {
                "loaded": mod.is_loaded,
                "load_time": mod.load_time,
                "error": str(mod.load_error) if mod.load_error else None,
            }
    return stats


def preload(*module_names: str) -> dict[str, bool]:
    """Eagerly load specified lazy modules.

    Useful for warming up modules in a background thread.

    Args:
        *module_names: Module names to preload

    Returns:
        Dict mapping module names to success status
    """
    results = {}
    for name in module_names:
        mod = lazy_import(name)
        try:
            mod._load()
            results[name] = True
        except ImportError:
            results[name] = False
    return results


def preload_in_background(
    *module_names: str,
    callback: Any | None = None,
    delay: float = 0.5,
) -> threading.Thread:
    """Start background preloading of heavy modules in a daemon thread.

    Allows the UI to start immediately while heavy modules (RAG, agents,
    pipelines, telemetry) load in parallel.

    Args:
        *module_names: Module names to preload
        callback: Optional callable(results: dict) called when done
        delay: Delay in seconds before starting preload

    Returns:
        The background thread (already started)
    """
    def _bg_preload():
        if delay > 0:
            time.sleep(delay)

        start = time.time()
        results = preload(*module_names)
        elapsed = time.time() - start

        loaded = sum(1 for v in results.values() if v)
        logger.info(
            f"Background preload complete: {loaded}/{len(results)} "
            f"modules in {elapsed:.3f}s"
        )

        if callback:
            try:
                callback(results)
            except Exception as e:
                logger.warning(f"Preload callback error: {e}")

    thread = threading.Thread(target=_bg_preload, daemon=True, name="lazy-preload")
    thread.start()
    return thread


# Modules considered "heavy" for lazy loading (S134 update)
HEAVY_MODULES = [
    "opti_oignon.rag_store",
    "opti_oignon.rag_hybrid_search",
    "opti_oignon.rag_external",
    "opti_oignon.rag_dashboard",
    "opti_oignon.coding_agent",
    "opti_oignon.coding_history",
    "opti_oignon.telemetry",
    "opti_oignon.telemetry_history",
    "opti_oignon.inference_profiler",
    "opti_oignon.performance_benchmark",
    "opti_oignon.benchmark_evaluator",
    "opti_oignon.benchmark_runner",
    "opti_oignon.benchmark_judge",
    "opti_oignon.benchmark_recommendations",
    "opti_oignon.benchmark_auto_trigger",
    "opti_oignon.benchmark_custom_profiles",
    "opti_oignon.fine_tune_export",
    "opti_oignon.fine_tune_tracker",
    "opti_oignon.plugin_index",
    "opti_oignon.plugin_installer",
    "opti_oignon.plugin_reviews",
    "opti_oignon.plugin_template",
    "opti_oignon.pipeline_manager",
    "opti_oignon.speculative_decoding",
    "opti_oignon.auto_tuner",
]


# Alias for compatibility with S134 prompt spec
get_lazy_module = lazy_import


class LazyAttr:
    """Proxy that lazily imports a module and delegates to a named attribute.

    Unlike LazyModule (which proxies the module itself), this proxies a
    specific attribute *within* the module. Useful when a module and its
    singleton share the same name (e.g. ``from mod import mod``).

    Example:
        # Defers import until first attribute access
        evaluator = LazyAttr("opti_oignon.benchmark_evaluator", "benchmark_evaluator")
        evaluator.available_profiles  # triggers import here
    """

    __slots__ = ("_module_name", "_attr_name", "_resolved", "_lock", "_error")

    def __init__(self, module_name: str, attr_name: str) -> None:
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_attr_name", attr_name)
        object.__setattr__(self, "_resolved", None)
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_error", None)

    def _resolve(self) -> Any:
        """Import the module and extract the target attribute."""
        resolved = object.__getattribute__(self, "_resolved")
        if resolved is not None:
            return resolved
        err = object.__getattribute__(self, "_error")
        if err is not None:
            raise err
        lock = object.__getattribute__(self, "_lock")
        with lock:
            # Double-check after acquiring the lock
            resolved = object.__getattribute__(self, "_resolved")
            if resolved is not None:
                return resolved
            mod_name = object.__getattribute__(self, "_module_name")
            attr_name = object.__getattribute__(self, "_attr_name")
            try:
                mod = importlib.import_module(mod_name)
                obj = getattr(mod, attr_name)
                object.__setattr__(self, "_resolved", obj)
                logger.debug("LazyAttr resolved: %s.%s", mod_name, attr_name)
                return obj
            except (ImportError, AttributeError) as exc:
                object.__setattr__(self, "_error", exc)
                logger.warning("LazyAttr failed: %s.%s: %s", mod_name, attr_name, exc)
                raise

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the resolved object."""
        return getattr(self._resolve(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow calling the proxy if the resolved attribute is callable."""
        return self._resolve()(*args, **kwargs)

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        resolved = object.__getattribute__(self, "_resolved")
        mod_name = object.__getattribute__(self, "_module_name")
        attr_name = object.__getattribute__(self, "_attr_name")
        if resolved is not None:
            return f"<LazyAttr({mod_name}.{attr_name}, resolved)>"
        return f"<LazyAttr({mod_name}.{attr_name}, deferred)>"


def get_startup_report() -> str:
    """Generate a loading report for all lazy modules.

    Returns:
        Formatted text report.
    """
    stats = get_lazy_stats()
    if not stats:
        return "No lazy modules registered"

    lines = ["Lazy Module Status:"]
    total_time = 0.0
    loaded_count = 0

    for name, info in sorted(stats.items()):
        status = "loaded" if info["loaded"] else ("FAILED" if info["error"] else "pending")
        t = info["load_time"]
        total_time += t
        if info["loaded"]:
            loaded_count += 1

        time_str = f" ({t:.3f}s)" if t > 0 else ""
        error_str = f" [{info['error']}]" if info["error"] else ""
        lines.append(f"  {name}: {status}{time_str}{error_str}")

    lines.append(f"  Total: {loaded_count}/{len(stats)} loaded, {total_time:.3f}s")
    return "\n".join(lines)
