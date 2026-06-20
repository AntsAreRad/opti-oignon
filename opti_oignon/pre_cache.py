#!/usr/bin/env python3
"""
PRE-CACHE -- Predictive Pre-Caching of Common Queries (S71)
=============================================================

Warms the S68 semantic cache with responses to commonly requested
query patterns. Pre-cache runs when Ollama is online and idle,
filling the cache proactively so that offline lookups have more
chance of a hit.

The set of queries to pre-cache is YAML-configurable and grouped
by task type.

Author: Leon
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Conditional imports
try:
    import ollama as _ollama_module
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    _ollama_module = None

try:
    from opti_oignon.semantic_cache import semantic_cache as _semantic_cache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False
    _semantic_cache = None


# =============================================================================
# CONSTANTS
# =============================================================================

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "pre_cache.yaml"

# Default queries to pre-cache if no config file exists
_DEFAULT_QUERIES = [
    {"query": "Hello", "task_type": "general", "model": ""},
    {"query": "Hi, how can you help me?", "task_type": "general", "model": ""},
    {"query": "What can you do?", "task_type": "general", "model": ""},
    {"query": "Summarize this text", "task_type": "general", "model": ""},
    {"query": "Explain this code", "task_type": "code_python", "model": ""},
    {"query": "Fix this bug", "task_type": "code_python", "model": ""},
    {"query": "Write a Python function", "task_type": "code_python", "model": ""},
    {"query": "Write an R script", "task_type": "code_r", "model": ""},
    {"query": "Analyze this data", "task_type": "data_analysis", "model": ""},
    {"query": "Help me plan", "task_type": "planning", "model": ""},
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PreCacheResult:
    """Result of a pre-cache warming run.

    Attributes:
        total: Number of queries attempted.
        cached: Number successfully cached.
        skipped: Number already in cache (skipped).
        failed: Number that failed to generate.
        duration_ms: Total wall-clock time in milliseconds.
        errors: List of error messages from failed queries.
    """
    total: int = 0
    cached: int = 0
    skipped: int = 0
    failed: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "total": self.total,
            "cached": self.cached,
            "skipped": self.skipped,
            "failed": self.failed,
            "duration_ms": round(self.duration_ms, 2),
            "errors": self.errors,
        }


# =============================================================================
# PRE-CACHE
# =============================================================================

class PreCache:
    """Predictive pre-caching for common query patterns.

    Generates LLM responses for a configured list of common queries
    and stores them in the S68 semantic cache proactively.

    Args:
        config_path: Path to YAML config. None uses default.
        cache: Semantic cache instance to populate. None uses the
            module-level singleton.
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        cache: Any = None,
    ):
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._cache = cache if cache is not None else _semantic_cache
        self._config: dict[str, Any] = {}
        self._queries: list[dict] = []
        self._last_result: PreCacheResult | None = None
        self._load_config()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _load_config(self) -> None:
        """Load pre-cache configuration from YAML."""
        defaults = {
            "enabled": True,
            "default_model": "",
            "max_tokens": 256,
            "temperature": 0.3,
        }
        queries = list(_DEFAULT_QUERIES)

        try:
            if self._config_path.exists():
                with open(self._config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                for k in defaults:
                    if k in loaded:
                        defaults[k] = loaded[k]
                if "queries" in loaded and isinstance(loaded["queries"], list):
                    queries = loaded["queries"]
        except Exception as e:
            logger.warning("Failed to load pre-cache config: %s", e)

        self._config = defaults
        self._queries = queries

    def get_config(self) -> dict:
        """Return config with current query list."""
        return {
            **self._config,
            "queries": list(self._queries),
            "query_count": len(self._queries),
        }

    @property
    def queries(self) -> list[dict]:
        """Return the list of pre-cache query definitions."""
        return list(self._queries)

    @property
    def last_result(self) -> PreCacheResult | None:
        """Return the result of the last warm_common_queries() run."""
        return self._last_result

    # -----------------------------------------------------------------
    # Pre-caching
    # -----------------------------------------------------------------

    def warm_common_queries(
        self,
        generate_fn: Any = None,
    ) -> PreCacheResult:
        """Pre-generate and cache responses for common queries.

        For each configured query:
        1. Check if already in cache -> skip
        2. Generate response via generate_fn or ollama.chat()
        3. Store in semantic cache

        Args:
            generate_fn: Optional callable(query, model, task_type) -> str.
                If None, uses ollama.chat() directly with default model.

        Returns:
            PreCacheResult with counts and timing.
        """
        result = PreCacheResult(total=len(self._queries))
        start_time = time.time()

        if not self._config.get("enabled", True):
            result.duration_ms = (time.time() - start_time) * 1000
            self._last_result = result
            return result

        for q_def in self._queries:
            query = q_def.get("query", "")
            task_type = q_def.get("task_type", "general")
            model = q_def.get("model", "") or self._config.get("default_model", "")

            if not query:
                continue

            # Check if already cached
            if self._cache is not None:
                try:
                    existing = self._cache.get(query, model=model)
                    if existing is not None:
                        result.skipped += 1
                        continue
                except Exception:
                    pass

            # Generate response
            try:
                if generate_fn is not None:
                    response = generate_fn(query, model, task_type)
                elif OLLAMA_AVAILABLE and model:
                    resp = _ollama_module.chat(
                        model=model,
                        messages=[{"role": "user", "content": query}],
                        options={
                            "num_predict": self._config.get("max_tokens", 256),
                            "temperature": self._config.get("temperature", 0.3),
                        },
                    )
                    # S193 PCH-01: handle both dict-form and object-form
                    # client responses. The previous dict-first .get() raised
                    # AttributeError on object responses BEFORE the object
                    # fallback below it could run (dead code) -- the
                    # BMK-01/MEM-06 idiom applied here.
                    if isinstance(resp, dict):
                        response = (resp.get("message") or {}).get("content", "") or ""
                    else:
                        _msg = getattr(resp, "message", None)
                        response = (getattr(_msg, "content", "") or "") if _msg is not None else ""
                else:
                    result.failed += 1
                    result.errors.append(f"No generator available for: {query[:50]}")
                    continue

                # Store in cache
                if self._cache is not None and response:
                    self._cache.put(
                        query=query,
                        response=response,
                        model=model,
                        metadata={"source": "pre_cache", "task_type": task_type},
                    )
                    result.cached += 1
                else:
                    result.failed += 1
                    result.errors.append(f"Empty response or no cache for: {query[:50]}")

            except Exception as e:
                result.failed += 1
                result.errors.append(f"{query[:50]}: {e}")
                logger.debug("Pre-cache generation failed for '%s': %s", query[:50], e)

        result.duration_ms = (time.time() - start_time) * 1000
        self._last_result = result
        logger.info(
            "Pre-cache warming done: %d cached, %d skipped, %d failed (%.0fms)",
            result.cached, result.skipped, result.failed, result.duration_ms,
        )
        return result


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

try:
    pre_cache = PreCache()
except Exception as e:
    logger.warning("Failed to create PreCache singleton: %s", e)
    pre_cache = None  # type: ignore[assignment]
