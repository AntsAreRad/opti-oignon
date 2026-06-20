#!/usr/bin/env python3
"""
Tests for PreCache -- S71 Step 2: Predictive pre-caching.

Covers:
- Config loading and defaults
- warm_common_queries with mock generator
- Cache hit skipping (already cached)
- Failed generation handling
- Disabled pre-cache
- PreCacheResult dataclass
- Custom query list from config
- Integration with mock semantic cache
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Direct module import
# ---------------------------------------------------------------------------

_mod_path = Path(__file__).resolve().parent.parent / "opti_oignon" / "pre_cache.py"
_spec = importlib.util.spec_from_file_location("pre_cache_mod", _mod_path)
_mod = importlib.util.module_from_spec(_spec)

# Ensure mock ollama and semantic_cache
sys.modules.setdefault("ollama", MagicMock())
if "opti_oignon.semantic_cache" not in sys.modules:
    _mock_sc_mod = types.ModuleType("opti_oignon.semantic_cache")
    _mock_sc_mod.semantic_cache = None
    sys.modules["opti_oignon.semantic_cache"] = _mock_sc_mod

_spec.loader.exec_module(_mod)

PreCache = _mod.PreCache
PreCacheResult = _mod.PreCacheResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Create a temporary pre_cache.yaml config."""
    cfg = {
        "enabled": True,
        "default_model": "test-model",
        "max_tokens": 128,
        "temperature": 0.2,
        "queries": [
            {"query": "Hello", "task_type": "general", "model": ""},
            {"query": "Write code", "task_type": "code_python", "model": ""},
        ],
    }
    if overrides:
        cfg.update(overrides)
    path = tmp_path / "pre_cache.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


class MockCache:
    """Mock semantic cache for testing."""

    def __init__(self):
        self._store = {}

    def get(self, query, model="", conversation_id=None):
        return self._store.get(query)

    def put(self, query, response, model="", metadata=None, conversation_id=None):
        self._store[query] = response
        return query


# ===========================================================================
# TEST CLASSES
# ===========================================================================


class TestPreCacheResult:
    """Tests for PreCacheResult dataclass."""

    def test_default_values(self):
        result = PreCacheResult()
        assert result.total == 0
        assert result.cached == 0
        assert result.skipped == 0
        assert result.failed == 0
        assert result.errors == []

    def test_to_dict(self):
        result = PreCacheResult(total=5, cached=3, skipped=1, failed=1, duration_ms=42.567)
        d = result.to_dict()
        assert d["total"] == 5
        assert d["duration_ms"] == 42.57


class TestPreCacheConfig:
    """Tests for config loading."""

    def test_default_queries_when_no_config(self, tmp_path):
        path = tmp_path / "nonexistent.yaml"
        pc = PreCache(config_path=path, cache=MockCache())
        assert len(pc.queries) > 0  # should use defaults

    def test_custom_queries_from_config(self, tmp_path):
        path = _make_config(tmp_path)
        pc = PreCache(config_path=path, cache=MockCache())
        assert len(pc.queries) == 2
        assert pc.queries[0]["query"] == "Hello"

    def test_get_config(self, tmp_path):
        path = _make_config(tmp_path)
        pc = PreCache(config_path=path, cache=MockCache())
        cfg = pc.get_config()
        assert cfg["enabled"] is True
        assert cfg["query_count"] == 2


class TestWarmCommonQueries:
    """Tests for warm_common_queries()."""

    def test_warm_with_generator(self, tmp_path):
        path = _make_config(tmp_path)
        cache = MockCache()
        pc = PreCache(config_path=path, cache=cache)

        def gen(query, model, task_type):
            return f"response to {query}"

        result = pc.warm_common_queries(generate_fn=gen)
        assert result.total == 2
        assert result.cached == 2
        assert result.skipped == 0
        assert result.failed == 0
        assert "Hello" in cache._store
        assert cache._store["Hello"] == "response to Hello"

    def test_warm_skips_cached(self, tmp_path):
        path = _make_config(tmp_path)
        cache = MockCache()
        cache._store["Hello"] = "already cached"
        pc = PreCache(config_path=path, cache=cache)

        def gen(query, model, task_type):
            return f"response to {query}"

        result = pc.warm_common_queries(generate_fn=gen)
        assert result.skipped == 1
        assert result.cached == 1
        assert cache._store["Hello"] == "already cached"  # not overwritten

    def test_warm_handles_generator_failure(self, tmp_path):
        path = _make_config(tmp_path)
        cache = MockCache()
        pc = PreCache(config_path=path, cache=cache)

        def gen(query, model, task_type):
            if query == "Hello":
                raise RuntimeError("model error")
            return "ok"

        result = pc.warm_common_queries(generate_fn=gen)
        assert result.failed == 1
        assert result.cached == 1
        assert len(result.errors) == 1

    def test_warm_disabled(self, tmp_path):
        path = _make_config(tmp_path, {"enabled": False})
        cache = MockCache()
        pc = PreCache(config_path=path, cache=cache)

        result = pc.warm_common_queries(generate_fn=lambda q, m, t: "ok")
        assert result.total == 2
        assert result.cached == 0
        assert result.skipped == 0

    def test_warm_no_cache_instance(self, tmp_path):
        path = _make_config(tmp_path)
        pc = PreCache(config_path=path, cache=None)

        def gen(query, model, task_type):
            return "response"

        result = pc.warm_common_queries(generate_fn=gen)
        # No cache to check or store -> all fail
        assert result.failed == 2

    def test_warm_records_duration(self, tmp_path):
        path = _make_config(tmp_path)
        cache = MockCache()
        pc = PreCache(config_path=path, cache=cache)

        result = pc.warm_common_queries(generate_fn=lambda q, m, t: "ok")
        assert result.duration_ms >= 0

    def test_last_result_property(self, tmp_path):
        path = _make_config(tmp_path)
        cache = MockCache()
        pc = PreCache(config_path=path, cache=cache)
        assert pc.last_result is None

        pc.warm_common_queries(generate_fn=lambda q, m, t: "ok")
        assert pc.last_result is not None
        assert pc.last_result.total == 2

    def test_warm_empty_query_skipped(self, tmp_path):
        cfg_path = tmp_path / "pre_cache.yaml"
        cfg = {
            "enabled": True,
            "queries": [
                {"query": "", "task_type": "general"},
                {"query": "valid", "task_type": "general"},
            ],
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        cache = MockCache()
        pc = PreCache(config_path=cfg_path, cache=cache)

        result = pc.warm_common_queries(generate_fn=lambda q, m, t: "ok")
        # Empty query is skipped (not counted as cached or failed)
        assert result.cached == 1
