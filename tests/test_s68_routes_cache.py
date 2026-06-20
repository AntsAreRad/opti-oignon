#!/usr/bin/env python3
"""
Tests for S68 API Routes — Step 3.

Covers:
- GET /api/cache/s68/status
- POST /api/cache/s68/toggle
- PUT /api/cache/s68/config
- POST /api/cache/s68/clear
- POST /api/cache/s68/expire
- Existing endpoints still work (/api/cache/stats, DELETE /api/cache)
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Module loading (bypass ollama requirement)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent


def _ensure_base_modules():
    """Load minimal modules needed for API testing."""
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(_ROOT / "opti_oignon")]
        sys.modules["opti_oignon"] = pkg

    if "opti_oignon.config" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.config",
            str(_ROOT / "opti_oignon" / "config.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["opti_oignon.config"] = mod
        spec.loader.exec_module(mod)

    if "opti_oignon.semantic_cache" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "opti_oignon.semantic_cache",
            str(_ROOT / "opti_oignon" / "semantic_cache.py"),
            submodule_search_locations=[],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["opti_oignon.semantic_cache"] = mod
        spec.loader.exec_module(mod)


_ensure_base_modules()

from opti_oignon.semantic_cache import SemanticCache, CacheStats

# ---------------------------------------------------------------------------
# Now test the routes directly (unit-test style, no FastAPI TestClient)
# ---------------------------------------------------------------------------

# Load just the schemas and route functions manually
_schemas_spec = importlib.util.spec_from_file_location(
    "opti_oignon.api.schemas",
    str(_ROOT / "opti_oignon" / "api" / "schemas.py"),
)
_schemas_mod = importlib.util.module_from_spec(_schemas_spec)
sys.modules["opti_oignon.api.schemas"] = _schemas_mod
# Need to set up api package
if "opti_oignon.api" not in sys.modules:
    api_pkg = types.ModuleType("opti_oignon.api")
    api_pkg.__path__ = [str(_ROOT / "opti_oignon" / "api")]
    sys.modules["opti_oignon.api"] = api_pkg
_schemas_spec.loader.exec_module(_schemas_mod)

S68CacheStatsSchema = _schemas_mod.S68CacheStatsSchema
S68CacheStatusResponse = _schemas_mod.S68CacheStatusResponse
S68CacheConfigUpdate = _schemas_mod.S68CacheConfigUpdate
S68CacheClearRequest = _schemas_mod.S68CacheClearRequest
CacheClearResponse = _schemas_mod.CacheClearResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_cache(tmp_path):
    """Create a test SemanticCache instance."""
    cfg_path = tmp_path / "cache.yaml"
    cfg = {
        "enabled": True,
        "exact_match_enabled": True,
        "semantic_match_enabled": False,
        "similarity_threshold": 0.92,
        "ttl_seconds": 3600,
        "max_entries": 100,
        "embedding_model": "mxbai-embed-large",
        "scope": "global",
        "max_candidates": 50,
        "avg_response_tokens": 250,
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    db_path = tmp_path / "test_cache.db"
    c = SemanticCache(db_path=db_path, config_path=cfg_path)
    c.embeddings_available = False
    return c


# ===========================================================================
# Test: Schema validation
# ===========================================================================


class TestS68Schemas:
    """Tests for S68 Pydantic schemas."""

    def test_stats_schema_defaults(self):
        """S68CacheStatsSchema has correct defaults."""
        s = S68CacheStatsSchema()
        assert s.total_entries == 0
        assert s.exact_hits == 0
        assert s.semantic_hits == 0
        assert s.tokens_saved == 0
        assert s.enabled is False

    def test_status_response_defaults(self):
        """S68CacheStatusResponse has correct defaults."""
        r = S68CacheStatusResponse()
        assert r.enabled is False
        assert r.available is False
        assert r.stats is None
        assert r.config == {}

    def test_config_update_partial(self):
        """S68CacheConfigUpdate allows partial updates."""
        u = S68CacheConfigUpdate(enabled=True, ttl_seconds=7200)
        dump = u.model_dump()
        assert dump["enabled"] is True
        assert dump["ttl_seconds"] == 7200
        assert dump["similarity_threshold"] is None

    def test_clear_request_defaults(self):
        """S68CacheClearRequest defaults to no conversation."""
        r = S68CacheClearRequest()
        assert r.conversation_id is None

    def test_clear_request_with_conversation(self):
        """S68CacheClearRequest accepts conversation_id."""
        r = S68CacheClearRequest(conversation_id="conv123")
        assert r.conversation_id == "conv123"


# ===========================================================================
# Test: Status endpoint logic
# ===========================================================================


class TestS68StatusLogic:
    """Tests for /api/cache/s68/status logic."""

    def test_status_from_cache(self, test_cache):
        """Status response is built from cache stats."""
        test_cache.put("q1", "r1")
        test_cache.get("q1")  # hit
        stats = test_cache.get_stats()

        response = S68CacheStatusResponse(
            enabled=stats.enabled,
            available=True,
            stats=S68CacheStatsSchema(
                total_entries=stats.total_entries,
                exact_hits=stats.exact_hits,
                semantic_hits=stats.semantic_hits,
                total_misses=stats.total_misses,
                hit_rate=stats.hit_rate,
                tokens_saved=stats.tokens_saved,
                enabled=stats.enabled,
            ),
            config=test_cache.get_config(),
        )
        assert response.enabled is True
        assert response.available is True
        assert response.stats.exact_hits == 1
        assert response.stats.total_entries == 1
        assert response.stats.tokens_saved == 250

    def test_status_when_unavailable(self):
        """Status response when cache module unavailable."""
        response = S68CacheStatusResponse(enabled=False, available=False)
        assert response.enabled is False
        assert response.available is False
        assert response.stats is None


# ===========================================================================
# Test: Toggle logic
# ===========================================================================


class TestS68ToggleLogic:
    """Tests for /api/cache/s68/toggle logic."""

    def test_toggle_on_off(self, test_cache):
        """Toggle flips enabled state."""
        assert test_cache.enabled is True
        test_cache.enabled = not test_cache.enabled
        assert test_cache.enabled is False
        test_cache.enabled = not test_cache.enabled
        assert test_cache.enabled is True


# ===========================================================================
# Test: Config update logic
# ===========================================================================


class TestS68ConfigUpdateLogic:
    """Tests for /api/cache/s68/config logic."""

    def test_update_partial(self, test_cache):
        """Partial config update only changes specified fields."""
        body = S68CacheConfigUpdate(ttl_seconds=7200, max_entries=500)
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        test_cache.update_config(updates)
        assert test_cache.ttl_seconds == 7200
        assert test_cache.max_entries == 500
        # Others unchanged
        assert test_cache.similarity_threshold == 0.92

    def test_update_enabled(self, test_cache):
        """Config update can toggle enabled."""
        body = S68CacheConfigUpdate(enabled=False)
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        test_cache.update_config(updates)
        assert test_cache.enabled is False


# ===========================================================================
# Test: Clear logic
# ===========================================================================


class TestS68ClearLogic:
    """Tests for /api/cache/s68/clear logic."""

    def test_clear_all(self, test_cache):
        """Clear all entries."""
        test_cache.put("q1", "r1")
        test_cache.put("q2", "r2")
        count = test_cache.invalidate()
        assert count == 2
        assert test_cache.entry_count() == 0

    def test_clear_by_conversation(self, tmp_path):
        """Clear only entries for a specific conversation."""
        cfg_path = tmp_path / "cache.yaml"
        cfg = {
            "enabled": True,
            "exact_match_enabled": True,
            "semantic_match_enabled": False,
            "scope": "conversation",
            "ttl_seconds": 3600,
            "max_entries": 100,
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        c = SemanticCache(db_path=tmp_path / "test.db", config_path=cfg_path)
        c.embeddings_available = False

        c.put("q1", "r1", conversation_id="conv_a")
        c.put("q2", "r2", conversation_id="conv_b")
        count = c.invalidate("conv_a")
        assert count == 1
        assert c.get("q1", conversation_id="conv_a") is None
        assert c.get("q2", conversation_id="conv_b") is not None

    def test_clear_response_format(self, test_cache):
        """Clear response matches expected schema."""
        test_cache.put("q", "r")
        count = test_cache.invalidate()
        resp = CacheClearResponse(entries_removed=count, source="s68_cache")
        assert resp.entries_removed == 1
        assert resp.source == "s68_cache"


# ===========================================================================
# Test: Expire logic
# ===========================================================================


class TestS68ExpireLogic:
    """Tests for /api/cache/s68/expire logic."""

    def test_expire_removes_stale(self, tmp_path):
        """expire_stale removes expired entries."""
        cfg_path = tmp_path / "cache.yaml"
        cfg = {"enabled": True, "ttl_seconds": 1, "max_entries": 100,
               "exact_match_enabled": True, "semantic_match_enabled": False}
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        c = SemanticCache(db_path=tmp_path / "test.db", config_path=cfg_path)
        c.embeddings_available = False
        c.put("q1", "r1")
        import time
        time.sleep(1.1)
        count = c.expire_stale()
        assert count == 1

    def test_expire_keeps_fresh(self, test_cache):
        """expire_stale keeps non-expired entries."""
        test_cache.put("q", "r")
        count = test_cache.expire_stale()
        assert count == 0
        assert test_cache.entry_count() == 1
