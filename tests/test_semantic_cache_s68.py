#!/usr/bin/env python3
"""
Tests for SemanticCache — S68 Step 1: Core functionality.

Covers:
- YAML config loading and defaults
- Exact hash get/put (Tier 1)
- Semantic similarity get/put (Tier 2) with mock embeddings
- TTL expiry
- LRU eviction
- Conversation scope
- Token savings tracking
- Stats reporting
- Invalidation
- Legacy S23 API backward compatibility
"""

import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Direct module import (bypass __init__.py which requires ollama)
# ---------------------------------------------------------------------------

_MODULE_PATH = Path(__file__).resolve().parent.parent / "opti_oignon" / "semantic_cache.py"


def _load_module():
    """Load semantic_cache.py directly from file path."""
    # Setup parent package
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = [str(_MODULE_PATH.parent)]
    sys.modules["opti_oignon"] = pkg

    # Load config.py
    config_spec = importlib.util.spec_from_file_location(
        "opti_oignon.config",
        str(_MODULE_PATH.parent / "config.py"),
    )
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules["opti_oignon.config"] = config_module
    config_spec.loader.exec_module(config_module)

    # Load semantic_cache.py
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.semantic_cache",
        str(_MODULE_PATH),
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.semantic_cache"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
SemanticCache = _mod.SemanticCache
CacheEntry = _mod.CacheEntry
CacheStats = _mod.CacheStats
SemanticMatch = _mod.SemanticMatch
SemanticCacheStats = _mod.SemanticCacheStats
cosine_similarity = _mod.cosine_similarity
_make_query_hash = _mod._make_query_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary DB path."""
    return tmp_path / "test_cache.db"


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary cache.yaml with test defaults."""
    cfg = {
        "enabled": True,
        "exact_match_enabled": True,
        "semantic_match_enabled": True,
        "similarity_threshold": 0.92,
        "ttl_seconds": 3600,
        "max_entries": 100,
        "embedding_model": "mxbai-embed-large",
        "scope": "global",
        "max_candidates": 50,
        "avg_response_tokens": 250,
    }
    path = tmp_path / "cache.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


@pytest.fixture
def cache(tmp_db, tmp_config):
    """Return a SemanticCache with test config, embeddings disabled."""
    c = SemanticCache(db_path=tmp_db, config_path=tmp_config)
    c.embeddings_available = False  # No Ollama in test env
    return c


@pytest.fixture
def cache_with_embeddings(tmp_db, tmp_config):
    """Return a SemanticCache with mock embedding support."""
    c = SemanticCache(db_path=tmp_db, config_path=tmp_config)
    c.embeddings_available = True
    return c


def _fake_embedding(text: str) -> list[float]:
    """Deterministic mock embedding: normalized hash-based vector."""
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()
    raw = [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm > 0 else raw


# ===========================================================================
# Test: cosine_similarity utility
# ===========================================================================


class TestCosineSimilarity:
    """Tests for the cosine_similarity utility function."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) + 1.0) < 1e-6

    def test_empty_vectors(self):
        """Empty vectors return 0.0."""
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_length(self):
        """Mismatched lengths return 0.0."""
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        """Zero vector returns 0.0."""
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ===========================================================================
# Test: query hash
# ===========================================================================


class TestQueryHash:
    """Tests for _make_query_hash."""

    def test_deterministic(self):
        """Same input produces same hash."""
        h1 = _make_query_hash("hello world")
        h2 = _make_query_hash("hello world")
        assert h1 == h2

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        h1 = _make_query_hash("hello")
        h2 = _make_query_hash("  hello  ")
        assert h1 == h2

    def test_different_input_different_hash(self):
        """Different inputs produce different hashes."""
        h1 = _make_query_hash("hello")
        h2 = _make_query_hash("world")
        assert h1 != h2


# ===========================================================================
# Test: Config loading
# ===========================================================================


class TestConfigLoading:
    """Tests for YAML config loading and defaults."""

    def test_loads_from_yaml(self, tmp_db, tmp_config):
        """Config values loaded from YAML file."""
        c = SemanticCache(db_path=tmp_db, config_path=tmp_config)
        cfg = c.get_config()
        assert cfg["similarity_threshold"] == 0.92
        assert cfg["max_entries"] == 100
        assert cfg["scope"] == "global"

    def test_defaults_when_no_yaml(self, tmp_db, tmp_path):
        """Default values used when YAML file does not exist."""
        missing = tmp_path / "nonexistent.yaml"
        c = SemanticCache(db_path=tmp_db, config_path=missing)
        cfg = c.get_config()
        assert cfg["enabled"] is False
        assert cfg["similarity_threshold"] == 0.92
        assert cfg["max_entries"] == 1000

    def test_explicit_overrides(self, tmp_db, tmp_config):
        """Constructor kwargs override YAML values."""
        c = SemanticCache(
            db_path=tmp_db,
            config_path=tmp_config,
            similarity_threshold=0.80,
            max_entries=50,
        )
        assert c.similarity_threshold == 0.80
        assert c.max_entries == 50

    def test_update_config(self, cache, tmp_config):
        """update_config persists changes to YAML."""
        cache.update_config({"ttl_seconds": 7200, "max_entries": 500})
        assert cache.ttl_seconds == 7200
        assert cache.max_entries == 500
        # Verify persisted
        with open(tmp_config) as f:
            saved = yaml.safe_load(f)
        assert saved["ttl_seconds"] == 7200

    def test_reload_config(self, cache, tmp_config):
        """reload_config re-reads from YAML."""
        with open(tmp_config) as f:
            cfg = yaml.safe_load(f)
        cfg["max_entries"] = 999
        with open(tmp_config, "w") as f:
            yaml.safe_dump(cfg, f)
        cache.reload_config()
        assert cache.max_entries == 999


# ===========================================================================
# Test: Exact match (Tier 1)
# ===========================================================================


class TestExactMatch:
    """Tests for exact hash get/put."""

    def test_put_and_get_exact(self, cache):
        """Put a query, get it back by exact match."""
        cache.put("What is Python?", "A programming language.", model="qwen3:32b")
        entry = cache.get("What is Python?")
        assert entry is not None
        assert entry.response == "A programming language."
        assert entry.match_type == "exact"
        assert entry.similarity == 1.0
        assert entry.model == "qwen3:32b"

    def test_miss_when_not_stored(self, cache):
        """Returns None for queries not in cache."""
        entry = cache.get("Unknown query")
        assert entry is None

    def test_disabled_returns_none(self, cache):
        """Returns None when cache is disabled."""
        cache.put("test", "response")
        cache.enabled = False
        assert cache.get("test") is None

    def test_put_returns_empty_when_disabled(self, cache):
        """put() returns empty string when disabled."""
        cache.enabled = False
        assert cache.put("test", "response") == ""

    def test_hit_count_increments(self, cache):
        """hit_count increases on repeated access."""
        cache.put("q1", "r1")
        e1 = cache.get("q1")
        assert e1.hit_count == 1
        e2 = cache.get("q1")
        assert e2.hit_count == 2

    def test_metadata_round_trip(self, cache):
        """Metadata dict is preserved through put/get."""
        cache.put("q", "r", metadata={"task": "code", "lang": "python"})
        entry = cache.get("q")
        assert entry.metadata["task"] == "code"
        assert entry.metadata["lang"] == "python"

    def test_model_filter(self, cache):
        """Model filter restricts matches."""
        cache.put("q", "r_a", model="model_a")
        # No match for different model
        entry = cache.get("q", model="model_b")
        assert entry is None
        # Match for correct model
        entry = cache.get("q", model="model_a")
        assert entry is not None
        assert entry.response == "r_a"

    def test_overwrite_existing(self, cache):
        """Putting same query overwrites the response."""
        cache.put("q", "old_response")
        cache.put("q", "new_response")
        entry = cache.get("q")
        assert entry.response == "new_response"
        assert entry.hit_count == 1  # First get after overwrite


# ===========================================================================
# Test: TTL expiry
# ===========================================================================


class TestTTLExpiry:
    """Tests for TTL-based entry expiration."""

    def test_expired_entry_returns_none(self, tmp_db, tmp_config):
        """Entries past TTL are treated as miss and deleted."""
        c = SemanticCache(db_path=tmp_db, config_path=tmp_config, ttl_seconds=1)
        c.embeddings_available = False
        c.put("q", "r")
        assert c.get("q") is not None
        time.sleep(1.1)
        assert c.get("q") is None

    def test_expire_stale_removes_old(self, tmp_db, tmp_config):
        """expire_stale() removes entries past TTL."""
        c = SemanticCache(db_path=tmp_db, config_path=tmp_config, ttl_seconds=1)
        c.embeddings_available = False
        c.put("q1", "r1")
        c.put("q2", "r2")
        time.sleep(1.1)
        removed = c.expire_stale()
        assert removed == 2
        assert c.entry_count() == 0

    def test_non_expired_entries_kept(self, cache):
        """Non-expired entries survive expire_stale()."""
        cache.put("q", "r")
        removed = cache.expire_stale()
        assert removed == 0
        assert cache.entry_count() == 1


# ===========================================================================
# Test: LRU eviction
# ===========================================================================


class TestLRUEviction:
    """Tests for max_entries LRU eviction."""

    def test_evicts_oldest_when_full(self, tmp_db, tmp_config):
        """Oldest entries evicted when max_entries exceeded."""
        c = SemanticCache(db_path=tmp_db, config_path=tmp_config, max_entries=3)
        c.embeddings_available = False
        c.put("q1", "r1")
        time.sleep(0.01)
        c.put("q2", "r2")
        time.sleep(0.01)
        c.put("q3", "r3")
        time.sleep(0.01)
        # This should evict q1 (oldest by last_accessed)
        c.put("q4", "r4")
        assert c.entry_count() == 3
        assert c.get("q1") is None
        assert c.get("q4") is not None

    def test_no_eviction_below_limit(self, cache):
        """No eviction when under max_entries."""
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        assert cache.entry_count() == 2


# ===========================================================================
# Test: Conversation scope
# ===========================================================================


class TestConversationScope:
    """Tests for conversation-scoped caching."""

    def test_global_scope_ignores_conversation_id(self, cache):
        """In global scope, conversation_id is ignored."""
        cache.put("q", "r", conversation_id="conv1")
        entry = cache.get("q", conversation_id="conv2")
        assert entry is not None  # Found because scope is global

    def test_conversation_scope_isolates(self, tmp_db, tmp_config):
        """In conversation scope, entries are isolated per conversation."""
        with open(tmp_config) as f:
            cfg = yaml.safe_load(f)
        cfg["scope"] = "conversation"
        with open(tmp_config, "w") as f:
            yaml.safe_dump(cfg, f)

        c = SemanticCache(db_path=tmp_db, config_path=tmp_config)
        c.embeddings_available = False
        c.put("q", "r_conv1", conversation_id="conv1")
        c.put("q", "r_conv2", conversation_id="conv2")

        e1 = c.get("q", conversation_id="conv1")
        e2 = c.get("q", conversation_id="conv2")
        assert e1.response == "r_conv1"
        assert e2.response == "r_conv2"

    def test_invalidate_by_conversation(self, tmp_db, tmp_config):
        """invalidate(conversation_id) only removes that conversation."""
        with open(tmp_config) as f:
            cfg = yaml.safe_load(f)
        cfg["scope"] = "conversation"
        with open(tmp_config, "w") as f:
            yaml.safe_dump(cfg, f)

        c = SemanticCache(db_path=tmp_db, config_path=tmp_config)
        c.embeddings_available = False
        c.put("q1", "r1", conversation_id="conv_a")
        c.put("q2", "r2", conversation_id="conv_b")
        removed = c.invalidate("conv_a")
        assert removed == 1
        assert c.get("q1", conversation_id="conv_a") is None
        assert c.get("q2", conversation_id="conv_b") is not None


# ===========================================================================
# Test: Semantic matching (Tier 2) with mock embeddings
# ===========================================================================


class TestSemanticMatch:
    """Tests for semantic similarity matching with mocked embeddings."""

    def test_semantic_hit_with_similar_query(self, cache_with_embeddings):
        """Semantic match returns entry for similar but not identical query."""
        c = cache_with_embeddings
        # Use a low threshold for testing
        c.update_config({"similarity_threshold": 0.5})

        # Manually insert an entry with a known embedding
        emb = [1.0, 0.0, 0.0, 0.0]
        norm = math.sqrt(sum(x * x for x in emb))
        emb = [x / norm for x in emb]
        blob = json.dumps(emb).encode("utf-8")

        conn = c._get_connection()
        now = time.time()
        conn.execute(
            """INSERT INTO cache_entries
               (query_hash, query_text, response, model, conversation_id,
                embedding, embedding_dim, metadata, created_at, last_accessed, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            ("hash_original", "What is Python?", "A language.", "m1", "",
             blob, len(emb), "{}", now, now),
        )
        conn.commit()
        conn.close()

        # Mock _get_embedding to return a very similar vector
        similar_emb = [0.99, 0.01, 0.0, 0.0]
        norm2 = math.sqrt(sum(x * x for x in similar_emb))
        similar_emb = [x / norm2 for x in similar_emb]

        with patch.object(_mod, "_get_embedding", return_value=similar_emb):
            entry = c.get("What is Python programming?")

        assert entry is not None
        assert entry.match_type == "semantic"
        assert entry.similarity > 0.5
        assert entry.response == "A language."

    def test_semantic_miss_below_threshold(self, cache_with_embeddings):
        """No match when similarity is below threshold."""
        c = cache_with_embeddings
        c.update_config({"similarity_threshold": 0.99})

        emb = [1.0, 0.0, 0.0, 0.0]
        blob = json.dumps(emb).encode("utf-8")

        conn = c._get_connection()
        now = time.time()
        conn.execute(
            """INSERT INTO cache_entries
               (query_hash, query_text, response, model, conversation_id,
                embedding, embedding_dim, metadata, created_at, last_accessed, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            ("hash_x", "query x", "response x", "m1", "",
             blob, len(emb), "{}", now, now),
        )
        conn.commit()
        conn.close()

        # Return an orthogonal vector
        with patch.object(_mod, "_get_embedding", return_value=[0.0, 1.0, 0.0, 0.0]):
            entry = c.get("completely different query")

        assert entry is None

    def test_exact_takes_priority_over_semantic(self, cache_with_embeddings):
        """Exact match is returned even when semantic match also exists."""
        c = cache_with_embeddings

        with patch.object(_mod, "_get_embedding", return_value=[1.0, 0.0]):
            c.put("exact query", "exact response", model="m1")

        entry = c.get("exact query")
        assert entry is not None
        assert entry.match_type == "exact"


# ===========================================================================
# Test: Stats
# ===========================================================================


class TestCacheStats:
    """Tests for get_stats()."""

    def test_initial_stats(self, cache):
        """Stats start at zero."""
        stats = cache.get_stats()
        assert stats.total_entries == 0
        assert stats.exact_hits == 0
        assert stats.semantic_hits == 0
        assert stats.total_misses == 0
        assert stats.hit_rate == 0.0
        assert stats.tokens_saved == 0
        assert stats.enabled is True  # Test config has enabled=True

    def test_stats_after_hits(self, cache):
        """Stats reflect cache activity."""
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        cache.get("q1")  # hit
        cache.get("q1")  # hit
        cache.get("unknown")  # miss
        stats = cache.get_stats()
        assert stats.total_entries == 2
        assert stats.exact_hits == 2
        assert stats.total_misses == 1
        assert stats.tokens_saved == 500  # 2 * 250
        assert abs(stats.hit_rate - 2 / 3) < 0.01

    def test_stats_config_reflected(self, cache):
        """Stats include config values."""
        stats = cache.get_stats()
        assert stats.max_entries == 100
        assert stats.ttl_seconds == 3600
        assert stats.similarity_threshold == 0.92
        assert stats.scope == "global"


# ===========================================================================
# Test: Invalidate and clear
# ===========================================================================


class TestInvalidation:
    """Tests for invalidate() and clear()."""

    def test_invalidate_all(self, cache):
        """invalidate() without argument clears everything."""
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        removed = cache.invalidate()
        assert removed == 2
        assert cache.entry_count() == 0

    def test_clear_resets_counters(self, cache):
        """clear() resets session counters."""
        cache.put("q", "r")
        cache.get("q")
        cache.clear()
        stats = cache.get_stats()
        assert stats.exact_hits == 0
        assert stats.tokens_saved == 0

    def test_entry_count(self, cache):
        """entry_count() returns correct total."""
        assert cache.entry_count() == 0
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        assert cache.entry_count() == 2


# ===========================================================================
# Test: Properties
# ===========================================================================


class TestProperties:
    """Tests for config property accessors."""

    def test_enabled_toggle(self, cache):
        """enabled property can be toggled."""
        assert cache.enabled is True
        cache.enabled = False
        assert cache.enabled is False

    def test_similarity_threshold_clamped(self, cache):
        """similarity_threshold is clamped to 0.5-0.99."""
        cache.similarity_threshold = 0.1
        assert cache.similarity_threshold == 0.5
        cache.similarity_threshold = 1.5
        assert cache.similarity_threshold == 0.99

    def test_scope_property(self, cache):
        """scope returns config value."""
        assert cache.scope == "global"

    def test_ttl_property(self, cache):
        """ttl_seconds returns config value."""
        assert cache.ttl_seconds == 3600

    def test_max_entries_property(self, cache):
        """max_entries returns config value."""
        assert cache.max_entries == 100


# ===========================================================================
# Test: Data classes
# ===========================================================================


class TestDataClasses:
    """Tests for CacheEntry and CacheStats dataclasses."""

    def test_cache_entry_defaults(self):
        """CacheEntry has correct defaults."""
        e = CacheEntry()
        assert e.query_hash == ""
        assert e.response == ""
        assert e.similarity == 1.0
        assert e.match_type == "exact"
        assert e.metadata == {}

    def test_cache_stats_defaults(self):
        """CacheStats has correct defaults."""
        s = CacheStats()
        assert s.total_entries == 0
        assert s.enabled is False

    def test_semantic_match_legacy(self):
        """SemanticMatch (S23 compat) has expected fields."""
        m = SemanticMatch(cache_key="k", similarity=0.95, model="m", query_text="q")
        assert m.cache_key == "k"
        assert m.similarity == 0.95

    def test_semantic_cache_stats_legacy(self):
        """SemanticCacheStats (S23 compat) has expected fields."""
        s = SemanticCacheStats()
        assert s.total_embeddings == 0
        assert s.embedding_model == "mxbai-embed-large"
