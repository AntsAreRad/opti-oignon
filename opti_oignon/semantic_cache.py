#!/usr/bin/env python3
"""
SEMANTIC CACHE — OPTI-OIGNON v1.7.0 (Session 68)
==================================================

Persistent dual-threshold cache that avoids redundant LLM calls:

- Tier 1 (Exact): SHA-256 hash of query text for identical matches.
- Tier 2 (Semantic): Cosine similarity of embeddings for semantically
  equivalent queries above a configurable threshold.

SQLite-backed with TTL expiry, LRU eviction, per-conversation or
global scope, and token-savings tracking.

Configuration loaded from config/cache.yaml with runtime overrides.

Backward-compatible: retains S23 methods (get_with_fallback,
put_with_embedding, store_embedding, find_similar) alongside the new
simplified get/put/invalidate/get_stats API.

Usage:
    from opti_oignon.semantic_cache import semantic_cache

    entry = semantic_cache.get("What is Python?")
    if entry is None:
        # ... call LLM ...
        semantic_cache.put("What is Python?", response_text, model="qwen3:32b")

Author: Leon
"""

import hashlib
import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import DATA_DIR

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    logger.warning(
        "db_utils unavailable: semantic_cache falling back to PLAINTEXT sqlite3. "
        "Cached queries/responses and embeddings are NOT encrypted at rest."
    )
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# =============================================================================
# PATHS
# =============================================================================

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "cache.yaml"
_DEFAULT_DB_PATH = DATA_DIR / "semantic_cache.db"

# =============================================================================
# DEFAULTS (fallback when YAML missing)
# =============================================================================

DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_TTL_SECONDS = 3600
DEFAULT_MAX_ENTRIES = 1000
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_SCOPE = "conversation"
DEFAULT_AVG_RESPONSE_TOKENS = 250

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CacheEntry:
    """Single cache entry returned by get().

    Attributes:
        query_hash: SHA-256 hash of the query text.
        query_text: Original query string.
        response: Cached LLM response.
        model: LLM model that generated the response.
        conversation_id: Conversation scope (empty for global).
        similarity: 1.0 for exact match, <1.0 for semantic match.
        match_type: 'exact', 'semantic', or 'none'.
        created_at: Unix timestamp of entry creation.
        last_accessed: Unix timestamp of last access.
        hit_count: Number of times this entry has been served.
        metadata: Arbitrary JSON metadata dict.
    """
    query_hash: str = ""
    query_text: str = ""
    response: str = ""
    model: str = ""
    conversation_id: str = ""
    similarity: float = 1.0
    match_type: str = "exact"
    created_at: float = 0.0
    last_accessed: float = 0.0
    hit_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache statistics returned by get_stats().

    Attributes:
        total_entries: Number of live (non-expired) entries.
        exact_hits: Exact hash hits this session.
        semantic_hits: Semantic similarity hits this session.
        total_misses: Total cache misses this session.
        hit_rate: Overall hit rate (0.0 - 1.0).
        exact_hit_rate: Exact-only hit rate.
        semantic_hit_rate: Semantic-only hit rate.
        tokens_saved: Estimated tokens not regenerated thanks to cache.
        size_bytes: Approximate DB size in bytes.
        max_entries: Configured max entries.
        ttl_seconds: Configured TTL.
        similarity_threshold: Configured similarity threshold.
        embedding_model: Configured embedding model name.
        scope: 'global' or 'conversation'.
        enabled: Whether the cache is currently enabled.
        embeddings_available: Whether the embedding model is reachable.
    """
    total_entries: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    exact_hit_rate: float = 0.0
    semantic_hit_rate: float = 0.0
    tokens_saved: int = 0
    size_bytes: int = 0
    max_entries: int = DEFAULT_MAX_ENTRIES
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    scope: str = DEFAULT_SCOPE
    enabled: bool = False
    embeddings_available: bool = False


# Legacy dataclass kept for backward compatibility (S23)
@dataclass
class SemanticMatch:
    """Result of a semantic similarity search (S23 compat).

    Attributes:
        cache_key: Key of the matching cache entry.
        similarity: Cosine similarity score (0.0 - 1.0).
        model: LLM model name.
        query_text: Original query text of the match.
    """
    cache_key: str
    similarity: float
    model: str
    query_text: str


# Legacy dataclass kept for backward compatibility (S23)
@dataclass
class SemanticCacheStats:
    """Legacy stats for S23 compatibility.

    Attributes:
        total_embeddings: Total stored embeddings.
        semantic_hits: Semantic hit count.
        semantic_misses: Semantic miss count.
        avg_similarity: Average similarity of hits.
        embedding_model: Embedding model name.
        threshold: Similarity threshold.
    """
    total_embeddings: int = 0
    semantic_hits: int = 0
    semantic_misses: int = 0
    avg_similarity: float = 0.0
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Cosine similarity between -1.0 and 1.0.
    """
    if len(vec_a) != len(vec_b) or len(vec_a) == 0:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _make_query_hash(query_text: str) -> str:
    """Generate SHA-256 hash for a query string.

    Args:
        query_text: Raw query text.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(query_text.strip().encode("utf-8")).hexdigest()


def _is_bulbe() -> bool:
    """True when the security mode is Bulbe.

    Fail-safe to False when the mode is undeterminable: the per-conversation
    default scope (TC-01) already prevents cross-conversation bleed, so the
    Bulbe override only tightens an explicitly-global config.
    """
    try:
        from opti_oignon.security_mode import is_bulbe

        return bool(is_bulbe())
    except Exception:
        return False


def _get_embedding(
    text: str, model: str = DEFAULT_EMBEDDING_MODEL
) -> list[float] | None:
    """Generate an embedding via ollama.embed().

    Args:
        text: Text to encode.
        model: Embedding model name.

    Returns:
        Embedding vector or None on failure.
    """
    try:
        import ollama

        result = ollama.embed(model=model, input=text)
        if result and "embeddings" in result and len(result["embeddings"]) > 0:
            return result["embeddings"][0]
        return None
    except Exception as e:
        logger.debug("Embedding generation failed: %s", e)
        return None


# =============================================================================
# SEMANTIC CACHE
# =============================================================================


class SemanticCache:
    """Persistent dual-threshold semantic cache.

    Stores query-response pairs with optional embeddings in SQLite.
    Lookup checks exact hash first (Tier 1), then falls back to
    cosine similarity over stored embeddings (Tier 2).

    Configuration loaded from cache.yaml with runtime overrides.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        config_path: Path | None = None,
        *,
        embedding_model: str | None = None,
        similarity_threshold: float | None = None,
        max_candidates: int | None = None,
        ttl_seconds: int | None = None,
        max_entries: int | None = None,
        scope: str | None = None,
    ):
        """Initialize the semantic cache.

        Args:
            db_path: Path to SQLite database (None = default).
            config_path: Path to cache.yaml (None = default).
            embedding_model: Override embedding model name.
            similarity_threshold: Override cosine threshold.
            max_candidates: Override max candidates per search.
            ttl_seconds: Override TTL in seconds.
            max_entries: Override max cached entries.
            scope: Override scope ('global' or 'conversation').
        """
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._lock = threading.Lock()

        # Load config from YAML, then apply explicit overrides
        self._config = self._load_config()
        if embedding_model is not None:
            self._config["embedding_model"] = embedding_model
        if similarity_threshold is not None:
            self._config["similarity_threshold"] = similarity_threshold
        if max_candidates is not None:
            self._config["max_candidates"] = max_candidates
        if ttl_seconds is not None:
            self._config["ttl_seconds"] = ttl_seconds
        if max_entries is not None:
            self._config["max_entries"] = max_entries
        if scope is not None:
            self._config["scope"] = scope

        # Session counters
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._similarity_sum = 0.0

        # Embedding availability (None = not yet tested)
        self._embeddings_available: bool | None = None

        # Initialize database
        self._init_db()

        logger.info(
            "SemanticCache initialized: model=%s, threshold=%.2f, "
            "ttl=%ds, max=%d, scope=%s",
            self._config["embedding_model"],
            self._config["similarity_threshold"],
            self._config["ttl_seconds"],
            self._config["max_entries"],
            self._config["scope"],
        )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def _load_config(self) -> dict:
        """Load configuration from cache.yaml with defaults.

        Returns:
            Merged config dict.
        """
        defaults = {
            "enabled": False,
            "exact_match_enabled": True,
            "semantic_match_enabled": True,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "ttl_seconds": DEFAULT_TTL_SECONDS,
            "max_entries": DEFAULT_MAX_ENTRIES,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "scope": DEFAULT_SCOPE,
            "max_candidates": DEFAULT_MAX_CANDIDATES,
            "avg_response_tokens": DEFAULT_AVG_RESPONSE_TOKENS,
        }
        try:
            if self._config_path.exists():
                with open(self._config_path, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh) or {}
                defaults.update(loaded)
        except Exception as exc:
            logger.warning("Could not load cache.yaml: %s", exc)
        return defaults

    def reload_config(self) -> dict:
        """Reload configuration from YAML file.

        Returns:
            Updated config dict.
        """
        self._config = self._load_config()
        logger.info("SemanticCache config reloaded")
        return dict(self._config)

    def get_config(self) -> dict:
        """Return current configuration as a dict.

        Returns:
            Copy of the config dict.
        """
        return dict(self._config)

    def update_config(self, updates: dict) -> dict:
        """Update config values at runtime and persist to YAML.

        Args:
            updates: Dict of config keys to update.

        Returns:
            Updated config dict.
        """
        allowed = {
            "enabled", "exact_match_enabled", "semantic_match_enabled",
            "similarity_threshold", "ttl_seconds", "max_entries",
            "embedding_model", "scope", "max_candidates",
            "avg_response_tokens",
        }
        for key, value in updates.items():
            if key in allowed:
                self._config[key] = value
        # Persist
        try:
            with open(self._config_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(self._config, fh, default_flow_style=False)
        except Exception as exc:
            logger.warning("Could not persist cache.yaml: %s", exc)
        return dict(self._config)

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()
        try:
            # S68: Unified cache entries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    query_hash TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    conversation_id TEXT NOT NULL DEFAULT '',
                    embedding BLOB,
                    embedding_dim INTEGER DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (query_hash, conversation_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_conv
                ON cache_entries(conversation_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_model
                ON cache_entries(model)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_accessed
                ON cache_entries(last_accessed)
            """)

            # S23 legacy: Keep old embeddings table for backward compat
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_embeddings (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    context_fingerprint TEXT NOT NULL DEFAULT ''
                )
            """)
            # S193 TC-04: guarded migration for pre-existing databases. Legacy
            # rows keep '' and therefore never match a requested fingerprint
            # (invalidation by attrition under the response-cache TTL).
            try:
                conn.execute(
                    "ALTER TABLE semantic_embeddings "
                    "ADD COLUMN context_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sem_model
                ON semantic_embeddings(model)
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Open a thread-safe SQLite connection."""
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------------------------------------------------------
    # Embedding helpers
    # -------------------------------------------------------------------------

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize an embedding vector to bytes (JSON).

        Args:
            embedding: Float vector.

        Returns:
            JSON-encoded bytes.
        """
        return json.dumps(embedding).encode("utf-8")

    def _deserialize_embedding(self, data: bytes) -> list[float]:
        """Deserialize bytes to an embedding vector.

        Args:
            data: JSON-encoded bytes.

        Returns:
            Float vector.
        """
        return json.loads(data.decode("utf-8"))

    def check_availability(self) -> bool:
        """Check if the embedding model is available via Ollama.

        Returns:
            True if embeddings can be generated.
        """
        if self._embeddings_available is not None:
            return self._embeddings_available

        try:
            embedding = _get_embedding("test", model=self._config["embedding_model"])
            self._embeddings_available = embedding is not None and len(embedding) > 0
        except Exception:
            self._embeddings_available = False

        logger.info(
            "Embeddings availability: %s (model=%s)",
            self._embeddings_available,
            self._config["embedding_model"],
        )
        return self._embeddings_available

    @property
    def embeddings_available(self) -> bool:
        """Whether the embedding model is reachable."""
        if self._embeddings_available is None:
            return self.check_availability()
        return self._embeddings_available

    @embeddings_available.setter
    def embeddings_available(self, value: bool) -> None:
        """Manually set embedding availability (for testing)."""
        self._embeddings_available = bool(value)

    # -------------------------------------------------------------------------
    # S68 Primary API: get / put / invalidate / get_stats
    # -------------------------------------------------------------------------

    def get(
        self,
        query: str,
        conversation_id: str | None = None,
        model: str = "",
        context_fingerprint: str = "",
    ) -> CacheEntry | None:
        """Look up a cached response for a query.

        Checks exact hash first (Tier 1), then embedding similarity
        (Tier 2) if semantic matching is enabled and embeddings are
        available.

        Args:
            query: The user query text.
            conversation_id: Optional conversation scope.
            model: LLM model filter (empty = any model).
            context_fingerprint: S193 TC-04 -- hash of the fully assembled
                generation context (system prompt). When non-empty, an entry
                only matches if it was stored under the same fingerprint, so
                a response generated under stale RAG/memory context is never
                served. Empty keeps the legacy behaviour (no filtering).

        Returns:
            CacheEntry if hit, None if miss.
        """
        if not self._config.get("enabled", False):
            return None

        # TC-01: in Bulbe, force conversation scope and fail closed -- never
        # serve a fuzzy hit from the shared (empty) bucket when no conversation
        # is in scope, so one conversation's response cannot bleed to another.
        bulbe = _is_bulbe()
        if bulbe and not (conversation_id or "").strip():
            self._misses += 1
            return None

        conv_id = self._resolve_conversation_id(
            conversation_id, force_conversation=bulbe
        )
        query_hash = _make_query_hash(query)
        now = time.time()
        ttl = self._config["ttl_seconds"]

        # Tier 1: Exact hash match
        if self._config.get("exact_match_enabled", True):
            entry = self._get_exact(
                query_hash, conv_id, model, now, ttl,
                context_fingerprint=context_fingerprint,
            )
            if entry is not None:
                self._exact_hits += 1
                self._tokens_saved += self._config.get(
                    "avg_response_tokens", DEFAULT_AVG_RESPONSE_TOKENS
                )
                return entry

        # Tier 2: Semantic similarity match
        if (
            self._config.get("semantic_match_enabled", True)
            and self.embeddings_available
        ):
            entry = self._get_semantic(
                query, conv_id, model, now, ttl,
                context_fingerprint=context_fingerprint,
            )
            if entry is not None:
                self._semantic_hits += 1
                self._similarity_sum += entry.similarity
                self._tokens_saved += self._config.get(
                    "avg_response_tokens", DEFAULT_AVG_RESPONSE_TOKENS
                )
                return entry

        self._misses += 1
        return None

    def put(
        self,
        query: str,
        response: str,
        model: str = "",
        metadata: dict | None = None,
        conversation_id: str | None = None,
        context_fingerprint: str = "",
    ) -> str:
        """Store a query-response pair in the cache.

        Generates and stores an embedding if the embedding model is
        available. Triggers LRU eviction if max_entries exceeded.

        Args:
            query: The user query text.
            response: The LLM response text.
            model: LLM model name.
            metadata: Optional metadata dict.
            conversation_id: Optional conversation scope.

        Returns:
            The query hash key.
        """
        if not self._config.get("enabled", False):
            return ""

        # TC-01: in Bulbe, force conversation scope and fail closed -- never
        # persist into the shared (empty) bucket when no conversation is in
        # scope, so a later query in another conversation cannot match it.
        bulbe = _is_bulbe()
        if bulbe and not (conversation_id or "").strip():
            return ""

        conv_id = self._resolve_conversation_id(
            conversation_id, force_conversation=bulbe
        )
        query_hash = _make_query_hash(query)
        now = time.time()
        # S193 TC-04: persist the generation-context fingerprint with the
        # entry so lookups can refuse stale-context matches.
        meta = dict(metadata or {})
        if context_fingerprint:
            meta["context_fingerprint"] = context_fingerprint
        meta_json = json.dumps(meta)

        # Generate embedding if available
        embedding_blob = None
        embedding_dim = 0
        if (
            self._config.get("semantic_match_enabled", True)
            and self.embeddings_available
        ):
            try:
                emb = _get_embedding(query, model=self._config["embedding_model"])
                if emb is not None:
                    embedding_blob = self._serialize_embedding(emb)
                    embedding_dim = len(emb)
            except Exception as exc:
                logger.debug("Embedding generation skipped: %s", exc)

        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO cache_entries
                   (query_hash, query_text, response, model, conversation_id,
                    embedding, embedding_dim, metadata, created_at,
                    last_accessed, hit_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                   ON CONFLICT(query_hash, conversation_id) DO UPDATE SET
                       response = excluded.response,
                       model = excluded.model,
                       embedding = excluded.embedding,
                       embedding_dim = excluded.embedding_dim,
                       metadata = excluded.metadata,
                       created_at = excluded.created_at,
                       last_accessed = excluded.last_accessed,
                       hit_count = 0
                """,
                (
                    query_hash, query, response, model, conv_id,
                    embedding_blob, embedding_dim, meta_json, now, now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # LRU eviction
        self._evict_if_needed()

        logger.debug(
            "Cache put: hash=%s, model=%s, conv=%s, embedding=%s",
            query_hash[:12], model, conv_id[:12] if conv_id else "global",
            embedding_dim > 0,
        )
        return query_hash

    def invalidate(self, conversation_id: str | None = None) -> int:
        """Evict cache entries.

        Args:
            conversation_id: If provided, only evict entries for this
                conversation. If None, clear ALL entries.

        Returns:
            Number of entries removed.
        """
        conn = self._get_connection()
        try:
            if conversation_id is not None:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE conversation_id = ?",
                    (conversation_id,),
                )
            else:
                cursor = conn.execute("DELETE FROM cache_entries")
                # Reset session counters on full clear
                self._exact_hits = 0
                self._semantic_hits = 0
                self._misses = 0
                self._tokens_saved = 0
            conn.commit()
            count = cursor.rowcount
            logger.info(
                "Cache invalidated: %d entries removed (conv=%s)",
                count,
                conversation_id or "all",
            )
            return count
        finally:
            conn.close()

    def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics.

        Returns:
            CacheStats with session counters and config.
        """
        now = time.time()
        ttl = self._config["ttl_seconds"]
        conn = self._get_connection()
        try:
            # Count non-expired entries
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM cache_entries WHERE (? - created_at) < ?",
                (now, ttl),
            ).fetchone()
            total_entries = row["cnt"] if row else 0
        finally:
            conn.close()

        # DB file size
        size_bytes = 0
        try:
            if self._db_path.exists():
                size_bytes = self._db_path.stat().st_size
        except Exception:
            pass

        total_requests = self._exact_hits + self._semantic_hits + self._misses
        total_hits = self._exact_hits + self._semantic_hits
        hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        exact_rate = self._exact_hits / total_requests if total_requests > 0 else 0.0
        sem_rate = self._semantic_hits / total_requests if total_requests > 0 else 0.0

        return CacheStats(
            total_entries=total_entries,
            exact_hits=self._exact_hits,
            semantic_hits=self._semantic_hits,
            total_misses=self._misses,
            hit_rate=hit_rate,
            exact_hit_rate=exact_rate,
            semantic_hit_rate=sem_rate,
            tokens_saved=self._tokens_saved,
            size_bytes=size_bytes,
            max_entries=self._config["max_entries"],
            ttl_seconds=self._config["ttl_seconds"],
            similarity_threshold=self._config["similarity_threshold"],
            embedding_model=self._config["embedding_model"],
            scope=self._config["scope"],
            enabled=self._config.get("enabled", False),
            embeddings_available=self._embeddings_available or False,
        )

    # -------------------------------------------------------------------------
    # Internal lookup helpers
    # -------------------------------------------------------------------------

    def _resolve_conversation_id(
        self, conversation_id: str | None, force_conversation: bool = False
    ) -> str:
        """Resolve conversation ID based on scope config.

        Args:
            conversation_id: Caller-provided conversation ID.
            force_conversation: When True (Bulbe), ignore a "global" config and
                always scope per conversation (TC-01).

        Returns:
            Empty string for global scope, or the conversation ID.
        """
        if not force_conversation and self._config.get("scope", DEFAULT_SCOPE) == "global":
            return ""
        return conversation_id or ""

    def _get_exact(
        self,
        query_hash: str,
        conv_id: str,
        model: str,
        now: float,
        ttl: int,
        context_fingerprint: str = "",
    ) -> CacheEntry | None:
        """Tier 1: Exact hash lookup.

        Args:
            query_hash: SHA-256 hash of query.
            conv_id: Conversation ID (empty for global).
            model: Model filter (empty = any).
            now: Current timestamp.
            ttl: TTL in seconds.
            context_fingerprint: S193 TC-04 -- when non-empty, the stored
                entry must carry the same fingerprint or the lookup misses.

        Returns:
            CacheEntry or None.
        """
        conn = self._get_connection()
        try:
            if model:
                row = conn.execute(
                    """SELECT * FROM cache_entries
                       WHERE query_hash = ? AND conversation_id = ?
                       AND model = ?""",
                    (query_hash, conv_id, model),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT * FROM cache_entries
                       WHERE query_hash = ? AND conversation_id = ?""",
                    (query_hash, conv_id),
                ).fetchone()

            if row is None:
                return None

            # TTL check
            if (now - row["created_at"]) >= ttl:
                conn.execute(
                    "DELETE FROM cache_entries WHERE query_hash = ? AND conversation_id = ?",
                    (query_hash, conv_id),
                )
                conn.commit()
                return None

            meta = {}
            try:
                meta = json.loads(row["metadata"])
            except Exception:
                pass

            # S193 TC-04: context fingerprint must match when requested.
            # Entries stored without one (legacy) never match a request that
            # carries a fingerprint -- treated as a plain miss.
            if context_fingerprint and meta.get("context_fingerprint", "") != context_fingerprint:
                return None

            # Update access stats
            new_hits = row["hit_count"] + 1
            conn.execute(
                """UPDATE cache_entries SET last_accessed = ?, hit_count = ?
                   WHERE query_hash = ? AND conversation_id = ?""",
                (now, new_hits, query_hash, conv_id),
            )
            conn.commit()

            return CacheEntry(
                query_hash=row["query_hash"],
                query_text=row["query_text"],
                response=row["response"],
                model=row["model"],
                conversation_id=row["conversation_id"],
                similarity=1.0,
                match_type="exact",
                created_at=row["created_at"],
                last_accessed=now,
                hit_count=new_hits,
                metadata=meta,
            )
        finally:
            conn.close()

    def _get_semantic(
        self,
        query: str,
        conv_id: str,
        model: str,
        now: float,
        ttl: int,
        context_fingerprint: str = "",
    ) -> CacheEntry | None:
        """Tier 2: Semantic similarity lookup.

        Args:
            query: Raw query text.
            conv_id: Conversation ID.
            model: Model filter (empty = any).
            now: Current timestamp.
            ttl: TTL in seconds.
            context_fingerprint: S193 TC-04 -- when non-empty, candidates
                stored under a different fingerprint are skipped.

        Returns:
            CacheEntry or None.
        """
        query_embedding = _get_embedding(
            query, model=self._config["embedding_model"]
        )
        if query_embedding is None:
            return None

        threshold = self._config["similarity_threshold"]
        max_cands = self._config["max_candidates"]

        conn = self._get_connection()
        try:
            if model:
                rows = conn.execute(
                    """SELECT query_hash, query_text, response, model,
                              conversation_id, embedding, metadata,
                              created_at, last_accessed, hit_count
                       FROM cache_entries
                       WHERE conversation_id = ? AND model = ?
                       AND (? - created_at) < ?
                       AND embedding IS NOT NULL
                       ORDER BY last_accessed DESC
                       LIMIT ?""",
                    (conv_id, model, now, ttl, max_cands),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT query_hash, query_text, response, model,
                              conversation_id, embedding, metadata,
                              created_at, last_accessed, hit_count
                       FROM cache_entries
                       WHERE conversation_id = ?
                       AND (? - created_at) < ?
                       AND embedding IS NOT NULL
                       ORDER BY last_accessed DESC
                       LIMIT ?""",
                    (conv_id, now, ttl, max_cands),
                ).fetchall()

            if not rows:
                return None

            best_row = None
            best_sim = 0.0

            for row in rows:
                try:
                    # S193 TC-04: skip candidates stored under a different
                    # generation context when a fingerprint is requested.
                    if context_fingerprint:
                        try:
                            cand_meta = json.loads(row["metadata"])
                        except Exception:
                            cand_meta = {}
                        if cand_meta.get("context_fingerprint", "") != context_fingerprint:
                            continue
                    candidate_emb = self._deserialize_embedding(row["embedding"])
                    sim = cosine_similarity(query_embedding, candidate_emb)
                    if sim >= threshold and sim > best_sim:
                        best_sim = sim
                        best_row = row
                except Exception as exc:
                    logger.debug("Error comparing embedding: %s", exc)
                    continue

            if best_row is None:
                return None

            # Update access stats on the matched entry
            new_hits = best_row["hit_count"] + 1
            conn.execute(
                """UPDATE cache_entries SET last_accessed = ?, hit_count = ?
                   WHERE query_hash = ? AND conversation_id = ?""",
                (now, new_hits, best_row["query_hash"], best_row["conversation_id"]),
            )
            conn.commit()

            meta = {}
            try:
                meta = json.loads(best_row["metadata"])
            except Exception:
                pass

            return CacheEntry(
                query_hash=best_row["query_hash"],
                query_text=best_row["query_text"],
                response=best_row["response"],
                model=best_row["model"],
                conversation_id=best_row["conversation_id"],
                similarity=best_sim,
                match_type="semantic",
                created_at=best_row["created_at"],
                last_accessed=now,
                hit_count=new_hits,
                metadata=meta,
            )
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Eviction
    # -------------------------------------------------------------------------

    def _evict_if_needed(self) -> int:
        """Evict oldest entries if max_entries exceeded (LRU).

        Returns:
            Number of entries evicted.
        """
        max_entries = self._config["max_entries"]
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM cache_entries"
            ).fetchone()
            total = row["cnt"] if row else 0

            if total <= max_entries:
                return 0

            excess = total - max_entries
            conn.execute(
                """DELETE FROM cache_entries WHERE rowid IN (
                       SELECT rowid FROM cache_entries
                       ORDER BY last_accessed ASC
                       LIMIT ?
                   )""",
                (excess,),
            )
            conn.commit()
            logger.debug("LRU eviction: removed %d entries", excess)
            return excess
        finally:
            conn.close()

    def expire_stale(self) -> int:
        """Remove all entries past their TTL.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        ttl = self._config["ttl_seconds"]
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM cache_entries WHERE (? - created_at) >= ?",
                (now, ttl),
            )
            conn.commit()
            count = cursor.rowcount
            if count > 0:
                logger.debug("TTL expiry: removed %d stale entries", count)
            return count
        finally:
            conn.close()

    def entry_count(self) -> int:
        """Get total number of cache entries (including expired).

        Returns:
            Count.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM cache_entries"
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Properties (config shortcuts)
    # -------------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the cache is enabled."""
        return self._config.get("enabled", False)

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the cache."""
        self._config["enabled"] = bool(value)
        logger.info("SemanticCache %s", "enabled" if value else "disabled")

    @property
    def similarity_threshold(self) -> float:
        """Current cosine similarity threshold."""
        return self._config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None:
        """Set similarity threshold (clamped to 0.5-0.99)."""
        self._config["similarity_threshold"] = max(0.5, min(0.99, float(value)))

    @property
    def embedding_model(self) -> str:
        """Current embedding model name."""
        return self._config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

    @property
    def scope(self) -> str:
        """Current scope ('global' or 'conversation')."""
        return self._config.get("scope", DEFAULT_SCOPE)

    @property
    def ttl_seconds(self) -> int:
        """Current TTL in seconds."""
        return self._config.get("ttl_seconds", DEFAULT_TTL_SECONDS)

    @property
    def max_entries(self) -> int:
        """Current max entries limit."""
        return self._config.get("max_entries", DEFAULT_MAX_ENTRIES)

    @property
    def semantic_hits(self) -> int:
        """Session semantic hit count."""
        return self._semantic_hits

    @property
    def exact_hits(self) -> int:
        """Session exact hit count."""
        return self._exact_hits

    @property
    def semantic_misses(self) -> int:
        """Session miss count (legacy compat name)."""
        return self._misses

    @property
    def tokens_saved(self) -> int:
        """Estimated tokens saved this session."""
        return self._tokens_saved

    # -------------------------------------------------------------------------
    # S23 Legacy API (backward compatibility)
    # -------------------------------------------------------------------------

    def store_embedding(
        self,
        cache_key: str,
        model: str,
        query_text: str,
        embedding: list[float] | None = None,
        context_fingerprint: str = "",
    ) -> bool:
        """Store an embedding in the legacy table (S23 compat).

        Args:
            cache_key: SHA-256 cache key.
            model: LLM model name.
            query_text: Original query text.
            embedding: Pre-computed embedding (generates if None).

        Returns:
            True if stored successfully.
        """
        if not self._config.get("enabled", False):
            return False

        if embedding is None:
            embedding = _get_embedding(
                query_text, model=self._config["embedding_model"]
            )
            if embedding is None:
                return False

        blob = self._serialize_embedding(embedding)
        dim = len(embedding)
        now = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO semantic_embeddings
                   (cache_key, model, query_text, embedding, embedding_dim,
                    created_at, context_fingerprint)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                       embedding = excluded.embedding,
                       embedding_dim = excluded.embedding_dim,
                       query_text = excluded.query_text,
                       created_at = excluded.created_at,
                       context_fingerprint = excluded.context_fingerprint
                """,
                (cache_key, model, query_text, blob, dim, now,
                 context_fingerprint),
            )
            conn.commit()
            return True
        except Exception as exc:
            logger.warning("Failed to store embedding: %s", exc)
            return False
        finally:
            conn.close()

    def find_similar(
        self,
        query_text: str,
        model: str,
        threshold: float | None = None,
        exclude_key: str | None = None,
        context_fingerprint: str = "",
    ) -> SemanticMatch | None:
        """Find the most similar entry in the legacy table (S23 compat).

        Args:
            query_text: Query to search for.
            model: LLM model filter.
            threshold: Similarity threshold override.
            exclude_key: Key to exclude from results.

        Returns:
            SemanticMatch or None.
        """
        if not self._config.get("enabled", False):
            return None

        threshold = threshold or self._config["similarity_threshold"]

        query_embedding = _get_embedding(
            query_text, model=self._config["embedding_model"]
        )
        if query_embedding is None:
            self._misses += 1
            return None

        return self.find_similar_by_embedding(
            query_embedding, model, threshold, exclude_key, query_text,
            context_fingerprint=context_fingerprint,
        )

    def find_similar_by_embedding(
        self,
        query_embedding: list[float],
        model: str,
        threshold: float | None = None,
        exclude_key: str | None = None,
        query_text: str = "",
        context_fingerprint: str = "",
    ) -> SemanticMatch | None:
        """Search by pre-computed embedding in legacy table (S23 compat).

        Args:
            query_embedding: Pre-computed embedding vector.
            model: LLM model filter.
            threshold: Similarity threshold.
            exclude_key: Key to exclude.
            query_text: Original text (for logging).

        Returns:
            SemanticMatch or None.
        """
        if not self._config.get("enabled", False):
            return None

        threshold = threshold or self._config["similarity_threshold"]
        max_cands = self._config["max_candidates"]

        # S193 TC-04: when a fingerprint is requested, only candidates
        # stored under the same generation context qualify (legacy rows keep
        # '' and are excluded by construction).
        fp_clause = " AND context_fingerprint = ?" if context_fingerprint else ""
        conn = self._get_connection()
        try:
            if exclude_key:
                params: list = [model, exclude_key]
                if context_fingerprint:
                    params.append(context_fingerprint)
                params.append(max_cands)
                rows = conn.execute(
                    f"""SELECT cache_key, model, query_text, embedding
                       FROM semantic_embeddings
                       WHERE model = ? AND cache_key != ?{fp_clause}
                       ORDER BY created_at DESC LIMIT ?""",
                    params,
                ).fetchall()
            else:
                params = [model]
                if context_fingerprint:
                    params.append(context_fingerprint)
                params.append(max_cands)
                rows = conn.execute(
                    f"""SELECT cache_key, model, query_text, embedding
                       FROM semantic_embeddings
                       WHERE model = ?{fp_clause}
                       ORDER BY created_at DESC LIMIT ?""",
                    params,
                ).fetchall()
        finally:
            conn.close()

        if not rows:
            self._misses += 1
            return None

        best_match = None
        best_sim = 0.0

        for row in rows:
            try:
                cand_emb = self._deserialize_embedding(row["embedding"])
                sim = cosine_similarity(query_embedding, cand_emb)
                if sim >= threshold and sim > best_sim:
                    best_sim = sim
                    best_match = SemanticMatch(
                        cache_key=row["cache_key"],
                        similarity=sim,
                        model=row["model"],
                        query_text=row["query_text"],
                    )
            except Exception:
                continue

        if best_match:
            self._semantic_hits += 1
            self._similarity_sum += best_sim
        else:
            self._misses += 1

        return best_match

    def get_with_fallback(
        self,
        response_cache: Any,
        cache_key: str,
        model: str,
        query_text: str,
        context_fingerprint: str = "",
    ) -> tuple[Any | None, float, str]:
        """Exact cache lookup with semantic fallback (S23 compat).

        Args:
            response_cache: ResponseCache instance.
            cache_key: Exact SHA-256 key.
            model: LLM model name.
            query_text: User query text.

        Returns:
            Tuple of (entry_or_none, similarity, match_type).
        """
        entry = response_cache.get(cache_key)
        if entry is not None:
            return entry, 1.0, "exact"

        if not self._config.get("enabled", False):
            return None, 0.0, "miss"

        match = self.find_similar(
            query_text=query_text,
            model=model,
            exclude_key=cache_key,
            context_fingerprint=context_fingerprint,
        )

        if match is None:
            return None, 0.0, "miss"

        semantic_entry = response_cache.get(match.cache_key)
        if semantic_entry is not None:
            return semantic_entry, match.similarity, "semantic"

        self.remove_embedding(match.cache_key)
        return None, 0.0, "miss"

    def put_with_embedding(
        self,
        response_cache: Any,
        model: str,
        system_prompt: str,
        user_content: str,
        response: str,
        task_type: str = "",
        ttl: int | None = None,
        explicit_key: str | None = None,
    ) -> tuple[str, bool]:
        """Store in response cache + generate embedding (S23 compat).

        Args:
            response_cache: ResponseCache instance.
            model: LLM model name.
            system_prompt: System prompt.
            user_content: User content.
            response: LLM response.
            task_type: Task type string.
            ttl: Optional TTL override.
            explicit_key: Pre-computed cache key.

        Returns:
            Tuple of (cache_key, embedding_stored).
        """
        cache_key = response_cache.put(
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            response=response,
            task_type=task_type,
            ttl=ttl,
            explicit_key=explicit_key,
        )

        if not cache_key:
            return "", False

        embedded = self.store_embedding(
            cache_key=cache_key,
            model=model,
            query_text=user_content,
        )

        return cache_key, embedded

    # -------------------------------------------------------------------------
    # Legacy cleanup methods
    # -------------------------------------------------------------------------

    def remove_embedding(self, cache_key: str) -> bool:
        """Remove a legacy embedding by key.

        Args:
            cache_key: Embedding cache key.

        Returns:
            True if removed.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM semantic_embeddings WHERE cache_key = ?",
                (cache_key,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def remove_embeddings_for_model(self, model: str) -> int:
        """Remove all legacy embeddings for a model.

        Args:
            model: LLM model name.

        Returns:
            Count of removed embeddings.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM semantic_embeddings WHERE model = ?",
                (model,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def clear(self) -> int:
        """Clear all cache entries and legacy embeddings.

        Returns:
            Total entries removed.
        """
        conn = self._get_connection()
        try:
            c1 = conn.execute("DELETE FROM cache_entries").rowcount
            c2 = conn.execute("DELETE FROM semantic_embeddings").rowcount
            conn.commit()
            total = c1 + c2
            self._exact_hits = 0
            self._semantic_hits = 0
            self._misses = 0
            self._tokens_saved = 0
            self._similarity_sum = 0.0
            logger.info("SemanticCache CLEAR: %d entries removed", total)
            return total
        finally:
            conn.close()

    def cleanup_orphans(self, response_cache: Any) -> int:
        """Remove legacy embeddings whose response cache entry is gone.

        Args:
            response_cache: ResponseCache instance.

        Returns:
            Number of orphans removed.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT cache_key FROM semantic_embeddings"
            ).fetchall()

            orphans = []
            for row in rows:
                key = row["cache_key"]
                entry = response_cache.get(key)
                if entry is None:
                    orphans.append(key)

            if orphans:
                placeholders = ",".join("?" for _ in orphans)
                conn.execute(
                    "DELETE FROM semantic_embeddings "
                    "WHERE cache_key IN ({})".format(placeholders),
                    orphans,
                )
                conn.commit()

            return len(orphans)
        finally:
            conn.close()

    def embedding_count(self) -> int:
        """Get total number of legacy embeddings.

        Returns:
            Count.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM semantic_embeddings"
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

semantic_cache = SemanticCache()
