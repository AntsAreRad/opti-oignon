#!/usr/bin/env python3
"""
RESPONSE CACHE - OPTI-OIGNON v1.4.0 (Session 18)
===================================================

A cache of LLM responses: a response already generated for an identical
request is served at once.

Features:
    - SHA-256 key based on: model + system_prompt + user content
    - Configurable expiration TTL (default: 1 hour)
    - Partitioning by model
    - Statistics (hits, misses, size)
    - Thread-safe via SQLite
    - Expired entries cleaned up automatically
    - Each entry keeps the conversation it was given in: a deletion forgets
      its answers (forget_conversation), and every deleted entry is
      overwritten

Architecture:
    - SQLite persistence (the same pattern as conversation.py)
    - Module-level singleton: response_cache
    - Conditional import in executor.py

Author: Leon
"""

import hashlib
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import DATA_DIR

logger = logging.getLogger(__name__)
# Audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    logger.warning(
        "db_utils unavailable: response_cache falling back to PLAINTEXT sqlite3. "
        "Cached responses are NOT encrypted at rest."
    )
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)



# =============================================================================
# CONSTANTES
# =============================================================================

# Default TTL: 1 hour (in seconds)
DEFAULT_TTL = 3600

# Default maximum cache size (number of entries)
DEFAULT_MAX_ENTRIES = 500

# Nettoyage automatique tous les N appels
CLEANUP_INTERVAL = 50

# The conversation an entry an older version cached came from, which it did not keep: not known.
_ORIGIN_UNKNOWN = "?"
# The conversations this process forgot: none is cached again, so a turn that outlived its deletion lands nowhere,
# until a refused deletion hands it back (``unforget_conversation``).
_FORGOTTEN: set[str] = set()
# Held while a mark is checked and an entry written, and while a forgetting marks and deletes.
_MARKS_LOCK = threading.Lock()
# How many times this process emptied the cache, and, by key, how many it had when the request whose lookup of that
# key missed began (``begin_request``): an answer a request already under way puts after an emptying -- a deletion of
# what it may have been drawn from -- is refused, its request older than the emptying. A key looked up again keeps the
# oldest count. The oldest lookups give way past the bound, far above the requests a process has under way.
_EMPTIED = [0]
_ASKED: OrderedDict[Any, int] = OrderedDict()
_ASKED_LIMIT = 4096
# The count when the request this thread is making began, before its prompt drew on any store; and the count a request
# hands back right before its put (``declare_request``), which that put spends.
_REQUEST = threading.local()
_PUTTING = threading.local()


def begin_request(since: int | None = None) -> int:
    """A request begins on this thread, before its prompt draws on any store: its lookups are marked from now, so an
    answer whose prompt a deletion outran is never put after it; the count, which the request hands back at its put
    (``declare_request``). ``since``: the count its turn began at, when the request is a later call of a turn (a
    pipeline's step, the agentic executor's next pass), kept when older: the call is judged by the turn's beginning."""
    with _MARKS_LOCK:
        count = _EMPTIED[0] if since is None else min(int(since), _EMPTIED[0])
        _REQUEST.count = count
        return count


def emptied() -> int:
    """How many times this cache was emptied so far, read without its lock: a turn's mark, taken where waiting on a
    put's disk work would stall (``cached_answers.mark_cached_turn``). An emptying racing the read is counted after
    the mark, so the turn is refused more often, never less."""
    return _EMPTIED[0]


def declare_request(count: int) -> None:
    """The request that began at ``count`` (``begin_request``) is about to put its answer on this thread: the next put
    here is refused when the cache was emptied since, whatever another request on the same key saw. Spent by that
    put."""
    _PUTTING.count = int(count)


def _declared() -> int | None:
    """The count the request putting now handed back on this thread, spent; None when none was."""
    count = getattr(_PUTTING, "count", None)
    _PUTTING.count = None
    return count


def _note_asked(key: Any) -> None:
    """A lookup of ``key`` missed: remember how many emptyings the request making it saw when it began -- this
    thread's, or now when it marked none --, the oldest when the key was already looked up."""
    with _MARKS_LOCK:
        seen = min(getattr(_REQUEST, "count", _EMPTIED[0]), _EMPTIED[0])
        _ASKED[key] = min(_ASKED.get(key, seen), seen)
        _ASKED.move_to_end(key)
        while len(_ASKED) > _ASKED_LIMIT:
            _ASKED.popitem(last=False)


def _outdated(key: Any, declared: int | None) -> bool:
    """Whether the answer about to be put under ``key`` was asked before the cache's last emptying: by the request's
    own count when it handed one back (``declared``), else by the mark its key's lookup left. Held under
    ``_MARKS_LOCK``; the key's mark is spent."""
    asked = _ASKED.pop(key, None)
    if declared is not None:
        return declared != _EMPTIED[0]
    return asked is not None and asked != _EMPTIED[0]


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """The columns ``table`` has in the file; empty when it has no such table."""
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CacheEntry:
    """Individual cache entry.

    Attributes:
        cache_key: SHA-256 hash identifiant unique
        model: Model name Ollama
        prompt_hash: System prompt hash
        query_hash: User content hash
        response: Reponse complete du LLM
        task_type: Type de tache (code_r, general, etc.)
        created_at: Timestamp de creation
        last_accessed: Dernier acces
        hit_count: Nombre d'acces
        ttl: Duree de vie en secondes
    """
    cache_key: str
    model: str
    prompt_hash: str
    query_hash: str
    response: str
    task_type: str = ""
    created_at: float = 0.0
    last_accessed: float = 0.0
    hit_count: int = 0
    ttl: int = DEFAULT_TTL


@dataclass
class CacheStats:
    """Global cache statistics.

    Attributes:
        total_entries: Nombre d'entrees actives
        total_hits: Nombre total de cache hits
        total_misses: Nombre total de cache misses
        hit_rate: Taux de hit (0.0 - 1.0)
        entries_by_model: Repartition par modele
        oldest_entry: Age de l'entree la plus ancienne (secondes)
        total_size_bytes: Taille approximative en octets
    """
    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    entries_by_model: dict[str, int] = field(default_factory=dict)
    oldest_entry: float = 0.0
    total_size_bytes: int = 0


# =============================================================================
# RESPONSE CACHE
# =============================================================================

class ResponseCache:
    """Cache de reponses LLM avec persistance SQLite.

    Utilise SHA-256 pour generer des cles uniques basees sur:
    - The model used
    - Le hash du system prompt
    - The hash of user content (question + document)

    La combinaison de ces trois elements garantit que seules
    des requetes strictement identiques sont servies depuis le cache.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        default_ttl: int = DEFAULT_TTL,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        """Initialize the response cache.

        Args:
            db_path: Path to SQLite database (default: DATA_DIR/response_cache.db)
            default_ttl: Default TTL in seconds for cache entries
            max_entries: Maximum number of cache entries
        """
        self._db_path = db_path or (DATA_DIR / "response_cache.db")
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._enabled = True
        self._lock = threading.Lock()

        # Compteurs de session (reset a each demarrage)
        self._session_hits = 0
        self._session_misses = 0
        self._call_count = 0

        # Dernier resultat pour affichage UI
        self._last_cache_hit = False
        self._last_cache_key = ""

        # The schema is built at the first connection, not here. This cache
        # is constructed at module scope, so building it here opened a
        # database on every import of the package -- with encryption not
        # enforced, at a moment where a refusal could not be handled.
        self._schema_ready = False

        logger.info(
            f"ResponseCache initialise: db={self._db_path}, "
            f"ttl={self._default_ttl}s, max={self._max_entries}"
        )

    def _init_db(self) -> None:
        """Create the cache table if it does not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    response TEXT NOT NULL,
                    task_type TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    ttl INTEGER NOT NULL,
                    conversation_id TEXT NOT NULL DEFAULT '?'
                )
            """)
            # The conversation each entry came from, '' for a request made outside any: a deletion reaches its
            # answers by it. Every entry an older file holds says its origin is not known (_ORIGIN_UNKNOWN), and
            # the first forgetting drops them. The table is asked, never a duplicate caught: the cipher's driver
            # raises its own classes.
            if "conversation_id" not in _columns(conn, "response_cache"):
                conn.execute(
                    "ALTER TABLE response_cache ADD COLUMN conversation_id TEXT NOT NULL "
                    f"DEFAULT '{_ORIGIN_UNKNOWN}'"
                )
            # Index for cleanup and search by model
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_model
                ON response_cache(model)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_created
                ON response_cache(created_at)
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Ouvre une connexion SQLite thread-safe."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # What any write frees -- an expiry, an eviction, a forgetting -- is overwritten, never left in free pages.
        conn.execute("PRAGMA secure_delete = ON")
        if not self._schema_ready:
            self._schema_ready = True
            self._init_db()
        return conn

    # -------------------------------------------------------------------------
    # Gestion cles
    # -------------------------------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate a SHA-256 hash for text.

        Args:
            text: Texte a hasher

        Returns:
            Hash SHA-256 hexadecimal
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def make_cache_key(model: str, system_prompt: str, user_content: str) -> str:
        """Generate a unique cache key from query components.

        The key is a SHA-256 hash of the concatenation of:
        - model name
        - system prompt hash
        - user content hash

        This ensures that different models, prompts, or queries
        produce distinct cache keys.

        Args:
            model: Ollama model name
            system_prompt: Full system prompt text
            user_content: User message (question + optional document)

        Returns:
            SHA-256 hex string as cache key
        """
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        query_hash = hashlib.sha256(user_content.encode("utf-8")).hexdigest()
        combined = f"{model}:{prompt_hash}:{query_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def make_conversation_cache_key(
        model: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        user_content: str,
    ) -> str:
        """Generate a cache key for multi-turn conversation context.

        Hashes the full conversation history (excluding system prompt message
        and current user message which are hashed separately) to produce a
        deterministic key. Two identical conversation states will always
        produce the same key.

        Args:
            model: Ollama model name
            system_prompt: Full system prompt text
            messages: Conversation messages as list of {role, content} dicts,
                system messages left out; the chat path passes every other
                message, the current turn included.
            user_content: Current user message

        Returns:
            SHA-256 hex string as cache key
        """
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        # Serialisation deterministe de l'historique
        history_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_parts.append(f"{role}:{content}")
        history_str = "\n".join(history_parts)
        history_hash = hashlib.sha256(history_str.encode("utf-8")).hexdigest()
        query_hash = hashlib.sha256(user_content.encode("utf-8")).hexdigest()
        combined = f"{model}:{prompt_hash}:{history_hash}:{query_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    # Operations principales
    # -------------------------------------------------------------------------

    def get(self, cache_key: str) -> CacheEntry | None:
        """Look up a cache entry by key.

        Returns the entry if found and not expired, None otherwise.
        Updates hit count and last_accessed on hit.
        Automatically removes expired entries.

        Args:
            cache_key: SHA-256 cache key

        Returns:
            CacheEntry if hit, None if miss or expired
        """
        if not self._enabled:
            self._session_misses += 1
            self._last_cache_hit = False
            return None

        self._call_count += 1

        # Nettoyage periodique
        if self._call_count % CLEANUP_INTERVAL == 0:
            self._cleanup_expired()

        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM response_cache WHERE cache_key = ?",
                (cache_key,)
            ).fetchone()

            if row is None:
                self._session_misses += 1
                self._last_cache_hit = False
                self._last_cache_key = cache_key
                _note_asked(cache_key)
                return None

            # Verification TTL
            now = time.time()
            created = row["created_at"]
            ttl = row["ttl"]
            if now - created > ttl:
                # Entree expiree, on la supprime
                conn.execute(
                    "DELETE FROM response_cache WHERE cache_key = ?",
                    (cache_key,)
                )
                conn.commit()
                self._session_misses += 1
                self._last_cache_hit = False
                self._last_cache_key = cache_key
                _note_asked(cache_key)
                logger.debug(f"Cache entry expired: {cache_key[:12]}...")
                return None

            # Cache hit: mise a jour des stats
            conn.execute(
                """UPDATE response_cache
                   SET hit_count = hit_count + 1, last_accessed = ?
                   WHERE cache_key = ?""",
                (now, cache_key)
            )
            conn.commit()

            entry = CacheEntry(
                cache_key=row["cache_key"],
                model=row["model"],
                prompt_hash=row["prompt_hash"],
                query_hash=row["query_hash"],
                response=row["response"],
                task_type=row["task_type"] or "",
                created_at=row["created_at"],
                last_accessed=now,
                hit_count=row["hit_count"] + 1,
                ttl=row["ttl"],
            )

            self._session_hits += 1
            self._last_cache_hit = True
            self._last_cache_key = cache_key
            logger.info(
                f"Cache HIT: {cache_key[:12]}... "
                f"(model={entry.model}, hits={entry.hit_count})"
            )
            return entry

        finally:
            conn.close()

    def put(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
        response: str,
        task_type: str = "",
        ttl: int | None = None,
        explicit_key: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """Store a response in the cache.

        Generates the cache key automatically from query components,
        or uses an explicit key if provided (e.g. for multi-turn
        conversation caching where the key includes history hash).

        Args:
            model: Ollama model name
            system_prompt: Full system prompt text
            user_content: User message content
            response: Complete LLM response to cache
            task_type: Optional task type for stats
            ttl: Optional TTL override (uses default if None)
            explicit_key: Optional pre-computed cache key
            conversation_id: The conversation the response was given in, None for a request made outside any; the
                entry keeps it, the last to put it, so that the conversation's deletion forgets it. A conversation
                this process forgot is never cached again: nothing is stored, and "" is returned.

        Returns:
            The cache key for the stored entry
        """
        declared = _declared()
        if not self._enabled:
            return ""

        # Utiliser la cle explicite si fournie (multi-turn), sinon generer
        cache_key = explicit_key or self.make_cache_key(model, system_prompt, user_content)
        prompt_hash = self._hash_text(system_prompt)
        query_hash = self._hash_text(user_content)
        now = time.time()
        entry_ttl = ttl if ttl is not None else self._default_ttl
        origin = conversation_id or ""

        # The mark is checked under the lock its forgetting takes, at the write: a put that took it first is
        # deleted by the forgetting, one that takes it after is refused.
        with _MARKS_LOCK:
            if origin and origin in _FORGOTTEN:
                return ""
            # An answer asked before the cache was last emptied may be drawn from what that emptying forgot.
            if _outdated(cache_key, declared):
                logger.debug("Cache put declined: asked before the cache was emptied")
                return ""
            conn = self._get_connection()
            try:
                # Evict the least recently used when the cache is full
                count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM response_cache"
                ).fetchone()["cnt"]

                if count >= self._max_entries:
                    self._evict_lru(conn, count - self._max_entries + 1)

                # Insert, or update the entry already under this key (UPSERT)
                conn.execute(
                    """INSERT INTO response_cache
                       (cache_key, model, prompt_hash, query_hash, response,
                        task_type, created_at, last_accessed, hit_count, ttl,
                        conversation_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                       ON CONFLICT(cache_key) DO UPDATE SET
                           response = excluded.response,
                           created_at = excluded.created_at,
                           last_accessed = excluded.last_accessed,
                           hit_count = 0,
                           ttl = excluded.ttl,
                           conversation_id = excluded.conversation_id
                    """,
                    (cache_key, model, prompt_hash, query_hash, response,
                     task_type, now, now, entry_ttl, origin)
                )
                conn.commit()

                logger.debug(
                    f"Cache PUT: {cache_key[:12]}... "
                    f"(model={model}, ttl={entry_ttl}s, "
                    f"response_len={len(response)})"
                )
                return cache_key

            finally:
                conn.close()

    def forget_conversation(self, conversation_id: str) -> int:
        """Delete every answer cached in a deleted conversation; how many.

        Unlike a lookup, a store that cannot be opened raises: a deletion that asks must never take it for one with
        nothing to forget. A file that is not there holds nothing, and is not created; the schema is never changed
        here. The entries an older version cached, which did not keep their conversation -- one of them may be this
        one's -- go with the first forgetting (an entry dropped is only asked again): marked of an origin not known,
        or, in a file still without the column, every entry. What is deleted is overwritten (``secure_delete``).
        From then on, the conversation is never cached again in this process (``_FORGOTTEN``), unless the
        forgetting fails.
        """
        try:
            with _MARKS_LOCK:
                # Marked under the puts' lock: a put that took it first is deleted below, one that takes it after
                # is refused.
                _FORGOTTEN.add(conversation_id)
                return self._forget_entries(conversation_id)
        except BaseException:
            _FORGOTTEN.discard(conversation_id)
            raise

    def _forget_entries(self, conversation_id: str) -> int:
        if not Path(self._db_path).exists():
            return 0
        conn = _safe_connect(self._db_path, check_same_thread=False)
        try:
            conn.execute("PRAGMA secure_delete = ON")
            columns = _columns(conn, "response_cache")
            if not columns:
                return 0
            if "conversation_id" in columns:
                gone = conn.execute(
                    "DELETE FROM response_cache WHERE conversation_id IN (?, ?)",
                    (conversation_id, _ORIGIN_UNKNOWN)).rowcount
            else:
                # A file its schema step could not give the column: every entry it holds is of an origin not known.
                gone = conn.execute("DELETE FROM response_cache").rowcount
            conn.commit()
            return gone
        finally:
            conn.close()

    def invalidate(self, cache_key: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            cache_key: SHA-256 cache key to remove

        Returns:
            True if entry was found and removed
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM response_cache WHERE cache_key = ?",
                (cache_key,)
            )
            conn.commit()
            removed = cursor.rowcount > 0
            if removed:
                logger.debug(f"Cache INVALIDATE: {cache_key[:12]}...")
            return removed
        finally:
            conn.close()

    def invalidate_model(self, model: str) -> int:
        """Remove all cache entries for a specific model.

        Args:
            model: Model name to clear

        Returns:
            Number of entries removed
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM response_cache WHERE model = ?",
                (model,)
            )
            conn.commit()
            count = cursor.rowcount
            logger.info(f"Cache INVALIDATE model={model}: {count} entries removed")
            return count
        finally:
            conn.close()

    def clear(self) -> int:
        """Remove all entries from the cache.

        Returns:
            Number of entries removed
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM response_cache")
            conn.commit()
            count = cursor.rowcount
            self._session_hits = 0
            self._session_misses = 0
            self._call_count = 0
            self._last_cache_hit = False
            logger.info(f"Cache CLEAR: {count} entries removed")
            return count
        finally:
            conn.close()

    def empty(self) -> int:
        """Erase every entry, overwritten; how many. A file that is not there holds none, and is not created. From
        then on, an answer whose lookup came before is never put: it may be drawn from what the emptying forgot."""
        with _MARKS_LOCK:
            _EMPTIED[0] += 1
        if not Path(self._db_path).exists():
            return 0
        return self.clear()

    # -------------------------------------------------------------------------
    # Statistiques
    # -------------------------------------------------------------------------

    def get_stats(self) -> CacheStats:
        """Get comprehensive cache statistics.

        Returns:
            CacheStats with current cache state
        """
        conn = self._get_connection()
        try:
            # Nombre d'entrees actives (non expirees)
            now = time.time()
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM response_cache "
                "WHERE created_at + ttl > ?",
                (now,)
            ).fetchone()["cnt"]

            # Repartition par modele
            rows = conn.execute(
                "SELECT model, COUNT(*) as cnt FROM response_cache "
                "WHERE created_at + ttl > ? "
                "GROUP BY model",
                (now,)
            ).fetchall()
            by_model = {row["model"]: row["cnt"] for row in rows}

            # Age de l'entree la plus ancienne
            oldest_row = conn.execute(
                "SELECT MIN(created_at) as oldest FROM response_cache "
                "WHERE created_at + ttl > ?",
                (now,)
            ).fetchone()
            oldest = now - oldest_row["oldest"] if oldest_row["oldest"] else 0.0

            # Taille approximative
            size_row = conn.execute(
                "SELECT SUM(LENGTH(response)) as total_size FROM response_cache "
                "WHERE created_at + ttl > ?",
                (now,)
            ).fetchone()
            total_size = size_row["total_size"] or 0

            # Taux de hit
            total_requests = self._session_hits + self._session_misses
            hit_rate = (
                self._session_hits / total_requests
                if total_requests > 0
                else 0.0
            )

            return CacheStats(
                total_entries=total,
                total_hits=self._session_hits,
                total_misses=self._session_misses,
                hit_rate=hit_rate,
                entries_by_model=by_model,
                oldest_entry=oldest,
                total_size_bytes=total_size,
            )

        finally:
            conn.close()

    @property
    def last_cache_hit(self) -> bool:
        """Whether the last get() was a cache hit."""
        return self._last_cache_hit

    @property
    def last_cache_key(self) -> str:
        """The cache key from the last get() call."""
        return self._last_cache_key

    @property
    def session_hits(self) -> int:
        """Total cache hits this session."""
        return self._session_hits

    @property
    def session_misses(self) -> int:
        """Total cache misses this session."""
        return self._session_misses

    @property
    def enabled(self) -> bool:
        """Whether the cache is currently enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the cache."""
        self._enabled = bool(value)
        logger.info(f"ResponseCache {'enabled' if self._enabled else 'disabled'}")

    @property
    def default_ttl(self) -> int:
        """Default TTL in seconds."""
        return self._default_ttl

    @default_ttl.setter
    def default_ttl(self, value: int) -> None:
        """Set default TTL (minimum 10 seconds)."""
        self._default_ttl = max(10, int(value))

    # -------------------------------------------------------------------------
    # Nettoyage interne
    # -------------------------------------------------------------------------

    def _cleanup_expired(self) -> None:
        """Delete entries whose TTL has expired."""
        conn = self._get_connection()
        try:
            now = time.time()
            cursor = conn.execute(
                "DELETE FROM response_cache WHERE created_at + ttl <= ?",
                (now,)
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.debug(
                    f"Cache cleanup: {cursor.rowcount} expired entries removed"
                )
        finally:
            conn.close()

    def _evict_lru(self, conn: sqlite3.Connection, count: int) -> None:
        """Evince les N entrees les moins recemment utilisees.

        Args:
            conn: Connexion SQLite existante
            count: Nombre d'entrees a evincer
        """
        if count <= 0:
            return
        conn.execute(
            """DELETE FROM response_cache
               WHERE cache_key IN (
                   SELECT cache_key FROM response_cache
                   ORDER BY last_accessed ASC
                   LIMIT ?
               )""",
            (count,)
        )
        logger.debug(f"Cache LRU eviction: {count} entries removed")

    # -------------------------------------------------------------------------
    # Utilitaires
    # -------------------------------------------------------------------------

    def entry_count(self) -> int:
        """Get total number of entries (including possibly expired).

        Returns:
            Total entry count
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM response_cache"
            ).fetchone()
            return row["cnt"]
        finally:
            conn.close()

    def get_entries_for_model(self, model: str) -> list[CacheEntry]:
        """Get all non-expired cache entries for a specific model.

        Args:
            model: Model name to filter by

        Returns:
            List of CacheEntry objects
        """
        conn = self._get_connection()
        try:
            now = time.time()
            rows = conn.execute(
                """SELECT * FROM response_cache
                   WHERE model = ? AND created_at + ttl > ?
                   ORDER BY last_accessed DESC""",
                (model, now)
            ).fetchall()
            return [
                CacheEntry(
                    cache_key=r["cache_key"],
                    model=r["model"],
                    prompt_hash=r["prompt_hash"],
                    query_hash=r["query_hash"],
                    response=r["response"],
                    task_type=r["task_type"] or "",
                    created_at=r["created_at"],
                    last_accessed=r["last_accessed"],
                    hit_count=r["hit_count"],
                    ttl=r["ttl"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_all_entries(self, limit: int = 100) -> list[CacheEntry]:
        """Get all non-expired cache entries, ordered by last access.

        Args:
            limit: Maximum entries to return (default 100)

        Returns:
            List of CacheEntry objects
        """
        conn = self._get_connection()
        try:
            now = time.time()
            rows = conn.execute(
                """SELECT * FROM response_cache
                   WHERE created_at + ttl > ?
                   ORDER BY last_accessed DESC
                   LIMIT ?""",
                (now, limit)
            ).fetchall()
            return [
                CacheEntry(
                    cache_key=r["cache_key"],
                    model=r["model"],
                    prompt_hash=r["prompt_hash"],
                    query_hash=r["query_hash"],
                    response=r["response"],
                    task_type=r["task_type"] or "",
                    created_at=r["created_at"],
                    last_accessed=r["last_accessed"],
                    hit_count=r["hit_count"],
                    ttl=r["ttl"],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def get_cached_models(self) -> list[str]:
        """Get list of models that have entries in the cache.

        Returns:
            Sorted list of model names
        """
        conn = self._get_connection()
        try:
            now = time.time()
            rows = conn.execute(
                """SELECT DISTINCT model FROM response_cache
                   WHERE created_at + ttl > ?
                   ORDER BY model""",
                (now,)
            ).fetchall()
            return [r["model"] for r in rows]
        finally:
            conn.close()

    def warm(
        self,
        entries: list[dict[str, str]],
        ttl: int | None = None,
    ) -> int:
        """Pre-populate cache with known question/response pairs.

        Useful for warming the cache with frequently asked queries
        to provide instant responses on first use.

        Args:
            entries: List of dicts with keys:
                - model: Ollama model name
                - system_prompt: System prompt text
                - user_content: User query
                - response: Expected response
                - task_type: Optional task type (default "warmed")
            ttl: Optional TTL override for warmed entries

        Returns:
            Number of entries successfully warmed
        """
        if not self._enabled:
            return 0

        warmed = 0
        for entry in entries:
            try:
                model = entry["model"]
                system_prompt = entry["system_prompt"]
                user_content = entry["user_content"]
                response = entry["response"]
                task_type = entry.get("task_type", "warmed")

                self.put(
                    model=model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    response=response,
                    task_type=task_type,
                    ttl=ttl,
                )
                warmed += 1
            except (KeyError, Exception) as e:
                logger.warning(f"Cache warm skip: {e}")
                continue

        logger.info(f"Cache warmed: {warmed}/{len(entries)} entries")
        return warmed

    @property
    def max_entries(self) -> int:
        """Maximum cache entries."""
        return self._max_entries


# =============================================================================
# SINGLETON
# =============================================================================

response_cache = ResponseCache()


def forget_conversation(conversation_id: str) -> int:
    """Forget what the response cache holds of a deleted conversation; how many entries."""
    return response_cache.forget_conversation(conversation_id)


def unforget_conversation(conversation_id: str) -> None:
    """Cache the conversation again: its deletion was refused, and it stays."""
    _FORGOTTEN.discard(conversation_id)
