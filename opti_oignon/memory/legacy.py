#!/usr/bin/env python3
"""
MEMORY — OPTI-OIGNON 1.4.0
============================

Cross-conversation memory: extracts, stores, and retrieves persistent
facts about the user across conversations.

This is the biggest gap vs. Claude/ChatGPT — every local conversation
starts from zero. This module bridges that gap by:

1. Extracting facts from conversations via a lightweight LLM call
2. Storing them in SQLite with categories and confidence scores
3. Deduplicating with string similarity (no embeddings needed)
4. Providing retrieval APIs for injection into system prompts (Session 11)

Architecture:
    - MemoryFact: dataclass for a single fact
    - MemoryManager: CRUD + extraction + deduplication
    - SQLite backend (memories.db), same pattern as conversation.py
    - Model fallback chain (same as context_summary.py)
    - Robust JSON parsing for LLM output

Author: Léon
"""

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

# S125: Data-at-rest encryption
try:
    from ..encryption import decrypt_field as _decrypt
    from ..encryption import encrypt_field as _encrypt
    _HAS_ENCRYPTION = True
except ImportError:
    _HAS_ENCRYPTION = False
    def _encrypt(v: str) -> str: return v  # type: ignore[misc]
    def _decrypt(v: str) -> str: return v  # type: ignore[misc]

# Import Ollama — needed for fact extraction
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from opti_oignon.db_utils import safe_connect

# Import conversation_manager to access messages
try:
    from ..conversation import conversation_manager
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False
    conversation_manager = None
    logger.warning("conversation_manager non disponible")


# =============================================================================
# CONSTANTES
# =============================================================================

VALID_CATEGORIES = {
    "preference",   # User preferences (language, style, format)
    "project",      # Projets en cours
    "skill",        # Skills, languages, tools mastered
    "personal",     # Personal info (name, studies, location)
    "tool",         # Technical environment (OS, editor, models)
    "context",      # General context, catch-all
}

DEFAULT_CATEGORY = "context"


# =============================================================================
# PROMPT D'EXTRACTION
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are a fact extractor. Your task is to analyze a conversation and extract PERSISTENT facts about the user.

## RULES
1. Extract facts that would be useful to remember for FUTURE conversations
2. Focus on: preferences, skills, projects, personal info, tools, environment
3. IGNORE: ephemeral questions, generic pleasantries, one-time requests
4. Each fact should be a SHORT, standalone sentence (max 15 words)
5. Categorize each fact as one of: preference, project, skill, personal, tool, context
6. Output ONLY a JSON array, no preamble, no explanation
7. If no memorable facts exist, output: []

## CATEGORIES
- **preference**: User preferences (language, code style, formatting, workflow)
- **project**: Projects the user is working on, research topics
- **skill**: Programming languages, frameworks, domains of expertise
- **personal**: Name, studies, location, occupation
- **tool**: OS, editor, software, hardware, models used
- **context**: General context that doesn't fit other categories

## OUTPUT FORMAT (strict JSON)
[
  {"fact": "User prefers French comments in code", "category": "preference"},
  {"fact": "User works with R and tidyverse", "category": "skill"}
]

## EXAMPLES OF GOOD EXTRACTIONS
- "User is an M2 student in IMABEE program" (personal)
- "User does bioacoustic research in Panama" (project)
- "User runs Kubuntu with Ollama" (tool)
- "User prefers dark mode interfaces" (preference)

## EXAMPLES OF WHAT TO SKIP
- "User asked about Python syntax" (too ephemeral)
- "User said hello" (pleasantry)
- "User wants to fix a bug" (one-time request, unless recurring)"""


def _reply_text(response: Any) -> str:
    """Pull the assistant text from an ollama-style chat response.

    Handles the dict shape and the object shape returned by newer ollama-python
    (a ChatResponse is not subscriptable), so an object-form response no longer
    raises a TypeError that the extraction would swallow into an empty result.
    Mirrors the helper in memory/extraction.py (the S189 dict-vs-object class).
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(response.get("content") or "")
    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "")
    return ""


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class MemoryFact:
    """A persistent fact extracted from a conversation.

    Attributes:
        id: UUID unique
        fact: Le fait en une phrase courte
        category: Fact category
        source_conversation_id: Source conversation ID
        created_at: Creation date ISO
        updated_at: ISO update date
        confidence: Score de confiance (0.0-1.0)
        active: Active or disabled by user
    """
    id: str
    fact: str
    category: str
    source_conversation_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    confidence: float = 1.0
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# MEMORY MANAGER
# =============================================================================

class MemoryManager:
    """Manages persistent cross-conversation memory.

    Extracts facts from conversations, stores them in SQLite,
    and provides retrieval APIs for injection into
    system prompts.

    Thread-safe via verrou interne.
    """

    # --- Model configuration ---
    EXTRACTION_MODEL = "qwen3:8b"
    FALLBACK_MODELS = [
        "qwen3:8b",
        "nemotron-3-nano:8b",
        "qwen3:4b",
        "qwen3:1.7b",
    ]
    EXTRACTION_TEMPERATURE = 0.2    # Deterministic for reliable JSON
    MAX_EXTRACTION_TOKENS = 600     # ~400 tokens de sortie max
    EXTRACTION_TIMEOUT = 20         # Secondes
    MAX_INPUT_MESSAGES = 30         # Derniers N messages pour extraction
    MAX_INPUT_TOKENS = 4000         # Budget tokens pour l'input

    # --- Deduplication thresholds ---
    DUPLICATE_THRESHOLD = 0.85      # Au-dessus → skip (doublon)
    MERGE_THRESHOLD = 0.70          # Au-dessus → update existant

    def __init__(self, db_path: Path | None = None):
        """Initialize the memory manager.

        Args:
            db_path: Path to SQLite database (default: DATA_DIR/memories.db)
        """
        self._db_path = db_path or (DATA_DIR / "memories.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Model cache (same pattern as context_summary)
        self._available_model: str | None = None
        self._model_checked_at: float = 0.0
        self._model_cache_ttl: float = 300.0  # 5 min

        # Schema initialization
        self._init_db()
        logger.info(f"MemoryManager initialized: {self._db_path}")

    # -----------------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Create a configured SQLite connection.

        S136 audit fix: routes through get_encrypted_connection() for
        SQLCipher support when available.
        """
        conn = safe_connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create the memories table if it does not exist."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        fact TEXT NOT NULL,
                        category TEXT NOT NULL DEFAULT 'context',
                        source_conversation_id TEXT DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        confidence REAL DEFAULT 1.0,
                        active INTEGER DEFAULT 1
                    );

                    CREATE INDEX IF NOT EXISTS idx_memories_category
                        ON memories(category);
                    CREATE INDEX IF NOT EXISTS idx_memories_active
                        ON memories(active);
                """)
                conn.commit()
            except Exception as e:
                logger.error(f"Memory DB initialization error: {e}")
                raise
            finally:
                conn.close()

    def _row_to_fact(self, row: sqlite3.Row) -> MemoryFact:
        """Convert a SQLite row to a MemoryFact."""
        return MemoryFact(
            id=row["id"],
            fact=_decrypt(row["fact"]),  # S125: transparent decryption
            category=row["category"],
            source_conversation_id=row["source_conversation_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            confidence=row["confidence"],
            active=bool(row["active"]),
        )

    # -----------------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------------

    def add_fact(
        self,
        fact: str,
        category: str,
        source_conversation_id: str = "",
        confidence: float = 1.0,
    ) -> MemoryFact | None:
        """Add a fact to memory.

        Args:
            fact: The fact to store
            category: Category (preference, skill, project, etc.)
            source_conversation_id: Source conversation ID
            confidence: Score de confiance (0.0-1.0)

        Returns:
            The created MemoryFact, or None on error
        """
        # Validation
        fact = fact.strip()
        if not fact:
            logger.warning("Empty fact, ignored")
            return None

        category = category.strip().lower()
        if category not in VALID_CATEGORIES:
            logger.warning(f"Unknown category '{category}', fallback → context")
            category = DEFAULT_CATEGORY

        confidence = max(0.0, min(1.0, confidence))

        now = datetime.now().isoformat()
        fact_id = str(uuid.uuid4())

        memory_fact = MemoryFact(
            id=fact_id,
            fact=fact,
            category=category,
            source_conversation_id=source_conversation_id,
            created_at=now,
            updated_at=now,
            confidence=confidence,
            active=True,
        )

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """INSERT INTO memories
                       (id, fact, category, source_conversation_id,
                        created_at, updated_at, confidence, active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory_fact.id,
                        _encrypt(memory_fact.fact),  # S125: encrypt at rest
                        memory_fact.category,
                        memory_fact.source_conversation_id,
                        memory_fact.created_at,
                        memory_fact.updated_at,
                        memory_fact.confidence,
                        1,
                    ),
                )
                conn.commit()
                logger.info(
                    f"Fact added [{category}]: {fact[:60]}..."
                    if len(fact) > 60 else f"Fact added [{category}]: {fact}"
                )
                return memory_fact
            except Exception as e:
                logger.error(f"Erreur ajout fait: {e}")
                return None
            finally:
                conn.close()

    def get_fact(self, fact_id: str) -> MemoryFact | None:
        """Retrieve a fact by ID.

        Args:
            fact_id: UUID du fait

        Returns:
            MemoryFact ou None
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (fact_id,)
                ).fetchone()
                return self._row_to_fact(row) if row else None
            except Exception as e:
                logger.error(f"Erreur lecture fait {fact_id}: {e}")
                return None
            finally:
                conn.close()

    def get_all_facts(
        self,
        active_only: bool = True,
        category: str | None = None,
    ) -> list[MemoryFact]:
        """Retrieve all facts.

        Args:
            active_only: Ne retourner que les faits actifs
            category: Filter by category (optional)

        Returns:
            Liste de MemoryFact
        """
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT * FROM memories WHERE 1=1"
                params: list = []

                if active_only:
                    query += " AND active = 1"
                if category:
                    query += " AND category = ?"
                    params.append(category)

                query += " ORDER BY category, created_at"
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_fact(r) for r in rows]
            except Exception as e:
                logger.error(f"Erreur lecture faits: {e}")
                return []
            finally:
                conn.close()

    def update_fact(
        self,
        fact_id: str,
        new_fact: str | None = None,
        new_category: str | None = None,
        new_confidence: float | None = None,
    ) -> bool:
        """Update an existing fact.

        Args:
            fact_id: UUID of the fact to modify
            new_fact: Nouveau texte (optionnel)
            new_category: New category (optional)
            new_confidence: Nouveau score de confiance (optionnel)

        Returns:
            True if updated, False otherwise
        """
        updates = []
        params: list = []

        if new_fact is not None:
            updates.append("fact = ?")
            params.append(new_fact.strip())
        if new_category is not None:
            cat = new_category.strip().lower()
            if cat not in VALID_CATEGORIES:
                cat = DEFAULT_CATEGORY
            updates.append("category = ?")
            params.append(cat)
        if new_confidence is not None:
            updates.append("confidence = ?")
            params.append(max(0.0, min(1.0, new_confidence)))

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(fact_id)

        with self._lock:
            conn = self._get_connection()
            try:
                result = conn.execute(
                    "UPDATE memories SET {} WHERE id = ?".format(
                        ", ".join(updates)
                    ),
                    params,
                )
                conn.commit()
                updated = result.rowcount > 0
                if updated:
                    logger.info(f"Fact updated: {fact_id[:8]}...")
                return updated
            except Exception as e:
                logger.error(f"Fact update error {fact_id}: {e}")
                return False
            finally:
                conn.close()

    def deactivate_fact(self, fact_id: str) -> bool:
        """Deactivate a fact (soft-delete).

        Args:
            fact_id: UUID du fait

        Returns:
            True if deactivated, False otherwise
        """
        with self._lock:
            conn = self._get_connection()
            try:
                result = conn.execute(
                    """UPDATE memories
                       SET active = 0, updated_at = ?
                       WHERE id = ? AND active = 1""",
                    (datetime.now().isoformat(), fact_id),
                )
                conn.commit()
                deactivated = result.rowcount > 0
                if deactivated:
                    logger.info(f"Fact deactivated: {fact_id[:8]}...")
                return deactivated
            except Exception as e:
                logger.error(f"Fact deactivation error {fact_id}: {e}")
                return False
            finally:
                conn.close()

    def activate_fact(self, fact_id: str) -> bool:
        """Reactivate a deactivated fact.

        Args:
            fact_id: UUID du fait

        Returns:
            True if reactivated, False otherwise
        """
        with self._lock:
            conn = self._get_connection()
            try:
                result = conn.execute(
                    """UPDATE memories
                       SET active = 1, updated_at = ?
                       WHERE id = ? AND active = 0""",
                    (datetime.now().isoformat(), fact_id),
                )
                conn.commit()
                return result.rowcount > 0
            except Exception as e:
                logger.error(f"Fact reactivation error {fact_id}: {e}")
                return False
            finally:
                conn.close()

    def delete_fact(self, fact_id: str) -> bool:
        """Permanently delete a fact.

        Args:
            fact_id: UUID du fait

        Returns:
            True if deleted, False otherwise
        """
        with self._lock:
            conn = self._get_connection()
            try:
                result = conn.execute(
                    "DELETE FROM memories WHERE id = ?", (fact_id,)
                )
                conn.commit()
                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"Fact deleted: {fact_id[:8]}...")
                return deleted
            except Exception as e:
                logger.error(f"Erreur suppression fait {fact_id}: {e}")
                return False
            finally:
                conn.close()

    def count_facts(self, active_only: bool = True) -> int:
        """Count the number of facts in memory.

        Args:
            active_only: Ne compter que les faits actifs

        Returns:
            Nombre de faits
        """
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT COUNT(*) as cnt FROM memories"
                if active_only:
                    query += " WHERE active = 1"
                row = conn.execute(query).fetchone()
                return row["cnt"] if row else 0
            except Exception as e:
                logger.error(f"Erreur comptage faits: {e}")
                return 0
            finally:
                conn.close()

    def clear_all(self) -> int:
        """Delete all facts (for tests).

        Returns:
            Number of deleted facts
        """
        with self._lock:
            conn = self._get_connection()
            try:
                result = conn.execute("DELETE FROM memories")
                conn.commit()
                count = result.rowcount
                logger.warning(f"Memory cleared: {count} facts deleted")
                return count
            except Exception as e:
                logger.error(f"Memory clear error: {e}")
                return 0
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------------

    def deduplicate(
        self,
        new_fact: str,
        threshold: float | None = None,
    ) -> tuple[str | None, float]:
        """Check if a similar fact already exists.

        Uses difflib.SequenceMatcher for textual similarity.

        Args:
            new_fact: The new fact to check
            threshold: Similarity threshold (default: DUPLICATE_THRESHOLD)

        Returns:
            Tuple (fact_id or None, max similarity score)
            - fact_id if duplicate found (score >= threshold), None otherwise
        """
        if threshold is None:
            threshold = self.DUPLICATE_THRESHOLD

        existing = self.get_all_facts(active_only=True)
        if not existing:
            return None, 0.0

        new_lower = new_fact.lower().strip()
        best_match_id = None
        best_score = 0.0

        for fact in existing:
            score = SequenceMatcher(
                None,
                new_lower,
                fact.fact.lower().strip(),
            ).ratio()

            if score > best_score:
                best_score = score
                best_match_id = fact.id

        if best_score >= threshold:
            return best_match_id, best_score

        return None, best_score

    # -----------------------------------------------------------------------
    # Extraction via LLM
    # -----------------------------------------------------------------------

    def _find_available_model(self) -> str | None:
        """Find an available model for extraction.

        Same pattern as context_summary: cache + fallback chain.

        Returns:
            Model name, or None
        """
        now = time.time()
        if (
            self._available_model
            and (now - self._model_checked_at) < self._model_cache_ttl
        ):
            return self._available_model

        if not OLLAMA_AVAILABLE:
            return None

        try:
            models_response = ollama.list()
            available_names = set()

            if hasattr(models_response, "models"):
                for m in models_response.models:
                    available_names.add(
                        m.model if hasattr(m, "model") else str(m)
                    )
            elif isinstance(models_response, dict) and "models" in models_response:
                for m in models_response["models"]:
                    name = m.get("model", m.get("name", ""))
                    available_names.add(name)

            # Search in order of preference
            for candidate in self.FALLBACK_MODELS:
                if candidate in available_names:
                    self._available_model = candidate
                    self._model_checked_at = now
                    logger.info(f"Memory extraction model: {candidate}")
                    return candidate
                # Partial match by prefix
                for avail in available_names:
                    if avail.startswith(candidate.split(":")[0] + ":"):
                        self._available_model = avail
                        self._model_checked_at = now
                        logger.info(
                            f"Extraction model (partial match): {avail}"
                        )
                        return avail

            # Fallback : premier disponible
            if available_names:
                first = sorted(available_names)[0]
                self._available_model = first
                self._model_checked_at = now
                logger.warning(
                    f"No preferred extraction model, using: {first}"
                )
                return first

        except Exception as e:
            logger.error(f"Extraction model search error: {e}")

        return None

    def _format_messages_for_extraction(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Formate les messages pour le prompt d'extraction.

        Args:
            messages: Liste de {role, content}

        Returns:
            Formatted text User:/Assistant:
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "").strip()
            if content and role in ("User", "Assistant"):
                # Truncate very long messages (code, etc.)
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estimation rapide du nombre de tokens."""
        return len(text) // 4

    def _truncate_messages(
        self,
        messages: list[dict[str, str]],
        max_messages: int,
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """Tronque les messages pour l'extraction.

        Keep the most recent messages (most relevant).

        Args:
            messages: Tous les messages
            max_messages: Nombre max de messages
            max_tokens: Budget tokens max

        Returns:
            Truncated messages
        """
        # First limit the number of messages (most recent)
        truncated = messages[-max_messages:] if len(messages) > max_messages else list(messages)

        # Puis limiter les tokens
        total = sum(
            self._estimate_tokens(m.get("content", "")) for m in truncated
        )
        while total > max_tokens and len(truncated) > 2:
            removed = truncated.pop(0)
            total -= self._estimate_tokens(removed.get("content", ""))

        return truncated

    def _clean_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from qwen3 responses."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
        return cleaned.strip()

    def _parse_extraction_response(self, raw: str) -> list[dict[str, str]]:
        """Parse the JSON response from the extraction model.

        Handles common cases: markdown fences, preamble, malformed JSON.

        Args:
            raw: Raw model response

        Returns:
            Liste de {"fact": ..., "category": ...} dicts
        """
        if not raw:
            return []

        # Nettoyer think tags
        cleaned = self._clean_think_tags(raw)

        # Retirer les fences markdown
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*", "", cleaned)

        # Trouver le premier [ et le dernier ]
        start = cleaned.find("[")
        end = cleaned.rfind("]")

        if start == -1 or end == -1 or end <= start:
            logger.warning(f"No JSON array found in response: {raw[:100]}")
            return []

        json_str = cleaned[start:end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Malformed JSON: {e} — raw: {json_str[:200]}")
            return []

        if not isinstance(data, list):
            logger.warning(f"Non-list response: {type(data)}")
            return []

        # Validate each entry
        valid = []
        for item in data:
            if not isinstance(item, dict):
                continue
            fact = item.get("fact", "").strip()
            category = item.get("category", DEFAULT_CATEGORY).strip().lower()

            if not fact or len(fact) < 5:
                continue
            if category not in VALID_CATEGORIES:
                category = DEFAULT_CATEGORY

            valid.append({"fact": fact, "category": category})

        return valid

    def extract_facts(
        self,
        conversation_id: str,
        model: str | None = None,
    ) -> list[dict[str, str]]:
        """Extract facts from a conversation via LLM.

        Sends recent messages to a lightweight model with the
        extraction prompt, and parses the JSON response.

        Args:
            conversation_id: Conversation ID to analyze
            model: Model override (otherwise auto-detection)

        Returns:
            Liste de {"fact": ..., "category": ...} dicts
        """
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama non disponible — extraction impossible")
            return []

        if not CONVERSATION_AVAILABLE or not conversation_manager:
            logger.warning("ConversationManager non disponible")
            return []

        # Retrieve messages
        messages = conversation_manager.get_context_messages(conversation_id)
        if not messages or len(messages) < 2:
            logger.info(
                f"Conversation {conversation_id[:8]} trop courte pour extraction"
            )
            return []

        # Tronquer
        truncated = self._truncate_messages(
            messages,
            self.MAX_INPUT_MESSAGES,
            self.MAX_INPUT_TOKENS,
        )

        # Formater
        formatted = self._format_messages_for_extraction(truncated)
        if not formatted.strip():
            return []

        # Model
        extraction_model = model or self._find_available_model()
        if not extraction_model:
            logger.warning("No model available for extraction")
            return []

        input_tokens = self._estimate_tokens(formatted)
        logger.info(
            f"Memory extraction: conv {conversation_id[:8]}, "
            f"{len(truncated)} messages (~{input_tokens}t) → {extraction_model}"
        )

        # Appel LLM
        start_time = time.time()
        try:
            response = ollama.chat(
                model=extraction_model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Extract memorable facts from this conversation:\n\n"
                            + formatted
                        ),
                    },
                ],
                options={
                    "temperature": self.EXTRACTION_TEMPERATURE,
                    "num_predict": self.MAX_EXTRACTION_TOKENS,
                },
            )

            elapsed = time.time() - start_time
            raw = _reply_text(response)

            if elapsed > self.EXTRACTION_TIMEOUT:
                logger.warning(
                    f"Extraction lente: {elapsed:.1f}s "
                    f"(timeout = {self.EXTRACTION_TIMEOUT}s)"
                )

            facts = self._parse_extraction_response(raw)
            logger.info(
                f"Extraction complete: {len(facts)} facts in {elapsed:.1f}s"
            )
            return facts

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Erreur extraction ({elapsed:.1f}s): {e}")
            return []

    def extract_facts_from_messages(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> list[dict[str, str]]:
        """Extract facts from a direct list of messages.

        Utile pour les tests sans conversation_manager.

        Args:
            messages: Liste de {role, content}
            model: Model override

        Returns:
            Liste de {"fact": ..., "category": ...} dicts
        """
        if not OLLAMA_AVAILABLE:
            return []

        if not messages or len(messages) < 2:
            return []

        truncated = self._truncate_messages(
            messages,
            self.MAX_INPUT_MESSAGES,
            self.MAX_INPUT_TOKENS,
        )
        formatted = self._format_messages_for_extraction(truncated)
        if not formatted.strip():
            return []

        extraction_model = model or self._find_available_model()
        if not extraction_model:
            return []

        start_time = time.time()
        try:
            response = ollama.chat(
                model=extraction_model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Extract memorable facts from this conversation:\n\n"
                            + formatted
                        ),
                    },
                ],
                options={
                    "temperature": self.EXTRACTION_TEMPERATURE,
                    "num_predict": self.MAX_EXTRACTION_TOKENS,
                },
            )
            elapsed = time.time() - start_time
            raw = _reply_text(response)
            facts = self._parse_extraction_response(raw)
            logger.info(
                f"Extraction (direct): {len(facts)} faits en {elapsed:.1f}s"
            )
            return facts
        except Exception as e:
            logger.error(f"Erreur extraction directe: {e}")
            return []

    # -----------------------------------------------------------------------
    # Combined extraction + storage
    # -----------------------------------------------------------------------

    def extract_and_store(
        self,
        conversation_id: str,
        model: str | None = None,
    ) -> int:
        """Extract facts from a conversation and store new ones.

        Full pipeline: extraction → deduplication → storage.

        Args:
            conversation_id: Conversation ID
            model: Model override

        Returns:
            Number of new facts added
        """
        facts = self.extract_facts(conversation_id, model=model)
        if not facts:
            return 0

        added = 0
        for fact_data in facts:
            fact_text = fact_data["fact"]
            category = fact_data["category"]

            # Deduplication
            existing_id, score = self.deduplicate(fact_text)

            if existing_id and score >= self.DUPLICATE_THRESHOLD:
                # Doublon exact → skip
                logger.debug(
                    f"Duplicate ignored (score={score:.2f}): {fact_text[:50]}"
                )
                continue

            if existing_id and score >= self.MERGE_THRESHOLD:
                # Similar → update existing fact
                self.update_fact(
                    existing_id,
                    new_fact=fact_text,
                    new_confidence=min(1.0, score + 0.1),
                )
                logger.info(
                    f"Fact merged (score={score:.2f}): {fact_text[:50]}"
                )
                continue

            # Nouveau fait → ajout
            result = self.add_fact(
                fact=fact_text,
                category=category,
                source_conversation_id=conversation_id,
            )
            if result:
                added += 1

        logger.info(
            f"extract_and_store: {len(facts)} extraits, "
            f"{added} new facts added for conv {conversation_id[:8]}"
        )
        return added

    # -----------------------------------------------------------------------
    # Formatting for injection (Session 11 preparation)
    # -----------------------------------------------------------------------

    def format_for_prompt(
        self,
        max_tokens: int = 500,
        active_only: bool = True,
    ) -> str:
        """Formate les faits pour injection dans un system prompt.

        Organize by category for easy model reading.

        Args:
            max_tokens: Budget tokens maximum
            active_only: Ne prendre que les faits actifs

        Returns:
            Formatted text ready for injection, or empty string
        """
        facts = self.get_all_facts(active_only=active_only)
        if not facts:
            return ""

        # Organize by category
        by_category: dict[str, list[str]] = {}
        for fact in facts:
            by_category.setdefault(fact.category, []).append(fact.fact)

        # Display priority order
        category_order = [
            "personal", "skill", "tool", "project", "preference", "context"
        ]
        category_labels = {
            "personal": "About the user",
            "skill": "Skills & expertise",
            "tool": "Tools & environment",
            "project": "Projects",
            "preference": "Preferences",
            "context": "Other context",
        }

        lines = ["[User Memory]"]
        total_tokens = self._estimate_tokens(lines[0])

        for cat in category_order:
            if cat not in by_category:
                continue
            label = category_labels.get(cat, cat.capitalize())
            header = f"\n{label}:"
            header_tokens = self._estimate_tokens(header)

            if total_tokens + header_tokens > max_tokens:
                break

            lines.append(header)
            total_tokens += header_tokens

            for fact_text in by_category[cat]:
                line = f"- {fact_text}"
                line_tokens = self._estimate_tokens(line)
                if total_tokens + line_tokens > max_tokens:
                    break
                lines.append(line)
                total_tokens += line_tokens

        return "\n".join(lines)


# =============================================================================
# INSTANCE GLOBALE
# =============================================================================

memory_manager = MemoryManager()

# Convenience exports
extract_facts = memory_manager.extract_facts
extract_and_store = memory_manager.extract_and_store
add_fact = memory_manager.add_fact
get_all_facts = memory_manager.get_all_facts
