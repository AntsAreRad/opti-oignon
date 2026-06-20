#!/usr/bin/env python3
"""
CONVERSATION - OPTI-OIGNON 1.3.0
=================================

Multi-turn conversation management with SQLite storage.

Replaces the single-turn JSON history with proper conversation
tracking, supporting multi-turn exchanges, search, and migration
from the legacy history format.

Features:
    - SQLite-backed conversation and message storage
    - CRUD operations for conversations
    - Token estimation per message (via context_manager)
    - Full-text search across conversations
    - Migration from legacy JSON history
    - Thread-safe database access
    - Ollama-ready message format output

Author: Leon
"""

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import DATA_DIR

from opti_oignon.db_utils import safe_connect

# EXP-01 (S194): real app version for export metadata (was hardcoded "1.6.3")
try:
    from .__version__ import __version__ as _APP_VERSION
except ImportError:  # pragma: no cover - standalone module loading
    _APP_VERSION = "0.0.0"

logger = logging.getLogger(__name__)

# ============================================================================
# S125: Data-at-rest encryption
# ============================================================================

try:
    from .encryption import encrypt_field as _encrypt, decrypt_field as _decrypt
    _HAS_ENCRYPTION = True
except ImportError:
    _HAS_ENCRYPTION = False
    def _encrypt(v: str) -> str: return v  # type: ignore[misc]
    def _decrypt(v: str) -> str: return v  # type: ignore[misc]

# ============================================================================
# Estimation de tokens - import avec fallback
# ============================================================================

# Tente d'importer depuis context_manager, sinon approximation simple
try:
    from .context_manager import estimate_tokens as _cm_estimate_tokens
    _HAS_CONTEXT_MANAGER = True
except ImportError:
    _HAS_CONTEXT_MANAGER = False
    logger.warning("context_manager non disponible, estimation tokens par approximation")


def _estimate_tokens(text: str, model: str | None = None) -> int:
    """Estime le nombre de tokens of a texte.

    Utilise context_manager si disponible, sinon len(text) / 4.
    """
    if not text:
        return 0
    if _HAS_CONTEXT_MANAGER:
        try:
            return _cm_estimate_tokens(text, model)
        except Exception:
            pass
    # Fallback: approximation grossiere
    return int(len(text) / 4)


# S138: Allowed column names for dynamic UPDATE queries
_CONV_UPDATE_COLS = frozenset({
    "title", "model", "task_type", "preset", "metadata", "updated_at",
})


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Message:
    """Un message dans une conversation.

    Attributes:
        id: Identifiant auto-incremente (SQLite)
        conversation_id: Parent conversation UUID
        role: 'system', 'user', ou 'assistant'
        content: Contenu du message
        timestamp: Horodatage ISO
        token_estimate: Estimation du nombre de tokens
        model: Model utilise (pour les messages assistant)
        metadata: Data supplementaires en JSON
    """
    id: int
    conversation_id: str
    role: str
    content: str
    timestamp: str
    token_estimate: int = 0
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_ollama_format(self) -> dict[str, str]:
        """Return the format expected by ollama.chat (role + content)."""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """Une conversation multi-tour.

    Attributes:
        id: UUID unique
        title: Conversation title
        created_at: Date de creation ISO
        updated_at: Derniere mise a jour ISO
        model: Dernier model utilise
        task_type: Dernier type de tache detecte
        preset: Preset utilise
        metadata: Data supplementaires en JSON
        messages: List of messages (loaded on demand)
    """
    id: str
    title: str
    created_at: str
    updated_at: str
    model: str | None = None
    task_type: str | None = None
    preset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)

    @property
    def message_count(self) -> int:
        """Nombre de messages loaded."""
        return len(self.messages)

    @property
    def total_tokens(self) -> int:
        """Somme des tokens estimes pour tous les messages loaded."""
        return sum(m.token_estimate for m in self.messages)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without messages for list views)."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "task_type": self.task_type,
            "preset": self.preset,
            "metadata": self.metadata,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
        }


# ============================================================================
# S199: Veilid sync publish hook (SYN-01, Bloc 0 lot 1)
# ============================================================================


def _sync_owner_id() -> str:
    """The owning user for a sync payload (the ``effective_user_id`` pattern).

    Conversations are install-scoped today (no ``user_id`` column), so this
    resolves to the single-user default. Scoping rides in the PAYLOAD, never
    the key: the conversation uuid stays the stable per-kind key on every
    device, and this payload field is the migration seam the memory and skills
    lots will carry the same way.
    """
    try:
        from opti_oignon.user_isolation import effective_user_id

        return effective_user_id(None)
    except Exception:  # pragma: no cover - isolation module is optional here
        return "local"


def _sync_publish_conversation(
    conv_id: str,
    payload_fn: Callable[[], dict[str, Any] | None] | None = None,
    *,
    deleted: bool = False,
    updated_at: str = "",
) -> None:
    """Journal a conversation change for Veilid sync, best-effort (SYN-01).

    Called by the conversation save paths AFTER the domain commit, while the
    manager lock is still held. ``payload_fn`` is a zero-arg callable building
    the full-state snapshot; it runs INSIDE this hook's protection, and only
    after the availability probe passes, so when sync is absent the save pays
    nothing at all (no snapshot reads, no journal append). The contract
    (ROADMAP_SYNC_CYCLE, Bloc 0):

    - A snapshot or journalling failure must never break the save: any error
      is logged and swallowed (at-least-once on the next save).
    - No-op when the optional veilid framework is absent
      (``guard.veilid_available`` is the cheap probe; the heavy framework is
      never imported on this path -- the sync modules below are pure Python).
    - Mode-free: producing and journalling are local-disk operations permitted
      in ANY mode (the documented ``producers.py`` posture). There is
      deliberately no mode gate here; only the wire is Daily-gated, downstream
      at the engine/guard.

    Clock discipline: the next clock for a local edit is the highest clock this
    device has journalled for the key, plus one (an unseen key yields 0, so the
    first clock is 1). Running under the manager lock serialises mint + append
    per process, keeping same-key clocks strictly monotonic. Lock order is
    conversation lock -> feed lock, and the feed never calls back into domain
    code, so the order is acyclic.
    """
    try:
        from opti_oignon.veilid.guard import veilid_available

        if not veilid_available():
            return
        payload: dict[str, Any] | None = None
        if not deleted:
            payload = payload_fn() if payload_fn is not None else None
            if payload is None:
                # The snapshot could not be built (row gone mid-save).
                # Publishing an empty non-tombstone payload would wipe the
                # conversation on peers under LWW -- skip instead.
                logger.debug(
                    "sync publish skipped for %s: no snapshot available", conv_id
                )
                return
        from opti_oignon.veilid.records import RecordKind
        from opti_oignon.veilid.sync_engine import get_sync_engine

        engine = get_sync_engine()
        clock = engine.current_clock(RecordKind.CONVERSATION, conv_id) + 1
        engine.publish_conversation(
            conv_id,
            payload,
            clock=clock,
            deleted=deleted,
            updated_at=updated_at,
        )
    except Exception:
        logger.warning(
            "veilid sync publish failed for conversation %s (save unaffected)",
            conv_id,
            exc_info=True,
        )


# ============================================================================
# CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """
    Manager de conversations multi-tour avec stockage SQLite.

    Provides CRUD operations, search, and migration
    depuis l'ancien format JSON. Thread-safe via un verrou global.

    Usage:
        manager = ConversationManager()
        conv = manager.create_conversation(title="Test")
        manager.add_message(conv.id, "user", "Bonjour")
        manager.add_message(conv.id, "assistant", "Salut!", model="qwen3-coder:30b")
        messages = manager.get_context_messages(conv.id)
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize the manager.

        Args:
            db_path: Chemin vers la base SQLite (default: DATA_DIR/conversations.db)
        """
        self._db_path = db_path or (DATA_DIR / "conversations.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Initialisation du schema
        self._init_db()
        logger.info(f"ConversationManager initialise: {self._db_path}")

    # -----------------------------------------------------------------------
    # Connexion et schema
    # -----------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Create a configured SQLite connection.

        S136 audit fix: routes through get_encrypted_connection() for
        SQLCipher support when available. Each call creates a new
        connection for multi-thread compatibility.
        """
        conn = safe_connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        title TEXT DEFAULT 'New conversation',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        model TEXT,
                        task_type TEXT,
                        preset TEXT,
                        metadata TEXT DEFAULT '{}'
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        token_estimate INTEGER DEFAULT 0,
                        model TEXT,
                        metadata TEXT DEFAULT '{}',
                        FOREIGN KEY (conversation_id)
                            REFERENCES conversations(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_messages_conv
                        ON messages(conversation_id);
                    CREATE INDEX IF NOT EXISTS idx_conversations_updated
                        ON conversations(updated_at DESC);
                """)
                conn.commit()
            except Exception as e:
                logger.error(f"Error initializing DB: {e}")
                raise
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Helpers internes
    # -----------------------------------------------------------------------

    def _row_to_conversation(self, row: sqlite3.Row) -> Conversation:
        """Convert a SQLite row to a Conversation object (without messages)."""
        metadata = {}
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        return Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            model=row["model"],
            task_type=row["task_type"],
            preset=row["preset"],
            metadata=metadata,
            messages=[],
        )

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Convert a SQLite row to a Message object."""
        metadata = {}
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=_decrypt(row["content"]),  # S125: transparent decryption
            timestamp=row["timestamp"],
            token_estimate=row["token_estimate"],
            model=row["model"],
            metadata=metadata,
        )

    def _sync_snapshot(
        self, conn: sqlite3.Connection, conv_id: str
    ) -> dict[str, Any] | None:
        """Build the full-state sync payload for a conversation (S199, SYN-01).

        The change feed is state-based LWW: ``since`` collapses to the latest
        record per key and CHF-02 compaction will delete superseded rows, so
        every journalled record must be a self-sufficient full state -- an
        event-delta payload would be destroyed by the collapse and a
        late-joining peer would never converge. The reads reuse the caller's
        ALREADY-OPEN connection (no second connection, no network); message
        content is decrypted to plaintext for cross-device portability (the
        S125 field key is per-install) -- at rest the change feed itself is
        SQLCipher via ``safe_connect``, in flight the Veilid route is E2E.
        Local SQLite message ids are device-local identities and are excluded;
        messages are ordered by timestamp then id. Cost: two SELECTs on the
        open connection, O(n) in conversation length -- recorded for the
        shakedown perf watch (MEM-02 class). Returns None when the conversation
        row is gone; the publish hook then skips rather than wiping peers.
        """
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not row:
            return None
        conv = self._row_to_conversation(row)
        msg_rows = conn.execute(
            """SELECT * FROM messages
               WHERE conversation_id = ?
               ORDER BY timestamp ASC, id ASC""",
            (conv_id,),
        ).fetchall()
        messages = []
        for r in msg_rows:
            m = self._row_to_message(r)
            messages.append(
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "token_estimate": m.token_estimate,
                    "model": m.model,
                    "metadata": m.metadata,
                }
            )
        return {
            "user_id": _sync_owner_id(),
            "conversation": {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "model": conv.model,
                "task_type": conv.task_type,
                "preset": conv.preset,
                "metadata": conv.metadata,
                "messages": messages,
            },
        }

    # -----------------------------------------------------------------------
    # CRUD - Conversations
    # -----------------------------------------------------------------------

    def create_conversation(
        self,
        title: str | None = None,
        model: str | None = None,
        task_type: str | None = None,
        preset: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        """Create a new conversation.

        Args:
            title: Conversation title (default: 'New conversation')
            model: Initial model name
            task_type: Initial task type
            preset: Preset used
            metadata: Additional metadata

        Returns:
            The created Conversation object
        """
        conv_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        title = title or "New conversation"
        meta_json = json.dumps(metadata or {})

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """INSERT INTO conversations
                       (id, title, created_at, updated_at, model, task_type, preset, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (conv_id, title, now, now, model, task_type, preset, meta_json),
                )
                conn.commit()
                # S199 SYN-01: domain commit first, then the sync publish
                # (best-effort; a snapshot or journalling failure never breaks
                # the save -- the hook builds the snapshot inside its guard).
                _sync_publish_conversation(
                    conv_id,
                    lambda: self._sync_snapshot(conn, conv_id),
                    updated_at=now,
                )
            except Exception as e:
                logger.error(f"Error creating conversation: {e}")
                raise
            finally:
                conn.close()

        logger.debug(f"Conversation creee: {conv_id} - {title}")

        return Conversation(
            id=conv_id,
            title=title,
            created_at=now,
            updated_at=now,
            model=model,
            task_type=task_type,
            preset=preset,
            metadata=metadata or {},
            messages=[],
        )

    def get_conversation(self, conv_id: str) -> Conversation | None:
        """Get a conversation with all its messages.

        Args:
            conv_id: Conversation UUID

        Returns:
            Conversation with messages, or None if not found
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Retrieve the conversation
                row = conn.execute(
                    "SELECT * FROM conversations WHERE id = ?", (conv_id,)
                ).fetchone()

                if not row:
                    return None

                conv = self._row_to_conversation(row)

                # Load all messages
                msg_rows = conn.execute(
                    """SELECT * FROM messages
                       WHERE conversation_id = ?
                       ORDER BY timestamp ASC, id ASC""",
                    (conv_id,),
                ).fetchall()

                conv.messages = [self._row_to_message(r) for r in msg_rows]
                return conv
            except Exception as e:
                logger.error(f"Error reading conversation {conv_id}: {e}")
                return None
            finally:
                conn.close()

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        """List conversations sorted by last update (newest first).

        Args:
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of Conversation objects (without messages loaded)
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    """SELECT * FROM conversations
                       ORDER BY updated_at DESC
                       LIMIT ? OFFSET ?""",
                    (limit, offset),
                ).fetchall()

                conversations = []
                for row in rows:
                    conv = self._row_to_conversation(row)
                    # Compte les messages sans les charger
                    count_row = conn.execute(
                        "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
                        (conv.id,),
                    ).fetchone()
                    # Stocke le nombre dans metadata pour reference
                    conv.metadata["_message_count"] = count_row["cnt"] if count_row else 0
                    conversations.append(conv)

                return conversations
            except Exception as e:
                logger.error(f"Error listing conversations: {e}")
                return []
            finally:
                conn.close()

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages.

        Args:
            conv_id: Conversation UUID

        Returns:
            True if deletion succeeded
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # ON DELETE CASCADE s'occupe des messages
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = ?", (conv_id,)
                )
                conn.commit()
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.debug(f"Conversation supprimee: {conv_id}")
                    # S199 SYN-01: a domain delete publishes a tombstone so the
                    # deletion converges on peers (empty payload, deleted=True).
                    _sync_publish_conversation(
                        conv_id,
                        deleted=True,
                        updated_at=datetime.now().isoformat(),
                    )
                return deleted
            except Exception as e:
                logger.error(f"Error deleting conversation {conv_id}: {e}")
                return False
            finally:
                conn.close()

    def rename_conversation(self, conv_id: str, new_title: str) -> bool:
        """Rename a conversation.

        Args:
            conv_id: Conversation UUID
            new_title: New title

        Returns:
            True if rename succeeded
        """
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (new_title, now, conv_id),
                )
                conn.commit()
                renamed = cursor.rowcount > 0
                if renamed:
                    logger.debug(f"Conversation renommee: {conv_id} -> {new_title}")
                    # S199 SYN-01: a rename is synced state; publish the new state.
                    _sync_publish_conversation(
                        conv_id,
                        lambda: self._sync_snapshot(conn, conv_id),
                        updated_at=now,
                    )
                return renamed
            except Exception as e:
                logger.error(f"Error renaming conversation {conv_id}: {e}")
                return False
            finally:
                conn.close()

    def update_conversation_metadata(
        self,
        conv_id: str,
        model: str | None = None,
        task_type: str | None = None,
        preset: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update conversation metadata fields.

        Only updates fields that are not None.

        Args:
            conv_id: Conversation UUID
            model: Model name to set
            task_type: Task type to set
            preset: Preset to set
            metadata: Extra metadata to merge

        Returns:
            True if update succeeded
        """
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                # Build query dynamically
                updates = ["updated_at = ?"]
                params: list = [now]

                if model is not None:
                    updates.append("model = ?")
                    params.append(model)
                if task_type is not None:
                    updates.append("task_type = ?")
                    params.append(task_type)
                if preset is not None:
                    updates.append("preset = ?")
                    params.append(preset)
                if metadata is not None:
                    # Fusionne avec les metadata existantes
                    row = conn.execute(
                        "SELECT metadata FROM conversations WHERE id = ?",
                        (conv_id,),
                    ).fetchone()
                    existing = {}
                    if row and row["metadata"]:
                        try:
                            existing = json.loads(row["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    existing.update(metadata)
                    updates.append("metadata = ?")
                    params.append(json.dumps(existing))

                params.append(conv_id)
                # S138: validate column names against allowlist
                for u in updates:
                    col = u.split("=")[0].strip()
                    assert col in _CONV_UPDATE_COLS, f"Invalid column: {col}"
                query = "UPDATE conversations SET {} WHERE id = ?".format(
                    ", ".join(updates)
                )

                cursor = conn.execute(query, params)
                conn.commit()
                updated = cursor.rowcount > 0
                if updated:
                    # S199 SYN-01: metadata is synced state; publish the new state.
                    _sync_publish_conversation(
                        conv_id,
                        lambda: self._sync_snapshot(conn, conv_id),
                        updated_at=now,
                    )
                return updated
            except Exception as e:
                logger.error(f"Error updating conversation {conv_id}: {e}")
                return False
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Messages
    # -----------------------------------------------------------------------

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message | None:
        """Add a message to a conversation.

        Estimates token count and updates conversation's updated_at.

        Args:
            conv_id: Conversation UUID
            role: Message role ('system', 'user', 'assistant')
            content: Message content
            model: Model used (for assistant messages)
            metadata: Additional metadata

        Returns:
            The created Message object, or None on error
        """
        now = datetime.now().isoformat()
        token_estimate = _estimate_tokens(content, model)
        meta_json = json.dumps(metadata or {})

        with self._lock:
            conn = self._get_connection()
            try:
                # Check that the conversation exists
                exists = conn.execute(
                    "SELECT id FROM conversations WHERE id = ?", (conv_id,)
                ).fetchone()
                if not exists:
                    logger.error(f"Conversation introuvable: {conv_id}")
                    return None

                # Insere le message
                # S125: Encrypt content at rest
                stored_content = _encrypt(content)
                cursor = conn.execute(
                    """INSERT INTO messages
                       (conversation_id, role, content, timestamp,
                        token_estimate, model, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (conv_id, role, stored_content, now, token_estimate, model, meta_json),
                )

                # Update updated_at and the conversation model
                update_fields = ["updated_at = ?"]
                update_params: list = [now]
                if model and role == "assistant":
                    update_fields.append("model = ?")
                    update_params.append(model)
                update_params.append(conv_id)

                # S138: validate column names against allowlist
                for u in update_fields:
                    col = u.split("=")[0].strip()
                    assert col in _CONV_UPDATE_COLS, f"Invalid column: {col}"
                _q = "UPDATE conversations SET {} WHERE id = ?".format(
                    ", ".join(update_fields)
                )
                conn.execute(_q, update_params)

                conn.commit()

                # S199 SYN-01: domain commit first, then the sync publish
                # (best-effort; the hook builds the full-state snapshot inside
                # its guard, on this already-open connection, and only when
                # sync is available -- the save never pays otherwise).
                _sync_publish_conversation(
                    conv_id,
                    lambda: self._sync_snapshot(conn, conv_id),
                    updated_at=now,
                )

                msg = Message(
                    id=cursor.lastrowid,
                    conversation_id=conv_id,
                    role=role,
                    content=content,
                    timestamp=now,
                    token_estimate=token_estimate,
                    model=model,
                    metadata=metadata or {},
                )
                logger.debug(
                    f"Message ajoute: conv={conv_id[:8]}... role={role} "
                    f"tokens~{token_estimate}"
                )
                return msg
            except Exception as e:
                logger.error(f"Error adding message: {e}")
                return None
            finally:
                conn.close()

    def get_messages(self, conv_id: str) -> list[Message]:
        """Get all messages for a conversation, ordered chronologically.

        Args:
            conv_id: Conversation UUID

        Returns:
            List of Message objects
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    """SELECT * FROM messages
                       WHERE conversation_id = ?
                       ORDER BY timestamp ASC, id ASC""",
                    (conv_id,),
                ).fetchall()
                return [self._row_to_message(r) for r in rows]
            except Exception as e:
                logger.error(f"Error reading messages {conv_id}: {e}")
                return []
            finally:
                conn.close()

    def get_context_messages(self, conv_id: str) -> list[dict[str, str]]:
        """Get messages in Ollama-ready format [{role, content}, ...].

        Excludes system messages (those are managed by the executor).

        Args:
            conv_id: Conversation UUID

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        messages = self.get_messages(conv_id)
        return [
            m.to_ollama_format()
            for m in messages
            if m.role in ("user", "assistant")
        ]

    def get_token_count(self, conv_id: str) -> int:
        """Get the total estimated token count for a conversation.

        Args:
            conv_id: Conversation UUID

        Returns:
            Sum of token_estimate for all messages
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    """SELECT COALESCE(SUM(token_estimate), 0) as total
                       FROM messages WHERE conversation_id = ?""",
                    (conv_id,),
                ).fetchone()
                return row["total"] if row else 0
            except Exception as e:
                logger.error(f"Error counting tokens {conv_id}: {e}")
                return 0
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def search_conversations(self, query: str, limit: int = 20) -> list[Conversation]:
        """Search conversations by title and message content.

        Args:
            query: Search text
            limit: Maximum results

        Returns:
            List of matching Conversation objects (without messages)
        """
        if not query or not query.strip():
            return []

        # Prepare le terme de search pour LIKE
        search_term = f"%{query.strip()}%"

        with self._lock:
            conn = self._get_connection()
            try:
                # Search dans les titres et le contenu des messages
                rows = conn.execute(
                    """SELECT DISTINCT c.* FROM conversations c
                       LEFT JOIN messages m ON c.id = m.conversation_id
                       WHERE c.title LIKE ? OR m.content LIKE ?
                       ORDER BY c.updated_at DESC
                       LIMIT ?""",
                    (search_term, search_term, limit),
                ).fetchall()
                return [self._row_to_conversation(r) for r in rows]
            except Exception as e:
                logger.error(f"Error searching '{query}': {e}")
                return []
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Statistiques
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored conversations.

        Returns:
            Dictionary with counts, models, task types, etc.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                conv_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM conversations"
                ).fetchone()["cnt"]

                msg_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM messages"
                ).fetchone()["cnt"]

                total_tokens = conn.execute(
                    "SELECT COALESCE(SUM(token_estimate), 0) as total FROM messages"
                ).fetchone()["total"]

                # Repartition par model
                model_rows = conn.execute(
                    """SELECT model, COUNT(*) as cnt FROM conversations
                       WHERE model IS NOT NULL
                       GROUP BY model ORDER BY cnt DESC"""
                ).fetchall()
                by_model = {r["model"]: r["cnt"] for r in model_rows}

                # Repartition par type de tache
                task_rows = conn.execute(
                    """SELECT task_type, COUNT(*) as cnt FROM conversations
                       WHERE task_type IS NOT NULL
                       GROUP BY task_type ORDER BY cnt DESC"""
                ).fetchall()
                by_task = {r["task_type"]: r["cnt"] for r in task_rows}

                return {
                    "total_conversations": conv_count,
                    "total_messages": msg_count,
                    "total_tokens_estimated": total_tokens,
                    "by_model": by_model,
                    "by_task_type": by_task,
                }
            except Exception as e:
                logger.error(f"Error computing statistics: {e}")
                return {}
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Export Markdown
    # -----------------------------------------------------------------------

    def export_conversation_markdown(self, conv_id: str) -> str | None:
        """Export a conversation as a formatted Markdown string.

        Includes conversation metadata (title, model, dates, stats)
        followed by all messages with role headers and timestamps.

        Args:
            conv_id: Conversation UUID

        Returns:
            Markdown string, or None if conversation not found
        """
        conv = self.get_conversation(conv_id)
        if not conv:
            return None

        lines = []

        # --- En-tete ---
        lines.append(f"# {conv.title}")
        lines.append("")

        # Bloc metadata
        lines.append("---")
        lines.append(f"- **Created:** {conv.created_at}")
        lines.append(f"- **Updated:** {conv.updated_at}")
        if conv.model:
            lines.append(f"- **Model:** {conv.model}")
        if conv.task_type:
            lines.append(f"- **Task type:** {conv.task_type}")
        if conv.preset:
            lines.append(f"- **Preset:** {conv.preset}")
        lines.append(f"- **Messages:** {conv.message_count}")
        lines.append(f"- **Tokens (est.):** ~{conv.total_tokens:,}")
        lines.append("---")
        lines.append("")

        # --- Messages ---
        for msg in conv.messages:
            # Role header
            role_label = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System",
            }.get(msg.role, msg.role.capitalize())

            # Timestamp court
            try:
                dt = datetime.fromisoformat(msg.timestamp)
                ts = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                ts = msg.timestamp or ""

            # Ligne d'en-tete du message
            header_parts = [f"### {role_label}"]
            if msg.model and msg.role == "assistant":
                header_parts.append(f"({msg.model})")
            if ts:
                header_parts.append(f"-- {ts}")
            lines.append(" ".join(header_parts))
            lines.append("")

            # Contenu
            lines.append(msg.content)
            lines.append("")

        # --- Pied de page ---
        lines.append("---")
        lines.append(f"*Exported from Opti-Oignon on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Export JSON
    # -----------------------------------------------------------------------

    def export_conversation_json(self, conv_id: str) -> str | None:
        """Export a conversation as a JSON string.

        Includes full conversation metadata and all messages in a
        structured format suitable for re-import or external processing.

        Args:
            conv_id: Conversation UUID

        Returns:
            JSON string (pretty-printed), or None if conversation not found
        """
        conv = self.get_conversation(conv_id)
        if not conv:
            return None

        # Structure d'export complete
        export_data = {
            "opti_oignon_version": _APP_VERSION,
            "export_format": "conversation_v1",
            "exported_at": datetime.now().isoformat(),
            "conversation": {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "model": conv.model,
                "task_type": conv.task_type,
                "preset": conv.preset,
                "metadata": conv.metadata,
                "stats": {
                    "message_count": conv.message_count,
                    "total_tokens": conv.total_tokens,
                },
            },
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "token_estimate": msg.token_estimate,
                    "model": msg.model,
                    "metadata": msg.metadata,
                }
                for msg in conv.messages
            ],
        }

        return json.dumps(export_data, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Export HTML
    # -----------------------------------------------------------------------

    def export_conversation_html(self, conv_id: str) -> str | None:
        """Export a conversation as a self-contained HTML file.

        Generates a standalone HTML document with embedded CSS for
        a clean, readable conversation view. No external dependencies.

        Args:
            conv_id: Conversation UUID

        Returns:
            HTML string, or None if conversation not found
        """
        conv = self.get_conversation(conv_id)
        if not conv:
            return None

        # Echappement HTML basique
        def _esc(text: str) -> str:
            return (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

        # Formater le contenu avec support basique des blocs de code
        def _format_content(content: str) -> str:
            lines = content.split("\n")
            result = []
            in_code = False
            for line in lines:
                if line.startswith("```") and not in_code:
                    lang = _esc(line[3:].strip())
                    result.append(f'<pre class="code-block"><code data-lang="{lang}">')
                    in_code = True
                elif line.startswith("```") and in_code:
                    result.append("</code></pre>")
                    in_code = False
                elif in_code:
                    result.append(_esc(line))
                else:
                    result.append(_esc(line))
            if in_code:
                result.append("</code></pre>")
            # EXP-03 (S194): .message-content is white-space: pre-wrap, so
            # plain newlines already render as line breaks; <br> joins
            # doubled the spacing on messages without code blocks.
            return "\n".join(result)

        # Construction du HTML
        title_esc = _esc(conv.title)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Messages HTML
        messages_html = []
        for msg in conv.messages:
            # EXP-02 (S194): whitelist the class value and escape the
            # timestamp fallback (defense in depth; values are
            # app-controlled today).
            role_class = (
                msg.role
                if msg.role in ("user", "assistant", "system")
                else "other"
            )
            role_label = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
                msg.role, msg.role.capitalize()
            )
            role_label = _esc(role_label)

            # Timestamp
            try:
                dt = datetime.fromisoformat(msg.timestamp)
                ts = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                ts = msg.timestamp or ""
            ts = _esc(ts)

            model_badge = ""
            if msg.model and msg.role == "assistant":
                model_esc = _esc(msg.model)
                model_badge = f' <span class="model-badge">{model_esc}</span>'

            content_html = _format_content(msg.content)

            messages_html.append(
                f'<div class="message {role_class}">'
                f'<div class="message-header">'
                f'<span class="role">{role_label}</span>{model_badge}'
                f'<span class="timestamp">{ts}</span>'
                f'</div>'
                f'<div class="message-content">{content_html}</div>'
                f'</div>'
            )

        messages_block = "\n".join(messages_html)

        # Metadata
        meta_parts = [f"Created: {_esc(conv.created_at)}", f"Updated: {_esc(conv.updated_at)}"]
        if conv.model:
            meta_parts.append(f"Model: {_esc(conv.model)}")
        if conv.task_type:
            meta_parts.append(f"Task: {_esc(conv.task_type)}")
        meta_parts.append(f"Messages: {conv.message_count}")
        meta_parts.append(f"Tokens (est.): ~{conv.total_tokens:,}")
        meta_html = " &middot; ".join(meta_parts)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_esc} — Opti-Oignon Export</title>
<style>
  :root {{
    --bg: #1a1a2e; --surface: #16213e; --border: #0f3460;
    --text: #e0e0e0; --text-muted: #8899aa;
    --user-bg: #1b2838; --assistant-bg: #0f3460; --system-bg: #2a1a3e;
    --accent: #e94560; --code-bg: #0d1117;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.6;
         max-width: 900px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: var(--accent); margin-bottom: 8px; font-size: 1.6em; }}
  .meta {{ color: var(--text-muted); font-size: 0.85em; margin-bottom: 24px;
           padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
  .message {{ margin-bottom: 16px; padding: 14px 18px; border-radius: 10px;
              border: 1px solid var(--border); }}
  .message.user {{ background: var(--user-bg); }}
  .message.assistant {{ background: var(--assistant-bg); }}
  .message.system {{ background: var(--system-bg); }}
  .message-header {{ display: flex; justify-content: space-between;
                     align-items: center; margin-bottom: 8px; }}
  .role {{ font-weight: 700; font-size: 0.9em; text-transform: uppercase;
           letter-spacing: 0.05em; }}
  .model-badge {{ background: var(--accent); color: #fff; font-size: 0.75em;
                  padding: 2px 8px; border-radius: 4px; margin-left: 8px; }}
  .timestamp {{ color: var(--text-muted); font-size: 0.8em; }}
  .message-content {{ white-space: pre-wrap; word-wrap: break-word; }}
  .code-block {{ background: var(--code-bg); padding: 12px; border-radius: 6px;
                 overflow-x: auto; font-family: 'JetBrains Mono', 'Fira Code', monospace;
                 font-size: 0.88em; margin: 8px 0; }}
  .footer {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border);
             color: var(--text-muted); font-size: 0.8em; text-align: center; }}
</style>
</head>
<body>
<h1>{title_esc}</h1>
<div class="meta">{meta_html}</div>
{messages_block}
<div class="footer">Exported from Opti-Oignon on {now_str}</div>
</body>
</html>"""

        return html

    # -----------------------------------------------------------------------
    # Delete the last message (for retry)
    # -----------------------------------------------------------------------

    def delete_last_message(self, conv_id: str, role: str | None = None) -> bool:
        """Delete the last message of a conversation.

        Used for retry: remove the failed assistant message before re-generating.

        Args:
            conv_id: Conversation UUID
            role: If specified, only delete if the last message has this role.
                  Prevents accidental deletion of user messages.

        Returns:
            True if a message was deleted
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Find the last message
                row = conn.execute(
                    """SELECT id, role FROM messages
                       WHERE conversation_id = ?
                       ORDER BY timestamp DESC, id DESC
                       LIMIT 1""",
                    (conv_id,),
                ).fetchone()

                if not row:
                    return False

                # Check le role si specifie
                if role and row["role"] != role:
                    logger.debug(
                        f"delete_last_message: last msg is '{row['role']}', "
                        f"attendu '{role}' -- pas de suppression"
                    )
                    return False

                conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
                conn.commit()
                logger.debug(f"Dernier message supprime: id={row['id']} conv={conv_id[:8]}")
                # S199 SYN-01: a retry-delete is an edit of the conversation,
                # not a deletion of it; publish the reduced full state.
                _sync_publish_conversation(
                    conv_id,
                    lambda: self._sync_snapshot(conn, conv_id),
                    updated_at=datetime.now().isoformat(),
                )
                return True

            except Exception as e:
                logger.error(f"Error deleting last message: {e}")
                return False
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Migration from the legacy JSON history
    # -----------------------------------------------------------------------

    def migrate_json_history(self, history_dir: Path | None = None) -> tuple[int, int]:
        """Import legacy JSON history as single-turn conversations.

        Each HistoryEntry becomes a conversation with one user message
        and one assistant message.

        Args:
            history_dir: Path to legacy history directory
                         (default: DATA_DIR/history)

        Returns:
            Tuple (imported_count, error_count)
        """
        history_dir = history_dir or (DATA_DIR / "history")

        if not history_dir.exists():
            logger.info("No history directory found, nothing to migrate")
            return (0, 0)

        imported = 0
        errors = 0

        # Iterate over all JSON history files
        json_files = sorted(history_dir.glob("history_*.json"))
        if not json_files:
            logger.info("No JSON history file found")
            return (0, 0)

        logger.info(f"Migration: {len(json_files)} fichiers trouves")

        for filepath in json_files:
            try:
                with open(filepath, encoding="utf-8") as f:
                    entries = json.load(f)

                if not isinstance(entries, list):
                    continue

                for entry in entries:
                    try:
                        self._migrate_single_entry(entry)
                        imported += 1
                    except Exception as e:
                        logger.warning(f"Error migrating entry: {e}")
                        errors += 1

            except Exception as e:
                logger.warning(f"Error reading file {filepath.name}: {e}")
                errors += 1

        logger.info(f"Migration terminee: {imported} importes, {errors} erreurs")
        return (imported, errors)

    def _migrate_single_entry(self, entry: dict[str, Any]) -> None:
        """Migrate a single HistoryEntry into a conversation.

        Create a conversation with a title derived from the question,
        then add a user message and an assistant message.
        """
        question = entry.get("question", "")
        response = entry.get("response", "")
        if not question and not response:
            return

        timestamp = entry.get("timestamp", datetime.now().isoformat())
        model = entry.get("model")
        task_type = entry.get("task_type")
        preset = entry.get("preset_used")

        # Titre: premiers 80 caracteres de la question
        title = question[:80].strip()
        if len(question) > 80:
            title += "..."
        if not title:
            title = "Imported entry"

        # Metadata de l'ancien format
        meta = {
            "migrated_from": "json_history",
            "original_id": entry.get("id", ""),
            "temperature": entry.get("temperature"),
            "duration_seconds": entry.get("duration_seconds"),
            "rating": entry.get("rating"),
            "tags": entry.get("tags", []),
        }
        # Clean up None values
        meta = {k: v for k, v in meta.items() if v is not None}

        conv_id = str(uuid.uuid4())
        meta_json = json.dumps(meta)

        with self._lock:
            conn = self._get_connection()
            try:
                # Create the conversation with original timestamps
                conn.execute(
                    """INSERT INTO conversations
                       (id, title, created_at, updated_at, model, task_type, preset, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (conv_id, title, timestamp, timestamp, model, task_type, preset, meta_json),
                )

                # Message user
                user_tokens = _estimate_tokens(question, model)
                conn.execute(
                    """INSERT INTO messages
                       (conversation_id, role, content, timestamp, token_estimate, model, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, '{}')""",
                    (conv_id, "user", question, timestamp, user_tokens, None),
                )

                # Message assistant
                if response:
                    assistant_tokens = _estimate_tokens(response, model)
                    conn.execute(
                        """INSERT INTO messages
                           (conversation_id, role, content, timestamp, token_estimate, model, metadata)
                           VALUES (?, ?, ?, ?, ?, ?, '{}')""",
                        (conv_id, "assistant", response, timestamp, assistant_tokens, model),
                    )

                conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                conn.close()


# ============================================================================
# INSTANCE GLOBALE
# ============================================================================

# Instance partagee, initialisee au premier import
conversation_manager = ConversationManager()


# ============================================================================
# FONCTIONS UTILITAIRES (acces rapide)
# ============================================================================

def create_conversation(**kwargs) -> Conversation:
    """Shortcut to create a conversation."""
    return conversation_manager.create_conversation(**kwargs)


def get_conversation(conv_id: str) -> Conversation | None:
    """Shortcut to retrieve a conversation."""
    return conversation_manager.get_conversation(conv_id)


def list_conversations(limit: int = 50, offset: int = 0) -> list[Conversation]:
    """Shortcut to list conversations."""
    return conversation_manager.list_conversations(limit, offset)


def add_message(conv_id: str, role: str, content: str, **kwargs) -> Message | None:
    """Shortcut to add a message."""
    return conversation_manager.add_message(conv_id, role, content, **kwargs)


def search_conversations(query: str) -> list[Conversation]:
    """Shortcut to search conversations."""
    return conversation_manager.search_conversations(query)


def export_conversation_markdown(conv_id: str) -> str | None:
    """Shortcut to export a conversation as Markdown."""
    return conversation_manager.export_conversation_markdown(conv_id)


def export_conversation_json(conv_id: str) -> str | None:
    """Shortcut to export a conversation as JSON."""
    return conversation_manager.export_conversation_json(conv_id)


def export_conversation_html(conv_id: str) -> str | None:
    """Shortcut to export a conversation as HTML."""
    return conversation_manager.export_conversation_html(conv_id)


def delete_last_message(conv_id: str, role: str | None = None) -> bool:
    """Shortcut to delete the last message."""
    return conversation_manager.delete_last_message(conv_id, role)


# ============================================================================
# CLI - TESTS
# ============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  CONVERSATION MANAGER - Tests CLI")
    print("=" * 60)
    print()

    # Utilise une DB temporaire pour les tests
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_conversations.db"
        manager = ConversationManager(db_path=db_path)

        # --- Test 1: Creation de conversation ---
        print("[TEST 1] Creation de conversation")
        conv = manager.create_conversation(
            title="Test: R debugging",
            model="qwen3-coder:30b",
            task_type="code_r",
        )
        assert conv is not None
        assert conv.title == "Test: R debugging"
        assert conv.id is not None
        print(f"  OK - id={conv.id[:12]}... title='{conv.title}'")

        # --- Test 2: Ajout de messages ---
        print("\n[TEST 2] Ajout de messages")
        msg1 = manager.add_message(conv.id, "user", "Comment calculer une ANOVA en R?")
        assert msg1 is not None
        assert msg1.role == "user"
        assert msg1.token_estimate > 0
        print(f"  OK - user msg id={msg1.id} tokens~{msg1.token_estimate}")

        msg2 = manager.add_message(
            conv.id, "assistant",
            "Pour une ANOVA a un facteur en R:\n\n"
            "```r\nresult <- aov(response ~ factor, data = df)\n"
            "summary(result)\n```\n\n"
            "Utilisez TukeyHSD(result) pour les comparaisons post-hoc.",
            model="qwen3-coder:30b",
        )
        assert msg2 is not None
        print(f"  OK - assistant msg id={msg2.id} tokens~{msg2.token_estimate}")

        msg3 = manager.add_message(conv.id, "user", "Et pour une ANOVA a deux facteurs?")
        msg4 = manager.add_message(
            conv.id, "assistant",
            "```r\nresult <- aov(response ~ factor1 * factor2, data = df)\n```",
            model="qwen3-coder:30b",
        )
        print("  OK - 4 messages au total")

        # --- Test 3: Lecture de conversation ---
        print("\n[TEST 3] Lecture de conversation avec messages")
        loaded = manager.get_conversation(conv.id)
        assert loaded is not None
        assert loaded.message_count == 4
        assert loaded.total_tokens > 0
        print(f"  OK - {loaded.message_count} messages, ~{loaded.total_tokens} tokens")

        # --- Test 4: Format Ollama ---
        print("\n[TEST 4] Format context messages (Ollama)")
        ctx = manager.get_context_messages(conv.id)
        assert len(ctx) == 4  # Pas de system message
        assert all("role" in m and "content" in m for m in ctx)
        print(f"  OK - {len(ctx)} messages prets pour Ollama")
        for m in ctx:
            preview = m["content"][:50].replace("\n", " ")
            print(f"       {m['role']:>9}: {preview}...")

        # --- Test 5: Token count ---
        print("\n[TEST 5] Comptage tokens")
        total = manager.get_token_count(conv.id)
        assert total > 0
        print(f"  OK - Total: ~{total} tokens")

        # --- Test 6: Liste des conversations ---
        print("\n[TEST 6] Listage conversations")
        # Create some additional conversations
        conv2 = manager.create_conversation(title="Python debugging", model="qwen3-coder:30b")
        manager.add_message(conv2.id, "user", "How to use pandas merge?")
        conv3 = manager.create_conversation(title="Scientific writing", model="nemotron-3-nano:30b")
        manager.add_message(conv3.id, "user", "Review my abstract about BCI biodiversity")

        conversations = manager.list_conversations()
        assert len(conversations) == 3
        print(f"  OK - {len(conversations)} conversations")
        for c in conversations:
            print(f"       [{c.updated_at[:10]}] {c.title}")

        # --- Test 7: Renommage ---
        print("\n[TEST 7] Renommage")
        ok = manager.rename_conversation(conv.id, "ANOVA en R - Session complete")
        assert ok
        renamed = manager.get_conversation(conv.id)
        assert renamed.title == "ANOVA en R - Session complete"
        print(f"  OK - Nouveau titre: '{renamed.title}'")

        # --- Test 8: Search ---
        print("\n[TEST 8] Search")
        results = manager.search_conversations("ANOVA")
        assert len(results) >= 1
        print(f"  OK - 'ANOVA' -> {len(results)} resultats")

        results2 = manager.search_conversations("pandas")
        assert len(results2) >= 1
        print(f"  OK - 'pandas' -> {len(results2)} resultats")

        results3 = manager.search_conversations("biodiversity")
        assert len(results3) >= 1
        print(f"  OK - 'biodiversity' -> {len(results3)} resultats")

        results4 = manager.search_conversations("xyznonexistent")
        assert len(results4) == 0
        print(f"  OK - 'xyznonexistent' -> {len(results4)} resultats")

        # --- Test 9: Update metadata ---
        print("\n[TEST 9] Update metadata")
        ok = manager.update_conversation_metadata(
            conv.id, task_type="code_r", metadata={"source": "test"}
        )
        assert ok
        updated = manager.get_conversation(conv.id)
        assert updated.task_type == "code_r"
        assert updated.metadata.get("source") == "test"
        print(f"  OK - task_type={updated.task_type}, meta={updated.metadata}")

        # --- Test 10: Statistiques ---
        print("\n[TEST 10] Statistiques")
        stats = manager.get_stats()
        assert stats["total_conversations"] == 3
        assert stats["total_messages"] >= 6
        print(f"  OK - {stats['total_conversations']} convs, "
              f"{stats['total_messages']} msgs, "
              f"~{stats['total_tokens_estimated']} tokens")
        print(f"       Models: {stats['by_model']}")
        print(f"       Taches: {stats['by_task_type']}")

        # --- Test 11: Suppression ---
        print("\n[TEST 11] Suppression")
        ok = manager.delete_conversation(conv2.id)
        assert ok
        assert manager.get_conversation(conv2.id) is None
        remaining = manager.list_conversations()
        assert len(remaining) == 2
        print(f"  OK - Conversation supprimee, {len(remaining)} restantes")

        # --- Test 12: Migration JSON ---
        print("\n[TEST 12] Migration JSON")
        # Create dummy history JSON files
        fake_history_dir = Path(tmpdir) / "history"
        fake_history_dir.mkdir()

        fake_entries = [
            {
                "id": "abc123",
                "timestamp": "2026-02-20T10:30:00",
                "question": "Comment installer vegan en R?",
                "refined_question": "Comment installer le package vegan en R?",
                "response": "install.packages('vegan')",
                "task_type": "code_r",
                "model": "qwen3-coder:30b",
                "temperature": 0.3,
                "duration_seconds": 1.5,
                "tags": ["r", "ecology"],
            },
            {
                "id": "def456",
                "timestamp": "2026-02-20T14:00:00",
                "question": "Explain Shannon diversity index",
                "refined_question": "Explain Shannon diversity index H'",
                "response": "The Shannon diversity index H' = -sum(pi * ln(pi))...",
                "task_type": "science",
                "model": "nemotron-3-nano:30b",
                "temperature": 0.5,
            },
        ]

        with open(fake_history_dir / "history_2026-02-20.json", "w") as f:
            json.dump(fake_entries, f)

        imported, errs = manager.migrate_json_history(fake_history_dir)
        assert imported == 2
        assert errs == 0
        print(f"  OK - {imported} entrees importees, {errs} erreurs")

        # Check that migrated conversations appear
        all_convs = manager.list_conversations(limit=100)
        migrated = [c for c in all_convs if "migrated_from" in c.metadata]
        print(f"  OK - {len(migrated)} conversations migrees trouvees")

        # --- Test 13: Conversation inexistante ---
        print("\n[TEST 13] Cas limites")
        assert manager.get_conversation("nonexistent-id") is None
        assert manager.delete_conversation("nonexistent-id") is False
        assert manager.rename_conversation("nonexistent-id", "test") is False
        assert manager.add_message("nonexistent-id", "user", "test") is None
        assert manager.get_messages("nonexistent-id") == []
        assert manager.get_token_count("nonexistent-id") == 0
        assert manager.search_conversations("") == []
        print("  OK - Tous les cas limites geres sans erreur")

        # --- Resume ---
        print()
        print("=" * 60)
        print("  TOUS LES TESTS PASSES")
        print("=" * 60)

        final_stats = manager.get_stats()
        print(f"\n  Base finale: {final_stats['total_conversations']} conversations, "
              f"{final_stats['total_messages']} messages")
