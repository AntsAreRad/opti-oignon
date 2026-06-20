#!/usr/bin/env python3
"""
Conversation Branching -- Opti-Oignon S97
==========================================

Fork conversations at any message, explore alternative paths,
compare branches side-by-side, and merge insights back.

Architecture:
    - Separate SQLite database (branches.db) for branch metadata
    - Branch messages are stored in the main conversations.db via
      the existing ConversationManager
    - Each branch references a fork_message_id (the last shared message)
    - Messages after the fork point belong to a specific branch
    - The "main" branch (branch_id=None) is the original conversation

SQLite schema:
    branches:
        branch_id TEXT PK
        conversation_id TEXT NOT NULL
        parent_branch_id TEXT (NULL for branches forked from main)
        fork_message_id INTEGER NOT NULL
        name TEXT
        color TEXT
        created_at TEXT
        updated_at TEXT
        metadata TEXT DEFAULT '{}'

    branch_messages:
        id INTEGER PK AUTOINCREMENT
        branch_id TEXT NOT NULL FK
        conversation_id TEXT NOT NULL
        role TEXT NOT NULL
        content TEXT NOT NULL
        timestamp TEXT NOT NULL
        token_estimate INTEGER DEFAULT 0
        model TEXT
        metadata TEXT DEFAULT '{}'

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
from typing import Any

from opti_oignon.db_utils import safe_connect

logger = logging.getLogger(__name__)

# S138: Allowed column names for dynamic UPDATE queries
_BRANCH_UPDATE_COLS = frozenset({"name", "color", "metadata"})

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_UNSET = object()

_DEFAULT_CONFIG: dict[str, Any] = {
    "branches": {
        "max_per_conversation": 50,
        "default_name_template": "Branch {n}",
        "color_palette": [
            "#B59E7D", "#7A857C", "#8A7E72", "#6B7F8E",
            "#9E8A76", "#7C8B7A", "#8E7D6D", "#6E7B86",
        ],
    },
    "merge": {
        "preserve_timestamps": True,
        "tag_merged_messages": True,
        "max_messages_per_merge": 200,
    },
    "display": {
        "sidebar_mode": True,
        "highlight_divergence": True,
        "max_tree_depth": 0,
    },
}


def _load_config() -> dict[str, Any]:
    """Load branches configuration from YAML with fallback to defaults."""
    try:
        import yaml
        config_path = Path(__file__).parent / "config" / "branches.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            # Merge with defaults
            merged = dict(_DEFAULT_CONFIG)
            for key in merged:
                if key in loaded and isinstance(loaded[key], dict):
                    merged[key] = {**merged[key], **loaded[key]}
            return merged
    except Exception as e:
        logger.warning("Failed to load branches.yaml, using defaults: %s", e)
    return dict(_DEFAULT_CONFIG)


# Module-level config singleton
branches_config = _load_config()

# ---------------------------------------------------------------------------
# Token estimation fallback
# ---------------------------------------------------------------------------

try:
    from .context_manager import estimate_tokens as _cm_estimate_tokens
    _HAS_CONTEXT_MANAGER = True
except ImportError:
    _HAS_CONTEXT_MANAGER = False


def _estimate_tokens(text: str, model: str | None = None) -> int:
    """Estimate token count for a text string."""
    if not text:
        return 0
    if _HAS_CONTEXT_MANAGER:
        try:
            return _cm_estimate_tokens(text, model)
        except Exception:
            pass
    return int(len(text) / 4)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    """A conversation branch (fork point)."""
    branch_id: str
    conversation_id: str
    parent_branch_id: str | None
    fork_message_id: int
    name: str
    color: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BranchMessage:
    """A message belonging to a specific branch."""
    id: int
    branch_id: str
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
        """Return Ollama-compatible format."""
        return {"role": self.role, "content": self.content}


@dataclass
class BranchTreeNode:
    """A node in the branch tree structure."""
    branch_id: str | None  # None = main conversation
    name: str
    color: str
    fork_message_id: int | None
    message_count: int
    last_model: str | None
    last_activity: str | None
    children: list["BranchTreeNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (recursive)."""
        d = {
            "branch_id": self.branch_id,
            "name": self.name,
            "color": self.color,
            "fork_message_id": self.fork_message_id,
            "message_count": self.message_count,
            "last_model": self.last_model,
            "last_activity": self.last_activity,
            "children": [c.to_dict() for c in self.children],
        }
        return d


@dataclass
class BranchComparison:
    """Side-by-side comparison of two branches."""
    branch_a_id: str | None
    branch_b_id: str | None
    branch_a_name: str
    branch_b_name: str
    shared_messages: list[dict[str, Any]]
    branch_a_messages: list[dict[str, Any]]
    branch_b_messages: list[dict[str, Any]]
    fork_message_id: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "branch_a_id": self.branch_a_id,
            "branch_b_id": self.branch_b_id,
            "branch_a_name": self.branch_a_name,
            "branch_b_name": self.branch_b_name,
            "shared_messages": self.shared_messages,
            "branch_a_messages": self.branch_a_messages,
            "branch_b_messages": self.branch_b_messages,
            "fork_message_id": self.fork_message_id,
        }


# ---------------------------------------------------------------------------
# Feature availability flag
# ---------------------------------------------------------------------------

BRANCHES_AVAILABLE = True


# ---------------------------------------------------------------------------
# ConversationBranchManager
# ---------------------------------------------------------------------------

class ConversationBranchManager:
    """
    Manages conversation branches with SQLite storage.

    Provides fork, rename, delete, merge, tree computation, and
    side-by-side comparison of conversation branches.

    Usage:
        manager = ConversationBranchManager()
        branch = manager.fork("conv-123", fork_message_id=5, name="Try GPT")
        messages = manager.get_branch_messages("conv-123", branch.branch_id)
        manager.add_branch_message(branch.branch_id, "conv-123", "user", "Hello")
    """

    def __init__(self, db_path: Path | None = None, config: dict | None = _UNSET):
        """Initialize the branch manager.

        Args:
            db_path: Path to branches SQLite database.
            config: Configuration dict (default: module-level singleton).
        """
        if config is _UNSET:
            config = branches_config

        self._config = config or _DEFAULT_CONFIG

        if db_path is None:
            try:
                from .config import DATA_DIR
                self._db_path = DATA_DIR / "branches.db"
            except ImportError:
                self._db_path = Path("data") / "branches.db"
        else:
            self._db_path = db_path

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()
        logger.info("ConversationBranchManager initialized: %s", self._db_path)

    # -------------------------------------------------------------------
    # Database setup
    # -------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Create a configured SQLite connection.

        S136 audit fix: routes through get_encrypted_connection().
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
                    CREATE TABLE IF NOT EXISTS branches (
                        branch_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        parent_branch_id TEXT,
                        fork_message_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        color TEXT NOT NULL DEFAULT '#B59E7D',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}',
                        FOREIGN KEY (parent_branch_id)
                            REFERENCES branches(branch_id) ON DELETE SET NULL
                    );

                    CREATE TABLE IF NOT EXISTS branch_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        branch_id TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        token_estimate INTEGER DEFAULT 0,
                        model TEXT,
                        metadata TEXT DEFAULT '{}',
                        FOREIGN KEY (branch_id)
                            REFERENCES branches(branch_id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_branches_conv
                        ON branches(conversation_id);
                    CREATE INDEX IF NOT EXISTS idx_branch_messages_branch
                        ON branch_messages(branch_id);
                    CREATE INDEX IF NOT EXISTS idx_branch_messages_conv
                        ON branch_messages(conversation_id);
                """)
                conn.commit()
            except Exception as e:
                logger.error("Failed to initialize branches DB: %s", e)
                raise
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _row_to_branch(self, row: sqlite3.Row) -> Branch:
        """Convert a SQLite row to a Branch object."""
        metadata = {}
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        return Branch(
            branch_id=row["branch_id"],
            conversation_id=row["conversation_id"],
            parent_branch_id=row["parent_branch_id"],
            fork_message_id=row["fork_message_id"],
            name=row["name"],
            color=row["color"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

    def _row_to_message(self, row: sqlite3.Row) -> BranchMessage:
        """Convert a SQLite row to a BranchMessage object."""
        metadata = {}
        try:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        return BranchMessage(
            id=row["id"],
            branch_id=row["branch_id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            timestamp=row["timestamp"],
            token_estimate=row["token_estimate"],
            model=row["model"],
            metadata=metadata,
        )

    def _next_color(self, conversation_id: str) -> str:
        """Get the next color from the palette for a conversation."""
        palette = self._config.get("branches", {}).get("color_palette", ["#B59E7D"])
        count = self._count_branches(conversation_id)
        return palette[count % len(palette)]

    def _next_name(self, conversation_id: str) -> str:
        """Generate the next default branch name."""
        template = self._config.get("branches", {}).get(
            "default_name_template", "Branch {n}"
        )
        count = self._count_branches(conversation_id)
        return template.replace("{n}", str(count + 1))

    def _count_branches(self, conversation_id: str) -> int:
        """Count existing branches for a conversation (no lock)."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM branches WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    # -------------------------------------------------------------------
    # CRUD: Fork
    # -------------------------------------------------------------------

    def fork(
        self,
        conversation_id: str,
        fork_message_id: int,
        name: str | None = None,
        color: str | None = None,
        parent_branch_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Branch | None:
        """Fork a conversation at a specific message.

        Creates a new branch that shares history up to fork_message_id.
        Messages added to this branch will diverge from the original.

        Args:
            conversation_id: The conversation to fork.
            fork_message_id: The message ID at the fork point (last shared).
            name: Branch name (auto-generated if None).
            color: Branch color hex (auto-assigned if None).
            parent_branch_id: Parent branch (None = forked from main).
            metadata: Extra metadata dict.

        Returns:
            The created Branch, or None on error.
        """
        # Check branch limit
        max_branches = self._config.get("branches", {}).get("max_per_conversation", 50)
        if max_branches > 0:
            current = self._count_branches(conversation_id)
            if current >= max_branches:
                logger.warning(
                    "Branch limit reached (%d) for conversation %s",
                    max_branches, conversation_id,
                )
                return None

        branch_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        branch_name = name or self._next_name(conversation_id)
        branch_color = color or self._next_color(conversation_id)
        meta_json = json.dumps(metadata or {})

        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """INSERT INTO branches
                       (branch_id, conversation_id, parent_branch_id,
                        fork_message_id, name, color, created_at, updated_at,
                        metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (branch_id, conversation_id, parent_branch_id,
                     fork_message_id, branch_name, branch_color,
                     now, now, meta_json),
                )
                conn.commit()
                branch = Branch(
                    branch_id=branch_id,
                    conversation_id=conversation_id,
                    parent_branch_id=parent_branch_id,
                    fork_message_id=fork_message_id,
                    name=branch_name,
                    color=branch_color,
                    created_at=now,
                    updated_at=now,
                    metadata=metadata or {},
                )
                logger.info(
                    "Branch created: %s (conv=%s, fork_msg=%d)",
                    branch_id[:8], conversation_id[:8], fork_message_id,
                )
                return branch
            except Exception as e:
                logger.error("Failed to create branch: %s", e)
                return None
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # CRUD: List / Get
    # -------------------------------------------------------------------

    def list_branches(self, conversation_id: str) -> list[Branch]:
        """List all branches for a conversation.

        Args:
            conversation_id: The conversation UUID.

        Returns:
            List of Branch objects ordered by creation time.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    """SELECT * FROM branches
                       WHERE conversation_id = ?
                       ORDER BY created_at ASC""",
                    (conversation_id,),
                ).fetchall()
                return [self._row_to_branch(r) for r in rows]
            except Exception as e:
                logger.error("Failed to list branches: %s", e)
                return []
            finally:
                conn.close()

    def get_branch(self, branch_id: str) -> Branch | None:
        """Get a single branch by ID.

        Args:
            branch_id: The branch UUID.

        Returns:
            Branch object or None if not found.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM branches WHERE branch_id = ?",
                    (branch_id,),
                ).fetchone()
                return self._row_to_branch(row) if row else None
            except Exception as e:
                logger.error("Failed to get branch %s: %s", branch_id, e)
                return None
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # CRUD: Update (rename / recolor)
    # -------------------------------------------------------------------

    def update_branch(
        self,
        branch_id: str,
        name: str | None = None,
        color: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Branch | None:
        """Update a branch's name, color, or metadata.

        Args:
            branch_id: The branch UUID.
            name: New name (unchanged if None).
            color: New color hex (unchanged if None).
            metadata: New metadata (merged if provided).

        Returns:
            Updated Branch object, or None on error.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM branches WHERE branch_id = ?",
                    (branch_id,),
                ).fetchone()
                if not row:
                    logger.warning("Branch not found: %s", branch_id)
                    return None

                now = datetime.now().isoformat()
                updates = ["updated_at = ?"]
                params: list[Any] = [now]

                if name is not None:
                    updates.append("name = ?")
                    params.append(name)
                if color is not None:
                    updates.append("color = ?")
                    params.append(color)
                if metadata is not None:
                    existing_meta = {}
                    try:
                        existing_meta = json.loads(row["metadata"]) if row["metadata"] else {}
                    except (json.JSONDecodeError, TypeError):
                        pass
                    existing_meta.update(metadata)
                    updates.append("metadata = ?")
                    params.append(json.dumps(existing_meta))

                params.append(branch_id)
                # S138: validate column names against allowlist
                for u in updates:
                    col = u.split("=")[0].strip()
                    assert col in _BRANCH_UPDATE_COLS, f"Invalid column: {col}"
                _q = "UPDATE branches SET {} WHERE branch_id = ?".format(
                    ", ".join(updates)
                )
                conn.execute(_q, params)
                conn.commit()
                return self.get_branch(branch_id)
            except Exception as e:
                logger.error("Failed to update branch %s: %s", branch_id, e)
                return None
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # CRUD: Delete
    # -------------------------------------------------------------------

    def delete_branch(self, branch_id: str) -> bool:
        """Delete a branch and all its messages.

        Child branches are re-parented to the deleted branch's parent
        (ON DELETE SET NULL on parent_branch_id FK).

        Args:
            branch_id: The branch UUID.

        Returns:
            True if deleted, False on error or not found.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Check existence
                row = conn.execute(
                    "SELECT branch_id, parent_branch_id FROM branches WHERE branch_id = ?",
                    (branch_id,),
                ).fetchone()
                if not row:
                    return False

                parent_id = row["parent_branch_id"]

                # Re-parent children to this branch's parent
                conn.execute(
                    """UPDATE branches SET parent_branch_id = ?
                       WHERE parent_branch_id = ?""",
                    (parent_id, branch_id),
                )

                # Delete messages (CASCADE should handle this, but be explicit)
                conn.execute(
                    "DELETE FROM branch_messages WHERE branch_id = ?",
                    (branch_id,),
                )

                # Delete branch
                conn.execute(
                    "DELETE FROM branches WHERE branch_id = ?",
                    (branch_id,),
                )
                conn.commit()
                logger.info("Branch deleted: %s", branch_id[:8])
                return True
            except Exception as e:
                logger.error("Failed to delete branch %s: %s", branch_id, e)
                return False
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # Branch messages
    # -------------------------------------------------------------------

    def add_branch_message(
        self,
        branch_id: str,
        conversation_id: str,
        role: str,
        content: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BranchMessage | None:
        """Add a message to a branch.

        Args:
            branch_id: The branch UUID.
            conversation_id: The conversation UUID.
            role: Message role (user, assistant, system).
            content: Message content.
            model: Model used (for assistant messages).
            metadata: Extra metadata.

        Returns:
            The created BranchMessage, or None on error.
        """
        now = datetime.now().isoformat()
        token_estimate = _estimate_tokens(content, model)
        meta_json = json.dumps(metadata or {})

        with self._lock:
            conn = self._get_connection()
            try:
                # Verify branch exists
                exists = conn.execute(
                    "SELECT branch_id FROM branches WHERE branch_id = ?",
                    (branch_id,),
                ).fetchone()
                if not exists:
                    logger.error("Branch not found: %s", branch_id)
                    return None

                cursor = conn.execute(
                    """INSERT INTO branch_messages
                       (branch_id, conversation_id, role, content,
                        timestamp, token_estimate, model, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (branch_id, conversation_id, role, content,
                     now, token_estimate, model, meta_json),
                )

                # Update branch updated_at
                conn.execute(
                    "UPDATE branches SET updated_at = ? WHERE branch_id = ?",
                    (now, branch_id),
                )
                conn.commit()

                msg = BranchMessage(
                    id=cursor.lastrowid,
                    branch_id=branch_id,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    timestamp=now,
                    token_estimate=token_estimate,
                    model=model,
                    metadata=metadata or {},
                )
                logger.debug(
                    "Branch message added: branch=%s role=%s tokens~%d",
                    branch_id[:8], role, token_estimate,
                )
                return msg
            except Exception as e:
                logger.error("Failed to add branch message: %s", e)
                return None
            finally:
                conn.close()

    def get_branch_only_messages(self, branch_id: str) -> list[BranchMessage]:
        """Get only the messages specific to a branch (after fork point).

        Args:
            branch_id: The branch UUID.

        Returns:
            List of BranchMessage objects ordered chronologically.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    """SELECT * FROM branch_messages
                       WHERE branch_id = ?
                       ORDER BY timestamp ASC, id ASC""",
                    (branch_id,),
                ).fetchall()
                return [self._row_to_message(r) for r in rows]
            except Exception as e:
                logger.error("Failed to get branch messages: %s", e)
                return []
            finally:
                conn.close()

    def get_branch_messages_full(
        self,
        conversation_id: str,
        branch_id: str,
        main_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Get full message history for a branch: shared + branch-specific.

        Shared history = main conversation messages up to fork_message_id.
        Branch-specific = messages in branch_messages table.

        Args:
            conversation_id: The conversation UUID.
            branch_id: The branch UUID.
            main_messages: Pre-fetched main conversation messages (optional).
                           Each dict must have at minimum 'id' key.

        Returns:
            Combined list of message dicts in chronological order.
        """
        branch = self.get_branch(branch_id)
        if not branch:
            return []

        fork_id = branch.fork_message_id

        # Get shared history (main messages up to and including fork point)
        if main_messages is not None:
            shared = [
                m for m in main_messages
                if m.get("id") is not None and m["id"] <= fork_id
            ]
        else:
            shared = self._get_main_messages_up_to(conversation_id, fork_id)

        # Get branch-specific messages
        branch_msgs = self.get_branch_only_messages(branch_id)
        branch_dicts = [m.to_dict() for m in branch_msgs]

        return shared + branch_dicts

    def _get_main_messages_up_to(
        self, conversation_id: str, max_message_id: int
    ) -> list[dict[str, Any]]:
        """Fetch main conversation messages up to a given message ID.

        Uses the conversations.db directly for read-only access.

        Args:
            conversation_id: The conversation UUID.
            max_message_id: Maximum message ID (inclusive).

        Returns:
            List of message dicts.
        """
        try:
            from .config import DATA_DIR
            conv_db = DATA_DIR / "conversations.db"
        except ImportError:
            conv_db = Path("data") / "conversations.db"

        if not conv_db.exists():
            return []

        try:
            conn = safe_connect(str(conv_db), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, conversation_id, role, content, timestamp,
                          token_estimate, model, metadata
                   FROM messages
                   WHERE conversation_id = ? AND id <= ?
                   ORDER BY timestamp ASC, id ASC""",
                (conversation_id, max_message_id),
            ).fetchall()
            result = []
            for r in rows:
                meta = {}
                try:
                    meta = json.loads(r["metadata"]) if r["metadata"] else {}
                except (json.JSONDecodeError, TypeError):
                    pass
                result.append({
                    "id": r["id"],
                    "conversation_id": r["conversation_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "token_estimate": r["token_estimate"],
                    "model": r["model"],
                    "metadata": meta,
                })
            return result
        except Exception as e:
            logger.error("Failed to read main messages: %s", e)
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # -------------------------------------------------------------------
    # Branch metadata / stats
    # -------------------------------------------------------------------

    def get_branch_stats(self, branch_id: str) -> dict[str, Any]:
        """Get statistics for a branch.

        Returns:
            Dict with message_count, last_model, last_activity, total_tokens.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    """SELECT
                        COUNT(*) as message_count,
                        MAX(timestamp) as last_activity,
                        COALESCE(SUM(token_estimate), 0) as total_tokens
                       FROM branch_messages
                       WHERE branch_id = ?""",
                    (branch_id,),
                ).fetchone()

                last_model_row = conn.execute(
                    """SELECT model FROM branch_messages
                       WHERE branch_id = ? AND model IS NOT NULL
                       ORDER BY timestamp DESC, id DESC LIMIT 1""",
                    (branch_id,),
                ).fetchone()

                return {
                    "message_count": row["message_count"] if row else 0,
                    "last_activity": row["last_activity"] if row else None,
                    "total_tokens": row["total_tokens"] if row else 0,
                    "last_model": last_model_row["model"] if last_model_row else None,
                }
            except Exception as e:
                logger.error("Failed to get branch stats: %s", e)
                return {
                    "message_count": 0,
                    "last_activity": None,
                    "total_tokens": 0,
                    "last_model": None,
                }
            finally:
                conn.close()

    # -------------------------------------------------------------------
    # Branch tree
    # -------------------------------------------------------------------

    def get_branch_tree(self, conversation_id: str) -> BranchTreeNode:
        """Build a tree structure of all branches for a conversation.

        The root node represents the main conversation (branch_id=None).
        Child nodes represent branches forked from the main or from
        other branches.

        Args:
            conversation_id: The conversation UUID.

        Returns:
            Root BranchTreeNode.
        """
        branches = self.list_branches(conversation_id)

        # Build stats map
        stats_map: dict[str, dict] = {}
        for b in branches:
            stats_map[b.branch_id] = self.get_branch_stats(b.branch_id)

        # Root node (main conversation)
        root = BranchTreeNode(
            branch_id=None,
            name="Main",
            color="#B59E7D",
            fork_message_id=None,
            message_count=0,  # Caller can fill this from conversation manager
            last_model=None,
            last_activity=None,
        )

        # Build nodes map
        nodes: dict[str | None, BranchTreeNode] = {None: root}
        for b in branches:
            stats = stats_map.get(b.branch_id, {})
            node = BranchTreeNode(
                branch_id=b.branch_id,
                name=b.name,
                color=b.color,
                fork_message_id=b.fork_message_id,
                message_count=stats.get("message_count", 0),
                last_model=stats.get("last_model"),
                last_activity=stats.get("last_activity"),
            )
            nodes[b.branch_id] = node

        # Link parent-child
        for b in branches:
            parent_key = b.parent_branch_id  # None means forked from main
            parent_node = nodes.get(parent_key, root)
            parent_node.children.append(nodes[b.branch_id])

        return root

    # -------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------

    def compare_branches(
        self,
        conversation_id: str,
        branch_a_id: str | None,
        branch_b_id: str | None,
        main_messages: list[dict[str, Any]] | None = None,
    ) -> BranchComparison | None:
        """Compare two branches side-by-side.

        Either branch can be None to represent the main conversation.
        Shows shared messages and divergent messages for each branch.

        Args:
            conversation_id: The conversation UUID.
            branch_a_id: First branch ID (None = main).
            branch_b_id: Second branch ID (None = main).
            main_messages: Pre-fetched main conversation messages.

        Returns:
            BranchComparison object, or None on error.
        """
        # Determine fork points
        fork_a = None
        fork_b = None
        name_a = "Main"
        name_b = "Main"

        if branch_a_id:
            ba = self.get_branch(branch_a_id)
            if not ba:
                return None
            fork_a = ba.fork_message_id
            name_a = ba.name

        if branch_b_id:
            bb = self.get_branch(branch_b_id)
            if not bb:
                return None
            fork_b = bb.fork_message_id
            name_b = bb.name

        # Find the common fork point (minimum of the two, or available one)
        if fork_a is not None and fork_b is not None:
            common_fork = min(fork_a, fork_b)
        elif fork_a is not None:
            common_fork = fork_a
        elif fork_b is not None:
            common_fork = fork_b
        else:
            # Both are main -- nothing to compare
            return None

        # Shared messages up to common fork
        shared = self._get_main_messages_up_to(conversation_id, common_fork)
        if main_messages is not None:
            shared = [
                m for m in main_messages
                if m.get("id") is not None and m["id"] <= common_fork
            ]

        # Branch A messages after fork
        if branch_a_id:
            a_msgs = [m.to_dict() for m in self.get_branch_only_messages(branch_a_id)]
        else:
            # Main conversation messages after fork
            a_msgs = self._get_main_messages_after(conversation_id, common_fork)

        # Branch B messages after fork
        if branch_b_id:
            b_msgs = [m.to_dict() for m in self.get_branch_only_messages(branch_b_id)]
        else:
            b_msgs = self._get_main_messages_after(conversation_id, common_fork)

        return BranchComparison(
            branch_a_id=branch_a_id,
            branch_b_id=branch_b_id,
            branch_a_name=name_a,
            branch_b_name=name_b,
            shared_messages=shared,
            branch_a_messages=a_msgs,
            branch_b_messages=b_msgs,
            fork_message_id=common_fork,
        )

    def _get_main_messages_after(
        self, conversation_id: str, after_message_id: int
    ) -> list[dict[str, Any]]:
        """Fetch main conversation messages after a given message ID."""
        try:
            from .config import DATA_DIR
            conv_db = DATA_DIR / "conversations.db"
        except ImportError:
            conv_db = Path("data") / "conversations.db"

        if not conv_db.exists():
            return []

        try:
            conn = safe_connect(str(conv_db), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, conversation_id, role, content, timestamp,
                          token_estimate, model, metadata
                   FROM messages
                   WHERE conversation_id = ? AND id > ?
                   ORDER BY timestamp ASC, id ASC""",
                (conversation_id, after_message_id),
            ).fetchall()
            result = []
            for r in rows:
                meta = {}
                try:
                    meta = json.loads(r["metadata"]) if r["metadata"] else {}
                except (json.JSONDecodeError, TypeError):
                    pass
                result.append({
                    "id": r["id"],
                    "conversation_id": r["conversation_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "token_estimate": r["token_estimate"],
                    "model": r["model"],
                    "metadata": meta,
                })
            return result
        except Exception as e:
            logger.error("Failed to read main messages after %d: %s", after_message_id, e)
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # -------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------

    def merge_messages(
        self,
        source_branch_id: str,
        target_branch_id: str,
        message_ids: list[int] | None = None,
    ) -> list[BranchMessage]:
        """Merge messages from one branch into another.

        Copies selected (or all) messages from the source branch
        to the target branch. Optionally tags merged messages.

        Args:
            source_branch_id: Source branch UUID.
            target_branch_id: Target branch UUID.
            message_ids: Specific message IDs to merge (None = all).

        Returns:
            List of newly created BranchMessage objects.
        """
        merge_cfg = self._config.get("merge", {})
        max_merge = merge_cfg.get("max_messages_per_merge", 200)
        tag_merged = merge_cfg.get("tag_merged_messages", True)
        preserve_ts = merge_cfg.get("preserve_timestamps", True)

        # Get source messages
        source_msgs = self.get_branch_only_messages(source_branch_id)
        if message_ids is not None:
            id_set = set(message_ids)
            source_msgs = [m for m in source_msgs if m.id in id_set]

        if len(source_msgs) > max_merge:
            logger.warning(
                "Merge capped at %d messages (requested %d)",
                max_merge, len(source_msgs),
            )
            source_msgs = source_msgs[:max_merge]

        # Get target branch info for conversation_id
        target = self.get_branch(target_branch_id)
        if not target:
            logger.error("Target branch not found: %s", target_branch_id)
            return []

        merged: list[BranchMessage] = []
        for msg in source_msgs:
            meta = dict(msg.metadata)
            if tag_merged:
                meta["merged_from"] = source_branch_id
                meta["original_message_id"] = msg.id

            new_msg = self.add_branch_message(
                branch_id=target_branch_id,
                conversation_id=target.conversation_id,
                role=msg.role,
                content=msg.content,
                model=msg.model,
                metadata=meta,
            )
            if new_msg:
                merged.append(new_msg)

        logger.info(
            "Merged %d messages from %s to %s",
            len(merged), source_branch_id[:8], target_branch_id[:8],
        )
        return merged

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    def delete_all_branches(self, conversation_id: str) -> int:
        """Delete all branches for a conversation.

        Args:
            conversation_id: The conversation UUID.

        Returns:
            Number of branches deleted.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                # Get branch IDs first
                rows = conn.execute(
                    "SELECT branch_id FROM branches WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchall()
                branch_ids = [r["branch_id"] for r in rows]

                if not branch_ids:
                    return 0

                # Delete messages for all branches
                placeholders = ",".join("?" * len(branch_ids))
                conn.execute(
                    "DELETE FROM branch_messages WHERE branch_id IN ({})".format(
                        placeholders
                    ),
                    branch_ids,
                )

                # Delete branches
                conn.execute(
                    "DELETE FROM branches WHERE conversation_id = ?",
                    (conversation_id,),
                )
                conn.commit()
                logger.info(
                    "Deleted %d branches for conversation %s",
                    len(branch_ids), conversation_id[:8],
                )
                return len(branch_ids)
            except Exception as e:
                logger.error("Failed to delete branches: %s", e)
                return 0
            finally:
                conn.close()

    def get_config(self) -> dict[str, Any]:
        """Return the current configuration."""
        return dict(self._config)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

try:
    branch_manager = ConversationBranchManager()
except Exception as _init_err:
    logger.error("Failed to initialize ConversationBranchManager: %s", _init_err)
    branch_manager = None
    BRANCHES_AVAILABLE = False
