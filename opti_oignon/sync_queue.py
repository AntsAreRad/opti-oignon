#!/usr/bin/env python3
"""
SYNC QUEUE -- SQLite-Backed Request Queue for Offline Mode (S71)
=================================================================

Stores pending LLM requests when Ollama is offline. Requests are
queued with priority and replayed (FIFO within priority) when
connectivity returns.

The queue persists across restarts via SQLite. Max queue size is
configurable to prevent unbounded growth.

Author: Leon
"""

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)



# =============================================================================
# CONSTANTS
# =============================================================================

_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "network.yaml"
_DEFAULT_DB_DIR = Path(__file__).parent / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "sync_queue.db"

DEFAULT_MAX_QUEUE_SIZE = 100
DEFAULT_PRIORITY = 5  # 1=highest, 10=lowest


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QueueEntry:
    """A pending request in the sync queue.

    Attributes:
        id: Unique entry identifier.
        query: The user query text.
        task_type: Task type string (e.g., 'code_python', 'general').
        priority: Priority level (1=highest, 10=lowest).
        created_at: Epoch timestamp when enqueued.
        status: One of 'pending', 'processing', 'completed', 'failed'.
        error: Error message if failed, else empty string.
        model: Optional preferred model name.
    """
    id: str = ""
    query: str = ""
    task_type: str = "general"
    priority: int = DEFAULT_PRIORITY
    created_at: float = 0.0
    status: str = "pending"
    error: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "task_type": self.task_type,
            "priority": self.priority,
            "created_at": self.created_at,
            "status": self.status,
            "error": self.error,
            "model": self.model,
        }


# =============================================================================
# SYNC QUEUE
# =============================================================================

class SyncQueue:
    """SQLite-backed request queue for offline mode.

    Stores pending LLM requests and replays them when Ollama comes
    back online. Entries are dequeued in priority-then-FIFO order.

    Args:
        db_path: Path to SQLite database file. None uses default.
        config_path: Path to config YAML. None uses default.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        config_path: Path | str | None = None,
    ):
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._config: dict[str, Any] = {}
        self._load_config()
        self._init_db()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _load_config(self) -> None:
        """Load queue-related settings from network.yaml."""
        defaults = {
            "max_queue_size": DEFAULT_MAX_QUEUE_SIZE,
        }
        try:
            if self._config_path.exists():
                with open(self._config_path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                # Queue settings might be nested under 'queue' key or at top level
                if "queue" in loaded and isinstance(loaded["queue"], dict):
                    defaults.update(loaded["queue"])
                elif "max_queue_size" in loaded:
                    defaults["max_queue_size"] = loaded["max_queue_size"]
        except Exception as e:
            logger.warning("Failed to load sync queue config: %s", e)
        self._config = defaults

    def get_config(self) -> dict:
        """Return a copy of queue configuration."""
        return dict(self._config)

    # -----------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the queue table if it does not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = _safe_connect(self._db_path, check_same_thread=False)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_queue (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    task_type TEXT NOT NULL DEFAULT 'general',
                    priority INTEGER NOT NULL DEFAULT 5,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_status_priority
                ON sync_queue(status, priority, created_at)
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new SQLite connection."""
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # -----------------------------------------------------------------
    # Core operations
    # -----------------------------------------------------------------

    def enqueue(
        self,
        query: str,
        task_type: str = "general",
        priority: int = DEFAULT_PRIORITY,
        model: str = "",
    ) -> QueueEntry | None:
        """Add a request to the queue.

        If the queue is at max capacity, the request is rejected.

        Args:
            query: User query text.
            task_type: Task type identifier.
            priority: Priority level (1=highest, 10=lowest).
            model: Optional preferred model name.

        Returns:
            QueueEntry if enqueued, None if queue is full.
        """
        max_size = self._config.get("max_queue_size", DEFAULT_MAX_QUEUE_SIZE)
        if self.size() >= max_size:
            logger.warning("Sync queue full (%d/%d), rejecting request", self.size(), max_size)
            return None

        entry = QueueEntry(
            id=str(uuid.uuid4()),
            query=query,
            task_type=task_type,
            priority=max(1, min(10, priority)),
            created_at=time.time(),
            status="pending",
            model=model,
        )

        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO sync_queue (id, query, task_type, priority, created_at, status, error, model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry.id, entry.query, entry.task_type, entry.priority,
                 entry.created_at, entry.status, entry.error, entry.model),
            )
            conn.commit()
        finally:
            conn.close()

        logger.debug("Enqueued request %s (type=%s, priority=%d)", entry.id, task_type, priority)
        return entry

    def dequeue(self) -> QueueEntry | None:
        """Get the next pending request from the queue.

        Returns the highest-priority (lowest number), then oldest entry.
        Marks it as 'processing'.

        Returns:
            QueueEntry if available, None if queue is empty.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                """SELECT * FROM sync_queue
                   WHERE status = 'pending'
                   ORDER BY priority ASC, created_at ASC
                   LIMIT 1""",
            ).fetchone()

            if row is None:
                return None

            entry = self._row_to_entry(row)
            conn.execute(
                "UPDATE sync_queue SET status = 'processing' WHERE id = ?",
                (entry.id,),
            )
            conn.commit()
            entry.status = "processing"
            return entry
        finally:
            conn.close()

    def mark_completed(self, entry_id: str) -> None:
        """Mark a queue entry as completed."""
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE sync_queue SET status = 'completed' WHERE id = ?",
                (entry_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_failed(self, entry_id: str, error: str = "") -> None:
        """Mark a queue entry as failed with an optional error message."""
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE sync_queue SET status = 'failed', error = ? WHERE id = ?",
                (error, entry_id),
            )
            conn.commit()
        finally:
            conn.close()

    def requeue_failed(self) -> int:
        """Reset all failed entries back to pending for retry.

        Returns:
            Number of entries re-queued.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "UPDATE sync_queue SET status = 'pending', error = '' WHERE status = 'failed'",
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def size(self, status: str | None = None) -> int:
        """Count entries in the queue, optionally filtered by status.

        Args:
            status: If provided, count only entries with this status.

        Returns:
            Number of matching entries.
        """
        conn = self._get_connection()
        try:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) FROM sync_queue WHERE status = ?", (status,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM sync_queue").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def list_entries(self, status: str | None = None, limit: int = 50) -> list[QueueEntry]:
        """List queue entries, optionally filtered by status.

        Args:
            status: Filter by status. None returns all.
            limit: Max entries to return.

        Returns:
            List of QueueEntry objects.
        """
        conn = self._get_connection()
        try:
            if status:
                rows = conn.execute(
                    """SELECT * FROM sync_queue WHERE status = ?
                       ORDER BY priority ASC, created_at ASC LIMIT ?""",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM sync_queue
                       ORDER BY priority ASC, created_at ASC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [self._row_to_entry(r) for r in rows]
        finally:
            conn.close()

    def clear(self, status: str | None = None) -> int:
        """Remove entries from the queue.

        Args:
            status: If provided, only clear entries with this status.
                    If None, clear all entries.

        Returns:
            Number of entries removed.
        """
        conn = self._get_connection()
        try:
            if status:
                cursor = conn.execute(
                    "DELETE FROM sync_queue WHERE status = ?", (status,)
                )
            else:
                cursor = conn.execute("DELETE FROM sync_queue")
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def process_queue(
        self,
        executor_fn: Callable[[QueueEntry], str] | None = None,
    ) -> list[dict]:
        """Drain pending entries and execute them.

        Dequeues entries one by one and calls executor_fn for each.
        If executor_fn is None, entries are simply marked completed
        (useful for testing).

        Args:
            executor_fn: Function that takes a QueueEntry and returns
                a response string. Should raise on failure.

        Returns:
            List of result dicts with 'id', 'status', 'response'/'error'.
        """
        results = []
        while True:
            entry = self.dequeue()
            if entry is None:
                break

            result = {"id": entry.id, "query": entry.query}
            try:
                if executor_fn is not None:
                    response = executor_fn(entry)
                    result["response"] = response
                result["status"] = "completed"
                self.mark_completed(entry.id)
            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)
                self.mark_failed(entry.id, str(e))

            results.append(result)

        return results

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> QueueEntry:
        """Convert a SQLite row to a QueueEntry."""
        return QueueEntry(
            id=row["id"],
            query=row["query"],
            task_type=row["task_type"],
            priority=row["priority"],
            created_at=row["created_at"],
            status=row["status"],
            error=row["error"],
            model=row["model"],
        )


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

try:
    sync_queue = SyncQueue()
except Exception as e:
    logger.warning("Failed to create SyncQueue singleton: %s", e)
    sync_queue = None  # type: ignore[assignment]
