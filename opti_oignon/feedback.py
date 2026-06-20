#!/usr/bin/env python3
"""
FEEDBACK STORE -- User Feedback Collection & Aggregation (S55)
================================================================

SQLite-backed storage for user feedback on model/pipeline responses.
Supports thumbs up/down and 1-5 star ratings, with aggregation
methods for analytics integration.

Author: Leon
"""

import csv
import io
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

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

_DATA_DIR = Path(__file__).parent / "data"
_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_DB_PATH = _DATA_DIR / "feedback.db"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "feedback.yaml"

# Valid rating types
RATING_TYPE_THUMBS = "thumbs"
RATING_TYPE_STARS = "stars"
VALID_RATING_TYPES = {RATING_TYPE_THUMBS, RATING_TYPE_STARS}

# Thumbs values: 1 = up, 0 = down
THUMBS_UP = 1
THUMBS_DOWN = 0

# Star rating range
MIN_STARS = 1
MAX_STARS = 5


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeedbackEntry:
    """Single feedback entry from a user."""

    # Identifiers
    feedback_id: str = ""
    conversation_id: str = ""
    message_id: str = ""

    # Rating data
    rating_type: str = RATING_TYPE_THUMBS  # "thumbs" or "stars"
    rating_value: int = 1  # 0/1 for thumbs, 1-5 for stars
    feedback_text: str = ""

    # Context metadata
    model_used: str = ""
    pipeline_used: str = ""
    task_type: str = ""

    # Timing
    timestamp: float = 0.0

    def __post_init__(self):
        """Generate feedback_id and timestamp if not provided."""
        if not self.feedback_id:
            self.feedback_id = str(uuid.uuid4())[:12]
        if self.timestamp <= 0:
            self.timestamp = time.time()

    def validate(self) -> tuple[bool, str]:
        """Validate feedback entry fields.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if self.rating_type not in VALID_RATING_TYPES:
            return False, f"Invalid rating_type: {self.rating_type}"

        if self.rating_type == RATING_TYPE_THUMBS:
            if self.rating_value not in (THUMBS_UP, THUMBS_DOWN):
                return False, f"Thumbs rating must be 0 or 1, got {self.rating_value}"
        elif self.rating_type == RATING_TYPE_STARS:
            if not (MIN_STARS <= self.rating_value <= MAX_STARS):
                return False, f"Star rating must be {MIN_STARS}-{MAX_STARS}, got {self.rating_value}"

        return True, ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackEntry":
        """Create from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @property
    def is_positive(self) -> bool:
        """Whether the feedback is positive."""
        if self.rating_type == RATING_TYPE_THUMBS:
            return self.rating_value == THUMBS_UP
        return self.rating_value >= 4

    @property
    def normalized_score(self) -> float:
        """Normalized score in 0.0-1.0 range."""
        if self.rating_type == RATING_TYPE_THUMBS:
            return float(self.rating_value)
        # Stars: map 1-5 to 0.0-1.0
        return (self.rating_value - MIN_STARS) / (MAX_STARS - MIN_STARS)


# =============================================================================
# AGGREGATION RESULT
# =============================================================================

@dataclass
class FeedbackStats:
    """Aggregated feedback statistics."""
    total_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    average_score: float = 0.0
    thumbs_up: int = 0
    thumbs_down: int = 0
    star_distribution: dict[int, int] = field(default_factory=dict)
    # Per-key breakdown
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_pipeline: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_task_type: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# FEEDBACK STORE
# =============================================================================

class FeedbackStore:
    """SQLite-backed feedback storage with CRUD and aggregation.

    Stores user feedback entries and provides aggregation methods
    for analytics dashboards and routing optimization.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        config_path: Path | None = None,
    ):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._config: dict[str, Any] = {}

        # Load config
        self._load_config()

        # Initialize database
        self._init_db()

        logger.info("FeedbackStore initialized (db=%s)", self._db_path)

    def _load_config(self) -> None:
        """Load feedback configuration from YAML."""
        try:
            if self._config_path.exists():
                with open(self._config_path) as f:
                    raw = yaml.safe_load(f) or {}
                self._config = raw.get("feedback", {})
            else:
                self._config = {}
        except Exception as e:
            logger.warning("Failed to load feedback config: %s", e)
            self._config = {}

    @property
    def enabled(self) -> bool:
        """Whether feedback collection is enabled."""
        return self._config.get("enabled", True)

    @property
    def max_text_length(self) -> int:
        """Maximum feedback text length."""
        return self._config.get("max_text_length", 2000)

    @property
    def auto_adjust_routing(self) -> bool:
        """Whether to auto-adjust routing scores based on feedback."""
        return self._config.get("auto_adjust_routing", False)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a SQLite connection."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create feedback table if it does not exist."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    conversation_id TEXT DEFAULT '',
                    message_id TEXT DEFAULT '',
                    rating_type TEXT NOT NULL DEFAULT 'thumbs',
                    rating_value INTEGER NOT NULL DEFAULT 1,
                    feedback_text TEXT DEFAULT '',
                    model_used TEXT DEFAULT '',
                    pipeline_used TEXT DEFAULT '',
                    task_type TEXT DEFAULT '',
                    timestamp REAL NOT NULL
                )
            """)
            # Indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_model
                ON feedback(model_used)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_pipeline
                ON feedback(pipeline_used)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_task
                ON feedback(task_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
                ON feedback(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_conversation
                ON feedback(conversation_id)
            """)
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================

    def add_feedback(self, entry: FeedbackEntry) -> FeedbackEntry:
        """Add a feedback entry to the store.

        Args:
            entry: FeedbackEntry to store.

        Returns:
            The stored entry (with generated id/timestamp if needed).

        Raises:
            ValueError: If the entry fails validation.
        """
        # Validate
        valid, error = entry.validate()
        if not valid:
            raise ValueError(f"Invalid feedback: {error}")

        # Truncate text if needed
        if len(entry.feedback_text) > self.max_text_length:
            entry.feedback_text = entry.feedback_text[:self.max_text_length]

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO feedback
                (feedback_id, conversation_id, message_id, rating_type,
                 rating_value, feedback_text, model_used, pipeline_used,
                 task_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.feedback_id,
                    entry.conversation_id,
                    entry.message_id,
                    entry.rating_type,
                    entry.rating_value,
                    entry.feedback_text,
                    entry.model_used,
                    entry.pipeline_used,
                    entry.task_type,
                    entry.timestamp,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.debug("Feedback added: %s (model=%s, pipeline=%s)",
                      entry.feedback_id, entry.model_used, entry.pipeline_used)
        return entry

    def get_feedback(self, feedback_id: str) -> FeedbackEntry | None:
        """Retrieve a single feedback entry by ID.

        Args:
            feedback_id: The feedback entry ID.

        Returns:
            FeedbackEntry or None if not found.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()
            if row:
                return FeedbackEntry(**dict(row))
            return None
        finally:
            conn.close()

    def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a feedback entry.

        Args:
            feedback_id: The feedback entry ID.

        Returns:
            True if deleted, False if not found.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM feedback WHERE feedback_id = ?",
                (feedback_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_feedback(
        self,
        limit: int = 100,
        offset: int = 0,
        since: float | None = None,
        until: float | None = None,
    ) -> list[FeedbackEntry]:
        """List feedback entries with optional time filtering.

        Args:
            limit: Maximum entries to return.
            offset: Pagination offset.
            since: Only entries after this timestamp.
            until: Only entries before this timestamp.

        Returns:
            List of FeedbackEntry objects.
        """
        conditions = []
        params: list = []

        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(until)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM feedback {where}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        conn = self._get_conn()
        try:
            rows = conn.execute(query, params).fetchall()
            return [FeedbackEntry(**dict(r)) for r in rows]
        finally:
            conn.close()

    def list_by_model(self, model: str, limit: int = 100) -> list[FeedbackEntry]:
        """List feedback entries for a specific model.

        Args:
            model: Model name to filter by.
            limit: Maximum entries to return.

        Returns:
            List of FeedbackEntry objects.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE model_used = ? ORDER BY timestamp DESC LIMIT ?",
                (model, limit),
            ).fetchall()
            return [FeedbackEntry(**dict(r)) for r in rows]
        finally:
            conn.close()

    def list_by_pipeline(self, pipeline: str, limit: int = 100) -> list[FeedbackEntry]:
        """List feedback entries for a specific pipeline.

        Args:
            pipeline: Pipeline name to filter by.
            limit: Maximum entries to return.

        Returns:
            List of FeedbackEntry objects.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE pipeline_used = ? ORDER BY timestamp DESC LIMIT ?",
                (pipeline, limit),
            ).fetchall()
            return [FeedbackEntry(**dict(r)) for r in rows]
        finally:
            conn.close()

    def list_by_conversation(self, conversation_id: str) -> list[FeedbackEntry]:
        """List all feedback for a conversation.

        Args:
            conversation_id: Conversation ID to filter by.

        Returns:
            List of FeedbackEntry objects.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE conversation_id = ? ORDER BY timestamp ASC",
                (conversation_id,),
            ).fetchall()
            return [FeedbackEntry(**dict(r)) for r in rows]
        finally:
            conn.close()

    def count(self) -> int:
        """Total number of feedback entries."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM feedback").fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def clear(self) -> int:
        """Delete all feedback entries.

        Returns:
            Number of entries deleted.
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute("DELETE FROM feedback")
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    def _aggregate_group(
        self,
        group_column: str,
        since: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Aggregate feedback stats grouped by a column.

        Args:
            group_column: Column name to group by.
            since: Only entries after this timestamp.

        Returns:
            Dict mapping group values to stat dicts.
        """
        conditions = [f"{group_column} != ''"]
        params: list = []

        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT
                {group_column} as group_key,
                COUNT(*) as total,
                AVG(CASE
                    WHEN rating_type = 'thumbs' THEN CAST(rating_value AS REAL)
                    WHEN rating_type = 'stars' THEN (CAST(rating_value AS REAL) - 1.0) / 4.0
                    ELSE 0.0
                END) as avg_score,
                SUM(CASE WHEN rating_type = 'thumbs' AND rating_value = 1 THEN 1 ELSE 0 END) as thumbs_up,
                SUM(CASE WHEN rating_type = 'thumbs' AND rating_value = 0 THEN 1 ELSE 0 END) as thumbs_down,
                SUM(CASE WHEN rating_type = 'stars' THEN 1 ELSE 0 END) as star_count,
                AVG(CASE WHEN rating_type = 'stars' THEN CAST(rating_value AS REAL) ELSE NULL END) as avg_stars
            FROM feedback
            {where}
            GROUP BY {group_column}
            ORDER BY total DESC
        """

        conn = self._get_conn()
        try:
            rows = conn.execute(query, params).fetchall()
            result = {}
            for row in rows:
                d = dict(row)
                key = d.pop("group_key")
                # Round floats
                d["avg_score"] = round(d["avg_score"] or 0.0, 4)
                d["avg_stars"] = round(d["avg_stars"] or 0.0, 2) if d["avg_stars"] else None
                result[key] = d
            return result
        finally:
            conn.close()

    def average_rating_by_model(self, since: float | None = None) -> dict[str, dict[str, Any]]:
        """Aggregate feedback stats grouped by model.

        Args:
            since: Only entries after this timestamp.

        Returns:
            Dict mapping model names to stat dicts.
        """
        return self._aggregate_group("model_used", since=since)

    def average_rating_by_pipeline(self, since: float | None = None) -> dict[str, dict[str, Any]]:
        """Aggregate feedback stats grouped by pipeline.

        Args:
            since: Only entries after this timestamp.

        Returns:
            Dict mapping pipeline names to stat dicts.
        """
        return self._aggregate_group("pipeline_used", since=since)

    def average_rating_by_task_type(self, since: float | None = None) -> dict[str, dict[str, Any]]:
        """Aggregate feedback stats grouped by task type.

        Args:
            since: Only entries after this timestamp.

        Returns:
            Dict mapping task types to stat dicts.
        """
        return self._aggregate_group("task_type", since=since)

    def get_stats(self, since: float | None = None) -> FeedbackStats:
        """Compute full feedback statistics.

        Args:
            since: Only entries after this timestamp.

        Returns:
            FeedbackStats object with all aggregated data.
        """
        conditions = []
        params: list = []

        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        conn = self._get_conn()
        try:
            # Global counts
            row = conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN rating_type = 'thumbs' AND rating_value = 1 THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN rating_type = 'thumbs' AND rating_value = 0 THEN 1 ELSE 0 END) as thumbs_down,
                    AVG(CASE
                        WHEN rating_type = 'thumbs' THEN CAST(rating_value AS REAL)
                        WHEN rating_type = 'stars' THEN (CAST(rating_value AS REAL) - 1.0) / 4.0
                        ELSE 0.0
                    END) as avg_score
                FROM feedback {where}
            """, params).fetchone()

            stats = FeedbackStats(
                total_count=row["total"] or 0,
                thumbs_up=row["thumbs_up"] or 0,
                thumbs_down=row["thumbs_down"] or 0,
                average_score=round(row["avg_score"] or 0.0, 4),
            )

            # Positive/negative counts
            stats.positive_count = stats.thumbs_up
            stats.negative_count = stats.thumbs_down

            # Star distribution
            star_rows = conn.execute(f"""
                SELECT rating_value, COUNT(*) as cnt
                FROM feedback
                {where + ' AND ' if where else 'WHERE '}rating_type = 'stars'
                GROUP BY rating_value
                ORDER BY rating_value
            """, params).fetchall()
            stats.star_distribution = {r["rating_value"]: r["cnt"] for r in star_rows}

            # Add star positive/negative counts
            for val, cnt in stats.star_distribution.items():
                if val >= 4:
                    stats.positive_count += cnt
                elif val <= 2:
                    stats.negative_count += cnt

        finally:
            conn.close()

        # Per-group breakdowns
        stats.by_model = self.average_rating_by_model(since=since)
        stats.by_pipeline = self.average_rating_by_pipeline(since=since)
        stats.by_task_type = self.average_rating_by_task_type(since=since)

        return stats

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_json(self, since: float | None = None) -> str:
        """Export all feedback as JSON string.

        Args:
            since: Only entries after this timestamp.

        Returns:
            JSON string with all feedback entries.
        """
        entries = self.list_feedback(limit=100000, since=since)
        data = [e.to_dict() for e in entries]
        return json.dumps(data, indent=2, ensure_ascii=False)

    def export_csv(self, since: float | None = None) -> str:
        """Export all feedback as CSV string.

        Args:
            since: Only entries after this timestamp.

        Returns:
            CSV string with all feedback entries.
        """
        entries = self.list_feedback(limit=100000, since=since)
        if not entries:
            return ""

        output = io.StringIO()
        fieldnames = list(entries[0].to_dict().keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.to_dict())

        return output.getvalue()


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

# S193 FBK-02: guard the singleton like the benchmark stores so an init
# failure degrades to None (deps already treats None as unavailable) instead
# of breaking the module import.
try:
    feedback_store = FeedbackStore()
except Exception as e:  # pragma: no cover
    logger.warning("FeedbackStore init failed: %s", e)
    feedback_store = None  # type: ignore[assignment]
