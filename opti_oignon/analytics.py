#!/usr/bin/env python3
"""
ANALYTICS ENGINE -- Performance Tracking & Trend Analysis (S55)
=================================================================

Tracks response times, token counts, pipeline/model usage, and
computes time-windowed trends. Optionally feeds metrics back to
the SmartRouter to adjust task_scores based on real performance.

SQLite-backed for persistence across restarts.

Author: Leon
"""

import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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

_DATA_DIR = Path(__file__).parent / "data"
_CONFIG_DIR = Path(__file__).parent / "config"
_DEFAULT_DB_PATH = _DATA_DIR / "analytics.db"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "feedback.yaml"

# Time window parsing map (suffix -> seconds)
_WINDOW_MULTIPLIERS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def _parse_window(window_str: str) -> int:
    """Parse a time window string like '24h', '7d', '1w' into seconds.

    Args:
        window_str: Time window string (e.g., '1h', '24h', '7d').

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If the format is invalid.
    """
    if not window_str or len(window_str) < 2:
        raise ValueError(f"Invalid window format: {window_str}")

    suffix = window_str[-1].lower()
    if suffix not in _WINDOW_MULTIPLIERS:
        raise ValueError(f"Unknown window suffix '{suffix}' in '{window_str}'")

    try:
        value = int(window_str[:-1])
    except ValueError:
        raise ValueError(f"Invalid numeric part in '{window_str}'")

    return value * _WINDOW_MULTIPLIERS[suffix]


# =============================================================================
# SAFE SQL QUERY BUILDER (S156 -- SA-155-020)
# =============================================================================

# Allowlist of valid SQL condition fragments for dynamic WHERE construction.
# Only fragments from this set are allowed in query building, preventing
# any injection via dynamic clause composition.
_ALLOWED_CONDITIONS: frozenset[str] = frozenset({
    "timestamp >= ?",
    "timestamp <= ?",
    "timestamp < ?",
    "model_used = ?",
    "pipeline_used = ?",
    "pipeline_used != ''",
    "model_used != ''",
    "task_type != ''",
    "was_routed = 1",
    "was_routed = 0",
})


def _build_where(conditions: list[str], params: list | None = None) -> str:
    """Build a safe SQL WHERE clause from validated condition fragments.

    Each condition must be in the _ALLOWED_CONDITIONS allowlist.
    Raises ValueError if an unknown condition is encountered.

    Args:
        conditions: List of SQL condition fragments (e.g., ["timestamp >= ?"]).
        params: Optional parameter list for validation (length check only).

    Returns:
        A WHERE clause string (including "WHERE") or empty string if no conditions.
    """
    if not conditions:
        return ""
    for cond in conditions:
        if cond not in _ALLOWED_CONDITIONS:
            raise ValueError(f"Disallowed SQL condition fragment: {cond!r}")
    return "WHERE " + " AND ".join(conditions)



@dataclass
class PerformanceRecord:
    """Single performance measurement for a request."""

    # Identifiers
    record_id: str = ""
    conversation_id: str = ""
    message_id: str = ""

    # Model & pipeline info
    model_used: str = ""
    pipeline_used: str = ""
    task_type: str = ""

    # Performance metrics
    response_time_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0

    # Routing info
    was_routed: bool = False
    routing_score: float = 0.0

    # Status
    success: bool = True
    error_message: str = ""

    # Timing
    timestamp: float = 0.0

    def __post_init__(self):
        """Generate record_id and timestamp if not provided."""
        if not self.record_id:
            self.record_id = str(uuid.uuid4())[:12]
        if self.timestamp <= 0:
            self.timestamp = time.time()
        # Auto-compute tokens_per_second
        if self.tokens_per_second <= 0 and self.completion_tokens > 0 and self.response_time_ms > 0:
            self.tokens_per_second = round(
                self.completion_tokens / (self.response_time_ms / 1000.0), 2
            )
        # Auto-compute total_tokens
        if self.total_tokens <= 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceRecord":
        """Create from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class TrendPoint:
    """Single point in a time-series trend."""
    window_start: float = 0.0
    window_end: float = 0.0
    count: int = 0
    avg_response_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    total_tokens: int = 0
    success_rate: float = 1.0


@dataclass
class AnalyticsOverview:
    """High-level analytics overview for the dashboard."""
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    success_rate: float = 1.0
    avg_response_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    total_tokens_processed: int = 0
    # Distributions
    pipeline_distribution: dict[str, int] = field(default_factory=dict)
    model_distribution: dict[str, int] = field(default_factory=dict)
    task_type_distribution: dict[str, int] = field(default_factory=dict)
    # Per-model performance
    model_performance: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per-pipeline performance
    pipeline_performance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

class PerformanceTracker:
    """Tracks individual request performance metrics in SQLite.

    Records response times, token counts, model/pipeline usage for
    every processed request. Provides query methods for analytics.
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._init_db()
        logger.info("PerformanceTracker initialized (db=%s)", self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a SQLite connection."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create performance table if it does not exist."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    record_id TEXT PRIMARY KEY,
                    conversation_id TEXT DEFAULT '',
                    message_id TEXT DEFAULT '',
                    model_used TEXT DEFAULT '',
                    pipeline_used TEXT DEFAULT '',
                    task_type TEXT DEFAULT '',
                    response_time_ms REAL DEFAULT 0.0,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    tokens_per_second REAL DEFAULT 0.0,
                    was_routed INTEGER DEFAULT 0,
                    routing_score REAL DEFAULT 0.0,
                    success INTEGER DEFAULT 1,
                    error_message TEXT DEFAULT '',
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_perf_model
                ON performance(model_used)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_perf_pipeline
                ON performance(pipeline_used)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_perf_timestamp
                ON performance(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_perf_task
                ON performance(task_type)
            """)
            conn.commit()
        finally:
            conn.close()

    def record(self, entry: PerformanceRecord) -> PerformanceRecord:
        """Record a performance measurement.

        Args:
            entry: PerformanceRecord to store.

        Returns:
            The stored record.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO performance
                (record_id, conversation_id, message_id, model_used,
                 pipeline_used, task_type, response_time_ms, prompt_tokens,
                 completion_tokens, total_tokens, tokens_per_second,
                 was_routed, routing_score, success, error_message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.record_id,
                    entry.conversation_id,
                    entry.message_id,
                    entry.model_used,
                    entry.pipeline_used,
                    entry.task_type,
                    entry.response_time_ms,
                    entry.prompt_tokens,
                    entry.completion_tokens,
                    entry.total_tokens,
                    entry.tokens_per_second,
                    int(entry.was_routed),
                    entry.routing_score,
                    int(entry.success),
                    entry.error_message,
                    entry.timestamp,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        logger.debug("Performance recorded: %s (model=%s, %.0fms)",
                      entry.record_id, entry.model_used, entry.response_time_ms)
        return entry

    def get_records(
        self,
        limit: int = 100,
        offset: int = 0,
        model: str | None = None,
        pipeline: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[PerformanceRecord]:
        """Query performance records with filters.

        Args:
            limit: Maximum records to return.
            offset: Pagination offset.
            model: Filter by model name.
            pipeline: Filter by pipeline name.
            since: Only records after this timestamp.
            until: Only records before this timestamp.

        Returns:
            List of PerformanceRecord objects.
        """
        conditions: list[str] = []
        params: list = []

        if model:
            conditions.append("model_used = ?")
            params.append(model)
        if pipeline:
            conditions.append("pipeline_used = ?")
            params.append(pipeline)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(until)

        where = _build_where(conditions)

        query = (
            "SELECT * FROM performance " + where
            + " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        conn = self._get_conn()
        try:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                # Convert integer booleans back
                d["was_routed"] = bool(d.get("was_routed", 0))
                d["success"] = bool(d.get("success", 1))
                results.append(PerformanceRecord.from_dict(d))
            return results
        finally:
            conn.close()

    def count(self, since: float | None = None) -> int:
        """Total number of performance records.

        Args:
            since: Only count records after this timestamp.

        Returns:
            Record count.
        """
        conditions: list[str] = []
        params: list = []

        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = _build_where(conditions)

        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM performance " + where, params
            ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def clear(self, before: float | None = None) -> int:
        """Delete performance records.

        Args:
            before: Only delete records before this timestamp.
                    If None, deletes all records.

        Returns:
            Number of records deleted.
        """
        conn = self._get_conn()
        try:
            if before is not None:
                cursor = conn.execute(
                    "DELETE FROM performance WHERE timestamp < ?", (before,)
                )
            else:
                cursor = conn.execute("DELETE FROM performance")
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()


# =============================================================================
# ANALYTICS ENGINE
# =============================================================================

class AnalyticsEngine:
    """Computes analytics and trends from performance data.

    Aggregates PerformanceTracker data into overview dashboards,
    time-series trends, and per-model/pipeline breakdowns.
    Optionally integrates with FeedbackStore for routing accuracy.
    """

    def __init__(
        self,
        tracker: PerformanceTracker | None = None,
        config_path: Path | None = None,
    ):
        self._tracker = tracker or PerformanceTracker()
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._config: dict[str, Any] = {}
        self._load_config()
        logger.info("AnalyticsEngine initialized")

    def _load_config(self) -> None:
        """Load analytics configuration from YAML."""
        try:
            if self._config_path.exists():
                with open(self._config_path) as f:
                    raw = yaml.safe_load(f) or {}
                self._config = raw.get("analytics", {})
            else:
                self._config = {}
        except Exception as e:
            logger.warning("Failed to load analytics config: %s", e)
            self._config = {}

    @property
    def enabled(self) -> bool:
        """Whether analytics tracking is enabled."""
        return self._config.get("enabled", True)

    @property
    def retention_seconds(self) -> int:
        """How long to retain performance records."""
        return self._config.get("retention_seconds", 2592000)

    @property
    def trend_windows(self) -> list[str]:
        """Configured trend time windows."""
        return self._config.get("trend_windows", ["1h", "24h", "7d", "30d"])

    @property
    def tracker(self) -> PerformanceTracker:
        """Access the underlying PerformanceTracker."""
        return self._tracker

    def get_overview(self, since: float | None = None) -> AnalyticsOverview:
        """Compute a full analytics overview.

        Args:
            since: Only include data after this timestamp.

        Returns:
            AnalyticsOverview with all metrics.
        """
        db_path = self._tracker._db_path
        conn = _safe_connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        conditions: list[str] = []
        params: list = []

        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        where = _build_where(conditions)

        try:
            # Global metrics
            row = conn.execute(
                "SELECT"
                " COUNT(*) as total,"
                " SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_cnt,"
                " SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_cnt,"
                " AVG(response_time_ms) as avg_rt,"
                " AVG(CASE WHEN tokens_per_second > 0 THEN tokens_per_second ELSE NULL END) as avg_tps,"
                " SUM(total_tokens) as sum_tokens"
                " FROM performance " + where,
                params,
            ).fetchone()

            total = row["total"] or 0
            overview = AnalyticsOverview(
                total_requests=total,
                success_count=row["success_cnt"] or 0,
                error_count=row["error_cnt"] or 0,
                success_rate=round((row["success_cnt"] or 0) / max(total, 1), 4),
                avg_response_time_ms=round(row["avg_rt"] or 0.0, 2),
                avg_tokens_per_second=round(row["avg_tps"] or 0.0, 2),
                total_tokens_processed=row["sum_tokens"] or 0,
            )

            # Pipeline distribution
            pipe_where = _build_where(conditions + ["pipeline_used != ''"])
            rows = conn.execute(
                "SELECT pipeline_used, COUNT(*) as cnt"
                " FROM performance " + pipe_where
                + " GROUP BY pipeline_used ORDER BY cnt DESC",
                params,
            ).fetchall()
            overview.pipeline_distribution = {r["pipeline_used"]: r["cnt"] for r in rows}

            # Model distribution
            model_where = _build_where(conditions + ["model_used != ''"])
            rows = conn.execute(
                "SELECT model_used, COUNT(*) as cnt"
                " FROM performance " + model_where
                + " GROUP BY model_used ORDER BY cnt DESC",
                params,
            ).fetchall()
            overview.model_distribution = {r["model_used"]: r["cnt"] for r in rows}

            # Task type distribution
            task_where = _build_where(conditions + ["task_type != ''"])
            rows = conn.execute(
                "SELECT task_type, COUNT(*) as cnt"
                " FROM performance " + task_where
                + " GROUP BY task_type ORDER BY cnt DESC",
                params,
            ).fetchall()
            overview.task_type_distribution = {r["task_type"]: r["cnt"] for r in rows}

            # Per-model performance
            rows = conn.execute(
                "SELECT"
                " model_used,"
                " COUNT(*) as cnt,"
                " AVG(response_time_ms) as avg_rt,"
                " AVG(CASE WHEN tokens_per_second > 0 THEN tokens_per_second ELSE NULL END) as avg_tps,"
                " SUM(total_tokens) as sum_tokens,"
                " AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as sr"
                " FROM performance " + model_where
                + " GROUP BY model_used ORDER BY cnt DESC",
                params,
            ).fetchall()
            for r in rows:
                overview.model_performance[r["model_used"]] = {
                    "count": r["cnt"],
                    "avg_response_time_ms": round(r["avg_rt"] or 0.0, 2),
                    "avg_tokens_per_second": round(r["avg_tps"] or 0.0, 2),
                    "total_tokens": r["sum_tokens"] or 0,
                    "success_rate": round(r["sr"] or 0.0, 4),
                }

            # Per-pipeline performance
            rows = conn.execute(
                "SELECT"
                " pipeline_used,"
                " COUNT(*) as cnt,"
                " AVG(response_time_ms) as avg_rt,"
                " AVG(CASE WHEN tokens_per_second > 0 THEN tokens_per_second ELSE NULL END) as avg_tps,"
                " AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as sr"
                " FROM performance " + pipe_where
                + " GROUP BY pipeline_used ORDER BY cnt DESC",
                params,
            ).fetchall()
            for r in rows:
                overview.pipeline_performance[r["pipeline_used"]] = {
                    "count": r["cnt"],
                    "avg_response_time_ms": round(r["avg_rt"] or 0.0, 2),
                    "avg_tokens_per_second": round(r["avg_tps"] or 0.0, 2),
                    "success_rate": round(r["sr"] or 0.0, 4),
                }

        finally:
            conn.close()

        return overview

    def get_trends(
        self,
        window: str = "24h",
        buckets: int = 24,
        model: str | None = None,
        pipeline: str | None = None,
    ) -> list[TrendPoint]:
        """Compute time-series trend points for a given window.

        Divides the window into equal-sized buckets and computes
        aggregate metrics for each bucket.

        Args:
            window: Time window string (e.g., '1h', '24h', '7d').
            buckets: Number of time buckets to divide the window into.
            model: Optional model filter.
            pipeline: Optional pipeline filter.

        Returns:
            List of TrendPoint objects, oldest first.
        """
        window_seconds = _parse_window(window)
        now = time.time()
        window_start = now - window_seconds
        bucket_size = window_seconds / max(buckets, 1)

        db_path = self._tracker._db_path
        conn = _safe_connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # Build filter conditions
        conditions = ["timestamp >= ?"]
        params: list = [window_start]

        if model:
            conditions.append("model_used = ?")
            params.append(model)
        if pipeline:
            conditions.append("pipeline_used = ?")
            params.append(pipeline)

        where = _build_where(conditions)

        try:
            # Fetch all records in the window
            rows = conn.execute(
                "SELECT * FROM performance " + where + " ORDER BY timestamp ASC",
                params,
            ).fetchall()
        finally:
            conn.close()

        # Build buckets
        trends: list[TrendPoint] = []
        for i in range(buckets):
            b_start = window_start + i * bucket_size
            b_end = b_start + bucket_size

            # Filter records for this bucket
            bucket_records = [
                r for r in rows
                if b_start <= r["timestamp"] < b_end
            ]

            count = len(bucket_records)
            if count == 0:
                trends.append(TrendPoint(
                    window_start=b_start,
                    window_end=b_end,
                    count=0,
                ))
                continue

            avg_rt = sum(r["response_time_ms"] for r in bucket_records) / count
            tps_values = [r["tokens_per_second"] for r in bucket_records if r["tokens_per_second"] > 0]
            avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0
            total_tok = sum(r["total_tokens"] for r in bucket_records)
            success_cnt = sum(1 for r in bucket_records if r["success"])
            sr = success_cnt / count

            trends.append(TrendPoint(
                window_start=b_start,
                window_end=b_end,
                count=count,
                avg_response_time_ms=round(avg_rt, 2),
                avg_tokens_per_second=round(avg_tps, 2),
                total_tokens=total_tok,
                success_rate=round(sr, 4),
            ))

        return trends

    def get_routing_accuracy(self, since: float | None = None) -> dict[str, Any]:
        """Compute routing accuracy by correlating routing with feedback.

        Compares routed vs non-routed request success rates and
        feedback scores when FeedbackStore is available.

        Args:
            since: Only include data after this timestamp.

        Returns:
            Dict with routing accuracy metrics.
        """
        db_path = self._tracker._db_path
        conn = _safe_connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        conditions: list[str] = []
        params: list = []

        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        routed_where = _build_where(conditions + ["was_routed = 1"])
        unrouted_where = _build_where(conditions + ["was_routed = 0"])

        try:
            # Routed vs non-routed performance
            row_routed = conn.execute(
                "SELECT"
                " COUNT(*) as cnt,"
                " AVG(response_time_ms) as avg_rt,"
                " AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as sr"
                " FROM performance " + routed_where,
                params,
            ).fetchone()

            row_unrouted = conn.execute(
                "SELECT"
                " COUNT(*) as cnt,"
                " AVG(response_time_ms) as avg_rt,"
                " AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as sr"
                " FROM performance " + unrouted_where,
                params,
            ).fetchone()

        finally:
            conn.close()

        return {
            "routed": {
                "count": row_routed["cnt"] or 0,
                "avg_response_time_ms": round(row_routed["avg_rt"] or 0.0, 2),
                "success_rate": round(row_routed["sr"] or 0.0, 4),
            },
            "unrouted": {
                "count": row_unrouted["cnt"] or 0,
                "avg_response_time_ms": round(row_unrouted["avg_rt"] or 0.0, 2),
                "success_rate": round(row_unrouted["sr"] or 0.0, 4),
            },
        }

    def cleanup_old_records(self) -> int:
        """Delete records older than the configured retention period.

        Returns:
            Number of records deleted.
        """
        cutoff = time.time() - self.retention_seconds
        deleted = self._tracker.clear(before=cutoff)
        if deleted > 0:
            logger.info("Analytics cleanup: removed %d old records", deleted)
        return deleted


# =============================================================================
# MODULE-LEVEL SINGLETONS
# =============================================================================

performance_tracker = PerformanceTracker()
analytics_engine = AnalyticsEngine(tracker=performance_tracker)
