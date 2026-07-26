#!/usr/bin/env python3
"""
Performance Monitor.

Real-time metrics collection for token throughput, latency, model utilization,
quality drift detection, and rule-based optimization recommendations.

Stores execution records in a dedicated SQLite database
(performance_metrics.db) with configurable retention.
"""

import logging
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
# Audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricsRecord:
    """Single execution record."""
    model: str
    task_type: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    quality_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LatencyStats:
    """Latency percentile statistics."""
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    count: int = 0


@dataclass
class DriftResult:
    """Drift detection result for a metric."""
    model: str
    metric: str
    baseline_value: float
    recent_value: float
    change_ratio: float
    is_drifted: bool
    direction: str  # "up" or "down"


@dataclass
class Recommendation:
    """Actionable optimization recommendation."""
    model: str
    metric: str
    message: str
    severity: str  # "info", "warning", "critical"
    value: float = 0.0


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "enabled": True,
    "retention_days": 7,
    "default_window_seconds": 300,
    "drift": {
        "window_seconds": 3600,
        "baseline_window_seconds": 86400,
        "threshold": 0.3,
    },
    "recommendation_rules": [],
}

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "performance.yaml"


def _load_config(config_path: Path | None = None) -> dict:
    """Load performance config from YAML with defaults fallback."""
    path = config_path or _CONFIG_PATH
    cfg = dict(_DEFAULT_CONFIG)
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            # Merge top-level keys
            for key in ("enabled", "retention_days", "default_window_seconds",
                        "recommendation_rules"):
                if key in raw:
                    cfg[key] = raw[key]
            # Merge drift sub-dict
            if "drift" in raw and isinstance(raw["drift"], dict):
                cfg["drift"] = {**cfg["drift"], **raw["drift"]}
    except Exception as e:
        logger.warning("Failed to load performance config: %s — using defaults", e)
    return cfg


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Collects execution metrics and provides aggregated statistics,
    drift detection, and optimization recommendations.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        config_path: Path | None = None,
    ):
        self._config = _load_config(config_path)
        self._enabled = self._config["enabled"]
        self._retention_days = self._config["retention_days"]
        self._default_window = self._config["default_window_seconds"]
        self._drift_cfg = self._config["drift"]
        self._rules = self._config.get("recommendation_rules", [])

        if db_path is None:
            db_path = Path(__file__).resolve().parent / "data" / "performance_metrics.db"
        self._db_path = str(db_path)

        self._lock = threading.Lock()
        # Opportunistic retention. performance_metrics.db had no
        # auto-purge (cleanup only via the manual /api/performance/cleanup
        # route), so it grew unbounded on a daily-use machine. Run cleanup at
        # most once per retention-check interval from record_execution, with
        # no extra background thread.
        self._last_cleanup_ts: float = 0.0
        self._cleanup_interval_s: float = 86400.0
        self._init_db()

    # -- Database ----------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        conn = _safe_connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create the metrics table if it does not exist."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_ts
                ON metrics(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_model_ts
                ON metrics(model, timestamp)
            """)
            conn.commit()
        finally:
            conn.close()

    # -- Recording ---------------------------------------------------------

    def record_execution(
        self,
        model: str,
        task_type: str,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        quality_score: float,
        timestamp: float | None = None,
    ) -> MetricsRecord | None:
        """
        Log a single execution metric.

        Returns the MetricsRecord if stored, or None if monitoring is disabled.
        """
        if not self._enabled:
            return None

        ts = timestamp or time.time()
        record = MetricsRecord(
            model=model,
            task_type=task_type,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            quality_score=quality_score,
            timestamp=ts,
        )

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO metrics
                   (model, task_type, latency_ms, tokens_in, tokens_out,
                    quality_score, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (model, task_type, latency_ms, tokens_in, tokens_out,
                 quality_score, ts),
            )
            conn.commit()
        finally:
            conn.close()

        # Opportunistic retention (no background thread).
        if ts - self._last_cleanup_ts > self._cleanup_interval_s:
            self._last_cleanup_ts = ts
            try:
                self.cleanup_old_records()
            except Exception as exc:  # pragma: no cover
                logger.debug("Opportunistic perf cleanup failed: %s", exc)

        return record

    # -- Throughput --------------------------------------------------------

    def get_throughput(self, window_seconds: int | None = None) -> dict:
        """
        Compute token throughput over a rolling window.

        Returns dict with tokens_in_per_sec, tokens_out_per_sec,
        total_tokens, request_count, window_seconds.
        """
        window = window_seconds or self._default_window
        cutoff = time.time() - window

        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT
                     COALESCE(SUM(tokens_in), 0) AS total_in,
                     COALESCE(SUM(tokens_out), 0) AS total_out,
                     COUNT(*) AS cnt
                   FROM metrics WHERE timestamp >= ?""",
                (cutoff,),
            ).fetchone()
        finally:
            conn.close()

        total_in = row["total_in"]
        total_out = row["total_out"]
        cnt = row["cnt"]

        return {
            "tokens_in_per_sec": total_in / window if window > 0 else 0.0,
            "tokens_out_per_sec": total_out / window if window > 0 else 0.0,
            "total_tokens": total_in + total_out,
            "request_count": cnt,
            "window_seconds": window,
        }

    # -- Latency -----------------------------------------------------------

    def get_latency_stats(
        self,
        model: str | None = None,
        window_seconds: int | None = None,
    ) -> LatencyStats:
        """
        Compute latency percentiles (p50, p95, p99) over a window.

        If model is None, aggregates across all models.
        """
        window = window_seconds or self._default_window
        cutoff = time.time() - window

        conn = self._get_conn()
        try:
            if model:
                rows = conn.execute(
                    "SELECT latency_ms FROM metrics WHERE model = ? AND timestamp >= ? ORDER BY latency_ms",
                    (model, cutoff),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT latency_ms FROM metrics WHERE timestamp >= ? ORDER BY latency_ms",
                    (cutoff,),
                ).fetchall()
        finally:
            conn.close()

        if not rows:
            return LatencyStats()

        latencies = [r["latency_ms"] for r in rows]
        n = len(latencies)

        return LatencyStats(
            p50=self._percentile(latencies, 50),
            p95=self._percentile(latencies, 95),
            p99=self._percentile(latencies, 99),
            mean=statistics.mean(latencies),
            count=n,
        )

    @staticmethod
    def _percentile(sorted_values: list[float], pct: int) -> float:
        """Compute a percentile from a sorted list."""
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        idx = (pct / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])

    # -- Utilization -------------------------------------------------------

    def get_model_utilization(
        self, window_seconds: int | None = None
    ) -> dict[str, float]:
        """
        Get model usage distribution over a window.

        Returns dict mapping model name to fraction of total requests (0.0-1.0).
        """
        window = window_seconds or 3600
        cutoff = time.time() - window

        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT model, COUNT(*) AS cnt
                   FROM metrics WHERE timestamp >= ?
                   GROUP BY model ORDER BY cnt DESC""",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {}

        total = sum(r["cnt"] for r in rows)
        return {r["model"]: r["cnt"] / total for r in rows}

    # -- Drift detection ---------------------------------------------------

    def detect_drift(
        self,
        model: str,
        metric: str = "latency",
        threshold: float | None = None,
    ) -> DriftResult | None:
        """
        Compare recent metric values vs historical baseline.

        Supported metrics: "latency", "quality".
        Returns DriftResult or None if insufficient data.
        """
        drift_threshold = threshold or self._drift_cfg["threshold"]
        recent_window = self._drift_cfg["window_seconds"]
        baseline_window = self._drift_cfg["baseline_window_seconds"]
        now = time.time()

        recent_cutoff = now - recent_window
        baseline_cutoff = now - baseline_window

        conn = self._get_conn()
        try:
            if metric == "latency":
                col = "latency_ms"
            elif metric == "quality":
                col = "quality_score"
            else:
                return None

            # Validate column name against allowlist
            _METRIC_COLS = frozenset({"latency_ms", "quality_score"})
            assert col in _METRIC_COLS, f"Invalid metric column: {col}"

            # Recent average
            row_recent = conn.execute(
                f"SELECT AVG({col}) AS val, COUNT(*) AS cnt FROM metrics "
                "WHERE model = ? AND timestamp >= ?",
                (model, recent_cutoff),
            ).fetchone()

            # Baseline average (older data only)
            row_baseline = conn.execute(
                f"SELECT AVG({col}) AS val, COUNT(*) AS cnt FROM metrics "
                "WHERE model = ? AND timestamp >= ? AND timestamp < ?",
                (model, baseline_cutoff, recent_cutoff),
            ).fetchone()
        finally:
            conn.close()

        if (
            not row_recent or row_recent["cnt"] == 0
            or not row_baseline or row_baseline["cnt"] == 0
        ):
            return None

        recent_val = row_recent["val"]
        baseline_val = row_baseline["val"]

        if baseline_val == 0:
            return None

        change = (recent_val - baseline_val) / abs(baseline_val)
        is_drifted = abs(change) >= drift_threshold
        direction = "up" if change > 0 else "down"

        return DriftResult(
            model=model,
            metric=metric,
            baseline_value=baseline_val,
            recent_value=recent_val,
            change_ratio=change,
            is_drifted=is_drifted,
            direction=direction,
        )

    def detect_all_drift(self) -> list[DriftResult]:
        """Run drift detection for all active models, all metrics."""
        results = []
        models = self._get_active_models()
        for model in models:
            for metric in ("latency", "quality"):
                dr = self.detect_drift(model, metric)
                if dr and dr.is_drifted:
                    results.append(dr)
        return results

    def _get_active_models(self) -> list[str]:
        """Get list of models with recent metrics."""
        cutoff = time.time() - self._drift_cfg["baseline_window_seconds"]
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT model FROM metrics WHERE timestamp >= ?",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        return [r["model"] for r in rows]

    # -- Recommendations ---------------------------------------------------

    def get_recommendations(self) -> list[Recommendation]:
        """
        Generate rule-based optimization recommendations.

        Evaluates each configured rule against current metrics.
        """
        recs = []
        models = self._get_active_models()

        for model in models:
            for rule in self._rules:
                rec = self._evaluate_rule(model, rule)
                if rec:
                    recs.append(rec)

        return recs

    def _evaluate_rule(self, model: str, rule: dict) -> Recommendation | None:
        """Evaluate a single recommendation rule for a model."""
        metric_name = rule.get("metric", "")
        condition = rule.get("condition", "gt")
        threshold = rule.get("threshold", 0)
        message_tpl = rule.get("message", "")

        value = self._get_rule_metric_value(model, metric_name)
        if value is None:
            return None

        triggered = False
        if condition == "gt" and value > threshold:
            triggered = True
        elif condition == "lt" and value < threshold:
            triggered = True
        elif condition == "gte" and value >= threshold:
            triggered = True
        elif condition == "lte" and value <= threshold:
            triggered = True

        if not triggered:
            return None

        # Determine severity
        severity = "warning"
        if abs(value) > threshold * 1.5 if threshold != 0 else False:
            severity = "critical"

        try:
            msg = message_tpl.format(model=model, value=value)
        except (KeyError, ValueError):
            msg = message_tpl

        return Recommendation(
            model=model,
            metric=metric_name,
            message=msg,
            severity=severity,
            value=value,
        )

    def _get_rule_metric_value(self, model: str, metric_name: str) -> float | None:
        """Resolve a metric value for rule evaluation."""
        if metric_name == "latency_p95":
            stats = self.get_latency_stats(model=model)
            return stats.p95 if stats.count > 0 else None

        elif metric_name == "latency_drift":
            dr = self.detect_drift(model, "latency")
            return dr.change_ratio if dr else None

        elif metric_name == "quality_drift":
            dr = self.detect_drift(model, "quality")
            return dr.change_ratio if dr else None

        elif metric_name == "utilization":
            util = self.get_model_utilization()
            return util.get(model)

        elif metric_name == "error_rate":
            # Placeholder: error_rate not yet tracked as a separate field
            return None

        return None

    # -- History -----------------------------------------------------------

    def get_history(
        self,
        model: str | None = None,
        hours: int = 24,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Retrieve raw metric history.

        Returns list of dicts, most recent first.
        """
        cutoff = time.time() - (hours * 3600)
        conn = self._get_conn()
        try:
            if model:
                rows = conn.execute(
                    """SELECT model, task_type, latency_ms, tokens_in, tokens_out,
                              quality_score, timestamp
                       FROM metrics WHERE model = ? AND timestamp >= ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (model, cutoff, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT model, task_type, latency_ms, tokens_in, tokens_out,
                              quality_score, timestamp
                       FROM metrics WHERE timestamp >= ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (cutoff, limit),
                ).fetchall()
        finally:
            conn.close()

        return [dict(r) for r in rows]

    # -- Cleanup -----------------------------------------------------------

    def cleanup_old_records(self) -> int:
        """Delete records older than retention_days. Returns count deleted."""
        cutoff = time.time() - (self._retention_days * 86400)
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            deleted = cursor.rowcount
        finally:
            conn.close()
        return deleted

    # -- Summary -----------------------------------------------------------

    def get_summary(self) -> dict:
        """Get a complete performance summary (throughput + latency + utilization)."""
        return {
            "throughput": self.get_throughput(),
            "latency": {
                "p50": self.get_latency_stats().p50,
                "p95": self.get_latency_stats().p95,
                "p99": self.get_latency_stats().p99,
                "mean": self.get_latency_stats().mean,
                "count": self.get_latency_stats().count,
            },
            "utilization": self.get_model_utilization(),
            "enabled": self._enabled,
        }

    # -- Properties --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def config(self) -> dict:
        return dict(self._config)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

try:
    performance_monitor = PerformanceMonitor()
    PERFORMANCE_MONITOR_AVAILABLE = True
    logger.info("PerformanceMonitor initialized (S72)")
except Exception as _e:
    logger.warning("PerformanceMonitor unavailable: %s", _e)
    performance_monitor = None
    PERFORMANCE_MONITOR_AVAILABLE = False
