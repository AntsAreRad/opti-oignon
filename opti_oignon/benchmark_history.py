#!/usr/bin/env python3
"""
Benchmark History -- SQLite persistence for benchmark runs.

Stores LLM benchmark results with full run/result tracking,
comparison support, trend analysis, and regression detection.

Usage:
    from opti_oignon.benchmark_history import benchmark_history

    # Save a completed run
    benchmark_history.save_run(run_data)

    # List past runs
    runs = benchmark_history.get_runs(limit=20)

    # Compare runs
    comparison = benchmark_history.compare_runs(["id1", "id2"])
"""

import json
import logging
import sqlite3
import statistics
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
# Audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# Default data directory
DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class BenchmarkRunRecord:
    """Persisted benchmark run metadata."""
    id: str
    run_type: str = "llm"
    started_at: str = ""
    completed_at: str = ""
    status: str = "running"
    models: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    total_tests: int = 0
    avg_score: float | None = None
    best_model: str | None = None
    duration_sec: float | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BenchmarkResultRecord:
    """Persisted individual test result."""
    id: str = ""
    run_id: str = ""
    model: str = ""
    task: str = ""
    task_name: str = ""
    category: str = ""
    score: float = 0.0
    auto_score: float = 0.0
    user_score: float | None = None
    time_seconds: float = 0.0
    status: str = "success"
    response_preview: str = ""
    keywords_found: list[str] = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    error_message: str | None = None


@dataclass
class BenchmarkComparison:
    """Comparison between multiple runs."""
    runs: list[dict[str, Any]] = field(default_factory=list)
    matrix: dict[str, dict[str, list[float | None]]] = field(default_factory=dict)
    deltas: list[dict[str, Any]] = field(default_factory=list)
    regressions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelTrend:
    """Performance trend for a single model."""
    model: str = ""
    run_ids: list[str] = field(default_factory=list)
    run_dates: list[str] = field(default_factory=list)
    avg_scores: list[float] = field(default_factory=list)
    avg_times: list[float] = field(default_factory=list)


# =============================================================================
# BENCHMARK HISTORY
# =============================================================================

class BenchmarkHistory:
    """SQLite-backed storage for benchmark run history.

    Thread-safe with a reentrant lock. Stores run metadata and
    individual test results in separate tables with foreign key
    relationships.
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize the benchmark history store.

        Args:
            db_path: Path to the SQLite database file.
                     Defaults to data/benchmark_history.db.
        """
        self._db_path = db_path or (DATA_DIR / "benchmark_history.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection with row factory."""
        conn = _safe_connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Create tables if they do not exist."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS benchmark_runs (
                        id TEXT PRIMARY KEY,
                        run_type TEXT NOT NULL DEFAULT 'llm',
                        started_at TEXT NOT NULL,
                        completed_at TEXT DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'running',
                        models TEXT DEFAULT '[]',
                        tasks TEXT DEFAULT '[]',
                        total_tests INTEGER DEFAULT 0,
                        avg_score REAL,
                        best_model TEXT,
                        duration_sec REAL,
                        config_snapshot TEXT DEFAULT '{}',
                        error TEXT
                    );

                    CREATE TABLE IF NOT EXISTS benchmark_results (
                        id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        task TEXT NOT NULL,
                        task_name TEXT DEFAULT '',
                        category TEXT DEFAULT '',
                        score REAL DEFAULT 0.0,
                        auto_score REAL DEFAULT 0.0,
                        user_score REAL,
                        time_seconds REAL DEFAULT 0.0,
                        status TEXT DEFAULT 'success',
                        response_preview TEXT DEFAULT '',
                        keywords_found TEXT DEFAULT '[]',
                        keywords_missing TEXT DEFAULT '[]',
                        error_message TEXT,
                        FOREIGN KEY (run_id) REFERENCES benchmark_runs(id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_results_run_id
                        ON benchmark_results(run_id);
                    CREATE INDEX IF NOT EXISTS idx_results_model
                        ON benchmark_results(model);
                    CREATE INDEX IF NOT EXISTS idx_runs_type
                        ON benchmark_runs(run_type);
                    CREATE INDEX IF NOT EXISTS idx_runs_status
                        ON benchmark_runs(status);
                """)
                conn.commit()
            finally:
                conn.close()

    # -------------------------------------------------------------------------
    # RUN CRUD
    # -------------------------------------------------------------------------

    def save_run(self, run: BenchmarkRunRecord) -> str:
        """Save or update a benchmark run record.

        Args:
            run: The run record to persist.

        Returns:
            The run ID.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO benchmark_runs
                       (id, run_type, started_at, completed_at, status,
                        models, tasks, total_tests, avg_score, best_model,
                        duration_sec, config_snapshot, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run.id, run.run_type, run.started_at, run.completed_at,
                        run.status, json.dumps(run.models), json.dumps(run.tasks),
                        run.total_tests, run.avg_score, run.best_model,
                        run.duration_sec, json.dumps(run.config_snapshot),
                        run.error,
                    ),
                )
                conn.commit()
                return run.id
            finally:
                conn.close()

    def save_result(self, result: BenchmarkResultRecord) -> str:
        """Save an individual test result.

        Args:
            result: The result record to persist.

        Returns:
            The result ID.
        """
        if not result.id:
            result.id = str(uuid.uuid4())
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO benchmark_results
                       (id, run_id, model, task, task_name, category,
                        score, auto_score, user_score, time_seconds,
                        status, response_preview, keywords_found,
                        keywords_missing, error_message)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        result.id, result.run_id, result.model, result.task,
                        result.task_name, result.category, result.score,
                        result.auto_score, result.user_score,
                        result.time_seconds, result.status,
                        result.response_preview,
                        json.dumps(result.keywords_found),
                        json.dumps(result.keywords_missing),
                        result.error_message,
                    ),
                )
                conn.commit()
                return result.id
            finally:
                conn.close()

    def get_runs(
        self,
        run_type: str = "llm",
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List benchmark runs, newest first.

        Args:
            run_type: Filter by run type ('llm' or 'perf').
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            List of run summary dicts.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM benchmark_runs
                       WHERE run_type = ?
                       ORDER BY started_at DESC
                       LIMIT ? OFFSET ?""",
                    (run_type, limit, offset),
                ).fetchall()
                return [self._row_to_run_dict(row) for row in rows]
            finally:
                conn.close()

    def get_run_detail(self, run_id: str) -> dict[str, Any] | None:
        """Get full run detail including all results.

        Args:
            run_id: The run ID.

        Returns:
            Run dict with nested results, or None if not found.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM benchmark_runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
                if not row:
                    return None

                run_dict = self._row_to_run_dict(row)

                # Fetch all results for this run
                result_rows = conn.execute(
                    """SELECT * FROM benchmark_results
                       WHERE run_id = ?
                       ORDER BY model, task""",
                    (run_id,),
                ).fetchall()

                run_dict["results"] = [
                    self._row_to_result_dict(r) for r in result_rows
                ]

                # Compute global ranking
                run_dict["global_ranking"] = self._compute_ranking(
                    run_dict["results"]
                )

                # Best by category
                run_dict["best_by_category"] = self._compute_best_by_category(
                    run_dict["results"]
                )

                return run_dict
            finally:
                conn.close()

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its results (cascade).

        Args:
            run_id: The run ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM benchmark_runs WHERE id = ?",
                    (run_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def get_run_count(self, run_type: str = "llm") -> int:
        """Count total runs of a given type.

        Args:
            run_type: The run type filter.

        Returns:
            Count of matching runs.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM benchmark_runs WHERE run_type = ?",
                    (run_type,),
                ).fetchone()
                return row["cnt"] if row else 0
            finally:
                conn.close()

    # -------------------------------------------------------------------------
    # COMPARISON
    # -------------------------------------------------------------------------

    def compare_runs(self, run_ids: list[str]) -> BenchmarkComparison:
        """Compare multiple runs side by side.

        Builds a matrix of model x task scores across runs,
        calculates deltas, and detects regressions.

        Args:
            run_ids: List of run IDs to compare (2+).

        Returns:
            BenchmarkComparison dataclass.
        """
        comparison = BenchmarkComparison()
        if len(run_ids) < 2:
            return comparison

        with self._lock:
            conn = self._get_conn()
            try:
                # Fetch all runs
                for rid in run_ids:
                    row = conn.execute(
                        "SELECT * FROM benchmark_runs WHERE id = ?",
                        (rid,),
                    ).fetchone()
                    if row:
                        comparison.runs.append(self._row_to_run_dict(row))

                if len(comparison.runs) < 2:
                    return comparison

                # Build matrix: {model: {task: [score_run1, score_run2, ...]}}
                matrix: dict[str, dict[str, list[float | None]]] = {}
                for idx, rid in enumerate(run_ids):
                    results = conn.execute(
                        "SELECT * FROM benchmark_results WHERE run_id = ?",
                        (rid,),
                    ).fetchall()
                    for r in results:
                        model = r["model"]
                        task = r["task"]
                        if model not in matrix:
                            matrix[model] = {}
                        if task not in matrix[model]:
                            matrix[model][task] = [None] * len(run_ids)
                        matrix[model][task][idx] = r["score"]

                comparison.matrix = matrix

                # Calculate deltas between last two runs
                for model, tasks in matrix.items():
                    for task, scores in tasks.items():
                        if len(scores) >= 2 and scores[-1] is not None and scores[-2] is not None:
                            delta = scores[-1] - scores[-2]
                            direction = "improved" if delta > 0 else (
                                "regressed" if delta < 0 else "stable"
                            )
                            delta_entry = {
                                "model": model,
                                "task": task,
                                "delta": round(delta, 2),
                                "direction": direction,
                                "score_before": scores[-2],
                                "score_after": scores[-1],
                            }
                            comparison.deltas.append(delta_entry)
                            # Detect regressions (> 1.5 points drop)
                            if delta <= -1.5:
                                comparison.regressions.append(delta_entry)

                return comparison
            finally:
                conn.close()

    # -------------------------------------------------------------------------
    # TRENDS
    # -------------------------------------------------------------------------

    def get_model_trends(
        self,
        model: str,
        last_n_runs: int = 10,
    ) -> ModelTrend:
        """Get performance trend for a model over recent runs.

        Args:
            model: Model name to track.
            last_n_runs: Number of recent runs to include.

        Returns:
            ModelTrend dataclass with scores and times per run.
        """
        trend = ModelTrend(model=model)
        with self._lock:
            conn = self._get_conn()
            try:
                # Find runs that include this model
                rows = conn.execute(
                    """SELECT DISTINCT r.run_id, br.started_at, br.id as run_pk
                       FROM benchmark_results r
                       JOIN benchmark_runs br ON br.id = r.run_id
                       WHERE r.model = ? AND br.status = 'completed'
                       ORDER BY br.started_at DESC
                       LIMIT ?""",
                    (model, last_n_runs),
                ).fetchall()

                # Process each run (reverse to get chronological order)
                for row in reversed(rows):
                    run_id = row["run_id"]
                    results = conn.execute(
                        """SELECT score, time_seconds FROM benchmark_results
                           WHERE run_id = ? AND model = ? AND status = 'success'""",
                        (run_id, model),
                    ).fetchall()
                    if results:
                        scores = [r["score"] for r in results]
                        times = [r["time_seconds"] for r in results]
                        trend.run_ids.append(run_id)
                        trend.run_dates.append(row["started_at"])
                        trend.avg_scores.append(
                            round(statistics.mean(scores), 2)
                        )
                        trend.avg_times.append(
                            round(statistics.mean(times), 2)
                        )

                return trend
            finally:
                conn.close()

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _row_to_run_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a SQLite row to a run dictionary."""
        d = dict(row)
        d["models"] = json.loads(d.get("models", "[]"))
        d["tasks"] = json.loads(d.get("tasks", "[]"))
        d["config_snapshot"] = json.loads(d.get("config_snapshot", "{}"))
        # Compute models_tested / tasks_tested aliases
        d["models_tested"] = d["models"]
        d["tasks_tested"] = d["tasks"]
        return d

    def _row_to_result_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a SQLite row to a result dictionary."""
        d = dict(row)
        d["keywords_found"] = json.loads(d.get("keywords_found", "[]"))
        d["keywords_missing"] = json.loads(d.get("keywords_missing", "[]"))
        return d

    def _compute_ranking(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Compute global ranking from result list."""
        model_scores: dict[str, list[float]] = {}
        model_times: dict[str, list[float]] = {}
        for r in results:
            model = r["model"]
            if r["status"] == "success":
                model_scores.setdefault(model, []).append(r["score"])
                model_times.setdefault(model, []).append(r["time_seconds"])

        ranking = []
        for model, scores in model_scores.items():
            avg_score = round(statistics.mean(scores), 2) if scores else 0.0
            avg_time = round(
                statistics.mean(model_times.get(model, [0])), 2
            )
            ranking.append({
                "model": model,
                "avg_score": avg_score,
                "avg_time": avg_time,
                "tests": len(scores),
            })

        ranking.sort(key=lambda x: x["avg_score"], reverse=True)
        for i, entry in enumerate(ranking):
            entry["rank"] = i + 1
        return ranking

    def _compute_best_by_category(
        self, results: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Find the best model per category."""
        category_scores: dict[str, dict[str, list[float]]] = {}
        for r in results:
            cat = r.get("category", "general")
            model = r["model"]
            if r["status"] == "success":
                category_scores.setdefault(cat, {}).setdefault(
                    model, []
                ).append(r["score"])

        best: dict[str, str] = {}
        for cat, models in category_scores.items():
            best_model = ""
            best_avg = -1.0
            for model, scores in models.items():
                avg = statistics.mean(scores) if scores else 0.0
                if avg > best_avg:
                    best_avg = avg
                    best_model = model
            if best_model:
                best[cat] = best_model
        return best


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

try:
    benchmark_history = BenchmarkHistory()
    logger.info("BenchmarkHistory initialized at %s", benchmark_history._db_path)
except Exception as e:
    logger.warning("BenchmarkHistory initialization failed: %s", e)
    benchmark_history = None
