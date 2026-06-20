#!/usr/bin/env python3
"""
Agent eval results store -- S230 (AGT_SPEC Section 7.3).

SQLite-backed storage on the ResultsStore idiom (benchmark_runner.py is
the source idiom: lock + safe_connect, executescript schema, parameterized
queries throughout). The store owns its OWN database,
data/agent_eval_results.db, kept deliberately SEPARATE from
benchmark_results.db: model-quality benchmarks and agent-capability evals
are different registers with different lifecycles, and folding them would
conflate the ATREST rows and the history semantics.

ATREST disposition (declared in AGT_SPEC 7.3, landed as a matrix row this
lot): kind DB, scope single-user, wipe pending-scoping (the FBK-01
family), backup excluded (telemetry-class data); any future config follows
the post-BK-06 additive rule, config-only.

Honest provenance columns (spec 7.2/7.3): governor_present on the run row,
admitted / admitted_ctx / failure_class (including "not_admitted") on every
task row -- visible, never masked.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module conventions (project-wide).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Encrypted DB connections (the S136 posture), with the established
# guarded fallback for standalone loading.
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:  # pragma: no cover - standalone loading fallback
    import sqlite3 as _sq3

    def _safe_connect(p, **kw):
        return _sq3.connect(str(p), **kw)


_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_DB_PATH = _DATA_DIR / "agent_eval_results.db"

# The failure taxonomy (AGT_SPEC 7.3, verbatim).
FAILURE_CLASSES = (
    "none",
    "test_fail",
    "timeout",
    "doom_loop",
    "refusal",
    "not_admitted",
    "error",
)

# Run lifecycle statuses (the benchmark register's vocabulary).
RUN_STATUSES = ("pending", "running", "completed", "failed", "cancelled")


class EvalResultsStore:
    """SQLite-backed storage for agent eval runs and task results."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or _DEFAULT_DB_PATH)
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS eval_runs (
                        run_id TEXT PRIMARY KEY,
                        started_at REAL NOT NULL,
                        finished_at REAL DEFAULT 0,
                        suite TEXT NOT NULL,
                        models TEXT NOT NULL,
                        repeats INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL DEFAULT 'pending',
                        governor_present INTEGER NOT NULL DEFAULT 0,
                        host_fingerprint TEXT DEFAULT '',
                        error TEXT DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS eval_task_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        model TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        repeat INTEGER NOT NULL DEFAULT 0,
                        passed INTEGER NOT NULL DEFAULT 0,
                        rounds INTEGER NOT NULL DEFAULT 0,
                        tool_calls INTEGER NOT NULL DEFAULT 0,
                        wall_s REAL NOT NULL DEFAULT 0,
                        failure_class TEXT NOT NULL DEFAULT 'none',
                        admitted TEXT NOT NULL DEFAULT 'absent',
                        admitted_ctx INTEGER,
                        spill_ref TEXT,
                        diagnostics_seen INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_eval_runs_started
                        ON eval_runs(started_at);
                    CREATE INDEX IF NOT EXISTS idx_eval_tasks_run
                        ON eval_task_results(run_id);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # -- run lifecycle -----------------------------------------------------

    def create_run(
        self,
        run_id: str,
        suite: str,
        models: list[str],
        repeats: int,
        governor_present: bool,
        host_fingerprint: str = "",
    ) -> None:
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.execute(
                    "INSERT INTO eval_runs (run_id, started_at, suite, models,"
                    " repeats, status, governor_present, host_fingerprint)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        time.time(),
                        suite,
                        json.dumps(list(models)),
                        int(repeats),
                        "running",
                        int(bool(governor_present)),
                        host_fingerprint,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def finish_run(self, run_id: str, status: str, error: str = "") -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"unknown run status: {status!r}")
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.execute(
                    "UPDATE eval_runs SET status = ?, finished_at = ?,"
                    " error = ? WHERE run_id = ?",
                    (status, time.time(), error, run_id),
                )
                conn.commit()
            finally:
                conn.close()

    # -- task rows -----------------------------------------------------------

    def record_task(
        self,
        run_id: str,
        model: str,
        task_id: str,
        repeat: int,
        passed: bool,
        rounds: int,
        tool_calls: int,
        wall_s: float,
        failure_class: str,
        admitted: str,
        admitted_ctx: int | None = None,
        spill_ref: str | None = None,
        diagnostics_seen: bool = False,
    ) -> None:
        if failure_class not in FAILURE_CLASSES:
            raise ValueError(f"unknown failure_class: {failure_class!r}")
        if admitted not in ("yes", "refused", "absent"):
            raise ValueError(f"unknown admitted value: {admitted!r}")
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                conn.execute(
                    "INSERT INTO eval_task_results (run_id, model, task_id,"
                    " repeat, passed, rounds, tool_calls, wall_s,"
                    " failure_class, admitted, admitted_ctx, spill_ref,"
                    " diagnostics_seen)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        model,
                        task_id,
                        int(repeat),
                        int(bool(passed)),
                        int(rounds),
                        int(tool_calls),
                        float(wall_s),
                        failure_class,
                        admitted,
                        admitted_ctx,
                        spill_ref,
                        int(bool(diagnostics_seen)),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    # -- reads ---------------------------------------------------------------

    @staticmethod
    def _run_row_to_dict(row: tuple) -> dict[str, Any]:
        models_raw = row[4]
        try:
            models = json.loads(models_raw)
        except (TypeError, ValueError):
            models = []
        return {
            "run_id": row[0],
            "started_at": row[1],
            "finished_at": row[2],
            "suite": row[3],
            "models": models,
            "repeats": row[5],
            "status": row[6],
            "governor_present": bool(row[7]),
            "host_fingerprint": row[8],
            "error": row[9],
        }

    _RUN_COLUMNS = (
        "run_id, started_at, finished_at, suite, models, repeats, status,"
        " governor_present, host_fingerprint, error"
    )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                cur = conn.execute(
                    "SELECT " + self._RUN_COLUMNS + " FROM eval_runs"
                    " WHERE run_id = ?",
                    (run_id,),
                )
                row = cur.fetchone()
            finally:
                conn.close()
        return self._run_row_to_dict(row) if row else None

    def get_task_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                cur = conn.execute(
                    "SELECT model, task_id, repeat, passed, rounds,"
                    " tool_calls, wall_s, failure_class, admitted,"
                    " admitted_ctx, spill_ref, diagnostics_seen"
                    " FROM eval_task_results WHERE run_id = ? ORDER BY id",
                    (run_id,),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        return [
            {
                "model": r[0],
                "task_id": r[1],
                "repeat": r[2],
                "passed": bool(r[3]),
                "rounds": r[4],
                "tool_calls": r[5],
                "wall_s": r[6],
                "failure_class": r[7],
                "admitted": r[8],
                "admitted_ctx": r[9],
                "spill_ref": r[10],
                "diagnostics_seen": bool(r[11]),
            }
            for r in rows
        ]

    @staticmethod
    def summarize(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Per-model register summary: pass rate, failure classes, means."""
        per_model: dict[str, dict[str, Any]] = {}
        for row in task_rows:
            bucket = per_model.setdefault(
                row["model"],
                {
                    "total": 0,
                    "passed": 0,
                    "failures": {},
                    "rounds_sum": 0,
                    "wall_sum": 0.0,
                },
            )
            bucket["total"] += 1
            if row["passed"]:
                bucket["passed"] += 1
            else:
                cls = row["failure_class"]
                bucket["failures"][cls] = bucket["failures"].get(cls, 0) + 1
            bucket["rounds_sum"] += int(row["rounds"])
            bucket["wall_sum"] += float(row["wall_s"])
        summary: dict[str, Any] = {}
        for model, bucket in per_model.items():
            total = bucket["total"] or 1
            summary[model] = {
                "total": bucket["total"],
                "passed": bucket["passed"],
                "failures": bucket["failures"],
                "rounds_avg": round(bucket["rounds_sum"] / total, 2),
                "wall_avg_s": round(bucket["wall_sum"] / total, 2),
            }
        return summary

    def get_run_details(self, run_id: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        tasks = self.get_task_rows(run_id)
        return {"run": run, "tasks": tasks, "summary": self.summarize(tasks)}

    def get_history(
        self, limit: int = 50, suite: str | None = None
    ) -> list[dict[str, Any]]:
        limit = max(1, int(limit))
        with self._lock:
            conn = _safe_connect(self._db_path)
            try:
                if suite:
                    cur = conn.execute(
                        "SELECT " + self._RUN_COLUMNS + " FROM eval_runs"
                        " WHERE suite = ? ORDER BY started_at DESC LIMIT ?",
                        (suite, limit),
                    )
                else:
                    cur = conn.execute(
                        "SELECT " + self._RUN_COLUMNS + " FROM eval_runs"
                        " ORDER BY started_at DESC LIMIT ?",
                        (limit,),
                    )
                rows = cur.fetchall()
            finally:
                conn.close()
        return [self._run_row_to_dict(r) for r in rows]
