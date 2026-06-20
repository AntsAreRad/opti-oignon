#!/usr/bin/env python3
"""
CODING HISTORY - OPTI-OIGNON v1.8.2 (S76/S78/S79/S80)
===================================================

SQLite-backed task history and checkpoint/resume for the coding agent.
Persists tasks, steps, test results, and human checkpoint decisions.
Supports resuming interrupted tasks from last checkpoint.

S78: Coding History Analytics (SQ-08) -- aggregated success rates,
step counts, model comparison, failure reasons, time trends, test
pass rates. All aggregation done via SQL.

S79: Export (JSON/CSV) and batch delete operations.

S80: Working memory persistence (working_memory table) for cross-step
context retention in the coding agent.

Separate database: coding_history.db (follows per-feature-domain pattern).

Author: Leon
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
# S136 audit fix: use encrypted DB connections
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "config", "coding_history.yaml"
)

_DEFAULT_CONFIG = {
    "enabled": True,
    "max_tasks": 200,
    "retention_days": 30,
    "max_output_length": 10000,
    "max_plan_size": 50000,
}

try:
    import yaml as _yaml
except ImportError:
    _yaml = None


def _load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML with safe defaults."""
    path = config_path or _CONFIG_PATH
    try:
        if _yaml is not None and os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                raw = _yaml.safe_load(fh) or {}
            return {**_DEFAULT_CONFIG, **raw}
    except Exception as exc:
        logger.warning("Failed to load coding history config: %s", exc)
    return dict(_DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Data classes for query results
# ---------------------------------------------------------------------------

@dataclass
class TaskSummary:
    """Summary of a coding agent task."""

    task_id: str
    task_text: str
    project_path: str
    model: str
    status: str
    step_count: int
    completed_steps: int
    test_runs: int
    last_passed: bool | None
    created_at: float
    completed_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "project_path": self.project_path,
            "model": self.model,
            "status": self.status,
            "step_count": self.step_count,
            "completed_steps": self.completed_steps,
            "test_runs": self.test_runs,
            "last_passed": self.last_passed,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class TaskDetail:
    """Full detail of a coding agent task with steps, tests, checkpoints."""

    task_id: str
    task_text: str
    project_path: str
    model: str
    status: str
    plan_json: dict[str, Any] | None
    created_at: float
    completed_at: float | None
    steps: list[dict[str, Any]] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "project_path": self.project_path,
            "model": self.model,
            "status": self.status,
            "plan_json": self.plan_json,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "steps": self.steps,
            "tests": self.tests,
            "checkpoints": self.checkpoints,
        }


@dataclass
class CheckpointState:
    """Snapshot of agent state at a checkpoint for resume."""

    task_id: str
    task_text: str
    project_path: str
    model: str
    plan_json: dict[str, Any] | None
    current_step: int
    phase: str
    originals_hash: str


# ---------------------------------------------------------------------------
# CodingHistoryStore
# ---------------------------------------------------------------------------

class CodingHistoryStore:
    """SQLite-backed persistence for coding agent tasks.

    Thread-safe. Uses WAL mode for concurrent reads. All aggregation
    is done via SQL rather than in-memory computation.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        config_path: str | None = None,
    ):
        self._config = _load_config(config_path)
        self._enabled = self._config["enabled"]
        self._max_tasks = self._config["max_tasks"]
        self._retention_days = self._config["retention_days"]
        self._max_output = self._config["max_output_length"]
        self._max_plan = self._config["max_plan_size"]

        if db_path is None:
            db_path = (
                Path(__file__).resolve().parent / "data" / "coding_history.db"
            )
        self._db_path = str(db_path)

        self._lock = threading.Lock()
        self._init_db()

    # -- Database setup ----------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        conn = _safe_connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_text TEXT NOT NULL,
                    project_path TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'started',
                    plan_json TEXT,
                    created_at REAL NOT NULL,
                    completed_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status);

                CREATE INDEX IF NOT EXISTS idx_tasks_created
                ON tasks(created_at);

                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    step_type TEXT NOT NULL DEFAULT '',
                    file_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    result TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                );

                CREATE INDEX IF NOT EXISTS idx_steps_task
                ON steps(task_id, step_number);

                CREATE TABLE IF NOT EXISTS tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    run_number INTEGER NOT NULL,
                    passed INTEGER NOT NULL DEFAULT 0,
                    output TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                );

                CREATE INDEX IF NOT EXISTS idx_tests_task
                ON tests(task_id, run_number);

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    decision TEXT NOT NULL DEFAULT '',
                    current_step INTEGER NOT NULL DEFAULT 0,
                    originals_hash TEXT NOT NULL DEFAULT '',
                    plan_snapshot TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoints_task
                ON checkpoints(task_id, timestamp);

                CREATE TABLE IF NOT EXISTS working_memory (
                    task_id TEXT PRIMARY KEY,
                    memory_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                );
            """)
            conn.commit()
        finally:
            conn.close()

    # -- Task lifecycle ----------------------------------------------------

    def record_task_start(
        self,
        task_id: str,
        task_text: str,
        project_path: str = "",
        model: str = "",
    ) -> None:
        """Record a new task start."""
        if not self._enabled:
            return
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tasks
                        (task_id, task_text, project_path, model, status, created_at)
                    VALUES (?, ?, ?, ?, 'started', ?)
                    """,
                    (task_id, task_text[:2000], project_path, model, time.time()),
                )
                conn.commit()
            finally:
                conn.close()

    def update_task_status(
        self, task_id: str, status: str, plan_json: dict | None = None
    ) -> None:
        """Update task status and optionally the plan."""
        if not self._enabled:
            return
        with self._lock:
            conn = self._get_conn()
            try:
                if plan_json is not None:
                    plan_str = json.dumps(plan_json)[:self._max_plan]
                    conn.execute(
                        """
                        UPDATE tasks
                        SET status = ?, plan_json = ?,
                            completed_at = CASE WHEN ? IN
                                ('completed','aborted','failed')
                            THEN ? ELSE completed_at END
                        WHERE task_id = ?
                        """,
                        (status, plan_str, status, time.time(), task_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE tasks
                        SET status = ?,
                            completed_at = CASE WHEN ? IN
                                ('completed','aborted','failed')
                            THEN ? ELSE completed_at END
                        WHERE task_id = ?
                        """,
                        (status, status, time.time(), task_id),
                    )
                conn.commit()
            finally:
                conn.close()

    # -- Step recording ----------------------------------------------------

    def record_step(
        self,
        task_id: str,
        step_number: int,
        step_type: str = "",
        file_path: str = "",
        status: str = "completed",
        result: str = "",
    ) -> None:
        """Record a completed step."""
        if not self._enabled:
            return
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO steps
                        (task_id, step_number, step_type, file_path,
                         status, result, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        step_number,
                        step_type,
                        file_path,
                        status,
                        result[:self._max_output],
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    # -- Test recording ----------------------------------------------------

    def record_test(
        self,
        task_id: str,
        run_number: int,
        passed: bool,
        output: str = "",
    ) -> None:
        """Record a test run result."""
        if not self._enabled:
            return
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO tests
                        (task_id, run_number, passed, output, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        run_number,
                        1 if passed else 0,
                        output[:self._max_output],
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    # -- Checkpoint recording and resume -----------------------------------

    def record_checkpoint(
        self,
        task_id: str,
        phase: str,
        decision: str = "",
        current_step: int = 0,
        originals_hash: str = "",
        plan_snapshot: dict | None = None,
    ) -> None:
        """Record a human checkpoint decision and agent state snapshot."""
        if not self._enabled:
            return
        with self._lock:
            conn = self._get_conn()
            try:
                plan_str = (
                    json.dumps(plan_snapshot)[:self._max_plan]
                    if plan_snapshot
                    else None
                )
                conn.execute(
                    """
                    INSERT INTO checkpoints
                        (task_id, phase, decision, current_step,
                         originals_hash, plan_snapshot, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        phase,
                        decision,
                        current_step,
                        originals_hash,
                        plan_str,
                        time.time(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_last_checkpoint(self, task_id: str) -> CheckpointState | None:
        """Get the last checkpoint state for resume.

        Returns None if task not found or no checkpoints recorded.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT
                    t.task_id, t.task_text, t.project_path, t.model,
                    c.phase, c.current_step, c.originals_hash,
                    c.plan_snapshot
                FROM checkpoints c
                JOIN tasks t ON t.task_id = c.task_id
                WHERE c.task_id = ?
                ORDER BY c.timestamp DESC
                LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            plan = None
            if row["plan_snapshot"]:
                try:
                    plan = json.loads(row["plan_snapshot"])
                except (json.JSONDecodeError, TypeError):
                    pass
            return CheckpointState(
                task_id=row["task_id"],
                task_text=row["task_text"],
                project_path=row["project_path"],
                model=row["model"],
                plan_json=plan,
                current_step=row["current_step"],
                phase=row["phase"],
                originals_hash=row["originals_hash"],
            )
        finally:
            conn.close()

    # -- Working memory (S80) ----------------------------------------------

    def save_working_memory(
        self, task_id: str, memory_data: dict[str, Any]
    ) -> None:
        """Save or update working memory for a task.

        Uses INSERT OR REPLACE to upsert the memory state.

        Args:
            task_id: The task identifier.
            memory_data: Working memory dict from WorkingMemory.to_dict().
        """
        if not self._enabled:
            return
        with self._lock:
            conn = self._get_conn()
            try:
                memory_str = json.dumps(memory_data)[:self._max_plan]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO working_memory
                        (task_id, memory_json, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (task_id, memory_str, time.time()),
                )
                conn.commit()
            finally:
                conn.close()

    def load_working_memory(self, task_id: str) -> dict[str, Any] | None:
        """Load working memory for a task.

        Args:
            task_id: The task identifier.

        Returns:
            Working memory dict, or None if not found.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT memory_json FROM working_memory WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                return json.loads(row["memory_json"])
            except (json.JSONDecodeError, TypeError):
                return None
        finally:
            conn.close()

    def delete_working_memory(self, task_id: str) -> bool:
        """Delete working memory for a task.

        Args:
            task_id: The task identifier.

        Returns:
            True if a row was deleted.
        """
        if not self._enabled:
            return False
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM working_memory WHERE task_id = ?",
                    (task_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def get_resumable_tasks(self) -> list[TaskSummary]:
        """List tasks that can be resumed (started or in-progress)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT
                    t.task_id, t.task_text, t.project_path, t.model,
                    t.status, t.created_at, t.completed_at,
                    COALESCE(
                        (SELECT COUNT(*) FROM steps s
                         WHERE s.task_id = t.task_id), 0
                    ) AS step_count,
                    COALESCE(
                        (SELECT COUNT(*) FROM steps s
                         WHERE s.task_id = t.task_id
                         AND s.status = 'completed'), 0
                    ) AS completed_steps,
                    COALESCE(
                        (SELECT COUNT(*) FROM tests te
                         WHERE te.task_id = t.task_id), 0
                    ) AS test_runs,
                    (SELECT te2.passed FROM tests te2
                     WHERE te2.task_id = t.task_id
                     ORDER BY te2.run_number DESC LIMIT 1
                    ) AS last_passed
                FROM tasks t
                WHERE t.status NOT IN ('completed', 'aborted', 'failed')
                ORDER BY t.created_at DESC
                """
            ).fetchall()
            return [self._row_to_summary(r) for r in rows]
        finally:
            conn.close()

    # -- Query methods -----------------------------------------------------

    def list_tasks(
        self, limit: int = 50, offset: int = 0, status: str | None = None
    ) -> list[TaskSummary]:
        """List tasks with optional status filter.

        Aggregation via SQL subqueries -- no in-memory computation.
        """
        conn = self._get_conn()
        try:
            if status:
                rows = conn.execute(
                    """
                    SELECT
                        t.task_id, t.task_text, t.project_path, t.model,
                        t.status, t.created_at, t.completed_at,
                        COALESCE(
                            (SELECT COUNT(*) FROM steps s
                             WHERE s.task_id = t.task_id), 0
                        ) AS step_count,
                        COALESCE(
                            (SELECT COUNT(*) FROM steps s
                             WHERE s.task_id = t.task_id
                             AND s.status = 'completed'), 0
                        ) AS completed_steps,
                        COALESCE(
                            (SELECT COUNT(*) FROM tests te
                             WHERE te.task_id = t.task_id), 0
                        ) AS test_runs,
                        (SELECT te2.passed FROM tests te2
                         WHERE te2.task_id = t.task_id
                         ORDER BY te2.run_number DESC LIMIT 1
                        ) AS last_passed
                    FROM tasks t
                    WHERE t.status = ?
                    ORDER BY t.created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        t.task_id, t.task_text, t.project_path, t.model,
                        t.status, t.created_at, t.completed_at,
                        COALESCE(
                            (SELECT COUNT(*) FROM steps s
                             WHERE s.task_id = t.task_id), 0
                        ) AS step_count,
                        COALESCE(
                            (SELECT COUNT(*) FROM steps s
                             WHERE s.task_id = t.task_id
                             AND s.status = 'completed'), 0
                        ) AS completed_steps,
                        COALESCE(
                            (SELECT COUNT(*) FROM tests te
                             WHERE te.task_id = t.task_id), 0
                        ) AS test_runs,
                        (SELECT te2.passed FROM tests te2
                         WHERE te2.task_id = t.task_id
                         ORDER BY te2.run_number DESC LIMIT 1
                        ) AS last_passed
                    FROM tasks t
                    ORDER BY t.created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            return [self._row_to_summary(r) for r in rows]
        finally:
            conn.close()

    def get_task_detail(self, task_id: str) -> TaskDetail | None:
        """Get full task detail including steps, tests, and checkpoints."""
        conn = self._get_conn()
        try:
            task_row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task_row is None:
                return None

            steps = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT step_number, step_type, file_path, status,
                           result, timestamp
                    FROM steps WHERE task_id = ?
                    ORDER BY step_number
                    """,
                    (task_id,),
                ).fetchall()
            ]

            tests = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT run_number, passed, output, timestamp
                    FROM tests WHERE task_id = ?
                    ORDER BY run_number
                    """,
                    (task_id,),
                ).fetchall()
            ]
            # Convert passed from int to bool in test results
            for t in tests:
                t["passed"] = bool(t["passed"])

            checkpoints = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT phase, decision, current_step,
                           originals_hash, timestamp
                    FROM checkpoints WHERE task_id = ?
                    ORDER BY timestamp
                    """,
                    (task_id,),
                ).fetchall()
            ]

            plan = None
            if task_row["plan_json"]:
                try:
                    plan = json.loads(task_row["plan_json"])
                except (json.JSONDecodeError, TypeError):
                    pass

            return TaskDetail(
                task_id=task_row["task_id"],
                task_text=task_row["task_text"],
                project_path=task_row["project_path"],
                model=task_row["model"],
                status=task_row["status"],
                plan_json=plan,
                created_at=task_row["created_at"],
                completed_at=task_row["completed_at"],
                steps=steps,
                tests=tests,
                checkpoints=checkpoints,
            )
        finally:
            conn.close()

    def count_tasks(self, status: str | None = None) -> int:
        """Count total tasks, optionally filtered by status."""
        conn = self._get_conn()
        try:
            if status:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM tasks WHERE status = ?",
                    (status,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM tasks"
                ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def delete_task(self, task_id: str) -> bool:
        """Delete a task and all associated records."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "DELETE FROM working_memory WHERE task_id = ?", (task_id,)
                )
                conn.execute(
                    "DELETE FROM checkpoints WHERE task_id = ?", (task_id,)
                )
                conn.execute(
                    "DELETE FROM tests WHERE task_id = ?", (task_id,)
                )
                conn.execute(
                    "DELETE FROM steps WHERE task_id = ?", (task_id,)
                )
                cursor = conn.execute(
                    "DELETE FROM tasks WHERE task_id = ?", (task_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    # -- Maintenance -------------------------------------------------------

    def prune(self) -> int:
        """Remove tasks older than retention_days and enforce max_tasks.

        Returns number of tasks pruned.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                pruned = 0

                # Age-based pruning
                cutoff = time.time() - (self._retention_days * 86400)
                old_ids = [
                    r["task_id"]
                    for r in conn.execute(
                        "SELECT task_id FROM tasks WHERE created_at < ?",
                        (cutoff,),
                    ).fetchall()
                ]
                for tid in old_ids:
                    conn.execute(
                        "DELETE FROM working_memory WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM checkpoints WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM tests WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM steps WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM tasks WHERE task_id = ?", (tid,)
                    )
                pruned += len(old_ids)

                # Count-based pruning
                total = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM tasks"
                ).fetchone()["cnt"]
                if total > self._max_tasks:
                    excess_ids = [
                        r["task_id"]
                        for r in conn.execute(
                            """
                            SELECT task_id FROM tasks
                            ORDER BY created_at ASC
                            LIMIT ?
                            """,
                            (total - self._max_tasks,),
                        ).fetchall()
                    ]
                    for tid in excess_ids:
                        conn.execute(
                            "DELETE FROM working_memory WHERE task_id = ?",
                            (tid,),
                        )
                        conn.execute(
                            "DELETE FROM checkpoints WHERE task_id = ?",
                            (tid,),
                        )
                        conn.execute(
                            "DELETE FROM tests WHERE task_id = ?", (tid,)
                        )
                        conn.execute(
                            "DELETE FROM steps WHERE task_id = ?", (tid,)
                        )
                        conn.execute(
                            "DELETE FROM tasks WHERE task_id = ?", (tid,)
                        )
                    pruned += len(excess_ids)

                conn.commit()
                return pruned
            finally:
                conn.close()

    # -- Export (S79) ------------------------------------------------------

    def export_tasks_json(self) -> list[dict[str, Any]]:
        """Export all tasks with steps and tests as JSON-serializable dicts.

        Returns a list of task dicts, each including nested steps, tests,
        and computed fields (step_count, pass_rate, duration_seconds).
        """
        conn = self._get_conn()
        try:
            tasks = conn.execute(
                """
                SELECT task_id, task_text, project_path, model, status,
                       created_at, completed_at
                FROM tasks ORDER BY created_at DESC
                """
            ).fetchall()

            result = []
            for t in tasks:
                tid = t["task_id"]
                steps = [
                    dict(r) for r in conn.execute(
                        """
                        SELECT step_number, step_type, file_path, status,
                               result, timestamp
                        FROM steps WHERE task_id = ?
                        ORDER BY step_number
                        """,
                        (tid,),
                    ).fetchall()
                ]
                tests = [
                    dict(r) for r in conn.execute(
                        """
                        SELECT run_number, passed, output, timestamp
                        FROM tests WHERE task_id = ?
                        ORDER BY run_number
                        """,
                        (tid,),
                    ).fetchall()
                ]
                for te in tests:
                    te["passed"] = bool(te["passed"])

                created = t["created_at"]
                completed = t["completed_at"]
                duration = (
                    round(completed - created, 1)
                    if completed and created and completed > created
                    else None
                )
                total_runs = len(tests)
                passed_runs = sum(1 for te in tests if te["passed"])

                result.append({
                    "task_id": tid,
                    "task_text": t["task_text"],
                    "project_path": t["project_path"],
                    "model": t["model"],
                    "status": t["status"],
                    "step_count": len(steps),
                    "test_runs": total_runs,
                    "pass_rate": (
                        round(100.0 * passed_runs / total_runs, 1)
                        if total_runs > 0 else 0.0
                    ),
                    "created_at": created,
                    "completed_at": completed,
                    "duration_seconds": duration,
                    "steps": steps,
                    "tests": tests,
                })
            return result
        finally:
            conn.close()

    def export_tasks_csv_rows(self) -> list[dict[str, Any]]:
        """Export tasks as flat dicts suitable for CSV serialization.

        Each row contains: task_id, task_text, model, status, step_count,
        test_runs, pass_rate, created_at, completed_at, duration_seconds.
        No nested structures.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT
                    t.task_id,
                    t.task_text,
                    t.model,
                    t.status,
                    COALESCE(
                        (SELECT COUNT(*) FROM steps s
                         WHERE s.task_id = t.task_id), 0
                    ) AS step_count,
                    COALESCE(
                        (SELECT COUNT(*) FROM tests te
                         WHERE te.task_id = t.task_id), 0
                    ) AS test_runs,
                    COALESCE(
                        (SELECT ROUND(
                            100.0 * SUM(
                                CASE WHEN te2.passed = 1 THEN 1 ELSE 0 END
                            ) / NULLIF(COUNT(*), 0), 1
                        ) FROM tests te2
                         WHERE te2.task_id = t.task_id), 0.0
                    ) AS pass_rate,
                    t.created_at,
                    t.completed_at,
                    CASE
                        WHEN t.completed_at IS NOT NULL
                             AND t.created_at IS NOT NULL
                             AND t.completed_at > t.created_at
                        THEN ROUND(t.completed_at - t.created_at, 1)
                        ELSE NULL
                    END AS duration_seconds
                FROM tasks t
                ORDER BY t.created_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- Batch delete (S79) ------------------------------------------------

    def batch_delete_by_ids(self, task_ids: list[str]) -> int:
        """Delete multiple tasks by their IDs.

        Args:
            task_ids: List of task IDs to delete.

        Returns:
            Number of tasks actually deleted.
        """
        if not task_ids:
            return 0
        deleted = 0
        with self._lock:
            conn = self._get_conn()
            try:
                for tid in task_ids:
                    conn.execute(
                        "DELETE FROM working_memory WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM checkpoints WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM tests WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM steps WHERE task_id = ?", (tid,)
                    )
                    cursor = conn.execute(
                        "DELETE FROM tasks WHERE task_id = ?", (tid,)
                    )
                    deleted += cursor.rowcount
                conn.commit()
                return deleted
            finally:
                conn.close()

    def batch_delete_before_date(self, before_timestamp: float) -> int:
        """Delete all tasks created before the given timestamp.

        Args:
            before_timestamp: Unix timestamp cutoff. Tasks with
                created_at < before_timestamp are deleted.

        Returns:
            Number of tasks deleted.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                task_ids = [
                    r["task_id"] for r in conn.execute(
                        "SELECT task_id FROM tasks WHERE created_at < ?",
                        (before_timestamp,),
                    ).fetchall()
                ]
                if not task_ids:
                    return 0

                for tid in task_ids:
                    conn.execute(
                        "DELETE FROM working_memory WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM checkpoints WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM tests WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM steps WHERE task_id = ?", (tid,)
                    )
                    conn.execute(
                        "DELETE FROM tasks WHERE task_id = ?", (tid,)
                    )
                conn.commit()
                return len(task_ids)
            finally:
                conn.close()

    # -- Stats -------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics across all tasks."""
        conn = self._get_conn()
        try:
            total = conn.execute(
                "SELECT COUNT(*) AS cnt FROM tasks"
            ).fetchone()["cnt"]

            by_status = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM tasks GROUP BY status"
            ).fetchall():
                by_status[row["status"]] = row["cnt"]

            total_steps = conn.execute(
                "SELECT COUNT(*) AS cnt FROM steps"
            ).fetchone()["cnt"]

            total_tests = conn.execute(
                "SELECT COUNT(*) AS cnt FROM tests"
            ).fetchone()["cnt"]

            passed_tests = conn.execute(
                "SELECT COUNT(*) AS cnt FROM tests WHERE passed = 1"
            ).fetchone()["cnt"]

            total_checkpoints = conn.execute(
                "SELECT COUNT(*) AS cnt FROM checkpoints"
            ).fetchone()["cnt"]

            return {
                "total_tasks": total,
                "by_status": by_status,
                "total_steps": total_steps,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "total_checkpoints": total_checkpoints,
            }
        finally:
            conn.close()

    # -- Analytics (S78 SQ-08) --------------------------------------------

    def get_success_rate_by_model(self) -> list[dict[str, Any]]:
        """Get task success rate grouped by model.

        Returns list of dicts with model, total, completed, success_rate.
        All aggregation done via SQL.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT
                    model,
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)
                        AS completed,
                    ROUND(
                        100.0 * SUM(
                            CASE WHEN status = 'completed' THEN 1 ELSE 0 END
                        ) / COUNT(*), 1
                    ) AS success_rate
                FROM tasks
                WHERE model != ''
                GROUP BY model
                ORDER BY success_rate DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_avg_steps_by_model(self) -> list[dict[str, Any]]:
        """Get average step count per task grouped by model.

        Returns list of dicts with model, avg_steps, min_steps,
        max_steps, task_count.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT
                    t.model,
                    ROUND(AVG(step_counts.cnt), 1) AS avg_steps,
                    MIN(step_counts.cnt) AS min_steps,
                    MAX(step_counts.cnt) AS max_steps,
                    COUNT(*) AS task_count
                FROM tasks t
                JOIN (
                    SELECT task_id, COUNT(*) AS cnt
                    FROM steps
                    GROUP BY task_id
                ) step_counts ON step_counts.task_id = t.task_id
                WHERE t.model != ''
                GROUP BY t.model
                ORDER BY avg_steps ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_avg_steps_overall(self) -> dict[str, Any]:
        """Get overall average step count across all tasks.

        Returns dict with avg_steps, min_steps, max_steps, task_count.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT
                    ROUND(AVG(step_counts.cnt), 1) AS avg_steps,
                    MIN(step_counts.cnt) AS min_steps,
                    MAX(step_counts.cnt) AS max_steps,
                    COUNT(*) AS task_count
                FROM (
                    SELECT task_id, COUNT(*) AS cnt
                    FROM steps
                    GROUP BY task_id
                ) step_counts
                """
            ).fetchone()
            if row is None:
                return {
                    "avg_steps": 0.0,
                    "min_steps": 0,
                    "max_steps": 0,
                    "task_count": 0,
                }
            return dict(row)
        finally:
            conn.close()

    def get_failure_reasons(self) -> list[dict[str, Any]]:
        """Get distribution of failure reasons.

        Uses the last checkpoint phase before failure as the failure
        point indicator. Returns list of dicts with phase, count.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT
                    COALESCE(last_cp.phase, 'unknown') AS failure_phase,
                    COUNT(*) AS count
                FROM tasks t
                LEFT JOIN (
                    SELECT task_id, phase
                    FROM checkpoints c1
                    WHERE c1.timestamp = (
                        SELECT MAX(c2.timestamp)
                        FROM checkpoints c2
                        WHERE c2.task_id = c1.task_id
                    )
                ) last_cp ON last_cp.task_id = t.task_id
                WHERE t.status = 'failed'
                GROUP BY failure_phase
                ORDER BY count DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_time_trends(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get time-to-completion trends for finished tasks.

        Returns list of dicts with task_id, model, created_at,
        completed_at, duration_seconds, ordered by created_at desc.
        Only includes tasks with both timestamps.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT
                    task_id,
                    model,
                    created_at,
                    completed_at,
                    ROUND(completed_at - created_at, 1)
                        AS duration_seconds
                FROM tasks
                WHERE completed_at IS NOT NULL
                    AND created_at IS NOT NULL
                    AND completed_at > created_at
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_test_pass_rate_per_task(
        self, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get test pass rate per task.

        Returns list of dicts with task_id, model, total_runs,
        passed_runs, pass_rate. Only includes tasks with test runs.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT
                    t.task_id,
                    t.model,
                    COUNT(*) AS total_runs,
                    SUM(CASE WHEN te.passed = 1 THEN 1 ELSE 0 END)
                        AS passed_runs,
                    ROUND(
                        100.0 * SUM(
                            CASE WHEN te.passed = 1 THEN 1 ELSE 0 END
                        ) / COUNT(*), 1
                    ) AS pass_rate
                FROM tasks t
                JOIN tests te ON te.task_id = t.task_id
                GROUP BY t.task_id
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_steps_distribution(self) -> list[dict[str, Any]]:
        """Get step count distribution across all tasks.

        Returns list of dicts with step_count and task_count,
        representing how many tasks had N steps.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT
                    step_counts.cnt AS step_count,
                    COUNT(*) AS task_count
                FROM (
                    SELECT task_id, COUNT(*) AS cnt
                    FROM steps
                    GROUP BY task_id
                ) step_counts
                GROUP BY step_counts.cnt
                ORDER BY step_counts.cnt ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_analytics(self) -> dict[str, Any]:
        """Get full analytics payload combining all analytics queries.

        Returns a single dict with all analytics data for the API
        endpoint. All aggregation is done via SQL.
        """
        conn = self._get_conn()
        try:
            total_tasks = conn.execute(
                "SELECT COUNT(*) AS cnt FROM tasks"
            ).fetchone()["cnt"]

            completed = conn.execute(
                "SELECT COUNT(*) AS cnt FROM tasks WHERE status = 'completed'"
            ).fetchone()["cnt"]

            overall_success_rate = (
                round(100.0 * completed / total_tasks, 1)
                if total_tasks > 0
                else 0.0
            )
        finally:
            conn.close()

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "overall_success_rate": overall_success_rate,
            "success_rate_by_model": self.get_success_rate_by_model(),
            "avg_steps_by_model": self.get_avg_steps_by_model(),
            "avg_steps_overall": self.get_avg_steps_overall(),
            "failure_reasons": self.get_failure_reasons(),
            "time_trends": self.get_time_trends(),
            "test_pass_rate_per_task": self.get_test_pass_rate_per_task(),
            "steps_distribution": self.get_steps_distribution(),
        }

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> TaskSummary:
        """Convert a query row to TaskSummary."""
        last_passed = row["last_passed"]
        if last_passed is not None:
            last_passed = bool(last_passed)
        return TaskSummary(
            task_id=row["task_id"],
            task_text=row["task_text"],
            project_path=row["project_path"],
            model=row["model"],
            status=row["status"],
            step_count=row["step_count"],
            completed_steps=row["completed_steps"],
            test_runs=row["test_runs"],
            last_passed=last_passed,
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

CODING_HISTORY_AVAILABLE = True

try:
    coding_history_store = CodingHistoryStore()
except Exception as _exc:
    logger.warning("CodingHistoryStore unavailable: %s", _exc)
    coding_history_store = None
    CODING_HISTORY_AVAILABLE = False
