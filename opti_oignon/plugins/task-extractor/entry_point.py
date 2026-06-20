"""
Task-extractor plugin for Opti-Oignon.

Automatically extracts action items and TODOs from LLM responses.
Maintains a persistent task list in SQLite. Tasks can be managed
via slash commands: /tasks, /tasks done <id>, /tasks clear.

Extraction uses configurable regex patterns to detect imperative
sentences, numbered steps, and common action phrases.
"""

import logging
import re
import sqlite3

# S136 audit fix
try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    _safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)
import time
from pathlib import Path
from typing import Any, Optional

__plugin_name__: str = "task-extractor"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_AUTO_EXTRACT = True
_MAX_TASKS = 200

DEFAULT_PATTERNS = [
    "you should",
    "next step",
    "todo:",
    "action item",
    "make sure to",
    "don't forget to",
    "remember to",
    "you need to",
    "you must",
    "you will need to",
    "important:",
    "recommendation:",
]

# =========================================================================
# Task extraction engine
# =========================================================================

# Numbered step pattern: "1. Do something", "1) Do something"
_NUMBERED_STEP_RE = re.compile(
    r"(?:^|\n)\s*\d+[.)]\s+(.+?)(?=\n\s*\d+[.)]|\n\n|\Z)",
    re.DOTALL,
)

# Imperative at line start (common verbs)
_IMPERATIVE_VERBS = (
    "install", "configure", "setup", "set up", "create", "add",
    "remove", "update", "check", "verify", "ensure", "run",
    "test", "deploy", "build", "fix", "implement", "write",
    "review", "backup", "migrate", "download", "upload",
)

_IMPERATIVE_RE = re.compile(
    r"(?:^|\n)\s*(?:" + "|".join(_IMPERATIVE_VERBS) + r")\s+.+?(?:\.|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Code block boundaries (to skip)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)


def _strip_code_blocks(text: str) -> str:
    """Remove code blocks from text before extraction."""
    return _CODE_BLOCK_RE.sub("", text)


def _build_pattern_regex(patterns: list[str]) -> re.Pattern[str]:
    """Build a combined regex from trigger patterns.

    Each pattern matches from the trigger phrase to the end of the
    sentence (period, newline, or end of text).
    """
    escaped = [re.escape(p) for p in patterns]
    combined = "|".join(escaped)
    return re.compile(
        rf"(?:^|(?<=\s))(?:{combined})\s*(.+?)(?:\.|;|\n|$)",
        re.IGNORECASE | re.MULTILINE,
    )


def extract_tasks(
    text: str,
    *,
    patterns: Optional[list[str]] = None,
    max_tasks: int = 10,
) -> list[dict[str, str]]:
    """Extract task items from response text.

    Parameters
    ----------
    text : str
        The LLM response text.
    patterns : list[str], optional
        Trigger patterns. Defaults to DEFAULT_PATTERNS.
    max_tasks : int
        Maximum number of tasks to extract.

    Returns
    -------
    list[dict]
        Extracted tasks with 'text' and 'source' keys.
    """
    if not text:
        return []

    clean_text = _strip_code_blocks(text)
    tasks: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add_task(task_text: str, source: str) -> None:
        normalized = task_text.strip().lower()
        # Skip very short or duplicate tasks
        if len(normalized) < 10 or normalized in seen:
            return
        if len(tasks) >= max_tasks:
            return
        seen.add(normalized)
        tasks.append({
            "text": task_text.strip(),
            "source": source,
        })

    # 1. Pattern-based extraction
    pat_list = patterns if patterns is not None else DEFAULT_PATTERNS
    pattern_re = _build_pattern_regex(pat_list)
    for m in pattern_re.finditer(clean_text):
        full_match = m.group(0).strip()
        _add_task(full_match, "pattern")

    # 2. Numbered steps
    for m in _NUMBERED_STEP_RE.finditer(clean_text):
        step_text = m.group(1).strip()
        _add_task(step_text, "numbered_step")

    # 3. Imperative sentences
    for m in _IMPERATIVE_RE.finditer(clean_text):
        imp_text = m.group(0).strip()
        _add_task(imp_text, "imperative")

    return tasks[:max_tasks]


# =========================================================================
# SQLite task store
# =========================================================================

class TaskDB:
    """SQLite-backed task storage.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    max_tasks : int
        Maximum number of stored tasks.
    """

    def __init__(self, db_path: str | Path, max_tasks: int = _MAX_TASKS) -> None:
        self.db_path = str(db_path)
        self.max_tasks = max_tasks
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = _safe_connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                done INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                completed_at REAL
            )
        """)
        conn.commit()

    def add_task(self, text: str, source: str = "manual") -> dict[str, Any]:
        """Add a new task.

        Returns
        -------
        dict
            The created task record, or error dict if limit reached.
        """
        conn = self._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE done = 0"
        ).fetchone()[0]
        if count >= self.max_tasks:
            return {
                "error": f"Task limit reached ({self.max_tasks}). "
                         "Complete or clear some tasks first.",
            }

        now = time.time()
        cursor = conn.execute(
            "INSERT INTO tasks (text, source, done, created_at) "
            "VALUES (?, ?, 0, ?)",
            (text, source, now),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "text": text,
            "source": source,
            "done": False,
            "created_at": now,
        }

    def list_tasks(
        self,
        *,
        include_done: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List tasks, pending first.

        Parameters
        ----------
        include_done : bool
            Whether to include completed tasks.
        limit : int
            Maximum number of tasks to return.

        Returns
        -------
        list[dict]
            List of task records.
        """
        conn = self._get_conn()
        if include_done:
            query = (
                "SELECT id, text, source, done, created_at, completed_at "
                "FROM tasks ORDER BY done ASC, created_at DESC LIMIT ?"
            )
            rows = conn.execute(query, (limit,)).fetchall()
        else:
            query = (
                "SELECT id, text, source, done, created_at, completed_at "
                "FROM tasks WHERE done = 0 "
                "ORDER BY created_at DESC LIMIT ?"
            )
            rows = conn.execute(query, (limit,)).fetchall()

        return [
            {
                "id": row["id"],
                "text": row["text"],
                "source": row["source"],
                "done": bool(row["done"]),
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
            }
            for row in rows
        ]

    def mark_done(self, task_id: int) -> bool:
        """Mark a task as completed.

        Returns True if a task was updated, False if not found.
        """
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute(
            "UPDATE tasks SET done = 1, completed_at = ? WHERE id = ? AND done = 0",
            (now, task_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def clear_done(self) -> int:
        """Remove all completed tasks.

        Returns the number of tasks removed.
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM tasks WHERE done = 1")
        conn.commit()
        return cursor.rowcount

    def clear_all(self) -> int:
        """Remove all tasks.

        Returns the number of tasks removed.
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM tasks")
        conn.commit()
        return cursor.rowcount

    def get_pending_count(self) -> int:
        """Return number of pending (not done) tasks."""
        conn = self._get_conn()
        return conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE done = 0"
        ).fetchone()[0]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =========================================================================
# Module-level DB instance (lazy init)
# =========================================================================

_db: Optional[TaskDB] = None


def _get_db(ctx: Any) -> TaskDB:
    """Get or create the task database."""
    global _db
    if _db is not None:
        return _db

    plugin_dir = ctx.metadata.get("plugin_dir", "")
    if plugin_dir:
        db_path = Path(plugin_dir) / "tasks.db"
    else:
        import tempfile
        db_path = Path(tempfile.gettempdir()) / "opti_tasks.db"

    max_tasks = ctx.config.get("max_tasks", _MAX_TASKS)
    _db = TaskDB(db_path, max_tasks=max_tasks)
    logger.debug("TaskDB initialized at %s", db_path)
    return _db


# =========================================================================
# Command parsing
# =========================================================================

_CMD_TASKS_LIST = re.compile(r"^/tasks\s*$")
_CMD_TASKS_ALL = re.compile(r"^/tasks\s+all\s*$")
_CMD_TASKS_DONE = re.compile(r"^/tasks\s+done\s+(\d+)\s*$")
_CMD_TASKS_CLEAR = re.compile(r"^/tasks\s+clear\s*$")
_CMD_TASKS_CLEAR_DONE = re.compile(r"^/tasks\s+clear\s+done\s*$")


def _format_task_list(tasks: list[dict[str, Any]], title: str = "Tasks") -> str:
    """Format a list of tasks for display."""
    if not tasks:
        return f"**{title}:** No tasks found."

    pending = [t for t in tasks if not t["done"]]
    done = [t for t in tasks if t["done"]]

    lines = [f"**{title}** ({len(pending)} pending):", ""]

    for task in pending:
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(task["created_at"])
        )
        lines.append(f"- [ ] **#{task['id']}** ({timestamp}): {task['text']}")

    if done:
        lines.append("")
        lines.append(f"**Completed** ({len(done)}):")
        lines.append("")
        for task in done:
            lines.append(f"- [x] **#{task['id']}**: {task['text']}")

    return "\n".join(lines)


# =========================================================================
# Hook implementations
# =========================================================================

def hook_post_inference(ctx: Any) -> Optional[dict[str, Any]]:
    """Post-inference hook: extract tasks from LLM response.

    Scans the response for action items and stores them in the
    task database. Appends a summary if tasks were found.
    """
    config = ctx.config or {}
    auto_extract = config.get("auto_extract", _AUTO_EXTRACT)

    if not auto_extract:
        return None

    response = ctx.data.get("response", "")
    if not response or len(response) < 30:
        return None

    # Parse custom patterns
    patterns_str = config.get("patterns", "")
    if patterns_str and isinstance(patterns_str, str):
        patterns = [p.strip() for p in patterns_str.split(",") if p.strip()]
    else:
        patterns = None

    # Extract tasks
    extracted = extract_tasks(response, patterns=patterns, max_tasks=10)
    if not extracted:
        return None

    # Store in database
    db = _get_db(ctx)
    added: list[dict[str, Any]] = []
    for task_info in extracted:
        result = db.add_task(task_info["text"], source=task_info["source"])
        if "error" not in result:
            added.append(result)

    if not added:
        return None

    # Append task summary to response
    task_summary = (
        f"\n\n---\n**Extracted {len(added)} task(s):** "
        + ", ".join(f"#{t['id']}" for t in added)
        + " (use `/tasks` to view)"
    )

    return {
        "response": response + task_summary,
        "extracted_tasks": added,
    }


def hook_tool_call(ctx: Any) -> Optional[dict[str, Any]]:
    """Tool call hook: handle task management slash commands.

    Commands:
        /tasks           List pending tasks
        /tasks all        List all tasks (including completed)
        /tasks done <id>  Mark a task as complete
        /tasks clear      Remove all tasks
        /tasks clear done Remove only completed tasks
    """
    user_input = ctx.data.get("user_input", "") or ctx.data.get("prompt", "")
    if not user_input:
        return None

    user_input = user_input.strip()
    if not user_input.startswith("/tasks"):
        return None

    db = _get_db(ctx)

    # /tasks — list pending
    if _CMD_TASKS_LIST.match(user_input):
        tasks = db.list_tasks(include_done=False)
        return {
            "response": _format_task_list(tasks, "Pending Tasks"),
            "handled": True,
        }

    # /tasks all — list all including done
    if _CMD_TASKS_ALL.match(user_input):
        tasks = db.list_tasks(include_done=True)
        return {
            "response": _format_task_list(tasks, "All Tasks"),
            "handled": True,
        }

    # /tasks done <id>
    m = _CMD_TASKS_DONE.match(user_input)
    if m:
        task_id = int(m.group(1))
        marked = db.mark_done(task_id)
        if marked:
            pending = db.get_pending_count()
            return {
                "response": f"Task #{task_id} marked as done. ({pending} pending)",
                "handled": True,
            }
        else:
            return {
                "response": f"Task #{task_id} not found or already completed.",
                "handled": True,
            }

    # /tasks clear done
    if _CMD_TASKS_CLEAR_DONE.match(user_input):
        removed = db.clear_done()
        return {
            "response": f"Cleared {removed} completed task(s).",
            "handled": True,
        }

    # /tasks clear
    if _CMD_TASKS_CLEAR.match(user_input):
        removed = db.clear_all()
        return {
            "response": f"Cleared all {removed} task(s).",
            "handled": True,
        }

    return None


# =========================================================================
# Hook registry
# =========================================================================

HOOKS = {
    "post_inference": hook_post_inference,
    "tool_call": hook_tool_call,
}


def init() -> None:
    """Plugin initialization."""
    pass


def shutdown() -> None:
    """Plugin shutdown: close database connection."""
    global _db
    if _db is not None:
        _db.close()
        _db = None
