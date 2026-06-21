#!/usr/bin/env python3
"""
Per-user data isolation layer for Opti-Oignon (S98).

Provides:
- Default "local" user for single-user mode backward compatibility
- Schema migration: adds user_id columns to existing user-facing tables
- Query helpers for per-user data filtering
- Per-user settings and preferences storage

Design decisions:
- Uses ALTER TABLE ADD COLUMN (non-destructive) for existing databases
- Default user_id = "local" ensures all existing data remains accessible
- Migration runs once on startup, idempotent via column existence check
- Only user-facing tables are isolated; system tables (benchmarks,
  analytics, cache) remain global

Tables migrated:
- conversations (conversation.py)
- memories (memory.py)
- projects (projects.py)
- feedback (feedback.py)
- preferences (session_fingerprint.py)
- branches inherit isolation from their parent conversation

Configuration: controlled by auth.yaml single_user_mode flag
"""

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
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
# Constants
# ---------------------------------------------------------------------------

# The default user ID used in single-user mode and for migrated data
DEFAULT_LOCAL_USER = "local"

# Tables that need user_id isolation and their database paths (relative to DATA_DIR)
# Format: (db_filename, table_name)
ISOLATION_TARGETS = [
    ("conversations.db", "conversations"),
    ("memory.db", "memories"),
    ("projects.db", "projects"),
    ("feedback.db", "feedback"),
    ("fingerprint.db", "preferences"),
]

# S138: Allowed table names for dynamic DDL queries
_ALLOWED_TABLES = frozenset(t[1] for t in ISOLATION_TARGETS)

USER_SETTINGS_AVAILABLE = True

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UserSettings:
    """Per-user settings and preferences."""
    user_id: str
    theme: str = "dark"
    default_model: str = ""
    default_preset: str = ""
    sidebar_open: bool = True
    language: str = "en"
    created_at: float = 0.0
    updated_at: float = 0.0
    preferences: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Migration engine
# ---------------------------------------------------------------------------


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        return column in columns
    except Exception:
        return False


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table exists."""
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def migrate_table(db_path: str | Path, table: str) -> bool:
    """Add user_id column to an existing table if it does not exist.

    All existing rows get user_id = 'local' (the default single-user).
    This is idempotent: safe to run multiple times.

    Returns True if migration was performed, False if already done or N/A.
    """
    path = Path(db_path)
    if not path.exists():
        return False

    try:
        conn = _safe_connect(path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")

        if not _table_exists(conn, table):
            conn.close()
            return False

        if _column_exists(conn, table, "user_id"):
            conn.close()
            return False  # Already migrated

        # S138: validate table name against allowlist
        assert table in _ALLOWED_TABLES, f"Invalid table: {table}"

        # Add column with default value
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN user_id TEXT DEFAULT '{DEFAULT_LOCAL_USER}'"
        )

        # Create index for efficient per-user queries
        index_name = f"idx_{table}_user_id"
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}(user_id)"
        )

        conn.commit()
        conn.close()
        logger.info("Migrated table %s in %s: added user_id column", table, path.name)
        return True
    except Exception as e:
        logger.warning("Migration failed for %s.%s: %s", db_path, table, e)
        return False


def run_migrations(data_dir: str | Path) -> dict[str, bool]:
    """Run all per-user isolation migrations.

    Args:
        data_dir: Path to the data directory containing SQLite databases.

    Returns:
        Dict mapping "db_file.table" to migration success/skip status.
    """
    data_path = Path(data_dir)
    results = {}

    for db_file, table in ISOLATION_TARGETS:
        db_path = data_path / db_file
        key = f"{db_file}.{table}"
        results[key] = migrate_table(db_path, table)

    migrated = [k for k, v in results.items() if v]
    if migrated:
        logger.info("Per-user migrations completed: %s", ", ".join(migrated))
    else:
        logger.debug("No per-user migrations needed")

    return results


# ---------------------------------------------------------------------------
# Per-user settings storage
# ---------------------------------------------------------------------------


class UserSettingsStore:
    """SQLite-backed per-user settings storage.

    Stored in the auth database alongside user accounts.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = _safe_connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    theme TEXT DEFAULT 'dark',
                    default_model TEXT DEFAULT '',
                    default_preset TEXT DEFAULT '',
                    sidebar_open INTEGER DEFAULT 1,
                    language TEXT DEFAULT 'en',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    preferences TEXT DEFAULT '{}'
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def get_settings(self, user_id: str) -> UserSettings:
        """Get settings for a user, creating defaults if none exist."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row:
                prefs = {}
                try:
                    prefs = json.loads(row["preferences"]) if row["preferences"] else {}
                except (json.JSONDecodeError, TypeError):
                    pass
                return UserSettings(
                    user_id=row["user_id"],
                    theme=row["theme"],
                    default_model=row["default_model"],
                    default_preset=row["default_preset"],
                    sidebar_open=bool(row["sidebar_open"]),
                    language=row["language"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    preferences=prefs,
                )
            # Create default settings
            return self._create_defaults(user_id)
        finally:
            conn.close()

    def _create_defaults(self, user_id: str) -> UserSettings:
        """Create default settings for a new user."""
        now = time.time()
        settings = UserSettings(
            user_id=user_id,
            created_at=now,
            updated_at=now,
        )
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO user_settings
                   (user_id, theme, default_model, default_preset, sidebar_open,
                    language, created_at, updated_at, preferences)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, settings.theme, settings.default_model,
                 settings.default_preset, int(settings.sidebar_open),
                 settings.language, now, now, "{}"),
            )
            conn.commit()
        finally:
            conn.close()
        return settings

    def update_settings(self, user_id: str, **kwargs: Any) -> UserSettings:
        """Update settings for a user. Creates defaults first if needed."""
        # Ensure settings exist
        self.get_settings(user_id)

        updates = []
        params: list[Any] = []

        allowed = ("theme", "default_model", "default_preset", "sidebar_open", "language")
        for key in allowed:
            if key in kwargs:
                val = kwargs[key]
                if key == "sidebar_open":
                    val = int(bool(val))
                updates.append(f"{key} = ?")
                params.append(val)

        if "preferences" in kwargs and isinstance(kwargs["preferences"], dict):
            # Merge with existing
            current = self.get_settings(user_id)
            merged = {**current.preferences, **kwargs["preferences"]}
            updates.append("preferences = ?")
            params.append(json.dumps(merged))

        if not updates:
            return self.get_settings(user_id)

        now = time.time()
        updates.append("updated_at = ?")
        params.append(now)
        params.append(user_id)

        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE user_settings SET {} WHERE user_id = ?".format(
                    ", ".join(updates)
                ),
                params,
            )
            conn.commit()
        finally:
            conn.close()

        return self.get_settings(user_id)

    def delete_settings(self, user_id: str) -> bool:
        """Delete all settings for a user."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "DELETE FROM user_settings WHERE user_id = ?", (user_id,)
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def user_filter_sql(user_id: str | None, single_user_mode: bool = True) -> tuple[str, list[str]]:
    """Generate a SQL WHERE clause fragment for per-user filtering.

    In single_user_mode or when user_id is None, returns an empty clause
    (no filtering). Otherwise, returns a clause filtering by user_id.

    Returns:
        Tuple of (sql_fragment, params_list).
        sql_fragment is either "" or " AND user_id = ?"
    """
    if single_user_mode or user_id is None:
        return ("", [])
    return (" AND user_id = ?", [user_id])


def user_filter_where(user_id: str | None, single_user_mode: bool = True) -> tuple[str, list[str]]:
    """Generate a SQL WHERE clause for per-user filtering.

    Same as user_filter_sql but uses WHERE instead of AND.

    Returns:
        Tuple of (sql_fragment, params_list).
        sql_fragment is either "" or " WHERE user_id = ?"
    """
    if single_user_mode or user_id is None:
        return ("", [])
    return (" WHERE user_id = ?", [user_id])


def effective_user_id(user_id: str | None, single_user_mode: bool = True) -> str:
    """Resolve the effective user ID.

    In single-user mode or when no user is provided, returns DEFAULT_LOCAL_USER.
    """
    if single_user_mode or user_id is None:
        return DEFAULT_LOCAL_USER
    return user_id


# ---------------------------------------------------------------------------
# Module-level singleton for settings store
# ---------------------------------------------------------------------------

try:
    _default_db = Path(__file__).parent.parent / "data" / "auth.db"
    user_settings_store = UserSettingsStore(db_path=_default_db)
except Exception as e:
    logger.error("Failed to initialize UserSettingsStore: %s", e)
    user_settings_store = None  # type: ignore
    USER_SETTINGS_AVAILABLE = False
