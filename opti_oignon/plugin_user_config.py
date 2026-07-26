#!/usr/bin/env python3
"""
Per-user plugin configuration store for Opti-Oignon.

Provides user-scoped plugin settings:
- Users can enable/disable plugins independently
- Per-user plugin preferences (JSON blob per plugin per user)
- Global allowlist remains in plugin_allowlist.py (admin-controlled)
- This module only handles user-level overrides

Storage: SQLite WAL mode in data/plugin_user_config.db
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from opti_oignon.db_utils import safe_connect as _safe_connect
except ImportError:
    import sqlite3 as _sq3
    _safe_connect = lambda p, **kw: _sq3.connect(str(p), **kw)  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_NAME = "plugin_user_config.db"


# ---------------------------------------------------------------------------
# PluginUserConfigStore
# ---------------------------------------------------------------------------


class PluginUserConfigStore:
    """SQLite-backed per-user plugin configuration.

    Each user has independent enable/disable and preferences for each plugin.
    The global allowlist (plugin_allowlist.py) takes precedence: if a plugin
    is not in the global allowlist, per-user config is irrelevant.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_DB_DIR / _DB_NAME
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = _safe_connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS plugin_user_config (
                    user_id TEXT NOT NULL,
                    plugin_name TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    preferences TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, plugin_name)
                );

                CREATE INDEX IF NOT EXISTS idx_puc_user_id
                    ON plugin_user_config(user_id);
                CREATE INDEX IF NOT EXISTS idx_puc_plugin_name
                    ON plugin_user_config(plugin_name);
            """)
            conn.commit()
        finally:
            conn.close()

    def get_config(
        self, user_id: str, plugin_name: str
    ) -> dict[str, Any] | None:
        """Get per-user config for a specific plugin.

        Returns None if no config exists (user hasn't customized this plugin).
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM plugin_user_config WHERE user_id = ? AND plugin_name = ?",
                (user_id, plugin_name),
            ).fetchone()
            if row is None:
                return None
            prefs = {}
            try:
                prefs = json.loads(row["preferences"]) if row["preferences"] else {}
            except (json.JSONDecodeError, TypeError):
                pass
            return {
                "user_id": row["user_id"],
                "plugin_name": row["plugin_name"],
                "enabled": bool(row["enabled"]),
                "preferences": prefs,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def set_config(
        self,
        user_id: str,
        plugin_name: str,
        enabled: bool | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Set or update per-user plugin config.

        Args:
            user_id: User identifier.
            plugin_name: Plugin name.
            enabled: Whether the plugin is enabled for this user.
            preferences: User-specific plugin preferences.

        Returns:
            The updated config dict.
        """
        now = time.time()
        existing = self.get_config(user_id, plugin_name)

        if existing is None:
            # Insert new
            en = enabled if enabled is not None else True
            prefs = preferences or {}
            conn = self._get_conn()
            inserted = False
            try:
                conn.execute(
                    """INSERT INTO plugin_user_config
                       (user_id, plugin_name, enabled, preferences, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, plugin_name, int(en), json.dumps(prefs), now, now),
                )
                conn.commit()
                inserted = True
            except sqlite3.IntegrityError:
                # PUC-2: a concurrent writer inserted the same
                # (user_id, plugin_name) between our get_config and this
                # insert. Fall through to the update path instead of
                # crashing with an unhandled IntegrityError.
                pass
            finally:
                conn.close()
            if not inserted:
                return self.set_config(
                    user_id, plugin_name,
                    enabled=enabled, preferences=preferences,
                )
            return {
                "user_id": user_id,
                "plugin_name": plugin_name,
                "enabled": en,
                "preferences": prefs,
                "created_at": now,
                "updated_at": now,
            }

        # Update existing
        updates: list[str] = []
        params: list[Any] = []

        if enabled is not None:
            updates.append("enabled = ?")
            params.append(int(enabled))

        if preferences is not None:
            merged = {**existing["preferences"], **preferences}
            updates.append("preferences = ?")
            params.append(json.dumps(merged))

        if not updates:
            return existing

        updates.append("updated_at = ?")
        params.append(now)
        params.extend([user_id, plugin_name])

        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE plugin_user_config SET {} WHERE user_id = ? AND plugin_name = ?".format(
                    ", ".join(updates)
                ),
                params,
            )
            conn.commit()
        finally:
            conn.close()

        return self.get_config(user_id, plugin_name) or existing

    def is_plugin_enabled(self, user_id: str, plugin_name: str) -> bool:
        """Check if a plugin is enabled for a user.

        Returns True if no config exists (default: enabled).
        """
        config = self.get_config(user_id, plugin_name)
        if config is None:
            return True  # Default: enabled
        return config["enabled"]

    def get_all_configs(self, user_id: str) -> list[dict[str, Any]]:
        """Get all plugin configs for a user."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM plugin_user_config WHERE user_id = ? ORDER BY plugin_name",
                (user_id,),
            ).fetchall()
            result = []
            for row in rows:
                prefs = {}
                try:
                    prefs = json.loads(row["preferences"]) if row["preferences"] else {}
                except (json.JSONDecodeError, TypeError):
                    pass
                result.append({
                    "user_id": row["user_id"],
                    "plugin_name": row["plugin_name"],
                    "enabled": bool(row["enabled"]),
                    "preferences": prefs,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                })
            return result
        finally:
            conn.close()

    def delete_config(self, user_id: str, plugin_name: str) -> bool:
        """Delete a specific plugin config for a user."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "DELETE FROM plugin_user_config WHERE user_id = ? AND plugin_name = ?",
                (user_id, plugin_name),
            )
            conn.commit()
            return result.rowcount > 0
        finally:
            conn.close()

    def delete_all_configs(self, user_id: str) -> int:
        """Delete all plugin configs for a user (account deletion)."""
        conn = self._get_conn()
        try:
            result = conn.execute(
                "DELETE FROM plugin_user_config WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return result.rowcount
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_plugin_user_config_store: PluginUserConfigStore | None = None


def get_plugin_user_config_store() -> PluginUserConfigStore:
    """Get or create the singleton PluginUserConfigStore."""
    global _plugin_user_config_store
    if _plugin_user_config_store is None:
        _plugin_user_config_store = PluginUserConfigStore()
    return _plugin_user_config_store


PLUGIN_USER_CONFIG_AVAILABLE = True
