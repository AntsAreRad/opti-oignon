#!/usr/bin/env python3
"""
Admin audit logging for Opti-Oignon.

Provides a dedicated audit table for admin actions on user data.
Separate from the signed_audit_log (which covers security events)
to enable GDPR compliance tracking and admin accountability.

Table: admin_audit_log
  - id: auto-increment primary key
  - admin_id: user_id of the admin performing the action
  - action: action name (e.g. "delete_user_data", "export_user_data")
  - target_type: type of target (e.g. "user", "conversation", "rag_collection")
  - target_id: identifier of the target
  - details: optional JSON details
  - timestamp: UNIX timestamp
  - ip_address: optional IP address of the admin

Storage: SQLite WAL mode in data/admin_audit.db
"""

from __future__ import annotations

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
_DB_NAME = "admin_audit.db"


# ---------------------------------------------------------------------------
# AdminAuditStore
# ---------------------------------------------------------------------------


class AdminAuditStore:
    """SQLite-backed admin audit log.

    Records all admin actions on user data for accountability
    and GDPR compliance.
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
                CREATE TABLE IF NOT EXISTS admin_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    details TEXT DEFAULT '',
                    timestamp REAL NOT NULL,
                    ip_address TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_admin_audit_admin_id
                    ON admin_audit_log(admin_id);
                CREATE INDEX IF NOT EXISTS idx_admin_audit_target
                    ON admin_audit_log(target_type, target_id);
                CREATE INDEX IF NOT EXISTS idx_admin_audit_timestamp
                    ON admin_audit_log(timestamp);
            """)
            conn.commit()
        finally:
            conn.close()

    def log_event(
        self,
        admin_id: str,
        action: str,
        target_type: str,
        target_id: str,
        details: str = "",
        ip_address: str = "",
    ) -> int:
        """Record an admin action.

        Args:
            admin_id: User ID of the admin.
            action: Action name (e.g. "delete_user_data").
            target_type: Type of target (e.g. "user").
            target_id: ID of the target.
            details: Optional details (JSON string or plain text).
            ip_address: Optional IP address.

        Returns:
            The ID of the inserted log entry.
        """
        now = time.time()
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO admin_audit_log
                   (admin_id, action, target_type, target_id, details, timestamp, ip_address)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (admin_id, action, target_type, target_id, details, now, ip_address),
            )
            conn.commit()
            entry_id = cursor.lastrowid or 0
            logger.info(
                "Admin audit: %s performed '%s' on %s/%s",
                admin_id, action, target_type, target_id,
            )
            return entry_id
        finally:
            conn.close()

    def get_events(
        self,
        admin_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query admin audit events with optional filters.

        Args:
            admin_id: Filter by admin user ID.
            target_type: Filter by target type.
            target_id: Filter by target ID.
            since: Filter events after this UNIX timestamp.
            limit: Max results (default 100).
            offset: Pagination offset.

        Returns:
            List of audit event dicts.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if admin_id is not None:
            clauses.append("admin_id = ?")
            params.append(admin_id)
        if target_type is not None:
            clauses.append("target_type = ?")
            params.append(target_type)
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(target_id)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)

        where = ""
        if clauses:
            where = " WHERE " + " AND ".join(clauses)

        sql = f"SELECT * FROM admin_audit_log{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._get_conn()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def count_events(
        self,
        admin_id: str | None = None,
        target_type: str | None = None,
    ) -> int:
        """Count audit events with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if admin_id is not None:
            clauses.append("admin_id = ?")
            params.append(admin_id)
        if target_type is not None:
            clauses.append("target_type = ?")
            params.append(target_type)

        where = ""
        if clauses:
            where = " WHERE " + " AND ".join(clauses)

        conn = self._get_conn()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM admin_audit_log{where}",
                params,
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def delete_events_for_target(
        self,
        target_type: str,
        target_id: str,
    ) -> int:
        """Delete audit events for a specific target (e.g. after user deletion).

        Note: in practice you may want to KEEP audit logs even after
        user deletion for compliance. This method is provided for
        completeness.

        Returns:
            Number of deleted events.
        """
        conn = self._get_conn()
        try:
            result = conn.execute(
                "DELETE FROM admin_audit_log WHERE target_type = ? AND target_id = ?",
                (target_type, target_id),
            )
            conn.commit()
            return result.rowcount
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_admin_audit_store: AdminAuditStore | None = None


def get_admin_audit_store() -> AdminAuditStore:
    """Get or create the singleton AdminAuditStore."""
    global _admin_audit_store
    if _admin_audit_store is None:
        _admin_audit_store = AdminAuditStore()
    return _admin_audit_store


def log_admin_event(
    admin_id: str,
    action: str,
    target_type: str,
    target_id: str,
    details: str = "",
    ip_address: str = "",
) -> int:
    """Convenience function to log an admin event via the singleton."""
    return get_admin_audit_store().log_event(
        admin_id=admin_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=details,
        ip_address=ip_address,
    )


ADMIN_AUDIT_AVAILABLE = True
