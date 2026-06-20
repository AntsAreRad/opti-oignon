#!/usr/bin/env python3
"""
TELEMETRY HISTORY STORE -- OPTI-OIGNON S114
=============================================

SQLite persistence for inference telemetry events.  Registers as a
telemetry consumer to capture inference_end events automatically and
stores them for historical analysis beyond the in-memory ring buffer.

Architecture:
    TelemetryHistoryStore  -- SQLite DB, consumer callback, queries
    get_history_store()    -- module-level singleton accessor

Features:
    - Automatic capture via telemetry consumer registration
    - Configurable retention policy (max age in days)
    - Paginated event history
    - Aggregation queries: events per hour, per model, latency trends
    - Thread-safe with WAL mode

Author: Leon
"""

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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

DEFAULT_DB_NAME = "telemetry_history.db"
DEFAULT_RETENTION_DAYS = 7
DEFAULT_MAX_EVENTS = 50_000


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class HistoryEvent:
    """A persisted inference event."""

    id: int = 0
    request_id: str = ""
    model: str = ""
    timestamp: float = 0.0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tok_per_sec: float = 0.0
    prompt_eval_ms: float = 0.0
    token_gen_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "model": self.model,
            "timestamp": self.timestamp,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tok_per_sec": round(self.tok_per_sec, 2),
            "prompt_eval_ms": round(self.prompt_eval_ms, 2),
            "token_gen_ms": round(self.token_gen_ms, 2),
        }


@dataclass
class TrendBucket:
    """Aggregated data for a time bucket (hourly)."""

    bucket_start: float = 0.0
    bucket_label: str = ""
    event_count: int = 0
    avg_latency_ms: float = 0.0
    avg_tok_per_sec: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_start": self.bucket_start,
            "bucket_label": self.bucket_label,
            "event_count": self.event_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_tok_per_sec": round(self.avg_tok_per_sec, 2),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
        }


# ---------------------------------------------------------------------------
# TelemetryHistoryStore
# ---------------------------------------------------------------------------


class TelemetryHistoryStore:
    """SQLite-backed telemetry event store.

    Registers as a telemetry consumer and persists inference_end events
    to a local SQLite database for historical analysis.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    retention_days : int
        Maximum age for stored events (older events are purged).
    max_events : int
        Hard cap on stored events.
    """

    def __init__(
        self,
        db_path: str | Path = "",
        retention_days: int = DEFAULT_RETENTION_DAYS,
        max_events: int = DEFAULT_MAX_EVENTS,
    ) -> None:
        self._db_path = str(db_path) if db_path else ""
        self._retention_days = max(1, retention_days)
        self._max_events = max(100, max_events)
        self._lock = threading.RLock()
        self._total_stored: int = 0
        self._auto_purge_enabled: bool = False
        self._auto_purge_timer: Optional[threading.Timer] = None

        if self._db_path:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            self._init_db()

        # Set __name__ for telemetry dashboard consumer display.
        self.consume.__func__.__name__ = "telemetry_history_consumer"  # type: ignore[attr-defined]

    # ----- Database init -----

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    timestamp REAL NOT NULL,
                    latency_ms REAL NOT NULL DEFAULT 0,
                    tokens_in INTEGER NOT NULL DEFAULT 0,
                    tokens_out INTEGER NOT NULL DEFAULT 0,
                    tok_per_sec REAL NOT NULL DEFAULT 0,
                    prompt_eval_ms REAL NOT NULL DEFAULT 0,
                    token_gen_ms REAL NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON inference_events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_model
                ON inference_events(model)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a SQLite connection (one per call, for thread safety)."""
        conn = _safe_connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ----- Telemetry consumer interface -----

    def consume(self, events: list) -> None:
        """Telemetry consumer callback.

        Persists inference_end events to SQLite.
        """
        if not self._db_path:
            return

        to_insert: list[tuple] = []
        for ev in events:
            etype = getattr(ev, "event_type", "") or ""
            if etype != "inference_end":
                continue

            rid = getattr(ev, "request_id", "") or ""
            if not rid:
                continue

            data = getattr(ev, "data", {}) or {}
            ts = getattr(ev, "timestamp", 0.0) or time.time()
            latency = data.get("latency_ms", 0.0)
            tokens_in = data.get("tokens_in", 0)
            tokens_out = data.get("tokens_out", 0)
            tok_s = (tokens_out / (latency / 1000.0)) if latency > 0 else 0.0

            to_insert.append((
                rid,
                getattr(ev, "model", "") or "",
                ts,
                latency,
                tokens_in,
                tokens_out,
                round(tok_s, 2),
                data.get("prompt_eval_ms", 0.0),
                data.get("token_gen_ms", 0.0),
            ))

        if not to_insert:
            return

        try:
            with self._lock:
                with self._get_conn() as conn:
                    conn.executemany(
                        """INSERT INTO inference_events
                           (request_id, model, timestamp, latency_ms,
                            tokens_in, tokens_out, tok_per_sec,
                            prompt_eval_ms, token_gen_ms)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        to_insert,
                    )
                    conn.commit()
                self._total_stored += len(to_insert)
        except Exception as exc:
            logger.warning("Failed to persist telemetry events: %s", exc)

    # ----- Public query API -----

    def get_history(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        model: str = "",
    ) -> dict[str, Any]:
        """Get paginated event history.

        Returns dict with 'events' (list of dicts), 'total', 'limit', 'offset'.
        """
        if not self._db_path:
            return {"events": [], "total": 0, "limit": limit, "offset": offset}

        with self._lock:
            with self._get_conn() as conn:
                # Count total
                if model:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM inference_events WHERE model = ?",
                        (model,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM inference_events"
                    ).fetchone()
                total = row[0] if row else 0

                # Fetch page
                if model:
                    rows = conn.execute(
                        """SELECT * FROM inference_events
                           WHERE model = ?
                           ORDER BY timestamp DESC
                           LIMIT ? OFFSET ?""",
                        (model, limit, offset),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM inference_events
                           ORDER BY timestamp DESC
                           LIMIT ? OFFSET ?""",
                        (limit, offset),
                    ).fetchall()

        events = []
        for r in rows:
            events.append(HistoryEvent(
                id=r["id"],
                request_id=r["request_id"],
                model=r["model"],
                timestamp=r["timestamp"],
                latency_ms=r["latency_ms"],
                tokens_in=r["tokens_in"],
                tokens_out=r["tokens_out"],
                tok_per_sec=r["tok_per_sec"],
                prompt_eval_ms=r["prompt_eval_ms"],
                token_gen_ms=r["token_gen_ms"],
            ).to_dict())

        return {
            "events": events,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_trends(
        self,
        *,
        hours: int = 24,
        model: str = "",
    ) -> list[dict[str, Any]]:
        """Get aggregated latency/throughput trends over time.

        Returns a list of hourly buckets with avg latency, tok/s, counts.
        """
        if not self._db_path:
            return []

        cutoff = time.time() - (hours * 3600)
        bucket_seconds = 3600  # 1 hour buckets

        with self._lock:
            with self._get_conn() as conn:
                if model:
                    rows = conn.execute(
                        """SELECT timestamp, latency_ms, tok_per_sec,
                                  tokens_in, tokens_out
                           FROM inference_events
                           WHERE timestamp >= ? AND model = ?
                           ORDER BY timestamp""",
                        (cutoff, model),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT timestamp, latency_ms, tok_per_sec,
                                  tokens_in, tokens_out
                           FROM inference_events
                           WHERE timestamp >= ?
                           ORDER BY timestamp""",
                        (cutoff,),
                    ).fetchall()

        # Group into buckets
        buckets: dict[int, list[dict]] = {}
        for r in rows:
            ts = r["timestamp"]
            bucket_key = int(ts // bucket_seconds) * bucket_seconds
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append({
                "latency_ms": r["latency_ms"],
                "tok_per_sec": r["tok_per_sec"],
                "tokens_in": r["tokens_in"],
                "tokens_out": r["tokens_out"],
            })

        # Build trend list
        result: list[dict[str, Any]] = []
        for bucket_ts in sorted(buckets.keys()):
            items = buckets[bucket_ts]
            n = len(items)
            avg_lat = sum(i["latency_ms"] for i in items) / n if n else 0
            avg_tok = sum(i["tok_per_sec"] for i in items) / n if n else 0
            total_in = sum(i["tokens_in"] for i in items)
            total_out = sum(i["tokens_out"] for i in items)

            from datetime import datetime
            label = datetime.fromtimestamp(bucket_ts).strftime("%Y-%m-%d %H:00")

            result.append(TrendBucket(
                bucket_start=bucket_ts,
                bucket_label=label,
                event_count=n,
                avg_latency_ms=avg_lat,
                avg_tok_per_sec=avg_tok,
                total_tokens_in=total_in,
                total_tokens_out=total_out,
            ).to_dict())

        return result

    def get_model_breakdown(self) -> list[dict[str, Any]]:
        """Get event count and average latency per model."""
        if not self._db_path:
            return []

        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    """SELECT model,
                              COUNT(*) as count,
                              AVG(latency_ms) as avg_latency,
                              AVG(tok_per_sec) as avg_toks,
                              SUM(tokens_in) as total_in,
                              SUM(tokens_out) as total_out
                       FROM inference_events
                       GROUP BY model
                       ORDER BY count DESC"""
                ).fetchall()

        return [
            {
                "model": r["model"],
                "event_count": r["count"],
                "avg_latency_ms": round(r["avg_latency"] or 0, 2),
                "avg_tok_per_sec": round(r["avg_toks"] or 0, 2),
                "total_tokens_in": r["total_in"] or 0,
                "total_tokens_out": r["total_out"] or 0,
            }
            for r in rows
        ]

    def purge(self, *, older_than_days: Optional[int] = None) -> int:
        """Delete old events based on retention policy.

        Parameters
        ----------
        older_than_days : int or None
            Override retention days. Uses configured value if None.

        Returns
        -------
        int
            Number of events purged.
        """
        if not self._db_path:
            return 0

        days = older_than_days if older_than_days is not None else self._retention_days
        cutoff = time.time() - (days * 86400)

        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM inference_events WHERE timestamp < ?",
                    (cutoff,),
                )
                deleted = cursor.rowcount
                conn.commit()

        if deleted > 0:
            logger.info("Purged %d telemetry events older than %d days", deleted, days)
        return deleted

    def purge_all(self) -> int:
        """Delete ALL stored events.

        Returns the number of events deleted.
        """
        if not self._db_path:
            return 0

        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute("DELETE FROM inference_events")
                deleted = cursor.rowcount
                conn.commit()

        if deleted > 0:
            logger.info("Purged all %d telemetry events", deleted)
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Quick overview stats."""
        if not self._db_path:
            return {
                "available": False,
                "total_stored": 0,
                "retention_days": self._retention_days,
                "auto_purge_enabled": self._auto_purge_enabled,
            }

        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM inference_events"
                ).fetchone()
                total = row[0] if row else 0

                oldest_row = conn.execute(
                    "SELECT MIN(timestamp) FROM inference_events"
                ).fetchone()
                oldest = oldest_row[0] if oldest_row and oldest_row[0] else 0

        return {
            "available": True,
            "total_stored": total,
            "retention_days": self._retention_days,
            "oldest_event_ts": oldest,
            "max_events": self._max_events,
            "auto_purge_enabled": self._auto_purge_enabled,
        }

    # ----- Settings management (S115) -----

    def update_settings(
        self,
        *,
        retention_days: Optional[int] = None,
        auto_purge_enabled: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Update retention configuration at runtime.

        Parameters
        ----------
        retention_days : int or None
            New retention period (1-365). None = keep current.
        auto_purge_enabled : bool or None
            Enable/disable daily auto-purge. None = keep current.

        Returns
        -------
        dict
            Updated settings snapshot.
        """
        with self._lock:
            if retention_days is not None:
                self._retention_days = max(1, min(365, retention_days))
            if auto_purge_enabled is not None:
                self._auto_purge_enabled = bool(auto_purge_enabled)
                if self._auto_purge_enabled:
                    self._start_auto_purge()
                else:
                    self._stop_auto_purge()

        logger.info(
            "Telemetry history settings updated: retention=%dd, auto_purge=%s",
            self._retention_days,
            self._auto_purge_enabled,
        )
        return {
            "retention_days": self._retention_days,
            "auto_purge_enabled": self._auto_purge_enabled,
        }

    def _start_auto_purge(self) -> None:
        """Start daily auto-purge background timer."""
        self._stop_auto_purge()
        if not self._auto_purge_enabled:
            return

        def _do_auto_purge():
            try:
                deleted = self.purge()
                if deleted > 0:
                    logger.info("Auto-purge removed %d events", deleted)
            except Exception as exc:
                logger.warning("Auto-purge failed: %s", exc)
            finally:
                # Reschedule for next day
                if self._auto_purge_enabled:
                    self._auto_purge_timer = threading.Timer(86400, _do_auto_purge)
                    self._auto_purge_timer.daemon = True
                    self._auto_purge_timer.start()

        self._auto_purge_timer = threading.Timer(86400, _do_auto_purge)
        self._auto_purge_timer.daemon = True
        self._auto_purge_timer.start()
        logger.info("Auto-purge scheduled (every 24h, retention=%dd)", self._retention_days)

    def _stop_auto_purge(self) -> None:
        """Cancel the auto-purge timer if running."""
        if self._auto_purge_timer is not None:
            self._auto_purge_timer.cancel()
            self._auto_purge_timer = None

    # ----- CSV export (S115) -----

    def export_csv(self, *, model: str = "") -> str:
        """Export event history as CSV string.

        Parameters
        ----------
        model : str
            Optional model filter. Empty = all models.

        Returns
        -------
        str
            CSV formatted string with header row.
        """
        if not self._db_path:
            return "id,request_id,model,timestamp,latency_ms,tokens_in,tokens_out,tok_per_sec,prompt_eval_ms,token_gen_ms\n"

        with self._lock:
            with self._get_conn() as conn:
                if model:
                    rows = conn.execute(
                        """SELECT * FROM inference_events
                           WHERE model = ?
                           ORDER BY timestamp DESC""",
                        (model,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT * FROM inference_events
                           ORDER BY timestamp DESC"""
                    ).fetchall()

        lines = ["id,request_id,model,timestamp,latency_ms,tokens_in,tokens_out,tok_per_sec,prompt_eval_ms,token_gen_ms"]
        for r in rows:
            # Escape commas in model names
            model_safe = str(r["model"]).replace(",", ";")
            lines.append(
                f'{r["id"]},{r["request_id"]},{model_safe},'
                f'{r["timestamp"]},{r["latency_ms"]},'
                f'{r["tokens_in"]},{r["tokens_out"]},'
                f'{r["tok_per_sec"]},{r["prompt_eval_ms"]},'
                f'{r["token_gen_ms"]}'
            )
        return "\n".join(lines) + "\n"

    def shutdown(self) -> None:
        """Cleanup before shutdown."""
        self._stop_auto_purge()
        self.purge()
        logger.info("TelemetryHistoryStore shutdown complete")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Optional[TelemetryHistoryStore] = None
_store_lock = threading.Lock()

TELEMETRY_HISTORY_AVAILABLE = True


def get_history_store(
    db_path: str | Path = "",
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> TelemetryHistoryStore:
    """Get or create the singleton TelemetryHistoryStore."""
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store

        if not db_path:
            try:
                from opti_oignon.config import DATA_DIR
                db_path = Path(DATA_DIR) / DEFAULT_DB_NAME
            except ImportError:
                db_path = Path("/tmp") / DEFAULT_DB_NAME

        _store = TelemetryHistoryStore(
            db_path=db_path,
            retention_days=retention_days,
        )

        # Auto-register as telemetry consumer.
        try:
            from opti_oignon.telemetry import get_telemetry
            collector = get_telemetry()
            collector.register_consumer(_store.consume)
            logger.info("TelemetryHistoryStore registered as telemetry consumer")
        except Exception as exc:
            logger.debug("Could not register history store with telemetry: %s", exc)

        return _store


def reset_history_store() -> None:
    """Reset the singleton (for testing)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.shutdown()
        _store = None
