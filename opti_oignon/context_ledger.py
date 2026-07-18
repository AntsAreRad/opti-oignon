#!/usr/bin/env python3
"""Per-request context ledger: numbers about a turn, never its words.

Optimizing a context pipeline starts with measuring it, and the pipeline
measured here already computes everything worth keeping -- token totals
per zone, cache hit types and similarities, archive retrieval scores, the
resource admission verdict -- then lets it all evaporate as in-memory
properties. This module is the sink that keeps those figures, one row per
request, so budget decisions can be judged against recorded reality
instead of intuition.

Three properties are load-bearing:

* Numbers and labels only. No prompt, no response, no document, no memory
  fact ever enters this file -- zone figures, method labels, bounded
  reason strings, identifiers. There is nothing user-authored to leak
  between users, nothing untrusted to launder into a summary, and nothing
  whose at-rest exposure would matter beyond the metadata it already is.

* Fail-open recording, fail-quiet reading. A write goes through
  ``record()``, which returns False on any fault and never raises: an
  observability sink that can take the chat path down is worse than no
  sink. Reads on an empty or unavailable store come back empty rather
  than erroring, because a metrics panel must degrade, not crash.

* A bounded table. Every insert enforces the retention cap by deleting
  the oldest overflow. A non-positive cap clamps to a floor of one row:
  misconfiguration shrinks the window, it never turns the ledger into an
  unbounded log or a self-erasing one.

All persistence rides the project's safe_connect seam (SQLCipher when
available); every statement is parameterised. When the seam itself is
absent the ledger reports itself unavailable instead of falling back to a
bare connection.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from .db_utils import safe_connect as _safe_connect

    SAFE_CONNECT_AVAILABLE = True
except ImportError:
    SAFE_CONNECT_AVAILABLE = False
    _safe_connect = None

try:
    from .config import DATA_DIR as _DATA_DIR
except ImportError:
    _DATA_DIR = Path(__file__).parent / "data"

DEFAULT_MAX_ROWS = 5000

_LABEL_MAX = 64
_ID_MAX = 128
_REASON_MAX = 256
_ZONES_MAX = 16
_RECENT_LIMIT_MAX = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_ledger_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    conversation_id TEXT,
    model TEXT NOT NULL,
    caller TEXT NOT NULL,
    outcome TEXT NOT NULL,
    token_method TEXT NOT NULL,
    tokens_system INTEGER,
    tokens_history INTEGER,
    tokens_user INTEGER,
    tokens_project INTEGER,
    tokens_manifest INTEGER,
    tokens_total INTEGER,
    zones_json TEXT,
    cache_hit INTEGER NOT NULL,
    cache_hit_type TEXT,
    cache_similarity REAL,
    cache_stored INTEGER NOT NULL,
    retrieval_count INTEGER NOT NULL,
    retrieval_top_score REAL,
    gov_action TEXT,
    gov_admitted INTEGER,
    gov_requested_ctx INTEGER,
    gov_num_ctx INTEGER,
    gov_conditional_eviction INTEGER,
    gov_keep_alive TEXT,
    gov_reason TEXT,
    duration_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_context_ledger_request
    ON context_ledger_entries(request_id);
CREATE INDEX IF NOT EXISTS idx_context_ledger_ts
    ON context_ledger_entries(ts);
"""

_COLUMNS = (
    "ts",
    "request_id",
    "conversation_id",
    "model",
    "caller",
    "outcome",
    "token_method",
    "tokens_system",
    "tokens_history",
    "tokens_user",
    "tokens_project",
    "tokens_manifest",
    "tokens_total",
    "zones_json",
    "cache_hit",
    "cache_hit_type",
    "cache_similarity",
    "cache_stored",
    "retrieval_count",
    "retrieval_top_score",
    "gov_action",
    "gov_admitted",
    "gov_requested_ctx",
    "gov_num_ctx",
    "gov_conditional_eviction",
    "gov_keep_alive",
    "gov_reason",
    "duration_ms",
)

_INSERT_SQL = (
    "INSERT INTO context_ledger_entries ("
    + ", ".join(_COLUMNS)
    + ") VALUES ("
    + ", ".join(":" + column for column in _COLUMNS)
    + ")"
)

_ZONE_KEYS = ("zone", "budgeted", "actual", "trimmed", "strategy")


# ---------------------------------------------------------------------------
# Field coercion (defensive: a malformed field is dropped, never propagated)
# ---------------------------------------------------------------------------

def _clip(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return text[:limit]


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return int(value) if isinstance(value, bool) else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_flag(value: Any) -> int | None:
    """Tri-state boolean: None stays None, everything else becomes 0/1."""
    if value is None:
        return None
    return 1 if bool(value) else 0


def _sanitize_zones(zones: Any) -> str | None:
    """Reduce zone reports to the allowed keys; anything else is dropped."""
    if not zones:
        return None
    out: list[dict[str, Any]] = []
    try:
        for item in zones:
            if len(out) >= _ZONES_MAX:
                break
            if not isinstance(item, dict):
                continue
            budgeted = _as_int(item.get("budgeted"))
            if budgeted is None:
                budgeted = _as_int(item.get("budgeted_tokens"))
            actual = _as_int(item.get("actual"))
            if actual is None:
                actual = _as_int(item.get("actual_tokens"))
            trimmed = _as_int(item.get("trimmed"))
            if trimmed is None:
                trimmed = _as_int(item.get("trimmed_tokens"))
            out.append(
                {
                    "zone": _clip(item.get("zone"), _LABEL_MAX) or "",
                    "budgeted": budgeted if budgeted is not None else 0,
                    "actual": actual if actual is not None else 0,
                    "trimmed": trimmed if trimmed is not None else 0,
                    "strategy": _clip(item.get("strategy"), _LABEL_MAX) or "",
                }
            )
    except Exception:
        return None
    if not out:
        return None
    return json.dumps(out, separators=(",", ":"))


def _parse_zones(zones_json: Any) -> list[dict[str, Any]]:
    if not zones_json:
        return []
    try:
        parsed = json.loads(zones_json)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class ContextLedger:
    """Bounded per-request measurement store.

    Args:
        db_path: Path to the SQLite file (default: DATA_DIR/context_ledger.db).
        max_rows: Retention cap. Non-positive or unparseable values clamp
            to a floor of one row; the table can shrink, never explode and
            never self-erase.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        max_rows: int | None = None,
    ):
        self._db_path = Path(db_path) if db_path is not None else (
            _DATA_DIR / "context_ledger.db"
        )
        raw_cap = DEFAULT_MAX_ROWS if max_rows is None else max_rows
        try:
            self._max_rows = max(1, int(raw_cap))
        except (TypeError, ValueError):
            self._max_rows = DEFAULT_MAX_ROWS
        self._lock = threading.Lock()
        self._ready = False
        if SAFE_CONNECT_AVAILABLE and _safe_connect is not None:
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                with contextlib.closing(self._connect()) as conn:
                    conn.executescript(_SCHEMA)
                    conn.commit()
                self._ready = True
            except Exception as exc:
                logger.warning("Context ledger unavailable: %s", exc)
        else:
            logger.warning(
                "Context ledger unavailable: the safe_connect seam is absent"
            )

    # -- plumbing ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = _safe_connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def available(self) -> bool:
        """Whether the store initialised and can be written and read."""
        return self._ready

    @property
    def max_rows(self) -> int:
        """The effective retention cap after clamping."""
        return self._max_rows

    @property
    def db_path(self) -> Path:
        """Where the ledger lives (informational)."""
        return self._db_path

    # -- writing -----------------------------------------------------------

    def record(
        self,
        *,
        request_id: str,
        model: str,
        outcome: str,
        conversation_id: str | None = None,
        caller: str = "chat",
        token_method: str = "estimated",
        tokens_system: int | None = None,
        tokens_history: int | None = None,
        tokens_user: int | None = None,
        tokens_project: int | None = None,
        tokens_manifest: int | None = None,
        tokens_total: int | None = None,
        zones: Any = None,
        cache_hit: bool = False,
        cache_hit_type: str | None = None,
        cache_similarity: float | None = None,
        cache_stored: bool = False,
        retrieval_count: int = 0,
        retrieval_top_score: float | None = None,
        gov_action: str | None = None,
        gov_admitted: bool | None = None,
        gov_requested_ctx: int | None = None,
        gov_num_ctx: int | None = None,
        gov_conditional_eviction: bool | None = None,
        gov_keep_alive: str | None = None,
        gov_reason: str | None = None,
        duration_ms: float | None = None,
        **_ignored: Any,
    ) -> bool:
        """Write one request row. Returns False on any fault, never raises.

        Unknown keyword fields are ignored rather than rejected, so an
        emitter one release ahead of this schema degrades to a partial row
        instead of a lost one.
        """
        if not self._ready:
            return False
        try:
            values = {
                "ts": time.time(),
                "request_id": _clip(request_id, _ID_MAX) or "",
                "conversation_id": _clip(conversation_id, _ID_MAX),
                "model": _clip(model, _ID_MAX) or "",
                "caller": _clip(caller, _LABEL_MAX) or "chat",
                "outcome": _clip(outcome, _LABEL_MAX) or "unknown",
                "token_method": _clip(token_method, _LABEL_MAX) or "estimated",
                "tokens_system": _as_int(tokens_system),
                "tokens_history": _as_int(tokens_history),
                "tokens_user": _as_int(tokens_user),
                "tokens_project": _as_int(tokens_project),
                "tokens_manifest": _as_int(tokens_manifest),
                "tokens_total": _as_int(tokens_total),
                "zones_json": _sanitize_zones(zones),
                "cache_hit": _as_flag(cache_hit) or 0,
                "cache_hit_type": _clip(cache_hit_type, _LABEL_MAX),
                "cache_similarity": _as_float(cache_similarity),
                "cache_stored": _as_flag(cache_stored) or 0,
                "retrieval_count": _as_int(retrieval_count) or 0,
                "retrieval_top_score": _as_float(retrieval_top_score),
                "gov_action": _clip(gov_action, _LABEL_MAX),
                "gov_admitted": _as_flag(gov_admitted),
                "gov_requested_ctx": _as_int(gov_requested_ctx),
                "gov_num_ctx": _as_int(gov_num_ctx),
                "gov_conditional_eviction": _as_flag(gov_conditional_eviction),
                "gov_keep_alive": _clip(gov_keep_alive, _LABEL_MAX),
                "gov_reason": _clip(gov_reason, _REASON_MAX),
                "duration_ms": _as_float(duration_ms),
            }
            with self._lock:
                with contextlib.closing(self._connect()) as conn:
                    conn.execute(_INSERT_SQL, values)
                    conn.execute(
                        "DELETE FROM context_ledger_entries WHERE id NOT IN ("
                        "SELECT id FROM context_ledger_entries "
                        "ORDER BY id DESC LIMIT ?)",
                        (self._max_rows,),
                    )
                    conn.commit()
            return True
        except Exception as exc:
            logger.debug("Context ledger write skipped: %s", exc)
            return False

    # -- reading -----------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        entry = dict(row)
        entry["zones"] = _parse_zones(entry.pop("zones_json", None))
        return entry

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest rows first; empty on an unavailable or faulted store."""
        if not self._ready:
            return []
        try:
            bounded = max(1, min(int(limit), _RECENT_LIMIT_MAX))
        except (TypeError, ValueError):
            bounded = 50
        try:
            with contextlib.closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT * FROM context_ledger_entries "
                    "ORDER BY id DESC LIMIT ?",
                    (bounded,),
                ).fetchall()
            return [self._row_to_dict(row) for row in rows]
        except Exception as exc:
            logger.debug("Context ledger read skipped: %s", exc)
            return []

    def get(self, request_id: str) -> dict[str, Any] | None:
        """The newest row for one request id, or None."""
        if not self._ready or not request_id:
            return None
        try:
            with contextlib.closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT * FROM context_ledger_entries "
                    "WHERE request_id = ? ORDER BY id DESC LIMIT 1",
                    (str(request_id),),
                ).fetchone()
            return self._row_to_dict(row) if row is not None else None
        except Exception as exc:
            logger.debug("Context ledger read skipped: %s", exc)
            return None

    def stats(self) -> dict[str, Any]:
        """Aggregate view: row count, outcome and method mix, averages."""
        if not self._ready:
            return {"available": False, "rows": 0, "max_rows": self._max_rows}
        try:
            with contextlib.closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT COUNT(*) AS n FROM context_ledger_entries"
                ).fetchone()["n"]
                outcomes = {
                    row["outcome"]: row["n"]
                    for row in conn.execute(
                        "SELECT outcome, COUNT(*) AS n "
                        "FROM context_ledger_entries GROUP BY outcome"
                    ).fetchall()
                }
                methods = {
                    row["token_method"]: row["n"]
                    for row in conn.execute(
                        "SELECT token_method, COUNT(*) AS n "
                        "FROM context_ledger_entries GROUP BY token_method"
                    ).fetchall()
                }
                averages = conn.execute(
                    "SELECT AVG(duration_ms) AS avg_duration_ms, "
                    "AVG(tokens_total) AS avg_tokens_total, "
                    "SUM(cache_hit) AS cache_hits "
                    "FROM context_ledger_entries"
                ).fetchone()
            return {
                "available": True,
                "rows": int(rows),
                "max_rows": self._max_rows,
                "outcomes": outcomes,
                "methods": methods,
                "avg_duration_ms": _as_float(averages["avg_duration_ms"]),
                "avg_tokens_total": _as_float(averages["avg_tokens_total"]),
                "cache_hits": _as_int(averages["cache_hits"]) or 0,
            }
        except Exception as exc:
            logger.debug("Context ledger stats skipped: %s", exc)
            return {"available": False, "rows": 0, "max_rows": self._max_rows}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_ledger: ContextLedger | None = None


def get_context_ledger() -> ContextLedger:
    """Return the shared ledger, building it on first use.

    The instance is always returned; its ``available`` property tells the
    truth about whether the store initialised.
    """
    global _ledger
    if _ledger is None:
        _ledger = ContextLedger()
    return _ledger


def reset_context_ledger() -> None:
    """Drop the shared ledger so the next resolution rebuilds it."""
    global _ledger
    _ledger = None
