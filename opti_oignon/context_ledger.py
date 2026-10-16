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

A turn the onion's journal laid adds the figures of its window: the rung
it was planned at and composed under, the context and the answer's room
it was sent with, its count and what it is accounted, its epoch, why that
epoch began and where the window first changed, whether it was cold or a
descent, how many of each kind of source it left out, whether the
engine's slot bound was read, and why it was refused with the whole
numbers of the refusal; on a cold window, the engine's own count of its
prompt. Numbers and bare-word labels again: a motive that is not one word
is dropped. A ledger written before these columns gains them when it is
opened, in one transaction a failure leaves undone, and every row it
already holds reads NULL there.

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
# The most whole numbers a refusal's figures keep, and the most kinds of
# source a window's cut names.
_FIGURES_MAX = 8
_CUT_MAX = 16

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
    duration_ms REAL,
    win_plan_rung INTEGER,
    win_refusal TEXT,
    win_figures_json TEXT,
    win_rung INTEGER,
    win_num_ctx INTEGER,
    win_num_predict INTEGER,
    win_tokens INTEGER,
    win_accounted INTEGER,
    win_epoch INTEGER,
    win_motive TEXT,
    win_first_diff INTEGER,
    win_cold INTEGER,
    win_descent INTEGER,
    win_cut_json TEXT,
    win_slot_unknown INTEGER,
    engine_prompt_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS idx_context_ledger_request
    ON context_ledger_entries(request_id);
CREATE INDEX IF NOT EXISTS idx_context_ledger_ts
    ON context_ledger_entries(ts);
"""

# The journal's figures and the engine's count of a cold prompt arrived
# after the table: a file written before them gains these columns, in this
# order, when it is opened (``_gain_window_columns``).
_WINDOW_COLUMNS = (
    ("win_plan_rung", "INTEGER"),
    ("win_refusal", "TEXT"),
    ("win_figures_json", "TEXT"),
    ("win_rung", "INTEGER"),
    ("win_num_ctx", "INTEGER"),
    ("win_num_predict", "INTEGER"),
    ("win_tokens", "INTEGER"),
    ("win_accounted", "INTEGER"),
    ("win_epoch", "INTEGER"),
    ("win_motive", "TEXT"),
    ("win_first_diff", "INTEGER"),
    ("win_cold", "INTEGER"),
    ("win_descent", "INTEGER"),
    ("win_cut_json", "TEXT"),
    ("win_slot_unknown", "INTEGER"),
    ("engine_prompt_tokens", "INTEGER"),
)

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
) + tuple(column for column, _definition in _WINDOW_COLUMNS)

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
    """A whole number SQLite can hold, or None: an infinity, a NaN or a number past 64 bits drops the field,
    never the row."""
    if value is None or isinstance(value, bool):
        return int(value) if isinstance(value, bool) else None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if -(1 << 63) <= number < (1 << 63) else None


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


def _word(value: Any) -> str | None:
    """A label that is one bare word -- ASCII letters, digits and
    underscores -- clipped to its bound, or None: a motive is a name from a
    closed list, never a sentence."""
    text = _clip(value, _LABEL_MAX) if isinstance(value, str) else None
    if text is None or not text.isascii() or not text.replace("_", "").isalnum():
        return None
    return text


def _sanitize_figures(figures: Any) -> str | None:
    """A refusal's figures, its whole numbers alone and at most
    ``_FIGURES_MAX`` of them; None when no list was given, so a refusal with
    no figures keeps its empty list."""
    if not isinstance(figures, (list, tuple)):
        return None
    kept = [value for value in figures if type(value) is int][:_FIGURES_MAX]
    return json.dumps(kept, separators=(",", ":"))


def _sanitize_cut(cut: Any) -> str | None:
    """How many items a window left out, by kind of source: each kind a bare
    word, each count a whole number, at most ``_CUT_MAX`` kinds; anything else
    dropped. None when no mapping was given."""
    if not isinstance(cut, dict):
        return None
    kept: dict[str, int] = {}
    try:
        for kind, count in cut.items():
            if len(kept) >= _CUT_MAX:
                break
            word = _word(kind)
            if word is not None and type(count) is int:
                kept[word] = count
    except Exception:
        return None
    return json.dumps(kept, sort_keys=True, separators=(",", ":"))


def _parse_figures(figures_json: Any) -> list[int] | None:
    if not figures_json:
        return None
    try:
        parsed = json.loads(figures_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    return [value for value in parsed if type(value) is int]


def _parse_cut(cut_json: Any) -> dict[str, int] | None:
    if not cut_json:
        return None
    try:
        parsed = json.loads(cut_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return {kind: count for kind, count in parsed.items() if isinstance(kind, str) and type(count) is int}


def _gain_window_columns(conn: sqlite3.Connection) -> None:
    """Bring a ledger written before the journal's figures to the table:
    each column of ``_WINDOW_COLUMNS`` it lacks is added, every row it
    already holds reading NULL there, in one transaction a failure leaves
    undone. The columns are read again under the write lock, so a file
    another process brought up meanwhile is left as it is."""

    def missing() -> list[tuple[str, str]]:
        present = {row[1] for row in conn.execute("PRAGMA table_info(context_ledger_entries)").fetchall()}
        return [(column, definition) for column, definition in _WINDOW_COLUMNS if column not in present]

    if not missing():
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        for column, definition in missing():
            conn.execute(f"ALTER TABLE context_ledger_entries ADD COLUMN {column} {definition}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


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
                    _gain_window_columns(conn)
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
        win_plan_rung: int | None = None,
        win_refusal: str | None = None,
        win_figures: Any = None,
        win_rung: int | None = None,
        win_num_ctx: int | None = None,
        win_num_predict: int | None = None,
        win_tokens: int | None = None,
        win_accounted: int | None = None,
        win_epoch: int | None = None,
        win_motive: str | None = None,
        win_first_diff: int | None = None,
        win_cold: bool | None = None,
        win_descent: bool | None = None,
        win_cut: Any = None,
        win_slot_unknown: bool | None = None,
        engine_prompt_tokens: int | None = None,
        **_ignored: Any,
    ) -> bool:
        """Write one request row. Returns False on any fault, never raises.

        Unknown keyword fields are ignored rather than rejected, so an
        emitter one release ahead of this schema degrades to a partial row
        instead of a lost one. The ``win_*`` fields are a journal window's
        figures, as the executor reports them; ``win_figures`` a refusal's
        whole numbers, ``win_cut`` the items left out by kind of source.
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
                "win_plan_rung": _as_int(win_plan_rung),
                "win_refusal": _word(win_refusal),
                "win_figures_json": _sanitize_figures(win_figures),
                "win_rung": _as_int(win_rung),
                "win_num_ctx": _as_int(win_num_ctx),
                "win_num_predict": _as_int(win_num_predict),
                "win_tokens": _as_int(win_tokens),
                "win_accounted": _as_int(win_accounted),
                "win_epoch": _as_int(win_epoch),
                "win_motive": _word(win_motive),
                "win_first_diff": _as_int(win_first_diff),
                "win_cold": _as_flag(win_cold),
                "win_descent": _as_flag(win_descent),
                "win_cut_json": _sanitize_cut(win_cut),
                "win_slot_unknown": _as_flag(win_slot_unknown),
                "engine_prompt_tokens": _as_int(engine_prompt_tokens),
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
        entry["win_figures"] = _parse_figures(entry.pop("win_figures_json", None))
        entry["win_cut"] = _parse_cut(entry.pop("win_cut_json", None))
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
