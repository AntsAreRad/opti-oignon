#!/usr/bin/env python3
"""Read-only API surface over the per-request context ledger.

Same shape as the governor surface: payload helpers at module level,
importable and testable where fastapi is absent, and a thin FastAPI
wrapper that only maps absence to HTTP codes. The surface never writes --
the ledger is fed by the execution hub, and this router exists so the
recorded figures can be read, not steered.

    GET /api/context/ledger/recent              -> newest rows, bounded limit
    GET /api/context/ledger/stats               -> row count, outcome and method mix
    GET /api/context/ledger/entry/{request_id}  -> one request's row, 404 when absent

Mode-free: reading local measurement rows behaves identically in Daily
and in Bulbe, and nothing here can reach user content -- the ledger never
stored any.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from ..context_ledger import get_context_ledger

    CONTEXT_LEDGER_AVAILABLE = True
except ImportError:
    CONTEXT_LEDGER_AVAILABLE = False
    get_context_ledger = None


def _resolve_ledger() -> Any:
    """The shared ledger instance, or None when the module is absent."""
    if not CONTEXT_LEDGER_AVAILABLE or get_context_ledger is None:
        return None
    try:
        return get_context_ledger()
    except Exception as exc:
        logger.debug("Context ledger resolution failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Web-free payload helpers
# ---------------------------------------------------------------------------

def recent_payload(ledger: Any, limit: int = 50) -> dict[str, Any]:
    """Newest ledger rows; empty and honest when the store is absent."""
    if ledger is None or not getattr(ledger, "available", False):
        return {"available": False, "count": 0, "entries": []}
    try:
        entries = ledger.recent(limit=limit)
    except Exception as exc:
        logger.debug("Ledger recent read failed: %s", exc)
        entries = []
    return {"available": True, "count": len(entries), "entries": entries}


def stats_payload(ledger: Any) -> dict[str, Any]:
    """The ledger's aggregate view; an honest stub when the store is absent."""
    if ledger is None:
        return {"available": False, "rows": 0}
    try:
        return ledger.stats()
    except Exception as exc:
        logger.debug("Ledger stats read failed: %s", exc)
        return {"available": False, "rows": 0}


def entry_payload(ledger: Any, request_id: str) -> dict[str, Any] | None:
    """One request's row, or None when unknown or the store is absent."""
    if ledger is None or not getattr(ledger, "available", False):
        return None
    try:
        return ledger.get(request_id)
    except Exception as exc:
        logger.debug("Ledger entry read failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Thin FastAPI wrapper
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

if _FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/api/context/ledger", tags=["context-ledger"])

    @router.get("/recent")
    def get_recent(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        """Newest recorded requests, most recent first."""
        return recent_payload(_resolve_ledger(), limit=limit)

    @router.get("/stats")
    def get_stats() -> dict[str, Any]:
        """Row count, retention cap, outcome and method mix, averages."""
        return stats_payload(_resolve_ledger())

    @router.get("/entry/{request_id}")
    def get_entry(request_id: str) -> dict[str, Any]:
        """One request's recorded row."""
        payload = entry_payload(_resolve_ledger(), request_id)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail="No ledger row for that request id",
            )
        return payload
else:
    router = None
