#!/usr/bin/env python3
"""
API routes for the Telemetry Dashboard -- S113.

Exposes collector statistics, consumer health, and manual flush
via a lightweight REST interface. All endpoints degrade gracefully
when the telemetry module is unavailable.
"""

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from .deps import (
    TELEMETRY_AVAILABLE,
    TELEMETRY_HISTORY_AVAILABLE,
    get_history_store,
    get_telemetry,
)
from .schemas import (
    HistoryEventSchema,
    ModelBreakdownSchema,
    TelemetryConsumerInfoSchema,
    TelemetryConsumersResponse,
    TelemetryFlushResponse,
    TelemetryHistoryPurgeResponse,
    TelemetryHistoryResponse,
    TelemetryHistorySettingsRequest,
    TelemetryHistorySettingsResponse,
    TelemetryHistoryStatsResponse,
    TelemetryStatsResponse,
    TelemetryTrendsResponse,
    TrendBucketSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


def _ensure_available():
    """Raise 503 if telemetry is not available."""
    if not TELEMETRY_AVAILABLE or get_telemetry is None:
        raise HTTPException(
            status_code=503,
            detail="Telemetry pipeline is not available",
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@router.get("/stats", response_model=TelemetryStatsResponse)
def get_telemetry_stats() -> dict:
    """Get telemetry collector statistics."""
    if not TELEMETRY_AVAILABLE or get_telemetry is None:
        return TelemetryStatsResponse(enabled=False)

    try:
        collector = get_telemetry()
        raw = collector.get_stats()
        return TelemetryStatsResponse(
            enabled=raw.get("enabled", False),
            total_events=raw.get("total_events", 0),
            total_requests=raw.get("total_requests", 0),
            total_tokens=raw.get("total_tokens", 0),
            active_requests=raw.get("active_requests", 0),
            buffer_size=raw.get("buffer_size", 0),
            buffer_max_size=collector.config.buffer_max_size,
            consumer_count=raw.get("consumer_count", 0),
        )
    except Exception as exc:
        logger.warning("Failed to get telemetry stats: %s", exc)
        return TelemetryStatsResponse(enabled=False)


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------


@router.get("/consumers", response_model=TelemetryConsumersResponse)
def get_telemetry_consumers() -> dict:
    """Get list of registered telemetry consumers with health status."""
    _ensure_available()
    collector = get_telemetry()

    consumers_info: list[TelemetryConsumerInfoSchema] = []
    # Access internal consumer list under lock.
    with collector._lock:
        for c in collector._consumers:
            name = getattr(c, "__name__", repr(c))
            # A consumer is considered healthy if it is still callable.
            healthy = callable(c)
            consumers_info.append(
                TelemetryConsumerInfoSchema(name=name, healthy=healthy)
            )

    return TelemetryConsumersResponse(
        consumers=consumers_info,
        count=len(consumers_info),
    )


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


@router.post("/flush", response_model=TelemetryFlushResponse)
def flush_telemetry_buffer() -> dict:
    """Manually flush the telemetry event buffer."""
    _ensure_available()
    collector = get_telemetry()
    flushed = collector.flush()
    return TelemetryFlushResponse(flushed_events=flushed)


# ---------------------------------------------------------------------------
# History (S114)
# ---------------------------------------------------------------------------


def _ensure_history_available():
    """Raise 503 if telemetry history store is not available."""
    if not TELEMETRY_HISTORY_AVAILABLE or get_history_store is None:
        raise HTTPException(
            status_code=503,
            detail="Telemetry history store is not available",
        )


@router.get("/history", response_model=TelemetryHistoryResponse)
def get_event_history(
    limit: int = Query(default=50, ge=1, le=500, description="Page size"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    model: str = Query(default="", description="Filter by model name"),
) -> dict:
    """Get paginated telemetry event history from SQLite."""
    if not TELEMETRY_HISTORY_AVAILABLE or get_history_store is None:
        return TelemetryHistoryResponse()

    try:
        store = get_history_store()
        result = store.get_history(limit=limit, offset=offset, model=model)
        events = [HistoryEventSchema(**e) for e in result.get("events", [])]
        return TelemetryHistoryResponse(
            events=events,
            total=result.get("total", 0),
            limit=result.get("limit", limit),
            offset=result.get("offset", offset),
        )
    except Exception as exc:
        logger.warning("Failed to get telemetry history: %s", exc)
        return TelemetryHistoryResponse()


@router.get("/trends", response_model=TelemetryTrendsResponse)
def get_telemetry_trends(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history"),
    model: str = Query(default="", description="Filter by model name"),
) -> dict:
    """Get aggregated latency/throughput trends over time."""
    if not TELEMETRY_HISTORY_AVAILABLE or get_history_store is None:
        return TelemetryTrendsResponse()

    try:
        store = get_history_store()
        raw_buckets = store.get_trends(hours=hours, model=model)
        buckets = [TrendBucketSchema(**b) for b in raw_buckets]
        return TelemetryTrendsResponse(
            buckets=buckets,
            hours=hours,
            model=model,
        )
    except Exception as exc:
        logger.warning("Failed to get telemetry trends: %s", exc)
        return TelemetryTrendsResponse()


@router.get("/history/models")
def get_history_model_breakdown() -> dict:
    """Get per-model event count and average stats from history."""
    if not TELEMETRY_HISTORY_AVAILABLE or get_history_store is None:
        return {"models": []}

    try:
        store = get_history_store()
        raw = store.get_model_breakdown()
        return {"models": [ModelBreakdownSchema(**m).model_dump() for m in raw]}
    except Exception as exc:
        logger.warning("Failed to get model breakdown: %s", exc)
        return {"models": []}


@router.get("/history/stats", response_model=TelemetryHistoryStatsResponse)
def get_history_stats() -> dict:
    """Get quick overview stats for the history store."""
    if not TELEMETRY_HISTORY_AVAILABLE or get_history_store is None:
        return TelemetryHistoryStatsResponse()

    try:
        store = get_history_store()
        raw = store.get_stats()
        return TelemetryHistoryStatsResponse(**raw)
    except Exception as exc:
        logger.warning("Failed to get history stats: %s", exc)
        return TelemetryHistoryStatsResponse()


@router.delete("/history", response_model=TelemetryHistoryPurgeResponse)
def purge_event_history(
    older_than_days: int = Query(
        default=0,
        ge=0,
        description="Purge events older than N days (0 = purge all)",
    ),
) -> dict:
    """Purge old events from telemetry history.

    If older_than_days is 0, purges ALL events.
    """
    _ensure_history_available()
    store = get_history_store()

    if older_than_days == 0:
        count = store.purge_all()
    else:
        count = store.purge(older_than_days=older_than_days)

    return TelemetryHistoryPurgeResponse(purged_count=count)


# ---------------------------------------------------------------------------
# Settings & Export (S115)
# ---------------------------------------------------------------------------


@router.put("/history/settings", response_model=TelemetryHistorySettingsResponse)
def update_history_settings(body: TelemetryHistorySettingsRequest) -> dict:
    """Update telemetry history retention and auto-purge settings."""
    _ensure_history_available()
    store = get_history_store()

    result = store.update_settings(
        retention_days=body.retention_days,
        auto_purge_enabled=body.auto_purge_enabled,
    )
    return TelemetryHistorySettingsResponse(**result)


@router.get("/history/export")
def export_history_csv(
    model: str = Query(default="", description="Filter by model name"),
) -> PlainTextResponse:
    """Export telemetry event history as CSV file."""
    if not TELEMETRY_HISTORY_AVAILABLE or get_history_store is None:
        raise HTTPException(
            status_code=503,
            detail="Telemetry history store is not available",
        )

    try:
        store = get_history_store()
        csv_data = store.export_csv(model=model)
        return PlainTextResponse(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=telemetry_history.csv",
            },
        )
    except Exception as exc:
        logger.warning("Failed to export telemetry history: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

