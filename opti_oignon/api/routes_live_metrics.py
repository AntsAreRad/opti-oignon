#!/usr/bin/env python3
"""
API routes for Live Performance Metrics.

Provides real-time inference metrics: current snapshot, recent history,
collector status, and a WebSocket stream for live frontend updates.

Endpoints:
    GET  /api/metrics/live    -- current live metrics snapshot
    GET  /api/metrics/history -- last N seconds of metrics history
    GET  /api/metrics/status  -- collector status and config
    WS   /api/metrics/stream  -- real-time metrics push (500ms updates)
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from .deps import (
    LIVE_METRICS_AVAILABLE,
    get_live_metrics,
)
from .schemas import (
    LiveMetricsConfigSchema,
    LiveMetricsHistoryResponse,
    LiveMetricsSampleSchema,
    LiveMetricsStatusResponse,
)

logger = logging.getLogger(__name__)

# RFC 6455 WebSocket close code for graceful server-side shutdown.
WS_CLOSE_INTERNAL_ERROR = 1011

router = APIRouter(
    prefix="/api/metrics",
    tags=["metrics"],
)


def _get_collector():
    """Get the live metrics collector or raise 503."""
    if not LIVE_METRICS_AVAILABLE or get_live_metrics is None:
        raise HTTPException(
            status_code=503,
            detail="Live metrics module not available",
        )
    return get_live_metrics()


@router.get("/live", response_model=LiveMetricsSampleSchema)
def get_live_snapshot() -> dict:
    """Get the current live metrics snapshot.

    Returns the most recent sampled values including tokens/sec,
    latency, GPU utilization, and memory usage.
    """
    collector = _get_collector()
    sample = collector.current_snapshot()
    return LiveMetricsSampleSchema(**sample.to_dict())


@router.get("/history", response_model=LiveMetricsHistoryResponse)
def get_metrics_history(
    seconds: int = Query(
        default=60,
        ge=1,
        le=3600,
        description="Number of seconds of history to return",
    ),
) -> dict:
    """Get recent metrics history.

    Returns a list of metrics samples from the rolling buffer,
    filtered to the last N seconds. Samples are ordered oldest-first.
    """
    collector = _get_collector()
    samples = collector.get_history(seconds=seconds)
    return LiveMetricsHistoryResponse(
        samples=samples,
        count=len(samples),
    )


@router.get("/status", response_model=LiveMetricsStatusResponse)
def get_metrics_status() -> dict:
    """Get live metrics collector status and configuration."""
    if not LIVE_METRICS_AVAILABLE or get_live_metrics is None:
        return LiveMetricsStatusResponse(available=False)

    collector = get_live_metrics()
    status = collector.get_status()

    return LiveMetricsStatusResponse(
        running=status["running"],
        config=LiveMetricsConfigSchema(**status["config"]),
        gpu_available=status["gpu_available"],
        history_size=status["history_size"],
        total_tokens_all_time=status["total_tokens_all_time"],
        is_generating=status["is_generating"],
        active_model=status["active_model"],
        available=True,
    )


@router.websocket("/stream")
async def metrics_stream_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time metrics streaming.

    Authenticates before processing.
    """
    await websocket.accept()

    # Audit fix: authenticate WebSocket connection
    try:
        from .routes_auth import authenticate_websocket
        user = await authenticate_websocket(websocket)
        if user is None:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Authentication required"},
            })
            await websocket.close(code=4001)
            return
    except Exception:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Authentication failed"},
        })
        await websocket.close(code=4001)
        return

    if not LIVE_METRICS_AVAILABLE or get_live_metrics is None:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Live metrics not available"},
        })
        await websocket.close()
        return

    collector = get_live_metrics()
    interval = collector.config.sample_interval_ms / 1000.0
    heartbeat_interval = 30.0
    last_heartbeat = asyncio.get_event_loop().time()

    try:
        while True:
            # Take a snapshot and send it.
            sample = collector.current_snapshot()
            await websocket.send_json({
                "type": "metrics",
                "data": sample.to_dict(),
            })

            # Wait for the sampling interval, but also check for
            # incoming messages (e.g. close requests).
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=interval,
                )
                if isinstance(msg, dict) and msg.get("type") == "close":
                    break
            except asyncio.TimeoutError:
                pass

            # Periodic heartbeat (in case client needs keep-alive).
            now = asyncio.get_event_loop().time()
            if now - last_heartbeat >= heartbeat_interval:
                await websocket.send_json({"type": "heartbeat"})
                last_heartbeat = now

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("Metrics WebSocket error: %s", exc)
        try:
            await websocket.close(code=WS_CLOSE_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
