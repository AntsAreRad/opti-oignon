#!/usr/bin/env python3
"""
API routes for Performance Dashboard -- Step 2.

Provides endpoints for real-time performance metrics, latency stats,
drift detection, optimization recommendations, and metric history.
"""

import logging

from fastapi import APIRouter, Query

from .deps import (
    PERFORMANCE_MONITOR_AVAILABLE,
    performance_monitor,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/summary")
def get_performance_summary() -> dict:
    """Get complete performance summary: throughput, latency, utilization."""
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        return {"available": False, "error": "Performance monitor not available"}

    try:
        summary = performance_monitor.get_summary()
        return {"available": True, **summary}
    except Exception as e:
        logger.error("Failed to get performance summary: %s", e)
        return {"available": False, "error": str(e)}


@router.get("/latency")
def get_latency_stats(
    model: str | None = Query(None, description="Filter by model name"),
    window: int = Query(300, description="Rolling window in seconds"),
) -> dict:
    """Get latency statistics (p50/p95/p99) for a model or all models."""
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        return {"available": False, "error": "Performance monitor not available"}

    try:
        stats = performance_monitor.get_latency_stats(
            model=model, window_seconds=window
        )
        return {
            "available": True,
            "model": model,
            "window_seconds": window,
            "p50": stats.p50,
            "p95": stats.p95,
            "p99": stats.p99,
            "mean": stats.mean,
            "count": stats.count,
        }
    except Exception as e:
        logger.error("Failed to get latency stats: %s", e)
        return {"available": False, "error": str(e)}


@router.get("/drift")
def get_drift_results() -> dict:
    """Get drift detection results for all active models."""
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        return {"available": False, "error": "Performance monitor not available", "drifts": []}

    try:
        drifts = performance_monitor.detect_all_drift()
        return {
            "available": True,
            "drifts": [
                {
                    "model": d.model,
                    "metric": d.metric,
                    "baseline_value": d.baseline_value,
                    "recent_value": d.recent_value,
                    "change_ratio": d.change_ratio,
                    "is_drifted": d.is_drifted,
                    "direction": d.direction,
                }
                for d in drifts
            ],
        }
    except Exception as e:
        logger.error("Failed to detect drift: %s", e)
        return {"available": False, "error": str(e), "drifts": []}


@router.get("/recommendations")
def get_recommendations() -> dict:
    """Get optimization recommendations based on current metrics."""
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        return {"available": False, "error": "Performance monitor not available", "recommendations": []}

    try:
        recs = performance_monitor.get_recommendations()
        return {
            "available": True,
            "recommendations": [
                {
                    "model": r.model,
                    "metric": r.metric,
                    "message": r.message,
                    "severity": r.severity,
                    "value": r.value,
                }
                for r in recs
            ],
        }
    except Exception as e:
        logger.error("Failed to get recommendations: %s", e)
        return {"available": False, "error": str(e), "recommendations": []}


@router.get("/history")
def get_performance_history(
    model: str | None = Query(None, description="Filter by model name"),
    hours: int = Query(24, description="How many hours of history"),
    limit: int = Query(500, description="Max records to return"),
) -> dict:
    """Get raw metric history records."""
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        return {"available": False, "error": "Performance monitor not available", "records": []}

    try:
        records = performance_monitor.get_history(
            model=model, hours=hours, limit=limit
        )
        return {
            "available": True,
            "model": model,
            "hours": hours,
            "count": len(records),
            "records": records,
        }
    except Exception as e:
        logger.error("Failed to get performance history: %s", e)
        return {"available": False, "error": str(e), "records": []}


@router.get("/throughput")
def get_throughput(
    window: int = Query(300, description="Rolling window in seconds"),
) -> dict:
    """Get token throughput over a rolling window."""
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        return {"available": False, "error": "Performance monitor not available"}

    try:
        tp = performance_monitor.get_throughput(window_seconds=window)
        return {"available": True, **tp}
    except Exception as e:
        logger.error("Failed to get throughput: %s", e)
        return {"available": False, "error": str(e)}


@router.get("/utilization")
def get_utilization(
    window: int = Query(3600, description="Rolling window in seconds"),
) -> dict:
    """Get model utilization distribution."""
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        return {"available": False, "error": "Performance monitor not available", "models": {}}

    try:
        util = performance_monitor.get_model_utilization(window_seconds=window)
        return {"available": True, "window_seconds": window, "models": util}
    except Exception as e:
        logger.error("Failed to get utilization: %s", e)
        return {"available": False, "error": str(e), "models": {}}


@router.post("/cleanup")
def cleanup_old_metrics() -> dict:
    """Delete metrics older than retention period."""
    if not PERFORMANCE_MONITOR_AVAILABLE or performance_monitor is None:
        return {"available": False, "error": "Performance monitor not available"}

    try:
        deleted = performance_monitor.cleanup_old_records()
        return {"available": True, "deleted": deleted}
    except Exception as e:
        logger.error("Failed to cleanup metrics: %s", e)
        return {"available": False, "error": str(e)}
