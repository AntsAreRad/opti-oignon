#!/usr/bin/env python3
"""
API routes for Feedback & Analytics (S55).

Provides endpoints for submitting feedback, querying feedback stats,
viewing analytics overviews, and time-series performance trends.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from opti_oignon.feedback import (
        RATING_TYPE_STARS,  # noqa: F401
        RATING_TYPE_THUMBS,  # noqa: F401
        FeedbackEntry,
        FeedbackStats,  # noqa: F401
        feedback_store,
    )
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    feedback_store = None

try:
    from opti_oignon.analytics import (
        PerformanceRecord,
        analytics_engine,
        performance_tracker,
    )
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    analytics_engine = None
    performance_tracker = None

router = APIRouter(prefix="/api", tags=["feedback", "analytics"])


# =============================================================================
# SCHEMAS
# =============================================================================

class FeedbackSubmitRequest(BaseModel):
    """Request body for submitting feedback."""
    conversation_id: str = ""
    message_id: str = ""
    rating_type: str = Field(default="thumbs", description="'thumbs' or 'stars'")
    rating_value: int = Field(default=1, description="0/1 for thumbs, 1-5 for stars")
    feedback_text: str = Field(default="", max_length=2000)
    model_used: str = ""
    pipeline_used: str = ""
    task_type: str = ""


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    feedback_id: str
    status: str = "ok"


class FeedbackEntryResponse(BaseModel):
    """Single feedback entry in API responses."""
    feedback_id: str = ""
    conversation_id: str = ""
    message_id: str = ""
    rating_type: str = "thumbs"
    rating_value: int = 1
    feedback_text: str = ""
    model_used: str = ""
    pipeline_used: str = ""
    task_type: str = ""
    timestamp: float = 0.0


class FeedbackStatsResponse(BaseModel):
    """Aggregated feedback statistics."""
    total_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    average_score: float = 0.0
    thumbs_up: int = 0
    thumbs_down: int = 0
    star_distribution: dict[int, int] = Field(default_factory=dict)
    by_model: dict[str, Any] = Field(default_factory=dict)
    by_pipeline: dict[str, Any] = Field(default_factory=dict)
    by_task_type: dict[str, Any] = Field(default_factory=dict)


class AnalyticsOverviewResponse(BaseModel):
    """Performance analytics overview."""
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    success_rate: float = 1.0
    avg_response_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    total_tokens_processed: int = 0
    pipeline_distribution: dict[str, int] = Field(default_factory=dict)
    model_distribution: dict[str, int] = Field(default_factory=dict)
    task_type_distribution: dict[str, int] = Field(default_factory=dict)
    model_performance: dict[str, Any] = Field(default_factory=dict)
    pipeline_performance: dict[str, Any] = Field(default_factory=dict)


class TrendPointResponse(BaseModel):
    """Single trend data point."""
    window_start: float = 0.0
    window_end: float = 0.0
    count: int = 0
    avg_response_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    total_tokens: int = 0
    success_rate: float = 1.0


class TrendsResponse(BaseModel):
    """Time-series trends response."""
    window: str = "24h"
    buckets: int = 24
    model: str | None = None
    pipeline: str | None = None
    data: list[TrendPointResponse] = Field(default_factory=list)


class RoutingAccuracyResponse(BaseModel):
    """Routing accuracy comparison."""
    routed: dict[str, Any] = Field(default_factory=dict)
    unrouted: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# FEEDBACK ENDPOINTS
# =============================================================================

@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackSubmitRequest) -> dict:
    """Submit feedback for a message.

    Accepts thumbs up/down or 1-5 star ratings with optional text.
    """
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback module not available")

    if not feedback_store.enabled:
        raise HTTPException(status_code=403, detail="Feedback collection is disabled")

    entry = FeedbackEntry(
        conversation_id=request.conversation_id,
        message_id=request.message_id,
        rating_type=request.rating_type,
        rating_value=request.rating_value,
        feedback_text=request.feedback_text,
        model_used=request.model_used,
        pipeline_used=request.pipeline_used,
        task_type=request.task_type,
    )

    try:
        stored = feedback_store.add_feedback(entry)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return FeedbackResponse(feedback_id=stored.feedback_id)


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
def get_feedback_stats(
    since: float | None = Query(None, description="Only entries after this timestamp"),
) -> dict:
    """Get aggregated feedback statistics."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback module not available")

    stats = feedback_store.get_stats(since=since)
    return FeedbackStatsResponse(
        total_count=stats.total_count,
        positive_count=stats.positive_count,
        negative_count=stats.negative_count,
        average_score=stats.average_score,
        thumbs_up=stats.thumbs_up,
        thumbs_down=stats.thumbs_down,
        star_distribution=stats.star_distribution,
        by_model=stats.by_model,
        by_pipeline=stats.by_pipeline,
        by_task_type=stats.by_task_type,
    )


@router.get("/feedback/by-model/{model}", response_model=list[FeedbackEntryResponse])
def get_feedback_by_model(
    model: str,
    limit: int = Query(100, ge=1, le=1000),
) -> list:
    """Get feedback entries for a specific model."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback module not available")

    entries = feedback_store.list_by_model(model, limit=limit)
    return [
        FeedbackEntryResponse(**e.to_dict())
        for e in entries
    ]


@router.get("/feedback/by-pipeline/{pipeline}", response_model=list[FeedbackEntryResponse])
def get_feedback_by_pipeline(
    pipeline: str,
    limit: int = Query(100, ge=1, le=1000),
) -> list:
    """Get feedback entries for a specific pipeline."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback module not available")

    entries = feedback_store.list_by_pipeline(pipeline, limit=limit)
    return [
        FeedbackEntryResponse(**e.to_dict())
        for e in entries
    ]


@router.get("/feedback/list", response_model=list[FeedbackEntryResponse])
def list_feedback(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    since: float | None = Query(None),
    until: float | None = Query(None),
) -> list:
    """List feedback entries with pagination and time filters."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback module not available")

    entries = feedback_store.list_feedback(
        limit=limit, offset=offset, since=since, until=until
    )
    return [FeedbackEntryResponse(**e.to_dict()) for e in entries]


@router.get("/feedback/{feedback_id}", response_model=FeedbackEntryResponse)
def get_feedback(feedback_id: str) -> dict:
    """Get a single feedback entry by ID."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback module not available")

    entry = feedback_store.get_feedback(feedback_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Feedback not found")

    return FeedbackEntryResponse(**entry.to_dict())


@router.delete("/feedback/{feedback_id}")
def delete_feedback(feedback_id: str) -> dict:
    """Delete a feedback entry."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback module not available")

    deleted = feedback_store.delete_feedback(feedback_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Feedback not found")

    return {"status": "deleted", "feedback_id": feedback_id}


@router.get("/feedback/export/json")
def export_feedback_json(
    since: float | None = Query(None),
) -> Response:
    """Export all feedback as JSON."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback module not available")

    data = feedback_store.export_json(since=since)
    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=feedback_export.json"},
    )


@router.get("/feedback/export/csv")
def export_feedback_csv(
    since: float | None = Query(None),
) -> Response:
    """Export all feedback as CSV."""
    if not FEEDBACK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Feedback module not available")

    data = feedback_store.export_csv(since=since)
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=feedback_export.csv"},
    )


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@router.get("/analytics/overview", response_model=AnalyticsOverviewResponse)
def get_analytics_overview(
    since: float | None = Query(None, description="Only data after this timestamp"),
) -> dict:
    """Get performance analytics overview for the dashboard."""
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analytics module not available")

    overview = analytics_engine.get_overview(since=since)
    return AnalyticsOverviewResponse(**overview.to_dict())


@router.get("/analytics/trends", response_model=TrendsResponse)
def get_analytics_trends(
    window: str = Query("24h", description="Time window (e.g., '1h', '24h', '7d')"),
    buckets: int = Query(24, ge=1, le=168, description="Number of time buckets"),
    model: str | None = Query(None, description="Filter by model"),
    pipeline: str | None = Query(None, description="Filter by pipeline"),
) -> dict:
    """Get time-series performance trends."""
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analytics module not available")

    try:
        trend_data = analytics_engine.get_trends(
            window=window, buckets=buckets, model=model, pipeline=pipeline
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    points = [
        TrendPointResponse(
            window_start=t.window_start,
            window_end=t.window_end,
            count=t.count,
            avg_response_time_ms=t.avg_response_time_ms,
            avg_tokens_per_second=t.avg_tokens_per_second,
            total_tokens=t.total_tokens,
            success_rate=t.success_rate,
        )
        for t in trend_data
    ]

    return TrendsResponse(
        window=window,
        buckets=buckets,
        model=model,
        pipeline=pipeline,
        data=points,
    )


@router.get("/analytics/routing-accuracy", response_model=RoutingAccuracyResponse)
def get_routing_accuracy(
    since: float | None = Query(None),
) -> dict:
    """Get routing accuracy comparison (routed vs unrouted)."""
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analytics module not available")

    data = analytics_engine.get_routing_accuracy(since=since)
    return RoutingAccuracyResponse(**data)


@router.post("/analytics/record")
def record_performance(
    model_used: str = "",
    pipeline_used: str = "",
    task_type: str = "",
    response_time_ms: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    success: bool = True,
    conversation_id: str = "",
    message_id: str = "",
) -> dict:
    """Manually record a performance data point (for testing/integration)."""
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analytics module not available")

    record = PerformanceRecord(
        model_used=model_used,
        pipeline_used=pipeline_used,
        task_type=task_type,
        response_time_ms=response_time_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        success=success,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    stored = performance_tracker.record(record)
    return {"status": "ok", "record_id": stored.record_id}


@router.post("/analytics/cleanup")
def cleanup_analytics() -> dict:
    """Remove old analytics records beyond retention period."""
    if not ANALYTICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Analytics module not available")

    deleted = analytics_engine.cleanup_old_records()
    return {"status": "ok", "deleted": deleted}
