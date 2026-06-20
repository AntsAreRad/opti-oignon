#!/usr/bin/env python3
"""
API routes for the Inference Profiler -- S113.

Provides per-request profiling summary and recent profile
listing.  Degrades gracefully when the profiler module is
unavailable.
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from .deps import (
    INFERENCE_PROFILER_AVAILABLE,
    get_profiler,
)
from .schemas import (
    InferenceProfileSchema,
    ProfilerRecentResponse,
    ProfilerSummaryResponse,
    ProfilerSummarySchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/profiler", tags=["profiler"])


def _ensure_available():
    """Raise 503 if inference profiler is not available."""
    if not INFERENCE_PROFILER_AVAILABLE or get_profiler is None:
        raise HTTPException(
            status_code=503,
            detail="Inference profiler is not available",
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@router.get("/summary", response_model=ProfilerSummaryResponse)
def get_profiler_summary() -> dict:
    """Get aggregated profiling stats per model."""
    if not INFERENCE_PROFILER_AVAILABLE or get_profiler is None:
        return ProfilerSummaryResponse()

    try:
        profiler = get_profiler()
        raw_summaries = profiler.get_summary()
        models = [ProfilerSummarySchema(**s) for s in raw_summaries]
        return ProfilerSummaryResponse(
            models=models,
            total_profiled_requests=profiler.total_profiled,
        )
    except Exception as exc:
        logger.warning("Failed to get profiler summary: %s", exc)
        return ProfilerSummaryResponse()


# ---------------------------------------------------------------------------
# Recent profiles
# ---------------------------------------------------------------------------


@router.get("/recent", response_model=ProfilerRecentResponse)
def get_recent_profiles(
    n: int = Query(default=20, ge=1, le=100, description="Number of recent profiles"),
) -> dict:
    """Get the most recent N inference profiles."""
    _ensure_available()
    profiler = get_profiler()
    raw = profiler.get_recent(n)
    profiles = [InferenceProfileSchema(**p) for p in raw]
    return ProfilerRecentResponse(
        profiles=profiles,
        count=len(profiles),
    )
