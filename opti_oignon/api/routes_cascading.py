#!/usr/bin/env python3
"""
API routes for Cascading Inference -- S69.

Provides endpoints for status, config update, and test cascade execution.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import CASCADING_AVAILABLE, cascading_inference
from .schemas import (
    CascadeConfigUpdate,
    CascadeResultSchema,
    CascadeStatusResponse,
    CascadeTierResultSchema,
    CascadeTierSchema,
    CascadeTestRequest,
    CascadeTestResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cascading", tags=["cascading"])


def _cascade_result_to_schema(result) -> CascadeResultSchema:
    """Convert a CascadeResult dataclass to a Pydantic schema."""
    attempts = []
    for a in result.attempts:
        attempts.append(CascadeTierResultSchema(
            tier_name=a.tier_name,
            model=a.model,
            response=a.response,
            score=a.score,
            latency_ms=a.latency_ms,
            escalation_reason=a.escalation_reason,
        ))
    return CascadeResultSchema(
        final_response=result.final_response,
        model_used=result.model_used,
        tier_index=result.tier_index,
        tier_name=result.tier_name,
        score=result.score,
        attempts=attempts,
        total_latency_ms=result.total_latency_ms,
        escalation_reasons=list(result.escalation_reasons),
    )


@router.get("/status", response_model=CascadeStatusResponse)
def get_cascading_status() -> dict:
    """Get cascading inference status, tier config, and last result summary."""
    if not CASCADING_AVAILABLE or cascading_inference is None:
        return CascadeStatusResponse(
            enabled=False,
            available=False,
        )

    status = cascading_inference.get_status()
    tiers = [
        CascadeTierSchema(**t) for t in status.get("tiers", [])
    ]

    return CascadeStatusResponse(
        enabled=status.get("enabled", False),
        available=True,
        tier_count=len(tiers),
        tiers=tiers,
        last_result=status.get("last_result"),
        config=cascading_inference.get_config(),
    )


@router.put("/config", response_model=CascadeStatusResponse)
def update_cascading_config(update: CascadeConfigUpdate) -> dict:
    """Update cascading inference configuration."""
    if not CASCADING_AVAILABLE or cascading_inference is None:
        raise HTTPException(status_code=503, detail="Cascading inference not available")

    cascading_inference.update_config(
        enabled=update.enabled,
        tiers=update.tiers,
        max_retries_per_tier=update.max_retries_per_tier,
        timeout_per_tier_seconds=update.timeout_per_tier_seconds,
        score_weights=update.score_weights,
    )

    # Return updated status
    return get_cascading_status()


@router.post("/test", response_model=CascadeTestResponse)
def test_cascade(request: CascadeTestRequest) -> dict:
    """Run a test cascade on a sample query and return the full result."""
    if not CASCADING_AVAILABLE or cascading_inference is None:
        raise HTTPException(status_code=503, detail="Cascading inference not available")

    if not cascading_inference.enabled:
        raise HTTPException(status_code=400, detail="Cascading inference is disabled")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = cascading_inference.cascade(
            query=request.query,
            task_type=request.task_type,
        )
        return CascadeTestResponse(
            result=_cascade_result_to_schema(result),
            config=cascading_inference.get_config(),
        )
    except Exception as e:
        logger.error("Cascade test failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Cascade test failed: {e}")
