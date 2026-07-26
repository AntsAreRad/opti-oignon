#!/usr/bin/env python3
"""
API routes for Speculative Generation.

Provides endpoints for status, config update, and test speculative generation.
Mutually exclusive with cascading inference: enabling one disables the other.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import (
    CASCADING_AVAILABLE,
    SPECULATIVE_AVAILABLE,
    cascading_inference,
    speculative_generator,
)
from .schemas import (
    SpeculativeConfigUpdate,
    SpeculativeResultSchema,
    SpeculativeStatusResponse,
    SpeculativeTestRequest,
    SpeculativeTestResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/speculative", tags=["speculative"])


def _result_to_schema(result) -> SpeculativeResultSchema:
    """Convert a SpeculativeResult dataclass to a Pydantic schema."""
    return SpeculativeResultSchema(
        final_response=result.final_response,
        draft_response=result.draft_response,
        verify_response=result.verify_response,
        draft_model=result.draft_model,
        verify_model=result.verify_model,
        draft_accepted=result.draft_accepted,
        iterations=result.iterations,
        total_latency_ms=result.total_latency_ms,
        draft_latency_ms=result.draft_latency_ms,
        verify_latency_ms=result.verify_latency_ms,
        convergence_score=result.convergence_score,
    )


@router.get("/status", response_model=SpeculativeStatusResponse)
def get_speculative_status() -> dict:
    """Get speculative generation status, model pair, and last result summary."""
    if not SPECULATIVE_AVAILABLE or speculative_generator is None:
        return SpeculativeStatusResponse(
            enabled=False,
            available=False,
        )

    status = speculative_generator.get_status()

    return SpeculativeStatusResponse(
        enabled=status.get("enabled", False),
        available=True,
        draft_model=status.get("draft_model", ""),
        verify_model=status.get("verify_model", ""),
        max_iterations=status.get("max_iterations", 2),
        convergence_threshold=status.get("convergence_threshold", 0.85),
        last_result=status.get("last_result"),
        config=speculative_generator.get_config(),
    )


@router.put("/config", response_model=SpeculativeStatusResponse)
def update_speculative_config(update: SpeculativeConfigUpdate) -> dict:
    """Update speculative generation configuration.

    If enabling speculative, automatically disables cascading (mutual exclusion).
    """
    if not SPECULATIVE_AVAILABLE or speculative_generator is None:
        raise HTTPException(status_code=503, detail="Speculative generation not available")

    # Mutual exclusion: enabling speculative disables cascading
    if update.enabled is True and CASCADING_AVAILABLE and cascading_inference is not None:
        if cascading_inference.enabled:
            cascading_inference.enabled = False
            cascading_inference._save_config()
            logger.info("Disabled cascading inference (mutual exclusion with speculative)")

    speculative_generator.update_config(
        enabled=update.enabled,
        draft_model=update.draft_model,
        verify_model=update.verify_model,
        max_iterations=update.max_iterations,
        convergence_threshold=update.convergence_threshold,
        draft_max_tokens=update.draft_max_tokens,
        verify_max_tokens=update.verify_max_tokens,
        draft_temperature=update.draft_temperature,
        verify_temperature=update.verify_temperature,
    )

    return get_speculative_status()


@router.post("/test", response_model=SpeculativeTestResponse)
def test_speculative(request: SpeculativeTestRequest) -> dict:
    """Run a test speculative generation on a sample query."""
    if not SPECULATIVE_AVAILABLE or speculative_generator is None:
        raise HTTPException(status_code=503, detail="Speculative generation not available")

    if not speculative_generator.enabled:
        raise HTTPException(status_code=400, detail="Speculative generation is disabled")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = speculative_generator.generate(
            query=request.query,
            task_type=request.task_type,
        )
        return SpeculativeTestResponse(
            result=_result_to_schema(result),
            config=speculative_generator.get_config(),
        )
    except Exception as e:
        logger.error("Speculative test failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Speculative test failed: {e}")
