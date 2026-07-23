#!/usr/bin/env python3
"""
API routes for Humanizer.

Provides endpoints for text humanization, config management,
A/B feedback submission, and feedback analytics.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import HUMANIZER_AVAILABLE, humanizer_engine
from .schemas import (
    HumanizerConfigResponse,
    HumanizerConfigUpdate,
    HumanizerFeedbackRequest,
    HumanizerFeedbackResponse,
    HumanizerRewriteRequest,
    HumanizerRewriteResponse,
    HumanizerStatsResponse,
    HumanizerStrategyStats,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/humanizer", tags=["humanizer"])


@router.post("/rewrite", response_model=HumanizerRewriteResponse)
def rewrite_text(request: HumanizerRewriteRequest) -> dict:
    """Humanize a text passage using configured strategies."""
    if not HUMANIZER_AVAILABLE or humanizer_engine is None:
        raise HTTPException(status_code=503, detail="Humanizer not available")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = humanizer_engine.humanize(
        text=request.text,
        model=request.model,
        mode=request.mode,
        intensity=request.intensity,
        formality=request.formality,
    )

    return HumanizerRewriteResponse(
        original=result.original,
        humanized=result.humanized,
        strategies_applied=result.strategies_applied,
        replacements_count=result.replacements_count,
        rewrite_model=result.rewrite_model,
        latency_ms=result.latency_ms,
        mode=result.mode,
        intensity=result.intensity,
        comparison_id=result.comparison_id,
    )


@router.get("/config", response_model=HumanizerConfigResponse)
def get_humanizer_config() -> dict:
    """Get current humanizer configuration."""
    if not HUMANIZER_AVAILABLE or humanizer_engine is None:
        return HumanizerConfigResponse(
            enabled=False,
            available=False,
        )

    config = humanizer_engine.get_config()
    return HumanizerConfigResponse(
        enabled=config.get("enabled", False),
        available=True,
        mode=config.get("mode", "rewrite"),
        intensity=config.get("intensity", "moderate"),
        formality=config.get("formality", "neutral"),
        rewrite_model=config.get("rewrite_model"),
        max_input_length=config.get("max_input_length", 8000),
        banned_phrases=config.get("banned_phrases", []),
        vocabulary_replacements=config.get("vocabulary_replacements", {}),
    )


@router.post("/config", response_model=HumanizerConfigResponse)
def update_humanizer_config(update: HumanizerConfigUpdate) -> dict:
    """Update humanizer configuration."""
    if not HUMANIZER_AVAILABLE or humanizer_engine is None:
        raise HTTPException(status_code=503, detail="Humanizer not available")

    kwargs = {}
    if update.enabled is not None:
        kwargs["enabled"] = update.enabled
    if update.mode is not None:
        kwargs["mode"] = update.mode
    if update.intensity is not None:
        kwargs["intensity"] = update.intensity
    if update.formality is not None:
        kwargs["formality"] = update.formality
    if update.rewrite_model is not None:
        kwargs["rewrite_model"] = update.rewrite_model
    if update.max_input_length is not None:
        kwargs["max_input_length"] = update.max_input_length
    if update.banned_phrases is not None:
        kwargs["banned_phrases"] = update.banned_phrases
    if update.vocabulary_replacements is not None:
        kwargs["vocabulary_replacements"] = update.vocabulary_replacements

    humanizer_engine.update_config(**kwargs)
    return get_humanizer_config()


@router.post("/feedback", response_model=HumanizerFeedbackResponse)
def submit_feedback(request: HumanizerFeedbackRequest) -> dict:
    """Submit an A/B comparison rating."""
    if not HUMANIZER_AVAILABLE or humanizer_engine is None:
        raise HTTPException(status_code=503, detail="Humanizer not available")

    if request.winner not in ("humanized", "original", "tie"):
        raise HTTPException(
            status_code=400,
            detail="Winner must be 'humanized', 'original', or 'tie'",
        )

    success = humanizer_engine.submit_feedback(
        comparison_id=request.comparison_id,
        winner=request.winner,
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Comparison '{request.comparison_id}' not found",
        )

    return HumanizerFeedbackResponse(
        success=True,
        comparison_id=request.comparison_id,
        winner=request.winner,
    )


@router.get("/stats", response_model=HumanizerStatsResponse)
def get_humanizer_stats() -> dict:
    """Get aggregated feedback statistics."""
    if not HUMANIZER_AVAILABLE or humanizer_engine is None:
        return HumanizerStatsResponse()

    stats = humanizer_engine.get_stats()
    by_strategy = {
        k: HumanizerStrategyStats(**v) for k, v in stats.by_strategy.items()
    }
    by_model = {
        k: HumanizerStrategyStats(**v) for k, v in stats.by_model.items()
    }
    by_intensity = {
        k: HumanizerStrategyStats(**v) for k, v in stats.by_intensity.items()
    }

    return HumanizerStatsResponse(
        total_ratings=stats.total_ratings,
        humanized_wins=stats.humanized_wins,
        original_wins=stats.original_wins,
        ties=stats.ties,
        win_rate=stats.win_rate,
        by_strategy=by_strategy,
        by_model=by_model,
        by_intensity=by_intensity,
    )
