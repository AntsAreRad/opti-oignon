#!/usr/bin/env python3
"""
API routes for the Context Optimizer (S123).

Provides endpoints for optimizer configuration, priority presets,
optimization reports, and enhanced budget calculation with preset support.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .deps import (
    CONTEXT_OPTIMIZER_AVAILABLE,
    PROMPT_OPTIMIZATION_AVAILABLE,
    get_context_optimizer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/context/optimizer", tags=["context-optimizer"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class OptimizerConfigUpdate(BaseModel):
    """Request body for updating optimizer config."""
    enabled: bool | None = None
    active_preset: str | None = None
    priority_presets: dict[str, dict[str, float]] | None = None
    custom_ratios: dict[str, float] | None = None


class BudgetPresetRequest(BaseModel):
    """Request body for budget calculation with preset."""
    model: str
    preset: str | None = None
    custom_ratios: dict[str, float] | None = None
    project_active: bool = False
    context_window_override: int = 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/config")
def get_optimizer_config() -> dict[str, Any]:
    """Get current optimizer configuration and priority presets.

    Returns the full config including enabled state, active preset,
    all available presets, and their ratio definitions.
    """
    if not CONTEXT_OPTIMIZER_AVAILABLE or get_context_optimizer is None:
        return {
            "available": False,
            "enabled": False,
            "config": {},
        }

    optimizer = get_context_optimizer()
    if optimizer is None:
        return {
            "available": True,
            "enabled": False,
            "config": {},
            "message": "Optimizer not initialized",
        }

    return {
        "available": True,
        "enabled": optimizer.enabled,
        "active_preset": optimizer.active_preset,
        "priority_presets": optimizer.priority_presets,
        "config": optimizer.config,
    }


@router.put("/config")
def update_optimizer_config(body: OptimizerConfigUpdate) -> dict[str, Any]:
    """Update optimizer configuration.

    Supports updating enabled state, active preset, preset definitions,
    and custom ratio overrides.
    """
    if not CONTEXT_OPTIMIZER_AVAILABLE or get_context_optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Context optimizer module not available",
        )

    optimizer = get_context_optimizer()
    if optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Context optimizer not initialized",
        )

    updates: dict[str, Any] = {}
    if body.enabled is not None:
        updates["enabled"] = body.enabled
    if body.active_preset is not None:
        updates["active_preset"] = body.active_preset
    if body.priority_presets is not None:
        updates["priority_presets"] = body.priority_presets
    if body.custom_ratios is not None:
        updates["custom_ratios"] = body.custom_ratios

    try:
        updated = optimizer.update_config(updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "updated",
        "enabled": optimizer.enabled,
        "active_preset": optimizer.active_preset,
        "config": updated,
    }


@router.get("/report")
def get_optimization_report(
    last_n: int = Query(1, ge=1, le=50, description="Number of reports to return"),
) -> dict[str, Any]:
    """Get the last optimization report(s).

    Returns per-zone metrics showing budgeted vs actual tokens,
    strategies applied, and trim summaries.
    """
    if not CONTEXT_OPTIMIZER_AVAILABLE or get_context_optimizer is None:
        return {
            "available": False,
            "reports": [],
        }

    optimizer = get_context_optimizer()
    if optimizer is None:
        return {
            "available": True,
            "reports": [],
            "message": "Optimizer not initialized",
        }

    reports = optimizer.reports
    # Return most recent N
    selected = reports[-last_n:] if last_n < len(reports) else reports

    return {
        "available": True,
        "count": len(selected),
        "total_retained": len(reports),
        "reports": [r.as_dict() for r in selected],
    }


@router.get("/presets")
def list_presets() -> dict[str, Any]:
    """List all available priority presets with their ratio definitions.

    Returns preset names, their zone ratios, and which preset is active.
    """
    if not CONTEXT_OPTIMIZER_AVAILABLE or get_context_optimizer is None:
        # Return hardcoded defaults even when optimizer unavailable
        return {
            "available": False,
            "active_preset": "balanced",
            "presets": {
                "balanced": {
                    "system_ratio": 0.10,
                    "project_ratio": 0.25,
                    "history_ratio": 0.40,
                    "user_ratio": 0.10,
                    "reserve_ratio": 0.15,
                },
                "rag_heavy": {
                    "system_ratio": 0.10,
                    "project_ratio": 0.35,
                    "history_ratio": 0.30,
                    "user_ratio": 0.10,
                    "reserve_ratio": 0.15,
                },
                "history_heavy": {
                    "system_ratio": 0.10,
                    "project_ratio": 0.15,
                    "history_ratio": 0.50,
                    "user_ratio": 0.10,
                    "reserve_ratio": 0.15,
                },
            },
        }

    optimizer = get_context_optimizer()
    if optimizer is None:
        return {
            "available": True,
            "active_preset": "balanced",
            "presets": {},
            "message": "Optimizer not initialized",
        }

    return {
        "available": True,
        "active_preset": optimizer.active_preset,
        "presets": optimizer.priority_presets,
    }


@router.post("/budget")
def calculate_budget_with_preset(body: BudgetPresetRequest) -> dict[str, Any]:
    """Calculate token budget with optional preset or custom ratios.

    Extends the existing /api/context/budget/{model} endpoint with
    support for priority presets and custom ratio overrides.
    """
    if not PROMPT_OPTIMIZATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Prompt optimization module not available",
        )

    # Import here to avoid circular dependency at module load
    try:
        from opti_oignon.prompt_optimization import PromptTokenBudgetManager
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PromptTokenBudgetManager not importable",
        )

    # Resolve priority overrides from preset or custom ratios
    overrides: dict[str, float] | None = None
    preset_name = body.preset

    if body.custom_ratios:
        overrides = body.custom_ratios
        preset_name = "custom"
    elif body.preset and body.preset != "balanced":
        # Fetch preset ratios from optimizer if available
        if CONTEXT_OPTIMIZER_AVAILABLE and get_context_optimizer is not None:
            optimizer = get_context_optimizer()
            if optimizer is not None:
                presets = optimizer.priority_presets
                if body.preset in presets:
                    overrides = presets[body.preset]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown preset '{body.preset}'. "
                               f"Available: {list(presets.keys())}",
                    )

    # Calculate budget using deps singleton
    from .deps import prompt_budget_manager
    if prompt_budget_manager is None:
        raise HTTPException(
            status_code=503,
            detail="PromptTokenBudgetManager not initialized",
        )

    budget = prompt_budget_manager.calculate_budget(
        model=body.model,
        project_active=body.project_active,
        context_window_override=body.context_window_override,
        priority_overrides=overrides,
    )

    return {
        "model": body.model,
        "preset": preset_name or "balanced",
        "budget": budget.as_dict(),
    }
