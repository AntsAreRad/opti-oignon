#!/usr/bin/env python3
"""
API routes for Smart Routing and Profile Management.

Provides endpoints for smart model selection, profile CRUD,
auto-detection, and router configuration.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .deps import (
    PROFILE_AVAILABLE,
    profile_manager,
)

# Conditional import of model health monitor
try:
    from opti_oignon.model_health import ModelHealthRecord, ModelStatus, model_health_monitor
    MODEL_HEALTH_AVAILABLE = True
except ImportError:
    MODEL_HEALTH_AVAILABLE = False
    model_health_monitor = None
    ModelHealthRecord = None
    ModelStatus = None

logger = logging.getLogger(__name__)

# Conditional import of smart router
try:
    from opti_oignon.smart_router import SmartRoutingResult, smart_router
    SMART_ROUTER_AVAILABLE = True
except ImportError:
    SMART_ROUTER_AVAILABLE = False
    smart_router = None
    SmartRoutingResult = None

# Conditional import of adaptive routing
try:
    from opti_oignon.adaptive_routing import feedback_routing_adapter
    ADAPTIVE_ROUTING_AVAILABLE = True
except ImportError:
    ADAPTIVE_ROUTING_AVAILABLE = False
    feedback_routing_adapter = None

router = APIRouter(prefix="/api/smart-routing", tags=["smart-routing"])


# =============================================================================
# SCHEMAS
# =============================================================================

class SmartRoutingResponse(BaseModel):
    """Response for smart model selection."""
    model: str
    score: float = 0.0
    task_score: float = 0.0
    speed_weight: float = 1.0
    context_fit: float = 1.0
    reason: str = ""
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    profile_used: bool = False
    fallback: bool = False
    feedback_adjusted: bool = False
    failover: bool = False
    original_model: str = ""


class PipelineRoutingResponse(BaseModel):
    """Response for pipeline-wide model selection."""
    selections: dict[str, SmartRoutingResponse] = Field(default_factory=dict)
    count: int = 0


class SmartRouterConfigResponse(BaseModel):
    """Smart router configuration."""
    enabled: bool = False
    profiles_available: bool = False
    operational: bool = False
    default_model: str = ""
    speed_preference: str = "balanced"
    speed_weights: dict[str, float] = Field(default_factory=dict)
    profile_count: int = 0


class SmartRouterConfigUpdate(BaseModel):
    """Request body for updating smart router config."""
    enabled: bool | None = None
    default_model: str | None = None
    speed_preference: str | None = None


class ProfileCreateRequest(BaseModel):
    """Request body for creating/updating a model profile."""
    display_name: str = ""
    capabilities: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    context_window: int = 32768
    speed_tier: str = "medium"
    quality_tier: str = "medium"
    recommended_for: list[str] = Field(default_factory=list)
    not_recommended_for: list[str] = Field(default_factory=list)
    task_scores: dict[str, float] = Field(default_factory=dict)


class TaskScoresUpdate(BaseModel):
    """Request body for updating task scores."""
    task_scores: dict[str, float]


# =============================================================================
# SMART ROUTING ENDPOINTS
# =============================================================================

@router.get("/select", response_model=SmartRoutingResponse)
def select_model_for_step(
    step_type: str = Query(..., description="Pipeline step type"),
    required_context: int | None = Query(None, description="Min context window"),
    prefer_speed: bool | None = Query(None, description="True=fast, False=quality"),
) -> dict:
    """Select the optimal model for a pipeline step type."""
    if not SMART_ROUTER_AVAILABLE or smart_router is None:
        raise HTTPException(status_code=503, detail="Smart router not available")

    result = smart_router.select_model(
        step_type=step_type,
        required_context=required_context,
        prefer_speed=prefer_speed,
    )
    return SmartRoutingResponse(**result.to_dict())


@router.post("/select-pipeline", response_model=PipelineRoutingResponse)
def select_models_for_pipeline(step_types: list[str]) -> dict:
    """Select optimal models for each step in a pipeline."""
    if not SMART_ROUTER_AVAILABLE or smart_router is None:
        raise HTTPException(status_code=503, detail="Smart router not available")

    results = smart_router.select_for_pipeline(step_types)
    selections = {
        st: SmartRoutingResponse(**res.to_dict())
        for st, res in results.items()
    }
    return PipelineRoutingResponse(selections=selections, count=len(selections))


# =============================================================================
# CONFIGURATION
# =============================================================================

@router.get("/config", response_model=SmartRouterConfigResponse)
def get_smart_router_config() -> dict:
    """Get current smart router configuration."""
    if not SMART_ROUTER_AVAILABLE or smart_router is None:
        return SmartRouterConfigResponse()

    config = smart_router.get_config()
    return SmartRouterConfigResponse(**{
        k: v for k, v in config.items()
        if k in SmartRouterConfigResponse.model_fields
    })


@router.put("/config")
def update_smart_router_config(body: SmartRouterConfigUpdate) -> dict:
    """Update smart router configuration."""
    if not SMART_ROUTER_AVAILABLE or smart_router is None:
        raise HTTPException(status_code=503, detail="Smart router not available")

    smart_router.configure(
        enabled=body.enabled,
        default_model=body.default_model,
        speed_preference=body.speed_preference,
    )
    return {"status": "ok", "config": smart_router.get_config()}


@router.post("/config/save")
def save_smart_router_config() -> dict:
    """Save smart router configuration to YAML file."""
    if not SMART_ROUTER_AVAILABLE or smart_router is None:
        raise HTTPException(status_code=503, detail="Smart router not available")

    saved = smart_router.save_config()
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save smart routing config")
    return {"status": "ok", "config": smart_router.get_config()}


# =============================================================================
# PROFILE CRUD
# =============================================================================

@router.put("/profiles/{model_name}")
def create_or_update_profile(model_name: str, body: ProfileCreateRequest) -> dict:
    """Create or update a model profile."""
    if not PROFILE_AVAILABLE or profile_manager is None:
        raise HTTPException(status_code=503, detail="Profile manager not available")

    try:
        from opti_oignon.model_profiles import ModelProfile
        profile = ModelProfile(
            name=model_name,
            display_name=body.display_name or model_name,
            capabilities=body.capabilities,
            strengths=body.strengths,
            weaknesses=body.weaknesses,
            context_window=body.context_window,
            speed_tier=body.speed_tier,
            quality_tier=body.quality_tier,
            recommended_for=body.recommended_for,
            not_recommended_for=body.not_recommended_for,
            task_scores=body.task_scores,
        )
        profile_manager.add_profile(profile)
        if SMART_ROUTER_AVAILABLE and smart_router:
            smart_router.clear_cache()
        return {"status": "ok", "profile": profile.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/profiles/{model_name}")
def delete_profile(model_name: str) -> dict:
    """Delete a model profile."""
    if not PROFILE_AVAILABLE or profile_manager is None:
        raise HTTPException(status_code=503, detail="Profile manager not available")

    removed = profile_manager.remove_profile(model_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Profile not found: {model_name}")

    if SMART_ROUTER_AVAILABLE and smart_router:
        smart_router.clear_cache()
    return {"status": "ok", "removed": model_name}


@router.put("/profiles/{model_name}/task-scores")
def update_task_scores(model_name: str, body: TaskScoresUpdate) -> dict:
    """Update task scores for a specific model."""
    if not PROFILE_AVAILABLE or profile_manager is None:
        raise HTTPException(status_code=503, detail="Profile manager not available")

    updated = profile_manager.update_task_scores(model_name, body.task_scores)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Profile not found: {model_name}")

    if SMART_ROUTER_AVAILABLE and smart_router:
        smart_router.clear_cache()

    profile = profile_manager.get_profile(model_name)
    return {"status": "ok", "task_scores": profile.task_scores if profile else {}}


@router.post("/profiles/{model_name}/auto-detect")
def auto_detect_model(model_name: str) -> dict:
    """Auto-detect model capabilities via ollama.show()."""
    if not PROFILE_AVAILABLE or profile_manager is None:
        raise HTTPException(status_code=503, detail="Profile manager not available")

    profile = profile_manager.auto_detect(model_name)
    if profile is None:
        raise HTTPException(
            status_code=500,
            detail=f"Auto-detection failed for {model_name}. Is Ollama running?"
        )

    if SMART_ROUTER_AVAILABLE and smart_router:
        smart_router.clear_cache()
    return {"status": "ok", "profile": profile.to_dict()}


@router.post("/profiles/save")
def save_profiles() -> dict:
    """Save all profiles to YAML file."""
    if not PROFILE_AVAILABLE or profile_manager is None:
        raise HTTPException(status_code=503, detail="Profile manager not available")

    saved = profile_manager.save()
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save profiles")
    return {"status": "ok", "count": profile_manager.count}


# =============================================================================
# FEEDBACK ADJUSTMENTS
# =============================================================================

@router.get("/feedback-adjustments")
def get_feedback_adjustments() -> dict:
    """Get current feedback-driven routing adjustments.

    Returns the full adaptive routing state, including all
    active and inactive score adjustments per model/task pair.
    """
    if not ADAPTIVE_ROUTING_AVAILABLE or feedback_routing_adapter is None:
        return {
            "enabled": False,
            "total_adjustments": 0,
            "active_adjustments": 0,
            "adjustments": [],
            "min_samples": 10,
            "max_adjustment": 0.15,
            "adjustment_factor": 0.05,
        }

    state = feedback_routing_adapter.get_all_adjustments()
    return state.to_dict()


@router.post("/feedback-adjustments/invalidate")
def invalidate_feedback_cache() -> dict:
    """Force recomputation of feedback-based routing adjustments.

    Clears the adapter cache so the next request recomputes
    adjustments from current feedback data.
    """
    if not ADAPTIVE_ROUTING_AVAILABLE or feedback_routing_adapter is None:
        raise HTTPException(status_code=503, detail="Adaptive routing not available")

    feedback_routing_adapter.invalidate_cache()
    # Also clear smart router cache since scores may change
    if SMART_ROUTER_AVAILABLE and smart_router:
        smart_router.clear_cache()

    return {"status": "ok", "message": "Feedback adjustment cache invalidated"}


# =============================================================================
# MODEL HEALTH ENDPOINTS
# =============================================================================

class ModelHealthResponse(BaseModel):
    """Response for a single model health record."""
    model: str
    status: str = "unknown"
    latency_ms: float = 0.0
    last_check: float = 0.0
    last_success: float = 0.0
    error_count: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    check_count: int = 0


class AllModelHealthResponse(BaseModel):
    """Response for all model health records."""
    records: dict[str, ModelHealthResponse] = Field(default_factory=dict)
    summary: dict[str, int] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


@router.get("/model-health", response_model=AllModelHealthResponse)
def get_all_model_health() -> dict:
    """Get health records for all tracked models."""
    if not MODEL_HEALTH_AVAILABLE or model_health_monitor is None:
        return AllModelHealthResponse(
            summary={"healthy": 0, "degraded": 0, "unavailable": 0},
            config={"enabled": False, "ollama_available": False},
        )

    state = model_health_monitor.to_dict()
    records_raw = state.get("records", {})
    records = {
        name: ModelHealthResponse(**data)
        for name, data in records_raw.items()
    }
    return AllModelHealthResponse(
        records=records,
        summary=state.get("summary", {}),
        config={
            k: v for k, v in state.items()
            if k not in ("records", "summary")
        },
    )


@router.get("/model-health/{model_name}", response_model=ModelHealthResponse)
def get_model_health(model_name: str) -> dict:
    """Get health record for a specific model."""
    if not MODEL_HEALTH_AVAILABLE or model_health_monitor is None:
        raise HTTPException(status_code=503, detail="Model health monitor not available")

    record = model_health_monitor.get_health(model_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No health record for model: {model_name}")

    return ModelHealthResponse(**record.to_dict())


@router.post("/model-health/check")
def force_health_check() -> dict:
    """Force an immediate health check on all tracked models.

    Returns updated health records after the check completes.
    """
    if not MODEL_HEALTH_AVAILABLE or model_health_monitor is None:
        raise HTTPException(status_code=503, detail="Model health monitor not available")

    results = model_health_monitor.check_all()
    records = {
        name: record.to_dict()
        for name, record in results.items()
    }
    return {
        "status": "ok",
        "checked": len(records),
        "records": records,
    }
