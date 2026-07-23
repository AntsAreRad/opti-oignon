#!/usr/bin/env python3
"""
API routes for Model Lifecycle Management.

Provides endpoints for pulling, deleting, updating models through
the Ollama API, managing model aliases, and detecting stale models.
All operations are optional and degrade gracefully when Ollama or
the lifecycle module is unavailable.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam

from .deps import (
    MODEL_LIFECYCLE_AVAILABLE,
    get_lifecycle_manager,
)
from .schemas import (
    ModelAliasesResponse,
    ModelAliasRequest,
    ModelDeleteResponse,
    ModelLifecycleStatusResponse,
    ModelPullJobSchema,
    ModelPullRequest,
    ModelUpdateCheckRequest,
    ModelUpdateInfoSchema,
    ModelUpdatesResponse,
    StaleModelsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/model-lifecycle", tags=["model-lifecycle"])


def _ensure_available():
    """Raise 503 if model lifecycle manager is not available."""
    if not MODEL_LIFECYCLE_AVAILABLE or get_lifecycle_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model lifecycle management is not available",
        )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@router.get("/status", response_model=ModelLifecycleStatusResponse)
def get_lifecycle_status() -> dict:
    """Get model lifecycle manager status."""
    if not MODEL_LIFECYCLE_AVAILABLE or get_lifecycle_manager is None:
        return ModelLifecycleStatusResponse(available=False, enabled=False)

    try:
        mgr = get_lifecycle_manager()
        return ModelLifecycleStatusResponse(
            available=True,
            enabled=mgr.enabled,
            ollama_base_url=mgr.config.ollama_base_url,
            max_concurrent_pulls=mgr.config.max_concurrent_pulls,
            active_pulls=mgr._active_pulls,
            alias_count=len(mgr.list_aliases()),
            stale_threshold_days=mgr.config.stale_threshold_days,
        )
    except Exception as exc:
        logger.warning("Failed to get lifecycle status: %s", exc)
        return ModelLifecycleStatusResponse(available=False, enabled=False)


# ---------------------------------------------------------------------------
# Pull operations
# ---------------------------------------------------------------------------


@router.post("/pull", response_model=ModelPullJobSchema)
def start_model_pull(body: ModelPullRequest) -> dict:
    """Start pulling a model in the background."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    if not mgr.enabled:
        raise HTTPException(status_code=400, detail="Model lifecycle is disabled")

    job = mgr.start_pull(body.model_name)
    return ModelPullJobSchema(**job.to_dict())


@router.get("/pull-progress/{job_id}", response_model=ModelPullJobSchema)
def get_pull_progress(
    job_id: str = PathParam(..., description="Pull job ID"),
) -> dict:
    """Get pull progress for a specific job."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    job = mgr.get_pull_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Pull job {job_id} not found")
    return ModelPullJobSchema(**job.to_dict())


@router.post("/pull-cancel/{job_id}")
def cancel_model_pull(
    job_id: str = PathParam(..., description="Pull job ID"),
) -> dict:
    """Cancel an active pull job."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    success = mgr.cancel_pull(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job {job_id} (not found or already finished)",
        )
    return {"success": True, "job_id": job_id}


@router.get("/pull-jobs")
def list_pull_jobs() -> dict:
    """List all pull jobs (active and completed)."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    jobs = mgr.list_pull_jobs()
    return {"jobs": jobs, "count": len(jobs)}


# ---------------------------------------------------------------------------
# Delete operations
# ---------------------------------------------------------------------------


@router.delete("/models/{model_name:path}", response_model=ModelDeleteResponse)
def delete_model(
    model_name: str = PathParam(..., description="Model name to delete"),
) -> dict:
    """Delete a locally stored model."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    if not mgr.enabled:
        raise HTTPException(status_code=400, detail="Model lifecycle is disabled")

    result = mgr.delete_model(model_name)
    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Delete failed"),
        )
    return ModelDeleteResponse(**result)


# ---------------------------------------------------------------------------
# Update check
# ---------------------------------------------------------------------------


@router.post("/update-check", response_model=ModelUpdatesResponse)
def check_model_updates(body: ModelUpdateCheckRequest) -> dict:
    """Check for newer versions of specified models."""
    _ensure_available()
    mgr = get_lifecycle_manager()

    if not body.model_names:
        # Check all locally installed models.
        models = mgr.list_models()
        names = [m["name"] for m in models if m.get("name")]
    else:
        names = body.model_names

    results = mgr.check_updates_batch(names)
    return ModelUpdatesResponse(
        results=[ModelUpdateInfoSchema(**r.to_dict()) for r in results],
    )


# ---------------------------------------------------------------------------
# Alias management
# ---------------------------------------------------------------------------


@router.get("/aliases", response_model=ModelAliasesResponse)
def list_aliases() -> dict:
    """List all model aliases."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    return ModelAliasesResponse(aliases=mgr.list_aliases())


@router.post("/aliases")
def set_alias(body: ModelAliasRequest) -> dict:
    """Create or update a model alias."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    success = mgr.set_alias(body.alias, body.model_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save alias")
    return {"success": True, "alias": body.alias, "model_name": body.model_name}


@router.delete("/aliases/{alias}")
def remove_alias(
    alias: str = PathParam(..., description="Alias to remove"),
) -> dict:
    """Remove a model alias."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    success = mgr.remove_alias(alias)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alias '{alias}' not found")
    return {"success": True, "alias": alias}


# ---------------------------------------------------------------------------
# Stale model detection
# ---------------------------------------------------------------------------


@router.get("/stale", response_model=StaleModelsResponse)
def detect_stale_models() -> dict:
    """Detect models that haven't been modified recently."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    stale = mgr.detect_stale_models()
    return StaleModelsResponse(
        models=stale,
        threshold_days=mgr.config.stale_threshold_days,
    )


# ---------------------------------------------------------------------------
# Model info (extended, through lifecycle manager)
# ---------------------------------------------------------------------------


@router.get("/models/{model_name:path}")
def get_model_detail(
    model_name: str = PathParam(..., description="Model name"),
) -> dict:
    """Get detailed information about a specific model."""
    _ensure_available()
    mgr = get_lifecycle_manager()
    info = mgr.get_model_info(model_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return info
