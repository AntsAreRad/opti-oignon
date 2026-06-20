#!/usr/bin/env python3
"""
API Routes for Inference Backend Management — S105
===================================================

Endpoints for listing backends, switching active backend,
browsing GGUF models, downloading models, and querying
model metadata.

ROUTE ORDERING: Specific paths (/models/all, /gguf/*) are
registered BEFORE the /{name} parametric route to prevent
the catch-all from swallowing them.
"""

import logging

from fastapi import APIRouter, HTTPException

from .schemas import (
    BackendActivateResponse,
    BackendListResponse,
    BackendModelsResponse,
    BackendStatusResponse,
    GGUFDownloadRequest,
    GGUFDownloadResponse,
    GGUFModelInfoResponse,
    GGUFModelListResponse,
    GGUFStorageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backends", tags=["backends"])

# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

try:
    from ..inference_backend import get_backend_registry
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    get_backend_registry = None

try:
    from ..model_manager import get_model_manager
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    get_model_manager = None


# ---------------------------------------------------------------------------
# Fixed-path endpoints (MUST be registered before /{name} catch-all)
# ---------------------------------------------------------------------------

@router.get("", response_model=BackendListResponse)
def list_backends() -> dict:
    """List all registered inference backends with their status."""
    if not BACKEND_AVAILABLE:
        return BackendListResponse(backends=[], active_backend=None).model_dump()

    registry = get_backend_registry()
    backends_data = registry.list_backends()

    backends = [BackendStatusResponse(**b) for b in backends_data]

    return BackendListResponse(
        backends=backends,
        active_backend=registry.active_name,
    ).model_dump()


@router.get("/models/all", response_model=BackendModelsResponse)
def list_all_models() -> dict:
    """List models from all healthy backends."""
    if not BACKEND_AVAILABLE:
        return BackendModelsResponse(models=[], count=0).model_dump()

    registry = get_backend_registry()
    models = registry.all_models()
    model_dicts = [m.to_dict() for m in models]

    return BackendModelsResponse(
        models=model_dicts,
        count=len(model_dicts),
    ).model_dump()


@router.get("/gguf/models", response_model=GGUFModelListResponse)
def list_gguf_models() -> dict:
    """List all GGUF model files from configured directories."""
    if not MODEL_MANAGER_AVAILABLE:
        return GGUFModelListResponse(models=[], count=0).model_dump()

    manager = get_model_manager()
    models = manager.scan_models()

    model_responses = [GGUFModelInfoResponse(**m) for m in models]

    return GGUFModelListResponse(
        models=model_responses,
        count=len(model_responses),
    ).model_dump()


@router.get("/gguf/models/{filename}/info", response_model=GGUFModelInfoResponse)
def get_gguf_model_info(filename: str) -> dict:
    """Get detailed metadata for a specific GGUF model file."""
    if not MODEL_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Model manager not available",
        )

    manager = get_model_manager()
    info = manager.get_model_info(filename)

    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"GGUF model not found: {filename}",
        )

    return GGUFModelInfoResponse(**info).model_dump()


@router.post("/gguf/download", response_model=GGUFDownloadResponse)
def download_gguf_model(request: GGUFDownloadRequest) -> dict:
    """Download a GGUF model from a direct URL."""
    if not MODEL_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Model manager not available",
        )

    manager = get_model_manager()

    try:
        result = manager.download_model(
            url=request.url,
            filename=request.filename,
            target_dir=request.target_dir,
        )

        return GGUFDownloadResponse(
            status=result.get("status", "error"),
            path=result.get("path", ""),
            filename=result.get("filename", ""),
            size=result.get("size", 0),
            size_human=result.get("size_human", ""),
            message=result.get("message", ""),
        ).model_dump()

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("GGUF download error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: {exc}",
        )


@router.get("/gguf/storage", response_model=GGUFStorageResponse)
def get_gguf_storage() -> dict:
    """Get storage usage for GGUF model directories."""
    if not MODEL_MANAGER_AVAILABLE:
        return GGUFStorageResponse(
            total_size=0,
            total_size_human="0B",
            model_count=0,
            directories=[],
        ).model_dump()

    manager = get_model_manager()
    usage = manager.get_storage_usage()

    return GGUFStorageResponse(**usage).model_dump()


# ---------------------------------------------------------------------------
# Parametric endpoints (/{name} AFTER all fixed paths)
# ---------------------------------------------------------------------------

@router.get("/{name}", response_model=BackendStatusResponse)
def get_backend(name: str) -> dict:
    """Get detailed status of a specific backend."""
    if not BACKEND_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Inference backend system not available",
        )

    registry = get_backend_registry()
    backend = registry.get(name)

    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=f"Backend not found: {name}",
        )

    healthy = False
    model_count = 0
    try:
        healthy = backend.health_check()
        if healthy:
            model_count = len(backend.list_models())
    except Exception:
        pass

    return BackendStatusResponse(
        name=backend.name,
        display_name=backend.display_name,
        healthy=healthy,
        active=backend.name == registry.active_name,
        model_count=model_count,
    ).model_dump()


@router.post("/{name}/activate", response_model=BackendActivateResponse)
def activate_backend(name: str) -> dict:
    """Switch the active inference backend."""
    if not BACKEND_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Inference backend system not available",
        )

    registry = get_backend_registry()

    if registry.get(name) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Backend not found: {name}",
        )

    success = registry.activate(name)

    return BackendActivateResponse(
        success=success,
        active_backend=registry.active_name or "",
        message=f"Backend '{name}' activated" if success else f"Failed to activate '{name}'",
    ).model_dump()
