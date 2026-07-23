#!/usr/bin/env python3
"""
API routes for Inference Auto-Tuner.

Provides endpoints to start/cancel tuning sessions, retrieve results,
apply tuned parameters, and check tuner status. Tuning runs on demand
only (user-initiated), never automatically.

POST /api/tuner/run now detects the active inference backend
(Ollama or llama.cpp) and creates a real benchmark function that
measures actual token generation speed. Falls back to mock if no
backend is available (testing/CI).
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam

from .deps import (
    AUTO_TUNER_AVAILABLE,
    INFERENCE_BACKEND_AVAILABLE,
    get_auto_tuner_manager,
    get_backend_registry,
)
from .schemas import (
    ParameterSpaceSchema,
    TunerConfigSchema,
    TunerJobSchema,
    TunerProfileSchema,
    TunerResultsResponse,
    TunerRunRequest,
    TunerStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/tuner",
    tags=["tuner"],
)


def _get_manager():
    """Get the auto-tuner manager or raise 503."""
    if not AUTO_TUNER_AVAILABLE or get_auto_tuner_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Auto-tuner module not available",
        )
    return get_auto_tuner_manager()


@router.get("/status", response_model=TunerStatusResponse)
def get_tuner_status() -> dict:
    """Get auto-tuner status, config, and active jobs."""
    if not AUTO_TUNER_AVAILABLE or get_auto_tuner_manager is None:
        return TunerStatusResponse(available=False)

    mgr = get_auto_tuner_manager()
    status = mgr.get_status()

    return TunerStatusResponse(
        config=TunerConfigSchema(**status["config"]),
        param_space=ParameterSpaceSchema(**status["param_space"]),
        active_jobs=status["active_jobs"],
        saved_profiles=status["saved_profiles"],
        available=status["available"],
    )


@router.post("/run", response_model=TunerJobSchema)
def start_tuning(body: TunerRunRequest) -> dict:
    """Start a tuning session for a model.

    S111: Detects the active inference backend (Ollama or llama.cpp)
    and creates a real benchmark function that measures actual token
    generation speed. Falls back to a mock if no backend is available
    (testing/CI mode). Returns a job object that can be polled for
    progress.
    """
    mgr = _get_manager()

    if not body.model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    benchmark_fn = _resolve_benchmark_fn(body.model_name)

    try:
        job = mgr.start_tuning(
            model_name=body.model_name,
            benchmark_fn=benchmark_fn,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return TunerJobSchema(**job.to_dict())


def _resolve_benchmark_fn(model_name: str):
    """Detect the active backend and return the appropriate benchmark function.

    Priority order:
        1. Ollama (if healthy and model is available)
        2. llama.cpp (if healthy and model is available)
        3. Mock fallback (always works)

    Returns:
        A callable(params: dict) -> BenchmarkResult.
    """
    from opti_oignon.auto_tuner import (
        create_llamacpp_benchmark_fn,
        create_mock_benchmark_fn,
        create_ollama_benchmark_fn,
    )

    # Try Ollama first.
    if INFERENCE_BACKEND_AVAILABLE and get_backend_registry is not None:
        try:
            registry = get_backend_registry()

            # Check Ollama backend.
            ollama_backend = registry.get_backend("ollama")
            if ollama_backend is not None and ollama_backend.health_check():
                logger.info(
                    "Tuner: using real Ollama benchmark for %s", model_name
                )
                return create_ollama_benchmark_fn(model_name)

            # Check llama.cpp backend.
            llamacpp_backend = registry.get_backend("llama_cpp")
            if llamacpp_backend is not None and llamacpp_backend.health_check():
                logger.info(
                    "Tuner: using real llama.cpp benchmark for %s",
                    model_name,
                )
                return create_llamacpp_benchmark_fn(
                    model_name, backend=llamacpp_backend
                )
        except Exception as exc:
            logger.debug("Backend detection failed, falling back to mock: %s", exc)

    # Fallback: mock benchmark (safe for CI / no-backend environments).
    logger.info("Tuner: no real backend available, using mock benchmark")
    return create_mock_benchmark_fn()


@router.get("/results", response_model=TunerResultsResponse)
def list_tuner_results() -> dict:
    """List all tuning results (per model)."""
    mgr = _get_manager()
    results = mgr.list_results()
    return TunerResultsResponse(
        results=results,
        count=len(results),
    )


@router.get("/results/{model_name}", response_model=TunerProfileSchema)
def get_tuner_result(
    model_name: str = PathParam(..., description="Model name"),
) -> dict:
    """Get best config for a specific model."""
    mgr = _get_manager()
    profile = mgr.get_result(model_name)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No tuning results for model: {model_name}",
        )
    return TunerProfileSchema(**profile.to_dict())


@router.post("/apply/{model_name}")
def apply_tuner_result(
    model_name: str = PathParam(..., description="Model name"),
) -> dict:
    """Apply tuned parameters as defaults for a model.

    Returns the best params dict that should be applied to the model's
    inference configuration.
    """
    mgr = _get_manager()
    params = mgr.apply_result(model_name)
    if params is None:
        raise HTTPException(
            status_code=404,
            detail=f"No tuning results to apply for model: {model_name}",
        )
    return {
        "status": "ok",
        "model_name": model_name,
        "applied_params": params,
        "message": f"Optimal parameters retrieved for {model_name}",
    }


@router.delete("/results/{model_name}")
def delete_tuner_result(
    model_name: str = PathParam(..., description="Model name"),
) -> dict:
    """Clear tuning data for a model."""
    mgr = _get_manager()
    deleted = mgr.delete_result(model_name)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No tuning results for model: {model_name}",
        )
    return {
        "status": "ok",
        "model_name": model_name,
        "message": f"Tuning results cleared for {model_name}",
    }


@router.post("/cancel/{model_name}")
def cancel_tuning(
    model_name: str = PathParam(..., description="Model name"),
) -> dict:
    """Cancel an active tuning session."""
    mgr = _get_manager()
    cancelled = mgr.cancel_tuning(model_name)
    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"No active tuning session for model: {model_name}",
        )
    return {
        "status": "ok",
        "model_name": model_name,
        "message": f"Tuning cancelled for {model_name}",
    }


@router.get("/job/{model_name}", response_model=TunerJobSchema)
def get_tuner_job(
    model_name: str = PathParam(..., description="Model name"),
) -> dict:
    """Get the current/last tuner job for a model."""
    mgr = _get_manager()
    job = mgr.get_job(model_name)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No tuner job for model: {model_name}",
        )
    return TunerJobSchema(**job.to_dict())


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@router.get("/recommendations/{model_name}")
def get_tuner_recommendations(
    model_name: str = PathParam(..., description="Model name"),
) -> dict:
    """Generate optimization recommendations from tuning results.

    Analyzes the tuning profile for a model and returns actionable
    recommendations sorted by estimated speedup.
    """
    from opti_oignon.auto_tuner import generate_recommendations

    mgr = _get_manager()
    profile = mgr.get_result(model_name)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"No tuning results for model: {model_name}",
        )

    recs = generate_recommendations(profile)
    return {
        "model_name": model_name,
        "recommendations": [r.to_dict() for r in recs],
        "count": len(recs),
    }
