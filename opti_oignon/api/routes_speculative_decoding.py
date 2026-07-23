#!/usr/bin/env python3
"""
API routes for Speculative Decoding.

Provides endpoints for status, config update, compatible draft model
listing, VRAM budget estimation, and acceptance rate history.
Speculative decoding is only available with the llama.cpp backend.
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from .deps import (
    INFERENCE_BACKEND_AVAILABLE,
    SPECULATIVE_DECODING_AVAILABLE,
    get_backend_registry,
    get_speculative_decoding_manager,
)
from .schemas import (
    CompatibleDraftsResponse,
    DraftCandidateSchema,
    SpeculativeDecodingConfigSchema,
    SpeculativeDecodingConfigUpdate,
    SpeculativeDecodingStatsSchema,
    SpeculativeDecodingStatusResponse,
    VRAMBudgetResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/speculative-decoding",
    tags=["speculative-decoding"],
)


def _get_manager():
    """Get the speculative decoding manager or raise 503."""
    if (
        not SPECULATIVE_DECODING_AVAILABLE
        or get_speculative_decoding_manager is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Speculative decoding module not available",
        )
    return get_speculative_decoding_manager()


def _get_llama_cpp_models() -> list[dict]:
    """Retrieve available models from the llama.cpp backend."""
    if not INFERENCE_BACKEND_AVAILABLE or get_backend_registry is None:
        return []
    try:
        registry = get_backend_registry()
        backend = registry.get_backend("llama_cpp")
        if backend is None:
            return []
        models = backend.list_models()
        result = []
        for m in models:
            info = m.to_dict() if hasattr(m, "to_dict") else {}
            result.append({
                "name": getattr(m, "name", "") or info.get("name", ""),
                "family": getattr(m, "family", "") or info.get("family", ""),
                "parameter_size_b": _extract_param_size(m),
                "quantization": (
                    getattr(m, "quantization_level", "")
                    or info.get("quantization_level", "")
                ),
                "path": getattr(m, "path", "") or info.get("path", ""),
            })
        return result
    except Exception as exc:
        logger.debug("Failed to list llama.cpp models: %s", exc)
        return []


def _extract_param_size(model_info) -> float:
    """Extract parameter size in billions from a BackendModelInfo."""
    raw = getattr(model_info, "parameter_size", None)
    if raw is None and hasattr(model_info, "to_dict"):
        raw = model_info.to_dict().get("parameter_size")
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        import re
        match = re.match(r"^([\d.]+)\s*[Bb]?$", raw.strip())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return 0.0


@router.get("/status", response_model=SpeculativeDecodingStatusResponse)
def get_speculative_decoding_status() -> dict:
    """Get speculative decoding status, config, and acceptance stats."""
    if (
        not SPECULATIVE_DECODING_AVAILABLE
        or get_speculative_decoding_manager is None
    ):
        return SpeculativeDecodingStatusResponse(
            available=False,
            backend_required="llama_cpp",
        )

    mgr = get_speculative_decoding_manager()
    status = mgr.get_status()

    return SpeculativeDecodingStatusResponse(
        config=SpeculativeDecodingConfigSchema(**status["config"]),
        stats=SpeculativeDecodingStatsSchema(**status["stats"]),
        available=status["available"],
        backend_required=status["backend_required"],
    )


@router.put("/config", response_model=SpeculativeDecodingConfigSchema)
def update_speculative_decoding_config(body: SpeculativeDecodingConfigUpdate) -> dict:
    """Enable/disable speculative decoding, select draft model, etc."""
    mgr = _get_manager()

    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(
            status_code=400,
            detail="No fields provided for update",
        )

    try:
        new_cfg = mgr.update_config(updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return SpeculativeDecodingConfigSchema(**new_cfg.to_dict())


@router.get("/compatible-drafts", response_model=CompatibleDraftsResponse)
def get_compatible_drafts(
    main_model: str = Query(
        ..., description="Main model name or family to find drafts for"
    ),
    main_family: str = Query(
        "", description="Main model family (if known)"
    ),
    main_params_b: float = Query(
        0.0, description="Main model parameter count in billions"
    ),
    main_quant: str = Query(
        "Q4_K_M", description="Main model quantization level"
    ),
) -> dict:
    """List compatible draft models for a given main model."""
    mgr = _get_manager()

    available_models = _get_llama_cpp_models()
    selector = mgr.get_draft_selector()

    family = main_family or main_model
    params_b = main_params_b if main_params_b > 0 else 70.0

    drafts = selector.find_compatible_drafts(
        family, params_b, main_quant, available_models,
    )

    return CompatibleDraftsResponse(
        main_model=main_model,
        drafts=[DraftCandidateSchema(**d.to_dict()) for d in drafts],
        count=len(drafts),
    )


@router.get("/vram-budget", response_model=VRAMBudgetResponse)
def check_vram_budget(
    main_params_b: float = Query(
        ..., description="Main model parameter count in billions"
    ),
    main_quant: str = Query(
        "Q4_K_M", description="Main model quantization level"
    ),
    draft_params_b: float = Query(
        ..., description="Draft model parameter count in billions"
    ),
    draft_quant: str = Query(
        "Q4_K_M", description="Draft model quantization level"
    ),
) -> dict:
    """Check if main + draft models fit within VRAM budget."""
    mgr = _get_manager()
    calc = mgr.get_vram_calculator()

    result = calc.check_fit(
        main_params_b, main_quant,
        draft_params_b, draft_quant,
    )

    return VRAMBudgetResponse(**result)


@router.post("/reset-stats")
def reset_speculative_stats() -> dict:
    """Clear acceptance rate statistics."""
    mgr = _get_manager()
    mgr.reset_stats()
    return {"status": "ok", "message": "Speculative decoding stats cleared"}


@router.get("/acceptance-history")
def get_acceptance_history(
    last_n: int = Query(
        default=50,
        ge=0,
        le=200,
        description="Number of recent records to return (0 = all)",
    ),
) -> dict:
    """Get per-request acceptance rate history (S111).

    Returns a list of AcceptanceRecord dicts, oldest first, with
    per-request draft/accepted counts, acceptance rates, and speedups.
    """
    mgr = _get_manager()
    history = mgr.get_acceptance_history(last_n=last_n)
    rolling = mgr.get_rolling_acceptance_rate(window=10)
    return {
        "history": history,
        "count": len(history),
        "rolling_acceptance_rate": round(rolling, 4),
    }
