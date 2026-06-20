#!/usr/bin/env python3
"""
Routes API for system presets (S84).

Infrastructure-level presets (Minimal/Balanced/Power) that configure
multiple YAML config files at once. Includes model detection,
preset recommendation, and onboarding state management.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import SYSTEM_PRESETS_AVAILABLE, system_presets_manager
from .schemas import (
    OnboardingStateResponse,
    SystemPresetApplyResponse,
    SystemPresetDetectResponse,
    SystemPresetInfo,
    SystemPresetListResponse,
    SystemPresetModelInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system-presets", tags=["system-presets"])


def _check_available():
    """Verify that the system presets module is available."""
    if not SYSTEM_PRESETS_AVAILABLE or system_presets_manager is None:
        raise HTTPException(
            status_code=503,
            detail="System presets module not available",
        )


@router.get("/list", response_model=SystemPresetListResponse)
def list_system_presets() -> dict:
    """List all available system presets with descriptions."""
    _check_available()

    presets = system_presets_manager.list_presets()
    return SystemPresetListResponse(
        presets=[
            SystemPresetInfo(
                id=p.id,
                name=p.name,
                description=p.description,
                icon=p.icon,
                recommended_vram_gb=p.recommended_vram_gb,
                recommended_ram_gb=p.recommended_ram_gb,
                model_strategy=p.model_strategy,
                pipelines=p.pipelines,
            )
            for p in presets
        ]
    )


@router.get("/detect", response_model=SystemPresetDetectResponse)
def detect_and_recommend() -> dict:
    """
    Auto-detect installed Ollama models and recommend a system preset.

    Scans the local Ollama installation, classifies models by size,
    and recommends the most appropriate preset.
    """
    _check_available()

    result = system_presets_manager.detect_and_recommend()
    return SystemPresetDetectResponse(
        models=[
            SystemPresetModelInfo(
                name=m.name,
                size_bytes=m.size_bytes,
                parameter_count_b=m.parameter_count_b,
                quantization=m.quantization,
                family=m.family,
                size_category=m.size_category,
            )
            for m in result.models
        ],
        recommended_preset=result.recommended_preset,
        reason=result.reason,
        model_counts=result.model_counts,
        total_estimated_vram_gb=result.total_estimated_vram_gb,
    )


@router.post("/apply/{preset_id}", response_model=SystemPresetApplyResponse)
def apply_system_preset(preset_id: str) -> dict:
    """
    Apply a system preset to all relevant configuration files.

    Creates backups of existing configs before applying changes.
    Updates the default model based on the preset's model strategy
    and the currently installed Ollama models.
    """
    _check_available()

    valid_ids = [p.id for p in system_presets_manager.list_presets()]
    if preset_id not in valid_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown system preset: {preset_id}. Valid: {valid_ids}",
        )

    result = system_presets_manager.apply_preset(preset_id)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return SystemPresetApplyResponse(
        applied=result.get("applied", False),
        preset_id=result.get("preset_id", ""),
        preset_name=result.get("preset_name", ""),
        selected_model=result.get("selected_model"),
        applied_configs=result.get("applied_configs", {}),
        pipelines=result.get("pipelines", []),
        warnings=result.get("warnings", []),
    )


@router.get("/onboarding", response_model=OnboardingStateResponse)
def get_onboarding_state() -> dict:
    """Get current onboarding state (whether user has initialized)."""
    _check_available()

    state = system_presets_manager.get_onboarding_state()
    return OnboardingStateResponse(
        user_initialized=state.get("user_initialized", False),
        applied_preset=state.get("applied_preset"),
        applied_at=state.get("applied_at"),
    )


@router.post("/onboarding/reset")
def reset_onboarding() -> dict:
    """Reset onboarding state to show the overlay again."""
    _check_available()

    system_presets_manager.reset_onboarding()
    return {"reset": True, "user_initialized": False}
