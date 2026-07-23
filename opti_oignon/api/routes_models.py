#!/usr/bin/env python3
"""
API routes for Ollama models.

Exposes the list of available models and effective model
resolution (presets, routing).
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from .deps import (
    ANALYZER_AVAILABLE,
    PRESET_AVAILABLE,
    PROFILE_AVAILABLE,
    ROUTER_AVAILABLE,
    analyzer,
    get_ollama_models,
    preset_manager,
    profile_manager,
)
from .deps import (
    router as model_router,
)
from .schemas import (
    EffectiveModelResponse,
    ModelInfo,
    ModelListResponse,
    ModelProfileInfo,
    ModelProfilesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


def _format_size(size_bytes) -> str | None:
    """Formate une taille en bytes en representation lisible."""
    if not size_bytes:
        return None
    try:
        size = int(size_bytes)
        if size >= 1_000_000_000:
            return f"{size / 1_000_000_000:.1f}GB"
        elif size >= 1_000_000:
            return f"{size / 1_000_000:.1f}MB"
        return f"{size}B"
    except (ValueError, TypeError):
        return str(size_bytes)


# ---------------------------------------------------------------------------
# MTP Detection — Multi-Token Prediction capability check
# ---------------------------------------------------------------------------

# Known MTP-capable model families/names. This list will grow as more
# models adopt MTP. Detection is by model name/family prefix matching.
_MTP_CAPABLE_PATTERNS: list[str] = [
    "deepseek-v3",
    "deepseek-r1",
    "deepseek-v2.5",
    "qwen3",
    "glm-4",
    "glm4",
]


def _detect_mtp_support(model_name: str, family: str | None = None) -> bool:
    """Detect if a model supports Multi-Token Prediction.

    Checks model name and family against known MTP-capable patterns.
    Detection only — MTP execution is not yet supported (llama.cpp MTP
    support still in development).

    Args:
        model_name: Full model name (e.g. "deepseek-v3:latest").
        family: Model family string if available.

    Returns:
        True if the model is known to support MTP.
    """
    name_lower = model_name.lower().strip()
    family_lower = (family or "").lower().strip()

    for pattern in _MTP_CAPABLE_PATTERNS:
        if pattern in name_lower or pattern in family_lower:
            return True

    return False


def _extract_model_info(model_data) -> ModelInfo:
    """Extract model info from an ollama.list() response entry."""
    # ollama-python >= 0.4: Model objects with .model attribute (not .name)
    # ollama-python < 0.4 or dict: key "name" or "model"
    if isinstance(model_data, dict):
        name = model_data.get("name", model_data.get("model", "unknown"))
        size = model_data.get("size")
        modified = model_data.get("modified_at")
        details = model_data.get("details", {})
        family = details.get("family") if isinstance(details, dict) else None
        param_size = details.get("parameter_size") if isinstance(details, dict) else None
        quant = details.get("quantization_level") if isinstance(details, dict) else None
    else:
        # Model object: .model contains the name (not .name in recent versions)
        name = getattr(model_data, "model", None) or getattr(model_data, "name", "unknown")
        size = getattr(model_data, "size", None)
        modified = getattr(model_data, "modified_at", None)
        details = getattr(model_data, "details", None)
        # details is an object with attributes, not a dict
        family = getattr(details, "family", None) if details else None
        param_size = getattr(details, "parameter_size", None) if details else None
        quant = getattr(details, "quantization_level", None) if details else None

    # Convert modified_at to string if it's a datetime object
    if modified and not isinstance(modified, str):
        try:
            modified = str(modified)
        except Exception:
            modified = None

    return ModelInfo(
        name=str(name),
        size=_format_size(size),
        modified_at=modified,
        family=family,
        parameter_size=param_size,
        quantization_level=quant,
        mtp_capable=_detect_mtp_support(str(name), family),
    )


@router.get("", response_model=ModelListResponse)
def list_models() -> dict:
    """List available Ollama models."""
    raw_models = get_ollama_models()
    models = [_extract_model_info(m) for m in raw_models]

    return ModelListResponse(
        models=models,
        count=len(models),
    ).model_dump()


@router.get("/effective", response_model=EffectiveModelResponse)
def get_effective_model(
    question: str = Query("", description="Question utilisateur pour le routing"),
    preset: str | None = Query(None, description="ID du preset a utiliser"),
    force_model: str | None = Query(None, description="Modele force"),
) -> dict:
    """Resolve the effective model by priority: forced > preset > auto-routing."""

    # 1. Modele force
    if force_model:
        return EffectiveModelResponse(model=force_model, source="forced")

    # 2. Preset explicite
    if preset and PRESET_AVAILABLE and preset_manager:
        try:
            p = preset_manager.get(preset)
            if p and p.model:
                return EffectiveModelResponse(model=p.model, source="preset")
        except Exception as e:
            logger.debug(f"Preset resolution error {preset}: {e}")

    # 3. Auto-detect by preset keywords
    if question.strip() and PRESET_AVAILABLE and preset_manager:
        try:
            detected = preset_manager.find_by_keywords(question)
            if detected and detected.model:
                return EffectiveModelResponse(model=detected.model, source="auto_preset")
        except Exception as e:
            logger.debug(f"Preset auto-detection error: {e}")

    # 4. Auto-routing via analyzer + router
    if question.strip() and ANALYZER_AVAILABLE and ROUTER_AVAILABLE:
        try:
            analysis = analyzer.analyze(question)
            routing = model_router.route(analysis)
            return EffectiveModelResponse(model=routing.model, source="auto_router")
        except Exception as e:
            logger.debug(f"Auto-routing error: {e}")

    # Fallback: no model determined
    return EffectiveModelResponse(model="", source="none")


# Model profiles endpoint
@router.get("/profiles", response_model=ModelProfilesResponse)
def list_model_profiles() -> dict:
    """List model profiles loaded from YAML configuration."""
    if not PROFILE_AVAILABLE or profile_manager is None:
        return ModelProfilesResponse(profiles={}, count=0).model_dump()

    data = profile_manager.to_dict()
    profiles = {}
    for name, pdata in data.get("profiles", {}).items():
        profiles[name] = ModelProfileInfo(**pdata)

    return ModelProfilesResponse(
        profiles=profiles,
        count=len(profiles),
    ).model_dump()


@router.get("/profiles/{model_name}")
def get_model_profile(model_name: str) -> dict:
    """Return the profile for a specific model."""
    if not PROFILE_AVAILABLE or profile_manager is None:
        raise HTTPException(status_code=404, detail="Model profiles not available")

    profile = profile_manager.get_profile(model_name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile for model: {model_name}")

    return ModelProfileInfo(**profile.to_dict()).model_dump()
