#!/usr/bin/env python3
"""
Vision configuration API routes (S94).

GET  /api/vision/config          -- Current vision config + effective model
PUT  /api/vision/config          -- Update vision model selection
GET  /api/vision/models          -- List vision-capable models
POST /api/vision/clear-cache     -- Clear capability probe cache
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])


# -- Schemas --

class VisionConfigResponse(BaseModel):
    """Vision configuration with resolved effective model."""
    vision_model: str = Field(description="Configured model or 'auto'")
    effective_model: str | None = Field(description="Resolved model name (None if unavailable)")
    detection_strategy: str = Field(description="Detection strategy: capabilities, patterns, or both")
    auto_detect_patterns: list[str] = Field(description="Patterns for name-based detection")
    vision_families: list[str] = Field(description="Capability families indicating vision support")
    known_vision_models: list[str] = Field(description="Manually declared vision-capable models")
    describe_prompt: str = Field(description="Default image description prompt")
    available_vision_models: list[str] = Field(description="All detected vision models")


class VisionConfigUpdate(BaseModel):
    """Request body for updating vision config."""
    vision_model: str | None = Field(default=None, description="Model name or 'auto'")
    describe_prompt: str | None = Field(default=None, description="Image description prompt")
    known_vision_models: list[str] | None = Field(default=None, description="Manual vision model list")


class VisionModelInfo(BaseModel):
    """Info about a vision-capable model."""
    name: str
    is_selected: bool = False
    detection_method: str = Field(description="How this model was detected: capability, pattern, or manual")


# -- Helpers --

def _get_available_model_names() -> list[str]:
    """Get list of all available Ollama model names."""
    try:
        from opti_oignon.api.deps import get_ollama_models
        models = get_ollama_models()
        names = []
        for m in models:
            if hasattr(m, "model"):
                names.append(m.model)
            elif hasattr(m, "name"):
                names.append(m.name)
            elif isinstance(m, dict):
                names.append(m.get("model", m.get("name", "")))
        return [n for n in names if n]
    except Exception as exc:
        logger.debug("Could not list Ollama models: %s", exc)
        return []


def _get_vision_config():
    """Get the vision config singleton, raising 503 if unavailable."""
    try:
        from opti_oignon.api.deps import vision_config, VISION_CONFIG_AVAILABLE
        if not VISION_CONFIG_AVAILABLE or vision_config is None:
            raise HTTPException(status_code=503, detail="Vision config not available")
        return vision_config
    except ImportError:
        raise HTTPException(status_code=503, detail="Vision config module not available")


def _get_detection_method(vc, model_name: str) -> str:
    """Determine how a model was detected as vision-capable."""
    if vc._is_known_vision(model_name):
        return "manual"
    if vc._probe_model_capabilities(model_name):
        return "capability"
    if vc._detect_by_patterns(model_name):
        return "pattern"
    return "unknown"


# -- Endpoints --

@router.get("/config", response_model=VisionConfigResponse)
async def get_vision_config() -> dict:
    """Return current vision configuration with resolved effective model."""
    vc = _get_vision_config()
    all_models = _get_available_model_names()
    vision_models = vc.detect_vision_models(all_models)
    effective = vc.get_effective_model(all_models)

    return VisionConfigResponse(
        vision_model=vc.vision_model,
        effective_model=effective,
        detection_strategy=vc.detection_strategy,
        auto_detect_patterns=vc.auto_detect_patterns,
        vision_families=vc.vision_families,
        known_vision_models=vc.known_vision_models,
        describe_prompt=vc.describe_prompt,
        available_vision_models=vision_models,
    )


@router.put("/config", response_model=VisionConfigResponse)
async def update_vision_config(body: VisionConfigUpdate) -> dict:
    """Update vision model selection, describe prompt, or known models."""
    vc = _get_vision_config()

    if body.vision_model is not None:
        vc.vision_model = body.vision_model

    if body.describe_prompt is not None:
        vc.describe_prompt = body.describe_prompt

    if body.known_vision_models is not None:
        vc.known_vision_models = body.known_vision_models

    all_models = _get_available_model_names()
    vision_models = vc.detect_vision_models(all_models)
    effective = vc.get_effective_model(all_models)

    return VisionConfigResponse(
        vision_model=vc.vision_model,
        effective_model=effective,
        detection_strategy=vc.detection_strategy,
        auto_detect_patterns=vc.auto_detect_patterns,
        vision_families=vc.vision_families,
        known_vision_models=vc.known_vision_models,
        describe_prompt=vc.describe_prompt,
        available_vision_models=vision_models,
    )


@router.get("/models", response_model=list[VisionModelInfo])
async def list_vision_models() -> list:
    """List all detected vision-capable models with detection method."""
    vc = _get_vision_config()
    all_models = _get_available_model_names()
    effective = vc.get_effective_model(all_models)
    vision_models = vc.detect_vision_models(all_models)

    return [
        VisionModelInfo(
            name=m,
            is_selected=(m == effective),
            detection_method=_get_detection_method(vc, m),
        )
        for m in vision_models
    ]


@router.post("/clear-cache")
async def clear_vision_cache() -> dict:
    """Clear the capability probe cache to force re-detection."""
    vc = _get_vision_config()
    vc.clear_cache()
    return {"status": "ok", "message": "Vision capability cache cleared"}
