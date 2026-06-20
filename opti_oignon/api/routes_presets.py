#!/usr/bin/env python3
"""
Routes API pour la gestion des presets.

Endpoints CRUD, search, matching par keywords, et duplication.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import PRESET_AVAILABLE, preset_manager
from .schemas import (
    PresetCreate,
    PresetDuplicateRequest,
    PresetInfo,
    PresetMatchResult,
    PresetUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/presets", tags=["presets"])


def _check_available():
    """Check that the presets module is available."""
    if not PRESET_AVAILABLE or preset_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Presets module not available",
        )


def _preset_to_info(preset) -> PresetInfo:
    """Convert a Preset object to PresetInfo schema."""
    return PresetInfo(
        id=preset.id,
        name=preset.name,
        description=preset.description,
        task=preset.task,
        model=preset.model,
        temperature=preset.temperature,
        prompt_variant=preset.prompt_variant,
        icon=preset.icon,
        tags=preset.tags or [],
        keywords=preset.keywords or [],
        detection_weight=preset.detection_weight,
        custom_prompt=preset.custom_prompt,
    )


@router.get("", response_model=list[PresetInfo])
def list_presets() -> list:
    """Liste tous les presets dans l'ordre configure."""
    _check_available()

    presets = preset_manager.get_ordered()
    return [_preset_to_info(p) for p in presets]


@router.get("/search", response_model=list[PresetInfo])
def search_presets(q: str = "") -> list:
    """Search de presets par nom, description, tags ou keywords."""
    _check_available()

    if not q.strip():
        return []

    results = preset_manager.search(q)
    return [_preset_to_info(p) for p in results]


@router.get("/match", response_model=list[PresetMatchResult])
def match_presets(text: str = "") -> list:
    """Trouve les presets correspondant a un texte via keywords."""
    _check_available()

    if not text.strip():
        return []

    results = preset_manager.find_by_keywords_with_scores(text)
    return [
        PresetMatchResult(
            preset=_preset_to_info(preset),
            score=score,
            matches=matches,
        )
        for preset, score, matches in results
    ]


@router.get("/{preset_id}", response_model=PresetInfo)
def get_preset(preset_id: str) -> dict:
    """Retrieve a preset by its ID."""
    _check_available()

    preset = preset_manager.get(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    return _preset_to_info(preset)


@router.post("", response_model=PresetInfo, status_code=201)
def create_preset(request: PresetCreate) -> dict:
    """Create a new preset."""
    _check_available()

    if not request.id.strip():
        raise HTTPException(status_code=422, detail="Preset ID cannot be empty")

    # Verifier si l'ID existe deja
    if preset_manager.get(request.id):
        raise HTTPException(status_code=409, detail="Preset ID already exists")

    preset = preset_manager.create(
        preset_id=request.id,
        name=request.name,
        task=request.task,
        model=request.model,
        temperature=request.temperature,
        prompt_variant=request.prompt_variant,
        description=request.description,
        icon=request.icon,
        tags=request.tags,
        keywords=request.keywords,
        detection_weight=request.detection_weight,
        custom_prompt=request.custom_prompt,
    )

    return _preset_to_info(preset)


@router.put("/{preset_id}", response_model=PresetInfo)
def update_preset(preset_id: str, request: PresetUpdate) -> dict:
    """Met a jour un preset existant."""
    _check_available()

    # Construire le dict des champs a mettre a jour (uniquement ceux fournis)
    update_fields = {}
    for field_name in request.model_fields:
        value = getattr(request, field_name)
        if value is not None:
            update_fields[field_name] = value

    if not update_fields:
        raise HTTPException(status_code=422, detail="No fields to update")

    preset = preset_manager.update(preset_id, **update_fields)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    return _preset_to_info(preset)


@router.delete("/{preset_id}")
def delete_preset(preset_id: str) -> dict:
    """Delete a preset."""
    _check_available()

    deleted = preset_manager.delete(preset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Preset not found or cannot be deleted")

    return {"deleted": True, "id": preset_id}


@router.post("/{preset_id}/duplicate", response_model=PresetInfo, status_code=201)
def duplicate_preset(preset_id: str, request: PresetDuplicateRequest) -> dict:
    """Duplicate an existing preset."""
    _check_available()

    if not request.new_id.strip():
        raise HTTPException(status_code=422, detail="New ID cannot be empty")

    # Verifier si le nouvel ID existe deja
    if preset_manager.get(request.new_id):
        raise HTTPException(status_code=409, detail="New preset ID already exists")

    new_preset = preset_manager.duplicate(preset_id, request.new_id, request.new_name)
    if not new_preset:
        raise HTTPException(status_code=404, detail="Source preset not found")

    return _preset_to_info(new_preset)
