#!/usr/bin/env python3
"""
Routes API pour la gestion des pipelines multi-agents.

Endpoints CRUD, duplication, agents/templates disponibles,
matching par keywords, statistiques et export YAML.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import PIPELINE_AVAILABLE, pipeline_manager
from .schemas import (
    PipelineCreate,
    PipelineDuplicateRequest,
    PipelineExportRequest,
    PipelineInfo,
    PipelineStats,
    PipelineStepSchema,
    PipelineUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipelines", tags=["pipelines"])


def _check_available():
    """Check that the pipelines module is available."""
    if not PIPELINE_AVAILABLE or pipeline_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Pipelines module not available",
        )


def _pipeline_to_info(pipeline) -> PipelineInfo:
    """Convert a Pipeline object to PipelineInfo schema."""
    steps = [
        PipelineStepSchema(
            name=s.name,
            agent=s.agent,
            prompt_template=s.prompt_template,
            description=s.description,
            system_prompt=s.system_prompt,
            model=s.model,
        )
        for s in pipeline.steps
    ]
    return PipelineInfo(
        id=pipeline.id,
        name=pipeline.name,
        description=pipeline.description,
        pattern=pipeline.pattern,
        emoji=pipeline.emoji,
        steps=steps,
        keywords=pipeline.keywords or [],
        detection_weight=pipeline.detection_weight,
        created_at=pipeline.created_at,
        is_builtin=pipeline.is_builtin,
        step_count=pipeline.step_count,
    )


@router.get("", response_model=list[PipelineInfo])
def list_pipelines(builtin_only: bool = False, custom_only: bool = False) -> list:
    """Liste tous les pipelines (builtin + custom)."""
    _check_available()

    if builtin_only:
        pipelines = pipeline_manager.list_builtin()
    elif custom_only:
        pipelines = pipeline_manager.list_custom()
    else:
        pipelines = pipeline_manager.list_all()

    return [_pipeline_to_info(p) for p in pipelines]


@router.get("/agents", response_model=list[str])
def list_agents() -> dict:
    """Liste les agents disponibles."""
    _check_available()
    return pipeline_manager.get_available_agents()


@router.get("/templates", response_model=list[str])
def list_templates() -> dict:
    """Liste les templates disponibles."""
    _check_available()
    return pipeline_manager.get_available_templates()


@router.get("/match")
def match_pipelines(text: str = "") -> dict:
    """Trouve le meilleur pipeline correspondant a un texte."""
    _check_available()

    if not text.strip():
        return {"match": None}

    pipeline = pipeline_manager.find_by_keywords(text)
    if not pipeline:
        return {"match": None}

    return {"match": _pipeline_to_info(pipeline)}


@router.get("/stats", response_model=PipelineStats)
def pipeline_stats() -> dict:
    """Return pipeline statistics."""
    _check_available()

    stats = pipeline_manager.get_stats()
    # Filtrer les cles None dans by_pattern
    by_pattern = {
        k or "unknown": v
        for k, v in stats.get("by_pattern", {}).items()
    }
    stats["by_pattern"] = by_pattern
    return PipelineStats(**stats)


@router.post("/export")
def export_pipelines(request: PipelineExportRequest) -> dict:
    """Export pipelines as YAML."""
    _check_available()

    if request.custom_only:
        content = pipeline_manager.export_custom()
    else:
        content = pipeline_manager.export_all()

    return {"format": "yaml", "content": content}


@router.get("/{pipeline_id}", response_model=PipelineInfo)
def get_pipeline(pipeline_id: str) -> dict:
    """Retrieve a pipeline by its ID."""
    _check_available()

    pipeline = pipeline_manager.get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return _pipeline_to_info(pipeline)


@router.post("", response_model=PipelineInfo, status_code=201)
def create_pipeline(request: PipelineCreate) -> dict:
    """Create a new custom pipeline."""
    _check_available()

    if not request.id.strip():
        raise HTTPException(status_code=422, detail="Pipeline ID cannot be empty")

    # Verifier si l'ID existe deja
    if pipeline_manager.get(request.id):
        raise HTTPException(status_code=409, detail="Pipeline ID already exists")

    # Importer la classe Pipeline pour creer l'objet
    try:
        from opti_oignon.pipeline_manager import Pipeline, PipelineStep
    except ImportError:
        raise HTTPException(status_code=503, detail="Pipeline module not available")

    steps = [
        PipelineStep(
            name=s.name,
            agent=s.agent,
            prompt_template=s.prompt_template,
            description=s.description,
            system_prompt=s.system_prompt,
            model=s.model,
        )
        for s in request.steps
    ]

    pipeline = Pipeline(
        id=request.id,
        name=request.name,
        description=request.description,
        pattern=request.pattern,
        emoji=request.emoji,
        steps=steps,
        keywords=request.keywords,
        detection_weight=request.detection_weight,
        is_builtin=False,
    )

    # PIP-05 (S192): validate before persisting (id format, name, steps,
    # agent/template existence). Mirrors the execution-pipeline routes.
    errors = pipeline_manager.validate_for_write(pipeline)
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    success = pipeline_manager.create(pipeline)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create pipeline")

    return _pipeline_to_info(pipeline)


@router.put("/{pipeline_id}", response_model=PipelineInfo)
def update_pipeline(pipeline_id: str, request: PipelineUpdate) -> dict:
    """Update an existing pipeline (custom only)."""
    _check_available()

    existing = pipeline_manager.get(pipeline_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if existing.is_builtin:
        raise HTTPException(status_code=403, detail="Cannot modify builtin pipeline")

    # Importer les classes necessaires
    try:
        from opti_oignon.pipeline_manager import Pipeline, PipelineStep
    except ImportError:
        raise HTTPException(status_code=503, detail="Pipeline module not available")

    # Build the updated pipeline
    steps = existing.steps
    if request.steps is not None:
        steps = [
            PipelineStep(
                name=s.name,
                agent=s.agent,
                prompt_template=s.prompt_template,
                description=s.description,
                system_prompt=s.system_prompt,
                model=s.model,
            )
            for s in request.steps
        ]

    updated = Pipeline(
        id=pipeline_id,
        name=request.name or existing.name,
        description=request.description if request.description is not None else existing.description,
        pattern=request.pattern or existing.pattern,
        emoji=request.emoji if request.emoji is not None else existing.emoji,
        steps=steps,
        keywords=request.keywords if request.keywords is not None else existing.keywords,
        detection_weight=request.detection_weight if request.detection_weight is not None else existing.detection_weight,
        is_builtin=False,
    )

    # PIP-05 (S192): validate before persisting, same as create.
    errors = pipeline_manager.validate_for_write(updated)
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    success = pipeline_manager.update(pipeline_id, updated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update pipeline")

    # Recuperer depuis le manager pour avoir les data coherentes
    result = pipeline_manager.get(pipeline_id)
    return _pipeline_to_info(result)


@router.delete("/{pipeline_id}")
def delete_pipeline(pipeline_id: str) -> dict:
    """Delete a custom pipeline."""
    _check_available()

    existing = pipeline_manager.get(pipeline_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if existing.is_builtin:
        raise HTTPException(status_code=403, detail="Cannot delete builtin pipeline")

    deleted = pipeline_manager.delete(pipeline_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete pipeline")

    return {"deleted": True, "id": pipeline_id}


@router.post("/{pipeline_id}/duplicate", response_model=PipelineInfo, status_code=201)
def duplicate_pipeline(pipeline_id: str, request: PipelineDuplicateRequest) -> dict:
    """Duplicate a pipeline (builtin or custom)."""
    _check_available()

    if not request.new_id.strip():
        raise HTTPException(status_code=422, detail="New ID cannot be empty")

    # Verifier si le nouvel ID existe deja
    if pipeline_manager.get(request.new_id):
        raise HTTPException(status_code=409, detail="New pipeline ID already exists")

    new_pipeline = pipeline_manager.duplicate(pipeline_id, request.new_id)
    if not new_pipeline:
        raise HTTPException(status_code=404, detail="Source pipeline not found")

    return _pipeline_to_info(new_pipeline)
