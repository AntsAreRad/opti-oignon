#!/usr/bin/env python3
"""
API routes for the execution pipelines.

CRUD endpoints for the ExecutionPipeline objects: list, create,
update, delete, duplicate, and the available step types.
Separate from the existing routes_pipelines.py, which serves the
orchestrator's multi-agent pipelines.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/execution-pipelines", tags=["execution-pipelines"])

# Conditional import of pipelines module
try:
    from opti_oignon.pipelines import (
        STEP_TYPE_DESCRIPTIONS,  # noqa: F401
        VALID_STEP_TYPES,  # noqa: F401
        ExecutionPipeline,
        ExecutionStep,
        get_pipeline_store,
    )
    EXEC_PIPELINES_AVAILABLE = True
except ImportError:
    EXEC_PIPELINES_AVAILABLE = False
    get_pipeline_store = None


# =============================================================================
# SCHEMAS PYDANTIC
# =============================================================================

class ExecStepSchema(BaseModel):
    """Schema of an execution step."""
    step_type: str
    label: str = ""
    model_override: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    condition: str | None = None
    pass_previous_output: bool = True


class ExecPipelineInfo(BaseModel):
    """Execution pipeline information."""
    id: str
    name: str
    description: str = ""
    steps: list[ExecStepSchema] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    is_builtin: bool = False
    step_count: int = 0
    step_types_summary: str = ""


class ExecPipelineCreate(BaseModel):
    """Request body for creating an execution pipeline."""
    id: str
    name: str
    description: str = ""
    steps: list[ExecStepSchema] = Field(default_factory=list)


class ExecPipelineUpdate(BaseModel):
    """Request body for modifying an execution pipeline."""
    name: str | None = None
    description: str | None = None
    steps: list[ExecStepSchema] | None = None


class ExecDuplicateRequest(BaseModel):
    """Request body for duplicating a pipeline."""
    new_id: str


class StepTypeInfo(BaseModel):
    """Information sur un type de step."""
    type: str
    description: str = ""


# =============================================================================
# HELPERS
# =============================================================================

def _check_available():
    """Check that the module is available."""
    if not EXEC_PIPELINES_AVAILABLE or get_pipeline_store is None:
        raise HTTPException(
            status_code=503,
            detail="Execution pipelines module not available",
        )


def _pipeline_to_info(p):
    """Convert an ExecutionPipeline to schema."""
    steps = [
        ExecStepSchema(
            step_type=s.step_type,
            label=s.label,
            model_override=s.model_override,
            parameters=s.parameters,
            condition=s.condition,
            pass_previous_output=s.pass_previous_output,
        )
        for s in p.steps
    ]
    return ExecPipelineInfo(
        id=p.id,
        name=p.name,
        description=p.description,
        steps=steps,
        created_at=p.created_at,
        updated_at=p.updated_at,
        is_builtin=p.is_builtin,
        step_count=p.step_count,
        step_types_summary=p.step_types_summary,
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("", response_model=list[ExecPipelineInfo])
def list_exec_pipelines(
    builtin_only: bool = False,
    custom_only: bool = False,
) -> list:
    """Liste tous les pipelines d'execution."""
    _check_available()
    store = get_pipeline_store()
    if builtin_only:
        pipelines = store.list_builtin()
    elif custom_only:
        pipelines = store.list_custom()
    else:
        pipelines = store.list_all()
    return [_pipeline_to_info(p) for p in pipelines]


@router.get("/step-types", response_model=list[StepTypeInfo])
def list_step_types() -> list:
    """Liste les types de step disponibles avec descriptions."""
    _check_available()
    store = get_pipeline_store()
    return [StepTypeInfo(**st) for st in store.get_step_types()]


@router.get("/{pipeline_id}", response_model=ExecPipelineInfo)
def get_exec_pipeline(pipeline_id: str) -> dict:
    """Retrieve an execution pipeline by ID."""
    _check_available()
    store = get_pipeline_store()
    pipeline = store.get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return _pipeline_to_info(pipeline)


@router.post("", response_model=ExecPipelineInfo, status_code=201)
def create_exec_pipeline(request: ExecPipelineCreate) -> dict:
    """Create a new execution pipeline."""
    _check_available()
    store = get_pipeline_store()

    if not request.id.strip():
        raise HTTPException(status_code=422, detail="Pipeline ID cannot be empty")

    if store.get(request.id):
        raise HTTPException(status_code=409, detail="Pipeline ID already exists")

    steps = [
        ExecutionStep(
            step_type=s.step_type,
            label=s.label,
            model_override=s.model_override,
            parameters=s.parameters,
            condition=s.condition,
            pass_previous_output=s.pass_previous_output,
        )
        for s in request.steps
    ]

    pipeline = ExecutionPipeline(
        id=request.id,
        name=request.name,
        description=request.description,
        steps=steps,
        is_builtin=False,
    )

    errors = pipeline.validate()
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    success = store.create(pipeline)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create pipeline")

    return _pipeline_to_info(pipeline)


@router.put("/{pipeline_id}", response_model=ExecPipelineInfo)
def update_exec_pipeline(pipeline_id: str, request: ExecPipelineUpdate) -> dict:
    """Update an execution pipeline (custom only)."""
    _check_available()
    store = get_pipeline_store()

    existing = store.get(pipeline_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if existing.is_builtin:
        raise HTTPException(status_code=403, detail="Cannot modify builtin pipeline")

    # Build the updated pipeline
    steps = existing.steps
    if request.steps is not None:
        steps = [
            ExecutionStep(
                step_type=s.step_type,
                label=s.label,
                model_override=s.model_override,
                parameters=s.parameters,
                condition=s.condition,
                pass_previous_output=s.pass_previous_output,
            )
            for s in request.steps
        ]

    updated = ExecutionPipeline(
        id=pipeline_id,
        name=request.name or existing.name,
        description=(
            request.description
            if request.description is not None
            else existing.description
        ),
        steps=steps,
        is_builtin=False,
    )

    errors = updated.validate()
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    success = store.update(pipeline_id, updated)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update pipeline")

    result = store.get(pipeline_id)
    return _pipeline_to_info(result)


@router.delete("/{pipeline_id}")
def delete_exec_pipeline(pipeline_id: str) -> dict:
    """Delete an execution pipeline (custom only)."""
    _check_available()
    store = get_pipeline_store()

    existing = store.get(pipeline_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    if existing.is_builtin:
        raise HTTPException(status_code=403, detail="Cannot delete builtin pipeline")

    deleted = store.delete(pipeline_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete pipeline")

    return {"deleted": True, "id": pipeline_id}


@router.post("/{pipeline_id}/duplicate", response_model=ExecPipelineInfo, status_code=201)
def duplicate_exec_pipeline(pipeline_id: str, request: ExecDuplicateRequest) -> dict:
    """Duplicate an execution pipeline."""
    _check_available()
    store = get_pipeline_store()

    if not request.new_id.strip():
        raise HTTPException(status_code=422, detail="New ID cannot be empty")

    if store.get(request.new_id):
        raise HTTPException(status_code=409, detail="New pipeline ID already exists")

    new_pipeline = store.duplicate(pipeline_id, request.new_id)
    if not new_pipeline:
        raise HTTPException(status_code=404, detail="Source pipeline not found")

    return _pipeline_to_info(new_pipeline)
