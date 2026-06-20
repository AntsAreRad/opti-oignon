#!/usr/bin/env python3
"""
API routes for Prompt Optimization — Opti-Oignon S65.

Endpoints for token budget inspection, template listing,
template retrieval, runtime overrides, and full config.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prompt", tags=["prompt"])


# ============================================================================
# Pydantic schemas
# ============================================================================

class TokenBudgetResponse(BaseModel):
    """Response for budget calculation endpoint."""
    model: str
    total_window: int
    system_tokens: int
    project_tokens: int
    history_tokens: int
    user_tokens: int
    reserve_tokens: int
    total_input_tokens: int
    total_allocated: int
    utilization: float


class TokenBudgetRequest(BaseModel):
    """Optional query params for budget calculation."""
    project_active: bool = False
    conversation_length: int = 0
    context_window_override: int = 0


class TemplateResponse(BaseModel):
    """Response for a single template."""
    task_type: str
    system_prompt: str
    temperature_override: float | None = None
    stop_sequences: list[str] = []
    source: str = "yaml"


class TemplateSummary(BaseModel):
    """Summary of a template for listing."""
    task_type: str
    has_temperature_override: bool
    temperature_override: float | None = None
    source: str
    prompt_length: int


class TemplateOverrideRequest(BaseModel):
    """Request body for setting a runtime template override."""
    system_prompt: str = Field(..., min_length=1, max_length=10000)
    temperature_override: float | None = None
    stop_sequences: list[str] | None = None


class PromptConfigResponse(BaseModel):
    """Full prompt optimization configuration."""
    enabled: bool
    budget: dict
    templates: dict


# ============================================================================
# Helper: get managers or 404
# ============================================================================

def _get_budget_manager():
    """Get PromptTokenBudgetManager or raise 503."""
    from .deps import PROMPT_OPTIMIZATION_AVAILABLE, prompt_budget_manager
    if not PROMPT_OPTIMIZATION_AVAILABLE or prompt_budget_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Prompt optimization module is not available",
        )
    return prompt_budget_manager


def _get_template_engine():
    """Get PromptTemplateEngine or raise 503."""
    from .deps import PROMPT_OPTIMIZATION_AVAILABLE, prompt_template_engine
    if not PROMPT_OPTIMIZATION_AVAILABLE or prompt_template_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Prompt optimization module is not available",
        )
    return prompt_template_engine


# ============================================================================
# Budget endpoints
# (IMPORTANT: specific routes MUST come before {model:path} catch-all)
# ============================================================================

@router.post("/budget/cache/clear")
def clear_budget_cache() -> dict:
    """Clear the context window cache."""
    mgr = _get_budget_manager()
    count = mgr.clear_cache()
    return {"cleared": count}


@router.get("/budget/cache/stats")
def get_cache_stats() -> dict:
    """Get context window cache statistics."""
    mgr = _get_budget_manager()
    return mgr.cache_stats()


@router.get("/budget/window/{model:path}")
def get_context_window(model: str) -> dict:
    """Get the detected context window size for a model."""
    mgr = _get_budget_manager()
    window = mgr.get_context_window(model)
    return {"model": model, "context_window": window}


@router.get("/budget/{model:path}", response_model=TokenBudgetResponse)
def get_budget(
    model: str,
    project_active: bool = False,
    conversation_length: int = 0,
    context_window_override: int = 0,
) -> dict:
    """Get token budget breakdown for a model.

    Args:
        model: Ollama model name (e.g. 'qwen3:32b').
        project_active: Whether a project context is active.
        conversation_length: Number of messages in conversation.
        context_window_override: Override context window (0 = auto-detect).
    """
    mgr = _get_budget_manager()
    budget = mgr.calculate_budget(
        model=model,
        conversation_length=conversation_length,
        project_active=project_active,
        context_window_override=context_window_override,
    )
    return budget.as_dict()


# ============================================================================
# Template endpoints
# ============================================================================

@router.get("/templates", response_model=list[TemplateSummary])
def list_templates() -> dict:
    """List all available prompt templates."""
    engine = _get_template_engine()
    return engine.list_templates()


@router.get("/templates/{task_type}", response_model=TemplateResponse)
def get_template(task_type: str, project_id: str | None = None) -> dict:
    """Get a specific prompt template by task type.

    Args:
        task_type: Task type (e.g. 'code_r', 'scientific_writing').
        project_id: Optional project ID for project-specific overrides.
    """
    engine = _get_template_engine()
    tpl = engine.get_template(task_type, project_id=project_id)
    # Interpolate to resolve {language_rule} etc.
    interpolated_prompt = engine.interpolate(tpl)
    return TemplateResponse(
        task_type=tpl.task_type,
        system_prompt=interpolated_prompt,
        temperature_override=tpl.temperature_override,
        stop_sequences=tpl.stop_sequences,
        source=tpl.source,
    )


@router.put("/templates/{task_type}", response_model=TemplateResponse)
def set_template_override(task_type: str, body: TemplateOverrideRequest) -> dict:
    """Set a runtime template override for a task type.

    Runtime overrides take highest priority and persist until cleared
    or the process restarts.
    """
    engine = _get_template_engine()
    tpl = engine.set_runtime_override(
        task_type=task_type,
        system_prompt=body.system_prompt,
        temperature_override=body.temperature_override,
        stop_sequences=body.stop_sequences,
    )
    return TemplateResponse(
        task_type=tpl.task_type,
        system_prompt=tpl.system_prompt,
        temperature_override=tpl.temperature_override,
        stop_sequences=tpl.stop_sequences,
        source=tpl.source,
    )


@router.delete("/templates/{task_type}/override")
def clear_template_override(task_type: str) -> dict:
    """Clear a runtime template override for a task type."""
    engine = _get_template_engine()
    removed = engine.clear_runtime_override(task_type)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"No runtime override for task_type='{task_type}'",
        )
    return {"cleared": task_type}


@router.delete("/templates/overrides/all")
def clear_all_template_overrides() -> dict:
    """Clear all runtime template overrides."""
    engine = _get_template_engine()
    count = engine.clear_all_runtime_overrides()
    return {"cleared": count}


# ============================================================================
# Config endpoint
# ============================================================================

@router.get("/config")
def get_prompt_config() -> dict:
    """Get full prompt optimization configuration."""
    from .deps import PROMPT_OPTIMIZATION_AVAILABLE
    mgr = _get_budget_manager()
    engine = _get_template_engine()
    return {
        "enabled": PROMPT_OPTIMIZATION_AVAILABLE,
        "budget": mgr.get_config(),
        "templates": engine.get_config(),
    }


@router.post("/config/reload")
def reload_prompt_config() -> dict:
    """Reload prompt optimization configuration from YAML files."""
    mgr = _get_budget_manager()
    engine = _get_template_engine()
    mgr.reload_config()
    engine.reload_config()
    return {"status": "reloaded"}
