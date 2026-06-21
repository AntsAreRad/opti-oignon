#!/usr/bin/env python3
"""
Fine-tuning data export and variant tracking API routes (S96).

POST   /api/fine-tune/export          -- Export conversations as training data
GET    /api/fine-tune/export/preview   -- Preview export with filters
GET    /api/fine-tune/quality          -- Get conversation quality scores
GET    /api/fine-tune/variants         -- List tracked fine-tuned variants
POST   /api/fine-tune/variants         -- Register new variant
DELETE /api/fine-tune/variants/{id}    -- Unregister variant
POST   /api/fine-tune/compare          -- Run A/B comparison
GET    /api/fine-tune/compare/{id}     -- Get comparison result
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fine-tune", tags=["fine-tune"])


# =============================================================================
# SCHEMAS
# =============================================================================

class ExportRequest(BaseModel):
    """Request body for training data export."""
    format: str = Field(default="sharegpt", description="Export format: sharegpt, alpaca, or jsonl")
    conversation_ids: list[str] | None = Field(default=None, description="Specific conversation IDs")
    date_from: str | None = Field(default=None, description="Start date (ISO format)")
    date_to: str | None = Field(default=None, description="End date (ISO format)")
    model: str | None = Field(default=None, description="Filter by model name")
    min_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum quality score")
    min_turns: int = Field(default=1, ge=1, description="Minimum conversation turns")


class VariantCreateRequest(BaseModel):
    """Request body for registering a fine-tuned variant."""
    name: str = Field(description="Human-readable variant name")
    base_model: str = Field(description="Base model name (e.g. qwen3:32b)")
    variant_model: str = Field(description="Fine-tuned model name in Ollama")
    description: str = Field(default="", description="Optional description")
    dataset_size: int = Field(default=0, ge=0, description="Training dataset size")
    epochs: int = Field(default=0, ge=0, description="Training epochs")
    learning_rate: float = Field(default=0.0, ge=0.0, description="Learning rate used")
    loss: float = Field(default=0.0, ge=0.0, description="Final training loss")
    training_duration_seconds: float = Field(default=0.0, ge=0.0, description="Training duration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class VariantResponse(BaseModel):
    """Response for a single variant."""
    variant_id: str
    name: str
    base_model: str
    variant_model: str
    status: str
    created_at: str
    updated_at: str
    description: str = ""
    dataset_size: int = 0
    epochs: int = 0
    learning_rate: float = 0.0
    loss: float = 0.0
    training_duration_seconds: float = 0.0
    metadata: dict[str, Any] = {}


class CompareRequest(BaseModel):
    """Request body for A/B comparison."""
    variant_id: str = Field(description="Variant ID to compare against base")
    prompts: list[str] = Field(description="Prompts to test both models with")


class CompareResponse(BaseModel):
    """Response for a comparison result."""
    comparison_id: str
    variant_id: str
    base_model: str
    variant_model: str
    status: str
    created_at: str
    completed_at: str = ""
    base_wins: int = 0
    variant_wins: int = 0
    ties: int = 0
    base_avg_latency_ms: float = 0.0
    variant_avg_latency_ms: float = 0.0
    summary: str = ""
    prompts: list[dict[str, Any]] = []


# =============================================================================
# HELPERS
# =============================================================================

def _get_exporter():
    """Get the fine-tune exporter singleton, raising 503 if unavailable."""
    try:
        from opti_oignon.api.deps import FINE_TUNE_EXPORT_AVAILABLE, fine_tune_exporter
        if not FINE_TUNE_EXPORT_AVAILABLE or fine_tune_exporter is None:
            raise HTTPException(status_code=503, detail="Fine-tune export not available")
        return fine_tune_exporter
    except ImportError:
        raise HTTPException(status_code=503, detail="Fine-tune export module not available")


def _get_tracker():
    """Get the fine-tune tracker singleton, raising 503 if unavailable."""
    try:
        from opti_oignon.api.deps import FINE_TUNE_TRACKER_AVAILABLE, fine_tune_tracker
        if not FINE_TUNE_TRACKER_AVAILABLE or fine_tune_tracker is None:
            raise HTTPException(status_code=503, detail="Fine-tune tracker not available")
        return fine_tune_tracker
    except ImportError:
        raise HTTPException(status_code=503, detail="Fine-tune tracker module not available")


# =============================================================================
# EXPORT ENDPOINTS
# =============================================================================

@router.post("/export")
def export_training_data(req: ExportRequest) -> dict:
    """Export conversations as training data in the requested format."""
    exporter = _get_exporter()

    from opti_oignon.fine_tune_export import ExportFilter

    filters = ExportFilter(
        conversation_ids=req.conversation_ids,
        date_from=req.date_from,
        date_to=req.date_to,
        model=req.model,
        min_quality=req.min_quality,
        min_turns=req.min_turns,
    )

    try:
        result = exporter.export(fmt=req.format, filters=filters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Return raw data as downloadable text for JSONL, JSON for others
    if req.format == "jsonl":
        return PlainTextResponse(
            content=result.data,
            media_type="application/jsonl",
            headers={
                "Content-Disposition": "attachment; filename=training_data.jsonl",
            },
        )

    return {
        "format": result.format,
        "conversation_count": result.conversation_count,
        "message_count": result.message_count,
        "data": result.data,
        "timestamp": result.timestamp,
        "filters_applied": result.filters_applied,
    }


@router.get("/export/preview")
def preview_export(
    format: str = Query(default="sharegpt", description="Export format"),
    model: str | None = Query(default=None, description="Filter by model"),
    min_quality: float = Query(default=0.0, ge=0.0, le=1.0),
    min_turns: int = Query(default=1, ge=1),
    max_preview: int = Query(default=3, ge=1, le=10),
) -> dict:
    """Preview export: shows count, sample entries, and quality scores."""
    exporter = _get_exporter()

    from opti_oignon.fine_tune_export import ExportFilter

    filters = ExportFilter(
        model=model,
        min_quality=min_quality,
        min_turns=min_turns,
    )

    try:
        preview = exporter.preview(fmt=format, filters=filters, max_preview=max_preview)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return preview


@router.get("/quality")
def get_quality_scores(
    conversation_ids: str | None = Query(default=None, description="Comma-separated IDs"),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """Get quality scores for conversations."""
    exporter = _get_exporter()

    ids = None
    if conversation_ids:
        ids = [cid.strip() for cid in conversation_ids.split(",") if cid.strip()]

    scores = exporter.get_quality_scores(conversation_ids=ids, limit=limit)
    return {
        "scores": [s.to_dict() for s in scores],
        "count": len(scores),
    }


# =============================================================================
# VARIANT ENDPOINTS
# =============================================================================

@router.get("/variants")
def list_variants(
    base_model: str | None = Query(default=None, description="Filter by base model"),
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """List registered fine-tuned variants."""
    tracker = _get_tracker()
    variants = tracker.list_variants(base_model=base_model, status=status, limit=limit)
    return {
        "variants": [v.to_dict() for v in variants],
        "count": len(variants),
    }


@router.post("/variants", status_code=201)
def register_variant(req: VariantCreateRequest) -> dict:
    """Register a new fine-tuned model variant."""
    tracker = _get_tracker()

    from opti_oignon.fine_tune_tracker import FineTuneVariant

    variant = FineTuneVariant(
        name=req.name,
        base_model=req.base_model,
        variant_model=req.variant_model,
        description=req.description,
        dataset_size=req.dataset_size,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        loss=req.loss,
        training_duration_seconds=req.training_duration_seconds,
        metadata=req.metadata,
    )

    try:
        registered = tracker.register_variant(variant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return registered.to_dict()


@router.delete("/variants/{variant_id}")
def unregister_variant(variant_id: str) -> dict:
    """Unregister a fine-tuned variant and its comparison history."""
    tracker = _get_tracker()

    deleted = tracker.unregister_variant(variant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Variant '{variant_id}' not found")

    return {"deleted": True, "variant_id": variant_id}


# =============================================================================
# COMPARISON ENDPOINTS
# =============================================================================

@router.post("/compare")
def run_comparison(req: CompareRequest) -> dict:
    """Create and run an A/B comparison between base and fine-tuned models.

    Note: actual inference requires an Ollama connection. Without one,
    the comparison is created but marked as failed.
    """
    tracker = _get_tracker()

    try:
        comparison = tracker.create_comparison(
            variant_id=req.variant_id,
            prompts=req.prompts,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Attempt to run with Ollama inference
    inference_fn = _get_inference_fn()
    result = tracker.run_comparison(
        comparison_id=comparison.comparison_id,
        inference_fn=inference_fn,
    )

    return result.to_dict()


@router.get("/compare/{comparison_id}")
def get_comparison(comparison_id: str) -> dict:
    """Get a comparison result by ID."""
    tracker = _get_tracker()

    result = tracker.get_comparison(comparison_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Comparison '{comparison_id}' not found")

    return result.to_dict()


# =============================================================================
# INFERENCE HELPER
# =============================================================================

def _get_inference_fn():
    """Build an inference function using Ollama, or return None."""
    try:
        import ollama

        def _infer(model: str, prompt: str) -> str:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(response, "message"):
                return response.message.content or ""
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            return str(response)

        return _infer
    except ImportError:
        logger.debug("Ollama not available for A/B comparison inference")
        return None
    except Exception as exc:
        logger.debug("Failed to create inference function: %s", exc)
        return None
