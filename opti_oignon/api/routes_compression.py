#!/usr/bin/env python3
"""
API routes for Conversation Compressor — Opti-Oignon.

Endpoints for compression config inspection, runtime updates,
per-conversation stats, and full-archive search.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/compression", tags=["compression"])


# ============================================================================
# Pydantic schemas
# ============================================================================

class CompressionConfigResponse(BaseModel):
    """Current compression configuration."""
    enabled: bool
    strategy: str
    recent_messages_keep: int
    compression_threshold_ratio: float
    llm_summary_model: str | None
    llm_summary_max_tokens: int
    llm_summary_temperature: float
    llm_summary_timeout: float
    rule_max_facts_per_message: int
    rule_min_message_length: int
    archive_retrieval_top_k: int
    archive_retrieval_min_score: float
    archive_retrieval_snippet_length: int
    retrieval_trigger_enabled: bool
    retrieval_trigger_min_confidence: float


class CompressionConfigUpdateRequest(BaseModel):
    """Partial update for compression configuration.

    All fields are optional; only provided fields are updated.
    """
    enabled: bool | None = None
    strategy: str | None = Field(
        None,
        description="Compression strategy: 'rule', 'llm', or 'hybrid'",
    )
    recent_messages_keep: int | None = Field(None, ge=1, le=50)
    compression_threshold_ratio: float | None = Field(None, ge=0.1, le=2.0)
    llm_summary_model: str | None = None
    llm_summary_max_tokens: int | None = Field(None, ge=50, le=2000)
    llm_summary_temperature: float | None = Field(None, ge=0.0, le=1.0)
    archive_retrieval_top_k: int | None = Field(None, ge=1, le=20)
    archive_retrieval_min_score: float | None = Field(None, ge=0.0, le=1.0)
    retrieval_trigger_enabled: bool | None = None
    retrieval_trigger_min_confidence: float | None = Field(None, ge=0.0, le=1.0)


class CompressedContextStats(BaseModel):
    """Stats from the last compression operation for a conversation."""
    conversation_id: str
    last_compression_available: bool
    summary: str | None = None
    original_count: int | None = None
    compressed_count: int | None = None
    strategy_used: str | None = None
    tokens_saved: int | None = None
    compression_ratio: float | None = None


class ArchiveSearchRequest(BaseModel):
    """Request body for archive search."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=20)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)


class ArchiveSearchResultItem(BaseModel):
    """Single archive search result."""
    message_id: int
    role: str
    snippet: str
    score: float
    timestamp: str


class ArchiveSearchResponse(BaseModel):
    """Response for archive search."""
    conversation_id: str
    query: str
    results: list[ArchiveSearchResultItem]
    total_found: int


# ============================================================================
# Helper: get compressor or 503
# ============================================================================

def _get_compressor():
    """Return the ConversationCompressor singleton or raise 503."""
    from .deps import CONVERSATION_COMPRESSOR_AVAILABLE, conversation_compressor
    if not CONVERSATION_COMPRESSOR_AVAILABLE or conversation_compressor is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation compressor module is not available",
        )
    return conversation_compressor


# ============================================================================
# Config endpoints
# ============================================================================

@router.get("/config", response_model=CompressionConfigResponse)
def get_compression_config() -> dict:
    """Get the current conversation compression configuration."""
    compressor = _get_compressor()
    cfg = compressor.get_config()
    return CompressionConfigResponse(
        enabled=bool(cfg.get("enabled", True)),
        strategy=str(cfg.get("strategy", "hybrid")),
        recent_messages_keep=int(cfg.get("recent_messages_keep", 6)),
        compression_threshold_ratio=float(cfg.get("compression_threshold_ratio", 1.0)),
        llm_summary_model=cfg.get("llm_summary_model"),
        llm_summary_max_tokens=int(cfg.get("llm_summary_max_tokens", 300)),
        llm_summary_temperature=float(cfg.get("llm_summary_temperature", 0.2)),
        llm_summary_timeout=float(cfg.get("llm_summary_timeout", 30)),
        rule_max_facts_per_message=int(cfg.get("rule_max_facts_per_message", 2)),
        rule_min_message_length=int(cfg.get("rule_min_message_length", 50)),
        archive_retrieval_top_k=int(cfg.get("archive_retrieval_top_k", 3)),
        archive_retrieval_min_score=float(cfg.get("archive_retrieval_min_score", 0.05)),
        archive_retrieval_snippet_length=int(cfg.get("archive_retrieval_snippet_length", 300)),
        retrieval_trigger_enabled=bool(cfg.get("retrieval_trigger_enabled", True)),
        retrieval_trigger_min_confidence=float(cfg.get("retrieval_trigger_min_confidence", 0.6)),
    )


@router.put("/config", response_model=CompressionConfigResponse)
def update_compression_config(body: CompressionConfigUpdateRequest) -> dict:
    """Update compression configuration at runtime.

    Only the fields present in the request body are updated.
    Changes are not persisted to disk; they reset on restart.
    """
    compressor = _get_compressor()

    # Validate strategy value if provided
    if body.strategy is not None and body.strategy not in ("rule", "llm", "hybrid"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy '{body.strategy}'. Must be one of: rule, llm, hybrid",
        )

    # Build update dict from non-None fields only
    updates = {
        key: value
        for key, value in body.model_dump().items()
        if value is not None
    }
    compressor.update_config(updates)

    # Return current (updated) config
    return get_compression_config()


@router.post("/config/reload")
def reload_compression_config() -> dict:
    """Reload compression configuration from compression.yaml.

    Runtime overrides are discarded; YAML values take effect.
    """
    compressor = _get_compressor()
    compressor.reload_config()
    return {"status": "reloaded"}


# ============================================================================
# Per-conversation stats endpoint
# ============================================================================

@router.get("/stats/{conversation_id}", response_model=CompressedContextStats)
def get_compression_stats(conversation_id: str) -> dict:
    """Get compression stats for a conversation.

    Returns the last known compression result for the given conversation
    as tracked by the executor singleton. If the executor has not yet
    compressed this conversation (or the conversation hasn't been queried
    since startup), last_compression_available will be False.
    """
    # Guard: compressor must be available
    _get_compressor()

    from .deps import EXECUTOR_AVAILABLE, executor

    last_result = None
    if EXECUTOR_AVAILABLE and executor is not None:
        last_result = getattr(executor, "last_compression_result", None)

    if last_result is None:
        return CompressedContextStats(
            conversation_id=conversation_id,
            last_compression_available=False,
        )

    return CompressedContextStats(
        conversation_id=conversation_id,
        last_compression_available=True,
        summary=last_result.summary if last_result.summary else None,
        original_count=last_result.original_count,
        compressed_count=last_result.compressed_count,
        strategy_used=last_result.strategy_used,
        tokens_saved=last_result.tokens_saved,
        compression_ratio=last_result.compression_ratio,
    )


# ============================================================================
# Archive search endpoint
# ============================================================================

@router.post("/archive/search/{conversation_id}", response_model=ArchiveSearchResponse)
def search_archive(conversation_id: str, body: ArchiveSearchRequest) -> dict:
    """Search the full uncompressed conversation archive.

    Performs keyword-based retrieval over the complete SQLite history,
    bypassing any compression that may have been applied to the working
    prompt. Useful for manually inspecting what context is available
    from older parts of a conversation.

    Args:
        conversation_id: UUID of the conversation to search.
        body: Query string, top_k, and min_score threshold.
    """
    compressor = _get_compressor()

    try:
        results = compressor.retrieve_from_archive(
            conversation_id=conversation_id,
            query=body.query,
            top_k=body.top_k,
        )
    except Exception as e:
        logger.error(f"Archive search failed for {conversation_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Archive search error: {e}",
        )

    # Filter by min_score (retrieve_from_archive uses config default;
    # here we apply the per-request override)
    filtered = [r for r in results if r.score >= body.min_score]

    return ArchiveSearchResponse(
        conversation_id=conversation_id,
        query=body.query,
        results=[
            ArchiveSearchResultItem(
                message_id=r.message_id,
                role=r.role,
                snippet=r.snippet,
                score=round(r.score, 4),
                timestamp=r.timestamp,
            )
            for r in filtered
        ],
        total_found=len(filtered),
    )
