#!/usr/bin/env python3
"""
API routes for cache management.

Provides endpoints for the response cache (S18) and semantic cache (S23/S68).
S68 adds: status, config update, clear by conversation, toggle.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .deps import (
    RESPONSE_CACHE_AVAILABLE,
    SEMANTIC_CACHE_AVAILABLE,
    response_cache,
    semantic_cache,
)
from .schemas import (
    CacheClearResponse,
    CacheCombinedStats,
    CacheStatsSchema,
    S68CacheClearRequest,
    S68CacheConfigUpdate,
    S68CacheStatsSchema,
    S68CacheStatusResponse,
    SemanticCacheStatsSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cache", tags=["cache"])


# =========================================================================
# Existing endpoints (S18/S23)
# =========================================================================


@router.get("/stats", response_model=CacheCombinedStats)
def cache_stats() -> dict:
    """Get combined statistics for both caches."""
    result = CacheCombinedStats()

    if RESPONSE_CACHE_AVAILABLE and response_cache is not None:
        try:
            stats = response_cache.get_stats()
            result.response_cache = CacheStatsSchema(
                total_entries=stats.total_entries,
                total_hits=stats.total_hits,
                total_misses=stats.total_misses,
                hit_rate=stats.hit_rate,
                entries_by_model=stats.entries_by_model,
                oldest_entry=stats.oldest_entry,
                total_size_bytes=stats.total_size_bytes,
            )
        except Exception as e:
            logger.error("Response cache stats error: %s", e)

    if SEMANTIC_CACHE_AVAILABLE and semantic_cache is not None:
        try:
            stats = semantic_cache.get_stats()
            result.semantic_cache = SemanticCacheStatsSchema(
                total_embeddings=stats.total_entries,
                semantic_hits=stats.semantic_hits,
                semantic_misses=stats.total_misses,
                avg_similarity=0.0,
                embedding_model=stats.embedding_model,
                threshold=stats.similarity_threshold,
            )
        except Exception as e:
            logger.error("Semantic cache stats error: %s", e)

    return result


@router.delete("", response_model=CacheClearResponse)
def clear_cache() -> dict:
    """Clear the response cache entirely."""
    if not RESPONSE_CACHE_AVAILABLE or response_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Response cache module not available",
        )

    try:
        count = response_cache.clear()
        return CacheClearResponse(
            entries_removed=count,
            source="response_cache",
        )
    except Exception as e:
        logger.error("Cache clear error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model}", response_model=CacheClearResponse)
def clear_cache_model(model: str) -> dict:
    """Clear the response cache for a specific model."""
    if not RESPONSE_CACHE_AVAILABLE or response_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Response cache module not available",
        )

    try:
        count = response_cache.invalidate_model(model)
        return CacheClearResponse(
            entries_removed=count,
            source=f"response_cache:{model}",
        )
    except Exception as e:
        logger.error("Cache clear model error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# S68 endpoints: Semantic Cache (enhanced)
# =========================================================================


@router.get("/s68/status", response_model=S68CacheStatusResponse)
def s68_cache_status() -> dict:
    """Get S68 semantic cache status, stats, and config."""
    if not SEMANTIC_CACHE_AVAILABLE or semantic_cache is None:
        return S68CacheStatusResponse(
            enabled=False,
            available=False,
        )

    try:
        stats = semantic_cache.get_stats()
        return S68CacheStatusResponse(
            enabled=stats.enabled,
            available=True,
            stats=S68CacheStatsSchema(
                total_entries=stats.total_entries,
                exact_hits=stats.exact_hits,
                semantic_hits=stats.semantic_hits,
                total_misses=stats.total_misses,
                hit_rate=stats.hit_rate,
                exact_hit_rate=stats.exact_hit_rate,
                semantic_hit_rate=stats.semantic_hit_rate,
                tokens_saved=stats.tokens_saved,
                size_bytes=stats.size_bytes,
                max_entries=stats.max_entries,
                ttl_seconds=stats.ttl_seconds,
                similarity_threshold=stats.similarity_threshold,
                embedding_model=stats.embedding_model,
                scope=stats.scope,
                enabled=stats.enabled,
                embeddings_available=stats.embeddings_available,
            ),
            config=semantic_cache.get_config(),
        )
    except Exception as e:
        logger.error("S68 cache status error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/s68/toggle", response_model=S68CacheStatusResponse)
def s68_cache_toggle() -> dict:
    """Toggle the S68 semantic cache on/off."""
    if not SEMANTIC_CACHE_AVAILABLE or semantic_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Semantic cache module not available",
        )

    try:
        semantic_cache.enabled = not semantic_cache.enabled
        return s68_cache_status()
    except Exception as e:
        logger.error("S68 cache toggle error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/s68/config", response_model=S68CacheStatusResponse)
def s68_cache_update_config(body: S68CacheConfigUpdate) -> dict:
    """Update S68 semantic cache configuration."""
    if not SEMANTIC_CACHE_AVAILABLE or semantic_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Semantic cache module not available",
        )

    try:
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if updates:
            semantic_cache.update_config(updates)
        return s68_cache_status()
    except Exception as e:
        logger.error("S68 cache config update error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/s68/clear", response_model=CacheClearResponse)
def s68_cache_clear(body: S68CacheClearRequest = S68CacheClearRequest()) -> dict:
    """Clear S68 semantic cache entries.

    If conversation_id is provided, only clears that conversation.
    Otherwise clears all entries.
    """
    if not SEMANTIC_CACHE_AVAILABLE or semantic_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Semantic cache module not available",
        )

    try:
        count = semantic_cache.invalidate(body.conversation_id)
        source = "s68_cache"
        if body.conversation_id:
            source = f"s68_cache:{body.conversation_id}"
        return CacheClearResponse(
            entries_removed=count,
            source=source,
        )
    except Exception as e:
        logger.error("S68 cache clear error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/s68/expire", response_model=CacheClearResponse)
def s68_cache_expire() -> dict:
    """Remove expired entries from S68 semantic cache."""
    if not SEMANTIC_CACHE_AVAILABLE or semantic_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Semantic cache module not available",
        )

    try:
        count = semantic_cache.expire_stale()
        return CacheClearResponse(
            entries_removed=count,
            source="s68_cache_expire",
        )
    except Exception as e:
        logger.error("S68 cache expire error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
