#!/usr/bin/env python3
"""
API routes for Network Manager / Offline-First Intelligence -- S71.

Provides endpoints for connectivity status, queue management,
and manual queue processing.
"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import (
    NETWORK_MANAGER_AVAILABLE,
    PRE_CACHE_AVAILABLE,
    SYNC_QUEUE_AVAILABLE,
    network_manager,
    pre_cache,
    sync_queue,
)
from .schemas import (
    NetworkStatusResponse,
    PreCacheResponse,
    QueueEntrySchema,
    QueueListResponse,
    QueueProcessResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/network", tags=["network"])


@router.get("/status", response_model=NetworkStatusResponse)
def get_network_status() -> dict:
    """Get current network connectivity status."""
    if not NETWORK_MANAGER_AVAILABLE or network_manager is None:
        return NetworkStatusResponse(
            available=False,
            online=False,
        )

    status = network_manager.status
    return NetworkStatusResponse(
        available=True,
        online=status.online,
        ollama_reachable=status.ollama_reachable,
        embedding_reachable=status.embedding_reachable,
        last_check=status.last_check,
        last_error=status.last_error,
        latency_ms=status.latency_ms,
        consecutive_failures=status.consecutive_failures,
        polling_active=network_manager.running,
        queue_size=sync_queue.size(status="pending") if SYNC_QUEUE_AVAILABLE and sync_queue else 0,
        config=network_manager.get_config(),
    )


@router.post("/poll", response_model=NetworkStatusResponse)
def poll_now() -> dict:
    """Trigger an immediate connectivity check."""
    if not NETWORK_MANAGER_AVAILABLE or network_manager is None:
        raise HTTPException(status_code=503, detail="Network manager not available")

    network_manager.poll_once()
    return get_network_status()


@router.get("/queue", response_model=QueueListResponse)
def get_queue_entries(status_filter: str | None = None, limit: int = 50) -> dict:
    """List queue entries, optionally filtered by status."""
    if not SYNC_QUEUE_AVAILABLE or sync_queue is None:
        return QueueListResponse(available=False, entries=[], total=0)

    entries = sync_queue.list_entries(status=status_filter, limit=limit)
    return QueueListResponse(
        available=True,
        entries=[
            QueueEntrySchema(
                id=e.id,
                query=e.query,
                task_type=e.task_type,
                priority=e.priority,
                created_at=e.created_at,
                status=e.status,
                error=e.error,
                model=e.model,
            )
            for e in entries
        ],
        total=sync_queue.size(),
        pending=sync_queue.size(status="pending"),
    )


@router.post("/queue/process", response_model=QueueProcessResponse)
def process_queue() -> dict:
    """Manually trigger queue processing (drains pending entries).

    Only works if Ollama is online. Uses no executor — entries are
    simply marked completed for now (full executor integration is
    a future enhancement).
    """
    if not SYNC_QUEUE_AVAILABLE or sync_queue is None:
        raise HTTPException(status_code=503, detail="Sync queue not available")

    if NETWORK_MANAGER_AVAILABLE and network_manager is not None:
        if not network_manager.is_online:
            raise HTTPException(
                status_code=409,
                detail="Cannot process queue while offline",
            )

    results = sync_queue.process_queue()
    return QueueProcessResponse(
        processed=len(results),
        results=results,
    )


@router.delete("/queue")
def clear_queue(status_filter: str | None = None) -> dict:
    """Clear queue entries. Optionally filter by status."""
    if not SYNC_QUEUE_AVAILABLE or sync_queue is None:
        raise HTTPException(status_code=503, detail="Sync queue not available")

    removed = sync_queue.clear(status=status_filter)
    return {"removed": removed}


@router.post("/pre-cache", response_model=PreCacheResponse)
def run_pre_cache() -> dict:
    """Trigger a pre-cache warming run."""
    if not PRE_CACHE_AVAILABLE or pre_cache is None:
        raise HTTPException(status_code=503, detail="Pre-cache not available")

    if NETWORK_MANAGER_AVAILABLE and network_manager is not None:
        if not network_manager.is_online:
            raise HTTPException(
                status_code=409,
                detail="Cannot pre-cache while offline",
            )

    result = pre_cache.warm_common_queries()
    return PreCacheResponse(
        total=result.total,
        cached=result.cached,
        skipped=result.skipped,
        failed=result.failed,
        duration_ms=result.duration_ms,
        errors=result.errors,
    )
