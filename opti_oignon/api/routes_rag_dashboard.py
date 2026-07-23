#!/usr/bin/env python3
"""
RAG DASHBOARD API routes.

GET    /api/rag/dashboard/stats       -- Overall dashboard statistics
GET    /api/rag/dashboard/usage       -- Query usage over time
GET    /api/rag/dashboard/sources     -- Source reliability ranking
GET    /api/rag/dashboard/health      -- Collection health metrics
POST   /api/rag/dashboard/refresh     -- Trigger auto-refresh check
GET    /api/rag/dashboard/stale       -- List stale sources
GET    /api/rag/dashboard/connectors  -- External connector status
GET    /api/rag/dashboard/backends    -- Available external backends
"""

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag/dashboard", tags=["rag-dashboard"])


# =========================================================================
# PYDANTIC SCHEMAS
# =========================================================================

class OverallStatsResponse(BaseModel):
    total_collections: int
    total_documents: int
    total_chunks: int
    total_citations: int
    total_queries_today: int
    total_queries_week: int
    total_queries_all: int
    avg_score: float
    storage_bytes: int


class UsageDataPointResponse(BaseModel):
    date: str
    query_count: int
    citation_count: int
    avg_score: float


class UsageResponse(BaseModel):
    data: list[UsageDataPointResponse]
    days: int


class SourceReliabilityResponse(BaseModel):
    source_file: str
    collection_name: str
    doc_id: str
    citation_count: int
    avg_score: float
    last_cited: float
    freshness_score: float
    reliability_score: float


class SourcesResponse(BaseModel):
    sources: list[SourceReliabilityResponse]
    total: int


class CollectionHealthResponse(BaseModel):
    name: str
    document_count: int
    chunk_count: int
    citation_count: int
    avg_chunk_size: float
    file_types: list[str]
    last_ingestion: float
    last_query: float
    freshness_score: float


class HealthResponse(BaseModel):
    collections: list[CollectionHealthResponse]
    total: int


class RefreshResponse(BaseModel):
    checked_at: float
    sources_checked: int
    sources_refreshed: int
    errors: list[str]


class StaleSourceResponse(BaseModel):
    doc_id: str
    source_file: str
    collection_name: str
    ingested_at: float
    age_days: float


class StaleResponse(BaseModel):
    sources: list[StaleSourceResponse]
    total: int


class ConnectorStatusResponse(BaseModel):
    name: str
    connector_type: str
    connected: bool
    document_count: int
    last_query_time_ms: float
    error: str | None = None


class ConnectorsResponse(BaseModel):
    connectors: list[ConnectorStatusResponse]
    total: int


class BackendsResponse(BaseModel):
    backends: dict[str, bool]


# =========================================================================
# HELPERS
# =========================================================================

def _get_dashboard():
    """Get the RAGDashboardStats singleton."""
    try:
        from opti_oignon.rag_dashboard import get_rag_dashboard
        return get_rag_dashboard()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RAG dashboard module not available",
        )
    except Exception as exc:
        logger.error("Failed to initialize RAG dashboard: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"RAG dashboard initialisation failed: {exc}",
        )


def _get_auto_refresh():
    """Get the RAGAutoRefresh singleton."""
    try:
        from opti_oignon.rag_dashboard import get_auto_refresh
        return get_auto_refresh()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RAG auto-refresh module not available",
        )
    except Exception as exc:
        logger.error("Failed to initialize RAG auto-refresh: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"RAG auto-refresh initialisation failed: {exc}",
        )


def _get_external_manager():
    """Get the ExternalVectorStoreManager singleton."""
    try:
        from opti_oignon.rag_external import get_external_manager
        return get_external_manager()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="RAG external stores module not available",
        )
    except Exception as exc:
        logger.error("Failed to initialize external manager: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"External manager initialisation failed: {exc}",
        )


# =========================================================================
# STATS ENDPOINT
# =========================================================================

@router.get("/stats", response_model=OverallStatsResponse)
def get_stats() -> dict:
    """Get overall RAG dashboard statistics."""
    dashboard = _get_dashboard()
    stats = dashboard.get_overall_stats()
    return OverallStatsResponse(
        total_collections=stats.total_collections,
        total_documents=stats.total_documents,
        total_chunks=stats.total_chunks,
        total_citations=stats.total_citations,
        total_queries_today=stats.total_queries_today,
        total_queries_week=stats.total_queries_week,
        total_queries_all=stats.total_queries_all,
        avg_score=stats.avg_score,
        storage_bytes=stats.storage_bytes,
    )


# =========================================================================
# USAGE ENDPOINT
# =========================================================================

@router.get("/usage", response_model=UsageResponse)
def get_usage(
    days: int = Query(30, ge=1, le=365, description="Number of days to show"),
) -> dict:
    """Get query usage over time (daily data points)."""
    dashboard = _get_dashboard()
    data = dashboard.get_usage_over_time(days=days)
    items = [
        UsageDataPointResponse(
            date=d.date,
            query_count=d.query_count,
            citation_count=d.citation_count,
            avg_score=d.avg_score,
        )
        for d in data
    ]
    return UsageResponse(data=items, days=days)


# =========================================================================
# SOURCES ENDPOINT
# =========================================================================

@router.get("/sources", response_model=SourcesResponse)
def get_sources(
    limit: int = Query(50, ge=1, le=500, description="Max sources to return"),
) -> dict:
    """Get source reliability ranking."""
    dashboard = _get_dashboard()
    sources = dashboard.get_source_reliability(limit=limit)
    items = [
        SourceReliabilityResponse(
            source_file=s.source_file,
            collection_name=s.collection_name,
            doc_id=s.doc_id,
            citation_count=s.citation_count,
            avg_score=s.avg_score,
            last_cited=s.last_cited,
            freshness_score=s.freshness_score,
            reliability_score=s.reliability_score,
        )
        for s in sources
    ]
    return SourcesResponse(sources=items, total=len(items))


# =========================================================================
# HEALTH ENDPOINT
# =========================================================================

@router.get("/health", response_model=HealthResponse)
def get_health() -> dict:
    """Get collection health metrics."""
    dashboard = _get_dashboard()
    health = dashboard.get_collection_health()
    items = [
        CollectionHealthResponse(
            name=h.name,
            document_count=h.document_count,
            chunk_count=h.chunk_count,
            citation_count=h.citation_count,
            avg_chunk_size=h.avg_chunk_size,
            file_types=h.file_types,
            last_ingestion=h.last_ingestion,
            last_query=h.last_query,
            freshness_score=h.freshness_score,
        )
        for h in health
    ]
    return HealthResponse(collections=items, total=len(items))


# =========================================================================
# REFRESH ENDPOINT
# =========================================================================

@router.post("/refresh", response_model=RefreshResponse)
def trigger_refresh() -> dict:
    """Trigger a manual auto-refresh check for stale sources."""
    refresher = _get_auto_refresh()
    result = refresher.check_and_refresh()
    return RefreshResponse(
        checked_at=result.checked_at,
        sources_checked=result.sources_checked,
        sources_refreshed=result.sources_refreshed,
        errors=result.errors,
    )


# =========================================================================
# STALE SOURCES ENDPOINT
# =========================================================================

@router.get("/stale", response_model=StaleResponse)
def get_stale_sources(
    max_age_days: float = Query(7.0, ge=0.1, le=365.0, description="Max age in days"),
) -> dict:
    """List sources that haven't been refreshed recently."""
    refresher = _get_auto_refresh()
    stale = refresher.get_stale_sources(max_age_days=max_age_days)
    items = [
        StaleSourceResponse(
            doc_id=s["doc_id"],
            source_file=s["source_file"],
            collection_name=s["collection_name"],
            ingested_at=s["ingested_at"],
            age_days=s["age_days"],
        )
        for s in stale
    ]
    return StaleResponse(sources=items, total=len(items))


# =========================================================================
# EXTERNAL CONNECTORS ENDPOINT
# =========================================================================

@router.get("/connectors", response_model=ConnectorsResponse)
def get_connectors() -> dict:
    """Get status of all external vector store connectors."""
    manager = _get_external_manager()
    statuses = manager.list_connectors()
    items = [
        ConnectorStatusResponse(
            name=s.name,
            connector_type=s.connector_type,
            connected=s.connected,
            document_count=s.document_count,
            last_query_time_ms=s.last_query_time_ms,
            error=s.error,
        )
        for s in statuses
    ]
    return ConnectorsResponse(connectors=items, total=len(items))


# =========================================================================
# AVAILABLE BACKENDS ENDPOINT
# =========================================================================

@router.get("/backends", response_model=BackendsResponse)
def get_backends() -> dict:
    """Check which external vector store backends are available."""
    manager = _get_external_manager()
    backends = manager.get_available_backends()
    return BackendsResponse(backends=backends)
