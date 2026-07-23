#!/usr/bin/env python3
"""
SEARCH ROUTES - Opti-Oignon API
==================================

Endpoints for web search proxy management, PII sanitization preview,
and search configuration.

"""

import logging

from fastapi import APIRouter, HTTPException

from .deps import WEB_SEARCH_AVAILABLE
from .schemas import (
    PIISanitizePreviewItem,
    PIISanitizePreviewRequest,
    PIISanitizePreviewResponse,
    ProxyConfigRequest,
    ProxyConfigResponse,
    ProxyStatusResponse,
    SearchConfigResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


def _get_searcher():
    """Get the web searcher singleton, raising 503 if unavailable."""
    if not WEB_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Web search module is not available",
        )
    from opti_oignon.web_search import web_searcher
    return web_searcher


# =============================================================================
# PROXY STATUS
# =============================================================================

@router.get("/proxy-status", response_model=ProxyStatusResponse)
def get_proxy_status() -> dict:
    """
    Check proxy health and connectivity.

    Returns proxy configuration state, reachability, latency,
    and Tor exit IP if applicable.
    """
    searcher = _get_searcher()
    status = searcher.check_proxy_status()
    return ProxyStatusResponse(
        configured=status.configured,
        proxy_url=status.proxy_url,
        reachable=status.reachable,
        latency_ms=status.latency_ms,
        exit_ip=status.exit_ip,
        error=status.error,
    )


# =============================================================================
# PROXY CONFIGURATION
# =============================================================================

@router.get("/proxy-config", response_model=ProxyConfigResponse)
def get_proxy_config() -> dict:
    """Get current proxy configuration."""
    searcher = _get_searcher()
    mode = "off"
    if searcher.proxy_configured:
        proxy = searcher.config.proxy or ""
        if "9050" in proxy:
            mode = "tor"
        else:
            mode = "custom"

    return ProxyConfigResponse(
        mode=mode,
        proxy_url=searcher.config.proxy,
        proxy_timeout=searcher.config.proxy_timeout,
        max_retries=searcher.config.max_retries,
        retry_backoff=searcher.config.retry_backoff,
        pii_sanitization_enabled=searcher.config.pii_sanitization_enabled,
    )


@router.post("/proxy-config", response_model=ProxyConfigResponse)
def update_proxy_config(request: ProxyConfigRequest) -> dict:
    """
    Update proxy configuration at runtime.

    Modes:
    - "off": Direct connection (no proxy)
    - "tor": Use socks5h://localhost:9050
    - "custom": Use provided proxy_url

    Does NOT persist to YAML (runtime only). Use settings API for persistence.
    """
    searcher = _get_searcher()

    if request.mode == "off":
        searcher.set_proxy(None)
    elif request.mode == "tor":
        searcher.set_proxy("socks5h://localhost:9050")
    elif request.mode == "custom":
        if not request.proxy_url:
            raise HTTPException(
                status_code=400,
                detail="proxy_url is required when mode is 'custom'",
            )
        searcher.set_proxy(request.proxy_url)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Use 'off', 'tor', or 'custom'.",
        )

    # Update optional config fields
    if request.proxy_timeout is not None:
        searcher.config.proxy_timeout = request.proxy_timeout
    if request.max_retries is not None:
        searcher.config.max_retries = request.max_retries
    if request.pii_sanitization_enabled is not None:
        searcher.config.pii_sanitization_enabled = request.pii_sanitization_enabled

    # Return updated config
    mode = request.mode
    return ProxyConfigResponse(
        mode=mode,
        proxy_url=searcher.config.proxy,
        proxy_timeout=searcher.config.proxy_timeout,
        max_retries=searcher.config.max_retries,
        retry_backoff=searcher.config.retry_backoff,
        pii_sanitization_enabled=searcher.config.pii_sanitization_enabled,
    )


# =============================================================================
# PII SANITIZATION PREVIEW
# =============================================================================

@router.post("/pii-preview", response_model=PIISanitizePreviewResponse)
def preview_pii_sanitization(request: PIISanitizePreviewRequest) -> dict:
    """
    Preview what PII would be stripped from a search query.

    Does not execute any search. Shows the user what will be sanitized
    before sending the query through the proxy.
    """
    searcher = _get_searcher()
    preview = searcher.preview_sanitization(request.query)

    items = [
        PIISanitizePreviewItem(
            original=item["original"],
            replacement=item["replacement"],
            category=item["category"],
        )
        for item in preview.get("items", [])
    ]

    return PIISanitizePreviewResponse(
        original=preview["original"],
        sanitized=preview["sanitized"],
        items=items,
        was_modified=preview["was_modified"],
    )


# =============================================================================
# SEARCH CONFIG (READ-ONLY OVERVIEW)
# =============================================================================

@router.get("/config", response_model=SearchConfigResponse)
def get_search_config() -> dict:
    """Get current search configuration overview."""
    searcher = _get_searcher()
    stats = searcher.get_cache_stats()

    return SearchConfigResponse(
        ddgs_available=stats.get("ddgs_available", False),
        pii_available=stats.get("pii_available", False),
        proxy_configured=stats.get("proxy_configured", False),
        cache_size=stats.get("cache_size", 0),
        total_searches=stats.get("total_searches", 0),
        cache_hits=stats.get("cache_hits", 0),
        errors=stats.get("errors", 0),
        retries=stats.get("retries", 0),
        pii_sanitizations=stats.get("pii_sanitizations", 0),
        proxy_searches=stats.get("proxy_searches", 0),
    )
