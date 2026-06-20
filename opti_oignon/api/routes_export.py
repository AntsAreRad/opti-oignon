#!/usr/bin/env python3
"""
Routes API pour l'export de conversations.

Endpoint pour exporter une conversation en Markdown, JSON, ou HTML.
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from .deps import CONVERSATION_AVAILABLE, conversation_manager
from .schemas import ExportResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["export"])

# Formats d'export supportes
VALID_FORMATS = {"markdown", "json", "html"}


@router.get("/conversations/{conv_id}/export", response_model=ExportResponse)
def export_conversation(
    conv_id: str,
    format: str = Query(default="markdown", description="Export format"),
) -> dict:
    """Export a conversation in the requested format.

    Formats supportes: markdown, json, html.
    """
    if not CONVERSATION_AVAILABLE or conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation module not available",
        )

    format_lower = format.lower()
    if format_lower not in VALID_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid format: {format}. "
            f"Supported: {', '.join(sorted(VALID_FORMATS))}",
        )

    # Delegation au conversation_manager
    export_methods = {
        "markdown": conversation_manager.export_conversation_markdown,
        "json": conversation_manager.export_conversation_json,
        "html": conversation_manager.export_conversation_html,
    }

    try:
        content = export_methods[format_lower](conv_id)
    except Exception as e:
        logger.error(f"Export error for {conv_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

    if content is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ExportResponse(
        conversation_id=conv_id,
        format=format_lower,
        content=content,
    )
