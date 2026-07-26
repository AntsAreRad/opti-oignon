#!/usr/bin/env python3
"""
Routes API pour les conversations.

Expose les operations CRUD sur les conversations via FastAPI,
en delegant au ConversationManager existant.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from .deps import CONVERSATION_AVAILABLE, conversation_manager
from .schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationRename,
    ConversationSummary,
    MessageItem,
)

logger = logging.getLogger(__name__)

# Audit fix: require authentication for all endpoints
try:
    from .routes_auth import _get_current_user
    _auth_dep = [Depends(_get_current_user)]
except ImportError:
    _auth_dep = []

router = APIRouter(prefix="/api/conversations", tags=["conversations"], dependencies=_auth_dep)


def _check_available():
    """Check that the conversation module is available."""
    if not CONVERSATION_AVAILABLE or conversation_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Conversation module unavailable",
        )


def _conv_to_summary(conv) -> dict:
    """Convert a Conversation object to ConversationSummary dict."""
    # Le message_count vient soit de la propriete (si messages charges),
    # soit du champ _message_count stocke dans metadata par list_conversations()
    msg_count = conv.metadata.get("_message_count", conv.message_count)
    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=msg_count,
        model=conv.model,
        task_type=conv.task_type,
        preset=conv.preset,
    ).model_dump()


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    q: str | None = Query(None, description="Terme de search"),
    limit: int = Query(50, ge=1, le=500, description="Nombre max de resultats"),
    offset: int = Query(0, ge=0, description="Offset pour pagination"),
) -> list:
    """Liste les conversations, avec search optionnelle."""
    _check_available()

    try:
        if q and q.strip():
            # Search dans titres et contenus
            convs = conversation_manager.search_conversations(q.strip(), limit=limit)
        else:
            convs = conversation_manager.list_conversations(limit=limit, offset=offset)

        return [_conv_to_summary(c) for c in convs]
    except Exception as e:
        logger.error(f"Erreur listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=ConversationSummary, status_code=201)
def create_conversation(body: ConversationCreate) -> dict:
    """Create a new conversation."""
    _check_available()

    try:
        conv = conversation_manager.create_conversation(
            title=body.title,
            model=body.model,
            preset=body.preset,
        )
        return _conv_to_summary(conv)
    except Exception as e:
        logger.error(f"Erreur creation conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conv_id}", response_model=ConversationDetail)
def get_conversation(conv_id: str) -> dict:
    """Retrieve a conversation with all its messages."""
    _check_available()

    conv = conversation_manager.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        messages=[m.to_dict() for m in conv.messages],
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        model=conv.model,
        task_type=conv.task_type,
        preset=conv.preset,
        message_count=conv.message_count,
        total_tokens=conv.total_tokens,
    ).model_dump()


@router.delete("/{conv_id}", status_code=204)
def delete_conversation(conv_id: str) -> Response:
    """Delete a conversation and all its messages."""
    _check_available()

    deleted = conversation_manager.delete_conversation(conv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return Response(status_code=204)


@router.patch("/{conv_id}", response_model=ConversationSummary)
def rename_conversation(conv_id: str, body: ConversationRename) -> dict:
    """Renomme une conversation."""
    _check_available()

    renamed = conversation_manager.rename_conversation(conv_id, body.title)
    if not renamed:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Reload conversation to return updated data
    conv = conversation_manager.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found after rename")

    return _conv_to_summary(conv)


@router.get("/{conv_id}/messages", response_model=list[MessageItem])
def get_messages(conv_id: str) -> list:
    """Retrieve the messages of a conversation."""
    _check_available()

    # Check that the conversation exists
    conv = conversation_manager.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = conversation_manager.get_messages(conv_id)
    return [
        MessageItem(
            id=m.id,
            role=m.role,
            content=m.content,
            timestamp=m.timestamp,
            model=m.model,
            token_estimate=m.token_estimate,
        ).model_dump()
        for m in messages
    ]


# =============================================================================
# TOOL HISTORY
# =============================================================================

# Conditional import of agentic executor for tool history
try:
    from opti_oignon.agentic_executor import agentic_executor as _agentic_executor
    _AGENTIC_AVAILABLE = True
except ImportError:
    _AGENTIC_AVAILABLE = False
    _agentic_executor = None


@router.get("/{conv_id}/tool-history")
def get_tool_history(conv_id: str) -> dict:
    """Get tool call history for a conversation.

    Returns all prior tool call results stored in memory
    for this conversation, enabling multi-turn tool reference.
    """
    if not _AGENTIC_AVAILABLE or _agentic_executor is None:
        return {"conversation_id": conv_id, "tool_calls": [], "count": 0}

    history = _agentic_executor.get_tool_history(conv_id)
    tool_calls = []
    for tc in history:
        tool_calls.append({
            "tool_name": getattr(tc, "tool_name", ""),
            "arguments": getattr(tc, "arguments", {}),
            "result": getattr(tc, "result", ""),
            "success": getattr(tc, "success", False),
            "execution_time": getattr(tc, "execution_time", 0.0),
            "reasoning": getattr(tc, "reasoning", ""),
        })

    return {
        "conversation_id": conv_id,
        "tool_calls": tool_calls,
        "count": len(tool_calls),
    }


@router.delete("/{conv_id}/tool-history")
def clear_tool_history(conv_id: str) -> dict:
    """Clear tool call history for a conversation."""
    if not _AGENTIC_AVAILABLE or _agentic_executor is None:
        return {"conversation_id": conv_id, "cleared": 0}

    cleared = _agentic_executor.clear_tool_history(conv_id)
    return {"conversation_id": conv_id, "cleared": cleared}
