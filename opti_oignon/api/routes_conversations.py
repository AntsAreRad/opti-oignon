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
    ConversationTitle,
    MessageItem,
    SourceWithdrawRequest,
    SourceWithdrawResponse,
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


def _refinement():
    """The title refinement's module, or None where it cannot be loaded: asked at the call."""
    try:
        from opti_oignon import title_refiner
    except Exception:  # noqa: BLE001 - absence is an answer: nothing pending, nothing to cancel
        return None
    return title_refiner


def _title_pending(conv_id: str) -> bool:
    """Whether the local model is refining the conversation's title now; False when that cannot be told."""
    refiner = _refinement()
    try:
        return bool(refiner.pending(conv_id)) if refiner is not None else False
    except Exception:  # noqa: BLE001 - a summary never fails for it
        return False


def _cancel_refinement(conv_id: str) -> None:
    """Cancel the refinement of the conversation's title, if one is queued or running; never raises."""
    refiner = _refinement()
    if refiner is not None:
        refiner.cancel(conv_id)


def _conv_to_summary(conv) -> dict:
    """Convert a Conversation object to ConversationSummary dict."""
    # Le message_count vient soit de la propriete (si messages charges),
    # soit du champ _message_count stocke dans metadata par list_conversations()
    msg_count = conv.metadata.get("_message_count", conv.message_count)
    return ConversationSummary(
        id=conv.id,
        title=conv.title,
        title_source=getattr(conv, "title_source", "default"),
        title_pending=_title_pending(conv.id),
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


@router.post("/withdraw", response_model=SourceWithdrawResponse)
def withdraw_source(request: SourceWithdrawRequest) -> dict:
    """The user's withdrawal of a source: every turn and branch message it reached is lowered.

    The source is a lineage entry (kind:identifier), or a kind and the value
    it is named by -- a page's address, a document's text, a fact's id. A
    fact of memory withdrawn is also set aside, restorable. A withdrawal only
    ever lowers; an entry outside the grammar is refused before anything is
    written.
    """
    if not CONVERSATION_AVAILABLE or conversation_manager is None:
        raise HTTPException(status_code=503, detail="Conversation store not available")
    from ..source_withdrawal import source_for, withdraw

    try:
        if request.kind and request.value is not None:
            source = source_for(request.kind, request.value)
        elif request.source:
            source = request.source
        else:
            raise ValueError("a source, or a kind and a value, is required")
        return withdraw(source, conversations=conversation_manager)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"withdrawal refused: {exc}") from None


@router.post("", response_model=ConversationSummary, status_code=201)
def create_conversation(body: ConversationCreate) -> dict:
    """Create a new conversation: the user's title when the body names one, else the store titles it."""
    _check_available()

    title = None
    if body.title is not None and body.title.strip():
        # A title the body names is the user's: one line within the bound,
        # or refused, and nothing is written.
        title = conversation_manager.fold_manual_title(body.title)
        if title is None:
            raise HTTPException(status_code=422, detail="A title is one line of text, not empty, within the bound")
    try:
        conv = conversation_manager.create_conversation(
            title=title,
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
        title_source=getattr(conv, "title_source", "default"),
        title_pending=_title_pending(conv.id),
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
    # A refinement of its title writes nothing now.
    _cancel_refinement(conv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return Response(status_code=204)


@router.patch("/{conv_id}", response_model=ConversationSummary)
def rename_conversation(conv_id: str, body: ConversationRename) -> dict:
    """Rename a conversation: the user's title, kept as typed on one line, never replaced by the store's rule."""
    _check_available()

    # One line, white space collapsed; empty or past titles.yaml's bound is
    # refused, and nothing is written.
    title = conversation_manager.fold_manual_title(body.title)
    if title is None:
        raise HTTPException(status_code=422, detail="A title is one line of text, not empty, within the bound")
    # The user's title: a refinement of the automatic one writes nothing now.
    _cancel_refinement(conv_id)
    renamed = conversation_manager.rename_conversation(conv_id, title)
    if not renamed:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Reload conversation to return updated data
    conv = conversation_manager.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found after rename")

    return _conv_to_summary(conv)


@router.post("/{conv_id}/title/auto", response_model=ConversationSummary)
def use_automatic_title(conv_id: str) -> dict:
    """Give the conversation's title back to the store's rule: derived from the typed words of its first turns."""
    _check_available()

    if not conversation_manager.use_automatic_title(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = conversation_manager.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _conv_to_summary(conv)


@router.get("/{conv_id}/title", response_model=ConversationTitle)
def get_conversation_title(conv_id: str) -> dict:
    """The conversation's title as it stands, and whether the local model is refining it: while it is, when to ask again."""
    _check_available()

    found = conversation_manager.title_of(conv_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    title, source = found
    pending = _title_pending(conv_id)
    retry_after_ms = None
    if pending:
        cfg = conversation_manager.titles_config()
        retry_after_ms = cfg.refine.poll_after_ms if cfg is not None else None
        pending = retry_after_ms is not None
    return ConversationTitle(
        title=title, title_source=source, title_pending=pending, retry_after_ms=retry_after_ms
    ).model_dump()


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
