#!/usr/bin/env python3
"""
API routes for persistent memory management.

Endpoints pour lister, ajouter, supprimer les faits memoire,
et extraire automatiquement des faits depuis une conversation.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from .deps import MEMORY_AVAILABLE, memory_manager
from .schemas import (
    MemoryAddRequest,
    MemoryEditRequest,
    MemoryExtractResponse,
    MemoryFactSchema,
    MemoryRecordSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _check_available():
    """Check that the memory module is available."""
    if not MEMORY_AVAILABLE or memory_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Memory module not available",
        )


@router.get("", response_model=list[MemoryFactSchema])
def list_facts(
    active_only: bool = True,
    category: str = "",
) -> list:
    """List all facts in memory."""
    _check_available()

    facts = memory_manager.get_all_facts(
        active_only=active_only,
        category=category or None,
    )
    return [
        MemoryFactSchema(
            id=f.id,
            fact=f.fact,
            category=f.category,
            source_conversation_id=f.source_conversation_id,
            created_at=f.created_at,
            updated_at=f.updated_at,
            confidence=f.confidence,
            active=f.active,
        )
        for f in facts
    ]


@router.post("", response_model=MemoryFactSchema)
def add_fact(request: MemoryAddRequest) -> dict:
    """Add a fact to memory."""
    _check_available()

    if not request.fact.strip():
        raise HTTPException(status_code=422, detail="Fact cannot be empty")

    result = memory_manager.add_fact(
        fact=request.fact,
        category=request.category,
        source_conversation_id=request.source_conversation_id,
        confidence=request.confidence,
    )
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to add fact")

    return MemoryFactSchema(
        id=result.id,
        fact=result.fact,
        category=result.category,
        source_conversation_id=result.source_conversation_id,
        created_at=result.created_at,
        updated_at=result.updated_at,
        confidence=result.confidence,
        active=result.active,
    )


@router.delete("/{fact_id}")
def delete_fact(fact_id: str) -> dict:
    """Delete a specific fact."""
    _check_available()

    deleted = memory_manager.delete_fact(fact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Fact not found")
    return {"deleted": True, "id": fact_id}


@router.delete("")
def clear_all_facts() -> dict:
    """Delete all facts in memory."""
    _check_available()

    count = memory_manager.clear_all()
    return {"cleared": True, "count": count}


@router.post("/extract/{conv_id}", response_model=MemoryExtractResponse)
def extract_facts(conv_id: str) -> dict:
    """Extract facts from a conversation and store them.

    Necessite un modele Ollama disponible pour l'extraction LLM.
    """
    _check_available()

    try:
        added = memory_manager.extract_and_store(conv_id)
        return MemoryExtractResponse(
            conversation_id=conv_id,
            facts_added=added,
        )
    except Exception as e:
        logger.error(f"Extraction error for {conv_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# S174: MemoryStore-backed surface (the two-tier memory store).
#
# Mounted under a distinct prefix /api/memories so the legacy /api/memory
# surface above stays intact. Exposes list, soft delete, restore, and edit over
# the coordinated MemoryStore (canonical SQLite WAL + oo_memories vector layer),
# per user (the auth subject; "local" in single-user mode), encrypted at rest
# through the store. All mutations route through the coordinated store, so the
# double dedup and cross-layer consistency apply uniformly.
# ---------------------------------------------------------------------------

try:
    from opti_oignon.memory.dedup import get_memory_store

    MEMORY_STORE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    MEMORY_STORE_AVAILABLE = False
    get_memory_store = None  # type: ignore[assignment]

try:
    from opti_oignon.memory import CATEGORIES as _MEMORY_CATEGORIES
except Exception:  # pragma: no cover
    _MEMORY_CATEGORIES = frozenset(
        {"identity", "preference", "fact", "contact", "project", "goal"}
    )

try:
    from .routes_auth import _get_current_user

    _memories_auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional
    _memories_auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


memories_router = APIRouter(
    prefix="/api/memories", tags=["memories"], dependencies=_memories_auth_dep
)


def _check_store() -> None:
    if not MEMORY_STORE_AVAILABLE or get_memory_store is None:
        raise HTTPException(status_code=503, detail="Memory store not available")


def _record_to_schema(record) -> MemoryRecordSchema:
    return MemoryRecordSchema(
        id=record.id,
        text=record.text,
        category=record.category,
        source=record.source,
        created_at=record.created_at,
        updated_at=record.updated_at,
        active=record.active,
        use_count=record.use_count,
    )


@memories_router.get("", response_model=list[MemoryRecordSchema])
def list_memories(
    active_only: bool = True,
    category: str = "",
    current_user: dict = Depends(_get_current_user),
) -> list:
    """List memories in the store, optionally filtered by category, per user."""
    _check_store()
    store = get_memory_store()
    user_id = current_user.get("sub")
    records = store.list(
        category=category or None, active_only=active_only, user_id=user_id
    )
    return [_record_to_schema(r) for r in records]


@memories_router.patch("/{fact_id}", response_model=MemoryRecordSchema)
def edit_memory(
    fact_id: str,
    request: MemoryEditRequest,
    current_user: dict = Depends(_get_current_user),
) -> dict:
    """Edit a stored memory's text and/or category (mirrored to both layers)."""
    _check_store()
    if request.text is not None and not request.text.strip():
        raise HTTPException(status_code=422, detail="Fact text cannot be empty")
    if request.category is not None and request.category not in _MEMORY_CATEGORIES:
        raise HTTPException(status_code=422, detail="Invalid category")
    store = get_memory_store()
    user_id = current_user.get("sub")
    record = store.update(
        fact_id, text=request.text, category=request.category, user_id=user_id
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _record_to_schema(record)


@memories_router.delete("/{fact_id}")
def soft_delete_memory(
    fact_id: str, current_user: dict = Depends(_get_current_user)
) -> dict:
    """Soft-delete a memory: clear its active flag and drop the vector entry; the
    canonical row is retained so it can be restored."""
    _check_store()
    store = get_memory_store()
    user_id = current_user.get("sub")
    ok = store.soft_delete(fact_id, user_id=user_id)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Memory not found or already inactive"
        )
    return {"soft_deleted": True, "id": fact_id}


@memories_router.post("/{fact_id}/restore", response_model=MemoryRecordSchema)
def restore_memory(
    fact_id: str, current_user: dict = Depends(_get_current_user)
) -> dict:
    """Restore a soft-deleted memory: set active and re-add the vector entry."""
    _check_store()
    store = get_memory_store()
    user_id = current_user.get("sub")
    ok = store.restore(fact_id, user_id=user_id)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Memory not found or already active"
        )
    record = store.get(fact_id, user_id=user_id)
    if record is None:  # pragma: no cover - race
        raise HTTPException(status_code=404, detail="Memory not found")
    return _record_to_schema(record)
