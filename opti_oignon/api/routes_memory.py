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

# ---------------------------------------------------------------------------
# M3: the legacy /api/memory surface (list/add/delete/clear/extract) is now
# backed by the coordinated MemoryStore -- a single source of truth -- mapped
# onto the existing MemoryFactSchema so the frontend (memory.ts, MemoryPanel)
# is unchanged. memory_manager is retained only for the one-shot migration and
# is otherwise frozen. Single-user mode resolves user_id=None to the local user.
# ---------------------------------------------------------------------------
try:
    from ..conversation import conversation_manager as _conv_manager
    from ..memory.dedup import get_memory_store as _get_store
    from ..memory.extraction import extract_and_store as _extract_and_store
    from ..memory.migration import migrate_legacy_to_store as _migrate_legacy

    _STORE_OK = True
except Exception:  # pragma: no cover - store optional
    _STORE_OK = False
    _get_store = _extract_and_store = _migrate_legacy = _conv_manager = None

try:
    from ..memory.canonical_store import CATEGORIES as _CANON_CATEGORIES

    _MEM_CATEGORIES = frozenset(_CANON_CATEGORIES)
except Exception:  # pragma: no cover
    _MEM_CATEGORIES = frozenset(
        {"identity", "preference", "fact", "contact", "project", "goal"}
    )


def _require_store():
    if not _STORE_OK or _get_store is None:
        raise HTTPException(status_code=503, detail="Memory store not available")
    return _get_store()


def _map_cat(category: str | None) -> str:
    cat = (category or "").strip().lower()
    return cat if cat in _MEM_CATEGORIES else "fact"


def _store_to_fact_schema(record) -> MemoryFactSchema:
    """Map a MemoryStore record onto the legacy MemoryFactSchema (frontend contract)."""
    return MemoryFactSchema(
        id=record.id,
        fact=record.text,
        category=record.category,
        source_conversation_id="",
        created_at=record.created_at or "",
        updated_at=record.updated_at or "",
        confidence=1.0,
        active=record.active,
    )


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
    """List facts from the unified memory store."""
    store = _require_store()
    records = store.list(
        category=category or None, active_only=active_only, user_id=None
    )
    return [_store_to_fact_schema(r) for r in records]


@router.post("", response_model=MemoryFactSchema)
def add_fact(request: MemoryAddRequest) -> dict:
    """Add a fact to the unified memory store (deduplicated)."""
    if not request.fact.strip():
        raise HTTPException(status_code=422, detail="Fact cannot be empty")

    store = _require_store()
    record, _decision = store.add(
        request.fact, _map_cat(request.category), source="manual", user_id=None
    )
    if record is None:
        raise HTTPException(status_code=500, detail="Failed to add fact")
    return _store_to_fact_schema(record)


@router.delete("/{fact_id}")
def delete_fact(fact_id: str) -> dict:
    """Soft-delete a fact (recoverable via the store's restore)."""
    store = _require_store()
    deleted = store.soft_delete(fact_id, user_id=None)
    if not deleted:
        raise HTTPException(status_code=404, detail="Fact not found")
    return {"deleted": True, "id": fact_id}


@router.delete("")
def clear_all_facts() -> dict:
    """Soft-delete all active facts (recoverable; the store has no hard clear)."""
    store = _require_store()
    records = store.list(active_only=True, user_id=None)
    count = 0
    for r in records:
        try:
            if store.soft_delete(r.id, user_id=None):
                count += 1
        except Exception as e:
            logger.warning(f"clear: soft-delete failed for {r.id}: {e}")
    return {"cleared": True, "count": count}


@router.post("/extract/{conv_id}", response_model=MemoryExtractResponse)
def extract_facts(conv_id: str) -> dict:
    """Extract facts from a conversation into the unified store.

    Necessite un modele Ollama disponible pour l'extraction LLM.
    """
    _require_store()
    if _extract_and_store is None or _conv_manager is None:
        raise HTTPException(status_code=503, detail="Extraction not available")

    try:
        messages = _conv_manager.get_context_messages(conv_id)
        results = _extract_and_store(messages, source=f"extract:{conv_id}")
        added = sum(
            1 for _r, d in results if getattr(d, "action", "add") != "merge"
        )
        return MemoryExtractResponse(conversation_id=conv_id, facts_added=added)
    except Exception as e:
        logger.error(f"Extraction error for {conv_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}",
        )


@router.post("/migrate")
def migrate_memory() -> dict:
    """Re-run the one-shot legacy -> store migration (idempotent; dedup-merged)."""
    if _migrate_legacy is None:
        raise HTTPException(status_code=503, detail="Migration not available")
    return _migrate_legacy(force=True)


@router.get("/health")
def memory_health() -> dict:
    """Memory store health.

    The canonical tier (keyword/recency over SQLite) is always available; the
    archive tier (semantic recall over the vector layer) is "degraded" when the
    embedder is down. ``degraded`` is the single overall flag the UI can surface.
    """
    result = {
        "canonical": "ok",
        "archive": "ok",
        "embedder": {"status": "unknown"},
        "degraded": False,
    }
    try:
        from ..memory.vector_store import get_vector_store

        emb = get_vector_store().health()
        result["embedder"] = emb
        if emb.get("status") != "ok":
            result["archive"] = "degraded"
            result["degraded"] = True
    except Exception as e:
        logger.debug(f"memory health probe failed: {e}")
        result["embedder"] = {"status": "unknown", "detail": str(e)}
        result["archive"] = "degraded"
        result["degraded"] = True
    return result


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
