"""Deduplication and coordinated CRUD for Opti-Oignon memory (S173, Theme 3).

Two responsibilities:

1. Double deduplication. On every add, two stages run in order: a Jaccard text
   similarity check at threshold 0.6, then a vector near-duplicate check via the
   vector layer's ``find_similar`` at cosine 0.92. A candidate that trips either
   check is merged into the existing fact rather than duplicated. The text stage
   runs first because it is cheap and needs no embedding.

2. Coordinated CRUD. ``MemoryStore`` wraps the canonical store and the vector
   layer and keeps them consistent: an insert writes to both, a merge reinforces
   the existing fact, an update propagates text and metadata to both, and a
   delete is mirrored (soft delete also drops the vector entry so it stops being
   surfaced, while the canonical row is retained for restore).

The module imports no sibling at module scope (the stores are injected), so it
loads and tests in isolation without ollama, fastapi, or chromadb.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Module sentinels (project convention).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

JACCARD_THRESHOLD = 0.6
COSINE_THRESHOLD = 0.92

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union else 0.0


@dataclass
class DedupDecision:
    """The outcome of a duplicate check.

    action is "insert" or "merge"; on a merge, target_id names the existing fact,
    reason is "jaccard" or "cosine", and score is the similarity that tripped it.
    """

    action: str
    target_id: str | None = None
    reason: str = ""
    score: float = 0.0


class MemoryDeduplicator:
    """The two-stage duplicate finder over the canonical and vector layers."""

    def __init__(
        self,
        canonical: Any,
        vector: Any,
        *,
        jaccard_threshold: float = JACCARD_THRESHOLD,
        cosine_threshold: float = COSINE_THRESHOLD,
        top_k: int = 5,
    ) -> None:
        self._canonical = canonical
        self._vector = vector
        self._jaccard_threshold = jaccard_threshold
        self._cosine_threshold = cosine_threshold
        self._top_k = top_k

    def find_duplicate(
        self,
        text: str,
        *,
        embedding: list[float] | None = None,
        user_id: str | None = None,
    ) -> DedupDecision:
        # Stage 1: Jaccard over the user's active facts (cheap, no embedding).
        best_id: str | None = None
        best_score = 0.0
        for record in self._canonical.list(active_only=True, user_id=user_id):
            score = jaccard_similarity(text, record.text)
            if score >= self._jaccard_threshold and score > best_score:
                best_id, best_score = record.id, score
        if best_id is not None:
            return DedupDecision("merge", best_id, "jaccard", best_score)

        # Stage 2: cosine near-duplicate via the vector layer.
        if embedding is not None:
            neighbours = self._vector.find_similar(
                embedding,
                user_id=user_id,
                top_k=self._top_k,
                threshold=self._cosine_threshold,
            )
            if neighbours:
                top = neighbours[0]
                return DedupDecision("merge", top.id, "cosine", top.similarity)

        return DedupDecision("insert")


class MemoryStore:
    """Coordinated facade keeping the canonical store and vector layer in sync."""

    def __init__(
        self,
        canonical: Any,
        vector: Any,
        *,
        deduplicator: MemoryDeduplicator | None = None,
    ) -> None:
        self._canonical = canonical
        self._vector = vector
        self._dedup = (
            deduplicator
            if deduplicator is not None
            else MemoryDeduplicator(canonical, vector)
        )

    def _uid(self, user_id: str | None) -> str:
        return self._canonical.resolve_user(user_id)

    def resolve_user(self, user_id: str | None = None) -> str:
        """Resolve the effective user id through the canonical store.

        Exposed so a coordinating layer (e.g. curation) can key its own state by
        the same resolved id the store uses, without touching the canonical
        store directly.
        """
        return self._uid(user_id)

    def _embed(self, text: str, embedding: list[float] | None) -> list[float] | None:
        if embedding is not None:
            return list(embedding)
        return self._vector.embed(text)

    # Create (with dedup)

    def add(
        self,
        text: str,
        category: str = "fact",
        *,
        source: str = "",
        user_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> tuple[Any, DedupDecision]:
        """Add a fact unless it duplicates an existing one, in which case merge.

        Returns the resulting record and the dedup decision.
        """
        uid = self._uid(user_id)
        emb = self._embed(text, embedding)
        decision = self._dedup.find_duplicate(text, embedding=emb, user_id=uid)

        if decision.action == "merge" and decision.target_id is not None:
            # Conservative merge: reinforce the existing fact, keep its text.
            self._canonical.touch(decision.target_id, user_id=uid)
            return self._canonical.get(decision.target_id, user_id=uid), decision

        record = self._canonical.add(text, category, source=source, user_id=uid)
        if emb is not None:
            self._vector.add(
                record.id,
                text,
                embedding=emb,
                user_id=record.user_id,
                category=record.category,
                source=source,
            )
        return record, decision

    # Read (canonical is the source of truth)

    def get(self, fact_id: str, *, user_id: str | None = None) -> Any | None:
        return self._canonical.get(fact_id, user_id=self._uid(user_id))

    def list(
        self,
        *,
        category: str | None = None,
        active_only: bool = True,
        user_id: str | None = None,
        order_by: str = "created_at",
        descending: bool = True,
        limit: int | None = None,
    ) -> list[Any]:
        return self._canonical.list(
            category=category,
            active_only=active_only,
            user_id=self._uid(user_id),
            order_by=order_by,
            descending=descending,
            limit=limit,
        )

    def count(self, *, active_only: bool = True, user_id: str | None = None) -> int:
        return self._canonical.count(active_only=active_only, user_id=self._uid(user_id))

    # Update (mirrored)

    def update(
        self,
        fact_id: str,
        *,
        text: str | None = None,
        category: str | None = None,
        source: str | None = None,
        user_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> Any | None:
        uid = self._uid(user_id)
        fields: dict[str, Any] = {}
        if text is not None:
            fields["text"] = text
        if category is not None:
            fields["category"] = category
        if source is not None:
            fields["source"] = source

        record = self._canonical.update(fact_id, user_id=uid, **fields)
        if record is None:
            return None

        vector_embedding = embedding
        if text is not None and vector_embedding is None:
            vector_embedding = self._vector.embed(text)
        self._vector.update(
            fact_id,
            text=text,
            embedding=vector_embedding,
            category=category,
            source=source,
            user_id=uid,
        )
        return record

    def touch(self, fact_id: str, *, user_id: str | None = None) -> bool:
        return self._canonical.touch(fact_id, user_id=self._uid(user_id))

    # Delete (mirrored); soft delete drops the vector entry, keeps the row

    def soft_delete(self, fact_id: str, *, user_id: str | None = None) -> bool:
        uid = self._uid(user_id)
        ok = self._canonical.soft_delete(fact_id, user_id=uid)
        if ok:
            self._vector.delete(fact_id, user_id=uid)
        return ok

    def restore(self, fact_id: str, *, user_id: str | None = None) -> bool:
        uid = self._uid(user_id)
        ok = self._canonical.restore(fact_id, user_id=uid)
        if ok:
            record = self._canonical.get(fact_id, user_id=uid)
            embedding = self._vector.embed(record.text)
            if embedding is not None:
                self._vector.add(
                    record.id,
                    record.text,
                    embedding=embedding,
                    user_id=record.user_id,
                    category=record.category,
                    source=record.source,
                )
        return ok

    def hard_delete(self, fact_id: str, *, user_id: str | None = None) -> bool:
        uid = self._uid(user_id)
        ok = self._canonical.hard_delete(fact_id, user_id=uid)
        self._vector.delete(fact_id, user_id=uid)
        return ok


# Module-level singleton with a reset for test isolation. The sibling stores are
# imported lazily here so this module stays importable in isolation.
_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        from .canonical_store import get_canonical_store
        from .vector_store import get_vector_store

        _memory_store = MemoryStore(get_canonical_store(), get_vector_store())
    return _memory_store


def reset_memory_store() -> None:
    global _memory_store
    _memory_store = None
