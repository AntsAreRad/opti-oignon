"""Hybrid retrieval for Opti-Oignon personal memory.

Retrieval combines three signals: vector similarity from the oo_memories layer,
keyword coverage of the query terms by a fact, and a category match driven by a
lightweight query-type detection step. The selected memories are then formatted
under the token budget from ``context_window.py`` and returned as a plain block,
ready for the untrusted-context wrapping the agent applies (this module
does not wrap; it only selects and formats).

Per-user isolation is enforced: both the vector query and the canonical scan are
scoped to the resolved user, so retrieval never crosses users. The module
imports its sibling stores by injection and only guard-imports the token
estimator, so it loads and tests in isolation.
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

# Hybrid scoring weights and defaults.
VECTOR_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.3
CATEGORY_WEIGHT = 0.1
VECTOR_TOP_K = 10
DEFAULT_TOP_N = 5
MEMORY_TOKEN_BUDGET = 512

# Token estimation reused from context_window.py, with an identical fallback so
# the module behaves the same when context_window is not importable.
try:
    from ..context_window import SlidingWindowManager

    def _estimate_tokens(text: str) -> int:
        return SlidingWindowManager._estimate_tokens(text)

    _HAS_CONTEXT_WINDOW = True
except Exception:
    _HAS_CONTEXT_WINDOW = False

    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text.split()) * 1.3))


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English words filtered from keyword scoring so a query does not match a
# fact merely through shared function words.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to",
        "in", "on", "at", "for", "and", "or", "but", "my", "me", "i", "you",
        "your", "it", "its", "this", "that", "these", "those", "with", "about",
        "as", "by", "from", "do", "does", "did", "have", "has", "had", "what",
        "which", "when", "where", "how", "who", "tell", "remind", "set", "get",
        "today", "now", "please", "up", "out", "into", "s",
    }
)

# Distinctive cues per category for query-type detection. The "fact" category is
# the fallback and has no cues.
_CATEGORY_CUES: dict[str, frozenset[str]] = {
    "identity": frozenset({"name", "who", "myself", "born", "age", "nickname"}),
    "preference": frozenset(
        {"prefer", "preference", "favorite", "favourite", "like", "likes", "love", "dislike", "rather"}
    ),
    "contact": frozenset({"email", "phone", "address", "contact", "number"}),
    "project": frozenset({"project", "projects", "repo", "repository", "codebase"}),
    "goal": frozenset({"goal", "goals", "objective", "aim", "plan", "plans"}),
}


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall((text or "").lower()))


def detect_query_type(query: str) -> str | None:
    """Return the most likely category for a query, or None if ambiguous.

    Rule-based: count overlap with each category's distinctive cues and pick the
    sole maximum. A tie returns None (treated as a general query, no category
    boost).
    """
    qtokens = _tokens(query)
    if not qtokens:
        return None
    counts = {cat: len(qtokens & cues) for cat, cues in _CATEGORY_CUES.items()}
    best = max(counts.values())
    if best == 0:
        return None
    winners = [cat for cat, n in counts.items() if n == best]
    return winners[0] if len(winners) == 1 else None


@dataclass
class QueryAnalysis:
    query: str
    category_hint: str | None
    tokens: frozenset[str]


@dataclass
class ScoredMemory:
    id: str
    text: str
    category: str
    score: float
    vector_similarity: float
    keyword_score: float
    category_match: bool
    record: Any = None


@dataclass
class DualLayerMemory:
    """The dual-layer view of personal memory for prompt assembly.

    block is the compressed working layer injected into the prompt (the budgeted
    selection); selected_ids names the facts it carries; total_active is the size
    of the full active archive, always >= the selection, so anything dropped from
    the working block stays recoverable through ``recover``. The block is
    unwrapped -- the agent applies the untrusted-context wrapping.
    """

    block: str
    selected_ids: list[str]
    total_active: int


class MemoryRetriever:
    """Hybrid retriever over the canonical store and the vector layer."""

    def __init__(
        self,
        canonical: Any,
        vector: Any,
        *,
        vector_weight: float = VECTOR_WEIGHT,
        keyword_weight: float = KEYWORD_WEIGHT,
        category_weight: float = CATEGORY_WEIGHT,
        vector_top_k: int = VECTOR_TOP_K,
    ) -> None:
        self._canonical = canonical
        self._vector = vector
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._category_weight = category_weight
        self._vector_top_k = vector_top_k

    def analyze_query(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(
            query=query,
            category_hint=detect_query_type(query),
            tokens=_tokens(query),
        )

    @staticmethod
    def _keyword_score(query_tokens: frozenset[str], text: str) -> float:
        """Coverage of the query's content terms by the fact text, in [0, 1]."""
        query_content = query_tokens - _STOPWORDS
        if not query_content:
            return 0.0
        fact_content = _tokens(text) - _STOPWORDS
        overlap = len(query_content & fact_content)
        return overlap / len(query_content)

    def retrieve(
        self,
        query: str,
        *,
        user_id: str | None = None,
        top_n: int = DEFAULT_TOP_N,
        query_embedding: list[float] | None = None,
        mark_used: bool = False,
    ) -> list[ScoredMemory]:
        uid = self._canonical.resolve_user(user_id)
        analysis = self.analyze_query(query)

        embedding = query_embedding
        if embedding is None:
            embedding = self._vector.embed(query)

        similarity_by_id: dict[str, float] = {}
        if embedding is not None:
            for neighbour in self._vector.find_similar(
                embedding, user_id=uid, top_k=self._vector_top_k
            ):
                similarity_by_id[neighbour.id] = neighbour.similarity

        scored: list[ScoredMemory] = []
        for record in self._canonical.list(active_only=True, user_id=uid):
            vector_similarity = max(0.0, similarity_by_id.get(record.id, 0.0))
            keyword_score = self._keyword_score(analysis.tokens, record.text)
            category_match = (
                analysis.category_hint is not None
                and record.category == analysis.category_hint
            )
            score = (
                self._vector_weight * vector_similarity
                + self._keyword_weight * keyword_score
                + (self._category_weight if category_match else 0.0)
            )
            if score <= 0.0:
                continue
            scored.append(
                ScoredMemory(
                    id=record.id,
                    text=record.text,
                    category=record.category,
                    score=score,
                    vector_similarity=vector_similarity,
                    keyword_score=keyword_score,
                    category_match=category_match,
                    record=record,
                )
            )

        scored.sort(
            key=lambda m: (m.score, m.vector_similarity, m.record.use_count),
            reverse=True,
        )
        selected = scored[: max(0, int(top_n))]

        if mark_used:
            for memory in selected:
                self._canonical.touch(memory.id, user_id=uid)

        return selected

    def fit_to_budget(
        self,
        memories: list[ScoredMemory],
        *,
        max_tokens: int = MEMORY_TOKEN_BUDGET,
        header: str = "Relevant memories:",
    ) -> list[ScoredMemory]:
        """Return the prefix of memories that fits within the token budget."""
        included: list[ScoredMemory] = []
        used = _estimate_tokens(header)
        for memory in memories:
            cost = _estimate_tokens(self._format_line(memory))
            if used + cost > max_tokens:
                break
            used += cost
            included.append(memory)
        return included

    def format_for_prompt(
        self,
        memories: list[ScoredMemory],
        *,
        max_tokens: int = MEMORY_TOKEN_BUDGET,
        header: str = "Relevant memories:",
    ) -> str:
        """Format selected memories as a plain block under the token budget.

        The result is unwrapped; the agent applies the untrusted-context wrapping
        downstream. Returns an empty string when nothing fits.
        """
        included = self.fit_to_budget(memories, max_tokens=max_tokens, header=header)
        if not included:
            return ""
        lines = [header] + [self._format_line(m) for m in included]
        return "\n".join(lines)

    @staticmethod
    def _format_line(memory: ScoredMemory) -> str:
        return "- [" + memory.category + "] " + memory.text

    # The dual-layer design. The working block is the compressed layer that goes
    # into the prompt; the full uncompressed archive stays searchable through the
    # canonical store and the vector layer, so a detail dropped from the working
    # block is always recoverable. The block stays unwrapped here; the agent
    # applies the untrusted-context wrapping.

    def recent_memories(
        self, *, user_id: str | None = None, top_n: int = DEFAULT_TOP_N
    ) -> list[ScoredMemory]:
        """The most-used then most-recently-updated active facts.

        Used as the working layer when there is no query to rank against (the
        "recent memories" half of the working set).
        """
        uid = self._canonical.resolve_user(user_id)
        records = self._canonical.list(active_only=True, user_id=uid)
        records = sorted(
            records, key=lambda r: (int(r.use_count), r.updated_at), reverse=True
        )[: max(0, int(top_n))]
        return [
            ScoredMemory(
                id=r.id,
                text=r.text,
                category=r.category,
                score=float(r.use_count),
                vector_similarity=0.0,
                keyword_score=0.0,
                category_match=False,
                record=r,
            )
            for r in records
        ]

    def working_block(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        top_n: int = DEFAULT_TOP_N,
        max_tokens: int = MEMORY_TOKEN_BUDGET,
        query_embedding: list[float] | None = None,
        mark_used: bool = False,
        header: str = "Relevant memories:",
    ) -> str:
        """The compressed working layer for the prompt.

        Query-relevant memories when a query is given, otherwise recent memories,
        formatted under the token budget. Unwrapped (the agent wraps it as
        untrusted context). Returns an empty string when nothing fits.
        """
        if query and query.strip():
            selected = self.retrieve(
                query,
                user_id=user_id,
                top_n=top_n,
                query_embedding=query_embedding,
                mark_used=mark_used,
            )
        else:
            selected = self.recent_memories(user_id=user_id, top_n=top_n)
            if mark_used:
                uid = self._canonical.resolve_user(user_id)
                for memory in selected:
                    self._canonical.touch(memory.id, user_id=uid)
        return self.format_for_prompt(selected, max_tokens=max_tokens, header=header)

    def composed_memories(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        salient_n: int = DEFAULT_TOP_N,
        relevant_n: int = DEFAULT_TOP_N,
        query_embedding: list[float] | None = None,
        mark_used: bool = False,
    ) -> list[ScoredMemory]:
        """Salient baseline + query-relevant facts, deduplicated, salient-first.

        The SALIENCE FLOOR (most-used then most-recent active facts) is ALWAYS
        included regardless of the query, so durable facts -- identity,
        preferences -- stay in the working set even on an unrelated turn; this is
        the property the query-only :meth:`retrieve` lacks (it drops every fact
        scoring 0). Query-relevant facts are appended after, skipping any already
        in the floor. The order is salient-first so the budget truncation (a tail
        cut in :meth:`fit_to_budget`) keeps the durable facts. ``mark_used``
        touches exactly the facts returned -- the reinforcement loop: an injected
        fact gains use_count, an unused one decays out of the working set but
        stays in the searchable archive.
        """
        uid = self._canonical.resolve_user(user_id)
        salient = self.recent_memories(user_id=uid, top_n=salient_n)
        seen = {m.id for m in salient}
        relevant: list[ScoredMemory] = []
        if query and query.strip():
            for m in self.retrieve(
                query,
                user_id=uid,
                top_n=relevant_n,
                query_embedding=query_embedding,
                mark_used=False,
            ):
                if m.id not in seen:
                    seen.add(m.id)
                    relevant.append(m)
        combined = salient + relevant
        if mark_used:
            for m in combined:
                self._canonical.touch(m.id, user_id=uid)
        return combined

    def recover(
        self,
        query: str,
        *,
        user_id: str | None = None,
        top_n: int = VECTOR_TOP_K,
        query_embedding: list[float] | None = None,
    ) -> list[ScoredMemory]:
        """Search the full uncompressed archive without the prompt budget.

        The recovery path of the dual layer: a detail dropped from the working
        block is found here, against the same canonical store and vector layer.
        """
        return self.retrieve(
            query,
            user_id=user_id,
            top_n=top_n,
            query_embedding=query_embedding,
            mark_used=False,
        )

    def assemble_dual_layer(
        self,
        query: str | None = None,
        *,
        user_id: str | None = None,
        top_n: int = DEFAULT_TOP_N,
        max_tokens: int = MEMORY_TOKEN_BUDGET,
        query_embedding: list[float] | None = None,
        mark_used: bool = False,
        header: str = "Relevant memories:",
    ) -> DualLayerMemory:
        """Assemble the dual-layer view: the working block, the ids it selected,
        and the full active-archive size (always >= the selection)."""
        uid = self._canonical.resolve_user(user_id)
        if query and query.strip():
            selected = self.retrieve(
                query,
                user_id=uid,
                top_n=top_n,
                query_embedding=query_embedding,
                mark_used=mark_used,
            )
        else:
            selected = self.recent_memories(user_id=uid, top_n=top_n)
        fitted = self.fit_to_budget(selected, max_tokens=max_tokens, header=header)
        block = self.format_for_prompt(selected, max_tokens=max_tokens, header=header)
        total_active = self._canonical.count(active_only=True, user_id=uid)
        return DualLayerMemory(
            block=block,
            selected_ids=[m.id for m in fitted],
            total_active=total_active,
        )


# Module-level singleton with a reset for test isolation. Sibling stores are
# imported lazily so this module stays importable in isolation.
_retriever: MemoryRetriever | None = None


def get_retriever() -> MemoryRetriever:
    global _retriever
    if _retriever is None:
        from .canonical_store import get_canonical_store
        from .vector_store import get_vector_store

        _retriever = MemoryRetriever(get_canonical_store(), get_vector_store())
    return _retriever


def reset_retriever() -> None:
    global _retriever
    _retriever = None


def working_memory_block(
    query: str | None = None,
    *,
    user_id: str | None = None,
    max_tokens: int = MEMORY_TOKEN_BUDGET,
    top_n: int = DEFAULT_TOP_N,
    mark_used: bool = False,
) -> str:
    """Module-level convenience: the working block over the singleton retriever.

    This is the seam the prompt-assembly path uses to inject the compressed
    memory layer; the full archive stays searchable via ``recover_memories``.
    """
    return get_retriever().working_block(
        query, user_id=user_id, max_tokens=max_tokens, top_n=top_n, mark_used=mark_used
    )


def _norm_text(text: str) -> str:
    """Whitespace/case-normalised form for cross-store deduplication."""
    return " ".join(str(text or "").strip().lower().split())


def build_memory_block(
    query: str | None = None,
    *,
    user_id: str | None = None,
    max_tokens: int = MEMORY_TOKEN_BUDGET,
    legacy_facts: Any = None,
    mark_used: bool = False,
    header: str = "Relevant memories:",
    retriever: MemoryRetriever | None = None,
) -> str:
    """The unified working-memory block injected into the prompt (M1).

    Composes, in ONE token budget and ONE deduplicated block:

      1. the salient baseline + query-relevant facts from the new MemoryStore
         (:meth:`MemoryRetriever.composed_memories`) -- so durable facts persist
         and relevant ones surface, replacing the either/or that let a stale
         query drop everything;
      2. a LEGACY BRIDGE: facts in ``legacy_facts`` (each exposing ``text`` /
         ``category`` -- the legacy memories.db rows), merged after the canonical
         facts and deduplicated against them by normalised text, so an existing
         legacy store keeps surfacing during the migration. This argument is
         temporary and is removed once the writes are unified.

    Graceful: a missing embedder only drops the vector contribution; the
    canonical (SQL) facts still rank and inject. The block is unwrapped (the
    agent wraps it as untrusted context). Returns "" when nothing fits.
    """
    r = retriever if retriever is not None else get_retriever()
    try:
        memories = list(
            r.composed_memories(query, user_id=user_id, mark_used=mark_used)
        )
    except Exception:
        logger.debug("composed_memories failed; memory block empty", exc_info=True)
        memories = []

    # Legacy bridge: wrap the legacy rows as ScoredMemory and dedup by text.
    seen_text = {_norm_text(m.text) for m in memories}
    for fact in legacy_facts or ():
        text = getattr(fact, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        norm = _norm_text(text)
        if norm in seen_text:
            continue
        seen_text.add(norm)
        category = getattr(fact, "category", "") or ""
        memories.append(
            ScoredMemory(
                id="legacy:" + norm[:48],
                text=text,
                category=str(category),
                score=0.0,
                vector_similarity=0.0,
                keyword_score=0.0,
                category_match=False,
                record=None,
            )
        )

    return r.format_for_prompt(memories, max_tokens=max_tokens, header=header)


def recover_memories(
    query: str,
    *,
    user_id: str | None = None,
    top_n: int = VECTOR_TOP_K,
) -> list[ScoredMemory]:
    """Module-level convenience: search the full archive for recovery."""
    return get_retriever().recover(query, user_id=user_id, top_n=top_n)
