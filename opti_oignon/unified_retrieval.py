#!/usr/bin/env python3
"""Unified retrieval across project documents, personal memory and the archive.

Three retrieval sources feed one prompt and each one runs its own engine
with its own score scale: project documents go through the hybrid store,
personal memory through its dual-layer retriever, the conversation archive
through its keyword scorer. None of them can see the others, so the same
passage can arrive twice under two labels, and which copy the model reads
is decided by concatenation order rather than by relevance.

This module puts one front in front of all three. Each source is reached
through the callable it was registered under -- the engines themselves are
wrapped, never modified -- and its results are carried in one envelope that
remembers where every snippet came from. Scores are compared through
per-source normalisation (each source's best is worth as much as every
other source's best; the order within a source is the source's own), the
cross-source duplicates are dropped before injection with the higher-ranked
copy surviving byte-for-byte, and the survivors are rendered with a
provenance header so the model can tell a project document from a memory.

An optional local cross-encoder can rerank the survivors. It is governed
twice over: a reranker that is not enabled is never consulted even when it
is present, and one that fails is named in the report while the fused order
stands. When no reranker runs, the report says ``fused-order`` -- absence
is reported, never papered over. Nothing here downloads anything: whether a
local model exists is the operator's decision, made elsewhere.

The layer itself is read-only and stateless: it never writes to any store,
it keeps no persistence of its own, and the conversation archive stays as
untouched here as it is everywhere else. Untrusted content gets no new
treatment on this path -- the layer only selects and orders what the
sources already produced, under the same envelope they produced it in.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .context_dedup import drop_duplicates

logger = logging.getLogger(__name__)

# A checkpoint is taken before any apply-type action. Hardcoded on purpose:
# this is a posture of the codebase, not a configuration surface.
checkpoint_before_apply = True

DEFAULT_PER_SOURCE_LIMIT = 5
DEFAULT_RESULT_LIMIT = 8
FUSED_ORDER = "fused-order"

SourceFn = Callable[[str, int], "list[RetrievedItem]"]
RerankFn = Callable[[str, "list[RetrievedItem]"], "list[RetrievedItem]"]


@dataclass
class RetrievedItem:
    """One snippet as its source produced it, plus stamped provenance."""

    source: str
    content: str
    score: float
    origin: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedRetrievalReport:
    """What one retrieval turn delivered, and on whose word."""

    query: str
    items: list[RetrievedItem]
    per_source: dict[str, int]
    dropped_duplicates: int
    ordering: str
    failures: dict[str, str]
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "items": [
                {
                    "source": it.source,
                    "content": it.content,
                    "score": round(it.score, 4),
                    "origin": it.origin,
                }
                for it in self.items
            ],
            "per_source": dict(self.per_source),
            "dropped_duplicates": self.dropped_duplicates,
            "ordering": self.ordering,
            "failures": dict(self.failures),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def _default_estimate(text: str) -> int:
    """The same rough measure the context zones use elsewhere."""
    return max(1, round(len(text) / 3.7))


class UnifiedRetriever:
    """One entry point in front of every retrieval source."""

    def __init__(
        self,
        sources: dict[str, SourceFn] | None = None,
        *,
        per_source_limit: int = DEFAULT_PER_SOURCE_LIMIT,
        dedup_threshold: float | None = None,
        reranker: tuple[str, RerankFn] | None = None,
        reranker_enabled: bool = False,
    ) -> None:
        """
        Args:
            sources: Mapping of source name to ``fn(query, limit)``. When
                None, the project's own engines are resolved lazily at each
                call; injected sources are used exactly as given.
            per_source_limit: How many items each source is asked for.
            dedup_threshold: Overlap ratio above which a candidate counts
                as already said. None keeps the deduplicator's own default.
            reranker: Optional ``(name, fn)``. The name is what the report
                announces when the reranker actually ran.
            reranker_enabled: Presence is not consent -- a reranker is
                consulted only when this is True.
        """
        self._sources = dict(sources) if sources is not None else None
        self._per_source_limit = max(1, int(per_source_limit))
        self._dedup_threshold = dedup_threshold
        self._reranker = reranker
        self._reranker_enabled = bool(reranker_enabled)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULT_LIMIT,
        conversation_id: str | None = None,
        project_id: str | None = None,
        already_composed: str = "",
    ) -> UnifiedRetrievalReport:
        """Query every source, drop cross-source repeats, order the rest.

        Args:
            query: What the prompt is being assembled for.
            limit: Maximum survivors handed back, after deduplication.
            conversation_id: Scopes the archive source in default mode;
                injected sources receive only the query and the limit.
            project_id: Scopes the document source in default mode.
            already_composed: Text the prompt already carries. A candidate
                that repeats it is dropped even as the sole result.
        """
        start = time.monotonic()
        sources = self._sources
        if sources is None:
            sources = _default_sources(
                conversation_id=conversation_id, project_id=project_id,
            )

        gathered: list[RetrievedItem] = []
        per_source: dict[str, int] = {}
        failures: dict[str, str] = {}

        for name, fn in sources.items():
            try:
                items = list(fn(query, self._per_source_limit))
            except Exception as exc:  # a broken engine costs only itself
                failures[name] = f"{type(exc).__name__}: {exc}"
                continue
            for item in items:
                # Provenance comes from the registry, not from the item: a
                # source cannot sign its material with another's name.
                item.source = name
            per_source[name] = len(items)
            gathered.extend(items)

        ordered = self._fused_order(gathered)

        kept, dropped = drop_duplicates(
            ordered,
            already_composed or "",
            key=lambda it: it.content,
            **(
                {"threshold": self._dedup_threshold}
                if self._dedup_threshold is not None
                else {}
            ),
        )
        kept = kept[: max(0, int(limit))]

        ordering = FUSED_ORDER
        if self._reranker is not None and self._reranker_enabled and kept:
            name, fn = self._reranker
            try:
                kept = list(fn(query, list(kept)))
                ordering = name
            except Exception as exc:  # absence and failure degrade the same way
                failures["reranker"] = f"{type(exc).__name__}: {exc}"
                ordering = FUSED_ORDER

        return UnifiedRetrievalReport(
            query=query,
            items=kept,
            per_source=per_source,
            dropped_duplicates=len(dropped),
            ordering=ordering,
            failures=failures,
            elapsed_ms=(time.monotonic() - start) * 1000.0,
        )

    def _fused_order(self, items: list[RetrievedItem]) -> list[RetrievedItem]:
        """Order across sources without touching any item's own score.

        Each source's scores are divided by that source's best, so every
        source's top is worth 1.0 and the order within a source is the
        source's own. Ties fall to the source's own score first -- when two
        copies of the same passage tie on rank, the one its engine trusted
        more should be the one that survives -- then to source name, origin
        and content, so the order is reproducible either way.
        """
        best: dict[str, float] = {}
        for it in items:
            if it.score > best.get(it.source, 0.0):
                best[it.source] = it.score

        def key(it: RetrievedItem) -> tuple[float, float, str, str, str]:
            top = best.get(it.source, 0.0)
            normalized = (it.score / top) if top > 0 else 0.0
            return (-normalized, -it.score, it.source, it.origin, it.content)

        return sorted(items, key=key)

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def format_for_injection(
        self,
        items: list[RetrievedItem],
        *,
        budget_tokens: int,
        estimate: Callable[[str], int] | None = None,
    ) -> str:
        """Render survivors under a provenance header, inside the budget.

        Items are emitted in the order given until the next one would not
        fit; that one and everything after it are dropped whole. Nothing is
        ever truncated inside an item, because half a snippet reads as a
        whole one.
        """
        measure = estimate or _default_estimate
        remaining = max(0, int(budget_tokens))
        blocks: list[str] = []
        for it in items:
            header = f"[{it.source}: {it.origin}]" if it.origin else f"[{it.source}]"
            block = f"{header}\n{it.content}"
            cost = measure(block)
            if cost > remaining:
                break
            blocks.append(block)
            remaining -= cost
        return "\n\n".join(blocks)


# ----------------------------------------------------------------------
# Default sources: the project's own engines, resolved lazily
# ----------------------------------------------------------------------

_default_compressor: Any = None


def _get_compressor() -> Any:
    """One archive reader for the module; built on first use."""
    global _default_compressor
    if _default_compressor is None:
        from .conversation_compressor import ConversationCompressor

        _default_compressor = ConversationCompressor()
    return _default_compressor


def _default_sources(
    conversation_id: str | None,
    project_id: str | None,
) -> dict[str, SourceFn]:
    """Bind the project's engines behind the (query, limit) shape.

    Every import happens inside the closure that needs it, so an engine
    that is not installed costs exactly one named failure at call time and
    nothing at import time.
    """

    def rag(query: str, limit: int) -> list[RetrievedItem]:
        from .rag_hybrid_search import get_hybrid_engine

        collection = f"project_{project_id}" if project_id else None
        response = get_hybrid_engine().search(
            query, collection=collection, n_results=limit,
        )
        items = []
        for r in response.results:
            origin = r.source_file
            if r.section:
                origin = f"{r.source_file} ({r.section})"
            elif r.total_chunks > 1:
                origin = f"{r.source_file} (chunk {r.chunk_index + 1}/{r.total_chunks})"
            items.append(RetrievedItem(
                source="rag", content=r.content, score=r.fused_score,
                origin=origin, metadata={"chunk_id": r.chunk_id},
            ))
        return items

    def memory(query: str, limit: int) -> list[RetrievedItem]:
        from .memory.retrieval import get_retriever

        return [
            RetrievedItem(
                source="memory", content=m.text, score=m.score,
                origin=m.id, metadata={"category": m.category},
            )
            for m in get_retriever().retrieve(query, top_n=limit)
        ]

    def archive(query: str, limit: int) -> list[RetrievedItem]:
        if not conversation_id:
            return []
        return [
            RetrievedItem(
                source="archive", content=r.snippet, score=r.score,
                origin=f"message {r.message_id} ({r.role})",
                metadata={"timestamp": r.timestamp},
            )
            for r in _get_compressor().retrieve_from_archive(
                conversation_id, query, top_k=limit,
            )
        ]

    return {"rag": rag, "memory": memory, "archive": archive}


# ----------------------------------------------------------------------
# Module singleton
# ----------------------------------------------------------------------

_retriever: UnifiedRetriever | None = None


def get_unified_retriever() -> UnifiedRetriever:
    """The shared front, with the project's own engines behind it."""
    global _retriever
    if _retriever is None:
        _retriever = UnifiedRetriever()
    return _retriever


def register_local_reranker(name: str, fn: RerankFn, *, enabled: bool) -> None:
    """Install a local reranker on the shared front.

    Whether a model exists locally is the operator's decision; this call
    only tells the front what to do with one that does. ``enabled`` is the
    consent switch and it is explicit on purpose.
    """
    front = get_unified_retriever()
    front._reranker = (name, fn)
    front._reranker_enabled = bool(enabled)
