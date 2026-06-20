"""opti_oignon.memory -- personal memory package (S173, Theme 3 / Odysseus Core).

A two-tier personal-memory store, aligned with the S66 dual-layer principle:

- ``canonical_store`` -- the SQLite WAL source of truth.
- ``vector_store``    -- the ``oo_memories`` ChromaDB layer (added in S173).
- ``dedup``           -- double deduplication (Jaccard then cosine).
- ``retrieval``       -- hybrid retrieval under the context-window budget.

Fold-in (resolved in S173): the former ``opti_oignon/memory.py`` and its
``MemoryManager`` / ``MemoryFact`` were relocated verbatim into
``opti_oignon/memory/legacy.py``. This ``__init__`` is the single compatibility
seam: it re-exports the legacy public surface so existing imports such as
``from opti_oignon.memory import MemoryManager`` keep working unchanged, and it
exposes the new package layer alongside it. The decision is recorded in
ODYSSEUS_SPEC.md Section 4.1.
"""

from __future__ import annotations

# New canonical store (S173).
from .canonical_store import (
    CATEGORIES,
    CanonicalMemoryStore,
    MemoryRecord,
    get_canonical_store,
    reset_canonical_store,
)

# New vector layer (S173).
from .vector_store import (
    COLLECTION_NAME,
    MemoryVectorStore,
    SimilarMemory,
    get_vector_store,
    reset_vector_store,
)

# New deduplication and coordinated CRUD (S173).
from .dedup import (
    COSINE_THRESHOLD,
    JACCARD_THRESHOLD,
    DedupDecision,
    MemoryDeduplicator,
    MemoryStore,
    get_memory_store,
    jaccard_similarity,
    reset_memory_store,
)

# New hybrid retrieval (S173) and S66 dual-layer assembly (S174).
from .retrieval import (
    MEMORY_TOKEN_BUDGET,
    DualLayerMemory,
    MemoryRetriever,
    QueryAnalysis,
    ScoredMemory,
    detect_query_type,
    get_retriever,
    recover_memories,
    reset_retriever,
    working_memory_block,
)

# Backward-compatible re-export of the folded-in legacy manager. Guarded so the
# package still imports (exposing the new layer) if the legacy import chain is
# unavailable in a constrained environment.
try:
    from .legacy import (
        OLLAMA_AVAILABLE,
        MemoryFact,
        MemoryManager,
        add_fact,
        extract_and_store,
        extract_facts,
        get_all_facts,
        memory_manager,
    )

    _LEGACY_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments only
    _LEGACY_AVAILABLE = False
    OLLAMA_AVAILABLE = False
    MemoryFact = None  # type: ignore[assignment,misc]
    MemoryManager = None  # type: ignore[assignment,misc]
    memory_manager = None  # type: ignore[assignment]
    add_fact = None  # type: ignore[assignment]
    extract_and_store = None  # type: ignore[assignment]
    extract_facts = None  # type: ignore[assignment]
    get_all_facts = None  # type: ignore[assignment]

__all__ = [
    # New layer.
    "CATEGORIES",
    "CanonicalMemoryStore",
    "MemoryRecord",
    "get_canonical_store",
    "reset_canonical_store",
    "COLLECTION_NAME",
    "MemoryVectorStore",
    "SimilarMemory",
    "get_vector_store",
    "reset_vector_store",
    "MemoryStore",
    "MemoryDeduplicator",
    "DedupDecision",
    "jaccard_similarity",
    "JACCARD_THRESHOLD",
    "COSINE_THRESHOLD",
    "get_memory_store",
    "reset_memory_store",
    "MemoryRetriever",
    "ScoredMemory",
    "QueryAnalysis",
    "detect_query_type",
    "MEMORY_TOKEN_BUDGET",
    "DualLayerMemory",
    "working_memory_block",
    "recover_memories",
    "get_retriever",
    "reset_retriever",
    # Legacy compatibility surface.
    "MemoryManager",
    "MemoryFact",
    "memory_manager",
    "OLLAMA_AVAILABLE",
    "add_fact",
    "extract_and_store",
    "extract_facts",
    "get_all_facts",
]
