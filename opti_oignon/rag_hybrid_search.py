#!/usr/bin/env python3
"""
RAG HYBRID SEARCH ENGINE -- Vector + Keyword search fusion (S100).

Provides:
- Vector search via existing ChromaDB store (rag_store)
- BM25-style keyword search on stored chunks
- Weighted score fusion (configurable alpha: vector vs keyword)
- Result deduplication across search modes
- Configurable via config/rag.yaml [hybrid_search] section

Usage::

    engine = HybridSearchEngine(store=rag_store_instance)
    results = engine.search(
        query="What is Shannon diversity?",
        collection="papers",
        alpha=0.7,  # 70% vector, 30% keyword
    )
"""

import hashlib
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Feature flag
HYBRID_SEARCH_AVAILABLE = True


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass
class HybridResult:
    """A single hybrid search result with fused score and provenance."""
    content: str
    vector_score: float
    keyword_score: float
    fused_score: float
    source_file: str
    file_type: str
    chunk_index: int
    total_chunks: int
    parent_doc_id: str
    collection_name: str
    chunk_id: str
    section: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    search_mode: str = "hybrid"  # "vector", "keyword", or "hybrid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "vector_score": round(self.vector_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "fused_score": round(self.fused_score, 4),
            "source_file": self.source_file,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "parent_doc_id": self.parent_doc_id,
            "collection_name": self.collection_name,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "page": self.page,
            "search_mode": self.search_mode,
        }


@dataclass
class HybridSearchResponse:
    """Full response from hybrid search."""
    query: str
    results: list[HybridResult]
    total_results: int
    alpha: float
    search_time_ms: float
    vector_count: int
    keyword_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "alpha": round(self.alpha, 2),
            "search_time_ms": round(self.search_time_ms, 2),
            "vector_count": self.vector_count,
            "keyword_count": self.keyword_count,
        }


# =========================================================================
# BM25 KEYWORD SCORING
# =========================================================================

class BM25Scorer:
    """
    Lightweight BM25-style keyword scorer for in-memory chunk scoring.

    Operates on a list of (chunk_id, content) pairs.
    No external dependency; uses standard BM25 formula with
    configurable k1 and b parameters.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tokenize_re = re.compile(r"[a-zA-Z0-9]+")

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + alphanumeric tokenizer."""
        return [t.lower() for t in self._tokenize_re.findall(text)]

    def score_chunks(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[tuple[str, float]]:
        """
        Score chunks against a query using BM25.

        Args:
            query: Search query string.
            chunks: List of dicts with at least 'chunk_id' and 'content' keys.

        Returns:
            List of (chunk_id, bm25_score) sorted descending by score.
        """
        if not chunks or not query.strip():
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Tokenize all documents
        doc_tokens: list[list[str]] = []
        for chunk in chunks:
            doc_tokens.append(self._tokenize(chunk.get("content", "")))

        n_docs = len(doc_tokens)
        avg_dl = sum(len(dt) for dt in doc_tokens) / max(1, n_docs)

        # Compute IDF for query terms
        idf: dict[str, float] = {}
        for term in set(query_terms):
            df = sum(1 for dt in doc_tokens if term in dt)
            # Standard BM25 IDF with smoothing
            idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # Score each document
        scores: list[tuple[str, float]] = []
        for i, chunk in enumerate(chunks):
            tokens = doc_tokens[i]
            dl = len(tokens)
            if dl == 0:
                scores.append((chunk["chunk_id"], 0.0))
                continue

            # Term frequency map
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            score = 0.0
            for term in query_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                score += idf.get(term, 0.0) * numerator / denominator

            scores.append((chunk["chunk_id"], score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def normalize_scores(
        self,
        scores: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Normalize BM25 scores to 0-1 range using min-max scaling."""
        if not scores:
            return []

        vals = [s for _, s in scores]
        min_val = min(vals)
        max_val = max(vals)
        spread = max_val - min_val

        if spread < 1e-9:
            # All scores identical: assign 0.5 if non-zero, else 0.0
            return [
                (cid, 0.5 if s > 0 else 0.0)
                for cid, s in scores
            ]

        return [
            (cid, (s - min_val) / spread)
            for cid, s in scores
        ]


# =========================================================================
# HYBRID SEARCH ENGINE
# =========================================================================

class HybridSearchEngine:
    """
    Combines vector search (ChromaDB) with keyword search (BM25)
    using weighted score fusion.

    The ``alpha`` parameter controls the blend:
    - alpha=1.0  -> pure vector search
    - alpha=0.0  -> pure keyword search
    - alpha=0.7  -> 70% vector + 30% keyword (default)
    """

    DEFAULT_ALPHA = 0.7
    DEFAULT_N_RESULTS = 5

    def __init__(
        self,
        store: Any = None,
        alpha: float | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Args:
            store: RAGVectorStore instance (lazy-loaded if None).
            alpha: Default vector weight (0.0 to 1.0).
            bm25_k1: BM25 k1 parameter (term frequency saturation).
            bm25_b: BM25 b parameter (document length normalization).
        """
        self._store = store
        self._alpha = alpha if alpha is not None else self.DEFAULT_ALPHA
        self._bm25 = BM25Scorer(k1=bm25_k1, b=bm25_b)
        self._config: dict[str, Any] | None = None

    @property
    def store(self):
        """Lazy-load the RAG store."""
        if self._store is None:
            try:
                from opti_oignon.rag_store import get_rag_store
                self._store = get_rag_store()
            except ImportError:
                import importlib.util
                import sys
                spec = importlib.util.spec_from_file_location(
                    "rag_store",
                    Path(__file__).parent / "rag_store.py",
                )
                mod = importlib.util.module_from_spec(spec)
                sys.modules["rag_store"] = mod  # Python 3.13: register before exec_module for dataclass safety
                spec.loader.exec_module(mod)
                self._store = mod.get_rag_store()
        return self._store

    def _load_config(self) -> dict[str, Any]:
        """Load hybrid search config from rag.yaml."""
        if self._config is not None:
            return self._config

        defaults: dict[str, Any] = {
            "alpha": self.DEFAULT_ALPHA,
            "n_results": self.DEFAULT_N_RESULTS,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "min_score": 0.1,
            "keyword_fetch_multiplier": 5,
        }
        try:
            import yaml
            config_path = Path(__file__).parent / "config" / "rag.yaml"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                hybrid_cfg = cfg.get("hybrid_search", {})
                if isinstance(hybrid_cfg, dict):
                    defaults.update(hybrid_cfg)
        except Exception as exc:
            logger.debug("Could not load hybrid_search config: %s", exc)

        self._config = defaults

        # RHS-03: honour the documented config/rag.yaml [hybrid_search] bm25_k1
        # / bm25_b keys. They were loaded into `defaults` but never applied to
        # the scorer (built in __init__ from constructor args), so a non-default
        # value in rag.yaml was silently ignored. Sync them here. Behaviour is
        # unchanged when the config matches the defaults (1.5 / 0.75).
        try:
            self._bm25.k1 = float(defaults.get("bm25_k1", self._bm25.k1))
            self._bm25.b = float(defaults.get("bm25_b", self._bm25.b))
        except (TypeError, ValueError):
            logger.debug("Invalid bm25_k1/bm25_b in config -- keeping current values")

        return defaults

    # -----------------------------------------------------------------
    # MAIN SEARCH
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        collection: str | None = None,
        n_results: int | None = None,
        alpha: float | None = None,
        min_score: float | None = None,
        source_filter: str | None = None,
        file_type_filter: str | None = None,
        track_citations: bool = True,
    ) -> HybridSearchResponse:
        """
        Perform hybrid search combining vector and keyword scoring.

        Args:
            query: Search query text.
            collection: Target collection (None = default).
            n_results: Max results to return.
            alpha: Vector weight override (0.0 to 1.0).
            min_score: Minimum fused score threshold.
            source_filter: Filter by source file path.
            file_type_filter: Filter by file type.
            track_citations: Log citations to SQLite.

        Returns:
            HybridSearchResponse with fused results.
        """
        start = time.time()
        cfg = self._load_config()

        collection = collection or "default"
        n_results = n_results or cfg.get("n_results", self.DEFAULT_N_RESULTS)
        alpha = alpha if alpha is not None else cfg.get("alpha", self._alpha)
        min_score = min_score if min_score is not None else cfg.get("min_score", 0.1)

        # Clamp alpha
        alpha = max(0.0, min(1.0, alpha))

        # Step 1: Vector search (via existing store)
        vector_results = self._vector_search(
            query=query,
            collection=collection,
            n_results=n_results * 3,
            source_filter=source_filter,
            file_type_filter=file_type_filter,
        )

        # Step 2: Keyword search (BM25 on stored chunks)
        keyword_results = self._keyword_search(
            query=query,
            collection=collection,
            n_results=n_results * cfg.get("keyword_fetch_multiplier", 5),
            source_filter=source_filter,
            file_type_filter=file_type_filter,
        )

        # Step 3: Fuse scores
        fused = self._fuse_scores(
            vector_results=vector_results,
            keyword_results=keyword_results,
            alpha=alpha,
        )

        # Step 4: Filter by min_score and trim
        filtered = [r for r in fused if r.fused_score >= min_score]
        filtered = filtered[:n_results]

        # Step 5: Track citations if requested
        if track_citations and filtered:
            self._track_citations(query, collection, filtered)

        elapsed_ms = (time.time() - start) * 1000

        return HybridSearchResponse(
            query=query,
            results=filtered,
            total_results=len(filtered),
            alpha=alpha,
            search_time_ms=elapsed_ms,
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
        )

    # -----------------------------------------------------------------
    # VECTOR SEARCH
    # -----------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        collection: str,
        n_results: int,
        source_filter: str | None = None,
        file_type_filter: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Run vector search via the RAG store and return a dict
        mapping chunk_id -> {score, content, metadata}.
        """
        results: dict[str, dict[str, Any]] = {}

        try:
            store = self.store
            if store is None:
                return results

            response = store.query(
                query_text=query,
                collection=collection,
                n_results=n_results,
                min_score=0.0,  # We filter later with fused score
                source_filter=source_filter,
                file_type_filter=file_type_filter,
                rerank=False,  # We do our own reranking
                track_citations=False,  # We track in the hybrid layer
            )

            for r in response.results:
                cid = self._chunk_key(r.parent_doc_id, r.chunk_index)
                results[cid] = {
                    "score": r.score,
                    "content": r.content,
                    "source_file": r.source_file,
                    "file_type": r.file_type,
                    "chunk_index": r.chunk_index,
                    "total_chunks": r.total_chunks,
                    "parent_doc_id": r.parent_doc_id,
                    "collection_name": r.collection_name,
                    "section": r.section,
                    "page": r.page,
                    "metadata": r.metadata,
                }

        except Exception as exc:
            logger.error("Vector search failed: %s", exc)

        return results

    # -----------------------------------------------------------------
    # KEYWORD SEARCH (BM25)
    # -----------------------------------------------------------------

    def _keyword_search(
        self,
        query: str,
        collection: str,
        n_results: int,
        source_filter: str | None = None,
        file_type_filter: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Run BM25 keyword search on stored chunks.

        Fetches all chunks from the ChromaDB collection, scores them
        with BM25, and returns top results as a dict mapping
        chunk_id -> {score, content, metadata}.
        """
        results: dict[str, dict[str, Any]] = {}

        try:
            store = self.store
            if store is None or store._chroma is None:
                return results

            # Fetch chunks from ChromaDB
            try:
                coll = store._chroma.get_collection(collection)
            except Exception:
                return results

            count = coll.count()
            if count == 0:
                return results

            # Build where filter
            where_filter = store._build_where(source_filter, file_type_filter)

            # Fetch all chunks (limited to a reasonable batch)
            fetch_kwargs: dict[str, Any] = {
                "include": ["documents", "metadatas"],
            }
            if where_filter:
                fetch_kwargs["where"] = where_filter

            raw = coll.get(**fetch_kwargs)

            if not raw or not raw.get("documents"):
                return results

            ids = raw["ids"]
            docs = raw["documents"]
            metas = raw["metadatas"]

            # Build chunk list for BM25
            chunks: list[dict[str, Any]] = []
            for i, (cid, doc, meta) in enumerate(zip(ids, docs, metas)):
                if doc:
                    chunks.append({
                        "chunk_id": cid,
                        "content": doc,
                        "meta": meta or {},
                    })

            if not chunks:
                return results

            # Score with BM25
            raw_scores = self._bm25.score_chunks(query, chunks)
            norm_scores = self._bm25.normalize_scores(raw_scores)

            # Build a lookup for normalized scores
            score_map = dict(norm_scores)  # noqa: F841

            # Build a chunk lookup
            chunk_lookup = {c["chunk_id"]: c for c in chunks}

            # Take top n_results
            for cid, score in norm_scores[:n_results]:
                if score <= 0:
                    continue
                chunk = chunk_lookup.get(cid)
                if not chunk:
                    continue
                meta = chunk.get("meta", {})
                doc_id = meta.get("parent_doc_id", "")
                chunk_idx = meta.get("chunk_index", 0)
                key = self._chunk_key(doc_id, chunk_idx)

                results[key] = {
                    "score": score,
                    "content": chunk["content"],
                    "source_file": meta.get("source_file", ""),
                    "file_type": meta.get("file_type", ""),
                    "chunk_index": chunk_idx,
                    "total_chunks": meta.get("total_chunks", 0),
                    "parent_doc_id": doc_id,
                    "collection_name": collection,
                    "section": meta.get("section") or None,
                    "page": meta.get("page", -1),
                    "metadata": meta,
                }

        except Exception as exc:
            logger.error("Keyword search failed: %s", exc)

        return results

    # -----------------------------------------------------------------
    # SCORE FUSION
    # -----------------------------------------------------------------

    def _fuse_scores(
        self,
        vector_results: dict[str, dict[str, Any]],
        keyword_results: dict[str, dict[str, Any]],
        alpha: float,
    ) -> list[HybridResult]:
        """
        Fuse vector and keyword scores using weighted combination.

        fused_score = alpha * vector_score + (1 - alpha) * keyword_score

        Deduplicates by chunk_id, keeping the merged entry.
        """
        # Collect all unique chunk keys. Sorted, because the iteration order
        # of a set of keys differs from one process to the next: chunks
        # carrying equal scores would be handed back in a different order on
        # every host, and any threshold measured on this engine would drift
        # with it. The sort below is stable, so the score still decides and
        # the chunk id only breaks ties.
        all_keys = sorted(set(vector_results.keys()) | set(keyword_results.keys()))
        fused: list[HybridResult] = []

        for key in all_keys:
            v_entry = vector_results.get(key)
            k_entry = keyword_results.get(key)

            v_score = v_entry["score"] if v_entry else 0.0
            k_score = k_entry["score"] if k_entry else 0.0

            # Determine search mode
            if v_entry and k_entry:
                mode = "hybrid"
            elif v_entry:
                mode = "vector"
            else:
                mode = "keyword"

            # Fused score
            fused_score = alpha * v_score + (1 - alpha) * k_score

            # Use whichever entry has data for metadata
            entry = v_entry or k_entry

            page_val = entry.get("page", -1)
            if isinstance(page_val, int) and page_val < 0:
                page_val = None

            fused.append(HybridResult(
                content=entry["content"],
                vector_score=v_score,
                keyword_score=k_score,
                fused_score=fused_score,
                source_file=entry.get("source_file", ""),
                file_type=entry.get("file_type", ""),
                chunk_index=entry.get("chunk_index", 0),
                total_chunks=entry.get("total_chunks", 0),
                parent_doc_id=entry.get("parent_doc_id", ""),
                collection_name=entry.get("collection_name", ""),
                chunk_id=key,
                section=entry.get("section"),
                page=page_val,
                metadata=entry.get("metadata", {}),
                search_mode=mode,
            ))

        # Sort by fused score descending
        fused.sort(key=lambda r: r.fused_score, reverse=True)
        return fused

    # -----------------------------------------------------------------
    # CITATION TRACKING
    # -----------------------------------------------------------------

    def _track_citations(
        self,
        query: str,
        collection: str,
        results: list[HybridResult],
    ) -> None:
        """Log citations to the RAG store SQLite database."""
        try:
            store = self.store
            if store is None:
                return

            now = time.time()
            for r in results:
                try:
                    from opti_oignon.rag_store import CitationRecord
                except ImportError:
                    return

                cid = uuid.uuid4().hex[:12]
                citation = CitationRecord(
                    citation_id=cid,
                    query=query,
                    collection_name=collection,
                    chunk_id=r.chunk_id,
                    parent_doc_id=r.parent_doc_id,
                    source_file=r.source_file,
                    section=r.section,
                    score=r.fused_score,
                    timestamp=now,
                )
                store.db.insert_citation(citation)
        except Exception as exc:
            logger.debug("Hybrid citation tracking error: %s", exc)

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    @staticmethod
    def _chunk_key(parent_doc_id: str, chunk_index: int) -> str:
        """Generate a unique key for deduplication."""
        return hashlib.sha256(
            f"{parent_doc_id}::{chunk_index}".encode()
        ).hexdigest()[:16]

    def get_config(self) -> dict[str, Any]:
        """Return the current hybrid search configuration."""
        cfg = self._load_config()
        return {
            **cfg,
            "alpha": self._alpha,
            "bm25_k1": self._bm25.k1,
            "bm25_b": self._bm25.b,
        }


# =========================================================================
# MODULE-LEVEL SINGLETON
# =========================================================================

_hybrid_engine: HybridSearchEngine | None = None


def get_hybrid_engine(
    store: Any = None,
    alpha: float | None = None,
) -> HybridSearchEngine:
    """Return the module-level HybridSearchEngine singleton."""
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridSearchEngine(store=store, alpha=alpha)
    return _hybrid_engine
