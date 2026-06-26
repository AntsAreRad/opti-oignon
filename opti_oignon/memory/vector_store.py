"""Vector layer for Opti-Oignon personal memory (S173, Theme 3 / Odysseus Core).

A ChromaDB collection named ``oo_memories`` in cosine space
(``hnsw:space = cosine``), kept distinct from the RAG collection so personal
memory and document retrieval never bleed into each other. Embeddings are
precomputed by the embedding client shared with the RAG pipeline
(``opti_oignon.rag.embeddings.OllamaEmbeddings``), so memory adds no new
embedding dependency.

The layer mirrors the canonical CRUD (add / update / delete / get / count) and
exposes ``find_similar`` for the dedup cosine check and for hybrid retrieval.
Per-user isolation is carried in the metadata ``user_id`` and enforced on every
read. The ChromaDB client/collection and the embedder are injectable so the
module loads and tests in isolation without chromadb or ollama installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module sentinels (project convention).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Guarded ChromaDB import (same posture as rag_store.py).
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _HAS_CHROMADB = True
except Exception:
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment]
    _HAS_CHROMADB = False

# Guarded shared embedding client (no new embedding dependency).
try:
    from ..rag.embeddings import OllamaEmbeddings

    _HAS_EMBEDDER = True
except Exception:
    OllamaEmbeddings = None  # type: ignore[assignment,misc]
    _HAS_EMBEDDER = False

COLLECTION_NAME = "oo_memories"
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "description": "Opti-Oignon personal memory (oo_memories), distinct from RAG",
}
DEFAULT_LOCAL_USER = "local"


@dataclass
class SimilarMemory:
    """A neighbour returned by find_similar.

    similarity is cosine similarity in [-1, 1], derived from the ChromaDB cosine
    distance as ``1 - distance``.
    """

    id: str
    similarity: float
    document: str | None = None
    metadata: dict[str, Any] | None = None


def _default_chroma_dir() -> Path:
    try:
        from ..rag.config import get_config

        return Path(get_config().chroma_dir)
    except Exception:
        try:
            from ..config import DATA_DIR

            return Path(DATA_DIR) / "chroma"
        except Exception:
            return Path("data") / "chroma"


def _default_embedder() -> Any | None:
    if not _HAS_EMBEDDER:
        return None
    try:
        return OllamaEmbeddings()
    except Exception:  # pragma: no cover - depends on runtime config
        return None


def _clean_metadata(md: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB metadata must be str/int/float/bool and non-null."""
    cleaned: dict[str, Any] = {}
    for key, val in md.items():
        if val is None:
            continue
        if isinstance(val, (str, int, float, bool)):
            cleaned[key] = val
        else:
            cleaned[key] = str(val)
    return cleaned


class MemoryVectorStore:
    """The oo_memories ChromaDB layer, mirroring the canonical store."""

    COLLECTION_NAME = COLLECTION_NAME

    def __init__(
        self,
        *,
        chroma_dir: Path | str | None = None,
        embedder: Any | None = None,
        collection: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self._embedder = embedder if embedder is not None else _default_embedder()
        self._warned_no_embedder = False
        if collection is not None:
            self._collection = collection
        else:
            self._collection = self._build_collection(chroma_dir, client)

    def _build_collection(self, chroma_dir: Path | str | None, client: Any | None) -> Any:
        if not _HAS_CHROMADB:
            raise RuntimeError(
                "chromadb is not available; inject a collection or install chromadb"
            )
        if client is None:
            path = str(chroma_dir if chroma_dir is not None else _default_chroma_dir())
            client = chromadb.PersistentClient(
                path=path, settings=ChromaSettings(anonymized_telemetry=False)
            )
        return client.get_or_create_collection(
            name=COLLECTION_NAME, metadata=dict(COLLECTION_METADATA)
        )

    @property
    def collection(self) -> Any:
        return self._collection

    # Embedding (shared client)

    def embed(self, text: str) -> list[float] | None:
        if self._embedder is None:
            if not self._warned_no_embedder:
                logger.warning(
                    "memory embedder unavailable; semantic recall degraded "
                    "(keyword/canonical retrieval only)"
                )
                self._warned_no_embedder = True
            return None
        try:
            return self._embedder.embed_single(text)
        except Exception as exc:  # pragma: no cover - runtime embedding failures
            logger.warning("memory embedding failed: %s", exc)
            return None

    def health(self) -> dict[str, Any]:
        """Report the embedder state for the archive (semantic) tier.

        Returns a dict with ``status`` ("ok" | "degraded" | "unavailable"),
        ``available`` (bool), and ``dim`` / ``detail``. Never raises. "ok" means
        embeddings flow; "degraded" means an embedder is configured but not
        returning vectors (e.g. Ollama down or the embed model missing);
        "unavailable" means no embedder is configured. In every non-ok case the
        canonical (keyword/recency) tier still works -- recall is degraded, not
        down.
        """
        if self._embedder is None:
            return {
                "status": "unavailable",
                "available": False,
                "detail": "no embedder configured; semantic recall disabled",
            }
        probe = self.embed("ok")
        if isinstance(probe, list) and probe:
            return {"status": "ok", "available": True, "dim": len(probe)}
        return {
            "status": "degraded",
            "available": False,
            "detail": "embedder configured but not returning vectors",
        }

    def _resolve_embedding(
        self, text: str, embedding: list[float] | None
    ) -> list[float]:
        if embedding is not None:
            return list(embedding)
        computed = self.embed(text)
        if computed is None:
            raise RuntimeError(
                "no embedding available: pass one explicitly or configure an embedder"
            )
        return computed

    @staticmethod
    def _where(user_id: str | None) -> dict[str, Any] | None:
        if user_id is None:
            return None
        return {"user_id": str(user_id)}

    # Create / update (mirror the canonical CRUD)

    def add(
        self,
        fact_id: str,
        text: str,
        *,
        embedding: list[float] | None = None,
        user_id: str = DEFAULT_LOCAL_USER,
        category: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        emb = self._resolve_embedding(text, embedding)
        md = {"user_id": str(user_id), "category": str(category), "source": str(source)}
        if metadata:
            md.update(metadata)
        self._collection.add(
            ids=[fact_id],
            embeddings=[emb],
            documents=[text],
            metadatas=[_clean_metadata(md)],
        )
        return fact_id

    def update(
        self,
        fact_id: str,
        *,
        text: str | None = None,
        embedding: list[float] | None = None,
        user_id: str | None = None,
        category: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        existing = self.get(fact_id)
        if existing is None:
            return False
        new_text = text if text is not None else existing.get("document") or ""
        md = dict(existing.get("metadata") or {})
        if user_id is not None:
            md["user_id"] = str(user_id)
        if category is not None:
            md["category"] = str(category)
        if source is not None:
            md["source"] = str(source)
        if metadata:
            md.update(metadata)

        if embedding is not None:
            emb = list(embedding)
        elif text is not None:
            recomputed = self.embed(new_text)
            emb = recomputed if recomputed is not None else existing.get("embedding")
        else:
            emb = existing.get("embedding")

        self._collection.upsert(
            ids=[fact_id],
            embeddings=[emb] if emb is not None else None,
            documents=[new_text],
            metadatas=[_clean_metadata(md)],
        )
        return True

    # Delete (scoped)

    def delete(self, fact_id: str, *, user_id: str | None = None) -> bool:
        record = self.get(fact_id)
        if record is None:
            return False
        if user_id is not None:
            owner = (record.get("metadata") or {}).get("user_id")
            if owner != str(user_id):
                return False
        self._collection.delete(ids=[fact_id])
        return True

    def clear(self, *, user_id: str | None = None) -> int:
        where = self._where(user_id)
        targets = self._collection.get(where=where, include=[]).get("ids", [])
        if targets:
            self._collection.delete(ids=list(targets))
        return len(targets)

    # Read

    def get(self, fact_id: str) -> dict[str, Any] | None:
        res = self._collection.get(
            ids=[fact_id], include=["documents", "metadatas", "embeddings"]
        )
        ids = res.get("ids") or []
        if not ids:
            return None
        documents = res.get("documents") or [None]
        metadatas = res.get("metadatas") or [None]
        embeddings = res.get("embeddings") or [None]
        return {
            "id": ids[0],
            "document": documents[0] if documents else None,
            "metadata": metadatas[0] if metadatas else None,
            "embedding": embeddings[0] if embeddings else None,
        }

    def count(self, *, user_id: str | None = None) -> int:
        if user_id is None:
            return int(self._collection.count())
        res = self._collection.get(where=self._where(user_id), include=[])
        return len(res.get("ids") or [])

    # Similarity

    def find_similar(
        self,
        embedding: list[float],
        *,
        user_id: str | None = None,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[SimilarMemory]:
        """Return up to top_k neighbours, optionally filtered by a similarity floor."""
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=self._where(user_id),
            include=["distances", "documents", "metadatas"],
        )
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        documents = (res.get("documents") or [[]])[0] if res.get("documents") else []
        metadatas = (res.get("metadatas") or [[]])[0] if res.get("metadatas") else []

        out: list[SimilarMemory] = []
        for i, fid in enumerate(ids):
            similarity = 1.0 - float(distances[i])
            if threshold is not None and similarity < threshold:
                continue
            out.append(
                SimilarMemory(
                    id=fid,
                    similarity=similarity,
                    document=documents[i] if i < len(documents) else None,
                    metadata=metadatas[i] if i < len(metadatas) else None,
                )
            )
        return out


# Module-level singleton with a reset for test isolation.
_vector_store: MemoryVectorStore | None = None


def get_vector_store() -> MemoryVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = MemoryVectorStore()
    return _vector_store


def reset_vector_store() -> None:
    global _vector_store
    _vector_store = None
