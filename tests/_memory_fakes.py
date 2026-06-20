"""Deterministic in-memory ChromaDB collection fake for S173 tests.

Not a test module (no ``test_`` prefix, so pytest does not collect it). It
mirrors the subset of the real ChromaDB Collection API used by
``opti_oignon/memory/vector_store.py`` -- ``add``, ``upsert``, ``get``,
``query``, ``delete``, ``count`` -- and implements genuine cosine semantics so
threshold logic (dedup at cosine 0.92, retrieval similarity) is exercised for
real, without installing chromadb in the sandbox. Cosine distance matches
ChromaDB's ``hnsw:space=cosine`` convention: ``distance = 1 - cosine_similarity``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FakeChromaCollection:
    def __init__(self, name: str = "oo_memories", metadata: dict | None = None):
        self.name = name
        self.metadata = dict(metadata or {})
        self._ids: list[str] = []
        self._data: dict[str, dict[str, Any]] = {}

    # Mutations

    def add(self, ids, embeddings, documents=None, metadatas=None):
        for i, _id in enumerate(ids):
            self._store(
                _id,
                embeddings[i] if embeddings is not None else None,
                documents[i] if documents is not None else None,
                metadatas[i] if metadatas is not None else None,
            )

    def upsert(self, ids, embeddings=None, documents=None, metadatas=None):
        for i, _id in enumerate(ids):
            prev = self._data.get(_id, {})
            emb = embeddings[i] if embeddings is not None else prev.get("embedding")
            doc = documents[i] if documents is not None else prev.get("document")
            md = metadatas[i] if metadatas is not None else prev.get("metadata")
            self._store(_id, emb, doc, md, _is_array=embeddings is None)

    def delete(self, ids=None, where=None):
        if ids is not None:
            targets = [i for i in ids if i in self._data]
        elif where is not None:
            targets = [i for i in self._ids if self._match(self._data[i]["metadata"], where)]
        else:
            targets = list(self._ids)
        for i in targets:
            self._data.pop(i, None)
            if i in self._ids:
                self._ids.remove(i)

    # Reads

    def get(self, ids=None, where=None, include=None):
        include = include or []
        selection = ids if ids is not None else list(self._ids)
        out_ids: list[str] = []
        out_docs: list[Any] = []
        out_meta: list[Any] = []
        out_emb: list[Any] = []
        for _id in selection:
            rec = self._data.get(_id)
            if rec is None:
                continue
            if where and not self._match(rec["metadata"], where):
                continue
            out_ids.append(_id)
            out_docs.append(rec["document"])
            out_meta.append(dict(rec["metadata"]))
            emb = rec["embedding"]
            out_emb.append(emb.tolist() if emb is not None else None)
        result: dict[str, Any] = {"ids": out_ids}
        want_docs = "documents" in include or not include
        want_meta = "metadatas" in include or not include
        result["documents"] = out_docs if want_docs else None
        result["metadatas"] = out_meta if want_meta else None
        result["embeddings"] = out_emb if "embeddings" in include else None
        return result

    def query(self, query_embeddings, n_results=5, where=None, include=None):
        q = np.asarray(query_embeddings[0], dtype=float)
        scored = []
        for _id in self._ids:
            rec = self._data[_id]
            if where and not self._match(rec["metadata"], where):
                continue
            if rec["embedding"] is None:
                continue
            scored.append((self._cosine_distance(q, rec["embedding"]), _id, rec))
        scored.sort(key=lambda t: t[0])
        scored = scored[: max(0, int(n_results))]
        return {
            "ids": [[s[1] for s in scored]],
            "distances": [[s[0] for s in scored]],
            "documents": [[s[2]["document"] for s in scored]],
            "metadatas": [[dict(s[2]["metadata"]) for s in scored]],
        }

    def count(self):
        return len(self._ids)

    # Helpers

    def _store(self, _id, embedding, document, metadata, _is_array=False):
        if _id not in self._data:
            self._ids.append(_id)
        arr = None
        if embedding is not None:
            arr = embedding if isinstance(embedding, np.ndarray) else np.asarray(embedding, dtype=float)
        self._data[_id] = {
            "embedding": arr,
            "document": document,
            "metadata": dict(metadata or {}),
        }

    @staticmethod
    def _match(md: dict, where: dict) -> bool:
        for key, val in where.items():
            if isinstance(val, dict):
                if "$eq" in val and md.get(key) != val["$eq"]:
                    return False
            elif md.get(key) != val:
                return False
        return True

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 1.0
        sim = float(np.dot(a, b) / (na * nb))
        return 1.0 - sim


class FakeEmbedder:
    """Deterministic stand-in for the shared RAG embedding client.

    Mirrors the ``embed_single`` / ``embed`` surface used by the vector layer.
    Mapped texts return their fixed vector (so a test can craft a near-duplicate
    that trips the cosine stage without tripping Jaccard); unmapped texts get a
    stable pseudo-random unit vector, near-orthogonal to others, so distinct
    facts do not accidentally collide above the 0.92 threshold.
    """

    def __init__(self, mapping: dict | None = None, dim: int = 16):
        self._mapping = {k: list(v) for k, v in (mapping or {}).items()}
        self._dim = dim

    def embed_single(self, text):
        if text in self._mapping:
            return list(self._mapping[text])
        import hashlib
        import random

        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self._dim)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed(self, texts):
        return [self.embed_single(t) for t in texts]

    def embed_batch(self, texts):
        return self.embed(texts)
