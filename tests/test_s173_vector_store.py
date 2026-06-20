#!/usr/bin/env python3
"""Tests for S173 -- the oo_memories ChromaDB vector layer.

Runtime checks load ``opti_oignon/memory/vector_store.py`` in isolation and
inject a deterministic cosine collection (``tests/_memory_fakes.py``), so the
cosine threshold logic is exercised for real without installing chromadb.
File-content checks assert the real wiring: the ``oo_memories`` collection name,
``hnsw:space = cosine``, the RAG-shared embedding client, and the persistent
client. Distinctness from the RAG collection is asserted via the collection
metadata note and the spec.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
MEM = OO / "memory"
VS_PATH = MEM / "vector_store.py"
SPEC = ROOT / "ODYSSEUS_SPEC.md"

sys.path.insert(0, str(ROOT / "tests"))
from _memory_fakes import FakeChromaCollection  # noqa: E402


def _load_vector_store():
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.memory" not in sys.modules:
        mpkg = types.ModuleType("opti_oignon.memory")
        mpkg.__path__ = [str(MEM)]
        sys.modules["opti_oignon.memory"] = mpkg
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.memory.vector_store", str(VS_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.memory.vector_store"] = mod
    spec.loader.exec_module(mod)
    return mod


vs = _load_vector_store()


def _fresh_collection():
    return FakeChromaCollection(name=vs.COLLECTION_NAME, metadata=dict(vs.COLLECTION_METADATA))


@pytest.fixture
def store():
    return vs.MemoryVectorStore(collection=_fresh_collection())


# Module sentinels


class TestModuleSentinels:
    def test_feature_available(self):
        assert vs.FEATURE_AVAILABLE is True

    def test_checkpoint_sentinel(self):
        assert vs.checkpoint_before_apply is True

    def test_singleton_helpers(self):
        assert hasattr(vs, "get_vector_store")
        assert hasattr(vs, "reset_vector_store")


# Collection wiring


class TestCollectionWiring:
    def test_collection_name(self):
        assert vs.COLLECTION_NAME == "oo_memories"

    def test_cosine_space_metadata(self):
        assert vs.COLLECTION_METADATA["hnsw:space"] == "cosine"

    def test_metadata_notes_distinct_from_rag(self):
        assert "RAG" in vs.COLLECTION_METADATA["description"]

    def test_shared_embedder_referenced(self):
        text = VS_PATH.read_text(encoding="utf-8")
        assert "rag.embeddings" in text
        assert "OllamaEmbeddings" in text

    def test_persistent_client_referenced(self):
        text = VS_PATH.read_text(encoding="utf-8")
        assert "PersistentClient" in text

    def test_get_or_create_with_cosine(self):
        text = VS_PATH.read_text(encoding="utf-8")
        assert "get_or_create_collection" in text
        assert "hnsw:space" in text

    def test_build_requires_chromadb_when_absent(self):
        if vs._HAS_CHROMADB:
            pytest.skip("chromadb installed; default constructor builds a real collection")
        with pytest.raises(RuntimeError):
            vs.MemoryVectorStore()


# CRUD mirror


class TestCrudMirror:
    def test_add_and_get(self, store):
        store.add("id1", "a fact", embedding=[1.0, 0.0, 0.0], category="fact")
        rec = store.get("id1")
        assert rec is not None and rec["document"] == "a fact"

    def test_add_stores_metadata(self, store):
        store.add(
            "id1",
            "pref",
            embedding=[1.0, 0.0, 0.0],
            user_id="local",
            category="preference",
            source="conv-9",
        )
        md = store.get("id1")["metadata"]
        assert md["user_id"] == "local"
        assert md["category"] == "preference"
        assert md["source"] == "conv-9"

    def test_count(self, store):
        store.add("a", "x", embedding=[1.0, 0.0, 0.0])
        store.add("b", "y", embedding=[0.0, 1.0, 0.0])
        assert store.count() == 2

    def test_update_text_and_metadata(self, store):
        store.add("id1", "old", embedding=[1.0, 0.0, 0.0], category="fact")
        assert store.update("id1", text="new", embedding=[0.0, 1.0, 0.0], category="identity") is True
        rec = store.get("id1")
        assert rec["document"] == "new"
        assert rec["metadata"]["category"] == "identity"

    def test_update_missing_returns_false(self, store):
        assert store.update("ghost", text="x", embedding=[1.0, 0.0, 0.0]) is False

    def test_delete_scoped(self, store):
        store.add("id1", "x", embedding=[1.0, 0.0, 0.0], user_id="local")
        assert store.delete("id1", user_id="someone-else") is False
        assert store.delete("id1", user_id="local") is True
        assert store.get("id1") is None

    def test_delete_missing_returns_false(self, store):
        assert store.delete("ghost") is False

    def test_clear_scoped(self, store):
        store.add("a", "x", embedding=[1.0, 0.0, 0.0], user_id="alice")
        store.add("b", "y", embedding=[0.0, 1.0, 0.0], user_id="alice")
        store.add("c", "z", embedding=[0.0, 0.0, 1.0], user_id="bob")
        removed = store.clear(user_id="alice")
        assert removed == 2
        assert store.count() == 1


# find_similar


class TestFindSimilar:
    def _seed(self, store):
        store.add("near", "Leon likes Kubuntu", embedding=[1.0, 0.0, 0.0], category="preference")
        store.add("alsonear", "Leon prefers Kubuntu", embedding=[0.97, 0.03, 0.0], category="preference")
        store.add("far", "Paris is the capital", embedding=[0.0, 1.0, 0.0], category="fact")

    def test_orders_by_similarity(self, store):
        self._seed(store)
        sims = store.find_similar([1.0, 0.0, 0.0], top_k=3)
        assert [s.id for s in sims] == ["near", "alsonear", "far"]
        assert sims[0].similarity >= sims[1].similarity >= sims[2].similarity

    def test_threshold_filters(self, store):
        self._seed(store)
        near = store.find_similar([1.0, 0.0, 0.0], top_k=5, threshold=0.92)
        assert {s.id for s in near} == {"near", "alsonear"}

    def test_similarity_from_distance(self, store):
        # Identical vector => cosine similarity 1.0 (distance 0.0).
        store.add("exact", "x", embedding=[0.5, 0.5, 0.0])
        sims = store.find_similar([0.5, 0.5, 0.0], top_k=1)
        assert sims[0].id == "exact"
        assert abs(sims[0].similarity - 1.0) < 1e-6

    def test_per_user_isolation_in_query(self, store):
        store.add("a", "alice fact", embedding=[1.0, 0.0, 0.0], user_id="alice")
        store.add("b", "bob fact", embedding=[1.0, 0.0, 0.0], user_id="bob")
        sims = store.find_similar([1.0, 0.0, 0.0], user_id="alice", top_k=5)
        assert {s.id for s in sims} == {"a"}

    def test_empty_collection_returns_empty(self, store):
        assert store.find_similar([1.0, 0.0, 0.0]) == []


# Per-user isolation (counts)


class TestPerUserIsolation:
    def test_count_per_user(self, store):
        store.add("a1", "x", embedding=[1.0, 0.0, 0.0], user_id="alice")
        store.add("a2", "y", embedding=[0.0, 1.0, 0.0], user_id="alice")
        store.add("b1", "z", embedding=[0.0, 0.0, 1.0], user_id="bob")
        assert store.count(user_id="alice") == 2
        assert store.count(user_id="bob") == 1
        assert store.count() == 3


# Spec registration


class TestSpecRegistration:
    def _spec(self) -> str:
        return SPEC.read_text(encoding="utf-8")

    def test_vector_store_registered(self):
        assert "opti_oignon/memory/vector_store.py" in self._spec()

    def test_spec_mentions_oo_memories_and_cosine(self):
        text = self._spec().lower()
        assert "oo_memories" in text
        assert "cosine" in text
