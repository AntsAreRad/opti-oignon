#!/usr/bin/env python3
"""Tests for S173 -- double deduplication and coordinated CRUD.

The Jaccard helper is pure and tested directly. The two-stage dedup (Jaccard 0.6
then cosine 0.92) and the cross-layer CRUD consistency run against a real
canonical store (tmp SQLite), the vector layer with an injected cosine
collection, and a deterministic embedder, all loaded in isolation.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
MEM = OO / "memory"
SPEC = ROOT / "ODYSSEUS_SPEC.md"

sys.path.insert(0, str(ROOT / "tests"))
from _memory_fakes import FakeChromaCollection, FakeEmbedder  # noqa: E402


def _ensure_pkg():
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.memory" not in sys.modules:
        mpkg = types.ModuleType("opti_oignon.memory")
        mpkg.__path__ = [str(MEM)]
        sys.modules["opti_oignon.memory"] = mpkg


def _ensure_real(name: str):
    full = f"opti_oignon.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(OO / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_mem(name: str):
    full = f"opti_oignon.memory.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(MEM / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg()
_ensure_real("db_encryption")
_ensure_real("user_isolation")
canon = _ensure_mem("canonical_store")
vec = _ensure_mem("vector_store")
ded = _ensure_mem("dedup")


# Texts and the vectors the embedder maps them to (4-dim, consistent).
BIO = "The user works in bioinformatics and ecology"
GENOMICS = "Genomics analysis fills most of the workday"
PARIS = "Paris is the capital of France"
KUB_LINUX = "Leon likes Kubuntu Linux"
KUB = "Leon likes Kubuntu"
DARK = "User prefers dark mode"
HIKE = "User enjoys hiking on weekends"

MAPPING = {
    BIO: [1.0, 0.0, 0.0, 0.0],
    GENOMICS: [0.99, 0.01, 0.0, 0.0],   # near-identical to BIO, disjoint tokens
    PARIS: [0.0, 1.0, 0.0, 0.0],        # orthogonal to BIO
    KUB_LINUX: [0.0, 0.0, 1.0, 0.0],
    KUB: [0.0, 0.0, 0.97, 0.03],        # high jaccard and high cosine to KUB_LINUX
    DARK: [0.5, 0.5, 0.0, 0.0],
    HIKE: [0.0, 0.0, 0.0, 1.0],
}


def _build(tmp_path, *, single_user_mode=True, name="mf.db"):
    canon_store = canon.CanonicalMemoryStore(
        tmp_path / name, single_user_mode=single_user_mode
    )
    embedder = FakeEmbedder(mapping=MAPPING, dim=4)
    vstore = vec.MemoryVectorStore(
        collection=FakeChromaCollection(name=vec.COLLECTION_NAME), embedder=embedder
    )
    store = ded.MemoryStore(canon_store, vstore)
    return store, canon_store, vstore


@pytest.fixture
def store_bundle(tmp_path):
    return _build(tmp_path)


# Module sentinels


class TestModuleSentinels:
    def test_feature_available(self):
        assert ded.FEATURE_AVAILABLE is True

    def test_checkpoint_sentinel(self):
        assert ded.checkpoint_before_apply is True

    def test_singleton_helpers(self):
        assert hasattr(ded, "get_memory_store")
        assert hasattr(ded, "reset_memory_store")


# Jaccard helper


class TestJaccard:
    def test_identical(self):
        assert ded.jaccard_similarity("alpha beta", "alpha beta") == 1.0

    def test_disjoint(self):
        assert ded.jaccard_similarity("alpha beta", "gamma delta") == 0.0

    def test_partial(self):
        assert abs(ded.jaccard_similarity("a b c", "a b") - (2 / 3)) < 1e-9

    def test_both_empty(self):
        assert ded.jaccard_similarity("", "") == 1.0

    def test_one_empty(self):
        assert ded.jaccard_similarity("alpha", "") == 0.0

    def test_case_and_punctuation_insensitive(self):
        assert ded.jaccard_similarity("Leon, Kubuntu!", "leon kubuntu") == 1.0

    def test_thresholds(self):
        assert ded.JACCARD_THRESHOLD == 0.6
        assert ded.COSINE_THRESHOLD == 0.92


# Dedup stages


class TestDedupStages:
    def test_insert_when_no_match(self, store_bundle):
        store, _, _ = store_bundle
        store.add(BIO, "fact")
        _, decision = store.add(PARIS, "fact")
        assert decision.action == "insert"

    def test_jaccard_merge(self, store_bundle):
        store, _, _ = store_bundle
        store.add(KUB_LINUX, "preference")
        _, decision = store.add(KUB, "preference")
        assert decision.action == "merge"
        assert decision.reason == "jaccard"
        assert decision.score >= 0.6

    def test_cosine_merge(self, store_bundle):
        store, _, _ = store_bundle
        store.add(BIO, "fact")
        _, decision = store.add(GENOMICS, "fact")
        assert decision.action == "merge"
        assert decision.reason == "cosine"
        assert decision.score >= 0.92

    def test_jaccard_takes_priority_over_cosine(self, store_bundle):
        # KUB trips both stages against KUB_LINUX; the text stage runs first.
        store, _, _ = store_bundle
        store.add(KUB_LINUX, "preference")
        _, decision = store.add(KUB, "preference")
        assert decision.reason == "jaccard"

    def test_exact_duplicate_merges(self, store_bundle):
        store, canon_store, _ = store_bundle
        store.add(PARIS, "fact")
        _, decision = store.add(PARIS, "fact")
        assert decision.action == "merge"
        assert canon_store.count() == 1


# Coordinated CRUD across both layers


class TestCoordinatedCrud:
    def test_add_writes_both_layers(self, store_bundle):
        store, canon_store, vstore = store_bundle
        record, _ = store.add(BIO, "fact")
        assert canon_store.count() == 1
        assert vstore.count() == 1
        assert vstore.get(record.id) is not None

    def test_merge_does_not_duplicate(self, store_bundle):
        store, canon_store, vstore = store_bundle
        store.add(BIO, "fact")
        store.add(GENOMICS, "fact")  # cosine merge
        assert canon_store.count() == 1
        assert vstore.count() == 1

    def test_merge_bumps_use_count(self, store_bundle):
        store, canon_store, _ = store_bundle
        record, _ = store.add(KUB_LINUX, "preference")
        assert canon_store.get(record.id).use_count == 0
        store.add(KUB, "preference")  # jaccard merge into record
        assert canon_store.get(record.id).use_count == 1

    def test_update_propagates_text_to_vector(self, store_bundle):
        store, _, vstore = store_bundle
        record, _ = store.add(BIO, "fact")
        store.update(record.id, text=PARIS, embedding=MAPPING[PARIS])
        rec = vstore.get(record.id)
        assert rec["document"] == PARIS

    def test_update_missing_returns_none(self, store_bundle):
        store, _, _ = store_bundle
        assert store.update("ghost", text="x") is None

    def test_soft_delete_removes_vector_keeps_canonical(self, store_bundle):
        store, canon_store, vstore = store_bundle
        record, _ = store.add(PARIS, "fact")
        assert store.soft_delete(record.id) is True
        assert canon_store.get(record.id).active is False
        assert canon_store.count(active_only=False) == 1
        assert vstore.get(record.id) is None

    def test_soft_delete_then_restore_readds_vector(self, store_bundle):
        store, _, vstore = store_bundle
        record, _ = store.add(PARIS, "fact")
        store.soft_delete(record.id)
        assert store.restore(record.id) is True
        assert vstore.get(record.id) is not None

    def test_hard_delete_removes_both(self, store_bundle):
        store, canon_store, vstore = store_bundle
        record, _ = store.add(PARIS, "fact")
        assert store.hard_delete(record.id) is True
        assert canon_store.get(record.id) is None
        assert vstore.get(record.id) is None

    def test_list_and_count_via_store(self, store_bundle):
        store, _, _ = store_bundle
        store.add(BIO, "fact")
        store.add(PARIS, "fact")
        store.add(HIKE, "fact")
        assert store.count() == 3
        assert len(store.list()) == 3


# Per-user isolation through the coordinated store


class TestPerUserIsolation:
    def test_no_cross_user_merge(self, tmp_path):
        store, canon_store, vstore = _build(tmp_path, single_user_mode=False)
        store.add(DARK, "preference", user_id="alice")
        # Same text for bob must insert, not merge with alice's fact.
        _, decision = store.add(DARK, "preference", user_id="bob")
        assert decision.action == "insert"
        assert canon_store.count(user_id="alice") == 1
        assert canon_store.count(user_id="bob") == 1
        assert vstore.count() == 2

    def test_delete_scoped_through_store(self, tmp_path):
        store, canon_store, _ = _build(tmp_path, single_user_mode=False)
        b, _ = store.add(DARK, "preference", user_id="bob")
        assert store.hard_delete(b.id, user_id="alice") is False
        assert store.get(b.id, user_id="bob") is not None


# Spec registration


class TestSpecRegistration:
    def _spec(self) -> str:
        return SPEC.read_text(encoding="utf-8")

    def test_dedup_registered(self):
        assert "opti_oignon/memory/dedup.py" in self._spec()

    def test_spec_mentions_thresholds(self):
        text = self._spec()
        assert "0.6" in text
        assert "0.92" in text
