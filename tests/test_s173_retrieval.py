#!/usr/bin/env python3
"""Tests for S173 -- hybrid retrieval, query-type detection, token budget.

Query-type detection is pure and tested directly. Hybrid scoring (vector +
keyword + category) and budgeted formatting run against a real canonical store
(tmp SQLite) seeded through the coordinated store, with an injected cosine
collection and a deterministic embedder. The token estimator is the one from
context_window.py (with an identical fallback), exercised via tiny budgets.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
MEM = OO / "memory"
RET_PATH = MEM / "retrieval.py"
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
_ensure_real("context_window")
canon = _ensure_mem("canonical_store")
vec = _ensure_mem("vector_store")
ded = _ensure_mem("dedup")
ret = _ensure_mem("retrieval")


F_NAME = "The user's name is Leon"
F_PREF = "Leon prefers dark mode interfaces"
F_PROJ = "Leon is building Opti-Oignon a local inference platform"
F_PARIS = "Paris is the capital of France"

Q_NAME = "what is my name"
Q_SEMANTIC = "describe the software i am developing"
Q_KEYWORD = "dark mode interfaces the user enabled"
Q_CATEGORY = "what is my favorite editor"
Q_IRRELEVANT = "tell me about the weather today"

FACTS = {
    F_NAME: ("identity", [1.0, 0.0, 0.0, 0.0]),
    F_PREF: ("preference", [0.0, 1.0, 0.0, 0.0]),
    F_PROJ: ("project", [0.0, 0.0, 1.0, 0.0]),
    F_PARIS: ("fact", [0.0, 0.0, 0.0, 1.0]),
}
QUERY_VECTORS = {
    Q_NAME: [1.0, 0.0, 0.0, 0.0],
    Q_SEMANTIC: [0.0, 0.0, 1.0, 0.0],
    Q_KEYWORD: [0.0, 0.0, 0.0, 0.0],
    Q_CATEGORY: [0.0, 0.0, 0.0, 0.0],
    Q_IRRELEVANT: [0.0, 0.0, 0.0, 0.0],
}
MAPPING = {text: v for text, (_c, v) in FACTS.items()}
MAPPING.update(QUERY_VECTORS)


def _build(tmp_path, *, single_user_mode=True, seed=True, name="mf.db"):
    canon_store = canon.CanonicalMemoryStore(
        tmp_path / name, single_user_mode=single_user_mode
    )
    embedder = FakeEmbedder(mapping=MAPPING, dim=4)
    vstore = vec.MemoryVectorStore(
        collection=FakeChromaCollection(name=vec.COLLECTION_NAME), embedder=embedder
    )
    store = ded.MemoryStore(canon_store, vstore)
    if seed:
        for text, (category, _v) in FACTS.items():
            store.add(text, category)
    retriever = ret.MemoryRetriever(canon_store, vstore)
    return retriever, canon_store, vstore, store


@pytest.fixture
def retriever_bundle(tmp_path):
    return _build(tmp_path)


# Module sentinels


class TestModuleSentinels:
    def test_feature_available(self):
        assert ret.FEATURE_AVAILABLE is True

    def test_checkpoint_sentinel(self):
        assert ret.checkpoint_before_apply is True

    def test_singleton_helpers(self):
        assert hasattr(ret, "get_retriever")
        assert hasattr(ret, "reset_retriever")


# Query-type detection


class TestQueryTypeDetection:
    def test_identity(self):
        assert ret.detect_query_type("what is my name") == "identity"
        assert ret.detect_query_type("who is the user") == "identity"

    def test_preference(self):
        assert ret.detect_query_type("what is my favorite editor") == "preference"
        assert ret.detect_query_type("which theme do I prefer") == "preference"

    def test_contact(self):
        assert ret.detect_query_type("what is my email") == "contact"

    def test_project(self):
        assert ret.detect_query_type("how is my project going") == "project"

    def test_goal(self):
        assert ret.detect_query_type("what are my goals") == "goal"

    def test_general_returns_none(self):
        assert ret.detect_query_type("tell me about the weather today") is None

    def test_empty_returns_none(self):
        assert ret.detect_query_type("") is None


# Hybrid retrieval


class TestHybridRetrieve:
    def test_keyword_path(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        results = retriever.retrieve(Q_KEYWORD, top_n=3)
        assert results
        assert results[0].text == F_PREF
        assert results[0].keyword_score > 0.0

    def test_vector_path(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        results = retriever.retrieve(Q_SEMANTIC, top_n=3)
        assert results[0].text == F_PROJ
        assert results[0].vector_similarity > 0.0
        assert results[0].keyword_score == 0.0

    def test_category_path(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        results = retriever.retrieve(Q_CATEGORY, top_n=3)
        # Only the preference fact matches via the category boost here.
        assert [m.text for m in results] == [F_PREF]
        assert results[0].category_match is True

    def test_combined_signals_rank_top(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        results = retriever.retrieve(Q_NAME, top_n=3)
        top = results[0]
        assert top.text == F_NAME
        assert top.vector_similarity > 0.0
        assert top.keyword_score > 0.0
        assert top.category_match is True

    def test_scores_descending(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        results = retriever.retrieve(Q_KEYWORD, top_n=5)
        scores = [m.score for m in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_limit(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        results = retriever.retrieve(Q_NAME, top_n=1)
        assert len(results) <= 1

    def test_irrelevant_returns_empty(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        assert retriever.retrieve(Q_IRRELEVANT, top_n=5) == []

    def test_mark_used_increments_use_count(self, retriever_bundle):
        retriever, canon_store, _, _ = retriever_bundle
        before = retriever.retrieve(Q_NAME, top_n=1)[0]
        assert canon_store.get(before.id).use_count == 0
        retriever.retrieve(Q_NAME, top_n=1, mark_used=True)
        assert canon_store.get(before.id).use_count == 1

    def test_mark_used_default_is_read_only(self, retriever_bundle):
        retriever, canon_store, _, _ = retriever_bundle
        hit = retriever.retrieve(Q_NAME, top_n=1)[0]
        assert canon_store.get(hit.id).use_count == 0


# Per-user isolation


class TestPerUserIsolation:
    def test_retrieve_scoped_to_user(self, tmp_path):
        retriever, _canon, _vec, store = _build(
            tmp_path, single_user_mode=False, seed=False
        )
        store.add(F_NAME, "identity", user_id="alice", embedding=MAPPING[F_NAME])
        store.add(F_PARIS, "fact", user_id="bob", embedding=MAPPING[F_PARIS])
        alice_hits = retriever.retrieve(Q_NAME, user_id="alice", top_n=5)
        assert all(m.record.user_id == "alice" for m in alice_hits)
        assert any(m.text == F_NAME for m in alice_hits)
        bob_hits = retriever.retrieve(Q_NAME, user_id="bob", top_n=5)
        assert all(m.text != F_NAME for m in bob_hits)


# Token-budget formatting


class TestFormatBudget:
    def test_format_has_header_and_line_shape(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        results = retriever.retrieve(Q_NAME, top_n=3)
        block = retriever.format_for_prompt(results)
        assert block.startswith("Relevant memories:")
        assert "- [identity] " + F_NAME in block

    def test_format_empty_when_no_memories(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        assert retriever.format_for_prompt([]) == ""

    def test_format_respects_tiny_budget(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        results = retriever.retrieve(Q_KEYWORD, top_n=5)
        # A budget that only fits the header cannot fit any line.
        assert retriever.format_for_prompt(results, max_tokens=2) == ""

    def test_fit_to_budget_truncates(self, tmp_path):
        retriever, _canon, _vec, store = _build(
            tmp_path, single_user_mode=False, seed=False
        )
        # Three same-user facts sharing only the query keyword (so they do not
        # merge under dedup) with orthogonal embeddings.
        facts = [
            ("alpha concerns morning routines", [1.0, 0.0, 0.0, 0.0]),
            ("alpha drives evening logging", [0.0, 1.0, 0.0, 0.0]),
            ("alpha shapes weekend planning", [0.0, 0.0, 1.0, 0.0]),
        ]
        for text, embedding in facts:
            store.add(text, "fact", user_id="local", embedding=embedding)
        results = retriever.retrieve(
            "alpha", user_id="local", top_n=5, query_embedding=[0.0, 0.0, 0.0, 0.0]
        )
        assert len(results) == 3
        full = retriever.fit_to_budget(results, max_tokens=1000)
        clipped = retriever.fit_to_budget(results, max_tokens=12)
        assert 0 < len(clipped) < len(full)

    def test_fit_to_budget_returns_subset(self, retriever_bundle):
        retriever, _, _, _ = retriever_bundle
        results = retriever.retrieve(Q_NAME, top_n=3)
        fitted = retriever.fit_to_budget(results, max_tokens=1000)
        assert fitted == results[: len(fitted)]


# Reuse and registration


class TestContextWindowReuse:
    def test_references_context_window(self):
        text = RET_PATH.read_text(encoding="utf-8")
        assert "context_window" in text


class TestSpecRegistration:
    def _spec(self) -> str:
        return SPEC.read_text(encoding="utf-8")

    def test_retrieval_registered(self):
        assert "opti_oignon/memory/retrieval.py" in self._spec()

    def test_spec_mentions_query_type_and_budget(self):
        text = self._spec().lower()
        assert "query-type" in text
        assert "context_window" in text
