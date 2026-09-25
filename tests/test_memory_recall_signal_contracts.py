#!/usr/bin/env python3
"""Contracts for what the default memory recall says about its own signals.

With the onion off, the working memory block comes from ``memory/retrieval.py``:
facts ranked by vector similarity, keyword coverage and a category cue. When
the vector layer cannot embed the query, the ranking falls back to keywords
and category -- rightly -- but every fact carried a vector similarity of 0.0,
exactly what a fact the layer measured and found unrelated carries. A
similarity nobody measured is now ``None``, and the retriever says once that
it recalled without its vector signal. Keywords are whole words, accents
included, so a French query no longer meets a fact through the fragments of
other words.

  * VS1 -- a similarity nobody measured is None -- the query could not be
    embedded, or no query was asked -- and 0.0 only when the vector layer
    answered and the fact was not among its neighbours; the retriever warns
    once when it recalls without its vector signal.
  * VS2 -- a French query matches a French fact on whole accented words, and
    no longer meets an unrelated fact through the fragments of its words.

Local-only (the public distribution ships no tests). The retriever is loaded
through the shared isolation window; the canonical store and the vector layer
are stand-ins with the interfaces the retriever calls.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_RETRIEVAL = "opti_oignon.memory.retrieval"


@dataclass
class _Record:
    id: str
    text: str
    category: str = "fact"
    use_count: int = 0
    updated_at: str = "t0"


@dataclass
class _Neighbour:
    id: str
    similarity: float


class _Canonical:
    def __init__(self, records):
        self._records = list(records)

    def resolve_user(self, user_id):
        return user_id or "local"

    def list(self, *, active_only=True, user_id=None):
        return list(self._records)

    def touch(self, fact_id, *, user_id=None):
        pass


class _Vector:
    """A vector layer that embeds, or cannot, and answers fixed neighbours."""

    def __init__(self, *, embedding=None, neighbours=None):
        self._embedding = embedding
        self._neighbours = neighbours or {}

    def embed(self, text):
        return self._embedding

    def find_similar(self, embedding, *, user_id=None, top_k=10):
        return [_Neighbour(i, s) for i, s in self._neighbours.items()]


def _open():
    loaded, restore = isolate(
        targets={_RETRIEVAL: source("memory", "retrieval.py")},
        packages=("opti_oignon.memory",),
    )
    return loaded[_RETRIEVAL], restore


_FACTS = [
    _Record("tea", "L\u00e9on pr\u00e9f\u00e8re le th\u00e9 vert.", use_count=3),
    _Record("loan", "Le pr\u00eat sera rembours\u00e9.", use_count=2),
    _Record("demo", "L\u00e9on pr\u00e9sente la d\u00e9mo lundi.", use_count=1),
]


# ---------------------------------------------------------------------------
# VS1 -- a similarity nobody measured is unknown, and said once
# ---------------------------------------------------------------------------
def test_vs1_a_similarity_nobody_measured_is_none_and_the_retriever_says_so_once(caplog):
    mod, restore = _open()
    try:
        blind = mod.MemoryRetriever(_Canonical(_FACTS), _Vector(embedding=None))
        with caplog.at_level(logging.WARNING, logger=_RETRIEVAL):
            first = blind.retrieve("Que pr\u00e9f\u00e8re L\u00e9on ?")
            second = blind.retrieve("L\u00e9on pr\u00e9sente quoi ?")
        assert first and second, "keywords still rank the facts without the vector signal"
        unmeasured = [(m.id, m.vector_similarity) for m in first + second]
        assert all(similarity is None for _id, similarity in unmeasured), unmeasured
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "vector" in r.getMessage()]
        assert len(warned) == 1, [r.getMessage() for r in caplog.records]
        recent = blind.recent_memories()
        assert recent and all(m.vector_similarity is None for m in recent), "no query was asked: nothing measured"
        seeing = mod.MemoryRetriever(_Canonical(_FACTS), _Vector(embedding=[1.0, 0.0], neighbours={"tea": 0.9}))
        found = {m.id: m.vector_similarity for m in seeing.retrieve("L\u00e9on pr\u00e9sente la d\u00e9mo")}
        assert found.get("tea") == 0.9, found
        assert found.get("demo") == 0.0, "measured and not a neighbour: 0.0, not unknown"
    finally:
        restore()


# ---------------------------------------------------------------------------
# VS2 -- whole French words, not fragments
# ---------------------------------------------------------------------------
def test_vs2_a_french_query_meets_a_french_fact_on_whole_words_not_on_fragments():
    mod, restore = _open()
    try:
        blind = mod.MemoryRetriever(_Canonical(_FACTS), _Vector(embedding=None))
        found = {m.id: m.keyword_score for m in blind.retrieve("Que pr\u00e9f\u00e8re L\u00e9on ?")}
        assert "loan" not in found, f"no word in common with the loan, only fragments of other words: {found}"
        assert set(found) == {"tea", "demo"}, found
        assert found["tea"] > found["demo"] > 0.0, found
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
