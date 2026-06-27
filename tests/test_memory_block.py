#!/usr/bin/env python3
"""Tests for the unified memory block composer (M1).

``retrieval.build_memory_block`` is the single working-memory block injected into
the prompt. It fixes the gap where the query-only ``working_block`` drops every
fact scoring 0 (so durable facts vanish on an unrelated turn), and unifies the
two memory stores during migration. This suite loads ``retrieval.py`` in
isolation (a fake canonical store + fake vector layer injected) and proves:

  * a SALIENCE FLOOR: the top facts by use_count x recency are always present,
    even when the query is unrelated -- and the contrast is asserted directly:
    the old ``working_block(query)`` returns "" on that same query;
  * query-relevant facts are added on top;
  * a LEGACY BRIDGE: facts passed in (the legacy memories.db) are merged and
    deduplicated against the canonical facts by text;
  * the token budget is enforced, salient facts surviving the tail truncation;
  * graceful degradation: a missing embedder removes only the vector
    contribution, the canonical facts still inject;
  * ``mark_used`` touches exactly the injected facts (the reinforcement loop);
  * nothing to show returns an empty string.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load():
    keys = ("opti_oignon", "opti_oignon.memory", "opti_oignon.memory.retrieval")
    saved = {k: sys.modules.get(k) for k in keys}
    for n in ("opti_oignon", "opti_oignon.memory"):
        pkg = types.ModuleType(n)
        pkg.__path__ = []
        sys.modules[n] = pkg
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.memory.retrieval", _OO / "memory" / "retrieval.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.memory.retrieval"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


# --- fakes mirroring the stores' interfaces the retriever calls ---

@dataclass
class FRec:
    id: str
    text: str
    category: str = "fact"
    use_count: int = 0
    updated_at: str = "t0"


@dataclass
class FNeighbour:
    id: str
    similarity: float


class FakeCanonical:
    def __init__(self, records):
        self._records = list(records)
        self.touched = []

    def resolve_user(self, user_id):
        return user_id or "local"

    def list(self, *, active_only=True, user_id=None):
        return list(self._records)

    def touch(self, fact_id, *, user_id=None):
        self.touched.append(fact_id)


class FakeVector:
    def __init__(self, *, embedding=None, sims=None):
        self._embedding = embedding
        self._sims = sims or {}

    def embed(self, text):
        return self._embedding  # None simulates a missing embedder

    def find_similar(self, embedding, *, user_id=None, top_k=10):
        return [FNeighbour(i, s) for i, s in self._sims.items()]


@dataclass
class FLegacy:
    text: str
    category: str = "fact"


def _retriever(mod, records, *, embedding=None, sims=None):
    return mod.MemoryRetriever(
        FakeCanonical(records), FakeVector(embedding=embedding, sims=sims)
    )


def test_salience_floor_present_on_unrelated_query():
    mod, restore = _load()
    try:
        recs = [
            FRec("n", "The user's name is Leon", "identity", use_count=10, updated_at="t3"),
            FRec("p", "The user prefers concise answers", "pref", use_count=5, updated_at="t2"),
        ]
        r = _retriever(mod, recs, embedding=None)  # embedder down -> no vector help
        q = "how do I bake sourdough bread"   # zero token overlap with the facts

        block = mod.build_memory_block(q, retriever=r, max_tokens=512)
        assert "Leon" in block            # durable identity fact still injected
        assert "concise" in block
        # the contrast: the old query-only path drops both (score 0) -> empty
        assert r.working_block(q) == ""
    finally:
        restore()


def test_query_relevant_added_on_top():
    mod, restore = _load()
    try:
        recs = [
            FRec("n", "The user's name is Leon", "identity", use_count=10),
            FRec("w", "The user works at Acme Corp", "work", use_count=1),
        ]
        r = _retriever(mod, recs, embedding=None)
        block = mod.build_memory_block("what about Acme Corp", retriever=r, max_tokens=512)
        assert "Acme" in block            # query-relevant fact surfaced
        assert "Leon" in block            # salient baseline still there
    finally:
        restore()


def test_legacy_bridge_merged_and_deduped():
    mod, restore = _load()
    try:
        recs = [FRec("n", "The user's name is Leon", "identity", use_count=10)]
        r = _retriever(mod, recs, embedding=None)
        legacy = [
            FLegacy("The user likes tea", "pref"),
            FLegacy("The user's name is Leon", "identity"),  # duplicate of canonical
        ]
        block = mod.build_memory_block(None, retriever=r, legacy_facts=legacy, max_tokens=512)
        assert "likes tea" in block       # legacy fact bridged in
        assert block.count("Leon") == 1   # duplicate not added twice
    finally:
        restore()


def test_budget_truncates_keeping_salient():
    mod, restore = _load()
    try:
        recs = [FRec(f"f{i}", f"fact number {i} body text", "fact",
                     use_count=100 - i, updated_at=f"t{100 - i}") for i in range(6)]
        r = _retriever(mod, recs, embedding=None)
        block = mod.build_memory_block(None, retriever=r, max_tokens=12)
        assert "fact number 0" in block          # most-salient survives
        assert "fact number 5" not in block       # least-salient truncated
        assert mod._estimate_tokens(block) <= 12  # budget respected
    finally:
        restore()


def test_embedder_down_graceful():
    mod, restore = _load()
    try:
        recs = [FRec("n", "The user's name is Leon", "identity", use_count=10)]
        r = _retriever(mod, recs, embedding=None)  # embed() returns None
        block = mod.build_memory_block("any question at all", retriever=r, max_tokens=512)
        assert "Leon" in block            # canonical facts still inject, no crash
    finally:
        restore()


def test_mark_used_touches_only_injected():
    mod, restore = _load()
    try:
        # 6 facts; the salience floor (top 5 by use_count) excludes f5
        recs = [FRec(f"f{i}", f"durable fact {i}", "fact", use_count=10 - i) for i in range(6)]
        fc = FakeCanonical(recs)
        r = mod.MemoryRetriever(fc, FakeVector(embedding=None))
        mod.build_memory_block("unrelated", retriever=r, max_tokens=512, mark_used=True)
        assert "f0" in fc.touched          # an injected salient fact is touched
        assert "f5" in [rr.id for rr in recs]
        assert "f5" not in fc.touched      # the non-injected fact is not touched
    finally:
        restore()


def test_empty_returns_empty():
    mod, restore = _load()
    try:
        r = _retriever(mod, [], embedding=None)
        assert mod.build_memory_block("x", retriever=r, legacy_facts=None, max_tokens=512) == ""
    finally:
        restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
