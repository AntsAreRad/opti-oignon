#!/usr/bin/env python3
"""Dedup contracts: two-stage duplicate checks, two layers kept in step.

Every fact enters through the coordinated store, where a cheap text
similarity stage runs before the vector stage and a duplicate reinforces the
existing fact instead of multiplying it. Deletions and restores mirror across
the canonical row and the vector entry so nothing lingers in one layer after
leaving the other. This suite pins that coordination:

  * DD1 -- a text-stage duplicate merges: the existing fact is reinforced,
    nothing is inserted in either layer, and the existing record is returned;
  * DD2 -- the text stage runs first and alone when it trips: the vector
    layer is never consulted, and a supplied embedding is taken as-is;
  * DD3 -- past the text stage, a vector near-duplicate merges at the cosine
    bar; below both bars the decision is an insert;
  * DD4 -- a soft delete mirrors into the vector layer only when the
    canonical row actually flipped, and a restore re-seeds the vector entry;
  * DD5 -- a hard delete always clears the vector entry, whatever the
    canonical outcome, and reports the canonical outcome;
  * DD6 -- without an embedding the insert lands canonical-only (degraded,
    never raising); with one, the vector entry mirrors the new record.

Loads the dedup module in isolation over recorder stores. Local-only. Runs
under pytest or the __main__ runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_MEMORY = _REPO / "opti_oignon" / "memory"


def _load():
    """Load the dedup module under a stand-in package.

    Every ``opti_oignon.*`` entry is snapshotted and evicted first so a
    previously imported real module cannot leak into the isolation window,
    then restored afterwards.
    """
    saved = {
        k: sys.modules[k]
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    }
    for k in saved:
        del sys.modules[k]

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    memory = types.ModuleType("opti_oignon.memory")
    memory.__path__ = []
    sys.modules["opti_oignon"] = root
    sys.modules["opti_oignon.memory"] = memory

    full = "opti_oignon.memory.dedup"
    spec = importlib.util.spec_from_file_location(full, _MEMORY / "dedup.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    memory.dedup = mod
    spec.loader.exec_module(mod)

    def restore():
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        for k, v in saved.items():
            sys.modules[k] = v

    return mod, restore


class _Rec:
    def __init__(self, rid, text, category="fact", source="", user_id="local"):
        self.id = rid
        self.text = text
        self.category = category
        self.source = source
        self.user_id = user_id
        self.use_count = 0
        self.created_at = "2026-01-01T00:00:00"
        self.updated_at = self.created_at


class _Canonical:
    """Recorder canonical store."""

    def __init__(self, records=(), soft_result=True, hard_result=True):
        self.records = {r.id: r for r in records}
        self.soft_result = soft_result
        self.hard_result = hard_result
        self.touched = []
        self.added = []
        self.soft_calls = []
        self.hard_calls = []
        self.restored = []

    def resolve_user(self, user_id=None):
        return "local"

    def list(self, *, active_only=True, user_id=None, **_kw):
        return list(self.records.values())

    def get(self, fact_id, *, user_id=None):
        return self.records.get(fact_id)

    def add(self, text, category="fact", *, source="", user_id=None):
        rec = _Rec(f"new{len(self.added)}", text, category, source)
        self.records[rec.id] = rec
        self.added.append(rec.id)
        return rec

    def touch(self, fact_id, *, user_id=None):
        self.touched.append(fact_id)
        return True

    def soft_delete(self, fact_id, *, user_id=None):
        self.soft_calls.append(fact_id)
        return self.soft_result

    def restore(self, fact_id, *, user_id=None):
        self.restored.append(fact_id)
        return True

    def hard_delete(self, fact_id, *, user_id=None):
        self.hard_calls.append(fact_id)
        return self.hard_result

    def update(self, fact_id, *, user_id=None, **fields):
        rec = self.records.get(fact_id)
        if rec is None:
            return None
        for key, val in fields.items():
            setattr(rec, key, val)
        return rec


class _Neighbour:
    def __init__(self, nid, similarity):
        self.id = nid
        self.similarity = similarity


class _Vector:
    """Recorder vector layer with configurable answers."""

    def __init__(self, embed_result=None, neighbours=()):
        self.embed_result = embed_result
        self.neighbours = list(neighbours)
        self.embed_calls = []
        self.find_calls = []
        self.add_calls = []
        self.delete_calls = []
        self.update_calls = []

    def embed(self, text):
        self.embed_calls.append(text)
        return self.embed_result

    def find_similar(self, embedding, *, user_id=None, top_k=5, threshold=None):
        self.find_calls.append((list(embedding), threshold))
        return list(self.neighbours)

    def add(self, fact_id, text, *, embedding=None, **kw):
        self.add_calls.append((fact_id, text, embedding, kw))
        return fact_id

    def delete(self, fact_id, *, user_id=None):
        self.delete_calls.append((fact_id, user_id))
        return True

    def update(self, fact_id, **kw):
        self.update_calls.append((fact_id, kw))
        return True


_TEXT_A = "the user brews green tea every single morning"
_TEXT_B = "the user brews green tea every morning"
_DISTINCT = "xray yankee zulu November papa quebec"


def test_dd1_a_text_stage_duplicate_reinforces_instead_of_inserting():
    mod, restore = _load()
    try:
        existing = _Rec("keep", _TEXT_A)
        canonical = _Canonical([existing])
        vector = _Vector(embed_result=None)
        store = mod.MemoryStore(canonical, vector)
        record, decision = store.add(_TEXT_B, "fact")
        assert decision.action == "merge"
        assert decision.reason == "jaccard"
        assert decision.target_id == "keep"
        assert canonical.touched == ["keep"]
        assert canonical.added == []
        assert vector.add_calls == []
        assert record.id == "keep"
    finally:
        restore()


def test_dd2_the_text_stage_runs_first_and_skips_the_vector_layer():
    mod, restore = _load()
    try:
        canonical = _Canonical([_Rec("keep", _TEXT_A)])
        vector = _Vector(embed_result=[9.0, 9.0])
        store = mod.MemoryStore(canonical, vector)
        _record, decision = store.add(_TEXT_B, "fact", embedding=[0.1, 0.2])
        assert decision.reason == "jaccard"
        assert vector.find_calls == []
        assert vector.embed_calls == [], "a supplied embedding is taken as-is"
    finally:
        restore()


def test_dd3_the_cosine_bar_merges_and_a_double_miss_inserts():
    mod, restore = _load()
    try:
        canonical = _Canonical([_Rec("far", "alpha bravo charlie delta")])
        vector = _Vector(neighbours=[_Neighbour("far", 0.95)])
        store = mod.MemoryStore(canonical, vector)
        record, decision = store.add(_DISTINCT, "fact", embedding=[0.5, 0.5])
        assert decision.action == "merge"
        assert decision.reason == "cosine"
        assert decision.target_id == "far"
        assert record.id == "far"

        canonical2 = _Canonical([_Rec("far", "alpha bravo charlie delta")])
        vector2 = _Vector(neighbours=[])
        store2 = mod.MemoryStore(canonical2, vector2)
        _record2, decision2 = store2.add(_DISTINCT, "fact", embedding=[0.5, 0.5])
        assert decision2.action == "insert"
        assert canonical2.added, "a double miss must insert"
    finally:
        restore()


def test_dd4_soft_delete_mirrors_only_on_success_and_restore_reseeds():
    mod, restore = _load()
    try:
        rec = _Rec("keep", _TEXT_A)
        canonical = _Canonical([rec])
        vector = _Vector(embed_result=[0.3, 0.4])
        store = mod.MemoryStore(canonical, vector)
        assert store.soft_delete("keep") is True
        assert vector.delete_calls == [("keep", "local")]

        canonical_miss = _Canonical([], soft_result=False)
        vector_miss = _Vector()
        store_miss = mod.MemoryStore(canonical_miss, vector_miss)
        assert store_miss.soft_delete("ghost") is False
        assert vector_miss.delete_calls == []

        assert store.restore("keep") is True
        assert vector.embed_calls == [_TEXT_A]
        assert [c[0] for c in vector.add_calls] == ["keep"]
    finally:
        restore()


def test_dd5_hard_delete_always_clears_the_vector_entry():
    mod, restore = _load()
    try:
        canonical = _Canonical([_Rec("keep", _TEXT_A)])
        vector = _Vector()
        store = mod.MemoryStore(canonical, vector)
        assert store.hard_delete("keep") is True
        assert vector.delete_calls == [("keep", "local")]

        canonical_miss = _Canonical([], hard_result=False)
        vector_miss = _Vector()
        store_miss = mod.MemoryStore(canonical_miss, vector_miss)
        assert store_miss.hard_delete("ghost") is False
        assert vector_miss.delete_calls == [("ghost", "local")]
    finally:
        restore()


def test_dd6_inserts_degrade_without_an_embedding_and_mirror_with_one():
    mod, restore = _load()
    try:
        canonical = _Canonical()
        vector = _Vector(embed_result=None)
        store = mod.MemoryStore(canonical, vector)
        record, decision = store.add(_DISTINCT, "fact")
        assert decision.action == "insert"
        assert canonical.added == [record.id]
        assert vector.add_calls == [], "no embedding means no vector write"

        canonical2 = _Canonical()
        vector2 = _Vector()
        store2 = mod.MemoryStore(canonical2, vector2)
        record2, _decision2 = store2.add(_DISTINCT, "goal", embedding=[0.7, 0.7])
        assert [c[0] for c in vector2.add_calls] == [record2.id]
        assert vector2.add_calls[0][2] == [0.7, 0.7]
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
