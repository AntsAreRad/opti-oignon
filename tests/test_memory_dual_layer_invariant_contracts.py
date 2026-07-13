#!/usr/bin/env python3
"""Dual-layer contracts: the working block compresses, the archive keeps all.

Personal memory rides two layers: a budgeted working block injected into the
prompt (the compressed layer) and the full active archive that stays
searchable regardless of what the budget dropped. Compression must never cost
access to detail, and surfacing facts must never mutate or shrink the
archive. This suite pins that invariant and the relative-decay mechanics
around it:

  * DL1 -- the dual-layer view reports the full active-archive size, always
    at least the budgeted selection (the selection is a subset, never the
    universe);
  * DL2 -- a fact dropped by the prompt budget is still found by the
    recovery search over the full archive (the invariant itself);
  * DL3 -- budget fitting is a strict prefix cut that counts the header:
    order is preserved and the header consumes budget;
  * DL4 -- surfacing without reinforcement is a pure read: no touch, no
    delete, no write of any kind reaches the store;
  * DL5 -- reinforcement touches exactly the facts returned, never the
    whole scored set;
  * DL6 -- decay is relative and non-destructive: an unused fact leaves the
    working set while remaining active in the archive, and retrieval never
    deletes;
  * DL7 -- an exhausted budget yields an empty block and an empty selection,
    never an exception.

Loads the retrieval module in isolation over recorder stores. Local-only.
Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_MEMORY = _REPO / "opti_oignon" / "memory"

_MODULES = ("retrieval",)


def _load():
    """Load the retrieval module under a stand-in package.

    Every ``opti_oignon.*`` entry is snapshotted and evicted first so a
    previously imported real module cannot leak into the isolation window
    through a lazy import, then restored afterwards.
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

    loaded = {}
    for m in _MODULES:
        full = f"opti_oignon.memory.{m}"
        spec = importlib.util.spec_from_file_location(full, _MEMORY / f"{m}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        setattr(memory, m, mod)
        spec.loader.exec_module(mod)
        loaded[m] = mod

    def restore():
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        for k, v in saved.items():
            sys.modules[k] = v

    return loaded["retrieval"], restore


class _Rec:
    """A minimal canonical record for the retriever."""

    def __init__(self, rid, text, category="fact", use_count=0, updated_at=""):
        self.id = rid
        self.text = text
        self.category = category
        self.use_count = use_count
        self.created_at = updated_at
        self.updated_at = updated_at


class _Canonical:
    """Recorder canonical store: reads served from memory, writes recorded."""

    def __init__(self, records):
        self.records = list(records)
        self.touched = []
        self.deleted = []

    def resolve_user(self, user_id=None):
        return "local"

    def list(self, *, active_only=True, user_id=None, **_kw):
        return list(self.records)

    def count(self, *, active_only=True, user_id=None):
        return len(self.records)

    def touch(self, fact_id, *, user_id=None):
        self.touched.append(fact_id)
        return True

    def soft_delete(self, fact_id, *, user_id=None):
        self.deleted.append(fact_id)
        return True

    def hard_delete(self, fact_id, *, user_id=None):
        self.deleted.append(fact_id)
        return True


class _Vector:
    """Vector layer stub: no embedder, no neighbours (keyword-only recall)."""

    def embed(self, text):
        return None

    def find_similar(self, embedding, *, user_id=None, top_k=5, threshold=None):
        return []


def _facts(n, use_count=0):
    return [
        _Rec(
            f"f{i}",
            f"durable note number {i} about the violet garden project",
            use_count=use_count,
            updated_at=f"2026-01-{i + 1:02d}T00:00:00",
        )
        for i in range(n)
    ]


def _budget_for(mod, retr, memories, keep, header="Relevant memories:"):
    """A token budget that fits the header plus exactly ``keep`` lines."""
    total = mod._estimate_tokens(header)
    for m in memories[:keep]:
        total += mod._estimate_tokens(retr._format_line(m))
    return total


def test_dl1_total_active_reports_the_full_archive_over_the_selection():
    mod, restore = _load()
    try:
        store = _Canonical(_facts(10))
        retr = mod.MemoryRetriever(store, _Vector())
        recent = retr.recent_memories(top_n=10)
        budget = _budget_for(mod, retr, recent, keep=2)
        view = retr.assemble_dual_layer(None, top_n=10, max_tokens=budget)
        assert view.total_active == 10
        assert len(view.selected_ids) < view.total_active
        assert view.total_active >= len(view.selected_ids)
    finally:
        restore()


def test_dl2_a_budget_dropped_fact_stays_recoverable_from_the_archive():
    mod, restore = _load()
    try:
        store = _Canonical(_facts(10))
        retr = mod.MemoryRetriever(store, _Vector())
        recent = retr.recent_memories(top_n=10)
        budget = _budget_for(mod, retr, recent, keep=2)
        view = retr.assemble_dual_layer(None, top_n=10, max_tokens=budget)
        selected = set(view.selected_ids)
        dropped = [r for r in store.records if r.id not in selected]
        assert dropped, "the budget must have dropped at least one fact"
        target = dropped[0]
        found = retr.recover(target.text)
        assert any(m.id == target.id for m in found)
    finally:
        restore()


def test_dl3_budget_fitting_is_a_header_counting_prefix_cut():
    mod, restore = _load()
    try:
        store = _Canonical([])
        retr = mod.MemoryRetriever(store, _Vector())
        header = "Working memory from earlier conversations with this user:"
        short = mod.ScoredMemory("s1", "tea", "fact", 1.0, 0.0, 0.0, False)
        tiny = mod.ScoredMemory("s2", "jam", "fact", 0.9, 0.0, 0.0, False)
        rest = mod.ScoredMemory("s3", "one more line here", "fact", 0.8, 0.0, 0.0, False)
        memories = [short, tiny, rest]
        # The header is longer than any line, so an implementation that
        # forgets to count it would admit a second line under this budget.
        assert mod._estimate_tokens(header) > mod._estimate_tokens(
            retr._format_line(tiny)
        )
        budget = mod._estimate_tokens(header) + mod._estimate_tokens(
            retr._format_line(short)
        )
        fitted = retr.fit_to_budget(memories, max_tokens=budget, header=header)
        assert fitted == [short]
        # Generous budget: the cut is a prefix in the original order.
        all_in = retr.fit_to_budget(memories, max_tokens=10_000, header=header)
        assert [m.id for m in all_in] == ["s1", "s2", "s3"]
        partial = retr.fit_to_budget(
            memories,
            max_tokens=budget + mod._estimate_tokens(retr._format_line(tiny)),
            header=header,
        )
        assert [m.id for m in partial] == ["s1", "s2"]
    finally:
        restore()


def test_dl4_surfacing_without_reinforcement_writes_nothing():
    mod, restore = _load()
    try:
        store = _Canonical(_facts(6))
        retr = mod.MemoryRetriever(store, _Vector())
        block = retr.working_block("violet garden", mark_used=False)
        assert block
        retr.recover("violet garden")
        assert store.touched == []
        assert store.deleted == []
    finally:
        restore()


def test_dl5_reinforcement_touches_exactly_the_returned_facts():
    mod, restore = _load()
    try:
        store = _Canonical(_facts(6))
        retr = mod.MemoryRetriever(store, _Vector())
        got = retr.retrieve("violet garden", top_n=2, mark_used=True)
        assert len(got) == 2
        assert sorted(store.touched) == sorted(m.id for m in got)
    finally:
        restore()


def test_dl6_decay_is_relative_and_never_destructive():
    mod, restore = _load()
    try:
        heavy = [
            _Rec("a", "alpha topic keeper", use_count=5, updated_at="2026-01-01"),
            _Rec("b", "bravo topic keeper", use_count=4, updated_at="2026-01-02"),
            _Rec("c", "charlie topic keeper", use_count=3, updated_at="2026-01-03"),
        ]
        idle = _Rec("z", "zulu idle note", use_count=0, updated_at="2026-02-01")
        store = _Canonical(heavy + [idle])
        retr = mod.MemoryRetriever(store, _Vector())
        working = retr.recent_memories(top_n=3)
        working_ids = [m.id for m in working]
        # The unused fact decays out of the working set despite being the
        # most recently updated one; the reinforced facts stay in.
        assert "z" not in working_ids
        assert working_ids == ["a", "b", "c"]
        # ... and the archive keeps it: still active, never deleted.
        assert any(r.id == "z" for r in store.list(active_only=True))
        assert store.deleted == []
    finally:
        restore()


def test_dl7_an_exhausted_budget_yields_empty_never_raises():
    mod, restore = _load()
    try:
        store = _Canonical(_facts(4))
        retr = mod.MemoryRetriever(store, _Vector())
        view = retr.assemble_dual_layer(None, top_n=4, max_tokens=0)
        assert view.block == ""
        assert view.selected_ids == []
        assert retr.working_block(None, max_tokens=0) == ""
        assert retr.retrieve("violet", top_n=0) == []
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
