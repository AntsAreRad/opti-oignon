#!/usr/bin/env python3
"""Canonical store contracts: identifiers allowlisted, users isolated.

The canonical memory store is the source of truth for personal facts. Every
value reaches SQL through a placeholder; the only assembled identifiers are
drawn from frozen allowlists; rows are scoped to their owner; and the sync
hook riding the writes is best-effort by construction. This suite pins those
properties on the real storage engine:

  * CS1 -- an unknown category is coerced to the default at insert time,
    never stored verbatim;
  * CS2 -- a hostile category filter yields an empty list without touching
    the table (the value never reaches the SQL text);
  * CS3 -- an update naming a column outside the allowlist is rejected with
    a ValueError and writes nothing; an invalid category update likewise;
  * CS4 -- an ordering column outside the allowlist is coerced to the
    default ordering and the query still succeeds;
  * CS5 -- rows are owner-scoped: reading, updating, or deleting another
    user's fact reads as absent and mutates nothing;
  * CS6 -- the soft-delete/restore flag round-trips, a second soft delete
    reads as a no-op, and the usage bump never engages the sync hook;
  * CS7 -- a failing sync layer never breaks the write: the fact lands even
    when the availability probe or the publish chain raises.

The canonical module is loaded through the shared isolation window with a
stubbed sync guard, so the publish hook resolves deterministically. The
owner-scoping module is loaded from its source, and the encryption module is
proven unreachable, so the store runs its plain SQLite path: the real engine,
and never a key. The sync modules CS7 neutralises are declared, so the window
puts them back. Local-only. Runs under pytest or the __main__ runner.
"""

import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_STORE = "opti_oignon.memory.canonical_store"


def _load(guard_available=False, guard_calls=None):
    """Load the canonical module in the shared window with a sync stub.

    The stubbed guard answers the availability probe deterministically and
    can record how often it was consulted.
    """
    guard = types.ModuleType("opti_oignon.veilid.guard")
    calls = guard_calls if guard_calls is not None else []

    def veilid_available():
        calls.append(1)
        return guard_available

    guard.veilid_available = veilid_available
    loaded, restore = isolate(
        targets={
            "opti_oignon.user_isolation": source("user_isolation.py"),
            _STORE: source("memory", "canonical_store.py"),
        },
        blocked=(
            "opti_oignon.db_encryption",
            "opti_oignon.veilid.records",
            "opti_oignon.veilid.sync_engine",
        ),
        seeded={"opti_oignon.veilid.guard": guard},
        packages=("opti_oignon.memory", "opti_oignon.veilid"),
    )
    return loaded[_STORE], restore


_HOSTILE = "'; DROP TABLE memory_facts; --"


def test_cs1_an_unknown_category_is_coerced_at_insert():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.CanonicalMemoryStore(Path(td) / "facts.db")
            odd = store.add("the user waters ferns", "not-a-category")
            assert odd.category == mod.DEFAULT_CATEGORY
            assert store.get(odd.id).category == mod.DEFAULT_CATEGORY
            kept = store.add("the user rows on weekends", "preference")
            assert store.get(kept.id).category == "preference"
        finally:
            restore()


def test_cs2_a_hostile_category_filter_reads_empty_and_leaves_the_table():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.CanonicalMemoryStore(Path(td) / "facts.db")
            store.add("alpha fact", "fact")
            store.add("bravo goal", "goal")
            assert store.list(category=_HOSTILE) == []
            assert store.count(active_only=False) == 2
            assert len(store.list()) == 2
        finally:
            restore()


def test_cs3_updates_outside_the_column_allowlist_are_rejected():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.CanonicalMemoryStore(Path(td) / "facts.db")
            rec = store.add("the user keeps bees", "fact")
            rejected = False
            try:
                store.update(rec.id, sneaky="x")
            except ValueError:
                rejected = True
            assert rejected, "a column outside the allowlist must raise ValueError"
            assert store.get(rec.id).text == "the user keeps bees"
            rejected_cat = False
            try:
                store.update(rec.id, category="bogus")
            except ValueError:
                rejected_cat = True
            assert rejected_cat
            assert store.get(rec.id).category == "fact"
        finally:
            restore()


def test_cs4_an_unlisted_ordering_column_is_coerced_to_the_default():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.CanonicalMemoryStore(Path(td) / "facts.db")
            first = store.add("first note", "fact")
            second = store.add("second note", "fact")
            rows = store.list(order_by="id; DROP TABLE memory_facts")
            assert [r.id for r in rows] == [second.id, first.id]
            assert store.count(active_only=False) == 2
        finally:
            restore()


def test_cs5_rows_are_owner_scoped_across_every_verb():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.CanonicalMemoryStore(
                Path(td) / "facts.db", single_user_mode=False
            )
            fact = store.add("alice grows tomatoes", "fact", user_id="alice")
            assert store.get(fact.id, user_id="bob") is None
            assert store.update(fact.id, user_id="bob", text="stolen") is None
            assert store.soft_delete(fact.id, user_id="bob") is False
            assert store.hard_delete(fact.id, user_id="bob") is False
            mine = store.get(fact.id, user_id="alice")
            assert mine is not None
            assert mine.text == "alice grows tomatoes"
            assert mine.active is True
        finally:
            restore()


def test_cs6_soft_delete_round_trips_and_the_usage_bump_stays_local():
    with tempfile.TemporaryDirectory() as td:
        calls = []
        mod, restore = _load(guard_calls=calls)
        try:
            store = mod.CanonicalMemoryStore(Path(td) / "facts.db")
            rec = store.add("the user sails", "fact")
            assert store.soft_delete(rec.id) is True
            gone = store.get(rec.id)
            assert gone.active is False
            assert all(r.id != rec.id for r in store.list(active_only=True))
            assert any(r.id == rec.id for r in store.list(active_only=False))
            assert store.soft_delete(rec.id) is False
            assert store.restore(rec.id) is True
            assert store.get(rec.id).active is True
            probes_before = len(calls)
            assert store.touch(rec.id) is True
            assert len(calls) == probes_before, "a usage bump must not probe sync"
            assert store.get(rec.id).use_count == 1
        finally:
            restore()


def test_cs7_a_failing_sync_layer_never_breaks_the_write():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(guard_available=True)
        try:
            # The probe passes but the publish chain is unimportable: the
            # write must land and the failure must stay inside the hook.
            sys.modules["opti_oignon.veilid.records"] = None
            rec = None
            store = mod.CanonicalMemoryStore(Path(td) / "facts.db")
            rec = store.add("the user hikes", "fact")
            assert rec is not None
            assert store.get(rec.id) is not None

            # The guard itself unimportable: same posture.
            sys.modules["opti_oignon.veilid.guard"] = None
            rec2 = store.add("the user paints", "fact")
            assert store.get(rec2.id) is not None
            assert store.count() == 2
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
