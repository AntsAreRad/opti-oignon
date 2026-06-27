#!/usr/bin/env python3
"""Tests for the legacy -> new-store migration (M3a).

``migration.migrate_legacy_to_store`` is the one-shot that copies the legacy
``memories.db`` facts into the coordinated ``MemoryStore`` so the two-store split
ends with a single source of truth. It is idempotent (the store's dedup merges a
re-run instead of duplicating), marker-guarded (so it runs once at startup), maps
unknown legacy categories onto the canonical set, and never raises (a migration
failure must not break startup). This suite loads ``migration.py`` in isolation
(a fake legacy manager + a fake store injected) and proves:

  * every legacy fact is added to the store, with counts (added/merged/scanned);
  * a store "merge" decision is counted as merged, not added;
  * an existing marker skips a re-run, and ``force=True`` overrides it;
  * an unknown legacy category is mapped to the default;
  * an empty legacy store is a clean no-op that still writes the marker;
  * a raising manager is swallowed and reported, never propagated.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load():
    keys = ("opti_oignon", "opti_oignon.memory", "opti_oignon.memory.migration")
    saved = {k: sys.modules.get(k) for k in keys}
    for n in ("opti_oignon", "opti_oignon.memory"):
        pkg = types.ModuleType(n)
        pkg.__path__ = []
        sys.modules[n] = pkg
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.memory.migration", _OO / "memory" / "migration.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.memory.migration"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


@dataclass
class FFact:
    fact: str
    category: str = "fact"


@dataclass
class FDecision:
    action: str = "add"


class FakeManager:
    def __init__(self, facts, *, raises=False):
        self._facts = facts
        self._raises = raises

    def get_all_facts(self, *, active_only=True, category=None):
        if self._raises:
            raise RuntimeError("db down")
        return list(self._facts)


class FakeStore:
    def __init__(self, *, merge_texts=()):
        self.added = []
        self._merge_texts = set(merge_texts)

    def add(self, text, category="fact", *, source="", user_id=None, embedding=None):
        self.added.append({"text": text, "category": category, "source": source})
        action = "merge" if text in self._merge_texts else "add"
        return (object(), FDecision(action=action))


def _marker():
    d = Path(tempfile.mkdtemp(prefix="oo-mig-"))
    return d / ".legacy_memory_migrated"


def test_migrates_all_facts():
    mod, restore = _load()
    try:
        mgr = FakeManager([FFact("name is Leon", "identity"),
                           FFact("likes tea", "preference"),
                           FFact("works at Acme", "project")])
        store = FakeStore()
        marker = _marker()
        res = mod.migrate_legacy_to_store(manager=mgr, store=store, marker_path=marker)
        assert res["scanned"] == 3
        assert res["added"] == 3
        assert res["merged"] == 0
        assert len(store.added) == 3
        assert all(a["source"] == "legacy-import" for a in store.added)
        assert marker.exists()
    finally:
        restore()


def test_counts_merges():
    mod, restore = _load()
    try:
        mgr = FakeManager([FFact("name is Leon"), FFact("dup fact")])
        store = FakeStore(merge_texts={"dup fact"})
        res = mod.migrate_legacy_to_store(manager=mgr, store=store, marker_path=_marker())
        assert res["added"] == 1
        assert res["merged"] == 1
    finally:
        restore()


def test_marker_skips_rerun():
    mod, restore = _load()
    try:
        marker = _marker()
        marker.write_text("migrated\n", encoding="utf-8")
        store = FakeStore()
        res = mod.migrate_legacy_to_store(
            manager=FakeManager([FFact("x")]), store=store, marker_path=marker
        )
        assert res["skipped_marker"] is True
        assert store.added == []
    finally:
        restore()


def test_force_ignores_marker():
    mod, restore = _load()
    try:
        marker = _marker()
        marker.write_text("migrated\n", encoding="utf-8")
        store = FakeStore()
        res = mod.migrate_legacy_to_store(
            manager=FakeManager([FFact("x")]), store=store, marker_path=marker, force=True
        )
        assert res["skipped_marker"] is False
        assert len(store.added) == 1
    finally:
        restore()


def test_unknown_category_mapped():
    mod, restore = _load()
    try:
        store = FakeStore()
        mod.migrate_legacy_to_store(
            manager=FakeManager([FFact("odd", "weird-category")]),
            store=store, marker_path=_marker(),
        )
        assert store.added[0]["category"] == "fact"   # mapped to default
    finally:
        restore()


def test_empty_legacy_noop():
    mod, restore = _load()
    try:
        marker = _marker()
        res = mod.migrate_legacy_to_store(
            manager=FakeManager([]), store=FakeStore(), marker_path=marker
        )
        assert res["scanned"] == 0
        assert res["added"] == 0
        assert marker.exists()
    finally:
        restore()


def test_manager_raises_swallowed():
    mod, restore = _load()
    try:
        res = mod.migrate_legacy_to_store(
            manager=FakeManager([], raises=True), store=FakeStore(), marker_path=_marker()
        )
        assert res["error"] is not None    # reported, not raised
    finally:
        restore()


# ---------------------------------------------------------------------------
# run_boot_migration: the startup adapter the app lifespan calls (M3a-startup).
# It delegates to migrate_legacy_to_store with defaults (idempotent +
# marker-guarded) and, like that function, never raises -- a migration problem
# must never break the boot.
# ---------------------------------------------------------------------------
def test_run_boot_migration_delegates_and_returns_result():
    mod, restore = _load()
    try:
        captured = {"called": 0, "kwargs": None}

        def _spy(**kwargs):
            captured["called"] += 1
            captured["kwargs"] = kwargs
            return {"scanned": 3, "added": 2, "merged": 1,
                    "skipped_marker": False, "error": None}

        mod.migrate_legacy_to_store = _spy            # swap the delegate
        res = mod.run_boot_migration()
        assert captured["called"] == 1                # called exactly once
        assert captured["kwargs"] == {}               # defaults: marker-guarded run
        assert res["added"] == 2 and res["merged"] == 1
        assert res["error"] is None
    finally:
        restore()


def test_run_boot_migration_swallows_errors():
    mod, restore = _load()
    try:
        def _boom(**kwargs):
            raise RuntimeError("migrate exploded")

        mod.migrate_legacy_to_store = _boom
        res = mod.run_boot_migration()                # must NOT raise
        assert res["error"] is not None               # reported in the result
        assert res["added"] == 0 and res["merged"] == 0
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
