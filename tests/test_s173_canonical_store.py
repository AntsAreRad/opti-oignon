#!/usr/bin/env python3
"""Tests for S173 -- canonical memory store and the memory.py fold-in.

Two kinds of checks:

- Runtime, on ``opti_oignon/memory/canonical_store.py`` loaded in isolation
  (``spec_from_file_location`` + ``sys.modules``), so they collect without
  ollama / fastapi. The real ``db_encryption`` and ``user_isolation`` modules
  are loaded (stdlib-only top imports) to exercise the genuine integration path;
  with SQLCipher absent the connection falls back to plain SQLite, which is the
  Daily-mode path.
- File-content, on the fold-in (the former ``opti_oignon/memory.py`` relocated
  to ``opti_oignon/memory/legacy.py`` with a compatibility re-export) and on the
  SQL-hygiene of the new module, plus the spec registration.
"""

import importlib.util
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
MEM = OO / "memory"
CANON_PATH = MEM / "canonical_store.py"
INIT_PATH = MEM / "__init__.py"
LEGACY_PATH = MEM / "legacy.py"
OLD_MEMORY_MODULE = OO / "memory.py"
SPEC = ROOT / "ODYSSEUS_SPEC.md"

EXPECTED_CATEGORIES = {
    "identity",
    "preference",
    "fact",
    "contact",
    "project",
    "goal",
}
EXPECTED_COLUMNS = {
    "id",
    "text",
    "category",
    "source",
    "user_id",
    "created_at",
    "updated_at",
    "active",
    "use_count",
}


def _load_real(name: str) -> None:
    """Load a real opti_oignon submodule by path (leak-safe, stdlib-only)."""
    if f"opti_oignon.{name}" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        f"opti_oignon.{name}", str(OO / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"opti_oignon.{name}"] = mod
    spec.loader.exec_module(mod)


def _load_canonical():
    """Load canonical_store.py directly, bypassing the package __init__."""
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.memory" not in sys.modules:
        mpkg = types.ModuleType("opti_oignon.memory")
        mpkg.__path__ = [str(MEM)]
        sys.modules["opti_oignon.memory"] = mpkg

    _load_real("db_encryption")
    _load_real("user_isolation")

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.memory.canonical_store", str(CANON_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.memory.canonical_store"] = mod
    spec.loader.exec_module(mod)
    return mod


cs = _load_canonical()


@pytest.fixture
def store(tmp_path):
    return cs.CanonicalMemoryStore(tmp_path / "mf.db")


@pytest.fixture
def multi_store(tmp_path):
    return cs.CanonicalMemoryStore(tmp_path / "mf_multi.db", single_user_mode=False)


def _raw(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


# Module sentinels


class TestModuleSentinels:
    def test_feature_available(self):
        assert cs.FEATURE_AVAILABLE is True

    def test_checkpoint_sentinel(self):
        assert cs.checkpoint_before_apply is True

    def test_has_reset_singleton(self):
        assert hasattr(cs, "reset_canonical_store")
        assert hasattr(cs, "get_canonical_store")


# Schema


class TestSchema:
    def test_table_exists(self, store):
        with _raw(store.db_path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (cs.TABLE_NAME,),
            ).fetchone()
        assert row is not None and row["name"] == "memory_facts"

    def test_columns(self, store):
        with _raw(store.db_path) as conn:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(memory_facts)")}
        assert cols == EXPECTED_COLUMNS

    def test_indexes_present(self, store):
        with _raw(store.db_path) as conn:
            idx = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                )
            }
        for expected in (
            "idx_memory_facts_category",
            "idx_memory_facts_active",
            "idx_memory_facts_user",
        ):
            assert expected in idx

    def test_insert_defaults(self, store):
        r = store.add("a fact", "fact")
        assert r.active is True
        assert r.use_count == 0
        assert r.created_at == r.updated_at


# WAL mode


class TestWalMode:
    def test_journal_mode_is_wal(self, store):
        assert store.journal_mode() == "wal"


# Categories


class TestCategories:
    def test_categories_is_frozenset(self):
        assert isinstance(cs.CATEGORIES, frozenset)
        assert set(cs.CATEGORIES) == EXPECTED_CATEGORIES

    def test_add_each_category(self, store):
        for cat in EXPECTED_CATEGORIES:
            r = store.add("text for " + cat, cat)
            assert r.category == cat

    def test_invalid_category_coerced_to_default(self, store):
        r = store.add("nonsense category", "not-a-category")
        assert r.category == cs.DEFAULT_CATEGORY == "fact"

    def test_update_invalid_category_raises(self, store):
        r = store.add("x", "fact")
        with pytest.raises(ValueError):
            store.update(r.id, category="bogus")


# CRUD


class TestCrud:
    def test_add_get_roundtrip(self, store):
        r = store.add("Leon uses Kubuntu", "fact", source="conv-1")
        got = store.get(r.id)
        assert got is not None
        assert got.text == "Leon uses Kubuntu"
        assert got.source == "conv-1"
        assert got.id

    def test_get_missing_returns_none(self, store):
        assert store.get("does-not-exist") is None

    def test_update_text(self, store):
        r = store.add("old text", "fact")
        time.sleep(0.002)
        updated = store.update(r.id, text="new text")
        assert updated is not None
        assert updated.text == "new text"
        assert updated.updated_at >= r.updated_at

    def test_update_unknown_column_raises(self, store):
        r = store.add("x", "fact")
        with pytest.raises(ValueError):
            store.update(r.id, not_a_column="nope")

    def test_update_missing_returns_none(self, store):
        assert store.update("ghost", text="x") is None

    def test_touch_increments_use_count(self, store):
        r = store.add("counts", "fact")
        assert store.touch(r.id) is True
        assert store.touch(r.id) is True
        assert store.get(r.id).use_count == 2

    def test_list_returns_active(self, store):
        store.add("one", "fact")
        store.add("two", "preference")
        assert len(store.list()) == 2

    def test_list_filter_by_category(self, store):
        store.add("one", "fact")
        store.add("pref", "preference")
        prefs = store.list(category="preference")
        assert len(prefs) == 1 and prefs[0].category == "preference"

    def test_list_unknown_category_empty(self, store):
        store.add("one", "fact")
        assert store.list(category="not-real") == []

    def test_list_order_by_use_count(self, store):
        a = store.add("a", "fact")
        b = store.add("b", "fact")
        store.add("c", "fact")
        store.touch(a.id)
        store.touch(a.id)
        store.touch(b.id)
        ordered = store.list(order_by="use_count", descending=True)
        assert [rec.text for rec in ordered] == ["a", "b", "c"]

    def test_list_invalid_order_by_falls_back(self, store):
        store.add("a", "fact")
        store.add("b", "fact")
        # An injection-looking order_by must be ignored, not interpolated.
        rows = store.list(order_by="use_count; DROP TABLE memory_facts")
        assert len(rows) == 2
        with _raw(store.db_path) as conn:
            still = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_facts'"
            ).fetchone()
        assert still is not None

    def test_list_limit(self, store):
        for i in range(5):
            store.add("fact " + str(i), "fact")
        assert len(store.list(limit=3)) == 3

    def test_count_active_vs_all(self, store):
        r = store.add("one", "fact")
        store.add("two", "fact")
        store.soft_delete(r.id)
        assert store.count() == 1
        assert store.count(active_only=False) == 2


# Delete: soft and hard


class TestDelete:
    def test_soft_delete_clears_active(self, store):
        r = store.add("temp", "fact")
        assert store.soft_delete(r.id) is True
        assert store.get(r.id).active is False
        assert store.list() == []
        assert len(store.list(active_only=False)) == 1

    def test_soft_delete_then_restore(self, store):
        r = store.add("temp", "fact")
        store.soft_delete(r.id)
        assert store.restore(r.id) is True
        assert store.get(r.id).active is True
        assert len(store.list()) == 1

    def test_hard_delete_removes_row(self, store):
        r = store.add("temp", "fact")
        assert store.hard_delete(r.id) is True
        assert store.get(r.id) is None
        assert store.count(active_only=False) == 0
        with _raw(store.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM memory_facts WHERE id=?", (r.id,)
            ).fetchone()
        assert row is None

    def test_soft_delete_missing_returns_false(self, store):
        assert store.soft_delete("ghost") is False

    def test_hard_delete_missing_returns_false(self, store):
        assert store.hard_delete("ghost") is False


# Per-user isolation


class TestPerUserIsolation:
    def test_list_scoped_to_user(self, multi_store):
        multi_store.add("alice fact", "fact", user_id="alice")
        multi_store.add("bob fact", "fact", user_id="bob")
        alice = multi_store.list(user_id="alice")
        assert len(alice) == 1 and alice[0].text == "alice fact"

    def test_get_other_user_returns_none(self, multi_store):
        b = multi_store.add("bob fact", "fact", user_id="bob")
        assert multi_store.get(b.id, user_id="alice") is None
        assert multi_store.get(b.id, user_id="bob") is not None

    def test_count_per_user(self, multi_store):
        multi_store.add("a1", "fact", user_id="alice")
        multi_store.add("a2", "fact", user_id="alice")
        multi_store.add("b1", "fact", user_id="bob")
        assert multi_store.count(user_id="alice") == 2
        assert multi_store.count(user_id="bob") == 1

    def test_hard_delete_scoped(self, multi_store):
        b = multi_store.add("bob fact", "fact", user_id="bob")
        # Attempt to delete bob's row as alice: must not succeed.
        assert multi_store.hard_delete(b.id, user_id="alice") is False
        assert multi_store.get(b.id, user_id="bob") is not None

    def test_soft_delete_scoped(self, multi_store):
        b = multi_store.add("bob fact", "fact", user_id="bob")
        assert multi_store.soft_delete(b.id, user_id="alice") is False
        assert multi_store.get(b.id, user_id="bob").active is True


# Singleton and reset


class TestSingleton:
    def test_get_is_stable_then_reset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cs, "_default_db_path", lambda: tmp_path / "singleton.db")
        cs.reset_canonical_store()
        first = cs.get_canonical_store()
        assert cs.get_canonical_store() is first
        cs.reset_canonical_store()
        assert cs.get_canonical_store() is not first
        cs.reset_canonical_store()


# SQL hygiene (file-content on the new module)


class TestSqlHygiene:
    SQL_KEYWORDS = (
        "select",
        "insert",
        "update",
        "delete",
        "create",
        "alter",
        "drop",
        "from",
        "where",
        "values",
    )

    def _source(self) -> str:
        return CANON_PATH.read_text(encoding="utf-8")

    def test_no_fstring_sql(self):
        text = self._source()
        for line in text.splitlines():
            lower = line.lower()
            has_fstring = 'f"' in line or "f'" in line
            if has_fstring and any(kw in lower for kw in self.SQL_KEYWORDS):
                pytest.fail("Possible f-string SQL: " + line.strip())

    def test_allowlists_are_frozensets(self):
        assert isinstance(cs._UPDATABLE_COLUMNS, frozenset)
        assert isinstance(cs._ORDERABLE_COLUMNS, frozenset)

    def test_uses_str_format_for_dynamic_clause(self):
        assert ".format(" in self._source()

    def test_parameterized_insert(self):
        text = self._source()
        assert "VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0)" in text

    def test_wal_pragma_in_source(self):
        assert "PRAGMA journal_mode=WAL" in self._source()

    def test_integration_wiring_referenced(self):
        text = self._source()
        assert "get_encrypted_connection" in text
        assert "db_encryption" in text
        assert "user_isolation" in text
        assert "effective_user_id" in text


# Fold-in of the former memory.py


class TestFoldIn:
    def test_legacy_module_present(self):
        assert LEGACY_PATH.exists()

    def test_old_memory_module_removed(self):
        assert not OLD_MEMORY_MODULE.exists()

    def test_legacy_contains_manager_and_fact(self):
        text = LEGACY_PATH.read_text(encoding="utf-8")
        assert "class MemoryManager" in text
        assert "class MemoryFact" in text

    def test_init_reexports_legacy_surface(self):
        text = INIT_PATH.read_text(encoding="utf-8")
        for name in (
            "MemoryManager",
            "MemoryFact",
            "memory_manager",
            "OLLAMA_AVAILABLE",
            "extract_facts",
            "extract_and_store",
            "add_fact",
            "get_all_facts",
        ):
            assert name in text, "Missing legacy re-export: " + name

    def test_init_exposes_new_layer(self):
        text = INIT_PATH.read_text(encoding="utf-8")
        for name in (
            "CanonicalMemoryStore",
            "MemoryRecord",
            "get_canonical_store",
            "reset_canonical_store",
        ):
            assert name in text, "Missing new-layer export: " + name


# Cartography / spec registration


class TestSpecRegistration:
    def _spec(self) -> str:
        return SPEC.read_text(encoding="utf-8")

    def test_canonical_registered(self):
        assert "opti_oignon/memory/canonical_store.py" in self._spec()

    def test_legacy_registered(self):
        assert "opti_oignon/memory/legacy.py" in self._spec()

    def test_section_4_1_records_final_shape(self):
        text = self._spec()
        assert "legacy.py" in text
        assert "facade" in text
        assert "relocated verbatim" in text
