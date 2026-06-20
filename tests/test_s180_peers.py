#!/usr/bin/env python3
"""Tests for S180 Goal 1 -- the per-peer registry and watermark store (Theme 4).

Covers opti_oignon/veilid/peers.py:

- Registry: add returns a stored record at watermark 0; get / has reflect
  presence; list is ordered by pairing time then peer id; remove deletes and
  reports; count tracks the set; a re-pair upserts the routing key and label while
  preserving the watermark and the original pairing time; bad inputs raise.
- The monotonic watermark: an unknown peer reads 0; advancing moves forward;
  a smaller or equal incoming value never regresses the stored watermark;
  advancing an unregistered peer is a no-op returning 0; bad inputs raise.
- SQL hygiene: WAL is the journal mode; the table identifier comes from a frozenset
  allowlist and the safe-table guard rejects anything else; the module uses no
  f-strings (so no f-string SQL); every SQL constant names only the allowed table.
- Isolation and the singleton: an injected root keeps the DB under a temp dir; the
  get / set / reset hooks behave; two stores on different roots are independent.

Loaded via spec_from_file_location with opti_oignon stubbed. The store imports
only the standard library, so it loads standalone; the data-directory import stays
unused because every store here is constructed with an explicit root.
"""

import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"


def _ensure_stubs() -> None:
    for name, sub in (("opti_oignon", OO), ("opti_oignon.veilid", VEILID)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod


def _load(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
peers = _load("peers")
PeerStore = peers.PeerStore
PeerRecord = peers.PeerRecord


@pytest.fixture
def store(tmp_path):
    s = PeerStore(root=tmp_path)
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _reset_singleton():
    peers.reset_peer_store()
    yield
    peers.reset_peer_store()


# Registry: add / get / has / list / remove / count


class TestRegistry:
    def test_add_returns_record_at_watermark_zero(self, store):
        rec = store.add_peer("dev-1", "RK1", label="laptop")
        assert isinstance(rec, PeerRecord)
        assert rec.peer_id == "dev-1"
        assert rec.routing_key == "RK1"
        assert rec.label == "laptop"
        assert rec.watermark == 0
        assert rec.added_at and rec.updated_at

    def test_add_defaults_empty_label(self, store):
        rec = store.add_peer("dev-1", "RK1")
        assert rec.label == ""

    def test_get_returns_record_and_none(self, store):
        store.add_peer("dev-1", "RK1")
        got = store.get_peer("dev-1")
        assert got is not None and got.peer_id == "dev-1"
        assert store.get_peer("missing") is None
        assert store.get_peer("") is None

    def test_has_peer(self, store):
        assert store.has_peer("dev-1") is False
        store.add_peer("dev-1", "RK1")
        assert store.has_peer("dev-1") is True

    def test_list_ordered_by_added_then_id(self, store):
        store.add_peer("b", "RKb")
        store.add_peer("a", "RKa")
        store.add_peer("c", "RKc")
        ids = [p.peer_id for p in store.list_peers()]
        # added_at ascending is the primary order; insertion order here.
        assert ids == ["b", "a", "c"]

    def test_count_tracks_set(self, store):
        assert store.count() == 0
        store.add_peer("a", "RKa")
        store.add_peer("b", "RKb")
        assert store.count() == 2
        store.remove_peer("a")
        assert store.count() == 1

    def test_remove_reports_and_deletes(self, store):
        store.add_peer("a", "RKa")
        assert store.remove_peer("a") is True
        assert store.has_peer("a") is False
        assert store.remove_peer("a") is False
        assert store.remove_peer("never") is False
        assert store.remove_peer("") is False

    def test_repair_preserves_watermark_and_added_at(self, store):
        first = store.add_peer("dev-1", "RK1", label="old")
        store.advance_watermark("dev-1", 7)
        again = store.add_peer("dev-1", "RK2", label="new")
        assert again.routing_key == "RK2"
        assert again.label == "new"
        assert again.watermark == 7  # preserved across the re-pair
        assert again.added_at == first.added_at  # original pairing time kept
        assert store.count() == 1  # upsert, not a second row

    def test_add_validates_inputs(self, store):
        with pytest.raises(ValueError):
            store.add_peer("", "RK1")
        with pytest.raises(ValueError):
            store.add_peer("dev-1", "")
        with pytest.raises(ValueError):
            store.add_peer("dev-1", "RK1", label=123)  # type: ignore[arg-type]


# The monotonic watermark


class TestMonotonicWatermark:
    def test_unknown_peer_reads_zero(self, store):
        assert store.get_watermark("nope") == 0
        assert store.get_watermark("") == 0

    def test_advance_moves_forward(self, store):
        store.add_peer("dev-1", "RK1")
        assert store.advance_watermark("dev-1", 5) == 5
        assert store.get_watermark("dev-1") == 5

    def test_advance_never_regresses(self, store):
        store.add_peer("dev-1", "RK1")
        store.advance_watermark("dev-1", 10)
        assert store.advance_watermark("dev-1", 3) == 10  # smaller ignored
        assert store.advance_watermark("dev-1", 10) == 10  # equal ignored
        assert store.advance_watermark("dev-1", 11) == 11  # larger advances
        assert store.get_watermark("dev-1") == 11

    def test_advance_unknown_is_noop(self, store):
        assert store.advance_watermark("ghost", 9) == 0
        assert store.has_peer("ghost") is False
        assert store.count() == 0

    def test_advance_validates_inputs(self, store):
        store.add_peer("dev-1", "RK1")
        with pytest.raises(ValueError):
            store.advance_watermark("dev-1", -1)
        with pytest.raises(ValueError):
            store.advance_watermark("dev-1", True)  # bool rejected though int subclass
        with pytest.raises(ValueError):
            store.advance_watermark("dev-1", 1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            store.advance_watermark("", 1)

    def test_zero_advance_is_allowed_and_noop(self, store):
        store.add_peer("dev-1", "RK1")
        store.advance_watermark("dev-1", 4)
        assert store.advance_watermark("dev-1", 0) == 4


# SQL hygiene and WAL


class TestSqlHygiene:
    def test_journal_mode_is_wal(self, store):
        assert store.journal_mode() == "wal"

    def test_table_allowlist_is_frozenset(self):
        assert isinstance(peers._TABLES, frozenset)
        assert peers.TABLE_NAME in peers._TABLES

    def test_safe_table_rejects_unknown(self):
        with pytest.raises(ValueError):
            peers._safe_table("evil; DROP TABLE x")

    def test_module_has_no_fstrings(self):
        src = (VEILID / "peers.py").read_text(encoding="utf-8")
        assert 'f"' not in src and "f'" not in src

    def test_sql_constants_reference_only_allowed_table(self):
        for sql in (
            peers._CREATE_TABLE,
            peers._UPSERT,
            peers._SELECT_ONE,
            peers._SELECT_ALL,
            peers._SELECT_WATERMARK,
            peers._ADVANCE,
            peers._DELETE_ONE,
        ):
            assert peers.TABLE_NAME in sql

    def test_advance_sql_uses_scalar_max(self):
        # The monotonic guarantee rides on the SQL scalar max(watermark, ?).
        assert "max(watermark, ?)" in peers._ADVANCE


# Isolation and the singleton


class TestIsolationSingleton:
    def test_db_path_under_injected_root(self, tmp_path):
        s = PeerStore(root=tmp_path)
        s.add_peer("dev-1", "RK1")
        assert s.db_path.parent == tmp_path
        assert s.db_path.name == peers.DB_FILENAME
        assert s.db_path.exists()
        s.close()

    def test_get_set_reset_singleton(self, tmp_path):
        s = PeerStore(root=tmp_path)
        peers.set_peer_store(s)
        assert peers.get_peer_store() is s
        peers.reset_peer_store()
        fresh = peers.get_peer_store(root=tmp_path)
        assert fresh is not s

    def test_two_stores_independent(self, tmp_path):
        a = PeerStore(root=tmp_path / "a")
        b = PeerStore(root=tmp_path / "b")
        a.add_peer("only-a", "RKa")
        assert a.has_peer("only-a") is True
        assert b.has_peer("only-a") is False
        a.close()
        b.close()

    def test_clear_empties_store(self, store):
        store.add_peer("a", "RKa")
        store.add_peer("b", "RKb")
        store.clear()
        assert store.count() == 0
        assert store.list_peers() == []

    def test_persists_across_reopen(self, tmp_path):
        s1 = PeerStore(root=tmp_path)
        s1.add_peer("dev-1", "RK1", label="laptop")
        s1.advance_watermark("dev-1", 5)
        s1.close()
        s2 = PeerStore(root=tmp_path)
        rec = s2.get_peer("dev-1")
        assert rec is not None
        assert rec.routing_key == "RK1"
        assert rec.watermark == 5
        s2.close()

    def test_stored_row_shape_is_what_we_query(self, store, tmp_path):
        store.add_peer("dev-1", "RK1", label="laptop")
        store.advance_watermark("dev-1", 3)
        conn = sqlite3.connect(str(store.db_path))
        try:
            row = conn.execute(
                "SELECT peer_id, routing_key, label, watermark FROM veilid_peers"
            ).fetchone()
        finally:
            conn.close()
        assert row == ("dev-1", "RK1", "laptop", 3)
