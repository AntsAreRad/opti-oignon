#!/usr/bin/env python3
"""Tests for S179 Goal 3 -- the per-device change feed (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/change_feed.py:

- Journalling and the watermark: recording assigns monotonic sequences; since(0)
  returns the whole current set; since(high_water) returns nothing; a delta carries
  the journal's high-water so a peer advances past consumed sequences.
- Delta semantics: a delta collapses to the latest version per key; a key changed
  several times since the watermark returns once, at its newest version.
- SQL hygiene: WAL is the journal mode; the table identifier comes from a frozenset
  allowlist; the module uses no f-strings (so no f-string SQL).
- Integrity: a record with a mismatched hash is refused; a corrupt journal payload
  is skipped on read rather than crashing.
- Isolation and the singleton: an injected root keeps the DB under a temp dir; the
  singleton get / set / reset hooks behave; two feeds on different roots are
  independent.

Loaded via spec_from_file_location with opti_oignon stubbed. records is loaded
first so change_feed's import resolves; the data-directory import stays unused
because every feed here is constructed with an explicit root.
"""

import dataclasses
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
records = _load("records")  # change_feed imports this; load it first
change_feed = _load("change_feed")
RecordKind = records.RecordKind


def _rec(record_id, clock, *, device="A", payload=None, kind=None, deleted=False):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
        deleted=deleted,
    )


@pytest.fixture
def feed(tmp_path):
    f = change_feed.ChangeFeed(root=tmp_path)
    yield f
    f.close()


@pytest.fixture(autouse=True)
def _reset_singleton():
    change_feed.reset_change_feed()
    yield
    change_feed.reset_change_feed()


# Journalling and the watermark


class TestJournalling:
    def test_record_returns_monotonic_sequences(self, feed):
        s1 = feed.record(_rec("a", 1))
        s2 = feed.record(_rec("b", 1))
        s3 = feed.record(_rec("c", 1))
        assert s1 < s2 < s3
        assert feed.count() == 3

    def test_high_water_tracks_max_sequence(self, feed):
        assert feed.high_water() == 0
        last = feed.record(_rec("a", 1))
        assert feed.high_water() == last

    def test_since_zero_returns_current_set(self, feed):
        feed.record(_rec("a", 1))
        feed.record(_rec("b", 1))
        delta = feed.since(0)
        assert {r.record_id for r in delta.records} == {"a", "b"}
        assert delta.high_water == feed.high_water()

    def test_since_high_water_returns_nothing(self, feed):
        feed.record(_rec("a", 1))
        hw = feed.high_water()
        delta = feed.since(hw)
        assert delta.records == []
        assert delta.high_water == hw

    def test_delta_advances_watermark(self, feed):
        feed.record(_rec("a", 1))
        first = feed.since(0)
        feed.record(_rec("b", 1))
        second = feed.since(first.high_water)
        assert {r.record_id for r in second.records} == {"b"}
        assert second.high_water > first.high_water


# Delta semantics: latest per key


class TestDeltaSemantics:
    def test_repeated_change_collapses_to_latest(self, feed):
        feed.record(_rec("a", 1, payload={"text": "first"}))
        feed.record(_rec("a", 2, payload={"text": "second"}))
        feed.record(_rec("a", 3, payload={"text": "third"}))
        delta = feed.since(0)
        assert len(delta.records) == 1
        assert delta.records[0].clock == 3
        assert delta.records[0].payload == {"text": "third"}

    def test_current_records_is_latest_per_key(self, feed):
        feed.record(_rec("a", 1, payload={"t": "a1"}))
        feed.record(_rec("a", 2, payload={"t": "a2"}))
        feed.record(_rec("b", 1, payload={"t": "b1"}))
        current = {r.record_id: r.clock for r in feed.current_records()}
        assert current == {"a": 2, "b": 1}

    def test_delta_only_includes_changes_after_watermark(self, feed):
        feed.record(_rec("a", 1))
        w = feed.high_water()
        feed.record(_rec("a", 2, payload={"text": "newer"}))
        feed.record(_rec("c", 1))
        delta = feed.since(w)
        ids = {r.record_id for r in delta.records}
        assert ids == {"a", "c"}
        a = next(r for r in delta.records if r.record_id == "a")
        assert a.clock == 2

    def test_tombstone_is_journalled_and_returned(self, feed):
        feed.record(_rec("a", 1))
        feed.record(_rec("a", 2, payload={}, deleted=True))
        delta = feed.since(0)
        assert len(delta.records) == 1
        assert delta.records[0].deleted is True


# SQL hygiene


class TestSqlHygiene:
    def test_journal_mode_is_wal(self, feed):
        feed.record(_rec("a", 1))
        assert feed.journal_mode() == "wal"

    def test_table_allowlist_is_frozenset(self):
        assert isinstance(change_feed._TABLES, frozenset)
        assert change_feed.TABLE_NAME in change_feed._TABLES

    def test_safe_table_rejects_unknown(self):
        with pytest.raises(ValueError):
            change_feed._safe_table("evil; DROP TABLE x")

    def test_module_has_no_fstrings(self):
        src = (VEILID / "change_feed.py").read_text(encoding="utf-8")
        assert 'f"' not in src and "f'" not in src

    def test_sql_constants_reference_only_allowed_table(self):
        for sql in (
            change_feed._INSERT,
            change_feed._SELECT_SINCE,
            change_feed._SELECT_ALL,
            change_feed._SELECT_MAX_SEQ,
        ):
            assert change_feed.TABLE_NAME in sql


# Integrity


class TestIntegrity:
    def test_refuses_mismatched_hash(self, feed):
        good = _rec("a", 1)
        bad = dataclasses.replace(good, content_hash="0" * 64)
        with pytest.raises(ValueError):
            feed.record(bad)

    def test_refuses_non_record(self, feed):
        with pytest.raises(TypeError):
            feed.record({"not": "a record"})

    def test_corrupt_payload_row_is_skipped(self, feed, tmp_path):
        feed.record(_rec("a", 1))
        feed.record(_rec("b", 1))
        # Corrupt one row's payload directly in the DB; the reader must skip it.
        conn = sqlite3.connect(str(feed.db_path))
        conn.execute(
            "UPDATE {} SET payload = ? WHERE record_id = ?".format(
                change_feed.TABLE_NAME
            ),
            ("{ not json", "a"),
        )
        conn.commit()
        conn.close()
        current = feed.current_records()
        assert {r.record_id for r in current} == {"b"}


# Isolation and the singleton


class TestSingletonAndIsolation:
    def test_db_lives_under_injected_root(self, tmp_path):
        f = change_feed.ChangeFeed(root=tmp_path)
        f.record(_rec("a", 1))
        assert f.db_path.parent == tmp_path
        assert f.db_path.exists()
        f.close()

    def test_two_roots_are_independent(self, tmp_path):
        a = change_feed.ChangeFeed(root=tmp_path / "A")
        b = change_feed.ChangeFeed(root=tmp_path / "B")
        a.record(_rec("x", 1))
        assert a.count() == 1
        assert b.count() == 0
        a.close()
        b.close()

    def test_singleton_get_set_reset(self, tmp_path):
        f = change_feed.ChangeFeed(root=tmp_path)
        change_feed.set_change_feed(f)
        assert change_feed.get_change_feed() is f
        change_feed.reset_change_feed()
        # After reset, get creates a fresh instance (not the old one).
        fresh = change_feed.get_change_feed(root=tmp_path)
        assert fresh is not f
        fresh.close()

    def test_clear_empties_journal(self, feed):
        feed.record(_rec("a", 1))
        feed.record(_rec("b", 1))
        assert feed.count() == 2
        feed.clear()
        assert feed.count() == 0
        assert feed.since(0).records == []
