#!/usr/bin/env python3
"""Tests for S202 -- sync cycle Bloc 0 lot 4: CHF-02 journal compaction.

Covers opti_oignon/veilid/change_feed.py:

- The transparent supersession rule: ``compact()`` deletes every row superseded
  by a later sequence for the same (kind, record_id) and keeps the latest per
  key; idempotent (a second run deletes 0); sequences are never reused after
  compaction (AUTOINCREMENT), so the watermark semantics hold.
- The transparency property as the test spine: for every watermark in the sweep
  (0, a mid value, exactly high_water, and the CHF-01 impossible watermark),
  ``since(w)`` returns identical results before and after compaction;
  ``high_water`` and ``current_records`` unchanged; the latest tombstone of a
  key survives (deletions keep converging to late-joining peers).
- The decision-2 invariant: per-key ``current_clock`` is unchanged across
  compaction under every current writer -- local mints (current+1), the
  equal-clock tie-break adoption (apply journals a different-content winner at
  the same clock), and the PRT-02 clock-only adoption (same content, higher
  clock).
- The every-N-appends trigger: in-process, OFF by default, validated
  constructor arg; fires at N, resets and fires again; a disabled trigger pays
  nothing; a batch ticks the counter once, after the commit.
- record_many folded into one transaction (the F9b note): all-or-nothing --
  pre-insert verification, single commit, mid-batch failure rolls the whole
  batch back and journals nothing; sequences returned in order. No production
  or test caller observed the old per-record commit (grep-verified), so no
  deselect-plus-reassert applies.
- Failure posture: a trigger-fired compaction failure never breaks the append
  (swallow-and-log, the sequence returns); the on-demand entry point
  propagates. Mode-free: compaction is local-disk maintenance, works in Bulbe,
  and the module references no mode gate (structural).
- Forward-compatibility (CHF-05): compaction touches feed rows only; a foreign
  table in the same database survives untouched.

Loaded via spec_from_file_location with opti_oignon stubbed (the lot-1/2/3 and
F9 idiom): security_mode stubbed and flippable, signed_audit_log a no-op,
modules registered in sys.modules before exec_module. Every feed is built on a
tmp root; the real guard/records/reconcile/protocol modules are used so the
apply-path invariant tests exercise the genuine writers.
"""

import importlib.util
import logging
import sqlite3
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
VEILID = OO / "veilid"

_MODE = {"fn": (lambda: "daily")}


def set_mode(value: str = "daily") -> None:
    def _gm() -> str:
        return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def _ensure_stubs() -> None:
    for name, sub in (("opti_oignon", OO), ("opti_oignon.veilid", VEILID)):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(sub)]
            sys.modules[name] = mod
    if "opti_oignon.security_mode" not in sys.modules:
        sm = types.ModuleType("opti_oignon.security_mode")
        sm.get_current_mode = _MODE["fn"]  # type: ignore[attr-defined]
        sys.modules["opti_oignon.security_mode"] = sm
    if "opti_oignon.signed_audit_log" not in sys.modules:
        al = types.ModuleType("opti_oignon.signed_audit_log")
        al.chain_log = lambda **kwargs: None  # type: ignore[attr-defined]
        sys.modules["opti_oignon.signed_audit_log"] = al


def _load(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, str(VEILID / f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod  # register before exec (3.12+ dataclass processing)
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load("guard")
records = _load("records")
reconcile = _load("reconcile")
change_feed = _load("change_feed")
protocol = _load("protocol")
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


def _snapshot(delta):
    """A comparable view of a Delta: per-key record identity plus high_water."""
    by_key = {
        records.key_of(r): (
            r.clock,
            r.content_hash,
            r.deleted,
            dict(r.payload),
            r.device,
            r.updated_at,
        )
        for r in delta.records
    }
    return (by_key, delta.high_water)


@pytest.fixture(autouse=True)
def _daily():
    set_mode("daily")
    yield
    set_mode("daily")


@pytest.fixture
def feed(tmp_path):
    f = change_feed.ChangeFeed(root=tmp_path)
    yield f
    f.close()


def _fill(feed):
    """A representative journal: multi-version keys, a tombstone, two kinds."""
    feed.record(_rec("a", 1))
    feed.record(_rec("a", 2))
    feed.record(_rec("b", 1))
    feed.record(_rec("a", 3))
    feed.record(_rec("c", 1))
    feed.record(_rec("c", 2, deleted=True, payload={}))
    feed.record(_rec("m", 1, kind=RecordKind.MEMORY_CANONICAL))
    feed.record(_rec("m", 2, kind=RecordKind.MEMORY_CANONICAL))


# The supersession rule


class TestCompactionBasics:
    def test_deletes_superseded_keeps_latest_per_key(self, feed):
        _fill(feed)
        assert feed.count() == 8
        deleted = feed.compact()
        assert deleted == 4
        assert feed.count() == 4
        served = {records.key_of(r): r for r in feed.current_records()}
        assert served[("conversation", "a")].clock == 3
        assert served[("conversation", "b")].clock == 1
        assert served[("memory_canonical", "m")].clock == 2

    def test_returns_deleted_count_and_is_idempotent(self, feed):
        _fill(feed)
        assert feed.compact() == 4
        assert feed.compact() == 0
        assert feed.count() == 4

    def test_empty_feed_compacts_to_zero(self, feed):
        assert feed.compact() == 0
        assert feed.count() == 0
        assert feed.high_water() == 0

    def test_single_version_keys_are_untouched(self, feed):
        feed.record(_rec("a", 1))
        feed.record(_rec("b", 1))
        assert feed.compact() == 0
        assert feed.count() == 2

    def test_sequences_are_not_reused_after_compaction(self, feed):
        for c in range(1, 5):
            feed.record(_rec("a", c))
        assert feed.high_water() == 4
        feed.compact()
        seq = feed.record(_rec("a", 5))
        # AUTOINCREMENT: the counter persists past deleted rows, so the
        # watermark a peer holds never aliases a new row.
        assert seq == 5
        assert feed.high_water() == 5


# Tombstones


class TestTombstones:
    def test_latest_tombstone_survives_compaction(self, feed):
        feed.record(_rec("a", 1))
        feed.record(_rec("a", 2, deleted=True, payload={}))
        assert feed.compact() == 1
        served = feed.since(0).records
        assert len(served) == 1
        assert served[0].deleted is True
        assert served[0].clock == 2

    def test_recreate_after_tombstone_keeps_only_recreate(self, feed):
        feed.record(_rec("a", 1))
        feed.record(_rec("a", 2, deleted=True, payload={}))
        feed.record(_rec("a", 3, payload={"v": "back"}))
        assert feed.compact() == 2
        served = feed.since(0).records
        assert len(served) == 1
        assert served[0].deleted is False
        assert served[0].clock == 3
        assert served[0].payload["v"] == "back"


# The transparency property


class TestTransparency:
    @pytest.mark.parametrize("which", ["zero", "mid", "exact", "impossible"])
    def test_since_sweep_identical_pre_post(self, feed, which):
        _fill(feed)
        hw = feed.high_water()
        w = {"zero": 0, "mid": hw // 2, "exact": hw, "impossible": hw + 7}[which]
        before = _snapshot(feed.since(w))
        feed.compact()
        after = _snapshot(feed.since(w))
        assert after == before

    def test_high_water_unchanged(self, feed):
        _fill(feed)
        hw = feed.high_water()
        feed.compact()
        assert feed.high_water() == hw

    def test_current_records_unchanged(self, feed):
        _fill(feed)
        before = {records.key_of(r): r.content_hash for r in feed.current_records()}
        feed.compact()
        after = {records.key_of(r): r.content_hash for r in feed.current_records()}
        assert after == before

    def test_chf01_backstop_serves_full_set_post_compaction(self, feed):
        _fill(feed)
        feed.compact()
        backstop = feed.since(feed.high_water() + 100)
        assert len(backstop.records) == 4
        assert backstop.high_water == feed.high_water()


# The decision-2 invariant: current_clock per key unchanged across compaction


class TestCurrentClockInvariant:
    def test_local_mints_keep_current_clock(self, feed):
        for c in range(1, 5):
            feed.record(_rec("a", c))
        assert feed.current_clock(RecordKind.CONVERSATION, "a") == 4
        feed.compact()
        assert feed.current_clock(RecordKind.CONVERSATION, "a") == 4

    def test_equal_clock_tiebreak_adoption_keeps_current_clock(self, feed):
        # Two candidates at the same clock with different content; the
        # reconciler's tie-break is (clock, content_hash, device, updated_at),
        # so order them at runtime and journal the LOSER locally first.
        x = _rec("k", 2, device="A", payload={"v": "left"})
        y = _rec("k", 2, device="B", payload={"v": "right"})
        loser, winner = sorted(
            (x, y), key=lambda r: (r.clock, r.content_hash, r.device, r.updated_at)
        )
        feed.record(_rec("k", 1, device=loser.device))
        feed.record(loser)
        batch = protocol.RecordBatch(
            device=winner.device, high_water=99, records=[winner], rejected=0
        )
        result = protocol.apply_record_batch(feed, batch)
        assert result.applied == 1  # different content at the same clock: adopted
        assert feed.current_clock(RecordKind.CONVERSATION, "k") == 2
        feed.compact()
        assert feed.current_clock(RecordKind.CONVERSATION, "k") == 2
        served = {records.key_of(r): r for r in feed.current_records()}
        assert served[("conversation", "k")].content_hash == winner.content_hash

    def test_prt02_clock_only_adoption_keeps_current_clock(self, feed):
        same_payload = {"v": "same"}
        r1 = _rec("k", 1, device="A", payload=same_payload)
        r3 = _rec("k", 3, device="A", payload=same_payload)
        assert r1.content_hash == r3.content_hash  # the PRT-02 premise
        feed.record(r1)
        batch = protocol.RecordBatch(
            device="B", high_water=99, records=[r3], rejected=0
        )
        result = protocol.apply_record_batch(feed, batch)
        assert result.applied == 1  # clock-only adoption journalled (PRT-02)
        assert feed.current_clock(RecordKind.CONVERSATION, "k") == 3
        feed.compact()
        assert feed.current_clock(RecordKind.CONVERSATION, "k") == 3


# The every-N-appends trigger


class TestTrigger:
    def test_fires_at_n(self, tmp_path):
        f = change_feed.ChangeFeed(root=tmp_path, compact_every=3)
        try:
            for c in range(1, 4):
                f.record(_rec("a", c))
            assert f.count() == 1  # the third append fired the compaction
            assert f.current_clock(RecordKind.CONVERSATION, "a") == 3
        finally:
            f.close()

    def test_counter_resets_and_fires_again(self, tmp_path):
        f = change_feed.ChangeFeed(root=tmp_path, compact_every=3)
        try:
            for c in range(1, 4):
                f.record(_rec("a", c))
            assert f.count() == 1
            for c in range(4, 7):
                f.record(_rec("a", c))
            assert f.count() == 1  # fired again after the next N appends
            assert f.current_clock(RecordKind.CONVERSATION, "a") == 6
        finally:
            f.close()

    def test_default_is_off(self, feed):
        for c in range(1, 8):
            feed.record(_rec("a", c))
        assert feed.count() == 7  # a bare constructor never auto-compacts

    def test_disabled_trigger_pays_nothing(self, tmp_path):
        f = change_feed.ChangeFeed(root=tmp_path, compact_every=None)
        try:
            for c in range(1, 6):
                f.record(_rec("a", c))
            assert f.count() == 5
            # The increment is conditional: disabled costs one None check and
            # the counter never moves.
            assert f._appends_since_compact == 0
        finally:
            f.close()

    @pytest.mark.parametrize("bad", [True, False, 0, -1, "3", 2.5])
    def test_constructor_validation_rejects(self, tmp_path, bad):
        with pytest.raises(ValueError):
            change_feed.ChangeFeed(root=tmp_path, compact_every=bad)

    @pytest.mark.parametrize("ok", [1, 7, 1000])
    def test_constructor_validation_accepts(self, tmp_path, ok):
        f = change_feed.ChangeFeed(root=tmp_path, compact_every=ok)
        f.close()

    def test_batch_ticks_counter_once_after_commit(self, tmp_path):
        f = change_feed.ChangeFeed(root=tmp_path, compact_every=3)
        try:
            f.record_many([_rec("a", 1), _rec("a", 2), _rec("a", 3)])
            assert f.count() == 1  # threshold met by the batch, fired post-commit
        finally:
            f.close()

    def test_counter_accumulates_across_batches(self, tmp_path):
        f = change_feed.ChangeFeed(root=tmp_path, compact_every=5)
        try:
            f.record_many([_rec("a", 1), _rec("a", 2), _rec("a", 3)])
            assert f.count() == 3  # below threshold: nothing fired
            f.record_many([_rec("a", 4), _rec("a", 5)])
            assert f.count() == 1  # cumulative 5 appends: fired
        finally:
            f.close()


# record_many folded into one transaction (F9b note)


class _PoisonSecondInsert:
    """A connection proxy that fails the second INSERT, delegating the rest."""

    def __init__(self, real):
        self._real = real
        self._inserts = 0

    def execute(self, sql, *args, **kwargs):
        if sql.lstrip().upper().startswith("INSERT"):
            self._inserts += 1
            if self._inserts == 2:
                raise sqlite3.OperationalError("poisoned mid-batch")
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestRecordManyFolded:
    def test_sequences_in_order_single_batch(self, feed):
        seqs = feed.record_many([_rec("a", 1), _rec("b", 1), _rec("a", 2)])
        assert seqs == [1, 2, 3]
        assert feed.count() == 3
        assert feed.high_water() == 3

    def test_bad_hash_journals_nothing(self, feed):
        import dataclasses

        good = _rec("a", 1)
        bad = dataclasses.replace(_rec("b", 1), content_hash="0" * 64)
        with pytest.raises(ValueError):
            feed.record_many([good, bad, _rec("c", 1)])
        # All-or-nothing: verification precedes any insert.
        assert feed.count() == 0
        assert feed.high_water() == 0

    def test_non_record_journals_nothing(self, feed):
        with pytest.raises(TypeError):
            feed.record_many([_rec("a", 1), {"not": "a record"}])
        assert feed.count() == 0

    def test_midbatch_failure_rolls_back_whole_batch(self, feed):
        feed.record(_rec("seed", 1))  # forces the connection open
        real = feed._connection
        feed._connection = _PoisonSecondInsert(real)
        try:
            with pytest.raises(sqlite3.OperationalError):
                feed.record_many([_rec("p", 1), _rec("q", 1)])
        finally:
            feed._connection = real
        # The first insert of the batch was rolled back with the failure.
        assert feed.count() == 1
        assert feed.high_water() == 1
        assert {r.record_id for r in feed.current_records()} == {"seed"}

    def test_empty_batch_returns_empty(self, feed):
        assert feed.record_many([]) == []
        assert feed.count() == 0


# Failure posture


class TestFailurePosture:
    def test_autocompact_failure_never_breaks_append(self, tmp_path, caplog):
        f = change_feed.ChangeFeed(root=tmp_path, compact_every=1)
        try:
            def _boom(conn):
                raise RuntimeError("compaction exploded")

            f._compact_locked = _boom  # type: ignore[method-assign]
            with caplog.at_level(
                logging.WARNING, logger="opti_oignon.veilid.change_feed"
            ):
                seq = f.record(_rec("a", 1))
            assert seq == 1
            assert f.count() == 1  # the append committed and survived
            assert any("auto-compaction failed" in m for m in caplog.messages)
            # The counter reset on fire whatever the outcome: restoring the
            # method, the very next append fires (and now succeeds).
            del f._compact_locked
            f.record(_rec("a", 2))
            assert f.count() == 1
            assert f.current_clock(RecordKind.CONVERSATION, "a") == 2
        finally:
            f.close()

    def test_on_demand_compact_propagates(self, feed):
        feed.record(_rec("a", 1))

        def _boom(conn):
            raise RuntimeError("compaction exploded")

        feed._compact_locked = _boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            feed.compact()


# Mode posture, forward-compatibility, hygiene


class TestModeAndHygiene:
    def test_compaction_works_in_bulbe(self, feed):
        set_mode("bulbe")
        for c in range(1, 4):
            feed.record(_rec("a", c))
        assert feed.compact() == 2  # local-disk maintenance, mode-free
        assert feed.count() == 1

    def test_module_references_no_mode_gate(self):
        src = (VEILID / "change_feed.py").read_text(encoding="utf-8")
        assert "assert_sync_allowed" not in src
        assert "get_current_mode" not in src

    def test_compaction_touches_only_the_feed_table(self, feed):
        _fill(feed)
        # A foreign table in the same database (the CHF-05 meta table to come)
        # must survive compaction untouched.
        side = sqlite3.connect(str(feed.db_path))
        side.execute("CREATE TABLE IF NOT EXISTS veilid_meta_fake (k TEXT)")
        side.execute("INSERT INTO veilid_meta_fake (k) VALUES ('epoch')")
        side.commit()
        side.close()
        feed.compact()
        side = sqlite3.connect(str(feed.db_path))
        rows = side.execute("SELECT k FROM veilid_meta_fake").fetchall()
        side.close()
        assert rows == [("epoch",)]

    def test_delete_constant_is_allowlisted_and_no_fstrings(self):
        assert change_feed.TABLE_NAME in change_feed._DELETE_SUPERSEDED
        src = (VEILID / "change_feed.py").read_text(encoding="utf-8")
        assert 'f"' not in src and "f'" not in src

    def test_vacuum_flag_smoke(self, feed):
        _fill(feed)
        deleted = feed.compact(vacuum=True)
        assert deleted == 4
        # The feed stays fully functional after the file rewrite.
        seq = feed.record(_rec("z", 1))
        assert seq == feed.high_water()
