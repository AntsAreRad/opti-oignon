#!/usr/bin/env python3
"""Tests for S204 -- sync cycle Bloc 1 lot 2 (CHF-05 feed epoch).

Covers the journal-reset repair added this lot:

- change_feed.feed_epoch: a random epoch minted once per journal file in a
  one-row meta table (the SYN-02 identity-row idiom); stable across reads and
  reconnects, distinct across feeds, re-minted only when the file is recreated;
  clear() keeps it (AUTOINCREMENT preserves the sequence counter, so a
  delete-all is not a reset in the CHF-01 sense); compaction never touches the
  meta table; mode-free.
- protocol: the batch envelope carries the responder feed's epoch (the normal
  chunk, the empty-delta batch, and the PRT-01 benign batch alike); a pre-epoch
  feed omits the field and a failing epoch reader degrades the same way;
  parse_record_batch reads the epoch defensively (missing/malformed -> None,
  never a reject); forward-compatibility with unknown fields holds both ways.
- peers: the nullable last_epoch column (NULL default; additive migration for
  pre-S204 registries; preserved on re-pair); get_last_epoch /
  set_last_epoch / reset_for_epoch, the reset being one atomic statement
  (watermark to 0 + the new epoch under one commit; a simulated failure leaves
  neither half applied).
- sync_engine.run_round: first contact stores the epoch with no reset; an
  unchanged epoch never resets; a changed epoch resets the watermark to 0 and
  converges a full resync over the normal bounded leg loop (never the
  backstop), repairing the low-but-possible divergence CHF-01 cannot see; a
  pre-epoch peer leaves the stored epoch untouched and still rides CHF-01; a
  prior round's deferred hold is re-offered from 0 after a reset; the
  epoch_reset diagnostic reaches RoundResult and the audit event; a store
  without the epoch accessors skips the handling; Bulbe refusal unchanged at
  the wire while feed_epoch stays mode-free.

Loaded via spec_from_file_location with opti_oignon stubbed; the security mode
is driven through a stubbed opti_oignon.security_mode and the audit log is a
recording stub, the S181/S202/S203 idiom.
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
_AUDIT = {"events": []}


def set_mode(value: str = "daily", *, raises: bool = False) -> None:
    if raises:
        def _gm() -> str:
            raise RuntimeError("mode undeterminable")
    else:
        def _gm() -> str:
            return value

    _MODE["fn"] = _gm
    sys.modules["opti_oignon.security_mode"].get_current_mode = _gm  # type: ignore[attr-defined]


def _record_audit(**kwargs):
    _AUDIT["events"].append(kwargs)


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
        sys.modules["opti_oignon.signed_audit_log"] = types.ModuleType(
            "opti_oignon.signed_audit_log"
        )
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]


def _load(name: str):
    full = f"opti_oignon.veilid.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_stubs()
guard = _load("guard")
records = _load("records")
reconcile = _load("reconcile")
change_feed = _load("change_feed")
peers = _load("peers")
protocol = _load("protocol")
producers = _load("producers")
sync_engine = _load("sync_engine")

RecordKind = records.RecordKind
VeilidDisabledInBulbe = guard.VeilidDisabledInBulbe


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    sys.modules["opti_oignon.signed_audit_log"].chain_log = _record_audit  # type: ignore[attr-defined]
    _AUDIT["events"].clear()
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    yield
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    set_mode("daily")


def _rec(record_id, clock, *, device="B", payload=None, kind=None):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
    )


def _feed(tmp_path, name="feed", seed=()):
    f = change_feed.ChangeFeed(root=tmp_path / name)
    for r in seed:
        f.record(r)
    return f


def _engine(tmp_path, device="A", seed=()):
    feed = change_feed.ChangeFeed(root=tmp_path / "local_feed")
    for r in seed:
        feed.record(r)
    store = peers.PeerStore(root=tmp_path / "store")
    return sync_engine.SyncEngine(device=device, feed=feed, store=store), feed, store


class HonestPeer:
    """Answers from its own feed through the real responder, bounded per leg."""

    def __init__(self, feed, device="B", max_count=None, max_bytes=None):
        self._feed = feed
        self._device = device
        self._max_count = max_count
        self._max_bytes = max_bytes

    def fetch(self, request):
        kwargs = {}
        if self._max_count is not None:
            kwargs["max_count"] = self._max_count
        if self._max_bytes is not None:
            kwargs["max_bytes"] = self._max_bytes
        return protocol.respond_to_request(
            self._feed, request, device=self._device, **kwargs
        )


class StrippingPeer(HonestPeer):
    """A pre-epoch peer: an old sender, simulated by stripping the field."""

    def fetch(self, request):
        batch = super().fetch(request)
        batch.pop("epoch", None)
        return batch


class TestFeedEpoch:
    def test_epoch_minted_once_and_stable(self, tmp_path):
        feed = _feed(tmp_path)
        first = feed.feed_epoch()
        assert isinstance(first, str) and len(first) == 32
        assert feed.feed_epoch() == first

    def test_epoch_stable_across_reconnect(self, tmp_path):
        feed = _feed(tmp_path)
        first = feed.feed_epoch()
        feed.close()
        reopened = change_feed.ChangeFeed(root=tmp_path / "feed")
        assert reopened.feed_epoch() == first

    def test_two_feeds_mint_distinct_epochs(self, tmp_path):
        a = _feed(tmp_path, "a")
        b = _feed(tmp_path, "b")
        assert a.feed_epoch() != b.feed_epoch()

    def test_recreated_feed_mints_new_epoch(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("k", 1)])
        first = feed.feed_epoch()
        path = feed.db_path
        feed.close()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(path) + suffix)
            if p.exists():
                p.unlink()
        recreated = change_feed.ChangeFeed(root=tmp_path / "feed")
        assert recreated.feed_epoch() != first

    def test_clear_keeps_epoch_and_sequences_do_not_restart(self, tmp_path):
        # AUTOINCREMENT preserves the sequence counter across a delete-all, so
        # clear() is not a journal reset in the CHF-01 sense and the epoch
        # deliberately survives it.
        feed = _feed(tmp_path, seed=[_rec("k", 1), _rec("k", 2)])
        epoch = feed.feed_epoch()
        high_before = feed.high_water()
        feed.clear()
        assert feed.feed_epoch() == epoch
        seq = feed.record(_rec("k", 3))
        assert seq > high_before

    def test_meta_table_allowlisted_and_outside_compaction_statement(self):
        assert change_feed.META_TABLE_NAME in change_feed._TABLES
        assert change_feed.TABLE_NAME in change_feed._DELETE_SUPERSEDED
        assert change_feed.META_TABLE_NAME not in change_feed._DELETE_SUPERSEDED
        # The meta statements name the meta table only.
        for stmt in (
            change_feed._CREATE_META,
            change_feed._SELECT_EPOCH,
            change_feed._INSERT_EPOCH,
        ):
            assert change_feed.META_TABLE_NAME in stmt
            assert change_feed.TABLE_NAME + " " not in stmt

    def test_feed_epoch_mode_free_in_bulbe(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("k", 1)])
        set_mode("bulbe")
        epoch = feed.feed_epoch()
        assert isinstance(epoch, str) and epoch
        # The wire stays refused while the local read works (the gate split).
        with pytest.raises(VeilidDisabledInBulbe):
            protocol.build_record_batch(feed, device="A", watermark=0)


class TestCompactionMetaIsolation:
    def test_compaction_never_touches_the_meta_table(self, tmp_path):
        # The lot-4 foreign-table proof, extended to the real meta table.
        feed = _feed(
            tmp_path,
            seed=[_rec("k", 1), _rec("k", 2), _rec("j", 1), _rec("j", 2)],
        )
        epoch = feed.feed_epoch()
        deleted = feed.compact()
        assert deleted == 2
        assert feed.feed_epoch() == epoch
        side = sqlite3.connect(str(feed.db_path))
        rows = side.execute(
            "SELECT COUNT(*) FROM {t}".format(t=change_feed.META_TABLE_NAME)
        ).fetchone()
        side.close()
        assert rows == (1,)

    def test_vacuum_compaction_keeps_the_epoch(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("k", 1), _rec("k", 2)])
        epoch = feed.feed_epoch()
        assert feed.compact(vacuum=True) == 1
        assert feed.feed_epoch() == epoch


class TestEnvelopeEpoch:
    def test_batch_carries_the_feed_epoch(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        batch = protocol.build_record_batch(feed, device="B", watermark=0)
        assert batch["epoch"] == feed.feed_epoch()

    def test_empty_delta_batch_carries_the_epoch(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        batch = protocol.build_record_batch(
            feed, device="B", watermark=feed.high_water()
        )
        assert batch["records"] == []
        assert batch["epoch"] == feed.feed_epoch()

    def test_benign_batch_carries_the_epoch_and_zero_high_water(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        batch = protocol.respond_to_request(feed, "garbage", device="B")
        assert batch["high_water"] == 0
        assert batch["records"] == []
        assert batch["epoch"] == feed.feed_epoch()

    def test_pre_epoch_feed_omits_the_field(self):
        class PreEpochFeed:
            def since_page(self, watermark, *, max_count, max_bytes):
                return change_feed.Delta(records=[], high_water=0)

        batch = protocol.build_record_batch(PreEpochFeed(), device="B", watermark=0)
        assert "epoch" not in batch

    def test_failing_epoch_reader_degrades_to_pre_epoch(self):
        class BrokenEpochFeed:
            def since_page(self, watermark, *, max_count, max_bytes):
                return change_feed.Delta(records=[], high_water=0)

            def feed_epoch(self):
                raise RuntimeError("meta table unreadable")

        batch = protocol.build_record_batch(BrokenEpochFeed(), device="B", watermark=0)
        assert "epoch" not in batch

    def test_parse_reads_the_epoch(self, tmp_path):
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        batch = protocol.build_record_batch(feed, device="B", watermark=0)
        parsed = protocol.parse_record_batch(batch)
        assert parsed is not None
        assert parsed.epoch == feed.feed_epoch()

    @pytest.mark.parametrize("bad", [None, "", 7, ["e"], {"e": 1}, True])
    def test_parse_treats_missing_or_malformed_epoch_as_none(self, bad):
        obj = {
            "v": protocol.PROTOCOL_VERSION,
            "type": protocol.MSG_RECORD_BATCH,
            "device": "B",
            "high_water": 3,
            "records": [],
        }
        if bad is not None:
            obj["epoch"] = bad
        parsed = protocol.parse_record_batch(obj)
        assert parsed is not None  # a pre-epoch peer, never a reject
        assert parsed.epoch is None

    def test_old_reader_forward_compat_both_ways(self, tmp_path):
        # A batch without the field parses (old sender); a batch with the field
        # plus an unknown extra field parses too (a future sender).
        feed = _feed(tmp_path, seed=[_rec("a", 1)])
        batch = protocol.build_record_batch(feed, device="B", watermark=0)
        without = dict(batch)
        without.pop("epoch")
        assert protocol.parse_record_batch(without) is not None
        with_future = dict(batch)
        with_future["future_field"] = {"x": 1}
        parsed = protocol.parse_record_batch(with_future)
        assert parsed is not None
        assert parsed.epoch == batch["epoch"]


class TestPeerStoreEpoch:
    def test_column_defaults_null_and_roundtrips(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "rk")
        assert store.get_last_epoch("B") is None
        assert store.get_peer("B").last_epoch is None
        assert store.set_last_epoch("B", "e1") is True
        assert store.get_last_epoch("B") == "e1"
        assert store.get_peer("B").last_epoch == "e1"

    def test_migration_adds_the_column_to_a_pre_s204_registry(self, tmp_path):
        # A registry created with the pre-S204 schema (no last_epoch) gains the
        # column on open and keeps its rows.
        root = tmp_path / "store"
        root.mkdir(parents=True)
        db = root / peers.DB_FILENAME
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE veilid_peers ("
            "peer_id TEXT PRIMARY KEY, routing_key TEXT NOT NULL, "
            "label TEXT NOT NULL DEFAULT '', watermark INTEGER NOT NULL DEFAULT 0, "
            "added_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO veilid_peers VALUES ('B', 'rk', '', 7, 't0', 't0')"
        )
        conn.commit()
        conn.close()
        store = peers.PeerStore(root=root)
        rec = store.get_peer("B")
        assert rec.watermark == 7
        assert rec.last_epoch is None
        assert store.set_last_epoch("B", "e1") is True
        assert store.get_last_epoch("B") == "e1"

    def test_repair_upsert_preserves_the_epoch(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "rk1")
        store.advance_watermark("B", 5)
        store.set_last_epoch("B", "e1")
        rec = store.add_peer("B", "rk2", label="renamed")
        assert rec.routing_key == "rk2"
        assert rec.watermark == 5
        assert rec.last_epoch == "e1"

    def test_unknown_peer_is_a_noop_returning_false(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        assert store.set_last_epoch("ghost", "e1") is False
        assert store.reset_for_epoch("ghost", "e1") is False
        assert store.get_last_epoch("ghost") is None
        assert store.get_last_epoch("") is None

    @pytest.mark.parametrize("bad", ["", 7, None, b"e", True])
    def test_epoch_validation_rejects_non_strings(self, tmp_path, bad):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "rk")
        with pytest.raises(ValueError):
            store.set_last_epoch("B", bad)
        with pytest.raises(ValueError):
            store.reset_for_epoch("B", bad)

    def test_reset_for_epoch_resets_and_stores_atomically(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "rk")
        store.advance_watermark("B", 9)
        store.set_last_epoch("B", "e1")
        assert store.reset_for_epoch("B", "e2") is True
        assert store.get_watermark("B") == 0
        assert store.get_last_epoch("B") == "e2"
        # The pair lands in ONE statement: structural proof of atomicity.
        assert "watermark = 0" in peers._RESET_FOR_EPOCH
        assert "last_epoch = ?" in peers._RESET_FOR_EPOCH

    def test_reset_simulated_failure_leaves_neither_half_applied(self, tmp_path):
        store = peers.PeerStore(root=tmp_path / "store")
        store.add_peer("B", "rk")
        store.advance_watermark("B", 9)
        store.set_last_epoch("B", "e1")
        real = store._conn()

        class FailingConn:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *args):
                if sql == peers._RESET_FOR_EPOCH:
                    raise sqlite3.OperationalError("simulated failure")
                return self._inner.execute(sql, *args)

            def commit(self):
                return self._inner.commit()

            def close(self):
                return self._inner.close()

        store._connection = FailingConn(real)
        with pytest.raises(sqlite3.OperationalError):
            store.reset_for_epoch("B", "e2")
        store._connection = real
        assert store.get_watermark("B") == 9
        assert store.get_last_epoch("B") == "e1"


class TestRunRoundEpoch:
    def test_first_contact_stores_the_epoch_without_reset(self, tmp_path):
        eng, local, store = _engine(tmp_path)
        feed_b = _feed(tmp_path, "feed_b", seed=[_rec(f"k{i}", 1) for i in range(3)])
        store.add_peer("B", "rk")
        store.advance_watermark("B", 2)
        res = eng.run_round("B", HonestPeer(feed_b))
        # No reset: only the delta after the stored watermark was served.
        assert res.epoch_reset is False
        assert res.applied == 1
        assert res.new_watermark == 3
        assert store.get_last_epoch("B") == feed_b.feed_epoch()

    def test_unchanged_epoch_never_resets(self, tmp_path):
        eng, local, store = _engine(tmp_path)
        feed_b = _feed(tmp_path, "feed_b", seed=[_rec("k1", 1)])
        store.add_peer("B", "rk")
        first = eng.run_round("B", HonestPeer(feed_b))
        assert first.epoch_reset is False
        feed_b.record(_rec("k2", 1))
        second = eng.run_round("B", HonestPeer(feed_b))
        assert second.epoch_reset is False
        assert second.applied == 1
        assert second.advanced is True
        assert store.get_last_epoch("B") == feed_b.feed_epoch()

    def test_changed_epoch_resets_and_converges_over_the_leg_loop(
        self, tmp_path, caplog
    ):
        # The divergence CHF-01 cannot see: the stale watermark (5) is
        # low-but-possible against the recreated journal (high 7), so without
        # the epoch the asker would silently skip the new journal's rows 1..5.
        eng, local, store = _engine(tmp_path)
        feed_b = _feed(
            tmp_path, "feed_b", seed=[_rec(f"old{i}", 1) for i in range(5)]
        )
        store.add_peer("B", "rk")
        first = eng.run_round("B", HonestPeer(feed_b))
        assert first.new_watermark == 5
        old_epoch = store.get_last_epoch("B")
        assert old_epoch == feed_b.feed_epoch()

        path = feed_b.db_path
        feed_b.close()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(path) + suffix)
            if p.exists():
                p.unlink()
        feed_b = _feed(
            tmp_path, "feed_b", seed=[_rec(f"new{i}", 1) for i in range(7)]
        )
        assert feed_b.feed_epoch() != old_epoch
        assert feed_b.high_water() == 7  # the stale watermark 5 is "possible"

        with caplog.at_level(logging.WARNING):
            res = eng.run_round("B", HonestPeer(feed_b, max_count=2))
        assert res.epoch_reset is True
        assert res.previous_watermark == 5
        assert res.new_watermark == 7
        assert res.advanced is True
        assert res.applied == 7
        # Leg 1 discarded at the stale cursor, then bounded pages from 0: the
        # resync rides the normal leg loop, never the CHF-01 backstop.
        assert res.legs > 2
        local_ids = {r.record_id for r in local.current_records()}
        assert {f"new{i}" for i in range(7)} <= local_ids
        assert store.get_last_epoch("B") == feed_b.feed_epoch()
        assert store.get_watermark("B") == 7
        text = caplog.text
        assert "epoch changed" in text
        assert "exceeds journal high-water" not in text  # no backstop ride

    def test_pre_epoch_peer_keeps_chf01_and_stored_epoch_untouched(self, tmp_path):
        # An old sender (no epoch on the wire) with an impossible watermark:
        # the CHF-01 backstop converges the data, the watermark is held by the
        # monotonic advance, and nothing is stored or reset.
        eng, local, store = _engine(tmp_path)
        feed_b = _feed(tmp_path, "feed_b", seed=[_rec(f"k{i}", 1) for i in range(3)])
        store.add_peer("B", "rk")
        store.advance_watermark("B", 99)
        res = eng.run_round("B", StrippingPeer(feed_b))
        assert res.epoch_reset is False
        assert res.applied == 3
        assert res.new_watermark == 99
        assert res.advanced is False
        assert store.get_last_epoch("B") is None

    def test_deferred_hold_is_reoffered_from_zero_after_a_reset(self, tmp_path):
        # Round 1 defers the skill (fail-secure hold). The peer's journal is
        # then recreated; the approving round resets and re-offers everything
        # from 0, skill included.
        eng, local, store = _engine(tmp_path)
        feed_b = _feed(
            tmp_path,
            "feed_b",
            seed=[_rec("c1", 1), _rec("s1", 1, kind=RecordKind.SKILL)],
        )
        store.add_peer("B", "rk")
        deny = lambda conv, label, args: False
        first = eng.run_round("B", HonestPeer(feed_b), approval_fn=deny)
        assert first.deferred == 1
        assert first.new_watermark == 0  # the single-chunk hold
        local_ids = {r.record_id for r in local.current_records()}
        assert "s1" not in local_ids and "c1" in local_ids

        path = feed_b.db_path
        feed_b.close()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(path) + suffix)
            if p.exists():
                p.unlink()
        feed_b = _feed(
            tmp_path,
            "feed_b",
            seed=[_rec("c1", 2), _rec("s1", 2, kind=RecordKind.SKILL)],
        )
        approve = lambda conv, label, args: True
        res = eng.run_round("B", HonestPeer(feed_b), approval_fn=approve)
        assert res.epoch_reset is True
        assert res.deferred == 0
        local_ids = {r.record_id for r in local.current_records()}
        assert {"c1", "s1"} <= local_ids
        assert store.get_watermark("B") == feed_b.high_water()

    def test_epoch_reset_reaches_the_audit_event(self, tmp_path):
        eng, local, store = _engine(tmp_path)
        feed_b = _feed(tmp_path, "feed_b", seed=[_rec("k1", 1)])
        store.add_peer("B", "rk")
        eng.run_round("B", HonestPeer(feed_b))
        path = feed_b.db_path
        feed_b.close()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(path) + suffix)
            if p.exists():
                p.unlink()
        feed_b = _feed(tmp_path, "feed_b", seed=[_rec("k2", 1)])
        eng.run_round("B", HonestPeer(feed_b))
        rounds = [
            e for e in _AUDIT["events"] if e.get("action") == "sync_round"
        ]
        assert rounds[-2]["epoch_reset"] is False
        assert rounds[-1]["epoch_reset"] is True

    def test_round_result_defaults_keep_the_old_shape(self):
        rr = sync_engine.RoundResult(peer_id="B")
        assert rr.epoch_reset is False
        assert rr.legs == 1

    def test_store_without_epoch_accessors_skips_the_handling(self, tmp_path):
        # A store that predates the epoch accessors (the since_page
        # forward-compat precedent): the round runs, nothing epoch-related is
        # attempted.
        feed_b = _feed(tmp_path, "feed_b", seed=[_rec("k1", 1)])
        local = change_feed.ChangeFeed(root=tmp_path / "local_feed")

        class LegacyStore:
            def __init__(self):
                self.watermark = 0

            def has_peer(self, peer_id):
                return True

            def get_watermark(self, peer_id):
                return self.watermark

            def advance_watermark(self, peer_id, watermark):
                self.watermark = max(self.watermark, int(watermark))
                return self.watermark

        eng = sync_engine.SyncEngine(device="A", feed=local, store=LegacyStore())
        res = eng.run_round("B", HonestPeer(feed_b))
        assert res.epoch_reset is False
        assert res.applied == 1
        assert res.new_watermark == 1

    def test_bulbe_refusal_unchanged_at_the_wire(self, tmp_path):
        eng, local, store = _engine(tmp_path)
        feed_b = _feed(tmp_path, "feed_b", seed=[_rec("k1", 1)])
        store.add_peer("B", "rk")
        set_mode("bulbe")
        with pytest.raises(VeilidDisabledInBulbe):
            eng.run_round("B", HonestPeer(feed_b))
        # The local epoch read stays mode-free while the wire refuses.
        assert isinstance(feed_b.feed_epoch(), str)
