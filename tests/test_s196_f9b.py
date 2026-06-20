#!/usr/bin/env python3
"""S196 F9b -- functional audit fixes for reconcile + change_feed (CRDT + watermark).

One tight test group per fix:

- CHF-01: ``since(w)`` with a watermark beyond the journal's high-water (the
  impossible-watermark signal: the journal file was reset and sequences
  restarted, or the asker's stored watermark is corrupt) serves the FULL current
  set instead of nothing, so the devices converge (idempotent apply) rather than
  silently diverging forever. The boundary (``since(high_water())`` -> empty)
  and the normal-delta path are unchanged.
- CHF-03: journal rows that fail integrity on read (unparseable payload or
  content-hash mismatch) are still skipped, but the aggregate count is surfaced
  as one warning per read instead of debug-only -- a corrupted journal no longer
  shrinks the served set silently.

reconcile.py needed no fix (RCN-01 is a recorded design note); the loader idiom
matches the s179/s196_f9a suites.
"""

from __future__ import annotations

import importlib.util
import logging
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
    spec = importlib.util.spec_from_file_location(full, str(VEILID / f"{name}.py"))
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
peers = _load("peers")
producers = _load("producers")
sync_engine = _load("sync_engine")
RecordKind = records.RecordKind

_CF_LOGGER = "opti_oignon.veilid.change_feed"


@pytest.fixture(autouse=True)
def _daily_and_reset():
    set_mode("daily")
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    yield
    change_feed.reset_change_feed()
    peers.reset_peer_store()
    sync_engine.reset_sync_engine()
    set_mode("daily")


def _rec(record_id, clock, *, device, payload=None, kind=None, deleted=False):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
        deleted=deleted,
    )


def _feed(tmp_path, name, seed=()):
    f = change_feed.ChangeFeed(root=tmp_path / name)
    for r in seed:
        f.record(r)
    return f


def _corrupt_row(feed, record_id, *, column, value):
    """Corrupt one journal row in place (test-only, through the feed's own conn)."""
    with feed._lock:
        conn = feed._conn()
        conn.execute(
            "UPDATE veilid_change_feed SET {} = ? WHERE record_id = ?".format(column),
            (value, record_id),
        )
        conn.commit()


# --- CHF-01: impossible watermark serves the full set ------------------------


class TestCHF01ResetBackstop:
    def test_beyond_high_water_serves_full_set(self, tmp_path):
        feed = _feed(
            tmp_path,
            "A",
            [_rec("a", 1, device="A"), _rec("b", 1, device="A"), _rec("a", 2, device="A")],
        )
        high = feed.high_water()
        delta = feed.since(high + 10)
        assert delta.high_water == high
        by_id = {r.record_id: r for r in delta.records}
        assert set(by_id) == {"a", "b"}
        assert by_id["a"].clock == 2  # latest per key, like since(0)

    def test_boundary_unchanged(self, tmp_path):
        # since(high_water()) still returns nothing: the boundary is not the
        # impossible case.
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        delta = feed.since(feed.high_water())
        assert delta.records == []
        assert delta.high_water == feed.high_water()

    def test_normal_delta_unchanged(self, tmp_path):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        w = feed.high_water()
        feed.record(_rec("b", 1, device="A"))
        delta = feed.since(w)
        assert {r.record_id for r in delta.records} == {"b"}

    def test_warning_emitted(self, tmp_path, caplog):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        with caplog.at_level(logging.WARNING, logger=_CF_LOGGER):
            feed.since(feed.high_water() + 1)
        assert any("CHF-01" in r.message for r in caplog.records)

    def test_round_converges_after_journal_reset(self, tmp_path):
        # End to end: A consumed B up to watermark 5, then B's journal was reset
        # (fresh feed, sequences restarted). Without the backstop A would receive
        # nothing forever; with it the records flow and A converges, while A's
        # watermark stays put (monotonic) until B's sequence passes it.
        feed_a = _feed(tmp_path, "A")
        store_a = peers.PeerStore(root=tmp_path / "pa")
        eng = sync_engine.SyncEngine(device="A", feed=feed_a, store=store_a)
        eng.register_peer("B", "rk-B")
        store_a.advance_watermark("B", 5)

        feed_b = _feed(tmp_path, "B", [_rec("b1", 1, device="B"), _rec("b2", 2, device="B")])
        assert feed_b.high_water() < 5  # the reset journal

        class HonestPeer:
            def fetch(self, request):
                return protocol.respond_to_request(feed_b, request, device="B")

        res = eng.run_round("B", HonestPeer())
        assert res.applied == 2
        assert res.advanced is False
        assert store_a.get_watermark("B") == 5  # held, never regressed
        assert {r.record_id for r in feed_a.current_records()} == {"b1", "b2"}


# --- CHF-03: corrupt journal rows are counted and surfaced -------------------


class TestCHF03CorruptRowVisibility:
    def test_unparseable_payload_warns_with_count(self, tmp_path, caplog):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A"), _rec("b", 1, device="A")])
        _corrupt_row(feed, "b", column="payload", value="not json")
        with caplog.at_level(logging.WARNING, logger=_CF_LOGGER):
            current = feed.current_records()
        assert {r.record_id for r in current} == {"a"}
        msgs = [r.message for r in caplog.records if "corrupt journal row" in r.message]
        assert msgs and "1" in msgs[0]

    def test_hash_mismatch_also_counted(self, tmp_path, caplog):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A"), _rec("b", 1, device="A")])
        _corrupt_row(feed, "b", column="content_hash", value="0" * 64)
        with caplog.at_level(logging.WARNING, logger=_CF_LOGGER):
            current = feed.current_records()
        assert {r.record_id for r in current} == {"a"}
        assert any("corrupt journal row" in r.message for r in caplog.records)

    def test_clean_read_no_warning(self, tmp_path, caplog):
        feed = _feed(tmp_path, "A", [_rec("a", 1, device="A")])
        with caplog.at_level(logging.WARNING, logger=_CF_LOGGER):
            current = feed.current_records()
        assert {r.record_id for r in current} == {"a"}
        assert not any("corrupt journal row" in r.message for r in caplog.records)
