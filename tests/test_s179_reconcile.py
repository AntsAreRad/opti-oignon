#!/usr/bin/env python3
"""Tests for S179 Goal 2 -- convergent reconciliation (Theme 4 / Veilid Sync).

Covers opti_oignon/veilid/reconcile.py:

- Last-writer-wins by logical clock: a strictly higher clock supersedes, cleanly,
  with no conflict logged.
- The deterministic tie-break on a clock tie: content hash, then device id, with no
  dependence on input order.
- The conflict log / retained history: concurrent divergence (a tie on the winning
  clock with different content) retains the loser; a clean supersession does not;
  one representative per distinct losing content is retained.
- Convergence and structure: one winner per key, keys present on only one side
  carried through, tombstones winning so deletions converge.
- Order-independence (commutativity): reconcile(a, b) == reconcile(b, a) for both
  the converged set and the conflict log; within-a-side order does not matter.
- Idempotence: reconciling a converged set with itself returns it unchanged with an
  empty conflict log; reconcile_many over the same union agrees with reconcile.

Loaded via spec_from_file_location with opti_oignon stubbed; records is loaded
first so reconcile's import of it resolves to the in-memory module.
"""

import importlib.util
import sys
import types
from pathlib import Path

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
records = _load("records")  # reconcile imports this; load it first
reconcile = _load("reconcile")
RecordKind = records.RecordKind


def _rec(record_id, clock, *, device, payload=None, kind=None, deleted=False):
    return records.new_record(
        kind=kind or RecordKind.CONVERSATION,
        record_id=record_id,
        payload=payload if payload is not None else {"v": clock},
        device=device,
        clock=clock,
        deleted=deleted,
    )


# Last-writer-wins by clock


class TestClockWins:
    def test_higher_clock_supersedes(self):
        old = _rec("c1", 1, device="A", payload={"text": "old"})
        new = _rec("c1", 2, device="B", payload={"text": "new"})
        result = reconcile.reconcile([old], [new])
        assert len(result.records) == 1
        assert result.records[0].clock == 2
        assert result.records[0].payload == {"text": "new"}

    def test_clean_supersession_logs_no_conflict(self):
        old = _rec("c1", 1, device="A", payload={"text": "old"})
        new = _rec("c1", 5, device="B", payload={"text": "new"})
        result = reconcile.reconcile([old], [new])
        assert result.conflicts == []

    def test_unique_keys_all_present(self):
        left = [_rec("a", 1, device="A"), _rec("b", 1, device="A")]
        right = [_rec("c", 1, device="B")]
        result = reconcile.reconcile(left, right)
        assert {r.record_id for r in result.records} == {"a", "b", "c"}


# Tie-break on a clock tie


class TestTieBreak:
    def test_tiebreak_is_deterministic_by_content_then_device(self):
        # Same clock, different content -> the tie-break decides, both orders agree.
        x = _rec("c1", 3, device="A", payload={"text": "alpha"})
        y = _rec("c1", 3, device="B", payload={"text": "beta"})
        w1 = reconcile.reconcile([x], [y]).records[0]
        w2 = reconcile.reconcile([y], [x]).records[0]
        assert w1 == w2

    def test_winner_is_the_max_selection_key(self):
        x = _rec("c1", 3, device="A", payload={"text": "alpha"})
        y = _rec("c1", 3, device="B", payload={"text": "beta"})
        expected = max([x, y], key=reconcile._selection_key)
        assert reconcile.reconcile([x], [y]).records[0] == expected

    def test_same_content_two_devices_is_not_a_conflict(self):
        # Identical content at the same clock from two devices: pick one, no conflict.
        a = _rec("c1", 3, device="A", payload={"text": "same"})
        b = _rec("c1", 3, device="B", payload={"text": "same"})
        result = reconcile.reconcile([a], [b])
        assert result.conflicts == []
        assert len(result.records) == 1
        assert result.records[0].content_hash == a.content_hash


# Conflict log / retained history


class TestConflictLog:
    def test_concurrent_divergence_retains_loser(self):
        x = _rec("c1", 4, device="A", payload={"text": "alpha"})
        y = _rec("c1", 4, device="B", payload={"text": "beta"})
        result = reconcile.reconcile([x], [y])
        assert len(result.conflicts) == 1
        entry = result.conflicts[0]
        assert entry.key == ("conversation", "c1")
        assert entry.winner == result.records[0]
        retained_hashes = {r.content_hash for r in entry.retained}
        loser = x if result.records[0] == y else y
        assert retained_hashes == {loser.content_hash}

    def test_superseded_lower_clock_not_retained(self):
        # The loser is at a lower clock: clean update, nothing retained.
        a = _rec("c1", 4, device="A", payload={"text": "alpha"})
        b = _rec("c1", 4, device="B", payload={"text": "beta"})
        old = _rec("c1", 1, device="C", payload={"text": "ancient"})
        result = reconcile.reconcile_many([[a], [b], [old]])
        entry = result.conflicts[0]
        retained_hashes = {r.content_hash for r in entry.retained}
        assert old.content_hash not in retained_hashes
        # only the at-clock-4 distinct loser is retained
        assert len(entry.retained) == 1

    def test_one_representative_per_distinct_losing_content(self):
        # Two losers with the same content collapse to one retained representative.
        winner = _rec("c1", 5, device="Z", payload={"text": "winner"})
        l1 = _rec("c1", 5, device="A", payload={"text": "loser"})
        l2 = _rec("c1", 5, device="B", payload={"text": "loser"})
        result = reconcile.reconcile_many([[winner], [l1], [l2]])
        entry = result.conflicts[0]
        assert len(entry.retained) == 1
        assert entry.retained[0].content_hash == l1.content_hash

    def test_conflict_log_ordered_by_key(self):
        a1 = _rec("a", 2, device="A", payload={"t": 1})
        a2 = _rec("a", 2, device="B", payload={"t": 2})
        z1 = _rec("z", 2, device="A", payload={"t": 1})
        z2 = _rec("z", 2, device="B", payload={"t": 2})
        result = reconcile.reconcile_many([[z1, a1], [z2, a2]])
        keys = [e.key for e in result.conflicts]
        assert keys == sorted(keys)


# Tombstones converge


class TestTombstones:
    def test_tombstone_wins_when_newer(self):
        live = _rec("c1", 1, device="A", payload={"text": "here"})
        gone = _rec("c1", 2, device="B", payload={}, deleted=True)
        result = reconcile.reconcile([live], [gone])
        assert result.records[0].deleted is True

    def test_tombstone_kept_in_converged_set(self):
        gone = _rec("c1", 3, device="A", payload={}, deleted=True)
        result = reconcile.reconcile([gone], [])
        assert len(result.records) == 1
        assert result.records[0].deleted is True


# Order-independence (commutativity)


class TestCommutativity:
    def _scenario(self):
        return (
            [
                _rec("a", 2, device="A", payload={"t": "a2"}),
                _rec("b", 1, device="A", payload={"t": "b1"}),
                _rec("c", 3, device="A", payload={"t": "cA"}),
            ],
            [
                _rec("a", 1, device="B", payload={"t": "a1"}),
                _rec("c", 3, device="B", payload={"t": "cB"}),
                _rec("d", 1, device="B", payload={"t": "d1"}),
            ],
        )

    def test_swapping_sides_gives_same_converged_set(self):
        left, right = self._scenario()
        ab = reconcile.reconcile(left, right).records
        ba = reconcile.reconcile(right, left).records
        assert ab == ba

    def test_swapping_sides_gives_same_conflict_log(self):
        left, right = self._scenario()
        ab = reconcile.reconcile(left, right).conflicts
        ba = reconcile.reconcile(right, left).conflicts
        assert ab == ba

    def test_within_side_order_does_not_matter(self):
        left, right = self._scenario()
        a = reconcile.reconcile(left, right).records
        b = reconcile.reconcile(list(reversed(left)), list(reversed(right))).records
        assert a == b


# Idempotence


class TestIdempotence:
    def test_reconcile_converged_with_itself_is_stable(self):
        left = [_rec("a", 2, device="A"), _rec("b", 1, device="A")]
        right = [_rec("a", 1, device="B"), _rec("c", 1, device="B")]
        converged = reconcile.reconcile(left, right).records
        again = reconcile.reconcile(converged, converged)
        assert again.records == converged
        assert again.conflicts == []

    def test_reconcile_many_agrees_with_reconcile(self):
        left = [_rec("a", 2, device="A", payload={"t": 1})]
        right = [_rec("a", 2, device="B", payload={"t": 2})]
        pair = reconcile.reconcile(left, right)
        many = reconcile.reconcile_many([left, right])
        assert pair.records == many.records
        assert pair.conflicts == many.conflicts
