#!/usr/bin/env python3
"""Contracts for the componion's recorder: what a journaled write adds of its own, and when.

The recorder reads the wall clock and the local offset once per journaled
write. It never moves a being backwards and never before its birth; it notes
a clock that was set back, once per head minute; and it records the offset
only when it differs from the one the law timeline has in force. Its facts
go in the same transaction as the gesture, before it, in a fixed order.

  * AK5 -- late facts land on the head: with the clock set back after the
    day-5 facts, an append lands at the latest minute with exactly one
    ``clock`` fact whose ``behind`` is the gap; a second behind append adds
    no scar, a forward jump adds none, the skew below which nothing is noted
    is the life settings' own, and a wall before birth, or past the last
    minute a fact may carry, is refused ``clock``.
    A late fact on the head invalidates the checkpoint kept at that minute,
    in its own transaction. After a set-back with no new fact, a view shows
    the being at the recorder's minute, never before the latest fact, and so
    possibly younger than an earlier view; the reducer refuses a fact at or
    before a state's minute as ``chain "order"``.
  * AK7 -- the recorder's ``tz`` facts: one is written before the gesture
    (``tz``, ``clock``, gesture, in consecutive ``oseq`` and links), in the
    same transaction, only when the observed offset differs from the one in
    force -- a landed ``tz`` fact counts; a reading that raises, is not an
    integer, is not a whole quarter hour or is out of range writes nothing
    and keeps the offset, while a quarter hour that is not a whole hour and
    the widest offsets, +-14 h, are written; a dropped gesture writes no
    ``tz`` and drops no checkpoint; a view before the next write keeps the
    old offset, and the next view's local minute follows the new one. An
    offset another device landed at the write's own minute, from an origin
    that sorts after this one's, ends that minute: a reading that differs
    is not written there, since it could not be in force, and is written at
    the next minute.

Local-only. The platform loads through the shared isolation window with the
platform's configuration, keys, mode, audit log and user modules proven
unreachable; every seam is injected (``tests/_allium_store_support.py``),
the offset seam included.
"""

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_store_support as support  # noqa: E402

BUDGET_S = {
    "test_ak5_late_facts_land_on_the_head_with_one_scar_and_a_view_follows_the_clock_back": 2.0,
    "test_ak7_a_tz_fact_is_written_before_the_gesture_only_when_the_offset_in_force_changes": 2.0,
}
DAY = 1440
B = support.WALL // 60
# The last minute of life a fact may carry: two days short of the largest integer.
LAST = (1 << 53) - 1 - 2 * DAY


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def p():
    platform, restore = support.open_platform()
    try:
        yield platform
    finally:
        restore()


def _rows(path):
    """``(seq, t, kind, origin, oseq, laws, body)`` of every linked fact, in seq order."""
    rows = support.read(path, "SELECT l.seq, f.t, f.kind, f.origin, f.oseq, f.laws, b.body FROM links l "
                              "JOIN facts f ON f.eid = l.eid LEFT JOIN bodies b ON b.eid = l.eid ORDER BY l.seq")
    return [(seq, t, kind, origin, oseq, laws, None if body is None else json.loads(bytes(body)))
            for seq, t, kind, origin, oseq, laws, body in rows]


def _gen(p, path):
    [(value,)] = support.read(path, "SELECT value FROM meta WHERE key = 'gen'")
    return p.wire.parse(bytes(value))


def _new(path, before):
    """The rows written since ``before``, as ``(kind, t, body)``; the seqs and oseqs are checked consecutive."""
    rows = _rows(path)
    assert rows[:len(before)] == before, "nothing already written moved"
    added = rows[len(before):]
    for index, row in enumerate(added):
        previous = before[-1] if index == 0 else added[index - 1]
        assert row[0] == previous[0] + 1, ("consecutive links", previous, row)
        if row[3] == previous[3]:
            assert row[4] == previous[4] + 1, ("consecutive oseq", previous, row)
    return [(kind, t, body) for _seq, t, kind, _origin, _oseq, _laws, body in added]


def _refused(p, call):
    try:
        call()
    except (p.membrane.MembraneRefused, p.store.StoreRefused) as refusal:
        return refusal.code
    return None


def _local_minute(offset, t):
    return (B + t + offset) % DAY


# ---------------------------------------------------------------------------
# AK5 -- late facts land on the head; one scar per head minute; a view follows the clock back
# ---------------------------------------------------------------------------
def test_ak5_late_facts_land_on_the_head_with_one_scar_and_a_view_follows_the_clock_back(p, tmp_path):
    cli = support.cli(p)
    given = support.seams(p, tmp_path, suite="ak5")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = support.sow(p, target)
        path = support.store_path(p, given)
        clock.advance_days(5)
        day5 = [being.append("act", {"act": act}, transport=cli) for act in ("water", "greet")]
        head = 5 * DAY
        assert [outcome.t for outcome in day5] == [head, head]
        first = being.view()
        assert (first.status, first.at) == ("current", head), first

        # Two checkpoints kept: one at the head minute, one before it, each holding the state a view shows there
        # (a view starts from a kept state, so a made-up one would be refused as a divergence).
        before_head = being.checkpoint_put(head - 1, being.laws, _eid(path, 0), being.view(to=head - 1).state)
        at_head = being.checkpoint_put(head, being.laws, day5[-1].eid, first.state)
        assert (before_head.t, at_head.t) == (head - 1, head)

        # Never backwards: the clock set back three days, the append lands on the head, and one scar says by
        # how much.
        clock.advance_days(-3)
        rows, gen = _rows(path), _gen(p, path)
        late = being.append("act", {"act": "touch"}, transport=cli)
        assert isinstance(late, p.membrane.Appended) and late.t == head, late
        assert _new(path, rows) == [("clock", head, {"behind": 3 * DAY}), ("act", head, {"act": "touch"})]
        assert _gen(p, path) == gen + 1, "the scar, the gesture and the invalidation are one transaction"
        assert support.read(path, "SELECT t FROM checkpoints ORDER BY t") == [(head - 1,)], \
            "the checkpoint at the head minute is gone, the one before it stays"

        # A second behind append: the head already carries its scar.
        rows = _rows(path)
        again = being.append("act", {"act": "play"}, transport=cli)
        assert again.t == head and _new(path, rows) == [("act", head, {"act": "play"})]

        # A view after the set-back: never before the latest fact.
        behind = being.view()
        assert (behind.status, behind.at) == ("current", head), behind

        # Forward: a view at day 9; then, with no new fact, the clock set back to day 7 shows day 7.
        clock.advance_days(7)
        ahead = being.view()
        assert ahead.at == 9 * DAY, ahead
        clock.advance_days(-2)
        younger = being.view()
        assert younger.at == 7 * DAY, "the view follows the clock back, never below the latest fact"
        assert head < younger.at < ahead.at, "witness: younger than the earlier view"
        assert younger.hash != ahead.hash

        # A forward jump adds no scar.
        clock.advance_days(1)
        rows = _rows(path)
        jumped = being.append("act", {"act": "warm"}, transport=cli)
        assert jumped.t == 8 * DAY and _new(path, rows) == [("act", 8 * DAY, {"act": "warm"})]

        # Never before birth: such a wall is refused, for a write and for a view, and nothing is written.
        saved = clock.wall
        clock.wall = support.WALL - 60
        rows, gen = _rows(path), _gen(p, path)
        assert _refused(p, lambda: being.append("act", {"act": "warm"}, transport=cli)) == "clock"
        assert _refused(p, lambda: being.view()) == "clock"
        assert (_rows(path), _gen(p, path)) == (rows, gen)
        # Nor past the last minute a fact may carry (two days short of the largest integer, so that a next
        # midnight stays in range): refused ``clock`` too, for a write and for a view, and nothing is written.
        clock.wall = support.WALL + (LAST + 1) * 60
        assert _refused(p, lambda: being.append("act", {"act": "warm"}, transport=cli)) == "clock"
        assert _refused(p, lambda: being.view()) == "clock"
        assert (_rows(path), _gen(p, path)) == (rows, gen)
        # Witness: the last minute itself is a wall a view takes (capped, since the life there is long).
        clock.wall = support.WALL + LAST * 60
        edge = being.view(cap=1)
        assert (edge.status, edge.owed > 0) == ("catching_up", True), edge
        assert (_rows(path), _gen(p, path)) == (rows, gen)
        clock.wall = saved
    finally:
        target.close()

    # The skew below which nothing is noted is the life settings' own: 30 minutes here, not 5.
    life = {"checkpoints": {"daily": 14, "monthly": True, "weekly": 52}, "skew_note_min": 30}
    target = support.store(p, dict(given, life=life))
    try:
        being = target.open("local")
        head = 8 * DAY
        clock.wall = saved - 30 * 60
        rows = _rows(path)
        assert being.append("act", {"act": "greet"}, transport=cli).t == head
        assert _new(path, rows) == [("act", head, {"act": "greet"})], "30 minutes behind: within the skew"
        clock.wall = saved - 31 * 60
        rows = _rows(path)
        assert being.append("act", {"act": "greet"}, transport=cli).t == head
        assert _new(path, rows) == [("clock", head, {"behind": 31}), ("act", head, {"act": "greet"})]

        # The reducer refuses a fact at or before a state's minute.
        clock.wall = saved
        seen = being.view()
        ordered = being.in_order()
        genesis, last = ordered[0][2], ordered[-1][2]
        assert last["t"] == seen.at == head

        def ask(state, facts):
            request = {"budget": 1 << 40, "facts": facts, "genesis": genesis, "op": "advance", "state": state,
                       "to": head, "v": 1}
            return p.wire.parse(p.engine.call(p.wire.emit(request)))

        refused = ask(seen.state, [last])
        assert (refused.get("refused"), refused.get("detail")) == ("chain", "order"), refused
        whole = ask(None, [entry[2] for entry in ordered[1:]])
        assert "refused" not in whole and whole["hash"] == seen.hash, "witness: from genesis it folds"
    finally:
        target.close()


def _eid(path, seq):
    [(eid,)] = support.read(path, "SELECT eid FROM links WHERE seq = ?", (seq,))
    return eid


# ---------------------------------------------------------------------------
# AK7 -- a tz fact before the gesture, only when the offset in force changes
# ---------------------------------------------------------------------------
def test_ak7_a_tz_fact_is_written_before_the_gesture_only_when_the_offset_in_force_changes(p, tmp_path):
    cli = support.cli(p)
    reading = {"value": 0}
    failing = {"seal": False}
    seal, opener = support.TestCipher().pair()

    def tz(now):
        value = reading["value"]
        if isinstance(value, BaseException):
            raise value
        return value

    def sealing(key, plaintext):
        if failing["seal"]:
            raise RuntimeError("the seal failed")
        return seal(key, plaintext)

    given = support.seams(p, tmp_path, suite="ak7", tz=tz, cipher=(sealing, opener))
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = support.sow(p, target)
        path = support.store_path(p, given)
        clock.advance_days(1)

        # The offset in force is the birth's: nothing but the gesture.
        rows = _rows(path)
        being.append("act", {"act": "water"}, transport=cli)
        assert _new(path, rows) == [("act", DAY, {"act": "water"})]

        # Travel: the seam reads +2 h. Until the next write, a view keeps the old offset.
        reading["value"] = 120
        clock.wall += 60 * 60
        now_t = DAY + 60
        old = being.view()
        assert (old.at, old.state["tz"], old.env["minute"]) == (now_t, 0, _local_minute(0, now_t)), old

        # The next write records it, before the gesture, in the gesture's transaction.
        rows, gen = _rows(path), _gen(p, path)
        outcome = being.append("act", {"act": "greet"}, transport=cli)
        assert _new(path, rows) == [("tz", now_t, {"quarters": 8}), ("act", now_t, {"act": "greet"})]
        assert _gen(p, path) == gen + 1, "one transaction"
        assert [row[5] for row in _rows(path)[len(rows):]] == [being.laws, being.laws], "the law in force"
        assert outcome.oseq == _rows(path)[-2][4] + 1
        new = being.view()
        assert (new.at, new.state["tz"]) == (now_t, 120), new
        assert new.env["minute"] == _local_minute(120, now_t) == (old.env["minute"] + 120) % DAY

        # The same offset again: nothing but the gesture.
        rows = _rows(path)
        being.append("act", {"act": "play"}, transport=cli)
        assert _new(path, rows) == [("act", now_t, {"act": "play"})]

        # Readings that are not an offset write nothing and keep the one in force.
        for value in (RuntimeError("no zone"), "120", 120.0, None, True, 125, 7, 855, -855, 1 << 60):
            reading["value"] = value
            rows = _rows(path)
            being.append("act", {"act": "touch"}, transport=cli)
            assert _new(path, rows) == [("act", now_t, {"act": "touch"})], value
        assert being.view().state["tz"] == 120, "the offset is kept"

        # Witness: a westward reading writes one.
        reading["value"] = -60
        rows = _rows(path)
        being.append("act", {"act": "warm"}, transport=cli)
        assert _new(path, rows) == [("tz", now_t, {"quarters": -4}), ("act", now_t, {"act": "warm"})]

        # With the clock set back as well: tz, then clock, then the gesture.
        reading["value"] = 60
        clock.wall -= 2 * 60 * 60
        rows = _rows(path)
        being.append("act", {"act": "greet"}, transport=cli)
        assert _new(path, rows) == [("tz", now_t, {"quarters": 4}), ("clock", now_t, {"behind": 120}),
                                    ("act", now_t, {"act": "greet"})]

        # An offset another device landed half an hour earlier is the one in force: the same reading
        # writes nothing.
        clock.advance_days(1)
        append_t = 2 * DAY - 60
        support.land(being, "f" * 16, 0, append_t - 30, "tz", {"quarters": 12})
        reading["value"] = 180
        rows = _rows(path)
        being.append("act", {"act": "play"}, transport=cli)
        assert _new(path, rows) == [("act", append_t, {"act": "play"})]
        reading["value"] = 60
        rows = _rows(path)
        being.append("act", {"act": "play"}, transport=cli)
        assert _new(path, rows) == [("tz", append_t, {"quarters": 4}), ("act", append_t, {"act": "play"})], \
            "witness: against the landed offset, the old one is a change"

        # A dropped gesture writes no tz; the next one that lands does.
        clock.advance_days(1)
        for _ in range(3):
            assert isinstance(being.append("move_pot", {}, transport=cli), p.membrane.Appended)
        reading["value"] = 240
        rows = _rows(path)
        # A checkpoint kept at the minute the dropped gesture would land on: a write that writes no fact
        # drops no checkpoint either.
        minute = (clock.wall - support.WALL) // 60
        being.checkpoint_put(minute, being.laws, being.in_order()[-1][1], being.view().state)
        marks = support.read(path, "SELECT t, laws, through, state_hash FROM checkpoints ORDER BY t")
        assert minute in [mark[0] for mark in marks], "presence: a checkpoint at that minute"
        dropped = being.append("move_pot", {}, transport=cli)
        assert dropped == p.membrane.Dropped("budget") and _rows(path) == rows
        assert support.read(path, "SELECT t, laws, through, state_hash FROM checkpoints ORDER BY t") == marks
        landed = being.append("act", {"act": "water"}, transport=cli)
        assert _new(path, rows) == [("tz", landed.t, {"quarters": 16}), ("act", landed.t, {"act": "water"})]
        assert landed.t == minute and support.read(path, "SELECT t FROM checkpoints WHERE t >= ?", (minute,)) == [], \
            "witness: the gesture that lands drops it"

        # A gesture whose own write fails takes its tz fact with it.
        reading["value"] = -300
        failing["seal"] = True
        rows, gen = _rows(path), _gen(p, path)
        with pytest.raises(RuntimeError):
            being.append("lang_teach", {}, transport=cli, payload="word")
        assert (_rows(path), _gen(p, path)) == (rows, gen), "nothing of the transaction stays"
        failing["seal"] = False
        taught = being.append("lang_teach", {}, transport=cli, payload="word")
        assert _new(path, rows) == [("tz", taught.t, {"quarters": -20}),
                                    ("lang_teach", taught.t, {"payload": _rows(path)[-1][6]["payload"]})]

        # Whole quarter hours that are not whole hours are offsets, and so are the widest ones: each is
        # written in quarters, and the next view's local minute follows it.
        for offset, quarters in ((330, 22), (345, 23), (840, 56), (-840, -56)):
            reading["value"] = offset
            rows = _rows(path)
            outcome = being.append("act", {"act": "greet"}, transport=cli)
            assert _new(path, rows) == [("tz", outcome.t, {"quarters": quarters}),
                                        ("act", outcome.t, {"act": "greet"})], offset
            seen = being.view()
            assert (seen.state["tz"], seen.env["minute"]) == (offset, _local_minute(offset, seen.at)), offset

        # An offset another device landed at this very minute, from an origin that sorts after this one's,
        # ends the minute whatever this device writes there: a reading that differs is not written, since it
        # could not be in force. At the next minute the same reading is written, and it is in force.
        now_t = (clock.wall - support.WALL) // 60
        assert now_t == outcome.t
        support.land(being, "f" * 16, 1, now_t, "tz", {"quarters": 4})
        assert being.view().state["tz"] == 60
        rows = _rows(path)
        same = being.append("act", {"act": "play"}, transport=cli)
        assert (same.t, _new(path, rows)) == (now_t, [("act", now_t, {"act": "play"})])
        assert being.view().state["tz"] == 60, "the landed offset ends the minute"
        clock.wall += 60
        rows = _rows(path)
        being.append("act", {"act": "play"}, transport=cli)
        assert _new(path, rows) == [("tz", now_t + 1, {"quarters": -56}), ("act", now_t + 1, {"act": "play"})]
        assert being.view().state["tz"] == -840, "witness: a minute later the reading is written and in force"
    finally:
        target.close()
