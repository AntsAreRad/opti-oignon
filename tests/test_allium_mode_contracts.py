#!/usr/bin/env python3
"""Contracts for the security mode as the garden reads it: once per action, and Bulbe's gates.

``Store.action()`` pins, for the length of one public action, the three
readings every store call of that action shares: the mode, whether the
platform runs for one person, and the wall clock. The next action reads
them again. A garden switched off or stopped builds no store and reads
nothing.

  * BM1 -- a look, a gesture, a deep verification and the laws screen each
    read the mode, the clock and the single-user seam exactly once, where
    the same store calls made outside an action read each at least twice; a mode moved from Daily to Bulbe between
    two looks is seen by the second, and a reader that raises serves the
    Bulbe layer as ``mode_unknown``; a resume and the look after it read it
    once; a garden switched off or stopped reads it never.
  * BM2 -- with no key and the glass jar allowed, Bulbe offers no jar: the
    sowing is refused as waiting for soil, with the Bulbe line, and nothing
    is written under the data directory; Daily discloses the jar on the card
    before it asks, and sows it labelled.

Local-only (the public distribution ships no tests). The platform and the
terminal load through the shared isolation window with the platform's
configuration, keys, mode, audit log and user modules proven unreachable;
every seam is injected (``tests/_allium_store_support.py``,
``tests/_allium_garden_support.py``); the glass jar is judged by the real
``store.probe``.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_garden_support as garden  # noqa: E402
import _allium_store_support as support  # noqa: E402

BUDGET_S = {
    "test_bm1_the_mode_is_read_once_per_action_and_again_for_the_next": 2.0,
    "test_bm2_bulbe_offers_no_glass_jar_and_daily_discloses_it_before_it_asks": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def p(monkeypatch, tmp_path):
    window, restore = garden.open_garden(monkeypatch, tmp_path)
    try:
        yield window
    finally:
        restore()


def _reads(mode, call):
    before = mode.reads
    result = call()
    return mode.reads - before, result


class _Counted:
    """A single-user seam that counts its reads."""

    def __init__(self, value):
        self.value = value
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.value


def _all_reads(given, call):
    """``(mode, clock, single-user)`` reads ``call`` made, and its result."""
    seams = (given["mode"], given["clock"], given["single_user"])
    before = [seam.reads for seam in seams]
    result = call()
    return tuple(seam.reads - start for seam, start in zip(seams, before)), result


# ---------------------------------------------------------------------------
# BM1 -- once per action
# ---------------------------------------------------------------------------
def test_bm1_the_mode_is_read_once_per_action_and_again_for_the_next(p, tmp_path):
    given = support.seams(p, tmp_path.joinpath("alive"), suite="bm1")
    given["single_user"] = _Counted(True)
    mode = given["mode"]
    target = support.store(p, given)
    try:
        support.sow(p, target)
    finally:
        target.close()
    given["clock"].advance_days(2)
    gardener = garden.garden(p, given, attended=True)()
    try:
        actions = {"look": gardener.look, "act": lambda: gardener.act("water"), "verify": gardener.verify,
                   "laws": gardener.laws}
        for name, call in actions.items():
            read, _result = _reads(mode, call)
            assert read == 1, (name, read)
            reads, _result = _all_reads(given, call)
            assert reads == (1, 1, 1), ("the mode, the clock and the single-user seam, once each", name, reads)

        # Witness: the same store calls outside an action read the seam each time.
        store = gardener.store()

        def outside():
            being = store.open("local")
            being.view()
            being.in_order()

        read, _result = _reads(mode, outside)
        assert read >= 2, read

        def twice():
            for _time in range(2):
                store.open("local").view()

        reads, _result = _all_reads(given, twice)
        assert min(reads) >= 2, ("witness: outside an action, each call reads each seam", reads)

        # The next action reads again: a mode moved between two looks is seen.
        first = gardener.look()
        mode.value = "bulbe"
        second = gardener.look()
        mode.value = RuntimeError("the mode cannot be read")
        third = gardener.look()
        mode.value = "daily"
    finally:
        gardener.close()
    assert ("bulbe" not in first.labels, first.habitat, first.mode) == (True, ("pot", "open"), "daily")
    assert ("bulbe" in second.labels, second.habitat, second.mode) == (True, ("pot", "bulbe"), "bulbe")
    assert ("mode_unknown" in third.labels, third.habitat, third.mode) == (True, ("pot", "bulbe"), "unknown")
    assert {first.status, second.status, third.status} == {"alive"}

    # A resume and the look after it: one read.
    broken = support.seams(p, tmp_path.joinpath("broken"), suite="bm1", index=1)
    target = support.store(p, broken)
    try:
        being = support.sow(p, target)
        for act in ("water", "greet"):
            broken["clock"].wall += 600
            being.append("act", {"act": act}, transport=support.cli(p))
    finally:
        target.close()
    path = support.store_path(p, broken)
    support.edit(path, lambda conn: conn.execute(
        "UPDATE bodies SET body = ? WHERE eid = (SELECT eid FROM links WHERE seq = 2)", (b'{"act":"none"}',)))
    gardener = garden.garden(p, broken, attended=True)()
    try:
        seen = gardener.look()
        assert (seen.status, seen.offer.kept_seq, seen.offer.discarded) == ("unreadable", 1, 1), seen
        read, after = _reads(broken["mode"], lambda: gardener.resume(1, 1))
    finally:
        gardener.close()
    assert (read, after.status) == (1, "alive"), (read, after)

    # Switched off, or stopped: no read at all.
    for name, factory in (("off", garden.garden(p, given, switch="off")),
                          ("stopped", garden.garden(p, given, stopped=lambda: True))):
        gardener = factory()
        try:
            read, look = _reads(mode, gardener.look)
        finally:
            gardener.close()
        assert (read, look.status, factory.stores) == (0, {"off": "disabled", "stopped": "stopped"}[name], 0), name


# ---------------------------------------------------------------------------
# BM2 -- Bulbe offers no glass jar
# ---------------------------------------------------------------------------
def _jar_seams(p, tmp_path, suite, mode):
    return support.seams(p, tmp_path, suite=suite, anchor_secret=lambda: ("none", None, "nokey"),
                         persistence={"busy_timeout_ms": 5000, "path": "allium", "require_encryption": False},
                         probe=p.store.probe, mode=support.Mode(mode))


def test_bm2_bulbe_offers_no_glass_jar_and_daily_discloses_it_before_it_asks(p, tmp_path):
    say = p.wording.say
    bulbe = _jar_seams(p, tmp_path.joinpath("bulbe"), "bm2", "bulbe")
    result = garden.invoke(p, ["sow"], input="Pip\nyes\n", factory=garden.garden(p, bulbe, attended=True))
    assert result.exit_code == 2, (result.exit_code, result.stdout, result.stderr, result.exception)
    for key in ("status.awaiting_soil", "status.awaiting_soil.bulbe"):
        assert garden.says(result.stderr, p, key), (key, result.stderr)
    data = Path(bulbe["data_dir"])
    assert not data.exists() or [path for path in data.rglob("*") if path.is_file()] == [], "nothing written"
    shown = garden.invoke(p, ["show"], factory=garden.garden(p, bulbe))
    assert shown.exit_code == 2 and garden.shows(shown.stdout, p, "status.awaiting_soil.bulbe"), shown.stdout

    daily = _jar_seams(p, tmp_path.joinpath("daily"), "bm2", "daily")
    result = garden.invoke(p, ["sow"], input="Pip\nyes\n", factory=garden.garden(p, daily, attended=True))
    assert result.exit_code == 0, (result.exit_code, result.stdout, result.stderr, result.exception)
    rows = result.stdout.splitlines()
    glass = garden.rows(p, say("sow.glass"))[0]
    confirm = garden.rows(p, say("sow.ask.confirm"))[0]
    label = garden.rows(p, say("label.glass_jar"))[0]
    assert glass in rows and label in rows, result.stdout
    assert rows.index(glass) < rows.index(confirm) < rows.index(label), result.stdout
    assert support.store_path(p, daily, suffix=".glass.db").exists()
    assert not support.store_path(p, daily).exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
