#!/usr/bin/env python3
"""Contracts for the componion's life in time: a life cut or sliced anywhere is the same life.

``advance`` folds a being from its genesis and its facts, minute by minute,
and returns the state at the minute it reached. That state may depend only
on the genesis, the facts up to that minute and the minute itself: never on
where earlier calls stopped, nor on how much work each was allowed.

  * AK1 -- cuts. Two fixture lives of 90 days -- a garden with waters and a
    warm, and a windowsill with sparse facts, dormant most of its life --
    whose facts include a jump west and a jump east of the time zone across
    local midnights while the being sleeps, and a params ``evolve`` whose
    ``effective_from`` a later ``tz`` fact moves off the local midnight.
    Advanced whole, and again through 25 seeded cuts plus a few chosen
    ones: every stop equals the stepped path's state at the same minute (one
    run with ``probe.fast_path: false`` through the same stops), and the last stop
    equals the whole run. Presence: a stop inside a span of the fast path,
    one at a fast boundary, one at a daily firing, a pair of stops around a
    ``tz`` fact, and one around a change of season; the calls' traces show
    that production ran the fast path and the stepped run never did.
  * AK2 -- slices. The same lives, sliced by seeded budgets of 5000 to
    20000 units from the genesis, and, over a window of 20 dormant days and
    the wake after it, by budgets of 1 to 200 units: every slice's state
    equals the stepped path's at the same minute, the last equals the
    unsliced run, and the slices' work adds up to the unsliced work.
    Presence: at least ten slices stopped by their budget, at least three of
    them at a daily firing of the fast path, as the call's trace shows.

The reference engine only; the twin answers the same requests later, in the
equivalence suite. Local-only. The modules load through the shared
isolation window.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_life_support as support  # noqa: E402
from _allium_window import open_allium  # noqa: E402

BUDGET_S = {
    "test_ak1_a_life_cut_anywhere_is_the_stepped_life_and_the_whole_life": 2.0,
    "test_ak2_a_life_sliced_by_any_budget_is_the_same_life_for_the_same_work": 2.0,
}

DAY = 1440
END = 90 * DAY


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def engine():
    loaded, restore = open_allium(native=False)
    try:
        yield support.Engine(loaded)
    finally:
        restore()


def _local(being, t, offset):
    """The minute of the local day at minute of life ``t``."""
    return (being.b + t + offset) % DAY


def _at_local(being, day, minute, offset):
    """The minute of life of local minute ``minute`` on the ``day``-th local day after the birth day."""
    first = (being.b + being.tz_birth) // DAY
    return (first + day) * DAY + minute - offset - being.b


def _garden(engine):
    """A garden life: waters, a warm while it sleeps in winter, jumps of the zone, and a params evolve moved by one."""
    being = support.Being(engine, suite="time", index=0, weather="garden")
    for day in (2, 7, 13, 19, 45, 52, 58, 83):
        being.act(_at_local(being, day, 9 * 60 + 5 * day, 0), "water")
    being.act(_at_local(being, 31, 10 * 60 + 7, 0), "warm")
    being.act(_at_local(being, 33, 11 * 60, 0), "greet")
    e = _at_local(being, 40, 12 * 60 + 3, 0)
    being.evolve(e, being.params(evap_awake=6000, sun_max=60000), effective_from=being.midnight(e, 0))
    being.tz(_at_local(being, 40, 18 * 60 + 11, 0), 60)
    being.tz(_at_local(being, 66, 60 + 60, 60), -600)
    being.tz(_at_local(being, 70, 22 * 60 + 1, -600), 840)
    return being, (e, being.midnight(e, 0))


def _windowsill(engine):
    """A windowsill life, south and short, dormant from drought most of its days, with sparse facts."""
    being = support.Being(engine, suite="time", index=1, weather="windowsill", hemisphere="south", band="short",
                          tz=345)
    being.act(_at_local(being, 3, 8 * 60 + 13, 345), "water")
    being.tz(_at_local(being, 30, 2 * 60 + 2, 345), -300)
    being.tz(_at_local(being, 41, 21 * 60 + 30, -300), 840)
    e = _at_local(being, 50, 9 * 60 + 29, 840)
    being.evolve(e, being.params(evap_dormant=2048, rain_gain=0), effective_from=being.midnight(e, 840))
    being.tz(_at_local(being, 50, 16 * 60 + 44, 840), 780)
    being.act(_at_local(being, 70, 7 * 60 + 58, 780), "water")
    return being, (e, being.midnight(e, 840))


def _lives(engine):
    return (("garden",) + _garden(engine), ("windowsill",) + _windowsill(engine))


def _fast(being, t):
    return (being.b + t) % 15 == 0


def _is_fact_minute(being, t):
    return any(fact["t"] == t for fact in being.facts)


# ---------------------------------------------------------------------------
# AK1 -- cuts
# ---------------------------------------------------------------------------
def test_ak1_a_life_cut_anywhere_is_the_stepped_life_and_the_whole_life(engine):
    met = {"fast path span": 0, "fast boundary": 0, "firing": 0, "across tz": 0, "across season": 0,
           "evolve moved": 0, "slept": 0, "fast path fired": 0}
    for name, being, (evolve_at, midnight) in _lives(engine):
        whole = being.advance(END)
        assert whole["done"] and whole["at"] == END and whole["alarm"] == 0
        draw = engine.rng.Stream(bytes(32), "test.time", 10 + len(name))
        stops = {1 + draw.below(END - 1) for _ in range(25)}
        # Chosen cuts: a daily firing while the being sleeps and the minute before it, the evolve and its minute.
        firing = _at_local(being, 60, 0, 780) if name == "windowsill" else _at_local(being, 27, 0, 0)
        stops |= {firing - 1, firing, evolve_at, midnight - 1, midnight}
        stops = sorted(stops) + [END]
        cut = support.cuts(being, stops, probe={"trace": True})
        stepped = support.cuts(being, stops, probe={"fast_path": False, "trace": True})
        wrong = [(stop, a["hash"], b["hash"]) for stop, a, b in zip(stops, cut, stepped) if a["hash"] != b["hash"]]
        assert wrong == [], (name, wrong[:3])
        assert [a["state"] for a in cut] == [b["state"] for b in stepped], name
        assert cut[-1]["state"] == whole["state"] and cut[-1]["hash"] == whole["hash"], name
        assert all(a["alarm"] == 0 for a in cut + stepped), name
        # The fast path is what production took, read off its trace, and what the stepped run never took.
        met["fast path fired"] += sum(1 for a in cut if a["trace"]["fast_path_days"] > 0)
        assert sum(b["trace"]["fast_path_days"] for b in stepped) == 0, name

        by_stop = dict(zip(stops, cut))
        assert by_stop[firing]["state"]["day"] == by_stop[firing - 1]["state"]["day"] + 1, "a daily firing"
        assert by_stop[firing]["state"]["organs"]["stage"]["dormant"], "the firing is in the fast path's span"
        met["firing"] += 1
        tz_minutes = [fact["t"] for fact in being.facts if fact["kind"] == "tz"]
        previous = None
        for stop, answer in zip(stops, cut):
            state = answer["state"]
            asleep = state["organs"]["stage"]["dormant"]
            met["slept"] += asleep
            if asleep and not _fast(being, stop) and not _is_fact_minute(being, stop):
                met["fast path span"] += 1
            met["fast boundary"] += _fast(being, stop)
            if previous is not None:
                met["across tz"] += any(previous[0] < t <= stop for t in tz_minutes)
                met["across season"] += previous[1]["organs"]["stage"]["season"] != state["organs"]["stage"]["season"]
            previous = (stop, state)
        # The evolve moved off its midnight is applied at its minute, and not a minute before.
        before, after = by_stop[midnight - 1]["state"], by_stop[midnight]["state"]
        pending = by_stop[evolve_at]["state"]["pending"]
        assert pending is not None and pending["effective_from"] == midnight, name
        assert before["pending"] == pending, ("pending until the minute before", name)
        assert after["pending"] is None, ("applied at its minute", name)
        assert after["params"] == pending["params"] and before["params"] != after["params"], name
        offset = by_stop[midnight]["state"]["tz"]
        assert _local(being, midnight, offset) != 0, ("the evolve's minute is no longer a local midnight", name)
        met["evolve moved"] += 1
    assert min(met.values()) >= 1, met
    assert met["fast path span"] >= 5 and met["across tz"] >= 3 and met["across season"] >= 3, met


# ---------------------------------------------------------------------------
# AK2 -- slices
# ---------------------------------------------------------------------------
def test_ak2_a_life_sliced_by_any_budget_is_the_same_life_for_the_same_work(engine):
    met = {"stopped": 0, "stopped at a fast path firing": 0, "woke in the window": 0, "stopped by the fast path": 0}
    for name, being, _evolve in _lives(engine):
        whole = being.advance(END)
        draw = engine.rng.Stream(bytes(32), "test.time", 20 + len(name))
        budgets = iter([5000 + draw.below(15001) for _ in range(400)])
        sliced = support.slices(being, END, budgets)
        stops = [answer["at"] for answer in sliced]
        stepped = support.cuts(being, stops, probe={"fast_path": False})
        assert [a["hash"] for a in sliced] == [b["hash"] for b in stepped], name
        assert sliced[-1]["state"] == whole["state"] and sliced[-1]["hash"] == whole["hash"], name
        assert sum(answer["work"] for answer in sliced) == whole["work"], name
        assert all(answer["work"] >= 5000 for answer in sliced[:-1]), name
        met["stopped"] += sum(1 for answer in sliced if not answer["done"])

    # A window of 20 dormant days and the wake after it, sliced finely from a state inside it.
    being, _evolve = _windowsill(engine)
    start = _at_local(being, 48, 5 * 60 + 17, 840)
    wake = [fact["t"] for fact in being.facts if fact["kind"] == "act" and fact["t"] > start][0]
    end = wake + DAY
    assert wake - start >= 20 * DAY
    origin = being.advance(start)
    assert origin["state"]["organs"]["stage"]["dormant"], "the window opens on a sleeping being"
    whole = being.advance(end, origin["state"])
    draw = engine.rng.Stream(bytes(32), "test.time", 30)
    # Mostly a day of the fast path or two per call (4 units a day on a windowsill), now and then many more.
    budgets = iter([1 + draw.below(5) if draw.below(10) else 1 + draw.below(200) for _ in range(4000)])
    sliced = support.slices(being, end, budgets, state=origin["state"], probe={"trace": True})
    stops = [answer["at"] for answer in sliced]
    stepped = support.cuts(being, stops, probe={"fast_path": False, "trace": True}, state=origin["state"])
    assert [a["hash"] for a in sliced] == [b["hash"] for b in stepped]
    assert sliced[-1]["hash"] == whole["hash"] and sum(answer["work"] for answer in sliced) == whole["work"]
    for answer in sliced:
        state = answer["state"]
        if answer["done"]:
            continue
        met["stopped"] += 1
        local = _local(being, answer["at"], state["tz"])
        met["stopped at a fast path firing"] += local == 0 and state["organs"]["stage"]["dormant"]
        # Read off the trace: the call ran the fast path and its budget stopped it at one of its firings.
        met["stopped by the fast path"] += (answer["trace"]["fast_path_days"] > 0 and local == 0
                                            and state["organs"]["stage"]["dormant"])
    met["woke in the window"] += not whole["state"]["organs"]["stage"]["dormant"]
    assert met["stopped"] >= 10 and met["stopped at a fast path firing"] >= 3 and met["woke in the window"] == 1, met
    assert met["stopped by the fast path"] >= 3, met
    assert sum(b["trace"]["fast_path_days"] for b in stepped) == 0, "the stepped run never takes the fast path"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
