#!/usr/bin/env python3
"""The life tier: ten-year lives of the componion on the native core, with the reference as their witness.

The t1 contracts live ninety days at most, in the Python reference, inside
two seconds. A being lives for years: these contracts live ten of them on
the full law, natively, and let the reference replay sampled windows of
them. They run only when asked by path (``bash scripts/ladder.sh life``):
the t1 sweep ignores this directory.

  * AK6 -- a decade, cut and sliced. Two lives of 3650 days on the full law,
    a garden and a windowsill, with about one fact a day (waters, warms,
    greetings, moves of the time zone, params ``evolve`` facts, and in the
    windowsill droughts it sleeps through). The native core lives each
    whole in one call, and again through 20 seeded cuts, each reached under
    seeded budgets: every cut is reached, the last state is the whole
    life's, and the work of the calls adds up to the whole life's work. Five
    seeded seven-day windows of each life, started from the native core's
    state at their first minute, are answered with the same bytes by the
    reference.
  * AQ9 -- the fast path over a decade. A windowsill that sleeps from a
    drought for ten years, and a garden that sleeps every winter, both with
    moves of the time zone across midnights while they sleep, one of them
    in their last thirty days: natively, the production path reaches the
    stepped path's state at the tenth year, for a fraction of its work, and
    the reference, given the native state thirty days before the end,
    answers those thirty days with the native core's bytes on the fast path
    and reaches the same state on the stepped path.

Both need the native core built and answering for the reference's world:
without it each contract is skipped as owed, and the ladder's life tier
says so and exits 3. Local-only. The modules load through the shared
isolation window.
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _allium_life_support as support  # noqa: E402
from _allium_window import open_allium  # noqa: E402

BUDGET_S = {
    "test_ak6_a_decade_is_the_same_life_whole_cut_sliced_or_replayed_by_the_reference": 60.0,
    "test_aq9_over_a_decade_the_fast_path_reaches_the_stepped_paths_state": 60.0,
}

DAY = 1440
YEARS = 3650
END = YEARS * DAY
OFFSETS = (-600, -345, -240, 0, 60, 345, 540, 780)
# Born 2026-01-20 (UTC): ten years later the being sleeps its winter through its last thirty days.
WINTER_WALL = 1768867200


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


class _Native:
    """The native core of one window, asked through its byte protocol."""

    def __init__(self, module, engine):
        self.module = module
        self.engine = engine
        self.calls = 0

    def bytes(self, request):
        self.calls += 1
        return bytes(self.module.allium_call(self.engine.wire.emit(request)))

    def ask(self, request):
        answer = self.engine.wire.parse(self.bytes(request))
        assert "refused" not in answer, answer
        return answer


def _open():
    """The window, the reference and the native core; skipped as owed when the core is absent or stale."""
    loaded, restore = open_allium(native=True)
    try:
        module = loaded["opti_oignon.native"].load()
        if module is None or getattr(module, "allium_call", None) is None:
            pytest.skip("OWED: the native core is not built here (scripts/build_oo_core.sh)")
        if not loaded["opti_oignon.allium.engine"].handshake(module):
            pytest.skip("OWED: the native core answers for another world, a stale build (scripts/build_oo_core.sh)")
    except BaseException:
        restore()
        raise
    engine = support.Engine(loaded)
    return engine, _Native(module, engine), restore


def _at_local(being, day, minute, offset):
    """The minute of life of local minute ``minute`` on the ``day``-th local day after the birth day."""
    first = (being.b + being.tz_birth) // DAY
    return (first + day) * DAY + minute - offset - being.b


# ---------------------------------------------------------------------------
# AK6 -- a decade, cut and sliced
# ---------------------------------------------------------------------------
def _decade(engine, weather, index):
    """A ten-year life on the full law with about one fact a day; the kinds of fact written, by name."""
    being = support.Being(engine, suite="life", index=index, law="v0_1", weather=weather, tz=60,
                          hemisphere=("north", "south")[index % 2], band=("medium", "short")[index % 2])
    draw = engine.rng.Stream(bytes(32), "test.life", index)
    offset = 60
    last = 0
    written = {}
    for day in range(1, YEARS):
        roll = draw.below(100)
        minute = 6 * 60 + draw.below(16 * 60)
        evolve = day in (1000, 2600)
        # The windowsill goes without water for a month now and then, and sleeps through a drought.
        if not evolve and (roll < 8 or (weather == "windowsill" and day % 173 < 30)):
            continue
        t = _at_local(being, day, minute, offset)
        if t <= last:
            continue
        if evolve:
            being.evolve(t, being.params(evap_awake=2048 + draw.below(8193), rain_gain=draw.below(131073)),
                         effective_from=being.midnight(t, offset))
            kind = "evolve"
        elif roll < 58:
            being.act(t, "water")
            kind = "water"
        elif roll < 68:
            being.act(t, "warm")
            kind = "warm"
        elif roll < 88:
            being.act(t, "greet")
            kind = "greet"
        else:
            offset = OFFSETS[draw.below(len(OFFSETS))]
            being.tz(t, offset)
            kind = "tz"
        written[kind] = written.get(kind, 0) + 1
        last = t
    return being, written


def test_ak6_a_decade_is_the_same_life_whole_cut_sliced_or_replayed_by_the_reference():
    engine, native, restore = _open()
    met = {"done false": 0, "dormant window": 0, "awake window": 0, "window": 0}
    try:
        for index, weather in ((0, "garden"), (1, "windowsill")):
            being, written = _decade(engine, weather, index)
            for kind in ("water", "warm", "greet", "tz", "evolve"):
                assert written.get(kind, 0) >= 1, (weather, kind, written)
            assert len(being.facts) >= 2500, (weather, len(being.facts))
            whole = native.ask(being.request(None, END, probe={"trace": True}))
            assert whole["done"] and whole["at"] == END and whole["alarm"] == 0, weather
            trace = whole["trace"]
            assert trace["fast_path_days"] >= 1 and trace["calls"]["chem"]["jump"] >= 1, (weather, "slept and woke")
            assert trace["facts"] == len(being.facts), weather

            draw = engine.rng.Stream(bytes(32), "test.life", 10 + index)
            stops = sorted({1 + draw.below(END - 1) for _ in range(20)}) + [END]
            state, work = None, 0
            for stop in stops:
                while True:
                    answer = native.ask(being.request(state, stop, budget=100000 + draw.below(2900001)))
                    state, work = answer["state"], work + answer["work"]
                    met["done false"] += not answer["done"]
                    if answer["done"]:
                        break
                assert answer["at"] == stop, (weather, stop, answer["at"])
            assert state == whole["state"] and answer["hash"] == whole["hash"], weather
            assert work == whole["work"], (weather, work, whole["work"])

            # Seven-day windows, from the native core's state at their first minute, in both engines.
            for _window in range(5):
                start = draw.below(END - 7 * DAY)
                origin = native.ask(being.request(None, start))["state"]
                request = being.request(origin, start + 7 * DAY)
                mine = engine.protocol.call(engine.wire.emit(request))
                assert native.bytes(request) == mine, (weather, start)
                answer = engine.wire.parse(mine)
                assert answer["done"] and "refused" not in answer, answer
                asleep = origin["organs"]["stage"]["dormant"] or answer["state"]["organs"]["stage"]["dormant"]
                met["dormant window" if asleep else "awake window"] += 1
                met["window"] += 1
    finally:
        restore()
    assert met["window"] == 10, met
    assert met["done false"] >= 5 and met["dormant window"] >= 1 and met["awake window"] >= 1, met
    assert native.calls >= 60, native.calls


# ---------------------------------------------------------------------------
# AQ9 -- the fast path over a decade
# ---------------------------------------------------------------------------
def _sleeper(engine, weather, index):
    """A decade on the full law that sleeps most of its time, with west and east moves of the zone asleep."""
    being = support.Being(engine, suite="life", index=index, law="v0_1", weather=weather, wall=WINTER_WALL,
                          tz=60, band="medium")
    offset = 60
    if weather == "garden":
        # Waters from spring to autumn, never in winter: the being sleeps from 1 December to its spring wake.
        for year in range(10):
            for week in range(12, 44, 2):
                being.act(_at_local(being, year * 365 + week * 7, 8 * 60 + week, offset), "water")
    # Moves of the zone across a local midnight while it sleeps: west in the small hours, so a day index
    # comes round again (and is not lived again), and east in the evening, so one is skipped.
    for day, minute, new in ((3 * 365 + 330, 2 * 60 + 7, -600), (3 * 365 + 350, 22 * 60 + 13, 60),
                             (YEARS - 20, 3 * 60 + 41, -345), (YEARS - 9, 21 * 60 + 29, 540)):
        being.tz(_at_local(being, day, minute, offset), new)
        offset = new
    return being


def test_aq9_over_a_decade_the_fast_path_reaches_the_stepped_paths_state():
    engine, native, restore = _open()
    met = {}
    try:
        for index, weather in ((20, "windowsill"), (21, "garden")):
            being = _sleeper(engine, weather, index)
            fast = native.ask(being.request(None, END, probe={"trace": True}))
            stepped = native.ask(being.request(None, END, probe={"fast_path": False, "trace": True}))
            assert fast["state"] == stepped["state"] and fast["hash"] == stepped["hash"], weather
            assert fast["done"] and fast["at"] == END and fast["alarm"] == 0, weather
            assert fast["state"]["organs"]["stage"]["dormant"], (weather, "asleep at the tenth year")
            trace, witness = fast["trace"], stepped["trace"]
            assert trace["fast_path_days"] >= 300 and trace["fast_path_skipped"] >= 1, (weather, trace["fast_path_days"], trace["fast_path_skipped"])
            assert witness["fast_path_days"] == 0 and witness["calls"]["chem"]["fast"]["dormant"] > 0, weather
            assert trace["calls"]["chem"]["fast"]["dormant"] == 0, weather
            assert fast["work"] < stepped["work"], (weather, fast["work"], stepped["work"])
            met[weather + " jumps"] = trace["calls"]["clock"]["jump"]

            # The last thirty days, from the native state, in the reference: its bytes on the fast path,
            # and the same state on the stepped path.
            start = END - 30 * DAY
            origin = native.ask(being.request(None, start))["state"]
            assert origin["organs"]["stage"]["dormant"], (weather, "the last thirty days are asleep")
            request = being.request(origin, END, probe={"trace": True})
            mine = engine.protocol.call(engine.wire.emit(request))
            assert native.bytes(request) == mine, weather
            window = engine.wire.parse(mine)
            assert window["hash"] == fast["hash"], weather
            assert window["trace"]["fast_path_days"] >= 1 and window["trace"]["fast_path_skipped"] >= 1, (weather, window["trace"])
            slow = engine.ask(being.request(origin, END, probe={"fast_path": False}))
            assert slow["hash"] == fast["hash"], weather
            met[weather] = met.get(weather, 0) + 1
    finally:
        restore()
    assert met.get("windowsill") == 1 and met.get("garden") == 1, met
    assert met["garden jumps"] >= 9, ("the garden wakes every spring", met)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
