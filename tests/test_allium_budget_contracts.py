#!/usr/bin/env python3
"""Contracts for the componion's work: the fast path, the day ceilings, the organ order and quiescence.

A being's life costs work units: each processed minute, each fact, each
day of the fast path, each weather draw, and each primitive an organ runs.
The law's unit table says what each costs, and the day ceilings it implies
bound every day of life.

  * AQ1 -- the fast path equals the steps. From dormant states, production
    and the stepped path (``probe.fast_path: false``) reach the same bytes: a
    windowsill span of 1, 2, 7, 30 and 365 days; a garden stretch of twenty
    days with a winter entry and its wake; a jump west across a midnight
    inside a dormant span (a repeated day index) and a jump east (a skipped
    one); a pending ``evolve`` whose minute a later ``tz`` fact moved off
    its midnight, reached at and just after that minute. Presence: days of
    the fast path, a quiescent organ's ``jump``, a wake inside a span, a
    repeated index passed over. Witness: the old rule, under which the fast
    path fires at every local midnight, differs on the jump west.
  * AQ3 -- the work of a day, by stage. Four founders, two in a garden and
    two on a windowsill, forty days each, each day of life its own call:
    every day costs at most the law's ``awake_day`` ceiling, a dormant day
    without a fact at most its ``dormant_day``, both under a million. The day
    the ceiling's proof describes -- every budgeted and capped fact at a
    minute of its own, two jumps east, three daily firings -- costs exactly
    ``awake_day`` in the garden. The ceilings the author script computes
    from the unit table, the budgets and the caps (loaded by path) are the
    ones both laws record. A dormant day without a fact that meets a
    pending ``evolve`` a later ``tz`` fact moved off its midnight costs at
    most ``dormant_day``, in both weathers: exactly one visit more than the
    same day with the ``evolve`` kept on its midnight, where the visit stands
    in for the fast path's day. Presence: the being sleeps before and after,
    no fact that day, the ``evolve`` comes due in it. Witness: in the garden
    the day costs ``dormant_day`` exactly, past the ceiling without the
    visit.
  * AQ4 -- a dormant year. A windowsill being asleep from drought, advanced
    365 days without a fact, costs at most 366 dormant days and under a
    million units, lives exactly 365 days of the fast path and stays asleep.
    With no ``evolve`` due, it costs at most 366 dormant days without the
    ceiling's visit.
    Witness: the stepped path costs 36135 units for the same state.
  * AQ5 -- Jacobi. All 24 orders of the organs give the same bytes over two
    days from three states: the day the soil falls under the dry threshold,
    a garden day with waters and a warm, and a winter wake. Witness: with a
    soil that leaks its new level onto the bus within the step, the orders
    give different bytes, so the order asked for is the order stepped in.
  * AQ6 -- the analytic count. On fixture lives with waters, warms, ``tz``
    facts, a noted fact and a params ``evolve``, in both weathers, on the
    fast path and off it, the work every trace accounts for by the law's
    unit table equals the work the engine counted.
  * AQ8 -- quiescence. Over a life that sleeps and wakes, the organs that
    ``jump`` are exactly the law's quiescent ones, they are never called
    while the being sleeps, and the soil and the stage are called asleep and
    never ``jump``. Witness: off the fast path the quiescent organs are called
    while the being sleeps.

The reference engine only; the twin answers the same requests later, in the
equivalence suite. Local-only. The modules load through the shared
isolation window.
"""

import itertools
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_life_support as support  # noqa: E402
from _allium_window import open_allium  # noqa: E402

BUDGET_S = {
    "test_aq1_the_fast_path_reaches_the_stepped_paths_bytes_from_every_dormant_state": 2.0,
    "test_aq3_no_day_of_life_costs_more_than_its_ceiling_and_the_ceilings_recompute": 2.0,
    "test_aq4_a_dormant_year_costs_a_few_units_a_day_on_the_fast_path": 2.0,
    "test_aq5_every_order_of_the_organs_gives_the_same_bytes": 2.0,
    "test_aq6_the_work_a_trace_accounts_for_is_the_work_counted": 2.0,
    "test_aq8_the_quiescent_organs_are_exactly_the_ones_skipped_while_the_being_sleeps": 2.0,
}

DAY = 1440
MILLION = 1000000
HEX64 = "ab" * 32
HEX32 = "cd" * 16


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


def _at_local(being, day, minute, offset):
    """The minute of life of local minute ``minute`` on the ``day``-th local day after the birth day."""
    first = (being.b + being.tz_birth) // DAY
    return (first + day) * DAY + minute - offset - being.b


def _pair(being, to, state, **probe):
    """Production and the stepped path from the same state to ``to``, both traced."""
    fast = being.advance(to, state, probe=dict(probe, trace=True))
    stepped = being.advance(to, state, probe=dict(probe, trace=True, fast_path=False))
    return fast, stepped


def _asleep(answer):
    return answer["state"]["organs"]["stage"]["dormant"]


# ---------------------------------------------------------------------------
# AQ1 -- the fast path equals the steps
# ---------------------------------------------------------------------------
def test_aq1_the_fast_path_reaches_the_stepped_paths_bytes_from_every_dormant_state(engine):
    met = {"fast path days": 0, "jump": 0, "wake in a span": 0, "skipped": 0, "evolve moved": 0}

    # A windowsill being asleep from drought: spans of 1, 2, 7, 30 and 365 days, from a minute off the grid.
    dry = support.Being(engine, suite="budget", index=0, weather="windowsill")
    base = dry.advance(21 * DAY + 187)
    assert _asleep(base)
    for days in (1, 2, 7, 30, 365):
        fast, stepped = _pair(dry, base["at"] + days * DAY, base["state"])
        assert fast["hash"] == stepped["hash"] and fast["state"] == stepped["state"], days
        assert fast["trace"]["fast_path_days"] == days and stepped["trace"]["fast_path_days"] == 0, days
        met["fast path days"] += fast["trace"]["fast_path_days"]

    # A garden stretch of twenty days: the winter entry and the wake, with no fact, so the wake is in a span.
    garden = support.Being(engine, suite="budget", index=1, weather="garden")
    start = garden.advance(20 * DAY + 611)
    assert not _asleep(start)
    fast, stepped = _pair(garden, 40 * DAY, start["state"])
    assert fast["hash"] == stepped["hash"], "the winter entry and the wake"
    assert not _asleep(fast) and fast["trace"]["fast_path_days"] >= 5, fast["trace"]
    jumps = fast["trace"]["calls"]["chem"]["jump"] + fast["trace"]["calls"]["clock"]["jump"]
    assert jumps == 2 and stepped["trace"]["calls"]["chem"]["jump"] == 1, "one jump per quiescent organ at the wake"
    met["jump"] += jumps
    met["wake in a span"] += jumps > 0 and not garden.after(start["at"], 40 * DAY)

    # Jumps of the zone asleep in a garden winter: west across a midnight (a repeated index), then east across two.
    zone = support.Being(engine, suite="budget", index=2, weather="garden")
    zone.tz(_at_local(zone, 26, 60, 0), -600)
    zone.tz(_at_local(zone, 27, 9 * 60 + 7, -600), -840)
    zone.tz(_at_local(zone, 29, 22 * 60 + 9, -840), 840)
    start = zone.advance(25 * DAY + 33)
    assert _asleep(start)
    for to in (27 * DAY + 5, 28 * DAY, 30 * DAY + 900, 38 * DAY):
        fast, stepped = _pair(zone, to, start["state"])
        assert fast["hash"] == stepped["hash"], to
    assert fast["trace"]["fast_path_skipped"] >= 1, "a repeated day index is passed over, not lived again"
    met["skipped"] += fast["trace"]["fast_path_skipped"]
    # Witness: under the old rule the fast path fires at every local midnight, and the jump west differs.
    world = engine.world
    kept = world._fast_path_fires
    world._fast_path_fires = lambda day, high: True
    try:
        old = zone.advance(28 * DAY, start["state"])
    finally:
        world._fast_path_fires = kept
    new = zone.advance(28 * DAY, start["state"])
    assert old["hash"] != new["hash"] and old["state"]["organs"] != new["state"]["organs"], \
        "witness: firing at a repeated index is seen"

    # A pending evolve whose minute a tz fact moved off its midnight, reached at that minute and just after it.
    moved = support.Being(engine, suite="budget", index=3, weather="windowsill")
    e = _at_local(moved, 25, 10 * 60 + 1, 0)
    effective = moved.midnight(e, 0)
    moved.evolve(e, moved.params(evap_dormant=3000), effective_from=effective)
    moved.tz(_at_local(moved, 25, 15 * 60 + 2, 0), 300)
    start = moved.advance(22 * DAY + 5)
    assert _asleep(start)
    after = moved.midnight(effective, 300)
    for to in (effective - 1, effective, effective + 1, effective + 7 * 60, after - 1):
        fast, stepped = _pair(moved, to, start["state"])
        assert fast["hash"] == stepped["hash"], to
        assert (fast["state"]["params"]["evap_dormant"] == 3000) == (to >= effective), to
    assert (engine.civil.local(moved.b, effective, 300)[1]) == 300, "the evolve's minute is off its midnight"
    met["evolve moved"] += 1
    assert min(met.values()) >= 1, met


# ---------------------------------------------------------------------------
# AQ3 -- the work of a day, by stage
# ---------------------------------------------------------------------------
def _constructed_day(engine, weather):
    """The day the awake ceiling's proof describes, on day of life 1: every fact a day allows, each on its own minute."""
    being = support.Being(engine, suite="budget", index=30, weather=weather, wall=1760010020, tz=-840)
    assert being.b % DAY == 700
    law = engine.lawfiles.law("fixture")
    first = DAY
    jumps = {first + 745: 0, first + 1345: 840}
    minutes = [t for t in range(first, first + DAY) if (being.b + t) % 15 and t not in jumps]
    slots = iter(minutes)
    offset = -840
    plan = []
    budgets = law["journal"]["budgets"]
    table = engine.lawfiles.table("journal_v1")["kinds"]
    bodies = {"act": {"act": "water"}, "dream_depth": {"depth": "normal"}, "forget_rhythm": {},
              "lang_forget": {"target": HEX64}, "lang_taboo_add": {"len": 3, "sha256": HEX64},
              "lang_teach": {"payload": HEX32}, "move_pot": {}, "name": {"name": "a name"}, "rest_begin": {},
              "rest_end": {}, "clock": {"behind": 1}, "owner": {"from": HEX32, "to": HEX32},
              "resumed": {"digest": HEX64, "removed": 0}}
    for kind, budget in sorted(budgets.items()):
        if table[kind]["body"] is None:
            continue
        for _ in range(budget):
            plan.append(kind)
    for kind, cap in sorted(law["work"]["caps"].items()):
        plan.extend([kind] * (cap - (2 if kind == "tz" else 0)))
    draw = engine.rng.Stream(bytes(32), "test.budget", 31)
    for i in range(len(plan) - 1, 0, -1):
        j = draw.below(i + 1)
        plan[i], plan[j] = plan[j], plan[i]
    pinned = False
    for t, kind in sorted([(next(slots), kind) for kind in plan] + [(t, "jump") for t in jumps]):
        for jump_t, to_offset in jumps.items():
            if jump_t <= t and offset != to_offset and jump_t == t:
                offset = to_offset
        if kind == "jump":
            being.tz(t, offset)
        elif kind == "tz":
            being.tz(t, offset)
        elif kind == "evolve":
            being.evolve(t, being.params(), effective_from=being.midnight(t, offset))
        elif kind in ("laws_pin", "laws_unpin"):
            being.fact(t, "laws_unpin" if pinned else "laws_pin", {})
            pinned = not pinned
        else:
            being.fact(t, kind, bodies[kind])
    return being


def test_aq3_no_day_of_life_costs_more_than_its_ceiling_and_the_ceilings_recompute(engine):
    law = engine.lawfiles.law("fixture")
    ceilings = law["work"]["ceilings"]
    awake_day, dormant_day = ceilings["awake_day"], ceilings["dormant_day"]
    assert (awake_day, dormant_day) == (4209, 7)
    met = {"dormant days": 0, "awake days": 0, "facts": 0, "wakes": 0}
    for index, weather in ((10, "garden"), (11, "garden"), (12, "windowsill"), (13, "windowsill")):
        being = support.Being(engine, suite="budget", index=index, weather=weather,
                              hemisphere="south" if index % 2 else "north")
        for day in (3, 11, 29, 36):
            being.act(_at_local(being, day, 8 * 60 + index, 0), "water")
        being.act(_at_local(being, 17, 20 * 60 + index, 0), "greet")
        being.tz(_at_local(being, 22, 23 * 60 + 1, 0), 540)
        state = None
        for k in range(1, 41):
            answer = being.advance(DAY * k - 1, state)
            facts = being.after(None if state is None else state["at"], DAY * k - 1)
            asleep = state is not None and state["organs"]["stage"]["dormant"] and _asleep(answer)
            assert answer["work"] <= awake_day <= MILLION, (index, k, answer["work"])
            if asleep and not facts:
                assert answer["work"] <= dormant_day, (index, k, answer["work"])
                met["dormant days"] += 1
            else:
                met["awake days"] += not asleep
            met["facts"] += len(facts)
            met["wakes"] += state is not None and state["organs"]["stage"]["dormant"] and not _asleep(answer)
            state = answer["state"]
    assert met["dormant days"] >= 20 and met["awake days"] >= 40 and met["wakes"] >= 2, met

    # The day of the proof, at the ceiling exactly in the garden, six units under it on a windowsill.
    for weather, expected in (("garden", awake_day), ("windowsill", awake_day - 6)):
        being = _constructed_day(engine, weather)
        facts = being.after(DAY - 1, 2 * DAY - 1)
        assert len(facts) == 155 + 326 and len({fact["t"] for fact in facts}) == len(facts)
        before = being.advance(DAY - 1)
        day = being.advance(2 * DAY - 1, before["state"], probe={"trace": True})
        trace = day["trace"]
        assert day["work"] == expected, (weather, day["work"], trace)
        assert trace["noted"] == 0 and trace["visits"] == 96 + 481 and trace["fast_path_days"] == 0
        assert trace["calls"]["stage"]["daily"]["awake"] == 3 and day["state"]["tz"] == 840, "three firings"
        assert not _asleep(day) and day["alarm"] == 0

    # The ceilings recompute from the unit table, the budgets and the caps, for both laws.
    author = support.load_script("allium_author_genome.py", "_aq3_author")
    table = engine.lawfiles.table("journal_v1")
    for name in engine.lawfiles.LAWS:
        carried = engine.lawfiles.law(name)
        assert author.ceilings(carried, table) == carried["work"]["ceilings"], name
        assert author.caps(carried, table) == carried["work"]["caps"], name
        assert carried["work"]["ceilings"]["awake_day"] <= MILLION, name
    assert author.ceilings(engine.lawfiles.law("v0_1"), table) == {"awake_day": 4591, "dormant_day": 7}

    # A factless dormant day that meets a pending evolve a later tz fact moved off its midnight: the
    # stepped path visits that minute while the being sleeps, and the ceiling counts the visit.
    visit = law["work"]["units"]["visit"]
    costs = {}
    for weather in ("garden", "windowsill"):
        for moved_by in (300, 0):
            being = support.Being(engine, suite="budget", index=14, weather=weather)
            e = _at_local(being, 27, 2 * 60 + 3, 0)
            effective = being.midnight(e, 0)
            being.evolve(e, being.params(evap_dormant=2000), effective_from=effective)
            being.tz(_at_local(being, 27, 4 * 60 + 1, 0), moved_by)
            k = effective // DAY
            start = being.advance(k * DAY - 1)
            day = being.advance((k + 1) * DAY - 1, start["state"], probe={"trace": True})
            assert being.after(start["at"], day["at"]) == [] and _asleep(start) and _asleep(day), \
                ("presence: a dormant day of life without a fact", weather, moved_by)
            assert start["state"]["pending"]["effective_from"] == effective and day["state"]["pending"] is None
            assert day["state"]["params"]["evap_dormant"] == 2000, "presence: the evolve came due that day"
            assert engine.civil.local(being.b, effective, moved_by)[1] == moved_by, "off its midnight, or on it"
            assert day["work"] <= dormant_day and day["trace"]["visits"] == 1, (weather, moved_by, day["trace"])
            costs[weather, moved_by] = day["work"]
    for weather in ("garden", "windowsill"):
        assert costs[weather, 300] == costs[weather, 0] + visit, ("the move costs one visit", costs)
    assert costs["garden", 300] == dormant_day > dormant_day - visit, ("witness: past the ceiling without it", costs)


# ---------------------------------------------------------------------------
# AQ4 -- a dormant year
# ---------------------------------------------------------------------------
def test_aq4_a_dormant_year_costs_a_few_units_a_day_on_the_fast_path(engine):
    being = support.Being(engine, suite="budget", index=40, weather="windowsill")
    start = being.advance(21 * DAY + 187)
    state = start["state"]
    assert state["organs"]["stage"]["dormant"] and state["organs"]["stage"]["cause"] == "dry"
    ceiling = engine.lawfiles.law("fixture")["work"]["ceilings"]["dormant_day"]
    year = being.advance(start["at"] + 365 * DAY, state, probe={"trace": True})
    assert year["work"] <= 366 * ceiling and year["work"] <= MILLION, year["work"]
    visit = engine.lawfiles.law("fixture")["work"]["units"]["visit"]
    assert state["pending"] is None and year["state"]["pending"] is None, "no evolve comes due in the year"
    assert year["work"] <= 366 * (ceiling - visit), "with no due evolve, the ceiling's visit is never paid"
    assert year["trace"]["fast_path_days"] == 365 and _asleep(year), year["trace"]
    assert year["work"] == 1460, "a day of the fast path, and the soil's and the stage's dormant steps"
    stepped = being.advance(start["at"] + 365 * DAY, state, probe={"fast_path": False})
    assert stepped["work"] == 36135 > 366 * ceiling, "witness: stepping every minute costs past the ceiling"
    assert stepped["hash"] == year["hash"]


# ---------------------------------------------------------------------------
# AQ5 -- Jacobi
# ---------------------------------------------------------------------------
def test_aq5_every_order_of_the_organs_gives_the_same_bytes(engine):
    orders = list(itertools.permutations(support.ORGANS))
    assert len(orders) == 24
    starts = []
    # The day the soil falls under the dry threshold: a windowsill loses 4096 a day from 32768.
    dry = support.Being(engine, suite="budget", index=50, weather="windowsill")
    starts.append(("dry threshold", dry, dry.advance(4 * DAY - 300)))
    theta_dry = engine.lawfiles.law("fixture")["constants"]["stage"]["theta_dry"]
    # A garden day with waters, a warm and a greet.
    garden = support.Being(engine, suite="budget", index=51, weather="garden")
    for minute, act in ((7 * 60 + 1, "water"), (12 * 60 + 2, "warm"), (13 * 60 + 3, "greet"), (19 * 60, "water")):
        garden.act(_at_local(garden, 6, minute, 0), act)
    starts.append(("garden day", garden, garden.advance(6 * DAY - 200)))
    # A winter wake, with no fact: the wake comes from the daily layer.
    winter = support.Being(engine, suite="budget", index=52, weather="garden")
    probe = winter.advance(40 * DAY, probe={"trace": True})
    starts.append(("winter wake", winter, None))
    seen = {}
    for name, being, start in starts:
        if name == "winter wake":
            day = 30
            while True:
                answer = being.advance(day * DAY)
                following = being.advance((day + 2) * DAY, answer["state"])
                if _asleep(answer) and not _asleep(following):
                    start = answer
                    break
                day += 1
                assert day < 40, "a winter wake between day 30 and day 40"
        state = start["state"]
        hashes = set()
        for order in orders:
            answer = being.advance(start["at"] + 2 * DAY, state, probe={"order": list(order)})
            hashes.add(answer["hash"])
            seen[name] = answer
        assert len(hashes) == 1, (name, len(hashes))
    assert seen["dry threshold"]["state"]["organs"]["soil"]["m"] < theta_dry <= starts[0][2]["state"]["organs"]["soil"]["m"]
    assert seen["dry threshold"]["state"]["organs"]["stage"]["dry"] >= 1
    assert seen["garden day"]["state"]["n"] - starts[1][2]["state"]["n"] == 4, "the day's four gestures"
    assert not _asleep(seen["winter wake"]) and probe["trace"]["calls"]["chem"]["jump"] >= 1

    # Witness: were the soil's new level read within the same daily step (a bus that leaks), the order would
    # show, so the order the probe asks for is the order the organs step in.
    soil = engine.world.MODULES["soil"]
    real = soil.daily

    def leaky(org, cx, day):
        real(org, cx, day)
        cx.bus = dict(cx.bus, moisture=org["m"])

    _name, being, start = starts[0]
    soil.daily = leaky
    try:
        leaked = {being.advance(start["at"] + 2 * DAY, start["state"], probe={"order": list(order)})["hash"]
                  for order in orders}
    finally:
        soil.daily = real
    assert len(leaked) > 1, "witness: a leaking bus gives the orders different bytes"


# ---------------------------------------------------------------------------
# AQ6 -- the analytic count
# ---------------------------------------------------------------------------
def test_aq6_the_work_a_trace_accounts_for_is_the_work_counted(engine):
    law = engine.lawfiles.law("fixture")
    met = {"noted": 0, "fast path days": 0, "dormant fast": 0, "draws": 0, "water": 0, "warm": 0, "evolve": 0, "tz": 0}
    for index, weather in ((60, "garden"), (61, "windowsill")):
        being = support.Being(engine, suite="budget", index=index, weather=weather, tz=-120)
        for day in (1, 9, 44, 47):
            being.act(_at_local(being, day, 7 * 60 + 11, -120), "water")
        being.act(_at_local(being, 31, 16 * 60, -120), "warm")
        being.act(_at_local(being, 32, 16 * 60, -120), "warm")
        for minute in (100, 200, 300, 400):
            being.fact(_at_local(being, 5, 600 + minute, -120), "move_pot", {})
        e = _at_local(being, 12, 13 * 60 + 13, -120)
        being.evolve(e, being.params(evap_awake=8000, rain_gain=70000), effective_from=being.midnight(e, -120))
        being.tz(_at_local(being, 15, 3 * 60 + 3, -120), 480)
        stops = [DAY * k + 17 * k for k in range(1, 50, 7)] + [50 * DAY]
        for fast_path in (True, False):
            state = None
            for stop in stops:
                answer = being.advance(stop, state, probe={"fast_path": fast_path, "trace": True})
                trace = answer["trace"]
                assert support.analytic(trace, law) == answer["work"], (weather, fast_path, stop, trace)
                met["noted"] += trace["noted"]
                met["fast path days"] += trace["fast_path_days"]
                met["dormant fast"] += trace["calls"]["chem"]["fast"]["dormant"]
                met["draws"] += trace["draws"]
                met["water"] += trace["acts"].get("water", 0)
                met["warm"] += trace["acts"].get("warm", 0)
                met["evolve"] += answer["state"]["params"]["evap_awake"] == 8000
                met["tz"] += answer["state"]["tz"] == 480
                state = answer["state"]
    assert min(met.values()) >= 1, met


# ---------------------------------------------------------------------------
# AQ8 -- quiescence
# ---------------------------------------------------------------------------
def test_aq8_the_quiescent_organs_are_exactly_the_ones_skipped_while_the_being_sleeps(engine):
    quiescent = engine.lawfiles.law("fixture")["quiescent_in_dormancy"]
    assert sorted(quiescent) == ["chem", "clock"]
    being = support.Being(engine, suite="budget", index=80, weather="windowsill")
    being.act(_at_local(being, 30, 9 * 60 + 9, 0), "water")
    stops = [DAY * k for k in range(5, 60, 5)]
    traces = {}
    for fast_path in (True, False):
        state = None
        parts = []
        for stop in stops:
            answer = being.advance(stop, state, probe={"fast_path": fast_path, "trace": True})
            parts.append(answer["trace"])
            state = answer["state"]
        traces[fast_path] = support.add_traces(parts)
    calls = traces[True]["calls"]
    jumped = sorted(name for name, layers in calls.items() if layers["jump"] > 0)
    assert jumped == sorted(quiescent), jumped
    for name in quiescent:
        assert calls[name]["fast"]["dormant"] == 0 and calls[name]["daily"]["dormant"] == 0, name
        assert calls[name]["fast"]["awake"] > 0, name
    for name in ("soil", "stage"):
        assert calls[name]["daily"]["dormant"] > 0 and calls[name]["jump"] == 0, name
    assert traces[True]["fast_path_days"] > 0
    stepped = traces[False]["calls"]
    assert stepped["chem"]["fast"]["dormant"] > 0 and stepped["clock"]["fast"]["dormant"] > 0, \
        "witness: off the fast path the quiescent organs are called while the being sleeps"
    assert sorted(name for name, layers in stepped.items() if layers["jump"] > 0) == sorted(quiescent)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
