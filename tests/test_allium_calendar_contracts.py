#!/usr/bin/env python3
"""Contracts for the componion's calendar and local time: a minute of life read as a civil date.

A being lives in minutes since its birth. Its local minute is the birth
minute in UTC, plus the minute of life, plus the offset in force; the local
day is that minute's day number since 1970-01-01, and the calendar reads it
as a civil date. Both engines compute it with integers only, and the
standard library's ``datetime`` is the independent opinion.

  * CY5 -- local time is exact. For every one of the 113 quarter-hour
    offsets, and for an offset list in which the last entry at or before a
    minute decides it, ``grid`` gives the local minute, the local day, the
    civil date, the fast boundary and the next local midnight that
    ``datetime`` gives with a fixed offset, in both engines. The
    reference's year position, season, daylength and sun agree with oracles
    written apart (the calendar from ``datetime``, the wrap of a year
    position below the equinox without a modulo), for both worlds of the
    law data and both hemispheres, and the light of one minute costs four
    units. Presence: negative offsets, a local midnight asked for itself,
    year positions on both sides of each equinox. And in a life, with the
    reference's ``advance``: across a jump of the time zone west that
    repeats a local day and one east that skips a local day, while the
    being sleeps, the high-water mark ``state.day`` rises once per index it
    lives, the skipped index is never lived, the repeated one is not lived
    again (the fast path passes it over), and the stage's daily step runs
    exactly once per rise.
  * CY6 -- the calendar equals ``datetime`` for every day from 1970-01-01
    to 2199-12-31, both ways; 2100, the one year of the range where the
    century rule decides, has no 29 February. Both engines answer ``grid``
    with the same bytes on 6000 seeded minutes under an offset list, on
    every 28 and 29 February, 1 March, 31 December and 1 January of the
    range, and at the edges of what a request may carry.

Local-only. The modules load through the shared isolation window; both
contracts need the native artefact that ``scripts/build_oo_core.sh`` builds.
"""

import datetime
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_life_support as life_support  # noqa: E402
from _allium_window import native_module, open_allium  # noqa: E402

BUDGET_S = {
    "test_cy5_local_time_is_exact_and_a_tz_jump_lives_each_local_day_once": 2.0,
    "test_cy6_the_calendar_equals_the_standard_library_for_every_day_from_1970_to_2199_in_both_engines": 2.0,
}

DAY = 1440
MAX_INT = (1 << 53) - 1
T_MAX = MAX_INT - 2880
WALL = 1760000000
OFFSETS = tuple(range(-840, 841, 15))
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
EPOCH_DATE = datetime.date(1970, 1, 1)
MINUTE = datetime.timedelta(minutes=1)

# The two worlds of the law data, as values: the calendar functions take them as arguments.
WORLDS = {
    "fixture": {
        "daylength": {"amp": {"long": 240, "medium": 150, "short": 60}, "equinox": 10, "mean": 720, "ramp": 60},
        "rain": {"max": [21845, 21845, 16384, 21845], "p_wet": [39322, 32768, 22938, 32768]},
        "season_shift": 5,
        "south_shift": 20,
        "year": {"days": 40, "kind": "fixed"},
    },
    "v0_1": {
        "daylength": {"amp": {"long": 240, "medium": 150, "short": 60}, "equinox": 79, "mean": 720, "ramp": 60},
        "rain": {"max": [21845, 21845, 16384, 21845], "p_wet": [39322, 32768, 22938, 32768]},
        "season_shift": 31,
        "south_shift": 183,
        "year": {"kind": "civil"},
    },
}
BANDS = ("long", "medium", "short")
HEMISPHERES = ("north", "south")


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


class _Engines:
    """The reference and the native engine of one window, asked the same requests."""

    def __init__(self, loaded):
        self.loaded = loaded
        self.wire = loaded["opti_oignon.allium.wire"]
        self.fx = loaded["opti_oignon.allium.fx"]
        self.rng = loaded["opti_oignon.allium.rng"]
        self.lawfiles = loaded["opti_oignon.allium.lawfiles"]
        self.civil = loaded["opti_oignon.allium.ref.civil"]
        self.protocol = loaded["opti_oignon.allium.ref.protocol"]
        self.native = native_module(loaded)
        self.native_calls = 0

    def both(self, request):
        data = self.wire.emit(request)
        ref = self.protocol.call(data)
        nat = bytes(self.native.allium_call(data))
        self.native_calls += 1
        assert ref == nat, (data[:160], ref[:240], nat[:240])
        return self.wire.parse(ref)


@pytest.fixture
def engines():
    loaded, restore = open_allium(native=True)
    try:
        yield _Engines(loaded)
    finally:
        restore()


def _oracle(b, t, z):
    """What ``datetime`` says of minute ``t`` of a life born in UTC minute ``b``, under the fixed offset ``z``."""
    zone = datetime.timezone(datetime.timedelta(minutes=z))
    instant = EPOCH + datetime.timedelta(minutes=b + t)
    here = instant.astimezone(zone)
    midnight = datetime.datetime.combine(here.date() + datetime.timedelta(days=1), datetime.time(0), tzinfo=zone)
    return {
        "civil": [here.year, here.month, here.day],
        "day": (here.date() - EPOCH_DATE).days,
        "fast": instant.minute % 15 == 0,
        "midnight": (midnight - EPOCH) // MINUTE - b,
        "minute": here.hour * 60 + here.minute,
        "offset": z,
    }


def _offset_oracle(entries, t, birth_tz):
    """The offset in force at ``t``: the last entry at or before it, read front to back."""
    offset = birth_tz
    for entry_t, entry_z in entries:
        if entry_t <= t:
            offset = entry_z
    return offset


def _date(day):
    return EPOCH_DATE + datetime.timedelta(days=day)


def _position_oracle(world, hemisphere, day):
    """The year position and year length from ``datetime``, with the southern shift added by a wrap."""
    if world["year"]["kind"] == "civil":
        date = _date(day)
        start = datetime.date(date.year, 1, 1)
        p = (date - start).days
        length = (datetime.date(date.year + 1, 1, 1) - start).days
    else:
        length = world["year"]["days"]
        p = day - (day // length) * length
    if hemisphere == "south":
        p += world["south_shift"]
        if p >= length:
            p -= length
    return p, length


def _season_oracle(world, p, length):
    q = p + world["season_shift"]
    if q >= length:
        q -= length
    season = 0
    while season < 3 and 4 * q >= (season + 1) * length:
        season += 1
    return season


def _daylength_oracle(world, band, p, length, sine):
    delta = p - world["daylength"]["equinox"]
    if delta < 0:
        delta += length
    bam = (delta * 1024) // length
    return world["daylength"]["mean"] + ((world["daylength"]["amp"][band] * sine[bam]) >> 15)


def _sun_oracle(world, minute, dl, sun_max):
    ramp = world["daylength"]["ramp"]
    rise = 720 - dl // 2
    light = min(max(minute - rise, 0), ramp, max(rise + dl - minute, 0))
    return (light * sun_max) // ramp


# ---------------------------------------------------------------------------
# CY5 -- local time is exact
# ---------------------------------------------------------------------------
def test_cy5_local_time_is_exact_and_a_tz_jump_lives_each_local_day_once(engines):
    e = engines
    draw = e.rng.Stream(bytes(32), "test.calendar", 5)
    met = {"negative": 0, "positive": 0, "zero": 0, "at midnight": 0, "fast": 0, "not fast": 0}
    days = set()

    # Every quarter-hour offset as the birth offset, with no entry: z(t) is birth.tz.
    for z in OFFSETS:
        wall = 86400 + draw.below(2 * WALL)
        b = wall // 60
        first_midnight = _oracle(b, 0, z)["midnight"]
        ts = [0, first_midnight - 1, first_midnight, first_midnight + DAY - 1, first_midnight + DAY]
        ts += [draw.below(400 * 366 * DAY) for _ in range(11)]
        answer = e.both({"birth": {"tz": z, "wall": wall}, "op": "grid", "ts": ts, "tz": [], "v": 1})
        assert answer["work"] == len(ts)
        wrong = [(t, got, _oracle(b, t, z)) for t, got in zip(ts, answer["out"]) if got != _oracle(b, t, z)]
        assert wrong == [], (z, wall, wrong[:3])
        assert len(answer["out"]) == len(ts)
        met["negative" if z < 0 else "positive" if z > 0 else "zero"] += 1
        for got in answer["out"]:
            met["at midnight"] += got["minute"] == 0
            met["fast" if got["fast"] else "not fast"] += 1
            days.add(got["day"])
        assert answer["out"][2]["minute"] == 0 and answer["out"][2]["midnight"] == first_midnight + DAY, \
            "a local midnight asked for itself gives the next one, strictly after it"
    assert met["negative"] == 56 and met["positive"] == 56 and met["zero"] == 1, met
    assert met["at midnight"] >= 113 and met["fast"] > 0 and met["not fast"] > 0, met

    # An offset list: the last entry at or before a minute decides it, birth.tz before the first.
    birth_tz = -345
    wall = WALL + 17
    b = wall // 60
    order = list(OFFSETS)
    for i in range(len(order) - 1, 0, -1):
        j = draw.below(i + 1)
        order[i], order[j] = order[j], order[i]
    entries = []
    t = 500
    for i, z in enumerate(order):
        entries.append([t, z])
        if i % 9 != 4:
            t += 1 + draw.below(9 * DAY)
    same_t = [entry_t for (entry_t, _), (next_t, _) in zip(entries, entries[1:]) if entry_t == next_t]
    assert len(same_t) >= 10, "witness: several entries share a minute, and the last of them wins"
    ts = [0, 499] + [entry_t for entry_t, _ in entries] + [entry_t - 1 for entry_t, _ in entries]
    ts += [draw.below(t + 2 * DAY) for _ in range(300)]
    answer = e.both({"birth": {"tz": birth_tz, "wall": wall}, "op": "grid", "ts": ts, "tz": entries, "v": 1})
    assert answer["work"] == len(ts)
    wanted = [_oracle(b, t, _offset_oracle(entries, t, birth_tz)) for t in ts]
    wrong = [(t, got, want) for t, got, want in zip(ts, answer["out"], wanted) if got != want]
    assert wrong == [], wrong[:3]
    assert len(answer["out"]) == len(ts)
    assert answer["out"][0]["offset"] == answer["out"][1]["offset"] == birth_tz, "before the first entry: birth.tz"
    last_of_pair = {entry_t: z for entry_t, z in entries}
    assert all(answer["out"][2 + i]["offset"] == last_of_pair[entry_t] for i, (entry_t, _) in enumerate(entries))
    for got in answer["out"]:
        days.add(got["day"])

    # The reference's year position, season, daylength and sun, on every local day met above.
    sine = e.lawfiles.sine()
    below_equinox = {}
    seasons = {}
    for name, world in WORLDS.items():
        for hemisphere in HEMISPHERES:
            wrong = []
            for day in sorted(days):
                p, length = _position_oracle(world, hemisphere, day)
                if e.civil.year_position(day, world, hemisphere) != (p, length):
                    wrong.append(("position", day))
                    continue
                season = e.civil.season(day, world, hemisphere)
                if season != _season_oracle(world, p, length):
                    wrong.append(("season", day, season))
                seasons.setdefault((name, hemisphere), set()).add(season)
                below = p < world["daylength"]["equinox"]
                below_equinox[(name, hemisphere, below)] = below_equinox.get((name, hemisphere, below), 0) + 1
                for band in BANDS:
                    work = e.fx.Work()
                    dl = e.civil.daylength(day, world, hemisphere, band, sine, work)
                    if dl != _daylength_oracle(world, band, p, length, sine) or work.units != 1:
                        wrong.append(("daylength", day, band, dl))
            assert wrong == [], (name, hemisphere, wrong[:4])
            assert below_equinox.get((name, hemisphere, True), 0) > 0, ("presence: p < equinox", name, hemisphere)
            assert below_equinox.get((name, hemisphere, False), 0) > 0, ("presence: p >= equinox", name, hemisphere)
            assert seasons[(name, hemisphere)] == {0, 1, 2, 3}, (name, hemisphere, seasons[(name, hemisphere)])

    # The civil world's northern winter runs from 1 December to 2 March (to 1 March in a leap year).
    civil_world = WORLDS["v0_1"]
    winters = {False: 0, True: 0}
    for day in range(10957, 10957 + 3 * 366):
        date = _date(day)
        leap = date.year % 4 == 0 and (date.year % 100 != 0 or date.year % 400 == 0)
        winter = date.month == 12 or date.month <= 2 or (date.month == 3 and date.day <= (1 if leap else 2))
        assert (e.civil.season(day, civil_world, "north") == 0) == winter, date
        winters[leap] += winter
    assert winters[True] > 0 and winters[False] > 0, winters
    # The fixture's ten-day seasons: born at WALL under offset 0, winter on life day 25 and spring on day 35.
    birth_day = (WALL // 60) // DAY
    assert birth_day == 20370 and e.civil.year_position(birth_day, WORLDS["fixture"], "north") == (10, 40)
    fixture_seasons = [e.civil.season(birth_day + k, WORLDS["fixture"], "north") for k in (24, 25, 34, 35)]
    assert fixture_seasons == [3, 0, 0, 1], fixture_seasons
    # Long days in June in the north and in December in the south, short ones the other way round.
    june, december = (datetime.date(2031, 6, 21) - EPOCH_DATE).days, (datetime.date(2031, 12, 21) - EPOCH_DATE).days
    for hemisphere, (long_day, short_day) in (("north", (june, december)), ("south", (december, june))):
        long_dl = e.civil.daylength(long_day, civil_world, hemisphere, "long", sine, e.fx.Work())
        short_dl = e.civil.daylength(short_day, civil_world, hemisphere, "long", sine, e.fx.Work())
        assert long_dl > 720 + 200 and short_dl < 720 - 200, (hemisphere, long_dl, short_dl)

    # The sun of a local minute: dark, a ramp, full light, a ramp, dark; the light of one minute costs 4 units.
    shapes = {"dark": 0, "ramp": 0, "full": 0}
    for dl in (0, 1, 59, 60, 61, 119, 120, 480, 719, 720, 721, 960, 1379, 1440):
        for sun_max in (16384, 65536):
            for minute in range(0, DAY, 7):
                work = e.fx.Work()
                got = e.civil.sun(minute, dl, civil_world, sun_max, work)
                want = _sun_oracle(civil_world, minute, dl, sun_max)
                assert got == want and work.units == 3 and work.alarm == 0, (dl, sun_max, minute, got, want)
                shapes["dark" if got == 0 else "full" if got == sun_max else "ramp"] += 1
    assert min(shapes.values()) > 0, shapes
    work = e.fx.Work()
    dl = e.civil.daylength(june, civil_world, "north", "medium", sine, work)
    e.civil.sun(700, dl, civil_world, 65536, work)
    assert work.units == 4 and work.alarm == 0, "the light of one minute costs 4 units"
    assert e.native_calls == 114, "the native engine answered every grid request"

    # In a life: a jump west repeats a local day, a jump east skips one; each index is lived once, while asleep.
    life = life_support.Engine(e.loaded)
    being = life_support.Being(life, suite="calendar", index=7, weather="windowsill")
    first = being.b // DAY
    west = (first + 22) * DAY + 12 * 60 - being.b
    east = (first + 26) * DAY + 21 * 60 + 840 - being.b
    being.tz(west, -840)
    being.tz(east, 840)
    start = being.advance(20 * DAY + 7)
    assert start["state"]["organs"]["stage"]["dormant"], "the jumps come while the being sleeps"
    end = 32 * DAY

    def offset_at(t):
        return 0 if t < west else -840 if t < east else 840

    points = {west - 1, west, east - 1, east, end}
    for t in range(start["at"] + 1, end + 1):
        if (being.b + t + offset_at(t)) % DAY == 0:
            points |= {t - 1, t}
    points = sorted(points)
    answers = life_support.cuts(being, points, probe={"trace": True}, state=start["state"])
    days = [start["state"]["day"]] + [answer["state"]["day"] for answer in answers]
    assert all(later >= earlier for earlier, later in zip(days, days[1:])), "the mark never goes back"
    rises = [later for earlier, later in zip(days, days[1:]) if later > earlier]
    trace = life_support.add_traces([answer["trace"] for answer in answers])
    assert len(rises) == len(set(rises)) and trace["calls"]["stage"]["daily"]["dormant"] == len(rises), \
        "one daily step per index lived, none twice"
    repeated = (being.b + west - 840) // DAY + 1
    skipped = (being.b + east - 840) // DAY + 1
    assert repeated in days and days.count(repeated) >= 3, "the repeated index was lived once, before the jump west"
    assert skipped not in days and skipped + 1 in rises, "the index skipped by the jump east is never lived"
    assert trace["fast_path_skipped"] >= 1 and trace["fast_path_days"] >= 5, trace
    assert all(answer["state"]["organs"]["stage"]["dormant"] for answer in answers)


# ---------------------------------------------------------------------------
# CY6 -- the calendar
# ---------------------------------------------------------------------------
def test_cy6_the_calendar_equals_the_standard_library_for_every_day_from_1970_to_2199_in_both_engines(engines):
    e = engines
    civil = e.civil
    first = EPOCH_DATE.toordinal()
    count = datetime.date(2199, 12, 31).toordinal() - first + 1
    assert count == 84006
    wrong = []
    for day in range(count):
        date = datetime.date.fromordinal(first + day)
        want = (date.year, date.month, date.day)
        if civil.civil_from_days(day) != want or civil.days_from_civil(date.year, date.month, date.day) != day:
            wrong.append((day, want, civil.civil_from_days(day)))
    assert wrong == [], wrong[:4]
    assert civil.civil_from_days(civil.days_from_civil(2100, 2, 28) + 1) == (2100, 3, 1), "2100 is not a leap year"
    assert civil.civil_from_days(civil.days_from_civil(2000, 2, 28) + 1) == (2000, 2, 29), "2000 is a leap year"
    leap_days = [y for y in range(1970, 2200) if civil.civil_from_days(civil.days_from_civil(y, 2, 28) + 1)[1] == 2]
    assert len(leap_days) == 56 and 2000 in leap_days and 2100 not in leap_days, leap_days

    # The dates where a calendar goes wrong first, under a negative offset from the earliest birth allowed.
    draw = e.rng.Stream(bytes(32), "test.calendar", 6)
    birth_tz = -60
    wall = 86400 + 59
    b = wall // 60
    targets = []
    for year in range(1970, 2200):
        targets += [(year, 1, 1), (year, 2, 28), (year, 3, 1), (year, 12, 31)]
        if year in leap_days:
            targets.append((year, 2, 29))
    ts = []
    for target in targets:
        day = (datetime.date(*target) - EPOCH_DATE).days
        minute = 1380 + draw.below(60) if day == 0 else draw.below(DAY)
        ts.append(day * DAY + minute - b - birth_tz)
    answer = e.both({"birth": {"tz": birth_tz, "wall": wall}, "op": "grid", "ts": ts, "tz": [], "v": 1})
    assert [tuple(got["civil"]) for got in answer["out"]] == targets, "every date asked for is the date answered"
    assert len(targets) == 230 * 4 + 56 and ts[0] < 60, "presence: from the first minute of 1970 on"

    # 6000 seeded minutes under an offset list, same-minute pairs included.
    entries = []
    t = 0
    for i in range(60):
        entries.append([t, OFFSETS[draw.below(len(OFFSETS))]])
        if i % 7 != 3:
            t += draw.below(2 * 366 * DAY)
    ts = [draw.below(t + 366 * DAY) for _ in range(6000)]
    answer = e.both({"birth": {"tz": 525, "wall": WALL + 31}, "op": "grid", "ts": ts, "tz": entries, "v": 1})
    assert answer["work"] == 6000 and len(answer["out"]) == 6000
    offsets_met = {got["offset"] for got in answer["out"]}
    assert len(offsets_met) >= 30 and any(z < 0 for z in offsets_met), sorted(offsets_met)

    # The edges of a request: the earliest birth and minute, the latest wall and minute, both extreme offsets.
    edges = [
        {"birth": {"tz": -840, "wall": 86400}, "op": "grid", "ts": [0, 1, 839, 840], "tz": [], "v": 1},
        {"birth": {"tz": 840, "wall": MAX_INT}, "op": "grid", "ts": [0, 1, T_MAX - 1, T_MAX],
         "tz": [[0, -840], [T_MAX, 840], [T_MAX, -840]], "v": 1},
        {"birth": {"tz": 0, "wall": MAX_INT}, "op": "grid", "ts": [], "tz": [], "v": 1},
    ]
    answers = [e.both(request) for request in edges]
    assert answers[0]["out"][0] == {"civil": [1970, 1, 1], "day": 0, "fast": True, "midnight": 840,
                                    "minute": 600, "offset": -840}
    assert answers[1]["out"][3]["offset"] == -840 and answers[1]["out"][3]["midnight"] <= MAX_INT
    assert answers[2] == {"out": [], "work": 0}
    assert e.native_calls == 5, "the native engine answered every grid request"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
