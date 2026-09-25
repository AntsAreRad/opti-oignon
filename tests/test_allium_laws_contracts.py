#!/usr/bin/env python3
"""Contracts for the componion's laws in a life: when an ``evolve`` applies, what a pin keeps, what is refused.

A being lives under a law and a few params its genesis froze. An ``evolve``
fact registers a change at the next local midnight; a pin keeps the law and
params in force; a law this engine does not carry is refused by name, never
approached. ``timeline`` folds the law kinds alone, and must say what
``advance`` says.

  * AV2 -- a params ``evolve`` changes nothing before its minute: against the
    same life without it, the organs, the bus and the params are the same the
    minute before, and differ two days later. An ``evolve`` written at a
    local midnight takes the next one. A ``tz`` fact at the evolve's minute,
    sorted before or after it, gives the same pending change: the minute is
    checked against the offset at the end of its ``tz`` facts. A wrong minute
    is noted ``evolve_when`` and nothing is kept.
  * AV3 -- an unknown law is refused by name: a genesis naming a law this
    engine does not carry (``law``), the fixture law under another digest
    (``law digest``), an ``evolve`` to a law not carried (``law``) or with
    the wrong version (``law version``), and an ``evolve`` from the fixture
    to the full law, which no migration joins (``migration``). With laws
    injected in the reference only: a defective life section (``life law``)
    and an organ code the engine does not implement (``code``).
  * AV4 -- the golden lives: six thirty-day lives whose requests are
    committed in hexadecimal under ``tests/allium_golden/v1`` (four on the
    fixture law -- a northern garden with waters, two moves of the time zone
    and the first day of winter; a southern windowsill through a drought and
    a water after its rest; a garden with a params ``evolve``; a pinned
    being that meets an ``evolve`` while pinned and one after the pin is
    lifted -- and a garden and a short-day windowsill on the full law). The
    reference and the native core answer each with the committed response
    and state digests, work, overhead and minute, and the author script's
    ``--check`` finds nothing to re-record. Each life shows what it was
    written for.
  * AV5 -- a change of law: both engines refuse, with the same bytes, an
    ``evolve`` between the two carried laws, which no migration joins. With
    an injected stable pair, the reference's identity migration gives the
    same bytes in two child processes under different string hash seeds,
    time zones and wall clocks of different parity, and the same bytes as in
    this process; across its minute, the organs, the bus and the day are
    the ones of the same being whose ``evolve`` keeps its law. A successor
    that lowers the ceiling of a level a state keeps (the soil's moisture,
    the stage's day counts) is refused ``migration``.
  * AV6 -- a pin keeps the old law: pinned on day 3, an ``evolve`` written on
    day 5 is noted ``evolve_pinned`` at its minute and the params are the
    same on day 10; unpinned on day 11, an ``evolve`` on day 12 applies.
  * AV9 -- the timeline is the reducer's. On two sixty-day windowsill lives
    with seeded sequences of the law kinds -- same-minute ``tz`` pairs, a
    ``tz`` and an ``evolve`` in the same minute in both canonical orders,
    pins, unpins, valid, invalid and moved ``evolve`` facts, a due ``evolve``
    while pinned and one met by a pin at its minute, a jump east across a
    local midnight -- at 50 seeded minutes,
    ``timeline`` (from the genesis, and from ``advance``'s previous state)
    gives the day, law, params, pending change, pin, versions seen and offset
    ``advance`` gives, and the next local midnight; the firing minutes it
    lists are exactly the minutes where ``advance``'s day rises.
  * AV12 -- the golden lives' author refuses a re-record its law does not
    allow. On copies of the committed files: an answer that changed, a
    request that changed and a life dropped, each under an unchanged law
    digest, are found stale by ``--check`` (exit 1) and refused by
    ``--write`` (exit 2), which writes nothing and names the reason; under a
    moved law digest the same change is re-recorded and the file is current.

AV4 and AV5 ask the native core too, and fail when it is not built here
(``scripts/build_oo_core.sh``); the others ask the reference engine only,
and the twin answers their requests in the equivalence suite. Local-only.
The modules load through the shared isolation window.
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_life_support as support  # noqa: E402
from _allium_window import native_module, open_allium  # noqa: E402
from _isolation import REPO  # noqa: E402

BUDGET_S = {
    "test_av2_a_params_evolve_changes_nothing_before_its_minute": 2.0,
    "test_av3_an_unknown_law_is_refused_by_name_and_never_approached": 2.0,
    "test_av4_the_golden_lives_are_answered_alike_by_both_engines_and_the_author_agrees": 2.0,
    "test_av5_a_change_of_law_is_refused_alike_and_the_identity_migration_reads_no_clock": 2.0,
    "test_av6_a_pin_keeps_the_law_and_params_in_force_until_it_is_lifted": 2.0,
    "test_av9_the_timeline_says_what_the_reducer_says": 2.0,
    "test_av12_the_life_author_refuses_to_re_record_a_life_under_an_unchanged_law": 2.0,
}

DAY = 1440
LOW = "0000000000000a01"
HIGH = "fffffffffffff001"
TIMELINE_KEYS = ("day", "law", "params", "pending", "pinned", "seen", "tz")


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


def _codes(answer):
    return [note["code"] for note in answer["notes"]]


# ---------------------------------------------------------------------------
# AV2 -- a params evolve changes nothing before its minute
# ---------------------------------------------------------------------------
def test_av2_a_params_evolve_changes_nothing_before_its_minute(engine):
    def life(with_evolve):
        being = support.Being(engine, suite="laws", index=0, weather="garden")
        for day in (2, 6, 9):
            being.act(_at_local(being, day, 8 * 60 + 30, 0), "water")
        e = _at_local(being, 10, 12 * 60 + 7, 0)
        if with_evolve:
            being.evolve(e, being.params(evap_awake=9000, sun_max=50000), effective_from=being.midnight(e, 0))
        return being, e, being.midnight(e, 0)

    changed, e, effective = life(True)
    same, _e, _effective = life(False)
    before = [changed.advance(effective - 1), same.advance(effective - 1)]
    for key in ("organs", "bus", "params"):
        assert before[0]["state"][key] == before[1]["state"][key], key
    assert before[0]["state"]["pending"]["effective_from"] == effective and before[1]["state"]["pending"] is None
    assert before[0]["state"]["params"] == changed.genesis["body"]["laws"]["params"]
    later = [changed.advance(effective + 2 * DAY), same.advance(effective + 2 * DAY)]
    assert later[0]["state"]["params"]["sun_max"] == 50000 and later[0]["state"]["pending"] is None
    assert later[0]["state"]["organs"] != later[1]["state"]["organs"], "witness: two days later the lives differ"
    assert later[0]["state"]["organs"]["chem"] != later[1]["state"]["organs"]["chem"]

    # Written at a local midnight, an evolve takes the next one; the midnight itself is a wrong minute.
    at_midnight = support.Being(engine, suite="laws", index=1, weather="garden")
    m = _at_local(at_midnight, 4, 0, 0)
    assert at_midnight.midnight(m, 0) == m + DAY
    at_midnight.evolve(m, at_midnight.params(rain_gain=1000), effective_from=m + DAY)
    at_midnight.evolve(m + 15, at_midnight.params(rain_gain=2000), effective_from=m, origin=HIGH)
    answer = at_midnight.advance(m + 20)
    assert answer["state"]["pending"]["effective_from"] == m + DAY, answer["state"]["pending"]
    assert answer["state"]["pending"]["params"]["rain_gain"] == 1000
    assert _codes(answer) == ["evolve_when"] and answer["notes"][0]["t"] == m + 15, answer["notes"]

    # A tz fact at the evolve's minute, sorted before or after it: the same pending change, at the new offset.
    pendings = []
    for tz_origin, evolve_origin in ((LOW, HIGH), (HIGH, LOW)):
        being = support.Being(engine, suite="laws", index=2, weather="windowsill", tz=120)
        e = _at_local(being, 7, 22 * 60 + 31, 120)
        effective = being.midnight(e, -300)
        being.tz(e, -300, origin=tz_origin)
        being.evolve(e, being.params(evap_dormant=500), effective_from=effective, origin=evolve_origin)
        kinds = [fact["kind"] for fact in being.facts if fact["t"] == e]
        pendings.append((kinds, being.advance(e + 1)["state"]["pending"]))
        wrong = support.Being(engine, suite="laws", index=2, weather="windowsill", tz=120)
        wrong.tz(e, -300, origin=tz_origin)
        wrong.evolve(e, wrong.params(evap_dormant=500), effective_from=wrong.midnight(e, 120), origin=evolve_origin)
        answer = wrong.advance(e + 1)
        assert answer["state"]["pending"] is None and _codes(answer) == ["evolve_when"], \
            "witness: the offset before the minute's tz fact is a wrong minute"
    assert [kinds for kinds, _ in pendings] == [["tz", "evolve"], ["evolve", "tz"]], "both canonical orders met"
    assert pendings[0][1] is not None and pendings[0][1] == pendings[1][1]
    assert pendings[0][1]["effective_from"] == effective


# ---------------------------------------------------------------------------
# AV3 -- an unknown law is refused by name
# ---------------------------------------------------------------------------
def _refusal(engine, request):
    answer = engine.ask(request)
    return answer.get("refused"), answer.get("detail")


def test_av3_an_unknown_law_is_refused_by_name_and_never_approached(engine):
    being = support.Being(engine, suite="laws", index=3, weather="garden")
    assert "refused" not in engine.ask(being.request(None, DAY)), "witness: the fixture being lives"

    def genesis_request(edit):
        request = being.request(None, DAY)
        request["genesis"] = dict(being.genesis, body=dict(being.genesis["body"]))
        request["genesis"]["body"]["laws"] = dict(being.genesis["body"]["laws"])
        edit(request["genesis"]["body"]["laws"])
        return request

    def rename(laws):
        laws["name"] = "fixturex"

    def redigest(laws):
        laws["sha256"] = "0" * 64

    assert _refusal(engine, genesis_request(rename)) == ("unknown_law", "law")
    assert _refusal(engine, genesis_request(redigest)) == ("unknown_law", "law digest")
    timeline = {"facts": [], "from": None, "genesis": genesis_request(redigest)["genesis"], "op": "timeline",
                "to": DAY, "v": 1}
    assert _refusal(engine, timeline) == ("unknown_law", "law digest"), "the timeline refuses by the same checks"

    full = engine.law("v0_1")
    e = _at_local(being, 2, 600, 0)
    cases = (
        (("nowhere", "1" * 64, 0), None, ("unknown_law", "law")),
        (("fixture", being.digest, 7), None, ("unknown_law", "law version")),
        (("fixture", "2" * 64, 0), None, ("unknown_law", "law digest")),
        (("v0_1", full["digest"], full["version"]), None, ("unknown_law", "migration")),
        (("fixture", being.digest, 0), ("v0_1", full["digest"]), ("unknown_law", "migration")),
    )
    for to, source, expected in cases:
        evolving = support.Being(engine, suite="laws", index=3, weather="garden")
        evolving.evolve(e, evolving.params(), effective_from=evolving.midnight(e, 0), to=to, source=source)
        assert _refusal(engine, evolving.request(None, 3 * DAY)) == expected, (to, source)
        before = evolving.request(None, e - 1)
        assert "refused" not in engine.ask(before), "the refusal is the evolve's, met when it is folded"

    # Laws the engine does not carry, injected in the reference only.
    def short_year(law):
        law["world"]["year"]["days"] = 3

    def other_code(law):
        law["code"] = "seed_" + "9"

    for edit, name, detail in ((short_year, "fixture_x", "life law"), (other_code, "fixture_y", "code")):
        files = support.defective(engine, edit, name)
        with support.injected(engine, files):
            digest = engine.lawfiles.digest(engine.wire.parse(files[name]))
            request = being.request(None, DAY)
            request["genesis"] = dict(being.genesis, body=dict(being.genesis["body"]))
            request["genesis"]["body"]["laws"] = dict(being.genesis["body"]["laws"], name=name, sha256=digest)
            assert _refusal(engine, request) == ("unknown_law", detail), name
        assert name not in engine.lawfiles.LAWS, "the carried laws are given back"


# ---------------------------------------------------------------------------
# AV4 -- the golden lives
# ---------------------------------------------------------------------------
GOLDEN = REPO / "tests" / "allium_golden" / "v1"
GOLDEN_LIVES = {"fixture": ("garden_north", "garden_evolve", "pinned", "windowsill_south"),
                "v0_1": ("garden", "windowsill")}
ENTRY_KEYS = ["at", "law_sha256", "overhead", "request", "response_sha256", "state_sha256", "work"]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fact_kinds(request, kind):
    return [fact for fact in request["facts"] if fact["kind"] == kind]


def _golden_shows(engine, law, name, request, answer, met):
    """What each golden life was written for, read off its request and its answer."""
    state = answer["state"]
    stage = state["organs"]["stage"]
    body = request["genesis"]["body"]
    trace = answer.get("trace")
    assert answer["provisional"] is True and state["law"]["name"] == law, (law, name)
    if (law, name) == ("fixture", "garden_north"):
        assert [fact["body"]["quarters"] * 15 for fact in _fact_kinds(request, "tz")] == [60, 345]
        assert state["tz"] == 345 and len(_fact_kinds(request, "act")) >= 3
        assert stage["dormant"] and stage["cause"] == "winter", stage
        assert trace["fast_path_days"] >= 1 and trace["draws"] >= 1, trace
        met["winter"] = met.get("winter", 0) + 1
    elif (law, name) == ("fixture", "windowsill_south"):
        assert body["weather"] == "windowsill" and body["hemisphere"] == "south"
        waters = [fact["t"] for fact in _fact_kinds(request, "act") if fact["body"]["act"] == "water"]
        assert not stage["dormant"] and stage["since"] > 0, stage
        assert any(t > stage["since"] for t in waters), "the water came after the entry into dormancy"
        # The minute before the water, the being sleeps from the drought, its rest over.
        water = min(t for t in waters if t > stage["since"])
        before = engine.ask(dict(request, facts=[f for f in request["facts"] if f["t"] < water], to=water - 1))
        asleep = before["state"]["organs"]["stage"]
        assert asleep["dormant"] and asleep["cause"] == "dry" and asleep["rest"] == 0, asleep
        assert trace["calls"]["chem"]["jump"] >= 1 and trace["calls"]["clock"]["jump"] >= 1, trace["calls"]
        assert trace["draws"] == 0 and trace["fast_path_days"] >= 1, trace
        met["drought and wake"] = met.get("drought and wake", 0) + 1
    elif (law, name) == ("fixture", "garden_evolve"):
        (evolve,) = _fact_kinds(request, "evolve")
        assert state["params"] == evolve["body"]["params"] != body["laws"]["params"], state["params"]
        assert state["pending"] is None and answer["notes"] == []
        met["params evolve"] = met.get("params evolve", 0) + 1
    elif (law, name) == ("fixture", "pinned"):
        kinds = [fact["kind"] for fact in request["facts"] if fact["kind"] != "act"]
        assert kinds == ["laws_pin", "evolve", "laws_unpin", "evolve"], kinds
        first, last = _fact_kinds(request, "evolve")
        assert answer["notes"] == [{"code": "evolve_pinned", "t": first["body"]["effective_from"]}], answer["notes"]
        assert state["params"] == last["body"]["params"] and not state["pinned"]
        met["pinned"] = met.get("pinned", 0) + 1
    elif (law, name) == ("v0_1", "garden"):
        assert body["weather"] == "garden" and "trace" not in answer
        met["full law"] = met.get("full law", 0) + 1
    else:
        assert body["weather"] == "windowsill" and body["band"] == "short" and "trace" not in answer
        met["full law"] = met.get("full law", 0) + 1


def test_av4_the_golden_lives_are_answered_alike_by_both_engines_and_the_author_agrees():
    loaded, restore = open_allium(native=True)
    met = {}
    try:
        engine = support.Engine(loaded)
        native = native_module(loaded)
        assert loaded["opti_oignon.allium.engine"].handshake(native), \
            "the native core answers for another world: run scripts/build_oo_core.sh"
        for law, names in GOLDEN_LIVES.items():
            golden = json.loads((GOLDEN / f"life_{law}.json").read_text(encoding="ascii"))
            assert sorted(golden) == sorted(names), (law, sorted(golden))
            digest = engine.law(law)["digest"]
            for name in names:
                entry = golden[name]
                assert sorted(entry) == ENTRY_KEYS, (law, name)
                assert entry["law_sha256"] == digest, ("a golden life is re-recorded with its law only", law, name)
                data = bytes.fromhex(entry["request"])
                request = engine.wire.parse(data)
                assert engine.wire.emit(request) == data
                assert request["op"] == "advance" and request["state"] is None and request["to"] == entry["at"]
                assert entry["at"] == 30 * DAY and request["genesis"]["body"]["laws"]["sha256"] == digest
                for label, answer_bytes in (("reference", engine.protocol.call(data)),
                                            ("native", bytes(native.allium_call(data)))):
                    assert _sha(answer_bytes) == entry["response_sha256"], (label, law, name)
                    answer = engine.wire.parse(answer_bytes)
                    assert answer["hash"] == entry["state_sha256"] == _sha(engine.wire.emit(answer["state"]))
                    assert (answer["done"], answer["at"], answer["work"], answer["overhead"], answer["alarm"]) == (
                        True, entry["at"], entry["work"], entry["overhead"], 0), (label, law, name)
                    met[label] = met.get(label, 0) + 1
                _golden_shows(engine, law, name, request, answer, met)

        # The author script recomputes every life with the reference and finds nothing to re-record.
        author = support.load_script("allium_author_life.py", "_av4_author")
        assert author.main(["--check"]) == 0
    finally:
        restore()
    assert met["reference"] == met["native"] == 6, met
    for what in ("winter", "drought and wake", "params evolve", "pinned"):
        assert met.get(what, 0) == 1, (what, met)
    assert met["full law"] == 2, met


# ---------------------------------------------------------------------------
# AV5 -- a change of law
# ---------------------------------------------------------------------------
_MIGRATE = r"""
import json, sys, time
from pathlib import Path
parity = int(sys.argv[3])
time.time = lambda: 1760000000.0 + parity
time.time_ns = lambda: 1760000000000000000 + parity
sys.path.insert(0, sys.argv[1])
from opti_oignon.allium import lawfiles
from opti_oignon.allium.ref import protocol
payload = json.loads(Path(sys.argv[2]).read_text(encoding="ascii"))
files = {name: bytes.fromhex(text) for name, text in payload["laws"].items()}
real = lawfiles.law_bytes
lawfiles.LAWS = lawfiles.LAWS + tuple(name for name in sorted(files) if name not in lawfiles.LAWS)
lawfiles.law_bytes = lambda name: files[name] if name in files else real(name)
print(protocol.call(bytes.fromhex(payload["request"])).hex())
"""


def test_av5_a_change_of_law_is_refused_alike_and_the_identity_migration_reads_no_clock(tmp_path):
    loaded, restore = open_allium(native=True)
    refused = 0
    try:
        engine = support.Engine(loaded)
        native = native_module(loaded)
        # Both engines refuse, byte for byte, an evolve between the two carried laws, in advance and in timeline.
        for law, target in (("fixture", "v0_1"), ("v0_1", "fixture")):
            being = support.Being(engine, suite="laws", index=50, law=law, weather="garden")
            e = _at_local(being, 2, 10 * 60 + 7, 0)
            info = engine.law(target)
            defaults = {k: spec["default"] for k, spec in engine.lawfiles.law(target)["params"].items()}
            being.evolve(e, defaults, effective_from=being.midnight(e, 0),
                         to=(target, info["digest"], info["version"]))
            timeline = {"facts": being.after(None, 3 * DAY), "from": None, "genesis": being.genesis,
                        "op": "timeline", "to": 3 * DAY, "v": 1}
            for request in (being.request(None, 3 * DAY), timeline):
                data = engine.wire.emit(request)
                mine = engine.protocol.call(data)
                assert bytes(native.allium_call(data)) == mine, (law, target, request["op"])
                assert engine.wire.parse(mine) == {"detail": "migration", "refused": "unknown_law"}, mine
                refused += 1

        # The identity migration, with a stable pair only the reference is given.
        files = support.stable_pair(engine)
        with support.injected(engine, files):
            second = engine.law("fixture_s2")

            def being_to(target):
                being = support.Being(engine, suite="laws", index=51, law="fixture_s", weather="garden")
                for day in (1, 3, 4):
                    being.act(_at_local(being, day, 8 * 60 + 11 * day, 0), "water")
                e = _at_local(being, 2, 15 * 60 + 29, 0)
                being.evolve(e, being.params(evap_awake=5000), effective_from=being.midnight(e, 0), to=target)
                return being, being.midnight(e, 0)

            moving, effective = being_to(("fixture_s2", second["digest"], second["version"]))
            staying, _same = being_to(None)
            before = [moving.advance(effective - 1)["state"], staying.advance(effective - 1)["state"]]
            for key in ("organs", "bus", "day", "params", "law", "seen"):
                assert before[0][key] == before[1][key], ("nothing moves before the minute", key)
            assert before[0]["law"]["name"] == "fixture_s" and before[0]["seen"] == [1]
            assert [state["pending"]["to"]["name"] for state in before] == ["fixture_s2", "fixture_s"]
            across = [moving.advance(effective)["state"], staying.advance(effective)["state"]]
            assert across[0]["law"] == {"name": "fixture_s2", "sha256": second["digest"], "v": 2}, across[0]["law"]
            assert across[1]["law"]["name"] == "fixture_s" and across[0]["seen"] == [1, 2] != across[1]["seen"]
            for key in ("organs", "bus", "day", "params", "genome"):
                assert across[0][key] == across[1][key], ("the identity migration moves no organ", key)
            request = moving.request(None, effective + 2 * DAY)
            data = engine.wire.emit(request)
            here = engine.protocol.call(data)
            answer = engine.wire.parse(here)
            assert "refused" not in answer and answer["state"]["law"]["name"] == "fixture_s2", answer
            payload = {"laws": {name: text.hex() for name, text in files.items()}, "request": data.hex()}

        # A successor that lowers the ceiling of a level the state keeps is no identity migration: the level
        # kept could lie above it, and the being would be refused on every call after the change.
        lowered_levels = 0
        for organ, key, value in (("soil", "m_max", 49152), ("stage", "rest_max", 500)):
            third = engine.wire.parse(files["fixture_s2"])
            third["name"] = "fixture_s3"
            third["version"] = 3
            third["constants"][organ][key] = value
            with support.injected(engine, dict(files, fixture_s3=engine.wire.emit(third))):
                info = engine.law("fixture_s3")
                assert "refused" not in info, ("the lowered law is sound on its own", organ, info)
                being = support.Being(engine, suite="laws", index=52, law="fixture_s", weather="garden")
                e = _at_local(being, 2, 15 * 60 + 29, 0)
                being.evolve(e, being.params(), effective_from=being.midnight(e, 0),
                             to=("fixture_s3", info["digest"], info["version"]))
                refusal = engine.ask(being.request(None, being.midnight(e, 0) + DAY))
                assert refusal == {"detail": "migration", "refused": "unknown_law"}, (organ, key, refusal)
                lowered_levels += 1
        assert lowered_levels == 2
        path = tmp_path / "stable_pair.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="ascii")
    finally:
        restore()

    outputs = []
    for hash_seed, zone, parity in (("1", "UTC0", "0"), ("2", "<+1245>-12:45", "1")):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, TZ=zone, PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run([sys.executable, "-c", _MIGRATE, str(REPO), str(path), parity], cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr[-2000:]
        outputs.append(run.stdout.strip())
    assert refused == 4
    assert outputs[0] == outputs[1] == here.hex(), "the migration answers the same under any seed, zone and clock"


# ---------------------------------------------------------------------------
# AV6 -- a pin keeps the old law
# ---------------------------------------------------------------------------
def test_av6_a_pin_keeps_the_law_and_params_in_force_until_it_is_lifted(engine):
    being = support.Being(engine, suite="laws", index=6, weather="garden")
    first = being.genesis["body"]["laws"]["params"]
    being.pin(_at_local(being, 3, 9 * 60, 0))
    e5 = _at_local(being, 5, 14 * 60 + 5, 0)
    being.evolve(e5, being.params(evap_awake=12000), effective_from=being.midnight(e5, 0))
    being.unpin(_at_local(being, 11, 9 * 60, 0))
    e12 = _at_local(being, 12, 10 * 60 + 12, 0)
    being.evolve(e12, being.params(evap_awake=2000, sun_max=40000), effective_from=being.midnight(e12, 0))
    stops = [_at_local(being, 4, 0, 0), e5 + 1, being.midnight(e5, 0), _at_local(being, 10, 12 * 60, 0),
             _at_local(being, 11, 12 * 60, 0), being.midnight(e12, 0) - 1, being.midnight(e12, 0) + DAY]
    answers = support.cuts(being, stops)
    pinned, registered, due, day10, unpinned, before, applied = answers
    assert pinned["state"]["pinned"] and registered["state"]["pending"] is not None
    assert due["notes"] == [{"code": "evolve_pinned", "t": being.midnight(e5, 0)}], due["notes"]
    assert due["state"]["pending"] is None and due["state"]["params"] == first
    assert day10["state"]["params"] == first and day10["state"]["law"]["v"] == being.v
    assert not unpinned["state"]["pinned"] and unpinned["state"]["params"] == first
    assert before["state"]["params"] == first and before["state"]["pending"] is not None
    assert applied["state"]["params"]["evap_awake"] == 2000 and applied["state"]["params"]["sun_max"] == 40000, \
        "witness: unpinned, an evolve applies"
    assert applied["notes"] == []


# ---------------------------------------------------------------------------
# AV9 -- the timeline is the reducer's
# ---------------------------------------------------------------------------
OFFSETS = (-600, -345, 0, 60, 345, 780, 840)
EVENTS = ("tz", "tz pair", "tz then evolve", "evolve then tz", "pin", "unpin", "evolve", "evolve wrong",
          "evolve params", "evolve moved", "pinned due", "pin at its minute", "east across midnight")


def _law_life(engine, index):
    """A windowsill life of sixty days with a seeded sequence of the law kinds; the events met, by name."""
    being = support.Being(engine, suite="laws", index=index, weather="windowsill")
    draw = engine.rng.Stream(bytes(32), "test.laws", index)
    offset = 0
    t = 300 + draw.below(DAY)
    met = {}
    order = list(EVENTS) * 3
    for i in range(len(order) - 1, 0, -1):
        j = draw.below(i + 1)
        order[i], order[j] = order[j], order[i]
    for event in order:
        t += 90 + draw.below(2 * DAY)
        if (being.b + t) % 15 == 0:
            t += 1
        new = OFFSETS[draw.below(len(OFFSETS))]
        params = being.params(evap_dormant=draw.below(8193), rain_gain=draw.below(131073))
        if event == "tz":
            being.tz(t, new)
            offset = new
        elif event == "tz pair":
            being.tz(t, OFFSETS[draw.below(len(OFFSETS))], origin=LOW)
            being.tz(t, new, origin=HIGH)
            offset = new
        elif event in ("tz then evolve", "evolve then tz"):
            first, second = (LOW, HIGH) if event == "tz then evolve" else (HIGH, LOW)
            being.tz(t, new, origin=first)
            offset = new
            being.evolve(t, params, effective_from=being.midnight(t, offset), origin=second)
        elif event == "pin":
            being.pin(t)
        elif event == "unpin":
            being.unpin(t)
        elif event == "evolve":
            being.evolve(t, params, effective_from=being.midnight(t, offset))
        elif event == "evolve wrong":
            being.evolve(t, params, effective_from=being.midnight(t, offset) + 15)
        elif event == "evolve params":
            being.evolve(t, dict(params, sun_max=1000), effective_from=being.midnight(t, offset))
        elif event == "evolve moved":
            being.evolve(t, params, effective_from=being.midnight(t, offset))
            t += 3 * 60 + 1
            offset = new if new != offset else OFFSETS[(OFFSETS.index(new) + 1) % len(OFFSETS)]
            being.tz(t, offset)
        elif event == "east across midnight":
            being.tz(t, -600)
            t = being.midnight(t, -600) + 23 * 60 + 1
            being.tz(t, 840)
            offset = 840
        elif event == "pinned due":
            being.pin(t)
            being.evolve(t + 1, params, effective_from=being.midnight(t + 1, offset))
            t = being.midnight(t + 1, offset) + 60
            being.unpin(t)
        else:
            being.unpin(t)
            being.evolve(t + 1, params, effective_from=being.midnight(t + 1, offset))
            t = being.midnight(t + 1, offset)
            being.pin(t)
            t += 1
            being.unpin(t)
        met[event] = met.get(event, 0) + 1
        if t > 58 * DAY:
            break
    return being, met


def _project(state):
    return {key: state[key] for key in ("at", "budget", "day", "law", "params", "pending", "pinned", "schema",
                                        "seen", "through", "tz")}


def test_av9_the_timeline_says_what_the_reducer_says(engine):
    end = 60 * DAY
    notes = {}
    for index in (90, 91):
        being, met = _law_life(engine, index)
        assert len(met) >= 10, met
        assert sorted(met) == sorted(EVENTS), ("every event of the sequence is met", sorted(set(EVENTS) - set(met)))
        draw = engine.rng.Stream(bytes(32), "test.laws", 100 + index)
        stops = sorted({1 + draw.below(end - 1) for _ in range(50)})
        answers = support.cuts(being, stops)
        previous = None
        for stop, answer in zip(stops, answers):
            state = answer["state"]
            fresh = being.timeline(stop)
            chained = being.timeline(stop, None if previous is None else _project(previous))
            for folded in (fresh, chained):
                for key in TIMELINE_KEYS:
                    assert folded["state"][key] == state[key], (index, stop, key, folded["state"][key], state[key])
                assert folded["state"]["at"] == stop
                assert folded["midnight"] == being.midnight(stop, state["tz"]), (index, stop)
            for note in answer["notes"]:
                notes[note["code"]] = notes.get(note["code"], 0) + 1
            previous = state

        # The firing minutes the timeline lists are exactly the minutes where the reducer's day rises.
        listed = being.timeline(end, midnights_after=0)
        firings = [m for m, _day, _civil in listed["midnights"]]
        assert len(firings) >= 50 and listed["work"] == len(firings) + len(
            [f for f in being.facts if f["t"] <= end]), len(firings)
        points = sorted({point for m in firings for point in (m - 1, m)} | {end})
        rising = support.cuts(being, points)
        day_at = {point: answer["state"]["day"] for point, answer in zip(points, rising)}
        for (m, day, civil), later in zip(listed["midnights"], firings[1:] + [end + 1]):
            assert day_at[m - 1] < day_at[m] == day, (index, m)
            assert civil == list(engine.civil.civil_from_days(day))
            assert day_at[later - 1] == day_at[m], ("no rise between two listed minutes", index, m, later)
        start = being.advance(0)["state"]["day"]
        assert day_at[firings[0] - 1] == start, "no rise before the first listed minute"
        assert listed["state"]["day"] == day_at[end]
        jumps = [m for m in firings if (being.b + m + rising[points.index(m)]["state"]["tz"]) % DAY != 0]
        assert jumps, ("witness: a firing at a tz fact's minute, not at a local midnight", index)
    for code in ("evolve_when", "evolve_params", "evolve_pinned", "evolve_superseded", "pin_twice", "unpin_unpinned"):
        assert notes.get(code, 0) >= 1, (code, notes)


# ---------------------------------------------------------------------------
# AV12 -- the life author refuses a re-record its law does not allow
# ---------------------------------------------------------------------------
def _md5s(folder):
    return {path.name: hashlib.md5(path.read_bytes()).hexdigest() for path in sorted(folder.iterdir())}


def test_av12_the_life_author_refuses_to_re_record_a_life_under_an_unchanged_law(tmp_path):
    loaded, restore = open_allium(native=False)
    try:
        author = support.load_script("allium_author_life.py", "_av12_author")
        fresh = author.author()
        # Every run below reads these entries, computed once, and the copies under tmp_path, never the tree's.
        author.author = lambda: fresh
        author.GOLDEN_DIR = tmp_path
        names = author.FILES

        def lay(edit=None):
            """The committed golden files, copied under tmp_path, with ``edit(entries)`` applied to the fixture's."""
            for law, name in names.items():
                entries = json.loads((GOLDEN / name).read_text(encoding="ascii"))
                if edit is not None and law == "fixture":
                    edit(entries)
                (tmp_path / name).write_text(author.render(entries), encoding="ascii")
            return _md5s(tmp_path)

        lay()
        assert author.main(["--check"]) == 0, "witness: the committed lives are current"
        runs = {}

        def answer_moved(entries):
            entries["pinned"]["response_sha256"] = "0" * 64

        def request_moved(entries):
            entries["garden_north"]["request"] = entries["garden_north"]["request"][:-2] + "00"

        def life_dropped(entries):
            entries["retired_life"] = dict(entries["pinned"])

        sentences = {answer_moved: "its answer changed under an unchanged law digest",
                     request_moved: "its request changed under an unchanged law",
                     life_dropped: "a golden life is dropped under an unchanged law"}
        for edit, sentence in sentences.items():
            before = lay(edit)
            assert author.main(["--check"]) == 1, (sentence, "the stale file is named")
            assert author.main(["--write"]) == 2, (sentence, "the re-record is refused")
            assert _md5s(tmp_path) == before, (sentence, "a refused re-record writes nothing")
            current = json.loads((tmp_path / names["fixture"]).read_text(encoding="ascii"))
            said = author.refusals("fixture", fresh["fixture"], current)
            assert len(said) == 1 and sentence in said[0], (sentence, said)
            runs[sentence] = True

        # Witness: under a moved law digest the same answer change is re-recorded, and the file is then current.
        def law_moved(entries):
            for entry in entries.values():
                entry["law_sha256"] = "1" * 64
            entries["pinned"]["response_sha256"] = "0" * 64

        before = lay(law_moved)
        assert author.main(["--write"]) == 0 and _md5s(tmp_path) != before, "a moved law re-records"
        assert (tmp_path / names["fixture"]).read_text(encoding="ascii") == author.render(fresh["fixture"])
        assert author.main(["--check"]) == 0
    finally:
        restore()
    assert len(runs) == 3, runs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
