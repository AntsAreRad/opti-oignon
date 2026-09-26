#!/usr/bin/env python3
"""Contracts for the componion's laws on the platform: the proposal, sowing, law updates and labels.

A being's law and params are frozen into its genesis at its birth, from the
proposal in the settings file; after that, only a law update moves them,
written by the laws writer as an ``evolve`` fact that takes effect at the
next local midnight. The platform learns what is in force from the engine's
law timeline alone.

  * AV1 -- the settings file does not move a being: sown under a proposal,
    grown, then opened by a fresh store under another proposal, the being
    shows the same hash and keeps the params its genesis froze; a being sown
    under the second proposal from the same draws, with the same facts, has
    that proposal's params and another state.
  * AV7 -- a view never writes an ``evolve``: sown under an injected stable
    law that a carried successor follows, three days of every look, each
    just after a local midnight, write nothing, and neither does another
    proposal alone; a gesture from a surface the law update's row does not
    hold writes none; the first gesture from the terminal writes exactly
    one, right after the gesture, carrying the params in force, from the law
    in force to the successor, at the next local midnight as the timeline
    gives it -- although the proposal had moved just before, which still
    waits for ``laws apply``. No look adds an audit entry or touches a file
    under the data directory either. Views before that minute show the
    first law, from it the successor, and the envelope of every fact
    follows; ``law_state`` says what ``advance`` says.
  * AV8 -- a prototype is labelled everywhere: the status, a view, the
    engine's own answer, ``law_state`` and a view catching up. A provisional
    law registered as retired and no longer carried reads
    ``retired_prototype`` with the label, never alive, ready or unreadable,
    and a resume is refused ``retired``; a law file edited in place reads
    ``unavailable`` / ``law`` (``retired_prototype`` once its old pair is
    registered); a genesis forged stable under the provisional fixture is
    refused by the engine (``bad_fact "genesis laws"``) and by the platform
    (``law``). A stable law's being carries no label, and a stable law in
    the register reads ``law``, not retired.
  * AV10 -- law writes on the platform: ``laws_apply`` is refused
    ``pinned``, ``params``, ``nothing`` and ``confirm`` by name and writes
    the ``evolve`` its diff confirmed; with a law update pending, it extends
    it to the same law. ``laws_pin`` twice is refused ``pinned`` and
    ``laws_unpin`` unpinned ``unpinned``. A generic append of ``evolve``,
    ``laws_pin`` or ``laws_unpin`` is refused ``producer`` from every surface
    of its row, and from any other ``surface``. With the injected stable
    pair, no automatic ``evolve`` is written while pinned, and one is at the
    first gesture after the unpin; a device that is not the being's home
    writes none.
  * AV11 -- the proposal and sowing: the proposal read from a settings file
    refuses by name, where it is used, a value of the wrong type, a boolean
    for an integer, an unknown symbol and one outside the law's range, and
    what cannot be read as a value; ``life`` falls back to its defaults and
    says so. A sowing the engine's dry run refuses (an injected defective
    law) leaves no file and raises ``LawsRefused("law")``; a wall clock
    before the second day of 1970 is refused ``clock``. Chain verification
    accepts a fact whose law version an ``evolve`` in the trunk names, and
    refuses one no ``evolve`` names, and an ``evolve`` naming a law this
    engine does not carry.
  * AV13 -- a law update takes effect at the next local midnight of the
    offset its own write records: a gesture from the terminal at 23:00
    local, as the offset moves to +2 h, writes ``tz``, the gesture and the
    automatic update in consecutive links, and the update takes the
    midnight 23 hours later, as the timeline gives it, with no note that it
    is off its minute; so does ``laws_apply`` confirmed at the old offset
    and written at the new one.
  * AV14 -- the automatic law update keeps an update the owner confirmed,
    and the day's budget: after ``laws_apply``, a gesture the same day
    writes the update to the successor with the confirmed params, which
    take effect with it; once ``laws_apply`` has spent the day's budget of
    law updates, a gesture that day writes none, and the first gesture of
    the next day writes it, with the params then in force.

Local-only. The platform loads through the shared isolation window with the
platform's configuration, keys, mode, audit log and user modules proven
unreachable; every seam is injected (``tests/_allium_store_support.py``),
the life's seams included. The injected laws live in the reference only, so
these contracts run the reference engine.
"""

import copy
import hashlib
import json
import logging
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_life_support as life_support  # noqa: E402
import _allium_store_support as support  # noqa: E402

BUDGET_S = {
    "test_av1_the_settings_file_does_not_move_a_being": 2.0,
    "test_av7_a_view_never_writes_an_evolve_and_the_first_gesture_writes_one": 2.0,
    "test_av8_a_prototype_is_labelled_everywhere_and_a_retired_one_is_said": 2.0,
    "test_av10_law_writes_are_refused_by_name_and_extend_a_pending_update": 2.0,
    "test_av11_the_proposal_and_sowing_are_refused_by_name_and_verification_knows_the_versions": 2.0,
    "test_av13_a_law_update_takes_effect_at_the_midnight_of_the_offset_its_own_write_records": 2.0,
    "test_av14_the_automatic_law_update_keeps_a_confirmed_update_and_skips_a_day_whose_budget_is_spent": 2.0,
}
DAY = 1440
B = support.WALL // 60
# The first local midnight after birth, at the offset every contract here keeps (UTC).
FIRST = DAY - B % DAY
ZERO = "0" * 64
OTHER = "f" * 16


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


def _proposal(sun_max=65536, evap_awake=4096, evap_dormant=1024, rain_gain=65536, band="long",
              hemisphere="north", weather="garden"):
    """A ``laws`` section of the settings file, as the store's seam takes it."""
    return {"light": {"sun_max": sun_max},
            "seasons": {"default_band": band, "default_hemisphere": hemisphere},
            "soil": {"evap_awake": evap_awake, "evap_dormant": evap_dormant, "rain_gain": rain_gain},
            "weather": {"mode": weather}}


def _params(section):
    return {"evap_awake": section["soil"]["evap_awake"], "evap_dormant": section["soil"]["evap_dormant"],
            "rain_gain": section["soil"]["rain_gain"], "sun_max": section["light"]["sun_max"]}


def _engine(p):
    """What the life support's law helpers read: the window's law files and codec."""
    return types.SimpleNamespace(lawfiles=p.lawfiles, wire=p.wire)


def _digest(p, data):
    return p.lawfiles.digest(p.wire.parse(data, lenient=True))


def _rows(path):
    """``(seq, t, kind, origin, oseq, laws, body)`` of every linked fact, in seq order."""
    rows = support.read(path, "SELECT l.seq, f.t, f.kind, f.origin, f.oseq, f.laws, b.body FROM links l "
                              "JOIN facts f ON f.eid = l.eid LEFT JOIN bodies b ON b.eid = l.eid ORDER BY l.seq")
    return [(seq, t, kind, origin, oseq, laws, None if body is None else json.loads(bytes(body)))
            for seq, t, kind, origin, oseq, laws, body in rows]


def _evolves(path):
    return [row for row in _rows(path) if row[2] == "evolve"]


def _snapshot(path):
    [(links,)] = support.read(path, "SELECT COUNT(*) FROM links")
    head = support.read(path, "SELECT seq, link FROM links ORDER BY seq DESC LIMIT 1")
    marks = support.read(path, "SELECT t, engine, laws, through, state_hash FROM checkpoints ORDER BY t")
    return links, head, marks, support.sha256_file(path)


def _surroundings(given):
    """What else a look could write: the audit log's entries (cross-anchors go there), and every file and
    directory under the data directory, with its size, modification time and bytes."""
    root = Path(given["data_dir"])
    listing = []
    for path in [root] + sorted(root.rglob("*")):
        stat = path.stat()
        if path.is_dir():
            listing.append((str(path.relative_to(root)), "dir", stat.st_mtime_ns))
        else:
            listing.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns, support.sha256_file(path)))
    return len(given["audit"].entries), listing


def _code(p, call):
    """What a call gave: ``None`` when it answered, else its refusal, named with its kind."""
    try:
        call()
    except p.membrane.MembraneRefused as refusal:
        return "membrane:" + refusal.code
    except p.store.StoreRefused as refusal:
        return "store:" + refusal.code
    except p.evolution.LawsRefused as refusal:
        return "laws:" + refusal.code
    except p.life.LifeRefused as refusal:
        return "life:" + refusal.code
    return None


def _refusal(p, call):
    """The refusal a call raised (a ``LawsRefused``, a ``MembraneRefused`` or a ``StoreRefused``)."""
    try:
        call()
    except (p.membrane.MembraneRefused, p.store.StoreRefused, p.evolution.LawsRefused) as refusal:
        return refusal
    raise AssertionError("the call answered")


def _at(clock, t):
    """Set the wall clock to minute ``t`` of life."""
    clock.wall = support.WALL + t * 60


def _status(p, given):
    target = support.store(p, given)
    try:
        return target.status("local")
    finally:
        target.close()


class _Recording:
    """The engine seam, keeping every ``advance`` answer, for the length of a ``with``."""

    def __init__(self, p):
        self._p = p
        self._real = p.engine.call
        self.answers = []

    def __enter__(self):
        self._p.engine.call = self._call
        return self

    def __exit__(self, *exc):
        self._p.engine.call = self._real

    def _call(self, request):
        answer = self._real(request)
        if self._p.wire.parse(request)["op"] == "advance":
            self.answers.append(self._p.wire.parse(answer))
        return answer


def _land(being, facts):
    """Land facts from another device in one transaction, each with the law version given."""

    def transaction(conn):
        meta = being._meta(conn)
        for origin, oseq, t, kind, body, laws in facts:
            head = being._head(conn)
            being._insert_fact(conn, origin=origin, oseq=oseq, t=t, kind=kind, body=body, head=head, laws=laws)
            name = "oseq_next:" + origin
            meta[name] = max(meta.get(name, 0), oseq + 1)
            being._put(conn, name, meta[name])
        gen = meta["gen"] + 1
        being._put(conn, "gen", gen)
        being._rewrite_anchor(conn, gen)

    being._write(transaction)


# ---------------------------------------------------------------------------
# AV1 -- the settings file does not move a being
# ---------------------------------------------------------------------------
P1 = _proposal(sun_max=40000, evap_awake=6000, evap_dormant=2048, rain_gain=50000)
P2 = _proposal(sun_max=30000, evap_awake=9000, evap_dormant=512, rain_gain=90000)


def _grow(p, being, clock):
    cli = support.cli(p)
    for act in ("water", "greet", "water", "warm"):
        clock.advance_days(1)
        being.append("act", {"act": act}, transport=cli)
    clock.advance_days(2)


def test_av1_the_settings_file_does_not_move_a_being(p, tmp_path):
    one = support.seams(p, tmp_path.joinpath("p1"), suite="av1", laws=copy.deepcopy(P1))
    target = support.store(p, one)
    try:
        being = support.sow(p, target)
        genesis = being.in_order()[0][2]
        assert genesis["body"]["laws"]["params"] == _params(P1), "the proposal is frozen into the genesis"
        _grow(p, being, one["clock"])
        first = being.view()
        facts = [(fact["t"], fact["kind"], fact["body"]) for _seq, _eid, fact in being.in_order()[1:]]
    finally:
        target.close()
    path = support.store_path(p, one)
    kept = support.sha256_file(path)

    # Another proposal: a fresh store shows the same being, and says what a law update would change.
    one["laws"] = copy.deepcopy(P2)
    fresh = support.store(p, one)
    try:
        again = fresh.open("local")
        second = again.view()
        assert (second.status, second.at, second.hash) == ("current", first.at, first.hash), second
        assert second.state["params"] == _params(P1)
        assert again.in_order()[0][2] == genesis
        diff = again.laws_diff()
        assert (diff.current, diff.proposed) == (_params(P1), _params(P2)), diff
        assert diff.changed == sorted(_params(P1)), diff
    finally:
        fresh.close()
    assert support.sha256_file(path) == kept, "the proposal read wrote nothing"

    # Witness: sown under the second proposal from the same draws, with the same facts.
    two = support.seams(p, tmp_path.joinpath("p2"), suite="av1", laws=copy.deepcopy(P2))
    other = support.store(p, two)
    try:
        twin = support.sow(p, other)
        assert twin.being == being.being, "presence: the same draws"
        _grow(p, twin, two["clock"])
        assert [(fact["t"], fact["kind"], fact["body"]) for _seq, _eid, fact in twin.in_order()[1:]] == facts
        view = twin.view()
        assert twin.in_order()[0][2]["body"]["laws"]["params"] == _params(P2)
        assert view.at == first.at and view.hash != first.hash, "the params move the state"
    finally:
        other.close()
    assert two["entropy"].chunks == one["entropy"].chunks


# ---------------------------------------------------------------------------
# AV7 -- a view never writes an evolve; the first gesture from the terminal writes one
# ---------------------------------------------------------------------------
def test_av7_a_view_never_writes_an_evolve_and_the_first_gesture_writes_one(p, tmp_path):
    engine = _engine(p)
    files = life_support.stable_pair(engine)
    first_law = {"name": "fixture_s", "sha256": _digest(p, files["fixture_s"])}
    successor = {"name": "fixture_s2", "sha256": _digest(p, files["fixture_s2"]), "v": 2}
    cli = support.cli(p)
    with life_support.injected(engine, files):
        given = support.seams(p, tmp_path, suite="av7")
        clock = given["clock"]
        target = support.store(p, given)
        try:
            being = support.sow(p, target, law="fixture_s")
            path = support.store_path(p, given)
            params = being.in_order()[0][2]["body"]["laws"]["params"]

            # Three days of every look, each just after a local midnight and at noon: nothing is written.
            looks = {
                "status": lambda: target.status("local"),
                "open": lambda: target.open("local"),
                "view": lambda: being.view(),
                "view at a minute": lambda: being.view(to=FIRST),
                "capped view": lambda: being.view(cap=1),
                "laws_diff": lambda: being.laws_diff(),
                "law_state": lambda: being.law_state(),
                "in_order": lambda: being.in_order(),
                "trunk": lambda: list(being.trunk()),
                "head": lambda: being.head(),
                "verify": lambda: being.verify(),
            }
            before = _snapshot(path)
            around = _surroundings(given)
            assert around[0] >= 1 and len(around[1]) >= 3, "presence: the audit log and the data directory hold rows"
            for day in range(3):
                for minute in (FIRST + day * DAY + 1, FIRST + day * DAY + DAY // 2):
                    _at(clock, minute)
                    for name, look in looks.items():
                        look()
                        assert _snapshot(path) == before, (day, minute, name)
                        assert _surroundings(given) == around, ("nor the audit log or any file", day, minute, name)
            assert being.view().state["law"]["name"] == "fixture_s"

            # Another proposal alone writes nothing either.
            moved = dict(given, laws=_proposal(evap_awake=8192))
            other = support.store(p, moved)
            try:
                seen = other.open("local")
                assert seen.laws_diff().changed == ["evap_awake"]
                seen.view()
                seen.law_state()
            finally:
                other.close()
            assert _snapshot(path) == before and _evolves(path) == []

            # A gesture from a surface the law update's row does not hold writes none.
            local = p.membrane.Transport("web", principal={"role": "admin", "sub": "local", "type": "access"})
            assert p.membrane.surface_of(local, clock.wall) == "web_local"
            assert isinstance(being.append("act", {"act": "touch"}, transport=local), p.membrane.Appended)
            assert _evolves(path) == []

            # The proposal moved before the first gesture from the terminal: the law update the gesture carries
            # keeps the params in force, and the proposal still waits for laws apply.
            given["laws"]["soil"]["evap_awake"] = 8192

            # The first gesture from the terminal writes exactly one, right after itself.
            rows = _rows(path)
            outcome = being.append("act", {"act": "greet"}, transport=cli)
            added = _rows(path)[len(rows):]
            assert [(row[2], row[1]) for row in added] == [("act", outcome.t), ("evolve", outcome.t)], added
            assert added[1][0] == added[0][0] + 1 and added[1][4] == added[0][4] + 1, "consecutive"
            body = added[1][6]
            midnight = p.evolution.timeline_at(being, outcome.t).midnight
            assert body == {"effective_from": midnight, "from": first_law, "params": params, "to": successor}, body
            assert midnight == FIRST + 3 * DAY and added[1][5] == 1, "the next local midnight, under the law in force"
            assert params["evap_awake"] != 8192 and being.laws_diff().changed == ["evap_awake"], "the proposal waits"
            assert _surroundings(given) != around, "witness: a gesture's writes show"
            again = being.append("act", {"act": "play"}, transport=cli)
            assert len(_evolves(path)) == 1, "a law update already pending is not written twice"
            assert again.t < midnight

            # Before the minute the first law, from it the successor; the envelope follows.
            state = being.law_state(to=midnight - 1)
            assert state.pending["to"] == successor and state.law["name"] == "fixture_s"
            assert being.view(to=midnight - 1).state["law"]["name"] == "fixture_s"
            after = being.view(to=midnight)
            assert after.state["law"] == successor and after.state["params"] == params, after.state["law"]
            _at(clock, midnight + 10)
            late = being.append("act", {"act": "water"}, transport=cli)
            assert [row[5] for row in _rows(path) if row[0] >= 1] == [1] * (len(_rows(path)) - 2) + [2], \
                "every fact before the minute carries the first law, the one after it the successor"
            assert late.t == midnight + 10 and len(_evolves(path)) == 1
            # Another device's fact under the successor, then the same process looks again, verifying only what
            # it has not read yet: it knows the version from the evolve it wrote itself.
            _land(being, [(OTHER, 0, midnight + 20, "act", {"act": "greet"}, 2)])
            status = target.status("local")
            assert (status.status, status.labels) == ("alive", ()), status
            assert _status(p, given).status == "alive", "and a fresh process, verifying from the genesis"

            # law_state says what advance says.
            for to in (midnight - 1, midnight, None):
                state = being.law_state(to=to)
                view = being.view(to=to)
                assert state.at == view.at
                for field in ("day", "law", "params", "pending", "pinned", "seen", "tz"):
                    assert getattr(state, field) == view.state[field], (to, field)
            assert being.law_state().law == successor and being.law_state().seen == [1, 2]
        finally:
            target.close()


# ---------------------------------------------------------------------------
# AV8 -- a prototype is labelled everywhere; a retired one is said
# ---------------------------------------------------------------------------
def _forge_genesis(p, path, change):
    """Rewrite the genesis of a store that holds only its genesis: its body, digest, eid, link and anchor."""

    def edit(conn):
        [(eid, being, origin, laws, body)] = conn.execute(
            "SELECT f.eid, f.being, f.origin, f.laws, b.body FROM facts f JOIN bodies b ON b.eid = f.eid").fetchall()
        genesis = p.wire.parse(bytes(body))
        change(genesis)
        fact = {"being": being, "body": genesis, "kind": "genesis", "laws": laws, "origin": origin, "oseq": 0,
                "t": 0}
        answer = p.wire.parse(p.engine.call(p.wire.emit({"fact": fact, "op": "fact_id", "v": 1})))
        forged = answer["eid"]
        conn.execute("UPDATE facts SET eid = ?, body_sha256 = ? WHERE eid = ?", (forged, answer["body"], eid))
        conn.execute("UPDATE bodies SET eid = ?, body = ? WHERE eid = ?", (forged, p.wire.emit(genesis), eid))
        conn.execute("UPDATE links SET eid = ?, link = ? WHERE seq = 0", (forged, p.chain.link(forged, ZERO, 0)))
        return fact

    fact = support.edit(path, edit)
    support.rewrite_anchor(p, path, seq=0, key_id=support.KEY_ID, anchor_key=support.KEY)
    return fact


def _register(p, register):
    """Give the window's law files back the register reader they had (or none)."""
    if register is None:
        if hasattr(p.lawfiles, "retired"):
            del p.lawfiles.retired
    else:
        p.lawfiles.retired = register


def test_av8_a_prototype_is_labelled_everywhere_and_a_retired_one_is_said(p, tmp_path):
    cli = support.cli(p)
    given = support.seams(p, tmp_path.joinpath("pot"), suite="av8")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = support.sow(p, target)
        clock.advance_days(2)
        being.append("act", {"act": "water"}, transport=cli)
        clock.advance_days(1)
        status = target.status("local")
        assert (status.status, status.labels) == ("alive", ("prototype",)), status
        with _Recording(p) as recording:
            view = being.view()
        assert (view.labels, view.provisional) == (("prototype",), True), view
        assert recording.answers and all(answer["provisional"] is True for answer in recording.answers)
        assert being.settle().done
        clock.advance_days(1)
        capped = being.view(cap=1)
        assert (capped.status, capped.labels, capped.provisional) == ("catching_up", ("prototype",), True)
    finally:
        target.close()
    path = support.store_path(p, given)
    sown = support.sha256_file(path)
    fixture = {"name": "fixture", "sha256": p.lawfiles.digest(p.lawfiles.law("fixture"))}

    # The law retired: its pair registered, its file no longer carried.
    carried, register, law_bytes = p.lawfiles.LAWS, getattr(p.lawfiles, "retired", None), p.lawfiles.law_bytes
    try:
        p.lawfiles.LAWS = tuple(name for name in carried if name != "fixture")
        p.lawfiles.retired = lambda: [dict(fixture)]
        status = _status(p, given)
        assert (status.status, status.labels, status.reason) == ("retired_prototype", ("prototype",), "retired")
        again = support.store(p, given)
        try:
            assert _code(p, lambda: again.open("local")) == "store:retired"
            assert _code(p, lambda: again.resume(transport=cli, confirm=(0, 0))) == "store:retired"
        finally:
            again.close()
        # Not registered: the same missing law is ``law``.
        p.lawfiles.retired = lambda: []
        assert (_status(p, given).status, _status(p, given).reason) == ("unavailable", "law")

        # The law file edited in place: its digest moved, so the being's law is not the one carried.
        p.lawfiles.LAWS = carried
        edited = p.lawfiles.law("fixture")
        edited["constants"]["soil"]["dose"] += 1
        p.lawfiles.law_bytes = lambda name: p.wire.emit(edited) if name == "fixture" else law_bytes(name)
        status = _status(p, given)
        assert (status.status, status.reason) == ("unavailable", "law"), status
        p.lawfiles.retired = lambda: [dict(fixture)]
        status = _status(p, given)
        assert (status.status, status.labels, status.reason) == ("retired_prototype", ("prototype",), "retired")
    finally:
        p.lawfiles.LAWS, p.lawfiles.law_bytes = carried, law_bytes
        _register(p, register)
    assert _status(p, given).status == "alive", "witness: the law carried again, the being opens"
    assert support.sha256_file(path) == sown, "looking wrote nothing"

    # A genesis forged stable under the provisional fixture: the engine and the platform refuse it.
    forged_seams = support.seams(p, tmp_path.joinpath("forged"), suite="av8", index=1)
    forger = support.store(p, forged_seams)
    try:
        support.sow(p, forger)
    finally:
        forger.close()
    forged = _forge_genesis(p, support.store_path(p, forged_seams),
                            lambda genesis: genesis["laws"].__setitem__("provisional", False))
    request = {"budget": 100, "facts": [], "genesis": forged, "op": "advance", "state": None, "to": 0, "v": 1}
    answer = p.wire.parse(p.engine.call(p.wire.emit(request)))
    assert (answer.get("refused"), answer.get("detail")) == ("bad_fact", "genesis laws"), answer
    status = _status(p, forged_seams)
    assert (status.status, status.reason) == ("unavailable", "law"), status

    # Witness: a stable law's being carries no label, and a stable law in the register is not retired.
    engine = _engine(p)
    files = life_support.stable_pair(engine)
    with life_support.injected(engine, files):
        stable = support.seams(p, tmp_path.joinpath("stable"), suite="av8", index=2)
        keeper = support.store(p, stable)
        try:
            kept = support.sow(p, keeper, law="fixture_s")
            stable["clock"].advance_days(1)
            status = keeper.status("local")
            assert (status.status, status.labels) == ("alive", ()), status
            view = kept.view()
            assert (view.labels, view.provisional) == ((), False)
            state = kept.law_state()
            assert (state.labels, state.provisional) == ((), False)
        finally:
            keeper.close()
        injected = p.lawfiles.LAWS
        try:
            p.lawfiles.LAWS = tuple(name for name in injected if name != "fixture_s")
            p.lawfiles.retired = lambda: [{"name": "fixture_s", "sha256": _digest(p, files["fixture_s"])}]
            status = _status(p, stable)
            assert (status.status, status.labels, status.reason) == ("unavailable", (), "law"), status
        finally:
            p.lawfiles.LAWS = injected
            _register(p, register)

    # law_state carries the label too.
    target = support.store(p, given)
    try:
        state = target.open("local").law_state()
        assert (state.labels, state.provisional) == (("prototype",), True), state
    finally:
        target.close()


# ---------------------------------------------------------------------------
# AV10 -- law writes on the platform
# ---------------------------------------------------------------------------
def test_av10_law_writes_are_refused_by_name_and_extend_a_pending_update(p, tmp_path):
    M = p.membrane
    cli = support.cli(p)
    given = support.seams(p, tmp_path.joinpath("pot"), suite="av10")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = support.sow(p, target)
        path = support.store_path(p, given)
        clock.advance_days(1)
        now = clock.wall
        law = {"name": "fixture", "sha256": p.lawfiles.digest(p.lawfiles.law("fixture"))}
        defaults = being.in_order()[0][2]["body"]["laws"]["params"]

        # A generic append of the laws writer's kinds is refused ``producer`` from every surface of its row.
        transports = {"cli_tty": cli, "web_session": support.web(p, "local", now),
                      "light_hook": M.Transport("light_hook"), "idle_timer": M.Transport("idle_timer"),
                      "web_local": M.Transport("web", principal={"role": "admin", "sub": "local",
                                                                 "type": "access"})}
        evolve = {"effective_from": FIRST + DAY, "from": law, "params": defaults, "to": dict(law, v=0)}
        bodies = {"evolve": evolve, "laws_pin": {}, "laws_unpin": {}}
        met = 0
        rows = _rows(path)
        for kind, body in bodies.items():
            for surface, transport in transports.items():
                expected = "membrane:producer" if surface in M.MATRIX[kind] else "membrane:surface"
                got = _code(p, lambda kind=kind, body=body, transport=transport: being.append(
                    kind, body, transport=transport))
                assert got == expected, (kind, surface, got)
                met += expected == "membrane:producer"
        assert met == 4 + 2 + 2 and _rows(path) == rows, "every surface of each row was met, and nothing written"

        # laws_apply: nothing to change under the law's own defaults.
        diff = being.laws_diff()
        assert (diff.current, diff.proposed, diff.changed) == (defaults, defaults, []), diff
        assert _code(p, lambda: being.laws_apply(cli, diff.confirm)) == "laws:nothing"
        # A proposal outside the law's range: ``params``, by name, before any confirmation is read.
        given["laws"]["soil"]["evap_awake"] = 99999
        refusal = _refusal(p, lambda: being.laws_apply(cli, "0" * 16))
        assert refusal.code == "params" and "soil.evap_awake 99999" in refusal.detail, refusal
        # Another proposal: the confirmation must be the diff's own.
        given["laws"]["soil"]["evap_awake"] = 8192
        diff = being.laws_diff()
        assert diff.changed == ["evap_awake"] and diff.law == dict(law, v=0), diff
        assert diff.confirm == hashlib.sha256(p.wire.emit({"from": diff.current, "law": law,
                                                           "to": diff.proposed})).hexdigest()[:16]
        assert _code(p, lambda: being.laws_apply(cli, "0" * 16)) == "laws:confirm"
        assert _code(p, lambda: being.laws_apply(transports["web_local"], diff.confirm)) == "membrane:surface"
        assert _rows(path) == rows, "no refusal wrote anything"
        applied = being.laws_apply(cli, diff.confirm)
        assert isinstance(applied, M.Appended), applied
        [row] = _evolves(path)
        assert row[6] == {"effective_from": FIRST + DAY, "from": law, "params": dict(defaults, evap_awake=8192),
                          "to": dict(law, v=0)}, row
        assert being.laws_diff().current == dict(defaults, evap_awake=8192), "the pending params are current"
        assert _code(p, lambda: being.laws_apply(cli, diff.confirm)) == "laws:nothing"

        # Pins: twice is ``pinned``, unpinned is ``unpinned``; a pinned being refuses laws_apply.
        assert isinstance(being.laws_pin(cli), M.Appended)
        assert _code(p, lambda: being.laws_pin(cli)) == "laws:pinned"
        given["laws"]["soil"]["evap_awake"] = 12000
        assert _code(p, lambda: being.laws_apply(cli, being.laws_diff().confirm)) == "laws:pinned"
        assert isinstance(being.laws_unpin(cli), M.Appended)
        assert _code(p, lambda: being.laws_unpin(cli)) == "laws:unpinned"
        assert _code(p, lambda: being.laws_pin(transports["web_local"])) == "membrane:surface"
        assert [row[2] for row in _rows(path)][-2:] == ["laws_pin", "laws_unpin"]
    finally:
        target.close()

    # With the injected stable pair: nothing while pinned, one at the first gesture after the unpin, and
    # laws_apply extends the pending update to the same law.
    engine = _engine(p)
    files = life_support.stable_pair(engine)
    first_law = {"name": "fixture_s", "sha256": _digest(p, files["fixture_s"])}
    successor = {"name": "fixture_s2", "sha256": _digest(p, files["fixture_s2"]), "v": 2}
    with life_support.injected(engine, files):
        stable = support.seams(p, tmp_path.joinpath("stable"), suite="av10", index=1)
        clock = stable["clock"]
        keeper = support.store(p, stable)
        try:
            being = support.sow(p, keeper, law="fixture_s")
            path = support.store_path(p, stable)
            params = being.in_order()[0][2]["body"]["laws"]["params"]
            clock.advance_days(1)
            assert isinstance(being.laws_pin(cli), M.Appended)
            being.append("act", {"act": "greet"}, transport=cli)
            clock.advance_days(1)
            being.append("act", {"act": "water"}, transport=cli)
            assert _evolves(path) == [], "no law update while pinned"
            assert isinstance(being.laws_unpin(cli), M.Appended)
            assert _evolves(path) == [], "the unpin is not a gesture"
            gesture = being.append("act", {"act": "warm"}, transport=cli)
            [row] = _evolves(path)
            assert (row[1], row[6]["to"], row[6]["params"]) == (gesture.t, successor, params), row

            stable["laws"]["soil"]["evap_awake"] = 8192
            diff = being.laws_diff()
            assert (diff.law, diff.current, diff.changed) == (successor, params, ["evap_awake"]), diff
            assert isinstance(being.laws_apply(cli, diff.confirm), M.Appended)
            last = _evolves(path)[-1]
            assert last[6]["from"] == first_law and last[6]["to"] == successor, "the pending law is kept"
            assert last[6]["params"] == dict(params, evap_awake=8192)
            shown = being.view(to=last[6]["effective_from"])
            assert shown.state["law"] == successor and shown.state["params"] == dict(params, evap_awake=8192)
        finally:
            keeper.close()

        # Only the being's home writes a law update: the same store under another device's origin writes none.
        away = support.seams(p, tmp_path.joinpath("away"), suite="av10", index=2)
        keeper = support.store(p, away)
        try:
            support.sow(p, keeper, law="fixture_s")
        finally:
            keeper.close()
        path = support.store_path(p, away)
        device = "e" * 16

        def move(conn):
            conn.execute("UPDATE meta SET value = ? WHERE key = 'origin'", (p.wire.emit(device),))
            conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("oseq_next:" + device, p.wire.emit(0)))

        support.edit(path, move)
        keeper = support.store(p, away)
        try:
            being = keeper.open("local")
            away["clock"].advance_days(1)
            assert isinstance(being.append("act", {"act": "greet"}, transport=cli), M.Appended)
            assert _rows(path)[-1][2:5] == ("act", device, 0), "presence: written as the other device"
            assert _evolves(path) == [], "a device that is not the being's home writes no law update"
        finally:
            keeper.close()


# ---------------------------------------------------------------------------
# AV11 -- the proposal and sowing; verification knows the law versions
# ---------------------------------------------------------------------------
_MALFORMED = (
    # (settings file text, where the refusal comes, its code, what it names)
    ("laws:\n  soil: {evap_awake: many}\n", "proposal", "params", "soil.evap_awake"),
    ("laws:\n  soil: {evap_dormant: true}\n", "proposal", "params", "soil.evap_dormant"),
    ("laws:\n  soil: {evap_awake: 99999}\n", "proposal", "params", "soil.evap_awake 99999"),
    ("laws:\n  light: {sun_max: 1000}\n", "proposal", "params", "light.sun_max 1000"),
    ("laws:\n  soil: {rain_gain: 1.5}\n", "proposal", "params", "soil.rain_gain"),
    ("laws:\n  seasons: {default_hemisphere: east}\n", "sowing", "sowing", "seasons.default_hemisphere"),
    ("laws:\n  seasons: {default_band: 3}\n", "sowing", "sowing", "seasons.default_band"),
    ("laws:\n  weather: {mode: balcony}\n", "sowing", "sowing", "weather.mode"),
    ("laws:\n  soil: 5\n", "proposal", "params", "soil is not a mapping"),
    ("laws:\n  soil: {evap_awak: 4096}\n", "proposal", "params", "soil.evap_awak"),
    ("laws:\n  lights: {sun_max: 65536}\n", "sowing", "sowing", "lights"),
    ("laws: [1, 2]\n", "proposal", "params", "laws is not a mapping"),
    ("laws:\n  soil: {evap_awake: [\n", "sowing", "sowing", "cannot be read"),
)


def test_av11_the_proposal_and_sowing_are_refused_by_name_and_verification_knows_the_versions(p, tmp_path,
                                                                                            caplog):
    evolution = p.evolution
    pin = p.membrane.law_pin("fixture")

    # A wall clock before the second day of 1970 cannot be a birth.
    early = support.seams(p, tmp_path.joinpath("early"), suite="av11", clock=support.Clock(86399))
    target = support.store(p, early)
    try:
        assert _code(p, lambda: support.sow(p, target)) == "membrane:clock"
        assert not support.directory(early).joinpath(p.anchors.owner_tag("local") + ".db").exists()
        early["clock"].wall = 86400
        assert support.sow(p, target).in_order()[0][2]["body"]["birth"]["wall"] == 86400, "witness: the first"
    finally:
        target.close()

    # Chain verification: a law version an evolve in the trunk names is accepted, another is refused.
    engine = _engine(p)
    files = life_support.stable_pair(engine)
    first_law = {"name": "fixture_s", "sha256": _digest(p, files["fixture_s"])}
    successor = {"name": "fixture_s2", "sha256": _digest(p, files["fixture_s2"]), "v": 2}
    with life_support.injected(engine, files):
        stable = support.seams(p, tmp_path.joinpath("stable"), suite="av11", index=1)
        keeper = support.store(p, stable)
        try:
            being = support.sow(p, keeper, law="fixture_s")
            params = being.in_order()[0][2]["body"]["laws"]["params"]
            evolve = {"effective_from": FIRST, "from": first_law, "params": params, "to": successor}
            _land(being, [(OTHER, 0, 10, "evolve", evolve, 1), (OTHER, 1, FIRST + 5, "act", {"act": "greet"}, 2)])
        finally:
            keeper.close()
        status = _status(p, stable)
        assert status.status == "alive", ("a version an evolve names is accepted", status)
        keeper = support.store(p, stable)
        try:
            _land(keeper.open("local"), [(OTHER, 2, FIRST + 6, "act", {"act": "warm"}, 3)])
        finally:
            keeper.close()
        status = _status(p, stable)
        assert (status.status, status.reason) == ("unreadable", "laws"), status

        # An evolve naming a law this engine does not carry leaves the being unavailable.
        named = support.seams(p, tmp_path.joinpath("named"), suite="av11", index=2)
        keeper = support.store(p, named)
        try:
            being = support.sow(p, keeper, law="fixture_s")
            stranger = dict(successor, sha256="ab" * 32)
            _land(being, [(OTHER, 0, 10, "evolve", dict(evolve, to=stranger), 1)])
        finally:
            keeper.close()
        status = _status(p, named)
        assert (status.status, status.reason) == ("unavailable", "law"), status

    # The proposal read from a settings file refuses each malformed value by name, where it is used.
    shipped = p.settings.laws()
    assert shipped["params"] == {"evap_awake": 4096, "evap_dormant": 1024, "rain_gain": 65536, "sun_max": 65536}
    assert shipped["sowing"] == {"band": "long", "hemisphere": "north", "weather": "garden"}, "the file is read"
    assert evolution.proposal(pin, shipped) == {name: spec["default"] for name, spec in pin["params"].items()}
    assert evolution.sowing(pin, shipped) == {"band": "long", "hemisphere": "north", "weather": "garden"}
    absent = p.settings.laws(tmp_path.joinpath("absent.yaml"))
    assert evolution.proposal(pin, absent) == evolution.proposal(pin, shipped), "a missing file proposes nothing"
    for index, (text, where, code, named_part) in enumerate(_MALFORMED):
        file = tmp_path.joinpath(f"case{index}.yaml")
        file.write_text(text, encoding="ascii")
        raw = p.settings.laws(file)
        use = (lambda raw=raw: evolution.proposal(pin, raw)) if where == "proposal" else \
            (lambda raw=raw: evolution.sowing(pin, raw))
        refusal = _refusal(p, use)
        assert (type(refusal).__name__, refusal.code) == ("LawsRefused", code), (text, refusal)
        assert named_part in refusal.detail, (text, refusal.detail)
    good = tmp_path.joinpath("good.yaml")
    good.write_text("laws:\n  soil: {evap_awake: 16384}\n  seasons: {default_hemisphere: south}\n",
                    encoding="ascii")
    raw = p.settings.laws(good)
    assert evolution.proposal(pin, raw)["evap_awake"] == 16384, "witness: the top of the range is taken"
    assert evolution.sowing(pin, raw)["hemisphere"] == "south"
    assert evolution.sowing(pin, raw, hemisphere="north")["hemisphere"] == "north", "the caller's own word wins"
    assert _refusal(p, lambda: evolution.sowing(pin, raw, weather="porch")).code == "sowing"

    # Sowing reads the proposal through the store's seam: a malformed one is refused before any draw.
    refused = support.seams(p, tmp_path.joinpath("refused"), suite="av11", index=3,
                            laws=_proposal(evap_awake=True))
    target = support.store(p, refused)
    try:
        assert _code(p, lambda: support.sow(p, target)) == "laws:params"
        # The store holds the seam's own mapping: it is changed in place, as a settings file is edited.
        refused["laws"].update(_proposal(band="huge"))
        assert _code(p, lambda: support.sow(p, target)) == "laws:sowing"
        assert refused["entropy"].calls == 0 and not support.directory(refused).exists()
        refused["laws"].update(_proposal(band="short", hemisphere="south", weather="windowsill", evap_dormant=0))
        genesis = support.sow(p, target).in_order()[0][2]["body"]
        assert (genesis["band"], genesis["hemisphere"], genesis["weather"]) == ("short", "south", "windowsill")
        assert genesis["laws"]["params"]["evap_dormant"] == 0
    finally:
        target.close()

    # A sowing the engine's dry run refuses leaves no file, whatever it drew, and names the law.
    faulty = life_support.defective(engine, lambda law: law.__setitem__("code", "seed_9"))
    with life_support.injected(engine, faulty):
        dry = support.seams(p, tmp_path.joinpath("dry"), suite="av11", index=4)
        target = support.store(p, dry)
        try:
            refusal = _refusal(p, lambda: support.sow(p, target, law="fixture_x"))
            assert (type(refusal).__name__, refusal.code) == ("LawsRefused", "law"), refusal
            assert "unknown_law" in refusal.detail and "code" in refusal.detail, refusal.detail
            assert dry["entropy"].calls >= 1, "presence: the draws were made, and are discarded"
            assert sorted(path.name for path in support.directory(dry).iterdir()) == [".sowing.lock"], \
                "no store and no sowing file is left"
            assert target.status("local").status == "ready"
        finally:
            target.close()

    # The machine policy falls back to its defaults, and says so.
    defaults = {"checkpoints": {"daily": 14, "monthly": True, "weekly": 52}, "skew_note_min": 5}
    name = p.settings.logger.name
    for section, said in ((7, "life"), ({"skew_note_min": "5"}, "skew_note_min"),
                          ({"skew_note_min": True}, "skew_note_min"), ({"skew_note_min": -1}, "skew_note_min"),
                          ({"checkpoints": [14]}, "checkpoints"), ({"checkpoints": {"daily": 1.5}}, "daily"),
                          ({"checkpoints": {"weekly": False}}, "weekly"),
                          ({"checkpoints": {"monthly": "yes"}}, "monthly")):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=name):
            assert p.settings.normalise_life(section) == defaults, section
        assert [r for r in caplog.records if said in r.getMessage()], ("the fallback is said", section)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=name):
        chosen = {"checkpoints": {"daily": 1, "monthly": False, "weekly": 0}, "skew_note_min": 30}
        assert p.settings.normalise_life(chosen) == chosen
        assert p.settings.life() == defaults, "the shipped file"
    assert not caplog.records, "witness: values in range are taken, and nothing is said"


# ---------------------------------------------------------------------------
# AV13 -- a law update's minute follows the offset its own write records
# ---------------------------------------------------------------------------
def test_av13_a_law_update_takes_effect_at_the_midnight_of_the_offset_its_own_write_records(p, tmp_path):
    engine = _engine(p)
    files = life_support.stable_pair(engine)
    first_law = {"name": "fixture_s", "sha256": _digest(p, files["fixture_s"])}
    successor = {"name": "fixture_s2", "sha256": _digest(p, files["fixture_s2"]), "v": 2}
    cli = support.cli(p)
    reading = {"value": 0}
    # 23:00 local at offset 0 is 01:00 of the next local day at +2 h, whose midnight comes 23 hours later.
    late = FIRST - 60
    moved = FIRST - 120 + DAY
    with life_support.injected(engine, files):
        # The automatic law update: a gesture from the terminal records the new offset and carries the update.
        given = support.seams(p, tmp_path.joinpath("auto"), suite="av13", tz=lambda now: reading["value"])
        target = support.store(p, given)
        try:
            being = support.sow(p, target, law="fixture_s")
            path = support.store_path(p, given)
            params = being.in_order()[0][2]["body"]["laws"]["params"]
            _at(given["clock"], late)
            reading["value"] = 120
            rows = _rows(path)
            outcome = being.append("act", {"act": "greet"}, transport=cli)
            added = _rows(path)[len(rows):]
            assert [(row[2], row[1]) for row in added] == [("tz", late), ("act", late), ("evolve", late)], added
            assert [row[0] for row in added] == [rows[-1][0] + 1, rows[-1][0] + 2, rows[-1][0] + 3], "consecutive"
            assert [row[4] for row in added] == [added[0][4], added[0][4] + 1, added[0][4] + 2], "consecutive oseq"
            assert (outcome.t, added[0][6]) == (late, {"quarters": 8})
            assert added[2][6] == {"effective_from": moved, "from": first_law, "params": params, "to": successor}
            assert p.evolution.timeline_at(being, late).midnight == moved, "the next local midnight at +2 h"
            state = being.law_state(to=late)
            assert (state.tz, state.pending["to"], state.pending["effective_from"]) == (120, successor, moved)
            before, after = being.view(to=moved - 1), being.view(to=moved)
            assert (before.state["law"]["name"], after.state["law"]) == ("fixture_s", successor)
            assert not [note for note in after.notes if note["code"] == "evolve_when"], after.notes
            assert len(_evolves(path)) == 1
            assert moved != FIRST + DAY, "witness: at the old offset the update would take another midnight"
        finally:
            target.close()

        # laws_apply: the confirmation is read at offset 0, the write records +2 h and takes that midnight.
        reading["value"] = 0
        given = support.seams(p, tmp_path.joinpath("apply"), suite="av13", index=1, tz=lambda now: reading["value"])
        target = support.store(p, given)
        try:
            being = support.sow(p, target, law="fixture_s")
            path = support.store_path(p, given)
            params = being.in_order()[0][2]["body"]["laws"]["params"]
            _at(given["clock"], late)
            given["laws"]["soil"]["evap_awake"] = 8192
            diff = being.laws_diff()
            reading["value"] = 120
            rows = _rows(path)
            assert isinstance(being.laws_apply(cli, diff.confirm), p.membrane.Appended)
            added = _rows(path)[len(rows):]
            assert [(row[2], row[1]) for row in added] == [("tz", late), ("evolve", late)], added
            confirmed = dict(params, evap_awake=8192)
            assert added[1][6] == {"effective_from": moved, "from": first_law, "params": confirmed,
                                   "to": dict(first_law, v=1)}, added[1][6]
            state = being.law_state(to=late)
            assert (state.tz, state.pending["params"], state.pending["effective_from"]) == (120, confirmed, moved)
            assert being.view(to=moved - 1).state["params"] == params
            shown = being.view(to=moved)
            assert shown.state["params"] == confirmed and not [n for n in shown.notes if n["code"] == "evolve_when"]
        finally:
            target.close()


# ---------------------------------------------------------------------------
# AV14 -- the automatic law update keeps a confirmed update, and the day's budget
# ---------------------------------------------------------------------------
def test_av14_the_automatic_law_update_keeps_a_confirmed_update_and_skips_a_day_whose_budget_is_spent(p,
                                                                                                    tmp_path):
    M = p.membrane
    engine = _engine(p)
    files = life_support.stable_pair(engine)
    first_law = {"name": "fixture_s", "sha256": _digest(p, files["fixture_s"])}
    successor = {"name": "fixture_s2", "sha256": _digest(p, files["fixture_s2"]), "v": 2}
    cli = support.cli(p)
    with life_support.injected(engine, files):
        # An update applied and confirmed, then a gesture the same day: the update to the successor carries
        # the confirmed params, and they take effect with it at the same midnight.
        given = support.seams(p, tmp_path.joinpath("kept"), suite="av14")
        target = support.store(p, given)
        try:
            being = support.sow(p, target, law="fixture_s")
            path = support.store_path(p, given)
            params = being.in_order()[0][2]["body"]["laws"]["params"]
            given["clock"].advance_days(1)
            given["laws"]["soil"]["evap_awake"] = 8192
            confirmed = dict(params, evap_awake=8192)
            assert isinstance(being.laws_apply(cli, being.laws_diff().confirm), M.Appended)
            assert being.law_state().pending["params"] == confirmed and confirmed != params
            gesture = being.append("act", {"act": "greet"}, transport=cli)
            applied, automatic = _evolves(path)
            assert applied[6]["to"] == dict(first_law, v=1) and automatic[1] == gesture.t
            assert automatic[6] == {"effective_from": FIRST + DAY, "from": first_law, "params": confirmed,
                                    "to": successor}, automatic[6]
            assert being.view(to=FIRST + DAY - 1).state["params"] == params
            shown = being.view(to=FIRST + DAY)
            assert (shown.state["law"], shown.state["params"]) == (successor, confirmed)
        finally:
            target.close()

        # The day's law updates spent by laws_apply: a gesture that day writes no automatic update; the first
        # gesture of the next day writes it, with the params then in force.
        given = support.seams(p, tmp_path.joinpath("spent"), suite="av14", index=1)
        clock = given["clock"]
        target = support.store(p, given)
        try:
            being = support.sow(p, target, law="fixture_s")
            path = support.store_path(p, given)
            params = being.in_order()[0][2]["body"]["laws"]["params"]
            budget = M.law_pin("fixture_s")["budgets"]["evolve"]
            values = (8192, 12000)
            clock.advance_days(1)
            for i in range(budget):
                given["laws"]["soil"]["evap_awake"] = values[i % 2]
                assert isinstance(being.laws_apply(cli, being.laws_diff().confirm), M.Appended), i
            assert [row[6]["to"] for row in _evolves(path)] == [dict(first_law, v=1)] * budget
            rows = _rows(path)
            gesture = being.append("act", {"act": "greet"}, transport=cli)
            assert [(row[2], row[1]) for row in _rows(path)[len(rows):]] == [("act", gesture.t)], \
                "no law update past the day's budget"
            clock.advance_days(1)
            rows = _rows(path)
            gesture = being.append("act", {"act": "greet"}, transport=cli)
            added = _rows(path)[len(rows):]
            assert [(row[2], row[1]) for row in added] == [("act", gesture.t), ("evolve", gesture.t)], added
            in_force = dict(params, evap_awake=values[(budget - 1) % 2])
            assert being.law_state(to=gesture.t).params == in_force, "the last update applied is in force"
            assert added[1][6] == {"effective_from": FIRST + 2 * DAY, "from": first_law, "params": in_force,
                                   "to": successor}, added[1][6]
        finally:
            target.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
