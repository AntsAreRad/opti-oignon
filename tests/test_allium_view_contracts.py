#!/usr/bin/env python3
"""Contracts for the componion's views and settle: looking writes nothing, and checkpoints are caches.

A view reads the facts in canonical order and the kept checkpoints under the
store's lock, then asks the engine outside it, from the latest checkpoint it
can use; it writes nothing. ``settle`` is the producer that keeps reducer
states at the local midnights the retention keeps, and drops the rest.

  * AK3 -- looking changes nothing: after one settle, every look, called
    across six local midnights, leaves the link count, the head link, the
    checkpoint rows and the file's bytes as they were. Dropping every
    checkpoint changes no view's hash at two minutes, although the views
    before the drop started from one. A capped view that does not reach its
    minute is ``catching_up`` with a stored state -- the checkpoint's hash
    and minute, or the minute-0 state when none is kept -- never the partial
    state the engine stopped at. A glass jar in Bulbe refuses every new
    action ``sealed``, a law update confirmed in Daily included. Nor does
    any look add an audit entry or touch a file under the data directory,
    and the engine never runs under the store's lock when a look or a
    settle asks it.
  * AK4 -- a clock jump is an honest absence, in the garden and on the
    windowsill: a life that jumps 88 days and a life looked at and settled
    every six days hold the same facts under the same eids and show the same
    final hash, each at the recorder's minute; only the second keeps
    checkpoints, and its last view starts from one. The garden sleeps
    through a winter and wakes, with more fructan than at sowing; the
    windowsill sleeps from drought, and the fast path lives its dormant days.
  * AK8 -- checkpoints as caches: settles over 120 days keep exactly the
    retention set (the last 14 local days, the Mondays of the last 52 weeks,
    the firsts of months, the latest), computed here with the calendar of
    the standard library, each holding the true state, and drop every row
    of another engine (with no daily, weekly or monthly retention, the
    latest alone is kept); a settle its budget stops keeps what it wrote, prunes
    nothing and says what is owed; a gesture drops the checkpoints from its
    minute on in its own transaction; a fact landed after a checkpoint's event
    and before its minute, or before its event, makes it unusable: the view
    starts from an earlier one and shows what a view with no checkpoint shows.
  * AK9 -- order and packing: ``in_order`` gives the genesis first, then
    canonical order, a fact landed at minute 0 from an origin below the
    genesis's included, and the view folds it -- a capped view with no
    checkpoint serves the minute-0 state with those facts folded; with the
    input limit set to 4 KiB, a 60-fact life is viewed in several requests,
    each within the limit, none splitting a minute although one more fact
    would have fit, with the hash of the unpacked view, and so it is with
    the item limit set to 7 facts; a minute whose facts alone do not fit
    refuses the view ``limit``.
  * AK10 -- stale checkpoints and lying ones: after a fact lands before the
    event some rows run through (they hold one fact too few), the next
    settle keeps the true states at the same minutes; after one lands after
    a kept row's event and before its minute, the next settle keeps one row
    there, through the landed fact. A kept row whose state is of another
    minute, whose law version is not its state's, or whose bytes are not
    the state it names refuses a view that would start from it
    ``divergence``, and nothing is written.

Local-only. The platform loads through the shared isolation window with the
platform's configuration, keys, mode, audit log and user modules proven
unreachable; every seam is injected (``tests/_allium_store_support.py``),
the life's seams included.
"""

import datetime
import sys
import threading
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_store_support as support  # noqa: E402

BUDGET_S = {
    "test_ak3_looking_changes_nothing_and_a_capped_view_serves_a_stored_state": 2.0,
    "test_ak4_a_clock_jump_equals_an_honest_absence_in_the_garden_and_on_the_windowsill": 2.0,
    "test_ak8_settle_keeps_the_retention_set_and_a_stale_checkpoint_is_never_a_start": 2.0,
    "test_ak9_the_genesis_comes_first_and_the_packer_never_splits_a_minute": 2.0,
    "test_ak10_a_settle_replaces_stale_checkpoints_and_a_checkpoint_that_lies_is_refused": 2.0,
}
DAY = 1440
MAX_INT = (1 << 53) - 1
B = support.WALL // 60
# The first local midnight after birth, at the offset every contract here keeps (UTC).
FIRST = DAY - B % DAY
LOW, HIGH = "0" * 16, "f" * 16
# The input limit the packing contract sets: a 60-fact life then takes four requests.
SMALL = 4096
# Gestures that bring no water: a windowsill being sleeps from drought, and its days cost the engine little.
DRY = ("greet", "play", "touch", "warm")


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


def _midnight(k):
    """The ``k``-th local midnight after birth, as a minute of life."""
    return FIRST + k * DAY


def _now_t(clock):
    return (clock.wall - support.WALL) // 60


def _sow(p, target, weather="garden"):
    return target.sow(transport=support.cli(p), law="fixture", tz_minutes=0, rhythm_consent=False, weather=weather)


def _rows(path):
    """Every checkpoint row, ``(t, engine, laws, through, state_hash, blob)``, in key order."""
    return support.read(path, "SELECT t, engine, laws, through, state_hash, blob FROM checkpoints "
                              "ORDER BY t, engine, laws, through")


def _gen(p, path):
    [(value,)] = support.read(path, "SELECT value FROM meta WHERE key = 'gen'")
    return p.wire.parse(bytes(value))


def _state(p, blob):
    return p.wire.parse(zlib.decompress(bytes(blob)))


def _from_genesis(p, being, to):
    """What a view with no checkpoint shows at ``to``: one engine call from the genesis, every fact to ``to``."""
    ordered = being.in_order()
    request = {"budget": MAX_INT, "facts": [fact for _seq, _eid, fact in ordered[1:] if fact["t"] <= to],
               "genesis": ordered[0][2], "op": "advance", "state": None, "to": to, "v": 1}
    answer = p.wire.parse(p.engine.call(p.wire.emit(request)))
    assert "refused" not in answer and answer["done"], answer
    return answer


class _Recording:
    """The engine seam, recording every request it answers and the answer, for the length of a ``with``."""

    def __init__(self, p):
        self._p = p
        self._real = p.engine.call
        self.sent = []

    def __enter__(self):
        self._p.engine.call = self._call
        return self

    def __exit__(self, *exc):
        self._p.engine.call = self._real

    def _call(self, request):
        answer = self._real(request)
        self.sent.append((bytes(request), bytes(answer)))
        return answer

    def advances(self):
        """``[(request, answer)]`` of the ``advance`` requests, parsed."""
        out = []
        for request, answer in self.sent:
            parsed = self._p.wire.parse(request)
            if parsed["op"] == "advance":
                out.append((parsed, self._p.wire.parse(answer)))
        return out


def _refused(p, call):
    """What a call gave: ``None`` when it answered, else the refusal's code."""
    try:
        call()
    except (p.membrane.MembraneRefused, p.store.StoreRefused, p.life.LifeRefused) as refusal:
        return refusal.code
    return None


def _land_many(being, facts):
    """Land several facts from other devices in one transaction, the way ``support.land`` lands one."""

    def transaction(conn):
        meta = being._meta(conn)
        for origin, oseq, t, kind, body in facts:
            head = being._head(conn)
            being._insert_fact(conn, origin=origin, oseq=oseq, t=t, kind=kind, body=body, head=head)
            name = "oseq_next:" + origin
            meta[name] = max(meta.get(name, 0), oseq + 1)
            being._put(conn, name, meta[name])
        gen = meta["gen"] + 1
        being._put(conn, "gen", gen)
        being._rewrite_anchor(conn, gen)

    being._write(transaction)


# ---------------------------------------------------------------------------
# AK3 -- looking changes nothing; a capped view serves a stored state
# ---------------------------------------------------------------------------
def _snapshot(path):
    [(links,)] = support.read(path, "SELECT COUNT(*) FROM links")
    head = support.read(path, "SELECT seq, link FROM links ORDER BY seq DESC LIMIT 1")
    return links, head, _rows(path), support.sha256_file(path)


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


def test_ak3_looking_changes_nothing_and_a_capped_view_serves_a_stored_state(p, tmp_path):
    cli = support.cli(p)
    given = support.seams(p, tmp_path.joinpath("pot"), suite="ak3")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = _sow(p, target, "windowsill")
        path = support.store_path(p, given)
        taught = None
        for i in range(20):
            clock.advance_days(4)
            if i == 7:
                taught = being.append("lang_teach", {}, transport=cli, payload="word")
            else:
                being.append("act", {"act": DRY[i % len(DRY)]}, transport=cli)
        clock.advance_days(10)
        [ref] = [fact["body"]["payload"] for _seq, eid, fact in being.in_order() if eid == taught.eid]
        settled = being.settle()
        assert settled.done and settled.written >= 1 and settled.owed == 0, settled

        # (a) Every look, across six local midnights: nothing moves, not even the checkpoints.
        before = _snapshot(path)
        assert len(before[2]) == settled.written, "presence: the settle kept checkpoints"
        middle = _midnight(40) + 100
        looks = {
            "status": lambda: target.status("local"),
            "open": lambda: target.open("local"),
            "view": lambda: being.view(),
            "view at a minute": lambda: being.view(to=middle),
            "capped view": lambda: being.view(cap=1),
            "in_order": lambda: being.in_order(),
            "trunk": lambda: list(being.trunk()),
            "head": lambda: being.head(),
            "verify": lambda: being.verify(),
            "unseal": lambda: being.unseal(ref),
            "laws_diff": lambda: being.laws_diff(),
            "law_state": lambda: being.law_state(),
            "law_state at a minute": lambda: being.law_state(to=middle),
        }
        around = _surroundings(given)
        assert around[0] >= 1 and len(around[1]) >= 3, "presence: the audit log and the data directory hold rows"
        for step in range(6):
            clock.advance_days(1)
            for name, look in looks.items():
                look()
                assert _snapshot(path) == before, (step, name)
                assert _surroundings(given) == around, ("nor the audit log or any file", step, name)

        # (b) Dropping every checkpoint changes no view's hash, at the recorder's minute and at an earlier one.
        seen = {}
        for to in (None, middle):
            with _Recording(p) as recording:
                seen[to] = being.view(to=to)
            first = recording.advances()[0][0]
            assert first["state"] is not None, ("presence: the view started from a checkpoint", to)
            assert (seen[to].status, seen[to].hash) == ("current", _from_genesis(p, being, seen[to].at)["hash"])
        count = len(_rows(path))
        assert count >= 1
        assert being.checkpoint_prune(keep=()) == count
        assert _rows(path) == []
        for to in (None, middle):
            with _Recording(p) as recording:
                again = being.view(to=to)
            assert recording.advances()[0][0]["state"] is None, "from the genesis now"
            assert (again.at, again.hash) == (seen[to].at, seen[to].hash), to

        # (c) A capped view that does not get there serves a stored state, never the partial one.
        # With no checkpoint kept: the minute-0 state, although the engine got past minute 0.
        goal = _now_t(clock)
        with _Recording(p) as recording:
            bare = being.view(cap=100)
        partial = [answer for _request, answer in recording.advances() if not answer["done"]]
        assert partial and partial[0]["at"] > 0, "presence: the engine stopped short, past minute 0"
        zero = _from_genesis(p, being, 0)
        assert (bare.status, bare.at, bare.hash, bare.state) == ("catching_up", 0, zero["hash"], zero["state"])
        assert partial[0]["hash"] != bare.hash
        awake_day = p.lawfiles.law("fixture")["work"]["ceilings"]["awake_day"]
        assert bare.owed == -(-goal // DAY) * awake_day > 0

        assert being.settle().done
        assert _surroundings(given)[1] != around[1], "witness: a settle's writes show in the files"
        rows = _rows(path)
        kept = rows[-1]
        clock.advance_days(2)
        goal = _now_t(clock)
        with _Recording(p) as recording:
            capped = being.view(cap=1)
        partial = [answer for _request, answer in recording.advances() if not answer["done"]]
        assert partial and all(answer["hash"] != kept[4] for answer in partial), "presence: stopped short"
        assert capped.status == "catching_up", capped
        assert (capped.at, capped.hash, capped.state) == (kept[0], kept[4], _state(p, kept[5])), capped
        assert capped.state["at"] == capped.at
        assert capped.owed == -(-(goal - capped.at) // DAY) * awake_day, capped.owed
        assert (capped.labels, capped.provisional) == (("prototype",), True)
        assert capped.env["day"] == (B + capped.at) // DAY, "the environment of the minute it shows"
        current = being.view()
        assert (current.status, current.at) == ("current", goal)
        assert current.hash != capped.hash and current.work > 1, "witness: one unit does not get there"
        assert _rows(path) == rows, "a capped view keeps nothing"

        # (e) The engine never runs under the store's lock when a look or a settle asks it.
        held = []
        real = p.engine.call

        def call(request):
            held.append(target._lock._is_owned())
            return real(request)

        p.engine.call = call
        try:
            clock.advance_days(1)
            for name, look in (("view", being.view), ("view at a minute", lambda: being.view(to=middle)),
                               ("law_state", being.law_state), ("laws_diff", being.laws_diff),
                               ("settle", being.settle)):
                del held[:]
                look()
                assert held and not any(held), (name, held)
            # Witness: a write asks the engine inside its transaction, under the lock; its day's cross-anchor
            # reaches the audit log.
            del held[:]
            being.append("act", {"act": "greet"}, transport=cli)
            assert any(held), "witness: the probe sees the lock held"
            assert _surroundings(given)[0] > around[0], "witness: a gesture's cross-anchor shows in the audit log"
        finally:
            p.engine.call = real
    finally:
        target.close()

    # (d) A glass jar in Bulbe refuses every new action by name; in Daily each answers.
    mode = support.Mode("daily")
    jar = support.seams(p, tmp_path.joinpath("jar"), suite="ak3", index=1, probe=None, mode=mode,
                        anchor_secret=lambda: ("none", None, "nokey"),
                        persistence={"busy_timeout_ms": 5000, "path": "allium", "require_encryption": False})
    glass = support.store(p, jar)
    try:
        being = _sow(p, glass)
        assert support.store_path(p, jar, suffix=".glass.db").exists()
        jar["clock"].advance_days(3)
        actions = {
            "view": lambda: being.view(),
            "view at a minute": lambda: being.view(to=10),
            "capped view": lambda: being.view(cap=1),
            "settle": lambda: being.settle(),
            "in_order": lambda: being.in_order(),
            "checkpoint_prune": lambda: being.checkpoint_prune(keep=()),
            "laws_diff": lambda: being.laws_diff(),
            "law_state": lambda: being.law_state(),
            "laws_pin": lambda: being.laws_pin(support.cli(p)),
            "laws_unpin": lambda: being.laws_unpin(support.cli(p)),
        }
        for name, action in actions.items():
            assert _refused(p, action) is None, name
        mode.value = "bulbe"
        for name, action in actions.items():
            assert _refused(p, action) == "sealed", name

        # laws_apply too: a law update confirmed in Daily is refused ``sealed`` in Bulbe, as the first action
        # on an open jar, and nothing is written; back in Daily the same confirmation writes it.
        mode.value = "daily"
        being = glass.open("local")
        jar["laws"]["soil"]["evap_awake"] = 8192
        confirm = being.laws_diff().confirm
        jar_path = support.store_path(p, jar, suffix=".glass.db")
        links = support.read(jar_path, "SELECT COUNT(*) FROM links")
        mode.value = "bulbe"
        assert _refused(p, lambda: being.laws_apply(support.cli(p), confirm)) == "sealed"
        assert support.read(jar_path, "SELECT COUNT(*) FROM links") == links
        mode.value = "daily"
        being = glass.open("local")
        assert isinstance(being.laws_apply(support.cli(p), confirm), p.membrane.Appended), "witness: Daily writes"
    finally:
        glass.close()


# ---------------------------------------------------------------------------
# AK4 -- a clock jump equals an honest absence
# ---------------------------------------------------------------------------
def _jumped(p, tmp_path, weather):
    """J: a water on day 1, 88 days away, a view, a greet on day 89, a view."""
    cli = support.cli(p)
    given = support.seams(p, tmp_path.joinpath("jumped_" + weather), suite="ak4")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = _sow(p, target, weather)
        clock.advance_days(1)
        being.append("act", {"act": "water"}, transport=cli)
        clock.advance_days(88)
        before = being.view()
        assert (before.status, before.at) == ("current", _now_t(clock)), before
        being.append("act", {"act": "greet"}, transport=cli)
        final = being.view()
        assert (final.status, final.at) == ("current", _now_t(clock)), final
        return being.in_order(), final, _rows(support.store_path(p, given)), []
    finally:
        target.close()


def _honest(p, tmp_path, weather):
    """H: a water on day 1, then fourteen times six days with a settle and a view, a greet on day 89, a view."""
    cli = support.cli(p)
    life = {"checkpoints": {"daily": 1, "monthly": False, "weekly": 0}, "skew_note_min": 5}
    given = support.seams(p, tmp_path.joinpath("honest_" + weather), suite="ak4", life=life)
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = _sow(p, target, weather)
        zero = being.view(to=0)
        clock.advance_days(1)
        being.append("act", {"act": "water"}, transport=cli)
        seen = [zero]
        for _ in range(14):
            clock.advance_days(6)
            assert being.settle().done
            look = being.view()
            assert (look.status, look.at) == ("current", _now_t(clock)), look
            seen.append(look)
        clock.advance_days(4)
        being.append("act", {"act": "greet"}, transport=cli)
        final = being.view()
        assert (final.status, final.at) == ("current", _now_t(clock)), final
        with _Recording(p) as recording:
            again = being.view()
        assert recording.advances()[0][0]["state"] is not None, "presence: H's last view starts from a checkpoint"
        assert (again.at, again.hash) == (final.at, final.hash)
        return being.in_order(), final, _rows(support.store_path(p, given)), seen
    finally:
        target.close()


def test_ak4_a_clock_jump_equals_an_honest_absence_in_the_garden_and_on_the_windowsill(p, tmp_path):
    for weather in ("garden", "windowsill"):
        facts, final, rows, _ = _jumped(p, tmp_path, weather)
        honest_facts, honest_final, honest_rows, seen = _honest(p, tmp_path, weather)
        assert [(eid, fact) for _seq, eid, fact in honest_facts] == [(eid, fact) for _seq, eid, fact in facts]
        assert [fact["kind"] for _seq, _eid, fact in facts] == ["genesis", "act", "act"], "no fact of the recorder"
        assert (honest_final.at, honest_final.hash) == (final.at, final.hash) == (89 * DAY, final.hash), weather
        assert rows == [] and len(honest_rows) >= 1, "only the life looked at keeps checkpoints"

        stages = [look.state["organs"]["stage"] for look in seen[1:]]
        if weather == "garden":
            winter = [i for i, stage in enumerate(stages) if stage["dormant"] and stage["cause"] == "winter"]
            assert winter, "witness: a winter dormancy"
            assert any(not stage["dormant"] for stage in stages[winter[0]:]), "witness: and a wake after it"
            sown = seen[0].state["organs"]["chem"]["fructan"]
            assert final.state["organs"]["chem"]["fructan"] > sown, "witness: fructan above sowing"
        else:
            stage = final.state["organs"]["stage"]
            assert stage["dormant"] and stage["cause"] == "dry", stage
            request = {"budget": MAX_INT, "facts": [fact for _seq, _eid, fact in facts[1:]], "genesis": facts[0][2],
                       "op": "advance", "probe": {"trace": True}, "state": None, "to": final.at, "v": 1}
            traced = p.wire.parse(p.engine.call(p.wire.emit(request)))
            assert traced["hash"] == final.hash and traced["trace"]["fast_path_days"] > 0, "witness: the fast path"


# ---------------------------------------------------------------------------
# AK8 -- checkpoints as caches
# ---------------------------------------------------------------------------
def _civil(k):
    """The civil date of the ``k``-th local midnight, by the standard library's calendar."""
    return datetime.date(1970, 1, 1) + datetime.timedelta(days=(B + _midnight(k)) // DAY)


def _retention(today, last, *, daily=14, weekly=52):
    """The kept midnights ``k`` up to ``last``, by the standard library's calendar: independent of the store."""
    kept = set()
    for k in range(last + 1):
        day = (B + _midnight(k)) // DAY
        civil = _civil(k)
        if day > today - daily or (civil.weekday() == 0 and day > today - 7 * weekly) or civil.day == 1 or k == last:
            kept.add(k)
    return kept


def test_ak8_settle_keeps_the_retention_set_and_a_stale_checkpoint_is_never_a_start(p, tmp_path):
    cli = support.cli(p)
    given = support.seams(p, tmp_path, suite="ak8")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = _sow(p, target, "windowsill")
        path = support.store_path(p, given)
        engine = p.store._engine_id()
        assert LOW < being.origin < HIGH

        # Settles over 120 days; a planted row of another engine before the last one.
        previous = None
        for step in range(3):
            for _ in range(8):
                clock.advance_days(5)
                being.append("act", {"act": support.ACTS[step % len(support.ACTS)]}, transport=cli)
            if step == 0:
                # A budget stops a settle: what it wrote stays, nothing is pruned, and the rest is owed.
                now = _now_t(clock)
                awake_day = p.lawfiles.law("fixture")["work"]["ceilings"]["awake_day"]
                first = _midnight(min(_retention((B + now) // DAY, (now - FIRST) // DAY)))
                assert being.settle(budget=1) == (False, 0, 0, 0, -(-now // DAY) * awake_day)
                assert _rows(path) == []
                short = being.settle(budget=being.view(to=first).work)
                assert (short.done, short.at, short.written, short.pruned) == (False, first, 1, 0), short
                assert short.owed == -(-(now - first) // DAY) * awake_day > 0
                assert [row[0] for row in _rows(path)] == [first], "the next settle starts from it"
            if step == 2:
                ordered = being.in_order()
                state = being.view(to=_midnight(3)).state
                p.store._ENGINE_ID["sha256"] = "e" * 64
                try:
                    being.checkpoint_put(_midnight(3), state["law"]["v"], ordered[0][1], state)
                finally:
                    p.store._ENGINE_ID["sha256"] = engine
                assert [row[1] for row in _rows(path)].count("e" * 64) == 1
            rows, gen = _rows(path), _gen(p, path)
            settled = being.settle()
            assert settled.done and settled.owed == 0, settled
            assert _gen(p, path) == gen + settled.written + (1 if settled.pruned else 0), "the prune is one write"
            if previous is not None:
                assert settled.pruned > 0, "presence: the retention dropped rows"
            previous = rows
        goal = _now_t(clock)
        assert goal == 120 * DAY
        today = (B + goal) // DAY
        last = (goal - FIRST) // DAY
        expected = _retention(today, last)
        older = [_civil(k) for k in expected if (B + _midnight(k)) // DAY <= today - 14]
        mondays = [civil for civil in older if civil.weekday() == 0]
        firsts = [civil for civil in older if civil.day == 1]
        assert mondays and firsts and len(expected) < last + 1, "presence: every rule keeps, and one drops"
        rows = _rows(path)
        assert all(row[1] == engine for row in rows), "no row of another engine is left"
        assert [row[0] for row in rows] == sorted(_midnight(k) for k in expected)
        for row in (rows[0], rows[len(rows) // 2], rows[-1]):
            state = _state(p, row[5])
            truth = _from_genesis(p, being, row[0])
            assert (row[4], state["at"]) == (truth["hash"], row[0]), "each kept row holds the true state"

        # A gesture drops the checkpoints from its minute on, in its own transaction: two rows kept past the
        # latest fact, then the clock set back to the first of them, on the day of the last write (so no
        # daily cross-anchor adds a write of its own).
        clock.wall = support.WALL + _midnight(121) * 60
        assert being.settle().done
        tail = [row[0] for row in _rows(path) if row[0] > _midnight(119)]
        assert tail == [_midnight(120), _midnight(121)], tail
        clock.wall = support.WALL + _midnight(120) * 60
        rows, gen = _rows(path), _gen(p, path)
        late = being.append("act", {"act": "warm"}, transport=cli)
        assert late.t == _midnight(120) and late.t // DAY == goal // DAY, late
        assert _gen(p, path) == gen + 1, "the gesture and the drop are one transaction"
        assert _rows(path) == [row for row in rows if row[0] < _midnight(120)]
        assert len(rows) - len(_rows(path)) == 2, "the row at its minute and the one after it"

        # Stale by its event: a fact landed after a checkpoint's event and at or before its minute.
        clock.wall = support.WALL + _midnight(122) * 60
        assert being.settle().done
        throughs = {row[0]: row[3] for row in _rows(path) if row[0] >= _midnight(120)}
        assert sorted(throughs) == [_midnight(k) for k in (120, 121, 122)], throughs
        assert set(throughs.values()) == {late.eid}, "the three rows run through the late gesture"
        support.land(being, HIGH, 0, _midnight(122) - 1, "act", {"act": "touch"})
        with _Recording(p) as recording:
            seen = being.view(to=_midnight(122))
        assert recording.advances()[0][0]["state"]["at"] == _midnight(121), "the latest usable row is the start"
        assert seen.hash == _from_genesis(p, being, _midnight(122))["hash"]

        # Stale by its count: a fact landed before a checkpoint's event, in canonical order (same minute,
        # an origin that sorts first).
        support.land(being, LOW, 0, late.t, "act", {"act": "touch"})
        with _Recording(p) as recording:
            seen = being.view(to=_midnight(121))
        start = recording.advances()[0][0]["state"]
        assert start["at"] == _midnight(119) < late.t, "every row through that event is skipped"
        truth = _from_genesis(p, being, _midnight(121))
        assert seen.hash == truth["hash"]
        [row] = [row for row in _rows(path) if row[0] == _midnight(121)]
        request = {"budget": MAX_INT, "facts": [fact for _seq, _eid, fact in being.in_order()[1:]
                                                if _midnight(121) < fact["t"] <= _midnight(121)],
                   "genesis": being.in_order()[0][2], "op": "advance", "state": _state(p, row[5]),
                   "to": _midnight(121), "v": 1}
        stale = p.wire.parse(p.engine.call(p.wire.emit(request)))
        assert stale["hash"] != truth["hash"], "witness: that row would have hidden the landed fact"
        assert being.checkpoint_prune(keep=()) > 0
        assert being.view(to=_midnight(121)).hash == truth["hash"], "the hash of a view with no checkpoint"

        # With no daily, weekly or monthly retention, a settle keeps the latest firing minute alone.
        latest = {"checkpoints": {"daily": 0, "monthly": False, "weekly": 0}, "skew_note_min": 5}
        alone = support.store(p, dict(given, life=latest))
        try:
            settled = alone.open("local").settle()
            assert (settled.done, settled.written) == (True, 1), settled
            assert [row[0] for row in _rows(path)] == [_midnight(122)]
        finally:
            alone.close()
    finally:
        target.close()


# ---------------------------------------------------------------------------
# AK10 -- a settle replaces stale checkpoints; a checkpoint that lies is refused
# ---------------------------------------------------------------------------
def test_ak10_a_settle_replaces_stale_checkpoints_and_a_checkpoint_that_lies_is_refused(p, tmp_path):
    cli = support.cli(p)
    given = support.seams(p, tmp_path, suite="ak10")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = _sow(p, target, "windowsill")
        path = support.store_path(p, given)
        assert LOW < being.origin < HIGH
        gestures = []
        for i in range(6):
            clock.advance_days(3)
            gestures.append(being.append("act", {"act": DRY[i % len(DRY)]}, transport=cli))
        clock.advance_days(3)
        assert being.settle().done

        # Stale by its count: a fact landed at a gesture's minute from an origin that sorts first. Every row
        # through that gesture or a later one still names the last fact by its minute, but holds one fact too
        # few; the next settle keeps the true states in their place.
        moment = gestures[2].t
        support.land(being, LOW, 0, moment, "act", {"act": "touch"})
        stale = [row for row in _rows(path) if row[0] >= moment]
        assert len(stale) >= 2, "presence: rows through the gesture or a later one"
        assert stale[0][4] != _from_genesis(p, being, stale[0][0])["hash"], "witness: that row now lies"
        settled = being.settle()
        assert settled.done and settled.owed == 0, settled
        rows = _rows(path)
        assert not {row[4] for row in rows} & {row[4] for row in stale}, "no stale state is kept"
        assert [row[0] for row in rows if row[0] >= moment] == [row[0] for row in stale], "the same minutes kept"
        for row in (rows[0], stale[0], rows[-1]):
            [kept] = [mine for mine in rows if mine[0] == row[0]]
            assert kept[4] == _from_genesis(p, being, kept[0])["hash"], "each kept row holds the true state"

        # Stale by its event: a fact landed the minute before the latest kept midnight, after the event that
        # row runs through. The next settle keeps one row there, through the landed fact, and the old row goes.
        latest = rows[-1]
        landed = support.land(being, HIGH, 0, latest[0] - 1, "act", {"act": "warm"})
        assert latest[3] != landed[0], "presence: the row ran through an earlier event"
        assert being.settle().done
        there = [row for row in _rows(path) if row[0] == latest[0]]
        assert [(row[3], row[4]) for row in there] == [(landed[0], _from_genesis(p, being, latest[0])["hash"])], \
            "one row at that minute, through the landed fact, holding the true state"

        # A checkpoint that does not hold the state it names is refused ``divergence`` and nothing is written:
        # a state of another minute, a law version that is not the state's, bytes that are not the state's.
        rows = _rows(path)
        kept = {row[0]: row[3] for row in rows}
        _t, laws, through, _hash, blob = rows[-1][0], rows[-1][2], rows[-1][3], rows[-1][4], rows[-1][5]
        planted = latest[0] + 1
        earlier = _state(p, blob)
        truth = being.view(to=planted)
        assert truth.state["at"] == planted != earlier["at"]
        # The bytes of a state of the same minute and law that is not the one the row names: one more unit of
        # sugar.
        forged = p.wire.parse(p.wire.emit(truth.state))
        forged["organs"]["chem"]["sugar"] += 1
        plants = {
            "a state of another minute": lambda: being.checkpoint_put(planted, laws, through, earlier),
            "another law version": lambda: being.checkpoint_put(planted, laws + 5, through, truth.state),
            "bytes that are not the state named": lambda: (
                being.checkpoint_put(planted, laws, through, truth.state),
                support.edit(path, lambda conn: conn.execute("UPDATE checkpoints SET blob = ? WHERE t = ?",
                                                             (zlib.compress(p.wire.emit(forged)), planted)))),
        }
        for name, plant in plants.items():
            plant()
            assert planted in [row[0] for row in _rows(path)], ("presence: planted", name)
            before = _snapshot(path)
            with pytest.raises(p.store.StoreRefused) as refusal:
                being.view(to=planted)
            assert refusal.value.code == "divergence", (name, refusal.value)
            assert _snapshot(path) == before, ("nothing written", name)
            assert being.checkpoint_prune(keep=kept) == 1, name
        seen = being.view(to=planted)
        assert (seen.status, seen.hash) == ("current", truth.hash), "witness: without the planted row it answers"
    finally:
        target.close()


# ---------------------------------------------------------------------------
# AK9 -- the genesis first, canonical order, and a packer that never splits a minute
# ---------------------------------------------------------------------------
def test_ak9_the_genesis_comes_first_and_the_packer_never_splits_a_minute(p, tmp_path):
    cli = support.cli(p)
    given = support.seams(p, tmp_path, suite="ak9")
    clock = given["clock"]
    target = support.store(p, given)
    try:
        being = _sow(p, target)
        assert LOW < being.origin < HIGH
        _land_many(being, [(LOW, 0, 0, "act", {"act": "touch"}), (HIGH, 0, 0, "act", {"act": "greet"})])
        minutes = []
        for i in range(12):
            clock.wall += 60 * 479
            minutes.append(_now_t(clock))
            for j in range(3):
                being.append("act", {"act": support.ACTS[(i + j) % len(support.ACTS)]}, transport=cli)
        landed = []
        for i, t in enumerate(minutes[:11]):
            landed += [(LOW, 1 + 2 * i, t, "act", {"act": "play"}), (LOW, 2 + 2 * i, t, "act", {"act": "warm"})]
        _land_many(being, landed)

        ordered = being.in_order()
        assert len(ordered) == 1 + 60, "the genesis and 60 facts"
        assert (ordered[0][0], ordered[0][2]["kind"]) == (0, "genesis"), "the genesis first"
        assert [(fact["t"], fact["origin"]) for _seq, _eid, fact in ordered[1:3]] == [(0, LOW), (0, HIGH)]
        keys = [(fact["t"], fact["origin"].encode("ascii"), fact["oseq"]) for _seq, _eid, fact in ordered[1:]]
        assert keys == sorted(keys) and len(set(keys)) == len(keys), "canonical order, text byte by byte"
        linked = support.read(support.store_path(p, given), "SELECT seq, eid FROM links ORDER BY seq")
        assert sorted((seq, eid) for seq, eid, _fact in ordered) == linked, "every linked fact, once"
        trunk = [(envelope, body) for envelope, body in being.trunk()]
        assert sorted((dict(envelope, body=body) for envelope, body in trunk[1:]),
                      key=lambda f: (f["t"], f["origin"].encode("ascii"), f["oseq"])) == \
            [fact for _seq, _eid, fact in ordered[1:]], "the facts as the trunk holds them"

        whole = being.view()
        assert whole.status == "current" and whole.state["n"] == len(ordered) - 1, "the landed minute-0 facts fold"
        assert whole.hash == _from_genesis(p, being, whole.at)["hash"]

        # A capped view with no checkpoint serves the state a view of minute 0 shows: minute 0's facts folded,
        # never a state without them.
        zero = being.view(to=0)
        assert (zero.status, zero.state["n"]) == ("current", 2), "presence: the two facts of minute 0 fold"
        assert _rows(support.store_path(p, given)) == [], "presence: no checkpoint to start from"
        capped = being.view(cap=1)
        assert (capped.status, capped.at, capped.hash, capped.state) == ("catching_up", 0, zero.hash, zero.state)

        # The input limit set to 4 KiB: several requests, whole minutes, the unpacked hash.
        limit = p.life.limits()[0]
        assert limit == 1 << 20, "the engine's own limit"
        p.life._LIMITS["input"] = SMALL
        try:
            with _Recording(p) as recording:
                packed = being.view()
        finally:
            p.life._LIMITS["input"] = limit
        requests = [request for request, _answer in recording.advances()]
        assert len(requests) >= 3, len(requests)
        assert all(len(raw) <= SMALL for raw, _answer in recording.sent)
        assert [fact for request in requests for fact in request["facts"]] == \
            [fact for _seq, _eid, fact in ordered[1:]], "every fact once, in order"
        snug = 0
        for before, after in zip(requests, requests[1:]):
            assert after["facts"] and before["to"] == after["facts"][0]["t"] - 1
            assert after["state"]["at"] == before["to"]
            assert {fact["t"] for fact in before["facts"]}.isdisjoint(fact["t"] for fact in after["facts"])
            one_more = dict(before, facts=before["facts"] + after["facts"][:1], to=after["facts"][0]["t"])
            snug += len(p.wire.emit(one_more)) <= SMALL
        assert snug >= 1, "presence: a cut where one more fact would have fit"
        assert (packed.status, packed.hash, packed.at) == ("current", whole.hash, whole.at)

        # The item limit set to 7 facts: every request within it, whole minutes, the unpacked hash.
        items = p.life.limits()[1]
        assert items >= len(ordered), "the engine's own item limit holds the whole life"
        p.life._LIMITS["items"] = 7
        try:
            with _Recording(p) as recording:
                counted = being.view()
        finally:
            p.life._LIMITS["items"] = items
        requests = [request for request, _answer in recording.advances()]
        assert len(requests) >= 3 and all(len(request["facts"]) <= 7 for request in requests), \
            [len(request["facts"]) for request in requests]
        assert [fact for request in requests for fact in request["facts"]] == \
            [fact for _seq, _eid, fact in ordered[1:]], "every fact once, in order"
        for before, after in zip(requests, requests[1:]):
            assert {fact["t"] for fact in before["facts"]}.isdisjoint(fact["t"] for fact in after["facts"])
        assert (counted.status, counted.hash, counted.at) == ("current", whole.hash, whole.at)

        # A minute whose facts alone do not fit refuses the view by name, and nothing moves.
        clock.wall += 60 * 30
        crowded = _now_t(clock)
        _land_many(being, [(HIGH, 1 + i, crowded, "act", {"act": "touch"}) for i in range(30)])
        p.life._LIMITS["input"] = SMALL
        try:
            with pytest.raises(p.life.LifeRefused) as refusal:
                being.view()
        finally:
            p.life._LIMITS["input"] = limit
        assert refusal.value.code == "limit" and str(crowded) in refusal.value.detail, refusal.value
        assert being.view().status == "current", "witness: under the engine's own limit it fits"
        assert being.in_order()[0] == ordered[0], "the genesis stays first, whatever lands at minute 0 or later"
    finally:
        target.close()
