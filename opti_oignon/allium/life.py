"""The being's life as the platform serves it: views, settle, and the packer that feeds the engine its facts.

A view is read-only: it writes no fact, no checkpoint, no meta and no
anchor. Under the store's lock it reads the mode gate, the facts in
canonical order (``BeingStore.in_order``), the latest recorded minute, the
wall clock and this engine's checkpoints; then it lets the lock go and asks
the engine's ``advance``, so the engine never runs under the lock.

The minute a view shows, when none is asked for, is the recorder's: the
wall clock's minute of life, never before the latest recorded fact. A being
is never shown younger than its last fact; after the clock is set back with
no fact since, it can be shown younger than an earlier view showed it,
since nothing records a look. A wall that cannot be read, or one before the
birth, refuses the view ``clock`` by name, never a guess. The minute does
not depend on the checkpoints, which are caches: dropping every one of them
changes no view.

Where a view starts. From the latest checkpoint of this engine at or before
its minute that it can use, else from the genesis. A checkpoint is used
only when the event it runs through is still the last fact, in canonical
order, at or before its minute (a fact landed after that event and by that
minute makes it stale), when its blob inflates to the bytes its hash names
and its state is at its own minute under its own law version (else the
store is refused ``divergence``), and when the state's count of facts is the
number of facts up to that event (a fact landed before the event makes it
stale). A stale checkpoint is skipped, never repaired; its blob is never
inflated.

A view asked with a ``cap`` of work that does not reach its minute serves
the state it started from -- the checkpoint's, or the minute-0 state, the
facts of minute 0 folded, as a view of minute 0 shows it -- as
``catching_up``, with what is still owed estimated from the law's awake-day
ceiling: a partial state is never served. The state it starts from is
asked within the same cap: no request of a capped view carries a budget
above it. The engine never splits a minute, so the minute-0 fold gives the
same answer under any budget. A view's ``work`` and ``notes`` are those of
every engine call it made, the one that gave the state it serves included.

``settle`` is a producer of caches. From the same start, it asks the law
timeline for the local midnights where the daily layer fires, keeps the
ones the retention names -- the last ``daily`` local days, the Mondays of
the last ``weekly`` weeks, every first of a month when ``monthly``, and the
latest -- relative to the local day the being has reached (the daily
layer's own count, which a westward offset never takes back), and advances
to each kept midnight after the start, one checkpoint each, through
``BeingStore.checkpoint_put``. The rows a fact landed before their event
made stale are dropped first, in one write: each still names the event it
runs through, and the true state kept at its minute would meet it under the
same key. When it gets through every kept midnight it drops every other
checkpoint, and every one of another engine, in one prune; a settle the
budget stops keeps what it has, since the latest kept state is where the
next settle starts.

The packer. One request carries at most the engine's input limit in bytes
and its item limit in facts, as the engine's own identity states them; the
packer measures the request it builds. It adds whole minutes of facts, in
canonical order, while the request fits; a request that leaves facts behind
runs to the minute before the next one, and the next request starts from the
state it returned. The facts of one minute are never split: a minute whose
facts alone do not fit refuses the whole call ``limit``. The same packer
chains ``advance`` and the law ``timeline``.

``stored`` serves the state a view starts from, as it is kept: what a look
shows, frozen, when the engine stops on a fault past it; it takes the
view's cap. ``deep_verify`` verifies the chain, then replays every kept
state and the state served now on the reference engine (``ask`` and
``chain`` take the engine to call) and says what agreed, what was stale and
what lies past the minute shown; a disagreement is ``Diverged``, a finding,
and nothing is written or repaired.

Refusals: an engine refusal is ``LifeRefused`` with the engine's own code and
detail; ``limit`` also comes from the packer, and ``bad_request`` from a
minute or a budget that is not one. None of them is new on the wire.

Nothing is imported at module level but the standard library.
"""

import bisect
import hashlib
from typing import NamedTuple

checkpoint_before_apply = True

MAX_INT = (1 << 53) - 1
DAY = 1440
# The local day index of a Monday, modulo 7: day 0, 1970-01-01, was a Thursday.
MONDAY = 4

_LIMITS = {}


class LifeRefused(ValueError):
    """A life the engine would not serve: ``code`` is a wire refusal code, ``detail`` the engine's words."""

    def __init__(self, code, detail=""):
        from . import wire

        if code not in wire.REFUSALS:
            raise ValueError(f"unknown life refusal: {code}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class View(NamedTuple):
    """What a view shows: the being's state at ``at``, and how it was reached."""

    status: str
    state: dict
    at: int
    hash: str
    env: dict
    labels: tuple
    law: dict
    provisional: bool
    owed: int
    work: int
    notes: list


class Settled(NamedTuple):
    """What a settle did: whether it got through, how far the kept states reach, rows written and dropped, work owed."""

    done: bool
    at: int
    written: int
    pruned: int
    owed: int


class Diverged(ValueError):
    """A replay on the reference engine that disagrees with what the store keeps or the engine serves.

    ``kind`` is ``kept`` (a kept state), ``blob`` (a kept state that does
    not hold what its hash names) or ``served`` (the state served now);
    ``t`` is its minute of life and ``day`` its day of life, ``laws`` the law
    version at that minute, and ``engine`` the engine that served the view
    (``native`` or ``reference``) when ``kind`` is ``served``. It is a
    finding, never a wire code, and nothing is replaced.
    """

    KINDS = ("kept", "blob", "served")

    def __init__(self, kind, t, laws, engine=None):
        if kind not in self.KINDS:
            raise ValueError(f"unknown divergence: {kind}")
        super().__init__(f"{kind} state at minute {t} disagrees with the reference replay")
        self.kind = kind
        self.code = kind
        self.t = t
        self.day = t // DAY if _is_int(t) else None
        self.laws = laws
        self.engine = engine


class Deep(NamedTuple):
    """What a deep verification found: facts and days of life, kept states compared, agreed, stale and ahead."""

    facts: int
    days: int
    kept: int
    agreed: int
    stale: int
    ahead: int
    engine: str
    version: object


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def limits():
    """``(input bytes, items)``: the most one request may carry, as the engine's identity states them.

    Read from the engine once and kept in ``_LIMITS``; a value already there
    is kept, so a caller may set the input limit lower.
    """
    if "input" not in _LIMITS or "items" not in _LIMITS:
        from . import engine, wire

        info = wire.parse(engine.call(wire.emit({"op": "engine", "v": 1})))
        _LIMITS.setdefault("input", info["limits"]["input"])
        _LIMITS.setdefault("items", info["limits"]["items"])
    return _LIMITS["input"], _LIMITS["items"]


def ask(request, call=None):
    """The engine's answer to one request, parsed; a refusal is ``LifeRefused`` with the engine's words.

    ``call`` is the engine asked (default: ``engine.call``, native when the
    handshake agreed); a deep verification passes the reference's own.
    """
    from . import engine, wire

    if call is None:
        call = engine.call
    answer = wire.parse(call(wire.emit(request)))
    if "refused" in answer:
        raise LifeRefused(answer["refused"], answer.get("detail", ""))
    return answer


def chain(base, key, state, facts, to, budget=None, call=None):
    """Ask the engine for ``facts`` (canonical order, none after ``to``) up to ``to``, packed; the answers.

    ``base`` is the request without its state, ``facts``, ``to`` and
    ``budget``; ``key`` names the state (``"state"`` for ``advance``,
    ``"from"`` for ``timeline``) and ``state`` is where the first request
    starts. With a ``budget``, each request carries what is left of it, and
    the chain stops at an answer that is not done or once it is spent: the
    last answer's ``at`` then says how far the life got. ``call`` is the
    engine asked (``ask``).
    """
    from . import wire

    limit, items = limits()
    sizes = [len(wire.emit(fact)) for fact in facts]
    answers = []
    start = 0
    spent = 0
    while True:
        request = dict(base, facts=[], to=0)
        request[key] = state
        if budget is not None:
            request["budget"] = budget - spent
        # The request with no fact and ``to`` 0: a fact adds its bytes and a comma, ``to`` its digits.
        empty = len(wire.emit(request)) - 1
        end = start
        carried = 0
        while end < len(facts):
            stop = end
            added = 0
            while stop < len(facts) and facts[stop]["t"] == facts[end]["t"]:
                added += sizes[stop]
                stop += 1
            last = to if stop == len(facts) else facts[stop]["t"] - 1
            if stop - start > items or empty + len(str(last)) + carried + added + (stop - start - 1) > limit:
                break
            carried += added
            end = stop
        if end == start and start < len(facts):
            raise LifeRefused("limit", f"the facts of minute {facts[start]['t']} do not fit in one request")
        request["facts"] = facts[start:end]
        request["to"] = to if end == len(facts) else facts[end]["t"] - 1
        if len(wire.emit(request)) > limit:
            raise LifeRefused("limit", "a packed request is over the engine's input limit")
        answer = ask(request, call)
        answers.append(answer)
        state = answer["state"]
        start = end
        if budget is not None:
            spent += answer["work"]
            if not answer["done"] or (start < len(facts) and spent >= budget):
                return answers
        if start == len(facts):
            return answers


def target(being, conn, now):
    """The minute a view or a settle shows when none is asked for: the recorder's minute of ``now``.

    ``max(wall minute, latest recorded minute)``; a wall that cannot be read
    or is before the birth is refused ``clock``.
    """
    from . import membrane
    from .store import T_MAX

    wall = membrane.recorder_wall(now, being.birth_wall)
    return membrane.recorder_t(wall, being.birth_wall, conn.execute(T_MAX).fetchone()[0])


def _read(being, to):
    """Under the store's lock, once: the facts in order, the minute to show, and this engine's checkpoints by it."""
    from .store import _guarded

    store = being._store
    with store._lock:
        being._gate()
        conn = being._live()

        def read():
            ordered = being._in_order(conn)
            goal = to if to is not None else target(being, conn, store._read_clock())
            return ordered, goal, being._checkpoint_rows(conn, goal)

        return _guarded(conn, read)


def _last_at(times, t):
    """The index, in the ordered facts, of the last one at or before minute ``t``; 0 (the genesis) when none is."""
    return bisect.bisect_right(times, t, 1) - 1


def start_point(ordered, rows, stale=None):
    """``(index, state)``: where a view or a settle starts -- the latest usable checkpoint, else the genesis.

    ``ordered`` is ``in_order()``; ``rows`` are this engine's checkpoints,
    latest first. ``index`` is the position in ``ordered`` of the event the
    state runs through (0 and ``None`` for the genesis). A row whose event is
    no longer the last fact by its minute, or whose count of facts is not
    the number up to that event, is stale and skipped; one whose blob does
    not hold the state it names is refused ``divergence``. ``stale``, when a
    list, receives each row skipped for its count, ``(t, laws, through,
    state_hash)``: it still names the last event by its minute, so only its
    state says it is stale.
    """
    from . import wire
    from .store import StoreRefused, _inflate

    times = [fact["t"] for _seq, _eid, fact in ordered]
    position = {eid: index for index, (_seq, eid, _fact) in enumerate(ordered)}
    for t, laws, through, state_hash, blob in rows:
        last = _last_at(times, t)
        if position.get(through) != last:
            continue
        canonical = _inflate(blob, limits()[0])
        if canonical is None or hashlib.sha256(canonical).hexdigest() != state_hash:
            raise StoreRefused("divergence", f"the checkpoint at t={t} does not hold the state it names")
        try:
            state = wire.parse(canonical)
        except Exception:  # noqa: BLE001 - bytes the codec refuses are no state
            state = None
        law = state.get("law") if isinstance(state, dict) else None
        if not isinstance(law, dict) or not _is_int(state.get("at")) or state["at"] != t or law.get("v") != laws:
            raise StoreRefused("divergence", f"the checkpoint at t={t} holds a state of another minute or law")
        if state.get("n") != last:
            if stale is not None:
                stale.append((t, laws, through, state_hash))
            continue
        return last, state
    return 0, None


def _owed(state_law, goal, at):
    """What is still owed from ``at`` to ``goal``: whole days at the law's awake-day ceiling."""
    from . import lawfiles

    awake_day = lawfiles.law(state_law["name"])["work"]["ceilings"]["awake_day"]
    return -(-(goal - at) // DAY) * awake_day


def view(being, to=None, cap=None):
    """The being at minute ``to`` (default: the recorder's minute of now); read-only.

    ``cap`` bounds the engine's work; a view that does not get there is
    ``catching_up`` and shows the state it started from.
    """
    from .evolution import labels

    if to is not None and (not _is_int(to) or not 0 <= to <= MAX_INT):
        raise LifeRefused("bad_request", "to")
    if cap is not None and (not _is_int(cap) or not 1 <= cap <= MAX_INT):
        raise LifeRefused("bad_request", "budget")
    ordered, goal, rows = _read(being, to)
    start, state = start_point(ordered, rows)
    base = {"genesis": ordered[0][2], "op": "advance", "v": 1}
    facts = [fact for _seq, _eid, fact in ordered[start + 1:] if fact["t"] <= goal]
    answers = chain(base, "state", state, facts, goal, budget=MAX_INT if cap is None else cap)
    last = answers[-1]
    if last["done"] and last["at"] == goal:
        work, notes = _spent(answers)
        return View("current", last["state"], last["at"], last["hash"], last["env"], labels(being),
                    last["state"]["law"], being.provisional, 0, work, notes)
    stored = _start_answers(base, ordered, state, budget=MAX_INT if cap is None else cap)
    served = stored[-1]
    work, notes = _spent(answers + stored)
    law = served["state"]["law"]
    return View("catching_up", served["state"], served["at"], served["hash"], served["env"], labels(being), law,
                being.provisional, _owed(law, goal, served["at"]), work, notes)


def _start_answers(base, ordered, state, budget=MAX_INT):
    """The engine's answers that show the state a view starts from, at its own minute.

    The minute-0 state as a view of minute 0 shows it (the facts of minute 0
    folded) when there is no usable checkpoint, asked within ``budget``;
    else the checkpoint's own minute, with no fact: the engine lives no
    minute and says what it shows.
    """
    if state is None:
        zero = [fact for _seq, _eid, fact in ordered[1:] if fact["t"] == 0]
        return chain(base, "state", None, zero, 0, budget=budget)
    return [ask(dict(base, budget=1, facts=[], state=state, to=state["at"]))]


def stored(being, to=None, cap=None):
    """The state a view at minute ``to`` starts from, served as it is kept; read-only.

    The latest usable checkpoint at or before the minute (``start_point``),
    else the minute-0 state with the facts of minute 0 folded, shown at its
    own minute: what a ``catching_up`` view serves, as a ``View`` of status
    ``stored`` that owes nothing. A look serves it when the engine stops on
    a fault past it. ``cap`` bounds every request, as a view's does.
    """
    from .evolution import labels

    if to is not None and (not _is_int(to) or not 0 <= to <= MAX_INT):
        raise LifeRefused("bad_request", "to")
    if cap is not None and (not _is_int(cap) or not 1 <= cap <= MAX_INT):
        raise LifeRefused("bad_request", "budget")
    ordered, _goal, rows = _read(being, to)
    _start, state = start_point(ordered, rows)
    base = {"genesis": ordered[0][2], "op": "advance", "v": 1}
    answers = _start_answers(base, ordered, state, budget=MAX_INT if cap is None else cap)
    served = answers[-1]
    work, notes = _spent(answers)
    return View("stored", served["state"], served["at"], served["hash"], served["env"], labels(being),
                served["state"]["law"], being.provisional, 0, work, notes)


def _spent(answers):
    """``(work, notes)``: what the engine's answers cost and noted, all of them."""
    work = 0
    for answer in answers:
        work += answer["work"]
    return work, [note for answer in answers for note in answer["notes"]]


def retention_keep(midnights, today, policy):
    """The firing minutes the retention keeps, of ``midnights`` (``[[t, day, [y, mo, d]], ...]``, ascending).

    Relative to the local day ``today``: a midnight is kept when its day is
    one of the last ``daily``, a Monday of the last ``weekly`` weeks, the
    first of a month when ``monthly``, or when it is the latest.
    """
    daily = policy["daily"]
    weekly = policy["weekly"]
    monthly = policy["monthly"]
    kept = set()
    for t, day, civil in midnights:
        if day > today - daily or (day % 7 == MONDAY and day > today - 7 * weekly) or (monthly and civil[2] == 1):
            kept.add(t)
    if midnights:
        kept.add(midnights[-1][0])
    return kept


def settle(being, budget=None):
    """Keep the states at the local midnights the retention keeps, from the latest usable one on; ``Settled``.

    Each kept midnight after the start is advanced to in order and kept
    through ``checkpoint_put``; ``budget`` bounds the engine's work over the
    whole settle, and a settle it stops says so (``done`` false, with what is
    still owed). The rows a fact landed before their event made stale are
    dropped before anything is kept; one that gets through prunes every
    other checkpoint.
    """
    from . import evolution

    if budget is not None and (not _is_int(budget) or not 1 <= budget <= MAX_INT):
        raise LifeRefused("bad_request", "budget")
    ordered, goal, rows = _read(being, None)
    policy = being._store._life_settings()["checkpoints"]
    stale = []
    start, state = start_point(ordered, rows, stale)
    pruned = being._checkpoint_drop(stale) if stale else 0
    genesis = ordered[0][2]
    laws = [fact for _seq, _eid, fact in ordered[1:] if fact["kind"] in evolution.KINDS and fact["t"] <= goal]
    line = evolution.timeline_of(genesis, laws, goal, midnights_after=0)
    keep = retention_keep(line.midnights, line.state["day"], policy)
    times = [fact["t"] for _seq, _eid, fact in ordered]
    eids = {(fact["t"], fact["origin"], fact["oseq"]): eid for _seq, eid, fact in ordered}
    base = {"genesis": genesis, "op": "advance", "v": 1}
    facts = [fact for _seq, _eid, fact in ordered[start + 1:] if fact["t"] <= goal]
    left = MAX_INT if budget is None else budget
    at = 0 if state is None else state["at"]
    written = 0
    done = True
    cursor = 0
    for minute in sorted(t for t in keep if t > at):
        if left < 1:
            done = False
            break
        upto = cursor
        while upto < len(facts) and facts[upto]["t"] <= minute:
            upto += 1
        answers = chain(base, "state", state, facts[cursor:upto], minute, budget=left)
        for answer in answers:
            left -= answer["work"]
        last = answers[-1]
        if not last["done"] or last["at"] != minute:
            done = False
            break
        state = last["state"]
        cursor = upto
        through = ordered[0][1] if state["through"] is None else eids[tuple(state["through"])]
        being.checkpoint_put(minute, state["law"]["v"], through, state)
        written += 1
        at = minute
    if done:
        pruned += being.checkpoint_prune(keep={minute: ordered[_last_at(times, minute)][1] for minute in keep})
        return Settled(True, at, written, pruned, 0)
    law = genesis["body"]["laws"] if state is None else state["law"]
    return Settled(False, at, written, pruned, _owed(law, goal, at))


def _deep_read(being):
    """Under the store's lock and the mode gate, once: the facts in order, the minute shown, every kept state.

    The kept states are this engine's at any minute, ascending by minute,
    law version and event.
    """
    from .store import _guarded

    store = being._store
    with store._lock:
        being._gate()
        conn = being._live()

        def read():
            ordered = being._in_order(conn)
            goal = target(being, conn, store._read_clock())
            rows = being._checkpoint_rows(conn, MAX_INT)
            return ordered, goal, rows

        ordered, goal, rows = _guarded(conn, read)
    return ordered, goal, sorted(rows, key=lambda row: (row[0], row[1], row[2]))


def deep_verify(being):
    """Verify the chain from genesis, then replay every kept state and the state served now on the reference.

    The chain is verified first (a ``ChainRefused`` is raised as it is).
    Then, from one read, each kept state at or before the minute shown now
    is compared with the reference engine's replay from the genesis: one
    whose event is no longer the last fact by its minute, or whose count of
    facts is not the count up to that event, is stale and skipped; one whose
    blob does not hold the state its hash names is ``Diverged("blob")``; one
    the replay disagrees with is ``Diverged("kept")``. A kept state past the
    minute shown (the clock set back since) is counted as ahead and
    skipped. Last, the replay reaches the minute shown and is compared with
    the view the engine serves there (native when the handshake agreed):
    ``Diverged("served")`` when they differ. Nothing is written and nothing
    is repaired; a ``Deep`` says what was compared.
    """
    from . import engine, wire
    from .ref import protocol
    from .store import _inflate

    being.verify()
    ordered, goal, rows = _deep_read(being)
    reference = protocol.call
    times = [fact["t"] for _seq, _eid, fact in ordered]
    position = {eid: index for index, (_seq, eid, _fact) in enumerate(ordered)}
    facts = [fact for _seq, _eid, fact in ordered[1:]]
    base = {"genesis": ordered[0][2], "op": "advance", "v": 1}
    state = None
    cursor = 0
    kept = agreed = stale = ahead = 0

    def replay(to):
        nonlocal state, cursor
        upto = cursor
        while upto < len(facts) and facts[upto]["t"] <= to:
            upto += 1
        answers = chain(base, "state", state, facts[cursor:upto], to, budget=MAX_INT, call=reference)
        state = answers[-1]["state"]
        cursor = upto
        return answers[-1]

    for t, laws, through, state_hash, blob in rows:
        if t > goal:
            ahead += 1
            continue
        kept += 1
        last = _last_at(times, t)
        if position.get(through) != last:
            stale += 1
            continue
        # A blob that does not inflate within the engine's input limit holds no state the engine could be given.
        canonical = _inflate(blob, limits()[0])
        if canonical is None or hashlib.sha256(canonical).hexdigest() != state_hash:
            raise Diverged("blob", t, laws)
        try:
            held = wire.parse(canonical)
        except Exception:  # noqa: BLE001 - bytes the codec refuses hold no state
            raise Diverged("blob", t, laws) from None
        if not isinstance(held, dict):
            raise Diverged("blob", t, laws)
        if held.get("n") != last:
            stale += 1
            continue
        if replay(t)["hash"] != state_hash:
            raise Diverged("kept", t, laws)
        agreed += 1
    reached = replay(goal)
    used = "native" if engine.native_in_use() else "reference"
    served = view(being, to=goal)
    if served.hash != reached["hash"]:
        raise Diverged("served", goal, reached["state"]["law"]["v"], engine=used)
    return Deep(len(ordered), goal // DAY, kept, agreed, stale, ahead, used, protocol.ENGINE_VERSION)
