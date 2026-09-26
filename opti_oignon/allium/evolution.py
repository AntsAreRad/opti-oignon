"""The laws of a being as the platform reads and writes them: the proposal, the timeline, law updates, pins.

A being's law and params are frozen into its genesis at its birth. They come
from the proposal in the settings file (``proposal`` for the params, checked
against the law's ranges; ``sowing`` for the hemisphere, the day-length band
and the weather, against the genesis's symbols), and before a genesis is
written the engine is asked to live its first minute (``dry_run``): a genesis
the engine refuses is never written. A present but malformed value of the
proposal is refused by name where it is used, never replaced by a default.
After the birth the proposal moves nothing by itself: a replay never reads
it, and a living being's params change only through a law update.

``timeline_at`` is the platform's only way to learn what is in force at a
minute -- the law and its params, the pin, a pending law update and the
local offset. It reads the trunk's facts of the kinds the law timeline folds
(``tz``, ``evolve``, ``laws_pin``, ``laws_unpin``), adds the ones the
current transaction plans to write, and asks the engine's ``timeline``
operation from the genesis, packed as a view is. The platform never folds
these kinds itself: the engine's fold is the one the reducer uses.

The laws writer. ``evolve``, ``laws_pin`` and ``laws_unpin`` are written by
the laws writer alone, through the store's journaled write and the
membrane's admission under the producer ``laws``, never by a generic append.
Each decides inside the write's own transaction, on the timeline at the
write's minute with the recorder's offset fact already planned:

* ``pin`` and ``unpin`` are refused ``pinned`` when the being is pinned
  already, ``unpinned`` when it is not;
* ``apply`` writes the law update a ``diff`` showed, checked in this order:
  the verb's surface and the owner, the mode, ``pinned``, the proposal
  (``params``), ``nothing`` to change, and the ``confirm`` the diff gave. A
  pending law update is extended, never cancelled: its law is kept, the
  proposal becomes its params;
* ``automatic`` is the law update a generic append carries, right after the
  gesture, when this device is the being's home, the gesture came from a
  surface the law update's row holds, the being is not pinned, a carried
  stable law follows the one in force, no pending update already goes to
  it, and the day's budget of law updates is not spent. It carries the
  params in force, or the params of an update already pending (one the
  owner confirmed is extended, never cancelled): the proposal reaches a
  being only through ``apply``.

Every law update takes effect at the next local midnight after its minute,
under the offset the minute ends with. A view writes nothing: ``diff`` and
``law_state`` only read, and the automatic update lives only in the
journaled write.

Refusals: ``LawsRefused`` with the codes ``law``, ``params``, ``pinned``,
``unpinned``, ``nothing``, ``confirm`` and ``sowing``. None is a wire code.

Nothing is imported at module level but the standard library.
"""

import hashlib
from typing import NamedTuple

checkpoint_before_apply = True

MAX_INT = (1 << 53) - 1
KINDS = ("evolve", "laws_pin", "laws_unpin", "tz")
# What a genesis reads when the proposal is silent on a sowing field.
SOWING_DEFAULTS = {"band": "long", "hemisphere": "north", "weather": "garden"}


class LawsRefused(ValueError):
    """A law write, a proposal or a sowing refused by name; ``code`` is one of ``CODES``. Nothing was written."""

    CODES = ("law", "params", "pinned", "unpinned", "nothing", "confirm", "sowing")

    def __init__(self, code, detail=""):
        if code not in self.CODES:
            raise ValueError(f"unknown laws refusal: {code}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class Timeline(NamedTuple):
    """The law timeline at a minute: its state, the next local midnight, notes, and firing minutes when asked."""

    state: dict
    midnight: int
    notes: list
    midnights: object


class Diff(NamedTuple):
    """What a law update from the proposal would change: the law it goes to, the params before and after."""

    law: dict
    current: dict
    proposed: dict
    changed: list
    confirm: str


class LawState(NamedTuple):
    """The laws of a being at a minute, as the engine's timeline gives them, and its label."""

    at: int
    day: int
    law: dict
    params: dict
    pending: object
    pinned: bool
    seen: list
    tz: int
    midnight: int
    provisional: bool
    labels: tuple


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def labels(being):
    """``("prototype",)`` when the being's genesis law is provisional: the label follows the genesis."""
    return ("prototype",) if being.provisional else ()


# ---------------------------------------------------------------------------
# The proposal and sowing
# ---------------------------------------------------------------------------
def proposal(pin, raw):
    """The params the proposal ``raw`` gives under law ``pin`` (``membrane.law_pin``); refused ``params`` by name.

    ``raw`` is the settings' proposal (``settings.laws`` or
    ``settings.normalise_laws``). A missing value reads the law's default; a
    value that is not an integer, a boolean, one outside the law's range or
    one the settings could not read is refused, naming where it is in the
    file.
    """
    from . import settings

    given = raw.get("params") if isinstance(raw, dict) else None
    given = given if isinstance(given, dict) else {}
    out = {}
    for name in sorted(pin["params"]):
        spec = pin["params"][name]
        section, key = settings.PARAM_KEYS.get(name, ("params", name))
        where = f"{section}.{key}"
        value = given.get(name)
        if isinstance(value, settings.Malformed):
            raise LawsRefused("params", value.detail)
        if value is None:
            out[name] = spec["default"]
            continue
        if isinstance(value, bool):
            raise LawsRefused("params", f"{where} is a boolean, not an integer")
        if not isinstance(value, int):
            raise LawsRefused("params", f"{where} is not an integer")
        if not spec["lo"] <= value <= spec["hi"]:
            raise LawsRefused("params", f"{where} {value} is outside {spec['lo']}..={spec['hi']}")
        out[name] = value
    return out


def sowing(pin, raw, *, hemisphere=None, band=None, weather=None):
    """The hemisphere, band and weather a genesis under law ``pin`` freezes; refused ``sowing`` by name.

    A field the caller names wins; otherwise the proposal's, otherwise north,
    long and garden. Each must be one of the genesis's own symbols.
    """
    from . import settings

    schema = pin["table"]["kinds"]["genesis"]["body"]
    named = {"band": band, "hemisphere": hemisphere, "weather": weather}
    given = raw.get("sowing") if isinstance(raw, dict) else None
    given = given if isinstance(given, dict) else {}
    out = {}
    for field in sorted(named):
        if named[field] is not None:
            value, where = named[field], field
        else:
            section, key = settings.SOWING_KEYS[field]
            value, where = given.get(field), f"{section}.{key}"
            if isinstance(value, settings.Malformed):
                raise LawsRefused("sowing", value.detail)
            if value is None:
                value = SOWING_DEFAULTS[field]
        symbols = schema[field]["of"]
        if not isinstance(value, str) or value not in symbols:
            raise LawsRefused("sowing", f"{where} {value!r} is not one of {', '.join(symbols)}")
        out[field] = value
    return out


def dry_run(genesis):
    """Ask the engine to live the first minute of ``genesis`` (an envelope); a refusal is ``LawsRefused``.

    A law the engine does not carry or cannot live (``unknown_law``) is
    ``law``; any other refusal is ``sowing``. What it computes is discarded.
    """
    from . import engine, wire

    request = {"budget": 1, "facts": [], "genesis": genesis, "op": "advance", "state": None, "to": 0, "v": 1}
    answer = wire.parse(engine.call(wire.emit(request)))
    if "refused" in answer:
        code = answer["refused"]
        detail = f"{code} {answer.get('detail', '')}".strip()
        raise LawsRefused("law" if code == "unknown_law" else "sowing", detail)


# ---------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------
def timeline_at(being, t, planned=(), midnights_after=None, conn=None):
    """What is in force at minute ``t``, with ``planned`` facts of the current transaction folded in.

    ``conn`` is the connection of a transaction in progress, whose caller
    holds the store's lock; the engine is then asked inside it, since a
    write decides on what it reads. Without one, the facts are read under
    the lock and the mode gate, as a view reads, and the engine is asked
    once the lock is let go. ``midnights_after`` asks for the daily firing
    minutes after it, as the engine lists them.
    """
    if conn is None:
        genesis, facts = _read_law_facts(being)
    else:
        genesis, facts = being._law_facts(conn)
    facts = [fact for fact in list(facts) + list(planned) if fact["t"] <= t]
    facts.sort(key=lambda fact: (fact["t"], fact["origin"], fact["oseq"]))
    return timeline_of(genesis, facts, t, midnights_after=midnights_after)


def _read_law_facts(being):
    """Under the store's lock and the mode gate: the genesis and the facts the law timeline folds."""
    from .store import _guarded

    with being._store._lock:
        being._gate()
        conn = being._live()
        return _guarded(conn, lambda: being._law_facts(conn))


def timeline_of(genesis, facts, t, midnights_after=None):
    """The engine's law timeline at minute ``t``, from the genesis and ``facts`` already read.

    ``facts`` are the folded kinds in canonical order, none after ``t``: a
    caller that read the whole trunk (a settle) passes them without reading
    the store again. Chained through the packer, as a view is.
    """
    from . import life

    base = {"genesis": genesis, "op": "timeline", "v": 1}
    if midnights_after is not None:
        base["midnights_after"] = midnights_after
    answers = life.chain(base, "from", None, facts, t)
    last = answers[-1]
    notes = [note for answer in answers for note in answer["notes"]]
    midnights = None if midnights_after is None else [m for answer in answers for m in answer["midnights"]]
    return Timeline(last["state"], last["midnight"], notes, midnights)


def _look(being, to):
    """A look at the law timeline: ``(t, timeline)``; nothing is written.

    Under the store's lock, once: the mode gate, the minute (``to``, or the
    one a view shows) and the facts the timeline folds. The engine is asked
    after the lock is let go, as a view asks it.
    """
    from . import life
    from .store import _guarded

    if to is not None and (not _is_int(to) or not 0 <= to <= MAX_INT):
        raise life.LifeRefused("bad_request", "to")
    store = being._store
    with store._lock:
        being._gate()
        conn = being._live()

        def read():
            t = to if to is not None else life.target(being, conn, store._read_clock())
            genesis, facts = being._law_facts(conn)
            return t, genesis, facts

        t, genesis, facts = _guarded(conn, read)
    return t, timeline_of(genesis, [fact for fact in facts if fact["t"] <= t], t)


def law_state(being, to=None):
    """The laws in force at minute ``to`` (default: the minute a view shows), as ``LawState``; nothing is written."""
    _t, line = _look(being, to)
    st = line.state
    return LawState(st["at"], st["day"], st["law"], st["params"], st["pending"], st["pinned"], st["seen"],
                    st["tz"], line.midnight, being.provisional, labels(being))


# ---------------------------------------------------------------------------
# Law updates
# ---------------------------------------------------------------------------
def confirmation(current, law, proposed):
    """The confirmation of a law update: the first 16 hex digits of the digest of what it changes."""
    from . import wire

    data = wire.emit({"from": current, "law": {"name": law["name"], "sha256": law["sha256"]}, "to": proposed})
    return hashlib.sha256(data).hexdigest()[:16]


def _diff(line, raw):
    """The diff at a timeline: the law an update goes to (the pending one's, else the law in force)."""
    from . import membrane

    st = line.state
    pending = st["pending"]
    law = dict(pending["to"]) if pending is not None else dict(st["law"])
    current = dict(pending["params"]) if pending is not None else dict(st["params"])
    proposed = proposal(membrane.law_pin(law["name"]), raw)
    changed = sorted(name for name in proposed if proposed[name] != current.get(name))
    return Diff(law, current, proposed, changed, confirmation(current, law, proposed))


def diff(being, to=None):
    """What a law update from the settings' proposal would change at minute ``to`` (default: a view's); a view.

    ``current`` is the pending update's params when one is pending, else the
    params in force; a proposal that is not one is refused ``params``.
    """
    raw = being._store._laws_settings()
    return _diff(_look(being, to)[1], raw)


def apply(being, transport, confirm):
    """Write the law update the proposal gives, once ``confirm`` is the diff's; ``Appended`` or ``Dropped``."""
    raw = being._store._laws_settings()

    def plan(line):
        st = line.state
        if st["pinned"]:
            raise LawsRefused("pinned", "the being is pinned to the law and params in force")
        found = _diff(line, raw)
        if not found.changed:
            raise LawsRefused("nothing", "the proposal is what is already in force or pending")
        if confirm != found.confirm:
            raise LawsRefused("confirm", "the confirmation is not the one the diff shows now")
        law = st["law"]
        return {"effective_from": line.midnight, "from": {"name": law["name"], "sha256": law["sha256"]},
                "params": found.proposed, "to": found.law}

    return being._journal("evolve", None, transport=transport, grant_ref=None, payload=None, producer="laws",
                          plan=plan, verb="laws_apply")


def pin(being, transport):
    """Pin the law and params in force: no law update applies until the unpin. Refused ``pinned`` when pinned."""

    def plan(line):
        if line.state["pinned"]:
            raise LawsRefused("pinned", "the being is already pinned")
        return {}

    return being._journal("laws_pin", {}, transport=transport, grant_ref=None, payload=None, producer="laws",
                          plan=plan)


def unpin(being, transport):
    """Lift the pin. Refused ``unpinned`` when the being is not pinned."""

    def plan(line):
        if not line.state["pinned"]:
            raise LawsRefused("unpinned", "the being is not pinned")
        return {}

    return being._journal("laws_unpin", {}, transport=transport, grant_ref=None, payload=None, producer="laws",
                          plan=plan)


def automatic(being, conn, t, in_force, surface, planned):
    """The body of the law update a generic append at minute ``t`` carries, or ``None``.

    ``in_force`` is the timeline at ``t``; ``planned()`` gives it with the
    recorder's offset fact of this transaction folded in, whose next local
    midnight the update takes effect at. Only a device that is the being's
    home writes it, and only after a gesture from a surface the law update's
    row holds.
    """
    from . import lawfiles, membrane

    if surface not in membrane.MATRIX["evolve"]:
        return None
    if being.origin != being._verified.genesis_origin:
        return None
    st = in_force.state
    if st["pinned"]:
        return None
    law = st["law"]
    found = lawfiles.successor(law["name"], law["sha256"])
    if found is None:
        return None
    pending = st["pending"]
    if pending is not None and pending["to"] == found:
        return None
    if being._day_count(conn, "evolve", membrane.day_of(t)) >= being._verified.pin["budgets"]["evolve"]:
        return None
    # A pending update the owner confirmed with ``apply`` is extended, never cancelled: its params ride with
    # the law update (the successor's ranges hold the old law's, so they stay in range); else the params in
    # force.
    params = pending["params"] if pending is not None else st["params"]
    return {"effective_from": planned().midnight, "from": {"name": law["name"], "sha256": law["sha256"]},
            "params": dict(params), "to": found}
