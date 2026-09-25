"""The life: a being's state, advanced from its genesis and facts, and the law timeline.

The reference. The Rust twin answers every request with the same bytes.

``advance`` folds a life. From the genesis (``state: null``) or from a state
it returned before, it takes the facts after that state, in canonical order
``(t, origin, oseq)``, and lives every minute up to ``to``, or until its
``budget`` of work units runs out after a whole processed minute or a whole
day of the fast path. The state it returns at minute ``at`` depends only on
the genesis, the facts with ``t <= at`` and ``at``: cut a life anywhere,
slice it under any budget, and the states and the summed work are the same.

Time. A minute of life ``t`` lies inside UTC minute ``b + t``, ``b`` the
birth minute; its local minute adds the offset in force, which ``tz`` facts
change. A fast boundary is a minute whose UTC minute is a multiple of 15,
so every local midnight is one. The event minutes are the fast boundaries,
the minutes that hold facts and the ``effective_from`` of a pending
``evolve``. Processing an event minute ``e``:

1. the visit (1 unit);
2. a pending ``evolve`` whose ``effective_from`` has come is applied;
3. the facts of ``e``, each one unit, in three passes, each in canonical
   order: the ``tz`` facts (the offset), then the law kinds ``laws_pin``,
   ``laws_unpin`` and ``evolve`` (checked against the offset after the first
   pass), then every other kind, handed to each organ in organ order. Each
   fact first passes the law-field check and its day's budget or cap; one
   over it is noted and not folded. A wake during the third pass runs the
   quiescent organs' ``jump``;
4. at a fast boundary, the fast layer: awake, the environment (the light)
   and each organ's fast step; dormant, the quiescent organs are not called;
5. when the local day is above the high-water mark ``state.day``, the daily
   layer: each organ's daily step, then the mark moves to that day, and a
   wake runs the quiescent organs' ``jump``. A day index skipped by a jump
   east is never lived, and one repeated by a jump west is not lived again;
6. the night layer, empty for now;
7. every organ publishes its channels on the bus, a quiescent organ 0 while
   the being sleeps. Every read in steps 3 to 5 is of the bus published at
   the previous processed minute, so the order of the organs never matters.

Minute 0 folds the genesis into the first state and is processed like any
other, except that the high-water mark is the local day of minute 0, set
after its ``tz`` facts, so the daily layer never fires there.

The fast path. While the being is dormant, nothing moves between daily
firings but the soil and the stage, and those only once a day: production
lives such a span a local midnight at a time, running only the non-quiescent
organs' daily steps (one unit for the day, and theirs). The span ends
before the next fact and before a pending ``effective_from``, where the
stepped path takes over. That minute costs a visit: at a local midnight it
stands in for the fast path's day, but an ``effective_from`` a later ``tz``
fact moved off its midnight is met by one visit more even while the being
sleeps, and the law's ``dormant_day`` ceiling counts it. A due ``evolve``
adds no visit only awake, where every fast boundary is visited anyway.
``probe.fast_path: false`` steps every minute instead, and
calls the quiescent organs as identities; the states are the same.

Laws. The law in force, its params and the pin live in the state. An
``evolve`` registers a pending change at the next local midnight, checked
against the carried laws; at its minute it is applied unless the being is
pinned or the law it starts from is no longer the one in force. A fact
carrying a law version never in force refuses the call; one in force before,
but not now, is noted and folded. Conflicts are notes, never refusals: a
refusal would replay forever.

``timeline`` folds the law kinds alone (``tz``, ``evolve``, ``laws_pin``,
``laws_unpin``) with the same functions, and computes the daily firing
minutes arithmetically instead of visiting minutes.
"""

import hashlib

from .. import fx, wire
from ..wire import Refused
from . import civil, journal, lawdata
from .organs import chem, clock, genome, soil, stage, weather
from .organs import compile as organ_compile

checkpoint_before_apply = True

DAY = civil.DAY
FAST = civil.FAST
ONE = fx.ONE
CMAX = fx.CMAX
I32_MAX = fx.I32_MAX
MAX_INT = wire.MAX_INT
T_MAX = civil.T_MAX
SCHEMA = 1
# The units the reducer counts itself; every organ step counts what its primitives count.
VISIT = 1
FACT = 1
FAST_PATH_DAY = 1
DOMAINS = weather.DOMAINS
MODULES = {"chem": chem, "clock": clock, "soil": soil, "stage": stage}
LAW_KINDS = ("evolve", "laws_pin", "laws_unpin")
REDUCER_KINDS = ("evolve", "laws_pin", "laws_unpin", "tz")
# The closed set of note codes a response may carry: diagnostics, never part of the state.
NOTE_CODES = ("budget", "evolve_from", "evolve_params", "evolve_pinned", "evolve_superseded", "evolve_when",
              "laws_stale", "pin_twice", "truncated", "unpin_unpinned")
STATE_KEYS = ("at", "being", "budget", "bus", "day", "genome", "law", "n", "organs", "params", "pending",
              "pinned", "schema", "seen", "through", "tz")
TSTATE_KEYS = ("at", "budget", "day", "law", "params", "pending", "pinned", "schema", "seen", "through", "tz")
CHANNELS = ("circadian", "dormant", "metab", "moisture")
CAUSES = ("dry", "none", "winter")
PROBE_KEYS = ("order", "fast_path", "trace")
_HEX = "0123456789abcdef"


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _int_in(value, low, high):
    return _is_int(value) and low <= value <= high


def _is_hex(value, length):
    if not isinstance(value, str) or len(value) != length:
        return False
    for char in value:
        if char not in _HEX:
            return False
    return True


def _exactly(value, names):
    if not isinstance(value, dict):
        return False
    for name in value:
        if name not in names:
            return False
    for name in names:
        if name not in value:
            return False
    return True


def _fields(request, required, optional=()):
    for name in request:
        if name not in required and name not in optional:
            raise Refused("bad_request", "fields")
    for name in required:
        if name not in request:
            raise Refused("bad_request", "fields")


def _list(request, key, items):
    value = request[key]
    if not isinstance(value, list):
        raise Refused("bad_request", key)
    if len(value) > items:
        raise Refused("limit", "items")
    return value


def _fast_path_fires(day, high):
    """Whether a local midnight of the fast path fires: its day is above the high-water mark."""
    return day > high


# ---------------------------------------------------------------------------
# The state and its schema
# ---------------------------------------------------------------------------

def _triple(value, low, high):
    if not isinstance(value, list) or len(value) != 3:
        return False
    for item in value:
        if not _int_in(item, low, high):
            return False
    return True


def _chem_ok(org):
    if not _exactly(org, ("burnt", "fructan", "k", "made", "metab", "sugar")):
        return False
    k = org["k"]
    if not _exactly(k, ("hy", "km_ps", "km_r", "ps", "r", "sy")):
        return False
    for name in ("hy", "ps", "r", "sy"):
        if not _int_in(k[name], 0, ONE):
            return False
    for name in ("km_ps", "km_r"):
        if not _int_in(k[name], 1, CMAX):
            return False
    for name in ("burnt", "fructan", "made", "metab", "sugar"):
        if not _int_in(org[name], 0, MAX_INT):
            return False
    return True


def _clock_ok(org):
    if not _exactly(org, ("k", "p")) or not _triple(org["p"], 0, CMAX):
        return False
    k = org["k"]
    if not _exactly(k, ("alpha", "beta", "k", "light", "n")):
        return False
    return (_triple(k["alpha"], 0, I32_MAX) and _triple(k["beta"], 0, I32_MAX) and _triple(k["k"], 1, CMAX)
            and _triple(k["n"], fx.N_MIN, fx.N_MAX) and _int_in(k["light"], 0, I32_MAX))


def _stage_ok(org):
    if not _exactly(org, ("cause", "dormant", "dry", "rest", "season", "since")):
        return False
    return (isinstance(org["cause"], str) and org["cause"] in CAUSES and isinstance(org["dormant"], bool)
            and _int_in(org["dry"], 0, MAX_INT) and _int_in(org["rest"], 0, MAX_INT)
            and _int_in(org["season"], 0, civil.SEASONS - 1) and _int_in(org["since"], 0, T_MAX))


def _organs_ok(organs, table):
    if not _exactly(organs, ("chem", "clock", "soil", "stage")):
        return False
    soil_org = organs["soil"]
    return (_chem_ok(organs["chem"]) and _clock_ok(organs["clock"]) and _stage_ok(organs["stage"])
            and _exactly(soil_org, ("m",)) and _int_in(soil_org["m"], 0, MAX_INT))


def _body_ok(schema, value):
    try:
        journal.check_body(schema, value)
    except Refused:
        return False
    return True


def _budget_ok(budget, table):
    if not _exactly(budget, ("counts", "day")) or not _int_in(budget["day"], 0, MAX_INT):
        return False
    counts = budget["counts"]
    if not isinstance(counts, dict):
        return False
    kinds = table["kinds"]
    for kind, count in counts.items():
        entry = kinds.get(kind)
        if entry is None or entry["scope"] != "trunk" or kind == "genesis" or not _int_in(count, 1, MAX_INT):
            return False
    return True


def _bus_ok(bus, table):
    if not _exactly(bus, CHANNELS):
        return False
    for name in CHANNELS:
        if not _int_in(bus[name], 0, ONE):
            return False
    return True


def _law_ok(law, table):
    return (_exactly(law, ("name", "sha256", "v")) and isinstance(law["name"], str) and 1 <= len(law["name"]) <= 32
            and _is_hex(law["sha256"], 64) and _int_in(law["v"], 0, 65535))


def _seen_ok(seen, table):
    if not isinstance(seen, list) or not seen:
        return False
    last = -1
    for v in seen:
        if not _int_in(v, 0, 65535) or v <= last:
            return False
        last = v
    return True


def _through_ok(through, table):
    if through is None:
        return True
    return (isinstance(through, list) and len(through) == 3 and _int_in(through[0], 0, T_MAX)
            and _is_hex(through[1], 16) and _int_in(through[2], 0, MAX_INT))


_STATE_CHECKS = {
    "at": lambda value, table: _int_in(value, 0, T_MAX),
    "being": lambda value, table: _is_hex(value, 32),
    "budget": _budget_ok,
    "bus": _bus_ok,
    "day": lambda value, table: _int_in(value, 0, MAX_INT),
    "genome": lambda value, table: _is_hex(value, 64),
    "law": _law_ok,
    "n": lambda value, table: _int_in(value, 0, MAX_INT),
    "organs": _organs_ok,
    "params": lambda value, table: _body_ok(lawdata.params_schema(table), value),
    "pending": lambda value, table: value is None or _body_ok(table["kinds"]["evolve"]["body"], value),
    "pinned": lambda value, table: isinstance(value, bool),
    "schema": lambda value, table: _is_int(value) and value == SCHEMA,
    "seen": _seen_ok,
    "through": _through_ok,
    "tz": lambda value, table: _is_int(value) and civil.offset_ok(value),
}


def _state_defect(value, keys, table):
    """The first defect of a state (or a timeline state) as the detail of its refusal, ``None`` when it is whole.

    The key set comes first (``state fields``), then each key in its sorted
    order (``state <key>``), then two rules across keys: the law in force is
    among the versions seen (``state seen``), and the last fact folded is not
    after the state's minute (``state through``).
    """
    if not isinstance(value, dict):
        return "state"
    if not _exactly(value, keys):
        return "state fields"
    for key in keys:
        if not _STATE_CHECKS[key](value[key], table):
            return "state " + key
    if value["law"]["v"] not in value["seen"]:
        return "state seen"
    if value["through"] is not None and value["through"][0] > value["at"]:
        return "state through"
    return None


def _bounds_ok(organs, life):
    """The stocks and levels of a state within the bounds of the law in force."""
    c = life.constants
    reserve = organs["chem"]
    rest_max = c["stage"]["rest_max"]
    return (reserve["sugar"] <= c["chem"]["sugar_max"]
            and c["chem"]["core"] <= reserve["fructan"] <= c["chem"]["fructan_max"]
            and organs["soil"]["m"] <= c["soil"]["m_max"]
            and organs["stage"]["dry"] <= rest_max and organs["stage"]["rest"] <= rest_max)


def _state_law(law):
    """The carried law a state names as in force: refused by name, digest or version, then by soundness."""
    law_value, digest = lawdata.law_file(law["name"])
    if law["sha256"] != digest:
        raise Refused("unknown_law", "law digest")
    if law["v"] != law_value.get("version"):
        raise Refused("unknown_law", "law version")
    return lawdata.life(law["name"])


# ---------------------------------------------------------------------------
# The checks of a request, in the order that decides which defect is named
# ---------------------------------------------------------------------------

def _genesis(request):
    """The genesis checked against the carried law it names; ``(genesis, life)``."""
    genesis = request["genesis"]
    journal.check_envelope(genesis, False)
    if genesis["kind"] != "genesis" or genesis["t"] != 0 or genesis["oseq"] != 0:
        raise Refused("bad_fact", "genesis")
    body = genesis["body"]
    laws = body.get("laws")
    if not isinstance(laws, dict) or not isinstance(laws.get("name"), str) or not isinstance(laws.get("sha256"), str):
        raise Refused("bad_fact", "genesis laws")
    _law, digest = lawdata.law_file(laws["name"])
    if laws["sha256"] != digest:
        raise Refused("unknown_law", "law digest")
    life = lawdata.life(laws["name"])
    journal.check_body(life.table["kinds"]["genesis"]["body"], body)
    if laws["v"] != life.version or laws["provisional"] != life.provisional:
        raise Refused("bad_fact", "genesis laws")
    birth = body["birth"]
    if birth["wall"] < civil.WALL_MIN or birth["tz"] % civil.TZ_STEP:
        raise Refused("bad_fact", "birth")
    name = lawdata.params_defect(laws["params"], life)
    if name is not None:
        raise Refused("bad_fact", "params " + name)
    if genesis["laws"] != laws["v"]:
        raise Refused("chain", "laws")
    return genesis, life


def _to(request):
    to = request["to"]
    if not _int_in(to, 0, T_MAX):
        raise Refused("bad_request", "to")
    return to


def _probe(request, life):
    """``(order, fast_path, trace)`` from the optional probe; the default is the law's order, the fast path, no trace."""
    if "probe" not in request:
        return life.organs, True, False
    probe = request["probe"]
    if not isinstance(probe, dict):
        raise Refused("bad_request", "probe")
    for key in probe:
        if key not in PROBE_KEYS:
            raise Refused("bad_request", "probe")
    order = life.organs
    if "order" in probe:
        order = probe["order"]
        if not lawdata._distinct_names(order) or sorted(order) != sorted(life.organs):
            raise Refused("bad_request", "probe")
        order = tuple(order)
    for key in ("fast_path", "trace"):
        if key in probe and not isinstance(probe[key], bool):
            raise Refused("bad_request", "probe")
    return order, probe.get("fast_path", True), probe.get("trace", False)


def _state(request, genesis, life, to, keys):
    """The state a request resumes from, checked, and the law in force; ``(None, life)`` from the genesis."""
    value = request["state" if keys is STATE_KEYS else "from"]
    if value is None:
        return None, life
    defect = _state_defect(value, keys, life.table)
    if defect is not None:
        raise Refused("bad_request", defect)
    if keys is STATE_KEYS and value["being"] != genesis["being"]:
        raise Refused("bad_request", "state being")
    in_force = _state_law(value["law"])
    if keys is STATE_KEYS and not _bounds_ok(value["organs"], in_force):
        raise Refused("bad_request", "state organs")
    if to < value["at"]:
        raise Refused("bad_request", "to")
    return value, in_force


def _facts(request, genesis, life, state, to, items, timeline):
    """The facts of a request, each checked, then in canonical order after the state, and none after ``to``."""
    facts = _list(request, "facts", items)
    being = genesis["being"]
    table = life.table
    floor = None
    if state is not None:
        through = state["through"]
        floor = (state["at"], None) if through is None else (state["at"], tuple(through))
    previous = None
    for fact in facts:
        body = fact.get("body") if isinstance(fact, dict) else None
        journal.check_envelope(fact, isinstance(body, str))
        t = fact["t"]
        if not 0 <= t <= T_MAX:
            raise Refused("bad_fact", "t")
        if isinstance(body, dict) and len(wire.emit(body)) > journal.BODY_LIMIT:
            raise Refused("limit", "body size")
        if fact["being"] != being:
            raise Refused("bad_fact", "being")
        journal.check_fact(fact, table)
        if timeline and fact["kind"] not in REDUCER_KINDS:
            raise Refused("bad_request", "timeline kind")
        key = (t, fact["origin"], fact["oseq"])
        if floor is not None and (t <= floor[0] or (floor[1] is not None and key <= floor[1])):
            raise Refused("chain", "order")
        if previous is not None and key <= previous:
            raise Refused("chain", "order")
        if t > to:
            raise Refused("bad_request", "to")
        previous = key
    return facts


# ---------------------------------------------------------------------------
# The fold both drivers share: a fact consumed, the law kinds, the due evolve
# ---------------------------------------------------------------------------

def _approach(target, source):
    """The carried law a change goes to, from the law it names as its source; refused by name otherwise.

    The target is carried, under its digest and its version; a change of
    law (another name or digest than the source) is a migration the target
    declares (``successor_ok``). A law this engine does not carry, or cannot
    migrate to, is never approached: neither when an ``evolve`` is
    registered, nor when a pending change read from a state comes due.
    """
    target_value, digest = lawdata.law_file(target["name"])
    if target["sha256"] != digest:
        raise Refused("unknown_law", "law digest")
    if target["v"] != target_value.get("version"):
        raise Refused("unknown_law", "law version")
    target_life = lawdata.life(target["name"])
    if (target["name"], target["sha256"]) != (source["name"], source["sha256"]) \
            and not lawdata.successor_ok(target_life, source):
        raise Refused("unknown_law", "migration")
    return target_life


class _Fold:
    """A state and the law in force, folding facts; ``advance`` and ``timeline`` differ in their drivers only."""

    def __init__(self, state, life, b, work):
        self.state = state
        self.life = life
        self.b = b
        self.work = work
        self.notes = []
        self.facts = 0
        self.noted = 0

    def note(self, code, t):
        self.notes.append({"code": code, "t": t})

    def consume(self, fact):
        """One fact consumed (one unit): its law field, then its day's budget or cap; whether it is folded."""
        st = self.state
        self.work.units += FACT
        self.facts += 1
        laws = fact["laws"]
        if laws not in st["seen"]:
            raise Refused("chain", "laws")
        kind = fact["kind"]
        t = fact["t"]
        budget = st["budget"]
        day = t // DAY
        if budget["day"] != day:
            budget = {"counts": {}, "day": day}
            st["budget"] = budget
        counts = budget["counts"]
        count = counts.get(kind, 0)
        if count >= self.life.limits[kind]:
            self.note("budget", t)
            self.noted += 1
            return False
        counts[kind] = count + 1
        if laws != st["law"]["v"]:
            self.note("laws_stale", t)
        return True

    def tz_fact(self, fact):
        """The first pass: a ``tz`` fact moves the offset in force."""
        if self.consume(fact):
            self.state["tz"] = fact["body"]["quarters"] * civil.TZ_STEP

    def law_fact(self, fact, e):
        """The second pass: a pin, an unpin, or an ``evolve`` registered against the offset after the first pass."""
        if not self.consume(fact):
            return
        st = self.state
        kind = fact["kind"]
        if kind == "laws_pin":
            if st["pinned"]:
                self.note("pin_twice", e)
                self.noted += 1
            else:
                st["pinned"] = True
            return
        if kind == "laws_unpin":
            if not st["pinned"]:
                self.note("unpin_unpinned", e)
                self.noted += 1
            else:
                st["pinned"] = False
            return
        self._register(fact["body"], e)

    def _register(self, body, e):
        target = body["to"]
        source = body["from"]
        target_life = _approach(target, source)
        st = self.state
        if body["effective_from"] != civil.next_midnight(self.b, e, st["tz"]):
            self.note("evolve_when", e)
            self.noted += 1
            return
        if lawdata.params_defect(body["params"], target_life) is not None:
            self.note("evolve_params", e)
            self.noted += 1
            return
        if st["pending"] is not None:
            self.note("evolve_superseded", e)
        st["pending"] = {
            "effective_from": body["effective_from"],
            "from": {"name": source["name"], "sha256": source["sha256"]},
            "params": dict(body["params"]),
            "to": {"name": target["name"], "sha256": target["sha256"], "v": target["v"]},
        }

    def due(self, e):
        """The pending ``evolve`` at its minute: dropped when pinned or from another law, else applied."""
        st = self.state
        pending = st["pending"]
        st["pending"] = None
        if st["pinned"]:
            self.note("evolve_pinned", e)
            return
        law = st["law"]
        if pending["from"]["name"] != law["name"] or pending["from"]["sha256"] != law["sha256"]:
            self.note("evolve_from", e)
            return
        target = pending["to"]
        change = target["name"] != law["name"] or target["sha256"] != law["sha256"]
        if change or target["v"] != law["v"]:
            # A pending change comes from a state the caller sent: it is approached as its registration was,
            # its version too, so a state this engine returns never names a law it would refuse.
            self.life = _approach(target, pending["from"])
        st["law"] = dict(target)
        st["params"] = dict(pending["params"])
        if target["v"] not in st["seen"]:
            st["seen"] = sorted(st["seen"] + [target["v"]])
        if change:
            self.law_changed()

    def law_changed(self):
        """What a change of law moves besides the law and its params; nothing in the timeline."""
        return None


# ---------------------------------------------------------------------------
# advance
# ---------------------------------------------------------------------------

class _Cx:
    """What an organ reads: the bus, the environment, the params and the law, and where to count."""

    __slots__ = ("bus", "sun", "params", "dormant", "constants", "world", "hemisphere", "weather", "seed",
                 "being_hi", "being_lo", "work", "minute", "day", "draws", "clips")


def _calls(organs):
    return {name: {"daily": {"awake": 0, "dormant": 0}, "fast": {"awake": 0, "dormant": 0}, "jump": 0}
            for name in organs}


class _Reducer(_Fold):
    """The stepped path, the fast path, and the organs' layers."""

    def __init__(self, state, life, genesis, order, fast_path, work):
        body = genesis["body"]
        super().__init__(state, life, body["birth"]["wall"] // 60, work)
        self.genesis = genesis
        self.order = order
        self.fast_path = fast_path
        self.calls = _calls(life.organs)
        self.acts = {}
        self.visits = 0
        self.env = 0
        self.fast_path_days = 0
        self.fast_path_skipped = 0
        self.sine = None
        cx = _Cx()
        cx.work = work
        cx.hemisphere = body["hemisphere"]
        cx.weather = body["weather"]
        cx.seed = bytes.fromhex(body["seed"])
        being = bytes.fromhex(genesis["being"])
        cx.being_hi = int.from_bytes(being[:8], "big")
        cx.being_lo = int.from_bytes(being[8:], "big")
        cx.draws = 0
        cx.clips = 0
        cx.sun = 0
        cx.minute = state["at"]
        cx.day = 0
        self.cx = cx
        self.band = body["band"]
        # Each layer's steps in the requested organ order: an organ without the layer is never called in it.
        self.fast_steps = tuple((name, MODULES[name].fast) for name in order if hasattr(MODULES[name], "fast"))
        self.daily_steps = tuple((name, MODULES[name].daily) for name in order if hasattr(MODULES[name], "daily"))
        self.fact_steps = tuple((name, MODULES[name].on_fact) for name in order
                                if hasattr(MODULES[name], "on_fact"))
        self._law_constants()

    def _law_constants(self):
        life = self.life
        self.cx.constants = life.constants
        self.cx.world = life.world
        self.skip = life.quiescent

    def law_changed(self):
        """A change of law: its constants and world, and the genome's constants derived again under it.

        The genome stays the one founded under the genesis law; the organs'
        states are otherwise unchanged (the identity migration).
        """
        self._law_constants()
        k = _derive(self.genesis, self.life)
        organs = self.state["organs"]
        organs["chem"]["k"] = k["chem"]
        organs["clock"]["k"] = k["clock"]

    # -- the layers ---------------------------------------------------------

    def _sine(self):
        if self.sine is None:
            self.sine = lawdata.sine()
        return self.sine

    def publish(self):
        st = self.state
        cx = self.cx
        organs = st["organs"]
        dormant = organs["stage"]["dormant"]
        bus = {}
        for name in self.life.organs:
            masked = dormant and name in self.skip
            for channel, value in MODULES[name].publish(organs[name], cx).items():
                bus[channel] = 0 if masked else value
        st["bus"] = bus

    def jumps(self, since, wake):
        organs = self.state["organs"]
        for name in self.order:
            if name in self.skip:
                MODULES[name].jump(organs[name], self.cx, since, wake)
                self.calls[name]["jump"] += 1

    def fast(self, e, day):
        st = self.state
        cx = self.cx
        organs = st["organs"]
        dormant = organs["stage"]["dormant"]
        cx.dormant = dormant
        cx.bus = st["bus"]
        cx.params = st["params"]
        if not dormant:
            self.env += 1
            minute = (self.b + e + st["tz"]) % DAY
            light = civil.daylength(day, cx.world, cx.hemisphere, self.band, self._sine(), self.work)
            cx.sun = civil.sun(minute, light, cx.world, st["params"]["sun_max"], self.work)
        state_name = "dormant" if dormant else "awake"
        for name, step in self.fast_steps:
            if dormant and self.fast_path and name in self.skip:
                continue
            step(organs[name], cx)
            self.calls[name]["fast"][state_name] += 1

    def daily(self, e, day):
        """The daily layer at minute ``e`` for local day ``day``; whether the being woke."""
        st = self.state
        cx = self.cx
        organs = st["organs"]
        stage_org = organs["stage"]
        dormant = stage_org["dormant"]
        since = stage_org["since"]
        cx.dormant = dormant
        cx.bus = st["bus"]
        cx.params = st["params"]
        cx.minute = e
        state_name = "dormant" if dormant else "awake"
        for name, step in self.daily_steps:
            if dormant and self.fast_path and name in self.skip:
                continue
            step(organs[name], cx, day)
            self.calls[name]["daily"][state_name] += 1
        st["day"] = day
        woke = dormant and not stage_org["dormant"]
        if woke:
            self.jumps(since, e)
        return woke

    # -- one event minute ---------------------------------------------------

    def minute(self, e, here, first):
        st = self.state
        cx = self.cx
        self.work.units += VISIT
        self.visits += 1
        pending = st["pending"]
        if pending is not None and pending["effective_from"] <= e:
            self.due(e)
        for fact in here:
            if fact["kind"] == "tz":
                self.tz_fact(fact)
        b = self.b
        day = (b + e + st["tz"]) // DAY
        if first:
            st["day"] = day
            st["organs"]["stage"]["season"] = civil.season(day, cx.world, cx.hemisphere)
        for fact in here:
            if fact["kind"] in LAW_KINDS:
                self.law_fact(fact, e)
        stage_org = st["organs"]["stage"]
        asleep = stage_org["dormant"]
        since = stage_org["since"]
        cx.bus = st["bus"]
        cx.params = st["params"]
        cx.minute = e
        cx.day = day
        organs = st["organs"]
        for fact in here:
            kind = fact["kind"]
            if kind in REDUCER_KINDS or not self.consume(fact):
                continue
            if kind == "act":
                act = fact["body"]["act"]
                self.acts[act] = self.acts.get(act, 0) + 1
            for name, handler in self.fact_steps:
                handler(organs[name], fact, cx)
        if asleep and not stage_org["dormant"]:
            self.jumps(since, e)
        if here:
            last = here[-1]
            st["through"] = [last["t"], last["origin"], last["oseq"]]
            st["n"] += len(here)
        if (b + e) % FAST == 0:
            self.fast(e, day)
        if day > st["day"]:
            self.daily(e, day)
        self.publish()

    # -- the driver ---------------------------------------------------------

    def run(self, facts, to, budget, fresh):
        """Live up to ``to`` or until the budget runs out; whether ``to`` was reached."""
        st = self.state
        work = self.work
        b = self.b
        count = len(facts)
        i = 0
        if fresh:
            j = i
            while j < count and facts[j]["t"] == 0:
                j += 1
            self.minute(0, facts[i:j], True)
            i = j
            st["at"] = 0
            if work.units >= budget and st["at"] < to:
                return False
        while st["at"] < to:
            at = st["at"]
            next_t = facts[i]["t"] if i < count else None
            pending = st["pending"]
            if self.fast_path and st["organs"]["stage"]["dormant"]:
                span = to
                if next_t is not None and next_t - 1 < span:
                    span = next_t - 1
                # Asleep, the due evolve's minute is visited (see the fast path above).
                if pending is not None and pending["effective_from"] - 1 < span:
                    span = pending["effective_from"] - 1
                if span > at:
                    z = st["tz"]
                    m = civil.next_midnight(b, at, z)
                    woke = False
                    while m <= span:
                        day = (b + m + z) // DAY
                        if _fast_path_fires(day, st["day"]):
                            work.units += FAST_PATH_DAY
                            self.fast_path_days += 1
                            woke = self.daily(m, day)
                            self.publish()
                            if woke:
                                st["at"] = m
                                break
                            if work.units >= budget and m < span:
                                st["at"] = m
                                return False
                        else:
                            self.fast_path_skipped += 1
                        m += DAY
                    if not woke:
                        st["at"] = span
                    continue
            e = at + 1 + (-(b + at + 1)) % FAST
            if next_t is not None and next_t < e:
                e = next_t
            if pending is not None and at < pending["effective_from"] < e:
                e = pending["effective_from"]
            if e > to:
                st["at"] = to
                break
            j = i
            while j < count and facts[j]["t"] == e:
                j += 1
            self.minute(e, facts[i:j], False)
            i = j
            st["at"] = e
            if work.units >= budget and e < to:
                return False
        return True

    def trace(self):
        return {
            "acts": dict(self.acts),
            "calls": self.calls,
            "clock_clip": self.cx.clips,
            "draws": self.cx.draws,
            "env": self.env,
            "facts": self.facts,
            "noted": self.noted,
            "fast_path_days": self.fast_path_days,
            "fast_path_skipped": self.fast_path_skipped,
            "visits": self.visits,
        }


def _found(genesis, life):
    """The being's genome, founded under the genesis law from its seed; ``(bytes, work)``."""
    law, lawview, _digest = lawdata.genome_law(life.name)
    alleles = lawdata.genome_pool(law)
    data, _chosen, work = genome.found(bytes.fromhex(genesis["body"]["seed"]), lawview, alleles)
    return data, work


def _derive(genesis, life):
    """The genome-derived constants under law ``life``, the genome founded under the genesis law."""
    genesis_life = lawdata.life(genesis["body"]["laws"]["name"])
    data, _found_work = _found(genesis, genesis_life)
    law, lawview, digest = lawdata.genome_law(life.name)
    tables, _work = organ_compile.compile_genome(data, law, lawview, digest)
    return lawdata.consts(tables, life.law)


def _initial(genesis, life):
    """The state before minute 0 is processed, and the work the genome operations reported."""
    data, found_work = _found(genesis, life)
    law, lawview, digest = lawdata.genome_law(life.name)
    tables, compile_work = organ_compile.compile_genome(data, law, lawview, digest)
    k = lawdata.consts(tables, life.law)
    body = genesis["body"]
    birth = body["birth"]
    b = birth["wall"] // 60
    organs = {}
    for name in sorted(life.organs):
        organs[name] = MODULES[name].init(life.constants[name], k.get(name))
    state = {
        "at": 0,
        "being": genesis["being"],
        "budget": {"counts": {}, "day": 0},
        "bus": {},
        "day": (b + birth["tz"]) // DAY,
        "genome": genome.sha256(data),
        "law": {"name": life.name, "sha256": life.digest, "v": life.version},
        "n": 0,
        "organs": organs,
        "params": dict(body["laws"]["params"]),
        "pending": None,
        "pinned": False,
        "schema": SCHEMA,
        "seen": [life.version],
        "through": None,
        "tz": birth["tz"],
    }
    return state, found_work + compile_work


def _env(state, reducer):
    """The environment at the state's minute, at no cost."""
    cx = reducer.cx
    work = fx.Work()
    day, minute = civil.local(reducer.b, state["at"], state["tz"])
    light = civil.daylength(day, cx.world, cx.hemisphere, reducer.band, reducer._sine(), work)
    return {
        "civil": list(civil.civil_from_days(day)),
        "day": day,
        "daylength": light,
        "minute": minute,
        "season": civil.season(day, cx.world, cx.hemisphere),
        "sun": civil.sun(minute, light, cx.world, state["params"]["sun_max"], work),
    }


def _capped(notes, limit, at):
    if len(notes) <= limit:
        return notes
    return notes[:limit - 1] + [{"code": "truncated", "t": at}]


def op_advance(request, size, limits):
    """The ``advance`` operation: a life folded up to ``to`` or its budget."""
    _fields(request, ("budget", "facts", "genesis", "op", "state", "to", "v"), ("probe",))
    genesis, life = _genesis(request)
    to = _to(request)
    budget = request["budget"]
    if not _int_in(budget, 1, MAX_INT):
        raise Refused("bad_request", "budget")
    order, fast_path, tracing = _probe(request, life)
    state, in_force = _state(request, genesis, life, to, STATE_KEYS)
    facts = _facts(request, genesis, life, state, to, limits["items"], False)
    overhead = (size + 63) // 64
    fresh = state is None
    if fresh:
        state, genome_work = _initial(genesis, life)
        overhead += genome_work
    work = fx.Work()
    reducer = _Reducer(state, in_force, genesis, order, fast_path, work)
    if fresh:
        reducer.publish()
    done = reducer.run(facts, to, budget, fresh)
    answer = {
        "alarm": work.alarm,
        "at": state["at"],
        "done": done,
        "env": _env(state, reducer),
        "hash": hashlib.sha256(wire.emit(state)).hexdigest(),
        "notes": _capped(reducer.notes, limits["notes"], state["at"]),
        "overhead": overhead,
        "provisional": genesis["body"]["laws"]["provisional"],
        "state": state,
        "work": work.units,
    }
    if tracing:
        answer["trace"] = reducer.trace()
    return answer


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------

class _Timeline(_Fold):
    """The law kinds folded alone, and the daily firing minutes computed per stretch of constant offset."""

    def __init__(self, state, life, b, work, window):
        super().__init__(state, life, b, work)
        self.window = window
        self.midnights = []
        self.cursor = state["at"]

    def fire(self, m, day):
        """A daily firing at minute ``m``: the mark moves, and the minute is listed when it is in the window."""
        self.state["day"] = day
        if self.window is not None and m > self.window[0]:
            if len(self.midnights) >= self.window[1]:
                raise Refused("limit", "items")
            self.midnights.append([m, day, list(civil.civil_from_days(day))])
            self.work.units += 1

    def rise(self, end):
        """Every local midnight in ``(cursor, end]`` under the offset in force whose day is above the mark."""
        st = self.state
        if end <= self.cursor:
            return
        z = st["tz"]
        m = civil.next_midnight(self.b, self.cursor, z)
        day = (self.b + m + z) // DAY
        if day <= st["day"]:
            m += (st["day"] - day + 1) * DAY
            day = st["day"] + 1
        if m <= end:
            if self.window is None:
                last = (self.b + end + z) // DAY
                st["day"] = last
            else:
                low = self.window[0]
                if m <= low:
                    skip = (low - m) // DAY + 1
                    m += skip * DAY
                    day += skip
                if m <= end and len(self.midnights) + (end - m) // DAY + 1 > self.window[1]:
                    raise Refused("limit", "items")
                while m <= end:
                    self.fire(m, day)
                    m += DAY
                    day += 1
                st["day"] = max(st["day"], (self.b + end + z) // DAY)
        self.cursor = end

    def run(self, facts, to, fresh):
        st = self.state
        count = len(facts)
        i = 0
        if fresh:
            j = i
            while j < count and facts[j]["t"] == 0:
                j += 1
            here = facts[i:j]
            for fact in here:
                if fact["kind"] == "tz":
                    self.tz_fact(fact)
            st["day"] = (self.b + st["tz"]) // DAY
            for fact in here:
                if fact["kind"] in LAW_KINDS:
                    self.law_fact(fact, 0)
            self._through(here)
            i = j
        while i < count:
            e = facts[i]["t"]
            j = i
            while j < count and facts[j]["t"] == e:
                j += 1
            here = facts[i:j]
            pending = st["pending"]
            if pending is not None and pending["effective_from"] <= e:
                self.due(pending["effective_from"])
            self.rise(e - 1)
            moved = False
            for fact in here:
                if fact["kind"] == "tz":
                    self.tz_fact(fact)
                    moved = True
            if moved:
                day = (self.b + e + st["tz"]) // DAY
                if day > st["day"]:
                    self.fire(e, day)
                self.cursor = e
            for fact in here:
                if fact["kind"] in LAW_KINDS:
                    self.law_fact(fact, e)
            self._through(here)
            i = j
        pending = st["pending"]
        if pending is not None and pending["effective_from"] <= to:
            self.due(pending["effective_from"])
        self.rise(to)
        st["at"] = to

    def _through(self, here):
        if here:
            last = here[-1]
            self.state["through"] = [last["t"], last["origin"], last["oseq"]]


def _initial_timeline(genesis, life):
    body = genesis["body"]
    return {
        "at": 0,
        "budget": {"counts": {}, "day": 0},
        "day": (body["birth"]["wall"] // 60 + body["birth"]["tz"]) // DAY,
        "law": {"name": life.name, "sha256": life.digest, "v": life.version},
        "params": dict(body["laws"]["params"]),
        "pending": None,
        "pinned": False,
        "schema": SCHEMA,
        "seen": [life.version],
        "through": None,
        "tz": body["birth"]["tz"],
    }


def op_timeline(request, limits):
    """The ``timeline`` operation: the law kinds folded up to ``to``, with the daily firing minutes."""
    _fields(request, ("facts", "from", "genesis", "op", "to", "v"), ("midnights_after",))
    genesis, life = _genesis(request)
    to = _to(request)
    after = None
    if "midnights_after" in request:
        after = request["midnights_after"]
        if not _int_in(after, 0, MAX_INT):
            raise Refused("bad_request", "midnights_after")
    state, in_force = _state(request, genesis, life, to, TSTATE_KEYS)
    facts = _facts(request, genesis, life, state, to, limits["items"], True)
    fresh = state is None
    if fresh:
        state = _initial_timeline(genesis, life)
    work = fx.Work()
    window = None if after is None else (max(after, state["at"]), limits["items"])
    fold = _Timeline(state, in_force, genesis["body"]["birth"]["wall"] // 60, work, window)
    fold.run(facts, to, fresh)
    answer = {
        "midnight": civil.next_midnight(fold.b, to, state["tz"]),
        "notes": _capped(fold.notes, limits["notes"], to),
        "state": state,
        "work": work.units,
    }
    if window is not None:
        answer["midnights"] = fold.midnights
    return answer
