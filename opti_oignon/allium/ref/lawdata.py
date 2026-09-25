"""The law data a life runs on: the carried laws, their life sections, and the constants a genome gives.

The reference. Its Rust twin reads the same files.

A carried law is one this engine embeds, named in ``lawfiles.LAWS``; its
identity is its name and the digest of its canonical bytes. Besides its
genome section, a law carries the sections its life runs on:

* ``code`` -- the revision of the organs' code; the engine implements a
  closed set (``CODES``), and each revision has its organs and the bus
  channel each writes;
* ``organs``, ``quiescent_in_dormancy`` and ``bus`` -- the default organ
  order, the organs never called while the being is dormant, and which
  organ writes each channel;
* ``world`` -- the year, the seasons, the provisional daylength and the rain;
* ``constants`` -- each organ's stocks, scales and thresholds, and the loci
  it reads;
* ``params`` -- the ranges of the few numbers a genesis freezes;
* ``work`` -- the unit table, the per-day caps of the budget-exempt kinds,
  and the day ceilings those imply, with their proofs;
* ``succeeds``, when present -- the stable law this one migrates from.

``life(name)`` checks the law's journal pin (the kinds table it names, by
digest, and a daily budget for every budgeted kind), then its life sections,
and remembers the result by the digests of the law's and the table's bytes.
A defect refuses ``unknown_law`` with the detail ``journal``, ``life law``
or, for an organ code this engine does not implement, ``code``.

``consts(tables, law)`` derives, from a genome's compiled tables, the
constants the organs keep in the state: the clock's production, decay,
repression and light gain, and the reserve's reaction rates. It costs no
unit: it runs at a being's first minute and on a change of law only.

The genome's law and founder pool are read here too, for the genome
operations and for a life's first minute alike.
"""

import hashlib

from .. import fx, lawfiles, wire
from ..wire import Refused
from . import journal
from .organs import genome

checkpoint_before_apply = True

ONE = fx.ONE
CMAX = fx.CMAX
MAX_INT = wire.MAX_INT
# The organ code revisions this engine implements, their organs, and the channel each organ writes.
CODES = ("seed_1",)
CODE_ORGANS = {"seed_1": ("chem", "clock", "soil", "stage")}
CODE_BUS = {"seed_1": {"circadian": "clock", "dormant": "stage", "metab": "chem", "moisture": "soil"}}
# The trunk kinds outside the membrane's daily budgets; all but the genesis are capped per day of life instead.
EXEMPT = ("clock", "genesis", "owner", "resumed", "tz")
BUDGET_MAX = 4096
LIFE_KEYS = ("bus", "code", "constants", "organs", "params", "quiescent_in_dormancy", "work", "world")
BANDS = ("long", "medium", "short")
SEASONS = 4
CIVIL_YEAR = 365
CHEM_KEYS = ("core", "enzymes", "fructan0", "fructan_max", "hyd_scale", "k_w", "metab_shift", "ps_scale",
             "resp_scale", "sugar0", "sugar_max", "syn_scale", "theta_h", "theta_s")
# Each reaction of the reserve, and the scale its summed turnover is taken at.
ROLES = (("hydrolysis", "hyd_scale", "hy"), ("photosynthesis", "ps_scale", "ps"),
         ("respiration", "resp_scale", "r"), ("synthesis", "syn_scale", "sy"))
FRUCTAN_LIMIT = 1 << 30
STAGE_KEYS = ("d_enter", "rest_dry", "rest_max", "rest_winter", "theta_dry", "theta_wet")
REST_LIMIT = 100000
# The unit table's exact shape; every leaf is a non-negative integer.
UNITS_SHAPE = {
    "act": {"water": {"soil": 0}},
    "draw": 0,
    "env": 0,
    "fact": 0,
    "fast_path_day": 0,
    "organs": {
        "chem": {"fast": {"awake": 0, "dormant": 0}},
        "clock": {"fast": {"awake": 0, "dormant": 0}},
        "soil": {"daily": {"awake": 0, "dormant": 0}},
        "stage": {"daily": {"awake": 0, "dormant": 0}},
    },
    "visit": 0,
}
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


def _is_text(value, most):
    if not isinstance(value, str) or not 1 <= len(value) <= most:
        return False
    for char in value:
        if not 0x20 <= ord(char) <= 0x7E:
            return False
    return value[0] != " " and value[-1] != " "


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


def _distinct_names(values):
    """A list of strings, none twice."""
    if not isinstance(values, list):
        return False
    met = {}
    for value in values:
        if not isinstance(value, str) or value in met:
            return False
        met[value] = True
    return True


# ---------------------------------------------------------------------------
# Carried files
# ---------------------------------------------------------------------------

# Parsed files and their canonical digests, by the SHA-256 of their bytes.
_FILES = {}


def _parsed(data, key=None):
    key = key or hashlib.sha256(data).hexdigest()
    if key not in _FILES:
        value = wire.parse(data, lenient=True)
        _FILES[key] = (value, lawfiles.digest(value))
    return _FILES[key]


def _law_entry(name):
    """The parsed carried law ``name``, its digest, and the SHA-256 of its bytes."""
    if not isinstance(name, str) or name not in lawfiles.LAWS:
        raise Refused("unknown_law", "law")
    data = lawfiles.law_bytes(name)
    key = hashlib.sha256(data).hexdigest()
    value, digest = _parsed(data, key)
    return value, digest, key


def law_file(name):
    """The parsed carried law ``name`` and its digest; ``unknown_law "law"`` for a law this engine does not carry."""
    value, digest, _key = _law_entry(name)
    return value, digest


def sine():
    """The frozen Q15 sine table, parsed once per content of its file."""
    value, _digest = _parsed(lawfiles.table_bytes("sine_q15_v1"))
    return value["entries"]


# The genome laws (value, codec view or None when unsound, digest) and pools, by the SHA-256 of their bytes.
_GENOME_LAWS = {}
_POOLS = {}


def genome_law(name):
    """A carried law, its genome codec view and its digest; ``unknown_law "genome law"`` when its genome is unsound."""
    if not isinstance(name, str) or name not in lawfiles.LAWS:
        raise Refused("unknown_law", "law")
    data = lawfiles.law_bytes(name)
    key = hashlib.sha256(data).hexdigest()
    if key not in _GENOME_LAWS:
        law = wire.parse(data, lenient=True)
        sound = not genome.validate_law(law)
        _GENOME_LAWS[key] = (law, genome.view(law) if sound else None, lawfiles.digest(law))
    law, lawview, digest = _GENOME_LAWS[key]
    if lawview is None:
        raise Refused("unknown_law", "genome law")
    return law, lawview, digest


def genome_pool(law):
    """The founder alleles a genome law pins, checked; refused by name when the pin or the pool is unsound."""
    pin = law.get("founders")
    if not isinstance(pin, dict) or pin.get("name") not in lawfiles.FOUNDERS:
        raise Refused("unknown_law", "founders digest")
    data = lawfiles.founders_bytes(pin["name"])
    key = hashlib.sha256(data).hexdigest()
    if key not in _POOLS:
        pool = wire.parse(data, lenient=True)
        _POOLS[key] = (pool, lawfiles.digest(pool))
    pool, digest = _POOLS[key]
    if pin.get("sha256") != digest:
        raise Refused("unknown_law", "founders digest")
    checked = (key, lawfiles.digest(law))
    if checked not in _POOLS:
        _POOLS[checked] = genome.pool_alleles(pool) if not genome.validate_pool(law, pool) else None
    alleles = _POOLS[checked]
    if alleles is None:
        raise Refused("unknown_law", "founders")
    return alleles


# ---------------------------------------------------------------------------
# The journal pin
# ---------------------------------------------------------------------------

def _journal_refused():
    return Refused("unknown_law", "journal")


def _journal(law):
    """The kinds table a law pins, its bytes, and the daily budgets of the budgeted kinds; refused ``journal``."""
    pin = law.get("journal") if isinstance(law, dict) else None
    if not _exactly(pin, ("budgets", "table")):
        raise _journal_refused()
    named = pin["table"]
    if not _exactly(named, ("name", "sha256")) or not isinstance(named["name"], str) \
            or named["name"] not in lawfiles.JOURNAL_TABLES:
        raise _journal_refused()
    data = lawfiles.table_bytes(named["name"])
    try:
        table, digest = _parsed(data)
    except Refused:
        raise _journal_refused() from None
    if named["sha256"] != digest:
        raise _journal_refused()
    try:
        journal.validate_table(table)
    except Refused:
        raise _journal_refused() from None
    budgeted = sorted(kind for kind, entry in table["kinds"].items()
                      if entry["scope"] == "trunk" and kind not in EXEMPT)
    budgets = pin["budgets"]
    if not isinstance(budgets, dict):
        raise _journal_refused()
    for kind in budgeted:
        if kind not in budgets or not _int_in(budgets[kind], 1, BUDGET_MAX):
            raise _journal_refused()
    for kind in budgets:
        if kind not in budgeted:
            raise _journal_refused()
    return table, data, {kind: budgets[kind] for kind in budgeted}


def params_schema(table):
    """The params a genesis freezes and an ``evolve`` carries, as the kinds table bounds them."""
    return table["kinds"]["genesis"]["body"]["laws"]["fields"]["params"]["fields"]


# ---------------------------------------------------------------------------
# The life sections
# ---------------------------------------------------------------------------

def _loci_kinds(law):
    """The kind of each locus of the law's genome, by name; nothing for a genome section it cannot read."""
    section = law.get("genome")
    loci = section.get("loci") if isinstance(section, dict) else None
    kinds = {}
    if not isinstance(loci, list):
        return kinds
    for entry in loci:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str) and isinstance(entry.get("kind"), str):
            kinds[entry["name"]] = entry["kind"]
    return kinds


def _world_ok(world):
    if not _exactly(world, ("daylength", "rain", "season_shift", "south_shift", "year")):
        return False
    year = world["year"]
    if _exactly(year, ("kind",)) and year["kind"] == "civil":
        length = CIVIL_YEAR
    elif _exactly(year, ("days", "kind")) and year["kind"] == "fixed" and _int_in(year["days"], 4, 1000):
        length = year["days"]
    else:
        return False
    light = world["daylength"]
    if not _exactly(light, ("amp", "equinox", "mean", "ramp")):
        return False
    for value in (light["equinox"], world["season_shift"], world["south_shift"]):
        if not _int_in(value, 0, length - 1):
            return False
    if not _int_in(light["ramp"], 1, 120) or not _is_int(light["mean"]) or not _exactly(light["amp"], BANDS):
        return False
    for band in BANDS:
        amp = light["amp"][band]
        if not _int_in(amp, 0, 720) or not amp <= light["mean"] <= 1440 - amp:
            return False
    rain = world["rain"]
    if not _exactly(rain, ("max", "p_wet")):
        return False
    for key in ("max", "p_wet"):
        values = rain[key]
        if not isinstance(values, list) or len(values) != SEASONS:
            return False
        for value in values:
            if not _int_in(value, 0, ONE):
                return False
    return True


def _chem_ok(chem, loci):
    if not _exactly(chem, CHEM_KEYS):
        return False
    for key in CHEM_KEYS:
        if key != "enzymes" and not _is_int(chem[key]):
            return False
    if not 0 <= chem["core"] <= chem["fructan0"] <= chem["fructan_max"] <= FRUCTAN_LIMIT:
        return False
    if not 0 <= chem["sugar0"] <= chem["sugar_max"] <= CMAX:
        return False
    if not 0 <= chem["theta_h"] <= chem["sugar_max"] or not 0 <= chem["theta_s"] <= chem["sugar_max"]:
        return False
    if not 1 <= chem["k_w"] <= ONE or not 0 <= chem["metab_shift"] <= 15:
        return False
    enzymes = chem["enzymes"]
    if not _exactly(enzymes, tuple(role for role, _, _ in ROLES)):
        return False
    for role, scale_key, _ in ROLES:
        names = enzymes[role]
        scale = chem[scale_key]
        if not _distinct_names(names) or not 1 <= len(names) <= 4 or scale < 0:
            return False
        for name in names:
            if loci.get(name) != "enz":
                return False
        if (len(names) * 65535 * scale) >> 16 > ONE:
            return False
    return True


def _clock_ok(clock, loci):
    if not _exactly(clock, ("genes", "init", "light")):
        return False
    genes, init = clock["genes"], clock["init"]
    if not _distinct_names(genes) or len(genes) != 3:
        return False
    for name in genes:
        if loci.get(name) != "tf":
            return False
    if not isinstance(init, list) or len(init) != 3:
        return False
    for value in init:
        if not _int_in(value, 0, CMAX):
            return False
    return isinstance(clock["light"], str) and loci.get(clock["light"]) == "rec"


def _soil_ok(soil):
    if not _exactly(soil, ("dose", "m0", "m_max")) or not _int_in(soil["m_max"], 1, ONE):
        return False
    return _int_in(soil["m0"], 0, soil["m_max"]) and _int_in(soil["dose"], 0, soil["m_max"])


def _stage_ok(stage, m_max):
    if not _exactly(stage, STAGE_KEYS) or not _int_in(stage["rest_max"], 1, REST_LIMIT):
        return False
    for key in ("d_enter", "rest_dry", "rest_winter"):
        if not _int_in(stage[key], 1, stage["rest_max"]):
            return False
    if not _is_int(stage["theta_dry"]) or not _is_int(stage["theta_wet"]):
        return False
    return 0 <= stage["theta_dry"] <= stage["theta_wet"] <= m_max


def _params_ok(params, table):
    bounds = params_schema(table)
    if not _exactly(params, tuple(bounds)):
        return False
    for name, spec in params.items():
        if not _exactly(spec, ("default", "hi", "lo")):
            return False
        if not (_is_int(spec["lo"]) and _is_int(spec["default"]) and _is_int(spec["hi"])):
            return False
        if not bounds[name]["lo"] <= spec["lo"] <= spec["default"] <= spec["hi"] <= bounds[name]["hi"]:
            return False
    return True


def _shaped(value, shape):
    """Whether ``value`` has exactly the keys of ``shape`` at every level, with non-negative integer leaves."""
    if isinstance(shape, dict):
        if not _exactly(value, tuple(shape)):
            return False
        for key in shape:
            if not _shaped(value[key], shape[key]):
                return False
        return True
    return _int_in(value, 0, MAX_INT)


def _work_ok(work, table):
    if not _exactly(work, ("caps", "ceilings", "proof", "units")):
        return False
    capped = tuple(kind for kind in EXEMPT if kind != "genesis"
                   and table["kinds"].get(kind, {}).get("scope") == "trunk")
    if not _exactly(work["caps"], capped):
        return False
    for kind in capped:
        if not _int_in(work["caps"][kind], 1, MAX_INT):
            return False
    if not _shaped(work["units"], UNITS_SHAPE):
        return False
    ceilings, proof = work["ceilings"], work["proof"]
    if not _exactly(ceilings, ("awake_day", "dormant_day")) or not _exactly(proof, ("awake_day", "dormant_day")):
        return False
    for key in ("awake_day", "dormant_day"):
        if not _is_int(ceilings[key]) or not isinstance(proof[key], str):
            return False
    return True


def _life_defect(law, table):
    """What is wrong with a law's life sections: ``None``, ``"code"`` or ``"life law"``."""
    for key in LIFE_KEYS:
        if key not in law:
            return "life law"
    code = law["code"]
    if not isinstance(code, str) or code not in CODES:
        return "code"
    if not _int_in(law.get("version"), 0, 65535) or not isinstance(law.get("provisional"), bool):
        return "life law"
    organs = law["organs"]
    if not _distinct_names(organs) or sorted(organs) != sorted(CODE_ORGANS[code]):
        return "life law"
    quiescent = law["quiescent_in_dormancy"]
    if not _distinct_names(quiescent):
        return "life law"
    for name in quiescent:
        if name not in organs:
            return "life law"
    if law["bus"] != CODE_BUS[code]:
        return "life law"
    if not _world_ok(law["world"]):
        return "life law"
    constants = law["constants"]
    loci = _loci_kinds(law)
    if not _exactly(constants, CODE_ORGANS[code]):
        return "life law"
    if not (_chem_ok(constants["chem"], loci) and _clock_ok(constants["clock"], loci)
            and _soil_ok(constants["soil"])):
        return "life law"
    if not _stage_ok(constants["stage"], constants["soil"]["m_max"]):
        return "life law"
    if not _params_ok(law["params"], table) or not _work_ok(law["work"], table):
        return "life law"
    if "succeeds" in law:
        succeeds = law["succeeds"]
        if not _exactly(succeeds, ("migrate", "name", "sha256")) or succeeds["migrate"] != "identity" \
                or not _is_text(succeeds["name"], 32) or not _is_hex(succeeds["sha256"], 64):
            return "life law"
    return None


class Life:
    """A carried law whose journal pin and life sections are sound, in the form the reducer reads."""

    __slots__ = ("name", "digest", "version", "provisional", "law", "table", "budgets", "caps", "limits",
                 "code", "organs", "quiescent", "world", "constants", "params")

    def __init__(self, name, digest, law, table, budgets):
        self.name = name
        self.digest = digest
        self.version = law["version"]
        self.provisional = law["provisional"]
        self.law = law
        self.table = table
        self.budgets = budgets
        self.caps = dict(law["work"]["caps"])
        # The per-day limit of every trunk kind but the genesis: its budget, or its cap.
        self.limits = dict(budgets)
        self.limits.update(self.caps)
        self.code = law["code"]
        self.organs = tuple(law["organs"])
        self.quiescent = tuple(law["quiescent_in_dormancy"])
        self.world = law["world"]
        self.constants = law["constants"]
        self.params = law["params"]


# Sound lives, or the refusal of an unsound one, by the digests of the law's and its table's bytes.
_LIVES = {}


def life(name):
    """Carried law ``name`` as a ``Life``; ``unknown_law`` ``law``, ``journal``, ``life law`` or ``code`` otherwise."""
    law, digest, key = _law_entry(name)
    pin = law.get("journal") if isinstance(law, dict) else None
    named = pin.get("table") if isinstance(pin, dict) else None
    table_name = named.get("name") if isinstance(named, dict) else None
    if isinstance(table_name, str) and table_name in lawfiles.JOURNAL_TABLES:
        key += hashlib.sha256(lawfiles.table_bytes(table_name)).hexdigest()
    if key not in _LIVES:
        try:
            table, _data, budgets = _journal(law)
        except Refused as refusal:
            _LIVES[key] = refusal.detail
        else:
            defect = _life_defect(law, table)
            _LIVES[key] = defect if defect is not None else Life(name, digest, law, table, budgets)
    found = _LIVES[key]
    if isinstance(found, str):
        raise Refused("unknown_law", found)
    return found


def params_defect(params, law_life):
    """The first param, by name, outside the law's range; ``None`` when every one is inside."""
    for name in sorted(law_life.params):
        spec = law_life.params[name]
        if not spec["lo"] <= params[name] <= spec["hi"]:
            return name
    return None


def successor_ok(target, source_pair):
    """Whether carried law ``target`` may follow the law named by ``source_pair`` (``{"name", "sha256"}``).

    Both are stable, ``target`` names the source as the law it migrates from
    by the identity migration and carries a higher version, their organs,
    bus, quiescent organs, world, journal pin, genome section and founders
    pin are the same, every stock bound of ``target`` contains the source's,
    so do the ceilings of the soil's moisture and of the stage's day counts
    (a state keeps those levels, and a lower ceiling could leave one above
    it), and every param range of ``target`` contains the source's.
    """
    name = source_pair["name"]
    if not isinstance(name, str) or name not in lawfiles.LAWS:
        return False
    try:
        source = life(name)
    except Refused:
        return False
    if source.digest != source_pair["sha256"] or source.provisional or target.provisional:
        return False
    if target.law.get("succeeds") != {"migrate": "identity", "name": source.name, "sha256": source.digest}:
        return False
    if target.version <= source.version:
        return False
    for key in ("organs", "bus", "quiescent_in_dormancy", "world", "journal", "genome", "founders"):
        if target.law.get(key) != source.law.get(key):
            return False
    new, old = target.constants["chem"], source.constants["chem"]
    if new["sugar_max"] < old["sugar_max"] or new["fructan_max"] < old["fructan_max"] or new["core"] > old["core"]:
        return False
    # The levels a state keeps are bounded by the law in force too: a lower ceiling could leave one above it.
    if target.constants["soil"]["m_max"] < source.constants["soil"]["m_max"]:
        return False
    if target.constants["stage"]["rest_max"] < source.constants["stage"]["rest_max"]:
        return False
    for param, spec in source.params.items():
        wider = target.params[param]
        if wider["lo"] > spec["lo"] or wider["hi"] < spec["hi"]:
            return False
    return True


# ---------------------------------------------------------------------------
# Constants a genome gives
# ---------------------------------------------------------------------------

def _column(tables, table, column):
    return wire.unpack_bulk(tables[table][column])[1]


def consts(tables, law):
    """The organs' genome-derived constants ``{"chem": k, "clock": k}`` from compiled tables under ``law``.

    The clock: for each gene named in ``constants.clock.genes``, the TF row
    whose gene's locus has that name gives ``alpha`` (its production) and
    ``beta`` (its decay), and the gene's first promoter edge gives ``k`` and
    ``n``; ``light`` is the gain of the REC row named
    ``constants.clock.light``. An absent row gives zeros, with ``k = ONE``
    and ``n = 1``; a present gene without an edge gives ``k = ONE`` and
    ``n = 1`` too. The reserve: each rate is the summed ``kcat`` of the
    role's loci present, times its scale, shifted down by 16; ``km_ps`` and
    ``km_r`` are the ``km`` of the first present locus of photosynthesis and
    respiration, else ``ONE``.
    """
    names = {}
    for entry in law["genome"]["loci"]:
        names[entry["id"]] = entry["name"]
    loci = _column(tables, "genes", "locus")
    starts = _column(tables, "genes", "edge_start")
    edge_k = _column(tables, "edges", "k")
    edge_n = _column(tables, "edges", "n")
    tf_rows = {}
    for row, gene in enumerate(_column(tables, "tf", "gene")):
        tf_rows[names[loci[gene]]] = (row, gene)
    prod = _column(tables, "tf", "prod")
    deg = _column(tables, "tf", "deg")
    clock = law["constants"]["clock"]
    alpha, beta, ks, ns = [], [], [], []
    for gene_name in clock["genes"]:
        found = tf_rows.get(gene_name)
        if found is None:
            alpha.append(0)
            beta.append(0)
            ks.append(ONE)
            ns.append(1)
            continue
        row, gene = found
        alpha.append(prod[row])
        beta.append(deg[row])
        if starts[gene + 1] > starts[gene]:
            ks.append(edge_k[starts[gene]])
            ns.append(edge_n[starts[gene]])
        else:
            ks.append(ONE)
            ns.append(1)
    light = 0
    gains = _column(tables, "rec", "gain")
    for row, gene in enumerate(_column(tables, "rec", "gene")):
        if names[loci[gene]] == clock["light"]:
            light = gains[row]
            break
    enz_rows = {}
    for row, gene in enumerate(_column(tables, "enz", "gene")):
        enz_rows[names[loci[gene]]] = row
    kcat = _column(tables, "enz", "kcat")
    km = _column(tables, "enz", "km")
    chem = law["constants"]["chem"]
    k = {}
    for role, scale_key, short in ROLES:
        total = 0
        for name in chem["enzymes"][role]:
            if name in enz_rows:
                total += kcat[enz_rows[name]]
        k[short] = (total * chem[scale_key]) >> 16
    for role, short in (("photosynthesis", "km_ps"), ("respiration", "km_r")):
        k[short] = ONE
        for name in chem["enzymes"][role]:
            if name in enz_rows:
                k[short] = km[enz_rows[name]]
                break
    return {
        "chem": k,
        "clock": {"alpha": alpha, "beta": beta, "k": ks, "light": light, "n": ns},
    }
