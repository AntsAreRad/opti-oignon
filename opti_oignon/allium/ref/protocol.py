"""The byte protocol of the engine, wire v1: the reference answer to every call.

``call(request_bytes) -> response_bytes`` never raises. A response is either
a result object or ``{"detail": ..., "refused": <code>}`` with a code from
``wire.REFUSALS``. The Rust twin (``rust/allium/src/lib.rs``) answers every
request with the same bytes; the order of the checks below is part of that
contract, since it decides which refusal a request with several defects gets.

Operations of the chassis:

* ``engine``  -- the engine's identity: versions, law and table digests,
  limits, operations and refusal codes. The handshake compares it whole.
* ``echo``    -- a document re-emitted: the codec, round trip.
* ``fx``      -- a batch of one fixed-point primitive.
* ``rng``     -- draws from a keyed stream, a key, or positional noise.
* ``bulk``    -- a compact integer array packed or unpacked.
* ``fact_id`` -- the identity of a fact: body digest and event id.
* ``fact_envelope`` -- the event ids of envelopes whose body is already a
  digest, as a redacted fact keeps it.
* ``law``     -- a law file's name, version, provisional flag and digest.
* ``genome_found``   -- a founder genome from a 32-byte seed and a law.
* ``genome_corner``  -- a genome at a corner of the law's box.
* ``genome_decode``  -- a genome checked against its law: record counts.
* ``genome_compile`` -- a genome's compiled tables and their digest.
* ``grid``    -- local time, the next local midnight and the civil date of
  explicit minutes of life, under a birth and an explicit offset list.
* ``advance`` -- a life folded from its genesis, or from a state returned
  before, over its facts up to a minute or a budget of work
  (``ref/world.py``).
* ``timeline`` -- the law kinds of a life folded alone: the law in force,
  its params and pin, the pending change, the offset and the daily firing
  minutes (``ref/world.py``).
* ``phon_table``       -- the phonology table as the engine parsed it, with a
  byte classifier, for an exhaustive comparison of the two engines.
* ``phon_lex``         -- a being's language block, from its genome or,
  as a fallback, from its seed alone.
* ``phon_inventory``   -- the phonology each block decodes to.
* ``phon_licit``       -- whether forms are sayable, and their syllables.
* ``phon_invent``      -- coinages for concepts, rejected candidates shown
  only by their digest.
* ``phon_first_sound`` -- a being's first sound.
* ``phon_sas``         -- the 2048-form word list of a block, and six words
  per digest (or digests back from six words).
* ``phon_taboo``       -- whether strings touch the taboo list.

The genome operations read the embedded law and pool files, validated once
per process and remembered by the SHA-256 of the file bytes
(``ref/lawdata.py``); ``decode`` and ``compile`` never read the pool.
Parsing a law is outside ``work``.
"""

import hashlib

from .. import fx, lawfiles, rng, wire
from ..wire import Refused
from . import civil, journal, lawdata, world
from .organs import compile as organ_compile
from .organs import genome, phon

checkpoint_before_apply = True

ENGINE_VERSION = "0.1.0"
WIRE_VERSION = 1

LIMITS = {
    "body": journal.BODY_LIMIT,
    "depth": wire.MAX_DEPTH,
    "input": 1 << 20,
    "items": 100000,
    "notes": 64,
    "state": 1 << 19,
    "steps": fx.STEPS_MAX,
}
OPS = ("advance", "bulk", "echo", "engine", "fact_envelope", "fact_id", "fx", "genome_compile", "genome_corner",
       "genome_decode", "genome_found", "grid", "law", "phon_first_sound", "phon_invent", "phon_inventory",
       "phon_lex", "phon_licit", "phon_sas", "phon_table", "phon_taboo", "rng", "timeline")
RNG_KINDS = ("below", "key", "noise", "stream", "unit")
_HEX = "0123456789abcdef"


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value, length):
    if not isinstance(value, str) or len(value) != length:
        return False
    for char in value:
        if char not in _HEX:
            return False
    return True


def _fields(request, required, optional=()):
    """Exactly the required keys, and optional ones only from their list."""
    for name in request:
        if name not in required and name not in optional:
            raise Refused("bad_request", "fields")
    for name in required:
        if name not in request:
            raise Refused("bad_request", "fields")


# The parsed value and canonical digest of each embedded file, by the SHA-256 of its bytes.
_FILES = {}


def _file(data):
    key = hashlib.sha256(data).hexdigest()
    if key not in _FILES:
        value = wire.parse(data, lenient=True)
        _FILES[key] = (value, lawfiles.digest(value))
    return _FILES[key]


def _engine():
    laws = {}
    for name in lawfiles.LAWS:
        law, digest = _file(lawfiles.law_bytes(name))
        laws[name] = {
            "digest": digest,
            "provisional": law["provisional"],
            "version": law["version"],
        }
    tables = {}
    for name in lawfiles.TABLES:
        tables[name] = _file(lawfiles.table_bytes(name))[1]
    founders = {}
    for name in lawfiles.FOUNDERS:
        founders[name] = _file(lawfiles.founders_bytes(name))[1]
    return {
        "codes": list(lawdata.CODES),
        "domains": sorted(genome.DOMAINS + phon.DOMAINS + world.DOMAINS),
        "engine": ENGINE_VERSION,
        "founders": founders,
        "genome_schema": genome.SCHEMA,
        "laws": laws,
        "limits": dict(LIMITS),
        "ops": list(OPS),
        "refusals": list(wire.REFUSALS),
        "tables": tables,
        "wire": WIRE_VERSION,
    }


def engine_info():
    """The engine's identity as OCJ bytes: what the handshake compares."""
    return wire.emit(_engine())


def _op_echo(request):
    _fields(request, ("doc", "op", "v"))
    doc = request["doc"]
    if len(wire.emit(doc)) > LIMITS["state"]:
        raise Refused("limit", "state size")
    return {"doc": doc}


def _op_fx(request):
    _fields(request, ("args", "fn", "op", "v"), ("budget", "table"))
    name = request["fn"]
    if not isinstance(name, str) or name not in fx.PRIMITIVES:
        raise Refused("unknown_op", "fx function")
    function, arity = fx.PRIMITIVES[name]
    table = None
    if name == "lut":
        if "table" not in request:
            raise Refused("bad_request", "table")
        table = request["table"]
        if not isinstance(table, list) or len(table) != 257:
            raise Refused("bad_request", "table")
        for entry in table:
            if not _is_int(entry) or entry < fx.I32_MIN or entry > fx.I32_MAX:
                raise Refused("bad_request", "table")
    elif "table" in request:
        raise Refused("bad_request", "fields")
    budget = request.get("budget")
    if budget is not None and (not _is_int(budget) or budget < 0):
        raise Refused("bad_request", "budget")
    args = request["args"]
    if not isinstance(args, list):
        raise Refused("bad_request", "args")
    if len(args) > LIMITS["items"]:
        raise Refused("limit", "items")
    steps = 0
    for item in args:
        if not isinstance(item, list) or len(item) != arity:
            raise Refused("bad_request", "args")
        for value in item:
            if not _is_int(value):
                raise Refused("bad_request", "args")
        if name == "decay_iter":
            steps += min(max(item[2], 0), fx.STEPS_MAX)
    if steps > LIMITS["steps"]:
        raise Refused("limit", "steps")
    sine = lawfiles.sine() if name in ("sin_b", "cos_b") else None
    work = fx.Work()
    out = []
    for item in args:
        if name in ("sin_b", "cos_b"):
            out.append(function(item[0], sine, work))
        elif name == "lut":
            out.append(function(item[0], table, work))
        else:
            out.append(function(*item, work))
    if budget is not None and work.units > budget:
        raise Refused("budget", "fx")
    return {"alarm": work.alarm, "out": out, "work": work.units}


def _op_rng(request):
    kind = request.get("kind")
    if kind == "below":
        _fields(request, ("bound", "domain", "index", "kind", "n", "op", "seed", "v"))
    else:
        _fields(request, ("domain", "index", "kind", "n", "op", "seed", "v"))
    if not isinstance(kind, str) or kind not in RNG_KINDS:
        raise Refused("bad_request", "kind")
    seed = request["seed"]
    if not _is_hex(seed, 64):
        raise Refused("bad_request", "seed")
    domain = request["domain"]
    if not isinstance(domain, str) or not domain:
        raise Refused("bad_request", "domain")
    index = request["index"]
    if not _is_int(index) or index < 0:
        raise Refused("bad_request", "index")
    count = request["n"]
    if not _is_int(count) or count < 1:
        raise Refused("bad_request", "n")
    if count > LIMITS["items"]:
        raise Refused("limit", "items")
    if kind == "key" and count != 1:
        raise Refused("bad_request", "n")
    seed_bytes = bytes.fromhex(seed)
    if kind == "key":
        return {"out": [rng.key(seed_bytes, domain, (index,)).hex()], "work": 1}
    if kind == "noise":
        k = int.from_bytes(rng.key(seed_bytes, domain, (index,))[:8], "big")
        return {"out": [format(rng.noise(k, c), "016x") for c in range(count)], "work": count}
    stream = rng.Stream(seed_bytes, domain, index)
    if kind == "stream":
        return {"out": [format(stream.next_u64(), "016x") for _ in range(count)], "work": count}
    if kind == "unit":
        return {"out": [stream.unit() for _ in range(count)], "work": count}
    bound = request["bound"]
    if not _is_int(bound) or bound < 1:
        raise Refused("bad_request", "bound")
    return {"out": [stream.below(bound) for _ in range(count)], "work": count}


def _op_bulk(request):
    if "pack" in request:
        _fields(request, ("op", "pack", "v"))
        pack = request["pack"]
        if not isinstance(pack, dict) or sorted(pack) != ["type", "values"]:
            raise Refused("bad_request", "pack")
        values = pack["values"]
        if not isinstance(values, list) or len(values) > LIMITS["items"]:
            raise Refused("bad_request", "pack")
        if not isinstance(pack["type"], str):
            raise Refused("bad_request", "pack")
        return {"text": wire.pack_bulk(pack["type"], values)}
    _fields(request, ("op", "unpack", "v"))
    kind, values = wire.unpack_bulk(request["unpack"])
    return {"type": kind, "values": values}


def _op_fact_id(request):
    _fields(request, ("fact", "op", "v"))
    return journal.fact_id(request["fact"])


def _op_fact_envelope(request):
    _fields(request, ("envelopes", "op", "v"))
    envelopes = request["envelopes"]
    if not isinstance(envelopes, list):
        raise Refused("bad_request", "envelopes")
    if len(envelopes) > LIMITS["items"]:
        raise Refused("limit", "items")
    return journal.fact_envelope(envelopes)


def _op_law(request):
    _fields(request, ("name", "op", "v"))
    name = request["name"]
    if not isinstance(name, str) or name not in lawfiles.LAWS:
        raise Refused("unknown_law", "law")
    law = lawfiles.law(name)
    return {
        "digest": lawfiles.digest(law),
        "name": name,
        "provisional": law["provisional"],
        "version": law["version"],
    }


def _genome_law(request):
    """The requested law, its codec view and its digest; refused by name when unsound."""
    return lawdata.genome_law(request["law"])


def _genome_pool(law):
    return lawdata.genome_pool(law)


def _genome_bytes(request, lawview):
    text = request["genome"]
    if not isinstance(text, str):
        raise Refused("bad_request", "genome hex")
    if len(text) > 2 * lawview.max_bytes:
        raise Refused("limit", "genome size")
    if not genome.is_genome_hex(text):
        raise Refused("bad_request", "genome hex")
    return bytes.fromhex(text)


def _op_genome_found(request):
    _fields(request, ("law", "op", "seed", "v"))
    law, lawview, _ = _genome_law(request)
    alleles = _genome_pool(law)
    seed = request["seed"]
    if not _is_hex(seed, 64):
        raise Refused("bad_request", "seed")
    data, chosen, work = genome.found(bytes.fromhex(seed), lawview, alleles)
    return {"alleles": wire.pack_bulk("u8", chosen), "genome": data.hex(), "sha256": genome.sha256(data),
            "work": work}


def _op_genome_corner(request):
    _fields(request, ("corner", "law", "op", "v"))
    _, lawview, _ = _genome_law(request)
    k = request["corner"]
    if not _is_int(k) or not 0 <= k <= genome.CORNER_MAX:
        raise Refused("bad_request", "corner")
    data, work = genome.corner(k, lawview)
    return {"genome": data.hex(), "sha256": genome.sha256(data), "work": work}


def _op_genome_decode(request):
    _fields(request, ("genome", "law", "op", "v"))
    _, lawview, _ = _genome_law(request)
    data = _genome_bytes(request, lawview)
    value = genome.decode(data, lawview)
    records = 0
    counts = []
    for chrom in value["chromosomes"]:
        counts.append(len(chrom))
        records += len(chrom)
    return {"chromosomes": counts, "sha256": genome.sha256(data), "work": (len(data) + 63) // 64 + records}


def _op_genome_compile(request):
    _fields(request, ("genome", "law", "op", "v"))
    law, lawview, digest = _genome_law(request)
    data = _genome_bytes(request, lawview)
    tables, work = organ_compile.compile_genome(data, law, lawview, digest)
    return {"sha256": organ_compile.tables_digest(tables), "tables": tables, "work": work}


# ---------------------------------------------------------------------------
# Civil time
# ---------------------------------------------------------------------------

def _grid_offsets(request):
    """The offset list of a ``grid`` request, ``[(t, z), ...]`` with ``t`` never decreasing."""
    entries = _list(request, "tz")
    checked = []
    last = 0
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2 or not _is_int(entry[0]) or not _is_int(entry[1]):
            raise Refused("bad_request", "tz")
        t, z = entry
        if not 0 <= t <= civil.T_MAX or t < last or not civil.offset_ok(z):
            raise Refused("bad_request", "tz")
        checked.append((t, z))
        last = t
    return checked


def _offset_at(entries, t, birth_tz):
    """The offset of the last entry at or before ``t`` (the last of a shared minute), else ``birth_tz``."""
    low, high = 0, len(entries)
    while low < high:
        middle = (low + high) // 2
        if entries[middle][0] <= t:
            low = middle + 1
        else:
            high = middle
    return entries[low - 1][1] if low else birth_tz


def _op_grid(request):
    _fields(request, ("birth", "op", "ts", "tz", "v"))
    birth = request["birth"]
    if not isinstance(birth, dict) or sorted(birth) != ["tz", "wall"]:
        raise Refused("bad_request", "birth")
    wall, birth_tz = birth["wall"], birth["tz"]
    if not _is_int(wall) or not civil.WALL_MIN <= wall <= wire.MAX_INT \
            or not _is_int(birth_tz) or not civil.offset_ok(birth_tz):
        raise Refused("bad_request", "birth")
    entries = _grid_offsets(request)
    ts = _list(request, "ts")
    for t in ts:
        if not _is_int(t) or not 0 <= t <= civil.T_MAX:
            raise Refused("bad_request", "ts")
    b = wall // 60
    out = []
    for t in ts:
        z = _offset_at(entries, t, birth_tz)
        day, minute = civil.local(b, t, z)
        out.append({
            "civil": list(civil.civil_from_days(day)),
            "day": day,
            "fast": (b + t) % civil.FAST == 0,
            "midnight": civil.next_midnight(b, t, z),
            "minute": minute,
            "offset": z,
        })
    return {"out": out, "work": len(ts)}


# ---------------------------------------------------------------------------
# Phonology
# ---------------------------------------------------------------------------

class _Lang:
    __slots__ = ("table", "taboo_value", "taboo", "phon_digest", "taboo_digest")


_LANGS = {}
_PIN_KEYS = ["name", "sha256"]


def _pin(lang, key, names, detail):
    pin = lang.get(key)
    if not isinstance(pin, dict) or sorted(pin) != _PIN_KEYS or pin["name"] not in names:
        raise Refused("unknown_law", detail)
    value, digest = _file(lawfiles.table_bytes(pin["name"]))
    if pin["sha256"] != digest:
        raise Refused("unknown_law", detail)
    return pin["name"], value, digest


def _phon_law(request):
    """The phonology and taboo tables the requested law pins, checked; refused by name when unsound."""
    name = request["law"]
    if not isinstance(name, str) or name not in lawfiles.LAWS:
        raise Refused("unknown_law", "law")
    law_data = lawfiles.law_bytes(name)
    law, _digest = _file(law_data)
    lang = law.get("lang")
    if not isinstance(lang, dict) or sorted(lang) != ["phon", "taboo"]:
        raise Refused("unknown_law", "phon digest")
    phon_name, phon_value, phon_digest = _pin(lang, "phon", lawfiles.PHON_TABLES, "phon digest")
    key = hashlib.sha256(law_data + lawfiles.table_bytes(phon_name)).hexdigest()
    if phon.validate_table(phon_value):
        raise Refused("unknown_law", "phon table")
    taboo_name, taboo_value, taboo_digest = _pin(lang, "taboo", lawfiles.TABOO_TABLES, "taboo digest")
    key = hashlib.sha256(key.encode("ascii") + lawfiles.table_bytes(taboo_name)).hexdigest()
    if key not in _LANGS:
        table = phon.Table(phon_value)
        if phon.validate_taboo(taboo_value, table):
            _LANGS[key] = None
        else:
            ctx = _Lang()
            ctx.table = table
            ctx.taboo_value = taboo_value
            ctx.taboo = [tuple(entry) for entry in taboo_value["entries"]]
            ctx.phon_digest = phon_digest
            ctx.taboo_digest = taboo_digest
            _LANGS[key] = ctx
    ctx = _LANGS[key]
    if ctx is None:
        raise Refused("unknown_law", "taboo table")
    return ctx


def _lex(value, table):
    if not isinstance(value, str) or len(value) != 2 * phon.LEX_BYTES or not genome.is_genome_hex(value):
        raise Refused("bad_request", "lex")
    block = bytes.fromhex(value)
    if not phon.lex_ok(block, table):
        raise Refused("bad_request", "lex")
    return block


def _seed(value):
    if not _is_hex(value, 64):
        raise Refused("bad_request", "seed")
    return bytes.fromhex(value)


def _inventory(value):
    if not _is_int(value) or not 0 <= value < (1 << phon.WEIGHTS) or not value & 0x1F or not value >> 5:
        raise Refused("bad_request", "inventory")
    return value


def _taboo_extra(request, table):
    if "taboo_extra" not in request:
        return []
    extra = request["taboo_extra"]
    if not isinstance(extra, list):
        raise Refused("bad_request", "taboo")
    if len(extra) > table.taboo_extra_max:
        raise Refused("limit", "taboo")
    for entry in extra:
        if not phon.taboo_pair(entry):
            raise Refused("bad_request", "taboo")
    return [tuple(entry) for entry in extra]


def _list(request, key, detail=None):
    value = request[key]
    if not isinstance(value, list):
        raise Refused("bad_request", detail or key)
    if len(value) > LIMITS["items"]:
        raise Refused("limit", "items")
    return value


def _forms(value, name, table):
    if not isinstance(value, list):
        raise Refused("bad_request", name)
    if len(value) > table.anchored_max:
        raise Refused("limit", name)
    for form in value:
        if not isinstance(form, str) or phon.FORM.fullmatch(form) is None:
            raise Refused("bad_request", name)
    return value


def _op_phon_table(request):
    _fields(request, ("law", "op", "v"))
    ctx = _phon_law(request)
    table = ctx.table
    value = table.value
    classify = [table.index.get(chr(byte), -1) for byte in range(256)]
    lengths = sorted({entry[0]: True for entry in ctx.taboo})
    return {
        "alphabet": value["alphabet"],
        "anchored_max": table.anchored_max,
        "anchored_total_max": table.anchored_total_max,
        "bias": [list(row) for row in table.bias],
        "classify": wire.pack_bulk("i8", classify),
        "features": [list(row) for row in table.features],
        "first_sound_exclude": list(table.first_exclude),
        "floor_consonants": value["floor_consonants"],
        "floor_vowels": list(value["floor_vowels"]),
        "fold": [list(entry) for entry in value["fold"]],
        "form_max": value["form_max"],
        "invent_tries": value["invent_tries"],
        "lex_box": [list(pair) for pair in table.lex_box],
        "phon": ctx.phon_digest,
        "potential_min": table.potential_min,
        "sas": dict(value["sas"]),
        "shown_max": value["shown_max"],
        "taboo": {"entries": len(ctx.taboo), "lengths": lengths, "sha256": ctx.taboo_digest},
        "taboo_extra_max": table.taboo_extra_max,
        "taboo_max": table.taboo_max,
        "taboo_window": list(value["taboo_window"]),
        "templates": [list(t) for t in value["templates"]],
        "work": phon.WEIGHTS + 256,
    }


def _op_phon_lex(request):
    _fields(request, ("law", "op", "v"), ("genome", "seed"))
    if ("genome" in request) == ("seed" in request):
        raise Refused("bad_request", "fields")
    ctx = _phon_law(request)
    if "genome" in request:
        law, lawview, digest = _genome_law(request)
        data = _genome_bytes(request, lawview)
        tables, work = organ_compile.compile_genome(data, law, lawview, digest)
        block = phon.lex_from_tables(tables)
        if not phon.lex_ok(block, ctx.table):
            raise Refused("engine_panic", "lex block")
        return {"lex": block.hex(), "source": "genome", "work": work + phon.LEX_BYTES}
    block, draws = phon.fallback_lex(_seed(request["seed"]), ctx.table)
    return {"lex": block.hex(), "source": "seed", "work": draws}


def _op_phon_inventory(request):
    _fields(request, ("law", "lex", "op", "v"))
    ctx = _phon_law(request)
    blocks = [_lex(value, ctx.table) for value in _list(request, "lex")]
    out = [phon.phonology_value(phon.decode(block, ctx.table)) for block in blocks]
    return {"out": out, "work": phon.LEX_BYTES * len(blocks)}


def _op_phon_licit(request):
    _fields(request, ("forms", "law", "lex", "op", "v"), ("inventory",))
    ctx = _phon_law(request)
    block = _lex(request["lex"], ctx.table)
    inventory = _inventory(request["inventory"]) if "inventory" in request else None
    forms = _list(request, "forms")
    for form in forms:
        if not isinstance(form, str):
            raise Refused("bad_request", "forms")
    ph = phon.decode(block, ctx.table, inventory)
    out = []
    work = phon.LEX_BYTES
    for form in forms:
        reason, syllables, splits = phon.licit(form, ph)
        out.append({"reason": reason} if reason else {"splits": splits, "syllables": syllables})
        work += len(form) + 1
    return {"out": out, "work": work}


_CASE_KEYS = ("anchored", "coin", "concept", "epoch", "lex", "others", "seed", "signs", "syllables")


def _op_phon_invent(request):
    _fields(request, ("cases", "law", "op", "v"), ("budget", "taboo_extra"))
    ctx = _phon_law(request)
    table = ctx.table
    budget = request.get("budget")
    if budget is not None and (not _is_int(budget) or budget < 0):
        raise Refused("bad_request", "budget")
    extra = _taboo_extra(request, table)
    cases = _list(request, "cases")
    checked = []
    lists = 0
    for case in cases:
        if not isinstance(case, dict) or any(k not in case for k in _CASE_KEYS) \
                or any(k not in _CASE_KEYS and k != "inventory" for k in case):
            raise Refused("bad_request", "case fields")
        block = _lex(case["lex"], table)
        seed = _seed(case["seed"])
        for key, high in (("concept", wire.MAX_INT), ("coin", wire.MAX_INT)):
            if not _is_int(case[key]) or not 0 <= case[key] <= high:
                raise Refused("bad_request", key)
        signs = case["signs"]
        if not isinstance(signs, list) or len(signs) != 4 or any(not _is_int(s) or s not in (-1, 0, 1) for s in signs):
            raise Refused("bad_request", "signs")
        if not _is_int(case["epoch"]) or not 0 <= case["epoch"] <= (1 << 32) - 1:
            raise Refused("bad_request", "epoch")
        if not _is_int(case["syllables"]) or not 0 <= case["syllables"] <= 3:
            raise Refused("bad_request", "syllables")
        inventory = _inventory(case["inventory"]) if "inventory" in case else None
        anchored = _forms(case["anchored"], "anchored", table)
        others = _forms(case["others"], "others", table)
        lists += len(anchored) + len(others)
        if lists > table.anchored_total_max:
            raise Refused("limit", "lists")
        checked.append((block, seed, case, inventory, anchored, others))
    taboo = phon.taboo_set(ctx.taboo, extra)
    out = []
    work = 0
    for block, seed, case, inventory, anchored, others in checked:
        ph = phon.decode(block, table, inventory)
        result, cost = phon.invent_case(ph, seed, case["concept"], case["coin"], case["signs"], case["epoch"],
                                        case["syllables"], anchored, others, taboo)
        work += cost
        if budget is not None and work > budget:
            raise Refused("budget", "phon invent")
        out.append(result)
    return {"out": out, "work": work}


def _op_phon_first_sound(request):
    _fields(request, ("cases", "law", "op", "v"))
    ctx = _phon_law(request)
    checked = []
    for case in _list(request, "cases"):
        if not isinstance(case, dict) or sorted(case) != ["lex", "seed"]:
            raise Refused("bad_request", "case fields")
        checked.append((_lex(case["lex"], ctx.table), _seed(case["seed"])))
    out = []
    work = 0
    for block, seed in checked:
        p, draws = phon.first_sound(seed, phon.decode(block, ctx.table))
        out.append(p)
        work += phon.LEX_BYTES + draws
    return {"out": out, "symbols": "".join(phon.ALPHABET[p] for p in out), "work": work}


def _op_phon_sas(request):
    _fields(request, ("law", "lex", "op", "v"), ("digests", "phrases", "taboo_extra"))
    ctx = _phon_law(request)
    block = _lex(request["lex"], ctx.table)
    extra = _taboo_extra(request, ctx.table)
    digests = None
    if "digests" in request:
        digests = _list(request, "digests")
        for digest in digests:
            if not _is_hex(digest, 64):
                raise Refused("bad_request", "digests")
    phrases = None
    if "phrases" in request:
        phrases = _list(request, "phrases")
        for phrase in phrases:
            if not isinstance(phrase, list) or len(phrase) != 6 \
                    or any(not isinstance(w, str) or phon.FORM.fullmatch(w) is None for w in phrase):
                raise Refused("bad_request", "phrases")
    words, candidates, work = phon.sas_list(phon.decode(block, ctx.table), phon.taboo_set(ctx.taboo, extra))
    out = {"candidates": candidates, "list": words, "sha256": hashlib.sha256(wire.emit(words)).hexdigest()}
    if digests is not None:
        indices = [phon.sas_indices(bytes.fromhex(digest))[0] for digest in digests]
        out["indices"] = indices
        out["words"] = [[words[i] for i in row] for row in indices]
        work += 6 * len(digests)
    if phrases is not None:
        positions = {word: i for i, word in enumerate(words)}
        out["parsed"] = [phon.sas_parse(phrase, positions) for phrase in phrases]
        work += 6 * len(phrases)
    out["work"] = work
    return out


def _op_phon_taboo(request):
    _fields(request, ("law", "op", "strings", "v"), ("taboo_extra",))
    ctx = _phon_law(request)
    extra = _taboo_extra(request, ctx.table)
    strings = _list(request, "strings")
    for text in strings:
        if not isinstance(text, str) or phon.SHOWN.fullmatch(text) is None:
            raise Refused("bad_request", "strings")
    taboo = phon.taboo_set(ctx.taboo, extra)
    out = [phon.taboo_hit(text, taboo) for text in strings]
    work = 0
    for text in strings:
        work += phon.taboo_work(len(text))
    return {"out": out, "work": work}


_HANDLERS = {
    "advance": lambda request, size: world.op_advance(request, size, LIMITS),
    "bulk": _op_bulk,
    "echo": _op_echo,
    "engine": lambda request: (_fields(request, ("op", "v")), _engine())[1],
    "fact_envelope": _op_fact_envelope,
    "fact_id": _op_fact_id,
    "fx": _op_fx,
    "genome_compile": _op_genome_compile,
    "genome_corner": _op_genome_corner,
    "genome_decode": _op_genome_decode,
    "genome_found": _op_genome_found,
    "grid": _op_grid,
    "law": _op_law,
    "phon_first_sound": _op_phon_first_sound,
    "phon_invent": _op_phon_invent,
    "phon_inventory": _op_phon_inventory,
    "phon_lex": _op_phon_lex,
    "phon_licit": _op_phon_licit,
    "phon_sas": _op_phon_sas,
    "phon_table": _op_phon_table,
    "phon_taboo": _op_phon_taboo,
    "rng": _op_rng,
    "timeline": lambda request: world.op_timeline(request, LIMITS),
}
# The operations whose answer counts the request's own size.
_SIZED = ("advance",)


def _answer(data):
    if len(data) > LIMITS["input"]:
        raise Refused("limit", "input size")
    request = wire.parse(data)
    if not isinstance(request, dict):
        raise Refused("bad_request", "request")
    op = request.get("op")
    if not isinstance(op, str):
        raise Refused("bad_request", "op")
    if request.get("v") != WIRE_VERSION or not _is_int(request.get("v")):
        raise Refused("unknown_op", "wire version")
    if op not in _HANDLERS:
        raise Refused("unknown_op", "op")
    if op in _SIZED:
        return _HANDLERS[op](request, len(data))
    return _HANDLERS[op](request)


def call(data):
    """Answer one request: OCJ bytes in, OCJ bytes out, never an exception."""
    try:
        return wire.emit(_answer(bytes(data)))
    except Refused as refusal:
        return wire.emit(refusal.as_value())
