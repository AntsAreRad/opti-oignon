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
* ``law``     -- a law file's name, version, provisional flag and digest.
* ``genome_found``   -- a founder genome from a 32-byte seed and a law.
* ``genome_corner``  -- a genome at a corner of the law's box.
* ``genome_decode``  -- a genome checked against its law: record counts.
* ``genome_compile`` -- a genome's compiled tables and their digest.

The genome operations read the embedded law and pool files, validated once
per process and remembered by the SHA-256 of the file bytes; ``decode`` and
``compile`` never read the pool. Parsing a law is outside ``work``.
"""

import hashlib

from .. import fx, lawfiles, rng, wire
from ..wire import Refused
from .organs import compile as organ_compile
from .organs import genome

checkpoint_before_apply = True

ENGINE_VERSION = "0.1.0"
WIRE_VERSION = 1

LIMITS = {
    "body": 4096,
    "depth": wire.MAX_DEPTH,
    "input": 1 << 20,
    "items": 100000,
    "state": 1 << 19,
    "steps": fx.STEPS_MAX,
}
OPS = ("bulk", "echo", "engine", "fact_id", "fx", "genome_compile", "genome_corner", "genome_decode",
       "genome_found", "law", "rng")
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


def _engine():
    laws = {}
    for name in lawfiles.LAWS:
        law = lawfiles.law(name)
        laws[name] = {
            "digest": lawfiles.digest(law),
            "provisional": law["provisional"],
            "version": law["version"],
        }
    tables = {}
    for name in lawfiles.TABLES:
        tables[name] = lawfiles.digest(lawfiles.table(name))
    founders = {}
    for name in lawfiles.FOUNDERS:
        founders[name] = lawfiles.digest(lawfiles.founders(name))
    return {
        "domains": list(genome.DOMAINS),
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


_FACT_FIELDS = ("being", "body", "kind", "laws", "origin", "oseq", "t")


def _op_fact_id(request):
    _fields(request, ("fact", "op", "v"))
    fact = request["fact"]
    if not isinstance(fact, dict):
        raise Refused("bad_fact", "fact")
    for name in fact:
        if name not in _FACT_FIELDS:
            raise Refused("bad_fact", "fields")
    for name in _FACT_FIELDS:
        if name not in fact:
            raise Refused("bad_fact", "fields")
    if not _is_hex(fact["being"], 32):
        raise Refused("bad_fact", "being")
    kind = fact["kind"]
    if not isinstance(kind, str) or not 1 <= len(kind) <= 32:
        raise Refused("bad_fact", "kind")
    for char in kind:
        if not ("a" <= char <= "z" or "0" <= char <= "9" or char == "_"):
            raise Refused("bad_fact", "kind")
    if not _is_int(fact["laws"]) or fact["laws"] < 0:
        raise Refused("bad_fact", "laws")
    if not _is_hex(fact["origin"], 16):
        raise Refused("bad_fact", "origin")
    if not _is_int(fact["oseq"]) or fact["oseq"] < 0:
        raise Refused("bad_fact", "oseq")
    if not _is_int(fact["t"]):
        raise Refused("bad_fact", "t")
    body = fact["body"]
    if not isinstance(body, dict):
        raise Refused("bad_fact", "body")
    body_bytes = wire.emit(body)
    if len(body_bytes) > LIMITS["body"]:
        raise Refused("limit", "body size")
    body_digest = hashlib.sha256(body_bytes).hexdigest()
    envelope = {name: fact[name] for name in _FACT_FIELDS}
    envelope["body"] = body_digest
    return {"body": body_digest, "eid": hashlib.sha256(wire.emit(envelope)).hexdigest()}


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


# Per-process memory of the genome laws and pools, by the SHA-256 of their file bytes.
_LAWS = {}
_POOLS = {}


def _genome_law(request):
    """The requested law, its codec view and its digest; refused by name when unsound."""
    name = request["law"]
    if not isinstance(name, str) or name not in lawfiles.LAWS:
        raise Refused("unknown_law", "law")
    data = lawfiles.law_bytes(name)
    key = hashlib.sha256(data).hexdigest()
    if key not in _LAWS:
        law = wire.parse(data, lenient=True)
        sound = not genome.validate_law(law)
        _LAWS[key] = (law, genome.view(law) if sound else None, lawfiles.digest(law))
    law, lawview, digest = _LAWS[key]
    if lawview is None:
        raise Refused("unknown_law", "genome law")
    return law, lawview, digest


def _genome_pool(law):
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


_HANDLERS = {
    "bulk": _op_bulk,
    "echo": _op_echo,
    "engine": lambda request: (_fields(request, ("op", "v")), _engine())[1],
    "fact_id": _op_fact_id,
    "fx": _op_fx,
    "genome_compile": _op_genome_compile,
    "genome_corner": _op_genome_corner,
    "genome_decode": _op_genome_decode,
    "genome_found": _op_genome_found,
    "law": _op_law,
    "rng": _op_rng,
}


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
    return _HANDLERS[op](request)


def call(data):
    """Answer one request: OCJ bytes in, OCJ bytes out, never an exception."""
    try:
        return wire.emit(_answer(bytes(data)))
    except Refused as refusal:
        return wire.emit(refusal.as_value())
