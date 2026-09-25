"""The genome, schema 1: layout, law and pool validation, codec, founders, corners.

A genome is diploid: ``pairs`` pairs of chromosomes, each chromosome a list of
fixed 16-byte records. The order of records on a chromosome is data -- the
promoter of a gene is the run of CIS records just before it -- so nothing
here ever sorts a chromosome.

Record layout (every multi-byte field big-endian)::

    0     kind    u8      1..16
    1     flags   u8      bits 0-1 dominance mode, 2 essential, 3 duplicable,
                          4 stress responsive, 5-7 zero; equal to the law's
    2-3   locus   u16     (home pair << 8) | ordinal; 0x8000 and up reserved
    4-5   stages  u16     stage mask, equal to the law's
    6-15  body    10      per kind; every unassigned byte or bit is zero

Genome framing: ``"ALG" 0x01``, schema u16 = 1, ploidy u8 = 2, pairs u8, then
for each chromosome (pair-major, homolog-minor) a u16 record count and the
records.

The layout of each kind -- field names and order, positions, types, the
fields a genome may not vary (structural, S), the fields that name a species
the gene writes (W) or reads (R) -- is compiled into both engines. The boxes,
the loci and the species classes are law data, and a law whose kind table
disagrees with this layout is refused.

The decoder is total: it raises nothing but ``wire.Refused``, and the first
defect it meets, scanning left to right, decides the refusal.
"""

import hashlib

from ... import rng
from ...wire import Refused

checkpoint_before_apply = True

MAGIC = b"ALG\x01"
SCHEMA = 1
PLOIDY = 2
STAGES = 11
RECORD = 16
BODY = 10
HEADER = 8
LOCUS_LIMIT = 0x8000
STAGE_MASK_MAX = (1 << STAGES) - 1

DOMAIN_FOUNDER = "genome.founder"
DOMAIN_CORNER = "genome.corner"
DOMAINS = (DOMAIN_CORNER, DOMAIN_FOUNDER)
CORNER_MAX = (1 << 32) - 1

DOM_MAX = 0
ADD = 1
LOAD_REC = 2
FLAG_ESSENTIAL = 0x04
FLAG_MODE = 0x03
FLAG_UNUSED = 0xE0

CLASSES = ("input", "token", "conserved", "open", "free")
SPECIES = 64
SPECIES_NONE = 255

# Field marks.
S = 1  # structural: never jittered or averaged; fixed on ADD and LOAD_REC loci
W = 2  # names a species the gene writes (or consumes): fixed on every locus
R = 4  # reads a species: never a token species
P = 8  # species reference: legal only when <= 63 or == 255


def _field(name, byte, bits=8, signed=False, marks=0, shift=0, scale=1):
    return (name, byte, shift, bits, signed, marks, scale)


def _u8(name, byte, marks=0):
    return _field(name, byte, 8, False, marks)


def _i8(name, byte, marks=0, scale=1):
    return _field(name, byte, 8, True, marks, scale=scale)


def _u16(name, byte, marks=0, scale=1):
    return _field(name, byte, 16, False, marks, scale=scale)


def _i16(name, byte, marks=0, scale=1):
    return _field(name, byte, 16, True, marks, scale=scale)


def _bits(name, byte, shift, bits, marks=0):
    return _field(name, byte, bits, False, marks, shift=shift)


# Field tuple positions.
F_NAME, F_BYTE, F_SHIFT, F_BITS, F_SIGNED, F_MARKS, F_SCALE = range(7)

# (code, name, fields, allowed modes, strength rule, reserved)
# Strength rules: ("field", i), ("abs", i), ("hash", None), ("sumabs", (i, ...)).
KINDS = (
    (1, "tf", (
        _u8("out", 0, S | W), _u16("prod", 1), _u16("deg", 3),
        _i16("bias", 5, scale=16), _u16("rate", 7),
    ), (DOM_MAX, ADD), ("field", 1), False),
    (2, "cis", (
        _u8("src", 0, S | R), _i16("w", 1, scale=16), _u16("K", 3, scale=8),
        _u8("n", 5), _u8("mode", 6, S),
    ), (DOM_MAX, ADD), ("abs", 1), False),
    (3, "enz", (
        _u8("s1", 0, S | W | R | P), _u8("s2", 1, S | W | R | P),
        _u8("p1", 2, S | W | P), _u8("p2", 3, S | W | P),
        _u16("kcat", 4), _u16("Km", 6, scale=8), _u16("yield", 8),
    ), (DOM_MAX, ADD), ("field", 4), False),
    (4, "rec", (
        _u8("channel", 0, S), _u8("species", 1, S | W),
        _i16("gain", 2, scale=16), _u16("threshold", 4),
    ), (DOM_MAX, ADD), ("abs", 2), False),
    (5, "morph", (
        _u8("pred", 0, S), _u8("guard", 1, S | R), _u16("threshold", 2, scale=8),
        _u8("succ", 4, S), _u16("rate", 5), _i8("angle", 7, scale=8),
    ), (DOM_MAX, ADD), ("field", 4), False),
    (6, "imm", (
        _u8("class", 0, S), _u16("strength", 1), _u16("cost", 3),
    ), (DOM_MAX,), ("field", 1), False),
    (7, "pig", (
        _u8("class", 0, S), _u8("allele", 1),
    ), (DOM_MAX,), ("field", 1), False),
    (8, "plast", (
        _u8("class", 0, S), _i8("A", 1), _i8("Aneg", 2), _i8("B", 3), _i8("C", 4), _i8("D", 5),
        _bits("LR", 6, 4, 4), _bits("TAU_X", 6, 0, 4), _bits("TAU_Y", 7, 4, 4), _bits("TAU_E", 7, 0, 4),
        _bits("M_SEL", 8, 5, 3, S), _bits("M_SIGN", 8, 4, 1, S),
    ), (DOM_MAX, ADD), ("abs", 1), True),
    (9, "temp", (
        _u8("channel", 0, S), _i16("weight", 1), _i16("base", 3), _u8("habituation", 5),
        _u16("curiosity", 6),
    ), (DOM_MAX, ADD), ("abs", 1), True),
    (10, "lex", (
        _u8("offset", 0, S),
        _u8("d0", 1), _u8("d1", 2), _u8("d2", 3), _u8("d3", 4), _u8("d4", 5),
        _u8("d5", 6), _u8("d6", 7), _u8("d7", 8), _u8("d8", 9),
    ), (DOM_MAX, ADD), ("hash", None), True),
    (11, "shape", (
        _u8("trait", 0, S), _u16("value", 1), _u16("window", 3),
    ), (DOM_MAX, ADD), ("field", 1), True),
    (12, "vern", (
        _u8("D_enter", 0), _u8("v_cold", 1), _u8("v_dorm", 2), _u16("VU_req", 3),
        _u8("stalk_days", 5), _u8("offsets", 6), _u8("rest_days", 7), _u16("a_min", 8),
    ), (ADD,), ("field", 3), True),
    (13, "prc", (
        _u8("block", 0, S), _i16("shift0", 1), _i16("shift1", 3), _i16("shift2", 5), _i16("shift3", 7),
    ), (ADD,), ("sumabs", (1, 2, 3, 4)), True),
    (14, "te", (
        _u16("activity", 0), _u8("target_bias", 2),
    ), (DOM_MAX, ADD), ("field", 0), False),
    (15, "load", (
        _u8("target", 0, S | W), _u16("effect", 1),
    ), (LOAD_REC,), ("field", 1), False),
    (16, "brain", (
        _u8("param", 0, S), _i16("value", 1),
    ), (DOM_MAX, ADD), ("abs", 1), True),
)

K_CODE, K_NAME, K_FIELDS, K_MODES, K_STRENGTH, K_RESERVED = range(6)
KIND_BY_CODE = {kind[K_CODE]: kind for kind in KINDS}
KIND_BY_NAME = {kind[K_NAME]: kind for kind in KINDS}
KIND_CIS = 2
KIND_LOAD = 15


def type_range(field):
    """The integer range a field's position can hold."""
    bits = field[F_BITS]
    if field[F_SIGNED]:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def _used_bits(fields):
    used = [0] * BODY
    for field in fields:
        byte, shift, bits = field[F_BYTE], field[F_SHIFT], field[F_BITS]
        if bits == 16:
            used[byte] = 0xFF
            used[byte + 1] = 0xFF
        elif bits == 8:
            used[byte] = 0xFF
        else:
            used[byte] |= ((1 << bits) - 1) << shift
    return tuple(used)


USED = {kind[K_CODE]: _used_bits(kind[K_FIELDS]) for kind in KINDS}


def read_field(body, field):
    byte, shift, bits = field[F_BYTE], field[F_SHIFT], field[F_BITS]
    if bits == 16:
        value = (body[byte] << 8) | body[byte + 1]
    elif bits == 8:
        value = body[byte]
    else:
        return (body[byte] >> shift) & ((1 << bits) - 1)
    if field[F_SIGNED] and value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _write_field(body, field, value):
    byte, shift, bits = field[F_BYTE], field[F_SHIFT], field[F_BITS]
    raw = value & ((1 << bits) - 1)
    if bits == 16:
        body[byte] = raw >> 8
        body[byte + 1] = raw & 0xFF
    elif bits == 8:
        body[byte] = raw
    else:
        body[byte] |= raw << shift


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


NAME_MAX = 48


def _is_name(value):
    """Lowercase ASCII letters, digits and underscores, 1 to 48 characters."""
    if not isinstance(value, str) or not 1 <= len(value) <= NAME_MAX:
        return False
    for char in value:
        if not ("a" <= char <= "z" or "0" <= char <= "9" or char == "_"):
            return False
    return True


def _keys_are(value, names):
    """True when ``value`` is a dict whose keys are exactly ``names``."""
    if not isinstance(value, dict) or len(value) != len(names):
        return False
    for name in names:
        if name not in value:
            return False
    return True


def _species_legal(value):
    return value <= 63 or value == SPECIES_NONE


# ---------------------------------------------------------------------------
# The law view: what the codec needs from a validated law
# ---------------------------------------------------------------------------

class Locus:
    __slots__ = ("id", "name", "kind", "flags", "stages", "boxes", "fixed")

    def __init__(self, ident, name, kind, flags, stages, boxes):
        self.id = ident
        self.name = name
        self.kind = kind
        self.flags = flags
        self.stages = stages
        self.boxes = boxes
        self.fixed = tuple(low == high for low, high in boxes)

    @property
    def mode(self):
        return self.flags & FLAG_MODE

    @property
    def essential(self):
        return bool(self.flags & FLAG_ESSENTIAL)


class View:
    """A validated law's genome, in the form the codec reads."""

    __slots__ = ("pairs", "max_records", "max_promoter", "max_bytes", "loci", "order",
                 "chromosomes", "classes", "deg", "token", "kind_boxes")

    def __init__(self, genome):
        self.pairs = genome["pairs"]
        self.max_records = genome["max_records"]
        self.max_promoter = genome["max_promoter"]
        self.max_bytes = genome["max_bytes"]
        self.kind_boxes = {}
        for entry in genome["kinds"]:
            self.kind_boxes[entry["code"]] = tuple((f["lo"], f["hi"]) for f in entry["fields"])
        self.loci = {}
        self.order = []
        for entry in genome["loci"]:
            kind = KIND_BY_NAME[entry["kind"]]
            boxes = list(self.kind_boxes[kind[K_CODE]])
            names = [field[F_NAME] for field in kind[K_FIELDS]]
            for override in entry["box"]:
                boxes[names.index(override["field"])] = (override["lo"], override["hi"])
            locus = Locus(entry["id"], entry["name"], kind[K_CODE], entry["flags"], entry["stages"],
                          tuple(boxes))
            self.loci[locus.id] = locus
            self.order.append(locus.id)
        self.chromosomes = tuple(tuple(ids) for ids in genome["chromosomes"])
        self.classes = tuple(entry["class"] for entry in genome["species"])
        self.deg = tuple(entry["deg"] for entry in genome["species"])
        self.token = tuple(cls == "token" for cls in self.classes)


def view(law):
    """The codec's view of a law already validated by :func:`validate_law`."""
    return View(law["genome"])


# ---------------------------------------------------------------------------
# Law validation
# ---------------------------------------------------------------------------

_GENOME_KEYS = ("chromosomes", "kinds", "loci", "max_bytes", "max_promoter", "max_records",
                "pairs", "schema", "species", "stages")


def _box_ok(field, low, high):
    lo_t, hi_t = type_range(field)
    return _is_int(low) and _is_int(high) and lo_t <= low <= high <= hi_t


def _validate_kinds(kinds, out):
    if not isinstance(kinds, list) or len(kinds) != len(KINDS):
        out.append("kinds: exactly 16 entries")
        return False
    good = True
    for entry, kind in zip(kinds, KINDS):
        code, name, fields = kind[K_CODE], kind[K_NAME], kind[K_FIELDS]
        if not _keys_are(entry, ("code", "fields", "name")) or entry["code"] != code \
                or not _is_int(entry["code"]) or entry["name"] != name:
            out.append(f"kinds: entry {code} is not the engine's {name}")
            good = False
            continue
        boxes = entry["fields"]
        if not isinstance(boxes, list) or len(boxes) != len(fields):
            out.append(f"kinds: {name} fields differ from the engine layout")
            good = False
            continue
        for box, field in zip(boxes, fields):
            if not _keys_are(box, ("hi", "lo", "name")) or box["name"] != field[F_NAME]:
                out.append(f"kinds: {name} fields differ from the engine layout")
                good = False
                break
            low, high = box["lo"], box["hi"]
            if not _box_ok(field, low, high):
                out.append(f"kinds: {name}.{field[F_NAME]} box outside its type range")
                good = False
                continue
            fname = field[F_NAME]
            if (name, fname) in (("cis", "K"), ("enz", "Km")) and low < 512:
                out.append(f"kinds: {name}.{fname} lo below 512")
                good = False
            if (name, fname) == ("cis", "n") and (low < 1 or high > 4):
                out.append("kinds: cis.n outside [1, 4]")
                good = False
            if name == "plast" and fname.startswith("TAU_") and (low < 1 or high > 15):
                out.append(f"kinds: plast.{fname} outside [1, 15]")
                good = False
            if field[F_MARKS] & P and not (_species_legal(low) and _species_legal(high)):
                out.append(f"kinds: {name}.{fname} species endpoints")
                good = False
    return good


def _validate_species(species, out):
    if not isinstance(species, list) or len(species) != SPECIES:
        out.append("species: exactly 64 entries")
        return False
    good = True
    for code, entry in enumerate(species):
        if not _keys_are(entry, ("class", "code", "deg", "name")) or not _is_int(entry["code"]) \
                or entry["code"] != code:
            out.append(f"species: entry {code}")
            good = False
            continue
        if not _is_name(entry["name"]):
            out.append(f"species: {code} name")
            good = False
        cls = entry["class"]
        if cls not in CLASSES:
            out.append(f"species: {code} class")
            good = False
            continue
        deg = entry["deg"]
        if not _is_int(deg) or not 0 <= deg <= 65535:
            out.append(f"species: {code} deg")
            good = False
        elif cls in ("input", "conserved") and deg != 0:
            out.append(f"species: {code} deg of an {cls} species")
            good = False
    return good


def _class_of(species, code):
    return species[code]["class"]


def _validate_locus_fields(entry, kind, kind_boxes, species, out):
    """Overrides, then the S, W and R rules, for one locus; its effective boxes or None."""
    name = entry["name"]
    fields = kind[K_FIELDS]
    names = [field[F_NAME] for field in fields]
    boxes = list(kind_boxes)
    overrides = entry["box"]
    if not isinstance(overrides, list):
        out.append(f"loci: {name} box")
        return None
    last = -1
    for override in overrides:
        if not _keys_are(override, ("field", "hi", "lo")) or override["field"] not in names:
            out.append(f"loci: {name} names a field its kind does not have")
            return None
        index = names.index(override["field"])
        if index <= last:
            out.append(f"loci: {name} overrides out of kind field order")
            return None
        last = index
        low, high = override["lo"], override["hi"]
        k_lo, k_hi = kind_boxes[index]
        if not _is_int(low) or not _is_int(high) or not k_lo <= low <= high <= k_hi:
            out.append(f"loci: {name}.{override['field']} outside the kind box")
            return None
        boxes[index] = (low, high)
    good = True
    mode = entry["flags"] & FLAG_MODE
    for index, field in enumerate(fields):
        marks = field[F_MARKS]
        low, high = boxes[index]
        fixed = low == high
        fname = field[F_NAME]
        if marks & S and mode in (ADD, LOAD_REC) and not fixed:
            out.append(f"loci: {name}.{fname} is structural and must be fixed on this mode")
            good = False
        if marks & W:
            if not fixed:
                out.append(f"loci: {name}.{fname} names a written species and must be fixed")
                good = False
            elif not _written_class_ok(kind[K_NAME], fname, low, species):
                out.append(f"loci: {name}.{fname} writes a species of a forbidden class")
                good = False
        if marks & R and fixed and low <= 63 and _class_of(species, low) == "token":
            out.append(f"loci: {name}.{fname} reads a token species")
            good = False
        if marks & P and not (_species_legal(low) and _species_legal(high)):
            out.append(f"loci: {name}.{fname} species endpoints")
            good = False
    return tuple(boxes) if good else None


def _written_class_ok(kind_name, field_name, value, species):
    if value == SPECIES_NONE and kind_name == "enz":
        return True
    if value > 63:
        return False
    cls = _class_of(species, value)
    if kind_name == "tf":
        return cls in ("open", "free")
    if kind_name == "rec":
        return cls in ("input", "token", "open", "free")
    if kind_name == "load":
        return cls in ("open", "free")
    if kind_name == "enz":
        return cls != "token"
    return False


def validate_law(law):
    """Every defect of a law's genome section, named; ``[]`` when it is sound."""
    out = []
    if not isinstance(law, dict) or not _keys_are(law.get("genome"), _GENOME_KEYS):
        return ["genome: missing, or its keys are not exactly the schema's"]
    genome = law["genome"]
    for key in ("max_bytes", "max_promoter", "max_records", "pairs", "schema", "stages"):
        if not _is_int(genome[key]):
            out.append(f"genome: {key} is not an integer")
    if out:
        return out
    if genome["schema"] != SCHEMA:
        out.append("genome: schema")
    if genome["stages"] != STAGES:
        out.append("genome: stages")
    pairs = genome["pairs"]
    if not 1 <= pairs <= 8:
        out.append("genome: pairs outside 1..8")
    if not 1 <= genome["max_records"] <= 64:
        out.append("genome: max_records outside 1..64")
    if not 1 <= genome["max_promoter"] <= 16:
        out.append("genome: max_promoter outside 1..16")
    kinds_ok = _validate_kinds(genome["kinds"], out)
    species_ok = _validate_species(genome["species"], out)
    if out or not kinds_ok or not species_ok:
        return out
    kind_boxes = {entry["code"]: tuple((f["lo"], f["hi"]) for f in entry["fields"])
                  for entry in genome["kinds"]}
    species = genome["species"]
    loci = genome["loci"]
    if not isinstance(loci, list) or not loci:
        return ["loci: a non-empty list"]
    names = {}
    previous = -1
    kinds_of = {}
    for entry in loci:
        if not _keys_are(entry, ("box", "flags", "id", "kind", "name", "stages")):
            out.append("loci: an entry's keys are not exactly the schema's")
            continue
        ident, flags, stages = entry["id"], entry["flags"], entry["stages"]
        if not (_is_int(ident) and _is_int(flags) and _is_int(stages)):
            out.append("loci: id, flags and stages are integers")
            continue
        label = entry["name"]
        if not _is_name(label):
            out.append(f"loci: {ident} name")
            continue
        if label in names:
            out.append(f"loci: {label} named twice")
        names[label] = True
        if ident <= previous or ident >= LOCUS_LIMIT or ident < 0 or (ident >> 8) >= pairs:
            out.append(f"loci: {label} id out of order or out of range")
        previous = max(previous, ident)
        if entry["kind"] not in KIND_BY_NAME:
            out.append(f"loci: {label} kind")
            continue
        kind = KIND_BY_NAME[entry["kind"]]
        kinds_of[ident] = kind[K_CODE]
        if not 0 <= flags <= 255 or flags & FLAG_UNUSED or (flags & FLAG_MODE) not in kind[K_MODES]:
            out.append(f"loci: {label} flags")
            continue
        if not 1 <= stages <= STAGE_MASK_MAX:
            out.append(f"loci: {label} stages")
        _validate_locus_fields(entry, kind, kind_boxes[kind[K_CODE]], species, out)
    if out:
        return out
    chromosomes = genome["chromosomes"]
    if not isinstance(chromosomes, list) or len(chromosomes) != pairs:
        return ["chromosomes: one list per pair"]
    seen = {}
    for pair, ids in enumerate(chromosomes):
        if not isinstance(ids, list) or len(ids) > genome["max_records"]:
            out.append(f"chromosomes: pair {pair} list")
            continue
        run = 0
        for ident in ids:
            if not _is_int(ident) or ident not in kinds_of or (ident >> 8) != pair or ident in seen:
                out.append(f"chromosomes: pair {pair} lists {ident!r} wrongly")
                continue
            seen[ident] = True
            if kinds_of[ident] == KIND_CIS:
                run += 1
                if run > genome["max_promoter"]:
                    out.append(f"chromosomes: pair {pair} template promoter too long")
            else:
                run = 0
    if len(seen) != len(kinds_of):
        out.append("chromosomes: every locus exactly once")
    if genome["max_bytes"] != HEADER + pairs * PLOIDY * 2 + PLOIDY * RECORD * len(loci):
        out.append("genome: max_bytes")
    return out


# ---------------------------------------------------------------------------
# Pool validation
# ---------------------------------------------------------------------------

def validate_pool(law, pool):
    """Every defect of a founder pool against its (validated) law; ``[]`` when sound."""
    lawview = view(law)
    if not _keys_are(pool, ("loci", "name", "schema", "version")):
        return ["pool: keys are not exactly the schema's"]
    out = []
    if not _is_int(pool["schema"]) or pool["schema"] != 1:
        out.append("pool: schema")
    if not _is_int(pool["version"]):
        out.append("pool: version")
    founders = law.get("founders")
    if not isinstance(founders, dict) or pool["name"] != founders.get("name"):
        out.append("pool: name is not the one the law pins")
    loci = pool["loci"]
    if not isinstance(loci, list) or len(loci) != len(lawview.order):
        return out + ["pool: exactly the law's loci"]
    for entry, ident in zip(loci, lawview.order):
        if not _keys_are(entry, ("alleles", "locus")) or not _is_int(entry["locus"]) \
                or entry["locus"] != ident:
            out.append(f"pool: locus {ident} missing or out of order")
            continue
        locus = lawview.loci[ident]
        alleles = entry["alleles"]
        if not isinstance(alleles, list) or not 2 <= len(alleles) <= 6:
            out.append(f"pool: locus {ident} holds 2 to 6 alleles")
            continue
        names = {}
        shapes = []
        for allele in alleles:
            defect = _allele_defect(allele, locus, lawview)
            if defect:
                out.append(f"pool: locus {ident} {defect}")
                continue
            if allele["name"] in names:
                out.append(f"pool: locus {ident} allele named twice")
            names[allele["name"]] = True
            shape = (allele["body"], allele["window"])
            if shape in shapes:
                out.append(f"pool: locus {ident} two identical alleles")
            shapes.append(shape)
    return out


def _allele_defect(allele, locus, lawview):
    if not _keys_are(allele, ("body", "freq", "name", "window")):
        return "allele keys"
    if not _is_int(allele["freq"]) or not 1 <= allele["freq"] <= 1000:
        return "freq"
    if not _is_name(allele["name"]):
        return "allele name"
    fields = KIND_BY_CODE[locus.kind][K_FIELDS]
    body, window = allele["body"], allele["window"]
    if not isinstance(body, list) or not isinstance(window, list) \
            or len(body) != len(fields) or len(window) != len(body):
        return "body or window length"
    for index, field in enumerate(fields):
        value, win = body[index], window[index]
        low, high = locus.boxes[index]
        if not _is_int(value) or not low <= value <= high:
            return f"body {field[F_NAME]} outside its box"
        marks = field[F_MARKS]
        if marks & P and not _species_legal(value):
            return f"body {field[F_NAME]} species"
        if marks & R and value <= 63 and lawview.token[value]:
            return f"body {field[F_NAME]} reads a token species"
        if not _is_int(win) or not 0 <= win <= high - low:
            return f"window {field[F_NAME]}"
        if win and (marks & (S | W) or low == high):
            return f"window {field[F_NAME]} on a field that may not vary"
    return None


def pool_alleles(pool):
    """The pool's alleles by locus id."""
    return {entry["locus"]: entry["alleles"] for entry in pool["loci"]}


# ---------------------------------------------------------------------------
# The codec
# ---------------------------------------------------------------------------

def _box_refusal(i, j, name):
    return Refused("bad_request", f"genome box c{i} r{j} {name}")


def _decode(data, lawview):
    """Decode and check a genome; chromosomes of ``(record value, body bytes)``."""
    data = bytes(data)
    if len(data) > lawview.max_bytes:
        raise Refused("limit", "genome size")
    if len(data) < HEADER:
        raise Refused("bad_request", "genome header")
    if data[0:4] != MAGIC:
        raise Refused("bad_request", "genome magic")
    if (data[4] << 8) | data[5] != SCHEMA:
        raise Refused("bad_request", "genome schema")
    if data[6] != PLOIDY:
        raise Refused("bad_request", "genome ploidy")
    if data[7] != lawview.pairs:
        raise Refused("bad_request", "genome pairs")
    pos = HEADER
    size = len(data)
    chromosomes = []
    presence = []
    loci = lawview.loci
    token = lawview.token
    for i in range(lawview.pairs * PLOIDY):
        pair = i // PLOIDY
        if size - pos < 2:
            raise Refused("bad_request", "genome truncated")
        count = (data[pos] << 8) | data[pos + 1]
        pos += 2
        if count > lawview.max_records:
            raise Refused("limit", f"genome records c{i}")
        if size - pos < count * RECORD:
            raise Refused("bad_request", "genome truncated")
        seen = {}
        run = 0
        records = []
        for j in range(count):
            raw = data[pos:pos + RECORD]
            pos += RECORD
            kind_code = raw[0]
            if not 1 <= kind_code <= len(KINDS):
                raise Refused("bad_request", f"genome kind c{i} r{j}")
            ident = (raw[2] << 8) | raw[3]
            locus = loci.get(ident)
            if locus is None:
                raise Refused("bad_request", f"genome locus c{i} r{j}")
            if kind_code != locus.kind:
                raise Refused("bad_request", f"genome locus kind c{i} r{j}")
            if ident >> 8 != pair:
                raise Refused("bad_request", f"genome locus pair c{i} r{j}")
            flags = raw[1]
            if flags & FLAG_UNUSED or flags != locus.flags:
                raise Refused("bad_request", f"genome flags c{i} r{j}")
            stages = (raw[4] << 8) | raw[5]
            if stages != locus.stages:
                raise Refused("bad_request", f"genome stage c{i} r{j}")
            if ident in seen:
                raise Refused("bad_request", f"genome duplicate c{i} r{j}")
            seen[ident] = True
            body = raw[6:RECORD]
            used = USED[kind_code]
            for b in range(BODY):
                if body[b] & ~used[b] & 0xFF:
                    raise Refused("bad_request", f"genome pad c{i} r{j}")
            fields = KIND_BY_CODE[kind_code][K_FIELDS]
            values = []
            for index, field in enumerate(fields):
                value = read_field(body, field)
                low, high = locus.boxes[index]
                if value < low or value > high:
                    raise _box_refusal(i, j, field[F_NAME])
                marks = field[F_MARKS]
                if marks & P and not _species_legal(value):
                    raise _box_refusal(i, j, field[F_NAME])
                if marks & R and value <= 63 and token[value]:
                    raise _box_refusal(i, j, field[F_NAME])
                values.append(value)
            if kind_code == KIND_CIS:
                run += 1
                if run > lawview.max_promoter:
                    raise Refused("limit", f"genome promoter c{i} r{j}")
            else:
                run = 0
            records.append(({"fields": values, "flags": flags, "kind": kind_code,
                             "locus": ident, "stages": stages}, body))
        chromosomes.append(records)
        presence.append(seen)
    if pos != size:
        raise Refused("bad_request", "genome trailing")
    for ident in lawview.order:
        locus = loci[ident]
        if not locus.essential:
            continue
        for h in range(PLOIDY):
            if ident not in presence[(ident >> 8) * PLOIDY + h]:
                raise Refused("bad_request", f"genome essential l{ident} h{h}")
    return chromosomes


def decode(data, lawview):
    """The genome value of canonical bytes, or a named refusal."""
    return {"chromosomes": [[record for record, _ in chrom] for chrom in _decode(data, lawview)]}


def encode_record(kind_code, flags, locus, stages, values, where="c0 r0"):
    """One record's 16 bytes; every field checked against its type range first."""
    kind = KIND_BY_CODE.get(kind_code)
    if kind is None or not isinstance(values, list) or len(values) != len(kind[K_FIELDS]):
        raise Refused("bad_request", f"genome value {where}")
    for part, limit in ((flags, 0xFF), (locus, 0xFFFF), (stages, 0xFFFF)):
        if not _is_int(part) or not 0 <= part <= limit:
            raise Refused("bad_request", f"genome value {where}")
    body = bytearray(BODY)
    for field, value in zip(kind[K_FIELDS], values):
        low, high = type_range(field)
        if not _is_int(value) or not low <= value <= high:
            raise Refused("bad_request", f"genome box {where} {field[F_NAME]}")
        _write_field(body, field, value)
    return bytes((kind_code, flags, locus >> 8, locus & 0xFF, stages >> 8, stages & 0xFF)) + bytes(body)


def encode(genome):
    """The canonical bytes of a genome value; the inverse of :func:`decode`."""
    if not isinstance(genome, dict) or not isinstance(genome.get("chromosomes"), list):
        raise Refused("bad_request", "genome value")
    chromosomes = genome["chromosomes"]
    if not chromosomes or len(chromosomes) % PLOIDY or len(chromosomes) // PLOIDY > 255:
        raise Refused("bad_request", "genome value")
    out = bytearray(MAGIC)
    out += SCHEMA.to_bytes(2, "big")
    out += bytes((PLOIDY, len(chromosomes) // PLOIDY))
    for i, chrom in enumerate(chromosomes):
        if not isinstance(chrom, list) or len(chrom) > 0xFFFF:
            raise Refused("bad_request", "genome value")
        out += len(chrom).to_bytes(2, "big")
        for j, record in enumerate(chrom):
            if not _keys_are(record, ("fields", "flags", "kind", "locus", "stages")):
                raise Refused("bad_request", f"genome value c{i} r{j}")
            out += encode_record(record["kind"], record["flags"], record["locus"], record["stages"],
                                 record["fields"], f"c{i} r{j}")
    return bytes(out)


def sha256(data):
    return hashlib.sha256(bytes(data)).hexdigest()


# ---------------------------------------------------------------------------
# Founders and corners
# ---------------------------------------------------------------------------

class CountingStream(rng.Stream):
    """A chassis stream that counts every word it draws, rejections included."""

    __slots__ = ("draws",)

    def __init__(self, seed, domain, index):
        super().__init__(seed, domain, index)
        self.draws = 0

    def next_u64(self):
        self.draws += 1
        return super().next_u64()


def _frame(lawview, chromosome_bytes):
    out = bytearray(MAGIC)
    out += SCHEMA.to_bytes(2, "big")
    out += bytes((PLOIDY, lawview.pairs))
    for count, records in chromosome_bytes:
        out += count.to_bytes(2, "big")
        out += records
    return bytes(out)


def found(seed, lawview, alleles_by_locus):
    """A founder genome from a 32-byte seed: ``(bytes, allele indices, work)``.

    One stream per (locus, homolog), indexed by ``locus*2 + h``: the allele
    first, then one draw per windowed field in kind order. Adding, removing
    or reordering loci moves no other locus's draws.
    """
    if not isinstance(seed, (bytes, bytearray)) or len(seed) != rng.SEED_BYTES:
        raise Refused("bad_request", "seed")
    draws = 0
    chosen_all = []
    parts = []
    for pair in range(lawview.pairs):
        for h in range(PLOIDY):
            records = bytearray()
            ids = lawview.chromosomes[pair]
            for ident in ids:
                locus = lawview.loci[ident]
                stream = CountingStream(bytes(seed), DOMAIN_FOUNDER, ident * 2 + h)
                alleles = alleles_by_locus[ident]
                total = 0
                for allele in alleles:
                    total += allele["freq"]
                if total < 1:
                    raise Refused("engine_panic", "genome found")
                pick = stream.below(total)
                chosen = 0
                cumulative = 0
                for index, allele in enumerate(alleles):
                    cumulative += allele["freq"]
                    if cumulative > pick:
                        chosen = index
                        break
                allele = alleles[chosen]
                values = []
                for index, base in enumerate(allele["body"]):
                    win = allele["window"][index]
                    value = base
                    if win > 0:
                        delta = stream.below(2 * win + 1) - win
                        low, high = locus.boxes[index]
                        value = min(max(base + delta, low), high)
                    values.append(value)
                records += encode_record(locus.kind, locus.flags, ident, locus.stages, values)
                draws += stream.draws
                chosen_all.append(chosen)
            parts.append((len(ids), bytes(records)))
    data = _frame(lawview, parts)
    return data, chosen_all, draws + (len(data) + 63) // 64


def corner(k, lawview):
    """A genome at a corner of the box: all lo (0), all hi (1), or keyed (k >= 2)."""
    if not _is_int(k) or not 0 <= k <= CORNER_MAX:
        raise Refused("bad_request", "corner")
    stream = CountingStream(bytes(rng.SEED_BYTES), DOMAIN_CORNER, k) if k >= 2 else None
    parts = []
    for pair in range(lawview.pairs):
        for _h in range(PLOIDY):
            records = bytearray()
            ids = lawview.chromosomes[pair]
            for ident in ids:
                locus = lawview.loci[ident]
                values = []
                for low, high in locus.boxes:
                    if k == 0:
                        values.append(low)
                    elif k == 1:
                        values.append(high)
                    else:
                        values.append(high if stream.next_u64() >> 63 == 1 else low)
                records += encode_record(locus.kind, locus.flags, ident, locus.stages, values)
            parts.append((len(ids), bytes(records)))
    data = _frame(lawview, parts)
    words = stream.draws if stream is not None else 0
    return data, words + (len(data) + 63) // 64


# ---------------------------------------------------------------------------
# Strength, for dominance
# ---------------------------------------------------------------------------

def strength(kind_code, values, body):
    """The key DOM_MAX compares first, in i64 from the decoded values."""
    rule, arg = KIND_BY_CODE[kind_code][K_STRENGTH]
    if rule == "field":
        return values[arg]
    if rule == "abs":
        return abs(values[arg])
    if rule == "sumabs":
        total = 0
        for index in arg:
            total += abs(values[index])
        return total
    return int.from_bytes(hashlib.sha256(bytes(body)).digest()[:4], "big")


def is_genome_hex(text):
    """True when ``text`` is an even-length lowercase hex string."""
    if not isinstance(text, str) or len(text) % 2:
        return False
    for char in text:
        if char not in "0123456789abcdef":
            return False
    return True
