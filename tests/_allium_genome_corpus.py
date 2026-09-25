#!/usr/bin/env python3
"""Genome corpora shared by the companion's genome contracts, and a test-local reader.

Every corpus is drawn at run time from the chassis stream on a fixed key
(``rng.Stream(bytes(32), "test.<suite>", i)``), never from ``random``: the
same inputs on every run and on every machine. A failing contract prints
the corpus index and the input.

The reader here is written from the published layout, independently of the
reference decoder, so that an acceptance by the engines is checked against
something that is not themselves.
"""

import re

# ---------------------------------------------------------------------------
# The published layout, transcribed: kind -> fields (name, offset, width, signed, shift)
# width 16 and 8 are whole bytes; 4, 3 and 1 are packed bits at ``shift``.
# ---------------------------------------------------------------------------

LAYOUT = {
    1: [("out", 0, 8, 0, 0), ("prod", 1, 16, 0, 0), ("deg", 3, 16, 0, 0), ("bias", 5, 16, 1, 0),
        ("rate", 7, 16, 0, 0)],
    2: [("src", 0, 8, 0, 0), ("w", 1, 16, 1, 0), ("K", 3, 16, 0, 0), ("n", 5, 8, 0, 0), ("mode", 6, 8, 0, 0)],
    3: [("s1", 0, 8, 0, 0), ("s2", 1, 8, 0, 0), ("p1", 2, 8, 0, 0), ("p2", 3, 8, 0, 0), ("kcat", 4, 16, 0, 0),
        ("Km", 6, 16, 0, 0), ("yield", 8, 16, 0, 0)],
    4: [("channel", 0, 8, 0, 0), ("species", 1, 8, 0, 0), ("gain", 2, 16, 1, 0), ("threshold", 4, 16, 0, 0)],
    5: [("pred", 0, 8, 0, 0), ("guard", 1, 8, 0, 0), ("threshold", 2, 16, 0, 0), ("succ", 4, 8, 0, 0),
        ("rate", 5, 16, 0, 0), ("angle", 7, 8, 1, 0)],
    6: [("class", 0, 8, 0, 0), ("strength", 1, 16, 0, 0), ("cost", 3, 16, 0, 0)],
    7: [("class", 0, 8, 0, 0), ("allele", 1, 8, 0, 0)],
    8: [("class", 0, 8, 0, 0), ("A", 1, 8, 1, 0), ("Aneg", 2, 8, 1, 0), ("B", 3, 8, 1, 0), ("C", 4, 8, 1, 0),
        ("D", 5, 8, 1, 0), ("LR", 6, 4, 0, 4), ("TAU_X", 6, 4, 0, 0), ("TAU_Y", 7, 4, 0, 4),
        ("TAU_E", 7, 4, 0, 0), ("M_SEL", 8, 3, 0, 5), ("M_SIGN", 8, 1, 0, 4)],
    9: [("channel", 0, 8, 0, 0), ("weight", 1, 16, 1, 0), ("base", 3, 16, 1, 0), ("habituation", 5, 8, 0, 0),
        ("curiosity", 6, 16, 0, 0)],
    10: [("offset", 0, 8, 0, 0)] + [(f"d{i}", 1 + i, 8, 0, 0) for i in range(9)],
    11: [("trait", 0, 8, 0, 0), ("value", 1, 16, 0, 0), ("window", 3, 16, 0, 0)],
    12: [("D_enter", 0, 8, 0, 0), ("v_cold", 1, 8, 0, 0), ("v_dorm", 2, 8, 0, 0), ("VU_req", 3, 16, 0, 0),
         ("stalk_days", 5, 8, 0, 0), ("offsets", 6, 8, 0, 0), ("rest_days", 7, 8, 0, 0), ("a_min", 8, 16, 0, 0)],
    13: [("block", 0, 8, 0, 0), ("shift0", 1, 16, 1, 0), ("shift1", 3, 16, 1, 0), ("shift2", 5, 16, 1, 0),
         ("shift3", 7, 16, 1, 0)],
    14: [("activity", 0, 16, 0, 0), ("target_bias", 2, 8, 0, 0)],
    15: [("target", 0, 8, 0, 0), ("effect", 1, 16, 0, 0)],
    16: [("param", 0, 8, 0, 0), ("value", 1, 16, 1, 0)],
}
KIND_NAMES = {"tf": 1, "cis": 2, "enz": 3, "rec": 4, "morph": 5, "imm": 6, "pig": 7, "plast": 8, "temp": 9,
              "lex": 10, "shape": 11, "vern": 12, "prc": 13, "te": 14, "load": 15, "brain": 16}
READERS = {(2, "src"), (3, "s1"), (3, "s2"), (5, "guard")}
SPECIES_REFS = {(3, "s1"), (3, "s2"), (3, "p1"), (3, "p2")}


def used_mask(kind):
    used = [0] * 10
    for _name, offset, width, _signed, shift in LAYOUT[kind]:
        if width == 16:
            used[offset] = used[offset + 1] = 0xFF
        elif width == 8:
            used[offset] = 0xFF
        else:
            used[offset] |= ((1 << width) - 1) << shift
    return used


def get_field(body, spec):
    _name, offset, width, signed, shift = spec
    if width == 16:
        value = (body[offset] << 8) | body[offset + 1]
    elif width == 8:
        value = body[offset]
    else:
        return (body[offset] >> shift) & ((1 << width) - 1)
    if signed and value >= 1 << (width - 1):
        value -= 1 << width
    return value


def put_field(body, spec, value):
    """Write ``value`` into a mutable body, masked to the field's width (no range check)."""
    _name, offset, width, _signed, shift = spec
    raw = value & ((1 << width) - 1)
    if width == 16:
        body[offset] = raw >> 8
        body[offset + 1] = raw & 0xFF
    elif width == 8:
        body[offset] = raw
    else:
        body[offset] = (body[offset] & ~(((1 << width) - 1) << shift) & 0xFF) | (raw << shift)


class Law:
    """The facts of a law's genome section a reader needs, read from the law value."""

    def __init__(self, law):
        genome = law["genome"]
        self.pairs = genome["pairs"]
        self.max_bytes = genome["max_bytes"]
        self.max_records = genome["max_records"]
        self.max_promoter = genome["max_promoter"]
        kind_boxes = {KIND_NAMES[k["name"]]: {f["name"]: (f["lo"], f["hi"]) for f in k["fields"]}
                      for k in genome["kinds"]}
        self.loci = {}
        for entry in genome["loci"]:
            kind = KIND_NAMES[entry["kind"]]
            boxes = dict(kind_boxes[kind])
            for override in entry["box"]:
                boxes[override["field"]] = (override["lo"], override["hi"])
            self.loci[entry["id"]] = {"kind": kind, "flags": entry["flags"], "stages": entry["stages"],
                                      "boxes": boxes, "name": entry["name"]}
        self.template = [list(ids) for ids in genome["chromosomes"]]
        self.token = [s["class"] == "token" for s in genome["species"]]


def read(data, law):
    """Chromosomes of ``(kind, flags, locus, stages, body)``, or a string naming the first defect found.

    Not the reference's check order: this reader only answers whether a
    genome is sound, so it can re-verify an acceptance.
    """
    if len(data) > law.max_bytes or len(data) < 8 or data[:4] != b"ALG\x01":
        return "header"
    if data[4:6] != b"\x00\x01" or data[6] != 2 or data[7] != law.pairs:
        return "header"
    pos, chromosomes = 8, []
    for index in range(2 * law.pairs):
        if len(data) - pos < 2:
            return "truncated"
        count = int.from_bytes(data[pos:pos + 2], "big")
        pos += 2
        if count > law.max_records or len(data) - pos < 16 * count:
            return "records"
        records, seen, run = [], [], 0
        for _ in range(count):
            rec = data[pos:pos + 16]
            pos += 16
            kind, flags, locus, stages, body = rec[0], rec[1], int.from_bytes(rec[2:4], "big"), \
                int.from_bytes(rec[4:6], "big"), rec[6:]
            entry = law.loci.get(locus)
            if entry is None or entry["kind"] != kind or locus >> 8 != index // 2:
                return "locus"
            if flags != entry["flags"] or stages != entry["stages"] or locus in seen:
                return "flags, stages or duplicate"
            seen.append(locus)
            if any(body[b] & ~used_mask(kind)[b] & 0xFF for b in range(10)):
                return "pad"
            for spec in LAYOUT[kind]:
                value = get_field(body, spec)
                low, high = entry["boxes"][spec[0]]
                if not low <= value <= high:
                    return f"box {spec[0]}"
                if (kind, spec[0]) in SPECIES_REFS and not (value <= 63 or value == 255):
                    return f"species {spec[0]}"
                if (kind, spec[0]) in READERS and value <= 63 and law.token[value]:
                    return f"token {spec[0]}"
            run = run + 1 if kind == 2 else 0
            if run > law.max_promoter:
                return "promoter"
            records.append((kind, flags, locus, stages, bytes(body)))
        chromosomes.append(records)
    if pos != len(data):
        return "trailing"
    for locus, entry in law.loci.items():
        if entry["flags"] & 4:
            for h in (0, 1):
                if locus not in [r[2] for r in chromosomes[2 * (locus >> 8) + h]]:
                    return "essential"
    return chromosomes


def frame(chromosomes, pairs):
    """Bytes of chromosomes given as lists of 16-byte records."""
    out = bytearray(b"ALG\x01\x00\x01\x02") + bytes([pairs])
    for records in chromosomes:
        out += len(records).to_bytes(2, "big")
        for record in records:
            out += record
    return bytes(out)


def split(data):
    """The header's pair count and the chromosomes of a well-framed genome as record lists."""
    pairs = data[7]
    pos, chromosomes = 8, []
    for _ in range(2 * pairs):
        count = int.from_bytes(data[pos:pos + 2], "big")
        pos += 2
        chromosomes.append([bytes(data[pos + 16 * i:pos + 16 * i + 16]) for i in range(count)])
        pos += 16 * count
    return pairs, chromosomes


def with_field(record, name, value):
    kind = record[0]
    spec = next(s for s in LAYOUT[kind] if s[0] == name)
    body = bytearray(record[6:])
    put_field(body, spec, value)
    return record[:6] + bytes(body)


def field_of(record, name):
    kind = record[0]
    spec = next(s for s in LAYOUT[kind] if s[0] == name)
    return get_field(record[6:], spec)


def locus_of(record):
    return int.from_bytes(record[2:4], "big")


# ---------------------------------------------------------------------------
# Detail grammar
# ---------------------------------------------------------------------------

_N = r"(0|[1-9][0-9]*)"
DETAIL_CLASSES = {
    "size": r"genome size",
    "header": r"genome header",
    "magic": r"genome magic",
    "schema": r"genome schema",
    "ploidy": r"genome ploidy",
    "pairs": r"genome pairs",
    "truncated": r"genome truncated",
    "records": rf"genome records c{_N}",
    "kind": rf"genome kind c{_N} r{_N}",
    "locus": rf"genome locus c{_N} r{_N}",
    "locus kind": rf"genome locus kind c{_N} r{_N}",
    "locus pair": rf"genome locus pair c{_N} r{_N}",
    "flags": rf"genome flags c{_N} r{_N}",
    "stage": rf"genome stage c{_N} r{_N}",
    "duplicate": rf"genome duplicate c{_N} r{_N}",
    "pad": rf"genome pad c{_N} r{_N}",
    "box": rf"genome box c{_N} r{_N} [A-Za-z_][A-Za-z0-9_]*",
    "promoter": rf"genome promoter c{_N} r{_N}",
    "trailing": r"genome trailing",
    "essential": rf"genome essential l{_N} h[01]",
}
REQUEST_CLASSES = {
    "fields": r"fields",
    "law": r"law",
    "genome hex": r"genome hex",
    "seed": r"seed",
    "corner": r"corner",
}
CODES = ("bad_request", "limit", "unknown_law")


def detail_class(detail):
    for name, pattern in DETAIL_CLASSES.items():
        if re.fullmatch(pattern, detail):
            return name
    for name, pattern in REQUEST_CLASSES.items():
        if re.fullmatch(pattern, detail):
            return "request " + name
    return None


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

def seeds(rng, domain, count):
    """``count`` 32-byte seeds, four words each from the stream of index ``i``."""
    out = []
    for i in range(count):
        stream = rng.Stream(bytes(32), domain, i)
        out.append(b"".join(stream.next_u64().to_bytes(8, "big") for _ in range(4)))
    return out


def targeted(founder, law):
    """One genome per detail class the fixture can reach, from a sound founder; ``[(class, bytes)]``."""
    pairs, chrom = split(founder)
    out = []

    def rebuild(chromosomes):
        return frame(chromosomes, pairs)

    out.append(("size", bytes(law.max_bytes + 1)))
    out.append(("header", founder[:5]))
    out.append(("magic", b"ALH" + founder[3:]))
    out.append(("schema", founder[:4] + b"\x00\x02" + founder[6:]))
    out.append(("ploidy", founder[:6] + b"\x01" + founder[7:]))
    out.append(("pairs", founder[:7] + bytes([pairs + 1]) + founder[8:]))
    out.append(("truncated", founder[:8 + 2 + 16 * 3 + 5]))
    c0 = list(chrom[0])
    out.append(("kind", rebuild([[bytes([17]) + c0[0][1:]] + c0[1:]] + chrom[1:])))
    bad = bytearray(c0[0])
    bad[2:4] = (0x7FFF).to_bytes(2, "big")
    out.append(("locus", rebuild([[bytes(bad)] + c0[1:]] + chrom[1:])))
    out.append(("locus kind", rebuild([[c0[0], bytes([1]) + c0[1][1:]] + c0[2:]] + chrom[1:])))
    out.append(("flags", rebuild([[c0[0][:1] + bytes([c0[0][1] | 0x20]) + c0[0][2:]] + c0[1:]] + chrom[1:])))
    out.append(("stage", rebuild([[c0[0][:4] + b"\x00\x01" + c0[0][6:]] + c0[1:]] + chrom[1:])))
    # A founder is exactly max_bytes long: a duplicate takes a non-essential record's place.
    spare = [k for k, rec in enumerate(c0) if not law.loci[locus_of(rec)]["flags"] & 4][-1]
    out.append(("duplicate", rebuild([c0[:spare] + [c0[0]] + c0[spare + 1:]] + chrom[1:])))
    pad = bytearray(c0[0])
    pad[15] |= 0x01  # the REC body's last byte is a pad
    out.append(("pad", rebuild([[bytes(pad)] + c0[1:]] + chrom[1:])))
    out.append(("box", rebuild([[with_field(c0[0], "gain", 32767)] + c0[1:]] + chrom[1:])))
    c1_short = [rec for rec in chrom[1] if rec != chrom[1][spare]]
    out.append(("trailing", rebuild([c0, c1_short] + chrom[2:]) + b"\x00"))
    essential = [r for r in chrom[1] if law.loci[locus_of(r)]["flags"] & 4][0]
    out.append(("essential", rebuild([chrom[0], [r for r in chrom[1] if r != essential]] + chrom[2:])))
    token = [r for r in c0 if r[0] == 2][0]
    index = c0.index(token)
    out.append(("box", rebuild([c0[:index] + [with_field(token, "src", 33)] + c0[index + 1:]] + chrom[1:])))
    return out


def v0_1_cases(corner_hi, law):
    """The detail classes only the full law reaches, and the widest promoter union."""
    pairs, chrom = split(corner_hi)
    out = []
    by_pair = {p: chrom[2 * p] for p in range(pairs)}
    cis = {p: [r for r in by_pair[p] if r[0] == 2] for p in range(pairs)}
    # A run of nine CIS records on pair 2's homolog 0.
    p2 = by_pair[2]
    nine = cis[2][:9]
    rest = [r for r in p2 if r not in nine]
    out.append(("promoter", frame(chrom[:4] + [nine + rest] + chrom[5:], pairs)))
    # A record count of 65 on chromosome 4: the count is read before the records, so the
    # genome keeps its size and the cap on records answers before the cap on bytes.
    data = bytearray(corner_hi)
    pos = 8
    for index in range(4):
        pos += 2 + 16 * len(chrom[index])
    data[pos:pos + 2] = (65).to_bytes(2, "big")
    out.append(("records", bytes(data)))
    # A pair-1 locus in place of a non-essential record of pair 0's chromosome.
    stray = by_pair[1][0]
    c0 = list(chrom[0])
    spare = [k for k, rec in enumerate(c0) if not law.loci[locus_of(rec)]["flags"] & 4][0]
    c0[spare] = stray
    out.append(("locus pair", frame([c0] + chrom[1:], pairs)))
    # Disjoint promoters: eight CIS before aba_synth on homolog 0, the other six on homolog 1.
    aba = [r for r in p2 if locus_of(r) == 0x20E][0]
    others = [r for r in p2 if r[0] != 2 and r != aba]
    first, second = cis[2][:8], cis[2][8:]
    h0 = second + others + first + [aba]
    h1 = first + others + second + [aba]
    out.append(("union", frame(chrom[:4] + [h0, h1] + chrom[6:], pairs)))
    return out


def mutations(rng, founders, law, count, domain):
    """``count`` seeded mutations of sound founders: ``[(index, bytes)]``."""
    out = []
    for i in range(count):
        s = rng.Stream(bytes(32), domain, i)
        base = founders[s.below(len(founders))]
        pairs, chrom = split(base)
        op = s.below(13)
        c = s.below(len(chrom))
        records = list(chrom[c])
        r = s.below(len(records))
        if op == 0:
            data = bytearray(base)
            data[s.below(len(data))] ^= 1 + s.below(255)
            out.append((i, bytes(data)))
            continue
        if op == 1:
            record = records[r]
            spec = LAYOUT[record[0]][s.below(len(LAYOUT[record[0]]))]
            low, high = law.loci[locus_of(record)]["boxes"][spec[0]]
            value = (low - 1, high + 1, low, high)[s.below(4)]
            records[r] = with_field(record, spec[0], value)
        elif op == 2:
            out.append((i, base[:s.below(len(base))]))
            continue
        elif op == 3:
            out.append((i, base + bytes(s.below(256) for _ in range(1 + s.below(20)))))
            continue
        elif op == 4:
            del records[r]
        elif op == 5:
            records.insert(s.below(len(records) + 1), records[r])
        elif op == 6:
            q = s.below(len(records))
            records[r], records[q] = records[q], records[r]
        elif op == 7:
            other = s.below(len(chrom))
            moved = records.pop(r)
            if other == c:
                records.insert(s.below(len(records) + 1), moved)
            else:
                chrom[other] = list(chrom[other]) + [moved]
        elif op == 8:
            data = bytearray(base)
            data[8:10] = s.below(80).to_bytes(2, "big")
            out.append((i, bytes(data)))
            continue
        elif op == 9:
            length = s.below(len(base) + 40)
            head = b"ALG\x01\x00\x01\x02\x01" if s.below(2) else b""
            out.append((i, head + bytes(s.below(256) for _ in range(length))))
            continue
        elif op == 10:
            data = bytearray(base)
            data[s.below(8)] = s.below(256)
            out.append((i, bytes(data)))
            continue
        elif op == 11:
            record = bytearray(records[r])
            used = used_mask(record[0])
            free = [(b, bit) for b in range(10) for bit in range(8) if not used[b] >> bit & 1]
            if free:
                b, bit = free[s.below(len(free))]
                record[6 + b] |= 1 << bit
            records[r] = bytes(record)
        else:
            cis_rows = [k for k, rec in enumerate(records) if rec[0] == 2]
            if cis_rows:
                k = cis_rows[s.below(len(cis_rows))]
                records[k] = with_field(records[k], "src", (1, 33, 34, 35)[s.below(4)])
        chrom[c] = records
        out.append((i, frame(chrom, pairs)))
    return out


def requests(genome_hex, max_bytes):
    """The request corpus: ``[(expected class or None, request)]``."""
    base = {"law": "fixture", "op": "genome_decode", "v": 1}
    return [
        ("request genome hex", dict(base, genome=genome_hex.upper())),
        ("request genome hex", dict(base, genome=genome_hex[:-1])),
        ("request genome hex", dict(base, genome=genome_hex[:-2] + "zz")),
        ("request genome hex", dict(base, genome=12)),
        ("size", dict(base, genome="00" * (max_bytes + 1))),
        ("request genome hex", dict(base, op="genome_compile", genome=None)),
        ("request corner", {"corner": True, "law": "fixture", "op": "genome_corner", "v": 1}),
        ("request corner", {"corner": -1, "law": "fixture", "op": "genome_corner", "v": 1}),
        ("request corner", {"corner": 1 << 32, "law": "fixture", "op": "genome_corner", "v": 1}),
        ("request seed", {"law": "fixture", "op": "genome_found", "seed": "AB" * 32, "v": 1}),
        ("request seed", {"law": "fixture", "op": "genome_found", "seed": "a" * 63, "v": 1}),
        ("request law", {"law": "v1", "op": "genome_found", "seed": "ab" * 32, "v": 1}),
        ("request law", {"law": 3, "op": "genome_corner", "corner": 0, "v": 1}),
        ("request fields", dict(base, genome=genome_hex, extra=1)),
        ("request fields", {"law": "fixture", "op": "genome_decode", "v": 1}),
        ("request fields", {"corner": 0, "law": "fixture", "op": "genome_found", "v": 1}),
        (None, {"corner": 4294967295, "law": "fixture", "op": "genome_corner", "v": 1}),
    ]
