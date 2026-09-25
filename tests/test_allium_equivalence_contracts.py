#!/usr/bin/env python3
"""Contracts for the componion's genome in both engines: the same bytes, request for request.

  * AE1 -- for the four genome operations, the native engine and the
    reference answer with the same bytes: founders from seeded seeds, box
    corners, single-homolog deletions, dominance cases built on the embedded
    laws, and the malformed-request corpus; each case the corpus exists to
    reach is shown to have been reached.
  * AE2 -- the same, over the whole decoder corpus of the reference suite:
    the targeted defects of every refusal class, the thousand seeded
    mutations and the cases only the full law reaches.

In both, every native answer also parses under the strict canonical parser,
and a counter around the native entry point proves the native engine
answered every request. The two share one corpus, split so each stays within
its time budget.

  * AE12 -- lives in both engines: twelve seeded lives and four lives whose
    genome constants come from the corners of the law's box, advanced
    through cuts and budget slices, answer with the same bytes, trace and
    environment included. At every returned state no fixed-point value was
    corrected, the reserve stays within its bounds and its ledger balances,
    and every trace accounts, by the law's unit table, for the work counted.
    Awake and dormant beings, a wake, ``tz`` facts, a params ``evolve``, a
    pin, a redacted fact, a noted fact, a call stopped by its budget, the
    trace, another organ order, the stepped path, a being awake before its
    year's equinox, and every note but ``laws_stale`` are each met.
  * AE13 -- requests in both engines: crafted defective ``advance``,
    ``grid`` and ``timeline`` requests reach every refusal a request can
    reach with the carried laws, some with several defects, which name the
    first in check order; ``timeline`` also folds seeded sequences of the
    law kinds, ``tz`` and ``evolve`` in one minute in both orders. Every
    answer is the same bytes.
  * AE14 -- what the seeded lives never ask, in both engines, byte for
    byte: ``advance`` over a ``tz`` fact and an ``evolve`` in one minute, in
    both canonical orders (the same pending change, checked against the
    offset at the end of the minute's ``tz`` facts); an ``evolve`` whose
    minute a later ``tz`` fact moved off its midnight, reached before, at
    and after it, awake and asleep, on the fast path and off it; a day with
    one fact past the per-day cap of each budget-exempt kind, each excess
    one noted ``budget`` and the day within the awake ceiling; states whose
    pending change names a law this engine cannot approach -- another law,
    an unknown name, another digest, another version -- refused by name in
    ``advance`` and ``timeline`` when it comes due and not a minute before;
    and a state whose versions seen hold one besides the law in force, where
    a fact carrying it is noted ``laws_stale`` and folded, or, over its
    day's budget, noted for the budget alone.
  * AE15 -- AE1, word for word but one assertion: the dominance cases'
    enzyme columns are equal whichever homolog carries the maximum, locus
    7's row is the maximum in both, and the column holds a row per enzyme
    locus of the fixture law, more than one.
  * AE16 -- AE2, word for word but its floor: the corpus compares at least
    the 1281 requests it reaches under the fixture's thirty-two loci.

Local-only. It needs the native artefact that ``scripts/build_oo_core.sh``
builds.
"""

import copy
import hashlib
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_genome_corpus as corpus  # noqa: E402
import _allium_life_support as support  # noqa: E402
from _allium_window import native_module, open_allium  # noqa: E402
from _isolation import REPO  # noqa: E402

GOLDEN = REPO / "tests" / "allium_golden" / "v1" / "genome.json"
OPS = ("genome_compile", "genome_corner", "genome_decode", "genome_found")

BUDGET_S = {
    "test_ae1_both_engines_answer_every_genome_request_with_the_same_bytes": 2.0,
    "test_ae2_both_engines_answer_the_whole_decoder_corpus_with_the_same_bytes": 2.0,
    "test_ae12_both_engines_live_the_same_lives_byte_for_byte": 2.0,
    "test_ae13_both_engines_refuse_every_defective_life_request_with_the_same_bytes": 2.0,
    "test_ae14_both_engines_agree_on_same_minute_evolves_exempt_floods_and_crafted_states": 2.0,
    "test_ae15_both_engines_answer_every_genome_request_with_the_same_bytes_for_each_enzyme_locus": 2.0,
    "test_ae16_both_engines_answer_the_whole_decoder_corpus_with_the_same_bytes_above_a_remeasured_floor": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


class _Counting:
    """Counts the native engine's answers per operation."""

    def __init__(self, native, wire):
        self.native = native
        self.wire = wire
        self.calls = {}

    def call(self, data, op):
        self.calls[op] = self.calls.get(op, 0) + 1
        return bytes(self.native.allium_call(data))


def _edit(data, chromosome, locus, **fields):
    pairs, chrom = corpus.split(data)
    records = list(chrom[chromosome])
    for index, record in enumerate(records):
        if corpus.locus_of(record) == locus:
            for name, value in fields.items():
                record = corpus.with_field(record, name, value)
            records[index] = record
    chrom[chromosome] = records
    return corpus.frame(chrom, pairs)


def _drop(data, chromosome, locus):
    pairs, chrom = corpus.split(data)
    chrom[chromosome] = [r for r in chrom[chromosome] if corpus.locus_of(r) != locus]
    return corpus.frame(chrom, pairs)


def _swap(data):
    pairs, chrom = corpus.split(data)
    out = []
    for p in range(pairs):
        out += [chrom[2 * p + 1], chrom[2 * p]]
    return corpus.frame(out, pairs)


def _cases(fixture_founder, v0_1_founder, v0_1_corner_hi, reader_v0_1):
    """Genomes built to reach one situation each: ``[(label, law, bytes)]``."""
    f = fixture_founder
    cases = [
        ("dom_max won by h0", "fixture", _edit(_edit(f, 0, 7, kcat=20000), 1, 7, kcat=16384)),
        ("dom_max won by h1", "fixture", _edit(_edit(f, 0, 7, kcat=16384), 1, 7, kcat=20000)),
        ("dom_max tie by bytes", "fixture", _edit(_edit(f, 0, 1, n=3), 1, 1, n=4)),
        ("lex by hash", "fixture", _edit(_edit(f, 0, 23, d0=1), 1, 23, d0=2)),
        ("add differing", "fixture", _edit(_edit(f, 0, 2, deg=3700), 1, 2, deg=4600)),
        ("add odd negative sum", "fixture", _edit(_edit(f, 0, 11, A=-3), 1, 11, A=0)),
        ("load row", "fixture", _edit(_edit(f, 0, 18, effect=16384), 1, 18, effect=16385)),
        ("load heterozygous", "fixture", _edit(_edit(f, 0, 18, effect=0), 1, 18, effect=16384)),
        ("load dose 1", "fixture", _drop(_edit(f, 0, 18, effect=16384), 1, 18)),
        ("dom_max signed minimum", "v0_1", _edit(_edit(v0_1_founder, 2, 0x100, w=-32768), 3, 0x100, w=32767)),
    ]
    union = dict(corpus.v0_1_cases(v0_1_corner_hi, reader_v0_1))["union"]
    cases.append(("union of disjoint promoters", "v0_1", union))
    cases.append(("swapped", "fixture", _swap(f)))
    return cases


class _Engines:
    """The reference and the native engine of one window, asked the same requests."""

    def __init__(self, loaded):
        self.wire = loaded["opti_oignon.allium.wire"]
        self.rng = loaded["opti_oignon.allium.rng"]
        self.lawfiles = loaded["opti_oignon.allium.lawfiles"]
        self.genome = loaded["opti_oignon.allium.ref.organs.genome"]
        self.protocol = loaded["opti_oignon.allium.ref.protocol"]
        self.native = _Counting(native_module(loaded), self.wire)
        self.laws = {name: self.lawfiles.law(name) for name in ("fixture", "v0_1")}
        self.views = {name: self.genome.view(law) for name, law in self.laws.items()}
        self.readers = {name: corpus.Law(law) for name, law in self.laws.items()}
        self.compared = 0
        self.answers = []

    def both(self, request):
        data = self.wire.emit(dict(request, v=1))
        mine = self.protocol.call(data)
        theirs = self.native.call(data, request.get("op"))
        assert theirs == mine, f"request {self.compared}: {data[:200]!r}\n{mine[:300]!r}\n{theirs[:300]!r}"
        # The bytes are equal, so the one strict parse of the native answer reads both.
        answer = self.wire.parse(theirs)
        self.compared += 1
        self.answers.append((request, answer))
        return answer

    def pair(self, law, data):
        self.both({"genome": data.hex(), "law": law, "op": "genome_decode"})
        return self.both({"genome": data.hex(), "law": law, "op": "genome_compile"})

    def column(self, tables, *path):
        node = tables
        for key in path:
            node = node[key]
        return self.wire.unpack_bulk(node)[1]


def test_ae1_both_engines_answer_every_genome_request_with_the_same_bytes():
    loaded, restore = open_allium(native=True)
    try:
        e = _Engines(loaded)
        founders = {}
        for law, count in (("fixture", 64), ("v0_1", 8)):
            founders[law] = []
            for seed in corpus.seeds(e.rng, "test.ae1", count):
                answer = e.both({"law": law, "op": "genome_found", "seed": seed.hex()})
                founders[law].append(bytes.fromhex(answer["genome"]))
                e.pair(law, founders[law][-1])
        corners = {}
        for law, count in (("fixture", 32), ("v0_1", 4)):
            corners[law] = []
            for k in range(count):
                answer = e.both({"corner": k, "law": law, "op": "genome_corner"})
                corners[law].append(bytes.fromhex(answer["genome"]))
                e.pair(law, corners[law][-1])
        base = founders["fixture"][0]
        for locus, entry in e.readers["fixture"].loci.items():
            if not entry["flags"] & 4:
                e.pair("fixture", _drop(base, 1, locus))
        reached = {}
        built = {}
        for label, law, data in _cases(base, founders["v0_1"][0], corners["v0_1"][1], e.readers["v0_1"]):
            answer = e.pair(law, data)
            assert "refused" not in answer, (label, answer)
            reached[label] = answer["tables"]
            built[label] = data
        refusals = {}
        for expected, request in corpus.requests(base.hex(), e.readers["fixture"].max_bytes):
            answer = e.both(request)
            if expected is not None:
                klass = corpus.detail_class(answer["detail"])
                refusals[klass] = refusals.get(klass, 0) + 1
    finally:
        restore()

    for op in OPS:
        assert e.native.calls.get(op, 0) >= 30, (op, e.native.calls)
    assert sum(e.native.calls.values()) == e.compared
    column = e.column

    def row_of(tables, table, locus):
        genes = column(tables, "genes", "locus")
        return column(tables, table, "gene").index(genes.index(locus))

    t = reached
    assert column(t["dom_max won by h0"], "enz", "kcat") == column(t["dom_max won by h1"], "enz", "kcat") == [20000]
    starts = column(t["dom_max tie by bytes"], "genes", "edge_start")
    assert column(t["dom_max tie by bytes"], "edges", "n")[0] == 4 and starts[2] - starts[1] == 1
    assert column(t["add differing"], "tf", "deg")[row_of(t["add differing"], "tf", 2)] == 4150
    assert column(t["add odd negative sum"], "reserved", "plast", "fields")[1] == -2
    assert column(t["load row"], "loads", "effect") == [16384]
    assert column(t["load heterozygous"], "loads", "gene") == []
    assert column(t["load dose 1"], "loads", "gene") == []
    tables = t["dom_max signed minimum"]
    genes = column(tables, "genes", "locus")
    starts = column(tables, "genes", "edge_start")
    row = genes.index(0x102)
    cis = column(tables, "edges", "cis")[starts[row]:starts[row + 1]]
    weights = column(tables, "edges", "w")[starts[row]:starts[row + 1]]
    assert weights[cis.index(0x100)] == -32768 * 16
    union = t["union of disjoint promoters"]
    starts = column(union, "genes", "edge_start")
    assert max(b - a for a, b in zip(starts, starts[1:])) == 14
    _pairs, chrom = corpus.split(built["lex by hash"])
    bodies = [next(r for r in chrom[h] if corpus.locus_of(r) == 23)[6:] for h in (0, 1)]
    assert bodies[0] != bodies[1]
    winner = max(bodies, key=lambda body: (int.from_bytes(hashlib.sha256(body).digest()[:4], "big"), body))
    lex = t["lex by hash"]
    lex_row = column(lex, "reserved", "lex", "gene").index(column(lex, "genes", "locus").index(23))
    assert column(lex, "reserved", "lex", "fields")[10 * lex_row + 1] == winner[1], "the hash key decides"

    compiled = [answer["tables"] for request, answer in e.answers
                if request.get("op") == "genome_compile" and "tables" in answer]
    doses = [d for tables in compiled for d in column(tables, "genes", "dose")]
    assert 1 in doses and 2 in doses, "dose 1 and dose 2 both compiled"
    assert any(column(tables, "loads", "gene") == [] for tables in compiled), "an empty column"
    k_lo = min(k for tables in compiled for k in column(tables, "edges", "k"))
    assert k_lo == 512 * 8, "a Hill constant at the floor of its box"
    kinds = {k for tables in compiled for k in column(tables, "genes", "kind")}
    edges_seen = sum(len(column(tables, "edges", "cis")) for tables in compiled)
    assert kinds == set(range(1, 17)) - {2} and edges_seen > 0, kinds
    for name in ("fields", "law", "genome hex", "seed", "corner"):
        assert refusals.get("request " + name, 0) >= 1, name
    assert e.compared >= 387, e.compared


def test_ae2_both_engines_answer_the_whole_decoder_corpus_with_the_same_bytes():
    loaded, restore = open_allium(native=True)
    try:
        e = _Engines(loaded)
        founders = [bytes.fromhex(entry["genome"]) for entry in
                    json.loads(GOLDEN.read_text(encoding="ascii"))["fixture"]["found"]]
        alleles = e.genome.pool_alleles(e.lawfiles.founders("fixture"))
        founders += [e.genome.found(seed, e.views["fixture"], alleles)[0]
                     for seed in corpus.seeds(e.rng, "test.gn3", 8)]
        corner_hi = e.genome.corner(1, e.views["v0_1"])[0]
        items = [("fixture", data, True) for _label, data in corpus.targeted(founders[0], e.readers["fixture"])]
        items += [("fixture", data, False) for _i, data in
                  corpus.mutations(e.rng, founders, e.readers["fixture"], 1000, "test.gn3")]
        items += [("v0_1", data, True) for _label, data in corpus.v0_1_cases(corner_hi, e.readers["v0_1"])]
        items.append(("v0_1", bytes(e.readers["v0_1"].max_bytes + 1), True))
        refusals = {}
        accepted = 0
        for law, data, targeted in items:
            answer = e.both({"genome": data.hex(), "law": law, "op": "genome_decode"})
            # Compile decodes first, so a random mutation decode refuses is refused by compile
            # with the same bytes; the targeted cases carry every refusal class through compile.
            if targeted or "refused" not in answer:
                answer = e.both({"genome": data.hex(), "law": law, "op": "genome_compile"})
            if "refused" in answer:
                klass = corpus.detail_class(answer["detail"])
                refusals[klass] = refusals.get(klass, 0) + 1
            else:
                accepted += 1
    finally:
        restore()

    assert sum(e.native.calls.values()) == e.compared
    assert e.native.calls.get("genome_decode", 0) >= 1000 and e.native.calls.get("genome_compile", 0) >= 100
    missing = [name for name in corpus.DETAIL_CLASSES if name not in refusals]
    assert missing == [], f"refusal classes never compared: {missing}"
    assert accepted >= 100, accepted
    assert e.compared >= 1284, e.compared


# ---------------------------------------------------------------------------
# AE15, AE16 -- AE1 and AE2 over the fixture as it is now: its enzyme loci and its corpus
# ---------------------------------------------------------------------------
def test_ae15_both_engines_answer_every_genome_request_with_the_same_bytes_for_each_enzyme_locus():
    loaded, restore = open_allium(native=True)
    try:
        e = _Engines(loaded)
        founders = {}
        for law, count in (("fixture", 64), ("v0_1", 8)):
            founders[law] = []
            for seed in corpus.seeds(e.rng, "test.ae1", count):
                answer = e.both({"law": law, "op": "genome_found", "seed": seed.hex()})
                founders[law].append(bytes.fromhex(answer["genome"]))
                e.pair(law, founders[law][-1])
        corners = {}
        for law, count in (("fixture", 32), ("v0_1", 4)):
            corners[law] = []
            for k in range(count):
                answer = e.both({"corner": k, "law": law, "op": "genome_corner"})
                corners[law].append(bytes.fromhex(answer["genome"]))
                e.pair(law, corners[law][-1])
        base = founders["fixture"][0]
        for locus, entry in e.readers["fixture"].loci.items():
            if not entry["flags"] & 4:
                e.pair("fixture", _drop(base, 1, locus))
        reached = {}
        built = {}
        for label, law, data in _cases(base, founders["v0_1"][0], corners["v0_1"][1], e.readers["v0_1"]):
            answer = e.pair(law, data)
            assert "refused" not in answer, (label, answer)
            reached[label] = answer["tables"]
            built[label] = data
        refusals = {}
        for expected, request in corpus.requests(base.hex(), e.readers["fixture"].max_bytes):
            answer = e.both(request)
            if expected is not None:
                klass = corpus.detail_class(answer["detail"])
                refusals[klass] = refusals.get(klass, 0) + 1
    finally:
        restore()

    for op in OPS:
        assert e.native.calls.get(op, 0) >= 30, (op, e.native.calls)
    assert sum(e.native.calls.values()) == e.compared
    column = e.column

    def row_of(tables, table, locus):
        genes = column(tables, "genes", "locus")
        return column(tables, table, "gene").index(genes.index(locus))

    t = reached
    h0, h1 = t["dom_max won by h0"], t["dom_max won by h1"]
    enzymes = sum(1 for entry in e.laws["fixture"]["genome"]["loci"] if entry["kind"] == "enz")
    assert column(h0, "enz", "kcat") == column(h1, "enz", "kcat"), "the homolog that carries the maximum does not matter"
    assert column(h0, "enz", "kcat")[row_of(h0, "enz", 7)] == column(h1, "enz", "kcat")[row_of(h1, "enz", 7)] == 20000
    assert len(column(h0, "enz", "kcat")) == enzymes > 1, ("a row per enzyme locus of the law", enzymes)
    starts = column(t["dom_max tie by bytes"], "genes", "edge_start")
    assert column(t["dom_max tie by bytes"], "edges", "n")[0] == 4 and starts[2] - starts[1] == 1
    assert column(t["add differing"], "tf", "deg")[row_of(t["add differing"], "tf", 2)] == 4150
    assert column(t["add odd negative sum"], "reserved", "plast", "fields")[1] == -2
    assert column(t["load row"], "loads", "effect") == [16384]
    assert column(t["load heterozygous"], "loads", "gene") == []
    assert column(t["load dose 1"], "loads", "gene") == []
    tables = t["dom_max signed minimum"]
    genes = column(tables, "genes", "locus")
    starts = column(tables, "genes", "edge_start")
    row = genes.index(0x102)
    cis = column(tables, "edges", "cis")[starts[row]:starts[row + 1]]
    weights = column(tables, "edges", "w")[starts[row]:starts[row + 1]]
    assert weights[cis.index(0x100)] == -32768 * 16
    union = t["union of disjoint promoters"]
    starts = column(union, "genes", "edge_start")
    assert max(b - a for a, b in zip(starts, starts[1:])) == 14
    _pairs, chrom = corpus.split(built["lex by hash"])
    bodies = [next(r for r in chrom[h] if corpus.locus_of(r) == 23)[6:] for h in (0, 1)]
    assert bodies[0] != bodies[1]
    winner = max(bodies, key=lambda body: (int.from_bytes(hashlib.sha256(body).digest()[:4], "big"), body))
    lex = t["lex by hash"]
    lex_row = column(lex, "reserved", "lex", "gene").index(column(lex, "genes", "locus").index(23))
    assert column(lex, "reserved", "lex", "fields")[10 * lex_row + 1] == winner[1], "the hash key decides"

    compiled = [answer["tables"] for request, answer in e.answers
                if request.get("op") == "genome_compile" and "tables" in answer]
    doses = [d for tables in compiled for d in column(tables, "genes", "dose")]
    assert 1 in doses and 2 in doses, "dose 1 and dose 2 both compiled"
    assert any(column(tables, "loads", "gene") == [] for tables in compiled), "an empty column"
    k_lo = min(k for tables in compiled for k in column(tables, "edges", "k"))
    assert k_lo == 512 * 8, "a Hill constant at the floor of its box"
    kinds = {k for tables in compiled for k in column(tables, "genes", "kind")}
    edges_seen = sum(len(column(tables, "edges", "cis")) for tables in compiled)
    assert kinds == set(range(1, 17)) - {2} and edges_seen > 0, kinds
    for name in ("fields", "law", "genome hex", "seed", "corner"):
        assert refusals.get("request " + name, 0) >= 1, name
    assert e.compared >= 387, e.compared


def test_ae16_both_engines_answer_the_whole_decoder_corpus_with_the_same_bytes_above_a_remeasured_floor():
    loaded, restore = open_allium(native=True)
    try:
        e = _Engines(loaded)
        founders = [bytes.fromhex(entry["genome"]) for entry in
                    json.loads(GOLDEN.read_text(encoding="ascii"))["fixture"]["found"]]
        alleles = e.genome.pool_alleles(e.lawfiles.founders("fixture"))
        founders += [e.genome.found(seed, e.views["fixture"], alleles)[0]
                     for seed in corpus.seeds(e.rng, "test.gn3", 8)]
        corner_hi = e.genome.corner(1, e.views["v0_1"])[0]
        items = [("fixture", data, True) for _label, data in corpus.targeted(founders[0], e.readers["fixture"])]
        items += [("fixture", data, False) for _i, data in
                  corpus.mutations(e.rng, founders, e.readers["fixture"], 1000, "test.gn3")]
        items += [("v0_1", data, True) for _label, data in corpus.v0_1_cases(corner_hi, e.readers["v0_1"])]
        items.append(("v0_1", bytes(e.readers["v0_1"].max_bytes + 1), True))
        refusals = {}
        accepted = 0
        for law, data, targeted in items:
            answer = e.both({"genome": data.hex(), "law": law, "op": "genome_decode"})
            # Compile decodes first, so a random mutation decode refuses is refused by compile
            # with the same bytes; the targeted cases carry every refusal class through compile.
            if targeted or "refused" not in answer:
                answer = e.both({"genome": data.hex(), "law": law, "op": "genome_compile"})
            if "refused" in answer:
                klass = corpus.detail_class(answer["detail"])
                refusals[klass] = refusals.get(klass, 0) + 1
            else:
                accepted += 1
    finally:
        restore()

    assert sum(e.native.calls.values()) == e.compared
    assert e.native.calls.get("genome_decode", 0) >= 1000 and e.native.calls.get("genome_compile", 0) >= 100
    missing = [name for name in corpus.DETAIL_CLASSES if name not in refusals]
    assert missing == [], f"refusal classes never compared: {missing}"
    assert accepted >= 100, accepted
    assert e.compared >= 1281, e.compared


# ---------------------------------------------------------------------------
# AE12, AE13 -- lives, and requests about lives, in both engines
# ---------------------------------------------------------------------------
DAY = 1440
LOW = "0000000000000a01"
HIGH = "fffffffffffff001"
NOTE_CODES = ("budget", "evolve_from", "evolve_params", "evolve_pinned", "evolve_superseded", "evolve_when",
              "pin_twice", "truncated", "unpin_unpinned")
OFFSETS = (-600, 0, 345, 840)
_DROP = object()


class _Lives:
    """The reference and the native engine of one window, asked the same requests about lives."""

    def __init__(self, loaded):
        self.ref = support.Engine(loaded)
        self.wire = self.ref.wire
        self.native = _Counting(native_module(loaded), self.wire)
        self.compared = 0
        self.constants = {}
        self.laws = {}

    def both(self, request):
        data = self.wire.emit(request)
        mine = self.ref.protocol.call(data)
        theirs = self.native.call(data, request.get("op"))
        assert theirs == mine, f"request {self.compared}: {data[:400]!r}\n{mine[:600]!r}\n{theirs[:600]!r}"
        self.compared += 1
        return self.wire.parse(theirs)

    def chem(self, name):
        if name not in self.constants:
            self.constants[name] = self.ref.lawfiles.law(name)["constants"]["chem"]
        return self.constants[name]

    def law(self, name):
        if name not in self.laws:
            self.laws[name] = self.ref.lawfiles.law(name)
        return self.laws[name]


def _count(seen, key):
    seen[key] = seen.get(key, 0) + 1


def _winter_wall(hemisphere):
    """A birth wall a day or two before the fixture's winter, in either hemisphere: the being sleeps early."""
    return support.WALL + (23 if hemisphere == "north" else 3) * 86400


def _offset(being, t):
    """The offset in force at the end of minute ``t``'s ``tz`` facts."""
    offset = being.tz_birth
    for fact in being.facts:
        if fact["t"] > t:
            break
        if fact["kind"] == "tz":
            offset = fact["body"]["quarters"] * 15
    return offset


def _evolve(being, t, params, **kw):
    """A well-formed ``evolve`` at ``t``, effective at the next local midnight under the offset in force."""
    return being.evolve(t, params, effective_from=being.midnight(t, _offset(being, t)), **kw)


def _shuffled(draw, items):
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = draw.below(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def _gestures(being, draw, end, count):
    for _ in range(count):
        being.act(1 + draw.below(end - 1), ("greet", "play", "touch", "water")[draw.below(4)])


def _ae12_life(ref, index, draw):
    """Life ``index`` of AE12: ``(being, end, mode, stops)``; each life is built to meet what it is named for."""
    hemisphere = ("north", "south")[draw.below(2)]
    band = ("long", "medium", "short")[draw.below(3)]
    tz = OFFSETS[draw.below(4)]
    law = "v0_1" if index in (10, 11) else "fixture"
    weather = "windowsill" if index in (1, 8, 11) else "garden"
    wall = _winter_wall(hemisphere) + draw.below(3600)
    params = None
    if index == 1:
        wall = support.WALL + draw.below(3600)
        params = {name: spec["default"] for name, spec in ref.lawfiles.law("fixture")["params"].items()}
        params["evap_awake"] = 16384
    being = support.Being(ref, suite="ae12", index=index, law=law, wall=wall, tz=tz, weather=weather,
                          hemisphere=hemisphere, band=band, params=params)
    end, mode, stops = 10 * DAY, (), None
    if index == 0:
        # Winter comes on the first days; a warm once the rest is over wakes the being.
        being.act(120 + draw.below(600), "water")
        being.act(8 * DAY + 600, "warm")
    elif index == 1:
        # A windowsill without rain dries out and sleeps; a water once the rest is over wakes it.
        end = 21 * DAY
        being.act(19 * DAY + draw.below(600), "water")
    elif index == 2:
        # Travel: east, then west over a local midnight; a params evolve on the way.
        being.tz(2 * DAY + 300 + draw.below(300), 780 if tz <= 0 else -600)
        being.tz(3 * DAY + 900 + draw.below(300), tz)
        _evolve(being, 4 * DAY + 77, being.params(evap_awake=6000, sun_max=40000))
        _gestures(being, draw, end, 4)
    elif index == 3:
        # Pins: twice, an evolve that comes due while pinned, an unpin twice, then an evolve that applies.
        being.pin(DAY + 10)
        being.pin(DAY + 20)
        _evolve(being, 2 * DAY + 5, being.params(rain_gain=30000))
        being.unpin(3 * DAY + 1)
        being.unpin(3 * DAY + 2)
        _evolve(being, 4 * DAY + 9, being.params(rain_gain=90000, evap_dormant=2000))
    elif index == 4:
        # Evolves superseded, at the wrong minute and out of the law's ranges; a redacted fact.
        end = 8 * DAY
        _evolve(being, DAY + 100, being.params(evap_awake=5000))
        _evolve(being, DAY + 200, being.params(evap_awake=7000))
        being.evolve(DAY + 300, being.params(), effective_from=being.midnight(DAY + 300, _offset(being, DAY + 300)) + 15)
        _evolve(being, DAY + 400, being.params(sun_max=1000))
        being.fact(2 * DAY + 50, "lang_teach", "ab" * 32)
        being.fact(2 * DAY + 60, "lang_teach", {"payload": "cd" * 16})
    elif index == 5:
        # An evolve that starts from a law the being is not under: noted at its minute.
        end = 6 * DAY
        other = ref.law("v0_1")
        defaults = {name: spec["default"] for name, spec in ref.lawfiles.law("v0_1")["params"].items()}
        _evolve(being, DAY + 30, defaults, to=("v0_1", other["digest"], other["version"]),
                source=("v0_1", other["digest"]))
    elif index == 6:
        # A flood of gestures in one day of life: over its budget, noted, the notes truncated.
        end = 4 * DAY
        for i in range(134):
            being.act(DAY + 17 + i, "greet")
        for i in range(9):
            being.pin(2 * DAY + 40 + i)
        stops = [DAY - 1, 3 * DAY, end]
    elif index == 7:
        mode = ("slices",)
        _gestures(being, draw, end, 6)
        being.act(8 * DAY + 700, "warm")
    elif index == 8:
        mode = ("fast path",)
        being.act(8 * DAY + 800, "warm")
        _gestures(being, draw, end, 3)
    elif index == 9:
        mode = ("order",)
        being.act(8 * DAY + 900, "warm")
        _gestures(being, draw, end, 5)
    else:
        end = 6 * DAY
        being.tz(DAY + 500, OFFSETS[(OFFSETS.index(tz) + 1) % 4])
        _gestures(being, draw, end, 3)
    return being, end, mode, stops


def _ae12_run(lives, being, end, draw, seen, mode=(), stops=None, state=None):
    """Advance a life through its stops in both engines; each answer is observed."""
    if stops is None:
        start = 0 if state is None else state["at"]
        stops = sorted({start + 1 + draw.below(end - start) for _ in range(7)} | {end})
    organs = list(support.ORGANS)
    dormant = None if state is None else state["organs"]["stage"]["dormant"]
    for stop in stops:
        while True:
            probe = {}
            if draw.below(4):
                probe["trace"] = True
            if "fast path" in mode or draw.below(5) == 0:
                probe["fast_path"] = False
            if "order" in mode or draw.below(5) == 0:
                probe["order"] = _shuffled(draw, organs)
            budget = 1500 + draw.below(8000) if "slices" in mode else support.MAX_INT
            request = being.request(state, stop, budget=budget, probe=probe or None)
            answer = lives.both(request)
            assert "refused" not in answer, answer
            _ae12_observe(lives, being, request, answer, seen)
            state = answer["state"]
            asleep = state["organs"]["stage"]["dormant"]
            if dormant and not asleep:
                _count(seen, "wake")
            dormant = asleep
            if answer["done"]:
                assert answer["at"] == stop, (stop, answer["at"])
                break
    return state


def _ae12_observe(lives, being, request, answer, seen):
    assert answer["alarm"] == 0, answer["alarm"]
    state = answer["state"]
    reserve = state["organs"]["chem"]
    c = lives.chem(state["law"]["name"])
    assert c["core"] <= reserve["fructan"] <= c["fructan_max"], reserve
    assert 0 <= reserve["sugar"] <= c["sugar_max"], reserve
    assert reserve["sugar"] + reserve["fructan"] == c["sugar0"] + c["fructan0"] + reserve["made"] - reserve["burnt"]
    _count(seen, "dormant" if state["organs"]["stage"]["dormant"] else "awake")
    if state["tz"] != being.tz_birth:
        _count(seen, "tz")
    if state["params"] != being.genesis["body"]["laws"]["params"]:
        _count(seen, "params evolve")
    if state["pinned"]:
        _count(seen, "pin")
    if not answer["done"]:
        _count(seen, "done false")
    probe = request.get("probe", {})
    if probe.get("fast_path") is False:
        _count(seen, "fast path off")
    if probe.get("order", list(support.ORGANS)) != list(support.ORGANS):
        _count(seen, "order")
    if any(isinstance(fact["body"], str) for fact in request["facts"]):
        _count(seen, "redacted")
    law = lives.law(state["law"]["name"])
    if "trace" in answer:
        # The work both engines counted is the work the trace accounts for by the law's unit table.
        assert support.analytic(answer["trace"], law) == answer["work"], (answer["work"], answer["trace"])
        _count(seen, "analytic")
    position, _length = lives.ref.civil.year_position(answer["env"]["day"], law["world"],
                                                       being.genesis["body"]["hemisphere"])
    if position < law["world"]["daylength"]["equinox"] and not state["organs"]["stage"]["dormant"]:
        _count(seen, "awake before the equinox")
    if "trace" in answer:
        _count(seen, "trace")
        if answer["trace"]["noted"]:
            _count(seen, "noted")
        if any(calls["jump"] for calls in answer["trace"]["calls"].values()):
            _count(seen, "wake")
    for note in answer["notes"]:
        _count(seen, "note " + note["code"])


def test_ae12_both_engines_live_the_same_lives_byte_for_byte():
    loaded, restore = open_allium(native=True)
    seen = {}
    days = 0
    try:
        lives = _Lives(loaded)
        ref = lives.ref
        for index in range(12):
            draw = ref.rng.Stream(bytes(32), "test.ae12", index)
            being, end, mode, stops = _ae12_life(ref, index, draw)
            _ae12_run(lives, being, end, draw, seen, mode, stops)
            days += end // DAY
        # Four lives whose genome constants sit at corners of the law's box, written into a minute-0 state.
        law = ref.lawfiles.law("fixture")
        for corner in range(4):
            draw = ref.rng.Stream(bytes(32), "test.ae12", 12 + corner)
            hemisphere = ("north", "south")[corner % 2]
            being = support.Being(ref, suite="ae12", index=12 + corner, wall=_winter_wall(hemisphere),
                                  hemisphere=hemisphere, tz=OFFSETS[corner])
            being.act(200 + draw.below(400), "water")
            being.act(8 * DAY + 60, "warm")
            genome = lives.both({"corner": corner, "law": "fixture", "op": "genome_corner", "v": 1})["genome"]
            tables = lives.both({"genome": genome, "law": "fixture", "op": "genome_compile", "v": 1})["tables"]
            k = ref.lawdata.consts(tables, law)
            state = lives.both(being.request(None, 0))["state"]
            state["organs"]["chem"]["k"] = k["chem"]
            state["organs"]["clock"]["k"] = k["clock"]
            _ae12_run(lives, being, 8 * DAY, draw, seen, state=state)
            days += 8
            _count(seen, "corner")
    finally:
        restore()

    assert days <= 200, days
    assert sum(lives.native.calls.values()) == lives.compared, "the native engine answered every request"
    assert lives.native.calls.get("advance", 0) >= 100, lives.native.calls
    for key in ("awake", "dormant", "wake", "tz", "params evolve", "pin", "redacted", "noted", "done false",
                "trace", "order", "fast path off"):
        assert seen.get(key, 0) >= 1, (key, seen)
    assert seen.get("analytic", 0) >= 50, ("the unit table accounts for the twin's work", seen)
    assert seen.get("awake before the equinox", 0) >= 1, ("the daylength's negative-safe position is met", seen)
    for code in NOTE_CODES:
        assert seen.get("note " + code, 0) >= 1, (code, seen)
    # The same count met every other code above: its zero here is a zero it could have missed. No seeded
    # life names a version besides the law in force; a crafted state that does is met in AE14.
    assert seen.get("note laws_stale", 0) == 0, "no carried successor, so no stale law field"
    assert seen.get("corner", 0) >= 4, seen


def _edited(base, *edits):
    """A deep copy of ``base`` with each ``(path, value)`` set, or removed when the value is ``_DROP``."""
    request = copy.deepcopy(base)
    for path, value in edits:
        node = request
        for key in path[:-1]:
            node = node[key]
        if value is _DROP:
            del node[path[-1]]
        else:
            node[path[-1]] = value
    return request


def _ae13_advance(lives, being, mid):
    """``(zone, expected, request)`` for defective ``advance`` requests; zones in check order."""
    ref = lives.ref
    base = being.request(mid, 3000)
    assert [fact["t"] for fact in base["facts"]] == [1500, 1600, 2000], base["facts"]
    other = ref.law("v0_1")
    g = ("genesis",)
    body = g + ("body",)
    laws = body + ("laws",)
    st = ("state",)
    f0 = ("facts", 0)
    f2 = ("facts", 2)
    evolve_body = {"effective_from": being.midnight(2000, 60),
                   "from": {"name": "fixture", "sha256": being.digest},
                   "params": being.params(),
                   "to": {"name": "fixture", "sha256": being.digest, "v": being.v}}
    evolve = dict(base["facts"][2], kind="evolve", body=evolve_body)

    def to_law(name, digest, v, source=None):
        edited = copy.deepcopy(evolve)
        edited["body"]["to"] = {"name": name, "sha256": digest, "v": v}
        if source is not None:
            edited["body"]["from"] = source
        return edited

    zones = [
        ("fields", [
            (("bad_request", "fields"), [(("extra",), 1)]),
            (("bad_request", "fields"), [(("budget",), _DROP)]),
            (("bad_request", "fields"), [(("state",), _DROP)]),
        ]),
        ("genesis", [
            (("bad_fact", "fact"), [(g, 5)]),
            (("bad_fact", "fields"), [(g + ("extra",), 1)]),
            (("bad_fact", "being"), [(g + ("being",), "zz")]),
            (("bad_fact", "kind"), [(g + ("kind",), "Genesis")]),
            (("bad_fact", "laws"), [(g + ("laws",), -1)]),
            (("bad_fact", "origin"), [(g + ("origin",), "0")]),
            (("bad_fact", "oseq"), [(g + ("oseq",), -1)]),
            (("bad_fact", "t"), [(g + ("t",), "0")]),
            (("bad_fact", "body"), [(body, "x")]),
            (("bad_fact", "genesis"), [(g + ("kind",), "act")]),
            (("bad_fact", "genesis"), [(g + ("t",), 3)]),
            (("bad_fact", "genesis"), [(g + ("oseq",), 1)]),
            (("bad_fact", "genesis laws"), [(laws, 5)]),
            (("bad_fact", "genesis laws"), [(laws + ("name",), 3)]),
            (("bad_fact", "genesis laws"), [(laws + ("sha256",), _DROP)]),
            (("unknown_law", "law"), [(laws + ("name",), "fixturex")]),
            (("unknown_law", "law digest"), [(laws + ("sha256",), "0" * 64)]),
            (("bad_fact", "body fields"), [(body + ("extra",), 1)]),
            (("bad_fact", "body band symbol"), [(body + ("band",), "huge")]),
            (("bad_fact", "body birth.wall range"), [(body + ("birth", "wall"), -1)]),
            (("bad_fact", "body laws.params.sun_max range"), [(laws + ("params", "sun_max"), 70000)]),
            (("bad_fact", "body seed hex"), [(body + ("seed",), "X" * 64)]),
            (("bad_fact", "genesis laws"), [(laws + ("v",), 1)]),
            (("bad_fact", "genesis laws"), [(laws + ("provisional",), False)]),
            (("bad_fact", "birth"), [(body + ("birth", "wall"), 86399)]),
            (("bad_fact", "birth"), [(body + ("birth", "tz"), 7)]),
            (("bad_fact", "params sun_max"), [(laws + ("params", "sun_max"), 100)]),
            (("bad_fact", "params evap_awake"), [(laws + ("params", "evap_awake"), 20000),
                                                 (laws + ("params", "sun_max"), 100)]),
            (("chain", "laws"), [(g + ("laws",), 1)]),
        ]),
        ("to", [
            (("bad_request", "to"), [(("to",), -1)]),
            (("bad_request", "to"), [(("to",), support.T_MAX + 1)]),
            (("bad_request", "to"), [(("to",), True)]),
        ]),
        ("budget", [
            (("bad_request", "budget"), [(("budget",), 0)]),
            (("bad_request", "budget"), [(("budget",), "x")]),
            (("bad_request", "budget"), [(("budget",), True)]),
        ]),
        ("probe", [
            (("bad_request", "probe"), [(("probe",), 5)]),
            (("bad_request", "probe"), [(("probe",), {"x": True})]),
            (("bad_request", "probe"), [(("probe",), {"order": ["chem", "clock", "soil"]})]),
            (("bad_request", "probe"), [(("probe",), {"order": ["chem", "chem", "soil", "stage"]})]),
            (("bad_request", "probe"), [(("probe",), {"fast_path": "yes"})]),
            (("bad_request", "probe"), [(("probe",), {"trace": 1})]),
        ]),
        ("state", [
            (("bad_request", "state"), [(st, 5)]),
            (("bad_request", "state fields"), [(st + ("n",), _DROP)]),
            (("bad_request", "state fields"), [(st + ("extra",), 0)]),
            (("bad_request", "state at"), [(st + ("at",), -1)]),
            (("bad_request", "state being"), [(st + ("being",), "zz")]),
            (("bad_request", "state being"), [(st + ("being",), "ab" * 16)]),
            (("bad_request", "state budget"), [(st + ("budget",), {"counts": {"zzz": 1}, "day": 0})]),
            (("bad_request", "state budget"), [(st + ("budget",), {"counts": {"genesis": 1}, "day": 0})]),
            (("bad_request", "state budget"), [(st + ("budget",), {"counts": {"act": 0}, "day": 0})]),
            (("bad_request", "state bus"), [(st + ("bus", "metab"), 65537)]),
            (("bad_request", "state day"), [(st + ("day",), -1)]),
            (("bad_request", "state genome"), [(st + ("genome",), "x")]),
            (("bad_request", "state law"), [(st + ("law", "v"), 70000)]),
            (("bad_request", "state n"), [(st + ("n",), -1)]),
            (("bad_request", "state organs"), [(st + ("organs", "chem", "k", "ps"), 65537)]),
            (("bad_request", "state organs"), [(st + ("organs", "stage", "cause"), "x")]),
            (("bad_request", "state organs"), [(st + ("organs", "chem", "sugar"), 10 ** 7)]),
            (("bad_request", "state organs"), [(st + ("organs", "chem", "fructan"), 1)]),
            (("bad_request", "state params"), [(st + ("params", "sun_max"), 70000)]),
            (("bad_request", "state pending"), [(st + ("pending",), {})]),
            (("bad_request", "state pinned"), [(st + ("pinned",), 1)]),
            (("bad_request", "state schema"), [(st + ("schema",), 2)]),
            (("bad_request", "state seen"), [(st + ("seen",), [])]),
            (("bad_request", "state seen"), [(st + ("seen",), [1])]),
            (("bad_request", "state through"), [(st + ("through",), [5, "zz", 0])]),
            (("bad_request", "state through"), [(st + ("through",), [1001, being.origin, 0])]),
            (("bad_request", "state tz"), [(st + ("tz",), 7)]),
            (("unknown_law", "law"), [(st + ("law", "name"), "fixturex")]),
            (("unknown_law", "law digest"), [(st + ("law", "sha256"), "0" * 64)]),
            (("unknown_law", "law version"), [(st + ("law", "v"), 3), (st + ("seen",), [0, 3])]),
            (("bad_request", "to"), [(("to",), 999)]),
        ]),
        ("facts", [
            (("bad_request", "facts"), [(("facts",), 5)]),
            (("bad_fact", "fact"), [(f0, 5)]),
            (("bad_fact", "fields"), [(f0 + ("extra",), 1)]),
            (("bad_fact", "being"), [(f0 + ("being",), "zz")]),
            (("bad_fact", "kind"), [(f0 + ("kind",), "Act")]),
            (("bad_fact", "laws"), [(f0 + ("laws",), -1)]),
            (("bad_fact", "origin"), [(f0 + ("origin",), "zz")]),
            (("bad_fact", "oseq"), [(f0 + ("oseq",), True)]),
            (("bad_fact", "t"), [(f0 + ("t",), "1500")]),
            (("bad_fact", "body"), [(f0 + ("body",), 5)]),
            (("bad_fact", "t"), [(f0 + ("t",), -1)]),
            (("limit", "body size"), [(f0 + ("body",), {"act": "water", "x" * 4200: 1})]),
            (("bad_fact", "being"), [(f0 + ("being",), "ab" * 16)]),
            (("bad_fact", "kind"), [(f0 + ("kind",), "zzz")]),
            (("bad_fact", "scope"), [(f0 + ("kind",), "presence_hour")]),
            (("bad_fact", "genesis"), [(f0 + ("kind",), "genesis")]),
            (("bad_fact", "reserved"), [(f0 + ("kind",), "braid")]),
            (("bad_fact", "body act symbol"), [(f0 + ("body",), {"act": "dance"})]),
            (("bad_fact", "body fields"), [(f0 + ("body",), {})]),
            (("bad_fact", "redaction"), [(f0 + ("body",), "ab" * 32)]),
            (("chain", "order"), [(f0 + ("t",), 900)]),
            (("chain", "order"), [(f0 + ("t",), 1600), (f0 + ("oseq",), base["facts"][1]["oseq"])]),
            (("chain", "order"), [(f0 + ("t",), 1700)]),
            (("bad_request", "to"), [(f0 + ("t",), 3001)]),
        ]),
        ("fold", [
            (("chain", "laws"), [(f2 + ("laws",), 1)]),
            (("unknown_law", "law"), [(f2, to_law("fixturex", being.digest, 0))]),
            (("unknown_law", "law digest"), [(f2, to_law("fixture", "0" * 64, 0))]),
            (("unknown_law", "law version"), [(f2, to_law("fixture", being.digest, 1))]),
            (("unknown_law", "migration"), [(f2, to_law("v0_1", other["digest"], other["version"]))]),
        ]),
    ]
    out = []
    for zone, singles in zones:
        for expected, edits in singles:
            out.append((zone, expected, _edited(base, *edits), edits))
    return base, out


def _ae13_timeline(lives):
    """``(expected, request)`` for defective ``timeline`` requests, on a being with a ``tz``, an evolve and a pin."""
    being = support.Being(lives.ref, suite="ae13", index=1, weather="windowsill")
    being.tz(1600, 60)
    _evolve(being, 1700, being.params(evap_dormant=3000))
    being.pin(2500)
    base = {"facts": list(being.facts), "from": None,
            "genesis": being.genesis, "op": "timeline", "to": 5000, "v": 1}
    tstate = lives.both(dict(base, facts=[f for f in base["facts"] if f["t"] <= 1000], to=1000))["state"]
    later = dict(base, facts=[f for f in base["facts"] if f["t"] > 1000], **{"from": tstate})
    other = lives.ref.law("v0_1")
    out = [
        (("bad_request", "fields"), _edited(base, (("from",), _DROP))),
        (("bad_request", "fields"), _edited(base, (("state",), None))),
        (("bad_fact", "genesis laws"), _edited(base, (("genesis", "body", "laws"), 5))),
        (("bad_fact", "birth"), _edited(base, (("genesis", "body", "birth", "tz"), 20))),
        (("bad_request", "to"), _edited(base, (("to",), -1))),
        (("bad_request", "midnights_after"), _edited(base, (("midnights_after",), -1))),
        (("bad_request", "midnights_after"), _edited(base, (("midnights_after",), "0"))),
        (("bad_request", "state"), _edited(later, (("from",), 5))),
        (("bad_request", "state fields"), _edited(later, (("from", "organs"), {}))),
        (("bad_request", "state tz"), _edited(later, (("from", "tz"), 900))),
        (("bad_request", "state pending"), _edited(later, (("from", "pending"), {"to": 1}))),
        (("unknown_law", "law"), _edited(later, (("from", "law", "name"), "zz"))),
        (("bad_request", "to"), _edited(later, (("to",), 999))),
        (("bad_request", "timeline kind"), _edited(base, (("facts",), base["facts"] + [
            dict(base["facts"][-1], kind="act", body={"act": "water"}, t=4000, oseq=99)]))),
        (("bad_fact", "oseq"), _edited(base, (("facts", 0, "oseq"), -1))),
        (("chain", "order"), _edited(later, (("facts", 0, "t"), 1000))),
        (("chain", "laws"), _edited(base, (("facts", 0, "laws"), 2))),
        (("unknown_law", "migration"), _edited(base, (("facts", 1, "body", "to"), {
            "name": "v0_1", "sha256": other["digest"], "v": other["version"]}))),
        (("unknown_law", "law version"), _edited(base, (("facts", 1, "body", "to", "v"), 9))),
        (("limit", "items"), _edited(base, (("to",), 100002 * DAY), (("midnights_after",), 0))),
        # Two defects: the genesis before the minute, the state before the facts.
        (("bad_fact", "birth"), _edited(base, (("genesis", "body", "birth", "tz"), 20), (("to",), -1))),
        (("bad_request", "state tz"), _edited(later, (("from", "tz"), 900), (("facts", 0, "t"), 1000))),
        (("bad_request", "midnights_after"), _edited(base, (("midnights_after",), -1),
                                                     (("facts", 0, "laws"), 2))),
    ]
    return out


def _ae13_grid():
    """``(expected, request)`` for defective ``grid`` requests."""
    base = {"birth": {"tz": 0, "wall": support.WALL}, "op": "grid", "ts": [0, 100, 5000],
            "tz": [[50, 60], [2000, -300]], "v": 1}
    # One list over the item limit is enough to meet the refusal; each costs the reference a long parse.
    return [
        (("bad_request", "fields"), _edited(base, (("x",), 1))),
        (("bad_request", "birth"), _edited(base, (("birth",), 5))),
        (("bad_request", "birth"), _edited(base, (("birth", "tz"), _DROP))),
        (("bad_request", "birth"), _edited(base, (("birth", "wall"), 86399))),
        (("bad_request", "birth"), _edited(base, (("birth", "wall"), True))),
        (("bad_request", "birth"), _edited(base, (("birth", "tz"), 7))),
        (("bad_request", "tz"), _edited(base, (("tz",), "x"))),
        (("bad_request", "tz"), _edited(base, (("tz", 0), [1]))),
        (("bad_request", "tz"), _edited(base, (("tz", 1, 0), 10))),
        (("bad_request", "tz"), _edited(base, (("tz", 0, 1), 900))),
        (("bad_request", "tz"), _edited(base, (("tz", 0, 0), -1))),
        (("bad_request", "tz"), _edited(base, (("tz", 1), [2000, "x"]))),
        (("bad_request", "ts"), _edited(base, (("ts",), 5))),
        (("bad_request", "ts"), _edited(base, (("ts", 1), support.T_MAX + 1))),
        (("bad_request", "ts"), _edited(base, (("ts", 1), False))),
        (("limit", "items"), _edited(base, (("ts",), [0] * 100001))),
        (("bad_request", "birth"), _edited(base, (("birth", "tz"), 7), (("tz", 0, 1), 900))),
        (("bad_request", "tz"), _edited(base, (("tz", 0, 1), 900), (("ts", 1), -1))),
        (("bad_request", "birth"), _edited(base, (("birth", "wall"), 5), (("ts", 0), "x"))),
    ]


def _ae13_sequences(lives, seen):
    """Seeded sequences of the law kinds folded by ``timeline`` in both engines, from the genesis and resumed."""
    ref = lives.ref
    law = ref.lawfiles.law("fixture")
    for index in range(12):
        draw = ref.rng.Stream(bytes(32), "test.ae13", 100 + index)
        being = support.Being(ref, suite="ae13", index=100 + index, weather="windowsill", tz=OFFSETS[index % 4])
        t = 0
        for _step in range(14):
            t += 1 + draw.below(3 * DAY)
            choice = draw.below(6)
            offset = _offset(being, t)
            params = {name: spec["lo"] + draw.below(spec["hi"] - spec["lo"] + 1)
                      for name, spec in law["params"].items()}
            if choice == 0:
                being.tz(t, max(-840, min(840, OFFSETS[draw.below(4)] + 15 * (draw.below(5) - 2))))
            elif choice == 1:
                _evolve(being, t, params)
            elif choice == 2:
                being.evolve(t, params, effective_from=being.midnight(t, offset) + 15 * draw.below(3))
            elif choice == 3:
                being.pin(t)
            elif choice == 4:
                being.unpin(t)
            else:
                first, second = (LOW, HIGH) if draw.below(2) else (HIGH, LOW)
                new = OFFSETS[draw.below(4)]
                being.tz(t, new, origin=first)
                being.evolve(t, params, effective_from=being.midnight(t, new), origin=second)
                _count(seen, "same minute tz first" if first == LOW else "same minute evolve first")
        tstate = None
        for stop in sorted({1 + draw.below(t + DAY) for _ in range(3)} | {t + 2 * DAY}):
            at = None if tstate is None else tstate["at"]
            if at is not None and stop <= at:
                continue
            request = {"facts": [f for f in being.after(at, stop) if f["kind"] in ("evolve", "laws_pin",
                                                                                  "laws_unpin", "tz")],
                       "from": tstate, "genesis": being.genesis, "op": "timeline", "to": stop, "v": 1}
            if draw.below(2):
                request["midnights_after"] = draw.below(stop + 1)
            answer = lives.both(request)
            assert "refused" not in answer, answer
            tstate = answer["state"]
            _count(seen, "timeline")
            for note in answer["notes"]:
                _count(seen, "timeline note " + note["code"])


def test_ae13_both_engines_refuse_every_defective_life_request_with_the_same_bytes():
    loaded, restore = open_allium(native=True)
    met = {}
    seen = {}
    try:
        lives = _Lives(loaded)
        ref = lives.ref
        being = support.Being(ref, suite="ae13", index=0, weather="garden")
        being.act(1500, "water")
        being.tz(1600, 60)
        being.act(2000, "greet")
        mid = lives.both(being.request(None, 1000))["state"]
        base, singles = _ae13_advance(lives, being, mid)
        answers = {}
        for zone, expected, request, _edits in singles:
            answer = lives.both(request)
            assert (answer.get("refused"), answer.get("detail")) == expected, (zone, expected, answer)
            met[expected] = met.get(expected, 0) + 1
            answers.setdefault(zone, []).append((request, answer))
        big = _edited(base, (("genesis", "body", "owner"), "a" * (1 << 20)))
        answer = lives.both(big)
        met[(answer["refused"], answer["detail"])] = 1
        # Two and three defects in different zones: the first in check order is the one named. The
        # edits of the later zones are made first, so an earlier zone's edit is never undone by a later one's.
        zones = ("fields", "genesis", "to", "budget", "probe", "state", "facts", "fold")
        draw = ref.rng.Stream(bytes(32), "test.ae13", 1)
        combined = 0
        by_zone = {}
        for zone, expected, request, edits in singles:
            by_zone.setdefault(zone, []).append((expected, edits))
        for _round in range(260):
            first, second = sorted(draw.below(len(zones)) for _ in range(2))
            if first == second:
                continue
            chosen = [by_zone[zones[first]][draw.below(len(by_zone[zones[first]]))],
                      by_zone[zones[second]][draw.below(len(by_zone[zones[second]]))]]
            if draw.below(3) == 0 and second + 1 < len(zones):
                third = zones[second + 1 + draw.below(len(zones) - second - 1)]
                chosen.append(by_zone[third][draw.below(len(by_zone[third]))])
            edits = [edit for _expected, group in reversed(chosen) for edit in group]
            answer = lives.both(_edited(base, *edits))
            assert (answer["refused"], answer["detail"]) == chosen[0][0], (chosen, answer)
            combined += 1
        for expected, request in _ae13_timeline(lives) + _ae13_grid():
            answer = lives.both(request)
            assert (answer.get("refused"), answer.get("detail")) == expected, (expected, answer)
            met[expected] = met.get(expected, 0) + 1
        _ae13_sequences(lives, seen)
    finally:
        restore()

    assert sum(lives.native.calls.values()) == lives.compared, "the native engine answered every request"
    for op in ("advance", "grid", "timeline"):
        assert lives.native.calls.get(op, 0) >= 19, (op, lives.native.calls)
    reachable = [
        ("bad_request", "fields"), ("bad_request", "to"), ("bad_request", "budget"), ("bad_request", "probe"),
        ("bad_request", "state"), ("bad_request", "state fields"), ("bad_request", "state seen"),
        ("bad_request", "state through"), ("bad_request", "state organs"), ("bad_request", "state being"),
        ("bad_request", "birth"), ("bad_request", "tz"), ("bad_request", "ts"), ("bad_request", "midnights_after"),
        ("bad_request", "timeline kind"), ("bad_request", "facts"),
        ("bad_fact", "fact"), ("bad_fact", "fields"), ("bad_fact", "being"), ("bad_fact", "kind"),
        ("bad_fact", "laws"), ("bad_fact", "origin"), ("bad_fact", "oseq"), ("bad_fact", "t"), ("bad_fact", "body"),
        ("bad_fact", "scope"), ("bad_fact", "genesis"), ("bad_fact", "genesis laws"), ("bad_fact", "reserved"),
        ("bad_fact", "redaction"), ("bad_fact", "birth"), ("bad_fact", "params sun_max"),
        ("bad_fact", "body act symbol"),
        ("chain", "order"), ("chain", "laws"),
        ("unknown_law", "law"), ("unknown_law", "law digest"), ("unknown_law", "law version"),
        ("unknown_law", "migration"),
        ("limit", "items"), ("limit", "body size"), ("limit", "input size"),
    ]
    missing = [pair for pair in reachable if pair not in met]
    assert missing == [], missing
    assert combined >= 60, combined
    assert seen.get("same minute tz first", 0) >= 1 and seen.get("same minute evolve first", 0) >= 1, seen
    assert seen.get("timeline", 0) >= 30, seen
    assert lives.compared >= 400, lives.compared


# ---------------------------------------------------------------------------
# AE14 -- what the seeded lives never ask, in both engines
# ---------------------------------------------------------------------------
HEX32 = "cd" * 16
HEX64 = "ab" * 32


def _off_grid(being, t):
    """``t``, or the minute after it when ``t`` is a fast boundary."""
    return t + 1 if (being.b + t) % 15 == 0 else t


def _ae14_cuts(lives, being, stops, state=None, probe=None):
    """Advance through ``stops`` in both engines, one call each; the answers, in order."""
    answers = []
    for stop in stops:
        answer = lives.both(being.request(state, stop, probe=probe))
        assert "refused" not in answer and answer["done"] and answer["at"] == stop, (stop, answer)
        state = answer["state"]
        answers.append(answer)
    return answers


def _ae14_same_minute(lives, seen):
    """A ``tz`` fact and an ``evolve`` in one minute, in both canonical orders: the same pending change."""
    pendings = []
    for index, (tz_origin, evolve_origin) in enumerate(((LOW, HIGH), (HIGH, LOW))):
        being = support.Being(lives.ref, suite="ae14", index=index, weather="garden")
        e = _off_grid(being, 2 * DAY + 437)
        effective = being.midnight(e, -300)
        being.tz(e, -300, origin=tz_origin)
        being.evolve(e, being.params(evap_awake=7000, sun_max=45000), effective_from=effective, origin=evolve_origin)
        kinds = [fact["kind"] for fact in being.facts if fact["t"] == e]
        before, at, after, due_before, due, later = _ae14_cuts(
            lives, being, [e - 1, e, e + 1, effective - 1, effective, effective + DAY], probe={"trace": True})
        assert before["state"]["pending"] is None and at["notes"] == [], (kinds, at["notes"])
        assert at["state"]["pending"]["effective_from"] == effective and at["state"]["tz"] == -300, kinds
        assert due_before["state"]["pending"] == at["state"]["pending"] and due["state"]["pending"] is None
        assert due["state"]["params"]["sun_max"] == 45000 != due_before["state"]["params"]["sun_max"], kinds
        pendings.append(at["state"]["pending"])
        _count(seen, "same minute " + " then ".join(kinds))
    assert pendings[0] == pendings[1], "both canonical orders register the same pending change"


def _ae14_moved(lives, seen):
    """An ``evolve`` whose minute a later ``tz`` fact moves off its midnight, awake and asleep, reached around it."""
    for index, weather in ((2, "garden"), (3, "windowsill")):
        being = support.Being(lives.ref, suite="ae14", index=index, weather=weather)
        day = 5 if weather == "garden" else 25
        e = _off_grid(being, day * DAY + 611)
        effective = being.midnight(e, 0)
        being.evolve(e, being.params(evap_dormant=3000, evap_awake=5000), effective_from=effective)
        being.tz(_off_grid(being, e + 5 * 60), 300)
        assert (being.b + effective + 300) % DAY != 0, "the evolve's minute is no longer a local midnight"
        start = lives.both(being.request(None, e - 7))["state"]
        stops = [effective - 1, effective, effective + 1, effective + 7 * 60, being.midnight(effective, 300) + DAY]
        runs = []
        for probe in ({"trace": True}, {"fast_path": False, "trace": True}):
            runs.append(_ae14_cuts(lives, being, stops, state=start, probe=probe))
        for fast, stepped in zip(*runs):
            assert fast["state"] == stepped["state"], ("the fast path and the steps agree", weather, fast["at"])
        applied = [answer["state"]["params"]["evap_dormant"] == 3000 for answer in runs[0]]
        assert applied == [False, True, True, True, True], (weather, applied)
        asleep = [answer["state"]["organs"]["stage"]["dormant"] for answer in runs[0]]
        if weather == "windowsill":
            assert all(asleep), "the windowsill sleeps across the moved minute"
            assert sum(answer["trace"]["fast_path_days"] for answer in runs[0]) >= 1, "the fast path ran"
        else:
            assert not any(asleep), "the garden is awake across the moved minute"
        _count(seen, "moved evolve " + weather)


def _ae14_exempt_flood(lives, seen):
    """One day of life with one fact past the cap of each budget-exempt kind: each excess one is noted."""
    ref = lives.ref
    law = ref.lawfiles.law("fixture")
    caps = law["work"]["caps"]
    bodies = {"clock": {"behind": 1}, "owner": {"from": HEX32, "to": HEX32},
              "resumed": {"digest": HEX64, "removed": 0}, "tz": {"quarters": 0}}
    assert sorted(caps) == sorted(bodies), caps
    being = support.Being(ref, suite="ae14", index=4, weather="garden")
    plan = [kind for kind in sorted(caps) for _ in range(caps[kind] + 1)]
    draw = ref.rng.Stream(bytes(32), "test.ae14", 4)
    for i in range(len(plan) - 1, 0, -1):
        j = draw.below(i + 1)
        plan[i], plan[j] = plan[j], plan[i]
    minutes = [t for t in range(DAY + 1, 2 * DAY) if (being.b + t) % 15]
    last = {}
    for t, kind in zip(minutes, plan):
        being.fact(t, kind, bodies[kind])
        last[kind] = t
    before = lives.both(being.request(None, DAY - 1))["state"]
    answer = lives.both(being.request(before, 2 * DAY - 1, probe={"trace": True}))
    assert "refused" not in answer, answer
    assert answer["notes"] == [{"code": "budget", "t": t} for t in sorted(last.values())], answer["notes"]
    assert answer["trace"]["noted"] == len(caps) and answer["trace"]["facts"] == len(plan), answer["trace"]
    assert answer["work"] <= law["work"]["ceilings"]["awake_day"], answer["work"]
    assert answer["state"]["budget"]["counts"] == caps, answer["state"]["budget"]
    _count(seen, "exempt flood")


def _ae14_pending(lives, seen, reached):
    """States whose pending change names a law this engine cannot approach: refused when it comes due, not before."""
    ref = lives.ref
    being = support.Being(ref, suite="ae14", index=5, weather="garden")
    mid = lives.both(being.request(None, 1000))["state"]
    effective = being.midnight(1000, 0)
    full = ref.law("v0_1")
    targets = (
        ({"name": "v0_1", "sha256": full["digest"], "v": full["version"]}, "migration"),
        ({"name": "fixturex", "sha256": being.digest, "v": being.v}, "law"),
        ({"name": "fixture", "sha256": "0" * 64, "v": being.v}, "law digest"),
        ({"name": "fixture", "sha256": being.digest, "v": being.v + 7}, "law version"),
    )
    for target, detail in targets:
        state = copy.deepcopy(mid)
        state["pending"] = {"effective_from": effective, "from": {"name": "fixture", "sha256": being.digest},
                            "params": being.params(), "to": target}
        tstate = {key: state[key] for key in TSTATE}
        for op in ("advance", "timeline"):
            for to in (effective - 1, effective):
                if op == "advance":
                    request = being.request(state, to)
                else:
                    request = {"facts": [], "from": tstate, "genesis": being.genesis, "op": "timeline", "to": to,
                               "v": 1}
                answer = lives.both(request)
                if to < effective:
                    assert "refused" not in answer and answer["state"]["pending"] == state["pending"], (op, detail)
                else:
                    assert answer == {"detail": detail, "refused": "unknown_law"}, (op, target, answer)
                    reached[(op, detail)] = reached.get((op, detail), 0) + 1
    _count(seen, "pending approached")


def _ae14_stale(lives, seen):
    """A state whose versions seen hold one besides the law in force: a fact carrying it is noted and folded."""
    ref = lives.ref
    lives_by_laws = {}
    for laws in (0, 3):
        being = support.Being(ref, suite="ae14", index=6, weather="garden")
        being.act(1500, "water", laws=laws)
        being.act(1600, "greet")
        lives_by_laws[laws] = being
    mid = lives.both(lives_by_laws[0].request(None, 1000))["state"]
    crafted = copy.deepcopy(mid)
    crafted["seen"] = [0, 3]
    honest = lives.both(lives_by_laws[0].request(mid, 3000, probe={"trace": True}))
    stale = lives.both(lives_by_laws[3].request(crafted, 3000, probe={"trace": True}))
    assert honest["notes"] == [] and stale["notes"] == [{"code": "laws_stale", "t": 1500}], stale["notes"]
    assert stale["trace"]["noted"] == honest["trace"]["noted"] == 0, "a stale fact is folded, not dropped"
    assert stale["state"]["organs"] == honest["state"]["organs"] and stale["state"]["n"] == honest["state"]["n"]
    assert stale["state"]["seen"] == [0, 3] and stale["state"]["law"] == honest["state"]["law"]
    _count(seen, "note laws_stale")
    # Over its day's budget, a stale fact is noted for the budget alone, and not folded.
    full = copy.deepcopy(crafted)
    full["budget"] = {"counts": {"act": ref.lawfiles.law("fixture")["journal"]["budgets"]["act"]}, "day": 1}
    over = lives.both(lives_by_laws[3].request(full, 3000, probe={"trace": True}))
    bare = support.Being(ref, suite="ae14", index=6, weather="garden")
    dry = lives.both(bare.request(mid, 3000))
    assert [note["code"] for note in over["notes"]] == ["budget", "budget"], over["notes"]
    assert over["trace"]["noted"] == 2 and over["state"]["organs"]["soil"] == dry["state"]["organs"]["soil"] != \
        stale["state"]["organs"]["soil"], "neither act is folded"
    _count(seen, "stale over budget")


TSTATE = ("at", "budget", "day", "law", "params", "pending", "pinned", "schema", "seen", "through", "tz")


def test_ae14_both_engines_agree_on_same_minute_evolves_exempt_floods_and_crafted_states():
    loaded, restore = open_allium(native=True)
    seen = {}
    reached = {}
    try:
        lives = _Lives(loaded)
        _ae14_same_minute(lives, seen)
        _ae14_moved(lives, seen)
        _ae14_exempt_flood(lives, seen)
        _ae14_pending(lives, seen, reached)
        _ae14_stale(lives, seen)
    finally:
        restore()

    assert sum(lives.native.calls.values()) == lives.compared, "the native engine answered every request"
    assert lives.native.calls.get("advance", 0) >= 40 and lives.native.calls.get("timeline", 0) >= 8, \
        lives.native.calls
    for key in ("same minute tz then evolve", "same minute evolve then tz", "moved evolve garden",
                "moved evolve windowsill", "exempt flood", "pending approached", "note laws_stale",
                "stale over budget"):
        assert seen.get(key, 0) >= 1, (key, seen)
    for op in ("advance", "timeline"):
        for detail in ("migration", "law", "law digest", "law version"):
            assert reached.get((op, detail), 0) >= 1, (op, detail, reached)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
