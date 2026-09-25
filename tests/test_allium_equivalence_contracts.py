#!/usr/bin/env python3
"""Contracts for the companion's genome in both engines: the same bytes, request for request.

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

Local-only. It needs the native artefact that ``scripts/build_oo_core.sh``
builds.
"""

import hashlib
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_genome_corpus as corpus  # noqa: E402
from _allium_window import native_module, open_allium  # noqa: E402
from _isolation import REPO  # noqa: E402

GOLDEN = REPO / "tests" / "allium_golden" / "v1" / "genome.json"
OPS = ("genome_compile", "genome_corner", "genome_decode", "genome_found")

BUDGET_S = {
    "test_ae1_both_engines_answer_every_genome_request_with_the_same_bytes": 2.0,
    "test_ae2_both_engines_answer_the_whole_decoder_corpus_with_the_same_bytes": 2.0,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
