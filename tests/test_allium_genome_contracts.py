#!/usr/bin/env python3
"""Contracts for the companion's genome, on the reference: codec, founders, compilation, bounds.

A genome is the being's fixed inheritance: a diploid set of 16-byte records
whose order is itself data (a gene's promoter is the run of CIS records just
before it). It is derived from a 32-byte seed through the law's founder
pool, never stored as the being's state -- the seed, the law and the pool
are -- and compiled into flat tables each time it is needed.

  * GN1 -- a founder is derived deterministically: the reference reproduces
    the committed golden genomes, digests and table digests, in this
    process and in two children under different string hash seeds.
  * GN2 -- a genome round-trips through its canonical bytes; every single
    pad bit set is refused by name, so is an unused flag bit, and encoding
    a field outside its bit width is refused rather than spilled.
  * GN3 -- the decoder is total: over a seeded corpus of mutations, every
    targeted defect, the full-law cases and the request corpus, each input
    is either accepted (and a reader written apart from the reference
    agrees it is sound) or refused with a code of the closed set and a
    detail of the published grammar; every detail class appears.
  * GN4 -- recompiling a genome reproduces its tables byte for byte, with
    each gene's edges in ascending CIS order.
  * GN5 -- each locus draws from its own stream: an extra allele at one
    locus changes that locus alone, a law that reverses the template yields
    the same bodies, and the corner and founder domains differ.
  * GN7 -- the bound analysis each law records equals the one recomputed
    from its boxes, byte for byte, and every bound meets its ceiling.
  * GN8 -- at the corners of the box, every factor a genome supplies keeps
    every fixed-point primitive inside its domain: no alarm.
  * GN9 -- dominance follows each locus's mode, and the tables do not
    depend on which homolog carries which allele.
  * GN10 -- each law and its pool have one source, pinned by digest: both
    validate, the pin matches, the authoring script reproduces all four
    files byte for byte, the native engine reports the same pins, and the
    validators name the defects they exist to catch.
  * GN11 -- the greenhouse draws alleles at the pool's frequencies,
    reproducibly, writes nothing, and counts how many genomes are distinct.

Local-only. The modules load through the shared isolation window; GN10
also needs the native artefact that ``scripts/build_oo_core.sh`` builds.
"""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_genome_corpus as corpus  # noqa: E402
from _allium_window import native_module, open_allium  # noqa: E402
from _isolation import REPO  # noqa: E402

GOLDEN = REPO / "tests" / "allium_golden" / "v1" / "genome.json"
LAWS_DIR = REPO / "opti_oignon" / "allium" / "laws"
SCRIPTS = REPO / "scripts"
ONE = 1 << 16
CMAX = 8 * ONE
I32_MAX = (1 << 31) - 1

BUDGET_S = {
    "test_gn1_founders_are_derived_deterministically_and_match_the_golden_vectors": 2.0,
    "test_gn2_a_genome_round_trips_through_its_canonical_bytes": 2.0,
    "test_gn3_the_decoder_accepts_inside_the_box_or_refuses_by_name_and_never_raises": 2.0,
    "test_gn4_recompiling_reproduces_the_tables_byte_for_byte": 2.0,
    "test_gn5_each_locus_draws_from_its_own_stream": 2.0,
    "test_gn7_the_recorded_bounds_equal_the_recomputed_ones_within_their_ceilings": 2.0,
    "test_gn8_at_the_box_corners_every_genome_factor_keeps_the_primitives_in_their_domain": 2.0,
    "test_gn9_dominance_follows_each_mode_and_ignores_which_homolog_carries_what": 2.0,
    "test_gn10_each_law_and_its_pool_have_one_source_pinned_by_digest": 2.0,
    "test_gn11_the_greenhouse_draws_at_the_pools_frequencies_and_writes_nothing": 2.0,
}


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


class Ref:
    """The reference modules of one window, and a few conveniences over them."""

    def __init__(self, loaded):
        self.loaded = loaded
        self.wire = loaded["opti_oignon.allium.wire"]
        self.rng = loaded["opti_oignon.allium.rng"]
        self.fx = loaded["opti_oignon.allium.fx"]
        self.lawfiles = loaded["opti_oignon.allium.lawfiles"]
        self.genome = loaded["opti_oignon.allium.ref.organs.genome"]
        self.compile = loaded["opti_oignon.allium.ref.organs.compile"]
        self.bounds = loaded["opti_oignon.allium.ref.organs.bounds"]
        self.protocol = loaded["opti_oignon.allium.ref.protocol"]

    def ask(self, request):
        return self.wire.parse(self.protocol.call(self.wire.emit(dict(request, v=1))))

    def law(self, name):
        law = self.lawfiles.law(name)
        return law, self.genome.view(law)

    def pool(self, law):
        return self.genome.pool_alleles(self.lawfiles.founders(law["founders"]["name"]))

    def tables(self, data, law, lawview=None):
        lawview = lawview or self.genome.view(law)
        tables, _work = self.compile.compile_genome(data, law, lawview, self.lawfiles.digest(law))
        return tables

    def column(self, tables, *path):
        node = tables
        for key in path:
            node = node[key]
        return self.wire.unpack_bulk(node)[1]


@pytest.fixture
def ref():
    loaded, restore = open_allium(native=False)
    try:
        yield Ref(loaded)
    finally:
        restore()


def _golden():
    return json.loads(GOLDEN.read_text(encoding="ascii"))


def _genome_golden(ask):
    seeds = {"fixture": [bytes(range(32)).hex(), "ff" * 32, "a5" * 32], "v0_1": [bytes(range(32)).hex(), "ff" * 32]}
    corners = {"fixture": 3, "v0_1": 2}
    out = {}
    for law, chosen in seeds.items():
        found = []
        for seed in chosen:
            founder = ask({"law": law, "op": "genome_found", "seed": seed})
            compiled = ask({"genome": founder["genome"], "law": law, "op": "genome_compile"})
            entry = {"seed": seed, "sha256": founder["sha256"], "tables_sha256": compiled["sha256"]}
            if law == "fixture":
                entry["genome"] = founder["genome"]
            found.append(entry)
        out[law] = {"corner_sha256": [ask({"corner": k, "law": law, "op": "genome_corner"})["sha256"]
                                      for k in range(corners[law])],
                    "found": found}
    return out


# ---------------------------------------------------------------------------
# GN1 -- founders are deterministic
# ---------------------------------------------------------------------------
_REPLAY = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from opti_oignon.allium import wire
from opti_oignon.allium.ref import protocol as p
def ask(obj):
    return wire.parse(p.call(wire.emit(dict(obj, v=1))))
seeds = {"fixture": [bytes(range(32)).hex(), "ff" * 32, "a5" * 32], "v0_1": [bytes(range(32)).hex(), "ff" * 32]}
corners = {"fixture": 3, "v0_1": 2}
out = {}
for law, chosen in seeds.items():
    found = []
    for seed in chosen:
        founder = ask({"law": law, "op": "genome_found", "seed": seed})
        compiled = ask({"genome": founder["genome"], "law": law, "op": "genome_compile"})
        entry = {"seed": seed, "sha256": founder["sha256"], "tables_sha256": compiled["sha256"]}
        if law == "fixture":
            entry["genome"] = founder["genome"]
        found.append(entry)
    out[law] = {"corner_sha256": [ask({"corner": k, "law": law, "op": "genome_corner"})["sha256"] for k in range(corners[law])], "found": found}
print(json.dumps(out, sort_keys=True))
"""


def test_gn1_founders_are_derived_deterministically_and_match_the_golden_vectors(ref, tmp_path):
    golden = _golden()
    assert len(golden["fixture"]["found"]) == 3 and len(golden["v0_1"]["found"]) == 2
    assert _genome_golden(ref.ask) == golden, "the reference reproduces the committed founders"
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run([sys.executable, "-c", _REPLAY, str(REPO)], cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr[-2000:]
        assert json.loads(run.stdout) == golden, f"the founders move under PYTHONHASHSEED={hash_seed}"
    assert list(tmp_path.iterdir()) == [], "the replay wrote nothing"


# ---------------------------------------------------------------------------
# GN2 -- round trip, canonical bytes, pads
# ---------------------------------------------------------------------------
def test_gn2_a_genome_round_trips_through_its_canonical_bytes(ref):
    g = ref.genome
    law, lawview = ref.law("fixture")
    golden = _golden()["fixture"]
    genomes = [bytes.fromhex(entry["genome"]) for entry in golden["found"]]
    genomes += [g.corner(k, lawview)[0] for k in range(16)]
    for index, data in enumerate(genomes):
        value = g.decode(data, lawview)
        assert g.encode(value) == data, f"genome {index} does not re-encode to itself"
        assert g.decode(g.encode(value), lawview) == value, f"genome {index} does not decode back"
    alleles = ref.pool(law)
    data, _chosen, _work = g.found(bytes(range(32)), lawview, alleles)
    assert data.hex() == golden["found"][0]["genome"]

    founder = genomes[0]
    pairs, chrom = corpus.split(founder)
    tried = 0
    kinds = {}
    for j, record in enumerate(chrom[0]):
        used = corpus.used_mask(record[0])
        for b in range(10):
            for bit in range(8):
                if used[b] >> bit & 1:
                    continue
                padded = bytearray(record)
                padded[6 + b] |= 1 << bit
                bad = corpus.frame([chrom[0][:j] + [bytes(padded)] + chrom[0][j + 1:], chrom[1]], pairs)
                with pytest.raises(ref.wire.Refused) as refused:
                    g.decode(bad, lawview)
                assert (refused.value.code, refused.value.detail) == ("bad_request", f"genome pad c0 r{j}")
                tried += 1
                kinds[record[0]] = True
    assert tried >= 300 and len(kinds) == 13, (tried, sorted(kinds))  # enz, lex and vern use all ten bytes

    flagged = bytearray(founder)
    flagged[8 + 2 + 1] |= 0x20
    with pytest.raises(ref.wire.Refused) as refused:
        g.decode(bytes(flagged), lawview)
    assert refused.value.detail == "genome flags c0 r0"

    value = g.decode(founder, lawview)
    plast = value["chromosomes"][0][11]
    assert plast["kind"] == 8
    plast["fields"][6] = 16
    with pytest.raises(ref.wire.Refused) as refused:
        g.encode(value)
    assert (refused.value.code, refused.value.detail) == ("bad_request", "genome box c0 r11 LR")


# ---------------------------------------------------------------------------
# GN3 -- the decoder is total
# ---------------------------------------------------------------------------
def _gn3_items(ref):
    fixture, fixture_view = ref.law("fixture")
    v0_1, v0_1_view = ref.law("v0_1")
    reader_fixture, reader_v0_1 = corpus.Law(fixture), corpus.Law(v0_1)
    alleles = ref.pool(fixture)
    founders = [bytes.fromhex(entry["genome"]) for entry in _golden()["fixture"]["found"]]
    founders += [ref.genome.found(seed, fixture_view, alleles)[0] for seed in corpus.seeds(ref.rng, "test.gn3", 8)]
    items = []
    for label, data in corpus.targeted(founders[0], reader_fixture):
        items.append((label, data, "fixture"))
    for index, data in corpus.mutations(ref.rng, founders, reader_fixture, 1000, "test.gn3"):
        items.append((f"mutation {index}", data, "fixture"))
    for k in range(4):
        items.append(("corner", ref.genome.corner(k, fixture_view)[0], "fixture"))
    corner_hi = ref.genome.corner(1, v0_1_view)[0]
    for label, data in corpus.v0_1_cases(corner_hi, reader_v0_1):
        items.append((label, data, "v0_1"))
    items.append(("size", bytes(reader_v0_1.max_bytes + 1), "v0_1"))
    items.append(("corner", ref.genome.corner(0, v0_1_view)[0], "v0_1"))
    views = {"fixture": (fixture_view, reader_fixture), "v0_1": (v0_1_view, reader_v0_1)}
    return items, views, founders


def test_gn3_the_decoder_accepts_inside_the_box_or_refuses_by_name_and_never_raises(ref):
    items, views, founders = _gn3_items(ref)
    seen = {}
    accepted = 0
    for index, (label, data, law_name) in enumerate(items):
        lawview, reader = views[law_name]
        try:
            ref.genome.decode(data, lawview)
        except ref.wire.Refused as refusal:
            assert refusal.code in corpus.CODES, (index, label, refusal.code, data.hex())
            klass = corpus.detail_class(refusal.detail)
            assert klass is not None, (index, label, refusal.detail, data.hex())
            seen[klass] = seen.get(klass, 0) + 1
            if label in corpus.DETAIL_CLASSES:
                assert klass == label, (index, label, refusal.detail)
            continue
        except Exception as exc:  # noqa: BLE001 - the property is that nothing else escapes
            pytest.fail(f"item {index} ({label}) raised {exc!r}: {data.hex()}")
        verdict = corpus.read(data, reader)
        assert not isinstance(verdict, str), f"item {index} ({label}) accepted but unsound: {verdict}: {data.hex()}"
        accepted += 1
        assert label not in corpus.DETAIL_CLASSES, f"{label} was accepted"
    requests = corpus.requests(founders[0].hex(), views["fixture"][1].max_bytes)
    for expected, request in requests:
        answer = ref.ask(request)
        if expected is None:
            assert "refused" not in answer, (request, answer)
            continue
        assert answer.get("refused") in corpus.CODES, (request, answer)
        klass = corpus.detail_class(answer["detail"])
        assert klass == expected, (request, answer)
        seen[klass] = seen.get(klass, 0) + 1
    missing = [name for name in corpus.DETAIL_CLASSES if name not in seen]
    assert missing == [], f"detail classes never produced: {missing}"
    for name in ("fields", "law", "genome hex", "seed", "corner"):
        assert seen.get("request " + name, 0) >= 1, name
    assert accepted >= 100, accepted


# ---------------------------------------------------------------------------
# GN4 -- recompilation
# ---------------------------------------------------------------------------
def test_gn4_recompiling_reproduces_the_tables_byte_for_byte(ref):
    golden = _golden()
    multi = 0
    for law_name in ("fixture", "v0_1"):
        law, lawview = ref.law(law_name)
        alleles = ref.pool(law)
        for entry in golden[law_name]["found"]:
            data, _chosen, _work = ref.genome.found(bytes.fromhex(entry["seed"]), lawview, alleles)
            first = ref.wire.emit(ref.tables(data, law, lawview))
            second = ref.wire.emit(ref.tables(data, law, lawview))
            assert first == second, "two compilations of one genome differ"
            assert hashlib.sha256(first).hexdigest() == entry["tables_sha256"]
            tables = ref.wire.parse(first)
            starts = ref.column(tables, "genes", "edge_start")
            cis = ref.column(tables, "edges", "cis")
            for row in range(len(starts) - 1):
                edges = cis[starts[row]:starts[row + 1]]
                assert edges == sorted(edges) and len(set(edges)) == len(edges), (law_name, row, edges)
                multi += len(edges) >= 2
    assert multi >= 10, f"only {multi} genes carry two edges or more"


# ---------------------------------------------------------------------------
# GN5 -- stream isolation
# ---------------------------------------------------------------------------
def _by_locus(data):
    pairs, chrom = corpus.split(data)
    out = {}
    for index, records in enumerate(chrom):
        for record in records:
            out[(corpus.locus_of(record), index % 2)] = record
    return out


def test_gn5_each_locus_draws_from_its_own_stream(ref):
    g = ref.genome
    law, lawview = ref.law("fixture")
    pool = ref.lawfiles.founders("fixture")
    extra = copy.deepcopy(pool)
    target = 2
    entry = next(e for e in extra["loci"] if e["locus"] == target)
    entry["alleles"].append({"body": [14, 49152, 4200, -8192, 16384], "freq": 4, "name": "extra",
                             "window": [0, 0, 0, 0, 0]})
    assert g.validate_pool(law, extra) == []
    original, widened = g.pool_alleles(pool), g.pool_alleles(extra)
    moved = 0
    for seed in corpus.seeds(ref.rng, "test.gn5", 20):
        before = _by_locus(g.found(seed, lawview, original)[0])
        after = _by_locus(g.found(seed, lawview, widened)[0])
        for key in before:
            if key[0] != target:
                assert before[key] == after[key], f"locus {key} moved when only locus {target} changed"
        moved += sum(before[(target, h)] != after[(target, h)] for h in (0, 1))
    assert moved >= 1, "the extra allele was never drawn"

    reversed_law = copy.deepcopy(law)
    reversed_law["genome"]["chromosomes"] = [list(reversed(ids)) for ids in law["genome"]["chromosomes"]]
    assert g.validate_law(reversed_law) == []
    reversed_view = g.view(reversed_law)
    for seed in corpus.seeds(ref.rng, "test.gn5", 5):
        forward = _by_locus(g.found(seed, lawview, original)[0])
        backward = _by_locus(g.found(seed, reversed_view, original)[0])
        assert forward == backward, "the template order moved a locus's draws"

    for k in range(2, 50):
        corner = g.CountingStream(bytes(32), g.DOMAIN_CORNER, k).next_u64()
        founder = g.CountingStream(bytes(32), g.DOMAIN_FOUNDER, k).next_u64()
        assert corner != founder, k


# ---------------------------------------------------------------------------
# GN7 -- the recorded bound analysis
# ---------------------------------------------------------------------------
def test_gn7_the_recorded_bounds_equal_the_recomputed_ones_within_their_ceilings(ref):
    b = ref.bounds
    for name in ref.lawfiles.LAWS:
        law = ref.lawfiles.law(name)
        computed = b.compute(law)
        assert law["genome_bounds"] == computed["genome_bounds"], f"{name}: genome_bounds not re-recorded"
        assert law["stock_bounds"] == computed["stock_bounds"], f"{name}: stock_bounds not re-recorded"
        assert b.defects(law) == []
        max_records = law["genome"]["max_records"]
        assert len(computed["genome_bounds"]) >= 11 and len(computed["stock_bounds"]) >= 4
        for bound, entry in computed["genome_bounds"].items():
            assert b.within(entry["ceiling"], entry["max_raw"], max_records), (name, bound, entry)
            for text in (entry["premise"], entry["proof"]):
                assert text and all(0x20 <= ord(c) <= 0x7E for c in text), (name, bound, text)
        for entry in computed["stock_bounds"].values():
            assert all(0x20 <= ord(c) <= 0x7E for c in entry["proof"] + entry["premise"] + entry["owed"])
    narrowed = ref.lawfiles.law("fixture")
    for kind in narrowed["genome"]["kinds"]:
        if kind["name"] == "cis":
            for field in kind["fields"]:
                if field["name"] == "w":
                    field["lo"], field["hi"] = -16384, 16383
    assert b.compute(narrowed) != b.compute(ref.lawfiles.law("fixture")), "the analysis reads the boxes"
    assert b.defects(narrowed) != [], "a box changed without re-recording is named"


# ---------------------------------------------------------------------------
# GN8 -- corners keep the primitives in their domain
# ---------------------------------------------------------------------------
def _drive(ref, tables, bounds_of, work):
    fx = ref.fx
    col = ref.column
    starts = col(tables, "genes", "edge_start")
    kinds = col(tables, "genes", "kind")
    ks, ns, ws, modes = (col(tables, "edges", name) for name in ("k", "n", "w", "mode"))
    tf_gene = col(tables, "tf", "gene")
    tf = {name: col(tables, "tf", name) for name in ("bias", "prod", "rate", "out", "deg")}
    edges_driven = 0
    for row in range(len(kinds)):
        lows, highs = [], []
        for e in range(starts[row], starts[row + 1]):
            assert fx.hill_up(ONE, ks[e], ns[e], work) > 0, "a Hill constant collapsed to a step"
            terms = []
            for c in (0, 1, ONE, CMAX):
                act = fx.hill_up(c, ks[e], ns[e], work) if modes[e] == 0 else fx.hill_down(c, ks[e], ns[e], work)
                terms.append(fx.mul(ws[e], act, work))
            lows.append(min(terms))
            highs.append(max(terms))
            edges_driven += 1
        bias = 0
        if row in tf_gene:
            bias = tf["bias"][tf_gene.index(row)]
        for a_g in (bias + sum(lows), bias + sum(highs)):
            assert abs(a_g) <= bounds_of["activation_max"]["max_raw"], (row, a_g)
            fx.sig(a_g, work)
    for index in range(len(tf_gene)):
        for x in (0, ONE):
            fx.mul(tf["prod"][index], x, work)
        for delta in (-ONE, ONE):
            fx.mul(tf["rate"][index], delta, work)
    deg = col(tables, "species", "deg")
    for code, value in enumerate(deg):
        for c in (0, CMAX):
            fx.mul(value, c, work)
        produced = sum(fx.mul(tf["prod"][i], ONE, work) for i in range(len(tf_gene)) if tf["out"][i] == code)
        assert produced <= I32_MAX, code
    for gain in col(tables, "rec", "gain"):
        for u in (0, ONE):
            fx.mul(gain, u, work)
    for kcat, km, yld in zip(col(tables, "enz", "kcat"), col(tables, "enz", "km"), col(tables, "enz", "yield")):
        for c in (0, CMAX):
            rate = fx.mm(c, km, work)
            fx.mul(kcat, rate, work)
            fx.mul(yld, rate, work)
    return edges_driven


def test_gn8_at_the_box_corners_every_genome_factor_keeps_the_primitives_in_their_domain(ref):
    fx = ref.fx
    work = fx.Work()
    driven = 0
    cases = []
    fixture, fixture_view = ref.law("fixture")
    v0_1, v0_1_view = ref.law("v0_1")
    for k in range(34):
        cases.append((fixture, fixture_view, ref.genome.corner(k, fixture_view)[0]))
    for k in range(4):
        cases.append((v0_1, v0_1_view, ref.genome.corner(k, v0_1_view)[0]))
    union = dict(corpus.v0_1_cases(ref.genome.corner(1, v0_1_view)[0], corpus.Law(v0_1)))["union"]
    cases.append((v0_1, v0_1_view, union))
    widest = 0
    for law, lawview, data in cases:
        tables = ref.tables(data, law, lawview)
        driven += _drive(ref, tables, law["genome_bounds"], work)
        starts = ref.column(tables, "genes", "edge_start")
        widest = max(widest, max(b - a for a, b in zip(starts, starts[1:])))
    assert work.alarm == 0, f"a corner raised {work.alarm} fixed-point alarms"
    assert widest == v0_1["genome_bounds"]["edges_per_gene_max"]["max_raw"], widest
    assert driven >= 200, driven
    probe = fx.Work()
    fx.hill_up(ONE, 0, 4, probe)
    assert probe.alarm >= 1, "the alarm can fire"


# ---------------------------------------------------------------------------
# GN9 -- dominance
# ---------------------------------------------------------------------------
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


def _reflag(data, locus, flags):
    """The records of ``locus`` carry the flags a law copy gives the locus."""
    pairs, chrom = corpus.split(data)
    for index, records in enumerate(chrom):
        chrom[index] = [r[:1] + bytes([flags]) + r[2:] if corpus.locus_of(r) == locus else r for r in records]
    return corpus.frame(chrom, pairs)


def _drop(data, chromosome, locus):
    pairs, chrom = corpus.split(data)
    chrom[chromosome] = [r for r in chrom[chromosome] if corpus.locus_of(r) != locus]
    return corpus.frame(chrom, pairs)


def _swap(data):
    pairs, chrom = corpus.split(data)
    return corpus.frame([chrom[1], chrom[0]], pairs)


def _widen(law, locus, field, box=None, flags=None):
    law = copy.deepcopy(law)
    entry = next(e for e in law["genome"]["loci"] if e["id"] == locus)
    if flags is not None:
        entry["flags"] = flags
    if box is not None:
        entry["box"] = [o for o in entry["box"] if o["field"] != field]
        if box != "kind":
            order = [f["name"] for f in next(k for k in law["genome"]["kinds"] if k["name"] == entry["kind"])["fields"]]
            entry["box"].append({"field": field, "hi": box[1], "lo": box[0]})
            entry["box"].sort(key=lambda o: order.index(o["field"]))
    return law


def _row(ref, tables, table, locus, *path):
    genes = ref.column(tables, "genes", "locus")
    row = genes.index(locus)
    rows = ref.column(tables, *(path or (table,)), "gene")
    return rows.index(row) if row in rows else None


def test_gn9_dominance_follows_each_mode_and_ignores_which_homolog_carries_what(ref):
    g = ref.genome
    law, lawview = ref.law("fixture")
    base = bytes.fromhex(_golden()["fixture"]["found"][0]["genome"])

    def compiled(data, law_value=law):
        assert g.validate_law(law_value) == []
        return ref.tables(data, law_value, g.view(law_value))

    def content(tables):
        # The tables carry the digest of the genome bytes, which a swap changes by definition.
        return ref.wire.emit({key: value for key, value in tables.items() if key != "genome"})

    def both(data, law_value=law):
        first = compiled(data, law_value)
        assert content(compiled(_swap(data), law_value)) == content(first), "homolog order matters"
        return first

    # DOM_MAX: the larger (strength, body bytes) key wins.
    tables = both(_edit(_edit(base, 0, 7, kcat=16384), 1, 7, kcat=20000))
    assert ref.column(tables, "enz", "kcat")[_row(ref, tables, "enz", 7)] == 20000

    wide = _widen(law, 1, "w", box="kind")
    tables = both(_edit(_edit(base, 0, 1, w=-32768), 1, 1, w=32767), wide)
    assert ref.column(tables, "edges", "w")[0] == -32768 * 16, "a signed minimum outranks the maximum"
    tables = both(_edit(_edit(base, 0, 1, w=4096), 1, 1, w=-4096), wide)
    assert ref.column(tables, "edges", "w")[0] == -4096 * 16, "a strength tie is broken by the body bytes"

    dom_plast = _widen(law, 11, "A", flags=0x04)
    tables = both(_reflag(_edit(_edit(base, 0, 11, A=-128), 1, 11, A=127), 11, 0x04), dom_plast)
    assert ref.column(tables, "reserved", "plast", "fields")[1] == -128

    # LEX enumerations are keyed by a hash of the body, whichever homolog holds it.
    pairs, chrom = corpus.split(base)
    lex_a = [r for r in chrom[0] if corpus.locus_of(r) == 23][0]
    lex_b = corpus.with_field(lex_a, "d0", 3 - corpus.field_of(lex_a, "d0"))  # onset 1 <-> 2
    assert lex_a != lex_b
    key = {r: (int.from_bytes(hashlib.sha256(r[6:]).digest()[:4], "big"), r[6:]) for r in (lex_a, lex_b)}
    winner = max((lex_a, lex_b), key=lambda r: key[r])
    data = corpus.frame([[lex_a if corpus.locus_of(r) == 23 else r for r in chrom[0]],
                         [lex_b if corpus.locus_of(r) == 23 else r for r in chrom[1]]], pairs)
    tables = both(data)
    rows = ref.column(tables, "reserved", "lex", "gene")
    fields = ref.column(tables, "reserved", "lex", "fields")
    lex_row = rows.index(ref.column(tables, "genes", "locus").index(23))
    assert fields[10 * lex_row + 1:10 * lex_row + 3] == [corpus.field_of(winner, "d0"), corpus.field_of(winner, "d1")]

    # ADD floors the mean.
    biased = _widen(law, 2, "bias", box="kind")
    tables = both(_edit(_edit(base, 0, 2, bias=-3), 1, 2, bias=0), biased)
    assert ref.column(tables, "tf", "bias")[_row(ref, tables, "tf", 2)] == -2 * 16
    tables = both(_edit(_edit(base, 0, 11, A=-128), 1, 11, A=127))
    assert ref.column(tables, "reserved", "plast", "fields")[1] == -1

    # Dose 1 keeps the allele as it is.
    single = _drop(_edit(base, 0, 17, activity=700), 1, 17)
    tables = compiled(single)
    row = ref.column(tables, "genes", "locus").index(17)
    assert ref.column(tables, "genes", "dose")[row] == 1
    assert ref.column(tables, "te", "activity")[ref.column(tables, "te", "gene").index(row)] == 700

    # LOAD_REC: a load row only when both homologs carry the load.
    tables = both(_edit(_edit(base, 0, 18, effect=0), 1, 18, effect=16384))
    assert ref.column(tables, "loads", "gene") == [], "a heterozygous load is recessive"
    tables = both(_edit(_edit(base, 0, 18, effect=16384), 1, 18, effect=16385))
    assert ref.column(tables, "loads", "effect") == [16384]
    tables = compiled(_drop(_edit(base, 0, 18, effect=16384), 1, 18))
    assert ref.column(tables, "loads", "gene") == [], "a load at dose 1 is not expressed"

    # One edge combines the CIS occurrences of both homologs.
    tables = both(_edit(_edit(base, 0, 1, n=3), 1, 1, n=4))
    starts = ref.column(tables, "genes", "edge_start")
    row = ref.column(tables, "genes", "locus").index(2)
    assert starts[row + 1] - starts[row] == 1
    assert ref.column(tables, "edges", "n")[starts[row]] == 4

    # Species decay takes the largest deg among the TFs writing one species.
    shared = _widen(law, 4, "out", box=(14, 14))
    data = _edit(_edit(base, 0, 4, out=14, deg=4600), 1, 4, out=14, deg=4600)
    data = _edit(_edit(data, 0, 2, deg=3700), 1, 2, deg=3700)
    tables = both(data, shared)
    species_deg = ref.column(tables, "species", "deg")
    assert species_deg[14] == 4600 and species_deg[15] == shared["genome"]["species"][15]["deg"]

    for entry in _golden()["fixture"]["found"]:
        data = bytes.fromhex(entry["genome"])
        assert content(compiled(_swap(data))) == content(compiled(data))


# ---------------------------------------------------------------------------
# GN10 -- one source per law and pool
# ---------------------------------------------------------------------------
def _load_script(name, module_name):
    path = SCRIPTS / name
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


def test_gn10_each_law_and_its_pool_have_one_source_pinned_by_digest():
    loaded, restore = open_allium(native=True)
    try:
        ref = Ref(loaded)
        g = ref.genome
        for name in ref.lawfiles.LAWS:
            law = ref.lawfiles.law(name)
            pool = ref.lawfiles.founders(law["founders"]["name"])
            assert g.validate_law(law) == [], name
            assert g.validate_pool(law, pool) == [], name
            assert law["founders"]["sha256"] == ref.lawfiles.digest(pool), f"{name} pins another pool"
        author = _load_script("allium_author_genome.py", "_gn10_author")
        files = author.author()
        assert sorted(files) == ["fixture.json", "founders_fixture.json", "founders_v1.json", "v0_1.json"]
        for file_name, text in files.items():
            assert LAWS_DIR.joinpath(file_name).read_text(encoding="ascii") == text, \
                f"{file_name} differs from what the authoring script writes"
        native = native_module(loaded)
        mine = json.loads(ref.protocol.engine_info())
        theirs = json.loads(bytes(native.allium_engine()))
        for key in ("domains", "founders", "genome_schema", "laws"):
            assert theirs[key] == mine[key], key
        assert len(mine["founders"]) == 2 and len(mine["laws"]) == 2

        law = ref.lawfiles.law("fixture")
        pool = ref.lawfiles.founders("fixture")

        def law_defects(edit):
            copy_of = copy.deepcopy(law)
            edit(copy_of["genome"])
            return g.validate_law(copy_of)

        def pool_defects(edit):
            copy_of = copy.deepcopy(pool)
            edit(copy_of)
            return g.validate_pool(law, copy_of)

        def kind_box(genome):
            genome["kinds"][0]["fields"][1]["hi"] = 70000

        def free_out(genome):
            locus = genome["loci"][2]
            locus["flags"] = 0x0C
            locus["box"] = [o for o in locus["box"] if o["field"] != "out"]

        def conserved_rec(genome):
            next(o for o in genome["loci"][0]["box"] if o["field"] == "species").update(lo=2, hi=2)

        def token_src(genome):
            next(o for o in genome["loci"][1]["box"] if o["field"] == "src").update(lo=1, hi=1)

        def bool_freq(value):
            value["loci"][0]["alleles"][0]["freq"] = True

        def short_body(value):
            value["loci"][0]["alleles"][0]["body"].pop()

        def twin_alleles(value):
            alleles = value["loci"][0]["alleles"]
            alleles[1] = dict(alleles[0], name="twin")

        expected = (
            (law_defects(kind_box), "type range"),
            (law_defects(free_out), "must be fixed"),
            (law_defects(conserved_rec), "forbidden class"),
            (law_defects(token_src), "token"),
            (pool_defects(bool_freq), "freq"),
            (pool_defects(short_body), "length"),
            (pool_defects(twin_alleles), "identical"),
        )
        for defects, word in expected:
            assert any(word in defect for defect in defects), (word, defects)
    finally:
        restore()


# ---------------------------------------------------------------------------
# GN11 -- the greenhouse
# ---------------------------------------------------------------------------
def _firewall():
    for module in list(sys.modules.values()):
        wall = getattr(module, "_FIREWALL", None)
        if wall is not None and hasattr(wall, "redirected"):
            return wall
    return None


def test_gn11_the_greenhouse_draws_at_the_pools_frequencies_and_writes_nothing(ref, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    greenhouse = _load_script("allium_greenhouse.py", "_gn11_greenhouse")
    root = bytes(range(32))
    report = greenhouse.report("fixture", 400, root)
    assert report["count"] == 400 and report["source"] == "static"
    assert report["distinct_genomes"] == 400
    draws = 800
    checked = 0
    for entry in report["loci"]:
        expected, observed = entry["expected"], entry["observed"]
        total = sum(expected)
        assert sum(observed) == draws, entry["name"]
        for freq, count in zip(expected, observed):
            assert count >= 1, (entry["name"], expected, observed)
            assert (count * total - draws * freq) ** 2 <= 16 * draws * freq * (total - freq), \
                (entry["name"], expected, observed)
            checked += 1
    assert checked >= 58, checked
    small = [ref.wire.emit(greenhouse.report("fixture", 16, root)) for _ in range(2)]
    assert small[0] == small[1], "the same cohort twice is the same report"
    pool = ref.lawfiles.founders("fixture")
    law = ref.lawfiles.law("fixture")
    lawview = ref.genome.view(law)
    for entry in pool["loci"]:
        locus = lawview.loci[entry["locus"]]
        first = entry["alleles"][0]["body"]
        second = entry["alleles"][1]["body"]
        width = len(first)
        entry["alleles"] = [
            {"body": list(first), "freq": 999, "name": "common", "window": [0] * width},
            {"body": list(second), "freq": 1, "name": "rare", "window": [0] * width},
        ]
        assert len(locus.boxes) == width
    narrow = greenhouse.report("fixture", 16, root, pool=pool)
    assert narrow["distinct_genomes"] < 16, "the count of distinct genomes can come out below the cohort"
    assert list(tmp_path.iterdir()) == [], "the greenhouse wrote nothing where it ran"
    wall = _firewall()
    assert wall is not None, "the data firewall is installed"
    assert not wall.redirected.get(wall.current), "the greenhouse reached for a data place"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
