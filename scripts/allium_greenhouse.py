#!/usr/bin/env python3
"""The greenhouse: a cohort of founder genomes, counted, never lived.

``python3 scripts/allium_greenhouse.py --law fixture --count 64 [--root HEX] [--native]``

Founds ``count`` genomes from seeds keyed on one root, decodes and compiles
each, and prints one canonical JSON object: how often each allele was
drawn against the pool's frequencies, the spread of a few parameters, the
colour classes the pigment loci allow, and how many of the genomes are
distinct. It simulates no day; it writes nothing anywhere; its numbers are
``"source": "static"`` -- properties of the law and the pool, not of a life.

``--native`` asks the native engine through the engine seam instead of
calling the reference organs.
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opti_oignon.allium import lawfiles, rng, wire  # noqa: E402
from opti_oignon.allium.ref.organs import compile as organ_compile  # noqa: E402
from opti_oignon.allium.ref.organs import genome as g  # noqa: E402

COHORT_DOMAIN = "greenhouse.cohort"
CLOCK_SPECIES = (14, 15, 16)
PHOTOPERIOD_CHANNEL = 10
COLOURS = ("golden", "red", "shallot", "white")


def _column(tables, table, name):
    return wire.unpack_bulk(tables[table][name])[1]


def _spread(values):
    return {"max": max(values), "min": min(values), "sum": sum(values)}


def _colour(alleles):
    """A provisional reading of the four pigment loci: I, C, R, G."""
    if alleles.get(0, 0) == 1:
        return "white"
    if alleles.get(1, 0) == 0:
        return "white"
    if alleles.get(2, 0) == 1:
        return "red"
    if alleles.get(3, 0) == 1:
        return "golden"
    return "shallot"


def _found_reference(seed, law, lawview, alleles, digest):
    data, chosen, _work = g.found(seed, lawview, alleles)
    tables, work = organ_compile.compile_genome(data, law, lawview, digest)
    return data, chosen, tables, work


def _found_native(seed, law_name):
    from opti_oignon.allium import engine

    def ask(request):
        answer = wire.parse(engine.call(wire.emit(dict(request, v=1))))
        if "refused" in answer:
            raise RuntimeError(f"the engine refused: {answer}")
        return answer

    founder = ask({"law": law_name, "op": "genome_found", "seed": seed.hex()})
    compiled = ask({"genome": founder["genome"], "law": law_name, "op": "genome_compile"})
    chosen = wire.unpack_bulk(founder["alleles"])[1]
    return bytes.fromhex(founder["genome"]), chosen, compiled["tables"], compiled["work"]


def report(law_name, count, root, pool=None, native=False):
    """The cohort report as a value; ``pool`` replaces the law's pool (reference path only)."""
    law = lawfiles.law(law_name)
    defects = g.validate_law(law)
    if defects:
        raise ValueError(f"law {law_name}: {defects[:3]}")
    lawview = g.view(law)
    if pool is None:
        pool = lawfiles.founders(law["founders"]["name"])
    defects = g.validate_pool(law, pool)
    if defects:
        raise ValueError(f"pool: {defects[:3]}")
    alleles = g.pool_alleles(pool)
    digest = lawfiles.digest(law)
    record_loci = []
    for pair in range(lawview.pairs):
        for _h in range(g.PLOIDY):
            record_loci.extend(lawview.chromosomes[pair])
    observed = {ident: [0] * len(alleles[ident]) for ident in lawview.order}
    genomes = {}
    clock, photoperiod, vu_req, works = [], [], [], []
    colours = {name: 0 for name in COLOURS}
    size = 0
    for i in range(count):
        seed = rng.key(root, COHORT_DOMAIN, (i,))
        if native:
            data, chosen, tables, work = _found_native(seed, law_name)
        else:
            data, chosen, tables, work = _found_reference(seed, law, lawview, alleles, digest)
        size = len(data)
        genomes[hashlib.sha256(data).hexdigest()] = True
        works.append(work)
        for ident, index in zip(record_loci, chosen):
            observed[ident][index] += 1
        outs = _column(tables, "tf", "out")
        for out, deg in zip(outs, _column(tables, "tf", "deg")):
            if out in CLOCK_SPECIES:
                clock.append(deg)
        for channel, threshold in zip(_column(tables, "rec", "channel"), _column(tables, "rec", "threshold")):
            if channel == PHOTOPERIOD_CHANNEL:
                photoperiod.append(threshold)
        vern = _column(tables["reserved"], "vern", "fields")
        if vern:
            vu_req.append(vern[3])
        pig = dict(zip(_column(tables, "pig", "class"), _column(tables, "pig", "allele")))
        colours[_colour(pig)] += 1
    out = {
        "clock_deg": _spread(clock),
        "colour_potential": colours,
        "compile_work": {"max": max(works), "min": min(works)},
        "count": count,
        "distinct_genomes": len(genomes),
        "genome_bytes": size,
        "law": law_name,
        "law_sha256": digest,
        "loci": [{"expected": [a["freq"] for a in alleles[ident]], "locus": ident,
                  "name": lawview.loci[ident].name, "observed": observed[ident]}
                 for ident in lawview.order],
        "root": bytes(root).hex(),
        "source": "static",
        "vu_req": _spread(vu_req),
    }
    if photoperiod:
        out["photoperiod"] = _spread(photoperiod)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--law", default="fixture", choices=lawfiles.LAWS)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--root", default=bytes(range(32)).hex(), help="64 lowercase hex characters")
    parser.add_argument("--native", action="store_true", help="ask the native engine through the seam")
    args = parser.parse_args(argv)
    if len(args.root) != 64:
        parser.error("--root takes 64 hex characters")
    if args.count < 1:
        parser.error("--count takes a positive number")
    value = report(args.law, args.count, bytes.fromhex(args.root), native=args.native)
    sys.stdout.write(wire.emit(value).decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
