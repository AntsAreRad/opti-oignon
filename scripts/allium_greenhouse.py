#!/usr/bin/env python3
"""The greenhouse: a cohort of founder genomes, counted, never lived.

``python3 scripts/allium_greenhouse.py --law fixture --count 64 [--root HEX] [--native]``

Founds ``count`` genomes from seeds keyed on one root, decodes and compiles
each, and prints one canonical JSON object: how often each allele was
drawn against the pool's frequencies, the spread of a few parameters, the
colour classes the pigment loci allow, and how many of the genomes are
distinct, the body levels and the adult beard and moustache each genome
allows. It
simulates no day; it writes nothing anywhere; its numbers are
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

# The beard's SHAPE trait codes (the full law's layout), and how they read.
MEL_TRAITS = (12, 13, 14, 16)
T_MC1R, T_ONSET, T_SPAN, T_CEILING, T_CREAM = 17, 18, 19, 20, 21
T_MOUSTACHE, T_FORM, T_SIZE, T_THICK = 22, 23, 24, 25
MEL_CUTS = (30, 58, 86, 114, 142, 170, 198)
BEARD_CLASSES = ("platinum", "cream", "golden", "honey", "light_chestnut", "chestnut", "brown", "dark_brown")
RED_SHADES = ("venetian", "copper", "red", "auburn")
FORMS = ("brush", "straight", "droopy", "curled")

# The body's continuous SHAPE trait codes, and how they read. A value is first
# rounded to its genotype step s = (value + 2048) >> 12: every allele is a
# multiple of 8192, so a genotype compiles to 4096 * (ka + kb) give or take the
# founder window, and s is ka + kb under any window below 2048. The cuts count
# on s, as MEL_CUTS count on the summed melanin; in value terms a cut c sits at
# 4096 * c - 2048, halfway between two steps.
T_RADIUS, T_HEIGHT, T_NECK, T_STIFFNESS = 0, 1, 3, 5
T_BREATH, T_TURN, T_STRIPE, T_SPECKLE = 6, 7, 10, 11
HW_CUTS = (7, 8, 9, 10)               # bulb half-width 5 + count
HB_CUTS = (5, 6, 7, 8, 9)             # bulb rows 8 + count
HH_CUTS = (7, 9, 11)                  # hat rows 7 + count
STIFFNESS_CUTS = (5, 6, 7, 8, 9, 10)  # stiffness 0..6; the tip bends 6 - stiffness
EYE_CUT = 9                           # one wide allele sets the eyes 6 px apart
TURN_CUTS = (7, 9, 11)                # left and gaze, left, right, right and gaze
STRIPE_CUT = 10                       # strong skin lines need two striped alleles
FLECK_CUTS = (7, 10)                  # 0, 1 or 2 flecks
DRY_CUT = 9                           # a dry hat tip
TURNS = ((-1, -1), (-1, 0), (1, 0), (1, 1))                 # (side, gaze) by turn class
SPECKLES = ((0, False), (1, False), (1, True), (2, True))   # (flecks, dry) by speckle class
BODY_DEFAULTS = {"dry": False, "eye": 2, "flop": 3, "gaze": 0, "hb": 10, "hh": 9, "hw": 7, "side": 1,
                 "speck": 0, "strong": False}
BODY_CLASSES = {"eye": 2, "flop": 7, "hb": 6, "hh": 4, "hw": 5, "speckle": 4, "stripe": 2, "turn": 4}


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


def shape_traits(tables):
    """The compiled SHAPE rows as ``{trait: value}``."""
    rows = _column(tables["reserved"], "shape", "fields")
    return {rows[k]: rows[k + 1] for k in range(0, len(rows), 3)}


def _step(value):
    return (value + 2048) >> 12


def _count(cuts, step):
    return sum(1 for cut in cuts if step >= cut)


def body_levels(shape):
    """The body's genetic display levels, each from its own compiled value by absolute cuts.

    No level depends on another genome: the sketch that ranked founders against
    each other is gone. A missing locus reads as the stated default -- the
    modal size, close-set eyes, leaning right without a gaze shift -- and never
    as a mark: no strong lines, no fleck, no dry tip.
    """
    out = dict(BODY_DEFAULTS)
    if T_RADIUS in shape:
        out["hw"] = 5 + _count(HW_CUTS, _step(shape[T_RADIUS]))
    if T_HEIGHT in shape:
        out["hb"] = 8 + _count(HB_CUTS, _step(shape[T_HEIGHT]))
    if T_NECK in shape:
        out["hh"] = 7 + _count(HH_CUTS, _step(shape[T_NECK]))
    if T_STIFFNESS in shape:
        out["flop"] = 6 - _count(STIFFNESS_CUTS, _step(shape[T_STIFFNESS]))
    if T_BREATH in shape:
        out["eye"] = 3 if _step(shape[T_BREATH]) >= EYE_CUT else 2
    if T_TURN in shape:
        out["side"], out["gaze"] = TURNS[_count(TURN_CUTS, _step(shape[T_TURN]))]
    if T_STRIPE in shape:
        out["strong"] = _step(shape[T_STRIPE]) >= STRIPE_CUT
    if T_SPECKLE in shape:
        step = _step(shape[T_SPECKLE])
        out["speck"] = _count(FLECK_CUTS, step)
        out["dry"] = step >= DRY_CUT
    return out


def body_classes(levels):
    """Each body trait's class index, for counting."""
    return {"eye": levels["eye"] - 2, "flop": levels["flop"], "hb": levels["hb"] - 8, "hh": levels["hh"] - 7,
            "hw": levels["hw"] - 5, "speckle": SPECKLES.index((levels["speck"], levels["dry"])),
            "stripe": 1 if levels["strong"] else 0, "turn": TURNS.index((levels["side"], levels["gaze"]))}


def beard_potential(shape):
    """The adult beard and moustache a genome allows, before any day is lived.

    A missing locus never invents a colour: without all four melanin loci,
    or with the dominant cream allele (mean 1 or 2), the beard keeps the
    cream every young gnome is born with; a missing mc1r reads as the
    functional genotype (2), never as red. ``colour`` names what an adult
    shows with no grey yet; ``glints`` marks the red carrier.
    """
    mel = [shape.get(trait) for trait in MEL_TRAITS]
    melanin = None if None in mel else sum(mel) >> 8
    mc1r = shape.get(T_MC1R)
    mc1r = 2 if mc1r is None else mc1r
    cream = (shape.get(T_CREAM) or 0) >= 1
    level = 1 if cream or melanin is None else sum(1 for cut in MEL_CUTS if melanin >= cut)
    colour = RED_SHADES[level >> 1] if mc1r == 0 else BEARD_CLASSES[level]
    grey = [shape.get(trait) for trait in (T_ONSET, T_SPAN, T_CEILING)]
    moustache = None
    if (shape.get(T_MOUSTACHE) or 0) >= 1 and None not in (shape.get(T_FORM), shape.get(T_SIZE)):
        moustache = {"form": FORMS[shape[T_FORM]], "size": shape[T_SIZE], "thick": shape.get(T_THICK) == 1}
    return {"colour": colour, "cream": cream, "glints": mc1r == 1, "grey": None if None in grey else grey,
            "level": level, "mc1r": mc1r, "melanin": melanin, "moustache": moustache}


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
    beards = {name: 0 for name in BEARD_CLASSES + RED_SHADES}
    glints = 0
    moustaches = {name: 0 for name in ("none",) + FORMS}
    sizes = {1: 0, 2: 0, 3: 0}
    thick = 0
    greys = []
    body = {name: [0] * size for name, size in BODY_CLASSES.items()}
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
        shape = shape_traits(tables)
        for name, index in body_classes(body_levels(shape)).items():
            body[name][index] += 1
        beard = beard_potential(shape)
        beards[beard["colour"]] += 1
        glints += beard["glints"]
        moustache = beard["moustache"]
        moustaches["none" if moustache is None else moustache["form"]] += 1
        if moustache is not None:
            sizes[moustache["size"]] += 1
            thick += moustache["thick"]
        if beard["grey"] is not None:
            greys.append(beard["grey"])
    out = {
        "beard_potential": {"colour": beards, "moustache": moustaches, "red_glints": glints,
                            "sizes": [sizes[1], sizes[2], sizes[3]], "thick": thick},
        "body_potential": body,
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
    if greys:
        out["beard_potential"]["grey"] = {name: _spread([row[k] for row in greys])
                                          for k, name in enumerate(("onset", "span", "ceiling"))}
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
