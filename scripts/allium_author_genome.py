#!/usr/bin/env python3
"""Author the companion's genome laws and founder pools from one table.

Writes, under ``opti_oignon/allium/laws/``:

* ``fixture.json`` -- the small law the t1 contracts run on: one pair, 29 loci
  (a three-node clock, one record of every kind, the whole language block,
  the four colour loci);
* ``v0_1.json`` -- the first provisional law of the full genome: eight
  pairs, 162 loci;
* ``founders_fixture.json`` and ``founders_v1.json`` -- the founder allele
  pools each law pins by digest.

The engines read the written files, never this script. The script is the
authoring tool: a contract runs it in-process and checks that it reproduces
all four files byte for byte, so a hand edit to a law or a pool without the
matching change here is caught.

Usage: ``python3 scripts/allium_author_genome.py [--check]``. ``--check``
writes nothing and exits 1 when a file on disk differs.

The v1 pool follows a rule rather than a list: a base allele ``a`` from the
kind's defaults, and an allele ``b`` that moves one field by an eighth of
its box, with named exceptions for the loci whose alleles mean something
(clock period, photoperiod, colour, resistance, transposons, loads).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAWS_DIR = ROOT.joinpath("opti_oignon", "allium", "laws")
TABLES_DIR = ROOT.joinpath("opti_oignon", "allium", "tables")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opti_oignon.allium import wire  # noqa: E402
from opti_oignon.allium.ref.organs import bounds  # noqa: E402
from opti_oignon.allium.ref.organs import genome as g  # noqa: E402

# ---------------------------------------------------------------------------
# Kind boxes (law data; the layout itself is the engine's)
# ---------------------------------------------------------------------------

I16 = (-32768, 32767)
U16 = (0, 65535)
I8 = (-128, 127)
U8 = (0, 255)

KIND_BOXES = {
    "tf": {"out": (0, 63), "prod": U16, "deg": U16, "bias": I16, "rate": U16},
    "cis": {"src": (0, 63), "w": I16, "K": (512, 65535), "n": (1, 4), "mode": (0, 1)},
    "enz": {"s1": U8, "s2": U8, "p1": U8, "p2": U8, "kcat": U16, "Km": (512, 65535), "yield": U16},
    "rec": {"channel": (0, 10), "species": (0, 63), "gain": I16, "threshold": U16},
    "morph": {"pred": (0, 15), "guard": (0, 63), "threshold": U16, "succ": (0, 15), "rate": U16,
              "angle": I8},
    "imm": {"class": (0, 1), "strength": U16, "cost": U16},
    "pig": {"class": (0, 3), "allele": (0, 1)},
    "plast": {"class": (0, 6), "A": I8, "Aneg": I8, "B": I8, "C": I8, "D": I8, "LR": (0, 15),
              "TAU_X": (1, 15), "TAU_Y": (1, 15), "TAU_E": (1, 15), "M_SEL": (0, 7), "M_SIGN": (0, 1)},
    "temp": {"channel": U8, "weight": I16, "base": I16, "habituation": U8, "curiosity": U16},
    "lex": dict([("offset", (0, 54))] + [(f"d{i}", U8) for i in range(9)]),
    "shape": {"trait": (0, 15), "value": U16, "window": U16},
    "vern": {"D_enter": (7, 21), "v_cold": (1, 2), "v_dorm": (1, 3), "VU_req": (45, 120),
             "stalk_days": (20, 40), "offsets": (2, 4), "rest_days": (30, 90), "a_min": (180, 420)},
    "prc": {"block": (0, 5), "shift0": I16, "shift1": I16, "shift2": I16, "shift3": I16},
    "te": {"activity": U16, "target_bias": U8},
    "load": {"target": (0, 63), "effect": U16},
    "brain": {"param": (0, 63), "value": I16},
}

SPECIES = (
    ("sun", "input"), ("lamp", "token"), ("sugar", "conserved"), ("fructan", "conserved"),
    ("structure", "conserved"), ("water", "conserved"), ("nitrogen", "open"), ("auxin", "open"),
    ("cytokinin", "open"), ("gibberellin", "open"), ("aba", "open"), ("ethylene", "open"),
    ("salicylate", "open"), ("jasmonate", "open"), ("clock_m", "open"), ("clock_d", "open"),
    ("clock_e", "open"), ("bulb_signal", "open"), ("bulb_block", "open"), ("flower_signal", "open"),
    ("cold_repressor", "open"), ("age_timer", "open"), ("dormancy_keeper", "open"),
    ("senescence", "open"), ("ros", "open"), ("damage", "open"), ("sleep_pressure", "open"),
    ("anthocyanin", "open"), ("quercetin", "open"), ("touch", "input"), ("temperature", "input"),
    ("pest_recognition", "input"), ("water_signal", "open"), ("m_rfr", "token"), ("m_blue", "token"),
    ("m_uv", "token"),
) + tuple((f"free_{code}", "free") for code in range(36, 64))

DEFAULT_DEG = 4096

# ---------------------------------------------------------------------------
# Loci
# ---------------------------------------------------------------------------

ALL = 0x7FF
MASKS = {"all": 0x7FF, "live": 0x7FE, "leaf": 0x41C, "sheath": 0x430}

CLOCK_TF = {"prod": (18724, 58256), "deg": (3641, 4681), "bias": -8192, "rate": 16384}
CLOCK_CIS = {"w": 4096, "K": 8192, "n": (3, 4), "mode": 1}
SHIFT_BOX = (-2048, 2048)
PRC_SHIFTS = {"shift0": SHIFT_BOX, "shift1": SHIFT_BOX, "shift2": SHIFT_BOX, "shift3": SHIFT_BOX}

# Per-byte narrowings of the language block, by record offset (d0..d8).
LEX_BOXES = {
    0: {}, 9: {}, 18: {}, 27: {},
    36: {"d6": (1, 255)},
    45: {"d0": (1, 2), "d1": (0, 1), "d2": (0, 127), "d3": (0, 1), "d4": (0, 2), "d5": (0, 2),
         "d6": (0, 1), "d7": (0, 4), "d8": (0, 4)},
    54: {"d0": (0, 1), "d1": (0, 1), "d3": 0, "d4": 0, "d5": 0, "d6": 0, "d7": 0, "d8": 0},
}

SHAPE_TRAITS = ("bulb_radius", "bulb_height", "taper", "neck", "meridians", "leaf_stiffness",
                "breath_base", "turn_speed", "ramp_offset", "leaf_tint", "stripe", "speckle")
SHAPE_VALUE_BOX = {2: (1, 3), 4: (0, 16), 8: (0, 2), 9: (0, 1)}


class L:
    """A law locus: id, name, kind, flags, stage mask, fixed or narrowed fields, founder values."""

    def __init__(self, ident, name, kind, flags, mask=ALL, founder=None, **fields):
        self.ident = ident
        self.name = name
        self.kind = kind
        self.flags = flags
        self.mask = mask
        self.fields = fields
        self.founder = founder or {}


def lex(ident, index, flags):
    offset = 9 * index
    return L(ident, f"lex_{index}", "lex", flags, offset=offset, **LEX_BOXES[offset])


FIXTURE_LOCI = (
    L(0, "rec_clock_light", "rec", 0x04, channel=0, species=14, gain=(256, 512), threshold=0),
    L(1, "cis_clock_m_by_e", "cis", 0x0C, src=16, **CLOCK_CIS),
    L(2, "clock_m", "tf", 0x0D, out=14, **CLOCK_TF),
    L(3, "cis_clock_d_by_m", "cis", 0x0C, src=14, **CLOCK_CIS),
    L(4, "clock_d", "tf", 0x0D, out=15, **CLOCK_TF),
    L(5, "cis_clock_e_by_d", "cis", 0x0C, src=15, **CLOCK_CIS),
    L(6, "clock_e", "tf", 0x0D, out=16, **CLOCK_TF),
    L(7, "enz_photosynthesis", "enz", 0x0C, s1=0, s2=5, p1=2, p2=255),
    L(8, "morph_leaf_init", "morph", 0x0C, MASKS["leaf"], pred=0, guard=2, succ=1),
    L(9, "imm_r0", "imm", 0x18, MASKS["live"], **{"class": 0}),
    L(10, "pig_r", "pig", 0x00, **{"class": 2}),
    L(11, "plast_relay_cortex", "plast", 0x05, M_SEL=0, M_SIGN=0, **{"class": 0}),
    L(12, "temp_curiosity", "temp", 0x11, channel=255, base=0),
    lex(13, 0, 0x05),
    L(14, "shape_bulb_radius", "shape", 0x01, trait=0, window=0),
    L(15, "vern", "vern", 0x05),
    L(16, "prc_5", "prc", 0x01, block=5, **PRC_SHIFTS),
    L(17, "te_0", "te", 0x10),
    L(18, "load_0", "load", 0x02, target=7),
    lex(19, 1, 0x05), lex(20, 2, 0x05), lex(21, 3, 0x05), lex(22, 4, 0x05),
    lex(23, 5, 0x04), lex(24, 6, 0x04),
    L(25, "pig_i", "pig", 0x00, **{"class": 0}),
    L(26, "pig_c", "pig", 0x00, **{"class": 1}),
    L(27, "pig_g", "pig", 0x00, **{"class": 3}),
    L(28, "brain_0", "brain", 0x01, param=0),
)


def _master(ident, target, name, src, mode):
    return L(ident, f"cis_{target}_by_{name}", "cis", 0x0C, founder={"src": src}, mode=mode)


def _v0_1_loci():
    loci = [
        L(0x000, "rec_clock_light", "rec", 0x04, channel=0, species=14, gain=(256, 512), threshold=0),
        L(0x001, "cis_clock_m_by_e", "cis", 0x0C, src=16, **CLOCK_CIS),
        L(0x002, "clock_m", "tf", 0x0D, out=14, **CLOCK_TF),
        L(0x003, "cis_clock_d_by_m", "cis", 0x0C, src=14, **CLOCK_CIS),
        L(0x004, "clock_d", "tf", 0x0D, out=15, **CLOCK_TF),
        L(0x005, "cis_clock_e_by_d", "cis", 0x0C, src=15, **CLOCK_CIS),
        L(0x006, "clock_e", "tf", 0x0D, out=16, **CLOCK_TF),
    ]
    for block in range(6):
        loci.append(L(0x007 + block, f"prc_{block}", "prc", 0x01, block=block, **PRC_SHIFTS))
    loci += [
        _master(0x100, "dormancy_keeper", "aba", 10, 0),
        _master(0x101, "dormancy_keeper", "gibberellin", 9, 1),
        L(0x102, "dormancy_keeper", "tf", 0x0D, out=22),
        _master(0x103, "age_timer", "structure", 4, 0),
        L(0x104, "age_timer", "tf", 0x0D, out=21),
        L(0x105, "rec_photoperiod", "rec", 0x05, channel=10, species=17, threshold=(30037, 40960)),
        _master(0x106, "bulb_signal", "bulb_block", 18, 1),
        L(0x107, "bulb_signal", "tf", 0x0D, out=17),
        _master(0x108, "bulb_block", "gibberellin", 9, 0),
        _master(0x109, "bulb_block", "sugar", 2, 1),
        L(0x10A, "bulb_block", "tf", 0x0D, out=18),
        _master(0x10B, "flower_signal", "age_timer", 21, 0),
        _master(0x10C, "flower_signal", "cold_repressor", 20, 1),
        L(0x10D, "flower_signal", "tf", 0x0D, out=19),
        _master(0x10E, "cold_repressor", "dormancy_keeper", 22, 0),
        _master(0x10F, "cold_repressor", "temperature", 30, 1),
        L(0x110, "cold_repressor", "tf", 0x0D, out=20),
        _master(0x111, "senescence", "age_timer", 21, 0),
        _master(0x112, "senescence", "ethylene", 11, 0),
        _master(0x113, "senescence", "cytokinin", 8, 1),
        L(0x114, "senescence", "tf", 0x0D, out=23),
        L(0x115, "vern", "vern", 0x05),
    ]
    hormones = (
        ("auxin", 7, (("sugar", 2, 0), ("cytokinin", 8, 1)), 0x0D),
        ("cytokinin", 8, (("nitrogen", 6, 0), ("auxin", 7, 1)), 0x0D),
        ("gibberellin", 9, (("sugar", 2, 0), ("aba", 10, 1)), 0x0D),
        ("aba", 10, (("dormancy_keeper", 22, 0), ("water", 5, 1)), 0x1D),
        ("ethylene", 11, (("damage", 25, 0), ("senescence", 23, 0)), 0x1D),
        ("salicylate", 12, (("pest_recognition", 31, 0), ("jasmonate", 13, 1)), 0x1D),
        ("jasmonate", 13, (("damage", 25, 0), ("salicylate", 12, 1)), 0x1D),
    )
    ident = 0x200
    for name, code, inputs, tf_flags in hormones:
        for source, src, mode in inputs:
            loci.append(_master(ident, name, source, src, mode))
            ident += 1
        loci.append(L(ident, f"{name}_synth", "tf", tf_flags, out=code))
        loci.append(L(ident + 1, f"{name}_catab", "enz", 0x0C, s1=code, s2=255, p1=255, p2=255))
        ident += 2
    loci += [
        L(0x300, "enz_photosynthesis", "enz", 0x0C, s1=0, s2=5, p1=2, p2=255),
        L(0x301, "enz_respiration", "enz", 0x0C, s1=2, s2=255, p1=255, p2=255),
        L(0x302, "enz_fructan_synth_1", "enz", 0x0C, s1=2, s2=255, p1=3, p2=255),
        L(0x303, "enz_fructan_synth_2", "enz", 0x08, s1=2, s2=255, p1=3, p2=255),
        L(0x304, "enz_fructan_hydrolysis", "enz", 0x0C, s1=3, s2=255, p1=2, p2=255),
        L(0x305, "enz_n_uptake", "enz", 0x0C, s1=255, s2=255, p1=6, p2=255),
        L(0x306, "enz_water_uptake", "enz", 0x0C, s1=255, s2=255, p1=5, p2=255),
        L(0x307, "rec_sun", "rec", 0x04, channel=0, species=0),
        L(0x308, "rec_lamp", "rec", 0x00, channel=1, species=1),
        L(0x309, "rec_water", "rec", 0x04, channel=2, species=32),
        L(0x30A, "rec_temperature", "rec", 0x04, channel=3, species=30),
        L(0x30B, "rec_touch", "rec", 0x00, channel=4, species=29),
        L(0x30C, "rec_pest_a", "rec", 0x10, channel=5, species=31),
        L(0x30D, "rec_pest_b", "rec", 0x10, channel=6, species=31),
        L(0x30E, "rec_phy_rfr", "rec", 0x01, channel=7, species=33, gain=(2048, 6144)),
        L(0x30F, "rec_blue", "rec", 0x01, channel=8, species=34, gain=(2048, 6144)),
        L(0x310, "rec_uv", "rec", 0x01, channel=9, species=35, gain=(2048, 6144)),
    ]
    immunity = (("r0", "salicylate", 12, 0), ("r1", "salicylate", 12, 0),
                ("r2", "jasmonate", 13, 1), ("r3", "jasmonate", 13, 1))
    ident = 0x400
    for name, source, src, cls in immunity:
        loci.append(L(ident, f"cis_imm_{name}_by_{source}", "cis", 0x08, founder={"src": src}, mode=0))
        loci.append(L(ident + 1, f"imm_{name}", "imm", 0x18, MASKS["live"], **{"class": cls}))
        ident += 2
    for offset, name in enumerate(("i", "c", "r", "g")):
        loci.append(L(0x408 + offset, f"pig_{name}", "pig", 0x00, **{"class": offset}))
    for index in range(4):
        loci.append(L(0x40C + index, f"te_{index}", "te", 0x10))
    for index, target in enumerate((6, 7, 8, 9, 10, 11, 14, 17)):
        loci.append(L(0x410 + index, f"load_{index}", "load", 0x02, target=target))
    morph = (
        (0x500, "leaf_init", 0x0C, "leaf", 0, 1, 2),
        (0x501, "leaf_elong", 0x0C, "leaf", 1, 1, 9),
        (0x502, "sheath_fill", 0x0C, "sheath", 2, 2, 3),
        (0x503, "scale_corr", 0x0C, "sheath", 1, 3, 17),
        (0x504, "root_init", 0x0C, "live", 0, 4, 7),
        (0x505, "root_tip", 0x0C, "live", 5, 5, 2),
        (0x506, "bolting", 0x08, 0x080, 0, 6, 19),
        (0x507, "fall_over", 0x08, 0x020, 1, 1, 23),
        (0x508, "tunic_dry", 0x08, 0x060, 3, 8, 10),
        (0x509, "umbel_open", 0x08, 0x100, 6, 7, 19),
        (0x50A, "bulbil", 0x08, 0x600, 7, 9, 3),
        (0x50B, "offset", 0x08, 0x400, 3, 10, 3),
    )
    for ident, name, flags, mask, pred, succ, guard in morph:
        mask = MASKS[mask] if isinstance(mask, str) else mask
        loci.append(L(ident, f"morph_{name}", "morph", flags, mask, pred=pred, guard=guard, succ=succ))
    for trait, name in enumerate(SHAPE_TRAITS):
        extra = {"value": SHAPE_VALUE_BOX[trait]} if trait in SHAPE_VALUE_BOX else {}
        loci.append(L(0x50C + trait, f"shape_{name}", "shape", 0x01, trait=trait, window=0, **extra))
    plast = (("relay_cortex", 0, 0), ("cortex_cortex", 0, 0), ("cortex_go", 1, 0), ("cortex_nogo", 1, 1),
             ("cortex_critic", 1, 0), ("cortex_prediction", 2, 0), ("hippo_cortex", 3, 0))
    for cls, (name, sel, sign) in enumerate(plast):
        loci.append(L(0x600 + cls, f"plast_{name}", "plast", 0x05, M_SEL=sel, M_SIGN=sign,
                      **{"class": cls}))
    for channel in range(7):
        loci.append(L(0x607 + channel, f"temp_homeo_{channel}", "temp", 0x01, channel=channel,
                      habituation=0, curiosity=0))
    loci.append(L(0x60E, "temp_curiosity", "temp", 0x11, channel=255, base=0))
    for index in range(5):
        loci.append(lex(0x700 + index, index, 0x05))
    loci.append(lex(0x705, 5, 0x04))
    loci.append(lex(0x706, 6, 0x04))
    for index in range(12):
        loci.append(L(0x707 + index, f"temp_social_{index}", "temp", 0x01, channel=16 + index,
                      weight=0, curiosity=0))
    return tuple(loci)


V0_1_LOCI = _v0_1_loci()

# ---------------------------------------------------------------------------
# Genome sections
# ---------------------------------------------------------------------------


def _kind_names(kind):
    return [field[g.F_NAME] for field in g.KIND_BY_NAME[kind][g.K_FIELDS]]


def kinds_section():
    out = []
    for code, name, fields, _modes, _strength, _reserved in g.KINDS:
        boxes = KIND_BOXES[name]
        out.append({
            "code": code,
            "fields": [{"hi": boxes[f[g.F_NAME]][1], "lo": boxes[f[g.F_NAME]][0], "name": f[g.F_NAME]}
                       for f in fields],
            "name": name,
        })
    return out


def species_section():
    out = []
    for code, (name, cls) in enumerate(SPECIES):
        deg = 0 if cls in ("input", "conserved") else DEFAULT_DEG
        out.append({"class": cls, "code": code, "deg": deg, "name": name})
    return out


def _box(value):
    return (value, value) if isinstance(value, int) else value


def locus_entry(locus):
    names = _kind_names(locus.kind)
    unknown = [name for name in locus.fields if name not in names]
    if unknown:
        raise ValueError(f"{locus.name}: unknown fields {unknown}")
    overrides = []
    for name in names:
        if name in locus.fields:
            low, high = _box(locus.fields[name])
            overrides.append({"field": name, "hi": high, "lo": low})
    return {"box": overrides, "flags": locus.flags, "id": locus.ident, "kind": locus.kind,
            "name": locus.name, "stages": locus.mask}


def genome_section(loci, pairs):
    chromosomes = [[] for _ in range(pairs)]
    for locus in sorted(loci, key=lambda item: item.ident):
        chromosomes[locus.ident >> 8].append(locus.ident)
    return {
        "chromosomes": chromosomes,
        "kinds": kinds_section(),
        "loci": [locus_entry(locus) for locus in sorted(loci, key=lambda item: item.ident)],
        "max_bytes": g.HEADER + pairs * g.PLOIDY * 2 + g.PLOIDY * g.RECORD * len(loci),
        "max_promoter": 8,
        "max_records": 64,
        "pairs": pairs,
        "schema": g.SCHEMA,
        "species": species_section(),
        "stages": g.STAGES,
    }


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

def allele(name, freq, body, window):
    return {"body": list(body), "freq": freq, "name": name, "window": list(window)}


def _rep(value, count):
    return [value] * count


# The fixture pool, literally: locus -> [(name, freq, body, window), ...].
_CLOCK_TF_W = [0, 512, 64, 0, 0]
_LEX_A = [160] * 9
_LEX_B = [96] * 9


def _clock_tf(out):
    return [("early", 1, [out, 49152, 4681, -8192, 16384], _CLOCK_TF_W),
            ("mid", 2, [out, 49152, 4096, -8192, 16384], _CLOCK_TF_W),
            ("late", 1, [out, 49152, 3641, -8192, 16384], _CLOCK_TF_W)]


def _clock_cis(src):
    return [("steep", 3, [src, 4096, 8192, 4, 1], [0] * 5), ("soft", 1, [src, 4096, 8192, 3, 1], [0] * 5)]


LEX_COLUMNS = {
    0: (_LEX_A, _LEX_B, [32] * 9),
    9: (_LEX_A, _LEX_B, [32] * 9),
    18: (_LEX_A, _LEX_B, [32] * 9),
    27: ([32, 128, 16, 64, 8, 8, 64, 128, 32], [64, 128, 32, 32, 0, 0, 96, 96, 64], [32] * 9),
    36: ([128, 128, 128, 128, 128, 128, 60, 128, 128], [96, 96, 96, 96, 96, 96, 90, 160, 160], [16] * 9),
    45: ([1, 1, 3, 0, 1, 0, 0, 1, 2], [2, 0, 1, 1, 0, 1, 1, 0, 4], [0] * 9),
    54: ([0, 0, 128, 0, 0, 0, 0, 0, 0], [1, 1, 128, 0, 0, 0, 0, 0, 0], [0, 0, 16, 0, 0, 0, 0, 0, 0]),
}


def _lex_alleles(offset, freq_a, freq_b, names=("a", "b")):
    a, b, win = LEX_COLUMNS[offset]
    return [(names[0], freq_a, [offset] + a, [0] + win), (names[1], freq_b, [offset] + b, [0] + win)]


FIXTURE_POOL = {
    0: [("slow", 3, [0, 14, 256, 0], [0, 0, 32, 0]), ("quick", 1, [0, 14, 512, 0], [0, 0, 32, 0])],
    1: _clock_cis(16),
    2: _clock_tf(14),
    3: _clock_cis(14),
    4: _clock_tf(15),
    5: _clock_cis(15),
    6: _clock_tf(16),
    7: [("common", 3, [0, 5, 2, 255, 16384, 2048, 49152], [0, 0, 0, 0, 512, 128, 512]),
        ("rich", 1, [0, 5, 2, 255, 20480, 2048, 49152], [0, 0, 0, 0, 512, 128, 512])],
    8: [("common", 3, [0, 2, 1024, 1, 8192, 64], [0, 0, 64, 0, 256, 2]),
        ("fast", 1, [0, 2, 1024, 1, 10240, 64], [0, 0, 64, 0, 256, 2])],
    9: [("resistant", 1, [0, 32768, 4096], [0, 1024, 256]), ("null", 1, [0, 0, 0], [0, 0, 0])],
    10: [("red", 1, [2, 1], [0, 0]), ("yellow", 1, [2, 0], [0, 0])],
    11: [("slow", 3, [0, 16, 0, 0, 0, 0, 6, 3, 3, 4, 0, 0], [0, 2] + [0] * 10),
         ("keen", 1, [0, 24, 0, 0, 0, 0, 5, 3, 3, 4, 0, 0], [0, 2] + [0] * 10)],
    12: [("calm", 1, [255, 128, 0, 2, 4096], [0, 16, 0, 0, 256]),
         ("curious", 1, [255, 64, 0, 1, 8192], [0, 16, 0, 0, 256])],
    13: _lex_alleles(0, 1, 1, ("full", "spare")),
    14: [("round", 1, [0, 32768, 0], [0, 1024, 0]), ("broad", 1, [0, 40960, 0], [0, 1024, 0])],
    15: [("temperate", 1, [14, 1, 2, 90, 30, 3, 60, 300], [2, 0, 0, 10, 4, 0, 10, 30]),
         ("quick", 1, [10, 2, 1, 60, 25, 2, 40, 240], [2, 0, 0, 10, 4, 0, 10, 30])],
    16: [("evening", 1, [5, 256, 256, 256, 256], [0, 32, 32, 32, 32]),
         ("sensitive", 1, [5, 512, 512, 512, 512], [0, 32, 32, 32, 32])],
    17: [("active", 1, [512, 0], [64, 0]), ("silent", 3, [0, 0], [0, 0])],
    18: [("clean", 7, [7, 0], [0, 0]), ("loaded", 1, [7, 16384], [0, 1024])],
    19: _lex_alleles(9, 1, 1),
    20: _lex_alleles(18, 1, 1),
    21: _lex_alleles(27, 1, 1),
    22: _lex_alleles(36, 1, 1),
    23: _lex_alleles(45, 1, 1),
    24: _lex_alleles(54, 1, 1),
    25: [("coloured", 3, [0, 0], [0, 0]), ("inhibitor", 1, [0, 1], [0, 0])],
    26: [("c_on", 3, [1, 1], [0, 0]), ("c_off", 1, [1, 0], [0, 0])],
    27: [("golden", 1, [3, 1], [0, 0]), ("shallot", 1, [3, 0], [0, 0])],
    28: [("a", 1, [0, 0], [0, 256]), ("b", 1, [0, 4096], [0, 256])],
}

# Defaults of the v1 rule: kind -> field -> (value, window).
DEFAULTS = {
    "tf": {"prod": (32768, 1024), "deg": (8192, 256), "bias": (0, 256), "rate": (16384, 512)},
    "cis": {"w": (4096, 256), "K": (4096, 256), "n": (2, 0)},
    "enz": {"kcat": (16384, 512), "Km": (2048, 128), "yield": (32768, 512)},
    "rec": {"gain": (4096, 256), "threshold": (0, 0)},
    "morph": {"threshold": (1024, 64), "rate": (8192, 256), "angle": (64, 2)},
    "plast": {"A": (16, 2), "Aneg": (0, 0), "B": (0, 0), "C": (0, 0), "D": (0, 0), "LR": (6, 0),
              "TAU_X": (3, 0), "TAU_Y": (3, 0), "TAU_E": (4, 0)},
    "shape": {"value": (32768, 1024)},
}
TEMP_DEFAULTS = {
    "homeo": {"weight": (256, 16), "base": (16384, 256)},
    "social": {"base": (512, 16), "habituation": (2, 0)},
    "curiosity": {"weight": (128, 16), "habituation": (2, 0), "curiosity": (4096, 256)},
}
SHAPE_DEFAULTS = {2: (2, 0), 4: (8, 0), 8: (0, 0), 9: (0, 0)}
PRC_DEFAULT = {0: 0, 1: -256, 2: -256, 3: 0, 4: 256, 5: 256}
STEP_FIELD = {"tf": "prod", "cis": "w", "enz": "kcat", "rec": "gain", "morph": "rate", "imm": "strength",
              "pig": "allele", "plast": "A", "temp": "weight", "shape": "value", "vern": "VU_req",
              "te": "activity", "load": "effect", "brain": "value"}


def _effective(entry):
    boxes = {f["name"]: (f["lo"], f["hi"]) for f in
             next(k for k in kinds_section() if k["name"] == entry["kind"])["fields"]}
    for override in entry["box"]:
        boxes[override["field"]] = (override["lo"], override["hi"])
    return boxes


def _rule_alleles(locus, entry):
    """Alleles ``a`` and ``b`` of a v1 locus by the authoring rule."""
    kind = locus.kind
    names = _kind_names(kind)
    boxes = _effective(entry)
    marks = {f[g.F_NAME]: f[g.F_MARKS] for f in g.KIND_BY_NAME[kind][g.K_FIELDS]}
    defaults = dict(DEFAULTS.get(kind, {}))
    if kind == "temp":
        if locus.name == "temp_curiosity":
            defaults = TEMP_DEFAULTS["curiosity"]
        elif locus.name.startswith("temp_social"):
            defaults = TEMP_DEFAULTS["social"]
        else:
            defaults = TEMP_DEFAULTS["homeo"]
    if kind == "plast" and locus.name == "plast_cortex_cortex":
        defaults["Aneg"] = (8, 0)
    if kind == "shape":
        trait = locus.fields["trait"]
        if trait in SHAPE_DEFAULTS:
            defaults = {"value": SHAPE_DEFAULTS[trait]}
    if kind == "prc":
        shift = PRC_DEFAULT[locus.fields["block"]]
        defaults = {f"shift{i}": (shift, 32) for i in range(4)}
    body, window = [], []
    for name in names:
        low, high = boxes[name]
        if low == high:
            body.append(low)
            window.append(0)
        elif marks[name] & g.S:
            body.append(locus.founder[name])
            window.append(0)
        else:
            value, win = defaults[name]
            body.append(value)
            window.append(win)
    step_name = STEP_FIELD.get(kind)
    if kind in ("prc", "lex") or step_name is None or boxes[step_name][0] == boxes[step_name][1]:
        step_name = next(name for name in names
                         if boxes[name][0] != boxes[name][1] and not marks[name] & g.S)
    index = names.index(step_name)
    low, high = boxes[step_name]
    step = max(1, (high - low) // 8)
    moved = list(body)
    moved[index] = min(body[index] + step, high)
    if moved[index] == body[index]:
        moved[index] = max(body[index] - step, low)
    return [("a", 3, body, window), ("b", 1, moved, window)]


def _v1_alleles(locus, entry):
    name, kind, fields = locus.name, locus.kind, locus.fields
    if name in ("clock_m", "clock_d", "clock_e"):
        return _clock_tf(fields["out"])
    if name.startswith("cis_clock_"):
        return _clock_cis(fields["src"])
    if name == "rec_clock_light":
        return FIXTURE_POOL[0]
    if name == "rec_photoperiod":
        return [("long_day", 1, [10, 17, 4096, 38229], [0, 0, 256, 546]),
                ("short_day", 1, [10, 17, 4096, 32768], [0, 0, 256, 546])]
    if kind == "imm":
        cls = fields["class"]
        return [("resistant", 1, [cls, 32768, 4096], [0, 1024, 256]), ("null", 1, [cls, 0, 0], [0, 0, 0])]
    if kind == "pig":
        return {"pig_i": [("coloured", 3, [0, 0], [0, 0]), ("inhibitor", 1, [0, 1], [0, 0])],
                "pig_c": [("c_on", 3, [1, 1], [0, 0]), ("c_off", 1, [1, 0], [0, 0])],
                "pig_r": [("red", 1, [2, 1], [0, 0]), ("yellow", 1, [2, 0], [0, 0])],
                "pig_g": [("golden", 1, [3, 1], [0, 0]), ("shallot", 1, [3, 0], [0, 0])]}[name]
    if kind == "te":
        return FIXTURE_POOL[17]
    if kind == "load":
        target = fields["target"]
        return [("clean", 7, [target, 0], [0, 0]), ("loaded", 1, [target, 16384], [0, 1024])]
    if kind == "vern":
        window = [2, 0, 0, 10, 4, 0, 10, 30]
        return [("a", 1, [14, 1, 2, 90, 30, 3, 60, 300], window),
                ("b", 1, [10, 2, 1, 60, 25, 2, 40, 240], window),
                ("c", 1, [18, 1, 3, 120, 35, 4, 85, 360], window)]
    if kind == "lex":
        return _lex_alleles(fields["offset"], 3, 1)
    return _rule_alleles(locus, entry)


def pool(name, loci, alleles_of):
    entries = []
    for locus in sorted(loci, key=lambda item: item.ident):
        chosen = alleles_of(locus, locus_entry(locus))
        entries.append({"alleles": [allele(*spec) for spec in chosen], "locus": locus.ident})
    return {"loci": entries, "name": name, "schema": 1, "version": 1}


# ---------------------------------------------------------------------------
# Laws
# ---------------------------------------------------------------------------

DECAYS = {"tag": {"kind": "lazy", "length": 256, "tau": 3}, "trace": {"kind": "iter", "tau": 2}}


def _digest(value):
    return hashlib.sha256(wire.emit(value)).hexdigest()


def _sine_digest():
    text = TABLES_DIR.joinpath("sine_q15_v1.json").read_bytes()
    return _digest(wire.parse(text, lenient=True))


def law(name, loci, pairs, founders_name, founders_value):
    value = {
        "decays": DECAYS,
        "founders": {"name": founders_name, "sha256": _digest(founders_value)},
        "genome": genome_section(loci, pairs),
        "name": name,
        "provisional": True,
        "quiescent_in_dormancy": [],
        "tables": {"sine_q15": _sine_digest()},
        "version": 0,
    }
    value.update(bounds.compute(value))
    return value


# ---------------------------------------------------------------------------
# Rendering: indented JSON, sorted keys, one small object or scalar list per line
# ---------------------------------------------------------------------------

def _flat(value):
    if isinstance(value, dict):
        return all(not isinstance(v, (dict, list)) or (isinstance(v, list) and _scalars(v))
                   for v in value.values())
    if isinstance(value, list):
        return _scalars(value)
    return True


def _scalars(items):
    return all(not isinstance(item, (dict, list)) for item in items)


def render(value, indent=0):
    if _flat(value) or not isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    pad = "  " * (indent + 1)
    end = "  " * indent
    if isinstance(value, dict):
        items = [f"{pad}{json.dumps(key)}: {render(value[key], indent + 1)}" for key in sorted(value)]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    items = [f"{pad}{render(item, indent + 1)}" for item in value]
    return "[\n" + ",\n".join(items) + "\n" + end + "]"


def author():
    """Every authored file, ``{file name: text}``; writes nothing."""
    fixture_pool = pool("fixture", FIXTURE_LOCI, lambda locus, entry: FIXTURE_POOL[locus.ident])
    v1_pool = pool("v1", V0_1_LOCI, _v1_alleles)
    laws = {
        "fixture.json": law("fixture", FIXTURE_LOCI, 1, "fixture", fixture_pool),
        "v0_1.json": law("v0_1", V0_1_LOCI, 8, "v1", v1_pool),
    }
    files = {name: render(value) + "\n" for name, value in laws.items()}
    files["founders_fixture.json"] = render(fixture_pool) + "\n"
    files["founders_v1.json"] = render(v1_pool) + "\n"
    for name in ("fixture.json", "v0_1.json"):
        value = laws[name]
        defects = g.validate_law(value) + bounds.defects(value)
        pool_value = fixture_pool if name == "fixture.json" else v1_pool
        defects += g.validate_pool(value, pool_value)
        if defects:
            raise ValueError(f"{name}: {defects[:5]}")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the files on disk, write nothing")
    args = parser.parse_args(argv)
    files = author()
    stale = []
    for name in sorted(files):
        path = LAWS_DIR.joinpath(name)
        current = path.read_text(encoding="ascii") if path.exists() else None
        if current != files[name]:
            stale.append(name)
            if not args.check:
                path.write_text(files[name], encoding="ascii")
    if args.check:
        print("stale: " + ", ".join(stale) if stale else "every authored file is current")
        return 1 if stale else 0
    print("written: " + ", ".join(stale) if stale else "nothing to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
