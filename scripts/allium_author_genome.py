#!/usr/bin/env python3
"""Author the componion's genome laws and founder pools from one table.

Writes, under ``opti_oignon/allium/laws/``:

* ``fixture.json`` -- the small law the t1 contracts run on: one pair, 32 loci
  (a three-node clock, one record of every kind, the whole language block,
  the four colour loci, and the three enzymes of the reserve the full law
  also carries: respiration, fructan synthesis and hydrolysis);
* ``v0_1.json`` -- the first provisional law of the full genome: eight
  pairs, 175 loci;
* ``founders_fixture.json`` and ``founders_v1.json`` -- the founder allele
  pools each law pins by digest.

Each law also pins, by name and digest, the phonology table and the taboo
table it speaks with, and the journal's table of kinds with the daily budget
of each budgeted kind; ``scripts/allium_author_phon.py`` and
``scripts/allium_author_journal.py`` write those tables, own those budgets
and run first.

Each law carries the sections its life runs on: the organ ``code`` revision
and its organs, the organs quiescent in dormancy, the bus, the ``world``
(year, seasons, daylength, rain), the organs' ``constants``, the ranges of
the ``params`` a genesis freezes, and the ``work`` section: the unit table,
the per-day caps of the budget-exempt kinds, and the day ceilings derived
from both, each with its proof. ``caps(law, table)`` and
``ceilings(law, table)`` compute them; a contract loads them by path and
recomputes the recorded ones. Every number there is a placeholder the later
organs retune; a change to what an organ does bumps ``code``.

``laws/retired.json`` lists the prototype laws that were rewritten in place
or removed, by name and digest, so that a being sown under one is named as a
retired prototype and never read under another digest. It is never
embedded. ``--retire NAME`` appends law ``NAME`` as it is on disk now; run
it before rewriting or removing a provisional law that a being outside the
contracts may have been sown under.

The engines read the written files, never this script. The script is the
authoring tool: a contract runs it in-process and checks that it reproduces
all four files byte for byte, so a hand edit to a law or a pool without the
matching change here is caught.

Usage: ``python3 scripts/allium_author_genome.py [--check] [--retire NAME ...]``.
``--check`` writes nothing and exits 1 when a file on disk differs.

The v1 pool follows a rule rather than a list: a base allele ``a`` from the
kind's defaults, and an allele ``b`` that moves one field by an eighth of
its box, with named exceptions for the loci whose alleles mean something
(clock period, photoperiod, colour, resistance, transposons, loads, beard,
body shape levels).
"""

import argparse
import hashlib
import importlib.util
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
    "shape": {"trait": (0, 31), "value": U16, "window": U16},
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

# Beard and moustache loci of the full law: (id, name, trait, value box).
# Each sits after every other locus of its pair, so no record lands inside a
# gene's promoter run and no other locus's founder stream moves. Trait 15
# stays free. A dominant reading (cream, moustache) uses alleles 2 and 0, so
# the additive mean shows a carrier as 1; mc1r reads the same way, 1 being
# the carrier with red glints.
BEARD_LOCI = (
    (0x00D, "shape_beard_mel_0", 12, (0, 16383)),
    (0x116, "shape_beard_cream", 21, (0, 2)),
    (0x21C, "shape_beard_mc1r", 17, (0, 2)),
    (0x311, "shape_beard_mel_1", 13, (0, 16383)),
    (0x418, "shape_moustache", 22, (0, 2)),
    (0x419, "shape_moustache_form", 23, (0, 3)),
    (0x41A, "shape_moustache_size", 24, (1, 3)),
    (0x41B, "shape_moustache_thick", 25, (0, 1)),
    (0x518, "shape_grey_onset", 18, (730, 4380)),
    (0x519, "shape_grey_span", 19, (730, 7300)),
    (0x51A, "shape_grey_ceiling", 20, (64, 256)),
    (0x60F, "shape_beard_mel_2", 14, (0, 16383)),
    (0x713, "shape_beard_mel_3", 16, (0, 16383)),
)

MEL = (("light", 3, 1024, 768), ("dark", 2, 15359, 768))
BEARD_ALLELES = {
    "shape_beard_mel_0": MEL, "shape_beard_mel_1": MEL, "shape_beard_mel_2": MEL, "shape_beard_mel_3": MEL,
    "shape_beard_cream": (("cream", 3, 2, 0), ("plain", 7, 0, 0)),
    "shape_beard_mc1r": (("func", 4, 2, 0), ("lof", 1, 0, 0)),
    "shape_moustache": (("yes", 4, 2, 0), ("no", 21, 0, 0)),
    "shape_moustache_form": (("brush", 5, 0, 0), ("straight", 7, 1, 0), ("droopy", 4, 2, 0), ("curled", 4, 3, 0)),
    "shape_moustache_size": (("small", 1, 1, 0), ("medium", 1, 2, 0), ("large", 1, 3, 0)),
    "shape_moustache_thick": (("thick", 2, 1, 0), ("thin", 3, 0, 0)),
    "shape_grey_onset": (("early", 1, 730, 30), ("mid", 2, 1095, 30), ("late", 1, 1825, 30)),
    "shape_grey_span": (("fast", 1, 1095, 60), ("mid", 2, 2190, 60), ("slow", 1, 3650, 60)),
    "shape_grey_ceiling": (("full", 4, 256, 0), ("retained", 1, 96, 0)),
}

# The eight continuous body shape loci: every allele sits on the lattice
# value = 8192 * k, so a genotype compiles to 4096 * (ka + kb) and a lone
# allele to its homozygote's value. The greenhouse reads a value by rounding
# it to that step and counting cuts that sit halfway between two steps, so
# each genotype stays 2048 from its nearest cut, twice the founder window.
BODY_WINDOW = 1024
BODY_ALLELES = {
    "shape_bulb_radius": (("slim", 1, 24576, BODY_WINDOW), ("round", 1, 32768, BODY_WINDOW),
                          ("broad", 1, 40960, BODY_WINDOW)),
    "shape_bulb_height": (("squat", 3, 16384, BODY_WINDOW), ("mid", 4, 24576, BODY_WINDOW),
                          ("tall", 4, 32768, BODY_WINDOW), ("towering", 1, 40960, BODY_WINDOW)),
    "shape_neck": (("short", 3, 24576, BODY_WINDOW), ("mid", 4, 32768, BODY_WINDOW),
                   ("long", 3, 49152, BODY_WINDOW)),
    "shape_leaf_stiffness": (("limp", 1, 16384, BODY_WINDOW), ("soft", 1, 24576, BODY_WINDOW),
                             ("firm", 1, 32768, BODY_WINDOW), ("stiff", 1, 40960, BODY_WINDOW)),
    "shape_breath_base": (("close", 7, 32768, BODY_WINDOW), ("wide", 3, 40960, BODY_WINDOW)),
    "shape_turn_speed": (("slow", 3, 24576, BODY_WINDOW), ("steady", 4, 32768, BODY_WINDOW),
                         ("quick", 3, 49152, BODY_WINDOW)),
    "shape_stripe": (("plain", 3, 32768, BODY_WINDOW), ("striped", 7, 40960, BODY_WINDOW)),
    "shape_speckle": (("clean", 3, 24576, BODY_WINDOW), ("freckled", 2, 32768, BODY_WINDOW),
                      ("weathered", 3, 40960, BODY_WINDOW)),
}


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


_FIXTURE_BASE = (
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
    for ident, name, trait, box in BEARD_LOCI:
        loci.append(L(ident, name, "shape", 0x01, trait=trait, window=0, value=box))
    return tuple(loci)


V0_1_LOCI = _v0_1_loci()


def _copied(ident, name):
    """Locus ``name`` of the full law, under the fixture id ``ident``: kind, flags, stages and boxes alike."""
    source = next(locus for locus in V0_1_LOCI if locus.name == name)
    return L(ident, source.name, source.kind, source.flags, source.mask, dict(source.founder), **source.fields)


# The enzymes the reserve reads, which the fixture's one chromosome carries
# after every other locus: no other locus's record or founder stream moves.
FIXTURE_ENZYMES = ((29, "enz_respiration"), (30, "enz_fructan_synth_1"), (31, "enz_fructan_hydrolysis"))
FIXTURE_LOCI = _FIXTURE_BASE + tuple(_copied(ident, name) for ident, name in FIXTURE_ENZYMES)

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
    table = BEARD_ALLELES if name in BEARD_ALLELES else BODY_ALLELES if name in BODY_ALLELES else None
    if table is not None:
        trait = fields["trait"]
        return [(label, freq, [trait, value, 0], [0, window, 0]) for label, freq, value, window in table[name]]
    return _rule_alleles(locus, entry)


def pool(name, loci, alleles_of):
    entries = []
    for locus in sorted(loci, key=lambda item: item.ident):
        chosen = alleles_of(locus, locus_entry(locus))
        entries.append({"alleles": [allele(*spec) for spec in chosen], "locus": locus.ident})
    return {"loci": entries, "name": name, "schema": 1, "version": 1}


# ---------------------------------------------------------------------------
# Life sections
# ---------------------------------------------------------------------------

# The organ code revision a law runs: the engine implements a closed set, and
# a change to what an organ does bumps it, so the law's digest moves with it.
CODE = "seed_1"
ORGANS = ["chem", "clock", "soil", "stage"]
QUIESCENT = ["chem", "clock"]
BUS = {"circadian": "clock", "dormant": "stage", "metab": "chem", "moisture": "soil"}

# Year, seasons, the provisional daylength (minutes, a sine by latitude band)
# and rain by season (winter, spring, summer, autumn; Q16 of ONE). The
# fixture's year is forty days, the full law's the civil calendar.
_RAIN = {"max": [21845, 21845, 16384, 21845], "p_wet": [39322, 32768, 22938, 32768]}
_AMP = {"long": 240, "medium": 150, "short": 60}
WORLD = {
    "fixture": {"daylength": {"amp": _AMP, "equinox": 10, "mean": 720, "ramp": 60}, "rain": _RAIN,
                "season_shift": 5, "south_shift": 20, "year": {"days": 40, "kind": "fixed"}},
    "v0_1": {"daylength": {"amp": _AMP, "equinox": 79, "mean": 720, "ramp": 60}, "rain": _RAIN,
             "season_shift": 31, "south_shift": 183, "year": {"kind": "civil"}},
}

_SYNTHESIS = {"fixture": ["enz_fructan_synth_1"], "v0_1": ["enz_fructan_synth_1", "enz_fructan_synth_2"]}
_STAGE_REST = {"fixture": (2, 5), "v0_1": (7, 60)}


def constants(name):
    """The organs' constants under law ``name``: stocks, scales and thresholds, and the loci they read."""
    rest_dry, rest_winter = _STAGE_REST[name]
    return {
        "chem": {"core": 32768, "enzymes": {"hydrolysis": ["enz_fructan_hydrolysis"],
                                            "photosynthesis": ["enz_photosynthesis"],
                                            "respiration": ["enz_respiration"], "synthesis": _SYNTHESIS[name]},
                 "fructan0": 131072, "fructan_max": 4194304, "hyd_scale": 512, "k_w": 16384, "metab_shift": 8,
                 "ps_scale": 1024, "resp_scale": 128, "sugar0": 16384, "sugar_max": 262144, "syn_scale": 512,
                 "theta_h": 8192, "theta_s": 16384},
        "clock": {"genes": ["clock_m", "clock_d", "clock_e"], "init": [65536, 32768, 0], "light": "rec_clock_light"},
        "soil": {"dose": 16384, "m0": 32768, "m_max": 65536},
        "stage": {"d_enter": 14, "rest_dry": rest_dry, "rest_max": 1000, "rest_winter": rest_winter,
                  "theta_dry": 16384, "theta_wet": 32768},
    }


# The ranges of the params a genesis freezes; the table's params schema bounds them.
PARAMS = {
    "evap_awake": {"default": 4096, "hi": 16384, "lo": 0},
    "evap_dormant": {"default": 1024, "hi": 8192, "lo": 0},
    "rain_gain": {"default": 65536, "hi": 131072, "lo": 0},
    "sun_max": {"default": 65536, "hi": 65536, "lo": 16384},
}

# What the life costs, in units: a visited minute, a consumed fact, a day of
# the fast path, a weather draw, the environment of an awake fast step, each
# organ's layer awake and dormant, and each act's cost to the organs it moves.
UNITS = {
    "act": {"water": {"soil": 1}}, "draw": 2, "env": 4, "fact": 1, "fast_path_day": 1,
    "organs": {"chem": {"fast": {"awake": 15, "dormant": 0}}, "clock": {"fast": {"awake": 13, "dormant": 0}},
               "soil": {"daily": {"awake": 2, "dormant": 2}}, "stage": {"daily": {"awake": 1, "dormant": 1}}},
    "visit": 1,
}
# The budget-exempt kinds a gesture never writes, capped per day of life so
# that a day's ceiling bounds every folded fact.
CAPPED_RARE = {"owner": 8, "resumed": 8}
# Any 1440 consecutive minutes hold 96 fast boundaries; offsets span 1680
# minutes, so one day of life holds at most three daily firings.
FAST_A_DAY = 96
FIRINGS_A_DAY = 3


def _budget_total(law, table):
    """``F``: the daily budgets of the budgeted kinds whose body is defined, summed."""
    kinds = table["kinds"]
    return sum(budget for kind, budget in law["journal"]["budgets"].items() if kinds[kind]["body"] is not None)


def caps(law, table):
    """Per-day caps of the exempt trunk kinds other than genesis: ``F`` for the recorder's, fixed for the rest."""
    total = _budget_total(law, table)
    return {"clock": total, "owner": CAPPED_RARE["owner"], "resumed": CAPPED_RARE["resumed"], "tz": total}


def _layer(units, organ, layer):
    costs = units["organs"][organ].get(layer)
    return 0 if costs is None else max(costs["awake"], costs["dormant"])


def _terms(law, table):
    """Every term of both ceilings, with the name it is written under in the proofs."""
    work = law["work"]
    units = work["units"]
    organs = law["organs"]
    budgets = law["journal"]["budgets"]
    kinds = table["kinds"]
    act = max((sum(costs.values()) for costs in units["act"].values()), default=0)
    extra = [(kind, budget, act) for kind, budget in sorted(budgets.items())
             if kinds[kind]["body"] is not None and kind == "act" and act > 0]
    return {
        "fast": [(f"{organ}.fast", _layer(units, organ, "fast")) for organ in organs
                 if "fast" in units["organs"][organ]],
        "daily": [(f"{organ}.daily", _layer(units, organ, "daily")) for organ in organs
                  if "daily" in units["organs"][organ]],
        "dormant": [(f"{organ}.daily", units["organs"][organ]["daily"]["dormant"]) for organ in organs
                    if "daily" in units["organs"][organ] and organ not in law["quiescent_in_dormancy"]],
        "budget": _budget_total(law, table),
        "extra": extra,
        "caps": sum(work["caps"].values()),
    }


def ceilings(law, table):
    """The day ceilings a law's unit table, budgets and caps imply: ``{"awake_day", "dormant_day"}``.

    ``awake_day`` bounds the units of any day of life (``t div 1440``): 96
    awake fast steps, three daily firings with their draws, every budgeted
    fact at its own minute with its largest organ cost, and every capped
    exempt fact at its own minute. ``dormant_day`` is one day of the fast
    path with no fact: the day, one visit, each non-quiescent organ's dormant
    daily layer, and the draw. The visit is there because the fast path stops
    before a pending ``evolve``'s ``effective_from`` and the stepped path
    visits that minute. At a local midnight that visit stands in for the fast
    path's day; once a later ``tz`` fact has moved the minute off its
    midnight, a due ``evolve`` is met by one visit even while the being
    sleeps, and the day's firing is paid as well. A due ``evolve`` adds no
    visit only awake, where every fast boundary is visited anyway.
    """
    units = law["work"]["units"]
    terms = _terms(law, table)
    visit, fact = units["visit"], units["fact"]
    awake = (FAST_A_DAY * (visit + units["env"] + sum(cost for _, cost in terms["fast"]))
             + FIRINGS_A_DAY * (sum(cost for _, cost in terms["daily"]) + units["draw"])
             + terms["budget"] * (visit + fact) + sum(budget * cost for _, budget, cost in terms["extra"])
             + terms["caps"] * (visit + fact))
    dormant = units["fast_path_day"] + visit + sum(cost for _, cost in terms["dormant"]) + units["draw"]
    return {"awake_day": awake, "dormant_day": dormant}


def ceiling_proofs(law, table):
    """Both ceilings as printable ASCII formulas with their numbers, as the bound analysis writes them."""
    units = law["work"]["units"]
    terms = _terms(law, table)
    visit, fact, env, draw = units["visit"], units["fact"], units["env"], units["draw"]
    found = ceilings(law, table)
    fast_names = " + ".join(["visit", "env"] + [name for name, _ in terms["fast"]])
    fast_values = " + ".join(str(value) for value in [visit, env] + [cost for _, cost in terms["fast"]])
    daily_names = " + ".join([name for name, _ in terms["daily"]] + ["draw"])
    daily_values = " + ".join(str(value) for value in [cost for _, cost in terms["daily"]] + [draw])
    extra_names = "".join(f" + {kind}*u_{kind}" for kind, _, _ in terms["extra"])
    extra_values = "".join(f" + {budget}*{cost}" for _, budget, cost in terms["extra"])
    parts = ([FAST_A_DAY * (visit + env + sum(cost for _, cost in terms["fast"])),
              FIRINGS_A_DAY * (sum(cost for _, cost in terms["daily"]) + draw),
              terms["budget"] * (visit + fact)] + [budget * cost for _, budget, cost in terms["extra"]]
             + [terms["caps"] * (visit + fact)])
    awake = (f"{FAST_A_DAY}*({fast_names}) + {FIRINGS_A_DAY}*({daily_names}) + F*(visit + fact){extra_names}"
             f" + caps*(visit + fact) = {FAST_A_DAY}*({fast_values}) + {FIRINGS_A_DAY}*({daily_values})"
             f" + {terms['budget']}*({visit} + {fact}){extra_values} + {terms['caps']}*({visit} + {fact})"
             f" = {' + '.join(str(part) for part in parts)} = {found['awake_day']}")
    dormant_names = " + ".join(["fast_path_day", "visit"] + [name for name, _ in terms["dormant"]] + ["draw"])
    dormant_values = " + ".join(str(value) for value in [units["fast_path_day"], visit]
                                + [cost for _, cost in terms["dormant"]] + [draw])
    dormant = f"{dormant_names} = {dormant_values} = {found['dormant_day']}"
    return {"awake_day": awake, "dormant_day": dormant}


def _journal_table():
    return wire.parse(TABLES_DIR.joinpath("journal_v1.json").read_bytes(), lenient=True)


def life_sections(name):
    """Every life section of law ``name`` but ``work``, which needs the rest of the law."""
    return {"bus": dict(BUS), "code": CODE, "constants": constants(name), "organs": list(ORGANS),
            "params": {key: dict(value) for key, value in PARAMS.items()},
            "quiescent_in_dormancy": list(QUIESCENT), "world": WORLD[name]}


def work_section(law, table):
    """The law's ``work`` section: its unit table, then the caps and ceilings those imply, with their proofs."""
    work = {"caps": caps(law, table), "units": UNITS}
    with_caps = dict(law, work=work)
    return dict(work, ceilings=ceilings(with_caps, table), proof=ceiling_proofs(with_caps, table))


def life_defects(law, table):
    """What the life sections get wrong against the law's own genome and the table, by name."""
    found = []
    work = law["work"]
    if work["caps"] != caps(law, table):
        found.append("work: caps not recomputed")
    if work["ceilings"] != ceilings(law, table) or work["proof"] != ceiling_proofs(law, table):
        found.append("work: ceilings not recomputed")
    kinds = {entry["name"]: entry["kind"] for entry in law["genome"]["loci"]}
    chem = law["constants"]["chem"]
    scales = {"hydrolysis": "hyd_scale", "photosynthesis": "ps_scale", "respiration": "resp_scale",
              "synthesis": "syn_scale"}
    for role, loci in chem["enzymes"].items():
        if not 1 <= len(loci) <= 4 or len(set(loci)) != len(loci) or any(kinds.get(n) != "enz" for n in loci):
            found.append(f"constants: chem {role} names 1..=4 distinct enzyme loci of the law")
        if (len(loci) * 65535 * chem[scales[role]]) >> 16 > 65536:
            found.append(f"constants: chem {role} rate bound")
    clock = law["constants"]["clock"]
    if any(kinds.get(n) != "tf" for n in clock["genes"]) or kinds.get(clock["light"]) != "rec":
        found.append("constants: clock loci")
    for name, spec in law["params"].items():
        bounds_of = table["kinds"]["genesis"]["body"]["laws"]["fields"]["params"]["fields"].get(name)
        if bounds_of is None or not bounds_of["lo"] <= spec["lo"] <= spec["default"] <= spec["hi"] <= bounds_of["hi"]:
            found.append(f"params: {name}")
    return found


# ---------------------------------------------------------------------------
# Laws
# ---------------------------------------------------------------------------

DECAYS = {"tag": {"kind": "lazy", "length": 256, "tau": 3}, "trace": {"kind": "iter", "tau": 2}}


def _digest(value):
    return hashlib.sha256(wire.emit(value)).hexdigest()


def _sine_digest():
    text = TABLES_DIR.joinpath("sine_q15_v1.json").read_bytes()
    return _digest(wire.parse(text, lenient=True))


def _table_digest(name):
    return _digest(wire.parse(TABLES_DIR.joinpath(f"{name}.json").read_bytes(), lenient=True))


def _journal_budgets(name):
    """The daily budgets law ``name`` pins, from the journal's authoring script (loaded by path)."""
    spec = importlib.util.spec_from_file_location(
        "_allium_author_journal", Path(__file__).resolve().parent.joinpath("allium_author_journal.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.BUDGETS[name])


def law(name, loci, pairs, founders_name, founders_value, taboo_name, table):
    value = {
        "decays": DECAYS,
        "founders": {"name": founders_name, "sha256": _digest(founders_value)},
        "genome": genome_section(loci, pairs),
        "journal": {"budgets": _journal_budgets(name),
                    "table": {"name": "journal_v1", "sha256": _table_digest("journal_v1")}},
        "lang": {"phon": {"name": "phon_v1", "sha256": _table_digest("phon_v1")},
                 "taboo": {"name": taboo_name, "sha256": _table_digest(taboo_name)}},
        "name": name,
        "provisional": True,
        "tables": {"sine_q15": _sine_digest()},
        "version": 0,
    }
    value.update(life_sections(name))
    value["work"] = work_section(value, table)
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


def _fixture_alleles(v1_pool):
    """The fixture's alleles: its own table, and the full law's pool by locus name for the copied loci."""
    ident_of = {locus.name: locus.ident for locus in V0_1_LOCI}
    by_ident = {entry["locus"]: entry["alleles"] for entry in v1_pool["loci"]}

    def alleles_of(locus, entry):
        if locus.ident in FIXTURE_POOL:
            return FIXTURE_POOL[locus.ident]
        return [(a["name"], a["freq"], a["body"], a["window"]) for a in by_ident[ident_of[locus.name]]]

    return alleles_of


def author():
    """Every authored file, ``{file name: text}``; writes nothing."""
    v1_pool = pool("v1", V0_1_LOCI, _v1_alleles)
    fixture_pool = pool("fixture", FIXTURE_LOCI, _fixture_alleles(v1_pool))
    table = _journal_table()
    laws = {
        "fixture.json": law("fixture", FIXTURE_LOCI, 1, "fixture", fixture_pool, "taboo_fixture_v1", table),
        "v0_1.json": law("v0_1", V0_1_LOCI, 8, "v1", v1_pool, "taboo_v1", table),
    }
    files = {name: render(value) + "\n" for name, value in laws.items()}
    files["founders_fixture.json"] = render(fixture_pool) + "\n"
    files["founders_v1.json"] = render(v1_pool) + "\n"
    for name in ("fixture.json", "v0_1.json"):
        value = laws[name]
        defects = g.validate_law(value) + bounds.defects(value)
        pool_value = fixture_pool if name == "fixture.json" else v1_pool
        defects += g.validate_pool(value, pool_value)
        defects += life_defects(value, table)
        if defects:
            raise ValueError(f"{name}: {defects[:5]}")
    return files


RETIRED = "retired.json"
_HEX = "0123456789abcdef"


def retired_text(pairs):
    """The register of retired prototype laws, rendered: ``{"retired": [{"name", "sha256"}, ...]}``."""
    for pair in pairs:
        if (sorted(pair) != ["name", "sha256"] or not isinstance(pair["name"], str)
                or not isinstance(pair["sha256"], str) or len(pair["sha256"]) != 64
                or any(char not in _HEX for char in pair["sha256"])):
            raise ValueError(f"{RETIRED}: an entry is not a law's name and digest")
    return render({"retired": [{"name": pair["name"], "sha256": pair["sha256"]} for pair in pairs]}) + "\n"


def _retired_on_disk():
    """The pairs the register holds, or ``None`` when it is absent."""
    path = LAWS_DIR.joinpath(RETIRED)
    if not path.exists():
        return None
    value = wire.parse(path.read_bytes(), lenient=True)
    if not isinstance(value, dict) or sorted(value) != ["retired"] or not isinstance(value["retired"], list):
        raise ValueError(f"{RETIRED}: not a register of retired laws")
    return value["retired"]


def retire(pairs, names):
    """``pairs`` with each named provisional law appended as it is on disk now, once."""
    out = list(pairs)
    for name in names:
        on_disk = wire.parse(LAWS_DIR.joinpath(f"{name}.json").read_bytes(), lenient=True)
        if on_disk.get("provisional") is not True:
            raise ValueError(f"{name}: only a provisional law is retired; a stable law is succeeded")
        pair = {"name": name, "sha256": _digest(on_disk)}
        if pair not in out:
            out.append(pair)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the files on disk, write nothing")
    parser.add_argument("--retire", action="append", default=[], choices=("fixture", "v0_1"), metavar="NAME",
                        help="append law NAME, as it is on disk now, to the register of retired prototypes")
    args = parser.parse_args(argv)
    if args.check and args.retire:
        parser.error("--retire writes the register; it does not go with --check")
    files = author()
    pairs = _retired_on_disk()
    files[RETIRED] = retired_text(retire(pairs or [], args.retire))
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
