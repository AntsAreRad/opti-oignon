#!/usr/bin/env python3
"""Author the componion's phonology table and its taboo tables.

Writes, under ``opti_oignon/allium/tables/``:

* ``phon_v1.json`` -- the alphabet's integer features, the iconic bias, the
  fold table for owner-typed words, the language block's boxes and the
  phonology's constants;
* ``taboo_fixture_v1.json`` -- three witness digests the contracts need to
  prove the taboo filter fires. They are derived here from the unfiltered
  generator, so no word is ever written in the tree;
* ``taboo_v1.json`` -- the full law's taboo list, created empty when it is
  missing and never rewritten here: ``scripts/allium_taboo_digest.py`` owns
  it once the maintainer supplies the English and French lists.

Run it before ``scripts/allium_author_genome.py``, which pins these tables
in the laws. ``--check`` writes nothing and exits 1 when a file on disk
differs. A contract runs ``author()`` in-process and compares.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT.joinpath("opti_oignon", "allium", "tables")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opti_oignon.allium import wire  # noqa: E402
from opti_oignon.allium.ref.organs import phon  # noqa: E402

FEATURES = (
    (0, 8, 7, 1, 7, 0, 1, 0), (0, 8, 7, 1, 7, 1, 0, 0), (0, 8, 7, 1, 7, 2, 0, 0), (0, 8, 7, 1, 7, 1, 2, 1),
    (0, 8, 7, 1, 7, 2, 2, 1),
    (1, 0, 0, 0, 1, 0, 0, 0), (1, 0, 0, 1, 1, 0, 0, 0), (1, 1, 0, 0, 1, 0, 0, 0), (1, 1, 0, 1, 1, 0, 0, 0),
    (1, 4, 0, 0, 1, 0, 0, 0), (1, 4, 0, 1, 1, 0, 0, 0), (1, 5, 0, 0, 1, 0, 0, 0), (1, 6, 0, 0, 1, 0, 0, 0),
    (1, 2, 1, 0, 2, 0, 0, 0), (1, 2, 1, 1, 2, 0, 0, 0), (1, 0, 2, 0, 3, 0, 0, 0), (1, 0, 2, 1, 3, 0, 0, 0),
    (1, 1, 2, 0, 3, 0, 0, 0), (1, 1, 2, 1, 3, 0, 0, 0), (1, 4, 2, 0, 3, 0, 0, 0), (1, 6, 2, 0, 3, 0, 0, 0),
    (1, 0, 3, 1, 4, 0, 0, 0), (1, 1, 3, 1, 4, 0, 0, 0), (1, 1, 4, 1, 5, 0, 0, 0), (1, 1, 5, 1, 5, 0, 0, 0),
    (1, 7, 6, 1, 6, 0, 0, 0), (1, 3, 6, 1, 6, 0, 0, 0),
)
# [size, bright, sharp, speed] per phoneme, in alphabet order: small, bright, sharp and fast are positive.
BIAS = (
    (-16, 0, 0, 0), (32, 32, 16, 16), (64, 64, 32, 32), (-32, -32, -32, -16), (-64, -64, -32, -32),
    (16, 0, 64, 32), (-32, 0, -32, -16), (16, 16, 64, 32), (-32, 0, -32, -16), (16, 0, 64, 32),
    (-32, 0, -32, -16), (0, -16, 32, 16), (0, 0, 32, 16), (16, 16, 32, 16), (-16, 0, -16, 0),
    (0, 0, 16, 16), (0, 0, -16, 0), (16, 32, 32, 32), (0, 0, 16, 0), (0, -16, 16, 0),
    (0, 0, 0, 16), (-32, -16, -64, -32), (0, 0, -16, 0), (-16, 0, -64, -16), (0, 0, 16, 16),
    (-16, -16, -32, -16), (16, 16, 0, 0),
)
# Code points of the Latin letters an owner may type, folded to the alphabet. Written as
# integers so the file stays ASCII.
_FOLDS = {
    "a": (192, 193, 194, 196, 224, 225, 226, 228),
    "c": (199, 231),
    "e": (200, 201, 202, 203, 232, 233, 234, 235),
    "i": (206, 207, 238, 239),
    "o": (212, 214, 244, 246),
    "u": (217, 219, 220, 249, 251, 252),
    "y": (255, 376),
    "oe": (338, 339),
    "ae": (198, 230),
    "'": (8217,),
    "": (32, 45),
}


def fold_table():
    pairs = [(code, chr(code).lower()) for code in range(65, 91)]
    for replacement, codes in _FOLDS.items():
        pairs.extend((code, replacement) for code in codes)
    return [[code, replacement] for code, replacement in sorted(pairs)]


def lex_box():
    box = [[0, 255] for _ in range(phon.LEX_BYTES)]
    for at, pair in ((42, (1, 255)), (45, (1, 2)), (46, (0, 1)), (47, (0, 127)), (48, (0, 1)), (49, (0, 2)),
                     (50, (0, 2)), (51, (0, 1)), (52, (0, 4)), (53, (0, 4)), (54, (0, 1)), (55, (0, 1))):
        box[at] = list(pair)
    for at in range(57, 63):
        box[at] = [0, 0]
    return box


def phon_table():
    return {
        "alphabet": phon.ALPHABET,
        "anchored_max": 4096,
        "anchored_total_max": 16384,
        "bias": [list(row) for row in BIAS],
        "features": [list(row) for row in FEATURES],
        "first_sound_exclude": [12],
        "floor_consonants": 6,
        "floor_vowels": [0, 2, 4],
        "fold": fold_table(),
        "form_max": 12,
        "invent_tries": 16,
        "lex_box": lex_box(),
        "name": "phon_v1",
        "potential_min": 128,
        "sas": dict(phon.SAS),
        "shown_max": 24,
        "taboo_extra_max": 4096,
        "taboo_max": 4096,
        "taboo_window": [3, 8],
        "templates": [list(t) for t in phon.TEMPLATES],
        "version": 1,
    }


def _candidate(ph, concept):
    """The attempt-0 candidate of a witness case, unfiltered, and whether it is licit."""
    out, _work = phon.invent_case(ph, bytes(32), concept, 0, [0, 0, 0, 0], 0, 0, [], [], {}, tries_max=1)
    return out.get("form")


def witnesses(table_value):
    """The three witness concepts and their taboo entries: whole form, a first trigram, eight letters inside."""
    table = phon.Table(table_value)
    ph = phon.decode(phon.law_lex(table), table)
    found = []
    concept = 0
    for rule in ("whole", "head", "inner"):
        while True:
            if concept >= 4096:
                raise RuntimeError(f"no witness for {rule} below concept 4096")
            form = _candidate(ph, concept)
            concept += 1
            if form is None:
                continue
            if rule == "whole":
                found.append((concept - 1, len(form), form))
                break
            if rule == "head" and len(form) >= 4:
                found.append((concept - 1, 3, form[:3]))
                break
            if rule == "inner" and len(form) >= 10:
                found.append((concept - 1, 8, form[1:9]))
                break
    entries = sorted([length, hashlib.sha256(text.encode("ascii")).hexdigest()] for _c, length, text in found)
    return [c for c, _l, _t in found], entries


def render(value, indent=0):
    flat = isinstance(value, list) and all(not isinstance(v, (dict, list)) for v in value)
    if flat or not isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list) and all(isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v)
                                       for v in value):
        pad = "  " * (indent + 1)
        return "[\n" + ",\n".join(pad + json.dumps(v) for v in value) + "\n" + "  " * indent + "]"
    pad = "  " * (indent + 1)
    end = "  " * indent
    if isinstance(value, dict):
        items = [f"{pad}{json.dumps(key)}: {render(value[key], indent + 1)}" for key in sorted(value)]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    items = [f"{pad}{render(item, indent + 1)}" for item in value]
    return "[\n" + ",\n".join(items) + "\n" + end + "]"


def author():
    """Every authored file, ``{file name: text}``; writes nothing."""
    table = phon_table()
    defects = phon.validate_table(wire.parse(wire.emit(table)))
    if defects:
        raise ValueError(defects[:5])
    _concepts, entries = witnesses(table)
    fixture = {"entries": entries, "name": "taboo_fixture_v1", "version": 1}
    if phon.validate_taboo(fixture, phon.Table(table)):
        raise ValueError("taboo_fixture_v1")
    return {
        "phon_v1.json": render(table) + "\n",
        "taboo_fixture_v1.json": render(fixture) + "\n",
    }


def empty_taboo():
    return render({"entries": [], "name": "taboo_v1", "version": 1}) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the files on disk, write nothing")
    args = parser.parse_args(argv)
    files = author()
    stale = []
    for name in sorted(files):
        path = TABLES_DIR.joinpath(name)
        current = path.read_text(encoding="ascii") if path.exists() else None
        if current != files[name]:
            stale.append(name)
            if not args.check:
                path.write_text(files[name], encoding="ascii")
    full = TABLES_DIR.joinpath("taboo_v1.json")
    if not full.exists() and not args.check:
        full.write_text(empty_taboo(), encoding="ascii")
        stale.append("taboo_v1.json (created empty)")
    if args.check:
        print("stale: " + ", ".join(stale) if stale else "every authored file is current")
        return 1 if stale else 0
    print("written: " + ", ".join(stale) if stale else "nothing to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
