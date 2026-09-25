#!/usr/bin/env python3
"""Author the componion's table of journal kinds, and the daily budgets the laws pin.

Writes ``opti_oignon/allium/tables/journal_v1.json``: every kind a being's
journal may hold, each with

* ``scope`` -- ``trunk`` (a fact of the chain, shared with the being's other
  devices) or ``rhythm`` (a row of the local rhythm layer, never a fact);
* ``body`` -- the schema of its body, or ``null`` for a reserved kind: one
  whose name is fixed now and whose body is written by the work that brings
  it, so appending it is refused until then;
* ``payload`` -- the schema of the plaintext a redactable body carries
  sealed apart, by reference, or ``null``;
* ``producer`` -- who alone may write it;
* ``redact_by`` -- the kind whose fact destroys its payload, or ``null``.

A body schema maps each field to its spec: ``int`` (``lo``..``hi``),
``bool``, ``symbol`` (one of a sorted list), ``hex`` (lowercase, ``len``
characters), ``text`` (printable ASCII, 1..``max`` characters, no space at
either end) or ``object`` (nested ``fields``). A payload spec is ``word``:
lowercase letters, 1..``max`` of them.

The daily budget of each budgeted trunk kind is law data, not table data:
``BUDGETS`` is exported here, and ``scripts/allium_author_genome.py``
writes it into each law beside the table's name and digest. Run this script
before that one. ``--check`` writes nothing and exits 1 when the file on
disk differs.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT.joinpath("opti_oignon", "allium", "tables")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opti_oignon.allium import wire  # noqa: E402

NAME = "journal_v1"
MAX_INT = wire.MAX_INT
SCOPES = ("rhythm", "trunk")
PRODUCERS = ("claim", "membrane", "recorder", "resume", "rhythm", "sow")
BUDGET_MAX = 4096
_KIND = re.compile(r"[a-z0-9_]{1,32}")


def _int(lo, hi):
    return {"hi": hi, "lo": lo, "type": "int"}


def _hex(length):
    return {"len": length, "type": "hex"}


def _symbol(*members):
    return {"of": sorted(members), "type": "symbol"}


def _text(most):
    return {"max": most, "type": "text"}


def _object(fields):
    return {"fields": fields, "type": "object"}


BOOL = {"type": "bool"}

GENESIS = {
    "birth": _object({"tz": _int(-840, 840), "wall": _int(0, MAX_INT)}),
    "derive": _int(1, 1),
    "laws": _object({"name": _text(32), "provisional": BOOL, "sha256": _hex(64), "v": _int(0, 65535)}),
    "owner": _hex(32),
    "rhythm_consent": BOOL,
    "seed": _hex(64),
    "soil": _symbol("encrypted", "glass"),
}

# kind: (scope, body schema, payload spec, redact_by, producer)
DEFINED = {
    "genesis": ("trunk", GENESIS, None, None, "sow"),
    "act": ("trunk", {"act": _symbol("greet", "play", "touch", "warm", "water")}, None, None, "membrane"),
    "dream_depth": ("trunk", {"depth": _symbol("deep", "normal", "shallow")}, None, None, "membrane"),
    "forget_rhythm": ("trunk", {}, None, None, "membrane"),
    "lang_forget": ("trunk", {"target": _hex(64)}, None, None, "membrane"),
    "lang_taboo_add": ("trunk", {"len": _int(1, 12), "sha256": _hex(64)}, None, None, "membrane"),
    "lang_teach": ("trunk", {"payload": _hex(32)}, {"max": 24, "type": "word"}, "lang_forget", "membrane"),
    "move_pot": ("trunk", {}, None, None, "membrane"),
    "name": ("trunk", {"name": _text(32)}, None, None, "membrane"),
    "owner": ("trunk", {"from": _hex(32), "to": _hex(32)}, None, None, "claim"),
    "rest_begin": ("trunk", {}, None, None, "membrane"),
    "rest_end": ("trunk", {}, None, None, "membrane"),
    "resumed": ("trunk", {"digest": _hex(64), "removed": _int(0, MAX_INT)}, None, None, "resume"),
    "presence_hour": ("rhythm", {"active": BOOL, "observed_hour": _int(0, 23)}, None, None, "rhythm"),
}

# Reserved trunk kinds: kind -> (redact_by, producer). Their bodies come with the work that brings them.
RESERVED = {
    "braid": (None, "membrane"),
    "bury": (None, "membrane"),
    "celebrate": (None, "membrane"),
    "clock": (None, "recorder"),
    "evolve": (None, "membrane"),
    "excavate": (None, "membrane"),
    "keep_dream": (None, "membrane"),
    "lang_borrow": ("lang_forget", "membrane"),
    "lang_insist": (None, "membrane"),
    "lang_reaction": (None, "membrane"),
    "lang_talk": (None, "membrane"),
    "laws_pin": (None, "membrane"),
    "laws_unpin": (None, "membrane"),
    "letter_sealed": (None, "membrane"),
    "light_day": (None, "membrane"),
    "meeting_in": (None, "membrane"),
    "offering_verdict": (None, "membrane"),
    "pollen_in": (None, "membrane"),
    "rhythm_seal": (None, "membrane"),
    "seal_broken": (None, "membrane"),
    "seed_in": (None, "membrane"),
    "tz": (None, "recorder"),
}

# Trunk kinds no daily budget applies to: written once, by a closed producer, or by the recorder.
EXEMPT = ("clock", "genesis", "owner", "resumed", "tz")

# Facts a day, per kind, under each law. The fixture's are small so that a contract can saturate them.
_V0_1 = {
    "act": 128, "dream_depth": 4, "forget_rhythm": 2, "lang_forget": 16, "lang_taboo_add": 16, "lang_teach": 16,
    "move_pot": 8, "name": 4, "rest_begin": 4, "rest_end": 4,
    "lang_reaction": 64, "lang_talk": 64, "lang_insist": 16, "lang_borrow": 8, "offering_verdict": 32,
    "keep_dream": 4, "braid": 4, "excavate": 4, "celebrate": 4, "letter_sealed": 4, "seal_broken": 4,
    "bury": 2, "laws_pin": 2, "laws_unpin": 2, "evolve": 2, "light_day": 2, "rhythm_seal": 2,
    "seed_in": 2, "pollen_in": 2, "meeting_in": 4,
}
_FIXTURE = {kind: 8 for kind in _V0_1}
_FIXTURE["act"] = 64
_FIXTURE["move_pot"] = 3
BUDGETS = {
    "fixture": {kind: _FIXTURE[kind] for kind in sorted(_FIXTURE)},
    "v0_1": {kind: _V0_1[kind] for kind in sorted(_V0_1)},
}


def journal_table():
    kinds = {}
    for kind, (scope, body, payload, redact_by, producer) in DEFINED.items():
        kinds[kind] = {"body": body, "payload": payload, "producer": producer, "redact_by": redact_by,
                       "scope": scope}
    for kind, (redact_by, producer) in RESERVED.items():
        kinds[kind] = {"body": None, "payload": None, "producer": producer, "redact_by": redact_by,
                       "scope": "trunk"}
    return {"kinds": {kind: kinds[kind] for kind in sorted(kinds)}, "name": NAME, "schema": 1}


def _field_names(schema):
    """Every field name of a body schema, at any depth."""
    names = []
    for name, spec in schema.items():
        names.append(name)
        if spec.get("type") == "object":
            names.extend(_field_names(spec["fields"]))
    return names


def defects(table):
    """What the authoring tables get wrong, by name; empty when the table and the budgets agree."""
    found = []
    kinds = table["kinds"]
    for kind, entry in kinds.items():
        body = entry["body"]
        if _KIND.fullmatch(kind) is None:
            found.append(f"{kind}: name")
        if entry["scope"] not in SCOPES or entry["producer"] not in PRODUCERS:
            found.append(f"{kind}: scope or producer")
        if entry["scope"] == "rhythm" and (entry["producer"] != "rhythm" or entry["redact_by"] is not None):
            found.append(f"{kind}: rhythm")
        if (entry["payload"] is not None) != (body is not None and "payload" in body):
            found.append(f"{kind}: payload")
        if body is not None and entry["redact_by"] is not None and body.get("payload") != _hex(32):
            found.append(f"{kind}: redactable without a payload reference")
        target = kinds.get(entry["redact_by"]) if entry["redact_by"] is not None else None
        if entry["redact_by"] is not None and (target is None or target["scope"] != "trunk"
                                              or target["redact_by"] is not None
                                              or target["body"] != {"target": _hex(64)}):
            found.append(f"{kind}: redact_by")
        if entry["scope"] == "trunk" and body is not None \
                and any(name in ("hour", "observed_hour") for name in _field_names(body)):
            found.append(f"{kind}: an hour in the trunk")
    budgeted = sorted(kind for kind, entry in kinds.items() if entry["scope"] == "trunk" and kind not in EXEMPT)
    for law, budgets in BUDGETS.items():
        if sorted(budgets) != budgeted:
            found.append(f"{law}: budget keys")
        for kind, budget in budgets.items():
            if not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= BUDGET_MAX:
                found.append(f"{law}: budget {kind}")
    return found


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
    table = journal_table()
    if wire.parse(wire.emit(table)) != table:
        raise ValueError(f"{NAME}: not canonical")
    found = defects(table)
    if found:
        raise ValueError(found[:5])
    return {f"{NAME}.json": render(table) + "\n"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the file on disk, write nothing")
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
    if args.check:
        print("stale: " + ", ".join(stale) if stale else "every authored file is current")
        return 1 if stale else 0
    print("written: " + ", ".join(stale) if stale else "nothing to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
