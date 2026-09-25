#!/usr/bin/env python3
"""Author the componion's golden lives: thirty-day lives on both laws, answered by the reference.

Writes, under ``tests/allium_golden/v1/``:

* ``life_fixture.json`` -- four lives on the fixture law: ``garden_north``
  (waters, the time zone moved twice, the first day of winter on day 25),
  ``windowsill_south`` (a drought, then a water after the rest),
  ``garden_evolve`` (a params ``evolve``) and ``pinned`` (a pin on day 3, an
  ``evolve`` on day 5 met while pinned, the pin lifted on day 10, an
  ``evolve`` on day 12);
* ``life_v0_1.json`` -- two lives on the full law: ``garden``, and
  ``windowsill`` under the short-day band.

Each entry is one ``advance`` request from the genesis to minute ``at``
(``30 * 1440``), written as the hexadecimal of its canonical bytes, with the
law's digest, the SHA-256 of the reference's response and of the state it
returns, and the work and overhead it counts. Every string is hexadecimal.
The fixture lives ask for the trace, so the unit accounting is pinned with
them; the full law's lives do not, so the served form of a response is
pinned as it is.

Usage: ``python3 scripts/allium_author_life.py [--check | --write]``.
``--check``, the default, recomputes every life with the reference, writes
nothing, and exits 1 when a file on disk differs. ``--write`` writes the
files, and refuses -- writing nothing, exit 2 -- to re-record an entry whose
law digest is unchanged but whose answer changed: the organs' code changed
under an unchanged law, and the law's ``code`` revision must be bumped
first. It refuses in the same way a life whose request changed, or a life
dropped, while its law's digest stayed the same: a golden life is
re-recorded with its law, never on its own.

The engines never read these files, and neither does the application. The
golden lives' contract replays each committed request in the reference and
in the native core, and runs ``main(["--check"])`` in-process.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT.joinpath("tests", "allium_golden", "v1")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opti_oignon.allium import lawfiles, rng, wire  # noqa: E402
from opti_oignon.allium.ref import civil, protocol  # noqa: E402

DAY = 1440
AT = 30 * DAY
MAX_INT = (1 << 53) - 1
WALL = 1760000000
DOMAIN = "golden.life"
FILES = {"fixture": "life_fixture.json", "v0_1": "life_v0_1.json"}
# What a changed answer is read from; the request is compared on its own.
ANSWER_KEYS = ("at", "overhead", "response_sha256", "state_sha256", "work")


class Life:
    """A genesis and its facts at the engine level, drawn from a fixed stream."""

    def __init__(self, law, index, *, weather, hemisphere, band, tz, trace, wall=WALL):
        draw = rng.Stream(bytes(32), DOMAIN, index)
        words = [draw.next_u64() for _ in range(9)]
        self.being = "".join(format(word, "016x") for word in words[0:2])
        self.origin = format(words[2], "016x")
        value = lawfiles.law(law)
        self.law = law
        self.v = value["version"]
        self.b = wall // 60
        self.tz = tz
        self.offset = tz
        self.trace = trace
        self.genesis = {
            "being": self.being,
            "body": {
                "band": band,
                "birth": {"tz": tz, "wall": wall},
                "derive": 1,
                "hemisphere": hemisphere,
                "laws": {"name": law, "params": self.defaults(), "provisional": value["provisional"],
                         "sha256": lawfiles.digest(value), "v": value["version"]},
                "owner": "".join(format(word, "016x") for word in words[7:9]),
                "rhythm_consent": False,
                "seed": "".join(format(word, "016x") for word in words[3:7]),
                "soil": "encrypted",
                "weather": weather,
            },
            "kind": "genesis",
            "laws": value["version"],
            "origin": self.origin,
            "oseq": 0,
            "t": 0,
        }
        self.facts = []

    def defaults(self):
        return {name: spec["default"] for name, spec in lawfiles.law(self.law)["params"].items()}

    def at(self, day, minute):
        """The minute of life of local minute ``minute`` on the ``day``-th local day, under the offset now in force."""
        first = (self.b + self.tz) // DAY
        return (first + day) * DAY + minute - self.offset - self.b

    def fact(self, t, kind, body):
        self.facts.append({"being": self.being, "body": body, "kind": kind, "laws": self.v, "origin": self.origin,
                           "oseq": len(self.facts) + 1, "t": t})
        if [f["t"] for f in self.facts] != sorted(f["t"] for f in self.facts):
            raise ValueError(f"the facts of a golden life are written in order: minute {t}")

    def act(self, day, minute, act):
        self.fact(self.at(day, minute), "act", {"act": act})

    def move(self, day, minute, offset):
        """A ``tz`` fact; the offset it names is in force from its minute on."""
        self.fact(self.at(day, minute), "tz", {"quarters": offset // 15})
        self.offset = offset

    def pin(self, day, minute):
        self.fact(self.at(day, minute), "laws_pin", {})

    def unpin(self, day, minute):
        self.fact(self.at(day, minute), "laws_unpin", {})

    def evolve(self, day, minute, **changes):
        """A params ``evolve`` that keeps the law, due at the next local midnight."""
        t = self.at(day, minute)
        law = self.genesis["body"]["laws"]
        params = self.defaults()
        params.update(changes)
        self.fact(t, "evolve", {"effective_from": civil.next_midnight(self.b, t, self.offset),
                                "from": {"name": law["name"], "sha256": law["sha256"]}, "params": params,
                                "to": {"name": law["name"], "sha256": law["sha256"], "v": law["v"]}})

    def request(self):
        request = {"budget": MAX_INT, "facts": self.facts, "genesis": self.genesis, "op": "advance",
                   "state": None, "to": AT, "v": 1}
        if self.trace:
            request["probe"] = {"trace": True}
        return request


# ---------------------------------------------------------------------------
# The lives
# ---------------------------------------------------------------------------

def garden_north():
    life = Life("fixture", 0, weather="garden", hemisphere="north", band="long", tz=0, trace=True)
    life.act(2, 8 * 60 + 12, "water")
    life.act(6, 8 * 60 + 40, "water")
    life.move(8, 14 * 60 + 7, 60)
    life.act(11, 7 * 60 + 53, "water")
    life.move(15, 20 * 60 + 31, 345)
    life.act(17, 9 * 60 + 2, "water")
    life.act(21, 7 * 60 + 45, "water")
    life.act(28, 12 * 60 + 30, "greet")
    return life


def windowsill_south():
    # Born six days later than the others, in the southern winter: no season turns to winter in its thirty days,
    # so the sleep is the drought's, and the water after its rest wakes it.
    life = Life("fixture", 1, weather="windowsill", hemisphere="south", band="medium", tz=-300, trace=True,
                wall=WALL + 6 * 86400)
    life.act(3, 18 * 60 + 20, "greet")
    life.act(24, 9 * 60 + 14, "water")
    return life


def garden_evolve():
    life = Life("fixture", 2, weather="garden", hemisphere="north", band="short", tz=345, trace=True)
    life.act(3, 7 * 60 + 31, "water")
    life.act(9, 8 * 60 + 5, "water")
    life.evolve(10, 12 * 60 + 7, evap_awake=6000, evap_dormant=2048, rain_gain=90000, sun_max=60000)
    life.act(14, 19 * 60 + 44, "water")
    life.act(18, 10 * 60 + 10, "warm")
    return life


def pinned():
    life = Life("fixture", 3, weather="garden", hemisphere="north", band="medium", tz=60, trace=True)
    life.act(2, 8 * 60 + 2, "water")
    life.pin(3, 9 * 60)
    life.evolve(5, 14 * 60 + 5, evap_awake=12000)
    life.act(8, 17 * 60 + 36, "water")
    life.unpin(10, 9 * 60)
    life.evolve(12, 10 * 60 + 12, evap_awake=2000, sun_max=40000)
    life.act(14, 8 * 60 + 50, "water")
    return life


def v0_1_garden():
    life = Life("v0_1", 4, weather="garden", hemisphere="north", band="medium", tz=60, trace=False)
    for day in (2, 5, 9):
        life.act(day, 7 * 60 + 3 * day, "water")
    life.act(11, 16 * 60 + 25, "warm")
    life.act(13, 7 * 60 + 39, "water")
    life.act(18, 7 * 60 + 54, "water")
    life.act(20, 12 * 60 + 1, "greet")
    life.act(22, 8 * 60 + 6, "water")
    life.act(27, 8 * 60 + 21, "water")
    return life


def v0_1_windowsill():
    life = Life("v0_1", 5, weather="windowsill", hemisphere="south", band="short", tz=-240, trace=False)
    for day in (1, 4, 8, 12):
        life.act(day, 18 * 60 + 5 * day, "water")
    life.act(14, 9 * 60 + 9, "greet")
    for day in (16, 20, 25, 29):
        life.act(day, 18 * 60 + 5 * day, "water")
    return life


LIVES = {
    "fixture": {"garden_evolve": garden_evolve, "garden_north": garden_north, "pinned": pinned,
                "windowsill_south": windowsill_south},
    "v0_1": {"garden": v0_1_garden, "windowsill": v0_1_windowsill},
}


# ---------------------------------------------------------------------------
# Authoring
# ---------------------------------------------------------------------------

def entry(life):
    """One golden entry: the request's bytes and what the reference answers to them."""
    data = wire.emit(life.request())
    answer_bytes = protocol.call(data)
    answer = wire.parse(answer_bytes)
    if "refused" in answer or not answer["done"] or answer["at"] != AT:
        raise ValueError(f"a golden life must be lived to its minute: {answer_bytes[:200]!r}")
    return {
        "at": answer["at"],
        "law_sha256": lawfiles.digest(lawfiles.law(life.law)),
        "overhead": answer["overhead"],
        "request": data.hex(),
        "response_sha256": hashlib.sha256(answer_bytes).hexdigest(),
        "state_sha256": answer["hash"],
        "work": answer["work"],
    }


def author():
    """Every golden file's entries, ``{law: {life: entry}}``; writes nothing."""
    return {law: {name: entry(build()) for name, build in sorted(lives.items())} for law, lives in LIVES.items()}


def render(entries):
    return json.dumps(entries, indent=2, sort_keys=True) + "\n"


def refusals(law, fresh, current):
    """Why ``fresh`` may not replace ``current`` (the entries on disk, or None): a list of sentences."""
    if current is None:
        return []
    out = []
    for name, old in sorted(current.items()):
        new = fresh.get(name)
        if new is None:
            if old.get("law_sha256") == lawfiles.digest(lawfiles.law(law)):
                out.append(f"{law}/{name}: a golden life is dropped under an unchanged law")
            continue
        if old.get("law_sha256") != new["law_sha256"]:
            continue
        if old.get("request") != new["request"]:
            out.append(f"{law}/{name}: its request changed under an unchanged law; "
                       "record the new life under a new name, or change the law")
        elif any(old.get(key) != new[key] for key in ANSWER_KEYS):
            out.append(f"{law}/{name}: its answer changed under an unchanged law digest -- the organs' code "
                       "changed under the law, so bump the law's code revision and re-run the authors first")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="recompute and compare, write nothing (the default)")
    mode.add_argument("--write", action="store_true", help="write the files, refusing a re-record the law does not allow")
    args = parser.parse_args(argv)
    entries = author()
    stale = []
    texts = {}
    refused = []
    for law, name in sorted(FILES.items()):
        path = GOLDEN_DIR.joinpath(name)
        current = path.read_text(encoding="ascii") if path.exists() else None
        texts[path] = render(entries[law])
        if current != texts[path]:
            stale.append(name)
            refused += refusals(law, entries[law], None if current is None else json.loads(current))
    if not args.write:
        print("stale: " + ", ".join(stale) if stale else "every golden life is current")
        return 1 if stale else 0
    if refused:
        for line in refused:
            print("refused: " + line)
        return 2
    for path, text in texts.items():
        if path.name in stale:
            path.write_text(text, encoding="ascii")
    print("written: " + ", ".join(stale) if stale else "nothing to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
