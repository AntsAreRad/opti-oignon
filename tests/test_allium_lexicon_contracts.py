#!/usr/bin/env python3
"""Contract for the componion's language draws: golden, replayable, unbiased.

  * LX1 -- the keyed draws of the language equal their committed goldens --
    each golden founder's block, inventory, first sound, twenty coinages and
    six-word list -- in this process and in two children under different
    string hash seeds; ``pick`` rejects exactly the words that would bias it
    and maps the first accepted word through its running sum; and the
    edit-distance-one test behind "too close to a word it has" holds its
    golden pairs, the same in both engines.

Local-only; the engine comparison needs the native artefact.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_phon_support as S  # noqa: E402
from _allium_window import open_allium  # noqa: E402
from _isolation import REPO  # noqa: E402

GOLDEN = REPO / "tests" / "allium_golden" / "v1" / "phon.json"

BUDGET_S = {
    "test_lx1_the_language_draws_equal_their_goldens_and_pick_is_unbiased": 2.0,
}

_REPLAY = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from opti_oignon.allium import rng, wire
from opti_oignon.allium.ref import protocol as p
def ask(obj):
    return wire.parse(p.call(wire.emit(dict(obj, v=1))))
def raw(obj):
    return p.call(wire.emit(dict(obj, v=1)))
SEEDS = {"fixture": [bytes(range(32)).hex(), "ff" * 32, "a5" * 32], "v0_1": [bytes(range(32)).hex(), "ff" * 32]}
MASK = sum(1 << b for b in (0, 2, 4, 5, 7, 9, 17, 21, 22, 23))
GOLDEN_DIGEST = hashlib.sha256(b"sas-golden").hexdigest()
out = {}
for law, seeds in SEEDS.items():
    rows = []
    for seed in seeds:
        genome = ask({"law": law, "op": "genome_found", "seed": seed})["genome"]
        lex = ask({"genome": genome, "law": law, "op": "phon_lex"})["lex"]
        inventory = ask({"law": law, "lex": [lex], "op": "phon_inventory"})["out"][0]
        first = ask({"cases": [{"lex": lex, "seed": seed}], "law": law, "op": "phon_first_sound"})["out"][0]
        cases = []
        for c in range(20):
            case = {"anchored": [], "coin": 0, "concept": c, "epoch": 0, "lex": lex, "others": [], "seed": seed,
                    "signs": [((c + f) % 3) - 1 for f in range(4)], "syllables": 0}
            if c >= 16:
                case["inventory"] = MASK
            cases.append(case)
        invent = raw({"cases": cases, "law": law, "op": "phon_invent"})
        sas = raw({"digests": [GOLDEN_DIGEST], "law": law, "lex": lex, "op": "phon_sas"})
        rows.append({"first_sound": "%02x" % first, "invent_sha256": hashlib.sha256(invent).hexdigest(),
                     "lex": lex, "mask": "%08x" % inventory["mask"], "sas_sha256": hashlib.sha256(sas).hexdigest(),
                     "seed": seed})
    out[law] = rows
table = ask({"law": "fixture", "op": "phon_table"})
lo = "".join("%02x" % pair[0] for pair in table["lex_box"])
floors = ask({"law": "fixture", "lex": [lo], "op": "phon_inventory"})["out"][0]["floors"]
out["lo_floors"] = "".join("%02x" % f for f in floors)
words = {}
for domain in ("lang.fallback", "lang.first_sound", "lang.invent", "lang.sas"):
    stream = rng.Stream.from_key(rng.key(bytes(32), domain, (0,)))
    words[domain] = ["%016x" % stream.next_u64() for _ in range(8)]
out["from_key"] = words
print(json.dumps(out, sort_keys=True))
"""


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


def _unfiltered(e, case):
    return e.ask({"cases": [case], "law": "v0_1", "op": "phon_invent"})["out"][0]


def test_lx1_the_language_draws_equal_their_goldens_and_pick_is_unbiased(tmp_path):
    golden = json.loads(GOLDEN.read_text(encoding="ascii"))
    assert len(golden["fixture"]) == 3 and len(golden["v0_1"]) == 2 and len(golden["from_key"]) == 4
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run([sys.executable, "-c", _REPLAY, str(REPO)], cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stderr[-2000:]
        assert json.loads(run.stdout) == golden, f"the language draws move under PYTHONHASHSEED={hash_seed}"
    assert list(tmp_path.iterdir()) == [], "the replay wrote nothing"

    loaded, restore = open_allium(native=True)
    try:
        e = S.Engines(loaded, native=True)
        rng, phon = e.rng, e.phon
        for domain, words in golden["from_key"].items():
            stream = rng.Stream.from_key(rng.key(bytes(32), domain, (0,)))
            assert ["%016x" % stream.next_u64() for _ in range(8)] == words, domain

        class Scripted(rng.Stream):
            __slots__ = ("words", "used")

            def next_u64(self):
                self.used += 1
                return self.words.pop(0)

        rejected = 0
        for total in range(1, 65):
            threshold = ((1 << 64) - total) % total
            assert ((1 << 64) - threshold) % total == 0, total
            stream = Scripted.from_key(bytes(32))
            stream.used = 0
            stream.words = ([threshold - 1] if threshold else []) + [threshold]
            index = phon.pick(stream, [1] * total)
            assert index == threshold % total, (total, index)
            assert stream.used == (2 if threshold else 1), total
            rejected += bool(threshold)
            weights = [0] * total
            weights[-1] = 1
            stream = Scripted.from_key(bytes(32))
            stream.used = 0
            stream.words = [threshold]
            assert phon.pick(stream, weights) == total - 1
        assert rejected >= 40, rejected

        for a, b, near in (("ab", "ba", False), ("abd", "abcd", True), ("abd", "abce", False), ("abc", "abcd", True),
                           ("abc", "abc", False), ("abc", "xbc", True), ("abc", "abcde", False)):
            assert phon.near1(a, b) is near and phon.near1(b, a) is near, (a, b)
        table = e.table()
        lex = S.hi_block(table).hex()
        near_seen = 0
        for concept in range(40):
            case = {"anchored": [], "coin": 0, "concept": concept, "epoch": 0, "lex": lex, "others": [],
                    "seed": "00" * 32, "signs": [0, 0, 0, 0], "syllables": 0}
            out = _unfiltered(e, case)
            if out["outcome"] != "word" or out["tries"] or len(out["form"]) < 3:
                continue
            form = out["form"]
            variants = (form[:-1], form + "a", ("e" if form[0] != "e" else "o") + form[1:], form[1] + form[0] + form[2:])
            for variant, expected in zip(variants, (True, True, True, False)):
                if variant == form:
                    continue
                answer = e.both({"cases": [dict(case, anchored=[variant])], "law": "v0_1", "op": "phon_invent"})
                first = answer["out"][0]["tries"][:1]
                got_near = bool(first) and first[0] == [S.sha(form), "near"]
                if expected:
                    assert got_near, (variant, answer)
                    near_seen += 1
                elif phon.near1(form, variant) is False:
                    assert not got_near, (variant, answer)
        assert near_seen >= 30, near_seen
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
