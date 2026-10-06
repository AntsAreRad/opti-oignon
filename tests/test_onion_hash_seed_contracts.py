#!/usr/bin/env python3
"""Contracts for the determinism of the onion's readings: the same turns give
the same probes, facts, decisions, digests and selection under any hash seed.

A decision key, a set of reporters or the subjects of a lexicon are sets:
their order is the hash seed's. Whatever the onion writes out in order --
a probe list, a reason, the facts left unasked, a digest, what the native
core is handed -- is read in a fixed order, a set sorted where it is
written, so that two runs of the same turns never differ.

  * HD1 -- under two hash seeds, a run over the labelled fixture set gives
    the same bytes: the probes drawn and their keys, the facts held, the
    gate's decisions with their reasons, claims and unasked facts, the new
    words of a summary, the receipt keys, the peel ids and digests, the
    selection and its scores, and what the native core would be handed.
  * HD2 -- every set the native core is handed arrives sorted: the words
    that are no entities, the function words, the subjects of the lexicon and
    every probe key.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_LABELLED = REPO / "tests" / "recall_set" / "labelled.json"

_REPLAY = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from opti_oignon.memory import peels, probes, receipts
probes._native = lambda: None
gate = peels.load_gate()
spans = [span["turns"] for span in probes.load_labelled(sys.argv[2])]
ADDED = " Then Ada, Bea, Cyd, Dov, Eli and Fay met on 2031-02-03 for 77 GB."
out = {"fingerprint": gate.lexicon.fingerprint,
       "native_lexicon": json.loads(json.dumps(probes._native_lexicon(gate.lexicon))),
       "native_tables": json.loads(json.dumps(probes._native_tables())),
       "spans": []}
cellar, ledger, tree = receipts.Cellar(), receipts.ReceiptLedger(), peels.PeelTree()
for turns in spans:
    drawn = probes.generate_probes(turns, gate.lexicon)
    text = " ".join(t["text"] for t in turns)
    starved = [p for p in drawn if p.kind != "entity"]
    rows = []
    for summary, given in ((text, drawn), (text + ADDED, drawn), (text, starved)):
        d = peels.decide(turns, given, summary, gate)
        rows.append([d.accepted, d.reason, [list(u) for u in d.unsupported], [list(u) for u in d.unasked],
                     d.facts, d.probe_coverage, d.novelty, d.length_ratio])
    receipt = receipts.Flesh([dict(t) for t in turns]).evict_span(len(turns), cellar, ledger)
    tree.add(peels.Peel(id=peels.peel_id(text, [receipt.key]), text=text, level=0, sources=(receipt.key,),
                        children=(), source_digest=peels.source_digest(cellar, [receipt.key]),
                        probes_passed=0, probes_total=0))
    out["spans"].append({
        "probes": [[p.kind, p.answer, p.turn_id, sorted(p.key), p.negated, p.negations, p.origin, p.role,
                    p.canonical] for p in drawn],
        "facts": [[f.kind, f.what, f.turn_id] for f in probes.held_facts(turns, gate.lexicon)],
        "novel": list(probes.novel_words(probes.holdings(turns), text + ADDED, gate.lexicon, gate.reporters)[1]),
        "decisions": rows,
        "receipt": receipt.key,
    })
out["peels"] = [[p.id, p.source_digest] for p in tree.all()]
out["selection"] = {query: [[c.provenance, c.score] for c in peels.select_peels(tree, query, cap=100000)]
                    for query in ("Oslo office budget", "decision Docker cluster", "2026 release review")}
print(json.dumps(out, sort_keys=True, ensure_ascii=True))
"""


def _window():
    loaded, restore = isolate(
        targets={
            "opti_oignon.memory.probes": source("memory", "probes.py"),
            "opti_oignon.memory.receipts": source("memory", "receipts.py"),
            "opti_oignon.memory.peels": source("memory", "peels.py"),
        },
        packages=("opti_oignon.memory",),
    )
    probes = loaded["opti_oignon.memory.probes"]
    probes._native = lambda: None
    return probes, loaded["opti_oignon.memory.peels"], restore


# ---------------------------------------------------------------------------
# HD1 -- the same bytes under two hash seeds
# ---------------------------------------------------------------------------
def test_hd1_a_run_over_the_labelled_set_gives_the_same_bytes_under_two_hash_seeds(tmp_path):
    outputs = []
    for hash_seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run([sys.executable, "-c", _REPLAY, str(REPO), str(_LABELLED)], cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr[-2000:]
        outputs.append(run.stdout)
    assert outputs[0] == outputs[1], "the onion's readings move with the hash seed"
    assert list(tmp_path.iterdir()) == [], "the replay wrote nothing"
    out = json.loads(outputs[0])
    spans = out["spans"]
    assert len(spans) == 18 and len(out["peels"]) == 18
    assert max(len(p[3]) for s in spans for p in s["probes"] if p[0] == "decision") >= 3, "control: a set to order"
    assert max(len(d[2]) for s in spans for d in s["decisions"]) >= 6, "control: claims to order"
    assert any(d[3] for s in spans for d in s["decisions"]), "control: unasked facts to order"
    assert any(s["novel"] for s in spans) and any(out["selection"].values()), "control: words and peels to order"
    assert len(out["native_lexicon"][0]) >= 3, "control: subjects to order"


# ---------------------------------------------------------------------------
# HD2 -- the native core is handed every set sorted
# ---------------------------------------------------------------------------
class _Recorder:
    """A core of the reference's generator that records what it is handed and answers nothing."""

    def __init__(self, version):
        self.probe_generator_version = version
        self.handed = []

    def probe_generate(self, texts, patterns, tables, not_entities, stopwords, markers, lexicon):
        self.handed += [("not entities", not_entities), ("stopwords", stopwords), ("subjects", list(lexicon[0]))]
        return None

    def probe_score(self, rows, text, patterns, tables, coverage, lexicon):
        self.handed += [("key", row[2]) for row in rows] + [("subjects", list(lexicon[0]))]
        return None


_SPAN = [
    {"turn_id": "k1", "role": "user", "origin": "typed",
     "text": "We keep the old cluster on 2026-03-04 with Alice Martin and Bob."},
    {"turn_id": "k2", "role": "assistant", "origin": "assistant", "text": "Noted: 16 Go per host."},
]


def test_hd2_every_set_the_native_core_is_handed_arrives_sorted():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        core = _Recorder(probes.GENERATOR_VERSION)
        probes._native = lambda: core
        drawn = probes.generate_probes(_SPAN, gate.lexicon)
        probes.score(drawn, "We keep the old cluster.")
        assert {name for name, _value in core.handed} == {"not entities", "stopwords", "subjects", "key"}, (
            "control: the core was handed each"
        )
        for name, value in core.handed:
            assert isinstance(value, list) and value == sorted(value), name
        assert max(len(value) for name, value in core.handed if name == "key") >= 3, "control: a key of members"
        assert min(len(value) for name, value in core.handed if name == "subjects") >= 3, "control: subjects"
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
