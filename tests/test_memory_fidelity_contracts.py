#!/usr/bin/env python3
"""Contracts for the two fixture metrics of the onion memory: compression
fidelity and the effective-context multiplier.

Fidelity resamples every peel against its Cellar spans: probes are
regenerated from the source and scored against the peel's text, and the
rate is what survives. The multiplier divides the source tokens a set of
root peels stands for by the tokens those peels cost in the window. Both
are produced by the code as it stands over injected summarisers, so both
carry ``source: "fixture"`` and neither is a claim about real recall
quality; the real summariser is a host measurement.

  * FM1 -- fidelity is 1.0 on a faithful tree, below 1.0 on a lossy one,
    and unknown (None, never 0.0) on an empty one; every figure says it is
    a fixture reading.
  * FM2 -- the multiplier is above one when the summariser shrinks, the
    source token count is the sum over the spans, and it is unknown when
    there is no peel.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _open():
    loaded, restore = isolate(
        targets={
            "opti_oignon.memory.probes": source("memory", "probes.py"),
            "opti_oignon.memory.receipts": source("memory", "receipts.py"),
            "opti_oignon.memory.peels": source("memory", "peels.py"),
        },
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.receipts"], restore


def _span(n):
    return [
        {"turn_id": f"s{n}a", "role": "user", "text": f"Service {n} was migrated by Alice on 2026-0{n}-10 with 1{n}0 connections and the dashboards were checked twice."},
        {"turn_id": f"s{n}b", "role": "assistant", "text": f"We agreed that service {n} stays on the new cluster unless the error rate rises."},
    ]


def _faithful(span):
    return " ".join(t["text"] for t in span)


def _short(span):
    return " ".join(t["text"].replace(" and the dashboards were checked twice", "") for t in span)


def _tree(peels, receipts, summarize, *, gate=None):
    gate = gate or peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
    cellar, tree = receipts.Cellar(), peels.PeelTree()
    for n in (1, 2, 3):
        peel, decision = peels.build_leaf(cellar.store(_span(n)), cellar, summarize, gate, tree)
        assert decision.accepted, decision.reason
    return cellar, tree


# ---------------------------------------------------------------------------
# FM1 -- fidelity
# ---------------------------------------------------------------------------
def test_fm1_fidelity_is_a_resample_against_the_cellar_and_unknown_when_empty():
    peels, receipts, restore = _open()
    try:
        cellar, tree = _tree(peels, receipts, _faithful)
        faithful = peels.fidelity(tree, cellar)
        assert faithful["source"] == "fixture"
        assert faithful["peels"] == 3 and faithful["probes"] >= 6
        assert faithful["rate"] == 1.0
        lossy = peels.Gate(decision_threshold=0.5, episodic_threshold=0.5, span_turns=2)
        cellar, tree = _tree(peels, receipts, lambda span: _faithful(span).replace("Alice", "Dave"), gate=lossy)
        swapped = peels.fidelity(tree, cellar)
        assert 0.0 < swapped["rate"] < 1.0
        assert swapped["failed"] == 3, "one entity per peel, three peels"
        empty = peels.fidelity(peels.PeelTree(), cellar)
        assert empty["rate"] is None and empty["peels"] == 0, "no peel means unknown, never 0.0"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FM2 -- multiplier
# ---------------------------------------------------------------------------
def test_fm2_the_multiplier_is_source_tokens_over_peel_tokens():
    peels, receipts, restore = _open()
    try:
        cellar, tree = _tree(peels, receipts, _short)
        m = peels.context_multiplier(tree, cellar)
        assert m["source"] == "fixture"
        expected_source = sum(peels.estimate_tokens(t["text"]) for n in (1, 2, 3) for t in _span(n))
        assert m["source_tokens"] == expected_source
        assert m["peel_tokens"] == sum(peels.estimate_tokens(p.text) for p in tree.roots())
        assert m["multiplier"] > 1.0
        assert m["multiplier"] == round(m["source_tokens"] / m["peel_tokens"], 4)
        cellar, tree = _tree(peels, receipts, _faithful)
        same = peels.context_multiplier(tree, cellar)
        assert same["multiplier"] == 1.0, "a faithful copy multiplies nothing"
        none = peels.context_multiplier(peels.PeelTree(), cellar)
        assert none["multiplier"] is None and none["peel_tokens"] == 0
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
