#!/usr/bin/env python3
"""Contracts for the baseline measurement of the existing dual-layer memory.

Before the memory block changes anything, it records what the current
mechanism does on corpora it names: how much a set of natural queries recalls
through the real retriever, how much of the salience floor survives, how many
facts the prompt budget drops and how many of those the archive search finds
again, and how far the rule-based compressor shrinks a conversation. Every
number is produced by the code as it stands, over recorder stores, with no
vector signal -- the keyword path alone -- so it is deterministic and can be
reproduced to the digit. The recorded figures are the baseline the block
will be measured against.

  * BL1 -- every named corpus is non-trivial, and every measure it yields is
    a finite number in its range.
  * BL2 -- the instrument is capable: recall is above zero, the compression
    ratio is a positive number, and the budget drops something on the long
    corpus. Whether the rule compressor shrinks a conversation is NOT
    assumed: the first measurement found it expanding the short corpus
    (a ratio above one), and that is the baseline's first finding, recorded
    rather than tuned away.
  * BL3 -- the measurement is reproducible and matches the recorded baseline
    exactly.
  * BL4 -- what the budget drops, the archive finds: the dual-layer
    invariant as a measured rate, not an assertion.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window with the real retriever and compressor as targets; the token
window and the conversation store are declared unreachable, so the retriever
takes its documented fallback estimator and the compressor its rule path.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_RANGES = {
    "recall_rate": (0.0, 1.0),
    "floor_retention": (0.0, 1.0),
    "recovery_rate": (0.0, 1.0),
    # A ratio, not a fraction: the rule compressor can and does exceed one.
    "rule_ratio": (0.0, 4.0),
}


def _open():
    loaded, restore = isolate(
        targets={
            "opti_oignon.memory.retrieval": source("memory", "retrieval.py"),
            "opti_oignon.conversation_compressor": source("conversation_compressor.py"),
            "opti_oignon.memory.baseline": source("memory", "baseline.py"),
        },
        blocked=(
            "opti_oignon.context_window",
            "opti_oignon.context_manager",
            "opti_oignon.conversation",
        ),
        packages=("opti_oignon.memory",),
    )
    return loaded["opti_oignon.memory.baseline"], restore


# ---------------------------------------------------------------------------
# BL1 -- corpora are non-trivial and measures are in range
# ---------------------------------------------------------------------------
def test_bl1_every_corpus_is_non_trivial_and_every_measure_in_range():
    mod, restore = _open()
    try:
        assert len(mod.CORPORA) >= 2, "more than one corpus, so no single shape is the baseline"
        for name in mod.CORPORA:
            corpus = mod.CORPORA[name]
            assert len(corpus.facts) >= 8, f"{name}: enough facts to mean something"
            assert len(corpus.queries) >= 4, f"{name}: enough queries to rate recall"
            measures = mod.measure(name)
            for key, (lo, hi) in _RANGES.items():
                value = measures[key]
                assert isinstance(value, float) and math.isfinite(value), f"{name}.{key} is a finite number"
                assert lo <= value <= hi, f"{name}.{key}={value} within [{lo}, {hi}]"
            assert measures["facts"] == len(corpus.facts)
            assert measures["budget_dropped"] >= 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# BL2 -- the instrument is capable of non-zero
# ---------------------------------------------------------------------------
def test_bl2_the_instrument_reads_something():
    mod, restore = _open()
    try:
        for name in mod.CORPORA:
            m = mod.measure(name)
            assert m["recall_rate"] > 0.0, f"{name}: the retriever recalls something"
            assert m["rule_ratio"] > 0.0, (
                f"{name}: the compressor produced text, so the ratio is a "
                "reading; whether it is below one is a finding, not a premise"
            )
        long_run = mod.measure(mod.LONG_CORPUS)
        assert long_run["budget_dropped"] >= 1, (
            "the long corpus exceeds the prompt budget, so the budget cut is exercised"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# BL3 -- reproducible, and equal to the recorded baseline
# ---------------------------------------------------------------------------
def test_bl3_the_measurement_reproduces_the_recorded_baseline():
    mod, restore = _open()
    try:
        assert set(mod.BASELINE) == set(mod.CORPORA), "one recorded entry per corpus"
        for name in mod.CORPORA:
            first = mod.measure(name)
            second = mod.measure(name)
            assert first == second, f"{name}: two runs agree to the digit"
            assert first == mod.BASELINE[name], (
                f"{name}: the code as it stands reproduces the recorded baseline; "
                f"a difference means the mechanism moved and the record must be "
                f"re-taken on purpose, never edited to fit -- got {first}"
            )
    finally:
        restore()


# ---------------------------------------------------------------------------
# BL4 -- what the budget drops, the archive finds
# ---------------------------------------------------------------------------
def test_bl4_what_the_budget_drops_the_archive_finds():
    mod, restore = _open()
    try:
        m = mod.measure(mod.LONG_CORPUS)
        assert m["budget_dropped"] >= 1, "control: something was dropped"
        assert m["recovery_rate"] == 1.0, (
            "every fact the prompt budget dropped is found by the archive search: "
            "the dual-layer invariant, as a rate"
        )
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
