#!/usr/bin/env python3
"""Contracts for probe-gated eviction: a span leaves the Flesh only once its
peel has answered for it.

The gate generates recall probes from the source span, scores the candidate
summary, and evicts only when every probe class present meets its
threshold. Below the gate the verbatim turns stay and the decision says why,
naming the failed probes. Two rules of the silent-zero family bind it: an
empty probe set on a rich span is a defect, and an unknown rate is not a
pass. The thresholds come from the YAML and are the design's proposed
defaults, not a calibration -- the last contract records what those
defaults let through on this fixture, so the arbitration has a number.

  * GE1 -- a faithful summary evicts: one receipt naming the span's turns,
    the span in the Cellar, the peel in the tree, the Flesh shortened.
  * GE2 -- fact inversion in the summary is refused (design blade 1).
  * GE3 -- decision deletion in the summary is refused (design blade 2).
  * GE4 -- an entity swap is caught by the probes and refused when the gate
    demands it (design blade 3).
  * GE5 -- a date shift is caught by the probes and refused when the gate
    demands it (design blade 4).
  * GE6 -- probe suppression: a rich span yields probes, a span with none
    is refused, never passed on an unknown rate (design blade 9).
  * GE7 -- the threshold is read, not hardcoded: the same summary passes
    one gate and fails a stricter one (design blade 10).
  * GE8 -- the gate is the YAML's, and an out-of-range gate is refused.
  * GE9 -- an empty Flesh evicts nothing and leaves no trace.
  * GE10 -- the finding, recorded: at the proposed defaults a single entity
    swap and a single date shift on the four-turn fixture pass the gate.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION_YAML = REPO / "opti_oignon" / "config" / "onion.yaml"


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


_SPAN = [
    {"turn_id": "t01", "role": "user", "text": "Alice and Bob met on 2026-03-04 to review the Harvest release with a budget of 1200 euros."},
    {"turn_id": "t02", "role": "assistant", "text": "We agreed that the demo will not use Docker because the venue in Oslo has no container runtime."},
    {"turn_id": "t03", "role": "user", "text": "Carol handles the Oslo account and the latency target is 45 milliseconds."},
    {"turn_id": "t04", "role": "assistant", "text": "Bob reviews the release on 2026-05-02 and the rollback keeps the old cluster warm for 7 days."},
]
_TAIL = [{"turn_id": f"t{i:02d}", "role": "user", "text": f"Service {i} moved to the new cluster on day {i}."} for i in range(5, 9)]


def _faithful(span):
    return " ".join(t["text"] for t in span)


def _edit(old, new):
    return lambda span: _faithful(span).replace(old, new)


def _setup(peels, receipts, *, gate=None):
    gate = gate or peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=4)
    return dict(flesh=receipts.Flesh(_SPAN + _TAIL), cellar=receipts.Cellar(),
                ledger=receipts.ReceiptLedger(), tree=peels.PeelTree(), gate=gate)


def _refused(peels, receipts, summarize, kind):
    s = _setup(peels, receipts)
    outcome = peels.evict_gated(summarize=summarize, **s)
    assert outcome.evicted is False
    assert outcome.receipt is None and outcome.peel is None
    assert len(s["flesh"].turns()) == 8, "the verbatim turns stay"
    assert s["ledger"].all() == [] and len(s["cellar"]) == 0 and s["tree"].all() == []
    kinds = {p.kind for p in outcome.decision.result.failures}
    assert kind in kinds, f"a {kind} probe failed: {outcome.decision.reason}"
    return outcome


# ---------------------------------------------------------------------------
# GE1 -- faithful
# ---------------------------------------------------------------------------
def test_ge1_a_faithful_summary_evicts_with_a_receipt_a_span_and_a_peel():
    peels, receipts, restore = _open()
    try:
        s = _setup(peels, receipts)
        outcome = peels.evict_gated(summarize=_faithful, **s)
        assert outcome.evicted is True, outcome.decision.reason
        assert outcome.receipt.turn_ids == ("t01", "t02", "t03", "t04")
        assert s["ledger"].open() == [outcome.receipt]
        assert s["cellar"].get(outcome.receipt.key) == _SPAN
        assert outcome.peel.sources == (outcome.receipt.key,)
        assert s["tree"].get(outcome.peel.id) == outcome.peel
        assert outcome.peel.probes_total >= 10 and outcome.peel.probes_passed == outcome.peel.probes_total
        assert [t["turn_id"] for t in s["flesh"].turns()] == ["t05", "t06", "t07", "t08"]
        assert outcome.decision.decision_rate == 1.0 and outcome.decision.episodic_rate == 1.0
    finally:
        restore()


# ---------------------------------------------------------------------------
# GE2 -- fact inversion (blade 1)
# ---------------------------------------------------------------------------
def test_ge2_a_fact_inversion_in_the_summary_is_refused():
    peels, receipts, restore = _open()
    try:
        outcome = _refused(peels, receipts, _edit("will not use Docker", "will use Docker"), "decision")
        assert outcome.decision.decision_rate == 0.0
        assert "decision" in outcome.decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# GE3 -- decision deletion (blade 2)
# ---------------------------------------------------------------------------
def test_ge3_a_deleted_decision_is_refused():
    peels, receipts, restore = _open()
    try:
        def drop_decision(span):
            return " ".join(t["text"] for t in span if "agreed" not in t["text"])
        outcome = _refused(peels, receipts, drop_decision, "decision")
        assert outcome.decision.decision_rate == 0.0
    finally:
        restore()


# ---------------------------------------------------------------------------
# GE4 -- entity swap (blade 3)
# ---------------------------------------------------------------------------
def test_ge4_an_entity_swap_is_caught_by_the_probes_and_refused_when_the_gate_demands():
    peels, receipts, restore = _open()
    try:
        s = _setup(peels, receipts, gate=peels.Gate(decision_threshold=0.9, episodic_threshold=1.0, span_turns=4))
        outcome = peels.evict_gated(summarize=_edit("Alice", "Dave"), **s)
        assert outcome.evicted is False
        failed = [(p.kind, p.answer, p.turn_id) for p in outcome.decision.result.failures]
        assert failed == [("entity", "Alice", "t01")], "exactly the swapped entity, with its turn"
        assert len(s["flesh"].turns()) == 8 and s["ledger"].all() == []
        assert outcome.decision.episodic_rate < 1.0
    finally:
        restore()


# ---------------------------------------------------------------------------
# GE5 -- date shift (blade 4)
# ---------------------------------------------------------------------------
def test_ge5_a_date_shift_is_caught_by_the_probes_and_refused_when_the_gate_demands():
    peels, receipts, restore = _open()
    try:
        s = _setup(peels, receipts, gate=peels.Gate(decision_threshold=0.9, episodic_threshold=1.0, span_turns=4))
        outcome = peels.evict_gated(summarize=_edit("2026-03-04", "2026-04-04"), **s)
        assert outcome.evicted is False
        failed = [(p.kind, p.answer, p.turn_id) for p in outcome.decision.result.failures]
        assert failed == [("date", "2026-03-04", "t01")]
        assert len(s["flesh"].turns()) == 8
    finally:
        restore()


# ---------------------------------------------------------------------------
# GE6 -- probe suppression (blade 9)
# ---------------------------------------------------------------------------
def test_ge6_a_span_without_probes_is_refused_never_passed_on_an_unknown_rate():
    peels, receipts, restore = _open()
    try:
        s = _setup(peels, receipts)
        rich = peels.evict_gated(summarize=_faithful, **s)
        assert rich.decision.result.passed + rich.decision.result.failed >= 10, "control: the rich span yields probes"
        bare = [{"turn_id": "b1", "role": "user", "text": "ok."}, {"turn_id": "b2", "role": "assistant", "text": "sure, thanks."}]
        s = _setup(peels, receipts)
        s["flesh"] = receipts.Flesh(bare)
        outcome = peels.evict_gated(summarize=_faithful, **s)
        assert outcome.decision.result.rate is None, "control: no probe existed"
        assert outcome.evicted is False
        assert "no probe" in outcome.decision.reason
        assert len(s["flesh"].turns()) == 2 and s["ledger"].all() == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# GE7 -- threshold read, not hardcoded (blade 10)
# ---------------------------------------------------------------------------
def test_ge7_the_same_summary_passes_one_gate_and_fails_a_stricter_one():
    peels, receipts, restore = _open()
    try:
        def drop_third(span):
            return " ".join(t["text"] for t in span if t["turn_id"] != "t03")
        loose = _setup(peels, receipts, gate=peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=4))
        passed = peels.evict_gated(summarize=drop_third, **loose)
        assert passed.evicted is True, passed.decision.reason
        assert 0.7 <= passed.decision.episodic_rate < 0.9, "the fixture sits between the two gates"
        strict = _setup(peels, receipts, gate=peels.Gate(decision_threshold=0.9, episodic_threshold=0.9, span_turns=4))
        refused = peels.evict_gated(summarize=drop_third, **strict)
        assert refused.evicted is False
        assert refused.decision.episodic_rate == passed.decision.episodic_rate
        assert "episodic" in refused.decision.reason and "0.9" in refused.decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# GE8 -- the gate is the YAML's
# ---------------------------------------------------------------------------
def test_ge8_the_gate_is_read_from_the_yaml_and_an_out_of_range_gate_is_refused():
    import yaml

    peels, receipts, restore = _open()
    try:
        raw = yaml.safe_load(_ONION_YAML.read_text(encoding="utf-8"))
        gate = peels.load_gate()
        assert gate.decision_threshold == float(raw["gate"]["decision_threshold"])
        assert gate.episodic_threshold == float(raw["gate"]["episodic_threshold"])
        assert gate.span_turns == int(raw["peels"]["span_turns"])
        assert gate.validate() == []
        assert 0.0 < gate.episodic_threshold <= gate.decision_threshold <= 1.0
        for bad in (
            peels.Gate(decision_threshold=1.5, episodic_threshold=0.7, span_turns=4),
            peels.Gate(decision_threshold=0.9, episodic_threshold=-0.1, span_turns=4),
            peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=0),
        ):
            assert bad.validate() != []
            with pytest.raises(peels.GateError):
                peels.evict_gated(summarize=_faithful, **_setup(peels, receipts, gate=bad))
    finally:
        restore()


# ---------------------------------------------------------------------------
# GE9 -- empty Flesh
# ---------------------------------------------------------------------------
def test_ge9_an_empty_flesh_evicts_nothing_and_leaves_no_trace():
    peels, receipts, restore = _open()
    try:
        s = _setup(peels, receipts)
        s["flesh"] = receipts.Flesh([])
        outcome = peels.evict_gated(summarize=_faithful, **s)
        assert outcome.evicted is False and outcome.receipt is None and outcome.peel is None
        assert "empty" in outcome.reason
        assert len(s["cellar"]) == 0 and s["ledger"].all() == [] and s["tree"].all() == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# GE10 -- the finding, recorded
# ---------------------------------------------------------------------------
def test_ge10_at_the_proposed_defaults_one_swap_and_one_shift_pass_the_gate():
    peels, receipts, restore = _open()
    try:
        gate = peels.load_gate()
        for summarize in (_edit("Alice", "Dave"), _edit("2026-03-04", "2026-04-04")):
            s = _setup(peels, receipts, gate=gate)
            outcome = peels.evict_gated(summarize=summarize, **s)
            assert outcome.decision.result.failed == 1, "control: the probe caught it"
            assert outcome.decision.episodic_rate == round(12 / 13, 4)
            assert outcome.evicted is True, (
                "recorded, not endorsed: at 0.7 a single wrong entity or date in a "
                "four-turn span passes the gate; the threshold is the arbitration's"
            )
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
