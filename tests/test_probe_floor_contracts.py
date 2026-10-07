#!/usr/bin/env python3
"""Contracts for the floor of the eviction gate: what share of the facts a
span holds its probes ask for.

The first face asks a summary the probes drawn from its span, and a rate
counts only the probes a summary answers: a generator that draws less makes
every rate rise. The gate does not trust the probe set it is handed. It
reads the span again as the second face reads a summary -- names by the
whole span, dates and numbers as read or as written, each fenced block by
its marker, a typed decision by its key and its polarity, piece by piece as
the origins bound them, and never by the native core -- and counts the
facts no probe asks for. Below the floor of ``onion.yaml`` the verbatim
turns stay, and the decision says why.

  * FL1 -- the facts of a span are its names, its dates and numbers, its
    fenced blocks by their markers and its typed decisions by their keys and
    polarity, each once, at the first turn that holds it, in reading order.
  * FL2 -- on every span of the labelled fixture set and of this suite the
    reference generator asks for every fact: the shipped floor refuses none
    of them, and the labelled set holds 69 facts.
  * FL3 -- the words no segment covers are no fact: what the executor writes
    between a typed question and its document is never counted, and the
    facts of either piece are.
  * FL4 -- a fenced block is one fact, its marker: the dates and numbers
    inside it are none, though the second face holds them.
  * FL5 -- only typed text decides: the same deciding sentence is no fact in
    the assistant's words, a document's, a refined question or a turn of no
    origin, and one fact typed.
  * FL6 -- a probe set that leaves a fact unasked is refused by name, with its
    coverage, its counts and the facts no probe asks for, though both faces
    and the bounds let the summary through.
  * FL7 -- the share counts facts, never probes: one fact asked three times
    leaves the others unasked, and a probe for a fact the span does not hold
    raises nothing.
  * FL8 -- a native twin that draws less than the reference is caught at the
    eviction: the verbatim turns stay and the decision names what it missed;
    a twin that draws the same evicts.
  * FL9 -- a generator that misses a class cannot raise the acceptance: a
    summary that loses the class, judged on such a generator's probes, is
    refused by the shipped floor, class by class, where the same gate without
    a floor lets it through although the reference probes refuse it.
  * FL10 -- a summary refused on every count gives the first face's reason,
    then its claims, then its bounds, then its floor.
  * FL11 -- a gate built by hand holds no floor: its coverage is measured and
    said, the floor it was held to is None, and it refuses nothing for it.
  * FL12 -- the gate reads its floor from ``onion.yaml`` and refuses by name a
    floor that is missing, null, a boolean, not a number, not finite or out
    of [0, 1].
  * FL13 -- a floor out of its range is refused by name before any summary is
    judged.
  * FL14 -- every decision carries its facts, its coverage, the facts no probe
    asks for and its floor, accepted or refused; a span with no fact has no
    coverage and is refused nothing for it.
  * FL15 -- the facts are never read by the native core: whatever a core
    draws, the span holds the same facts, and counting them never asks it.
  * FL16 -- the runbook reports the floor, the spread of the coverage of the
    spans it judged with how many fall under the floor, and the facts no
    probe asked for, kind by kind.
  * FL17 -- a leaf and a parent are held to the floor.
  * FL18 -- FL9 held word for word on the rich fixture with its code turn
    showing its block rather than ordering it run: the second face now
    refuses a summary that repeats an order the user did not type, the
    assistant's included.
  * FL19 to FL22 -- FL6, FL8, FL14 and FL17 held word for word on the same
    fixture, for the same reason.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION = REPO / "opti_oignon" / "config" / "onion.yaml"
_LABELLED = REPO / "tests" / "recall_set" / "labelled.json"


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
    return probes, loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.receipts"], restore


def _turn(turn_id, role, origin, text, **declared):
    return {"turn_id": turn_id, "role": role, "origin": origin, "text": text, **declared}


def _verbatim(span):
    return " ".join(t["text"] for t in span)


def _held(facts):
    return [(f.kind, f.what, f.turn_id) for f in facts]


_BLOCK = "print('farm')"
_RICH = [
    _turn("r1", "user", "typed", "The review is on 2026-03-04."),
    _turn("r2", "user", "typed", "It is led by Alice Martin and Bob."),
    _turn("r3", "user", "typed", "We keep the old cluster."),
    _turn("r4", "assistant", "assistant", "The farm has 16 Go per host."),
    _turn("r5", "assistant", "assistant", "Run this:\n\n```python\n" + _BLOCK + "\n```"),
]
_NAMES = (("entity", "Alice Martin", "r2"), ("entity", "Bob", "r2"))

_QUESTION = "Where does Alice live?"
_GAP = "\n\n[Attached by Zelda on 2031-01-01]\n\n"
_DOCUMENT = "Alice lives in Lyon since 2019."
_CONTENT = _QUESTION + _GAP + _DOCUMENT
_SEGMENTED = [_turn("s1", "user", "typed", _CONTENT, segments=[
    [0, len(_QUESTION), "typed"], [len(_QUESTION) + len(_GAP), len(_CONTENT), "document"],
])]

_INSIDE = "deadline = '2026-01-01'\nretries = 42"
_CODE_ONLY = [_turn("c1", "assistant", "assistant", "Set it:\n\n```\n" + _INSIDE + "\n```")]

_DECIDES = "We keep the old cluster."
_THREE = [_turn("h1", "user", "typed", "On 2026-03-04 the team met Alice and Bob.")]
_QUIET = [_turn("q1", "user", "typed", "hello there")]
_TAIL = [_turn(f"x{i}", "user", "typed", f"Service {i} moved on day {i}.") for i in range(1, 4)]


# ---------------------------------------------------------------------------
# FL1 -- the facts of a span
# ---------------------------------------------------------------------------
def test_fl1_the_facts_of_a_span_are_its_names_dates_numbers_blocks_and_typed_decisions_each_once_in_order():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        expected = [
            ("date", "2026-03-04", "r1"),
            ("entity", "Alice Martin", "r2"),
            ("entity", "Bob", "r2"),
            ("decision", _DECIDES, "r3"),
            ("number", "16 GB", "r4"),
            ("code", probes.code_marker(_BLOCK), "r5"),
        ]
        assert _held(probes.held_facts(_RICH, gate.lexicon)) == expected
        again = _RICH + [_turn("r6", "user", "typed", "Bob checks the review on 2026-03-04.")]
        assert _held(probes.held_facts(again, gate.lexicon)) == expected, "each fact once, at the first turn"
        inverted = _RICH + [_turn("r6", "user", "typed", "We do not keep the old cluster.")]
        assert _held(probes.held_facts(inverted, gate.lexicon)) == expected + [
            ("decision", "We do not keep the old cluster.", "r6"),
        ], "a decision of the other polarity is another fact"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL2 -- the reference asks for every fact of the fixtures
# ---------------------------------------------------------------------------
def test_fl2_the_reference_generator_asks_for_every_fact_of_the_fixtures_and_the_shipped_floor_refuses_none():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        labelled = [span["turns"] for span in probes.load_labelled(_LABELLED)]
        assert len(labelled) == 18
        total = 0
        for turns in labelled + [_RICH, _SEGMENTED, _CODE_ONLY, _THREE]:
            drawn = probes.generate_probes(turns, gate.lexicon)
            coverage = probes.probe_coverage(turns, drawn, gate.lexicon)
            assert coverage.facts >= 1 and coverage.asked == coverage.facts and coverage.unasked == (), coverage
            decision = peels.decide(turns, drawn, _verbatim(turns), gate)
            assert decision.probe_coverage == 1.0 and "probe coverage" not in decision.reason, decision.reason
            if turns in labelled:
                total += coverage.facts
        assert total == 69
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL3 -- the words no segment covers
# ---------------------------------------------------------------------------
def test_fl3_the_words_no_segment_covers_are_no_fact_and_the_facts_of_either_piece_are():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        assert _held(probes.held_facts(_SEGMENTED, gate.lexicon)) == [
            ("entity", "Alice", "s1"), ("number", "2019", "s1"), ("entity", "Lyon", "s1"),
        ]
        held = probes.holdings(_SEGMENTED)
        assert "zelda" in held.words and "2031-01-01" in held.dates, "control: the second face holds the gap's words"
        whole = [dict(_SEGMENTED[0], segments=[])]
        read = _held(probes.held_facts(whole, gate.lexicon))
        assert ("entity", "Zelda", "s1") in read and ("date", "2031-01-01", "s1") in read, (
            "control: the same words in a turn of one piece are facts"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL4 -- a fenced block is one fact
# ---------------------------------------------------------------------------
def test_fl4_a_fenced_block_is_one_fact_its_marker_and_what_it_holds_is_none():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        assert _held(probes.held_facts(_CODE_ONLY, gate.lexicon)) == [("code", probes.code_marker(_INSIDE), "c1")]
        held = probes.holdings(_CODE_ONLY)
        assert "2026-01-01" in held.dates and "42" in held.numbers, "control: the second face holds what the block holds"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL5 -- only typed text decides
# ---------------------------------------------------------------------------
def test_fl5_a_deciding_sentence_is_a_fact_only_where_it_was_typed():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        for role, origin in (("assistant", "assistant"), ("user", "document"), ("user", "refined"),
                             ("user", "legacy")):
            assert probes.held_facts([_turn("d1", role, origin, _DECIDES)], gate.lexicon) == (), origin
        typed = probes.held_facts([_turn("d1", "user", "typed", _DECIDES)], gate.lexicon)
        assert _held(typed) == [("decision", _DECIDES, "d1")]
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL6 -- a fact left unasked refuses the summary by name
# ---------------------------------------------------------------------------
def test_fl6_a_probe_set_that_leaves_a_fact_unasked_is_refused_by_name_with_its_figures():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_RICH, gate.lexicon)
        starved = [q for q in drawn if q.kind != "entity"]
        text = _verbatim(_RICH)
        assert peels.judge(starved, text, gate).accepted is True, "control: the first face lets it through"
        assert peels.faithfulness(_RICH, starved, text, gate) == (), "control: it says nothing its span lacks"
        refused = peels.decide(_RICH, starved, text, gate)
        assert refused.accepted is False
        assert refused.facts == 6 and refused.probe_coverage == 4 / 6 and refused.probe_floor == 1.0
        assert refused.unasked == _NAMES
        assert refused.reason == (f"probe coverage {round(4 / 6, 4)} under 1.0 (4 of 6 facts asked, unasked: "
                                  "entity:Alice Martin@r2, entity:Bob@r2)")
        assert refused.unsupported == (), "the floor is said in the reason, never as a claim"
        held = peels.decide(_RICH, drawn, text, gate)
        assert held.accepted is True and held.probe_coverage == 1.0 and held.unasked == (), held.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL7 -- the share counts facts, never probes
# ---------------------------------------------------------------------------
def test_fl7_the_share_counts_facts_never_probes():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_THREE, gate.lexicon)
        assert [(q.kind, q.answer) for q in drawn] == [("date", "2026-03-04"), ("entity", "Alice"), ("entity", "Bob")]
        date = drawn[0]
        thrice = peels.decide(_THREE, [date, date, date], _verbatim(_THREE), gate)
        assert thrice.accepted is False and thrice.facts == 3 and thrice.probe_coverage == 1 / 3
        assert thrice.unasked == (("entity", "Alice", "h1"), ("entity", "Bob", "h1"))
        stranger = dataclasses.replace(drawn[1], answer="Zelda")
        swapped = peels.decide(_THREE, [drawn[0], stranger, drawn[2]], _verbatim(_THREE), gate)
        assert swapped.facts == 3 and swapped.probe_coverage == 2 / 3, "a probe for a fact the span lacks asks nothing"
        assert swapped.unasked == (("entity", "Alice", "h1"),)
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL8 -- a twin that draws less is caught at the eviction
# ---------------------------------------------------------------------------
def _setup(peels, receipts, gate, span):
    return dict(flesh=receipts.Flesh(span + _TAIL), cellar=receipts.Cellar(), ledger=receipts.ReceiptLedger(),
                tree=peels.PeelTree(), gate=dataclasses.replace(gate, span_turns=len(span)))


def test_fl8_a_native_twin_that_draws_less_than_the_reference_is_caught_at_the_eviction():
    probes, peels, receipts, restore = _window()
    try:
        gate = peels.load_gate()
        reference = probes.generate_probes(_RICH, gate.lexicon)
        probes._native_draw = lambda pieces, lexicon: [q for q in reference if q.kind != "entity"]
        s = _setup(peels, receipts, gate, _RICH)
        outcome = peels.evict_gated(summarize=_verbatim, **s)
        assert outcome.evicted is False and outcome.receipt is None and outcome.peel is None
        assert len(s["flesh"].turns()) == len(_RICH) + len(_TAIL), "the verbatim turns stay"
        assert outcome.decision.result.failed == 0, "control: the summary answers every probe the twin drew"
        assert outcome.decision.unasked == _NAMES
        assert "4 of 6 facts asked, unasked: entity:Alice Martin@r2, entity:Bob@r2" in outcome.reason
        probes._native_draw = lambda pieces, lexicon: list(reference)
        s = _setup(peels, receipts, gate, _RICH)
        assert peels.evict_gated(summarize=_verbatim, **s).evicted is True, "control: a twin that draws the same evicts"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL9 -- a generator that misses a class cannot raise the acceptance
# ---------------------------------------------------------------------------
_LOST = (("date", "r1"), ("entity", "r2"), ("decision", "r3"), ("number", "r4"), ("code", "r5"))
# Where the shipped thresholds refuse the loss on the reference probes: one
# date or one number of four episodic probes is within the episodic threshold.
_RISES = ("entity", "decision", "code")


def test_fl9_a_generator_that_misses_a_class_cannot_raise_the_acceptance():
    probes, peels, _receipts, restore = _window()
    try:
        shipped = peels.load_gate()
        unfloored = dataclasses.replace(shipped, probe_floor=None)
        reference = probes.generate_probes(_RICH, shipped.lexicon)
        for kind, turn_id in _LOST:
            missing = [q for q in reference if q.kind != kind]
            summary = _verbatim([t for t in _RICH if t["turn_id"] != turn_id])
            if kind in _RISES:
                assert peels.decide(_RICH, reference, summary, unfloored).accepted is False, (kind, "control")
                assert peels.decide(_RICH, missing, summary, unfloored).accepted is True, (kind, "the rate rises")
            refused = peels.decide(_RICH, missing, summary, shipped)
            assert refused.accepted is False and refused.reason.startswith("probe coverage "), (kind, refused.reason)
            assert {k for k, _what, _turn in refused.unasked} == {kind}, (kind, refused.unasked)
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL10 -- every reason, the floor's last
# ---------------------------------------------------------------------------
_ADRIFT = ("Violet lanterns, sprawling meadows and gorgeous dunes greeted Dave near quiet harbors with painted kites "
           "and amber lamps.")


def test_fl10_a_summary_refused_on_every_count_gives_the_floor_after_the_bounds():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        starved = [q for q in probes.generate_probes(_RICH, gate.lexicon) if q.kind != "entity"]
        first = peels.judge(starved, _ADRIFT, gate)
        assert first.accepted is False, "control: the first face refuses it"
        decision = peels.decide(_RICH, starved, _ADRIFT, gate)
        reason = decision.reason
        assert decision.accepted is False and reason.startswith(first.reason)
        assert (len(first.reason) < reason.index("entity:Dave") < reason.index("novelty 1.0 over 0.5")
                < reason.index("probe coverage"))
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL11 -- a gate built by hand holds no floor
# ---------------------------------------------------------------------------
def test_fl11_a_gate_built_by_hand_holds_no_floor_and_its_decision_says_so():
    probes, peels, _receipts, restore = _window()
    try:
        shipped = peels.load_gate()
        hand = peels.Gate(decision_threshold=shipped.decision_threshold,
                          episodic_threshold=shipped.episodic_threshold, span_turns=1, lexicon=shipped.lexicon)
        assert hand.probe_floor is None and hand.validate() == []
        starved = [q for q in probes.generate_probes(_RICH, shipped.lexicon) if q.kind != "entity"]
        decision = peels.decide(_RICH, starved, _verbatim(_RICH), hand)
        assert decision.accepted is True, decision.reason
        assert decision.probe_floor is None, "no floor, and it says so"
        assert decision.probe_coverage == 4 / 6 and decision.unasked == _NAMES, "measured all the same"
        assert peels.decide(_RICH, starved, _verbatim(_RICH), shipped).accepted is False, "control: the shipped floor"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL12 -- the floor is the YAML's
# ---------------------------------------------------------------------------
def _loaded(peels, tmp_path, mutate):
    raw = yaml.safe_load(_ONION.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "onion.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=False), encoding="utf-8")
    return peels.load_gate(path)


def _set(section, key, value):
    def mutate(raw):
        raw[section][key] = value
    return mutate


def _drop(section, key):
    def mutate(raw):
        del raw[section][key]
    return mutate


def test_fl12_the_floor_is_read_from_onion_yaml_and_refused_by_name(tmp_path):
    _probes, peels, _receipts, restore = _window()
    try:
        raw = yaml.safe_load(_ONION.read_text(encoding="utf-8"))
        gate = peels.load_gate()
        assert gate.probe_floor == raw["gate"]["probe_floor"] == 1.0
        assert gate.validate() == []
        assert _loaded(peels, tmp_path, _set("gate", "probe_floor", 0)).probe_floor == 0.0, "a floor of none, read"
        assert _loaded(peels, tmp_path, _set("gate", "probe_floor", 0.8)).probe_floor == 0.8
        with pytest.raises(peels.GateError, match="probe_floor"):
            _loaded(peels, tmp_path, _drop("gate", "probe_floor"))
        for value in (None, True, "all", float("nan"), float("inf"), 1.5, -0.1):
            with pytest.raises(peels.GateError, match="probe_floor"):
                _loaded(peels, tmp_path, _set("gate", "probe_floor", value))
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL13 -- a floor out of its range is refused before any summary is judged
# ---------------------------------------------------------------------------
def test_fl13_a_floor_out_of_its_range_is_refused_by_name_before_any_summary_is_judged():
    probes, peels, _receipts, restore = _window()
    try:
        shipped = peels.load_gate()
        drawn = probes.generate_probes(_RICH, shipped.lexicon)
        for value in (True, "all", float("nan"), -0.1, 1.5):
            gate = dataclasses.replace(shipped, probe_floor=value)
            assert [e for e in gate.validate() if e.startswith("probe_floor:")], value
            with pytest.raises(peels.GateError, match="probe_floor"):
                peels.decide(_RICH, drawn, _verbatim(_RICH), gate)
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL14 -- every decision carries its figures
# ---------------------------------------------------------------------------
def test_fl14_every_decision_carries_its_facts_coverage_unasked_facts_and_floor():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_RICH, gate.lexicon)
        accepted = peels.decide(_RICH, drawn, _verbatim(_RICH), gate)
        assert accepted.accepted is True, accepted.reason
        assert (accepted.facts, accepted.probe_coverage, accepted.unasked, accepted.probe_floor) == (6, 1.0, (), 1.0)
        claimed = peels.decide(_RICH, drawn, _verbatim(_RICH) + " They met Dave.", gate)
        assert claimed.accepted is False and claimed.unsupported, "control: refused by a claim"
        assert (claimed.facts, claimed.probe_coverage, claimed.unasked, claimed.probe_floor) == (6, 1.0, (), 1.0)
        quiet = peels.decide(_QUIET, probes.generate_probes(_QUIET, gate.lexicon), _verbatim(_QUIET), gate)
        assert quiet.facts == 0 and quiet.probe_coverage is None and quiet.unasked == ()
        assert quiet.accepted is False and "probe coverage" not in quiet.reason, "no fact: no coverage, no refusal for it"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL15 -- the facts are never read by the native core
# ---------------------------------------------------------------------------
def test_fl15_the_facts_of_a_span_are_never_read_by_the_native_core():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        facts = probes.held_facts(_RICH, gate.lexicon)
        asked = []

        def twin(pieces, lexicon):
            asked.append(len(pieces))
            return []

        probes._native_draw = twin
        probes._native = lambda: object()
        assert probes.held_facts(_RICH, gate.lexicon) == facts, "whatever a core draws, the span holds the same facts"
        assert probes.generate_probes(_RICH, gate.lexicon) == [] and asked == [5], "control: the generator asks it"
        asked.clear()
        coverage = probes.probe_coverage(_RICH, [], gate.lexicon)
        assert coverage.facts == 6 and coverage.asked == 0 and asked == [], "counting the facts never asks the core"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL16 -- the runbook reports the floor
# ---------------------------------------------------------------------------
def _runbook():
    """The runbook script, loaded from its file; the path it adds is taken back."""
    spec = importlib.util.spec_from_file_location("oo_onion_runbook", REPO / "scripts" / "onion_runbook.py")
    module = importlib.util.module_from_spec(spec)
    path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = path
    return module


def test_fl16_the_runbook_reports_the_floor_the_spread_of_coverage_and_the_unasked_facts():
    probes, peels, _receipts, restore = _window()
    try:
        runbook = _runbook()
        gate = peels.load_gate()
        assert runbook._gate_entry(gate)["probe_floor"] == 1.0
        drawn = probes.generate_probes(_RICH, gate.lexicon)
        judged = [
            (_RICH, drawn, _verbatim(_RICH)),
            (_RICH, [q for q in drawn if q.kind != "entity"], _verbatim(_RICH)),
            (_RICH, [q for q in drawn if q.kind not in ("entity", "code")], _verbatim(_RICH)),
            (_QUIET, [], _verbatim(_QUIET)),
        ]
        face = runbook._second_face(judged, gate)
        assert face["probe_coverage"] == {"measured": 3, "unmeasured": 1, "under": 2,
                                          "min": 0.5, "p50": 0.6667, "p90": 1.0, "max": 1.0}
        assert face["unasked"] == {"code": 1, "entity": 4}
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL17 -- a leaf and a parent are held to the floor
# ---------------------------------------------------------------------------
def test_fl17_a_leaf_and_a_parent_are_held_to_the_floor():
    probes, peels, receipts, restore = _window()
    try:
        gate = peels.load_gate()
        s = _setup(peels, receipts, gate, _RICH)
        evicted = peels.evict_gated(summarize=_verbatim, **s)
        assert evicted.evicted is True, evicted.reason
        reference = probes.generate_probes(_RICH, gate.lexicon)
        probes._native_draw = lambda pieces, lexicon: [q for q in reference if q.kind != "entity"]
        leaf, decision = peels.build_leaf(evicted.receipt.key, s["cellar"], _verbatim, s["gate"], s["tree"])
        assert leaf is None and decision.unasked == _NAMES, decision.reason
        parent, decision = peels.build_parent([evicted.peel.id], s["cellar"], _verbatim, s["gate"], s["tree"])
        assert parent is None and decision.unasked == _NAMES, decision.reason
        assert len(s["tree"].all()) == 1, "nothing was added"
        probes._native_draw = lambda pieces, lexicon: None
        leaf, decision = peels.build_leaf(evicted.receipt.key, s["cellar"], _verbatim, s["gate"], s["tree"])
        assert leaf is not None, decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL18 -- FL9 on the rich fixture whose code turn tells rather than orders
# ---------------------------------------------------------------------------
_TOLD_RICH = _RICH[:4] + [_turn("r5", "assistant", "assistant", "The code:\n\n```python\n" + _BLOCK + "\n```")]


def test_fl18_a_generator_that_misses_a_class_cannot_raise_the_acceptance():
    probes, peels, _receipts, restore = _window()
    try:
        shipped = peels.load_gate()
        unfloored = dataclasses.replace(shipped, probe_floor=None)
        reference = probes.generate_probes(_TOLD_RICH, shipped.lexicon)
        for kind, turn_id in _LOST:
            missing = [q for q in reference if q.kind != kind]
            summary = _verbatim([t for t in _TOLD_RICH if t["turn_id"] != turn_id])
            if kind in _RISES:
                assert peels.decide(_TOLD_RICH, reference, summary, unfloored).accepted is False, (kind, "control")
                assert peels.decide(_TOLD_RICH, missing, summary, unfloored).accepted is True, (kind, "the rate rises")
            refused = peels.decide(_TOLD_RICH, missing, summary, shipped)
            assert refused.accepted is False and refused.reason.startswith("probe coverage "), (kind, refused.reason)
            assert {k for k, _what, _turn in refused.unasked} == {kind}, (kind, refused.unasked)
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL19 -- FL6 on the rich fixture whose code turn tells rather than orders
# ---------------------------------------------------------------------------
def test_fl19_a_probe_set_that_leaves_a_fact_unasked_is_refused_by_name_with_its_figures():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_TOLD_RICH, gate.lexicon)
        starved = [q for q in drawn if q.kind != "entity"]
        text = _verbatim(_TOLD_RICH)
        assert peels.judge(starved, text, gate).accepted is True, "control: the first face lets it through"
        assert peels.faithfulness(_TOLD_RICH, starved, text, gate) == (), "control: it says nothing its span lacks"
        refused = peels.decide(_TOLD_RICH, starved, text, gate)
        assert refused.accepted is False
        assert refused.facts == 6 and refused.probe_coverage == 4 / 6 and refused.probe_floor == 1.0
        assert refused.unasked == _NAMES
        assert refused.reason == (f"probe coverage {round(4 / 6, 4)} under 1.0 (4 of 6 facts asked, unasked: "
                                  "entity:Alice Martin@r2, entity:Bob@r2)")
        assert refused.unsupported == (), "the floor is said in the reason, never as a claim"
        held = peels.decide(_TOLD_RICH, drawn, text, gate)
        assert held.accepted is True and held.probe_coverage == 1.0 and held.unasked == (), held.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL20 -- FL8 on the rich fixture whose code turn tells rather than orders
# ---------------------------------------------------------------------------
def test_fl20_a_native_twin_that_draws_less_than_the_reference_is_caught_at_the_eviction():
    probes, peels, receipts, restore = _window()
    try:
        gate = peels.load_gate()
        reference = probes.generate_probes(_TOLD_RICH, gate.lexicon)
        probes._native_draw = lambda pieces, lexicon: [q for q in reference if q.kind != "entity"]
        s = _setup(peels, receipts, gate, _TOLD_RICH)
        outcome = peels.evict_gated(summarize=_verbatim, **s)
        assert outcome.evicted is False and outcome.receipt is None and outcome.peel is None
        assert len(s["flesh"].turns()) == len(_TOLD_RICH) + len(_TAIL), "the verbatim turns stay"
        assert outcome.decision.result.failed == 0, "control: the summary answers every probe the twin drew"
        assert outcome.decision.unasked == _NAMES
        assert "4 of 6 facts asked, unasked: entity:Alice Martin@r2, entity:Bob@r2" in outcome.reason
        probes._native_draw = lambda pieces, lexicon: list(reference)
        s = _setup(peels, receipts, gate, _TOLD_RICH)
        assert peels.evict_gated(summarize=_verbatim, **s).evicted is True, "control: a twin that draws the same evicts"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL21 -- FL14 on the rich fixture whose code turn tells rather than orders
# ---------------------------------------------------------------------------
def test_fl21_every_decision_carries_its_facts_coverage_unasked_facts_and_floor():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_TOLD_RICH, gate.lexicon)
        accepted = peels.decide(_TOLD_RICH, drawn, _verbatim(_TOLD_RICH), gate)
        assert accepted.accepted is True, accepted.reason
        assert (accepted.facts, accepted.probe_coverage, accepted.unasked, accepted.probe_floor) == (6, 1.0, (), 1.0)
        claimed = peels.decide(_TOLD_RICH, drawn, _verbatim(_TOLD_RICH) + " They met Dave.", gate)
        assert claimed.accepted is False and claimed.unsupported, "control: refused by a claim"
        assert (claimed.facts, claimed.probe_coverage, claimed.unasked, claimed.probe_floor) == (6, 1.0, (), 1.0)
        quiet = peels.decide(_QUIET, probes.generate_probes(_QUIET, gate.lexicon), _verbatim(_QUIET), gate)
        assert quiet.facts == 0 and quiet.probe_coverage is None and quiet.unasked == ()
        assert quiet.accepted is False and "probe coverage" not in quiet.reason, "no fact: no coverage, no refusal for it"
    finally:
        restore()


# ---------------------------------------------------------------------------
# FL22 -- FL17 on the rich fixture whose code turn tells rather than orders
# ---------------------------------------------------------------------------
def test_fl22_a_leaf_and_a_parent_are_held_to_the_floor():
    probes, peels, receipts, restore = _window()
    try:
        gate = peels.load_gate()
        s = _setup(peels, receipts, gate, _TOLD_RICH)
        evicted = peels.evict_gated(summarize=_verbatim, **s)
        assert evicted.evicted is True, evicted.reason
        reference = probes.generate_probes(_TOLD_RICH, gate.lexicon)
        probes._native_draw = lambda pieces, lexicon: [q for q in reference if q.kind != "entity"]
        leaf, decision = peels.build_leaf(evicted.receipt.key, s["cellar"], _verbatim, s["gate"], s["tree"])
        assert leaf is None and decision.unasked == _NAMES, decision.reason
        parent, decision = peels.build_parent([evicted.peel.id], s["cellar"], _verbatim, s["gate"], s["tree"])
        assert parent is None and decision.unasked == _NAMES, decision.reason
        assert len(s["tree"].all()) == 1, "nothing was added"
        probes._native_draw = lambda pieces, lexicon: None
        leaf, decision = peels.build_leaf(evicted.receipt.key, s["cellar"], _verbatim, s["gate"], s["tree"])
        assert leaf is not None, decision.reason
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
