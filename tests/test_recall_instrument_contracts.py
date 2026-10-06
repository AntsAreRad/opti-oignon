#!/usr/bin/env python3
"""Contracts for the recall instrument: what the probes recall of the facts a
reader would keep, and the gate figures that carry it.

An acceptance rate says how many probes a summary answers; it says nothing
of the facts no probe was drawn for. The instrument reads a labelled set --
spans whose facts a reader marked by class, at the turn that first writes
them -- and counts, class by class, the facts the generator draws a probe
for. The figure read on the fixture set of these tests is stated in
``onion.yaml`` with the version of the generator and the fingerprint of the
lexicon it was read with, measured again here, and carried by every gate
decision, every fidelity reading and every rate of the runbook -- or, when
the generator or the lexicon differs, None with the reason: a recall read on
another generator or another lexicon is never borrowed.

  * RA1 -- the fixture set holds every class in both languages, five facts
    at least each.
  * RA2 -- a fact is recalled only by a probe of its own class.
  * RA3 -- a fact is recalled only by a probe drawn at its own turn.
  * RA4 -- a fact is recalled only by a probe with its own answer.
  * RA5 -- a decision is recalled only by a probe of its own polarity.
  * RA6 -- a decision is recalled by its acts, dates and numbers, whatever
    other words the generator keys it with, and by no other set of them.
  * RA7 -- a code block is recalled by the marker of its own body.
  * RA8 -- a fact is counted once, however many probes answer it.
  * RA9 -- a malformed labelled set is refused by name.
  * RA10 -- a refusal names where the defect lies, never what the text says.
  * RA11 -- read again on the fixture set, the recall is the one
    ``onion.yaml`` states: generator, lexicon and every count.
  * RA12 -- every fact the set marks as a known miss is missed.
  * RA13 -- the gate loads the stated recall field by field, its classes in
    their fixed order whatever the order of the file.
  * RA14 -- a malformed stated recall is refused by name.
  * RA15 -- a gate file that states no recall loads a gate with none, and
    its decisions say so.
  * RA16 -- a decision of the shipped gate carries the stated recall.
  * RA17 -- a recall read with another generator is not carried, and the
    decision names both versions.
  * RA18 -- a recall read with another lexicon is not carried, and the
    decision names both fingerprints.
  * RA19 -- a gate built by hand carries no recall, and its decisions say
    so.
  * RA20 -- a summary the second face refuses carries the recall too.
  * RA21 -- a fidelity reading states the recall that holds for its probes,
    and never another.
  * RA22 -- every rate the runbook reports stands beside the recall that
    holds for it.
  * RA23 -- the runbook reads a labelled set for its counts only: no
    writing and no turn of the set reaches its output.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION = REPO / "opti_oignon" / "config" / "onion.yaml"
_SET = REPO / "tests" / "recall_set" / "labelled.json"
_DROP = object()


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


_TEXT = "We keep the Oslo office from 2027-03-14."
_DECISION = {"class": "decision", "turn_id": "t1", "writing": _TEXT, "negated": False, "acts": ["keep"],
             "dates": ["2027-03-14"], "numbers": []}


def _valid():
    return {"format": "labelled-recall-set", "version": 1, "spans": [{
        "span_id": "m-1", "language": "en",
        "turns": [
            {"turn_id": "t1", "role": "user", "origin": "typed", "text": _TEXT},
            {"turn_id": "t2", "role": "assistant", "origin": "assistant", "text": "Noted, the office has 12 GB."},
        ],
        "facts": [
            {"class": "entity", "turn_id": "t1", "writing": "Oslo", "answer": "Oslo"},
            {"class": "date", "turn_id": "t1", "writing": "2027-03-14", "answer": "2027-03-14"},
            {"class": "number", "turn_id": "t2", "writing": "12 GB", "answer": "12 GB"},
            dict(_DECISION),
        ],
    }]}


def _spans(probes, *facts):
    raw = _valid()
    raw["spans"][0]["facts"] = list(facts)
    return probes.labelled_spans(raw)


def _count(reading, kind):
    """``(recalled, size)`` of one class of a reading."""
    for name, recalled, size in reading.probe_recall.classes:
        if name == kind:
            return recalled, size
    raise AssertionError(f"the reading has no class {kind}")


def _drawn(*probes_):
    return lambda span, lexicon=None: list(probes_)


def _probe(probes, kind, answer, turn_id="t1", **extra):
    return probes.Probe(kind, f"{kind}?", answer, turn_id, **extra)


def _gate_with(peels, tmp_path, recall):
    raw = yaml.safe_load(_ONION.read_text(encoding="utf-8"))
    if recall is _DROP:
        raw["gate"].pop("probe_recall", None)
    else:
        raw["gate"]["probe_recall"] = recall
    path = tmp_path / "onion.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return peels.load_gate(path)


def _stated(probes, generator, lexicon):
    return {"source": "fixture", "generator": generator, "lexicon": lexicon,
            "classes": {kind: {"size": 4, "recalled": 3} for kind in probes.RECALL_CLASSES}}


def _span():
    return [{"turn_id": "t1", "role": "user", "origin": "typed", "text": "On 2027-03-14, Alice met Bob in Oslo."}]


def _fields(recall):
    return (recall.source, recall.generator, recall.lexicon, recall.classes)


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


def test_ra1_the_fixture_set_holds_every_class_in_both_languages_five_facts_at_least_each():
    probes, peels, restore = _window()
    try:
        spans = probes.load_labelled(_SET)
        counts = Counter((fact["class"], span["language"]) for span in spans for fact in span["facts"])
        short = {(kind, language): counts[(kind, language)] for kind in probes.RECALL_CLASSES
                 for language in ("fr", "en") if counts[(kind, language)] < 5}
        assert short == {}
        assert len({span["span_id"] for span in spans}) == len(spans) >= 16
    finally:
        restore()


def test_ra2_a_fact_is_recalled_only_by_a_probe_of_its_own_class():
    probes, peels, restore = _window()
    try:
        spans = _spans(probes,{"class": "date", "turn_id": "t1", "writing": "2027-03-14", "answer": "2027-03-14"})
        other = probes.measure_recall(spans, generate=_drawn(_probe(probes, "number", "2027-03-14")))
        own = probes.measure_recall(spans, generate=_drawn(_probe(probes, "date", "2027-03-14")))
        assert _count(other, "date") == (0, 1)
        assert _count(own, "date") == (1, 1)
    finally:
        restore()


def test_ra3_a_fact_is_recalled_only_by_a_probe_drawn_at_its_own_turn():
    probes, peels, restore = _window()
    try:
        spans = _spans(probes,{"class": "entity", "turn_id": "t1", "writing": "Oslo", "answer": "Oslo"})
        elsewhere = probes.measure_recall(spans, generate=_drawn(_probe(probes, "entity", "Oslo", "t2")))
        here = probes.measure_recall(spans, generate=_drawn(_probe(probes, "entity", "Oslo", "t1")))
        assert _count(elsewhere, "entity") == (0, 1)
        assert _count(here, "entity") == (1, 1)
    finally:
        restore()


def test_ra4_a_fact_is_recalled_only_by_a_probe_with_its_own_answer():
    probes, peels, restore = _window()
    try:
        spans = _spans(probes,{"class": "number", "turn_id": "t2", "writing": "12 GB", "answer": "12 GB"})
        wrong = probes.measure_recall(spans, generate=_drawn(_probe(probes, "number", "12", "t2")))
        right = probes.measure_recall(spans, generate=_drawn(_probe(probes, "number", "12 GB", "t2")))
        assert _count(wrong, "number") == (0, 1)
        assert _count(right, "number") == (1, 1)
    finally:
        restore()


def test_ra5_a_decision_is_recalled_only_by_a_probe_of_its_own_polarity():
    probes, peels, restore = _window()
    try:
        spans = _spans(probes,dict(_DECISION))
        key = frozenset({"act:keep", "date:2027-03-14", "oslo", "office"})
        flipped = probes.measure_recall(spans, generate=_drawn(_probe(probes, "decision", _TEXT, key=key, negated=True)))
        same = probes.measure_recall(spans, generate=_drawn(_probe(probes, "decision", _TEXT, key=key, negated=False)))
        assert _count(flipped, "decision") == (0, 1)
        assert _count(same, "decision") == (1, 1)
    finally:
        restore()


def test_ra6_a_decision_is_recalled_by_its_acts_dates_and_numbers_whatever_other_words_key_it():
    probes, peels, restore = _window()
    try:
        spans = _spans(probes,dict(_DECISION))

        def read(*members):
            probe = _probe(probes, "decision", _TEXT, key=frozenset(members))
            return _count(probes.measure_recall(spans, generate=_drawn(probe)), "decision")

        assert read("act:keep", "date:2027-03-14", "oslo", "office", "anything") == (1, 1)
        assert read("act:keep", "date:2027-03-14") == (1, 1)
        assert read("act:keep", "oslo", "office") == (0, 1)
        assert read("act:keep", "act:drop", "date:2027-03-14") == (0, 1)
        assert read("act:keep", "date:2027-03-14", "number:12 GB") == (0, 1)
    finally:
        restore()


def test_ra7_a_code_block_is_recalled_by_the_marker_of_its_own_body():
    probes, peels, restore = _window()
    try:
        raw = _valid()
        body = "port: 9443\nworkers: 8"
        raw["spans"][0]["turns"][1]["text"] = f"Here it is:\n```yaml\n{body}\n```"
        raw["spans"][0]["facts"] = [
            {"class": "code", "turn_id": "t2", "writing": body, "answer": probes.code_key(body)},
        ]
        spans = probes.labelled_spans(raw)
        other = probes.measure_recall(spans, generate=_drawn(_probe(probes, "code", probes.code_marker("port: 9443"), "t2")))
        own = probes.measure_recall(spans, generate=_drawn(_probe(probes, "code", probes.code_marker(body), "t2")))
        assert _count(other, "code") == (0, 1)
        assert _count(own, "code") == (1, 1)
    finally:
        restore()


def test_ra8_a_fact_is_counted_once_however_many_probes_answer_it():
    probes, peels, restore = _window()
    try:
        spans = _spans(probes,{"class": "entity", "turn_id": "t1", "writing": "Oslo", "answer": "Oslo"})
        twice = _probe(probes, "entity", "Oslo")
        reading = probes.measure_recall(spans, generate=_drawn(twice, twice, twice))
        assert _count(reading, "entity") == (1, 1)
        assert reading.missed == ()
    finally:
        restore()


def _facts(raw):
    return raw["spans"][0]["facts"]


def _turns(raw):
    return raw["spans"][0]["turns"]


_MALFORMED = {
    "format": (lambda raw: raw.update(format="recall"), "format"),
    "version": (lambda raw: raw.update(version=2), "version"),
    "no span": (lambda raw: raw.update(spans=[]), "no span"),
    "span twice": (lambda raw: raw["spans"].append(copy.deepcopy(raw["spans"][0])), "span_id"),
    "language": (lambda raw: raw["spans"][0].update(language="de"), "language"),
    "turn twice": (lambda raw: _turns(raw)[1].update(turn_id="t1"), "turn_id"),
    "turn without text": (lambda raw: _turns(raw)[1].pop("text"), "text"),
    "origin": (lambda raw: _turns(raw)[0].update(origin="typed+web"), "origin"),
    "class": (lambda raw: _facts(raw)[0].update({"class": "place"}), "class"),
    "turn": (lambda raw: _facts(raw)[0].update(turn_id="t9"), "turn"),
    "absent": (lambda raw: _facts(raw)[0].update(writing="Bergen", answer="Bergen"), "not in its turn"),
    "earlier": (lambda raw: (_turns(raw)[1].update(text="Oslo has 12 GB."), _facts(raw)[0].update(turn_id="t2")),
                "earlier turn"),
    "empty answer": (lambda raw: _facts(raw)[1].update(answer=""), "answer"),
    "code key": (lambda raw: _facts(raw).append(
        {"class": "code", "turn_id": "t2", "writing": "12 GB", "answer": "0" * 12}), "code key"),
    "negated": (lambda raw: _facts(raw)[3].update(negated=0), "negated"),
    "no act": (lambda raw: _facts(raw)[3].update(acts=[]), "acts"),
    "field": (lambda raw: _facts(raw)[0].update(guess=True), "field"),
    "tag": (lambda raw: _facts(raw)[0].update(tags=["guess"]), "tag"),
    "twice": (lambda raw: _facts(raw).append(dict(_facts(raw)[0])), "twice"),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_ra9_a_malformed_labelled_set_is_refused_by_name(case):
    probes, peels, restore = _window()
    try:
        mutate, named = _MALFORMED[case]
        raw = _valid()
        mutate(raw)
        with pytest.raises(probes.LabelledSetError) as refused:
            probes.labelled_spans(raw)
        assert named in str(refused.value)
        probes.labelled_spans(_valid())
    finally:
        restore()


def test_ra10_a_refusal_names_where_the_defect_lies_never_what_the_text_says():
    probes, peels, restore = _window()
    try:
        raw = _valid()
        raw["spans"][0]["span_id"] = "m-7"
        raw["spans"][0]["turns"][0]["text"] = "Zebulon Quartz keeps the Oslo office from 2027-03-14."
        raw["spans"][0]["facts"] = [{"class": "entity", "turn_id": "t1", "writing": "Yolanda Ferrix",
                                     "answer": "Yolanda Ferrix"}]
        with pytest.raises(probes.LabelledSetError) as refused:
            probes.labelled_spans(raw)
        said = str(refused.value)
        assert "m-7" in said and "t1" in said and "fact 0" in said
        assert not any(word in said for word in ("Zebulon", "Quartz", "Yolanda", "Ferrix", "Oslo"))
    finally:
        restore()


def test_ra11_read_again_on_the_fixture_set_the_recall_is_the_one_onion_yaml_states():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        reading = probes.measure_recall(probes.load_labelled(_SET), gate.lexicon)
        assert gate.probe_recall is not None
        assert _fields(reading.probe_recall) == _fields(gate.probe_recall)
        assert gate.probe_recall.generator == probes.GENERATOR_VERSION
        assert gate.probe_recall.lexicon == gate.lexicon.fingerprint
    finally:
        restore()


def test_ra12_every_fact_the_set_marks_as_a_known_miss_is_missed():
    probes, peels, restore = _window()
    try:
        spans = probes.load_labelled(_SET)
        known = {(span["span_id"], index) for span in spans for index, fact in enumerate(span["facts"])
                 if "known-miss" in fact.get("tags", ())}
        reading = probes.measure_recall(spans, peels.load_gate().lexicon)
        missed = {(span_id, index) for span_id, _turn_id, _kind, index in reading.missed}
        assert len(known) == 4
        assert known <= missed
    finally:
        restore()


# ---------------------------------------------------------------------------
# The stated recall and the gate
# ---------------------------------------------------------------------------


def test_ra13_the_gate_loads_the_stated_recall_field_by_field_in_the_fixed_order_of_its_classes(tmp_path):
    probes, peels, restore = _window()
    try:
        classes = {kind: {"size": 10 + i, "recalled": i} for i, kind in enumerate(probes.RECALL_CLASSES)}
        stated = {"source": "fixture", "generator": 7, "lexicon": "0123456789ab",
                  "classes": dict(reversed(list(classes.items())))}
        gate = _gate_with(peels, tmp_path, stated)
        assert _fields(gate.probe_recall) == ("fixture", 7, "0123456789ab", tuple(
            (kind, i, 10 + i) for i, kind in enumerate(probes.RECALL_CLASSES)))
    finally:
        restore()


def _bad_classes(change):
    def mutate(stated):
        change(stated["classes"])
    return mutate


_BAD_RECALL = {
    "not a mapping": (lambda stated: stated.clear() or stated.update(_list=True), "recall holds"),
    "missing key": (lambda stated: stated.pop("lexicon"), "recall holds"),
    "extra key": (lambda stated: stated.update(rate=0.9), "recall holds"),
    "source": (lambda stated: stated.update(source="labelled"), "source"),
    "generator bool": (lambda stated: stated.update(generator=True), "generator"),
    "generator zero": (lambda stated: stated.update(generator=0), "generator"),
    "generator text": (lambda stated: stated.update(generator="5"), "generator"),
    "lexicon short": (lambda stated: stated.update(lexicon="0123"), "lexicon"),
    "lexicon upper": (lambda stated: stated.update(lexicon="0123456789AB"), "lexicon"),
    "lexicon number": (lambda stated: stated.update(lexicon=123456789012), "lexicon"),
    "class missing": (_bad_classes(lambda classes: classes.pop("code")), "classes"),
    "class unknown": (_bad_classes(lambda classes: classes.update(place={"size": 1, "recalled": 1})), "classes"),
    "row": (_bad_classes(lambda classes: classes.update(code={"size": 4})), "code"),
    "size zero": (_bad_classes(lambda classes: classes.update(code={"size": 0, "recalled": 0})), "size"),
    "size bool": (_bad_classes(lambda classes: classes.update(code={"size": True, "recalled": 1})), "size"),
    "over size": (_bad_classes(lambda classes: classes.update(code={"size": 4, "recalled": 5})), "recalled"),
    "negative": (_bad_classes(lambda classes: classes.update(code={"size": 4, "recalled": -1})), "recalled"),
}


@pytest.mark.parametrize("case", sorted(_BAD_RECALL))
def test_ra14_a_malformed_stated_recall_is_refused_by_name(case, tmp_path):
    probes, peels, restore = _window()
    try:
        mutate, named = _BAD_RECALL[case]
        stated = _stated(probes, 5, "0123456789ab")
        mutate(stated)
        if "_list" in stated:
            stated = ["fixture", 5]
        with pytest.raises(peels.GateError) as refused:
            _gate_with(peels, tmp_path, stated)
        assert named in str(refused.value)
        assert _gate_with(peels, tmp_path, _stated(probes, 5, "0123456789ab")).probe_recall is not None
    finally:
        restore()


def test_ra15_a_gate_file_that_states_no_recall_loads_a_gate_with_none_and_its_decisions_say_so(tmp_path):
    probes, peels, restore = _window()
    try:
        gate = _gate_with(peels, tmp_path, _DROP)
        assert gate.probe_recall is None
        span = _span()
        decision = peels.decide(span, probes.generate_probes(span, gate.lexicon), span[0]["text"], gate)
        assert decision.probe_recall is None
        assert "no recall stated" in decision.probe_recall_note
    finally:
        restore()


def test_ra16_a_decision_of_the_shipped_gate_carries_the_stated_recall():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        span = _span()
        decision = peels.decide(span, probes.generate_probes(span, gate.lexicon), span[0]["text"], gate)
        assert decision.accepted
        assert _fields(decision.probe_recall) == _fields(gate.probe_recall)
        assert decision.probe_recall_note == ""
    finally:
        restore()


def test_ra17_a_recall_read_with_another_generator_is_not_carried_and_both_versions_are_named(tmp_path):
    probes, peels, restore = _window()
    try:
        version = probes.GENERATOR_VERSION
        lexicon = peels.load_gate().lexicon.fingerprint
        gate = _gate_with(peels, tmp_path, _stated(probes, version + 1, lexicon))
        span = _span()
        decision = peels.judge(probes.generate_probes(span, gate.lexicon), span[0]["text"], gate)
        assert decision.probe_recall is None
        assert f"generator {version}" in decision.probe_recall_note
        assert f"generator {version + 1}" in decision.probe_recall_note
    finally:
        restore()


def test_ra18_a_recall_read_with_another_lexicon_is_not_carried_and_both_fingerprints_are_named(tmp_path):
    probes, peels, restore = _window()
    try:
        gate = _gate_with(peels, tmp_path, _stated(probes, probes.GENERATOR_VERSION, "0" * 12))
        span = _span()
        decision = peels.judge(probes.generate_probes(span, gate.lexicon), span[0]["text"], gate)
        assert decision.probe_recall is None
        assert gate.lexicon.fingerprint in decision.probe_recall_note
        assert "0" * 12 in decision.probe_recall_note
    finally:
        restore()


def test_ra19_a_gate_built_by_hand_carries_no_recall_and_its_decisions_say_so():
    probes, peels, restore = _window()
    try:
        gate = peels.Gate(0.9, 0.7, 4)
        assert gate.probe_recall is None
        span = _span()
        decision = peels.judge(probes.generate_probes(span), span[0]["text"], gate)
        assert decision.probe_recall is None
        assert "no recall stated" in decision.probe_recall_note
    finally:
        restore()


def test_ra20_a_summary_the_second_face_refuses_carries_the_recall_too():
    probes, peels, restore = _window()
    try:
        gate = peels.load_gate()
        span = _span()
        text = span[0]["text"] + " They met with Dave in Oslo."
        decision = peels.decide(span, probes.generate_probes(span, gate.lexicon), text, gate)
        assert not decision.accepted and decision.unsupported
        assert _fields(decision.probe_recall) == _fields(gate.probe_recall)
        assert decision.probe_recall_note == ""
    finally:
        restore()


def test_ra21_a_fidelity_reading_states_the_recall_that_holds_for_its_probes_and_never_another(tmp_path):
    probes, peels, restore = _window()
    try:
        receipts = sys.modules["opti_oignon.memory.receipts"]
        gate = peels.load_gate()
        tree, cellar = peels.PeelTree(), receipts.Cellar()
        held = peels.fidelity(tree, cellar, gate.lexicon, gate.probe_recall)
        assert held["probe_recall"] == gate.probe_recall.entry() and held["probe_recall_note"] == ""
        assert held["probe_recall"]["classes"]["code"]["size"] == dict(
            (kind, size) for kind, _recalled, size in gate.probe_recall.classes)["code"]
        other = _gate_with(peels, tmp_path, _stated(probes, probes.GENERATOR_VERSION + 1, gate.lexicon.fingerprint))
        moved = peels.fidelity(tree, cellar, gate.lexicon, other.probe_recall)
        assert moved["probe_recall"] is None and f"generator {probes.GENERATOR_VERSION + 1}" in moved["probe_recall_note"]
        none = peels.fidelity(tree, cellar, gate.lexicon)
        assert none["probe_recall"] is None and "no recall stated" in none["probe_recall_note"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# The runbook
# ---------------------------------------------------------------------------


def test_ra22_every_rate_the_runbook_reports_stands_beside_the_recall_that_holds_for_it(tmp_path):
    probes, peels, restore = _window()
    try:
        runbook = _runbook()
        receipts = sys.modules["opti_oignon.memory.receipts"]
        gate = peels.load_gate()
        stated = {"probe_recall": gate.probe_recall.entry(), "probe_recall_note": ""}
        tree, cellar = peels.PeelTree(), receipts.Cellar()
        at_gate = runbook._at_gate(3, 1, tree, cellar, gate)
        row = runbook._sweep_row(4, 3, gate)
        assert {k: runbook._gate_entry(gate)[k] for k in stated} == stated
        assert {k: at_gate[k] for k in stated} == stated
        assert {k: at_gate["fidelity"][k] for k in stated} == stated
        assert {k: row[k] for k in stated} == stated
        assert (at_gate["spans_accepted"], at_gate["spans_refused"], row["spans"], row["accepted"]) == (3, 1, 4, 3)
        other = _gate_with(peels, tmp_path, _stated(probes, probes.GENERATOR_VERSION + 1, gate.lexicon.fingerprint))
        assert runbook._sweep_row(4, 3, other)["probe_recall"] is None
        assert f"generator {probes.GENERATOR_VERSION}" in runbook._sweep_row(4, 3, other)["probe_recall_note"]
    finally:
        restore()


def test_ra23_the_runbook_reads_a_labelled_set_for_its_counts_only(capsys):
    probes, peels, restore = _window()
    try:
        runbook = _runbook()
        assert runbook.main(["--probe-recall", str(_SET)]) == 0
        out = capsys.readouterr().out
        report = json.loads(out)
        spans = probes.load_labelled(_SET)
        reading = probes.measure_recall(spans, peels.load_gate().lexicon)
        assert report["source"] == "labelled" and report["spans"] == len(spans)
        assert report["classes"] == {name: {"size": size, "recalled": recalled,
                                            "recall": round(recalled / size, 4)}
                                     for name, recalled, size in reading.probe_recall.classes}
        texts = [fact["writing"] for span in spans for fact in span["facts"]]
        texts += [turn["text"] for span in spans for turn in span["turns"]]
        forms = [(text, json.dumps(text)[1:-1]) for text in texts if len(text) >= 4]
        leaked = [text[:24] for text, escaped in forms if text in out or escaped in out]
        assert len(forms) > 100
        assert leaked == []
    finally:
        restore()
