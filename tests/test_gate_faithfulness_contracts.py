#!/usr/bin/env python3
"""Contracts for the second face of the eviction gate: what a summary says
that its span does not hold.

The first face asks the summary the probes drawn from its span. A summary
can answer every one of them and still add what no turn said. The second
face reads the summary as a source is read and refuses, by name, every
claim its span does not hold: a name, a date, a number, a code block or its
marker, and a sentence that decides without a typed decision behind it.
The span is read as leniently as the reader allows -- every origin, every
block, the head of a sentence and inline code included, a role as a word
-- and the summary as strictly: a claim is held only where the span holds
it. A summary may say less than its span, a month for a day or a number
without its unit; it may not say more.

  * GF1 -- a name the summary gives that its span does not hold is refused
    by name, though the summary answers every probe of the first face.
  * GF2 -- a name is held wherever its span holds its words: at the head of
    a sentence, in another case or without its accents, in inline code or a
    fenced block, in the assistant's words or a document's, or as a role.
  * GF3 -- a date and a number the span does not hold are refused by name,
    each once, names before dates before numbers.
  * GF4 -- a number is held by its reading: in another writing, without the
    unit its span gives it, as a part of a date of its span, or in its code;
    another unit is refused, never guessed.
  * GF5 -- a date is held by its reading or by a finer date of its span: in
    either language, as the month of a day, as the day without its year; a
    date finer than its span's, or one put for a relative date, is refused.
  * GF6 -- a code marker or a fenced block the span holds no block for is
    refused by name; the span's own marker, or its block verbatim, is held.
  * GF7 -- a sentence of the summary that decides needs a typed decision of
    its span behind it: one the span never makes is refused, and so is one
    that only the assistant or a document makes.
  * GF8 -- a deciding sentence that inverts a typed decision is refused as an
    inversion, naming the turn it inverts.
  * GF9 -- a sentence whose subject is a reporter (the assistant, a
    document, a tool) reports another speaker's words and needs no typed
    decision, in either language; only its subject makes it a report, and a
    gate without reporters holds every deciding sentence to a typed decision.
  * GF10 -- a sentence that does not decide needs no typed decision, whoever
    said its words.
  * GF11 -- code is a class of its own: at the shipped gate one lost marker
    refuses the span while every other class meets its threshold; a gate
    built without a code threshold judges code with the episodic class and
    gives no code rate.
  * GF12 -- the gate reads its code threshold and its reporters from
    ``onion.yaml``, and refuses by name a threshold that is missing, a
    boolean, not a number, not finite or outside [0, 1], and reporters that
    are missing or malformed; an empty table of reporters is a stricter
    gate, read as one.
  * GF13 -- the eviction judges with both faces: a summary that answers every
    probe but gives a name its span does not hold leaves the verbatim turns
    in place, and the decision names it.
  * GF14 -- a leaf and a parent are judged with both faces.
  * GF15 -- the finding GE16 recorded is closed: at the proposed defaults a
    single entity swap and a single date shift still pass the first face,
    and the second refuses each of them by name, and nothing else.
  * GF16 -- the quieting the next contracts run on changes a fixture's
    decision and nothing else: the same lengths, the same probes, and the
    loud fixture is refused for its decision alone.
  * GF17 to GF42 -- FM1, FM2, OR4, LB4, LB5, LB6, LB8, LB15 to LB19, RO1 to
    RO4, OS1, OS2, OS3, OS5 and PT1 to PT6 held word for word on their
    fixtures quieted: their turns of no origin asserted a decision no probe
    ever asked, which the second face now refuses to see in a summary. FM1
    loses its name where it swapped it, a swap being refused whatever the
    thresholds.

The second face has two bounds besides its claims. A summary whose content
words its span mostly does not hold has drifted from it, whatever it
answers; a summary longer than its span saves nothing, and the verbatim
turns are both shorter and exact. Each bound refuses by name, with its
figure, in the reason and never as a claim.

  * GF43 -- a summary whose content words its span mostly does not hold is
    refused with its novelty and the new words, though both faces let it
    through; a faithful one has none.
  * GF44 -- a summary longer than its span is refused with its length ratio
    and its counts; word for word is exactly the bound, and one word over it
    is over.
  * GF45 -- novelty counts no function word, no figure, no word decisions
    are told with (the lexicon, the markers), no reporter, no code marker and
    no fenced code, in either language; a summary with no content word has
    no novelty and is refused nothing for it.
  * GF46 -- a word is held by its key: its final s taken off, then its first
    five letters; a word of another key is new.
  * GF47 -- the length ratio is every word of the summary over every word of
    its span's texts, code included and roles not; a span with no word has
    no ratio.
  * GF48 -- the gate reads its bounds from ``onion.yaml`` and refuses by name
    a bound that is missing, null, a boolean, not a number, not finite, or
    out of its range: [0, 1] for novelty, above 0 for the length ratio.
  * GF49 -- a gate built by hand holds no bound: its figures are measured and
    said, the bounds it was held to are None, and it refuses nothing for them.
  * GF50 -- a bound out of its range is refused by name before any summary
    is judged.
  * GF51 -- the eviction, a leaf and a parent are held to the bounds.
  * GF52 -- a summary refused on every count gives every reason, the first
    face's, then its claims, then its bounds, and only its claims as claims.
  * GF53 -- the runbook reports each bound and the spread of its figures,
    span by span: measured, unmeasured, over the bound, and the nearest-rank
    minimum, median, ninetieth percentile and maximum.
  * GF54 -- the librarian is asked to write in the language of the turns and
    to attribute each decision to its source.
  * GF55 -- at the proposed defaults, a faithful summary that tells who said
    what answers both faces and stays far under the novelty bound, yet is
    longer than a short span and refused for its length alone: the verbatim
    turns stay, shorter and exact.
  * GF56 -- inline code in a summary is held only by the same inline code of
    its span: a number, a name, a command or a negation in backticks its span
    never wrote so is refused by name as code; the span's own inline code,
    its flags included, is held.
  * GF57 -- a negation written as inline code inverts a decision as a plain
    one does, whatever inline code the span holds elsewhere; a flag in code,
    such as --no-cache, still reads as no negation.
  * GF58 -- a negation inside longer inline code ("will not", "won't",
    "don't", "not in") inverts a decision as a plain one does, though an
    attached document or the assistant holds the same code; a faithful
    summary that copies a flag such as --no-cache is held, a negation fused
    into a flag being none.
  * GF59 -- a negation fused into a flag reads as none in prose as in code: a
    summary that drops the flag's backticks is held, and one that also drops
    the decision's negation is refused.
  * GF60 -- GF47 held on a code turn that shows its block rather than orders
    it run: the second face now refuses a summary that repeats an order the
    user did not type, the assistant's included, so GF47's fixture gives way
    to one that tells; every figure is the one GF47 states.
  * GF61 -- GF58 held on a planted sentence that tells rather than orders:
    the planted imperative, held by a document or the assistant, is now an
    order the user did not type; the negation each inline code carries is
    the one GF58 holds to.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import ast
import dataclasses
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION = REPO / "opti_oignon" / "config" / "onion.yaml"
_E = chr(0xE9)
_EG = chr(0xE8)


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


def _turn(turn_id, role, origin, text):
    return {"turn_id": turn_id, "role": role, "origin": origin, "text": text}


def _verbatim(span):
    return " ".join(t["text"] for t in span)


def _second(probes, peels, span, text, gate):
    """What the second face finds in ``text`` against ``span``."""
    return peels.faithfulness(span, probes.generate_probes(span, gate.lexicon), text, gate)


def _kinds(found):
    return [(kind, what) for kind, what, _turn_id in found]


_SPAN = [
    _turn("t1", "user", "typed",
          "On 2026-03-04, Alice and Bob met in Oslo about the Harvest release. We keep Docker for the build farm."),
    _turn("t2", "assistant", "assistant", "Noted: the farm has 16 Go per host, and Carol owns the Oslo account."),
]
_ADDED = " They met with Dave in Oslo."
_TAIL = [_turn(f"x{i}", "user", "typed", f"Service {i} moved on day {i}.") for i in range(1, 4)]


# ---------------------------------------------------------------------------
# GF1 -- a name the span does not hold
# ---------------------------------------------------------------------------
def test_gf1_a_name_its_span_does_not_hold_is_refused_by_name_though_every_probe_is_answered():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_SPAN, gate.lexicon)
        faithful = _verbatim(_SPAN)
        added = faithful + _ADDED
        first = peels.judge(drawn, added, gate)
        assert first.accepted is True and first.result.failed == 0, "control: the first face lets the name through"
        held = peels.decide(_SPAN, drawn, faithful, gate)
        assert held.accepted is True and held.unsupported == (), held.reason
        refused = peels.decide(_SPAN, drawn, added, gate)
        assert refused.accepted is False
        assert refused.unsupported == (("entity", "Dave", ""),)
        assert "entity:Dave" in refused.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF2 -- a name is held wherever the span holds its words
# ---------------------------------------------------------------------------
_HELD = [
    _turn("h1", "user", "typed", "Zo" + _E + " opens the review today. Run `kubectl apply` on the cluster first."),
    _turn("h2", "assistant", "assistant", "The server is ready:\n\n```python\nimport fastapi\n```"),
    _turn("h3", "user", "document", "Invoice from contoso for the cluster."),
]


def test_gf2_a_name_is_held_wherever_its_span_holds_its_words():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        summary = "The review is with Zoe; the Assistant ran Kubectl with FASTAPI for Contoso."
        assert _second(probes, peels, _HELD, summary, gate) == ()
        found = _second(probes, peels, _HELD, summary.replace("Contoso", "Mallory"), gate)
        assert _kinds(found) == [("entity", "Mallory")], "control: the face reads these names"
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF3 -- a date and a number the span does not hold
# ---------------------------------------------------------------------------
def test_gf3_a_date_and_a_number_its_span_does_not_hold_are_refused_by_name_each_once_in_order():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        summary = _verbatim(_SPAN) + " Later Dave counted 5 hosts on 2026-03-05, and Dave saw 5 racks."
        found = _second(probes, peels, _SPAN, summary, gate)
        assert _kinds(found) == [("entity", "Dave"), ("date", "2026-03-05"), ("number", "5")]
        decision = peels.decide(_SPAN, probes.generate_probes(_SPAN, gate.lexicon), summary, gate)
        assert decision.accepted is False
        assert "date:2026-03-05" in decision.reason and "number:5" in decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF4 -- a number is held by its reading
# ---------------------------------------------------------------------------
_NUMBERS = [
    _turn("n1", "user", "typed", "The farm has 16 Go per host, 1 000 users and 2,5 km of cable since 7 octobre 2026."),
    _turn("n2", "assistant", "assistant", "Start it with:\n\n```js\napp.listen(8080)\n```"),
]


def test_gf4_a_number_is_held_by_its_reading_and_another_unit_is_refused():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        for sentence in (
            "Each host has 16 GB.",
            "There are 1000 users.",
            "The cable runs 2.5 km.",
            "Memory is 16 per host.",
            "It started in 2026.",
            "It started on day 7.",
            "The app listens on 8080.",
        ):
            assert _second(probes, peels, _NUMBERS, sentence, gate) == (), sentence
        found = _second(probes, peels, _NUMBERS, "Each host has 16 GiB.", gate)
        assert _kinds(found) == [("number", "16 GiB")]
        found = _second(probes, peels, _NUMBERS, "There are 1200 users.", gate)
        assert _kinds(found) == [("number", "1200")]
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF5 -- a date is held by its reading or by a finer date
# ---------------------------------------------------------------------------
_DAY = [_turn("d1", "user", "typed", "The audit is on 2026-03-04.")]
_MONTH = [_turn("d2", "user", "typed", "The launch is in March 2026.")]
_RELATIVE = [_turn("d3", "user", "typed", "The demo is tomorrow.")]


def test_gf5_a_date_is_held_by_its_reading_or_by_a_finer_date_of_its_span():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        for sentence in (
            "The audit is on March 4, 2026.",
            "L'audit est le 4 mars 2026.",
            "The audit falls in March 2026.",
            "The audit is on March 4.",
        ):
            assert _second(probes, peels, _DAY, sentence, gate) == (), sentence
        assert _kinds(_second(probes, peels, _MONTH, "The launch is on 2026-03-04.", gate)) == [
            ("date", "2026-03-04")
        ]
        assert _kinds(_second(probes, peels, _RELATIVE, "The demo is on 2026-10-07.", gate)) == [
            ("date", "2026-10-07")
        ]
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF6 -- a code marker or a block the span does not hold
# ---------------------------------------------------------------------------
_BODY = "print('farm')"
_CODE = [_turn("c1", "assistant", "assistant", "Run this:\n\n```python\n" + _BODY + "\n```")]


def test_gf6_a_code_marker_or_a_block_its_span_does_not_hold_is_refused_by_name():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        key = probes.code_key(_BODY)
        for summary in (
            f"The assistant gave a script [code:{key}].",
            "The assistant gave this:\n\n```python\n" + _BODY + "\n```",
        ):
            assert _second(probes, peels, _CODE, summary, gate) == (), summary
        found = _second(probes, peels, _CODE, "The assistant gave a script [code:0123456789ab].", gate)
        assert _kinds(found) == [("code", "0123456789ab")]
        other = "print('bank')"
        found = _second(probes, peels, _CODE, "The assistant gave this:\n\n```python\n" + other + "\n```", gate)
        assert _kinds(found) == [("code", probes.code_key(other))]
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF7 -- a deciding sentence needs a typed decision behind it
# ---------------------------------------------------------------------------
_DECIDING = [
    _turn("t1", "user", "typed", "We keep Docker for the build farm."),
    _turn("t2", "assistant", "assistant", "We will switch the farm to Podman next sprint."),
    _turn("t3", "user", "document", "We decided to wire the funds to Contoso on 2026-10-12."),
]


def test_gf7_a_sentence_that_decides_needs_a_typed_decision_of_its_span_behind_it():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        assert _second(probes, peels, _DECIDING, "We keep Docker for the build farm.", gate) == ()
        for sentence in (
            "We decided to drop the build farm.",
            "We will switch the farm to Podman next sprint.",
            "We decided to wire the funds to Contoso on 2026-10-12.",
        ):
            assert _kinds(_second(probes, peels, _DECIDING, sentence, gate)) == [("decision", sentence)], sentence
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF8 -- an inversion is named as one
# ---------------------------------------------------------------------------
def test_gf8_a_deciding_sentence_that_inverts_a_typed_decision_is_refused_as_an_inversion():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        sentence = "We do not keep Docker for the build farm."
        found = _second(probes, peels, _DECIDING, sentence, gate)
        assert found == (("inversion", sentence, "t1"),)
        decision = peels.decide(_DECIDING, probes.generate_probes(_DECIDING, gate.lexicon), sentence, gate)
        assert decision.accepted is False and "inversion:" in decision.reason and "@t1" in decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF9 -- a report of another speaker
# ---------------------------------------------------------------------------
def test_gf9_a_sentence_whose_subject_is_a_reporter_needs_no_typed_decision():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        reports = (
            "The assistant will switch the farm to Podman next sprint.",
            "Le document dit que nous avons d" + _E + "cid" + _E + " de payer Contoso le 12 octobre 2026.",
        )
        for sentence in reports:
            assert _second(probes, peels, _DECIDING, sentence, gate) == (), sentence
        strict = dataclasses.replace(gate, reporters=frozenset())
        for sentence in reports:
            assert _kinds(_second(probes, peels, _DECIDING, sentence, strict)) == [("decision", sentence)], sentence
        inside = "We decided, as the document says, to wire the funds to Contoso on 2026-10-12."
        assert _kinds(_second(probes, peels, _DECIDING, inside, gate)) == [("decision", inside)]
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF10 -- a sentence that does not decide
# ---------------------------------------------------------------------------
_REPORTED = [
    _turn("r1", "assistant", "assistant",
          "Bob reviews the release on 2026-05-02 and the rollback keeps the old cluster warm for 7 days."),
]


def test_gf10_a_sentence_that_does_not_decide_needs_no_typed_decision():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        assert _second(probes, peels, _REPORTED, _verbatim(_REPORTED), gate) == ()
        control = "We decided to keep the old cluster warm."
        assert _kinds(_second(probes, peels, _REPORTED, control, gate)) == [("decision", control)], (
            "control: the face reads decisions in this span"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF11 -- code is a class of its own
# ---------------------------------------------------------------------------
_BUILD, _TEST = "make build", "make test"
_TWO_BLOCKS = [
    _turn("k1", "user", "typed",
          "On 2026-03-04, Alice and Bob met in Oslo about the Harvest release with 3 reviewers."),
    _turn("k2", "assistant", "assistant",
          "First:\n\n```sh\n" + _BUILD + "\n```\n\nThen:\n\n```sh\n" + _TEST + "\n```"),
]


def test_gf11_code_is_a_class_of_its_own_at_the_shipped_gate():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_TWO_BLOCKS, gate.lexicon)
        assert sum(1 for p in drawn if p.kind == "code") == 2, "control: two code probes"
        summary = _TWO_BLOCKS[0]["text"] + " First " + probes.code_marker(_BUILD) + "."
        decision = peels.decide(_TWO_BLOCKS, drawn, summary, gate)
        assert decision.accepted is False and decision.unsupported == ()
        assert decision.code_rate == 0.5 and decision.episodic_rate == 1.0
        assert "code probes at 0.5 against 1.0" in decision.reason and "episodic" not in decision.reason
        built = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=4, lexicon=gate.lexicon)
        loose = peels.judge(drawn, summary, built)
        assert loose.accepted is True and loose.code_rate is None
        assert loose.episodic_rate == round(7 / 8, 4)
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF12 -- the code threshold and the reporters are the YAML's
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


def test_gf12_the_code_threshold_and_the_reporters_are_read_from_onion_yaml_and_refused_by_name(tmp_path):
    _probes, peels, _receipts, restore = _window()
    try:
        raw = yaml.safe_load(_ONION.read_text(encoding="utf-8"))
        gate = peels.load_gate()
        assert gate.code_threshold == raw["gate"]["code_threshold"] == 1.0
        assert {"assistant", "document", "documents", "outil", "outils"} <= gate.reporters
        assert gate.validate() == []
        for mutate in (
            _drop("gate", "code_threshold"),
            _set("gate", "code_threshold", True),
            _set("gate", "code_threshold", "high"),
            _set("gate", "code_threshold", float("nan")),
            _set("gate", "code_threshold", 1.5),
            _set("gate", "code_threshold", -0.1),
        ):
            with pytest.raises(peels.GateError, match="code_threshold"):
                _loaded(peels, tmp_path, mutate)
        for mutate in (
            _drop("gate", "reporters"),
            _set("gate", "reporters", None),
            _set("gate", "reporters", ["assistant"]),
            _set("gate", "reporters", {"de": ["dokument"]}),
            _set("gate", "reporters", {"en": "assistant"}),
            _set("gate", "reporters", {"en": ["Assistant"]}),
            _set("gate", "reporters", {"en": ["mod" + _EG + "le"]}),
            _set("gate", "reporters", {"en": ["search engine"]}),
        ):
            with pytest.raises(peels.GateError, match="reporters"):
                _loaded(peels, tmp_path, mutate)
        assert _loaded(peels, tmp_path, _set("gate", "reporters", {})).reporters == frozenset(), (
            "an empty table is a stricter gate, never a malformed one"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF13 -- the eviction judges with both faces
# ---------------------------------------------------------------------------
def _setup(peels, receipts, gate, span):
    return dict(flesh=receipts.Flesh(span + _TAIL), cellar=receipts.Cellar(), ledger=receipts.ReceiptLedger(),
                tree=peels.PeelTree(), gate=dataclasses.replace(gate, span_turns=len(span)))


def test_gf13_the_eviction_judges_with_both_faces():
    _probes, peels, receipts, restore = _window()
    try:
        gate = peels.load_gate()
        s = _setup(peels, receipts, gate, _SPAN)
        outcome = peels.evict_gated(summarize=lambda span: _verbatim(span) + _ADDED, **s)
        assert outcome.evicted is False and outcome.receipt is None and outcome.peel is None
        assert len(s["flesh"].turns()) == len(_SPAN) + len(_TAIL), "the verbatim turns stay"
        assert s["ledger"].all() == [] and len(s["cellar"]) == 0 and s["tree"].all() == []
        assert outcome.decision.result.failed == 0, "control: every probe was answered"
        assert outcome.decision.unsupported == (("entity", "Dave", ""),)
        assert "entity:Dave" in outcome.reason
        s = _setup(peels, receipts, gate, _SPAN)
        assert peels.evict_gated(summarize=_verbatim, **s).evicted is True, "control: the faithful summary evicts"
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF14 -- a leaf and a parent are judged with both faces
# ---------------------------------------------------------------------------
def test_gf14_a_leaf_and_a_parent_are_judged_with_both_faces():
    _probes, peels, receipts, restore = _window()
    try:
        gate = peels.load_gate()
        s = _setup(peels, receipts, gate, _SPAN)
        evicted = peels.evict_gated(summarize=_verbatim, **s)
        assert evicted.evicted is True, evicted.reason
        added = lambda turns: _verbatim(turns) + _ADDED  # noqa: E731
        leaf, decision = peels.build_leaf(evicted.receipt.key, s["cellar"], added, s["gate"], s["tree"])
        assert leaf is None and decision.unsupported == (("entity", "Dave", ""),)
        parent, decision = peels.build_parent([evicted.peel.id], s["cellar"], added, s["gate"], s["tree"])
        assert parent is None and decision.unsupported == (("entity", "Dave", ""),)
        assert len(s["tree"].all()) == 1, "nothing was added"
        leaf, decision = peels.build_leaf(evicted.receipt.key, s["cellar"], _verbatim, s["gate"], s["tree"])
        assert leaf is not None, decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF15 -- the finding of GE16, closed
# ---------------------------------------------------------------------------
# The typed fixture of tests/test_gated_eviction_contracts.py, word for word.
_GE_TYPED_SPAN = [
    {"turn_id": "t01", "role": "user", "origin": "typed",
     "text": "On 2026-03-04, Alice and Bob met to review the Harvest release with a budget of 1200 euros."},
    {"turn_id": "t02", "role": "assistant", "origin": "assistant",
     "text": "The venue in Oslo has no container runtime, and Carol handles the Oslo account with a latency "
             "target of 45 milliseconds."},
    {"turn_id": "t03", "role": "user", "origin": "typed",
     "text": "We agreed that the demo will not use Docker for the Oslo venue."},
    {"turn_id": "t04", "role": "assistant", "origin": "assistant",
     "text": "Bob reviews the release on 2026-05-02 and the rollback keeps the old cluster warm for 7 days."},
]


def test_gf15_one_swap_and_one_shift_pass_the_first_face_and_are_refused_by_the_second():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_GE_TYPED_SPAN, gate.lexicon)
        for old, new, kind in (("Alice", "Dave", "entity"), ("2026-03-04", "2026-04-04", "date")):
            summary = _verbatim(_GE_TYPED_SPAN).replace(old, new)
            first = peels.judge(drawn, summary, gate)
            assert first.result.failed == 1 and first.episodic_rate == round(12 / 13, 4), "the GE16 figures"
            assert first.accepted is True, "the first face alone lets it through"
            decision = peels.decide(_GE_TYPED_SPAN, drawn, summary, gate)
            assert decision.accepted is False
            assert decision.unsupported == ((kind, new, ""),)
            assert f"{kind}:{new}" in decision.reason
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF16-GF42 -- twenty-six contracts held on turns that decide nothing
# ---------------------------------------------------------------------------
# Their fixtures say "we agreed that service N stays on the new cluster" in
# turns of no origin. No probe ever asked that decision -- a turn of no
# origin decides nothing -- and the second face now refuses a summary that
# asserts it. Each contract runs word for word on the same fixture, its
# words of decision replaced by words of the same length that decide
# nothing, so that every count, size and figure it reads stays the same.
_QUIET = (("agreed", "stated"), ("stays", "lives"))


def _quiet(text):
    for old, new in _QUIET:
        text = text.replace(old, new)
    return text


def _quieted(monkeypatch, name):
    """The suite ``name``, its fixture's turns quieted for the length of the contract."""
    module = importlib.import_module(name)
    for fixture, key in (("_messages", "content"), ("_span", "text")):
        made = getattr(module, fixture, None)
        if made is not None:
            monkeypatch.setattr(module, fixture, lambda n, made=made, key=key: [
                dict(turn, **{key: _quiet(turn[key])}) for turn in made(n)
            ])
    return module


def test_gf16_the_quieting_changes_the_decision_and_nothing_else():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        messages = importlib.import_module("test_librarian_contracts")._messages(4)
        for loud in (
            importlib.import_module("test_peel_tree_contracts")._span(1),
            importlib.import_module("test_memory_fidelity_contracts")._span(2),
            [{"turn_id": str(i), "role": m["role"], "text": m["content"]} for i, m in enumerate(messages)],
        ):
            quiet = [dict(turn, text=_quiet(turn["text"])) for turn in loud]
            assert [len(t["text"]) for t in quiet] == [len(t["text"]) for t in loud]
            drawn = probes.generate_probes(loud, gate.lexicon)
            assert [(p.kind, p.answer, p.turn_id) for p in drawn] == [
                (p.kind, p.answer, p.turn_id) for p in probes.generate_probes(quiet, gate.lexicon)
            ], "the same probes: no word of decision was ever probed"
            assert {kind for kind, _what, _turn in peels.faithfulness(loud, drawn, _verbatim(loud), gate)} == {
                "decision"
            }, "the loud fixture is refused for its decision alone"
            assert peels.faithfulness(quiet, drawn, _verbatim(quiet), gate) == ()
    finally:
        restore()


def test_gf17_fm1_holds_on_turns_that_decide_nothing_with_a_name_lost_not_swapped(monkeypatch):
    # FM1 built its lossy peels by swapping a name, which the second face now
    # refuses whatever the thresholds (GF13): here the name is lost instead,
    # and every assertion of FM1 stands as it was.
    fm = _quieted(monkeypatch, "test_memory_fidelity_contracts")
    peels, receipts, restore = fm._open()
    try:
        cellar, tree = fm._tree(peels, receipts, fm._faithful)
        faithful = peels.fidelity(tree, cellar)
        assert faithful["source"] == "fixture"
        assert faithful["peels"] == 3 and faithful["probes"] >= 6
        assert faithful["rate"] == 1.0
        lossy = peels.Gate(decision_threshold=0.5, episodic_threshold=0.5, span_turns=2)
        cellar, tree = fm._tree(peels, receipts, lambda span: fm._faithful(span).replace("Alice", "someone"), gate=lossy)
        lost = peels.fidelity(tree, cellar)
        assert 0.0 < lost["rate"] < 1.0
        assert lost["failed"] == 3, "one entity per peel, three peels"
        empty = peels.fidelity(peels.PeelTree(), cellar)
        assert empty["rate"] is None and empty["peels"] == 0, "no peel means unknown, never 0.0"
    finally:
        restore()


def test_gf18_fm2_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_memory_fidelity_contracts").test_fm2_the_multiplier_is_source_tokens_over_peel_tokens()


def test_gf19_or4_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_onion_routes_contracts").test_or4_the_handlers_reach_the_librarian_as_the_user_and_a_recall_leaves_the_receipt_open()


def test_gf20_lb4_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_librarian_contracts").test_lb4_curation_evicts_through_the_gate_until_the_flesh_fits()


def test_gf21_lb5_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_librarian_contracts").test_lb5_the_memory_block_carries_core_receipts_and_peels_as_data()


def test_gf22_lb6_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_librarian_contracts").test_lb6_the_memory_block_is_within_the_sum_of_its_layer_caps()


def test_gf23_lb8_holds_on_turns_that_decide_nothing(monkeypatch, tmp_path):
    _quieted(monkeypatch, "test_librarian_contracts").test_lb8_the_state_is_saved_and_comes_back_across_a_restart_with_its_cursor(tmp_path)


def test_gf24_lb15_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_librarian_contracts").test_lb15_a_close_empties_the_flesh_through_the_gate_and_returns_digest_and_root()


def test_gf25_lb16_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_librarian_contracts").test_lb16_a_refused_span_stops_the_close_and_stays_verbatim_with_its_failures_named()


def test_gf26_lb17_holds_on_turns_that_decide_nothing(monkeypatch, tmp_path):
    _quieted(monkeypatch, "test_librarian_contracts").test_lb17_a_close_is_saved_through_the_store_with_its_refused_remainder(tmp_path)


def test_gf27_lb18_holds_on_turns_that_decide_nothing(monkeypatch, tmp_path):
    _quieted(monkeypatch, "test_librarian_contracts").test_lb18_an_open_after_a_restart_returns_the_persisted_block_or_refuses_by_name(tmp_path)


def test_gf28_lb19_holds_on_turns_that_decide_nothing(monkeypatch, tmp_path):
    _quieted(monkeypatch, "test_librarian_contracts").test_lb19_a_recall_reads_the_span_a_user_close_is_saved_and_every_mutation_is_saved(tmp_path)


def test_gf29_ro1_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_recall_read_only_contracts").test_ro1_a_recall_through_the_route_hands_back_the_span_and_leaves_the_receipt_open()


def test_gf30_ro2_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_recall_read_only_contracts").test_ro2_the_librarian_s_recall_changes_no_receipt()


def test_gf31_ro3_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_recall_read_only_contracts").test_ro3_the_user_closes_a_receipt_through_its_route()


def test_gf32_ro4_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_recall_read_only_contracts").test_ro4_a_close_by_any_other_actor_is_refused_by_name_and_changes_nothing()


def test_gf33_os1_holds_on_turns_that_decide_nothing(monkeypatch, tmp_path):
    _quieted(monkeypatch, "test_onion_store_contracts").test_os1_a_state_round_trips_byte_for_byte_and_composes_the_same_block(tmp_path)


def test_gf34_os2_holds_on_turns_that_decide_nothing(monkeypatch, tmp_path):
    _quieted(monkeypatch, "test_onion_store_contracts").test_os2_a_moved_byte_is_refused_by_name_and_never_leaves_a_partial_state(tmp_path)


def test_gf35_os3_holds_on_turns_that_decide_nothing(monkeypatch, tmp_path):
    _quieted(monkeypatch, "test_onion_store_contracts").test_os3_plaintext_is_refused_by_name_unless_allowed_and_a_wrong_key_is_not_an_empty_state(tmp_path)


def test_gf36_os5_holds_on_turns_that_decide_nothing(monkeypatch, tmp_path):
    _quieted(monkeypatch, "test_onion_store_contracts").test_os5_the_root_is_a_pure_function_of_the_four_stores(tmp_path)


def test_gf37_pt1_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_peel_tree_contracts").test_pt1_a_leaf_points_at_its_cellar_span_and_verifies()


def test_gf38_pt2_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_peel_tree_contracts").test_pt2_a_parent_is_summarised_from_the_cellar_turns_never_from_child_text()


def test_gf39_pt3_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_peel_tree_contracts").test_pt3_a_pointer_that_does_not_resolve_is_refused_by_name()


def test_gf40_pt4_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_peel_tree_contracts").test_pt4_a_peel_that_diverged_from_its_spans_or_its_id_is_refused()


def test_gf41_pt5_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_peel_tree_contracts").test_pt5_selection_ranks_the_subject_first_never_stacks_and_fits_the_cap()


def test_gf42_pt6_holds_on_turns_that_decide_nothing(monkeypatch):
    _quieted(monkeypatch, "test_peel_tree_contracts").test_pt6_one_bad_peel_refuses_the_whole_tree()


# ---------------------------------------------------------------------------
# GF43-GF54 -- the bounds of a summary against its span
# ---------------------------------------------------------------------------
_LONG = [_turn("w1", "user", "typed", "Alice reviews the Harvest release on 2026-03-04 with a long checklist of many "
                                     "steps written by the whole team over several weeks.")]
# Both faces let it through: it answers the date and the name, and every
# word it adds is lower-case and decides nothing.
_NOVEL = "Alice reviews Harvest on 2026-03-04; gorgeous violet lanterns illuminate sprawling meadows."
_FAITHFUL = "Alice reviews the Harvest release on 2026-03-04 with a long checklist."
_OVER = _NOVEL + " Quiet harbors, painted kites, amber lamps, silver dunes, velvet skies, golden fields and copper bells " \
                 "shimmer softly."
_AUDIT = [_turn("a1", "user", "typed", "The audit is on 2026-03-04.")]


def _decide(probes, peels, span, text, gate):
    return peels.decide(span, probes.generate_probes(span, gate.lexicon), text, gate)


def test_gf43_a_summary_whose_words_its_span_mostly_does_not_hold_is_refused_with_its_novelty():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_LONG, gate.lexicon)
        first = peels.judge(drawn, _NOVEL, gate)
        assert first.accepted is True and first.result.failed == 0, "control: the first face lets it through"
        assert peels.faithfulness(_LONG, drawn, _NOVEL, gate) == (), "control: it says no claim its span lacks"
        refused = peels.decide(_LONG, drawn, _NOVEL, gate)
        assert refused.accepted is False
        assert refused.novelty == 6 / 9 and refused.max_novelty == 0.5
        assert f"novelty {round(6 / 9, 4)} over 0.5 (6 of 9 content words new: gorgeous, violet" in refused.reason
        assert refused.unsupported == (), "a bound is said in the reason, never as a claim"
        held = peels.decide(_LONG, drawn, _FAITHFUL, gate)
        assert held.accepted is True and held.novelty == 0.0, held.reason
    finally:
        restore()


def test_gf44_a_summary_longer_than_its_span_is_refused_with_its_length_ratio():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_SPAN, gate.lexicon)
        verbatim = _verbatim(_SPAN)
        at = peels.decide(_SPAN, drawn, verbatim, gate)
        assert at.accepted is True and at.length_ratio == 1.0 and at.max_length_ratio == 1.0, at.reason
        doubled = verbatim + " " + verbatim
        assert peels.judge(drawn, doubled, gate).accepted is True, "control: the first face lets it through"
        refused = peels.decide(_SPAN, drawn, doubled, gate)
        assert refused.accepted is False and refused.length_ratio == 2.0 and refused.novelty == 0.0
        assert "length ratio 2.0 over 1.0 (70 words for 35)" in refused.reason
        assert refused.unsupported == (), "a bound is said in the reason, never as a claim"
        over = peels.decide(_SPAN, drawn, verbatim + " Oslo.", gate)
        assert over.accepted is False and over.length_ratio == 36 / 35, "one word over its span is over"
    finally:
        restore()


def test_gf45_novelty_counts_no_function_word_figure_word_of_decision_reporter_or_code():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        held = probes.holdings(_AUDIT)
        told = ("We decided to keep it, and the assistant will choose 42. Nous avons d" + _E + "cid" + _E
                + " de garder 3 outils. " + probes.code_marker("velvet()") + "\n\n```\nvelvet\n```\n")
        assert probes.novel_words(held, told, gate.lexicon, gate.reporters) == ((), ())
        content, new = probes.novel_words(held, told + " Velvet.", gate.lexicon, gate.reporters)
        assert content == ("velvet",) and new == ("velvet",), "control: a word outside them is counted"
        decision = _decide(probes, peels, _AUDIT, "On 2026-03-04.", gate)
        assert decision.accepted is True and decision.novelty is None, "no content word: no novelty, nothing refused"
    finally:
        restore()


def test_gf46_a_word_is_held_by_its_key_its_final_s_off_then_its_first_five_letters():
    probes, _peels, _receipts, restore = _window()
    try:
        held = probes.holdings([_turn("k1", "user", "typed", "The host runs one service with a transfer queue.")])
        content, new = probes.novel_words(held, "The hosts run services with a transport queue by the hostel.")
        assert content == ("hosts", "run", "services", "transport", "queue", "hostel")
        assert new == ("hostel",), "transport has the key of transfer; hostel has a key of its own"
    finally:
        restore()


def test_gf47_the_length_ratio_is_every_word_of_the_summary_over_every_word_of_its_span_code_included():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        assert probes.word_count(_CODE[0]["text"]) == 5, "run, this, and the block: python, print, farm"
        verbatim = _decide(probes, peels, _CODE, _CODE[0]["text"], gate)
        assert verbatim.accepted is True and verbatim.length_ratio == 1.0, "the role is no word of the span"
        marked = _decide(probes, peels, _CODE, "Run this: " + probes.code_marker(_BODY), gate)
        assert marked.accepted is True and marked.length_ratio == 4 / 5, "a marker is two words, its block three"
        silent = _decide(probes, peels, [dict(_CODE[0], text="")], "Run.", gate)
        assert silent.length_ratio is None, "a span with no word has no ratio"
    finally:
        restore()


def test_gf48_the_bounds_are_read_from_onion_yaml_and_refused_by_name(tmp_path):
    _probes, peels, _receipts, restore = _window()
    try:
        raw = yaml.safe_load(_ONION.read_text(encoding="utf-8"))
        gate = peels.load_gate()
        assert gate.max_novelty == raw["gate"]["max_novelty"] == 0.5
        assert gate.max_length_ratio == raw["gate"]["max_length_ratio"] == 1.0
        assert gate.validate() == []
        assert _loaded(peels, tmp_path, _set("gate", "max_novelty", 0)).max_novelty == 0.0, "the strictest bound"
        assert _loaded(peels, tmp_path, _set("gate", "max_length_ratio", 3)).max_length_ratio == 3.0
        for name, bad in (
            ("max_novelty", (None, True, "high", float("nan"), float("inf"), 1.5, -0.1)),
            ("max_length_ratio", (None, True, "long", float("nan"), float("inf"), 0, -1.0)),
        ):
            with pytest.raises(peels.GateError, match=name):
                _loaded(peels, tmp_path, _drop("gate", name))
            for value in bad:
                with pytest.raises(peels.GateError, match=name):
                    _loaded(peels, tmp_path, _set("gate", name, value))
    finally:
        restore()


def test_gf49_a_gate_built_by_hand_holds_no_bound_and_its_decision_says_so():
    probes, peels, _receipts, restore = _window()
    try:
        shipped = peels.load_gate()
        hand = peels.Gate(decision_threshold=shipped.decision_threshold,
                          episodic_threshold=shipped.episodic_threshold, span_turns=1, lexicon=shipped.lexicon)
        assert hand.max_novelty is None and hand.max_length_ratio is None and hand.validate() == []
        decision = _decide(probes, peels, _LONG, _OVER, hand)
        assert decision.accepted is True, decision.reason
        assert decision.max_novelty is None and decision.max_length_ratio is None, "no bound, and it says so"
        assert decision.novelty == 22 / 25 and decision.length_ratio == 30 / 24, "measured all the same"
        refused = _decide(probes, peels, _LONG, _OVER, shipped)
        assert refused.accepted is False, "control: the shipped gate holds both"
        assert "novelty 0.88 over 0.5" in refused.reason and "length ratio 1.25 over 1.0" in refused.reason
    finally:
        restore()


def test_gf50_a_bound_out_of_its_range_is_refused_by_name_before_any_summary_is_judged():
    probes, peels, _receipts, restore = _window()
    try:
        shipped = peels.load_gate()
        for name, bad in (
            ("max_novelty", (True, "half", float("nan"), -0.1, 1.5)),
            ("max_length_ratio", (True, "long", float("nan"), float("inf"), 0, -1.0)),
        ):
            for value in bad:
                gate = dataclasses.replace(shipped, **{name: value})
                assert [e for e in gate.validate() if e.startswith(name + ":")], (name, value)
                with pytest.raises(peels.GateError, match=name):
                    _decide(probes, peels, _LONG, _FAITHFUL, gate)
    finally:
        restore()


def test_gf51_the_eviction_a_leaf_and_a_parent_are_held_to_the_bounds():
    _probes, peels, receipts, restore = _window()
    try:
        gate = peels.load_gate()
        doubled = lambda turns: _verbatim(turns) + " " + _verbatim(turns)  # noqa: E731
        s = _setup(peels, receipts, gate, _SPAN)
        outcome = peels.evict_gated(summarize=doubled, **s)
        assert outcome.evicted is False and outcome.receipt is None and outcome.peel is None
        assert len(s["flesh"].turns()) == len(_SPAN) + len(_TAIL), "the verbatim turns stay"
        assert outcome.decision.result.failed == 0, "control: every probe was answered"
        assert "length ratio 2.0 over 1.0" in outcome.reason
        s = _setup(peels, receipts, gate, _SPAN)
        evicted = peels.evict_gated(summarize=_verbatim, **s)
        assert evicted.evicted is True, evicted.reason
        leaf, decision = peels.build_leaf(evicted.receipt.key, s["cellar"], doubled, s["gate"], s["tree"])
        assert leaf is None and "length ratio 2.0 over 1.0" in decision.reason
        parent, decision = peels.build_parent([evicted.peel.id], s["cellar"], doubled, s["gate"], s["tree"])
        assert parent is None and "length ratio 2.0 over 1.0" in decision.reason
        assert len(s["tree"].all()) == 1, "nothing was added"
    finally:
        restore()


# Dave stands inside his sentence: a name at the head of its only sentence
# is not read (the same reader as a source's).
_ADRIFT = ("Violet lanterns, sprawling meadows and gorgeous dunes greeted Dave near quiet harbors with painted kites "
           "and amber lamps.")


def test_gf52_a_summary_refused_on_every_count_gives_every_reason_in_order_and_only_its_claims_as_claims():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_SPAN, gate.lexicon)
        first = peels.judge(drawn, _ADRIFT, gate)
        assert first.accepted is False, "control: the first face refuses it"
        decision = peels.decide(_SPAN, drawn, _ADRIFT, gate)
        assert decision.accepted is False and decision.unsupported == (("entity", "Dave", ""),)
        reason = decision.reason
        assert reason.startswith(first.reason)
        assert len(first.reason) < reason.index("entity:Dave") < reason.index("novelty 1.0 over 0.5")
    finally:
        restore()


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


def test_gf53_the_runbook_reports_each_bound_with_the_spread_of_its_figures():
    probes, peels, _receipts, restore = _window()
    try:
        runbook = _runbook()
        gate = peels.load_gate()
        entry = runbook._gate_entry(gate)
        assert entry["max_novelty"] == 0.5 and entry["max_length_ratio"] == 1.0
        verbatim = _verbatim(_SPAN)
        judged = [(span, probes.generate_probes(span, gate.lexicon), text) for span, text in (
            (_SPAN, verbatim), (_SPAN, verbatim + " " + verbatim), (_LONG, _NOVEL), (_AUDIT, "On 2026-03-04."),
        )]
        face = runbook._second_face(judged, gate)
        assert face["spans"] == 4 and face["spans_refused"] == 0 and face["claims_refused"] == {}
        assert face["novelty"] == {"measured": 3, "unmeasured": 1, "over": 1,
                                   "min": 0.0, "p50": 0.0, "p90": 0.6667, "max": 0.6667}
        assert face["length_ratio"] == {"measured": 4, "unmeasured": 0, "over": 1,
                                        "min": 0.5417, "p50": 0.5714, "p90": 2.0, "max": 2.0}
    finally:
        restore()


def _system_prompt():
    """The librarian's system prompt, read from its source without importing it."""
    tree = ast.parse((REPO / "opti_oignon" / "memory" / "librarian.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_SYSTEM_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("the librarian holds no system prompt")


def test_gf54_the_librarian_is_asked_for_the_language_of_the_turns_and_the_source_of_each_decision():
    prompt = _system_prompt()
    assert "Write the summary in the language of the turns." in prompt
    assert ("Attribute each decision to its source: it is the user's only if the user typed it; what the "
            "assistant, a document or a tool said is reported with them as the subject.") in prompt


# A summary of the GE16 fixture as the prompt asks for one: every probe
# answered, the user's decision told as the user's, the assistant's words
# told as the assistant's.
_TOLD = ("On 2026-03-04 Alice and Bob reviewed the Harvest release with a budget of 1200 euros. The assistant "
         "explained that the Oslo venue has no container runtime and that Carol handles the Oslo account with a 45 "
         "milliseconds latency target. The user agreed that the demo will not use Docker for the Oslo venue. The "
         "assistant said Bob reviews the release on 2026-05-02 and that the rollback keeps the old cluster warm for "
         "7 days.")


def test_gf55_a_faithful_summary_longer_than_a_short_span_is_refused_for_its_length_alone():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        drawn = probes.generate_probes(_GE_TYPED_SPAN, gate.lexicon)
        first = peels.judge(drawn, _TOLD, gate)
        assert first.accepted is True and first.result.failed == 0, "control: every probe is answered"
        assert peels.faithfulness(_GE_TYPED_SPAN, drawn, _TOLD, gate) == (), "control: no claim its span lacks"
        decision = peels.decide(_GE_TYPED_SPAN, drawn, _TOLD, gate)
        assert decision.novelty == 2 / 35, "explained and said, the words of the telling"
        assert decision.accepted is False and decision.length_ratio == 78 / 72
        assert decision.reason == f"length ratio {round(78 / 72, 4)} over 1.0 (78 words for 72)", "its length alone"
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF56 -- inline code is held only by the same inline code of its span
# ---------------------------------------------------------------------------
_BILLING = [
    _turn("b1", "user", "typed", "The billing service runs on port 8080, owned by Alice Martin."),
    _turn("b2", "user", "typed", "We will delete the staging backups on 2026-11-01."),
    _turn("b3", "user", "typed", "We will use `--no-cache` for the builds."),
]
_BILLING_TOLD = ("The billing service runs on port 8080, owned by Alice Martin. We will delete the staging backups on "
                 "2026-11-01. We will use `--no-cache` for the builds.")


def test_gf56_inline_code_in_a_summary_is_held_only_by_the_same_inline_code_of_its_span():
    probes, peels, _receipts, restore = _window()
    try:
        gate = dataclasses.replace(peels.load_gate(), max_length_ratio=None, max_novelty=None)
        drawn = probes.generate_probes(_BILLING, gate.lexicon)
        held = peels.decide(_BILLING, drawn, _BILLING_TOLD, gate)
        assert held.accepted is True and held.unsupported == (), "control: the span's own inline code is held"
        for told, inline in (
            (_BILLING_TOLD.replace("port 8080", "port 8080 and also on `9090`"), "9090"),
            (_BILLING_TOLD.replace("Alice Martin.", "Alice Martin with `Mallory Kane`."), "Mallory Kane"),
            (_BILLING_TOLD + " Then (`rm -rf /srv/billing`).", "rm -rf /srv/billing"),
            (_BILLING_TOLD.replace("We will delete", "We will `not` delete"), "not"),
        ):
            refused = peels.decide(_BILLING, drawn, told, gate)
            assert refused.accepted is False and ("code", inline, "") in refused.unsupported, (inline, refused.reason)
    finally:
        restore()


_PYTHON_NOT = "In Python, `not` inverts a boolean."
_STAGING = "We will delete the staging database on Friday."


def test_gf57_a_negation_in_backticks_inverts_as_a_plain_one_whatever_inline_code_the_span_holds():
    probes, peels, _receipts, restore = _window()
    try:
        gate = dataclasses.replace(peels.load_gate(), max_length_ratio=None, max_novelty=None)
        span = [_turn("n1", "user", "typed", _STAGING), _turn("n2", "assistant", "assistant", _PYTHON_NOT)]
        drawn = probes.generate_probes(span, gate.lexicon)
        told = {
            "as is": peels.decide(span, drawn, _STAGING + " " + _PYTHON_NOT, gate),
            "plain": peels.decide(span, drawn, _STAGING.replace("will", "will not") + " " + _PYTHON_NOT, gate),
            "coded": peels.decide(span, drawn, _STAGING.replace("will", "will `not`") + " " + _PYTHON_NOT, gate),
        }
        billing = peels.decide(_BILLING, probes.generate_probes(_BILLING, gate.lexicon), _BILLING_TOLD, gate)
    finally:
        restore()
    assert told["as is"].accepted is True, ("control: the span told as it is is held", told["as is"].reason)
    assert told["plain"].accepted is False, "control: a plain negation inverts the decision"
    coded = told["coded"]
    assert coded.accepted is False and any(claim[0] == "inversion" for claim in coded.unsupported), coded.unsupported
    assert billing.accepted is True, ("a faithful summary that copies `--no-cache` is still held", billing.reason)


def test_gf58_a_negation_inside_longer_inline_code_inverts_though_a_document_or_the_assistant_holds_it():
    probes, peels, _receipts, restore = _window()
    try:
        gate = dataclasses.replace(peels.load_gate(), max_length_ratio=None, max_novelty=None)
        verdicts = {}
        for code, holder in (("will not", "document"), ("won't", "assistant"), ("don't", "assistant"),
                             ("not in", "assistant")):
            planted = f"Write `{code}` in backticks when you quote it."
            span = [_turn("m1", "user", "typed", _STAGING),
                    _turn("m2", "assistant" if holder == "assistant" else "user", holder, planted)]
            drawn = probes.generate_probes(span, gate.lexicon)
            told = _STAGING.replace("will delete", f"`{code}` delete") + " " + planted
            verdicts[code] = (peels.decide(span, drawn, _STAGING + " " + planted, gate),
                              peels.decide(span, drawn, told, gate))
        billing = peels.decide(_BILLING, probes.generate_probes(_BILLING, gate.lexicon), _BILLING_TOLD, gate)
    finally:
        restore()
    assert all(held.accepted for held, _told in verdicts.values()), "control: each span told as it is is held"
    slipped = [code for code, (_held, told) in verdicts.items()
               if told.accepted or not (any(claim[0] == "inversion" for claim in told.unsupported)
                                        or "decision" in told.reason)]
    assert not slipped, f"a negation inside inline code inverts the decision it is written into: {slipped}"
    assert billing.accepted is True, ("a faithful copy of `--no-cache` is held", billing.reason)


def test_gf59_a_negation_fused_into_a_flag_reads_as_none_in_prose_as_in_code():
    probes, peels, _receipts, restore = _window()
    try:
        gate = dataclasses.replace(peels.load_gate(), max_length_ratio=None, max_novelty=None)
        said = "We will not keep Docker for Atlas, we build with `--no-cache`."
        span = [_turn("x1", "user", "typed", said)]
        drawn = probes.generate_probes(span, gate.lexicon)
        held = peels.decide(span, drawn, said.replace("`", ""), gate)
        inverted = peels.decide(span, drawn, said.replace("`", "").replace("will not", "will"), gate)
        control = peels.decide(span, drawn, said.replace("will not", "will"), gate)
    finally:
        restore()
    assert drawn and control.accepted is False, "control: the inversion with the flag in code is refused"
    assert held.accepted is True, ("the flag without its backticks reads as no negation", held.reason)
    assert inverted.accepted is False, "the inversion with the flag's backticks dropped is refused"


# ---------------------------------------------------------------------------
# GF60 -- GF47 on a code turn that tells rather than orders
# ---------------------------------------------------------------------------
_TOLD_CODE = [_turn("c1", "assistant", "assistant", "The code:\n\n```python\n" + _BODY + "\n```")]


def test_gf60_the_length_ratio_is_every_word_of_the_summary_over_every_word_of_its_span_code_included():
    probes, peels, _receipts, restore = _window()
    try:
        gate = peels.load_gate()
        assert probes.word_count(_TOLD_CODE[0]["text"]) == 5, "the, code, and the block: python, print, farm"
        verbatim = _decide(probes, peels, _TOLD_CODE, _TOLD_CODE[0]["text"], gate)
        assert verbatim.accepted is True and verbatim.length_ratio == 1.0, "the role is no word of the span"
        marked = _decide(probes, peels, _TOLD_CODE, "The code: " + probes.code_marker(_BODY), gate)
        assert marked.accepted is True and marked.length_ratio == 4 / 5, "a marker is two words, its block three"
        silent = _decide(probes, peels, [dict(_TOLD_CODE[0], text="")], "The code.", gate)
        assert silent.length_ratio is None, "a span with no word has no ratio"
    finally:
        restore()


# ---------------------------------------------------------------------------
# GF61 -- GF58 on a planted sentence that tells rather than orders
# ---------------------------------------------------------------------------
def test_gf61_a_negation_inside_longer_inline_code_inverts_though_a_document_or_the_assistant_holds_it():
    probes, peels, _receipts, restore = _window()
    try:
        gate = dataclasses.replace(peels.load_gate(), max_length_ratio=None, max_novelty=None)
        verdicts = {}
        for code, holder in (("will not", "document"), ("won't", "assistant"), ("don't", "assistant"),
                             ("not in", "assistant")):
            planted = f"The guide writes `{code}` in backticks when quoting it."
            span = [_turn("m1", "user", "typed", _STAGING),
                    _turn("m2", "assistant" if holder == "assistant" else "user", holder, planted)]
            drawn = probes.generate_probes(span, gate.lexicon)
            told = _STAGING.replace("will delete", f"`{code}` delete") + " " + planted
            verdicts[code] = (peels.decide(span, drawn, _STAGING + " " + planted, gate),
                              peels.decide(span, drawn, told, gate))
        billing = peels.decide(_BILLING, probes.generate_probes(_BILLING, gate.lexicon), _BILLING_TOLD, gate)
    finally:
        restore()
    assert all(held.accepted for held, _told in verdicts.values()), "control: each span told as it is is held"
    slipped = [code for code, (_held, told) in verdicts.items()
               if told.accepted or not (any(claim[0] == "inversion" for claim in told.unsupported)
                                        or "decision" in told.reason)]
    assert not slipped, f"a negation inside inline code inverts the decision it is written into: {slipped}"
    assert billing.accepted is True, ("a faithful copy of `--no-cache` is held", billing.reason)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
