#!/usr/bin/env python3
"""Contracts for decision probes: who may decide, and what a decision is.

A decision probe obliges a summary to keep a decision, with its polarity, or
the verbatim turns stay. That obligation is also a lever: whatever text can
make a decision probe can make the memory carry its words. Only the user's
typed words may hold it. The assistant's words, a document, a refined
question and a turn of unknown origin still yield their dates, numbers,
names and code, but never a decision.

A decision need not say that it is one. The ``decisions`` section of
``onion.yaml`` holds, language by language, the subjects and the acts a
decision is said with, class by class: keep, drop, choose, prefer, avoid,
switch. The lexicon built from it is refused by name when it is malformed,
travels with the gate, and is named, with the generator's version, on every
figure its probes are scored into.

  * DP1 -- a decision is drawn from a typed piece and from no other: the
    same sentence as the assistant's, a tool's, a document's, a refined
    question's or a legacy turn's draws its episodic probes and no decision,
    and in a turn that mixes typed text and a document only the typed
    segment decides.
  * DP2 -- whatever a native twin of the current version draws, a decision
    from a piece that is not typed is dropped.
  * DP3 -- a lexicon is built by rule: French verbs in -er and -ir and
    English verbs are inflected, a member of several words by its first
    word, a form listed as written stands as written, and each language
    keeps its own subjects.
  * DP4 -- a malformed lexicon is refused by name: a form in two classes, a
    member carrying a negation or a subject of its language, a capital or a
    letter outside ASCII, an empty class, an unknown language or key, a
    French member no rule inflects.
  * DP5 -- the gate reads its lexicon from ``onion.yaml``: the shipped
    section holds the six classes in both languages, and a configuration
    without the section, or with a malformed one, is refused by name.
  * DP6 -- a lexicon's fingerprint is canonical: the same lexicon written in
    another order gives the same twelve hexadecimal digits, and a change to
    a form, a class or a subject gives others.
  * DP7 -- every figure names the generator and the lexicon its probes were
    drawn with: a gate decision, a built peel, an eviction and a fidelity
    reading.
  * DP8 -- a gate never judges probes drawn with a lexicon other than its
    own.
  * DP9 -- without a marker, a typed sentence decides when a subject of an
    act's language stands before the act; without a lexicon, only a marker
    decides.
  * DP10 -- a subject of another language makes no decision: "Click on
    Select" does not decide, "We select Podman" does.
  * DP11 -- a subject decides within the reach its language declares, and
    not one word beyond it.
  * DP12 -- an act that opens a sentence or an item is an instruction and
    decides; an act further in, with no subject before it, does not.
  * DP13 -- a decision's key holds its acts by class and its dates and
    numbers by their reading, never by their words.
  * DP14 -- a sentence that covers a decision's words but says another act
    does not answer it.
  * DP15 -- a date or a number of a decision answers in any writing that
    reads to it; moved or lost, it fails the decision.
  * DP16 to DP21 -- for each class, keep, drop, choose, prefer, avoid and
    switch: another word of the class, or the other language, answers; the
    decision negated, or said with its opposite act, does not.
  * DP22 -- a native twin is handed the lexicon it draws with and the one
    it scores with; probes drawn with two lexicons are scored by the
    reference.
  * DP23 -- a lexicon's reach is a whole number of words, one or more, per
    language, refused by name otherwise and named in its fingerprint.
  * DP24 -- a decision's marker words stay in its key: reworded into a
    weaker status or modality -- discussed, proposed, envisaged, may, peut
    -- it does not answer. Taking them out was measured, and let these
    through.
  * DP25 -- what the generator decides with a lexicon is pinned with its
    version: a change to what it draws raises the version.
  * DP26 -- a French decision typed without its accents decides as typed
    with them: its marker is matched folded, the same acts, dates and
    polarity are asked for, and the accented summary answers it.
  * DP27 -- an accented marker decides, folded, wherever it opens a word,
    with a subject or none, its accents typed or not and in either Unicode
    form; inside a word ("helicopter", "undecidable") it decides nothing. An
    English word that opens with one is read as a decision: the safe side,
    where the span stays verbatim.
  * DP28 -- a decision decides however its text is written: an act whose
    accents are decomposed, a marker whose last word is elided ("prevu
    d'utiliser"), a marker of two words split by a line break or a no-break
    space.

Local-only (the public distribution ships no tests).
"""

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_DECISION = "We decided to ship Harvest to Oslo on 2026-10-09 with 3 reviewers."
_DOCUMENT = "We decided to wire the funds to Contoso on 2026-10-12."


def _probes():
    loaded, restore = isolate(
        targets={"opti_oignon.memory.probes": source("memory", "probes.py")},
        packages=("opti_oignon.memory",),
    )
    mod = loaded["opti_oignon.memory.probes"]
    mod._native = lambda: None
    return mod, restore


def _turn(role, origin, text, segments=None):
    turn = {"turn_id": "t1", "role": role, "text": text}
    if origin is not None:
        turn["origin"] = origin
    if segments is not None:
        turn["segments"] = segments
    return turn


def _kinds(drawn):
    return sorted({p.kind for p in drawn})


# ---------------------------------------------------------------------------
# DP1 -- only typed text decides
# ---------------------------------------------------------------------------
def test_dp1_a_decision_is_drawn_from_typed_text_and_from_nothing_else():
    mod, restore = _probes()
    try:
        typed = mod.generate_probes([_turn("user", "typed", _DECISION)])
        decisions = [p for p in typed if p.kind == "decision"]
        assert [(p.answer, p.origin, p.role) for p in decisions] == [(_DECISION, "typed", "user")], typed
        episodic = sorted((p.kind, p.answer) for p in typed if p.kind != "decision")
        for role, origin in (("assistant", "assistant"), ("assistant", "assistant+tool"), ("assistant", "assistant+web"),
                             ("user", "document"), ("user", "refined"), ("user", "legacy"), ("user", None),
                             ("assistant", None)):
            drawn = mod.generate_probes([_turn(role, origin, _DECISION)])
            assert "decision" not in _kinds(drawn), f"{role} {origin} draws no decision"
            assert sorted((p.kind, p.answer) for p in drawn) == episodic, f"{role} {origin} keeps its episodic probes"
        mixed = _DECISION + " " + _DOCUMENT
        cut = len(_DECISION) + 1
        drawn = mod.generate_probes([_turn("user", "typed", mixed, [[0, cut, "typed"], [cut, len(mixed), "document"]])])
        assert [(p.answer, p.origin) for p in drawn if p.kind == "decision"] == [(_DECISION, "typed")], drawn
        assert "Contoso" in [p.answer for p in drawn if p.kind == "entity"], "the document's names are still asked"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP2 -- a twin cannot hand a decision to a piece that is not typed
# ---------------------------------------------------------------------------
class _Twin:
    """A native core of the current version that draws one decision from every piece it is handed."""

    def __init__(self, version):
        self.probe_generator_version = version

    def probe_generate(self, texts, *_tables):
        return [(index, "decision", text, ["decided", "ship"], 0, False) for index, text in enumerate(texts)]


def test_dp2_a_decision_a_twin_draws_from_a_piece_that_is_not_typed_is_dropped():
    mod, restore = _probes()
    try:
        twin = _Twin(mod.GENERATOR_VERSION)
        mod._native = lambda: twin
        mixed = _DECISION + " " + _DOCUMENT
        cut = len(_DECISION) + 1
        span = [
            _turn("user", "typed", mixed, [[0, cut, "typed"], [cut, len(mixed), "document"]]),
            _turn("assistant", "assistant", _DECISION),
            _turn("user", None, _DECISION),
        ]
        drawn = mod.generate_probes(span)
        assert [(p.answer, p.origin) for p in drawn] == [(_DECISION + " ", "typed")], drawn
    finally:
        restore()


# ---------------------------------------------------------------------------
# The lexicon: the words a decision is said with
# ---------------------------------------------------------------------------
_E = chr(0xE9)
_DELETE = object()

# A lexicon that exercises every rule: French verbs in -er, one with a stem
# in -g, and in -ir; English verbs that take -es, lose a silent e, end in a
# consonant and y, or end on one vowel and one consonant, in one syllable or
# in two; members of two words; forms listed as written; and the French
# subject "on" beside the English "settle on", where it is no subject.
_SECTION = {
    "fr": {
        "subjects": ["on", "nous"],
        "reach": 3,
        "acts": {"keep": ["garder", "rester sur", "figer"], "drop": ["renoncer"], "choose": ["choisir"]},
        "as_written": {"choose": ["retiens", "partir sur"]},
    },
    "en": {
        "subjects": ["we", "let"],
        "reach": 3,
        "acts": {
            "keep": ["keep", "stick with"], "drop": ["drop", "ditch"], "choose": ["settle on", "rely on", "go with"],
            "prefer": ["prefer"], "switch": ["replace"],
        },
        "as_written": {"keep": ["kept"], "prefer": ["d rather"]},
    },
}

_GATE_SPAN = [
    {"turn_id": "g1", "role": "user", "origin": "typed",
     "text": "We decided to ship Harvest to Oslo on 2026-10-09 with 3 reviewers."},
    {"turn_id": "g2", "role": "assistant", "origin": "assistant", "text": "Carol reviews the Oslo release on 2026-10-12."},
]


def _forms(lexicon):
    return {(language, " ".join(words), cls) for language, words, cls in lexicon.forms}


def _altered(*changes):
    """A copy of the fixture section with each ``(path, value)`` set, or deleted for ``_DELETE``."""
    section = copy.deepcopy(_SECTION)
    for path, value in changes:
        *parents, last = path
        node = section
        for key in parents:
            node = node[key]
        if value is _DELETE:
            del node[last]
        else:
            node[last] = value
    return section


def _reordered(section):
    """The same section with every mapping and list reversed and every list's first entry repeated."""
    out = {}
    for language in reversed(list(section)):
        entry = {}
        for key in reversed(list(section[language])):
            value = section[language][key]
            if isinstance(value, list):
                entry[key] = list(reversed(value)) + value[:1]
            elif isinstance(value, dict):
                entry[key] = {cls: list(reversed(members)) + members[:1] for cls, members in reversed(list(value.items()))}
            else:
                entry[key] = value
        out[language] = entry
    return out


def _gate_window():
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


def _verbatim(span):
    return " ".join(t["text"] for t in span)


# ---------------------------------------------------------------------------
# DP3 -- a lexicon is built by rule
# ---------------------------------------------------------------------------
def test_dp3_a_lexicon_inflects_its_lemmas_by_rule_and_keeps_its_written_forms():
    mod, restore = _probes()
    try:
        lexicon = mod.build_lexicon(_SECTION)
        forms = _forms(lexicon)
        expected = {
            ("fr", "keep"): ("garder", "garde", "gardes", "gardons", "gardez", "gardent", "gardee", "gardees", "gardant",
                             "gardait", "gardions", "gardaient", "gardera", "garderons", "garderait", "garderiez",
                             "reste sur", "restons sur", "restera sur", "fige", "figeons", "figeant", "figeait", "figera"),
            ("fr", "drop"): ("renoncer", "renonce", "renoncons", "renoncait", "renoncera"),
            ("fr", "choose"): ("choisir", "choisis", "choisit", "choisissons", "choisissez", "choisissent", "choisi",
                               "choisie", "choisissant", "choisissait", "choisira", "choisirait", "choisisse",
                               "retiens", "partir sur"),
            ("en", "keep"): ("keep", "keeps", "keeping", "stick with", "sticks with", "sticking with", "kept"),
            ("en", "drop"): ("drop", "drops", "dropping", "dropped", "ditch", "ditches", "ditching", "ditched"),
            ("en", "choose"): ("settle on", "settles on", "settling on", "settled on", "rely on", "relies on",
                               "relying on", "relied on", "go with", "goes with", "going with"),
            ("en", "prefer"): ("prefer", "prefers", "preferring", "prefering", "preferred", "prefered", "d rather"),
            ("en", "switch"): ("replace", "replaces", "replacing", "replaced"),
        }
        for (language, cls), listed in expected.items():
            missing = [form for form in listed if (language, form, cls) not in forms]
            assert not missing, f"{language} {cls} lacks {missing}"
        written = {form for _language, form, _cls in forms}
        never = ("figons", "figant", "figait", "partis sur", "partissons sur", "kepts", "kepted", "ds rather",
                 "keepping", "keepped", "droping", "droped", "replaceing", "replaceed", "relys on", "relyed on",
                 "ditchs", "settle ons", "stick withs")
        assert not [form for form in never if form in written], "a form no rule makes"
        assert {cls for _language, _form, cls in forms} == {"keep", "drop", "choose", "prefer", "switch"}
        assert lexicon.subjects == frozenset({("fr", "on"), ("fr", "nous"), ("en", "we"), ("en", "let")})
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP4 -- a malformed lexicon is refused by name
# ---------------------------------------------------------------------------
def test_dp4_a_malformed_lexicon_is_refused_by_name():
    mod, restore = _probes()
    try:
        cases = (
            (_altered((("en", "acts", "drop"), ["drop", "keeping"])), "'keeping'", "two classes"),
            (_altered((("en", "acts", "avoid"), ["garde"])), "'garde'", "two classes"),
            (_altered((("en", "acts", "drop"), ["not keep"])), "'not keep'", "negation"),
            (_altered((("fr", "acts", "drop"), ["ne plus garder"])), "'ne plus garder'", "negation"),
            (_altered((("fr", "acts", "keep"), ["on garder"])), "'on garder'", "subject"),
            (_altered((("en", "as_written", "keep"), ["we kept"])), "'we kept'", "subject"),
            (_altered((("en", "acts", "keep"), ["Keep"])), "'Keep'", "lower-case ASCII"),
            (_altered((("fr", "acts", "prefer"), ["pr" + _E + "f" + _E + "rer"])), "'pr", "lower-case ASCII"),
            (_altered((("en", "acts", "keep"), ["stick  with"])), "'stick  with'", "lower-case ASCII"),
            (_altered((("en", "acts", "avoid"), [])), "acts.avoid", "no member"),
            (_altered((("en", "as_written", "prefer"), [])), "as_written.prefer", "no member"),
            (_altered((("en", "acts", "keep"), "keep")), "acts.keep", "list"),
            (_altered((("de",), {"subjects": ["wir"], "acts": {"keep": ["behalten"]}})), "'de'", "language"),
            (_altered((("fr", "acts", "keep"), ["faire"])), "'faire'", "as_written"),
            (_altered((("en", "subjects"), ["We"])), "'We'", "subject"),
            (_altered((("en", "subjects"), _DELETE)), "decisions.en", "subjects"),
            (_altered((("en", "acts"), _DELETE)), "decisions.en", "acts"),
            (_altered((("fr", "verbs"), {"keep": ["garder"]})), "'verbs'", "key"),
            (_altered((("en", "acts", "Keep"), ["retain"])), "'Keep'", "class"),
            ([], "decisions", "mapping"),
            ({}, "decisions", "language"),
        )
        for section, culprit, reason in cases:
            with pytest.raises(mod.LexiconError) as caught:
                mod.build_lexicon(section)
            message = str(caught.value)
            assert culprit in message and reason in message, (culprit, reason, message)
        assert mod.build_lexicon(_SECTION).forms, "control: the fixture itself builds"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP5 -- the gate reads its lexicon from onion.yaml
# ---------------------------------------------------------------------------
def test_dp5_the_gate_reads_its_lexicon_from_onion_yaml_and_refuses_a_missing_or_malformed_one(tmp_path):
    _probes_mod, peels, _receipts, restore = _gate_window()
    try:
        gate = peels.load_gate()
        forms = _forms(gate.lexicon)
        classes = {"keep", "drop", "choose", "prefer", "avoid", "switch"}
        for language in ("fr", "en"):
            assert {cls for lang, _form, cls in forms if lang == language} == classes, language
        shipped = (("fr", "gardons", "keep"), ("fr", "restons sur", "keep"), ("fr", "laisse tomber", "drop"),
                   ("fr", "renoncons", "drop"), ("fr", "retiens", "choose"), ("fr", "retenons", "choose"),
                   ("fr", "partons sur", "choose"), ("fr", "optons", "choose"), ("fr", "prefere", "prefer"),
                   ("fr", "evitons", "avoid"), ("fr", "basculons", "switch"), ("fr", "remplace", "switch"),
                   ("en", "kept", "keep"), ("en", "stuck with", "keep"), ("en", "gave up", "drop"),
                   ("en", "given up", "drop"), ("en", "ditched", "drop"), ("en", "went with", "choose"),
                   ("en", "settled on", "choose"), ("en", "chosen", "choose"), ("en", "would rather", "prefer"),
                   ("en", "d rather", "prefer"), ("en", "avoiding", "avoid"), ("en", "switched", "switch"),
                   ("en", "migrating", "switch"))
        missing = [entry for entry in shipped if entry not in forms]
        assert not missing, missing
        written = {form for _language, form, _cls in forms}
        common = {"use", "uses", "utiliser", "utilise", "prendre", "prend", "take", "passer a", "passe a", "move to"}
        assert not written & common, "verbs too common outside a decision stay out"
        assert gate.lexicon.subjects == frozenset({("fr", "on"), ("fr", "je"), ("fr", "j"), ("fr", "nous"),
                                                   ("en", "we"), ("en", "i"), ("en", "let")})
        raw = yaml.safe_load(Path(peels._CONFIG).read_text(encoding="utf-8"))
        del raw["decisions"]
        without = tmp_path / "without.yaml"
        without.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(peels.GateError) as caught:
            peels.load_gate(without)
        assert "decisions" in str(caught.value) and "missing" in str(caught.value), str(caught.value)
        raw["decisions"] = {"en": {"subjects": ["we"], "acts": {"keep": ["Keep"]}}}
        malformed = tmp_path / "malformed.yaml"
        malformed.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(peels.GateError) as caught:
            peels.load_gate(malformed)
        assert "'Keep'" in str(caught.value), "the lexicon's refusal reaches the gate by name"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP6 -- a lexicon's fingerprint is canonical
# ---------------------------------------------------------------------------
def test_dp6_a_lexicon_fingerprint_is_canonical():
    mod, restore = _probes()
    try:
        first = mod.build_lexicon(_SECTION)
        assert re.fullmatch(r"[0-9a-f]{12}", first.fingerprint), first.fingerprint
        again = mod.build_lexicon(_reordered(_SECTION))
        assert again.fingerprint == first.fingerprint and again == first, "order and repeats name nothing"
        changed = (
            _altered((("en", "acts", "keep"), ["keep", "stick with", "retain"])),
            _altered((("en", "acts", "drop"), ["drop"]), (("en", "acts", "prefer"), ["prefer", "ditch"])),
            _altered((("fr", "subjects"), ["on"]), (("en", "subjects"), ["we", "let", "nous"])),
            _altered((("en", "as_written", "keep"), _DELETE), (("en", "acts", "keep"), ["keep", "stick with", "kept"])),
            _altered((("en", "acts", "switch"), _DELETE), (("en", "acts", "swap"), ["replace"])),
        )
        prints = [mod.build_lexicon(section).fingerprint for section in changed]
        assert first.fingerprint not in prints and len(set(prints)) == len(prints), prints
        empty = mod.EMPTY_LEXICON
        assert empty.forms == () and empty.subjects == frozenset()
        assert re.fullmatch(r"[0-9a-f]{12}", empty.fingerprint) and empty.fingerprint != first.fingerprint
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP7 -- every figure names the generator and the lexicon
# ---------------------------------------------------------------------------
def test_dp7_every_figure_names_the_generator_and_the_lexicon_its_probes_were_drawn_with():
    probes, peels, receipts, restore = _gate_window()
    try:
        shipped = peels.load_gate()
        bare = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        version = probes.GENERATOR_VERSION
        assert shipped.lexicon.fingerprint != probes.EMPTY_LEXICON.fingerprint, "control: two lexicons to tell apart"
        for gate, fingerprint in ((shipped, shipped.lexicon.fingerprint), (bare, probes.EMPTY_LEXICON.fingerprint)):
            drawn = probes.generate_probes(_GATE_SPAN, gate.lexicon)
            assert drawn and {p.lexicon.fingerprint for p in drawn} == {fingerprint}
            judged = peels.judge(drawn, _verbatim(_GATE_SPAN), gate)
            assert judged.accepted and (judged.generator, judged.lexicon) == (version, fingerprint)
            nothing = peels.judge([], "anything", gate)
            assert not nothing.accepted and (nothing.generator, nothing.lexicon) == (version, fingerprint)
            cellar, tree = receipts.Cellar(), peels.PeelTree()
            _peel, built = peels.build_leaf(cellar.store([dict(t) for t in _GATE_SPAN]), cellar, _verbatim, gate, tree)
            assert built.accepted and (built.generator, built.lexicon) == (version, fingerprint)
            flesh = receipts.Flesh([dict(t) for t in _GATE_SPAN])
            outcome = peels.evict_gated(flesh=flesh, cellar=receipts.Cellar(), ledger=receipts.ReceiptLedger(),
                                        tree=peels.PeelTree(), gate=gate, summarize=_verbatim)
            assert outcome.evicted and (outcome.decision.generator, outcome.decision.lexicon) == (version, fingerprint)
            reading = peels.fidelity(tree, cellar, gate.lexicon)
            assert reading["rate"] == 1.0 and (reading["generator"], reading["lexicon"]) == (version, fingerprint)
        assert peels.fidelity(tree, cellar)["lexicon"] == probes.EMPTY_LEXICON.fingerprint, "no lexicon is the empty one"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP8 -- a gate judges only probes drawn with its own lexicon
# ---------------------------------------------------------------------------
def test_dp8_a_gate_never_judges_probes_drawn_with_another_lexicon():
    probes, peels, _receipts, restore = _gate_window()
    try:
        shipped = peels.load_gate()
        bare = peels.Gate(decision_threshold=0.9, episodic_threshold=0.7, span_turns=2)
        text = _verbatim(_GATE_SPAN)
        assert peels.judge(probes.generate_probes(_GATE_SPAN, shipped.lexicon), text, shipped).accepted, "control"
        for drawn_with, gate in ((None, shipped), (shipped.lexicon, bare)):
            drawn = probes.generate_probes(_GATE_SPAN, drawn_with)
            with pytest.raises(peels.GateError) as caught:
                peels.judge(drawn, text, gate)
            message = str(caught.value)
            assert shipped.lexicon.fingerprint in message and probes.EMPTY_LEXICON.fingerprint in message, message
        by_hand = [probes.Probe("entity", "q", "Harvest", "g1")]
        assert peels.judge(by_hand, "Harvest", bare).accepted, "a probe made without a lexicon is the empty one's"
        with pytest.raises(peels.GateError):
            peels.judge(by_hand, "Harvest", shipped)
    finally:
        restore()


# ---------------------------------------------------------------------------
# A decision without a marker: the shipped lexicon at work
# ---------------------------------------------------------------------------
_EG = chr(0xE8)
_ONION = Path(__file__).resolve().parents[1] / "opti_oignon" / "config" / "onion.yaml"


def _shipped(mod):
    """The lexicon the shipped ``onion.yaml`` holds, built by the module under test."""
    return mod.build_lexicon(yaml.safe_load(_ONION.read_text(encoding="utf-8"))["decisions"])


def _decisions(mod, text, lexicon):
    """The decision probes one typed turn of ``text`` draws with ``lexicon``."""
    return [p for p in mod.generate_probes([_turn("user", "typed", text)], lexicon) if p.kind == "decision"]


def _answers(mod, probe, text):
    return mod.score([probe], text).passed == 1


# ---------------------------------------------------------------------------
# DP9 -- a subject of its language before an act decides without a marker
# ---------------------------------------------------------------------------
_UNMARKED = (
    "On garde Docker.",
    "je pr" + _E + "f" + _EG + "re " + _E + "viter Kubernetes",
    "Nous migrons vers Podman.",
    "On part sur Nix.",
    "J'abandonne Helm.",
    "Let's go with Podman.",
    "I'd rather drop Helm.",
    "We ditch Jenkins.",
)


def test_dp9_a_subject_of_its_language_before_an_act_decides_without_a_marker():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        for text in _UNMARKED:
            assert not _decisions(mod, text, mod.EMPTY_LEXICON), f"control: {text!r} carries no marker"
            assert [p.answer for p in _decisions(mod, text, lexicon)] == [text], text
        assert [p.answer for p in _decisions(mod, _DECISION, mod.EMPTY_LEXICON)] == [_DECISION], "a marker decides"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP10 -- a subject decides only for an act of its own language
# ---------------------------------------------------------------------------
def test_dp10_a_subject_of_another_language_makes_no_decision():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        pairs = (
            ("Click on Select.", "We select Podman."),
            ("Focus on dropping the cache.", "We are dropping the cache."),
            ("Je drop la table.", "Je laisse tomber la table."),
        )
        for foreign, own in pairs:
            assert [p.answer for p in _decisions(mod, own, lexicon)] == [own], f"control: {own!r} decides"
            assert not _decisions(mod, foreign, lexicon), f"{foreign!r}: its subject is not of its act's language"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP11 -- a subject decides within its language's reach
# ---------------------------------------------------------------------------
def test_dp11_a_subject_decides_within_the_reach_of_its_language_and_not_beyond():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        assert dict(lexicon.reach) == {"fr": 3, "en": 3}, lexicon.reach
        cases = (
            ("We really truly prefer Podman.", "We really truly always prefer Podman."),
            ("On a vraiment gard" + _E + " Docker.", "On a vraiment toujours gard" + _E + " Docker."),
        )
        for within, beyond in cases:
            assert [p.answer for p in _decisions(mod, within, lexicon)] == [within], within
            assert not _decisions(mod, beyond, lexicon), beyond
        section = yaml.safe_load(_ONION.read_text(encoding="utf-8"))["decisions"]
        section["fr"]["reach"] = 4
        wider = mod.build_lexicon(section)
        assert [p.answer for p in _decisions(mod, cases[1][1], wider)] == [cases[1][1]], "the reach is read per language"
        assert not _decisions(mod, cases[0][1], wider), "the English reach is its own"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP12 -- an act at the head of a sentence or an item is an instruction
# ---------------------------------------------------------------------------
def test_dp12_an_act_that_opens_a_sentence_or_an_item_decides_and_one_further_in_without_a_subject_does_not():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        heads = (
            ("Keep Docker.", "Keep Docker."),
            ("Garde Docker.", "Garde Docker."),
            (_E.upper() + "vitons Kubernetes.", _E.upper() + "vitons Kubernetes."),
            ("- Switch the runners to Podman", "Switch the runners to Podman"),
        )
        for text, answer in heads:
            assert [p.answer for p in _decisions(mod, text, lexicon)] == [answer], text
            assert not _decisions(mod, text, mod.EMPTY_LEXICON), f"control: {text!r} carries no marker"
        for text in ("Teams keep Docker images small.", "Les " + _E + "quipes gardent Docker."):
            assert not _decisions(mod, text, lexicon), text
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP13 -- a decision key is canonical
# ---------------------------------------------------------------------------
def test_dp13_a_decision_key_holds_its_acts_dates_and_numbers_by_their_reading_not_their_words():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        [probe] = _decisions(mod, "We keep Docker until 2026-10-05 on 3 hosts.", lexicon)
        assert probe.key == frozenset({"docker", "until", "hosts", "act:keep", "date:2026-10-05", "number:3"}), probe.key
        [probe] = _decisions(mod, "On garde `docker-compose` et on abandonne Swarm le 5 octobre 2026.", lexicon)
        assert probe.key == frozenset({"docker", "compose", "swarm", "act:keep", "act:drop", "date:2026-10-05"}), probe.key
        [probe] = _decisions(mod, "Let's go with Podman.", lexicon)
        assert probe.key == frozenset({"podman", "act:choose"}), "a subject is no content word: " + str(sorted(probe.key))
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP14 -- the acts of a decision are required
# ---------------------------------------------------------------------------
_LONG = "We keep Docker for the build farm, the staging cluster, the nightly runners and the release pipeline."


def test_dp14_a_decision_said_with_another_act_is_refused_even_where_its_words_are_covered():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        [probe] = _decisions(mod, _LONG, lexicon)
        assert "act:keep" in probe.key and (len(probe.key) - 1) / len(probe.key) >= mod.DECISION_COVERAGE, probe.key
        assert _answers(mod, probe, _LONG), "control: the decision answers for itself"
        assert _answers(mod, probe, _LONG.replace("We keep", "We retain")), "another member of its class answers"
        for act in ("drop", "switch", "avoid"):
            swapped = _LONG.replace("keep", act)
            assert not _answers(mod, probe, swapped), swapped
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP15 -- the dates and numbers of a decision are required, in any writing
# ---------------------------------------------------------------------------
_DATED = "We keep Docker on the build farm until 2026-10-05 with 3 hosts and 2 spares."


def test_dp15_a_decision_keeps_its_dates_and_numbers_in_any_writing_and_fails_them_moved_or_lost():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        [probe] = _decisions(mod, _DATED, lexicon)
        assert {"date:2026-10-05", "number:3", "number:2"} <= probe.key, probe.key
        assert (len(probe.key) - 2) / len(probe.key) >= mod.DECISION_COVERAGE, "control: words alone would cover it"
        assert _answers(mod, probe, _DATED.replace("2026-10-05", "5 October 2026")), "a date written otherwise answers"
        for altered in (
            _DATED.replace("2026-10-05", "2026-10-06"),
            _DATED.replace(" until 2026-10-05", ""),
            _DATED.replace("3 hosts", "4 hosts"),
            _DATED.replace("3 hosts", "hosts"),
        ):
            assert not _answers(mod, probe, altered), altered
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP16 to DP21 -- each class answers in its words and languages, never inverted
# ---------------------------------------------------------------------------
def _holds_its_class(cls, said, other_word, translated, negated, inverted):
    """A decision of ``cls`` answers to another word of its class and to the other language, never inverted."""
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        [probe] = _decisions(mod, said, lexicon)
        act = "act:" + cls
        assert act in probe.key and (len(probe.key) - 1) / len(probe.key) >= mod.DECISION_COVERAGE, probe.key
        assert _answers(mod, probe, said), "control: the decision answers for itself"
        assert _answers(mod, probe, other_word), f"another word of {cls} answers: {other_word!r}"
        assert _answers(mod, probe, translated), f"{cls} answers in the other language: {translated!r}"
        assert not _answers(mod, probe, negated), f"negated, the decision is inverted: {negated!r}"
        assert not _answers(mod, probe, inverted), f"said with its opposite act, the decision is inverted: {inverted!r}"
    finally:
        restore()


def test_dp16_a_keep_decision_answers_in_its_words_and_languages_and_never_inverted():
    _holds_its_class(
        "keep",
        "We keep Docker for Harvest, Oslo and Bergen.",
        "We stick with Docker for Harvest, Oslo and Bergen.",
        "On conserve Docker pour Harvest, Oslo et Bergen.",
        "We never keep Docker for Harvest, Oslo and Bergen.",
        "We drop Docker for Harvest, Oslo and Bergen.",
    )


def test_dp17_a_drop_decision_answers_in_its_words_and_languages_and_never_inverted():
    _holds_its_class(
        "drop",
        "We drop Helm for Harvest, Oslo and Bergen.",
        "We give up Helm for Harvest, Oslo and Bergen.",
        "On laisse tomber Helm pour Harvest, Oslo et Bergen.",
        "We do not drop Helm for Harvest, Oslo and Bergen.",
        "We keep Helm for Harvest, Oslo and Bergen.",
    )


def test_dp18_a_choose_decision_answers_in_its_words_and_languages_and_never_inverted():
    _holds_its_class(
        "choose",
        "We choose Podman for Harvest, Oslo and Bergen.",
        "We settle on Podman for Harvest, Oslo and Bergen.",
        "Nous choisissons Podman pour Harvest, Oslo et Bergen.",
        "We never choose Podman for Harvest, Oslo and Bergen.",
        "We avoid Podman for Harvest, Oslo and Bergen.",
    )


def test_dp19_a_prefer_decision_answers_in_its_words_and_languages_and_never_inverted():
    _holds_its_class(
        "prefer",
        "I prefer Nix for Harvest, Oslo and Bergen.",
        "I would rather run Nix for Harvest, Oslo and Bergen.",
        "Je pr" + _E + "f" + _EG + "re Nix pour Harvest, Oslo et Bergen.",
        "I do not prefer Nix for Harvest, Oslo and Bergen.",
        "I avoid Nix for Harvest, Oslo and Bergen.",
    )


def test_dp20_an_avoid_decision_answers_in_its_words_and_languages_and_never_inverted():
    _holds_its_class(
        "avoid",
        "We avoid Kubernetes for Harvest, Oslo and Bergen.",
        "We are avoiding Kubernetes for Harvest, Oslo and Bergen.",
        "On " + _E + "vite Kubernetes pour Harvest, Oslo et Bergen.",
        "We never avoid Kubernetes for Harvest, Oslo and Bergen.",
        "We choose Kubernetes for Harvest, Oslo and Bergen.",
    )


def test_dp21_a_switch_decision_answers_in_its_words_and_languages_and_never_inverted():
    _holds_its_class(
        "switch",
        "We replace Docker by Podman on Harvest, Oslo and Bergen.",
        "We migrate Docker to Podman on Harvest, Oslo and Bergen.",
        "On remplace Docker par Podman sur Harvest, Oslo et Bergen.",
        "We do not replace Docker by Podman on Harvest, Oslo and Bergen.",
        "We keep Docker over Podman on Harvest, Oslo and Bergen.",
    )


# ---------------------------------------------------------------------------
# DP22 -- a native twin is handed the lexicon
# ---------------------------------------------------------------------------
class _LexiconTwin:
    """A native core of the current version that records the lexicon it is handed, and draws and fails nothing."""

    def __init__(self, version):
        self.probe_generator_version = version
        self.handed = []

    def probe_generate(self, texts, *rest):
        self.handed.append(("generate", rest[-1]))
        return []

    def probe_score(self, rows, text, *rest):
        self.handed.append(("score", rest[-1]))
        return []


def _table(lexicon):
    return (tuple(sorted(lexicon.subjects)), lexicon.forms, lexicon.reach)


def test_dp22_a_native_twin_is_handed_the_lexicon_it_draws_and_scores_with():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        span = [_turn("user", "typed", "On garde Docker.")]
        drawn = mod.generate_probes(span, lexicon)
        bare = mod.generate_probes([_turn("user", "typed", _DECISION)])
        mixed = drawn + bare
        assert "decision" in _kinds(drawn) and "decision" in _kinds(bare), "control: both lexicons decide"
        reference = mod.score(mixed, "On garde Docker.")
        twin = _LexiconTwin(mod.GENERATOR_VERSION)
        mod._native = lambda: twin
        assert mod.generate_probes(span, lexicon) == [] and twin.handed == [("generate", _table(lexicon))]
        assert mod.generate_probes(span) == [] and twin.handed[-1] == ("generate", _table(mod.EMPTY_LEXICON))
        twin.handed.clear()
        assert mod.score(drawn, "anything").passed == len(drawn) and twin.handed == [("score", _table(lexicon))]
        twin.handed.clear()
        result = mod.score(mixed, "On garde Docker.")
        assert twin.handed == [], "probes drawn with two lexicons are scored by the reference"
        assert (result.passed, result.failed) == (reference.passed, reference.failed)
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP23 -- a lexicon's reach
# ---------------------------------------------------------------------------
def test_dp23_a_lexicon_reach_is_a_whole_number_per_language_refused_by_name_and_fingerprinted():
    mod, restore = _probes()
    try:
        first = mod.build_lexicon(_SECTION)
        assert dict(first.reach) == {"fr": 3, "en": 3}, first.reach
        wider = mod.build_lexicon(_altered((("en", "reach"), 4)))
        assert dict(wider.reach) == {"fr": 3, "en": 4} and wider.fingerprint != first.fingerprint
        for value in (_DELETE, 0, -1, True, 2.5, "3", None):
            with pytest.raises(mod.LexiconError) as caught:
                mod.build_lexicon(_altered((("fr", "reach"), value)))
            message = str(caught.value)
            assert "decisions.fr" in message and "reach" in message, (value, message)
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP24 -- a decision's marker words stay in its key
# ---------------------------------------------------------------------------
def test_dp24_a_decision_reworded_into_a_weaker_status_or_modality_does_not_answer():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        weakened = (
            ("We decided to drop Docker.", "We discussed dropping Docker."),
            ("We agreed to keep Postgres.", "Someone proposed keeping Postgres."),
            ("On a d" + _E + "cid" + _E + " d'abandonner Helm.", "On a envisag" + _E + " d'abandonner Helm."),
            ("On doit garder Docker.", "On peut garder Docker."),
            ("We must keep Postgres.", "We may keep Postgres."),
        )
        for said, reworded in weakened:
            [probe] = _decisions(mod, said, lexicon)
            assert _answers(mod, probe, said), f"control: {said!r} answers for itself"
            assert not _answers(mod, probe, reworded), f"{reworded!r} weakens {said!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP25 -- what the generator decides with a lexicon is pinned with its version
# ---------------------------------------------------------------------------
# The digest of the decisions drawn from the corpus of DP25 with the
# fixture's lexicon, by version. A new version adds its line; a line is
# never rewritten.
_DECISION_PINS = {
    5: "4cba6b87d33d9061430e23368ffd385fbd6bf985527657bf1a40f3f89f84dca5",
    6: "4cba6b87d33d9061430e23368ffd385fbd6bf985527657bf1a40f3f89f84dca5",
    7: "4cba6b87d33d9061430e23368ffd385fbd6bf985527657bf1a40f3f89f84dca5",
}
_PINNED = (
    "On garde Docker jusqu'au 5 octobre 2026.",
    "Nous renoncons a Swarm.",
    "Let's go with Podman on 3 hosts.",
    "Keep `docker-compose` for 2 weeks.",
    "We decided to drop Helm.",
    "We never replace Postgres.",
    "Click on Select.",
    "We really truly always prefer Nix.",
)


def test_dp25_what_the_generator_decides_with_a_lexicon_is_pinned_with_its_version():
    mod, restore = _probes()
    try:
        span = [dict(_turn("user", "typed", text), turn_id=f"p{i}") for i, text in enumerate(_PINNED)]
        drawn = mod.generate_probes(span, mod.build_lexicon(_SECTION))
        rows = [[p.answer, sorted(p.key), p.negations] for p in drawn if p.kind == "decision"]
        assert len(rows) == 6, f"the corpus decides six times: {len(rows)}"
        digest = hashlib.sha256(json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest()
        assert mod.GENERATOR_VERSION == max(_DECISION_PINS), "the version is the last one pinned"
        assert _DECISION_PINS[mod.GENERATOR_VERSION] == digest, (
            f"the generator decides otherwise than its version pinned: raise GENERATOR_VERSION and pin {digest}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# DP26 -- a French decision typed without its accents
# ---------------------------------------------------------------------------
# It decides by its marker alone: "supprimer" is no act of the lexicon.
_ACCENTED = "On a d" + _E + "cid" + _E + " de supprimer les sauvegardes du serveur Atlas le 5 octobre 2026."
_BARE = "On a decide de supprimer les sauvegardes du serveur Atlas le 5 octobre 2026."


def test_dp26_a_french_decision_typed_without_its_accents_decides_as_with_them():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        accented, bare = _decisions(mod, _ACCENTED, lexicon), _decisions(mod, _BARE, lexicon)
        assert len(accented) == 1, "control: written with its accents, the sentence decides"
        assert len(bare) == 1, "written without them, it decides as well"
        typed = ("act:", "date:", "number:")
        assert ({m for m in bare[0].key if m.startswith(typed)}, bare[0].negated) == (
            {m for m in accented[0].key if m.startswith(typed)}, accented[0].negated)
        assert _answers(mod, bare[0], _ACCENTED), "the same decision with its accents answers it"
        assert not _answers(mod, bare[0], "Les sauvegardes du serveur Atlas, le 5 octobre 2026."), "a summary that drops it fails"
    finally:
        restore()


_ACUTE = chr(0x0301)  # a combining acute accent: the decomposed form a keyboard may type


def test_dp27_an_accented_marker_decides_folded_wherever_it_opens_a_word_and_never_inside_one():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        decides = [
            _BARE, _ACCENTED,
            "Il a ete decide de supprimer les sauvegardes du serveur Atlas.",
            "L'equipe a decide de garder Docker.",
            "Nous avons donc finalement decide de garder Docker.",
            "C'est decide, on garde Docker.",
            "Vous avez opte pour Podman.",
            "L'" + _E + "quipe a d" + "e" + _ACUTE + "cide" + _ACUTE + " de supprimer les sauvegardes.",
        ]
        inside = ["The helicopter photos are in the archive.", "Is this grammar undecidable in general?"]
        drawn = {text: len(_decisions(mod, text, lexicon)) for text in decides + inside}
    finally:
        restore()
    assert drawn[_ACCENTED] == 1 and drawn[_BARE] == 1, "control: a marker decides, accents typed or not"
    missed = [text for text in decides if drawn[text] != 1]
    assert not missed, f"a French decision decides wherever its marker opens a word, with a subject or none: {missed}"
    assert all(drawn[text] == 0 for text in inside), {text: drawn[text] for text in inside}


_GRAVE = chr(0x0300)  # a combining grave accent


def test_dp28_a_decision_decides_however_its_text_is_written():
    mod, restore = _probes()
    try:
        lexicon = _shipped(mod)
        composed = ["Je pr" + _E + "f" + chr(0xE8) + "re Docker.", "On " + _E + "vite Docker.",
                    "Il faut supprimer Docker.", "We will ship Harvest to Oslo."]
        written = [
            "Je pre" + _ACUTE + "fe" + _GRAVE + "re Docker.",
            "On e" + _ACUTE + "vite Docker.",
            "Il est pr" + _E + "vu d'utiliser Docker.",
            "Nous avons pr" + _E + "vu d'abandonner Docker.",
            "Il\nfaut supprimer Docker.",
            "On va\nsupprimer Docker.",
            "We will\nship Harvest to Oslo.",
            "Je vais" + chr(0xA0) + "supprimer Docker.",
        ]
        drawn = {text: len(_decisions(mod, text, lexicon)) for text in composed + written}
    finally:
        restore()
    assert all(drawn[text] == 1 for text in composed), {text: drawn[text] for text in composed}
    missed = [ascii(text) for text in written if drawn[text] != 1]
    assert not missed, f"a decision decides however its text is written: {missed}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
