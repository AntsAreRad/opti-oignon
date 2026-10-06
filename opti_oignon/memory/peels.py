#!/usr/bin/env python3
"""Peels: the summary tree of the onion memory, and the gate that grows it.

A peel is a summary that has answered for the span it replaces. It carries
the Cellar keys of its source spans, a digest of their content, and the
probe score it earned when it was made. Two rules keep the tree honest.
Recompress from source: a parent peel is summarised from the union of its
children's Cellar spans, never from the children's text, so a photocopy of
a photocopy cannot be produced here and is refused when handed in -- a
source that resolves to no span, a parent whose sources are not its
children's, a digest that no longer matches the spans, a text that no
longer matches the id, each by name. Probe-gated eviction: a span leaves
the Flesh only once its candidate peel answers the recall probes drawn from
the span, class by class, at the thresholds of ``onion.yaml``. Below the
gate the verbatim turns stay and the decision says which probes failed. An
empty probe set is refused, not passed: an unknown rate is not a pass. The
peel must also say nothing its span does not hold, and stay within two
bounds against it -- the share of its content words the span holds no word
for, and its length -- each refused by name with its figure. Nor is a probe
set taken on trust: the span's facts are read again, and a set that asks
for less of them than the floor is refused with the facts it leaves unasked.

Selection at query time is deterministic and keyword-based, in any script:
a term is a word as the probes read one, in lower case with its accents
taken off, the function and question words of both languages aside. The vector
layer for peels is a host decision (which store, which embedder, measured
against the existing one) and nothing here claims it. The summariser is a
seam: the librarian's model on the host, a recording fake in the
contracts. The two metrics at the end are fixture readings and say so in
their ``source`` field. The executor reaches this module through the
librarian and nothing else does; a contract on the tree says so.
"""

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

checkpoint_before_apply = True

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "onion.yaml"
# The words a query asks with rather than about, in both languages, folded;
# the function words are the reader's, ``probes.FUNCTION_WORDS``.
_QUESTION_WORDS = frozenset({
    "what", "which", "who", "how", "happened", "about",
    "quoi", "quel", "quelle", "quels", "quelles", "comment",
})


class PeelIntegrityError(ValueError):
    """A peel does not stand on what it claims to stand on."""


class GateError(ValueError):
    """The gate cannot be applied as configured."""


def estimate_tokens(text):
    """The fallback estimate, the same as the composer's and the retriever's."""
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


def _canonical(spans):
    return json.dumps([list(s) for s in spans], sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _native():
    """The native core, or None: asked at the call, never at import."""
    try:
        from opti_oignon.native import load
    except Exception:  # noqa: BLE001 - absence is the reference path
        return None
    return load()


def source_digest(cellar, sources):
    """The digest of the spans behind ``sources``, read from the Cellar now."""
    spans = [cellar.get(k) for k in sources]
    core = _native()
    if core is not None:
        try:
            return core.digest_spans(spans)
        except TypeError:
            pass  # a value shape the core refuses: the reference formats it
    return hashlib.sha256(_canonical(spans).encode("utf-8")).hexdigest()


def peel_id(text, sources):
    core = _native()
    if core is not None:
        return core.peel_id(text, [str(s) for s in sources])
    return hashlib.sha256(json.dumps([text, list(sources)], ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Gate:
    decision_threshold: float
    episodic_threshold: float
    span_turns: int
    # The decision lexicon the gate's probes are drawn with; None is the
    # empty one.
    lexicon: object = None
    # The threshold of the code class. None leaves code probes in the
    # episodic class, as a gate built without one judged them.
    code_threshold: float = None
    # The words that make a deciding sentence of a summary a report of
    # another speaker's words, when one is its subject. None is none: every
    # deciding sentence then needs a typed decision behind it.
    reporters: frozenset = None
    # The bounds of a summary against its span: the most of its content
    # words its span may hold no word for, and the most words it may write
    # for each word of the span. None is no bound: the figure is measured
    # and said, and refuses nothing.
    max_novelty: float = None
    max_length_ratio: float = None
    # What the gate's probes recall of the facts a reader keeps, a
    # ``ProbeRecall`` as the gate's file states it. None states none, as a
    # gate built by hand does. It refuses nothing: every decision carries
    # it, or says why it does not hold.
    probe_recall: object = None
    # The least share of the facts its span holds that a summary's probes
    # must ask for, the facts read again by ``probes.held_facts``. None is no
    # floor: the share is measured and said, and refuses nothing.
    probe_floor: float = None

    def validate(self):
        errors = []
        for name in ("decision_threshold", "episodic_threshold"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"{name}: {value!r} is not within [0, 1]")
        if not isinstance(self.span_turns, int) or self.span_turns < 1:
            errors.append(f"span_turns: {self.span_turns!r} is not a positive integer")
        if self.code_threshold is not None and not _fraction(self.code_threshold):
            errors.append(f"code_threshold: {self.code_threshold!r} is not a number within [0, 1]")
        if self.reporters is not None and not (
            isinstance(self.reporters, frozenset)
            and all(isinstance(word, str) and _REPORTER.fullmatch(word) for word in self.reporters)
        ):
            errors.append(f"reporters: {self.reporters!r} is not a set of lower-case ASCII words")
        if self.max_novelty is not None and not _fraction(self.max_novelty):
            errors.append(f"max_novelty: {self.max_novelty!r} is not a number within [0, 1]")
        if self.max_length_ratio is not None and not _ratio(self.max_length_ratio):
            errors.append(f"max_length_ratio: {self.max_length_ratio!r} is not a finite number above 0")
        if self.probe_floor is not None and not _fraction(self.probe_floor):
            errors.append(f"probe_floor: {self.probe_floor!r} is not a number within [0, 1]")
        if self.probe_recall is not None:
            from .probes import ProbeRecall

            if not isinstance(self.probe_recall, ProbeRecall):
                errors.append(f"probe_recall: a {type(self.probe_recall).__name__} is not a stated probe recall")
        return errors


# A reporter is one lower-case ASCII word, written in one of the languages
# the decision lexicon knows; its plural in -s is read with it.
_REPORTER = re.compile(r"[a-z]+")
_REPORTER_LANGUAGES = ("fr", "en")


def _fraction(value):
    """True for a finite number within [0, 1]; a boolean is no number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def _ratio(value):
    """True for a finite number above 0; a boolean is no number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 < value < float("inf")


# The bounds ``onion.yaml`` must give, each with its test and what it must be.
_BOUNDS = (
    ("max_novelty", _fraction, "a number within [0, 1]"),
    ("max_length_ratio", _ratio, "a finite number above 0"),
)


def _reporters(section):
    """The reporters of ``onion.yaml``, each word with its plural, refused by name when malformed."""
    if section is None:
        raise GateError("onion gate: reporters are missing, and a gate needs to know whose words a report gives")
    if not isinstance(section, dict):
        raise GateError(f"onion gate: reporters {section!r} do not map a language to its words")
    words = set()
    for language, members in section.items():
        if language not in _REPORTER_LANGUAGES:
            raise GateError(f"onion gate: reporters: unknown language {language!r}")
        if not isinstance(members, list):
            raise GateError(f"onion gate: reporters: {language} holds {members!r}, not a list of words")
        for member in members:
            if not isinstance(member, str) or not _REPORTER.fullmatch(member):
                raise GateError(f"onion gate: reporters: {member!r} is not one lower-case ASCII word")
            words.update((member, member + "s"))
    return frozenset(words)


def load_gate(path=None):
    """The gate of ``onion.yaml`` with its lexicon, code threshold, reporters and bounds, refused if out of range or malformed."""
    import yaml

    from .probes import LexiconError, build_lexicon

    raw = yaml.safe_load(Path(path or _CONFIG).read_text(encoding="utf-8")) or {}
    try:
        gate = Gate(
            decision_threshold=float(raw["gate"]["decision_threshold"]),
            episodic_threshold=float(raw["gate"]["episodic_threshold"]),
            span_turns=int(raw["peels"]["span_turns"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError(f"onion gate is incomplete or malformed: {exc!r}") from exc
    errors = gate.validate()
    if errors:
        raise GateError("; ".join(errors))
    if "code_threshold" not in raw["gate"]:
        raise GateError("onion gate: code_threshold is missing, and a gate needs the threshold of its code class")
    code = raw["gate"]["code_threshold"]
    if not _fraction(code):
        raise GateError(f"onion gate: code_threshold {code!r} is not a number within [0, 1]")
    reporters = _reporters(raw["gate"].get("reporters"))
    bounds = []
    for name, valid, said in _BOUNDS:
        if name not in raw["gate"]:
            raise GateError(f"onion gate: {name} is missing, and a gate needs the bounds a summary is held to")
        if not valid(raw["gate"][name]):
            raise GateError(f"onion gate: {name} {raw['gate'][name]!r} is not {said}")
        bounds.append(float(raw["gate"][name]))
    if "probe_floor" not in raw["gate"]:
        raise GateError("onion gate: probe_floor is missing, and a gate needs the share of a span's facts its probes "
                        "must ask for")
    floor = raw["gate"]["probe_floor"]
    if not _fraction(floor):
        raise GateError(f"onion gate: probe_floor {floor!r} is not a number within [0, 1]")
    if "decisions" not in raw:
        raise GateError("onion decisions: the section is missing, and a gate needs its decision lexicon")
    try:
        lexicon = build_lexicon(raw["decisions"])
    except LexiconError as exc:
        raise GateError(f"onion {exc}") from exc
    stated = _probe_recall(raw["gate"].get("probe_recall"))
    return Gate(
        gate.decision_threshold, gate.episodic_threshold, gate.span_turns, lexicon, float(code), reporters, *bounds,
        probe_recall=stated, probe_floor=float(floor),
    )


# The keys of a stated probe recall, in the order its refusals name them.
_RECALL_KEYS = ("classes", "generator", "lexicon", "source")
# The one set a contract can read again: the labelled fixture set of the
# tests. A figure read on any other set is reported, never stated.
_RECALL_SOURCES = ("fixture",)
_NO_RECALL = "no recall stated: the probes were not read on a labelled set"


def _count(value, low):
    """True for a whole number from ``low``; a boolean is no count."""
    return type(value) is int and value >= low


def _probe_recall(section):
    """The probe recall the gate's file states, refused by name if malformed; None when it states none."""
    from .probes import LEXICON_FINGERPRINT_LENGTH, RECALL_CLASSES, ProbeRecall

    if section is None:
        return None
    if not isinstance(section, dict) or sorted(map(str, section)) != sorted(_RECALL_KEYS):
        held = sorted(map(str, section)) if isinstance(section, dict) else type(section).__name__
        raise GateError(f"onion gate: probe_recall holds {held}, and a stated recall is {', '.join(_RECALL_KEYS)}")
    source, generator, lexicon = section["source"], section["generator"], section["lexicon"]
    if source not in _RECALL_SOURCES:
        raise GateError(f"onion gate: probe_recall source {str(source)[:24]!r} is not fixture, the set a contract reads again")
    if not _count(generator, 1):
        raise GateError(f"onion gate: probe_recall generator {generator!r} is not a version, a whole number from 1")
    if not (isinstance(lexicon, str) and re.fullmatch("[0-9a-f]{%d}" % LEXICON_FINGERPRINT_LENGTH, lexicon)):
        raise GateError(f"onion gate: probe_recall lexicon {str(lexicon)[:24]!r} is not a lexicon fingerprint "
                        f"({LEXICON_FINGERPRINT_LENGTH} lower-case hexadecimal digits, quoted)")
    classes = section["classes"]
    if not isinstance(classes, dict) or sorted(map(str, classes)) != sorted(RECALL_CLASSES):
        held = sorted(map(str, classes)) if isinstance(classes, dict) else type(classes).__name__
        raise GateError(f"onion gate: probe_recall classes are {held}, and a stated recall gives each of "
                        f"{', '.join(RECALL_CLASSES)}")
    rows = []
    for name in RECALL_CLASSES:
        row = classes[name]
        if not isinstance(row, dict) or sorted(map(str, row)) != ["recalled", "size"]:
            raise GateError(f"onion gate: probe_recall of {name} is not a size and a recalled count")
        size, recalled = row["size"], row["recalled"]
        if not _count(size, 1):
            raise GateError(f"onion gate: probe_recall of {name}: size {size!r} is not a count of facts, a whole number "
                            "from 1")
        if not (_count(recalled, 0) and recalled <= size):
            raise GateError(f"onion gate: probe_recall of {name}: recalled {recalled!r} is not a count from 0 to its size "
                            f"{size}")
        rows.append((name, recalled, size))
    return ProbeRecall(source, generator, lexicon, tuple(rows))


def stated_probe_recall(stated, lexicon):
    """The probe recall that holds for probes this generator draws with ``lexicon``, and what is said of it.

    Returns ``(stated, note)``: the stated probe recall and an empty note
    when it was read with this version of the generator and this lexicon;
    otherwise None and the reason. A figure read on another generator or
    another lexicon is never borrowed: the lexicon is the user's to change.
    """
    from .probes import GENERATOR_VERSION

    if stated is None:
        return None, _NO_RECALL
    if stated.generator != GENERATOR_VERSION:
        return None, (f"no probe recall for generator {GENERATOR_VERSION}: the stated one was read with generator "
                      f"{stated.generator}")
    fingerprint = _fingerprint(lexicon)
    if stated.lexicon != fingerprint:
        return None, (f"no probe recall for lexicon {fingerprint}: the stated one was read with lexicon "
                      f"{stated.lexicon}")
    return stated, ""


def _fingerprint(lexicon):
    """The fingerprint of a decision lexicon; None is the empty one."""
    from .probes import EMPTY_LEXICON

    return (EMPTY_LEXICON if lexicon is None else lexicon).fingerprint


@dataclass(frozen=True)
class Peel:
    """A summary with the spans it stands on and the score it earned."""

    id: str
    text: str
    level: int
    sources: tuple
    children: tuple
    source_digest: str
    probes_passed: int
    probes_total: int


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    result: object
    decision_rate: float = None
    episodic_rate: float = None
    # What the rates are rates of: the version of the generator and the
    # fingerprint of the lexicon the probes were drawn with.
    generator: int = None
    lexicon: str = None
    # The rate of the code class, None when the gate has none or the span
    # no code.
    code_rate: float = None
    # What the second face found: each claim of the summary its span does
    # not hold, as ``(kind, what, turn)``.
    unsupported: tuple = ()
    # The figures of the bounds: the share of the summary's content words
    # its span holds no word for, None without a content word; its words
    # over its span's, None for a span with no word. Measured whatever the
    # gate holds; ``max_novelty`` and ``max_length_ratio`` are the bounds
    # they were held to, None for none.
    novelty: float = None
    length_ratio: float = None
    max_novelty: float = None
    max_length_ratio: float = None
    # What the probes recall of the facts a reader keeps: the gate's stated
    # ``ProbeRecall`` when it was read with this generator and this lexicon,
    # with an empty note; otherwise None, and the note says why. A rate is
    # read with it: a fact no probe was drawn for is a fact no rate counts.
    probe_recall: object = None
    probe_recall_note: str = _NO_RECALL
    # What share of the facts its span holds the probes ask for, measured
    # whatever the gate holds: the facts, read again by ``probes.held_facts``;
    # the share, None for a span with no fact; the facts no probe asks for,
    # each ``(kind, what, turn)``; and the floor the share was held to, None
    # for none. ``judge`` alone reads no span and leaves them unset.
    facts: int = None
    probe_coverage: float = None
    unasked: tuple = ()
    probe_floor: float = None


@dataclass(frozen=True)
class Selected:
    """One selected peel, in the shape the composer takes."""

    text: str
    provenance: str
    score: float = 0.0


@dataclass(frozen=True)
class Eviction:
    evicted: bool
    reason: str
    receipt: object = None
    peel: object = None
    decision: object = None


class PeelTree:
    def __init__(self):
        self._peels = {}
        self._order = []

    def add(self, peel):
        if peel.id not in self._peels:
            self._order.append(peel.id)
        self._peels[peel.id] = peel
        return peel.id

    def get(self, peel_id_):
        return self._peels[peel_id_]

    def has(self, peel_id_):
        return peel_id_ in self._peels

    def all(self):
        return [self._peels[i] for i in self._order]

    def leaves(self):
        return [p for p in self.all() if not p.children]

    def roots(self):
        covered = {c for p in self.all() for c in p.children}
        return [p for p in self.all() if p.id not in covered]

    def verify(self, cellar):
        for peel in self.all():
            verify_peel(peel, cellar, self)
        return None


def _rate(probes, result_failures):
    if not probes:
        return None
    failed = sum(1 for p in result_failures if p in probes)
    return round((len(probes) - failed) / len(probes), 4)


def judge(probes, text, gate):
    """Score ``text`` against ``probes`` and decide, class by class.

    The decision names the generator and the lexicon the probes were drawn
    with. Probes drawn with a lexicon other than the gate's are refused: no
    figure names a lexicon its probes were not drawn with. Code probes are
    a class of their own when the gate holds a code threshold: a peel is
    the only index the model reads a code block's key from, so a lost marker
    is a block the model no longer knows exists.
    """
    from .probes import GENERATOR_VERSION, score

    errors = gate.validate()
    if errors:
        raise GateError("; ".join(errors))
    lexicon = _fingerprint(gate.lexicon)
    foreign = sorted({_fingerprint(p.lexicon) for p in probes} - {lexicon})
    if foreign:
        raise GateError(f"probes drawn with lexicon {foreign[0]} cannot be judged by a gate holding lexicon {lexicon}")
    stated, note = stated_probe_recall(gate.probe_recall, gate.lexicon)
    named = {"generator": GENERATOR_VERSION, "lexicon": lexicon, "probe_recall": stated, "probe_recall_note": note}
    result = score(probes, text)
    if not probes:
        return GateDecision(
            False, "no probe could be drawn from the span: an unknown rate is not a pass", result, **named
        )
    coded = gate.code_threshold is not None
    decision = [p for p in probes if p.kind == "decision"]
    code = [p for p in probes if coded and p.kind == "code"]
    episodic = [p for p in probes if p.kind != "decision" and not (coded and p.kind == "code")]
    decision_rate = _rate(decision, result.failures)
    episodic_rate = _rate(episodic, result.failures)
    named["code_rate"] = _rate(code, result.failures)
    short = []
    if decision_rate is not None and decision_rate < gate.decision_threshold:
        short.append(f"decision probes at {decision_rate} against {gate.decision_threshold}")
    if episodic_rate is not None and episodic_rate < gate.episodic_threshold:
        short.append(f"episodic probes at {episodic_rate} against {gate.episodic_threshold}")
    if named["code_rate"] is not None and named["code_rate"] < gate.code_threshold:
        short.append(f"code probes at {named['code_rate']} against {gate.code_threshold}")
    if short:
        failed = ", ".join(f"{p.kind}:{p.answer[:24]}@{p.turn_id}" for p in result.failures)
        reason = "; ".join(short) + f" -- failed: {failed}"
        return GateDecision(False, reason, result, decision_rate, episodic_rate, **named)
    reason = "every probe class present meets its threshold"
    return GateDecision(True, reason, result, decision_rate, episodic_rate, **named)


def faithfulness(span, probes, text, gate):
    """The second face of the gate: what ``text`` says that ``span`` does not hold, each ``(kind, what, turn)``.

    Names, dates, numbers and code come first, read by
    ``probes.unsupported_claims``; then the sentences that decide with no
    typed decision of the span behind them, read by
    ``probes.unbacked_decisions`` with the gate's lexicon and reporters.
    """
    from .probes import holdings

    return _unheld(holdings(span), probes, text, gate)


def _unheld(held, probes, text, gate):
    from .probes import unbacked_decisions, unsupported_claims

    found = unsupported_claims(held, text)
    found += unbacked_decisions(probes, text, gate.lexicon, gate.reporters or frozenset())
    return tuple(found)


def decide(span, probes, text, gate):
    """Both faces of the gate on a summary of ``span``: every site that judges a summary judges it here.

    The first face is ``judge``: the probes the summary must answer. The
    second is ``faithfulness``: whatever the rates, a claim the span does
    not hold refuses the summary by name; and its bounds: a summary whose
    content words its span mostly holds no word for has drifted from it, and
    one longer than its span saves nothing, each refused with its figure in
    the reason, never as a claim. The probes are not taken on trust: the
    span's facts are read again, and a set that leaves more of them unasked
    than the floor allows judged the summary on less than its span holds,
    refused with its share and the facts no probe asks for. A decision
    refused on several counts gives every reason: the first face's, the
    claims, the bounds, the floor.
    """
    from .probes import holdings, novel_words, probe_coverage, word_count

    first = judge(probes, text, gate)
    held = holdings(span)
    found = _unheld(held, probes, text, gate)
    content, new = novel_words(held, text, gate.lexicon, gate.reporters or frozenset())
    words, span_words = word_count(text), sum(word_count(t) for t in held.texts)
    novelty = len(new) / len(content) if content else None
    ratio = words / span_words if span_words else None
    coverage = probe_coverage(span, probes, gate.lexicon)
    figures = dict(novelty=novelty, length_ratio=ratio, max_novelty=gate.max_novelty,
                   max_length_ratio=gate.max_length_ratio, facts=coverage.facts, probe_coverage=coverage.share,
                   unasked=coverage.unasked, probe_floor=gate.probe_floor)
    over = []
    if novelty is not None and gate.max_novelty is not None and novelty > gate.max_novelty:
        shown = ", ".join(list(dict.fromkeys(word[:24] for word in new))[:5])
        over.append(f"novelty {round(novelty, 4)} over {gate.max_novelty} "
                    f"({len(new)} of {len(content)} content words new: {shown})")
    if ratio is not None and gate.max_length_ratio is not None and ratio > gate.max_length_ratio:
        over.append(f"length ratio {round(ratio, 4)} over {gate.max_length_ratio} ({words} words for {span_words})")
    share = coverage.share
    if share is not None and gate.probe_floor is not None and share < gate.probe_floor:
        shown = ", ".join(f"{kind}:{what[:24]}" + (f"@{turn}" if turn else "") for kind, what, turn in coverage.unasked[:5])
        over.append(f"probe coverage {round(share, 4)} under {gate.probe_floor} "
                    f"({coverage.asked} of {coverage.facts} facts asked, unasked: {shown})")
    if not found and not over:
        return replace(first, **figures)
    reasons = [] if first.accepted else [first.reason]
    if found:
        said = ", ".join(f"{kind}:{what[:24]}" + (f"@{turn}" if turn else "") for kind, what, turn in found)
        reasons.append(f"the summary says what its span does not hold -- {said}")
    return GateDecision(
        False, "; ".join(reasons + over), first.result, first.decision_rate, first.episodic_rate,
        generator=first.generator, lexicon=first.lexicon, code_rate=first.code_rate, unsupported=found, **figures,
        probe_recall=first.probe_recall, probe_recall_note=first.probe_recall_note,
    )


def _summarise(sources, cellar, summarize, gate):
    from .probes import generate_probes

    spans = [cellar.get(k) for k in sources]
    turns = [t for span in spans for t in span]
    probes = generate_probes(turns, gate.lexicon)
    text = str(summarize(turns))
    return spans, probes, text, decide(turns, probes, text, gate)


def _make(text, sources, level, children, decision, cellar):
    result = decision.result
    return Peel(
        id=peel_id(text, sources),
        text=text,
        level=level,
        sources=tuple(sources),
        children=tuple(children),
        source_digest=source_digest(cellar, sources),
        probes_passed=result.passed,
        probes_total=result.passed + result.failed,
    )


def build_leaf(key, cellar, summarize, gate, tree):
    """A level-0 peel over one Cellar span, added to the tree only if the gate accepts."""
    if not cellar.has(key):
        raise PeelIntegrityError(f"source {key} resolves to no Cellar span")
    _spans, _probes, text, decision = _summarise((key,), cellar, summarize, gate)
    if not decision.accepted:
        return None, decision
    peel = _make(text, (key,), 0, (), decision, cellar)
    tree.add(peel)
    return peel, decision


def build_parent(child_ids, cellar, summarize, gate, tree):
    """A peel over the union of its children's spans, summarised from the Cellar, never from the children."""
    children = []
    for cid in child_ids:
        if not tree.has(cid):
            raise PeelIntegrityError(f"child {cid} is not in the tree")
        children.append(tree.get(cid))
    sources = []
    for child in children:
        for key in child.sources:
            if key not in sources:
                sources.append(key)
    if not sources:
        raise PeelIntegrityError("a parent needs at least one child with a source")
    _spans, _probes, text, decision = _summarise(tuple(sources), cellar, summarize, gate)
    if not decision.accepted:
        return None, decision
    level = max(c.level for c in children) + 1
    peel = _make(text, tuple(sources), level, tuple(c.id for c in children), decision, cellar)
    tree.add(peel)
    return peel, decision


def verify_peel(peel, cellar, tree=None):
    """Refuse, by name, a peel that does not stand on what it claims."""
    for key in peel.sources:
        if not cellar.has(key):
            raise PeelIntegrityError(f"peel {peel.id}: source {key} resolves to no Cellar span")
    if peel_id(peel.text, peel.sources) != peel.id:
        raise PeelIntegrityError(f"peel {peel.id}: text or sources no longer hash to the id")
    if peel.children:
        if tree is None:
            raise PeelIntegrityError(f"peel {peel.id}: has children and no tree to resolve them in")
        expected = []
        for cid in peel.children:
            if not tree.has(cid):
                raise PeelIntegrityError(f"peel {peel.id}: child {cid} is not in the tree")
            child = tree.get(cid)
            if child.level >= peel.level:
                raise PeelIntegrityError(f"peel {peel.id}: child {cid} is not below it")
            for key in child.sources:
                if key not in expected:
                    expected.append(key)
        if list(peel.sources) != expected:
            raise PeelIntegrityError(
                f"peel {peel.id}: sources are not the union of its children's sources; "
                "a summary must stand on the Cellar, not on other summaries"
            )
    if source_digest(cellar, peel.sources) != peel.source_digest:
        raise PeelIntegrityError(f"peel {peel.id}: source digest no longer matches the Cellar spans")
    return None


def peel_origins(peel, cellar):
    """The union of the origins of the turns a peel stands on, read from the Cellar now."""
    from .receipts import span_origins

    found = set()
    for key in peel.sources:
        if not cellar.has(key):
            raise PeelIntegrityError(f"peel {peel.id}: source {key} resolves to no Cellar span")
        found.update(span_origins(cellar.get(key)))
    return tuple(sorted(found))


def _terms(text):
    """The terms of a text: its words in any script, folded as the probes fold them, function and question words aside."""
    from .probes import FUNCTION_WORDS, folded_words

    return {w for w in folded_words(text) if w not in FUNCTION_WORDS and w not in _QUESTION_WORDS}


def _lineage(tree, peel):
    """Ids of the peel, its ancestors and its descendants."""
    related = {peel.id}
    stack = list(peel.children)
    while stack:
        cid = stack.pop()
        if cid in related or not tree.has(cid):
            continue
        related.add(cid)
        stack.extend(tree.get(cid).children)
    parents = {c: p.id for p in tree.all() for c in p.children}
    current = peel.id
    while current in parents:
        current = parents[current]
        related.add(current)
    return related


def select_peels(tree, query, cap, estimate=None):
    """Peels for the query, best first, whole items, never an ancestor with its descendant.

    A peel scores the share of the query's terms its text holds; ties go to
    the lower level, then the lower id, so the order is the same in every run.
    """
    estimate = estimate or estimate_tokens
    terms = _terms(query)
    if not terms:
        return []
    scored = []
    for peel in tree.all():
        hit = len(terms & _terms(peel.text)) / len(terms)
        if hit > 0:
            scored.append((-hit, peel.level, peel.id, peel))
    scored.sort()
    chosen, taken, used = [], set(), 0
    for neg, _level, _pid, peel in scored:
        if peel.id in taken:
            continue
        tokens = estimate(peel.text)
        if used + tokens > cap:
            continue
        used += tokens
        taken |= _lineage(tree, peel)
        chosen.append(Selected(text=peel.text, provenance=f"peel:{peel.id[:12]}:L{peel.level}", score=round(-neg, 4)))
    return chosen


def evict_gated(*, flesh, cellar, ledger, tree, gate, summarize):
    """One gated eviction step: the oldest span leaves only if its peel answers for it."""
    from .probes import generate_probes

    errors = gate.validate()
    if errors:
        raise GateError("; ".join(errors))
    turns = flesh.turns()
    if not turns:
        return Eviction(False, "the Flesh is empty; nothing to evict")
    span = turns[: gate.span_turns]
    probes = generate_probes(span, gate.lexicon)
    text = str(summarize([dict(t) for t in span]))
    decision = decide(span, probes, text, gate)
    if not decision.accepted:
        return Eviction(False, decision.reason, decision=decision)
    receipt = flesh.evict_span(len(span), cellar, ledger)
    peel = _make(text, (receipt.key,), 0, (), decision, cellar)
    tree.add(peel)
    return Eviction(True, decision.reason, receipt=receipt, peel=peel, decision=decision)


def fidelity(tree, cellar, lexicon=None, probe_recall=None):
    """Probe pass rate of every peel resampled against its Cellar spans. A fixture reading.

    The probes are drawn with ``lexicon``, the empty one when none is given,
    and the reading names it with the version of the generator. Beside the
    rate stands ``probe_recall``, a gate's stated probe recall, when it
    holds for these probes; otherwise None, and the note says why.
    """
    from .probes import EMPTY_LEXICON, GENERATOR_VERSION, generate_probes, score

    lexicon = EMPTY_LEXICON if lexicon is None else lexicon
    passed = failed = 0
    for peel in tree.all():
        turns = [t for k in peel.sources for t in cellar.get(k)]
        result = score(generate_probes(turns, lexicon), peel.text)
        passed += result.passed
        failed += result.failed
    total = passed + failed
    held, note = stated_probe_recall(probe_recall, lexicon)
    return {
        "source": "fixture",
        "peels": len(tree.all()),
        "probes": total,
        "passed": passed,
        "failed": failed,
        "rate": None if total == 0 else round(passed / total, 4),
        "generator": GENERATOR_VERSION,
        "lexicon": lexicon.fingerprint,
        "probe_recall": None if held is None else held.entry(),
        "probe_recall_note": note,
    }


def context_multiplier(tree, cellar, estimate=None):
    """Source tokens the root peels stand for, over the tokens they cost. A fixture reading."""
    estimate = estimate or estimate_tokens
    roots = tree.roots()
    source_tokens = sum(estimate(str(t.get("text", ""))) for p in roots for k in p.sources for t in cellar.get(k))
    peel_tokens = sum(estimate(p.text) for p in roots)
    return {
        "source": "fixture",
        "roots": len(roots),
        "source_tokens": source_tokens,
        "peel_tokens": peel_tokens,
        "multiplier": None if peel_tokens == 0 else round(source_tokens / peel_tokens, 4),
    }
