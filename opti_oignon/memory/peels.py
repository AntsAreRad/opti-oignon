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

The queue does not stop at a refusal. ``advance`` takes the oldest span down
a ladder below the gate: a span no probe can judge leaves bare, with no
peel; a refused summary is asked for once more, handed the probes it
failed; then it is repaired in the user's own words -- the sentences of the
summary its span holds, and the fewest typed segments answering what they
miss -- and kept while it saves enough; else the span is held, the places
of its typed units kept as anchors in the Cellar. The gate itself never
yields: every peel the ladder makes has passed it.

The user's words never enter a peel as a copy. A peel is a summary and
references: each whole segment the user typed that it keeps, by its turn,
its place and the digest of its bytes, read from the Cellar each time the
peel is shown and shown only while it still answers to that digest. No
summary gives an order, the user's restated included, even word for word:
each segment the user typed that orders is referenced wherever a peel is
made, and only a reference may carry an order. A sentence the repair drops
is kept by its motive and its digest, never its words, and the peel says how
many it lost. A peel is shown with the label of what it shows: a summary is
memory and carries the context and lineage of every turn it stands on; a
reference carries its own turn's; a part a withdrawal reached is not shown.

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


def _ref_tokens(refs):
    """The references of a peel as the tokens its id reads after its sources: none for a peel that makes none."""
    return [f"ref:{turn_id}:{int(start)}:{int(stop)}:{digest}" for turn_id, start, stop, digest in refs]


def peel_id(text, sources, refs=()):
    """A peel's id: its text and its sources, then each reference it makes as a token after them.

    A peel that makes no reference keeps the id it always had, so a stored
    peel answers to it; the native core reads the same list, unchanged.
    """
    keys = [str(s) for s in sources] + _ref_tokens(refs)
    core = _native()
    if core is not None:
        return core.peel_id(text, keys)
    return hashlib.sha256(json.dumps([text, keys], ensure_ascii=False).encode("utf-8")).hexdigest()


def segment_digest(words):
    """The digest a reference holds of the bytes it points at: SHA-256 of their UTF-8, total over any string."""
    return hashlib.sha256(str(words).encode("utf-8", "surrogatepass")).hexdigest()


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
    # The forms an order is given in, a ``probes.Directives`` as the gate's
    # file states them. None reads no order: a gate built by hand judges a
    # summary's claims and decisions only, each decision by its sentence.
    directives: object = None

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
        if self.directives is not None:
            from .probes import Directives

            if not isinstance(self.directives, Directives):
                errors.append(f"directives: a {type(self.directives).__name__} is not a table of directive forms")
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
    """The gate of ``onion.yaml`` with its lexicon, code threshold, reporters, bounds and directives, refused if out of range or malformed."""
    import yaml

    from .probes import DirectivesError, LexiconError, build_directives, build_lexicon

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
    if "directives" not in raw:
        raise GateError("onion directives: the section is missing, and a gate needs the forms an order is given in")
    try:
        directives = build_directives(raw["directives"])
    except DirectivesError as exc:
        raise GateError(f"onion {exc}") from exc
    stated = _probe_recall(raw["gate"].get("probe_recall"))
    return Gate(
        gate.decision_threshold, gate.episodic_threshold, gate.span_turns, lexicon, float(code), reporters, *bounds,
        probe_recall=stated, probe_floor=float(floor), directives=directives,
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
    # The rung of the queue that made it: "accepted" for the librarian's
    # summary as the gate judged it, "reasked" for its second summary,
    # "repaired" for one the queue rebuilt from the user's words.
    rung: str = "accepted"
    # A peel made before references: each run of the user's its text holds
    # as a copy, ``(turn_id, start, stop)`` in its turn. None is made so now.
    stitched: tuple = ()
    # The probes it still fails, ``(kind, answer, turn_id)``: what it lost
    # under the thresholds, kept with it and never logged.
    residual: tuple = ()
    # Each whole segment the user typed that it keeps, ``(turn_id, start,
    # stop, sha256)``: read from the Cellar when it is shown, never copied.
    refs: tuple = ()
    # Each sentence of the summary the repair dropped, ``(motive, sha256)``:
    # what it lost and why, never the words.
    dropped: tuple = ()


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
    # The fingerprint of the table of directive forms the second face read
    # orders with; None when the gate holds none and reads no order.
    directives: str = None


@dataclass(frozen=True)
class Selected:
    """One selected peel, in the shape the composer takes."""

    text: str
    provenance: str
    score: float = 0.0
    # What it shows may carry that the user never endorsed: ``(context,
    # lineage)``. Memory as a whole unless whoever made it said better.
    label: tuple = (("memory",), ())
    # What a query is matched against, when whoever made it says: what it
    # holds -- its summary and the words its references read, shown or not --
    # never the words of a note the queue wrote; a note shown in place of a
    # withheld part stands where the query reaches what it holds.
    reach: str = ""
    # What its showing counted, ``(motive, n)``: the block counts the events
    # of what it places, once.
    events: tuple = ()
    # False when it shows nothing but a note: it then keeps no related peel
    # out of the block.
    shows: bool = True


@dataclass(frozen=True)
class Eviction:
    evicted: bool
    reason: str
    receipt: object = None
    peel: object = None
    decision: object = None
    # How the step ended, by name: "bare", "accepted", "reasked",
    # "repaired" or "held" for the rung its span left on, "stale" for a
    # span that left the head of the Flesh while its summary was written;
    # empty where nothing was attempted or the gate refused.
    rung: str = ""
    # Why the gate refused the summaries the step was given, by name from
    # ``REFUSAL_MOTIVES``, one entry per refusal and motive.
    refused: tuple = ()
    # The sentences the repair dropped from the peel it made, ``(motive,
    # sha256)``, as the peel keeps them.
    dropped: tuple = ()


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
    named = {"generator": GENERATOR_VERSION, "lexicon": lexicon, "probe_recall": stated, "probe_recall_note": note,
             "directives": None if gate.directives is None else gate.directives.fingerprint}
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
    ``probes.unsupported_claims``; then the sentences and clauses that
    decide with no typed decision of the span behind them, read by
    ``probes.unbacked_decisions`` with the gate's lexicon, reporters and
    directives; then the clauses that order with no typed sentence or typed
    decision of the span behind them, read by ``probes.unbacked_directives``
    with the gate's table: a peel is read by every later turn, and only the
    user's words may order there, referenced, never in a summary.
    """
    from .probes import holdings

    return _unheld(holdings(span, gate.directives), probes, text, gate)


def _unheld(held, probes, text, gate):
    from .probes import unbacked_decisions, unsupported_claims

    reporters = gate.reporters or frozenset()
    found = unsupported_claims(held, text)
    found += unbacked_decisions(probes, text, gate.lexicon, reporters, gate.directives, held.typed)
    found += _directives_in_summary(held, probes, text, gate)
    return tuple(found)


def decide(span, probes, text, gate, refs=()):
    """Both faces of the gate on a peel of ``span``: its summary ``text`` and its references ``refs``.

    Every site that judges a peel judges it here. The first face is
    ``judge``: the probes the peel must answer, read in what the model will
    read of it -- the summary and the words its references show, the heads of
    the markers they may hold not yet defanged; the window defangs a head
    alone and keeps every word around it (``composer.defanged``). The second
    is ``faithfulness``, on the summary alone: whatever the rates, a claim
    the span does not hold refuses it by name, and so does an order, even
    one the user gave, even word for word -- the user's words stand in a
    peel only referenced. Then its bounds: a summary whose content words its
    span mostly holds no word for has drifted from it, read in the summary
    alone so that the user's words shown beside it dilute nothing; a peel
    longer than its span saves nothing, read in all it shows; each refused
    with its figure in the reason, never as a claim. The probes are not
    taken on trust: the span's facts are read again, and a set that leaves
    more of them unasked than the floor allows judged the peel on less than
    its span holds, refused with its share and the facts no probe asks for.
    A decision refused on several counts gives every reason: the first
    face's, the claims, the bounds, the floor.
    """
    from .probes import holdings, novel_words, probe_coverage, word_count

    shown = joined(text, shown_words(span, refs))[0]
    first = judge(probes, shown, gate)
    held = holdings(span, gate.directives)
    found = _unheld(held, probes, text, gate)
    content, new = novel_words(held, text, gate.lexicon, gate.reporters or frozenset())
    words, span_words = word_count(shown), sum(word_count(t) for t in held.texts)
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
        probe_recall=first.probe_recall, probe_recall_note=first.probe_recall_note, directives=first.directives,
    )


def _summarise(sources, cellar, summarize, gate):
    from .probes import generate_probes

    spans = [cellar.get(k) for k in sources]
    turns = [t for span in spans for t in span]
    probes = generate_probes(turns, gate.lexicon)
    text, refs = _with_orders(turns, probes, _unmarked(summarize(turns), turns), gate)
    return spans, probes, text, decide(turns, probes, text, gate, refs), refs


def _make(text, sources, level, children, decision, cellar, *, rung="accepted", refs=(), dropped=()):
    result = decision.result
    return Peel(
        id=peel_id(text, sources, refs),
        text=text,
        level=level,
        sources=tuple(sources),
        children=tuple(children),
        source_digest=source_digest(cellar, sources),
        probes_passed=result.passed,
        probes_total=result.passed + result.failed,
        rung=rung,
        residual=tuple((p.kind, p.answer, p.turn_id) for p in result.failures),
        refs=tuple(tuple(ref) for ref in refs),
        dropped=tuple(tuple(entry) for entry in dropped),
    )


def build_leaf(key, cellar, summarize, gate, tree):
    """A level-0 peel over one Cellar span, added to the tree only if the gate accepts."""
    if not cellar.has(key):
        raise PeelIntegrityError(f"source {key} resolves to no Cellar span")
    _spans, _probes, text, decision, refs = _summarise((key,), cellar, summarize, gate)
    if not decision.accepted:
        return None, decision
    peel = _make(text, (key,), 0, (), decision, cellar, refs=refs)
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
    _spans, _probes, text, decision, refs = _summarise(tuple(sources), cellar, summarize, gate)
    if not decision.accepted:
        return None, decision
    level = max(c.level for c in children) + 1
    peel = _make(text, tuple(sources), level, tuple(c.id for c in children), decision, cellar, refs=refs)
    tree.add(peel)
    return peel, decision


def verify_peel(peel, cellar, tree=None):
    """Refuse, by name, a peel that does not stand on what it claims."""
    for key in peel.sources:
        if not cellar.has(key):
            raise PeelIntegrityError(f"peel {peel.id}: source {key} resolves to no Cellar span")
    if any(not _well_formed(ref) for ref in peel.refs):
        raise PeelIntegrityError(f"peel {peel.id}: a reference is no place in a turn with the digest of its bytes")
    if peel_id(peel.text, peel.sources, peel.refs) != peel.id:
        raise PeelIntegrityError(f"peel {peel.id}: text, sources or references no longer hash to the id")
    if peel.refs:
        # The bytes, here: a reference names a place in a user's turn of the peel's spans whose bytes answer to its
        # digest. Whether that place still reads as a whole typed segment is the reader's to say, at each showing:
        # it depends on the grammar of origins, which may change, and a peel that no longer shows one is not refused.
        turns = {str(t.get("turn_id", "")): t for key in peel.sources for t in cellar.get(key)}
        for ref in peel.refs:
            turn = turns.get(ref[0])
            text = str(turn.get("text", "") or "") if turn is not None else ""
            if (turn is None or turn.get("role") != "user" or ref[2] > len(text)
                    or segment_digest(text[ref[1]:ref[2]]) != ref[3]):
                raise PeelIntegrityError(
                    f"peel {peel.id}: a reference no longer reads as the bytes of a user's turn it was made with"
                )
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


# ---------------------------------------------------------------------------
# References: the user's words by their place, never a copy
# ---------------------------------------------------------------------------

# The lineage entry that says a lineage is not whole: cut at the grammar's
# limit, or never recorded.
CUT = "lineage:truncated"
# What a turn id may show in a reference's marker: no bracket, no blank, no
# sign a frame or an envelope is written with.
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


def _received(turn):
    """True for a turn a peer sent, or whose context does not read as a list: no one vouches for who typed it."""
    context = turn.get("context", [])
    return not isinstance(context, (list, tuple)) or "received" in context


def typed_segments(span):
    """The whole segments of a span the user typed, ``(turn_id, start, stop)`` in order: the only words a peel references.

    A segment the origin grammar reads typed -- or a turn of origin typed
    that declares no segment, whole -- of a turn whose id is its own in the
    span and that no peer sent (``received``). Typed words that sit against
    a pasted part are the document's, as the grammar reads them, and none
    of these. A segment of blanks alone is none.
    """
    from .probes import read_origin

    seen = {}
    for turn in span:
        turn_id = str(turn.get("turn_id", ""))
        seen[turn_id] = seen.get(turn_id, 0) + 1
    found = []
    for turn in span:
        turn_id = str(turn.get("turn_id", ""))
        if not turn_id or seen[turn_id] > 1 or _received(turn):
            continue
        text = str(turn.get("text", "") or "")
        origin, segments, _defect = read_origin(turn)
        parts = segments if segments else [(0, len(text), origin)]
        found.extend((turn_id, int(start), int(stop)) for start, stop, label in parts
                     if label == "typed" and text[start:stop].strip())
    return found


def references(span, places):
    """The references a peel makes to keep ``places``: each whole typed segment one of them falls in.

    ``places`` are ``(turn_id, start, stop)``; a segment one of them overlaps
    is referenced whole, ``(turn_id, start, stop, sha256)``, in span order
    and once. A place in no typed segment -- another origin's words, a turn
    a peer sent -- references nothing.
    """
    texts = {str(t.get("turn_id", "")): str(t.get("text", "") or "") for t in span}
    wanted = [(str(turn_id), int(start), int(stop)) for turn_id, start, stop in places]
    return tuple(
        (turn_id, start, stop, segment_digest(texts[turn_id][start:stop]))
        for turn_id, start, stop in typed_segments(span)
        if any(at == turn_id and begin < stop and start < end for at, begin, end in wanted)
    )


def _well_formed(ref):
    """True for a reference of the shape a peel makes: a turn id, two integer places in order, a SHA-256 digest."""
    try:
        turn_id, start, stop, digest = ref
    except (TypeError, ValueError):
        return False
    places = all(isinstance(n, int) and not isinstance(n, bool) for n in (start, stop))
    return (isinstance(turn_id, str) and bool(turn_id) and places and 0 <= start < stop and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None)


def _reread(ref, texts, placed):
    """The words a reference points at, when they still are a whole typed segment answering to its digest; else None."""
    if not _well_formed(ref):
        return None
    turn_id, start, stop, digest = ref
    if (turn_id, start, stop) not in placed:
        return None
    words = texts[turn_id][start:stop]
    return words if segment_digest(words) == digest else None


def read_references(span, refs):
    """The references of ``refs`` that still read in ``span``, ``(turn_id, words)`` in order; the others left out.

    One reads while its place is a whole typed segment of the span and its
    bytes answer to its digest; its words are the turn's bytes, as held.
    """
    texts = {str(t.get("turn_id", "")): str(t.get("text", "") or "") for t in span}
    placed = set(typed_segments(span))
    found = []
    for ref in refs:
        words = _reread(ref, texts, placed)
        if words is not None:
            found.append((str(ref[0]), words))
    return found


def shown_words(span, refs):
    """What the references of ``refs`` show of ``span``: ``(turn_id, words)``, a fenced block as its marker, never its code."""
    from .probes import mask_code

    return [(turn_id, mask_code(words)) for turn_id, words in read_references(span, refs)]


def _header(turn_id, altered):
    """The marker a reference's words follow: its turn, and how many markers the frames' rule defanged in them."""
    shown = _SAFE_ID.sub("", str(turn_id))
    if not altered:
        return f"[{shown}]"
    return f"[{shown}, {altered} marker{'' if altered == 1 else 's'} defanged]"


def _as_is(text):
    return text, 0


def joined(summary, shown, notes=(), defang=_as_is):
    """A peel's parts as one text, and how many markers ``defang`` took in its references' words.

    The summary, then each reference's words after the marker of its turn,
    then one line of what the peel does not show and why, never a word of
    it; joined by a line break. Each part goes through ``defang`` on its own:
    as it is for the gate, which reads what a peel holds; by the composer's
    rule for the model (``peel_text``).
    """
    parts, altered = [], 0
    if summary:
        parts.append(defang(summary)[0])
    for turn_id, words in shown:
        clean, count = defang(words)
        altered += count
        parts.append(f"{_header(turn_id, count)} {clean}")
    if notes:
        parts.append(defang("[not shown: " + "; ".join(notes) + "]")[0])
    return "\n".join(parts), altered


def peel_text(summary, shown, notes=()):
    """What the model reads of a peel, and how many markers its references' words took.

    ``joined``, each part defanged on its own by the composer's rule --
    frames, a marker left open at its end, the envelope's -- so that no frame
    or envelope downstream changes a byte of what is shown here: a
    reference whose words held a marker says so and how many in its own
    marker, and one that held none shows the bytes its turn holds, a fenced
    block as its marker.
    """
    from .composer import defanged

    return joined(summary, shown, notes, defanged)


def join_labels(labels):
    """The union of several ``(context, lineage)``; past the grammar's limit the lineage says it was cut.

    The wrapper's own join, kept here so the onion stays free of the agent
    package; a contract holds the two alike.
    """
    from .probes import _LINEAGE_LIMIT

    context, lineage = set(), set()
    for part_context, part_lineage in labels:
        context.update(part_context)
        lineage.update(part_lineage)
    entries = sorted(lineage)
    if len(entries) > _LINEAGE_LIMIT:
        entries = sorted(entries[: _LINEAGE_LIMIT - 1] + [CUT])
    return sorted(context), entries


def turn_label(turn, lineage=None):
    """The ``(context, lineage)`` of a turn the onion mirrored: what its words may carry that the user never endorsed.

    Its context as mirrored; a turn that declares none reads as its own
    parts give a user turn, legacy for an answer. Its lineage as recorded
    beside it (``lineage``, by turn id); a user turn's, when none was, from
    its own parts as the store derives it; an answer's never recorded is
    unknown, and reads cut. A turn whose parts do not read is legacy, cut.
    """
    from .probes import _user_context, _user_lineage

    role = str(turn.get("role", "") or "")
    text = str(turn.get("text", "") or "")
    origin, segments = turn.get("origin", "legacy"), turn.get("segments") or []
    context = turn.get("context")
    try:
        if not isinstance(context, (list, tuple)):
            context = _user_context(origin, segments) if role == "user" else ["legacy"]
        recorded = (lineage or {}).get(str(turn.get("turn_id", "")))
        if recorded is not None:
            found = list(recorded)
        elif role == "user":
            found = _user_lineage(text, segments)
        else:
            found = [CUT]
        return sorted(set(context)), sorted(set(found))
    except (TypeError, ValueError, IndexError, KeyError):
        return ["legacy"], [CUT]


def withheld(label, withdrawn):
    """True when a withdrawal reached what ``label`` describes.

    Its context says so, its lineage names a source of ``withdrawn``, or its
    lineage was cut -- or never recorded -- while any source is withdrawn.
    """
    context, lineage = label
    if "withdrawn" in context:
        return True
    if not withdrawn:
        return False
    return CUT in lineage or any(entry in withdrawn for entry in lineage)


def _copies_read(peel, turns):
    """True when every run a peel made before references copied still reads in its text after its turn's marker."""
    from .probes import mask_code

    texts = {str(t.get("turn_id", "")): str(t.get("text", "") or "") for t in turns}
    for turn_id, start, stop in peel.stitched:
        held = texts.get(str(turn_id))
        if held is None or not 0 <= int(start) < int(stop) <= len(held):
            return False
        words = held[int(start):int(stop)]
        if f"[{turn_id}] {words}" not in peel.text and f"[{turn_id}] {mask_code(words)}" not in peel.text:
            return False
    return True


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _dropped_note(dropped):
    """What a peel says of the sentences the repair dropped from it: how many and why, never a word; none for none."""
    if not dropped:
        return []
    motives = ", ".join(sorted({str(motive) for motive, _digest in dropped}))
    return [f"{_plural(len(dropped), 'sentence')} the repair dropped from the summary ({motives})"]


def render_peel(peel, cellar, *, lineage=None, withdrawn=()):
    """What the model reads of ``peel`` now: a ``Selected`` with the label of what it shows, or None.

    Read from the Cellar at the call. The summary is shown unless a
    withdrawal reached a turn it stands on (``withheld``); it is memory and
    carries the context and lineage of every turn it stands on. Each
    reference is shown while its words still read as the whole typed
    segment it was made with, its own turn's label beside it; not shown,
    when they no longer read or a withdrawal reached its turn. A peel made
    before references shows its text as it was saved, memory, legacy too
    when a run it copied no longer reads in it, and only a note once a
    withdrawal reached it. What a peel does not show is said in its last
    line, by number and reason, and counted in its events; a query reaches
    it by what it holds, so the note stands where the peel would have.
    """
    withdrawn = frozenset(withdrawn or ())
    turns = [t for key in peel.sources if cellar.has(key) for t in cellar.get(key)]
    labels = {str(t.get("turn_id", "")): turn_label(t, lineage) for t in turns}
    every = join_labels(labels.values())
    summary_label = (sorted({"memory", *every[0]}), every[1])
    provenance = f"peel:{peel.id[:12]}:L{peel.level}"
    if peel.stitched and not peel.refs:
        if withheld(summary_label, withdrawn):
            note = peel_text("", (), ["the peel, a turn it stands on was reached by a withdrawal"])[0]
            return Selected(text=note, provenance=provenance, label=([], []), reach=peel.text,
                            events=(("peels_withheld", 1),), shows=False)
        context = summary_label[0] if _copies_read(peel, turns) else sorted({"legacy", *summary_label[0]})
        return Selected(text=peel.text, provenance=provenance, label=(context, summary_label[1]), reach=peel.text)
    from .probes import mask_code

    notes, events, shown, shown_labels, held = [], [], [], [], [peel.text]
    summary = peel.text
    if summary and withheld(summary_label, withdrawn):
        summary = ""
        notes.append("the summary, a turn it stands on was reached by a withdrawal")
        events.append(("summaries_withheld", 1))
    elif summary:
        shown_labels.append(summary_label)
    texts = {str(t.get("turn_id", "")): str(t.get("text", "") or "") for t in turns}
    placed = set(typed_segments(turns))
    unplaced = reached = 0
    for ref in peel.refs:
        words = _reread(ref, texts, placed)
        if words is None:
            unplaced += 1
            continue
        held.append(words)
        turn_id = str(ref[0])
        if withheld(labels[turn_id], withdrawn):
            reached += 1
            continue
        shown.append((turn_id, mask_code(words)))
        shown_labels.append(labels[turn_id])
    if reached:
        notes.append(f"{_plural(reached, 'reference')} to words of a turn a withdrawal reached")
        events.append(("references_withdrawn", reached))
    if unplaced:
        notes.append(f"{_plural(unplaced, 'reference')} no longer reading as the words it was made with")
        events.append(("references_unplaced", unplaced))
    notes += _dropped_note(peel.dropped)
    if not summary and not shown and not notes:
        return None
    text, altered = peel_text(summary, shown, notes)
    if altered:
        events.append(("references_altered", altered))
    # A query reaches a peel by what it holds -- its summary and the words its references read -- never by the
    # words of its note, which the queue wrote.
    return Selected(text=text, provenance=provenance, label=tuple(join_labels(shown_labels)),
                    reach="\n".join(part for part in held if part), events=tuple(events),
                    shows=bool(summary or shown))


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


def select_peels(tree, query, cap, estimate=None, render=None):
    """Peels for the query, best first, whole items, never an ancestor with its descendant.

    A peel scores the share of the query's terms what it holds holds; ties
    go to the lower level, then the lower id, so the order is the same in
    every run. ``render`` shows a peel as the model will read it
    (``render_peel``), None for one that shows nothing, and says what it
    holds (``reach``), never the words of its note; without it a peel shows
    its text, memory as a whole. A peel placed keeps its ancestors and
    descendants out, unless all it shows is a note.
    """
    estimate = estimate or estimate_tokens
    terms = _terms(query)
    if not terms:
        return []
    scored = []
    for peel in tree.all():
        shown = (render(peel) if render is not None
                 else Selected(text=peel.text, provenance=f"peel:{peel.id[:12]}:L{peel.level}"))
        if shown is None:
            continue
        hit = len(terms & _terms(shown.reach if render is not None else shown.text)) / len(terms)
        if hit > 0:
            scored.append((-hit, peel.level, peel.id, peel, shown))
    scored.sort(key=lambda entry: entry[:3])
    chosen, taken, used = [], set(), 0
    for neg, _level, _pid, peel, shown in scored:
        if peel.id in taken:
            continue
        tokens = estimate(shown.text)
        if used + tokens > cap:
            continue
        used += tokens
        taken |= _lineage(tree, peel) if shown.shows else {peel.id}
        chosen.append(replace(shown, score=round(-neg, 4)))
    return chosen


def evict_gated(*, flesh, cellar, ledger, tree, gate, summarize):
    """One gated eviction step: the oldest span leaves only if its peel answers for it.

    A step for a Flesh no other writer shares: the queue's own step, read
    and committed under the state's lock, is ``advance``.
    """
    from .probes import generate_probes

    errors = gate.validate()
    if errors:
        raise GateError("; ".join(errors))
    turns = flesh.turns()
    if not turns:
        return Eviction(False, "the Flesh is empty; nothing to evict")
    span = turns[: gate.span_turns]
    probes = generate_probes(span, gate.lexicon)
    text, refs = _with_orders(span, probes, _unmarked(summarize([dict(t) for t in span]), span), gate)
    decision = decide(span, probes, text, gate, refs)
    if not decision.accepted:
        return Eviction(False, decision.reason, decision=decision)
    receipt = flesh.evict_span(len(span), cellar, ledger, kind="accepted")
    peel = _make(text, (receipt.key,), 0, (), decision, cellar, refs=refs)
    tree.add(peel)
    return Eviction(True, decision.reason, receipt=receipt, peel=peel, decision=decision, rung="accepted")


# ---------------------------------------------------------------------------
# The queue's ladder: below the gate, a span always leaves
# ---------------------------------------------------------------------------

_LADDER_KEYS = ("rho", "exact_cover", "anchors", "proposals_per_day", "copy_shared_words")


@dataclass(frozen=True)
class Ladder:
    """What the rungs below the gate are held to, read from the ``queue`` section of ``onion.yaml``.

    ``rho``: a repaired peel longer than ``rho`` times its span saves too
    little, and the span is held instead. ``exact_cover``: at or under this
    many candidate units a cover is the fewest; above, greedy. ``anchors``:
    the tokens of the peels layer the anchors of held spans may take.
    ``proposals_per_day``: the most typed decisions offered to the Core per
    conversation and per day, those superseded since included.
    ``copy_shared_words``: the fewest content words a summary's sentence
    shares with a document, a tool or the web for the repair to drop it as
    their copy.
    """

    rho: float
    exact_cover: int
    anchors: int
    proposals_per_day: int
    copy_shared_words: int

    def validate(self):
        errors = []
        shared = self.copy_shared_words
        if isinstance(shared, bool) or not isinstance(shared, int) or shared < 1:
            errors.append(f"copy_shared_words: {shared!r} is not a positive integer")
        if isinstance(self.rho, bool) or not isinstance(self.rho, (int, float)) or not 0.0 < float(self.rho) <= 1.0:
            errors.append(f"rho: {self.rho!r} is not a number in (0, 1]")
        if isinstance(self.exact_cover, bool) or not isinstance(self.exact_cover, int) or not 0 <= self.exact_cover <= 20:
            errors.append(f"exact_cover: {self.exact_cover!r} is not an integer in [0, 20]")
        for name in ("anchors", "proposals_per_day"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"{name}: {value!r} is not a non-negative integer")
        return errors


def load_ladder(path=None):
    """The ladder from ``onion.yaml``, refused by name when a key is missing, unknown or out of range."""
    import yaml

    raw = yaml.safe_load(Path(path or _CONFIG).read_text(encoding="utf-8")) or {}
    section = raw.get("queue")
    if not isinstance(section, dict):
        raise GateError("queue: the section is missing or not a mapping")
    unknown = sorted(set(section) - set(_LADDER_KEYS))
    missing = [key for key in _LADDER_KEYS if key not in section]
    if unknown or missing:
        raise GateError(f"queue: unknown {unknown or 'none'}, missing {missing or 'none'}")
    ladder = Ladder(**{key: section[key] for key in _LADDER_KEYS})
    errors = ladder.validate()
    if errors:
        raise GateError("queue: " + "; ".join(errors))
    return ladder


def cover(units, targets, limit):
    """The fewest ``units`` answering every probe of ``targets``, as indices in order; None when one is answered by none.

    Exact while at most ``limit`` units answer anything: the fewest units,
    then the fewest characters, then the earliest. Above it, greedy and
    deterministic: the unit answering most of what is left, the earliest
    on a tie. A probe that no single unit answers leaves no cover.
    """
    from itertools import combinations

    from .probes import answers

    need = frozenset(range(len(targets)))
    if not need:
        return ()
    candidates = []
    for index, unit in enumerate(units):
        hits = frozenset(i for i, probe in enumerate(targets) if answers(probe, unit.text))
        if hits:
            candidates.append((index, hits, len(unit.text)))
    if frozenset().union(*(hits for _index, hits, _size in candidates)) != need:
        return None
    if len(candidates) <= limit:
        for size in range(1, len(candidates) + 1):
            best = None
            for combo in combinations(candidates, size):
                if frozenset().union(*(hits for _index, hits, _size in combo)) == need:
                    cost = (sum(n for _index, _hits, n in combo), tuple(index for index, _hits, _n in combo))
                    best = cost if best is None or cost < best else best
            if best is not None:
                return best[1]
    chosen, covered = [], frozenset()
    while covered != need:
        index, hits, _size = max(candidates, key=lambda c: (len(c[1] - covered), -c[0]))
        chosen.append(index)
        covered |= hits
    return tuple(sorted(chosen))


def _span_tokens(span):
    return sum(estimate_tokens(str(t.get("text", ""))) for t in span)


def _saves_enough(tokens, span_tokens, rho):
    """True when ``tokens`` are at most ``rho`` times ``span_tokens``, compared as the ratio ``rho`` is written.

    A ratio, never a product: ``0.29 * 100`` is ``28.999...`` in floating
    point, and a repair of exactly 29 tokens would be refused.
    """
    return span_tokens > 0 and tokens / span_tokens <= rho


# Why a step went on without a summary, by name: a class of probes under its
# threshold, a claim its span does not hold, a bound overrun, too few of its
# span's facts asked for, a call that never answered, and a call the queue
# did not make -- the resource governor did not admit it, the span did not
# fit the window, the run had spent its time on the model. A span with no
# probe is no refusal: it leaves bare, counted as such.
REFUSAL_MOTIVES = ("decision", "episodic", "code", "unsupported", "novelty", "length", "coverage", "call_failed",
                   "not_admitted", "over_window", "spent")


class CallRefused(Exception):
    """A call to the model the queue did not make, and why, by a motive of ``REFUSAL_MOTIVES``; never a word of a span."""

    def __init__(self, motive, detail=""):
        super().__init__(f"{motive}: {detail}" if detail else motive)
        self.motive = motive


def refusal_motives(decision, gate):
    """Why ``gate`` refused ``decision``, by name from ``REFUSAL_MOTIVES``: never a word of the summary or its span."""
    if decision is None or decision.accepted:
        return ()
    found = []
    for motive, rate, floor in (("decision", decision.decision_rate, gate.decision_threshold),
                                ("episodic", decision.episodic_rate, gate.episodic_threshold),
                                ("code", decision.code_rate, gate.code_threshold),
                                ("coverage", decision.probe_coverage, gate.probe_floor)):
        if rate is not None and floor is not None and rate < floor:
            found.append(motive)
    if decision.unsupported:
        found.append("unsupported")
    for motive, value, ceiling in (("novelty", decision.novelty, gate.max_novelty),
                                   ("length", decision.length_ratio, gate.max_length_ratio)):
        if value is not None and ceiling is not None and value > ceiling:
            found.append(motive)
    return tuple(found)


def refusal_fingerprint(failures, gate, asker=""):
    """The mark of a refusal: the probes a summary failed, under the generator, the gate and the call refused.

    Never an answer: each failed probe enters as the SHA-256 of its kind,
    answer and turn, and the mark is the SHA-256 of their sorted list with
    the generator's version, the lexicon's fingerprint, every threshold and
    bound of the gate, its reporting verbs, and ``asker``, what names the
    second asking (its model, temperature, seed and prompt), so a change of
    any of them is another mark.
    """
    from .probes import GENERATOR_VERSION

    failed = sorted(
        hashlib.sha256(json.dumps([p.kind, p.answer, p.turn_id], ensure_ascii=False).encode("utf-8")).hexdigest()
        for p in failures
    )
    bounds = [gate.decision_threshold, gate.episodic_threshold, gate.code_threshold, gate.max_novelty,
              gate.max_length_ratio, gate.probe_floor, gate.span_turns]
    payload = json.dumps({"generator": GENERATOR_VERSION, "lexicon": _fingerprint(gate.lexicon), "bounds": bounds,
                          "reporters": sorted(gate.reporters or ()), "asker": str(asker), "failed": failed},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _typed_units(span):
    """The units of a span the user typed: the only words the queue may keep verbatim.

    A document, a tool, the assistant or a turn of no origin is summarised,
    judged and recallable like any other, but its text is never referenced
    by a peel nor kept as an anchor: shown verbatim in what every turn
    reads, an instruction it carries would persist. Nor is a turn a peer
    sent (``received``), whatever origin it declares: no peer vouches for
    who typed it.
    """
    from .probes import units

    received = {str(t.get("turn_id", "")) for t in span if _received(t)}
    return [unit for unit in units(span) if unit.origin == "typed" and unit.turn_id not in received]


def _within_reach(targets, found):
    """The probes of ``targets`` some unit of ``found`` answers."""
    from .probes import answers

    return [p for p in targets if any(answers(p, unit.text) for unit in found)]


# The origins whose words are the conversation's own, for the repair's copy
# filter: what the user typed or had refined, what the assistant answered,
# and a legacy turn -- the least trusted origin, which decides nothing (see
# probes), yet whose words the first face keeps too. A document, or a label
# carrying a flag -- a tool, the web (``assistant+tool``) -- came from
# outside it, and so did an answer whose context says a source outside the
# conversation reached it, or a turn a peer sent (see ``_lowered_turns``).
_CONVERSATION = ("typed", "refined", "assistant", "legacy")


def _lowered_turns(span):
    """The turns of a span whose declared context says their words are not the conversation's own.

    An answer whose context is not the clean one: a document, a page, a tool
    reached it; an answer read from the store before contexts were kept
    declares ``legacy``, and is one of them. A turn of either role a peer
    sent (``received``): no peer vouches for who typed it, so a planted
    question is no more the user's than a planted answer. A turn that declares
    no context at all is judged by its origin alone, as before contexts were
    kept: the store's mirror read declares one for every turn it hands.
    """
    return {
        str(t.get("turn_id", "")) for t in span
        if "context" in t and (
            (t.get("role") == "assistant" and t.get("context") != [])
            or "received" in (t.get("context") or ())
        )
    }


# The innermost bracketed run of a text, by any opening and closing mark a
# marker could be dressed in: every bracket and quotation mark Unicode names
# (categories Ps, Pe, Pi, Pf) and the angle brackets of ASCII, built once on
# first use.
_BRACKETED = []
# The controls of bidirectional text -- marks, embeddings, overrides,
# isolates -- each of which makes a run display in another order than the one
# it is written and read in.
_BIDI_CONTROLS = frozenset(chr(c) for c in (0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)))


def _bracketed():
    if not _BRACKETED:
        import unicodedata

        opening = "<" + "".join(chr(c) for c in range(0x10000) if unicodedata.category(chr(c)) in ("Ps", "Pi"))
        closing = ">" + "".join(chr(c) for c in range(0x10000) if unicodedata.category(chr(c)) in ("Pe", "Pf"))
        _BRACKETED.append(re.compile(
            r"[ \t]*[" + re.escape(opening) + r"]([^" + re.escape(opening + closing) + r"]*)["
            + re.escape(closing) + r"][ \t]*"
        ))
    return _BRACKETED[0]


def _unmarked(text, span):
    """The model's ``text`` without a marker it wrote of what only the queue writes in a peel.

    A marker is a bracket that opens on a turn id -- ``t`` and digits, past
    any punctuation between them when the digits are four or more, as an
    id's are, or an id of ``span``, for any turn -- alone, as a reference's
    marker shows it,
    or with more after the id, as the marker of a reference whose words were
    defanged shows it ("[t0001, 2 markers defanged]"); or a bracket that
    opens on the words of a peel's note ("[not shown: ...]"). The controls
    of bidirectional text are taken out of the text first, so it displays
    in the order it is written and read. A marker is read as a reader would
    see it -- any bracket, any case, any spacing or punctuation between the
    note's letters, compatibility forms folded, the marks a letter carries
    taken off, invisible characters dropped, a trailing colon -- and taken
    out until none is left, one
    inside another included: a summary can write nothing a reader would
    take for the user's words or for the queue's note. Two known
    misses, counted: a bracket is not read past one it holds that stays, and
    a letter of another script that only looks like one of these is not
    read as it. A bracket the user typed, as written inside one unit of
    theirs, is their own words and stays; one a document, the assistant or
    a peer wrote protects nothing, nor does one made of the end of a unit
    and the start of the next.
    """
    import unicodedata

    def fold(text):
        """``text`` with its compatibility forms folded and the marks its letters carry taken off: a t with a caron reads t."""
        bare = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.category(ch).startswith("M"))
        return unicodedata.normalize("NFKC", bare)

    ids = {fold(str(t.get("turn_id", ""))).casefold() for t in span} - {""}
    typed = [unit.text for unit in _typed_units(span)]

    def name(inner, spaced):
        folded = fold(inner)
        kept = "".join((" " if ch.isspace() else ch) if spaced else ("" if ch.isspace() else ch)
                       for ch in folded if unicodedata.category(ch) != "Cf")
        return " ".join(kept.split()).casefold().rstrip(":") if spaced else kept.casefold().rstrip(":")

    def opens_on_id(said, head=r"t\d+"):
        found = re.match(head, said)
        if found and (found.end() == len(said) or not (said[found.end()].isalnum() or said[found.end()] == ".")):
            return True
        return any(said == i or (said.startswith(i) and not said[len(i)].isalnum()) for i in ids)

    def drop(match):
        said, spaced = name(match.group(1), False), name(match.group(1), True)
        # Read with its spaces dropped, ``t`` and the four digits or more of an id open on one past any punctuation
        # between them -- an index such as ``[t-1]`` stays; and the note's words are its letters alone.
        letters = "".join(ch for ch in said if ch.isalnum())
        marker = (opens_on_id(said, r"t[\W_]*\d{4,}") or opens_on_id(spaced)
                  or letters.startswith("notshown"))
        written = match.group(0).strip()
        return " " if marker and not any(written in unit for unit in typed) else match.group(0)

    out, pattern = "".join(ch for ch in str(text) if ch not in _BIDI_CONTROLS), _bracketed()
    while True:
        again = pattern.sub(drop, out)
        if again == out:
            return out.strip()
        out = again


def _asked(model, span, refused, *extra):
    """What ``model`` wrote for ``span``, unmarked; None when the call failed or was not made, counted in ``refused``, never raised."""
    try:
        return _unmarked(model([dict(t) for t in span], *extra), span)
    except CallRefused as exc:
        import logging

        motive = exc.motive if exc.motive in REFUSAL_MOTIVES else "call_failed"
        logging.getLogger(__name__).info("onion queue: a call was not made (%s); the step goes on without the model",
                                         motive)
        refused.append(motive)
        return None
    except Exception as exc:  # noqa: BLE001 - a call that fails is a summary that never came
        import logging

        # By its class alone: a failure's message may carry the span's words.
        logging.getLogger(__name__).warning("onion queue: a call failed (%s); the step goes on without the model",
                                            type(exc).__name__)
        refused.append("call_failed")
        return None


# Why the repair drops a sentence of the summary, in the order it asks: an
# order, given or told, a claim its span does not hold, a copy of words from
# outside the conversation, a sentence the summary did not end.
DROP_MOTIVES = ("order", "unheld", "copy", "unended")


def _repair(span, probes, text, gate, ladder):
    """The summary's sentences its span holds, then the fewest typed segments answering what they miss.

    Returns ``(text, refs, dropped)``: the sentences kept, the references to
    the user's words, and each sentence dropped as ``(motive, sha256)`` from
    ``DROP_MOTIVES``, never its words. A sentence that copies a unit from
    outside the conversation -- a document, an answer the assistant gave
    with a tool or the web or in sight of anything but the user's words --
    is dropped like one its span does not hold: a summary the gate refused
    would otherwise come back as a peel that keeps those words verbatim, an
    instruction among them. The words of the conversation itself -- typed,
    refined, the assistant's clean answers, a turn written before origins --
    are kept as the first face keeps them. What only words of another origin
    than the user's typing answer stays missed, and the gate judges the
    repair with it missing. Every typed segment holding a run of the user's
    that orders is referenced (``order_ranges``), then the fewest typed
    units answering what the sentences and those segments still miss, each
    by the whole segment it stands in: its condition, its quote, its label
    and its retraction stay with it, as the user typed them.
    """
    from .probes import (
        _ENDED,
        _names_held,
        _order_prose,
        copies,
        holdings,
        order_ranges,
        score,
        sentences,
        units,
    )

    held = holdings(span, gate.directives)
    texts = {str(t.get("turn_id", "")): str(t.get("text", "") or "") for t in span}
    lowered = _lowered_turns(span)
    others = [texts.get(u.turn_id, "")[u.start:u.stop] for u in units(span)
              if u.origin not in _CONVERSATION or u.turn_id in lowered]
    # An order the whole summary gives is dropped wherever it stands: a list
    # item judged alone would lose the label that addressed it to the reader.
    # The clause is compared in the prose it was read in, inline code as
    # words; a sentence of the repair never ends inside a sentence an order
    # is read in. A sentence the summary did not end ("On every later turn,"
    # or a verb alone) is dropped too: no word of the summary's is read with
    # the user's.
    refused = [what for _kind, what, _turn in _directives_in_summary(held, probes, text, gate)]
    places = []
    if gate.directives is not None:
        places = [(run.turn_id, run.start, run.stop)
                  for run in order_ranges(span, gate.directives, _names_held(probes))]
    # A sentence the user's orders, referenced, already show as written is said once: neither kept nor dropped,
    # nothing being lost.
    shown = _shown_sentences(span, references(span, places))
    kept, dropped = [], []
    for sentence in sentences(text):
        if sentence.strip() in shown:
            continue
        if (any(clause in _order_prose(sentence) for clause in refused)
                or _directives_in_summary(held, probes, sentence, gate)):
            motive = "order"
        elif _unheld(held, probes, sentence, gate):
            motive = "unheld"
        elif any(copies(sentence, other, ladder.copy_shared_words) for other in others):
            motive = "copy"
        elif not _ENDED.search(sentence.rstrip()):
            motive = "unended"
        else:
            kept.append(sentence)
            continue
        dropped.append((motive, segment_digest(sentence)))
    found = _typed_units(span)
    missing = score(probes, joined(" ".join(kept), shown_words(span, references(span, places)))[0]).failures
    chosen = _choose(found, _within_reach(missing, found), ladder)
    places += [(found[i].turn_id, found[i].start, found[i].stop) for i in chosen]
    return " ".join(kept), references(span, places), tuple(dropped)


def _shown_sentences(span, refs):
    """The sentences the words ``refs`` show hold, each stripped, as the probes split a text."""
    from .probes import sentences

    return {s.strip() for _turn_id, words in shown_words(span, refs) for s in sentences(words)}


def _said_once(text, span, refs):
    """``text`` without the sentences the words ``refs`` show already hold as written: each said once, nothing lost,
    the model reading it in the user's own words. A sentence goes only when the user typed it whole, never when it
    is a piece of one of theirs; a text that repeats none is returned as it was written."""
    from .probes import sentences

    if not refs or not text:
        return text
    shown = _shown_sentences(span, refs)
    said = sentences(text)
    kept = [s for s in said if s.strip() not in shown]
    return text if len(kept) == len(said) else " ".join(kept)


def _directives_in_summary(held, probes, text, gate):
    """The orders ``text`` gives as a summary: every one, no block of the user's words being honoured in it."""
    from .probes import unbacked_directives

    return unbacked_directives(replace(held, stitchable=()), probes, text, gate.directives)


def _with_orders(span, probes, text, gate):
    """``text`` and the references to each whole typed segment holding a run of the user's that orders.

    The user's orders stand in a peel only so, by reference to what they
    typed (``order_ranges``), and no summary restates them. A sentence of
    ``text`` those segments already show as written is said once
    (``_said_once``): the gate judges what the model will read.
    """
    if gate.directives is None:
        return text, ()
    from .probes import _names_held, order_ranges

    runs = order_ranges(span, gate.directives, _names_held(probes))
    refs = references(span, [(run.turn_id, run.start, run.stop) for run in runs])
    return _said_once(text, span, refs), refs


def _choose(found, targets, ladder):
    """The units of ``found`` kept for ``targets``: each decision's own sentence, then the fewest for the rest.

    A decision is kept in its own words. Another sentence of its turn may
    carry enough of its key to answer it -- the question that led to it, a
    remark on it -- and the fewest units would keep that one in its place:
    so a decision drawn from a sentence of ``found`` takes that sentence,
    and the cover answers only what it leaves. Indices, in order.
    """
    from .probes import answers

    own = set()
    for probe in targets:
        if probe.kind == "decision":
            index = next((i for i, unit in enumerate(found)
                          if unit.turn_id == probe.turn_id and unit.text == probe.answer), None)
            if index is not None:
                own.add(index)
    rest = [probe for probe in targets if not any(answers(probe, found[i].text) for i in own)]
    return tuple(sorted(own | set(cover(found, rest, ladder.exact_cover) or ())))


def _anchors(span, probes, ladder):
    """What a held span keeps: the places in the Cellar of each typed decision, then of the fewest typed units
    answering what else typed words can."""
    found = _typed_units(span)
    chosen = _choose(found, _within_reach(probes, found), ladder)
    return tuple((found[i].turn_id, found[i].start, found[i].stop) for i in chosen)


def _commit(guard, flesh, cellar, ledger, tree, span, *, rung, kind, reason, anchors=(), made=None, refused=()):
    """Evict ``span`` under ``guard`` only while it is still the head of the Flesh, with its receipt and its peel."""
    read = [t.get("turn_id") for t in span]
    decision = made[1] if made is not None else None
    with guard:
        if [t.get("turn_id") for t in flesh.turns()[: len(span)]] != read:
            return Eviction(
                False, "stale: the span left the head of the Flesh while its summary was written; nothing evicted",
                decision=decision, rung="stale", refused=tuple(refused),
            )
        receipt = flesh.evict_span(len(span), cellar, ledger, kind=kind, anchors=anchors)
        peel = None
        if made is not None:
            text, decision, refs, dropped = made
            peel = _make(text, (receipt.key,), 0, (), decision, cellar, rung=rung, refs=refs, dropped=dropped)
            tree.add(peel)
    return Eviction(True, reason, receipt=receipt, peel=peel, decision=decision, rung=rung, refused=tuple(refused),
                    dropped=peel.dropped if peel is not None else ())


def advance(*, flesh, cellar, ledger, tree, gate, ladder, summarize=None, reask=None, refusals=None, lock=None):
    """One step of the queue: the oldest span always leaves, under what best answers for it.

    The rungs, each tried only when the one before it is refused:

    * no probe: a span the probes draw nothing from leaves bare. No model is
      asked and no peel placed: the gate cannot judge it, so nothing may
      speak for it, and it stays recallable in the Cellar.
    * the summary: the librarian's, with a reference to each whole segment
      the user typed that orders (``order_ranges``), judged by both faces of
      the gate. An accepted peel keeps what it failed with it, its
      residual. A call that
      fails -- the model absent, stopped, past its deadline -- is counted as
      ``call_failed`` and the step goes on down the rungs that need no
      model: no call stops the queue.
    * the second asking: a summary the first face refused is asked for once
      more through ``reask``, handed the probes it failed and no other. A
      second refusal is marked in ``refusals`` under the span's key, and a
      span that comes back with the same mark is not asked for again. A
      refusal on the span's probe coverage is not asked for again at all:
      no text can change it.
    * the repair: the sentences its span holds of the better summary, then
      references to the fewest typed segments of the span answering what
      they miss, judged again by both faces, and kept only while all it
      shows is at most ``rho`` times its span; each sentence it drops is
      kept by its motive and digest, and said in what the peel shows.
    * the hold: no peel; the receipt keeps the places in the Cellar of the
      fewest units answering every probe of the span, its anchors.

    With no summariser the queue starts at the repair. The span is read
    under ``lock``, the model is asked without it, and the step commits
    under it only while the span is still the head of the Flesh: otherwise
    nothing leaves and the step says it was stale. No reason given here
    carries a word of the span. A turn marker the model writes is taken out
    of its text before the gate reads it: in a peel, a marker is one a
    reference shows.
    """
    from contextlib import nullcontext
    from functools import partial

    from .probes import generate_probes

    errors = gate.validate() + ladder.validate()
    if errors:
        raise GateError("; ".join(errors))
    guard = nullcontext() if lock is None else lock
    with guard:
        turns = flesh.turns()
    if not turns:
        return Eviction(False, "the Flesh is empty; nothing to evict")
    span = turns[: gate.span_turns]
    commit = partial(_commit, guard, flesh, cellar, ledger, tree, span)
    probes = generate_probes(span, gate.lexicon)
    if not probes:
        return commit(rung="bare", kind="bare", reason="no probe could be drawn from the span: it leaves bare, with no peel")
    text, refused = "", []
    first = None if summarize is None else _asked(summarize, span, refused)
    if first is not None:
        text = first
        ordered, kept_orders = _with_orders(span, probes, text, gate)
        decision = decide(span, probes, ordered, gate, kept_orders)
        if decision.accepted:
            return commit(rung="accepted", kind="accepted", reason=decision.reason,
                          made=(ordered, decision, kept_orders, ()))
        motives = refusal_motives(decision, gate)
        refused.extend(motives)
        failed = decision.result.failures
        if reask is not None and failed and "coverage" not in motives:
            from .receipts import span_key

            key, mark = span_key(span), refusal_fingerprint(failed, gate, getattr(reask, "identity", ""))
            if refusals is None or refusals.get(key) != mark:
                again = _asked(reask, span, refused, [(p.kind, p.answer, p.turn_id) for p in failed])
                if again is not None:
                    ordered, kept_orders = _with_orders(span, probes, again, gate)
                    judged = decide(span, probes, ordered, gate, kept_orders)
                    if judged.accepted:
                        return commit(rung="reasked", kind="accepted", made=(ordered, judged, kept_orders, ()),
                                      refused=refused,
                                      reason="accepted on the second asking, handed the probes the first one failed")
                    refused.extend(refusal_motives(judged, gate))
                    if refusals is not None:
                        with guard:
                            refusals[key] = mark
                    if len(judged.result.failures) + len(judged.unsupported) < len(failed) + len(decision.unsupported):
                        text = again
    repaired, refs, dropped = _repair(span, probes, text, gate, ladder)
    repaired = _said_once(repaired, span, refs) if refs else repaired
    if repaired or refs:
        judged = decide(span, probes, repaired, gate, refs)
        shown = joined(repaired, shown_words(span, refs))[0]
        if judged.accepted and _saves_enough(estimate_tokens(shown), _span_tokens(span), ladder.rho):
            return commit(rung="repaired", kind="accepted", made=(repaired, judged, refs, dropped), refused=refused,
                          reason=f"repaired with {len(refs)} reference(s) to the user's words")
    return commit(rung="held", kind="held", anchors=_anchors(span, probes, ladder), refused=refused,
                  reason="held: no peel answers for the span within the compression floor; its anchors stay")


def select_anchors(ledger, cellar, query, cap, estimate=None, dropped=None, unplaced=None, *, lineage=None,
                   withdrawn=(), reached=None):
    """The anchors of open held receipts for ``query``, their words read from the Cellar, whole, within ``cap``.

    An anchor ranks as a peel does, by the share of the query's terms its
    words hold, then the newest receipt first, then its place in its span.
    It shows what a peel would of its unit, re-read from its place: a
    sentence as written, a code block by its marker, never its code; with
    the label of its turn (``turn_label``), and defanged as a reference is,
    its turn's marker saying how many markers that took when it took any. A
    place that is no typed unit of its span -- a turn a peer sent included
    -- shows nothing. ``dropped``, a list, receives the provenance of each
    anchor the query reached that ``cap`` left out, ``unplaced`` of each it
    reached whose place reads as no typed unit any more, and ``reached`` of
    each it reached whose turn a withdrawal reached, not shown.
    """
    from .composer import defanged

    estimate = estimate or estimate_tokens
    withdrawn = frozenset(withdrawn or ())
    terms = _terms(query)
    if not terms or cap <= 0:
        return []
    scored = []
    held = [r for r in ledger.open() if r.kind == "held" and r.anchors]
    for age, receipt in enumerate(reversed(held)):
        span = cellar.get(receipt.key)
        texts = {str(t.get("turn_id", "")): str(t.get("text", "")) for t in span}
        labels = {str(t.get("turn_id", "")): turn_label(t, lineage) for t in span}
        typed = {(u.turn_id, u.start, u.stop): u.text for u in _typed_units(span)}
        for place, (turn_id, start, stop) in enumerate(receipt.anchors):
            shown = typed.get((turn_id, start, stop))
            hit = len(terms & _terms(texts.get(turn_id, "")[start:stop])) / len(terms)
            if hit <= 0:
                continue
            provenance = f"anchor:{receipt.key[:12]}:{turn_id}"
            if not shown:
                if unplaced is not None:
                    unplaced.append(provenance)
                continue
            if withheld(labels[turn_id], withdrawn):
                if reached is not None:
                    reached.append(provenance)
                continue
            words, altered = defanged(shown)
            if altered:
                words = f"{_header(turn_id, altered)} {words}"
            events = (("anchors_altered", altered),) if altered else ()
            scored.append((-hit, age, place, words, provenance, tuple(labels[turn_id]), events))
    scored.sort(key=lambda entry: entry[:3])
    chosen, used = [], 0
    for neg, _age, _place, words, provenance, label, events in scored:
        tokens = estimate(words)
        if used + tokens > cap:
            if dropped is not None:
                dropped.append(provenance)
            continue
        used += tokens
        chosen.append(Selected(text=words, provenance=provenance, score=round(-neg, 4), label=label, events=events))
    return chosen


def fidelity(tree, cellar, lexicon=None, probe_recall=None):
    """Probe pass rate of every peel resampled against its Cellar spans. A fixture reading.

    The probes are drawn with ``lexicon``, the empty one when none is given,
    and the reading names it with the version of the generator. A peel is
    read as it shows: its summary and the words its references read. Beside
    the rate stands ``probe_recall``, a gate's stated probe recall, when it
    holds for these probes; otherwise None, and the note says why.
    """
    from .probes import EMPTY_LEXICON, GENERATOR_VERSION, generate_probes, score

    lexicon = EMPTY_LEXICON if lexicon is None else lexicon
    passed = failed = 0
    for peel in tree.all():
        turns = [t for k in peel.sources for t in cellar.get(k)]
        result = score(generate_probes(turns, lexicon), _holds(peel, turns))
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


def _holds(peel, turns):
    """What a peel holds as it shows: its summary, then the words its references read in ``turns``."""
    return joined(peel.text, shown_words(turns, peel.refs))[0] if peel.refs else peel.text


def context_multiplier(tree, cellar, estimate=None):
    """Source tokens the root peels stand for, over the tokens they cost as they show. A fixture reading."""
    estimate = estimate or estimate_tokens
    roots = tree.roots()
    source_tokens = sum(estimate(str(t.get("text", ""))) for p in roots for k in p.sources for t in cellar.get(k))
    peel_tokens = sum(estimate(_holds(p, [t for k in p.sources for t in cellar.get(k)])) for p in roots)
    return {
        "source": "fixture",
        "roots": len(roots),
        "source_tokens": source_tokens,
        "peel_tokens": peel_tokens,
        "multiplier": None if peel_tokens == 0 else round(source_tokens / peel_tokens, 4),
    }
