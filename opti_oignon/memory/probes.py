#!/usr/bin/env python3
"""Recall probes: grounded questions a compressed memory must still answer.

A summary may replace verbatim text only once it has been shown to answer for
it. The instrument is a set of probes drawn from the source span -- the
entities it names, the numbers and dates it states, the decisions it records
-- each carrying the turn it came from, and a scorer that says whether a
candidate text answers them. Deterministic, standard library only, and free
of any model: the librarian may later add richer probes on the host, but the
gate itself has to be provable here.

Two rules of the silent-zero family are load-bearing. A generator that finds
nothing on a rich span is a defect, and the contracts hold a rich fixture
against it. A scorer with no probes to score reports an unknown rate, never
0.0: an absence of questions is not a total failure.

Nothing on the chat path imports this module. It is measurement, not
pipeline, and a contract on the tree says so.
"""

import re
from dataclasses import dataclass, field

checkpoint_before_apply = True

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"(?<![\w-])\d+(?:[.,]\d+)?(?![\w-])")
_WORD = re.compile(r"[a-z0-9]+")
_CAPITALISED = re.compile(r"\b[A-Z][a-zA-Z]+\b")
_NEGATION = re.compile(r"\b(not|never|no|cannot)\b|n't\b", re.IGNORECASE)

# Markers that make a sentence a recorded decision.
_DECISION_MARKERS = ("decided", "agreed", "will ", "must", "chose", "plan to", "shall")

# Capitalised words that are not names of anything.
_NOT_ENTITIES = frozenset({
    "the", "a", "an", "we", "i", "you", "he", "she", "it", "they", "this",
    "that", "these", "those", "our", "your", "their", "my", "in", "on", "at",
    "and", "but", "or", "if", "when", "then", "there", "here", "yes", "no",
})

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "we", "i",
    "you", "he", "she", "it", "they", "this", "that", "our", "your", "their",
    "not", "never", "no", "cannot", "do", "does", "did", "don", "doesn", "t",
})

# The share of a decision's content words a sentence must carry to count as
# the same decision. Below it, the decision is absent, not merely reworded.
DECISION_COVERAGE = 0.7


def _tokens(text):
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class Probe:
    """One grounded question, with the turn that grounds it."""

    kind: str
    question: str
    answer: str
    turn_id: str
    key: frozenset = field(default_factory=frozenset)
    negated: bool = False
    # How many negation tokens the decision sentence carries. A boolean was
    # not enough: "will use Docker because the venue has no runtime" still
    # reads as negated, and the inversion of "will not use Docker" passed.
    negations: int = 0


@dataclass
class ProbeResult:
    """The outcome of scoring a candidate text against a probe set."""

    passed: int
    failed: int
    failures: list

    @property
    def rate(self):
        """Fraction answered, or None when there was nothing to answer.

        None and 0.0 are different answers: 0.0 means every question was
        asked and none was answered; None means no question existed.
        """
        total = self.passed + self.failed
        return None if total == 0 else self.passed / total


def _negations(sentence):
    return len(_NEGATION.findall(sentence))


def _sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _is_decision(sentence):
    lowered = sentence.lower()
    return any(marker in lowered for marker in _DECISION_MARKERS)


def _decision_key(sentence):
    return frozenset(t for t in _tokens(sentence) if t not in _STOPWORDS)


def generate_probes(span):
    """Probes drawn from a span of turns, each a mapping with turn_id and text.

    Every probe carries the turn it was drawn from, so a failing probe names
    where the lost information lived. Order is deterministic: turn order,
    then sentence order, then kind.
    """
    probes = []
    for turn in span:
        turn_id = str(turn.get("turn_id", ""))
        text = str(turn.get("text", "") or "")
        for sentence in _sentences(text):
            seen = set()
            for date in _DATE.findall(sentence):
                if ("date", date) not in seen:
                    seen.add(("date", date))
                    probes.append(Probe("date", f"Which date is stated in {turn_id}?", date, turn_id))
            without_dates = _DATE.sub(" ", sentence)
            for number in _NUMBER.findall(without_dates):
                if ("number", number) not in seen:
                    seen.add(("number", number))
                    probes.append(Probe("number", f"Which number is stated in {turn_id}?", number, turn_id))
            for name in _CAPITALISED.findall(sentence):
                if name.lower() in _NOT_ENTITIES or ("entity", name) in seen:
                    continue
                seen.add(("entity", name))
                probes.append(Probe("entity", f"Who or what is named in {turn_id}?", name, turn_id))
            if _is_decision(sentence):
                key = _decision_key(sentence)
                if key:
                    probes.append(Probe(
                        "decision",
                        f"What was decided in {turn_id}?",
                        sentence,
                        turn_id,
                        key=key,
                        negated=bool(_NEGATION.search(sentence)),
                        negations=_negations(sentence),
                    ))
    return probes


def answers(probe, text):
    """True when the candidate text answers the probe.

    Entities, numbers and dates are answered by presence of the exact token.
    A decision is answered by a sentence carrying enough of its content words
    with the same polarity: the same words with the opposite polarity is the
    decision inverted, and that is a failure, not a match. Polarity is the
    count of negation tokens, not their presence: one negation dropped from
    a sentence that carried two is an inversion too.
    """
    if probe.kind == "decision":
        for sentence in _sentences(text):
            words = set(_tokens(sentence))
            coverage = len(probe.key & words) / len(probe.key)
            if coverage >= DECISION_COVERAGE:
                if _negations(sentence) == probe.negations:
                    return True
        return False
    if probe.kind in ("date", "number"):
        pattern = r"(?<![\w-])" + re.escape(probe.answer) + r"(?![\w-])"
        return re.search(pattern, text) is not None
    return probe.answer.lower() in set(_tokens(text))


def score(probes, text):
    """Score a candidate text against a probe set."""
    failures = [p for p in probes if not answers(p, text)]
    return ProbeResult(passed=len(probes) - len(failures), failed=len(failures), failures=failures)
