#!/usr/bin/env python3
"""The drift ledger: facts with provenance that are superseded, never edited.

A memory fact today carries a sentence, a category, a free-form source and a
use count. It cannot say which turn it came from, how sure anyone was, when
it stopped holding, or what replaced it. The ledger record can. Its one rule
is that a fact is never changed in place: a new fact supersedes it, the
original stays with a link to its successor, and a reader who follows the
chain arrives at the fact that currently holds. Drift then stops being an
impression and becomes a count.

The deterministic half of contradiction detection lives here too: a
negation of a held fact, a different value for an attribute a subject can
only have one of, a number or a date that disagrees for the same subject.
The templates are narrow on purpose. Liking a city and living in it are not
exclusive; a preference with a qualifier is not the same claim. What the
templates cannot decide is left to the model judge on the host, never
guessed here. An injection harness proves the templates capable: known
contradictions go in and must come back by name.

This module holds the schema and its semantics in memory. Moving the physical
table to it is the last session of the block. Nothing on the chat path
imports this module; a contract on the tree says so.
"""

import re
from dataclasses import dataclass, field, replace

checkpoint_before_apply = True

KINDS = frozenset({"fact", "preference", "correction", "project"})

# Predicates a subject can hold only one value of at a time.
_EXCLUSIVE_PREDICATES = frozenset({"live in", "work at", "be on", "be born on", "be called"})

_LEMMA = {
    "lives": "live", "live": "live", "works": "work", "work": "work",
    "is": "be", "are": "be", "was": "be", "were": "be",
    "prefers": "prefer", "prefer": "prefer", "likes": "like", "like": "like",
}

_STATEMENT = re.compile(
    r"^\s*(?P<subject>.+?)\s+"
    r"(?P<neg>does not |do not |did not |is not |are not |was not |never )?"
    r"(?P<verb>lives?|works?|is|are|was|were|prefers?|likes?)"
    r"(?:\s+(?P<particle>in|at|on|called|born on))?"
    r"\s+(?P<object>.+?)\s*\.?\s*$",
    re.IGNORECASE,
)
_VALUE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d+(?:[.,]\d+)?\b")


@dataclass(frozen=True)
class LedgerFact:
    """One statement, where it came from, how sure, and what replaced it."""

    id: str
    statement: str
    kind: str
    provenance: list = field(default_factory=list)
    confidence: float = 1.0
    valid_from: str = ""
    valid_until: str = ""
    superseded_by: str = None

    @property
    def status(self):
        return "superseded" if self.superseded_by else "active"


def validate_fact(fact):
    """Every reason the record cannot be held, or an empty list."""
    errors = []
    if not str(fact.id or "").strip():
        errors.append("id: empty")
    if not str(fact.statement or "").strip():
        errors.append("statement: empty")
    if fact.kind not in KINDS:
        errors.append(f"kind: unknown {fact.kind!r}; expected one of {', '.join(sorted(KINDS))}")
    if not fact.provenance:
        errors.append("provenance: a fact with no turn behind it cannot be held")
    try:
        confidence = float(fact.confidence)
    except (TypeError, ValueError):
        confidence = None
    if confidence is None or not 0.0 <= confidence <= 1.0:
        errors.append(f"confidence: {fact.confidence!r} is not within [0, 1]")
    if fact.valid_from and fact.valid_until and fact.valid_until < fact.valid_from:
        errors.append(
            f"valid window ends before it starts: {fact.valid_until} < {fact.valid_from}"
        )
    return errors


class DriftLedger:
    """An in-memory ledger with supersession and no edit."""

    def __init__(self):
        self._facts = {}

    def add(self, fact):
        errors = validate_fact(fact)
        if errors:
            raise ValueError("; ".join(errors))
        if fact.id in self._facts:
            raise ValueError(f"fact {fact.id!r} is already held; supersede it, never overwrite it")
        self._facts[fact.id] = fact
        return fact.id

    def supersede(self, old_id, new_fact):
        """Hold ``new_fact`` and link the old one to it. The old text is untouched."""
        old = self._facts.get(old_id)
        if old is None:
            raise KeyError(old_id)
        if old.superseded_by:
            raise ValueError(f"fact {old_id!r} is already superseded by {old.superseded_by!r}")
        new_id = self.add(new_fact)
        # A new record carrying the link, so the object handed in is never mutated.
        self._facts[old_id] = replace(old, superseded_by=new_id)
        return new_id

    def get(self, fact_id):
        return self._facts[fact_id]

    def head(self, fact_id):
        """Follow supersession from ``fact_id`` to the fact that currently holds."""
        fact = self._facts[fact_id]
        seen = {fact_id}
        while fact.superseded_by:
            if fact.superseded_by in seen:
                raise ValueError(f"supersession cycle at {fact.superseded_by!r}")
            seen.add(fact.superseded_by)
            fact = self._facts[fact.superseded_by]
        return fact

    def active(self):
        return [f for f in self._facts.values() if f.status == "active"]

    def all(self):
        return list(self._facts.values())


@dataclass(frozen=True)
class Contradiction:
    """A held fact a new statement cannot be true alongside."""

    fact_id: str
    template: str
    detail: str


@dataclass(frozen=True)
class _Claim:
    subject: str
    predicate: str
    obj: str
    negated: bool
    value: str


def _parse(statement):
    m = _STATEMENT.match(statement or "")
    if not m:
        return None
    verb = _LEMMA.get(m.group("verb").lower(), m.group("verb").lower())
    particle = (m.group("particle") or "").lower()
    predicate = f"{verb} {particle}".strip()
    obj = re.sub(r"\s+", " ", m.group("object").strip().rstrip(".")).lower()
    value = _VALUE.search(obj)
    return _Claim(
        subject=re.sub(r"\s+", " ", m.group("subject").strip()).lower(),
        predicate=predicate,
        obj=obj,
        negated=bool(m.group("neg")),
        value=value.group(0) if value else "",
    )


def _statement_of(fact):
    return fact.statement if hasattr(fact, "statement") else fact[1]


def _id_of(fact):
    return fact.id if hasattr(fact, "id") else fact[0]


def find_contradictions(facts, statement):
    """Held facts the statement contradicts, by template.

    ``facts`` is a sequence of ledger facts or of ``(id, statement)`` pairs.
    A statement that does not fit the templates contradicts nothing here --
    that is the model judge's territory, and its absence is reported as an
    empty list, not as agreement.
    """
    claim = _parse(statement)
    if claim is None:
        return []
    found = []
    for fact in facts:
        held = _parse(_statement_of(fact))
        if held is None or held.subject != claim.subject or held.predicate != claim.predicate:
            continue
        fact_id = _id_of(fact)
        if held.obj == claim.obj:
            if held.negated != claim.negated:
                found.append(Contradiction(fact_id, "negation", f"{statement!r} negates {_statement_of(fact)!r}"))
            continue
        if held.negated or claim.negated:
            continue
        if held.value and claim.value and held.value != claim.value:
            found.append(Contradiction(fact_id, "value", f"{claim.value} disagrees with {held.value}"))
        elif held.predicate in _EXCLUSIVE_PREDICATES:
            found.append(Contradiction(fact_id, "exclusive", f"{claim.obj!r} and {held.obj!r} cannot both hold"))
    return found


def _shift_value(value):
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        year, month, day = value.split("-")
        month = int(month) % 12 + 1
        return f"{year}-{month:02d}-{day}"
    return str(int(float(value.replace(",", "."))) + 100)


def inject_contradictions(facts, count):
    """Statements that contradict the first ``count`` eligible held facts.

    Returns ``(statements, expected_fact_ids)`` in the same order. A fact
    that carries a value gets a shifted value; one with an exclusive
    attribute gets another value for it; anything else is negated. Each
    injection is detectable by the templates, which is what makes this a
    proof of capability rather than a fixture.
    """
    statements, expected = [], []
    for fact in facts:
        if len(statements) >= max(0, int(count)):
            break
        text = _statement_of(fact)
        claim = _parse(text)
        if claim is None or claim.negated:
            continue
        m = _STATEMENT.match(text)
        head = text[: m.start("object")]
        if claim.value:
            injected = head + text[m.start("object"):].replace(claim.value, _shift_value(claim.value), 1)
        elif claim.predicate in _EXCLUSIVE_PREDICATES:
            injected = head + "Elsewhere"
        else:
            verb = m.group("verb")
            base = _LEMMA.get(verb.lower(), verb.lower())
            negated = "is not" if base == "be" else f"does not {base}"
            injected = text[: m.start("verb")] + negated + text[m.end("verb"):]
        statements.append(injected)
        expected.append(_id_of(fact))
    return statements, expected
