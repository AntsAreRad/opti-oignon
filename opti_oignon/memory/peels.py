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
empty probe set is refused, not passed: an unknown rate is not a pass.

Selection at query time is deterministic and keyword-based. The vector
layer for peels is a host decision (which store, which embedder, measured
against the existing one) and nothing here claims it. The summariser is a
seam: the librarian's model on the host, a recording fake in the
contracts. The two metrics at the end are fixture readings and say so in
their ``source`` field. Nothing on the chat path imports this module yet,
and a contract on the tree says so.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

checkpoint_before_apply = True

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "onion.yaml"
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "what", "which",
    "who", "how", "did", "does", "do", "happened", "about",
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


def source_digest(cellar, sources):
    """The digest of the spans behind ``sources``, read from the Cellar now."""
    return hashlib.sha256(_canonical([cellar.get(k) for k in sources]).encode("utf-8")).hexdigest()


def peel_id(text, sources):
    return hashlib.sha256(json.dumps([text, list(sources)], ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Gate:
    decision_threshold: float
    episodic_threshold: float
    span_turns: int

    def validate(self):
        errors = []
        for name in ("decision_threshold", "episodic_threshold"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"{name}: {value!r} is not within [0, 1]")
        if not isinstance(self.span_turns, int) or self.span_turns < 1:
            errors.append(f"span_turns: {self.span_turns!r} is not a positive integer")
        return errors


def load_gate(path=None):
    """The gate of ``onion.yaml``, refused if out of range."""
    import yaml

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
    return gate


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
    """Score ``text`` against ``probes`` and decide, class by class."""
    from .probes import score

    errors = gate.validate()
    if errors:
        raise GateError("; ".join(errors))
    result = score(probes, text)
    if not probes:
        return GateDecision(False, "no probe could be drawn from the span: an unknown rate is not a pass", result)
    decision = [p for p in probes if p.kind == "decision"]
    episodic = [p for p in probes if p.kind != "decision"]
    decision_rate = _rate(decision, result.failures)
    episodic_rate = _rate(episodic, result.failures)
    short = []
    if decision_rate is not None and decision_rate < gate.decision_threshold:
        short.append(f"decision probes at {decision_rate} against {gate.decision_threshold}")
    if episodic_rate is not None and episodic_rate < gate.episodic_threshold:
        short.append(f"episodic probes at {episodic_rate} against {gate.episodic_threshold}")
    if short:
        failed = ", ".join(f"{p.kind}:{p.answer[:24]}@{p.turn_id}" for p in result.failures)
        return GateDecision(False, "; ".join(short) + f" -- failed: {failed}", result, decision_rate, episodic_rate)
    return GateDecision(True, "every probe class present meets its threshold", result, decision_rate, episodic_rate)


def _summarise(sources, cellar, summarize, gate):
    from .probes import generate_probes

    spans = [cellar.get(k) for k in sources]
    turns = [t for span in spans for t in span]
    probes = generate_probes(turns)
    text = str(summarize(turns))
    return spans, probes, text, judge(probes, text, gate)


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


def _terms(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


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
    """Peels for the query, best first, whole items, never an ancestor with its descendant."""
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
    probes = generate_probes(span)
    text = str(summarize([dict(t) for t in span]))
    decision = judge(probes, text, gate)
    if not decision.accepted:
        return Eviction(False, decision.reason, decision=decision)
    receipt = flesh.evict_span(len(span), cellar, ledger)
    peel = _make(text, (receipt.key,), 0, (), decision, cellar)
    tree.add(peel)
    return Eviction(True, decision.reason, receipt=receipt, peel=peel, decision=decision)


def fidelity(tree, cellar):
    """Probe pass rate of every peel resampled against its Cellar spans. A fixture reading."""
    from .probes import generate_probes, score

    passed = failed = 0
    for peel in tree.all():
        turns = [t for k in peel.sources for t in cellar.get(k)]
        result = score(generate_probes(turns), peel.text)
        passed += result.passed
        failed += result.failed
    total = passed + failed
    return {
        "source": "fixture",
        "peels": len(tree.all()),
        "probes": total,
        "passed": passed,
        "failed": failed,
        "rate": None if total == 0 else round(passed / total, 4),
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
