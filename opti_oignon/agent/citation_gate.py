#!/usr/bin/env python3
"""Citation gate: the additive wiring that a producing path will call to verify
a produced answer's cited claims.

The verification arc is complete through its parts: the role S267
(``opti_oignon.agent.claim_verification``) checks one (claim, source) pair and
returns a fail-secure verdict; the aggregation S271
(``opti_oignon.agent.claim_aggregation``) folds a list of (claim, source) pairs
through the role into one per-answer verdict; the extractor S273
(``opti_oignon.agent.citation_extraction``) parses a produced answer plus its
ordered sources into exactly those pairs; the route S274
(``opti_oignon.api.routes_citation_verification``) exposes the whole pipeline
over HTTP for a UI. Both the S271 and S273 docstrings record that the final
piece -- calling ``verify_answer`` on the extracted pairs inside a live
producing path -- is "a later lot". This module is the staging for that lot: it
composes the extractor and the aggregation into a single caller-driven seam so
the eventual ``loop.py`` (or ``routes_chat.py``) call is a one-liner that
imports and calls it. No producing-path edit lands with this module; the
producing path stays untouched until a separate lot wires the one-liner behind
the activation flag.

Design notes:

- Composition, not reimplementation. This module adds no parsing, no verdict
  taxonomy, no anti-injection wrapping, and no fail-secure mapping of its own.
  It calls ``citation_extraction.extract_pairs`` for the pairs and
  ``claim_aggregation.make_answer_verifier`` for the per-answer verifier, which
  in turn composes the S267 role. Each extracted claim and its cited source are
  wrapped as untrusted data under one policy header by the role; this module
  places no untrusted text in a trusted message and interprets nothing.
- Not a model-reachable tool. Like the role, the aggregation, and the
  extractor, this is a caller-driven surface: a producing path (or a UI, or a
  later agent step) hands in an answer and its sources, never the model's tool
  calling. It defines no tool schema and registers nothing in the agent tool
  registry, so it grows no schema-count or allowlist pin.
- No egress, no mode gate. The wiring reaches no network -- only the extractor
  (pure stdlib) and the role's injected inference seam -- so it runs identically
  in Daily and Bulbe with no mode gate, the verification arc's posture carried
  up to the wiring.
- Fail-secure end to end. The extractor never raises and omits any pair it
  cannot resolve; the aggregation never raises and defaults to uncertain on an
  empty pair list. So an answer with no citations, an empty answer, an empty
  sources list, an out-of-range marker, or an unavailable model all yield a
  clean fail-secure aggregate (uncertain, not ok), never an exception.
- Dependency injection. The inference seam is the role's, threaded through the
  aggregation's ``make_answer_verifier``: a callable over the built messages, or
  an object exposing ``stream``, injected by the caller. An un-injected client
  yields a clean per-pair failure rather than guessing a model, so the gate
  degrades cleanly. ``run_gate`` additionally takes the built ``verify_answer``
  directly, so the composition can be exercised by pytest with an injected seam
  and no fastapi / ollama chain (the S243 lesson).
- Activation behind a flag. ``GATE_ANSWERS_ON_PRODUCE`` is False by default. The
  gate always computes the verdict (a harmless read), but the blocking decision
  ``should_block_answer`` honours the flag: it returns False while the flag is
  off, so a producing path that imports the wiring stays inert until the flag is
  flipped in a later lot. Even when enabled the decision is conservative: it
  blocks only on a positively-refuted ``unsupported`` verdict, never on the
  fail-secure ``uncertain`` (an answer whose citations are merely unverifiable
  is not blocked).

``checkpoint_before_apply`` is hardcoded True and never overridable;
``FEATURE_AVAILABLE`` gates graceful degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence, Tuple

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation is
# applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

# The activation flag for gating a produced answer on its citation verdict. Off
# by default, so importing this wiring into a producing path is inert until a
# later lot flips it. The gate always computes the verdict; only the blocking
# decision honours this flag.
GATE_ANSWERS_ON_PRODUCE = False

try:
    from opti_oignon.agent.citation_extraction import extract_pairs
    from opti_oignon.agent.claim_aggregation import make_answer_verifier
    from opti_oignon.agent.claim_verification import (
        VERDICT_SUPPORTED,
        VERDICT_UNSUPPORTED,
        VERDICT_UNCERTAIN,
    )

    FEATURE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    FEATURE_AVAILABLE = False
    extract_pairs = None  # type: ignore[assignment]
    make_answer_verifier = None  # type: ignore[assignment]
    VERDICT_SUPPORTED = "supported"
    VERDICT_UNSUPPORTED = "unsupported"
    VERDICT_UNCERTAIN = "uncertain"


@dataclass
class CitationGateResult:
    """The outcome of gating one produced answer on its cited claims.

    ``verdict`` is the per-answer aggregate (supported / unsupported /
    uncertain). ``ok`` is the aggregation's own ``ok`` (True only when at least
    one pair was extracted and every pair verified cleanly). ``reason`` carries a
    brief summary on a not-ok aggregate. ``results`` is the per-pair list of the
    role's results. ``pairs`` is the (claim, source) pairs the extractor derived
    from the answer, positionally aligned with ``results`` (the S274 result
    shape). The gate never raises.
    """

    verdict: str
    ok: bool
    reason: str = ""
    results: list = field(default_factory=list)
    pairs: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "ok": self.ok,
            "reason": self.reason,
            "results": [
                r.to_dict() if hasattr(r, "to_dict") else r for r in self.results
            ],
            "pairs": [
                {"claim": claim, "source": source} for claim, source in self.pairs
            ],
        }


def _empty_result(reason: str) -> CitationGateResult:
    """A clean fail-secure aggregate when nothing could be verified."""
    return CitationGateResult(
        verdict=VERDICT_UNCERTAIN, ok=False, reason=reason, results=[], pairs=[]
    )


def run_gate(
    answer: Any,
    sources: Sequence[Any],
    verify_answer: Callable[[Sequence[Any]], Any],
) -> CitationGateResult:
    """Extract the cited claims from ``answer`` and apply ``verify_answer``.

    The pure composition: ``extract_pairs(answer, sources)`` derives the
    (claim, source) pairs (fail-closed by omission, never raising), the supplied
    ``verify_answer`` aggregates them (never raising), and the aggregate is
    surfaced together with the extracted pairs. ``verify_answer`` is taken
    directly so a test can drive the composition with an injected seam and no
    model. Never raises.
    """
    if extract_pairs is None:
        return _empty_result("Citation extraction surface not available.")
    pairs = extract_pairs(answer, sources)
    result = verify_answer(pairs)
    return CitationGateResult(
        verdict=str(getattr(result, "verdict", VERDICT_UNCERTAIN)),
        ok=bool(getattr(result, "ok", False)),
        reason=str(getattr(result, "reason", "") or ""),
        results=list(getattr(result, "results", []) or []),
        pairs=list(pairs),
    )


def make_citation_gate(
    model_client: Any = None,
) -> Callable[[Any, Sequence[Any]], CitationGateResult]:
    """Build a gate over a produced answer and its ordered sources.

    Wires the real per-answer verifier via the aggregation's
    ``make_answer_verifier`` (injecting the role's one-shot inference seam) and
    returns ``gate(answer, sources)``. There is deliberately no mode gate: the
    wiring reaches no network and runs the same in Daily and Bulbe. The returned
    gate never raises.
    """
    if not FEATURE_AVAILABLE or make_answer_verifier is None:
        def _unavailable(answer: Any, sources: Sequence[Any]) -> CitationGateResult:
            return _empty_result("Citation verification surface not available.")

        return _unavailable

    verify_answer = make_answer_verifier(model_client=model_client)

    def gate(answer: Any, sources: Sequence[Any]) -> CitationGateResult:
        return run_gate(answer, sources, verify_answer)

    return gate


def verify_answer_citations(
    answer: Any,
    sources: Sequence[Any],
    model_client: Any = None,
) -> CitationGateResult:
    """Extract and verify one produced answer's cited claims, fail-secure.

    The one-liner a producing path calls: it builds the gate over the user's
    selected model client and runs it on the answer and its ordered sources,
    returning the per-answer aggregate plus the extracted pairs. Never raises.
    """
    return make_citation_gate(model_client)(answer, sources)


def should_block_answer(result: Any, *, enabled: Any = None) -> bool:
    """Decide whether a produced answer should be blocked on its verdict.

    Behind the ``GATE_ANSWERS_ON_PRODUCE`` flag (overridable per-call via
    ``enabled`` for a producing path that gates conditionally): returns False
    while gating is off, so importing the wiring is inert. When gating is on the
    decision is conservative -- it blocks only on a positively-refuted
    ``unsupported`` verdict, never on the fail-secure ``uncertain`` (an answer
    whose cited claims are merely unverifiable is not blocked). Never raises.
    """
    active = GATE_ANSWERS_ON_PRODUCE if enabled is None else bool(enabled)
    if not active:
        return False
    return getattr(result, "verdict", None) == VERDICT_UNSUPPORTED
