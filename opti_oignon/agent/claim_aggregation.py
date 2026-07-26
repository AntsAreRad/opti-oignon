#!/usr/bin/env python3
"""Per-answer claim aggregation over the claim-vs-source verification role.

The higher-value continuation of the verification arc (the role, the route,
the UI, the nav): a standalone module that gates a whole
answer's cited claims through the verification role. It takes a list of
(claim, source) pairs -- the kind a citation-extraction step would hand it --
runs each one through the role's injected one-shot seam, and aggregates the
per-pair verdicts fail-secure into a single per-answer verdict. It is local,
deterministic in its plumbing, and routes inference only through the role's
injected seam, so it is 100% local / Python / Ollama with no backend coupling at
module load.

Design notes:

- Composition, not reimplementation. This module builds on the role
  (``opti_oignon.agent.claim_verification``) rather than duplicating the verdict
  taxonomy, the anti-injection wrapping, or the fail-secure mapping. Each
  (claim, source) pair is verified by the role's ``verify``, so the claim and
  the cited source are each wrapped as untrusted data under one policy header by
  the role; this module places no untrusted text in a
  trusted message and interprets nothing.
- Not a model-reachable tool. Like the role and the N.3 note-actions surface,
  this is a caller-driven surface: a citation-extraction step (or a UI) hands in
  the pairs, never the model's tool calling. It defines no tool schema and
  registers nothing in the agent tool registry, so it grows no schema-count or
  allowlist pin. The activation -- extracting the pairs from a live produced
  answer and acting on the aggregate verdict -- is a later lot; this module is
  the aggregation logic that lot will call.
- Fail-secure aggregation, mirroring the role's asymmetry. The per-answer
  verdict is UNSUPPORTED if any pair is unsupported; otherwise UNCERTAIN if any
  pair is uncertain (which also captures any pair the role could not verify,
  since those are held at uncertain); SUPPORTED only when every pair is
  supported; an empty pair list or any unknown verdict defaults to UNCERTAIN,
  never to SUPPORTED. An answer whose every cited claim is unverifiable never
  asserts support, exactly as a single indeterminate verification never asserts
  support.
- Clean ``ok`` is conservative. The aggregate ``ok`` is True only when at least
  one pair was supplied and every pair verified cleanly (the role's own ``ok``
  on each). A single empty-source pair, an unavailable model, or a raising seam
  marks the aggregate not-ok while the verdict stays fail-secure; nothing here
  raises.
- No egress, no mode gate. Verification reads only the supplied sources plus the
  local model and reaches no network, so the aggregation runs identically in
  Daily and Bulbe with no mode resolution and no mode provider on the factory,
  the role's posture carried up one level.
- Dependency injection. The one-shot inference seam is the role's: a callable
  over the built messages, or an object exposing ``stream``, injected by the
  caller. An un-injected verifier reports a clean per-pair failure rather than
  guessing a model, so the aggregate degrades cleanly. Nothing imports the
  backend at module load, so the surface is exercised directly by pytest with no
  fastapi / ollama chain.

``checkpoint_before_apply`` is hardcoded True and never overridable;
``FEATURE_AVAILABLE`` gates graceful degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .claim_verification import (
    VERDICT_SUPPORTED,
    VERDICT_UNCERTAIN,
    VERDICT_UNSUPPORTED,
    ClaimVerificationResult,
    make_claim_verifier,
)

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation is
# applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True


@dataclass
class AnswerVerificationResult:
    """The outcome of verifying every cited claim in one answer.

    ``verdict`` is the aggregate (supported / unsupported / uncertain). ``ok`` is
    True only when at least one pair was supplied and every pair verified cleanly
    (each pair's own ``ok``). ``results`` is the per-pair list of the role's
    :class:`ClaimVerificationResult`. ``reason`` carries a brief summary on a
    not-ok aggregate. The verifier never raises.
    """

    verdict: str
    ok: bool
    results: list[ClaimVerificationResult] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "ok": self.ok,
            "reason": self.reason,
            "results": [r.to_dict() for r in self.results],
        }


def aggregate_verdicts(verdicts: Iterable[str]) -> str:
    """Aggregate per-pair verdicts into a per-answer verdict, fail-secure.

    Mirrors the role's asymmetry: unsupported dominates; otherwise uncertain if
    any pair is uncertain; supported only when every pair is supported. An empty
    sequence, or any verdict outside the taxonomy with no unsupported present,
    defaults to uncertain -- support is never asserted on absence or ambiguity.
    """
    vs = [str(v) for v in verdicts]
    if not vs:
        return VERDICT_UNCERTAIN
    if any(v == VERDICT_UNSUPPORTED for v in vs):
        return VERDICT_UNSUPPORTED
    if any(v == VERDICT_UNCERTAIN for v in vs):
        return VERDICT_UNCERTAIN
    if all(v == VERDICT_SUPPORTED for v in vs):
        return VERDICT_SUPPORTED
    # No unsupported and not all supported means an unknown verdict slipped in;
    # never promote to supported on an unrecognised value.
    return VERDICT_UNCERTAIN


def _coerce_pair(pair: Any) -> tuple[str, str]:
    """Coerce one item to a (claim, source) string pair, fail-secure to empties.

    A malformed item (not indexable, too short) yields ("", "") so the role
    refuses that pair cleanly rather than the aggregation raising.
    """
    try:
        claim = "" if pair[0] is None else str(pair[0])
        source = "" if pair[1] is None else str(pair[1])
    except (TypeError, IndexError, KeyError):
        return "", ""
    return claim, source


def _summarize_failures(results: Sequence[ClaimVerificationResult]) -> str:
    """A brief, deterministic reason for a not-ok aggregate."""
    total = len(results)
    if total == 0:
        return "No claim/source pairs to verify."
    failed = sum(1 for r in results if not r.ok)
    if failed:
        return str(failed) + " of " + str(total) + " pair(s) could not be verified."
    return ""


def make_answer_verifier(
    model_client: Any = None,
) -> Callable[[Sequence[Any]], AnswerVerificationResult]:
    """Build a per-answer verifier, injecting the role's one-shot inference seam.

    ``model_client`` is the same seam the role takes (a callable over the
    built messages, or an object with ``stream``); when None the role's default
    resolver is used (which yields a clean per-pair failure unless wired). There
    is deliberately no mode provider: the aggregation reaches no network and runs
    the same in Daily and Bulbe.

    The returned ``verify_answer(pairs)`` runs each (claim, source) pair through
    the role's verify (so each pair is wrapped as untrusted data by the role),
    aggregates the verdicts fail-secure, marks ``ok`` only when at least one pair
    was supplied and every pair verified cleanly, and returns an
    :class:`AnswerVerificationResult`. It never raises.
    """

    verify = make_claim_verifier(model_client)

    def verify_answer(pairs: Sequence[Any]) -> AnswerVerificationResult:
        results: list[ClaimVerificationResult] = []
        for pair in (pairs or []):
            claim, source = _coerce_pair(pair)
            results.append(verify(claim, source))
        verdict = aggregate_verdicts(r.verdict for r in results)
        ok = bool(results) and all(r.ok for r in results)
        reason = "" if ok else _summarize_failures(results)
        return AnswerVerificationResult(
            verdict=verdict, ok=ok, results=results, reason=reason
        )

    return verify_answer
