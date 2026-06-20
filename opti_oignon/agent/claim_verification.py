#!/usr/bin/env python3
"""Claim-vs-source verification role (the gated verification surface).

A first implementation lot for the logged DEBT_LOT_S261 roadmap item: a role
that checks a model-generated claim against its cited source and returns a
fail-secure verdict. It is local, deterministic in its plumbing, and routes its
single inference call through an injected one-shot seam, so it is 100% local /
Python / Ollama with no backend coupling at module load.

Design notes:

- This is NOT a model-reachable tool. Unlike the N.4 ``manage_notes`` tool,
  this surface is driven by a caller handing in a (claim, source) pair, not by
  the model's tool calling. It defines no ``ToolSchema`` and registers nothing
  in the agent tool registry, so it grows no schema-count or allowlist pin.
- Anti-injection. Both the claim (model-generated, untrusted) and the cited
  source (external, untrusted) are wrapped as untrusted data under one policy
  header via :func:`opti_oignon.agent.untrusted_context.untrusted_message_many`
  (the S175 / Odysseus core). The verification instruction is the only trusted
  message; both pieces ride the user role inside untrusted-data markers, so
  injection-looking text in either piece cannot steer the model.
- Fail-secure verdict. The taxonomy is supported / unsupported / uncertain. The
  mapping of free-text model output is asymmetric on purpose: an unparseable or
  ambiguous reply defaults to UNCERTAIN, never to SUPPORTED, and only an
  explicit unsupported signal moves a lead-ambiguous reply off uncertain. A
  verification role that rubber-stamped "supported" on ambiguity would be
  dangerous; an indeterminate verification never asserts support.
- No egress, no mode gate. Verification reads only the supplied source plus the
  local model and reaches no network, so the role runs identically in Daily and
  Bulbe with no mode resolution and no web action (unlike the N.3
  fact-check-with-web action). There is deliberately no mode provider.
- Dependency injection. The model client is a one-shot inference seam the caller
  injects (a callable over the built messages, or an object exposing
  ``stream``). An un-injected verifier reports a clean failure rather than
  guessing a model, exactly the N.3 posture. A later route / UI lot wires the
  client from the user's selected model.
- The inference seam is injectable for tests; nothing here imports the backend
  at module load, so the surface is exercised directly by pytest with no
  fastapi / ollama chain (the S243 lesson).

``checkpoint_before_apply`` is hardcoded True and never overridable;
``FEATURE_AVAILABLE`` gates graceful degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .untrusted_context import untrusted_message_many

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation is
# applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True

# The source labels the two untrusted pieces are wrapped under (sanitised by
# untrusted_context to safe tag attributes).
SOURCE_CLAIM = "claim"
SOURCE_SOURCE = "source"

# The verdict taxonomy.
VERDICT_SUPPORTED = "supported"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_UNCERTAIN = "uncertain"

ALL_VERDICTS: frozenset[str] = frozenset(
    {VERDICT_SUPPORTED, VERDICT_UNSUPPORTED, VERDICT_UNCERTAIN}
)

# The trusted verification instruction. It names the job, the source-only rule,
# and asks the model to lead with the verdict word so the mapping is robust.
_VERIFY_INSTRUCTION = (
    "You are a verification role. The untrusted-data block below contains a "
    "claim labelled 'claim' and a cited source labelled 'source'. Decide "
    "whether the claim is supported by the source, using only the source and "
    "no outside knowledge. Begin your answer with exactly one word: SUPPORTED, "
    "UNSUPPORTED, or UNCERTAIN, then give a brief reason grounded in the "
    "source. Answer UNCERTAIN when the source does not settle the claim either "
    "way; do not guess."
)

# Verdict markers, scanned fail-secure. Unsupported and uncertain are checked
# before supported because "unsupported" contains "supported" as a substring;
# the lead is authoritative, with a whole-text unsupported override but never a
# whole-text supported promotion.
_UNSUPPORTED_MARKERS = (
    "unsupported",
    "not supported",
    "not support",
    "no support",
    "contradict",
    "refute",
)
_UNCERTAIN_MARKERS = (
    "uncertain",
    "unclear",
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "insufficient",
    "not enough",
    "no information",
    "not addressed",
    "does not address",
    "doesn't address",
    "not sure",
    "unsure",
    "ambiguous",
    "maybe",
)
_SUPPORTED_MARKERS = (
    "supported",
    "support the claim",
    "supports the claim",
    "confirm",
    "consistent with",
    "corroborat",
)


@dataclass
class ClaimVerificationResult:
    """The outcome of a claim-vs-source verification.

    ``ok`` True carries the mapped ``verdict`` and the model's ``raw_text``; any
    failure is ``ok`` False with a ``reason``, the ``verdict`` held at the
    fail-secure ``uncertain``. The verifier never raises.
    """

    verdict: str
    ok: bool
    reason: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "ok": self.ok,
            "reason": self.reason,
            "raw_text": self.raw_text,
        }


def _has(markers: tuple[str, ...], hay: str) -> bool:
    return any(m in hay for m in markers)


def normalize_verdict(text: Any) -> str:
    """Map free-text model output to a verdict, fail-secure to uncertain.

    The lead (first line) is authoritative; unsupported and uncertain are tested
    before supported so the "unsupported" substring is never read as support.
    When the lead carries no verdict word, only an explicit whole-text
    unsupported signal moves off uncertain -- an ambiguous reply is never
    promoted to supported.
    """
    if text is None or not str(text).strip():
        return VERDICT_UNCERTAIN
    low = str(text).strip().lower()
    head = low.split("\n", 1)[0]
    if _has(_UNSUPPORTED_MARKERS, head):
        return VERDICT_UNSUPPORTED
    if _has(_UNCERTAIN_MARKERS, head):
        return VERDICT_UNCERTAIN
    if _has(_SUPPORTED_MARKERS, head):
        return VERDICT_SUPPORTED
    if _has(_UNSUPPORTED_MARKERS, low):
        return VERDICT_UNSUPPORTED
    return VERDICT_UNCERTAIN


def build_messages(claim: str, source: str) -> list[dict[str, str]]:
    """Build the one-shot [system, user] messages for a verification.

    The system message is the trusted verification instruction; the user message
    wraps the claim and the cited source as untrusted data under one policy
    header (the anti-injection core). Raises ``ValueError`` on an empty claim or
    empty source -- the runner guards both before building.
    """
    c = "" if claim is None else str(claim)
    s = "" if source is None else str(source)
    if not c.strip():
        raise ValueError("Empty claim: nothing to verify.")
    if not s.strip():
        raise ValueError("Empty source: nothing to verify the claim against.")
    user_msg = untrusted_message_many(
        [(SOURCE_CLAIM, c), (SOURCE_SOURCE, s)]
    )
    if user_msg is None:  # pragma: no cover - guarded by the strip checks above
        raise ValueError("No untrusted content to wrap.")
    return [{"role": "system", "content": _VERIFY_INSTRUCTION}, user_msg]


def _default_model_client() -> Any:
    """No process-default model client: the caller injects a one-shot client.

    Returns None so an un-injected verifier reports a clean failure rather than
    guessing a model. A later route / UI lot wires the client from the user's
    selected model, the same dependency-injection posture as the N.3 surface.
    """
    return None


def _invoke_once(model_client: Any, messages: list[dict[str, str]]) -> str:
    """Invoke the one-shot inference seam and coerce its output to text.

    Mirrors the N.3 tolerance: ``model_client`` may expose ``stream`` or be a
    plain callable taking the messages. The return may be a string or an
    iterable of chunks (strings, ``{"content": ...}`` dicts, or objects with
    ``content``).
    """
    fn = getattr(model_client, "stream", None)
    if fn is None and callable(model_client):
        fn = model_client
    if fn is None:
        raise TypeError("model client is not callable and has no stream method")
    out = fn(messages)
    if isinstance(out, str):
        return out
    parts: list[str] = []
    for chunk in out:
        if isinstance(chunk, str):
            parts.append(chunk)
        elif isinstance(chunk, dict):
            parts.append(str(chunk.get("content", "")))
        else:
            parts.append(str(getattr(chunk, "content", "")))
    return "".join(parts)


def make_claim_verifier(
    model_client: Any = None,
) -> Callable[[str, str], ClaimVerificationResult]:
    """Build a claim-vs-source verifier, injecting the one-shot inference seam.

    ``model_client`` is the inference seam (a callable over the built messages,
    or an object with ``stream``); when None the default resolver is used (which
    returns None unless wired by a caller, yielding a clean failure). There is
    deliberately no mode provider: the role reaches no network and runs the same
    in Daily and Bulbe.

    The returned ``verify(claim, source)`` refuses an empty claim or source with
    a structured fail-secure result, wraps both as untrusted data, invokes the
    model once, maps the output to a verdict (fail-secure to uncertain), and
    returns a :class:`ClaimVerificationResult`. It never raises.
    """

    def verify(claim: str, source: str) -> ClaimVerificationResult:
        c = "" if claim is None else str(claim)
        s = "" if source is None else str(source)
        if not c.strip():
            return ClaimVerificationResult(
                verdict=VERDICT_UNCERTAIN,
                ok=False,
                reason="Empty claim: nothing to verify.",
            )
        if not s.strip():
            return ClaimVerificationResult(
                verdict=VERDICT_UNCERTAIN,
                ok=False,
                reason="Empty source: nothing to verify the claim against.",
            )
        client = model_client if model_client is not None else _default_model_client()
        if client is None:
            return ClaimVerificationResult(
                verdict=VERDICT_UNCERTAIN,
                ok=False,
                reason="Model client unavailable.",
            )
        try:
            messages = build_messages(c, s)
            text = _invoke_once(client, messages)
        except Exception as exc:
            return ClaimVerificationResult(
                verdict=VERDICT_UNCERTAIN,
                ok=False,
                reason="Verification failed: " + str(exc),
            )
        verdict = normalize_verdict(text)
        return ClaimVerificationResult(verdict=verdict, ok=True, raw_text=text)

    return verify
