#!/usr/bin/env python3
"""FastAPI citation-verify route (composing the S273 citation extractor with the
S271 per-answer aggregation): expose the whole producer -> aggregation pipeline
over HTTP.

The producer half landed at S273 (``opti_oignon.agent.citation_extraction``): a
pure-stdlib parser whose ``extract_pairs(answer, sources)`` turns a produced
answer carrying inline numeric citation markers, plus the ordered sources those
markers index, into the (claim, source) pairs the aggregation consumes. The
aggregator landed at S271 (``opti_oignon.agent.claim_aggregation``): its
``make_answer_verifier`` builds a ``verify_answer(pairs)`` that runs each pair
through the S267 verification role and aggregates the per-pair verdicts
fail-secure into a single per-answer verdict. The S272 route
(``routes_answer_verification``) exposed the aggregation over HTTP, but it takes
a pre-built batch of (claim, source) pairs: a caller still had to construct the
pairs by hand. This module is the join: a single per-user ``POST`` that runs
``extract_pairs`` over a submitted answer plus its ordered sources and hands the
result to ``verify_answer``, so a caller submits a raw answer and gets the
verdict directly. Registered on the app exactly like
``answer_verification_router``, the S272 precedent.

It mirrors the S268 / S272 route idiom precisely; the differences are the payload
(a raw answer plus its ordered sources, not a pre-built batch) and the result
(the aggregate, the per-pair list, and the extracted (claim, source) pairs for
transparency, the extracted pairs positionally aligned with the per-pair
results). The router is a DISTINCT object from the S268 ``claim_verification_router``
and the S272 ``answer_verification_router`` and shares the ``/api/claims`` prefix
with a new path, so neither of those routes' surfaces is touched.

Design notes:

- Not a model-reachable tool. Like the S268 / S272 routes and the N.3
  note-actions route, this surface is caller-driven (a UI, a producing path, or a
  later agent step submits a raw answer and its sources), not tool-called; it
  defines no tool schema and registers nothing in the agent tool registry. It is
  a thin composition wrapper: it interprets nothing, issues no direct database
  query, and delegates the parsing to ``citation_extraction`` and the wrapping,
  the verdict taxonomy, the fail-secure mapping, the aggregation, and the result
  shape to ``claim_aggregation`` (which composes ``claim_verification``).
- Untrusted wrapping is the role's. Each extracted pair's claim and cited source
  are wrapped as untrusted data under one policy header by the verification role
  (via ``agent.untrusted_context``): the verification instruction is the only
  trusted (system-role) message, both pieces ride the user role inside the
  untrusted-data markers, so injection-looking text in the answer or a source
  cannot steer the model. This route never places a claim or source in a
  system-role message.
- Model client. The one-shot inference seam is built from the user's selected
  model the way ``routes_answer_verification`` builds its one-shot client: a
  one-shot TEXT completion (non-streaming) that the role's ``_invoke_once``
  coerces directly. The builder is a FastAPI dependency seam so tests inject a
  fake client through ``app.dependency_overrides`` and ollama is never invoked
  in-container.
- No mode gate (CV-D4). The verification surface reaches no network and has no
  mode gate: it runs identically in Daily and Bulbe. So this route carries no
  mode dependency seam and builds the verifier with no mode provider, exactly as
  the S268 / S272 routes.
- Fail-closed end to end. The extractor never raises and omits any pair it cannot
  resolve (out-of-range, ``[0]``, malformed, or empty-source citations); the
  aggregation never raises and defaults to uncertain on an empty pair list. So an
  answer with no citations, an empty answer, an empty sources list, or an
  unavailable model all yield a clean fail-secure aggregate (uncertain, not ok),
  never an error. The one HTTP error code is the availability guard (503),
  mirroring ``routes_answer_verification._check``.

``checkpoint_before_apply`` is hardcoded True and never overridable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Hardcoded checkpoint discipline for every new module; never overridable.
checkpoint_before_apply = True

try:
    from opti_oignon.agent.citation_extraction import extract_pairs
    from opti_oignon.agent.claim_aggregation import make_answer_verifier

    FEATURE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    FEATURE_AVAILABLE = False
    extract_pairs = None  # type: ignore[assignment]
    make_answer_verifier = None  # type: ignore[assignment]

try:
    from .routes_auth import _get_current_user

    _cv_auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional

    _cv_auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


class ClaimSourcePair(BaseModel):
    """One cited claim and the source it is checked against.

    On the wire these are both the extracted-pairs view (what the parser derived
    from the answer) and the natural per-pair record: each pair is verified
    independently and the per-answer verdict is the fail-secure aggregate.
    """

    claim: str
    source: str


class CitationVerificationRequest(BaseModel):
    """A produced answer plus its ordered sources, and the user's model.

    ``answer`` carries inline numeric citation markers ``[n]`` (1-based) that
    index ``sources`` by position. The model is optional: an absent model yields
    a clean fail-secure result (the builder returns None), rather than guessing a
    model. An answer with no citations, an empty answer, or an empty ``sources``
    list extracts no pairs and is a clean fail-secure failure (the aggregate
    defaults to uncertain, ok False), not an error.
    """

    answer: str
    sources: List[str]
    model: Optional[str] = None


class ClaimVerificationResultSchema(BaseModel):
    """One per-pair verification result crossing the wire.

    Mirrors ``claim_verification.ClaimVerificationResult.to_dict()``: the mapped
    verdict (supported / unsupported / uncertain), ok, an optional reason on a
    fail-secure failure, and the model's raw text on success.
    """

    verdict: str
    ok: bool
    reason: str = ""
    raw_text: str = ""


class CitationVerificationResultSchema(BaseModel):
    """The structured per-answer aggregate crossing the wire.

    Mirrors ``claim_aggregation.AnswerVerificationResult.to_dict()`` plus the
    extracted-pairs view: the aggregate verdict, ok (True only when at least one
    pair was extracted and every pair verified cleanly), an optional reason on a
    not-ok aggregate, the per-pair results, and the (claim, source) pairs the
    parser extracted from the answer, positionally aligned with ``results``.
    """

    verdict: str
    ok: bool
    reason: str = ""
    results: List[ClaimVerificationResultSchema] = []
    pairs: List[ClaimSourcePair] = []


citation_verification_router = APIRouter(
    prefix="/api/claims", tags=["claims"], dependencies=_cv_auth_dep
)


def _check() -> None:
    if not FEATURE_AVAILABLE or make_answer_verifier is None or extract_pairs is None:
        raise HTTPException(
            status_code=503, detail="Citation verification surface not available"
        )


class _OneShotOllamaClient:
    """A one-shot TEXT completion over the built messages, for ``_invoke_once``.

    A plain callable that runs a non-streaming Ollama chat and returns the reply
    text, which is exactly what the verification role's ``_invoke_once`` coerces.
    The ``ollama`` import is lazy so this module loads without it; resolution
    failure surfaces as the runner's clean fail-secure failure.
    """

    def __init__(self, model: str, *, host: Optional[str] = None) -> None:
        self._model = model
        self._host = host

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        import ollama

        client = ollama.Client(host=self._host) if self._host else ollama.Client()
        resp = client.chat(model=self._model, messages=messages, stream=False)
        try:
            return str(resp["message"]["content"])
        except (KeyError, TypeError, IndexError):
            msg = getattr(resp, "message", None)
            return str(getattr(msg, "content", "") if msg is not None else "")


def _resolve_one_shot_client(model: Optional[str]) -> Any:
    """Build a one-shot model client from the selected model, or None.

    None when no model is selected (so the runner reports a clean fail-secure
    failure rather than guessing a model), otherwise a one-shot client over the
    user's chosen model.
    """
    if not model:
        return None
    try:
        return _OneShotOllamaClient(model)
    except Exception:  # pragma: no cover - defensive
        return None


def _client_builder_dep() -> Callable[[Optional[str]], Any]:
    """The one-shot client builder seam (a model -> client callable).

    A FastAPI dependency so tests inject a fake builder through
    ``app.dependency_overrides`` without touching ollama; the live builder wires
    the user's selected model.
    """
    return _resolve_one_shot_client


@citation_verification_router.post(
    "/verify-citations", response_model=CitationVerificationResultSchema
)
def run_citation_verification(
    request: CitationVerificationRequest,
    build_client: Callable[[Optional[str]], Any] = Depends(_client_builder_dep),
    current_user: dict = Depends(_get_current_user),
) -> CitationVerificationResultSchema:
    """Extract the cited claims from one answer and verify them, fail-secure.

    Per-user via the auth dependency. The (claim, source) pairs are parsed from
    the answer and its ordered sources by ``citation_extraction.extract_pairs``
    (fail-closed by omission, never raising); the model client is built from the
    user's selected model; each pair's claim and source are wrapped as untrusted
    context by the verification role; there is deliberately no mode gate (CV-D4:
    the surface reaches no network and runs identically in Daily and Bulbe). The
    aggregation never raises; its structured fail-secure aggregate, the per-pair
    results, and the extracted pairs (aligned with the results) cross the wire.
    """
    _check()
    client = build_client(request.model or None)
    verify_answer = make_answer_verifier(model_client=client)
    pairs = extract_pairs(request.answer, request.sources)
    result = verify_answer(pairs)
    data = result.to_dict()
    return CitationVerificationResultSchema(
        verdict=data["verdict"],
        ok=data["ok"],
        reason=data["reason"],
        results=data["results"],
        pairs=[ClaimSourcePair(claim=claim, source=source) for claim, source in pairs],
    )
