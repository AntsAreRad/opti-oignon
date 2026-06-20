#!/usr/bin/env python3
"""FastAPI per-answer verification route (the S271 aggregation module's wiring
lot): expose the per-answer claim aggregation over HTTP.

The answer-aggregation module landed at S271
(``opti_oignon.agent.claim_aggregation``): a standalone, caller-driven surface
that takes a list of (claim, source) pairs, runs each through the S267
verification role, and aggregates the per-pair verdicts fail-secure into a single
per-answer verdict. The route / UI exposure was deferred there. This module is
that wire: a single per-user ``POST`` that runs one ``verify_answer`` over a
submitted batch of (claim, source) pairs and returns the structured aggregate for
the caller to show and act on. Registered on the app exactly like
``claim_verification_router``, the S268 precedent.

It mirrors the S268 single-pair route precisely; the only differences are the
payload (a batch of pairs, not one pair) and the result (the aggregate plus the
per-pair list). The router is a DISTINCT object from the S268
``claim_verification_router`` and shares the ``/api/claims`` prefix with a new
path, so the single-pair route's surface is untouched.

Design notes:

- Not a model-reachable tool. Like the S268 route and the N.3 note-actions route,
  this surface is caller-driven (a UI, a citation-extraction step, or a later
  agent step submits a batch of pairs), not tool-called; it defines no tool
  schema and registers nothing in the agent tool registry. It is a thin wrapper:
  it interprets nothing, issues no direct database query, and delegates the
  wrapping, the verdict taxonomy, the fail-secure mapping, the aggregation, and
  the result shape to ``claim_aggregation`` (which composes ``claim_verification``).
- Untrusted wrapping is the role's. Each pair's claim and cited source are
  wrapped as untrusted data under one policy header by the verification role (via
  ``agent.untrusted_context``): the verification instruction is the only trusted
  (system-role) message, both pieces ride the user role inside the untrusted-data
  markers, so injection-looking text in either piece cannot steer the model. This
  route never places a claim or source in a system-role message.
- Model client. The one-shot inference seam is built from the user's selected
  model the way ``routes_claim_verification`` builds its one-shot client: a
  one-shot TEXT completion (non-streaming) that the role's ``_invoke_once``
  coerces directly. The builder is a FastAPI dependency seam so tests inject a
  fake client through ``app.dependency_overrides`` and ollama is never invoked
  in-container.
- No mode gate (CV-D4). The verification surface reaches no network and has no
  mode gate: it runs identically in Daily and Bulbe. So this route carries no
  mode dependency seam and builds the verifier with no mode provider, exactly as
  the S268 route. The user's own caller-driven surface, with no egress, is not
  route-mode-gated.
- The aggregation never raises; its structured ``AnswerVerificationResult``
  crosses the wire as ``AnswerVerificationResultSchema`` (the aggregate verdict,
  ok or a clean fail-secure failure, and the per-pair list, all carried in the
  body). The one HTTP error code is the availability guard (503), mirroring
  ``routes_claim_verification._check``.

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
    from opti_oignon.agent.claim_aggregation import make_answer_verifier

    FEATURE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    FEATURE_AVAILABLE = False
    make_answer_verifier = None  # type: ignore[assignment]

try:
    from .routes_auth import _get_current_user

    _av_auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional

    _av_auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


class ClaimSourcePair(BaseModel):
    """One cited claim and the source it is checked against.

    A batch of these is the kind of input a citation-extraction step would hand
    in: each pair is verified independently and the per-answer verdict is the
    fail-secure aggregate.
    """

    claim: str
    source: str


class AnswerVerificationRequest(BaseModel):
    """A batch of (claim, source) pairs to verify, plus the user's model.

    The model is optional: an absent model yields a clean fail-secure result (the
    builder returns None), rather than guessing a model. An empty ``pairs`` list
    is a clean fail-secure failure (the aggregate defaults to uncertain, ok
    False), not an error.
    """

    pairs: List[ClaimSourcePair]
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


class AnswerVerificationResultSchema(BaseModel):
    """The structured per-answer aggregate crossing the wire.

    Mirrors ``claim_aggregation.AnswerVerificationResult.to_dict()``: the
    aggregate verdict, ok (True only when at least one pair was supplied and every
    pair verified cleanly), an optional reason on a not-ok aggregate, and the
    per-pair results.
    """

    verdict: str
    ok: bool
    reason: str = ""
    results: List[ClaimVerificationResultSchema] = []


answer_verification_router = APIRouter(
    prefix="/api/claims", tags=["claims"], dependencies=_av_auth_dep
)


def _check() -> None:
    if not FEATURE_AVAILABLE or make_answer_verifier is None:
        raise HTTPException(
            status_code=503, detail="Answer verification surface not available"
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


@answer_verification_router.post(
    "/verify-answer", response_model=AnswerVerificationResultSchema
)
def run_answer_verification(
    request: AnswerVerificationRequest,
    build_client: Callable[[Optional[str]], Any] = Depends(_client_builder_dep),
    current_user: dict = Depends(_get_current_user),
) -> AnswerVerificationResultSchema:
    """Verify every (claim, source) pair in one answer and aggregate fail-secure.

    Per-user via the auth dependency. The model client is built from the user's
    selected model; each pair's claim and source are wrapped as untrusted context
    by the verification role; there is deliberately no mode gate (CV-D4: the
    surface reaches no network and runs identically in Daily and Bulbe). The
    aggregation never raises; its structured fail-secure aggregate (and the
    per-pair results) crosses the wire.
    """
    _check()
    client = build_client(request.model or None)
    verify_answer = make_answer_verifier(model_client=client)
    pairs = [(p.claim, p.source) for p in request.pairs]
    result = verify_answer(pairs)
    return AnswerVerificationResultSchema(**result.to_dict())
