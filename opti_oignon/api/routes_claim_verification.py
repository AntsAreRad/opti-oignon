#!/usr/bin/env python3
"""FastAPI claim-verification route (the verification role's wiring lot):
expose the claim-vs-source verification role over HTTP.

The verification role (``opti_oignon.agent.claim_verification``):
a caller-driven surface that wraps a model-generated claim and its cited source
as untrusted data under one policy header and returns a fail-secure verdict
(supported / unsupported / uncertain, defaulting to uncertain). The route / UI
wiring was deferred there. This module is that wire: a single per-user ``POST``
that runs one verification over a submitted (claim, source) pair and returns the
structured result for the caller to show and act on. Registered on the app
exactly like ``note_actions_router``, the precedent.

Design notes:

- Not a model-reachable tool. Like the N.3 note-actions route, this surface is
  caller-driven (a UI action, or a later agent step submits a claim and its
  source), not tool-called; it defines no tool schema and registers nothing in
  the agent tool registry. It is a thin wrapper: it interprets nothing, issues
  no direct database query, and delegates the wrapping, the verdict taxonomy,
  the fail-secure mapping, and the result shape to ``claim_verification``.
- Untrusted wrapping is the role's. Both the claim and the cited source are
  wrapped as untrusted data by ``claim_verification.build_messages`` (via
  ``agent.untrusted_context``): the verification instruction is the only trusted
  (system-role) message, both pieces ride the user role inside the
  untrusted-data markers, so injection-looking text in either piece cannot steer
  the model. This route never places the claim or source in a system-role
  message.
- Model client. The one-shot inference seam is built from the user's selected
  model the way ``routes_note_actions`` builds its one-shot client: a one-shot
  TEXT completion (non-streaming) that ``claim_verification._invoke_once``
  coerces directly. The builder is a FastAPI dependency seam so tests inject a
  fake client through ``app.dependency_overrides`` and ollama is never invoked
  in-container.
- No mode gate (CV-D4). The verification role reaches no network and has no mode
  gate: it runs identically in Daily and Bulbe. So this route carries no mode
  dependency seam and builds the verifier with no mode provider, exactly as
  ``routes_memory`` carries no mode gate even though ``manage_memory`` is a
  Bulbe-forbidden tool. The user's own caller-driven surface, with no egress, is
  not route-mode-gated.
- The verifier never raises; its structured ``ClaimVerificationResult`` crosses
  the wire as ``ClaimVerificationResultSchema`` (ok or a clean fail-secure
  failure, all carried in the body). The one HTTP error code is the availability
  guard (503), mirroring ``routes_note_actions._check``.

``checkpoint_before_apply`` is hardcoded True and never overridable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Hardcoded checkpoint discipline for every new module; never overridable.
checkpoint_before_apply = True

try:
    from opti_oignon.agent.claim_verification import make_claim_verifier

    FEATURE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    FEATURE_AVAILABLE = False
    make_claim_verifier = None  # type: ignore[assignment]

try:
    from .routes_auth import _get_current_user

    _cv_auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional

    _cv_auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


class ClaimVerificationRequest(BaseModel):
    """A claim and its cited source to verify, plus the user's selected model.

    The model is optional: an absent model yields a clean fail-secure result (the
    builder returns None), rather than guessing a model.
    """

    claim: str
    source: str
    model: str | None = None


class ClaimVerificationResultSchema(BaseModel):
    """The structured verification result crossing the wire.

    Mirrors ``claim_verification.ClaimVerificationResult.to_dict()``: the mapped
    verdict (supported / unsupported / uncertain), ok, an optional reason on a
    fail-secure failure, and the model's raw text on success.
    """

    verdict: str
    ok: bool
    reason: str = ""
    raw_text: str = ""


claim_verification_router = APIRouter(
    prefix="/api/claims", tags=["claims"], dependencies=_cv_auth_dep
)


def _check() -> None:
    if not FEATURE_AVAILABLE or make_claim_verifier is None:
        raise HTTPException(
            status_code=503, detail="Claim verification surface not available"
        )


class _OneShotOllamaClient:
    """A one-shot TEXT completion over the built messages, for ``_invoke_once``.

    A plain callable that runs a non-streaming Ollama chat and returns the reply
    text, which is exactly what ``claim_verification._invoke_once`` coerces. The
    ``ollama`` import is lazy so this module loads without it; resolution failure
    surfaces as the runner's clean fail-secure failure.
    """

    def __init__(self, model: str, *, host: str | None = None) -> None:
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


def _resolve_one_shot_client(model: str | None) -> Any:
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


def _client_builder_dep() -> Callable[[str | None], Any]:
    """The one-shot client builder seam (a model -> client callable).

    A FastAPI dependency so tests inject a fake builder through
    ``app.dependency_overrides`` without touching ollama; the live builder wires
    the user's selected model.
    """
    return _resolve_one_shot_client


@claim_verification_router.post(
    "/verify", response_model=ClaimVerificationResultSchema
)
def run_claim_verification(
    request: ClaimVerificationRequest,
    build_client: Callable[[str | None], Any] = Depends(_client_builder_dep),
    current_user: dict = Depends(_get_current_user),
) -> ClaimVerificationResultSchema:
    """Verify one claim against its cited source.

    Per-user via the auth dependency. The model client is built from the user's
    selected model; the claim and source are wrapped as untrusted context by
    ``claim_verification``; there is deliberately no mode gate (CV-D4: the role
    reaches no network and runs identically in Daily and Bulbe). The verifier
    never raises; its structured fail-secure result crosses the wire.
    """
    _check()
    client = build_client(request.model or None)
    verifier = make_claim_verifier(model_client=client)
    result = verifier(request.claim, request.source)
    return ClaimVerificationResultSchema(**result.to_dict())
