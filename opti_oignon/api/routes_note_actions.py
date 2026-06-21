#!/usr/bin/env python3
"""FastAPI note-actions route (N.3 backend half): expose the S246 selection-action
runner over HTTP.

The agent-side selection-action surface landed at S246
(``opti_oignon.agent.note_actions``): from a note selection the user asks one of
five local actions -- fact-check, develop, summarize, rewrite, make-checklist --
or the Daily-only fact-check-with-web. This module is the HTTP surface the
SvelteKit notes UI (N.2 proper) calls: a single per-user ``POST`` that runs one
selection action and returns the structured result for the UI to show alongside
the note and insert. Registered on the app exactly like ``notes_router``.

Design notes:

- Not a model-reachable tool. N.3 is UI-driven (the user selects text and picks
  an action), not tool-called; this route defines no ``ToolSchema`` and registers
  nothing in the agent tool registry. It is a thin wrapper: it interprets
  nothing, issues no SQL, and delegates the wrapping, the action-to-prompt
  mapping, the mode gate, and the result shape to ``note_actions``.
- Untrusted wrapping is the surface's. The selected text is wrapped as untrusted
  data by ``note_actions.build_messages`` (via ``agent.untrusted_context``): the
  action's instruction is the only trusted (system-role) message, the selection
  rides the user role inside the untrusted-data markers, so injection-looking
  note text cannot steer the model. This route never places the selection in a
  system-role message.
- Model client. The one-shot inference seam is built from the user's selected
  model the way ``api/routes_agent.py`` builds the agent loop's client, except
  it is a one-shot TEXT completion (non-streaming) so ``note_actions._invoke_once``
  coerces it directly, rather than the loop's ``{"message": {"content"}}`` stream
  shape. The builder is a FastAPI dependency seam so tests inject a fake client
  through ``app.dependency_overrides`` and ollama is never invoked in-container.
- Daily-only web gate. The web action needs egress and is Daily-only. The gate is
  enforced at the route by injecting the live security mode (fail-secure to
  Bulbe) into the runner's ``mode_provider``: ``note_actions`` then returns a
  structured refusal (``refused=True``) for a web action outside Daily, never a
  silent local downgrade, and this route returns that refusal verbatim. The mode
  is a dependency seam so tests drive Daily / Bulbe directly.
- The runner never raises; its structured ``NoteActionResult`` crosses the wire
  as ``NoteActionResultSchema`` (ok / refused / a clean failure all carried in
  the body). The one HTTP error code is the availability guard (503), mirroring
  ``routes_notes._check_store``.

``checkpoint_before_apply`` is hardcoded True and never overridable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from .schemas import NoteActionRequest, NoteActionResultSchema

logger = logging.getLogger(__name__)

# Hardcoded checkpoint discipline for every new module; never overridable.
checkpoint_before_apply = True

try:
    from opti_oignon.agent.note_actions import make_note_action_runner

    FEATURE_AVAILABLE = True
except Exception:  # pragma: no cover - constrained environments
    FEATURE_AVAILABLE = False
    make_note_action_runner = None  # type: ignore[assignment]

try:
    from .routes_auth import _get_current_user

    _na_auth_dep = [Depends(_get_current_user)]
except ImportError:  # pragma: no cover - auth optional

    _na_auth_dep = []

    def _get_current_user() -> dict:  # type: ignore[misc]
        return {"sub": None}


note_actions_router = APIRouter(
    prefix="/api/notes/actions", tags=["notes"], dependencies=_na_auth_dep
)


def _check() -> None:
    if not FEATURE_AVAILABLE or make_note_action_runner is None:
        raise HTTPException(
            status_code=503, detail="Note actions surface not available"
        )


class _OneShotOllamaClient:
    """A one-shot TEXT completion over the built messages, for ``_invoke_once``.

    Unlike the loop's streaming ``_OllamaModelClient`` (which yields
    ``{"message": {"content", "tool_calls"}}`` chunks), this is a plain callable
    that runs a non-streaming Ollama chat and returns the reply text, which is
    exactly what ``note_actions._invoke_once`` coerces. The ``ollama`` import is
    lazy so this module loads without it; resolution failure surfaces as the
    runner's clean failure.
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

    Mirrors ``routes_agent._resolve_model_client``: None when no model is
    selected (so the runner reports a clean failure rather than guessing a
    model), otherwise a one-shot client over the user's chosen model.
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
    the user's selected model the ``routes_agent`` way.
    """
    return _resolve_one_shot_client


def _live_mode() -> str:
    """The live security mode, fail-secure to Bulbe when undeterminable."""
    try:
        from opti_oignon.security_mode import get_current_mode

        return str(get_current_mode() or "").strip().lower() or "bulbe"
    except Exception:  # pragma: no cover - defensive guard
        return "bulbe"


def _mode_dep() -> str:
    """The mode seam for the Daily-only web gate (overridable in tests)."""
    return _live_mode()


@note_actions_router.post("/run", response_model=NoteActionResultSchema)
def run_note_action(
    request: NoteActionRequest,
    build_client: Callable[[str | None], Any] = Depends(_client_builder_dep),
    mode: str = Depends(_mode_dep),
    current_user: dict = Depends(_get_current_user),
) -> NoteActionResultSchema:
    """Run one selection action over the user's selected note text.

    Per-user via the auth dependency. The model client is built from the user's
    selected model; the selection is wrapped as untrusted context by
    ``note_actions``; the Daily-only web gate is enforced via the injected live
    mode (a structured refusal, never a silent local downgrade). The runner never
    raises; its structured result crosses the wire.
    """
    _check()
    client = build_client(request.model or None)
    runner = make_note_action_runner(model_client=client, mode_provider=lambda: mode)
    result = runner(request.action, request.selection)
    return NoteActionResultSchema(**result.to_dict())
