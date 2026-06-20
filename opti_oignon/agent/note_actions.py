#!/usr/bin/env python3
"""LLM-from-note selection actions (N.3): the agent-side selection-action surface.

From a note selection the user asks one of five local actions -- fact-check,
develop, summarize, rewrite, make-checklist -- or the Daily-only
fact-check-with-web. The selected text is wrapped as untrusted data via
:mod:`opti_oignon.agent.untrusted_context` (the S175 / Odysseus anti-injection
core), so injection-looking note text cannot steer the model: the action's
instruction is the only trusted message, the selection rides the user role inside
the untrusted-data markers, and the policy header forbids the model from treating
the enclosed text as instructions. The model is invoked once and the result is
returned for the UI to show alongside the note and insert.

Design notes:

- This is NOT a model-reachable tool. Unlike the N.4 ``manage_notes`` tool, this
  surface is driven by the user selecting text and choosing an action, not by the
  model's tool-calling. It defines no ``ToolSchema`` and registers nothing in the
  agent tool registry, so it grows no schema-count or allowlist pin.
- Mode and egress. The five local actions run in both Daily and Bulbe (they reach
  no network). fact-check-with-web needs web egress and is Daily-only: the runner
  refuses it with a structured result (never a silent local downgrade) outside
  Daily. The egress itself rides ``web_search`` (in ``NETWORK_TOOLS``, forbidden
  in Bulbe); this gate is the surface's own refusal so the model is never invoked
  for a web action outside Daily. The mode resolution is fail-secure: an
  undeterminable mode is treated as Bulbe.
- Dependency injection. The model client is a one-shot inference seam the caller
  injects -- a callable taking the built messages and returning the completion
  text (or an object exposing ``stream``). The agent loop is likewise invoked
  with a model client the route builds from the user's selected model
  (api/routes_agent.py); this surface follows the same posture, so the model and
  its parameters are the caller's choice, not this module's. A later N.3 route/UI
  session wires the client; until then an un-injected runner reports a clean
  failure rather than guessing a model.
- The mode provider is injectable for tests; the inference seam is injectable for
  tests; nothing here imports the backend at module load, so the surface is
  exercised directly by pytest with no fastapi / ollama chain (the S243 lesson).

``checkpoint_before_apply`` is hardcoded True and never overridable;
``FEATURE_AVAILABLE`` gates graceful degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .untrusted_context import untrusted_message

logger = logging.getLogger(__name__)

# Hardcoded and never overridable: a checkpoint is taken before any mutation is
# applied. Project-wide non-negotiable for every new module.
checkpoint_before_apply = True

FEATURE_AVAILABLE = True

# Mode literals mirrored locally so the fail-secure default is a constant here and
# the live mode comes from security_mode (resolved lazily and guarded).
MODE_DAILY = "daily"
MODE_BULBE = "bulbe"

# The source label the note selection is wrapped under (sanitised by
# untrusted_context to a safe tag attribute).
SOURCE_NOTE = "note"

# The action names. Five local, one web-only.
ACTION_FACT_CHECK = "fact_check"
ACTION_FACT_CHECK_WEB = "fact_check_web"
ACTION_DEVELOP = "develop"
ACTION_SUMMARIZE = "summarize"
ACTION_REWRITE = "rewrite"
ACTION_MAKE_CHECKLIST = "make_checklist"

LOCAL_ACTIONS: frozenset[str] = frozenset(
    {
        ACTION_FACT_CHECK,
        ACTION_DEVELOP,
        ACTION_SUMMARIZE,
        ACTION_REWRITE,
        ACTION_MAKE_CHECKLIST,
    }
)

# Web-egress actions: Daily-only. Kept a set so a later web-backed action joins
# here and inherits the same gate without touching the runner.
WEB_ACTIONS: frozenset[str] = frozenset({ACTION_FACT_CHECK_WEB})

ALL_ACTIONS: frozenset[str] = LOCAL_ACTIONS | WEB_ACTIONS

# The trusted instruction per action. Each references the untrusted-data block
# that follows in the user role, so the model knows the selection is data.
_ACTION_INSTRUCTIONS: dict[str, str] = {
    ACTION_FACT_CHECK: (
        "You are fact-checking the user's note. Assess the factual accuracy of "
        "the claims in the untrusted-data block below using only your own "
        "knowledge. Do not browse the web. List each notable claim with a "
        "verdict (supported, unsupported, or uncertain) and a brief reason."
    ),
    ACTION_FACT_CHECK_WEB: (
        "You are fact-checking the user's note with web search. Verify the "
        "claims in the untrusted-data block below against current web sources "
        "and cite them. List each notable claim with a verdict and its source."
    ),
    ACTION_DEVELOP: (
        "You are developing an idea from the user's note. Expand and deepen the "
        "idea in the untrusted-data block below: add structure, supporting "
        "points, and concrete next steps, staying on the author's topic."
    ),
    ACTION_SUMMARIZE: (
        "You are summarizing the user's note. Produce a concise summary of the "
        "untrusted-data block below, preserving the key points and the author's "
        "intent."
    ),
    ACTION_REWRITE: (
        "You are rewriting the user's note. Rewrite the text in the "
        "untrusted-data block below for clarity and flow without changing its "
        "meaning. Return only the rewritten text."
    ),
    ACTION_MAKE_CHECKLIST: (
        "You are turning the user's note into a checklist. Convert the text in "
        "the untrusted-data block below into an actionable markdown checklist, "
        "one '- [ ] ' item per task, preserving order where it matters."
    ),
}


@dataclass
class NoteActionResult:
    """The outcome of a selection action.

    ``ok`` True carries the model ``text``; ``refused`` True marks the
    structured egress refusal (a web action outside Daily); any other failure is
    ``ok`` False with a ``reason`` and ``refused`` False. The handler never
    raises.
    """

    action: str
    ok: bool
    text: str = ""
    refused: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "ok": self.ok,
            "text": self.text,
            "refused": self.refused,
            "reason": self.reason,
        }


def _normalize_action(action: Any) -> str:
    return str(action or "").strip().lower()


def requires_web(action: str) -> bool:
    """True when the action needs web egress (Daily-only)."""
    return _normalize_action(action) in WEB_ACTIONS


def build_messages(action: str, selection: str) -> list[dict[str, str]]:
    """Build the one-shot [system, user] messages for an action.

    The system message is the trusted action instruction; the user message wraps
    the selection as untrusted data (the anti-injection core). Raises
    ``ValueError`` on an unknown action. Callers that may receive an empty
    selection should guard it before building (the runner does).
    """
    act = _normalize_action(action)
    if act not in ALL_ACTIONS:
        raise ValueError("Unknown note action: " + repr(action))
    instruction = _ACTION_INSTRUCTIONS[act]
    user_msg = untrusted_message(str(selection), source=SOURCE_NOTE)
    return [{"role": "system", "content": instruction}, user_msg]


def _default_mode() -> str:
    """Resolve the live security mode, fail-secure to Bulbe when undeterminable.

    Lazy and guarded so the module loads without the backend; the live mode comes
    from security_mode. An undeterminable mode is Bulbe (the project-wide
    fail-secure posture), which refuses any web action.
    """
    try:
        from ..security_mode import get_current_mode

        mode = str(get_current_mode() or "").strip().lower()
        return mode or MODE_BULBE
    except Exception:  # pragma: no cover - defensive guard
        return MODE_BULBE


def _resolve_mode(mode_provider: Optional[Callable[[], str]]) -> str:
    if mode_provider is None:
        return _default_mode()
    try:
        return str(mode_provider() or "").strip().lower() or MODE_BULBE
    except Exception:
        # Fail-secure: an undeterminable mode is Bulbe.
        return MODE_BULBE


def _default_model_client() -> Any:
    """No process-default model client: the caller injects a one-shot client.

    The agent loop is invoked with a model client the route builds from the
    user's selected model (api/routes_agent.py); this surface follows the same
    dependency-injection posture, so the model and its parameters are the
    caller's choice, not this module's. Returns None so an un-injected runner
    reports a clean failure rather than guessing a model. A later N.3 route/UI
    session wires the client.
    """
    return None


def _invoke_once(model_client: Any, messages: list[dict[str, str]]) -> str:
    """Invoke the one-shot inference seam and coerce its output to text.

    Mirrors the loop's tolerance: ``model_client`` may expose ``stream`` or be a
    plain callable taking the messages. The return may be a string or an iterable
    of chunks (strings, ``{"content": ...}`` dicts, or objects with ``content``).
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


def make_note_action_runner(
    model_client: Any = None,
    *,
    mode_provider: Optional[Callable[[], str]] = None,
) -> Callable[[str, str], NoteActionResult]:
    """Build a selection-action runner, injecting the inference seam and mode.

    ``model_client`` is the one-shot inference seam (a callable over the built
    messages, or an object with ``stream``); when None the default resolver is
    used (which returns None unless wired by a caller, yielding a clean failure).
    ``mode_provider`` resolves the live security mode for the web-egress gate;
    when None the live mode comes from security_mode, fail-secure to Bulbe.

    The returned ``run(action, selection)`` validates the action, refuses an
    empty selection, gates web actions to Daily (a structured refusal before any
    generation), wraps the selection as untrusted data, invokes the model once,
    and returns a :class:`NoteActionResult`. It never raises.
    """

    def run(action: str, selection: str) -> NoteActionResult:
        act = _normalize_action(action)
        if act not in ALL_ACTIONS:
            return NoteActionResult(
                action=act,
                ok=False,
                reason="Unknown action: " + repr(action),
            )
        if not selection or not str(selection).strip():
            return NoteActionResult(
                action=act,
                ok=False,
                reason="Empty selection: nothing to act on.",
            )
        if requires_web(act):
            mode = _resolve_mode(mode_provider)
            if mode != MODE_DAILY:
                return NoteActionResult(
                    action=act,
                    ok=False,
                    refused=True,
                    reason=(
                        "fact-check-with-web requires Daily mode; refused in "
                        + mode
                        + " mode."
                    ),
                )
        client = model_client if model_client is not None else _default_model_client()
        if client is None:
            return NoteActionResult(
                action=act,
                ok=False,
                reason="Model client unavailable.",
            )
        try:
            messages = build_messages(act, str(selection))
            text = _invoke_once(client, messages)
        except Exception as exc:
            return NoteActionResult(
                action=act,
                ok=False,
                reason="Action '" + act + "' failed: " + str(exc),
            )
        return NoteActionResult(action=act, ok=True, text=text)

    return run
