#!/usr/bin/env python3
"""Tool-call dispatch for the agent loop (S175, Theme 3 / Odysseus Core).

Two responsibilities (ODYSSEUS_SPEC.md Section 5.2 and Section 5.3):

1. Dual dispatch. A round either emits native function-calling tool calls or
   text the model wrote in one of the local-model conventions. ``resolve_tool_
   calls`` decides which path a round used -- native (reusing the
   ``structured_output`` ``ToolCallRequest`` schema and ``json_repair`` for
   tolerant argument parsing) when the model emits them, otherwise the parser
   in ``tool_parsing`` -- and normalises both into a single ``ToolCall``.

2. The sandbox dispatch invariant. Every filesystem / shell / code tool runs
   ONLY through the S73/S74 disposable bwrap sandbox via the injected
   ``sandbox_tools.SandboxToolSession``. There is no in-process, tempdir, or
   host path in this module: the only way a sandboxed tool executes is by
   calling a method on the session object, and the dispatch refuses to act
   unless that session is backed by an available bwrap. When bwrap is
   unavailable the agent refuses; it never falls back to the host. Copy-out of
   results stays behind the human approval gate (Daily at copy-out, Bulbe
   per-call). The concrete sandboxed tool set lands in S176; S175 lands this
   seam and proves the invariant.

Gating is delegated to ``allowlists``: the dispatch consults the active mode's
allowlist before any tool runs, and in Bulbe routes the call through the
human-approval gate (fail-secure). A refused or failed tool becomes a
``DispatchResult`` observation, never an exception, so the loop never raises
into the conversation path.

Importlib-isolatable: the sibling agent modules and ``json_repair`` are pure or
self-guarding; ``structured_output`` is guarded. The sandbox is injected (duck
typed), so this module loads and its dispatch is exercised without the backend.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from opti_oignon.agent import allowlists
from opti_oignon.agent.tool_parsing import ParsedToolCall, parse_tool_blocks

logger = logging.getLogger(__name__)

# Module conventions (Theme 3).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Tolerant JSON for string-encoded native arguments (stdlib-only, guarded).
try:
    from opti_oignon.json_repair import repair_json as _repair_json

    JSON_REPAIR_AVAILABLE = True
except Exception:  # pragma: no cover - defensive guard
    _repair_json = None
    JSON_REPAIR_AVAILABLE = False

# Reuse the native function-calling schema for normalisation when available.
try:
    from opti_oignon.structured_output import ToolCallRequest as _ToolCallRequest

    STRUCTURED_OUTPUT_AVAILABLE = True
except Exception:  # pragma: no cover - defensive guard
    _ToolCallRequest = None
    STRUCTURED_OUTPUT_AVAILABLE = False

# Which path a round used.
PATH_NATIVE = "native"
PATH_TEXT = "text"

# DispatchResult reason codes (gate reasons are reused from ``allowlists``).
REASON_EXECUTED = "executed"
REASON_SANDBOX_UNAVAILABLE = "sandbox_unavailable"
REASON_NO_EXECUTOR = "no_executor"
REASON_ERROR = "error"


@dataclass
class ToolCall:
    """A single normalised tool call, from either dispatch path.

    ``source`` is ``"native"`` or the textual format the parser recovered it
    from (one of ``tool_parsing.SUPPORTED_FORMATS``). ``raw`` keeps the original
    payload for observation and audit.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments), "source": self.source}


@dataclass
class DispatchResult:
    """The outcome of dispatching one ``ToolCall``.

    ``executed`` says whether the tool ran at all; ``observation`` is the text
    fed back to the loop (tool output or a refusal explanation); ``reason`` is a
    machine code. A refusal or an error always sets ``executed`` False and never
    raises.
    """

    tool_name: str
    executed: bool
    observation: str
    reason: str
    source: str = ""
    mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "executed": self.executed,
            "reason": self.reason,
            "observation": self.observation,
            "source": self.source,
            "mode": self.mode,
        }


# Coercion helpers


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        if _repair_json is not None:
            try:
                return _repair_json(text)
            except Exception:
                return None
        return None


def _safe_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, sort_keys=True)
    except Exception:
        return repr(obj)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce a tool argument to bool ('true'/'false' strings included)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no", ""):
            return False
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_args(raw: Any) -> dict[str, Any]:
    """Coerce native arguments to a dict (a JSON-string payload is parsed)."""
    if isinstance(raw, str):
        parsed = _loads(raw)
        raw = parsed if isinstance(parsed, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    return raw


def _make_tool_call(name: str, args: dict[str, Any], source: str, raw: str) -> ToolCall:
    """Build a ToolCall, validating through ToolCallRequest when available.

    This is the single normalisation seam shared by both dispatch paths.
    """
    nm = str(name).strip()
    arguments = dict(args)
    if _ToolCallRequest is not None:
        try:
            req = _ToolCallRequest(tool_name=nm, arguments=arguments)
            nm = req.tool_name
            arguments = dict(req.arguments)
        except Exception:
            pass
    return ToolCall(name=nm, arguments=arguments, source=source, raw=raw)


# Path resolution and normalisation


def _get_message(response: Any) -> Any:
    if response is None:
        return None
    if isinstance(response, dict):
        return response.get("message")
    return getattr(response, "message", None)


def extract_native_calls(response: Any) -> list[ToolCall] | None:
    """Native function-calling tool calls, or None if the round has none.

    None signals the caller to fall back to the text parser. Supports the
    Ollama shape (``message.tool_calls[].function.{name,arguments}``, arguments
    as a dict) and the OpenAI shape (arguments as a JSON string).
    """
    message = _get_message(response)
    tool_calls = _get_attr(message, "tool_calls", None)
    if not tool_calls:
        tool_calls = _get_attr(response, "tool_calls", None)
    if not tool_calls:
        return None
    calls: list[ToolCall] = []
    for tc in tool_calls:
        fn = _get_attr(tc, "function", None)
        name = _get_attr(fn, "name", None) if fn is not None else None
        if not name:
            name = _get_attr(tc, "name", None)
        if not name:
            continue
        raw_args = _get_attr(fn, "arguments", None) if fn is not None else None
        if raw_args is None:
            raw_args = _get_attr(tc, "arguments", {})
        calls.append(
            _make_tool_call(str(name).strip(), _normalize_args(raw_args), PATH_NATIVE, _safe_dumps(tc))
        )
    return calls


def extract_text(response: Any) -> str:
    """The assistant text content of a response, or an empty string."""
    message = _get_message(response)
    content = _get_attr(message, "content", None)
    if content is None:
        content = _get_attr(response, "content", "")
    return content or ""


def _from_parsed(pc: ParsedToolCall) -> ToolCall:
    return _make_tool_call(pc.name, dict(pc.arguments), pc.source, pc.raw)


def resolve_tool_calls(response: Any) -> tuple[list[ToolCall], str]:
    """Resolve a round into normalised tool calls and the path that produced them.

    Native function calls are preferred when present; otherwise the text is run
    through the parser. Both paths yield the same ``ToolCall`` representation.
    """
    native = extract_native_calls(response)
    if native is not None:
        return native, PATH_NATIVE
    parsed = parse_tool_blocks(extract_text(response))
    return [_from_parsed(p) for p in parsed], PATH_TEXT


# The sandbox seam (the invariant)


def sandbox_ready(session: Any) -> bool:
    """Whether the injected sandbox session is backed by an available bwrap.

    This is the physical invariant: the agent acts only when true isolation is
    available. A missing session, a missing manager, or an unavailable bwrap
    all return False, so the dispatch refuses rather than touching the host.
    There is deliberately no tempdir or degraded path here.
    """
    if session is None:
        return False
    mgr = getattr(session, "sandbox_manager", None)
    if mgr is None:
        return False
    return bool(getattr(mgr, "bwrap_available", False))


# The only execution path for sandboxed tools: methods on the session object.
_SANDBOX_DISPATCH: dict[str, Callable[[Any, dict[str, Any]], str]] = {
    "bash": lambda s, a: s.bash(_as_str(a.get("command")), _as_int(a.get("timeout"), 30)),
    "view": lambda s, a: s.view(
        _as_str(a.get("path")), _as_int(a.get("start_line"), 0), _as_int(a.get("end_line"), 0)
    ),
    "create_file": lambda s, a: s.create_file(_as_str(a.get("path")), _as_str(a.get("content"))),
    "str_replace": lambda s, a: s.str_replace(
        _as_str(a.get("path")), _as_str(a.get("old_str")), _as_str(a.get("new_str"))
    ),
    # S228 (AGT_SPEC 5.1/5.5): the three read-only workspace tools, methods on
    # the same session object; argument names match the schemas exactly.
    "grep": lambda s, a: s.grep(
        _as_str(a.get("pattern")),
        _as_str(a.get("path") or "."),
        glob=_as_str(a.get("glob")),
        is_regex=_as_bool(a.get("is_regex"), False),
        case_sensitive=_as_bool(a.get("case_sensitive"), False),
        context_lines=_as_int(a.get("context_lines"), 0),
        max_results=_as_int(a.get("max_results"), 100),
    ),
    "glob": lambda s, a: s.glob(
        _as_str(a.get("pattern")),
        _as_str(a.get("path") or "."),
        max_results=_as_int(a.get("max_results"), 200),
    ),
    "ls": lambda s, a: s.ls(
        _as_str(a.get("path") or "."),
        max_entries=_as_int(a.get("max_entries"), 200),
    ),
}


def _refusal_text(name: str, decision: allowlists.GateDecision) -> str:
    if decision.reason == allowlists.REASON_NOT_ALLOWED:
        return f"Tool '{name}' is not permitted in {decision.mode} mode."
    if decision.reason == allowlists.REASON_DENIED:
        return f"Tool call '{name}' was not approved."
    return f"Tool '{name}' was refused: {decision.reason}."


def dispatch_tool_call(
    call: ToolCall,
    *,
    mode: str | None = None,
    conversation_id: str = "",
    sandbox: Any = None,
    approval_fn: Callable[[str, str, dict[str, Any]], bool] | None = None,
    tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> DispatchResult:
    """Gate then execute a single tool call, returning an observation result.

    Order: the allowlist gate (plus the Bulbe human gate) first; then, for a
    sandboxed tool, the sandbox-readiness invariant and execution through the
    session; for a non-sandbox tool, an injected handler if one is registered.
    Never raises -- every refusal or error is a ``DispatchResult``.
    """
    decision = allowlists.evaluate(
        call.name,
        call.arguments,
        mode=mode,
        conversation_id=conversation_id,
        approval_fn=approval_fn,
    )
    if not decision.allowed:
        return DispatchResult(
            tool_name=call.name,
            executed=False,
            observation=_refusal_text(call.name, decision),
            reason=decision.reason,
            source=call.source,
            mode=decision.mode,
        )

    if allowlists.is_sandbox_tool(call.name):
        if not sandbox_ready(sandbox):
            return DispatchResult(
                tool_name=call.name,
                executed=False,
                observation=(
                    f"Tool '{call.name}' requires the disposable bwrap sandbox, which is "
                    "not available; the agent refuses to run filesystem, shell, or code "
                    "tools on the host."
                ),
                reason=REASON_SANDBOX_UNAVAILABLE,
                source=call.source,
                mode=decision.mode,
            )
        if not bool(getattr(sandbox, "active", False)):
            return DispatchResult(
                tool_name=call.name,
                executed=False,
                observation=f"Tool '{call.name}' has no active sandbox session.",
                reason=REASON_SANDBOX_UNAVAILABLE,
                source=call.source,
                mode=decision.mode,
            )
        try:
            output = _SANDBOX_DISPATCH[call.name](sandbox, call.arguments)
        except Exception as exc:
            return DispatchResult(
                tool_name=call.name,
                executed=False,
                observation=f"Tool '{call.name}' raised an error: {exc}",
                reason=REASON_ERROR,
                source=call.source,
                mode=decision.mode,
            )
        return DispatchResult(
            tool_name=call.name,
            executed=True,
            observation=_as_str(output),
            reason=REASON_EXECUTED,
            source=call.source,
            mode=decision.mode,
        )

    # Allowed non-sandbox tool. No executor ships in S175; an injected handler
    # is the forward hook for the S176 tool set.
    handler = (tool_handlers or {}).get(call.name)
    if handler is None:
        return DispatchResult(
            tool_name=call.name,
            executed=False,
            observation=f"Tool '{call.name}' has no executor in this build.",
            reason=REASON_NO_EXECUTOR,
            source=call.source,
            mode=decision.mode,
        )
    try:
        output = handler(call.arguments)
    except Exception as exc:
        return DispatchResult(
            tool_name=call.name,
            executed=False,
            observation=f"Tool '{call.name}' raised an error: {exc}",
            reason=REASON_ERROR,
            source=call.source,
            mode=decision.mode,
        )
    return DispatchResult(
        tool_name=call.name,
        executed=True,
        observation=_as_str(output),
        reason=REASON_EXECUTED,
        source=call.source,
        mode=decision.mode,
    )


def dispatch_round(
    response: Any,
    *,
    mode: str | None = None,
    conversation_id: str = "",
    sandbox: Any = None,
    approval_fn: Callable[[str, str, dict[str, Any]], bool] | None = None,
    tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
) -> tuple[list[DispatchResult], str]:
    """Resolve a model response and dispatch every tool call it produced."""
    calls, path = resolve_tool_calls(response)
    results = [
        dispatch_tool_call(
            c,
            mode=mode,
            conversation_id=conversation_id,
            sandbox=sandbox,
            approval_fn=approval_fn,
            tool_handlers=tool_handlers,
        )
        for c in calls
    ]
    return results, path
