#!/usr/bin/env python3
"""Tool-call dispatch for the agent loop.

Two responsibilities:

1. Dual dispatch. A round either emits native function-calling tool calls or
   text the model wrote in one of the local-model conventions. ``resolve_tool_
   calls`` decides which path a round used -- native (reusing the
   ``structured_output`` ``ToolCallRequest`` schema and ``json_repair`` for
   tolerant argument parsing) when the model emits them, otherwise the parser
   in ``tool_parsing`` -- and normalises both into a single ``ToolCall``.

2. The sandbox dispatch invariant. Every filesystem / shell / code tool runs
   ONLY through the disposable bwrap sandbox via the injected
   ``sandbox_tools.SandboxToolSession``. There is no in-process, tempdir, or
   host path in this module: the only way a sandboxed tool executes is by
   calling a method on the session object, and the dispatch refuses to act
   unless that session is backed by an available bwrap. When bwrap is
   unavailable the agent refuses; it never falls back to the host. Copy-out of
   results stays behind the human approval gate (Daily at copy-out, Bulbe
   per-call). The concrete sandboxed tool set comes later; this stage lands this
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

# The provenance gate. The package always carries it; a window that loads this
# module alone without it dispatches as before the gate existed, and says so.
try:
    from opti_oignon import provenance as _provenance
except Exception as _gate_missing:  # noqa: BLE001 - absence is said, not raised
    _provenance = None
    logger.warning("the provenance gate cannot be loaded (%s): the agent's tool calls run ungated", _gate_missing)

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
    raises. ``provenance`` is what the gate found: the call's class, the label
    of each argument as it would run, the decision, and whether a person was
    asked; empty when the call never reached the gate.
    """

    tool_name: str
    executed: bool
    observation: str
    reason: str
    source: str = ""
    mode: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "executed": self.executed,
            "reason": self.reason,
            "observation": self.observation,
            "source": self.source,
            "mode": self.mode,
            "provenance": dict(self.provenance),
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
    """Whether the injected sandbox session runs its commands under bwrap.

    This is the physical invariant: the agent acts only when true isolation is
    in use. A missing session, a missing manager, or a manager that does not
    run bwrap (absent, or installed while the resolved backend is tempdir) all
    return False, so the dispatch refuses rather than touching the host.
    There is deliberately no tempdir or degraded path here. A seam that does
    not report ``bwrap_in_use`` is judged on availability alone; the
    SandboxManager always reports it.
    """
    if session is None:
        return False
    mgr = getattr(session, "sandbox_manager", None)
    if mgr is None:
        return False
    in_use = getattr(mgr, "bwrap_in_use", None)
    if in_use is not None:
        return bool(in_use)
    return bool(getattr(mgr, "bwrap_available", False))


# What each sandboxed tool's session call receives: the arguments coerced as the
# call has always coerced them, with the value each takes when the model gives
# none. The gate labels these values, and the session receives exactly them.
_SANDBOX_ARGUMENTS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "bash": lambda a: {"command": _as_str(a.get("command")), "timeout": _as_int(a.get("timeout"), 30)},
    "view": lambda a: {
        "path": _as_str(a.get("path")),
        "start_line": _as_int(a.get("start_line"), 0),
        "end_line": _as_int(a.get("end_line"), 0),
    },
    "create_file": lambda a: {"path": _as_str(a.get("path")), "content": _as_str(a.get("content"))},
    "str_replace": lambda a: {
        "path": _as_str(a.get("path")),
        "old_str": _as_str(a.get("old_str")),
        "new_str": _as_str(a.get("new_str")),
    },
    # The three read-only workspace tools; argument names match the schemas exactly.
    "grep": lambda a: {
        "pattern": _as_str(a.get("pattern")),
        "path": _as_str(a.get("path") or "."),
        "glob": _as_str(a.get("glob")),
        "is_regex": _as_bool(a.get("is_regex"), False),
        "case_sensitive": _as_bool(a.get("case_sensitive"), False),
        "context_lines": _as_int(a.get("context_lines"), 0),
        "max_results": _as_int(a.get("max_results"), 100),
    },
    "glob": lambda a: {
        "pattern": _as_str(a.get("pattern")),
        "path": _as_str(a.get("path") or "."),
        "max_results": _as_int(a.get("max_results"), 200),
    },
    "ls": lambda a: {"path": _as_str(a.get("path") or "."), "max_entries": _as_int(a.get("max_entries"), 200)},
}

# The value each of those arguments takes when the model gives none.
_SANDBOX_DEFAULTS: dict[str, dict[str, Any]] = {
    "bash": {"timeout": 30},
    "view": {"start_line": 0, "end_line": 0},
    "create_file": {},
    "str_replace": {},
    "grep": {"path": ".", "glob": "", "is_regex": False, "case_sensitive": False, "context_lines": 0,
             "max_results": 100},
    "glob": {"path": ".", "max_results": 200},
    "ls": {"path": ".", "max_entries": 200},
}

# The only execution path for sandboxed tools: methods on the session object,
# handed the arguments above.
_SANDBOX_DISPATCH: dict[str, Callable[[Any, dict[str, Any]], str]] = {
    "bash": lambda s, a: s.bash(a["command"], a["timeout"]),
    "view": lambda s, a: s.view(a["path"], a["start_line"], a["end_line"]),
    "create_file": lambda s, a: s.create_file(a["path"], a["content"]),
    "str_replace": lambda s, a: s.str_replace(a["path"], a["old_str"], a["new_str"]),
    "grep": lambda s, a: s.grep(
        a["pattern"],
        a["path"],
        glob=a["glob"],
        is_regex=a["is_regex"],
        case_sensitive=a["case_sensitive"],
        context_lines=a["context_lines"],
        max_results=a["max_results"],
    ),
    "glob": lambda s, a: s.glob(a["pattern"], a["path"], max_results=a["max_results"]),
    "ls": lambda s, a: s.ls(a["path"], max_entries=a["max_entries"]),
}


def _sink_arguments(call: ToolCall, handler: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The arguments the call's sink receives, and the value each takes when the model gives none.

    A sandboxed tool's are its session call's; a handler that declares how it
    reads its arguments (``canonical_arguments``, ``argument_defaults``) is
    read that way; any other handler receives the arguments as the model gave
    them.
    """
    if allowlists.is_sandbox_tool(call.name):
        return _SANDBOX_ARGUMENTS[call.name](call.arguments), dict(_SANDBOX_DEFAULTS[call.name])
    canonical = getattr(handler, "canonical_arguments", None)
    defaults = getattr(handler, "argument_defaults", None)
    arguments = canonical(call.arguments) if callable(canonical) else dict(call.arguments)
    return dict(arguments), dict(defaults) if isinstance(defaults, dict) else {}


def _labelled_approval(approval_fn: Any, assessment: Any, asked: list) -> Any:
    """The approval ``evaluate`` asks: the caller's (or the default queue's), shown each argument's label and the class.

    ``asked`` records that a person was asked. A run with no one to ask keeps
    its own refusal, which ``evaluate`` reports with its reason.
    """
    if assessment is None or isinstance(approval_fn, allowlists.NoApprovalChannel):
        return approval_fn
    labels, effect = dict(assessment.labels), assessment.effect

    def ask(conversation_id: str, tool_name: str, arguments: dict[str, Any]) -> bool:
        asked.append(tool_name)
        if approval_fn is None:
            return allowlists.request_approval(conversation_id, tool_name, arguments, labels=labels, effect=effect)
        if _provenance.accepts_labels(approval_fn):
            return bool(approval_fn(conversation_id, tool_name, arguments, labels=labels, effect=effect))
        return bool(approval_fn(conversation_id, tool_name, arguments))

    return ask


def _refusal_text(name: str, decision: allowlists.GateDecision) -> str:
    if decision.reason == allowlists.REASON_NOT_ALLOWED:
        return f"Tool '{name}' is not permitted in {decision.mode} mode."
    if decision.reason == allowlists.REASON_DENIED:
        return f"Tool call '{name}' was not approved."
    if decision.reason == allowlists.REASON_NO_CHANNEL:
        return (f"Tool '{name}' needs a person's approval in {decision.mode} mode, and this run has no one to "
                f"ask ({decision.detail}); nothing ran.")
    return f"Tool '{name}' was refused: {decision.reason}."


def dispatch_tool_call(
    call: ToolCall,
    *,
    mode: str | None = None,
    conversation_id: str = "",
    sandbox: Any = None,
    approval_fn: Callable[[str, str, dict[str, Any]], bool] | None = None,
    tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    provenance: Any = None,
) -> DispatchResult:
    """Gate then execute a single tool call, returning an observation result.

    Order: the arguments as the sink will receive them, each labelled by
    ``provenance`` (the run's turn; None endorses nothing); the allowlist gate
    (plus the Bulbe human gate, shown those arguments and their labels); the
    provenance gate's network policy; then, for a sandboxed tool, the
    sandbox-readiness invariant and execution through the session; for a
    non-sandbox tool, an injected handler if one is registered. The sink
    receives exactly the arguments that were labelled. With the gate loaded,
    the mode is read again at each call and never looser than the machine's
    (an escalation holds a running agent's next call to Bulbe), and a call
    that cannot run -- no handler, no sandbox -- is refused before anyone is
    asked about it. Never raises -- every refusal or error is a
    ``DispatchResult``.
    """
    handler = (tool_handlers or {}).get(call.name)
    if _provenance is not None:
        mode = allowlists.floor_mode(mode)
    try:
        arguments, defaults = _sink_arguments(call, handler)
    except Exception as exc:
        return DispatchResult(
            tool_name=call.name,
            executed=False,
            observation=f"Tool '{call.name}' raised an error: {exc}",
            reason=REASON_ERROR,
            source=call.source,
            mode=mode or "",
        )
    assessment = (
        _provenance.assess(call.name, arguments, provenance, defaults=defaults)
        if _provenance is not None else None
    )
    if assessment is not None and allowlists.is_tool_allowed(call.name, mode):
        unrunnable = _unrunnable(call, handler, sandbox, mode, assessment)
        if unrunnable is not None:
            return unrunnable
    asked: list[str] = []
    decision = allowlists.evaluate(
        call.name,
        arguments if assessment is None else dict(assessment.arguments),
        mode=mode,
        conversation_id=conversation_id,
        approval_fn=_labelled_approval(approval_fn, assessment, asked),
    )
    if not decision.allowed:
        return DispatchResult(
            tool_name=call.name,
            executed=False,
            observation=_refusal_text(call.name, decision),
            reason=decision.reason,
            source=call.source,
            mode=decision.mode,
            provenance=_refused_metadata(assessment, decision.reason, bool(asked)),
        )
    metadata: dict[str, Any] = {}
    if assessment is not None:
        verdict = _provenance.decide(call.name, assessment, provenance, mode=decision.mode)
        metadata = dict(verdict.metadata(), asked=bool(asked) or verdict.asked)
        if not verdict.allowed:
            return DispatchResult(
                tool_name=call.name,
                executed=False,
                observation=verdict.message,
                reason=verdict.reason,
                source=call.source,
                mode=decision.mode,
                provenance=metadata,
            )
        arguments = verdict.arguments

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
                provenance=metadata,
            )
        if not bool(getattr(sandbox, "active", False)):
            return DispatchResult(
                tool_name=call.name,
                executed=False,
                observation=f"Tool '{call.name}' has no active sandbox session.",
                reason=REASON_SANDBOX_UNAVAILABLE,
                source=call.source,
                mode=decision.mode,
                provenance=metadata,
            )
        try:
            output = _SANDBOX_DISPATCH[call.name](sandbox, arguments)
        except Exception as exc:
            return DispatchResult(
                tool_name=call.name,
                executed=False,
                observation=f"Tool '{call.name}' raised an error: {exc}",
                reason=REASON_ERROR,
                source=call.source,
                mode=decision.mode,
                provenance=metadata,
            )
        return DispatchResult(
            tool_name=call.name,
            executed=True,
            observation=_as_str(output),
            reason=REASON_EXECUTED,
            source=call.source,
            mode=decision.mode,
            provenance=metadata,
        )

    # Allowed non-sandbox tool. No executor ships; an injected handler
    # is the forward hook for the tool set.
    if handler is None:
        return DispatchResult(
            tool_name=call.name,
            executed=False,
            observation=f"Tool '{call.name}' has no executor in this build.",
            reason=REASON_NO_EXECUTOR,
            source=call.source,
            mode=decision.mode,
            provenance=metadata,
        )
    try:
        output = handler(arguments)
    except Exception as exc:
        return DispatchResult(
            tool_name=call.name,
            executed=False,
            observation=f"Tool '{call.name}' raised an error: {exc}",
            reason=REASON_ERROR,
            source=call.source,
            mode=decision.mode,
            provenance=metadata,
        )
    return DispatchResult(
        tool_name=call.name,
        executed=True,
        observation=_as_str(output),
        reason=REASON_EXECUTED,
        source=call.source,
        mode=decision.mode,
        provenance=metadata,
    )


def _unrunnable(call: ToolCall, handler: Any, sandbox: Any, mode: Any, assessment: Any) -> DispatchResult | None:
    """The refusal of a call that could not run whatever anyone answered, or None when it can run."""
    if allowlists.is_sandbox_tool(call.name):
        if sandbox_ready(sandbox) and bool(getattr(sandbox, "active", False)):
            return None
        observation = (
            f"Tool '{call.name}' requires the disposable bwrap sandbox, which is not available; the agent "
            "refuses to run filesystem, shell, or code tools on the host."
            if not sandbox_ready(sandbox) else f"Tool '{call.name}' has no active sandbox session."
        )
        reason = REASON_SANDBOX_UNAVAILABLE
    elif handler is None:
        observation, reason = f"Tool '{call.name}' has no executor in this build.", REASON_NO_EXECUTOR
    else:
        return None
    shown_mode = mode if mode in allowlists.VALID_MODES else allowlists.MODE_BULBE
    return DispatchResult(
        tool_name=call.name,
        executed=False,
        observation=observation,
        reason=reason,
        source=call.source,
        mode=shown_mode,
        provenance=_refused_metadata(assessment, reason, False),
    )


def _refused_metadata(assessment: Any, reason: str, asked: bool) -> dict[str, Any]:
    """The provenance a call refused before the network policy carries; empty when it was never assessed."""
    if assessment is None:
        return {}
    return {"effect": assessment.effect, "labels": dict(assessment.labels), "decision": reason, "asked": asked}


def dispatch_round(
    response: Any,
    *,
    mode: str | None = None,
    conversation_id: str = "",
    sandbox: Any = None,
    approval_fn: Callable[[str, str, dict[str, Any]], bool] | None = None,
    tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    provenance: Any = None,
) -> tuple[list[DispatchResult], str]:
    """Resolve a model response and dispatch every tool call it produced, each judged by ``provenance``."""
    calls, path = resolve_tool_calls(response)
    results = [
        dispatch_tool_call(
            c,
            mode=mode,
            conversation_id=conversation_id,
            sandbox=sandbox,
            approval_fn=approval_fn,
            tool_handlers=tool_handlers,
            provenance=provenance,
        )
        for c in calls
    ]
    return results, path
