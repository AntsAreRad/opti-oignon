"""Robust tool-calling primitives (Lot 1 + Lot 2 of the agentic robustness cycle).

Layered defense for tool selection, so the agentic loop is sturdy regardless of
how cooperative a given model is:

  Layer 1  native_tool_schemas / parse_native_tool_calls
           Ollama native function-calling: build schemas from the registry's
           ToolDefinition.parameters and parse the structured message.tool_calls
           directly. This matches what qwen2.5-coder / qwen3 were trained on, so
           it selects tools far more reliably than a custom format= wrapper, and
           it supports parallel calls.

  Layer 2a forced_decision_model
           When a tool is clearly needed but the model returns none/no call, a
           format= schema whose tool_name is an enum of the available tools ONLY
           (no "none") makes the sampler unable to refuse -> a tool selection is
           guaranteed. This forces an action without depending on tool_choice
           (whose Ollama support is uneven).

  Layer 2b transpile_intent
           The deterministic floor: if the model still narrates code/commands in
           prose instead of calling a tool, recover (filename, code) and
           synthesize write_file / execute_code by parsing, not by asking. Does
           not depend on the model at all.

  model_supports_native_tools is the capability gate (the think=True/400 cycle's
  option A): native function-calling for models that support it, with the
  existing format= path as the fallback for those that do not.

These functions take duck-typed registry objects (anything exposing .name,
.description, .parameters, and per-parameter .type/.description/.required) so the
suite can exercise them without importing the full registry / ollama chain.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Layer 1 -- native Ollama function-calling
# ---------------------------------------------------------------------------
_JSON_TYPE = {
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}


def native_tool_schemas(tools: list) -> list[dict]:
    """Build Ollama native function-call schemas from registry tool definitions."""
    schemas = []
    for tool in tools:
        properties: dict[str, dict] = {}
        required: list[str] = []
        for pname, pdef in getattr(tool, "parameters", {}).items():
            ptype = _JSON_TYPE.get(getattr(pdef, "type", "string"), "string")
            prop: dict[str, Any] = {
                "type": ptype,
                "description": getattr(pdef, "description", "") or "",
            }
            if ptype == "array":
                prop["items"] = {"type": "string"}
            properties[pname] = prop
            if getattr(pdef, "required", True):
                required.append(pname)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return schemas


def _get(obj: Any, key: str) -> Any:
    """Dict-or-object accessor (the codebase's dict-vs-object class)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def parse_native_tool_calls(response: Any) -> list[tuple[str, dict]]:
    """Extract (tool_name, arguments) pairs from an Ollama chat response.

    Handles both the object and dict response forms, and arguments delivered
    either as a dict or as a JSON string.
    """
    message = _get(response, "message")
    raw_calls = _get(message, "tool_calls") or []
    out: list[tuple[str, dict]] = []
    for call in raw_calls:
        fn = _get(call, "function")
        name = _get(fn, "name")
        if not name:
            continue
        args = _get(fn, "arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append((name, args))
    return out


# ---------------------------------------------------------------------------
# Layer 2a -- enum-forced decision (guaranteed selection, no "none")
# ---------------------------------------------------------------------------
def forced_decision_schema(tool_names: list[str]) -> dict:
    """A JSON schema for a forced tool decision: tool_name is an enum of the
    available tools (no "none"), so the format= sampler cannot decline.

    Returned as a plain JSON schema (not a Pydantic class) so it can be passed
    straight to ollama.chat(format=...) without constructing a dynamic model.
    """
    if not tool_names:
        raise ValueError("forced_decision_schema requires at least one tool")
    return {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string", "enum": list(tool_names)},
            "arguments": {"type": "object"},
            "reasoning": {"type": "string"},
        },
        "required": ["tool_name", "arguments"],
    }


# ---------------------------------------------------------------------------
# Layer 2b -- deterministic intent transpiler (the salvage floor)
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_EXT = (
    r"(?:py|js|ts|sh|json|ya?ml|txt|md|html|css|c|cpp|h|hpp|rs|go|java|sql|toml)"
)
_FNAME_PATTERNS = [
    re.compile(r"[Ff]ich(?:ier|er)[ :]+`([\w./-]+\." + _EXT + r")`"),
    re.compile(r"[Ff]ile[ :]+`([\w./-]+\." + _EXT + r")`"),
    re.compile(r"`([\w./-]+\." + _EXT + r")`"),
    re.compile(r"\b([\w-]+\." + _EXT + r")\b"),
]
_RUN_INTENT = re.compile(
    r"\b(ex[eé]cut|execute|run|lance|then run)\b", re.IGNORECASE
)
_LANG_EXT = {
    "python": "py", "py": "py", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "bash": "sh", "sh": "sh", "shell": "sh",
}


def transpile_intent(
    response: str,
    user_message: str = "",
    available: set[str] | None = None,
) -> list[tuple[str, dict]]:
    """Recover tool calls from a prose response that narrated code/commands.

    Returns synthesized (tool_name, arguments) pairs. Only emits calls for tools
    that are in ``available`` (when provided), so the salvage never invents an
    unavailable tool.
    """
    if not response or "```" not in response:
        return []
    can = (lambda name: True) if available is None else (lambda name: name in available)

    blocks = _FENCE.findall(response)
    if not blocks:
        return []

    head = response[: response.find("```")]
    fname = None
    for pat in _FNAME_PATTERNS:
        m = pat.search(head) or pat.search(response)
        if m:
            fname = m.group(1)
            break

    wants_run = bool(_RUN_INTENT.search(user_message) or _RUN_INTENT.search(response))
    calls: list[tuple[str, dict]] = []
    for i, (lang, code) in enumerate(blocks):
        code = code.rstrip("\n") + "\n"
        if i == 0 and fname:
            name = fname
        else:
            name = f"snippet_{i + 1}.{_LANG_EXT.get(lang.lower(), 'txt')}"
        if can("write_file"):
            calls.append(("write_file", {"filename": name, "content": code}))
        if wants_run and name.endswith(".py") and can("execute_code"):
            calls.append(("execute_code", {"filename": name}))
    return calls


# ---------------------------------------------------------------------------
# Capability gate (option A) -- native tools for models that support them
# ---------------------------------------------------------------------------
# Families known to support Ollama native function-calling well. Used as the
# default when a model profile carries no explicit `native_tools` capability.
_NATIVE_TOOL_FAMILIES = (
    "qwen2.5", "qwen3", "qwen2", "llama3.1", "llama3.2", "llama3.3",
    "mistral-nemo", "mistral-small", "mistral-large", "firefunction",
    "command-r", "hermes3", "granite3",
)


def model_supports_native_tools(model: str, capability_lookup=None) -> bool:
    """Whether ``model`` should use the native function-calling path.

    If ``capability_lookup`` is given (the model-profile system -- option A), it
    is consulted first: a callable(model) -> bool | None. None means "no opinion"
    and falls through to the name heuristic.
    """
    if capability_lookup is not None:
        try:
            verdict = capability_lookup(model)
            if verdict is not None:
                return bool(verdict)
        except Exception:
            pass
    name = (model or "").lower()
    return any(fam in name for fam in _NATIVE_TOOL_FAMILIES)


# ---------------------------------------------------------------------------
# Lot 3 -- argument auto-repair (tolerate near-miss argument names)
# ---------------------------------------------------------------------------
# A model often emits the right intent under a slightly wrong key ("path" for
# "filename", "code" for "content"). Rather than fail the call on a missing
# required parameter, remap a provided alias onto the schema name. Alias keys
# are compared in normalized form (lowercased, separators stripped).
_ARG_ALIASES = {
    "filename": (
        "path", "file", "filepath", "name", "fname", "target", "outputpath",
        "outfile", "destination",
    ),
    "content": (
        "text", "code", "data", "body", "source", "contents", "filecontent",
        "filecontents",
    ),
    "command": ("cmd", "shell", "bash", "script", "commandline"),
    "code": ("source", "script", "snippet", "program", "content"),
    "path": ("filepath", "dir", "directory", "folder", "location"),
    "query": ("q", "search", "searchquery", "question", "prompt"),
    "directory": ("dir", "folder", "path", "location"),
}


def _norm(s: str) -> str:
    return s.lower().replace("_", "").replace("-", "").replace(" ", "")


def repair_arguments(param_names, arguments: dict) -> dict:
    """Remap near-miss argument keys onto the schema parameter names.

    Non-destructive: provided keys are kept; a schema name absent from
    ``arguments`` is filled from a normalized exact match or an alias, when one
    is present. Unknown extra keys are left untouched (the caller's resolve step
    drops them).
    """
    if not isinstance(arguments, dict):
        return {}
    out = dict(arguments)
    provided = {_norm(k): k for k in arguments}
    for pname in param_names:
        if pname in out:
            continue
        npname = _norm(pname)
        cand = provided.get(npname)
        if cand is not None and cand != pname:
            out[pname] = arguments[cand]
            continue
        for alias in _ARG_ALIASES.get(pname, ()):
            key = provided.get(_norm(alias))
            if key is not None:
                out[pname] = arguments[key]
                break
    return out
