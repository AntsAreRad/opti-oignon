#!/usr/bin/env python3
"""The agent loop.

A multi-turn streaming loop (ODYSSEUS_SPEC.md Section 5.1) with a configurable
round cap whose default is the Odysseus reference value of 20. Each round
streams a model response, resolves any tool calls (``dispatch``), executes them
through the disposable bwrap sandbox seam (never the host), observes the
results -- wrapped as untrusted data (``untrusted_context``) -- and continues
until the model returns a final answer with no tool calls or the cap is reached.
A bounded verifier mirrors the reference, capped at ``_VERIFIER_MAX_ROUNDS`` so
it never loops forever.

The loop never raises into the conversation path: a model-stream failure, a
parser miss, a refused tool, or a tool error all become observations, not
exceptions. The model client and the sandbox are injected, so the loop is
isolatable and its runtime tests collect without the backend. The model client
is either an object with ``stream(messages, tools=None)`` or a callable with the
same signature, yielding Ollama-shaped chunks (``{"message": {"content": ...,
"tool_calls": ...}}``); a single response dict is also accepted.

The memory working block (``retrieval.working_block``), which the memory layer
leaves unwrapped, is consumed here through ``untrusted_context`` and injected as
an untrusted user-role message before the task.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from opti_oignon.agent import allowlists, dispatch, untrusted_context
from opti_oignon.agent import tools as agent_tools

# Guarded PyYAML import for the local hardening loader (the
# load_diagnostics_config precedent: config_loader.py is deliberately not
# edited; a missing dependency falls back to the in-module defaults).
try:
    import yaml as _yaml
except Exception:  # pragma: no cover - PyYAML is a pinned dependency
    _yaml = None

# Guarded governor seam (AGT_SPEC 6.6): when the run's executor path holds a
# thread-local admission ticket, the loop derives its budgets from the
# admitted num_ctx; absent module, absent ticket, or a non-admission action,
# the static floors hold (honest provenance, never a fabricated claim).
try:
    from opti_oignon.resource_governor import get_active_ticket as _get_active_ticket
except Exception:  # pragma: no cover - governor is optional at import time
    _get_active_ticket = None

logger = logging.getLogger(__name__)

# Module conventions (Theme 3).
checkpoint_before_apply = True
FEATURE_AVAILABLE = True

# Round caps. The default mirrors the Odysseus reference; the hard ceiling
# guards against an accidental unbounded configuration.
MAX_AGENT_ROUNDS = 20
_HARD_ROUND_CEILING = 1000
_VERIFIER_MAX_ROUNDS = 2

# Bounded subagent cap. The effective child cap is
# min(requested, TASK_CHILD_CAP, parent_rounds_remaining - 1), and the rounds
# a child uses are DEBITED from the parent budget, so a run's total work stays
# under the round cap regardless of task usage.
TASK_CHILD_CAP = 6

# Stop reasons.
STOP_DONE = "done"
STOP_MAX_ROUNDS = "max_rounds"
STOP_ERROR = "error"
STOP_CANCELLED = "cancelled"
# Structured doom-loop abort.
STOP_DOOM_LOOP = "doom_loop"

# ---------------------------------------------------------------------------
# Loop-hardening constants.
#
# The static values below are the spec defaults and the FLOORS of the
# governor-fed form (6.6): a fed derivation can only raise them. The
# per-observation caps are config-overridable with hard floors of their own
# (6.1); no configuration can go below those floors.
# ---------------------------------------------------------------------------
AGENT_OBS_MAX_BYTES = 16384
AGENT_OBS_MAX_LINES = 256
_OBS_BYTES_FLOOR = 4096
_OBS_LINES_FLOOR = 64
AGENT_ROUND_OBS_BUDGET = 49152
# Over-budget observations in a round truncate HARDER, never drop (6.1).
_OBS_OVERBUDGET_BYTES = 1024
_OBS_OVERBUDGET_LINES = 16
PRUNE_TRIGGER_CHARS = 98304
PRUNE_TARGET_CHARS = 65536
PRUNE_PROTECT_ROUNDS = 3
DOOM_LOOP_THRESHOLD = 3
_DOOM_THRESHOLD_FLOOR = 2
OBS_FRACTION_DEFAULT = 0.35
_SPILL_DIR = ".agent/spill"
_BASH_ELISION_MARKER = "\n[... output elided ...]\n"

# The 6.5 max-steps reminder: trusted (ours, not external content), appended
# once per run when parent rounds remaining falls to 2.
MAX_STEPS_REMINDER = (
    "[2 rounds remain; converge: finish the task or state what is done and "
    "what remains]"
)

HARDENING_DEFAULTS: dict[str, Any] = {
    "obs_max_bytes": AGENT_OBS_MAX_BYTES,
    "obs_max_lines": AGENT_OBS_MAX_LINES,
    "round_obs_budget": AGENT_ROUND_OBS_BUDGET,
    "prune_trigger_chars": PRUNE_TRIGGER_CHARS,
    "prune_target_chars": PRUNE_TARGET_CHARS,
    "prune_protect_rounds": PRUNE_PROTECT_ROUNDS,
    "summarize_compaction": False,
    "doom_loop_threshold": DOOM_LOOP_THRESHOLD,
    "obs_fraction": OBS_FRACTION_DEFAULT,
}


def load_hardening_config() -> dict[str, Any]:
    """The ``hardening:`` block of agent/config.yaml, defaults applied.

    A tiny guarded reader local to this module, on the
    ``load_diagnostics_config`` precedent (config_loader.py deliberately not
    edited): a missing file, missing PyYAML, a malformed file, or a missing
    block all yield ``HARDENING_DEFAULTS``. The 6.1 hard floors are enforced
    here, so no configuration can lower a cap below them, and the prune
    trigger is kept strictly above the prune target.
    """
    cfg = dict(HARDENING_DEFAULTS)
    if _yaml is None:
        return cfg
    try:
        config_path = Path(__file__).parent / "config.yaml"
        raw = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return cfg
    block = raw.get("hardening") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        return cfg

    def _int_key(key: str, floor: int) -> None:
        try:
            cfg[key] = max(floor, int(block.get(key, cfg[key])))
        except Exception:
            logger.debug("hardening config key %s unusable; default kept", key)

    _int_key("obs_max_bytes", _OBS_BYTES_FLOOR)
    _int_key("obs_max_lines", _OBS_LINES_FLOOR)
    _int_key("round_obs_budget", _OBS_BYTES_FLOOR)
    _int_key("prune_target_chars", _OBS_BYTES_FLOOR)
    _int_key("prune_trigger_chars", _OBS_BYTES_FLOOR)
    _int_key("prune_protect_rounds", 1)
    _int_key("doom_loop_threshold", _DOOM_THRESHOLD_FLOOR)
    cfg["summarize_compaction"] = bool(
        block.get("summarize_compaction", cfg["summarize_compaction"])
    )
    try:
        fraction = float(block.get("obs_fraction", cfg["obs_fraction"]))
        cfg["obs_fraction"] = min(1.0, max(0.05, fraction))
    except Exception:
        logger.debug("hardening obs_fraction unusable; default kept")
    # Invariant: pruning must have somewhere to go.
    if cfg["prune_trigger_chars"] <= cfg["prune_target_chars"]:
        cfg["prune_trigger_chars"] = cfg["prune_target_chars"] + _OBS_BYTES_FLOOR
    return cfg


def _derive_budgets(
    cfg: dict[str, Any], admitted_num_ctx: int | None
) -> tuple[int, int, int]:
    """Round budget and prune thresholds, static or governor-fed (6.6).

    The fed form derives ``budget_chars = admitted_num_ctx * 4 *
    obs_fraction`` and sets the round budget and prune trigger from it,
    NEVER below the static floors; the prune target preserves the static
    trigger:target ratio (2:3). Returns (round_budget, prune_trigger,
    prune_target).
    """
    round_budget = int(cfg["round_obs_budget"])
    trigger = int(cfg["prune_trigger_chars"])
    target = int(cfg["prune_target_chars"])
    if admitted_num_ctx is None or admitted_num_ctx <= 0:
        return round_budget, trigger, target
    try:
        budget_chars = int(admitted_num_ctx * 4 * float(cfg["obs_fraction"]))
    except Exception:
        return round_budget, trigger, target
    round_budget = max(round_budget, budget_chars)
    trigger = max(trigger, 2 * budget_chars)
    target = max(target, (2 * trigger) // 3)
    return round_budget, trigger, target


def _ticket_num_ctx() -> int | None:
    """The admitted num_ctx of the thread's active governor ticket, if any.

    Honest provenance (6.6): only a ticket whose action is an admission
    ("admit" / "downsize") feeds the budgets; no governor, no ticket, or a
    refusal yields None and the static floors hold. Never raises.
    """
    if _get_active_ticket is None:
        return None
    try:
        ticket = _get_active_ticket()
    except Exception:
        return None
    if ticket is None:
        return None
    action = getattr(ticket, "action", None)
    num_ctx = getattr(ticket, "num_ctx", None)
    if action not in ("admit", "downsize") or not num_ctx:
        return None
    try:
        return int(num_ctx)
    except Exception:
        return None

# A short verification instruction (trusted, from us) appended for the verifier.
VERIFIER_PROMPT = (
    "Verify the result above against the original task. Check it is correct and "
    "complete. Respond with PASS if it holds or FAIL with a brief reason if it "
    "does not. Use a tool only if you must re-check something."
)


@dataclass
class AgentEvent:
    """A single observable step, surfaced to an optional ``on_event`` callback."""

    kind: str
    round: int
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierResult:
    """The outcome of the bounded verifier subagent."""

    rounds: int
    verdict: str
    observations: list[str] = field(default_factory=list)
    bounded: bool = True


@dataclass
class AgentRunResult:
    """The outcome of a full agent run."""

    final_text: str
    rounds: int
    stop_reason: str
    tool_results: list[dispatch.DispatchResult] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    verifier: VerifierResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_text": self.final_text,
            "rounds": self.rounds,
            "stop_reason": self.stop_reason,
            "tool_results": [r.to_dict() for r in self.tool_results],
            "verifier": (
                {
                    "rounds": self.verifier.rounds,
                    "verdict": self.verifier.verdict,
                    "bounded": self.verifier.bounded,
                }
                if self.verifier is not None
                else None
            ),
        }


# Helpers


def _clamp_rounds(max_rounds: Any) -> int:
    try:
        value = int(max_rounds)
    except Exception:
        value = MAX_AGENT_ROUNDS
    if value < 1:
        value = 1
    return min(value, _HARD_ROUND_CEILING)


def _emit(on_event: Callable[[AgentEvent], None] | None, kind: str, rnd: int, data: dict) -> None:
    if on_event is None:
        return
    try:
        on_event(AgentEvent(kind=kind, round=rnd, data=data))
    except Exception:  # an observer must never break the loop
        logger.debug("on_event callback raised; ignoring", exc_info=True)


def _invoke_model(model_client: Any, messages: list[dict], tools: Any):
    """Call the injected model client's streaming entry point.

    Accepts an object with ``stream`` or a plain callable; tolerates a client
    that does not take a ``tools`` keyword.
    """
    fn = getattr(model_client, "stream", None)
    if fn is None and callable(model_client):
        fn = model_client
    if fn is None:
        raise TypeError("model client has no 'stream' method and is not callable")
    try:
        return fn(messages, tools=tools)
    except TypeError:
        return fn(messages)


def _iter_chunks(result: Any):
    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    try:
        iter(result)
    except TypeError:
        return [result]
    return result


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _consume_stream(result: Any) -> tuple[str, Any]:
    """Accumulate streamed content and capture any native tool calls."""
    parts: list[str] = []
    tool_calls: Any = None
    for chunk in _iter_chunks(result):
        message = _get(chunk, "message")
        if message is None:
            continue
        content = _get(message, "content")
        if content:
            parts.append(content)
        tc = _get(message, "tool_calls")
        if tc:
            tool_calls = tc
    return "".join(parts), tool_calls


def _last_assistant_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "") or ""
    return ""


def _initial_messages(
    system_prompt: str,
    task: str,
    include_memory: bool,
    memory_provider: Callable[..., str] | None,
    memory_query: str | None,
    user_id: str | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if include_memory:
        mem = untrusted_context.memory_untrusted_message(
            memory_query if memory_query is not None else task,
            user_id=user_id,
            provider=memory_provider,
        )
        if mem is not None:
            messages.append(mem)
    messages.append({"role": "user", "content": task})
    return messages


def _observations_message(results: list[dispatch.DispatchResult]) -> dict[str, Any] | None:
    """Wrap a round's tool results as a single untrusted-data user message."""
    items = [("tool", f"{r.tool_name}: {r.observation}") for r in results]
    return untrusted_context.untrusted_message_many(items)


# ---------------------------------------------------------------------------
# Loop-hardening helpers: output caps and spill (6.1), the
# deterministic prune stage and the flag-gated summarize stage (6.2), and the
# doom-loop window (6.3). The verifier deliberately keeps the uncapped
# ``_observations_message`` path: ``_run_verifier`` is unchanged.
# ---------------------------------------------------------------------------


def _utf8_clip(text: str, max_bytes: int) -> str:
    """The leading ``max_bytes`` of ``text``, safe at UTF-8 boundaries."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def _utf8_clip_tail(text: str, max_bytes: int) -> str:
    """The trailing ``max_bytes`` of ``text``, safe at UTF-8 boundaries."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[-max_bytes:].decode("utf-8", errors="ignore")


def _truncate_observation(
    text: str, max_bytes: int, max_lines: int, head_and_tail: bool
) -> tuple[str, bool]:
    """Clip an observation to the byte and line caps (6.1).

    Head by default; bash observations keep head and tail halves around the
    elision marker (errors cluster at both ends). Returns
    ``(clipped, truncated)``.
    """
    lines = text.split("\n")
    if len(text.encode("utf-8")) <= max_bytes and len(lines) <= max_lines:
        return text, False
    if not head_and_tail:
        head = "\n".join(lines[:max_lines])
        return _utf8_clip(head, max_bytes), True
    head_lines = max(1, max_lines // 2)
    tail_lines = max(1, max_lines - head_lines)
    head_budget = max(1, max_bytes // 2)
    tail_budget = max(1, max_bytes - head_budget)
    head_src = "\n".join(lines[:head_lines])
    tail_src = "\n".join(lines[-tail_lines:])
    clipped = (
        _utf8_clip(head_src, head_budget)
        + _BASH_ELISION_MARKER
        + _utf8_clip_tail(tail_src, tail_budget)
    )
    return clipped, True


def _spill_full_output(
    sandbox: Any, spill_counter: dict[str, int], rnd: int, text: str
) -> str | None:
    """Write the full pre-truncation text inside the workspace (6.1).

    An ORDINARY workspace write through the session (the bwrap boundary is
    untouched; ``.agent/`` is agent-internal and the copy-out diff excludes
    it by the manifest rule). Returns the workspace-relative spill path, or
    None when no session can take the write (handler-only runs, an inactive
    session, an error result) -- the stub then omits the path, per the spec.
    Never raises.
    """
    create = getattr(sandbox, "create_file", None)
    if not callable(create):
        return None
    spill_counter["k"] += 1
    rel = f"{_SPILL_DIR}/obs_{rnd}_{spill_counter['k']}.txt"
    try:
        out = create(rel, text)
    except Exception:
        logger.debug("spill write failed; the stub omits the path", exc_info=True)
        return None
    if isinstance(out, str) and out and not out.startswith("Error"):
        return rel
    return None


def _capped_observations_message(
    results: list[dispatch.DispatchResult],
    *,
    cfg: dict[str, Any],
    round_budget: int,
    rnd: int,
    sandbox: Any,
    spill_counter: dict[str, int],
) -> tuple[dict[str, Any] | None, list[str]]:
    """The 6.1 cap-and-spill transform, applied BEFORE untrusted wrapping.

    Builds exactly what ``_observations_message`` would build, over clipped
    texts. The ``DispatchResult`` objects are never mutated (events and
    ``tool_results`` keep the full observation; only the transcript copy is
    clipped). Later observations in an over-budget round truncate harder
    (the over-budget floor), never drop. Returns ``(message_or_None,
    spill_paths)``.
    """
    items: list[tuple[str, str]] = []
    spills: list[str] = []
    remaining = int(round_budget)
    for r in results:
        text = r.observation or ""
        if remaining >= int(cfg["obs_max_bytes"]):
            max_bytes = int(cfg["obs_max_bytes"])
            max_lines = int(cfg["obs_max_lines"])
        else:
            max_bytes = max(_OBS_OVERBUDGET_BYTES, remaining)
            max_lines = _OBS_OVERBUDGET_LINES
        clipped, truncated = _truncate_observation(
            text, max_bytes, max_lines, head_and_tail=(r.tool_name == "bash")
        )
        if truncated:
            orig_bytes = len(text.encode("utf-8"))
            orig_lines = len(text.split("\n"))
            spill_path = _spill_full_output(sandbox, spill_counter, rnd, text)
            if spill_path is not None:
                spills.append(spill_path)
                stub = (
                    f"[truncated: {orig_bytes} bytes, {orig_lines} lines; "
                    f"full output: {spill_path}]"
                )
            else:
                stub = f"[truncated: {orig_bytes} bytes, {orig_lines} lines]"
            joiner = "" if (not clipped or clipped.endswith("\n")) else "\n"
            clipped = clipped + joiner + stub
        remaining = max(0, remaining - len(clipped.encode("utf-8")))
        items.append(("tool", f"{r.tool_name}: {clipped}"))
    return untrusted_context.untrusted_message_many(items), spills


def _estimate_transcript_chars(messages: list[dict[str, Any]]) -> int:
    """Chars-based transcript size estimate (6.2): sum of content lengths.

    Honest about being an estimate, monotone, and dependency-free.
    """
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def _prune_transcript(
    messages: list[dict[str, Any]],
    prunable: list[dict[str, Any]],
    *,
    current_round: int,
    trigger: int,
    target: int,
    protect_rounds: int,
    on_event: Callable[[AgentEvent], None] | None,
) -> None:
    """The deterministic, model-free prune stage (6.2).

    Walks the loop's own observation registry oldest-first, replacing each
    tool-output message outside the protect window with the one-line stub,
    until the estimate falls under ``target`` or only protected content
    remains. The system prompt, the original task, the memory block and any
    ``skill_message`` are protected BY CONSTRUCTION: only messages this loop
    registered as observations are ever candidates; the last
    ``protect_rounds`` rounds stay verbatim. Spilled originals remain
    re-readable in the workspace, so pruning a spilled observation loses
    nothing.
    """
    estimate = _estimate_transcript_chars(messages)
    if estimate <= trigger:
        return
    before = estimate
    pruned = 0
    for entry in prunable:
        if estimate <= target:
            break
        if entry.get("pruned"):
            continue
        if entry["round"] > current_round - protect_rounds:
            continue
        index = entry["index"]
        spill = entry.get("spill") or "none"
        stub = (
            f"[pruned observation, round {entry['round']}, "
            f"{entry['bytes']} bytes; spill: {spill}]"
        )
        old_content = messages[index].get("content")
        old_len = len(old_content) if isinstance(old_content, str) else 0
        messages[index]["content"] = stub
        entry["pruned"] = True
        pruned += 1
        estimate += len(stub) - old_len
    if pruned:
        _emit(
            on_event,
            "compaction",
            current_round,
            {"pruned": pruned, "before": before, "after": estimate},
        )


def _summarize_pruned_span(
    model_client: Any,
    messages: list[dict[str, Any]],
    prunable: list[dict[str, Any]],
    on_event: Callable[[AgentEvent], None] | None,
    current_round: int,
) -> None:
    """Stage two of 6.2: flag-gated summarization of the pruned span.

    OFF by default (``summarize_compaction: false``); called only when the
    flag is on and pruning alone did not reach the target. One model call
    summarizes the pruned stubs into a single UNTRUSTED summary message (a
    model output is data) inserted at the prune boundary. The container
    asserts this structure with a fake client; the live behaviour -- summary
    quality, token economics -- is HOST-ASSURED and is what the Lot 3
    harness measures before anyone turns the flag on. Never raises.
    """
    span = [e for e in prunable if e.get("pruned") and not e.get("summarized")]
    if not span:
        return
    try:
        body = "\n".join(
            messages[e["index"]].get("content", "") or "" for e in span
        )
        prompt = (
            "Summarize the pruned tool observations below in at most five "
            "sentences, preserving file paths and error names.\n\n" + body
        )
        content, _calls = _consume_stream(
            _invoke_model(model_client, [{"role": "user", "content": prompt}], None)
        )
    except Exception:
        logger.debug("summarize compaction failed; skipping", exc_info=True)
        return
    if not content:
        return
    boundary = max(e["index"] for e in span)
    messages.insert(
        boundary + 1,
        untrusted_context.tool_output_message("[compaction summary] " + content),
    )
    for entry in prunable:
        if entry["index"] > boundary:
            entry["index"] += 1
        if entry.get("pruned"):
            entry["summarized"] = True
    _emit(on_event, "compaction", current_round, {"summarized": len(span)})


def _doom_signature(call: dispatch.ToolCall) -> str:
    """The 6.3 window key: tool name plus canonical JSON of the arguments."""
    try:
        args = json.dumps(call.arguments or {}, sort_keys=True, default=str)
    except Exception:
        args = str(call.arguments)
    return f"{call.name}:{args}"


def _new_doom_state() -> dict[str, Any]:
    """A fresh rolling window: identical READS count; refusals never enter."""
    return {
        "sig": None,
        "count": 0,
        "asked": set(),
        "corrected": set(),
        "approved": set(),
    }


def _doom_should_abort(
    state: dict[str, Any], threshold: int, calls: list[dispatch.ToolCall]
) -> str | None:
    """The no-approval fourth-identical abort, checked BEFORE dispatch (6.3).

    Only a signature that already received the one corrective observation
    can abort here; the approval branch never reaches this (a granted
    approval exempts the signature, a denial aborted at the trip). Returns
    the offending tool name, or None.
    """
    sig = state["sig"]
    if sig is None or state["count"] < threshold:
        return None
    if sig not in state["corrected"] or sig in state["approved"]:
        return None
    for call in calls:
        if _doom_signature(call) == sig:
            return call.name
        return None  # a different leading call breaks consecutiveness
    return None


def _doom_update(
    state: dict[str, Any],
    threshold: int,
    calls: list[dispatch.ToolCall],
    results: list[dispatch.DispatchResult],
    *,
    approval_fn: Callable[[str, str, dict], bool] | None,
    conversation_id: str,
    messages: list[dict[str, Any]],
    on_event: Callable[[AgentEvent], None] | None,
    rnd: int,
) -> bool:
    """Update the rolling window over the round's EXECUTED calls (6.3).

    The window counts identical consecutive ``(tool, canonical args)``
    entries per executed call; identical reads count; a different executed
    call resets; refusals neither count nor reset. At the threshold: with
    ``approval_fn`` present a synthetic approval is asked ONCE per signature
    (the Bulbe-shaped human interrupt) -- deny or an exception aborts the
    run; without it, ONE corrective untrusted observation is injected, and
    a further identical executed call aborts (the in-batch safety net behind
    ``_doom_should_abort``). Returns True when the run must abort.
    """
    for call, result in zip(calls, results):
        if not getattr(result, "executed", False):
            continue
        sig = _doom_signature(call)
        if sig == state["sig"]:
            state["count"] += 1
        else:
            state["sig"] = sig
            state["count"] = 1
        if sig in state["approved"]:
            continue
        if (
            approval_fn is None
            and state["count"] > threshold
            and sig in state["corrected"]
        ):
            _emit(on_event, "aborted", rnd, {"reason": "doom_loop", "tool": call.name})
            return True
        if state["count"] == threshold:
            if approval_fn is not None:
                if sig in state["asked"]:
                    continue
                state["asked"].add(sig)
                message = (
                    f"doom_loop: the agent repeated {call.name} with identical "
                    f"arguments {threshold} times; continue?"
                )
                try:
                    approved = bool(
                        approval_fn(
                            conversation_id,
                            "doom_loop",
                            {"tool": call.name, "message": message},
                        )
                    )
                except Exception:
                    approved = False  # fail-secure, the evaluate() idiom
                if approved:
                    state["approved"].add(sig)
                    continue
                _emit(
                    on_event, "aborted", rnd, {"reason": "doom_loop", "tool": call.name}
                )
                return True
            if sig not in state["corrected"]:
                state["corrected"].add(sig)
                messages.append(
                    untrusted_context.tool_output_message(
                        f"[doom-loop detected: {call.name} repeated {threshold} "
                        f"times with identical arguments; vary the approach or "
                        f"conclude]"
                    )
                )
    return False


# Helpers: the todo binding and the bounded task subagent.


def _native_tool_names(tools: Any) -> set[str]:
    """Tool names advertised by a native function-calling tools list."""
    names: set[str] = set()
    if not tools:
        return names
    try:
        for entry in tools:
            fn = _get(entry, "function")
            name = _get(fn, "name") if fn is not None else None
            if not name:
                name = _get(entry, "name")
            if name:
                names.add(str(name))
    except TypeError:
        return names
    return names


def _resolved_mode(mode: str | None) -> str:
    """Resolve a mode argument the allowlists way (fail-secure to Bulbe)."""
    if mode in allowlists.VALID_MODES:
        return mode  # type: ignore[return-value]
    if mode is None:
        return allowlists.current_mode()
    return allowlists.MODE_BULBE


def _bind_todo_handler(
    effective_handlers: dict[str, Callable[[dict[str, Any]], Any]],
    exposed_names: set[str],
    on_event: Callable[[AgentEvent], None] | None,
    round_ref: dict[str, int],
) -> None:
    """Ensure a per-run todo handler exists and emits ``todo_updated``.

    Gated on the run actually exposing todo (its schema in the advertised
    tools, or a caller-injected handler), so runs without todo keep the
    prior behaviour byte for byte. A caller-injected handler is kept; its
    ``on_update`` is bound only when unset.
    """
    if (
        agent_tools.TOOL_TODO not in exposed_names
        and agent_tools.TOOL_TODO not in effective_handlers
    ):
        return
    handler = effective_handlers.get(agent_tools.TOOL_TODO)
    if handler is None:
        handler = agent_tools.make_todo_handler()
        effective_handlers[agent_tools.TOOL_TODO] = handler

    def _todo_update(payload: dict[str, Any]) -> None:
        _emit(on_event, "todo_updated", round_ref.get("round", 0), dict(payload))

    try:
        if getattr(handler, "on_update", None) is None:
            handler.on_update = _todo_update  # type: ignore[attr-defined]
    except Exception:  # a foreign handler may refuse attributes; not fatal
        logger.debug("todo handler does not accept on_update binding", exc_info=True)


def _child_task_surface() -> tuple[list[dict[str, Any]], str]:
    """The child subagent tool surface: exactly the sandbox set (5.4).

    Native schemas plus the prompt section for the seven sandboxed tools; no
    handler map (the empty map is the hard gate: any non-sandbox call the
    child emits falls through to dispatch's safe no-executor observation, and
    a nested task can never run).
    """
    schemas = [s for s in agent_tools.ALL_SCHEMAS if s.sandboxed]
    ts = agent_tools.ToolSet(mode="task-child", schemas=schemas, tool_handlers={})
    return ts.native_tools(), ts.system_prompt_section()


def _run_task_child(
    description: str,
    prompt: str,
    *,
    model_client: Any,
    child_cap: int,
    mode: str | None,
    conversation_id: str,
    sandbox: Any,
    approval_fn: Callable[[str, str, dict], bool] | None,
    on_event: Callable[[AgentEvent], None] | None,
    hardening_cfg: dict[str, Any] | None = None,
    round_budget: int | None = None,
    spill_counter: dict[str, int] | None = None,
) -> tuple[str, int]:
    """Run one bounded child task; returns (final_text, rounds_used).

    The child shares the parent's SandboxToolSession, mode and approval_fn
    (every child sandbox call rides the same Bulbe per-call approval), and
    its events are re-emitted with a ``task`` marker so the panel can nest
    them. The child's observations ride the same 6.1 caps and spill
    (the spill counter is shared with the parent, so paths never collide);
    the doom-loop window and the prune stage stay PARENT-ONLY by design --
    the child is already bounded by ``child_cap`` and debited from the
    parent budget. Never raises.
    """
    label = description or "subtask"

    def child_emit(kind: str, rnd: int, data: dict[str, Any]) -> None:
        payload = dict(data)
        payload["task"] = label
        _emit(on_event, kind, rnd, payload)

    native, prompt_section = _child_task_surface()
    framing = (
        "You are a focused sub-task agent. Complete ONLY the task below and "
        "reply with your result as plain text.\n\n" + prompt_section
    )
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": framing},
        {"role": "user", "content": prompt},
    ]
    final_text = ""
    rounds = 0
    while rounds < child_cap:
        rounds += 1
        child_emit("round_start", rounds, {})
        try:
            content, native_calls = _consume_stream(_invoke_model(model_client, msgs, native))
        except Exception as exc:
            final_text = f"[task model error] {exc}"
            child_emit("error", rounds, {"error": str(exc)})
            break
        msgs.append({"role": "assistant", "content": content})
        child_emit("model_output", rounds, {"content": content})
        try:
            results, _path = dispatch.dispatch_round(
                {"message": {"content": content, "tool_calls": native_calls}},
                mode=mode,
                conversation_id=conversation_id,
                sandbox=sandbox,
                approval_fn=approval_fn,
                tool_handlers={},
            )
        except Exception:  # dispatch is built not to raise; defensive
            results = []
        if not results:
            final_text = content
            break
        if hardening_cfg is not None:
            obs_msg, _spills = _capped_observations_message(
                results,
                cfg=hardening_cfg,
                round_budget=(
                    round_budget
                    if round_budget is not None
                    else int(hardening_cfg["round_obs_budget"])
                ),
                rnd=rounds,
                sandbox=sandbox,
                spill_counter=(
                    spill_counter if spill_counter is not None else {"k": 0}
                ),
            )
        else:
            obs_msg = _observations_message(results)
        if obs_msg is not None:
            msgs.append(obs_msg)
        for r in results:
            child_emit("tool_result", rounds, r.to_dict())
    else:
        final_text = final_text or _last_assistant_text(msgs)
    return final_text, rounds


def _dispatch_with_tasks(
    calls: list[dispatch.ToolCall],
    *,
    model_client: Any,
    cap: int,
    rounds: int,
    mode: str | None,
    conversation_id: str,
    sandbox: Any,
    approval_fn: Callable[[str, str, dict], bool] | None,
    tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] | None,
    on_event: Callable[[AgentEvent], None] | None,
    hardening_cfg: dict[str, Any] | None = None,
    round_budget: int | None = None,
    spill_counter: dict[str, int] | None = None,
) -> tuple[list[dispatch.DispatchResult], int]:
    """Dispatch a round that contains task calls; returns (results, new_cap).

    Non-task calls take the normal per-call dispatch (identical semantics to
    ``dispatch_round``, which is resolve + per-call dispatch). Every task
    bound of AGT_SPEC 5.4 is enforced here: depth 1 via the child's empty
    handler map, child_cap = min(requested, TASK_CHILD_CAP,
    parent_rounds_remaining - 1), and the child's rounds debited from the
    parent budget through the returned cap.
    """
    resolved_mode = _resolved_mode(mode)
    results: list[dispatch.DispatchResult] = []
    for call in calls:
        if call.name != agent_tools.TOOL_TASK:
            results.append(
                dispatch.dispatch_tool_call(
                    call,
                    mode=mode,
                    conversation_id=conversation_id,
                    sandbox=sandbox,
                    approval_fn=approval_fn,
                    tool_handlers=tool_handlers,
                )
            )
            continue
        if not allowlists.is_tool_allowed(agent_tools.TOOL_TASK, mode):
            results.append(
                dispatch.DispatchResult(
                    tool_name=call.name,
                    executed=False,
                    observation=f"Tool '{call.name}' is not permitted in {resolved_mode} mode.",
                    reason=allowlists.REASON_NOT_ALLOWED,
                    source=call.source,
                    mode=resolved_mode,
                )
            )
            continue
        args = call.arguments or {}
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            results.append(
                dispatch.DispatchResult(
                    tool_name=call.name,
                    executed=False,
                    observation="task requires a non-empty 'prompt'.",
                    reason=dispatch.REASON_ERROR,
                    source=call.source,
                    mode=resolved_mode,
                )
            )
            continue
        try:
            requested = int(args.get("max_rounds"))
        except Exception:
            requested = TASK_CHILD_CAP
        if requested < 1:
            requested = TASK_CHILD_CAP
        remaining = cap - rounds
        child_cap = min(requested, TASK_CHILD_CAP, remaining - 1)
        if child_cap < 1:
            results.append(
                dispatch.DispatchResult(
                    tool_name=call.name,
                    executed=False,
                    observation="task refused: insufficient round budget remaining.",
                    reason="task_budget_exhausted",
                    source=call.source,
                    mode=resolved_mode,
                )
            )
            continue
        description = str(args.get("description") or "").strip()
        final_text, used = _run_task_child(
            description,
            prompt,
            model_client=model_client,
            child_cap=child_cap,
            mode=mode,
            conversation_id=conversation_id,
            sandbox=sandbox,
            approval_fn=approval_fn,
            on_event=on_event,
            hardening_cfg=hardening_cfg,
            round_budget=round_budget,
            spill_counter=spill_counter,
        )
        cap -= used  # the debit: child rounds come out of the parent budget
        bound_report = f"task used {used} rounds of {child_cap}"
        observation = (final_text + "\n\n" if final_text else "") + bound_report
        results.append(
            dispatch.DispatchResult(
                tool_name=call.name,
                executed=True,
                observation=observation,
                reason=dispatch.REASON_EXECUTED,
                source=call.source,
                mode=resolved_mode,
            )
        )
    return results, cap


# Verdict patterns, compiled once. Failure tokens are matched first and with
# word boundaries, so "incorrect" / "wrong" read as a fail and are never
# mistaken for a pass via the "correct" substring inside "incorrect".
_VERDICT_FAIL_RE = re.compile(r"\b(?:incorrect|wrong|fail(?:ed|ing|s)?)\b", re.IGNORECASE)
_VERDICT_PASS_RE = re.compile(r"\b(?:pass(?:ed|ing|es)?|correct|looks good)\b", re.IGNORECASE)


def _extract_verdict(content: str) -> str:
    text = content or ""
    if _VERDICT_FAIL_RE.search(text):
        return "fail"
    if _VERDICT_PASS_RE.search(text):
        return "pass"
    return "unknown"


def _run_verifier(
    model_client: Any,
    base_messages: list[dict],
    *,
    max_rounds: int = _VERIFIER_MAX_ROUNDS,
    mode: str | None = None,
    conversation_id: str = "",
    sandbox: Any = None,
    approval_fn: Callable[[str, str, dict], bool] | None = None,
    tool_handlers: dict[str, Callable[[dict], Any]] | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
) -> VerifierResult:
    """A bounded verifier subagent that never exceeds the reference cap.

    The cap is the minimum of the requested ``max_rounds`` and
    ``_VERIFIER_MAX_ROUNDS``, so even a larger request cannot loop forever.
    """
    cap = max(1, min(_clamp_rounds(max_rounds), _VERIFIER_MAX_ROUNDS))
    msgs = list(base_messages) + [{"role": "user", "content": VERIFIER_PROMPT}]
    observations: list[str] = []
    verdict = "unknown"
    rounds = 0
    while rounds < cap:
        rounds += 1
        try:
            content, native = _consume_stream(_invoke_model(model_client, msgs, None))
        except Exception as exc:
            observations.append(f"[verifier round {rounds} error] {exc}")
            break
        msgs.append({"role": "assistant", "content": content})
        observations.append(content)
        _emit(on_event, "verifier_output", rounds, {"content": content})
        try:
            results, _ = dispatch.dispatch_round(
                {"message": {"content": content, "tool_calls": native}},
                mode=mode,
                conversation_id=conversation_id,
                sandbox=sandbox,
                approval_fn=approval_fn,
                tool_handlers=tool_handlers,
            )
        except Exception:
            results = []
        if not results:
            verdict = _extract_verdict(content)
            break
        obs_msg = _observations_message(results)
        if obs_msg is not None:
            msgs.append(obs_msg)
    return VerifierResult(rounds=rounds, verdict=verdict, observations=observations, bounded=rounds <= cap)


def run(
    task: str,
    *,
    model_client: Any,
    sandbox: Any = None,
    mode: str | None = None,
    conversation_id: str = "",
    system_prompt: str = "",
    tools: Any = None,
    approval_fn: Callable[[str, str, dict], bool] | None = None,
    tool_handlers: dict[str, Callable[[dict], Any]] | None = None,
    max_rounds: int = MAX_AGENT_ROUNDS,
    include_memory: bool = True,
    memory_provider: Callable[..., str] | None = None,
    memory_query: str | None = None,
    user_id: str | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    verify: bool = False,
    admitted_num_ctx: int | None = None,
) -> AgentRunResult:
    """Run the multi-turn streaming agent loop.

    Streams a model response each round, dispatches any tool calls through the
    sandbox seam, feeds the results back as untrusted observations, and stops
    when the model returns a final answer (no tool calls) or the round cap is
    reached. Never raises: failures become observations or a terminal result.

    Every observation is capped and spilled per
    6.1 before entering the transcript, the deterministic prune stage runs
    when the chars estimate crosses the trigger (6.2), a doom-loop window
    watches executed calls (6.3), and one trusted reminder lands when two
    rounds remain (6.5). ``admitted_num_ctx`` is the 6.6 governor feed: when
    a caller passes the admitted value (or the thread holds an admission
    ticket), the round budget and prune thresholds derive from it, never
    below the static floors; otherwise the static defaults hold.
    """
    cap = _clamp_rounds(max_rounds)
    messages = _initial_messages(
        system_prompt, task, include_memory, memory_provider, memory_query, user_id
    )
    tool_results_all: list[dispatch.DispatchResult] = []
    final_text = ""
    stop_reason = STOP_DONE

    # The per-run handler map. The todo closure (pure session state) is
    # created or bound here, gated on the run actually exposing todo, so the
    # prior paths stay byte-identical.
    effective_handlers: dict[str, Callable[[dict[str, Any]], Any]] = dict(tool_handlers or {})
    round_ref: dict[str, int] = {"round": 0}
    _bind_todo_handler(effective_handlers, _native_tool_names(tools), on_event, round_ref)

    # Per-run hardening state. Budgets are static
    # or governor-fed (6.6); the spill counter is run-global and shared with
    # task children so spill paths never collide; the prunable registry holds
    # the indices of the observation messages this loop appends (everything
    # else -- the system prompt, the task, the memory block, skill messages
    # -- is protected from pruning by construction).
    hardening_cfg = load_hardening_config()
    fed_num_ctx = admitted_num_ctx if admitted_num_ctx is not None else _ticket_num_ctx()
    round_budget, prune_trigger, prune_target = _derive_budgets(hardening_cfg, fed_num_ctx)
    doom_threshold = int(hardening_cfg["doom_loop_threshold"])
    doom_state = _new_doom_state()
    spill_counter: dict[str, int] = {"k": 0}
    prunable: list[dict[str, Any]] = []
    reminder_sent = False

    if model_client is None:
        return AgentRunResult(
            final_text="",
            rounds=0,
            stop_reason=STOP_ERROR,
            tool_results=[],
            messages=messages,
            verifier=None,
        )

    rounds = 0
    while rounds < cap:
        if should_continue is not None and not should_continue():
            stop_reason = STOP_CANCELLED
            final_text = final_text or _last_assistant_text(messages)
            break
        rounds += 1
        round_ref["round"] = rounds
        # One trusted reminder when, counting this
        # round, exactly two rounds remain (the cap is mutable under task
        # debits, so the condition re-evaluates each round; sent once).
        if not reminder_sent and (cap - rounds + 1) == 2:
            reminder_sent = True
            messages.append({"role": "user", "content": MAX_STEPS_REMINDER})
        _emit(on_event, "round_start", rounds, {})
        try:
            content, native_calls = _consume_stream(_invoke_model(model_client, messages, tools))
        except Exception as exc:
            messages.append({"role": "assistant", "content": ""})
            messages.append(
                untrusted_context.tool_output_message(f"[round {rounds} model error] {exc}")
            )
            _emit(on_event, "error", rounds, {"error": str(exc)})
            stop_reason = STOP_ERROR
            break

        messages.append({"role": "assistant", "content": content})
        _emit(on_event, "model_output", rounds, {"content": content})

        response = {"message": {"content": content, "tool_calls": native_calls}}
        try:
            # The loop intercepts task calls before
            # dispatch_round -- the bounded child needs the model client and
            # the round budget that dispatch does not hold. A round without a
            # task call takes the prior path unchanged.
            pre_calls, _pre_path = dispatch.resolve_tool_calls(response)
            # After the one corrective observation, a
            # further identical call aborts BEFORE execution.
            doom_tool = _doom_should_abort(doom_state, doom_threshold, pre_calls)
            if doom_tool is not None:
                _emit(on_event, "aborted", rounds, {"reason": "doom_loop", "tool": doom_tool})
                stop_reason = STOP_DOOM_LOOP
                final_text = final_text or _last_assistant_text(messages)
                break
            if any(c.name == agent_tools.TOOL_TASK for c in pre_calls):
                results, cap = _dispatch_with_tasks(
                    pre_calls,
                    model_client=model_client,
                    cap=cap,
                    rounds=rounds,
                    mode=mode,
                    conversation_id=conversation_id,
                    sandbox=sandbox,
                    approval_fn=approval_fn,
                    tool_handlers=effective_handlers,
                    on_event=on_event,
                    hardening_cfg=hardening_cfg,
                    round_budget=round_budget,
                    spill_counter=spill_counter,
                )
            else:
                results, _path = dispatch.dispatch_round(
                    response,
                    mode=mode,
                    conversation_id=conversation_id,
                    sandbox=sandbox,
                    approval_fn=approval_fn,
                    tool_handlers=effective_handlers,
                )
        except Exception as exc:  # dispatch is built not to raise; defensive
            results = []
            messages.append(
                untrusted_context.tool_output_message(f"[round {rounds} dispatch error] {exc}")
            )

        if not results:
            final_text = content
            stop_reason = STOP_DONE
            _emit(on_event, "done", rounds, {"final_text": final_text})
            break

        tool_results_all.extend(results)
        obs_msg, round_spills = _capped_observations_message(
            results,
            cfg=hardening_cfg,
            round_budget=round_budget,
            rnd=rounds,
            sandbox=sandbox,
            spill_counter=spill_counter,
        )
        if obs_msg is not None:
            messages.append(obs_msg)
            prunable.append(
                {
                    "index": len(messages) - 1,
                    "round": rounds,
                    "bytes": len((obs_msg.get("content") or "").encode("utf-8")),
                    "spill": ", ".join(round_spills) if round_spills else None,
                }
            )
        for r in results:
            _emit(on_event, "tool_result", rounds, r.to_dict())
        if _doom_update(
            doom_state,
            doom_threshold,
            pre_calls,
            results,
            approval_fn=approval_fn,
            conversation_id=conversation_id,
            messages=messages,
            on_event=on_event,
            rnd=rounds,
        ):
            stop_reason = STOP_DOOM_LOOP
            final_text = final_text or _last_assistant_text(messages)
            break
        _prune_transcript(
            messages,
            prunable,
            current_round=rounds,
            trigger=prune_trigger,
            target=prune_target,
            protect_rounds=int(hardening_cfg["prune_protect_rounds"]),
            on_event=on_event,
        )
        if (
            hardening_cfg.get("summarize_compaction")
            and _estimate_transcript_chars(messages) > prune_target
        ):
            _summarize_pruned_span(model_client, messages, prunable, on_event, rounds)
    else:
        stop_reason = STOP_MAX_ROUNDS
        final_text = final_text or _last_assistant_text(messages)

    verifier_result = None
    if verify and stop_reason == STOP_DONE:
        verifier_result = _run_verifier(
            model_client,
            messages,
            max_rounds=_VERIFIER_MAX_ROUNDS,
            mode=mode,
            conversation_id=conversation_id,
            sandbox=sandbox,
            approval_fn=approval_fn,
            tool_handlers=effective_handlers,
            on_event=on_event,
        )

    return AgentRunResult(
        final_text=final_text,
        rounds=rounds,
        stop_reason=stop_reason,
        tool_results=tool_results_all,
        messages=messages,
        verifier=verifier_result,
    )
