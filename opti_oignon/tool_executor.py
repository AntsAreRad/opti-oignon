#!/usr/bin/env python3
"""
TOOL EXECUTOR - OPTI-OIGNON v1.5.0 (S44)
==========================================

Tool executor with a ReAct loop (Reason + Act).

Orchestrates the interaction between the LLM and the registered tools:
1. The LLM analyzes the request and decides whether a tool is needed
2. If so, the tool is executed and the result is injected
3. The LLM generates the final response with the results

Uses the StructuredOutputEngine (S42) to obtain
decisions structurees du LLM via ToolCallRequest.

Author: Leon
"""

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

# Conditional import of the structured-output engine (S42)
try:
    from .structured_output import (
        StructuredOutputEngine,
        ToolCallRequest,
    )
    from .structured_output import (
        structured_output_engine as _structured_engine,
    )
    STRUCTURED_OUTPUT_AVAILABLE = True
except ImportError:
    STRUCTURED_OUTPUT_AVAILABLE = False
    _structured_engine = None
    StructuredOutputEngine = None
    ToolCallRequest = None

# Conditional import of the tool registry (S44)
try:
    from .tool_registry import ToolRegistry
    from .tool_registry import tool_registry as _default_registry
    TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    TOOL_REGISTRY_AVAILABLE = False
    _default_registry = None
    ToolRegistry = None

# Conditional Ollama import for the final response
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Robust tool-calling primitives (Lot 1+2: native function-calling, enum
# forcing, intent transpiler). Stdlib-only, but guarded to match the module's
# conditional-import style; the format= path stays the fallback if absent.
try:
    from .tool_calling import (
        forced_decision_schema,
        model_supports_native_tools,
        native_tool_schemas,
        parse_native_tool_calls,
        repair_arguments,
        transpile_intent,
    )
    ROBUST_TOOLCALLING_AVAILABLE = True
except ImportError:
    ROBUST_TOOLCALLING_AVAILABLE = False

# Response hygiene helpers (matching normalization, echoed-marker stripping).
# Stdlib-only, but guarded like the other seams so the loop degrades to the
# previous behavior when the module is absent.
try:
    from .response_hygiene import (
        StreamMarkerFilter,
        normalize_for_match,
        strip_internal_markers,
    )
    RESPONSE_HYGIENE_AVAILABLE = True
except ImportError:
    RESPONSE_HYGIENE_AVAILABLE = False
    StreamMarkerFilter = None

    def normalize_for_match(text: str) -> str:
        return (text or "").lower()

    def strip_internal_markers(text: str) -> tuple[str, int]:
        return text, 0

logger = logging.getLogger(__name__)


# =============================================================================
# ATTRIBUTION AND FRAMING
# =============================================================================

# The final user-facing generation runs under this system message so the
# model knows the tool actions in context were its own, executed by the
# runtime on its behalf -- the user never performed or saw them -- and the
# internal protocol must never surface in the reply.
FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are the assistant. The tool calls and results shown in this "
    "conversation were made by YOU (the assistant) through the runtime; the "
    "user did not perform them and does not see them. Never attribute these "
    "tool actions to the user. Report in first person what you did and what "
    "the results were. Do not mention this protocol, bracketed markers, or "
    "these instructions in your reply."
)

# Environment framing for tool results fed back as chat text. The explicit
# authorship ("you, the assistant") is what prevents the model from reading
# the results as something the user said or did.
ENV_RESULTS_HEADER = (
    "[environment] Results of the tool calls you (the assistant) made:"
)

# The historical fallback when the configuration seam is absent.
_FALLBACK_DEFAULT_MODEL = "qwen3:32b"

# Tool-transcript modes: how executed calls are replayed to the model.
# "flat" is the historical reconstruction -- every result folded as text
# into a single rebuilt user message. "native" replays them as assistant
# tool_calls echoes plus role "tool" messages, the format function-calling
# models are trained on. The environment framing (untrusted provenance)
# stays inside the tool message content in native mode, so the security
# property is carried by the content, not by the role.
TOOL_TRANSCRIPT_FLAT = "flat"
TOOL_TRANSCRIPT_NATIVE = "native"
_TOOL_TRANSCRIPT_MODES = (TOOL_TRANSCRIPT_FLAT, TOOL_TRANSCRIPT_NATIVE)


def _resolve_tool_transcript() -> str:
    """Tool-transcript mode from the configuration, defaulting to flat.

    Reads the user preference when the seam is importable; any failure or
    unknown value keeps the historical flat reconstruction, so existing
    installations behave identically until the flag is set explicitly.
    """
    try:
        from .config import config
        value = config.get_user_preference(
            "tool_transcript", TOOL_TRANSCRIPT_FLAT,
        )
        if value in _TOOL_TRANSCRIPT_MODES:
            return value
    except Exception:
        pass
    return TOOL_TRANSCRIPT_FLAT


def _extract_message_content(resp) -> str | None:
    """The assistant text of an Ollama chat response, dict/attr-safe.

    Returns the message content when it is a non-empty string, else None.
    Used to capture the direct answer a capable model puts in a zero-call
    decision, independent of the object/dict response shape.
    """
    if resp is None:
        return None
    msg = resp.get("message") if isinstance(resp, dict) else getattr(
        resp, "message", None,
    )
    if msg is None:
        return None
    content = msg.get("content") if isinstance(msg, dict) else getattr(
        msg, "content", None,
    )
    return content if isinstance(content, str) and content else None


def _native_call_echo(tool_name: str, arguments: dict) -> dict:
    """The assistant message echoing one executed call, native shape.

    One echo per executed call (the arguments are the post-repair ones --
    what actually ran). The content is empty: the call itself is the
    assistant's contribution to the turn.
    """
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": tool_name,
                "arguments": dict(arguments or {}),
            },
        }],
    }


def _native_tool_result(
    tool_name: str, result: str, earlier: bool = False,
) -> dict:
    """The role "tool" message carrying one result, native shape.

    The environment/untrusted framing is INSIDE the content: the security
    property (tool output is marked untrusted provenance, never elevated)
    is carried by the text itself, independent of the role.
    """
    when = "earlier tool result" if earlier else "tool result"
    return {
        "role": "tool",
        "content": (
            f"[environment] {when} (untrusted content): {tool_name}\n"
            f"Result: {result}"
        ),
    }



def _resolve_default_model() -> str:
    """Default tool model from the configuration, with a stable fallback.

    Reads the routing configuration when the seam is importable so the tool
    loop follows the user's model choices instead of a hardcoded name; any
    failure keeps the historical fallback.
    """
    try:
        from .config import get_model
        name = get_model("general", "primary")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return _FALLBACK_DEFAULT_MODEL


def _call_signature(tool_name: str, arguments: dict) -> str:
    """Stable identity of a tool call (name + canonical args) for the Lot 4
    anti-spin progress guard. Argument order and value types are normalized so
    two calls that differ only cosmetically still compare equal.
    """
    try:
        canonical = json.dumps(arguments or {}, sort_keys=True, default=str)
    except Exception:
        canonical = repr(sorted(
            (str(k), str(v)) for k, v in (arguments or {}).items()
        ))
    return f"{tool_name}\x00{canonical}"


# Lot 4 verification pass (deterministic-light). The artifact-producing handlers
# (execute_code / write_file) never raise -- they capture every outcome as text
# and the tool call is reported success=True regardless -- so a failed run or a
# failed write cannot be seen through ToolCallResult.success. These are the
# well-known FAILURE markers their result text carries. The set is intentionally
# conservative (clear failure signals only): a false retry costs more than a
# missed check, which merely leaves behavior as it was.
_EXEC_FAILURE_MARKERS = (
    "Traceback (most recent call last):",
    "Execution Failed (return code:",
    "Code execution error:",
)
_WRITE_FAILURE_MARKERS = (
    "Write file error:",
)


def _verification_hint(tool_calls: list) -> str | None:
    """Deterministic-light success check off executed results.

    Scans back to the most recent artifact-producing call (execute_code /
    write_file) and returns a corrective observation when its result text shows
    a clear failure, else None (criterion met, or nothing to verify). Only that
    last artifact action is judged -- it is the success criterion for the turn.
    """
    for tc in reversed(tool_calls):
        name = getattr(tc, "tool_name", "")
        result = getattr(tc, "result", "") or ""
        if name == "execute_code":
            if any(m in result for m in _EXEC_FAILURE_MARKERS):
                return (
                    "[environment] verification: the last code execution "
                    "reported an error. Inspect the output above, fix the "
                    "code, and run it again."
                )
            return None
        if name == "write_file":
            if any(m in result for m in _WRITE_FAILURE_MARKERS):
                return (
                    "[environment] verification: the last file write failed. "
                    "Check the path and content, then write the file again."
                )
            return None
    return None


# =============================================================================
# RESULT MODELS
# =============================================================================

class ToolCallResult(BaseModel):
    """Result of a single tool call."""
    tool_name: str
    arguments: dict[str, Any] = {}
    result: str = ""
    success: bool = True
    execution_time: float = 0.0
    reasoning: str = ""
    # Lot 3: a soft failure the model could fix by adjusting its call (missing
    # argument, handler error). Hard failures (approval denied, tool not found
    # or disabled) leave this False so the loop stops instead of retrying.
    retryable: bool = False


class ToolExecutionResult(BaseModel):
    """Full result of a tool execution."""
    response: str = ""
    tool_calls: list[ToolCallResult] = []
    model: str = ""
    total_time: float = 0.0
    # Corrective observations injected by the loop's verification pass
    # (one extra iteration after a failed artifact call). Additive with
    # a zero default: every construction site stays valid.
    verification_hints: int = 0


# Extended schema for the LLM decision (can indicate "none")
class ToolDecision(BaseModel):
    """LLM decision on tool usage."""
    tool_name: str  # "none" if no tool needed
    arguments: dict[str, Any] = {}
    reasoning: str = ""


# =============================================================================
# TOOL EXECUTOR
# =============================================================================

class ToolExecutor:
    """Tool executor with a ReAct loop.

    Orchestrates the planning, tool execution, and
    final response generation integrating tool results.
    """

    def __init__(
        self,
        registry=None,
        structured_engine=None,
        max_tool_calls: int = 5,
        default_model: str | None = None,
        max_tool_retries: int = 2,
        tool_transcript: str | None = None,
    ):
        self.registry = registry or _default_registry
        self.structured_engine = structured_engine or _structured_engine
        self.max_tool_calls = max_tool_calls
        # None resolves from the configuration (an explicit name always wins).
        self.default_model = default_model or _resolve_default_model()
        # Lot 3: consecutive retryable tool failures tolerated before the ReAct
        # loop gives up (the model gets the error fed back to self-correct).
        self.max_tool_retries = max_tool_retries
        # How executed tool calls are replayed to the model; an explicit
        # valid value wins, otherwise the configured preference (flat by
        # default -- the historical behavior).
        self.tool_transcript = (
            tool_transcript
            if tool_transcript in _TOOL_TRANSCRIPT_MODES
            else _resolve_tool_transcript()
        )
        # S128: Optional pre-execution approval hook.
        # Callable(tool_name, arguments) -> bool.
        # If set, called before each tool execution. Returns True to
        # proceed, False to deny. May block (e.g. waiting for human
        # approval in Bulbe mode).
        # S185 (EX-02): this attribute is the legacy process-wide fallback
        # only. The live request path passes a per-invocation approval_fn
        # through execute_with_tools -> _execute_tool instead, so that
        # concurrent generation threads (overlapping Bulbe sessions) cannot
        # clobber or drop each other's gate via this shared attribute.
        self.pre_tool_call_hook = None

        # Single-pass reuse: on the native path, a capable model that
        # calls no tool usually carries its direct answer in the decision
        # content. It is stashed here for the current turn so the caller can
        # reuse it as the final answer instead of paying a second
        # generation. Reset at the start of every tool loop and drained by
        # the caller, so it never leaks across turns.
        self._pending_direct_answer: str | None = None

    def should_use_tools(self, message: str, model: str = None) -> bool:
        """Determine if the request would benefit from tool usage.

        Fast heuristic based on keywords before doing
        LLM call for structured analysis.
        """
        if not self.registry:
            return False

        available = self.registry.list_available()
        if not available:
            return False

        # Accent-stripped, apostrophe-folded view so accented French
        # phrasings match the unaccented indicator entries below.
        msg_lower = normalize_for_match(message)

        # Tool-need indicators
        search_indicators = [
            "search", "cherche", "find", "look up", "what is the latest",
            "current", "today", "actualite", "news", "search",
        ]
        code_indicators = [
            "run", "execute", "calcul", "compute", "test this code",
            "lance", "essaie ce code", "execute ce code",
        ]
        file_indicators = [
            "read file", "write file", "list files", "lis le fichier",
            "ecris dans", "liste les fichiers", "show me the file",
        ]

        for ind in search_indicators:
            if ind in msg_lower and self.registry.is_available("web_search"):
                return True

        for ind in code_indicators:
            if ind in msg_lower and self.registry.is_available("execute_code"):
                return True

        for ind in file_indicators:
            if ind in msg_lower and (
                self.registry.is_available("read_file")
                or self.registry.is_available("write_file")
                or self.registry.is_available("list_files")
            ):
                return True

        return False

    @staticmethod
    def _notify_tool_call(
        on_tool_call: Callable[["ToolCallResult"], None] | None,
        call_result: "ToolCallResult",
    ) -> None:
        """Invoke the progress callback; its errors never break the loop."""
        if on_tool_call is None:
            return
        try:
            on_tool_call(call_result)
        except Exception as exc:
            logger.warning("Tool progress callback error: %s", exc)

    def execute_with_tools(
        self,
        message: str,
        model: str = None,
        conversation_messages: list[dict] | None = None,
        tool_history: list["ToolCallResult"] | None = None,
        approval_fn: Callable[[str, dict], bool] | None = None,
        on_tool_call: Callable[["ToolCallResult"], None] | None = None,
        manifest=None,
    ) -> ToolExecutionResult:
        """Execute a ReAct loop: plan -> tool -> observe -> respond.

        Args:
            message: User message
            model: Ollama model to use
            conversation_messages: Optional conversation history
            tool_history: Prior tool call results from earlier turns (S62)
            approval_fn: Optional per-invocation approval gate
                (tool_name, arguments) -> bool, called before each tool
                execution. S185 (EX-02): bound to this call rather than to a
                shared singleton attribute, so overlapping Bulbe sessions
                cannot clobber or drop each other's gate. Takes precedence
                over the legacy pre_tool_call_hook attribute.
            on_tool_call: Optional progress callback invoked with each
                ToolCallResult right after the tool executes, DURING the
                loop, so a caller can surface live activity instead of
                waiting for the final answer. Callback errors are logged
                and never interrupt the loop.

        Returns:
            ToolExecutionResult with final response and call history
        """
        start_time = time.time()
        _model = model or self.default_model
        # The final generation is shown what it could call this turn;
        # empty when no manifest is present, so behavior is unchanged then.
        _manifest_block = (
            getattr(manifest, "prompt_block", "") if manifest is not None else ""
        )

        (
            tool_calls, tool_results_context, native_transcript,
            context_messages, fatal, verification_hints,
        ) = self._run_tool_loop(
            message, _model, conversation_messages, tool_history,
            approval_fn, on_tool_call,
            **({"manifest": manifest} if manifest is not None else {}),
        )
        if fatal is not None:
            return ToolExecutionResult(
                response=fatal,
                tool_calls=tool_calls,
                model=_model,
                total_time=time.time() - start_time,
                verification_hints=verification_hints,
            )

        # Drain the zero-call reuse slot for this turn (always, so it
        # never carries over). It is only consumed when no tool ran.
        _candidate = self._take_direct_answer_candidate()
        final_response = None
        if not tool_calls and ROBUST_TOOLCALLING_AVAILABLE:
            final_response = self._salvage_from_narration(
                message, _model, context_messages, tool_calls,
                tool_results_context, approval_fn, on_tool_call,
                **({"candidate": _candidate} if _candidate is not None else {}),
            )
        if final_response is None:
            final_response = self._generate_final_response(
                message, _model, tool_calls, tool_results_context,
                context_messages, native_transcript,
                manifest_block=_manifest_block,
            )

        # A model sometimes echoes the internal scaffolding into its answer;
        # those lines are runtime plumbing, never user-facing content.
        final_response, stripped = strip_internal_markers(final_response)
        if stripped:
            logger.info(
                "Response hygiene: removed %d internal marker line(s) from "
                "the final answer", stripped,
            )

        return ToolExecutionResult(
            response=final_response,
            tool_calls=tool_calls,
            model=_model,
            total_time=time.time() - start_time,
            verification_hints=verification_hints,
        )

    def stream_with_tools(
        self,
        message: str,
        model: str = None,
        conversation_messages: list[dict] | None = None,
        tool_history: list["ToolCallResult"] | None = None,
        approval_fn: Callable[[str, dict], bool] | None = None,
        on_tool_call: Callable[["ToolCallResult"], None] | None = None,
        manifest=None,
    ):
        """Streaming variant of ``execute_with_tools``.

        Runs the same ReAct loop (tool activity flows through the
        ``on_tool_call`` progress callback), then streams the FINAL answer
        chunk by chunk instead of returning it in one block. Echoed internal
        markers are filtered incrementally, so the user never sees them even
        transiently. The generator's return value is the ToolExecutionResult;
        its ``response`` equals exactly the emitted text.
        """
        start_time = time.time()
        _model = model or self.default_model
        # The final generation is shown what it could call this turn;
        # empty when no manifest is present, so behavior is unchanged then.
        _manifest_block = (
            getattr(manifest, "prompt_block", "") if manifest is not None else ""
        )

        (
            tool_calls, tool_results_context, native_transcript,
            context_messages, fatal, verification_hints,
        ) = self._run_tool_loop(
            message, _model, conversation_messages, tool_history,
            approval_fn, on_tool_call,
            **({"manifest": manifest} if manifest is not None else {}),
        )
        if fatal is not None:
            yield fatal
            return ToolExecutionResult(
                response=fatal,
                tool_calls=tool_calls,
                model=_model,
                total_time=time.time() - start_time,
                verification_hints=verification_hints,
            )

        # Drain the zero-call reuse slot for this turn (always, so it
        # never carries over). It is only consumed when no tool ran.
        _candidate = self._take_direct_answer_candidate()
        pre_final = None
        if not tool_calls and ROBUST_TOOLCALLING_AVAILABLE:
            pre_final = self._salvage_from_narration(
                message, _model, context_messages, tool_calls,
                tool_results_context, approval_fn, on_tool_call,
                **({"candidate": _candidate} if _candidate is not None else {}),
            )

        emitted: list[str] = []
        if pre_final is not None:
            # Nothing was salvaged: the candidate IS the final answer.
            text, dropped = strip_internal_markers(pre_final)
            if dropped:
                logger.info(
                    "Response hygiene: removed %d internal marker line(s) "
                    "from the final answer", dropped,
                )
            if text:
                emitted.append(text)
                yield text
        else:
            final_messages = self._final_messages(
                message, tool_results_context, context_messages,
                native_transcript, manifest_block=_manifest_block,
            )
            marker_filter = (
                StreamMarkerFilter() if RESPONSE_HYGIENE_AVAILABLE else None
            )
            for piece in self._stream_final_response(final_messages, _model):
                out = marker_filter.feed(piece) if marker_filter else piece
                if out:
                    emitted.append(out)
                    yield out
            tail = marker_filter.flush() if marker_filter else ""
            if tail:
                emitted.append(tail)
                yield tail
            if marker_filter is not None and marker_filter.dropped:
                logger.info(
                    "Response hygiene: removed %d internal marker line(s) "
                    "from the streamed answer", marker_filter.dropped,
                )

        return ToolExecutionResult(
            response="".join(emitted),
            tool_calls=tool_calls,
            model=_model,
            total_time=time.time() - start_time,
            verification_hints=verification_hints,
        )

    def _run_tool_loop(
        self,
        message: str,
        _model: str,
        conversation_messages: list[dict] | None,
        tool_history: list["ToolCallResult"] | None,
        approval_fn: Callable[[str, dict], bool] | None,
        on_tool_call: Callable[["ToolCallResult"], None] | None,
        manifest=None,
    ) -> tuple[
        list["ToolCallResult"], list[str], list[dict], list[dict],
        str | None, int,
    ]:
        """The shared ReAct loop behind both execution fronts.

        Returns ``(tool_calls, tool_results_context, native_transcript,
        context_messages, fatal, verification_hints)`` where ``fatal`` is a
        terminal user-facing message when the prerequisites are missing
        (else None) and ``verification_hints`` counts the corrective
        observations the verification pass injected. The flat
        results context and the native transcript are BOTH accumulated on
        every run: the flat form keeps feeding the format= fallback, the
        salvage path and the verification hint, so a native-mode run
        degrades coherently when the native path is unavailable mid-loop;
        the message builders pick one representation at build time.
        """
        tool_calls: list[ToolCallResult] = []

        # A fresh turn owns a fresh reuse slot; anything a prior turn
        # stashed must never carry over.
        self._pending_direct_answer = None

        # Verify prerequisites
        if not self.registry:
            return tool_calls, [], [], [], "Tool registry not available.", 0

        # Build initial context. With a capability manifest, the prompt is
        # rendered from ITS tool set (the per-request truth); without one,
        # the live availability view is unchanged legacy behavior.
        if manifest is not None:
            tools_prompt = self.registry.get_tools_prompt(
                tools=list(getattr(manifest, "tools", ()) or ()),
            )
        else:
            tools_prompt = self.registry.get_tools_prompt()
        if not tools_prompt:
            return tool_calls, [], [], [], "No tools available.", 0

        # Conversation context
        context_messages = list(conversation_messages or [])

        # S62: Include prior tool history in context
        tool_results_context = []
        # Native transcript: the same history in the format function-calling
        # models are trained on. Prior-turn calls are replayed as
        # assistant-echo plus tool-result pairs before the current user
        # message (their true chronology); the environment/untrusted framing
        # lives inside the tool message content.
        native_transcript: list[dict] = []
        if tool_history:
            for prior_tc in tool_history:
                tool_results_context.append(
                    f"[environment] earlier tool call by assistant: "
                    f"{prior_tc.tool_name}\n"
                    f"Arguments: {prior_tc.arguments}\n"
                    f"Result: {prior_tc.result}"
                )
                native_transcript.append(_native_call_echo(
                    prior_tc.tool_name, prior_tc.arguments,
                ))
                native_transcript.append(_native_tool_result(
                    prior_tc.tool_name, prior_tc.result, earlier=True,
                ))
        native_transcript.append({"role": "user", "content": message})

        # ReAct loop: decide -> execute -> re-decide. _decide_tools layers
        # native function-calling (Layer 1) over the existing format= path and
        # can return multiple calls per turn (native parallel tool calls). Lot 3:
        # a retryable failure (bad argument, handler error) is fed back into the
        # next decision instead of stopping the loop, so the model can correct
        # itself; bounded by max_tool_retries consecutive retryable failures.
        consecutive_failures = 0
        # No-progress guard: a bounded window of the most recent SUCCESSFUL
        # call signatures. A new call identical to ANY signature in the
        # window -- immediate or alternating (A-B-A) -- makes no further
        # progress, so the loop stops instead of repeating it. Only
        # successful calls enter the window, so the retry path for a FAILING
        # call keeps its own budget (max_tool_retries, error fed back).
        recent_signatures: deque[str] = deque(maxlen=3)
        # Lot 4 verification pass: when the model stops issuing calls, confirm
        # the success criterion off the executed results before declaring done.
        # A single corrective iteration is injected at most (verification_done).
        verification_done = False
        # Injections actually performed, threaded into the result so the
        # caller's instrumentation reports what really played.
        verification_hints = 0
        for iteration in range(self.max_tool_calls):
            # Forwarded only when a manifest exists: the no-manifest call
            # keeps its exact historical shape, so anything wrapping the
            # decision hook stays compatible unchanged.
            decisions = self._decide_tools(
                message, _model, context_messages, tool_results_context,
                native_transcript=native_transcript,
                **({"manifest": manifest} if manifest is not None else {}),
            )
            if not decisions:
                if not verification_done and tool_calls:
                    hint = _verification_hint(tool_calls)
                    if hint is not None:
                        verification_done = True
                        verification_hints += 1
                        tool_results_context.append(hint)
                        native_transcript.append(
                            {"role": "user", "content": hint}
                        )
                        logger.info(
                            "Verification: success criterion not met; running "
                            "one corrective iteration"
                        )
                        continue
                break

            hard_stop = False
            retryable_failure = False
            spin_detected = False
            for tool_name, arguments in decisions:
                signature = _call_signature(tool_name, arguments)
                if signature in recent_signatures:
                    logger.info(
                        "No-progress guard: call to %s repeats a recent "
                        "successful call; stopping the ReAct loop",
                        tool_name,
                    )
                    spin_detected = True
                    break
                call_result = self._execute_tool(
                    tool_name, arguments, "", approval_fn=approval_fn,
                )
                tool_calls.append(call_result)
                self._notify_tool_call(on_tool_call, call_result)
                tool_results_context.append(
                    f"[environment] tool call by assistant: "
                    f"{call_result.tool_name}\n"
                    f"Arguments: {call_result.arguments}\n"
                    f"Result: {call_result.result}"
                )
                native_transcript.append(_native_call_echo(
                    call_result.tool_name, call_result.arguments,
                ))
                native_transcript.append(_native_tool_result(
                    call_result.tool_name, call_result.result,
                ))
                if not call_result.success:
                    if call_result.retryable:
                        retryable_failure = True
                    else:
                        hard_stop = True
                    break
                recent_signatures.append(signature)
            if spin_detected:
                break
            if hard_stop:
                break
            if retryable_failure:
                consecutive_failures += 1
                if consecutive_failures >= self.max_tool_retries:
                    break
                continue  # re-decide; the error is now in the context
            consecutive_failures = 0

        return tool_calls, tool_results_context, native_transcript, \
            context_messages, None, verification_hints

    def _resolve_direct_answer_reuse(self) -> bool:
        """Whether the single-pass reuse is enabled.

        Reads the user preference when the config seam is importable; any
        failure or absence keeps it ON, so the optimization applies by
        default and only an explicit opt-out disables it.
        """
        try:
            from .config import config
            return bool(config.get_user_preference(
                "tool_direct_answer_reuse", True,
            ))
        except Exception:
            return True

    def _take_direct_answer_candidate(self) -> str | None:
        """Drain the stashed zero-call answer; return it only if reusable.

        The slot is ALWAYS cleared (so it never leaks past the turn),
        independent of eligibility. None is returned -- and the historical
        generation path runs -- when there is nothing stashed, when the
        reuse is opted out, or when the content is empty once the internal
        scaffold is stripped. The RAW text is returned otherwise: the single
        hygiene pass happens downstream, exactly where the generated answer
        would also be cleaned, so reuse and generation share one code path.
        """
        raw = self._pending_direct_answer
        self._pending_direct_answer = None
        if not raw:
            return None
        if not self._resolve_direct_answer_reuse():
            return None
        cleaned, _ = strip_internal_markers(raw)
        if not cleaned.strip():
            return None
        return raw

    def _salvage_from_narration(
        self,
        message: str,
        _model: str,
        context_messages: list[dict],
        tool_calls: list["ToolCallResult"],
        tool_results_context: list[str],
        approval_fn: Callable[[str, dict], bool] | None,
        on_tool_call: Callable[["ToolCallResult"], None] | None,
        candidate: str | None = None,
    ) -> str | None:
        """Layer 2b -- deterministic salvage when no tool fired.

        The model may have narrated code/commands in prose instead of calling
        a tool. A candidate answer is generated; when tool calls can be
        transpiled out of it they are executed (mutating ``tool_calls`` and
        ``tool_results_context`` in place) and None is returned so the caller
        produces a fresh final answer over the results. When nothing can be
        salvaged the candidate itself is returned as the final answer.

        when ``candidate`` is supplied (the zero-call decision already
        carried a direct answer), it is reused verbatim instead of paying a
        second generation. The transpile safety net still runs on it, so a
        narrated tool action is executed exactly as before; only the
        redundant generation is skipped.
        """
        if candidate is not None:
            logger.info(
                "Single-pass reuse: the zero-call decision answer is reused "
                "as the final answer (no second generation)"
            )
        else:
            candidate = self._generate_final_response(
                message, _model, [], [], context_messages,
            )
        available_names = (
            {t.name for t in self.registry.list_available()}
            if self.registry else set()
        )
        salvaged = transpile_intent(candidate, message, available_names)
        if not salvaged:
            return candidate
        logger.info(
            "Salvaged %d tool call(s) from a narrated response",
            len(salvaged),
        )
        for tool_name, arguments in salvaged:
            call_result = self._execute_tool(
                tool_name, arguments, "intent-transpiled",
                approval_fn=approval_fn,
            )
            tool_calls.append(call_result)
            self._notify_tool_call(on_tool_call, call_result)
            tool_results_context.append(
                f"[environment] tool call by assistant: "
                f"{call_result.tool_name}\n"
                f"Arguments: {call_result.arguments}\n"
                f"Result: {call_result.result}"
            )
            if not call_result.success:
                break
        return None

    def _ask_llm_for_tool(
        self,
        message: str,
        model: str,
        tools_prompt: str,
        previous_results: list[str],
        context_messages: list[dict],
    ) -> ToolDecision | None:
        """Ask the LLM which tool to use via the StructuredOutputEngine.

        Returns:
            ToolDecision ou None en cas d'erreur
        """
        if not STRUCTURED_OUTPUT_AVAILABLE or self.structured_engine is None:
            return None

        # Build the messages
        messages = list(context_messages)

        # Add previous results if this is a subsequent iteration
        if previous_results:
            results_text = "\n\n".join(previous_results)
            user_content = (
                f"{message}\n\n"
                f"{ENV_RESULTS_HEADER}\n{results_text}\n\n"
                f"Based on these results, do you need another tool? "
                f'If not, set tool_name to "none". Never attribute these '
                f"tool actions to the user."
            )
        else:
            user_content = message

        messages.append({"role": "user", "content": user_content})

        # Appel structure
        result = self.structured_engine.generate_structured(
            messages=messages,
            schema=ToolDecision,
            model=model,
            extra_system_prompt=tools_prompt,
            temperature=0.0,
            max_retries=2,
        )

        if result.success and result.data:
            return result.data
        return None

    def _decide_tools(
        self,
        message: str,
        model: str,
        context_messages: list[dict],
        previous_results: list[str],
        force: bool = False,
        native_transcript: list[dict] | None = None,
        manifest=None,
    ) -> list[tuple[str, dict]]:
        """Decide which tool(s) to call next; an empty list means stop.

        Layer 1: native Ollama function-calling for capable models -- it matches
        what the model was trained on, so selection is far more reliable, and
        parallel calls come for free. Layer 2a: when a capable model declines
        but ``force`` is set, an enum-constrained format= schema (no "none")
        guarantees a selection. Non-capable models, or any failure, fall back to
        the existing format= ToolDecision path with unchanged behavior.
        """
        # With a capability manifest the decision sees ITS tool set -- the
        # per-request truth (mode, killswitch, overrides already applied);
        # otherwise the live availability view, unchanged legacy behavior.
        if manifest is not None:
            available = list(getattr(manifest, "tools", ()) or ())
        else:
            available = self.registry.list_available() if self.registry else []
        if not available:
            return []
        manifest_block = (
            getattr(manifest, "prompt_block", "") if manifest is not None else ""
        )

        use_native = (
            ROBUST_TOOLCALLING_AVAILABLE
            and OLLAMA_AVAILABLE
            and model_supports_native_tools(model)
        )
        if use_native:
            messages = self._build_decision_messages(
                message, context_messages, previous_results,
                native_transcript,
            )
            if manifest_block:
                messages = (
                    [{"role": "system", "content": manifest_block}] + messages
                )
            resp = None
            try:
                resp = ollama.chat(
                    model=model,
                    messages=messages,
                    tools=native_tool_schemas(available),
                    options={"temperature": 0.0},
                )
            except Exception as e:
                logger.warning(
                    "Native tool call failed (%s); falling back to format=", e,
                )
            if resp is not None:
                calls = parse_native_tool_calls(resp)
                if calls:
                    return calls
                if force:
                    forced = self._enum_force_tool(
                        messages, model, [t.name for t in available],
                    )
                    return [forced] if forced else []
                # A capable model that called nothing is done. It usually
                # carries its direct answer here: stash it so the
                # caller can reuse it as the final answer rather than paying
                # a second generation. Only in the non-forced branch: a
                # forced decision needs a tool, so there is no answer to
                # reuse.
                self._pending_direct_answer = _extract_message_content(resp)
                return []

        # Fallback: the existing format= ToolDecision path. With a manifest,
        # the decision prompt carries the same truth as the native path: the
        # capability block plus the manifest-filtered tool descriptions.
        if manifest is not None:
            base_prompt = (
                self.registry.get_tools_prompt(tools=available)
                if self.registry else ""
            )
            tools_prompt = (
                f"{manifest_block}\n\n{base_prompt}" if manifest_block
                else base_prompt
            )
        else:
            tools_prompt = self.registry.get_tools_prompt()
        decision = self._ask_llm_for_tool(
            message, model, tools_prompt,
            previous_results, context_messages,
        )
        if decision is None or decision.tool_name in ("none", ""):
            return []
        return [(decision.tool_name, decision.arguments)]

    def _build_decision_messages(
        self,
        message: str,
        context_messages: list[dict],
        previous_results: list[str],
        native_transcript: list[dict] | None = None,
    ) -> list[dict]:
        """Assemble the chat messages for a tool decision (with prior results).

        In native mode the transcript (user message, assistant tool_calls
        echoes, role "tool" results) is replayed as-is -- the format the
        model was trained on -- instead of being folded into one rebuilt
        user message.
        """
        if (
            self.tool_transcript == TOOL_TRANSCRIPT_NATIVE
            and native_transcript
        ):
            messages = list(context_messages)
            messages.extend(native_transcript)
            return messages
        messages = list(context_messages)
        if previous_results:
            results_text = "\n\n".join(previous_results)
            content = (
                f"{message}\n\n"
                f"{ENV_RESULTS_HEADER}\n{results_text}\n\n"
                f"Call the next tool if needed; otherwise answer directly. "
                f"Never attribute these tool actions to the user."
            )
        else:
            content = message
        messages.append({"role": "user", "content": content})
        return messages

    def _enum_force_tool(
        self, messages: list[dict], model: str, tool_names: list[str],
    ) -> tuple[str, dict] | None:
        """Force a tool selection via an enum-constrained format= schema.

        The schema's tool_name enum excludes "none", so the sampler cannot
        decline -- a tool is guaranteed when the caller knows one is needed.
        """
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                format=forced_decision_schema(tool_names),
                options={"temperature": 0.0},
            )
            raw = (
                resp["message"]["content"]
                if isinstance(resp, dict)
                else resp.message.content
            )
            data = json.loads(raw)
            name = data.get("tool_name")
            if name in tool_names:
                args = data.get("arguments")
                return (name, args if isinstance(args, dict) else {})
        except Exception as e:
            logger.warning("Enum-force tool selection failed: %s", e)
        return None

    def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any],
        reasoning: str = "",
        approval_fn: Callable[[str, dict], bool] | None = None,
    ) -> ToolCallResult:
        """Execute a single tool call."""
        start_time = time.time()

        # S128: Pre-execution approval hook (Bulbe mode tool call approval).
        # S185 (EX-02): an explicit per-invocation approval_fn takes precedence
        # over the legacy process-wide pre_tool_call_hook attribute. Binding the
        # gate to the call (not a shared attribute) means two concurrent Bulbe
        # generation threads cannot clobber each other's hook or drop the gate
        # when one of them finishes.
        hook = approval_fn if approval_fn is not None else self.pre_tool_call_hook
        if hook is not None:
            try:
                approved = hook(tool_name, arguments)
                if not approved:
                    return ToolCallResult(
                        tool_name=tool_name,
                        arguments=arguments,
                        result="Tool call denied by approval gate",
                        success=False,
                        execution_time=time.time() - start_time,
                        reasoning=reasoning,
                    )
            except Exception as hook_exc:
                # Fail-secure: deny on hook error
                return ToolCallResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=f"Tool call denied (approval error: {hook_exc})",
                    success=False,
                    execution_time=time.time() - start_time,
                    reasoning=reasoning,
                )

        # Verify the tool exists and is available
        tool = self.registry.get(tool_name) if self.registry else None
        if tool is None:
            return ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                result=f"Tool not found: {tool_name}",
                success=False,
                execution_time=time.time() - start_time,
                reasoning=reasoning,
            )

        if not tool.enabled:
            return ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                result=f"Tool disabled: {tool_name}",
                success=False,
                execution_time=time.time() - start_time,
                reasoning=reasoning,
            )

        if tool.handler is None:
            return ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                result=f"No handler for tool: {tool_name}",
                success=False,
                execution_time=time.time() - start_time,
                reasoning=reasoning,
            )

        # Lot 3: auto-repair near-miss argument names (path -> filename, code ->
        # content, etc.) before resolving, so a slightly mis-keyed call succeeds
        # instead of failing on a missing required parameter.
        if ROBUST_TOOLCALLING_AVAILABLE:
            arguments = repair_arguments(list(tool.parameters), arguments)

        # Apply default values for optional parameters
        resolved_args = {}
        for param_name, param_def in tool.parameters.items():
            if param_name in arguments:
                resolved_args[param_name] = arguments[param_name]
            elif not param_def.required and param_def.default is not None:
                resolved_args[param_name] = param_def.default

        # Verify the required parameters
        for param_name, param_def in tool.parameters.items():
            if param_def.required and param_name not in resolved_args:
                return ToolCallResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=(
                        f"Missing required parameter: {param_name}. "
                        f"Provide it and call the tool again."
                    ),
                    success=False,
                    retryable=True,
                    execution_time=time.time() - start_time,
                    reasoning=reasoning,
                )

        # Execute the handler
        try:
            result_str = tool.handler(**resolved_args)
            execution_time = time.time() - start_time

            return ToolCallResult(
                tool_name=tool_name,
                arguments=resolved_args,
                result=str(result_str),
                success=True,
                execution_time=execution_time,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"Erreur execution outil {tool_name}: {e}")
            return ToolCallResult(
                tool_name=tool_name,
                arguments=resolved_args,
                result=(
                    f"Execution error: {e}. "
                    f"Adjust the arguments or approach and retry."
                ),
                success=False,
                retryable=True,
                execution_time=time.time() - start_time,
                reasoning=reasoning,
            )

    def _final_messages(
        self,
        message: str,
        tool_results_context: list[str],
        context_messages: list[dict],
        native_transcript: list[dict] | None = None,
        manifest_block: str = "",
    ) -> list[dict]:
        """Chat messages for the final user-facing answer.

        The attribution system message leads: the model must report the tool
        actions as its own, never as something the user did. Shared by the
        single-shot and streaming fronts so both carry identical framing.

        When a capability block is present it is pinned right after the
        attribution message, so the FINAL generation knows what it could
        call this turn; an empty block is the exact prior behavior.

        In native mode the conversation ends on the last role "tool" message
        with no synthetic user turn -- the trained final-answer position --
        and the leading system message carries the reporting instruction.
        """
        lead: list[dict] = [
            {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT},
        ]
        if manifest_block:
            lead.append({"role": "system", "content": manifest_block})
        if (
            self.tool_transcript == TOOL_TRANSCRIPT_NATIVE
            and native_transcript
        ):
            messages = list(lead)
            messages.extend(context_messages)
            messages.extend(native_transcript)
            return messages
        messages = list(lead)
        messages.extend(context_messages)

        if tool_results_context:
            results_text = "\n\n".join(tool_results_context)
            messages.append({
                "role": "user",
                "content": (
                    f"{message}\n\n"
                    f"{ENV_RESULTS_HEADER}\n{results_text}\n\n"
                    f"Write the final user-facing answer, reporting in "
                    f"first person what you did and what the results were."
                ),
            })
        else:
            messages.append({"role": "user", "content": message})
        return messages

    def _stream_final_response(self, messages: list[dict], model: str):
        """Stream the final answer's raw chunks for the given messages.

        Falls back to a single-shot generation (yielded once) when the
        backend cannot stream, so the streaming front never loses the answer.
        """
        if not OLLAMA_AVAILABLE:
            yield "Cannot generate response: Ollama not available."
            return
        try:
            stream = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.3},
                stream=True,
            )
            produced = False
            for chunk in stream:
                part = chunk.get("message") if isinstance(chunk, dict) \
                    else getattr(chunk, "message", None)
                content = part.get("content") if isinstance(part, dict) \
                    else getattr(part, "content", "")
                if content:
                    produced = True
                    yield content
            if produced:
                return
        except Exception as exc:
            logger.warning(
                "Streaming final response failed (%s); falling back to a "
                "single-shot generation", exc,
            )
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.3},
            )
            yield response.message.content
        except Exception as exc:
            logger.error(f"Final response generation error: {exc}")
            yield f"Error generating response: {exc}"

    def _generate_final_response(
        self,
        message: str,
        model: str,
        tool_calls: list[ToolCallResult],
        tool_results_context: list[str],
        context_messages: list[dict],
        native_transcript: list[dict] | None = None,
        manifest_block: str = "",
    ) -> str:
        """Generate the final LLM response integrating tool results.

        If no tools were called, generate a direct response.
        """
        if not OLLAMA_AVAILABLE:
            # Fallback: assemble raw results
            if tool_results_context:
                return "\n\n".join(tool_results_context)
            return "Cannot generate response: Ollama not available."

        messages = self._final_messages(
            message, tool_results_context, context_messages,
            native_transcript, manifest_block=manifest_block,
        )

        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.3},
            )
            return response.message.content
        except Exception as e:
            logger.error(f"Final response generation error: {e}")
            # Fallback with raw results
            if tool_results_context:
                return (
                    "I encountered an error generating the final response, "
                    "but here are the tool results:\n\n"
                    + "\n\n".join(tool_results_context)
                )
            return f"Error generating response: {e}"


# =============================================================================
# SINGLETON
# =============================================================================

tool_executor = ToolExecutor()
