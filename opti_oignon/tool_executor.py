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

logger = logging.getLogger(__name__)


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
        default_model: str = "qwen3:32b",
        max_tool_retries: int = 2,
    ):
        self.registry = registry or _default_registry
        self.structured_engine = structured_engine or _structured_engine
        self.max_tool_calls = max_tool_calls
        self.default_model = default_model
        # Lot 3: consecutive retryable tool failures tolerated before the ReAct
        # loop gives up (the model gets the error fed back to self-correct).
        self.max_tool_retries = max_tool_retries
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

        msg_lower = message.lower()

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

    def execute_with_tools(
        self,
        message: str,
        model: str = None,
        conversation_messages: list[dict] | None = None,
        tool_history: list["ToolCallResult"] | None = None,
        approval_fn: Callable[[str, dict], bool] | None = None,
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

        Returns:
            ToolExecutionResult with final response and call history
        """
        start_time = time.time()
        _model = model or self.default_model
        tool_calls: list[ToolCallResult] = []

        # Verify prerequisites
        if not self.registry:
            return ToolExecutionResult(
                response="Tool registry not available.",
                model=_model,
                total_time=time.time() - start_time,
            )

        # Build initial context
        tools_prompt = self.registry.get_tools_prompt()
        if not tools_prompt:
            return ToolExecutionResult(
                response="No tools available.",
                model=_model,
                total_time=time.time() - start_time,
            )

        # Conversation context
        context_messages = list(conversation_messages or [])

        # S62: Include prior tool history in context
        tool_results_context = []
        if tool_history:
            for prior_tc in tool_history:
                tool_results_context.append(
                    f"[Prior tool call: {prior_tc.tool_name}] "
                    f"Arguments: {prior_tc.arguments}\n"
                    f"Result: {prior_tc.result}"
                )

        # ReAct loop: decide -> execute -> re-decide. _decide_tools layers
        # native function-calling (Layer 1) over the existing format= path and
        # can return multiple calls per turn (native parallel tool calls). Lot 3:
        # a retryable failure (bad argument, handler error) is fed back into the
        # next decision instead of stopping the loop, so the model can correct
        # itself; bounded by max_tool_retries consecutive retryable failures.
        consecutive_failures = 0
        for iteration in range(self.max_tool_calls):
            decisions = self._decide_tools(
                message, _model, context_messages, tool_results_context,
            )
            if not decisions:
                break

            hard_stop = False
            retryable_failure = False
            for tool_name, arguments in decisions:
                call_result = self._execute_tool(
                    tool_name, arguments, "", approval_fn=approval_fn,
                )
                tool_calls.append(call_result)
                tool_results_context.append(
                    f"[Tool: {call_result.tool_name}] "
                    f"Arguments: {call_result.arguments}\n"
                    f"Result: {call_result.result}"
                )
                if not call_result.success:
                    if call_result.retryable:
                        retryable_failure = True
                    else:
                        hard_stop = True
                    break
            if hard_stop:
                break
            if retryable_failure:
                consecutive_failures += 1
                if consecutive_failures >= self.max_tool_retries:
                    break
                continue  # re-decide; the error is now in the context
            consecutive_failures = 0

        # Layer 2b -- deterministic salvage: if no tool fired, the model may
        # have narrated code/commands in prose instead of calling a tool.
        # Recover those calls and execute them, then regenerate with results.
        if not tool_calls and ROBUST_TOOLCALLING_AVAILABLE:
            candidate = self._generate_final_response(
                message, _model, [], [], context_messages,
            )
            available_names = (
                {t.name for t in self.registry.list_available()}
                if self.registry else set()
            )
            salvaged = transpile_intent(candidate, message, available_names)
            if salvaged:
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
                    tool_results_context.append(
                        f"[Tool: {call_result.tool_name}] "
                        f"Arguments: {call_result.arguments}\n"
                        f"Result: {call_result.result}"
                    )
                    if not call_result.success:
                        break
                final_response = self._generate_final_response(
                    message, _model, tool_calls, tool_results_context,
                    context_messages,
                )
            else:
                final_response = candidate
        else:
            final_response = self._generate_final_response(
                message, _model, tool_calls, tool_results_context,
                context_messages,
            )

        return ToolExecutionResult(
            response=final_response,
            tool_calls=tool_calls,
            model=_model,
            total_time=time.time() - start_time,
        )

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
                f"Previous tool results:\n{results_text}\n\n"
                f"Based on these results, do you need another tool? "
                f'If not, set tool_name to "none".'
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
    ) -> list[tuple[str, dict]]:
        """Decide which tool(s) to call next; an empty list means stop.

        Layer 1: native Ollama function-calling for capable models -- it matches
        what the model was trained on, so selection is far more reliable, and
        parallel calls come for free. Layer 2a: when a capable model declines
        but ``force`` is set, an enum-constrained format= schema (no "none")
        guarantees a selection. Non-capable models, or any failure, fall back to
        the existing format= ToolDecision path with unchanged behavior.
        """
        available = self.registry.list_available() if self.registry else []
        if not available:
            return []

        use_native = (
            ROBUST_TOOLCALLING_AVAILABLE
            and OLLAMA_AVAILABLE
            and model_supports_native_tools(model)
        )
        if use_native:
            messages = self._build_decision_messages(
                message, context_messages, previous_results,
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
                return []  # a capable model that called nothing is done

        # Fallback: the existing format= ToolDecision path (unchanged behavior).
        decision = self._ask_llm_for_tool(
            message, model, self.registry.get_tools_prompt(),
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
    ) -> list[dict]:
        """Assemble the chat messages for a tool decision (with prior results)."""
        messages = list(context_messages)
        if previous_results:
            results_text = "\n\n".join(previous_results)
            content = (
                f"{message}\n\n"
                f"Previous tool results:\n{results_text}\n\n"
                f"Call the next tool if needed; otherwise answer directly."
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

    def _generate_final_response(
        self,
        message: str,
        model: str,
        tool_calls: list[ToolCallResult],
        tool_results_context: list[str],
        context_messages: list[dict],
    ) -> str:
        """Generate the final LLM response integrating tool results.

        If no tools were called, generate a direct response.
        """
        if not OLLAMA_AVAILABLE:
            # Fallback: assemble raw results
            if tool_results_context:
                return "\n\n".join(tool_results_context)
            return "Cannot generate response: Ollama not available."

        # Build the messages for the final response
        messages = list(context_messages)

        if tool_results_context:
            results_text = "\n\n".join(tool_results_context)
            messages.append({
                "role": "user",
                "content": (
                    f"{message}\n\n"
                    f"Here are the tool results:\n{results_text}\n\n"
                    f"Please provide a comprehensive answer based on "
                    f"the question and the tool results above."
                ),
            })
        else:
            messages.append({"role": "user", "content": message})

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
