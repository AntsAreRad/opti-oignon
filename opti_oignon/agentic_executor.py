#!/usr/bin/env python3
"""
AGENTIC EXECUTOR - OPTI-OIGNON v1.5.0 (S45)
=============================================

Unified agentic executor -- the single entry point that orchestrates
every available capability: tool calling, code verification,
thinking mode, and web search.

Analyzes the user request via the StructuredOutputEngine to
automatically determine the optimal pipeline, then delegates
execution to the specialized components.

Architecture:
    User message
        |
        v
    Task analysis (TaskAnalysis via StructuredOutputEngine)
        |
        +--> Simple question       --> Direct Executor
        |
        +--> Tools needed          --> ToolExecutor (ReAct loop)
        |
        +--> Code question         --> Executor + Verification
        |
        +--> Complex question      --> Think mode + optional tools
        |
        +--> Web search            --> Web search + LLM synthesis

S261: comments, docstrings, and log literals normalised to English
(the project rule: no French anywhere in code); the French detection
keywords below are functional data and stay. S261 also threads the
per-invocation approval_fn (S185, EX-02) through the tools and
think+tools dispatch, which previously dropped it.

Author: Leon
"""

import logging
import time
from collections.abc import Callable, Generator
from typing import Any

# Conditional import of the existing Executor
try:
    from .executor import Executor
    from .executor import executor as _default_executor
    from .router import RoutingResult
    EXECUTOR_AVAILABLE = True
except ImportError:
    EXECUTOR_AVAILABLE = False
    _default_executor = None
    Executor = None
    RoutingResult = None

# Conditional import of the ToolExecutor (S44)
try:
    from .tool_executor import (
        ToolCallResult,
        ToolExecutionResult,
        ToolExecutor,
    )
    from .tool_executor import (
        tool_executor as _default_tool_executor,
    )
    TOOL_EXECUTOR_AVAILABLE = True
except ImportError:
    TOOL_EXECUTOR_AVAILABLE = False
    _default_tool_executor = None
    ToolExecutor = None
    ToolExecutionResult = None
    ToolCallResult = None

# Conditional import of the StructuredOutputEngine (S42)
try:
    from .structured_output import (
        StructuredOutputEngine,
        TaskAnalysis,
    )
    from .structured_output import (
        structured_output_engine as _default_structured_engine,
    )
    STRUCTURED_OUTPUT_AVAILABLE = True
except ImportError:
    STRUCTURED_OUTPUT_AVAILABLE = False
    _default_structured_engine = None
    StructuredOutputEngine = None
    TaskAnalysis = None

# Conditional import of the VerificationEngine (S43)
try:
    from .verification import (
        VerificationResult,
    )
    from .verification import (
        verification_engine as _default_verification_engine,
    )
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False
    _default_verification_engine = None
    VerificationResult = None

# Conditional import of the ReasoningEngine (S49)
try:
    from .reasoning import (
        ReasoningConfig,
        ReasoningEngine,
        ReasoningResult,
        ReasoningStep,
    )
    from .reasoning import (
        reasoning_engine as _default_reasoning_engine,
    )
    REASONING_AVAILABLE = True
except ImportError:
    REASONING_AVAILABLE = False
    _default_reasoning_engine = None
    ReasoningEngine = None
    ReasoningResult = None
    ReasoningStep = None
    ReasoningConfig = None

# Conditional import of the ConsensusEngine (S50)
try:
    from .consensus import (
        ConsensusConfig,
        ConsensusEngine,
        ConsensusResult,
    )
    from .consensus import (
        ModelResponse as ConsensusModelResponse,
    )
    from .consensus import (
        consensus_engine as _default_consensus_engine,
    )
    CONSENSUS_AVAILABLE = True
except ImportError:
    CONSENSUS_AVAILABLE = False
    _default_consensus_engine = None
    ConsensusEngine = None
    ConsensusResult = None
    ConsensusConfig = None
    ConsensusModelResponse = None

# Conditional import of the SelfCorrectionEngine (S51)
try:
    from .self_correction import (
        SelfCorrectionConfig,
        SelfCorrectionEngine,
        SelfCorrectionResult,
    )
    from .self_correction import (
        self_correction_engine as _default_self_correction_engine,
    )
    SELF_CORRECTION_AVAILABLE = True
except ImportError:
    SELF_CORRECTION_AVAILABLE = False
    _default_self_correction_engine = None
    SelfCorrectionEngine = None
    SelfCorrectionResult = None
    SelfCorrectionConfig = None

# Conditional import of the CascadingInference (S69)
try:
    from .cascading import (
        CascadeResult,
        CascadingInference,
    )
    from .cascading import (
        cascading_inference as _default_cascading_inference,
    )
    CASCADING_INFERENCE_AVAILABLE = True
except ImportError:
    CASCADING_INFERENCE_AVAILABLE = False
    _default_cascading_inference = None
    CascadingInference = None
    CascadeResult = None

# Conditional import of the SpeculativeGenerator (S70)
try:
    from .speculative import (
        SpeculativeGenerator,
        SpeculativeResult,
    )
    from .speculative import (
        speculative_generator as _default_speculative_generator,
    )
    SPECULATIVE_GENERATION_AVAILABLE = True
except ImportError:
    SPECULATIVE_GENERATION_AVAILABLE = False
    _default_speculative_generator = None
    SpeculativeGenerator = None
    SpeculativeResult = None

# Conditional import of the ToolRegistry (S73 security)
try:
    from .tool_registry import tool_registry as _default_tool_registry
    TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    TOOL_REGISTRY_AVAILABLE = False
    _default_tool_registry = None

logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE CONSTANTS
# =============================================================================

PIPELINE_DIRECT = "direct"
PIPELINE_TOOLS = "tools"
PIPELINE_CODE_VERIFY = "code_verify"
PIPELINE_THINK = "think"
PIPELINE_WEB_SEARCH = "web_search"
PIPELINE_THINK_TOOLS = "think_tools"
PIPELINE_REASONING = "reasoning"  # S49
PIPELINE_CONSENSUS = "consensus"  # S50
PIPELINE_SELF_CORRECT = "self_correct"  # S51
PIPELINE_CASCADING = "cascading"  # S69
PIPELINE_SPECULATIVE = "speculative"  # S70


# =============================================================================
# COMPLEXITY ANALYSIS -- FAST HEURISTICS
# =============================================================================

# The keyword lists below are DETECTION DATA: the French entries match
# French user phrasings and are deliberately kept (S261).

# Tool-need detection keywords
_TOOL_KEYWORDS = [
    "search", "cherche", "find", "look up", "what is the latest",
    "current", "today", "actualite", "news", "search",
    "run", "execute", "calcul", "compute", "test this code",
    "lance", "essaie ce code", "execute ce code",
    "read file", "write file", "list files", "lis le fichier",
    "ecris dans", "liste les fichiers", "show me the file",
]

# Web-search keywords
_WEB_SEARCH_KEYWORDS = [
    "search", "cherche sur", "google", "find online", "latest news",
    "actualite", "what happened", "recent", "aujourd'hui", "today",
    "current price", "weather", "meteo",
]

# Code keywords
_CODE_KEYWORDS = [
    "code", "python", "script", "function", "classe", "class",
    "def ", "import ", "library", "package", "bug", "error",
    "debug", "fix", "compile", "syntax", "r script", "ggplot",
    "tidyverse", "dataframe", "pandas", "numpy", "matplotlib",
]

# Complexity keywords (favour the think mode)
_COMPLEXITY_KEYWORDS = [
    "explain", "compare", "analyze", "architecture", "design",
    "optimize", "refactor", "why", "how does", "pros and cons",
    "tradeoff", "strategy", "plan", "step by step",
    "explique", "compare", "analyse", "pourquoi", "comment",
    "avantages", "inconvenients", "strategie", "etape par etape",
]

# S49: Advanced-reasoning keywords (CoT, decomposition)
_REASONING_KEYWORDS = [
    "step by step", "etape par etape",
    "break down", "decompose", "decomposer",
    "analyze in detail", "analyse en detail",
    "think through", "reflechis a",
    "systematic", "systematique",
    "comprehensive analysis", "analyse complete",
    "evaluate different approaches", "evalue differentes approches",
    "compare and contrast", "compare et contraste",
    "multi-step", "multi-etape",
    "consider all options", "considere toutes les options",
]


def _quick_classify(message: str) -> dict:
    """Fast heuristic classification of the request.

    Return a dict with pipeline indicators.
    Never calls the LLM -- used as the fallback when the
    StructuredOutputEngine is not available.
    """
    msg_lower = message.lower()

    needs_tools = any(kw in msg_lower for kw in _TOOL_KEYWORDS)
    needs_web = any(kw in msg_lower for kw in _WEB_SEARCH_KEYWORDS)
    is_code = any(kw in msg_lower for kw in _CODE_KEYWORDS)
    is_complex = any(kw in msg_lower for kw in _COMPLEXITY_KEYWORDS)
    # S49: Advanced-reasoning need detection
    needs_reasoning = any(kw in msg_lower for kw in _REASONING_KEYWORDS)

    # Length heuristic: long messages are often complex
    word_count = len(message.split())
    if word_count > 80:
        is_complex = True
    # S49: Very long messages with several questions suggest reasoning
    if word_count > 100 and message.count("?") >= 2:
        needs_reasoning = True

    # Inline code blocks present
    if "```" in message:
        is_code = True

    return {
        "needs_tools": needs_tools,
        "needs_web": needs_web,
        "is_code": is_code,
        "is_complex": is_complex,
        "needs_reasoning": needs_reasoning,
    }


def _select_pipeline(
    classification: dict,
    think_override: bool | None,
    web_search_override: bool | None,
    tool_executor_available: bool,
    verification_available: bool,
    reasoning_available: bool = False,
    sandbox_active: bool = False,
) -> str:
    """Select the optimal pipeline according to classification.

    Args:
        classification: Result of _quick_classify or a converted TaskAnalysis
        think_override: True/False forces, None = auto
        web_search_override: True/False forces, None = auto
        tool_executor_available: Whether the ToolExecutor is available
        verification_available: Whether the VerificationEngine is available
        reasoning_available: Whether the ReasoningEngine is available (S49)

    Returns:
        Pipeline name to use (PIPELINE_*)
    """
    # When the disposable sandbox is active, the file/exec tools are always
    # available and the MODEL decides whether to call them. This drops the
    # classifier as a gate on tool ACCESS, so create/read/edit/modify/run
    # works in any language with no keyword misses; the classifier still
    # shapes the rest (think vs direct, web, reasoning).
    tools_armed = tool_executor_available and (
        bool(classification.get("needs_tools")) or sandbox_active
    )

    # Explicit overrides take priority
    if web_search_override is True:
        return PIPELINE_WEB_SEARCH

    if think_override is True:
        # Think + tools when available
        if tools_armed:
            return PIPELINE_THINK_TOOLS
        return PIPELINE_THINK

    if think_override is False and web_search_override is False:
        # Forced off -- but tools stay available (CODE_VERIFY is unsafe in
        # sandbox mode, so armed tools take precedence over it).
        if tools_armed:
            return PIPELINE_TOOLS
        if classification.get("is_code") and verification_available:
            return PIPELINE_CODE_VERIFY
        return PIPELINE_DIRECT

    # Auto-detection
    if classification.get("needs_web"):
        return PIPELINE_WEB_SEARCH

    # S49: Advanced reasoning when detected and available
    if classification.get("needs_reasoning") and reasoning_available:
        return PIPELINE_REASONING

    if tools_armed:
        if classification.get("is_complex"):
            return PIPELINE_THINK_TOOLS
        return PIPELINE_TOOLS

    if classification.get("is_complex"):
        return PIPELINE_THINK

    if classification.get("is_code") and verification_available:
        return PIPELINE_CODE_VERIFY

    return PIPELINE_DIRECT


# =============================================================================
# AGENTIC EXECUTOR
# =============================================================================

class AgenticExecutor:
    """Unified agentic executor.

    The single entry point that orchestrates every capability:
    tool calling, code verification, thinking, web search.
    Automatically determines the optimal pipeline or honours the
    explicit user overrides.
    """

    def __init__(
        self,
        executor=None,
        tool_executor=None,
        structured_engine=None,
        verification_engine=None,
        reasoning_engine=None,
        consensus_engine=None,
        self_correction_engine=None,
        cascading_inference=None,
        speculative_generator=None,
        default_model: str = "qwen3:32b",
    ):
        """Initialize the agentic executor.

        Args:
            executor: Executor instance (or None for the singleton)
            tool_executor: ToolExecutor instance (or None for the singleton)
            structured_engine: StructuredOutputEngine instance (or None)
            verification_engine: VerificationEngine instance (or None)
            reasoning_engine: ReasoningEngine instance (or None, S49)
            consensus_engine: ConsensusEngine instance (or None, S50)
            self_correction_engine: SelfCorrectionEngine instance (or None, S51)
            cascading_inference: CascadingInference instance (or None, S69)
            speculative_generator: SpeculativeGenerator instance (or None, S70)
            default_model: Default model for task analysis
        """
        self._executor = executor or _default_executor
        self._tool_executor = tool_executor or _default_tool_executor
        self._structured_engine = structured_engine or _default_structured_engine
        self._verification_engine = verification_engine or _default_verification_engine
        self._reasoning_engine = reasoning_engine or _default_reasoning_engine
        self._consensus_engine = consensus_engine or _default_consensus_engine
        self._self_correction_engine = self_correction_engine or _default_self_correction_engine
        self._cascading_inference = cascading_inference or _default_cascading_inference
        self._speculative_generator = speculative_generator or _default_speculative_generator
        self._default_model = default_model

        # Results of the last execution
        self._last_tool_calls: list = []
        self._last_verification_results: list = []
        self._last_pipeline: str = PIPELINE_DIRECT
        self._last_task_analysis: Any | None = None
        # S49: Last reasoning result
        self._last_reasoning_result: Any | None = None
        # S50: Last consensus result
        self._last_consensus_result: Any | None = None
        # S51: Last self-correction result
        self._last_correction_result: Any | None = None
        # S69: Last cascading result
        self._last_cascade_result: Any | None = None
        # S70: Last speculative result
        self._last_speculative_result: Any | None = None

        # S62: Per-conversation tool call history for multi-turn tool use
        self._tool_call_history: dict[str, list] = {}
        self._max_history_per_conversation: int = 20

        # Callback for real-time events (tool calls, etc.)
        self._on_tool_call: Callable | None = None
        # S49: Callback for the reasoning steps
        self._on_reasoning_step: Callable | None = None

    # -----------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------

    @property
    def last_tool_calls(self) -> list:
        """Tool calls made during the last execution."""
        return self._last_tool_calls

    @property
    def last_verification_results(self) -> list:
        """Verification results of the last execution."""
        return self._last_verification_results

    @property
    def last_pipeline(self) -> str:
        """Pipeline used during the last execution."""
        return self._last_pipeline

    @property
    def last_task_analysis(self) -> Any | None:
        """TaskAnalysis of the last execution (or None)."""
        return self._last_task_analysis

    @property
    def last_prompt_budget(self):
        """S65: Last calculated PromptTokenBudget from the executor, or None."""
        if self._executor is not None and hasattr(self._executor, "last_prompt_budget"):
            return self._executor.last_prompt_budget
        return None

    @property
    def last_compression_result(self):
        """S66: Last CompressedContext from the executor, or None."""
        if self._executor is not None and hasattr(self._executor, "last_compression_result"):
            return self._executor.last_compression_result
        return None

    @property
    def s68_cache_hit(self) -> bool:
        """S68: Whether the last call was served from the S68 semantic cache."""
        if self._executor is not None and hasattr(self._executor, "s68_cache_hit"):
            return self._executor.s68_cache_hit
        return False

    @property
    def s68_cache_key(self) -> str:
        """S68: The cache key used for the last S68 cache lookup."""
        if self._executor is not None and hasattr(self._executor, "s68_cache_key"):
            return self._executor.s68_cache_key
        return ""

    @property
    def last_cascade_result(self):
        """S69: Last CascadeResult from cascading inference, or None."""
        return self._last_cascade_result

    @property
    def cascading_available(self) -> bool:
        """S69: Whether cascading inference is available and enabled."""
        return (
            CASCADING_INFERENCE_AVAILABLE
            and self._cascading_inference is not None
            and self._cascading_inference.enabled
        )

    @property
    def last_speculative_result(self):
        """S70: Last SpeculativeResult from speculative generation, or None."""
        return self._last_speculative_result

    @property
    def speculative_available(self) -> bool:
        """S70: Whether speculative generation is available and enabled."""
        return (
            SPECULATIVE_GENERATION_AVAILABLE
            and self._speculative_generator is not None
            and self._speculative_generator.enabled
        )

    @property
    def available(self) -> bool:
        """Whether the agentic executor is operational.

        Requires at least the base Executor.
        """
        return self._executor is not None

    @property
    def tool_executor_available(self) -> bool:
        """Whether the ToolExecutor is available."""
        return (
            TOOL_EXECUTOR_AVAILABLE
            and self._tool_executor is not None
        )

    @property
    def verification_available(self) -> bool:
        """Whether the VerificationEngine is available.

        SECURITY (S73): Returns False when the tool_registry is in
        sandbox mode. This prevents the code_verify pipeline from
        auto-executing LLM-generated code on the host, bypassing
        the sandbox. In sandbox mode, code execution must go through
        sandbox_bash instead.
        """
        # Block verification when sandbox mode is active
        if self._is_sandbox_mode_active():
            return False

        return (
            VERIFICATION_AVAILABLE
            and self._verification_engine is not None
            and hasattr(self._verification_engine, 'available')
            and self._verification_engine.available
        )

    @property
    def structured_available(self) -> bool:
        """Whether the StructuredOutputEngine is available."""
        return (
            STRUCTURED_OUTPUT_AVAILABLE
            and self._structured_engine is not None
        )

    @property
    def reasoning_available(self) -> bool:
        """Whether the ReasoningEngine is available (S49)."""
        return (
            REASONING_AVAILABLE
            and self._reasoning_engine is not None
            and self._reasoning_engine.available
        )

    def _is_sandbox_mode_active(self) -> bool:
        """Check if the tool_registry is in sandbox mode.

        When sandbox mode is active, code verification and other
        unsandboxed execution paths must be blocked to prevent
        the LLM from bypassing the sandbox.

        Returns:
            True if sandbox mode is active.
        """
        try:
            registry = _default_tool_registry
            if registry is not None and hasattr(registry, 'sandbox_mode'):
                return registry.sandbox_mode
        except Exception:
            pass
        return False

    def _is_quick_sandbox_active(self) -> bool:
        """Check if the tool_registry is in QUICK sandbox mode.

        Quick sandbox redirects the file/exec tools to a disposable
        sandbox. When it is on, those tools are always available and the
        model decides whether to call them, so the tools pipeline must be
        armed for every turn -- independently of the hard sandbox flag.
        """
        try:
            registry = _default_tool_registry
            if registry is not None and hasattr(
                registry, 'quick_sandbox_mode'
            ):
                return registry.quick_sandbox_mode
        except Exception:
            pass
        return False

    @property
    def last_reasoning_result(self) -> Any | None:
        """Last reasoning result (S49)."""
        return self._last_reasoning_result

    @property
    def consensus_available(self) -> bool:
        """Whether the ConsensusEngine is available (S50)."""
        return (
            CONSENSUS_AVAILABLE
            and self._consensus_engine is not None
            and self._consensus_engine.available
        )

    @property
    def last_consensus_result(self) -> Any | None:
        """Last consensus result (S50)."""
        return self._last_consensus_result

    @property
    def self_correction_available(self) -> bool:
        """Whether the SelfCorrectionEngine is available (S51)."""
        return (
            SELF_CORRECTION_AVAILABLE
            and self._self_correction_engine is not None
            and self._self_correction_engine.available
        )

    @property
    def last_correction_result(self) -> Any | None:
        """Last self-correction result (S51)."""
        return self._last_correction_result

    # -----------------------------------------------------------------
    # S62: Multi-turn tool call history
    # -----------------------------------------------------------------

    def get_tool_history(self, conversation_id: str) -> list:
        """Get prior tool call results for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of ToolCallResult objects from prior turns
        """
        if not conversation_id:
            return []
        return list(self._tool_call_history.get(conversation_id, []))

    def clear_tool_history(self, conversation_id: str) -> int:
        """Clear tool call history for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Number of entries cleared
        """
        if conversation_id in self._tool_call_history:
            count = len(self._tool_call_history[conversation_id])
            del self._tool_call_history[conversation_id]
            return count
        return 0

    def clear_all_tool_history(self) -> int:
        """Clear all tool call history across all conversations.

        Returns:
            Number of conversations cleared
        """
        count = len(self._tool_call_history)
        self._tool_call_history.clear()
        return count

    def _record_tool_calls(
        self, conversation_id: str | None, tool_calls: list,
    ) -> None:
        """Record tool calls in the per-conversation history.

        Args:
            conversation_id: Conversation identifier
            tool_calls: List of ToolCallResult from current turn
        """
        if not conversation_id or not tool_calls:
            return

        if conversation_id not in self._tool_call_history:
            self._tool_call_history[conversation_id] = []

        self._tool_call_history[conversation_id].extend(tool_calls)

        # Trim to max history size
        history = self._tool_call_history[conversation_id]
        if len(history) > self._max_history_per_conversation:
            self._tool_call_history[conversation_id] = history[
                -self._max_history_per_conversation:
            ]

    # -----------------------------------------------------------------
    # Task analysis
    # -----------------------------------------------------------------

    def analyze_task(self, message: str, model: str | None = None) -> Any | None:
        """Analyze the request to determine the optimal pipeline.

        Uses the StructuredOutputEngine when available, otherwise
        returns None (fast heuristics will be used).

        Args:
            message: User message
            model: Model for the analysis (default: self._default_model)

        Returns:
            TaskAnalysis or None when the structured analysis failed
        """
        if not self.structured_available or TaskAnalysis is None:
            return None

        _model = model or self._default_model

        try:
            result = self._structured_engine.generate_structured(
                messages=[{"role": "user", "content": message}],
                schema=TaskAnalysis,
                model=_model,
                extra_system_prompt=(
                    "Analyze this user message and determine:\n"
                    "- task_type: what kind of task this is\n"
                    "- complexity: simple, moderate, or complex\n"
                    "- requires_tools: list of tool names needed "
                    "(web_search, code_execution, read_file, etc.) or empty\n"
                    "- requires_thinking: whether deep reasoning is needed\n"
                    "- language: detected language (en, fr, auto)\n"
                    "- confidence: your confidence 0.0-1.0"
                ),
                temperature=0.0,
                max_retries=2,
            )
            if result.success and result.data:
                return result.data
        except Exception as e:
            logger.warning(f"Task analysis failed: {e}")

        return None

    def _classify_message(
        self,
        message: str,
        model: str | None = None,
        use_llm: bool = True,
    ) -> dict:
        """Classify the request by combining heuristics and the LLM.

        Tries the LLM analysis first, then falls back to the fast
        heuristics when it is unavailable.

        Returns:
            dict with needs_tools, needs_web, is_code, is_complex
        """
        # Fast heuristics (always available)
        quick = _quick_classify(message)

        # Try the LLM analysis when requested and available
        if use_llm and self.structured_available:
            analysis = self.analyze_task(message, model)
            if analysis is not None:
                self._last_task_analysis = analysis

                # Convert the TaskAnalysis into a classification dict
                return {
                    "needs_tools": bool(analysis.requires_tools),
                    "needs_web": "web_search" in analysis.requires_tools,
                    "is_code": analysis.task_type in (
                        "code_python", "code_r", "debug", "code",
                        "code_review", "refactor",
                    ),
                    "is_complex": (
                        analysis.complexity == "complex"
                        or analysis.requires_thinking
                    ),
                    "task_analysis": analysis,
                }

        return quick

    # -----------------------------------------------------------------
    # Main execution
    # -----------------------------------------------------------------

    def execute(
        self,
        message: str,
        routing: Any,
        conversation_id: str | None = None,
        think: bool | None = None,
        web_search: bool | None = None,
        consensus: bool | None = None,
        consensus_models: list[str] | None = None,
        consensus_strategy: str | None = None,
        self_correct: bool | None = None,
        cascading: bool | None = None,
        speculative: bool | None = None,
        on_status: Callable[[str], None] | None = None,
        on_tool_call: Callable | None = None,
        on_reasoning_step: Callable | None = None,
        on_consensus_model: Callable | None = None,
        on_correction_step: Callable | None = None,
        use_llm_analysis: bool = False,
        approval_fn: Callable[[str, dict], bool] | None = None,
    ) -> Generator:
        """Execute with intelligent pipeline selection.

        Yields streaming chunks:
        - str for normal tokens
        - ("thinking", str) for the thinking content
        - ("reasoning_step", ReasoningStep) for the reasoning steps (S49)
        - ("reasoning_done", ReasoningResult) end of reasoning (S49)
        - ("consensus_model_done", ModelResponse) individual response (S50)
        - ("consensus_done", ConsensusResult) consensus result (S50)
        - ("correction_step", dict) self-correction step (S51)
        - ("correction_done", SelfCorrectionResult) correction result (S51)
        - ("cascade_done", CascadeResult) cascading result (S69)
        Tool calls are signalled via the on_tool_call callback.

        Args:
            message: User message
            routing: RoutingResult from the router
            conversation_id: Optional conversation ID
            think: None = auto-decide, True = force, False = disable
            web_search: None = auto-decide, True = force, False = disable
            consensus: None = auto, True = force, False = disable (S50)
            consensus_models: Model list for the consensus (S50)
            consensus_strategy: Consensus strategy (S50)
            self_correct: None = auto, True = force, False = disable (S51)
            cascading: None = auto, True = force, False = disable (S69)
            on_status: Optional status callback
            on_tool_call: Callback invoked for each tool call
                Signature: on_tool_call(tool_call_result: ToolCallResult)
            on_reasoning_step: Callback for each reasoning step (S49)
                Signature: on_reasoning_step(step: ReasoningStep)
            on_consensus_model: Callback for each model response (S50)
                Signature: on_consensus_model(model_response: ModelResponse)
            on_correction_step: Callback for each correction step (S51)
                Signature: on_correction_step(step_info: dict)
            use_llm_analysis: When True, use the LLM to analyze the task
                (slower but more precise). Defaults to False for speed.
            approval_fn: Optional per-invocation tool-approval gate
                (tool_name, arguments) -> bool, forwarded to the tool executor.
                S185 (EX-02): bound to this call rather than to a shared
                singleton attribute, so concurrent Bulbe sessions cannot
                clobber or drop each other's gate.

        Yields:
            Response chunks (str or tuples)
        """
        start_time = time.time()

        # Reset the results of the last execution
        self._last_tool_calls = []
        self._last_verification_results = []
        self._last_task_analysis = None
        # S49: Last reasoning result
        self._last_reasoning_result = None
        # S50: Last consensus result
        self._last_consensus_result = None
        # S51: Last self-correction result
        self._last_correction_result = None
        # S69: Last cascading result
        self._last_cascade_result = None
        # S70: Last speculative result
        self._last_speculative_result = None
        self._on_tool_call = on_tool_call
        self._on_reasoning_step = on_reasoning_step
        self._on_consensus_model = on_consensus_model
        self._on_correction_step = on_correction_step

        # Check the base executor is available
        if self._executor is None:
            yield "[ERR] Executor not available"
            return

        # S70: Speculative override -- if explicitly requested, use
        # speculative pipeline directly (mutually exclusive with cascading)
        if speculative is True and self.speculative_available:
            self._last_pipeline = PIPELINE_SPECULATIVE
            logger.info("AgenticExecutor: explicit speculative, pipeline=speculative")
            yield from self._execute_speculative_pipeline(
                message, routing, conversation_id, on_status,
            )
            duration = time.time() - start_time
            logger.info(
                f"AgenticExecutor: finished in {duration:.2f}s, "
                f"pipeline={PIPELINE_SPECULATIVE}"
            )
            return

        # S69: Cascading override -- when explicitly requested, use
        # the cascading pipeline directly
        if cascading is True and self.cascading_available:
            self._last_pipeline = PIPELINE_CASCADING
            logger.info("AgenticExecutor: explicit cascading, pipeline=cascading")
            yield from self._execute_cascading_pipeline(
                message, routing, conversation_id, on_status,
            )
            duration = time.time() - start_time
            logger.info(
                f"AgenticExecutor: finished in {duration:.2f}s, "
                f"pipeline={PIPELINE_CASCADING}"
            )
            return

        # S51: Self-correct override -- when explicitly requested, use
        # the self_correct pipeline directly
        if self_correct is True and self.self_correction_available:
            self._last_pipeline = PIPELINE_SELF_CORRECT
            logger.info("AgenticExecutor: explicit self_correct, pipeline=self_correct")
            yield from self._execute_self_correct_pipeline(
                message, routing, conversation_id, on_status,
            )
            duration = time.time() - start_time
            logger.info(
                f"AgenticExecutor: finished in {duration:.2f}s, "
                f"pipeline={PIPELINE_SELF_CORRECT}"
            )
            return

        # S50: Consensus override -- when explicitly requested, use
        # the consensus pipeline directly without classification
        if consensus is True and self.consensus_available:
            self._last_pipeline = PIPELINE_CONSENSUS
            logger.info("AgenticExecutor: explicit consensus, pipeline=consensus")
            yield from self._execute_consensus_pipeline(
                message, routing, conversation_id, on_status,
                models=consensus_models,
                strategy=consensus_strategy,
            )
            duration = time.time() - start_time
            logger.info(
                f"AgenticExecutor: finished in {duration:.2f}s, "
                f"pipeline={PIPELINE_CONSENSUS}"
            )
            return

        # Classify the request
        classification = self._classify_message(
            message,
            model=getattr(routing, 'model', self._default_model),
            use_llm=use_llm_analysis,
        )

        # Select the pipeline
        pipeline = _select_pipeline(
            classification=classification,
            think_override=think,
            web_search_override=web_search,
            tool_executor_available=self.tool_executor_available,
            verification_available=self.verification_available,
            reasoning_available=self.reasoning_available,
            sandbox_active=(
                self._is_sandbox_mode_active()
                or self._is_quick_sandbox_active()
            ),
        )
        self._last_pipeline = pipeline

        logger.info(
            f"AgenticExecutor: pipeline={pipeline}, "
            f"classification={classification}"
        )

        # Dispatch to the selected pipeline
        if pipeline == PIPELINE_TOOLS:
            yield from self._execute_tools_pipeline(
                message, routing, conversation_id, on_status,
                approval_fn=approval_fn,
            )

        elif pipeline == PIPELINE_THINK_TOOLS:
            yield from self._execute_think_tools_pipeline(
                message, routing, conversation_id, on_status,
                approval_fn=approval_fn,
            )

        elif pipeline == PIPELINE_REASONING:
            yield from self._execute_reasoning_pipeline(
                message, routing, conversation_id, on_status,
            )

        elif pipeline == PIPELINE_THINK:
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
                think=True, web_search=False,
                verify_code=classification.get("is_code", False),
            )

        elif pipeline == PIPELINE_WEB_SEARCH:
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
                think=False, web_search=True,
                verify_code=False,
            )

        elif pipeline == PIPELINE_CODE_VERIFY:
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
                think=False, web_search=False,
                verify_code=True,
            )

        else:
            # PIPELINE_DIRECT
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
                think=False, web_search=False,
                verify_code=False,
            )

        duration = time.time() - start_time
        logger.info(
            f"AgenticExecutor: finished in {duration:.2f}s, "
            f"pipeline={pipeline}, "
            f"tool_calls={len(self._last_tool_calls)}, "
            f"verifications={len(self._last_verification_results)}"
        )

    # -----------------------------------------------------------------
    # Execution pipelines
    # -----------------------------------------------------------------

    def _execute_direct_pipeline(
        self,
        message: str,
        routing: Any,
        conversation_id: str | None,
        on_status: Callable | None,
        think: bool = False,
        web_search: bool = False,
        verify_code: bool = False,
    ) -> Generator:
        """Direct pipeline: LLM call via the existing Executor.

        Optionally enables the think mode, the web search, and/or
        the code verification after the generation.
        """
        full_response = ""

        try:
            gen = self._executor.execute(
                question=message,
                routing=routing,
                document=None,
                refine=False,
                conversation_id=conversation_id,
                think=think,
                web_search=web_search,
                on_status=on_status,
            )
            for chunk in gen:
                if chunk:
                    if isinstance(chunk, tuple):
                        # Thinking chunk or other tuple
                        yield chunk
                        if len(chunk) == 2 and chunk[0] != "thinking":
                            full_response += chunk[1]
                    else:
                        full_response += chunk
                        yield chunk

        except Exception as e:
            logger.error(f"Direct pipeline error: {e}")
            yield f"\n\n[Error: {e}]"
            return

        # Transfer verification results from the executor
        if hasattr(self._executor, 'last_verification_results'):
            self._last_verification_results = self._executor.last_verification_results

        # Additional code verification when requested and not already done
        if (
            verify_code
            and not self._last_verification_results
            and self.verification_available
            and full_response
        ):
            self._run_code_verification(
                full_response, message,
                getattr(routing, 'model', self._default_model),
            )

    def _execute_tools_pipeline(
        self,
        message: str,
        routing: Any,
        conversation_id: str | None,
        on_status: Callable | None,
        approval_fn: Callable[[str, dict], bool] | None = None,
    ) -> Generator:
        """Tools pipeline: ReAct loop via the ToolExecutor.

        Executes the necessary tools, then yields the final response.
        Tool calls are signalled via callback and stored.
        S62: Prior tool call history is passed for multi-turn context.
        S261: approval_fn is the per-invocation tool-approval gate
        (S185, EX-02), threaded from execute() and forwarded to
        execute_with_tools; the pre-S261 dispatch dropped it and this
        method then raised an unbound-name error into its fallback.
        """
        if not self.tool_executor_available:
            # Fall back to the direct pipeline
            logger.warning("ToolExecutor unavailable, falling back to direct")
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
            )
            return

        model = getattr(routing, 'model', self._default_model)

        # Fetch the conversation context when available
        conv_messages = self._get_conversation_context(conversation_id)

        # S62: Retrieve prior tool call history for multi-turn context
        prior_tool_history = self.get_tool_history(conversation_id) if conversation_id else []

        try:
            result = self._tool_executor.execute_with_tools(
                message=message,
                model=model,
                conversation_messages=conv_messages,
                tool_history=prior_tool_history if prior_tool_history else None,
                approval_fn=approval_fn,
            )

            # Store the tool calls
            self._last_tool_calls = list(result.tool_calls)

            # S62: Record tool calls in per-conversation history
            self._record_tool_calls(conversation_id, result.tool_calls)

            # Signal each tool call via the callback
            for tc in result.tool_calls:
                self._emit_tool_call(tc)

            # Yield the final response
            if result.response:
                yield result.response

            # Save to the conversation when needed
            self._save_to_conversation(
                conversation_id, message, result.response, model,
            )

        except Exception as e:
            logger.error(f"Tools pipeline error: {e}")
            # Fall back to the direct pipeline
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
            )

    def _execute_think_tools_pipeline(
        self,
        message: str,
        routing: Any,
        conversation_id: str | None,
        on_status: Callable | None,
        approval_fn: Callable[[str, dict], bool] | None = None,
    ) -> Generator:
        """Think+tools pipeline: thinking first, then tools when needed.

        Combines the think mode with tool use for complex requests
        that need both reasoning and execution.
        S261: approval_fn is the per-invocation tool-approval gate
        (S185, EX-02), threaded from execute() into the tools phase.
        """
        # Phase 1: direct call with think mode for the reasoning pass
        full_response = ""

        try:
            gen = self._executor.execute(
                question=message,
                routing=routing,
                document=None,
                refine=False,
                conversation_id=conversation_id,
                think=True,
                web_search=False,
                on_status=on_status,
            )
            for chunk in gen:
                if chunk:
                    if isinstance(chunk, tuple):
                        yield chunk
                        if len(chunk) == 2 and chunk[0] != "thinking":
                            full_response += chunk[1]
                    else:
                        full_response += chunk
                        yield chunk

        except Exception as e:
            logger.error(f"Think phase error in think+tools pipeline: {e}")
            yield f"\n\n[Error during thinking: {e}]"
            return

        # Phase 2: run the tools when the tool_executor detects a need
        if (
            self.tool_executor_available
            and self._tool_executor.should_use_tools(
                message, getattr(routing, 'model', self._default_model)
            )
        ):
            model = getattr(routing, 'model', self._default_model)
            conv_messages = self._get_conversation_context(conversation_id)

            # S62: Prior tool call history for multi-turn context
            prior_tool_history = self.get_tool_history(conversation_id) if conversation_id else []

            try:
                result = self._tool_executor.execute_with_tools(
                    message=message,
                    model=model,
                    conversation_messages=conv_messages,
                    tool_history=prior_tool_history if prior_tool_history else None,
                    approval_fn=approval_fn,
                )
                self._last_tool_calls = list(result.tool_calls)

                # S62: Record tool calls in per-conversation history
                self._record_tool_calls(conversation_id, result.tool_calls)

                for tc in result.tool_calls:
                    self._emit_tool_call(tc)

                # When the tools produced results, append them
                if result.tool_calls and result.response:
                    yield "\n\n"
                    yield result.response

            except Exception as e:
                logger.warning(f"Tools phase failed (think+tools): {e}")

        # Transfer the verifications from the executor
        if hasattr(self._executor, 'last_verification_results'):
            self._last_verification_results = self._executor.last_verification_results

    def _execute_reasoning_pipeline(
        self,
        message: str,
        routing: Any,
        conversation_id: str | None,
        on_status: Callable | None,
    ) -> Generator:
        """Reasoning pipeline (S49): multi-step decomposition and solving.

        Uses the ReasoningEngine to decompose the request into
        sub-steps, solve them sequentially, then synthesize a final
        response. Steps are emitted via the on_reasoning_step callback.
        """
        if not self.reasoning_available:
            # Fall back to the think pipeline
            logger.warning("ReasoningEngine unavailable, falling back to think")
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
                think=True, web_search=False, verify_code=False,
            )
            return

        model = getattr(routing, 'model', self._default_model)

        def _on_step(step):
            """Internal callback emitting the reasoning steps."""
            if self._on_reasoning_step is not None:
                try:
                    self._on_reasoning_step(step)
                except Exception as e:
                    logger.debug(f"reasoning_step callback failed: {e}")

        try:
            gen = self._reasoning_engine.execute_reasoning(
                question=message,
                # RSN-04 (S192): None lets the engine resolve the strategy
                # from reasoning.yaml (default_strategy) instead of pinning
                # "decompose" and leaving the other strategies unreachable.
                strategy=None,
                model=model,
                on_step=_on_step,
            )

            full_response = ""
            for chunk in gen:
                if isinstance(chunk, tuple):
                    chunk_type, chunk_data = chunk
                    if chunk_type == "reasoning_step":
                        yield ("reasoning_step", chunk_data)
                    elif chunk_type == "reasoning_done":
                        self._last_reasoning_result = chunk_data
                        yield ("reasoning_done", chunk_data)
                else:
                    # Final response text
                    full_response += chunk
                    yield chunk

            # Save to the conversation
            self._save_to_conversation(
                conversation_id, message, full_response, model,
            )

        except Exception as e:
            logger.error(f"Reasoning pipeline error: {e}")
            # Fall back to the direct pipeline with think
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
                think=True, web_search=False, verify_code=False,
            )

    # -----------------------------------------------------------------
    # Consensus pipeline (S50)
    # -----------------------------------------------------------------

    def _execute_consensus_pipeline(
        self,
        message: str,
        routing: Any,
        conversation_id: str | None,
        on_status: Callable | None,
        models: list[str] | None = None,
        strategy: str | None = None,
    ) -> Generator:
        """Consensus pipeline (S50): query N models and merge.

        Uses the ConsensusEngine to query several models in parallel,
        compare the responses, and select the best one.
        """
        if not self.consensus_available:
            # Fall back to the direct pipeline
            logger.warning("ConsensusEngine unavailable, falling back to direct")
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
                think=False, web_search=False, verify_code=False,
            )
            return

        model = getattr(routing, 'model', self._default_model)

        def _on_model_done(model_resp):
            """Internal callback emitting the individual responses."""
            if self._on_consensus_model is not None:
                try:
                    self._on_consensus_model(model_resp)
                except Exception as e:
                    logger.debug(f"consensus_model callback failed: {e}")

        try:
            gen = self._consensus_engine.execute_consensus(
                query=message,
                models=models,
                strategy=strategy,
                on_model_done=_on_model_done,
            )

            full_response = ""
            for chunk in gen:
                if isinstance(chunk, tuple):
                    chunk_type, chunk_data = chunk
                    if chunk_type == "consensus_model_done":
                        yield ("consensus_model_done", chunk_data)
                    elif chunk_type == "consensus_done":
                        self._last_consensus_result = chunk_data
                        yield ("consensus_done", chunk_data)
                else:
                    # Selected response text
                    full_response += chunk
                    yield chunk

            # Save to the conversation
            self._save_to_conversation(
                conversation_id, message, full_response, model,
            )

        except Exception as e:
            logger.error(f"Consensus pipeline error: {e}")
            # Fall back to the direct pipeline
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
                think=False, web_search=False, verify_code=False,
            )

    # -----------------------------------------------------------------
    # Self-correction pipeline (S51)
    # -----------------------------------------------------------------

    def _execute_self_correct_pipeline(
        self,
        message: str,
        routing: Any,
        conversation_id: str | None,
        on_status: Callable | None,
    ) -> Generator:
        """Self-correction pipeline (S51): generate then correct.

        1. Generate the initial response via the direct pipeline
        2. Run the self-correction over the response
        3. Return the best version

        Falls back to direct when the SelfCorrectionEngine is
        unavailable.
        """
        if not self.self_correction_available:
            logger.warning("SelfCorrectionEngine unavailable, falling back to direct")
            yield from self._execute_direct_pipeline(
                message, routing, conversation_id, on_status,
                think=False, web_search=False, verify_code=False,
            )
            return

        model = getattr(routing, 'model', self._default_model)

        # Phase 1: generate the initial response (no streaming)
        if on_status:
            on_status("Generating initial response...")

        initial_response = ""
        try:
            gen = self._executor.execute(
                question=message,
                routing=routing,
                document=None,
                refine=False,
                conversation_id=conversation_id,
                think=False,
                web_search=False,
                on_status=on_status,
            )
            for chunk in gen:
                if chunk:
                    if isinstance(chunk, tuple):
                        if len(chunk) == 2 and chunk[0] != "thinking":
                            initial_response += chunk[1]
                    else:
                        initial_response += chunk
        except Exception as e:
            logger.error(f"Initial generation error: {e}")
            yield f"\n\n[Error: {e}]"
            return

        if not initial_response:
            yield "[ERR] No initial response generated"
            return

        # Phase 2: self-correction
        if on_status:
            on_status("Running self-correction...")

        try:
            gen = self._self_correction_engine.execute_self_correction(
                user_message=message,
                response=initial_response,
                model=model,
            )

            full_response = ""
            for chunk in gen:
                if isinstance(chunk, tuple):
                    chunk_type, chunk_data = chunk
                    if chunk_type == "correction_step":
                        yield ("correction_step", chunk_data)
                        if self._on_correction_step is not None:
                            try:
                                self._on_correction_step(chunk_data)
                            except Exception as e:
                                logger.debug(f"correction_step callback failed: {e}")
                    elif chunk_type == "correction_done":
                        self._last_correction_result = chunk_data
                        yield ("correction_done", chunk_data)
                else:
                    full_response += chunk
                    yield chunk

            # Save to the conversation
            self._save_to_conversation(
                conversation_id, message, full_response, model,
            )

        except Exception as e:
            logger.error(f"self_correct pipeline error: {e}")
            # Fallback: stream the initial response
            yield initial_response
            self._save_to_conversation(
                conversation_id, message, initial_response, model,
            )

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def _run_code_verification(
        self, response_text: str, question: str, model: str,
    ) -> None:
        """Run the code verification over the response."""
        if not self.verification_available:
            return

        try:
            results = self._verification_engine.verify_response_code_blocks(
                response_text=response_text,
                original_question=question,
                model=model,
                timeout=30,
            )
            if results:
                self._last_verification_results = results
                for vr in results:
                    logger.info(
                        f"AgenticExecutor verification ({vr.language}): "
                        f"status={vr.status}, iterations={vr.iterations}"
                    )
        except Exception as e:
            logger.warning(f"Code verification failed: {e}")

    def _emit_tool_call(self, tool_call_result) -> None:
        """Signal a tool call via the callback."""
        if self._on_tool_call is not None:
            try:
                self._on_tool_call(tool_call_result)
            except Exception as e:
                logger.warning(f"tool_call callback error: {e}")

    def _get_conversation_context(
        self, conversation_id: str | None,
    ) -> list[dict]:
        """Retrieve conversation context for the ToolExecutor.

        Convert conversation messages to Ollama format.
        """
        if not conversation_id:
            return []

        try:
            from .conversation import conversation_manager
            if conversation_manager is None:
                return []

            messages = conversation_manager.get_messages(conversation_id)
            if not messages:
                return []

            # Convert to Ollama format (role + content)
            ollama_messages = []
            for msg in messages[-10:]:  # Keep the last 10 messages
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
            return ollama_messages

        except Exception as e:
            logger.debug(f"Could not load the conversation context: {e}")
            return []

    def _save_to_conversation(
        self,
        conversation_id: str | None,
        user_message: str,
        assistant_response: str,
        model: str,
    ) -> None:
        """Save messages in the conversation.

        Used by the pipelines that do not go through the Executor
        (which handles saving internally).
        """
        if not conversation_id or not assistant_response:
            return

        try:
            from .conversation import conversation_manager
            if conversation_manager is None:
                return

            # Save the user message
            conversation_manager.add_message(
                conv_id=conversation_id,
                role="user",
                content=user_message,
            )
            # Save the response
            conversation_manager.add_message(
                conv_id=conversation_id,
                role="assistant",
                content=assistant_response,
                metadata={"model": model},
            )

        except Exception as e:
            logger.debug(f"Conversation save failed: {e}")

    def _execute_cascading_pipeline(
        self,
        message: str,
        routing: Any,
        conversation_id: str | None,
        on_status: Callable | None,
    ) -> Generator:
        """S69: Cascading pipeline -- route through progressive tiers.

        Uses CascadingInference to try fast -> standard -> power models,
        stopping at the first tier whose response meets the quality threshold.
        """
        if not self.cascading_available:
            yield "[ERR] Cascading inference not available"
            return

        if on_status:
            on_status("[>] Running cascading inference...")

        try:
            result = self._cascading_inference.cascade(
                query=message,
                task_type=getattr(routing, "task_type", None),
            )
            self._last_cascade_result = result

            if result.final_response:
                yield result.final_response

            # Emit cascade result as structured event
            yield ("cascade_done", result)

            if on_status:
                on_status(
                    f"[OK] Cascade resolved at tier {result.tier_index} "
                    f"({result.tier_name}), score={result.score:.3f}, "
                    f"latency={result.total_latency_ms:.0f}ms"
                )

            # Also store on the base executor for property access
            if self._executor is not None and hasattr(self._executor, "_last_cascade_result"):
                self._executor._last_cascade_result = result

        except Exception as e:
            logger.error("Cascading pipeline error: %s", e)
            yield f"\n\n[ERR] Cascading inference failed: {e}"

    def _execute_speculative_pipeline(
        self,
        message: str,
        routing: Any,
        conversation_id: str | None,
        on_status: Callable | None,
    ) -> Generator:
        """S70: Speculative pipeline -- draft-verify pattern.

        Uses SpeculativeGenerator: a fast model drafts, a larger model verifies.
        Mutually exclusive with cascading inference (S69).
        """
        if not self.speculative_available:
            yield "[ERR] Speculative generation not available"
            return

        if on_status:
            on_status("[>] Running speculative generation (draft-verify)...")

        try:
            result = self._speculative_generator.generate(
                query=message,
                task_type=getattr(routing, "task_type", None),
            )
            self._last_speculative_result = result

            if result.final_response:
                yield result.final_response

            # Emit speculative result as structured event
            yield ("speculative_done", result)

            if on_status:
                accepted = "draft accepted" if result.draft_accepted else "verify used"
                on_status(
                    f"[OK] Speculative: {accepted}, iterations={result.iterations}, "
                    f"convergence={result.convergence_score:.3f}, "
                    f"latency={result.total_latency_ms:.0f}ms"
                )

            # Also store on the base executor for property access
            if self._executor is not None and hasattr(self._executor, "_last_speculative_result"):
                self._executor._last_speculative_result = result

        except Exception as e:
            logger.error("Speculative pipeline error: %s", e)
            yield f"\n\n[ERR] Speculative generation failed: {e}"

    def reset(self) -> None:
        """Reset state between executions."""
        self._last_tool_calls = []
        self._last_verification_results = []
        self._last_pipeline = PIPELINE_DIRECT
        self._last_task_analysis = None
        self._last_reasoning_result = None
        self._last_correction_result = None
        self._last_cascade_result = None
        self._last_speculative_result = None
        self._on_tool_call = None
        self._on_reasoning_step = None
        self._on_correction_step = None

    def cancel(self) -> None:
        """Cancel the in-flight execution by delegating to the executor."""
        if self._executor is not None and hasattr(self._executor, 'cancel'):
            self._executor.cancel()


# =============================================================================
# SINGLETON
# =============================================================================

agentic_executor = AgenticExecutor()
