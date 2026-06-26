#!/usr/bin/env python3
"""
EXECUTOR - OPTI-OIGNON 2.0
==========================

Execute queries via inference backends (Ollama, llama.cpp, etc.)
with appropriate system prompts.

This module handles:
- System prompt loading
- Question refinement
- Streaming query execution
- Error and timeout handling
- Request cancellation
- Context validation (NEW: Phase A4)
- Multi-backend inference (S105: backend abstraction layer)

MULTILINGUAL LOGIC:
- The code and interface are in English
- BUT if the user asks in French -> response in French
- If user asks in English -> response in English
- The system detects user language and responds accordingly

Author: Leon
"""

import hashlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Generator
from typing import Any, Optional

import ollama

from .config import config

# S193 TC-04: sentinel context fingerprint for execution paths that build no
# assembled system prompt (cascade S69, speculative S70). Partitions their
# cache entries away from full-context responses in both directions: a
# context-free lookup never serves a document/RAG-grounded response, and a
# full-context lookup never serves a context-free one.
_CTX_FP_NOCTX = hashlib.sha256(b"opti-oignon:no-context").hexdigest()
from .router import RoutingResult

# S105: Inference backend abstraction -- use backend when available,
# fall back to direct ollama calls for backward compatibility.
try:
    from .inference_backend import get_backend_registry
    INFERENCE_BACKEND_AVAILABLE = True
except ImportError:
    INFERENCE_BACKEND_AVAILABLE = False
    get_backend_registry = None

# Context management import
try:
    from .context_manager import (
        ContextCheck,
        get_context_manager,
    )
    from .context_manager import (
        check_context as cm_check_context,
    )
    from .context_manager import (
        estimate_tokens as cm_estimate_tokens,
    )
    from .context_manager import (
        get_model_limits as cm_get_model_limits,
    )
    from .context_manager import (
        smart_truncate as cm_smart_truncate,  # noqa: F401
    )
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    CONTEXT_MANAGER_AVAILABLE = False
    ContextCheck = None

# Conversation management import
try:
    from .conversation import conversation_manager
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False
    conversation_manager = None

# Context summarization import (v1.4.0 -- F2)
try:
    from .context_summary import (
        context_summarizer,
        extract_summary_text,
        is_summary_message,
    )
    CONTEXT_SUMMARY_AVAILABLE = True
except ImportError:
    CONTEXT_SUMMARY_AVAILABLE = False
    context_summarizer = None

# Cross-conversation memory import (v1.4.0 -- F1, Session 11)
try:
    from .memory import memory_manager as _memory_manager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    _memory_manager = None

# S174 (Theme 3): the S66 dual-layer working block from the new MemoryStore-backed
# retriever. The compressed block is injected into the prompt; the full archive
# stays searchable for recovery. The block is unwrapped (the agent wraps it as
# untrusted context, S175).
try:
    from .memory.retrieval import working_memory_block as _working_memory_block
    from .memory.retrieval import build_memory_block as _build_memory_block
    DUAL_LAYER_MEMORY_AVAILABLE = True
except Exception:
    DUAL_LAYER_MEMORY_AVAILABLE = False
    _working_memory_block = None
    _build_memory_block = None

# Intelligent sliding window (v1.4.0 -- S16/S17)
try:
    from .context_window import sliding_window_manager, token_budget_manager
    CONTEXT_WINDOW_AVAILABLE = True
except ImportError:
    CONTEXT_WINDOW_AVAILABLE = False
    sliding_window_manager = None
    token_budget_manager = None

# Response cache (v1.4.0 -- S18/C3)
try:
    from .response_cache import response_cache as _response_cache
    RESPONSE_CACHE_AVAILABLE = True
except ImportError:
    RESPONSE_CACHE_AVAILABLE = False
    _response_cache = None

# Semantic cache (v1.4.0 -- S23 G1)
try:
    from .semantic_cache import semantic_cache as _semantic_cache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False
    _semantic_cache = None

# Model warm-up / keepalive (v1.4.0 -- S24 F2)
try:
    from .model_warmup import MODEL_WARMUP_AVAILABLE
    from .model_warmup import model_warmup as _model_warmup
except ImportError:
    MODEL_WARMUP_AVAILABLE = False
    _model_warmup = None

# Code verification (v1.5.0 -- S43)
try:
    from .verification import verification_engine as _verification_engine
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False
    _verification_engine = None

# Project context injection (v1.5.9 -- S58)
try:
    from .project_context import project_context_builder as _project_context_builder
    from .project_triggers import trigger_detector as _trigger_detector
    from .projects import project_store as _project_store
    PROJECT_CONTEXT_AVAILABLE = True
except ImportError:
    PROJECT_CONTEXT_AVAILABLE = False
    _project_context_builder = None
    _trigger_detector = None
    _project_store = None

# Prompt optimization (v1.6.4 -- S65)
try:
    from .prompt_optimization import (
        prompt_budget_manager as _prompt_budget_manager,
    )
    from .prompt_optimization import (
        prompt_template_engine as _prompt_template_engine,
    )
    PROMPT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PROMPT_OPTIMIZATION_AVAILABLE = False
    _prompt_budget_manager = None
    _prompt_template_engine = None

# Conversation compressor (v1.6.5 -- S66)
try:
    from .conversation_compressor import (
        CompressedContext,
    )
    from .conversation_compressor import (
        check_retrieval_trigger as _check_retrieval_trigger,
    )
    from .conversation_compressor import (
        conversation_compressor as _conversation_compressor,
    )
    CONVERSATION_COMPRESSOR_AVAILABLE = True
except ImportError:
    CONVERSATION_COMPRESSOR_AVAILABLE = False
    _conversation_compressor = None
    CompressedContext = None
    _check_retrieval_trigger = None

# Cascading inference (v1.7.1 -- S69)
try:
    from .cascading import CascadeResult as _CascadeResult
    from .cascading import cascading_inference as _cascading_inference
    CASCADING_AVAILABLE = True
except ImportError:
    CASCADING_AVAILABLE = False
    _cascading_inference = None
    _CascadeResult = None

# Speculative generation (v1.7.2 -- S70)
try:
    from .speculative import SpeculativeResult as _SpeculativeResult
    from .speculative import speculative_generator as _speculative_generator
    SPECULATIVE_AVAILABLE = True
except ImportError:
    SPECULATIVE_AVAILABLE = False
    _speculative_generator = None
    _SpeculativeResult = None

# Resource Governor admission (S224, R-01 spec Section 4) -- the executor
# is a named semantic-seam funnel. Lazy, fail-open: an absent or erroring
# governor leaves every path exactly as it was.
try:
    from .resource_governor import (
        clear_active_ticket as _governor_clear_ticket,
    )
    from .resource_governor import (
        get_resource_governor as _get_resource_governor,
    )
    from .resource_governor import (
        set_active_ticket as _governor_set_ticket,
    )
    from .resource_governor import (
        ticket_scope as _governor_ticket_scope,
    )
    RESOURCE_GOVERNOR_AVAILABLE = True
except ImportError:
    RESOURCE_GOVERNOR_AVAILABLE = False
    _get_resource_governor = None
    _governor_set_ticket = None
    _governor_clear_ticket = None
    _governor_ticket_scope = None


def _governor_admit(
    model: str,
    requested_ctx: int | None,
    caller: str = "chat",
    extra_models: list[str] | None = None,
):
    """Funnel-side admission (S224). None when the governor is absent,
    disabled, or errors (fail-open); an AdmissionDecision otherwise."""
    if not RESOURCE_GOVERNOR_AVAILABLE or _get_resource_governor is None:
        return None
    try:
        governor = _get_resource_governor()
        if not governor.config.enabled:
            return None
        return governor.admit(
            model,
            requested_ctx=requested_ctx,
            caller=caller,
            extra_models=extra_models,
        )
    except Exception as e:
        logger.debug(f"Governor admission failed open: {e}")
        return None


def _governor_hold_ticket(decision) -> None:
    """Set the thread-local admission ticket (the 4.4 pass-through);
    no-op when the governor or the decision is absent."""
    if decision is None or _governor_set_ticket is None:
        return
    try:
        _governor_set_ticket(decision)
    except Exception as e:
        logger.debug(f"Governor ticket set failed open: {e}")


def _governor_release_ticket() -> None:
    """Clear the thread-local admission ticket; never raises."""
    if _governor_clear_ticket is None:
        return
    try:
        _governor_clear_ticket()
    except Exception as e:
        logger.debug(f"Governor ticket clear failed open: {e}")


def _native_think_kwargs(think: bool | None) -> dict:
    """Ollama-native think kwargs, tri-state (S259).

    None means "do not steer": nothing is sent and the client call is
    byte-identical to the historical one (a reasoning model keeps its
    own default). True engages the native switch; False is the explicit
    suppression a prompt-level tag cannot deliver -- the only way to
    actually stop a thinking-by-default model (qwen3.x class) from
    spending reasoning tokens. Callers merge the result into the client
    kwargs; the streaming call-site maps its historical boolean
    conservatively (truthy -> True, falsy -> None) because the live
    behaviour of think=False across model families is host-verified,
    never assumed in-container (INFERENCE_PERF_S259.md).
    """
    if think is None:
        return {}
    return {"think": bool(think)}


def _governor_account_load(model: str, num_ctx: int | None) -> None:
    """invalidate_on_load wiring for funnels whose transport is a direct
    ollama call out of the mechanical seam's reach (speculative, cascade,
    vision): the funnel accounts right after a positive admission so the
    Bloc 0 attribution learns real costs."""
    if not RESOURCE_GOVERNOR_AVAILABLE or _get_resource_governor is None:
        return
    try:
        _get_resource_governor().invalidate_on_load(model, num_ctx)
    except Exception as e:
        logger.debug(f"Governor load accounting failed open: {e}")


# Network manager (v1.7.3 -- S71)
try:
    from .network_manager import network_manager as _network_manager
    NETWORK_MANAGER_AVAILABLE = True
except ImportError:
    NETWORK_MANAGER_AVAILABLE = False
    _network_manager = None

# Sync queue (v1.7.3 -- S71)
try:
    from .sync_queue import sync_queue as _sync_queue
    SYNC_QUEUE_AVAILABLE = True
except ImportError:
    SYNC_QUEUE_AVAILABLE = False
    _sync_queue = None

# Performance monitor (v1.7.4 -- S72)
try:
    from .performance_monitor import performance_monitor as _performance_monitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    _performance_monitor = None

# Vision delegation pipeline (v1.9.7 -- S95)
try:
    from .vision_pipeline import vision_pipeline as _vision_pipeline
    VISION_PIPELINE_AVAILABLE = True
except ImportError:
    VISION_PIPELINE_AVAILABLE = False
    _vision_pipeline = None

# Context optimizer (v2.4.0 -- S123)
try:
    from .context_optimizer import get_optimizer as _get_context_optimizer
    CONTEXT_OPTIMIZER_AVAILABLE = True
except ImportError:
    CONTEXT_OPTIMIZER_AVAILABLE = False
    _get_context_optimizer = None

logger = logging.getLogger(__name__)

# =============================================================================
# SYSTEM PROMPTS WITH MULTILINGUAL SUPPORT
# =============================================================================
# Note: These prompts instruct the model to respond in the user's language

PROMPTS = {
    # ----- R CODE -----
    "code_r": {
        "standard": """You are a senior R expert specialized in bioinformatics and ecology.

## YOUR RULES
1. **Tidyverse style**: Use pipe |> or %>%, dplyr, tidyr
2. **Commented code**: Explain each important step
3. **Error handling**: Include tryCatch() or stopifnot() when relevant
4. **Reproducibility**: set.seed() for randomness

## RESPONSE FORMAT
```r
# [SHORT DESCRIPTION]
library(...)
# [CODE WITH COMMENTS]
```

## LANGUAGE RULE
Respond in the same language as the user's question. If they ask in French, respond in French. If they ask in English, respond in English.

Now answer the user's request.""",

        "reasoning": """You are a senior R expert. THINK OUT LOUD BEFORE coding.

## MANDATORY PROCESS
<thinking>
1. Rephrase the problem
2. List necessary steps
3. Identify potential pitfalls
4. Which packages to use?
</thinking>

## THEN CODE with tidyverse style.

LANGUAGE: Respond in the user's language (French if asked in French, English if asked in English).

User question:""",

        "fast": """R Expert. Tidyverse style. Respond in user's language. Direct and concise code.

Question:""",
    },

    # ----- PYTHON CODE -----
    "code_python": {
        "standard": """You are a senior Python developer specialized in data science.

## YOUR RULES
1. **Type hints**: Always type functions
2. **Docstrings**: Google format (Args, Returns)
3. **PEP 8**: Properly formatted code
4. **Error handling**: try/except with clear messages

## FORMAT
```python
#!/usr/bin/env python3
\"\"\"Script description\"\"\"

from typing import ...

def my_function(arg: type) -> type:
    \"\"\"Description.\"\"\"
    pass
```

## LANGUAGE RULE
Respond in the same language as the user's question.

Answer the request.""",

        "reasoning": """You are a senior Python dev. REASON BEFORE CODING.

<thinking>
1. What is the exact problem?
2. What are the inputs/outputs?
3. Which modules to use?
4. Edge cases to handle?
</thinking>

Then code with type hints and docstrings.
LANGUAGE: Match the user's language.

Question:""",

        "fast": """Python dev. Type hints. Respond in user's language. Concise code.

Question:""",
    },

    # ----- DEBUG -----
    "debug_r": {
        "standard": """You are an R debugging expert. Your approach is METHODICAL.

## DEBUG PROCESS
1. **READ** the error carefully
2. **IDENTIFY** the probable cause
3. **FIX** with working code
4. **EXPLAIN** to avoid in the future

## RESPONSE FORMAT
### Error Analysis
[Error explanation]

### Probable Cause
[What causes the problem]

### Fixed Code
```r
# Corrected code with comments
```

### Tip
[How to avoid this problem]

## LANGUAGE: Respond in the user's language.

Now analyze the user's error.""",

        "reasoning": """R debugging expert. Reason step by step.
Respond in the user's language.

<thinking>
1. What exactly does the error say?
2. Which line/function is affected?
3. What data type is problematic?
4. What's the solution?
</thinking>

Then provide analysis and fixed code.

Error:""",

        "fast": """R Debug. Identify error, give fixed code. User's language.

Error:""",
    },

    "debug_python": {
        "standard": """You are a Python debugging expert. Your approach is METHODICAL.

## DEBUG PROCESS
1. **READ** the traceback carefully
2. **IDENTIFY** the probable cause
3. **FIX** with working code
4. **EXPLAIN** to avoid in the future

## RESPONSE FORMAT
### Error Analysis
[Traceback explanation]

### Probable Cause
[What causes the problem]

### Fixed Code
```python
# Corrected code with comments
```

### Tip
[How to avoid this problem]

## LANGUAGE: Respond in the user's language.

Now analyze the error.""",

        "reasoning": """Python debugging expert. Reason step by step before fixing.
Respond in user's language.""",

        "fast": """Python Debug. Identify error, give fixed code. User's language.""",
    },

    # ----- SCIENTIFIC WRITING -----
    "scientific_writing": {
        "standard": """You are an expert scientific writer.

## YOUR RULES
1. **Academic style**: Objective, precise, no unnecessary jargon
2. **Clear structure**: Follow conventions for the document type
3. **Data**: Include statistics and exact values when relevant
4. **Citations**: (Author, Year) format if you invent any

## DOCUMENT TYPES
- Abstract: 250 words max, Background-Methods-Results-Conclusion
- Methods: Reproducibility, technical details, statistics
- Results: Objective, precise numbers, no interpretation
- Discussion: Interpretation, limitations, perspectives

## LANGUAGE: Respond in the user's language.

Write according to the request.""",

        "reasoning": """Scientific writer. Structure your thoughts before writing.
Respond in user's language.

<thinking>
1. What type of document?
2. What structure to adopt?
3. What key points to include?
4. What tone to use?
</thinking>

Then write the requested text.""",

        "fast": """Concise scientific writing. Academic style. User's language.""",
    },

    # ----- PLANNING -----
    "planning": {
        "standard": """You are an expert in organization and planning.

## YOUR METHOD
1. **Understand** the final objective
2. **Break down** into actionable steps
3. **Prioritize** by importance/urgency
4. **Anticipate** obstacles

## FORMAT
### Objective
[Clear rephrasing of the objective]

### Steps
1. [Step 1 - actionable]
2. [Step 2 - actionable]
...

### Points of Attention
- [Risk or pitfall to avoid]

### Next Action
[The first concrete thing to do]

## LANGUAGE: Respond in the user's language.

Plan the user's task.""",

        "reasoning": """Planning expert. Reason through the approach.
Respond in user's language.""",

        "fast": """Planner. Concise action steps. User's language.""",
    },

    # ----- GENERAL -----
    "general": {
        "standard": """You are a helpful assistant.

## YOUR APPROACH
1. Understand the question completely
2. Provide accurate, relevant information
3. Be concise but thorough
4. Use examples when helpful

## LANGUAGE: Respond in the user's language.

Answer the question.""",

        "reasoning": """Thoughtful assistant. Reason through your answer.
Respond in user's language.""",

        "fast": """Concise assistant. Direct answers. User's language.""",
    },
}

# Default prompt if task type not found
DEFAULT_PROMPT = PROMPTS["general"]["standard"]


# =============================================================================
# REFINEMENT PROMPT
# =============================================================================

REFINE_PROMPT = """You are a prompt engineering expert. Your task is to improve user questions.

## CONTEXT
{context}

## ORIGINAL QUESTION
{question}

## YOUR MISSION
Rewrite this question to be:
1. More specific and detailed
2. Clear about expected output format
3. Including relevant technical context
4. Well-structured if complex

## RULES
- Keep the same language as the original (French->French, English->English)
- Don't change the intent
- Don't add unnecessary complexity
- If the question is already good, make minimal changes

## OUTPUT
Return ONLY the improved question, nothing else."""


# =============================================================================
# EXECUTOR CLASS
# =============================================================================

class Executor:
    """
    Execute LLM queries with refinement and streaming.

    Handles:
    - System prompt selection
    - Question refinement
    - Streaming execution
    - Context validation (NEW: Phase A4)
    - Cancellation
    """

    def __init__(self):
        """Initialize the executor."""
        self._cancel_event = threading.Event()
        self._current_task: str | None = None
        self._last_refined_question: str | None = None
        self._last_context_check: ContextCheck | None = None
        self._last_window_stats: dict[str, Any] = {}
        self._memory_enabled: bool = True  # Session 11: memory injection on by default
        self._cache_enabled: bool = True  # Session 18: response caching on by default
        self._last_cache_hit: bool = False  # Whether the last call was a cache hit
        self._last_verification_results: list = []  # S43: verification results
        self._last_tool_calls: list = []  # S45: tool-call results
        self._prompt_optimization_enabled: bool = True  # S65: prompt template + budget
        self._last_prompt_budget = None  # S65: last calculated PromptTokenBudget
        self._compression_enabled: bool = True  # S66: conversation compressor
        self._last_compression_result = None  # S66: last CompressedContext or None
        self._s68_cache_hit: bool = False  # S68: last call was S68 cache hit
        self._s68_cache_key: str = ""  # S68: last S68 cache key used
        self._last_cascade_result = None  # S69: last CascadeResult or None
        self._last_speculative_result = None  # S70: last SpeculativeResult or None
        self._last_offline_queued: bool = False  # S71: last call was queued offline
        self._last_vision_meta: dict = {}  # S95: last vision delegation metadata
        self._last_optimization_report = None  # S123: last OptimizationReport or None

    @property
    def last_refined_question(self) -> str | None:
        """Get the last refined question."""
        return self._last_refined_question

    @property
    def last_context_check(self) -> Optional['ContextCheck']:
        """Get the last context check result."""
        return self._last_context_check

    @property
    def last_window_stats(self) -> dict[str, Any]:
        """Get the last sliding window stats from _build_conversation_messages.

        Returns dict with keys: strategy, kept, dropped, total_tokens,
        available_for_input, context_window, history_count, etc.
        Empty dict if no multi-turn call was made.
        """
        return self._last_window_stats

    @property
    def memory_enabled(self) -> bool:
        """Whether memory facts are injected into system prompts."""
        return self._memory_enabled

    @memory_enabled.setter
    def memory_enabled(self, value: bool) -> None:
        self._memory_enabled = bool(value)

    @property
    def cache_enabled(self) -> bool:
        """Whether response caching is active for this executor."""
        return self._cache_enabled and RESPONSE_CACHE_AVAILABLE

    @cache_enabled.setter
    def cache_enabled(self, value: bool) -> None:
        self._cache_enabled = bool(value)

    @property
    def last_cache_hit(self) -> bool:
        """Whether the last execute() call was served from cache."""
        return self._last_cache_hit

    @property
    def s68_cache_hit(self) -> bool:
        """S68: Whether the last call was served from the S68 semantic cache."""
        return self._s68_cache_hit

    @property
    def s68_cache_key(self) -> str:
        """S68: The cache key used for the last S68 cache lookup."""
        return self._s68_cache_key

    @property
    def last_verification_results(self) -> list:
        """S43: Verification results of the last execute().

        Returns:
            List of VerificationResult (one per verified code block).
            Empty if no verification or no code blocks.
        """
        return self._last_verification_results

    @property
    def last_tool_calls(self) -> list:
        """S45: Tool-call results of the last execute().

        Returns:
            Liste de ToolCallResult (via l'AgenticExecutor).
            Vide si pas d'appels d'outils.
        """
        return self._last_tool_calls

    @property
    def prompt_optimization_enabled(self) -> bool:
        """S65: Whether prompt optimization (templates + budget) is active."""
        return self._prompt_optimization_enabled and PROMPT_OPTIMIZATION_AVAILABLE

    @prompt_optimization_enabled.setter
    def prompt_optimization_enabled(self, value: bool) -> None:
        self._prompt_optimization_enabled = bool(value)

    @property
    def last_prompt_budget(self) -> object | None:
        """S65: Last calculated PromptTokenBudget, or None."""
        return self._last_prompt_budget

    @property
    def compression_enabled(self) -> bool:
        """S66: Whether conversation compression is active."""
        return (
            self._compression_enabled
            and CONVERSATION_COMPRESSOR_AVAILABLE
            and _conversation_compressor is not None
            and _conversation_compressor.enabled
        )

    @compression_enabled.setter
    def compression_enabled(self, value: bool) -> None:
        self._compression_enabled = bool(value)

    @property
    def last_compression_result(self) -> object | None:
        """S66: Last CompressedContext from _build_conversation_messages, or None."""
        return self._last_compression_result

    @property
    def last_optimization_report(self) -> object | None:
        """S123: Last OptimizationReport from context optimizer, or None."""
        return self._last_optimization_report

    @property
    def last_cascade_result(self) -> object | None:
        """S69: Last CascadeResult from cascading inference, or None."""
        return self._last_cascade_result

    @property
    def last_speculative_result(self) -> object | None:
        """S70: Last SpeculativeResult from speculative generation, or None."""
        return self._last_speculative_result

    @property
    def last_offline_queued(self) -> bool:
        """S71: Whether the last execute() call was queued due to offline state."""
        return self._last_offline_queued

    @property
    def last_vision_meta(self) -> dict:
        """S95: Vision delegation metadata from the last execute() call."""
        return self._last_vision_meta

    # -------------------------------------------------------------------------
    # System Prompts
    # -------------------------------------------------------------------------

    def get_system_prompt(self, task_type: str, variant: str = "standard") -> str:
        """
        Get the system prompt for a task type.

        Args:
            task_type: Task type (code_r, debug_python, etc.)
            variant: Prompt variant (standard, reasoning, fast)

        Returns:
            System prompt string
        """
        task_prompts = PROMPTS.get(task_type, PROMPTS.get("general", {}))
        return task_prompts.get(variant, task_prompts.get("standard", DEFAULT_PROMPT))

    # -------------------------------------------------------------------------
    # Refinement
    # -------------------------------------------------------------------------

    def refine_question(
        self,
        question: str,
        document: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> tuple[str, str | None]:
        """
        Refine a question using an LLM.

        Args:
            question: Original question
            document: Optional content (code, text) for context
            model: Model to use for refinement
            temperature: Temperature for refinement

        Returns:
            (refined_question, error) - error is None if success
        """
        model = model or config.get_model("code", "primary")

        # Build context
        context_parts = []
        if document:
            # Detect document type
            if any(p in document for p in ["library(", "<-", "function("]):
                context_parts.append(f"R code provided:\n```r\n{document[:2000]}\n```")
            elif any(p in document for p in ["import ", "def ", "class "]):
                context_parts.append(f"Python code provided:\n```python\n{document[:2000]}\n```")
            else:
                context_parts.append(f"Document provided:\n{document[:2000]}")

        context = "\n\n".join(context_parts) if context_parts else "No document provided."

        # Build refinement prompt
        refine_prompt = REFINE_PROMPT.format(context=context, question=question)

        try:
            # Retrieve keep_alive duration
            ka = "30m"
            if MODEL_WARMUP_AVAILABLE and _model_warmup:
                ka = _model_warmup.keep_alive

            messages = [
                {"role": "system", "content": "You are a prompt improvement expert."},
                {"role": "user", "content": refine_prompt}
            ]
            options = {"temperature": temperature}

            # S224: governor admission (R-01). Refinement is auxiliary --
            # a refusal degrades to the original question through the
            # established error contract of this helper.
            _admission = _governor_admit(model, None, caller="chat")
            if _admission is not None and not _admission.admitted:
                _msg = _admission.refusal_payload().get(
                    "message", "resource admission refused"
                )
                logger.warning(
                    f"Refinement admission refused for {model}: {_msg}"
                )
                return question, _msg
            # S225: per-decision keep_alive override (Section 5 step 1) --
            # the governor's soft-pressure value takes precedence over
            # the warmup default for THIS call only.
            if _admission is not None and _admission.keep_alive:
                ka = _admission.keep_alive

            # S105: Use backend abstraction when available
            if INFERENCE_BACKEND_AVAILABLE and get_backend_registry:
                backend = get_backend_registry().active
                if backend:
                    # S224: ticket pass-through (thread-local, 4.4).
                    _governor_hold_ticket(_admission)
                    try:
                        resp = backend.generate(
                            model=model,
                            messages=messages,
                            options=options,
                            keep_alive=ka,
                        )
                    finally:
                        _governor_release_ticket()
                    refined = resp.content.strip()
                    logger.debug(f"Refined question: {refined[:100]}...")
                    return refined, None

            # Fallback: direct ollama call
            response = ollama.chat(
                model=model,
                messages=messages,
                options=options,
                keep_alive=ka,
            )

            refined = response["message"]["content"].strip()
            logger.debug(f"Refined question: {refined[:100]}...")
            return refined, None

        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return question, str(e)

    # -------------------------------------------------------------------------
    # Context Validation (NEW: Phase A4)
    # -------------------------------------------------------------------------

    def validate_context(
        self,
        question: str,
        document: str,
        system_prompt: str,
        model: str,
        auto_truncate: bool = False
    ) -> tuple[str, Optional['ContextCheck'], str | None]:
        """
        Validate and optionally adjust context for model limits.

        Args:
            question: User's question
            document: Document/code content
            system_prompt: System prompt being used
            model: Target model
            auto_truncate: If True, automatically truncate if needed

        Returns:
            Tuple of (adjusted_document, context_check, warning_message)
        """
        if not CONTEXT_MANAGER_AVAILABLE:
            return document, None, None

        # Perform context check
        context_check = cm_check_context(
            prompt=question,
            document=document,
            system_prompt=system_prompt,
            model=model
        )

        self._last_context_check = context_check
        warning = context_check.warning_message

        # Handle truncation if needed
        if context_check.truncation_needed and auto_truncate:
            manager = get_context_manager()
            truncated_doc, tokens_removed = manager.smart_truncate(
                text=document,
                max_tokens=context_check.available_for_input - context_check.prompt_tokens - context_check.system_tokens - 1000,
                model=model
            )

            warning = f"Document truncated: removed ~{tokens_removed:,} tokens to fit context window"
            logger.info(f"Auto-truncated document: {tokens_removed} tokens removed")

            return truncated_doc, context_check, warning

        return document, context_check, warning

    # -------------------------------------------------------------------------
    # Conversation History (NEW: v1.3.0 - Multi-turn)
    # -------------------------------------------------------------------------

    # Context window management thresholds for history
    CONTEXT_SOFT_LIMIT = 0.70   # 70%: start of sliding window
    CONTEXT_HARD_LIMIT = 0.90   # 90%: warning + forced truncation

    def _estimate_tokens(self, text: str, model: str = "") -> int:
        """Estimate token count for text, with fallback.

        Args:
            text: Text to estimate
            model: Model name for more accurate estimation

        Returns:
            Estimated token count
        """
        if CONTEXT_MANAGER_AVAILABLE:
            return cm_estimate_tokens(text, model or None)
        # Fallback : approximation simple
        return len(text) // 4

    def _summarize_old_messages(
        self,
        history: list[dict[str, str]],
        total_tokens: int,
        soft_limit: int,
        model: str,
        conversation_id: str | None = None,
    ) -> bool:
        """Attempt to summarize old messages to reduce context.

        Compresses the oldest messages in history into a summary,
        modifies history in place (replaces old messages with summary).

        Handles cumulative summaries: if the first message is already
        a summary, incorporates it into the new summary.

        Args:
            history: Conversation history (MODIFIED in place)
            total_tokens: Current total token count
            soft_limit: Target token limit
            model: Current model name (for estimation)
            conversation_id: Optional conv ID for metadata storage

        Returns:
            True if summarization succeeded, False for fallback to drop
        """
        if not CONTEXT_SUMMARY_AVAILABLE or context_summarizer is None:
            return False

        try:
            # Determine the number of pairs to summarize
            # We want to reduce enough to get under soft_limit
            tokens_to_free = total_tokens - soft_limit

            # Detect an existing summary at position 0
            existing_summary = None
            start_idx = 0
            if history and is_summary_message(history[0]):
                existing_summary = extract_summary_text(history[0])
                start_idx = 1  # Don't re-summarize the summary message

            # Compute how many messages to summarize
            pairs_to_summarize = 0
            tokens_freed = 0
            idx = start_idx
            while tokens_freed < tokens_to_free and idx + 1 < len(history):
                t1 = self._estimate_tokens(history[idx]["content"], model)
                t2 = self._estimate_tokens(history[idx + 1]["content"], model)
                tokens_freed += t1 + t2
                pairs_to_summarize += 1
                idx += 2

            if pairs_to_summarize == 0:
                return False

            # Messages to summarize (excluding existing summary)
            end_idx = start_idx + pairs_to_summarize * 2
            messages_to_summarize = history[start_idx:end_idx]

            input_tokens = sum(
                self._estimate_tokens(m["content"], model)
                for m in messages_to_summarize
            )

            # Call the summarizer
            summary = context_summarizer.summarize_messages(
                messages=messages_to_summarize,
                existing_summary=existing_summary,
            )

            if summary is None:
                logger.warning("Summary failed -- falling back to deletion")
                return False

            summary_msg = context_summarizer.create_summary_message(summary)
            summary_tokens = self._estimate_tokens(summary, model)

            # Rebuild history: [summary] + remaining messages
            remaining = history[end_idx:]
            history.clear()
            history.append(summary_msg)
            history.extend(remaining)

            # Log
            if existing_summary:
                existing_tokens = self._estimate_tokens(existing_summary, model)
                logger.info(
                    f"Cumulative summary: merged with existing summary "
                    f"({existing_tokens}t) + {len(messages_to_summarize)} msgs "
                    f"({input_tokens}t) -> {summary_tokens}t"
                )
            else:
                logger.info(
                    f"Context summary: compressed {len(messages_to_summarize)} "
                    f"messages ({input_tokens}t) -> {summary_tokens}t"
                )

            # Store the summary in the conversation metadata
            if (
                conversation_id
                and CONVERSATION_AVAILABLE
                and conversation_manager is not None
            ):
                try:
                    conversation_manager.update_conversation_metadata(
                        conversation_id,
                        metadata={
                            "context_summary": summary,
                            "summary_msg_count": (
                                len(messages_to_summarize)
                                + (1 if existing_summary else 0)
                            ),
                            "summary_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        },
                    )
                except Exception as e:
                    logger.warning(f"Unable to save summary: {e}")

            return True

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return False

    def _inject_memory(self, system_prompt: str, question: str | None = None) -> str:
        """Append the memory block to the system prompt if available and enabled.

        Builds the unified working block (M1): the salient durable facts (always
        present, ranked by use_count x recency) plus query-relevant facts plus a
        bridge over the legacy store, composed in one token budget and
        deduplicated. The full uncompressed archive remains searchable for
        recovery; the block is unwrapped (the agent wraps it as untrusted
        context, S175).

        Args:
            system_prompt: the current system prompt
            question: the current question, used to rank the working block

        Returns:
            system prompt with memory section appended, or unchanged
        """
        if not self._memory_enabled:
            return system_prompt

        # M1: unified working block -- the salient durable facts (always present)
        # + query-relevant facts + a legacy bridge, composed in one token budget.
        # The bridge reads the legacy memories.db so an existing store keeps
        # surfacing during the migration; it is dropped once writes are unified.
        legacy_facts = None
        if MEMORY_AVAILABLE and _memory_manager is not None:
            try:
                legacy_facts = _memory_manager.get_all_facts(active_only=True)
            except Exception as e:
                logger.debug(f"Legacy memory read skipped: {e}")
                legacy_facts = None

        memory_block = ""
        if DUAL_LAYER_MEMORY_AVAILABLE and _build_memory_block is not None:
            try:
                memory_block = _build_memory_block(
                    question,
                    max_tokens=500,
                    legacy_facts=legacy_facts,
                    mark_used=True,
                ) or ""
            except Exception as e:
                logger.debug(f"Unified memory block skipped: {e}")
                memory_block = ""

        # Belt-and-suspenders: if the unified composer is entirely unavailable
        # but the legacy store is, fall back to its flat block (prior behaviour).
        if not memory_block and MEMORY_AVAILABLE and _memory_manager is not None:
            try:
                memory_block = _memory_manager.format_for_prompt(max_tokens=500) or ""
            except Exception as e:
                logger.debug(f"Memory injection skipped: {e}")
                memory_block = ""

        if memory_block:
            return system_prompt + "\n\n" + memory_block
        return system_prompt

    def _inject_project_context(
        self,
        system_prompt: str,
        question: str,
        conversation_id: str | None,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Inject project context into the system prompt if applicable.

        Checks if the conversation is linked to a project, runs trigger
        detection, and if relevant, retrieves RAG context and appends it.
        Always injects system_instructions for project conversations,
        even if RAG retrieval is skipped.

        Args:
            system_prompt: The current system prompt.
            question: The user's question (for trigger detection + RAG query).
            conversation_id: The conversation ID (to find linked project).
            on_status: Optional status callback.

        Returns:
            System prompt with project context appended, or unchanged.
        """
        if not PROJECT_CONTEXT_AVAILABLE:
            return system_prompt

        if not conversation_id or _project_store is None:
            return system_prompt

        # Find linked project
        try:
            project_id = _project_store.get_project_for_conversation(conversation_id)
        except Exception:
            return system_prompt

        if not project_id:
            return system_prompt

        def _status(msg: str):
            if on_status:
                on_status(msg)

        # Run trigger detection. L3 (LLM classification) is intentionally
        # enabled on this path; the latency trade-off is tracked as PTR-02
        # (S192) for the live shakedown.
        use_rag = False
        if _trigger_detector is not None:
            try:
                relevance = _trigger_detector.detect(question, project_id, skip_l3=False)
                use_rag = relevance.relevant
                if use_rag:
                    _status(
                        f"[>] Project context: L{relevance.trigger_level} trigger "
                        f"(confidence={relevance.confidence:.2f})"
                    )
            except Exception as e:
                logger.debug("Project trigger detection failed: %s", e)

        # Build context
        if _project_context_builder is None:
            return system_prompt

        try:
            if use_rag and _project_context_builder.available:
                # Full RAG context retrieval
                ctx = _project_context_builder.build_context(project_id, question)
            else:
                # Fallback: system_instructions only (always for project convos)
                ctx = _project_context_builder.build_system_instructions_only(project_id)

            if ctx.context_text:
                _status(
                    f"[OK] Project context injected: "
                    f"{ctx.chunks_used} chunks, ~{ctx.total_tokens_estimate} tokens"
                )
                return system_prompt + "\n\n" + ctx.context_text

        except Exception as e:
            logger.warning("Project context injection failed: %s", e)

        return system_prompt

    def _build_conversation_messages(
        self,
        system_prompt: str,
        conversation_id: str,
        current_message: str,
        model: str,
    ) -> tuple[list[dict[str, str]], int, dict[str, Any]]:
        """Build the full messages array with conversation history.

        Loads conversation history from the conversation backend,
        applies intelligent sliding window if context limits are approached,
        and returns an Ollama-ready messages list with window stats.

        Trimming strategy (S17):
        1. Intelligent summary (context_summary) if the threshold is reached
        2. Importance-based sliding window (context_window S16) as fallback
        3. Aggressive emergency dropping if everything still exceeds

        Args:
            system_prompt: System prompt for this task
            conversation_id: UUID of the conversation
            current_message: Current user message to append
            model: Model name (for token estimation and limits)

        Returns:
            Tuple of (messages_list, total_token_estimate, window_stats)
            window_stats contient strategy, kept, dropped, etc.
        """
        # Sliding-window stats (surfaced to the UI)
        window_stats: dict[str, Any] = {}

        # Retrieve the model limits
        if CONTEXT_MANAGER_AVAILABLE:
            limits = cm_get_model_limits(model)
            context_window = limits.context_window
            output_reserve = limits.max_output
        else:
            # Fallback conservateur
            context_window = 32768
            output_reserve = 4096

        available_for_input = context_window - output_reserve

        # Estimate the tokens of the fixed parts (system + current message)
        system_tokens = self._estimate_tokens(system_prompt, model)
        current_tokens = self._estimate_tokens(current_message, model)
        fixed_tokens = system_tokens + current_tokens

        # Retrieve conversation history
        history = []
        history_tokens = 0
        if CONVERSATION_AVAILABLE and conversation_manager:
            history = conversation_manager.get_context_messages(conversation_id)

            # Inject a saved summary if available and if history
            # does not already contain a summary message (avoids duplicates)
            if (
                CONTEXT_SUMMARY_AVAILABLE
                and history
                and not (history and is_summary_message(history[0]))
            ):
                try:
                    conv = conversation_manager.get_conversation(conversation_id)
                    if conv and conv.metadata.get("context_summary"):
                        stored_summary = conv.metadata["context_summary"]
                        summary_msg = context_summarizer.create_summary_message(
                            stored_summary
                        )
                        history.insert(0, summary_msg)
                        logger.info(
                            "Context summary restored from metadata"
                        )
                except Exception as e:
                    logger.debug(f"No saved summary to restore: {e}")

            # Estimate the history tokens
            for msg in history:
                history_tokens += self._estimate_tokens(msg["content"], model)

        total_tokens = fixed_tokens + history_tokens

        # S66: Conversation compression -- triggered before sliding window if
        # history exceeds the S65 budget's history_tokens allocation.
        # The full archive in SQLite is never modified; compression only
        # affects what goes into the prompt.
        self._last_compression_result = None
        if (
            self.compression_enabled
            and history
            and self._last_prompt_budget is not None
        ):
            budget_history_tokens = self._last_prompt_budget.history_tokens
            if history_tokens > budget_history_tokens:
                try:
                    compressed = _conversation_compressor.compress(
                        messages=history,
                        budget_tokens=budget_history_tokens,
                        model=model,
                    )
                    if compressed.compressed_count > 0 and compressed.summary:
                        # Rebuild history: summary block + verbatim recent messages
                        summary_block = {
                            "role": "system",
                            "content": compressed.summary,
                        }
                        history = [summary_block] + list(compressed.recent_messages)
                        history_tokens = sum(
                            self._estimate_tokens(m["content"], model)
                            for m in history
                        )
                        total_tokens = fixed_tokens + history_tokens
                        self._last_compression_result = compressed
                        logger.info(
                            f"S66 compression ({compressed.strategy_used}): "
                            f"{compressed.original_count} -> "
                            f"{compressed.compressed_count} compressed + "
                            f"{len(compressed.recent_messages)} kept verbatim, "
                            f"{compressed.tokens_saved}t saved"
                        )
                except Exception as e:
                    logger.warning(f"S66 compression failed, proceeding without: {e}")

        # Sliding window: if we exceed the soft threshold, summarize or prune
        soft_limit = int(available_for_input * self.CONTEXT_SOFT_LIMIT)
        hard_limit = int(available_for_input * self.CONTEXT_HARD_LIMIT)

        if total_tokens > soft_limit and len(history) > 2:
            logger.info(
                f"Contexte a {total_tokens}/{available_for_input} tokens "
                f"({total_tokens * 100 / available_for_input:.0f}%), "
                f"application du sliding window"
            )

            # --- Phase 1: Intelligent summary (F2, v1.4.0) ---
            summarized = False
            if (
                CONTEXT_SUMMARY_AVAILABLE
                and context_summarizer is not None
                and len(history) >= context_summarizer.SUMMARY_THRESHOLD
            ):
                summarized = self._summarize_old_messages(
                    history, total_tokens, soft_limit, model, conversation_id
                )
                if summarized:
                    # Recompute tokens after summary (history modified in-place)
                    history_tokens = sum(
                        self._estimate_tokens(m["content"], model) for m in history
                    )
                    total_tokens = fixed_tokens + history_tokens

            # --- Phase 2: Intelligent sliding window (S16/S17) ---
            # Use SlidingWindowManager if the summary was not enough
            # or was not available
            if (
                not summarized
                and CONTEXT_WINDOW_AVAILABLE
                and sliding_window_manager is not None
                and total_tokens > soft_limit
            ):
                try:
                    trimmed_history, sw_stats = sliding_window_manager.prepare_messages(
                        history, model, system_tokens=fixed_tokens
                    )
                    window_stats = sw_stats
                    dropped = sw_stats.get("dropped", 0)
                    if dropped > 0:
                        logger.info(
                            f"Sliding window (importance): {sw_stats.get('kept', 0)} gardes, "
                            f"{dropped} supprimes ({sw_stats.get('strategy', '?')})"
                        )
                    history = trimmed_history
                    history_tokens = sw_stats.get("total_tokens", 0)
                    total_tokens = fixed_tokens + history_tokens
                except Exception as e:
                    logger.error(f"Intelligent sliding window error: {e}")

            # --- Phase 3: Simple-dropping fallback (v1.3.0 behaviour) ---
            # If neither summary nor intelligent window worked
            if total_tokens > soft_limit and len(history) > 2:
                trimmed_history = list(history)
                while total_tokens > soft_limit and len(trimmed_history) > 2:
                    if len(trimmed_history) >= 2:
                        removed_1 = trimmed_history.pop(0)
                        removed_2 = trimmed_history.pop(0)
                        removed_tokens = (
                            self._estimate_tokens(removed_1["content"], model)
                            + self._estimate_tokens(removed_2["content"], model)
                        )
                        total_tokens -= removed_tokens
                        history_tokens -= removed_tokens
                    else:
                        break

                pairs_removed = (len(history) - len(trimmed_history)) // 2
                logger.info(
                    f"Sliding window (drop legacy) : {pairs_removed} paire(s) supprimee(s), "
                    f"contexte reduit a {total_tokens} tokens"
                )
                history = trimmed_history
                window_stats["fallback_legacy"] = True

        # Hard limit: if we still exceed despite the trim, force truncation
        if total_tokens > hard_limit:
            logger.warning(
                f"Context still too large after trim: "
                f"{total_tokens}/{available_for_input} tokens "
                f"({total_tokens * 100 / available_for_input:.0f}%). "
                f"Additional truncation of old messages."
            )
            trimmed_history = list(history)
            while total_tokens > hard_limit and len(trimmed_history) > 2:
                removed = trimmed_history.pop(0)
                removed_tokens = self._estimate_tokens(removed["content"], model)
                total_tokens -= removed_tokens
                history_tokens -= removed_tokens
            history = trimmed_history

        # Build the final messages array
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": current_message})

        logger.info(
            f"Messages built: system({system_tokens}t) + "
            f"history({history_tokens}t, {len(history)} msgs) + "
            f"user({current_tokens}t) = {total_tokens}t / "
            f"{available_for_input}t available "
            f"({total_tokens * 100 / available_for_input:.0f}%)"
        )

        # Enrich the stats with the global info
        window_stats.setdefault("strategy", "keep_all")
        window_stats["system_tokens"] = system_tokens
        window_stats["history_tokens"] = history_tokens
        window_stats["total_tokens"] = total_tokens
        window_stats["available_for_input"] = available_for_input
        window_stats["context_window"] = context_window
        window_stats["history_count"] = len(history)

        return messages, total_tokens, window_stats

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def execute(
        self,
        question: str,
        routing: RoutingResult,
        document: str | None = None,
        refine: bool = True,
        on_status: Callable[[str], None] | None = None,
        auto_truncate: bool = False,
        validate_context: bool = True,
        conversation_id: str | None = None,
        system_prompt_suffix: str | None = None,
        think: bool = False,
        web_search: bool = False,
        images: list[str] | None = None,
        no_cache: bool = False,
        persist: bool = True,
    ) -> Generator[str, None, tuple[str, str]]:
        """
        Execute a complete query with streaming.

        Args:
            question: User's question
            routing: Routing result (model, temperature, etc.)
            document: Optional document/code for context
            refine: If True, refine question before execution
            on_status: Callback for status updates
            auto_truncate: If True, auto-truncate document if context exceeded
            validate_context: If True, validate context against model limits
            conversation_id: Optional conversation UUID for multi-turn mode.
                When provided, loads conversation history and saves messages
                after execution. When None, behaves as single-turn (backward compatible).
            system_prompt_suffix: Optional text appended to the system prompt.
                Used by web search integration to inject search instructions.
            think: If True, enable chain-of-thought reasoning via Ollama think=True.
                Thinking tokens are yielded as ("thinking", content) tuples (S42).
            web_search: If True, run web search before LLM call and inject
                results into the system prompt (S42).
            images: Optional list of base64-encoded images for vision models (S48).
                Passed directly to ollama.chat() via the images parameter.
            no_cache: S68: If True, bypass all cache layers for this call.
            persist: If True (default), save the user and assistant messages
                to the conversation after execution. Callers that own the
                final persistence themselves (e.g. the think+tools pipeline,
                which appends a tool-output block after this call) pass False
                to avoid a duplicated user message and a truncated assistant
                message.

        Yields:
            Response chunks in streaming. When think=True, thinking chunks
            are yielded as ("thinking", content) tuples before normal chunks.

        Returns:
            Tuple (refined_question, full_response) at the end

        Note:
            The return value is also stored in self.last_refined_question
            for easy retrieval after iteration completes.
        """
        self._cancel_event.clear()
        self._current_task = routing.task_type
        self._last_context_check = None
        self._last_compression_result = None  # S66: reset per-call
        self._last_offline_queued = False  # S71: reset per-call
        self._last_vision_meta: dict = {}  # S95: reset per-call

        def status(msg: str) -> None:
            if on_status:
                on_status(msg)
            logger.info(msg)

        # Step 0: Context validation (NEW: Phase A4)
        adjusted_document = document or ""
        if validate_context and document and CONTEXT_MANAGER_AVAILABLE:
            system_prompt = self.get_system_prompt(routing.task_type, routing.prompt_variant)

            adjusted_document, context_check, context_warning = self.validate_context(
                question=question,
                document=document,
                system_prompt=system_prompt,
                model=routing.model,
                auto_truncate=auto_truncate
            )

            if context_check:
                if context_check.exceeds_limit and not auto_truncate:
                    yield f"[ERR] Context exceeds model limit: {context_warning}"
                    yield f"\n\nEstimated tokens: ~{context_check.total_tokens:,}"
                    yield f"\nAvailable: {context_check.available_for_input:,}"
                    yield "\n\nOptions:"
                    yield "\n- Enable auto-truncation"
                    yield "\n- Summarize the document first"
                    yield "\n- Use a model with larger context (e.g., nemotron-3-nano:30b)"
                    return question, f"[Context exceeded: {context_check.total_tokens} > {context_check.available_for_input}]"

                if context_warning:
                    status(f"[!] {context_warning}")

        # Step 0b: Vision delegation (S95)
        # When images are present and the current model lacks vision,
        # delegate to the user's preferred vision model for a description
        # and inject it into the question. Images are consumed by the
        # vision model so the text model only sees the augmented message.
        _vision_meta: dict = {}
        if VISION_PIPELINE_AVAILABLE and _vision_pipeline is not None and images:
            # S224: governor admission of the vision-delegation model
            # BEFORE it loads (spec 4.1). Only when delegation would
            # actually trigger (the pipeline's own public check); the
            # model is resolved defensively (vision_pipeline is not
            # edited). A refusal refuses the REQUEST with the typed body
            # naming the vision model -- D3 chat semantics, structured,
            # never a silent strip. The admission itself never breaks the
            # vision path: any error in it fails open.
            _vision_refusal_msg = None
            try:
                if _vision_pipeline.detect_needs_delegation(
                    message=question,
                    images=images,
                    current_model=routing.model,
                ):
                    _vresolver = getattr(
                        _vision_pipeline, "_resolve_vision_model", None
                    )
                    _vision_model = (
                        _vresolver() if callable(_vresolver) else None
                    )
                    if _vision_model:
                        _vadmission = _governor_admit(
                            str(_vision_model), None, caller="chat"
                        )
                        if _vadmission is not None and not _vadmission.admitted:
                            _vision_refusal_msg = (
                                _vadmission.refusal_payload().get(
                                    "message", "resource admission refused"
                                )
                            )
                        elif (
                            _vadmission is not None
                            and _vadmission.load_expected
                        ):
                            _governor_account_load(str(_vision_model), None)
            except Exception as exc:
                logger.debug("S224: vision admission failed open: %s", exc)
            if _vision_refusal_msg is not None:
                status("[!] Resource admission refused for the vision model")
                _vmsg = f"[ERR] {_vision_refusal_msg}"
                yield _vmsg
                self._current_task = None
                return question, _vmsg
            try:
                question, images, _vision_meta = _vision_pipeline.process(
                    message=question,
                    images=images,
                    current_model=routing.model,
                    on_status=on_status,
                )
                self._last_vision_meta = _vision_meta
            except Exception as exc:
                logger.warning("Vision delegation failed: %s", exc)
                # Continue with original question and images on failure

        # BUG-09 S108: Safety net -- if images still present but no vision
        # pipeline available, strip them to prevent 500 from non-vision model.
        if images and (not VISION_PIPELINE_AVAILABLE or _vision_pipeline is None):
            logger.warning(
                "Images provided but vision pipeline unavailable. "
                "Stripping images to avoid model error."
            )
            status(
                "No vision-capable model found. "
                "Install llava, llama3.2-vision, or similar to analyze images."
            )
            images = None

        # Step 1: Refinement (optional)
        refined_question = question
        if refine:
            status(f"[>] Refining question with {routing.model}...")
            refined_question, error = self.refine_question(
                question, adjusted_document, routing.model, config.get_temperature("refining")
            )
            if error:
                status(f"[!] Refinement failed: {error}")
            else:
                status("[OK] Question refined")

        # Store refined question in instance for later retrieval
        self._last_refined_question = refined_question

        if self._cancel_event.is_set():
            yield "[Cancelled]"
            return refined_question, "[Cancelled]"

        # Step 2: Get system prompt
        # S65: Use prompt template engine when available and enabled
        _template_temp_override = None
        self._last_prompt_budget = None
        _active_project_id = None  # S123: always initialize for optimizer access

        if (
            self.prompt_optimization_enabled
            and _prompt_template_engine is not None
            and _prompt_budget_manager is not None
        ):
            # Determine project context for template resolution
            if conversation_id and PROJECT_CONTEXT_AVAILABLE and _project_store:
                try:
                    _active_project_id = _project_store.get_project_for_conversation(
                        conversation_id
                    )
                except Exception:
                    pass

            # Get template for detected task type
            template = _prompt_template_engine.get_template(
                routing.task_type, project_id=_active_project_id,
            )

            # Interpolate variables
            system_prompt = _prompt_template_engine.interpolate(
                template,
                context={
                    "model_name": routing.model,
                    "task_type": routing.task_type,
                    "project_name": _active_project_id or "",
                },
            )

            # Capture temperature override from template
            if template.temperature_override is not None:
                _template_temp_override = template.temperature_override

            # Calculate token budget
            self._last_prompt_budget = _prompt_budget_manager.calculate_budget(
                model=routing.model,
                project_active=_active_project_id is not None,
            )

            logger.debug(
                f"S65: template='{template.task_type}' source={template.source}, "
                f"budget={self._last_prompt_budget.total_window}t "
                f"(sys={self._last_prompt_budget.system_tokens}/"
                f"proj={self._last_prompt_budget.project_tokens}/"
                f"hist={self._last_prompt_budget.history_tokens}/"
                f"user={self._last_prompt_budget.user_tokens}/"
                f"res={self._last_prompt_budget.reserve_tokens})"
            )
        else:
            # Fallback: use original hardcoded prompt system
            system_prompt = self.get_system_prompt(routing.task_type, routing.prompt_variant)

        # Step 2b: Append suffix if provided (web search instructions, etc.)
        if system_prompt_suffix:
            system_prompt = system_prompt + system_prompt_suffix

        # Step 2c: Inject memory facts (Session 11 -- F1; S174 dual-layer)
        system_prompt = self._inject_memory(system_prompt, refined_question)

        # S123: Check if context optimizer handles project injection
        _s123_optimizer_active = (
            CONTEXT_OPTIMIZER_AVAILABLE
            and _get_context_optimizer is not None
            and _get_context_optimizer() is not None
            and _get_context_optimizer().enabled
        )

        # Step 2c-bis: Inject project context (S58)
        # Skipped when S123 optimizer is active (it handles RAG with budget passthrough)
        if not _s123_optimizer_active:
            system_prompt = self._inject_project_context(
                system_prompt, question, conversation_id, on_status=status,
            )

        # Step 2d: Web search injection (S42)
        # If web_search is enabled, run a search and inject results.
        # SR-02 (S185): the availability gate previously imported
        # SearchInterceptor/wrap_system_prompt but never used them (this path
        # injects results directly via web_search_engine). Gate on the real
        # dependency instead. The <search>-tag SearchInterceptor state machine
        # is not wired into this streaming path (see search_integration.py).
        if web_search:
            # SR-03: gate the live chat egress on the kill switch too. The
            # security middleware only blocks the standalone /api/search
            # endpoints, and only in Bulbe; consult the kill switch here so an
            # engaged kill switch actually stops chat-triggered web search.
            try:
                from opti_oignon.search_killswitch import search_killswitch as _ks
                _search_killed = _ks.is_killed()
            except Exception:
                _search_killed = False

            try:
                from opti_oignon.web_search import web_search_engine
                SEARCH_AVAILABLE = True
            except ImportError:
                web_search_engine = None
                SEARCH_AVAILABLE = False

            if _search_killed:
                status("[!] Web search skipped (kill switch engaged)")
            elif SEARCH_AVAILABLE:
                status(f"[>] Web search for: {question[:80]}...")
                try:
                    results = web_search_engine.search(question, max_results=5)
                    if results:
                        # Format results as additional context
                        search_context = "\n\n--- Web Search Results ---\n"
                        for i, r in enumerate(results, 1):
                            title = getattr(r, 'title', r.get('title', '')) if isinstance(r, dict) else getattr(r, 'title', str(r))
                            snippet = getattr(r, 'snippet', r.get('snippet', '')) if isinstance(r, dict) else getattr(r, 'snippet', str(r))
                            url = getattr(r, 'url', r.get('url', '')) if isinstance(r, dict) else getattr(r, 'url', '')
                            search_context += f"\n[{i}] {title}\n{snippet}\nSource: {url}\n"
                        search_context += "\n--- End of Search Results ---\n"
                        search_context += "\nUse the search results above to inform your response. Cite sources when relevant."
                        system_prompt = system_prompt + search_context
                        status(f"[OK] {len(results)} search results injected")
                    else:
                        status("[!] Web search returned no results")
                except Exception as e:
                    status(f"[!] Web search failed: {e}")
                    logger.warning(f"Web search error: {e}")

        # Step 3: Build messages (multi-turn ou single-turn)
        # Final user content (refined question + possible document)
        user_content = refined_question
        if adjusted_document:
            user_content += f"\n\n---\nDocument provided:\n{adjusted_document}"

        # Multi-turn mode: load the conversation history
        use_conversation = (
            conversation_id is not None
            and CONVERSATION_AVAILABLE
            and conversation_manager is not None
        )

        # S66: Archive retrieval trigger -- if the user references past context
        # ("you said...", "we discussed..."), inject relevant archive snippets
        # into the system prompt so the LLM can answer accurately even after
        # compression has reduced the working history.
        if (
            use_conversation
            and self.compression_enabled
            and conversation_id
            and _check_retrieval_trigger is not None
            and _check_retrieval_trigger(
                refined_question,
                min_confidence=_conversation_compressor.get_config().get(
                    "retrieval_trigger_min_confidence", 0.6
                ) if _conversation_compressor else 0.6,
            )
        ):
            try:
                archive_results = _conversation_compressor.retrieve_from_archive(
                    conversation_id, refined_question
                )
                if archive_results:
                    archive_context = "\n\n--- Retrieved from conversation archive ---\n"
                    for res in archive_results:
                        archive_context += f"[{res.role}] {res.snippet}\n"
                    archive_context += "--- End of archive retrieval ---\n"
                    system_prompt = system_prompt + archive_context
                    status(
                        f"[>] Archive retrieval: {len(archive_results)} relevant "
                        f"message(s) injected from history"
                    )
                    logger.info(
                        f"S66 archive retrieval: {len(archive_results)} result(s) "
                        f"injected for conversation {conversation_id}"
                    )
            except Exception as e:
                logger.warning(f"S66 archive retrieval failed: {e}")

        if use_conversation:
            # S123: Use context optimizer when active (replaces manual pipeline)
            if _s123_optimizer_active:
                self._last_optimization_report = None
                optimizer = _get_context_optimizer()

                # Load conversation history for optimizer
                _conv_history = []
                if CONVERSATION_AVAILABLE and conversation_manager:
                    _conv_history = conversation_manager.get_context_messages(
                        conversation_id
                    )

                try:
                    opt_result = optimizer.optimize(
                        model=routing.model,
                        system_prompt=system_prompt,
                        user_message=user_content,
                        conversation_history=_conv_history,
                        conversation_id=conversation_id,
                        project_id=_active_project_id,
                        rag_query=refined_question,
                        project_active=_active_project_id is not None,
                    )
                    messages = opt_result.messages
                    context_tokens = opt_result.total_tokens
                    self._last_optimization_report = opt_result.report
                    system_prompt = opt_result.system_prompt

                    # Build window_stats compatible with existing UI
                    rpt = opt_result.report
                    window_stats = {
                        "strategy": "s123_optimizer",
                        "kept": len(messages) - 2,  # minus system + user
                        "dropped": rpt.total_trimmed,
                        "total_tokens": context_tokens,
                        "overflow": rpt.overflow,
                        "preset": rpt.preset_used,
                        "duration_ms": rpt.duration_ms,
                    }
                    self._last_window_stats = window_stats

                    status(
                        f"[>] S123 Optimizer: {len(messages)-2} messages, "
                        f"~{context_tokens:,} tokens "
                        f"(preset={rpt.preset_used}, "
                        f"trimmed={rpt.total_trimmed}t, "
                        f"{rpt.duration_ms:.0f}ms)"
                    )
                except Exception as e:
                    logger.warning(
                        "S123 optimizer failed, falling back to manual pipeline: %s", e
                    )
                    # Fallback: use manual pipeline
                    messages, context_tokens, window_stats = self._build_conversation_messages(
                        system_prompt=system_prompt,
                        conversation_id=conversation_id,
                        current_message=user_content,
                        model=routing.model,
                    )
                    self._last_window_stats = window_stats
            else:
                messages, context_tokens, window_stats = self._build_conversation_messages(
                    system_prompt=system_prompt,
                    conversation_id=conversation_id,
                    current_message=user_content,
                    model=routing.model,
                )
                # Store the stats for external access (context bar UI)
                self._last_window_stats = window_stats

            # Multi-turn status with trimming info (A3)
            if not _s123_optimizer_active:
                strategy = window_stats.get("strategy", "keep_all")
                dropped = window_stats.get("dropped", 0)
                if dropped > 0:
                    status(
                        f"[>] Multi-turn: {len(messages)-2} messages kept, "
                        f"{dropped} trimmed ({strategy}), ~{context_tokens:,} tokens"
                    )
                else:
                    status(
                        f"[>] Multi-turn: {len(messages)-2} previous messages, "
                        f"~{context_tokens:,} tokens"
                    )
        else:
            # Mode single-turn classique (backward compatible)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            self._last_window_stats = {}

        # Step 3b: Cache lookup (S18 -- C3, S19 -- G3 multi-turn)
        # Cache s'applique en single-turn ET multi-turn (conversation hashing)
        self._last_cache_hit = False
        self._s68_cache_hit = False
        self._s68_cache_key = ""
        cache_key = ""

        # S193 TC-04: fingerprint of the fully assembled generation context
        # (system prompt AFTER memory / project-RAG / web-search / archive /
        # optimizer injection). Passed to every semantic-cache get/put so a
        # response generated under a different context is never served for a
        # merely similar (or even identical) query.
        _ctx_fp = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

        # S68: Semantic cache check (new get/put API) -- single-turn only
        if (
            not no_cache
            and not use_conversation
            and SEMANTIC_CACHE_AVAILABLE
            and _semantic_cache is not None
            and _semantic_cache.enabled
        ):
            try:
                s68_entry = _semantic_cache.get(
                    user_content,
                    conversation_id=conversation_id,
                    model=routing.model,
                    context_fingerprint=_ctx_fp,
                )
                if s68_entry is not None:
                    self._last_cache_hit = True
                    self._s68_cache_hit = True
                    self._s68_cache_key = s68_entry.query_hash
                    hit_label = "S68-" + s68_entry.match_type.upper()
                    status(
                        f"[{hit_label}] Hit for {routing.model} "
                        f"(sim={s68_entry.similarity:.4f})"
                    )
                    yield s68_entry.response
                    self._current_task = None
                    return refined_question, s68_entry.response
            except Exception as e:
                logger.debug("S68 cache lookup error: %s", e)

        if (
            not no_cache
            and self._cache_enabled
            and RESPONSE_CACHE_AVAILABLE
            and _response_cache is not None
            and _response_cache.enabled
        ):
            if not use_conversation:
                # Single-turn: key based on model + prompt + query
                cache_key = _response_cache.make_cache_key(
                    routing.model, system_prompt, user_content
                )
            else:
                # Multi-turn (S19 G3): key based on model + prompt + history + query
                # Use the history messages (without system and current user)
                history_msgs = [
                    m for m in messages
                    if m.get("role") != "system"
                ][:-1]  # Exclude the last one (current message)
                cache_key = _response_cache.make_conversation_cache_key(
                    routing.model, system_prompt, history_msgs, user_content
                )
            cached = _response_cache.get(cache_key)

            # S23 G1: Semantic fallback on exact miss (single-turn only)
            semantic_hit = False
            if (
                cached is None
                and not use_conversation
                and SEMANTIC_CACHE_AVAILABLE
                and _semantic_cache is not None
                and _semantic_cache.enabled
            ):
                try:
                    sem_entry, sim, match_type = _semantic_cache.get_with_fallback(
                        _response_cache, cache_key, routing.model, user_content,
                        context_fingerprint=_ctx_fp,
                    )
                    if sem_entry is not None and match_type == "semantic":
                        cached = sem_entry
                        semantic_hit = True
                        logger.info(
                            f"Semantic cache hit: sim={sim:.4f}, "
                            f"key={sem_entry.cache_key[:12]}..."
                        )
                except Exception as e:
                    logger.debug(f"Semantic cache fallback error: {e}")

            if cached is not None:
                # Cache hit: serve the response instantly
                self._last_cache_hit = True
                hit_type = "SEMANTIC" if semantic_hit else "CACHE"
                status(f"[{hit_type}] Hit for {routing.model} (key={cache_key[:8]}...)")
                yield cached.response

                # Save multi-turn even on cache hit (S19 G3)
                if use_conversation and persist and cached.response:
                    try:
                        conversation_manager.add_message(
                            conversation_id, "user", user_content
                        )
                        conversation_manager.add_message(
                            conversation_id, "assistant", cached.response,
                            model=routing.model
                        )
                    except Exception as e:
                        logger.error(f"Conversation save error (cache hit): {e}")

                self._current_task = None
                return refined_question, cached.response

        # Step 4: Execute with streaming (with keepalive for Gradio)

        # S71: Offline check -- if Ollama is unreachable, enqueue for later
        if (
            NETWORK_MANAGER_AVAILABLE
            and _network_manager is not None
            and not _network_manager.is_online
        ):
            offline_msg = (
                "[Offline] Ollama is currently unreachable. "
            )
            if SYNC_QUEUE_AVAILABLE and _sync_queue is not None:
                entry = _sync_queue.enqueue(
                    query=user_content,
                    task_type=routing.task_type,
                    model=routing.model,
                )
                if entry is not None:
                    offline_msg += (
                        "Your request has been queued and will be "
                        "processed when connectivity returns."
                    )
                    logger.info("S71: Request queued offline (id=%s)", entry.id)
                else:
                    offline_msg += "The offline queue is full. Please try again later."
            else:
                offline_msg += "Please check your Ollama connection."

            status("[!] Ollama offline -- request queued")
            self._last_offline_queued = True
            yield offline_msg
            self._current_task = None
            return refined_question, offline_msg

        # S65: Use template temperature override if available
        effective_temperature = routing.temperature
        if _template_temp_override is not None:
            effective_temperature = _template_temp_override
            logger.debug(
                f"S65: template temperature override "
                f"{routing.temperature} -> {effective_temperature}"
            )

        status(f"[>] Generating with {routing.model} (temp={effective_temperature})...")

        # S224: governor admission (R-01, chat semantics D3: downsize then
        # refuse, never a silent queue). The requested ctx is the measured
        # prompt total plus the model's output reserve (spec 4.2); the
        # admitted value is sent as options["num_ctx"] in stream_thread --
        # the first num_ctx this project sends at all.
        _gov_requested_ctx = None
        try:
            if use_conversation:
                _gov_measured = int(context_tokens)
            else:
                _gov_measured = sum(
                    self._estimate_tokens(
                        str(m.get("content", "")), routing.model
                    )
                    for m in messages
                )
            _gov_reserve = 4096
            if CONTEXT_MANAGER_AVAILABLE:
                _gov_reserve = int(
                    cm_get_model_limits(routing.model).max_output
                )
            _gov_requested_ctx = _gov_measured + _gov_reserve
        except Exception as e:
            logger.debug(f"S224: requested_ctx estimate failed open: {e}")
        _gov_decision = _governor_admit(
            routing.model, _gov_requested_ctx, caller="chat"
        )
        if _gov_decision is not None and not _gov_decision.admitted:
            _gov_msg = _gov_decision.refusal_payload().get(
                "message", "resource admission refused"
            )
            status("[!] Resource admission refused")
            refusal_msg = f"[ERR] {_gov_msg}"
            yield refusal_msg
            self._current_task = None
            return refined_question, refusal_msg

        full_response = ""
        start_time = time.time()

        # Use queue and thread for keepalive (prevents Gradio timeout during model loading)
        chunk_queue = queue.Queue()
        thread_result = {"error": None, "done": False}

        def stream_thread() -> None:
            """Run inference in separate thread, push chunks to queue."""
            # S224: the admission ticket is thread-local and the backend
            # hook runs on the consuming thread (a generator head executes
            # at first iteration), so the ticket is held HERE and released
            # in the finally below.
            _governor_hold_ticket(_gov_decision)
            try:
                # Retrieve keep_alive duration from warmup manager
                ka = "30m"
                if MODEL_WARMUP_AVAILABLE and _model_warmup:
                    ka = _model_warmup.keep_alive
                # S225: per-decision keep_alive override (Section 5
                # step 1) -- the governor's soft-pressure value takes
                # precedence over the warmup default for THIS call only.
                if _gov_decision is not None and _gov_decision.keep_alive:
                    ka = _gov_decision.keep_alive

                options = {"temperature": effective_temperature}
                # S224: the admitted context (spec 4.2).
                if _gov_decision is not None and _gov_decision.num_ctx:
                    options["num_ctx"] = int(_gov_decision.num_ctx)
                _vision_images = images or getattr(routing, "images", None)

                # S105: Use backend abstraction when available
                _use_backend = (
                    INFERENCE_BACKEND_AVAILABLE
                    and get_backend_registry
                    and get_backend_registry().active is not None
                )

                if _use_backend:
                    backend = get_backend_registry().active
                    stream_iter = backend.stream(
                        model=routing.model,
                        messages=messages,
                        options=options,
                        keep_alive=ka,
                        think=bool(think),
                        images=_vision_images,
                    )

                    for chunk in stream_iter:
                        if self._cancel_event.is_set():
                            chunk_queue.put(("cancel", None))
                            break

                        # StreamChunk has .thinking and .content directly
                        if think and chunk.thinking:
                            chunk_queue.put(("thinking", chunk.thinking))
                        if chunk.content:
                            chunk_queue.put(("chunk", chunk.content))

                        if time.time() - start_time > routing.timeout:
                            chunk_queue.put(("timeout", None))
                            break
                else:
                    # Fallback: direct ollama.chat() call
                    chat_kwargs = dict(
                        model=routing.model,
                        messages=messages,
                        options=options,
                        stream=True,
                        keep_alive=ka,
                    )
                    # S259: the native think switch rides the tri-state
                    # helper. The historical default is preserved exactly
                    # (truthy -> {"think": True}, falsy -> nothing sent);
                    # explicit suppression (False) is the helper's third
                    # state, threaded end to end as a host-verified
                    # follow-up per INFERENCE_PERF_S259.md.
                    chat_kwargs.update(
                        _native_think_kwargs(True if think else None)
                    )
                    # S48/S92: Embed images in the last user message
                    if _vision_images:
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                msg["images"] = _vision_images
                                break

                    stream = ollama.chat(**chat_kwargs)

                    for chunk in stream:
                        if self._cancel_event.is_set():
                            chunk_queue.put(("cancel", None))
                            break

                        # S42: Handle thinking content in stream
                        if think and "message" in chunk:
                            msg = chunk["message"]
                            thinking_text = ""
                            content_text = ""

                            if hasattr(msg, "thinking"):
                                thinking_text = msg.thinking or ""
                            elif isinstance(msg, dict) and "thinking" in msg:
                                thinking_text = msg.get("thinking", "") or ""

                            if hasattr(msg, "content"):
                                content_text = msg.content or ""
                            elif isinstance(msg, dict) and "content" in msg:
                                content_text = msg.get("content", "") or ""

                            if thinking_text:
                                chunk_queue.put(("thinking", thinking_text))
                            if content_text:
                                chunk_queue.put(("chunk", content_text))
                        elif "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            chunk_queue.put(("chunk", content))

                        if time.time() - start_time > routing.timeout:
                            chunk_queue.put(("timeout", None))
                            break

            except Exception as e:
                thread_result["error"] = str(e)
            finally:
                # S224: release the thread-local admission ticket.
                _governor_release_ticket()
                thread_result["done"] = True
                chunk_queue.put(("done", None))

        # Start streaming thread
        stream_thread_obj = threading.Thread(target=stream_thread, daemon=True)
        stream_thread_obj.start()

        # Process chunks with keepalive
        last_yield_time = time.time()
        thinking_buffer = ""
        while True:
            try:
                # Wait for chunk with timeout (keeps Gradio connection alive)
                event_type, content = chunk_queue.get(timeout=2.0)

                if event_type == "done":
                    break
                elif event_type == "thinking":
                    # S42: Emit the thinking content as a tuple
                    thinking_buffer += content
                    yield ("thinking", content)
                    last_yield_time = time.time()
                elif event_type == "chunk":
                    full_response += content
                    yield content
                    last_yield_time = time.time()  # noqa: F841
                elif event_type == "cancel":
                    full_response += "\n\n[Generation cancelled]"
                    yield "\n\n[Generation cancelled]"
                    break
                elif event_type == "timeout":
                    full_response += "\n\n[Timeout reached]"
                    yield "\n\n[Timeout reached]"
                    break

            except queue.Empty:
                # No chunk received, yield keepalive to prevent Gradio timeout
                elapsed = time.time() - start_time
                # Yield empty string to keep connection alive (invisible to user)
                # But log progress for debugging
                logger.debug(f"Keepalive: waiting for model response... ({elapsed:.0f}s)")
                # Don't yield visible text, just keep the generator active
                yield ""

        # Wait for thread to finish
        stream_thread_obj.join(timeout=5.0)

        # Check for thread errors
        if thread_result["error"]:
            error_msg = f"\n\n[ERR] Error: {thread_result['error']}"
            full_response += error_msg
            yield error_msg
            status(f"[ERR] Error: {thread_result['error']}")
        else:
            elapsed = time.time() - start_time
            status(f"[OK] Completed in {elapsed:.1f}s")

        # S72: Record performance metrics (non-blocking)
        if (
            PERFORMANCE_MONITOR_AVAILABLE
            and _performance_monitor is not None
            and _performance_monitor.enabled
            and full_response
            and not thread_result["error"]
        ):
            try:
                # Estimate token counts from text lengths (chars / 4)
                _est_tokens_in = max(1, len(user_content) // 4)
                _est_tokens_out = max(1, len(full_response) // 4)
                _quality_est = min(1.0, len(full_response) / max(1, len(user_content)))
                _performance_monitor.record_execution(
                    model=routing.model,
                    task_type=routing.task_type,
                    latency_ms=elapsed * 1000,
                    tokens_in=_est_tokens_in,
                    tokens_out=_est_tokens_out,
                    quality_score=min(1.0, _quality_est),
                )
            except Exception as _perf_err:
                logger.debug("S72: performance recording skipped: %s", _perf_err)

        # Step 5: Multi-turn save (NEW: v1.3.0)
        # Save messages to conversation after full reception
        if use_conversation and persist and full_response and not thread_result["error"]:
            try:
                conversation_manager.add_message(
                    conversation_id, "user", user_content
                )
                conversation_manager.add_message(
                    conversation_id, "assistant", full_response,
                    model=routing.model
                )
                conversation_manager.update_conversation_metadata(
                    conversation_id,
                    model=routing.model,
                    task_type=routing.task_type,
                )
                logger.info(
                    f"Conversation {conversation_id[:8]}... updated "
                    f"(+2 messages, model={routing.model})"
                )
            except Exception as e:
                logger.error(f"Conversation save error: {e}")

        # Step 6: Cache storage (S18 -- C3, S19 -- G3 multi-turn)
        # Store the response in cache for successful requests (single AND multi-turn)
        if (
            cache_key
            and full_response
            and not thread_result["error"]
            and RESPONSE_CACHE_AVAILABLE
            and _response_cache is not None
            and _response_cache.enabled
        ):
            try:
                _response_cache.put(
                    model=routing.model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    response=full_response,
                    task_type=routing.task_type,
                    explicit_key=cache_key,
                )
                logger.debug(f"Response cached: {cache_key[:12]}...")

                # S23 G1: Store the embedding for semantic search
                # (single-turn seulement, en arriere-plan)
                if (
                    not use_conversation
                    and SEMANTIC_CACHE_AVAILABLE
                    and _semantic_cache is not None
                    and _semantic_cache.enabled
                ):
                    try:
                        _semantic_cache.store_embedding(
                            cache_key=cache_key,
                            model=routing.model,
                            query_text=user_content,
                            context_fingerprint=_ctx_fp,
                        )
                    except Exception as e:
                        logger.debug(f"Semantic embedding storage skipped: {e}")

            except Exception as e:
                logger.error(f"Cache storage error: {e}")

        # S68: Store in semantic cache (new get/put API)
        if (
            not no_cache
            and full_response
            and not thread_result["error"]
            and not use_conversation
            and SEMANTIC_CACHE_AVAILABLE
            and _semantic_cache is not None
            and _semantic_cache.enabled
        ):
            try:
                s68_key = _semantic_cache.put(
                    query=user_content,
                    response=full_response,
                    model=routing.model,
                    metadata={"task_type": routing.task_type},
                    conversation_id=conversation_id,
                    context_fingerprint=_ctx_fp,
                )
                if s68_key:
                    self._s68_cache_key = s68_key
                    logger.debug("S68 cache put: %s", s68_key[:12])
            except Exception as e:
                logger.debug("S68 cache put skipped: %s", e)

        # Step 7: Code verification (S43)
        # If the response contains Python/R code blocks, verify them
        # SECURITY (S73): Skip when sandbox mode is active to prevent
        # LLM-generated code from being auto-executed on the host.
        self._last_verification_results = []
        _sandbox_mode_active = False
        try:
            from .tool_registry import tool_registry as _tr
            _sandbox_mode_active = (
                _tr is not None
                and hasattr(_tr, 'sandbox_mode')
                and _tr.sandbox_mode
            )
        except Exception:
            pass

        if (
            VERIFICATION_AVAILABLE
            and _verification_engine is not None
            and _verification_engine.available
            and full_response
            and not thread_result["error"]
            and not self._cancel_event.is_set()
            and not _sandbox_mode_active
        ):
            try:
                # Check whether the response contains executable blocks
                vresults = _verification_engine.verify_response_code_blocks(
                    response_text=full_response,
                    original_question=question,
                    model=routing.model,
                    timeout=30,
                )
                if vresults:
                    self._last_verification_results = vresults
                    # Log the result
                    for vr in vresults:
                        logger.info(
                            f"Code verification ({vr.language}): "
                            f"status={vr.status}, iterations={vr.iterations}"
                        )
            except Exception as e:
                logger.warning(f"Code verification failed: {e}")

        self._current_task = None
        return refined_question, full_response

    def execute_cascade(
        self,
        question: str,
        task_type: str | None = None,
        no_cache: bool = False,
        conversation_id: str | None = None,
    ) -> dict | None:
        """S69: Execute a query using cascading inference.

        Routes through progressively larger models, stopping at the first
        whose response meets the quality threshold.

        Args:
            question: User query.
            task_type: Optional task type hint.
            no_cache: If True, bypass S68 cache.
            conversation_id: Optional conversation ID for cache scope.

        Returns:
            CascadeResult with final response and tier details,
            or None if cascading is unavailable.
        """
        self._last_cascade_result = None

        if not CASCADING_AVAILABLE or _cascading_inference is None:
            logger.debug("Cascading inference not available")
            return None

        if not _cascading_inference.enabled:
            logger.debug("Cascading inference is disabled")
            return None

        # S68: Check cache before cascade
        if (
            not no_cache
            and SEMANTIC_CACHE_AVAILABLE
            and _semantic_cache is not None
            and _semantic_cache.enabled
        ):
            try:
                s68_entry = _semantic_cache.get(
                    question,
                    conversation_id=conversation_id,
                    context_fingerprint=_CTX_FP_NOCTX,
                )
                if s68_entry is not None:
                    self._s68_cache_hit = True
                    self._s68_cache_key = s68_entry.query_hash
                    logger.info(
                        "S69: S68 cache hit before cascade (sim=%.4f)",
                        s68_entry.similarity,
                    )
                    # Build a synthetic CascadeResult for the cached response
                    if _CascadeResult is not None:
                        result = _CascadeResult(
                            final_response=s68_entry.response,
                            model_used="cache",
                            tier_index=-1,
                            tier_name="cache",
                            score=1.0,
                        )
                        self._last_cascade_result = result
                        return result
            except Exception as e:
                logger.debug("S69: S68 cache lookup error: %s", e)

        # S224: governor admission of the FIRST tier model -- the one the
        # cascade is guaranteed to load. A refusal answers None (the
        # documented unavailability contract); later tiers are direct
        # callers inside cascading and ride the Section 8 residual this
        # bloc records. Defensive reads only: cascading is not edited.
        try:
            _tiers = getattr(_cascading_inference, "tiers", None) or []
            _first_tier_model = (
                str(getattr(_tiers[0], "model", "") or "") if _tiers else ""
            )
        except Exception:
            _first_tier_model = ""
        if _first_tier_model:
            _admission = _governor_admit(
                _first_tier_model, None, caller="chat"
            )
            if _admission is not None and not _admission.admitted:
                logger.warning(
                    "S224: cascade admission refused for first tier %s: %s",
                    _first_tier_model,
                    _admission.reason,
                )
                return None
            if _admission is not None and _admission.load_expected:
                _governor_account_load(_first_tier_model, _admission.num_ctx)

        # Run the cascade
        try:
            result = _cascading_inference.cascade(
                query=question,
                task_type=task_type,
            )
            self._last_cascade_result = result

            # S68: Store the final response in cache
            if (
                not no_cache
                and result.final_response
                and not result.final_response.startswith("[ERR]")
                and SEMANTIC_CACHE_AVAILABLE
                and _semantic_cache is not None
                and _semantic_cache.enabled
            ):
                try:
                    s68_key = _semantic_cache.put(
                        query=question,
                        response=result.final_response,
                        model=result.model_used,
                        metadata={"task_type": task_type or "", "cascade_tier": result.tier_name},
                        conversation_id=conversation_id,
                        context_fingerprint=_CTX_FP_NOCTX,
                    )
                    if s68_key:
                        self._s68_cache_key = s68_key
                        logger.debug("S69: S68 cache put after cascade: %s", s68_key[:12])
                except Exception as e:
                    logger.debug("S69: S68 cache put skipped: %s", e)

            return result

        except Exception as e:
            logger.error("Cascading inference error: %s", e)
            return None

    def execute_speculative(
        self,
        question: str,
        task_type: str | None = None,
        no_cache: bool = False,
        conversation_id: str | None = None,
    ) -> dict | None:
        """S70: Execute a query using speculative generation.

        Uses a fast draft model to generate a response, then a larger
        verify model to evaluate and correct it.

        Args:
            question: User query.
            task_type: Optional task type hint.
            no_cache: If True, bypass S68 cache.
            conversation_id: Optional conversation ID for cache scope.

        Returns:
            SpeculativeResult with final response and phase details,
            or None if speculative generation is unavailable.
        """
        self._last_speculative_result = None

        if not SPECULATIVE_AVAILABLE or _speculative_generator is None:
            logger.debug("Speculative generation not available")
            return None

        if not _speculative_generator.enabled:
            logger.debug("Speculative generation is disabled")
            return None

        # S68: Check cache before speculative generation
        if (
            not no_cache
            and SEMANTIC_CACHE_AVAILABLE
            and _semantic_cache is not None
            and _semantic_cache.enabled
        ):
            try:
                s68_entry = _semantic_cache.get(
                    question,
                    conversation_id=conversation_id,
                    context_fingerprint=_CTX_FP_NOCTX,
                )
                if s68_entry is not None:
                    self._s68_cache_hit = True
                    self._s68_cache_key = s68_entry.query_hash
                    logger.info(
                        "S70: S68 cache hit before speculative (sim=%.4f)",
                        s68_entry.similarity,
                    )
                    if _SpeculativeResult is not None:
                        result = _SpeculativeResult(
                            final_response=s68_entry.response,
                            draft_response="",
                            verify_response="",
                            draft_model="cache",
                            verify_model="cache",
                            draft_accepted=True,
                            iterations=0,
                            total_latency_ms=0.0,
                            draft_latency_ms=0.0,
                            verify_latency_ms=0.0,
                            convergence_score=1.0,
                        )
                        self._last_speculative_result = result
                        return result
            except Exception as e:
                logger.debug("S70: S68 cache lookup error: %s", e)

        # S224: governor admission folding the draft+verify pair into ONE
        # decision (spec Section 8). A refusal answers None -- the
        # documented unavailability contract of this funnel; the decision
        # sits in the ring and the caller's fallback path runs its own
        # admission. The pair's transport is a direct ollama call out of
        # the mechanical seam's reach, so the funnel accounts the load.
        try:
            _verify_model = str(
                getattr(_speculative_generator, "verify_model", "") or ""
            )
            _draft_model = str(
                getattr(_speculative_generator, "draft_model", "") or ""
            )
        except Exception:
            _verify_model, _draft_model = "", ""
        if _verify_model:
            _admission = _governor_admit(
                _verify_model,
                None,
                caller="chat",
                extra_models=[_draft_model] if _draft_model else None,
            )
            if _admission is not None and not _admission.admitted:
                logger.warning(
                    "S224: speculative admission refused for %s (+%s): %s",
                    _verify_model,
                    _draft_model or "no draft",
                    _admission.reason,
                )
                return None
            if _admission is not None and _admission.load_expected:
                _governor_account_load(_verify_model, _admission.num_ctx)
                if _draft_model:
                    _governor_account_load(_draft_model, None)

        # Run speculative generation
        try:
            result = _speculative_generator.generate(
                query=question,
                task_type=task_type,
            )
            self._last_speculative_result = result

            # S68: Store the final response in cache
            if (
                not no_cache
                and result.final_response
                and not result.final_response.startswith("[ERR]")
                and SEMANTIC_CACHE_AVAILABLE
                and _semantic_cache is not None
                and _semantic_cache.enabled
            ):
                try:
                    model_used = (
                        result.draft_model
                        if result.draft_accepted
                        else result.verify_model
                    )
                    s68_key = _semantic_cache.put(
                        query=question,
                        response=result.final_response,
                        model=model_used,
                        metadata={
                            "task_type": task_type or "",
                            "speculative_draft_accepted": result.draft_accepted,
                        },
                        conversation_id=conversation_id,
                        context_fingerprint=_CTX_FP_NOCTX,
                    )
                    if s68_key:
                        self._s68_cache_key = s68_key
                        logger.debug("S70: S68 cache put after speculative: %s", s68_key[:12])
                except Exception as e:
                    logger.debug("S70: S68 cache put skipped: %s", e)

            return result

        except Exception as e:
            logger.error("Speculative generation error: %s", e)
            return None

    def execute_simple(
        self,
        question: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.5,
    ) -> str:
        """
        Simple execution without streaming or refinement.

        Args:
            question: The question
            model: Model to use
            system_prompt: System prompt to use
            temperature: Temperature

        Returns:
            Complete response
        """
        try:
            # Retrieve keep_alive duration
            ka = "30m"
            if MODEL_WARMUP_AVAILABLE and _model_warmup:
                ka = _model_warmup.keep_alive

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            options = {"temperature": temperature}

            # S224: governor admission (R-01). The refusal rides the
            # established "Error: ..." contract of this helper.
            _admission = _governor_admit(model, None, caller="chat")
            if _admission is not None and not _admission.admitted:
                _msg = _admission.refusal_payload().get(
                    "message", "resource admission refused"
                )
                logger.warning(
                    f"Simple execution admission refused for {model}: {_msg}"
                )
                return f"Error: {_msg}"
            # S225: per-decision keep_alive override (Section 5 step 1) --
            # the governor's soft-pressure value takes precedence over
            # the warmup default for THIS call only.
            if _admission is not None and _admission.keep_alive:
                ka = _admission.keep_alive

            # S105: Use backend abstraction when available
            if INFERENCE_BACKEND_AVAILABLE and get_backend_registry:
                backend = get_backend_registry().active
                if backend:
                    # S224: ticket pass-through (thread-local, 4.4).
                    _governor_hold_ticket(_admission)
                    try:
                        resp = backend.generate(
                            model=model,
                            messages=messages,
                            options=options,
                            keep_alive=ka,
                        )
                    finally:
                        _governor_release_ticket()
                    return resp.content

            # Fallback: direct ollama call
            response = ollama.chat(
                model=model,
                messages=messages,
                options=options,
                keep_alive=ka,
            )
            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Simple execution error: {e}")
            return f"Error: {str(e)}"

    # -------------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel current generation."""
        self._cancel_event.set()
        logger.info("Cancellation requested")

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_event.is_set()

    def reset(self) -> None:
        """Reset executor state."""
        self._cancel_event.clear()
        self._current_task = None
        self._last_refined_question = None
        self._last_context_check = None
        self._last_window_stats = {}
        self._last_cache_hit = False
        self._last_verification_results = []  # S43
        self._last_tool_calls = []  # S45


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

executor = Executor()


def execute(
    question: str,
    routing: RoutingResult,
    document: str | None = None,
    refine: bool = True,
    conversation_id: str | None = None,
    images: list[str] | None = None,
) -> Generator[str, None, tuple[str, str]]:
    """Convenience function to execute a query."""
    return executor.execute(
        question, routing, document, refine,
        conversation_id=conversation_id,
        images=images,
    )


def get_prompt(task_type: str, variant: str = "standard") -> str:
    """Convenience function to get a prompt."""
    return executor.get_system_prompt(task_type, variant)


# =============================================================================
# TEST CLI
# =============================================================================

if __name__ == "__main__":
    from .analyzer import analyze
    from .router import router

    print("=== Executor Test ===\n")

    # Simple test
    question = "How to calculate the mean in R?"
    print(f"Question: {question}")

    # Analyze and route
    analysis = analyze(question)
    routing = router.route(analysis)

    print(f"Model: {routing.model}")
    print(f"Task: {routing.task_type}")
    print(f"Variant: {routing.prompt_variant}")
    print()

    # Show prompt
    prompt = executor.get_system_prompt(routing.task_type, routing.prompt_variant)
    print("System Prompt (excerpt):")
    print(prompt[:200] + "...")
    print()

    # Execute (no streaming for test)
    print("Response:")
    response = executor.execute_simple(
        question,
        routing.model,
        prompt,
        routing.temperature
    )
    print(response[:500] + "..." if len(response) > 500 else response)
