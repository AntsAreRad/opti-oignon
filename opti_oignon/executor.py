#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXECUTOR - OPTI-OIGNON 1.0
==========================

Execute queries to Ollama with appropriate system prompts.

This module handles:
- System prompt loading
- Question refinement
- Streaming query execution
- Error and timeout handling
- Request cancellation
- Context validation (NEW: Phase A4)

MULTILINGUAL LOGIC:
- The code and interface are in English
- BUT if the user asks in French → response in French
- If user asks in English → response in English
- The system detects user language and responds accordingly

Author: Léon
"""

from typing import Dict, Generator, Optional, Tuple, Callable, Any
from pathlib import Path
import threading
import queue
import time
import logging
import ollama

from .config import config
from .router import RoutingResult

# Context management import
try:
    from .context_manager import (
        get_context_manager,
        ContextCheck,
        check_context as cm_check_context,
        smart_truncate as cm_smart_truncate
    )
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    CONTEXT_MANAGER_AVAILABLE = False
    ContextCheck = None

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
- Keep the same language as the original (French→French, English→English)
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
        self._current_task: Optional[str] = None
        self._last_refined_question: Optional[str] = None
        self._last_context_check: Optional['ContextCheck'] = None
    
    @property
    def last_refined_question(self) -> Optional[str]:
        """Get the last refined question."""
        return self._last_refined_question
    
    @property
    def last_context_check(self) -> Optional['ContextCheck']:
        """Get the last context check result."""
        return self._last_context_check
    
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
        document: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Tuple[str, Optional[str]]:
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
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a prompt improvement expert."},
                    {"role": "user", "content": refine_prompt}
                ],
                options={"temperature": temperature},
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
    ) -> Tuple[str, Optional['ContextCheck'], Optional[str]]:
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
    # Execution
    # -------------------------------------------------------------------------
    
    def execute(
        self,
        question: str,
        routing: RoutingResult,
        document: Optional[str] = None,
        refine: bool = True,
        on_status: Optional[Callable[[str], None]] = None,
        auto_truncate: bool = False,
        validate_context: bool = True,
    ) -> Generator[str, None, Tuple[str, str]]:
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
            
        Yields:
            Response chunks in streaming
            
        Returns:
            Tuple (refined_question, full_response) at the end
            
        Note:
            The return value is also stored in self.last_refined_question
            for easy retrieval after iteration completes.
        """
        self._cancel_event.clear()
        self._current_task = routing.task_type
        self._last_context_check = None
        
        def status(msg: str):
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
                    yield f"\n\nOptions:"
                    yield f"\n- Enable auto-truncation"
                    yield f"\n- Summarize the document first"
                    yield f"\n- Use a model with larger context (e.g., nemotron-3-nano:30b)"
                    return question, f"[Context exceeded: {context_check.total_tokens} > {context_check.available_for_input}]"
                
                if context_warning:
                    status(f"[!] {context_warning}")
        
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
        system_prompt = self.get_system_prompt(routing.task_type, routing.prompt_variant)
        
        # Step 3: Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": refined_question}
        ]
        
        # Add document if present (always, regardless of refinement)
        if adjusted_document:
            messages[-1]["content"] += f"\n\n---\nDocument provided:\n{adjusted_document}"
        
        # Step 4: Execute with streaming (with keepalive for Gradio)
        status(f"[>] Generating with {routing.model} (temp={routing.temperature})...")
        
        full_response = ""
        start_time = time.time()
        
        # Use queue and thread for keepalive (prevents Gradio timeout during model loading)
        chunk_queue = queue.Queue()
        thread_result = {"error": None, "done": False}
        
        def stream_thread():
            """Run ollama.chat in separate thread, push chunks to queue."""
            try:
                stream = ollama.chat(
                    model=routing.model,
                    messages=messages,
                    options={"temperature": routing.temperature},
                    stream=True,
                )
                
                for chunk in stream:
                    if self._cancel_event.is_set():
                        chunk_queue.put(("cancel", None))
                        break
                    
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        chunk_queue.put(("chunk", content))
                    
                    # Check timeout
                    if time.time() - start_time > routing.timeout:
                        chunk_queue.put(("timeout", None))
                        break
                        
            except Exception as e:
                thread_result["error"] = str(e)
            finally:
                thread_result["done"] = True
                chunk_queue.put(("done", None))
        
        # Start streaming thread
        stream_thread_obj = threading.Thread(target=stream_thread, daemon=True)
        stream_thread_obj.start()
        
        # Process chunks with keepalive
        last_yield_time = time.time()
        while True:
            try:
                # Wait for chunk with timeout (keeps Gradio connection alive)
                event_type, content = chunk_queue.get(timeout=2.0)
                
                if event_type == "done":
                    break
                elif event_type == "chunk":
                    full_response += content
                    yield content
                    last_yield_time = time.time()
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
        
        self._current_task = None
        return refined_question, full_response
    
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
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                options={"temperature": temperature},
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


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

executor = Executor()


def execute(
    question: str,
    routing: RoutingResult,
    document: Optional[str] = None,
    refine: bool = True,
) -> Generator[str, None, Tuple[str, str]]:
    """Convenience function to execute a query."""
    return executor.execute(question, routing, document, refine)


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
