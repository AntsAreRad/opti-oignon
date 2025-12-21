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
import time
import logging
import ollama

from .config import config
from .router import RoutingResult

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
1. 🔍 **READ** the error carefully
2. 🎯 **IDENTIFY** the probable cause
3. ✅ **FIX** with working code
4. 💡 **EXPLAIN** to avoid in the future

## RESPONSE FORMAT
### 🔍 Error Analysis
[Error explanation]

### 🎯 Probable Cause
[What causes the problem]

### ✅ Fixed Code
```r
# Corrected code with comments
```

### 💡 Tip
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
1. 🔍 **READ** the traceback carefully
2. 🎯 **IDENTIFY** the probable cause
3. ✅ **FIX** with working code
4. 💡 **EXPLAIN** to avoid in the future

## RESPONSE FORMAT
### 🔍 Error Analysis
[Traceback explanation]

### 🎯 Probable Cause
[What causes the problem]

### ✅ Fixed Code
```python
# Corrected code with comments
```

### 💡 Tip
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

Help the user with their planning.""",

        "reasoning": """Planning expert. Think deeply. User's language.

<thinking>
1. What's the real objective?
2. What are the constraints?
3. What are the options?
4. What approach is optimal?
</thinking>

Then provide a structured plan.""",

        "fast": """Quick planning. Objective, key steps, next action. User's language.""",
    },
    
    "planning_deep": {
        "standard": """You are a strategic advisor expert in deep analysis.
Respond in the user's language.

## YOUR METHOD
1. **Analyze** all aspects of the problem
2. **Evaluate** options with pros/cons
3. **Reason** step by step
4. **Recommend** with clear justification

Provide comprehensive analysis and recommendations.""",
        
        "reasoning": """Strategic advisor. Deep analysis. User's language.""",
        "fast": """Strategic advisor. Quick analysis. User's language.""",
    },
    
    # ----- SIMPLE QUESTION -----
    "simple_question": {
        "standard": """You are a helpful assistant. Answer clearly and concisely.
Respond in the user's language (French if asked in French, English if asked in English).

Question:""",
        "reasoning": """Helpful assistant. Think before answering. User's language.""",
        "fast": """Helpful assistant. Quick answer. User's language.""",
    },
    
    # ----- LINUX -----
    "linux": {
        "standard": """You are a Linux/Bash expert.

## YOUR RULES
1. Explain commands clearly
2. Include safety warnings when relevant
3. Provide working examples
4. Mention alternatives when useful

## LANGUAGE: Respond in the user's language.

Answer the request.""",
        "reasoning": """Linux expert. Explain step by step. User's language.""",
        "fast": """Linux expert. Direct commands. User's language.""",
    },
}

# Default prompts for unknown task types
DEFAULT_PROMPTS = {
    "standard": """You are a helpful AI assistant.
Respond in the user's language (match their language).
Answer clearly and helpfully.""",
    "reasoning": """Think step by step. User's language.""",
    "fast": """Quick answer. User's language.""",
}

# Refinement prompt
REFINE_PROMPT = """Improve this question to get a better answer from an AI.
Keep the same language as the original question.
Make it clearer and more specific without changing the intent.

Context:
{context}

Original question:
{question}

Improved question (same language):"""


# =============================================================================
# MAIN CLASS
# =============================================================================

class Executor:
    """
    Query executor for Ollama models.
    
    Handles system prompt loading, question refinement,
    streaming execution, and cancellation.
    """
    
    def __init__(self):
        """Initialize the executor."""
        self._cancel_event = threading.Event()
        self._current_task: Optional[str] = None
        # FIX: Store the last refined question for retrieval after streaming
        self._last_refined_question: Optional[str] = None
    
    @property
    def last_refined_question(self) -> Optional[str]:
        """
        Get the last refined question from the most recent execute() call.
        
        This property allows retrieval of the refined question after
        streaming iteration completes, since Python generators don't
        expose their return value through normal for-loop iteration.
        
        Returns:
            The refined question, or None if no execution has occurred.
        """
        return self._last_refined_question
    
    # -------------------------------------------------------------------------
    # System Prompts
    # -------------------------------------------------------------------------
    
    def get_system_prompt(self, task_type: str, variant: str = "standard") -> str:
        """
        Get the system prompt for a task type and variant.
        
        Args:
            task_type: Task type (code_r, debug_python, etc.)
            variant: Variant (standard, reasoning, fast)
            
        Returns:
            System prompt string
        """
        task_prompts = PROMPTS.get(task_type, DEFAULT_PROMPTS)
        prompt = task_prompts.get(variant, task_prompts.get("standard", DEFAULT_PROMPTS["standard"]))
        return prompt
    
    def get_available_tasks(self) -> list:
        """Return list of available task types."""
        return list(PROMPTS.keys())
    
    def get_available_variants(self) -> list:
        """Return list of available prompt variants."""
        return ["standard", "reasoning", "fast"]
    
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
        Refine a question for better results.
        
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
    # Execution
    # -------------------------------------------------------------------------
    
    def execute(
        self,
        question: str,
        routing: RoutingResult,
        document: Optional[str] = None,
        refine: bool = True,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, Tuple[str, str]]:
        """
        Execute a complete query with streaming.
        
        Args:
            question: User's question
            routing: Routing result (model, temperature, etc.)
            document: Optional document/code for context
            refine: If True, refine question before execution
            on_status: Callback for status updates
            
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
        
        def status(msg: str):
            if on_status:
                on_status(msg)
            logger.info(msg)
        
        # Step 1: Refinement (optional)
        refined_question = question
        if refine:
            status(f"[>] Refining question with {routing.model}...")
            refined_question, error = self.refine_question(
                question, document, routing.model, config.get_temperature("refining")
            )
            if error:
                status(f"[!] Refinement failed: {error}")
            else:
                status("[OK] Question refined")
        
        # FIX: Store refined question in instance for later retrieval
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
        
        # Add document if present
        if document and not refine:
            messages[-1]["content"] += f"\n\n---\nDocument provided:\n{document}"
        
        # Step 4: Execute with streaming
        status(f"[>] Generating with {routing.model} (temp={routing.temperature})...")
        
        full_response = ""
        start_time = time.time()
        
        try:
            stream = ollama.chat(
                model=routing.model,
                messages=messages,
                options={"temperature": routing.temperature},
                stream=True,
            )
            
            for chunk in stream:
                if self._cancel_event.is_set():
                    full_response += "\n\n[Generation cancelled]"
                    yield "\n\n[Generation cancelled]"
                    break
                
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    full_response += content
                    yield content
                
                # Check timeout
                if time.time() - start_time > routing.timeout:
                    full_response += "\n\n[Timeout reached]"
                    yield "\n\n[Timeout reached]"
                    break
            
            elapsed = time.time() - start_time
            status(f"[OK] Completed in {elapsed:.1f}s")
            
        except Exception as e:
            error_msg = f"\n\n[ERR] Error: {str(e)}"
            full_response += error_msg
            yield error_msg
            status(f"[ERR] Error: {e}")
        
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
        self._last_refined_question = None  # Also reset refined question


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
