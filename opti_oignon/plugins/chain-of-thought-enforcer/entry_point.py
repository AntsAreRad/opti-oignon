"""
Chain-of-thought enforcer plugin for Opti-Oignon.

Pre-inference: detects complex questions (math, logic, multi-step
reasoning) and injects a "think step by step" instruction into the
system message.

Post-inference: parses the response to separate reasoning from the
final answer and formats them with a clear visual separator.
"""

import logging
import re
from typing import Any

__plugin_name__: str = "chain-of-thought-enforcer"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Complexity detection
# =========================================================================

# Default keywords that suggest a complex question
DEFAULT_COMPLEXITY_KEYWORDS = [
    "why", "how", "compare", "contrast", "explain", "analyze",
    "evaluate", "prove", "derive", "calculate", "what if",
    "difference between", "pros and cons", "advantages",
    "step by step", "reasoning",
]

# Math/logic indicators
_MATH_OPERATORS = re.compile(
    r"[\+\-\*/\^=]|"
    r"\b(?:sum|product|integral|derivative|equation|solve|simplify|factor)\b",
    re.IGNORECASE,
)

# Multi-part question indicators
_MULTI_PART_PATTERN = re.compile(
    r"(?:first|second|third|1\)|2\)|3\)|a\)|b\)|c\)|\band\b.*\?)",
    re.IGNORECASE,
)

# Default CoT injection template
DEFAULT_INJECTION = (
    "Think step by step, show your reasoning, "
    "then give your final answer."
)

# =========================================================================
# Reasoning / answer separation patterns
# =========================================================================

# Markers that typically precede the final answer
_ANSWER_MARKERS = [
    r"(?:^|\n)\s*(?:\*\*)?(?:final answer|answer|conclusion|result|therefore|thus|in summary|to summarize|in conclusion|so,?)(?:\*\*)?[:\s]",
    r"(?:^|\n)\s*(?:\*\*)?(?:TL;?DR|TLDR)(?:\*\*)?[:\s]",
    r"(?:^|\n)\s*---+\s*\n",
]

_ANSWER_MARKER_RE = re.compile(
    "|".join(_ANSWER_MARKERS),
    re.IGNORECASE | re.MULTILINE,
)

# Markers that indicate reasoning sections
_REASONING_MARKERS = [
    r"(?:^|\n)\s*(?:\*\*)?(?:reasoning|thought process|analysis|let me think|let's think|step \d|thinking)(?:\*\*)?[:\s]",
]

_REASONING_MARKER_RE = re.compile(
    "|".join(_REASONING_MARKERS),
    re.IGNORECASE | re.MULTILINE,
)


# =========================================================================
# Complexity assessment
# =========================================================================

def is_complex_question(
    prompt: str,
    *,
    keywords: list[str] | None = None,
    min_words: int = 5,
) -> bool:
    """Determine if a prompt contains a complex question.

    A question is considered complex if it meets any of:
    - Contains complexity keywords (configurable)
    - Contains math/logic operators
    - Appears to be a multi-part question
    - Is sufficiently long (word count >= min_words) AND contains a question mark

    Parameters
    ----------
    prompt : str
        The user prompt to assess.
    keywords : list[str], optional
        Complexity trigger keywords. Defaults to built-in list.
    min_words : int
        Minimum word count for length-based trigger.

    Returns
    -------
    bool
        True if the prompt appears to be complex.
    """
    if not prompt or not prompt.strip():
        return False

    prompt_lower = prompt.lower().strip()
    words = prompt_lower.split()

    if len(words) < min_words:
        return False

    # Check complexity keywords
    kw_list = keywords or DEFAULT_COMPLEXITY_KEYWORDS
    for kw in kw_list:
        if kw.lower() in prompt_lower:
            return True

    # Check math/logic operators
    if _MATH_OPERATORS.search(prompt):
        return True

    # Check multi-part questions
    if _MULTI_PART_PATTERN.search(prompt):
        return True

    # Long question with question mark
    if "?" in prompt and len(words) >= min_words * 2:
        return True

    return False


# =========================================================================
# Response formatting
# =========================================================================

def split_reasoning_and_answer(text: str) -> tuple[str, str]:
    """Split a response into reasoning and final answer sections.

    Looks for answer markers to find the boundary. If no clear
    marker is found, returns the full text as answer with empty
    reasoning.

    Parameters
    ----------
    text : str
        The LLM response text.

    Returns
    -------
    tuple[str, str]
        (reasoning, answer) where either may be empty.
    """
    if not text:
        return ("", "")

    # Try to find an answer marker
    match = _ANSWER_MARKER_RE.search(text)
    if match:
        reasoning = text[:match.start()].strip()
        answer = text[match.end():].strip()
        # Include the marker label in answer if it is not just "---"
        marker_text = match.group(0).strip().strip("-").strip()
        if marker_text:
            answer = marker_text + " " + answer
        if reasoning and answer:
            return (reasoning, answer)

    # Try reasoning marker: everything after marker until answer marker
    reason_match = _REASONING_MARKER_RE.search(text)
    if reason_match:
        # Everything before reasoning marker is preamble
        after_reasoning = text[reason_match.end():]
        answer_match = _ANSWER_MARKER_RE.search(after_reasoning)
        if answer_match:
            reasoning = after_reasoning[:answer_match.start()].strip()
            answer = after_reasoning[answer_match.end():].strip()
            if reasoning and answer:
                return (reasoning, answer)

    # No clear separation found
    return ("", text)


def format_response(
    reasoning: str,
    answer: str,
    *,
    style: str = "separator",
) -> str:
    """Format reasoning and answer according to the chosen style.

    Parameters
    ----------
    reasoning : str
        The reasoning/thought process section.
    answer : str
        The final answer section.
    style : str
        Formatting style: "separator", "collapsible", or "plain".

    Returns
    -------
    str
        Formatted response.
    """
    if not reasoning:
        return answer

    if style == "collapsible":
        return (
            f"<details>\n<summary>Reasoning</summary>\n\n"
            f"{reasoning}\n\n</details>\n\n"
            f"**Answer:**\n\n{answer}"
        )
    elif style == "plain":
        return f"{reasoning}\n\n{answer}"
    else:
        # Default: separator style
        return (
            f"**Reasoning:**\n\n{reasoning}\n\n"
            f"---\n\n"
            f"**Answer:**\n\n{answer}"
        )


# =========================================================================
# Hook implementations
# =========================================================================

def hook_pre_inference(ctx: Any) -> dict[str, Any] | None:
    """Pre-inference hook: inject CoT instruction for complex questions.

    Reads ctx.data["prompt"] (or ctx.data["messages"]) and injects
    the CoT instruction into the system message if the question is
    deemed complex.
    """
    config = ctx.config or {}

    # Parse config
    kw_str = config.get("complexity_keywords", "")
    if kw_str and isinstance(kw_str, str):
        keywords = [k.strip() for k in kw_str.split(",") if k.strip()]
    else:
        keywords = None

    min_words = config.get("min_question_words", 5)
    template = config.get("injection_template", DEFAULT_INJECTION)

    # Get the user prompt
    prompt = ctx.data.get("prompt", "")
    messages = ctx.data.get("messages", [])

    # Extract last user message if using messages format
    user_text = prompt
    if not user_text and messages:
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

    if not user_text:
        return None

    # Check complexity
    if not is_complex_question(user_text, keywords=keywords, min_words=min_words):
        return None

    logger.debug("CoT enforcer: complex question detected, injecting instruction")

    # Inject into system message
    system_message = ctx.data.get("system_message", "")
    if system_message:
        enhanced_system = f"{system_message}\n\n{template}"
    else:
        enhanced_system = template

    result: dict[str, Any] = {
        "system_message": enhanced_system,
        "_cot_injected": True,
    }

    return result


def hook_post_inference(ctx: Any) -> dict[str, Any] | None:
    """Post-inference hook: format reasoning vs answer in response.

    Only activates if CoT was injected (checks _cot_injected flag)
    or if the response naturally contains reasoning markers.
    """
    response = ctx.data.get("response", "")
    if not response:
        return None

    config = ctx.config or {}
    style = config.get("format_style", "separator")

    # Check if CoT was injected or if response has reasoning markers
    cot_injected = ctx.data.get("_cot_injected", False)
    has_reasoning = _REASONING_MARKER_RE.search(response) is not None
    has_answer = _ANSWER_MARKER_RE.search(response) is not None

    if not (cot_injected or has_reasoning or has_answer):
        return None

    reasoning, answer = split_reasoning_and_answer(response)

    if not reasoning:
        return None

    formatted = format_response(reasoning, answer, style=style)
    return {"response": formatted}


# =========================================================================
# Hook registry
# =========================================================================

HOOKS = {
    "pre_inference": hook_pre_inference,
    "post_inference": hook_post_inference,
}


def init() -> None:
    """Plugin initialization."""
    pass


def shutdown() -> None:
    """Plugin shutdown."""
    pass
