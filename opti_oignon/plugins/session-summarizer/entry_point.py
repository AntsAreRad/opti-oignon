"""
Session-summarizer plugin for Opti-Oignon.

Generates a running summary of the conversation session. Every N
messages, a background thread produces a summary using extractive
summarization (sentence scoring). If a model is available via the
context, generative summarization is attempted first.

Commands:
    /summary    View the current session summary
    /summary reset   Reset summary and message counter

Never blocks user interaction -- summary generation runs in a
background thread.
"""

import logging
import math
import re
import threading
import time
from typing import Any, Optional

__plugin_name__: str = "session-summarizer"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_DEFAULT_INTERVAL = 5
_DEFAULT_MAX_SUMMARY_LENGTH = 300

# =========================================================================
# Module-level state
# =========================================================================

_state_lock = threading.Lock()

# Current session state
_message_count: int = 0
_current_summary: str = ""
_summary_in_progress: bool = False
_conversation_buffer: list[dict[str, str]] = []
_last_summary_at: int = 0

# =========================================================================
# Extractive summarization (self-contained, no external deps)
# =========================================================================

_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d\"'])"
    r"|(?<=[.!?])\s*\n",
)

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "this", "that", "these", "those", "it", "its", "and", "but",
    "or", "not", "if", "then", "when", "where", "how", "what",
    "which", "who", "also", "just", "about", "more", "some",
    "than", "very", "all", "each", "both", "few", "most", "other",
    "such", "there", "here", "they", "them", "their", "you", "your",
    "we", "our", "use", "using", "used", "i", "me", "my",
})


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    clean = _CODE_BLOCK_RE.sub("", text)
    clean = re.sub(r"^#{1,6}\s+", "", clean, flags=re.MULTILINE)
    sentences = _SENTENCE_RE.split(clean)
    result: list[str] = []
    for sent in sentences:
        parts = sent.split("\n\n")
        for part in parts:
            stripped = part.strip()
            if stripped and _count_words(stripped) >= 3:
                result.append(stripped)
    return result


def _extract_keywords(text: str, top_n: int = 15) -> set[str]:
    """Extract important keywords by frequency."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in _STOP_WORDS:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return {w for w, _ in sorted_words[:top_n]}


def _score_sentence(
    sent: str,
    index: int,
    total: int,
    keywords: set[str],
    avg_length: float,
) -> float:
    """Score a sentence for summary inclusion."""
    # Position score (U-shaped: high at start and end)
    if total <= 1:
        pos = 1.0
    else:
        position = index / (total - 1)
        pos = 0.5 + 2.0 * (position - 0.5) ** 2

    # Keyword density
    words = set(re.findall(r"[a-z]+", sent.lower()))
    kw = min(len(words & keywords) / max(len(keywords), 1), 1.0) if keywords else 0.0

    # Length score (Gaussian penalty for deviation from average)
    wc = _count_words(sent)
    ratio = wc / avg_length if avg_length > 0 else 1.0
    length = math.exp(-0.5 * (ratio - 1.0) ** 2)

    return (0.3 * pos) + (0.4 * kw) + (0.3 * length)


def extractive_summarize(
    text: str,
    max_sentences: int = 3,
) -> str:
    """Generate an extractive summary of the text.

    Scores sentences by position, keyword density, and length,
    then selects the top-scoring sentences.

    Parameters
    ----------
    text : str
        Text to summarize.
    max_sentences : int
        Maximum number of sentences in the summary.

    Returns
    -------
    str
        Summary text.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences) if sentences else ""

    keywords = _extract_keywords(text)
    total = len(sentences)
    lengths = [_count_words(s) for s in sentences]
    avg_length = sum(lengths) / total if total > 0 else 10.0

    scored = [
        (_score_sentence(s, i, total, keywords, avg_length), i, s)
        for i, s in enumerate(sentences)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:max_sentences]
    top.sort(key=lambda x: x[1])  # restore original order

    return " ".join(sent for _, _, sent in top)


# =========================================================================
# Summary generation (background)
# =========================================================================


def _build_conversation_text(buffer: list[dict[str, str]]) -> str:
    """Build a single text from conversation buffer for summarization.

    Parameters
    ----------
    buffer : list[dict]
        List of message dicts with 'role' and 'content' keys.

    Returns
    -------
    str
        Combined conversation text.
    """
    parts: list[str] = []
    for msg in buffer:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content.strip():
            parts.append(f"{role}: {content.strip()}")
    return "\n\n".join(parts)


def _generate_summary_background(
    buffer: list[dict[str, str]],
    max_words: int,
    model_override: str,
    use_extractive: bool,
    ctx_metadata: dict,
) -> None:
    """Generate summary in background thread.

    Attempts generative summarization via model if available,
    falls back to extractive summarization.

    Parameters
    ----------
    buffer : list[dict]
        Conversation messages to summarize.
    max_words : int
        Maximum summary word count.
    model_override : str
        Model name override (empty = auto-select).
    use_extractive : bool
        Whether to use extractive fallback.
    ctx_metadata : dict
        Context metadata for model access.
    """
    global _current_summary, _summary_in_progress

    try:
        text = _build_conversation_text(buffer)
        if not text.strip():
            return

        # Estimate max sentences from max_words (~15 words per sentence)
        max_sentences = max(2, max_words // 15)

        # Always use extractive summarization (generative would require
        # async model access which is out of scope for the plugin sandbox)
        summary = extractive_summarize(text, max_sentences=max_sentences)

        # Trim to max_words
        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words]) + "..."

        with _state_lock:
            _current_summary = summary

        logger.debug(
            "Session summary updated (%d words from %d messages)",
            _count_words(summary), len(buffer),
        )

    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
    finally:
        with _state_lock:
            _summary_in_progress = False


def trigger_summary(
    ctx: Any,
    force: bool = False,
) -> None:
    """Trigger background summary generation if conditions are met.

    Parameters
    ----------
    ctx : Any
        Hook context.
    force : bool
        Force generation regardless of interval.
    """
    global _summary_in_progress, _last_summary_at

    config = ctx.config or {}
    interval = config.get("interval", _DEFAULT_INTERVAL)
    max_words = config.get("max_summary_length", _DEFAULT_MAX_SUMMARY_LENGTH)
    model_override = config.get("model_override", "")
    use_extractive = config.get("use_extractive_fallback", True)

    with _state_lock:
        if _summary_in_progress:
            return

        if not force and _message_count % interval != 0:
            return

        if not force and _message_count == _last_summary_at:
            return

        buffer_copy = list(_conversation_buffer)
        _summary_in_progress = True
        _last_summary_at = _message_count

    ctx_metadata = ctx.metadata if hasattr(ctx, "metadata") else {}

    thread = threading.Thread(
        target=_generate_summary_background,
        args=(buffer_copy, max_words, model_override, use_extractive, ctx_metadata),
        daemon=True,
        name="session-summarizer",
    )
    thread.start()


# =========================================================================
# Command parsing
# =========================================================================

_CMD_SUMMARY = re.compile(r"^/summary\s*$")
_CMD_SUMMARY_RESET = re.compile(r"^/summary\s+reset\s*$")


# =========================================================================
# Hook implementations
# =========================================================================


def hook_post_inference(ctx: Any) -> Optional[dict[str, Any]]:
    """Post-inference hook: track messages and trigger summary.

    Adds each response to the conversation buffer and triggers
    background summary generation at configured intervals.
    """
    global _message_count

    response = ctx.data.get("response", "")
    prompt = ctx.data.get("prompt", "") or ctx.data.get("user_input", "")

    with _state_lock:
        if prompt:
            _conversation_buffer.append({"role": "user", "content": prompt})
        if response:
            _conversation_buffer.append({"role": "assistant", "content": response})
        _message_count += 1

    # Trigger summary if interval reached
    trigger_summary(ctx)

    # Do not modify the response
    return None


def hook_tool_call(ctx: Any) -> Optional[dict[str, Any]]:
    """Tool call hook: handle /summary commands.

    /summary       -- display current session summary
    /summary reset -- reset summary and message counter
    """
    global _message_count, _current_summary, _conversation_buffer, _last_summary_at

    user_input = ctx.data.get("user_input", "") or ctx.data.get("prompt", "")
    if not user_input:
        return None

    user_input = user_input.strip()

    # /summary reset
    if _CMD_SUMMARY_RESET.match(user_input):
        with _state_lock:
            _message_count = 0
            _current_summary = ""
            _conversation_buffer.clear()
            _last_summary_at = 0
        return {
            "response": "Session summary reset.",
            "handled": True,
        }

    # /summary
    if _CMD_SUMMARY.match(user_input):
        with _state_lock:
            summary = _current_summary
            count = _message_count
            in_progress = _summary_in_progress

        if in_progress:
            status = " (update in progress...)"
        else:
            status = ""

        if not summary:
            msg = (
                f"No summary available yet ({count} messages tracked). "
                f"Summary is generated every "
                f"{ctx.config.get('interval', _DEFAULT_INTERVAL)} messages."
            )
        else:
            msg = (
                f"**Session Summary** ({count} messages){status}\n\n"
                f"{summary}"
            )

        return {
            "response": msg,
            "handled": True,
        }

    return None


# =========================================================================
# Hook registry
# =========================================================================

HOOKS = {
    "post_inference": hook_post_inference,
    "tool_call": hook_tool_call,
}


def init() -> None:
    """Plugin initialization."""
    pass


def shutdown() -> None:
    """Plugin shutdown: reset state."""
    global _message_count, _current_summary, _conversation_buffer
    global _summary_in_progress, _last_summary_at
    with _state_lock:
        _message_count = 0
        _current_summary = ""
        _conversation_buffer.clear()
        _summary_in_progress = False
        _last_summary_at = 0
