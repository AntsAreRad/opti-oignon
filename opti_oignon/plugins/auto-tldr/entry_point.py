"""
Auto-TLDR plugin for Opti-Oignon.

Detects long LLM responses and prepends a concise TL;DR summary.
Uses extractive summarization: sentences are scored by position,
keyword density, and length, then the top-scoring sentences are
selected as the summary.

Pure text processing — no external dependencies, no permissions needed.
"""

import logging
import math
import re
from typing import Any

__plugin_name__: str = "auto-tldr"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_WORD_THRESHOLD = 300
_MAX_SUMMARY_SENTENCES = 2
_SEPARATOR = "---"

# =========================================================================
# Text processing utilities
# =========================================================================

# Sentence boundary detection (handles abbreviations, decimals, etc.)
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d\"'])"
    r"|(?<=[.!?])\s*\n",
)

# Code block removal
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)

# Markdown header removal for scoring
_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Common filler / low-information phrases
_FILLER_PHRASES = frozenset({
    "in other words", "that being said", "it is worth noting",
    "as mentioned above", "as previously mentioned", "in addition to this",
    "on the other hand", "having said that", "it should be noted",
    "for what it is worth", "at the end of the day",
})


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Handles common edge cases: abbreviations, decimal numbers,
    markdown headers, and list items.

    Parameters
    ----------
    text : str
        Input text (code blocks should be removed first).

    Returns
    -------
    list[str]
        Non-empty sentence strings.
    """
    # Remove markdown headers for cleaner splitting
    clean = _HEADER_RE.sub("", text)

    # Split on sentence boundaries
    sentences = _SENTENCE_RE.split(clean)

    # Further split on double newlines (paragraph breaks)
    result: list[str] = []
    for sent in sentences:
        parts = sent.split("\n\n")
        for part in parts:
            stripped = part.strip()
            if stripped and count_words(stripped) >= 3:
                result.append(stripped)

    return result


# =========================================================================
# Sentence scoring
# =========================================================================

def _position_score(index: int, total: int) -> float:
    """Score based on sentence position.

    First and last sentences get higher scores (they tend to
    contain topic sentences and conclusions).

    Parameters
    ----------
    index : int
        Zero-based sentence index.
    total : int
        Total number of sentences.

    Returns
    -------
    float
        Position score between 0.0 and 1.0.
    """
    if total <= 1:
        return 1.0

    # Normalize to [0, 1]
    position = index / (total - 1)

    # U-shaped curve: high at start and end, low in middle
    return 0.5 + 2.0 * (position - 0.5) ** 2


def _keyword_density_score(sentence: str, keywords: set[str]) -> float:
    """Score based on keyword overlap.

    Parameters
    ----------
    sentence : str
        The sentence to score.
    keywords : set[str]
        Set of important keywords extracted from the full text.

    Returns
    -------
    float
        Density score between 0.0 and 1.0.
    """
    if not keywords:
        return 0.0

    words = set(re.findall(r"[a-z]+", sentence.lower()))
    if not words:
        return 0.0

    overlap = len(words & keywords)
    return min(overlap / max(len(keywords), 1), 1.0)


def _length_score(sentence: str, avg_length: float) -> float:
    """Score based on sentence length relative to average.

    Sentences close to average length score highest. Very short
    or very long sentences score lower.

    Parameters
    ----------
    sentence : str
        The sentence to score.
    avg_length : float
        Average sentence word count.

    Returns
    -------
    float
        Length score between 0.0 and 1.0.
    """
    word_count = count_words(sentence)
    if avg_length <= 0:
        return 0.5

    ratio = word_count / avg_length
    # Gaussian-like penalty for deviation from average
    return math.exp(-0.5 * (ratio - 1.0) ** 2)


def _filler_penalty(sentence: str) -> float:
    """Penalty for sentences containing filler phrases.

    Returns
    -------
    float
        Penalty between 0.0 (no filler) and 0.3 (heavy filler).
    """
    lower = sentence.lower()
    count = sum(1 for phrase in _FILLER_PHRASES if phrase in lower)
    return min(count * 0.15, 0.3)


def extract_keywords(text: str, top_n: int = 15) -> set[str]:
    """Extract important keywords from text by frequency.

    Filters out common stop words and short words.

    Parameters
    ----------
    text : str
        Full response text.
    top_n : int
        Number of top keywords to extract.

    Returns
    -------
    set[str]
        Set of keyword strings.
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "this", "that", "these", "those", "it", "its", "and", "but",
        "or", "not", "if", "then", "when", "where", "how", "what",
        "which", "who", "also", "just", "about", "more", "some",
        "than", "very", "all", "each", "both", "few", "most", "other",
        "such", "there", "here", "they", "them", "their", "you", "your",
        "we", "our", "use", "using", "used",
    }

    words = re.findall(r"[a-z]{3,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in stop_words:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return {w for w, _ in sorted_words[:top_n]}


def score_sentences(
    sentences: list[str],
    keywords: set[str],
) -> list[tuple[float, int, str]]:
    """Score all sentences and return sorted by score.

    Parameters
    ----------
    sentences : list[str]
        List of sentence strings.
    keywords : set[str]
        Important keywords from the full text.

    Returns
    -------
    list[tuple[float, int, str]]
        Sorted list of (score, original_index, sentence).
    """
    if not sentences:
        return []

    total = len(sentences)
    lengths = [count_words(s) for s in sentences]
    avg_length = sum(lengths) / total if total > 0 else 10.0

    scored: list[tuple[float, int, str]] = []
    for i, sent in enumerate(sentences):
        pos = _position_score(i, total)
        kw = _keyword_density_score(sent, keywords)
        length = _length_score(sent, avg_length)
        filler = _filler_penalty(sent)

        # Weighted combination
        score = (0.3 * pos) + (0.4 * kw) + (0.2 * length) - filler
        scored.append((score, i, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# =========================================================================
# Summary generation
# =========================================================================

def generate_summary(
    text: str,
    *,
    max_sentences: int = _MAX_SUMMARY_SENTENCES,
) -> str:
    """Generate an extractive summary of the text.

    Parameters
    ----------
    text : str
        The full response text.
    max_sentences : int
        Maximum number of sentences in the summary.

    Returns
    -------
    str
        The summary text, or empty string if summarization fails.
    """
    # Remove code blocks for analysis
    clean = _CODE_BLOCK_RE.sub("[code block]", text)

    sentences = split_sentences(clean)
    if len(sentences) <= max_sentences:
        return ""

    keywords = extract_keywords(clean)
    scored = score_sentences(sentences, keywords)

    # Pick top sentences, then reorder by original position
    top = scored[:max_sentences]
    top.sort(key=lambda x: x[1])  # sort by original index

    summary_parts = [sent for _, _, sent in top]
    return " ".join(summary_parts)


# =========================================================================
# Hook implementation
# =========================================================================

def hook_post_inference(ctx: Any) -> dict[str, Any] | None:
    """Post-inference hook: prepend TL;DR to long responses.

    Checks if the response exceeds the word threshold, generates
    an extractive summary, and prepends it to the response.
    """
    response = ctx.data.get("response", "")
    if not response:
        return None

    config = ctx.config or {}
    threshold = config.get("word_threshold", _WORD_THRESHOLD)
    max_sentences = config.get("max_summary_sentences", _MAX_SUMMARY_SENTENCES)
    separator = config.get("separator", _SEPARATOR)

    word_count = count_words(response)
    if word_count < threshold:
        return None

    summary = generate_summary(response, max_sentences=max_sentences)
    if not summary:
        return None

    # Prepend TL;DR block
    tldr_block = f"**TL;DR:** {summary}\n\n{separator}\n\n"
    annotated = tldr_block + response

    return {
        "response": annotated,
        "tldr_summary": summary,
        "tldr_word_count": word_count,
    }


# =========================================================================
# Hook registry
# =========================================================================

HOOKS = {
    "post_inference": hook_post_inference,
}


def init() -> None:
    """Plugin initialization."""
    pass


def shutdown() -> None:
    """Plugin shutdown."""
    pass
