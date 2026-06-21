"""
Response-stats plugin for Opti-Oignon.

Appends a statistics footer (or header) to LLM responses with:
- Estimated token count (word-based heuristic)
- Reading time estimate
- Flesch-Kincaid readability score (adapted for technical content)
- Word, sentence, paragraph counts
- Complexity label (simple / moderate / complex / advanced)

Pure text processing -- no external dependencies, no permissions needed.
"""

import logging
import math
import re
from typing import Any

__plugin_name__: str = "response-stats"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_DEFAULT_STATS = [
    "words", "sentences", "paragraphs",
    "tokens", "reading_time", "readability", "complexity",
]
_DEFAULT_POSITION = "footer"
_DEFAULT_STYLE = "compact"
_MIN_WORDS = 20

# Average reading speed in words per minute
_WPM = 238

# Average tokens-per-word ratio for English text (LLM tokenizer heuristic)
_TOKENS_PER_WORD = 1.33

# =========================================================================
# Code block handling
# =========================================================================

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _strip_code_for_analysis(text: str) -> str:
    """Remove code blocks and inline code for readability analysis."""
    cleaned = _CODE_BLOCK_RE.sub("", text)
    cleaned = _INLINE_CODE_RE.sub("", cleaned)
    return cleaned


# =========================================================================
# Text analysis utilities
# =========================================================================

# Sentence boundary detection
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d\"'])"
    r"|(?<=[.!?])\s*\n",
)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count sentences in text.

    Uses sentence boundary regex. Falls back to counting
    terminal punctuation if regex yields zero.
    """
    if not text.strip():
        return 0

    parts = _SENTENCE_RE.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if parts:
        return len(parts)

    # Fallback: count terminal punctuation
    terminals = len(re.findall(r"[.!?]+", text))
    return max(terminals, 1)


def count_paragraphs(text: str) -> int:
    """Count paragraphs (separated by blank lines)."""
    paras = re.split(r"\n\s*\n", text.strip())
    return len([p for p in paras if p.strip()])


def count_syllables(word: str) -> int:
    """Estimate syllable count for an English word.

    Uses a vowel-group heuristic with adjustments for
    common patterns (silent e, -le endings, etc.).

    Parameters
    ----------
    word : str
        A single word (lowercase).

    Returns
    -------
    int
        Estimated syllable count (minimum 1).
    """
    word = word.lower().strip()
    if not word:
        return 0

    # Remove trailing 'e' (silent e)
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]

    # Count vowel groups
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)

    # Adjust for common patterns
    if word.endswith("le") and len(word) > 2 and word[-3] not in "aeiouy":
        count += 1

    return max(count, 1)


def count_syllables_text(text: str) -> int:
    """Count total syllables in text."""
    words = re.findall(r"[a-zA-Z]+", text)
    return sum(count_syllables(w) for w in words)


# =========================================================================
# Readability metrics
# =========================================================================


def flesch_kincaid_grade(text: str) -> float:
    """Calculate Flesch-Kincaid Grade Level.

    FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

    Parameters
    ----------
    text : str
        Input text (code blocks should be stripped).

    Returns
    -------
    float
        Grade level score. Higher = more complex.
        Clamped to [0, 25] range.
    """
    words = count_words(text)
    if words == 0:
        return 0.0

    sentences = count_sentences(text)
    if sentences == 0:
        sentences = 1

    syllables = count_syllables_text(text)

    grade = (
        0.39 * (words / sentences)
        + 11.8 * (syllables / words)
        - 15.59
    )

    return max(0.0, min(grade, 25.0))


def flesch_reading_ease(text: str) -> float:
    """Calculate Flesch Reading Ease score.

    FRE = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)

    Parameters
    ----------
    text : str
        Input text (code blocks should be stripped).

    Returns
    -------
    float
        Reading ease score. Higher = easier to read.
        Clamped to [0, 100] range.
    """
    words = count_words(text)
    if words == 0:
        return 100.0

    sentences = count_sentences(text)
    if sentences == 0:
        sentences = 1

    syllables = count_syllables_text(text)

    ease = (
        206.835
        - 1.015 * (words / sentences)
        - 84.6 * (syllables / words)
    )

    return max(0.0, min(ease, 100.0))


def complexity_label(fk_grade: float) -> str:
    """Map Flesch-Kincaid grade to a complexity label.

    Parameters
    ----------
    fk_grade : float
        Flesch-Kincaid grade level.

    Returns
    -------
    str
        One of: simple, moderate, complex, advanced.
    """
    if fk_grade < 6.0:
        return "simple"
    elif fk_grade < 10.0:
        return "moderate"
    elif fk_grade < 14.0:
        return "complex"
    else:
        return "advanced"


# =========================================================================
# Token estimation
# =========================================================================


def estimate_tokens(text: str) -> int:
    """Estimate token count using word-based heuristic.

    Approximation: ~1.33 tokens per word for English text.
    Code blocks count more heavily (variable names, symbols).

    Parameters
    ----------
    text : str
        Full response text.

    Returns
    -------
    int
        Estimated token count.
    """
    # Count words in prose
    prose = _strip_code_for_analysis(text)
    prose_words = count_words(prose)

    # Count code block content separately (higher token density)
    code_blocks = _CODE_BLOCK_RE.findall(text)
    code_text = " ".join(code_blocks)
    code_words = count_words(code_text)

    # Code has ~1.8 tokens per word (symbols, operators, etc.)
    tokens = prose_words * _TOKENS_PER_WORD + code_words * 1.8

    return int(math.ceil(tokens))


# =========================================================================
# Reading time
# =========================================================================


def reading_time_seconds(word_count: int) -> int:
    """Estimate reading time in seconds.

    Parameters
    ----------
    word_count : int
        Number of words.

    Returns
    -------
    int
        Estimated reading time in seconds.
    """
    if word_count <= 0:
        return 0
    return int(math.ceil(word_count / _WPM * 60))


def format_reading_time(seconds: int) -> str:
    """Format reading time as human-readable string.

    Parameters
    ----------
    seconds : int
        Reading time in seconds.

    Returns
    -------
    str
        Formatted string like '< 1 min', '2 min', '1 min 30s'.
    """
    if seconds < 30:
        return "< 1 min"
    minutes = seconds // 60
    remaining = seconds % 60
    if remaining < 15:
        return f"{minutes} min"
    return f"{minutes} min {remaining}s"


# =========================================================================
# Stats computation
# =========================================================================


def compute_stats(text: str) -> dict[str, Any]:
    """Compute all response statistics.

    Parameters
    ----------
    text : str
        Full LLM response text.

    Returns
    -------
    dict
        Dictionary with all computed stats.
    """
    # Analyze prose (without code)
    prose = _strip_code_for_analysis(text)

    words = count_words(text)
    prose_words = count_words(prose)  # noqa: F841
    sentences = count_sentences(prose)
    paragraphs = count_paragraphs(text)
    tokens = estimate_tokens(text)
    read_secs = reading_time_seconds(words)
    fk_grade = flesch_kincaid_grade(prose)
    fre = flesch_reading_ease(prose)
    label = complexity_label(fk_grade)

    return {
        "words": words,
        "sentences": sentences,
        "paragraphs": paragraphs,
        "tokens": tokens,
        "reading_time_seconds": read_secs,
        "reading_time": format_reading_time(read_secs),
        "fk_grade": round(fk_grade, 1),
        "reading_ease": round(fre, 1),
        "complexity": label,
    }


# =========================================================================
# Formatting
# =========================================================================


def format_stats_compact(
    stats: dict[str, Any],
    enabled: list[str],
) -> str:
    """Format statistics as a compact single-line string.

    Parameters
    ----------
    stats : dict
        Computed statistics.
    enabled : list[str]
        Which stats to include.

    Returns
    -------
    str
        Formatted stats line.
    """
    parts: list[str] = []

    if "words" in enabled:
        parts.append(f"{stats['words']} words")
    if "sentences" in enabled:
        parts.append(f"{stats['sentences']} sentences")
    if "paragraphs" in enabled:
        parts.append(f"{stats['paragraphs']} paragraphs")
    if "tokens" in enabled:
        parts.append(f"~{stats['tokens']} tokens")
    if "reading_time" in enabled:
        parts.append(stats["reading_time"])
    if "readability" in enabled:
        parts.append(f"FK {stats['fk_grade']}")
    if "complexity" in enabled:
        parts.append(stats["complexity"])

    return " | ".join(parts)


def format_stats_detailed(
    stats: dict[str, Any],
    enabled: list[str],
) -> str:
    """Format statistics as a detailed multi-line block.

    Parameters
    ----------
    stats : dict
        Computed statistics.
    enabled : list[str]
        Which stats to include.

    Returns
    -------
    str
        Formatted stats block.
    """
    lines: list[str] = ["**Response Statistics:**"]

    if "words" in enabled:
        lines.append(f"- Words: {stats['words']}")
    if "sentences" in enabled:
        lines.append(f"- Sentences: {stats['sentences']}")
    if "paragraphs" in enabled:
        lines.append(f"- Paragraphs: {stats['paragraphs']}")
    if "tokens" in enabled:
        lines.append(f"- Estimated tokens: ~{stats['tokens']}")
    if "reading_time" in enabled:
        lines.append(f"- Reading time: {stats['reading_time']}")
    if "readability" in enabled:
        lines.append(
            f"- Readability: FK grade {stats['fk_grade']} "
            f"(ease: {stats['reading_ease']})"
        )
    if "complexity" in enabled:
        lines.append(f"- Complexity: {stats['complexity']}")

    return "\n".join(lines)


# =========================================================================
# Hook implementation
# =========================================================================


def hook_post_inference(ctx: Any) -> dict[str, Any] | None:
    """Post-inference hook: compute and append response statistics.

    Computes word count, sentence count, paragraph count, estimated
    tokens, reading time, Flesch-Kincaid readability, and complexity
    label. Appends or prepends as configured.
    """
    response = ctx.data.get("response", "")
    if not response:
        return None

    config = ctx.config or {}
    enabled = config.get("enabled_stats", _DEFAULT_STATS)
    position = config.get("position", _DEFAULT_POSITION)
    style = config.get("style", _DEFAULT_STYLE)
    min_words = config.get("min_words", _MIN_WORDS)

    word_count = count_words(response)
    if word_count < min_words:
        return None

    stats = compute_stats(response)

    if style == "detailed":
        stats_block = format_stats_detailed(stats, enabled)
    else:
        stats_block = f"*{format_stats_compact(stats, enabled)}*"

    if position == "header":
        annotated = f"{stats_block}\n\n{response}"
    else:
        annotated = f"{response}\n\n{stats_block}"

    return {
        "response": annotated,
        "stats": stats,
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
