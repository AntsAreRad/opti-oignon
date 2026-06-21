"""
Diff-tracker plugin for Opti-Oignon.

Detects code blocks in LLM responses that are iterations of
previously seen code in the conversation. Computes a unified diff
(additions/deletions) and appends it after the code block.

Matching strategy:
1. Extract function/class names from both blocks
2. If names overlap, compute diff between those blocks
3. Otherwise, fall back to SequenceMatcher similarity ratio
4. Only show diff if similarity exceeds the configured threshold

Uses difflib from stdlib -- no external dependencies.
"""

import difflib
import logging
import re
import threading
from typing import Any

__plugin_name__: str = "diff-tracker"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_DEFAULT_SIMILARITY_THRESHOLD = 0.4
_DEFAULT_DIFF_FORMAT = "inline"
_DEFAULT_SHOW_STATS = True
_DEFAULT_MAX_HISTORY = 50

# =========================================================================
# Module-level state
# =========================================================================

_history_lock = threading.Lock()

# List of previously seen code blocks: (language, code, names)
_code_history: list[dict[str, Any]] = []

# =========================================================================
# Code block extraction
# =========================================================================

_CODE_BLOCK_RE = re.compile(
    r"```(\w*)\n([\s\S]*?)```",
    re.DOTALL,
)


def extract_code_blocks(text: str) -> list[dict[str, Any]]:
    """Extract fenced code blocks from markdown text.

    Parameters
    ----------
    text : str
        Markdown text containing code blocks.

    Returns
    -------
    list[dict]
        List of dicts with 'language', 'code', 'start', 'end' keys.
        'start' and 'end' are character positions of the full match.
    """
    blocks: list[dict[str, Any]] = []
    for m in _CODE_BLOCK_RE.finditer(text):
        lang = m.group(1).lower() or "unknown"
        code = m.group(2)
        blocks.append({
            "language": lang,
            "code": code,
            "full_match": m.group(0),
            "start": m.start(),
            "end": m.end(),
        })
    return blocks


# =========================================================================
# Name extraction (function/class identifiers)
# =========================================================================

# Python: def name(...), class Name
_PY_DEF_RE = re.compile(r"^\s*(?:def|class)\s+(\w+)", re.MULTILINE)
# JavaScript/TypeScript: function name(, const name =, class Name
_JS_DEF_RE = re.compile(
    r"^\s*(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=|class\s+(\w+))",
    re.MULTILINE,
)
# R: name <- function
_R_DEF_RE = re.compile(r"^\s*(\w+)\s*<-\s*function", re.MULTILINE)
# Rust/Go/Java: fn name(, func name(, void/int/... name(
_GENERIC_FUNC_RE = re.compile(
    r"^\s*(?:fn|func|pub\s+fn|pub\s+func)\s+(\w+)",
    re.MULTILINE,
)


def extract_names(code: str, language: str = "unknown") -> set[str]:
    """Extract function/class/variable names from code.

    Parameters
    ----------
    code : str
        Source code text.
    language : str
        Programming language hint.

    Returns
    -------
    set[str]
        Set of identifier names found.
    """
    names: set[str] = set()

    # Always try Python patterns (most common in LLM output)
    for m in _PY_DEF_RE.finditer(code):
        names.add(m.group(1))

    # JS/TS patterns
    if language in ("javascript", "typescript", "js", "ts", "jsx", "tsx", "unknown"):
        for m in _JS_DEF_RE.finditer(code):
            name = m.group(1) or m.group(2) or m.group(3)
            if name:
                names.add(name)

    # R patterns
    if language in ("r", "unknown"):
        for m in _R_DEF_RE.finditer(code):
            names.add(m.group(1))

    # Generic patterns (Rust, Go, etc.)
    for m in _GENERIC_FUNC_RE.finditer(code):
        names.add(m.group(1))

    return names


# =========================================================================
# Similarity computation
# =========================================================================


def compute_similarity(code_a: str, code_b: str) -> float:
    """Compute similarity ratio between two code blocks.

    Uses both line-level and character-level SequenceMatcher
    and returns the higher score. Line-level is better for large
    blocks with shared lines; character-level catches small edits
    within lines.

    Parameters
    ----------
    code_a : str
        First code block.
    code_b : str
        Second code block.

    Returns
    -------
    float
        Similarity ratio between 0.0 and 1.0.
    """
    line_ratio = difflib.SequenceMatcher(
        None,
        code_a.splitlines(),
        code_b.splitlines(),
    ).ratio()

    char_ratio = difflib.SequenceMatcher(
        None,
        code_a,
        code_b,
    ).ratio()

    return max(line_ratio, char_ratio)


def find_best_match(
    block: dict[str, Any],
    history: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    """Find the best matching previous code block.

    Strategy:
    1. Check for name overlap (function/class names)
    2. Among name-overlapping candidates, pick highest similarity
    3. If no name overlap, check raw similarity

    Parameters
    ----------
    block : dict
        Current code block with 'language', 'code', 'names' keys.
    history : list[dict]
        Previous code blocks.
    threshold : float
        Minimum similarity ratio to accept a match.

    Returns
    -------
    dict or None
        Best matching previous block, or None.
    """
    current_names = block["names"]
    current_code = block["code"]
    best_match: dict[str, Any] | None = None
    best_score: float = 0.0

    for prev in history:
        # Skip blocks in different languages (unless unknown)
        if (block["language"] != "unknown"
                and prev["language"] != "unknown"
                and block["language"] != prev["language"]):
            continue

        prev_names = prev["names"]

        # Name overlap bonus
        name_overlap = bool(current_names and prev_names and (current_names & prev_names))

        similarity = compute_similarity(current_code, prev["code"])

        # With name overlap, lower the threshold slightly
        effective_threshold = threshold * 0.7 if name_overlap else threshold

        if similarity >= effective_threshold and similarity > best_score:
            # Avoid matching identical blocks
            if similarity >= 0.999:
                continue
            best_score = similarity
            best_match = prev

    return best_match


# =========================================================================
# Diff generation
# =========================================================================


def generate_diff(
    old_code: str,
    new_code: str,
    show_stats: bool = True,
) -> str:
    """Generate a unified diff between two code blocks.

    Parameters
    ----------
    old_code : str
        Previous version of the code.
    new_code : str
        New version of the code.
    show_stats : bool
        Whether to include addition/deletion counts.

    Returns
    -------
    str
        Formatted diff string.
    """
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="previous",
        tofile="current",
        lineterm="",
    ))

    if not diff_lines:
        return ""

    # Count additions and deletions
    additions = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    deletions = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

    result_parts: list[str] = []

    if show_stats:
        result_parts.append(
            f"**Diff:** +{additions} additions, -{deletions} deletions"
        )

    # Format as a fenced diff block
    diff_text = "\n".join(diff_lines)
    result_parts.append(f"```diff\n{diff_text}\n```")

    return "\n".join(result_parts)


# =========================================================================
# Response annotation
# =========================================================================


def annotate_response(
    text: str,
    threshold: float,
    show_stats: bool,
    max_history: int,
) -> tuple[str, int]:
    """Annotate code blocks in response with diffs against history.

    Parameters
    ----------
    text : str
        Full LLM response text.
    threshold : float
        Minimum similarity threshold.
    show_stats : bool
        Show addition/deletion counts.
    max_history : int
        Maximum history size.

    Returns
    -------
    tuple[str, int]
        Annotated text and number of diffs inserted.
    """
    blocks = extract_code_blocks(text)
    if not blocks:
        return text, 0

    diffs_inserted = 0

    # Process blocks in reverse order to preserve positions
    with _history_lock:
        history_snapshot = list(_code_history)

    insertions: list[tuple[int, str]] = []

    for block in blocks:
        names = extract_names(block["code"], block["language"])
        block["names"] = names

        match = find_best_match(block, history_snapshot, threshold)
        if match:
            diff = generate_diff(match["code"], block["code"], show_stats)
            if diff:
                insertions.append((block["end"], diff))
                diffs_inserted += 1

    # Apply insertions in reverse order to preserve positions
    insertions.sort(key=lambda x: x[0], reverse=True)
    for pos, diff_text in insertions:
        text = text[:pos] + "\n\n" + diff_text + text[pos:]

    # Add current blocks to history
    with _history_lock:
        for block in blocks:
            _code_history.append({
                "language": block["language"],
                "code": block["code"],
                "names": block.get("names", set()),
            })
            # Trim history
            while len(_code_history) > max_history:
                _code_history.pop(0)

    return text, diffs_inserted


# =========================================================================
# Hook implementation
# =========================================================================


def hook_post_inference(ctx: Any) -> dict[str, Any] | None:
    """Post-inference hook: detect code iterations and append diffs.

    Scans the response for code blocks, matches them against
    previously seen code in the conversation, and appends unified
    diffs for modified blocks.
    """
    response = ctx.data.get("response", "")
    if not response:
        return None

    # Quick check: any code blocks?
    if "```" not in response:
        return None

    config = ctx.config or {}
    threshold = config.get("similarity_threshold", _DEFAULT_SIMILARITY_THRESHOLD)
    show_stats = config.get("show_stats", _DEFAULT_SHOW_STATS)
    max_history = config.get("max_history", _DEFAULT_MAX_HISTORY)

    annotated, diff_count = annotate_response(
        response, threshold, show_stats, max_history,
    )

    if diff_count == 0:
        return None

    return {
        "response": annotated,
        "diffs_found": diff_count,
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
    """Plugin shutdown: clear history."""
    global _code_history
    with _history_lock:
        _code_history.clear()
