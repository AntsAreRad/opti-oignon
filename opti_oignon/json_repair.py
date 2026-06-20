#!/usr/bin/env python3
"""
JSON REPAIR - OPTI-OIGNON v1.8.2 (S80)
========================================

Tolerant JSON extraction and repair for local LLM outputs.

Local models (8K-32K context) frequently produce malformed JSON:
- Markdown fences wrapping (```json ... ```)
- Trailing commas in objects and arrays
- Single-quoted strings instead of double-quoted
- JSON embedded in explanatory text
- Unescaped control characters
- Missing closing brackets
- Comments (// or /* */) inside JSON

This module provides progressive repair strategies:
1. Direct parse (fast path)
2. Strip markdown fences
3. Extract JSON substring from mixed text
4. Fix common syntax errors (commas, quotes, brackets)
5. Fallback: parse numbered list into structured steps

Author: Leon
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

# JRP-01 (S192): the previous pattern ```...\s*\n?(.*?)\n?\s*``` had an
# ambiguous whitespace sandwich around the lazy group and backtracked
# catastrophically on an unclosed fence followed by whitespace (5k chars
# already took seconds; 20k effectively hung). The rewrite anchors the lazy
# group on the closing-fence literal (linear scan); the caller strips the
# captured group, which preserves the previous trailing/leading-whitespace
# trimming behaviour.
_FENCE_PATTERN = re.compile(
    r"```(?:json|JSON|js|javascript)?[ \t]*\n?(.*?)```",
    re.DOTALL,
)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text.

    Handles ```json, ```JSON, ```js, and bare ``` fences.
    If multiple fenced blocks exist, returns the first one.

    Args:
        text: Raw text possibly containing markdown fences.

    Returns:
        Text with fences removed, or original text if no fences found.
    """
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    # Handle case where opening fence exists but no closing fence
    lines = text.strip().split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# JSON substring extraction
# ---------------------------------------------------------------------------

def extract_json_substring(text: str) -> str | None:
    """Find and extract the first valid JSON object or array from text.

    Scans for the first '{' or '[' and attempts to find the matching
    closing bracket using a bracket-counting approach that handles
    strings (including escaped quotes).

    Args:
        text: Text potentially containing embedded JSON.

    Returns:
        Extracted JSON string, or None if no JSON structure found.
    """
    # Find first { or [
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return None

    # Pick whichever comes first
    if obj_start == -1:
        start = arr_start
        open_char, close_char = "[", "]"
    elif arr_start == -1:
        start = obj_start
        open_char, close_char = "{", "}"
    elif obj_start <= arr_start:
        start = obj_start
        open_char, close_char = "{", "}"
    else:
        start = arr_start
        open_char, close_char = "[", "]"

    # Bracket-counting with string awareness
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            if in_string:
                escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # If we never balanced, return from start to end (might be truncated)
    return text[start:]


# ---------------------------------------------------------------------------
# Syntax repair
# ---------------------------------------------------------------------------

def fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing brackets.

    Handles: {key: value,} and [item,]

    Args:
        text: JSON string with possible trailing commas.

    Returns:
        JSON string with trailing commas removed.
    """
    # Remove comma followed by optional whitespace/newlines then } or ]
    return re.sub(r",\s*([}\]])", r"\1", text)


def fix_single_quotes(text: str) -> str:
    """Replace single-quoted strings with double-quoted strings.

    Only replaces quotes that appear to be JSON string delimiters,
    not apostrophes within words. Uses a state-machine approach
    to avoid replacing apostrophes in contractions.

    Args:
        text: JSON string with possible single quotes.

    Returns:
        JSON string with double quotes.
    """
    result = []
    i = 0
    in_double_string = False
    in_single_string = False
    # JRP-02 (S192): track the last significant (non-whitespace) appended
    # character incrementally instead of rebuilding and rstripping the whole
    # buffer at every quote ('"".join(result).rstrip()' made the pass
    # O(n^2) on quote-dense input -- 1.3s at 20k quotes).
    last_significant = ""

    def _track(s: str) -> None:
        nonlocal last_significant
        stripped = s.rstrip()
        if stripped:
            last_significant = stripped[-1]

    while i < len(text):
        ch = text[i]

        # Handle escape sequences
        if i + 1 < len(text) and ch == "\\":
            result.append(ch)
            result.append(text[i + 1])
            _track(ch)
            _track(text[i + 1])
            i += 2
            continue

        if ch == '"' and not in_single_string:
            in_double_string = not in_double_string
            result.append(ch)
            _track(ch)
        elif ch == "'" and not in_double_string:
            if in_single_string:
                # Closing single quote -> double quote
                in_single_string = False
                result.append('"')
                _track('"')
            else:
                # Check if this looks like a JSON string delimiter
                # (after :, [, {, ( or , or at start)
                if not last_significant or last_significant in ":,[{(":
                    in_single_string = True
                    result.append('"')
                    _track('"')
                else:
                    # Likely an apostrophe in text, keep as-is
                    result.append(ch)
                    _track(ch)
        else:
            # If inside single-quoted string and char is ", escape it
            if in_single_string and ch == '"':
                result.append('\\"')
                _track('"')
            else:
                result.append(ch)
                _track(ch)
        i += 1

    return "".join(result)


def strip_comments(text: str) -> str:
    """Remove C-style comments from JSON text.

    Handles // line comments and /* block comments */.
    Respects string boundaries (does not strip inside strings).

    Args:
        text: JSON string with possible comments.

    Returns:
        JSON string with comments removed.
    """
    result = []
    i = 0
    in_string = False
    escape_next = False

    while i < len(text):
        ch = text[i]

        if escape_next:
            result.append(ch)
            escape_next = False
            i += 1
            continue

        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            i += 1
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if in_string:
            result.append(ch)
            i += 1
            continue

        # Check for line comment
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            # Skip until end of line
            while i < len(text) and text[i] != "\n":
                i += 1
            continue

        # Check for block comment
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            i += 2
            while i + 1 < len(text):
                if text[i] == "*" and text[i + 1] == "/":
                    i += 2
                    break
                i += 1
            else:
                i += 1  # Unclosed block comment
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def fix_unescaped_newlines(text: str) -> str:
    """Escape literal newlines inside JSON string values.

    Args:
        text: JSON string with possible unescaped newlines in values.

    Returns:
        JSON string with newlines properly escaped inside strings.
    """
    result = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue

        if ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if in_string and ch == "\n":
            result.append("\\n")
            continue

        if in_string and ch == "\t":
            result.append("\\t")
            continue

        result.append(ch)

    return "".join(result)


def fix_missing_closing(text: str) -> str:
    """Attempt to close unclosed brackets at end of truncated JSON.

    Counts unmatched { and [ and appends matching closers.

    Args:
        text: Possibly truncated JSON string.

    Returns:
        JSON string with closing brackets appended if needed.
    """
    stack = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack:
            stack.pop()

    # Append missing closers in reverse order
    if stack:
        text = text.rstrip()
        # Remove trailing comma if present
        if text.endswith(","):
            text = text[:-1]
        text += "".join(reversed(stack))

    return text


# ---------------------------------------------------------------------------
# Numbered list fallback parser
# ---------------------------------------------------------------------------

_NUMBERED_STEP_PATTERN = re.compile(
    r"^\s*(\d+)\.\s+(.+?)$", re.MULTILINE
)


def parse_numbered_list(text: str) -> list[dict[str, Any]] | None:
    """Parse a numbered list into plan steps as a JSON fallback.

    When the LLM fails to produce valid JSON after retries, it may
    output a numbered list like:
        1. Create file utils.py with helper functions
        2. Edit main.py to import utils
        3. Run tests

    This parses such lists into PlanStep-compatible dicts.

    Args:
        text: Text containing a numbered list.

    Returns:
        List of step dicts, or None if no numbered list found.
    """
    matches = _NUMBERED_STEP_PATTERN.findall(text)
    if not matches or len(matches) < 1:
        return None

    steps = []
    for num_str, description in matches:
        description = description.strip()
        step = {
            "step_type": _infer_step_type(description),
            "description": description,
            "file_path": _extract_file_path(description),
            "command": _extract_command(description),
            "content": "",
            "old_str": "",
            "new_str": "",
        }
        steps.append(step)

    return steps


def _infer_step_type(description: str) -> str:
    """Infer step type from description text.

    Args:
        description: Step description string.

    Returns:
        One of 'create', 'edit', 'test', 'bash'.
    """
    lower = description.lower()
    if any(w in lower for w in ("create", "write", "new file", "add file")):
        return "create"
    if any(w in lower for w in ("edit", "modify", "update", "change", "fix", "replace")):
        return "edit"
    if any(w in lower for w in ("test", "pytest", "run test", "verify")):
        return "test"
    return "bash"


_FILE_PATH_PATTERN = re.compile(
    r"(?:^|\s)([\w./\-]+\.(?:py|js|ts|yaml|yml|json|toml|cfg|md|txt|html|css|svelte))"
)


def _extract_file_path(description: str) -> str:
    """Extract a file path from description text.

    Args:
        description: Step description string.

    Returns:
        Extracted file path or empty string.
    """
    match = _FILE_PATH_PATTERN.search(description)
    return match.group(1) if match else ""


_COMMAND_PATTERN = re.compile(r"`([^`]+)`")


def _extract_command(description: str) -> str:
    """Extract a command from backtick-delimited text in description.

    Args:
        description: Step description string.

    Returns:
        Extracted command or empty string.
    """
    match = _COMMAND_PATTERN.search(description)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Main repair pipeline
# ---------------------------------------------------------------------------

def repair_json(text: str) -> dict[str, Any] | list[Any]:
    """Attempt to parse and repair malformed JSON from LLM output.

    Applies progressive repair strategies in order:
    1. Direct parse (fast path for well-formed JSON)
    2. Strip markdown fences + parse
    3. Extract JSON substring from mixed text + parse
    4. Apply syntax fixes (commas, quotes, comments, escapes) + parse
    5. Fix missing closing brackets + parse

    Args:
        text: Raw LLM output text.

    Returns:
        Parsed JSON as dict or list.

    Raises:
        ValueError: If all repair strategies fail.
    """
    if not text or not text.strip():
        raise ValueError("Empty input text")

    original = text.strip()

    # Strategy 1: direct parse
    # JRP-03 (S192): deep-nesting input can make json.loads raise
    # RecursionError instead of JSONDecodeError; catch it at every strategy
    # so the documented ValueError contract holds on adversarial input.
    try:
        return json.loads(original)
    except (json.JSONDecodeError, RecursionError):
        pass

    # Strategy 2: strip markdown fences
    stripped = strip_markdown_fences(original)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, RecursionError):
        pass

    # Strategy 3: extract JSON substring
    extracted = extract_json_substring(stripped)
    if extracted:
        try:
            return json.loads(extracted)
        except (json.JSONDecodeError, RecursionError):
            pass
    else:
        extracted = stripped

    # Strategy 4: syntax fixes on extracted text
    fixed = extracted
    fixed = strip_comments(fixed)
    fixed = fix_trailing_commas(fixed)
    fixed = fix_single_quotes(fixed)
    fixed = fix_unescaped_newlines(fixed)

    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, RecursionError):
        pass

    # Strategy 5: fix missing closing brackets
    closed = fix_missing_closing(fixed)
    try:
        return json.loads(closed)
    except (json.JSONDecodeError, RecursionError) as exc:
        logger.debug(
            "All JSON repair strategies failed. Last error: %s", exc
        )
        raise ValueError(
            f"Failed to repair JSON after all strategies: {exc}"
        ) from exc


def repair_json_or_list(
    text: str,
) -> tuple[dict[str, Any] | list[Any] | None, list[dict[str, Any]] | None]:
    """Attempt JSON repair, falling back to numbered list parsing.

    Returns a tuple of (json_result, list_result). Exactly one will
    be non-None if parsing succeeded, both None if everything failed.

    Args:
        text: Raw LLM output text.

    Returns:
        Tuple of (parsed_json, parsed_list). At most one is non-None.
    """
    try:
        result = repair_json(text)
        return result, None
    except ValueError:
        pass

    # Fallback: numbered list
    steps = parse_numbered_list(text)
    if steps:
        return None, steps

    return None, None


# ---------------------------------------------------------------------------
# Prompt reinforcement for retries
# ---------------------------------------------------------------------------

JSON_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous response was not valid JSON. "
    "Respond ONLY with a single JSON object. "
    "Do NOT include any markdown fences (```), explanatory text, "
    "or comments. Start your response with { and end with }."
)

SIMPLIFIED_PLAN_SUFFIX = (
    "\n\nYour previous JSON responses could not be parsed. "
    "Instead, respond with a simple numbered list:\n"
    "1. Description of step one\n"
    "2. Description of step two\n"
    "...\n"
    "Include file paths and commands in each step description."
)
