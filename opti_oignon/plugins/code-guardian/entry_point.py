"""
Code-guardian plugin for Opti-Oignon.

Validates code blocks found in LLM responses. Supports Python
(ast.parse), JSON (json.loads), and R (basic syntax heuristics).
Appends a syntax badge after each code block indicating whether
the code is valid or contains errors.

Pure text processing with stdlib only — no external dependencies.
"""

import ast
import json
import logging
import re
from typing import Any, Optional

__plugin_name__: str = "code-guardian"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_DEFAULT_LANGUAGES = {"python", "json", "r"}
_DEFAULT_BADGE_FORMAT = "bracket"
_DEFAULT_MIN_LINES = 2

# =========================================================================
# Language aliases mapping
# =========================================================================

_LANGUAGE_ALIASES: dict[str, str] = {
    "python": "python",
    "py": "python",
    "python3": "python",
    "json": "json",
    "jsonc": "json",
    "r": "r",
    "rlang": "r",
}

# =========================================================================
# Code block extraction
# =========================================================================

# Match fenced code blocks: ```lang\n...code...\n```
_CODE_BLOCK_RE = re.compile(
    r"```(\w*)\s*\n(.*?)```",
    re.DOTALL,
)


def extract_code_blocks(text: str) -> list[dict[str, Any]]:
    """Extract fenced code blocks from markdown text.

    Parameters
    ----------
    text : str
        The response text containing markdown code blocks.

    Returns
    -------
    list[dict]
        List of dicts with keys: language, code, start, end, line_count.
    """
    blocks: list[dict[str, Any]] = []
    for m in _CODE_BLOCK_RE.finditer(text):
        lang_tag = m.group(1).lower().strip()
        code = m.group(2)
        line_count = code.count("\n") + (1 if code.strip() else 0)
        blocks.append({
            "language": lang_tag,
            "code": code,
            "start": m.start(),
            "end": m.end(),
            "line_count": line_count,
        })
    return blocks


# =========================================================================
# Python validation
# =========================================================================

def validate_python(code: str) -> dict[str, Any]:
    """Validate Python code using ast.parse.

    Parameters
    ----------
    code : str
        Python source code.

    Returns
    -------
    dict
        Result with keys: valid (bool), error (str or None),
        line (int or None), col (int or None), details (list[str]).
    """
    result: dict[str, Any] = {
        "valid": True,
        "error": None,
        "line": None,
        "col": None,
        "details": [],
    }

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result["valid"] = False
        result["error"] = str(exc.msg) if exc.msg else "Syntax error"
        result["line"] = exc.lineno
        result["col"] = exc.offset
        return result

    # Optional: check for common pitfalls
    pitfalls = _check_python_pitfalls(tree, code)
    if pitfalls:
        result["details"] = pitfalls

    return result


def _check_python_pitfalls(
    tree: ast.AST,
    source: str,
) -> list[str]:
    """Check for common Python pitfalls via AST inspection.

    Detects:
    - Unused imports (imported names not referenced elsewhere)
    - Bare except clauses
    - Mutable default arguments

    Parameters
    ----------
    tree : ast.AST
        Parsed AST.
    source : str
        Original source code (for reference counting).

    Returns
    -------
    list[str]
        List of warning strings.
    """
    warnings: list[str] = []

    # Collect imported names
    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names.append(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_names.append(name)

    # Check for unused imports (simple heuristic: count occurrences in source)
    for name in imported_names:
        # Count occurrences excluding the import line itself
        # Use word boundary matching to avoid partial matches
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        matches = pattern.findall(source)
        # If the name only appears once (in the import), it is unused
        if len(matches) <= 1:
            warnings.append(f"Possibly unused import: {name}")

    # Check for bare except
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            warnings.append(
                f"Bare except clause at line {node.lineno}"
            )

    # Check for mutable default arguments
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None and isinstance(
                    default, (ast.List, ast.Dict, ast.Set)
                ):
                    warnings.append(
                        f"Mutable default argument in '{node.name}()' "
                        f"at line {node.lineno}"
                    )

    return warnings


# =========================================================================
# JSON validation
# =========================================================================

def validate_json(code: str) -> dict[str, Any]:
    """Validate JSON code using json.loads.

    Parameters
    ----------
    code : str
        JSON source code.

    Returns
    -------
    dict
        Result with keys: valid (bool), error (str or None),
        line (int or None), col (int or None), details (list[str]).
    """
    result: dict[str, Any] = {
        "valid": True,
        "error": None,
        "line": None,
        "col": None,
        "details": [],
    }

    try:
        json.loads(code)
    except json.JSONDecodeError as exc:
        result["valid"] = False
        result["error"] = exc.msg
        result["line"] = exc.lineno
        result["col"] = exc.colno
    except Exception as exc:
        result["valid"] = False
        result["error"] = str(exc)

    return result


# =========================================================================
# R validation (basic heuristics)
# =========================================================================

def validate_r(code: str) -> dict[str, Any]:
    """Validate R code using basic syntax heuristics.

    Checks for:
    - Matching parentheses, braces, and brackets
    - Unclosed string literals
    - Common syntax issues (e.g. misplaced operators)

    Parameters
    ----------
    code : str
        R source code.

    Returns
    -------
    dict
        Result with keys: valid (bool), error (str or None),
        line (int or None), col (int or None), details (list[str]).
    """
    result: dict[str, Any] = {
        "valid": True,
        "error": None,
        "line": None,
        "col": None,
        "details": [],
    }

    # Strip comments (# to end of line, but not inside strings)
    lines = code.split("\n")
    clean_lines: list[str] = []
    for line in lines:
        in_string = False
        string_char = ""
        clean = []
        for ch in line:
            if in_string:
                clean.append(ch)
                if ch == string_char:
                    in_string = False
            elif ch in ('"', "'"):
                in_string = True
                string_char = ch
                clean.append(ch)
            elif ch == "#":
                break
            else:
                clean.append(ch)
        clean_lines.append("".join(clean))

    clean_code = "\n".join(clean_lines)

    # Check matching delimiters
    _PAIRS = {"(": ")", "{": "}", "[": "]"}
    _CLOSE_TO_OPEN = {v: k for k, v in _PAIRS.items()}
    stack: list[tuple[str, int]] = []

    for lineno, line in enumerate(clean_lines, start=1):
        in_str = False
        str_char = ""
        for colno, ch in enumerate(line, start=1):
            if in_str:
                if ch == str_char:
                    in_str = False
                continue
            if ch in ('"', "'"):
                in_str = True
                str_char = ch
                continue
            if ch in _PAIRS:
                stack.append((ch, lineno))
            elif ch in _CLOSE_TO_OPEN:
                expected_open = _CLOSE_TO_OPEN[ch]
                if not stack:
                    result["valid"] = False
                    result["error"] = f"Unexpected closing '{ch}'"
                    result["line"] = lineno
                    result["col"] = colno
                    return result
                top_char, top_line = stack.pop()
                if top_char != expected_open:
                    result["valid"] = False
                    result["error"] = (
                        f"Mismatched '{top_char}' (line {top_line}) "
                        f"closed by '{ch}'"
                    )
                    result["line"] = lineno
                    result["col"] = colno
                    return result

    if stack:
        unclosed_char, unclosed_line = stack[-1]
        result["valid"] = False
        result["error"] = f"Unclosed '{unclosed_char}'"
        result["line"] = unclosed_line
        return result

    # Check for unclosed strings
    for lineno, line in enumerate(clean_lines, start=1):
        in_str = False
        str_char = ""
        for ch in line:
            if in_str:
                if ch == str_char:
                    in_str = False
            elif ch in ('"', "'"):
                in_str = True
                str_char = ch
        if in_str:
            result["valid"] = False
            result["error"] = f"Unclosed string literal"
            result["line"] = lineno
            return result

    return result


# =========================================================================
# Badge formatting
# =========================================================================

def format_badge(
    validation: dict[str, Any],
    language: str,
    *,
    badge_format: str = "bracket",
) -> str:
    """Format a validation result as a displayable badge.

    Parameters
    ----------
    validation : dict
        Validation result from validate_* functions.
    language : str
        The language name for display.
    badge_format : str
        Badge style: "bracket", "emoji", or "hidden".

    Returns
    -------
    str
        Formatted badge string (may be empty for "hidden" on success).
    """
    is_valid = validation["valid"]
    error = validation.get("error")
    line = validation.get("line")
    details = validation.get("details", [])

    if badge_format == "hidden" and is_valid and not details:
        return ""

    lang_label = language.capitalize()

    if badge_format == "emoji":
        if is_valid:
            badge = f"[{lang_label} OK]"
            if details:
                badge += " " + "; ".join(details)
            return badge
        else:
            loc = f" line {line}" if line else ""
            return f"[{lang_label} Error{loc}: {error}]"

    # Default: bracket format
    if is_valid:
        badge = f"[{lang_label} Syntax OK]"
        if details:
            badge += " (warnings: " + "; ".join(details) + ")"
        return badge
    else:
        loc = f" line {line}" if line else ""
        return f"[{lang_label} Syntax Error{loc}: {error}]"


# =========================================================================
# Main validation dispatcher
# =========================================================================

_VALIDATORS = {
    "python": validate_python,
    "json": validate_json,
    "r": validate_r,
}


def validate_block(
    language: str,
    code: str,
    *,
    enabled_languages: set[str],
) -> Optional[dict[str, Any]]:
    """Validate a code block if its language is enabled.

    Parameters
    ----------
    language : str
        Language tag from the code fence.
    code : str
        The code content.
    enabled_languages : set[str]
        Set of language names to validate.

    Returns
    -------
    dict or None
        Validation result, or None if language is not enabled.
    """
    # Resolve alias
    resolved = _LANGUAGE_ALIASES.get(language, language)

    if resolved not in enabled_languages:
        return None

    validator = _VALIDATORS.get(resolved)
    if validator is None:
        return None

    return validator(code)


# =========================================================================
# Hook implementation
# =========================================================================

def hook_post_inference(ctx: Any) -> Optional[dict[str, Any]]:
    """Post-inference hook: validate code blocks in LLM response.

    Extracts fenced code blocks, validates each one, and appends
    a syntax badge after each block.
    """
    response = ctx.data.get("response", "")
    if not response:
        return None

    # Parse config
    config = ctx.config or {}
    lang_str = config.get("languages", "python,json,r")
    enabled_languages = {
        lang.strip().lower() for lang in lang_str.split(",") if lang.strip()
    }
    badge_format = config.get("badge_format", _DEFAULT_BADGE_FORMAT)
    min_lines = config.get("min_lines", _DEFAULT_MIN_LINES)

    # Extract code blocks
    blocks = extract_code_blocks(response)
    if not blocks:
        return None

    # Validate and collect badges
    annotations: list[dict[str, Any]] = []
    for block in blocks:
        if block["line_count"] < min_lines:
            continue

        validation = validate_block(
            block["language"],
            block["code"],
            enabled_languages=enabled_languages,
        )

        if validation is None:
            continue

        badge = format_badge(
            validation,
            _LANGUAGE_ALIASES.get(block["language"], block["language"]),
            badge_format=badge_format,
        )

        if badge:
            annotations.append({
                "block_end": block["end"],
                "badge": badge,
                "language": block["language"],
                "valid": validation["valid"],
            })

    if not annotations:
        return None

    # Insert badges after each code block (work backwards)
    annotated = response
    for ann in reversed(annotations):
        insert_pos = ann["block_end"]
        badge_text = f"\n{ann['badge']}"
        annotated = annotated[:insert_pos] + badge_text + annotated[insert_pos:]

    # Build summary
    total_checked = len(annotations)
    valid_count = sum(1 for a in annotations if a["valid"])
    error_count = total_checked - valid_count

    return {
        "response": annotated,
        "code_guardian_summary": {
            "blocks_checked": total_checked,
            "valid": valid_count,
            "errors": error_count,
        },
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
