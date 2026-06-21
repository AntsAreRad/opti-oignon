"""
Code formatter plugin for Opti-Oignon.

Formats code blocks in LLM responses. Supports Python (AST-based
validation + indentation normalization) and JSON (pretty-printing).
"""

import ast
import json
import re
import textwrap
from typing import Any

__plugin_name__: str = "code-formatter"
__plugin_version__: str = "1.0.0"

# Configuration defaults
_PYTHON_INDENT = 4
_JSON_INDENT = 2
_MAX_CODE_LENGTH = 50000

# Regex to match fenced code blocks in markdown
_CODE_BLOCK_RE = re.compile(
    r"```(\w*)\n(.*?)```",
    re.DOTALL,
)


class FormatError(Exception):
    """Raised when code formatting fails."""


def format_python(code: str, indent: int = _PYTHON_INDENT) -> str:
    """Format Python code via AST round-trip and indentation normalization.

    Validates syntax first; if invalid, returns original code unchanged.
    """
    code = code.rstrip()
    if len(code) > _MAX_CODE_LENGTH:
        return code

    # Dedent first
    dedented = textwrap.dedent(code)

    # Validate syntax
    try:
        ast.parse(dedented)
    except SyntaxError:
        # Cannot parse: return original
        return code

    # Normalize indentation: replace tabs with spaces
    lines = dedented.splitlines()
    normalized: list[str] = []
    for line in lines:
        # Replace leading tabs with indent spaces
        stripped = line.lstrip()
        if not stripped:
            normalized.append("")
            continue
        leading = line[: len(line) - len(stripped)]
        leading = leading.replace("\t", " " * indent)
        # Normalize leading spaces to multiples of indent
        space_count = len(leading)
        level = space_count // indent
        normalized.append(" " * (level * indent) + stripped)

    result = "\n".join(normalized)

    # Final validation
    try:
        ast.parse(result)
    except SyntaxError:
        return code

    return result


def format_json(code: str, indent: int = _JSON_INDENT) -> str:
    """Pretty-print JSON code."""
    code = code.strip()
    if len(code) > _MAX_CODE_LENGTH:
        return code

    try:
        parsed = json.loads(code)
        return json.dumps(parsed, indent=indent, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return code


def format_code_block(language: str, code: str) -> str:
    """Format a code block based on its language tag."""
    lang = language.lower().strip()

    if lang in ("python", "py", "python3"):
        return format_python(code)
    elif lang in ("json", "jsonl"):
        return format_json(code)
    else:
        # No formatting for unknown languages
        return code


def format_response_blocks(text: str) -> tuple[str, int]:
    """Find and format all code blocks in a response text.

    Returns (formatted_text, count_of_blocks_formatted).
    """
    count = 0

    def _replace(match: re.Match) -> str:
        nonlocal count
        lang = match.group(1) or ""
        code = match.group(2)
        if not lang:
            return match.group(0)
        formatted = format_code_block(lang, code)
        if formatted != code:
            count += 1
        return f"```{lang}\n{formatted}```"

    result = _CODE_BLOCK_RE.sub(_replace, text)
    return result, count


# =========================================================================
# Hook implementations
# =========================================================================

def hook_post_inference(ctx: Any) -> dict[str, Any] | None:
    """Format code blocks in LLM response text."""
    response = ctx.data.get("response", "")
    if not response or "```" not in response:
        return None

    formatted, count = format_response_blocks(response)
    if count > 0:
        return {
            "response": formatted,
            "code_blocks_formatted": count,
        }
    return None


def hook_tool_call(ctx: Any) -> dict[str, Any] | None:
    """Handle direct format requests via tool_call.

    Expects ctx.data:
        tool_name: "format_code" or "code_formatter"
        code: str
        language: str (optional, default "python")
    """
    tool_name = ctx.data.get("tool_name", "")
    if tool_name not in ("format_code", "code_formatter"):
        return None

    code = ctx.data.get("code", "")
    language = ctx.data.get("language", "python")

    if not code:
        return {"result": None, "error": "No code provided"}

    formatted = format_code_block(language, code)
    return {
        "result": formatted,
        "language": language,
        "changed": formatted != code,
        "error": None,
    }


HOOKS = {
    "post_inference": hook_post_inference,
    "tool_call": hook_tool_call,
}


def init() -> None:
    """Plugin initialization."""
    pass


def shutdown() -> None:
    """Plugin shutdown."""
    pass
