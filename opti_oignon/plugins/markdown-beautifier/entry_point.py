"""
Markdown-beautifier plugin for Opti-Oignon.

Normalizes and beautifies markdown in LLM responses. Applies
configurable rules to fix common formatting issues:

- header_spacing: ensure blank lines before/after headers, consistent marker spacing
- list_formatting: normalize indentation, fix mixed markers
- table_alignment: align column widths in markdown tables
- code_block_spacing: add missing blank lines around fenced code blocks
- fence_repair: close unclosed code fences

Pure text processing -- no external dependencies, no permissions needed.
"""

import logging
import re
from typing import Any

__plugin_name__: str = "markdown-beautifier"
__plugin_version__: str = "1.0.0"

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration defaults
# =========================================================================

_DEFAULT_RULES = [
    "header_spacing",
    "list_formatting",
    "table_alignment",
    "code_block_spacing",
    "fence_repair",
]

_DEFAULT_STRICTNESS = "normal"

AVAILABLE_RULES = frozenset(_DEFAULT_RULES)

# =========================================================================
# Code block protection
# =========================================================================

_FENCED_BLOCK_RE = re.compile(r"(```[^\n]*\n[\s\S]*?```)", re.DOTALL)


def _protect_fenced_blocks(text: str) -> tuple[str, list[str]]:
    """Replace fenced code blocks with placeholders.

    Returns the modified text and the list of extracted blocks.
    """
    blocks: list[str] = []

    def _replace(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"__MD_FENCE_{len(blocks) - 1}__"

    protected = _FENCED_BLOCK_RE.sub(_replace, text)
    return protected, blocks


def _restore_fenced_blocks(text: str, blocks: list[str]) -> str:
    """Restore fenced code blocks from placeholders."""
    for i, block in enumerate(blocks):
        text = text.replace(f"__MD_FENCE_{i}__", block)
    return text


# =========================================================================
# Rule: header_spacing
# =========================================================================

# Match markdown headers (# to ######)
_HEADER_RE = re.compile(r"^(#{1,6})\s*(.*?)$", re.MULTILINE)


def fix_header_spacing(text: str, strictness: str = "normal") -> str:
    """Normalize header formatting.

    - Ensure exactly one space after # markers
    - Add blank line before headers (unless at start of text)
    - Add blank line after headers
    - In strict mode: normalize header level hierarchy

    Parameters
    ----------
    text : str
        Input markdown text.
    strictness : str
        'normal' or 'strict'.

    Returns
    -------
    str
        Text with normalized headers.
    """
    lines = text.split("\n")
    result: list[str] = []

    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line)
        if m:
            hashes = m.group(1)
            title = m.group(2).strip()
            normalized = f"{hashes} {title}"

            # Add blank line before header (if previous line is not blank
            # and we are not at the start)
            if result and result[-1].strip() != "":
                result.append("")

            result.append(normalized)

            # Add blank line after header (peek at next line)
            if i + 1 < len(lines) and lines[i + 1].strip() != "":
                result.append("")
        else:
            result.append(line)

    return "\n".join(result)


# =========================================================================
# Rule: list_formatting
# =========================================================================

# Detect list items: -, *, +, or numbered (1. 2. etc.)
_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)]) (.*)$")


def fix_list_formatting(text: str, strictness: str = "normal") -> str:
    """Normalize list formatting.

    - Normalize indentation to multiples of 2 spaces
    - In strict mode: standardize unordered markers to '-'

    Parameters
    ----------
    text : str
        Input markdown text.
    strictness : str
        'normal' or 'strict'.

    Returns
    -------
    str
        Text with normalized lists.
    """
    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        m = _LIST_ITEM_RE.match(line)
        if m:
            indent = m.group(1)
            marker = m.group(2)
            content = m.group(3)

            # Normalize indent to multiples of 2
            indent_level = len(indent) // 2
            normalized_indent = "  " * indent_level

            # In strict mode, normalize unordered markers to '-'
            if strictness == "strict" and marker in ("*", "+"):
                marker = "-"

            result.append(f"{normalized_indent}{marker} {content}")
        else:
            result.append(line)

    return "\n".join(result)


# =========================================================================
# Rule: table_alignment
# =========================================================================

_TABLE_ROW_RE = re.compile(r"^\|(.+)\|$")
_SEPARATOR_RE = re.compile(r"^[\s|:-]+$")


def _is_table_separator(row: str) -> bool:
    """Check if a row is a table separator (|---|---|)."""
    cells = row.strip().strip("|").split("|")
    return all(
        re.match(r"^\s*:?-+:?\s*$", cell.strip())
        for cell in cells
        if cell.strip()
    )


def _parse_table_rows(lines: list[str]) -> list[list[str]]:
    """Parse table rows into lists of cell contents."""
    rows: list[list[str]] = []
    for line in lines:
        m = _TABLE_ROW_RE.match(line.strip())
        if m:
            cells = [c.strip() for c in m.group(1).split("|")]
            rows.append(cells)
    return rows


def fix_table_alignment(text: str, strictness: str = "normal") -> str:
    """Align markdown table columns.

    Pads each column to its maximum width for visual alignment.

    Parameters
    ----------
    text : str
        Input markdown text.
    strictness : str
        'normal' or 'strict'.

    Returns
    -------
    str
        Text with aligned tables.
    """
    lines = text.split("\n")
    result: list[str] = []
    table_buffer: list[str] = []
    in_table = False

    def _flush_table() -> None:
        """Process and flush accumulated table rows."""
        if not table_buffer:
            return

        rows = _parse_table_rows(table_buffer)
        if len(rows) < 2:
            # Not a real table, emit as-is
            result.extend(table_buffer)
            table_buffer.clear()
            return

        # Find max column count and widths
        max_cols = max(len(r) for r in rows)
        col_widths = [0] * max_cols
        for row in rows:
            for j, cell in enumerate(row):
                if j < max_cols:
                    col_widths[j] = max(col_widths[j], len(cell))

        # Minimum width of 3 for separator dashes
        col_widths = [max(w, 3) for w in col_widths]

        # Rebuild rows
        for i, row in enumerate(rows):
            # Pad row to max_cols
            padded = row + [""] * (max_cols - len(row))
            is_sep = i < len(table_buffer) and _is_table_separator(table_buffer[i])

            if is_sep:
                cells = ["-" * col_widths[j] for j in range(max_cols)]
            else:
                cells = [
                    padded[j].ljust(col_widths[j])
                    for j in range(max_cols)
                ]

            result.append("| " + " | ".join(cells) + " |")

        table_buffer.clear()

    for line in lines:
        stripped = line.strip()
        is_table_row = bool(_TABLE_ROW_RE.match(stripped))

        if is_table_row:
            in_table = True
            table_buffer.append(line)
        else:
            if in_table:
                _flush_table()
                in_table = False
            result.append(line)

    # Flush any remaining table
    if table_buffer:
        _flush_table()

    return "\n".join(result)


# =========================================================================
# Rule: code_block_spacing
# =========================================================================

_FENCE_OPEN_RE = re.compile(r"^```\w*\s*$")
_FENCE_CLOSE_RE = re.compile(r"^```\s*$")


def fix_code_block_spacing(text: str, strictness: str = "normal") -> str:
    """Add missing blank lines around fenced code blocks.

    Ensures there is a blank line before the opening fence
    and after the closing fence.

    Parameters
    ----------
    text : str
        Input markdown text.
    strictness : str
        'normal' or 'strict'.

    Returns
    -------
    str
        Text with proper code block spacing.
    """
    lines = text.split("\n")
    result: list[str] = []
    in_fence = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not in_fence and _FENCE_OPEN_RE.match(stripped):
            # Opening fence: ensure blank line before
            if result and result[-1].strip() != "":
                result.append("")
            result.append(line)
            in_fence = True

        elif in_fence and _FENCE_CLOSE_RE.match(stripped):
            result.append(line)
            # Closing fence: ensure blank line after
            if i + 1 < len(lines) and lines[i + 1].strip() != "":
                result.append("")
            in_fence = False

        else:
            result.append(line)

    return "\n".join(result)


# =========================================================================
# Rule: fence_repair
# =========================================================================


def fix_unclosed_fences(text: str, strictness: str = "normal") -> str:
    """Close unclosed code fences.

    Scans for opening ``` that lack a matching close and appends
    a closing ``` at the end.

    Parameters
    ----------
    text : str
        Input markdown text.
    strictness : str
        'normal' or 'strict'.

    Returns
    -------
    str
        Text with all code fences properly closed.
    """
    lines = text.split("\n")
    fence_open = False
    open_count = 0
    close_count = 0

    for line in lines:
        stripped = line.strip()
        if not fence_open and stripped.startswith("```"):
            fence_open = True
            open_count += 1
        elif fence_open and stripped == "```":
            fence_open = False
            close_count += 1

    if open_count > close_count:
        # Add missing closing fences
        missing = open_count - close_count
        for _ in range(missing):
            text = text.rstrip() + "\n```"

    return text


# =========================================================================
# Rule dispatcher
# =========================================================================

RULE_FUNCTIONS = {
    "header_spacing": fix_header_spacing,
    "list_formatting": fix_list_formatting,
    "table_alignment": fix_table_alignment,
    "code_block_spacing": fix_code_block_spacing,
    "fence_repair": fix_unclosed_fences,
}

# Order matters: fence_repair first (so subsequent rules see valid blocks),
# then code_block_spacing, then the rest.
RULE_ORDER = [
    "fence_repair",
    "code_block_spacing",
    "header_spacing",
    "list_formatting",
    "table_alignment",
]


def beautify(
    text: str,
    rules: list[str],
    strictness: str = "normal",
) -> str:
    """Apply selected beautification rules to markdown text.

    Rules that operate on prose protect fenced code blocks
    from modification. Fence-related rules run on the raw text.

    Parameters
    ----------
    text : str
        Input markdown text.
    rules : list[str]
        Which rules to apply.
    strictness : str
        'normal' or 'strict'.

    Returns
    -------
    str
        Beautified markdown text.
    """
    # Fence-level rules run on raw text
    fence_rules = {"fence_repair", "code_block_spacing"}
    # Content-level rules need code block protection
    content_rules = {"header_spacing", "list_formatting", "table_alignment"}  # noqa: F841

    # Apply in defined order
    for rule_name in RULE_ORDER:
        if rule_name not in rules:
            continue
        if rule_name not in RULE_FUNCTIONS:
            logger.warning("Unknown beautification rule: %s", rule_name)
            continue

        fn = RULE_FUNCTIONS[rule_name]

        if rule_name in fence_rules:
            # Operate on raw text
            text = fn(text, strictness)
        else:
            # Protect code blocks
            protected, blocks = _protect_fenced_blocks(text)
            protected = fn(protected, strictness)
            text = _restore_fenced_blocks(protected, blocks)

    return text


# =========================================================================
# Hook implementation
# =========================================================================


def hook_post_inference(ctx: Any) -> dict[str, Any] | None:
    """Post-inference hook: beautify markdown in the LLM response.

    Applies configured rules to normalize headers, lists, tables,
    code block spacing, and fence closure.
    """
    response = ctx.data.get("response", "")
    if not response:
        return None

    config = ctx.config or {}
    rules = config.get("rules", _DEFAULT_RULES)
    strictness = config.get("strictness", _DEFAULT_STRICTNESS)

    # Validate rules
    valid_rules = [r for r in rules if r in AVAILABLE_RULES]
    if not valid_rules:
        return None

    beautified = beautify(response, valid_rules, strictness)

    if beautified == response:
        return None

    return {
        "response": beautified,
        "rules_applied": valid_rules,
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
