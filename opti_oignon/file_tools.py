#!/usr/bin/env python3
"""
FILE TOOLS - OPTI-OIGNON v1.7.5 (S73)
=======================================

Four sandboxed filesystem tools for LLM agentic execution:
- sandbox_bash: Execute bash commands inside the sandbox
- sandbox_view: Read file contents within the sandbox (with line range)
- sandbox_create_file: Create or overwrite a file within the sandbox
- sandbox_str_replace: Find-and-replace within a sandbox file

All tools enforce strict path validation: no path traversal, no symlink
escape, no access outside the sandbox workspace. Every operation goes
through the SandboxManager, which provides either bwrap kernel isolation
or (degraded) tempdir isolation with command blocklist.

Each tool produces a ToolDefinition compatible with the existing
tool_registry (S44) and tool_executor (S44) interfaces.

Author: Leon
"""

import difflib
import logging
import os

logger = logging.getLogger(__name__)

# Conditional import of sandbox manager
try:
    from opti_oignon.sandbox_manager import (
        SANDBOX_AVAILABLE,
        SandboxManager,
        validate_sandbox_path,
    )
    from opti_oignon.sandbox_manager import (
        sandbox_manager as _default_sandbox_manager,
    )
except ImportError:
    SANDBOX_AVAILABLE = False
    _default_sandbox_manager = None
    SandboxManager = None
    validate_sandbox_path = None

# Conditional import of tool registry types
try:
    from opti_oignon.tool_registry import ToolDefinition, ToolParam
    TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    TOOL_REGISTRY_AVAILABLE = False
    ToolDefinition = None
    ToolParam = None

# Maximum file size readable via sandbox_view (1 MB)
MAX_VIEW_SIZE = 1024 * 1024

# Maximum file size writable via sandbox_create_file (2 MB)
MAX_CREATE_SIZE = 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helper: resolve sandbox path safely
# ---------------------------------------------------------------------------

def _resolve_path(
    sandbox_mgr: SandboxManager,
    session_id: str,
    path: str,
) -> tuple[bool, str, str]:
    """Resolve a path within a sandbox session.

    Returns (success, resolved_absolute_path, error_message).
    """
    workspace = sandbox_mgr.get_workspace_path(session_id)
    if workspace is None:
        return False, "", f"Session not found or inactive: {session_id}"

    valid, resolved, err = validate_sandbox_path(workspace, path)
    if not valid:
        return False, "", f"Path rejected: {err}"

    return True, resolved, ""


# ---------------------------------------------------------------------------
# Tool: sandbox_bash
# ---------------------------------------------------------------------------

def _handle_sandbox_bash(
    session_id: str,
    command: str,
    timeout: int = 30,
    _sandbox_manager: SandboxManager | None = None,
) -> str:
    """Execute a bash command inside a sandbox session.

    The command runs in the sandbox workspace with full isolation
    (bwrap) or restricted environment (tempdir fallback). The command
    blocklist is applied regardless of backend.

    Args:
        session_id: Active sandbox session ID.
        command: Bash command to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        Formatted string with stdout, stderr, and return code.
    """
    mgr = _sandbox_manager or _default_sandbox_manager
    if mgr is None:
        return "Error: Sandbox manager not available"

    try:
        result = mgr.execute_command(session_id, command, timeout=timeout)
    except ValueError as exc:
        return f"Error: {exc}"

    if result.blocked:
        return (
            f"BLOCKED: Command rejected for security reasons.\n"
            f"Reason: {result.block_reason}\n"
            f"The command was not executed."
        )

    parts = []
    if result.timed_out:
        parts.append(f"TIMEOUT: Command timed out after {timeout}s")

    if result.stdout:
        output = result.stdout
        if result.truncated_stdout:
            output += "\n[stdout truncated]"
        parts.append(output)

    if result.stderr:
        stderr = result.stderr
        if result.truncated_stderr:
            stderr += "\n[stderr truncated]"
        parts.append(f"STDERR:\n{stderr}")

    if not parts:
        status = "success" if result.return_code == 0 else "failed"
        parts.append(
            f"Command {status} (return code: {result.return_code})"
        )
    elif result.return_code != 0 and not result.timed_out:
        parts.append(f"[exit code: {result.return_code}]")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool: sandbox_view
# ---------------------------------------------------------------------------

def _handle_sandbox_view(
    session_id: str,
    path: str,
    start_line: int = 0,
    end_line: int = 0,
    _sandbox_manager: SandboxManager | None = None,
) -> str:
    """Read file contents or list a directory within the sandbox.

    For files: returns content, optionally filtered to a line range.
    For directories: returns a listing of entries (files and subdirs).

    Args:
        session_id: Active sandbox session ID.
        path: Path relative to workspace or /workspace/ prefix.
        start_line: First line to show (1-indexed, 0 = from start).
        end_line: Last line to show (0 = to end, -1 = to end).

    Returns:
        File content with line numbers, or directory listing.
    """
    mgr = _sandbox_manager or _default_sandbox_manager
    if mgr is None:
        return "Error: Sandbox manager not available"

    ok, resolved, err = _resolve_path(mgr, session_id, path)
    if not ok:
        return f"Error: {err}"

    # Directory listing
    if os.path.isdir(resolved):
        return _list_directory(resolved, mgr.get_workspace_path(session_id))

    # File reading
    if not os.path.isfile(resolved):
        return f"Error: Path not found: {path}"

    # Size check
    try:
        size = os.path.getsize(resolved)
    except OSError as exc:
        return f"Error reading file: {exc}"

    if size > MAX_VIEW_SIZE:
        return (
            f"Error: File too large ({size:,} bytes, "
            f"max {MAX_VIEW_SIZE:,} bytes)"
        )

    try:
        with open(resolved, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return f"Error reading file: {exc}"

    total = len(lines)

    # Apply line range
    if start_line > 0 or end_line != 0:
        s = max(start_line - 1, 0)  # Convert to 0-indexed
        if end_line == -1 or end_line == 0:
            e = total
        else:
            e = min(end_line, total)
        lines = lines[s:e]
        line_offset = s
    else:
        line_offset = 0

    # Format with line numbers
    numbered = []
    for i, line in enumerate(lines):
        lineno = line_offset + i + 1
        numbered.append(f"{lineno:>6}\t{line.rstrip()}")

    header = f"File: {path} ({total} lines total)"
    if start_line > 0 or end_line != 0:
        show_start = line_offset + 1
        show_end = line_offset + len(lines)
        header += f" [showing lines {show_start}-{show_end}]"

    return header + "\n" + "\n".join(numbered)


def _list_directory(dir_path: str, workspace_root: str) -> str:
    """List directory contents within sandbox."""
    try:
        entries = sorted(os.listdir(dir_path))
    except OSError as exc:
        return f"Error listing directory: {exc}"

    if not entries:
        rel = os.path.relpath(dir_path, workspace_root)
        return f"Empty directory: {rel}"

    lines = []
    for entry in entries[:200]:  # Cap at 200 entries
        full = os.path.join(dir_path, entry)
        if os.path.isdir(full):
            lines.append(f"  [DIR]  {entry}/")
        elif os.path.islink(full):
            lines.append(f"  [LINK] {entry}")
        else:
            try:
                size = os.path.getsize(full)
                lines.append(f"  [FILE] {entry} ({size:,} bytes)")
            except OSError:
                lines.append(f"  [FILE] {entry}")

    rel = os.path.relpath(dir_path, workspace_root)
    header = f"Directory: /workspace/{rel} ({len(entries)} entries)"
    if len(entries) > 200:
        header += " [truncated to 200]"
    return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: sandbox_create_file
# ---------------------------------------------------------------------------

def _handle_sandbox_create_file(
    session_id: str,
    path: str,
    content: str,
    _sandbox_manager: SandboxManager | None = None,
) -> str:
    """Create or overwrite a file within the sandbox workspace.

    Parent directories are created automatically. The path must stay
    within the sandbox workspace.

    Args:
        session_id: Active sandbox session ID.
        path: File path relative to workspace or /workspace/ prefix.
        content: Content to write to the file.

    Returns:
        Success message with file path and size, or error.
    """
    mgr = _sandbox_manager or _default_sandbox_manager
    if mgr is None:
        return "Error: Sandbox manager not available"

    ok, resolved, err = _resolve_path(mgr, session_id, path)
    if not ok:
        return f"Error: {err}"

    # Size check
    content_bytes = len(content.encode("utf-8"))
    if content_bytes > MAX_CREATE_SIZE:
        return (
            f"Error: Content too large ({content_bytes:,} bytes, "
            f"max {MAX_CREATE_SIZE:,} bytes)"
        )

    # Create parent directories
    parent = os.path.dirname(resolved)
    if parent:
        workspace = mgr.get_workspace_path(session_id)
        # Validate parent path is also within workspace
        parent_valid, _, parent_err = validate_sandbox_path(
            workspace, os.path.relpath(parent, workspace)
        )
        if not parent_valid:
            return f"Error: Parent directory path rejected: {parent_err}"
        os.makedirs(parent, exist_ok=True)

    try:
        with open(resolved, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as exc:
        return f"Error writing file: {exc}"

    # S81: Register created file for write-then-execute detection
    try:
        mgr.register_created_file(path, content)
    except Exception:
        pass  # Non-critical: validator tracking is defense-in-depth

    return f"File created: {path} ({content_bytes:,} bytes)"


# ---------------------------------------------------------------------------
# Tool: sandbox_str_replace
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# S229 (AGT_SPEC 6.4): the conservative str_replace recovery chain.
#
# Exact match stays first. On a MISS (zero exact occurrences) the three
# conservative replacers run in order -- line-trimmed, whitespace-normalized,
# indentation-flexible -- and a stage applies ONLY when it yields exactly one
# candidate line window: zero candidates continues down the chain; two or
# more fails the whole call ("old_str matched K regions after normalization;
# make it unique"). The multi-exact-match path (count > 1) and the rejected
# replacers of the spec's Section 4 map (BlockAnchor, ContextAware,
# MultiOccurrence, TrimmedBoundary, EscapeNormalized) are untouched: nothing
# rejected re-enters. Candidates are located as line windows and the file is
# rebuilt by line INDICES, never by str.replace, so an identical region
# elsewhere can never be touched by accident. On total failure the
# structured miss hint names up to three closest lines by difflib
# similarity, so the model's next attempt is informed rather than blind.
# ---------------------------------------------------------------------------

_RECOVERY_HINT_LINES = 3
_RECOVERY_HINT_LINE_CLIP = 80


def _window_candidates(content_lines, old_lines, line_match):
    """Start indices of every window where ``line_match`` holds per line."""
    n = len(old_lines)
    if n == 0 or n > len(content_lines):
        return []
    starts = []
    for i in range(len(content_lines) - n + 1):
        if all(line_match(content_lines[i + j], old_lines[j]) for j in range(n)):
            starts.append(i)
    return starts


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _uniform_indent_delta(window, old_lines):
    """The single uniform indent adjustment between a window and old_str.

    Blank lines are exempt. Returns ``None`` when the difference is not one
    uniform delta, ``("", "")`` when indents already match exactly (the
    line-trimmed case), ``("+", prefix)`` when every non-blank window line
    is the old line indented by ``prefix``, and ``("-", prefix)`` when every
    non-blank window line is the old line OUTDENTED by ``prefix``.
    """
    delta = None
    for win_line, old_line in zip(window, old_lines):
        if not old_line.strip():
            continue
        win_indent = _leading_ws(win_line)
        old_indent = _leading_ws(old_line)
        if win_indent == old_indent:
            current = ("", "")
        elif win_indent.endswith(old_indent):
            current = ("+", win_indent[: len(win_indent) - len(old_indent)])
        elif old_indent.endswith(win_indent):
            current = ("-", old_indent[: len(old_indent) - len(win_indent)])
        else:
            return None
        if delta is None:
            delta = current
        elif delta != current:
            return None
    return delta if delta is not None else ("", "")


def _apply_indent_delta(new_str: str, delta) -> str:
    """Re-indent ``new_str`` by the window's uniform delta (blank-exempt)."""
    sign, prefix = delta
    if not prefix:
        return new_str
    out = []
    for line in new_str.split("\n"):
        if not line.strip():
            out.append(line)
        elif sign == "+":
            out.append(prefix + line)
        else:
            out.append(line[len(prefix):] if line.startswith(prefix) else line)
    return "\n".join(out)


def _find_recovery_window(content_lines, old_lines):
    """Run the 6.4 chain; the first stage with EXACTLY ONE candidate wins.

    Returns ``("hit", start, strategy, delta)``, ``("multi", count, None,
    None)`` or ``("miss", 0, None, None)``. ``delta`` is the uniform indent
    adjustment of the indentation-flexible strategy (None for the other
    two, whose replacement lands verbatim).
    """
    trimmed = _window_candidates(
        content_lines, old_lines, lambda a, b: a.strip() == b.strip()
    )
    if len(trimmed) > 1:
        return ("multi", len(trimmed), None, None)
    if len(trimmed) == 1:
        start = trimmed[0]
        delta = _uniform_indent_delta(
            content_lines[start : start + len(old_lines)], old_lines
        )
        if delta is not None and delta[1]:
            return ("hit", start, "indentation-flexible", delta)
        return ("hit", start, "line-trimmed", None)

    def _norm(s: str) -> str:
        return " ".join(s.split())

    normalized = _window_candidates(
        content_lines, old_lines, lambda a, b: _norm(a) == _norm(b)
    )
    if len(normalized) > 1:
        return ("multi", len(normalized), None, None)
    if len(normalized) == 1:
        return ("hit", normalized[0], "whitespace-normalized", None)
    return ("miss", 0, None, None)


def _attempt_recovery(resolved: str, content: str, old_str: str, new_str: str):
    """Run the recovery chain and, on a hit, perform the edit.

    Returns the final message (the strategy-naming success or the K-regions
    error) or ``None`` on a clean miss -- the caller then appends the
    structured hint to the unchanged not-found message. The recovered edit
    flows through the same file write as any edit (and, at the session
    layer, the same diagnostics-after-write and the same copy-out review).
    """
    content_lines = content.split("\n")
    old_lines = old_str.split("\n")
    if len(old_lines) > 1 and old_lines[-1] == "":
        old_lines = old_lines[:-1]
    if not old_lines or not any(line.strip() for line in old_lines):
        return None
    kind, value, strategy, delta = _find_recovery_window(content_lines, old_lines)
    if kind == "multi":
        return (
            f"Error: old_str matched {value} regions after normalization; "
            f"make it unique"
        )
    if kind != "hit":
        return None
    start = value
    replacement = new_str if delta is None else _apply_indent_delta(new_str, delta)
    repl_lines = replacement.split("\n")
    if len(repl_lines) > 1 and repl_lines[-1] == "":
        repl_lines = repl_lines[:-1]
    if new_str == "":
        repl_lines = []
    new_content = "\n".join(
        content_lines[:start] + repl_lines + content_lines[start + len(old_lines):]
    )
    try:
        with open(resolved, "w", encoding="utf-8") as fh:
            fh.write(new_content)
    except OSError as exc:
        return f"Error writing file: {exc}"
    return f"Replaced (matched via {strategy} normalization)."


def _miss_hint(content_lines, old_str: str) -> str:
    """Up to three closest lines by difflib similarity, with line numbers.

    The probe is the first non-empty line of ``old_str``; ties resolve by
    line number, and hint lines are clipped so the hint stays one line.
    """
    probe = next(
        (line.strip() for line in old_str.split("\n") if line.strip()),
        old_str.strip(),
    )
    if not probe:
        return ""
    scored = []
    for lineno, line in enumerate(content_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        ratio = difflib.SequenceMatcher(None, probe, stripped).ratio()
        scored.append((-ratio, lineno, stripped))
    if not scored:
        return ""
    scored.sort()
    parts = []
    for _neg_ratio, lineno, text in scored[:_RECOVERY_HINT_LINES]:
        if len(text) > _RECOVERY_HINT_LINE_CLIP:
            text = text[:_RECOVERY_HINT_LINE_CLIP] + "..."
        parts.append(f"{lineno}: {text}")
    return "closest lines: " + "; ".join(parts)


def _handle_sandbox_str_replace(
    session_id: str,
    path: str,
    old_str: str,
    new_str: str = "",
    _sandbox_manager: SandboxManager | None = None,
) -> str:
    """Find and replace a unique string in a file within the sandbox.

    The old_str must appear exactly once in the file (safety: prevents
    accidental mass replacements). Use new_str="" to delete the match.

    Args:
        session_id: Active sandbox session ID.
        path: File path relative to workspace or /workspace/ prefix.
        old_str: String to find (must be unique in the file).
        new_str: Replacement string (empty to delete).

    Returns:
        Success message or error (including if old_str not found
        or found multiple times).
    """
    mgr = _sandbox_manager or _default_sandbox_manager
    if mgr is None:
        return "Error: Sandbox manager not available"

    ok, resolved, err = _resolve_path(mgr, session_id, path)
    if not ok:
        return f"Error: {err}"

    if not os.path.isfile(resolved):
        return f"Error: File not found: {path}"

    if not old_str:
        return "Error: old_str cannot be empty"

    # Read file
    try:
        with open(resolved, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError as exc:
        return f"Error reading file: {exc}"

    # Count occurrences
    count = content.count(old_str)
    if count == 0:
        # S229 (AGT_SPEC 6.4): exact match failed -- run the conservative
        # recovery chain; on a clean miss, append the structured hint to the
        # unchanged not-found message so the next attempt is informed.
        recovered = _attempt_recovery(resolved, content, old_str, new_str)
        if recovered is not None:
            return recovered
        hint = _miss_hint(content.split("\n"), old_str)
        not_found = (
            f"Error: String not found in {path}. "
            f"The file has {len(content):,} characters. "
            f"Make sure the search string matches exactly "
            f"(including whitespace and indentation)."
        )
        return not_found + ("\n" + hint if hint else "")
    if count > 1:
        return (
            f"Error: String found {count} times in {path}. "
            f"The search string must be unique. "
            f"Add more surrounding context to make it unique."
        )

    # Perform replacement
    new_content = content.replace(old_str, new_str, 1)

    # Write back
    try:
        with open(resolved, "w", encoding="utf-8") as fh:
            fh.write(new_content)
    except OSError as exc:
        return f"Error writing file: {exc}"

    if new_str:
        return (
            f"Replacement successful in {path}: "
            f"replaced {len(old_str)} chars with {len(new_str)} chars"
        )
    return f"Deletion successful in {path}: removed {len(old_str)} chars"


# ---------------------------------------------------------------------------
# ToolDefinition factories
# ---------------------------------------------------------------------------

def get_sandbox_bash_definition(
    sandbox_mgr: SandboxManager | None = None,
) -> "ToolDefinition":
    """Create a ToolDefinition for the sandbox bash tool."""
    mgr = sandbox_mgr or _default_sandbox_manager

    def handler(session_id: str, command: str, timeout: int = 30) -> str:
        return _handle_sandbox_bash(
            session_id, command, timeout, _sandbox_manager=mgr
        )

    return ToolDefinition(
        name="sandbox_bash",
        description=(
            "Execute a bash command inside the isolated sandbox. "
            "The command runs in /workspace/ with no access to the "
            "host filesystem or network. Dangerous commands are blocked."
        ),
        parameters={
            "session_id": ToolParam(
                name="session_id",
                type="string",
                description="Active sandbox session ID",
                required=True,
            ),
            "command": ToolParam(
                name="command",
                type="string",
                description="Bash command to execute",
                required=True,
            ),
            "timeout": ToolParam(
                name="timeout",
                type="int",
                description="Maximum execution time in seconds",
                required=False,
                default=30,
            ),
        },
        handler=handler,
        requires=["sandbox"],
        enabled=SANDBOX_AVAILABLE and mgr is not None,
    )


def get_sandbox_view_definition(
    sandbox_mgr: SandboxManager | None = None,
) -> "ToolDefinition":
    """Create a ToolDefinition for the sandbox view tool."""
    mgr = sandbox_mgr or _default_sandbox_manager

    def handler(
        session_id: str,
        path: str,
        start_line: int = 0,
        end_line: int = 0,
    ) -> str:
        return _handle_sandbox_view(
            session_id, path, start_line, end_line,
            _sandbox_manager=mgr,
        )

    return ToolDefinition(
        name="sandbox_view",
        description=(
            "Read file contents or list a directory within the sandbox. "
            "Files are shown with line numbers. Use start_line/end_line "
            "to view a specific range (1-indexed). Directories show a "
            "listing of their entries."
        ),
        parameters={
            "session_id": ToolParam(
                name="session_id",
                type="string",
                description="Active sandbox session ID",
                required=True,
            ),
            "path": ToolParam(
                name="path",
                type="string",
                description=(
                    "File or directory path (relative to workspace "
                    "or /workspace/ prefix)"
                ),
                required=True,
            ),
            "start_line": ToolParam(
                name="start_line",
                type="int",
                description="First line to show (1-indexed, 0 = start)",
                required=False,
                default=0,
            ),
            "end_line": ToolParam(
                name="end_line",
                type="int",
                description="Last line to show (0 or -1 = end)",
                required=False,
                default=0,
            ),
        },
        handler=handler,
        requires=["sandbox"],
        enabled=SANDBOX_AVAILABLE and mgr is not None,
    )


def get_sandbox_create_file_definition(
    sandbox_mgr: SandboxManager | None = None,
) -> "ToolDefinition":
    """Create a ToolDefinition for the sandbox create_file tool."""
    mgr = sandbox_mgr or _default_sandbox_manager

    def handler(session_id: str, path: str, content: str) -> str:
        return _handle_sandbox_create_file(
            session_id, path, content, _sandbox_manager=mgr,
        )

    return ToolDefinition(
        name="sandbox_create_file",
        description=(
            "Create or overwrite a file within the sandbox workspace. "
            "Parent directories are created automatically. "
            "Path must stay within /workspace/."
        ),
        parameters={
            "session_id": ToolParam(
                name="session_id",
                type="string",
                description="Active sandbox session ID",
                required=True,
            ),
            "path": ToolParam(
                name="path",
                type="string",
                description=(
                    "File path (relative to workspace "
                    "or /workspace/ prefix)"
                ),
                required=True,
            ),
            "content": ToolParam(
                name="content",
                type="string",
                description="Content to write to the file",
                required=True,
            ),
        },
        handler=handler,
        requires=["sandbox"],
        enabled=SANDBOX_AVAILABLE and mgr is not None,
    )


def get_sandbox_str_replace_definition(
    sandbox_mgr: SandboxManager | None = None,
) -> "ToolDefinition":
    """Create a ToolDefinition for the sandbox str_replace tool."""
    mgr = sandbox_mgr or _default_sandbox_manager

    def handler(
        session_id: str,
        path: str,
        old_str: str,
        new_str: str = "",
    ) -> str:
        return _handle_sandbox_str_replace(
            session_id, path, old_str, new_str,
            _sandbox_manager=mgr,
        )

    return ToolDefinition(
        name="sandbox_str_replace",
        description=(
            "Find and replace a unique string in a file within the "
            "sandbox. The search string must appear exactly once "
            "(prevents accidental mass replacements). "
            "Use new_str='' to delete the match."
        ),
        parameters={
            "session_id": ToolParam(
                name="session_id",
                type="string",
                description="Active sandbox session ID",
                required=True,
            ),
            "path": ToolParam(
                name="path",
                type="string",
                description="File path within the sandbox",
                required=True,
            ),
            "old_str": ToolParam(
                name="old_str",
                type="string",
                description="String to find (must be unique in file)",
                required=True,
            ),
            "new_str": ToolParam(
                name="new_str",
                type="string",
                description="Replacement string (empty to delete)",
                required=False,
                default="",
            ),
        },
        handler=handler,
        requires=["sandbox"],
        enabled=SANDBOX_AVAILABLE and mgr is not None,
    )


# ---------------------------------------------------------------------------
# Convenience: get all sandbox tool definitions
# ---------------------------------------------------------------------------

def get_all_sandbox_tool_definitions(
    sandbox_mgr: SandboxManager | None = None,
) -> list:
    """Return all 4 sandbox tool definitions.

    Args:
        sandbox_mgr: Optional SandboxManager override (for testing).

    Returns:
        List of ToolDefinition instances.
    """
    return [
        get_sandbox_bash_definition(sandbox_mgr),
        get_sandbox_view_definition(sandbox_mgr),
        get_sandbox_create_file_definition(sandbox_mgr),
        get_sandbox_str_replace_definition(sandbox_mgr),
    ]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

FILE_TOOLS_AVAILABLE = SANDBOX_AVAILABLE and TOOL_REGISTRY_AVAILABLE
