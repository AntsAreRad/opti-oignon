#!/usr/bin/env python3
"""
FILE TOOLS - OPTI-OIGNON v1.7.5
=======================================

Four sandboxed filesystem tools for LLM agentic execution:
- sandbox_bash: Execute bash commands inside the sandbox
- sandbox_view: Read file contents within the sandbox (with line range)
- sandbox_create_file: Create or overwrite a file within the sandbox
- sandbox_str_replace: Find-and-replace within a sandbox file

All tools enforce strict path validation: no path traversal, no symlink
escape, no access outside the sandbox workspace. A checked path is then
reached beneath the workspace one component at a time without following a
link, so a link the sandbox puts on it after the check is refused, never
followed. Every operation goes through the SandboxManager, which provides
either bwrap kernel isolation or (degraded) tempdir isolation with command
blocklist.

Each tool produces a ToolDefinition compatible with the existing
tool_registry and tool_executor interfaces.

Author: Leon
"""

import contextlib
import difflib
import errno
import io
import logging
import os
import stat

logger = logging.getLogger(__name__)

# Conditional import of sandbox manager
try:
    from opti_oignon.sandbox_manager import (
        SANDBOX_AVAILABLE,
        SandboxManager,
        _folder_beneath,
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
    _folder_beneath = None

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
# Helper: hold a tool's session while it works in its workspace
# ---------------------------------------------------------------------------

def _held(
    sandbox_mgr: SandboxManager,
    session_id: str,
    work,
    *,
    writes: bool = False,
) -> str:
    """``work()`` -- a tool's answer -- run with its session in use
    (``SandboxManager.in_use``).

    The idle sweep never takes the session while the tool reads or writes its
    workspace, and the call counts as activity on the session it held when it
    ends, its answer an error or not: the agent calling a tool is the agent
    working. A session unknown, inactive or without its workspace is not
    held: ``work`` answers it as before, and nothing is counted. An explicit
    destroy is not held off: a tool that writes (``writes``) whose session it
    ended meanwhile answers so, whatever its write did -- landed in a
    workspace being removed, or failed for the removal, an answer or an
    exception; a read keeps its own answer.
    """
    if sandbox_mgr is None:
        return work()
    with contextlib.ExitStack() as stack:
        try:
            session = stack.enter_context(sandbox_mgr.in_use(session_id))
        except ValueError:
            return work()
        destroyed = (
            f"Error: Session {session_id} was destroyed while the file "
            f"was written"
        )
        try:
            answer = work()
        except Exception:
            # The destroy removed the workspace under the write, which failed
            # for it: the destroy is the answer, not its trace
            if writes and not session.active:
                logger.debug(
                    "write in %s failed under its destroy", session_id,
                    exc_info=True,
                )
                return destroyed
            raise
        finally:
            # Counted before the session is let go: a sweep in between would
            # find it idle and in use by nothing
            sandbox_mgr.touch_activity(session_id, held=session)
        if writes and not session.active:
            return destroyed
        return answer


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

# A file made where nothing stood is looked for again this many times when
# something else is put at its name in between
_CREATE_TRIES = 3


def _checked_parts(
    sandbox_mgr: SandboxManager,
    session_id: str,
    resolved: str,
) -> tuple[str, list[str]]:
    """The workspace's real path and the components of ``resolved`` -- a path
    ``_resolve_path`` checked and resolved -- beneath it.

    A tool reaches them one at a time without following a link: the links
    inside the workspace were resolved at the check (a venv's ``bin/python``,
    a link the model made), so a link met on the way now was put there after
    the check, and is refused (``OSError``), never followed.
    """
    workspace = sandbox_mgr.get_workspace_path(session_id)
    if workspace is None:
        raise OSError(errno.ENOENT, "Session not found or inactive", session_id)
    root = os.path.realpath(workspace)
    relative = os.path.relpath(resolved, root)
    parts = [] if relative == os.curdir else relative.split(os.sep)
    if os.pardir in parts:
        raise OSError(errno.EINVAL, "not a path beneath the workspace", resolved)
    return root, parts


def _named(exc: OSError, resolved: str) -> OSError:
    """``exc`` named after the checked path, as an open by path named it; a
    link met at the name -- one put there after the check, or a loop the
    check could not resolve -- says so."""
    if exc.errno == errno.ELOOP:
        return OSError(errno.ELOOP, "a link stands there: not followed", resolved)
    return type(exc)(exc.errno, exc.strerror, resolved)


def _open_checked(root: str, parts: list[str], flags: int, resolved: str) -> int:
    """A descriptor of the entry ``parts`` name under ``root`` (the workspace
    itself when ``parts`` is empty), opened with ``flags``, reached without
    following a link and without waiting: a FIFO is never waited on."""
    try:
        folder = _folder_beneath(root, parts[:-1])
        try:
            return os.open(
                parts[-1] if parts else os.curdir,
                flags | _NOFOLLOW | _NONBLOCK | _CLOEXEC,
                dir_fd=folder,
            )
        finally:
            os.close(folder)
    except OSError as exc:
        raise _named(exc, resolved) from exc


def _is_folder(root: str, parts: list[str], resolved: str) -> bool:
    """Whether the checked path names a folder, looked at without reading it
    (``O_PATH``) and without following a link."""
    try:
        fd = _open_checked(root, parts, getattr(os, "O_PATH", os.O_RDONLY), resolved)
    except OSError:
        return False
    try:
        return stat.S_ISDIR(os.fstat(fd).st_mode)
    finally:
        os.close(fd)


def _read_at_most(fd: int, limit: int) -> bytes:
    """At most ``limit`` bytes read from ``fd``."""
    chunks = []
    while limit > 0:
        chunk = os.read(fd, min(limit, 1 << 16))
        if not chunk:
            break
        chunks.append(chunk)
        limit -= len(chunk)
    return b"".join(chunks)


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data`` to ``fd``."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _write_checked(root: str, parts: list[str], data: bytes, resolved: str) -> None:
    """Write ``data`` as the file ``parts`` name under ``root``.

    Its folders are reached without following a link, a missing one made
    beneath its parent, as ``makedirs`` made it. A regular file at the name is
    written in place, truncated first; where nothing stands, the file is made
    (``O_EXCL``, mode 0o666 under the umask, as ``open(..., "w")`` made it).
    Anything else at the name -- a link put there after the check, a FIFO, a
    folder -- is refused (``OSError``), never written through nor waited on.
    """
    if not parts:
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), resolved)
    try:
        folder = _folder_beneath(root, parts[:-1], make=True)
        try:
            for _ in range(_CREATE_TRIES):
                try:
                    fd = os.open(
                        parts[-1],
                        os.O_WRONLY | _NOFOLLOW | _NONBLOCK | _CLOEXEC,
                        dir_fd=folder,
                    )
                    break
                except FileNotFoundError:
                    pass
                try:
                    fd = os.open(
                        parts[-1],
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC,
                        0o666,
                        dir_fd=folder,
                    )
                    break
                except FileExistsError:
                    continue
            else:
                raise FileExistsError(
                    errno.EEXIST, "the name keeps changing while it is written"
                )
        finally:
            os.close(folder)
    except OSError as exc:
        if exc.errno == errno.ENXIO:
            # A FIFO or a socket at the name, refused without waiting
            raise OSError(errno.ENXIO, "not a regular file", resolved) from exc
        raise _named(exc, resolved) from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", resolved)
        os.ftruncate(fd, 0)
        _write_all(fd, data)
    finally:
        os.close(fd)


def _rewrite(fd: int, writable: bool, resolved: str, text: str) -> None:
    """Write ``text`` over the file open on ``fd``: the file written is the file
    read. Encoded before the file is cut, so a text that cannot be written
    leaves it whole."""
    data = text.encode("utf-8")
    if not writable:
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), resolved)
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    _write_all(fd, data)


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
    The read runs with its session in use and counts as activity
    (``_held``); a destroy that overtakes it leaves its answer standing.

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
    return _held(
        mgr, session_id,
        lambda: _view(mgr, session_id, path, start_line, end_line),
    )


def _view(
    mgr: SandboxManager,
    session_id: str,
    path: str,
    start_line: int,
    end_line: int,
) -> str:
    """``_handle_sandbox_view``'s read, its session held."""
    ok, resolved, err = _resolve_path(mgr, session_id, path)
    if not ok:
        return f"Error: {err}"

    # The entry is opened once, without following a link; what it is, its
    # size and its content all come from that descriptor
    try:
        root, parts = _checked_parts(mgr, session_id, resolved)
        fd = _open_checked(root, parts, os.O_RDONLY, resolved)
    except (FileNotFoundError, NotADirectoryError):
        return f"Error: Path not found: {path}"
    except OSError as exc:
        if exc.errno == errno.ENXIO:
            return f"Error: Path not found: {path}"
        if isinstance(exc, PermissionError) and _is_folder(root, parts, resolved):
            return f"Error listing directory: {exc}"
        return f"Error reading file: {exc}"

    try:
        st = os.fstat(fd)

        # Directory listing
        if stat.S_ISDIR(st.st_mode):
            return _list_directory(
                fd, resolved, mgr.get_workspace_path(session_id) or root
            )

        # File reading
        if not stat.S_ISREG(st.st_mode):
            return f"Error: Path not found: {path}"

        # Size check
        size = st.st_size
        if size > MAX_VIEW_SIZE:
            return (
                f"Error: File too large ({size:,} bytes, "
                f"max {MAX_VIEW_SIZE:,} bytes)"
            )

        # Never more than the cap is read: a file that grew since its size
        # was taken is refused as too large
        try:
            data = _read_at_most(fd, MAX_VIEW_SIZE + 1)
            if len(data) > MAX_VIEW_SIZE:
                size = max(os.fstat(fd).st_size, len(data))
                return (
                    f"Error: File too large ({size:,} bytes, "
                    f"max {MAX_VIEW_SIZE:,} bytes)"
                )
        except OSError as exc:
            return f"Error reading file: {exc}"
    finally:
        os.close(fd)

    lines = io.TextIOWrapper(
        io.BytesIO(data), encoding="utf-8", errors="replace"
    ).readlines()

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


def _list_directory(folder: int, dir_path: str, workspace_root: str) -> str:
    """List directory contents within sandbox.

    The folder is read through ``folder``, a descriptor of ``dir_path``, and
    each entry's kind and size come from the entry itself, never through a
    link: a link shows as ``[LINK]``, whatever it points at.
    """
    try:
        with os.scandir(folder) as it:
            entries = sorted(it, key=lambda entry: entry.name)
    except OSError as exc:
        return f"Error listing directory: {exc}"

    if not entries:
        rel = os.path.relpath(dir_path, workspace_root)
        return f"Empty directory: {rel}"

    lines = []
    for entry in entries[:200]:  # Cap at 200 entries
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            lines.append(f"  [FILE] {entry.name}")
            continue
        if stat.S_ISLNK(st.st_mode):
            lines.append(f"  [LINK] {entry.name}")
        elif stat.S_ISDIR(st.st_mode):
            lines.append(f"  [DIR]  {entry.name}/")
        else:
            lines.append(f"  [FILE] {entry.name} ({st.st_size:,} bytes)")

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
    within the sandbox workspace. The file and its folders are reached
    without following a link: a link put on the checked path after the
    check is refused, never written through. The write runs with its
    session in use and counts as activity (``_held``); a destroy that
    overtakes it is the answer.

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
    return _held(
        mgr, session_id,
        lambda: _create_file(mgr, session_id, path, content),
        writes=True,
    )


def _create_file(
    mgr: SandboxManager,
    session_id: str,
    path: str,
    content: str,
) -> str:
    """``_handle_sandbox_create_file``'s write, its session held."""
    ok, resolved, err = _resolve_path(mgr, session_id, path)
    if not ok:
        return f"Error: {err}"

    # Size check
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        return f"Error writing file: {exc}"
    content_bytes = len(data)
    if content_bytes > MAX_CREATE_SIZE:
        return (
            f"Error: Content too large ({content_bytes:,} bytes, "
            f"max {MAX_CREATE_SIZE:,} bytes)"
        )

    # Validate the parent; its directories are created as the file is written
    parent = os.path.dirname(resolved)
    if parent:
        workspace = mgr.get_workspace_path(session_id)
        # Validate parent path is also within workspace
        parent_valid, _, parent_err = validate_sandbox_path(
            workspace, os.path.relpath(parent, workspace)
        )
        if not parent_valid:
            return f"Error: Parent directory path rejected: {parent_err}"

    try:
        root, parts = _checked_parts(mgr, session_id, resolved)
        _write_checked(root, parts, data, resolved)
    except OSError as exc:
        return f"Error writing file: {exc}"

    # Register created file for write-then-execute detection
    try:
        mgr.register_created_file(path, content)
    except Exception:
        pass  # Non-critical: validator tracking is defense-in-depth

    return f"File created: {path} ({content_bytes:,} bytes)"


# ---------------------------------------------------------------------------
# Tool: sandbox_str_replace
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The conservative str_replace recovery chain.
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


def _attempt_recovery(rewrite, content: str, old_str: str, new_str: str):
    """Run the recovery chain and, on a hit, perform the edit.

    Returns the final message (the strategy-naming success or the K-regions
    error) or ``None`` on a clean miss -- the caller then appends the
    structured hint to the unchanged not-found message. The recovered edit
    flows through the same file write as any edit, ``rewrite`` (and, at the
    session layer, the same diagnostics-after-write and the same copy-out
    review).
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
        rewrite(new_content)
    except (OSError, UnicodeEncodeError) as exc:
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
    The file is opened once, without following a link, and read and
    rewritten through that descriptor: the file written is the file read.
    The edit runs with its session in use and counts as activity
    (``_held``); a destroy that overtakes it is the answer.

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
    return _held(
        mgr, session_id,
        lambda: _str_replace(mgr, session_id, path, old_str, new_str),
        writes=True,
    )


def _str_replace(
    mgr: SandboxManager,
    session_id: str,
    path: str,
    old_str: str,
    new_str: str,
) -> str:
    """``_handle_sandbox_str_replace``'s edit, its session held."""
    ok, resolved, err = _resolve_path(mgr, session_id, path)
    if not ok:
        return f"Error: {err}"

    # Opened for writing too, or for reading only when its owner may not
    # write it: the edit then fails at its write, as it did
    writable = True
    try:
        root, parts = _checked_parts(mgr, session_id, resolved)
        try:
            fd = _open_checked(root, parts, os.O_RDWR, resolved)
        except PermissionError:
            writable = False
            fd = _open_checked(root, parts, os.O_RDONLY, resolved)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return f"Error: File not found: {path}"
    except OSError as exc:
        if exc.errno == errno.ENXIO:
            return f"Error: File not found: {path}"
        return f"Error reading file: {exc}"

    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return f"Error: File not found: {path}"
        return _replace_in(fd, writable, resolved, path, old_str, new_str)
    finally:
        os.close(fd)


def _replace_in(
    fd: int,
    writable: bool,
    resolved: str,
    path: str,
    old_str: str,
    new_str: str,
) -> str:
    """str_replace on the file open on ``fd``: read, matched and rewritten
    through that one descriptor."""
    if not old_str:
        return "Error: old_str cannot be empty"

    def rewrite(text: str) -> None:
        _rewrite(fd, writable, resolved, text)

    # Read file
    try:
        with open(fd, encoding="utf-8", errors="replace", closefd=False) as fh:
            content = fh.read()
    except OSError as exc:
        return f"Error reading file: {exc}"

    # Count occurrences
    count = content.count(old_str)
    if count == 0:
        # Exact match failed -- run the conservative
        # recovery chain; on a clean miss, append the structured hint to the
        # unchanged not-found message so the next attempt is informed.
        recovered = _attempt_recovery(rewrite, content, old_str, new_str)
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
        rewrite(new_content)
    except (OSError, UnicodeEncodeError) as exc:
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
