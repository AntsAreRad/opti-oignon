#!/usr/bin/env python3
"""
SANDBOX TOOLS - OPTI-OIGNON v1.7.5
==========================================

Session-aware wrapper layer for sandboxed file tools.

Provides a SandboxToolSession that binds a sandbox session to the
4 file tools, so the LLM sees simplified tool interfaces WITHOUT
needing to manage session_id:

  bash(command)           instead of  sandbox_bash(session_id, command)
  view(path)              instead of  sandbox_view(session_id, path)
  create_file(path, ...)  instead of  sandbox_create_file(session_id, ...)
  str_replace(path, ...)  instead of  sandbox_str_replace(session_id, ...)

The session_id is injected automatically by the wrapper.

Usage in the coding_agent:
    session = SandboxToolSession(sandbox_manager)
    session.start("coding-task-1")
    # Register session.get_tool_definitions() in tool_registry
    # LLM uses tools without knowing about sessions
    session.stop()

Author: Leon
"""

import errno
import fnmatch
import logging
import os
import re
import shlex
import stat
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from opti_oignon.sandbox_manager import (
        SANDBOX_AVAILABLE,
        SandboxManager,
        SandboxSession,
        validate_sandbox_path,
    )
    from opti_oignon.sandbox_manager import (
        sandbox_manager as _default_sandbox_manager,
    )
except ImportError:
    SANDBOX_AVAILABLE = False
    _default_sandbox_manager = None
    SandboxManager = None
    SandboxSession = None
    validate_sandbox_path = None

try:
    from opti_oignon.file_tools import (
        _handle_sandbox_bash,
        _handle_sandbox_create_file,
        _handle_sandbox_str_replace,
        _handle_sandbox_view,
    )
    FILE_TOOLS_AVAILABLE = True
except ImportError:
    FILE_TOOLS_AVAILABLE = False
    _handle_sandbox_bash = None
    _handle_sandbox_view = None
    _handle_sandbox_create_file = None
    _handle_sandbox_str_replace = None

try:
    from opti_oignon.tool_registry import ToolDefinition, ToolParam, ToolRegistry
    from opti_oignon.tool_registry import tool_registry as _default_tool_registry
    TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    TOOL_REGISTRY_AVAILABLE = False
    ToolDefinition = None
    ToolParam = None
    ToolRegistry = None
    _default_tool_registry = None

# Guarded YAML import for the [agent.diagnostics] config block. The
# loader falls back to the built-in defaults when PyYAML or the file is
# unavailable, so behaviour never depends on the file being present.
try:
    import yaml as _yaml
except ImportError:  # pragma: no cover - defensive guard
    _yaml = None

# ---------------------------------------------------------------------------
# Constants: read-only workspace tools and diagnostics
# ---------------------------------------------------------------------------

# Hard caps for the read-only tools (AGT_SPEC 5.1). Requested values are
# clamped into [1, cap]; defaults live on the schemas in agent/tools.py.
GREP_MAX_RESULTS_CAP = 500
GREP_CONTEXT_CAP = 5
GLOB_MAX_RESULTS_CAP = 1000
LS_MAX_ENTRIES_CAP = 1000

# Files larger than this are skipped by grep (the view 1 MiB posture).
MAX_GREP_FILE_SIZE = 1024 * 1024

# grep searches no deeper than this many folders below its target: its walk
# holds one descriptor per level, and says how many folders it left.
GREP_MAX_DEPTH = 128

# Null-byte binary sniff window: a NUL anywhere in the first chunk marks the
# file binary and grep skips it.
_BINARY_SNIFF_BYTES = 8192

# Diagnostics-after-write (AGT_SPEC 5.2): suffix map and built-in defaults.
# The Python ladder runs INSIDE the disposable bwrap sandbox via
# execute_command (validator + signed audit log included); the Svelte check is
# a trusted host-side read of the workspace copy (no execution). Both are
# gated on the bwrap backend so the container's clean path stays
# byte-identical.
DIAGNOSTICS_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "tools": ["ruff", "pyflakes", "py_compile"],
    "type_tool": "mypy",
    "timeout_s": 10,
    "max_block_bytes": 4096,
    "max_findings": 25,
}
_DIAG_PROBE_COMMANDS = {
    "ruff": "command -v ruff",
    "pyflakes": "command -v pyflakes",
    "py_compile": "command -v python3",
    "mypy": "command -v mypy",
}
_DIAG_TRUNCATED_MARKER = "[diagnostics truncated]"

# HTML void elements never need a closing tag (tag-balance check).
_VOID_ELEMENTS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
)
_SVELTE_BLOCK_KINDS = ("if", "each", "await", "key", "snippet")
# A tag's head, and the end its attributes stop at; ``_tags`` reads the two, once per place.
_TAG_HEAD_RE = re.compile(r"<\s*(/?)([A-Za-z][A-Za-z0-9:._-]*)(?![A-Za-z0-9:._-])")
_TAG_END_RE = re.compile(r"(/?)(?<!\s)\s*>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b", re.IGNORECASE)
_SCRIPT_STYLE_ENDS = {
    "script": re.compile(r"</script\s*>", re.IGNORECASE),
    "style": re.compile(r"</style\s*>", re.IGNORECASE),
}
# Whether a closing tag's name is the opening's as a case-blind back-reference compares them: ``name + "\x00" +
# closing name``. A pattern's literal reads dotted and dotless i and long s otherwise.
_SAME_NAME_RE = re.compile(r"([^\x00]*)\x00\1", re.IGNORECASE)


def _resolve_workspace_path(
    mgr: Any, session_id: str | None, path: str
) -> tuple[bool, str, str, str]:
    """Resolve a path inside the session workspace (the view precedent).

    Returns (ok, workspace_root, resolved_absolute_path, error_message).
    Confinement is exactly ``validate_sandbox_path``; refusals reuse the
    established message shapes.
    """
    if mgr is None or validate_sandbox_path is None:
        return False, "", "", "Sandbox manager not available"
    workspace = mgr.get_workspace_path(session_id)
    if workspace is None:
        return False, "", "", f"Session not found or inactive: {session_id}"
    valid, resolved, err = validate_sandbox_path(workspace, path)
    if not valid:
        return False, workspace, "", f"Path rejected: {err}"
    return True, workspace, resolved, ""


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_PATH = getattr(os, "O_PATH", os.O_RDONLY)


def _open_resolved(workspace: str, resolved: str, flags: int) -> int:
    """A descriptor of ``resolved`` -- a path ``_resolve_workspace_path``
    checked and resolved -- opened with ``flags`` without waiting (a FIFO is
    never waited on).

    The path is reached beneath the workspace one component at a time
    without following a link, as the file tools reach theirs: the links
    inside the workspace were resolved at the check, so a link met on the way
    now was put there after it, and is refused (``OSError``), never followed.
    """
    root = os.path.realpath(workspace)
    relative = os.path.relpath(resolved, root)
    parts = [] if relative == os.curdir else relative.split(os.sep)
    if os.pardir in parts:
        raise OSError(errno.EINVAL, "not a path beneath the workspace", resolved)
    try:
        folder = os.open(root, _PATH | _DIRECTORY | _CLOEXEC)
        try:
            for part in parts[:-1]:
                inner = os.open(part, _PATH | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=folder)
                os.close(folder)
                folder = inner
            return os.open(
                parts[-1] if parts else os.curdir,
                flags | _NOFOLLOW | _NONBLOCK | _CLOEXEC,
                dir_fd=folder,
            )
        finally:
            os.close(folder)
    except OSError as exc:
        raise _named(exc, resolved) from exc


def _named(exc: OSError, resolved: str) -> OSError:
    """``exc`` named after the checked path, as an open by path named it; a
    link met at the name says so."""
    if exc.errno == errno.ELOOP:
        return OSError(errno.ELOOP, "a link stands there: not followed", resolved)
    return type(exc)(exc.errno, exc.strerror, resolved)


def _read_fd(fd: int, limit: int | None) -> bytes:
    """The bytes of the regular file open on ``fd``, at most ``limit`` of
    them; anything but a regular file is refused (``OSError``)."""
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise OSError(errno.EINVAL, "not a regular file")
    chunks = []
    left = limit
    while left is None or left > 0:
        chunk = os.read(fd, 1 << 16 if left is None else min(left, 1 << 16))
        if not chunk:
            break
        chunks.append(chunk)
        if left is not None:
            left -= len(chunk)
    return b"".join(chunks)


def _read_resolved(workspace: str, resolved: str, limit: int | None = None) -> bytes:
    """The bytes of the regular file at the checked path ``resolved``, at most
    ``limit`` of them, read through one descriptor opened beneath the
    workspace (``_open_resolved``); anything but a regular file is refused
    (``OSError``)."""
    fd = _open_resolved(workspace, resolved, os.O_RDONLY)
    try:
        return _read_fd(fd, limit)
    finally:
        os.close(fd)


def _listing(folder: int) -> list[tuple[str, str, int]]:
    """The entries of the folder open on ``folder`` as (sort key, name,
    mode), sorted by key, each entry read itself without following a link
    (lstat beneath the folder's descriptor); links are left out. A folder's
    key ends with a slash, so a depth-first walk meets the files in the order
    of their paths. An entry whose status cannot be read keeps mode 0, unless
    the listing saw a folder there; a folder that cannot be listed has no
    entries, as ``os.walk`` read it."""
    try:
        with os.scandir(folder) as it:
            seen = [(entry.name, entry.is_dir(follow_symlinks=False)) for entry in it]
    except OSError:
        return []
    out = []
    for name, seen_folder in seen:
        try:
            mode = os.stat(name, dir_fd=folder, follow_symlinks=False).st_mode
        except OSError:
            if seen_folder:
                continue
            mode = 0
        if stat.S_ISLNK(mode):
            continue  # symlinks are skipped in every walk
        out.append((name + "/" if stat.S_ISDIR(mode) else name, name, mode))
    out.sort()
    return out


def _grep_files(workspace: str, resolved: str, glob_filter: str, cut: dict[str, int]):
    """What grep scans at the checked path ``resolved``, as (relpath, bytes)
    in the order of their paths -- the bytes at most one past the size cap,
    or None for a file that could not be read -- or None when the path is not
    there.

    The target is looked at without being opened for reading (``O_PATH``),
    so a FIFO named as the target is never opened. A file target yields
    itself; a folder is walked depth-first, beneath the workspace through no
    link: each folder opened from its parent's descriptor, each entry read
    itself, each regular file read from its folder's descriptor, a FIFO or
    any other special file never opened (counted as skipped), at most
    ``GREP_MAX_DEPTH`` folders deep (``cut["folders"]`` counts the folders
    left); an optional fnmatch filter applies to the workspace-relative path.
    """
    try:
        fd = _open_resolved(workspace, resolved, _PATH)
        try:
            mode = os.fstat(fd).st_mode
        finally:
            os.close(fd)
    except OSError:
        return None
    if stat.S_ISREG(mode):
        return _grep_target_file(workspace, resolved)
    if stat.S_ISDIR(mode):
        return _grep_walk(workspace, resolved, glob_filter, cut)
    return None


def _grep_target_file(workspace: str, resolved: str):
    """The file grep was pointed at, as ``_grep_files`` yields it."""
    try:
        data = _read_resolved(workspace, resolved, MAX_GREP_FILE_SIZE + 1)
    except OSError:
        data = None
    yield os.path.relpath(resolved, workspace), data


def _grep_walk(workspace: str, resolved: str, glob_filter: str, cut: dict[str, int]):
    """The files under the folder grep was pointed at, as ``_grep_files``
    yields them."""
    try:
        top = _open_resolved(workspace, resolved, os.O_RDONLY | _DIRECTORY)
    except OSError:
        return  # a folder that cannot be listed is skipped, as os.walk skipped it
    # (descriptor, path, entries left, depth) of each folder on the current path
    stack = [(top, resolved, iter(_listing(top)), 0)]
    try:
        while stack:
            folder, here, rest, depth = stack[-1]
            item = next(rest, None)
            if item is None:
                stack.pop()
                os.close(folder)
                continue
            _key, name, mode = item
            full = os.path.join(here, name)
            if stat.S_ISDIR(mode):
                if depth >= GREP_MAX_DEPTH:
                    cut["folders"] += 1
                    continue
                try:
                    inner = os.open(
                        name,
                        os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _NONBLOCK | _CLOEXEC,
                        dir_fd=folder,
                    )
                except OSError:
                    continue
                stack.append((inner, full, iter(_listing(inner)), depth + 1))
                continue
            rel = os.path.relpath(full, workspace)
            if glob_filter and not fnmatch.fnmatch(rel, glob_filter):
                continue
            data = None
            if stat.S_ISREG(mode):
                try:
                    data = _read_beneath(folder, name, MAX_GREP_FILE_SIZE + 1)
                except OSError:
                    pass
            yield rel, data
    finally:
        for folder, _here, _rest, _depth in stack:
            os.close(folder)


def _read_beneath(folder: int, name: str, limit: int) -> bytes:
    """At most ``limit`` bytes of the regular file ``name`` in the folder open
    on ``folder``, opened without following a link and without waiting."""
    fd = os.open(name, os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _CLOEXEC, dir_fd=folder)
    try:
        return _read_fd(fd, limit)
    finally:
        os.close(fd)


def _glob_segments(pattern: str) -> tuple[int, list[str]]:
    """The leading ``..`` segments of a glob pattern, counted, and its other
    segments (empty and ``.`` ones dropped, consecutive ``**`` read as one,
    as pathlib reads them). An absolute pattern is refused as ``Path.glob``
    refused it (``NotImplementedError``), and so is a pattern that names
    nothing but its base, and a ``..`` past the pattern's start
    (``ValueError``). A pattern ending with a slash names folders only:
    no segment is left, so it finds no file."""
    if pattern.startswith("/"):
        raise NotImplementedError("Non-relative patterns are unsupported")
    if "\x00" in pattern:
        raise ValueError("lstat: embedded null character in path")  # pathlib's own words
    parts = [part for part in pattern.split("/") if part not in ("", ".")]
    up = 0
    while parts and parts[0] == "..":
        up += 1
        parts.pop(0)
    if not parts and not up:
        raise ValueError("Unacceptable pattern: PosixPath('.')")  # pathlib's own words
    if ".." in parts:
        raise ValueError("'..' is only read at the pattern's start")
    if pattern.endswith("/"):
        return up, []
    collapsed: list[str] = []
    for part in parts:
        if not (part == "**" and collapsed and collapsed[-1] == "**"):
            collapsed.append(part)
    return up, collapsed


def _glob_closure(parts: list[str], positions: set[int]) -> set[int]:
    """``positions`` with every position a ``**`` may skip (it matches zero
    folders too)."""
    closed = set(positions)
    pending = list(positions)
    while pending:
        i = pending.pop()
        if i < len(parts) and parts[i] == "**" and i + 1 not in closed:
            closed.add(i + 1)
            pending.append(i + 1)
    return closed


_GLOB_MAGIC = re.compile(r"[*?[]")


def _glob_walk(base: int, here: str, parts: list[str], found: list[tuple[float, str]], cut: dict[str, int]):
    """The regular files under the folder held on ``base`` (an ``O_PATH``
    descriptor) whose path from it matches ``parts``, added to ``found`` as
    (mtime, path from ``here``).

    Walked through descriptors, each folder opened from its parent's without
    following a link (``O_PATH``: passing through a folder asks only to
    search it, as a path walk does), each entry read itself (lstat beneath
    its folder): a link is neither walked nor returned. A folder is entered
    only when a position of the pattern can still match below it, so a
    pattern with no ``**`` reads no deeper than its segments; ``**`` matches
    zero or more folders, and last it matches every file below, as pathlib
    3.13 reads it. A folder that may be searched but not listed is looked
    into by its literal segments alone, as pathlib looked. At most
    ``GREP_MAX_DEPTH`` folders deep."""
    last = len(parts) - 1

    def walk(folder: int, rel: str, positions: set[int], depth: int) -> None:
        try:
            lister = os.open(os.curdir, os.O_RDONLY | _DIRECTORY | _CLOEXEC, dir_fd=folder)
            try:
                with os.scandir(lister) as it:
                    names = sorted(entry.name for entry in it)
            finally:
                os.close(lister)
        except OSError:
            # Not listed: the pattern's literal names are looked for one by one
            names = sorted({
                parts[i] for i in positions
                if i <= last and parts[i] != "**" and not _GLOB_MAGIC.search(parts[i])
            })
        for name in names:
            try:
                st = os.stat(name, dir_fd=folder, follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                continue  # symlinks are skipped in every walk
            path = os.path.join(rel, name) if rel else name
            if stat.S_ISREG(st.st_mode):
                if any(
                    (parts[i] == "**" and i == last)
                    or (i == last and parts[i] != "**" and fnmatch.fnmatchcase(name, parts[i]))
                    for i in positions
                ):
                    found.append((st.st_mtime, path))
                continue
            if not stat.S_ISDIR(st.st_mode):
                continue
            below = set()
            for i in positions:
                if i > last:
                    continue
                if parts[i] == "**":
                    below.add(i)
                elif i < last and fnmatch.fnmatchcase(name, parts[i]):
                    below.add(i + 1)
            below = {i for i in _glob_closure(parts, below) if i <= last}
            if not below:
                continue
            if depth >= GREP_MAX_DEPTH:
                cut["folders"] += 1
                continue
            try:
                inner = os.open(name, _PATH | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=folder)
            except OSError:
                continue
            try:
                walk(inner, path, below, depth + 1)
            finally:
                os.close(inner)

    if parts:
        walk(base, here, {i for i in _glob_closure(parts, {0}) if i <= last}, 0)


def _is_binary(data: bytes) -> bool:
    """Null-byte sniff: True when the first chunk contains a NUL byte."""
    return b"\x00" in data[:_BINARY_SNIFF_BYTES]


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(int(value), hi))


def load_diagnostics_config() -> dict[str, Any]:
    """The [agent.diagnostics] block of agent/config.yaml, defaults applied.

    A tiny guarded reader local to this module (the agent config_loader is
    deliberately left unedited): a missing file, missing PyYAML, a
    malformed file, or a missing block all yield ``DIAGNOSTICS_DEFAULTS``.
    """
    cfg = dict(DIAGNOSTICS_DEFAULTS)
    cfg["tools"] = list(DIAGNOSTICS_DEFAULTS["tools"])
    if _yaml is None:
        return cfg
    try:
        config_path = Path(__file__).parent / "agent" / "config.yaml"
        raw = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return cfg
    block = raw.get("diagnostics") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        return cfg
    cfg["enabled"] = bool(block.get("enabled", cfg["enabled"]))
    tools = block.get("tools")
    if isinstance(tools, (list, tuple)) and tools:
        cfg["tools"] = [str(t) for t in tools if str(t) in _DIAG_PROBE_COMMANDS]
    try:
        cfg["timeout_s"] = max(1, int(block.get("timeout_s", cfg["timeout_s"])))
    except Exception:
        pass
    try:
        cfg["max_block_bytes"] = max(
            256, int(block.get("max_block_bytes", cfg["max_block_bytes"]))
        )
    except Exception:
        pass
    try:
        cfg["max_findings"] = max(1, int(block.get("max_findings", cfg["max_findings"])))
    except Exception:
        pass
    return cfg


def _closing_tag(text: str, name: str, at: int) -> "re.Match[str] | None":
    """The first closing tag at or after ``at`` whose name is ``name`` as a case-blind back-reference reads it."""
    ends = _SCRIPT_STYLE_ENDS["script" if len(name) == len("script") else "style"]
    while True:
        found = ends.search(text, at)
        if found is None:
            return None
        closing_name = text[found.start() + 2:found.start() + 2 + len(name)]
        if _SAME_NAME_RE.fullmatch(name + "\x00" + closing_name):
            return found
        at = found.start() + 1


def _blank_script_style(text: str) -> str:
    """``text`` with the inner content of each script and style element taken out, the tags kept.

    What ``re.sub(r"(<(script|style)\\b[^>]*>).*?(</\\2\\s*>)", lambda m: m.group(1) + m.group(3), text,
    flags=re.IGNORECASE | re.DOTALL)`` gives, in time linear in the text: that pattern read from every opening left
    open to the end of the text. Here the ``>`` after an opening is searched once per stretch (the next one is kept
    with the place it was searched from), and so is the closing tag of each name as the back-reference compares names
    (kept per name, lowered): an opening whose name has no closing tag after it is passed at once.
    """
    out, last, at = [], 0, 0
    angle = (0, -2)  # (searched from, the next ">" from there); -2 before any search, -1 for none
    closings: dict[str, tuple] = {}  # name lowered -> (searched from, closing tag or None)
    while True:
        head = _SCRIPT_STYLE_RE.search(text, at)
        if head is None:
            break
        if not (angle[0] <= head.end() and (angle[1] == -1 or head.end() <= angle[1])):
            angle = (head.end(), text.find(">", head.end()))
        if angle[1] == -1:
            break  # no ">" after this opening, so none after any opening after it
        body = angle[1] + 1
        name = head.group(1)
        seen = closings.get(name.lower())
        if seen is None or not (seen[0] <= body and (seen[1] is None or body <= seen[1].start())):
            seen = (body, _closing_tag(text, name, body))
            closings[name.lower()] = seen
        closing = seen[1]
        if closing is None:
            at = head.start() + 1
            continue
        out.append(text[last:body])
        out.append(closing.group())
        last = at = closing.end()
    out.append(text[last:])
    return "".join(out)


def _tags(text: str) -> list[tuple[str, str, str, str]]:
    """The tags of ``text`` as ``(closing slash, name, attributes, self-closing slash)``, in time linear in the text.

    What ``re.findall`` gives with the pattern
    ``<\\s*(/?)([A-Za-z][A-Za-z0-9:._-]*)(?![A-Za-z0-9:._-])((?:"[^"]*"|'[^']*'|[^>"'])*?)(/?)(?<!\\s)\\s*>``,
    which read the attributes again from every ``<`` of a long run with no ``>``. From the end of a tag's name, the
    attributes are walked one unit at a time -- a quoted string, or one character that is neither a quote nor
    ``>`` -- and stop at the first place where ``_TAG_END_RE`` matches; a quote with no closing one, a ``>`` that is
    not an end, or the end of the text fails the tag. Where a walk stops depends on the place it passes only, so it
    is kept for every place passed: each place is walked once.
    """
    stops: dict[int, tuple | None] = {}

    def walk(place):
        passed, stop = [], None
        while True:
            if place in stops:
                stop = stops[place]
                break
            passed.append(place)
            end = _TAG_END_RE.match(text, place)
            if end is not None:
                stop = (place, end)
                break
            char = text[place:place + 1]
            if char in ("\"", "'"):
                quote = text.find(char, place + 1)
                if quote == -1:
                    break
                place = quote + 1
            elif char and char != ">":
                place += 1
            else:
                break
        for step in passed:
            stops[step] = stop
        return stop

    found, at = [], 0
    while True:
        head = _TAG_HEAD_RE.search(text, at)
        if head is None:
            return found
        stop = walk(head.end())
        if stop is None:
            at = head.start() + 1
            continue
        attributes_end, end = stop
        found.append((head.group(1), head.group(2), text[head.end():attributes_end], end.group(1)))
        at = end.end()


def _strip_for_tag_scan(text: str) -> str:
    """Reduce Svelte source to scannable markup (heuristic, documented).

    Comments and the inner content of script/style elements are removed (the
    tags themselves are kept), and simple ``{...}`` template expressions that
    are not block markers are blanked so comparison operators inside them do
    not read as tags. The checker is a conservative heuristic; the
    host-assured pass exercises it on real Svelte trees.
    """
    # Linear on a comment left open: one with no closing after it takes the rest at once -- no comment after it can
    # close -- and is kept, as only a comment that reached its closing sets ``real``.
    text = re.sub(r"<!--(?:.*?-->(?P<real>)|.*)",
                  lambda m: "" if m.group("real") is not None else m.group(0), text, flags=re.DOTALL)
    text = _blank_script_style(text)
    # Blank non-block brace expressions with a small non-nesting scanner.
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{" and i + 1 < n and text[i + 1] not in "#/:":
            depth = 1
            j = i + 1
            while j < n and depth:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            out.append(" ")
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def svelte_tag_findings(text: str) -> list[str]:
    """Tag-balance findings for a Svelte component (pure, deterministic).

    Checks per-name HTML open/close counts (void and self-closing elements
    excluded) and per-kind Svelte block balance ({#if}/{/if} and friends).
    Returns an empty list when balanced.
    """
    findings: list[str] = []
    scannable = _strip_for_tag_scan(text)
    opens: dict[str, int] = {}
    closes: dict[str, int] = {}
    for closing, name, _attrs, selfclose in _tags(scannable):
        lowered = name.lower()
        if lowered in _VOID_ELEMENTS or (selfclose and not closing):
            continue
        if closing:
            closes[lowered] = closes.get(lowered, 0) + 1
        else:
            opens[lowered] = opens.get(lowered, 0) + 1
    for name in sorted(set(opens) | set(closes)):
        o, c = opens.get(name, 0), closes.get(name, 0)
        if o != c:
            findings.append(f"unbalanced tag <{name}>: {o} opened, {c} closed")
    for kind in _SVELTE_BLOCK_KINDS:
        o = len(re.findall(r"\{#" + kind + r"\b", scannable))
        c = len(re.findall(r"\{/" + kind + r"\s*\}", scannable))
        if o != c:
            findings.append(
                f"unbalanced Svelte block {{#{kind}}}: {o} opened, {c} closed"
            )
    return findings


class SandboxToolSession:
    """Binds a sandbox session to simplified tool interfaces.

    Manages the lifecycle of a sandbox session and produces
    ToolDefinition instances where session_id is pre-bound.
    The LLM sees clean tool interfaces: bash, view, create_file,
    str_replace -- without needing to know about sessions.

    SECURITY: When started, activates sandbox mode on the tool_registry,
    which DISABLES all unsafe unsandboxed tools (execute_code, read_file,
    write_file, list_files). This prevents the LLM from bypassing the
    sandbox. Tools are re-enabled when the session stops.
    """

    def __init__(
        self,
        sandbox_mgr: SandboxManager | None = None,
        tool_registry: "ToolRegistry | None" = None,
    ):
        self._mgr = sandbox_mgr or _default_sandbox_manager
        self._registry = tool_registry or _default_tool_registry
        self._session: SandboxSession | None = None
        self._session_id: str | None = None
        # Diagnostics: the linter ladder is probed once per session and
        # cached; the cache is cleared on every lifecycle change.
        self._diag_probed: bool = False
        self._diag_tool: str | None = None
        self._type_probed: bool = False
        self._type_tool: str | None = None

    @property
    def active(self) -> bool:
        """Whether a sandbox session is currently active."""
        return self._session is not None and self._session.active

    @property
    def session_id(self) -> str | None:
        """Current session ID, or None if not started."""
        return self._session_id

    @property
    def sandbox_manager(self) -> SandboxManager | None:
        """Access the underlying sandbox manager."""
        return self._mgr

    def start(
        self,
        session_id: str | None = None,
    ) -> str:
        """Start a new sandbox session.

        Args:
            session_id: Custom ID, or auto-generated UUID if None.

        Returns:
            The session ID.

        Raises:
            RuntimeError: If sandbox manager unavailable or session
                already active.
        """
        if self._mgr is None:
            raise RuntimeError("Sandbox manager not available")
        if self.active:
            raise RuntimeError(
                f"Session already active: {self._session_id}. "
                f"Call stop() first."
            )

        sid = session_id or f"tool-session-{uuid.uuid4().hex[:12]}"
        self._session = self._mgr.create_sandbox(sid)
        self._session_id = sid
        self._reset_diag_cache()

        # SECURITY: Disable unsafe unsandboxed tools in the registry
        # so the LLM cannot bypass the sandbox.
        self._apply_registry_lockout()

        logger.info("SandboxToolSession started: %s", sid)
        return sid

    def attach(self, session_id: str) -> str:
        """Bind this wrapper to an EXISTING managed workspace.

        The conversation-binding seam (SANDBOX_WORKSPACE_SPEC section 4.1):
        unlike start(), nothing is created; unlike stop(), the matching
        detach() destroys nothing. The set_sandbox_mode lockout of the
        unsandboxed tools applies exactly as for start() -- the invariant is
        unchanged whenever a workspace is active.

        Args:
            session_id: An existing, active sandbox session id.

        Returns:
            The session ID.

        Raises:
            RuntimeError: If the manager is unavailable or a session is
                already active on this wrapper.
            ValueError: If the session does not exist or is inactive.
        """
        if self._mgr is None:
            raise RuntimeError("Sandbox manager not available")
        if self.active:
            raise RuntimeError(
                f"Session already active: {self._session_id}. "
                f"Call stop() or detach() first."
            )
        session = self._mgr.get_session(session_id)
        if session is None or not session.active:
            raise ValueError(f"Session not found or inactive: {session_id}")

        self._session = session
        self._session_id = session_id
        self._reset_diag_cache()
        self._apply_registry_lockout()
        logger.info("SandboxToolSession attached: %s", session_id)
        return session_id

    def detach(self) -> bool:
        """Release an attached workspace WITHOUT destroying it.

        Re-enables the tools the lockout disabled; the workspace and its
        files persist (the conversation binding owns the lifetime).

        Returns:
            True if a session was released, False if none was active.
        """
        if not self.active:
            return False
        self._release_registry_lockout()
        sid = self._session_id
        self._session = None
        self._session_id = None
        self._reset_diag_cache()
        logger.info("SandboxToolSession detached: %s", sid)
        return True

    def _apply_registry_lockout(self) -> None:
        """Disable the unsafe unsandboxed tools while a workspace is active.

        Factored from start() so attach() enforces the identical
        invariant; behaviour is unchanged.
        """
        if self._registry is None:
            return
        disabled = self._registry.set_sandbox_mode(True)
        if disabled:
            logger.info(
                "Sandbox mode activated: disabled %d unsafe tools: %s",
                len(disabled), disabled,
            )

        # Optionally disable web_search (data exfiltration risk)
        if (
            self._mgr is not None
            and hasattr(self._mgr.config, 'disable_web_search_in_sandbox')
            and self._mgr.config.disable_web_search_in_sandbox
        ):
            ws_tool = self._registry.get("web_search")
            if ws_tool is not None and ws_tool.enabled:
                ws_tool.enabled = False
                self._registry._disabled_by_sandbox.add("web_search")
                logger.info(
                    "Sandbox mode: disabled web_search "
                    "(disable_web_search_in_sandbox=true)"
                )

    def _release_registry_lockout(self) -> None:
        """Re-enable the tools the lockout disabled (stop and detach)."""
        if self._registry is None:
            return
        restored = self._registry.set_sandbox_mode(False)
        if restored:
            logger.info(
                "Sandbox mode deactivated: re-enabled %d tools: %s",
                len(restored), restored,
            )

    def stop(self) -> bool:
        """Stop and destroy the current sandbox session.

        Re-enables any unsafe tools that were disabled by sandbox mode.

        Returns:
            True if destroyed, False if no active session.
        """
        if not self.active or self._mgr is None:
            return False

        # SECURITY: Re-enable unsafe tools now that sandbox is closing
        self._release_registry_lockout()

        result = self._mgr.destroy_sandbox(self._session_id)
        self._session = None
        self._session_id = None
        self._reset_diag_cache()
        return result

    def inject_files(self, file_paths: list[str]) -> list[str]:
        """Inject files into the active sandbox.

        Returns list of injected file paths within the workspace.
        """
        if not self.active or self._mgr is None:
            raise RuntimeError("No active sandbox session")
        return self._mgr.inject_files(self._session_id, file_paths)

    def inject_directory(self, src_dir: str, dest_subdir: str = "") -> int:
        """Inject a directory tree into the active sandbox.

        Returns count of files copied.
        """
        if not self.active or self._mgr is None:
            raise RuntimeError("No active sandbox session")
        return self._mgr.inject_directory(
            self._session_id, src_dir, dest_subdir
        )

    def extract_files(self) -> list[dict[str, Any]]:
        """List files available for extraction from the sandbox."""
        if not self.active or self._mgr is None:
            raise RuntimeError("No active sandbox session")
        return self._mgr.extract_files(self._session_id)

    # -----------------------------------------------------------------
    # Simplified tool handlers (session_id auto-injected)
    # -----------------------------------------------------------------

    def bash(self, command: str, timeout: int = 30) -> str:
        """Execute a bash command in the sandbox."""
        self._check_active()
        return _handle_sandbox_bash(
            self._session_id, command, timeout,
            _sandbox_manager=self._mgr,
        )

    def view(
        self,
        path: str,
        start_line: int = 0,
        end_line: int = 0,
    ) -> str:
        """Read a file or list a directory in the sandbox."""
        self._check_active()
        return _handle_sandbox_view(
            self._session_id, path, start_line, end_line,
            _sandbox_manager=self._mgr,
        )

    def create_file(self, path: str, content: str) -> str:
        """Create or overwrite a file in the sandbox.

        When the write succeeds and the filename matches the
        diagnostics suffix map, an in-sandbox linter pass (or the host-side
        Svelte tag-balance read) may append a ``[diagnostics]`` block; a
        clean write returns byte-identical output to the prior shape.
        """
        self._check_active()
        result = _handle_sandbox_create_file(
            self._session_id, path, content,
            _sandbox_manager=self._mgr,
        )
        return result + self._maybe_diagnostics(path, result)

    def str_replace(
        self,
        path: str,
        old_str: str,
        new_str: str = "",
    ) -> str:
        """Find and replace a unique string in a sandbox file.

        A successful edit may append a ``[diagnostics]`` block exactly
        as for ``create_file``; the clean path is byte-identical.
        """
        self._check_active()
        result = _handle_sandbox_str_replace(
            self._session_id, path, old_str, new_str,
            _sandbox_manager=self._mgr,
        )
        return result + self._maybe_diagnostics(path, result)

    # -----------------------------------------------------------------
    # Read-only workspace tools: grep, glob, ls.
    # Trusted host-side reads on the view precedent: path-confined via
    # validate_sandbox_path, then reached beneath the workspace through no
    # link (_open_resolved, glob's pattern walked so too), active session required,
    # deterministic sorted output, truncated flags, null-byte binary sniff,
    # 1 MiB skip.
    # -----------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: str = ".",
        *,
        glob: str = "",
        is_regex: bool = False,
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_results: int = 100,
    ) -> str:
        """Search workspace file contents; one 'relpath:lineno: text' per match."""
        self._check_active()
        ok, workspace, resolved, err = _resolve_workspace_path(
            self._mgr, self._session_id, path
        )
        if not ok:
            return f"Error: {err}"
        if not pattern:
            return "Error: grep requires a non-empty 'pattern'"
        max_results = _clamp(max_results, 1, GREP_MAX_RESULTS_CAP)
        context_lines = _clamp(context_lines, 0, GREP_CONTEXT_CAP)

        if is_regex:
            from opti_oignon.linear_pattern import CostlyPattern, compile_linear
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                rx = compile_linear(pattern, flags)
            except CostlyPattern as exc:
                return (f"Error: regex pattern refused, it cannot be read in linear time on a long line ({exc}); "
                        f"search a fixed string (is_regex=false), or a pattern with no back-reference, conditional "
                        f"or atomic group")
            except re.error as exc:
                return f"Error: invalid regex pattern: {exc}"

            def matches(line: str) -> bool:
                return rx.search(line) is not None
        else:
            needle = pattern if case_sensitive else pattern.casefold()

            def matches(line: str) -> bool:
                hay = line if case_sensitive else line.casefold()
                return needle in hay

        cut = {"folders": 0}
        files = _grep_files(workspace, resolved, glob, cut)
        if files is None:
            return f"Error: Path not found: {path}"

        match_lines: list[str] = []
        matched_files: set[str] = set()
        skipped = 0
        truncated = False
        count = 0
        try:
            for rel, data in files:
                if truncated:
                    break
                # Read beneath the workspace through no link, never more than
                # the size cap: unreadable counts as skipped, never as a crash
                if data is None or len(data) > MAX_GREP_FILE_SIZE or _is_binary(data):
                    skipped += 1
                    continue
                lines = data.decode("utf-8", errors="replace").splitlines()
                for idx, line in enumerate(lines):
                    if not matches(line):
                        continue
                    if count >= max_results:
                        truncated = True
                        break
                    count += 1
                    matched_files.add(rel)
                    match_lines.append(f"{rel}:{idx + 1}: {line}")
                    if context_lines:
                        lo = max(0, idx - context_lines)
                        hi = min(len(lines), idx + 1 + context_lines)
                        for c in range(lo, hi):
                            if c == idx:
                                continue
                            match_lines.append(f"  {c + 1}| {lines[c]}")
        finally:
            files.close()  # the walk's descriptors, even when grep stops early
        header = f"{count} match(es) in {len(matched_files)} file(s)"
        if truncated:
            header += " [truncated]"
        if skipped:
            header += f" [{skipped} file(s) skipped: binary or >1 MiB]"
        if cut["folders"]:
            header += f" [{cut['folders']} folder(s) deeper than {GREP_MAX_DEPTH} levels not searched]"
        if not match_lines:
            return header
        return header + "\n" + "\n".join(match_lines)

    def glob(self, pattern: str, path: str = ".", *, max_results: int = 200) -> str:
        """Find workspace files by glob pattern, newest first then by name."""
        self._check_active()
        ok, workspace, resolved, err = _resolve_workspace_path(
            self._mgr, self._session_id, path
        )
        if not ok:
            return f"Error: {err}"
        if not pattern:
            return "Error: glob requires a non-empty 'pattern'"
        try:
            os.close(_open_resolved(workspace, resolved, _PATH | _DIRECTORY))
        except OSError:
            return f"Error: Path not found: {path}"
        max_results = _clamp(max_results, 1, GLOB_MAX_RESULTS_CAP)
        try:
            up, parts = _glob_segments(pattern)
        except (ValueError, NotImplementedError) as exc:
            return f"Error: invalid glob pattern: {exc}"
        # The pattern is walked beneath the workspace through no link
        # (_glob_walk), from the base folder -- moved up by the pattern's
        # leading ``..`` segments, never above the workspace
        root = os.path.realpath(workspace)
        relative = os.path.relpath(resolved, root)
        base_parts = [] if relative == os.curdir else relative.split(os.sep)
        entries: list[tuple[float, str]] = []
        cut = {"folders": 0}
        if up <= len(base_parts):
            base_parts = base_parts[: len(base_parts) - up]
            try:
                base = _open_resolved(workspace, os.path.join(root, *base_parts), _PATH | _DIRECTORY)
            except OSError:
                return f"Error: Path not found: {path}"
            try:
                _glob_walk(base, os.path.join(*base_parts) if base_parts else "", parts, entries, cut)
            finally:
                os.close(base)
        entries.sort(key=lambda e: (-e[0], e[1]))
        truncated = len(entries) > max_results
        shown = entries[:max_results]
        header = f"{len(shown)} file(s)"
        if truncated:
            header += " [truncated]"
        if cut["folders"]:
            header += f" [{cut['folders']} folder(s) deeper than {GREP_MAX_DEPTH} levels not searched]"
        if not shown:
            return header
        return header + "\n" + "\n".join(rel for _mtime, rel in shown)

    def ls(self, path: str = ".", *, max_entries: int = 200) -> str:
        """List a workspace directory: 'type size name', dirs first, name-sorted."""
        self._check_active()
        ok, workspace, resolved, err = _resolve_workspace_path(
            self._mgr, self._session_id, path
        )
        if not ok:
            return f"Error: {err}"
        # The folder is opened beneath the workspace through no link, listed
        # through that descriptor, and each entry read itself (lstat)
        try:
            folder = _open_resolved(workspace, resolved, os.O_RDONLY | _DIRECTORY)
        except (FileNotFoundError, NotADirectoryError):
            return f"Error: Path not found: {path}"
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                return f"Error: Path not found: {path}"
            return f"Error listing directory: {exc}"
        max_entries = _clamp(max_entries, 1, LS_MAX_ENTRIES_CAP)
        try:
            try:
                with os.scandir(folder) as it:
                    names = sorted(entry.name for entry in it)
            except OSError as exc:
                return f"Error listing directory: {_named(exc, resolved)}"
            dirs: list[str] = []
            files: list[tuple[str, int]] = []
            for name in names:
                try:
                    st = os.stat(name, dir_fd=folder, follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISLNK(st.st_mode):
                    continue  # symlinks are skipped in every walk
                if stat.S_ISDIR(st.st_mode):
                    dirs.append(name)
                elif stat.S_ISREG(st.st_mode):
                    files.append((name, st.st_size))
        finally:
            os.close(folder)
        lines = [f"dir 0 {name}" for name in dirs]
        lines.extend(f"file {size} {name}" for name, size in files)
        if not lines:
            return "0 entries"
        truncated = len(lines) > max_entries
        shown = lines[:max_entries]
        if truncated:
            shown.append(f"[truncated at {max_entries} entries]")
        return "\n".join(shown)

    # -----------------------------------------------------------------
    # ToolDefinition generation (for tool_registry)
    # -----------------------------------------------------------------

    def get_tool_definitions(self) -> list:
        """Generate ToolDefinition instances with session_id pre-bound.

        Returns 4 tools: bash, view, create_file, str_replace.
        These are the simplified versions the LLM will see.
        """
        if not TOOL_REGISTRY_AVAILABLE:
            return []

        return [
            self._make_bash_def(),
            self._make_view_def(),
            self._make_create_file_def(),
            self._make_str_replace_def(),
        ]

    def _make_bash_def(self) -> "ToolDefinition":
        """ToolDefinition for bash (session pre-bound)."""
        def handler(command: str, timeout: int = 30) -> str:
            return self.bash(command, timeout)

        return ToolDefinition(
            name="bash",
            description=(
                "Execute a bash command in the isolated sandbox workspace. "
                "The sandbox has no access to the host filesystem or network. "
                "Working directory is /workspace/."
            ),
            parameters={
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
            requires=[],
            enabled=True,
        )

    def _make_view_def(self) -> "ToolDefinition":
        """ToolDefinition for view (session pre-bound)."""
        def handler(
            path: str,
            start_line: int = 0,
            end_line: int = 0,
        ) -> str:
            return self.view(path, start_line, end_line)

        return ToolDefinition(
            name="view",
            description=(
                "Read file contents with line numbers, or list a directory. "
                "Use start_line/end_line for a specific range (1-indexed). "
                "Paths are relative to /workspace/."
            ),
            parameters={
                "path": ToolParam(
                    name="path",
                    type="string",
                    description="File or directory path",
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
                    description="Last line to show (0 = end)",
                    required=False,
                    default=0,
                ),
            },
            handler=handler,
            requires=[],
            enabled=True,
        )

    def _make_create_file_def(self) -> "ToolDefinition":
        """ToolDefinition for create_file (session pre-bound)."""
        def handler(path: str, content: str) -> str:
            return self.create_file(path, content)

        return ToolDefinition(
            name="create_file",
            description=(
                "Create or overwrite a file in the sandbox workspace. "
                "Parent directories are created automatically. "
                "Paths are relative to /workspace/."
            ),
            parameters={
                "path": ToolParam(
                    name="path",
                    type="string",
                    description="File path to create",
                    required=True,
                ),
                "content": ToolParam(
                    name="content",
                    type="string",
                    description="File content to write",
                    required=True,
                ),
            },
            handler=handler,
            requires=[],
            enabled=True,
        )

    def _make_str_replace_def(self) -> "ToolDefinition":
        """ToolDefinition for str_replace (session pre-bound)."""
        def handler(
            path: str,
            old_str: str,
            new_str: str = "",
        ) -> str:
            return self.str_replace(path, old_str, new_str)

        return ToolDefinition(
            name="str_replace",
            description=(
                "Find and replace a unique string in a file. "
                "The search string must appear exactly once. "
                "Use new_str='' to delete the match."
            ),
            parameters={
                "path": ToolParam(
                    name="path",
                    type="string",
                    description="File path in the sandbox",
                    required=True,
                ),
                "old_str": ToolParam(
                    name="old_str",
                    type="string",
                    description="String to find (must be unique)",
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
            requires=[],
            enabled=True,
        )

    # -----------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------

    def _check_active(self) -> None:
        """Raise if no active session."""
        if not self.active:
            raise RuntimeError("No active sandbox session")

    # -- Diagnostics-after-write --

    def _reset_diag_cache(self) -> None:
        """Clear the per-session diagnostics probe cache."""
        self._diag_probed = False
        self._diag_tool = None
        self._type_probed = False
        self._type_tool = None

    def _backend_is_bwrap(self) -> bool:
        """Whether the active session runs on the bwrap isolation backend.

        The whole diagnostics feature is gated on this: the in-sandbox ladder
        must never execute on a degraded backend, and gating the Svelte read
        too keeps the container's clean-path outputs byte-identical.
        """
        backend = getattr(self._session, "isolation_backend", None)
        value = getattr(backend, "value", backend)
        return value == "bwrap"

    def _maybe_diagnostics(self, path: str, write_result: str) -> str:
        """The ``[diagnostics]`` suffix block for a successful write, or ''.

        Fail-silent by contract: any unavailability, error, or timeout is a
        logger.debug skip, never an output change and never a fabricated
        success. Only findings produce a non-empty return.
        """
        try:
            if not write_result or write_result.startswith("Error"):
                return ""
            ext = os.path.splitext(path)[1].lower()
            if ext not in (".py", ".svelte"):
                return ""
            if not self._backend_is_bwrap():
                return ""
            cfg = load_diagnostics_config()
            if not cfg.get("enabled", True):
                return ""
            ok, workspace, resolved, _err = _resolve_workspace_path(
                self._mgr, self._session_id, path
            )
            if not ok or not os.path.isfile(resolved):
                return ""
            if ext == ".svelte":
                # Trusted host-side read of the workspace copy, beneath the
                # workspace through no link; no execution.
                try:
                    text = _read_resolved(workspace, resolved).decode(
                        "utf-8", errors="replace"
                    )
                except OSError:
                    return ""
                findings = svelte_tag_findings(text)
            else:
                rel = os.path.relpath(resolved, workspace)
                findings = self._run_python_ladder(rel, cfg)
                # Complementary type-aware pass (AGT borrow from OpenCode's
                # LSP feedback): mypy surfaces semantic/type errors the
                # linter ladder cannot. Merged into the same suffix block.
                findings = findings + self._run_python_type_check(rel, cfg)
            if not findings:
                return ""
            return self._format_diag_block(findings, cfg)
        except Exception:  # the write result must never be endangered
            logger.debug("diagnostics pass skipped", exc_info=True)
            return ""

    def _probe_diag_tool(self, cfg: dict[str, Any]) -> str | None:
        """First available ladder tool, probed once per session and cached.

        Probes run inside the sandbox via ``execute_command`` (validator and
        signed audit log included), so the audit chain sees them like any
        other execution.
        """
        if self._diag_probed:
            return self._diag_tool
        self._diag_probed = True
        self._diag_tool = None
        for tool in cfg.get("tools", []):
            probe = _DIAG_PROBE_COMMANDS.get(tool)
            if probe is None:
                continue
            try:
                result = self._mgr.execute_command(
                    self._session_id, probe, timeout=int(cfg["timeout_s"])
                )
            except Exception:
                logger.debug("diagnostics probe failed for %s", tool, exc_info=True)
                continue
            if getattr(result, "blocked", False) or getattr(result, "timed_out", False):
                continue
            if getattr(result, "return_code", 1) == 0:
                self._diag_tool = tool
                break
        return self._diag_tool

    def _run_python_ladder(self, rel: str, cfg: dict[str, Any]) -> list[str]:
        """Run the probed linter on the workspace copy, inside the sandbox."""
        tool = self._probe_diag_tool(cfg)
        if tool is None:
            return []
        quoted = shlex.quote(rel)
        if tool == "ruff":
            cmd = f"ruff check --quiet -- {quoted}"
        elif tool == "pyflakes":
            cmd = f"pyflakes {quoted}"
        else:
            cmd = f"python3 -m py_compile {quoted}"
        try:
            result = self._mgr.execute_command(
                self._session_id, cmd, timeout=int(cfg["timeout_s"])
            )
        except Exception:
            logger.debug("diagnostics run failed for %s", rel, exc_info=True)
            return []
        if getattr(result, "blocked", False) or getattr(result, "timed_out", False):
            return []
        rc = getattr(result, "return_code", 0)
        if rc == 0:
            return []
        text = (getattr(result, "stdout", "") or "") + "\n" + (
            getattr(result, "stderr", "") or ""
        )
        lines = [line for line in text.splitlines() if line.strip()]
        return lines or [f"{tool} exited {rc}"]

    def _probe_type_tool(self, cfg: dict[str, Any]) -> str | None:
        """Probe the optional type-checker (mypy), once per session, cached.

        Mirrors _probe_diag_tool: the probe runs inside the sandbox via
        execute_command (validator and signed audit log included), so the
        audit chain sees it like any other execution.
        """
        if self._type_probed:
            return self._type_tool
        self._type_probed = True
        self._type_tool = None
        tool = cfg.get("type_tool")
        if not tool:
            return None
        probe = _DIAG_PROBE_COMMANDS.get(tool)
        if probe is None:
            return None
        try:
            result = self._mgr.execute_command(
                self._session_id, probe, timeout=int(cfg["timeout_s"])
            )
        except Exception:
            logger.debug("type-check probe failed for %s", tool, exc_info=True)
            return None
        if getattr(result, "blocked", False) or getattr(result, "timed_out", False):
            return None
        if getattr(result, "return_code", 1) == 0:
            self._type_tool = tool
        return self._type_tool

    def _run_python_type_check(self, rel: str, cfg: dict[str, Any]) -> list[str]:
        """Complementary type-aware pass (mypy) on the workspace copy.

        Runs IN ADDITION to the linter ladder: a type checker surfaces
        semantic errors (wrong types, bad signatures, undefined attributes)
        that ruff/pyflakes/py_compile do not. The disposable sandbox holds
        only the edited file, so missing-import noise is suppressed to keep
        findings about THIS file. Fail-silent like the rest of the path.
        """
        tool = self._probe_type_tool(cfg)
        if tool is None:
            return []
        quoted = shlex.quote(rel)
        cmd = (
            "mypy --no-error-summary --no-color-output "
            "--ignore-missing-imports --follow-imports=silent "
            f"--show-column-numbers {quoted}"
        )
        try:
            result = self._mgr.execute_command(
                self._session_id, cmd, timeout=int(cfg["timeout_s"])
            )
        except Exception:
            logger.debug("type-check run failed for %s", rel, exc_info=True)
            return []
        if getattr(result, "blocked", False) or getattr(result, "timed_out", False):
            return []
        if getattr(result, "return_code", 0) == 0:
            return []
        text = (getattr(result, "stdout", "") or "") + "\n" + (
            getattr(result, "stderr", "") or ""
        )
        return [line for line in text.splitlines() if line.strip()]

    def _format_diag_block(self, findings: list[str], cfg: dict[str, Any]) -> str:
        """Assemble the capped suffix block (count, lines, truncation marker)."""
        total = len(findings)
        max_findings = int(cfg.get("max_findings", 25))
        shown = list(findings[:max_findings])
        if total > max_findings:
            shown.append(_DIAG_TRUNCATED_MARKER)
        header = f"\n\n[diagnostics] {total} finding(s):\n"
        block = header + "\n".join(shown)
        cap = int(cfg.get("max_block_bytes", 4096))
        while len(block.encode("utf-8")) > cap and len(shown) > 1:
            if shown[-1] == _DIAG_TRUNCATED_MARKER:
                shown.pop(-2)
            else:
                shown.pop()
                shown.append(_DIAG_TRUNCATED_MARKER)
            block = header + "\n".join(shown)
        return block


# ---------------------------------------------------------------------------
# Module-level availability
# ---------------------------------------------------------------------------

SANDBOX_TOOLS_AVAILABLE = (
    SANDBOX_AVAILABLE
    and FILE_TOOLS_AVAILABLE
    and TOOL_REGISTRY_AVAILABLE
)
