#!/usr/bin/env python3
"""
SANDBOX TOOLS - OPTI-OIGNON v1.7.5 (S73)
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

Usage in S74's coding_agent:
    session = SandboxToolSession(sandbox_manager)
    session.start("coding-task-1")
    # Register session.get_tool_definitions() in tool_registry
    # LLM uses tools without knowing about sessions
    session.stop()

Author: Leon
"""

import fnmatch
import logging
import os
import re
import shlex
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

# Guarded YAML import for the [agent.diagnostics] config block (S228). The
# loader falls back to the built-in defaults when PyYAML or the file is
# unavailable, so behaviour never depends on the file being present.
try:
    import yaml as _yaml
except ImportError:  # pragma: no cover - defensive guard
    _yaml = None

# ---------------------------------------------------------------------------
# S228 (AGT Lot 1) constants: read-only workspace tools and diagnostics
# ---------------------------------------------------------------------------

# Hard caps for the read-only tools (AGT_SPEC 5.1). Requested values are
# clamped into [1, cap]; defaults live on the schemas in agent/tools.py.
GREP_MAX_RESULTS_CAP = 500
GREP_CONTEXT_CAP = 5
GLOB_MAX_RESULTS_CAP = 1000
LS_MAX_ENTRIES_CAP = 1000

# Files larger than this are skipped by grep (the view 1 MiB posture).
MAX_GREP_FILE_SIZE = 1024 * 1024

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
    "timeout_s": 10,
    "max_block_bytes": 4096,
    "max_findings": 25,
}
_DIAG_PROBE_COMMANDS = {
    "ruff": "command -v ruff",
    "pyflakes": "command -v pyflakes",
    "py_compile": "command -v python3",
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
_TAG_RE = re.compile(
    r"<\s*(/?)([A-Za-z][A-Za-z0-9:._-]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*?)(/?)\s*>"
)
_SCRIPT_STYLE_RE = re.compile(
    r"(<(script|style)\b[^>]*>).*?(</\2\s*>)", re.IGNORECASE | re.DOTALL
)


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


def _is_binary_file(abs_path: str) -> bool:
    """Null-byte sniff: True when the first chunk contains a NUL byte."""
    try:
        with open(abs_path, "rb") as fh:
            return b"\x00" in fh.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True  # unreadable counts as skipped, never as a crash


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(int(value), hi))


def load_diagnostics_config() -> dict[str, Any]:
    """The [agent.diagnostics] block of agent/config.yaml, defaults applied.

    A tiny guarded reader local to this module (the agent config_loader is
    deliberately not edited at S228): a missing file, missing PyYAML, a
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


def _strip_for_tag_scan(text: str) -> str:
    """Reduce Svelte source to scannable markup (heuristic, documented).

    Comments and the inner content of script/style elements are removed (the
    tags themselves are kept), and simple ``{...}`` template expressions that
    are not block markers are blanked so comparison operators inside them do
    not read as tags. The checker is a conservative heuristic; the
    host-assured pass exercises it on real Svelte trees.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = _SCRIPT_STYLE_RE.sub(lambda m: m.group(1) + m.group(3), text)
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
    for closing, name, _attrs, selfclose in _TAG_RE.findall(scannable):
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
    str_replace — without needing to know about sessions.

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
        # S228 diagnostics: the linter ladder is probed once per session and
        # cached; the cache is cleared on every lifecycle change.
        self._diag_probed: bool = False
        self._diag_tool: str | None = None

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
        allow_degraded: bool = False,
    ) -> str:
        """Start a new sandbox session.

        Args:
            session_id: Custom ID, or auto-generated UUID if None.
            allow_degraded: Allow tempdir mode without confirmation.

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
        self._session = self._mgr.create_sandbox(
            sid, allow_degraded=allow_degraded
        )
        self._session_id = sid
        self._reset_diag_cache()

        # SECURITY: Disable unsafe unsandboxed tools in the registry
        # so the LLM cannot bypass the sandbox.
        self._apply_registry_lockout()

        logger.info("SandboxToolSession started: %s", sid)
        return sid

    def attach(self, session_id: str) -> str:
        """Bind this wrapper to an EXISTING managed workspace (S210).

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
        """Release an attached workspace WITHOUT destroying it (S210).

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

        Factored from start() at S210 so attach() enforces the identical
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

        S228: when the write succeeds and the filename matches the
        diagnostics suffix map, an in-sandbox linter pass (or the host-side
        Svelte tag-balance read) may append a ``[diagnostics]`` block; a
        clean write returns byte-identical output to the pre-S228 shape.
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

        S228: a successful edit may append a ``[diagnostics]`` block exactly
        as for ``create_file``; the clean path is byte-identical.
        """
        self._check_active()
        result = _handle_sandbox_str_replace(
            self._session_id, path, old_str, new_str,
            _sandbox_manager=self._mgr,
        )
        return result + self._maybe_diagnostics(path, result)

    # -----------------------------------------------------------------
    # S228 read-only workspace tools (AGT_SPEC 5.1): grep, glob, ls.
    # Trusted host-side reads on the view precedent: path-confined via
    # validate_sandbox_path, active session required, deterministic sorted
    # output, truncated flags, null-byte binary sniff, 1 MiB skip.
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
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                rx = re.compile(pattern, flags)
            except re.error as exc:
                return f"Error: invalid regex pattern: {exc}"

            def matches(line: str) -> bool:
                return rx.search(line) is not None
        else:
            needle = pattern if case_sensitive else pattern.casefold()

            def matches(line: str) -> bool:
                hay = line if case_sensitive else line.casefold()
                return needle in hay

        files = self._grep_candidates(workspace, resolved, glob)
        if files is None:
            return f"Error: Path not found: {path}"

        match_lines: list[str] = []
        matched_files: set[str] = set()
        skipped = 0
        truncated = False
        count = 0
        for rel, abs_path in files:
            if truncated:
                break
            try:
                size = os.path.getsize(abs_path)
            except OSError:
                skipped += 1
                continue
            if size > MAX_GREP_FILE_SIZE or _is_binary_file(abs_path):
                skipped += 1
                continue
            try:
                with open(abs_path, encoding="utf-8", errors="replace") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                skipped += 1
                continue
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

        header = f"{count} match(es) in {len(matched_files)} file(s)"
        if truncated:
            header += " [truncated]"
        if skipped:
            header += f" [{skipped} file(s) skipped: binary or >1 MiB]"
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
        if not os.path.isdir(resolved):
            return f"Error: Path not found: {path}"
        max_results = _clamp(max_results, 1, GLOB_MAX_RESULTS_CAP)
        workspace_real = os.path.realpath(workspace)
        try:
            candidates = list(Path(resolved).glob(pattern))
        except (ValueError, NotImplementedError) as exc:
            return f"Error: invalid glob pattern: {exc}"
        entries: list[tuple[float, str]] = []
        for cand in candidates:
            try:
                if cand.is_symlink() or not cand.is_file():
                    continue
                real = os.path.realpath(str(cand))
                # Defense-in-depth: results must stay inside the workspace.
                if not real.startswith(workspace_real + os.sep) and real != workspace_real:
                    continue
                rel = os.path.relpath(str(cand), workspace)
                entries.append((os.path.getmtime(str(cand)), rel))
            except OSError:
                continue
        entries.sort(key=lambda e: (-e[0], e[1]))
        truncated = len(entries) > max_results
        shown = entries[:max_results]
        header = f"{len(shown)} file(s)"
        if truncated:
            header += " [truncated]"
        if not shown:
            return header
        return header + "\n" + "\n".join(rel for _mtime, rel in shown)

    def ls(self, path: str = ".", *, max_entries: int = 200) -> str:
        """List a workspace directory: 'type size name', dirs first, name-sorted."""
        self._check_active()
        ok, _workspace, resolved, err = _resolve_workspace_path(
            self._mgr, self._session_id, path
        )
        if not ok:
            return f"Error: {err}"
        if not os.path.isdir(resolved):
            return f"Error: Path not found: {path}"
        max_entries = _clamp(max_entries, 1, LS_MAX_ENTRIES_CAP)
        try:
            names = sorted(os.listdir(resolved))
        except OSError as exc:
            return f"Error listing directory: {exc}"
        dirs: list[str] = []
        files: list[tuple[str, int]] = []
        for name in names:
            full = os.path.join(resolved, name)
            if os.path.islink(full):
                continue  # symlinks are skipped in every S228 walk
            if os.path.isdir(full):
                dirs.append(name)
            elif os.path.isfile(full):
                try:
                    files.append((name, os.path.getsize(full)))
                except OSError:
                    continue
        lines = [f"dir 0 {name}" for name in dirs]
        lines.extend(f"file {size} {name}" for name, size in files)
        if not lines:
            return "0 entries"
        truncated = len(lines) > max_entries
        shown = lines[:max_entries]
        if truncated:
            shown.append(f"[truncated at {max_entries} entries]")
        return "\n".join(shown)

    def _grep_candidates(
        self, workspace: str, resolved: str, glob_filter: str
    ) -> list[tuple[str, str]] | None:
        """The deterministic (relpath, abspath) walk grep scans, or None.

        A file target yields itself; a directory is walked depth-first with
        sorted entries, symlinks never followed and never read; an optional
        fnmatch filter applies to the workspace-relative path.
        """
        if os.path.isfile(resolved):
            rel = os.path.relpath(resolved, workspace)
            return [(rel, resolved)]
        if not os.path.isdir(resolved):
            return None
        out: list[tuple[str, str]] = []
        for root, dirnames, filenames in os.walk(resolved, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames):
                full = os.path.join(root, name)
                if os.path.islink(full):
                    continue
                rel = os.path.relpath(full, workspace)
                if glob_filter and not fnmatch.fnmatch(rel, glob_filter):
                    continue
                out.append((rel, full))
        out.sort(key=lambda e: e[0])
        return out

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

    # -- S228 diagnostics-after-write (AGT_SPEC 5.2) --

    def _reset_diag_cache(self) -> None:
        """Clear the per-session diagnostics probe cache."""
        self._diag_probed = False
        self._diag_tool = None

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
                # Trusted host-side read of the workspace copy; no execution.
                try:
                    with open(resolved, encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    return ""
                findings = svelte_tag_findings(text)
            else:
                rel = os.path.relpath(resolved, workspace)
                findings = self._run_python_ladder(rel, cfg)
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
