#!/usr/bin/env python3
"""
CODE EXECUTOR -- OPTI-OIGNON 1.4.0 (F3)

Sandboxed code execution for Python, R, and Bash.

Every run is a client of the server's sandbox manager, the runner the agent's
tools use: the script is written into a sandbox workspace by the shared file
handler, so the command validator reads it, and runs through
``SandboxManager.execute_command``. There is no other runner. Without a usable
sandbox a run is refused, never moved to this machine:
- Off unless the user turns on the ``code_execution`` setting
- Timeout enforcement (default 30s, at most 120s)
- Output, memory and file size limits: the sandbox's own
- One sandbox per run, destroyed after it; one per conversation in
  persistent mode, destroyed by a reset
- Output images copied out only as regular files, never through a link

Architecture:
    - CodeBlock: dataclass for a parsed code block from LLM output
    - ExecutionResult: dataclass for execution outcome
    - CodeExecutor: main engine (execute, detect_language, extract_code_blocks)
    - Module-level singleton: code_executor

Author: Leon
"""

import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CodeBlock:
    """A fenced code block parsed from LLM output."""
    code: str
    language: str
    start_pos: int
    end_pos: int


@dataclass
class ExecutionResult:
    """Result of a code execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    language: str
    truncated: bool = False
    error_message: str = ""
    output_files: list[str] = field(default_factory=list)
    working_dir: str = ""
    # False when the code never ran: execution is off, the language or its
    # runtime is missing, or the sandbox is unavailable or refused the run.
    ran: bool = True


# Regex to match fenced code blocks.
# Handles: optional leading spaces, language tag with special chars (c++, c#),
# optional newline after language tag, alternative fence styles.
_CODE_BLOCK_RE = re.compile(
    r"[ \t]*```([^\n`]*?)[ \t]*\n(.*?)[ \t]*```",
    re.DOTALL,
)

# Image file extensions to detect as output
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".pdf"}

# Script files to exclude from output detection
_SCRIPT_FILES = {"script.py", "script.R", "script.sh"}

# Stable output directory for images from ephemeral executions
_OUTPUT_DIR = None

def _get_output_dir() -> str:
    """Get or create a stable directory for execution output files."""
    global _OUTPUT_DIR
    if _OUTPUT_DIR is None or not os.path.isdir(_OUTPUT_DIR):
        try:
            from .config import DATA_DIR
            _OUTPUT_DIR = os.path.join(DATA_DIR, "exec_outputs")
        except ImportError:
            _OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "opti_exec_outputs")
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR

# The user setting that turns code execution on, and its value when it is
# unset or cannot be read. The settings screen writes it.
CODE_EXECUTION_SETTING = "code_execution"
CODE_EXECUTION_DEFAULT = False

_NO_SANDBOX = "Code execution needs the sandbox, which is off or unavailable."


def _user_setting(key: str, default):
    """A user setting, or ``default`` when it is unset or cannot be read."""
    try:
        from .config import config
        return config.get_user_preference(key, default)
    except Exception:
        return default


def _server_sandbox():
    """The server's sandbox manager, or None when the sandbox is off or missing."""
    try:
        from .sandbox_manager import sandbox_manager
    except Exception:
        return None
    return sandbox_manager


def _write_script(manager, session_id: str, name: str, code: str) -> str:
    """Write a script into a sandbox workspace through the shared file handler.

    The handler validates the path, holds its size limit and registers the
    content with the command validator, as it does for every sandbox client.
    Returns "" once the script is written, or the handler's error.
    """
    from .file_tools import _handle_sandbox_create_file
    reply = _handle_sandbox_create_file(session_id, name, code, _sandbox_manager=manager)
    return "" if reply.startswith("File created:") else reply

# Language aliases for normalization
_LANGUAGE_ALIASES = {
    "python": "python",
    "python3": "python",
    "py": "python",
    "r": "r",
    "rlang": "r",
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
}

# Heuristics for language detection when no language tag is given
_PYTHON_INDICATORS = [
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",
    r"\bdef\s+\w+\s*\(",
    r"\bclass\s+\w+",
    r"\bprint\s*\(",
    r"\bif\s+__name__\b",
    r"\bpd\.DataFrame\b",
    r"\bnp\.\w+",
    r"^\s*#\s*!.*python",
]

_R_INDICATORS = [
    r"\blibrary\s*\(",
    r"\brequire\s*\(",
    r"<-\s*\w+",
    r"\w+\s*<-",
    r"\bc\s*\(",
    r"\bdata\.frame\s*\(",
    r"\bggplot\s*\(",
    r"\bmutate\s*\(",
    r"\bfilter\s*\(",
    r"\bpipe\b|%>%",
    r"^\s*#\s*!.*Rscript",
]

_BASH_INDICATORS = [
    r"^\s*#!/bin/(ba)?sh",
    r"\bsudo\s+",
    r"\bapt(-get)?\s+",
    r"\bpip\s+install\b",
    r"\becho\s+",
    r"\bcd\s+",
    r"\bls\b",
    r"\bgrep\b",
    r"\bawk\b",
    r"\bsed\b",
    r"\bcat\s+",
    r"\bchmod\b",
    r"\bmkdir\b",
]


class CodeExecutor:
    """Execute Python, R, and Bash code inside the server's sandbox."""

    SUPPORTED_LANGUAGES = {"python", "r", "bash"}

    # Safety limits (output, memory and file size are the sandbox's own)
    DEFAULT_TIMEOUT = 30       # seconds
    MAX_TIMEOUT = 120          # absolute maximum

    def __init__(self, sandbox_mgr=None):
        self._sandbox_mgr = sandbox_mgr  # None: the server's own, looked up per run
        self._persistent_mode = False  # one sandbox per conversation
        self._persistent_sessions = {}  # conv_id -> sandbox session id
        self._sessions_lock = threading.Lock()
        self._detect_available_languages()

    def _detect_available_languages(self):
        """Check which language runtimes are available on the system."""
        self._available = {}
        # Python
        for cmd in ["python3", "python"]:
            if shutil.which(cmd):
                self._available["python"] = cmd
                break
        # R
        if shutil.which("Rscript"):
            self._available["r"] = "Rscript"
        # Bash
        for cmd in ["/bin/bash", "/usr/bin/bash"]:
            if os.path.isfile(cmd) and os.access(cmd, os.X_OK):
                self._available["bash"] = cmd
                break
        if "bash" not in self._available and shutil.which("bash"):
            self._available["bash"] = shutil.which("bash")
        logger.info(f"Code executor: available languages = {list(self._available.keys())}")

    @property
    def enabled(self) -> bool:
        """Whether the user turned code execution on; only a real True does."""
        return _user_setting(CODE_EXECUTION_SETTING, CODE_EXECUTION_DEFAULT) is True

    @property
    def persistent_mode(self) -> bool:
        """Whether each conversation keeps one sandbox across runs."""
        return self._persistent_mode

    @persistent_mode.setter
    def persistent_mode(self, value: bool):
        self._persistent_mode = bool(value)
        if not value:
            self.cleanup_all_persistent_dirs()

    def _manager(self):
        """The sandbox manager runs go to, or None when there is none."""
        return self._sandbox_mgr if self._sandbox_mgr is not None else _server_sandbox()

    @staticmethod
    def _destroy(manager, session_id: str):
        try:
            manager.destroy_sandbox(session_id)
        except Exception as e:
            logger.warning(f"Could not destroy sandbox {session_id}: {e}")

    def _conversation_session(self, manager, conv_id: str) -> str:
        """The live sandbox of a conversation, made on its first run."""
        with self._sessions_lock:
            session_id = self._persistent_sessions.get(conv_id)
            if session_id is not None and manager.get_workspace_path(session_id):
                return session_id
            session = manager.create_sandbox(None, label="code execution")
            self._persistent_sessions[conv_id] = session.session_id
            logger.info(f"Created persistent sandbox for {conv_id[:8]}")
            return session.session_id

    def reset_persistent_dir(self, conv_id: str) -> bool:
        """Destroy the sandbox of a conversation.

        Returns:
            True if a sandbox was destroyed, False if none existed.
        """
        with self._sessions_lock:
            session_id = self._persistent_sessions.pop(conv_id, None)
        manager = self._manager()
        if session_id is None or manager is None:
            return False
        self._destroy(manager, session_id)
        logger.info(f"Reset persistent sandbox for {conv_id[:8]}")
        return True

    def cleanup_all_persistent_dirs(self):
        """Destroy the sandbox of every conversation."""
        with self._sessions_lock:
            session_ids = list(self._persistent_sessions.values())
            self._persistent_sessions.clear()
        manager = self._manager()
        if manager is None:
            return
        for session_id in session_ids:
            self._destroy(manager, session_id)
        if session_ids:
            logger.info(f"Cleaned up {len(session_ids)} persistent sandboxes")

    def list_persistent_files(self, conv_id: str) -> list[str]:
        """List the regular files in the sandbox of a conversation.

        Returns:
            List of filenames (not full paths), or empty list.
        """
        with self._sessions_lock:
            session_id = self._persistent_sessions.get(conv_id)
        manager = self._manager()
        if session_id is None or manager is None:
            return []
        workspace = manager.get_workspace_path(session_id)
        if not workspace:
            return []
        return sorted(
            name for name in self._regular_files(workspace)
            if not name.startswith("script.")
        )

    @staticmethod
    def _regular_files(directory: str) -> list[str]:
        """Names of the regular files directly in ``directory``; a link is never followed."""
        try:
            names = os.listdir(directory)
        except OSError:
            return []
        found = []
        for name in names:
            try:
                mode = os.lstat(os.path.join(directory, name)).st_mode
            except OSError:
                continue
            if stat.S_ISREG(mode):
                found.append(name)
        return found

    def get_available_languages(self) -> list[str]:
        """Return list of languages with available runtimes."""
        return list(self._available.keys())

    def is_language_available(self, language: str) -> bool:
        """Check if a specific language runtime is installed."""
        lang = self._normalize_language(language)
        return lang in self._available

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
        conv_id: str | None = None,
    ) -> ExecutionResult:
        """Execute code inside a sandbox session and return the result.

        Args:
            code: source code to execute
            language: one of python/r/bash (or alias)
            timeout: max seconds (None = DEFAULT_TIMEOUT, capped at MAX_TIMEOUT)
            conv_id: if provided and persistent_mode is on, reuse its sandbox

        Returns:
            ExecutionResult with stdout, stderr, timing, etc.; ``ran`` is
            False when the code never ran.
        """
        start_time = time.monotonic()
        language = self._normalize_language(language)

        if not self.enabled:
            return self._refusal(language, "Code execution is disabled. Enable it in Settings.")

        if language not in self.SUPPORTED_LANGUAGES:
            return self._refusal(language, f"Unsupported language: {language}")

        if language not in self._available:
            return self._refusal(
                language,
                f"Runtime not found for {language}. "
                f"Available: {', '.join(self._available.keys()) or 'none'}",
            )

        manager = self._manager()
        if manager is None:
            return self._refusal(language, _NO_SANDBOX)

        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT
        timeout = min(timeout, self.MAX_TIMEOUT)

        # The conversation's own sandbox, or one made for this run only
        use_persistent = self._persistent_mode and bool(conv_id)
        try:
            if use_persistent:
                session_id = self._conversation_session(manager, conv_id)
            else:
                session_id = manager.create_sandbox(None, label="code execution").session_id
        except Exception as e:
            return self._refusal(language, f"The sandbox could not open a session for this run: {e}")

        try:
            result = self._run_in_sandbox(
                manager, session_id, code, language, timeout, copy_out=not use_persistent,
            )
        except Exception as e:
            logger.exception(f"Code execution failed: {e}")
            result = self._refusal(language, f"Internal error: {e}")
        finally:
            if not use_persistent:
                self._destroy(manager, session_id)
        result.execution_time = time.monotonic() - start_time
        return result

    @staticmethod
    def _refusal(language: str, reason: str) -> ExecutionResult:
        """The result of a run that never happened."""
        return ExecutionResult(
            success=False, stdout="", stderr="",
            return_code=-1, execution_time=0.0,
            language=language,
            error_message=reason,
            ran=False,
        )

    def _detect_output_files(
        self, workspace: str, files_before: set,
    ) -> list[str]:
        """Find new image files created during execution.

        Only regular files count: a link is never followed out of the
        workspace, whatever it points at.

        Args:
            workspace: the sandbox workspace
            files_before: set of filenames present before execution

        Returns:
            List of absolute paths to new output files (images only).
        """
        new_names = set(self._regular_files(workspace)) - files_before - _SCRIPT_FILES
        output_paths = []
        for name in sorted(new_names):
            ext = os.path.splitext(name)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                output_paths.append(os.path.join(workspace, name))
        return output_paths

    @staticmethod
    def _copy_out(paths: list[str]) -> list[str]:
        """Copy images out of a sandbox that is about to go.

        Each file is opened without following a link, and copied only while
        it is a regular file with a single name; anything else stays behind.
        """
        output_dir = _get_output_dir()
        stable_paths = []
        for fpath in paths:
            fname = os.path.basename(fpath)
            # Add timestamp prefix to avoid collisions
            stable_path = os.path.join(output_dir, f"{int(time.time())}_{fname}")
            try:
                fd = os.open(fpath, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            except OSError as e:
                logger.debug(f"Could not open output file: {e}")
                continue
            with os.fdopen(fd, "rb") as src:
                try:
                    st = os.fstat(src.fileno())
                    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                        continue
                    with open(stable_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    stable_paths.append(stable_path)
                except OSError as e:
                    logger.debug(f"Could not copy output file: {e}")
        return stable_paths

    @staticmethod
    def _detect_table_output(stdout: str) -> str | None:
        """Detect tabular output in stdout and convert to markdown table.

        Handles pandas DataFrame repr and simple aligned columns.
        Returns markdown table string or None if no table detected.
        """
        lines = stdout.strip().split("\n")
        if len(lines) < 2:
            return None

        # Detect pandas-style DataFrame output:
        # Has an index column (numbers), aligned whitespace columns
        # Pattern: lines with consistent column alignment
        # Check if first data line starts with a number or whitespace+number (index)
        # and has at least 2 columns of data

        # Strategy: find runs of lines that look tabular
        # A line is "tabular" if it has 2+ whitespace-separated fields
        tabular_runs = []
        current_run = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if len(current_run) >= 3:
                    tabular_runs.append(current_run)
                current_run = []
                continue

            # Split by 2+ spaces (common in DataFrame repr and column output)
            fields = re.split(r"\s{2,}", stripped)
            if len(fields) >= 2:
                current_run.append(stripped)
            else:
                if len(current_run) >= 3:
                    tabular_runs.append(current_run)
                current_run = []

        if len(current_run) >= 3:
            tabular_runs.append(current_run)

        if not tabular_runs:
            return None

        # Convert the longest tabular run to markdown
        longest = max(tabular_runs, key=len)

        # Split each line into columns using 2+ whitespace
        rows = []
        for line in longest:
            fields = re.split(r"\s{2,}", line.strip())
            rows.append(fields)

        if not rows:
            return None

        # Normalize column count
        max_cols = max(len(r) for r in rows)
        if max_cols < 2:
            return None

        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # Build markdown table
        # First row is header
        md_parts = []
        header = rows[0]
        md_parts.append("| " + " | ".join(header) + " |")
        md_parts.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows[1:]:
            md_parts.append("| " + " | ".join(row) + " |")

        return "\n".join(md_parts)

    def _run_in_sandbox(
        self,
        manager,
        session_id: str,
        code: str,
        language: str,
        timeout: int,
        copy_out: bool,
    ) -> ExecutionResult:
        """Write the script into the session's workspace and run it there."""
        workspace = manager.get_workspace_path(session_id) or ""
        files_before = set(os.listdir(workspace)) if os.path.isdir(workspace) else set()

        script, command = self._command(language)
        error = _write_script(manager, session_id, script, code)
        if error:
            return self._refusal(language, error)

        outcome = manager.execute_command(session_id, command, timeout=timeout)
        if outcome.blocked:
            return self._refusal(language, outcome.block_reason or "The sandbox refused the command.")

        if outcome.timed_out:
            result = ExecutionResult(
                success=False,
                stdout=outcome.stdout,
                stderr=f"Execution timed out after {timeout}s",
                return_code=-1,
                execution_time=float(timeout),
                language=language,
                truncated=outcome.truncated_stdout,
                error_message=f"Timeout: code exceeded {timeout}s limit",
            )
        else:
            result = ExecutionResult(
                success=(outcome.return_code == 0),
                stdout=outcome.stdout,
                stderr=outcome.stderr,
                return_code=outcome.return_code,
                execution_time=0.0,  # filled by caller
                language=language,
                truncated=outcome.truncated_stdout or outcome.truncated_stderr,
            )
        result.working_dir = workspace

        # Detect new output files (images)
        new_files = self._detect_output_files(workspace, files_before)
        if new_files and copy_out:
            # The sandbox goes with this run: copy images to the stable output dir
            result.output_files = self._copy_out(new_files)
        elif new_files:
            # Persistent mode: files stay in the conversation's sandbox
            result.output_files = new_files
        return result

    def _command(self, language: str) -> tuple[str, str]:
        """The script name and the command that runs it in the workspace.

        Python and Bash take the script as their first argument: the command
        validator reads a script it was told about only when the command
        names it right after the interpreter.
        """
        if language == "python":
            return "script.py", f"{self._available['python']} script.py"
        elif language == "r":
            return "script.R", "Rscript --vanilla script.R"
        elif language == "bash":
            return "script.sh", "bash script.sh"
        else:
            raise ValueError(f"No command builder for language: {language}")

    @staticmethod
    def _normalize_language(lang: str) -> str:
        """Normalize a language name/alias to canonical form."""
        if not lang:
            return "python"
        lang_lower = lang.strip().lower()
        return _LANGUAGE_ALIASES.get(lang_lower, lang_lower)

    def detect_language(self, code: str) -> str:
        """Auto-detect code language from content.

        Uses simple heuristic scoring: count indicator matches for each language.

        Returns:
            "python", "r", "bash", or "python" as default
        """
        scores = {"python": 0, "r": 0, "bash": 0}

        for pattern in _PYTHON_INDICATORS:
            if re.search(pattern, code, re.MULTILINE):
                scores["python"] += 1

        for pattern in _R_INDICATORS:
            if re.search(pattern, code, re.MULTILINE):
                scores["r"] += 1

        for pattern in _BASH_INDICATORS:
            if re.search(pattern, code, re.MULTILINE):
                scores["bash"] += 1

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "python"  # default fallback
        return best

    def extract_code_blocks(self, response: str) -> list[CodeBlock]:
        """Extract fenced code blocks from an LLM response.

        Matches patterns like:
            ```python
            print("hello")
            ```

        Returns:
            List of CodeBlock with code, language, positions
        """
        blocks = []
        for match in _CODE_BLOCK_RE.finditer(response):
            raw_lang = match.group(1).strip()
            code = match.group(2)
            # Strip trailing whitespace but keep leading (indentation matters)
            code = code.rstrip()

            # Normalize language; if empty, try auto-detect
            if raw_lang:
                language = self._normalize_language(raw_lang)
            else:
                language = self.detect_language(code)

            # Only include if it looks like executable code
            # Skip tiny blocks that are probably inline examples
            if len(code.strip()) < 3:
                continue

            # Skip blocks tagged with non-executable languages
            if language not in self.SUPPORTED_LANGUAGES:
                continue

            blocks.append(CodeBlock(
                code=code,
                language=language,
                start_pos=match.start(),
                end_pos=match.end(),
            ))
        return blocks

    def format_result(self, result: ExecutionResult) -> str:
        """Format an ExecutionResult as a readable markdown string for display.

        Includes:
        - Inline images for output files (matplotlib, ggplot, etc.)
        - Markdown tables for tabular stdout
        - Syntax-highlighted error blocks
        """
        lang_label = result.language.capitalize()
        time_str = f"{result.execution_time:.1f}s"
        parts = []

        if result.error_message:
            parts.append(f"**Code Execution -- {lang_label}**\n")
            parts.append(f"Error: {result.error_message}")
            return "\n".join(parts)

        status = "Success" if result.success else f"Failed (exit code {result.return_code})"
        parts.append(f"**Code Execution -- {lang_label}, {time_str}**\n")
        parts.append(f"Status: {status}")
        if result.truncated:
            parts.append("(output was truncated)")

        if result.stdout.strip():
            # Try to detect and render tabular output
            table_md = self._detect_table_output(result.stdout)
            if table_md:
                parts.append(f"\n{table_md}")
                # If there are non-table lines, show them as raw output
                non_table_lines = []
                in_table = False
                for line in result.stdout.strip().split("\n"):
                    fields = re.split(r"\s{2,}", line.strip())
                    if len(fields) >= 2:
                        in_table = True
                    elif in_table:
                        in_table = False
                    if not in_table and line.strip():
                        non_table_lines.append(line)
                if non_table_lines:
                    parts.append(f"\n```\n{''.join(non_table_lines).rstrip()}\n```")
            else:
                parts.append(f"\n```\n{result.stdout.rstrip()}\n```")

        if result.stderr.strip():
            # Syntax-highlighted error output
            err_lang = "python" if result.language == "python" else "r" if result.language == "r" else ""
            label = "Warnings" if result.success else "Errors"
            parts.append(f"\n{label}:")
            parts.append(f"```{err_lang}\n{result.stderr.rstrip()}\n```")

        if not result.stdout.strip() and not result.stderr.strip():
            parts.append("\n(no output)")

        # Inline images for output files
        if result.output_files:
            parts.append("")
            for fpath in result.output_files:
                fname = os.path.basename(fpath)
                ext = os.path.splitext(fname)[1].lower()
                if ext == ".svg":
                    # SVG rendered inline if possible, otherwise as image
                    parts.append(f"![{fname}]({fpath})")
                elif ext == ".pdf":
                    parts.append(f"[{fname}]({fpath}) (PDF output)")
                else:
                    parts.append(f"![{fname}]({fpath})")

        return "\n".join(parts)


# Module-level singleton
code_executor = CodeExecutor()

# Convenience exports
execute_code = code_executor.execute
extract_code_blocks = code_executor.extract_code_blocks
detect_language = code_executor.detect_language
format_result = code_executor.format_result
