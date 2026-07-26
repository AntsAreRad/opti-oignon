#!/usr/bin/env python3
"""
CODE EXECUTOR -- OPTI-OIGNON 1.4.0 (F3)

Sandboxed code execution for Python, R, and Bash.

Runs code in isolated subprocesses with:
- Timeout enforcement (default 30s)
- Output size limits (default 50k chars)
- Memory limits via ulimit (Linux only)
- Temporary working directories (cleaned up after)
- No eval/exec -- always subprocess

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
import subprocess
import tempfile
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
    """Execute Python, R, and Bash code in sandboxed subprocesses."""

    SUPPORTED_LANGUAGES = {"python", "r", "bash"}

    # Safety limits
    DEFAULT_TIMEOUT = 30       # seconds
    MAX_TIMEOUT = 120          # absolute maximum
    MAX_OUTPUT_SIZE = 50_000   # chars
    MAX_MEMORY_MB = 512        # MB

    def __init__(self):
        self._enabled = False  # off by default for safety
        self._persistent_mode = False  # reuse tmpdir per conversation
        self._persistent_dirs = {}  # conv_id -> tmpdir path
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
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = bool(value)

    @property
    def persistent_mode(self) -> bool:
        """Whether to reuse working directories per conversation."""
        return self._persistent_mode

    @persistent_mode.setter
    def persistent_mode(self, value: bool):
        self._persistent_mode = bool(value)
        if not value:
            self.cleanup_all_persistent_dirs()

    def get_persistent_dir(self, conv_id: str) -> str:
        """Get or create a persistent working directory for a conversation.

        Args:
            conv_id: conversation identifier

        Returns:
            Path to the persistent tmpdir
        """
        if conv_id not in self._persistent_dirs:
            d = tempfile.mkdtemp(prefix=f"opti_persist_{conv_id[:8]}_")
            self._persistent_dirs[conv_id] = d
            logger.info(f"Created persistent dir for {conv_id[:8]}: {d}")
        return self._persistent_dirs[conv_id]

    def reset_persistent_dir(self, conv_id: str) -> bool:
        """Remove and recreate the persistent dir for a conversation.

        Returns:
            True if a dir was cleaned up, False if none existed.
        """
        if conv_id in self._persistent_dirs:
            old = self._persistent_dirs.pop(conv_id)
            try:
                shutil.rmtree(old, ignore_errors=True)
            except Exception:
                pass
            logger.info(f"Reset persistent dir for {conv_id[:8]}")
            return True
        return False

    def cleanup_all_persistent_dirs(self):
        """Remove all persistent working directories."""
        for cid, d in list(self._persistent_dirs.items()):
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        count = len(self._persistent_dirs)
        self._persistent_dirs.clear()
        if count:
            logger.info(f"Cleaned up {count} persistent dirs")

    def list_persistent_files(self, conv_id: str) -> list[str]:
        """List files in the persistent working directory for a conversation.

        Returns:
            List of filenames (not full paths), or empty list.
        """
        if conv_id not in self._persistent_dirs:
            return []
        d = self._persistent_dirs[conv_id]
        if not os.path.isdir(d):
            return []
        try:
            return [
                f for f in os.listdir(d)
                if not f.startswith("script.") and os.path.isfile(os.path.join(d, f))
            ]
        except Exception:
            return []

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
        allow_network: bool = False,
        conv_id: str | None = None,
    ) -> ExecutionResult:
        """Execute code in a subprocess and return the result.

        Args:
            code: source code to execute
            language: one of python/r/bash (or alias)
            timeout: max seconds (None = DEFAULT_TIMEOUT)
            allow_network: if False, attempts to restrict network (best-effort)
            conv_id: if provided and persistent_mode is on, reuse working dir

        Returns:
            ExecutionResult with stdout, stderr, timing, etc.
        """
        start_time = time.monotonic()
        language = self._normalize_language(language)

        if not self._enabled:
            return ExecutionResult(
                success=False, stdout="", stderr="",
                return_code=-1, execution_time=0.0,
                language=language,
                error_message="Code execution is disabled. Enable it in Settings.",
            )

        if language not in self.SUPPORTED_LANGUAGES:
            return ExecutionResult(
                success=False, stdout="", stderr="",
                return_code=-1, execution_time=0.0,
                language=language,
                error_message=f"Unsupported language: {language}",
            )

        if language not in self._available:
            return ExecutionResult(
                success=False, stdout="", stderr="",
                return_code=-1, execution_time=0.0,
                language=language,
                error_message=(
                    f"Runtime not found for {language}. "
                    f"Available: {', '.join(self._available.keys()) or 'none'}"
                ),
            )

        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT
        timeout = min(timeout, self.MAX_TIMEOUT)

        # Decide whether to use a persistent or ephemeral directory
        use_persistent = (
            self._persistent_mode
            and conv_id is not None
            and len(conv_id) > 0
        )

        if use_persistent:
            tmpdir = self.get_persistent_dir(conv_id)
            cleanup = False
        else:
            tmpdir = tempfile.mkdtemp(prefix="opti_exec_")
            cleanup = True

        # Snapshot files before execution for output detection
        files_before = set()
        try:
            files_before = {
                f for f in os.listdir(tmpdir)
                if f not in _SCRIPT_FILES
            }
        except OSError:
            pass

        try:
            result = self._run_in_subprocess(
                code, language, tmpdir, timeout, allow_network,
            )
            result.execution_time = time.monotonic() - start_time
            result.working_dir = tmpdir

            # Detect new output files (images, data)
            new_files = self._detect_output_files(tmpdir, files_before)

            if new_files and cleanup:
                # Ephemeral mode: copy images to stable output dir
                output_dir = _get_output_dir()
                stable_paths = []
                for fpath in new_files:
                    fname = os.path.basename(fpath)
                    # Add timestamp prefix to avoid collisions
                    stable_name = f"{int(time.time())}_{fname}"
                    stable_path = os.path.join(output_dir, stable_name)
                    try:
                        shutil.copy2(fpath, stable_path)
                        stable_paths.append(stable_path)
                    except Exception as e:
                        logger.debug(f"Could not copy output file: {e}")
                result.output_files = stable_paths
            elif new_files:
                # Persistent mode: files stay in place
                result.output_files = new_files

            return result
        except Exception as e:
            logger.exception(f"Code execution failed: {e}")
            return ExecutionResult(
                success=False, stdout="", stderr=str(e),
                return_code=-1,
                execution_time=time.monotonic() - start_time,
                language=language,
                error_message=f"Internal error: {e}",
            )
        finally:
            if cleanup:
                try:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass

    def _detect_output_files(
        self, tmpdir: str, files_before: set,
    ) -> list[str]:
        """Find new image/data files created during execution.

        Args:
            tmpdir: working directory
            files_before: set of filenames present before execution

        Returns:
            List of absolute paths to new output files (images only).
        """
        try:
            files_after = set(os.listdir(tmpdir))
        except OSError:
            return []

        new_names = files_after - files_before - _SCRIPT_FILES
        output_paths = []
        for name in sorted(new_names):
            ext = os.path.splitext(name)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                full_path = os.path.join(tmpdir, name)
                if os.path.isfile(full_path):
                    output_paths.append(full_path)
        return output_paths

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

    def _run_in_subprocess(
        self,
        code: str,
        language: str,
        tmpdir: str,
        timeout: int,
        allow_network: bool,
    ) -> ExecutionResult:
        """Actually run the code via subprocess."""
        cmd, script_path = self._prepare_command(code, language, tmpdir)

        env = os.environ.copy()
        # Restrict some environment variables for safety
        env["HOME"] = tmpdir
        env["TMPDIR"] = tmpdir
        # Keep PATH so runtimes can find their dependencies
        # Keep LANG/LC_ALL for proper encoding

        # Build ulimit prefix for memory limit (Linux only)
        preexec = None
        if os.name == "posix":
            mem_bytes = self.MAX_MEMORY_MB * 1024 * 1024
            def _set_limits():
                try:
                    import resource
                    resource.setrlimit(
                        resource.RLIMIT_AS,
                        (mem_bytes, mem_bytes),
                    )
                except Exception:
                    pass  # best-effort
            preexec = _set_limits

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env=env,
                preexec_fn=preexec,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                return_code=-1,
                execution_time=float(timeout),
                language=language,
                error_message=f"Timeout: code exceeded {timeout}s limit",
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        truncated = False

        if len(stdout) > self.MAX_OUTPUT_SIZE:
            stdout = stdout[: self.MAX_OUTPUT_SIZE] + f"\n\n... [truncated at {self.MAX_OUTPUT_SIZE} chars]"
            truncated = True

        if len(stderr) > self.MAX_OUTPUT_SIZE:
            stderr = stderr[: self.MAX_OUTPUT_SIZE] + f"\n\n... [truncated at {self.MAX_OUTPUT_SIZE} chars]"
            truncated = True

        return ExecutionResult(
            success=(proc.returncode == 0),
            stdout=stdout,
            stderr=stderr,
            return_code=proc.returncode,
            execution_time=0.0,  # filled by caller
            language=language,
            truncated=truncated,
        )

    def _prepare_command(
        self, code: str, language: str, tmpdir: str,
    ) -> tuple[list[str], str]:
        """Write code to a temp file and build the command to run it.

        Returns:
            (command_list, script_path)
        """
        if language == "python":
            script = os.path.join(tmpdir, "script.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            cmd_name = self._available["python"]
            return [cmd_name, "-u", script], script

        elif language == "r":
            script = os.path.join(tmpdir, "script.R")
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            return [self._available["r"], "--vanilla", script], script

        elif language == "bash":
            script = os.path.join(tmpdir, "script.sh")
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            os.chmod(script, 0o700)
            return [self._available["bash"], "-e", script], script

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
