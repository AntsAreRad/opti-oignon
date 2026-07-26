#!/usr/bin/env python3
"""
Output formatting utilities -- Opti-Oignon CLI.

Provides coloured terminal output, spinner animations, and
human-friendly formatters for model lists, status dashboards,
and error messages.  Respects ``NO_COLOR`` / ``--no-color``.
"""

import itertools
import sys
import threading
from typing import Any

# -- ANSI colour helpers ---------------------------------------------------

class _Colours:
    """ANSI escape sequences for terminal colouring."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


_C = _Colours


def _col(text: str, colour: str, *, bold: bool = False, enabled: bool = True) -> str:
    """Wrap *text* with ANSI colour codes if *enabled*."""
    if not enabled:
        return text
    prefix = colour
    if bold:
        prefix = _C.BOLD + colour
    return f"{prefix}{text}{_C.RESET}"


# -- Public helpers --------------------------------------------------------

def echo_error(msg: str, *, color: bool = True) -> None:
    """Print an error message to stderr."""
    prefix = _col("Error:", _C.RED, bold=True, enabled=color)
    click_echo = _safe_echo()
    click_echo(f"{prefix} {msg}", err=True)


def echo_success(msg: str, *, color: bool = True) -> None:
    """Print a success message."""
    prefix = _col("OK", _C.GREEN, bold=True, enabled=color)
    click_echo = _safe_echo()
    click_echo(f"{prefix} {msg}")


def _safe_echo():
    """Return click.echo if available, else a basic fallback."""
    try:
        import click
        return click.echo
    except ImportError:
        def _fallback(msg: str, err: bool = False, **kw: Any) -> None:
            dest = sys.stderr if err else sys.stdout
            print(msg, file=dest)
        return _fallback


# -- Spinner ---------------------------------------------------------------

class Spinner:
    """Simple terminal spinner for long-running operations.

    Use as a context manager::

        with Spinner("Loading"):
            do_work()
    """

    _FRAMES = [".", "..", "...", "   "]

    def __init__(self, message: str = "", *, enabled: bool = True) -> None:
        self.message = message
        self.enabled = enabled and sys.stderr.isatty()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Spinner":
        if self.enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2)
            # Clear the spinner line
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()

    def _spin(self) -> None:
        frames = itertools.cycle(self._FRAMES)
        while not self._stop_event.is_set():
            frame = next(frames)
            sys.stderr.write(f"\r  {self.message}{frame}")
            sys.stderr.flush()
            self._stop_event.wait(0.3)


# -- Formatters ------------------------------------------------------------

def format_models_table(models: list[dict[str, Any]], *, color: bool = True) -> str:
    """Format a list of model dicts as a terminal-friendly table."""
    if not models:
        return "No models available."

    lines: list[str] = []
    # Header
    header = f"{'Name':<40} {'Size':>10} {'Family':<15} {'Quant':<8}"
    lines.append(_col(header, _C.BOLD, enabled=color))
    lines.append("-" * len(header))

    for m in models:
        name = m.get("name", "?")
        size = m.get("size_display") or m.get("size", "?")
        family = m.get("family", "?") or "?"
        quant = m.get("quantization", "") or ""
        lines.append(f"{name:<40} {str(size):>10} {family:<15} {quant:<8}")

    lines.append(f"\n{_col(str(len(models)), _C.CYAN, bold=True, enabled=color)} model(s) available.")
    return "\n".join(lines)


def format_status(data: dict[str, Any], *, color: bool = True) -> str:
    """Format health dashboard data as a readable status report."""
    lines: list[str] = []

    lines.append(_col("Opti-Oignon Status", _C.BOLD, enabled=color))
    lines.append("=" * 40)

    # General
    version = data.get("version", "?")
    uptime = data.get("uptime_seconds")
    lines.append(f"  Version:  {version}")
    if uptime is not None:
        mins = int(uptime) // 60
        lines.append(f"  Uptime:   {mins} min")

    # Models
    model_count = data.get("model_count") or data.get("models_count", "?")
    lines.append(f"  Models:   {model_count}")

    # Warmup
    warmup = data.get("warmup_status", {})
    if isinstance(warmup, dict):
        warmed = warmup.get("warmed_models", [])
        if warmed:
            lines.append(f"  Warmed:   {', '.join(warmed[:5])}")

    # Ollama
    ollama = data.get("ollama_status") or data.get("ollama", {})
    if isinstance(ollama, dict):
        oll_ok = ollama.get("connected", ollama.get("available", False))
        status_str = _col("connected", _C.GREEN, enabled=color) if oll_ok else _col(
            "disconnected", _C.RED, enabled=color)
        lines.append(f"  Ollama:   {status_str}")

    # Context health
    ctx = data.get("context_health", {})
    if isinstance(ctx, dict) and ctx.get("available"):
        lines.append(f"  Context:  {_col('healthy', _C.GREEN, enabled=color)}")

    return "\n".join(lines)
