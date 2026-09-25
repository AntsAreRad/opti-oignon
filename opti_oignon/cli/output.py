#!/usr/bin/env python3
"""
Output formatting utilities -- Opti-Oignon CLI.

Provides coloured terminal output, spinner animations, the wait
animation of ``oo chat``, and human-friendly formatters for model lists,
status dashboards, and error messages.  Respects ``NO_COLOR`` /
``--no-color``.
"""

import itertools
import os
import sys
import threading
import time
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

def _run_color() -> bool:
    """The colour of the run: its configuration inside a command, else ``NO_COLOR``.

    A caller that says nothing gets the colour the run was given --
    ``--no-color``, ``NO_COLOR`` or the file -- and not a default of its own.
    """
    try:
        import click
    except ImportError:
        click = None
    ctx = click.get_current_context(silent=True) if click is not None else None
    if ctx is not None:
        obj = ctx.find_root().obj
        color = getattr(obj.get("config") if isinstance(obj, dict) else None, "color", None)
        if isinstance(color, bool):
            return color
    return "NO_COLOR" not in os.environ


def echo_error(msg: str, *, color: bool | None = None) -> None:
    """Print an error message to stderr, coloured as the run is unless told."""
    prefix = _col("Error:", _C.RED, bold=True, enabled=_run_color() if color is None else color)
    click_echo = _safe_echo()
    click_echo(f"{prefix} {msg}", err=True)


def echo_success(msg: str, *, color: bool | None = None) -> None:
    """Print a success message, coloured as the run is unless told."""
    prefix = _col("OK", _C.GREEN, bold=True, enabled=_run_color() if color is None else color)
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


# -- Wait line ---------------------------------------------------------------
#
# One line of ASCII art on stderr while the user really waits in ``oo chat``:
# from Enter to the first visible answer, and during /close and /open. It is
# erased before anything else is shown, never touches stdout, and writes no
# escape sequence and no newline -- only carriage returns and printable
# ASCII -- so an interrupted or killed process leaves no hidden cursor and
# no colour behind. The art is content, not configuration: the frames stay
# here, where the ASCII and width contracts can hold them.

WAIT_WIDTH = 32

# Thirteen columns each; the core ``@`` stays in the same column while the
# layers grow, shrink, fall or sprout around it.
WAIT_FRAMES = {
    "waiting": (
        "     @       ",
        "    (@)      ",
        "   ((@))     ",
        "  (((@)))    ",
        "   ((@))     ",
        "    (-)      ",
    ),
    # The first five grow once, then the last four sway.
    "requested": (
        "  (((@)))    ",
        "  (((@)))_   ",
        "  (((@)))_.  ",
        "  (((@)))_v  ",
        "  (((@)))_\\|/",
        "  (((@)))_||/",
        "  (((@)))_\\|/",
        "  (((@)))_\\||",
    ),
    # Peels drop onto a pile and the core stays; the loop visibly starts
    # over, so it claims no progress it cannot measure.
    "closing": (
        "  (((@)))    ",
        "   ((@)) )   ",
        "   ((@))    _",
        "    (@)  )  _",
        "    (@)    __",
        "     @   ) __",
        "     @    ___",
    ),
    "opening": (
        "     @       ",
        "    (@)      ",
        "   ((@))     ",
        "  (((@)))    ",
        "  (((@)))    ",
    ),
}

_REQUESTED_GROWTH = 4
WAIT_CLEAR = "\r" + " " * WAIT_WIDTH + "\r"
WAIT_BYE = "  (@)/  bye\n"
_ELAPSED_CAP = 9999


def _wait_art(kind: str, index: int) -> str:
    frames = WAIT_FRAMES[kind]
    index = max(int(index), 0)
    if kind == "requested":
        if index < _REQUESTED_GROWTH:
            return frames[index]
        loop = frames[_REQUESTED_GROWTH:]
        return loop[(index - _REQUESTED_GROWTH) % len(loop)]
    return frames[index % len(frames)]


def render_wait_line(kind: str, index: int, elapsed_s: float) -> str:
    """The 32 columns of one frame: the label, the whole seconds since Enter, the art."""
    seconds = min(max(int(elapsed_s), 0), _ELAPSED_CAP)
    return f"  {kind:<9} {seconds:>4}s  {_wait_art(kind, index)}"


def wait_enabled(cfg: Any, stream: Any, env: Any) -> bool:
    """True only when every switch allows the animation and the stream is a terminal."""
    if not getattr(cfg, "animations", False) or not getattr(cfg, "color", False):
        return False
    if "NO_COLOR" in env:
        return False
    if str(env.get("TERM", "") or "").strip().lower() == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001 - a stream that cannot say is not a terminal
        return False


def _stream_columns(stream: Any) -> int:
    try:
        return int(os.get_terminal_size(stream.fileno()).columns)
    except Exception:  # noqa: BLE001 - an unreadable width draws nothing
        return 0


class WaitLine:
    """The wait animation of ``oo chat``: armed after Enter, stopped before any output.

    ``arm`` starts one wait and its daemon frame thread; ``request`` turns a
    plain wait into the sprout once the executor says the request is with
    the model; ``stop`` is final for that wait, returns within ``stop_ms``
    even if a write is stalled, and after it returns nothing more is drawn;
    ``bye`` writes the one-line farewell. A write that fails switches the
    waiter off for the rest of the session instead of raising.
    """

    def __init__(self, stream: Any, *, clock: Any, columns: Any, interval_ms: int,
                 delay_ms: int, stop_ms: int, thread_factory: Any = threading.Thread) -> None:
        self._stream = stream
        self._clock = clock
        self._columns = columns
        self._interval = interval_ms / 1000.0
        self._delay = delay_ms / 1000.0
        self._stop_s = stop_ms / 1000.0
        self._thread_factory = thread_factory
        self._lock = threading.Lock()
        self._gen = 0
        self._kind = "waiting"
        self._armed_at: float | None = None
        self._origin = 0.0
        self._phase_at: float | None = None
        self._stopped = True
        self._stop_event: threading.Event | None = None
        self._drawn = False
        self._last: str | None = None
        self._off = False

    def now(self) -> float:
        return self._clock()

    def arm(self, kind: str, origin: float) -> None:
        """Start a wait of ``kind``; the elapsed seconds count from ``origin``."""
        if self._off:
            return
        if not self._lock.acquire(timeout=self._stop_s):
            return
        try:
            self._gen += 1
            gen = self._gen
            self._kind = kind
            self._armed_at = self._clock()
            self._origin = origin
            self._phase_at = None
            self._stopped = False
            self._last = None
            event = self._stop_event = threading.Event()
        finally:
            self._lock.release()
        thread = self._thread_factory(target=self._run, args=(gen, event), name="oo-wait", daemon=True)
        thread.start()

    def request(self) -> None:
        """The executor's keepalive: the request is with the model, so the wait becomes the sprout."""
        if self._kind == "waiting" and self._phase_at is None and not self._stopped:
            self._phase_at = self._clock()

    def tick(self, now: float | None = None) -> bool:
        """Draw the frame due at ``now`` if it differs from the one on screen."""
        return self._tick(self._gen, now)

    def _run(self, gen: int, event: threading.Event) -> None:
        while not event.is_set():
            self._tick(gen)
            event.wait(self._interval)

    def _tick(self, gen: int, now: float | None = None) -> bool:
        with self._lock:
            if self._off or self._stopped or gen != self._gen or self._armed_at is None:
                return False
            now = self._clock() if now is None else now
            start = self._armed_at + self._delay
            if now < start:
                return False
            try:
                columns = int(self._columns())
            except Exception:  # noqa: BLE001 - an unreadable width draws nothing
                columns = 0
            if columns <= WAIT_WIDTH:
                return False
            if self._phase_at is not None:
                kind = "requested"
                index = int((now - max(self._phase_at, start)) // self._interval)
            else:
                kind = self._kind
                index = int((now - start) // self._interval)
            line = render_wait_line(kind, index, now - self._origin)
            if line == self._last:
                return False
            if not self._write("\r" + line):
                return False
            self._last = line
            self._drawn = True
            return True

    def stop(self) -> None:
        """End the current wait: clear the line if one was drawn, within the stop bound."""
        self._stopped = True
        event = self._stop_event
        if event is not None:
            event.set()
        if not self._lock.acquire(timeout=self._stop_s):
            return
        try:
            if self._drawn and not self._off:
                self._write(WAIT_CLEAR)
            self._drawn = False
            self._last = None
        finally:
            self._lock.release()

    def bye(self) -> None:
        """The one-line farewell after /quit."""
        if not self._off:
            self._write(WAIT_BYE)

    def _write(self, text: str) -> bool:
        try:
            self._stream.write(text)
            self._stream.flush()
            return True
        except OSError:
            self._off = True
            return False


def make_wait_line(cfg: Any, *, stream: Any = None, clock: Any = None, columns: Any = None,
                   env: Any = None, thread_factory: Any = None) -> "WaitLine | None":
    """The waiter for ``oo chat``, or None when any switch says no animation."""
    stream = sys.stderr if stream is None else stream
    env = os.environ if env is None else env
    if not wait_enabled(cfg, stream, env):
        return None
    return WaitLine(
        stream,
        clock=clock or time.monotonic,
        columns=columns or (lambda: _stream_columns(stream)),
        interval_ms=int(getattr(cfg, "animation_interval_ms", 150)),
        delay_ms=int(getattr(cfg, "animation_delay_ms", 400)),
        stop_ms=int(getattr(cfg, "animation_stop_ms", 100)),
        thread_factory=thread_factory or threading.Thread,
    )


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
