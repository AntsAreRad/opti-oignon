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
import re
import sys
import threading
import time
import unicodedata
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


# What each label says of an argument, for the person deciding.
_LABEL_MEANING = {
    "typed": "you typed it in this turn",
    "default": "the tool's own default",
    "unendorsed": "not typed by you: the model chose it",
}


# Characters a terminal does not print as themselves: a carriage return or an
# escape sequence can rewrite the line, a direction override can reorder it,
# and the code points Unicode lets a screen draw as nothing (variation
# selectors, tags, fillers, the symbols drawn blank) can carry bytes unseen.
# The approval queue has already written each of them as its escape and
# doubled every backslash; the terminal escapes any it is still sent, and
# leaves the queue's own writing as it is (a contract holds both).
_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp", "Co", "Cs", "Cn"})
_IGNORABLE = ((0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5),
              (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x2800, 0x2800),
              (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8),
              (0x16FE4, 0x16FE4), (0x1BCA0, 0x1BCA3), (0x1D159, 0x1D159), (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF))


def _hidden(ch: str) -> bool:
    if ch == "\n":
        return False
    category = unicodedata.category(ch)
    if category in _HIDDEN_CATEGORIES or (category == "Zs" and ch != " "):
        return True
    point = ord(ch)
    return any(low <= point <= high for low, high in _IGNORABLE)


def _plain(text: Any) -> str:
    """``text`` as a terminal prints it as itself: every character that would hide what follows written as its escape.

    Applied here whatever the backend sent, so a value cannot rewrite the
    line the person reads before deciding; what the approval queue already
    wrote (its escapes, its doubled backslashes) is printed unchanged.
    """
    return "".join(ascii(ch)[1:-1] if _hidden(ch) else ch for ch in str(text))


def _line(text: Any) -> str:
    """``text`` printed as itself on one row: a line break written as its escape too."""
    return _plain(text).replace("\n", "\\n")


# A name printed on a row of the terminal's own: a plain identifier. Any
# other name is printed behind a bar, as a value is.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")


def _units(line: str) -> list[str]:
    """``line`` in the pieces a row keeps whole: an escape (a backslash, then
    ``x`` and two digits, ``u`` and four, ``U`` and eight, or one character),
    or a single character."""
    units, start = [], 0
    while start < len(line):
        step = 1 if line[start] != "\\" else {"x": 4, "u": 6, "U": 10}.get(line[start + 1:start + 2], 2)
        units.append(line[start:start + step])
        start += step
    return units


def _cells(text: str) -> int:
    """The most cells any terminal gives ``text``: one for printable ASCII, two for anything else."""
    return sum(1 if " " <= ch <= "~" else 2 for ch in text)


def _rows(first: str, text: str, width: int, rest: str | None = None) -> list[str]:
    """``text`` on rows no wider than ``width``: behind ``first``, then behind ``rest`` (``first`` when None).

    A row wider than the terminal wraps to its left edge, where the rest of
    a value could pass for a row of the terminal's own; so each line is cut
    here, between the pieces ``_units`` keeps whole, and every row carries
    its prefix.
    """
    rest = first if rest is None else rest
    rows: list[str] = []
    for line in text.split("\n"):
        prefix = first if not rows else rest
        row, used = "", _cells(prefix)
        for unit in _units(line):
            cells = _cells(unit)
            if row and used + cells > width:
                rows.append(prefix + row)
                prefix = rest
                row, used = "", _cells(prefix)
            row += unit
            used += cells
        rows.append(prefix + row)
    return rows


def _size(size: Any) -> str:
    """A value's own length and lines as the queue counted them, or nothing when the backend sent no count."""
    try:
        chars, count = int(size["chars"]), int(size["lines"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return ""
    return f"{chars} characters" + (f", {count} lines" if count > 1 else "")


def terminal_width(stream: Any = None) -> int:
    """The narrowest width in cells the rows may land on: ``COLUMNS``, and every terminal the process is attached to
    -- ``stream`` (standard error by default), standard output, standard input and the controlling terminal, since a
    piped stream (``2>&1 | tee``) still lands on one of them; else 80. A width wider than the terminal would let a
    row wrap at its left edge."""
    widths = []
    try:
        widths.append(int(os.environ.get("COLUMNS", "")))
    except ValueError:
        pass
    for attached in (stream or sys.stderr, sys.stdout, sys.stdin):
        try:
            widths.append(os.get_terminal_size(attached.fileno()).columns)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    try:
        # Read-only, never taken as this process's controlling terminal, and
        # never waited on (a serial line can hold an open for its carrier).
        tty = os.open("/dev/tty", os.O_RDONLY | getattr(os, "O_NOCTTY", 0) | getattr(os, "O_NONBLOCK", 0))
    except (AttributeError, OSError):
        tty = None
    if tty is not None:
        try:
            widths.append(os.get_terminal_size(tty).columns)
        except (OSError, ValueError):
            pass
        finally:
            os.close(tty)
    widths = [width for width in widths if width > 0]
    return min(widths) if widths else 80


def format_approval(meta: dict[str, Any], *, color: bool = True, width: int | None = None) -> str:
    """Format a tool call held for the user's answer: each value, where it came from, how to answer.

    Each argument's own row comes first -- its name and where it came from
    -- then its value, every line of it behind a bar. A name that is not a
    plain identifier is printed behind a bar of its own, and the summary an
    older backend sends behind the value's bar: no text the model chose is
    printed on a row that could pass for the terminal's own. No row is wider
    than ``width`` (the terminal's, by default).
    """
    width = int(width or terminal_width())
    aid = _line(meta.get("approval_id", "?"))
    tool = _line(meta.get("tool_name", "?"))
    effect = _line(meta.get("effect") or "")
    labels = meta.get("labels") or {}
    arguments = meta.get("arguments") or {}
    sizes = meta.get("sizes") if isinstance(meta.get("sizes"), dict) else {}
    head = f"Tool call waiting for your answer: {tool}" + (f" ({effect})" if effect else "")
    # A row of the terminal's own goes on behind four spaces: two, then text,
    # is an argument's place, and no field's text may take it.
    lines = [_col(row, _C.YELLOW, bold=True, enabled=color) for row in _rows("", head, width, "    ")]
    if not arguments:
        summary = _plain(meta.get("arguments_summary") or "")
        if summary:
            lines.extend(_rows("    | ", summary, width))
    names = list(arguments) + [n for n in labels if n not in arguments]
    for index, name in enumerate(names, start=1):
        tag = _line(labels.get(name, ""))
        plain = isinstance(name, str) and bool(_NAME.match(name))
        header = name if plain else f"argument {index}"
        if tag:
            header += f"   [{tag}: {_LABEL_MEANING.get(tag, tag)}]"
        size = _size(sizes.get(name))
        if size:
            header += f"   ({size})"
        rows = _rows("  ", header, width, "    ")
        # Green only for a label the terminal knows as the user's or the
        # tool's own; one it does not know is left uncoloured, never safe.
        shade = _C.RED if tag == "unendorsed" else _C.GREEN if tag in ("typed", "default") else ""
        if tag and color and shade:
            rows = [row.replace(f"[{tag}:", f"[{_col(tag, shade, enabled=True)}:", 1) for row in rows]
        lines.extend(rows)
        if not plain:
            lines.extend(_rows("    name | ", _plain(name), width))
        if name in arguments:
            value = arguments[name]
            text = _plain(value) if isinstance(value, str) else _plain(repr(value))
            lines.extend(_rows("    | ", text, width))
    lines.extend(_rows("  ", f"Answer with: oo approve {aid}   or   oo deny {aid}", width, "    "))
    return "\n".join(lines)


def format_resolution(meta: dict[str, Any], *, color: bool = True) -> str:
    """Format how a held tool call ended, whoever answered it."""
    allowed = bool(meta.get("approved"))
    verdict = _col("Allowed" if allowed else "Refused", _C.GREEN if allowed else _C.RED, bold=True, enabled=color)
    return f"{verdict}: {_line(meta.get('tool_name', '?'))}"


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
