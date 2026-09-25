#!/usr/bin/env python3
"""Contracts for the wait animations of ``oo chat``.

While the user really waits -- from Enter to the first visible answer, and
during /close and /open -- one line of ASCII art is drawn on stderr and
erased before anything else is shown. It never touches stdout, never
delays or reorders a token, and stays silent wherever it could get in the
way.

Gate:
  * TW1 -- stderr not a terminal: nothing is written.
  * TW2 -- NO_COLOR (even empty) or --no-color: nothing is written.
  * TW3 -- ``animations: false`` in cli.yaml: nothing is written.
  * TW4 -- TERM=dumb: nothing is written.

Shape:
  * TW5 -- every frame, label and the farewell are printable ASCII, and the
    CLI sources hold no character above 127.
  * TW6 -- every rendered line is exactly 32 columns, the elapsed seconds
    capped at 9999.
  * TW7 -- a wait writes only carriage returns and printable ASCII: no
    escape byte, no newline.
  * TW8 -- every write is flushed.
  * TW9 -- nothing is drawn before the grace delay.
  * TW10 -- a terminal of 32 columns or fewer, or one whose width cannot be
    read, draws nothing.

Placement in the chat loop:
  * TW11 -- the line is cleared before the first byte of any visible event,
    and nothing is drawn after it.
  * TW12 -- a keepalive keeps the animation; the first real token clears it.
  * TW13 -- the wait resumes after the first turn's conversation line.
  * TW14 -- the elapsed seconds count from Enter across a resumed wait.
  * TW15 -- once the answer has started nothing is drawn.
  * TW16 -- the sprout appears only after the executor's keepalive.
  * TW17 -- the executor yields that keepalive only once its request thread
    has started, which is what makes the sprout's label true.
  * TW18 -- /close and /open draw their own frames, in order.
  * TW19 -- stdout is byte-identical with and without animations.
  * TW20 -- replaying everything the waiter wrote leaves a blank screen.
  * TW21 -- Ctrl-C during a wait clears the line and exits 130.

Engine:
  * TW22 -- a stopped wait writes nothing; a new one draws again.
  * TW23 -- stop returns within its bound when a write is stalled.
  * TW24 -- a broken stream switches the waiter off, never raising.
  * TW25 -- the frame thread starts at arm, as a daemon, not at construction.
  * TW31 -- the frame thread's own loop draws one frame per interval and
    ends when its wait is stopped (every other contract ticks by hand).

Configuration:
  * TW26 -- a bad animation value falls back alone; the others survive.
  * TW27 -- the settings survive a save and are not switched off by colour.
  * TW28 -- ``oo config set`` takes the switch and refuses an out-of-range
    interval by name, leaving the file as it was.

Ends:
  * TW29 -- /quit says goodbye in one line; nothing else does.
  * TW30 -- a last line without a newline arms no wait.

Local-only (the public distribution ships no tests). The CLI modules are
loaded through the shared isolation window; the stream, the clock, the
terminal width, the environment and the frame thread are seams, so every
frame is decided by a fake clock and read from a fake terminal.
"""

import ast
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_CLI = ("config", "client", "output", "main")
_KINDS = ("waiting", "requested", "closing", "opening")


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------
class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _lengths():
    """How many bytes stdout and stderr hold right now; -1 when unreadable."""
    out = []
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
            out.append(len(stream.buffer.getvalue()))
        except Exception:
            out.append(-1)
    return tuple(out)


class _TTY:
    """A terminal stand-in: records every write with the stdout/stderr lengths at that moment."""

    def __init__(self, tty=True, fail=0):
        self.tty = tty
        self.fail = fail
        self.writes = []
        self.flushes = 0

    def isatty(self):
        return self.tty

    def write(self, text):
        if self.fail:
            self.fail -= 1
            raise OSError("broken pipe")
        self.writes.append((text, _lengths()))
        return len(text)

    def flush(self):
        self.flushes += 1

    @property
    def data(self):
        return "".join(text for text, _ in self.writes)


class _Threads:
    """A thread factory that starts nothing: the contract ticks the waiter itself."""

    def __init__(self):
        self.made = []

    def __call__(self, target=None, args=(), name=None, daemon=None, **kwargs):
        made = SimpleNamespace(target=target, args=args, name=name, daemon=daemon, started=False)
        made.start = lambda: setattr(made, "started", True)
        self.made.append(made)
        return made

    def waiters(self):
        seen = []
        for made in self.made:
            owner = getattr(made.target, "__self__", None)
            if owner is not None and owner not in seen:
                seen.append(owner)
        return seen


class _Script:
    """A chat session whose turns are scripts; ("advance", s) moves the clock and ticks the waiter."""

    def __init__(self, scripts, clock, threads):
        self.scripts, self.clock, self.threads = scripts, clock, threads

    def handle(self, line):
        for step in self.scripts.get(line.strip(), []):
            if step[0] == "advance":
                self.clock.advance(step[1])
                for waiter in self.threads.waiters():
                    waiter.tick()
            elif step[0] == "interrupt":
                raise KeyboardInterrupt
            elif step[0] == "event":
                yield step[1]
            else:
                yield SimpleNamespace(kind=step[0], text=step[1])


class _RaisingText:
    """An info event whose text raises Ctrl-C in the loop body, after the waiter has drawn."""

    kind = "info"

    @property
    def text(self):
        raise KeyboardInterrupt


def _open(monkeypatch, tmp_path, cli_yaml=None):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("NO_COLOR", raising=False)
    if cli_yaml is not None:
        folder = tmp_path / "xdg" / "opti-oignon"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "cli.yaml").write_text(yaml.safe_dump(cli_yaml), encoding="utf-8")
    loaded, restore = isolate(
        targets={f"opti_oignon.cli.{m}": source("cli", f"{m}.py") for m in _CLI},
        packages=("opti_oignon.cli",),
    )
    return loaded, restore


def _unit(out, *, tty=None, clock=None, columns=80, interval=150, delay=400, stop=100, threads=None):
    stream = tty if tty is not None else _TTY()
    clock = clock if clock is not None else _Clock()
    threads = threads if threads is not None else _Threads()
    cols = columns if callable(columns) else (lambda: columns)
    waiter = out.WaitLine(
        stream, clock=clock, columns=cols, interval_ms=interval, delay_ms=delay,
        stop_ms=stop, thread_factory=threads,
    )
    return waiter, stream, clock, threads


def _invoke(loaded, lines, scripts, *, tty=True, args=(), env=None, columns=80):
    from click.testing import CliRunner

    main = loaded["opti_oignon.cli.main"]
    clock, threads, stream = _Clock(), _Threads(), _TTY(tty)
    session = _Script(scripts, clock, threads)
    seams = {
        "stream": stream, "clock": clock, "columns": lambda: columns,
        "env": env if env is not None else {"TERM": "xterm-256color"},
        "thread_factory": threads,
    }
    result = CliRunner(mix_stderr=False).invoke(
        main.cli, [*args, "chat"], input="".join(lines),
        obj={"chat_session": lambda model, conversation_id: session, "wait_seams": seams},
    )
    return result, stream


def _frames(out, stream):
    return [text for text, _ in stream.writes if text.startswith("\r") and text != out.WAIT_CLEAR]


def _clears(out, stream):
    return [i for i, (text, _) in enumerate(stream.writes) if text == out.WAIT_CLEAR]


_TURN = {"hello": [("advance", 1.0), ("token", "Hi there")]}


# ---------------------------------------------------------------------------
# TW1-TW4 -- the gate
# ---------------------------------------------------------------------------
def test_tw1_without_a_terminal_the_waiter_writes_nothing(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        result, stream = _invoke(loaded, ["hello\n"], _TURN, tty=False)
        assert result.exit_code == 0, result.output
        assert stream.writes == [], "not a terminal: not one byte"
        control, drawn = _invoke(loaded, ["hello\n"], _TURN, tty=True)
        assert control.exit_code == 0 and len(drawn.writes) > 0, "control: a terminal draws"
    finally:
        restore()


def test_tw2_no_color_in_any_form_switches_the_waiter_off(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        _, empty = _invoke(loaded, ["hello\n"], _TURN, env={"TERM": "xterm", "NO_COLOR": ""})
        assert empty.writes == [], "NO_COLOR present, even empty: nothing"
        _, flag = _invoke(loaded, ["hello\n"], _TURN, args=("--no-color",))
        assert flag.writes == [], "--no-color: nothing"
        _, control = _invoke(loaded, ["hello\n"], _TURN)
        assert len(control.writes) > 0, "control: colour on draws"
    finally:
        restore()


def test_tw3_the_animations_setting_switches_the_waiter_off(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path, cli_yaml={"animations": False})
    try:
        _, stream = _invoke(loaded, ["hello\n"], _TURN)
        assert stream.writes == [], "animations: false writes nothing"
    finally:
        restore()
    loaded, restore = _open(monkeypatch, tmp_path, cli_yaml={"animations": True})
    try:
        _, stream = _invoke(loaded, ["hello\n"], _TURN)
        assert len(stream.writes) > 0, "control: animations: true draws"
    finally:
        restore()


def test_tw4_a_dumb_terminal_switches_the_waiter_off(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        _, dumb = _invoke(loaded, ["hello\n"], _TURN, env={"TERM": "dumb"})
        assert dumb.writes == [], "TERM=dumb: nothing"
        _, control = _invoke(loaded, ["hello\n"], _TURN, env={"TERM": "xterm"})
        assert len(control.writes) > 0, "control: xterm draws"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TW5-TW10 -- the shape of what is drawn
# ---------------------------------------------------------------------------
def _printable(text):
    return all(0x20 <= ord(ch) <= 0x7E for ch in text)


def test_tw5_every_frame_and_the_cli_source_are_printable_ascii(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        assert set(out.WAIT_FRAMES) == set(_KINDS)
        for kind, frames in out.WAIT_FRAMES.items():
            assert frames and all(_printable(f) for f in frames), kind
            assert _printable(kind)
        assert _printable(out.WAIT_BYE.rstrip("\n")) and out.WAIT_BYE.endswith("\n")
    finally:
        restore()
    census = {p.name: sum(ord(ch) > 127 for ch in p.read_text(encoding="utf-8")) for p in (REPO / "opti_oignon" / "cli").glob("*.py")}
    assert census and sum(census.values()) == 0, census
    control = (REPO / "opti_oignon" / "redteam" / "strategies.py").read_text(encoding="utf-8")
    assert sum(ord(ch) > 127 for ch in control) == 3, "control: the census sees non-ASCII where it is"


def test_tw6_every_rendered_line_is_exactly_thirty_two_columns(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        assert out.WAIT_WIDTH == 32
        for kind in _KINDS:
            for k in range(0, 24):
                for elapsed in (0, 9, 9999, 20000):
                    line = out.render_wait_line(kind, k, elapsed)
                    assert len(line) == 32 and _printable(line), (kind, k, elapsed, line)
        assert "9999s" in out.render_wait_line("waiting", 0, 20000), "the elapsed seconds are capped"
    finally:
        restore()


def _full_wait(out, kind):
    waiter, stream, clock, _ = _unit(out)
    waiter.arm(kind, clock())
    for _ in range(12):
        clock.advance(0.2)
        waiter.tick()
        if kind == "waiting" and clock() > 1001.0:
            waiter.request()
    waiter.stop()
    return stream


def test_tw7_a_wait_writes_only_carriage_returns_and_printable_ascii(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        for kind in ("waiting", "closing", "opening"):
            stream = _full_wait(out, kind)
            assert len(stream.writes) > 2, "control: the wait drew and cleared"
            assert all(ch == "\r" or 0x20 <= ord(ch) <= 0x7E for ch in stream.data), kind
            assert "\x1b" not in stream.data and "\n" not in stream.data
    finally:
        restore()


def test_tw8_every_write_is_flushed(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        stream = _full_wait(out, "waiting")
        assert len(stream.writes) > 0 and stream.flushes == len(stream.writes)
    finally:
        restore()


def test_tw9_nothing_is_drawn_before_the_grace_delay(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        waiter, stream, clock, _ = _unit(out, delay=400, interval=150)
        armed = clock()
        waiter.arm("waiting", armed)
        waiter.tick(armed + 0.399)
        assert stream.writes == [], "a wait shorter than the grace delay shows nothing"
        waiter.tick(armed + 0.400 + 0.150)
        assert len(stream.writes) == 1, "past the delay, one frame"
    finally:
        restore()


def test_tw10_a_terminal_of_thirty_two_columns_or_fewer_draws_nothing(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]

        def unreadable():
            raise OSError("no terminal size")

        for columns, expected in ((32, 0), (33, 1), (unreadable, 0)):
            waiter, stream, clock, _ = _unit(out, columns=columns)
            waiter.arm("waiting", clock())
            waiter.tick(clock() + 1.0)
            assert len(stream.writes) == expected, columns
    finally:
        restore()


# ---------------------------------------------------------------------------
# TW11-TW21 -- placement in the chat loop
# ---------------------------------------------------------------------------
def test_tw11_the_line_is_cleared_before_the_first_byte_of_any_visible_event(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        scripts = {
            "hello": [("advance", 1.0), ("token", "Hi"), ("advance", 3.0)],
            "bad": [("advance", 1.0), ("refusal", "no backend"), ("advance", 3.0)],
        }
        result, stream = _invoke(loaded, ["hello\n"], scripts)
        clears = _clears(out, stream)
        assert _frames(out, stream) and clears, "control: frames were drawn, then cleared"
        assert clears[-1] == len(stream.writes) - 1, "nothing is drawn after the clear"
        assert stream.writes[clears[-1]][1][0] == 0, "cleared before the token's first byte reached stdout"
        assert result.stdout.startswith("Hi")

        result, stream = _invoke(loaded, ["bad\n"], scripts)
        clears = _clears(out, stream)
        before = len(result.stderr.split("Error:")[0].encode("utf-8"))
        assert clears and stream.writes[clears[0]][1][1] == before, "cleared before the refusal reached stderr"
        assert "no backend" in result.stderr
    finally:
        restore()


def test_tw12_keepalives_keep_the_animation_and_the_first_real_token_clears_it(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        scripts = {"hello": [("advance", 1.0), ("token", ""), ("advance", 1.0), ("token", ""), ("advance", 1.0), ("token", "Hi")]}
        _, stream = _invoke(loaded, ["hello\n"], scripts)
        clears = _clears(out, stream)
        assert len(clears) == 1 and clears[0] == len(stream.writes) - 1, "one clear, at the first real token"
        assert len(_frames(out, stream)) >= 3, "the frames went on across the keepalives"
    finally:
        restore()


def test_tw13_the_wait_resumes_after_the_first_turns_conversation_line(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        scripts = {"hello": [("advance", 1.0), ("info", "conversation conv-1"), ("advance", 1.0), ("token", "Hi")]}
        _, stream = _invoke(loaded, ["hello\n"], scripts)
        info_len = len("conversation conv-1\n")
        after = [text for text, (out_len, _) in stream.writes if out_len == info_len and text != out.WAIT_CLEAR]
        assert after, "frames were drawn between the conversation line and the first token"
    finally:
        restore()


def test_tw14_elapsed_counts_from_enter_across_a_re_arm(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        scripts = {"hello": [("advance", 10.0), ("info", "conversation conv-1"), ("advance", 2.3), ("token", "Hi")]}
        _, stream = _invoke(loaded, ["hello\n"], scripts)
        frames = _frames(out, stream)
        assert frames and "  12s" in frames[-1], frames[-1] if frames else frames
    finally:
        restore()


def test_tw15_nothing_is_drawn_once_the_answer_has_started(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        for first in (("token", "Hi"), ("thinking", "hmm")):
            scripts = {"hello": [("advance", 1.0), first, ("advance", 5.0), ("info", "note"), ("advance", 5.0), ("token", " there")]}
            _, stream = _invoke(loaded, ["hello\n"], scripts)
            clears = _clears(out, stream)
            assert clears and clears[0] == len(stream.writes) - 1, f"{first[0]}: nothing after the answer started"
    finally:
        restore()


def test_tw16_the_sprout_appears_only_after_the_executors_keepalive(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        scripts = {"hello": [("advance", 1.5), ("token", ""), ("advance", 1.5), ("token", "Hi")]}
        _, stream = _invoke(loaded, ["hello\n"], scripts)
        frames = _frames(out, stream)
        kinds = [f[2:11].strip() for f in (frame[1:] for frame in frames)]
        assert "waiting" in kinds and "requested" in kinds, kinds
        first_requested = kinds.index("requested")
        assert all(k == "waiting" for k in kinds[:first_requested]), "no sprout before the keepalive"
        assert all(k == "requested" for k in kinds[first_requested:]), "only the sprout after it"
    finally:
        restore()


def _empty_yields(src):
    """The empty-string yields of Executor.execute, with whether each sits in `except queue.Empty` after the thread start."""
    tree = ast.parse(src)
    execute = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Executor":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "execute":
                    execute = item
    assert execute is not None, "Executor.execute is where the chat stream comes from"
    starts = [
        n.lineno for n in ast.walk(execute)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "start"
        and isinstance(n.func.value, ast.Name) and n.func.value.id == "stream_thread_obj"
    ]
    handlers = [
        h for h in ast.walk(execute)
        if isinstance(h, ast.ExceptHandler) and isinstance(h.type, ast.Attribute)
        and h.type.attr == "Empty" and isinstance(h.type.value, ast.Name) and h.type.value.id == "queue"
    ]
    found = []
    for n in ast.walk(execute):
        if isinstance(n, ast.Yield) and isinstance(n.value, ast.Constant) and n.value.value == "":
            inside = any(h.lineno <= n.lineno <= h.end_lineno for h in handlers)
            after = bool(starts) and min(starts) < n.lineno
            found.append((n.lineno, inside and after))
    return found


def test_tw17_the_executor_yields_its_keepalive_only_after_the_request_is_sent():
    src = (REPO / "opti_oignon" / "executor.py").read_text(encoding="utf-8")
    found = _empty_yields(src)
    assert len(found) == 1 and found[0][1], (
        f"exactly one empty yield, in `except queue.Empty` after the request thread started: {found}"
    )
    marker = "        stream_thread_obj.start()\n"
    assert src.count(marker) == 1
    control = src.replace(marker, marker + '        yield ""\n')
    assert len(_empty_yields(control)) == 2, "control: the census sees a second empty yield"


def test_tw18_each_command_draws_its_own_frames_in_order(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        for kind in ("waiting", "closing", "opening"):
            frames = out.WAIT_FRAMES[kind]
            waiter, stream, clock, _ = _unit(out)
            armed = clock()
            waiter.arm(kind, armed)
            drawn = []
            for k in range(2 * len(frames)):
                before = len(stream.writes)
                waiter.tick(armed + 0.400 + k * 0.150 + 0.001)
                if len(stream.writes) > before:
                    drawn.append((k, stream.writes[-1][0]))
            for k, text in drawn:
                assert text[1:].endswith(frames[k % len(frames)]), (kind, k, text)
            assert len(drawn) >= len(set(frames)), kind

        scripts = {"/close": [("advance", 1.0), ("info", "closed")], "/open x": [("advance", 1.0), ("info", "opened")],
                   "plain": [("advance", 1.0), ("token", "ok")]}
        for line, label in (("/close\n", "closing"), ("/open x\n", "opening"), ("plain\n", "waiting")):
            _, stream = _invoke(loaded, [line], scripts)
            frames = _frames(out, stream)
            assert frames and all(f[1:].strip().startswith(label) for f in frames), (line, frames)
    finally:
        restore()


_SESSION = {
    "hello": [("advance", 1.0), ("info", "conversation conv-1"), ("advance", 2.5), ("token", ""), ("advance", 1.0), ("token", "Hi"), ("token", " there")],
    "/close": [("advance", 1.5), ("info", "closed conv-1: 1 span(s) evicted")],
    "/open conv-1": [("advance", 1.0), ("info", "opened conv-1")],
}


def test_tw19_stdout_is_byte_identical_with_and_without_animations(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        lines = ["hello\n", "/close\n", "/open conv-1\n"]
        on, drawn = _invoke(loaded, lines, _SESSION, tty=True)
        off, silent = _invoke(loaded, lines, _SESSION, tty=False)
        assert _frames(out, drawn) and silent.writes == [], "control: frames with, none without"
        assert on.stdout == off.stdout, "stdout does not depend on the animations"
        assert "\r" not in on.stdout and "(@)" not in on.stdout
    finally:
        restore()


def _screen(data):
    rows, col = [[]], 0
    for ch in data:
        if ch == "\r":
            col = 0
        elif ch == "\n":
            rows.append([])
            col = 0
        else:
            row = rows[-1]
            while len(row) < col:
                row.append(" ")
            if col < len(row):
                row[col] = ch
            else:
                row.append(ch)
            col += 1
    return ["".join(row) for row in rows]


def test_tw20_replaying_everything_the_waiter_wrote_leaves_a_blank_screen(monkeypatch, tmp_path):
    assert _screen("ab\rc") == ["cb"], "control: the emulator overwrites on carriage return"
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        _, stream = _invoke(loaded, ["hello\n", "/close\n", "/open conv-1\n"], _SESSION)
        labels = {f[1:].strip().split()[0] for f in _frames(out, stream)}
        assert labels == set(_KINDS), f"control: every kind was drawn at least once ({labels})"
        rows = _screen(stream.data)
        assert all(row.strip() == "" for row in rows), rows
    finally:
        restore()


def test_tw21_ctrl_c_during_a_wait_clears_the_line_and_exits_130(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        scripts = {
            "in-generator": [("advance", 1.0), ("interrupt",)],
            "in-loop": [("advance", 1.0), ("event", _RaisingText())],
        }
        for line in ("in-generator\n", "in-loop\n"):
            result, stream = _invoke(loaded, [line], scripts)
            assert result.exit_code == 130, (line, result.exit_code)
            assert _frames(out, stream), "control: a frame was drawn before Ctrl-C"
            assert stream.writes[-1][0] == out.WAIT_CLEAR, "the last write is the clear"
            assert "Error: interrupted" in result.stderr
    finally:
        restore()


# ---------------------------------------------------------------------------
# TW22-TW25 -- the engine
# ---------------------------------------------------------------------------
def test_tw22_a_stopped_wait_writes_nothing_and_a_new_one_draws_again(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        waiter, stream, clock, _ = _unit(out)
        waiter.arm("waiting", clock())
        clock.advance(1.0)
        waiter.tick()
        waiter.stop()
        count = len(stream.writes)
        assert count == 2 and stream.writes[-1][0] == out.WAIT_CLEAR
        clock.advance(1.0)
        waiter.tick()
        waiter.stop()
        assert len(stream.writes) == count, "a tick or a second stop after the stop writes nothing"
        waiter.arm("waiting", clock())
        clock.advance(1.0)
        waiter.tick()
        assert len(stream.writes) == count + 1, "a new wait draws again"
    finally:
        restore()


def test_tw23_stop_returns_within_its_bound_when_a_write_is_stalled(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        waiter, stream, clock, _ = _unit(out, stop=50)
        waiter.arm("waiting", clock())
        clock.advance(1.0)
        waiter.tick()
        count = len(stream.writes)
        waiter._lock.acquire()
        stopper = threading.Thread(target=waiter.stop, daemon=True)
        try:
            stopper.start()
            stopper.join(2.0)
            assert not stopper.is_alive(), "stop returned while the frame lock was held"
        finally:
            waiter._lock.release()
        stopper.join(2.0)
        clock.advance(1.0)
        waiter.tick()
        assert len(stream.writes) == count, "nothing is written after a stop that could not clear"
    finally:
        restore()


def test_tw24_a_broken_stream_switches_the_waiter_off_without_raising(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        waiter, stream, clock, _ = _unit(out, tty=_TTY(fail=1))
        waiter.arm("waiting", clock())
        clock.advance(1.0)
        waiter.tick()
        waiter.stop()
        waiter.bye()
        waiter.arm("waiting", clock())
        clock.advance(1.0)
        waiter.tick()
        waiter.stop()
        assert stream.writes == [], "after one failed write the waiter stays off"
    finally:
        restore()


def test_tw25_the_frame_thread_starts_at_arm_and_is_a_daemon(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        waiter, _, clock, threads = _unit(out)
        assert threads.made == [], "constructing the waiter starts nothing"
        waiter.arm("waiting", clock())
        assert len(threads.made) == 1
        made = threads.made[0]
        assert made.daemon is True and made.started is True and made.name == "oo-wait"
    finally:
        restore()


class _SteppingEvent(threading.Event):
    """An event whose every wait moves the fake clock one interval, and which sets itself after `limit` waits."""

    def __init__(self, clock, limit):
        super().__init__()
        self.clock, self.limit, self.waits = clock, limit, 0

    def wait(self, timeout=None):
        self.waits += 1
        self.clock.advance(timeout)
        if self.waits >= self.limit:
            self.set()
        return self.is_set()


def test_tw31_the_frame_threads_loop_draws_per_interval_and_ends_when_stopped(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        waiter, stream, clock, threads = _unit(out, delay=400, interval=150)
        waiter.arm("waiting", clock())
        made = threads.made[0]
        gen, _ = made.args
        stepping = _SteppingEvent(clock, limit=10)
        made.target(gen, stepping)
        assert stepping.waits == 10, "the loop waited one interval at a time until its wait was stopped"
        frames = _frames(out, stream)
        assert len(frames) >= 5, "past the grace delay, the loop drew a frame per interval"
        assert frames[0][1:].strip().startswith("waiting")

        silent = _SteppingEvent(clock, limit=3)
        silent.set()
        before = len(stream.writes)
        made.target(gen, silent)
        assert len(stream.writes) == before and silent.waits == 0, "a stopped wait's loop draws nothing and ends at once"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TW26-TW28 -- configuration
# ---------------------------------------------------------------------------
def test_tw26_a_bad_animation_setting_falls_back_alone_and_keeps_the_others(monkeypatch, tmp_path):
    bad = {
        "api_url": "http://remote:9000", "timeout": 7, "animations": "maybe",
        "animation_interval_ms": "fast", "animation_delay_ms": 5, "animation_stop_ms": True,
    }
    loaded, restore = _open(monkeypatch, tmp_path, cli_yaml=bad)
    try:
        cfg = loaded["opti_oignon.cli.config"].load_config()
        assert cfg.api_url == "http://remote:9000" and cfg.timeout == 7, "the other settings survive"
        assert cfg.animations is False, "an unreadable switch is off"
        assert (cfg.animation_interval_ms, cfg.animation_delay_ms, cfg.animation_stop_ms) == (150, 400, 100)
    finally:
        restore()
    good = {"animations": "yes", "animation_interval_ms": 200, "animation_delay_ms": 800, "animation_stop_ms": 50}
    loaded, restore = _open(monkeypatch, tmp_path, cli_yaml=good)
    try:
        cfg = loaded["opti_oignon.cli.config"].load_config()
        assert cfg.animations is True
        assert (cfg.animation_interval_ms, cfg.animation_delay_ms, cfg.animation_stop_ms) == (200, 800, 50)
    finally:
        restore()


def test_tw27_the_animation_settings_survive_a_save_and_are_not_switched_off_by_colour(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        config = loaded["opti_oignon.cli.config"]
        path = tmp_path / "saved.yaml"
        config.CLIConfig(animations=False, animation_interval_ms=300, animation_delay_ms=900, animation_stop_ms=20).save(path)
        again = config.load_config(path)
        assert (again.animations, again.animation_interval_ms, again.animation_delay_ms, again.animation_stop_ms) == (False, 300, 900, 20)
        monkeypatch.setenv("NO_COLOR", "1")
        coloured_off = config.CLIConfig().to_dict()
        assert coloured_off["color"] is False and coloured_off["animations"] is True, (
            "NO_COLOR silences the waiter at run time; it is never written into the animations setting"
        )
    finally:
        restore()


def test_tw28_config_set_accepts_the_switch_and_refuses_an_out_of_range_interval(monkeypatch, tmp_path):
    from click.testing import CliRunner

    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        main = loaded["opti_oignon.cli.main"]
        path = tmp_path / "xdg" / "opti-oignon" / "cli.yaml"
        runner = CliRunner(mix_stderr=False)
        ok = runner.invoke(main.cli, ["config", "set", "animations", "false"])
        assert ok.exit_code == 0, ok.output
        assert yaml.safe_load(path.read_text(encoding="utf-8"))["animations"] is False
        before = path.read_bytes()
        refused = runner.invoke(main.cli, ["config", "set", "animation_interval_ms", "20"])
        assert refused.exit_code == 1
        assert "animation_interval_ms" in refused.stderr and "50" in refused.stderr, refused.stderr
        assert path.read_bytes() == before, "a refused value leaves the file as it was"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TW29-TW30 -- the ends
# ---------------------------------------------------------------------------
def test_tw29_quit_says_bye_and_nothing_else_does(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        out = loaded["opti_oignon.cli.output"]
        scripts = {"/quit": [("quit", "bye")], "hello": [("advance", 1.0), ("token", "Hi")], "stop": [("advance", 1.0), ("interrupt",)]}
        _, stream = _invoke(loaded, ["/quit\n"], scripts)
        assert stream.data.endswith(out.WAIT_BYE) and out.WAIT_BYE == "  (@)/  bye\n"
        _, silent = _invoke(loaded, ["/quit\n"], scripts, tty=False)
        assert silent.writes == [], "no farewell with the gate off"
        _, eof = _invoke(loaded, ["hello\n"], scripts)
        assert out.WAIT_BYE not in eof.data, "no farewell at end of input"
        _, interrupted = _invoke(loaded, ["stop\n"], scripts)
        assert out.WAIT_BYE not in interrupted.data, "no farewell on Ctrl-C"
    finally:
        restore()


def test_tw30_a_last_line_without_a_newline_arms_no_wait(monkeypatch, tmp_path):
    loaded, restore = _open(monkeypatch, tmp_path)
    try:
        _, bare = _invoke(loaded, ["hello"], _TURN)
        assert bare.writes == [], "a line cut off by end of input arms nothing"
        _, control = _invoke(loaded, ["hello\n"], _TURN)
        assert len(control.writes) > 0, "control: the same line with a newline draws"
    finally:
        restore()
