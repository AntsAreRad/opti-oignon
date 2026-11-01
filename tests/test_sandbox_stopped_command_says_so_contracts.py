"""Contracts for a command a stop or a destroy ended: its answer says so.

A sandbox command killed by the stop button (``stop_command``) or by its sandbox's destroy (the destroy route,
``cleanup_all``, the emergency stop's "destroy sandboxes" step) was answered as any command a signal ended: its
return code, its output so far, and nothing else. The panel, the chat's code execution and the agent's bash tool --
which hands the answer to the model -- could not tell it from a command that killed itself or ran out of memory.
Now the stop and the destroy mark each process they take whose leader still runs, under the manager's lock, before
they kill it (the first mark kept); the spawn both runners use takes the mark out when the process ends and hands it
on with its answer when its leader died of their SIGKILL; and the answer's stderr ends with one line, after any
truncation, its ``stopped`` saying which.

  * SB1 -- A STOP SAYS SO: a command running when ``stop_command`` comes, through ``execute_command`` on the tempdir
    backend, and through the spawn and the output processing the bubblewrap runner chains: its stderr so far, then
    "Command stopped by the stop button"; ``stopped`` "stop"; its stdout so far kept; its return code SIGKILL's, as
    before.
  * SB2 -- A DESTROY SAYS SO: a command running when its sandbox is destroyed: "Command stopped: its sandbox was
    destroyed", ``stopped`` "destroy", also when the destroy has dropped the session's registry before the spawn
    untracks the command (round 2); a stop then a destroy on one command (the destroy between the stop's mark and its
    kill): the stop's line alone; a destroy then a stop: the destroy's alone.
  * SB3 -- WHAT STAYS: a command that kills itself (``kill -9 $$``, its leader dead of SIGKILL where no PID namespace
    stands between: round 1) or ends on its own says nothing more; a timeout is answered as before; a stderr past its
    cap is truncated, the line after the truncation mark. A command whose leader had ended when the stop or the
    destroy took it says nothing (rounds 1 and 2): already reaped, not yet untracked; ended but not reaped, a job of
    its group holding its output, which the stop then kills -- the leader exited or killed itself; and a leader that
    ends on its own between the stop's look and its kill. No mark outlives its process.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402

BUDGET_S = {
    "test_sb1_a_stop_says_so": 10.0,
    "test_sb2_a_destroy_says_so": 10.0,
    "test_sb3_what_a_command_answer_keeps": 10.0,
}

_STOP_LINE = "Command stopped by the stop button"
_DESTROY_LINE = "Command stopped: its sandbox was destroyed"
_PATH = {"PATH": "/usr/bin:/bin"}


def _ready(workspace, name="ready", deadline=5.0):
    """Whether the command wrote ``name`` in its workspace before the deadline."""
    path = Path(workspace) / name
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


def _running(manager, sid, command, answers, then="sleep 40"):
    """Run ``command`` in the session through ``execute_command`` in a thread, its answer landing in ``answers``;
    returns the thread once the command wrote ``ready`` in its workspace, then runs ``then`` (by default it sleeps
    until killed, or its 30 s run out)."""
    workspace = manager.get_workspace_path(sid)
    # A command run before in the session left its own
    (Path(workspace) / "ready").unlink(missing_ok=True)

    def run():
        try:
            answers.append(manager.execute_command(sid, f"{command}; : > ready; {then}", timeout=30))
        except Exception as exc:  # noqa: BLE001 - the answer is what the contract reads
            answers.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert _ready(workspace), f"the command never ran: {answers}"
    return thread


def _spawning(manager, sid, script, answers):
    """Run ``bash -c script`` in the session through the tracked spawn alone (no namespace), in a thread."""
    workspace = manager.get_workspace_path(sid)
    thread = threading.Thread(target=lambda: answers.append(manager._spawn_tracked(
        ["bash", "-c", script], timeout=30, session_id=sid, cwd=workspace, env=_PATH)), daemon=True)
    thread.start()
    return thread


def _state(pid):
    """The state letter of process ``pid`` ("Z": ended, not reaped), None once it is gone -- reaped before its file
    was opened, or between its opening and its reading (ESRCH)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as stat_file:
            return stat_file.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return None


def _leader_ended(manager, sid, deadline=5.0):
    """Whether the session's one tracked process ended -- a zombie, not yet reaped -- before the deadline."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        with manager._lock:
            procs = list(manager._running_procs.get(sid, ()))
        if len(procs) == 1 and _state(procs[0].pid) == "Z":
            return True
        time.sleep(0.02)
    return False


class _Gate:
    """The manager's lock, at whose next taking one thread, once named, waits until let through."""

    def __init__(self, lock):
        self.lock = lock
        self.thread = None
        self.waiting = threading.Event()
        self.through = threading.Event()

    def __enter__(self):
        if threading.current_thread() is self.thread and not self.through.is_set():
            self.waiting.set()
            self.through.wait(5)
        return self.lock.__enter__()

    def __exit__(self, *exc):
        return self.lock.__exit__(*exc)


def _said(answers):
    assert len(answers) == 1 and not isinstance(answers[0], Exception), answers
    said = answers[0]
    return said.stdout, said.stderr, getattr(said, "stopped", None), said.return_code, said.timed_out


def test_sb1_a_stop_says_so(tmp_path):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid = manager.create_sandbox("sb1").session_id
        answers = []
        thread = _running(manager, sid, "echo sb1-out; echo sb1-err >&2", answers)
        stopped = manager.stop_command(sid)
        thread.join(timeout=8)
        # The bubblewrap runner's path: the same spawn, its answer through the same output processing
        workspace = manager.get_workspace_path(sid)
        spawned = []
        spawn = _spawning(manager, sid, "echo sb1-raw; : > ready2; sleep 40", spawned)
        raw_ran = _ready(workspace, "ready2")
        raw_stopped = manager.stop_command(sid)
        spawn.join(timeout=8)
        raw = manager._process_output(spawned[0], 30) if spawned else None
        marks = dict(getattr(manager, "_stopped", {}))
    finally:
        restore()
    assert stopped is True, "the stop found no command running"
    assert _said(answers) == ("sb1-out\n", "sb1-err\n" + _STOP_LINE, "stop", -signal.SIGKILL, False), (
        f"a command the stop button ended is answered otherwise: {_said(answers)}")
    assert raw_ran and raw_stopped is True and len(spawned) == 1, (raw_ran, raw_stopped, spawned)
    assert (getattr(spawned[0], "stopped", None), spawned[0].returncode) == ("stop", -signal.SIGKILL), (
        f"the spawn's answer does not say the stop ended it: {spawned[0]!r}")
    assert (raw.stdout, raw.stderr, getattr(raw, "stopped", None), raw.return_code) == (
        "sb1-raw\n", _STOP_LINE, "stop", -signal.SIGKILL), (
        f"the bubblewrap runner's processing answers a stopped command otherwise: {raw!r}")
    assert marks == {}, f"a mark outlived its process: {marks}"


def test_sb2_a_destroy_says_so(tmp_path, monkeypatch):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid = manager.create_sandbox("sb2").session_id
        # The spawn held at its untracking until the destroy, which drops the session's registry, has returned
        gate = manager._lock = _Gate(manager._lock)
        answers = []
        thread = _running(manager, sid, "echo sb2-out", answers)
        gate.thread = thread
        destroyed = manager.destroy_sandbox(sid)
        held = gate.waiting.wait(5)
        gate.through.set()
        thread.join(timeout=8)
        manager._lock = gate.lock
        # One command, the second word coming between the first's mark and its kill
        real_kill = sm._kill_group
        orders = {}
        for first, second in (("stop", "destroy"), ("destroy", "stop")):
            sid = manager.create_sandbox(f"sb2-{first}").session_id
            calls = []

            def kill(proc, sid=sid, second=second, calls=calls):
                if not calls:
                    calls.append(proc)
                    (manager.destroy_sandbox if second == "destroy" else manager.stop_command)(sid)
                real_kill(proc)

            monkeypatch.setattr(sm, "_kill_group", kill)
            got = []
            thread = _running(manager, sid, "true", got)
            (manager.stop_command if first == "stop" else manager.destroy_sandbox)(sid)
            thread.join(timeout=8)
            monkeypatch.setattr(sm, "_kill_group", real_kill)
            orders[first] = (got, len(calls))
        marks = dict(getattr(manager, "_stopped", {}))
    finally:
        restore()
    assert destroyed is True and held, (destroyed, held)
    assert _said(answers) == ("sb2-out\n", _DESTROY_LINE, "destroy", -signal.SIGKILL, False), (
        f"a command its sandbox's destroy ended is answered otherwise: {_said(answers)}")
    stop_first, stops = orders["stop"]
    destroy_first, destroys = orders["destroy"]
    assert stops == 1 and destroys == 1, "the second word never came between the first's mark and its kill"
    assert _said(stop_first) == ("", _STOP_LINE, "stop", -signal.SIGKILL, False), (
        f"a stop then a destroy: the destroy changed what the answer says: {_said(stop_first)}")
    assert _said(destroy_first) == ("", _DESTROY_LINE, "destroy", -signal.SIGKILL, False), (
        f"a destroy then a stop: the stop changed what the answer says: {_said(destroy_first)}")
    assert marks == {}, f"a mark outlived its process: {marks}"


def test_sb3_what_a_command_answer_keeps(tmp_path, monkeypatch):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path, max_stderr_bytes=64)
        sid = manager.create_sandbox("sb3").session_id
        itself = manager.execute_command(sid, "echo a; kill -9 $$; echo b")
        alone = []
        _spawning(manager, sid, "kill -9 $$", alone).join(timeout=8)
        own = manager.execute_command(sid, "echo fine; echo own >&2; exit 4")
        late = manager.execute_command(sid, "sleep 40", timeout=1)
        # A stderr past its cap, then the stop
        capped = []
        thread = _running(manager, sid, "head -c 200 /dev/zero | tr '\\0' e >&2", capped)
        manager.stop_command(sid)
        thread.join(timeout=8)
        # A leader that ended -- exited, or killed itself -- while a job of its group holds its output: the stop or
        # the destroy kills the job, and the command is answered as its leader ended
        holding = {}
        for name, end, word in (("exited", "exit 0", "stop"), ("killed", "kill -9 $$", "stop"),
                                ("destroyed", "kill -9 $$", "destroy")):
            held_sid = sid if word == "stop" else manager.create_sandbox("sb3-held").session_id
            got = []
            hold = _spawning(manager, held_sid, f"sleep 40 & : > {name}; {end}", got)
            ended = _ready(manager.get_workspace_path(held_sid), name) and _leader_ended(manager, held_sid)
            took = (manager.stop_command if word == "stop" else manager.destroy_sandbox)(held_sid)
            hold.join(timeout=8)
            holding[name] = (ended, took, got)
        # A stop or a destroy that takes a command already ended on its own and reaped, before its spawn untracks it
        reaped = {}
        for word in ("stop", "destroy"):
            sid = manager.create_sandbox(f"sb3-{word}").session_id
            gate = manager._lock = _Gate(manager._lock)
            got = []
            thread = _running(manager, sid, "true", got, then="sleep 0.2")
            gate.thread = thread
            at_gate = gate.waiting.wait(5)
            took = (manager.stop_command if word == "stop" else manager.destroy_sandbox)(sid)
            gate.through.set()
            thread.join(timeout=8)
            manager._lock = gate.lock
            reaped[word] = (at_gate, took, got)
        # A leader that ends on its own between the stop's look and its kill
        sid = manager.create_sandbox("sb3-late").session_id
        real_kill = sm._kill_group

        def late_kill(proc):
            end = time.monotonic() + 5
            while time.monotonic() < end and _state(proc.pid) not in ("Z", None):
                time.sleep(0.01)
            real_kill(proc)

        monkeypatch.setattr(sm, "_kill_group", late_kill)
        slow = []
        thread = _running(manager, sid, "true", slow, then="sleep 0.3")
        slow_took = manager.stop_command(sid)
        thread.join(timeout=8)
        monkeypatch.setattr(sm, "_kill_group", real_kill)
        marks = dict(getattr(manager, "_stopped", {}))
    finally:
        restore()
    assert (itself.stdout, itself.stderr.strip(), getattr(itself, "stopped", None)) == ("a\n", "", ""), (
        f"a command that killed itself is answered otherwise: {itself!r}")
    assert len(alone) == 1 and (alone[0].returncode, getattr(alone[0], "stopped", None)) == (-signal.SIGKILL, ""), (
        f"a command whose leader killed itself is told stopped: {alone}")
    assert (own.stdout, own.stderr, own.return_code, getattr(own, "stopped", None)) == ("fine\n", "own\n", 4, ""), (
        f"a command that ended on its own is answered otherwise: {own!r}")
    assert (late.timed_out, late.stderr, late.return_code, getattr(late, "stopped", None)) == (
        True, "Command timed out after 1s", -1, ""), f"a timeout is answered otherwise: {late!r}"
    said = _said(capped)
    assert said == ("", "e" * 64 + "\n... [STDERR TRUNCATED]\n" + _STOP_LINE, "stop", -signal.SIGKILL, False), (
        f"a stopped command's stderr past its cap is answered otherwise: {said}")
    assert capped[0].truncated_stderr is True
    for name, code in (("exited", 0), ("killed", -signal.SIGKILL), ("destroyed", -signal.SIGKILL)):
        ended, took, got = holding[name]
        assert ended and took is True and len(got) == 1, (name, ended, took, got)
        assert (got[0].returncode, getattr(got[0], "stopped", None)) == (code, ""), (
            f"a command whose leader had ended ({name}) when the word came is told stopped: {got[0]!r}")
    for word, (at_gate, took, got) in reaped.items():
        assert at_gate and took is True, f"the {word} never took a command already reaped: {(at_gate, took)}"
        assert _said(got) == ("", "", "", 0, False), (
            f"a {word} that took a command already reaped says it stopped it: {_said(got)}")
    assert slow_took is True and _said(slow) == ("", "", "", 0, False), (
        f"a leader that ended on its own before the stop's kill is told stopped: {slow_took} {_said(slow)}")
    assert marks == {}, f"a mark outlived its process: {marks}"
