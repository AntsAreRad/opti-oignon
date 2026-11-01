"""Contracts for the tempdir backend's commands: every process of a command dies with it, where a PID namespace can.

On the tempdir backend (no bubblewrap) a command ran as ``unshare --user --net -- bash -c COMMAND``: a process that
outlived it -- a background job once the command returned, or one moved out of its group with ``setsid`` -- ran on,
out of reach of the stop button and of a destroy, on a workspace being removed. Now the command runs, where the
system allows it, in a PID namespace of its own as well (``--pid --fork --kill-child``), whose first process is a
shell that runs the command as its foreground child: when that shell ends -- the command returned, or a stop or a
destroy killed it -- the kernel kills every process left in the namespace, and the command can neither make that
shell leave its group nor clear its death signal (round 1; a foreground child since round 2, so that the command's
signals, traps and stdin are as before). Where the PID namespace is refused, the user and network namespace alone, then plain bash,
as before, and the log says which.

  * PJ1 -- NOTHING OUTLIVES A COMMAND: through ``execute_command`` on the tempdir backend where the PID form works, a
    command that starts a background job and a ``setsid`` one and returns: neither survives it; a command whose jobs
    run when a destroy comes: none survives the destroy; a command that sets out to stay -- ``exec setsid`` into a
    process that clears its own death signal -- does not survive the destroy either (round 1).
  * PJ2 -- THE STRONGEST FORM THAT WORKS: the detection tries the PID form first, then the user and network form, then
    none, and says which (a stand-in ``subprocess.run`` refusing the forms one by one); the command line follows it.
  * PJ3 -- WHAT STAYS: a command's output and return code as before, and its network cut (its namespace's
    ``/proc/net/dev`` lists the loopback alone); a command's shell is no namespace's first process: it is PID 2, as
    under bubblewrap, and a command that dies of a signal is answered 128 plus its number, as under bubblewrap
    (``kill $$``: 143; before, -15), with no line of the first process's in its stderr (round 1, round 2); a trap on
    SIGINT catches the command's own SIGINT, as before (round 2: a background child would have had SIGINT ignored).
    A background job's output after its command returned is lost, as under bubblewrap.

The PID form is probed on the machine; where it is refused, PJ1 and PJ3 say so and skip.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402

BUDGET_S = {
    "test_pj1_nothing_outlives_a_command": 10.0,
    "test_pj2_the_strongest_form_that_works": 5.0,
    "test_pj3_what_a_namespaced_command_keeps": 5.0,
}

_PID_FORM = ["unshare", "--user", "--net", "--pid", "--fork", "--kill-child", "--"]
_INIT = 'exec 4>&2 2>/dev/null; bash -c "$1" 2>&4 4>&-; exit "$?"'

# This run's markers: a concurrent run of the suite never sees, nor kills, another's processes
_RUN = f"{random.randrange(10**5, 10**6)}"


def _marker(n):
    """The sleep argument of this run's process ``n``: ``47.<n><run>``."""
    return f"47.{n}{_RUN}"


def _pid_form_works():
    try:
        done = subprocess.run([*_PID_FORM, "echo", "pj-ok"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and "pj-ok" in done.stdout


def _running_with(marker):
    """The pids of this user's processes that run ``sleep`` (or a ``python3``) on ``marker``, this process never."""
    found = []
    me, mine = os.getuid(), os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == mine:
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid != me:
                continue
            with open(f"/proc/{entry}/cmdline", "rb") as cmdline:
                words = cmdline.read().split(b"\0")
        except OSError:
            continue
        if words[0] in (b"sleep", b"python3") and any(marker.encode("ascii") in word for word in words[1:]):
            found.append(int(entry))
    return found


def _settle(marker, deadline=3.0):
    """The processes holding ``marker`` once they had time to die, or what still runs at the deadline."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        alive = _running_with(marker)
        if not alive:
            return []
        time.sleep(0.05)
    return _running_with(marker)


def _kill_left(*markers):
    for marker in markers:
        for pid in _running_with(marker):
            try:
                os.kill(pid, 9)
            except OSError:
                pass


def test_pj1_nothing_outlives_a_command(tmp_path):
    if not _pid_form_works():
        pytest.skip("this system refuses an unprivileged PID namespace: the tempdir backend runs without one")
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path, strict_mode=False)
        sid = manager.create_sandbox("pj1").session_id
        m = {n: _marker(n) for n in range(1, 7)}
        result = manager.execute_command(
            sid, f"sleep {m[1]} > /dev/null 2>&1 & setsid sleep {m[2]} > /dev/null 2>&1 & echo pj1")
        returned = _settle(m[1]) + _settle(m[2])
        # A destroy while the command's jobs run
        answers = []
        thread = threading.Thread(
            target=lambda: answers.append(
                manager.execute_command(sid, f"sleep {m[3]} & setsid sleep {m[4]} & sleep {m[5]}")),
            daemon=True)
        thread.start()
        end = time.monotonic() + 5
        while time.monotonic() < end and not (_running_with(m[3]) and _running_with(m[4])):
            time.sleep(0.05)
        started = bool(_running_with(m[3])) and bool(_running_with(m[4]))
        manager.destroy_sandbox(sid)
        thread.join(timeout=8)
        destroyed = _settle(m[3]) + _settle(m[4]) + _settle(m[5])
        # A command that sets out to stay: out of its group, its death signal cleared
        sid = manager.create_sandbox("pj1-staying").session_id
        staying = threading.Thread(target=lambda: manager.execute_command(sid, (
            'exec setsid python3 -c "import ctypes, time; ctypes.CDLL(None).prctl(1, 0, 0, 0, 0); '
            f'time.sleep({m[6]})"')), daemon=True)
        staying.start()
        end = time.monotonic() + 5
        while time.monotonic() < end and not _running_with(m[6]):
            time.sleep(0.05)
        stayed_up = bool(_running_with(m[6]))
        manager.destroy_sandbox(sid)
        staying.join(timeout=8)
        outstayed = _settle(m[6])
    finally:
        _kill_left(*(_marker(n) for n in range(1, 7)))
        restore()
    assert result.return_code == 0 and "pj1" in result.stdout, (result.return_code, result.stdout, result.stderr)
    assert returned == [], f"a background job or a setsid one outlived its command: {returned}"
    assert started, "the command's jobs never ran"
    assert destroyed == [], f"a command's job outlived its sandbox's destroy: {destroyed}"
    assert stayed_up, "the staying command never ran"
    assert outstayed == [] and not staying.is_alive(), (
        f"a command that left its group and cleared its death signal outlived the destroy: {outstayed}")


def test_pj2_the_strongest_form_that_works(tmp_path, monkeypatch, caplog):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    real_run = subprocess.run
    outcomes = {}
    try:
        for refused in ((), ("--pid",), ("--pid", "--net")):
            tried = []

            def standing_in(argv, *args, _refused=refused, _tried=tried, **kwargs):
                if argv and argv[0] == "unshare":
                    _tried.append(tuple(argv))
                    if any(flag in argv for flag in _refused):
                        return subprocess.CompletedProcess(argv, 1, "", "unshare: refused")
                    return subprocess.CompletedProcess(argv, 0, "unshare-ok\n", "")
                return real_run(argv, *args, **kwargs)

            monkeypatch.setattr(subprocess, "run", standing_in)
            caplog.clear()
            caplog.set_level(logging.INFO)
            manager = ce_t._manager(sm, tmp_path)
            line = manager._build_tempdir_command("echo pj2")
            monkeypatch.undo()
            said = [record.getMessage() for record in caplog.records if "Tempdir mode" in record.getMessage()]
            outcomes[refused] = (line, len(tried), said[-1] if said else "")
    finally:
        monkeypatch.undo()
        restore()
    strongest, net_only, none = outcomes[()], outcomes[("--pid",)], outcomes[("--pid", "--net")]
    assert strongest[0] == [*_PID_FORM, "bash", "-c", _INIT, "sandbox-init", "echo pj2"] and strongest[1] == 1, strongest
    assert "--pid available" in strongest[2], strongest[2]
    assert net_only[0] == ["unshare", "--user", "--net", "--", "bash", "-c", "echo pj2"] and net_only[1] == 2, net_only
    assert "no PID namespace" in net_only[2], net_only[2]
    assert none[0] == ["bash", "-c", "echo pj2"] and none[1] == 2, none
    assert "NO network isolation" in none[2], none[2]


def test_pj3_what_a_namespaced_command_keeps(tmp_path):
    if not _pid_form_works():
        pytest.skip("this system refuses an unprivileged PID namespace: the tempdir backend runs without one")
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path, strict_mode=False)
        sid = manager.create_sandbox("pj3").session_id
        said = manager.execute_command(sid, "echo pj3-out; echo pj3-err >&2; exit 3")
        net = manager.execute_command(sid, "cat /proc/net/dev")
        itself = manager.execute_command(sid, "echo pid=$$; kill $$; echo after")
        killed = manager.execute_command(sid, "echo a; kill -9 $$; echo b")
        # A command inherits this process's signal dispositions, and a shell cannot trap a signal ignored when it
        # starts: SIGINT handled here (a handler resets to the default in a child), whatever the runner ignores
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            trapped = manager.execute_command(sid, 'trap "echo trapped" INT; kill -INT $$; echo after')
        finally:
            signal.signal(signal.SIGINT, previous)
        manager.destroy_sandbox(sid)
    finally:
        restore()
    assert said.return_code == 3 and "pj3-out" in said.stdout and "pj3-err" in said.stderr, (
        said.return_code, said.stdout, said.stderr)
    interfaces = [line.split(":")[0].strip() for line in net.stdout.splitlines() if ":" in line]
    assert net.return_code == 0 and interfaces == ["lo"], f"the command saw the network: {interfaces}"
    assert (itself.return_code, itself.stdout.strip()) == (143, "pid=2"), (
        f"a command's shell is its namespace's first process: {itself.return_code} {itself.stdout!r}")
    assert (killed.return_code, killed.stdout.strip(), killed.stderr.strip()) == (137, "a", ""), (
        f"a command killed by a signal is answered otherwise: {killed.return_code} {killed.stdout!r} {killed.stderr!r}")
    assert trapped.stdout.split() == ["trapped", "after"], (
        f"a command's trap on SIGINT never ran (SIGINT ignored): {trapped.stdout!r}")
