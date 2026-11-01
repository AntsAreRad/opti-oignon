"""Contracts for a sandbox's running commands: a destroy stops them, and every one of them is tracked.

``destroy_sandbox`` never stopped the session's running command: it removed the workspace and dropped the session
while the command's process group ran on until it ended or timed out -- through the destroy route, ``cleanup_all``
and the emergency stop's "destroy sandboxes" step alike. The registry of running commands held one process per
session: a second command of the same session overwrote the first's entry, and the first's end removed the second's,
which then ran untracked -- out of reach of ``stop_command`` and of a destroy, the session shown idle. And a command
whose session a destroy ended after its caller checked it registered anyway. Now the registry keeps every running
process of a session, a destroy kills them before it removes the workspace, and a spawn whose session is no longer
active kills what it started and refuses.

  * KR1 -- A DESTROY STOPS WHAT RUNS: a command running in the session (a real ``sleep``, through the spawn both
    runners use): the destroy kills its process group before it removes the workspace -- dead when the removal
    starts -- and the command's caller gets its answer at once; a second destroy racing the first kills it too before
    its own removal (round 1), and so does a destroy racing a stop (round 2). (On the tempdir backend a process that
    outlives its command -- a background job once the command returned, or one moved out of its group with
    ``setsid`` -- is out of reach, as of the stop button; bubblewrap's PID namespace takes it down.)
  * KR2 -- EVERY COMMAND IS TRACKED: two commands of one session at once: both tracked, the session running until
    both end; ``stop_command`` kills both; a short command ending beside a long one leaves the long one tracked and
    the session running.
  * KR3 -- NOTHING STARTS FOR A DESTROYED SESSION: a command whose session a destroy ends between its caller's check
    and its spawn, the destroy still removing the workspace (held there): killed, never left running, and its
    caller told.
  * KR4 -- WHAT STAYS: ``stop_command`` on an idle session answers False; a command that ends on its own leaves the
    session idle, its registry entry gone.
  * KR5 -- THE IDLE SWEEP NEVER KILLS A COMMAND (round 1): the sweep decides on a snapshot; a command started since
    -- or one starting now, which counts as activity -- keeps its session: the destroy checks again under its lock;
    so do activity since the snapshot, a binding since, and a destroy already begun (round 2); a command refused
    before it runs is no activity (round 2).
  * KR6 -- A FREED PID IS NEVER SIGNALLED (round 1): a command whose leader is already reaped, still in the registry,
    and one whose time runs out after a destroy killed it: no signal is sent to its pid.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402

BUDGET_S = {
    "test_kr1_a_destroy_stops_what_runs": 10.0,
    "test_kr2_every_command_is_tracked": 10.0,
    "test_kr3_nothing_starts_for_a_destroyed_session": 10.0,
    "test_kr4_what_the_running_registry_keeps": 5.0,
    "test_kr5_the_idle_sweep_never_kills_a_command": 10.0,
    "test_kr6_a_freed_pid_is_never_signalled": 10.0,
}

# Long enough that a command still running when its contract ends was never stopped
_SLEEP = "40"


def _alive(pid):
    """Whether the process ``pid`` still runs (a zombie does not)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as stat_file:
            return stat_file.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _running(manager, sid, count, deadline=5.0):
    """The session's tracked processes once ``count`` run, or what is tracked when the deadline passes."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        with manager._lock:
            procs = manager._running_procs.get(sid)
            tracked = list(procs) if isinstance(procs, (set, list, tuple)) else ([procs] if procs else [])
        if len(tracked) >= count:
            return tracked
        time.sleep(0.02)
    return tracked


def _command(manager, sid, answers):
    """Run a real ``sleep`` in the session through the tracked spawn, in a thread; its answer lands in ``answers``."""
    workspace = manager.get_workspace_path(sid)

    def run():
        try:
            answers.append(manager._spawn_tracked(
                ["sleep", _SLEEP], timeout=60, session_id=sid, cwd=workspace, env={"PATH": "/usr/bin:/bin"}))
        except Exception as exc:  # noqa: BLE001 - the answer is what the contract reads
            answers.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _reap(procs):
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_kr1_a_destroy_stops_what_runs(tmp_path, monkeypatch):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    tracked = []
    at_removal = []
    try:
        manager = ce_t._manager(sm, tmp_path)
        real_remove = sm._remove_tree

        def watched_remove(path):
            # What still runs when a removal starts
            at_removal.append([proc.pid for proc in tracked if _alive(proc.pid)])
            return real_remove(path)

        monkeypatch.setattr(sm, "_remove_tree", watched_remove)
        sid = manager.create_sandbox("kr1").session_id
        workspace = manager.get_workspace_path(sid)
        answers = []
        thread = _command(manager, sid, answers)
        tracked += _running(manager, sid, 1)
        started = time.monotonic()
        destroyed = manager.destroy_sandbox(sid)
        thread.join(timeout=8)
        answered_in = time.monotonic() - started
        alive = [proc.pid for proc in tracked if _alive(proc.pid)]
        gone = not os.path.lexists(workspace)
        # A second destroy racing the first, inside the first's kill: it kills too before it removes
        sid = manager.create_sandbox("kr1-raced").session_id
        raced_answers = []
        raced_thread = _command(manager, sid, raced_answers)
        raced = _running(manager, sid, 1)
        tracked += raced
        real_kill = getattr(sm, "_kill_group", None)
        nested, racing = [], []

        def racing_kill(proc):
            if not racing:
                racing.append(proc)
                nested.append(manager.destroy_sandbox(sid))
            return real_kill(proc)

        monkeypatch.setattr(sm, "_kill_group", racing_kill, raising=False)
        raced_destroyed = manager.destroy_sandbox(sid)
        raced_thread.join(timeout=8)
        # A destroy racing a stop, inside the stop's kill: it kills too before it removes
        sid = manager.create_sandbox("kr1-stopped").session_id
        stopped_answers = []
        stopped_thread = _command(manager, sid, stopped_answers)
        tracked += _running(manager, sid, 1)
        racing.clear()
        nested.clear()
        stopped = manager.stop_command(sid)
        stopped_thread.join(timeout=8)
        monkeypatch.undo()
    finally:
        monkeypatch.undo()
        _reap(tracked)
        restore()
    assert len(tracked) == 3, f"a command was never tracked: {tracked}"
    assert destroyed is True and gone, (destroyed, gone)
    assert alive == [], f"a destroy left the session's command running: {alive}"
    assert not thread.is_alive() and answered_in < 8, f"the command's caller waited {answered_in:.1f} s"
    assert answers and getattr(answers[0], "returncode", None) == -9, f"the command was not killed: {answers}"
    assert (nested, raced_destroyed, stopped) == ([True], True, True), (nested, raced_destroyed, stopped)
    assert at_removal and all(still == [] for still in at_removal), (
        f"a removal started while the session's command ran: {at_removal}")


def test_kr2_every_command_is_tracked(tmp_path):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    tracked = []
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid = manager.create_sandbox("kr2").session_id
        answers = []
        threads = [_command(manager, sid, answers), _command(manager, sid, answers)]
        tracked = _running(manager, sid, 2)
        running_with_two = manager.is_running(sid)
        stopped = manager.stop_command(sid)
        for thread in threads:
            thread.join(timeout=8)
        alive = [proc.pid for proc in tracked if _alive(proc.pid)]
        running_after = manager.is_running(sid)
        # A short command ends beside a long one: the long one stays tracked
        long_answers = []
        long_thread = _command(manager, sid, long_answers)
        long_run = _running(manager, sid, 1)
        tracked += long_run
        manager._spawn_tracked(["true"], timeout=30, session_id=sid,
                               cwd=manager.get_workspace_path(sid), env={"PATH": "/usr/bin:/bin"})
        still_running = manager.is_running(sid)
        still_tracked = _running(manager, sid, 1, deadline=0.1)
        manager.destroy_sandbox(sid)
        long_thread.join(timeout=8)
    finally:
        _reap(tracked)
        restore()
    assert len(tracked) == 3, f"two commands of one session, {len(tracked) - 1} tracked"
    assert running_with_two and stopped is True, (running_with_two, stopped)
    assert alive == [], f"stop_command left a command of the session running: {alive}"
    assert not any(thread.is_alive() for thread in threads), "a command's caller still waits"
    assert running_after is False, "the session shows running once its commands are stopped"
    assert still_running and still_tracked == long_run, (
        f"a short command's end untracked the long one beside it: {still_running} {still_tracked}")
    assert not long_thread.is_alive(), "the long command outlived its session's destroy"


def test_kr3_nothing_starts_for_a_destroyed_session(tmp_path, monkeypatch):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    started = []
    real_popen = subprocess.Popen

    def recording(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        started.append(proc)
        return proc

    removing, release = threading.Event(), threading.Event()
    destroyers = []
    try:
        manager = ce_t._manager(sm, tmp_path, strict_mode=False)
        sid = manager.create_sandbox("kr3").session_id
        real_remove = sm._remove_tree

        def held_remove(path):
            # The destroy has ended the session; its workspace is still there
            removing.set()
            release.wait(timeout=8)
            return real_remove(path)

        def destroyed_meanwhile(command):
            # The caller has checked the session; a destroy ends it before the spawn
            destroyer = threading.Thread(target=manager.destroy_sandbox, args=(sid,), daemon=True)
            destroyer.start()
            destroyers.append(destroyer)
            removing.wait(timeout=5)
            return ["sleep", _SLEEP]

        manager._build_tempdir_command = destroyed_meanwhile
        monkeypatch.setattr(sm, "_remove_tree", held_remove)
        monkeypatch.setattr(subprocess, "Popen", recording)
        began = time.monotonic()
        result = manager.execute_command(sid, "echo kr3")
        took = time.monotonic() - began
        release.set()
        for destroyer in destroyers:
            destroyer.join(timeout=8)
        monkeypatch.undo()
        alive = [proc.pid for proc in started if _alive(proc.pid)]
        tracked = dict(manager._running_procs)
    finally:
        release.set()
        monkeypatch.undo()
        _reap(started)
        restore()
    assert len(started) == 1, f"the command was never spawned: {started} ({result.return_code}, {result.stderr!r})"
    assert alive == [], f"a command started for a destroyed session runs on: {alive}"
    assert took < 8, f"the caller waited {took:.1f} s for a command of a destroyed session"
    assert result.return_code == -1 and "was destroyed" in result.stderr, (result.return_code, result.stderr)
    assert tracked == {}, f"a destroyed session's command is tracked: {tracked}"


def test_kr4_what_the_running_registry_keeps(tmp_path):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid = manager.create_sandbox("kr4").session_id
        idle_stop = manager.stop_command(sid)
        done = manager._spawn_tracked(["true"], timeout=30, session_id=sid,
                                      cwd=manager.get_workspace_path(sid), env={"PATH": "/usr/bin:/bin"})
        running_after = manager.is_running(sid)
        tracked = dict(manager._running_procs)
        manager.destroy_sandbox(sid)
    finally:
        restore()
    assert idle_stop is False, idle_stop
    assert done.returncode == 0, done
    assert running_after is False and tracked == {}, (running_after, tracked)


def test_kr5_the_idle_sweep_never_kills_a_command(tmp_path):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    tracked = []
    try:
        manager = ce_t._manager(sm, tmp_path, idle_ttl_seconds=60, strict_mode=False)
        # Idle past the TTL by its last activity, a command started since the sweep's snapshot
        sid = manager.create_sandbox("kr5").session_id
        manager.get_session(sid).last_activity = time.time() - 3600
        answers = []
        real_destroy = manager.destroy_sandbox
        threads = []

        def started_since_the_snapshot(session_id, **kwargs):
            if session_id == sid and not threads:
                threads.append(_command(manager, sid, answers))
                tracked.extend(_running(manager, sid, 1))
            return real_destroy(session_id, **kwargs)

        manager.destroy_sandbox = started_since_the_snapshot
        manager.list_sessions()
        del manager.destroy_sandbox
        kept = manager.get_session(sid) is not None
        alive = [proc.pid for proc in tracked if _alive(proc.pid)]
        # A command starting counts as activity
        seen = []
        began = time.time()
        if kept:
            manager._build_tempdir_command = lambda command: seen.append(manager.get_session(sid).last_activity) or ["true"]
            manager.get_session(sid).last_activity = time.time() - 3600
            manager.execute_command(sid, "echo kr5")
            manager.destroy_sandbox(sid)
        for thread in threads:
            thread.join(timeout=8)
        # Every clause of the check made again: activity since the snapshot, a binding since, a destroy begun
        clauses = {}
        for clause in ("activity", "binding", "ending"):
            changed = manager.create_sandbox(f"kr5-{clause}").session_id
            manager.get_session(changed).last_activity = time.time() - 3600

            def changed_since(session_id, _clause=clause, _changed=changed, **kwargs):
                session = manager.get_session(_changed)
                if session_id == _changed and session is not None:
                    if _clause == "activity":
                        session.last_activity = time.time()
                    elif _clause == "binding":
                        session.bound_conversation_id = "kr5-conversation"
                    else:
                        session.active = False
                return real_destroy(session_id, **kwargs)

            manager.destroy_sandbox = changed_since
            reaped = manager._sweep_idle_sessions()
            del manager.destroy_sandbox
            clauses[clause] = (reaped, manager.get_session(changed) is not None)
            if manager.get_session(changed) is not None:
                manager.get_session(changed).active = True
                manager.destroy_sandbox(changed)
        # A command refused before it runs is no activity
        strict = ce_t._manager(sm, tmp_path, idle_ttl_seconds=60, strict_mode=True)
        refused_sid = strict.create_sandbox("kr5-refused").session_id
        long_ago = time.time() - 3600
        strict.get_session(refused_sid).last_activity = long_ago
        refused = strict.execute_command(refused_sid, "echo kr5")
        refused_activity = strict.get_session(refused_sid).last_activity
        strict.destroy_sandbox(refused_sid)
    finally:
        _reap(tracked)
        restore()
    assert len(tracked) == 1, f"the command was never tracked: {tracked}"
    assert kept and alive == [p.pid for p in tracked], (
        f"the idle sweep took a session whose command ran: kept={kept} alive={alive}")
    assert seen and seen[0] >= began, f"a command starting did not count as activity: {seen} {began}"
    assert clauses == {clause: (0, True) for clause in ("activity", "binding", "ending")}, (
        f"the idle sweep took a session changed since its snapshot: {clauses}")
    assert refused.blocked and refused_activity == long_ago, (
        f"a command refused before it ran counted as activity: {refused.blocked} {refused_activity} {long_ago}")


def test_kr6_a_freed_pid_is_never_signalled(tmp_path, monkeypatch):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    signalled = []
    real_killpg, real_kill = os.killpg, os.kill
    leftovers = []
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid = manager.create_sandbox("kr6").session_id
        workspace = manager.get_workspace_path(sid)
        # A command ended and reaped, its entry not yet dropped
        done = subprocess.Popen(["true"], cwd=workspace, start_new_session=True)
        done.wait(timeout=10)
        with manager._lock:
            entry = manager._running_procs.get(sid)
            if isinstance(entry, set):
                entry.add(done)
            else:
                manager._running_procs[sid] = {done}
        monkeypatch.setattr(os, "killpg", lambda pid, sig: signalled.append(("killpg", pid)) or real_killpg(pid, sig))
        monkeypatch.setattr(os, "kill", lambda pid, sig: signalled.append(("kill", pid)) or real_kill(pid, sig))
        manager.destroy_sandbox(sid)
        reaped_signalled = list(signalled)
        signalled.clear()
        # A command whose time runs out after a destroy killed and reaped it
        sid = manager.create_sandbox("kr6-late").session_id
        killed_by_destroy = []
        real_communicate = subprocess.Popen.communicate

        def late(self, *args, **kwargs):
            if kwargs.get("timeout") == 30 and not killed_by_destroy:
                killed_by_destroy.append(self.pid)
                manager.destroy_sandbox(sid)
                raise subprocess.TimeoutExpired(self.args, 30)
            return real_communicate(self, *args, **kwargs)

        monkeypatch.setattr(subprocess.Popen, "communicate", late)
        try:
            manager._spawn_tracked(["sleep", _SLEEP], timeout=30, session_id=sid,
                                   cwd=manager.get_workspace_path(sid), env={"PATH": "/usr/bin:/bin"})
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        monkeypatch.undo()
        late_signals = [signal_ for signal_ in signalled if signal_[1] in killed_by_destroy]
    finally:
        monkeypatch.undo()
        _reap(leftovers)
        restore()
    assert [pid for _how, pid in reaped_signalled if pid == done.pid] == [], (
        f"a destroy signalled a command already reaped: {reaped_signalled}")
    assert timed_out and len(killed_by_destroy) == 1, (timed_out, killed_by_destroy)
    assert [how for how, _pid in late_signals] == ["killpg"], (
        f"a command killed by a destroy was signalled again when its time ran out: {late_signals}")
