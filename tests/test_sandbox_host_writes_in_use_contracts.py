"""Contracts for the host's work on a workspace: the idle sweep never takes one while the host copies into or out of it.

The idle sweep, which every list of the panel runs, destroys an unbound session idle past the TTL that runs no
command. The host's own work on a workspace -- an upload, a copy of files or of a folder, a clone of a shared folder,
the apply, which reads the workspace and writes the host's files, and the copy-out of approved files -- is no
command: it counted as activity only when it ended (the copy of files and the copy-out never), and nothing marked it
as running. A sweep that came while one ran removed the workspace under it: an upload's files landed in a removed
tree, a clone was cut, an apply or a copy-out wrote part of what was asked. Now each runs with its session in use
(``SandboxManager.in_use``, counted on the session), which the sweep's snapshot skips and its check made again
declines; a copy in that an explicit destroy overtook says so.

  * HU1 -- NO SWEEP TAKES A WORKSPACE THE HOST WORKS IN: an upload, a copy of files, a copy of a folder, a clone, an
    apply, a copy-out of two files and of one, each from a session idle past the TTL; the panel's list comes at the
    operation's first write (before the copy-out's first file), the session's last activity past the TTL again: the
    session is kept and every file lands (the apply's and the copy-out's on the host); once a copy in or an apply
    ends, the next list keeps the session (its end is activity, the copy of files' too) (round 1: the copy-out).
  * HU2 -- THE SWEEP'S CHECK MADE AGAIN: the sweep's snapshot taken, an upload begins before its destroy: the destroy
    declines and the upload completes; two uploads at once: the first ending leaves the session in use, the second
    frees it, and the next sweep takes the session once idle past the TTL.
  * HU3 -- A REFUSAL LEAVES NOTHING IN USE: an unknown session (to a copy in, an apply or a copy-out), an inactive one
    (answered as before), an upload over a cap, an invalid destination, a clone outside the shares, a copy of a
    folder that fails, an apply on a stale review, a workspace folder gone, a copy-out of a file not approved: none
    counts activity, and the session idle past the TTL is swept after each.
  * HU4 -- A DESTROY DURING A COPY: an explicit destroy while an upload writes ends the session at once (it is not held
    off), and a session made again under its id is not held by the upload (round 1); each copy in a destroy overtakes
    answers that the session was destroyed -- whether it returned, never with files that went with the workspace, or
    failed for the removal (round 1: a copy of a folder, an upload into a folder) -- and the workspace does not stay.
    An apply or a copy-out a destroy overtakes keeps its answer: what it wrote is on the host (round 2: the copy-out).

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_hu1_no_sweep_takes_a_workspace_the_host_writes_into": 5.0,
    "test_hu2_the_sweeps_check_made_again_declines_a_copy": 5.0,
    "test_hu3_a_refusal_leaves_nothing_in_use": 5.0,
    "test_hu4_a_destroy_during_a_copy_answers_so": 5.0,
}


def _window(tmp_path):
    """The real manager on the tempdir backend, sharing ``tmp_path/share``, and the real workspace module (the
    review and the apply) over it: (sandbox_manager, sandbox_workspace, manager, closer). No route: the routes'
    import reads their export root."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *a, **k: sqlite3.connect(path)
    (tmp_path / "share").mkdir()
    loaded, restore = isolate(
        targets={
            "opti_oignon.process_secrets": source("process_secrets.py"),
            "opti_oignon.sandbox_manager": copy,
            "opti_oignon.sandbox_workspace": source("sandbox_workspace.py"),
        },
        seeded={
            "opti_oignon.db_utils": db_utils,
            "opti_oignon.config": ce_t._config(tmp_path, ce_t._ON),
        },
    )
    try:
        sm = loaded["opti_oignon.sandbox_manager"]
        if sm.sandbox_manager is not None:
            raise AssertionError("the import-time singleton must stay off in this window")
        manager = ce_t._manager(sm, tmp_path, host_share_roots=[str(tmp_path / "share")])
    except BaseException:
        restore()
        raise
    return sm, loaded["opti_oignon.sandbox_workspace"], manager, restore


def _workspace(manager, label):
    sid = manager.create_sandbox(None, label=label).session_id
    workspace = Path(manager.get_workspace_path(sid))
    return sid, workspace, os.path.realpath(workspace)


def _refusal(call, *args, **kwargs):
    """(the exception's name, its text) of what ``call`` raised, or ("answered", its answer)."""
    try:
        return "answered", call(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- the contract reads it
        return type(exc).__name__, str(exc)


class _Stream:
    """An upload's stream: ``data`` in one read; ``before`` runs at the first read."""

    def __init__(self, data, before=None):
        self.data, self.before, self.done = data, before, False

    def read(self, size=-1):
        if self.before is not None:
            before, self.before = self.before, None
            before()
        if self.done:
            return b""
        self.done = True
        return self.data


def _idle(manager, sid):
    """The session idle past the TTL by its last activity; the value set."""
    long_ago = time.time() - 2 * manager._config.idle_ttl_seconds
    manager.get_session(sid).last_activity = long_ago
    return long_ago


def _sweep_at_first(manager, sid, kept):
    """At its first call: the session idle past the TTL again, then the panel's list (the sweep); ``kept`` says
    whether the session stayed."""

    def sweep():
        if kept:
            return
        _idle(manager, sid)
        manager.list_sessions()
        kept.append(manager.get_session(sid) is not None)

    return sweep


def _racing(module, name, before, after=None):
    """``module.name`` running ``before`` (and ``after``) around its real self; the real one."""
    real = getattr(module, name)

    def racing(*args, **kwargs):
        before()
        done = real(*args, **kwargs)
        if after is not None:
            after()
        return done

    setattr(module, name, racing)
    return real


def _in_thread(call, *args, **kwargs):
    """``call`` run in a thread; (thread, outcome): outcome holds ("answered", value) or ("raised", exception)."""
    outcome = []

    def run():
        try:
            outcome.append(("answered", call(*args, **kwargs)))
        except Exception as exc:  # noqa: BLE001 -- the contract reads it
            outcome.append(("raised", exc))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _blocked_stream(data):
    """An upload's stream that, at its first read, says it is reading and waits to be let go: (stream, reading, go)."""
    reading, go = threading.Event(), threading.Event()

    def wait():
        reading.set()
        go.wait(timeout=10)

    return _Stream(data, wait), reading, go


def _host_sources(tmp_path):
    """A shared folder to clone, host files and a host folder to copy in: (share, host)."""
    share = tmp_path / "share"
    (share / "proj" / "sub").mkdir(parents=True)
    (share / "proj" / "c.txt").write_bytes(b"clone-c")
    (share / "proj" / "sub" / "d.txt").write_bytes(b"clone-d")
    (share / "out").mkdir()
    host = tmp_path / "host"
    (host / "tree" / "sub").mkdir(parents=True)
    (host / "one.txt").write_bytes(b"inject-one")
    (host / "tree" / "e.txt").write_bytes(b"tree-e")
    (host / "tree" / "sub" / "f.txt").write_bytes(b"tree-f")
    return share, host


def test_hu1_no_sweep_takes_a_workspace_the_host_writes_into(tmp_path):
    sm, ws, manager, restore = _window(tmp_path)
    share, host = _host_sources(tmp_path)
    exported = tmp_path / "exported"
    seen = {}
    try:
        # name: (the seam's owner, the seam, the operation, whether its end counts as activity)
        operations = {
            "upload": (None, None, lambda sid, sweep: manager.upload_files(
                sid, [("up.txt", _Stream(b"uploaded", sweep), 8)]), True),
            "inject_files": (sm, "_replace_beneath", lambda sid, sweep: manager.inject_files(
                sid, [str(host / "one.txt")]), True),
            "inject_directory": (sm, "_copy_tree_beneath", lambda sid, sweep: manager.inject_directory(
                sid, str(host / "tree")), True),
            "clone": (sm, "_new_file_beneath", lambda sid, sweep: manager.clone_directory(
                sid, str(share / "proj")), True),
            "apply": (ws, "_write_file_into_target", lambda sid, sweep: ws.apply_workspace_changes(
                sid, ws.generate_workspace_diff(sid, manager=manager).diff_hash, manager=manager,
                target_dir=str(share / "out")), True),
            # The panel's copy out: the sweep before its first file is copied; one file copied out, the sweep at
            # its write
            "copy_out": (manager, "copy_out_file", lambda sid, sweep: manager.copy_out_batch(
                sid, ["k.txt", "l.txt"], str(exported / "batch")), False),
            "copy_out_file": (sm, "_create_free_export", lambda sid, sweep: manager.copy_out_file(
                sid, "k.txt", str(exported / "one")), False),
        }
        for name, (owner, seam, operate, _counts) in operations.items():
            sid, workspace, _real = _workspace(manager, f"hu1-{name}")
            if name == "apply":
                (workspace / "a.txt").write_bytes(b"applied")
                manager.approve_files(sid, ["a.txt"])
            if name.startswith("copy_out"):
                (workspace / "k.txt").write_bytes(b"out-k")
                (workspace / "l.txt").write_bytes(b"out-l")
                manager.approve_files(sid, ["k.txt", "l.txt"])
            kept = []
            sweep = _sweep_at_first(manager, sid, kept)
            real = _racing(owner, seam, sweep) if owner is not None else None
            _idle(manager, sid)
            try:
                answer = _refusal(operate, sid, sweep)
            finally:
                if owner is manager:
                    del manager.copy_out_file
                elif owner is not None:
                    setattr(owner, seam, real)
            manager.list_sessions()
            after = manager.get_session(sid) is not None
            files = sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file())
            seen[name] = (answer, kept, after, files)
            if manager.get_session(sid) is not None:
                manager.destroy_sandbox(sid)
        out = (share / "out" / "a.txt").read_bytes() if (share / "out" / "a.txt").exists() else None
        copied_out = sorted(str(p.relative_to(exported)) for p in exported.rglob("*") if p.is_file())
    finally:
        restore()
    for name, (answer, kept, _after, _files) in seen.items():
        assert answer[0] == "answered", f"{name}: {seen[name]}"
        assert kept == [True], f"{name}: the sweep took a workspace the host was working in: {seen[name]}"
    assert seen["upload"][3] == ["up.txt"], seen["upload"]
    assert seen["inject_files"][3] == ["one.txt"], seen["inject_files"]
    assert seen["inject_directory"][3] == ["tree/e.txt", "tree/sub/f.txt"], seen["inject_directory"]
    assert seen["clone"][3] == ["proj/c.txt", "proj/sub/d.txt"], seen["clone"]
    assert out == b"applied", f"the apply's file never reached the host: {out!r}"
    assert [entry["src_path"] for entry in seen["copy_out"][0][1]] == ["k.txt", "l.txt"], seen["copy_out"]
    assert copied_out == ["batch/k.txt", "batch/l.txt", "one/k.txt"], copied_out
    for name, (_answer, _kept, after, _files) in seen.items():
        if operations[name][3]:
            assert after, f"{name}: its end counted no activity, the next list took the session: {seen[name]}"


def test_hu2_the_sweeps_check_made_again_declines_a_copy(tmp_path):
    _sm, _ws, manager, restore = _window(tmp_path)
    try:
        # The sweep's snapshot taken, an upload begun before its destroy
        sid, workspace, _real = _workspace(manager, "hu2")
        stream, reading, go = _blocked_stream(b"late")
        threads = []
        real_destroy = manager.destroy_sandbox

        def begun_since_the_snapshot(session_id, **kwargs):
            if session_id == sid and not threads:
                threads.append(_in_thread(manager.upload_files, sid, [("late.txt", stream, 4)]))
                reading.wait(timeout=5)
            return real_destroy(session_id, **kwargs)

        _idle(manager, sid)
        manager.destroy_sandbox = begun_since_the_snapshot
        try:
            reaped = manager._sweep_idle_sessions()
        finally:
            del manager.destroy_sandbox
        kept = manager.get_session(sid) is not None
        go.set()
        for thread, _outcome in threads:
            thread.join(timeout=8)
        late = threads[0][1] if threads else []
        landed = (workspace / "late.txt").read_bytes() if (workspace / "late.txt").exists() else None
        if manager.get_session(sid) is not None:
            manager.destroy_sandbox(sid)
        # Two at once: the session in use until both end
        sid, workspace, _real = _workspace(manager, "hu2-two")
        first, first_reading, first_go = _blocked_stream(b"one")
        second, second_reading, second_go = _blocked_stream(b"two")
        one = _in_thread(manager.upload_files, sid, [("one.txt", first, 3)])
        two = _in_thread(manager.upload_files, sid, [("two.txt", second, 3)])
        both = first_reading.wait(timeout=5) and second_reading.wait(timeout=5)
        first_go.set()
        one[0].join(timeout=8)
        _idle(manager, sid)
        manager.list_sessions()
        while_second = manager.get_session(sid) is not None
        second_go.set()
        two[0].join(timeout=8)
        if while_second:
            _idle(manager, sid)
            manager.list_sessions()
        after_both = manager.get_session(sid) is not None
        if after_both:
            manager.destroy_sandbox(sid)
    finally:
        restore()
    assert (reaped, kept) == (0, True), f"the sweep's destroy took a session a copy had begun in: {reaped} {kept}"
    assert late and late[0][0] == "answered", late
    assert landed == b"late", f"the upload begun before the sweep's destroy did not land: {landed!r}"
    assert both and one[1][0][0] == "answered" and two[1][0][0] == "answered", (both, one[1], two[1])
    assert while_second, "the first copy's end freed a session the second still wrote into"
    assert not after_both, "a session no copy uses any more, idle past the TTL, was never swept"


def test_hu3_a_refusal_leaves_nothing_in_use(tmp_path):
    _sm, ws, manager, restore = _window(tmp_path)
    share = tmp_path / "share"
    (share / "out").mkdir()
    outside = tmp_path / "outside"
    (outside / "src").mkdir(parents=True)
    (outside / "src" / "x.txt").write_bytes(b"x")
    failing = tmp_path / "failing"
    failing.mkdir()
    (failing / "ok.txt").write_bytes(b"ok")
    os.mkfifo(failing / "pipe")
    seen = {}
    try:
        unknown = _refusal(manager.upload_files, "hu3-nobody", [("u.txt", _Stream(b"u"), 1)])
        unknown_apply = _refusal(
            ws.apply_workspace_changes, "hu3-nobody", "0" * 64, manager=manager, target_dir=str(share / "out"))
        unknown_out = _refusal(manager.copy_out_batch, "hu3-nobody", ["a.txt"], str(tmp_path / "exported"))
        cap = manager._config.upload_max_file_bytes
        refusals = {
            "inactive": lambda sid: manager.upload_files(sid, [("i.txt", _Stream(b"i"), 1)]),
            "over_a_cap": lambda sid: manager.upload_files(sid, [("big.txt", _Stream(b"b"), cap + 1)]),
            "invalid_destination": lambda sid: manager.upload_files(sid, [("d.txt", _Stream(b"d"), 1)], "../away"),
            "clone_outside": lambda sid: manager.clone_directory(sid, str(outside / "src")),
            "copy_fails": lambda sid: manager.inject_directory(sid, str(failing)),
            "stale_review": lambda sid: ws.apply_workspace_changes(
                sid, "0" * 64, manager=manager, target_dir=str(share / "out")),
            "workspace_gone": lambda sid: manager.inject_files(sid, [str(failing / "ok.txt")]),
            "copy_out_unapproved": lambda sid: manager.copy_out_file(sid, "a.txt", str(tmp_path / "exported")),
        }
        for name, refuse in refusals.items():
            sid, workspace, _real = _workspace(manager, f"hu3-{name}")
            if name == "inactive":
                manager.get_session(sid).active = False
            if name == "workspace_gone":
                os.rename(workspace, tmp_path / "moved-away")
            if name == "copy_out_unapproved":
                (workspace / "a.txt").write_bytes(b"a")
            long_ago = _idle(manager, sid)
            answer = _refusal(refuse, sid)
            session = manager.get_session(sid)
            activity = session.last_activity if session is not None else None
            if name == "inactive":
                session.active = True
            if name == "workspace_gone":
                os.rename(tmp_path / "moved-away", workspace)
            manager.list_sessions()
            seen[name] = (answer, activity == long_ago, manager.get_session(sid) is None)
            if manager.get_session(sid) is not None:
                manager.destroy_sandbox(sid)
    finally:
        restore()
    assert unknown == ("ValueError", "Session not found: hu3-nobody"), unknown
    assert unknown_apply == ("ValueError", "Session not found: hu3-nobody"), unknown_apply
    assert unknown_out == ("ValueError", "Session not found: hu3-nobody"), unknown_out
    assert seen["inactive"][0][0] == "ValueError" and seen["inactive"][0][1].startswith("Session is inactive"), seen
    assert seen["over_a_cap"][0][0] == "WorkspaceQuotaExceeded", seen["over_a_cap"]
    assert seen["invalid_destination"][0][0] == "ValueError", seen["invalid_destination"]
    assert seen["clone_outside"][0][0] in ("PermissionError", "ValueError"), seen["clone_outside"]
    assert seen["copy_fails"][0][0] == "Error", seen["copy_fails"]
    assert seen["stale_review"][0][0] == "WorkspaceReviewDrift", seen["stale_review"]
    assert seen["workspace_gone"][0][0] == "ValueError" and "missing" in seen["workspace_gone"][0][1], seen
    assert seen["copy_out_unapproved"][0][0] == "PermissionError", seen["copy_out_unapproved"]
    for name, (answer, no_activity, swept) in seen.items():
        assert no_activity, f"{name}: a refusal counted as activity: {answer}"
        assert swept, f"{name}: a refusal left its session in use, the sweep never took it: {answer}"


def _destroy_at_first(manager, sid, ended):
    """At its first call: an explicit destroy of the session (``ended`` holds its answer); never again, even from
    within the destroy."""

    def destroy():
        if not ended:
            ended.append(None)
            ended[0] = manager.destroy_sandbox(sid)

    return destroy


def test_hu4_a_destroy_during_a_copy_answers_so(tmp_path):
    sm, ws, manager, restore = _window(tmp_path)
    share, host = _host_sources(tmp_path)
    copies = {}
    try:
        # An upload: the destroy comes while it reads its stream; a session made again under the same id
        sid, workspace, _real = _workspace(manager, "hu4-upload")
        stream, reading, go = _blocked_stream(b"gone")
        thread, upload = _in_thread(manager.upload_files, sid, [("gone.txt", stream, 4)])
        reading.wait(timeout=5)
        began = time.monotonic()
        destroyed = manager.destroy_sandbox(sid)
        took = time.monotonic() - began
        manager.create_sandbox(sid, label="hu4-again")
        again = Path(manager.get_workspace_path(sid))
        _idle(manager, sid)
        manager.list_sessions()
        again_swept = manager.get_session(sid) is None
        go.set()
        thread.join(timeout=8)
        manager.list_sessions()
        upload_left = workspace.exists()
        # Every copy in, the destroy at its first write: the copy fails for it, or returns
        cases = {
            "upload_into_a_folder": (sm, "_folder_beneath", lambda sid: manager.upload_files(
                sid, [("u.txt", _Stream(b"u"), 1)], "sub")),
            "inject_files": (sm, "_replace_beneath", lambda sid: manager.inject_files(sid, [str(host / "one.txt")])),
            "inject_directory": (sm, "_copy_tree_beneath", lambda sid: manager.inject_directory(
                sid, str(host / "tree"))),
            "clone": (sm, "_new_file_beneath", lambda sid: manager.clone_directory(sid, str(share / "proj"))),
        }
        for name, (owner, seam, copy) in cases.items():
            sid, workspace, _real = _workspace(manager, f"hu4-{name}")
            ended = []
            real = _racing(owner, seam, _destroy_at_first(manager, sid, ended))
            try:
                answer = _refusal(copy, sid)
            finally:
                setattr(owner, seam, real)
            manager.list_sessions()
            copies[name] = (answer, ended, workspace.exists())
        # An apply: the destroy comes after its first write
        sid, workspace, _real = _workspace(manager, "hu4-apply")
        (workspace / "a.txt").write_bytes(b"applied-a")
        (workspace / "b.txt").write_bytes(b"applied-b")
        manager.approve_files(sid, ["a.txt", "b.txt"])
        diff_hash = ws.generate_workspace_diff(sid, manager=manager).diff_hash
        ended = []
        real = _racing(ws, "_write_file_into_target", lambda: None, _destroy_at_first(manager, sid, ended))
        try:
            applied = _refusal(
                ws.apply_workspace_changes, sid, diff_hash, manager=manager, target_dir=str(share / "out"))
        finally:
            ws._write_file_into_target = real
        on_host = sorted(p.name for p in (share / "out").iterdir())
        # A copy-out: the destroy comes after its first file
        sid, workspace, _real = _workspace(manager, "hu4-copy-out")
        (workspace / "k.txt").write_bytes(b"out-k")
        (workspace / "l.txt").write_bytes(b"out-l")
        manager.approve_files(sid, ["k.txt", "l.txt"])
        out_ended = []
        _racing(manager, "copy_out_file", lambda: None, _destroy_at_first(manager, sid, out_ended))
        try:
            copied_out = _refusal(manager.copy_out_batch, sid, ["k.txt", "l.txt"], str(tmp_path / "exported"))
        finally:
            del manager.copy_out_file
        exported = sorted(p.name for p in (tmp_path / "exported").iterdir())
    finally:
        restore()
    assert destroyed and took < 2.0, f"an explicit destroy was held off by a copy: {destroyed} after {took:.1f} s"
    assert again_swept and not again.exists(), (
        "a session made again under the id of one an upload was copying into was held by that upload")
    assert upload and upload[0][0] == "raised" and "was destroyed" in str(upload[0][1]), (
        f"an upload a destroy overtook answered with files that went with the workspace: {upload}")
    assert not upload_left, "the workspace an upload wrote into after its destroy stayed"
    for name, (answer, ended, left) in copies.items():
        assert ended == [True], f"{name}: the destroy never came: {copies[name]}"
        assert answer[0] == "ValueError" and "was destroyed" in answer[1], (
            f"{name}: a copy a destroy overtook did not say so: {answer}")
        assert not left, f"{name}: the workspace a copy wrote into after its destroy stayed"
    assert ended == [True] and applied[0] == "answered", (ended, applied)
    assert [entry["path"] for entry in applied[1]["applied"]] == [on_host[0]] and len(on_host) == 1, (
        f"an apply a destroy overtook did not say what it wrote: {applied[1]} {on_host}")
    assert out_ended == [True] and copied_out[0] == "answered", (out_ended, copied_out)
    assert [entry["src_path"] for entry in copied_out[1]] == ["k.txt"] and exported == ["k.txt"], (
        f"a copy-out a destroy overtook did not say what it copied: {copied_out} {exported}")
