"""Contracts for the agent's file tools: the sweep never takes a workspace one of them works in, and each call counts.

The agent's tool session is unbound: the idle sweep, which every list of the panel runs, destroys it once idle past the
TTL with no command running and no host operation in use. Its file tools -- ``create_file`` and ``str_replace``, which
write into the workspace, ``view``, and the reads ``grep``, ``glob`` and ``ls`` -- ran outside ``in_use`` and counted
no activity: an agent that wrote and read files without running a command between them was idle to the sweep, its
session destroyed between two calls (the next answered "Session not found", the work gone) or under a write (the tool
answered "File created" when its write landed before the removal reached it). Now each runs with its session in use
(``file_tools._held``) and counts as activity when it ends, its answer an error or not: the agent calling a tool is the
agent working. A destroy that overtakes a write is the write's answer.

  * FH1 -- NO SWEEP TAKES A WORKSPACE AN AGENT TOOL WORKS IN: ``create_file``, ``str_replace``, ``view``, ``grep``,
    ``glob`` and ``ls``, each through the agent's tool session on a session idle past the TTL; the panel's list comes
    at the tool's first write or read, the session's last activity past the TTL again: the session is kept and the
    tool answers what it did; once it ends, the next list keeps the session (its end counted as activity).
  * FH2 -- A DESTROY DURING A WRITE: an explicit destroy is not held off; ``create_file`` and ``str_replace`` it
    overtakes -- before their write, which then fails for the removal or lands in an unlinked file, or after it, or
    between their path's check and their write, where ``create_file`` raised (round 1) -- answer that the session was
    destroyed, never that the file was written; the workspace does not stay and nothing stays in use. A ``view`` it
    overtakes once its file is open keeps its answer. A session made again under the id meanwhile is not touched by
    the old call's end, nor written (round 1).
  * FH3 -- WHAT A CALL COUNTS: a session unknown to the manager ("Session not found or inactive"), or one already
    ended under the agent's tool session (its ``RuntimeError``), answers as before and nothing is held, not even a
    session made again under its id (round 2: the read tools held it before their check); a refused call (a path out
    of the workspace, content too large, no match, a file or a folder not found, an empty pattern) counts activity:
    the next list keeps the session idle past the TTL before it; nothing stays in use after any call, a tool that
    raised included, and the sweep then takes the session.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import sqlite3
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_fh1_no_sweep_takes_a_workspace_an_agent_tool_works_in": 5.0,
    "test_fh2_a_destroy_during_a_write_answers_so": 5.0,
    "test_fh3_what_a_call_counts": 5.0,
}


def _window(tmp_path):
    """The real manager on the tempdir backend, the real file tools and the agent's real tool session over it:
    (sandbox_manager, file_tools, sandbox_tools, manager, closer)."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *a, **k: sqlite3.connect(path)
    loaded, restore = isolate(
        targets={
            "opti_oignon.process_secrets": source("process_secrets.py"),
            "opti_oignon.sandbox_manager": copy,
            "opti_oignon.file_tools": source("file_tools.py"),
            "opti_oignon.sandbox_tools": source("sandbox_tools.py"),
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
        manager = ce_t._manager(sm, tmp_path)
    except BaseException:
        restore()
        raise
    return sm, loaded["opti_oignon.file_tools"], loaded["opti_oignon.sandbox_tools"], manager, restore


def _session(st, manager, label):
    """A session holding ``a.txt`` ("old text") and the agent's tool session on it: (sid, workspace, tools)."""
    sid = manager.create_sandbox(None, label=label).session_id
    workspace = Path(manager.get_workspace_path(sid))
    (workspace / "a.txt").write_bytes(b"old text")
    tools = st.SandboxToolSession(sandbox_mgr=manager, tool_registry=None)
    tools.attach(sid)
    return sid, workspace, tools


def _idle(manager, sid):
    """The session idle past the TTL by its last activity."""
    manager.get_session(sid).last_activity = time.time() - 2 * manager._config.idle_ttl_seconds


def _racing(module, name, before=None, after=None):
    """``module.name`` running ``before`` (and ``after``) around its real self, once; the real one."""
    real = getattr(module, name)
    pending = {"before": before, "after": after}

    def racing(*args, **kwargs):
        first, pending["before"] = pending["before"], None
        if first is not None:
            first()
        done = real(*args, **kwargs)
        then, pending["after"] = pending["after"], None
        if then is not None:
            then()
        return done

    setattr(module, name, racing)
    return real


def _outcome(call, *args, **kwargs):
    """("answered", the answer) or (the exception's name, its text)."""
    try:
        return "answered", call(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- the contract reads it
        return type(exc).__name__, str(exc)


# name: (the seam's module, "ft" or "st"; the seam, called at the tool's first write or read; the call)
_TOOLS = {
    "create_file": ("ft", "_write_checked", lambda tools: tools.create_file("new.txt", "made")),
    "str_replace": ("ft", "_rewrite", lambda tools: tools.str_replace("a.txt", "old", "new")),
    "view": ("ft", "_read_at_most", lambda tools: tools.view("a.txt")),
    "grep": ("st", "_grep_files", lambda tools: tools.grep("old")),
    "glob": ("st", "_glob_walk", lambda tools: tools.glob("*.txt")),
    "ls": ("st", "_open_resolved", lambda tools: tools.ls(".")),
}


def test_fh1_no_sweep_takes_a_workspace_an_agent_tool_works_in(tmp_path):
    sm, ft, st, manager, restore = _window(tmp_path)
    modules = {"ft": ft, "st": st}
    seen = {}
    try:
        for name, (where, seam, call) in _TOOLS.items():
            sid, workspace, tools = _session(st, manager, f"fh1-{name}")
            kept = []

            def sweep(sid=sid, kept=kept):
                _idle(manager, sid)
                manager.list_sessions()
                kept.append(manager.get_session(sid) is not None)

            _idle(manager, sid)
            real = _racing(modules[where], seam, before=sweep)
            try:
                answer = _outcome(call, tools)
            finally:
                setattr(modules[where], seam, real)
            manager.list_sessions()
            after = manager.get_session(sid) is not None
            files = {p.name: p.read_bytes() for p in workspace.iterdir() if p.is_file()} if workspace.is_dir() else {}
            seen[name] = (answer, kept, after, files)
            if manager.get_session(sid) is not None:
                manager.destroy_sandbox(sid)
    finally:
        restore()
    for name, (answer, kept, after, _files) in seen.items():
        assert kept == [True], f"{name}: the sweep took a workspace an agent tool worked in: {seen[name]}"
        assert after, f"{name}: the tool's call counted no activity, the next list took the session: {seen[name]}"
        assert answer[0] == "answered" and not answer[1].startswith("Error"), (name, seen[name])
    assert seen["create_file"][0][1] == "File created: new.txt (4 bytes)", seen["create_file"]
    assert seen["create_file"][3] == {"a.txt": b"old text", "new.txt": b"made"}, seen["create_file"]
    assert seen["str_replace"][3] == {"a.txt": b"new text"}, seen["str_replace"]
    assert seen["view"][0][1] == "File: a.txt (1 lines total)\n     1\told text", seen["view"]
    assert seen["grep"][0][1] == "1 match(es) in 1 file(s)\na.txt:1: old text", seen["grep"]
    assert seen["glob"][0][1] == "1 file(s)\na.txt", seen["glob"]
    assert seen["ls"][0][1] == "file 8 a.txt", seen["ls"]


def test_fh2_a_destroy_during_a_write_answers_so(tmp_path):
    sm, ft, st, manager, restore = _window(tmp_path)
    cases = {
        # name: (the seam, the call, whether the destroy comes before the seam's real self)
        "create_file before": ("_write_checked", lambda tools: tools.create_file("new.txt", "made"), True),
        "create_file after": ("_write_checked", lambda tools: tools.create_file("new.txt", "made"), False),
        "create_file checked": ("_resolve_path", lambda tools: tools.create_file("new.txt", "made"), False),
        "str_replace before": ("_rewrite", lambda tools: tools.str_replace("a.txt", "old", "new"), True),
        "str_replace after": ("_rewrite", lambda tools: tools.str_replace("a.txt", "old", "new"), False),
        "str_replace checked": ("_resolve_path", lambda tools: tools.str_replace("a.txt", "old", "new"), False),
        "view before": ("_read_at_most", lambda tools: tools.view("a.txt"), True),
    }
    seen = {}
    try:
        for name, (seam, call, first) in cases.items():
            sid, workspace, tools = _session(st, manager, f"fh2-{name.replace(' ', '-')}")
            session = manager.get_session(sid)
            destroyed = []

            def destroy(sid=sid, destroyed=destroyed):
                destroyed.append(manager.destroy_sandbox(sid))

            real = _racing(ft, seam, **({"before": destroy} if first else {"after": destroy}))
            try:
                answer = _outcome(call, tools)
            finally:
                setattr(ft, seam, real)
            seen[name] = (sid, answer, destroyed, workspace.exists(), manager.get_session(sid) is None,
                          session.host_operations)
        # A session made again under the id while a write is held: the old call's end neither touches nor writes it
        remade_sid, _workspace, tools = _session(st, manager, "fh2-remade")
        remade = {}

        def remake():
            manager.destroy_sandbox(remade_sid)
            remade["session"] = manager.create_sandbox(remade_sid, label="fh2-remade-new")
            _idle(manager, remade_sid)
            remade["since"] = remade["session"].last_activity

        real = _racing(ft, "_write_checked", before=remake)
        try:
            again = _outcome(tools.create_file, "new.txt", "made")
        finally:
            ft._write_checked = real
        new = remade["session"]
        remade_seen = (again, new.last_activity == remade["since"], new.host_operations,
                       sorted(p.name for p in Path(new.workspace_path).iterdir()))
        manager.destroy_sandbox(remade_sid)
    finally:
        restore()
    assert remade_seen == (
        ("answered", f"Error: Session {remade_sid} was destroyed while the file was written"), True, 0, []), (
        f"a session made again under the id is touched or written by the old call: {remade_seen}")
    for name, (sid, answer, destroyed, stays, gone, held) in seen.items():
        assert destroyed == [True] and gone and not stays, f"{name}: the destroy was held off: {seen[name]}"
        assert held == 0, f"{name}: the session stayed in use: {seen[name]}"
        if name.startswith("view"):
            assert answer == ("answered", "File: a.txt (1 lines total)\n     1\told text"), (
                f"a view a destroy overtook lost its answer: {seen[name]}")
        else:
            assert answer == ("answered", f"Error: Session {sid} was destroyed while the file was written"), (
                f"{name}: a write a destroy overtook is answered otherwise: {seen[name]}")


def test_fh3_what_a_call_counts(tmp_path):
    sm, ft, st, manager, restore = _window(tmp_path)
    try:
        # Unknown to the manager: answered as before
        unknown = (
            ft._handle_sandbox_create_file("no-such", "a.txt", "x", _sandbox_manager=manager),
            ft._handle_sandbox_str_replace("no-such", "a.txt", "a", "b", _sandbox_manager=manager),
            ft._handle_sandbox_view("no-such", "a.txt", _sandbox_manager=manager),
        )
        # Ended under the agent's tool session, its id made again for another session idle past the TTL: answered as
        # before, and the other session neither held nor touched (round 2)
        ended_sid, _workspace, ended = _session(st, manager, "fh3-ended")
        manager.destroy_sandbox(ended_sid)
        other = manager.create_sandbox(ended_sid, label="fh3-other")
        _idle(manager, ended_sid)
        since = other.last_activity
        after_end = {name: _outcome(call, ended) for name, (_where, _seam, call) in _TOOLS.items()}
        other_seen = (other.last_activity == since, other.host_operations)
        manager.destroy_sandbox(ended_sid)
        # Refused calls, each on a session of its own idle past the TTL: each counts, nothing stays in use
        refusals = {
            "path out": lambda tools: tools.create_file("../out.txt", "x"),
            "too large": lambda tools: tools.create_file("big.txt", "x" * (ft.MAX_CREATE_SIZE + 1)),
            "no match": lambda tools: tools.str_replace("a.txt", "absent", "b"),
            "file not found": lambda tools: tools.str_replace("missing.txt", "a", "b"),
            "view not found": lambda tools: tools.view("missing.txt"),
            "grep empty": lambda tools: tools.grep(""),
            "glob empty": lambda tools: tools.glob(""),
            "ls not found": lambda tools: tools.ls("missing"),
        }
        counted = {}
        for name, call in refusals.items():
            sid, _workspace, tools = _session(st, manager, f"fh3-{name.replace(' ', '-')}")
            session = manager.get_session(sid)
            _idle(manager, sid)
            answer = _outcome(call, tools)
            manager.list_sessions()
            counted[name] = (answer, manager.get_session(sid) is not None, session.host_operations)
            if manager.get_session(sid) is not None:
                manager.destroy_sandbox(sid)
        # A tool that raised: nothing stays in use, the sweep then takes the idle session
        sid, workspace, tools = _session(st, manager, "fh3-raised")
        session = manager.get_session(sid)
        real = ft._write_checked

        def broken(*args, **kwargs):
            raise RuntimeError("stand-in failure")

        ft._write_checked = broken
        try:
            raised = _outcome(tools.create_file, "c.txt", "x")
        finally:
            ft._write_checked = real
        held_after_raise = session.host_operations
        _idle(manager, sid)
        manager.list_sessions()
        swept = manager.get_session(sid) is None
        stays = workspace.exists()
    finally:
        restore()
    assert unknown == ("Error: Session not found or inactive: no-such",) * 3, unknown
    for name, answer in after_end.items():
        assert answer == ("RuntimeError", "No active sandbox session"), (
            f"{name} on a session already ended ({ended_sid}) is answered otherwise: {after_end}")
    assert other_seen == (True, 0), (
        f"a call of an ended tool session held or touched its id's new session: {other_seen}")
    for name, (answer, kept, held) in counted.items():
        assert answer[0] == "answered" and answer[1].startswith("Error"), (name, counted[name])
        assert kept, f"a refused call ({name}) counted no activity, the sweep took the session: {counted}"
        assert held == 0, f"a refused call ({name}) left the session in use: {counted}"
    assert raised == ("RuntimeError", "stand-in failure"), raised
    assert held_after_raise == 0 and swept and not stays, (
        f"a tool that raised left the session in use: {(held_after_raise, swept, stays)}")
