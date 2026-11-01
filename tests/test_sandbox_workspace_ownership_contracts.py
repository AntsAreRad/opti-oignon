"""Contracts for the sandbox's workspaces: one in use is never reaped by another manager.

A manager starting with ``reconcile_on_start`` (as shipped) removed every ``sandbox-*`` directory of its workspace
base that its own sessions did not hold, and a new manager holds none: importing the module (its singleton) from a
command line, or the note routes' manager built per request inside the server, removed the live workspaces of every
other manager -- a coding session's files under a command still running. Now each live workspace's directory is held
by its manager under an exclusive flock for the session's life, and a start removes only a directory whose lock it
can take.

  * WL1 -- A START LEAVES WHAT ANOTHER MANAGER HOLDS: a manager starting on a base where another manager of this
    process holds a live workspace leaves it with its files, removes a true orphan, and says what it left (an INFO
    line); a workspace another process holds is left too, and removed by the next start once that process is gone.
  * WL2 -- NO WINDOW AT CREATION: another manager's reconciliation run between a workspace's making and its hold
    never costs the new session its workspace: the session returned is live, held, and its directory is there; the
    same id created twice at once leaves one session, and the create that finishes second keeps neither its
    workspace nor its hold (round 1).
  * WL3 -- THE HOLD LASTS AS LONG AS THE SESSION: held while live, on a descriptor a command run in the workspace
    does not inherit; destroying removes the tree, then releases it, and says a tree it cannot remove whole; a
    manager dropped with live sessions releases them when it is dropped, with no cycle collection (round 1), and the
    next start removes what they left.
  * WL4 -- THROUGH THE REAL START: the module's import-time manager, on a base where another manager holds a live
    workspace (the command line's case beside the server), leaves it; a directory it cannot open is left and said
    as not checked, and one it cannot remove whole is said and not counted as removed (WARNINGs; round 1).
  * WL5 -- WHERE THE FILE SYSTEM REFUSES THE LOCK, AND ONLY THERE (flock over NFS on a directory opened read-only
    fails EBADF): a workspace is still made, unheld, and said; a start there removes none (round 1); where the base
    locks, a hold that fails otherwise (descriptors exhausted) refuses the create and leaves nothing (round 2).

The manager module is loaded from a copy of its file next to a configuration of the test's own (the import-time
singleton off, but in WL4, where its workspace base is the test's), so nothing reaches the configured base. Bubblewrap
detection is stood in for, but in WL4, whose import-time start runs the real one (`bwrap --version` and its
namespace check); no command of the tests runs in a sandbox. Local-only.
"""
from __future__ import annotations

import errno
import fcntl
import gc
import logging
import os
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_wl1_a_start_leaves_what_another_manager_holds": 6.0,
    "test_wl2_no_window_at_creation": 3.0,
    "test_wl3_the_hold_lasts_as_long_as_the_session": 3.0,
    "test_wl4_through_the_real_start": 6.0,
    "test_wl5_where_the_file_system_refuses_the_lock_and_only_there": 3.0,
}

_SM = "opti_oignon.sandbox_manager"
# A process outside the package holding a directory as a manager does: it says "held", then waits for its stdin.
_HOLDER = ("import fcntl, os, sys\n"
           "fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY)\n"
           "fcntl.flock(fd, fcntl.LOCK_EX)\n"
           "print('held', flush=True)\n"
           "sys.stdin.read()\n")


class _Said(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []
        self.levels = []

    def emit(self, record):
        self.lines.append(record.getMessage())
        self.levels.append(record.levelno)

    def at(self, level):
        """What was said at ``level`` or above: what an operator's log shows."""
        return [line for line, said in zip(self.lines, self.levels) if said >= level]


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _loaded(tmp_path, config="enabled: false\n", *, secrets=False):
    """sandbox_manager.py loaded from a byte-exact copy beside ``config``, with what it logs; the closer restores.
    ``secrets`` loads the real process_secrets beside it, which a start with the sandbox on reaches."""
    package = tmp_path / f"copy{len(list(tmp_path.glob('copy*')))}"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text(config, encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    said = _Said()
    logger = logging.getLogger(_SM)
    level = logger.level
    logger.addHandler(said)
    logger.setLevel(logging.DEBUG)
    db_utils = _module("opti_oignon.db_utils", safe_connect=lambda path, *a, **k: sqlite3.connect(path))
    try:
        targets = {"opti_oignon.process_secrets": source("process_secrets.py")} if secrets else {}
        targets[_SM] = copy
        loaded, restore = isolate(targets=targets, seeded={"opti_oignon.db_utils": db_utils})
    except BaseException:
        logger.removeHandler(said)
        logger.setLevel(level)
        raise

    def close():
        try:
            restore()
        finally:
            logger.removeHandler(said)
            logger.setLevel(level)

    return loaded[_SM], said, close


def _manager(sm, base, *, reconcile=True):
    sm._detect_bwrap = lambda: (True, "stand-in bwrap")
    config = sm.SandboxConfig(isolation_backend="tempdir", strict_mode=False, require_degraded_confirmation=False,
                              workspace_base=str(base), reconcile_on_start=reconcile)
    return sm.SandboxManager(config)


def _held_elsewhere(path):
    """Whether another open description holds ``path``'s directory: a lock this test cannot take."""
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def _fds_on(path):
    """This process's descriptors open on ``path``."""
    found = []
    for name in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{name}") == str(path):
                found.append(int(name))
        except OSError:
            continue
    return found


def _fds_on_prefix(base, prefix):
    """This process's descriptors open on a directory of ``base`` whose name starts with ``prefix``, kept or gone."""
    found = []
    for name in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if target.startswith(str(base / prefix)):
            found.append(int(name))
    return found


def _orphan(base, name, words="left behind\n"):
    path = base / name
    path.mkdir(mode=0o700)
    (path / "left.txt").write_text(words, encoding="utf-8")
    return path


def test_wl1_a_start_leaves_what_another_manager_holds(tmp_path):
    sm, said, close = _loaded(tmp_path)
    base = tmp_path / "workspaces"
    holder = None
    try:
        server = _manager(sm, base)
        live = server.create_sandbox("chat")
        Path(live.workspace_path, "work.py").write_text("print('mine')\n", encoding="utf-8")
        orphan = _orphan(base, "sandbox-gone-0000")
        said.lines.clear()
        said.levels.clear()

        _manager(sm, base)
        assert Path(live.workspace_path, "work.py").is_file(), "a start removed a workspace another manager holds"
        assert Path(live.workspace_path, "work.py").read_text(encoding="utf-8") == "print('mine')\n"
        assert server.get_workspace_path("chat") == live.workspace_path
        assert not orphan.exists(), "a true orphan stays"
        left = [line for line in said.at(logging.INFO) if os.path.basename(live.workspace_path) in line]
        assert left and "holds" in left[0], said.lines

        elsewhere = _orphan(base, "sandbox-elsewhere-0000", "another process's\n")
        holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(elsewhere)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)
        assert holder.stdout.readline().strip() == "held"
        _manager(sm, base)
        assert (elsewhere / "left.txt").is_file(), "a start removed a workspace another process holds"
        assert (elsewhere / "left.txt").read_text(encoding="utf-8") == "another process's\n"
        holder.stdin.close()
        assert holder.wait(timeout=5) == 0
        holder = None
        _manager(sm, base)
        assert not elsewhere.exists(), "the workspace of a process gone stays"
        assert Path(live.workspace_path, "work.py").exists()
        assert server.destroy_sandbox("chat") and not Path(live.workspace_path).exists()
    finally:
        if holder is not None:
            holder.kill()
            holder.wait()
        close()


def test_wl2_no_window_at_creation(tmp_path):
    sm, said, close = _loaded(tmp_path)
    base = tmp_path / "workspaces"
    real = sm.tempfile
    try:
        server = _manager(sm, base)
        other = _manager(sm, base, reconcile=False)
        raced = []

        def mkdtemp(*args, **kwargs):
            made = real.mkdtemp(*args, **kwargs)
            if not raced:
                raced.append(other.reconcile_workspaces())
            return made

        sm.tempfile = types.SimpleNamespace(mkdtemp=mkdtemp, gettempdir=real.gettempdir)
        session = server.create_sandbox("raced")
        sm.tempfile = real
        assert raced, "the hook ran"
        path = Path(session.workspace_path)
        assert path.is_dir(), "the new session's workspace was removed between its making and its hold"
        assert server.get_workspace_path("raced") == str(path)
        assert _held_elsewhere(path), "the new workspace is not held"
        assert server.destroy_sandbox("raced")

        # The same id created twice at once (round 1): the create that finishes second keeps nothing.
        inner, started = [], []

        def twice(*args, **kwargs):
            made = real.mkdtemp(*args, **kwargs)
            if not started:
                started.append(made)
                inner.append(server.create_sandbox("twice"))
            return made

        sm.tempfile = types.SimpleNamespace(mkdtemp=twice, gettempdir=real.gettempdir)
        try:
            server.create_sandbox("twice")
        except ValueError:
            refused = True
        else:
            refused = False
        sm.tempfile = real
        first = Path(inner[0].workspace_path)
        assert refused, "the same id was created twice"
        assert server.get_workspace_path("twice") == str(first) and _held_elsewhere(first)
        made = [p for p in base.glob("sandbox-twice-*") if p != first]
        assert made == [], ("the refused create left its workspace", made)
        stray = [fd for fd in _fds_on_prefix(base, "sandbox-twice-")
                 if os.readlink(f"/proc/self/fd/{fd}") != str(first)]
        assert stray == [], "the refused create left its hold open"
        assert server.destroy_sandbox("twice") and _fds_on_prefix(base, "sandbox-twice-") == []
    finally:
        sm.tempfile = real
        close()


def test_wl3_the_hold_lasts_as_long_as_the_session(tmp_path):
    sm, said, close = _loaded(tmp_path)
    base = tmp_path / "workspaces"
    try:
        server = _manager(sm, base)
        session = server.create_sandbox("kept")
        path = Path(session.workspace_path)
        assert _held_elsewhere(path), "a live workspace is not held"
        fds = _fds_on(path)
        assert len(fds) == 1, fds
        assert not os.get_inheritable(fds[0]), "a command run in the workspace would inherit its hold"
        assert server.destroy_sandbox("kept")
        assert not path.exists() and _fds_on(path) == [], "destroying left the tree or the hold"

        # A tree destroying cannot remove whole is said (round 1).
        stuck = Path(server.create_sandbox("stuck").workspace_path)
        (stuck / "sealed").mkdir()
        (stuck / "sealed" / "kept.txt").write_text("x\n", encoding="utf-8")
        (stuck / "sealed").chmod(0o500)
        try:
            said.lines.clear()
            said.levels.clear()
            assert server.destroy_sandbox("stuck")
            assert (stuck / "sealed" / "kept.txt").exists()
            assert [line for line in said.at(logging.WARNING) if "could not be removed whole" in line], said.lines
        finally:
            (stuck / "sealed").chmod(0o700)

        dropped = _manager(sm, base)
        left = Path(dropped.create_sandbox("dropped").workspace_path)
        (left / "work.py").write_text("x = 1\n", encoding="utf-8")
        assert _held_elsewhere(left)
        collecting = gc.isenabled()
        gc.disable()
        try:
            del dropped
            assert _fds_on(left) == [], "a manager dropped with a live session keeps its hold until a collection"
        finally:
            if collecting:
                gc.enable()
        _manager(sm, base)
        assert not left.exists(), "the next start keeps what a dropped manager left"
    finally:
        close()


def test_wl4_through_the_real_start(tmp_path):
    sm, said, close = _loaded(tmp_path)
    base = tmp_path / "workspaces"
    late = None
    try:
        server = _manager(sm, base)
        live = server.create_sandbox("server")
        Path(live.workspace_path, "work.py").write_text("print('server')\n", encoding="utf-8")
        shut = base / "sandbox-shut-0000"
        shut.mkdir(mode=0o700)
        (shut / "inside.txt").write_text("kept\n", encoding="utf-8")
        shut.chmod(0)
        stuck = _orphan(base, "sandbox-stuck-0000")
        (stuck / "sealed").mkdir()
        (stuck / "sealed" / "kept.txt").write_text("x\n", encoding="utf-8")
        (stuck / "sealed").chmod(0o500)
        config = ("enabled: true\nisolation_backend: tempdir\nrequire_degraded_confirmation: false\n"
                  f"reconcile_on_start: true\nworkspace_base: {base}\n")
        late, late_said, late_close = _loaded(tmp_path, config, secrets=True)
        try:
            assert late.sandbox_manager is not None, ("the import-time manager started", late_said.lines)
            assert Path(live.workspace_path, "work.py").is_file(), (
                "the import-time manager removed the server's live workspace")
            assert Path(live.workspace_path, "work.py").read_text(encoding="utf-8") == "print('server')\n"
            shut.chmod(0o700)
            assert (shut / "inside.txt").exists()
            unchecked = [line for line in late_said.at(logging.WARNING) if "sandbox-shut-0000" in line]
            assert unchecked and "not checked" in unchecked[0], late_said.lines
            assert (stuck / "sealed" / "kept.txt").exists()
            whole = [line for line in late_said.at(logging.WARNING) if "sandbox-stuck-0000" in line]
            assert whole and "removed whole" in whole[0], late_said.lines
            assert not [line for line in late_said.lines if "reaped" in line], ("counted as removed", late_said.lines)
        finally:
            shut.chmod(0o700)
            (stuck / "sealed").chmod(0o700)
            late_close()
        assert server.destroy_sandbox("server")
    finally:
        close()


def test_wl5_where_the_file_system_refuses_the_lock_and_only_there(tmp_path):
    sm, said, close = _loaded(tmp_path)
    base = tmp_path / "workspaces"
    real_fcntl, real_hold = getattr(sm, "fcntl", None), getattr(sm, "_hold_workspace", None)
    try:
        def refused(fd, how):
            raise OSError(errno.EBADF, os.strerror(errno.EBADF))

        sm.fcntl = types.SimpleNamespace(flock=refused, LOCK_EX=fcntl.LOCK_EX, LOCK_SH=fcntl.LOCK_SH,
                                         LOCK_NB=fcntl.LOCK_NB, LOCK_UN=fcntl.LOCK_UN)
        server = _manager(sm, base)
        session = server.create_sandbox("unheld")
        path = Path(session.workspace_path)
        assert path.is_dir() and server.get_workspace_path("unheld") == str(path), "no workspace where locks fail"
        (path / "work.py").write_text("print('unheld')\n", encoding="utf-8")
        assert [line for line in said.at(logging.WARNING) if "cannot be locked" in line], said.lines
        said.lines.clear()
        said.levels.clear()

        _manager(sm, base)
        assert (path / "work.py").is_file(), "a start removed a workspace it could not lock"
        assert [line for line in said.at(logging.WARNING) if path.name in line and "not checked" in line], said.lines
        assert server.destroy_sandbox("unheld") and not path.exists()

        # Only there (round 2): where the base locks, a hold that fails otherwise -- descriptors exhausted --
        # refuses the create and leaves nothing, never a workspace unheld that another start would remove.
        sm.fcntl = real_fcntl
        locking = _manager(sm, base)

        def exhausted(path):
            raise OSError(errno.EMFILE, os.strerror(errno.EMFILE))

        sm._hold_workspace = exhausted
        try:
            locking.create_sandbox("exhausted")
        except OSError as exc:
            assert exc.errno == errno.EMFILE, exc
        else:
            raise AssertionError("a workspace was made unheld on a base that locks")
        finally:
            sm._hold_workspace = real_hold
        assert locking.get_session("exhausted") is None
        assert list(base.glob("sandbox-exhausted-*")) == [], "the refused create left its directory"
    finally:
        sm.fcntl, sm._hold_workspace = real_fcntl, real_hold
        close()
