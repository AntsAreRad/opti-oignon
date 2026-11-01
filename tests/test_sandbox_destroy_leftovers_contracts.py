"""Contracts for a sandbox's destroy: a workspace it could not remove stays counted, and the sandbox cannot freeze one.

``destroy_sandbox`` removed a workspace with ``shutil.rmtree(ignore_errors=True)``, then dropped the session and closed
its hold whether or not the tree was gone: a tree it could not remove whole was only logged, counted by nothing,
retried by nothing, its hold released. The sandbox's own code made such a tree in one command: a folder whose write bit
it removed. Now the removal gives a frozen folder back its owner's rights, reached through no link, and a tree that
still stands is kept: counted against the limit on concurrent sessions, removed again by the idle sweep and by
``cleanup_all``, its hold closed once it is gone.

  * LO1 -- A FROZEN WORKSPACE IS REMOVED WHOLE: folders of modes ``0o500``, ``0o000`` and ``0o600`` with files
    inside, the workspace itself ``0o500``, a link to a host folder of mode ``0o555`` and one to a host file: the
    destroy removes the workspace whole, its hold closed, and the host keeps its modes and its files -- also where
    ``/proc`` cannot serve the rights given back, for a folder its owner may still read and search; a start removes a frozen orphan whole the same way, one whose own
    folder its sandbox shut (``0o000``) too (round 1).
  * LO2 -- A TREE THAT STANDS STAYS COUNTED: a file the removal cannot unlink (a stand-in for what the owner cannot
    undo): the destroy answers True, the tree stays, its hold kept, and with a limit of one a new creation is refused
    with a message that names it, before it makes a workspace -- and so is one whose workspace was being made when
    the tree was left; once the file can go, the next creation's sweep removes the tree, closes its hold,
    and registers; ``cleanup_all`` does the same; a reconcile of the same manager leaves it to the sweep, never taking
    it for another manager's; the status counts it with the active sessions (round 1).
  * LO3 -- WHAT STAYS: a plain destroy answers True and leaves nothing; an unknown id answers False; the hold of a
    destroyed workspace is closed.
  * LO4 -- TWO DESTROYS RECORD THE TREE ONCE: a second destroy of the same session while the first removes its tree,
    neither able to remove it whole: the tree is kept once, with its hold, which the sweep closes once it is gone.
    (Before this block no tree was kept at all: on 207 the contract finds no second destroy to watch.)
  * LO5 -- A SWAP NEVER REACHES THE HOST (round 1): a frozen folder swapped for a link to a host folder between its
    stat and its opening, between its opening and its rights given back, and past the depth where folders are moved
    up instead of entered (round 2): the host keeps its mode and its file, and the next sweep removes what the swap
    left.
  * LO6 -- A TREE OF ANY DEPTH GOES (round 1): a chain of folders deeper than the descriptors left to the process,
    every level frozen: the destroy removes it whole; and where no folder can be moved up (a running command can
    make each move fail), the walk still holds no more descriptors than its depth bound (round 2).
  * LO7 -- WL3, SUPERSEDED (the hold lasts as long as the session): its cases and assertions unchanged, a tree a
    destroy cannot remove whole stood for by a file whose removal is refused -- a folder of mode ``0o500`` now goes.
  * LO8 -- WL4, SUPERSEDED (through the real start): its cases and assertions unchanged, a directory the start
    cannot check stood for by a folder of mode ``0o000`` whose rights cannot be given back (another owner's), and one
    it cannot remove whole by a file whose removal is refused -- both now go when their owner's rights do.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import gc
import logging
import os
import resource
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402
import test_sandbox_workspace_ownership_contracts as wl_t  # noqa: E402

BUDGET_S = {
    "test_lo1_a_frozen_workspace_is_removed_whole": 5.0,
    "test_lo2_a_tree_that_stands_stays_counted": 5.0,
    "test_lo3_what_a_destroy_keeps": 5.0,
    "test_lo4_two_destroys_record_the_tree_once": 5.0,
    "test_lo5_a_swap_never_reaches_the_host": 5.0,
    "test_lo6_a_tree_of_any_depth_goes": 5.0,
    "test_lo7_wl3_the_hold_lasts_as_long_as_the_session": 5.0,
    "test_lo8_wl4_through_the_real_start": 10.0,
}


def _hold(manager, sid):
    """The session's hold and the folder it holds: its number may be another's once it is closed."""
    fd = manager._holds[sid]
    held = os.fstat(fd)
    return fd, (held.st_dev, held.st_ino)


def _closed(hold):
    fd, folder = hold
    try:
        now = os.fstat(fd)
    except OSError:
        return True
    return (now.st_dev, now.st_ino) != folder


def _freeze(workspace, host, host_file):
    """A workspace as the sandbox's code can leave it: folders it froze, links out of it."""
    (workspace / "frozen" / "inner").mkdir(parents=True)
    (workspace / "frozen" / "f.txt").write_bytes(b"f")
    (workspace / "frozen" / "inner" / "g.txt").write_bytes(b"g")
    (workspace / "shut" / "deep").mkdir(parents=True)
    (workspace / "shut" / "deep" / "h.txt").write_bytes(b"h")
    (workspace / "unsearchable" / "deep").mkdir(parents=True)
    (workspace / "unsearchable" / "i.txt").write_bytes(b"i")
    (workspace / "door").symlink_to(host)
    (workspace / "frozen" / "file-door").symlink_to(host_file)
    os.chmod(workspace / "frozen" / "inner", 0o500)
    os.chmod(workspace / "frozen", 0o500)
    os.chmod(workspace / "shut", 0o000)
    os.chmod(workspace / "unsearchable", 0o600)
    os.chmod(workspace, 0o500)


def _thaw(base):
    """Owner rights back on whatever a failed contract left under ``base``, for the temporary folder's removal."""
    if os.path.lexists(base):
        for folder, dirs, _files in os.walk(base):
            for name in dirs:
                os.chmod(os.path.join(folder, name), 0o700)


def test_lo1_a_frozen_workspace_is_removed_whole(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    (host / "keep.txt").write_bytes(b"host")
    host_file = tmp_path / "host.txt"
    host_file.write_bytes(b"host file")
    os.chmod(host_file, 0o400)
    os.chmod(host, 0o555)
    base = tmp_path / "workspaces"
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid = manager.create_sandbox("lo1").session_id
        workspace = Path(manager.get_workspace_path(sid))
        _freeze(workspace, host, host_file)
        hold = _hold(manager, sid)
        destroyed = manager.destroy_sandbox(sid)
        left = os.path.lexists(workspace)
        leftovers = dict(getattr(manager, "_leftovers", {}))
        # Where /proc cannot serve the rights given back
        sid = manager.create_sandbox("lo1-no-proc").session_id
        workspace_no_proc = Path(manager.get_workspace_path(sid))
        (workspace_no_proc / "frozen").mkdir()
        (workspace_no_proc / "frozen" / "f.txt").write_bytes(b"f")
        os.chmod(workspace_no_proc / "frozen", 0o500)
        real_chmod = os.chmod

        def no_proc(path, *args, **kwargs):
            if os.fsdecode(path).startswith("/proc/"):
                raise FileNotFoundError(2, "No such file or directory", path)
            return real_chmod(path, *args, **kwargs)

        os.chmod = no_proc
        try:
            manager.destroy_sandbox(sid)
        finally:
            os.chmod = real_chmod
        left_no_proc = os.path.lexists(workspace_no_proc)
        # Orphans of a manager gone, as frozen, one with its own folder shut: a start removes them whole
        orphan = base / "sandbox-lo1-orphan"
        orphan.mkdir()
        _freeze(orphan, host, host_file)
        shut = base / "sandbox-lo1-shut"
        (shut / "inner").mkdir(parents=True)
        (shut / "inner" / "s.txt").write_bytes(b"s")
        os.chmod(shut, 0o000)
        ce_t._manager(sm, tmp_path, reconcile_on_start=True)
        orphan_left = os.path.lexists(orphan)
        shut_left = os.path.lexists(shut)
    finally:
        restore()
        _thaw(base)
    host_mode = stat.S_IMODE(os.stat(host).st_mode)
    host_file_mode = stat.S_IMODE(os.stat(host_file).st_mode)
    os.chmod(host, 0o755)
    kept = (host / "keep.txt").read_bytes(), host_file.read_bytes()
    assert destroyed is True, destroyed
    assert not left, "a workspace the sandbox froze survived its destroy"
    assert leftovers == {}, leftovers
    assert _closed(hold), "the destroyed workspace's hold was left open"
    assert not left_no_proc, "a frozen workspace survived its destroy where /proc cannot serve"
    assert not orphan_left, "a start left a frozen orphan standing"
    assert not shut_left, "a start left an orphan whose own folder its sandbox shut"
    assert (host_mode, host_file_mode) == (0o555, 0o400), (
        f"the removal changed the host's modes through a link: {oct(host_mode)} {oct(host_file_mode)}")
    assert kept == (b"host", b"host file"), kept


def test_lo2_a_tree_that_stands_stays_counted(tmp_path, monkeypatch, caplog):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path, max_concurrent_sessions=1)
        sid = manager.create_sandbox("lo2").session_id
        workspace = Path(manager.get_workspace_path(sid))
        (workspace / "stuck.txt").write_bytes(b"stuck")
        hold = _hold(manager, sid)
        real_unlink = os.unlink

        def refusing(path, *args, **kwargs):
            # What the owner cannot undo: this one name refuses its removal
            if os.path.basename(os.fsdecode(path)) == "stuck.txt":
                raise PermissionError(1, "Operation not permitted", path)
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", refusing)
        destroyed = manager.destroy_sandbox(sid)
        stands = os.path.lexists(workspace)
        real_make = manager._make_workspace
        made_at_limit = []
        manager._make_workspace = lambda sid: made_at_limit.append(sid) or real_make(sid)
        try:
            # Past the limit while the tree stands: a place given here is given back
            manager.destroy_sandbox(manager.create_sandbox("lo2-next").session_id)
            refused = None
        except ValueError as exc:
            refused = str(exc)
        manager._make_workspace = real_make
        hold_open_while_counted = not _closed(hold)
        status_count = manager.active_session_count
        caplog.set_level(logging.INFO)
        caplog.clear()
        manager.reconcile_workspaces()
        taken_for_another = [r.getMessage() for r in caplog.records if "another manager" in r.getMessage()]
        kept_by_reconcile = os.path.lexists(workspace) and workspace.as_posix() in getattr(manager, "_leftovers", {})
        monkeypatch.undo()
        after = manager.create_sandbox("lo2-after").session_id
        gone = not os.path.lexists(workspace)
        hold_closed = _closed(hold)
        manager.destroy_sandbox(after)

        # A tree left while a creation made its workspace: the registration counts it
        def leaving(sid):
            made = real_make(sid)
            manager._make_workspace = real_make
            inner = manager.create_sandbox("lo2-inner").session_id
            (Path(manager.get_workspace_path(inner)) / "stuck.txt").write_bytes(b"stuck")
            monkeypatch.setattr(os, "unlink", refusing)
            manager.destroy_sandbox(inner)
            monkeypatch.undo()
            return made

        manager._make_workspace = leaving
        try:
            manager.destroy_sandbox(manager.create_sandbox("lo2-raced").session_id)
            raced = None
        except ValueError as exc:
            raced = str(exc)
        manager._make_workspace = real_make
        # cleanup_all removes a leftover too
        sid2 = manager.create_sandbox("lo2-again").session_id
        workspace2 = Path(manager.get_workspace_path(sid2))
        (workspace2 / "stuck.txt").write_bytes(b"stuck")
        monkeypatch.setattr(os, "unlink", refusing)
        manager.destroy_sandbox(sid2)
        monkeypatch.undo()
        stands2 = os.path.lexists(workspace2)
        manager.cleanup_all()
        gone2 = not os.path.lexists(workspace2)
        leftovers = dict(getattr(manager, "_leftovers", {}))
    finally:
        monkeypatch.undo()
        restore()
    assert destroyed is True and stands, (destroyed, stands)
    limit = "Maximum concurrent sessions reached (1): 1 destroyed workspace(s) not yet removed count against it"
    assert refused == limit, f"a tree a destroy could not remove was not counted against the limit: {refused}"
    assert made_at_limit == [], f"a creation refused for a tree left standing made a workspace first: {made_at_limit}"
    assert raced == limit, f"a registration did not count a tree left while its workspace was made: {raced}"
    assert hold_open_while_counted, "the hold of a tree still counted was closed"
    assert kept_by_reconcile, "the manager's own reconcile took a tree it still counts"
    assert taken_for_another == [], f"the manager's own reconcile took its tree for another manager's: {taken_for_another}"
    assert status_count == 1, f"the status counts {status_count} place(s) where the limit counts 1"
    assert after == "lo2-after" and gone and hold_closed, (after, gone, hold_closed)
    assert stands2 and gone2 and leftovers == {}, (stands2, gone2, leftovers)


def test_lo3_what_a_destroy_keeps(tmp_path):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid = manager.create_sandbox("lo3").session_id
        workspace = Path(manager.get_workspace_path(sid))
        (workspace / "a" / "b").mkdir(parents=True)
        (workspace / "a" / "b" / "c.txt").write_bytes(b"c")
        hold = _hold(manager, sid)
        destroyed = manager.destroy_sandbox(sid)
        unknown = manager.destroy_sandbox("lo3-nobody")
        left = os.path.lexists(workspace)
        leftovers = dict(getattr(manager, "_leftovers", {}))
    finally:
        restore()
    assert destroyed is True and unknown is False, (destroyed, unknown)
    assert not left and leftovers == {} and _closed(hold), (left, leftovers)


def test_lo4_two_destroys_record_the_tree_once(tmp_path, monkeypatch):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid = manager.create_sandbox("lo4").session_id
        workspace = Path(manager.get_workspace_path(sid))
        (workspace / "stuck.txt").write_bytes(b"stuck")
        hold = _hold(manager, sid)
        real_unlink = os.unlink
        real_remove = getattr(sm, "_remove_tree", None)
        inner = []
        started = []

        def refusing(path, *args, **kwargs):
            if os.path.basename(os.fsdecode(path)) == "stuck.txt":
                raise PermissionError(1, "Operation not permitted", path)
            return real_unlink(path, *args, **kwargs)

        def raced(path):
            # The second destroy runs while the first removes: it ends the session first
            if not started:
                started.append(path)
                inner.append(manager.destroy_sandbox(sid))
            return real_remove(path)

        monkeypatch.setattr(os, "unlink", refusing)
        monkeypatch.setattr(sm, "_remove_tree", raced, raising=False)
        outer = manager.destroy_sandbox(sid)
        monkeypatch.undo()
        recorded = dict(getattr(manager, "_leftovers", {}))
        manager.cleanup_all()
        gone = not os.path.lexists(workspace)
        closed = _closed(hold)
    finally:
        monkeypatch.undo()
        restore()
    assert (inner, outer) == ([True], True), (inner, outer)
    assert list(recorded.values()) == [hold[0]], f"the tree was not recorded once with its hold: {recorded}"
    assert gone and closed, f"the tree's hold was lost to the second destroy: gone={gone} closed={closed}"


def test_lo5_a_swap_never_reaches_the_host(tmp_path, monkeypatch):
    host = tmp_path / "host"
    host.mkdir()
    (host / "keep.txt").write_bytes(b"host")
    os.chmod(host, 0o555)
    base = tmp_path / "workspaces"
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    outcomes = {}
    try:
        manager = ce_t._manager(sm, tmp_path)
        depth = getattr(sm, "_REMOVE_DEPTH", 32)
        for case, seam, levels in (("stat", "stat", 0), ("fstat", "fstat", 0), ("deep", "stat", depth - 1)):
            sid = manager.create_sandbox(f"lo5-{case}").session_id
            workspace = Path(manager.get_workspace_path(sid))
            holder = workspace.joinpath(*(["d"] * levels))
            (holder / "victim").mkdir(parents=True)
            (holder / "victim" / "v.txt").write_bytes(b"v")
            os.chmod(holder / "victim", 0o500)
            victim = os.stat(holder / "victim").st_ino
            real = getattr(os, seam)
            swapped = []

            def seamed(*args, _real=real, _holder=holder, _seam=seam, _swapped=swapped, **kwargs):
                result = _real(*args, **kwargs)
                if _swapped:
                    return result
                # The stat of the folder before its opening, or the check of the descriptor opened on it
                if (_seam == "stat" and args[0] == "victim" and kwargs.get("dir_fd") is not None) or (
                        _seam == "fstat" and stat.S_ISDIR(result.st_mode) and result.st_ino == victim):
                    _swapped.append(_seam)
                    os.rename(_holder / "victim", _holder / "moved")
                    os.symlink(host, _holder / "victim")
                return result

            monkeypatch.setattr(os, seam, seamed)
            try:
                manager.destroy_sandbox(sid)
            finally:
                monkeypatch.undo()
            outcomes[case] = (
                swapped == [seam],
                stat.S_IMODE(os.stat(host).st_mode),
                (host / "keep.txt").exists(),
            )
        manager.cleanup_all()
        swept = sorted(os.listdir(base))
    finally:
        monkeypatch.undo()
        restore()
        _thaw(base)
        os.chmod(host, 0o755)
    assert outcomes == {case: (True, 0o555, True) for case in ("stat", "fstat", "deep")}, (
        f"a folder swapped for a link to the host reached the host: {outcomes}")
    assert swept == [], f"the sweep left what the swaps left: {swept}"


def _chain(workspace, levels):
    """A chain of ``levels`` folders under ``workspace``, a file in each, every level frozen (``0o500``)."""
    chain = [workspace]
    for _ in range(levels):
        chain.append(chain[-1] / "d")
    chain[-1].mkdir(parents=True)
    for level in chain[1:]:
        (level / "f.txt").write_bytes(b"f")
    for level in reversed(chain[1:]):
        os.chmod(level, 0o500)


def test_lo6_a_tree_of_any_depth_goes(tmp_path, monkeypatch):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid = manager.create_sandbox("lo6").session_id
        workspace = Path(manager.get_workspace_path(sid))
        _chain(workspace, 300)
        # Fewer descriptors left to the process than the chain has levels
        opened = len(os.listdir("/proc/self/fd"))
        resource.setrlimit(resource.RLIMIT_NOFILE, (opened + 96, hard))
        try:
            destroyed = manager.destroy_sandbox(sid)
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        left = os.path.lexists(workspace)
        leftovers = dict(getattr(manager, "_leftovers", {}))
        # No folder can be moved up: the walk holds no more descriptors than its depth bound, and the tree stands
        sid = manager.create_sandbox("lo6-unmoved").session_id
        unmoved = Path(manager.get_workspace_path(sid))
        _chain(unmoved, 300)
        baseline = len(os.listdir("/proc/self/fd"))
        held = []

        def refused(*args, **kwargs):
            held.append(len(os.listdir("/proc/self/fd")) - baseline)
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(os, "rename", refused)
        try:
            manager.destroy_sandbox(sid)
        finally:
            monkeypatch.undo()
        stood = os.path.lexists(unmoved)
        manager.cleanup_all()
        unmoved_gone = not os.path.lexists(unmoved)
    finally:
        monkeypatch.undo()
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        restore()
        _thaw(tmp_path / "workspaces")
    depth = getattr(sm, "_REMOVE_DEPTH", 32)
    assert destroyed is True, destroyed
    assert not left and leftovers == {}, "a chain deeper than the descriptors left survived its destroy"
    assert held and max(held) <= depth + 8, (
        f"with no folder moved up the walk held {max(held, default=0)} descriptors past its bound of {depth}")
    assert stood and unmoved_gone, (stood, unmoved_gone)


def _refusing(name, unlink):
    """``os.unlink`` refusing the one entry ``name``: what the owner cannot undo."""
    def refusing(path, *args, **kwargs):
        if os.path.basename(os.fsdecode(path)) == name:
            raise PermissionError(1, "Operation not permitted", path)
        return unlink(path, *args, **kwargs)
    return refusing


def _no_rights(chmod):
    """``os.chmod`` refusing the rights given back through a descriptor: a folder another owner holds."""
    def refused(path, *args, **kwargs):
        if os.fsdecode(path).startswith("/proc/"):
            raise PermissionError(1, "Operation not permitted", path)
        return chmod(path, *args, **kwargs)
    return refused


def test_lo7_wl3_the_hold_lasts_as_long_as_the_session(tmp_path, monkeypatch):
    sm, said, close = wl_t._loaded(tmp_path)
    base = tmp_path / "workspaces"
    try:
        server = wl_t._manager(sm, base)
        session = server.create_sandbox("kept")
        path = Path(session.workspace_path)
        assert wl_t._held_elsewhere(path), "a live workspace is not held"
        fds = wl_t._fds_on(path)
        assert len(fds) == 1, fds
        assert not os.get_inheritable(fds[0]), "a command run in the workspace would inherit its hold"
        assert server.destroy_sandbox("kept")
        assert not path.exists() and wl_t._fds_on(path) == [], "destroying left the tree or the hold"

        # A tree destroying cannot remove whole is said (round 1).
        stuck = Path(server.create_sandbox("stuck").workspace_path)
        (stuck / "sealed").mkdir()
        (stuck / "sealed" / "kept.txt").write_text("x\n", encoding="utf-8")
        monkeypatch.setattr(os, "unlink", _refusing("kept.txt", os.unlink))
        try:
            said.lines.clear()
            said.levels.clear()
            assert server.destroy_sandbox("stuck")
            assert (stuck / "sealed" / "kept.txt").exists()
            assert [line for line in said.at(logging.WARNING) if "could not be removed whole" in line], said.lines
        finally:
            monkeypatch.undo()

        dropped = wl_t._manager(sm, base)
        left = Path(dropped.create_sandbox("dropped").workspace_path)
        (left / "work.py").write_text("x = 1\n", encoding="utf-8")
        assert wl_t._held_elsewhere(left)
        collecting = gc.isenabled()
        gc.disable()
        try:
            del dropped
            assert wl_t._fds_on(left) == [], "a manager dropped with a live session keeps its hold until a collection"
        finally:
            if collecting:
                gc.enable()
        wl_t._manager(sm, base)
        assert not left.exists(), "the next start keeps what a dropped manager left"
    finally:
        close()


def test_lo8_wl4_through_the_real_start(tmp_path, monkeypatch):
    sm, said, close = wl_t._loaded(tmp_path)
    base = tmp_path / "workspaces"
    late = None
    try:
        server = wl_t._manager(sm, base)
        live = server.create_sandbox("server")
        Path(live.workspace_path, "work.py").write_text("print('server')\n", encoding="utf-8")
        shut = base / "sandbox-shut-0000"
        shut.mkdir(mode=0o700)
        (shut / "inside.txt").write_text("kept\n", encoding="utf-8")
        shut.chmod(0)
        stuck = wl_t._orphan(base, "sandbox-stuck-0000")
        (stuck / "sealed").mkdir()
        (stuck / "sealed" / "kept.txt").write_text("x\n", encoding="utf-8")
        config = ("enabled: true\nisolation_backend: tempdir\nrequire_degraded_confirmation: false\n"
                  f"reconcile_on_start: true\nworkspace_base: {base}\n")
        monkeypatch.setattr(os, "unlink", _refusing("kept.txt", os.unlink))
        monkeypatch.setattr(os, "chmod", _no_rights(os.chmod))
        try:
            late, late_said, late_close = wl_t._loaded(tmp_path, config, secrets=True)
        finally:
            monkeypatch.undo()
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
            late_close()
        assert server.destroy_sandbox("server")
    finally:
        close()
