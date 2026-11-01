"""Contracts for the agent's file tools: a tool never follows a link the sandbox puts on a checked path.

The file tools -- view, create_file, str_replace -- are what the agents, the sandbox's execute route and the code
executor use inside a workspace. Each checked a path (``validate_sandbox_path``: links inside the workspace resolved,
nothing outside) and then worked on the resolved path BY PATH: a folder test, a size, an ``open`` to read, an
``open(..., "w")`` to write, a listing whose entries were tested through their links. A link the sandbox put on the
path between the check and the use -- a command running at the same time -- was followed: the model's content written
into a host file, a host file's bytes shown to the model, a host folder shown as a folder of the workspace. Now the
resolved path is reached beneath the workspace one component at a time without following a link, the file is opened
once and read, sized and written through that descriptor, and a listing reads each entry itself.

  * FW1 -- A CREATED FILE IS NEVER WRITTEN THROUGH A LINK RACED IN: a link the sandbox puts at a new file's name, at a
    folder's place on its way, or in place of an existing file the moment after any check of it: no host byte
    written, each call says it could not write or lands the file in the workspace; a link put at a new folder's place
    right after the check refuses the file (round 1); a file swapped for a link once the tool holds it open still
    takes the content, in the workspace.
  * FW2 -- AN EDIT READS AND WRITES THE FILE IT OPENED: str_replace with a link raced in at the file's name, whose
    target holds the string or not: no host byte read into the reply or written; with the file swapped for a link
    once the tool holds it open, the exact and the recovered replacement both land in the workspace's file.
  * FW3 -- A VIEW SHOWS ONLY THE WORKSPACE: a file swapped for a link to a host file the moment after any check of it,
    or once the tool holds it open, or right after the check (round 1), and a folder on the way swapped for a link to a
    host folder, at each look or right after the check: the reply holds the workspace's bytes or says the path is not
    there or is a link, never the host's.
  * FW4 -- A LISTING READS EACH ENTRY ITSELF: links to a host folder, a host file, nowhere, and a folder of the
    workspace show as ``[LINK]``, with no size, nothing of what they point at; a listed folder swapped for a link to a
    host folder after its check still lists the workspace's folder.
  * FW5 -- WHAT STAYS: a new file in new folders with ``open``'s modes, a file overwritten in place keeping its mode, a
    second name of a file written in place, a path through a link inside the workspace (a venv's folder, an alias)
    created, viewed and edited as before; the messages, line ranges, size caps, the recovery chain and its hint; an
    edit of a file its owner may not write fails at its write, and its miss still hints; an edit whose text cannot be
    written leaves the file whole and the tool answers (round 1: it raised); a FIFO at a name answered at once by each
    tool (create_file waited on it), and one a reader holds refused, never written into; a link loop named as a link,
    a folder its owner may not read as a folder that cannot be listed, and a file that grows after its size was
    taken refused as too large, never read past the cap (round 1).
  * FW6 -- THE AGENT'S SEARCH TOOLS READ ONLY THE WORKSPACE (round 1): grep on a file and through a folder, ls, glob
    and the Svelte check after a write, with a link put in place of a file or a folder right after the tool's own
    look at it -- 203's checks (``islink``, ``isfile``), the check, the walk's own status read of an entry (round 2:
    the lab called none of 203's), or once the file is held: never a host file's bytes, size or kind in the reply;
    the tools' results otherwise as before; a FIFO in a searched folder counted as skipped, and one named as grep's
    target, never opened (a process waiting to write to it stays waiting); grep searches 128 folders deep and says
    how many it left (round 2: its walk cost the square of a tree's depth).

Round 1 found the agent's search tools (grep, glob, ls) and the Svelte check after a write still reading by path
after their checks, and the contracts never holding a link at view's last open or at a new folder of create_file;
FW6 and the cases marked round 1 answer it. Round 2 found grep's new walk costing the square of a tree's depth, and
FW6 racing on checks the lab no longer made; FW6's cases marked round 2 answer it.

The races are played by the test itself: ``_Racing`` wraps the path functions of ``os`` so that each look at a
watched name finds what the workspace holds there and leaves a link in its place; a tool that checks with any of them
and then works by path follows it, a tool that works through no link never meets it.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import os
import sqlite3
import stat
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_fw1_a_created_file_is_never_written_through_a_link_raced_in": 5.0,
    "test_fw2_an_edit_reads_and_writes_the_file_it_opened": 5.0,
    "test_fw3_a_view_shows_only_the_workspace": 5.0,
    "test_fw4_a_listing_reads_each_entry_itself": 5.0,
    "test_fw5_what_the_file_tools_keep": 5.0,
    "test_fw6_the_agents_search_tools_read_only_the_workspace": 5.0,
}


def _tree(root):
    """Every entry under ``root``: a link's target, a folder's mode, a file's bytes and mode, a special file's kind."""
    seen = {}
    for folder, dirs, files in os.walk(root):
        for path in [Path(folder) / name for name in dirs + files]:
            rel = path.relative_to(root).as_posix()
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode):
                seen[rel] = ("link", os.readlink(path))
            elif stat.S_ISDIR(st.st_mode):
                seen[rel] = ("folder", oct(stat.S_IMODE(st.st_mode)))
            elif stat.S_ISREG(st.st_mode):
                seen[rel] = ("file", path.read_bytes(), oct(stat.S_IMODE(st.st_mode)))
            else:
                # A FIFO is described, never read: a read would wait for a writer
                seen[rel] = ("special", oct(stat.S_IFMT(st.st_mode)))
    return seen


def _host(tmp_path):
    """Host files a sandbox would aim at: keys in a folder of mode 0o700, a file holding the string an edit looks
    for, and a secret."""
    host = tmp_path / "host"
    (host / "dotssh").mkdir(parents=True)
    keys = host / "dotssh" / "authorized_keys"
    keys.write_bytes(b"host-keys TOKEN\n")
    os.chmod(keys, 0o600)
    (host / "secret.txt").write_bytes(b"host-secret line\n")
    os.chmod(host / "dotssh", 0o700)
    return host


class _Racing:
    """The sandbox's code racing the checks a tool makes. Just before a look at one of ``names`` -- or at a path
    beneath one -- what the race put there is taken away: the name holds what the workspace holds, its entry or
    nothing. The moment after, the entry is moved aside (when there is one) and a link to ``aim`` stands at the name.
    A look is a stat, an existence, file, folder or link test, a resolution, an access test, a listing of the name's
    folder, an open that writes nothing (``O_PATH``, read-only) -- by path, beneath a folder's descriptor, or through
    ``/proc/self/fd`` --, a listing of a folder's descriptor, and a status read on a descriptor of the entry
    (``fstat``). An open that writes is the tool's
    act, not a look: the link is left for it to meet. ``looks`` narrows the calls that count: ``("fstat",)`` swaps
    the entry only once the tool holds it open, so a tool that reads or writes by path after that meets the link, and
    one that keeps to its descriptor never does; the check's own calls (``realpath``, ``islink``) leave the link for
    whatever the tool does next. With ``stays``, the race is won once: the link put after the first look stays, and a
    later look at the name -- a second look by path after the tool's own -- meets it."""

    _OS = ("stat", "lstat", "access", "listdir", "scandir", "open", "fstat")
    _PATH = ("exists", "lexists", "isfile", "isdir", "islink", "realpath", "getsize")
    _WRITES = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def __init__(self, monkeypatch, names, aim, aside, looks=None, stays=False):
        self._names = sorted(os.path.abspath(name) for name in names)
        self._aim = str(aim)
        aside = Path(aside)
        aside.mkdir(exist_ok=True)
        self._aside = {name: str(aside / f"{len(list(aside.iterdir()))}-{i}") for i, name in enumerate(self._names)}
        self._lstat, self._unlink, self._symlink = os.lstat, os.unlink, os.symlink
        self._readlink, self._rename = os.readlink, os.rename
        self._put = set()
        self._moved = set()
        self._stays = stays
        for owner, attrs in ((os, self._OS), (os.path, self._PATH)):
            for attr in attrs:
                if looks is None or attr in looks:
                    monkeypatch.setattr(owner, attr, self._racing(attr, getattr(owner, attr)))

    def _racing(self, attr, real):
        def look(*args, **kwargs):
            where = self._watched(attr, args[0] if args else kwargs.get("path", kwargs.get("fd")), kwargs.get("dir_fd"))
            acts = attr == "open" and (args[1] if len(args) > 1 else kwargs.get("flags", 0)) & self._WRITES
            if not acts and not self._stays:
                for name in where & self._put:
                    self._take_back(name)
            try:
                return real(*args, **kwargs)
            finally:
                for name in where - self._put:
                    try:
                        self._lstat(name)
                        self._rename(name, self._aside[name])
                        self._moved.add(name)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        continue
                    self._symlink(self._aim, name)
                    self._put.add(name)
        return look

    def _take_back(self, name):
        self._put.discard(name)
        self._unlink(name)
        if name in self._moved:
            self._moved.discard(name)
            self._rename(self._aside[name], name)

    def settle(self):
        """Every name back to what the workspace holds."""
        for name in list(self._put):
            self._take_back(name)

    def _watched(self, attr, path, dir_fd):
        """The watched names a look at ``path`` (beneath ``dir_fd``) concerns: the name itself or a folder above it,
        the entry a descriptor is open on, or for a listing every watched name of that folder."""
        if attr in ("listdir", "scandir") and path is None:
            path = "."
        if isinstance(path, int):
            if attr not in ("fstat", "listdir", "scandir"):
                return set()
            try:
                path = self._readlink(f"/proc/self/fd/{path}")
            except OSError:
                return set()
        if path is None:
            return set()
        where = os.fsdecode(os.fspath(path))
        if dir_fd is not None and not os.path.isabs(where):
            where = os.path.join(self._readlink(f"/proc/self/fd/{dir_fd}"), where)
        parts = os.path.abspath(where).split(os.sep)
        if parts[1:4] == ["proc", "self", "fd"] and len(parts) > 4:
            try:
                where = os.path.join(self._readlink(os.sep.join(parts[:5])), *parts[5:])
            except OSError:
                return set()
        where = os.path.abspath(where)
        if attr in ("listdir", "scandir"):
            return {name for name in self._names if os.path.dirname(name) == where}
        return {name for name in self._names if where == name or where.startswith(name + os.sep)}


def _answer(call, *args, **kwargs):
    """What a tool answered, or the exception it raised, named: a tool answers, it does not raise."""
    try:
        return call(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- the contract reads it
        return f"raised {exc!r}"


def _window(tmp_path):
    """The real file tools over the real manager on the tempdir backend, and a closer."""
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        ft = sys.modules["opti_oignon.file_tools"]
        manager = ce_t._manager(sm, tmp_path)
    except BaseException:
        restore()
        raise
    return ft, manager, restore


def _workspace(manager, label):
    sid = manager.create_sandbox(None, label=label).session_id
    workspace = Path(manager.get_workspace_path(sid))
    return sid, workspace, os.path.realpath(workspace)


def test_fw1_a_created_file_is_never_written_through_a_link_raced_in(tmp_path, monkeypatch):
    host = _host(tmp_path)
    keys = host / "dotssh" / "authorized_keys"
    before = _tree(host)
    ft, manager, restore = _window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "fw1")
        (workspace / "old.txt").write_bytes(b"fw1-old")
        (workspace / "late.txt").write_bytes(b"fw1-late-before")

        def create(path, content):
            return _answer(ft._handle_sandbox_create_file, sid, path, content, _sandbox_manager=manager)

        race = _Racing(monkeypatch, [os.path.join(real, name) for name in ("new.txt", "deep", "old.txt")], keys,
                       tmp_path / "aside")
        replies = {
            "new.txt": create("new.txt", "fw1-new"),
            "deep/inner.txt": create("deep/inner.txt", "fw1-deep"),
            "old.txt": create("/workspace/old.txt", "fw1-overwrite"),
        }
        race.settle()
        monkeypatch.undo()
        race = _Racing(monkeypatch, [os.path.join(real, "deep2")], host / "dotssh", tmp_path / "aside",
                       looks=("realpath", "islink"))
        after_the_check = create("deep2/inner.txt", "fw1-deep2")
        race.settle()
        monkeypatch.undo()
        race = _Racing(monkeypatch, [os.path.join(real, "late.txt")], keys, tmp_path / "aside", looks=("fstat",))
        late = create("late.txt", "fw1-late")
        race.settle()
        monkeypatch.undo()
        after = _tree(host)
        landed = _tree(workspace)
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert after == before, f"a created file was written through a link the sandbox raced in: {after}"
    for name in ("new.txt", "old.txt"):
        assert replies[name] == (
            f"Error writing file: [Errno 40] a link stands there: not followed: '{os.path.join(real, name)}'"
        ), replies[name]
    deep2 = os.path.join(real, "deep2", "inner.txt")
    assert after_the_check == f"Error writing file: [Errno 20] Not a directory: '{deep2}'", after_the_check
    for path, reply in replies.items():
        content = {"new.txt": b"fw1-new", "deep/inner.txt": b"fw1-deep", "old.txt": b"fw1-overwrite"}[path]
        if reply.startswith("File created:"):
            assert landed[path][:2] == ("file", content), (path, reply, landed.get(path))
        else:
            assert reply.startswith("Error writing file:"), (path, reply)
            assert all(seen[1] != content for seen in landed.values() if seen[0] == "file"), (path, landed)
    assert late == "File created: late.txt (8 bytes)", late
    assert landed["late.txt"][:2] == ("file", b"fw1-late"), f"a file held open took its content elsewhere: {landed}"


def test_fw2_an_edit_reads_and_writes_the_file_it_opened(tmp_path, monkeypatch):
    host = _host(tmp_path)
    keys = host / "dotssh" / "authorized_keys"
    secret = host / "secret.txt"
    before = _tree(host)
    ft, manager, restore = _window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "fw2")
        (workspace / "edit.txt").write_bytes(b"ws-edit TOKEN\n")
        (workspace / "leak.txt").write_bytes(b"ws-leak line\n")
        (workspace / "held.txt").write_bytes(b"ws-held TOKEN\n")
        (workspace / "rec.txt").write_bytes(b"def f():\n    alpha  beta\n    return 1\n")

        def edit(path, old, new):
            return _answer(ft._handle_sandbox_str_replace, sid, path, old, new, _sandbox_manager=manager)

        race = _Racing(monkeypatch, [os.path.join(real, "edit.txt")], keys, tmp_path / "aside")
        raced = edit("edit.txt", "TOKEN", "CHANGED")
        race.settle()
        monkeypatch.undo()
        race = _Racing(monkeypatch, [os.path.join(real, "leak.txt")], secret, tmp_path / "aside")
        leak = edit("leak.txt", "host-secret lime", "x")
        race.settle()
        monkeypatch.undo()
        race = _Racing(monkeypatch, [os.path.join(real, name) for name in ("held.txt", "rec.txt")], keys,
                       tmp_path / "aside", looks=("fstat",))
        held = edit("held.txt", "TOKEN", "CHANGED")
        recovered = edit("rec.txt", "alpha beta", "gamma")
        race.settle()
        monkeypatch.undo()
        after = _tree(host)
        landed = _tree(workspace)
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert after == before, f"an edit wrote through a link the sandbox raced in: {after}"
    for reply in (raced, leak, held, recovered):
        assert "host-" not in reply, f"an edit showed a host file's bytes: {reply}"
    assert raced.startswith("Error") or landed["edit.txt"][1] == b"ws-edit CHANGED\n", (raced, landed["edit.txt"])
    assert leak.startswith("Error"), leak
    assert held == "Replacement successful in held.txt: replaced 5 chars with 7 chars", held
    assert landed["held.txt"][:2] == ("file", b"ws-held CHANGED\n"), f"an edit wrote elsewhere: {landed}"
    assert recovered == "Replaced (matched via whitespace-normalized normalization).", recovered
    assert landed["rec.txt"][:2] == ("file", b"def f():\ngamma\n    return 1\n"), landed["rec.txt"]


def test_fw3_a_view_shows_only_the_workspace(tmp_path, monkeypatch):
    host = _host(tmp_path)
    secret = host / "secret.txt"
    ft, manager, restore = _window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "fw3")
        (workspace / "v.txt").write_bytes(b"ws-v one\nws-v two\n")
        (workspace / "held.txt").write_bytes(b"ws-held\n")
        (workspace / "v3.txt").write_bytes(b"ws-v3\n")
        for name in ("sub", "sub2"):
            (workspace / name).mkdir()
            (workspace / name / "authorized_keys").write_bytes(b"ws-keys\n")

        def view(path, *lines):
            return _answer(ft._handle_sandbox_view, sid, path, *lines, _sandbox_manager=manager)

        race = _Racing(monkeypatch, [os.path.join(real, "v.txt")], secret, tmp_path / "aside")
        raced = view("v.txt")
        ranged = view("v.txt", 2, 2)
        race.settle()
        monkeypatch.undo()
        race = _Racing(monkeypatch, [os.path.join(real, "sub")], host / "dotssh", tmp_path / "aside")
        on_the_way = view("sub/authorized_keys")
        race.settle()
        monkeypatch.undo()
        race = _Racing(monkeypatch, [os.path.join(real, "sub2")], host / "dotssh", tmp_path / "aside",
                       looks=("realpath", "islink"))
        after_the_check = view("sub2/authorized_keys")
        race.settle()
        monkeypatch.undo()
        race = _Racing(monkeypatch, [os.path.join(real, "v3.txt")], secret, tmp_path / "aside",
                       looks=("realpath", "islink"))
        at_the_name = view("v3.txt")
        race.settle()
        monkeypatch.undo()
        race = _Racing(monkeypatch, [os.path.join(real, "held.txt")], secret, tmp_path / "aside", looks=("fstat",))
        held = view("held.txt")
        race.settle()
        monkeypatch.undo()
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    for reply in (raced, ranged, on_the_way, after_the_check, at_the_name, held):
        assert "host-" not in reply, f"a view showed a host file's bytes: {reply}"
    assert raced == "File: v.txt (2 lines total)\n     1\tws-v one\n     2\tws-v two", raced
    assert ranged == "File: v.txt (2 lines total) [showing lines 2-2]\n     2\tws-v two", ranged
    assert on_the_way.startswith("Error") or on_the_way.endswith("\tws-keys"), on_the_way
    assert after_the_check == "Error: Path not found: sub2/authorized_keys", after_the_check
    v3 = os.path.join(real, "v3.txt")
    assert at_the_name == f"Error reading file: [Errno 40] a link stands there: not followed: '{v3}'", at_the_name
    assert held == "File: held.txt (1 lines total)\n     1\tws-held", held


def test_fw4_a_listing_reads_each_entry_itself(tmp_path, monkeypatch):
    host = _host(tmp_path)
    ft, manager, restore = _window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "fw4")
        (workspace / "sub").mkdir()
        (workspace / "a.txt").write_bytes(b"abc")
        (workspace / "hostdir").symlink_to(host / "dotssh")
        (workspace / "hostfile").symlink_to(host / "secret.txt")
        (workspace / "dangling").symlink_to(tmp_path / "nowhere")
        (workspace / "inner").symlink_to(workspace / "sub")
        (workspace / "lst").mkdir()
        (workspace / "lst" / "mine.txt").write_bytes(b"mine")
        (workspace / "empty").mkdir()

        def view(path):
            return _answer(ft._handle_sandbox_view, sid, path, _sandbox_manager=manager)

        listing = view("/workspace")
        empty = view("empty")
        race = _Racing(monkeypatch, [os.path.join(real, "lst")], host / "dotssh", tmp_path / "aside")
        raced = view("lst")
        race.settle()
        monkeypatch.undo()
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert listing.split("\n") == [
        "Directory: /workspace/. (8 entries)",
        "  [FILE] a.txt (3 bytes)",
        "  [LINK] dangling",
        "  [DIR]  empty/",
        "  [LINK] hostdir",
        "  [LINK] hostfile",
        "  [LINK] inner",
        "  [DIR]  lst/",
        "  [DIR]  sub/",
    ], f"a listing looked through a link: {listing}"
    assert empty == "Empty directory: empty", empty
    assert raced == "Directory: /workspace/lst (1 entries)\n  [FILE] mine.txt (4 bytes)", raced


def test_fw5_what_the_file_tools_keep(tmp_path, monkeypatch):
    reference = tmp_path / "reference"
    with open(reference, "w", encoding="utf-8") as fh:
        fh.write("r")
    made_mode = stat.S_IMODE(reference.stat().st_mode)
    (tmp_path / "reference-folder").mkdir()
    folder_mode = stat.S_IMODE((tmp_path / "reference-folder").stat().st_mode)
    ft, manager, restore = _window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "fw5")

        def create(path, content):
            return _answer(ft._handle_sandbox_create_file, sid, path, content, _sandbox_manager=manager)

        def view(path, *lines):
            return _answer(ft._handle_sandbox_view, sid, path, *lines, _sandbox_manager=manager)

        def edit(path, old, new=""):
            return _answer(ft._handle_sandbox_str_replace, sid, path, old, new, _sandbox_manager=manager)

        replies = {"new": create("pkg/deep/mod.py", "x = 1\n")}
        (workspace / "kept.txt").write_bytes(b"kept-before")
        os.chmod(workspace / "kept.txt", 0o640)
        kept_inode = (workspace / "kept.txt").stat().st_ino
        replies["kept"] = create("kept.txt", "kept-after")
        kept_in_place = (workspace / "kept.txt").stat().st_ino == kept_inode
        (workspace / "a.txt").write_bytes(b"a-before")
        os.link(workspace / "a.txt", workspace / "b.txt")
        replies["second"] = create("b.txt", "b-after")
        (workspace / "real-bin").mkdir()
        (workspace / "venv").mkdir()
        (workspace / "venv" / "bin").symlink_to(workspace / "real-bin")
        replies["venv"] = create("venv/bin/tool", "#!/bin/sh\n")
        (workspace / "target.txt").write_bytes(b"one\ntwo\nthree\n")
        (workspace / "alias.txt").symlink_to("target.txt")
        replies["view-alias"] = view("alias.txt", 2, 3)
        replies["edit-alias"] = edit("alias.txt", "two", "TWO")
        replies["create-alias"] = create("alias.txt", "via alias\n")
        (workspace / "s.txt").write_bytes(b"keep me\n")
        replies["surrogate"] = edit("s.txt", "keep", "\ud800")
        replies["create-surrogate"] = create("s2.txt", "\ud800")
        (workspace / "loop").symlink_to("loop")
        replies["view-loop"] = view("loop")
        replies["edit-loop"] = edit("loop", "x", "y")
        (workspace / "noread").mkdir()
        os.chmod(workspace / "noread", 0o300)
        replies["noread"] = view("noread")
        os.chmod(workspace / "noread", 0o700)
        grow = workspace / "grow.txt"
        grow.write_bytes(b"small\n")
        real_fstat = os.fstat

        def growing(fd):
            # A sandbox command appends to the file the moment its size is taken
            st = real_fstat(fd)
            if os.readlink(f"/proc/self/fd/{fd}") == str(Path(real) / "grow.txt") and st.st_size < 100:
                with open(grow, "ab") as fh:
                    fh.write(b"y" * (ft.MAX_VIEW_SIZE + 10))
            return st

        monkeypatch.setattr(os, "fstat", growing)
        replies["grow"] = view("grow.txt")
        monkeypatch.undo()
        (workspace / "e.txt").write_bytes(b"one two\nthree\none two\n")
        replies["twice"] = edit("e.txt", "one two", "x")
        replies["delete"] = edit("e.txt", "three\n")
        replies["empty-old"] = edit("e.txt", "")
        replies["miss"] = edit("e.txt", "one twx", "y")
        (workspace / "i.py").write_bytes(b"class A:\n    def f(self):\n        return 1\n")
        replies["indent"] = edit("i.py", "def f(self):\n    return 1\n", "def g(self):\n    return 2\n")
        (workspace / "ro.txt").write_bytes(b"ro one\n")
        os.chmod(workspace / "ro.txt", 0o444)
        replies["ro"] = edit("ro.txt", "ro one", "ro two")
        replies["ro-miss"] = edit("ro.txt", "ro onx", "ro two")
        (workspace / "big.txt").write_bytes(b"x" * (ft.MAX_VIEW_SIZE + 1))
        replies["big"] = view("big.txt")
        replies["too-much"] = create("huge.txt", "x" * (ft.MAX_CREATE_SIZE + 1))
        (workspace / "folder").mkdir()
        replies["folder"] = create("folder", "x")
        replies["edit-folder"] = edit("folder", "x", "y")
        replies["missing"] = view("missing.txt")
        replies["edit-missing"] = edit("missing.txt", "x", "y")
        replies["outside"] = create("../escape.txt", "x")
        os.mkfifo(workspace / "pipe")
        os.mkfifo(workspace / "read-pipe")
        reader = os.open(workspace / "read-pipe", os.O_RDONLY | os.O_NONBLOCK)
        released = threading.Event()

        def release():
            # A tool that waits on the FIFO, to read or to write, is let go after three seconds, never left hanging
            if not released.wait(3.0):
                os.close(os.open(workspace / "pipe", os.O_RDWR | os.O_NONBLOCK))

        releaser = threading.Thread(target=release, daemon=True)
        releaser.start()
        started = time.monotonic()
        replies["pipe-create"] = create("pipe", "x")
        replies["pipe-edit"] = edit("pipe", "x", "y")
        replies["pipe-view"] = view("pipe")
        waited = time.monotonic() - started
        released.set()
        releaser.join()
        replies["read-pipe"] = create("read-pipe", "x")
        os.close(reader)
        landed = _tree(workspace)
        manager.destroy_sandbox(sid)
    finally:
        restore()
    assert replies["new"] == "File created: pkg/deep/mod.py (6 bytes)", replies["new"]
    assert landed["pkg/deep/mod.py"] == ("file", b"x = 1\n", oct(made_mode)), landed["pkg/deep/mod.py"]
    assert landed["pkg"] == landed["pkg/deep"] == ("folder", oct(folder_mode)), (landed["pkg"], landed["pkg/deep"])
    assert replies["kept"] == "File created: kept.txt (10 bytes)", replies["kept"]
    assert landed["kept.txt"] == ("file", b"kept-after", oct(0o640)), landed["kept.txt"]
    assert kept_in_place, "a file overwritten was replaced, not written in place"
    assert replies["second"] == "File created: b.txt (7 bytes)", replies["second"]
    assert landed["a.txt"][1] == landed["b.txt"][1] == b"b-after", "a second name of a file was not written in place"
    assert replies["venv"] == "File created: venv/bin/tool (10 bytes)", replies["venv"]
    assert landed["venv/bin"] == ("link", str(workspace / "real-bin")), landed["venv/bin"]
    assert landed["real-bin/tool"][1] == b"#!/bin/sh\n", landed.get("real-bin/tool")
    assert replies["view-alias"] == "File: alias.txt (3 lines total) [showing lines 2-3]\n     2\ttwo\n     3\tthree"
    assert replies["edit-alias"] == "Replacement successful in alias.txt: replaced 3 chars with 3 chars"
    assert replies["create-alias"] == "File created: alias.txt (10 bytes)", replies["create-alias"]
    assert landed["alias.txt"] == ("link", "target.txt") and landed["target.txt"][1] == b"via alias\n"
    assert replies["twice"] == (
        "Error: String found 2 times in e.txt. The search string must be unique. "
        "Add more surrounding context to make it unique."
    ), replies["twice"]
    assert replies["delete"] == "Deletion successful in e.txt: removed 6 chars", replies["delete"]
    assert replies["empty-old"] == "Error: old_str cannot be empty", replies["empty-old"]
    assert replies["miss"] == (
        "Error: String not found in e.txt. The file has 16 characters. Make sure the search string matches exactly "
        "(including whitespace and indentation).\nclosest lines: 1: one two; 2: one two"
    ), replies["miss"]
    assert landed["e.txt"][1] == b"one two\none two\n", landed["e.txt"]
    assert replies["indent"] == "Replaced (matched via indentation-flexible normalization).", replies["indent"]
    assert landed["i.py"][1] == b"class A:\n    def g(self):\n        return 2\n", landed["i.py"]
    ro = os.path.join(real, "ro.txt")
    assert replies["ro"] == f"Error writing file: [Errno 13] Permission denied: '{ro}'", replies["ro"]
    assert replies["ro-miss"].endswith("\nclosest lines: 1: ro one"), replies["ro-miss"]
    assert landed["ro.txt"] == ("file", b"ro one\n", oct(0o444)), landed["ro.txt"]
    assert replies["big"] == (
        f"Error: File too large ({ft.MAX_VIEW_SIZE + 1:,} bytes, max {ft.MAX_VIEW_SIZE:,} bytes)"
    ), replies["big"]
    assert replies["too-much"].startswith("Error: Content too large"), replies["too-much"]
    folder = os.path.join(real, "folder")
    assert replies["folder"] == f"Error writing file: [Errno 21] Is a directory: '{folder}'", replies["folder"]
    assert replies["edit-folder"] == "Error: File not found: folder", replies["edit-folder"]
    assert replies["missing"] == "Error: Path not found: missing.txt", replies["missing"]
    assert replies["edit-missing"] == "Error: File not found: missing.txt", replies["edit-missing"]
    assert replies["outside"].startswith("Error: Path rejected: Path traversal detected"), replies["outside"]
    assert not (tmp_path / "workspaces" / "escape.txt").exists()
    assert waited < 1.0, f"a FIFO at a name was waited on: {waited:.1f} s"
    for name, errno_text in (("pipe", "[Errno 6]"), ("read-pipe", "[Errno 22]")):
        reply = replies["pipe-create" if name == "pipe" else name]
        assert reply == f"Error writing file: {errno_text} not a regular file: '{os.path.join(real, name)}'", reply
    assert replies["pipe-edit"] == "Error: File not found: pipe", replies["pipe-edit"]
    assert replies["pipe-view"] == "Error: Path not found: pipe", replies["pipe-view"]
    assert landed["pipe"] == landed["read-pipe"] == ("special", oct(stat.S_IFIFO)), landed["pipe"]
    assert landed["s.txt"][1] == b"keep me\n", f"an edit that cannot be written cut the file: {landed['s.txt']}"
    unencodable = "Error writing file: 'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed"
    assert replies["surrogate"] == replies["create-surrogate"] == unencodable, (replies["surrogate"],
                                                                                replies["create-surrogate"])
    assert "s2.txt" not in landed, landed.get("s2.txt")
    loop = os.path.join(real, "loop")
    for name in ("view-loop", "edit-loop"):
        assert replies[name] == f"Error reading file: [Errno 40] a link stands there: not followed: '{loop}'", (
            name, replies[name])
    noread = os.path.join(real, "noread")
    assert replies["noread"] == f"Error listing directory: [Errno 13] Permission denied: '{noread}'", replies["noread"]
    assert replies["grow"].startswith(f"Error: File too large ({ft.MAX_VIEW_SIZE + 16:,} bytes, "), (
        f"a file that grew after its size was taken was read past the cap: {replies['grow'][:120]}")


def _search_window(tmp_path):
    """The real search tools (``sandbox_tools``) over the real file tools and the real manager on the tempdir backend,
    as ``ce_t._window`` loads the file tools, and a closer."""
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
    return loaded["opti_oignon.sandbox_tools"], manager, restore


def test_fw6_the_agents_search_tools_read_only_the_workspace(tmp_path, monkeypatch):
    host = _host(tmp_path)
    secret = host / "secret.txt"
    (host / "tags.svelte").write_bytes(b"<hostsecrettag>\n")
    before = _tree(host)
    st, manager, restore = _search_window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "fw6")
        tools = st.SandboxToolSession(sandbox_mgr=manager, tool_registry=None)
        tools.attach(sid)
        (workspace / "g").mkdir()
        (workspace / "g" / "a.txt").write_bytes(b"ws-a line\n")
        (workspace / "g" / "b.txt").write_bytes(b"ws-b line\n")
        os.mkfifo(workspace / "g" / "pipe")
        (workspace / "g" / "hostlink").symlink_to(secret)
        (workspace / "one.txt").write_bytes(b"ws-one line\n")
        (workspace / "d").mkdir()
        (workspace / "d" / "f").write_bytes(b"x")
        (workspace / "d" / "e").write_bytes(b"x")
        (workspace / "d" / "sub").mkdir()
        (workspace / "d" / "hostlink").symlink_to(host / "dotssh")
        (workspace / "gl").mkdir()
        (workspace / "gl" / "k.txt").write_bytes(b"k")
        (workspace / "gl" / "m.txt").write_bytes(b"m")
        (workspace / "gl" / "doorway").symlink_to(host)
        chain = workspace / "deep"
        for level in range(1, 131):
            chain = chain / "c"
            if level == 128:
                searched = chain
        chain.mkdir(parents=True)
        (searched / "y.txt").write_bytes(b"deepline\n")
        (chain / "x.txt").write_bytes(b"deepline\n")
        os.mkfifo(workspace / "wpipe")
        writer_in = threading.Event()

        def write():
            # A sandbox process waiting to write to a FIFO, until something opens it to read
            os.close(os.open(workspace / "wpipe", os.O_WRONLY))
            writer_in.set()

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        time.sleep(0.2)
        replies_first = {"target-fifo": _answer(tools.grep, "x", "wpipe")}
        released_by_grep = writer_in.wait(0.3)
        if not released_by_grep:
            os.close(os.open(workspace / "wpipe", os.O_RDONLY | os.O_NONBLOCK))
        writer.join(5.0)
        started = time.monotonic()
        replies_first["deep"] = _answer(tools.grep, "deepline", "deep")
        deep_took = time.monotonic() - started

        def raced(names, aim, looks, call, *args, stays=False, **kwargs):
            race = _Racing(monkeypatch, [os.path.join(real, name) for name in names], aim, tmp_path / "aside",
                           looks=looks, stays=stays)
            try:
                return _answer(call, *args, **kwargs)
            finally:
                race.settle()
                monkeypatch.undo()

        released = threading.Event()

        def release():
            # A search that opens the FIFO is let go every three seconds, never left hanging
            while not released.wait(3.0):
                os.close(os.open(workspace / "g" / "pipe", os.O_RDWR | os.O_NONBLOCK))

        releaser = threading.Thread(target=release, daemon=True)
        releaser.start()
        started = time.monotonic()
        grep_plain = _answer(tools.grep, "line", "g")
        waited = time.monotonic() - started
        replies = {
            "grep-plain": grep_plain,
            "grep-walk": raced(["g/a.txt"], secret, ("islink",), tools.grep, "line", "g"),
            "grep-file": raced(["one.txt"], secret, ("realpath", "islink"), tools.grep, "line", "one.txt"),
            "grep-held": raced(["g/a.txt"], secret, ("fstat",), tools.grep, "line", "g"),
            "grep-stat": raced(["g/a.txt"], secret, ("stat",), tools.grep, "line", "g", stays=True),
            "ls-plain": _answer(tools.ls, "d"),
            "ls": raced(["d/f", "d/e"], secret, ("islink",), tools.ls, "d"),
            "ls-folder": raced(["d/e"], host / "dotssh", ("islink",), tools.ls, "d"),
            "ls-after-check": raced(["d"], host / "dotssh", ("realpath", "islink"), tools.ls, "d"),
            "ls-stat": raced(["d/f"], secret, ("stat",), tools.ls, "d", stays=True),
            "glob-plain": _answer(tools.glob, "*.txt", "gl"),
            "glob": raced(["gl/k.txt"], secret, ("lstat",), tools.glob, "*.txt", "gl"),
            "glob-doorway": _answer(tools.glob, "doorway/*", "gl"),
        }
        released.set()
        releaser.join()
        tools._backend_is_bwrap = lambda: True
        replies["svelte"] = raced(["c.svelte"], host / "tags.svelte", ("isfile",), tools.create_file, "c.svelte",
                                  "<div></div>\n")
        replies["svelte-held"] = raced(["c.svelte"], host / "tags.svelte", ("open",), tools._maybe_diagnostics,
                                       "c.svelte", "File created: c.svelte (12 bytes)")
        landed = _tree(workspace)
        after = _tree(host)
        tools.detach()
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert after == before, f"a search tool changed a host file: {after}"
    for name, reply in replies.items():
        assert "host-" not in reply and "hostsecret" not in reply, f"{name} showed a host file's bytes: {reply}"
    assert waited < 1.0, f"grep opened a FIFO in the folder it searched: {waited:.1f} s"
    for name, reply in replies_first.items():
        assert "host-" not in reply, f"{name} showed a host file's bytes: {reply}"
    assert replies_first["target-fifo"] == "Error: Path not found: wpipe", replies_first["target-fifo"]
    assert not released_by_grep, "grep opened a FIFO named as its target, letting a waiting writer go"
    rel_y = "deep/" + "c/" * 128 + "y.txt"
    assert replies_first["deep"] == (
        f"1 match(es) in 1 file(s) [1 folder(s) deeper than 128 levels not searched]\n{rel_y}:1: deepline"
    ), replies_first["deep"][:200]
    assert deep_took < 1.0, f"grep took {deep_took:.1f} s on a tree 130 folders deep"
    assert replies["grep-plain"] == (
        "2 match(es) in 2 file(s) [1 file(s) skipped: binary or >1 MiB]\ng/a.txt:1: ws-a line\ng/b.txt:1: ws-b line"
    ), replies["grep-plain"]
    assert replies["grep-stat"] == (
        "1 match(es) in 1 file(s) [2 file(s) skipped: binary or >1 MiB]\ng/b.txt:1: ws-b line"
    ), replies["grep-stat"]
    assert replies["grep-walk"] == replies["grep-held"] == replies["grep-plain"], (replies["grep-walk"],
                                                                                   replies["grep-held"])
    assert replies["grep-file"] == "Error: Path not found: one.txt", replies["grep-file"]
    assert replies["ls-plain"] == "dir 0 sub\nfile 1 e\nfile 1 f", replies["ls-plain"]
    assert replies["ls"] == replies["ls-folder"] == replies["ls-stat"] == replies["ls-plain"], (
        replies["ls"], replies["ls-folder"], replies["ls-stat"])
    assert replies["ls-after-check"] == "Error: Path not found: d", replies["ls-after-check"]
    assert replies["glob-plain"].split("\n")[0] == "2 file(s)", replies["glob-plain"]
    assert sorted(replies["glob-plain"].split("\n")[1:]) == ["gl/k.txt", "gl/m.txt"], replies["glob-plain"]
    assert replies["glob"] in ("1 file(s)\ngl/m.txt", replies["glob-plain"]), replies["glob"]
    assert replies["glob-doorway"] == "0 file(s)", replies["glob-doorway"]
    assert replies["svelte"] == "File created: c.svelte (12 bytes)", replies["svelte"]
    assert replies["svelte-held"] == "", f"the Svelte check read a file swapped once held: {replies['svelte-held']}"
    assert landed["c.svelte"][:2] == ("file", b"<div></div>\n"), landed["c.svelte"]
