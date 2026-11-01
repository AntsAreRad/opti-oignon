"""Contracts for the files copied into a sandbox: a copy in never writes through a link the sandbox put in its workspace.

Four paths copy host files into a workspace. A coding task's project went in with ``shutil.copytree``, merged into what
was there, making folders, writing files and setting status by path: a link the sandbox's code had left at the
project's name, or below it at a name the project holds, sent the copy onto host files. An injected file
(``inject_files``) was written to the name its check resolved, through a link put there after the check, and in place
into a second name of a host file. An upload and a clone opened each file by path after checking that nothing stood
there, so a link put there in between was followed. Now every copy in reaches its folders one component at a time
without following a link, an upload or a clone creates each file only where nothing stands, and an injected file is
written under a fresh name and renamed over its own.

  * IN1 -- A LINK AT AN INJECTED TREE'S DESTINATION IS NEVER WRITTEN THROUGH: through the real manager, a link left at
    the project's name, and links below it at a folder's name and at a file's name: the host's files and folders keep
    their bytes, modes and times, nothing appears among them; the link at the file's name is replaced by the file, the
    rest is copied, and the copy names what it refused.
  * IN2 -- AN INJECTED TREE LANDS AS COPYTREE LANDED IT: nested folders, files' bytes, modes and times, a folder's mode
    and times, a file link of the source copied as its content and a folder link as the folder, a merge that
    overwrites a file and keeps another, a missing ``dest_subdir`` made, a source written with a trailing slash landing
    in the destination itself; the count is the files written; a source written ``x/..`` is refused and nothing lands
    beside the workspace; a link in the source back to a folder above it is refused, the rest copied, where
    ``copytree`` copied the folder into itself until the path failed; a project taken again into a folder its owner may
    not write (copied so from the source) is written in place, as ``copytree`` wrote it (round 1); a folder of the
    workspace injected onto itself keeps its bytes, each file refused (round 2).
  * IN3 -- AN INJECTED FILE REPLACES THE ENTRY AT ITS NAME: a link to a host file at the name, a second name of a host
    file, and a link the sandbox puts at a name the moment after any check of it (round 1: ``_Hostile``): the host
    files keep their bytes and modes; each name is a regular file of one name with the source's bytes, times and mode
    without setuid or setgid; a folder at a name is skipped, and no copy is left beside it; a file of the workspace
    injected onto itself keeps its bytes, refused (round 2); a FIFO at a name is replaced at once, never waited on
    (round 2: a copy that waits is let go after three seconds, so a wait shows as time, never as a hang).
  * IN4 -- AN UPLOAD CREATES EACH FILE WHERE NOTHING STANDS: a link the sandbox puts at an upload's name the moment
    after any check of it, and one put at a name before the upload looks at it, are never followed, the host file
    untouched -- the first refused or never met, the second refused as an existing name; a name already taken is
    refused, the others written with their sizes and digests; a link on the way to the destination folder, even one
    that stays inside the workspace, refuses the upload.
  * IN5 -- A CLONE WRITES THROUGH NO LINK PUT IN ITS FOLDERS WHILE IT RUNS: a link put at a file's name in the folder
    the clone made, and one put at a name the moment after any check of it: the files skipped, the host's bytes and
    modes untouched; a link put at a sub-folder's place refuses the clone as an existing destination, the route's 409
    (round 1: it was a 500; round 2: skipping it said nothing); the clone keeps its rules: links and special files
    skipped and counted, the manifest's digests, the status copied, the destination refused when it is there.

The races are played by the test itself: ``_Hostile`` wraps the path functions of ``os`` (round 2: listings, opens
that write nothing, ``/proc/self/fd`` paths too) so that each look at a watched name finds it empty and leaves a link
there; a copy that checks with any of them and then writes by path follows it (202's code, and the check-then-write
shapes of the book), a copy that writes through no link never meets it.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import hashlib
import io
import os
import stat
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402

BUDGET_S = {
    "test_in1_a_link_at_an_injected_trees_destination_is_never_written_through": 5.0,
    "test_in2_an_injected_tree_lands_as_copytree_landed_it": 5.0,
    "test_in3_an_injected_file_replaces_the_entry_at_its_name": 5.0,
    "test_in4_an_upload_creates_each_file_where_nothing_stands": 5.0,
    "test_in5_a_clone_writes_through_no_link_put_in_its_folders_while_it_runs": 5.0,
}

_TIME_NS = 1_700_000_000_123_456_789


def _tree(root):
    """Every entry under ``root``, ``root`` itself included: a link's target, a folder's mode and time, a file's bytes,
    mode, time and names, a special file's kind."""
    seen = {}
    for folder, dirs, files in os.walk(root):
        for path in [Path(folder)] + [Path(folder) / name for name in dirs + files]:
            rel = path.relative_to(root).as_posix()
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode):
                seen[rel] = ("link", os.readlink(path))
            elif stat.S_ISDIR(st.st_mode):
                seen[rel] = ("folder", oct(stat.S_IMODE(st.st_mode)), st.st_mtime_ns)
            elif stat.S_ISREG(st.st_mode):
                seen[rel] = ("file", path.read_bytes(), oct(stat.S_IMODE(st.st_mode)), st.st_mtime_ns, st.st_nlink)
            else:
                # A FIFO is described, never read: a read would wait for a writer
                seen[rel] = ("special", oct(stat.S_IFMT(st.st_mode)))
    return seen


def _host(tmp_path):
    """Host files a sandbox would aim at: keys of mode 0o600 in a folder of mode 0o700, another file."""
    host = tmp_path / "host"
    (host / "dotssh").mkdir(parents=True)
    keys = host / "dotssh" / "authorized_keys"
    keys.write_bytes(b"in-host-keys")
    os.chmod(keys, 0o600)
    other = host / "other.txt"
    other.write_bytes(b"in-host-other")
    os.chmod(other, 0o644)
    os.chmod(host / "dotssh", 0o700)
    return host


def _raised(call, *args, **kwargs):
    """The exception ``call`` raised, or None."""
    try:
        call(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- the contract reads which one
        return exc
    return None


class _Hostile:
    """The sandbox's code racing the checks a copy makes. Just before the copy looks at one of ``names``, the link put
    there is taken away, so the look finds nothing; the moment after, a link to ``aim`` is put there. A look is a
    stat, an existence, file, folder or link test, a resolution, an access test, a removal, a listing of the name's
    folder, or an open that writes nothing (``O_PATH``, read-only) -- by path, beneath a folder's descriptor, or
    through ``/proc/self/fd``. An open that writes or creates is the copy's act, not a look: the link is left for it
    to meet. So the name holds the link right after the last check of any of these kinds; a copy that writes through
    no link never meets it, nor needs to look."""

    _OS = ("stat", "lstat", "access", "unlink", "remove", "listdir", "scandir", "open")
    _PATH = ("exists", "lexists", "isfile", "isdir", "islink", "realpath", "getsize")
    _WRITES = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def __init__(self, monkeypatch, names, aim):
        self._names = {os.path.abspath(name) for name in names}
        self._aim = str(aim)
        self._lstat, self._unlink, self._symlink, self._readlink = os.lstat, os.unlink, os.symlink, os.readlink
        self._put = set()
        for owner, attrs in ((os, self._OS), (os.path, self._PATH)):
            for attr in attrs:
                monkeypatch.setattr(owner, attr, self._racing(attr, getattr(owner, attr)))

    def _racing(self, attr, real):
        def look(*args, **kwargs):
            where = self._watched(attr, args[0] if args else kwargs.get("path"), kwargs.get("dir_fd"))
            acts = attr == "open" and (args[1] if len(args) > 1 else kwargs.get("flags", 0)) & self._WRITES
            if not acts:
                for name in where & self._put:
                    self._put.discard(name)
                    self._unlink(name)
            try:
                return real(*args, **kwargs)
            finally:
                for name in where - self._put:
                    try:
                        self._lstat(name)
                    except FileNotFoundError:
                        self._symlink(self._aim, name)
                        self._put.add(name)
                    except OSError:
                        pass
        return look

    def _watched(self, attr, path, dir_fd):
        """The watched names a look at ``path`` (beneath ``dir_fd``) concerns: the name itself, or for a listing every
        watched name of that folder."""
        if attr in ("listdir", "scandir") and path is None:
            path = "."
        if path is None or isinstance(path, int):
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
        return {where} & self._names


class _ThenPlant(io.BytesIO):
    """An upload's stream that, once read to its end, lets ``plant`` run: the sandbox's code puts a link at a later
    file's name before anything of the upload has looked at it."""

    def __init__(self, data, plant):
        super().__init__(data)
        self._plant = plant

    def read(self, size=-1):
        chunk = super().read(size)
        if not chunk and self._plant is not None:
            plant, self._plant = self._plant, None
            plant()
        return chunk


def _workspace(manager, label):
    sid = manager.create_sandbox(None, label=label).session_id
    return sid, Path(manager.get_workspace_path(sid))


def test_in1_a_link_at_an_injected_trees_destination_is_never_written_through(tmp_path):
    host = _host(tmp_path)
    project = tmp_path / "src" / "proj"
    (project / "sub").mkdir(parents=True)
    (project / "authorized_keys").write_bytes(b"in1-from-the-project")
    (project / "sub" / "x.txt").write_bytes(b"in1-x")
    (project / "notes.txt").write_bytes(b"in1-notes")
    before = _tree(host)
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        top, ws_top = _workspace(manager, "in1-top")
        (ws_top / "proj").symlink_to(host / "dotssh")
        at_top = _raised(manager.inject_directory, top, str(project))
        below, ws_below = _workspace(manager, "in1-below")
        (ws_below / "proj").mkdir()
        (ws_below / "proj" / "sub").symlink_to(host / "dotssh")
        (ws_below / "proj" / "authorized_keys").symlink_to(host / "dotssh" / "authorized_keys")
        at_below = _raised(manager.inject_directory, below, str(project))
        after = _tree(host)
        landed = _tree(ws_below / "proj")
        manager.destroy_sandbox(top)
        manager.destroy_sandbox(below)
    finally:
        restore()
    assert after == before, f"an injected tree wrote through a link the sandbox left: {after}"
    assert isinstance(at_top, OSError) and str(project) in str(at_top), at_top
    assert isinstance(at_below, OSError) and str(project / "sub") in str(at_below), at_below
    assert "authorized_keys" not in str(at_below), f"a link at a file's name was refused, not replaced: {at_below}"
    assert landed["sub"] == ("link", str(host / "dotssh")), "the link at a folder's place was replaced"
    assert landed["authorized_keys"][:2] == ("file", b"in1-from-the-project"), landed["authorized_keys"]
    assert landed["notes.txt"][:2] == ("file", b"in1-notes"), "the rest of the tree was not copied"


def test_in2_an_injected_tree_lands_as_copytree_landed_it(tmp_path):
    source = tmp_path / "src"
    project = source / "proj"
    (project / "pkg" / "deep").mkdir(parents=True)
    (project / "run.sh").write_bytes(b"#!/bin/sh\necho in2\n")
    os.chmod(project / "run.sh", 0o755)
    (project / "pkg" / "mod.py").write_bytes(b"in2-mod")
    os.chmod(project / "pkg" / "mod.py", 0o640)
    os.utime(project / "pkg" / "mod.py", ns=(_TIME_NS, _TIME_NS))
    (project / "pkg" / "deep" / "data.bin").write_bytes(b"\x00in2-data")
    (project / "linked.txt").symlink_to(project / "pkg" / "mod.py")
    shared = source / "shared"
    shared.mkdir()
    (shared / "lib.txt").write_bytes(b"in2-lib")
    (project / "shared").symlink_to(shared)
    os.chmod(project / "pkg", 0o750)
    os.utime(project / "pkg", ns=(_TIME_NS, _TIME_NS))
    loop = source / "loop"
    loop.mkdir()
    (loop / "f.txt").write_bytes(b"in2-loop")
    (loop / "up").symlink_to(loop)
    frozen = source / "frozen"
    frozen.mkdir()
    (frozen / "r.txt").write_bytes(b"in2-first")
    os.chmod(frozen, 0o555)
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid, workspace = _workspace(manager, "in2")
        (workspace / "work" / "in" / "proj").mkdir(parents=True)
        (workspace / "work" / "in" / "proj" / "run.sh").write_bytes(b"in2-older")
        (workspace / "work" / "in" / "proj" / "keep.txt").write_bytes(b"in2-keep")
        count = manager.inject_directory(sid, str(project), "work/in")
        landed = _tree(workspace / "work" / "in" / "proj")
        slash, ws_slash = _workspace(manager, "in2-slash")
        slash_count = manager.inject_directory(slash, str(project) + os.sep)
        at_root = sorted(p.name for p in ws_slash.iterdir())
        looping, ws_loop = _workspace(manager, "in2-loop")
        looped = _raised(manager.inject_directory, looping, str(loop))
        cycled = _tree(ws_loop / "loop")
        manager.inject_directory(looping, str(frozen))
        (frozen / "r.txt").write_bytes(b"in2-again")
        again = _raised(manager.inject_directory, looping, str(frozen))
        refrozen = _tree(ws_loop / "frozen")
        os.chmod(ws_loop / "frozen", 0o755)
        os.chmod(frozen, 0o755)
        (ws_loop / "own").mkdir()
        (ws_loop / "own" / "o.txt").write_bytes(b"in2-own")
        onto_itself = _raised(manager.inject_directory, looping, str(ws_loop / "own"))
        own = (ws_loop / "own" / "o.txt").read_bytes()
        base = Path(manager.get_workspace_path(sid)).parent
        beside_before = sorted(p.name for p in base.iterdir())
        climbing = _raised(manager.inject_directory, sid, str(project / "pkg" / ".."))
        beside_after = sorted(p.name for p in base.iterdir())
        manager.destroy_sandbox(sid)
        manager.destroy_sandbox(slash)
        manager.destroy_sandbox(looping)
    finally:
        restore()
    assert beside_after == beside_before, f"a source written x/.. landed beside the workspace: {beside_after}"
    assert isinstance(climbing, ValueError), climbing
    files = {rel: entry[1] for rel, entry in landed.items() if entry[0] == "file"}
    assert files == {
        "run.sh": b"#!/bin/sh\necho in2\n",
        "keep.txt": b"in2-keep",
        "linked.txt": b"in2-mod",
        "pkg/mod.py": b"in2-mod",
        "pkg/deep/data.bin": b"\x00in2-data",
        "shared/lib.txt": b"in2-lib",
    }, files
    assert landed["shared"][0] == "folder", "a folder link of the source did not land as the folder"
    assert landed["run.sh"][2] == oct(0o755) and landed["pkg/mod.py"][2:4] == (oct(0o640), _TIME_NS), landed
    assert landed["pkg"][1:] == (oct(0o750), _TIME_NS), f"a folder's status was not copied: {landed['pkg']}"
    assert count == 5, f"the count is not the files written: {count}"
    assert slash_count == 5 and "run.sh" in at_root and "proj" not in at_root, at_root
    assert sorted(cycled) == [".", "f.txt"], f"a link back to a folder above it was copied: {sorted(cycled)}"
    assert isinstance(looped, OSError) and str(loop / "up") in str(looped), looped
    assert again is None, f"a project taken again into a folder its owner may not write failed: {again}"
    assert refrozen["r.txt"][1] == b"in2-again" and refrozen["."][1] == oct(0o555), refrozen
    assert own == b"in2-own", f"a folder injected onto itself lost its bytes: {own}"
    assert isinstance(onto_itself, OSError) and "o.txt" in str(onto_itself), onto_itself


def test_in3_an_injected_file_replaces_the_entry_at_its_name(tmp_path, monkeypatch):
    host = _host(tmp_path)
    keys = host / "dotssh" / "authorized_keys"
    source = tmp_path / "src"
    source.mkdir()
    for name in ("a.txt", "b.txt", "c.txt", "d.txt"):
        (source / name).write_bytes(f"in3-{name}".encode())
        os.chmod(source / name, 0o6640)
        os.utime(source / name, ns=(_TIME_NS, _TIME_NS))
    before = _tree(host)
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid, workspace = _workspace(manager, "in3")
        real = os.path.realpath(workspace)
        (workspace / "a.txt").symlink_to(keys)
        os.link(host / "other.txt", workspace / "b.txt")
        (workspace / "d.txt").mkdir()
        (workspace / "e.txt").write_bytes(b"in3-e-itself")
        os.mkfifo(workspace / "f.txt")
        _Hostile(monkeypatch, [os.path.join(real, name) for name in ("a.txt", "b.txt", "c.txt")], keys)
        injected = manager.inject_files(sid, [str(source / name) for name in ("a.txt", "b.txt", "c.txt", "d.txt")])
        monkeypatch.undo()
        itself = _raised(manager.inject_files, sid, [str(workspace / "e.txt")])
        released = threading.Event()

        def release():
            # A copy that waits on the FIFO is let go after three seconds, never left hanging
            if not released.wait(3.0):
                os.close(os.open(workspace / "f.txt", os.O_RDONLY | os.O_NONBLOCK))

        releaser = threading.Thread(target=release, daemon=True)
        releaser.start()
        (source / "f.txt").write_bytes(b"in3-f")
        started = time.monotonic()
        try:
            fifo = manager.inject_files(sid, [str(source / "f.txt")])
        except OSError as exc:
            fifo = exc
        waited = time.monotonic() - started
        released.set()
        releaser.join()
        after = _tree(host)
        landed = _tree(workspace)
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert after == before, f"an injected file wrote through a link or a second name: {after}"
    assert injected == [os.path.join(real, name) for name in ("a.txt", "b.txt", "c.txt")], injected
    assert landed["e.txt"][:2] == ("file", b"in3-e-itself"), f"a file injected onto itself lost its bytes: {landed}"
    assert itself is None, f"a file injected onto itself was not refused quietly, as the other skips: {itself}"
    assert waited < 1.0 and fifo == [os.path.join(real, "f.txt")], f"a FIFO at a name was waited on: {waited:.1f} s"
    assert landed["f.txt"][:2] == ("file", b"in3-f"), landed["f.txt"]
    for name in ("a.txt", "b.txt", "c.txt"):
        assert landed[name] == ("file", f"in3-{name}".encode(), oct(0o640), _TIME_NS, 1), (name, landed[name])
    assert landed["d.txt"][0] == "folder" and not any(r.startswith("d.txt/") for r in landed), landed
    assert sorted(r for r in landed if "/" not in r) == [".", "a.txt", "b.txt", "c.txt", "d.txt", "e.txt", "f.txt"], (
        sorted(landed))


def test_in4_an_upload_creates_each_file_where_nothing_stands(tmp_path, monkeypatch):
    host = _host(tmp_path)
    keys = host / "dotssh" / "authorized_keys"
    before = _tree(host)
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path)
        sid, workspace = _workspace(manager, "in4")
        folder = Path(os.path.realpath(workspace)) / "up"
        folder.mkdir()
        (folder / "taken.txt").write_bytes(b"in4-taken")
        _Hostile(monkeypatch, [folder / "two.txt"], keys)
        items = [
            ("one.txt", _ThenPlant(b"in4-one", lambda: (folder / "three.txt").symlink_to(keys)), 7),
            ("two.txt", io.BytesIO(b"in4-two"), 7),
            ("three.txt", io.BytesIO(b"in4-three"), 9),
            ("taken.txt", io.BytesIO(b"in4-again"), 9),
        ]
        result = manager.upload_files(sid, items, dest_subdir="up")
        monkeypatch.undo()
        (workspace / "inside").mkdir()
        (workspace / "lnk").symlink_to(workspace / "inside")
        through = _raised(manager.upload_files, sid, [("four.txt", io.BytesIO(b"in4-four"), 8)],
                          dest_subdir="lnk/deeper")
        after = _tree(host)
        landed = _tree(workspace)
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert after == before, f"an upload wrote through a link put at its name: {after}"
    written = {w["name"]: (w["size"], w["sha256"]) for w in result["written"]}
    refused = {r["name"]: r["reason"] for r in result["refused"]}
    # two.txt meets a link the moment after any check of it: refused with the link left, or written where nothing was
    two_written = "two.txt" in written
    assert landed["up/two.txt"][:2] == (("file", b"in4-two") if two_written else ("link", str(keys))), landed
    expected = {"one.txt": (7, hashlib.sha256(b"in4-one").hexdigest())}
    if two_written:
        expected["two.txt"] = (7, hashlib.sha256(b"in4-two").hexdigest())
    assert written == expected, result
    exists = "Destination already exists (not overwritten)"
    assert refused == {"three.txt": exists, "taken.txt": exists, **({} if two_written else {"two.txt": exists})}, refused
    assert landed["up/three.txt"] == ("link", str(keys)) and landed["up/taken.txt"][1] == b"in4-taken", landed
    assert not any(r.startswith("inside/") for r in landed), "an upload went through a link on its way"
    assert isinstance(through, ValueError) and "Invalid destination" in str(through), through


def test_in5_a_clone_writes_through_no_link_put_in_its_folders_while_it_runs(tmp_path, monkeypatch):
    host = _host(tmp_path)
    keys = host / "dotssh" / "authorized_keys"
    project = tmp_path / "share" / "proj"
    (project / "sub").mkdir(parents=True)
    (project / "a.txt").write_bytes(b"in5-a")
    (project / "d.txt").write_bytes(b"in5-d")
    (project / "sub" / "b.txt").write_bytes(b"in5-b")
    second = tmp_path / "share" / "proj2"
    (second / "sub2").mkdir(parents=True)
    (second / "t.txt").write_bytes(b"in5-t")
    (second / "sub2" / "c.txt").write_bytes(b"in5-c")
    os.chmod(project / "sub" / "b.txt", 0o604)
    os.utime(project / "sub" / "b.txt", ns=(_TIME_NS, _TIME_NS))
    (project / "ln").symlink_to(project / "a.txt")
    os.mkfifo(project / "pipe")
    before = _tree(host)
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path, host_share_roots=[str(tmp_path / "share")])
        sid, workspace = _workspace(manager, "in5")
        dest = Path(os.path.realpath(workspace)) / "proj"
        made = hashlib.sha256
        calls = []

        def digest_then_plant(*args, **kwargs):
            # A clone's first file of its top folder, before its sub-folders: the sandbox's code puts a link at
            # a.txt in the folder the clone just made; in the second clone, one at sub2's place
            calls.append(1)
            if len(calls) == 1:
                (dest / "a.txt").symlink_to(keys)
            elif (dest.parent / "proj2").is_dir() and not os.path.lexists(dest.parent / "proj2" / "sub2"):
                (dest.parent / "proj2" / "sub2").symlink_to(host / "dotssh")
            return made(*args, **kwargs)

        monkeypatch.setattr(hashlib, "sha256", digest_then_plant)
        _Hostile(monkeypatch, [dest / "d.txt"], keys)
        result = manager.clone_directory(sid, str(project))
        replaced = _raised(manager.clone_directory, sid, str(second))
        monkeypatch.undo()
        again = _raised(manager.clone_directory, sid, str(project))
        after = _tree(host)
        landed = _tree(dest)
        at_sub2 = _tree(dest.parent / "proj2").get("sub2")
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert after == before, f"a clone wrote through a link put in its folder: {after}"
    assert landed["a.txt"] == ("link", str(keys)), "the planted link was replaced or followed"
    assert isinstance(replaced, FileExistsError) and "sub2" in str(replaced), (
        f"a clone that met a link at a folder's place was not refused: {replaced}")
    assert at_sub2 == ("link", str(host / "dotssh")), f"the link at sub2's place was replaced: {at_sub2}"
    # d.txt meets a link the moment after any check of it: skipped with the link left, or copied where nothing was
    d_copied = landed["d.txt"][0] == "file"
    assert landed["d.txt"][:2] == (("file", b"in5-d") if d_copied else ("link", str(keys))), landed["d.txt"]
    assert {k: result[k] for k in ("dest", "copied_files", "skipped_symlinks", "skipped_special")} == {
        "dest": "proj", "copied_files": 1 + d_copied, "skipped_symlinks": 1, "skipped_special": 1,
    }, result
    manifest = {"proj/sub/b.txt": hashlib.sha256(b"in5-b").hexdigest()}
    if d_copied:
        manifest["proj/d.txt"] = hashlib.sha256(b"in5-d").hexdigest()
    assert result["manifest"] == manifest, result["manifest"]
    assert landed["sub/b.txt"] == ("file", b"in5-b", oct(0o604), _TIME_NS, 1), landed["sub/b.txt"]
    assert isinstance(again, FileExistsError), f"a clone over its own destination was not refused: {again}"
