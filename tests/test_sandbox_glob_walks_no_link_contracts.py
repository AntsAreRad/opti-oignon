"""Contracts for the agent's glob: its pattern walk goes through no link.

The agent's glob checked its base folder beneath the workspace (204) and then walked the pattern with ``Path.glob``,
by path: pathlib's wildcard and literal steps went through a link to a folder, so a link the model made to a host
folder -- ``/`` itself -- made the server walk that host tree, in the agent's thread. Nothing of it reached the model
(each result was checked beneath the workspace), but the walk was a host traversal the model steered. Now the pattern
is walked through descriptors, each folder opened from its parent's without following a link, each entry read itself,
a folder entered only while the pattern can still match below it.

  * GW1 -- A LINK ON THE PATTERN'S WAY IS NEVER WALKED: a link to a host folder under a literal segment, a ``*`` and a
    ``**``, and a folder swapped for one right after the walk's own look at it (round 1): the host folder is never
    opened nor listed (every listing and open recorded by the identity of what it reached), and nothing of it is
    returned.
  * GW2 -- THE SAME ANSWERS AS BEFORE ON A TREE WITHOUT LINKS: pattern by pattern, the reply equals the reply 204's
    glob gave (``Path.glob``, regular files only, newest first then by name): ``*``, ``?``, ``[...]``, literals,
    ``**`` first, in the middle, last, hidden files, nested folders, no match, a leading ``..`` inside the
    workspace, empty segments, a trailing slash (folders only, so no file: round 1), the clamp and its marker. The
    reference is the running Python's pathlib, read as 3.13 reads it: the suite runs on 3.13.
  * GW3 -- LINKS INSIDE THE WORKSPACE ARE NEITHER RETURNED NOR WALKED: a link to a file and a link to a folder of the
    workspace; a FIFO is not returned and never opened.
  * GW4 -- THE REFUSALS AND THE FOLDERS THAT CANNOT BE LISTED: an absolute pattern, a pattern naming nothing but its
    base (round 1), a ``..`` past the pattern's start (also before a trailing slash, round 2) and a NUL (round 2) are
    invalid patterns; a leading ``..`` above the workspace finds nothing; a base that is not a folder is not found; a
    folder that may be searched but not listed finds nothing for a wildcard and what pathlib finds for a literal name
    (round 2: it hid every file below it).
  * GW5 -- THE WALK IS BOUNDED: a pattern with no ``**`` reads no deeper than its segments; a tree deeper than 128
    folders is searched 128 deep and the reply says how many folders it left, in under a second; a pattern of many
    ``**`` costs what one does (round 1).

Round 2 found the folders that may be searched but not listed hiding their files from literal names, and GW5's
timing too close to its bound to tell a blade; the cases marked round 2 answer it. Round 1 found a trailing slash
read as no slash, the walk's only guard against a folder swapped for a link right
after its look left without a contract, GW3's FIFO check unable to fail, and repeated ``**`` costing each its walk;
the cases marked round 1 answer it.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_sandbox_file_tools_follow_no_link_contracts as fw_t  # noqa: E402

BUDGET_S = {
    "test_gw1_a_link_on_the_patterns_way_is_never_walked": 5.0,
    "test_gw2_the_same_answers_as_before_on_a_tree_without_links": 5.0,
    "test_gw3_links_inside_the_workspace_are_neither_returned_nor_walked": 5.0,
    "test_gw4_the_refusals": 5.0,
    "test_gw5_the_walk_is_bounded": 5.0,
}


class _Seen:
    """What the walk lists and opens, by identity (device, inode): ``os.scandir`` and ``os.listdir`` (the folders
    listed), ``os.open`` (the entries opened) -- by path, beneath a descriptor, or on a descriptor -- each recorded
    once the call succeeded, as the entry it reached. pathlib's globbers hold their own ``os.scandir`` (Python 3.13);
    theirs is recorded too."""

    def __init__(self, monkeypatch):
        self.listed = set()
        self.opened = set()
        self._stat, self._fstat = os.stat, os.fstat
        for attr, into in (("scandir", self.listed), ("listdir", self.listed), ("open", self.opened)):
            monkeypatch.setattr(os, attr, self._recording(getattr(os, attr), into))
        for module in [m for name, m in list(sys.modules.items()) if name == "glob" or name.startswith("pathlib")]:
            for cls in [obj for obj in vars(module).values() if isinstance(obj, type)]:
                held = cls.__dict__.get("scandir")
                if isinstance(held, staticmethod):
                    monkeypatch.setattr(cls, "scandir", staticmethod(self._recording(held.__func__, self.listed)))

    def _identity(self, path, dir_fd):
        try:
            if isinstance(path, int):
                st = self._fstat(path)
            else:
                st = self._stat("." if path is None else path, dir_fd=dir_fd)
        except OSError:
            return None
        return st.st_dev, st.st_ino

    def _recording(self, real, into):
        def look(*args, **kwargs):
            result = real(*args, **kwargs)
            identity = self._identity(args[0] if args else kwargs.get("path"), kwargs.get("dir_fd"))
            if identity is not None:
                into.add(identity)
            return result
        return look


def _identities(*roots):
    """The identities of every folder and file under ``roots``, the roots included (a root that is a file or a FIFO
    too)."""
    out = set()
    for root in roots:
        st = os.lstat(root)
        out.add((st.st_dev, st.st_ino))
        for folder, dirs, files in os.walk(root):
            for path in [folder] + [os.path.join(folder, name) for name in dirs + files]:
                st = os.lstat(path)
                out.add((st.st_dev, st.st_ino))
    return out


def _session(tmp_path):
    st, manager, restore = fw_t._search_window(tmp_path)
    try:
        sid, workspace, real = fw_t._workspace(manager, "gw")
        tools = st.SandboxToolSession(sandbox_mgr=manager, tool_registry=None)
        tools.attach(sid)
    except BaseException:
        restore()
        raise
    return tools, manager, sid, workspace, real, restore


def _make(root, rels, start=1_700_000_000):
    """Files under ``root``, each a second newer than the one before."""
    for n, rel in enumerate(rels):
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rel.encode())
        os.utime(path, (start + n, start + n))


def _pathlib_reply(workspace, base, pattern, max_results=200):
    """204's glob, as it answered on a tree without links: ``Path.glob`` from the base, regular files that are not
    links, newest first then by name, the clamp and its marker."""
    entries = []
    for cand in Path(base).glob(pattern):
        if cand.is_symlink() or not cand.is_file():
            continue
        entries.append((os.path.getmtime(cand), os.path.relpath(str(cand), workspace)))
    entries.sort(key=lambda e: (-e[0], e[1]))
    shown = entries[:max_results]
    header = f"{len(shown)} file(s)" + (" [truncated]" if len(entries) > max_results else "")
    return header if not shown else header + "\n" + "\n".join(rel for _m, rel in shown)


def test_gw1_a_link_on_the_patterns_way_is_never_walked(tmp_path, monkeypatch):
    host = fw_t._host(tmp_path)
    (host / "deep" / "deeper").mkdir(parents=True)
    (host / "deep" / "deeper" / "secret.py").write_bytes(b"host")
    tools, manager, sid, workspace, real, restore = _session(tmp_path)
    try:
        _make(workspace, ["mine.py", "sub/also.py"])
        (workspace / "root").symlink_to(host)
        (workspace / "sub" / "door").symlink_to(host / "deep")
        (workspace / "victim" / "inner").mkdir(parents=True)
        (workspace / "victim" / "inner" / "own.py").write_bytes(b"own")
        host_entries = _identities(host)
        seen = _Seen(monkeypatch)
        replies = [tools.glob(pattern) for pattern in ("root/*", "root/deep/*/*.py", "*/*.py", "**/*.py", "*/*/*/*")]
        # A folder swapped for a link to the host the moment after the walk's own look at it, the link staying
        fw_t._Racing(monkeypatch, [os.path.join(real, "victim")], host / "deep", tmp_path / "aside", looks=("stat",),
                     stays=True)
        replies.append(tools.glob("victim/**/*.py"))
        monkeypatch.undo()
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    walked = (seen.listed | seen.opened) & host_entries
    assert walked == set(), f"the glob listed or opened {len(walked)} host entries through a link"
    for reply in replies:
        assert "secret" not in reply and "root/" not in reply and "door" not in reply, reply
    assert replies[5] == "0 file(s)", f"the glob walked a folder swapped for a link after its look: {replies[5]}"
    assert replies[3] == "3 file(s)\nvictim/inner/own.py\nsub/also.py\nmine.py", replies[3]


def test_gw2_the_same_answers_as_before_on_a_tree_without_links(tmp_path):
    if sys.version_info < (3, 13):
        pytest.skip("the reference is pathlib as Python 3.13 reads a pattern")
    tools, manager, sid, workspace, real, restore = _session(tmp_path)
    try:
        _make(workspace, ["t.py", ".h.py", "a/b/x.py", "a/b/y.txt", "a/z.py", "ab.txt", "b.txt", "c/c/c.py",
                          "a/b/c/d/e.md", "q1.txt", "q22.txt", "[x].txt"])
        (workspace / "emptydir" / "inner").mkdir(parents=True)
        cases = [(".", p) for p in ("*", "*.py", "?.txt", "q?.txt", "[ab].txt", "[!a]*.txt", "a/*", "**/*.py",
                                    "**/b/*", "a/**/x.py", "**", "a/**", ".h*", "nope*", "*/*", "a**", "**/*",
                                    "*/**/*.md", "c/**/c.py", "**/c/*.py", "a/*/*", "**/**/*.py", "emptydir/**",
                                    "[[]x].txt", "*/", "**/", "t.py/", "a/", "**//*.py", "a//b/*", "a/./b/*")]
        cases += [("a/b", "../*.py"), ("a/b", "../../*.txt"), ("a", "../c/*/*"), ("a/b", ".."), ("a/b", "../")]
        replies = [(base, pattern, tools.glob(pattern, base), _pathlib_reply(workspace, Path(real) / base, pattern))
                   for base, pattern in cases]
        clamped = (tools.glob("**/*", ".", max_results=3), _pathlib_reply(workspace, Path(real), "**/*", 3))
        manager.destroy_sandbox(sid)
    finally:
        restore()
    differ = [(base, pattern, got, before) for base, pattern, got, before in replies if got != before]
    assert differ == [], f"glob answered differently from before on a tree without links: {differ[:3]}"
    assert clamped[0] == clamped[1] and clamped[0].startswith("3 file(s) [truncated]"), clamped


def test_gw3_links_inside_the_workspace_are_neither_returned_nor_walked(tmp_path, monkeypatch):
    tools, manager, sid, workspace, real, restore = _session(tmp_path)
    try:
        _make(workspace, ["venv/lib/site.py", "real.py"])
        (workspace / "venv" / "lib64").symlink_to("lib")
        (workspace / "alias.py").symlink_to("real.py")
        os.mkfifo(workspace / "pipe.py")
        pipe = _identities(workspace / "pipe.py")
        seen = _Seen(monkeypatch)
        through = tools.glob("venv/*/site.py")
        top = tools.glob("*.py")
        monkeypatch.undo()
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert through == "1 file(s)\nvenv/lib/site.py", f"a link to a folder of the workspace was walked: {through}"
    assert top == "1 file(s)\nreal.py", f"a link or a FIFO was returned: {top}"
    assert not seen.opened & pipe, "the glob opened a FIFO"


def test_gw4_the_refusals(tmp_path):
    tools, manager, sid, workspace, real, restore = _session(tmp_path)
    try:
        _make(workspace, ["a/x.py", "f.txt", "locked/l.py", "xo/sub/deep.py"])
        os.chmod(workspace / "locked", 0o300)
        os.chmod(workspace / "xo", 0o100)
        unlisted = {(base, pattern): (tools.glob(pattern, base), _pathlib_reply(workspace, Path(real) / base, pattern))
                    for base, pattern in (("locked", "*"), ("locked", "l.py"), (".", "*/l.py"), (".", "**/l.py"),
                                          ("xo", "sub/*.py"), ("xo", "*/*.py"))}
        os.chmod(workspace / "locked", 0o700)
        os.chmod(workspace / "xo", 0o700)
        replies = {
            "absolute": tools.glob("/etc/*"),
            "inner-dotdot": tools.glob("a/../*.txt"),
            "above": tools.glob("../*", "a"),
            "far-above": tools.glob("../../*", "a"),
            "not-a-folder": tools.glob("*", "f.txt"),
            "empty": tools.glob(""),
            "dot": [tools.glob(p) for p in (".", "./", "./.")],
            "inner-dotdot-slash": tools.glob("a/../"),
            "nul": tools.glob("a\x00b"),
        }
        manager.destroy_sandbox(sid)
    finally:
        restore()
    assert replies["absolute"] == "Error: invalid glob pattern: Non-relative patterns are unsupported", replies
    assert replies["inner-dotdot"] == "Error: invalid glob pattern: '..' is only read at the pattern's start", replies
    assert replies["above"] == "1 file(s)\nf.txt", replies["above"]
    assert replies["far-above"] == "0 file(s)", replies["far-above"]
    assert replies["not-a-folder"] == "Error: Path not found: f.txt", replies["not-a-folder"]
    assert replies["empty"] == "Error: glob requires a non-empty 'pattern'", replies["empty"]
    assert replies["dot"] == ["Error: invalid glob pattern: Unacceptable pattern: PosixPath('.')"] * 3, replies["dot"]
    assert replies["inner-dotdot-slash"] == replies["inner-dotdot"], replies["inner-dotdot-slash"]
    assert replies["nul"] == "Error: invalid glob pattern: lstat: embedded null character in path", replies["nul"]
    differ = {key: pair for key, pair in unlisted.items() if pair[0] != pair[1]}
    assert differ == {}, f"a folder that may be searched but not listed answered unlike pathlib: {differ}"
    assert unlisted[("locked", "*")][0] == "0 file(s)" and unlisted[("locked", "l.py")][0].startswith("1 file(s)")


def test_gw5_the_walk_is_bounded(tmp_path, monkeypatch):
    tools, manager, sid, workspace, real, restore = _session(tmp_path)
    try:
        _make(workspace, ["top.py", "one/two/three/deep.py"])
        chain = workspace / "deep"
        for level in range(1, 131):
            chain = chain / "c"
            if level == 127:
                kept = chain
        chain.mkdir(parents=True)
        (kept / "y.py").write_bytes(b"y")
        (chain / "x.py").write_bytes(b"x")
        below = _identities(workspace / "one") | _identities(workspace / "deep") - _identities(workspace / "deep" / "c")
        below |= {(os.stat(workspace / "deep" / "c").st_dev, os.stat(workspace / "deep" / "c").st_ino)}
        seen = _Seen(monkeypatch)
        shallow = tools.glob("*.py")
        monkeypatch.undo()
        started = time.monotonic()
        deep = tools.glob("deep/**/*.py")
        deep_took = time.monotonic() - started
        _make(workspace, [f"wide/{n}/f{m}.py" for n in range(20) for m in range(50)])
        started = time.monotonic()
        many = tools.glob("wide/" + "**/" * 100000 + "*.py")
        many_took = time.monotonic() - started
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert not seen.listed & below, "a pattern with no ** listed a folder below its segments"
    assert shallow == "1 file(s)\ntop.py", shallow
    rel_y = "deep/" + "c/" * 127 + "y.py"
    assert deep == f"1 file(s) [1 folder(s) deeper than 128 levels not searched]\n{rel_y}", deep[:200]
    assert deep_took < 1.0, f"a ** over a tree 130 folders deep took {deep_took:.1f} s"
    assert many.startswith("200 file(s) [truncated]") and many_took < 1.0, (
        f"a pattern of many ** took {many_took:.1f} s: {many[:40]}")
