"""Contracts for the panel's reads of a workspace: none follows a link the sandbox put in its workspace.

Host-side reads of a workspace serve the panel: the provision reads the requirements file, the file list walks the
workspace, the preview shows a file, the download sends one, the diff review hashes the workspace and the apply copies
the reviewed files out. Each read by path: the provision and the file list
followed a link the sandbox made with no race at all -- the requirements file a link to a host file, its lines sent
back in the refusal's detail; a link to a host file listed with that file's size and time; a folder swapped for a link
while the list walked, descended into -- and the preview and the download followed a link put after their check. Now
each reaches the workspace's files beneath it through no link: the provision and the download through one descriptor
opened component by component, the preview through the export's reader, the file list by a walk through descriptors
that reads each entry itself.

  * WR1 -- A PROVISION READS ONLY THE WORKSPACE'S REQUIREMENTS FILE: a requirements file that is a link to a host
    file, or lies past a link to a host folder, is not a file in the workspace (the 400 as before), and no host line
    appears in the answer; the refused lines of a requirements file of the workspace come back as before, read with
    universal newlines, through a link inside the workspace too (round 1), and a file past the validator's size bound
    gets the validator's refusal (round 1) -- by its bytes, a CRLF file whose text would fit included (round 2: its
    start alone was validated, its whole handed to pip).
  * WR2 -- THE FILE LIST SHOWS ONLY THE WORKSPACE: links to a host file and to a host folder are not listed, a folder
    swapped for a link to a host folder while the list walks is listed as the workspace's folder, or left when the
    link stays; the list otherwise as ``os.walk`` gave it, its folders now by name: a folder's files by name, then its
    folders' trees, a FIFO listed, sizes and times; 128 folders deep.
  * WR3 -- A PREVIEW SHOWS ONLY THE WORKSPACE'S BYTES: a text or a binary file swapped for a link to a host file at
    each look, right after the check, or once held: never a host byte, and a link met is refused, saying so.
  * WR4 -- A DOWNLOAD SENDS ONLY THE FILE IT OPENED: an approved file swapped for a link right after the check is
    refused; swapped once held, the bytes sent are still the workspace's -- through ``/proc/self/fd``, from the
    descriptor where there is none, and declining a server's offer to send the path later (round 1); the name,
    length and type sent as before, and no body to a HEAD request (round 2); an unapproved path refused as before.
  * WR5 -- WHAT STAYS: a plain download's bytes and headers, nested files listed, previews of text, binary and long
    files, a FIFO answered at once by the preview and the download, an escaping path refused; what changes: a second
    name of a file refused by the preview and the download, saying why (it was served); no descriptor left open by
    any of them (round 1).
  * WR6 -- A REVIEW AND AN APPLY READ ONLY THE WORKSPACE (round 1): the diff review with a file and a folder swapped
    for links at each look hashes and sizes only the workspace's files, and a folder swapped right after the walk's
    look, or one the review may not search, refuses the review -- 409 at the diff route (round 2: an unsearchable
    folder's files read as deleted, a drift was a 500); the apply with the file swapped at each look writes the
    workspace's bytes, never a host file's, refuses a second name saying so, and a file changed while the apply holds
    it is refused, the target left as it was: what is written is what was reviewed, or nothing.

The races are played by the test itself, with 204's seam (``_Racing``). Round 1 found the diff review and the apply
still reading by path, the download reading by path where no ``/proc`` exists, a link inside the workspace refused as
a requirements file, a second name refused without a word, and no contract on descriptors or caps; the cases marked
round 1 and WR6 answer it. Round 2 found the provision validating only the start of a file past its bound, an
unsearchable folder reading as deleted under review, a drift answered with a 500, and refusals naming the wrong
cause; the cases marked round 2 answer it.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402
import test_sandbox_file_tools_follow_no_link_contracts as fw_t  # noqa: E402
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_wr1_a_provision_reads_only_the_workspaces_requirements_file": 5.0,
    "test_wr2_the_file_list_shows_only_the_workspace": 5.0,
    "test_wr3_a_preview_shows_only_the_workspaces_bytes": 5.0,
    "test_wr4_a_download_sends_only_the_file_it_opened": 5.0,
    "test_wr5_what_the_panels_reads_keep": 5.0,
    "test_wr6_a_review_and_an_apply_read_only_the_workspace": 5.0,
}


def _window(tmp_path):
    """The real sandbox routes over the real manager on the tempdir backend, and a closer. The routes' other
    neighbours are left out (their imports are guarded): no authentication, no binding store, no quick sandbox."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *a, **k: sqlite3.connect(path)
    (tmp_path / "share").mkdir()
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.FILE_TOOLS_AVAILABLE = False
    deps.SANDBOX_AVAILABLE = True
    deps.sandbox_manager = None
    loaded, restore = isolate(
        targets={
            "opti_oignon.process_secrets": source("process_secrets.py"),
            "opti_oignon.sandbox_manager": copy,
            "opti_oignon.sandbox_egress": source("sandbox_egress.py"),
            "opti_oignon.sandbox_workspace": source("sandbox_workspace.py"),
            "opti_oignon.api.schemas": source("api", "schemas.py"),
            "opti_oignon.api.routes_sandbox": source("api", "routes_sandbox.py"),
        },
        seeded={
            "opti_oignon.db_utils": db_utils,
            "opti_oignon.config": ce_t._config(tmp_path, ce_t._ON),
            "opti_oignon.api.deps": deps,
        },
        packages=("opti_oignon.api",),
    )
    try:
        sm = loaded["opti_oignon.sandbox_manager"]
        if sm.sandbox_manager is not None:
            raise AssertionError("the import-time singleton must stay off in this window")
        manager = ce_t._manager(sm, tmp_path, host_share_roots=[str(tmp_path / "share")])
        routes = loaded["opti_oignon.api.routes_sandbox"]
        routes.sandbox_manager = manager
    except BaseException:
        restore()
        raise
    return routes, manager, restore


def _workspace(manager, label):
    sid = manager.create_sandbox(None, label=label).session_id
    workspace = Path(manager.get_workspace_path(sid))
    return sid, workspace, os.path.realpath(workspace)


def _refusal(call, *args, **kwargs):
    """(status, detail) of the HTTP refusal ``call`` raised, or ("answered", its answer)."""
    try:
        return "answered", call(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- the contract reads it
        return getattr(exc, "status_code", type(exc).__name__), getattr(exc, "detail", str(exc))


def _sent(response, extensions=None, method="GET"):
    """What a response sends over ASGI: (status, headers, body); a ``pathsend`` message's path counts as its body,
    tagged."""
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "method": method, "path": "/", "headers": [], "query_string": b"",
             "extensions": dict(extensions or {})}
    asyncio.run(response(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in start["headers"]}
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    body += b"".join(b"pathsend:" + os.fsencode(m["path"]) for m in messages if m["type"] == "http.response.pathsend")
    return start["status"], headers, body


def test_wr1_a_provision_reads_only_the_workspaces_requirements_file(tmp_path, monkeypatch):
    host = fw_t._host(tmp_path)
    (host / "reqs").mkdir()
    (host / "reqs" / "requirements.txt").write_bytes(b"host-pinned==1.0\n")
    before = fw_t._tree(host)
    routes, manager, restore = _window(tmp_path)
    try:
        sid, workspace, _real = _workspace(manager, "wr1")
        manager.get_session(sid).network_enabled = True
        routes._eg.network_allowed = lambda: True
        (workspace / "linked.txt").symlink_to(host / "secret.txt")
        (workspace / "reqs").symlink_to(host / "reqs")
        (workspace / "plain.txt").write_bytes(b"pkg\r\nother==1.0\r\n")
        (workspace / "inner").mkdir()
        (workspace / "inner" / "base.txt").write_bytes(b"pkg\n")
        (workspace / "requirements.txt").symlink_to("inner/base.txt")
        (workspace / "huge.txt").write_bytes(b"# x\n" * (routes._eg._REQUIREMENTS_MAX_BYTES // 4 + 1))
        bound = routes._eg._REQUIREMENTS_MAX_BYTES
        (workspace / "crlf.txt").write_bytes(b"# x\r\n" * (bound // 5) + b"--extra-index-url http://elsewhere\n")
        (workspace / "late.txt").write_bytes(b"pkg\n")
        (workspace / "dir").mkdir()
        (workspace / "dir" / "requirements.txt").write_bytes(b"pkg\n")
        user = {"sub": None}

        def provision(rel):
            request = routes.SandboxProvisionRequest(requirements_path=rel)
            return _refusal(routes.provision_workspace, sid, request, current_user=user)

        linked = provision("linked.txt")
        past = provision("reqs/requirements.txt")
        plain = provision("plain.txt")
        inside = provision("requirements.txt")
        huge = provision("huge.txt")
        crlf = provision("crlf.txt")
        late = {}
        for rel, name, aim in (("late.txt", "late.txt", host / "secret.txt"),
                               ("dir/requirements.txt", "dir", host / "reqs")):
            # The link put the moment after the provision resolved the path
            race = fw_t._Racing(monkeypatch, [str(workspace / name)], aim, tmp_path / "aside", looks=("realpath",))
            late[rel] = provision(rel)
            race.settle()
            monkeypatch.undo()
        after = fw_t._tree(host)
        manager.destroy_sandbox(sid)
    finally:
        restore()
    assert after == before
    for name, answer in (("linked", linked), ("past", past), *late.items()):
        assert "host-" not in str(answer[1]), f"a provision showed a host file's lines ({name}): {answer}"
    assert linked == (400, "requirements_path does not name a file in the workspace: linked.txt"), linked
    assert past == (400, "requirements_path does not name a file in the workspace: reqs/requirements.txt"), past
    assert plain[0] == 400, plain
    refused = plain[1]["refused"]
    assert [(r["line"], r["text"]) for r in refused] == [(1, "pkg"), (2, "other==1.0")], refused
    assert inside[0] == 400 and [(r["line"], r["text"]) for r in inside[1]["refused"]] == [(1, "pkg")], (
        f"a requirements file reached through a link inside the workspace was not read: {inside}")
    for rel, answer in late.items():
        assert answer == (400, f"requirements_path does not name a file in the workspace: {rel}"), (rel, answer)
    for answer in (huge, crlf):
        assert answer[0] == 400 and [(r["line"], r["reason"]) for r in answer[1]["refused"]] == [
            (0, "requirements file exceeds the size bound")], (
            f"a requirements file past the size bound was validated in part: {answer}")


def test_wr2_the_file_list_shows_only_the_workspace(tmp_path, monkeypatch):
    host = fw_t._host(tmp_path)
    routes, manager, restore = _window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "wr2")
        (workspace / "a.txt").write_bytes(b"aaa")
        (workspace / "sub").mkdir()
        (workspace / "sub" / "b.txt").write_bytes(b"bb")
        (workspace / "sub2").mkdir()
        (workspace / "sub2" / "c.txt").write_bytes(b"c")
        (workspace / "lnk").symlink_to(host / "secret.txt")
        (workspace / "dlnk").symlink_to(host / "dotssh")
        os.mkfifo(workspace / "p")
        os.utime(workspace / "a.txt", ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "sub2")], host / "dotssh", tmp_path / "aside")
        raced = _refusal(manager.extract_files, sid)
        race.settle()
        monkeypatch.undo()
        listed = _refusal(routes.list_sandbox_files, sid)
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "sub2")], host / "dotssh", tmp_path / "aside",
                            looks=("stat",), stays=True)
        after_look = _refusal(manager.extract_files, sid)
        race.settle()
        monkeypatch.undo()
        chain = workspace / "zdeep"
        for level in range(1, 131):
            chain = chain / "c"
            if level == 127:
                kept = chain
        chain.mkdir(parents=True)
        (kept / "y.txt").write_bytes(b"y")
        (chain / "x.txt").write_bytes(b"x")
        deep = [f["path"] for f in manager.extract_files(sid) if f["path"].startswith("zdeep/")]
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    assert raced[0] == "answered", raced
    rows = [(f["path"], f["size"]) for f in raced[1]]
    assert rows == [("a.txt", 3), ("p", 0), ("sub/b.txt", 2), ("sub2/c.txt", 1)], (
        f"the file list showed a host file or followed a link: {rows}")
    assert raced[1][0]["modified"] == 1_700_000_000.0, raced[1][0]
    assert after_look[0] == "answered", after_look
    assert [(f["path"], f["size"]) for f in after_look[1]] == [("a.txt", 3), ("p", 0), ("sub/b.txt", 2)], (
        f"the file list followed a folder swapped for a link after its look: {after_look[1]}")
    assert deep == ["zdeep/" + "c/" * 127 + "y.txt"], f"the file list walked past 128 folders: {len(deep)}"
    assert listed[0] == "answered", listed
    assert [(e.path, e.size) for e in listed[1].files] == rows, listed[1]


def test_wr3_a_preview_shows_only_the_workspaces_bytes(tmp_path, monkeypatch):
    host = fw_t._host(tmp_path)
    secret = host / "secret.txt"
    routes, manager, restore = _window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "wr3")
        for name in ("v.txt", "w.txt", "h.txt"):
            (workspace / name).write_bytes(f"ws-{name}\n".encode())
        (workspace / "hb.bin").write_bytes(b"\xff\xfews-binary")
        answers = {}
        for name, looks in (("v.txt", None), ("w.txt", ("realpath", "islink")), ("h.txt", ("fstat",)),
                            ("hb.bin", ("fstat",))):
            race = fw_t._Racing(monkeypatch, [os.path.join(real, name)], secret, tmp_path / "aside", looks=looks)
            answers[name] = _refusal(routes.preview_sandbox_file, sid, name)
            race.settle()
            monkeypatch.undo()
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    for name, answer in answers.items():
        assert "host-" not in str(answer[1]), f"a preview showed a host file's bytes ({name}): {answer}"
    assert answers["v.txt"][0] == "answered" and answers["v.txt"][1].content == "ws-v.txt\n", answers["v.txt"]
    assert answers["w.txt"] == (404, "A link stands there, not followed: w.txt"), answers["w.txt"]
    assert answers["h.txt"][0] == "answered" and answers["h.txt"][1].content == "ws-h.txt\n", answers["h.txt"]
    assert answers["hb.bin"][0] == "answered", answers["hb.bin"]
    assert answers["hb.bin"][1].content == b"\xff\xfews-binary".hex(), answers["hb.bin"]


def test_wr4_a_download_sends_only_the_file_it_opened(tmp_path, monkeypatch):
    host = fw_t._host(tmp_path)
    secret = host / "secret.txt"
    routes, manager, restore = _window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "wr4")
        for name in ("d.txt", "e.txt"):
            (workspace / name).write_bytes(f"ws-{name}-bytes".encode())
        assert manager.approve_files(sid, ["d.txt", "e.txt"]) == ["d.txt", "e.txt"]
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "d.txt")], secret, tmp_path / "aside",
                            looks=("realpath", "islink"))
        after_check = _refusal(routes.download_sandbox_file, sid, "d.txt")
        sent_after_check = _sent(after_check[1]) if after_check[0] == "answered" else None
        race.settle()
        monkeypatch.undo()
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "e.txt")], secret, tmp_path / "aside", looks=("fstat",))
        held = _refusal(routes.download_sandbox_file, sid, "e.txt")
        sent_held = _sent(held[1]) if held[0] == "answered" else None
        race.settle()
        monkeypatch.undo()
        (workspace / "n.txt").write_bytes(b"ws-n.txt-bytes")
        (workspace / "q.txt").write_bytes(b"ws-q.txt-bytes")
        assert manager.approve_files(sid, ["n.txt", "q.txt"]) == ["n.txt", "q.txt"]
        real_isdir = os.path.isdir
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "n.txt")], secret, tmp_path / "aside", looks=("fstat",))
        monkeypatch.setattr(os.path, "isdir", lambda p: False if p == "/proc/self/fd" else real_isdir(p))
        no_proc = _refusal(routes.download_sandbox_file, sid, "n.txt")
        sent_no_proc = _sent(no_proc[1]) if no_proc[0] == "answered" else None
        race.settle()
        monkeypatch.undo()
        monkeypatch.setattr(os.path, "isdir", lambda p: False if p == "/proc/self/fd" else real_isdir(p))
        head_no_proc = _sent(routes.download_sandbox_file(sid, "n.txt"), method="HEAD")
        monkeypatch.undo()
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "q.txt")], secret, tmp_path / "aside", looks=("fstat",))
        offered = _refusal(routes.download_sandbox_file, sid, "q.txt")
        sent_offered = _sent(offered[1], {"http.response.pathsend": {}}) if offered[0] == "answered" else None
        race.settle()
        monkeypatch.undo()
        unapproved = _refusal(routes.download_sandbox_file, sid, "a.txt")
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()
    for answer in (sent_after_check, sent_held, sent_no_proc, sent_offered):
        assert answer is None or b"host-" not in answer[2], f"a download sent a host file's bytes: {answer}"
    assert after_check == (404, "A link stands there, not followed: d.txt"), after_check
    assert sent_no_proc is not None and sent_no_proc[:1] + sent_no_proc[2:] == (200, b"ws-n.txt-bytes"), sent_no_proc
    assert sent_no_proc[1]["content-length"] == "14", sent_no_proc[1]
    assert (head_no_proc[0], head_no_proc[1]["content-length"], head_no_proc[2]) == (200, "14", b""), head_no_proc
    assert sent_offered is not None and sent_offered[2] == b"ws-q.txt-bytes", (
        f"a download let the server send the path after the descriptor was closed: {sent_offered}")
    assert held[0] == "answered", held
    status, headers, body = sent_held
    assert (status, body) == (200, b"ws-e.txt-bytes"), (status, body)
    assert headers["content-length"] == str(len(b"ws-e.txt-bytes")), headers
    assert headers["content-disposition"] == 'attachment; filename="e.txt"', headers
    assert headers["content-type"] == "application/octet-stream", headers
    assert unapproved[0] == 403, unapproved


def test_wr5_what_the_panels_reads_keep(tmp_path):
    routes, manager, restore = _window(tmp_path)
    try:
        sid, workspace, real = _workspace(manager, "wr5")
        (workspace / "deep" / "er").mkdir(parents=True)
        (workspace / "deep" / "er" / "n.txt").write_bytes(b"nested")
        (workspace / "t.txt").write_bytes(b"line one\r\nline two\n")
        (workspace / "b.bin").write_bytes(b"\xff\xfe\x00binary")
        (workspace / "big.txt").write_bytes(b"x" * 70000)
        (workspace / "a.txt").write_bytes(b"a-bytes")
        os.link(workspace / "a.txt", workspace / "a2.txt")
        os.mkfifo(workspace / "pipe")
        assert manager.approve_files(sid, ["t.txt", "a2.txt"]) == ["t.txt", "a2.txt"]
        manager.get_session(sid).approved_paths.add("pipe")
        opened = len(os.listdir("/proc/self/fd"))
        listed = [(f["path"], f["size"]) for f in manager.extract_files(sid)]
        text = routes.preview_sandbox_file(sid, "t.txt")
        binary = routes.preview_sandbox_file(sid, "b.bin")
        big = routes.preview_sandbox_file(sid, "big.txt")
        second = _refusal(routes.preview_sandbox_file, sid, "a2.txt")
        released = threading.Event()

        def release():
            # A read that waits on the FIFO is let go every three seconds, never left hanging
            while not released.wait(3.0):
                os.close(os.open(workspace / "pipe", os.O_RDWR | os.O_NONBLOCK))

        releaser = threading.Thread(target=release, daemon=True)
        releaser.start()
        started = time.monotonic()
        pipe_preview = _refusal(routes.preview_sandbox_file, sid, "pipe")
        pipe_download = _refusal(routes.download_sandbox_file, sid, "pipe")
        waited = time.monotonic() - started
        released.set()
        releaser.join()
        sent = _sent(routes.download_sandbox_file(sid, "t.txt"))
        second_download = _refusal(routes.download_sandbox_file, sid, "a2.txt")
        dropped = routes.download_sandbox_file(sid, "t.txt")
        del dropped
        manager.extract_files(sid)
        left_open = len(os.listdir("/proc/self/fd")) - opened
        escaping = _refusal(routes.download_sandbox_file, sid, "../x")
        manager.destroy_sandbox(sid)
    finally:
        restore()
    assert listed == [("a.txt", 7), ("a2.txt", 7), ("b.bin", 9), ("big.txt", 70000), ("pipe", 0), ("t.txt", 19),
                      ("deep/er/n.txt", 6)], listed
    assert (text.content, text.size, text.truncated, text.is_binary) == ("line one\nline two\n", 19, False, False)
    assert (binary.content, binary.size, binary.truncated, binary.is_binary) == (
        b"\xff\xfe\x00binary".hex(), 9, False, True)
    assert (len(big.content), big.size, big.truncated, big.is_binary) == (65536, 70000, True, False)
    assert second == (404, "A second name of a file (a hard link): a2.txt"), second
    assert waited < 1.0, f"a FIFO at a name was waited on: {waited:.1f} s"
    assert pipe_preview == (404, "Not a regular file: pipe"), pipe_preview
    assert sent[0] == 200 and sent[2] == b"line one\r\nline two\n", sent
    assert second_download == (404, "A second name of a file (a hard link): a2.txt"), second_download
    assert pipe_download == (404, "Not a regular file: pipe"), pipe_download
    assert left_open == 0, f"the panel's reads left {left_open} descriptor(s) open"
    assert escaping[0] in (400, 403), escaping


def test_wr6_a_review_and_an_apply_read_only_the_workspace(tmp_path, monkeypatch):
    import hashlib

    host = fw_t._host(tmp_path)
    secret = host / "secret.txt"
    before = fw_t._tree(host)
    routes, manager, restore = _window(tmp_path)
    try:
        ws = routes._ws
        target = tmp_path / "share" / "out"
        target.mkdir()
        sid, workspace, real = _workspace(manager, "wr6")
        (workspace / "f.txt").write_bytes(b"ws-f")
        (workspace / "sub").mkdir()
        (workspace / "sub" / "g.txt").write_bytes(b"ws-g")
        (workspace / "h.txt").write_bytes(b"ws-h")
        (workspace / "k1.txt").write_bytes(b"ws-k")
        os.link(workspace / "k1.txt", workspace / "k2.txt")
        assert manager.approve_files(sid, ["f.txt", "sub/g.txt", "h.txt", "k1.txt"]) == [
            "f.txt", "sub/g.txt", "h.txt", "k1.txt"]
        (workspace / "shut").mkdir()
        (workspace / "shut" / "in").mkdir()
        (workspace / "shut" / "in" / "x.txt").write_bytes(b"x")
        (workspace / "shut" / "y.txt").write_bytes(b"y")
        os.chmod(workspace / "shut", 0o400)
        shut = _refusal(routes._compute_session_diff, sid)
        os.chmod(workspace / "shut", 0o700)
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "f.txt")], secret, tmp_path / "aside")
        race_sub = fw_t._Racing(monkeypatch, [os.path.join(real, "sub")], host / "dotssh", tmp_path / "aside2")
        reviewed = _refusal(ws.generate_workspace_diff, sid, manager=manager)
        race_sub.settle()
        race.settle()
        monkeypatch.undo()
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "sub")], host / "dotssh", tmp_path / "aside",
                            looks=("stat",), stays=True)
        drift = _refusal(ws.generate_workspace_diff, sid, manager=manager)
        race.settle()
        monkeypatch.undo()
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "sub")], host / "dotssh", tmp_path / "aside",
                            looks=("stat",), stays=True)
        drift_route = _refusal(routes._compute_session_diff, sid)
        race.settle()
        monkeypatch.undo()
        diff_hash = ws.generate_workspace_diff(sid, manager=manager).diff_hash
        real_fstat = os.fstat

        def changing(fd):
            # A sandbox command rewrites h.txt the moment the apply holds it open
            st = real_fstat(fd)
            if sys._getframe(1).f_code.co_name == "_open_beneath" and os.readlink(f"/proc/self/fd/{fd}") == str(
                    Path(real) / "h.txt"):
                with open(Path(real) / "h.txt", "r+b") as fh:
                    fh.write(b"XX")
            return st

        # The race played on the apply's own looks at its source, after its review of the workspace
        race = fw_t._Racing(monkeypatch, [os.path.join(real, "f.txt")], secret, tmp_path / "aside",
                            looks=("realpath", "isfile", "lstat"))
        raced = _refusal(ws.apply_workspace_changes, sid, diff_hash, manager=manager, target_dir=str(target))
        race.settle()
        monkeypatch.undo()
        after_race = fw_t._tree(target)
        monkeypatch.setattr(os, "fstat", changing)
        applied = _refusal(ws.apply_workspace_changes, sid, diff_hash, manager=manager, target_dir=str(target))
        monkeypatch.undo()
        written = fw_t._tree(target)
        after = fw_t._tree(host)
        manager.destroy_sandbox(sid)
    finally:
        monkeypatch.undo()
        restore()

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    assert after == before, f"a review or an apply changed a host file: {after}"
    assert raced[0] == "answered", raced
    assert after_race.get("f.txt", ("none", b""))[1] != secret.read_bytes(), (
        f"an apply wrote a host file's bytes through a link put after its check: {after_race}")
    assert reviewed[0] == "answered", reviewed
    entries = {e.path: e.current_hash for e in reviewed[1].entries}
    assert entries == {"f.txt": sha(b"ws-f"), "sub/g.txt": sha(b"ws-g"), "h.txt": sha(b"ws-h"),
                       "k1.txt": sha(b"ws-k"), "k2.txt": sha(b"ws-k"), "shut/in/x.txt": sha(b"x"),
                       "shut/y.txt": sha(b"y")}, (
        f"a review did not hash the workspace's own files, as they are: {entries}")
    assert drift[0] == "WorkspaceReviewDrift", f"a folder swapped for a link under review was walked: {drift}"
    assert drift_route[0] == 409, drift_route
    assert shut == (409, "The workspace could not be reviewed whole: shut/in could not be read (Permission denied); "
                    "review again"), f"a folder the review could not search read as deleted: {shut}"
    assert [(r["path"], r["error"]) for r in raced[1]["refused"]] == [
        ("f.txt", "workspace source invalid: a link stands there, not followed"),
        ("k1.txt", "workspace source invalid: a second name of a file (a hard link)")], raced[1]["refused"]
    assert {a["path"] for a in raced[1]["applied"]} == {"sub/g.txt", "h.txt"}, raced[1]
    assert applied[0] == "answered", applied
    assert {a["path"] for a in applied[1]["applied"]} == {"f.txt", "sub/g.txt"}, applied[1]
    assert [(r["path"], "changed since the reviewed diff" in r["error"]) for r in applied[1]["refused"]] == [
        ("h.txt", True), ("k1.txt", False)], applied[1]["refused"]
    assert {k: v[1] for k, v in written.items() if v[0] == "file"} == {
        "f.txt": b"ws-f", "sub/g.txt": b"ws-g", "h.txt": b"ws-h"}, f"an apply wrote what was not reviewed: {written}"
