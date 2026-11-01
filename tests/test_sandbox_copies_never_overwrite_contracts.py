"""Contracts for the files copied out of a sandbox: a copy never overwrites a file, nor writes through a link.

Two paths copy a sandbox's files to the host. A code run whose sandbox goes with it copies its new images into the code
output directory (``code_executor._copy_into``), named by the second and the file's name: two runs in the same second
writing the same name shared one copy, and the first run's result named the second run's image. The file manager's
"Approve & download" (``POST /api/sandbox/{id}/copy-out``, ``SandboxManager.copy_out_file``) wrote ``dest/<name>``
with ``shutil.copy2``: an export of the same name -- another session's, another user's, in the one export folder --
was overwritten, and a link at that name was followed. Both now create each copy with ``O_CREAT | O_EXCL |
O_NOFOLLOW`` under a name no file holds, and remove a copy that fails. The route's default destination was relative to
the server's current directory, and refused by its own check wherever the server was not started from the checkout's
root: it is under the export root now.

  * UQ1 -- TWO RUNS IN ONE SECOND KEEP A COPY EACH: through the real executor and sandbox, the second frozen, two runs
    of one conversation writing one name keep two copies, each its own bytes; what follows the first ``_`` is still
    the file's name.
  * UQ2 -- A LINK OR A FILE AT A COPY'S NAME IS LEFT AS IT IS: a link planted at the name a run's copy would take, and a
    file at the next one, are untouched -- the link's target unchanged -- and the copy takes a free name, mode 0o600.
  * UQ3 -- AN EXPORT NEVER OVERWRITES: through the real manager, two exports of one name, a file already in the folder
    and a link planted at an export's name all keep their bytes; each export lands under a free name, which its result
    names, its status copied from the workspace's file.
  * UQ4 -- THE DEFAULT DESTINATION DOES NOT DEPEND ON THE CURRENT DIRECTORY: a copy-out with no destination, the
    server's current directory elsewhere, reaches the manager under the export root, not refused.
  * UQ5 -- A COPY THAT FAILS LEAVES NOTHING: a run's copy and an export that fail while their bytes are written leave no
    file behind.
  * UQ6 -- A FAILED CLEAN-UP KEEPS THE RUN (round 1): a run's copy that fails, and whose overwrite fails too (a full
    disk), is removed as it is; the run keeps its success, its output and its other images.
  * UQ7 -- AN EXPORT READS NO LINK AND CARRIES NO SPECIAL BIT (round 1): an approved file the sandbox replaces by a
    link after the checks, or whose folder it replaces by one, is not exported and the host file not read, nor is a
    file with a second name (a hard link to a host file); a FIFO put in a file's place is refused at once, never
    waited on (round 2); a file under a folder its owner may walk but not list is exported (round 2); an export keeps
    the source's times and its mode without setuid, setgid, sticky or write for anyone but its owner.
  * UQ8 -- EVERY NAME FINDS A FREE ONE (round 1): an export whose name and its hundred counted marks are taken lands
    under a random mark; a name of 255 bytes taken once lands cut to fit; a run's image with a long name is copied,
    its name cut to fit. Round 2: a name is cut where its characters end, never inside one (an accented or CJK name
    stays UTF-8); a suffix longer than a name leaves room for is cut too; a path written with a trailing slash lands
    under its name.
  * UQ9 -- THE FILE MANAGER NAMES WHAT IT WROTE (round 1): "Approve & download" shows the name of each file written,
    read from the result, and says how many were not; no message is built from the count alone.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402
import test_sandbox_rest_confinement_contracts as rc_t  # noqa: E402

BUDGET_S = {
    "test_uq1_two_runs_in_one_second_keep_a_copy_each": 6.0,
    "test_uq2_a_link_or_a_file_at_a_copys_name_is_left_as_it_is": 6.0,
    "test_uq3_an_export_never_overwrites": 5.0,
    "test_uq4_the_default_destination_does_not_depend_on_the_current_directory": 2.0,
    "test_uq5_a_copy_that_fails_leaves_nothing": 6.0,
    "test_uq6_a_failed_clean_up_keeps_the_run": 6.0,
    "test_uq7_an_export_reads_no_link_and_carries_no_special_bit": 5.0,
    "test_uq8_every_name_finds_a_free_one": 6.0,
    "test_uq9_the_file_manager_names_what_it_wrote": 2.0,
}

_SECOND = 1793450000


def _image(words):
    """A run that writes ``words`` to an image file of its workspace."""
    return f"open('chart.png', 'wb').write({words!r})\n"


def _frozen(ce):
    """The executor's clock stopped at one second; monotonic time still runs."""
    ce.time = types.SimpleNamespace(time=lambda: float(_SECOND), monotonic=time.monotonic)


def _folder(tmp_path, conv):
    """The folder a run of ``conv`` copies its images into, under the window's data directory."""
    return tmp_path / "data" / "exec_outputs" / ("c-" + hashlib.sha256(conv.encode("utf-8")).hexdigest()[:32])


def test_uq1_two_runs_in_one_second_keep_a_copy_each(tmp_path):
    sm, ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        _frozen(ce)
        executor = ce.CodeExecutor(sandbox_mgr=ce_t._manager(sm, tmp_path))
        runs = [executor.execute(_image(words), "python", conv_id="conv-uq1") for words in (b"uq1-first", b"uq1-second")]
    finally:
        restore()
    for result in runs:
        assert result.success and len(result.output_files) == 1, (result.output_files, result.error_message)
    first, second = (Path(r.output_files[0]) for r in runs)
    assert first != second, f"two runs in one second share one copy: {first}"
    assert first.read_bytes() == b"uq1-first", "the first run's result names another run's image"
    assert second.read_bytes() == b"uq1-second"
    assert {first.name.split("_", 1)[-1], second.name.split("_", 1)[-1]} == {"chart.png"}, (first.name, second.name)
    assert first.parent == second.parent == _folder(tmp_path, "conv-uq1")


def test_uq2_a_link_or_a_file_at_a_copys_name_is_left_as_it_is(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    secret = host / "uq2-secret.png"
    secret.write_bytes(b"uq2-host-bytes")
    folder = _folder(tmp_path, "conv-uq2")
    folder.mkdir(parents=True)
    link = folder / f"{_SECOND}_chart.png"
    link.symlink_to(secret)
    taken = folder / f"{_SECOND}-1_chart.png"
    taken.write_bytes(b"uq2-earlier-copy")
    sm, ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        _frozen(ce)
        executor = ce.CodeExecutor(sandbox_mgr=ce_t._manager(sm, tmp_path))
        result = executor.execute(_image(b"uq2-run"), "python", conv_id="conv-uq2")
    finally:
        restore()
    assert result.success and len(result.output_files) == 1, (result.output_files, result.error_message)
    assert secret.read_bytes() == b"uq2-host-bytes", "a copy wrote through a link"
    assert link.is_symlink() and os.readlink(link) == str(secret), "the planted link was replaced"
    assert taken.read_bytes() == b"uq2-earlier-copy", "a copy overwrote an earlier one"
    copied = Path(result.output_files[0])
    assert copied.name == f"{_SECOND}-2_chart.png" and copied.read_bytes() == b"uq2-run", copied
    assert stat.S_IMODE(copied.lstat().st_mode) == 0o600, oct(copied.lstat().st_mode)


def _session(sm, tmp_path, files):
    """A real manager on the tempdir backend, a session whose workspace holds ``files`` (name -> bytes, mode)."""
    manager = ce_t._manager(sm, tmp_path)
    sid = manager.create_sandbox(None, label="uq").session_id
    workspace = Path(manager.get_workspace_path(sid))
    for name, (data, mode) in files.items():
        (workspace / name).write_bytes(data)
        os.chmod(workspace / name, mode)
    return manager, sid, workspace


def test_uq3_an_export_never_overwrites(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    secret = host / "uq3-secret.txt"
    secret.write_bytes(b"uq3-host-bytes")
    dest = tmp_path / "exports"
    dest.mkdir()
    (dest / "notes.txt").write_bytes(b"uq3-notes-already-there")
    (dest / "link.txt").symlink_to(secret)
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager, sid, workspace = _session(sm, tmp_path, {
            "report.txt": (b"uq3-report-first", 0o640),
            "notes.txt": (b"uq3-notes-exported", 0o600),
            "link.txt": (b"uq3-link-exported", 0o600),
        })
        manager.approve_files(sid, ["report.txt", "notes.txt", "link.txt"])
        first = manager.copy_out_batch(sid, ["report.txt", "notes.txt", "link.txt"], str(dest))
        (workspace / "report.txt").write_bytes(b"uq3-report-second")
        second = manager.copy_out_batch(sid, ["report.txt"], str(dest))
        manager.destroy_sandbox(sid)
    finally:
        restore()
    written = {r["src_path"]: Path(r["dest_path"]) for r in first}
    written.update({r["src_path"] + " again": Path(r["dest_path"]) for r in second})
    assert {k: (p.name, p.read_bytes()) for k, p in written.items()} == {
        "report.txt": ("report.txt", b"uq3-report-first"),
        "notes.txt": ("notes-1.txt", b"uq3-notes-exported"),
        "link.txt": ("link-1.txt", b"uq3-link-exported"),
        "report.txt again": ("report-1.txt", b"uq3-report-second"),
    }, written
    assert (dest / "notes.txt").read_bytes() == b"uq3-notes-already-there", "an export overwrote a file"
    assert secret.read_bytes() == b"uq3-host-bytes", "an export wrote through a link"
    assert (dest / "link.txt").is_symlink(), "the planted link was replaced"
    assert stat.S_IMODE(written["report.txt"].stat().st_mode) == 0o640, "the export's status was not copied"


def test_uq4_the_default_destination_does_not_depend_on_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = rc_t._load()
    try:
        mgr = rc_t._Manager(workspace="unused")
        rc_t._wire(ctx, mgr)
        ctx.mod.copy_out_sandbox_files("sid-uq4", SimpleNamespace(paths=["a.txt"], dest_dir=None))
    finally:
        ctx.restore()
    assert len(mgr.copy_out_calls) == 1, mgr.copy_out_calls
    _sid, _paths, dest = mgr.copy_out_calls[0]
    assert dest == os.path.realpath(str(rc_t._REPO / "data" / "sandbox_exports")), dest


class _Breaking:
    """``shutil`` as the module sees it, but a copy of the content that fails half-way."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def copyfileobj(src, dst, *a, **k):
        dst.write(src.read(4))
        dst.flush()
        raise OSError("uq5: the disk is full")


def test_uq5_a_copy_that_fails_leaves_nothing(tmp_path):
    dest = tmp_path / "exports"
    sm, ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        ce.shutil = _Breaking(ce.shutil)
        run = ce.CodeExecutor(sandbox_mgr=ce_t._manager(sm, tmp_path)).execute(
            _image(b"uq5-run"), "python", conv_id="conv-uq5")
        manager, sid, _workspace = _session(sm, tmp_path, {"report.txt": (b"uq5-report", 0o600)})
        manager.approve_files(sid, ["report.txt"])
        sm.shutil = _Breaking(sm.shutil)
        exported = manager.copy_out_batch(sid, ["report.txt"], str(dest))
        manager.destroy_sandbox(sid)
    finally:
        restore()
    assert run.success and run.output_files == [], run.output_files
    left = sorted(p.name for p in _folder(tmp_path, "conv-uq5").iterdir())
    assert left == [], f"a run's failed copy stayed: {left}"
    assert exported == [] and sorted(p.name for p in dest.iterdir()) == [], "a failed export stayed"


class _BreakingOnce(_Breaking):
    """``shutil`` whose first copy of a content fails half-way; the later ones copy."""

    def __init__(self, real):
        super().__init__(real)
        self.broken = 0

    def copyfileobj(self, src, dst, *a, **k):
        if self.broken:
            return self._real.copyfileobj(src, dst, *a, **k)
        self.broken += 1
        return _Breaking.copyfileobj(src, dst)


def test_uq6_a_failed_clean_up_keeps_the_run(tmp_path):
    code = "open('a.png', 'wb').write(b'uq6-a')\nopen('b.png', 'wb').write(b'uq6-b')\nprint('uq6 ran')\n"
    sm, ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        ce.shutil = _BreakingOnce(ce.shutil)

        def full_disk(path, said=None):
            raise OSError(errno.ENOSPC, "uq6: no space to overwrite")

        ce._overwrite_and_remove = full_disk
        run = ce.CodeExecutor(sandbox_mgr=ce_t._manager(sm, tmp_path)).execute(code, "python", conv_id="conv-uq6")
    finally:
        restore()
    assert run.success and run.stdout.strip() == "uq6 ran", (run.success, run.error_message, run.stdout)
    assert [Path(p).read_bytes() for p in run.output_files] == [b"uq6-b"], run.output_files
    left = sorted(p.name for p in _folder(tmp_path, "conv-uq6").iterdir())
    assert left == [Path(run.output_files[0]).name], f"the failed copy stayed: {left}"


def test_uq7_an_export_reads_no_link_and_carries_no_special_bit(tmp_path):
    host = tmp_path / "host"
    (host / "sub").mkdir(parents=True)
    secret = host / "secret.txt"
    secret.write_bytes(b"uq7-host-secret")
    (host / "sub" / "report.txt").write_bytes(b"uq7-host-folder-secret")
    dest = tmp_path / "exports"
    swaps = {}
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager, sid, workspace = _session(sm, tmp_path, {
            "plain.txt": (b"uq7-plain", 0o600), "bits.sh": (b"uq7-bits", 0o4777)})
        (workspace / "sub").mkdir()
        (workspace / "sub" / "report.txt").write_bytes(b"uq7-sub")
        os.link(secret, workspace / "hard.txt")
        (workspace / "pipe.txt").write_bytes(b"uq7-pipe")
        (workspace / "locked").mkdir()
        (workspace / "locked" / "inner.txt").write_bytes(b"uq7-inner")
        os.chmod(workspace / "locked", 0o300)
        os.utime(workspace / "bits.sh", ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
        asked = ["plain.txt", "sub/report.txt", "bits.sh", "hard.txt", "pipe.txt", "locked/inner.txt"]
        os.chmod(workspace / "locked", 0o700)
        manager.approve_files(sid, asked)
        os.chmod(workspace / "locked", 0o300)
        checked = sm.validate_sandbox_path

        def check_then_swap(root, path):
            """The sandbox, still running, puts a link in place once the checks are done."""
            out = checked(root, path)
            if path == "plain.txt" and "plain" not in swaps:
                swaps["plain"] = True
                os.unlink(workspace / "plain.txt")
                os.symlink(secret, workspace / "plain.txt")
            if path == "sub/report.txt" and "sub" not in swaps:
                swaps["sub"] = True
                os.rename(workspace / "sub", workspace / "sub-before")
                os.symlink(host / "sub", workspace / "sub")
            return out

        opened = sm._open_beneath

        def fifo_then_open(root, relative):
            """The sandbox puts a FIFO in the file's place after every check, just before the export opens it."""
            if relative == "pipe.txt" and "pipe" not in swaps:
                swaps["pipe"] = True
                os.unlink(workspace / "pipe.txt")
                os.mkfifo(workspace / "pipe.txt")
            return opened(root, relative)

        sm.validate_sandbox_path = check_then_swap
        sm._open_beneath = fifo_then_open
        exported = []
        worker = threading.Thread(target=lambda: exported.extend(manager.copy_out_batch(sid, asked, str(dest))))
        worker.start()
        worker.join(5)
        waited = worker.is_alive()
        if waited:  # a reader waits on the FIFO: a writer lets it go, and the contract fails below
            os.close(os.open(workspace / "pipe.txt", os.O_WRONLY | os.O_NONBLOCK))
            worker.join(5)
        os.chmod(workspace / "locked", 0o700)
        manager.destroy_sandbox(sid)
    finally:
        restore()
    assert not waited, "an export waited on a FIFO put in a file's place"
    assert swaps == {"plain": True, "sub": True, "pipe": True}, swaps
    assert sorted(r["src_path"] for r in exported) == ["bits.sh", "locked/inner.txt"], exported
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["bits.sh", "inner.txt"], f"a file read through a link, or not regular, was exported: {names}"
    assert (dest / "inner.txt").read_bytes() == b"uq7-inner", "a file under a folder that cannot be listed was lost"
    exported_bits = (dest / "bits.sh").stat()
    assert stat.S_IMODE(exported_bits.st_mode) == 0o755, oct(exported_bits.st_mode)
    assert exported_bits.st_mtime_ns == 1_000_000_000_000_000_000, exported_bits.st_mtime_ns


def test_uq8_every_name_finds_a_free_one(tmp_path):
    long_name = "x" * 251 + ".txt"
    wide_name = "a" + "\u65e5" * 83 + ".txt"  # 254 bytes: its "-1" lands inside a three-byte character
    suffix_name = "a." + "x" * 253  # a suffix of 254 bytes: no stem to cut leaves the name too long
    dest = tmp_path / "exports"
    dest.mkdir()
    taken = ["report.txt"] + [f"report-{n}.txt" for n in range(1, 100)] + [long_name, wide_name, suffix_name, "slash.txt"]
    for name in taken:
        (dest / name).write_bytes(b"uq8-already-there")
    image = "y" * 247 + ".png"
    wide_image = "a" + "\u00e9" * 125 + ".png"  # 255 bytes: its stamp's prefix cuts inside a two-byte character
    sm, ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager, sid, _workspace = _session(sm, tmp_path, {
            "report.txt": (b"uq8-report", 0o600), long_name: (b"uq8-long", 0o600), wide_name: (b"uq8-wide", 0o600),
            suffix_name: (b"uq8-suffix", 0o600), "slash.txt": (b"uq8-slash", 0o600)})
        asked = ["report.txt", long_name, wide_name, suffix_name, "slash.txt/"]
        manager.approve_files(sid, asked)
        exported = manager.copy_out_batch(sid, asked, str(dest))
        manager.destroy_sandbox(sid)
        run = ce.CodeExecutor(sandbox_mgr=ce_t._manager(sm, tmp_path)).execute(
            f"open({image!r}, 'wb').write(b'uq8-image')\nopen({wide_image!r}, 'wb').write(b'uq8-wide-image')\n",
            "python", conv_id="conv-uq8")
    finally:
        restore()
    landed = {r["src_path"]: Path(r["dest_path"]) for r in exported}
    assert set(landed) == set(asked), sorted(landed)
    for path in landed.values():
        path.name.encode("utf-8")  # a name cut inside a character holds a lone surrogate, and raises here
        assert len(os.fsencode(path.name)) <= 255, path.name
    assert landed[wide_name].name == "a" + "\u65e5" * 82 + "-1.txt", landed[wide_name].name
    assert landed[wide_name].read_bytes() == b"uq8-wide"
    assert landed[suffix_name].name.startswith("-1.x"), landed[suffix_name].name
    assert landed[suffix_name].read_bytes() == b"uq8-suffix"
    assert (landed["slash.txt/"].name, landed["slash.txt/"].read_bytes()) == ("slash-1.txt", b"uq8-slash")
    assert re.fullmatch(r"report-[0-9a-f]{12}\.txt", landed["report.txt"].name), landed["report.txt"].name
    assert landed["report.txt"].read_bytes() == b"uq8-report"
    cut = landed[long_name].name
    assert len(os.fsencode(cut)) <= 255 and cut.endswith("-1.txt") and cut.startswith("x" * 200), cut
    assert landed[long_name].read_bytes() == b"uq8-long"
    assert run.success and len(run.output_files) == 2, (run.output_files, run.error_message)
    copies = {Path(p).read_bytes(): Path(p) for p in run.output_files}
    assert set(copies) == {b"uq8-image", b"uq8-wide-image"}, sorted(copies)
    for copied in copies.values():
        copied.name.encode("utf-8")
        assert len(os.fsencode(copied.name)) <= 255 and copied.name.endswith(".png"), copied.name
    assert copies[b"uq8-wide-image"].name.split("_", 1)[1] == "a" + "\u00e9" * 119 + ".png", (
        copies[b"uq8-wide-image"].name)


def test_uq9_the_file_manager_names_what_it_wrote():
    text = (rc_t._REPO / "frontend" / "src" / "lib" / "components" / "panels" / "SandboxFileManager.svelte").read_text(
        encoding="utf-8")
    body = text[text.index("async function handleApproveAndDownload"):]
    body = body[:body.index("\n\t}\n")]
    assert re.search(r"result\.copied\.map\(\(c\) => c\.dest_path", body), "the names written are not read"
    assert re.search(r"successMsg = `[^`]*\$\{written\.join\(", body), "the names written are not shown"
    assert re.search(r"if \(written\.length < paths\.length\) \{\s*error = ", body), "a file not written is not said"
    assert "copied_count" not in body, "a message is still built from the count alone"
