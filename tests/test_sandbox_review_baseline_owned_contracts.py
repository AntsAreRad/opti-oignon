"""Contracts for the review's baseline: it belongs to the workspace it was recorded for, never to a session made again under the id.

The review's baseline (``sandbox_workspace.WorkspaceManifests``: the hashes recorded at copy-in, the write-once
cloned root and its mount) was kept by session id alone, and only the destroy route forgot it: the idle sweep,
``cleanup_all`` and an explicit destroy by the manager left it. A session made again under the id of one so ended --
an id the create route takes from the request, and the chat's coding agent reuses after every sweep -- inherited it:
its review listed the old workspace's files as deleted, its apply defaulted onto the old clone's host folder, and a
deletion the user confirmed from that review removed host files the new workspace never held. Now a baseline is
recorded for the workspace the copy wrote (each session made has its own folder) and read for the workspace the
review holds.

  * OB1 -- A SESSION MADE AGAIN UNDER A SWEPT ONE'S ID INHERITS NOTHING: a clone recorded as the route records it,
    the session swept, another made under its id: its review has no baseline, lists none of the old files and has no
    cloned root nor mount; with the old files' deletions confirmed, its apply with no target is refused (no implicit
    target) and the host keeps them (round 1: in that order). The old session's own review saw its baseline (the
    control).
  * OB2 -- WHATEVER ENDED THE OLD ONE: the same after ``cleanup_all`` and after the manager's explicit destroy; the
    store's ``drop`` (the destroy route's) forgets every workspace of the id.
  * OB3 -- A LATE RECORD: a record for a workspace whose session ended, coming after another session was made under
    the id and recorded its own (a copy whose session was destroyed and made again before the route recorded it):
    the new session's review reads only its own baseline; the late entries never evict the new ones (that folder
    stands), and they are dropped at the id's next record (their folder is gone), unlike those of a folder the store
    cannot look at (kept: not seen gone; round 1).
  * OB4 -- THE ROUTES RECORD AND READ BY WORKSPACE (round 1), through the routes module under the REST suite's
    stand-in package: the clone and the upload record for the workspace of the session they checked, also when
    another is made under the id while the copy runs (round 2: never re-read by id), and the listing says a session
    has a cloned baseline only when its own workspace has one.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_sandbox_host_writes_in_use_contracts as hu_t  # noqa: E402
import test_sandbox_rest_confinement_contracts as rc_t  # noqa: E402

BUDGET_S = {
    "test_ob1_a_session_made_again_under_a_swept_ones_id_inherits_nothing": 5.0,
    "test_ob2_whatever_ended_the_old_one": 5.0,
    "test_ob3_a_late_record": 5.0,
    "test_ob4_the_routes_record_and_read_by_workspace": 5.0,
}

_window = hu_t._window
_refusal = hu_t._refusal
_idle = hu_t._idle


def _share(tmp_path):
    """A shared folder to clone, two files in it: (share, proj)."""
    share = tmp_path / "share"
    proj = share / "proj"
    proj.mkdir(parents=True)
    (proj / "keep1.txt").write_bytes(b"host-1")
    (proj / "keep2.txt").write_bytes(b"host-2")
    return share, proj


def _cloned(ws, manager, sid, proj):
    """A session under ``sid`` with ``proj`` cloned in, its baseline recorded as the clone route records it."""
    session = manager.create_sandbox(sid, label=sid)
    result = manager.clone_directory(sid, str(proj))
    ws.get_workspace_manifests().record(
        sid, result["manifest"], cloned_root=result["cloned_root"], cloned_mount=result["dest"],
        workspace=session.workspace_path)
    return session


def _seen(ws, manager, sid):
    """The review of ``sid``: (baseline present, (path, kind) each, cloned root, cloned mount), or what it raised."""
    answer = _refusal(ws.generate_workspace_diff, sid, manager=manager)
    if answer[0] != "answered":
        return answer
    diff = answer[1]
    return (diff.baseline_present, [(c.path, c.kind) for c in diff.entries], diff.cloned_root, diff.cloned_mount)


def test_ob1_a_session_made_again_under_a_swept_ones_id_inherits_nothing(tmp_path):
    _sm, ws, manager, restore = _window(tmp_path)
    _share_dir, proj = _share(tmp_path)
    try:
        sid = "ob1-reused"
        _cloned(ws, manager, sid, proj)
        before = _seen(ws, manager, sid)
        _idle(manager, sid)
        manager.list_sessions()
        swept = manager.get_session(sid) is None
        again = manager.create_sandbox(sid, label="ob1-made-again")
        after = _seen(ws, manager, sid)
        # The user confirms the deletions an inherited review would show, then applies with no target
        manager.confirm_deletions(sid, ["proj/keep1.txt", "proj/keep2.txt"])
        no_target = _refusal(ws.apply_workspace_changes, sid, ws.generate_workspace_diff(
            sid, manager=manager).diff_hash, manager=manager)
        out = tmp_path / "share" / "out"
        out.mkdir()
        applied = _refusal(ws.apply_workspace_changes, sid, ws.generate_workspace_diff(
            sid, manager=manager).diff_hash, manager=manager, target_dir=str(out))
        listed = ws.get_workspace_manifests().get_cloned_root(sid, again.workspace_path)
        host = sorted(p.name for p in proj.iterdir())
    finally:
        restore()
    assert before[0] is True and before[1] == [] and before[2] == str(proj) and before[3] == "proj", (
        f"control: the cloned session's own review did not see its baseline: {before}")
    assert swept, "control: the idle session was not swept"
    assert after == (False, [], None, None), (
        f"a session made again under a swept one's id inherited its baseline: {after}")
    assert no_target[0] == "WorkspaceApplyTargetError", (
        f"the apply of a session made again under a swept one's id had an implicit target: {no_target}")
    assert applied[0] == "answered" and applied[1]["deleted"] == [], (
        f"the apply of a session made again under a swept one's id deleted host files: {applied}")
    assert listed is None, "the listing says a session made again under a swept one's id has a cloned baseline"
    assert host == ["keep1.txt", "keep2.txt"], f"host files were removed: {host}"


def test_ob2_whatever_ended_the_old_one(tmp_path):
    _sm, ws, manager, restore = _window(tmp_path)
    _share_dir, proj = _share(tmp_path)
    seen = {}
    try:
        for how in ("cleanup_all", "destroy"):
            sid = f"ob2-{how}"
            _cloned(ws, manager, sid, proj)
            if how == "cleanup_all":
                manager.cleanup_all()
            else:
                manager.destroy_sandbox(sid)
            gone = manager.get_session(sid) is None
            manager.create_sandbox(sid, label=f"{sid}-again")
            seen[how] = (gone, _seen(ws, manager, sid))
            manager.destroy_sandbox(sid)
        # The destroy route's drop: every workspace of the id (an ended one's recorded late, as in OB3)
        store = ws.get_workspace_manifests()
        first = manager.create_sandbox("ob2-drop", label="ob2-drop")
        manager.destroy_sandbox("ob2-drop")
        second = manager.create_sandbox("ob2-drop", label="ob2-drop-again")
        store.record("ob2-drop", {"b.txt": "1" * 64}, cloned_root=str(proj), cloned_mount="proj",
                     workspace=second.workspace_path)
        store.record("ob2-drop", {"a.txt": "0" * 64}, workspace=first.workspace_path)
        recorded = store.snapshot().get("ob2-drop")
        store.drop("ob2-drop")
        dropped = (store.snapshot().get("ob2-drop"), store.get_manifest("ob2-drop", second.workspace_path),
                   store.get_cloned_root("ob2-drop", second.workspace_path),
                   store.get_cloned_mount("ob2-drop", second.workspace_path))
    finally:
        restore()
    for how, (gone, after) in seen.items():
        assert gone, f"control: {how} did not end the session"
        assert after == (False, [], None, None), (
            f"{how}: a session made again under the id inherited the old one's baseline: {after}")
    assert recorded == 2, f"control: the two workspaces' entries were not both recorded: {recorded}"
    assert dropped == (None, None, None, None), f"the destroy route's drop left a baseline of the id: {dropped}"


def test_ob3_a_late_record(tmp_path):
    _sm, ws, manager, restore = _window(tmp_path)
    _share_dir, proj = _share(tmp_path)
    try:
        store = ws.get_workspace_manifests()
        sid = "ob3-late"
        old = manager.create_sandbox(sid, label="ob3-old")
        manager.destroy_sandbox(sid)
        new = manager.create_sandbox(sid, label="ob3-new")
        (Path(new.workspace_path) / "own.txt").write_bytes(b"own")
        store.record(sid, {"own.txt": ws.manifest_hash_file(str(Path(new.workspace_path) / "own.txt"))},
                     workspace=new.workspace_path)
        # The old copy's record, late
        store.record(sid, {"late.txt": "2" * 64}, cloned_root=str(proj), cloned_mount="proj",
                     workspace=old.workspace_path)
        after_late = (_seen(ws, manager, sid), store.snapshot().get(sid))
        # The id's next record: the late entries' folder is gone
        (Path(new.workspace_path) / "more.txt").write_bytes(b"more")
        store.record(sid, {"more.txt": ws.manifest_hash_file(str(Path(new.workspace_path) / "more.txt"))},
                     workspace=new.workspace_path)
        after_next = (store.snapshot().get(sid), store.get_manifest(sid, old.workspace_path),
                      store.get_cloned_root(sid, old.workspace_path))
        # A late record for a folder the store cannot look at (its parent a link to itself: ELOOP, whoever runs
        # this): kept at the id's next record
        loop = tmp_path / "loop"
        os.symlink("loop", loop)
        store.record(sid, {"unseen.txt": "3" * 64}, workspace=str(loop / "sandbox-unseen"))
        store.record(sid, {"more.txt": ws.manifest_hash_file(str(Path(new.workspace_path) / "more.txt"))},
                     workspace=new.workspace_path)
        unseen = store.get_manifest(sid, str(loop / "sandbox-unseen"))
    finally:
        restore()
    assert after_late[0] == (True, [], None, None), (
        f"the new session's review did not read its own baseline alone (a late record read, or its own evicted): "
        f"{after_late[0]}")
    assert after_late[1] == 2, f"a late record evicted the new session's baseline, or was lost: {after_late[1]}"
    assert after_next == (2, None, None), (
        f"the late entries of a workspace gone were kept past the id's next record: {after_next}")
    assert unseen == {"unseen.txt": "3" * 64}, (
        f"the entries of a folder the store could not look at were dropped as gone: {unseen}")


def test_ob4_the_routes_record_and_read_by_workspace():
    ctx = rc_t._load()
    try:
        mgr = rc_t._Manager(workspace="unused")
        rc_t._wire(ctx, mgr)
        records = []
        roots = {("sid-1", "/ws/ob4-checked"): "/share/proj", ("sid-2", "/ws/ob4-ended"): "/share/old"}

        # Tolerant of a call that names no workspace (it is recorded as None): the claim, not the call, fails
        def record(sid, entries, cloned_root=None, cloned_mount=None, *, workspace=None):
            records.append((sid, sorted(entries), cloned_root, cloned_mount, workspace))
            return len(entries)

        ctx.ws.get_workspace_manifests = lambda: SimpleNamespace(
            record=record, get_cloned_root=lambda sid, workspace=None: roots.get((sid, workspace)))
        ctx.ws.get_conversation_for = lambda sid: None

        def checked_then_made_again(sid):
            """The session the route checks, and one made again under its id while the copy runs."""
            checked = rc_t._session(owner="alice")
            checked.workspace_path = "/ws/ob4-checked"
            mgr.sessions[sid] = checked

            def made_again():
                again = rc_t._session(owner="alice")
                again.workspace_path = "/ws/ob4-made-again"
                mgr.sessions[sid] = again

            return made_again

        # The clone: the session made again under the id before the copy returns
        made_again = checked_then_made_again("sid-1")
        mgr.clone_result = {
            "dest": "proj", "cloned_root": "/share/proj", "copied_files": 1, "copied_bytes": 1,
            "skipped_symlinks": 0, "skipped_special": 0, "manifest": {"proj/a.txt": "0" * 64},
        }
        real_clone = mgr.clone_directory

        def clone_directory(session_id, src_path, dest_subdir=""):
            made_again()
            return real_clone(session_id, src_path, dest_subdir)

        mgr.clone_directory = clone_directory
        ctx.mod.clone_host_directory(
            "sid-1", SimpleNamespace(src_path="/share/proj", dest_subdir=""), current_user={"sub": "alice"})
        # The upload: the same
        made_again = checked_then_made_again("sid-3")

        def upload_files(session_id, items, dest_subdir=""):
            made_again()
            return {"written": [{"relative_path": "up.txt", "sha256": "1" * 64}], "refused": [], "written_bytes": 2}

        mgr.upload_files = upload_files
        ctx.mod.upload_workspace_files(
            "sid-3", [SimpleNamespace(filename="up.txt", file=io.BytesIO(b"up"))], "", current_user={"sub": "alice"})
        mgr.list_sessions = lambda: [
            {"session_id": "sid-1", "workspace_path": "/ws/ob4-checked"},
            {"session_id": "sid-2", "workspace_path": "/ws/ob4-made-again"},
        ]
        rows = [(row.session_id, row.has_cloned_baseline) for row in ctx.mod.list_sessions()]
    finally:
        ctx.restore()
    assert records == [
        ("sid-1", ["proj/a.txt"], "/share/proj", "proj", "/ws/ob4-checked"),
        ("sid-3", ["up.txt"], None, None, "/ws/ob4-checked"),
    ], f"a copy route did not record for the workspace of the session it checked: {records}"
    assert rows == [("sid-1", True), ("sid-2", False)], (
        f"the listing did not read the cloned baseline of each row's own workspace: {rows}")
