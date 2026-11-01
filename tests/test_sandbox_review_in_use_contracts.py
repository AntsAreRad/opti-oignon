"""Contracts for the review of a workspace: the sweep never takes one under review; a review a destroy overtook says so.

The review (``sandbox_workspace.generate_workspace_diff``, the panel's "review changes", and the apply's own
recomputation) walks the live workspace and compares it with the baseline recorded at copy-in: a baseline file the
walk does not find is "deleted". It ran with its session in no use: a sweep that came during the walk removed the
tree under it, and the files the walk had not reached yet read as deleted -- a false diff, its ``diff_hash`` one the
user could send to the apply. An explicit destroy did the same, and a root already gone raised a bare
``FileNotFoundError``. Now the walk runs with its session in use (``SandboxManager.in_use``), which the sweep's
snapshot skips and its check made again declines; a review an explicit destroy overtook raises ``ValueError`` (the
routes answer 404), never a diff of the removal.

  * RH1 -- NO SWEEP TAKES A WORKSPACE UNDER REVIEW: a session idle past the TTL, the panel's list at the walk's first
    file: the session is kept and the review is whole (no file it did not reach reads as deleted); inside the apply,
    the review's hold nests in the apply's (two, then none); a review counts no activity, and the next list takes
    the session.
  * RH2 -- A DESTROY DURING A REVIEW SAYS SO: an explicit destroy at the walk's first file, before the walk (the root
    gone: the walk's own error chained) or after it: the review raises ``ValueError`` (the session destroyed while it
    was reviewed), never a diff; the apply whose review a destroy overtook answers the same (404), never drift (409),
    and writes nothing; a session made again under the same id is not held by the review of the old one, nor
    reviewed by the apply of the old one when it is made between the apply's hold and its review (round 1), nor
    lends its deletion confirmations, its approvals or its activity to that apply when it is made after the review
    (round 2).
  * RH3 -- WHAT A REVIEW LEAVES: an unknown session, an inactive one and a missing workspace answer as before, an
    unknown one also when a session is made under its id between the review's count and its check (round 1: it is
    not taken for it); a walk that raised with its session alive (the bound exceeded, a folder it cannot read) raises
    as before; after each, as after a review that answered, nothing stays in use, no activity is counted, and the
    next list takes the session.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_sandbox_host_writes_in_use_contracts as hu_t  # noqa: E402

BUDGET_S = {
    "test_rh1_no_sweep_takes_a_workspace_under_review": 5.0,
    "test_rh2_a_destroy_during_a_review_says_so": 5.0,
    "test_rh3_what_a_review_leaves": 5.0,
}

_window = hu_t._window
_workspace = hu_t._workspace
_refusal = hu_t._refusal
_idle = hu_t._idle
_racing = hu_t._racing


def _baseline(ws, sid, workspace):
    """Three files in the workspace (one in a folder), recorded as its baseline."""
    (workspace / "sub").mkdir()
    for rel, data in (("a.txt", b"rh-a"), ("sub/b.txt", b"rh-b"), ("z.txt", b"rh-z")):
        (workspace / rel).write_bytes(data)
    ws.get_workspace_manifests().record(
        sid, {rel: ws.manifest_hash_file(str(workspace / rel)) for rel in ("a.txt", "sub/b.txt", "z.txt")},
        workspace=str(workspace))


def _changes(answer):
    """The review's classified changes, (path, kind) each, or what it raised."""
    if answer[0] != "answered":
        return answer
    return [(change.path, change.kind) for change in answer[1].entries]


def _destroyed(sid):
    return ("ValueError", f"Session {sid} was destroyed while it was reviewed")


def test_rh1_no_sweep_takes_a_workspace_under_review(tmp_path):
    _sm, ws, manager, restore = _window(tmp_path)
    share = tmp_path / "share"
    (share / "out").mkdir()
    try:
        # The panel's review: the sweep at the walk's first file
        sid, workspace, _real = _workspace(manager, "rh1")
        _baseline(ws, sid, workspace)
        kept = []
        real = _racing(ws, "_hash_fd", hu_t._sweep_at_first(manager, sid, kept))
        _idle(manager, sid)
        try:
            review = _refusal(ws.generate_workspace_diff, sid, manager=manager)
        finally:
            ws._hash_fd = real
        session = manager.get_session(sid)
        in_use_after = session.host_operations if session is not None else None
        idle_after = session is not None and (
            time.time() - session.last_activity > manager._config.idle_ttl_seconds)
        manager.list_sessions()
        swept_after = manager.get_session(sid) is None
        # The apply's review: its hold nests in the apply's
        sid, workspace, _real = _workspace(manager, "rh1-apply")
        (workspace / "a.txt").write_bytes(b"rh-applied")
        manager.approve_files(sid, ["a.txt"])
        diff_hash = ws.generate_workspace_diff(sid, manager=manager).diff_hash
        counts = []

        def count():
            if not counts:
                counts.append(manager.get_session(sid).host_operations)

        real = _racing(ws, "_hash_fd", count)
        try:
            applied = _refusal(
                ws.apply_workspace_changes, sid, diff_hash, manager=manager, target_dir=str(share / "out"))
        finally:
            ws._hash_fd = real
        apply_session = manager.get_session(sid)
        apply_in_use_after = apply_session.host_operations if apply_session is not None else None
        on_host = sorted(p.name for p in (share / "out").iterdir())
    finally:
        restore()
    assert kept == [True], f"the sweep took a workspace while it was reviewed: {_changes(review)}"
    assert review[0] == "answered", review
    assert _changes(review) == [] and review[1].unchanged == 3, (
        f"a review the sweep came into read files it had not reached as deleted: {_changes(review)}")
    assert in_use_after == 0, f"a review left its session in use: {in_use_after}"
    assert idle_after, "a review counted as activity"
    assert swept_after, "a session no review uses any more, idle past the TTL, was never swept"
    assert counts == [2], f"the apply's review did not hold its session within the apply's hold: {counts}"
    assert applied[0] == "answered" and on_host == ["a.txt"], (applied, on_host)
    assert apply_in_use_after == 0, f"the apply and its review left the session in use: {apply_in_use_after}"


def test_rh2_a_destroy_during_a_review_says_so(tmp_path):
    _sm, ws, manager, restore = _window(tmp_path)
    share = tmp_path / "share"
    (share / "out").mkdir()
    reviews = {}
    try:
        # name: (the seam, whether the destroy comes before it, else after it)
        cases = {
            "at_a_file": ("_hash_fd", True),
            "root_gone": ("_walk_workspace_hashes", True),
            "after_the_walk": ("_walk_workspace_hashes", False),
        }
        for name, (seam, before) in cases.items():
            sid, workspace, _real = _workspace(manager, f"rh2-{name}")
            _baseline(ws, sid, workspace)
            ended = []
            destroy = hu_t._destroy_at_first(manager, sid, ended)
            real = _racing(ws, seam, destroy) if before else _racing(ws, seam, lambda: None, destroy)
            try:
                answer = ("answered", _changes(("answered", ws.generate_workspace_diff(sid, manager=manager))))
                cause = None
            except Exception as exc:  # noqa: BLE001 -- the contract reads it
                answer, cause = (type(exc).__name__, str(exc)), exc.__cause__
            finally:
                setattr(ws, seam, real)
            reviews[name] = (sid, answer, ended, cause, workspace.exists())
        # The apply: the destroy at its review's first file
        sid, workspace, _real = _workspace(manager, "rh2-apply")
        (workspace / "a.txt").write_bytes(b"rh-applied")
        (workspace / "b.txt").write_bytes(b"rh-applied")
        manager.approve_files(sid, ["a.txt", "b.txt"])
        diff_hash = ws.generate_workspace_diff(sid, manager=manager).diff_hash
        apply_ended = []
        real = _racing(ws, "_hash_fd", hu_t._destroy_at_first(manager, sid, apply_ended))
        try:
            applied = _refusal(
                ws.apply_workspace_changes, sid, diff_hash, manager=manager, target_dir=str(share / "out"))
        finally:
            ws._hash_fd = real
        apply_sid = sid
        on_host = sorted(p.name for p in (share / "out").iterdir())
        # A session made again under the same id while the old one is reviewed
        sid, workspace, _real = _workspace(manager, "rh2-again")
        _baseline(ws, sid, workspace)
        again = []

        def destroy_and_make_again():
            if not again:
                again.append(manager.destroy_sandbox(sid))
                manager.create_sandbox(sid, label="rh2-made-again")

        real = _racing(ws, "_hash_fd", destroy_and_make_again)
        try:
            old_review = _refusal(ws.generate_workspace_diff, sid, manager=manager)
        finally:
            ws._hash_fd = real
        again_sid = sid
        new = manager.get_session(sid)
        new_in_use = new.host_operations if new is not None else None
        _idle(manager, sid)
        manager.list_sessions()
        new_swept = manager.get_session(sid) is None
        # The apply: its session destroyed and another made under its id between its hold and its review
        sid, workspace, _real = _workspace(manager, "rh2-apply-again")
        (workspace / "c.txt").write_bytes(b"rh-applied")
        manager.approve_files(sid, ["c.txt"])
        diff_hash = ws.generate_workspace_diff(sid, manager=manager).diff_hash
        apply_again = []

        def made_again_before_the_review():
            if not apply_again:
                apply_again.append(manager.destroy_sandbox(sid))
                manager.create_sandbox(sid, label="rh2-apply-made-again")
                made = Path(manager.get_workspace_path(sid))
                (made / "c.txt").write_bytes(b"rh-applied")
                manager.approve_files(sid, ["c.txt"])

        real = _racing(ws, "_apply_reviewed", made_again_before_the_review)
        try:
            applied_again = _refusal(
                ws.apply_workspace_changes, sid, diff_hash, manager=manager, target_dir=str(share / "out"))
        finally:
            ws._apply_reviewed = real
        apply_new = manager.get_session(sid)
        apply_new_in_use = apply_new.host_operations if apply_new is not None else None
        apply_again_sid = sid
        on_host_again = sorted(p.name for p in (share / "out").iterdir())
        # The apply: its session destroyed after its review and another made under its id, which confirms the
        # deletion the review showed and approves the file it added
        sid, workspace, _real = _workspace(manager, "rh2-apply-gates")
        ws.get_workspace_manifests().record(sid, {"gone.txt": "0" * 64}, workspace=str(workspace))
        (workspace / "d.txt").write_bytes(b"rh-added")
        gates = share / "gates"
        gates.mkdir()
        (gates / "gone.txt").write_bytes(b"host-kept")
        diff_hash = ws.generate_workspace_diff(sid, manager=manager).diff_hash
        gates_again = []

        def made_again_after_the_review():
            if not gates_again:
                gates_again.append(manager.destroy_sandbox(sid))
                manager.create_sandbox(sid, label="rh2-gates-made-again")
                (Path(manager.get_workspace_path(sid)) / "d.txt").write_bytes(b"rh-added")
                manager.confirm_deletions(sid, ["gone.txt"])
                gates_again.append(manager.approve_files(sid, ["d.txt"]))
                gates_again.append(_idle(manager, sid))

        real = _racing(ws, "generate_workspace_diff", lambda: None, made_again_after_the_review)
        try:
            gated = _refusal(ws.apply_workspace_changes, sid, diff_hash, manager=manager, target_dir=str(gates))
        finally:
            ws.generate_workspace_diff = real
        gates_new = manager.get_session(sid)
        gates_new_activity = gates_new.last_activity if gates_new is not None else None
        on_host_gates = sorted(p.name for p in gates.iterdir())
    finally:
        restore()
    for name, (rid, answer, ended, _cause, left) in reviews.items():
        assert ended == [True], f"{name}: the destroy never came: {reviews[name]}"
        assert answer == _destroyed(rid), f"{name}: a review a destroy overtook did not say so: {answer}"
        assert not left, f"{name}: the workspace of a review a destroy overtook stayed"
    assert isinstance(reviews["root_gone"][3], FileNotFoundError), (
        f"the walk's own error is not chained to the destroy's answer: {reviews['root_gone']}")
    assert apply_ended == [True], apply_ended
    assert applied == _destroyed(apply_sid), (
        f"an apply whose review a destroy overtook did not say so (drift is 409, not 404): {applied}")
    assert on_host == [], f"an apply whose review a destroy overtook wrote: {on_host}"
    assert again == [True], again
    assert old_review == _destroyed(again_sid), (
        f"the review of a session destroyed and made again under its id did not say so: {old_review}")
    assert new_in_use == 0, f"a session made again under the id was held by the old one's review: {new_in_use}"
    assert new_swept, "a session made again under the id of one under review was never swept"
    assert apply_again == [True], apply_again
    assert applied_again == _destroyed(apply_again_sid), (
        f"the apply of a session destroyed and made again under its id reviewed the new one: {applied_again}")
    assert on_host_again == [], f"the apply of a destroyed session wrote the new one's files: {on_host_again}"
    assert apply_new_in_use == 0, f"the new session was left in use by the old one's apply: {apply_new_in_use}"
    assert gates_again[0] is True and gates_again[1] and len(gates_again) == 3, gates_again
    assert gated[0] == "answered" and gated[1]["deleted"] == [] and gated[1]["applied"] == [], (
        f"the apply of a destroyed session used the gates of the session made again under its id: {gated}")
    assert (gated[1]["skipped_unapproved"], gated[1]["skipped_unconfirmed"], gated[1]["refused"]) == (1, 1, []), (
        f"the apply of a destroyed session took the new one's approval or confirmation for its own: {gated}")
    assert on_host_gates == ["gone.txt"], (
        f"the apply of a destroyed session deleted or wrote on the new one's confirmations: {on_host_gates}")
    assert gates_new_activity == gates_again[2], "the apply of a destroyed session counted the new one's activity"


class _AfterFirstRelease:
    """The manager's lock, running ``hook`` once, right after the first release that follows its arming."""

    def __init__(self, lock, hook):
        self.lock, self.hook, self.armed = lock, hook, False

    def __enter__(self):
        return self.lock.__enter__()

    def __exit__(self, *exc):
        done = self.lock.__exit__(*exc)
        if self.armed:
            self.armed = False
            self.hook()
        return done


def test_rh3_what_a_review_leaves(tmp_path):
    _sm, ws, manager, restore = _window(tmp_path)
    seen = {}
    try:
        unknown = _refusal(ws.generate_workspace_diff, "rh3-nobody", manager=manager)
        # A session made under an unknown id between the review's count and its check
        late = []
        lock = _AfterFirstRelease(
            manager._lock, lambda: late.append(manager.create_sandbox("rh3-late", label="rh3-late").session_id))
        manager._lock = lock
        try:
            lock.armed = True
            made_late = _refusal(ws.generate_workspace_diff, "rh3-late", manager=manager)
        finally:
            manager._lock = lock.lock
        late_session = manager.get_session("rh3-late")
        late_in_use = late_session.host_operations if late_session is not None else None
        for name in ("answered", "inactive", "workspace_gone", "over_the_bound", "unreadable"):
            sid, workspace, _real = _workspace(manager, f"rh3-{name}")
            _baseline(ws, sid, workspace)
            if name == "inactive":
                manager.get_session(sid).active = False
            if name == "workspace_gone":
                os.rename(workspace, tmp_path / "moved-away")
            if name == "unreadable":
                os.chmod(workspace / "sub", 0)
            long_ago = _idle(manager, sid)
            answer = _refusal(
                ws.generate_workspace_diff, sid, manager=manager, max_entries=1 if name == "over_the_bound" else None)
            session = manager.get_session(sid)
            in_use, activity = session.host_operations, session.last_activity
            if name == "inactive":
                session.active = True
            if name == "workspace_gone":
                os.rename(tmp_path / "moved-away", workspace)
            if name == "unreadable":
                os.chmod(workspace / "sub", 0o755)
            manager.list_sessions()
            seen[name] = (answer, in_use, activity == long_ago, manager.get_session(sid) is None)
            if manager.get_session(sid) is not None:
                manager.destroy_sandbox(sid)
    finally:
        restore()
    assert unknown == ("ValueError", "Session not found: rh3-nobody"), unknown
    assert late == ["rh3-late"], late
    assert made_late == ("ValueError", "Session not found: rh3-late"), (
        f"a session made under the id between the review's count and its check was taken for it: {made_late}")
    assert late_in_use == 0, f"the session made late was left in use: {late_in_use}"
    assert seen["answered"][0][0] == "answered" and _changes(seen["answered"][0]) == [], seen["answered"]
    assert seen["inactive"][0][0] == "ValueError" and seen["inactive"][0][1].startswith("Session is inactive"), seen
    assert seen["workspace_gone"][0][0] == "ValueError" and "missing" in seen["workspace_gone"][0][1], seen
    assert seen["over_the_bound"][0][0] == "WorkspaceDiffBoundExceeded", seen["over_the_bound"]
    assert seen["unreadable"][0][0] == "WorkspaceReviewDrift", (
        f"a walk that raised with its session alive was answered otherwise than before: {seen['unreadable']}")
    for name, (answer, in_use, no_activity, swept) in seen.items():
        assert in_use == 0, f"{name}: a review left its session in use: {answer}"
        assert no_activity, f"{name}: a review counted as activity: {answer}"
        assert swept, f"{name}: the session idle past the TTL was never swept after its review: {answer}"
