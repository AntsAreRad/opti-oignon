"""Contracts for the host's copies: each works on the session it holds, never on one made again under its id.

The host's copies run with their session in use (``_copying_in``: an upload, a copy of files or of a folder, a clone;
``_copying_out``: a copy-out of one file or of several). Inside the hold each still found its workspace, read its
approvals and counted its activity by the id. A session destroyed and made again under the same id between the hold
and that lookup -- the id chosen by the caller, or the chat's coding agent making its session again after a sweep --
took the copy's place: a copy in wrote into the new workspace and answered that the session was destroyed, a copy
out copied the new session's approved files to the host under the old one's request, and a copy's end counted
activity on the new session. Now the decorators hand the held session to the copy, which works on it alone.

  * HO1 -- A COPY IN WORKS ON THE SESSION IT HOLDS: an upload, a copy of files, a copy of a folder and a clone, each
    with its session destroyed and another made under its id right after the hold checked it: the copy answers that
    its session was destroyed, the new workspace holds nothing of it, and the new session's activity is untouched;
    destroyed there with its workspace still standing, each answers the destroy and writes nothing into it (round 2).
  * HO2 -- A COPY OUT WORKS ON THE SESSION IT HOLDS: a copy-out of one file and of two, the same: nothing of the new
    session (its own approved files at the same names) reaches the host; a destroy whose workspace still stands (a
    leftover, or its removal under way) stops a copy-out as before: the batch's later files and a single file are
    not copied (round 1), also when it comes while the copy-out reads its gates (round 2).
  * HO3 -- WHAT A HOLD COUNTS: the batch's files are copied within the batch's one hold, a copy-out of one file holds
    itself once, and neither looks its session up by id again (one check each, at one count); nothing stays in use
    after either; without a remake each copy lands as before.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_sandbox_host_writes_in_use_contracts as hu_t  # noqa: E402

BUDGET_S = {
    "test_ho1_a_copy_in_works_on_the_session_it_holds": 5.0,
    "test_ho2_a_copy_out_works_on_the_session_it_holds": 5.0,
    "test_ho3_what_a_hold_counts": 5.0,
}

_window = hu_t._window
_refusal = hu_t._refusal
_Stream = hu_t._Stream


def _made_again_after_the_check(manager, sid, made):
    """Right after the hold checked the session (``_active_workspace_of``, once): the session destroyed and another
    made under its id, idle since long ago; ``made`` holds (the new session, its last activity)."""
    real = manager._active_workspace_of

    def checked(session_id, session):
        workspace = real(session_id, session)
        if session_id == sid and not made:
            made.append(None)  # once, whatever looks the session up through here meanwhile
            manager.destroy_sandbox(sid)
            again = manager.create_sandbox(sid, label=f"{sid}-made-again")
            again.last_activity = time.time() - 3600
            made[0] = (again, again.last_activity)
        return workspace

    manager._active_workspace_of = checked


def _files(folder):
    folder = Path(folder)
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


def test_ho1_a_copy_in_works_on_the_session_it_holds(tmp_path):
    sm, _ws, manager, restore = _window(tmp_path)
    share, host = hu_t._host_sources(tmp_path)
    seen = {}
    try:
        copies = {
            "upload": lambda sid: manager.upload_files(sid, [("up.txt", _Stream(b"up"), 2)]),
            "inject_files": lambda sid: manager.inject_files(sid, [str(host / "one.txt")]),
            "inject_directory": lambda sid: manager.inject_directory(sid, str(host / "tree")),
            "clone": lambda sid: manager.clone_directory(sid, str(share / "proj")),
        }
        for name, copy in copies.items():
            sid = f"ho1-{name}"
            manager.create_sandbox(sid, label=sid)
            made = []
            _made_again_after_the_check(manager, sid, made)
            try:
                answer = _refusal(copy, sid)
            finally:
                del manager._active_workspace_of
            again, activity = made[0] if made else (None, None)
            seen[name] = (answer, bool(made), _files(again.workspace_path) if again else None,
                          again.last_activity == activity if again else None, again.host_operations if again else None)
            manager.destroy_sandbox(sid)
        # A destroy right after the hold checked the session, its workspace still standing: nothing written there
        left = {}
        real_remove = sm._remove_tree
        sm._remove_tree = lambda path: False
        try:
            for name, copy in copies.items():
                sid = f"ho1-left-{name}"
                session = manager.create_sandbox(sid, label=sid)
                ended = []
                real = manager._active_workspace_of

                def destroyed_after_the_check(session_id, checked, sid=sid, ended=ended, real=real):
                    workspace = real(session_id, checked)
                    hu_t._destroy_at_first(manager, sid, ended)()
                    return workspace

                manager._active_workspace_of = destroyed_after_the_check
                try:
                    answer = _refusal(copy, sid)
                finally:
                    del manager._active_workspace_of
                left[name] = (answer, ended, _files(session.workspace_path))
        finally:
            sm._remove_tree = real_remove
    finally:
        restore()
    for name, (answer, ended, files) in left.items():
        assert ended == [True], f"{name}: control: the destroy did not come"
        assert answer[0] == "ValueError" and "was destroyed" in answer[1], (
            f"{name}: a copy whose session was destroyed after the hold's check did not answer so: {answer}")
        assert files == [], f"{name}: a copy wrote into a destroyed session's standing workspace: {files}"
    for name, (answer, remade, files, untouched, in_use) in seen.items():
        assert remade, f"{name}: control: the session was not made again"
        assert answer[0] == "ValueError" and "was destroyed" in answer[1], (
            f"{name}: a copy whose session was made again under its id did not answer the destroy: {answer}")
        assert files == [], f"{name}: a copy wrote into the session made again under its id: {files}"
        assert untouched, f"{name}: a copy counted its activity on the session made again under its id"
        assert in_use == 0, f"{name}: the session made again was left in use: {in_use}"


def test_ho2_a_copy_out_works_on_the_session_it_holds(tmp_path):
    sm, _ws, manager, restore = _window(tmp_path)
    exported = tmp_path / "exported"
    seen = {}
    try:
        copies = {
            "copy_out_file": lambda sid: manager.copy_out_file(sid, "k.txt", str(exported / "one")),
            "copy_out_batch": lambda sid: manager.copy_out_batch(sid, ["k.txt", "l.txt"], str(exported / "batch")),
        }
        for name, copy in copies.items():
            sid = f"ho2-{name}"
            session = manager.create_sandbox(sid, label=sid)
            for rel, data in (("k.txt", b"old-k"), ("l.txt", b"old-l")):
                (Path(session.workspace_path) / rel).write_bytes(data)
            manager.approve_files(sid, ["k.txt", "l.txt"])
            made = []
            real = manager._active_workspace_of

            def made_again_with_its_own_files(session_id, checked, sid=sid, made=made, real=real):
                workspace = real(session_id, checked)
                if session_id == sid and not made:
                    made.append(None)  # once: the approval below looks the session up through here too
                    manager.destroy_sandbox(sid)
                    again = manager.create_sandbox(sid, label=f"{sid}-made-again")
                    for rel, data in (("k.txt", b"new-k"), ("l.txt", b"new-l")):
                        (Path(again.workspace_path) / rel).write_bytes(data)
                    made[0] = manager.approve_files(sid, ["k.txt", "l.txt"])
                return workspace

            manager._active_workspace_of = made_again_with_its_own_files
            try:
                answer = _refusal(copy, sid)
            finally:
                del manager._active_workspace_of
            seen[name] = (answer, made)
            manager.destroy_sandbox(sid)
        landed = {str(p.relative_to(exported)): p.read_bytes() for p in exported.rglob("*") if p.is_file()}
        # A destroy whose workspace still stands (a leftover, or its removal under way) stops a copy-out as before
        leftovers = tmp_path / "leftovers"
        real_remove = sm._remove_tree
        sm._remove_tree = lambda path: False
        try:
            sid = "ho2-leftover-batch"
            session = manager.create_sandbox(sid, label=sid)
            for rel in ("k.txt", "l.txt", "m.txt"):
                (Path(session.workspace_path) / rel).write_bytes(b"left-" + rel.encode())
            manager.approve_files(sid, ["k.txt", "l.txt", "m.txt"])
            ended = []
            hu_t._racing(manager, "copy_out_file", lambda: None, hu_t._destroy_at_first(manager, sid, ended))
            try:
                batch_left = _refusal(manager.copy_out_batch, sid, ["k.txt", "l.txt", "m.txt"], str(leftovers / "batch"))
            finally:
                del manager.copy_out_file
            sid = "ho2-leftover-one"
            session = manager.create_sandbox(sid, label=sid)
            (Path(session.workspace_path) / "k.txt").write_bytes(b"left-one")
            manager.approve_files(sid, ["k.txt"])
            one_ended = []
            real = manager._active_workspace_of

            def destroyed_after_the_check(session_id, checked):
                workspace = real(session_id, checked)
                hu_t._destroy_at_first(manager, sid, one_ended)()
                return workspace

            manager._active_workspace_of = destroyed_after_the_check
            try:
                one_left = _refusal(manager.copy_out_file, sid, "k.txt", str(leftovers / "one"))
            finally:
                del manager._active_workspace_of
            left_standing = Path(session.workspace_path).exists()
            sid = "ho2-leftover-gates"
            session = manager.create_sandbox(sid, label=sid)
            (Path(session.workspace_path) / "k.txt").write_bytes(b"left-gates")
            manager.approve_files(sid, ["k.txt"])
            gates_ended = []
            hu_t._racing(manager, "is_file_approved", hu_t._destroy_at_first(manager, sid, gates_ended))
            try:
                gates_left = _refusal(manager.copy_out_file, sid, "k.txt", str(leftovers / "gates"))
            finally:
                del manager.is_file_approved
        finally:
            sm._remove_tree = real_remove
        left_landed = sorted(str(p.relative_to(leftovers)) for p in leftovers.rglob("*") if p.is_file())
    finally:
        restore()
    assert ended == [True] and one_ended == [True] and left_standing, (
        f"control: the destroys did not come, or their workspace did not stay: {ended} {one_ended} {left_standing}")
    assert batch_left[0] == "answered" and [r["src_path"] for r in batch_left[1]] == ["k.txt"], (
        f"a batch a destroy overtook went on copying out of its standing workspace: {batch_left}")
    assert one_left[0] == "ValueError" and gates_ended == [True] and gates_left[0] in ("ValueError", "PermissionError"), (
        f"a copy-out whose session was destroyed copied out of its standing workspace: {one_left} {gates_left}")
    assert left_landed == ["batch/k.txt"], f"files of a destroyed session reached the host: {left_landed}"
    for name, (answer, made) in seen.items():
        assert made and made[0], f"{name}: control: the session made again approved nothing"
    assert not [content for content in landed.values() if content.startswith(b"new-")], (
        f"a copy-out copied the files of the session made again under its id to the host: {landed}")
    assert seen["copy_out_batch"][0] == ("answered", []), (
        f"the batch answered otherwise than with nothing copied: {seen['copy_out_batch'][0]}")


def test_ho3_what_a_hold_counts(tmp_path):
    _sm, _ws, manager, restore = _window(tmp_path)
    exported = tmp_path / "exported"
    try:
        sid = "ho3-batch"
        session = manager.create_sandbox(sid, label=sid)
        for rel in ("k.txt", "l.txt"):
            (Path(session.workspace_path) / rel).write_bytes(rel.encode())
        manager.approve_files(sid, ["k.txt", "l.txt"])
        checks = {"batch": [], "one": []}
        real_check = manager._active_workspace_of

        def checking(session_id, checked):
            checks[current[0]].append(checked.host_operations if checked is not None else None)
            return real_check(session_id, checked)

        current = ["batch"]
        manager._active_workspace_of = checking
        try:
            batch = _refusal(manager.copy_out_batch, sid, ["k.txt", "l.txt"], str(exported / "batch"))
            after_batch = session.host_operations
            current[0] = "one"
            one = _refusal(manager.copy_out_file, sid, "k.txt", str(exported / "one"))
        finally:
            del manager._active_workspace_of
        after_one = session.host_operations
        # Without a remake: an upload and a copy of files answer as before
        upload = _refusal(manager.upload_files, sid, [("up.txt", _Stream(b"up"), 2)])
        inject = _refusal(manager.inject_files, sid, [str(exported / "one" / "k.txt")])
        landed = _files(session.workspace_path)
    finally:
        restore()
    assert checks["batch"] == [1], (
        f"the batch's files were not copied within its one hold, or a copy looked its session up by id: {checks}")
    assert batch[0] == "answered" and [r["src_path"] for r in batch[1]] == ["k.txt", "l.txt"], batch
    assert checks["one"] == [1] and one[0] == "answered", (
        f"a copy-out of one file did not hold itself once, or looked its session up by id: {checks} {one}")
    assert (after_batch, after_one) == (0, 0), f"a copy-out left its session in use: {after_batch} {after_one}"
    assert upload[0] == "answered" and inject[0] == "answered", (upload, inject)
    assert landed == ["k.txt", "l.txt", "up.txt"], f"a copy without a remake did not land as before: {landed}"
