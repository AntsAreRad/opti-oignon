"""Contracts for the mode lock that is there and gives no field: it gives Bulbe.

``security_mode._read_mode`` trusted security.yaml alone whenever ``_read_lockfile`` gave no field -- for a lock
that is not there (a fresh install), but also for one that is there and cannot be read (its rights, a folder at its
name, bytes that are not UTF-8, a link to nothing or to a device), that cannot be looked at (its folder not
searchable), or that holds none: whoever could spoil the lock and write ``security_mode: daily`` had Daily mode
without the ceremony; and a FIFO at its name held every reader of the mode. Now one read of the lock -- a regular
file only, its type checked before the open and again once open, opened without blocking, no longer than any lock
written -- tells the three
apart, and a lock that is there, or may be, and gives no field fails secure to Bulbe (``lockfile_unreadable``, the
reason recorded). Only a lock that is not there leaves security.yaml alone to say the mode. The mode cache sees the
entry at the lock's name itself, the mode status says such a lock, and the emergency stop writes Bulbe over a Bulbe
that a fault gives, which would last only until the fault is repaired.

  * LK1 -- A LOCK THAT IS THERE AND GIVES NO FIELD GIVES BULBE: its rights refusing it (unless root), a folder at
    its name, bytes that are not UTF-8, an empty file, a link to nothing, a link to a device (not opened: its type
    is checked first), a FIFO (answered at once), a file longer than any lock written, its folder not searchable
    (unless root): the mode reads Bulbe whatever security.yaml says, the event recorded with its reason, the reason
    check gives is the case's, and the mode's state says the lock is there -- or, its folder not searchable, that
    this is not known; no lock at all: security.yaml, as before.
  * LK2 -- THE WAY BACK: on one manager, the lock made readable again reads the mode it says, the verdict recorded
    once while the files are unchanged; a link to nothing that comes at the lock's name reads Bulbe, and once it
    goes the mode security.yaml says.
  * LK3 -- CHECK AND THE STATUS SAY IT (supersedes mv6[lock_unread] of test_mode_verdict_contracts.py, its case
    kept): check says Bulbe for a lock it cannot read, and for one that holds no field, as the platform reads it,
    recording nothing; the mode status says the lock is there, why it gives no field, no agreement, nothing verified.
  * LK4 -- THE EMERGENCY STOP OVER IT (supersedes mv12[none_lockfile_a_folder] of test_mode_verdict_contracts.py,
    its case kept): a folder at the lock's name, something in it, no key file; a lock its rights refuse, signed
    under the key file; a Bulbe kept from a key file that did not load, which loads since: the mode reads Bulbe
    before the stop, recorded; the forced stop writes Bulbe into security.yaml all the same and the mode reads Bulbe
    once the fault is repaired; a stop that cannot write the lock and leaves one that gives no field says the mode
    reads Bulbe from it, never that none is there.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_mode_verdict_contracts as mv_t  # noqa: E402

# Each case by its name in the junit file: the ladder reads a parametrized contract's budget case by case.
BUDGET_S = {
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[its_rights]": 5.0,
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[a_folder]": 5.0,
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[not_UTF-8]": 5.0,
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[empty]": 5.0,
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[longer_than_any_written]": 5.0,
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[a_FIFO]": 5.0,
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[a_link_to_nothing]": 5.0,
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[a_link_to_a_device]": 5.0,
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[its_folder_not_searchable]": 5.0,
    "test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe[no_lock]": 5.0,
    "test_lk2_the_way_back": 5.0,
    "test_lk3_check_and_the_status_say_it[lock_unread]": 5.0,
    "test_lk3_check_and_the_status_say_it[holds_no_field]": 5.0,
    "test_lk4_the_emergency_stop_over_it[a_folder]": 5.0,
    "test_lk4_the_emergency_stop_over_it[its_rights]": 5.0,
    "test_lk4_the_emergency_stop_over_it[a_stale_Bulbe]": 5.0,
    "test_lk4_the_emergency_stop_over_it[cut]": 5.0,
}

_NEEDS_A_USER = ("its rights", "its folder not searchable")
_REASON = {"its rights": "cannot be read here (PermissionError)", "a folder": "is not a regular file",
           "not UTF-8": "cannot be read here (UnicodeDecodeError)", "empty": "holds no field",
           "longer than any written": "holds more than 4096 bytes", "a FIFO": "is not a regular file",
           "a link to nothing": "cannot be read here (FileNotFoundError)", "a link to a device": "is not a regular file",
           "its folder not searchable": "cannot be looked at here (PermissionError)", "no lock": None}


def _case_id(value):
    """A case's id with no space, so that one blade names one case."""
    return value.replace(" ", "_")


def _recorder(world):
    """The mode's records, each event with its details, kept in a list instead of the chain."""
    recorded = []
    world.sm._audit_log = lambda event, severity="INFO", **details: recorded.append((event, details))
    return recorded


def _events(recorded):
    return [event for event, _details in recorded]


def _spoil(world, how):
    """The lock made unreadable, a folder, bytes that are not UTF-8, empty, longer than any written, a FIFO, a link
    to nothing or to a device, or its folder not searchable."""
    if how == "its rights":
        os.chmod(world.lock, 0)
        return
    if how == "its folder not searchable":
        os.chmod(world.data, 0o600)
        return
    os.chmod(world.lock, 0o600)
    if how == "not UTF-8":
        world.lock.write_bytes(b"MODE:\xffdaily\n")
    elif how == "empty":
        world.lock.write_bytes(b"")
    elif how == "longer than any written":
        world.lock.write_bytes(world.lock.read_bytes() + b"#" * 5000 + b"\n")
    else:
        world.lock.unlink()
        if how == "a folder":
            world.lock.mkdir()
        elif how == "a FIFO":
            os.mkfifo(world.lock)
        elif how == "a link to a device":
            world.lock.symlink_to("/dev/null")
        else:
            world.lock.symlink_to(world.data / "nothing here")


def _give_back(world):
    """The folder searchable and the lock readable again, where they can be."""
    os.chmod(world.data, 0o755)
    if world.lock.is_symlink() or not world.lock.is_file():
        return
    os.chmod(world.lock, 0o400)


@pytest.mark.parametrize("how", ["its rights", "a folder", "not UTF-8", "empty", "longer than any written", "a FIFO",
                                 "a link to nothing", "a link to a device", "its folder not searchable", "no lock"],
                         ids=_case_id)
def test_lk1_a_lock_that_is_there_and_gives_no_field_gives_bulbe(tmp_path, monkeypatch, how):
    if how in _NEEDS_A_USER and os.geteuid() == 0:
        pytest.skip("root reads a file of mode 0 and searches a folder of mode 0o600")
    world = mv_t._world(tmp_path, monkeypatch)
    opened, real_open = [], os.open

    def recording_open(path, flags, *args, **kwargs):
        opened.append(os.path.realpath(path))
        return real_open(path, flags, *args, **kwargs)

    try:
        mv_t._sources(world, "daily", None if how == "no lock" else "daily", None)
        if how != "no lock":
            _spoil(world, how)
        recorded = _recorder(world)
        monkeypatch.setattr(os, "open", recording_open)
        still, mode = mv_t._in_a_thread(world.sm.SecurityModeManager().get_current_mode)
        monkeypatch.setattr(os, "open", real_open)
        if still:
            # The reader waits on the FIFO for a writer: one comes and goes, so it lets go
            with contextlib.suppress(OSError):
                os.close(os.open(world.lock, os.O_WRONLY | os.O_NONBLOCK))
        else:
            reason = world.sm._lock_unread()
            recorded = list(recorded)
            state = world.sm.SecurityModeManager().get_state()
    finally:
        monkeypatch.setattr(os, "open", real_open)
        _give_back(world)
        world.restore()
    assert not still, f"{how}: the mode's reader still waits on the lock"
    if how == "no lock":
        assert (mode, recorded) == ("daily", []), f"no lock no longer reads security.yaml alone: {mode} {recorded}"
    else:
        assert mode == "bulbe", f"{how}: a lock that is there and gives no field let security.yaml choose {mode}"
        assert _events(recorded) == ["lockfile_unreadable"], f"{how}: the verdict was not recorded: {recorded}"
        assert recorded[0][1].get("lock_unread") == _REASON[how], f"{how}: the record says no reason: {recorded}"
    assert reason == _REASON[how], f"{how}: check gives the reason {reason!r}"
    if how == "a link to a device":
        assert "/dev/null" not in opened, "the device behind the link at the lock's name was opened"
    there = None if how == "its folder not searchable" else how != "no lock"
    assert (state.lockfile_exists, state.lock_unread) == (there, _REASON[how]), (
        f"{how}: the mode's state says {state.lockfile_exists} {state.lock_unread!r}")


def test_lk2_the_way_back(tmp_path, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root reads a file of mode 0")
    world = mv_t._world(tmp_path, monkeypatch)
    try:
        mv_t._sources(world, "daily", "daily", None)
        recorded = _recorder(world)
        manager = world.sm.SecurityModeManager()
        seen = [manager.get_current_mode()]
        os.chmod(world.lock, 0)
        seen += [manager.get_current_mode(), manager.get_current_mode()]
        os.chmod(world.lock, 0o400)
        seen.append(manager.get_current_mode())
        os.chmod(world.lock, 0o600)
        world.lock.unlink()
        seen.append(manager.get_current_mode())
        world.lock.symlink_to(world.data / "nothing here")
        seen.append(manager.get_current_mode())
        world.lock.unlink()
        seen.append(manager.get_current_mode())
    finally:
        _give_back(world)
        world.restore()
    assert seen == ["daily", "bulbe", "bulbe", "daily", "daily", "bulbe", "daily"], (
        f"read, made unreadable twice, readable again, removed, a link to nothing, removed: {seen}")
    assert _events(recorded) == ["lockfile_unreadable", "lockfile_unreadable"], (
        f"each verdict is recorded once, while the files are unchanged: {recorded}")


_SAID = {
    "lock_unread": "security mode: bulbe, failing secure: its lock file cannot be read here (PermissionError); "
                   "security.yaml says daily",
    "holds_no_field": "security mode: bulbe, failing secure: its lock file holds no field; security.yaml says daily",
}
_STATUS = {"lock_unread": "cannot be read here (PermissionError)", "holds_no_field": "holds no field"}


@pytest.mark.parametrize("state", list(_SAID))
def test_lk3_check_and_the_status_say_it(tmp_path, monkeypatch, capsys, state):
    if state == "lock_unread" and os.geteuid() == 0:
        pytest.skip("root reads a file of mode 0")
    world = mv_t._world(tmp_path, monkeypatch)
    try:
        mv_t._sources(world, "daily", "daily", mv_t.K0)
        _spoil(world, "its rights" if state == "lock_unread" else "empty")
        recorded = _recorder(world)
        report = world.sm.mode_report()
        assert recorded == [], "check records nothing"
        assert report["mode"] == world.sm.SecurityModeManager().get_current_mode(), "said as the platform reads it"
        assert world.dbe.main(["check"]) == 0
        said = [line for line in capsys.readouterr().out.splitlines() if line.startswith("security mode: ")]
        status = world.sm.SecurityModeManager().status()
    finally:
        _give_back(world)
        world.restore()
    assert len(said) == 1 and _SAID[state] in said[0], said
    told = (status["mode"], status["lockfile_exists"], status["lock_unread"], status["sources_agree"],
            status["hmac_valid"])
    assert told == ("bulbe", True, _STATUS[state], False, None), f"the mode status says: {told}"


@pytest.mark.parametrize("left", ["a folder", "its rights", "a stale Bulbe", "cut"], ids=_case_id)
def test_lk4_the_emergency_stop_over_it(tmp_path, monkeypatch, left):
    if left == "its rights" and os.geteuid() == 0:
        pytest.skip("root reads a file of mode 0")
    world = mv_t._world(tmp_path, monkeypatch)
    try:
        sm = world.sm
        if left == "its rights":
            mv_t._key_file(world, monkeypatch)
            mv_t._sources(world, "daily", "daily", mv_t.K1)
            os.chmod(world.lock, 0)
        elif left == "a stale Bulbe":
            mv_t._key_file(world, monkeypatch, wrapped=True)
            mv_t._sources(world, "daily", "daily", mv_t.K1)
            monkeypatch.delenv("OPTI_KEYFILE_PASSPHRASE")
        else:
            mv_t._sources(world, "daily")
        recorded = _recorder(world)
        if left == "cut":
            write = sm._write_lockfile

            def cut(mode, user_id, key):
                world.lock.write_bytes(b"")
                raise OSError(28, "No space left on device")

            monkeypatch.setattr(sm, "_write_lockfile", cut)
        else:
            if left == "a folder":
                (world.lock / "kept").mkdir(parents=True)
        manager = sm.SecurityModeManager()
        before = manager.get_current_mode()
        if left == "a stale Bulbe":
            # The key file loads from now on; the files are unchanged, so the manager keeps the Bulbe it read
            monkeypatch.setenv("OPTI_KEYFILE_PASSPHRASE", mv_t.PHRASE)
        forced = manager.escalate_to_bulbe("emergency-stop", force=True)
        said = world.yaml.read_text(encoding="utf-8")
        if left == "cut":
            monkeypatch.setattr(sm, "_write_lockfile", write)
        # The fault repaired: the folder gone, the lock readable
        if left == "a folder":
            shutil.rmtree(world.lock)
        _give_back(world)
        after = (manager.get_current_mode(), sm.SecurityModeManager().get_current_mode())
    finally:
        _give_back(world)
        world.restore()
    assert forced["success"] is True and forced["mode"] == "bulbe", forced
    assert "security_mode: bulbe" in said, f"the stop left security.yaml saying: {said!r} ({forced})"
    assert after == ("bulbe", "bulbe"), f"the mode once the fault is repaired: {after}"
    if left == "cut":
        assert before == "daily", before
        assert ("the lockfile could not be written (OSError): it holds no field, and the mode reads Bulbe from that"
                in forced["message"]), forced
        assert "lockfile_unreadable" in _events(recorded), f"the cut lock's verdict was not recorded: {recorded}"
    else:
        fault = "lockfile_unverifiable" if left == "a stale Bulbe" else "lockfile_unreadable"
        assert before == "bulbe" and _events(recorded)[:1] == [fault], (
            f"{left}: the fault let security.yaml choose {before}: {recorded}")
        assert forced["changed"] is True, f"{left}: the stop wrote nothing over a Bulbe a fault gives: {forced}"
    if left == "a folder":
        assert ("the lockfile could not be written (IsADirectoryError): it is not a regular file, and the mode reads "
                "Bulbe from that" in forced["message"]), forced
