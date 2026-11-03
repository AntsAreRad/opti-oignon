#!/usr/bin/env python3
"""Contracts: the audit chain's anchor never re-anchors in silence.

The signed audit log compares its chain with an anchor file (the anchored height and tip, an HMAC-SHA256 under a key
derived from the master key; a plain SHA-256 without one) when a process first connects. It re-anchored
in silence an anchor that was missing beside entries, one it did not recognize, and one written under another key
id -- an anchor deleted, garbled or forged with the key id ``nokey`` passed under a keyed install, and a chain
truncated with it was never noticed --; it never compared the anchored tip, so a chain rewritten at the anchored
height passed; and every verdict was one log line of one process, the anchor moved by the next append.

  * TA1 -- an anchor missing beside entries, keyed or not, is written into the chain as ``audit_anchor_missing``,
    a WARNING with the chain's height, whichever call opens the chain first; the chain then verifies, re-anchored
    past that entry, and the next process says nothing. Neither an entry nor an anchor (a first run) is nothing.
  * TA2 -- an anchor not written here (not five fields, not v2, a height that is no number, bytes that are not
    UTF-8 in any field) and one written under another key id (a key change, or one forged with ``nokey`` and a
    plain checksum under a keyed install) are each a WARNING entry, never silent; the forged one's height is said,
    never believed.
  * TA3 -- a chain rewritten and rehashed at the anchored height, as long or longer, is ``audit_anchor_rewritten``:
    CRITICAL under a key, a WARNING without one -- though the chain verifies.
  * TA4 -- a chain truncated (or replaced by an empty one) and a keyed anchor whose tag fails are written into the
    chain, CRITICAL under a key (an unkeyed checksum that fails is a WARNING), with the anchored and current heights;
    the entry outlives the anchor it was read from: after the next append, the next process sees an anchor that
    verifies and the entry is still there.
  * TA5 -- an anchor that cannot be looked at (a FIFO, one swapped in after it was looked at, a directory, a link
    -- to an anchor, looping, or leading nowhere --, a file longer than any anchor) is ``audit_anchor_not_checked``,
    a WARNING naming why; the first connection never waits on it.
  * TA6 -- the anchor is written whole and never through a link at its name: a reader that opened it before a save
    reads the earlier anchor whole, a file a link at its name points to is left as it was, and the anchor is 0600.
  * TA7 -- ``get_status`` reports what the check saw of the anchor: its verdict and whether it is keyed; an anchor
    that verifies, and a chain grown past it, write no entry.
  * TA8 -- the anchor never moves past a chain that was not compared with it: a comparison that fails is said in
    the chain once, runs again at the next connection and then names the truncation; a walk of the chain that fails
    (a row that does not decode) never keeps the anchor from being compared, nor its verdict from being written; a
    verdict whose first write fails is written at the next append; another writer appending while the check runs
    never makes the next start say the chain was rewritten, and a first entry landing while it reads is no missing
    anchor; two threads opening the chain at once write one verdict.
  * TA9 -- a process that holds no key where one is configured, beside an anchor written under one, writes no
    verdict and leaves the anchor as it is, its own entries included; the keyed process's next start finds it sound.
    Where no key is configured, such an anchor is another key's, said. A process whose check ran under a key never
    writes an anchor without it, its key failing to load at one append.
  * TA10 -- a chain truncated, put back from an older copy, or rewritten while a process runs is said at that
    process's next entry, CRITICAL under a key, and the anchor moves only past that verdict.

Local-only (the public distribution ships no tests). The chain module is loaded from source on the shared isolation
window, its store opened by plain sqlite3; the anchor's secret is set per instance, keyed with a key that is not in
the tree, or unkeyed. Each new instance stands for a new process opening the same chain.
"""

import hashlib
import logging
import os
import sqlite3
import stat
import sys
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

# Seconds per junit case, the parametrized ids included: the ladder reads this literal alone.
BUDGET_S = {
    "test_ta1_an_anchor_missing_beside_entries_is_written_into_the_chain[keyed-status]": 2.0,
    "test_ta1_an_anchor_missing_beside_entries_is_written_into_the_chain[keyed-append]": 2.0,
    "test_ta1_an_anchor_missing_beside_entries_is_written_into_the_chain[unkeyed-status]": 2.0,
    "test_ta1_an_anchor_missing_beside_entries_is_written_into_the_chain[unkeyed-append]": 2.0,
    "test_ta1_neither_an_entry_nor_an_anchor_is_a_first_run_and_says_nothing[keyed]": 2.0,
    "test_ta1_neither_an_entry_nor_an_anchor_is_a_first_run_and_says_nothing[unkeyed]": 2.0,
    "test_ta2_an_anchor_not_written_here_is_said_never_silent[empty]": 2.0,
    "test_ta2_an_anchor_not_written_here_is_said_never_silent[four_fields]": 2.0,
    "test_ta2_an_anchor_not_written_here_is_said_never_silent[garbage]": 2.0,
    "test_ta2_an_anchor_not_written_here_is_said_never_silent[height_not_a_number]": 2.0,
    "test_ta2_an_anchor_not_written_here_is_said_never_silent[not_utf8]": 2.0,
    "test_ta2_an_anchor_not_written_here_is_said_never_silent[not_utf8_key_id]": 2.0,
    "test_ta2_an_anchor_not_written_here_is_said_never_silent[not_utf8_tag]": 2.0,
    "test_ta2_an_anchor_not_written_here_is_said_never_silent[not_utf8_tip]": 2.0,
    "test_ta2_an_anchor_not_written_here_is_said_never_silent[v1]": 2.0,
    "test_ta2_an_anchor_under_another_key_is_said_and_its_height_never_believed[forged_nokey]": 2.0,
    "test_ta2_an_anchor_under_another_key_is_said_and_its_height_never_believed[other_key]": 2.0,
    "test_ta3_a_chain_rewritten_at_the_anchored_height_is_said_though_it_verifies[keyed-as_long]": 2.0,
    "test_ta3_a_chain_rewritten_at_the_anchored_height_is_said_though_it_verifies[keyed-grown]": 2.0,
    "test_ta3_a_chain_rewritten_at_the_anchored_height_is_said_though_it_verifies[unkeyed-as_long]": 2.0,
    "test_ta3_a_chain_rewritten_at_the_anchored_height_is_said_though_it_verifies[unkeyed-grown]": 2.0,
    "test_ta4_a_truncated_or_tampered_chain_is_written_into_it_and_outlives_the_anchor[truncated-keyed]": 2.0,
    "test_ta4_a_truncated_or_tampered_chain_is_written_into_it_and_outlives_the_anchor[truncated-unkeyed]": 2.0,
    "test_ta4_a_truncated_or_tampered_chain_is_written_into_it_and_outlives_the_anchor[replaced-keyed]": 2.0,
    "test_ta4_a_truncated_or_tampered_chain_is_written_into_it_and_outlives_the_anchor[replaced-unkeyed]": 2.0,
    "test_ta4_a_truncated_or_tampered_chain_is_written_into_it_and_outlives_the_anchor[tampered-keyed]": 2.0,
    "test_ta4_a_truncated_or_tampered_chain_is_written_into_it_and_outlives_the_anchor[tampered-unkeyed]": 2.0,
    "test_ta5_an_anchor_that_cannot_be_looked_at_is_not_checked_and_never_waited_on[fifo]": 2.0,
    "test_ta5_an_anchor_that_cannot_be_looked_at_is_not_checked_and_never_waited_on[fifo_swapped_in]": 2.0,
    "test_ta5_an_anchor_that_cannot_be_looked_at_is_not_checked_and_never_waited_on[directory]": 2.0,
    "test_ta5_an_anchor_that_cannot_be_looked_at_is_not_checked_and_never_waited_on[link_to_an_anchor]": 2.0,
    "test_ta5_an_anchor_that_cannot_be_looked_at_is_not_checked_and_never_waited_on[looping_link]": 2.0,
    "test_ta5_an_anchor_that_cannot_be_looked_at_is_not_checked_and_never_waited_on[dangling_link]": 2.0,
    "test_ta5_an_anchor_that_cannot_be_looked_at_is_not_checked_and_never_waited_on[oversize]": 2.0,
    "test_ta6_the_anchor_is_written_whole_and_never_through_a_link_at_its_name": 2.0,
    "test_ta7_the_status_reports_the_anchor_and_a_sound_one_writes_nothing[keyed]": 2.0,
    "test_ta7_the_status_reports_the_anchor_and_a_sound_one_writes_nothing[unkeyed]": 2.0,
    "test_ta7_a_first_run_reports_neither_an_entry_nor_an_anchor[keyed]": 2.0,
    "test_ta7_a_first_run_reports_neither_an_entry_nor_an_anchor[unkeyed]": 2.0,
    "test_ta8_a_check_that_fails_is_said_once_runs_again_and_then_names_the_truncation[read]": 2.0,
    "test_ta8_a_check_that_fails_is_said_once_runs_again_and_then_names_the_truncation[append]": 2.0,
    "test_ta8_a_verdict_whose_first_write_fails_is_written_at_the_next_append": 2.0,
    "test_ta8_another_writer_appending_while_the_check_runs_never_makes_the_next_start_say_rewritten": 2.0,
    "test_ta8_a_walk_that_fails_never_keeps_the_anchor_from_being_compared": 2.0,
    "test_ta8_a_first_entry_landing_while_the_check_reads_is_growth_not_a_missing_anchor": 2.0,
    "test_ta8_two_threads_opening_the_chain_at_once_write_one_verdict": 2.0,
    "test_ta9_a_process_without_the_key_leaves_a_keyed_anchor_as_it_is[status]": 2.0,
    "test_ta9_a_process_without_the_key_leaves_a_keyed_anchor_as_it_is[append]": 2.0,
    "test_ta9_where_no_key_is_configured_an_anchor_under_one_is_another_keys_and_said": 2.0,
    "test_ta9_a_key_that_does_not_load_at_one_append_never_turns_the_anchor_into_a_checksum": 2.0,
    "test_ta10_a_chain_changed_while_a_process_runs_is_said_at_its_next_entry[rows_deleted-keyed]": 2.0,
    "test_ta10_a_chain_changed_while_a_process_runs_is_said_at_its_next_entry[rows_deleted-unkeyed]": 2.0,
    "test_ta10_a_chain_changed_while_a_process_runs_is_said_at_its_next_entry[older_copy_put_back-keyed]": 2.0,
    "test_ta10_a_chain_changed_while_a_process_runs_is_said_at_its_next_entry[older_copy_put_back-unkeyed]": 2.0,
    "test_ta10_a_chain_changed_while_a_process_runs_is_said_at_its_next_entry[rewritten-keyed]": 2.0,
    "test_ta10_a_chain_changed_while_a_process_runs_is_said_at_its_next_entry[rewritten-unkeyed]": 2.0,
}

_KEY = hashlib.sha256(b"ta-contract-key-not-in-the-tree").digest()
_KEY_ID = "7a11c0de5eed0001"
_OTHER_KEY = hashlib.sha256(b"ta-contract-other-key").digest()
_OTHER_KEY_ID = "7a11c0de5eed0002"


class _InsertRefused:
    """A connection whose next INSERT fails, as a store that cannot take an entry does once."""

    def __init__(self, conn, hooks):
        self._conn, self._hooks = conn, hooks

    def execute(self, sql, *args):
        if sql.lstrip().startswith("INSERT") and self._hooks.get("refuse_insert", 0) > 0:
            self._hooks["refuse_insert"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.fixture
def hooks():
    """What the store stand-in does: ``on_connect()`` before each connection, ``refuse_insert`` INSERTs refused."""
    return {}


@pytest.fixture
def sal(hooks):
    """The chain module from source, its store opened by plain sqlite3."""
    db_utils = types.ModuleType("opti_oignon.db_utils")

    def safe_connect(path, *, check_same_thread=True, timeout=5.0, read_only=False):
        if hooks.get("on_connect") is not None:
            hooks["on_connect"]()
        conn = sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=timeout)
        return _InsertRefused(conn, hooks) if hooks.get("refuse_insert") else conn

    db_utils.safe_connect = safe_connect
    loaded, restore = isolate(targets={"opti_oignon.signed_audit_log": source("signed_audit_log.py")},
                              seeded={"opti_oignon.db_utils": db_utils})
    try:
        yield loaded["opti_oignon.signed_audit_log"]
    finally:
        restore()


def _open(sal, db, keyed=True, key=None):
    """A new instance on ``db`` -- a new process --, its anchor's secret set: keyed (``key``: ``(bytes, id)``), or
    not."""
    log = sal.SignedAuditLog(db)
    secret = key if key is not None else ((_KEY, _KEY_ID) if keyed else (None, "nokey"))
    log._anchor_secret = lambda: secret
    return log


def _chain_of(sal, tmp_path, n, keyed=True, key=None):
    """A chain of ``n`` entries, anchored by the process that wrote them; its store's path."""
    db = tmp_path / "audit" / "audit_chain.db"
    log = _open(sal, db, keyed, key)
    for i in range(n):
        log.append_event("test_event", source="contracts", action=f"action {i}", details={"i": i})
    return db


def _anchor(db):
    return db.parent / ".audit_chain_anchor"


def _events(log):
    """The chain's entries, oldest first."""
    return sorted(log.get_events(limit=500), key=lambda event: event["id"])


def _said(log):
    """The anchor verdicts written into the chain: ``[(event_type, severity, details)]``, oldest first."""
    return [(e["event_type"], e["severity"], e["details"]) for e in _events(log)
            if e["event_type"].startswith("audit_anchor_")]


def _seen(log):
    """What the status says the check saw of the anchor; empty where it says nothing."""
    return log.get_status().get("anchor") or {}


def _rehash_from(sal, db, first):
    """Rehash every entry from ``first`` on, as one who rewrote them would: the chain verifies again."""
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT id, timestamp, event_type, source, action, severity, details_json, prev_hash "
                            "FROM audit_chain ORDER BY id").fetchall()
        prev = None
        for eid, ts, etype, src, act, sev, djson, phash in rows:
            if eid >= first:
                phash = prev
                ehash = sal._compute_entry_hash(eid, ts, etype, src, act, sev, djson, phash)
                conn.execute("UPDATE audit_chain SET prev_hash = ?, entry_hash = ? WHERE id = ?", (phash, ehash, eid))
            prev = conn.execute("SELECT entry_hash FROM audit_chain WHERE id = ?", (eid,)).fetchone()[0]
        conn.commit()
    finally:
        conn.close()


def _append_raw(sal, db, action):
    """One entry chained at the end by hand, the anchor not moved: a chain grown by one who rewrote it."""
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute("SELECT id, entry_hash FROM audit_chain ORDER BY id DESC LIMIT 1").fetchone()
        eid, prev = row if row else (0, "0" * 128)
        ehash = sal._compute_entry_hash(eid + 1, 1.0, "test_event", "contracts", action, "INFO", "{}", prev)
        conn.execute("INSERT INTO audit_chain (id, timestamp, event_type, source, action, severity, details_json, "
                     "prev_hash, entry_hash) VALUES (?, 1.0, 'test_event', 'contracts', ?, 'INFO', '{}', ?, ?)",
                     (eid + 1, action, prev, ehash))
        conn.commit()
    finally:
        conn.close()


def _first_connection_in_a_thread(log, call, seconds=5.0):
    """``call(log)`` on a thread of its own, waited on for ``seconds``: ``(still running, what it returned)``."""
    got = {}
    worker = threading.Thread(target=lambda: got.setdefault("value", call(log)), daemon=True)
    worker.start()
    worker.join(seconds)
    return worker.is_alive(), got.get("value", "nothing returned")


# ---------------------------------------------------------------------------
# TA1 -- an anchor missing beside entries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("opened_by", ["status", "append"])
@pytest.mark.parametrize("keyed", [True, False], ids=["keyed", "unkeyed"])
def test_ta1_an_anchor_missing_beside_entries_is_written_into_the_chain(sal, tmp_path, keyed, opened_by, caplog):
    db = _chain_of(sal, tmp_path, 3, keyed)
    _anchor(db).unlink()
    caplog.set_level(logging.WARNING, logger="opti_oignon.signed_audit_log")
    log = _open(sal, db, keyed)
    if opened_by == "status":
        log.get_status()
    else:
        log.append_event("next_event", source="contracts")
    said = _said(log)
    assert [(t, s) for t, s, _ in said] == [("audit_anchor_missing", "WARNING")], said
    assert said[0][2]["entries"] == 3 and said[0][2]["keyed"] is keyed, said[0][2]
    if opened_by == "append":
        assert [e["event_type"] for e in _events(log)][-2:] == ["next_event", "audit_anchor_missing"], \
            "chained right after the entry whose opening saw it"
    assert any("missing" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records), \
        "said in the log too"
    seen = _seen(log)
    assert seen.get("verdict") == "missing" and seen.get("keyed") is keyed, seen
    total = len(_events(log))
    assert log.verify_chain() == (True, None, total)
    assert _anchor(db).is_file(), "re-anchored past the entry that says why"
    again = _open(sal, db, keyed)
    assert len(_events(again)) == total, "the next process sees an anchor that verifies, and says nothing"
    assert _seen(again).get("verdict") == "ok"


@pytest.mark.parametrize("keyed", [True, False], ids=["keyed", "unkeyed"])
def test_ta1_neither_an_entry_nor_an_anchor_is_a_first_run_and_says_nothing(sal, tmp_path, keyed):
    db = tmp_path / "audit" / "audit_chain.db"
    log = _open(sal, db, keyed)
    assert log.get_status()["total_entries"] == 0 and _said(log) == []
    assert not _anchor(db).exists()


# ---------------------------------------------------------------------------
# TA2 -- an anchor not written here, or under another key
# ---------------------------------------------------------------------------
_NOT_WRITTEN_HERE = {
    "garbage": b"not an anchor",
    "four_fields": b"v2|7a11c0de5eed0001|3|abc",
    "v1": b"v1|7a11c0de5eed0001|3|abc|def",
    "height_not_a_number": b"v2|7a11c0de5eed0001|three|abc|def",
    "not_utf8": b"v\xff|7a11c0de5eed0001|3|abc|def",
    "not_utf8_key_id": b"v2|7a11c0de5eed\xff\xfe|3|abc|def",
    "not_utf8_tip": b"v2|7a11c0de5eed0001|3|ab\xff|def",
    "not_utf8_tag": b"v2|7a11c0de5eed0001|3|abc|de\xff",
    "empty": b"",
}


@pytest.mark.parametrize("written", sorted(_NOT_WRITTEN_HERE))
def test_ta2_an_anchor_not_written_here_is_said_never_silent(sal, tmp_path, written):
    db = _chain_of(sal, tmp_path, 3)
    _anchor(db).write_bytes(_NOT_WRITTEN_HERE[written])
    log = _open(sal, db)
    said = _said(log)
    assert [(t, s) for t, s, _ in said] == [("audit_anchor_unrecognized", "WARNING")], said
    assert said[0][2]["entries"] == 3
    assert _seen(log).get("verdict") == "unrecognized"
    again = _open(sal, db)
    assert _said(again) == said and _seen(again).get("verdict") == "ok", "re-anchored past the entry"


@pytest.mark.parametrize("anchored_by", ["forged_nokey", "other_key"])
def test_ta2_an_anchor_under_another_key_is_said_and_its_height_never_believed(sal, tmp_path, anchored_by):
    if anchored_by == "forged_nokey":
        db = _chain_of(sal, tmp_path, 4)
        # Made without the key: a plain checksum, the key id of an unkeyed install, a height that hides nothing.
        content = "v2|nokey|1|" + "0" * 128
        _anchor(db).write_text(f"{content}|{hashlib.sha256(content.encode()).hexdigest()}", encoding="utf-8")
        anchored = 1
    else:
        # A key change: the chain was written and anchored under another key.
        db = _chain_of(sal, tmp_path, 4, key=(_OTHER_KEY, _OTHER_KEY_ID))
        anchored = 4
    log = _open(sal, db)
    said = _said(log)
    assert [(t, s) for t, s, _ in said] == [("audit_anchor_other_key", "WARNING")], said
    details = said[0][2]
    assert details["anchored_entries"] == anchored and details["entries"] == 4 and details["key_id"] == _KEY_ID, \
        details
    assert details["anchored_key_id"] == ("nokey" if anchored_by == "forged_nokey" else _OTHER_KEY_ID)
    seen = _seen(log)
    assert seen.get("verdict") == "other_key" and seen.get("keyed") is True, seen
    assert _seen(_open(sal, db)).get("verdict") == "ok"


# ---------------------------------------------------------------------------
# TA3 -- a chain rewritten at the anchored height
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grown", [False, True], ids=["as_long", "grown"])
@pytest.mark.parametrize("keyed", [True, False], ids=["keyed", "unkeyed"])
def test_ta3_a_chain_rewritten_at_the_anchored_height_is_said_though_it_verifies(sal, tmp_path, keyed, grown):
    db = _chain_of(sal, tmp_path, 3, keyed)
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE audit_chain SET action = 'nothing happened' WHERE id = 2")
    conn.commit()
    conn.close()
    _rehash_from(sal, db, 2)
    if grown:
        _append_raw(sal, db, "after the rewrite")
    log = _open(sal, db, keyed)
    height = 4 if grown else 3
    said = _said(log)
    severity = "CRITICAL" if keyed else "WARNING"
    assert [(t, s) for t, s, _ in said] == [("audit_anchor_rewritten", severity)], said
    assert said[0][2]["anchored_entries"] == 3 and said[0][2]["entries"] == height, said[0][2]
    assert log.verify_chain() == (True, None, height + 1), "the rewritten chain verifies, the entry that says so too"
    assert _seen(log).get("verdict") == "rewritten"


# ---------------------------------------------------------------------------
# TA4 -- truncated, replaced or tampered: written into the chain, and outliving the anchor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("keyed", [True, False], ids=["keyed", "unkeyed"])
@pytest.mark.parametrize("how", ["truncated", "replaced", "tampered"])
def test_ta4_a_truncated_or_tampered_chain_is_written_into_it_and_outlives_the_anchor(sal, tmp_path, how, keyed,
                                                                                         caplog):
    db = _chain_of(sal, tmp_path, 5, keyed)
    if how == "truncated":
        conn = sqlite3.connect(str(db))
        conn.execute("DELETE FROM audit_chain WHERE id > 3")
        conn.commit()
        conn.close()
        verdict, entries, anchored = "truncated", 3, 5
    elif how == "replaced":
        db.unlink()
        verdict, entries, anchored = "truncated", 0, 5
    else:
        text = _anchor(db).read_text(encoding="utf-8").split("|")
        text[2] = "9"
        _anchor(db).write_text("|".join(text), encoding="utf-8")
        verdict, entries, anchored = ("tampered" if keyed else "checksum_mismatch"), 5, 9
    caplog.set_level(logging.WARNING, logger="opti_oignon.signed_audit_log")
    log = _open(sal, db, keyed)
    severity = "CRITICAL" if keyed and verdict != "checksum_mismatch" else "WARNING"
    said = _said(log)
    assert [(t, s) for t, s, _ in said] == [(f"audit_anchor_{verdict}", severity)], said
    assert said[0][2]["entries"] == entries and said[0][2]["anchored_entries"] == anchored, said[0][2]
    level = logging.CRITICAL if severity == "CRITICAL" else logging.WARNING
    assert any(r.levelno == level and r.name == "opti_oignon.signed_audit_log" for r in caplog.records)
    assert _seen(log).get("verdict") == verdict
    log.append_event("after", source="contracts")
    later = _open(sal, db, keyed)
    assert [t for t, _, _ in _said(later)] == [f"audit_anchor_{verdict}"], "the entry outlives the anchor"
    assert _seen(later).get("verdict") == "ok", "the anchor moved past the entry"


# ---------------------------------------------------------------------------
# TA5 -- an anchor that cannot be looked at
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("anchor_is", ["fifo", "fifo_swapped_in", "directory", "link_to_an_anchor", "looping_link",
                                       "dangling_link", "oversize"])
def test_ta5_an_anchor_that_cannot_be_looked_at_is_not_checked_and_never_waited_on(sal, tmp_path, anchor_is,
                                                                                   monkeypatch):
    db = _chain_of(sal, tmp_path, 2)
    path = _anchor(db)
    if anchor_is == "link_to_an_anchor":
        moved = tmp_path / "moved_anchor"
        os.replace(path, moved)
        os.symlink(moved, path)
    elif anchor_is == "oversize":
        path.write_bytes(path.read_bytes() + b" " * 5000)
    else:
        path.unlink()
    if anchor_is == "fifo":
        os.mkfifo(path)
    elif anchor_is == "fifo_swapped_in":
        # Looked at as a regular file, a FIFO by the time it is opened.
        os.mkfifo(path)
        regular = os.stat(tmp_path / "audit" / "audit_chain.db")
        real_lstat = os.lstat
        monkeypatch.setattr(os, "lstat", lambda p, *a, **k: regular if str(p) == str(path) else real_lstat(p, *a, **k))
    elif anchor_is == "directory":
        path.mkdir()
    elif anchor_is == "looping_link":
        os.symlink(path.name, path)
    elif anchor_is == "dangling_link":
        os.symlink(tmp_path / "nowhere", path)
    log = _open(sal, db)
    running, status = _first_connection_in_a_thread(log, lambda chain: chain.get_status())
    assert not running, "the first connection waited on the anchor"
    said = _said(log)
    assert [(t, s) for t, s, _ in said] == [("audit_anchor_not_checked", "WARNING")], said
    assert said[0][2]["reason"] and said[0][2]["entries"] == 2, said[0][2]
    seen = status.get("anchor") or {}
    assert seen.get("verdict") == "not_checked" and seen.get("reason") == said[0][2]["reason"], seen


# ---------------------------------------------------------------------------
# TA6 -- written whole, never through a link
# ---------------------------------------------------------------------------
def test_ta6_the_anchor_is_written_whole_and_never_through_a_link_at_its_name(sal, tmp_path):
    db = _chain_of(sal, tmp_path, 2)
    log = _open(sal, db)
    before = _anchor(db).read_bytes()
    with open(_anchor(db), "rb") as reader:
        log.append_event("moves_the_anchor", source="contracts")
        assert reader.read() == before, "a reader that opened the anchor before a save reads it whole"
    assert _anchor(db).read_bytes() != before, "control: the anchor moved"
    assert stat.S_IMODE(os.stat(_anchor(db)).st_mode) == 0o600

    victim = tmp_path / "victim.txt"
    victim.write_text("left as it was", encoding="utf-8")
    _anchor(db).unlink()
    os.symlink(victim, _anchor(db))
    later = _open(sal, db)
    later.append_event("after_the_link", source="contracts")
    assert victim.read_text(encoding="utf-8") == "left as it was", "written through the link at its name"
    assert not os.path.islink(_anchor(db)) and stat.S_ISREG(os.lstat(_anchor(db)).st_mode)
    assert [t for t, _, _ in _said(later)] == ["audit_anchor_not_checked"], "a link is never followed"


# ---------------------------------------------------------------------------
# TA7 -- the status says what the check saw
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("keyed", [True, False], ids=["keyed", "unkeyed"])
def test_ta7_the_status_reports_the_anchor_and_a_sound_one_writes_nothing(sal, tmp_path, keyed):
    db = _chain_of(sal, tmp_path, 3, keyed)
    log = _open(sal, db, keyed)
    status = log.get_status()
    assert status.get("anchor") == {"verdict": "ok", "keyed": keyed, "entries": 3, "anchored_entries": 3}, status
    assert status["total_entries"] == 3 and _said(log) == []

    _append_raw(sal, db, "grown by another writer")
    grown = _open(sal, db, keyed)
    assert _seen(grown) == {"verdict": "grown", "keyed": keyed, "entries": 4, "anchored_entries": 3}
    assert _said(grown) == [], "a chain grown past its anchor writes nothing"
    assert _seen(_open(sal, db, keyed)).get("verdict") == "ok", "re-anchored at the grown height"


@pytest.mark.parametrize("keyed", [True, False], ids=["keyed", "unkeyed"])
def test_ta7_a_first_run_reports_neither_an_entry_nor_an_anchor(sal, tmp_path, keyed):
    log = _open(sal, tmp_path / "audit" / "audit_chain.db", keyed)
    assert _seen(log) == {"verdict": "empty", "keyed": keyed, "entries": 0}


# ---------------------------------------------------------------------------
# TA8 -- the anchor never moves past a chain that was not compared with it
# ---------------------------------------------------------------------------
def _keep_first(db, n):
    """The chain truncated to its first ``n`` entries, the anchor left as it was."""
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM audit_chain WHERE id > ?", (n,))
    conn.commit()
    conn.close()


@pytest.mark.parametrize("opened_by", ["read", "append"])
def test_ta8_a_check_that_fails_is_said_once_runs_again_and_then_names_the_truncation(sal, tmp_path, opened_by):
    db = _chain_of(sal, tmp_path, 5)
    _keep_first(db, 2)
    before = _anchor(db).read_bytes()
    log = _open(sal, db)
    real_secret, fails = log._anchor_secret, [2]

    def secret_failing_twice():
        if fails[0]:
            fails[0] -= 1
            raise sqlite3.OperationalError("database is locked")
        return real_secret()

    log._anchor_secret = secret_failing_twice
    if opened_by == "read":
        log.get_events(limit=1)
    else:
        log.append_event("during", source="contracts")
    assert _anchor(db).read_bytes() == before, "the anchor waits for a check that concluded"
    log.get_events(limit=1)
    assert fails == [0] and _anchor(db).read_bytes() == before
    log.get_events(limit=1)
    said = _said(log)
    assert [(t, s) for t, s, _ in said] == [("audit_anchor_not_checked", "WARNING"),
                                            ("audit_anchor_truncated", "CRITICAL")], said
    assert "database is locked" in said[0][2]["reason"] and said[1][2]["anchored_entries"] == 5, said
    later = _open(sal, db)
    assert _said(later) == said and _seen(later).get("verdict") == "ok"


def test_ta8_a_verdict_whose_first_write_fails_is_written_at_the_next_append(sal, hooks, tmp_path):
    db = _chain_of(sal, tmp_path, 5)
    _keep_first(db, 2)
    before = _anchor(db).read_bytes()
    log = _open(sal, db)
    hooks["refuse_insert"] = 1
    with pytest.raises(sqlite3.OperationalError):
        log.append_event("refused", source="contracts")
    assert hooks.pop("refuse_insert") == 0, "control: the INSERT was refused"
    assert _anchor(db).read_bytes() == before, "the anchor waits for the verdict"
    log.append_event("next", source="contracts")
    said = _said(log)
    assert [(t, s) for t, s, _ in said] == [("audit_anchor_truncated", "CRITICAL")], said
    assert [e["event_type"] for e in _events(log)][-2:] == ["next", "audit_anchor_truncated"]
    assert _seen(_open(sal, db)).get("verdict") == "ok"


def test_ta8_another_writer_appending_while_the_check_runs_never_makes_the_next_start_say_rewritten(sal, hooks,
                                                                                                    tmp_path):
    db = _chain_of(sal, tmp_path, 3)
    _append_raw(sal, db, "one past the anchor")
    written = []

    def another_writer():
        if len(written) < 8:
            written.append(len(written))
            _append_raw(sal, db, f"by another writer {len(written)}")

    hooks["on_connect"] = another_writer
    log = _open(sal, db)
    log.get_status()
    hooks.pop("on_connect")
    assert written, "control: the other writer appended while the check ran"
    later = _open(sal, db)
    assert _said(log) == [] and _said(later) == [], "nothing was rewritten"
    assert _seen(later).get("verdict") in ("ok", "grown")


def test_ta8_a_walk_that_fails_never_keeps_the_anchor_from_being_compared(sal, tmp_path):
    db = _chain_of(sal, tmp_path, 6)
    _keep_first(db, 4)
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE audit_chain SET action = CAST(X'FF' AS TEXT) WHERE id = 2")
    conn.commit()
    conn.close()
    log = _open(sal, db)
    started = time.monotonic()
    running, count = _first_connection_in_a_thread(log, lambda chain: chain.entry_count())
    assert not running and time.monotonic() - started < 2.0, "the first connection waited on its own walk"
    conn = sqlite3.connect(str(db))
    said = conn.execute("SELECT event_type, severity FROM audit_chain WHERE event_type LIKE 'audit_anchor_%' "
                        "ORDER BY id").fetchall()
    conn.close()
    assert said == [("audit_anchor_truncated", "CRITICAL")], said
    assert count == 5, "the four entries left and the verdict"


def test_ta8_a_first_entry_landing_while_the_check_reads_is_growth_not_a_missing_anchor(sal, tmp_path, monkeypatch):
    db = tmp_path / "audit" / "audit_chain.db"
    real_read, landed = getattr(sal, "_read_anchor_text", None), []

    def read_then_another_writer(path):
        text = real_read(path)
        if not landed:
            landed.append(True)
            _append_raw(sal, db, "the first entry, by another writer")
        return text

    # The anchor's reader, where the module has one: an entry lands right after the anchor was looked at.
    monkeypatch.setattr(sal, "_read_anchor_text", read_then_another_writer, raising=False)
    log = _open(sal, db)
    log.get_status()
    monkeypatch.undo()
    assert landed, "control: the entry landed while the check read"
    assert _said(log) == [] and _seen(log).get("verdict") == "empty"
    later = _open(sal, db)
    assert _said(later) == [] and _seen(later).get("verdict") == "ok", "anchored at the entry that landed"


def test_ta8_two_threads_opening_the_chain_at_once_write_one_verdict(sal, hooks, tmp_path, monkeypatch):
    db = _chain_of(sal, tmp_path, 3)
    _anchor(db).unlink()
    arrived, barrier = [], threading.Barrier(2, timeout=5)
    real_read, readers, finished, met = getattr(sal, "_read_anchor_text", None), [], [], threading.Condition()

    def together():
        if len(arrived) < 2:
            arrived.append(threading.get_ident())
            barrier.wait()

    def read_together(path):
        with met:
            readers.append(threading.get_ident())
            met.notify_all()
            met.wait_for(lambda: len(readers) > 1 or finished, timeout=5)
        return real_read(path)

    def opens_the_chain():
        try:
            log.get_status()
        finally:
            with met:
                finished.append(threading.get_ident())
                met.notify_all()

    hooks["on_connect"] = together
    # A thread that reads the anchor waits for the other to read it too, or to be done: a second check reads it
    # before the first has written it, at every run.
    monkeypatch.setattr(sal, "_read_anchor_text", read_together, raising=False)
    log = _open(sal, db)
    threads = [threading.Thread(target=opens_the_chain, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    monkeypatch.undo()
    hooks.pop("on_connect")
    assert len(set(arrived)) == 2 and not any(thread.is_alive() for thread in threads), arrived
    said = _said(log)
    assert [(t, s) for t, s, _ in said] == [("audit_anchor_missing", "WARNING")], said


# ---------------------------------------------------------------------------
# TA9 -- a process without the key leaves a keyed anchor as it is
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("opened_by", ["status", "append"])
def test_ta9_a_process_without_the_key_leaves_a_keyed_anchor_as_it_is(sal, tmp_path, opened_by):
    db = _chain_of(sal, tmp_path, 3)
    before = _anchor(db).read_bytes()
    keyless = _open(sal, db, keyed=False)
    keyless._key_configured = lambda: True
    if opened_by == "status":
        keyless.get_status()
    else:
        keyless.append_event("by_a_process_without_the_key", source="contracts")
    assert _said(keyless) == [], "no verdict from a process that cannot verify"
    assert _anchor(db).read_bytes() == before, "the keyed anchor left as it is"
    seen = _seen(keyless)
    assert seen.get("verdict") == "key_not_loaded" and seen.get("keyed") is False, seen
    keyed = _open(sal, db)
    assert _said(keyed) == [] and _seen(keyed).get("verdict") == ("grown" if opened_by == "append" else "ok")


def test_ta9_where_no_key_is_configured_an_anchor_under_one_is_another_keys_and_said(sal, tmp_path):
    db = _chain_of(sal, tmp_path, 3)
    keyless = _open(sal, db, keyed=False)
    keyless._key_configured = lambda: False
    said = _said(keyless)
    assert [(t, s) for t, s, _ in said] == [("audit_anchor_other_key", "WARNING")], said
    assert said[0][2]["anchored_key_id"] == _KEY_ID and said[0][2]["key_id"] == "nokey"
    again = _open(sal, db, keyed=False)
    again._key_configured = lambda: False
    assert _said(again) == said and _seen(again).get("verdict") == "ok", "re-anchored without a key"


def test_ta9_a_key_that_does_not_load_at_one_append_never_turns_the_anchor_into_a_checksum(sal, tmp_path):
    db = _chain_of(sal, tmp_path, 3)
    log = _open(sal, db)
    assert _seen(log).get("verdict") == "ok"
    log._anchor_secret = lambda: (None, "nokey")
    log.append_event("while_the_key_does_not_load", source="contracts")
    assert _anchor(db).read_text(encoding="utf-8").split("|")[1] == _KEY_ID, "the anchor stays keyed"
    later = _open(sal, db)
    assert _said(later) == [] and _seen(later).get("verdict") == "grown"


# ---------------------------------------------------------------------------
# TA10 -- truncated or rewritten while a process runs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("keyed", [True, False], ids=["keyed", "unkeyed"])
@pytest.mark.parametrize("how", ["rows_deleted", "older_copy_put_back", "rewritten"])
def test_ta10_a_chain_changed_while_a_process_runs_is_said_at_its_next_entry(sal, tmp_path, how, keyed):
    db = _chain_of(sal, tmp_path, 2, keyed)
    older = tmp_path / "older_copy.db"
    older.write_bytes(db.read_bytes())
    server = _open(sal, db, keyed)
    for name in ("server_started", "a_login", "another_login"):
        server.append_event(name, source="contracts")
    if how == "rows_deleted":
        _keep_first(db, 2)
        verdict, entries = "truncated", 2
    elif how == "older_copy_put_back":
        db.write_bytes(older.read_bytes())
        verdict, entries = "truncated", 2
    else:
        conn = sqlite3.connect(str(db))
        conn.execute("UPDATE audit_chain SET action = 'nothing happened' WHERE id = 3")
        conn.commit()
        conn.close()
        _rehash_from(sal, db, 3)
        verdict, entries = "rewritten", 5
    server.append_event("next_login", source="contracts")
    severity = "CRITICAL" if keyed else "WARNING"
    said = _said(server)
    assert [(t, s) for t, s, _ in said] == [(f"audit_anchor_{verdict}", severity)], said
    assert said[0][2]["anchored_entries"] == 5 and said[0][2]["entries"] == entries, said[0][2]
    assert [e["event_type"] for e in _events(server)][-2:] == ["next_login", f"audit_anchor_{verdict}"]
    assert _seen(server).get("verdict") == verdict
    later = _open(sal, db, keyed)
    assert _said(later) == said and _seen(later).get("verdict") == "ok", "the anchor moved past the verdict alone"
