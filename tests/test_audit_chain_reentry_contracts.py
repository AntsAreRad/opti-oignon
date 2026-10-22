#!/usr/bin/env python3
"""Contracts for the audit chain asked again by the thread that is writing an entry.

Opening the chain's store asks the security mode, and a mode that fails
secure records why -- through the chain, on the same thread, while that
thread holds the chain. The chain never makes that thread wait on itself.

  * JR1 -- CHAINED RIGHT AFTER, NEVER WAITED ON: an entry asked for on the
    thread that is writing another -- while the store opens, a fresh store
    or one with entries already -- is chained right after the entry being
    written, in the order asked; the call that asked returns at once, with
    None; the chain verifies.
  * JR2 -- ITS LEADER NOT WRITTEN, SAID BY NAME: when the entry being
    written is not, the entry asked meanwhile is said as not written, by its
    event type and action, never dropped in silence; the next append on the
    same thread is written at once.
  * JR3 -- ANOTHER THREAD WAITS ITS TURN: an entry asked for on another
    thread while one is being written waits, then is chained after it, and
    returns its id.
  * JR4 -- A FRESH STORE OPENED BY A READ: an entry recorded while a read
    opens a store that has no table yet finds the table made, and is
    written; a store whose first opening fails, or whose table is refused
    the first time, makes its table at the next call.

The chain module is loaded from source in an isolation window; its store is
opened by a stand-in for ``safe_connect`` that calls a hook before each
connection, as the security mode's read does in the real one.
"""

import logging
import sqlite3
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _window(state):
    """The chain module from source; its store opened by plain sqlite3 after ``state["hook"]()``, when set."""
    db_utils = types.ModuleType("opti_oignon.db_utils")

    def safe_connect(path, *, check_same_thread=True, timeout=5.0, read_only=False):
        hook = state.get("hook")
        if hook is not None:
            hook()
        conn = sqlite3.connect(str(path), check_same_thread=check_same_thread, timeout=timeout)
        return _TableRefusedOnce(conn, state) if state.get("table_refused") else conn

    db_utils.safe_connect = safe_connect
    loaded, restore = isolate(targets={"opti_oignon.signed_audit_log": source("signed_audit_log.py")},
                              seeded={"opti_oignon.db_utils": db_utils})
    return loaded["opti_oignon.signed_audit_log"], restore


class _TableRefusedOnce:
    """A connection whose first CREATE TABLE fails, as a store that cannot make its schema does once."""

    def __init__(self, conn, state):
        self._conn, self._state = conn, state

    def execute(self, sql, *args):
        if "CREATE TABLE" in sql and self._state.pop("table_refused", False):
            raise sqlite3.OperationalError("the table is not made")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _in_a_thread(call, name="writer"):
    """``call()`` on a thread of its own, waited on for ten seconds: ``(still running, what it returned)``."""
    got = {}
    worker = threading.Thread(target=lambda: got.setdefault("value", call()), name=name, daemon=True)
    worker.start()
    worker.join(10)
    return worker.is_alive(), got.get("value", "nothing returned")


def _types(log):
    """The chain's event types, oldest first."""
    return [event["event_type"] for event in sorted(log.get_events(limit=50), key=lambda event: event["id"])]


# ---------------------------------------------------------------------------
# JR1 -- chained right after, never waited on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("before", [0, 2])
def test_jr1_an_entry_asked_on_the_writing_thread_is_chained_right_after_never_waited_on(tmp_path, before):
    state = {}
    sal, restore = _window(state)
    try:
        log = sal.SignedAuditLog(tmp_path / "audit" / "audit_chain.db")
        earlier = [f"earlier_{n}" for n in range(before)]
        for name in earlier:
            log.append_event(name, source="test")
        asked = []

        def ask_once():
            if not asked:
                asked.append(log.append_event("inner_event", source="test", action="inner"))

        state["hook"] = ask_once
        running, outer = _in_a_thread(lambda: log.append_event("outer_event", source="test", action="outer"))
        state.pop("hook")
        assert not running, "the thread writing an entry waited on itself"
        assert outer == before + 1 and asked == [None], (outer, asked)
        assert _types(log) == [*earlier, "outer_event", "inner_event"], "chained right after, in the order asked"
        assert log.verify_chain() == (True, None, before + 2)
        stamps = [event["timestamp"] for event in sorted(log.get_events(limit=50), key=lambda event: event["id"])]
        assert stamps == sorted(stamps), "ids and times agree"
    finally:
        restore()


# ---------------------------------------------------------------------------
# JR2 -- its leader not written, said by name
# ---------------------------------------------------------------------------
def test_jr2_an_entry_whose_leader_is_not_written_is_said_by_name_and_the_next_append_stands(tmp_path, caplog):
    state = {}
    sal, restore = _window(state)
    try:
        log = sal.SignedAuditLog(tmp_path / "audit" / "audit_chain.db")
        assert log.append_event("first", source="test") == 1
        asked = []

        def ask_then_fail():
            state.pop("hook")
            asked.append(log.append_event("inner_event", source="test", action="inner"))
            raise sqlite3.OperationalError("the store does not open")

        state["hook"] = ask_then_fail
        caplog.set_level(logging.ERROR, logger="opti_oignon.signed_audit_log")
        with pytest.raises(sqlite3.OperationalError):
            log.append_event("outer_event", source="test", action="outer")
        assert asked == [None]
        said = [record.getMessage() for record in caplog.records
                if "AUDIT CHAIN APPEND FAILED" in record.getMessage()]
        assert len(said) == 1 and "event_type=inner_event" in said[0] and "action=inner" in said[0], said
        assert _types(log) == ["first"], "neither entry is in the chain"
        assert log.append_event("after", source="test") == 2, "the next append on the same thread is written at once"
        assert _types(log) == ["first", "after"] and log.verify_chain() == (True, None, 2)
    finally:
        restore()


# ---------------------------------------------------------------------------
# JR3 -- another thread waits its turn
# ---------------------------------------------------------------------------
def test_jr3_an_entry_asked_on_another_thread_waits_its_turn_and_is_chained_after(tmp_path):
    state = {}
    sal, restore = _window(state)
    try:
        log = sal.SignedAuditLog(tmp_path / "audit" / "audit_chain.db")
        assert log.append_event("first", source="test") == 1
        opening, release = threading.Event(), threading.Event()

        def hold_the_writer():
            if threading.current_thread().name == "writer":
                state.pop("hook")
                opening.set()
                release.wait(10)

        state["hook"] = hold_the_writer
        got = {}
        writer = threading.Thread(target=lambda: got.setdefault("writer", log.append_event("outer_event", source="test")),
                                  name="writer", daemon=True)
        writer.start()
        assert opening.wait(10)
        other = threading.Thread(target=lambda: got.setdefault("other", log.append_event("other_event", source="test")),
                                 name="other", daemon=True)
        other.start()
        other.join(0.5)
        assert other.is_alive() and "other" not in got, "another thread waits while an entry is written"
        release.set()
        writer.join(10)
        other.join(10)
        assert got == {"writer": 2, "other": 3}, got
        assert _types(log) == ["first", "outer_event", "other_event"] and log.verify_chain() == (True, None, 3)
    finally:
        restore()


# ---------------------------------------------------------------------------
# JR4 -- a fresh store opened by a read
# ---------------------------------------------------------------------------
def test_jr4_an_entry_recorded_while_a_read_opens_a_fresh_store_finds_its_table(tmp_path):
    state = {}
    sal, restore = _window(state)
    try:
        log = sal.SignedAuditLog(tmp_path / "audit" / "audit_chain.db")

        def record_once():
            # As the mode records through chain_log, which says a failure and never raises it.
            if "asked" not in state:
                state["asked"] = "asking"
                try:
                    state["asked"] = log.append_event("verdict", source="test", action="recorded as the store opens")
                except Exception as exc:  # noqa: BLE001 - what the record met is the finding
                    state["asked"] = f"{type(exc).__name__}: {exc}"

        state["hook"] = record_once
        running, events = _in_a_thread(lambda: log.get_events(limit=10))
        state.pop("hook")
        assert not running and state["asked"] == 1, state
        assert [event["event_type"] for event in events] == ["verdict"] and log.verify_chain() == (True, None, 1)

        again = sal.SignedAuditLog(tmp_path / "again" / "audit_chain.db")

        def fail_once():
            state.pop("hook")
            raise sqlite3.OperationalError("the store does not open")

        state["hook"] = fail_once
        with pytest.raises(sqlite3.OperationalError):
            again.append_event("first", source="test")
        assert again.append_event("second", source="test") == 1, "a table not made is made at the next call"
        assert _types(again) == ["second"]

        third = sal.SignedAuditLog(tmp_path / "third" / "audit_chain.db")
        state["table_refused"] = True
        with pytest.raises(sqlite3.OperationalError):
            third.append_event("first", source="test")
        assert "table_refused" not in state, "control: the store opened and its table was refused"
        assert third.append_event("second", source="test") == 1, "a table refused once is made at the next call"
        assert _types(third) == ["second"]
    finally:
        restore()
