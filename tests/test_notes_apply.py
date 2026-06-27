#!/usr/bin/env python3
"""Tests for the note materialization apply method (SYN-01 receive half).

``NotesStore.apply_synced_note`` is the receiving half of a sync round for
notes. It loads ``notes_store.py`` in isolation (``safe_connect`` to a plain
sqlite3 connection, ``effective_user_id`` to the single-user default, the
publish hook stubbed) and proves the two load-bearing invariants plus the
basics:

  * a record materialises the note row (create), seeding the body;
  * an UPDATE to an existing note touches metadata ONLY -- ``body_crdt`` (owned
    by NOTE_UPDATE/CRDT) and ``mobile_allowed`` (N9-D3, the phone opt-in) are
    never overwritten by a received record;
  * applying the same record twice is idempotent;
  * a deleted record tombstones (``deleted = 1``), CRDT-safe, never a hard
    delete;
  * a malformed payload fails secure (returns False, raises nothing).

Local-only. Runs under pytest or the __main__ runner.
"""

import base64
import importlib.util
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load(tmpdir: str):
    keys = (
        "opti_oignon", "opti_oignon.notes", "opti_oignon.db_utils",
        "opti_oignon.user_isolation", "opti_oignon.notes.notes_store",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    notes_pkg = types.ModuleType("opti_oignon.notes")
    notes_pkg.__path__ = []
    sys.modules["opti_oignon.notes"] = notes_pkg

    db = types.ModuleType("opti_oignon.db_utils")

    def _safe_connect(path, **kw):
        return sqlite3.connect(path, check_same_thread=kw.get("check_same_thread", False))

    db.safe_connect = _safe_connect
    sys.modules["opti_oignon.db_utils"] = db

    ui = types.ModuleType("opti_oignon.user_isolation")
    ui.DEFAULT_LOCAL_USER = "local"
    ui.effective_user_id = lambda user_id, single_user_mode=True: (
        "local" if (single_user_mode or user_id is None) else user_id
    )
    sys.modules["opti_oignon.user_isolation"] = ui

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.notes.notes_store", _OO / "notes" / "notes_store.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.notes.notes_store"] = mod
    spec.loader.exec_module(mod)

    # The publish hook is hook-free in apply, but the create/setup helpers call
    # it; stub it so setup never reaches the (absent) veilid framework.
    mod._sync_publish_note = lambda *a, **k: None

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


def _note_payload(title="Hello", body=b"BODY1", tags='["a"]', pinned=True):
    return {
        "title": title,
        "body_crdt_b64": base64.b64encode(body).decode("ascii"),
        "tags": tags,
        "pinned": pinned,
        "created_at": "t0",
    }


def _raw(dbpath):
    c = sqlite3.connect(dbpath)
    c.row_factory = sqlite3.Row
    return c


def test_apply_creates_note_with_secure_defaults():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            assert store.apply_synced_note("n1", _note_payload()) is True

            raw = _raw(str(Path(td) / mod.DB_FILENAME))
            try:
                row = raw.execute("SELECT * FROM note WHERE id = ?", ("n1",)).fetchone()
                assert row is not None
                assert row["title"] == "Hello"
                assert row["tags"] == '["a"]'
                assert row["pinned"] == 1
                assert bytes(row["body_crdt"]) == b"BODY1"   # body seeded on create
                assert row["mobile_allowed"] == 0            # secure default, never from the wire
                assert row["deleted"] == 0
            finally:
                raw.close()
        finally:
            restore()


def test_update_never_clobbers_body_or_mobile_allowed():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            # Seed an existing note locally with a known body, then opt it onto
            # the phone (the deliberate local gesture).
            store.add_note(title="Local", body_crdt=b"LOCALBODY", note_id="n1")
            store.set_mobile_allowed("n1", True)

            # A received full-state record carries a different title AND a
            # different body; only the title (metadata) may land.
            assert store.apply_synced_note(
                "n1", _note_payload(title="Remote", body=b"REMOTEBODY")
            ) is True

            raw = _raw(str(Path(td) / mod.DB_FILENAME))
            try:
                row = raw.execute("SELECT * FROM note WHERE id = ?", ("n1",)).fetchone()
                assert row["title"] == "Remote"               # metadata updated
                assert bytes(row["body_crdt"]) == b"LOCALBODY"  # body NOT clobbered (CRDT owns it)
                assert row["mobile_allowed"] == 1              # opt-in NOT reset by the wire
            finally:
                raw.close()
        finally:
            restore()


def test_apply_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            store.apply_synced_note("n1", _note_payload())
            store.apply_synced_note("n1", _note_payload())  # twice

            raw = _raw(str(Path(td) / mod.DB_FILENAME))
            try:
                n = raw.execute(
                    "SELECT COUNT(*) AS c FROM note WHERE id = ?", ("n1",)
                ).fetchone()["c"]
                assert n == 1
            finally:
                raw.close()
        finally:
            restore()


def test_apply_tombstone_marks_deleted():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            store.apply_synced_note("n1", _note_payload())
            assert store.apply_synced_note("n1", None, deleted=True) is True

            raw = _raw(str(Path(td) / mod.DB_FILENAME))
            try:
                row = raw.execute("SELECT deleted FROM note WHERE id = ?", ("n1",)).fetchone()
                assert row is not None and row["deleted"] == 1   # tombstone, not hard delete
            finally:
                raw.close()
        finally:
            restore()


def test_apply_fails_secure_on_malformed_payload():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            assert store.apply_synced_note("", _note_payload()) is False
            assert store.apply_synced_note("n1", None) is False        # no payload, not deleted
            assert store.apply_synced_note("n1", 5) is False
        finally:
            restore()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'} - {failures} failure(s)")
    sys.exit(1 if failures else 0)
