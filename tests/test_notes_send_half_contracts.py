#!/usr/bin/env python3
"""SYN-01 backfill: the notes SEND half (the _sync_publish_note save hooks).

Companion suite (additive; originals never edited). The notes suites STUB
``_sync_publish_note`` to a no-op so they can exercise the receive half, which
means the producer's TRIGGERING on a write is never pinned. This companion
pins it at the store layer with a spy, the mirror of S278's NOTE_UPDATE G1:

  * a create journals AFTER the commit (commit-then-publish), as a non-deleted
    record (``add_note``);
  * a delete journals the TOMBSTONE (``deleted=True``) -- never a regular
    update, which would resurrect/clobber the note on peers under LWW
    (``delete_note``);
  * a REPEAT delete does NOT re-publish: the tombstone fires only when a live
    row actually flips (``deleted = 0`` -> ``1``), so a repeat is a no-op and
    cannot start an echo.

``notes_store.py`` is loaded in isolation with the stubbed ``db_utils`` /
``user_isolation`` idiom the sibling notes suites use; the spy replaces the
module-level ``_sync_publish_note`` (the same monkeypatch the receive suites
use to stub it), so the producer's own veilid availability is irrelevant -- we
assert only that the WRITE seam invoked it, and with which coordinates.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load():
    keys = ("opti_oignon", "opti_oignon.notes", "opti_oignon.db_utils",
            "opti_oignon.user_isolation", "opti_oignon.notes.notes_store")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    notes_pkg = types.ModuleType("opti_oignon.notes")
    notes_pkg.__path__ = []
    sys.modules["opti_oignon.notes"] = notes_pkg

    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda path, **kw: sqlite3.connect(
        path, check_same_thread=kw.get("check_same_thread", False))
    sys.modules["opti_oignon.db_utils"] = db

    ui = types.ModuleType("opti_oignon.user_isolation")
    ui.DEFAULT_LOCAL_USER = "local"
    ui.effective_user_id = lambda user_id, single_user_mode=True: (
        "local" if (single_user_mode or user_id is None) else user_id)
    sys.modules["opti_oignon.user_isolation"] = ui

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.notes.notes_store", _OO / "notes" / "notes_store.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.notes.notes_store"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, restore


def _spy(mod):
    """Install a recording spy over the module-level _sync_publish_note."""
    calls: list[dict] = []

    def rec(note_id, payload_fn=None, *, deleted=False, updated_at=""):
        calls.append({"note_id": note_id, "deleted": deleted,
                      "updated_at": updated_at})

    mod._sync_publish_note = rec
    return calls


def test_create_publishes_after_commit():
    """add_note journals a non-deleted record after the commit."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.NotesStore(root=td)
            calls = _spy(mod)
            store.add_note(title="N", body_crdt=b"B", note_id="n1")
            assert store.get_note("n1") is not None     # the row landed
            assert len(calls) == 1                       # and was journalled
            assert calls[0]["note_id"] == "n1"
            assert calls[0]["deleted"] is False          # a create, not a tombstone
        finally:
            restore()


def test_delete_publishes_tombstone():
    """delete_note journals the tombstone (deleted=True), not a regular update."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.NotesStore(root=td)
            store.add_note(title="N", body_crdt=b"B", note_id="n1")
            calls = _spy(mod)                            # spy only the delete
            assert store.delete_note("n1") is True
            assert len(calls) == 1
            assert calls[0]["note_id"] == "n1"
            assert calls[0]["deleted"] is True           # a tombstone publish
        finally:
            restore()


def test_repeat_delete_does_not_republish():
    """A repeat delete is a no-op: the tombstone fires only on a real flip."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.NotesStore(root=td)
            store.add_note(title="N", body_crdt=b"B", note_id="n1")
            calls = _spy(mod)
            assert store.delete_note("n1") is True
            assert len(calls) == 1                       # first flip publishes
            calls.clear()
            assert store.delete_note("n1") is False      # already tombstoned
            assert calls == []                           # no echo on the repeat
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
