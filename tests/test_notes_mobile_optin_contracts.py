#!/usr/bin/env python3
"""Per-item phone-sync opt-in contracts for the notes store.

Companion suite (additive; the existing notes suites are never edited). The
sibling send-half suite pins create/delete/repeat-delete at the publish seam
but leaves ``set_mobile_allowed`` -- the desktop trust decision that opts a
single note into phone-class sync -- unpinned. This suite covers it:

  * an EFFECTIVE opt-in republishes the note's full state, so a phone whose
    watermark has already advanced past the previously-filtered entry still
    receives the newly allowed note (republish is delivery, not security --
    the serve-time filter's live lookup stays the authority);
  * a tombstoned or unknown note is NEVER re-allowed and NEVER republished:
    flipping the flag cannot resurrect a deleted note into the phone-sync
    surface, and the call returns ``False``;
  * the outbound payload NEVER carries ``mobile_allowed``: the flag is local
    desktop trust state; were it to ride
    the wire, a receiving device's apply path could become a writer of it.

``notes_store.py`` is loaded through the shared isolation window with
stubbed ``db_utils`` and ``user_isolation``; the sync package is unreachable
there, so the store's own publish hook is the quiet no-op its comment
describes. The module-level ``_sync_publish_note`` is replaced by a recording
spy (the same monkeypatch the receive suites use), so the producer's own
framework availability is irrelevant -- we assert only that the WRITE seam
invoked it, with which coordinates, and with which payload.

Local-only. Runs under pytest or the ``__main__`` runner.
"""

import sqlite3
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_STORE = "opti_oignon.notes.notes_store"


def _load():
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda path, **kw: sqlite3.connect(
        path, check_same_thread=kw.get("check_same_thread", False))
    ui = types.ModuleType("opti_oignon.user_isolation")
    ui.DEFAULT_LOCAL_USER = "local"
    ui.effective_user_id = lambda user_id, single_user_mode=True: (
        "local" if (single_user_mode or user_id is None) else user_id)
    loaded, restore = isolate(
        targets={_STORE: source("notes", "notes_store.py")},
        seeded={"opti_oignon.db_utils": db, "opti_oignon.user_isolation": ui},
        packages=("opti_oignon.notes",),
    )
    return loaded[_STORE], restore


def _spy(mod):
    """Spy the module-level _sync_publish_note; build and capture the payload.

    The recorder runs ``payload_fn`` for non-tombstone publishes so a test can
    inspect exactly what would cross the wire.
    """
    calls: list[dict] = []

    def rec(note_id, payload_fn=None, *, deleted=False, updated_at=""):
        payload = None
        if payload_fn is not None and not deleted:
            payload = payload_fn()
        calls.append({"note_id": note_id, "deleted": deleted,
                      "updated_at": updated_at, "payload": payload})

    mod._sync_publish_note = rec
    return calls


def test_opt_in_republishes_full_state():
    """An effective opt-in journals a fresh full-state record (no flag on it)."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.NotesStore(root=td)
            store.add_note(title="Hello", body_crdt=b"BODY", note_id="n1")
            calls = _spy(mod)                          # spy only the flip
            assert store.set_mobile_allowed("n1", True) is True
            assert len(calls) == 1                     # the opt-in republished
            assert calls[0]["note_id"] == "n1"
            assert calls[0]["deleted"] is False        # full state, not a tombstone
            payload = calls[0]["payload"]
            assert payload is not None
            assert payload["title"] == "Hello"         # carries the real state
            assert "body_crdt_b64" in payload
            assert "attachments" in payload
            assert "mobile_allowed" not in payload     # the flag never rides
        finally:
            restore()


def test_tombstoned_or_unknown_note_is_not_re_allowed():
    """A deleted or unknown note is never re-allowed and never republished."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.NotesStore(root=td)
            store.add_note(title="Hello", body_crdt=b"BODY", note_id="n1")
            assert store.delete_note("n1") is True     # tombstone the note
            calls = _spy(mod)
            assert store.set_mobile_allowed("n1", True) is False   # never revived
            assert calls == []                         # and never republished
            assert store.set_mobile_allowed("ghost", True) is False  # unknown id
            assert calls == []
        finally:
            restore()


def test_outbound_payload_omits_the_mobile_flag():
    """The wire payload never carries mobile_allowed, even when the flag is set."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load()
        try:
            store = mod.NotesStore(root=td)
            store.add_note(title="Hello", body_crdt=b"BODY", note_id="n1")
            store.set_mobile_allowed("n1", True)       # the flag is now ON
            record = store.get_note("n1")
            payload = mod._note_sync_payload(
                record, store.list_attachments("n1"))
            assert "mobile_allowed" not in payload     # absent from the wire
            assert payload["title"] == "Hello"
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
