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
    desktop trust state (MOBILE_THREAT_MODEL.md section 3); were it to ride
    the wire, a receiving device's apply path could become a writer of it.

``notes_store.py`` is loaded in isolation with the same stubbed
``db_utils`` / ``user_isolation`` idiom the sibling notes suites use, and the
module-level ``_sync_publish_note`` is replaced by a recording spy (the same
monkeypatch the receive suites use), so the producer's own framework
availability is irrelevant -- we assert only that the WRITE seam invoked it,
with which coordinates, and with which payload.

Local-only. Runs under pytest or the ``__main__`` runner.
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
