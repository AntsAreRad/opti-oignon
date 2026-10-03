#!/usr/bin/env python3
"""The CRDT NOTE_UPDATE store, each contract proven by directed mutation.

Companion suite (additive; the original suites are never edited). The existing
``test_note_updates_store.py`` pins the store contracts but is full-stack DEP
(it imports the package and so pulls fastapi/ollama); it does not run in this
container. ``note_updates_store.py`` is import-isolable, so this companion
re-pins the load-bearing store contracts under the same stubbed
``db_utils`` / ``user_isolation`` idiom the notes suites use -- making them both
runnable here AND proven by directed mutation -- and fills two gaps the DEP suite leaves:

  * the ``sync_publish`` gate at the STORE layer: a remote-apply landing
    (``sync_publish=False``) must NOT re-publish the received update (the DEP
    suite forces ``sync_publish=False`` everywhere, so the honouring of the
    flag is never exercised);
  * the DEFAULT parent-liveness probe ``_sibling_lookup`` against the sibling
    ``notes.db``: a tombstoned (``deleted = 1``) parent is NOT live, so the
    append seam refuses (the DEP suite injects a fake ``parent_lookup``, so the
    real sibling probe is never exercised).

The append seam, the watermark, and the destructive tombstone prune are the
section-5 / section-4 posture of the notes CRDT specification. The engine sink
(``veilid/sync_engine._update_sink_for``) is host-side and out of scope here.

Local-only. Runs under pytest or the __main__ runner.
"""

import sqlite3
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


def _load_store():
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda path, **kw: sqlite3.connect(
        path, check_same_thread=kw.get("check_same_thread", False))

    ui = types.ModuleType("opti_oignon.user_isolation")
    ui.DEFAULT_LOCAL_USER = "local"
    ui.effective_user_id = lambda user_id, single_user_mode=True: (
        "local" if (single_user_mode or user_id is None) else user_id)

    loaded, restore = isolate(
        targets={"opti_oignon.notes.note_updates_store":
                 source("notes", "note_updates_store.py")},
        seeded={"opti_oignon.db_utils": db, "opti_oignon.user_isolation": ui},
        packages=("opti_oignon.notes",),
    )
    return loaded["opti_oignon.notes.note_updates_store"], restore


def _live_store(mod, td):
    """A store whose parent is always live (the injected-gate path)."""
    return mod.NoteUpdatesStore(root=td, parent_lookup=lambda n, u: True)


# --------------------------------------------------------------------------- #
# Ported store contracts (DEP suite -> isolated, proven by mutation)          #
# --------------------------------------------------------------------------- #
def test_duplicate_seq_never_replaces():
    """A duplicate (user, note, seq) is refused; the original row is intact.

    The append-only integrity jewel. Pins the IntegrityError->refuse on the
    unique ``(user_id, note_id, seq)`` index (a plain INSERT, never a REPLACE).
    """
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_store()
        try:
            store = _live_store(mod, td)
            store.append_update("n1", b"first", seq=1, sync_publish=False)
            try:
                store.append_update("n1", b"second", seq=1, sync_publish=False)
                raised = False
            except mod.NoteUpdateRefused:
                raised = True
            assert raised
            rows = store.list_updates("n1")
            assert len(rows) == 1
            assert rows[0].update_blob == b"first"   # never overwritten
        finally:
            restore()


def test_append_refuses_dead_parent():
    """An unknown or dead parent refuses; nothing is persisted (section 5)."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_store()
        try:
            store = mod.NoteUpdatesStore(root=td, parent_lookup=lambda n, u: False)
            try:
                store.append_update("n1", b"x", sync_publish=False)
                raised = False
            except mod.NoteUpdateRefused:
                raised = True
            assert raised
            assert store.count_updates("n1") == 0
        finally:
            restore()


def test_watermark_regression_rejected():
    """The checkpoint watermark is monotonic; a regression is a no-op."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_store()
        try:
            store = _live_store(mod, td)
            assert store.set_checkpoint_watermark("n1", 5) is True
            assert store.set_checkpoint_watermark("n1", 3) is False
            assert store.get_checkpoint_watermark("n1") == 5
        finally:
            restore()


def test_tombstone_prune_refuses_live_parent():
    """The destructive full-tail prune refuses while the parent is still live.

    The inverse of the append gate: prune proceeds only when the parent is
    affirmatively NOT live, so a live note never loses its tail.
    """
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_store()
        try:
            store = _live_store(mod, td)
            store.append_update("n1", b"a", seq=1, sync_publish=False)
            store.append_update("n1", b"b", seq=2, sync_publish=False)
            assert store.prune_for_tombstone("n1") == 0
            assert store.count_updates("n1") == 2
        finally:
            restore()


# --------------------------------------------------------------------------- #
# Gap G1: the sync_publish gate at the store layer (no-republish)             #
# --------------------------------------------------------------------------- #
def test_sync_publish_gate_suppresses_on_remote_apply():
    """A remote-apply append (sync_publish=False) must NOT re-publish.

    Re-publishing a received record would re-sign the author's update as ours.
    A local append (sync_publish=True) does publish through the glue. Pins the
    ``if sync_publish:`` branch in ``append_update``.
    """
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_store()
        try:
            calls: list = []
            mod._sync_publish_note_update = lambda *a, **k: calls.append(a)
            store = _live_store(mod, td)
            # Remote-apply landing: suppressed.
            store.append_update("n1", b"remote", seq=3, sync_publish=False)
            assert calls == []
            # Local edit: published.
            store.append_update("n1", b"local", sync_publish=True)
            assert len(calls) == 1
        finally:
            restore()


# --------------------------------------------------------------------------- #
# Gap G2: the default sibling parent-liveness probe respects the tombstone     #
# --------------------------------------------------------------------------- #
def test_default_sibling_probe_respects_tombstone():
    """The real ``_sibling_lookup`` treats a tombstoned parent as NOT live.

    With no injected ``parent_lookup``, the seam probes the sibling ``notes.db``
    for a non-tombstoned row; a ``deleted = 1`` parent must refuse the append,
    so the append gate honours tombstone-wins. Pins the ``AND deleted = 0`` of
    the sibling probe SELECT.
    """
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_store()
        try:
            sib = Path(td) / mod._NOTES_DB_FILENAME
            con = sqlite3.connect(str(sib))
            con.execute("CREATE TABLE note (id TEXT, user_id TEXT, deleted INTEGER)")
            con.execute(
                "INSERT INTO note (id, user_id, deleted) VALUES ('n1', 'local', 1)")
            con.commit()
            con.close()
            store = mod.NoteUpdatesStore(root=td)   # default _sibling_lookup
            try:
                store.append_update("n1", b"x", sync_publish=False)
                raised = False
            except mod.NoteUpdateRefused:
                raised = True
            assert raised
            assert store.count_updates("n1") == 0
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
