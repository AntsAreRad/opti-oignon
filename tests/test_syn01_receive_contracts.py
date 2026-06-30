#!/usr/bin/env python3
"""Gap-fill contracts for the SYN-01 receive half (companion to the per-store apply suites).

The per-store apply suites (``test_conversation_apply``, ``test_notes_apply``,
``test_skill_apply``, ``test_memory_canonical_apply``) each pin their store's
materialisation contracts. This companion closes two gaps surfaced while
mutation-proving that cluster, WITHOUT editing the original suites:

  * the conversation and note apply paths are HOOK-FREE -- like the canonical
    apply (which already proves it with a publish-hook spy), they must never
    call their store's ``_sync_publish_*`` glue, or an apply would re-publish a
    received record and start an apply -> write -> publish echo that inflates
    the clock and ping-pongs between devices forever. The conversation and note
    suites did not pin this; the canonical suite did. This restores symmetry.
  * a note tombstone for a note NEVER SEEN on this device still records a
    ``deleted = 1`` row (it is not a silent no-op), so a later out-of-order
    non-tombstone is reconciled by the LWW relay instead of silently
    resurrecting a deletion. The note suite only exercised the already-exists
    tombstone path.

Each store is loaded in isolation (``safe_connect`` stubbed to a plain sqlite3
connection, the user-isolation and config seams stubbed) exactly as the
sibling suites do, so no SQLCipher / fastapi / ollama / veilid is required.
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


def _load_conversation(tmpdir: str):
    """Load conversation.py in isolation (mirrors test_conversation_apply._load)."""
    keys = (
        "opti_oignon", "opti_oignon.db_utils", "opti_oignon.config",
        "opti_oignon.conversation",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    db = types.ModuleType("opti_oignon.db_utils")

    def _safe_connect(path, **kw):
        return sqlite3.connect(path, check_same_thread=kw.get("check_same_thread", False))

    db.safe_connect = _safe_connect
    sys.modules["opti_oignon.db_utils"] = db

    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(tmpdir)
    sys.modules["opti_oignon.config"] = cfg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.conversation", _OO / "conversation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.conversation"] = mod
    spec.loader.exec_module(mod)

    # Reversible marker for the at-rest field key (apply re-encrypts messages).
    mod._encrypt = lambda v: "E:" + v
    if hasattr(mod, "_decrypt"):
        mod._decrypt = lambda v: v[2:] if isinstance(v, str) and v.startswith("E:") else v

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


def _load_notes(tmpdir: str):
    """Load notes_store.py in isolation (mirrors test_notes_apply._load)."""
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

    # Hook-free in apply, but create/setup helpers call it; stub by default.
    mod._sync_publish_note = lambda *a, **k: None

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


def _conv_payload(conv_id="c1"):
    return {
        "user_id": "u",
        "conversation": {
            "id": conv_id,
            "title": "Hello",
            "created_at": "t0",
            "updated_at": "t1",
            "model": "m",
            "task_type": None,
            "preset": None,
            "metadata": {},
            "messages": [
                {"role": "user", "content": "hi", "timestamp": "t0",
                 "token_estimate": 1, "model": None, "metadata": {}},
            ],
        },
    }


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


def test_conversation_apply_does_not_republish():
    """apply_synced_conversation is hook-free: it never calls the publish glue."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_conversation(td)
        try:
            calls = []
            mod._sync_publish_conversation = lambda *a, **k: calls.append((a, k))
            mgr = mod.ConversationManager(db_path=Path(td) / "c.db")

            assert mgr.apply_synced_conversation(_conv_payload()) is True
            assert mgr.apply_synced_conversation(
                {"conversation": {"id": "c1"}}, deleted=True
            ) is True

            assert calls == []  # apply never re-publishes (no apply->write->publish echo)
        finally:
            restore()


def test_note_apply_does_not_republish():
    """apply_synced_note is hook-free: it never calls the publish glue."""
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_notes(td)
        try:
            calls = []
            mod._sync_publish_note = lambda *a, **k: calls.append((a, k))
            store = mod.NotesStore(root=td)

            assert store.apply_synced_note("n1", _note_payload()) is True
            assert store.apply_synced_note("n1", None, deleted=True) is True

            assert calls == []  # apply never re-publishes
        finally:
            restore()


def test_note_tombstone_for_unseen_note_records_tombstone():
    """A tombstone for a never-seen note records deleted=1, not a silent no-op.

    The row must exist (with the secure mobile_allowed default) so a later
    out-of-order non-tombstone is reconciled by the LWW relay instead of
    silently resurrecting the deletion.
    """
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_notes(td)
        try:
            store = mod.NotesStore(root=td)

            assert store.apply_synced_note("ghost", None, deleted=True) is True

            raw = _raw(str(Path(td) / mod.DB_FILENAME))
            try:
                row = raw.execute(
                    "SELECT deleted, mobile_allowed FROM note WHERE id = ?", ("ghost",)
                ).fetchone()
                assert row is not None              # the tombstone is recorded
                assert row["deleted"] == 1          # as deleted, not a no-op
                assert row["mobile_allowed"] == 0   # secure default preserved
            finally:
                raw.close()
        finally:
            restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
