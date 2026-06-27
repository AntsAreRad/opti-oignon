#!/usr/bin/env python3
"""Tests for the eager vault attachment manifest (SYN-01).

The manifest rides INSIDE the NOTE record: ``_note_sync_payload`` emits the
authoritative attachment set (id, kind, mime, byte_size, blob_ref, plaintext
content_hash, small thumbnail -- but NOT the bytes), and
``apply_synced_note`` reconciles a peer's attachment rows to exactly that set.

Proven here, all against ``notes_store.py`` loaded in isolation:

  * the published payload carries the manifest with the thumbnail and the
    plaintext content_hash;
  * applying a record UPSERTS each manifest row WITHOUT any blob bytes (the
    bytes are fetched on demand later);
  * reconciliation PRUNES an attachment dropped from the manifest;
  * the UPSERT PRESERVES an already-downloaded blob's local ``nonce`` (and so
    never orphans the sealed bytes) while still refreshing manifest fields;
  * a tombstoned note clears its attachment rows.

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
    db.safe_connect = lambda path, **kw: sqlite3.connect(
        path, check_same_thread=kw.get("check_same_thread", False)
    )
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
    mod._sync_publish_note = lambda *a, **k: None  # setup never reaches veilid

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return mod, restore


def _mani(aid, *, kind="image", mime="image/png", size=2048,
          blob="blob://x", ch="HASH", thumb=b"THUMB"):
    return {
        "id": aid,
        "kind": kind,
        "mime": mime,
        "byte_size": size,
        "blob_ref": blob,
        "content_hash": ch,
        "thumbnail_b64": base64.b64encode(thumb).decode("ascii"),
    }


def _note_payload(attachments):
    return {
        "title": "N",
        "body_crdt_b64": base64.b64encode(b"BODY").decode("ascii"),
        "tags": "[]",
        "pinned": False,
        "created_at": "t0",
        "attachments": attachments,
    }


def _att_rows(mod, td, note_id):
    raw = sqlite3.connect(str(Path(td) / mod.DB_FILENAME))
    raw.row_factory = sqlite3.Row
    try:
        return raw.execute(
            "SELECT * FROM attachment WHERE note_id = ? ORDER BY id", (note_id,)
        ).fetchall()
    finally:
        raw.close()


def test_payload_carries_attachment_manifest():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            store.add_note(title="N", body_crdt=b"B", note_id="n1")
            store.add_attachment(
                "n1", "image", blob_ref="blob://a1", mime="image/png",
                byte_size=2048, content_hash="HASH1", thumbnail=b"THUMB",
                attachment_id="a1",
            )
            rec = store.get_note("n1")
            payload = mod._note_sync_payload(rec, store.list_attachments("n1"))
            atts = payload["attachments"]
            assert len(atts) == 1
            e = atts[0]
            assert e["id"] == "a1"
            assert e["kind"] == "image"
            assert e["mime"] == "image/png"
            assert e["byte_size"] == 2048
            assert e["blob_ref"] == "blob://a1"
            assert e["content_hash"] == "HASH1"
            assert base64.b64decode(e["thumbnail_b64"]) == b"THUMB"
        finally:
            restore()


def test_apply_upserts_manifest_without_bytes():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            ok = store.apply_synced_note(
                "n1", _note_payload([_mani("a1", thumb=b"TH1", ch="H1", blob="blob://1")])
            )
            assert ok is True
            rows = _att_rows(mod, td, "n1")
            assert len(rows) == 1
            r = rows[0]
            assert r["id"] == "a1"
            assert r["kind"] == "image"
            assert r["blob_ref"] == "blob://1"
            assert r["content_hash"] == "H1"
            assert bytes(r["thumbnail"]) == b"TH1"
            assert r["nonce"] == ""            # no local seal yet -- bytes not downloaded
        finally:
            restore()


def test_apply_prunes_removed_attachment():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            store.apply_synced_note("n1", _note_payload([_mani("a1"), _mani("a2")]))
            assert {r["id"] for r in _att_rows(mod, td, "n1")} == {"a1", "a2"}
            # The authoritative set now omits a2 -> it must be pruned on apply.
            store.apply_synced_note("n1", _note_payload([_mani("a1")]))
            assert {r["id"] for r in _att_rows(mod, td, "n1")} == {"a1"}
        finally:
            restore()


def test_apply_preserves_downloaded_blob_nonce():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            store.add_note(title="N", body_crdt=b"B", note_id="n1")
            # Simulate a blob already downloaded + sealed locally: the row holds
            # a device-local nonce. A re-applied manifest must NOT clobber it.
            store.add_attachment(
                "n1", "image", blob_ref="blob://1", nonce="NONCE1",
                content_hash="OLD", attachment_id="a1",
            )
            store.apply_synced_note(
                "n1", _note_payload([_mani("a1", ch="NEW", blob="blob://1")])
            )
            r = _att_rows(mod, td, "n1")[0]
            assert r["nonce"] == "NONCE1"      # downloaded blob NOT orphaned
            assert r["content_hash"] == "NEW"  # manifest field still refreshed
        finally:
            restore()


def test_tombstone_clears_attachments():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            store = mod.NotesStore(root=td)
            store.apply_synced_note("n1", _note_payload([_mani("a1"), _mani("a2")]))
            assert len(_att_rows(mod, td, "n1")) == 2
            store.apply_synced_note("n1", None, deleted=True)
            assert _att_rows(mod, td, "n1") == []
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
