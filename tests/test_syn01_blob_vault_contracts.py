#!/usr/bin/env python3
"""SYN-01 backfill: blob framing / transfer / vault-manifest contract gaps.

Companion suite (additive; the original blob/vault suites are never edited).
Each test pins a security contract on the SYN-01 receive surface that the
existing suites left unpinned, and each is red-before-green (it fails when the
guarding source line is neutralised, passes on the untouched source):

  * ``open_stream`` REFUSES a framed blob whose header version is not the
    supported one -- a forged header cannot coerce a different frame parse
    (``blob_store.py``);
  * ``pull_chunk`` REJECTS a cursor that is neither the next nor the just-served
    one and DROPS the session -- a peer cannot skip ahead in a fetch
    (``blob_transfer.py``);
  * ``apply_synced_note`` prune is SCOPED to the reconciled note -- another
    note's attachment rows survive a reconcile, so a manifest can never reach
    across note boundaries to orphan unrelated media (``notes_store.py``).

All three source modules are loaded in isolation with the same stubbed
``opti_oignon.encryption`` (real AES-256-GCM) / ``db_utils`` / ``user_isolation``
the sibling suites use, so this runs in-container without fastapi / ollama.

Local-only. Runs under pytest or the __main__ runner.
"""

import base64
import importlib.util
import os
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"
_KEY = bytes(range(32))
_KEY_A = bytes(range(32))


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


# --------------------------------------------------------------------------- #
# blob_store loader (stub encryption backed by real AES-256-GCM)               #
# --------------------------------------------------------------------------- #
def _load_blob_store():
    keys = ("opti_oignon", "opti_oignon.notes", "opti_oignon.encryption",
            "opti_oignon.notes.blob_store")
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    npkg = types.ModuleType("opti_oignon.notes")
    npkg.__path__ = []
    sys.modules["opti_oignon.notes"] = npkg

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    enc = types.ModuleType("opti_oignon.encryption")
    enc.encrypt_bytes = lambda key, pt: bytes([1]) + (
        lambda n: n + AESGCM(key).encrypt(n, pt, None)
    )(os.urandom(12))
    enc.decrypt_bytes = lambda key, blob: AESGCM(key).decrypt(
        blob[1:13], blob[13:], None
    )
    enc.get_encryption_key = lambda: None
    sys.modules["opti_oignon.encryption"] = enc

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.notes.blob_store", _OO / "notes" / "blob_store.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.notes.blob_store"] = mod
    spec.loader.exec_module(mod)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, restore


def _load_transfer():
    spec = importlib.util.spec_from_file_location(
        "blob_transfer_under_test_v", _OO / "veilid" / "blob_transfer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# notes_store loader (stub db_utils / user_isolation; sqlite on a temp file)   #
# --------------------------------------------------------------------------- #
def _load_notes_store():
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
    mod._sync_publish_note = lambda *a, **k: None

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
        "id": aid, "kind": kind, "mime": mime, "byte_size": size,
        "blob_ref": blob, "content_hash": ch,
        "thumbnail_b64": base64.b64encode(thumb).decode("ascii"),
    }


def _note_payload(attachments):
    return {
        "title": "N",
        "body_crdt_b64": base64.b64encode(b"BODY").decode("ascii"),
        "tags": "[]", "pinned": False, "created_at": "t0",
        "attachments": attachments,
    }


def _att_ids(mod, td, note_id):
    raw = sqlite3.connect(str(Path(td) / mod.DB_FILENAME))
    raw.row_factory = sqlite3.Row
    try:
        return {
            r["id"] for r in raw.execute(
                "SELECT id FROM attachment WHERE note_id = ?", (note_id,)
            ).fetchall()
        }
    finally:
        raw.close()


# --------------------------------------------------------------------------- #
# Contracts                                                                    #
# --------------------------------------------------------------------------- #
def test_framed_version_mismatch_refused():
    """A framed blob whose header version is not FRAMED_VERSION is refused.

    The version byte lives in the PLAINTEXT header, so flipping it leaves every
    frame's GCM tag intact -- only the version gate can reject the blob. Pins
    ``if version != FRAMED_VERSION: raise`` in ``open_stream``.
    """
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_blob_store()
        try:
            store = mod.NotesBlobStore(root=td, master_key=_KEY)
            store.seal_stream("v", [os.urandom(120)], chunk_size=64)
            path = store._blob_path("v")
            raw = bytearray(path.read_bytes())
            vpos = len(mod.FRAMED_MAGIC)          # the version byte's offset
            assert raw[vpos] == mod.FRAMED_VERSION
            raw[vpos] = (raw[vpos] + 7) & 0xFF    # an unsupported version
            path.write_bytes(bytes(raw))
            assert _raises(lambda: list(store.open_stream("v")))
        finally:
            restore()


def test_cursor_out_of_order_rejected():
    """pull_chunk rejects a skip-ahead cursor and drops the session.

    After serving chunk 0, a cursor that is neither the next (1) nor the
    just-served (0) must yield ``None`` (not the next sequential chunk) and the
    session must be gone. Pins ``if cursor != self._next: raise`` in
    ``_FetchSession.serve``.
    """
    with tempfile.TemporaryDirectory() as ta:
        bs, restore = _load_blob_store()
        tx = _load_transfer()
        tx.reset_blob_transfer()
        try:
            producer = bs.NotesBlobStore(root=ta, master_key=_KEY_A)
            # 12 KiB at 4 KiB frames -> >= 3 chunks, so chunk 0 does not exhaust.
            producer.seal_stream("a1", [os.urandom(12_000)], chunk_size=4096)
            assert tx.open_fetch("peerA", "r", "a1", blob_store=producer) is True
            c0 = tx.pull_chunk("peerA", "r", 0)
            assert c0 is not None and c0["done"] is False and c0["cursor"] == 1
            # Skip ahead: neither a retry (0) nor the next (1).
            assert tx.pull_chunk("peerA", "r", 5) is None
            assert tx.active_fetches() == 0       # session dropped on the error
        finally:
            tx.reset_blob_transfer()
            restore()


def test_prune_scoped_to_reconciled_note():
    """Manifest prune is scoped to the note; another note's media survives.

    Reconciling note n1 (dropping a2) must never reach across to n2's rows.
    Pins the ``note_id = ?`` scope of the prune SELECT in ``apply_synced_note``.
    """
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load_notes_store()
        try:
            store = mod.NotesStore(root=td)
            store.apply_synced_note("n1", _note_payload([_mani("a1"), _mani("a2")]))
            store.apply_synced_note("n2", _note_payload([_mani("b1")]))
            assert _att_ids(mod, td, "n1") == {"a1", "a2"}
            assert _att_ids(mod, td, "n2") == {"b1"}
            # Reconcile n1 down to {a1}; n2's b1 must be untouched.
            store.apply_synced_note("n1", _note_payload([_mani("a1")]))
            assert _att_ids(mod, td, "n1") == {"a1"}
            assert _att_ids(mod, td, "n2") == {"b1"}
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
