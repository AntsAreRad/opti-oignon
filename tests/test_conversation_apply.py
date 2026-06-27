#!/usr/bin/env python3
"""Tests for the conversation materialization apply method (SYN-01 receive half).

``ConversationManager.apply_synced_conversation`` is the receiving half of a
sync round: a winning CONVERSATION record is written into the local store so it
surfaces on this device. This suite loads ``conversation.py`` in isolation --
``safe_connect`` stubbed to a plain sqlite3 connection, ``DATA_DIR`` to a tmp
dir, and the S125 field key swapped for a reversible marker -- and proves:

  * a record materialises the conversation row and its messages;
  * message content is RE-ENCRYPTED at rest (the marker form is stored, never
    the plaintext);
  * applying the same record twice is idempotent (one conversation, no message
    duplication) -- the property that keeps the apply -> write loop from
    inflating;
  * a tombstone removes the conversation and its messages;
  * a malformed payload fails secure (returns False, raises nothing).

The engine-side wiring (the round lander that calls this) is validated by the
maintainer's engine harness, not here. Local-only. Runs under pytest or the
__main__ runner.
"""

import importlib.util
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _load(tmpdir: str):
    """Load conversation.py in isolation with a plain-sqlite safe_connect.

    sys.modules is saved/restored so sibling suites stay clean. The S125
    ``_encrypt``/``_decrypt`` module globals are swapped for a reversible marker
    AFTER load, so the at-rest encryption path is observable.
    """
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

    # Reversible marker for the at-rest field key, so re-encryption is visible.
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


def _payload(conv_id="c1", title="Hello", msgs=(("user", "hi"), ("assistant", "yo"))):
    return {
        "user_id": "u",
        "conversation": {
            "id": conv_id,
            "title": title,
            "created_at": "t0",
            "updated_at": "t1",
            "model": "m",
            "task_type": None,
            "preset": None,
            "metadata": {"k": "v"},
            "messages": [
                {
                    "role": r,
                    "content": c,
                    "timestamp": f"t{i}",
                    "token_estimate": 1,
                    "model": None,
                    "metadata": {},
                }
                for i, (r, c) in enumerate(msgs)
            ],
        },
    }


def test_apply_materialises_conversation_and_reencrypts():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            dbp = str(Path(td) / "c.db")
            mgr = mod.ConversationManager(db_path=Path(dbp))
            assert mgr.apply_synced_conversation(_payload()) is True

            raw = sqlite3.connect(dbp)
            try:
                title = raw.execute(
                    "SELECT title FROM conversations WHERE id = ?", ("c1",)
                ).fetchone()
                assert title is not None and title[0] == "Hello"
                rows = raw.execute(
                    "SELECT role, content FROM messages "
                    "WHERE conversation_id = ? ORDER BY timestamp ASC",
                    ("c1",),
                ).fetchall()
                assert [r[0] for r in rows] == ["user", "assistant"]
                # Re-encrypted at rest: the marker form, never the plaintext.
                assert rows[0][1] == "E:hi" and rows[1][1] == "E:yo"
            finally:
                raw.close()
        finally:
            restore()


def test_apply_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            dbp = str(Path(td) / "c.db")
            mgr = mod.ConversationManager(db_path=Path(dbp))
            mgr.apply_synced_conversation(_payload())
            mgr.apply_synced_conversation(_payload())  # twice

            raw = sqlite3.connect(dbp)
            try:
                conv_n = raw.execute(
                    "SELECT COUNT(*) FROM conversations WHERE id = ?", ("c1",)
                ).fetchone()[0]
                msg_n = raw.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", ("c1",)
                ).fetchone()[0]
                assert conv_n == 1            # one conversation, not duplicated
                assert msg_n == 2             # messages cleared+reinserted, not doubled
            finally:
                raw.close()
        finally:
            restore()


def test_apply_tombstone_removes_conversation():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            dbp = str(Path(td) / "c.db")
            mgr = mod.ConversationManager(db_path=Path(dbp))
            mgr.apply_synced_conversation(_payload())
            assert mgr.apply_synced_conversation(
                {"conversation": {"id": "c1"}}, deleted=True
            ) is True

            raw = sqlite3.connect(dbp)
            try:
                assert raw.execute(
                    "SELECT COUNT(*) FROM conversations WHERE id = ?", ("c1",)
                ).fetchone()[0] == 0
                assert raw.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", ("c1",)
                ).fetchone()[0] == 0
            finally:
                raw.close()
        finally:
            restore()


def test_apply_fails_secure_on_malformed_payload():
    with tempfile.TemporaryDirectory() as td:
        mod, restore = _load(td)
        try:
            mgr = mod.ConversationManager(db_path=Path(td) / "c.db")
            assert mgr.apply_synced_conversation({}) is False
            assert mgr.apply_synced_conversation({"conversation": {}}) is False
            assert mgr.apply_synced_conversation({"conversation": {"id": ""}}) is False
            assert mgr.apply_synced_conversation({"conversation": 5}) is False
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
