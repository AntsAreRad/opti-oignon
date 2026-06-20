"""S184 audit fix VL-02 -- Veilid change-feed journal at-rest encryption.

The change-feed journal (veilid_change_feed.db) stores the synced record payloads
(conversations, memory, etc.). It previously opened a plain ``sqlite3.connect``, so
the synced content sat in plaintext at rest -- inconsistent with sync_queue.py, which
uses the encrypted ``db_utils.safe_connect`` helper (the S136 pattern), and with the
project's encrypted-everywhere posture. The fix routes the journal connection through
the same ``_safe_connect`` helper.

These are source-level assertions: the veilid subpackage's import chain (records,
db_utils, config) makes standalone module loading heavy, and whether SQLCipher is
actually present is an environment property. Pinning that the encrypted helper is wired
in (and the bare sqlite3.connect is gone from the journal connection) is the robust,
deterministic check.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHANGE_FEED = _REPO_ROOT / "opti_oignon" / "veilid" / "change_feed.py"
_SYNC_QUEUE = _REPO_ROOT / "opti_oignon" / "sync_queue.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_change_feed_imports_safe_connect():
    src = _read(_CHANGE_FEED)
    assert "from opti_oignon.db_utils import safe_connect as _safe_connect" in src


def test_change_feed_connection_uses_encrypted_helper():
    src = _read(_CHANGE_FEED)
    # Isolate the _conn method body.
    start = src.index("def _conn(")
    nxt = src.index("\n    def ", start + 1)
    body = src[start:nxt]
    assert "_safe_connect(" in body, "change-feed journal must use the encrypted helper"
    assert "sqlite3.connect(" not in body, (
        "change-feed journal must not open a bare sqlite3 connection"
    )


def test_change_feed_matches_sync_queue_pattern():
    # sync_queue is the reference: both must wire the same encrypted-connection helper.
    sq = _read(_SYNC_QUEUE)
    cf = _read(_CHANGE_FEED)
    assert "safe_connect" in sq
    assert "_safe_connect" in cf
