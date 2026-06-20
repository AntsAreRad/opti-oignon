"""S185 audit fix TC-02 -- the plaintext DB fallback is loud, not silent.

response_cache, semantic_cache, connection_pool and veilid.change_feed each fall
back to a bare ``sqlite3.connect`` lambda when ``db_utils`` cannot be imported,
with no log line. And ``db_utils.safe_connect`` itself, when ``db_encryption`` is
unavailable, fell back to plain sqlite3 silently AND without applying the Bulbe
fail-closed enforcement that ``get_encrypted_connection`` applies -- the real
exposure: a Bulbe deployment with a broken db_encryption import would write
plaintext silently while the code assumes encryption.

The fix: safe_connect now warns loudly (once) on the plaintext fallback and fails
closed in Bulbe (raises); get_encrypted_connection's own Daily plaintext fallback
is upgraded from debug to a once-gated warning; and the four fallback sites log a
prominent warning.

db_utils is loaded in isolation: opti_oignon is stubbed so the db_encryption
import fails (_ENCRYPTION_AVAILABLE False), exercising the fallback. is_bulbe is
controlled via a stub opti_oignon.security_mode module.
"""

import importlib.util
import logging
import sqlite3
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "db_utils.py"


def _load():
    spec = importlib.util.spec_from_file_location("db_utils_tc02", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12 dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


du = _load()


def _set_mode(monkeypatch, bulbe: bool):
    stub = types.ModuleType("opti_oignon.security_mode")
    stub.is_bulbe = lambda: bulbe
    monkeypatch.setitem(sys.modules, "opti_oignon.security_mode", stub)


# ---------------------------------------------------------------------------
# db_utils.safe_connect with db_encryption unavailable
# ---------------------------------------------------------------------------

def test_encryption_unavailable_in_isolation():
    # The isolated load cannot import db_encryption, so the fallback is active.
    assert du._ENCRYPTION_AVAILABLE is False


def test_daily_warns_loudly_and_connects(tmp_path, monkeypatch, caplog):
    _set_mode(monkeypatch, bulbe=False)
    monkeypatch.setattr(du, "_plaintext_fallback_warned", False)
    with caplog.at_level(logging.WARNING):
        conn = du.safe_connect(tmp_path / "x.db")
    assert isinstance(conn, sqlite3.Connection)
    conn.execute("SELECT 1")
    conn.close()
    assert any(
        "plaintext" in r.message.lower() for r in caplog.records
    ), "a prominent plaintext warning must be emitted"


def test_bulbe_fails_closed(tmp_path, monkeypatch):
    _set_mode(monkeypatch, bulbe=True)
    monkeypatch.setattr(du, "_plaintext_fallback_warned", False)
    with pytest.raises(RuntimeError):
        du.safe_connect(tmp_path / "y.db")


def test_warning_emitted_once(tmp_path, monkeypatch, caplog):
    _set_mode(monkeypatch, bulbe=False)
    monkeypatch.setattr(du, "_plaintext_fallback_warned", False)
    with caplog.at_level(logging.WARNING):
        du.safe_connect(tmp_path / "a.db").close()
        du.safe_connect(tmp_path / "b.db").close()
    plaintext_warnings = [
        r for r in caplog.records if "plaintext" in r.message.lower()
    ]
    assert len(plaintext_warnings) == 1


# ---------------------------------------------------------------------------
# The four fallback sites and db_encryption now warn (source assertions)
# ---------------------------------------------------------------------------

def _src(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def test_fallback_sites_warn_on_plaintext():
    for rel in (
        "opti_oignon/response_cache.py",
        "opti_oignon/semantic_cache.py",
        "opti_oignon/connection_pool.py",
        "opti_oignon/veilid/change_feed.py",
    ):
        src = _src(rel)
        # The ImportError fallback block must carry a prominent warning.
        block = src.split("from opti_oignon.db_utils import safe_connect", 1)[1]
        block = block.split("\n\n", 1)[0]
        assert ".warning(" in block and "PLAINTEXT" in block.upper(), (
            f"{rel} must warn loudly on the db_utils plaintext fallback"
        )


def test_db_encryption_plaintext_fallback_is_loud():
    src = _src("opti_oignon/db_encryption.py")
    fn = src.split("def get_encrypted_connection", 1)[1].split("\ndef ", 1)[0]
    # The plaintext fallback path uses a warning, not a silent debug.
    assert "logger.warning(" in fn
    assert "PLAINTEXT" in fn.upper()
