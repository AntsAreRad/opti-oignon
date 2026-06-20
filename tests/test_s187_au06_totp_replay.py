"""S187 auth fix AU-06 -- TOTP codes are single-use within their validity window.

Before the fix, totp_validate verified a code with valid_window=1 and never
recorded which time-step it consumed, so a captured live code was replayable for
the ~90s it stayed inside the +/-1 step window.

The fix records the consumed step per user (a new totp_config.last_step column,
added idempotently for pre-existing databases) and rejects a code whose step is at
or below the stored one. The accept decision for validity is unchanged
(totp.verify(code, valid_window=1)); the replay check is layered on top.

The module is loaded in isolation via spec_from_file_location with a stubbed
opti_oignon.db_utils; the optional encryption import degrades to a base64 secret
store, which the validate path decrypts transparently. The 2FA DB path is
redirected to a tmp file per test. The matched step is a property of the
(secret, code) pair, so the tests are robust to a step rollover during execution.
"""

import importlib.util
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest

pyotp = pytest.importorskip("pyotp")

# Bare package + a db_utils stub so auth_2fa imports.
sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))

_db_utils_stub = types.ModuleType("opti_oignon.db_utils")


def _safe_connect(db_path, *, check_same_thread: bool = True, timeout: float = 5.0):
    return sqlite3.connect(
        str(db_path), check_same_thread=check_same_thread, timeout=timeout
    )


_db_utils_stub.safe_connect = _safe_connect
sys.modules["opti_oignon.db_utils"] = _db_utils_stub

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "auth_2fa.py"


def _load():
    spec = importlib.util.spec_from_file_location("auth_2fa_au06", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12+ dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


a2fa = _load()


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(a2fa, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(a2fa, "_2FA_DB_PATH", tmp_path / "auth_2fa.db")
    a2fa._init_2fa_db()
    return a2fa.TwoFactorAuthManager()


def _setup_active_totp(manager, user_id="user-1"):
    res = manager.totp_setup(user_id)
    assert res["success"], res
    secret = res["secret"]
    vr = manager.totp_verify(user_id, pyotp.TOTP(secret).now())
    assert vr["success"], vr
    return secret


def _stored_last_step(user_id="user-1"):
    conn = a2fa._get_2fa_conn()
    try:
        return conn.execute(
            "SELECT last_step FROM totp_config WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
    finally:
        conn.close()


def _set_last_step(value, user_id="user-1"):
    conn = a2fa._get_2fa_conn()
    try:
        conn.execute(
            "UPDATE totp_config SET last_step = ? WHERE user_id = ?", (value, user_id)
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Core property: a code accepted once is rejected on immediate reuse
# ---------------------------------------------------------------------------

def test_code_accepted_once_then_rejected_on_reuse(mgr):
    secret = _setup_active_totp(mgr)
    code = pyotp.TOTP(secret).now()
    assert mgr.totp_validate("user-1", code) is True
    assert mgr.totp_validate("user-1", code) is False


# ---------------------------------------------------------------------------
# A step greater than the stored one is accepted, then recorded
# ---------------------------------------------------------------------------

def test_greater_step_accepted_and_recorded(mgr):
    secret = _setup_active_totp(mgr)
    assert _stored_last_step() == 0  # setup/verify do not consume a step
    code = pyotp.TOTP(secret).now()
    assert mgr.totp_validate("user-1", code) is True
    assert _stored_last_step() > 0


# ---------------------------------------------------------------------------
# A step at or below the stored one is rejected
# ---------------------------------------------------------------------------

def test_step_at_or_below_stored_is_rejected(mgr):
    secret = _setup_active_totp(mgr)
    totp = pyotp.TOTP(secret)
    code = totp.now()
    matched = a2fa._totp_matched_step(totp, code, time.time())
    assert matched is not None
    _set_last_step(matched)  # simulate this step already consumed
    assert mgr.totp_validate("user-1", code) is False


def test_strictly_later_step_accepted_with_lower_stored_step(mgr):
    secret = _setup_active_totp(mgr)
    totp = pyotp.TOTP(secret)
    code = totp.now()
    matched = a2fa._totp_matched_step(totp, code, time.time())
    assert matched is not None
    _set_last_step(matched - 1)
    assert mgr.totp_validate("user-1", code) is True


# ---------------------------------------------------------------------------
# Migration: fresh DB and pre-existing DB both end up with the column
# ---------------------------------------------------------------------------

def test_fresh_db_has_last_step_column(tmp_path, monkeypatch):
    monkeypatch.setattr(a2fa, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(a2fa, "_2FA_DB_PATH", tmp_path / "fresh.db")
    a2fa._init_2fa_db()
    conn = a2fa._get_2fa_conn()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(totp_config)").fetchall()}
    finally:
        conn.close()
    assert "last_step" in cols


def test_migration_adds_last_step_to_pre_existing_db(tmp_path, monkeypatch):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE totp_config (
            user_id TEXT PRIMARY KEY,
            secret_encrypted TEXT NOT NULL,
            verified INTEGER DEFAULT 0,
            created_at REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO totp_config (user_id, secret_encrypted, verified, created_at) "
        "VALUES ('u', 'B64:xx', 1, 0)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(a2fa, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(a2fa, "_2FA_DB_PATH", db)
    a2fa._init_2fa_db()  # idempotent ALTER ADD COLUMN

    conn = sqlite3.connect(str(db))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(totp_config)").fetchall()}
        val = conn.execute(
            "SELECT last_step FROM totp_config WHERE user_id = 'u'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert "last_step" in cols
    assert val == 0  # existing row defaulted, not invalidated

    # Idempotent: a second init does not raise (column already present).
    a2fa._init_2fa_db()


# ---------------------------------------------------------------------------
# Source guards
# ---------------------------------------------------------------------------

def test_totp_validate_consults_and_updates_last_step_source():
    src = _PATH.read_text(encoding="utf-8")
    start = src.index("def totp_validate(")
    end = src.index("\n    def totp_disable(", start)
    body = src[start:end]
    assert "last_step" in body
    assert "_totp_matched_step(" in body
    assert "UPDATE totp_config SET last_step" in body


def test_helper_exists_and_uses_constant_time_compare():
    src = _PATH.read_text(encoding="utf-8")
    start = src.index("def _totp_matched_step(")
    end = src.index("class TwoFactorAuthManager", start)
    body = src[start:end]
    assert "compare_digest" in body
