"""S187 auth fix AU-03 -- recovery/app-password hashes are keyed off the master key.

Before the fix, recovery codes and app passwords were HMAC'd with a public
constant ("oo-recovery-hmac-key-" + kdf), which is no secret under Kerckhoffs
(the project is open source), so the HMAC added nothing over plain SHA-256 against
a DB-leak adversary.

The fix derives the code-hashing subkey off the master encryption key on its own
HKDF domain (distinct from the SQLCipher subkey, the learned-router MAC and the
audit anchor) and versions stored hashes with a "v2:" prefix. Migration, with no
hard failure: a pre-AU-03 recovery code still validates and the remaining codes
are flagged for re-issue (regenerate to re-key); a pre-AU-03 app password
validates and is transparently rehashed to v2 on use. Without a master key the
legacy scheme is retained and the at-rest protection rests on SQLCipher.

opti_oignon.encryption is stubbed with a togglable master key so the real
derivation path in auth_2fa is exercised; the 2FA DB path is redirected to tmp.
"""

import hashlib
import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

import pytest

# Bare package + a db_utils stub so auth_2fa imports.
sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))

_db_utils_stub = types.ModuleType("opti_oignon.db_utils")


def _safe_connect(db_path, *, check_same_thread: bool = True, timeout: float = 5.0):
    return sqlite3.connect(
        str(db_path), check_same_thread=check_same_thread, timeout=timeout
    )


_db_utils_stub.safe_connect = _safe_connect
sys.modules["opti_oignon.db_utils"] = _db_utils_stub

# Togglable master key, exposed through a stubbed encryption module.
_MASTER = {"key": None}


class _FakeSecureBytes:
    def __init__(self, raw):
        self._raw = raw

    def as_bytes(self):
        return self._raw


_enc_stub = types.ModuleType("opti_oignon.encryption")


def _get_encryption_key():
    return _FakeSecureBytes(_MASTER["key"]) if _MASTER["key"] else None


def _get_encryption_status():
    return {"key_available": bool(_MASTER["key"]), "kdf": "argon2id"}


class _EncryptionManager:
    enabled = False
    has_key = False


_enc_stub.get_encryption_key = _get_encryption_key
_enc_stub.get_encryption_status = _get_encryption_status
_enc_stub.EncryptionManager = _EncryptionManager
sys.modules["opti_oignon.encryption"] = _enc_stub

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "auth_2fa.py"


def _load():
    spec = importlib.util.spec_from_file_location("auth_2fa_au03", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12+ dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


a2fa = _load()


@pytest.fixture(autouse=True)
def _reset_master():
    _MASTER["key"] = None
    yield
    _MASTER["key"] = None


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(a2fa, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(a2fa, "_2FA_DB_PATH", tmp_path / "auth_2fa.db")
    a2fa._init_2fa_db()
    return a2fa.TwoFactorAuthManager()


def _recovery_hashes(user_id="u"):
    conn = a2fa._get_2fa_conn()
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT code_hash FROM recovery_codes WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The key is derived (not a constant) and on a distinct domain
# ---------------------------------------------------------------------------

def test_domain_is_distinct_and_key_is_derived():
    assert a2fa._CODE_HMAC_INFO == b"oo-2fa-code-hmac-v2"
    known = {
        b"opti-oignon-sqlcipher-v1",
        b"oo-learned-router-mac-v1",
        b"opti-oignon-audit-anchor-v1",
        b"opti-oignon-audit-anchor-keyid-v1",
    }
    assert a2fa._CODE_HMAC_INFO not in known

    _MASTER["key"] = b"M" * 32
    subkey = a2fa._derive_2fa_code_key()
    assert subkey is not None
    assert isinstance(subkey, (bytes, bytearray)) and len(subkey) == 32

    h = a2fa._hash_code("abc123")
    assert h.startswith("v2:")
    # The re-key changes the output relative to the legacy scheme.
    assert h != a2fa._hash_code_legacy("abc123")


def test_no_master_key_falls_back_to_legacy():
    _MASTER["key"] = None
    assert a2fa._derive_2fa_code_key() is None
    assert not a2fa._hash_code("abc123").startswith("v2:")


# ---------------------------------------------------------------------------
# Round-trip under the new key
# ---------------------------------------------------------------------------

def test_recovery_round_trip_v2(mgr):
    _MASTER["key"] = b"M" * 32
    codes = mgr.generate_recovery_codes("u")
    assert codes
    stored = _recovery_hashes()
    assert stored and all(s.startswith("v2:") for s in stored)
    assert mgr.validate_recovery_code("u", codes[0]) is True
    assert mgr.validate_recovery_code("u", codes[0]) is False  # one-time use
    assert mgr.recovery_reissue_required("u") is False  # all v2


def test_app_password_round_trip_v2(mgr):
    _MASTER["key"] = b"M" * 32
    res = mgr.create_app_password("u", "cli")
    pw = res["password"]
    conn = a2fa._get_2fa_conn()
    try:
        stored = conn.execute(
            "SELECT password_hash FROM app_passwords WHERE user_id = 'u'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert stored.startswith("v2:")
    assert mgr.validate_app_password("u", pw) is True
    assert mgr.validate_app_password("u", "wrong") is False


# ---------------------------------------------------------------------------
# Migration: a legacy recovery hash is recognized and triggers re-issue
# ---------------------------------------------------------------------------

def test_legacy_recovery_recognized_and_reissue_then_cleared(mgr):
    _MASTER["key"] = b"M" * 32  # master key present -> new scheme is v2
    plain_a, plain_b = "aaaa1111bbbb2222", "cccc3333dddd4444"
    legacy_a = a2fa._hash_code_legacy(plain_a)
    legacy_b = a2fa._hash_code_legacy(plain_b)
    assert not legacy_a.startswith("v2:")

    conn = a2fa._get_2fa_conn()
    try:
        for h in (legacy_a, legacy_b):
            conn.execute(
                "INSERT INTO recovery_codes (user_id, code_hash, created_at) "
                "VALUES ('u', ?, 0)",
                (h,),
            )
        conn.commit()
    finally:
        conn.close()

    # Recognized, not a hard failure.
    assert mgr.validate_recovery_code("u", plain_a) is True
    # The remaining legacy code reports a needed re-issue.
    assert mgr.recovery_reissue_required("u") is True
    # Regenerating re-keys under the new scheme and clears the flag.
    assert mgr.generate_recovery_codes("u")
    assert mgr.recovery_reissue_required("u") is False


def test_legacy_sha256_recovery_recognized_without_master_key(mgr):
    _MASTER["key"] = None  # no master key ever -> old hashes were plain SHA-256
    plain = "sha256-only-code-1234"
    sha = hashlib.sha256(plain.encode("utf-8")).hexdigest()
    conn = a2fa._get_2fa_conn()
    try:
        conn.execute(
            "INSERT INTO recovery_codes (user_id, code_hash, created_at) "
            "VALUES ('u', ?, 0)",
            (sha,),
        )
        conn.commit()
    finally:
        conn.close()
    assert mgr.validate_recovery_code("u", plain) is True


# ---------------------------------------------------------------------------
# Migration: a legacy app password is recognized and rehashed to v2 on use
# ---------------------------------------------------------------------------

def test_legacy_app_password_rehashed_on_use(mgr):
    _MASTER["key"] = b"M" * 32
    plain = "legacy-app-password-xyz"
    legacy = a2fa._hash_code_legacy(plain)
    assert not legacy.startswith("v2:")

    conn = a2fa._get_2fa_conn()
    try:
        conn.execute(
            "INSERT INTO app_passwords "
            "(password_id, user_id, name, password_hash, created_at) "
            "VALUES ('p1', 'u', 'cli', ?, 0)",
            (legacy,),
        )
        conn.commit()
    finally:
        conn.close()

    assert mgr.validate_app_password("u", plain) is True
    # The stored hash is upgraded in place to the v2 scheme.
    conn = a2fa._get_2fa_conn()
    try:
        stored = conn.execute(
            "SELECT password_hash FROM app_passwords WHERE password_id = 'p1'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert stored.startswith("v2:")
    # Still validates after the upgrade.
    assert mgr.validate_app_password("u", plain) is True


# ---------------------------------------------------------------------------
# Source guards
# ---------------------------------------------------------------------------

def test_source_guards():
    src = _PATH.read_text(encoding="utf-8")
    assert 'b"oo-2fa-code-hmac-v2"' in src
    assert "def _derive_2fa_code_key(" in src
    assert "get_encryption_key" in src
    assert "def recovery_reissue_required(" in src
    assert "_verify_code(" in src

    start = src.index("def validate_app_password(")
    end = src.index("\n    def revoke_app_password(", start)
    body = src[start:end]
    assert "password_hash = ?" in body
    assert "_hash_code(password)" in body
