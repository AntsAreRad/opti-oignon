#!/usr/bin/env python3
"""Contracts for the recovery codes, the TOTP secret and the token when the key they lean on is not there.

With no master key that loads, recovery codes and app passwords were hashed with an HMAC under a constant anyone can
read (``oo-recovery-hmac-key-<kdf>``), under the key file's text, or with plain SHA-256: a read of the second-factor
store let a 64-bit recovery code be searched offline at SHA-256's speed. A sealed TOTP secret that did not decrypt was
handed on as the secret, so every code was invalid, counted, and the account lost its factor without a word. And a
signed token with no ``exp``, or one that is not a finite number, never expired.

  * CK1 -- with no key, a code is hashed ``s1:``: scrypt (n = 2^14, r = 8, p = 1) under a salt of its own, never the
    old scheme, and before the store's write opens; each verifies; with a key, switched on or off, ``v2:``.
  * CK2 -- the old hashes are still read: a recovery code once, the account told to make new ones; an app password
    rehashed on use, to ``v2:`` or ``s1:``; an ``s1:`` code asks no reissue; a ``v2:`` code no key can check is said.
  * CK3 -- an ``s1:`` hash that is not well formed matches no code and raises nothing.
  * CK4 -- the routes that make codes run the hashing off the event loop.
  * CK5 -- a sealed TOTP secret that does not decrypt (no key, another key, a cipher that fails, a manager that cannot
    be built) raises ``EncryptionUnavailable`` with the reason, at the sign-in and at the confirmation; switched off
    with a key that loads, it reads; a ``B64:`` secret and one stored before encryption read as before; the prefixes
    are encryption's.
  * CK6 -- the second step of the login (a TOTP code, a keyed recovery code), the enrolment and the confirmation
    answer 503 with the reason, the try counted; with the key back, the same code passes; any other error is no 503.
  * CK7 -- a token whose ``exp`` is missing, null, a string, a boolean, NaN, an infinity, too large for a float, or
    past is refused; a finite future one is read.
  * CK8 -- a8 and a9 of tests/test_auth_core_contracts.py, said again with an ``exp``.
  * CK9 -- the texts say it.
  * CK10 -- of two tries that read the same unused recovery code together, one uses it.
  * CK11 -- tf22 of tests/test_two_factor_login_frontend_contracts.py, with the confirmation's reason shown.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import base64
import hashlib
import hmac
import json
import shutil
import sqlite3
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_auth_2fa_routes_seam_contracts as seam  # noqa: E402
import test_auth_core_contracts as core  # noqa: E402
import test_two_factor_account_contracts as tf  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_ck1_with_no_key_a_code_is_hashed_with_scrypt_under_its_own_salt": 6.0,
    "test_ck2_the_old_hashes_are_read_and_replaced": 6.0,
    "test_ck3_an_s1_hash_that_is_not_well_formed_matches_nothing": 3.0,
    "test_ck4_the_routes_that_make_codes_hash_off_the_event_loop": 2.0,
    "test_ck5_a_sealed_secret_that_does_not_decrypt_is_said": 4.0,
    "test_ck6_the_routes_answer_503_with_the_reason": 8.0,
    "test_ck7_a_token_that_does_not_say_when_it_ends_is_refused": 3.0,
    "test_ck8_a8_and_a9_with_an_exp": 3.0,
    "test_ck9_the_texts_say_it": 2.0,
    "test_ck10_two_tries_of_one_recovery_code_use_it_once": 3.0,
    "test_ck11_tf22_with_the_confirmations_reason_shown": 2.0,
}

TFA = "opti_oignon.auth_2fa"
A, B = "uid-alice", "uid-bob"
K1, K2 = bytes(range(1, 33)), bytes(range(101, 133))
_CONSTANT = "oo-recovery-hmac-key-argon2id"
_KEYFILE_TEXT = "a key file's text, the first thirty-two characters of which were a key"
# A wrapped key file's text begins the same way on every machine.
_ENVELOPE = '{"version": "envelope-v1", "kdf": "argon2id", "salt": "c2FsdA==", "nonce": "bm9uY2U=", "ct": "Y3Q="}'


# --------------------------------------------------------------------------- the second-factor module alone


class _Keys:
    """What the stand-in encryption module loads: ``key`` (None: no key loads), whether the switch is on, whether its
    manager can be built, and what its cipher raises."""

    def __init__(self, key=K1):
        self.key, self.on, self.builds, self.raises = key, True, True, None


def _encryption_seed(keys):
    """The encryption module as the second-factor module reads it: a manager that seals ``ENC2:`` under the key it
    loads, loads none with the switch off unless one is given, and hands a value it cannot open back as it is, as
    the real one does."""
    module = types.ModuleType("opti_oignon.encryption")

    class EncryptionUnavailable(RuntimeError):
        pass

    EncryptionUnavailable.__module__ = "opti_oignon.encryption"

    class EncryptionManager:
        def __init__(self, key=None, enabled=None):
            if not keys.builds:
                raise OSError("the manager cannot be built")
            on = keys.on if enabled is None else enabled
            self._key = key if key is not None else (keys.key if on else None)
            self.enabled = bool(on and self._key is not None)
            self.refusing = "encryption is on and no key loads" if on and self._key is None else None

        @property
        def has_key(self):
            return self._key is not None

        def encrypt(self, plaintext):
            return "ENC2:" + base64.b64encode(self._key[:4] + plaintext.encode()).decode()

        def decrypt(self, ciphertext):
            if keys.raises is not None:
                raise keys.raises
            if self._key is None or not ciphertext.startswith("ENC2:"):
                return ciphertext
            raw = base64.b64decode(ciphertext[5:])
            return raw[4:].decode() if raw[:4] == self._key[:4] else ciphertext

    module.EncryptionManager = EncryptionManager
    module.EncryptionUnavailable = EncryptionUnavailable
    module.get_encryption_key = lambda: keys.key
    return module


class _FileStore:
    """``safe_connect`` as the product's: a connection of its own for each call, on a file in ``folder``."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def connect(self, db_path, **_kwargs):
        conn = sqlite3.connect(self.folder / Path(str(db_path)).name, timeout=5, check_same_thread=False)
        return conn

    def close_all(self):
        pass


def _world(key=K1, db=None):
    """The real second-factor module in a window, its store in memory (or ``db``), its clock and its key in the
    test's hands."""
    keys, clock, mode = _Keys(key), seam._Clock(), seam._ModeState()
    db = db if db is not None else seam._DbSeam()
    loaded, restore = isolate(
        targets={TFA: source("auth_2fa.py")},
        blocked=("ollama", "qrcode", "fido2", "fido2.server", "fido2.webauthn"),
        seeded={"opti_oignon.db_utils": seam._db_utils_seed(db), "opti_oignon.encryption": _encryption_seed(keys),
                "opti_oignon.security_mode": seam._security_mode_seed(mode), "pyotp": seam._pyotp_seed(clock)},
    )
    tfa = loaded[TFA]
    tfa.time = types.SimpleNamespace(time=clock.time)

    def close():
        restore()
        db.close_all()

    return SimpleNamespace(tfa=tfa, mgr=tfa.two_factor_manager, keys=keys, clock=clock, close=close,
                           refusal=sys.modules["opti_oignon.encryption"].EncryptionUnavailable)


def _rows(w, table, column):
    conn = w.tfa._get_2fa_conn()
    try:
        return [r[0] for r in conn.execute(f"SELECT {column} FROM {table} ORDER BY rowid").fetchall()]
    finally:
        conn.close()


def _insert(w, sql, params):
    conn = w.tfa._get_2fa_conn()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _old_forms(code):
    """The hashes of the old scheme: under the key file's text (a plain one, a wrapped one), plain SHA-256, and the
    constant the old code named."""
    return {hmac.new(_CONSTANT.encode(), code.encode(), hashlib.sha256).hexdigest(),
            hmac.new(_KEYFILE_TEXT[:32].encode(), code.encode(), hashlib.sha256).hexdigest(),
            hmac.new(_ENVELOPE[:32].encode(), code.encode(), hashlib.sha256).hexdigest(),
            hashlib.sha256(code.encode()).hexdigest()}


def _s1(stored, code):
    """Whether ``stored`` is ``s1:<salt>:<digest>`` with the digest scrypt makes of ``code`` at the stated cost."""
    prefix, salt_hex, digest_hex = stored.split(":")
    salt = bytes.fromhex(salt_hex)
    return (prefix == "s1" and len(salt) == 16
            and digest_hex == hashlib.scrypt(code.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32).hex())


def _v2(stored, code, key):
    subkey = hmac.new(key, b"oo-2fa-code-hmac-v2", hashlib.sha256).digest()
    return stored == "v2:" + hmac.new(subkey, code.encode(), hashlib.sha256).hexdigest()


def test_ck1_with_no_key_a_code_is_hashed_with_scrypt_under_its_own_salt(tmp_path):
    w = _world(key=None)
    try:
        # The key file's text is there to be read: no hash is made under it.
        (tmp_path / ".keyfile").write_text(_KEYFILE_TEXT, encoding="utf-8")
        w.tfa._DATA_DIR = tmp_path
        real_hash, held = w.tfa._hash_code_s1, []

        def watching(code):
            held.append(w.tfa._get_2fa_conn().in_transaction)
            return real_hash(code)

        real_derive, lookups = w.tfa._derive_2fa_code_key, []

        def derive():
            lookups.append(1)
            return real_derive()

        w.tfa._hash_code_s1, w.tfa._derive_2fa_code_key = watching, derive
        codes = w.mgr.generate_recovery_codes(A)
        w.tfa._hash_code_s1, w.tfa._derive_2fa_code_key = real_hash, real_derive
        assert held == [False] * 10, "each code is hashed before the store's write opens"
        assert len(lookups) == 1, "the key is looked up once for the set: its ten codes are of one scheme"
        stored = _rows(w, "recovery_codes", "code_hash")
        assert len(codes) == len(stored) == 10 and all(len(c) == 16 for c in codes)
        assert all(_s1(h, c) for h, c in zip(stored, codes)), stored
        assert len({h.split(":")[1] for h in stored}) == 10, "each code has a salt of its own"
        assert not any(h.split(":")[2] in _old_forms(c) or h in _old_forms(c) for h, c in zip(stored, codes))
        assert all(w.tfa._verify_code(c, h) == (True, False) for h, c in zip(stored, codes))
        assert w.tfa._verify_code(codes[0], stored[1]) == (False, False)
        assert w.mgr.validate_recovery_code(A, codes[3]) is True
        assert w.mgr.validate_recovery_code(A, codes[3]) is False, "a recovery code is used once"
        assert w.mgr.get_status(A).recovery_reissue_required is False

        made = w.mgr.create_app_password(A, "a client")
        held = _rows(w, "app_passwords", "password_hash")
        assert len(held) == 1 and _s1(held[0], made["password"]) and held[0] not in _old_forms(made["password"])
        assert w.mgr.validate_app_password(A, made["password"]) is True
        assert _rows(w, "app_passwords", "password_hash") == held, "an s1 hash is not rehashed"

        twice = [w.tfa._hash_code("the same code"), w.tfa._hash_code("the same code")]
        assert twice[0] != twice[1] and all(_s1(h, "the same code") for h in twice)

        # A key that loads: v2, as before, the switch on or off.
        w.keys.key = K1
        for on in (True, False):
            w.keys.on = on
            keyed = w.tfa._hash_code("the same code")
            assert _v2(keyed, "the same code", K1) and w.tfa._verify_code("the same code", keyed) == (True, False)
        w.keys.on = True
        # The s1 hashes made with no key still verify once a key loads.
        assert w.tfa._verify_code(codes[0], stored[0]) == (True, False)
    finally:
        w.close()


def _held(w):
    """The recovery tries the account's window holds this hour."""
    return [t for t in w.mgr._recovery_attempts.get(A, []) if w.clock.time() - t < 3600]


def test_ck2_the_old_hashes_are_read_and_replaced(tmp_path):
    # With no key, the old scheme wrote an HMAC under the key file's first 32 characters when a key file was there (a
    # wrapped one's are the same on every machine), else plain SHA-256: the forms the product wrote and still reads.
    (tmp_path / ".keyfile").write_text(_ENVELOPE, encoding="utf-8")

    def under_file(code):
        return hmac.new(_ENVELOPE[:32].encode(), code.encode(), hashlib.sha256).hexdigest()

    def plain(code):
        return hashlib.sha256(code.encode()).hexdigest()

    for key in (None, K1):
        w = _world(key=key)
        try:
            w.tfa._DATA_DIR = tmp_path
            old_codes = ["0123456789abcdef", "fedcba9876543210"]
            for old in (plain(old_codes[0]), under_file(old_codes[1])):
                _insert(w, "INSERT INTO recovery_codes (user_id, code_hash, created_at) VALUES (?, ?, ?)",
                        (A, old, 1.0))
            assert w.mgr.get_status(A).recovery_reissue_required is True
            assert w.mgr.recovery_reissue_required(A) is True
            assert w.mgr.validate_recovery_code(A, old_codes[0]) is True
            assert w.mgr.validate_recovery_code(A, old_codes[0]) is False
            assert w.mgr.get_status(A).recovery_reissue_required is True, "an old code is left"
            assert w.mgr.validate_recovery_code(A, old_codes[1]) is True
            w.clock.advance(3601)  # past the hour's three recovery tries
            made = w.mgr.generate_recovery_codes(A)
            assert w.mgr.get_status(A).recovery_reissue_required is False, key
            assert w.mgr.recovery_reissue_required(A) is False, key
            if key is not None:
                # Keyed codes, and then no key loads: not one can be checked; that is said, and the try goes back.
                w.keys.key = None
                with pytest.raises(w.refusal) as refused:
                    w.mgr.validate_recovery_code(A, made[0])
                assert "10 recovery code(s) are keyed and no key loads" in str(refused.value)
                assert _held(w) == [], "not one code was checked: the try goes back"
                # A set that mixes the two (a key file put back or removed while codes were made): with no key the
                # s1 codes read, and a code that matches none of them is said, its try counted.
                s1_codes = ["00112233445566ff", "ffeeddccbbaa0099"]
                for code in s1_codes:
                    _insert(w, "INSERT INTO recovery_codes (user_id, code_hash, created_at) VALUES (?, ?, ?)",
                            (A, w.tfa._hash_code_s1(code), 2.0))
                assert w.mgr.validate_recovery_code(A, s1_codes[0]) is True
                with pytest.raises(w.refusal):
                    w.mgr.validate_recovery_code(A, "0000000000000000")
                assert len(_held(w)) == 2, "the s1 code was checked: counted"
                # Switched off with the key that loads, a keyed code reads.
                w.keys.key, w.keys.on = key, False
                assert w.mgr.validate_recovery_code(A, made[0]) is True, "with its key the code reads"
                w.keys.on = True

            password = "an-app-password-made-before-the-new-scheme"
            for row, old in (("pw-1", plain(password)), ("pw-2", under_file(password + "2"))):
                _insert(w, "INSERT INTO app_passwords (password_id, user_id, name, password_hash, created_at) "
                        "VALUES (?, ?, ?, ?, ?)", (row, B, row, old, 1.0))
            assert w.mgr.validate_app_password(B, password) is True
            assert w.mgr.validate_app_password(B, password + "2") is True
            for held, word in zip(_rows(w, "app_passwords", "password_hash"), (password, password + "2")):
                assert (_s1(held, word) if key is None else _v2(held, word, key)), (key, held)
                assert w.mgr.validate_app_password(B, word) is True
        finally:
            w.close()


def test_ck3_an_s1_hash_that_is_not_well_formed_matches_nothing():
    w = _world(key=None)
    try:
        good = w.tfa._hash_code("0123456789abcdef")
        _prefix, salt, digest = good.split(":")
        assert w.tfa._verify_code("0123456789abcdef", good) == (True, False)
        assert w.tfa._verify_code("0123456789abcdee", good) == (False, False)
        for bad in ("s1:", "s1:" + salt, "s1:" + salt + ":", "s1::" + digest, f"s1:{salt}:{digest}:00",
                    f"s1:{salt[:-2]}:{digest}", f"s1:{salt}:{digest[:-2]}", f"s1:{salt}00:{digest}",
                    f"s1:{'g' * 32}:{digest}", f"s1:{salt}:{'z' * 64}", f"s1:{salt} :{digest[1:]}"):
            assert w.tfa._verify_code("0123456789abcdef", bad) == (False, False), bad
    finally:
        w.close()


# --------------------------------------------------------------------------- the event loop


_HASHING = {"generate_recovery_codes", "create_app_password", "validate_recovery_code", "validate_app_password",
            "validate_2fa"}


def _hashing_calls(path):
    """Each call of a manager method that hashes codes in a route of ``path``: (route, kind, method, off the loop)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) or not any(
                isinstance(d, ast.Call) and ast.unparse(d.func).startswith("router.") for d in node.decorator_list):
            continue
        threaded = set()
        for call in ast.walk(node):
            if (isinstance(call, ast.Call) and ast.unparse(call.func) == "run_in_threadpool" and call.args
                    and isinstance(call.args[0], ast.Lambda)):
                threaded |= {id(c) for c in ast.walk(call.args[0].body)}
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in _HASHING:
                found.append((node.name, type(node).__name__, call.func.attr, id(call) in threaded))
    return found


def test_ck4_the_routes_that_make_codes_hash_off_the_event_loop():
    security = _hashing_calls(source("api", "routes_security.py"))
    logins = _hashing_calls(source("api", "routes_auth.py"))
    assert {(r, m) for r, _k, m, _t in security} >= {("generate_recovery_codes", "generate_recovery_codes"),
                                                     ("create_app_password", "create_app_password")}, security
    assert ("login_2fa", "FunctionDef", "validate_2fa", False) in logins, logins
    for route, kind, method, threaded in security + logins:
        assert kind == "FunctionDef" or threaded, (route, method, "hashes on the event loop")


# --------------------------------------------------------------------------- the TOTP secret


def _sealed_prefixes():
    tree = ast.parse(source("encryption.py").read_text(encoding="utf-8"))
    found = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name) and t.id in ("_ENCRYPTED_PREFIX_V1", "_ENCRYPTED_PREFIX_V2")}
    return found["_ENCRYPTED_PREFIX_V2"], found["_ENCRYPTED_PREFIX_V1"]


def test_ck5_a_sealed_secret_that_does_not_decrypt_is_said():
    w = _world()
    try:
        assert w.tfa._SEALED_PREFIXES == _sealed_prefixes()
        secret = w.mgr.totp_setup(A, label="alice")["secret"]
        stored = _rows(w, "totp_config", "secret_encrypted")[0]
        assert stored.startswith("ENC2:") and secret not in stored and w.tfa._decrypt_secret(stored) == secret

        def code():
            return seam._totp_code(secret, int(w.clock.time() // 30))

        assert w.mgr.totp_verify(A, code())["success"] is True
        w.clock.advance(31)
        cases = (("key", None, None, True, "the TOTP secret is sealed and no key loads"),
                 ("key", K2, None, True, "the TOTP secret does not decrypt with the key that loads"),
                 ("raises", K1, ValueError("tag"), True, "the TOTP secret does not decrypt (ValueError)"),
                 ("builds", K1, None, False, "the TOTP secret does not decrypt (OSError)"))
        for _name, key, raises, builds, reason in cases:
            w.keys.key, w.keys.raises, w.keys.builds = key, raises, builds
            with pytest.raises(w.refusal) as refused:
                w.tfa._decrypt_secret(stored)
            assert str(refused.value).startswith(reason), str(refused.value)
            with pytest.raises(w.refusal):
                w.mgr.totp_validate(A, code())
        assert w.mgr._totp_attempts.get(A, []) == [], "no code was checked: each try went back"
        w.keys.key, w.keys.raises, w.keys.builds, w.keys.on = K1, None, True, False
        assert w.tfa._decrypt_secret(stored) == secret, "switched off, the key that loads reads what it sealed"
        assert w.mgr.totp_validate(A, code()) is True, "with its key the secret reads, the tries above counted"
        w.keys.on = True

        # The confirmation of a setup reads the secret the same way.
        other = w.mgr.totp_setup(B, label="bob")["secret"]
        w.keys.key = None
        with pytest.raises(w.refusal):
            w.mgr.totp_verify(B, seam._totp_code(other, int(w.clock.time() // 30)))
        assert w.mgr._totp_confirm_failures.get(B, []) == [], "no code was checked: the try went back"
        w.keys.key = K1
        assert w.mgr.totp_verify(B, seam._totp_code(other, int(w.clock.time() // 30)))["success"] is True

        # A B64 secret and one stored before encryption need no manager.
        w.keys.builds = False
        assert w.tfa._decrypt_secret("B64:" + base64.b64encode(b"JBSWY3DPEHPK3PXP").decode()) == "JBSWY3DPEHPK3PXP"
        assert w.tfa._decrypt_secret("JBSWY3DPEHPK3PXP") == "JBSWY3DPEHPK3PXP"
    finally:
        w.close()


def _refusing(w, loads):
    """The routes' world: its encryption stand-in given the refusal class the routes ask by name, and a key that
    loads or none at all."""
    enc = sys.modules["opti_oignon.encryption"]
    if not hasattr(enc, "EncryptionUnavailable"):
        refusal = type("EncryptionUnavailable", (RuntimeError,), {"__module__": "opti_oignon.encryption"})
        enc.EncryptionUnavailable = refusal
    enc.EncryptionManager.has_key = loads
    enc.get_encryption_key = (lambda: b"\x11" * 32) if loads else (lambda: None)


def _broken(*_args, **_kwargs):
    raise RuntimeError("an error that is no refusal of the encryption")


def test_ck6_the_routes_answer_503_with_the_reason():
    reason = "the TOTP secret is sealed and no key loads"
    w = tf._world()
    try:
        _refusing(w, True)
        secret = tf._enroll_totp(w, tf.A, tf.PW_A)
        codes = w.tfa.two_factor_manager.generate_recovery_codes(tf.A)
        first = tf._post(w, "/api/auth/login", {"username": "alice", "password": tf.PW_A}).json()
        assert first["methods"] == ["totp", "recovery"], first
        challenge = first["challenge_id"]
        _refusing(w, False)
        answer = tf._post(w, "/api/auth/login/2fa", {"challenge_id": challenge, "method": "totp",
                                                     "code": tf._code(w, secret)})
        assert answer.status_code == 503 and reason in answer.json()["detail"], answer.text
        assert w.routes._challenge_store.get(challenge)["attempts"] == 1, "the try is counted"
        keyed = tf._post(w, "/api/auth/login/2fa", {"challenge_id": challenge, "method": "recovery",
                                                    "code": codes[0]})
        assert keyed.status_code == 503 and "keyed and no key loads" in keyed.json()["detail"], keyed.text
        assert w.routes._challenge_store.get(challenge)["attempts"] == 2, "the try is counted"
        _refusing(w, True)
        signed = tf._post(w, "/api/auth/login/2fa", {"challenge_id": challenge, "method": "totp",
                                                     "code": tf._code(w, secret)})
        assert signed.status_code == 200 and signed.json()["access_token"] == "tok-" + tf.A, signed.text
        again = tf._post(w, "/api/auth/login", {"username": "alice", "password": tf.PW_A}).json()["challenge_id"]
        assert tf._post(w, "/api/auth/login/2fa", {"challenge_id": again, "method": "recovery",
                                                   "code": codes[0]}).status_code == 200, "the code reads with its key"

        # Any other error is no refusal: it is not answered 503 with its text.
        mgr = w.tfa.two_factor_manager
        again = tf._post(w, "/api/auth/login", {"username": "alice", "password": tf.PW_A}).json()["challenge_id"]
        mgr.validate_2fa = _broken
        with pytest.raises(RuntimeError, match="no refusal"):
            tf._post(w, "/api/auth/login/2fa", {"challenge_id": again, "method": "totp", "code": "123456"})
        del mgr.validate_2fa

        tf._as(w, tf.B)
        setup = tf._post(w, "/api/security/2fa/totp/setup", {"password": tf.PW_B})
        assert setup.status_code == 200, setup.text
        _refusing(w, False)
        confirm = tf._post(w, "/api/security/2fa/totp/verify", {"code": tf._code(w, setup.json()["secret"])})
        assert confirm.status_code == 503 and reason in confirm.json()["detail"], confirm.text
        _refusing(w, True)
        mgr.totp_verify = _broken
        with pytest.raises(RuntimeError, match="no refusal"):
            tf._post(w, "/api/security/2fa/totp/verify", {"code": tf._code(w, setup.json()["secret"])})
        del mgr.totp_verify
        confirm = tf._post(w, "/api/security/2fa/totp/verify", {"code": tf._code(w, setup.json()["secret"])})
        assert confirm.status_code == 200, confirm.text
    finally:
        w.close()

    w = tf._world(bulbe=True)
    try:
        _refusing(w, True)
        challenges = {}
        for name, password in (("alice", tf.PW_A), ("bob", tf.PW_B)):
            login = tf._post(w, "/api/auth/login", {"username": name, "password": password}).json()
            assert login.get("requires_2fa_enrolment") is True, login
            challenges[name] = login["challenge_id"]
        begun = tf._post(w, "/api/auth/login/2fa/enrol/begin", {"challenge_id": challenges["alice"]})
        assert begun.status_code == 200, begun.text
        _refusing(w, False)
        done = tf._post(w, "/api/auth/login/2fa/enrol/complete",
                        {"challenge_id": challenges["alice"], "code": tf._code(w, begun.json()["secret"])})
        assert done.status_code == 503 and reason in done.json()["detail"], done.text
        assert w.routes._challenge_store.get(challenges["alice"])["attempts"] == 1, "the try is counted"
        _refusing(w, True)
        done = tf._post(w, "/api/auth/login/2fa/enrol/complete",
                        {"challenge_id": challenges["alice"], "code": tf._code(w, begun.json()["secret"])})
        assert done.status_code == 200, done.text

        begun = tf._post(w, "/api/auth/login/2fa/enrol/begin", {"challenge_id": challenges["bob"]})
        assert begun.status_code == 200, begun.text
        w.tfa.two_factor_manager.totp_verify = _broken
        with pytest.raises(RuntimeError, match="no refusal"):
            tf._post(w, "/api/auth/login/2fa/enrol/complete",
                     {"challenge_id": challenges["bob"], "code": tf._code(w, begun.json()["secret"])})
        del w.tfa.two_factor_manager.totp_verify
    finally:
        w.close()


# --------------------------------------------------------------------------- the token


def _token_world():
    auth, restore, _counter, tmpdir = core._load()

    def close():
        restore()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return auth, close


def _with_exp(auth, exp, secret="s", algorithm="HS512"):
    """A token signed as jwt_encode signs, its payload's exp written as JSON writes it (NaN and Infinity too)."""
    payload = {"sub": "u", "type": "access"}
    if exp is not _ABSENT:
        payload["exp"] = exp
    return auth.jwt_encode(payload, secret, algorithm)


_ABSENT = object()


def test_ck7_a_token_that_does_not_say_when_it_ends_is_refused():
    auth, close = _token_world()
    try:
        now = int(time.time())
        nan = _with_exp(auth, float("nan"))
        assert '"exp":NaN' in base64.urlsafe_b64decode(nan.split(".")[1] + "==").decode(), "witness: NaN is sent"
        for exp in (_ABSENT, None, str(now + 600), True, False, float("nan"), float("inf"), float("-inf"),
                    10 ** 400, [now + 600], {"at": now + 600}, now - 10, now - 0.5):
            assert auth.jwt_decode(_with_exp(auth, exp), "s") is None, exp
        for exp in (now + 600, now + 600.5):
            decoded = auth.jwt_decode(_with_exp(auth, exp), "s")
            assert decoded is not None and decoded["exp"] == exp, exp
        manager = SimpleNamespace(config={"jwt": {"algorithm": "HS512"}}, _get_jwt_secret=lambda: "s")
        assert auth.AuthManager.validate_token(manager, _with_exp(auth, _ABSENT)) is None
        assert auth.AuthManager.validate_token(manager, _with_exp(auth, now + 600))["sub"] == "u"
        # No further ahead than the configured lifetime from now (60 minutes here), a minute's grace.
        assert auth.AuthManager.validate_token(manager, _with_exp(auth, now + 3600 + 30))["sub"] == "u"
        for exp in (now + 3600 + 120, now + 10 ** 9, 1e300, sys.float_info.max):
            assert auth.AuthManager.validate_token(manager, _with_exp(auth, exp)) is None, exp
        manager.config["jwt"]["access_token_expiry_minutes"] = 600
        assert auth.AuthManager.validate_token(manager, _with_exp(auth, now + 3600 + 120))["sub"] == "u"
        for lifetime in ("nan", "inf", None, "an hour"):
            manager.config["jwt"]["access_token_expiry_minutes"] = lifetime
            assert auth.AuthManager.validate_token(manager, _with_exp(auth, now + 600)) is None, lifetime
    finally:
        close()


def test_ck8_a8_and_a9_with_an_exp():
    auth, close = _token_world()
    try:
        exp = int(time.time()) + 600
        hs256 = _with_exp(auth, exp, "sekret", "HS256")
        assert auth.jwt_decode(hs256, "sekret", "HS512") is None
        assert auth.jwt_decode(hs256, "sekret", "HS256") is not None
        good = _with_exp(auth, exp, "sekret")
        assert auth.jwt_decode(good, "sekret") is not None, "witness: the token reads, so each refusal below is its own"
        assert auth.jwt_decode("not.a.jwt", "sekret") is None
        assert auth.jwt_decode("only.two", "sekret") is None
        assert auth.jwt_decode(good, "WRONG-SECRET") is None
        head, body, _sig = good.split(".")
        assert auth.jwt_decode(f"{head}.{body}.deadbeef", "sekret") is None
        forged = json.loads(base64.urlsafe_b64decode(body + "=="))
        forged["exp"] = exp + 10 ** 6
        other = base64.urlsafe_b64encode(json.dumps(forged).encode()).decode().rstrip("=")
        assert auth.jwt_decode(f"{head}.{other}.{_sig}", "sekret") is None, "a payload changed under its signature"
    finally:
        close()


# --------------------------------------------------------------------------- the texts


def test_ck9_the_texts_say_it():
    log = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    doc = (REPO / "docs" / "security" / "authentication.md").read_text(encoding="utf-8")
    tfa = source("auth_2fa.py").read_text(encoding="utf-8")
    flat = " ".join(log.split())
    assert "Recovery codes and app passwords are no longer hashed under a public constant" in flat
    assert "Contracts ck1-ck11 in `tests/test_codes_secret_and_token_hold_without_their_key_contracts.py`" in flat
    flat_doc = " ".join(doc.split())
    assert "rests on SQLCipher" not in flat_doc
    assert "it is scrypt under a salt of its own: it needs no secret" in flat_doc
    assert "A sealed secret that does not decrypt (no key loads, or another key does) is said" in flat_doc
    assert "a signed token with no expiry, or one that is not a finite number, is refused" in flat_doc
    assert "and so is one whose end lies further ahead than the configured access lifetime" in flat_doc
    assert "the account's tries are given back, since no code was checked" in flat_doc
    assert "reads as a wrong code: nothing in it tells the two apart" in flat_doc
    assert "falls back to the legacy scheme" not in " ".join(tfa.split())
    page = (REPO / "frontend" / "src" / "lib" / "components" / "settings" / "TOTPSetup.svelte").read_text(
        encoding="utf-8")
    assert "verifyError = httpRefusal(e)?.detail || e.detail || 'Verification failed.';" in page, \
        "the settings page shows the server's reason for a confirmation it refused"


# --------------------------------------------------------------------------- one recovery code, two tries


def test_ck10_two_tries_of_one_recovery_code_use_it_once(tmp_path):
    w = _world(key=None, db=_FileStore(tmp_path))
    try:
        codes = w.mgr.generate_recovery_codes(A)
        real_verify, meeting = w.tfa._verify_code, threading.Barrier(2, timeout=5)

        def verify(code, stored):
            """Both tries find the code before either writes it used."""
            out = real_verify(code, stored)
            if out[0]:
                try:
                    meeting.wait()
                except threading.BrokenBarrierError:
                    pass
            return out

        w.tfa._verify_code = verify
        with ThreadPoolExecutor(2) as pool:
            answers = list(pool.map(lambda _n: w.mgr.validate_recovery_code(A, codes[0]), range(2)))
        assert not meeting.broken, "witness: the two tries met after reading the code"
        assert sorted(answers) == [False, True], answers
        w.tfa._verify_code = real_verify
        assert w.mgr.validate_recovery_code(A, codes[1]) is True, "another code is still there"
    finally:
        w.close()


# --------------------------------------------------------------------------- tf22


def test_ck11_tf22_with_the_confirmations_reason_shown(monkeypatch):
    import test_two_factor_login_frontend_contracts as fe

    page = "frontend/src/lib/components/settings/TOTPSetup.svelte"
    monkeypatch.setitem(fe._REFUSAL_SITES, page, 1)
    with pytest.raises(AssertionError, match="the server's own reason"):
        fe.test_tf22_the_page_answers_the_second_step_and_sends_the_step_up()
    monkeypatch.setitem(fe._REFUSAL_SITES, page, 2)
    fe.test_tf22_the_page_answers_the_second_step_and_sends_the_step_up()
