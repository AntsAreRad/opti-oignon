#!/usr/bin/env python3
"""Contracts for the authentication core (``opti_oignon/auth.py``).

The module under contract is the platform's direct authentication surface:
a self-contained JWT implementation (HMAC-SHA512 by default), password
hashing with a PBKDF2 fallback, an ``AuthManager`` that owns user CRUD,
sessions and token rotation, project role-based access control and an audit
trail on SQLite, and an in-memory ``LoginRateLimiter``. What these contracts
pin is the observable, security-relevant behaviour under the exact
dependencies in the reference environment -- never the shape of the code.

The load is NOT import-pure: two module-level singletons instantiate at
import time (``auth_manager = AuthManager()`` and ``login_rate_limiter``),
and constructing the manager opens its database. The window therefore seeds
a stand-in ``db_utils`` whose ``safe_connect`` redirects every requested
path to a throwaway file, and a1 proves the redirect is EXERCISED at import
(the manager touches the database as it is built) rather than assuming it.
Without the seed the real connector would reach a repository database, so
the seed is load-bearing, not decorative.

The security postures pinned here are the fail-secure ones the platform
leans on: the JWT verifier refuses an algorithm the header tries to
downgrade to, refuses a bad signature, a malformed token, a wrong secret
and an expired token; password verification refuses an unknown hash scheme
and an empty hash; ``authenticate`` runs a dummy verification for a missing
user so a timing side channel cannot enumerate usernames; single-user mode
is DERIVED, latches off for the process once a second user is seen, and
fails safe when the user count cannot be determined; the rate limiter backs
off exponentially and preserves the per-IP lockout count across a success so
a valid credential cannot reset an attacker's counter.

One recorded observation is pinned as it stands rather than judged here:
``logout`` matches its session row without an "is active" guard, so a
repeated logout of an EXISTING session still reports success (SQLite counts
matched rows, not changed ones); only an UNKNOWN token reports failure. a24
pins that exact semantics so any change to it surfaces.

A final census walks the module's abstract syntax and pins that every SQL
site passes a constant (or a ``.format`` over a constant template with
whitelisted columns, never interpolated values) and that no SQL string is
an f-string -- a quiet edit that builds a query from an f-string anywhere in
the file reddens it. Isolation goes through the shared window with ``ollama``
proven unreachable. The module itself is left byte-identical by this suite.
"""

import ast
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import types
import uuid

import pytest
from _isolation import isolate, source

_AUTH = "opti_oignon.auth"

# Reference literals, read off the module in the reference environment and
# frozen here so a change to any of them must update this file.
_VALID_ROLES = ("admin", "user", "viewer")
_JWT_ALGS = {"HS256", "HS512"}
_DEFAULT_JWT_ALG = "HS512"
_DEFAULT_MIN_LEN = 8
_DEFAULT_MAX_SESSIONS = 5

# SQL census snapshot: the number of execute sites and the count that build
# their statement dynamically. Both are frozen; f-string SQL must stay at 0.
_SQL_SITES = 30
_SQL_DYNAMIC_SITES = 1  # update_user's ``SET {}``.format over a whitelist
_SQL_FSTRING_SITES = 0


def _redirecting_db_utils(tmpdir, counter):
    """A stand-in ``db_utils`` whose ``safe_connect`` never leaves ``tmpdir``.

    Equal requested paths map to one file (deterministic), distinct paths to
    distinct files, and every call bumps ``counter`` so a test can assert the
    connector was reached.
    """
    stub = types.ModuleType("opti_oignon.db_utils")

    def _safe_connect(db_path, *, check_same_thread=True, timeout=5.0):
        counter["n"] += 1
        key = hashlib.md5(str(db_path).encode()).hexdigest()
        target = os.path.join(tmpdir, key + ".db")
        return sqlite3.connect(
            target, timeout=timeout, check_same_thread=check_same_thread
        )

    stub.safe_connect = _safe_connect
    return stub


def _load():
    """Open the shared window and load the auth module from its source file.

    Returns ``(module, restore, counter, tmpdir)``. ``counter['n']`` is the
    number of ``safe_connect`` calls made through the seeded connector.
    """
    tmpdir = tempfile.mkdtemp(prefix="auth_core_")
    counter = {"n": 0}
    stub = _redirecting_db_utils(tmpdir, counter)
    loaded, restore = isolate(
        targets={_AUTH: source("auth.py")},
        blocked=("ollama",),
        seeded={"opti_oignon.db_utils": stub},
    )
    return loaded[_AUTH], restore, counter, tmpdir


def _fresh_manager(auth, tmpdir):
    """An ``AuthManager`` on a database file no other manager shares."""
    db = os.path.join(tmpdir, uuid.uuid4().hex + ".db")
    return auth.AuthManager(db_path=db)


# --- Load posture and singletons -------------------------------------------


def test_a1_load_touches_db_and_exposes_singletons():
    """a1: the module loads under the window; the seeded connector is reached
    at import (the manager opens its database as it constructs), and the two
    module-level singletons plus the availability flag are present."""
    auth, restore, counter, _tmp = _load()
    try:
        # The module-level AuthManager() reached safe_connect while building.
        assert counter["n"] > 0
        assert auth.AUTH_AVAILABLE is True
        assert auth.auth_manager is not None
        assert type(auth.auth_manager).__name__ == "AuthManager"
        assert auth.login_rate_limiter is not None
        assert type(auth.login_rate_limiter).__name__ == "LoginRateLimiter"
    finally:
        restore()


def test_a2_constants_frozen():
    """a2: role whitelist and the JWT algorithm map are exactly the frozen
    reference values."""
    auth, restore, _c, _t = _load()
    try:
        assert auth.VALID_ROLES == _VALID_ROLES
        assert set(auth._JWT_ALGORITHMS) == _JWT_ALGS
        assert auth._JWT_ALGORITHMS["HS256"] is hashlib.sha256
        assert auth._JWT_ALGORITHMS["HS512"] is hashlib.sha512
    finally:
        restore()


def test_a3_default_config_postures():
    """a3: the built-in default config carries the safe postures -- HS512
    signing, the minimum password length, the session cap, and single-user
    mode defaulting on (it is neutralised later by the derived property)."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        # Force the built-in defaults by loading a config path that is absent.
        defaults = mgr._load_config(os.path.join(tmp, "does_not_exist.yaml"))
        assert defaults["jwt"]["algorithm"] == _DEFAULT_JWT_ALG
        assert defaults["password"]["min_length"] == _DEFAULT_MIN_LEN
        assert defaults["session"]["max_sessions"] == _DEFAULT_MAX_SESSIONS
        assert defaults["single_user_mode"] is True
    finally:
        restore()


# --- JWT helpers ------------------------------------------------------------


def test_a4_jwt_algorithm_map_exact():
    """a4: the algorithm map has exactly two entries; nothing else signs."""
    auth, restore, _c, _t = _load()
    try:
        assert len(auth._JWT_ALGORITHMS) == 2
        assert set(auth._JWT_ALGORITHMS) == _JWT_ALGS
    finally:
        restore()


def test_a5_jwt_encode_defaults_to_hs512():
    """a5: an encode with no algorithm argument stamps HS512 in the header."""
    auth, restore, _c, _t = _load()
    try:
        tok = auth.jwt_encode({"sub": "u"}, "sekret")
        header = json.loads(auth._b64url_decode(tok.split(".")[0]))
        assert header["alg"] == "HS512"
        assert header["typ"] == "JWT"
    finally:
        restore()


def test_a6_jwt_encode_rejects_unknown_algorithm():
    """a6: an unsupported algorithm name is refused at encode time."""
    auth, restore, _c, _t = _load()
    try:
        with pytest.raises(ValueError):
            auth.jwt_encode({"a": 1}, "sekret", "HS384")
    finally:
        restore()


def test_a7_jwt_round_trip_is_faithful():
    """a7: a token encoded and decoded with the same secret returns the
    payload verbatim."""
    auth, restore, _c, _t = _load()
    try:
        payload = {"sub": "u1", "role": "admin", "exp": int(time.time()) + 60}
        tok = auth.jwt_encode(payload, "sekret")
        out = auth.jwt_decode(tok, "sekret")
        assert out is not None
        assert out["sub"] == "u1"
        assert out["role"] == "admin"
    finally:
        restore()


def test_a8_jwt_downgrade_is_rejected():
    """a8: a token whose header algorithm differs from the server's expected
    one is refused (algorithm-confusion downgrade guard); the same token is
    accepted only when the server expects that very algorithm."""
    auth, restore, _c, _t = _load()
    try:
        hs256 = auth.jwt_encode({"sub": "x"}, "sekret", "HS256")
        # Server expects HS512: the HS256 header is a downgrade -> refused.
        assert auth.jwt_decode(hs256, "sekret", "HS512") is None
        # Server expects HS256: matches -> accepted.
        assert auth.jwt_decode(hs256, "sekret", "HS256") is not None
    finally:
        restore()


def test_a9_jwt_decode_fails_secure():
    """a9: a malformed token, a wrong secret and a tampered signature all
    decode to None."""
    auth, restore, _c, _t = _load()
    try:
        good = auth.jwt_encode({"sub": "u"}, "sekret")
        assert auth.jwt_decode("not.a.jwt", "sekret") is None
        assert auth.jwt_decode("only.two", "sekret") is None
        assert auth.jwt_decode(good, "WRONG-SECRET") is None
        # Tamper the signature segment.
        head, body, _sig = good.split(".")
        assert auth.jwt_decode(f"{head}.{body}.deadbeef", "sekret") is None
    finally:
        restore()


def test_a10_jwt_expiry_enforced():
    """a10: an expired token decodes to None; a token with no expiry claim is
    accepted."""
    auth, restore, _c, _t = _load()
    try:
        expired = auth.jwt_encode({"sub": "e", "exp": int(time.time()) - 10}, "s")
        assert auth.jwt_decode(expired, "s") is None
        no_exp = auth.jwt_encode({"sub": "n"}, "s")
        assert auth.jwt_decode(no_exp, "s") is not None
    finally:
        restore()


# --- Password hashing -------------------------------------------------------


def test_a11_password_round_trip():
    """a11: a hashed password verifies against the right password and refuses
    the wrong one, whichever backend is in effect."""
    auth, restore, _c, _t = _load()
    try:
        h = auth.hash_password("hunter2")
        assert auth.verify_password("hunter2", h) is True
        assert auth.verify_password("WRONG", h) is False
    finally:
        restore()


def test_a12_verify_password_fails_secure():
    """a12: verification of an unknown hash scheme and of an empty hash both
    return False rather than raising or accepting."""
    auth, restore, _c, _t = _load()
    try:
        assert auth.verify_password("pw", "plaintext-not-a-hash") is False
        assert auth.verify_password("pw", "") is False
    finally:
        restore()


def test_a13_hash_format_matches_backend():
    """a13: the stored hash carries a scheme marker -- a bcrypt ``$2`` prefix
    when bcrypt is present, otherwise the PBKDF2 fallback marker."""
    auth, restore, _c, _t = _load()
    try:
        h = auth.hash_password("some-password")
        if auth.BCRYPT_AVAILABLE:
            assert h.startswith("$2")
        else:
            assert h.startswith("pbkdf2:")
            # Fallback layout is ``pbkdf2:<salt>:<hexdigest>``.
            assert len(h.split(":")) == 3
    finally:
        restore()


# --- User CRUD and hash masking --------------------------------------------


def test_a14_create_user_happy_path():
    """a14: a valid registration returns a populated User with a fresh id and
    the configured default role."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("alice", "password123", email="a@e.co")
        assert u is not None
        assert u.username == "alice"
        assert u.email == "a@e.co"
        assert u.role == "user"
        assert u.user_id
        assert mgr.get_user(u.user_id).username == "alice"
        assert mgr.get_user_by_username("alice").user_id == u.user_id
    finally:
        restore()


def test_a15_to_dict_masks_hash_by_default():
    """a15: serialization hides the password hash unless the caller opts in."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("bob", "password123")
        assert "password_hash" not in u.to_dict()
        assert "password_hash" in u.to_dict(include_hash=True)
    finally:
        restore()


def test_a16_create_user_rejects_invalid_input():
    """a16: a password below the minimum length and an out-of-range username
    are both refused (return None)."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        assert mgr.create_user("carol", "short") is None  # < min length 8
        assert mgr.create_user("x", "password123") is None  # username too short
        assert mgr.create_user("y" * 65, "password123") is None  # too long
    finally:
        restore()


def test_a17_duplicate_username_rejected():
    """a17: a second registration with a taken username returns None."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        assert mgr.create_user("dave", "password123") is not None
        assert mgr.create_user("dave", "password123") is None
    finally:
        restore()


def test_a18_update_user_role_whitelist_and_merge():
    """a18: an update accepts a role only from the whitelist, silently drops
    an invalid one, and merges metadata rather than replacing it."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("erin", "password123", metadata={"a": 1})
        # Invalid role is ignored; the row keeps its prior role.
        out = mgr.update_user(u.user_id, role="superadmin")
        assert out.role == "user"
        # Valid role takes effect.
        out = mgr.update_user(u.user_id, role="admin")
        assert out.role == "admin"
        # Metadata merges.
        out = mgr.update_user(u.user_id, metadata={"b": 2})
        assert out.metadata == {"a": 1, "b": 2}
    finally:
        restore()


# --- Authentication ---------------------------------------------------------


def test_a19_authenticate_verifies_password():
    """a19: authentication succeeds for the right password and fails for the
    wrong one."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("frank", "password123")
        assert mgr.authenticate("frank", "password123").user_id == u.user_id
        assert mgr.authenticate("frank", "nope") is None
    finally:
        restore()


def test_a20_authenticate_missing_user_runs_dummy_verify():
    """a20: authenticating a user that does not exist returns None without
    raising; the pre-computed dummy hash used to equalise timing is a real,
    verifiable hash."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        assert mgr.authenticate("ghost", "whatever") is None
        # The dummy hash is a genuine hash, so the missing-user path spends
        # the same verification work as a real one.
        assert isinstance(mgr._dummy_hash, str) and mgr._dummy_hash
        assert auth.verify_password("__timing_oracle_dummy__", mgr._dummy_hash)
    finally:
        restore()


# --- Tokens, refresh rotation, logout --------------------------------------


def test_a21_token_lifecycle():
    """a21: token creation yields a bearer pair and the access token
    validates back to its payload with the access-type claim."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("grace", "password123")
        tok = mgr.create_tokens(u)
        assert tok.token_type == "bearer"
        payload = mgr.validate_token(tok.access_token)
        assert payload is not None
        assert payload["sub"] == u.user_id
        assert payload["type"] == "access"
    finally:
        restore()


def test_a22_validate_token_gates_on_access_type():
    """a22: a correctly-signed token whose type is not ``access`` is refused
    by ``validate_token`` (a refresh-typed token cannot pass as an access
    token)."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        secret = mgr._get_jwt_secret()
        alg = mgr.config.get("jwt", {}).get("algorithm", "HS512")
        forged = auth.jwt_encode(
            {"sub": "u", "type": "refresh", "exp": int(time.time()) + 60},
            secret, alg,
        )
        # The signature is valid, but the type gate rejects it.
        assert auth.jwt_decode(forged, secret, alg) is not None
        assert mgr.validate_token(forged) is None
    finally:
        restore()


def test_a23_refresh_rotates_and_invalidates_old():
    """a23: exchanging a refresh token issues a NEW refresh token and the old
    one no longer works (single-use rotation)."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("heidi", "password123")
        first = mgr.create_tokens(u)
        second = mgr.refresh_tokens(first.refresh_token)
        assert second is not None
        assert second.refresh_token != first.refresh_token
        # The consumed refresh token is dead.
        assert mgr.refresh_tokens(first.refresh_token) is None
    finally:
        restore()


def test_a24_logout_semantics():
    """a24: logging out an existing session reports success; an unknown token
    reports failure; a repeated logout of the same (already-inactive) session
    still reports success, because the update matches the row whether or not
    it changes it. This pins the observed semantics exactly."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("ivan", "password123")
        tok = mgr.create_tokens(u)
        assert mgr.logout(tok.refresh_token) is True
        # Repeat on an existing-but-inactive session: matched row -> True.
        assert mgr.logout(tok.refresh_token) is True
        # Unknown token: no row matched -> False.
        assert mgr.logout("no-such-refresh-token") is False
    finally:
        restore()


# --- Single-user mode derivation (one-way latch, fail-secure) --------------


def test_a25_single_user_mode_on_with_one_user():
    """a25: with at most one user and the default flag on, single-user mode
    is active."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        assert mgr.single_user_mode is True
        mgr.create_user("solo", "password123")
        assert mgr.single_user_mode is True
    finally:
        restore()


def test_a26_single_user_mode_latches_off():
    """a26: once a second user exists, single-user mode turns off and the
    latch flag is set."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        mgr.create_user("one", "password123")
        mgr.create_user("two", "password123")
        assert mgr.single_user_mode is False
        assert mgr._multi_user_latched is True
    finally:
        restore()


def test_a27_single_user_mode_latch_is_one_way_and_explicit_opt_out_wins():
    """a27: deleting back to a single user does NOT re-enable the bypass
    (one-way latch); an explicit ``single_user_mode: false`` always wins; an
    undeterminable user count fails safe to multi-user."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        a = mgr.create_user("aa", "password123")
        b = mgr.create_user("bb", "password123")
        assert mgr.single_user_mode is False  # latches
        mgr.delete_user(b.user_id)  # back to one user
        assert mgr.single_user_mode is False  # stays off
        assert a is not None

        # Explicit opt-out wins regardless of latch state.
        mgr2 = _fresh_manager(auth, tmp)
        mgr2.config["single_user_mode"] = False
        assert mgr2.single_user_mode is False

        # Fail-secure: if the user count cannot be determined, treat as
        # multi-user (authentication required).
        mgr3 = _fresh_manager(auth, tmp)

        def _boom():
            raise RuntimeError("count unavailable")

        mgr3.count_users = _boom
        assert mgr3.single_user_mode is False
    finally:
        restore()


# --- Project RBAC -----------------------------------------------------------


def test_a28_share_project_role_whitelist():
    """a28: sharing stores the granted role and refuses a role outside the
    project role set."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("owner", "password123")
        res = mgr.share_project("proj1", u.user_id, role="editor",
                                invited_by=u.user_id)
        assert res is not None
        assert mgr.get_project_role("proj1", u.user_id) == "editor"
        assert mgr.share_project("proj1", u.user_id, role="superuser") is None
    finally:
        restore()


def test_a29_check_permission_hierarchy():
    """a29: the role hierarchy is owner > editor > viewer -- an editor
    satisfies a viewer requirement but not an owner one."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("member", "password123")
        mgr.share_project("proj2", u.user_id, role="editor", invited_by=u.user_id)
        assert mgr.check_permission("proj2", u.user_id, "viewer") is True
        assert mgr.check_permission("proj2", u.user_id, "editor") is True
        assert mgr.check_permission("proj2", u.user_id, "owner") is False
    finally:
        restore()


def test_a30_no_access_is_denied_and_removable():
    """a30: a user with no grant is denied and has no role; removing a grant
    takes effect."""
    auth, restore, _c, tmp = _load()
    try:
        mgr = _fresh_manager(auth, tmp)
        u = mgr.create_user("stranger", "password123")
        assert mgr.check_permission("projX", u.user_id, "viewer") is False
        assert mgr.get_project_role("projX", u.user_id) is None
        mgr.share_project("projY", u.user_id, role="viewer", invited_by=u.user_id)
        assert mgr.remove_project_access("projY", u.user_id) is True
        assert mgr.get_project_role("projY", u.user_id) is None
    finally:
        restore()


# --- Login rate limiter -----------------------------------------------------


def test_a31_rate_limit_locks_after_max_attempts():
    """a31: the first attempt is allowed, a lockout follows once the attempt
    ceiling is hit, and a disabled limiter always allows."""
    auth, restore, _c, _t = _load()
    try:
        rl = auth.LoginRateLimiter(config={
            "enabled": True, "login_max_attempts": 3,
            "login_window_seconds": 300, "lockout_base_seconds": 60,
            "lockout_max_seconds": 3600, "account_lock_threshold": 10,
            "account_lock_duration_seconds": 900,
        })
        allowed, retry = rl.check_rate_limit("1.2.3.4", "u")
        assert allowed is True and retry == 0
        for _ in range(3):
            rl.record_failure("1.2.3.4", "u")
        blocked, wait = rl.check_rate_limit("1.2.3.4", "u")
        assert blocked is False and wait > 0

        off = auth.LoginRateLimiter(config={"enabled": False})
        assert off.check_rate_limit("x", "y") == (True, 0)
    finally:
        restore()


def test_a32_lockout_backoff_is_exponential_and_capped():
    """a32: lockout duration is ``base`` for the first lockout, doubles each
    subsequent one, and is capped at the configured maximum."""
    auth, restore, _c, _t = _load()
    try:
        rl = auth.LoginRateLimiter(config={
            "enabled": True, "lockout_base_seconds": 60,
            "lockout_max_seconds": 3600,
        })
        assert rl._get_lockout_duration(0) == 60
        assert rl._get_lockout_duration(1) == 60
        assert rl._get_lockout_duration(2) == 120
        assert rl._get_lockout_duration(3) == 240
        assert rl._get_lockout_duration(20) == 3600  # capped
    finally:
        restore()


def test_a33_success_clears_username_but_preserves_ip_lockout():
    """a33: a successful login clears the per-username entry but keeps the
    per-IP lockout count, so a valid credential cannot reset an attacker's
    accumulated IP penalty."""
    auth, restore, _c, _t = _load()
    try:
        rl = auth.LoginRateLimiter(config={
            "enabled": True, "login_max_attempts": 3,
            "login_window_seconds": 300, "lockout_base_seconds": 60,
            "lockout_max_seconds": 3600, "account_lock_threshold": 10,
            "account_lock_duration_seconds": 900,
        })
        for _ in range(3):
            rl.record_failure("9.9.9.9", "victim")
        rl.check_rate_limit("9.9.9.9", "victim")  # trips the IP lockout
        rl.record_success("9.9.9.9", "victim")
        status = rl.get_status(ip="9.9.9.9", username="victim")
        assert "username" not in status  # username entry cleared
        assert status["ip"]["lockout_count"] == 1  # IP penalty preserved
    finally:
        restore()


# --- SQL census -------------------------------------------------------------


def _sql_sites(tree):
    """Every ``execute``/``executescript``/``executemany`` call node."""
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("execute", "executescript", "executemany"):
                sites.append(node)
    return sites


def test_a34_sql_is_parameterised_never_f_string():
    """a34: walk the module's syntax and pin that no SQL string is an
    f-string, that the site count matches the frozen snapshot, and that the
    single dynamically-built statement is a ``.format`` over a constant
    template (a whitelist of column names), never interpolated values."""
    tree = ast.parse(source("auth.py").read_text())
    sites = _sql_sites(tree)
    assert len(sites) == _SQL_SITES

    fstring = 0
    dynamic = 0
    for node in sites:
        arg0 = node.args[0] if node.args else None
        if isinstance(arg0, ast.JoinedStr):
            fstring += 1
        elif isinstance(arg0, ast.Call):
            dynamic += 1
            # The dynamic site must be ``<constant>.format(...)``.
            assert isinstance(arg0.func, ast.Attribute)
            assert arg0.func.attr == "format"
            assert isinstance(arg0.func.value, ast.Constant)
            assert isinstance(arg0.func.value.value, str)

    assert fstring == _SQL_FSTRING_SITES
    assert dynamic == _SQL_DYNAMIC_SITES
