#!/usr/bin/env python3
"""Grouped contracts: the auth routes and the two-factor module, one seam.

Two modules are under contract TOGETHER because production wires them
together: the route layer's lazy manager lookup resolves to the two-factor
module's singleton, and the login flow leans on that seam -- password step,
server-side challenge, second-factor step, tokens. A suite that stubbed one
side could not pin the seam itself, so this one loads BOTH from their source
files in a single shared window and proves the lookup lands on the identical
in-window singleton before driving the flow end to end (a1, a2).

Around that seam the suite pins the route-level fail-closed postures the
platform leans on and that no other suite of either module covers: the
password refusal is one fixed message that never embeds the username, so a
refusal cannot enumerate accounts (b1); the forwarded-for header is trusted
only when the TCP peer is loopback, so a remote client cannot spoof its way
past per-address limits (b2); the current-user dependency reads the
credential cookie first, falls back to a bearer header, and refuses an
absent or invalid token with 401 (b3); the hardened mode overrides the
single-user bypass on BOTH the request path and the socket path, so
disabling accounts never disables authentication there (b4); and the socket
handshake refuses a foreign browser origin even when the caller presents a
valid credential cookie (b5).

The second-factor step is bounded server-side: the challenge store's
time-to-live and attempt cap are frozen, an expired or unknown challenge is
refused, failed attempts count down and then lock the challenge (c1, c2); a
missing code is a 400 and an unreachable two-factor module is a 503 on the
second step while the password step degrades gracefully to direct tokens
(c3); and a failed refresh clears every credential cookie so no stale
credential survives in the browser (c4). A final census walks the
two-factor module's syntax tree and freezes its statement sites: every
database statement is a constant, none is dynamic, none is an f-string --
a quiet edit that interpolates a value anywhere in that file reddens it
(c5).

Isolation goes through the shared window. The two-factor module is NOT
import-pure -- it initializes its database schema as it loads -- so the
window seeds a recording database connector and a1 proves the seed is
EXERCISED at import rather than assuming it. The hardware-key and QR
libraries are proven unreachable; the one-time-password library and the
clock are deterministic stand-ins shared between the module under test and
the code generator. Both modules are left byte-identical by this suite.
"""

import ast
import asyncio
import hashlib
import hmac
import json
import sqlite3
import types

import pytest
from _isolation import isolate, source

_TFA = "opti_oignon.auth_2fa"
_ROUTES = "opti_oignon.api.routes_auth"

# Reference literals, read off the modules in the reference environment and
# frozen here so a change to any of them must update this file.
_CHALLENGE_TTL = 300
_CHALLENGE_MAX_ATTEMPTS = 5
_ACCESS_COOKIE = "oo_access_token"
_REFRESH_COOKIE = "oo_refresh_token"
_CSRF_COOKIE = "oo_csrf_token"
_LOGIN_REFUSAL = "Invalid username or password."
_UNKNOWN_CHALLENGE_REFUSAL = "Invalid or expired 2FA challenge. Please log in again."
_LOCKED_REFUSAL = "Too many 2FA attempts. Please log in again."
_MISSING_CODE_REFUSAL = "A 2FA code is required for this method."
_ABSENT_MODULE_REFUSAL = "2FA module not available."

# Statement census snapshot of the two-factor module: the number of
# execute/executescript sites, the count that build their statement
# dynamically, and the count that are f-strings. All three are frozen.
_SQL_SITES = 36
_SQL_DYNAMIC_SITES = 0
_SQL_FSTRING_SITES = 0

_RIGHT_PASSWORD = "correct-horse-battery"
_RESIDENT = "resident"
_RESIDENT_ID = "uid-resident"


class _Clock:
    """Deterministic clock shared by both modules and the code stub."""

    def __init__(self, start=1_700_000_000.0):
        self.now = float(start)

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


def _totp_code(secret, step):
    """Deterministic six-digit code for a (secret, time-step) pair."""
    digest = hashlib.sha256(f"{secret}:{step}".encode()).hexdigest()
    return str(int(digest[:8], 16) % 1_000_000).zfill(6)


def _pyotp_seed(clock):
    """Deterministic one-time-password library keyed to the shared clock."""
    module = types.ModuleType("pyotp")

    class _TOTP:
        def __init__(self, secret, *args, **kwargs):
            self.secret = secret
            self.interval = 30

        def at(self, for_time):
            return _totp_code(self.secret, int(for_time // self.interval))

        def verify(self, code, valid_window=0):
            current = int(clock.time() // self.interval)
            for offset in range(-valid_window, valid_window + 1):
                candidate = _totp_code(self.secret, current + offset)
                if hmac.compare_digest(candidate, str(code)):
                    return True
            return False

        def provisioning_uri(self, name="", issuer_name=""):
            return f"otpauth://totp/{issuer_name}:{name}"

    module.TOTP = _TOTP
    module.random_base32 = lambda: "TESTSEEDBASE32AB"
    return module


def _encryption_seed():
    """Encryption module stand-in: manager enabled, master key present."""
    import base64

    module = types.ModuleType("opti_oignon.encryption")

    class _Manager:
        enabled = True
        has_key = True

        def encrypt(self, plaintext):
            return "ENC:" + base64.b64encode(plaintext.encode()).decode()

        def decrypt(self, ciphertext):
            if not ciphertext.startswith("ENC:"):
                raise ValueError("not a stand-in ciphertext")
            return base64.b64decode(ciphertext[4:]).decode()

    module.EncryptionManager = _Manager
    module.get_encryption_key = lambda: b"\x11" * 32
    module.get_encryption_status = lambda: {"key_available": True, "kdf": "t"}
    return module


class _ModeState:
    """Flippable state behind the seeded security-mode module."""

    def __init__(self, bulbe=False):
        self.bulbe = bulbe
        self.audit = []


def _security_mode_seed(state):
    module = types.ModuleType("opti_oignon.security_mode")
    module.is_bulbe = lambda: state.bulbe

    def _audit_log(event, **kwargs):
        state.audit.append((event, kwargs))

    module._audit_log = _audit_log
    return module


class _DbSeam:
    """Recording stand-in for ``safe_connect``.

    Routes each requested path to one persistent in-memory store, counts
    every call, records every path, and absorbs ``close`` so the store
    survives the module's open/close-per-operation pattern.
    """

    def __init__(self):
        self.calls = 0
        self.paths = []
        self._stores = {}

    def connect(self, db_path, **kwargs):
        self.calls += 1
        key = str(db_path)
        self.paths.append(key)
        real = self._stores.get(key)
        if real is None:
            real = sqlite3.connect(":memory:", check_same_thread=False)
            self._stores[key] = real
        seam = self

        class _Wrap:
            def __getattr__(self, name):
                return getattr(real, name)

            def close(self):
                return None

            @property
            def row_factory(self):
                return real.row_factory

            @row_factory.setter
            def row_factory(self, value):
                real.row_factory = value

        _ = seam
        return _Wrap()

    def close_all(self):
        for real in self._stores.values():
            try:
                real.close()
            except Exception:
                pass
        self._stores.clear()


def _db_utils_seed(seam):
    module = types.ModuleType("opti_oignon.db_utils")
    module.safe_connect = seam.connect
    return module


class _User:
    def __init__(self, user_id, username=_RESIDENT):
        self.user_id = user_id
        self.username = username
        self.role = "user"

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": "",
            "role": self.role,
            "created_at": 0.0,
            "updated_at": 0.0,
            "metadata": {},
        }


class _AuthManagerStandIn:
    """Recording authentication manager behind the seeded dependency module.

    ``authenticate`` accepts exactly one (username, password) pair;
    ``validate_token`` resolves from an explicit mapping; ``refresh_tokens``
    refuses by default so the failure path is drivable; every audit call is
    recorded.
    """

    def __init__(self, single_user=False):
        self.single_user_mode = single_user
        self.config = {"users": {"allow_registration": True}}
        self.tokens = {}
        self.audit = []

    def authenticate(self, username, password):
        if username == _RESIDENT and password == _RIGHT_PASSWORD:
            return _User(_RESIDENT_ID)
        return None

    def get_user(self, user_id):
        return _User(user_id)

    def create_tokens(self, user):
        payload = {
            "access_token": "access-token-value",
            "refresh_token": "refresh-token-value",
            "token_type": "bearer",
            "expires_in": 3600,
            "user_id": user.user_id,
        }
        return types.SimpleNamespace(
            access_token=payload["access_token"],
            refresh_token=payload["refresh_token"],
            expires_in=payload["expires_in"],
            to_dict=lambda: dict(payload),
        )

    def validate_token(self, token):
        return self.tokens.get(token)

    def refresh_tokens(self, refresh_token):
        return None

    def logout(self, refresh_token):
        return True

    def count_users(self):
        return 1

    def _log_audit(self, *args, **kwargs):
        self.audit.append((args, kwargs))


def _deps_seed(manager):
    module = types.ModuleType("opti_oignon.api.deps")
    module.AUTH_AVAILABLE = True
    module.auth_manager = manager
    module.USER_SETTINGS_AVAILABLE = True
    module.user_settings_store = object()
    return module


class _Request:
    """Minimal request stand-in for the route helpers."""

    def __init__(self, host="127.0.0.1", headers=None, cookies=None):
        self.client = types.SimpleNamespace(host=host)
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.method = "POST"
        self.url = "http://127.0.0.1/probe"


class _Response:
    """Response stand-in recording cookie writes and deletions."""

    def __init__(self):
        self.set_keys = []
        self.deleted_keys = []

    def set_cookie(self, **kwargs):
        self.set_keys.append(kwargs.get("key"))

    def delete_cookie(self, **kwargs):
        self.deleted_keys.append(kwargs.get("key"))


class _Socket:
    """Minimal socket handshake stand-in."""

    def __init__(self, origin=None, cookies=None, query=None):
        self.headers = {} if origin is None else {"origin": origin}
        self.cookies = cookies or {}
        self.query_params = query or {}


def _load(*, bulbe=False, single_user=False, with_second_factor=True):
    """Open the shared window and load the modules from their source files.

    With the second factor present, BOTH modules are targets of one window
    so the route layer's lazy lookup resolves to the in-window singleton.
    Without it, only the route module loads and the two-factor module is
    proven unreachable, which is the degraded posture under contract in c3.
    Both modules' clocks are replaced by one deterministic clock shared with
    the code generator.
    """
    clock = _Clock()
    seam = _DbSeam()
    mode = _ModeState(bulbe=bulbe)
    manager = _AuthManagerStandIn(single_user=single_user)

    seeded = {
        "opti_oignon.security_mode": _security_mode_seed(mode),
        "opti_oignon.api.deps": _deps_seed(manager),
    }
    if with_second_factor:
        targets = {
            _TFA: source("auth_2fa.py"),
            _ROUTES: source("api", "routes_auth.py"),
        }
        blocked = ("ollama", "fido2", "qrcode")
        seeded["opti_oignon.db_utils"] = _db_utils_seed(seam)
        seeded["pyotp"] = _pyotp_seed(clock)
        seeded["opti_oignon.encryption"] = _encryption_seed()
    else:
        targets = {_ROUTES: source("api", "routes_auth.py")}
        blocked = ("ollama", _TFA)

    loaded, restore = isolate(
        targets=targets,
        blocked=blocked,
        seeded=seeded,
        packages=("opti_oignon.api",),
    )

    def close():
        restore()
        seam.close_all()

    routes = loaded[_ROUTES]
    routes.time = types.SimpleNamespace(time=clock.time)
    tfa = loaded.get(_TFA)
    if tfa is not None:
        tfa.time = types.SimpleNamespace(time=clock.time)

    return types.SimpleNamespace(
        routes=routes,
        tfa=tfa,
        manager=manager,
        clock=clock,
        seam=seam,
        mode=mode,
        restore=close,
    )


def _enroll(ctx, user_id=_RESIDENT_ID):
    """Enroll and verify a one-time-password secret on the shared manager."""
    setup = ctx.tfa.two_factor_manager.totp_setup(user_id)
    assert setup["success"], f"enrollment must succeed, got {setup}"
    secret = setup["secret"]
    step = int(ctx.clock.time() // 30)
    verified = ctx.tfa.two_factor_manager.totp_verify(user_id, _totp_code(secret, step))
    assert verified["success"], f"activation must succeed, got {verified}"
    return secret


def _password_step(ctx):
    """Run the password step and return the issued challenge body."""
    out = ctx.routes.login(
        ctx.routes.LoginRequest(username=_RESIDENT, password=_RIGHT_PASSWORD),
        _Request(),
        _Response(),
    )
    body = json.loads(out.body)
    assert body.get("requires_2fa") is True, f"a challenge must be issued, got {body}"
    return body


# --- The seam: load posture and the end-to-end flow -------------------------


def test_a1_grouped_load_exercises_seam_and_shares_the_manager():
    """a1: one window loads both modules; the seeded connector is exercised
    at import (the two-factor module initializes its schema as it loads, on
    its own database path); and the route layer's lazy lookup resolves to
    the IDENTICAL in-window singleton -- the seam this grouping exists for."""
    ctx = _load()
    try:
        assert ctx.seam.calls > 0, (
            "the seeded connector must be reached at import; the seed is "
            "load-bearing, not decorative"
        )
        assert str(ctx.tfa._2FA_DB_PATH) in ctx.seam.paths, (
            "the import-time touch must land on the module's own database "
            f"path, got {ctx.seam.paths}"
        )
        assert str(ctx.tfa._2FA_DB_PATH).endswith("auth_2fa.db")
        looked_up = ctx.routes._get_2fa_manager()
        assert looked_up is ctx.tfa.two_factor_manager, (
            "the route layer's lookup must resolve to the in-window "
            "singleton itself, not a copy and not None"
        )
        assert ctx.routes.router.prefix == "/api/auth"
        assert isinstance(ctx.routes._challenge_store, ctx.routes._ChallengeStore)
        assert ctx.tfa.TOTP_AVAILABLE is True
        assert ctx.tfa.WEBAUTHN_AVAILABLE is False
        assert ctx.tfa.QRCODE_AVAILABLE is False
    finally:
        ctx.restore()


def test_a2_login_flow_reaches_tokens_through_the_shared_manager():
    """a2: with an enrollment active on the SHARED singleton, the password
    step issues a challenge instead of tokens; the second step with a live
    code consumes the challenge, audits the method used, sets both
    credential cookies and returns bearer tokens for the challenged user."""
    ctx = _load()
    try:
        secret = _enroll(ctx)
        body = _password_step(ctx)
        assert body["methods"] == ["totp"], (
            f"the offered methods must reflect the enrollment, got {body}"
        )
        challenge_id = body["challenge_id"]

        ctx.clock.advance(30)
        step = int(ctx.clock.time() // 30)
        response = _Response()
        tokens = ctx.routes.login_2fa(
            ctx.routes.TwoFALoginRequest(
                challenge_id=challenge_id,
                code=_totp_code(secret, step),
                method="totp",
            ),
            _Request(),
            response,
        )
        assert tokens.token_type == "bearer"
        assert tokens.user_id == _RESIDENT_ID
        assert tokens.access_token and tokens.refresh_token
        assert {_ACCESS_COOKIE, _REFRESH_COOKIE} <= set(response.set_keys), (
            "both credential cookies must be set on success, got "
            f"{response.set_keys}"
        )
        assert ctx.routes._challenge_store.get(challenge_id) is None, (
            "the challenge must be consumed on success"
        )
        audited = [
            (args, kwargs)
            for args, kwargs in ctx.manager.audit
            if args and args[1] == "login_2fa"
        ]
        assert audited, f"the second step must be audited, got {ctx.manager.audit}"
        assert audited[0][1].get("details", {}).get("method") == "totp"
    finally:
        ctx.restore()


# --- Route-level fail-closed postures ---------------------------------------


def test_b1_login_refusal_carries_no_username_oracle():
    """b1: a wrong password for an existing name and any password for an
    unknown name produce the SAME fixed refusal, and the submitted username
    never appears in it -- a refusal cannot enumerate accounts."""
    from fastapi import HTTPException

    ctx = _load()
    try:
        details = {}
        for username in (_RESIDENT, "ghost-nobody"):
            with pytest.raises(HTTPException) as caught:
                ctx.routes.login(
                    ctx.routes.LoginRequest(username=username, password="wrong-pw"),
                    _Request(),
                    _Response(),
                )
            assert caught.value.status_code == 401
            details[username] = caught.value.detail
        assert details[_RESIDENT] == details["ghost-nobody"] == _LOGIN_REFUSAL, (
            f"the refusal must be one fixed message, got {details}"
        )
        for username, detail in details.items():
            assert username not in detail, (
                f"the submitted username must never echo into the refusal: {detail!r}"
            )
    finally:
        ctx.restore()


def test_b2_forwarded_header_trusted_only_from_localhost_peer():
    """b2: the forwarded-for header is attacker-controlled, so it is trusted
    only when the TCP peer is loopback (a local reverse proxy); a remote
    peer's spoofed header is ignored and the socket address is used."""
    ctx = _load()
    try:
        extract = ctx.routes._extract_client_ip
        spoofed = _Request(host="203.0.113.9", headers={"X-Forwarded-For": "1.2.3.4"})
        assert extract(spoofed) == "203.0.113.9", (
            "a remote peer's forwarded header must be ignored"
        )
        proxied = _Request(
            host="127.0.0.1",
            headers={"X-Forwarded-For": "198.51.100.7, 203.0.113.9"},
        )
        assert extract(proxied) == "198.51.100.7", (
            "behind a loopback proxy the first forwarded hop is the client"
        )
        bare = _Request(host="127.0.0.1")
        assert extract(bare) == "127.0.0.1"
    finally:
        ctx.restore()


def test_b3_current_user_cookie_first_bearer_fallback_closed_refusals():
    """b3: the current-user dependency resolves the credential cookie first,
    falls back to a bearer header, and refuses an absent or an invalid
    token with 401 -- never a pass-through."""
    from fastapi import HTTPException

    ctx = _load()
    try:
        ctx.manager.tokens["cookie-token"] = {
            "sub": "uid-cookie", "username": _RESIDENT, "role": "user", "type": "access",
        }
        ctx.manager.tokens["header-token"] = {
            "sub": "uid-header", "username": _RESIDENT, "role": "user", "type": "access",
        }
        via_cookie = ctx.routes._get_current_user(
            _Request(cookies={_ACCESS_COOKIE: "cookie-token"}), authorization=None
        )
        assert via_cookie["sub"] == "uid-cookie"
        via_header = ctx.routes._get_current_user(
            _Request(), authorization="Bearer header-token"
        )
        assert via_header["sub"] == "uid-header"
        for request, authorization in (
            (_Request(), None),
            (_Request(cookies={_ACCESS_COOKIE: "forged"}), None),
        ):
            with pytest.raises(HTTPException) as caught:
                ctx.routes._get_current_user(request, authorization=authorization)
            assert caught.value.status_code == 401, (
                "an absent or invalid token must be refused with 401, got "
                f"{caught.value.status_code}"
            )
    finally:
        ctx.restore()


def test_b4_hardened_mode_overrides_the_single_user_bypass():
    """b4: with accounts disabled the request path and the socket path both
    return a synthetic local admin -- UNLESS the hardened mode is active, in
    which case authentication is required regardless of that setting, on
    both paths."""
    from fastapi import HTTPException

    relaxed = _load(single_user=True, bulbe=False)
    try:
        who = relaxed.routes._get_current_user(_Request(), authorization=None)
        assert who["sub"] == "local" and who["role"] == "admin"
        via_socket = asyncio.run(relaxed.routes.authenticate_websocket(_Socket()))
        assert via_socket is not None and via_socket["sub"] == "local"
    finally:
        relaxed.restore()

    hardened = _load(single_user=True, bulbe=True)
    try:
        with pytest.raises(HTTPException) as caught:
            hardened.routes._get_current_user(_Request(), authorization=None)
        assert caught.value.status_code == 401, (
            "the hardened mode must require authentication even with "
            "accounts disabled"
        )
        via_socket = asyncio.run(hardened.routes.authenticate_websocket(_Socket()))
        assert via_socket is None, (
            "the socket path must apply the same override, got "
            f"{via_socket!r}"
        )
    finally:
        hardened.restore()


def test_b5_socket_handshake_refuses_foreign_origins():
    """b5: a socket handshake carrying a foreign browser origin is refused
    even when the caller presents a valid credential cookie; loopback
    origins and origin-less non-browser clients authenticate normally."""
    ctx = _load()
    try:
        ctx.manager.tokens["valid-token"] = {
            "sub": "uid-socket", "username": _RESIDENT, "role": "user", "type": "access",
        }
        hijacked = asyncio.run(
            ctx.routes.authenticate_websocket(
                _Socket(
                    origin="https://evil.example.com",
                    cookies={_ACCESS_COOKIE: "valid-token"},
                )
            )
        )
        assert hijacked is None, (
            "a foreign origin must be refused even with a valid credential, "
            f"got {hijacked!r}"
        )
        local = asyncio.run(
            ctx.routes.authenticate_websocket(
                _Socket(
                    origin="http://localhost:5173",
                    cookies={_ACCESS_COOKIE: "valid-token"},
                )
            )
        )
        assert local is not None and local["sub"] == "uid-socket"
        headless = asyncio.run(
            ctx.routes.authenticate_websocket(_Socket(query={"token": "valid-token"}))
        )
        assert headless is not None and headless["sub"] == "uid-socket"
        anonymous = asyncio.run(
            ctx.routes.authenticate_websocket(_Socket(origin="http://127.0.0.1:8000"))
        )
        assert anonymous is None, "no token must mean no identity on the socket"
    finally:
        ctx.restore()


# --- The second step: challenge bounds and degraded postures ----------------


def test_c1_challenge_lifecycle_expires_and_unknowns_refuse():
    """c1: the challenge store's time-to-live and attempt cap carry the
    frozen values; a stored challenge round-trips, dies when its window
    elapses, and an unknown challenge is refused with the fixed message."""
    from fastapi import HTTPException

    ctx = _load()
    try:
        store = ctx.routes._challenge_store
        assert ctx.routes._ChallengeStore.CHALLENGE_TTL_SECONDS == _CHALLENGE_TTL
        assert ctx.routes._ChallengeStore.MAX_ATTEMPTS == _CHALLENGE_MAX_ATTEMPTS
        challenge_id = store.create("uid-life", ["totp"])
        entry = store.get(challenge_id)
        assert entry is not None
        assert entry["user_id"] == "uid-life"
        assert entry["attempts"] == 0 and entry["locked"] is False
        ctx.clock.advance(_CHALLENGE_TTL + 1)
        assert store.get(challenge_id) is None, (
            "an expired challenge must be gone, not merely stale"
        )
        with pytest.raises(HTTPException) as caught:
            ctx.routes.login_2fa(
                ctx.routes.TwoFALoginRequest(
                    challenge_id="never-issued", code="000000", method="totp"
                ),
                _Request(),
                _Response(),
            )
        assert caught.value.status_code == 403
        assert caught.value.detail == _UNKNOWN_CHALLENGE_REFUSAL
    finally:
        ctx.restore()


def test_c2_second_step_attempt_cap_counts_down_then_locks():
    """c2: failed second-step attempts count down through the fixed
    remaining-attempts messages, the cap locks the challenge, and a locked
    challenge stays refused."""
    from fastapi import HTTPException

    ctx = _load()
    try:
        _enroll(ctx)
        challenge_id = _password_step(ctx)["challenge_id"]
        observed = []
        for _ in range(_CHALLENGE_MAX_ATTEMPTS + 1):
            with pytest.raises(HTTPException) as caught:
                ctx.routes.login_2fa(
                    ctx.routes.TwoFALoginRequest(
                        challenge_id=challenge_id, code="111111", method="totp"
                    ),
                    _Request(),
                    _Response(),
                )
            observed.append((caught.value.status_code, caught.value.detail))
        expected = [
            (403, f"Invalid 2FA code. {remaining} attempt(s) remaining.")
            for remaining in (4, 3, 2, 1)
        ] + [(403, _LOCKED_REFUSAL), (403, _LOCKED_REFUSAL)]
        assert observed == expected, (
            "the countdown, the lock and the locked refusal must all hold, "
            f"got {observed}"
        )
    finally:
        ctx.restore()


def test_c3_missing_code_and_absent_second_factor_fail_correctly():
    """c3: on the second step a missing code is a 400 and an unreachable
    two-factor module is a 503; on the password step the same unreachable
    module degrades gracefully to direct tokens instead of failing."""
    from fastapi import HTTPException

    grouped = _load()
    try:
        _enroll(grouped)
        challenge_id = _password_step(grouped)["challenge_id"]
        with pytest.raises(HTTPException) as caught:
            grouped.routes.login_2fa(
                grouped.routes.TwoFALoginRequest(
                    challenge_id=challenge_id, code="", method="totp"
                ),
                _Request(),
                _Response(),
            )
        assert caught.value.status_code == 400
        assert caught.value.detail == _MISSING_CODE_REFUSAL
    finally:
        grouped.restore()

    degraded = _load(with_second_factor=False)
    try:
        assert degraded.routes._get_2fa_manager() is None, (
            "the lookup must degrade to None when the module is unreachable"
        )
        tokens = degraded.routes.login(
            degraded.routes.LoginRequest(username=_RESIDENT, password=_RIGHT_PASSWORD),
            _Request(),
            _Response(),
        )
        assert tokens.token_type == "bearer" and tokens.user_id == _RESIDENT_ID, (
            "the password step must degrade to direct tokens, got "
            f"{tokens!r}"
        )
        challenge_id = degraded.routes._challenge_store.create("uid-stranded", ["totp"])
        with pytest.raises(HTTPException) as caught:
            degraded.routes.login_2fa(
                degraded.routes.TwoFALoginRequest(
                    challenge_id=challenge_id, code="123456", method="totp"
                ),
                _Request(),
                _Response(),
            )
        assert caught.value.status_code == 503
        assert caught.value.detail == _ABSENT_MODULE_REFUSAL
    finally:
        degraded.restore()


def test_c4_failed_refresh_clears_every_credential_cookie():
    """c4: a refusal on the refresh path is a 401 AND clears the access,
    refresh and cross-site cookies -- no stale credential survives in the
    browser after a failed refresh."""
    from fastapi import HTTPException

    ctx = _load()
    try:
        response = _Response()
        with pytest.raises(HTTPException) as caught:
            ctx.routes.refresh_token(
                _Request(cookies={_REFRESH_COOKIE: "stale-token"}),
                response,
                ctx.routes.RefreshRequest(refresh_token="stale-token"),
            )
        assert caught.value.status_code == 401
        assert {_ACCESS_COOKIE, _REFRESH_COOKIE, _CSRF_COOKIE} <= set(
            response.deleted_keys
        ), (
            "every credential cookie must be cleared on a failed refresh, "
            f"got {response.deleted_keys}"
        )
    finally:
        ctx.restore()


# --- Statement census -------------------------------------------------------


def test_c5_statement_census_is_frozen():
    """c5: every database statement in the two-factor module is a constant;
    the numbers of statement sites, dynamically built statements and
    f-string statements are frozen, so a quiet edit that interpolates a
    value into a statement anywhere in that file reddens this contract."""
    tree = ast.parse(source("auth_2fa.py").read_text(encoding="utf-8"))
    keywords = ("SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "PRAGMA")

    def _statementish(text):
        upper = text.upper()
        return any(keyword in upper for keyword in keywords)

    sites = 0
    dynamic = 0
    fstrings = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in ("execute", "executescript"):
            continue
        sites += 1
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant):
            continue
        if isinstance(first, ast.JoinedStr):
            parts = [
                value.value
                for value in first.values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            ]
            if any(_statementish(part) for part in parts):
                fstrings += 1
                dynamic += 1
            continue
        dynamic += 1

    assert sites == _SQL_SITES, f"statement sites moved: {sites} != {_SQL_SITES}"
    assert dynamic == _SQL_DYNAMIC_SITES, f"dynamic statements appeared: {dynamic}"
    assert fstrings == _SQL_FSTRING_SITES, f"f-string statements appeared: {fstrings}"


if __name__ == "__main__":
    import sys

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
