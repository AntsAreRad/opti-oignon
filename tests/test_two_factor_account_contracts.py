#!/usr/bin/env python3
"""Contracts for the second factor that belongs to the account (block 177).

Up to block 176, every second factor enrolled from the settings was stored under the literal identity ``admin``: the
login read the account's own id and never asked it, any session could write or overwrite it, and the second step of
the login took any method the client named, app passwords among them. Escalating to Bulbe, the kill switch, the
emergency stop and the tool-call approvals recorded the same literal as their actor.

  * TF6 -- no route of the security API acts under a literal identity: every route that reaches the second-factor
    manager, the kill switch, the mode, the emergency stop or a tool-call approval takes the session; the one
    ``"admin"`` left is a role comparison; that session is the session dependency's (``Depends``), never a value
    the request could carry; the routes no caller used (``/2fa/validate``, ``/2fa/webauthn/auth/*``) are gone.
  * TF7 -- a factor enrolled through the routes belongs to the session's account, and the login asks it of that
    account alone: the other account's status is empty and its login gives tokens at once.
  * TF8 -- the account a session stands for: no account named, or one gone, is 401 -- the status included; the
    synthetic session of
    single-user mode stands for the single owner, and with none is 403 -- the status and the lists say an account
    is needed instead; an unreadable account store is 503; no refusal writes a row.
  * TF9 -- each of the eight routes that add or remove a credential asks the session account's own password: absent
    422, the limiter untouched; wrong 403, counted, the rows unchanged; held back 429, not checked; one bcrypt 5
    refuses 503; each refusal audited by name with no secret; right, done; and one check at a time per account --
    a second step-up sent while the first is checked is 429.
  * TF10 -- own factors only, both ways: another account's key or app password is 404 and stays, another account's
    disable-all leaves the first untouched; and the session's own acts take effect -- its key listed and removed, its
    app password revoked, its TOTP disabled, and after its disable-all the login asks no second factor.
  * TF11 -- setting TOTP up over a verified secret is 409: the secret stays, and the login still asks it; a setup
    never verified can be begun again.
  * TF12 -- the login's second step takes only what its first step offered: an app password, or a method not
    offered, is a failed try with no token, and so is a recovery code made since the first step; the code is read
    in its plain form -- the spaces an app shows inside it ignored, fullwidth digits read as plain ones -- and tried
    as a recovery code only when it has one's shape (sixteen hexadecimal characters), whichever method names it and
    whatever the first step offered, ``auto`` trying any other as a TOTP code -- so that mistyped TOTP codes never
    spend the recovery codes' tries, while a well-formed wrong one spends one; and of two answers that pass together,
    one signs in.
  * TF13 -- a security key at login, the library stood in for as python-fido2 2.x takes its entities (by keyword
    alone): the first step carries its request options and the second completes against them, an assertion that
    fails leaving the challenge to the next try, and another login of the same account leaving it alone; the state
    is claimed while one answer is checked -- the same answer sent meanwhile finds no challenge --; an answer without
    an assertion is 400; a store that cannot be read when the answer arrives fails that answer and keeps the
    challenge for the next; a store that cannot record the key's last use does not refuse a verified key, whose
    challenge stays spent; a login's stale states are swept by the next; a key that cannot be asked here is not
    offered, nor a TOTP whose library is missing, and an account left with no method that can be checked is refused
    (503).
  * TF14 -- the factors stored under ``admin`` are retired on the data block 176 wrote (tests/fixtures/
    two_factor_admin_v176.sql): none remains, a real account's rows are byte-identical, one record and one audit
    carry the counts alone, nothing is decrypted, a second process retires nothing, and a store that refuses its
    deletes, or a store opened read-only, keeps the rows, audits the refusal and still serves -- its status read
    without the record's table; the record's read answers nothing, and raises nothing, where the store cannot be
    opened.
  * TF15 -- the owner is told, while no factor of theirs is active, that earlier factors were retired; another
    account never is; ``asked_at_login`` is false while single-user mode lets every request through.
  * TF16 -- escalating to Bulbe, the kill, the emergency stop and resume, and a tool call's approval or denial are
    recorded in the name of the session's account -- the owner's for the synthetic session -- and go through,
    recorded by the session's own name, when the account store cannot be read.
  * TF17 -- asking to re-enable search, and cancelling the request, are the owner's, and the confirmation is refused
    to any other account than the one that asked.
  * TF18 -- on the real account store, the real bcrypt and the real modules: the first registered account enrolls
    through the routes with its own password, the login asks it and lets the second account straight through, and
    the second account's session cannot pass the step-up with the first one's password.
  * TF19 -- deleting an account deletes its second factors, and no other's, and forgets its pending key logins: one
    begun before the deletion no longer completes, its state gone from where the login recorded it; an account
    deleted while its right answer is checked gets no token, and the page is told to log in again.
  * TF20 -- in place of seam c5: the second-factor module's statement census at its new count, every statement a
    constant.
  * TF23 -- a real security key at login, on the installed python-fido2: a key enrolled as the library stores one,
    the login's first step offering it with the library's own request options, an assertion signed by the key's
    private key passing the library's own check, and one signed by another key refused; the state spent by the
    answer that passes; a key removed while a login waits no longer answers it.
  * TF25 -- a real security key registered through the routes as the page sends it -- the library's own
    attestation, the name in ``credential_name`` --, listed under that name, then asked at login and passed: the
    registration's entities built as python-fido2 2.x takes them.

Every module under contract is loaded from its file through the shared isolation window; the routes are driven
through a minimal FastAPI application holding the real routers, the session given by an override of the session
dependency. The second-factor store is the seam suite's in-memory connector, or a file under ``tmp_path``. Nothing
reads or writes the platform's data.

Local-only (the public distribution ships no tests).
"""

import ast
import asyncio
import base64
import hashlib
import sqlite3
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_auth_2fa_routes_seam_contracts as seam  # noqa: E402
from _isolation import isolate, source  # noqa: E402

TFA = "opti_oignon.auth_2fa"
FACTORS = "opti_oignon.ceremony_factors"
ROUTES = "opti_oignon.api.routes_auth"
SEC = "opti_oignon.api.routes_security"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "two_factor_admin_v176.sql"
KEEPER = "5f0c2a8e-3b1d-4c7e-9a6f-2d4b8e1c7a90"
A, B = "uid-alice", "uid-bob"
PW_A, PW_B = "alice-own-password", "bob-own-password"
LONG = "y" * 73
_TABLES = ("totp_config", "webauthn_credentials", "recovery_codes", "app_passwords")

BUDGET_S = {
    "test_tf6_no_route_acts_under_a_literal_identity": 2.0,
    "test_tf7_a_factor_enrolled_belongs_to_the_session_account_and_the_login_asks_it": 4.0,
    "test_tf8_the_account_a_session_stands_for": 4.0,
    "test_tf9_each_route_that_adds_or_removes_a_credential_asks_the_accounts_password": 6.0,
    "test_tf10_a_session_acts_on_its_own_factors_alone": 4.0,
    "test_tf11_setting_totp_up_over_a_verified_secret_is_refused": 4.0,
    "test_tf12_the_second_step_takes_only_what_the_first_offered": 6.0,
    "test_tf13_a_security_key_at_login": 4.0,
    "test_tf14_the_factors_stored_under_admin_are_retired_on_old_data": 4.0,
    "test_tf15_the_owner_is_told_of_retired_factors": 4.0,
    "test_tf16_protective_actions_are_recorded_in_the_sessions_name": 3.0,
    "test_tf17_asking_to_re_enable_search_is_the_owners_and_bound_to_the_asker": 3.0,
    "test_tf18_the_real_account_store_and_bcrypt_carry_the_second_factor": 12.0,
    "test_tf19_deleting_an_account_deletes_its_factors": 4.0,
    "test_tf20_the_second_factor_statement_census": 2.0,
    "test_tf23_a_real_security_key_at_login": 6.0,
    "test_tf25_a_real_security_key_registered_then_asked": 6.0,
}


# --------------------------------------------------------------------------- stand-ins


class _User:
    """An account as ``auth.User`` holds one, for the routes and the step-up."""

    def __init__(self, user_id, username, password, role="user"):
        self.user_id, self.username, self.role = user_id, username, role
        self.password_hash = "hash:" + password
        self.email = ""

    def to_dict(self):
        return {"user_id": self.user_id, "username": self.username, "email": "", "role": self.role,
                "created_at": 0.0, "updated_at": 0.0, "metadata": {}}


class _Accounts:
    """The account store, shared by the auth module (the step-up, the session's account) and the API's dependency
    module (the login, the deletion). Accounts are registered in the order given; ``broken`` makes every read raise
    as an unreadable store does."""

    def __init__(self, *people):
        self.users, self.order = {}, []
        for person in people:
            self.add(*person)
        self.single_user_mode = False
        self.config = {"users": {"allow_registration": True}, "single_user_mode": False}
        self.tokens, self.audit, self.broken = {}, [], False

    def add(self, user_id, username, password, role="user"):
        self.users[user_id] = _User(user_id, username, password, role)
        self.order.append(user_id)

    def _read(self):
        if self.broken:
            raise sqlite3.OperationalError("unable to open database file")

    def get_user(self, user_id):
        self._read()
        return self.users.get(user_id)

    def owners(self):
        self._read()
        admins = sorted(uid for uid in self.order if self.users[uid].role == "admin")
        if admins or not self.order:
            return admins
        return [self.order[0]]

    def is_owner(self, user_id):
        return bool(user_id) and user_id in self.owners()

    def count_users(self):
        self._read()
        return len(self.order)

    def list_users(self, limit=100, offset=0):
        self._read()
        return [self.users[uid] for uid in self.order[offset:offset + limit]]

    def authenticate(self, username, password):
        self._read()
        return next((u for u in self.users.values() if u.username == username and u.password_hash == "hash:" + password),
                    None)

    def create_tokens(self, user):
        token = "tok-" + user.user_id
        self.tokens[token] = {"sub": user.user_id, "username": user.username, "role": user.role, "type": "access"}
        payload = {"access_token": token, "refresh_token": "ref-" + user.user_id, "token_type": "bearer",
                   "expires_in": 3600, "user_id": user.user_id}
        return SimpleNamespace(access_token=token, refresh_token=payload["refresh_token"], expires_in=3600,
                               to_dict=lambda: dict(payload))

    def validate_token(self, token):
        return self.tokens.get(token)

    def delete_user(self, user_id):
        if user_id not in self.users:
            return False
        del self.users[user_id]
        self.order.remove(user_id)
        return True

    def _log_audit(self, *args, **kwargs):
        self.audit.append(args)


class _Limiter:
    def __init__(self, allowed=True):
        self.allowed, self.asked, self.failures = allowed, [], []

    def check_rate_limit(self, ip, username):
        self.asked.append((ip, username))
        return (True, 0) if self.allowed else (False, 600)

    def record_failure(self, ip, username):
        self.failures.append((ip, username))

    def record_success(self, ip, username):
        pass


class _Verifier:
    """``verify_password`` stood in for, as bcrypt 5 answers: a password of more than 72 bytes raises."""

    def __init__(self):
        self.calls, self.threads = 0, []

    def __call__(self, password, password_hash):
        self.calls += 1
        self.threads.append(threading.get_ident())
        if len(password.encode("utf-8")) > 72:
            raise ValueError("password cannot be longer than 72 bytes, truncate manually if necessary")
        return password_hash == "hash:" + password


def _fido2_seeds(log):
    """A security-key library stood in for: its request options in python-fido2's shape, an assertion that passes
    when it signs the challenge it was given."""
    root = types.ModuleType("fido2")
    server = types.ModuleType("fido2.server")
    webauthn = types.ModuleType("fido2.webauthn")

    class Rp:
        def __init__(self, *, id, name):  # noqa: A002 - python-fido2 2.x takes its fields by keyword alone
            self.id, self.name = id, name

    class UserEntity:
        def __init__(self, *, id, name, display_name):  # noqa: A002
            self.id, self.name, self.display_name = id, name, display_name

    class Attested:
        def __init__(self, data):
            self.data = bytes(data)

    class Fido2Server:
        def __init__(self, rp):
            self.rp = rp

        def authenticate_begin(self, credentials):
            log.append(("begin", len(credentials)))
            challenge = f"Y2hhbGxlbmdl{len(log)}"  # a new one at every begin, as the library's
            options = {"publicKey": {"challenge": challenge, "rpId": self.rp.id,
                                     "allowCredentials": [{"type": "public-key", "id": "a2V5LTE"}]}}
            return options, {"challenge": challenge}

        def authenticate_complete(self, state, credentials, response):
            log.append(("complete", response.get("signature")))
            if response.get("signature") != "signed:" + state["challenge"]:
                raise ValueError("the assertion does not sign the challenge")
            return credentials[0]

    server.Fido2Server = Fido2Server
    webauthn.AttestedCredentialData = Attested
    webauthn.AuthenticatorData = object
    webauthn.PublicKeyCredentialRpEntity = Rp
    webauthn.PublicKeyCredentialUserEntity = UserEntity
    return {"fido2": root, "fido2.server": server, "fido2.webauthn": webauthn}


# --------------------------------------------------------------------------- the world


def _auth_module(accounts, limiter, verifier):
    module = types.ModuleType("opti_oignon.auth")
    module.auth_manager, module.login_rate_limiter, module.verify_password = accounts, limiter, verifier
    return module


def _deps_module(accounts):
    module = types.ModuleType("opti_oignon.api.deps")
    module.AUTH_AVAILABLE = True
    module.auth_manager = accounts
    module.USER_SETTINGS_AVAILABLE = True
    module.user_settings_store = object()
    return module


def _world(*, accounts=None, limiter=None, fido=False, bulbe=False, real_fido=False):
    """The real second-factor module, the step-up, the login and the security routes in one window, behind a FastAPI
    application; the session is ``w.current``."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    accounts = accounts if accounts is not None else _Accounts((A, "alice", PW_A), (B, "bob", PW_B))
    limiter = limiter if limiter is not None else _Limiter()
    verifier = _Verifier()
    clock, db, mode, fido_log = seam._Clock(), seam._DbSeam(), seam._ModeState(bulbe=bulbe), []
    seeded = {
        "opti_oignon.security_mode": seam._security_mode_seed(mode),
        "opti_oignon.api.deps": _deps_module(accounts),
        "opti_oignon.db_utils": seam._db_utils_seed(db),
        "pyotp": seam._pyotp_seed(clock),
        "opti_oignon.encryption": seam._encryption_seed(),
        "opti_oignon.auth": _auth_module(accounts, limiter, verifier),
    }
    blocked = ["ollama", "qrcode"]
    if fido:
        seeded.update(_fido2_seeds(fido_log))
    elif not real_fido:
        blocked += ["fido2", "fido2.server", "fido2.webauthn"]
    loaded, restore = isolate(
        targets={TFA: source("auth_2fa.py"), FACTORS: source("ceremony_factors.py"),
                 ROUTES: source("api", "routes_auth.py"), SEC: source("api", "routes_security.py")},
        blocked=tuple(blocked), seeded=seeded, packages=("opti_oignon.api",),
    )
    tfa, routes, sec = loaded[TFA], loaded[ROUTES], loaded[SEC]
    tfa.time = types.SimpleNamespace(time=clock.time)
    routes.time = types.SimpleNamespace(time=clock.time)
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(sec.router)
    w = SimpleNamespace(tfa=tfa, routes=routes, sec=sec, accounts=accounts, limiter=limiter, verifier=verifier,
                        clock=clock, db=db, mode=mode, fido_log=fido_log, current={})
    app.dependency_overrides[routes._get_current_user] = lambda: dict(w.current)
    w.client = TestClient(app)

    def close():
        w.client.close()
        restore()
        db.close_all()

    w.close = close
    return w


def _as(w, user_id, role=None):
    user = w.accounts.users.get(user_id)
    w.current = {"sub": user_id, "username": user.username if user else user_id,
                 "role": role or (user.role if user else "user"), "type": "access"}


def _post(w, path, body=None):
    return w.client.post(path, json=body or {})


def _delete(w, path, body=None):
    return w.client.request("DELETE", path, json=body or {})


def _rows(w):
    conn = w.tfa._get_2fa_conn()
    try:
        return {t: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}").fetchall()) for t in _TABLES}
    finally:
        conn.close()


def _of(rows, user_id):
    """The rows of one account, every table: the id is the first column but in recovery codes (the second)."""
    return {t: [r for r in rs if user_id in (r[0], r[1] if len(r) > 1 else None, r[2] if len(r) > 2 else None)]
            for t, rs in rows.items()}


def _code(w, secret):
    return seam._totp_code(secret, int(w.clock.time() // 30))


def _enroll_totp(w, user_id, password):
    _as(w, user_id)
    setup = _post(w, "/api/security/2fa/totp/setup", {"password": password})
    assert setup.status_code == 200, setup.text
    verify = _post(w, "/api/security/2fa/totp/verify", {"code": _code(w, setup.json()["secret"])})
    assert verify.status_code == 200, verify.text
    w.clock.advance(31)  # past the step the verification consumed
    return setup.json()["secret"]


def _key_row(w, user_id, credential_id="key-1"):
    conn = w.tfa._get_2fa_conn()
    try:
        conn.execute("INSERT INTO webauthn_credentials (credential_id, public_key, user_id, name, created_at) "
                     "VALUES (?, ?, ?, ?, ?)",
                     (credential_id, base64.urlsafe_b64encode(b"public-key").decode().rstrip("="), user_id, "Key",
                      w.clock.time()))
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- TF6


def test_tf6_no_route_acts_under_a_literal_identity():
    tree = ast.parse(source("api", "routes_security.py").read_text(encoding="utf-8"))
    literals = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "admin"]
    roles = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
             and any(isinstance(c, ast.Constant) and c.value == "admin" for c in n.comparators)]
    assert len(literals) == 1 and len(roles) == 1 and "role" in ast.unparse(roles[0]), [ast.unparse(n)
                                                                                         for n in literals]
    routes = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and any(
                isinstance(d, ast.Call) and ast.unparse(d.func).startswith("router.") for d in node.decorator_list):
            routes[node.name] = node
    holders = {"two_factor_manager", "search_killswitch", "security_mode_manager", "_emergency_stop",
               "tool_call_approval"}
    acting = {"kill", "request_reenable", "escalate_to_bulbe", "stop", "resume", "approve", "deny"}
    reached = []
    for name, node in routes.items():
        calls = [ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)]
        touches = [c for c in calls if c.split(".")[0] in holders
                   and (c.startswith("two_factor_manager.") or c.split(".")[-1] in acting)]
        if touches:
            reached.append(name)
            args = node.args.args + node.args.kwonlyargs
            defaults = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):] if node.args.defaults
                                else [], node.args.defaults))
            defaults.update({a.arg: d for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None})
            params = [a.arg for a in args]
            assert "current" in params or "account" in params, (name, touches)
            for param, dependency in (("current", "Depends(_get_current_user)"),
                                      ("account", "Depends(_session_account)")):
                if param in params:
                    assert param in defaults and ast.unparse(defaults[param]) == dependency, (name, param)
    assert len(reached) >= 20, reached
    assert not {"validate_2fa", "webauthn_auth_begin", "webauthn_auth_complete"} & set(routes)


# --------------------------------------------------------------------------- TF7


def test_tf7_a_factor_enrolled_belongs_to_the_session_account_and_the_login_asks_it():
    w = _world()
    try:
        _as(w, A)
        setup = _post(w, "/api/security/2fa/totp/setup", {"password": PW_A})
        assert setup.status_code == 200 and setup.json()["uri"].endswith(":alice"), setup.text
        assert _post(w, "/api/security/2fa/totp/verify",
                     {"code": _code(w, setup.json()["secret"])}).status_code == 200
        rows = _rows(w)
        assert [r[0] for r in rows["totp_config"]] == [A], rows["totp_config"]
        _as(w, B)
        assert w.client.get("/api/security/2fa/status").json()["any_method_active"] is False
        _as(w, A)
        assert w.client.get("/api/security/2fa/status").json()["any_method_active"] is True
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert first.get("requires_2fa") is True and first["methods"] == ["totp"], first
        straight = _post(w, "/api/auth/login", {"username": "bob", "password": PW_B}).json()
        assert straight.get("access_token") == "tok-" + B and "requires_2fa" not in straight, straight
    finally:
        w.close()


# --------------------------------------------------------------------------- TF8


def test_tf8_the_account_a_session_stands_for():
    w = _world()
    try:
        before = _rows(w)
        for session, status in (({}, 401), ({"sub": ""}, 401), ({"sub": "uid-gone", "role": "user"}, 401)):
            w.current = dict(session)
            answer = _post(w, "/api/security/2fa/totp/verify", {"code": "000000"})
            assert answer.status_code == status, (session, answer.text)
            looked = w.client.get("/api/security/2fa/status")
            assert looked.status_code == status, (session, looked.text)
        w.accounts.broken = True
        _as(w, A)
        assert _post(w, "/api/security/2fa/totp/verify", {"code": "000000"}).status_code == 503
        w.accounts.broken = False
        assert _rows(w) == before, "no refusal writes a row"
    finally:
        w.close()
    # The synthetic session of single-user mode stands for the single owner; with no account, it stands for none.
    for people, expected in ((((A, "alice", PW_A),), "owner"), ((), "none")):
        w = _world(accounts=_Accounts(*people))
        try:
            w.current = {"sub": "local", "username": "local", "role": "admin", "type": "access"}
            status = w.client.get("/api/security/2fa/status")
            assert status.status_code == 200, status.text
            setup = _post(w, "/api/security/2fa/totp/setup", {"password": PW_A})
            if expected == "owner":
                assert "account_required" not in status.json() and setup.status_code == 200, setup.text
                assert [r[0] for r in _rows(w)["totp_config"]] == [A]
            else:
                assert status.json().get("account_required") is True, status.json()
                assert setup.status_code == 403 and "Create the account first" in setup.json()["detail"]
                assert _rows(w)["totp_config"] == []
                for path, key in (("/api/security/2fa/webauthn/credentials", "credentials"),
                                  ("/api/security/2fa/app-passwords", "passwords")):
                    listed = w.client.get(path)
                    assert listed.status_code == 200 and listed.json()[key] == [], (path, listed.text)
                    assert listed.json().get("account_required") is True, (path, listed.json())
        finally:
            w.close()


# --------------------------------------------------------------------------- TF9

_MUTATING = (
    ("POST", "/api/security/2fa/totp/setup", {}),
    ("DELETE", "/api/security/2fa/totp", {}),
    ("POST", "/api/security/2fa/webauthn/register/begin", {}),
    ("DELETE", "/api/security/2fa/webauthn/credentials/key-1", {}),
    ("POST", "/api/security/2fa/recovery-codes/generate", {}),
    ("POST", "/api/security/2fa/app-passwords", {"name": "cli"}),
    ("DELETE", "/api/security/2fa/app-passwords/pw-none", {}),
    ("DELETE", "/api/security/2fa/all", {}),
)


def _send(w, method, path, body):
    return _post(w, path, body) if method == "POST" else _delete(w, path, body)


def test_tf9_each_route_that_adds_or_removes_a_credential_asks_the_accounts_password():
    for method, path, extra in _MUTATING:
        w = _world()
        try:
            _as(w, A)
            before = _rows(w)
            # Absent: 422 before anything, the limiter untouched.
            assert _send(w, method, path, dict(extra)).status_code == 422, path
            assert w.limiter.asked == [] and w.verifier.calls == 0, path
            # Wrong: 403, counted for this account, nothing changed, audited with no secret.
            wrong = _send(w, method, path, {**extra, "password": "not-alices"})
            assert wrong.status_code == 403, (path, wrong.text)
            assert w.limiter.failures == [("testclient", "alice")], (path, w.limiter.failures)
            assert _rows(w) == before, path
            refused = [d for e, d in w.mode.audit if e == "account_password_refused"]
            assert refused and refused[-1]["error"] == "wrong_password" and refused[-1]["user_id"] == A, path
            assert "not-alices" not in repr(w.mode.audit) + wrong.text, path
            # One bcrypt 5 refuses: 503.
            assert _send(w, method, path, {**extra, "password": LONG}).status_code == 503, path
            # Right: done (or, where there is nothing to act on, its own answer -- never a refusal of the password).
            done = _send(w, method, path, {**extra, "password": PW_A})
            assert done.status_code in (200, 400, 404), (path, done.status_code, done.text)
            assert done.status_code != 400 or "register/begin" in path, (path, done.text)
        finally:
            w.close()
        # Held back: 429, the password never checked.
        w = _world(limiter=_Limiter(allowed=False))
        try:
            _as(w, A)
            held = _send(w, method, path, {**extra, "password": PW_A})
            assert held.status_code == 429 and w.verifier.calls == 0, (path, held.text)
        finally:
            w.close()
    # One check at a time per account: a step-up sent while another is checked is refused at once, unchecked.
    w = _world()
    try:
        cf = sys.modules[FACTORS]
        entered, release = threading.Event(), threading.Event()
        real_verify = w.verifier.__call__

        def slow(password, password_hash):
            entered.set()
            release.wait(10)
            return real_verify(password, password_hash)

        sys.modules["opti_oignon.auth"].verify_password = slow
        first = []
        worker = threading.Thread(target=lambda: first.append(
            cf.check_own_password({"sub": A}, PW_A, client_ip="198.51.100.7")))
        worker.start()
        assert entered.wait(10)
        busy = cf.check_own_password({"sub": A}, PW_A, client_ip="198.51.100.7")
        assert (busy.passed, busy.status, busy.error) == (False, 429, "password_busy"), busy
        release.set()
        worker.join(10)
        assert first and first[0].passed
    finally:
        w.close()


# --------------------------------------------------------------------------- TF10


def test_tf10_a_session_acts_on_its_own_factors_alone():
    w = _world()
    try:
        _enroll_totp(w, A, PW_A)
        _key_row(w, A)
        made = _post(w, "/api/security/2fa/app-passwords", {"name": "cli", "password": PW_A})
        assert made.status_code == 200
        alice = _of(_rows(w), A)
        _as(w, B)
        assert _delete(w, "/api/security/2fa/webauthn/credentials/key-1", {"password": PW_B}).status_code == 404
        password_id = made.json()["password_id"]
        assert _delete(w, f"/api/security/2fa/app-passwords/{password_id}", {"password": PW_B}).status_code == 404
        assert _delete(w, "/api/security/2fa/all", {"password": PW_B}).status_code == 200
        assert _of(_rows(w), A) == alice, "another account's actions leave the first one's factors as they were"
        assert w.client.get("/api/security/2fa/webauthn/credentials").json()["credentials"] == []
        assert w.client.get("/api/security/2fa/app-passwords").json()["passwords"] == []
        # The session's own acts take effect.
        _as(w, A)
        listed = w.client.get("/api/security/2fa/webauthn/credentials").json()["credentials"]
        assert [c["credential_id"] for c in listed] == ["key-1"], listed
        assert _delete(w, "/api/security/2fa/webauthn/credentials/key-1", {"password": PW_A}).status_code == 200
        assert _of(_rows(w), A)["webauthn_credentials"] == [], "its key is removed"
        revoked = _delete(w, f"/api/security/2fa/app-passwords/{password_id}", {"password": PW_A})
        assert revoked.status_code == 200, revoked.text
        assert [(p["password_id"], p["revoked"]) for p in
                w.client.get("/api/security/2fa/app-passwords").json()["passwords"]] == [(password_id, 1)], \
            "its app password is revoked"
        disabled = _delete(w, "/api/security/2fa/totp", {"password": PW_A})
        assert disabled.status_code == 200 and disabled.json().get("success") is True, disabled.text
        assert w.client.get("/api/security/2fa/status").json()["totp_verified"] is False, "its TOTP is off"
        _enroll_totp(w, A, PW_A)
        _key_row(w, A, "key-2")
        assert _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json().get("requires_2fa")
        _as(w, A)
        assert _delete(w, "/api/security/2fa/all", {"password": PW_A}).status_code == 200
        after = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert "requires_2fa" not in after and after.get("access_token") == "tok-" + A, after
    finally:
        w.close()


# --------------------------------------------------------------------------- TF11


def test_tf11_setting_totp_up_over_a_verified_secret_is_refused():
    w = _world()
    try:
        _enroll_totp(w, A, PW_A)
        before = _rows(w)
        again = _post(w, "/api/security/2fa/totp/setup", {"password": PW_A})
        assert again.status_code == 409 and "disable it first" in again.json()["detail"], again.text
        assert _rows(w) == before
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert first.get("requires_2fa") is True and "totp" in first["methods"], first
        # A setup never verified is no active factor: another account begins one, and begins it again.
        _as(w, B)
        assert _post(w, "/api/security/2fa/totp/setup", {"password": PW_B}).status_code == 200
        again = _post(w, "/api/security/2fa/totp/setup", {"password": PW_B})
        assert again.status_code == 200, again.text
    finally:
        w.close()


# --------------------------------------------------------------------------- TF12


def test_tf12_the_second_step_takes_only_what_the_first_offered():
    w = _world()
    try:
        secret = _enroll_totp(w, A, PW_A)
        app_password = _post(w, "/api/security/2fa/app-passwords",
                             {"name": "cli", "password": PW_A}).json()["password"]
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert first["methods"] == ["totp"], first
        challenge = first["challenge_id"]
        for method, code in (("app_password", app_password), ("recovery", "0" * 16), ("webauthn", "")):
            answer = _post(w, "/api/auth/login/2fa", {"challenge_id": challenge, "method": method, "code": code})
            assert answer.status_code == 403 and "not offered" in answer.json()["detail"], (method, answer.text)
            assert "access_token" not in answer.text and not answer.cookies, method
        assert w.routes._challenge_store.get(challenge)["attempts"] == 3, "each one is a failed try"
        auto = _post(w, "/api/auth/login/2fa", {"challenge_id": challenge, "method": "auto", "code": app_password})
        assert auto.status_code == 403, "auto stays within the offered methods"
        right = _post(w, "/api/auth/login/2fa", {"challenge_id": challenge, "method": "auto", "code": _code(w, secret)})
        assert right.status_code == 200 and right.json()["access_token"] == "tok-" + A, right.text
    finally:
        w.close()
    # A recovery code made since the first step was not offered by it: auto does not try it.
    w = _world()
    try:
        _enroll_totp(w, A, PW_A)
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert first["methods"] == ["totp"], first
        _as(w, A)
        codes = _post(w, "/api/security/2fa/recovery-codes/generate", {"password": PW_A}).json()["codes"]
        late = _post(w, "/api/auth/login/2fa", {"challenge_id": first["challenge_id"], "method": "auto",
                                                "code": codes[0]})
        assert late.status_code == 403 and "access_token" not in late.text, late.text
        assert not w.tfa.two_factor_manager._recovery_attempts.get(A), "a method not offered is not tried"
    finally:
        w.close()
    # The code's shape: mistyped TOTP codes -- a digit dropped, a space, a wrong six -- never spend the recovery
    # codes' tries; a recovery code passes typed in capitals with a space; a well-formed wrong one spends one try;
    # the right TOTP code passes with the space the apps show.
    w = _world()
    try:
        secret = _enroll_totp(w, A, PW_A)
        codes = _post(w, "/api/security/2fa/recovery-codes/generate", {"password": PW_A}).json()["codes"]
        assert all(len(c) == 16 for c in codes) and w.tfa.two_factor_manager.recovery_shaped(codes[0]), codes
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert first["methods"] == ["totp", "recovery"], first
        step = int(w.clock.time() // 30)
        valid = {seam._totp_code(secret, step + d) for d in (-1, 0, 1)}
        wrong6 = next(c for c in ("000000", "111111", "222222", "333333") if c not in valid)
        spaced = next(c for c in ("000 001", "111 112", "222 223", "333 334") if c.replace(" ", "") not in valid)
        for wrong in (wrong6, "42725", spaced):
            miss = _post(w, "/api/auth/login/2fa", {"challenge_id": first["challenge_id"], "method": "auto",
                                                    "code": wrong})
            assert miss.status_code == 403, (wrong, miss.text)
        assert not w.tfa.two_factor_manager._recovery_attempts.get(A), "no recovery try was spent"
        typed = codes[0][:8].upper() + " " + codes[0][8:].upper()
        recovered = _post(w, "/api/auth/login/2fa", {"challenge_id": first["challenge_id"], "method": "auto",
                                                     "code": typed})
        assert recovered.status_code == 200 and recovered.json()["access_token"] == "tok-" + A, recovered.text
        second = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        spent = len(w.tfa.two_factor_manager._recovery_attempts.get(A, []))
        shaped = _post(w, "/api/auth/login/2fa", {"challenge_id": second["challenge_id"], "method": "auto",
                                                  "code": "f" * 16})
        assert shaped.status_code == 403, shaped.text
        assert len(w.tfa.two_factor_manager._recovery_attempts.get(A, [])) == spent + 1, \
            "a recovery code's shape is one recovery try"
        right = _code(w, secret)
        spaced = _post(w, "/api/auth/login/2fa", {"challenge_id": second["challenge_id"], "method": "auto",
                                                  "code": right[:3] + " " + right[3:]})
        assert spaced.status_code == 200 and spaced.json()["access_token"] == "tok-" + A, spaced.text
        # The next step's code, typed in fullwidth digits with a no-break space and a tab: its plain form passes.
        w.clock.advance(30)
        plain = _code(w, secret)
        fullwidth = "".join(chr(0xFF10 + int(d)) for d in plain)
        third = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        wide = fullwidth[:2] + "\u00a0" + fullwidth[2:4] + "\t" + fullwidth[4:]
        folded = _post(w, "/api/auth/login/2fa", {"challenge_id": third["challenge_id"], "method": "auto",
                                                  "code": wide})
        assert folded.status_code == 200 and folded.json()["access_token"] == "tok-" + A, folded.text
    finally:
        w.close()
    # The shape holds whatever the first step offered and whatever method is named: beside a key, a code of six
    # digits spends no recovery try; sixteen characters that are not hexadecimal are no recovery code; "recovery"
    # named with a code of another shape is refused without a try.
    w = _world(fido=True)
    try:
        _key_row(w, A)
        _as(w, A)
        codes = _post(w, "/api/security/2fa/recovery-codes/generate", {"password": PW_A}).json()["codes"]
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert first["methods"] == ["webauthn", "recovery"], first
        for method, wrong in (("auto", "123456"), ("auto", "recovery-code-xy"), ("recovery", "42725")):
            miss = _post(w, "/api/auth/login/2fa", {"challenge_id": first["challenge_id"], "method": method,
                                                    "code": wrong})
            assert miss.status_code == 403, (method, wrong, miss.text)
        assert not w.tfa.two_factor_manager._recovery_attempts.get(A), "no recovery try was spent"
        named = _post(w, "/api/auth/login/2fa", {"challenge_id": first["challenge_id"], "method": "recovery",
                                                 "code": codes[0]})
        assert named.status_code == 200 and named.json()["access_token"] == "tok-" + A, named.text
    finally:
        w.close()
    # Of two answers that pass together, one signs in: the challenge is consumed once.
    w = _world()
    try:
        _enroll_totp(w, A, PW_A)
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        together = threading.Barrier(2, timeout=10)

        def passing(user_id, code, method):
            together.wait()
            return {"success": True, "method": method}

        w.tfa.two_factor_manager.validate_2fa = passing
        answers = []

        def send():
            answers.append(_post(w, "/api/auth/login/2fa", {"challenge_id": first["challenge_id"], "method": "auto",
                                                            "code": "123456"}))

        senders = [threading.Thread(target=send) for _ in range(2)]
        for sender in senders:
            sender.start()
        for sender in senders:
            sender.join(20)
        assert not together.broken, "both answers passed their check together"
        assert sorted(a.status_code for a in answers) == [200, 403], [a.text for a in answers]
        assert sum("access_token" in a.text for a in answers) == 1
    finally:
        w.close()


# --------------------------------------------------------------------------- TF13


def test_tf13_a_security_key_at_login():
    w = _world(fido=True)
    try:
        _key_row(w, A)
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert first["methods"] == ["webauthn"], first
        given = first["webauthn_options"]["publicKey"]["challenge"]
        wrong = _post(w, "/api/auth/login/2fa", {"challenge_id": first["challenge_id"], "method": "webauthn",
                                                 "webauthn_response": {"signature": "signed:another"}})
        assert wrong.status_code == 403, wrong.text
        # Another login of the same account, begun meanwhile, leaves this one's challenge alone.
        other = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert other["webauthn_options"]["publicKey"]["challenge"] != given, other
        # The same challenge, tried again with the key's right answer: a failed assertion did not spend it.
        right = _post(w, "/api/auth/login/2fa", {"challenge_id": first["challenge_id"], "method": "webauthn",
                                                 "webauthn_response": {"signature": "signed:" + given}})
        assert right.status_code == 200 and right.json()["access_token"] == "tok-" + A, right.text
        # No assertion at all: 400, before anything is checked.
        third = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        bare = _post(w, "/api/auth/login/2fa", {"challenge_id": third["challenge_id"], "method": "webauthn"})
        assert bare.status_code == 400, bare.text
    finally:
        w.close()
    # The state is claimed while one answer is checked: the same answer sent meanwhile finds no challenge.
    w = _world(fido=True)
    try:
        _key_row(w, A)
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        given = first["webauthn_options"]["publicKey"]["challenge"]
        slot = w.routes._challenge_store.get(first["challenge_id"])["webauthn_key"]
        server = sys.modules["fido2.server"].Fido2Server
        real_complete = server.authenticate_complete
        inside, release = threading.Event(), threading.Event()

        def held(self, state, credentials, response):
            inside.set()
            release.wait(10)
            return real_complete(self, state, credentials, response)

        server.authenticate_complete = held
        manager = w.tfa.two_factor_manager
        outcome = []
        checking = threading.Thread(target=lambda: outcome.append(
            manager.webauthn_auth_complete(A, {"signature": "signed:" + given}, key=slot)))
        checking.start()
        assert inside.wait(10)
        meanwhile = manager.webauthn_auth_complete(A, {"signature": "signed:" + given}, key=slot)
        release.set()
        checking.join(10)
        server.authenticate_complete = real_complete
        assert meanwhile == {"success": False, "error": "no_challenge"}, meanwhile
        assert outcome and outcome[0]["success"] is True, outcome
        # A login's stale states are swept by the next login's.
        _post(w, "/api/auth/login", {"username": "alice", "password": PW_A})
        w.clock.advance(301)
        _post(w, "/api/auth/login", {"username": "alice", "password": PW_A})
        assert len([k for k in manager._webauthn_challenges if k.startswith("login:")]) == 1
    finally:
        w.close()
    # A store that cannot be read when the answer arrives fails that answer and keeps the challenge for the next.
    w = _world(fido=True)
    try:
        _key_row(w, A)
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        given = first["webauthn_options"]["publicKey"]["challenge"]
        manager = w.tfa.two_factor_manager
        held, reads = manager._held_credentials, []

        def failing_once(user_id):
            reads.append(user_id)
            if len(reads) == 1:
                raise sqlite3.OperationalError("database is locked")
            return held(user_id)

        manager._held_credentials = failing_once
        answer = {"challenge_id": first["challenge_id"], "method": "webauthn",
                  "webauthn_response": {"signature": "signed:" + given}}
        assert _post(w, "/api/auth/login/2fa", answer).status_code == 403
        kept = _post(w, "/api/auth/login/2fa", answer)
        assert kept.status_code == 200 and kept.json()["access_token"] == "tok-" + A, kept.text
    finally:
        w.close()
    # A store that cannot record the key's last use does not refuse a verified key; its challenge stays spent.
    w = _world(fido=True)
    try:
        _key_row(w, A)
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        given = first["webauthn_options"]["publicKey"]["challenge"]
        opened = w.tfa._get_2fa_conn

        class _NoWrites:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args):
                if sql.lstrip().upper().startswith("UPDATE"):
                    raise sqlite3.OperationalError("attempt to write a readonly database")
                return self._real.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._real, name)

        w.tfa._get_2fa_conn = lambda: _NoWrites(opened())
        passed = _post(w, "/api/auth/login/2fa", {"challenge_id": first["challenge_id"], "method": "webauthn",
                                                  "webauthn_response": {"signature": "signed:" + given}})
        assert passed.status_code == 200 and passed.json()["access_token"] == "tok-" + A, passed.text
        assert not [k for k in w.tfa.two_factor_manager._webauthn_challenges if k.startswith("login:")]
    finally:
        w.close()
    # Without the library: a key cannot be asked. Alone, it leaves no method: 503. Beside a TOTP, it is not offered.
    w = _world()
    try:
        _key_row(w, A)
        alone = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A})
        assert alone.status_code == 503 and "access_token" not in alone.text, alone.text
        _enroll_totp(w, A, PW_A)
        beside = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert beside["methods"] == ["totp"] and "webauthn_options" not in beside, beside
        # Without the library that checks a TOTP code, a verified TOTP is not offered either: nothing is left.
        w.tfa.TOTP_AVAILABLE = False
        none = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A})
        assert none.status_code == 503 and "access_token" not in none.text, none.text
    finally:
        w.close()


# --------------------------------------------------------------------------- TF14


def _file_store(tmp_path, name, *, refuse_deletes=False, read_only=False):
    """The real second-factor module over a file store under ``tmp_path``; ``refuse_deletes`` makes every DELETE
    fail as a read-only store does."""
    target = tmp_path / name

    class _Refusing:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, attr):
            return getattr(self._real, attr)

        def execute(self, sql, *args):
            if refuse_deletes and sql.lstrip().upper().startswith("DELETE"):
                raise sqlite3.OperationalError("attempt to write a readonly database")
            return self._real.execute(sql, *args)

        @property
        def row_factory(self):
            return self._real.row_factory

        @row_factory.setter
        def row_factory(self, value):
            self._real.row_factory = value

    connector = types.ModuleType("opti_oignon.db_utils")
    connector.safe_connect = lambda db_path, **kw: _Refusing(
        sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=5.0) if read_only
        else sqlite3.connect(str(target), timeout=5.0))
    mode = seam._ModeState()
    loaded, restore = isolate(
        targets={TFA: source("auth_2fa.py")},
        blocked=("ollama", "qrcode", "fido2", "fido2.server", "fido2.webauthn"),
        seeded={"opti_oignon.db_utils": connector, "opti_oignon.security_mode": seam._security_mode_seed(mode),
                "pyotp": seam._pyotp_seed(seam._Clock()), "opti_oignon.encryption": seam._encryption_seed()},
    )
    return loaded[TFA], mode, target, restore


def _raw(target, where, args=()):
    conn = sqlite3.connect(str(target))
    try:
        return {t: sorted(conn.execute(f"SELECT * FROM {t} WHERE {where}", args).fetchall()) for t in _TABLES}
    finally:
        conn.close()


def test_tf14_the_factors_stored_under_admin_are_retired_on_old_data(tmp_path):
    fixture = FIXTURE.read_text(encoding="utf-8")
    seeded = tmp_path / "two_factor.db"
    conn = sqlite3.connect(str(seeded))
    conn.executescript(fixture)
    conn.close()
    unbound = _raw(seeded, "user_id = ?", ("admin",))
    keeper = _raw(seeded, "user_id = ?", (KEEPER,))
    assert [len(unbound[t]) for t in _TABLES] == [1, 1, 10, 1], "the fixture holds what 176 wrote under admin"
    tfa, mode, target, restore = _file_store(tmp_path, "two_factor.db")
    try:
        decrypted = []
        tfa._decrypt_secret = lambda ciphertext: decrypted.append(ciphertext) or ""
        status = tfa.two_factor_manager.get_status(KEEPER)
        assert status.any_method_active is True
    finally:
        restore()
    assert _raw(target, "user_id = ?", ("admin",)) == {t: [] for t in _TABLES}, "no factor of no account remains"
    assert _raw(target, "user_id = ?", (KEEPER,)) == keeper, "a real account's rows are byte-identical"
    check = sqlite3.connect(str(target))
    try:
        records = check.execute("SELECT totp, webauthn, recovery_codes, app_passwords FROM retired_factors").fetchall()
    finally:
        check.close()
    assert records == [(1, 1, 10, 1)], records
    events = [(e, d) for e, d in mode.audit if e == "two_factor_unbound_retired"]
    assert events == [("two_factor_unbound_retired",
                       {"severity": "WARNING", "totp": 1, "webauthn": 1, "recovery_codes": 10, "app_passwords": 1})]
    assert decrypted == [], "the retirement never decrypts"
    # A second process finds nothing: no record, no audit.
    tfa, mode, target, restore = _file_store(tmp_path, "two_factor.db")
    try:
        tfa.two_factor_manager.get_status(KEEPER)
    finally:
        restore()
    check = sqlite3.connect(str(target))
    try:
        assert check.execute("SELECT COUNT(*) FROM retired_factors").fetchone()[0] == 1
    finally:
        check.close()
    assert not [e for e, _d in mode.audit if e == "two_factor_unbound_retired"]
    # A store that refuses the deletes keeps the rows, records nothing, and still serves.
    refusing = tmp_path / "refusing.db"
    conn = sqlite3.connect(str(refusing))
    conn.executescript(fixture)
    conn.close()
    tfa, mode, target, restore = _file_store(tmp_path, "refusing.db", refuse_deletes=True)
    try:
        assert tfa.two_factor_manager.get_status(KEEPER).any_method_active is True
    finally:
        restore()
    assert _raw(target, "user_id = ?", ("admin",)) == unbound
    assert not [e for e, _d in mode.audit if e == "two_factor_unbound_retired"]
    assert [d for e, d in mode.audit if e == "two_factor_unbound_not_retired"], "the refusal is audited"
    # A store opened read-only -- already in WAL mode, as the module leaves one: 176's schema read it without a
    # write, and the first connection still reads; the retirement is refused and audited, the status is read without
    # the record's table, and nothing raises.
    frozen = tmp_path / "frozen.db"
    conn = sqlite3.connect(str(frozen))
    conn.executescript(fixture)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    tfa, mode, target, restore = _file_store(tmp_path, "frozen.db", read_only=True)
    try:
        assert tfa.two_factor_manager.get_status(KEEPER).any_method_active is True
        assert tfa.two_factor_manager.retired_unbound() is None

        def unopenable():
            raise sqlite3.OperationalError("unable to open database file")

        opened, tfa._get_2fa_conn = tfa._get_2fa_conn, unopenable
        try:
            assert tfa.two_factor_manager.retired_unbound() is None, "it informs, it never refuses"
        finally:
            tfa._get_2fa_conn = opened
    finally:
        restore()
    assert _raw(target, "user_id = ?", ("admin",)) == unbound
    assert [d for e, d in mode.audit if e == "two_factor_unbound_not_retired"]


# --------------------------------------------------------------------------- TF15


def _seed_old_store(w):
    """The store as 176 left it, before the module's first connection: its retirement then runs on it."""
    raw = w.db.connect(str(w.tfa._2FA_DB_PATH))
    raw.executescript(FIXTURE.read_text(encoding="utf-8"))


def test_tf15_the_owner_is_told_of_retired_factors():
    w = _world()
    try:
        _seed_old_store(w)
        _as(w, A)
        told = w.client.get("/api/security/2fa/status").json()
        assert told["retired_unbound"] == {"at": told["retired_unbound"]["at"], "totp": 1, "webauthn": 1,
                                           "recovery_codes": 10, "app_passwords": 1}, told
        assert told["asked_at_login"] is True
        _as(w, B)
        assert "retired_unbound" not in w.client.get("/api/security/2fa/status").json(), "another account is not told"
        _enroll_totp(w, A, PW_A)
        assert "retired_unbound" not in w.client.get("/api/security/2fa/status").json(), "an active factor ends it"
    finally:
        w.close()
    w = _world(accounts=_Accounts((A, "alice", PW_A)))
    try:
        w.accounts.single_user_mode = True
        _as(w, A)
        assert w.client.get("/api/security/2fa/status").json()["asked_at_login"] is False
        w.mode.bulbe = True
        assert w.client.get("/api/security/2fa/status").json()["asked_at_login"] is True, "Bulbe asks it"
    finally:
        w.close()


# --------------------------------------------------------------------------- TF16, TF17


def _actor_routes(accounts):
    """routes_security with recording stand-ins for the mode, the kill switch, the emergency stop and the approvals."""
    acted = []
    mode = types.ModuleType("opti_oignon.security_mode")
    mode.MODE_BULBE, mode.MODE_DAILY, mode.VALID_MODES = "bulbe", "daily", ("bulbe", "daily")
    mode.is_bulbe = lambda: False
    mode._audit_log = lambda event, **details: None
    mode.security_mode_manager = SimpleNamespace(
        escalate_to_bulbe=lambda user_id, force=False: acted.append(("escalate", user_id)) or {"success": True})
    switch = types.ModuleType("opti_oignon.search_killswitch")
    switch.search_killswitch = SimpleNamespace(
        kill=lambda user_id, reason: acted.append(("kill", user_id)) or {"success": True},
        request_reenable=lambda user_id: acted.append(("request_reenable", user_id)) or {"success": True},
        cancel_reenable=lambda: acted.append(("cancel_reenable", A)) or {"success": True})
    stop = types.ModuleType("opti_oignon.emergency_stop")
    stop.stop = lambda user_id="", drop_to_bulbe=False: acted.append(("stop", user_id)) or {"stopped": True}
    stop.resume = lambda user_id="", warmup_model=None: acted.append(("resume", user_id)) or {"stopped": False}
    approvals = types.ModuleType("opti_oignon.tool_call_approval")
    approvals.tool_call_approval = SimpleNamespace(
        approve=lambda approval_id, user_id: acted.append(("approve", user_id)) or True,
        deny=lambda approval_id, user_id: acted.append(("deny", user_id)) or True)
    auth_routes = types.ModuleType("opti_oignon.api.routes_auth")
    auth_routes._get_current_user = lambda: {}
    auth_routes._extract_client_ip = lambda request: "198.51.100.7"
    loaded, restore = isolate(
        targets={FACTORS: source("ceremony_factors.py"), SEC: source("api", "routes_security.py")},
        seeded={"opti_oignon.auth": _auth_module(accounts, _Limiter(), _Verifier()),
                "opti_oignon.api.routes_auth": auth_routes, "opti_oignon.security_mode": mode,
                "opti_oignon.search_killswitch": switch, "opti_oignon.emergency_stop": stop,
                "opti_oignon.tool_call_approval": approvals},
        packages=("opti_oignon.api",),
    )
    return loaded[SEC], acted, restore


def _act(sec, current):
    run = asyncio.run
    run(sec.set_security_mode(sec.SecurityModeChangeRequest(mode="bulbe"), dict(current)))
    run(sec.kill_search(sec.SearchKillRequest(reason="manual"), dict(current)))
    run(sec.engage_emergency_stop(sec.EmergencyStopRequest(), dict(current)))
    run(sec.resume_from_emergency_stop(sec.EmergencyResumeRequest(), dict(current)))
    run(sec.approve_tool_call("approval-1", dict(current)))
    run(sec.deny_tool_call("approval-2", dict(current)))


def test_tf16_protective_actions_are_recorded_in_the_sessions_name():
    actions = ["escalate", "kill", "stop", "resume", "approve", "deny"]
    for people, current, expected in (
            (((A, "alice", PW_A), (B, "bob", PW_B)), {"sub": B}, B),
            ((((A, "alice", PW_A)),), {"sub": "local", "role": "admin"}, A),
            ((), {"sub": "local", "role": "admin"}, "local")):
        accounts = _Accounts(*people)
        sec, acted, restore = _actor_routes(accounts)
        try:
            _act(sec, current)
            assert acted == [(a, expected) for a in actions], (current, acted)
            # An account store that cannot be read stops none of them: the session's own name is recorded.
            accounts.broken = True
            acted.clear()
            _act(sec, current)
            assert acted == [(a, current["sub"]) for a in actions], (current, acted)
        finally:
            restore()


def test_tf17_asking_to_re_enable_search_is_the_owners_and_bound_to_the_asker(tmp_path):
    from fastapi import HTTPException

    accounts = _Accounts((A, "alice", PW_A), (B, "bob", PW_B))
    sec, acted, restore = _actor_routes(accounts)
    try:
        with pytest.raises(HTTPException) as caught:
            asyncio.run(sec.request_search_reenable(account=sec._session_account({"sub": B})))
        assert caught.value.status_code == 403 and acted == [], "nothing is pending"
        asyncio.run(sec.request_search_reenable(account=sec._session_account({"sub": A})))
        assert acted == [("request_reenable", A)]
        with pytest.raises(HTTPException) as caught:
            asyncio.run(sec.cancel_search_reenable(account=sec._session_account({"sub": B})))
        assert caught.value.status_code == 403 and acted == [("request_reenable", A)], "another cancels nothing"
        asyncio.run(sec.cancel_search_reenable(account=sec._session_account({"sub": A})))
        assert acted[-1] == ("cancel_reenable", A)
    finally:
        restore()
    # The real switch: the account that asked confirms; any other's confirmation is refused.
    mode = seam._ModeState()
    loaded, restore = isolate(targets={"opti_oignon.search_killswitch": source("search_killswitch.py")},
                              seeded={"opti_oignon.security_mode": seam._security_mode_seed(mode)})
    try:
        mod = loaded["opti_oignon.search_killswitch"]
        mod._STATE_PATH = tmp_path / "state.json"
        clock = [1_000_000.0]
        mod.time = SimpleNamespace(time=lambda: clock[0])
        switch = mod.search_killswitch
        assert switch.kill(reason="manual").get("persisted") is True
        ask = switch.request_reenable(A)
        code = switch.get_reenable_visual_code()
        clock[0] += 301
        other = switch.confirm_reenable(request_id=ask["request_id"], visual_code=code, user_id=B)
        assert other.get("error") == "user_mismatch" and switch.is_killed() is True, other
        own = switch.confirm_reenable(request_id=ask["request_id"], visual_code=code, user_id=A)
        assert own.get("success") is True and switch.is_killed() is False, own
    finally:
        restore()


# --------------------------------------------------------------------------- TF18


def test_tf18_the_real_account_store_and_bcrypt_carry_the_second_factor(tmp_path, monkeypatch):
    pytest.importorskip("bcrypt")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    connector = types.ModuleType("opti_oignon.db_utils")

    def safe_connect(db_path, *, check_same_thread=True, timeout=5.0):
        name = hashlib.md5(str(db_path).encode()).hexdigest()[:12] + ".db"
        return sqlite3.connect(str(tmp_path / name), timeout=timeout, check_same_thread=False)

    connector.safe_connect = safe_connect
    deps = _deps_module(None)
    clock, mode = seam._Clock(), seam._ModeState()
    loaded, restore = isolate(
        targets={"opti_oignon.auth": source("auth.py"), TFA: source("auth_2fa.py"),
                 FACTORS: source("ceremony_factors.py"), ROUTES: source("api", "routes_auth.py"),
                 SEC: source("api", "routes_security.py")},
        blocked=("ollama", "qrcode", "fido2", "fido2.server", "fido2.webauthn"),
        seeded={"opti_oignon.db_utils": connector, "opti_oignon.api.deps": deps,
                "opti_oignon.security_mode": seam._security_mode_seed(mode), "pyotp": seam._pyotp_seed(clock),
                "opti_oignon.encryption": seam._encryption_seed()},
        packages=("opti_oignon.api",),
    )
    client = None
    try:
        auth, tfa, routes, sec = loaded["opti_oignon.auth"], loaded[TFA], loaded[ROUTES], loaded[SEC]
        tfa.time = types.SimpleNamespace(time=clock.time)
        manager = auth.auth_manager
        deps.auth_manager = manager
        first = manager.create_user("first-account", "First-Passw0rd-of-tf18")
        later = manager.create_user("later-account", "Later-Passw0rd-of-tf18")
        assert first is not None and later is not None and manager.owners() == [first.user_id]
        app = FastAPI()
        app.include_router(routes.router)
        app.include_router(sec.router)
        session = {}
        app.dependency_overrides[routes._get_current_user] = lambda: dict(session)
        client = TestClient(app)
        session.update({"sub": later.user_id, "username": "later-account", "role": "user", "type": "access"})
        borrowed = client.post("/api/security/2fa/totp/setup", json={"password": "First-Passw0rd-of-tf18"})
        assert borrowed.status_code == 403, "a session passes the step-up with its own account's password alone"
        session.update({"sub": first.user_id, "username": "first-account"})
        setup = client.post("/api/security/2fa/totp/setup", json={"password": "First-Passw0rd-of-tf18"})
        assert setup.status_code == 200, setup.text
        code = seam._totp_code(setup.json()["secret"], int(clock.time() // 30))
        assert client.post("/api/security/2fa/totp/verify", json={"code": code}).status_code == 200
        asked = client.post("/api/auth/login", json={"username": "first-account",
                                                     "password": "First-Passw0rd-of-tf18"}).json()
        assert asked.get("requires_2fa") is True and asked["methods"] == ["totp"], asked
        straight = client.post("/api/auth/login", json={"username": "later-account",
                                                        "password": "Later-Passw0rd-of-tf18"}).json()
        assert straight.get("access_token") and "requires_2fa" not in straight, straight
    finally:
        if client is not None:
            client.close()
        restore()


# --------------------------------------------------------------------------- TF19


def test_tf19_deleting_an_account_deletes_its_factors():
    w = _world(accounts=_Accounts((A, "alice", PW_A, "admin"), (B, "bob", PW_B)))
    try:
        _enroll_totp(w, B, PW_B)
        _key_row(w, B)
        assert _post(w, "/api/security/2fa/app-passwords", {"name": "cli", "password": PW_B}).status_code == 200
        assert _post(w, "/api/security/2fa/recovery-codes/generate", {"password": PW_B}).status_code == 200
        _enroll_totp(w, A, PW_A)
        alice = _of(_rows(w), A)
        _as(w, A)
        gone = w.client.delete(f"/api/auth/users/{B}")
        assert gone.status_code == 200, gone.text
        rows = _rows(w)
        assert _of(rows, B) == {t: [] for t in _TABLES}, rows
        assert _of(rows, A) == alice
    finally:
        w.close()
    # A key login begun before the deletion no longer completes: its state is forgotten with the account.
    w = _world(accounts=_Accounts((A, "alice", PW_A, "admin"), (B, "bob", PW_B)), fido=True)
    try:
        _key_row(w, B)
        pending = _post(w, "/api/auth/login", {"username": "bob", "password": PW_B}).json()
        given = pending["webauthn_options"]["publicKey"]["challenge"]
        slot = w.routes._challenge_store.get(pending["challenge_id"])["webauthn_key"]
        manager = w.tfa.two_factor_manager
        assert slot in manager._webauthn_challenges
        _as(w, A)
        assert w.client.delete(f"/api/auth/users/{B}").status_code == 200
        assert slot not in manager._webauthn_challenges, "the state is gone from where the login recorded it"
        late = _post(w, "/api/auth/login/2fa", {"challenge_id": pending["challenge_id"], "method": "webauthn",
                                                "webauthn_response": {"signature": "signed:" + given}})
        assert late.status_code == 403 and "access_token" not in late.text, late.text
    finally:
        w.close()
    # An account deleted while its right answer is checked: no token, and the page is told to log in again.
    w = _world()
    try:
        _enroll_totp(w, B, PW_B)
        pending = _post(w, "/api/auth/login", {"username": "bob", "password": PW_B}).json()

        def passing_then_gone(user_id, code, method):
            w.accounts.delete_user(user_id)
            return {"success": True, "method": method}

        w.tfa.two_factor_manager.validate_2fa = passing_then_gone
        gone = _post(w, "/api/auth/login/2fa", {"challenge_id": pending["challenge_id"], "method": "auto",
                                                "code": "123456"})
        assert gone.status_code == 403 and "access_token" not in gone.text, gone.text
        assert "log in again" in gone.json()["detail"].lower(), gone.text
    finally:
        w.close()


# --------------------------------------------------------------------------- TF20

_SQL_SITES_177 = 52


def test_tf20_the_second_factor_statement_census():
    """In place of seam c5, whose walk it repeats word for word at the module's new count: the retirement of the
    factors no account held, its record and an account's purge add constant statements alone."""
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

    assert sites == _SQL_SITES_177, f"statement sites moved: {sites} != {_SQL_SITES_177}"
    assert dynamic == 0, f"dynamic statements appeared: {dynamic}"
    assert fstrings == 0, f"f-string statements appeared: {fstrings}"


# --------------------------------------------------------------------------- TF23


def _b64url(data):
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _signed_assertion(challenge, signer, credential_id):
    """What a security key answers to a login's challenge, signed with ``signer`` (an EC P-256 private key): the
    browser's client data for the page's origin, the authenticator data for ``localhost`` with the user present."""
    import json

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    client_data = json.dumps({"type": "webauthn.get", "challenge": challenge,
                              "origin": "http://localhost:5173", "crossOrigin": False}).encode()
    authenticator_data = hashlib.sha256(b"localhost").digest() + bytes([0x01]) + (1).to_bytes(4, "big")
    signature = signer.sign(authenticator_data + hashlib.sha256(client_data).digest(), ec.ECDSA(hashes.SHA256()))
    return {"id": _b64url(credential_id), "rawId": _b64url(credential_id), "type": "public-key",
            "response": {"clientDataJSON": _b64url(client_data), "authenticatorData": _b64url(authenticator_data),
                         "signature": _b64url(signature), "userHandle": None},
            "clientExtensionResults": {}}


def test_tf23_a_real_security_key_at_login():
    pytest.importorskip("fido2")
    from cryptography.hazmat.primitives.asymmetric import ec
    from fido2.cose import ES256
    from fido2.webauthn import Aaguid, AttestedCredentialData

    key = ec.generate_private_key(ec.SECP256R1())
    credential_id = b"tf23-credential-id"
    attested = AttestedCredentialData.create(Aaguid.NONE, credential_id, ES256.from_cryptography_key(key.public_key()))
    w = _world(real_fido=True)
    try:
        assert w.tfa.WEBAUTHN_AVAILABLE is True, "the installed library is loaded"
        conn = w.tfa._get_2fa_conn()
        try:
            conn.execute("INSERT INTO webauthn_credentials (credential_id, public_key, user_id, name, created_at) "
                         "VALUES (?, ?, ?, ?, ?)", (_b64url(credential_id), _b64url(bytes(attested)), A, "Key",
                                                    w.clock.time()))
            conn.commit()
        finally:
            conn.close()

        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A})
        assert first.status_code == 200, first.text
        options = first.json()["webauthn_options"]["publicKey"]
        assert first.json()["methods"] == ["webauthn"] and options["rpId"] == "localhost", first.text
        assert options["allowCredentials"][0]["id"] == _b64url(credential_id), options
        other = ec.generate_private_key(ec.SECP256R1())
        forged = _post(w, "/api/auth/login/2fa", {"challenge_id": first.json()["challenge_id"], "method": "webauthn",
                                                  "webauthn_response": _signed_assertion(options["challenge"], other,
                                                                                         credential_id)})
        assert forged.status_code == 403 and "access_token" not in forged.text, forged.text
        signed = _post(w, "/api/auth/login/2fa", {"challenge_id": first.json()["challenge_id"], "method": "webauthn",
                                                  "webauthn_response": _signed_assertion(options["challenge"], key,
                                                                                         credential_id)})
        assert signed.status_code == 200 and signed.json()["access_token"] == "tok-" + A, signed.text
        manager = w.tfa.two_factor_manager
        assert not [k for k in manager._webauthn_challenges if k.startswith("login:")], "spent by the answer that passes"
        # A key removed while a login waits no longer answers it.
        waiting = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        _as(w, A)
        removed = _delete(w, f"/api/security/2fa/webauthn/credentials/{_b64url(credential_id)}", {"password": PW_A})
        assert removed.status_code == 200, removed.text
        late = _post(w, "/api/auth/login/2fa", {
            "challenge_id": waiting["challenge_id"], "method": "webauthn",
            "webauthn_response": _signed_assertion(waiting["webauthn_options"]["publicKey"]["challenge"], key,
                                                   credential_id)})
        assert late.status_code == 403 and "access_token" not in late.text, late.text
    finally:
        w.close()


# --------------------------------------------------------------------------- TF25


def test_tf25_a_real_security_key_registered_then_asked():
    pytest.importorskip("fido2")
    import json

    from cryptography.hazmat.primitives.asymmetric import ec
    from fido2.cose import ES256
    from fido2.webauthn import Aaguid, AttestationObject, AttestedCredentialData, AuthenticatorData

    key = ec.generate_private_key(ec.SECP256R1())
    credential_id = b"tf25-registered-key"
    w = _world(real_fido=True)
    try:
        assert w.tfa.WEBAUTHN_AVAILABLE is True, "the installed library is loaded"
        _as(w, A)
        begun = _post(w, "/api/security/2fa/webauthn/register/begin", {"password": PW_A})
        assert begun.status_code == 200 and begun.json().get("success") is True, begun.text
        options = begun.json()["options"]["publicKey"]
        assert options["rp"]["id"] == "localhost" and options["user"]["name"] == "alice", options
        attested = AttestedCredentialData.create(Aaguid.NONE, credential_id,
                                                 ES256.from_cryptography_key(key.public_key()))
        authenticator_data = AuthenticatorData.create(hashlib.sha256(b"localhost").digest(),
                                                      AuthenticatorData.FLAG.UP | AuthenticatorData.FLAG.AT, 0,
                                                      attested)
        client_data = json.dumps({"type": "webauthn.create", "challenge": options["challenge"],
                                  "origin": "http://localhost:5173", "crossOrigin": False}).encode()
        attestation = AttestationObject.create("none", authenticator_data, {})
        complete = _post(w, "/api/security/2fa/webauthn/register/complete", {
            "credential_name": "Desk key",
            "response": {"id": _b64url(credential_id), "rawId": _b64url(credential_id), "type": "public-key",
                         "response": {"clientDataJSON": _b64url(client_data),
                                      "attestationObject": _b64url(bytes(attestation))},
                         "clientExtensionResults": {}}})
        assert complete.status_code == 200 and complete.json().get("success") is True, complete.text
        listed = w.client.get("/api/security/2fa/webauthn/credentials").json()["credentials"]
        assert [(c["credential_id"], c["name"]) for c in listed] == [(_b64url(credential_id), "Desk key")], listed
        first = _post(w, "/api/auth/login", {"username": "alice", "password": PW_A}).json()
        assert first["methods"] == ["webauthn"], first
        signed = _post(w, "/api/auth/login/2fa", {
            "challenge_id": first["challenge_id"], "method": "webauthn",
            "webauthn_response": _signed_assertion(first["webauthn_options"]["publicKey"]["challenge"], key,
                                                   credential_id)})
        assert signed.status_code == 200 and signed.json()["access_token"] == "tok-" + A, signed.text
    finally:
        w.close()
