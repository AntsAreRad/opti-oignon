#!/usr/bin/env python3
"""Contracts for the checks that decide allow or deny: none of them fails open.

An exception, a missing account, an import that fails, a write that fails: each used to end as an approval
somewhere. A plugin batch was approved with any password its check could not handle -- bcrypt 5 raises on one of more
than 72 bytes, and ``except Exception: password_valid = True`` took the raise for a match --, and the check looked for
an account named ``admin`` that need not exist. Leaving Bulbe mode passed the password when no account could be
resolved, or when checking it raised. The allowlist trusted a batch's entries in memory before it wrote them, so a
write that failed left them verified until a restart. A plugin loader that could not import the security mode, or in
Bulbe mode the allowlist, loaded the plugin without its check.

  * NF1 -- ``ceremony_factors.check_password`` passes the right password of the owner's account alone, asked through
    the owner's session, naming the account -- the owner is an admin's account or, while none is an admin's, the
    first registered; the synthetic session of single-user mode stands for the owner when exactly one account is an owner's.
    It refuses by name, with the status a route answers: no account behind the session (403 ``no_account``); a
    session that is not the owner's (403 ``not_owner``); a password for the same account being checked already --
    one check at a time per account, refused rather than queued (429 ``password_busy``); an account or an address
    the login limiter holds back, asked before the password is (429 ``password_locked``); a wrong password, recorded by the limiter as a failed
    try of the account the session stands for (403 ``wrong_password``, never 401); and anything that raised -- an
    account store that did not load or cannot be read, a password the library refuses, as bcrypt 5 refuses one of
    more than 72 bytes -- (503 ``password_unverifiable``). No message holds the password.
  * NF2 -- the plugin batch approval, the Bulbe-to-Daily confirmation and the search re-enable take that check
    before anything is approved, confirmed or re-enabled, the check in a worker thread and the change back on the
    event loop, where the revocations and the cancellations run: a refusal raises its status and is audited by
    name, with no secret, and nothing is changed; a right password approves and re-enables in the name of the
    account it names, and confirms for the session that asked. An allowlist that cannot be written answers 503.
  * NF3 -- the allowlist writes a batch's entries before it trusts them: a write that fails answers ``not_recorded``,
    the entries in memory as they were and the batch still pending; a file the failed write cut short reads as no
    allowlist -- approvals lost, never one gained; once a write succeeds, the batch is approved and a fresh manager
    verifies it.
  * NF4 -- the plugin loader starts nothing when the security mode cannot be read, nor in Bulbe mode when the
    allowlist cannot be imported or its check raises, and says why; in Daily mode it loads without importing the
    allowlist.
  * NF5 -- the global authentication middleware checks every HTTP request -- one that merely says ``Upgrade:
    websocket`` included --, serves nothing but the public paths when the account store did not load (503), and
    never takes a security mode it cannot read for Daily: the single-user bypass holds in Daily mode alone.
  * NF6 -- the role checks and the CSRF middleware take a security mode they cannot read for Bulbe mode.
  * NF7 -- the security mode compares a lock file's HMAC, a downgrade's request id and its visual code on their
    bytes: a text beyond ASCII is a mismatch, never an error.
  * NF8 -- the app registers its security-mode, CSRF and authentication middlewares outside any ``try``: one that
    cannot be registered stops the app, which never serves a route without it.
  * NF9 -- the CSRF middleware checks every state-changing HTTP request, one that merely says ``Upgrade:
    websocket`` included.
  * NF10 -- on the real account store, its real accounts and the real bcrypt, the ceremony's check passes the owner's
    right password and nothing else -- the first registered account while registration gave every account the
    default role, by the store's order and never by a clock stepped back, the admins once one exists; none on an
    empty store --: a later account's own password, a wrong one, and one bcrypt refuses are each refused by name.
  * NF11 -- every WebSocket route of the API checks its handshake -- ``authenticate_websocket`` -- and closes the
    socket when it fails: no middleware sees a socket.
  * NF12 -- the page's reconnecting socket, run under Node with a socket stood in for, takes a handshake the server
    refused (close code 4001) as its end -- one final close, no reconnect, even after an open --, and still
    reconnects a connection lost otherwise.
  * NF13 -- the three ceremonies' panels and the authentication switch show the server's own reason for a refusal
    (``httpRefusal``) -- another account's session, a wrong password, a wait -- rather than the bare status, and
    the switch shows the mode the server answers is in force.

Every module under contract is loaded from its file through the shared isolation window, with the account store,
the limiter, the security mode, the allowlist manager and the plugin worker stood in for where a contract says so;
NF3 runs the real allowlist, its key file and its paths under ``tmp_path``. Nothing reads or writes the platform's
data.

Local-only (the public distribution ships no tests).
"""

import ast
import asyncio
import inspect
import sqlite3
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

FACTORS = "opti_oignon.ceremony_factors"
ROUTES = "opti_oignon.api.routes_security"
ALLOW = "opti_oignon.plugin_allowlist"
LOADER = "opti_oignon.plugin_loader"
RIGHT = "the account's own ceremony password"
WRONG = "not the password at all"
LONG = "y" * 73  # bcrypt 5 raises on more than 72 bytes
CLIENT = "198.51.100.7"
K1 = bytes(range(1, 33))

# Seconds per contract, read by the ladder's budget reader: NF12 and NF13 use the frontend helper, which makes this
# suite one of the frontend family, where every contract carries a budget.
BUDGET_S = {
    "test_nf1_the_ceremony_password_is_the_owners_or_refused_by_name": 2.0,
    "test_nf2_the_ceremonies_change_nothing_on_a_password_that_is_not_the_owners": 3.0,
    "test_nf3_the_allowlist_trusts_a_batch_only_once_it_is_written": 2.0,
    "test_nf4_the_loader_starts_nothing_it_could_not_check": 2.0,
    "test_nf5_the_global_middleware_checks_every_request_and_fails_closed": 1.0,
    "test_nf6_role_checks_and_csrf_take_an_unreadable_mode_for_bulbe": 1.0,
    "test_nf7_the_mode_compares_its_codes_on_bytes_and_never_raises": 1.0,
    "test_nf8_the_app_never_starts_without_its_security_middlewares": 1.0,
    "test_nf9_csrf_checks_a_request_that_only_says_it_upgrades": 1.0,
    "test_nf10_the_real_account_store_passes_the_owners_password_alone": 6.0,
    "test_nf11_every_websocket_route_checks_its_handshake": 2.0,
    "test_nf12_the_pages_socket_stops_once_its_handshake_is_refused": 5.0,
    "test_nf13_the_ceremonies_panels_show_the_servers_reason": 1.0,
}


# --------------------------------------------------------------------------- stand-ins


class _Account:
    """An account as ``auth.User`` holds one -- id, name, role, creation time, hash, and nothing else NF10 does not also
    find on the real one."""

    made = 0

    def __init__(self, user_id, username, password, *, role="admin"):
        self.user_id, self.username, self.role = user_id, username, role
        self.password_hash = "hash:" + password
        _Account.made += 1
        self.created_at = float(_Account.made)


class _Accounts:
    """The account store stood in for; ``broken`` makes every read raise as an unreadable store does."""

    def __init__(self, *accounts, broken=False):
        self.accounts, self.broken = list(accounts), broken

    def _read(self):
        if self.broken:
            raise sqlite3.OperationalError("unable to open database file")

    def get_user(self, user_id):
        self._read()
        return next((account for account in self.accounts if account.user_id == user_id), None)

    def count_users(self):
        self._read()
        return len(self.accounts)

    def list_users(self, limit=100, offset=0):
        self._read()
        return self.accounts[offset:offset + limit]

    def owners(self):
        """The rule ``AuthManager.owners`` holds: the admins' accounts, or the first registered alone while none is an
        admin's (the stand-in's accounts are registered in the order they were made)."""
        self._read()
        admins = sorted(account.user_id for account in self.accounts if account.role == "admin")
        if admins or not self.accounts:
            return admins
        return [min(self.accounts, key=lambda account: (account.created_at, account.user_id)).user_id]

    def is_owner(self, user_id):
        return bool(user_id) and user_id in self.owners()


class _Limiter:
    def __init__(self, allowed=True):
        self.allowed, self.asked, self.failures = allowed, [], []

    def check_rate_limit(self, ip, username):
        self.asked.append((ip, username))
        return (True, 0) if self.allowed else (False, 600)

    def record_failure(self, ip, username):
        self.failures.append((ip, username))


class _Verifier:
    """``verify_password`` stood in for, as bcrypt 5 answers: a password of more than 72 bytes raises."""

    def __init__(self):
        self.calls, self.threads = 0, []
        self.entered, self.release = None, None

    def __call__(self, password, password_hash):
        self.calls += 1
        self.threads.append(threading.get_ident())
        if self.entered is not None:
            self.entered.set()
            self.release.wait(10)
        if len(password.encode("utf-8")) > 72:
            raise ValueError("password cannot be longer than 72 bytes, truncate manually if necessary")
        return password_hash == "hash:" + password


def _auth(accounts, limiter, verifier):
    module = types.ModuleType("opti_oignon.auth")
    module.auth_manager, module.login_rate_limiter, module.verify_password = accounts, limiter, verifier
    return module


RESIDENT = _Account("uid-resident", "resident", RIGHT)
SESSION = {"sub": "uid-resident", "username": "resident", "type": "access"}


def _factors(accounts=None, limiter=None, *, auth_blocked=False):
    """``ceremony_factors`` in a window, the account store and the limiter stood in for (or ``auth`` unreachable)."""
    accounts = accounts if accounts is not None else _Accounts(RESIDENT)
    limiter = limiter if limiter is not None else _Limiter()
    verifier = _Verifier()
    seeded = {} if auth_blocked else {"opti_oignon.auth": _auth(accounts, limiter, verifier)}
    blocked = ("opti_oignon.auth",) if auth_blocked else ()
    loaded, restore = isolate(targets={FACTORS: source("ceremony_factors.py")}, blocked=blocked, seeded=seeded)
    return SimpleNamespace(cf=loaded[FACTORS], accounts=accounts, limiter=limiter, verifier=verifier,
                           restore=restore)


def _secret_free(verdict):
    return not any(secret in verdict.message for secret in (RIGHT, WRONG, LONG))


# --------------------------------------------------------------------------- NF1


def test_nf1_the_ceremony_password_is_the_owners_or_refused_by_name():
    w = _factors()
    try:
        verdict = w.cf.check_password(SESSION, RIGHT, client_ip=CLIENT)
        assert verdict.passed and verdict.account_id == "uid-resident" and verdict.status == 200
        assert w.limiter.asked == [(CLIENT, "resident")] and w.limiter.failures == []
        # A wrong one: refused with a 403, never a 401 that would end the session's page, and counted.
        verdict = w.cf.check_password(SESSION, WRONG, client_ip=CLIENT)
        assert (verdict.passed, verdict.status, verdict.error) == (False, 403, "wrong_password")
        assert w.limiter.failures == [(CLIENT, "resident")] and _secret_free(verdict)
        # A password the library refuses -- bcrypt 5 on more than 72 bytes -- is not checked, so not taken.
        verdict = w.cf.check_password(SESSION, LONG, client_ip=CLIENT)
        assert (verdict.passed, verdict.status, verdict.error) == (False, 503, "password_unverifiable")
        assert _secret_free(verdict) and "ValueError" in verdict.message
        # No account behind the session: none named, or one that is not there.
        calls = w.verifier.calls
        for session in (None, {}, {"sub": ""}, {"sub": "uid-nobody"}, {"username": "resident"}, "uid-resident"):
            verdict = w.cf.check_password(session, RIGHT, client_ip=CLIENT)
            assert (verdict.passed, verdict.status, verdict.error) == (False, 403, "no_account"), session
        assert w.verifier.calls == calls, "no account, no password checked"
    finally:
        w.restore()
    # A session that is not the owner's: an account registered later approves nothing with its own right password.
    user = _Account("uid-user", "registered", RIGHT, role="user")
    w = _factors(_Accounts(RESIDENT, user))
    try:
        verdict = w.cf.check_password({"sub": "uid-user"}, RIGHT, client_ip=CLIENT)
        assert (verdict.passed, verdict.status, verdict.error) == (False, 403, "not_owner")
        assert w.verifier.calls == 0 and w.limiter.asked == []
    finally:
        w.restore()
    # While no account is an admin's, the first registered is the owner's -- registration gives the default role.
    first = _Account("uid-first", "first", RIGHT, role="user")
    later = _Account("uid-later", "later", RIGHT, role="user")
    w = _factors(_Accounts(later, first))
    try:
        assert w.cf.check_password({"sub": "uid-first"}, RIGHT, client_ip=CLIENT).passed
        assert w.cf.check_password({"sub": "uid-later"}, RIGHT, client_ip=CLIENT).error == "not_owner"
    finally:
        w.restore()
    # The synthetic session of single-user mode stands for the owner when exactly one account is an owner's: the
    # one admin among users, or the first registered of accounts that are all users'; for none when two are admins',
    # or none.
    local = {"sub": "local", "username": "local"}
    oldest = _Account("uid-oldest", "oldest", RIGHT, role="user")
    newer = _Account("uid-newer", "newer", RIGHT, role="user")
    for accounts, owner in ((_Accounts(RESIDENT), "resident"), (_Accounts(newer, RESIDENT), "resident"),
                            (_Accounts(newer, oldest), "oldest"), (_Accounts(), None),
                            (_Accounts(RESIDENT, _Account("uid-other", "other", RIGHT)), None)):
        w = _factors(accounts)
        try:
            verdict = w.cf.check_password(local, RIGHT, client_ip=CLIENT)
            assert verdict.passed is (owner is not None), owner
            assert owner is not None or verdict.error == "no_account"
            # The limiter counts the account the session stands for, never the session's own name.
            assert w.limiter.asked == ([(CLIENT, owner)] if owner else []), w.limiter.asked
            if owner:
                assert w.cf.check_password(local, WRONG, client_ip=CLIENT).error == "wrong_password"
                assert w.limiter.failures == [(CLIENT, owner)], w.limiter.failures
        finally:
            w.restore()
    # One check at a time per account: a second try sent while the first is checked is refused at once (429), its
    # password never checked, and the account is free again once the first ends; another owner is not held back.
    other_admin = _Account("uid-other-admin", "other-admin", RIGHT)
    w = _factors(_Accounts(RESIDENT, other_admin))
    try:
        w.verifier.entered, w.verifier.release = threading.Event(), threading.Event()
        first = []
        worker = threading.Thread(target=lambda: first.append(w.cf.check_password(SESSION, WRONG, client_ip=CLIENT)))
        worker.start()
        assert w.verifier.entered.wait(10)
        busy = w.cf.check_password(SESSION, RIGHT, client_ip=CLIENT)
        assert (busy.passed, busy.status, busy.error) == (False, 429, "password_busy") and _secret_free(busy)
        assert w.verifier.calls == 1 and w.limiter.asked == [(CLIENT, "resident")]
        w.verifier.entered = None
        assert w.cf.check_password({"sub": "uid-other-admin"}, RIGHT, client_ip=CLIENT).passed
        w.verifier.release.set()
        worker.join(10)
        assert first and first[0].error == "wrong_password"
        assert w.cf.check_password(SESSION, RIGHT, client_ip=CLIENT).passed, "the account is free once a check ends"
        # A check that raised -- bcrypt 5 on more than 72 bytes -- frees the account too.
        assert w.cf.check_password(SESSION, LONG, client_ip=CLIENT).error == "password_unverifiable"
        assert w.cf.check_password(SESSION, RIGHT, client_ip=CLIENT).passed, "a check that raised frees the account"
    finally:
        w.restore()
    # An account the limiter holds back is refused before its password is checked.
    w = _factors(limiter=_Limiter(allowed=False))
    try:
        verdict = w.cf.check_password(SESSION, RIGHT, client_ip=CLIENT)
        assert (verdict.passed, verdict.status, verdict.error) == (False, 429, "password_locked")
        assert w.verifier.calls == 0
    finally:
        w.restore()
    # An account store that cannot be read, or did not load, or an auth module that does not import: 503.
    for name in ("unreadable", "not loaded", "not importable"):
        w = (_factors(_Accounts(RESIDENT, broken=True)) if name == "unreadable"
             else _factors(auth_blocked=True) if name == "not importable" else _factors())
        try:
            if name == "not loaded":
                sys.modules["opti_oignon.auth"].auth_manager = None
            verdict = w.cf.check_password(SESSION, RIGHT, client_ip=CLIENT)
            assert (verdict.passed, verdict.status, verdict.error) == (False, 503, "password_unverifiable"), name
        finally:
            w.restore()


# --------------------------------------------------------------------------- NF2


def _routes(accounts=None, limiter=None, *, approve_answer=None):
    """``routes_security`` in a window, the check real, the account store, the limiter, the security mode, the
    allowlist manager and the search kill switch stood in for; each change records the thread it ran on."""
    accounts = accounts if accounts is not None else _Accounts(RESIDENT, _Account("uid-user", "registered", RIGHT,
                                                                                    role="user"))
    limiter = limiter if limiter is not None else _Limiter()
    verifier = _Verifier()
    audit, approved, confirmed, reenabled, changes = [], [], [], [], []

    auth_routes = types.ModuleType("opti_oignon.api.routes_auth")
    auth_routes._get_current_user = lambda: dict(SESSION)
    auth_routes._extract_client_ip = lambda request: CLIENT

    mode = types.ModuleType("opti_oignon.security_mode")
    mode.MODE_BULBE, mode.MODE_DAILY, mode.VALID_MODES = "bulbe", "daily", ("bulbe", "daily")

    def confirm_downgrade(**kwargs):
        changes.append(threading.get_ident())
        confirmed.append(kwargs)
        return {"success": True, "mode": "daily"}

    mode.security_mode_manager = SimpleNamespace(confirm_downgrade=confirm_downgrade)
    mode._audit_log = lambda event, severity="INFO", **details: audit.append((event, severity, details))

    allow = types.ModuleType(ALLOW)

    def approve_batch(batch_id, user_id):
        changes.append(threading.get_ident())
        approved.append((batch_id, user_id))
        return approve_answer or {"success": True, "entries_added": 1, "batch_id": batch_id}

    allow.plugin_allowlist_manager = SimpleNamespace(approve_batch=approve_batch)

    switch = types.ModuleType("opti_oignon.search_killswitch")

    def confirm_reenable(*, request_id, visual_code, user_id):
        changes.append(threading.get_ident())
        reenabled.append((request_id, user_id))
        return {"success": True, "search_enabled": True}

    switch.search_killswitch = SimpleNamespace(confirm_reenable=confirm_reenable)

    loaded, restore = isolate(
        targets={FACTORS: source("ceremony_factors.py"), ROUTES: source("api", "routes_security.py")},
        seeded={"opti_oignon.auth": _auth(accounts, limiter, verifier), "opti_oignon.api.routes_auth": auth_routes,
                "opti_oignon.security_mode": mode, ALLOW: allow, "opti_oignon.search_killswitch": switch},
        packages=("opti_oignon.api",),
    )
    return SimpleNamespace(routes=loaded[ROUTES], audit=audit, approved=approved, confirmed=confirmed,
                           reenabled=reenabled, changes=changes, verifier=verifier, limiter=limiter,
                           restore=restore)


def _on_a_loop(w, coroutine):
    """Run a route's coroutine on an event loop of its own, and say on which thread the loop ran."""
    w.loop_thread = threading.get_ident()
    return asyncio.run(coroutine)


def _session(session):
    return dict(session) if isinstance(session, dict) else session


def _approve(w, password, session=SESSION):
    body = w.routes.BatchApproveRequest(batch_id="batch-1", visual_code="000000", password=password)
    return _on_a_loop(w, w.routes.approve_plugin_batch(body, object(), _session(session)))


def _confirm(w, password, session=SESSION):
    body = w.routes.DowngradeConfirmRequest(request_id="req-1", visual_code="123456", password=password)
    return _on_a_loop(w, w.routes.confirm_mode_downgrade(body, object(), _session(session)))


def _reenable(w, password, session=SESSION):
    body = w.routes.SearchReenableConfirm(request_id="req-2", visual_code="654321", password=password)
    return _on_a_loop(w, w.routes.confirm_search_reenable(body, object(), _session(session)))


def test_nf2_the_ceremonies_change_nothing_on_a_password_that_is_not_the_owners():
    from fastapi import HTTPException

    w = _routes()
    try:
        routes = w.routes
        for handler in ("approve_plugin_batch", "confirm_mode_downgrade", "confirm_search_reenable", "revoke_plugin",
                        "cancel_mode_downgrade"):
            assert inspect.iscoroutinefunction(getattr(routes, handler)), handler
        for ceremony, run, done in (("plugin_batch", _approve, w.approved), ("downgrade", _confirm, w.confirmed),
                                    ("search_reenable", _reenable, w.reenabled)):
            for password, session, status, error in ((WRONG, SESSION, 403, "wrong_password"),
                                                     (LONG, SESSION, 503, "password_unverifiable"),
                                                     (RIGHT, {"sub": "uid-nobody"}, 403, "no_account"),
                                                     (RIGHT, {"sub": "uid-user"}, 403, "not_owner")):
                w.audit.clear()
                with pytest.raises(HTTPException) as caught:
                    run(w, password, session)
                assert caught.value.status_code == status, (ceremony, error)
                assert done == [], (ceremony, error)
                assert [(event, details["ceremony"], details["error"], details["user_id"])
                        for event, _sev, details in w.audit] == [
                    ("ceremony_password_refused", ceremony, error, session["sub"])], (ceremony, error)
                said = repr(w.audit) + str(caught.value.detail)
                assert not any(secret in said for secret in (RIGHT, WRONG, LONG)), (ceremony, error)
        # The limiter saw the client's address, as the login's helper reads it.
        assert set(w.limiter.asked) == {(CLIENT, "resident")} and set(w.limiter.failures) == {(CLIENT, "resident")}
        # The right password: the batch approved and search re-enabled in the name of the account it names, the
        # downgrade confirmed for the session that asked.
        assert _approve(w, RIGHT)["success"] is True
        assert w.approved == [("batch-1", "uid-resident")]
        assert _confirm(w, RIGHT)["success"] is True
        assert [(c["user_id"], c["request_id"]) for c in w.confirmed] == [("uid-resident", "req-1")]
        assert _reenable(w, RIGHT)["success"] is True
        assert w.reenabled == [("req-2", "uid-resident")]
        # bcrypt in a worker thread, never on the event loop; each change back on the loop, where the revocations
        # and the cancellations run, so that none of them interleaves with it.
        assert w.verifier.threads and w.loop_thread not in w.verifier.threads, (w.loop_thread, w.verifier.threads)
        assert w.changes == [w.loop_thread] * 3, (w.loop_thread, w.changes)
    finally:
        w.restore()
    # The synthetic single-user session stands for the owner: the batch is approved in the owner's name.
    w = _routes(accounts=_Accounts(_Account("uid-later", "later", RIGHT, role="user"), RESIDENT))
    try:
        assert _approve(w, RIGHT, {"sub": "local", "username": "local"})["success"] is True
        assert w.approved == [("batch-1", "uid-resident")]
    finally:
        w.restore()
    # The limiter holds the account back: 429, nothing approved.
    w = _routes(limiter=_Limiter(allowed=False))
    try:
        with pytest.raises(HTTPException) as caught:
            _approve(w, RIGHT)
        assert caught.value.status_code == 429 and w.approved == []
    finally:
        w.restore()
    # An allowlist that cannot be written answers 503, not a plain 400.
    w = _routes(approve_answer={"success": False, "error": "not_recorded",
                                "message": "The allowlist could not be written (OSError): nothing was approved."})
    try:
        with pytest.raises(HTTPException) as caught:
            _approve(w, RIGHT)
        assert caught.value.status_code == 503 and "nothing was approved" in caught.value.detail
    finally:
        w.restore()


# --------------------------------------------------------------------------- NF3


def _allow_world(tmp_path, monkeypatch):
    """encryption, the security mode and the allowlist in a window, every path they reach under ``tmp_path``."""
    pytest.importorskip("cryptography")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    names = ("process_secrets", "encryption", "security_mode", "plugin_allowlist")
    loaded, restore = isolate(targets={f"opti_oignon.{name}": source(f"{name}.py") for name in names},
                              blocked=("opti_oignon.secure_bytes",))
    try:
        enc, sm, pa = loaded["opti_oignon.encryption"], loaded["opti_oignon.security_mode"], loaded[ALLOW]
        data = tmp_path / "data"
        data.mkdir(parents=True)
        keyfile = data / ".keyfile"
        enc._DEFAULT_KEYFILE = sm._DEFAULT_KEYFILE = keyfile
        sm._DATA_DIR, sm._LOCKFILE_PATH, sm._SECURITY_YAML = (data, data / ".security_mode_lock",
                                                              tmp_path / "security.yaml")
        sm._audit_log = lambda event, severity="INFO", **details: None
        pa._DATA_DIR, pa._ALLOWLIST_PATH = data, data / "plugin_allowlist.json"
        enc.save_keyfile(K1, allow_unprotected=True)
        return SimpleNamespace(pa=pa, allowlist=pa._ALLOWLIST_PATH, restore=restore)
    except BaseException:
        restore()
        raise


def test_nf3_the_allowlist_trusts_a_batch_only_once_it_is_written(tmp_path, monkeypatch):
    w = _allow_world(tmp_path, monkeypatch)
    try:
        pa = w.pa
        plugin = tmp_path / "first"
        plugin.mkdir()
        (plugin / "plugin.py").write_text("VALUE = 1\n", encoding="utf-8")
        (plugin / "manifest.yaml").write_text("id: first\n", encoding="utf-8")
        manager = pa.PluginAllowlistManager()
        manifest = manager.prepare_batch([{"plugin_id": "first", "plugin_dir": str(plugin), "permissions": []}])
        saved = pa._save_allowlist

        def full_disk(entries):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(pa, "_save_allowlist", full_disk)
        result = manager.approve_batch(manifest.batch_id, "tester")
        assert (result["success"], result["error"]) == (False, "not_recorded"), result
        assert "nothing was approved" in result["message"]
        # Nothing trusted, here or on disk; the batch still pending.
        assert manager.verify_plugin("first", plugin, [])["allowed"] is False
        assert not w.allowlist.exists() and manager.get_pending_batch() is manifest
        # A write that the disk cuts short leaves a file no manager trusts anything in.
        real_dump = pa.json.dump

        def cut_short(value, handle, **kwargs):
            handle.write('[{"plugin_id": "first", "code_')
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(pa, "_save_allowlist", saved)
        monkeypatch.setattr(pa.json, "dump", cut_short)
        result = manager.approve_batch(manifest.batch_id, "tester")
        assert (result["success"], result["error"]) == (False, "not_recorded"), result
        monkeypatch.setattr(pa.json, "dump", real_dump)
        assert manager.verify_plugin("first", plugin, [])["allowed"] is False
        assert pa.PluginAllowlistManager().verify_plugin("first", plugin, [])["allowed"] is False
        # Once a write succeeds: approved, and a fresh manager verifies it from the file.
        monkeypatch.setattr(pa, "_save_allowlist", saved)
        assert manager.approve_batch(manifest.batch_id, "tester")["success"] is True
        assert pa.PluginAllowlistManager().verify_plugin("first", plugin, [])["allowed"] is True
    finally:
        w.restore()


# --------------------------------------------------------------------------- NF4


class _Workers:
    """The plugin worker manager stood in for: what was started."""

    def __init__(self):
        self.started = []

    def start_plugin(self, **kwargs):
        self.started.append(kwargs["plugin_name"])


def _loader(tmp_path, *, mode, allowlist):
    """The plugin loader in a window: the security mode stood in for (``mode``: "daily", "bulbe" or "unreadable"),
    the allowlist stood in for ("allows", "raises") or unreachable ("absent")."""
    seeded, blocked = {}, ["opti_oignon.plugin_subprocess"]
    state = types.ModuleType("opti_oignon.security_mode")

    def is_bulbe():
        if mode == "unreadable":
            raise RuntimeError("the lock file cannot be read")
        return mode == "bulbe"

    state.is_bulbe = is_bulbe
    seeded["opti_oignon.security_mode"] = state
    asked = []
    if allowlist == "absent":
        blocked.append(ALLOW)
    else:
        allow = types.ModuleType(ALLOW)

        def verify_plugin(plugin_id, plugin_dir, permissions):
            asked.append(plugin_id)
            if allowlist == "raises":
                raise OSError("the allowlist cannot be read")
            if allowlist == "refuses":
                return {"allowed": False, "reason": "Not in allowlist"}
            return {"allowed": True, "reason": "Verified"}

        allow.plugin_allowlist_manager = SimpleNamespace(verify_plugin=verify_plugin)
        seeded[ALLOW] = allow
    loaded, restore = isolate(
        targets={"opti_oignon.plugin_manifest": source("plugin_manifest.py"), LOADER: source("plugin_loader.py")},
        blocked=tuple(blocked), seeded=seeded,
    )
    plugin = tmp_path / f"plugin-{mode}-{allowlist}"
    plugin.mkdir()
    (plugin / "manifest.yaml").write_text(
        "name: checked-plugin\nversion: 1.0.0\nauthor: tester\ndescription: a fixture\n"
        "entry_point: entry_point.py\nhooks:\n  - pre_inference\n", encoding="utf-8")
    (plugin / "entry_point.py").write_text("def hook_pre_inference(context):\n    return None\n", encoding="utf-8")
    workers = _Workers()
    loader = loaded[LOADER].PluginLoader(subprocess_manager=workers)
    return SimpleNamespace(loader=loader, errors=loaded[LOADER].PluginLoadError, plugin=plugin, workers=workers,
                           asked=asked, restore=restore)


def test_nf4_the_loader_starts_nothing_it_could_not_check(tmp_path):
    for mode, allowlist, said in (("unreadable", "allows", "The security mode cannot be read"),
                                  ("bulbe", "absent", "not checked"),
                                  ("bulbe", "raises", "not checked"),
                                  ("bulbe", "refuses", "not allowed")):
        w = _loader(tmp_path, mode=mode, allowlist=allowlist)
        try:
            with pytest.raises(w.errors) as caught:
                w.loader.load_plugin(w.plugin)
            assert said in str(caught.value), (mode, allowlist, str(caught.value))
            assert w.workers.started == [], (mode, allowlist)
        finally:
            w.restore()
    # Bulbe with a check that allows: started. Daily: started, the allowlist never imported.
    for mode, allowlist, asked in (("bulbe", "allows", ["checked-plugin"]), ("daily", "absent", [])):
        w = _loader(tmp_path, mode=mode, allowlist=allowlist)
        try:
            w.loader.load_plugin(w.plugin)
            assert w.workers.started == ["checked-plugin"] and w.asked == asked, (mode, allowlist)
        finally:
            w.restore()


# --------------------------------------------------------------------------- NF5


MIDDLEWARE = "opti_oignon.api.auth_middleware"


def _mode_module(mode):
    state = types.ModuleType("opti_oignon.security_mode")

    def is_bulbe():
        if mode == "unreadable":
            raise RuntimeError("the lock file cannot be read")
        return mode == "bulbe"

    state.is_bulbe = is_bulbe
    return state


def _middleware(*, manager, mode):
    """The authentication middleware in a window: the account store stood in for ("accounts": a token "good" is
    valid; "single": single-user mode; "none": it did not load) and the security mode ("daily", "bulbe",
    "unreadable")."""
    auth = types.ModuleType("opti_oignon.auth")
    auth.auth_manager = None if manager == "none" else SimpleNamespace(
        single_user_mode=(manager == "single"),
        validate_token=lambda token: {"sub": "uid-resident"} if token == "good" else None)
    loaded, restore = isolate(targets={MIDDLEWARE: source("api", "auth_middleware.py")},
                              seeded={"opti_oignon.auth": auth, "opti_oignon.security_mode": _mode_module(mode)},
                              packages=("opti_oignon.api",))
    return loaded[MIDDLEWARE], restore


def _ask(module, path, headers=()):
    """One HTTP request through the middleware: its status, and whether the route behind it was served."""
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse

    scope = {"type": "http", "method": "GET", "path": path, "raw_path": path.encode("ascii"), "query_string": b"",
             "headers": [(name.lower().encode("ascii"), value.encode("ascii")) for name, value in headers],
             "scheme": "http", "server": ("127.0.0.1", 8001), "client": ("127.0.0.1", 50000), "root_path": ""}
    served = []

    async def call_next(request):
        served.append(request.url.path)
        return PlainTextResponse("served")

    response = asyncio.run(module.AuthMiddleware(app=None).dispatch(Request(scope), call_next))
    return response.status_code, bool(served)


def test_nf5_the_global_middleware_checks_every_request_and_fails_closed():
    module, restore = _middleware(manager="accounts", mode="daily")
    try:
        # A request that only says it upgrades to a WebSocket is an HTTP request, checked as one.
        for connection in ("keep-alive", "upgrade", "Upgrade"):
            assert _ask(module, "/api/chat/send", [("Upgrade", "websocket"), ("Connection", connection)]) == (
                401, False), connection
        assert _ask(module, "/api/chat/send") == (401, False)
        assert _ask(module, "/api/chat/send", [("Cookie", "oo_access_token=good")]) == (200, True)
        assert _ask(module, "/api/health") == (200, True)
    finally:
        restore()
    # No account store: nothing but the public paths.
    module, restore = _middleware(manager="none", mode="daily")
    try:
        assert _ask(module, "/api/chat/send", [("Cookie", "oo_access_token=good")]) == (503, False)
        assert _ask(module, "/api/health") == (200, True)
    finally:
        restore()
    # The single-user bypass holds in Daily mode alone; a mode that cannot be read is not Daily.
    for mode, answer in (("daily", (200, True)), ("bulbe", (401, False)), ("unreadable", (401, False))):
        module, restore = _middleware(manager="single", mode=mode)
        try:
            assert _ask(module, "/api/chat/send") == answer, mode
        finally:
            restore()


# --------------------------------------------------------------------------- NF6


def test_nf6_role_checks_and_csrf_take_an_unreadable_mode_for_bulbe():
    from fastapi import HTTPException

    single = types.ModuleType("opti_oignon.auth")
    single.auth_manager = SimpleNamespace(single_user_mode=True, validate_token=lambda token: None)
    for name, path in (("opti_oignon.rbac_enforcement", source("rbac_enforcement.py")),
                       ("opti_oignon.api.csrf_middleware", source("api", "csrf_middleware.py"))):
        for mode, bulbe in (("unreadable", True), ("bulbe", True), ("daily", False)):
            loaded, restore = isolate(targets={name: path},
                                      seeded={"opti_oignon.security_mode": _mode_module(mode), "opti_oignon.auth": single},
                                      packages=("opti_oignon.api",))
            try:
                module = loaded[name]
                assert module._is_bulbe() is bulbe, (name, mode)
                # The decisions, not the helper alone: no single-user bypass unless the mode reads as Daily.
                if name.endswith("csrf_middleware"):
                    assert module._is_single_user_unauthenticated() is (not bulbe), mode
                else:
                    request = SimpleNamespace(cookies={}, headers={}, client=SimpleNamespace(host="127.0.0.1"))
                    if bulbe:
                        with pytest.raises(HTTPException) as caught:
                            module.get_current_user(request, authorization=None)
                        assert caught.value.status_code == 401, mode
                    else:
                        assert module.get_current_user(request, authorization=None)["sub"] == "local"
            finally:
                restore()


# --------------------------------------------------------------------------- NF7


def test_nf7_the_mode_compares_its_codes_on_bytes_and_never_raises(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    loaded, restore = isolate(targets={"opti_oignon.encryption": source("encryption.py"),
                                       "opti_oignon.security_mode": source("security_mode.py")},
                              blocked=("opti_oignon.secure_bytes",))
    try:
        sm = loaded["opti_oignon.security_mode"]
        beyond = "\u00e9" * 64
        assert sm._same_text("123456", "123456") is True
        for given, held in (("12345\u0663", "123456"), (beyond, "e" * 64), ("123456", "1234567"), ("", "1"),
                            ("\ud800", "123456"), ("12345\udfff", "123456")):
            assert sm._same_text(given, held) is False, given
        # In constant time: the comparison is hmac's.
        same = next(node for node in ast.walk(ast.parse(source("security_mode.py").read_text(encoding="utf-8")))
                    if isinstance(node, ast.FunctionDef) and node.name == "_same_text")
        assert [ast.unparse(call.func) for call in ast.walk(same) if isinstance(call, ast.Call)
                and ast.unparse(call.func).endswith("compare_digest")] == ["_hmac.compare_digest"]
        fields = {"MODE": "bulbe", "TIMESTAMP": "1700000000", "USER_ID": "uid-resident"}
        good = sm._compute_lockfile_hmac("bulbe", 1700000000.0, "uid-resident", K1)
        assert sm._verify_lockfile(dict(fields, HMAC=good), K1) is True
        assert sm._verify_lockfile(dict(fields, HMAC=beyond), K1) is False
        assert sm._verify_lockfile(dict(fields, HMAC=good[:-1] + "\u00e9"), K1) is False
        # The downgrade's request id and visual code, typed by a client, are compared the same way.
        tree = ast.parse(source("security_mode.py").read_text(encoding="utf-8"))
        confirm = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "confirm_downgrade")
        compared = [ast.unparse(call) for call in ast.walk(confirm)
                    if isinstance(call, ast.Call) and ast.unparse(call.func) in ("_same_text", "_hmac.compare_digest")]
        assert compared == ["_same_text(request_id, req.request_id)", "_same_text(visual_code, req.visual_code)"], (
            compared)
    finally:
        restore()


# --------------------------------------------------------------------------- NF8


def test_nf8_the_app_never_starts_without_its_security_middlewares():
    # Each middleware imported and registered by a statement of the module's own body -- never inside a try, a with,
    # an except*, a condition or a function --, so that one that cannot be registered stops the app.
    tree = ast.parse(source("api", "app.py").read_text(encoding="utf-8"))
    for middleware in ("SecurityModeMiddleware", "CSRFMiddleware", "AuthMiddleware"):
        imported = [stmt for stmt in tree.body if isinstance(stmt, ast.ImportFrom)
                    and any(alias.name == middleware for alias in stmt.names)]
        registered = [stmt for stmt in tree.body if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                      and ast.unparse(stmt.value) == f"app.add_middleware({middleware})"]
        assert len(imported) == 1 and len(registered) == 1, middleware
        anywhere = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and ast.unparse(node) == f"app.add_middleware({middleware})"]
        assert len(anywhere) == 1, (middleware, "registered once, at the top")


# --------------------------------------------------------------------------- NF9


def test_nf9_csrf_checks_a_request_that_only_says_it_upgrades():
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse

    auth = types.ModuleType("opti_oignon.auth")
    auth.auth_manager = SimpleNamespace(single_user_mode=False)
    name = "opti_oignon.api.csrf_middleware"
    loaded, restore = isolate(targets={name: source("api", "csrf_middleware.py")},
                              seeded={"opti_oignon.auth": auth, "opti_oignon.security_mode": _mode_module("daily")},
                              packages=("opti_oignon.api",))
    try:
        middleware = loaded[name].CSRFMiddleware(app=None)
        assert middleware._cookie_mode and middleware._csrf_enabled

        def post(headers):
            scope = {"type": "http", "method": "POST", "path": "/api/chat/send", "raw_path": b"/api/chat/send",
                     "query_string": b"", "scheme": "http", "server": ("127.0.0.1", 8001),
                     "client": ("127.0.0.1", 50000), "root_path": "",
                     "headers": [(k.lower().encode("ascii"), v.encode("ascii")) for k, v in headers]}
            served = []

            async def call_next(request):
                served.append(True)
                return PlainTextResponse("served")

            response = asyncio.run(middleware.dispatch(Request(scope), call_next))
            return response.status_code, bool(served)

        session = ("Cookie", "oo_access_token=good; oo_csrf_token=the-token")
        assert post([session, ("Upgrade", "websocket"), ("Connection", "upgrade")]) == (403, False)
        assert post([session, ("Upgrade", "websocket")]) == (403, False)
        assert post([session]) == (403, False)
        assert post([session, ("X-CSRF-Token", "the-token"), ("Upgrade", "websocket")]) == (200, True)
    finally:
        restore()


# --------------------------------------------------------------------------- NF10


def test_nf10_the_real_account_store_passes_the_owners_password_alone(tmp_path, monkeypatch):
    bcrypt = pytest.importorskip("bcrypt")
    for name in ("OPTI_ENCRYPTION_KEY", "OPTI_KEYFILE_PASSPHRASE", "RAG_HOME"):
        monkeypatch.delenv(name, raising=False)
    connector = types.ModuleType("opti_oignon.db_utils")
    connector.safe_connect = lambda db_path, *, check_same_thread=True, timeout=5.0: sqlite3.connect(
        str(tmp_path / "accounts.db"), timeout=timeout, check_same_thread=check_same_thread)
    loaded, restore = isolate(targets={"opti_oignon.auth": source("auth.py"), FACTORS: source("ceremony_factors.py")},
                              blocked=("ollama",), seeded={"opti_oignon.db_utils": connector})
    try:
        auth, cf = loaded["opti_oignon.auth"], loaded[FACTORS]
        assert auth.auth_manager is not None
        strong = "Ceremony-Passw0rd-only-these-contracts"
        manager = auth.auth_manager
        # An empty store has no owner: the single-user session stands for no one.
        assert manager.owners() == []
        assert cf.check_password({"sub": "local"}, strong, client_ip=CLIENT).error == "no_account"
        # Registration gives the default role: the first registered account is the owner while none is an admin's,
        # by the store's order -- a clock stepped back when the later one registered changes nothing.
        first = manager.create_user("first-account", strong)
        later = manager.create_user("registered-later", strong)
        assert first is not None and later is not None and first.role == later.role == "user"
        conn = manager._get_conn()
        try:
            conn.execute("UPDATE users SET created_at = ? WHERE user_id = ?", (first.created_at - 3600.0,
                                                                                later.user_id))
            conn.commit()
        finally:
            conn.close()
        assert manager.owners() == [first.user_id]
        verdict = cf.check_password({"sub": first.user_id}, strong, client_ip=CLIENT)
        assert verdict.passed and verdict.account_id == first.user_id
        assert cf.check_password({"sub": later.user_id}, strong, client_ip=CLIENT).error == "not_owner"
        assert cf.check_password({"sub": "local"}, strong, client_ip=CLIENT).account_id == first.user_id
        # Once an account is an admin's, the admins alone are owners.
        admin = manager.create_user("an-admin", strong, role="admin")
        assert cf.check_password({"sub": admin.user_id}, strong, client_ip=CLIENT).passed
        assert cf.check_password({"sub": first.user_id}, strong, client_ip=CLIENT).error == "not_owner"
        assert cf.check_password({"sub": admin.user_id}, WRONG, client_ip=CLIENT).error == "wrong_password"
        refused = cf.check_password({"sub": admin.user_id}, LONG, client_ip=CLIENT)
        assert not refused.passed
        if int(bcrypt.__version__.split(".")[0]) >= 5:
            assert refused.error == "password_unverifiable", refused
    finally:
        restore()


# --------------------------------------------------------------------------- NF11


def _socket_routes(api):
    """Every WebSocket handler of the API, its modules walked to the bottom: a decorated function that takes a
    ``WebSocket``, or one a ``.websocket`` decorator registers. A socket registered by a call is out of sight: an
    error."""
    found = []
    for path in sorted(api.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(("add_api_websocket_route",
                                                                              "add_websocket_route")):
                raise AssertionError(f"{path.name}: a socket registered by a call, out of the census's sight")
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) or not node.decorator_list:
                continue
            takes_socket = any(arg.annotation is not None and ast.unparse(arg.annotation).endswith("WebSocket")
                               for arg in node.args.args + node.args.kwonlyargs)
            registered = any(isinstance(d, ast.Call) and ast.unparse(d.func).endswith((".websocket",
                                                                                      ".websocket_route"))
                             for d in node.decorator_list)
            if takes_socket or registered:
                found.append((f"{path.relative_to(api)}:{node.name}", node))
    return found


def _refusal(stmt, user, socket):
    """Whether ``stmt`` is ``if <user> is None:`` that closes the socket with 4001 and returns."""
    return (isinstance(stmt, ast.If) and ast.unparse(stmt.test) == f"{user} is None"
            and f"{socket}.close(code=4001)" in ast.unparse(stmt) and isinstance(stmt.body[-1], ast.Return))


def test_nf11_every_websocket_route_checks_its_handshake():
    api = Path(__file__).resolve().parent.parent / "opti_oignon" / "api"
    sockets = _socket_routes(api)
    names = [where for where, _node in sockets]
    assert "routes_agent.py:agent_stream" in names and len(names) >= 6, names
    for where, node in sockets:
        body = [stmt for stmt in node.body
                if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))]
        socket = node.args.args[0].arg
        # The socket accepted, then its handshake checked before anything else is done.
        assert ast.unparse(body[0]) == f"await {socket}.accept()", where
        guard = body[1]
        assert isinstance(guard, ast.Try), (where, "the handshake is the first thing after accept")
        bound = [stmt for stmt in guard.body if isinstance(stmt, ast.Assign)
                 and ast.unparse(stmt.value) == f"await authenticate_websocket({socket})"]
        assert len(bound) == 1 and len(bound[0].targets) == 1 and isinstance(bound[0].targets[0], ast.Name), where
        user = bound[0].targets[0].id
        # A refusal that closes with 4001 and returns: in the guard, or right after it.
        inside = any(_refusal(stmt, user, socket) for stmt in guard.body)
        after = _refusal(body[2], user, socket)
        assert inside or after, where
        # A handshake that raised gives no user: each except branch refuses, or leaves the user None for the
        # refusal right after it -- none hands the route a user, none lets it go on.
        assert guard.handlers and not guard.orelse and not guard.finalbody, where
        for handler in guard.handlers:
            refuses = f"{socket}.close(code=4001)" in ast.unparse(handler) and isinstance(handler.body[-1],
                                                                                         ast.Return)
            nothing = all(isinstance(stmt, ast.Assign) and ast.unparse(stmt) == f"{user} = None"
                          for stmt in handler.body)
            assert refuses or (nothing and after), (where, ast.unparse(handler))
        # Nothing but the check and its refusal is awaited before the refusal is decided.
        checked = [guard] + ([body[2]] if after else [])
        awaited = [ast.unparse(n.value) for stmt in checked for n in ast.walk(stmt) if isinstance(n, ast.Await)]
        others = [text for text in awaited
                  if text != f"authenticate_websocket({socket})" and not text.startswith(
                      (f"{socket}.close(", f"{socket}.send_json(", f"_send_token({socket},"))]
        assert others == [], (where, others)


# --------------------------------------------------------------------------- NF12, NF13

_SOCKET_DRIVER = r"""
const clause = process.argv[2];
const made = [];
class StoodIn {
  constructor(url) { this.url = url; this.readyState = 0; made.push(this); }
  send() {}
  close() {}
}
StoodIn.OPEN = 1;
globalThis.WebSocket = StoodIn;
const timers = [];
globalThis.setTimeout = (fn, ms) => { timers.push(ms); return timers.length; };
globalThis.clearTimeout = () => {};
const { ReconnectingWebSocket, WS_AUTH_REFUSED } = await import(process.env.OO_CLIENT);
function fail(why) { console.log('FAIL ' + why); process.exit(1); }
if (WS_AUTH_REFUSED !== 4001) fail('the refused code is ' + WS_AUTH_REFUSED);
const refused = new ReconnectingWebSocket('ws://localhost/api/agent/stream');
let closes = 0, reconnects = 0;
refused.onclose = (ev) => { closes += 1; if (ev.code !== 4001) fail('the close code is lost'); };
refused.onreconnect = () => { reconnects += 1; };
made[0].onopen({});
made[0].onclose({ code: 4001 });
if (made.length !== 1 || timers.length !== 0 || reconnects !== 0 || closes !== 1) {
  fail(`refused: sockets ${made.length}, timers ${timers.length}, reconnects ${reconnects}, closes ${closes}`);
}
const lost = new ReconnectingWebSocket('ws://localhost/api/agent/stream');
let lostCloses = 0;
lost.onclose = () => { lostCloses += 1; };
made[1].onopen({});
made[1].onclose({ code: 1006 });
if (timers.length !== 1 || lostCloses !== 0) fail(`lost: timers ${timers.length}, closes ${lostCloses}`);
console.log('PASS ' + clause);
"""


def test_nf12_the_pages_socket_stops_once_its_handshake_is_refused():
    from _frontend import run_ts

    run_ts({"OO_CLIENT": "frontend/src/lib/api/client.ts"}, _SOCKET_DRIVER, "refused")


_PANELS = {
    "frontend/src/lib/components/settings/SecurityModePanel.svelte": "confirmError = ",
    "frontend/src/lib/components/settings/SearchKillSwitchPanel.svelte": "confirmError = ",
    "frontend/src/lib/components/settings/PluginAllowlistPanel.svelte": "approveError = ",
    "frontend/src/lib/components/settings/sections/AccountAuthMode.svelte": "toastError(",
}


def test_nf13_the_ceremonies_panels_show_the_servers_reason():
    from _frontend import read

    for path, sink in _PANELS.items():
        text = read(path)
        assert "import { httpRefusal } from '$lib/switches/serverSwitch';" in text, path
        catches = [chunk.split("}", 1)[0] for chunk in text.split("catch (e")[1:]]
        shown = [line.strip() for chunk in catches for line in chunk.splitlines()
                 if sink in line and "httpRefusal(e)?.detail ||" in line]
        assert shown, (path, "a refusal shown without the server's reason")
    # The switch shows the mode the server answers is in force, not the one it asked for.
    switch = read("frontend/src/lib/components/settings/sections/AccountAuthMode.svelte")
    assert "singleUserMode = answer.single_user_mode ?? newMode;" in switch
    assert "singleUserMode = newMode;" not in switch
