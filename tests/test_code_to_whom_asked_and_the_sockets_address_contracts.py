#!/usr/bin/env python3
"""Contracts for the ceremonies' visual codes and the address a limiter counts a client under.

The visual codes of leaving Bulbe mode and of re-enabling search were served to any session, whoever had asked, and
said in their docstrings that they never left the API as JSON; any session could ask to leave Bulbe, replacing the
owner's request. A re-enable request ended in its reply only. Three readers of a client's address took
X-Forwarded-For -- from a loopback peer, or from anyone -- the cookies' secure context took X-Forwarded-Proto from
anyone, and uvicorn, started with its defaults, already took both from 127.0.0.1, where every local client connects
from: each could pick the address the limiters counted it under, and the scheme the Host gate and the cookies read.

  * VA1 -- leaving Bulbe is the owner's to ask; its code is served to the account that asked only, 403 to another,
    404 once the request is confirmed or past its end, where the confirmation is refused (410); the request, the code
    and the confirmation read the session's identity alike.
  * VA2 -- the re-enable's code is served to the owner's account that asked only, 403 to another account or one that
    is no owner's, 404 past the request's end (ten minutes after its cooldown, as the downgrade's), where the
    confirmation is refused (410), the request closed and the status no longer pending; while it lives, the status
    names it and its end, for a page reloaded mid-ceremony.
  * VA3 -- the login's, the upload's and the account routes' readers give the client the Host gate hands on, whatever
    X-Forwarded-For says, loopback peer or not (b2's first and third findings said again); the cookies' secure
    context reads the gate's scheme, not X-Forwarded-Proto; no module but the gate names a forwarded header.
  * VA4 -- the proxies ``api.trusted_proxies`` of ``security.yaml`` lists, read whole or not at all, each entry as
    written; the gate takes that list; the launcher hands uvicorn no forwarded header (``proxy_headers=False``); tg4 of
    tests/test_revoked_device_never_reaches_the_api_contracts.py run whole with that key.
  * VA5 -- a real uvicorn server on the launcher's arguments, the real gate in front of the app: a listed proxy's
    forwarded client and scheme are taken, as uvicorn would read them where both read one; any other forwarded
    request is handed on as ``unlisted-proxy``, even under uvicorn's own defaults, which, the witness, take any local
    client's. Where the gate is stricter: every X-Forwarded-For line, a hop that is no address written alone or a
    loopback one never read, a listed proxy's one X-Forwarded-Proto read on its own, each forwarded header a trigger,
    a websocket handshake read, the scheme read before the site check.
  * VA6 -- every launcher of the tree, and every command the docs give, starts uvicorn with its proxy headers off;
    ul14 of tests/test_launcher_unlock_contracts.py run whole with the backend script's flag.
  * VA7 -- the texts say it; op7 of tests/test_another_port_is_another_site_contracts.py run whole with the sentence
    that names the list.
  * VA8 -- a request an unlisted proxy forwarded is never a local one: the localhost-only routes refuse it, the
    remote check calls it remote, Bulbe's local-only layer rejects it (said as a forwarded request, not as a bind
    guard's breach), its cookies get no loopback exemption.

Every module under contract is loaded from its file through the shared isolation window, the account store, the
limiter and the managers stood in for where a contract says so; VA1 and VA2 run the real managers on their own
clocks, VA5 a real server on an ephemeral loopback port. Nothing reads or writes the platform's data.

Local-only (the public distribution ships no tests).
"""

import ast
import asyncio
import http.client
import inspect
import json
import logging
import re
import socket
import sys
import textwrap
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_census  # noqa: E402
import test_another_port_is_another_site_contracts as op  # noqa: E402
import test_auth_2fa_routes_seam_contracts as seam  # noqa: E402
import test_bulbe_paths_contracts as bp  # noqa: E402
import test_checks_fail_closed_contracts as nf  # noqa: E402
import test_launcher_unlock_contracts as ul  # noqa: E402
import test_revoked_device_never_reaches_the_api_contracts as tg  # noqa: E402
import test_search_gates_contracts as sg  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402
from test_revoked_device_never_reaches_the_api_contracts import (
    world,  # noqa: E402,F401  (tg4's fixture, for va4)
)

_SM = "opti_oignon.security_mode"
_KS = "opti_oignon.search_killswitch"
_HG = "opti_oignon.api.host_gate"
_MAIN = "opti_oignon.main"
_GUARD = "opti_oignon.remote_session_guard"

BUDGET_S = {
    "test_va1_leaving_bulbe_is_the_owners_to_ask_and_its_code_the_asking_accounts": 1.5,
    "test_va2_the_re_enable_code_is_the_asking_accounts_and_ends_with_its_request": 1.5,
    "test_va3_the_limiters_read_the_gates_client_and_no_other_module_names_a_forwarded_header": 8.0,
    "test_va4_the_listed_proxies_are_read_as_written_and_uvicorn_reads_no_forwarded_header": 3.0,
    "test_va5_a_real_server_hands_a_forwarded_client_on_from_a_listed_proxy_only": 6.0,
    "test_va6_every_launcher_starts_uvicorn_with_its_proxy_headers_off": 6.0,
    "test_va7_the_texts_say_it": 2.0,
    "test_va8_a_request_an_unlisted_proxy_forwarded_is_never_a_local_one": 2.0,
}

OWNER = nf.RESIDENT
SECOND_OWNER = nf._Account("uid-second", "second", nf.RIGHT)
USER = nf._Account("uid-user", "registered", nf.RIGHT, role="user")
UNLISTED = "unlisted-proxy"


def _session(account):
    return {"sub": account.user_id, "username": account.username, "type": "access"}


def _status(coroutine):
    """A route's coroutine run on a loop of its own: ``(status, answer)``, the HTTPException's when it raises one."""
    from fastapi import HTTPException

    try:
        return 200, asyncio.run(coroutine)
    except HTTPException as exc:
        return exc.status_code, exc.detail


def _mode_manager(clock):
    """The real security mode module in a window, on ``clock``, its audit kept in memory; a manager in Bulbe mode."""
    loaded, restore = isolate(targets={_SM: source("security_mode.py")})
    sm = loaded[_SM]
    sm.time = SimpleNamespace(time=lambda: clock[0])
    sm.audit = []
    sm._audit_log = lambda event, severity="INFO", **details: sm.audit.append(event)
    manager = sm.SecurityModeManager()
    manager.get_current_mode = lambda: "bulbe"
    return sm, manager, restore


def _routes(manager=None, switch=None):
    """``routes_security`` in a window, the ceremony check real, the account store holding two owners and a user,
    the managers the ones given (or empty stand-ins)."""
    accounts = nf._Accounts(OWNER, SECOND_OWNER, USER)
    auth_routes = types.ModuleType("opti_oignon.api.routes_auth")
    auth_routes._get_current_user = lambda: _session(OWNER)
    auth_routes._extract_client_ip = lambda request: nf.CLIENT
    mode = types.ModuleType(_SM)
    mode.MODE_BULBE, mode.MODE_DAILY, mode.VALID_MODES = "bulbe", "daily", ("bulbe", "daily")
    mode.security_mode_manager = manager or SimpleNamespace()
    mode._audit_log = lambda event, severity="INFO", **details: None
    mode.get_current_mode, mode.is_bulbe = (lambda: "daily"), (lambda: False)
    allow = types.ModuleType(nf.ALLOW)
    allow.plugin_allowlist_manager = SimpleNamespace()
    ks = types.ModuleType(_KS)
    ks.search_killswitch = switch or SimpleNamespace()
    loaded, restore = isolate(
        targets={nf.FACTORS: source("ceremony_factors.py"), nf.ROUTES: source("api", "routes_security.py")},
        seeded={"opti_oignon.auth": nf._auth(accounts, nf._Limiter(), nf._Verifier()),
                "opti_oignon.api.routes_auth": auth_routes, _SM: mode, nf.ALLOW: allow, _KS: ks},
        packages=("opti_oignon.api",),
    )
    return loaded[nf.ROUTES], restore


def _flat(text):
    return " ".join(text.split())


def _read(*parts):
    return REPO.joinpath(*parts).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- VA1


def test_va1_leaving_bulbe_is_the_owners_to_ask_and_its_code_the_asking_accounts():
    clock = [1_000_000.0]
    sm, manager, restore_mode = _mode_manager(clock)
    try:
        # The manager: the code is read with the request it belongs to; a request ends ten minutes past its cooldown.
        asked = manager.request_downgrade("uid-resident")
        assert asked["pending"] is True and "visual_code" not in asked, asked
        assert asked["expires_at"] == clock[0] + sm.DOWNGRADE_COOLDOWN_SECONDS + 600, asked
        found, code = manager.visual_code_for("uid-resident")
        assert found == "ok" and re.fullmatch(r"[0-9]{6}", code or ""), (found, code)
        assert manager.get_pending_visual_code() == code
        assert manager.visual_code_for("uid-second") == ("other", None)
        clock[0] = asked["expires_at"]
        assert manager.visual_code_for("uid-resident") == ("ok", code), "the request lives up to its end"
        clock[0] = asked["expires_at"] + 1
        assert manager.visual_code_for("uid-resident") == ("none", None), "a request past its end serves no code"
        assert manager.get_pending_visual_code() is None
        asked = manager.request_downgrade("uid-resident")
        manager._pending_downgrade.confirmed = True
        assert manager.visual_code_for("uid-resident") == ("none", None), "a confirmed request serves no code"
        assert manager.get_pending_visual_code() is None
        manager._pending_downgrade = None
        assert manager.visual_code_for("uid-resident") == ("none", None)

        # The routes, on the real manager.
        routes, restore = _routes(manager=manager)
        try:
            def ask(account):
                return _status(routes.request_mode_downgrade(_session(account),
                                                             routes._session_account(_session(account))))

            status, asked = ask(OWNER)
            assert status == 200 and asked["pending"] is True and "visual_code" not in asked, (status, asked)
            held = manager._pending_downgrade
            # Another account's session asks nothing, and the owner's request stays.
            status, detail = ask(USER)
            assert status == 403 and "owner" in str(detail), (status, detail)
            assert manager._pending_downgrade is held, "a user's session replaced the owner's request"
            status, served = _status(routes.get_visual_code(_session(OWNER)))
            assert status == 200 and served == {"visual_code": held.visual_code}, served
            for other in (_session(SECOND_OWNER), _session(USER), {"user_id": "uid-resident-not"}, None):
                status, detail = _status(routes.get_visual_code(other))
                assert status == 403 and held.visual_code not in str(detail), (other, detail)
                assert "account that asked" in str(detail), detail
            clock[0] = asked["expires_at"] + 1
            assert _status(routes.get_visual_code(_session(OWNER)))[0] == 404, "past its end, nothing is served"
            body = routes.DowngradeConfirmRequest(request_id=asked["request_id"], visual_code=held.visual_code,
                                                  password=nf.RIGHT)
            status, detail = _status(routes.confirm_mode_downgrade(body, object(), _session(OWNER)))
            assert status == 410 and "expired" in str(detail).lower(), (status, detail)
            # One identity for the three: the session's sub, else its user_id, else "local".
            for current, identity in ((_session(OWNER), "uid-resident"), ({"user_id": "uid-x"}, "uid-x"),
                                      ({"sub": "uid-a", "user_id": "uid-b"}, "uid-a"),
                                      ({"sub": "", "user_id": ""}, "local"), (None, "local"), ("token", "local")):
                assert routes._ceremony_user_id(current) == identity, (current, identity)
            for name in ("request_mode_downgrade", "get_visual_code", "confirm_mode_downgrade"):
                text = inspect.getsource(getattr(routes, name))
                assert "_ceremony_user_id(current)" in text, name
                assert '.get("sub")' not in text, (name, "the identity read apart")
        finally:
            restore()
    finally:
        restore_mode()


# --------------------------------------------------------------------------- VA2


def test_va2_the_re_enable_code_is_the_asking_accounts_and_ends_with_its_request(tmp_path):
    state = tmp_path / "state.json"
    mod, _sm, restore_switch = sg._ks_window(tmp_path, state)
    try:
        switch = mod.search_killswitch
        clock = [1_000_000.0]
        mod.time = SimpleNamespace(time=lambda: clock[0])
        assert mod.REENABLE_GRACE_SECONDS == 600, "ten minutes past the cooldown, as the downgrade's and the doc say"
        assert switch.kill(reason="manual").get("persisted") is True
        # The switch: the code is read with the request it belongs to, while the request lives.
        asked = switch.request_reenable("uid-resident")
        assert asked["pending"] is True and "visual_code" not in asked, asked
        assert asked["expires_at"] == clock[0] + mod.REENABLE_COOLDOWN_SECONDS + 600, asked
        found, code = switch.reenable_visual_code_for("uid-resident")
        assert found == "ok" and re.fullmatch(r"[0-9]{6}", code or ""), (found, code)
        assert switch.get_reenable_visual_code() == code
        assert switch.reenable_visual_code_for("uid-second") == ("other", None)
        status = switch.status()
        assert status["reenable_pending"] is True and status["reenable_expires_at"] == asked["expires_at"], status
        assert status["reenable_request_id"] == asked["request_id"], "a page reloaded mid-ceremony confirms it"
        clock[0] = asked["expires_at"] + 1
        assert switch.reenable_visual_code_for("uid-resident") == ("none", None), "past its end, no code"
        assert switch.get_reenable_visual_code() is None
        status = switch.status()
        assert status["reenable_pending"] is False and status["reenable_expires_at"] is None, status
        assert status["reenable_request_id"] is None, status
        refused = switch.confirm_reenable(request_id=asked["request_id"], visual_code=code, user_id="uid-resident")
        assert refused["success"] is False and refused["error"] == "expired", refused
        assert switch.status()["reenable_pending"] is False and switch.is_killed(), "the request is closed"
        again = switch.confirm_reenable(request_id=asked["request_id"], visual_code=code, user_id="uid-resident")
        assert again["error"] == "no_pending_request", again
        # Witness: inside its end, the same ceremony re-enables.
        asked = switch.request_reenable("uid-resident")
        code = switch.get_reenable_visual_code()
        clock[0] = asked["expires_at"]
        done = switch.confirm_reenable(request_id=asked["request_id"], visual_code=code, user_id="uid-resident")
        assert done["success"] is True and switch.is_enabled(), done

        # The routes, on the real switch: the owner's account that asked, no other.
        assert switch.kill(reason="manual").get("persisted") is True
        routes, restore = _routes(switch=switch)
        try:
            asked = switch.request_reenable("uid-resident")
            code = switch.get_reenable_visual_code()
            owner = routes._session_account(_session(OWNER))
            status, served = _status(routes.get_search_reenable_code(owner))
            assert status == 200 and served == {"visual_code": code}, (status, served)
            for account in (SECOND_OWNER, USER):
                held = routes._session_account(_session(account))
                status, detail = _status(routes.get_search_reenable_code(held))
                assert status == 403 and code not in str(detail), (account.user_id, status, detail)
            clock[0] = asked["expires_at"] + 1
            assert _status(routes.get_search_reenable_code(owner))[0] == 404, "past its end, nothing is served"
            body = routes.SearchReenableConfirm(request_id=asked["request_id"], visual_code=code, password=nf.RIGHT)
            status, detail = _status(routes.confirm_search_reenable(body, object(), _session(OWNER)))
            assert status == 410 and "expired" in str(detail), (status, detail)
            assert switch.is_killed() and switch.status()["reenable_pending"] is False
        finally:
            restore()
    finally:
        restore_switch()


# --------------------------------------------------------------------------- VA3

# The headers a proxy forwards a client's address or scheme in, as a module would spell them.
_FORWARDED = ("x-forwarded-for", "x-real-ip", "x-forwarded-proto", "x-forwarded-host", "x-client-ip",
              "true-client-ip", "cf-connecting-ip", "forwarded")


def _forwarded_names(text, path="<module>"):
    """The forwarded headers a module's code names, each with the top-level statement it is in: its string and bytes
    constants that are such a header, any case, docstrings left out (a docstring may say what is never read)."""
    tree = ast.parse(text, path)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    found = []
    for top in tree.body:
        name = getattr(top, "name", None) or next(
            (target.id for target in getattr(top, "targets", ()) if isinstance(target, ast.Name)), None)
        for node in ast.walk(top):
            if isinstance(node, ast.Constant) and id(node) not in docstrings and isinstance(node.value, (str, bytes)):
                value = node.value.decode("latin-1") if isinstance(node.value, bytes) else node.value
                if value.strip().lower() in _FORWARDED:
                    found.append((path, name, value))
    return found


def test_va3_the_limiters_read_the_gates_client_and_no_other_module_names_a_forwarded_header():
    remote = seam._Request(host="203.0.113.9", headers={"X-Forwarded-For": "1.2.3.4"})
    proxied = seam._Request(host="127.0.0.1", headers={"X-Forwarded-For": "198.51.100.7, 203.0.113.9",
                                                         "x-forwarded-for": "198.51.100.7", "X-Real-IP": "6.6.6.6"})
    bare = seam._Request(host="127.0.0.1")
    marked = seam._Request(host=UNLISTED, headers={"X-Forwarded-For": "6.6.6.6"})
    nobody = seam._Request()
    nobody.client = None
    readers = []
    ctx = seam._load()
    try:
        readers.append(("routes_auth", ctx.routes._extract_client_ip))
        for name in ("routes_files", "routes_users"):
            loaded, restore = isolate(targets={"opti_oignon.api.schemas": source("api", "schemas.py"),
                                               f"opti_oignon.api.{name}": source("api", f"{name}.py")},
                                      packages=("opti_oignon.api",))
            readers.append((name, loaded[f"opti_oignon.api.{name}"]._get_client_ip))
            restore()
        for name, read in readers:
            # b2's first and third findings, said again.
            assert read(remote) == "203.0.113.9", (name, "a remote peer's forwarded header is ignored")
            assert read(bare) == "127.0.0.1", name
            # The loopback peer's header is no longer the client's: any local client could send one.
            assert read(proxied) == "127.0.0.1", (name, read(proxied))
            assert read(marked) == UNLISTED, name
            assert read(nobody) == "unknown", name
        # The cookies' secure context reads the URL's scheme (the gate's), never a forwarded one a client sent.
        ctx.routes._get_security_jwt_config = lambda: {"cookie_secure": False}
        claimed = seam._Request(host="203.0.113.9", headers={"X-Forwarded-Proto": "https"})
        assert ctx.routes._detect_secure_context(claimed) is False, "a client's forwarded scheme is not https"
        claimed.url = "https://onion.lan/probe"
        assert ctx.routes._detect_secure_context(claimed) is True, "witness: the gate's https is"
    finally:
        ctx.restore()

    # No module of the package but the gate names a forwarded header in its code: the gate only in its list and in
    # the one reading of it.
    witness = ('def f(request):\n    """X-Forwarded-For is never read."""\n'
               '    return request.headers.get("X-Forwarded-For")\n')
    assert _forwarded_names(witness) == [("<module>", "f", "X-Forwarded-For")], (
        "witness: a read is seen, a docstring not")
    assert _forwarded_names('h = dict(scope["headers"])[b"x-real-ip"]\n'), "witness: bytes, another header"
    found, read = [], 0
    for path in _allium_census.modules().values():
        found += _forwarded_names(Path(path).read_text(encoding="utf-8"), str(Path(path).relative_to(REPO)))
        read += 1
    assert read > 300, read
    gate = [entry for entry in found if entry[0] == "opti_oignon/api/host_gate.py"]
    assert {entry[1] for entry in gate} == {"FORWARDED_HEADERS", "through_proxy", "_forwarded_scheme"}, gate
    assert sorted({entry[2] for entry in gate if entry[1] == "FORWARDED_HEADERS"}) == [
        "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "x-real-ip"], gate
    assert [entry for entry in found if entry[0] != "opti_oignon/api/host_gate.py"] == [], found


# --------------------------------------------------------------------------- VA4


def _gate():
    loaded, restore = isolate(targets={_HG: source("api", "host_gate.py")}, packages=("opti_oignon.api",))
    return loaded[_HG], restore


def _launcher():
    """``main`` in a window, ``host_gate`` unreachable from it: the launcher needs no reading of its own."""
    loaded, restore = isolate(targets={"opti_oignon.__version__": source("__version__.py"), _MAIN: source("main.py")},
                              blocked=(_HG,), packages=("opti_oignon.api",))
    return loaded[_MAIN], restore


def test_va4_the_listed_proxies_are_read_as_written_and_uvicorn_reads_no_forwarded_header(
        tmp_path, caplog, world, capsys):  # noqa: F811
    gate, restore = _gate()
    own = tmp_path / "va4"  # tg4's world keeps its own security.yaml at the top of tmp_path
    own.mkdir()
    try:
        path = own / "security.yaml"
        assert gate.trusted_proxies(own / "absent.yaml") == ()
        for text, listed in (("security_mode: daily\n", ()), ("api:\n  hosts: [onion.lan]\n", ()),
                             ("api:\n  trusted_proxies: []\n", ()),
                             ("api:\n  trusted_proxies: [10.0.0.5, 10.0.0.0/8, '::1', 10.0.0.5]\n",
                              ("10.0.0.5", "10.0.0.0/8", "::1")),
                             ("api:\n  trusted_proxies: [0.0.0.0/1]\n", ("0.0.0.0/1",))):
            path.write_text(text, encoding="utf-8")
            caplog.clear()
            assert gate.trusted_proxies(path) == listed, text
            assert not caplog.records, (text, caplog.text)
        for text in ("api:\n  trusted_proxies: ['*']\n", "api:\n  trusted_proxies: [0.0.0.0/0]\n",
                     "api:\n  trusted_proxies: ['::/0']\n", "api:\n  trusted_proxies: [0.0.0.0/1, 128.0.0.0/1]\n",
                     "api:\n  trusted_proxies: ['::/1', '8000::/1']\n", "api:\n  trusted_proxies: [proxy.example]\n",
                     "api:\n  trusted_proxies: [10.0.0.5/24]\n", "api:\n  trusted_proxies: ['::ffff:127.0.0.1']\n",
                     "api:\n  trusted_proxies: ['fe80::1%eth0']\n", "api:\n  trusted_proxies: [10.0.0.5, 7]\n",
                     "api:\n  trusted_proxies: [' 10.0.0.5']\n", "api:\n  trusted_proxies: 10.0.0.5\n",
                     "api:\n  trusted_proxies: [10.0.0.5, null]\n", "api: [unclosed\n"):
            path.write_text(text, encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                assert gate.trusted_proxies(path) == (), text
            assert "no proxy is trusted" in caplog.text, (text, caplog.text)
        # The gate takes the list security.yaml gives, unless told another; app.py tells it none.
        init = inspect.getsource(gate.HostGateMiddleware.__init__)
        assert "trusted_proxies() if proxies is None" in init, init
        assert gate.HostGateMiddleware(None, hosts=(), origins=(), proxies=("10.0.0.0/8", "::1")).networks == (
            gate.ipaddress.ip_network("10.0.0.0/8"), gate.ipaddress.ip_network("::1"))
        app = _read("opti_oignon", "api", "app.py")
        assert "app.add_middleware(HostGateMiddleware, origins=_listed_origins)" in app and "proxies=" not in app
    finally:
        restore()

    # The launcher hands uvicorn no forwarded header, needing no reading of its own.
    main, restore = _launcher()
    try:
        assert main._proxy_kwargs() == {"proxy_headers": False}
        assert "**ssl_kwargs, **_proxy_kwargs())" in inspect.getsource(main.cmd_api)
    finally:
        restore()

    # tg4, run whole, its two key sets holding the proxy key the launcher now gives.
    text = textwrap.dedent(inspect.getsource(tg.test_tg4_a_network_address_is_served_with_the_whole_mtls_or_not_at_all))
    tree = ast.parse(text)
    replaced = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Set) and sorted(getattr(e, "value", None) for e in node.elts) == ["host", "port"]:
            node.elts.append(ast.Constant("proxy_headers"))
            replaced += 1
    assert replaced == 2, replaced
    namespace = dict(vars(tg))
    exec(compile(ast.fix_missing_locations(tree), tg.__file__, "exec"), namespace)
    namespace["test_tg4_a_network_address_is_served_with_the_whole_mtls_or_not_at_all"](world, capsys)
    assert world.runs and all(run.get("proxy_headers") is False for run in world.runs), world.runs


# --------------------------------------------------------------------------- VA5


async def _echo(scope, receive, send):
    """An ASGI app that answers the client and the scheme it was handed."""
    if scope["type"] != "http":
        return
    body = json.dumps({"client": list(scope.get("client") or ()), "scheme": scope.get("scheme")}).encode()
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


def _seen(app, kwargs, headers):
    """What a real uvicorn server on ``kwargs`` hands ``app`` for a local client that sends ``headers``."""
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, lifespan="off",
                                           **kwargs))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "the server did not start"
            time.sleep(0.02)
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.request("GET", "/", headers=headers)
            answer = json.loads(connection.getresponse().read())
        finally:
            connection.close()
        return answer["client"][0], answer["scheme"]
    finally:
        server.should_exit = True
        thread.join(10)


def test_va5_a_real_server_hands_a_forwarded_client_on_from_a_listed_proxy_only():
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    main, restore = _launcher()
    try:
        launched = main._proxy_kwargs()
    finally:
        restore()
    gate, restore = _gate()
    try:
        forwarded = {"X-Forwarded-For": "6.6.6.6", "X-Forwarded-Proto": "https"}
        assert _seen(_echo, {}, forwarded) == ("6.6.6.6", "https"), (
            "witness: uvicorn's defaults take any local client's")

        def gated(proxies):
            return gate.HostGateMiddleware(_echo, hosts=(), origins=(), proxies=proxies)

        # None listed, as shipped: the socket's client, or one unlisted client for anything forwarded.
        assert _seen(gated(()), launched, {}) == ("127.0.0.1", "http")
        assert _seen(gated(()), launched, forwarded) == (UNLISTED, "http")
        assert _seen(gated(()), launched, {"X-Real-IP": "6.6.6.6"}) == (UNLISTED, "http")
        # Under uvicorn's own defaults too, the gate hands no address a client chose.
        assert _seen(gated(()), {}, forwarded)[0] == UNLISTED
        # The loopback listed: the proxy's forwarded client and scheme.
        assert _seen(gated(("127.0.0.1",)), launched, forwarded) == ("6.6.6.6", "https")
        assert _seen(gated(("127.0.0.1",)), launched, {"X-Forwarded-For": "not-an-address"}) == (UNLISTED, "http")
        assert _seen(gated(("127.0.0.1",)), launched, {"X-Forwarded-For": "127.0.0.1"}) == (UNLISTED, "http"), (
            "a chain of proxies alone names no client")
    finally:
        restore()

    # The gate reads a listed proxy's headers as uvicorn would.
    gate, restore = _gate()
    try:
        seen = {}

        async def keep(scope, receive, send):
            seen["scope"] = scope

        for listed, chain, proto in ((("127.0.0.1",), "6.6.6.6", "https"), (("127.0.0.1",), "6.6.6.6, 7.7.7.7", "http"),
                                     (("127.0.0.1", "10.0.0.0/8"), "6.6.6.6, 10.1.2.3", "https"),
                                     (("127.0.0.0/8",), "2001:db8::1, 127.0.0.9", None)):
            headers = [(b"host", b"127.0.0.1:8001"), (b"x-forwarded-for", chain.encode())]
            if proto:
                headers.append((b"x-forwarded-proto", proto.encode()))
            scope = {"type": "http", "client": ("127.0.0.1", 50000), "scheme": "http", "headers": headers}
            ours = gate.through_proxy(scope, tuple(gate.ipaddress.ip_network(entry) for entry in listed))
            asyncio.run(ProxyHeadersMiddleware(keep, trusted_hosts=list(listed))(dict(scope), None, None))
            theirs = seen["scope"]
            assert (ours["client"][0], ours["scheme"]) == (theirs["client"][0], theirs["scheme"]), (listed, chain)

        # Where the gate is stricter than uvicorn, and every line and word it reads.
        def through(headers, *, listed=("127.0.0.1",), peer=("127.0.0.1", 50000), kind="http"):
            scope = {"type": kind, "client": peer, "scheme": "ws" if kind == "websocket" else "http",
                     "headers": [(b"host", b"127.0.0.1:8001"), *headers]}
            got = gate.through_proxy(scope, tuple(gate.ipaddress.ip_network(entry) for entry in listed))
            return got["client"][0], got["scheme"]

        xff, xfp = b"x-forwarded-for", b"x-forwarded-proto"
        for headers, listed, peer, kind, expected in (
            ([(xff, b"6.6.6.6, unknown")], ("127.0.0.1",), ("127.0.0.1", 1), "http", (UNLISTED, "http")),
            ([(xff, b"127.0.0.1, 203.0.113.9:5")], ("10.0.0.5",), ("10.0.0.5", 1), "http", (UNLISTED, "http")),
            ([(xff, b"6.6.6.6"), (xff, b"7.7.7.7")], ("127.0.0.1",), ("127.0.0.1", 1), "http", ("7.7.7.7", "http")),
            ([(xff, b"::1")], ("127.0.0.1",), ("127.0.0.1", 1), "http", (UNLISTED, "http")),
            ([(xff, b"203.0.113.9, ::ffff:127.0.0.1")], ("127.0.0.1",), ("127.0.0.1", 1), "http", (UNLISTED, "http")),
            ([(xfp, b"https")], ("127.0.0.1",), ("127.0.0.1", 1), "http", (UNLISTED, "https")),
            ([(xff, b"127.0.0.1"), (xfp, b"https")], ("127.0.0.1",), ("127.0.0.1", 1), "http", (UNLISTED, "https")),
            ([(xff, b"6.6.6.6"), (xfp, b"https"), (xfp, b"http")], ("127.0.0.1",), ("127.0.0.1", 1), "http",
             ("6.6.6.6", "http")),
            ([(xff, b"6.6.6.6"), (xfp, b"httpsx")], ("127.0.0.1",), ("127.0.0.1", 1), "http", ("6.6.6.6", "http")),
            ([(xff, b"6.6.6.6"), (xfp, b"wss")], ("127.0.0.1",), ("127.0.0.1", 1), "http", ("6.6.6.6", "https")),
            ([(xff, b"6.6.6.6"), (xfp, b"https")], ("127.0.0.1",), ("127.0.0.1", 1), "websocket", ("6.6.6.6", "wss")),
            ([(xff, b"6.6.6.6"), (xfp, b"https")], ("127.0.0.1",), ("127.0.0.1", 0), "http", (UNLISTED, "http")),
            ([(b"forwarded", b"for=6.6.6.6")], (), ("127.0.0.1", 1), "http", (UNLISTED, "http")),
            ([(b"x-forwarded-host", b"onion.lan")], (), ("127.0.0.1", 1), "http", (UNLISTED, "http")),
            ([(b"X-Real-IP", b"6.6.6.6")], (), ("127.0.0.1", 1), "http", (UNLISTED, "http")),
            ([(xff, b"6.6.6.6")], (), ("127.0.0.1", 1), "websocket", (UNLISTED, "ws")),
        ):
            assert through(headers, listed=listed, peer=peer, kind=kind) == expected, (headers, listed, peer, kind)

        # Through the gate itself: a websocket handshake is read too, and a listed TLS front's page may change
        # things, as the scheme it says is read before the site check; an unlisted one's may not.
        async def sent_through(app, scope):
            messages = []

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            await app(scope, receive, send)
            return messages

        handed = {}

        async def inner(scope, receive, send):
            handed["scope"] = scope
            if scope["type"] == "http":
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b""})

        socket_scope = {"type": "websocket", "client": ("127.0.0.1", 50000), "scheme": "ws", "path": "/ws",
                        "headers": [(b"host", b"127.0.0.1:8001"), (b"x-forwarded-for", b"6.6.6.6")]}
        asyncio.run(sent_through(gate.HostGateMiddleware(inner, hosts=(), origins=(), proxies=()), socket_scope))
        assert handed["scope"]["client"][0] == UNLISTED, "a websocket handshake is read as a request is"
        page = {"type": "http", "method": "POST", "path": "/api/x", "client": ("127.0.0.1", 50000), "scheme": "http",
                "headers": [(b"host", b"onion.lan"), (b"origin", b"https://onion.lan"),
                            (b"x-forwarded-for", b"203.0.113.9"), (b"x-forwarded-proto", b"https")]}
        for proxies, status in ((("127.0.0.1",), 200), ((), 403)):
            handed.clear()
            app = gate.HostGateMiddleware(inner, hosts=("onion.lan",), origins=(), proxies=proxies)
            answer = asyncio.run(sent_through(app, dict(page)))
            assert answer[0]["status"] == status, (proxies, answer[0])
        assert handed == {}, "the unlisted front's change reached the app"
    finally:
        restore()


# --------------------------------------------------------------------------- VA6

_APP = "opti_oignon.api.app:app"


def _shell_commands(text):
    """The commands of a shell or Markdown text that start the API with uvicorn, continuation lines joined."""
    joined = re.sub(r"\\\n", " ", text)
    return [line for line in joined.splitlines() if "uvicorn" in line and _APP in line]


def _python_starts(text, path):
    """The argument lists a module builds to start the API with uvicorn: list literals holding both words."""
    found = []
    for node in ast.walk(ast.parse(text, path)):
        if isinstance(node, ast.List):
            words = [element.value for element in node.elts if isinstance(element, ast.Constant)]
            if "uvicorn" in words and _APP in words:
                found.append(words)
    return found


def _cmd(text):
    """The exec-form CMD of a Dockerfile's text, continuation lines joined: its list of words."""
    joined = re.sub(r"\\\n", " ", text)
    commands = [line[len("CMD"):].strip() for line in joined.splitlines() if line.startswith("CMD ")]
    assert len(commands) == 1, commands
    return json.loads(commands[0])


def test_va6_every_launcher_starts_uvicorn_with_its_proxy_headers_off(tmp_path):
    assert _shell_commands("exec python3 -m uvicorn opti_oignon.api.app:app \\\n    --port 1\n"), "witness: a command"
    assert _python_starts('argv = [sys.executable, "-m", "uvicorn", "opti_oignon.api.app:app"]\n', "w.py")
    shell, python = [], []
    for path in sorted([*(REPO / "scripts").glob("*.sh"), *(REPO / "scripts").glob("*.py")]):
        shell += [(path.name, line) for line in _shell_commands(path.read_text(encoding="utf-8"))]
    for path in [REPO / "README.md", REPO / "frontend" / "README.md", *sorted((REPO / "docs").rglob("*.md"))]:
        shell += [(str(path.relative_to(REPO)), line) for line in _shell_commands(path.read_text(encoding="utf-8"))]
    for path in _allium_census.modules().values():
        text = Path(path).read_text(encoding="utf-8")
        python += [(str(Path(path).relative_to(REPO)), words) for words in _python_starts(text, str(path))]
    named = {where for where, _line in shell} | {where for where, _words in python}
    assert {"dev_backend.sh", "smoke_test.sh", "demo.py", "docs/getting-started/installation.md",
            "frontend/README.md", "opti_oignon/ui.py"} <= named, named
    assert [entry for entry in shell if "--no-proxy-headers" not in entry[1]] == [], shell
    assert [entry for entry in python if "--no-proxy-headers" not in entry[1]] == [], python
    # The image: uvicorn directly, its proxy headers off; the compose file starts it no other way.
    text = _read("Dockerfile.backend")
    words = _cmd(text)
    assert words[:3] == ["python", "-m", "uvicorn"] and _APP in words and "--no-proxy-headers" in words, words
    assert not {"--proxy-headers", "--forwarded-allow-ips"} & set(words), words
    assert "--no-proxy-headers" not in _cmd(text.replace(', "--no-proxy-headers"', "")), "witness: read where it is"
    compose = _read("docker-compose.yml")
    assert not [line for line in compose.splitlines() if line.strip().startswith(("command:", "entrypoint:"))], (
        "the compose file starts the backend another way")
    assert "FORWARDED_ALLOW_IPS" not in compose and "FORWARDED_ALLOW_IPS" not in text

    # ul14, run whole, the server's argv holding the flag.
    source_text = textwrap.dedent(inspect.getsource(ul.test_ul14_the_backend_script_runs_no_program_but_the_server))
    tree = ast.parse(source_text)
    old = ("argv: [-m] [uvicorn] [opti_oignon.api.app:app] [--host] [127.0.0.1] [--port] [8001] "
           "[--timeout-graceful-shutdown] [10]")
    replaced = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == old:
            node.value = old + " [--no-proxy-headers]"
            replaced += 1
    assert replaced == 1, replaced
    namespace = dict(vars(ul))
    exec(compile(ast.fix_missing_locations(tree), ul.__file__, "exec"), namespace)
    namespace["test_ul14_the_backend_script_runs_no_program_but_the_server"](tmp_path)


# --------------------------------------------------------------------------- VA7


def test_va7_the_texts_say_it():
    routes = _read("opti_oignon", "api", "routes_security.py")
    mode = _read("opti_oignon", "security_mode.py")
    switch = _read("opti_oignon", "search_killswitch.py")
    for name, text in (("routes_security", routes), ("security_mode", mode), ("search_killswitch", switch)):
        for old in ("DOM injection", "invisible to the LLM", "template renderer", "DOM only", "not in API JSON",
                    "NOT the JSON API", "NOT in this JSON response", "to the session that asked"):
            assert old not in text, (name, old)
    assert "to the account that asked for it only, any of its sessions" in _flat(routes)
    assert "serves it to the account that asked only" in _flat(mode)
    assert "to the account that asked only" in _flat(switch)
    gate = _flat(_read("opti_oignon", "api", "host_gate.py"))
    for said in ("from an address ``api.trusted_proxies`` of ``security.yaml`` lists",
                 "Uvicorn reads no forwarded header: every launcher of the tree starts it with ``--no-proxy-headers``",
                 "No other module reads ``X-Forwarded-For``."):
        assert said in gate, said
    for name, function, said in (("routes_auth.py", "_extract_client_ip", "X-Forwarded-For is never read here"),
                                 ("routes_files.py", "_get_client_ip", "never an X-Forwarded-For the client sent"),
                                 ("routes_users.py", "_get_client_ip", "never an X-Forwarded-For the client sent")):
        tree = ast.parse(_read("opti_oignon", "api", name))
        reader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
        assert said in _flat(ast.get_docstring(reader) or ""), (name, said)
    doc = _flat(_read("docs", "security", "authentication.md"))
    for said in ("Leaving Bulbe mode and re-enabling search are the owner's to ask",
                 "to the account that asked for the ceremony only, any of its sessions -- another account's gets a 403",
                 "ten minutes past its cooldown, after which its confirmation is refused (410) and nothing is pending",
                 "No route reads X-Forwarded-For, and uvicorn reads none: every launcher starts it with "
                 "`--no-proxy-headers`",
                 "A request that carries a forwarded header from a proxy that is not listed is one client, "
                 "`unlisted-proxy`, never a local one",
                 "A front that adds no forwarded header (nginx's bare `proxy_pass`, a TCP forwarder) is not seen",
                 "listing `127.0.0.1` takes the header of every client of this machine"):
        assert said in doc, said
    security = _flat(_read("SECURITY.md"))
    for said in ("no route and not uvicorn reads X-Forwarded-For, the Host gate does, from a listed proxy only",
                 "passes the scheme on (`X-Forwarded-Proto`, from an address `api.trusted_proxies` lists",
                 'api: trusted_proxies: ["127.0.0.1"]',
                 "is handed on as that one client too: never the loopback",
                 "`proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`",
                 "read only when it is an IP address written alone (no port, no name) and no loopback one",
                 "A front that adds no forwarded header at all (nginx's `proxy_pass` with no such line, a TCP "
                 "forwarder such as `ssh -L` or socat) is not seen",
                 "a newly listed proxy needs a restart"):
        assert said in security, said
    assert "from an address uvicorn trusts" not in security and "uvicorn trusts the addresses" not in security
    changelog = _flat(_read("CHANGELOG.md"))
    assert "A ceremony's visual code is served to the account that asked for it only" in changelog
    assert "A front that adds no forwarded header (nginx's bare `proxy_pass`, a TCP forwarder) is not seen" in changelog
    assert "A request forwarded by any other is one client" not in changelog, "the qualifier is dropped"
    assert "Contracts va1-va8 in `tests/test_code_to_whom_asked_and_the_sockets_address_contracts.py`; b2, tg4, op7" \
           " and ul14 give way to them." in changelog

    # op7, run whole, with the sentence that names the list.
    text = textwrap.dedent(inspect.getsource(op.test_op7_the_texts_say_the_rule))
    tree = ast.parse(text)
    old = "passes the scheme on (`X-Forwarded-Proto`, from an address uvicorn trusts)"
    replaced = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == old:
            node.value = "passes the scheme on (`X-Forwarded-Proto`, from an address `api.trusted_proxies` lists"
            replaced += 1
    assert replaced == 1, replaced
    namespace = dict(vars(op))
    exec(compile(ast.fix_missing_locations(tree), op.__file__, "exec"), namespace)
    namespace["test_op7_the_texts_say_the_rule"]()
    with pytest.raises(AssertionError, match="uvicorn trusts"):
        op.test_op7_the_texts_say_the_rule()


# --------------------------------------------------------------------------- VA8


def test_va8_a_request_an_unlisted_proxy_forwarded_is_never_a_local_one():
    from fastapi import HTTPException

    # The localhost-only routes (device certificates, remote access) refuse it.
    routes, restore = _routes()
    try:
        with pytest.raises(HTTPException) as refused:
            routes._require_localhost(seam._Request(host=UNLISTED))
        assert refused.value.status_code == 403
        assert routes._require_localhost(seam._Request(host="127.0.0.1")) is None, "witness: the loopback passes"
    finally:
        restore()
    # The remote check calls it remote: the remote headers and the hardened cookies.
    loaded, restore = isolate(targets={_GUARD: source("remote_session_guard.py")})
    try:
        guard = loaded[_GUARD]
        assert guard.is_remote_request(UNLISTED) is True and guard.is_remote_request("127.0.0.1") is False
    finally:
        restore()
    # Its cookies get no loopback exemption.
    ctx = seam._load()
    try:
        ctx.routes._get_security_jwt_config = lambda: {"cookie_secure": True}
        assert ctx.routes._detect_secure_context(seam._Request(host=UNLISTED)) is True
        assert ctx.routes._detect_secure_context(seam._Request(host="127.0.0.1")) is False, "witness: loopback"
    finally:
        ctx.restore()
    # Bulbe's local-only layer rejects it (on a route it does not let through by its path: the health route is),
    # said as a forwarded request: no bind-guard breach is recorded for it, as one is for a non-local address.
    gate, restore = _gate()
    unlisted = gate.UNLISTED_PROXY
    restore()
    events = []
    chain = types.ModuleType("opti_oignon.signed_audit_log")
    chain.chain_log = lambda **entry: events.append(entry)
    with bp._middleware("bulbe", seeded={"opti_oignon.signed_audit_log": chain}) as w:
        assert w.mw._FORWARDED_CLIENT == unlisted == UNLISTED, "the middleware's word is the gate's"
        status, body = bp._ask(w.app, "GET", "/api/searchx", client=UNLISTED)
        assert status == 403, (status, body)
        assert [(e["event_type"], e["severity"]) for e in events] == [("forwarded_request_rejected", "WARNING")]
        status, body = bp._ask(w.app, "GET", "/api/searchx", client="203.0.113.9")
        assert status == 403 and events[-1]["event_type"] == "non_local_request_rejected", events
        assert events[-1]["severity"] == "CRITICAL", events
        status, body = bp._ask(w.app, "GET", "/api/searchx", client="127.0.0.1")
        assert status == 200, ("witness: the loopback passes", status, body)
