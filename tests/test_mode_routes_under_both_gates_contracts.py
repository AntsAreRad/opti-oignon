#!/usr/bin/env python3
"""Contracts for the security mode's routes under the CSRF check and the Bulbe middleware, on every path.

The routes under ``/api/security/mode`` were exempt from the CSRF double-submit check and, in Bulbe, waved through by
the security-mode middleware before it read the mode, with the sign-in routes and the schema; any signed-in session
could cancel the owner's request to leave Bulbe. So in Bulbe a client of another machine, or a request a proxy
forwarded, could sign in and run the mode's ceremony, a Bearer token was taken there, and the session cookies, set by
the sign-in routes alone, never got Bulbe's SameSite=Strict. A websocket never met the middleware at all; the CSRF
check's exemption of the health check, compared as a string, exempted the benchmark routes below it.

  * MB1 -- the CSRF check covers the mode's routes: in Bulbe, a signed-in session's POST to each without the token is
    refused before its handler, with it served; the lists as written, compared by whole segments, the health check
    exactly; of the mounted routes that change state they exempt the sign-in routes only (the benchmarks below the
    health check ask the token).
  * MB2 -- cancelling a request to leave Bulbe is an owner's: a session of an account that is not an owner's gets 403
    and the request stands, the same request with the same code; an owner cancels it, a second owner too; a session
    that names no account gets 401, the single-user session with no account 403.
  * MB3 -- in Bulbe, and when the mode cannot be read, every path but the health check is refused to a client of
    another address and to a request a proxy forwarded, before its handler: each route of the mode, each sign-in
    route; this machine's client is served, with Bulbe's rate-limit headers; the health check is served to anyone;
    in Daily nothing changes.
  * MB4 -- in Bulbe, and when the mode cannot be read, a Bearer token is refused on the mode's routes, as on any
    route; the sign-in routes and the schema still take one (written, so that a client holding a token learns how
    to sign in); in Daily nothing changes.
  * MB5 -- the session cookies the real sign-in writer sets for this machine's client (Lax as shipped) leave a
    sign-in route Strict in Bulbe, every one of them, and Lax in Daily; the writer's own rule for another machine's
    client (Strict) is the witness.
  * MB6 -- the census: the prefixes the middleware spares, as written, cover the mounted sign-in routes and nothing
    else, none under ``/api/security``; the health check is spared exactly; every route of the mode that changes
    state depends on a session, and the cancel on the account's (``_session_account``), as its pendant, the
    re-enable's cancel, does.
  * MB7 -- the texts say it.
  * MB8 -- in Bulbe, and when the mode cannot be read, a websocket handshake from a client of another address or one
    a proxy forwarded (port 0, as the Host gate hands it on; a token in its query) is closed before it is accepted,
    its endpoint never run, the refusal audited as the HTTP one's; this machine's client, and one with no address,
    are accepted; in Daily every client is. Such a client's socket accepted in Daily is closed at its first message
    once Bulbe began, sent or received, its endpoint told the client went; this machine's goes on.
  * MB9 -- only a client Bulbe serves may enter it: the escalation answers 403 to any other, forced or not, and the
    emergency stop stops without its drop to Bulbe and says why, each refusal audited (``bulbe_entry_refused``);
    this machine's client, and one with no address, enter it; the routes' rule is the middleware's, address for
    address.
  * MB10 -- the pages say a refusal: a status refused with 403 installs no synthetic user (any other failure keeps
    the single-user fallback), the settings panel shows a refused cancel and a refused escalation in the server's
    words and keeps them shown, the stop button the refused drop to Bulbe.

Every module under contract is loaded from its file through the shared isolation window, the mode, the switch and
the audit stood in for; MB2 runs the real security mode manager on its own clock. Nothing reads or writes the
platform's data.

Local-only (the public distribution ships no tests).
"""

import ast
import asyncio
import json
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_auth_2fa_routes_seam_contracts as seam  # noqa: E402
import test_bulbe_paths_contracts as bp  # noqa: E402
import test_code_to_whom_asked_and_the_sockets_address_contracts as va  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_mb1_the_csrf_check_covers_the_modes_routes": 2.0,
    "test_mb2_cancelling_leaving_bulbe_is_an_owners": 1.5,
    "test_mb3_bulbe_serves_this_machines_clients_on_every_path_but_the_health_check": 3.0,
    "test_mb4_a_bearer_token_is_refused_on_the_modes_routes_in_bulbe": 2.0,
    "test_mb5_the_session_cookies_leave_a_sign_in_route_strict_in_bulbe": 2.0,
    "test_mb6_the_spared_prefixes_cover_the_sign_in_alone_and_each_change_asks_a_session": 2.0,
    "test_mb7_the_texts_say_it": 1.0,
    "test_mb8_a_websocket_from_a_client_bulbe_does_not_serve_is_closed_before_it_is_accepted": 2.0,
    "test_mb9_only_a_client_bulbe_serves_may_enter_it": 1.5,
    "test_mb10_the_pages_say_a_refusal": 1.0,
}

_CSRF = "opti_oignon.api.csrf_middleware"
_MODE_PREFIX = "/api/security/mode"
_REMOTE = "203.0.113.9"
_TOKEN = "the-token"
_SIGN_IN = [("POST", "/api/auth/login"), ("POST", "/api/auth/login/2fa"), ("POST", "/api/auth/login/2fa/enrol/begin"),
            ("POST", "/api/auth/login/2fa/enrol/complete"), ("POST", "/api/auth/refresh"),
            ("POST", "/api/auth/register"), ("GET", "/api/auth/status")]
_SOCKETS = ("/api/chat/stream", "/api/agent/stream", "/ws/coding/live")
# Clients the middleware does not serve in Bulbe, and clients it does (None: a request with no client).
_REFUSED = (_REMOTE, "unlisted-proxy", "::ffff:127.0.0.1", "127.0.0.2", "172.17.0.1", "localhost", "testclient")
_SERVED = ("127.0.0.1", "::1", None, "")


def _mounted():
    """Every route the package mounts, ``(method, path)``, read from its source by the Bulbe paths' census."""
    routes, unread = bp._route_census(bp._API)
    assert unread == [], unread
    assert len(routes) >= 500, len(routes)
    return routes


def _below(path, prefix):
    return path == prefix or path.startswith(prefix + "/")


def _mode_routes(routes):
    found = sorted({(method, path) for method, path in routes if _below(path, _MODE_PREFIX)})
    assert len(found) >= 7, found
    return found


def _concrete(path):
    """A path the router serves for a route template: each parameter given a value."""
    return re.sub(r"\{[^}]+\}", "p", path)


def _raise_unreadable():
    raise OSError("the mode cannot be read")


@contextmanager
def _gate(mode, routes, *, unreadable=False, cookies=(), sockets=(), echoes=(), pushers=()):
    """The real Bulbe middleware over a minimal application serving ``routes`` and the websockets ``sockets``; each
    handler it lets through is recorded in ``calls``, and answers with the ``cookies`` given as its Set-Cookie
    headers (a socket accepts, then closes). A socket of ``echoes`` sends back each text it receives, one of
    ``pushers`` sends three words, running ``hooks`` after the first; either records ``gone <code>`` when told the
    client went. The audit is kept."""
    from fastapi import FastAPI, WebSocket
    from starlette.responses import Response
    from starlette.websockets import WebSocketDisconnect

    targets = {}
    for name, path in ((bp._WG, source("web_gate.py")), (bp._EG, source("egress.py"))):
        if path.exists():
            targets[name] = path
    targets[bp._MW] = source("api", "security_mode_middleware.py")
    sm = bp._mode_module(mode)
    if unreadable:
        sm.get_current_mode = _raise_unreadable
    events = []
    chain = types.ModuleType("opti_oignon.signed_audit_log")
    chain.chain_log = lambda **entry: events.append(entry)
    loaded, restore = isolate(
        targets=targets,
        seeded={bp._SM: sm, bp._KS: bp._switch_module(False), "opti_oignon.signed_audit_log": chain},
        packages=("opti_oignon.api",),
    )
    try:
        mw = loaded[bp._MW]
        calls = []
        application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        def handler_for(name):
            def handler():
                calls.append(name)
                response = Response(content=json.dumps({"handler": name}), media_type="application/json")
                for value in cookies:
                    response.headers.append("set-cookie", value)
                return response

            return handler

        def socket_for(name):
            async def endpoint(websocket: WebSocket):
                calls.append(name)
                await websocket.accept()
                await websocket.close()

            return endpoint

        hooks = []

        def echo_for(name):
            async def endpoint(websocket: WebSocket):
                calls.append(name)
                await websocket.accept()
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        calls.append(f"gone {message.get('code')}")
                        return
                    await websocket.send_text(message.get("text") or "")

            return endpoint

        def pusher_for(name):
            async def endpoint(websocket: WebSocket):
                calls.append(name)
                await websocket.accept()
                try:
                    for word in ("one", "two", "three"):
                        await websocket.send_text(word)
                        for hook in hooks:
                            hook()
                        del hooks[:]
                except WebSocketDisconnect as gone:
                    calls.append(f"gone {gone.code}")

            return endpoint

        for method, path in routes:
            application.add_api_route(path, handler_for(f"{method} {path}"), methods=[method])
        for path in sockets:
            application.add_api_websocket_route(path, socket_for(f"WS {path}"))
        for path in echoes:
            application.add_api_websocket_route(path, echo_for(f"ECHO {path}"))
        for path in pushers:
            application.add_api_websocket_route(path, pusher_for(f"PUSH {path}"))
        application.add_middleware(mw.SecurityModeMiddleware)
        yield SimpleNamespace(mw=mw, app=application, calls=calls, sm=sm, events=events, hooks=hooks)
    finally:
        restore()


def _answer(application, method, path, *, client=bp._LOCAL, headers=()):
    """One ASGI request with a session cookie: ``(status, body, response headers)``; ``client`` None sends none."""
    sent = []
    full = _concrete(path)
    scope = {
        "type": "http", "method": method, "path": full, "raw_path": full.encode("ascii"), "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:8001"), (b"cookie", b"oo_session=local"),
                    *((key.lower().encode("ascii"), value.encode("latin-1")) for key, value in headers)],
        "client": None if client is None else (client, 50000), "server": ("127.0.0.1", 8001), "scheme": "http",
        "http_version": "1.1", "root_path": "",
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(application(scope, receive, send))
    start = sent[0]
    body = b"".join(message.get("body", b"") for message in sent[1:])
    try:
        body = json.loads(body)
    except ValueError:
        pass
    return start["status"], body, [(key.lower().decode("latin-1"), value.decode("latin-1"))
                                   for key, value in start["headers"]]


def _ask(application, method, path, *, client=bp._LOCAL, headers=()):
    """One ASGI request with a session cookie: ``(status, body, set-cookie values)``."""
    status, body, answered = _answer(application, method, path, client=client, headers=headers)
    return status, body, [value for key, value in answered if key == "set-cookie"]


def _knock(application, path, *, client, port=50000, query=b"", script=()):
    """One websocket handshake, then ``script``: each text sent to the socket in turn, each callable run before the
    next one is read; the messages the application sent (``client`` None sends none)."""
    sent = []
    scope = {
        "type": "websocket", "path": path, "raw_path": path.encode("ascii"), "query_string": query,
        "headers": [(b"host", b"127.0.0.1:8001"), (b"origin", b"http://127.0.0.1:8001")],
        "client": None if client is None else (client, port), "server": ("127.0.0.1", 8001), "scheme": "ws",
        "http_version": "1.1", "root_path": "", "subprotocols": [],
    }
    incoming = [{"type": "websocket.connect"}, *script]

    async def receive():
        while incoming and callable(incoming[0]):
            incoming.pop(0)()
        if not incoming:
            return {"type": "websocket.disconnect", "code": 1000}
        item = incoming.pop(0)
        return item if isinstance(item, dict) else {"type": "websocket.receive", "text": item}

    async def send(message):
        sent.append(message)

    asyncio.run(application(scope, receive, send))
    return sent


# --------------------------------------------------------------------------- MB1


def test_mb1_the_csrf_check_covers_the_modes_routes():
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse

    routes = _mounted()
    changing = [(method, path) for method, path in _mode_routes(routes) if method in ("POST", "PUT", "DELETE", "PATCH")]
    assert {path for _method, path in changing} >= {
        _MODE_PREFIX, _MODE_PREFIX + "/request-downgrade", _MODE_PREFIX + "/confirm-downgrade",
        _MODE_PREFIX + "/cancel-downgrade"}, changing
    auth = types.ModuleType("opti_oignon.auth")
    auth.auth_manager = SimpleNamespace(single_user_mode=False)
    loaded, restore = isolate(targets={_CSRF: source("api", "csrf_middleware.py")},
                              seeded={"opti_oignon.auth": auth, "opti_oignon.security_mode": bp._mode_module("bulbe")},
                              packages=("opti_oignon.api",))
    try:
        csrf = loaded[_CSRF]
        middleware = csrf.CSRFMiddleware(app=None)
        assert middleware._cookie_mode and middleware._csrf_enabled, "witness: the shipped configuration checks"

        def send(method, path, headers):
            full = _concrete(path)
            scope = {"type": "http", "method": method, "path": full, "raw_path": full.encode("ascii"),
                     "query_string": b"", "scheme": "http", "server": ("127.0.0.1", 8001),
                     "client": ("127.0.0.1", 50000), "root_path": "",
                     "headers": [(k.lower().encode("ascii"), v.encode("ascii")) for k, v in headers]}
            served = []

            async def call_next(request):
                served.append(True)
                return PlainTextResponse("served")

            response = asyncio.run(middleware.dispatch(Request(scope), call_next))
            return response.status_code, bool(served)

        session = ("Cookie", f"oo_access_token=good; oo_csrf_token={_TOKEN}")
        for method, path in changing:
            assert send(method, path, [session]) == (403, False), (method, path, "served without the token")
            assert send(method, path, [session, ("X-CSRF-Token", "another")]) == (403, False), (method, path)
            assert send(method, path, [session, ("X-CSRF-Token", _TOKEN)]) == (200, True), (method, path)
        # The sign-in stays exempt (witness: the list still exempts what it names).
        assert send("POST", "/api/auth/login", [session]) == (200, True)
        # The lists as written, and what they exempt of the mounted routes that change state, by the check's own
        # reading: the sign-in routes, nothing else -- not the mode's, not the benchmarks below the health check.
        assert csrf._CSRF_EXEMPT_PREFIXES == ("/api/auth/login", "/api/auth/register", "/api/auth/status",
                                             "/api/auth/refresh", "/docs", "/openapi.json", "/redoc")
        assert csrf._CSRF_EXEMPT_EXACT == ("/api/health",)
        exempted = sorted({(method, path) for method, path in routes
                           if method in ("POST", "PUT", "DELETE", "PATCH") and csrf._exempt(_concrete(path))})
        assert exempted == sorted(route for route in _SIGN_IN if route[0] == "POST"), exempted
        for path in ("/api/health/benchmarks", "/api/health/benchmarks/{name}"):
            assert send("POST", path, [session]) == (403, False), (path, "served without the token")
            assert send("POST", path, [session, ("X-CSRF-Token", _TOKEN)]) == (200, True), path
        assert send("POST", "/api/health", [session]) == (200, True), "the health check itself is exempt"
        # Whole segments: a name that merely begins like an exempt one is not exempt.
        assert csrf._exempt("/api/auth/login/2fa") and csrf._exempt("/docs/oauth2-redirect")
        for path in ("/api/auth/loginx", "/api/healthx", "/docsx", "/api/health/dashboard"):
            assert not csrf._exempt(path), path
    finally:
        restore()


# --------------------------------------------------------------------------- MB2


def test_mb2_cancelling_leaving_bulbe_is_an_owners():
    from fastapi import HTTPException

    clock = [1_000_000.0]
    _sm, manager, restore_mode = va._mode_manager(clock)
    try:
        routes, restore = va._routes(manager=manager)
        try:
            def cancel(current):
                return va._status(routes.cancel_mode_downgrade(routes._session_account(current)))

            def ask():
                return va._status(routes.request_mode_downgrade(
                    va._session(va.OWNER), routes._session_account(va._session(va.OWNER))))

            status, asked = ask()
            assert status == 200 and asked["pending"] is True, (status, asked)
            held = manager._pending_downgrade
            code = held.visual_code
            # A session of an account that is not an owner's cancels nothing: the same request, the same code.
            assert routes._session_account(va._session(va.USER)).owner is False
            status, detail = cancel(va._session(va.USER))
            assert status == 403 and "owner" in str(detail) and "stands" in str(detail), (status, detail)
            assert manager._pending_downgrade is held and held.visual_code == code, "a user's session cancelled it"
            assert manager.visual_code_for("uid-resident") == ("ok", code)
            # A session that names no account, and the single-user session with no account, cancel nothing either.
            for current, refused in (({}, 401), ({"sub": ""}, 401), ({"sub": "local"}, 403),
                                     ({"sub": "uid-gone"}, 401)):
                with pytest.raises(HTTPException) as caught:
                    routes._session_account(current)
                assert caught.value.status_code == refused, (current, caught.value.status_code)
            assert manager._pending_downgrade is held
            # The owner cancels it.
            status, answer = cancel(va._session(va.OWNER))
            assert status == 200 and answer.get("success") is True, (status, answer)
            assert manager._pending_downgrade is None
            assert manager.visual_code_for("uid-resident") == ("none", None)
            status, answer = cancel(va._session(va.OWNER))
            assert status == 200 and "No pending" in answer.get("message", ""), (status, answer)
            # A second owner is an owner: cancelling is an owner's, as asking is.
            assert routes._session_account(va._session(va.SECOND_OWNER)).owner is True
            status, asked = ask()
            assert status == 200 and manager._pending_downgrade is not None, (status, asked)
            status, answer = cancel(va._session(va.SECOND_OWNER))
            assert status == 200 and answer.get("success") is True and manager._pending_downgrade is None, answer
        finally:
            restore()
    finally:
        restore_mode()


# --------------------------------------------------------------------------- MB3


def test_mb3_bulbe_serves_this_machines_clients_on_every_path_but_the_health_check():
    routes = _mounted()
    mode_routes = _mode_routes(routes)
    with _gate("bulbe", []) as probe:
        spared = probe.mw._ALWAYS_ALLOWED_PREFIXES
        forwarded = probe.mw._FORWARDED_CLIENT
    sign_in = sorted({(method, path) for method, path in routes
                      if any(_below(path, prefix) for prefix in spared)})
    assert set(_SIGN_IN) <= set(sign_in), sign_in
    schema = [("GET", "/docs"), ("GET", "/openapi.json")]
    checked = mode_routes + sign_in + schema
    health = ("GET", "/api/health")
    for mode, unreadable in (("bulbe", False), ("daily", True)):
        with _gate(mode, checked + [health], unreadable=unreadable) as w:
            for method, path in checked:
                for client, event in ((_REMOTE, "non_local_request_rejected"),
                                      (forwarded, "forwarded_request_rejected")):
                    del w.calls[:]
                    status, body, _cookies = _ask(w.app, method, path, client=client)
                    assert status == 403 and w.calls == [], (mode, unreadable, method, path, client, status, body)
                    assert body.get("restriction") == "non_local_rejected", (method, path, body)
                    assert w.events and w.events[-1]["event_type"] == event, (method, path, w.events[-1:])
                    assert w.events[-1]["path"] == _concrete(path)
                for client in ("127.0.0.1", "::1", None):
                    del w.calls[:]
                    status, body, answered = _answer(w.app, method, path, client=client)
                    assert (status, w.calls) == (200, [f"{method} {path}"]), (mode, method, path, client, status)
                    named = dict(answered)
                    assert (named.get("x-ratelimit-mode"), named.get("x-ratelimit-maxattempts"),
                            named.get("x-ratelimit-window")) == (("bulbe", "3", "300") if path.startswith("/api/")
                                                                 else (None, None, None)), (path, answered)
            for client in (_REMOTE, forwarded, bp._LOCAL):
                del w.calls[:]
                status, body, _cookies = _ask(w.app, *health, client=client)
                assert (status, w.calls) == (200, ["GET /api/health"]), (client, status, body)
    # Daily: every client is served, the change is Bulbe's alone.
    with _gate("daily", checked) as w:
        for method, path in checked:
            for client in (_REMOTE, forwarded):
                del w.calls[:]
                status, body, _cookies = _ask(w.app, method, path, client=client)
                assert (status, w.calls) == (200, [f"{method} {path}"]), (method, path, client, status, body)


# --------------------------------------------------------------------------- MB4


def test_mb4_a_bearer_token_is_refused_on_the_modes_routes_in_bulbe():
    routes = _mounted()
    mode_routes = _mode_routes(routes)
    sign_in = _SIGN_IN + [("GET", "/docs"), ("GET", "/openapi.json")]
    bearer = [("Authorization", "Bearer a-token")]
    for mode, unreadable in (("bulbe", False), ("daily", True)):
        with _gate(mode, mode_routes + sign_in, unreadable=unreadable) as w:
            for method, path in mode_routes:
                del w.calls[:]
                status, body, _cookies = _ask(w.app, method, path, headers=bearer)
                assert status == 403 and w.calls == [], (mode, unreadable, method, path, status, body)
                assert body.get("restriction") == "bearer_rejected", (method, path, body)
                del w.calls[:]
                status, body, _cookies = _ask(w.app, method, path)
                assert (status, w.calls) == (200, [f"{method} {path}"]), ("witness: a cookie session", method, path)
            for method, path in sign_in:
                del w.calls[:]
                status, body, _cookies = _ask(w.app, method, path, headers=bearer)
                assert (status, w.calls) == (200, [f"{method} {path}"]), (mode, method, path, status, body)
    with _gate("daily", mode_routes) as w:
        for method, path in mode_routes:
            del w.calls[:]
            status, body, _cookies = _ask(w.app, method, path, headers=bearer)
            assert (status, w.calls) == (200, [f"{method} {path}"]), (method, path, status, body)


# --------------------------------------------------------------------------- MB5


def _written_cookies(host):
    """The Set-Cookie values the real sign-in writer (``routes_auth._set_auth_cookies``) sets for a client at
    ``host``, on the shipped configuration's defaults, the real remote check beside it (the second factor out of
    reach, as the seam's degraded window has it)."""
    from starlette.responses import Response

    loaded, restore = isolate(
        targets={seam._ROUTES: source("api", "routes_auth.py"),
                 "opti_oignon.remote_session_guard": source("remote_session_guard.py")},
        blocked=("ollama", seam._TFA),
        seeded={"opti_oignon.security_mode": seam._security_mode_seed(seam._ModeState(bulbe=False)),
                "opti_oignon.api.deps": seam._deps_seed(seam._AuthManagerStandIn(single_user=False))},
        packages=("opti_oignon.api",),
    )
    try:
        routes = loaded[seam._ROUTES]
        routes._get_security_jwt_config = lambda: {}
        response = Response()
        routes._set_auth_cookies(seam._Request(host=host), response, "an-access-token", "a-refresh-token")
        return [value.decode("latin-1") for key, value in response.raw_headers if key == b"set-cookie"]
    finally:
        restore()


def _samesite(cookie):
    found = re.findall(r"(?i)samesite=([a-z]+)", cookie)
    assert len(found) == 1, cookie
    return found[0].lower()


def test_mb5_the_session_cookies_leave_a_sign_in_route_strict_in_bulbe():
    local = _written_cookies(bp._LOCAL)
    names = sorted(cookie.split("=", 1)[0] for cookie in local)
    assert names == ["oo_access_token", "oo_csrf_token", "oo_refresh_token"], local
    assert {_samesite(cookie) for cookie in local} == {"lax"}, ("witness: Lax for this machine's client", local)
    remote = _written_cookies(_REMOTE)
    assert {_samesite(cookie) for cookie in remote} == {"strict"}, ("witness: the writer's own remote rule", remote)
    sign_in = [route for route in _SIGN_IN if route[0] == "POST"]
    for mode, unreadable, expected in (("bulbe", False, "strict"), ("daily", True, "strict"),
                                       ("daily", False, "lax")):
        with _gate(mode, sign_in, unreadable=unreadable, cookies=local) as w:
            for method, path in sign_in:
                status, body, cookies = _ask(w.app, method, path)
                assert status == 200, (mode, path, status, body)
                assert len(cookies) == len(local), (mode, path, cookies)
                assert {_samesite(cookie) for cookie in cookies} == {expected}, (mode, unreadable, path, cookies)
                assert sorted(cookie.split("=", 1)[0] for cookie in cookies) == names, cookies


# --------------------------------------------------------------------------- MB6


def _decorated(tree):
    """``{function name: (method, path)}`` of the routes a module's ``router`` decorates."""
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and getattr(decorator.func.value, "id", "") == "router" and decorator.args
                        and isinstance(decorator.args[0], ast.Constant)):
                    found[node.name] = (decorator.func.attr.upper(), decorator.args[0].value)
    return found


def _depends_on(function):
    """The names of the callables a function's parameters ``Depends`` on."""
    names = set()
    arguments = function.args
    for default in [*arguments.defaults, *arguments.kw_defaults]:
        if (isinstance(default, ast.Call) and getattr(default.func, "id", "") == "Depends" and default.args
                and isinstance(default.args[0], ast.Name)):
            names.add(default.args[0].id)
    return names


def test_mb6_the_spared_prefixes_cover_the_sign_in_alone_and_each_change_asks_a_session():
    routes = _mounted()
    with _gate("bulbe", []) as w:
        spared = w.mw._ALWAYS_ALLOWED_PREFIXES
        exact = w.mw._ALWAYS_ALLOWED_EXACT
        matches = w.mw._matches
    assert exact == ("/api/health",), exact
    assert spared == ("/api/auth/login", "/api/auth/register", "/api/auth/status", "/api/auth/refresh", "/docs",
                      "/openapi.json"), spared
    covered = sorted({(method, path) for method, path in routes for prefix in spared if matches(path, prefix)})
    assert covered == sorted(_SIGN_IN + [("GET", "/docs"), ("GET", "/openapi.json")]), covered
    # Every route of the mode that changes state depends on a session; the cancel on the account's, as its pendant.
    text = source("api", "routes_security.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    decorated = _decorated(tree)
    changing = {name: route for name, route in decorated.items()
                if _below(route[1], "/mode") and route[0] in ("POST", "PUT", "DELETE")}
    assert set(changing) >= {"set_security_mode", "request_mode_downgrade", "confirm_mode_downgrade",
                             "cancel_mode_downgrade"}, changing
    for name in changing:
        assert _depends_on(functions[name]) & {"_get_current_user", "_session_account"}, (name, "asks no session")
    assert "_session_account" in _depends_on(functions["cancel_mode_downgrade"])
    assert "_session_account" in _depends_on(functions["cancel_search_reenable"]), "witness: the pendant"
    cancel = ast.get_source_segment(text, functions["cancel_mode_downgrade"])
    assert "account.owner" in cancel and "cancel_downgrade()" in cancel, cancel


# --------------------------------------------------------------------------- MB7


def _flat(text):
    return " ".join(text.split())


def test_mb7_the_texts_say_it():
    security = _flat(REPO.joinpath("SECURITY.md").read_text(encoding="utf-8"))
    for said in ("state-changing endpoints of a signed-in browser session, the security mode's included",
                 "sign-in, registration, the session's status and refresh, the health check itself (not the routes "
                 "below it), the schema",
                 "Bulbe's local-only check rejects it (on every path but the health check, its sockets included)",
                 "This machine's clients only: on every path but the health check, the sign-in and the security "
                 "mode's own routes included, a client of another address or a request a proxy forwarded is refused, "
                 "and its websocket closed before it is accepted",
                 "and its websocket closed before it is accepted (one it opened in Daily, at its first message once "
                 "Bulbe began)",
                 "Such a client may not enter Bulbe either -- the escalation answers 403, and the emergency stop stops "
                 "without its drop to Bulbe, each refusal audited --, so that Bulbe is entered from where it can be "
                 "left",
                 "A Bearer token is refused on every HTTP path but the health check, the sign-in routes and the "
                 "schema (a websocket may still carry its token in its query)",
                 "every cookie the API sets is SameSite=Strict, the session's included",
                 "SameSite=Strict in Bulbe mode and for another machine's client",
                 "a backend in a container sees every client at the bridge's address: from a browser, Bulbe can be "
                 "neither entered nor used there"):
        assert said in security, said
    assert "SameSite=Strict)" not in security, "a Strict said with no condition"
    auth = _flat(REPO.joinpath("docs", "security", "authentication.md").read_text(encoding="utf-8"))
    for said in ("are the owner's to ask, and to cancel (a session of an account that is not an owner's gets a 403, "
                 "and the request stands)",
                 "Strict in Bulbe mode and for a client of another machine",
                 "the value `jwt.cookie_samesite` of `security.yaml` gives (Lax as shipped)"):
        assert said in auth, said
    assert "**SameSite=Strict**" not in auth and "another account's session gets a 403, and" not in auth
    layers = _flat(REPO.joinpath("docs", "architecture", "security-layers.md").read_text(encoding="utf-8"))
    assert ("SameSite=Strict in Bulbe mode and for another machine's client (`jwt.cookie_samesite`, Lax as shipped, "
            "for this machine's browser in Daily mode)") in layers
    assert "SameSite=Strict flags" not in layers
    bulbe = _flat(REPO.joinpath("docs", "security", "bulbe-mode.md").read_text(encoding="utf-8"))
    for said in ("**This machine's clients only** -- on every path but `/api/health`, the sign-in and the security "
                 "mode's routes included, the API refuses a client of another address or a request a proxy forwarded",
                 "closes such a client's websocket before it is accepted, or, for one it opened in Daily, at its first "
                 "message once Bulbe began",
                 "Only a client Bulbe serves may enter it, so that it is entered from where it can be left: from "
                 "another address, or through a proxy, the escalation answers 403, and the emergency stop stops "
                 "without its drop to Bulbe."):
        assert said in bulbe, said
    changelog = _flat(REPO.joinpath("CHANGELOG.md").read_text(encoding="utf-8"))
    entry = changelog.split("- [SECURITY] The security mode's routes pass both gates", 1)
    assert len(entry) == 2, "no entry"
    entry = entry[1].split("- [SECURITY]", 1)[0]
    for said in ("cancelling is an owner's (403 for a session of an account that is not an owner's",
                 "on every path but `/api/health`", "closes such a client's websocket before it is accepted",
                 "Strict, the session's included", "the escalation answers 403, and the emergency stop stops without "
                 "its drop to Bulbe and says so", "The CSRF exemptions are compared by whole segments, the health "
                 "check exactly", "The sign-in routes and the schema still take a Bearer token in Bulbe",
                 "(one it opened in Daily, at its first message once Bulbe began)",
                 "each refusal audited (`bulbe_entry_refused`)",
                 "Contracts mb1-mb10 in `tests/test_mode_routes_under_both_gates_contracts.py`"):
        assert said in entry, said
    checklist = _flat(REPO.joinpath("docs", "getting-started", "post-install-checklist.md").read_text(encoding="utf-8"))
    assert ("requires a single authenticated request, from this machine: Bulbe serves this machine's clients only, "
            "so another address, or a request through a proxy, may not enter it") in checklist
    agent = _flat(source("api", "routes_agent.py").read_text(encoding="utf-8"))
    assert "no middleware sees a socket" not in agent and "the HTTP middlewares never see a socket" in agent
    middleware = _flat(source("api", "security_mode_middleware.py").read_text(encoding="utf-8"))
    assert "the Bearer refusal alone" in middleware and "but a revoked certificate" in middleware
    assert '"/api/security/mode"' not in middleware
    assert '"/api/security/mode"' not in source("api", "csrf_middleware.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- MB8


def test_mb8_a_websocket_from_a_client_bulbe_does_not_serve_is_closed_before_it_is_accepted():
    for mode, unreadable in (("bulbe", False), ("daily", True)):
        with _gate(mode, [], unreadable=unreadable, sockets=_SOCKETS) as w:
            for path in _SOCKETS:
                for client, event, port, query in (
                        (_REMOTE, "non_local_request_rejected", 50000, b""),
                        (w.mw._FORWARDED_CLIENT, "forwarded_request_rejected", 0, b""),
                        (_REMOTE, "non_local_request_rejected", 0, b"token=a-token")):
                    del w.calls[:]
                    sent = _knock(w.app, path, client=client, port=port, query=query)
                    assert sent == [{"type": "websocket.close", "code": 1008}], (mode, unreadable, path, client, sent)
                    assert w.calls == [], (path, client, "the endpoint ran")
                    assert w.events and w.events[-1]["event_type"] == event and w.events[-1]["path"] == path
                for client in ("127.0.0.1", "::1", None):
                    del w.calls[:]
                    sent = _knock(w.app, path, client=client)
                    assert sent and sent[0]["type"] == "websocket.accept", (mode, path, client, sent)
                    assert w.calls == [f"WS {path}"], (path, client)
    with _gate("daily", [], sockets=_SOCKETS) as w:
        for path in _SOCKETS:
            for client in (_REMOTE, w.mw._FORWARDED_CLIENT, "127.0.0.1"):
                del w.calls[:]
                sent = _knock(w.app, path, client=client)
                assert sent and sent[0]["type"] == "websocket.accept", (path, client, sent)
                assert w.calls == [f"WS {path}"]
                assert not w.events, w.events
    # A socket accepted in Daily: Bulbe begins while it is open. A client Bulbe does not serve is closed at its next
    # message, received or sent, and its endpoint told it went; this machine's client goes on.
    for unreadable in (False, True):
        with _gate("daily", [], echoes=("/ws/echo",), pushers=("/ws/push",)) as w:
            def bulbe():
                if unreadable:
                    w.sm.get_current_mode = _raise_unreadable
                else:
                    w.sm.mode = "bulbe"

            for client, event in ((_REMOTE, "non_local_request_rejected"),
                                  (w.mw._FORWARDED_CLIENT, "forwarded_request_rejected")):
                w.sm.mode, w.sm.get_current_mode = "daily", (lambda: w.sm.mode)
                del w.calls[:], w.events[:]
                sent = _knock(w.app, "/ws/echo", client=client, port=0, script=("a", bulbe, "b", "c"))
                assert [m["type"] for m in sent] == ["websocket.accept", "websocket.send", "websocket.close"], sent
                assert sent[1].get("text") == "a" and sent[2]["code"] == 1008, sent
                assert w.calls == ["ECHO /ws/echo", "gone 1008"], (client, w.calls)
                assert [e["event_type"] for e in w.events] == [event], w.events
                w.sm.mode, w.sm.get_current_mode = "daily", (lambda: w.sm.mode)
                del w.calls[:], w.events[:]
                w.hooks.append(bulbe)
                sent = _knock(w.app, "/ws/push", client=client)
                assert [m["type"] for m in sent] == ["websocket.accept", "websocket.send", "websocket.close"], sent
                assert sent[1].get("text") == "one" and sent[2]["code"] == 1008, sent
                assert w.calls == ["PUSH /ws/push", "gone 1008"], (client, w.calls)
                assert [e["event_type"] for e in w.events] == [event], w.events
            w.sm.mode, w.sm.get_current_mode = "daily", (lambda: w.sm.mode)
            del w.calls[:], w.events[:]
            sent = _knock(w.app, "/ws/echo", client="127.0.0.1", script=("a", bulbe, "b"))
            assert [m.get("text") for m in sent if m["type"] == "websocket.send"] == ["a", "b"], sent
            assert not w.events, w.events


# --------------------------------------------------------------------------- MB9


def test_mb9_only_a_client_bulbe_serves_may_enter_it():
    escalated, stopped, audited = [], [], []
    manager = SimpleNamespace(escalate_to_bulbe=lambda user_id, force=False: escalated.append(user_id)
                              or {"success": True, "mode": "bulbe"})
    routes, restore = va._routes(manager=manager)
    try:
        routes.SECURITY_MODE_AVAILABLE = True
        routes.EMERGENCY_STOP_AVAILABLE = True
        routes._emergency_stop = SimpleNamespace(
            stop=lambda user_id="", drop_to_bulbe=False: stopped.append(drop_to_bulbe)
            or {"stopped": True, "drop_to_bulbe": drop_to_bulbe, "steps": [], "failed_steps": []})
        sys.modules["opti_oignon.security_mode"]._audit_log = (
            lambda event, severity="INFO", **details: audited.append((event, details.get("via"),
                                                                      details.get("client_host"))))

        def request(host):
            return SimpleNamespace(client=None if host is None else SimpleNamespace(host=host))

        owner = va._session(va.OWNER)
        for host in _REFUSED:
            for force in (False, True):
                del escalated[:], audited[:]
                status, detail = va._status(routes.set_security_mode(
                    routes.SecurityModeChangeRequest(mode="bulbe", force=force), owner, request(host)))
                assert status == 403 and "this machine" in str(detail) and escalated == [], (host, force, detail)
                assert audited == [("bulbe_entry_refused", "escalation", host)], (host, audited)
            del stopped[:], audited[:]
            status, answer = va._status(routes.engage_emergency_stop(
                routes.EmergencyStopRequest(drop_to_bulbe=True), owner, request(host)))
            assert status == 200 and answer["stopped"] is True and stopped == [False], (host, answer, stopped)
            assert "this machine" in answer.get("bulbe_refused", ""), answer
            assert audited == [("bulbe_entry_refused", "emergency_stop", host)], (host, audited)
            del stopped[:], audited[:]
            status, answer = va._status(routes.engage_emergency_stop(routes.EmergencyStopRequest(), owner,
                                                                     request(host)))
            assert status == 200 and stopped == [False] and "bulbe_refused" not in answer, (host, answer)
            assert audited == [], audited
        for host in _SERVED:
            del escalated[:], stopped[:], audited[:]
            status, answer = va._status(routes.set_security_mode(
                routes.SecurityModeChangeRequest(mode="bulbe"), owner, request(host)))
            assert status == 200 and escalated == ["uid-resident"], (host, status, answer)
            status, answer = va._status(routes.engage_emergency_stop(
                routes.EmergencyStopRequest(drop_to_bulbe=True), owner, request(host)))
            assert status == 200 and stopped == [True] and "bulbe_refused" not in answer, (host, answer)
            assert audited == [], audited
        # Called with no request at all (no client), as a test or another route would: served.
        del escalated[:]
        assert va._status(routes.set_security_mode(routes.SecurityModeChangeRequest(mode="bulbe"), owner))[0] == 200
        assert escalated == ["uid-resident"]
        rule = routes._from_this_machine
    finally:
        restore()
    # The routes' rule is the middleware's, address for address.
    with _gate("bulbe", []) as w:
        for host in _REFUSED + _SERVED:
            served = not w.mw._refused_in_bulbe(host, "/probe")
            assert rule(SimpleNamespace(client=None if host is None else SimpleNamespace(host=host))) == served, host
            assert served == (host in _SERVED), host


# --------------------------------------------------------------------------- MB10


def _body(text, opening):
    """The body of the function that ``opening`` begins, up to its closing brace at the first indentation."""
    return text.split(opening, 1)[1].split("\n\t}\n", 1)[0]


def test_mb10_the_pages_say_a_refusal():
    front = REPO / "frontend" / "src" / "lib"
    auth = (front / "stores" / "auth.ts").read_text(encoding="utf-8")
    status_call = auth.split("status = await authApi.getAuthStatus();", 1)
    assert len(status_call) == 2, "the status is no longer fetched alone"
    refused = status_call[1].split("authStatus.set(status);", 1)[0]
    assert "if (httpRefusal(err)?.status !== 403) throw err;" in refused, refused
    assert "single_user_mode: false" in refused and "currentUser.set(null);" in refused, refused
    assert not re.search(r"role:\s*['\"`]admin", refused), refused
    assert "import { httpRefusal } from '$lib/switches/serverSwitch';" in auth
    # The single-user fallback is still the network failure's (and any other's) after the status.
    assert re.search(r"Backend unreachable, assume single-user mode[\s\S]*role:\s*'admin'", auth.split(refused, 1)[1])
    panel = (front / "components" / "settings" / "SecurityModePanel.svelte").read_text(encoding="utf-8")
    cancel = _body(panel, "async function handleCancelDowngrade()")
    caught = cancel.split("} catch (e: any) {", 1)
    assert len(caught) == 2, cancel
    refusal = caught[1].split("return;", 1)
    assert len(refusal) == 2 and "resetDowngradeState();" in refusal[1], cancel
    assert re.findall(r"\berror = [^;]*;", refusal[0]) == [
        "error = httpRefusal(e)?.detail || e.message || 'Cancelling the downgrade failed';"], refusal[0]
    assert "/* ignore */" not in cancel
    escalate = _body(panel, "async function handleEscalate()")
    caught = escalate.split("} catch (e: any) {", 1)[1].split("} finally {", 1)[0]
    assert re.findall(r"\berror = [^;]*;", caught) == ["error = httpRefusal(e)?.detail || e.message || "
                                                         "'Escalation failed';"], caught
    stop = (front / "stores" / "estop.ts").read_text(encoding="utf-8")
    engage = stop.split("export async function engageStop", 1)[1].split("\n}\n", 1)[0]
    assert ("error: [stepErrors('Stopped', result), result.bulbe_refused ?? ''].filter(Boolean).join('. ')"
            in engage), engage
    assert "bulbe_refused?: string;" in (front / "api" / "estop.ts").read_text(encoding="utf-8")
