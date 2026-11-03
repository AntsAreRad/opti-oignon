"""Contracts for the origin rule: a page of another port of this machine is another site.

The Host gate took a change from any page whose ``Origin`` had a loopback name, on any port; CORS answered any
loopback origin on any port, with credentials; and a websocket handshake was asked only its Origin's host name, by
``authenticate_websocket``. A dev server or another local app (``localhost:3000``), or a page of another scheme
(``tauri://localhost``), could change anything in single-user Daily, read every answer, and open every socket -- the
synthetic admin's stream. Now ``api/host_gate.site_allowed`` takes a page as this site's when its Origin has the name
and the port of the request's Host header (the scheme's default when none is written; ``https`` for a name other than
a loopback one), or when it is exactly one of the origins ``cors.origins`` of ``security.yaml`` lists
(``listed_origins``); the gate refuses another site's change, as before, and now closes its websocket handshake before
the app; CORS answers the listed origins alone (``cors_settings``).

  * OP1 -- ANOTHER PORT, NAME OR SCHEME CHANGES NOTHING: a POST from a page of another local port, another loopback
    name on the same port, the scheme's default port against a written one, another scheme -- the request's own
    scheme against the page's, both on their default ports (round 1) --, a listed name over ``http``, a listed
    origin's neighbour: 403 before the app, the fixed words.
  * OP2 -- THE PAGES OF THIS SITE ARE ANSWERED: through the dev proxy, the API's own address, ``[::1]``, a listed name
    over ``https`` with or without its port, a default port written in the Host, capitals, a listed origin across
    ports or names, ``Sec-Fetch-Site`` ``same-origin`` or ``none``, the CLI; a GET, a HEAD, an OPTIONS from another
    port still pass (what xs2 held besides the ports).
  * OP3 -- ANOTHER SITE'S SOCKET IS CLOSED AT THE GATE: a handshake from another port, name, scheme or site, ``null``,
    two Origins, or no Origin and another site's ``Sec-Fetch-Site`` is closed 1008 before the app; the same origin, a
    listed one and a client with no Origin pass; the first refusal is said once, without the origin.
  * OP4 -- THE LIST IS READ WHOLE OR NOT AT ALL: ``cors.origins`` as a list or a string, as a browser spells it, the
    variable when the file lists none, ``*``; one entry that is not an exact origin (a path, a wildcard, a listed name
    over ``http``, capitals, a port out of range, an address spelled otherwise) refuses the list and says so; an
    absent file, key or value lists none in silence; the suites drop the variable a shell lists (round 1).
  * OP5 -- CORS ANSWERS THE LISTED ORIGINS ALONE: the app's CORS middleware (``ListedOriginsCORS``, Starlette's)
    built from ``cors_settings`` answers a listed origin with credentials and a page of another port no CORS header
    at all, simple or preflight (Starlette's sent it Allow-Credentials, round 2); ``*`` answers without credentials;
    ``app.py`` reads the list once and hands it to CORS and to the gate, with no pattern.
  * OP6 -- THE FRONTEND OPENS ITS SOCKETS ON ITS OWN ORIGIN: ``wsUrl`` under Node gives the page's origin; the coding
    socket goes through it, port-free; the dev server and the nginx image proxy ``/ws`` with the Host kept, its port
    included, nginx to the backend image's port (round 1); the dev server answers no CORS of its own (round 2); the
    launcher's rewriter moves both targets.
  * OP7 -- THE TEXTS SAY THE RULE: SECURITY.md, the gate's docstring, ``app.py``, ``authenticate_websocket``, the
    frontend README, the proxy line of the garden's document and the CHANGELOG.
  * OP8 -- THE ROUTES TAKE THE SOCKETS THE GATE TAKES: the gate marks a handshake it lets through
    (``ORIGIN_CHECKED`` in the scope), and ``authenticate_websocket``, which held every Origin to a loopback name,
    takes it as it is -- a listed name over wss, a listed origin; a socket the gate did not mark is held to the
    loopback names as before (round 1).

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_api_answers_only_its_host_names_contracts as hg_t  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_op1_another_port_name_or_scheme_changes_nothing": 5.0,
    "test_op2_the_pages_of_this_site_are_answered": 5.0,
    "test_op3_another_sites_socket_is_closed_at_the_gate": 5.0,
    "test_op4_the_list_is_read_whole_or_not_at_all": 10.0,
    "test_op5_cors_answers_the_listed_origins_alone": 5.0,
    "test_op6_the_frontend_opens_its_sockets_on_its_own_origin": 15.0,
    "test_op7_the_texts_say_the_rule": 5.0,
    "test_op8_the_routes_take_the_sockets_the_gate_takes": 5.0,
}

_HOSTS = ("oo.lan",)
_SITE_REFUSED = b'{"detail":"This server takes no change from that page."}'


def _gate(gate_mod, origins=()):
    return gate_mod.HostGateMiddleware(None, hosts=_HOSTS, origins=origins)


def _drive(gate, headers, kind, method="GET", scheme=None):
    """Drive ``gate`` with one ASGI scope of ``kind`` made over ``scheme`` (none: the key left out, as ASGI allows):
    (what the gate sent, the scope the app was given or None)."""
    given = []

    async def app(scope, receive, send):
        given.append(scope)
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": f"{kind}.disconnect"}

    scope = {"type": kind, "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers]}
    if kind == "http":
        scope.update(method=method, path="/api/conversations")
    if scheme is not None:
        scope["scheme"] = scheme
    gate.app = app
    asyncio.run(gate(scope, receive, send))
    return sent, (given[0] if given else None)


def _change(gate_mod, extra, host, method="POST", origins=(), scheme=None):
    """A request to ``host`` with the ``extra`` headers: (status, whether the app ran, body)."""
    sent, given = _drive(_gate(gate_mod, origins), [("host", host)] + list(extra), "http", method, scheme)
    return hg_t._status(sent), given is not None, b"".join(m.get("body", b"") for m in sent)


def _socket(gate_mod, extra, host, origins=(), gate=None, scheme=None):
    """A websocket handshake to ``host``: (what the gate sent, the scope the app was given or None)."""
    return _drive(gate if gate is not None else _gate(gate_mod, origins), [("host", host)] + list(extra),
                  "websocket", scheme=scheme)


def test_op1_another_port_name_or_scheme_changes_nothing():
    gate_mod, restore = hg_t._gate_module()
    seen = {}
    try:
        for label, origin, host, origins, scheme in (
            ("an https page against the http API's default port", "https://localhost", "localhost", (), "http"),
            ("an http page against the https API's default port", "http://localhost", "localhost", (), "https"),
            ("an http page against the https API on its port", "http://localhost:8443", "localhost:8443", (),
             "https"),
            ("a TLS page before a proxy that does not pass the scheme on", "https://oo.lan", "oo.lan", (), "http"),
        ):
            seen[label] = _change(gate_mod, [("origin", origin)], host, origins=origins, scheme=scheme)
        for label, origin, host, origins in (
            ("another local port", "http://localhost:3000", "localhost:8001", ()),
            ("a printer's page", "http://127.0.0.1:631", "127.0.0.1:8001", ()),
            ("another loopback name, the same port", "http://127.0.0.1:5173", "localhost:5173", ()),
            ("ipv6 against ipv4", "http://[::1]:8001", "127.0.0.1:8001", ()),
            ("the default port against a written one", "http://localhost", "localhost:8001", ()),
            ("a written port against the default", "http://localhost:8001", "localhost", ()),
            ("https's default against http's", "https://oo.lan", "oo.lan:80", ()),
            ("another scheme", "tauri://localhost", "localhost:8001", ()),
            ("a socket's scheme", "ws://localhost:8001", "localhost:8001", ()),
            ("a listed name over http", "http://oo.lan:8443", "oo.lan:8443", ()),
            ("a listed origin's neighbour", "http://localhost:5174", "localhost:8001", ("http://localhost:5173",)),
            ("a listed origin over another scheme", "https://localhost:5173", "localhost:8001",
             ("http://localhost:5173",)),
            ("the wildcard is no origin", "http://localhost:3000", "localhost:8001", ("*",)),
        ):
            seen[label] = _change(gate_mod, [("origin", origin)], host, origins=origins)
        for method in ("PUT", "PATCH", "DELETE"):
            seen[method] = _change(gate_mod, [("origin", "http://localhost:3000")], "localhost:8001", method=method)
    finally:
        restore()
    for label, (status, ran, body) in seen.items():
        assert status == 403 and not ran, f"{label}: a page of another site changed something: {status} {ran}"
        assert body == _SITE_REFUSED, f"{label}: not the fixed words: {body!r}"


def test_op2_the_pages_of_this_site_are_answered():
    gate_mod, restore = hg_t._gate_module()
    seen = {}
    try:
        for label, extra, host, origins in (
            ("through the dev proxy", [("origin", "http://localhost:5173"), ("sec-fetch-site", "same-origin")],
             "localhost:5173", ()),
            ("the API's own address", [("origin", "http://127.0.0.1:8001")], "127.0.0.1:8001", ()),
            ("ipv6 loopback", [("origin", "http://[::1]:8001")], "[::1]:8001", ()),
            ("http's default port written in the Host", [("origin", "http://localhost")], "localhost:80", ()),
            ("capitals", [("origin", "HTTP://LOCALHOST:5173")], "LOCALHOST:5173", ()),
            ("a listed origin across ports", [("origin", "http://localhost:5173"), ("sec-fetch-site", "same-site")],
             "localhost:8001", ("http://localhost:5173",)),
            ("a listed origin in capitals", [("origin", "HTTP://LOCALHOST:5173")], "localhost:8001",
             ("http://localhost:5173",)),
            ("a listed origin of another name", [("origin", "https://ui.example")], "localhost:8001",
             ("https://ui.example",)),
            ("the same origin", [("sec-fetch-site", "same-origin")], "localhost:8001", ()),
            ("typed by the user", [("sec-fetch-site", "none")], "localhost:8001", ()),
            ("a fetch site in capitals", [("sec-fetch-site", "Same-Origin")], "localhost:8001", ()),
            ("the CLI", [], "localhost:8001", ()),
        ):
            seen[label] = _change(gate_mod, extra, host, origins=origins)
        for label, origin, host, origins, scheme in (
            ("a listed name over https", "https://oo.lan", "oo.lan", (), "https"),
            ("a listed name with its port", "https://oo.lan:8443", "oo.lan:8443", (), "https"),
            ("https's default port written in the Host", "https://oo.lan", "oo.lan:443", (), "https"),
            ("a loopback page over https", "https://localhost:8443", "localhost:8443", (), "https"),
            ("the http API named as such", "http://localhost:8001", "localhost:8001", (), "http"),
            ("a TLS proxy's page, listed", "https://oo.lan", "oo.lan", ("https://oo.lan",), "http"),
        ):
            seen[label] = _change(gate_mod, [("origin", origin)], host, origins=origins, scheme=scheme)
        for method in ("GET", "HEAD", "OPTIONS"):
            seen[f"a {method} from another port"] = _change(gate_mod, [("origin", "http://localhost:3000")],
                                                            "localhost:8001", method=method)
    finally:
        restore()
    for label, (status, ran, _body) in seen.items():
        assert ran and status == 200, f"{label}: a page of this site, the CLI or a safe method was refused: {status}"


def test_op3_another_sites_socket_is_closed_at_the_gate(caplog):
    gate_mod, restore = hg_t._gate_module()
    refused, passed = {}, {}
    try:
        for label, extra, host, origins in (
            ("another local port", [("origin", "http://localhost:3000")], "localhost:8001", ()),
            ("another site", [("origin", "https://evil.example")], "localhost:8001", ()),
            ("another loopback name", [("origin", "http://127.0.0.1:8001")], "localhost:8001", ()),
            ("another scheme", [("origin", "tauri://localhost")], "localhost:8001", ()),
            ("an extension", [("origin", "chrome-extension://abcdef")], "localhost:8001", ()),
            ("a file", [("origin", "file://")], "localhost:8001", ()),
            ("opaque", [("origin", "null")], "localhost:8001", ()),
            ("two origins", [("origin", "http://localhost:8001"), ("origin", "http://localhost:8001")],
             "localhost:8001", ()),
            ("a listed name over http", [("origin", "http://oo.lan:8001")], "oo.lan:8001", ()),
            ("not the listed one", [("origin", "http://localhost:3000")], "localhost:8001",
             ("http://localhost:5173",)),
            ("no origin, another site's fetch", [("sec-fetch-site", "cross-site")], "localhost:8001", ()),
        ):
            refused[label] = _socket(gate_mod, extra, host, origins)
        refused["an https page's socket over ws"] = _socket(gate_mod, [("origin", "https://localhost")], "localhost",
                                                            scheme="ws")
        refused["an http page's socket over wss"] = _socket(gate_mod, [("origin", "http://oo.lan:8443")],
                                                            "oo.lan:8443", scheme="wss")
        for label, extra, host, origins, scheme in (
            ("the same origin", [("origin", "http://localhost:5173")], "localhost:5173", (), None),
            ("the same origin over ws", [("origin", "http://localhost:5173")], "localhost:5173", (), "ws"),
            ("a listed name over wss", [("origin", "https://oo.lan:8443")], "oo.lan:8443", (), "wss"),
            ("a listed origin", [("origin", "http://localhost:5173")], "localhost:8001", ("http://localhost:5173",),
             None),
            ("a client with no origin", [], "localhost:8001", (), None),
        ):
            passed[label] = _socket(gate_mod, extra, host, origins, scheme=scheme)
        marked = gate_mod.ORIGIN_CHECKED
        gate = _gate(gate_mod)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            for _n in range(5):
                _socket(gate_mod, [("origin", "http://localhost:3000")], "localhost:8001", gate=gate)
        lines = [record.getMessage() for record in caplog.records]
    finally:
        restore()
    for label, (sent, given) in refused.items():
        assert given is None and sent == [{"type": "websocket.close", "code": 1008}], (
            f"{label}: a socket of another site was not closed before the app: {sent} {given}")
    for label, (sent, given) in passed.items():
        assert given is not None and sent == [], f"{label}: a socket of this site, or of no browser, was closed: {sent}"
        assert given.get(marked) is True, f"{label}: the gate let the socket through unmarked: {sorted(given)}"
    said = [line for line in lines if line.startswith("Host gate: refused a socket opened by a page of another site")]
    assert len(said) == 1 and "3000" not in said[0], f"the refused socket was not said once: {lines}"


def test_op4_the_list_is_read_whole_or_not_at_all(tmp_path, caplog):
    gate_mod, restore = hg_t._gate_module()
    files = {
        "a list": 'cors:\n  origins: ["http://localhost:5173", "https://oo.lan:443", "https://ui.example:8443",'
                  ' "http://[::1]:5173", "http://localhost:5173"]\n',
        "a string": 'cors:\n  origins: "http://127.0.0.1:5173, https://ui.example"\n',
        "the wildcard": 'cors:\n  origins: ["*", "http://localhost:5173"]\n',
        "no key": "mode: daily\n",
        "no value": "cors:\n  origins:\n",
        "an empty list": "cors:\n  origins: []\n",
        "a path": 'cors:\n  origins: ["http://localhost:5173/"]\n',
        "a wildcard name": 'cors:\n  origins: ["https://*.example"]\n',
        "a listed name over http": 'cors:\n  origins: ["http://oo.lan"]\n',
        "capitals": 'cors:\n  origins: ["HTTP://LOCALHOST:5173"]\n',
        "no scheme": 'cors:\n  origins: ["localhost:5173"]\n',
        "another scheme": 'cors:\n  origins: ["tauri://localhost"]\n',
        "a port out of range": 'cors:\n  origins: ["http://localhost:99999"]\n',
        "port zero": 'cors:\n  origins: ["http://localhost:0"]\n',
        "a padded quad": 'cors:\n  origins: ["https://192.168.001.020"]\n',
        "an unspecified address": 'cors:\n  origins: ["https://0.0.0.0"]\n',
        "a number among them": 'cors:\n  origins: ["http://localhost:5173", 5173]\n',
        "a good one beside a bad one": 'cors:\n  origins: ["http://localhost:5173", "http://oo.lan"]\n',
        "a value of another type": "cors:\n  origins: 5173\n",
        "a section of another type": 'cors: ["http://localhost:5173"]\n',
        "not yaml": "cors: [\n",
    }
    lists, warned = {}, {}
    try:
        for name, text in files.items():
            path = tmp_path / f"{name.replace(' ', '_')}.yaml"
            path.write_text(text, encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                lists[name] = gate_mod.listed_origins(path, environ={})
            warned[name] = any("no origin is listed" in r.getMessage() for r in caplog.records)
        absent = gate_mod.listed_origins(tmp_path / "absent.yaml", environ={})
        env = {"OPTI_CORS_ORIGINS": "http://127.0.0.1:5173,https://ui.example"}
        from_env = gate_mod.listed_origins(tmp_path / "no_key.yaml", environ=env)
        file_first = gate_mod.listed_origins(tmp_path / "a_string.yaml", environ={"OPTI_CORS_ORIGINS": "*"})
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            bad_env = gate_mod.listed_origins(tmp_path / "no_key.yaml", environ={"OPTI_CORS_ORIGINS": "http://oo.lan"})
        env_warned = [r.getMessage() for r in caplog.records if "no origin is listed" in r.getMessage()]
    finally:
        restore()
    assert lists["a list"] == ("http://localhost:5173", "https://oo.lan", "https://ui.example:8443",
                               "http://[::1]:5173"), lists["a list"]
    assert lists["a string"] == ("http://127.0.0.1:5173", "https://ui.example"), lists["a string"]
    assert lists["the wildcard"] == ("*",), lists["the wildcard"]
    for name in ("no key", "no value", "an empty list"):
        assert lists[name] == () and not warned[name], f"{name}: an absent list was said or read: {lists[name]}"
    assert absent == (), absent
    for name in ("a path", "a wildcard name", "a listed name over http", "capitals", "no scheme", "another scheme",
                 "a port out of range", "port zero", "a padded quad", "an unspecified address", "a number among them",
                 "a good one beside a bad one", "a value of another type", "a section of another type", "not yaml"):
        assert lists[name] == () and warned[name], f"{name}: a list not read whole was not refused and said: {lists}"
    assert from_env == ("http://127.0.0.1:5173", "https://ui.example"), from_env
    assert file_first == ("http://127.0.0.1:5173", "https://ui.example"), "the variable was read over the file"
    assert bad_env == () and len(env_warned) == 1 and "OPTI_CORS_ORIGINS" in env_warned[0], (bad_env, env_warned)
    assert "oo.lan" not in env_warned[0], f"the refusal repeated the value: {env_warned}"

    # The variable a shell lists never reaches a contract: a gate built without its origins lists none (round 1).
    conftest = Path(__file__).resolve().parent / "conftest.py"
    env = dict(os.environ, OPTI_CORS_ORIGINS="http://localhost:3000")
    tell = "import os; print('OPTI_CORS_ORIGINS' in os.environ)"
    plain = subprocess.run([sys.executable, "-c", tell], env=env, capture_output=True, text=True, timeout=60)
    loaded = subprocess.run([sys.executable, "-c", "import runpy, sys; runpy.run_path(sys.argv[1]); " + tell,
                             str(conftest)], env=env, capture_output=True, text=True, timeout=60)
    assert plain.stdout.strip() == "True", ("control: the child is given the variable", plain.stdout, plain.stderr)
    assert loaded.returncode == 0 and loaded.stdout.strip().splitlines()[-1] == "False", (
        "the suites' configuration does not drop the origins a shell lists", loaded.stdout, loaded.stderr)
    assert "OPTI_CORS_ORIGINS" not in os.environ, "this session carries the shell's origins"


def _cors(gate_mod, origins, origin, preflight=False):
    """The app's CORS middleware built from ``cors_settings(origins)``, asked by a page of ``origin``: (status,
    headers answered, whether the app ran)."""
    ran = []

    async def app(scope, receive, send):
        ran.append(scope["method"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = gate_mod.ListedOriginsCORS(app, **gate_mod.cors_settings(origins))
    headers = [(b"host", b"localhost:8001"), (b"origin", origin.encode("latin-1"))]
    if preflight:
        headers.append((b"access-control-request-method", b"POST"))
    scope = {"type": "http", "method": "OPTIONS" if preflight else "GET", "path": "/api/conversations",
             "headers": headers, "query_string": b"", "scheme": "http", "server": ("127.0.0.1", 8001)}
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    answered = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in start.get("headers", [])}
    return start["status"], answered, bool(ran)


def test_op5_cors_answers_the_listed_origins_alone(caplog):
    gate_mod, restore = hg_t._gate_module()
    try:
        listed = ("http://localhost:5173",)
        listed_simple = _cors(gate_mod, listed, "http://localhost:5173")
        listed_preflight = _cors(gate_mod, listed, "http://localhost:5173", preflight=True)
        other_simple = _cors(gate_mod, listed, "http://localhost:3000")
        other_preflight = _cors(gate_mod, listed, "http://localhost:3000", preflight=True)
        unlisted = _cors(gate_mod, (), "http://localhost:5173")
        unlisted_preflight = _cors(gate_mod, (), "http://127.0.0.1:5173", preflight=True)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            wildcard = _cors(gate_mod, ("*",), "http://localhost:3000")
        wildcard_said = [r.getMessage() for r in caplog.records if "wildcard" in r.getMessage()]
        settings = gate_mod.cors_settings(listed)
    finally:
        restore()
    for label, (_status, answered, _ran) in (("simple", listed_simple), ("preflight", listed_preflight)):
        assert answered.get("access-control-allow-origin") == "http://localhost:5173", (label, answered)
        assert answered.get("access-control-allow-credentials") == "true", (label, answered)
    assert listed_preflight[0] == 200 and not listed_preflight[2], listed_preflight
    for label, (status, answered, ran) in (("simple", other_simple), ("listed none", unlisted)):
        assert ran and status == 200, (label, status)
        cors = [key for key in answered if key.startswith("access-control-")]
        assert cors == [], f"another port's page was answered CORS headers ({label}): {cors} (round 2)"
    for label, (status, answered, ran) in (("another port", other_preflight), ("listed none", unlisted_preflight)):
        cors = [key for key in answered if key.startswith("access-control-")]
        assert status == 400 and not ran and cors == [], (label, status, answered)
    assert wildcard[1].get("access-control-allow-origin") == "*", wildcard
    assert "access-control-allow-credentials" not in wildcard[1], wildcard
    assert len(wildcard_said) == 1, wildcard_said
    assert "allow_origin_regex" not in settings and settings["allow_origins"] == ["http://localhost:5173"], settings

    text = source("api", "app.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
                "listed_origins", "cors_settings"):
            calls.setdefault(node.func.id, []).append(ast.unparse(node))
    assigned = {node.targets[0].id: ast.unparse(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    added = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "add_middleware"
             and node.args and isinstance(node.args[0], ast.Name)
             and node.args[0].id in ("CORSMiddleware", "ListedOriginsCORS", "HostGateMiddleware")]
    assert calls == {"listed_origins": ["listed_origins()"], "cors_settings": ["cors_settings(_listed_origins)"]}, calls
    assert assigned.get("_listed_origins") == "listed_origins()" and assigned.get("_cors") == (
        "cors_settings(_listed_origins)"), "app.py does not read the list once for CORS and the gate"
    assert added == ["app.add_middleware(ListedOriginsCORS, **_cors)",
                     "app.add_middleware(HostGateMiddleware, origins=_listed_origins)"], added
    assert "allow_origin_regex" not in text and "_LOCALHOST_RE" not in text, "a pattern of origins is answered"
    assert "CORSMiddleware" not in text, "app.py names Starlette's CORS, which answers an unlisted origin"


_WS_DRIVER = r"""
const clause = process.argv[2];
function fail(why) { console.log('FAIL ' + why); process.exit(1); }
const { wsUrl } = await import(process.env.OO_CLIENT);
for (const [origin, wanted] of [
  ['http://localhost:5173', 'ws://localhost:5173/ws/coding/live'],
  ['https://oo.lan:8443', 'wss://oo.lan:8443/ws/coding/live'],
  ['http://127.0.0.1:8001', 'ws://127.0.0.1:8001/ws/coding/live'],
]) {
  globalThis.window = { location: { origin } };
  const got = wsUrl('/ws/coding/live');
  if (got !== wanted) fail(origin + ' gave ' + got);
}
console.log('PASS ' + clause);
"""


def _block(config, key):
    """The braces of the entry ``key`` (quoted or not) in vite's config text."""
    start = re.search(r"""['"]?""" + re.escape(key) + r"""['"]?\s*:\s*\{""", config)
    assert start, f"no {key} proxy entry"
    depth = 0
    for index in range(start.end() - 1, len(config)):
        depth += {"{": 1, "}": -1}.get(config[index], 0)
        if depth == 0:
            return config[start.end() - 1:index + 1]
    raise AssertionError(f"the {key} proxy entry does not close")


def test_op6_the_frontend_opens_its_sockets_on_its_own_origin(tmp_path):
    from _frontend import read, run_ts

    run_ts({"OO_CLIENT": "frontend/src/lib/api/client.ts"}, _WS_DRIVER, "own origin")

    coding = read("frontend/src/lib/api/codingAgent.ts")
    assert "import { apiGet, apiPost, wsUrl } from './client';" in coding
    assert coding.count("new WebSocket(") == 1 and "new WebSocket(wsUrl('/ws/coding/live'))" in coding, (
        "the coding socket is not opened on the page's own origin")
    assert "8001" not in coding and "location.hostname" not in coding and "location.protocol" not in coding, (
        "the coding socket still names a host or a port of its own")

    config = read("frontend/vite.config.ts")
    server = _block(config, "server")
    assert re.findall(r"\bcors\s*:\s*(\w+)", server) == ["false"], (
        "the dev server answers CORS of its own: its default lets a page of any loopback port read (round 2)")
    api, ws = _block(config, "/api"), _block(config, "/ws")
    target = re.findall(r"target\s*:\s*'([^']*)'", api)
    assert len(target) == 1 and re.findall(r"target\s*:\s*'([^']*)'", ws) == target, (api, ws)
    for entry in (api, ws):
        assert re.findall(r"changeOrigin\s*:\s*(\w+)", entry) == ["false"], entry
        assert re.findall(r"\bws\s*:\s*(\w+)", entry) == ["true"], entry

    nginx = read("frontend/nginx.conf")
    hosts = re.findall(r"proxy_set_header\s+Host\s+(\S+);", nginx)
    assert hosts == ["$http_host", "$http_host"], f"nginx does not forward the Host with its port: {hosts}"
    location = nginx.split("location /ws/ {", 1)
    assert len(location) == 2, "nginx does not proxy /ws/"
    body = location[1].split("}", 1)[0]
    for line in ("proxy_set_header Upgrade $http_upgrade;", 'proxy_set_header Connection "upgrade";',
                 "proxy_http_version 1.1;"):
        assert line in body, f"/ws/ is not proxied as a socket: {line}"
    served = re.findall(r'"--port", "([0-9]+)"', (REPO / "Dockerfile.backend").read_text(encoding="utf-8"))
    targets = re.findall(r"proxy_pass\s+http://backend:([0-9]+);", nginx)
    assert len(served) == 1 and targets == served * 2, (
        f"nginx proxies to a port the backend image does not serve: {targets} against {served} (round 1)")

    loaded, restore = isolate(targets={"opti_oignon.ui": source("ui.py")})
    try:
        root = tmp_path / "project"
        (root / "frontend").mkdir(parents=True)
        (root / "frontend" / "vite.config.ts").write_text(config, encoding="utf-8")
        loaded["opti_oignon.ui"]._update_vite_proxy(root, 8123)
        moved = (root / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
    finally:
        restore()
    for key in ("/api", "/ws"):
        assert re.findall(r"target\s*:\s*'([^']*)'", _block(moved, key)) == ["http://localhost:8123"], key


def test_op7_the_texts_say_the_rule():
    security = (REPO / "SECURITY.md").read_text(encoding="utf-8")
    gate = source("api", "host_gate.py").read_text(encoding="utf-8")
    app = source("api", "app.py").read_text(encoding="utf-8")
    auth = source("api", "routes_auth.py").read_text(encoding="utf-8")
    readme = (REPO / "frontend" / "README.md").read_text(encoding="utf-8")
    garden = (REPO / "docs" / "architecture" / "onion-memory.md").read_text(encoding="utf-8")
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    flat = " ".join(security.split())

    for said in ("A page of another port of this machine (a dev server, a local app) or of another scheme is "
                 "another site.", "and every websocket handshake that a page of another site sends",
                 "origins listed under `cors.origins`", "proxy_set_header Host $http_host",
                 "one entry that is not refuses the whole list"):
        assert said in flat, f"SECURITY.md does not say: {said}"
    for gone in ("a loopback name, on any port", "`proxy_set_header Host $host`", "turns the gate off"):
        assert gone not in flat, f"SECURITY.md still says: {gone}"
    for said in ("a page of an origin not listed gets no CORS header at all",
                 "refuses every change and every socket of the proxy's own pages"):
        assert said in flat, f"SECURITY.md does not say: {said} (round 2)"
    flat_gate = " ".join(gate.split())
    assert "on any port -- a page of this machine" not in flat_gate
    assert "never compared" not in flat_gate and "compared here with the Origin's" in flat_gate, (
        "the gate's docstring says ports are never compared (round 2)")
    assert "A page of another port, of another name or of another scheme of this very machine" in flat_gate
    assert "Always include localhost variants" not in app and "any port" not in app
    doc = ast.get_docstring(next(node for node in ast.parse(auth).body
                                 if isinstance(node, ast.AsyncFunctionDef) and node.name == "authenticate_websocket"))
    flat_doc = " ".join(doc.split())
    for said in ("api/host_gate.py", "``ORIGIN_CHECKED`` in the scope", "A socket the gate did not see is held here"):
        assert said in flat_doc, f"authenticate_websocket's docstring does not say: {said}"
    assert "malicious website cannot" not in flat_doc and "a looser one" not in flat_doc, doc
    for said in ("passes the scheme on (`X-Forwarded-Proto`, from an address uvicorn trusts)",
                 "The routes' own socket check takes what the gate let through"):
        assert said in flat, f"SECURITY.md does not say: {said}"
    assert "under `cors.origins` of `security.yaml`" in readme
    assert "proxy_set_header Host $http_host" in garden and "proxy_set_header Host $host`" not in garden
    flat_garden = " ".join(garden.split())
    assert "The garden compares names, not ports" in flat_garden, "the garden's document hides its port rule"
    assert "so a page of another site cannot make the garden look" not in flat_garden
    fixed = changelog.split("### Fixed\n\n", 1)[1]
    assert fixed.startswith("- [SECURITY] A page of another port of this machine is now another site."), fixed[:80]


def test_op8_the_routes_take_the_sockets_the_gate_takes():
    import test_auth_2fa_routes_seam_contracts as seam
    from starlette.websockets import WebSocket

    gate_mod, restore = hg_t._gate_module()
    restore()
    ctx = seam._load()
    held = {}
    try:
        ctx.manager.tokens["valid-token"] = {
            "sub": "uid-socket", "username": seam._RESIDENT, "role": "user", "type": "access",
        }
        cookie = f"{seam._ACCESS_COOKIE}=valid-token".encode("latin-1")

        def through_the_gate(label, origin, host, origins=(), scheme="wss"):
            """A handshake through the gate to a route that asks ``authenticate_websocket``: what the gate sent."""
            sent = []

            async def route(scope, receive, send):
                held[label] = await ctx.routes.authenticate_websocket(WebSocket(scope, receive, send))

            async def send(message):
                sent.append(message)

            async def receive():
                return {"type": "websocket.disconnect"}

            gate = gate_mod.HostGateMiddleware(route, hosts=_HOSTS, origins=origins)
            scope = {"type": "websocket", "scheme": scheme, "path": "/api/chat/stream", "query_string": b"",
                     "headers": [(b"host", host.encode("latin-1")), (b"origin", origin.encode("latin-1")),
                                 (b"cookie", cookie)]}
            asyncio.run(gate(scope, receive, send))
            return sent

        listed_name = through_the_gate("a listed name over wss", "https://oo.lan:8443", "oo.lan:8443")
        listed_origin = through_the_gate("a listed origin", "https://ui.example", "localhost:8443",
                                         origins=("https://ui.example",))
        through_the_gate("the same loopback origin", "http://localhost:5173", "localhost:5173", scheme="ws")
        unseen = {}
        for label, scope_extra in (("unmarked", {}), ("marked otherwise", {gate_mod.ORIGIN_CHECKED: "yes"})):
            socket = seam._Socket(origin="https://oo.lan:8443", cookies={seam._ACCESS_COOKIE: "valid-token"})
            socket.scope = {"type": "websocket", **scope_extra}
            unseen[label] = asyncio.run(ctx.routes.authenticate_websocket(socket))
        evil = seam._Socket(origin="https://evil.example", cookies={seam._ACCESS_COOKIE: "valid-token"})
        refused = asyncio.run(ctx.routes.authenticate_websocket(evil))
        route_text = source("api", "routes_auth.py").read_text(encoding="utf-8")
    finally:
        ctx.restore()
    assert listed_name == [] and listed_origin == [], (listed_name, listed_origin)
    for label in ("a listed name over wss", "a listed origin", "the same loopback origin"):
        assert held.get(label) is not None and held[label]["sub"] == "uid-socket", (
            f"{label}: the route refused a socket the gate let through: {held}")
    for label, user in unseen.items():
        assert user is None, f"{label}: a socket the gate did not mark was taken by a listed name: {user}"
    assert refused is None, "a socket of another site was taken by the route"
    assert gate_mod.ORIGIN_CHECKED == "opti_oignon.origin_checked"
    assert f'scope.get("{gate_mod.ORIGIN_CHECKED}") is True' in route_text, "the route reads another mark"
