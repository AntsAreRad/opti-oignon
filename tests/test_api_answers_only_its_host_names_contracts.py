"""Contracts for the Host gate: the API answers only the names this machine is served as.

Nothing read the Host header. A page from anywhere can rebind its own name to 127.0.0.1 (DNS rebinding): the browser
then sends the page's requests to the local API under the page's own origin -- same-origin for the browser, so CORS
never applies -- with ``Host: attacker.example:PORT``; in single-user Daily mode the synthetic admin answered them,
CSRF skipped. Now ``api/host_gate.HostGateMiddleware``, the outermost middleware, refuses every request (403) and every
websocket handshake (closed before it is accepted) whose Host header is not exactly one loopback name or a name listed
under ``api.hosts`` of ``security.yaml``, by the garden's routes' grammar.

  * HG1 -- A REBOUND NAME REACHES NOTHING: a GET, a POST and a websocket handshake addressed to a rebound name are
    refused before the app runs (the fixed body, the name never repeated); the loopback names, with a port or without,
    in any case, are answered; no Host, two Host headers, and the forms the garden refuses are refused; a forwarded
    name is never read; the lifespan passes; a flood of names is logged by name for the first 32, then counted every
    hundred refusals (round 1).
  * HG2 -- THE NAMES THE MACHINE IS SERVED AS: the names listed under ``api.hosts`` are answered, others refused; an
    absent file or key lists none; a file or a list that cannot be read whole lists none and says so, an entry the
    garden's rule refuses included (an unspecified address, an IPv4 address spelled otherwise than as its dotted quad;
    round 1); the gate's grammar and loopback names are the garden's.
  * HG3 -- REGISTERED OUTSIDE EVERYTHING: in ``api/app.py`` the gate is imported unconditionally and added after every
    other middleware (Starlette runs the last added first), and no middleware runs outside that chain (no
    ``@app.middleware``, no wrapping of ``app``; round 1), so a refused request never reaches CORS, the security
    headers, the authentication or the CSRF check.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_hg1_a_rebound_name_reaches_nothing": 5.0,
    "test_hg2_the_names_the_machine_is_served_as": 5.0,
    "test_hg3_registered_outside_everything": 5.0,
}


def _gate_module():
    """The gate's module loaded from its source in an isolation window: (module, restore)."""
    loaded, restore = isolate(targets={"opti_oignon.api.host_gate": source("api", "host_gate.py")})
    return loaded["opti_oignon.api.host_gate"], restore


def _ask(gate_mod, headers, kind="http", hosts=(), method="GET", gate=None):
    """Drive the gate (a new one around a recording app, or ``gate``) with one ASGI scope: (what the gate sent,
    whether the app ran)."""
    ran = []

    async def app(scope, receive, send):
        ran.append(scope["type"])
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": f"{kind}.disconnect"} if kind != "lifespan" else {"type": "lifespan.startup"}

    scope = {"type": kind, "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers]}
    if kind == "http":
        scope.update(method=method, path="/api/conversations")
    if gate is None:
        gate = gate_mod.HostGateMiddleware(app, hosts=hosts)
    else:
        gate.app = app
    asyncio.run(gate(scope, receive, send))
    return sent, bool(ran)


def _status(sent):
    starts = [m for m in sent if m["type"] == "http.response.start"]
    return starts[0]["status"] if starts else None


def test_hg1_a_rebound_name_reaches_nothing(caplog):
    gate_mod, restore = _gate_module()
    try:
        rebound = {}
        for method in ("GET", "POST"):
            sent, ran = _ask(gate_mod, [("host", "attacker.example:8001")], method=method)
            rebound[method] = (_status(sent), ran, b"".join(m.get("body", b"") for m in sent))
        ws_sent, ws_ran = _ask(gate_mod, [("host", "attacker.example:8001")], kind="websocket")
        answered = {host: _ask(gate_mod, [("host", host)])
                    for host in ("localhost", "localhost:8001", "127.0.0.1:5173", "[::1]:8001", "LocalHost:8001")}
        refused = {}
        for label, headers in {
            "no host": [],
            "two hosts": [("host", "localhost"), ("host", "localhost")],
            "trailing dot": [("host", "127.0.0.1.")],
            "user part": [("host", "user@127.0.0.1")],
            "at evil": [("host", "127.0.0.1@evil")],
            "port out of range": [("host", "127.0.0.1:99999")],
            "port zero": [("host", "localhost:0")],
            "loopback as a label": [("host", "localhost.evil.com")],
            "forwarded name": [("host", "evil.example"), ("x-forwarded-host", "127.0.0.1")],
            "a good origin": [("host", "evil.example:8001"), ("origin", "http://localhost:5173")],
        }.items():
            refused[label] = _ask(gate_mod, headers)
        lifespan_sent, lifespan_ran = _ask(gate_mod, [], kind="lifespan")
        # A flood: 32 names spent, one of them refused again and again, then the operator's own unlisted name
        flooded = gate_mod.HostGateMiddleware(None, hosts=())
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            for n in range(32):
                _ask(gate_mod, [("host", f"name{n}.example")], gate=flooded)
            for _n in range(168):
                _ask(gate_mod, [("host", "name0.example")], gate=flooded)
            for _n in range(100):
                _ask(gate_mod, [("host", "phone.lan")], gate=flooded)
        lines = [record.getMessage() for record in caplog.records]
    finally:
        restore()
    named = [line for line in lines if "refused a request addressed to" in line]
    counted = [line for line in lines if "requests refused since the start" in line]
    assert len(named) == 32 and len(counted) == 3 and "300 requests refused" in counted[-1], (
        f"a flood of names hid the refusals past the first ones: {len(named)} named, {counted}")
    assert counted[-1].endswith("addressed to phone.lan"), f"the latest refused name was never said: {counted}"
    for method, (status, ran, body) in rebound.items():
        assert status == 403 and not ran, f"{method}: a rebound name reached the app: {status} {ran}"
        assert body == b'{"detail":"This server does not answer that host name."}', body
        assert b"attacker" not in body, "the refusal repeated the name refused"
    assert not ws_ran and ws_sent == [{"type": "websocket.close", "code": 1008}], (
        f"a websocket handshake addressed to a rebound name was not closed before the app: {ws_sent} {ws_ran}")
    for host, (sent, ran) in answered.items():
        assert ran and _status(sent) == 200, f"a loopback name was not answered: {host} {sent}"
    for label, (sent, ran) in refused.items():
        assert not ran and _status(sent) == 403, f"{label}: a request not addressed to one allowed name ran: {sent}"
    assert lifespan_ran and lifespan_sent == [], "the gate stopped the lifespan"


def test_hg2_the_names_the_machine_is_served_as(tmp_path, caplog):
    gate_mod, restore = _gate_module()
    files = {
        "listed": "api:\n  hosts: [\"oo.lan\", \"192.168.1.20\", \"[fd00::1]\"]\n",
        "no key": "mode: daily\n",
        "not yaml": "api: [\n",
        "not a list": "api:\n  hosts: oo.lan\n",
        "a name of another form": "api:\n  hosts: [\"oo.lan\", \"*.lan\"]\n",
        "upper case": "api:\n  hosts: [\"OO.lan\"]\n",
        "unspecified v4": "api:\n  hosts: [\"0.0.0.0\"]\n",
        "unspecified v6": "api:\n  hosts: [\"[::]\"]\n",
        "mapped unspecified": "api:\n  hosts: [\"[::ffff:0.0.0.0]\"]\n",
        "a decimal address": "api:\n  hosts: [\"2130706433\"]\n",
        "a padded quad": "api:\n  hosts: [\"192.168.001.020\"]\n",
        "a long v6 spelling": "api:\n  hosts: [\"[fd00:0::1]\"]\n",
        "an invalid date": "since: 2026-13-45\napi:\n  hosts: [\"oo.lan\"]\n",
    }
    lists, warned = {}, {}
    try:
        for name, text in files.items():
            path = tmp_path / f"{name.replace(' ', '_')}.yaml"
            path.write_text(text, encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                lists[name] = gate_mod.listed_hosts(path)
            warned[name] = any("loopback names only" in r.getMessage() for r in caplog.records)
        absent = gate_mod.listed_hosts(tmp_path / "absent.yaml")
        hosts = lists["listed"]
        served = {host: _ask(gate_mod, [("host", host)], hosts=hosts)[1]
                  for host in ("oo.lan", "oo.lan:8443", "192.168.1.20", "[fd00::1]:8001", "localhost")}
        not_served = {host: _ask(gate_mod, [("host", host)], hosts=hosts)[1]
                      for host in ("other.lan", "oo.lan.evil", "192.168.1.21")}
        gate_text = source("api", "host_gate.py").read_text(encoding="utf-8")
        garden_text = source("api", "routes_allium.py").read_text(encoding="utf-8")
    finally:
        restore()
    assert lists["listed"] == ("oo.lan", "192.168.1.20", "[fd00::1]") and absent == () and lists["no key"] == (), lists
    for name in ("not yaml", "not a list", "a name of another form", "upper case", "unspecified v4", "unspecified v6",
                 "mapped unspecified", "a decimal address", "a padded quad", "a long v6 spelling", "an invalid date"):
        assert lists[name] == () and warned[name], f"{name}: a list not read whole was not refused and said: {lists}"
    assert not warned["no key"], "an absent key was said as an unreadable list"
    assert all(served.values()), f"a listed name was not answered: {served}"
    assert not any(not_served.values()), f"a name not listed was answered: {not_served}"
    for line in ('LOOPBACK = ("127.0.0.1", "localhost", "[::1]")',
                 '_NAME = r"(?P<name>\\[[0-9a-f:]+\\]|[a-z0-9-]+(?:\\.[a-z0-9-]+)*)"',
                 '_PORT = r"(?::(?P<port>[0-9]{1,5}))?"',
                 '_LISTED = re.compile(r"[a-z0-9-]+(\\.[a-z0-9-]+)*|\\[[0-9a-f:]+\\]")'):
        assert line in gate_text and line in garden_text, f"the gate's rule is not the garden's: {line}"


def test_hg3_registered_outside_everything():
    tree = ast.parse(source("api", "app.py").read_text(encoding="utf-8"))
    added = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_middleware":
            given = node.args[0] if node.args else next(
                (kw.value for kw in node.keywords if kw.arg == "middleware_class"), None)
            added.append((node.lineno, given.id if isinstance(given, ast.Name) else ast.unparse(given or node)))
    added.sort()
    # No other way to put code before the gate: no middleware decorator or call, no hand on the middleware list or
    # the built stack, one app made once, never aliased
    text = source("api", "app.py").read_text(encoding="utf-8")
    other_ways = [word for word in (".middleware(", "user_middleware", "middleware_stack", "build_middleware_stack")
                  if word in text]
    made = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", "")) in ("FastAPI", "Starlette")]
    bound = [node.lineno for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
             and any(isinstance(target, ast.Name) and target.id == "app"
                     for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    aliased = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and isinstance(node.value, ast.Name) and node.value.id == "app"]
    decorated = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "middleware"
                         for d in node.decorator_list)]
    rebound = [node.lineno for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "app" for target in node.targets)
               and not (isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "FastAPI")]
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "host_gate"
               and any(alias.name == "HostGateMiddleware" for alias in node.names)]
    guarded = [node for node in ast.walk(tree) if isinstance(node, ast.Try)
               and any(isinstance(inner, ast.ImportFrom) and inner.module == "host_gate" for inner in ast.walk(node))]
    assert added and added[-1][1] == "HostGateMiddleware", (
        f"the gate is not the last middleware added (the first to run): {added}")
    assert [name for _line, name in added].count("HostGateMiddleware") == 1, added
    assert imports and not guarded, "the gate's import is not unconditional at the module's top level"
    assert not decorated and not rebound and not other_ways, (
        f"a middleware outside the add_middleware chain could run before the gate: {decorated} {rebound} {other_ways}")
    assert len(made) == 1 and len(bound) == 1 and not aliased, (
        f"the app is not made once and bound once under one name: {made} {bound} {aliased}")
