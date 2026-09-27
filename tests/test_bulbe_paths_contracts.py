"""Contracts for the Bulbe middleware's paths: it refuses what the router would serve, by the router's reading.

The security-mode middleware decides by path, before the router runs. A
decision by string prefix on a path the router reads differently refuses
routes it never meant and serves routes it meant to refuse. These contracts
drive the real middleware over a minimal application, one ASGI request at a
time, and read its lists against the routes the package mounts.

  * BP1 -- the middleware matches what the router matches: whole path
    segments, on the path the router reads (a mount prefix removed); every
    entry of its lists names a mounted route, and every web-only route is
    exactly one mounted route with that method.
  * BP2 -- the web-only routes are refused in Bulbe before their handler, in
    the web gate's words; in Daily each reaches its handler, and the
    marketplace listing stays open in Bulbe.
  * BP3 -- the always-allowed list admits the health route exactly: a route
    below it is not admitted with it.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window.
"""

import ast
import asyncio
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_bp1_the_middleware_matches_what_the_router_matches": 2.0,
    "test_bp2_the_web_only_routes_are_refused_in_bulbe_before_their_handler": 2.0,
    "test_bp3_the_always_allowed_list_admits_the_health_route_exactly": 2.0,
    "test_bp4_the_plugin_allowlist_reads_the_plugin_from_the_path_the_router_reads": 2.0,
}

_MW = "opti_oignon.api.security_mode_middleware"
_WG = "opti_oignon.web_gate"
_EG = "opti_oignon.egress"
_SM = "opti_oignon.security_mode"
_KS = "opti_oignon.search_killswitch"
_ALLOW = "opti_oignon.plugin_allowlist"

_API = REPO / "opti_oignon" / "api"
_WEB_MODE_TEXT = "Web search is refused outside Daily mode."
_WEB_ONLY = (
    ("POST", "/api/plugins/marketplace/install"),
    ("POST", "/api/backends/gguf/download"),
    ("POST", "/api/model-lifecycle/pull"),
)
_MINIMAL_ROUTES = (
    ("GET", "/api/search"),
    ("GET", "/api/searchx"),
    ("GET", "/api/search/config"),
    *_WEB_ONLY,
    ("GET", "/api/plugins/marketplace"),
    ("GET", "/api/health"),
    ("POST", "/api/health/benchmarks"),
    ("POST", "/api/plugins/install/allowed-plugin"),
)
_LOCAL = "127.0.0.1"
_REMOTE = "203.0.113.9"


# ---------------------------------------------------------------------------
# The window: the real middleware, a mode and a switch standing in.
# ---------------------------------------------------------------------------
def _mode_module(mode):
    """The security mode: its value, and a Bulbe policy as strict as the shipped one."""
    sm = types.ModuleType(_SM)
    sm.mode = mode
    sm.get_current_mode = lambda: sm.mode
    sm.get_policy = lambda: SimpleNamespace(
        bearer_auth_allowed=False, plugin_allowlist_required=True, cookie_samesite="Strict",
        rate_limit_max_attempts=3, rate_limit_window=300,
    )
    sm.is_bulbe = lambda: sm.mode == "bulbe"
    sm._audit_log = lambda *args, **kwargs: None
    return sm


def _switch_module(engaged):
    ks = types.ModuleType(_KS)
    ks.search_killswitch = SimpleNamespace(is_killed=lambda: engaged, is_enabled=lambda: not engaged)
    return ks


@contextmanager
def _middleware(mode, *, engaged=False, seeded=None):
    """The middleware over a minimal application; every handler it lets through is recorded in ``calls``."""
    from fastapi import FastAPI

    targets = {}
    for name, path in ((_WG, source("web_gate.py")), (_EG, source("egress.py"))):
        if path.exists():
            targets[name] = path
    targets[_MW] = source("api", "security_mode_middleware.py")
    sm = _mode_module(mode)
    loaded, restore = isolate(
        targets=targets, seeded={_SM: sm, _KS: _switch_module(engaged), **(seeded or {})},
        packages=("opti_oignon.api",),
    )
    try:
        mw = loaded[_MW]
        calls = []
        application = FastAPI()
        for method, path in _MINIMAL_ROUTES:
            application.add_api_route(path, _handler(calls, f"{method} {path}"), methods=[method])
        application.add_middleware(mw.SecurityModeMiddleware)
        yield SimpleNamespace(mw=mw, app=application, calls=calls, sm=sm, wg=loaded.get(_WG))
    finally:
        restore()


def _handler(calls, name):
    def handler():
        calls.append(name)
        return {"handler": name}

    return handler


def _ask(application, method, path, *, client=_LOCAL, root_path=""):
    """One ASGI request, with a session cookie and no bearer token: (status, body as JSON)."""
    sent = []
    full = root_path + path
    scope = {
        "type": "http", "method": method, "path": full, "raw_path": full.encode("ascii"), "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:8001"), (b"cookie", b"oo_session=local")],
        "client": (client, 50000), "server": ("127.0.0.1", 8001), "scheme": "http", "http_version": "1.1",
        "root_path": root_path,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(application(scope, receive, send))
    body = b"".join(message.get("body", b"") for message in sent[1:])
    try:
        return sent[0]["status"], json.loads(body)
    except ValueError:
        return sent[0]["status"], body


# ---------------------------------------------------------------------------
# The route census: every route the package mounts, read from its source.
# ---------------------------------------------------------------------------
_VERBS = ("get", "post", "put", "delete", "patch", "head", "options")
_FASTAPI_OWN = (("docs_url", "/docs"), ("redoc_url", "/redoc"), ("openapi_url", "/openapi.json"))


def _callee(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword(call, name):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _route_census(api_dir):
    """(routes, unread): every (METHOD, path) a router or application decorator mounts, and what could not be read.

    A router's prefix is its ``APIRouter(prefix=...)``; an application is
    ``FastAPI(...)``, which also mounts its own documentation routes unless
    the call turns them off. A prefix or a path that is not a string literal
    is reported in ``unread`` rather than guessed.
    """
    routes, unread = [], []
    for path in sorted(api_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        prefixes = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            kind = _callee(node.value)
            if kind not in ("APIRouter", "FastAPI"):
                continue
            given = _keyword(node.value, "prefix")
            if given is None:
                prefix = ""
            elif isinstance(given, ast.Constant) and isinstance(given.value, str):
                prefix = given.value
            else:
                unread.append(f"{path.name}:{node.lineno}: a router prefix that is not a literal")
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    prefixes.setdefault(target.id, set()).add(prefix)
            if kind == "FastAPI":
                for name, default in _FASTAPI_OWN:
                    value = _keyword(node.value, name)
                    if value is None:
                        routes.append(("GET", default))
                    elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                        routes.append(("GET", value.value))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id in prefixes):
                    continue
                verb = decorator.func.attr
                if verb in _VERBS:
                    methods = [verb.upper()]
                elif verb == "websocket":
                    methods = ["WEBSOCKET"]
                elif verb == "api_route":
                    listed = _keyword(decorator, "methods")
                    methods = ([e.value.upper() for e in listed.elts if isinstance(e, ast.Constant)]
                               if isinstance(listed, (ast.List, ast.Tuple)) else ["GET"])
                else:
                    continue
                first = decorator.args[0] if decorator.args else _keyword(decorator, "path")
                owners = prefixes[decorator.func.value.id]
                if not (isinstance(first, ast.Constant) and isinstance(first.value, str)) or len(owners) != 1:
                    unread.append(f"{path.name}:{decorator.lineno}: a route whose path cannot be read")
                    continue
                routes += [(method, next(iter(owners)) + first.value) for method in methods]
    return routes, unread


def _segment_match(path, prefix):
    return path == prefix or path.startswith(prefix + "/")


# ---------------------------------------------------------------------------
# BP1 -- the middleware matches what the router matches
# ---------------------------------------------------------------------------
def test_bp1_the_middleware_matches_what_the_router_matches():
    # c1 -- Bulbe, the switch engaged: a route that only shares the search
    # prefix's letters reaches its handler; the search route and a route
    # below it are refused.
    with _middleware("bulbe", engaged=True) as w:
        status, body = _ask(w.app, "GET", "/api/searchx")
        assert (status, w.calls) == (200, ["GET /api/searchx"]), (status, body, w.calls)
        for path in ("/api/search", "/api/search/config"):
            del w.calls[:]
            status, body = _ask(w.app, "GET", path)
            assert status == 403 and w.calls == [], (path, status, body, w.calls)
        assert w.mw._matches("/api/search", "/api/search") is True
        assert w.mw._matches("/api/searchx", "/api/search") is False

        # c1b -- mounted below a root path, the path the router reads is the
        # one the middleware decides on.
        del w.calls[:]
        status, body = _ask(w.app, "GET", "/api/search", root_path="/oo")
        assert status == 403 and w.calls == [], (status, body, w.calls)

    # c2 -- every entry of the three lists names a mounted route, and every
    # web-only route is exactly one mounted route with that method.
    routes, unread = _route_census(_API)
    assert unread == [], unread
    assert len(routes) >= 500, len(routes)
    fixture, fixture_unread = _route_census_of(
        "from fastapi import APIRouter\nrouter = APIRouter(prefix='/api/x')\n\n"
        "@router.post('/y')\ndef y():\n    return None\n"
    )
    assert (fixture, fixture_unread) == ([("POST", "/api/x/y")], [])
    paths = [path for _method, path in routes]
    with _middleware("bulbe") as w:
        lists = {
            name: getattr(w.mw, name)
            for name in ("_ALWAYS_ALLOWED_PREFIXES", "_SEARCH_PREFIXES", "_PLUGIN_INSTALL_PREFIXES")
        }
        web_only = getattr(w.mw, "_WEB_ROUTES", None)
    for name, entries in lists.items():
        assert len(entries) >= 1, name
        dead = [entry for entry in entries if not any(_segment_match(path, entry) for path in paths)]
        assert dead == [], (name, dead)
    assert web_only is not None, "absence: the middleware names no web-only routes"
    assert len(web_only) >= 1
    for entry in web_only:
        assert routes.count(tuple(entry)) == 1, (entry, routes.count(tuple(entry)))


def _route_census_of(text):
    """The census of one fixture module, to show the census can find a route."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "routes_fixture.py").write_text(text, encoding="utf-8")
        return _route_census(Path(folder))


# ---------------------------------------------------------------------------
# BP2 -- the web-only routes are refused in Bulbe before their handler
# ---------------------------------------------------------------------------
def test_bp2_the_web_only_routes_are_refused_in_bulbe_before_their_handler():
    # c1 -- Bulbe, a local client with its session cookie: each web-only route
    # answers 403 in the web gate's words, and no handler runs.
    with _middleware("bulbe") as w:
        if w.wg is not None:
            assert w.wg._REFUSALS["mode"] == _WEB_MODE_TEXT
        for method, path in _WEB_ONLY:
            status, body = _ask(w.app, method, path)
            assert w.calls == [], (path, status, w.calls)
            assert status == 403, (path, status, body)
            assert body == {"detail": _WEB_MODE_TEXT, "mode": "bulbe", "restriction": "web_refused"}, (path, body)

        # c2 -- the marketplace listing is not a web-only route.
        status, body = _ask(w.app, "GET", "/api/plugins/marketplace")
        assert (status, w.calls) == (200, ["GET /api/plugins/marketplace"]), (status, body, w.calls)

    # c2 -- Daily: each web-only route reaches its handler once.
    with _middleware("daily") as w:
        for method, path in _WEB_ONLY:
            status, body = _ask(w.app, method, path)
            assert status == 200, (path, status, body)
        assert w.calls == [f"{method} {path}" for method, path in _WEB_ONLY], w.calls


# ---------------------------------------------------------------------------
# BP3 -- the always-allowed list admits the health route exactly
# ---------------------------------------------------------------------------
def test_bp3_the_always_allowed_list_admits_the_health_route_exactly():
    # c1 -- Bulbe, a client that is not this machine: a route below the health
    # route is refused before its handler, and the health route is admitted.
    with _middleware("bulbe") as w:
        status, body = _ask(w.app, "POST", "/api/health/benchmarks", client=_REMOTE)
        assert status == 403 and w.calls == [], (status, body, w.calls)
        status, body = _ask(w.app, "GET", "/api/health", client=_REMOTE)
        assert (status, w.calls) == (200, ["GET /api/health"]), (status, body, w.calls)


# ---------------------------------------------------------------------------
# BP4 -- the plugin allowlist reads the plugin from the path the router reads
# ---------------------------------------------------------------------------
def _allowlist(asked):
    """The plugin allowlist: it records every plugin it is asked about and allows one."""
    allow = types.ModuleType(_ALLOW)
    allow.plugin_allowlist_manager = SimpleNamespace(
        is_allowed=lambda plugin_id: asked.append(plugin_id) or plugin_id == "allowed-plugin",
    )
    return allow


def test_bp4_the_plugin_allowlist_reads_the_plugin_from_the_path_the_router_reads():
    asked = []
    with _middleware("bulbe", seeded={_ALLOW: _allowlist(asked)}) as w:
        # c2 -- witness: at the root, the plugin is the path's segment after
        # the install route, and an allowed one reaches its handler.
        status, body = _ask(w.app, "POST", "/api/plugins/install/allowed-plugin")
        assert asked == ["allowed-plugin"], asked
        assert (status, w.calls) == (200, ["POST /api/plugins/install/allowed-plugin"]), (status, body, w.calls)

        # c1 -- below a mount prefix, the same plugin is asked about and admitted.
        del asked[:], w.calls[:]
        status, body = _ask(w.app, "POST", "/api/plugins/install/allowed-plugin", root_path="/oo")
        assert asked == ["allowed-plugin"], asked
        assert (status, w.calls) == (200, ["POST /api/plugins/install/allowed-plugin"]), (status, body, w.calls)

        # c3 -- a plugin the list does not allow stays refused below a mount
        # prefix, before any handler.
        del asked[:], w.calls[:]
        status, body = _ask(w.app, "POST", "/api/plugins/install/other-plugin", root_path="/oo")
        assert asked == ["other-plugin"], asked
        assert status == 403 and w.calls == [], (status, body, w.calls)
        assert body.get("restriction") == "plugin_not_allowed", body
