#!/usr/bin/env python3
"""Shared support for the contracts of the garden's API routes: the window, the stand-ins, the client.

A route contract opens the platform window with the garden's API modules
loaded after it (``open_api``): the schemas, the platform's auth routes
(whose user dependency the garden's router carries) and the router itself,
under a stand-in ``opti_oignon.api`` package. What those modules reach for
in the platform is seeded here and nowhere else:

* ``opti_oignon.api.deps`` -- ``AUTH_AVAILABLE`` and ``auth_manager``, an
  ``AuthStandIn`` whose ``single_user_mode`` counts its reads and whose
  ``validate_token`` answers a payload of ``_allium_store_support.principal``
  for a token it was given (``auth=None``: no auth module at all);
* ``opti_oignon.emergency_stop`` -- ``is_stopped``, a ``StopStandIn`` that
  counts its calls and answers a value, or raises one;
* ``opti_oignon.security_mode`` -- only when asked (``bulbe=True``, or a
  module of the caller's own), and then taken out of the blocked names.
  Otherwise the platform's mode module stays proven unreachable, and the auth
  routes take their ``ImportError`` path for it.

``app`` mounts the router on a bare FastAPI application; ``client`` is a
``TestClient`` that never raises a server exception, used as a context
manager so its portal thread is gone before a contract counts threads;
``raw`` sends one ASGI request with exactly the raw headers given (no Host,
two Hosts). ``StandInGarden`` answers a fixed ``Look`` (or raises) and
builds nothing; ``GardenFactory`` counts how often the router asks for a
garden; ``api_garden`` is a real ``Garden`` over a test's store seams, wired
as the API's is (no attended verb, the API's view cap, a caller required).
``Recorder`` keeps what every engine request asked and what it answered;
``settings_file`` points the settings reader and the router at a temporary
``allium.yaml``.

Local-only (the public distribution ships no tests).
"""

import asyncio
import json
import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_store_support as support  # noqa: E402
from _allium_window import open_allium  # noqa: E402

BASE = "http://127.0.0.1:8001"
CLIENT = ("127.0.0.1", 50000)
STATUS = "/api/allium/status"
# The platform's synthetic principal in single-user mode, outside Bulbe: no ``iat``, no ``exp``.
SYNTHETIC = {"sub": "local", "username": "local", "role": "admin", "type": "access"}
TOKEN_COOKIE = "oo_access_token"


class AuthStandIn:
    """The auth manager as the platform's user dependency reads it; every read and validation counted.

    ``tokens`` maps a token to ``(sub, now)``: a known token validates to
    ``support.principal(sub, now)``, anything else to ``None``.
    """

    def __init__(self, single_user, tokens=None):
        self.single_user = single_user
        self.tokens = dict(tokens or {})
        self.reads = 0
        self.validations = 0
        self._lock = threading.Lock()

    @property
    def single_user_mode(self):
        with self._lock:
            self.reads += 1
        return self.single_user

    def validate_token(self, token):
        with self._lock:
            self.validations += 1
        found = self.tokens.get(token)
        return None if found is None else support.principal(*found)

    @property
    def calls(self):
        return self.reads + self.validations


class StopStandIn:
    """``emergency_stop.is_stopped``: a value (or an exception, raised), and a count of the calls."""

    def __init__(self, value=False):
        self.value = value
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


def deps_module(manager):
    module = types.ModuleType("opti_oignon.api.deps")
    module.AUTH_AVAILABLE = manager is not None
    module.auth_manager = manager
    return module


def stop_module(stop):
    module = types.ModuleType("opti_oignon.emergency_stop")
    module.is_stopped = stop
    return module


def bulbe_module(active=True):
    """A stand-in ``opti_oignon.security_mode`` whose ``is_bulbe`` answers its ``active`` attribute."""
    module = types.ModuleType("opti_oignon.security_mode")
    module.active = active
    module.is_bulbe = lambda: module.active
    return module


class ApiWindow(support.Platform):
    """The modules of an API window: the platform, the router, its schemas, the auth routes and the stand-ins."""

    def __init__(self, loaded, *, deps, manager, stop):
        super().__init__(loaded)
        self.routes = loaded["opti_oignon.api.routes_allium"]
        self.schemas = loaded["opti_oignon.api.schemas_allium"]
        self.auth = loaded["opti_oignon.api.routes_auth"]
        self.deps = deps
        self.manager = manager
        self.stop = stop


def open_api(monkeypatch, tmp_path, *, auth="single", stopped=None, bulbe=False, tokens=None, security_mode=None,
             seeded=None, blocked=None, **kw):
    """One window with the platform and the garden's API modules loaded; ``(ApiWindow, restore)``.

    ``auth`` is ``"single"`` (a single-user auth manager), ``"multi"`` (a
    login is required) or ``None`` (no auth module). ``stopped`` is what the
    emergency stop answers (``False`` when not given). ``bulbe=True`` seeds a
    security mode whose ``is_bulbe`` is true; ``security_mode`` seeds a module
    of the caller's own instead. ``kw`` goes to ``open_allium``.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(Path(tmp_path) / "xdg"))
    manager = None if auth is None else AuthStandIn(auth == "single", tokens)
    if auth not in (None, "single", "multi"):
        raise ValueError(f"unknown auth: {auth}")
    deps = deps_module(manager)
    stop = StopStandIn(False if stopped is None else stopped)
    given = {"opti_oignon.api.deps": deps, "opti_oignon.emergency_stop": stop_module(stop)}
    mode = security_mode if security_mode is not None else (bulbe_module(True) if bulbe else None)
    if mode is not None:
        given["opti_oignon.security_mode"] = mode
    given.update(seeded or {})
    names = support.BLOCKED if blocked is None else tuple(blocked)
    names = tuple(name for name in names if name not in given)
    loaded, restore = open_allium(native=False, platform=True, api=True, blocked=names, seeded=given, **kw)
    return ApiWindow(loaded, deps=deps, manager=manager, stop=stop), restore


def app(w):
    """A bare FastAPI application with the garden's router, and nothing else."""
    from fastapi import FastAPI

    application = FastAPI()
    application.include_router(w.routes.router)
    return application


def client(w, base=BASE, client=CLIENT, application=None):
    """A ``TestClient`` of ``app(w)`` (or ``application``) that answers a server fault as a response."""
    from fastapi.testclient import TestClient

    return TestClient(application if application is not None else app(w), base_url=base,
                      raise_server_exceptions=False, client=client)


def raw(application, headers, method="GET", path=STATUS):
    """One ASGI request with exactly ``headers`` (``[(bytes, bytes)]``); ``(status, body as JSON or bytes)``."""
    sent = []
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode("ascii"), "query_string": b"",
             "headers": list(headers), "client": CLIENT, "server": ("127.0.0.1", 8001), "scheme": "http",
             "http_version": "1.1", "root_path": "", "app": application}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(application(scope, receive, send))
    status = sent[0]["status"]
    body = b"".join(message.get("body", b"") for message in sent[1:])
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


class StandInGarden:
    """A garden that answers ``look`` with a fixed look (or raises ``raises``), records callers, builds nothing."""

    def __init__(self, look=None, raises=None):
        self.answer = look
        self.raises = raises
        self.callers = []
        self.closed = 0

    def look(self, caller=None, cap=None):
        self.callers.append(caller)
        if self.raises is not None:
            raise self.raises
        return self.answer

    def close(self):
        self.closed += 1


class GardenFactory:
    """A ``garden_factory``: each call counted, answered by ``make()``."""

    def __init__(self, make):
        self.make = make
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
        return self.make()


class ApiStores:
    """A garden's store factory over a test's seams: the stores it built, and how often it was asked."""

    def __init__(self, w, given):
        self.w = w
        self.given = given
        self.calls = 0
        self.stores = []

    def __call__(self):
        self.calls += 1
        built = self.w.store.Store(**self.given)
        self.stores.append(built)
        return built


def api_garden(w, given, *, law="fixture", stopped=None):
    """A ``Garden`` wired as the API's: the settings' switch, no attended verb, the API's view cap, a caller required.

    Its store is built on the test's seams ``given`` (``garden.store_factory``
    counts it); ``_allium_garden_support.Factory`` has no view cap.
    """
    return w.service.Garden(store_factory=ApiStores(w, given), switch=w.settings.switch, stopped=stopped,
                            attended=None, law=law, view_cap=w.service.api_view_cap, caller_required=True)


class Recorder:
    """Wraps the engine seam: ``(op, budget, to, work, done)`` of every request, in order; ``close`` puts it back.

    ``call`` is the engine answered through (default: the one in place).
    """

    def __init__(self, w, call=None):
        self.w = w
        self.real = w.engine.call
        self.call = call if call is not None else self.real
        self.requests = []
        self._lock = threading.Lock()
        w.engine.call = self

    def __call__(self, request):
        answer = self.call(request)
        asked = self.w.wire.parse(bytes(request))
        said = self.w.wire.parse(bytes(answer))
        with self._lock:
            self.requests.append((asked.get("op"), asked.get("budget"), asked.get("to"), said.get("work"),
                                  said.get("done")))
        return answer

    def mark(self):
        return len(self.requests)

    def since(self, mark):
        return list(self.requests[mark:])

    def close(self):
        self.w.engine.call = self.real


def work(requests):
    """The work the engine answered over the ``advance`` requests of ``requests``."""
    return sum(entry[3] or 0 for entry in requests if entry[0] == "advance")


def settings_file(w, tmp_path, text, name="allium.yaml"):
    """Write ``text`` as a temporary settings file; point the settings reader and the router at it; its path."""
    folder = Path(tmp_path) / "settings"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    w.settings.config_file = lambda: path
    w.routes.ALLIUM_YAML = path
    return path


def get(w, headers=None, base=BASE, peer=CLIENT, application=None, method="GET"):
    """One request to the status route through a fresh client (``peer`` its address); the response."""
    with client(w, base=base, client=peer, application=application) as http:
        return http.request(method, STATUS, headers=headers or {})


__all__ = ["BASE", "CLIENT", "STATUS", "SYNTHETIC", "TOKEN_COOKIE", "AuthStandIn", "StopStandIn", "ApiWindow",
           "open_api", "app", "client", "raw", "StandInGarden", "GardenFactory", "ApiStores", "api_garden",
           "Recorder", "work", "settings_file", "get", "deps_module", "stop_module", "bulbe_module"]
