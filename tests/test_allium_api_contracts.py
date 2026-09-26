#!/usr/bin/env python3
"""Contracts for the garden's routes in the API: the router, the projection it serves, and its boundaries.

The API serves the componion read only, through one router under
``/api/allium`` (``opti_oignon/api/routes_allium.py``) and its own schemas
(``opti_oignon/api/schemas_allium.py``). Every garden route carries a closed
Host/Origin/Fetch-Metadata check first and the platform's explicit user
dependency after it; the caller of the garden is built from that
dependency's principal alone; every refusal is a closed body of the garden's
own lines; a view is capped by the API's own bound.

  * AF3 -- every garden route carries the host check and the user
    dependency, and the surface comes from the transport alone: the walk of
    the router, the auth answers (unavailable, no token, single-user, a
    session by cookie and by Bearer, Bulbe with one account), the synthetic
    principal's keys, what the route module reads and calls (no rebound
    principal, no other handler parameter, no part of a request read past
    the host check, with a planted witness), a query, header or cookie that
    names another account changing nothing, an API garden that refuses a
    missing caller, and no middleware that waves the route through.
  * AF4 -- the garden's schemas live in their own module and match the
    served projection, at every level, for every status.
  * AF5 -- the API's garden is built once, on first use with the switch on;
    the health key and the switched-off route build nothing; the router's
    readers of the switch and of the listed names agree with the garden's;
    the production garden takes the auth manager's own single-user rule and
    the emergency stop; the shutdown closes what was built.
  * AF6 -- a view under the API cap does at most the cap plus two awake days
    of work, and says it is catching up; the cap settings fall back by name;
    a cap below one awake day is raised to it, so the view after a terminal
    write is current, as the served line says.
  * AF7 -- no API handler asks the engine past the cap, catches up or
    writes; a frozen view and a native one keep their bounds and write
    nothing either.
  * AF9 -- a rebound Host, a foreign Origin and a page of another site are
    refused before anything runs; the loopback names, the listed names (an
    address spelled otherwise than as its dotted quad refuses the list), the
    platform's redirect of a trailing slash, and the Vite proxy's line.
  * AF15 -- every word the API serves passes the nets, and no request value
    or exception text is repeated: the API's own lines, every served form
    (the doctrine last), the OpenAPI words, the route class's bodies and its
    log records, and a refusal whose catalogue cannot be imported.
  * AF16 -- a long-lived API process is safe to call from many threads: one
    garden under eight threads, the engine handshake (a raising loader asked
    once), the router's one garden under eight requests, the mode reading
    (on a stand-in, then on the platform's own module: its tamper evidence
    written once per change of the files, never per reading), the settings
    caches.
  * AF17 -- with the switch off, the API imports nothing of the being, in a
    fresh process behind the data firewall: called directly, and through the
    router for a listed name; a refusal imports the catalogue and its nets
    only.

Local-only (the public distribution ships no tests). The platform and the
router load through the shared isolation window with the platform's
configuration, keys, mode, audit log and user modules proven unreachable;
the auth manager, the emergency stop and every store seam are stand-ins
(``tests/_allium_api_support.py``); the child of AF17 imports this tree.
"""

import ast
import inspect
import itertools
import json
import logging
import os
import re
import sys
import threading
import time
import types
import typing
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_api_support as api  # noqa: E402
import _allium_garden_support as garden  # noqa: E402
import _allium_store_support as support  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_af3_every_garden_route_carries_the_host_check_and_the_user_dependency_and_the_surface_is_the_transports": 2.0,
    "test_af4_the_garden_schemas_live_in_their_own_module_and_match_the_served_projection": 2.0,
    "test_af5_the_api_garden_is_built_once_on_first_use_and_the_health_key_and_the_off_route_build_nothing": 2.0,
    "test_af6_a_view_under_the_api_cap_does_at_most_the_cap_plus_two_awake_days_of_work": 2.0,
    "test_af7_no_api_handler_asks_the_engine_past_the_cap_catches_up_or_writes": 2.0,
    "test_af9_a_rebound_host_a_foreign_origin_and_a_page_of_another_site_are_refused_before_anything_runs": 2.0,
    "test_af15_every_word_the_api_serves_passes_the_nets_and_no_request_value_or_exception_text_is_repeated": 2.0,
    "test_af16_a_long_lived_api_process_is_safe_to_call_from_many_threads": 2.0,
    "test_af17_with_the_switch_off_the_api_imports_nothing_of_the_being": 2.0,
}
PACKAGE = REPO / "opti_oignon"
TESTS = REPO / "tests"
WALL = support.WALL
DAY = 1440
DAY_S = DAY * 60
MAX_INT = (1 << 53) - 1
ON = "enabled: true\n"
OFF = "enabled: false\n"
CAPPED = ON + "api:\n  python_cap: 5000\n  native_cap: 50000\n"

# The API's own lines, in their order, and the codes of its refusals.
WEB_KEYS = ("web.host", "web.origin", "web.site", "web.sign_in", "web.auth_unavailable", "web.request", "web.fault",
            "web.catching_up")
REFUSAL_CODES = ("host", "origin", "site", "sign_in", "auth_unavailable", "request", "fault")
REFUSAL_HTTP = {"host": 403, "origin": 403, "site": 403, "sign_in": 401, "auth_unavailable": 503, "request": 422,
                "fault": 500}
# The statuses the service serves with labels, and the ones served with no being and no habitat.
LABELLED = ("unavailable", "sealed_bulbe", "missing", "retired_prototype", "unreadable")
BARE = ("disabled", "stopped", "ready", "awaiting_soil")
# The shared stop line as it read before the API served it: it presupposes an onion.
STOPPED_BEFORE = ("The emergency stop is on in this server: it computes nothing for this onion until the stop is "
                  "lifted. Its time runs on.")
# A name of the longest length a name may have.
LONG_NAME = "Abcdefghij Klmnopqrst Uvwxyz-012"
# The garden's public methods (the route module may call only ``look``) and its models, by the projection's keys.
GARDEN_METHODS = ("look", "sow", "sow_card", "act", "name", "name_card", "verify", "laws", "laws_diff", "laws_apply",
                  "laws_pin", "laws_unpin", "resume", "finish", "confirm_share")
NESTED = {"as_of": "AlliumAsOf", "being": "AlliumBeing", "habitat": "AlliumHabitat", "law": "AlliumLaw"}
MIDDLEWARE_PREFIXES = (("auth_middleware.py", "_PUBLIC_EXACT"), ("auth_middleware.py", "_PUBLIC_PREFIXES"),
                       ("csrf_middleware.py", "_CSRF_EXEMPT_PREFIXES"),
                       ("security_mode_middleware.py", "_ALWAYS_ALLOWED_PREFIXES"))


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


@pytest.fixture
def w(monkeypatch, tmp_path):
    window, restore = api.open_api(monkeypatch, tmp_path, auth="single")
    try:
        yield window
    finally:
        restore()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _code(template, **values):
    """A child's code with its placeholders filled by the Python literals of ``values``."""
    out = template
    for name, value in values.items():
        out = out.replace("__" + name.upper() + "__", repr(value))
    assert "__" not in out.replace("__name__", "").replace("__init__", ""), "every placeholder is filled"
    return out


def _refusal(w, code):
    return {"detail": w.wording.web("web." + code).text, "refusal": code}


def _web_caller(w, principal=None):
    return w.service.transport("web", principal=dict(api.SYNTHETIC if principal is None else principal))


def _ordered(w, labels):
    order = w.describe.LABEL_ORDER
    return tuple(sorted(set(labels), key=order.index))


def _local(wall):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(wall))


def _looks(w):
    """Synthetic looks of every status and, for the statuses served with labels, every label combination.

    ``(look, what)``: ``what`` names the case. A glass jar's label goes with a
    jar's habitat; ``catching_up`` and ``frozen`` go only to an alive being,
    never together; a mode label is one of ``bulbe`` and ``mode_unknown``.
    """
    out = [(garden.look(w, status), status) for status in w.service.STATUSES]
    out.append((garden.look(w, "disabled", reason="unreadable"), "disabled.unreadable"))
    out.append((garden.look(w, "stopped", reason="unknown"), "stopped.unknown"))
    cores = [core for n in range(4) for core in itertools.combinations(("prototype", "retired", "glass_jar"), n)]
    modes = ((), ("bulbe",), ("mode_unknown",))
    for status in LABELLED:
        for core, mode in itertools.product(cores, modes):
            labels = _ordered(w, core + mode)
            jar = "glass_jar" in labels
            layer = "open" if not mode else ("sealed" if jar else "bulbe")
            fields = {"labels": labels}
            if status in ("sealed_bulbe", "retired_prototype"):
                fields["habitat"] = ("jar" if jar else "pot", layer)
            out.append((garden.look(w, status, **fields), status + ":" + ",".join(labels)))
            if status == "unavailable":
                opened = dict(fields, habitat=("jar" if jar else "pot", layer), reason="busy",
                              being=garden.being_info(w, soil="glass" if jar else "encrypted"))
                out.append((garden.look(w, status, **opened), "unavailable.view:" + ",".join(labels)))
    for core, mode, late in itertools.product(cores, modes, ((), ("catching_up",), ("frozen",))):
        labels = _ordered(w, core + mode + late)
        jar = "glass_jar" in labels
        layer = "open" if not mode else ("sealed" if jar else "bulbe")
        felt = garden.felt(w, labels=labels, jar=jar, layer=layer)
        look = garden.look(w, "alive", labels=labels, felt=felt, habitat=("jar" if jar else "pot", layer),
                           being=garden.being_info(w, soil="glass" if jar else "encrypted"))
        out.append((look, "alive:" + ",".join(labels)))
    labels = _ordered(w, ("prototype", "retired", "glass_jar", "catching_up", "mode_unknown"))
    felt = garden.felt(w, labels=labels, jar=True, layer="sealed", name=LONG_NAME, day=36500)
    out.append((garden.look(w, "alive", labels=labels, felt=felt, habitat=("jar", "sealed"),
                            being=garden.being_info(w, soil="glass", name=LONG_NAME)), "alive:largest"))
    return out


def _literal(annotation):
    """The values of the ``Literal`` inside ``annotation`` (through ``Optional`` and ``list``), or ``None``."""
    if typing.get_origin(annotation) is typing.Literal:
        return set(typing.get_args(annotation))
    for arg in typing.get_args(annotation):
        found = _literal(arg)
        if found is not None:
            return found
    return None


def _flat(dependant):
    out = []
    for sub in dependant.dependencies:
        out.append(sub.call)
        out.extend(_flat(sub))
    return out


def _walk_router(router, routes, auth):
    """``(paths walked, paths flagged)``: a route is flagged unless it is a ``GardenRoute`` whose dependencies
    start with the host check and contain the platform's user dependency, under ``/api/allium``."""
    from fastapi.routing import APIRoute

    walked, flagged = [], []
    for route in router.routes:
        path = getattr(route, "path", repr(route))
        walked.append(path)
        if not isinstance(route, APIRoute):
            flagged.append(path)
            continue
        calls = _flat(route.dependant)
        if not (isinstance(route, routes.GardenRoute) and calls and calls[0] is routes.host_origin
                and any(call is auth._get_current_user for call in calls) and path.startswith("/api/allium/")):
            flagged.append(path)
    return walked, flagged


def _routes_tree():
    return ast.parse((PACKAGE / "api" / "routes_allium.py").read_text(encoding="utf-8"))


def _enclosing(tree):
    """``{id(node): the innermost function that holds it, or None}`` for every node of ``tree``."""
    owner = {}

    def visit(node, name):
        for child in ast.iter_child_nodes(node):
            owner[id(child)] = name
            visit(child, child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else name)

    visit(tree, None)
    return owner


def _functions_under(tree, name):
    """Every node inside the function ``name``, nested ones included."""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            inside.update(id(child) for child in ast.walk(node))
    return inside


# What a request carries beside the principal: never read by the route module outside the host check.
REQUEST_PARTS = ("query_params", "path_params", "url", "client", "body", "json", "form", "stream", "cookies",
                 "headers", "scope")
ROUTE_VERBS = ("get", "post", "put", "patch", "delete", "head", "options", "api_route")


def _request_reads(tree):
    """``[(kind, where, what)]``: every way the route module could let a request choose what the garden reads.

    * ``rebinds`` -- a function rebinds the parameter it takes from the user
      dependency (an assignment, a walrus, a loop, a ``with`` or a ``del``);
    * ``parameter`` -- a route handler takes a parameter that is not the user
      dependency (a ``Request``, a header, a query, a cookie, a body);
    * ``reads`` -- a part of a request is read outside the host check.
    """
    found = []
    inside = _functions_under(tree, "host_origin")
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        defaults = ([None] * (len(fn.args.posonlyargs + fn.args.args) - len(fn.args.defaults))
                    + list(fn.args.defaults) + list(fn.args.kw_defaults))
        dependency = {argument.arg for argument, default in zip(arguments, defaults)
                      if isinstance(default, ast.Call) and getattr(default.func, "id", "") == "Depends"
                      and default.args and getattr(default.args[0], "id", "") == "_get_current_user"}
        routed = any(isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                     and decorator.func.attr in ROUTE_VERBS for decorator in fn.decorator_list)
        if routed:
            extra = [argument.arg for argument in arguments + [fn.args.vararg, fn.args.kwarg]
                     if argument is not None and argument.arg not in dependency]
            found.extend(("parameter", fn.name, name) for name in extra)
        for node in ast.walk(fn):
            bound = []
            if isinstance(node, (ast.Assign, ast.Delete)):
                bound = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
                bound = [node.target]
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                bound = [item.optional_vars for item in node.items if item.optional_vars is not None]
            names = {name.id for target in bound for name in ast.walk(target) if isinstance(name, ast.Name)}
            found.extend(("rebinds", fn.name, name) for name in sorted(names & dependency))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in REQUEST_PARTS and id(node) not in inside:
            found.append(("reads", node.lineno, node.attr))
    return found


def _middleware_prefixes():
    """``{(file, name): values}`` of the middlewares' public and exempt path lists, read from their source."""
    out = {}
    for file, name in MIDDLEWARE_PREFIXES:
        tree = ast.parse((PACKAGE / "api" / file).read_text(encoding="utf-8"))
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                node, ast.AnnAssign) else []
            if any(getattr(target, "id", None) == name for target in targets):
                value = node.value
                if isinstance(value, ast.Call) and getattr(value.func, "id", "") in ("frozenset", "set", "tuple"):
                    value = value.args[0]
                out[(file, name)] = tuple(ast.literal_eval(value))
    return out


def _listing(folder):
    """``{file: sha256}`` of every file under ``folder``."""
    folder = Path(folder)
    return {str(file.relative_to(folder)): support.sha256_file(file) for file in sorted(folder.rglob("*"))
            if file.is_file()}


def _checkpoints(path):
    return support.read(path, "SELECT t, laws, through, state_hash FROM checkpoints ORDER BY t")


def _laws_named(w, given):
    """The laws a being's facts name: its genesis law and every law update's target."""
    target = w.store.Store(**given)
    try:
        names = set()
        for _seq, _eid, fact in target.open("local").in_order():
            if fact["kind"] == "genesis":
                names.add(fact["body"]["laws"]["name"])
            elif fact["kind"] == "evolve":
                names.add(fact["body"]["to"]["name"])
        return names
    finally:
        target.close()


def _sow(w, given, **kw):
    """Sow the one seed of a test store from an attended terminal."""
    sower = garden.garden(w, given, attended=True, law="fixture")()
    try:
        return sower.sow(**kw)
    finally:
        sower.close()


class _Counting:
    def __init__(self, real):
        self.real = real
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
        return self.real(*args, **kwargs)


def _one_assignment(tree, function):
    """How many statements of ``function`` write a module-level cache (an item or attribute of a module name,
    or a module name declared global)."""
    module_names = set()
    for node in tree.body:
        for target in (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []):
            if isinstance(target, ast.Name):
                module_names.add(target.id)
    [fn] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
    declared = {name for node in ast.walk(fn) if isinstance(node, ast.Global) for name in node.names}
    count = 0
    for node in ast.walk(fn):
        targets = (node.targets if isinstance(node, ast.Assign) else
                   [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        for target in targets:
            base = target
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and ((base is not target and base.id in module_names)
                                               or base.id in declared):
                count += 1
                break
    return count


# ---------------------------------------------------------------------------
# AF15 -- every served word passes the nets; nothing of a request or an exception is repeated
# ---------------------------------------------------------------------------
def _masked(line, name):
    if line["key"] == "identity" and name:
        return line["text"].replace(name, "x")
    return line["text"]


def test_af15_every_word_the_api_serves_passes_the_nets_and_no_request_value_or_exception_text_is_repeated(
        w, tmp_path, caplog):
    e, wording = w.ethics, w.wording
    printable = re.compile(r"[ -~]*")

    # The API's own lines: a closed dict, counted in its literal, each through the nets.
    assert tuple(wording.WEB) == WEB_KEYS
    tree = ast.parse((PACKAGE / "allium" / "wording.py").read_text(encoding="ascii"))
    literal = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(getattr(target, "id", "") == "WEB" for target in node.targets)]
    assert len(literal) == 1 and isinstance(literal[0], ast.Dict)
    assert len(literal[0].keys) == len(wording.WEB) == len(WEB_KEYS)
    for key, text in wording.WEB.items():
        assert e.check(text) == () and printable.fullmatch(text), (key, e.check(text))
        line = wording.web(key)
        assert (line.key, line.text) == (key, text)
    assert e.check(wording.WEB_FALLBACK) == () and printable.fullmatch(wording.WEB_FALLBACK)
    with pytest.raises(KeyError):
        wording.web("web.nothing")
    wording.WEB["web.planted"] = "It misses you."
    try:
        with pytest.raises(wording.CopyRefused) as refused:
            wording.web("web.planted")
        assert refused.value.key == "web.planted", "witness: a line that fails the nets is never served"
    finally:
        del wording.WEB["web.planted"]

    # Every form, over every status and every label combination the service serves.
    forms = [(look, what, w.describe.web_fields(look)) for look, what in _looks(w)]
    doctrine = wording.say("doctrine").text
    for _look, what, payload in forms:
        if payload["status"] != "disabled":
            # The being is always called a simulation: the doctrine closes every form that says anything.
            assert payload["lines"] and payload["lines"][-1] == {"key": "doctrine", "text": doctrine}, (
                what, [line["key"] for line in payload["lines"]])
    assert sum(1 for _l, _w, payload in forms if payload["status"] != "disabled") > 100, "presence: the forms said"
    met = {"statuses": set(), "jar": 0, "prototype": 0, "catching_up": 0, "bare": 0, "largest": 0}
    checked = set()
    for look, what, payload in forms:
        met["statuses"].add(look.status)
        keys = [line["key"] for line in payload["lines"]]
        name = look.felt.name if look.felt is not None else None
        for line in payload["lines"]:
            text = _masked(line, name)
            if text not in checked:
                checked.add(text)
                assert e.check(text) == (), (what, line, e.check(text))
            assert printable.fullmatch(line["text"]), (what, line)
        assert "path.line" not in keys, what
        if look.status == "disabled":
            assert payload["lines"] == [], what
        elif look.status == "alive":
            assert "identity" in keys and "conditions" in keys, what
            assert not any(key.startswith("status.") for key in keys), what
        else:
            assert any(key.startswith("status.") for key in keys), ("the headline is a status line", what, keys)
        jar = "glass_jar" in look.labels or (payload["habitat"] or {}).get("container") == "jar"
        if jar:
            met["jar"] += 1
            assert "label.glass_jar" in keys, what
        if "prototype" in look.labels:
            met["prototype"] += 1
            assert any(key.startswith("label.prototype") or key == "label.retired" for key in keys), what
        web_keys = [key for key in keys if key.startswith("web.")]
        if "catching_up" in look.labels and look.status == "alive":
            met["catching_up"] += 1
            assert web_keys == ["web.catching_up"], what
            assert keys[keys.index("label.catching_up") + 1] == "web.catching_up", (what, keys)
        else:
            assert web_keys == [], (what, keys)
        if look.status in BARE or (look.status == "unavailable" and look.being is None and look.habitat is None):
            met["bare"] += 1
            assert not any("this onion" in line["text"] for line in payload["lines"]), (what, payload["lines"])
        assert e.check_fields(payload) == (), (what, e.check_fields(payload))
        compact = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("ascii")
        assert len(compact) <= 2048, (what, len(compact))
        if what == "alive:largest":
            met["largest"] = len(compact)
    assert met["statuses"] == set(w.service.STATUSES)
    assert met["jar"] and met["prototype"] and met["catching_up"] and met["bare"] >= 6, met
    assert met["largest"] > 1024, ("presence: the largest form is large", met["largest"])
    saved = wording.TEMPLATES["status.stopped"]
    wording.TEMPLATES["status.stopped"] = STOPPED_BEFORE
    try:
        planted = w.describe.web_fields(garden.look(w, "stopped"))
    finally:
        wording.TEMPLATES["status.stopped"] = saved
    assert any("this onion" in line["text"] for line in planted["lines"]), "witness: the old stop line is found"

    # The OpenAPI words: the handler, every model and every field description.
    assert e.check(inspect.getdoc(w.routes.garden_status)) == ()
    models = [value for name, value in vars(w.schemas).items() if name.startswith("Allium") and isinstance(value, type)]
    assert len(models) >= 7, [model.__name__ for model in models]
    descriptions = 0
    for model in models:
        doc = model.__dict__.get("__doc__")
        assert doc and e.check(doc) == (), (model.__name__, doc)
        for field_name, field in model.model_fields.items():
            if field.description is not None:
                descriptions += 1
                assert e.check(field.description) == (), (model.__name__, field_name, e.check(field.description))
    assert descriptions >= 10, descriptions

    # The route class, on a scratch router: a body of the wrong type and a fault repeat nothing.
    from fastapi import APIRouter, FastAPI
    from pydantic import BaseModel

    canary = support.canary(w, "af15")

    class ProbeBody(BaseModel):
        n: int

    scratch = APIRouter(route_class=w.routes.GardenRoute)

    @scratch.post("/probe")
    def probe(body: ProbeBody):
        return {"n": body.n}

    @scratch.get("/boom")
    def boom():
        raise ValueError(canary)

    application = FastAPI()
    application.include_router(scratch)
    caplog.set_level(logging.DEBUG)
    with api.client(w, application=application) as http:
        caplog.clear()
        answer = http.post("/probe", json={"n": canary})
        assert answer.status_code == 422 and answer.json() == _refusal(w, "request"), answer.text
        assert canary not in answer.text and canary not in str(answer.headers)
        refusal_records = [r for r in caplog.records if r.name.startswith("opti_oignon.api")]
        assert refusal_records and all(r.levelno == logging.DEBUG for r in refusal_records), refusal_records
        assert any("request" in r.getMessage() for r in refusal_records)
        caplog.clear()
        answer = http.get("/boom")
        assert answer.status_code == 500 and answer.json() == _refusal(w, "fault"), answer.text
        assert canary not in answer.text and canary not in str(answer.headers)
        faults = [r for r in caplog.records if r.name.startswith("opti_oignon.api") and r.levelno >= logging.WARNING]
        assert faults and any("ValueError" in r.getMessage() for r in faults), caplog.records
        assert not any(canary in r.getMessage() for r in caplog.records)

    # The real route: a fault the service does not map, a line that fails, and a host refused.
    api.settings_file(w, tmp_path, ON)
    stand = api.StandInGarden(raises=RuntimeError(canary))
    w.routes.garden_factory = api.GardenFactory(lambda: stand)
    with api.client(w) as http:
        caplog.clear()
        answer = http.get(api.STATUS)
        assert answer.status_code == 500 and answer.json() == _refusal(w, "fault"), answer.text
        assert canary not in answer.text
        faults = [r for r in caplog.records if r.name.startswith("opti_oignon.api") and r.levelno >= logging.WARNING]
        assert faults and any("RuntimeError" in r.getMessage() for r in faults), caplog.records
        stand.raises = w.wording.CopyRefused("identity")
        caplog.clear()
        answer = http.get(api.STATUS)
        assert answer.status_code == 500 and answer.json() == _refusal(w, "fault"), answer.text
        faults = [r for r in caplog.records if r.name.startswith("opti_oignon.api") and r.levelno >= logging.WARNING]
        assert any("CopyRefused" in r.getMessage() and "identity" in r.getMessage() for r in faults), caplog.records
        stand.raises = RuntimeError(canary)
        real_web = w.wording.web

        def failing(key):
            raise w.wording.CopyRefused(key)

        w.wording.web = failing
        try:
            answer = http.get(api.STATUS)
        finally:
            w.wording.web = real_web
        assert answer.status_code == 500, answer.text
        assert answer.json() == {"detail": w.wording.WEB_FALLBACK, "refusal": "fault"}, answer.text
        caplog.clear()
        answer = http.get(api.STATUS, headers={"Host": canary + ".example"})
        assert answer.status_code == 403 and answer.json() == _refusal(w, "host"), answer.text
        refusal_records = [r for r in caplog.records if r.name.startswith("opti_oignon.api")]
        assert refusal_records and all(r.levelno == logging.DEBUG for r in refusal_records), refusal_records
        assert any("host" in r.getMessage() for r in refusal_records)
        assert not any(canary in r.getMessage() for r in caplog.records)
    assert len(stand.callers) == 3, "the refused request never reached the garden"

    # A refusal whose catalogue cannot even be imported still answers a closed body: the catalogue's own
    # fallback, which the route module holds as a constant equal to it.
    assert w.routes.FALLBACK == w.wording.WEB_FALLBACK
    package = sys.modules["opti_oignon.allium"]
    held, cached = package.__dict__.pop("wording"), sys.modules["opti_oignon.allium.wording"]
    sys.modules["opti_oignon.allium.wording"] = None
    try:
        answer = api.get(w, headers={"Host": "evil.example"})
    finally:
        sys.modules["opti_oignon.allium.wording"] = cached
        package.wording = held
    assert (answer.status_code, answer.json()) == (403, {"detail": w.wording.WEB_FALLBACK, "refusal": "host"}), \
        answer.text


# ---------------------------------------------------------------------------
# AF4 -- the schemas: their own module, the served projection's shape
# ---------------------------------------------------------------------------
def test_af4_the_garden_schemas_live_in_their_own_module_and_match_the_served_projection(w):
    # The module loads with no project module seeded, and imports typing and pydantic only, at module level.
    loaded, restore = isolate(targets={"opti_oignon.api.schemas_allium": source("api", "schemas_allium.py")},
                              packages=("opti_oignon.api",))
    try:
        assert hasattr(loaded["opti_oignon.api.schemas_allium"], "AlliumStatus")
    finally:
        restore()
    tree = ast.parse((PACKAGE / "api" / "schemas_allium.py").read_text(encoding="utf-8"))
    top = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    every = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert top and len(top) == len(every), "imports at module level only"
    named = {node.module if isinstance(node, ast.ImportFrom) else alias.name
             for node in top for alias in node.names}
    assert named == {"typing", "pydantic"}, named
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert classes and all(name.startswith("Allium") for name in classes), classes
    from pydantic import BaseModel

    for name in classes:
        assert issubclass(getattr(w.schemas, name), BaseModel), name

    # No other module of the API defines an Allium model, and the platform's schemas no Allium name.
    platform = ast.parse((PACKAGE / "api" / "schemas.py").read_text(encoding="utf-8"))
    names = [node.name for node in ast.walk(platform) if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    names += [target.id for node in ast.walk(platform) if isinstance(node, ast.Assign)
              for target in node.targets if isinstance(target, ast.Name)]
    names += [alias.asname or alias.name for node in ast.walk(platform) if isinstance(node, ast.ImportFrom)
              for alias in node.names]
    assert len(names) > 50 and not [name for name in names if name.startswith("Allium")]
    scanned = 0
    for path in sorted((PACKAGE / "api").glob("*.py")):
        if path.name == "schemas_allium.py":
            continue
        scanned += 1
        others = [node.name for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                  if isinstance(node, ast.ClassDef) and node.name.startswith("Allium")]
        assert others == [], (path.name, others)
    assert scanned > 50

    # The closed codes are the garden's own.
    s = w.schemas
    status = s.AlliumStatus.model_fields
    being = s.AlliumBeing.model_fields
    assert _literal(status["status"].annotation) == set(w.service.STATUSES)
    assert _literal(status["labels"].annotation) == set(w.describe.LABEL_ORDER)
    assert _literal(status["source"].annotation) == {"simulation"}
    assert _literal(being["life"].annotation) == set(w.describe.LIVES)
    assert _literal(being["light"].annotation) == set(w.describe.LIGHTS)
    assert _literal(being["place"].annotation) == set(w.describe.PLACES)
    assert _literal(being["soil"].annotation) == set(w.describe.SOILS)
    assert _literal(being["season"].annotation) == set(w.describe.SEASON_CODES)
    assert _literal(s.AlliumRefusal.model_fields["refusal"].annotation) == set(w.routes.REFUSAL_STATUS)
    assert set(w.routes.REFUSAL_STATUS) == set(REFUSAL_CODES)
    layers = {w.habitat.layer(soil, mode) for soil in ("encrypted", "glass") for mode in ("daily", "bulbe", "unknown")}
    habitat = s.AlliumHabitat.model_fields
    assert _literal(habitat["container"].annotation) == {pair[0] for pair in layers}
    assert _literal(habitat["layer"].annotation) == {pair[1] for pair in layers}
    description = being["light"].description
    assert "sun" in description and "Not a count" in description, description

    # Every form validates to itself, and the model's fields are the payload's keys at every level.
    stages, met = set(), set()
    for look, what in _looks(w):
        payload = w.describe.web_fields(look)
        assert s.AlliumStatus.model_validate(payload).model_dump() == payload, what
        assert set(status) == set(payload), (what, set(status) ^ set(payload))
        met.add(look.status)
        for key, model in NESTED.items():
            if payload[key] is not None:
                met.add(key)
                fields = set(getattr(s, model).model_fields)
                assert fields == set(payload[key]), (what, key, fields ^ set(payload[key]))
        for line in payload["lines"]:
            assert set(s.AlliumLine.model_fields) == set(line), (what, line)
        if payload["being"] is not None:
            stages.add(payload["being"]["stage"])
    assert met >= set(w.service.STATUSES) | set(NESTED), met
    assert stages and _literal(being["stage"].annotation) == stages, stages
    for reason in (None, "unreadable"):
        assert w.routes.DISABLED == w.describe.web_fields(garden.look(w, "disabled", reason=reason)), reason

    # The OpenAPI of the router alone: the garden's components and FastAPI's own, none qualified by module.
    document = api.app(w).openapi()
    components = set(document.get("components", {}).get("schemas", {}))
    assert {"AlliumStatus", "AlliumRefusal"} <= components, components
    for component in components:
        assert component.startswith("Allium") or component in ("HTTPValidationError", "ValidationError"), component
        assert "__" not in component, component
    from fastapi.routing import APIRoute

    routes = [route for route in w.routes.router.routes if isinstance(route, APIRoute)]
    assert routes
    for route in routes:
        assert route.response_model is not None and route.response_model.__module__ == w.schemas.__name__, route.path


# ---------------------------------------------------------------------------
# AF16 -- a long-lived process, many threads
# ---------------------------------------------------------------------------
def _mode_module(tmp_path):
    """A stand-in security mode: a counting singleton, and a manager class that reads a temporary file."""
    folder = Path(tmp_path) / "mode"
    folder.mkdir(parents=True, exist_ok=True)
    module = types.ModuleType("opti_oignon.security_mode")
    module._SECURITY_YAML = folder / "security.yaml"
    module._LOCKFILE_PATH = folder / ".security_mode_lock"
    module._SECURITY_YAML.write_text("mode: daily\n", encoding="ascii")
    module._LOCKFILE_PATH.write_text("lock\n", encoding="ascii")

    class SecurityModeManager:
        def get_current_mode(self):
            return Path(module._SECURITY_YAML).read_text(encoding="ascii").splitlines()[0].split(":", 1)[1].strip()

        def invalidate_cache(self):
            return None

    class Singleton:
        invalidations = 0
        reads = 0

        def invalidate_cache(self):
            Singleton.invalidations += 1

        def get_current_mode(self):
            Singleton.reads += 1
            return "daily"

    module.SecurityModeManager = SecurityModeManager
    module.security_mode_manager = Singleton()
    module.Singleton = Singleton
    module.is_bulbe = lambda: False
    return module


def test_af16_a_long_lived_api_process_is_safe_to_call_from_many_threads(monkeypatch, tmp_path):
    mode_module = _mode_module(tmp_path)
    w, restore = api.open_api(monkeypatch, tmp_path, auth="single", security_mode=mode_module)
    try:
        # Eight threads look at once through one garden.
        api.settings_file(w, tmp_path, CAPPED)
        given = support.seams(w, tmp_path, suite="af16")
        _sow(w, given)
        given["clock"].advance_days(1)
        served = api.api_garden(w, given)
        caller = _web_caller(w)
        start = threading.Barrier(8)
        payloads, faults = [], []

        def look():
            try:
                start.wait(timeout=30)
                payloads.append(json.dumps(w.describe.web_fields(served.look(caller=caller)), sort_keys=True))
            except BaseException as exc:  # noqa: BLE001 - every fault is reported below
                faults.append(repr(exc))

        threads = [threading.Thread(target=look) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        served.close()
        assert faults == [] and len(payloads) == 8, faults
        assert len(set(payloads)) == 1 and json.loads(payloads[0])["status"] == "alive"
        assert served.store_factory.calls == 1, "one store for the process"

        # The engine handshake is answered once for every thread.
        protocol = w.loaded["opti_oignon.allium.ref.protocol"]
        entered, release = threading.Event(), threading.Event()
        loads = []

        def slow_loader():
            loads.append(1)
            entered.set()
            release.wait(timeout=30)
            return types.SimpleNamespace(allium_engine=protocol.engine_info, allium_call=protocol.call)

        answers = {}
        second_started = threading.Event()

        def ask(name, flag=None):
            if flag is not None:
                flag.set()
            answers[name] = w.engine.native_in_use()

        w.engine.reset()
        real_loader = w.engine._load_native
        w.engine._load_native = slow_loader
        try:
            first = threading.Thread(target=ask, args=("first",))
            first.start()
            assert entered.wait(timeout=30), "the first thread is loading"
            second = threading.Thread(target=ask, args=("second", second_started))
            second.start()
            assert second_started.wait(timeout=30)
            second.join(timeout=0.2)
            release.set()
            first.join(timeout=30)
            second.join(timeout=30)
        finally:
            release.set()
            w.engine._load_native = real_loader
            w.engine.reset()
        assert answers == {"first": True, "second": True}, answers
        assert loads == [1], "the loader ran once"
        # A loader that raises is asked once: its caller sees the error, and every later call the reference.
        attempts = []

        def raising_loader():
            attempts.append(1)
            raise RuntimeError("the native core cannot be loaded")

        w.engine._load_native = raising_loader
        try:
            with pytest.raises(RuntimeError):
                w.engine.native_in_use()
            later = [w.engine.native_in_use(), w.engine.native_in_use()]
        finally:
            w.engine._load_native = real_loader
            w.engine.reset()
        assert later == [False, False] and attempts == [1], (later, attempts)

        # The router builds its garden once, however many requests ask for it at the same time.
        w.routes._held["garden"] = None
        gate = threading.Barrier(8)

        def slow_garden():
            time.sleep(0.05)
            return api.StandInGarden(garden.look(w, "ready"))

        made = api.GardenFactory(slow_garden)
        w.routes.garden_factory = made
        got, errors = [], []

        def ask_garden():
            try:
                gate.wait(timeout=30)
                got.append(w.routes._garden())
            except BaseException as exc:  # noqa: BLE001 - every fault is reported below
                errors.append(repr(exc))

        askers = [threading.Thread(target=ask_garden) for _ in range(8)]
        for thread in askers:
            thread.start()
        for thread in askers:
            thread.join(timeout=60)
        w.routes._held["garden"] = None
        assert errors == [] and len(got) == 8 and made.calls == 1, (errors, len(got), made.calls)
        assert len({id(one) for one in got}) == 1, "one garden for every request"

        # The mode reading follows the files through a manager of its own; the platform's is never touched.
        singleton = mode_module.Singleton
        assert [w.mode.live_reading(), w.mode.live_reading()] == ["daily", "daily"]
        mode_module._SECURITY_YAML.write_text("mode: bulbe\n# rewritten\n", encoding="ascii")
        assert w.mode.live_reading() == "bulbe", "witness: the reader follows the file"
        assert (singleton.invalidations, singleton.reads) == (0, 0), (singleton.invalidations, singleton.reads)

        # The switch and the api readers keep their cache in one assignment, keyed by the size too.
        tree = ast.parse((PACKAGE / "allium" / "settings.py").read_text(encoding="ascii"))
        assert _one_assignment(tree, "switch") == 1
        assert _one_assignment(tree, "api") == 1
        path = api.settings_file(w, tmp_path, ON + "api:\n  python_cap: 5000\n", name="sized.yaml")
        stamp = os.stat(path)
        assert w.settings.api(path).python_cap == 5000
        path.write_text(ON + "api:\n  python_cap: 60000\n# a longer file\n", encoding="utf-8")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert os.stat(path).st_mtime_ns == stamp.st_mtime_ns and os.stat(path).st_size != stamp.st_size
        assert w.settings.api(path).python_cap == 60000, "a file rewritten with its time forced back is read again"
    finally:
        restore()

    # The mode reading on the platform's own mode module: a change of the files is read by a manager of the
    # garden's own, a mismatch between them is recorded as tamper evidence once per change of the files, never
    # once per reading, and the server's own manager is never touched.
    audit, chain = [], []
    auth_module = types.ModuleType("opti_oignon.auth")
    auth_module.auth_manager = types.SimpleNamespace(_log_audit_event=lambda **kw: audit.append(kw["event_type"]))
    chain_module = types.ModuleType("opti_oignon.signed_audit_log")
    chain_module.chain_log = lambda **kw: chain.append(kw["event_type"])
    loaded, restore = isolate(targets={"opti_oignon.security_mode": source("security_mode.py"),
                                       "opti_oignon.allium.mode": source("allium", "mode.py")},
                              seeded={"opti_oignon.auth": auth_module, "opti_oignon.signed_audit_log": chain_module},
                              blocked=("opti_oignon.encryption",), packages=("opti_oignon.allium",))
    try:
        platform_mode, reading = loaded["opti_oignon.security_mode"], loaded["opti_oignon.allium.mode"]
        folder = tmp_path / "platform-mode"
        folder.mkdir()
        platform_mode._SECURITY_YAML = folder / "security.yaml"
        platform_mode._LOCKFILE_PATH = folder / ".security_mode_lock"
        platform_mode._DEFAULT_KEYFILE = folder / "no-keyfile"
        platform_mode._SECURITY_YAML.write_text("security_mode: daily\n", encoding="ascii")
        platform_mode._LOCKFILE_PATH.write_text("MODE:daily\nTIMESTAMP:1\n", encoding="ascii")
        platform_mode.security_mode_manager._cached_mode = "daily"
        assert [reading.live_reading(), reading.live_reading()] == ["daily", "daily"] and audit == chain == []
        platform_mode._LOCKFILE_PATH.write_text("MODE:bulbe\nTIMESTAMP:12\n", encoding="ascii")
        assert [reading.live_reading(), reading.live_reading()] == ["bulbe", "bulbe"]
        assert audit == chain == ["security_mode_mismatch"], (audit, chain)
        platform_mode._LOCKFILE_PATH.write_text("MODE:bulbe\nTIMESTAMP:123\n", encoding="ascii")
        assert reading.live_reading() == "bulbe"
        assert audit == chain == ["security_mode_mismatch"] * 2, "witness: each change of the files is recorded"
        assert platform_mode.security_mode_manager._cached_mode == "daily", "the server's own manager is untouched"
    finally:
        restore()


# ---------------------------------------------------------------------------
# AF3 -- the host check and the user dependency on every route; the surface from the transport
# ---------------------------------------------------------------------------
def test_af3_every_garden_route_carries_the_host_check_and_the_user_dependency_and_the_surface_is_the_transports(
        monkeypatch, tmp_path):
    mode = api.bulbe_module(False)
    token = "token-" + "a" * 24
    w, restore = api.open_api(monkeypatch, tmp_path, auth="single", security_mode=mode, tokens={token: ("leon", WALL)})
    try:
        routes, auth = w.routes, w.auth
        walked, flagged = _walk_router(routes.router, routes, auth)
        assert walked and flagged == [], (walked, flagged)
        assert routes.router.prefix == "/api/allium"
        from fastapi import APIRouter

        scratch = APIRouter(prefix="/api/allium", route_class=routes.GardenRoute)

        @scratch.get("/scratch")
        def scratch_route():
            return {}

        assert _walk_router(scratch, routes, auth)[1] == ["/api/allium/scratch"], "witness: the walk flags a route"

        # The auth answers, the garden a stand-in that records who asked.
        api.settings_file(w, tmp_path, ON)
        stand = api.StandInGarden(garden.look(w, "ready"))
        factory = api.GardenFactory(lambda: stand)
        routes.garden_factory = factory
        w.deps.AUTH_AVAILABLE, w.deps.auth_manager = False, None
        answer = api.get(w)
        assert answer.status_code == 503 and answer.json() == _refusal(w, "auth_unavailable"), answer.text
        assert factory.calls == 0 and stand.callers == []
        multi = api.AuthStandIn(False, {token: ("leon", WALL)})
        w.deps.AUTH_AVAILABLE, w.deps.auth_manager = True, multi
        answer = api.get(w)
        assert answer.status_code == 401 and answer.json() == _refusal(w, "sign_in"), answer.text
        assert factory.calls == 0 and stand.callers == []
        surfaces = {}
        single = api.AuthStandIn(True, {token: ("leon", WALL)})
        w.deps.auth_manager = single
        assert api.get(w).status_code == 200
        local = stand.callers[-1]
        assert local == w.membrane.Transport("web", principal=api.SYNTHETIC), local
        assert set(local.principal) == {"sub", "username", "role", "type"}, "the synthetic principal, as it is"
        assert "iat" not in local.principal and "exp" not in local.principal
        surfaces["single"] = w.membrane.surface_of(local, WALL)
        w.deps.auth_manager = multi
        for how, headers in (("cookie", {"Cookie": api.TOKEN_COOKIE + "=" + token}),
                             ("bearer", {"Authorization": "Bearer " + token})):
            answer = api.get(w, headers=headers)
            assert answer.status_code == 200, (how, answer.text)
            caller = stand.callers[-1]
            assert caller == w.membrane.Transport("web", principal=support.principal("leon", WALL)), (how, caller)
            surfaces[how] = w.membrane.surface_of(caller, WALL)
        assert surfaces == {"single": "web_local", "cookie": "web_session", "bearer": "web_session"}, surfaces
        assert factory.calls == 1
        # A query, a header or a cookie that names another account changes nothing of the caller.
        w.deps.auth_manager = single
        plain = api.get(w).json()
        for how, headers, query in (("query", {}, "?as=zz-other&sub=zz-other"), ("header", {"X-User": "zz-other"}, ""),
                                    ("cookie", {"Cookie": "sub=zz-other; account=zz-other"}, "")):
            with api.client(w) as http:
                answer = http.get(api.STATUS + query, headers=headers)
            assert answer.status_code == 200 and answer.json() == plain, (how, answer.text)
            assert stand.callers[-1] == w.membrane.Transport("web", principal=api.SYNTHETIC), (how, stand.callers[-1])
        w.deps.auth_manager = multi
        answer = api.get(w, headers={"Cookie": api.TOKEN_COOKIE + "=" + token + "; sub=zz-other"})
        assert stand.callers[-1] == w.membrane.Transport("web", principal=support.principal("leon", WALL)), answer.text

        # Bulbe with one account: a login is required, and the account is the local one.
        mode.active = True
        w.deps.auth_manager = single
        answer = api.get(w)
        assert answer.status_code == 401 and answer.json() == _refusal(w, "sign_in"), answer.text
        answer = api.get(w, headers={"Cookie": api.TOKEN_COOKIE + "=" + token})
        assert answer.status_code == 200, answer.text
        caller = stand.callers[-1]
        assert w.membrane.surface_of(caller, WALL) == "web_session"
        assert w.membrane.actor_of(caller, routes._single_user(), WALL) == "local"

        # What the route module reads and calls.
        tree = _routes_tree()
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args.args + node.args.kwonlyargs
                defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults) \
                    + list(node.args.kw_defaults)
                for argument, default in zip(arguments, defaults):
                    if (isinstance(default, ast.Call) and getattr(default.func, "id", "") == "Depends"
                            and default.args and getattr(default.args[0], "id", "") == "_get_current_user"):
                        guarded.add((node.name, argument.arg))
        owner = _enclosing(tree)
        built = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, (ast.Name, ast.Attribute)) and (getattr(func, "id", None) == "Transport"
                                                                or getattr(func, "attr", None) == "Transport"):
                raise AssertionError("the route module builds a Transport itself")
            if isinstance(func, ast.Attribute) and func.attr == "transport":
                built += 1
                principal = [k.value for k in node.keywords if k.arg == "principal"]
                where = owner.get(id(node))
                assert (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "web"
                        and len(principal) == 1 and isinstance(principal[0], ast.Name)
                        and (where, principal[0].id) in guarded), ast.dump(node)
            if isinstance(func, ast.Attribute) and func.attr in GARDEN_METHODS:
                assert "caller" in {k.arg for k in node.keywords}, ("a garden call passes caller=", ast.dump(node))
        assert built >= 1 and guarded, (built, guarded)
        inside = _functions_under(tree, "host_origin")
        assert inside, "the host check is a function of the route module"
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "cookies":
                raise AssertionError("the route module reads a cookie")
            if isinstance(node, ast.Attribute) and node.attr in ("headers", "scope"):
                assert id(node) in inside, ("a header is read outside the host check", node.lineno)
        # Nothing a request carries beside its principal can choose the caller: the dependency's parameter is
        # never rebound, a handler takes nothing else, and no part of a request is read past the host check.
        assert _request_reads(tree) == [], _request_reads(tree)
        text = (PACKAGE / "api" / "routes_allium.py").read_text(encoding="utf-8")
        planted = _request_reads(ast.parse(text + (
            "\n\n@router.get('/planted')\n"
            "def planted(request: Request, principal: dict = Depends(_get_current_user)) -> dict:\n"
            "    principal = dict(principal, sub=request.query_params.get('as'))\n"
            "    return principal\n")))
        assert {kind for kind, _where, _what in planted} == {"parameter", "rebinds", "reads"}, planted

        # An API garden refuses a missing caller on every method that takes one; the terminal's does not.
        api.settings_file(w, tmp_path, ON)
        wired = w.service.Garden.production(stopped=lambda: True, terminal=False, view_cap=w.service.api_view_cap,
                                            single_user=lambda: True)
        args = {"act": ("water",), "name": ("Pip",), "laws_apply": ("0" * 16,), "resume": (0, 0), "finish": ("0" * 8,),
                "confirm_share": ("K7QX-MPLR",)}
        methods = [name for name, member in inspect.getmembers(w.service.Garden, inspect.isfunction)
                   if not name.startswith("_") and "caller" in inspect.signature(member).parameters
                   and inspect.signature(member).parameters["caller"].default is None]
        assert set(GARDEN_METHODS) <= set(methods), methods
        for name in methods:
            with pytest.raises(TypeError):
                getattr(wired, name)(*args.get(name, ()))
        assert wired._store is None, "nothing was built"
        api.settings_file(w, tmp_path, OFF)
        terminal = w.service.Garden.production()
        assert terminal.look().status == "disabled", "witness: the terminal's garden has a default caller"
        terminal.close()

        # No middleware waves the route through.
        prefixes = _middleware_prefixes()
        assert len(prefixes) == len(MIDDLEWARE_PREFIXES) and all(prefixes.values()), prefixes
        for (file, name), values in prefixes.items():
            for value in values:
                assert not (api.STATUS == value or api.STATUS.startswith(value)), (file, name, value)
    finally:
        restore()


# ---------------------------------------------------------------------------
# AF9 -- the Host, the Origin and the page's site, before anything runs
# ---------------------------------------------------------------------------
def test_af9_a_rebound_host_a_foreign_origin_and_a_page_of_another_site_are_refused_before_anything_runs(
        w, tmp_path):
    api.settings_file(w, tmp_path, ON)
    canary = support.canary(w, "af9")
    stand = api.StandInGarden(garden.look(w, "ready"))
    factory = api.GardenFactory(lambda: stand)
    w.routes.garden_factory = factory
    application = api.app(w)
    from fastapi.routing import APIRoute

    paths = [route.path for route in w.routes.router.routes if isinstance(route, APIRoute)]
    assert paths, "presence: the router has routes"
    counted = []

    def untouched():
        return (w.manager.calls, factory.calls, len(stand.callers))

    def refused(answer, code, what):
        status, body = (answer.status_code, answer.json()) if hasattr(answer, "status_code") else answer
        assert (status, body) == (403, _refusal(w, code)), (what, status, body)
        assert canary not in json.dumps(body), what
        counted.append(code)

    def accepted(answer, what):
        assert answer.status_code == 200 and answer.json()["status"] == "ready", (what, answer.text)

    # A first accepted request builds the garden, so each refusal below is seen not to touch it.
    with api.client(w, application=application) as http:
        accepted(http.get(api.STATUS), "the loopback address")
    for path in paths:
        before = untouched()
        with api.client(w, base="http://evil.example:8001", application=application) as evil:
            refused(evil.get(path), "host", "a rebound name")
        with api.client(w, application=application) as http:
            for host in ("testserver", "127.0.0.1.", "user@127.0.0.1", "127.0.0.1@evil", "127.0.0.1:99999",
                         "localhost.evil.com", canary + ".example:8001"):
                refused(http.get(path, headers={"Host": host}), "host", host)
            for origin in ("null", "http://evil.example", "file://x", "http://localhost:5173/path",
                           "http://" + canary + ".example"):
                refused(http.get(path, headers={"Origin": origin}), "origin", origin)
            refused(http.get(path, headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors",
                                            "Sec-Fetch-Dest": "image"}), "site", "an image of another site")
            refused(http.get(path, headers={"Sec-Fetch-Site": "same-site"}), "site", "same-site with no origin")
            refused(http.get(path, headers={"Host": "evil.example", "X-Forwarded-Host": "127.0.0.1"}), "host",
                    "a forwarded name is never read")
            refused(http.get(path, headers={"Host": "evil.example:8001", "Origin": "http://localhost:5173"}), "host",
                    "an accepted origin never excuses a rebound name")
            refused(http.get(path, headers={"Host": "evil.example", "Sec-Fetch-Site": "same-origin"}), "host",
                    "the page's own site never excuses a rebound name")
        loop = (b"host", b"127.0.0.1:8001")
        refused(api.raw(application, [], path=path), "host", "no Host")
        refused(api.raw(application, [loop, (b"host", b"evil.example")], path=path), "host", "two Hosts")
        refused(api.raw(application, [loop, (b"origin", b"http://localhost:5173"),
                                      (b"origin", b"http://localhost:5173")], path=path), "origin", "two Origins")
        refused(api.raw(application, [loop, (b"sec-fetch-site", b"same-origin"),
                                      (b"sec-fetch-site", b"same-origin")], path=path), "site", "two sites")
        assert untouched() == before, "a refusal runs neither the auth manager nor the garden"
    assert set(counted) == {"host", "origin", "site"}

    # Accepted: the loopback names and the page's own site.
    with api.client(w, application=application) as http:
        for host in ("127.0.0.1:8001", "localhost:8001", "LOCALHOST", "[::1]:8001"):
            accepted(http.get(api.STATUS, headers={"Host": host}), host)
        accepted(http.get(api.STATUS, headers={"Origin": "http://localhost:5173"}), "the dev server's origin")
        accepted(http.get(api.STATUS), "no origin")
        for site in ("same-origin", "none"):
            accepted(http.get(api.STATUS, headers={"Sec-Fetch-Site": site}), site)
        accepted(http.get(api.STATUS, headers={"Sec-Fetch-Site": "cross-site", "Origin": "http://localhost:5173"}),
                 "another site whose origin is accepted")
        accepted(http.get(api.STATUS, headers={"Host": "127.0.0.1:8001", "X-Forwarded-Host": "evil.example"}),
                 "a forwarded name is never read")
        before = untouched()
        answer = http.post(api.STATUS)
        assert answer.status_code == 405 and untouched() == before, answer.text
    # Under the prefix, the platform's own answers run no garden work: a trailing slash is redirected by the
    # router before any dependency, whatever the name it was sent to.
    for base in (api.BASE, "http://evil.example:8001"):
        before = untouched()
        with api.client(w, base=base, application=application) as http:
            answer = http.get(api.STATUS + "/", follow_redirects=False)
        assert answer.status_code == 307 and untouched() == before, (base, answer.status_code, answer.text)

    # A listed name: over https for its origin; a list that cannot be read allows the loopback names only.
    api.settings_file(w, tmp_path, ON + "api:\n  hosts: [pc.lan]\n")
    assert "pc.lan" in w.settings.api().hosts
    with api.client(w, application=application) as http:
        accepted(http.get(api.STATUS, headers={"Host": "pc.lan:8001", "Sec-Fetch-Site": "same-origin"}), "pc.lan")
        accepted(http.get(api.STATUS, headers={"Origin": "https://pc.lan:5173", "Sec-Fetch-Site": "same-site"}),
                 "a listed origin over https")
        refused(http.get(api.STATUS, headers={"Origin": "http://pc.lan:5173"}), "origin", "a listed origin in clear")
    for hosts in ('"pc.lan"', '["*"]', '["http://pc.lan"]', '["pc.lan:80"]', '["0.0.0.0"]', '["[::]"]',
                  '["pc.lan", "*"]', '["pc.lan", "0.0.0.0"]', '["pc.lan", "[::]"]', '["pc.lan", "PC.LAN"]'):
        api.settings_file(w, tmp_path, ON + "api:\n  hosts: " + hosts + "\n")
        assert tuple(w.settings.api().hosts) == (), hosts
        with api.client(w, application=application) as http:
            refused(http.get(api.STATUS, headers={"Host": "pc.lan:8001"}), "host", hosts)
            accepted(http.get(api.STATUS), hosts)
    api.settings_file(w, tmp_path, ON + "api:\n  hosts: [localhost]\n")
    with api.client(w, application=application) as http:
        accepted(http.get(api.STATUS, headers={"Host": "localhost:8001"}), "a listed loopback name")
        refused(http.get(api.STATUS, headers={"Host": "pc.lan:8001"}), "host", "a listed loopback name")
    # The spellings a browser reads as the unspecified address refuse the list, as 0.0.0.0 does.
    for hosts in ('["0"]', '["0.0"]', '["000.000.000.000"]', '["0x0"]', '["pc.lan", "0x0"]'):
        api.settings_file(w, tmp_path, ON + "api:\n  hosts: " + hosts + "\n")
        assert tuple(w.settings.api().hosts) == () == w.routes.listed_hosts(), hosts
        with api.client(w, application=application) as http:
            refused(http.get(api.STATUS, headers={"Host": "pc.lan:8001"}), "host", hosts)

    # The dev proxy hands the browser's Host through.
    config = (REPO / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
    start = re.search(r"""['"]/api['"]\s*:\s*\{""", config)
    assert start, "the /api proxy entry is found"
    depth, end = 0, start.end() - 1
    for index in range(start.end() - 1, len(config)):
        depth += {"{": 1, "}": -1}.get(config[index], 0)
        if depth == 0:
            end = index
            break
    entry = config[start.end() - 1:end + 1]
    assert "target" in entry
    assert re.findall(r"changeOrigin\s*:\s*(\w+)", entry) == ["false"], entry


# ---------------------------------------------------------------------------
# AF5 -- one garden, built on first use; the health key and the switched-off route build nothing
# ---------------------------------------------------------------------------
def _holds_garden(w, value, depth=0):
    if isinstance(value, w.service.Garden):
        return True
    if depth < 2 and isinstance(value, dict):
        return any(_holds_garden(w, item, depth + 1) for item in value.values())
    if depth < 2 and isinstance(value, (list, tuple)):
        return any(_holds_garden(w, item, depth + 1) for item in value)
    return False


def test_af5_the_api_garden_is_built_once_on_first_use_and_the_health_key_and_the_off_route_build_nothing(
        w, tmp_path):
    # The application wires the health key and the shutdown; the platform's dependencies never name the being.
    tree = ast.parse((PACKAGE / "api" / "app.py").read_text(encoding="utf-8"))
    local = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("routes_allium"):
            for alias in node.names:
                local[alias.name] = alias.asname or alias.name
    assert {"health_flag", "close_garden"} <= set(local), local
    [health] = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "health_check"]
    maps = [value for node in ast.walk(health) if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "modules" and isinstance(value, ast.Dict)]
    assert len(maps) == 1
    keys = [key.value if isinstance(key, ast.Constant) else None for key in maps[0].keys]
    assert keys.count("allium") == 1, keys
    value = maps[0].values[keys.index("allium")]
    assert (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == local["health_flag"]
            and not value.args and not value.keywords), ast.dump(value)
    [lifespan] = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"]
    after = min(node.lineno for node in ast.walk(lifespan) if isinstance(node, ast.Yield))
    closing = [node for node in ast.walk(lifespan) if isinstance(node, ast.Try) and node.lineno > after
               and any(isinstance(call, ast.Call) and getattr(call.func, "id", None) == local["close_garden"]
                       for statement in node.body for call in ast.walk(statement))]
    assert closing, "the shutdown closes the garden inside a try"
    deps = ast.parse((PACKAGE / "api" / "deps.py").read_text(encoding="utf-8"))
    for node in ast.walk(deps):
        words = [getattr(node, "id", None), getattr(node, "attr", None), getattr(node, "module", None),
                 node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None]
        words += [alias.name for alias in getattr(node, "names", []) if isinstance(alias, ast.alias)]
        for word in words:
            assert not (isinstance(word, str) and ("allium" in word.lower() or "componion" in word.lower())), word

    routes = w.routes
    assert routes.ALLIUM_YAML == w.settings.config_file()
    assert [name for name, value in vars(routes).items() if _holds_garden(w, value)] == [], "loading builds no garden"

    # The router reads the switch by the garden's own rule, without importing the being.
    folder = tmp_path / "switch"
    folder.mkdir()
    corpus = {"true": "enabled: true\n", "yes": "enabled: yes\n", "false": "enabled: false\n",
              "quoted": 'enabled: "true"\n', "one": "enabled: 1\n", "no_key": "persistence: {}\n",
              "list": "- enabled\n- true\n", "unparsed": "enabled: [true\n"}
    files = {}
    for name, text in corpus.items():
        files[name] = folder / (name + ".yaml")
        files[name].write_text(text, encoding="utf-8")
    files["no_file"] = folder / "absent.yaml"
    answers = {}
    for name, path in files.items():
        answers[name] = routes.switched_on(path)
        assert answers[name] is (w.settings.switch(path) == "on"), name
        assert routes.health_flag(path) is answers[name], name
    assert {name for name, on in answers.items() if on} == {"true", "yes"}, answers

    # The router reads the names of api.hosts from the same file, without importing the being, by the rule the
    # garden's settings read them: the two readers agree on every list, good or not.
    hosts_corpus = {
        "none": "enabled: true\n", "one": "api:\n  hosts: [pc.lan]\n",
        "several": 'api:\n  hosts: [pc.lan, 192.168.1.20, "[fe80::1]", localhost]\n',
        "off_listed": "enabled: false\napi:\n  hosts: [pc.lan]\n", "string": 'api:\n  hosts: "pc.lan"\n',
        "star": 'api:\n  hosts: ["*"]\n', "scheme": 'api:\n  hosts: ["http://pc.lan"]\n',
        "port": 'api:\n  hosts: ["pc.lan:80"]\n', "upper": 'api:\n  hosts: ["PC.LAN"]\n',
        "any4": 'api:\n  hosts: ["0.0.0.0"]\n', "any6": 'api:\n  hosts: ["[::]"]\n',
        "mapped": 'api:\n  hosts: ["[::ffff:0.0.0.0]"]\n', "zero": 'api:\n  hosts: ["0"]\n',
        "zero2": 'api:\n  hosts: ["0.0"]\n', "zero4": 'api:\n  hosts: ["000.000.000.000"]\n',
        "hex": 'api:\n  hosts: ["0x0"]\n', "bracket": 'api:\n  hosts: ["[zz]"]\n', "number": "api:\n  hosts: [1]\n",
        "empty": "api:\n  hosts: []\n", "null": "api:\n  hosts:\n", "api_list": "api: [pc.lan]\n",
        "top_list": "- api\n", "unparsed": "api: [pc.lan\n"}
    hosts_answers = {}
    for name, text in hosts_corpus.items():
        path = folder / ("hosts_" + name + ".yaml")
        path.write_text(text, encoding="utf-8")
        hosts_answers[name] = routes.listed_hosts(path)
        assert hosts_answers[name] == tuple(w.settings.api(path).hosts), (name, hosts_answers[name])
    hosts_answers["no_file"] = routes.listed_hosts(folder / "absent-hosts.yaml")
    assert hosts_answers["no_file"] == tuple(w.settings.api(folder / "absent-hosts.yaml").hosts) == ()
    listed_ones = {name for name, found in hosts_answers.items() if found}
    assert listed_ones == {"one", "several", "off_listed"}, hosts_answers
    assert hosts_answers["several"] == ("pc.lan", "192.168.1.20", "[fe80::1]", "localhost"), hosts_answers

    # Switched off: three reads, no garden. Switched on: three reads, one garden.
    given = support.seams(w, tmp_path, suite="af5")
    built = []

    def make():
        built.append(api.api_garden(w, given))
        return built[-1]

    factory = api.GardenFactory(make)
    routes.garden_factory = factory
    routes.close_garden()
    api.settings_file(w, tmp_path, OFF)
    for _ in range(3):
        answer = api.get(w)
        assert answer.status_code == 200 and answer.json() == routes.DISABLED, answer.text
    assert factory.calls == 0 and w.stop.calls == 0
    real_reader = w.service.platform_single_user
    reader = _Counting(real_reader)
    w.service.platform_single_user = reader
    api.settings_file(w, tmp_path, ON)
    reads = w.manager.reads
    for _ in range(3):
        answer = api.get(w)
        assert answer.status_code == 200 and answer.json()["status"] == "ready", answer.text
    assert factory.calls == 1 and len(built) == 1
    assert w.manager.reads - reads == 3, "the user dependency reads the manager once per request"
    assert reader.calls == 0

    # The shutdown closes what was built, once.
    [store] = built[0].store_factory.stores
    closes = _Counting(store.close)
    store.close = closes
    routes.close_garden()
    routes.close_garden()
    assert closes.calls == 1

    # The production garden: the manager's rule, the emergency stop, no attended verb, the API's cap.
    records = []
    real_store = w.store.Store

    def recording(**keywords):
        records.append(dict(keywords))
        return real_store(**dict(given, single_user=keywords["single_user"]))

    w.store.Store = recording
    try:
        wired = routes._production_garden()
        assert records == [], "the store is built at the first look"
        assert wired.attended is None and wired.view_cap is w.service.api_view_cap and wired.caller_required is True
        caller = _web_caller(w)
        reads, stops = w.manager.reads, w.stop.calls
        looks = []
        for value in (True, RuntimeError("the stop cannot be read"), False, False):
            w.stop.value = value
            looks.append(wired.look(caller=caller))
        assert [look.status for look in looks] == ["stopped", "stopped", "ready", "ready"]
        heads = [w.describe.web_fields(look)["lines"][0]["key"] for look in looks[:3]]
        assert heads == ["status.stopped", "status.stopped.unknown", "status.ready"], heads
        assert w.stop.calls - stops >= 3
        assert records == [{"single_user": routes._single_user}], records
        assert w.manager.reads - reads == 2, "the garden's single-user reading is the manager's, once per action"
        assert reader.calls == 0, "the terminal's reader is never asked"
        wired.close()
    finally:
        w.store.Store = real_store
        w.service.platform_single_user = real_reader


# ---------------------------------------------------------------------------
# AF17 -- switched off, a fresh process imports nothing of the being
# ---------------------------------------------------------------------------
CHILD_OFF = '''
import json, os, sys, threading
sys.path.insert(0, __REPO__)
sys.path.insert(0, __TESTS__)
from pathlib import Path
from _data_firewall import DataFirewall
firewall = DataFirewall(__REPO__, seed=False)
firewall.install()
try:
    os.stat(os.path.join(__REPO__, "data", "planted-" + __CANARY__))
except OSError:
    pass
planted = sum(len(found) for found in firewall.redirected.values())


def being(names):
    return sorted(name for name in names if name == "opti_oignon.allium" or name.startswith("opti_oignon.allium."))


import opti_oignon.api.routes_allium as routes
baseline = set(sys.modules)
off = Path(__OFF__)
flag_off = routes.health_flag(off)
routes.ALLIUM_YAML = off
served_off = routes.garden_status(principal=dict(__SYNTHETIC__))
added_off = being(set(sys.modules) - baseline)
counter_off = sum(len(found) for found in firewall.redirected.values())
threads_off = threading.active_count()

# Through the router: the host check and the user dependency run before the handler. The auth manager and
# the security mode are stand-ins, seeded before any request, so the platform's own are never imported.
import asyncio, types
deps = types.ModuleType("opti_oignon.api.deps")
deps.AUTH_AVAILABLE = True
deps.auth_manager = types.SimpleNamespace(single_user_mode=True)
sys.modules["opti_oignon.api.deps"] = deps
mode = types.ModuleType("opti_oignon.security_mode")
mode.is_bulbe = lambda: False
sys.modules["opti_oignon.security_mode"] = mode
from fastapi import FastAPI
application = FastAPI()
application.include_router(routes.router)


def routed(host):
    sent = []
    scope = {"type": "http", "method": "GET", "path": "/api/allium/status", "raw_path": b"/api/allium/status",
             "query_string": b"", "headers": [(b"host", host), (b"sec-fetch-site", b"same-origin")],
             "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8001), "scheme": "http",
             "http_version": "1.1", "root_path": "", "app": application}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(application(scope, receive, send))
    return sent[0]["status"], json.loads(b"".join(m.get("body", b"") for m in sent[1:]))


routes.ALLIUM_YAML = Path(__LISTED__)
before = set(sys.modules)
listed_status, listed_body = routed(b"pc.lan:8001")
added_listed = being(set(sys.modules) - before)
before = set(sys.modules)
refused_status, refused_body = routed(b"evil.example:8001")
added_refused = being(set(sys.modules) - before)
counter_routed = sum(len(found) for found in firewall.redirected.values())


class StandIn:
    def look(self, caller=None, cap=None):
        from opti_oignon.allium import service
        return service.Look("ready", (), "daily", None, None, None, None, None, None, None, 0, False)


on = Path(__ON__)
routes.ALLIUM_YAML = on
before = set(sys.modules)
flag_on = routes.health_flag(on)
added_flag = being(set(sys.modules) - before)
routes.garden_factory = StandIn
served_on = routes.garden_status(principal=dict(__SYNTHETIC__))
added_on = being(set(sys.modules) - baseline)
counter_on = sum(len(found) for found in firewall.redirected.values())
firewall.uninstall()
sys.stderr.write(json.dumps({"added_flag": added_flag, "added_off": added_off, "added_on": added_on,
                             "counter_off": counter_off, "counter_on": counter_on, "flag_off": flag_off,
                             "flag_on": flag_on, "planted": planted, "served_off": served_off,
                             "served_on": served_on, "threads_off": threads_off,
                             "listed_status": listed_status, "listed_body": listed_body,
                             "added_listed": added_listed, "refused_status": refused_status,
                             "refused_body": refused_body, "added_refused": added_refused,
                             "counter_routed": counter_routed}) + "\\n")
'''


def test_af17_with_the_switch_off_the_api_imports_nothing_of_the_being(tmp_path):
    settings = tmp_path / "settings"
    settings.mkdir()
    off, on, listed = settings / "off.yaml", settings / "on.yaml", settings / "listed.yaml"
    off.write_text(OFF, encoding="utf-8")
    on.write_text(ON, encoding="utf-8")
    listed.write_text(OFF + "api:\n  hosts: [pc.lan]\n", encoding="utf-8")
    code = _code(CHILD_OFF, repo=str(REPO), tests=str(TESTS), canary="af17canary", off=str(off), on=str(on),
                 listed=str(listed), synthetic=api.SYNTHETIC)
    code_, out, err, report = garden.finish(garden.child(tmp_path, code, cwd=tmp_path / "cwd"))
    assert code_ == 0 and report is not None, err[-2000:]
    assert report["planted"] == 1, "the firewall's counter can count"
    assert report["flag_off"] is False and report["served_off"] == {
        "as_of": None, "being": None, "habitat": None, "labels": [], "law": None, "lines": [], "source": "simulation",
        "status": "disabled"}, report
    assert report["added_off"] == [] and report["counter_off"] == 1 and report["threads_off"] == 1, report
    # Through the router, switched off: a request addressed by a listed name is answered with nothing of the
    # being imported; a refused one imports the catalogue and its nets to say its line, and nothing else.
    assert (report["listed_status"], report["listed_body"]) == (200, report["served_off"]), report
    assert report["added_listed"] == [], report["added_listed"]
    assert report["refused_status"] == 403 and report["refused_body"]["refusal"] == "host", report
    assert report["added_refused"] == ["opti_oignon.allium", "opti_oignon.allium.ethics",
                                       "opti_oignon.allium.wording"], report["added_refused"]
    assert report["counter_routed"] == 1, report
    # The witness: the probe counts what a status call switched on imports.
    assert report["flag_on"] is True and report["added_flag"] == [], report
    assert report["served_on"]["status"] == "ready", report["served_on"]
    added = set(report["added_on"])
    assert {"opti_oignon.allium.service", "opti_oignon.allium.describe"} <= added, added
    assert not added & {"opti_oignon.allium.store", "opti_oignon.allium.life", "opti_oignon.allium.engine",
                        "opti_oignon.allium.mode"}, "a stand-in garden builds no store"
    assert report["counter_on"] == 1, report


# ---------------------------------------------------------------------------
# AF6 -- the cap plus two awake days, and the settings of the caps
# ---------------------------------------------------------------------------
def test_af6_a_view_under_the_api_cap_does_at_most_the_cap_plus_two_awake_days_of_work(w, tmp_path, caplog):
    api.settings_file(w, tmp_path, ON + "api:\n  python_cap: 5000\n")
    assert w.engine.native_in_use() is False
    given = support.seams(w, tmp_path, suite="af6")
    served = api.api_garden(w, given)
    w.routes.garden_factory = api.GardenFactory(lambda: served)
    recorder = api.Recorder(w)
    try:
        mark = recorder.mark()
        answer = api.get(w)
        assert answer.status_code == 200 and answer.json()["status"] == "ready", answer.text
        ready_work = api.work(recorder.since(mark))
        _sow(w, given)
        laws = _laws_named(w, given)
        assert laws == {"fixture"}, laws
        ceilings = {name: w.lawfiles.law(name)["work"]["ceilings"] for name in w.lawfiles.LAWS}
        for name, ceiling in ceilings.items():
            assert ceiling["dormant_day"] <= ceiling["awake_day"], name
        awake = max(ceilings[name]["awake_day"] for name in laws)
        assert awake == 4209
        bound = 5000 + 2 * awake
        assert ready_work <= bound
        forms, done = {}, {}
        elapsed = 0
        for age in (1, 10):
            given["clock"].advance_days(age - elapsed)
            elapsed = age
            mark = recorder.mark()
            answer = api.get(w)
            assert answer.status_code == 200, answer.text
            forms[age], done[age] = answer.json(), api.work(recorder.since(mark))
            assert done[age] <= bound, (age, done[age], bound)
        assert "catching_up" not in forms[1]["labels"], forms[1]["labels"]
        assert forms[1]["as_of"]["local"] == _local(WALL + DAY_S)
        assert "catching_up" in forms[10]["labels"], forms[10]["labels"]
        keys = [line["key"] for line in forms[10]["lines"]]
        assert keys[keys.index("label.catching_up") + 1] == "web.catching_up", keys
        assert forms[10]["as_of"]["local"] == _local(WALL) != _local(WALL + 10 * DAY_S), forms[10]["as_of"]
        assert f"Shown as of {_local(WALL)} (+00:00)" in forms[10]["lines"][keys.index("label.catching_up")]["text"]

        # The witness: the terminal's garden looks at the same being uncapped, past the bound.
        looker = garden.garden(w, given, law="fixture")()
        mark = recorder.mark()
        try:
            assert looker.look().status == "alive"
        finally:
            looker.close()
        assert api.work(recorder.since(mark)) > bound
    finally:
        recorder.close()
        served.close()

    # The caps read their defaults, by name, when they are not ones.
    caplog.set_level(logging.WARNING)
    for index, value in enumerate(("0", "-1", "true", '"5000"', str(2 ** 53))):
        path = api.settings_file(w, tmp_path, ON + "api:\n  python_cap: " + value + "\n", name=f"cap{index}.yaml")
        caplog.clear()
        assert w.settings.api(path).python_cap == 200000, value
        assert [r for r in caplog.records if r.levelno == logging.WARNING], ("the fallback is logged", value)
    path = api.settings_file(w, tmp_path, ON, name="plain.yaml")
    read = w.settings.api(path)
    assert (read.python_cap, read.native_cap, tuple(read.hosts)) == (200000, 5000000, ())

    # A cap below one awake day of the laws this engine carries is raised to that day, by name, so the line
    # a capped view says holds: after a write made in the terminal, the next capped view is current.
    floor = max(w.lawfiles.law(name)["work"]["ceilings"]["awake_day"] for name in w.lawfiles.LAWS)
    assert floor == 4591
    low = api.settings_file(w, tmp_path, ON + "api:\n  python_cap: 1000\n  native_cap: 10\n", name="low.yaml")
    caplog.clear()
    assert w.service.api_view_cap(low) == floor
    assert [r for r in caplog.records if r.levelno == logging.WARNING and "awake day" in r.getMessage()], \
        "the raise is logged"
    real_native = w.engine.native_in_use
    w.engine.native_in_use = lambda: True
    try:
        assert w.service.api_view_cap(low) == floor
    finally:
        w.engine.native_in_use = real_native
    kept = api.settings_file(w, tmp_path, ON + "api:\n  python_cap: 5000\n", name="kept.yaml")
    assert w.service.api_view_cap(kept) == 5000, "witness: a cap above the day is kept as written"
    api.settings_file(w, tmp_path, ON + "api:\n  python_cap: 1000\n")
    given = support.seams(w, tmp_path / "low", suite="af6low")
    _sow(w, given)
    given["clock"].advance_days(10)
    served = api.api_garden(w, given)
    try:
        caller = _web_caller(w)
        assert "catching_up" in served.look(caller=caller).labels
        carer = garden.garden(w, given, law="fixture")()
        try:
            carer.act("water")
        finally:
            carer.close()
        after = served.look(caller=caller)
        assert after.status == "alive" and "catching_up" not in after.labels, after.labels
    finally:
        served.close()


# ---------------------------------------------------------------------------
# AF7 -- no handler asks past the cap, catches up or writes
# ---------------------------------------------------------------------------
def test_af7_no_api_handler_asks_the_engine_past_the_cap_catches_up_or_writes(w, tmp_path):
    api.settings_file(w, tmp_path, CAPPED)
    given = support.seams(w, tmp_path, suite="af7")
    _sow(w, given)
    served = api.api_garden(w, given)
    w.routes.garden_factory = api.GardenFactory(lambda: served)
    path = support.store_path(w, given)
    folder = support.directory(given)
    settles = _Counting(w.life.settle)
    w.life.settle = settles

    def kept():
        return (support.sha256_file(path), _listing(folder), _checkpoints(path),
                json.dumps(given["audit"].entries, sort_keys=True))

    recorder = api.Recorder(w)
    try:
        # Two looks, a day and ten days after the sowing: every request within the cap, nothing written.
        given["clock"].advance_days(1)
        before = kept()
        mark = recorder.mark()
        statuses = []
        for step in (0, 9):
            given["clock"].advance_days(step)
            answer = api.get(w)
            assert answer.status_code == 200, answer.text
            statuses.append((answer.json()["status"], "catching_up" in answer.json()["labels"]))
        assert statuses == [("alive", False), ("alive", True)], statuses
        asked = recorder.since(mark)
        assert {entry[0] for entry in asked} <= {"engine", "fact_envelope", "advance"}, asked
        budgets = [entry[1] for entry in asked if entry[0] == "advance"]
        assert budgets and all(isinstance(b, int) and b <= 5000 for b in budgets), budgets
        assert kept() == before, "a look writes nothing"
        assert settles.calls == 0 and _checkpoints(path) == [], "a look never catches up"

        # The platform: a view of a being with no kept state asks within its cap, the minute-0 fold included.
        target = w.store.Store(**given)
        mark = recorder.mark()
        try:
            view = target.open("local").view(cap=5000)
        finally:
            target.close()
        assert view.status == "catching_up"
        budgets = [entry[1] for entry in recorder.since(mark) if entry[0] == "advance"]
        assert len(budgets) >= 2 and all(b <= 5000 for b in budgets), budgets
    finally:
        recorder.close()

    # Frozen: the engine stops on a fault past minute 0; the kept state is asked for within the cap.
    mid = kept()
    real = w.engine.call

    def panicking(request):
        asked = w.wire.parse(request)
        if asked.get("op") == "advance" and asked.get("to", 0) > 0:
            return w.wire.emit({"detail": "x", "refused": "engine_panic"})
        return real(request)

    recorder = api.Recorder(w, call=panicking)
    try:
        answer = api.get(w)
        body = answer.json()
        assert answer.status_code == 200 and body["status"] == "alive" and "frozen" in body["labels"], body
        folds = [entry[1] for entry in recorder.requests if entry[0] == "advance" and entry[2] == 0]
        assert folds and all(b <= 5000 for b in folds), recorder.requests
    finally:
        recorder.close()

    # Native: the native cap, above the reference's and within its own.
    real_native = w.engine.native_in_use
    w.engine.native_in_use = lambda: True
    recorder = api.Recorder(w)
    try:
        answer = api.get(w)
        assert answer.status_code == 200 and "catching_up" not in answer.json()["labels"], answer.text
        budgets = [entry[1] for entry in recorder.requests if entry[0] == "advance"]
        assert budgets and 5000 < max(budgets) <= 50000, budgets
    finally:
        recorder.close()
        w.engine.native_in_use = real_native
    assert kept() == mid and settles.calls == 0, "a frozen look and a native one write nothing either"

    # Witnesses: the terminal's look is uncapped; a terminal gesture writes, and a settle keeps a state.
    recorder = api.Recorder(w)
    try:
        looker = garden.garden(w, given, law="fixture")()
        try:
            looker.look()
        finally:
            looker.close()
        assert MAX_INT in [entry[1] for entry in recorder.requests if entry[0] == "advance"]
    finally:
        recorder.close()
    before = kept()
    carer = garden.garden(w, given, law="fixture")()
    try:
        carer.act("water")
    finally:
        carer.close()
    assert api.get(w).status_code == 200
    after = kept()
    assert [a != b for a, b in zip(after, before)] == [True, True, True, True], "witness: a gesture moves every probe"
    assert settles.calls >= 1
    served.close()
