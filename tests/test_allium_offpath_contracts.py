#!/usr/bin/env python3
"""Contracts for the componion's absence from everything that is not the garden: the package, the chat path.

The being is reached through two doors only -- the terminal's ``oo garden``
and the API's read route -- and only by a closed, named set of producers.
Nothing on the chat path imports it; the backend and the emergency stop
never name it; the light hook the backend carries is passive until a sink
is registered, and no head calls it yet.

  * AO1 -- importing the package, or the route module, loads nothing of the
    being, opens no database and starts no thread: the facade's static
    tables, then a fresh process.
  * AO2 -- the backend and the emergency stop name the being nowhere, at any
    depth; the light sink is bound once, to nothing, and rebound only by its
    registration.
  * AO3 -- the chat path never reaches the being: the module-scope imports of
    its four modules, its import closure, and the eager closure of the CLI.
  * AO4 -- the being is entered only through the terminal and the read route,
    by a closed, named set of producers: its importers, what the application
    and the route module may call, every write site, the one place a
    transport is built, and no registration of a light sink.
  * AO5 -- with no light sink, the hook and a chat turn import and stat
    nothing of the being; a registered sink sees raw counts only; no head
    reports to it yet.

Local-only (the public distribution ships no tests). The censuses read the
syntax trees (``tests/_allium_census.py``) and never import the package; the
backend loads through the shared isolation window with its client library
scripted; the child of AO1 imports this tree behind the data firewall.
"""

import ast
import builtins
import importlib
import json
import logging
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_census as census  # noqa: E402
import _allium_garden_support as garden  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_ao1_importing_the_package_or_the_route_module_loads_nothing_of_the_being": 2.0,
    "test_ao2_the_backend_and_the_emergency_stop_name_the_being_nowhere_and_the_sink_is_passive": 2.0,
    "test_ao3_the_chat_path_never_reaches_the_being": 2.0,
    "test_ao4_the_being_is_entered_only_through_the_terminal_and_the_read_route_by_named_producers": 2.0,
    "test_ao5_with_no_light_sink_the_hook_and_a_chat_turn_import_and_stat_nothing_of_the_being": 2.0,
}
PACKAGE = REPO / "opti_oignon"
TESTS = REPO / "tests"
# The two modules that open the doors to the being: the terminal's commands and the API's router.
ENTRIES = ("opti_oignon.cli.garden", "opti_oignon.api.routes_allium")
CHAT_EAGER = ("cli.main", "cli.session", "executor", "inference_backend")
CHAT_ROOTS = ("cli.session", "executor", "inference_backend", "core_daemon", "api.routes_chat")
# The garden's methods the route module may not call (it calls ``look``, and ``close`` in its shutdown only).
FORBIDDEN = ("sow", "sow_card", "act", "name", "name_card", "verify", "laws", "laws_diff", "laws_apply", "laws_pin",
             "laws_unpin", "resume", "finish", "confirm_share", "gated", "store", "settle")
# The producers: the store's and the being's write methods (``append`` with a transport only), and the
# service's write verbs.
STORE_WRITES = ("sow", "finish_sowing", "resume", "append", "rhythm_put", "heard_note", "heard_end_season",
                "checkpoint_put", "checkpoint_prune", "settle", "laws_apply", "laws_pin", "laws_unpin")
SERVICE_VERBS = ("sow", "act", "name", "laws_apply", "laws_pin", "laws_unpin", "resume", "finish")
PRODUCERS = frozenset({
    ("opti_oignon.allium.life", "settle", "checkpoint_put"),
    ("opti_oignon.allium.life", "settle", "checkpoint_prune"),
    ("opti_oignon.allium.service", "Garden.sow", "sow"),
    ("opti_oignon.allium.service", "Garden.sow", "append"),
    ("opti_oignon.allium.service", "Garden._settle", "settle"),
    ("opti_oignon.allium.service", "Garden.act", "append"),
    ("opti_oignon.allium.service", "Garden.name", "append"),
    ("opti_oignon.allium.service", "Garden.laws_apply", "laws_apply"),
    ("opti_oignon.allium.service", "Garden.laws_pin", "laws_pin"),
    ("opti_oignon.allium.service", "Garden.laws_unpin", "laws_unpin"),
    ("opti_oignon.allium.service", "Garden.resume", "resume"),
    ("opti_oignon.allium.service", "Garden.finish", "finish_sowing"),
    ("opti_oignon.allium.store", "BeingStore.settle", "settle"),
    ("opti_oignon.cli.garden", "run_sow", "sow"),
    ("opti_oignon.cli.garden", "run_care", "act"),
    ("opti_oignon.cli.garden", "run_name", "name"),
    ("opti_oignon.cli.garden", "run_laws_apply", "laws_apply"),
    ("opti_oignon.cli.garden", "_pinning", "laws_pin"),
    ("opti_oignon.cli.garden", "_pinning", "laws_unpin"),
    ("opti_oignon.cli.garden", "run_resume", "resume"),
    ("opti_oignon.cli.garden", "run_finish", "finish"),
})
_BACKEND = "opti_oignon.inference_backend"
# The four head shapes the hook is called with, and what a sink must receive for them.
HEADS = (("generate", {"eval_count": 5, "prompt_eval_count": 11}), ("stream", {"eval_count": 3, "prompt_eval_count": 7}),
         ("embed", {"prompt_eval_count": 4}), ("embed_many", {"prompt_eval_count": 4}))
RECORDS = [("generate", "ollama", 11, "reported", 5, "reported"), ("stream", "ollama", 7, "reported", 3, "reported"),
           ("embed", "ollama", 4, "reported", None, "not_generated"),
           ("embed_many", "ollama", 4, "reported", None, "not_generated")]
SLOTS = ("head", "backend", "prompt", "prompt_source", "completion", "completion_source")
REPLY = "canary-reply-that-no-sink-may-see"
PROMPT = "canary-prompt-that-no-sink-may-see"
MODEL = "canary-model-name"


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


def _code(template, **values):
    out = template
    for name, value in values.items():
        out = out.replace("__" + name.upper() + "__", repr(value))
    assert "__" not in out.replace("__name__", "").replace("__init__", ""), "every placeholder is filled"
    return out


def _being(names):
    """The names among ``names`` that are the being or one of its two doors."""
    return sorted(name for name in names
                  if name == "opti_oignon.allium" or name.startswith("opti_oignon.allium.") or name in ENTRIES)


# ---------------------------------------------------------------------------
# AO2 -- the backend and the stop never name the being; the sink is passive
# ---------------------------------------------------------------------------
def _names_the_being(text):
    """Every import, alias, name, attribute, definition or string of ``text`` that holds the being's name."""
    hits = []
    for node in ast.walk(ast.parse(text)):
        values = [getattr(node, "id", None), getattr(node, "attr", None)]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            values.append(getattr(node, "module", None))
            values += [part for alias in node.names for part in (alias.name, alias.asname)]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            values.append(node.name)
        if isinstance(node, ast.arg):
            values.append(node.arg)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append(node.value)
        for value in values:
            if isinstance(value, str) and ("allium" in value.lower() or "componion" in value.lower()):
                hits.append(value)
    return hits


def _module_statements(body):
    """Every statement run at module scope: the module's own, and those of its ``try``, ``if``, ``with`` and loops."""
    for node in body:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(node, field, None)
            if isinstance(inner, list):
                yield from _module_statements(inner)
        for handler in getattr(node, "handlers", None) or ():
            yield from _module_statements(handler.body)
        for case in getattr(node, "cases", None) or ():
            yield from _module_statements(case.body)


def _sink_bindings(tree):
    """``(module-scope bindings of _LIGHT_SINK, functions that rebind it, functions declaring it global)``."""
    module_level = []
    for node in _module_statements(tree.body):
        targets = (node.targets if isinstance(node, ast.Assign) else
                   [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name) and name.id == "_LIGHT_SINK":
                    module_level.append(node.value)
    rebinding, declaring, foreign = set(), set(), []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Global) and "_LIGHT_SINK" in node.names:
                declaring.add(fn.name)
            targets = (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name) and name.id == "_LIGHT_SINK":
                        rebinding.add(fn.name)
                    if isinstance(name, ast.Attribute) and name.attr == "_LIGHT_SINK":
                        foreign.append(fn.name)
    return module_level, rebinding, declaring, foreign


def test_ao2_the_backend_and_the_emergency_stop_name_the_being_nowhere_and_the_sink_is_passive():
    for rel in ("inference_backend.py", "emergency_stop.py"):
        text = (PACKAGE / rel).read_text(encoding="utf-8")
        assert len(text) > 1000
        assert _names_the_being(text) == [], (rel, _names_the_being(text))
    assert _names_the_being('import importlib\n\n\ndef f():\n    importlib.import_module("opti_oignon.allium.light")\n')
    assert _names_the_being("def g():\n    from opti_oignon.allium import settings\n")
    tree = ast.parse((PACKAGE / "inference_backend.py").read_text(encoding="utf-8"))
    module_level, rebinding, declaring, foreign = _sink_bindings(tree)
    assert len(module_level) == 1 and isinstance(module_level[0], ast.Constant) and module_level[0].value is None, \
        "the sink is bound at module scope once, to nothing"
    assert rebinding == {"set_light_sink"} and declaring == {"set_light_sink"}, (rebinding, declaring)
    assert foreign == []
    planted = ast.parse("_LIGHT_SINK = None\n\n\ndef set_light_sink(sink):\n    global _LIGHT_SINK\n"
                        "    _LIGHT_SINK = sink\n\n\ndef _resolve():\n    global _LIGHT_SINK\n    _LIGHT_SINK = print\n")
    assert _sink_bindings(planted)[1] == {"set_light_sink", "_resolve"}, "witness: a second binder is found"
    nested = ast.parse("_LIGHT_SINK = None\ntry:\n    _LIGHT_SINK = print\nexcept Exception:\n    pass\n"
                       "if True:\n    with open(__file__) as f:\n        _LIGHT_SINK = f\n")
    assert len(_sink_bindings(nested)[0]) == 3, "witness: a module-scope binding inside a try, an if or a with"


# ---------------------------------------------------------------------------
# AO3 -- the chat path never reaches the being
# ---------------------------------------------------------------------------
def test_ao3_the_chat_path_never_reaches_the_being():
    met = set()
    for name in CHAT_EAGER:
        eager = census.imports(census.full(name), eager=True)
        met.update(eager)
        assert _being(eager) == [], (name, _being(eager))
    assert {"opti_oignon.cli.output", "opti_oignon.inference_backend"} <= met, "presence: module-scope imports read"
    reached = census.closure(CHAT_ROOTS)
    assert "opti_oignon.inference_backend" in reached and len(reached) > 100, len(reached)
    assert _being(reached) == [], _being(reached)
    eager = census.closure(["cli.main"], eager=True)
    assert "opti_oignon.cli.output" in eager, sorted(eager)
    assert _being(eager) == [], _being(eager)
    session = (PACKAGE / "cli" / "session.py").read_text(encoding="utf-8")
    overrides = {
        "opti_oignon.cli.session": session + "\n\ndef _planted_hop():\n    from opti_oignon import zz_hop1  # noqa\n",
        "opti_oignon.zz_hop1": "from opti_oignon import zz_hop2  # noqa\n",
        "opti_oignon.zz_hop2": "from opti_oignon.allium import service  # noqa\n",
    }
    planted = census.closure(CHAT_ROOTS, overrides=overrides)
    assert {"opti_oignon.zz_hop1", "opti_oignon.zz_hop2", "opti_oignon.allium.service"} <= planted, \
        "witness: a chain whose first hop is in a function is followed"
    for spelling in ('importlib.import_module("opti_oignon.allium.service")',
                     '__import__("opti_oignon.allium.service")',
                     'importlib.import_module(".allium.service", "opti_oignon")',
                     '__import__("opti_oignon.allium", fromlist=["service"])'):
        literal = {"opti_oignon.cli.session": session + "\n\ndef _planted_literal():\n    import importlib\n"
                                                        "    return " + spelling + "\n"}
        assert "opti_oignon.allium.service" in census.closure(CHAT_ROOTS, overrides=literal), \
            ("witness: an import by a literal name is followed", spelling)
    main = (PACKAGE / "cli" / "main.py").read_text(encoding="utf-8")
    early = census.closure(["cli.main"], eager=True,
                           overrides={"opti_oignon.cli.main": "from .garden import run_show  # noqa\n" + main})
    assert "opti_oignon.cli.garden" in _being(early), "witness: a module-scope door is found"


# ---------------------------------------------------------------------------
# AO1 -- the package and the route module load nothing of the being
# ---------------------------------------------------------------------------
CHILD_IMPORT = '''
import json, os, sqlite3, sys, threading
sys.path.insert(0, __REPO__)
sys.path.insert(0, __TESTS__)
from _data_firewall import DataFirewall
firewall = DataFirewall(__REPO__, seed=False)
firewall.install()
try:
    os.stat(os.path.join(__REPO__, "data", "planted-" + __CANARY__))
except OSError:
    pass
planted = sum(len(found) for found in firewall.redirected.values())
connects = []
sys.addaudithook(lambda event, args: connects.append(str(args[0]) if args else "")
                 if event == "sqlite3.connect" else None)


def project():
    return sorted(name for name in sys.modules if name == "opti_oignon" or name.startswith("opti_oignon."))


import opti_oignon
package = project()
package_threads = threading.active_count()
package_connects = list(connects)
import opti_oignon.api.routes_allium
routes = project()
routes_threads = threading.active_count()
routes_connects = list(connects)
sqlite3.connect(":memory:").close()
witness = len(connects) - len(routes_connects)
counter = sum(len(found) for found in firewall.redirected.values())
firewall.uninstall()
sys.stderr.write(json.dumps({"counter": counter, "package": package, "package_connects": package_connects,
                             "package_threads": package_threads, "planted": planted, "routes": routes,
                             "routes_connects": routes_connects, "routes_threads": routes_threads,
                             "witness": witness}) + "\\n")
'''


def test_ao1_importing_the_package_or_the_route_module_loads_nothing_of_the_being(tmp_path):
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("_EXPORTS", "_COLLIDING"):
                    values[target.id] = node.value
    assert set(values) == {"_EXPORTS", "_COLLIDING"}
    exports = ast.literal_eval(values["_EXPORTS"])
    assert len(exports) > 50
    # The colliding names are the exported names that are also modules of the package: read by that rule.
    assert "_EXPORTS" in {getattr(node, "id", None) for node in ast.walk(values["_COLLIDING"])}
    colliding = {name for name in exports if (PACKAGE / f"{name}.py").exists()}
    assert colliding, "presence: some exported names are modules"
    words = list(exports) + [part for value in exports.values() for part in value] + sorted(colliding)
    for node in tree.body:
        if isinstance(node, ast.Import):
            words += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            words += [node.module or ""] + [alias.name for alias in node.names]
    assert not [word for word in words if "allium" in str(word).lower()], words
    code = _code(CHILD_IMPORT, repo=str(REPO), tests=str(TESTS), canary="ao1canary")
    returncode, _out, err, report = garden.finish(garden.child(tmp_path, code, cwd=tmp_path / "cwd"))
    assert returncode == 0 and report is not None, err[-2000:]
    assert report["planted"] == 1 and report["counter"] == 1, "the firewall counts, and nothing else was reached"
    assert report["package"] == ["opti_oignon", "opti_oignon.__version__"], report["package"]
    assert report["package_threads"] == 1 and report["package_connects"] == []
    assert _being(report["routes"]) == ["opti_oignon.api.routes_allium"], report["routes"]
    assert report["routes_threads"] == 1 and report["routes_connects"] == []
    assert report["witness"] == 1, "the audit hook counts a connect"


# ---------------------------------------------------------------------------
# AO4 -- the doors and the producers, closed and named
# ---------------------------------------------------------------------------
def _producers(overrides=None):
    names = census.package_modules("allium") + list(ENTRIES)
    names += [name for name in (overrides or ()) if name not in names]
    found = set()
    for module, where, method, keywords in census.calls(names, set(STORE_WRITES) | set(SERVICE_VERBS),
                                                        overrides=overrides, keep=True):
        if method == "append" and "transport" not in keywords:
            continue
        found.add((module, where, method))
    return found


def _transport_sites(names, overrides=None):
    sites = []
    for module, tree in census.trees(names, overrides, contains=("Transport",)):
        visitor = census._Scoped()

        def visit_Call(node, visitor=visitor, module=module):
            func = node.func
            if getattr(func, "id", None) == "Transport" or getattr(func, "attr", None) == "Transport":
                sites.append((module, visitor.where()))
            visitor.generic_visit(node)

        visitor.visit_Call = visit_Call
        visitor.visit(tree)
    return sites


def _sink_writers(texts):
    """The modules, outside the backend, that call ``set_light_sink`` or assign ``_LIGHT_SINK``."""
    found = []
    for module, text in texts.items():
        if module == _BACKEND or ("set_light_sink" not in text and "_LIGHT_SINK" not in text):
            continue
        for node in ast.walk(ast.parse(text)):
            called = isinstance(node, ast.Call) and "set_light_sink" in (getattr(node.func, "id", None),
                                                                         getattr(node.func, "attr", None))
            targets = (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
            assigned = any(getattr(t, "id", None) == "_LIGHT_SINK" or getattr(t, "attr", None) == "_LIGHT_SINK"
                           for target in targets for t in ast.walk(target))
            if called or assigned:
                found.append(module)
                break
    return found


def _route_calls(overrides=None):
    return census.calls(["api.routes_allium"], set(FORBIDDEN) | {"look", "close"}, overrides=overrides)


def test_ao4_the_being_is_entered_only_through_the_terminal_and_the_read_route_by_named_producers():
    assert census.importers() == frozenset(ENTRIES), sorted(census.importers())
    doors = {"opti_oignon.zz_door": "def f():\n    from opti_oignon.allium import service\n    return service\n",
             "opti_oignon.zz_by_name": "def f():\n    import importlib\n"
                                       "    return importlib.import_module('opti_oignon.allium.service')\n"}
    assert census.importers(overrides=doors) == frozenset(ENTRIES) | set(doors), \
        "witness: a new importer is found, by a statement and by a literal name"
    app = ast.parse((PACKAGE / "api" / "app.py").read_text(encoding="utf-8"))
    named = [alias.name for node in ast.walk(app) if isinstance(node, ast.ImportFrom)
             and (node.module or "").endswith("routes_allium") for alias in node.names]
    assert sorted(named) == ["close_garden", "health_flag", "router"], named
    assert not [node for node in ast.walk(app) if isinstance(node, ast.Import)
                and any("allium" in alias.name for alias in node.names)]
    assert _being(census.imports("opti_oignon.api.app")) == ["opti_oignon.api.routes_allium"]

    # The route module looks, and closes at shutdown; it never writes, and never reaches a method by name.
    calls = _route_calls()
    assert [call for call in calls if call[2] in FORBIDDEN] == [], calls
    assert [call for call in calls if call[2] == "look"], "presence: the route module looks"
    closes = [call[1] for call in calls if call[2] == "close"]
    assert closes and set(closes) == {"close_garden"}, closes
    routes_tree = census.parse_file(PACKAGE / "api" / "routes_allium.py")
    for node in ast.walk(routes_tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr":
            name = node.args[1] if len(node.args) > 1 else None
            assert isinstance(name, ast.Constant) and isinstance(name.value, str), ("getattr by a name", node.lineno)
            assert name.value not in set(FORBIDDEN) | {"look", "close"}, ("getattr on a garden", name.value)
    routes_text = (PACKAGE / "api" / "routes_allium.py").read_text(encoding="utf-8")
    planted = _route_calls({"opti_oignon.api.routes_allium": routes_text
                            + "\n\ndef _planted(garden, caller):\n    return garden.act('water', caller=caller)\n"})
    assert [call[2] for call in planted if call[2] in FORBIDDEN] == ["act"], "witness: a planted write is found"

    # Every producer, named; and the one place a transport is built.
    found = _producers()
    assert found == PRODUCERS, (sorted(found - PRODUCERS), sorted(PRODUCERS - found))
    describe = (PACKAGE / "allium" / "describe.py").read_text(encoding="utf-8")
    scratch = describe + "\n\ndef _planted(being, t):\n    return being.append('act', {'act': 'water'}, transport=t)\n"
    assert _producers({"opti_oignon.allium.describe": scratch}) - PRODUCERS == {
        ("opti_oignon.allium.describe", "_planted", "append")}, "witness: a planted producer is counted"
    everything = census.package_modules("opti_oignon")
    assert _transport_sites(everything) == [("opti_oignon.allium.service", "transport")]
    planted_transport = {"opti_oignon.zz_scratch": "from opti_oignon.allium import membrane\n\n\ndef f():\n"
                                                   "    return membrane.Transport('cli')\n"}
    assert ("opti_oignon.zz_scratch", "f") in _transport_sites(everything + ["opti_oignon.zz_scratch"],
                                                               planted_transport)

    # No module registers a light sink or assigns the backend's.
    known = census.modules()
    texts = {name: path.read_text(encoding="utf-8") for name, path in known.items()}
    assert len(texts) > 300
    assert _sink_writers(texts) == []
    texts["opti_oignon.zz_wire"] = "from opti_oignon import inference_backend\ninference_backend.set_light_sink(print)\n"
    assert _sink_writers(texts) == ["opti_oignon.zz_wire"], "witness: a registration is found"


# ---------------------------------------------------------------------------
# AO5 -- the passive light hook
# ---------------------------------------------------------------------------
class _Beneath:
    """A profile function: every event inside the hook's frame or below it."""

    def __init__(self, code):
        self.code = code
        self.depth = 0
        self.entered = 0
        self.events = []

    def __call__(self, frame, event, arg):
        if frame.f_code is self.code and event == "call":
            self.entered += 1
            self.depth += 1
            return
        if frame.f_code is self.code and event == "return":
            self.depth -= 1
            return
        if self.depth:
            self.events.append((event, getattr(arg, "__name__", frame.f_code.co_name)))


class _Recording:
    """While open: every ``os.stat``/``os.lstat``, every import attempt, and every profiler event beneath
    ``code`` when one is given; the ``allium`` keys the module cache gained.

    An import attempt is every call of ``builtins.__import__`` (each ``import``
    statement, cached or not, with its ``from`` names) and of
    ``importlib.import_module``, which raises no audit event of its own. The
    two are wrapped only while the recording is open: an audit hook could not
    be taken back, and would be paid by every suite that runs after.
    """

    def __init__(self, code=None):
        self.beneath = _Beneath(code) if code is not None else None
        self.stats = []
        self.imports = []

    def __enter__(self):
        self._stat, self._lstat, self._import_module = os.stat, os.lstat, importlib.import_module
        self._import = builtins.__import__
        stats, imports = self.stats, self.imports
        real_stat, real_lstat, real_import_module = self._stat, self._lstat, self._import_module
        real_import = self._import

        def stat(path, *args, **kwargs):
            stats.append(os.fspath(path) if isinstance(path, (str, bytes, os.PathLike)) else path)
            return real_stat(path, *args, **kwargs)

        def lstat(path, *args, **kwargs):
            stats.append(os.fspath(path) if isinstance(path, (str, bytes, os.PathLike)) else path)
            return real_lstat(path, *args, **kwargs)

        def import_module(name, package=None):
            imports.append(name)
            return real_import_module(name, package)

        def import_(name, globals=None, locals=None, fromlist=(), level=0):
            imports.append(name)
            imports.extend(name + "." + str(item) for item in (fromlist or ()))
            return real_import(name, globals, locals, fromlist, level)

        self._before = {key for key in sys.modules if "allium" in key}
        os.stat, os.lstat, importlib.import_module = stat, lstat, import_module
        builtins.__import__ = import_
        self._profile = sys.getprofile()
        if self.beneath is not None:
            sys.setprofile(self.beneath)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self._profile)
        builtins.__import__ = self._import
        os.stat, os.lstat, importlib.import_module = self._stat, self._lstat, self._import_module
        self.new = {key for key in sys.modules if "allium" in key} - self._before
        return False

    def named(self, word):
        return [name for name in self.imports if word in name]


class _Ollama:
    """The client library, scripted: a chat, a stream, an embedding; ``touch`` stats and imports when it chats."""

    def __init__(self, touch=None):
        self.touch = touch

    def list(self):
        return {"models": [{"name": MODEL}]}

    def show(self, name):
        return {"details": {}, "model_info": {}}

    def chat(self, **kw):
        if self.touch is not None:
            try:
                os.stat(self.touch)
            except OSError:
                pass
            try:
                import zz_planted  # noqa: F401 -- the attempt is the witness
            except ImportError:
                pass
        if kw.get("stream"):
            return iter([{"message": {"content": REPLY}, "done": False},
                         {"message": {"content": ""}, "done": True, "eval_count": 3, "prompt_eval_count": 7}])
        return {"message": {"content": REPLY}, "eval_count": 5, "prompt_eval_count": 11}

    def embed(self, model, input):
        n = 1 if isinstance(input, str) else len(input)
        return {"embeddings": [[0.5, 0.25]] * n, "prompt_eval_count": 4}


def _open_backend(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    loaded, restore = isolate(targets={_BACKEND: source("inference_backend.py")}, packages=("opti_oignon",))
    # The being is absent from the cache, so an attempt to import it is an import, and is seen.
    sys.modules.pop("opti_oignon.allium", None)
    return loaded[_BACKEND], restore


def _turn(mod, client):
    mod.OLLAMA_AVAILABLE = True
    mod._ollama_module = client
    mod._live_mode = lambda: "daily"
    registry = mod.BackendRegistry()
    registry.register(mod.OllamaBackend())
    registry.activate("ollama")
    backend = registry.resolve_backend(MODEL)
    messages = [{"role": "user", "content": PROMPT}]
    assert backend.generate(MODEL, messages).content == REPLY
    assert [chunk.content for chunk in backend.stream(MODEL, messages)][0] == REPLY
    assert backend.embed(MODEL, PROMPT) is not None
    assert backend.embed_many(MODEL, [PROMPT, PROMPT]) is not None


def _row(count):
    return tuple(getattr(count, slot) for slot in SLOTS)


def test_ao5_with_no_light_sink_the_hook_and_a_chat_turn_import_and_stat_nothing_of_the_being(
        monkeypatch, tmp_path, caplog):
    mod, restore = _open_backend(monkeypatch)
    try:
        # (1) Nothing is registered, and the hook does nothing beneath itself.
        assert mod._LIGHT_SINK is None
        for head, counts in HEADS:
            with _Recording(mod._light.__code__) as seen:
                assert mod._light(head, "ollama", dict(counts)) is None
            assert seen.beneath.entered == 1 and seen.beneath.events == [], (head, seen.beneath.events[:8])
            assert seen.stats == [] and seen.named("allium") == [] and seen.new == set(), head

        # (2) A witness sink: it stats and tries to import the being, and the recorder sees both.
        witness = str(tmp_path / "witness")
        records = []

        def sink(count):
            records.append(count)
            try:
                os.stat(witness)
            except OSError:
                pass
            try:
                import opti_oignon.allium  # noqa: F401 -- refused in the window; the attempt is the witness
            except ImportError:
                pass

        assert mod.set_light_sink(sink) is None
        with _Recording(mod._light.__code__) as seen:
            for head, counts in HEADS:
                mod._light(head, "ollama", dict(counts))
        assert seen.beneath.entered == 4 and seen.beneath.events, "witness: the profiler sees work beneath the hook"
        assert seen.stats.count(witness) == 4 and len(seen.named("opti_oignon.allium")) == 4, (seen.stats,
                                                                                               seen.imports)
        assert [_row(count) for count in records] == RECORDS, [_row(count) for count in records]

        # (3) A record is closed: slots, each a code, a count or nothing.
        for count in records:
            assert not hasattr(count, "__dict__") and tuple(type(count).__slots__) == SLOTS
            assert all(value is None or type(value) in (int, str) for value in _row(count)), _row(count)

        # (4) A sink that raises never breaks the hook; its record names the class alone.
        canary = "canary-of-a-broken-sink"

        def broken(count):
            raise RuntimeError(canary)

        assert mod.set_light_sink(broken) is sink
        caplog.set_level(logging.DEBUG)
        caplog.clear()
        assert mod._light("generate", "ollama", {"eval_count": 1}) is None
        said = [record.getMessage() for record in caplog.records if record.name == _BACKEND]
        assert any("RuntimeError" in message for message in said) and not any(canary in m for m in said), said

        # (5) Registration: the previous sink back; a sink that cannot be called refused.
        assert mod.set_light_sink(None) is broken and mod._LIGHT_SINK is None
        with pytest.raises(TypeError):
            mod.set_light_sink(42)
        assert mod._LIGHT_SINK is None

        # (6) A count that is not one is unknown, never a number; reported exactly when there is a number.
        records.clear()
        mod.set_light_sink(records.append)
        for bad in ("5", True, -1):
            mod._light("generate", "ollama", {"eval_count": bad, "prompt_eval_count": bad})
        mod._light("stream", "ollama", {})
        for head, counts in HEADS:
            mod._light(head, "ollama", dict(counts))
        for row in map(_row, records[:4]):
            assert row[2:] == (None, "unknown", None, "unknown"), row
        for row in map(_row, records):
            assert (row[2] is None) == (row[3] != "reported") and (row[4] is None) == (row[5] != "reported"), row
        assert mod.set_light_sink(None) == records.append

        # (7) A chat turn with no sink: nothing of the being imported or stat-ed.
        with _Recording() as seen:
            _turn(mod, _Ollama())
        assert seen.stats == [] and seen.named("allium") == [] and seen.new == set(), (seen.stats, seen.imports)
        touched = str(tmp_path / "touched")
        with _Recording() as seen:
            _turn(mod, _Ollama(touch=touched))
        assert touched in seen.stats and seen.named("zz_planted"), "witness: the recorder sees a stat and an import"
        records.clear()
        mod.set_light_sink(records.append)
        _turn(mod, _Ollama())
        # No head reports yet (clause 8): the turn gives the sink nothing. The scan below has a subject once
        # a later change adds the call sites, and supersedes this pin by name.
        assert records == [], [_row(count) for count in records]
        for row in map(_row, records):
            for value in row:
                assert value is None or not any(word in str(value) for word in (REPLY, PROMPT, MODEL)), row
        mod.set_light_sink(None)
    finally:
        restore()

    # (8) Nothing in the package calls the hook yet.
    def hook_calls(names, overrides=None):
        found = []
        for module, tree in census.trees(names, overrides, contains=("_light",)):
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and "_light" in (getattr(node.func, "id", None),
                                                               getattr(node.func, "attr", None)):
                    found.append(module)
        return found

    everything = census.package_modules("opti_oignon")
    assert hook_calls(everything) == []
    planted = {"opti_oignon.zz_head": 'def f():\n    _light("generate", "x", {})\n'}
    assert hook_calls(["opti_oignon.zz_head"], planted) == ["opti_oignon.zz_head"], "witness: a call is found"
    assert json.dumps(RECORDS)
