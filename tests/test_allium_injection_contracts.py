#!/usr/bin/env python3
"""Contracts that keep the componion out of reach of the model, of remote requests and of plugins.

The being is the person's simulation, not a tool: nothing the model can call
imports it, no untrusted block is labelled with it, no remote request may
name it, and a sandboxed plugin cannot import it or the two modules that
open it. What it may reach of the platform is a closed list.

  * IJ1 -- no model-tool entry and no plugin loader reaches the being: the
    import closure of the tool entries (an import by a literal name
    followed), and the dynamic imports that closure cannot follow, pinned by
    name.
  * IJ2 -- no tool schema or allowlist names the being, and remote inference
    refuses each of its names as a forbidden capability, by name.
  * IJ3 -- a sandboxed plugin cannot import the being or its entry modules,
    cached or not: through the finder by a real statement, through a
    platform helper (a statement, importlib by name, importlib's own
    ``__import__``), by a relative import whose package it forges, or after
    renaming itself in a real load; a platform caller is never refused.
  * IJ8 -- the executor hub never imports the being, nor labels an untrusted
    source with it: every source constant, every labelled call, and the
    labels that are not literals pinned by where they are made.
  * IJ11 -- the being reaches only its closed list of platform modules, and
    only the names it reads of each.

Local-only (the public distribution ships no tests). The censuses read the
syntax trees (``tests/_allium_census.py``); remote inference and the plugin
loader load through the shared isolation window with their platform
dependencies stood in.
"""

import ast
import builtins
import re
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _allium_census as census  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_ij1_no_model_tool_entry_and_no_plugin_loader_reaches_the_being": 2.0,
    "test_ij2_no_tool_schema_or_allowlist_names_the_being_and_remote_inference_refuses_it_by_name": 2.0,
    "test_ij3_a_sandboxed_plugin_cannot_import_the_being_or_its_entry_modules_cached_or_not": 2.0,
    "test_ij8_the_executor_hub_never_imports_the_being_nor_labels_an_untrusted_source_with_it": 2.0,
    "test_ij11_the_being_reaches_only_its_closed_list_of_platform_modules": 2.0,
}
PACKAGE = REPO / "opti_oignon"
ENTRIES = ("opti_oignon.cli.garden", "opti_oignon.api.routes_allium")
TOOL_ROOTS = ("agent.tools", "agent.dispatch", "agent.skills", "agent.loop", "tool_registry", "tool_executor",
              "tool_calling", "agentic_executor", "plugin_loader")
HUB_ROOTS = ("executor", "agent.loop", "tool_executor", "agentic_executor")
# What the closure of the tool entries imports by a name it computes: it cannot follow these, so they are named.
DYNAMIC_SITES = frozenset({
    ("opti_oignon", "_import", "import_module"),
    ("opti_oignon.api.deps", "_module_exists", "find_spec"),
    ("opti_oignon.db_encryption", "<module>", "__import__"),
    ("opti_oignon.lazy_loader", "LazyAttr._resolve", "import_module"),
    ("opti_oignon.lazy_loader", "LazyModule._load", "import_module"),
    ("opti_oignon.learned_router", "<module>", "find_spec"),
    ("opti_oignon.plugin_loader", "PluginLoader._load_plugin_inprocess", "spec_from_file_location"),
    ("opti_oignon.tool_registry", "ToolRegistry._check_requirement", "__import__"),
})
# The untrusted-context calls, the ones that take their labels as ``(label, text)`` items, and the calls whose
# label is neither a literal nor a source constant, by where they are made.
UNTRUSTED = ("wrap", "untrusted_message", "wrap_items", "untrusted_message_many", "memory_untrusted_message")
ITEMS = ("wrap_items", "untrusted_message_many")
LABEL_SITES = frozenset({
    ("opti_oignon.agent.loop", "_capped_observations_message", "untrusted_message_many"),
    ("opti_oignon.agent.loop", "_observations_message", "untrusted_message_many"),
    ("opti_oignon.agent.teacher", "_failure_context", "wrap_items"),
    ("opti_oignon.agent.untrusted_context", "memory_untrusted_message", "untrusted_message"),
    ("opti_oignon.agent.untrusted_context", "untrusted_message", "wrap"),
    ("opti_oignon.agent.untrusted_context", "untrusted_message_many", "wrap_items"),
})
LABEL_TOKENS = ("allium", "componion", "garden", "pet", "beetle")
SOURCE_NAME = re.compile(r"(?:^|_)SOURCE_\w+$")
# The model's tool surface and the four names of the being a remote request may never carry.
TOOL_FILES = ("opti_oignon/agent/tools.py", "opti_oignon/agent/allowlists.py", "opti_oignon/agent/dispatch.py",
              "opti_oignon/tool_registry.py", "opti_oignon/tool_calling.py", "opti_oignon/tool_executor.py")
BEING_NAMES = ("allium", "componion", "garden", "pet")
REQUEST = {"v": 1, "type": "remote_infer", "device": "phone", "request_id": "r1", "prompt": "hello"}
# What the being may reach of the platform: each module, and the names it reads of it.
REACH = {
    "opti_oignon.native": {"load"},
    "opti_oignon.security_mode": {"SecurityModeManager", "_SECURITY_YAML", "_LOCKFILE_PATH"},
    "opti_oignon.db_utils": {"safe_connect"},
    "opti_oignon.db_encryption": {"SQLCIPHER_AVAILABLE"},
    "opti_oignon.encryption": {"_DEFAULT_KEYFILE", "_ENV_KEY_NAME", "decrypt_bytes", "encrypt_bytes",
                               "get_encryption_key"},
    "opti_oignon.config": {"DATA_DIR"},
    "opti_oignon.signed_audit_log": {"signed_audit_log"},
}
BLOCKED_ENTRIES = ("opti_oignon.allium", "opti_oignon.allium.service", "opti_oignon.api.routes_allium",
                   "opti_oignon.cli.garden")


@pytest.fixture(autouse=True)
def _no_thread_left():
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before, "a contract left a thread running"


def _being(names):
    return sorted(name for name in names
                  if name == "opti_oignon.allium" or name.startswith("opti_oignon.allium.") or name in ENTRIES)


def _text(rel):
    return (REPO / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# IJ11 -- the being's outgoing reach
# ---------------------------------------------------------------------------
def _own_imports(body):
    """The import statements of one scope, not of the functions or classes nested in it."""
    out = []
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        else:
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(node, field, None)
                if isinstance(inner, list):
                    out.extend(_own_imports(inner))
            for handler in getattr(node, "handlers", None) or ():
                out.extend(_own_imports(handler.body))
    return out


def _outside(name):
    return census._project(name) and not (name == "opti_oignon.allium" or name.startswith("opti_oignon.allium."))


def reach(overrides=None):
    """``{platform module: names read}`` of every module under ``opti_oignon/allium``.

    A module a scope imports is bound to its local name there; the names read
    are its attributes and the literal names ``getattr`` reads on it, in that
    scope and the ones nested in it. ``from opti_oignon import X`` binds the
    module ``X`` when ``X`` is one, else reads ``X`` of the facade; a bare
    ``import opti_oignon.X`` binds the facade. A string constant that names a
    module outside the package counts that module.
    """
    known = census.modules()
    names = census.package_modules("allium") + [n for n in (overrides or ()) if n.startswith("opti_oignon.allium")]
    table = {}
    for module, tree in census.trees(sorted(set(names)), overrides, keep=True):
        scopes = [tree] + [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for scope in scopes:
            bound = {}
            for node in _own_imports(scope.body):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _outside(alias.name) or alias.name == "opti_oignon":
                            target = alias.name if alias.asname else "opti_oignon"
                            table.setdefault(target, set())
                            bound[alias.asname or "opti_oignon"] = target
                    continue
                base = census._base(node, module, census._is_package(module, known))
                if not base or not census._project(base) or not (_outside(base) or base == "opti_oignon"):
                    continue
                for alias in node.names:
                    child = base + "." + alias.name
                    if child in known or (overrides and child in overrides):
                        table.setdefault(child, set())
                        bound[alias.asname or alias.name] = child
                    else:
                        table.setdefault(base, set()).add(alias.name)
            for node in ast.walk(scope):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bound:
                    table[bound[node.value.id]].add(node.attr)
                if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr" and len(node.args) > 1
                        and isinstance(node.args[0], ast.Name) and node.args[0].id in bound
                        and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
                    table[bound[node.args[0].id]].add(node.args[1].value)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and re.fullmatch(r"opti_oignon(\.[A-Za-z_]\w*)+", node.value) and _outside(node.value)):
                table.setdefault(node.value, set())
    return table


def test_ij11_the_being_reaches_only_its_closed_list_of_platform_modules():
    found = reach()
    assert len(found) == 7, sorted(found)
    assert found == REACH, {name: sorted(found.get(name, ())) for name in set(found) | set(REACH)
                            if found.get(name) != REACH.get(name)}
    service = (PACKAGE / "allium" / "service.py").read_text(encoding="utf-8")
    mode = (PACKAGE / "allium" / "mode.py").read_text(encoding="utf-8")
    planted = reach({"opti_oignon.allium.service": service + "\n\ndef _planted():\n"
                                                             "    from opti_oignon.notes import notes_store\n"
                                                             "    return notes_store\n"})
    assert "opti_oignon.notes" in planted or "opti_oignon.notes.notes_store" in planted, "witness: a new module"
    planted = reach({"opti_oignon.allium.service": service + "\n\ndef _planted():\n    import opti_oignon\n"
                                                             "    return opti_oignon.analyze\n"})
    assert "analyze" in planted.get("opti_oignon", set()), "witness: the facade is reached"
    marker = "from opti_oignon import security_mode as sm"
    assert mode.count(marker) == 1
    planted = reach({"opti_oignon.allium.mode": mode.replace(marker, marker + "\n        getattr(sm, '_PRIVATE')")})
    assert "_PRIVATE" in planted["opti_oignon.security_mode"], "witness: a name read by getattr"


# ---------------------------------------------------------------------------
# IJ1 -- the model's tools and the plugin loader
# ---------------------------------------------------------------------------
def test_ij1_no_model_tool_entry_and_no_plugin_loader_reaches_the_being():
    reached = census.closure(TOOL_ROOTS)
    assert "opti_oignon.inference_backend" in reached and len(reached) > 100, len(reached)
    assert _being(reached) == [], _being(reached)
    dynamic = set(census.dynamic_imports(sorted(reached)))
    assert dynamic == DYNAMIC_SITES, (sorted(dynamic - DYNAMIC_SITES), sorted(DYNAMIC_SITES - dynamic))
    calling = _text("opti_oignon/tool_calling.py")
    chain = {"opti_oignon.tool_calling": calling + "\n\ndef _planted_hop():\n    from opti_oignon import zz_one  # noqa\n",
             "opti_oignon.zz_one": "from opti_oignon import zz_two  # noqa\n",
             "opti_oignon.zz_two": "from opti_oignon import zz_three  # noqa\n",
             "opti_oignon.zz_three": "from opti_oignon.allium import service  # noqa\n"}
    assert "opti_oignon.allium.service" in census.closure(TOOL_ROOTS, overrides=chain), "witness: a deep chain"
    literal = {"opti_oignon.tool_calling": calling + "\n\ndef _planted_literal():\n    import importlib\n"
                                                     "    return importlib.import_module('opti_oignon.allium.service')\n"}
    assert "opti_oignon.allium.service" in census.closure(TOOL_ROOTS, overrides=literal), \
        "witness: an import by a literal name is followed"
    dispatch = _text("opti_oignon/agent/dispatch.py")
    planted = {"opti_oignon.agent.dispatch": dispatch + "\n\ndef _planted(name):\n    import importlib\n"
                                                        "    return importlib.import_module(name)\n"}
    found = census.dynamic_imports(["agent.dispatch"], overrides=planted)
    assert ("opti_oignon.agent.dispatch", "_planted", "import_module") in found, "witness: a computed import"


# ---------------------------------------------------------------------------
# IJ8 -- the hub, and the labels of untrusted blocks
# ---------------------------------------------------------------------------
def _module_constants(tree):
    """``{name: value}`` of every string constant bound at module scope (in a ``try`` or ``if`` too)."""
    out = {}

    def walk(statements):
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        out.setdefault(target.id, []).append(node.value.value)
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(node, field, None)
                if isinstance(inner, list):
                    walk(inner)
            for handler in getattr(node, "handlers", None) or ():
                walk(handler.body)

    walk(tree.body)
    return out


def labels(overrides=None):
    """``(source constants, labels checked, sites)`` over the package outside ``allium/``.

    Source constants: every module-scope string bound to a ``SOURCE_`` name.
    A label is the ``source=`` of ``wrap``, ``untrusted_message`` and
    ``memory_untrusted_message``, or the first element of each ``(label,
    text)`` item of ``wrap_items`` and ``untrusted_message_many``: a literal
    is checked, and so is a ``SOURCE_`` name bound to a constant; anything else
    makes the call a site, ``(module, function, callee)``, and every tuple
    literal's first string in that function is checked too.
    """
    names = census.package_modules("opti_oignon") + [n for n in (overrides or ()) if n not in census.modules()]
    trees = [(module, tree) for module, tree in census.trees(names, overrides, contains=("SOURCE_", "untrusted_context"))
             if not (module == "opti_oignon.allium" or module.startswith("opti_oignon.allium."))]
    constants = {}
    for module, tree in trees:
        for name, values in _module_constants(tree).items():
            if SOURCE_NAME.search(name):
                constants.setdefault(name, []).extend((module, value) for value in values)
    checked, sites, functions = [], set(), {}
    for module, tree in trees:
        local = {}
        own = module.endswith(".untrusted_context")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("untrusted_context"):
                for alias in node.names:
                    if alias.name in UNTRUSTED:
                        local[alias.asname or alias.name] = alias.name
        visitor = census._Scoped()
        stack = []

        def label(node, where, callee, module=module):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                checked.append((module, where, node.value))
                return True
            ident = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if ident and SOURCE_NAME.search(ident) and ident in constants:
                checked.extend((module, where, value) for _m, value in constants[ident])
                return True
            return False

        def visit_FunctionDef(node, visitor=visitor):
            stack.append(node)
            visitor.stack.append(node.name)
            visitor.generic_visit(node)
            visitor.stack.pop()
            stack.pop()

        def visit_Call(node, visitor=visitor, module=module, local=local, own=own):
            func, callee = node.func, None
            if isinstance(func, ast.Attribute) and func.attr in UNTRUSTED:
                holder = func.value
                if getattr(holder, "id", None) == "untrusted_context" or getattr(holder, "attr", None) == \
                        "untrusted_context":
                    callee = func.attr
            elif isinstance(func, ast.Name):
                callee = local.get(func.id) or (func.id if own and func.id in UNTRUSTED else None)
            if callee is not None:
                where = visitor.where()
                literal = True
                if callee in ITEMS:
                    items = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "items"),
                                                                 None)
                    if isinstance(items, (ast.List, ast.Tuple)):
                        for item in items.elts:
                            if not (isinstance(item, ast.Tuple) and item.elts and label(item.elts[0], where, callee)):
                                literal = False
                    elif items is not None:
                        literal = False
                else:
                    given = next((k.value for k in node.keywords if k.arg == "source"), None)
                    if given is not None and not label(given, where, callee):
                        literal = False
                if not literal:
                    sites.add((module, where, callee))
                    if stack:
                        functions[(module, where)] = stack[-1]
            visitor.generic_visit(node)

        visitor.visit_FunctionDef = visit_FunctionDef
        visitor.visit_AsyncFunctionDef = visit_FunctionDef
        visitor.visit_Call = visit_Call
        visitor.visit(tree)
    tuples = []
    for (module, where), fn in functions.items():
        for node in ast.walk(fn):
            if isinstance(node, ast.Tuple) and node.elts and isinstance(node.elts[0], ast.Constant) \
                    and isinstance(node.elts[0].value, str):
                tuples.append((module, where, node.elts[0].value))
    checked.extend(tuples)
    return constants, checked, sites, tuples


def _named(values):
    return [value for value in values if set(census.words(value[-1])) & set(LABEL_TOKENS)]


def test_ij8_the_executor_hub_never_imports_the_being_nor_labels_an_untrusted_source_with_it():
    reached = census.closure(HUB_ROOTS)
    assert "opti_oignon.inference_backend" in reached and len(reached) > 100, len(reached)
    assert _being(reached) == [], _being(reached)
    executor = _text("opti_oignon/executor.py")
    for planted_hop in ("    from opti_oignon import zz_hub  # noqa\n",
                        "    import importlib\n    importlib.import_module('opti_oignon.zz_hub')\n"):
        hub = {"opti_oignon.executor": executor + "\n\ndef _planted_hop():\n" + planted_hop,
               "opti_oignon.zz_hub": "def g():\n    from opti_oignon.allium import service  # noqa\n"}
        assert "opti_oignon.allium.service" in census.closure(HUB_ROOTS, overrides=hub), \
            ("witness: a chain under the hub whose first hop is in a function", planted_hop)
    constants, checked, sites, tuples = labels()
    values = [(name, module, value) for name, found in constants.items() for module, value in found]
    assert len(values) >= 20 and _named(values) == [], _named(values)
    assert len(checked) >= 10 and _named(checked) == [], _named(checked)
    assert tuples and {value for _m, _w, value in tuples} >= {"tool"}, tuples
    assert sites == LABEL_SITES, (sorted(sites - LABEL_SITES), sorted(LABEL_SITES - sites))
    planted = {"opti_oignon.agent.zz_planted": (
        "from opti_oignon.agent import untrusted_context\n"
        "from .untrusted_context import untrusted_message\n\n\n"
        "def f(x):\n    return untrusted_message(x, source='garden')\n\n\n"
        "def g(y):\n    return untrusted_context.wrap_items([('componion', y)])\n\n\n"
        "def h(x, label):\n    return untrusted_message(x, source=label)\n")}
    _constants, checked, sites, _tuples = labels(planted)
    named = {(module, where, value) for module, where, value in _named(checked)}
    assert {("opti_oignon.agent.zz_planted", "f", "garden"), ("opti_oignon.agent.zz_planted", "g", "componion")} \
        <= named, named
    assert ("opti_oignon.agent.zz_planted", "h", "untrusted_message") in sites, "witness: a new site is listed"


# ---------------------------------------------------------------------------
# IJ2 -- the tool surface, and remote inference
# ---------------------------------------------------------------------------
def _veilid_stand_ins():
    guard = types.ModuleType("opti_oignon.veilid.guard")

    class VeilidDisabledInBulbe(Exception):
        pass

    guard.VeilidDisabledInBulbe = VeilidDisabledInBulbe
    guard.NETWORK_ISOLATED = False
    guard.assert_sync_allowed = lambda: None
    protocol = types.ModuleType("opti_oignon.veilid.protocol")
    protocol.PROTOCOL_VERSION = 1
    protocol.MSG_REMOTE_INFER = "remote_infer"
    protocol.MSG_REMOTE_INFER_CONT = "remote_infer_cont"
    streaming = types.ModuleType("opti_oignon.veilid.remote_streaming")
    streaming.check_rate = lambda peer: True
    streaming.open_session = lambda origin, rid, chunks: None
    streaming.pull = lambda origin, rid, cursor: None
    return {"opti_oignon.veilid.guard": guard, "opti_oignon.veilid.protocol": protocol,
            "opti_oignon.veilid.remote_streaming": streaming}


def test_ij2_no_tool_schema_or_allowlist_names_the_being_and_remote_inference_refuses_it_by_name():
    hits, read = census.strings(TOOL_FILES, BEING_NAMES)
    assert read > 1000 and hits == [], (read, hits)
    tools = _text(TOOL_FILES[0])
    planted, _read = census.strings(TOOL_FILES[:1], BEING_NAMES,
                                    texts={TOOL_FILES[0]: tools + '\nPLANTED = {"name": "garden_look"}\n'})
    assert [(value, word) for _file, value, word in planted] == [("garden_look", "garden")], "witness: a schema name"

    loaded, restore = isolate(targets={"opti_oignon.veilid.remote_inference": source("veilid", "remote_inference.py")},
                              seeded=_veilid_stand_ins(), packages=("opti_oignon.veilid",))
    try:
        remote = loaded["opti_oignon.veilid.remote_inference"]
        assert "prompt" in remote._ALLOWED_FIELDS and "cursor" in remote._CONT_ALLOWED_FIELDS
        for name in BEING_NAMES:
            assert name not in remote._ALLOWED_FIELDS and name not in remote._CONT_ALLOWED_FIELDS, name
            assert name in remote._FORBIDDEN_FIELDS, name
        assert remote._enforce_bounded_surface(dict(REQUEST), "r1") is None, "the request itself is in surface"
        forbidden = remote._enforce_bounded_surface(dict(REQUEST, tool=1), "r1")
        assert forbidden["reason"] == "out_of_surface" and "names a capability" in forbidden["detail"], forbidden
        for name in BEING_NAMES:
            refusal = remote._enforce_bounded_surface(dict(REQUEST, **{name: 1}), "r1")
            assert refusal is not None and refusal["reason"] == "out_of_surface", (name, refusal)
            detail = refusal["detail"]
            assert "'" + name + "'" in detail and "names a capability" in detail and "componion" in detail, (name, detail)
        control = remote._enforce_bounded_surface(dict(REQUEST, zzz=1), "r1")
        assert control["reason"] == "out_of_surface" and "is not part of the tier 1 bounded surface" in \
            control["detail"] and "componion" not in control["detail"], control
    finally:
        restore()


# ---------------------------------------------------------------------------
# IJ3 -- the plugin sandbox
# ---------------------------------------------------------------------------
PLUGIN_IMPORTS = '''
def attempt(violation):
    outcomes = []
    try:
        import opti_oignon.allium  # noqa: F401
        outcomes.append("passed")
    except violation:
        outcomes.append("refused")
    try:
        from opti_oignon import allium  # noqa: F401
        outcomes.append("passed")
    except violation:
        outcomes.append("refused")
    try:
        from opti_oignon.allium import service  # noqa: F401
        outcomes.append("passed")
    except violation:
        outcomes.append("refused")
    try:
        import opti_oignon.api.routes_allium  # noqa: F401
        outcomes.append("passed")
    except violation:
        outcomes.append("refused")
    try:
        import json  # noqa: F401
        outcomes.append("passed")
    except violation:
        outcomes.append("refused")
    return outcomes


def find(importer, names, violation):
    outcomes = []
    for name in names:
        try:
            outcomes.append(importer.find_spec(name))
        except violation:
            outcomes.append("refused")
    return outcomes
'''


# Plugin code that reaches the being another way: through a platform helper, by a relative import whose
# package it forges, or by a real import statement that only the finder can answer.
PLUGIN_TRICKS = '''
def through(helper, violation):
    try:
        helper()
        return "passed"
    except violation:
        return "refused"


def forged(violation, importer=None):
    class Spec:
        parent = "opti_oignon"

    outcomes = []
    for scope in ({"__spec__": Spec(), "__name__": "zz"}, {"__package__": None, "__spec__": Spec(), "__name__": "zz"}):
        try:
            if importer is None:
                __import__("allium", scope, None, ("service",), 1)
            else:
                importer(scope)
            outcomes.append("passed")
        except violation:
            outcomes.append("refused")
        except ImportError:
            outcomes.append("import_error")
    return outcomes


def real_import(violation):
    try:
        import opti_oignon.allium  # noqa: F401
        return "passed"
    except violation:
        return "refused"
    except ImportError:
        return "import_error"


def own_relative(violation):
    try:
        __import__("notes", {"__package__": "opti_oignon", "__name__": "zz"}, None, (), 1)
        return "passed"
    except violation:
        return "refused"
    except ImportError:
        return "import_error"


def shifting(violation, importer):
    class Shifting:
        read = 0

        @property
        def parent(self):
            Shifting.read += 1
            return "zz_elsewhere" if Shifting.read == 1 else "opti_oignon"

    try:
        importer({"__spec__": Shifting(), "__name__": "zz"})
        return "passed"
    except violation:
        return "refused"
    except ImportError:
        return "import_error"
'''
# A platform helper a plugin may call: it imports by a statement, by name through importlib, or relatively.
PLATFORM_HELPER = '''
import importlib


def by_statement():
    import opti_oignon.allium  # noqa: F401


def by_name():
    return importlib.import_module("opti_oignon.allium.service")


def by_importlib_import():
    return importlib.__import__("opti_oignon.allium", None, None, ("service",), 0)


def relative(scope):
    return __import__("allium", scope, None, ("service",), 1)
'''
# A plugin that renames itself before it imports the being: it is still plugin code.
RENAMED_PLUGIN = (
    "__name__ = 'json'\n"
    "try:\n"
    "    from opti_oignon import allium  # noqa: F401\n"
    "    OUTCOME = 'passed'\n"
    "except Exception as exc:  # noqa: BLE001\n"
    "    OUTCOME = type(exc).__name__\n")


def _as_module(name, text=PLUGIN_IMPORTS):
    """``text`` (``PLUGIN_IMPORTS`` by default) compiled as the code of a module called ``name``: its functions."""
    namespace = {"__name__": name, "__builtins__": builtins}
    # Compiled and run by the test itself, before any sandbox is entered.
    exec(compile(text, "<" + name + ">", "exec"), namespace)
    return namespace


def _manifest_stand_in():
    module = types.ModuleType("opti_oignon.plugin_manifest")

    class PluginManifestError(Exception):
        pass

    class PluginManifest:
        @classmethod
        def from_dict(cls, data):
            return types.SimpleNamespace(name=data["name"], version=str(data["version"]),
                                         entry_point=data["entry_point"], permissions=[], hooks=[])

    module.PluginManifest, module.PluginManifestError = PluginManifest, PluginManifestError
    return module


def _stand_in(name, package=False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    return module


def test_ij3_a_sandboxed_plugin_cannot_import_the_being_or_its_entry_modules_cached_or_not(tmp_path):
    plugin = _as_module("_opti_plugin_x")
    platform = _as_module("zz_platform_code")
    tricks = _as_module("_opti_plugin_tricks", PLUGIN_TRICKS)
    helper = _as_module("zz_platform_helper", PLATFORM_HELPER)

    # Uncached: none of the names is in the module cache, and the finder answers.
    loaded, restore = isolate(targets={"opti_oignon.plugin_loader": source("plugin_loader.py")},
                              blocked=("opti_oignon.config", "opti_oignon.plugin_manifest"))
    try:
        loader = loaded["opti_oignon.plugin_loader"]
        for name in BLOCKED_ENTRIES:
            sys.modules.pop(name, None)
        importer = loader._RestrictedImporter(loader._BLOCKED_IMPORTS, loader._NETWORK_MODULES)
        refused = plugin["find"](importer, BLOCKED_ENTRIES, loader.PluginSandboxViolation)
        assert refused == ["refused"] * 4, refused
        allowed = plugin["find"](importer, ("opti_oignon.alliumx", "opti_oignon.notes", "json"),
                                 loader.PluginSandboxViolation)
        assert allowed == [None, None, None], allowed
        assert platform["find"](importer, ("opti_oignon.allium",), loader.PluginSandboxViolation) == [None], \
            "witness: the rule holds only with a plugin on the stack"
        # A real import statement, with only the finder in place: the import system stands between the plugin's
        # frame and the finder, and the plugin is still on the stack.
        sys.meta_path.insert(0, importer)
        try:
            real = tricks["real_import"](loader.PluginSandboxViolation)
        finally:
            sys.meta_path.remove(importer)
        assert real == "refused", real
        assert "opti_oignon.allium" not in sys.modules
    finally:
        restore()

    # Cached: the modules stand in the cache, and the import wrapper answers.
    allium = _stand_in("opti_oignon.allium", package=True)
    service = _stand_in("opti_oignon.allium.service")
    allium.service = service
    seeded = {"opti_oignon.allium": allium, "opti_oignon.allium.service": service,
              "opti_oignon.api.routes_allium": _stand_in("opti_oignon.api.routes_allium"),
              "opti_oignon.cli.garden": _stand_in("opti_oignon.cli.garden")}
    loaded, restore = isolate(targets={"opti_oignon.plugin_loader": source("plugin_loader.py")}, seeded=seeded,
                              packages=("opti_oignon.api", "opti_oignon.cli"),
                              blocked=("opti_oignon.config", "opti_oignon.plugin_manifest"))
    try:
        loader = loaded["opti_oignon.plugin_loader"]
        violation = loader.PluginSandboxViolation
        with loader._RestrictedBuiltins():
            from_plugin = plugin["attempt"](violation)
            from_platform = platform["attempt"](violation)
        assert from_plugin == ["refused", "refused", "refused", "refused", "passed"], from_plugin
        assert from_platform == ["passed"] * 5, from_platform
        assert all(name in sys.modules for name in seeded), "nothing is hidden from the module cache"
        # Through a platform helper the plugin calls: a statement, importlib by name, importlib's own
        # ``__import__``; and a relative import whose package the plugin forges, made by itself or by the helper.
        with loader._RestrictedBuiltins():
            helped = [tricks["through"](helper[name], violation)
                      for name in ("by_statement", "by_name", "by_importlib_import")]
            forged = tricks["forged"](violation) + tricks["forged"](violation, helper["relative"])
            unhelped = [helper[name]() is not None for name in ("by_name", "by_importlib_import")]
        assert helped == ["refused"] * 3, helped
        assert forged == ["refused"] * 4, forged
        # A plugin module is top-level: a relative import of its own names a package it forged, whatever it
        # names; and a package that answers differently when read twice is read once, by the rule that judges:
        # the import goes where the rule looked (a package that does not exist), never to the being.
        with loader._RestrictedBuiltins():
            own = tricks["own_relative"](violation)
            shifted = tricks["shifting"](violation, helper["relative"])
        assert (own, shifted) == ("refused", "import_error"), (own, shifted)
        assert unhelped == [True, True], "witness: the helpers work for a platform caller"
    finally:
        restore()

    # A plugin that renames itself, loaded by the loader: its frames are still a plugin's.
    folder = tmp_path / "renamer"
    folder.mkdir()
    (folder / "manifest.yaml").write_text("name: renamer\nversion: 1.0.0\nauthor: t\ndescription: d\n"
                                          "entry_point: plugin.py\n", encoding="ascii")
    (folder / "plugin.py").write_text(RENAMED_PLUGIN, encoding="ascii")
    seeded = {"opti_oignon.allium": _stand_in("opti_oignon.allium", package=True),
              "opti_oignon.plugin_manifest": _manifest_stand_in()}
    loaded, restore = isolate(targets={"opti_oignon.plugin_loader": source("plugin_loader.py")}, seeded=seeded,
                              blocked=("opti_oignon.config",))
    try:
        loader = loaded["opti_oignon.plugin_loader"]
        renamed = loader.PluginLoader(subprocess_mode="inprocess")._load_plugin_inprocess(folder)
        assert renamed.module.OUTCOME == "PluginSandboxViolation", renamed.module.OUTCOME
        assert renamed.module.__name__ == "json", "the plugin did rename itself"
        assert not loader._LOADING, "no plugin is held as loading once its load ends"
    finally:
        restore()
