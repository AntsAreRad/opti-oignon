#!/usr/bin/env python3
"""Contracts for a failed import, which never opens a door (block 230; W5, D13, M8 and W8 of the security census).

Five reads of the security mode took a mode module that did not import as Daily: the database encryption was not
enforced, the rate limits stayed Daily's, the conversation was not wiped after the turn, the sandbox's strict
settings were not pinned; search could be re-enabled and the cache served its shared bucket the same way. A
``db_encryption`` that did not import opened every store in plaintext in Daily, and a mode that could not be read
there counted as Daily. Twenty-one routers dropped their authentication when the auth routes did not import, twelve
of them with an identity of no one, one with a synthetic admin. The security headers and the content security policy
were skipped, logged, when they could not be registered.

The rule: a module that is not in the tree is a feature the tree does not have; a module that is in the tree and does
not import (a circular import, a dependency of its own missing) is a failure, read closed. Python tells the two apart:
the first raises ``ModuleNotFoundError`` named after that module, the second another ``ImportError`` or a
``ModuleNotFoundError`` named after the missing dependency. The authentication is never a feature to go without:
any failure to import it refuses.

  * FI1 -- A MODE MODULE THAT DOES NOT IMPORT READS AS BULBE in the rate limiter, the per-turn wipe and the sandbox,
    whether it fails in place or for a dependency of its own; a tree with no mode module reads Daily as before. What
    each world raises is witnessed.
  * FI2 -- SEARCH STAYS OFF: the re-enable is refused as in Bulbe, asked and confirmed, in both failing worlds; the
    request goes through in a tree with no mode module.
  * FI3 -- THE CACHE KEEPS TO THE CONVERSATION AND THE SWAP CHECK REFUSES: the semantic cache reads Bulbe in both
    failing worlds and when the mode read raises; the startup swap check refuses unencrypted swap there.
  * FI4 -- A STORE NEVER OPENS IN PLAINTEXT FOR A FAILED IMPORT: ``get_encrypted_connection`` without a key refuses in
    both failing worlds, nothing created; ``safe_connect`` with a ``db_encryption`` that does not import refuses in
    every mode, and with none in the tree refuses in Bulbe, in both failing worlds and when the mode read raises,
    nothing created; a tree with neither module opens plaintext as before.
  * FI5 -- NO ROUTE IS SERVED WITHOUT THE AUTHENTICATION IT ASKS: every handler of an import of the auth routes under
    ``opti_oignon/api`` defines only functions that refuse with 503 and binds only dependencies on them, no name
    bound to an empty list anywhere; three routers loaded with the auth routes absent refuse through their dependency.
  * FI6 -- THE APP NEVER SERVES WITHOUT ITS HEADERS: every middleware of ``api/app.py`` is registered outside any
    ``try``, the security headers and the content security policy among them, their imports too.
  * FI7 -- in place of ft17 and ft19 of the facts suite, which called the memory routes in a window without the auth
    routes and were served with no authentication: both whole, their window given the auth routes with the identity
    they ran under (the old fallback's, no account); without it the first route answers 503.
  * FI8 -- THE TEXTS: the encryption page says what a module that does not import does, and where a tree with no
    mode module opens as Daily; the changelog says the change, and no per-turn wipe, which nothing calls.
  * FI10 -- in place of tl41, tl70 and tl71 of the turn label suite, whose terminal window seeded an empty
    ``db_encryption`` to mean "no encryption layer": the three whole, the layer absent from their window; with the
    empty module the memory store does not import.
  * FI11 -- A LAYER THE COVERED TREE LACKS READS AS ABSENT: under a stand-in package with no path, as four suites'
    loaders build it, importing ``db_utils`` or ``db_encryption`` raises ``ModuleNotFoundError`` named after it,
    whether the data firewall refuses the copy an install finds elsewhere or no finder finds one.
  * FI9 -- NO STORE KEEPS A FALLBACK OF ITS OWN FOR A LAYER THAT FAILS: every guarded import of ``db_utils`` or
    ``db_encryption`` under ``opti_oignon`` (their own modules and the plugins, which keep their stores in plaintext
    in their own process, aside) re-raises first unless the layer is absent from the tree, or the module is loaded
    outside the package; a module the write census seals waits for its debt (rag_dashboard, may only shrink); a
    queue, the notes store and the pending writes, loaded with the layer failing in place or half written, do not
    import, and load as before with it absent.

The modules run in isolation windows (``tests/_isolation.py``) on files under a temporary folder; the app is read as
source, never imported. Nothing reads or writes the platform's data.

Local-only (the public distribution ships no tests).
"""

import ast
import asyncio
import sqlite3
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_fi1_a_mode_module_that_does_not_import_reads_as_bulbe": 6.0,
    "test_fi2_search_stays_off": 4.0,
    "test_fi3_the_cache_keeps_to_the_conversation_and_the_swap_check_refuses": 4.0,
    "test_fi4_a_store_never_opens_in_plaintext_for_a_failed_import": 6.0,
    "test_fi5_no_route_is_served_without_the_authentication_it_asks": 8.0,
    "test_fi6_the_app_never_serves_without_its_headers": 2.0,
    "test_fi7_the_memory_routes_given_their_authentication": 12.0,
    "test_fi8_the_texts": 2.0,
    "test_fi9_no_store_keeps_a_fallback_of_its_own_for_a_layer_that_fails": 6.0,
    "test_fi10_the_terminal_session_with_the_layer_absent": 6.0,
    "test_fi11_a_layer_the_covered_tree_lacks_reads_as_absent": 4.0,
}

_SM = "opti_oignon.security_mode"
_DBE = "opti_oignon.db_encryption"
_API = source("api")


def _failing(name):
    """A module that is there and does not import, as a circular import meets it: ``from <name> import x`` raises an
    ImportError named after the module, not a ModuleNotFoundError."""
    return types.ModuleType(name)


def _dependency_missing(name, dependency="yaml"):
    """A module whose own import lacks a dependency: ``from <name> import x`` raises a ModuleNotFoundError named after
    the dependency."""
    module = types.ModuleType(name)

    def __getattr__(attr):
        raise ModuleNotFoundError(f"No module named '{dependency}'", name=dependency)

    module.__getattr__ = __getattr__
    return module


def _raising(name, error):
    """A module whose own import raises ``error`` (a file half written: a SyntaxError)."""
    module = types.ModuleType(name)

    def __getattr__(attr):
        raise error

    module.__getattr__ = __getattr__
    return module


def _mode(bulbe):
    module = types.ModuleType(_SM)
    if isinstance(bulbe, BaseException):
        def is_bulbe():
            raise bulbe
    else:
        def is_bulbe():
            return bulbe
    module.is_bulbe = is_bulbe
    module._audit_log = lambda *a, **k: None
    return module


# (world, the mode module seeded or None, Bulbe by the rule, what ``from opti_oignon.security_mode import is_bulbe``
# raises there)
_WORLDS = (
    ("absent", None, False, ("ModuleNotFoundError", _SM)),
    ("fails in place", lambda: _failing(_SM), True, ("ImportError", _SM)),
    ("a dependency missing", lambda: _dependency_missing(_SM), True, ("ModuleNotFoundError", "yaml")),
)


# The mode module there and answering: the readings the rule leaves as they were.
_ANSWERING = (
    ("answers Bulbe", lambda: _mode(True), True, None),
    ("answers Daily", lambda: _mode(False), False, None),
)


def _window(targets, seeded=None, packages=()):
    row = sqlite3.Row
    loaded, restore = isolate(targets=targets, seeded=seeded or {}, packages=packages)

    def close():
        try:
            restore()
        finally:
            sqlite3.Row = row

    return loaded, close


def _seen():
    """What importing the mode reader raises in the window open now."""
    try:
        from opti_oignon.security_mode import is_bulbe  # noqa: F401
    except ImportError as exc:
        return type(exc).__name__, exc.name
    return None


def _seeded(make):
    return {} if make is None else {_SM: make()}


def test_fi1_a_mode_module_that_does_not_import_reads_as_bulbe(tmp_path):
    from test_bulbe_pins_the_sandbox_contracts import _manager
    from test_bulbe_pins_the_sandbox_contracts import _window as _sandbox_window

    class _Seed:
        def __init__(self, make):
            self.make = make

        def module(self):
            return self.make()

    for world, make, bulbe, raised in (*_WORLDS, *_ANSWERING):
        loaded, close = _window({"opti_oignon.rate_limiter": source("rate_limiter.py"),
                                 "opti_oignon.conversation_wipe": source("conversation_wipe.py")}, _seeded(make))
        try:
            if raised is not None:
                assert _seen() == raised, (world, _seen())
            limiter = loaded["opti_oignon.rate_limiter"].RateLimiter()
            assert limiter._is_bulbe() is bulbe, (world, "the rate limiter")
            wiper = loaded["opti_oignon.conversation_wipe"].ConversationWipeManager()
            wiper._config["bulbe_wipe_per_turn"] = True
            wiped = wiper.on_bulbe_response("c1")
            assert (wiped is not None) is bulbe, (world, "the per-turn wipe", wiped)
        finally:
            close()
        sm, close = _sandbox_window(tmp_path / world.replace(" ", "_"), None if make is None else _Seed(make))
        try:
            strict = _manager(sm, tmp_path / world.replace(" ", "_"), bwrap=False).strict_mode
        finally:
            close()
        assert strict is bulbe, (world, "the sandbox's strict settings")


def test_fi2_search_stays_off(tmp_path):
    ks = "opti_oignon.search_killswitch"
    raising = ("the read raises", lambda: _mode(RuntimeError("the mode cannot be read")), True, None)
    for world, make, bulbe, raised in (*_WORLDS, *_ANSWERING, raising):
        # asked: the mode module in this world from the start.
        loaded, close = _window({ks: source("search_killswitch.py")}, _seeded(make))
        try:
            if raised is not None:
                assert _seen() == raised, (world, _seen())
            switch = loaded[ks].SearchKillSwitch(state_path=tmp_path / f"{world}-asked.json")
            switch.kill("owner", "contract")
            asked = switch.request_reenable("owner")
        finally:
            close()
        if bulbe:
            assert asked.get("error") == "bulbe_mode" and not switch._state.reenable_pending, (world, asked)
        else:
            assert asked.get("pending") is True, (world, asked)
        # confirmed: asked in Daily, the mode module failing by the confirmation.
        loaded, close = _window({ks: source("search_killswitch.py")}, {_SM: _mode(False)})
        try:
            switch = loaded[ks].SearchKillSwitch(state_path=tmp_path / f"{world}-confirmed.json")
            switch.kill("owner", "contract")
            asked = switch.request_reenable("owner")
            assert asked.get("pending") is True, (world, asked)
            code = switch.get_reenable_visual_code()
            if make is None:
                sys.modules[_SM] = None
            else:
                sys.modules[_SM] = make()
            loaded[ks].time = types.SimpleNamespace(time=lambda: asked["expires_at"] - 30)
            confirmed = switch.confirm_reenable(request_id=asked["request_id"], visual_code=code, user_id="owner")
            killed, pending = switch.is_killed(), switch._state.reenable_pending
        finally:
            close()
        if bulbe:
            assert confirmed.get("error") == "bulbe_mode" and killed and not pending, (world, confirmed, pending)
        else:
            assert confirmed.get("success") is True and not killed, (world, confirmed)


def test_fi3_the_cache_keeps_to_the_conversation_and_the_swap_check_refuses(tmp_path):
    config = types.ModuleType("opti_oignon.config")
    config.DATA_DIR = tmp_path / "data"
    worlds = [*_WORLDS, *_ANSWERING,
              ("the read raises", lambda: _mode(RuntimeError("the mode cannot be read")), True, None)]
    for world, make, bulbe, raised in worlds:
        seeded = {"opti_oignon.config": config, **_seeded(make)}
        loaded, close = _window({"opti_oignon.semantic_cache": source("semantic_cache.py"),
                                 "opti_oignon.secure_bytes": source("secure_bytes.py")}, seeded)
        try:
            if raised is not None:
                assert _seen() == raised, (world, _seen())
            assert loaded["opti_oignon.semantic_cache"]._is_bulbe() is bulbe, (world, "the semantic cache")
            if world == "the read raises":
                continue
            sb = loaded["opti_oignon.secure_bytes"]
            sb.check_swap_encrypted = lambda: sb.SwapCheckResult(
                swap_enabled=True, encrypted=False, devices=[{"device": "/dev/sda2", "encrypted": False}], safe=False,
                platform_supported=True)
            try:
                sb.swap_startup_check()
                refused = None
            except RuntimeError as exc:
                refused = str(exc)
        finally:
            close()
        if bulbe:
            assert refused is not None and "Bulbe mode requires encrypted swap" in refused, (world, refused)
        else:
            assert refused is None, (world, refused)


def test_fi4_a_store_never_opens_in_plaintext_for_a_failed_import(tmp_path):
    # get_encrypted_connection, no key.
    for world, make, bulbe, raised in _WORLDS:
        loaded, close = _window({_DBE: source("db_encryption.py")}, _seeded(make))
        path = tmp_path / f"dbe-{world}.db"
        try:
            dbe = loaded[_DBE]
            dbe._get_db_encryption_key = lambda: None
            try:
                dbe.get_encrypted_connection(str(path)).close()
                refused = None
            except RuntimeError as exc:
                refused = str(exc)
        finally:
            close()
        if bulbe:
            assert refused is not None and not path.exists(), (world, refused)
        else:
            assert refused is None and path.exists(), (world, refused)
    # safe_connect, db_encryption absent from the tree, failing in place, or lacking a dependency of its own.
    du = "opti_oignon.db_utils"
    encryption = (
        ("absent", None, None),
        ("fails in place", lambda: _failing(_DBE), "ImportError"),
        ("a dependency missing", lambda: _dependency_missing(_DBE, "sqlcipher3"), "ModuleNotFoundError"),
        ("half written", lambda: _raising(_DBE, SyntaxError("invalid syntax")), "SyntaxError"),
        ("the package missing", lambda: _dependency_missing(_DBE, "opti_oignon"), None),
    )
    modes = [*_WORLDS, ("Daily", lambda: _mode(False), False, None), ("Bulbe", lambda: _mode(True), True, None),
             ("the read raises", lambda: _mode(RuntimeError("the mode cannot be read")), True, None)]
    for kind, made, broken in encryption:
        for world, make, bulbe, _raised in modes:
            seeded = {**_seeded(make), **({} if made is None else {_DBE: made()})}
            loaded, close = _window({du: source("db_utils.py")}, seeded)
            path = tmp_path / f"du-{kind}-{world}.db"
            try:
                assert loaded[du]._ENCRYPTION_AVAILABLE is False, (kind, world)
                refused = {}
                for read_only in (False, True):
                    try:
                        loaded[du].safe_connect(path, read_only=read_only).close()
                        refused[read_only] = None
                    except RuntimeError as exc:
                        refused[read_only] = str(exc)
            finally:
                close()
            for said in refused.values():
                if broken is not None:
                    assert said is not None and f"({broken}: " in said and not path.exists(), (kind, world, refused)
                elif bulbe:
                    assert said is not None and "Bulbe mode requires encryption" in said and not path.exists(), (
                        kind, world, refused)
                else:
                    assert said is None and path.exists(), (kind, world, refused)


def _auth_sites():
    """Every try under opti_oignon/api that imports the router's authentication from the auth routes, with its
    module's name (a socket's handshake check, which imports another name, refuses on its own)."""
    for path in sorted(_API.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and any(
                    isinstance(s, ast.ImportFrom) and s.level == 1 and s.module == "routes_auth"
                    and "_get_current_user" in [a.name for a in s.names] for s in node.body):
                yield path.name, node


def _refuses(function):
    """A function whose body, a docstring aside, is one raise of HTTPException(503)."""
    body = [s for s in function.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or not isinstance(body[0].exc, ast.Call):
        return False
    call = body[0].exc
    status = [k.value for k in call.keywords if k.arg == "status_code"] or call.args[:1]
    return (isinstance(call.func, ast.Name) and call.func.id == "HTTPException" and len(status) == 1
            and isinstance(status[0], ast.Constant) and status[0].value == 503)


def test_fi5_no_route_is_served_without_the_authentication_it_asks(tmp_path):
    seen, bound = set(), {}
    for name, node in _auth_sites():
        for handler in node.handlers:
            functions = {s.name for s in handler.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
            bound.setdefault(name, set()).update(
                t.id for s in handler.body if isinstance(s, ast.Assign) for t in s.targets if isinstance(t, ast.Name))
            for stmt in handler.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert _refuses(stmt), (name, handler.lineno, stmt.name, "a function that does not refuse")
                elif isinstance(stmt, ast.Assign):
                    value = stmt.value
                    assert isinstance(value, ast.List) and value.elts, (name, handler.lineno, ast.unparse(stmt))
                    for elt in value.elts:
                        assert (isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name)
                                and elt.func.id == "Depends" and len(elt.args) == 1
                                and isinstance(elt.args[0], ast.Name) and elt.args[0].id in functions), (
                            name, handler.lineno, ast.unparse(stmt))
                else:
                    raise AssertionError((name, handler.lineno, ast.unparse(stmt)))
            assert functions, (name, handler.lineno, "no refusal defined")
        seen.add(name)
    expected = {f"routes_{m}.py" for m in (
        "agent_eval", "answer_verification", "backup", "citation_verification", "claim_verification",
        "conversations", "governor", "memory", "note_actions", "note_updates", "notes", "notes_attachments",
        "notes_caption", "notes_transcription", "pending_writes", "plugin_marketplace", "plugins", "projects", "rag",
        "sandbox", "security", "sync")}
    assert expected <= seen, sorted(expected - seen)
    for path in sorted(_API.glob("*.py")):
        used = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and not node.value.elts:
                for target in node.targets:
                    assert not (isinstance(target, ast.Name) and "auth" in target.id), (path.name, node.lineno)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "APIRouter":
                for keyword in node.keywords:
                    if keyword.arg == "dependencies":
                        assert not (isinstance(keyword.value, ast.List) and not keyword.value.elts), (
                            path.name, node.lineno, "a router given no dependency")
                        if isinstance(keyword.value, ast.Name):
                            used.add(keyword.value.id)
        assert bound.get(path.name, set()) <= used, (path.name, bound.get(path.name), used)
    # Three routers loaded with the auth routes absent from the tree: their dependency refuses.
    for module, name in (("routes_projects", "router"), ("routes_plugins", "router"),
                         ("routes_pending_writes", "pending_writes_router")):
        dotted = f"opti_oignon.api.{module}"
        loaded, close = _window({dotted: source("api", f"{module}.py")}, packages=("opti_oignon.api",))
        try:
            try:
                from opti_oignon.api.routes_auth import _get_current_user  # noqa: F401
                raise AssertionError("the auth routes must be absent from this window")
            except ModuleNotFoundError as exc:
                assert exc.name == "opti_oignon.api.routes_auth", exc.name
            router = getattr(loaded[dotted], name)
            assert router.dependencies, module
            for dep in router.dependencies:
                try:
                    asyncio.run(dep.dependency())
                    status = None
                except HTTPException as exc:
                    status = exc.status_code
                assert status == 503, (module, status)
        finally:
            close()


def test_fi6_the_app_never_serves_without_its_headers():
    # A registration is a statement of the module itself: no try, with, if or function around it.
    tree = ast.parse(source("api", "app.py").read_text(encoding="utf-8"))
    top = {id(stmt) for stmt in tree.body} | {id(stmt.value) for stmt in tree.body if isinstance(stmt, ast.Expr)}
    registered = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_middleware":
            assert id(node) in top, (node.lineno, ast.unparse(node))
            registered.append(ast.unparse(node.args[0]))
    assert {"SecurityHeadersMiddleware", "_CSPMiddleware"} <= set(registered), registered
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and {a.name for a in node.names} & {"SecurityHeadersMiddleware", "CSPMiddleware"}]
    assert len(imports) == 2 and all(id(node) in top for node in imports), [(n.lineno, n.module) for n in imports]
    # Nor is the stack they build touched afterwards: no read or write of it in the module.
    touched = [(node.lineno, node.attr) for node in ast.walk(tree)
               if isinstance(node, ast.Attribute) and node.attr in ("user_middleware", "middleware_stack")]
    assert not touched, touched


def test_fi7_the_memory_routes_given_their_authentication(tmp_path, monkeypatch):
    import test_facts_forgotten_contracts as ft

    auth = types.ModuleType("opti_oignon.api.routes_auth")
    auth._get_current_user = lambda: {"sub": None}
    real = ft.isolate

    def given(*, targets, seeded=None, **kwargs):
        seeded = dict(seeded or {})
        if "opti_oignon.api.routes_memory" in targets:
            seeded["opti_oignon.api.routes_auth"] = auth
        return real(targets=targets, seeded=seeded, **kwargs)

    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(AssertionError, match="503"):
        ft.test_ft17_a_fact_the_user_made_theirs_stays_when_its_conversation_goes(bare)
    monkeypatch.setattr(ft, "isolate", given)
    for test in (ft.test_ft17_a_fact_the_user_made_theirs_stays_when_its_conversation_goes,
                 ft.test_ft19_a_restore_racing_the_forgetting_leaves_no_vector_and_the_trash_keeps_its_clock):
        directory = tmp_path / test.__name__[5:9]
        directory.mkdir()
        test(directory)


def test_fi8_the_texts():
    def flat(path):
        return " ".join(path.read_text(encoding="utf-8").split())

    page = flat(REPO / "docs" / "security" / "encryption.md")
    assert ("Bulbe mode refuses to open a database in plaintext through the platform's connection layer "
            "(`safe_connect`); a store that opens its own connection when that layer is missing from the install "
            "altogether does not refuse yet.") in page
    assert ("An encryption module that is there and does not import (a circular import, a dependency of its own "
            "missing, a file half written) refuses every database, in either mode, and the server does not start; "
            "a module that opens a database does not import when its database layer is there and does not, but the "
            "knowledge base dashboard (`rag_dashboard.py`), which keeps its own plaintext path for now. A security "
            "mode that cannot be read, or whose module is there and does not import, counts as Bulbe here. Only a "
            "tree with no security mode module at all opens as Daily here (the HTTP gate reads it as Bulbe).") in page
    log = flat(REPO / "CHANGELOG.md")
    assert ("- [SECURITY] A failed import of the security mode, the database layer or the auth routes no longer "
            "opens a door, and the app no longer starts without its security headers.") in log
    assert "thirty-seven modules that open a database also kept a plaintext fallback of their own" in log
    assert "refuses: the server does not start, `safe_connect` refuses every database" in log
    assert "a conversation was not wiped after its turn" not in log
    assert "Now they refuse every request (503), as the security routes did." in log
    sandbox, cache = flat(source("sandbox_manager.py")), flat(source("semantic_cache.py"))
    assert "in a tree with no security-mode module there is no Bulbe to pin for." in sandbox
    assert "Fail-safe to False when the mode is undeterminable" not in cache
    assert "A mode that cannot be read reads as Bulbe, and so does a mode module that is there and does not import" in (
        cache)


def _layer_sites():
    """Every try under opti_oignon/ whose body imports db_utils or db_encryption, outside those two and plugins/; the
    package walked as the census walks it, never into a data directory."""
    from _allium_census import ROOT, modules

    for path in sorted(modules().values()):
        rel = path.relative_to(ROOT)
        if rel.name in ("db_utils.py", "db_encryption.py") or rel.parts[0] == "plugins":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Try):
                hits = [s for s in node.body if isinstance(s, ast.ImportFrom) and s.module
                        and s.module.split(".")[-1] in ("db_utils", "db_encryption")]
                if hits:
                    yield str(rel), node, hits[0]


def _sealed():
    """The modules the write census seals until their debt is paid (its LEDGER, read as source): a byte of theirs
    that moves fails that guard, so the split waits for the debt."""
    tree = ast.parse((REPO / ".github" / "scripts" / "write_census_guard.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["LEDGER"]:
            return {Path(key.value).relative_to("opti_oignon").as_posix() for key in node.value.keys}
    raise AssertionError("the write census keeps no LEDGER")


def test_fi9_no_store_keeps_a_fallback_of_its_own_for_a_layer_that_fails(tmp_path):
    sites, owed, sealed = 0, set(), _sealed()
    for rel, node, imp in _layer_sites():
        dotted = "opti_oignon." + imp.module.split(".")[-1]
        handler = node.handlers[0]
        first = handler.body[0]
        test = ast.unparse(first.test) if isinstance(first, ast.If) else ""
        if imp.level:
            want = (f'(__package__ or "").split(".")[0] == "opti_oignon" and not ('
                    f'isinstance(_exc, ModuleNotFoundError) and _exc.name == "{dotted}")')
        else:
            want = f'not (isinstance(_exc, ModuleNotFoundError) and _exc.name in ("opti_oignon", "{dotted}"))'
        # The guard word for word: a widened type, another segment, an "or True" is another rule.
        split = (len(node.handlers) == 1 and handler.name == "_exc" and isinstance(first, ast.If)
                 and len(first.body) == 1 and isinstance(first.body[0], ast.Raise) and first.body[0].exc is None
                 and not first.orelse and test == ast.unparse(ast.parse(want, mode="eval").body))
        if not split and rel in sealed:
            owed.add(rel)
            continue
        assert split, (rel, handler.lineno, test)
        sites += 1
    assert sites >= 41, sites
    # Waits for its write debt (BACKLOG 230 f); may only shrink.
    assert owed <= {"rag_dashboard.py"}, owed
    # Loaded: the layer failing in place, or half written, the module does not import; absent, it loads as before.
    du = "opti_oignon.db_utils"
    cases = (
        ("opti_oignon.sync_queue", ("sync_queue.py",), du, ()),
        ("opti_oignon.notes.notes_store", ("notes", "notes_store.py"), du, ("opti_oignon.notes",)),
        ("opti_oignon.pending_writes", ("pending_writes.py",), _DBE, ()),
    )
    for module, parts, layer, packages in cases:
        for kind, made, error in (("fails in place", lambda: _failing(layer), ImportError),
                                  ("half written", lambda: _raising(layer, SyntaxError("invalid syntax")),
                                   SyntaxError)):
            row = sqlite3.Row
            try:
                with pytest.raises(error):
                    isolate(targets={module: source(*parts)}, seeded={layer: made()}, packages=packages)
            finally:
                sqlite3.Row = row
        loaded, close = _window({module: source(*parts)}, packages=packages)
        try:
            assert loaded[module] is not None, (module, "absent")
        finally:
            close()


def test_fi10_the_terminal_session_with_the_layer_absent(monkeypatch):
    import test_turn_label_memory_contracts as tl

    real = tl.isolate

    def absent(*, targets, seeded=None, **kwargs):
        seeded = {name: module for name, module in (seeded or {}).items()
                  if not (name == _DBE and not hasattr(module, "get_encrypted_connection"))}
        return real(targets=targets, seeded=seeded, **kwargs)

    with pytest.raises(ImportError, match="SQLCIPHER_AVAILABLE"):
        tl.test_tl41_the_terminal_adopts_a_fact_by_the_digest_shown_and_withdraws_a_source()
    monkeypatch.setattr(tl, "isolate", absent)
    tl.test_tl41_the_terminal_adopts_a_fact_by_the_digest_shown_and_withdraws_a_source()
    tl.test_tl70_the_terminal_close_hands_the_conversation_as_the_store_reads_it_and_closes_without_it()
    tl.test_tl71_a_withdrawal_by_text_that_lowered_nothing_says_how_to_name_the_exact_bytes()


def test_fi11_a_layer_the_covered_tree_lacks_reads_as_absent():
    import importlib

    names = ("opti_oignon", "opti_oignon.db_utils", "opti_oignon.db_encryption")
    saved = {name: sys.modules.get(name) for name in names}
    stand_in = types.ModuleType("opti_oignon")
    stand_in.__path__ = []
    try:
        sys.modules["opti_oignon"] = stand_in
        for name in names[1:]:
            sys.modules.pop(name, None)
            with pytest.raises(ModuleNotFoundError) as caught:
                importlib.import_module(name)
            assert caught.value.name == name, (name, caught.value.name, str(caught.value))
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
