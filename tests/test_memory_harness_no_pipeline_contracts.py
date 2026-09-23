#!/usr/bin/env python3
"""Contract that the memory harness is measurement only: nothing on the chat
path imports it.

The memory block's first session builds the instrument and writes no
pipeline. That is a rule about the tree, and a rule about the tree is
checked on the tree: the three harness modules exist, and no other module in
the package imports any of them -- not the executor, not the agent, not the
routes, and not the memory package's own facade, which would pull them into
every import of the package.

  * ZP1 -- the three harness modules exist and are pure at module scope.
  * ZP2 -- no module outside the harness imports a harness module.
  * ZP3 -- the composer, the Core store and the receipts exist, are pure at
    module scope, and are imported by nothing else yet: wiring them into
    the chat path is the block's last session, not a side effect.
  * ZP4 -- supersedes ZP2 once the gate exists: the harness is imported by
    the onion's own modules only (the probes gate eviction, by design), and
    still by nothing on the chat path -- not the executor, not the agent,
    not the routes, not the package facade.
  * ZP5 -- supersedes ZP3 once the executor is wired: the onion modules are
    imported by one another and by the executor's guarded import of the
    librarian, and by nothing else -- not the routes, not the agent, not
    the package facade.
  * ZP6 -- supersedes ZP5 once the user's surface exists: the onion is
    imported by the executor's guarded import and by the memory routes'
    handlers, both of them the librarian and nothing else; the model's
    tools and the chat tool registry reach no onion module.
  * ZP7 -- supersedes ZP6 once the terminal session exists: the chat
    session under ``cli/`` is the third user surface. The same properties
    over three importers, each importing the librarian and nothing else,
    and the two surfaces that are not the executor reach it inside a
    function, never at module scope.

Local-only (the public distribution ships no tests). Reads the tree; loads
nothing.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO  # noqa: E402

_PACKAGE = REPO / "opti_oignon"
_HARNESS = ("probes", "drift", "baseline")
_HARNESS_PATHS = {_PACKAGE / "memory" / f"{m}.py" for m in _HARNESS}
_ONION = ("composer", "core_store", "receipts", "peels", "librarian", "ledger_store", "onion_store")
_ONION_PATHS = {_PACKAGE / "memory" / f"{m}.py" for m in _ONION}
_STDLIB_ONLY_AT_SCOPE = {"dataclasses", "re", "typing", "collections", "hashlib", "json", "math", "itertools", "logging"}


def _imports_of(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
    return names


# ---------------------------------------------------------------------------
# ZP1 -- the harness exists and is pure at module scope
# ---------------------------------------------------------------------------
def test_zp1_the_harness_modules_exist_and_are_pure_at_module_scope():
    for path in sorted(_HARNESS_PATHS):
        assert path.is_file(), f"{path.name} is part of the harness"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = {
            n for n in _imports_of(ast.Module(body=[s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))], type_ignores=[]))
        }
        offenders = {n for n in top if n.split(".")[0] not in _STDLIB_ONLY_AT_SCOPE and not n.startswith("__future__")}
        assert offenders == set(), (
            f"{path.name} imports only the standard library at module scope; "
            f"anything heavier is imported inside the function that needs it: {offenders}"
        )


# ---------------------------------------------------------------------------
# ZP2 -- nothing outside the harness imports it
# ---------------------------------------------------------------------------
def test_zp2_nothing_on_the_chat_path_imports_the_harness():
    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path in _HARNESS_PATHS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _imports_of(tree):
            bare = name.lstrip(".")
            last = bare.split(".")[-1] if bare else ""
            if (
                bare.startswith("opti_oignon.memory.") and last in _HARNESS
            ) or (name.startswith(".") and last in _HARNESS and "memory" in str(path)):
                offenders.append(f"{path.relative_to(REPO)} imports {name}")
    assert offenders == [], (
        "the harness is measurement, not pipeline: nothing imports it yet. "
        f"Found: {offenders}"
    )


# ---------------------------------------------------------------------------
# ZP3 -- the onion modules exist, are pure at scope, and are not wired yet
# ---------------------------------------------------------------------------
def test_zp3_the_onion_modules_exist_and_nothing_imports_them_yet():
    allowed = _STDLIB_ONLY_AT_SCOPE | {"pathlib"}
    for path in sorted(_ONION_PATHS):
        assert path.is_file(), f"{path.name} is part of the onion"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = _imports_of(ast.Module(body=[s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))], type_ignores=[]))
        offenders = {n for n in top if n.split(".")[0] not in allowed and not n.startswith("__future__")}
        assert offenders == set(), f"{path.name} imports only the standard library at module scope: {offenders}"
    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path in _ONION_PATHS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _imports_of(tree):
            bare = name.lstrip(".")
            last = bare.split(".")[-1] if bare else ""
            if (
                bare.startswith("opti_oignon.memory.") and last in _ONION
            ) or (name.startswith(".") and last in _ONION and "memory" in str(path)):
                offenders.append(f"{path.relative_to(REPO)} imports {name}")
    assert offenders == [], f"the onion is not on the chat path yet. Found: {offenders}"


# ---------------------------------------------------------------------------
# ZP4 -- the harness is imported by the onion only
# ---------------------------------------------------------------------------
def test_zp4_the_harness_is_imported_by_the_onion_only_and_never_by_the_chat_path():
    importers = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path in _HARNESS_PATHS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _imports_of(tree):
            bare = name.lstrip(".")
            last = bare.split(".")[-1] if bare else ""
            if (
                bare.startswith("opti_oignon.memory.") and last in _HARNESS
            ) or (name.startswith(".") and last in _HARNESS and "memory" in str(path)):
                importers.setdefault(path, []).append(name)
    assert len(importers) >= 1, "control: the gate imports the probes, so the census reads non-zero"
    outside = {f"{p.relative_to(REPO)} imports {n}" for p, names in importers.items() if p not in _ONION_PATHS for n in names}
    assert outside == set(), f"only the onion may import the harness. Found: {sorted(outside)}"
    assert set(importers) <= _ONION_PATHS
    facade = _PACKAGE / "memory" / "__init__.py"
    assert facade not in importers, "the facade would pull the harness into every import of the package"


# ---------------------------------------------------------------------------
# ZP5 -- the onion is imported by the executor's guarded import only
# ---------------------------------------------------------------------------
def test_zp5_the_onion_is_imported_by_the_executor_only_and_only_the_librarian():
    allowed = _STDLIB_ONLY_AT_SCOPE | {"pathlib", "threading", "logging", "contextlib", "datetime"}
    for path in sorted(_ONION_PATHS):
        assert path.is_file(), f"{path.name} is part of the onion"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = _imports_of(ast.Module(body=[s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))], type_ignores=[]))
        offenders = {n for n in top if n.split(".")[0] not in allowed and not n.startswith("__future__")}
        assert offenders == set(), f"{path.name} imports only the standard library at module scope: {offenders}"
    importers = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path in _ONION_PATHS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _imports_of(tree):
            bare = name.lstrip(".")
            last = bare.split(".")[-1] if bare else ""
            if (
                bare.startswith("opti_oignon.memory.") and last in _ONION
            ) or (
                name.startswith(".") and last in _ONION
                and (bare.startswith("memory.") or "memory" in str(path))
            ):
                importers.setdefault(path.relative_to(REPO).as_posix(), set()).add(last)
    assert importers.get("opti_oignon/executor.py") == {"librarian"}, (
        f"the executor imports the librarian and nothing else of the onion: {importers}"
    )
    assert set(importers) == {"opti_oignon/executor.py"}, f"no other module imports the onion: {importers}"
    facade = (_PACKAGE / "memory" / "__init__.py").read_text(encoding="utf-8")
    assert not any(f".{m} import" in facade or f"memory.{m}" in facade for m in _ONION), "the facade stays out of it"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# ZP6 -- the executor and the memory routes import the librarian, nothing else does
# ---------------------------------------------------------------------------
_USER_SURFACES = {"opti_oignon/executor.py", "opti_oignon/api/routes_memory.py"}
_MODEL_TOOLS = ("agent/tools.py", "tool_registry.py", "tool_executor.py", "agent/loop.py")


def test_zp6_the_executor_and_the_memory_routes_import_the_librarian_and_the_models_tools_reach_no_onion():
    allowed = _STDLIB_ONLY_AT_SCOPE | {"pathlib", "threading", "logging", "contextlib", "datetime"}
    for path in sorted(_ONION_PATHS):
        assert path.is_file(), f"{path.name} is part of the onion"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = _imports_of(ast.Module(body=[s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))], type_ignores=[]))
        offenders = {n for n in top if n.split(".")[0] not in allowed and not n.startswith("__future__")}
        assert offenders == set(), f"{path.name} imports only the standard library at module scope: {offenders}"
    importers = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path in _ONION_PATHS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _imports_of(tree):
            bare = name.lstrip(".")
            last = bare.split(".")[-1] if bare else ""
            if (
                bare.startswith("opti_oignon.memory.") and last in _ONION
            ) or (
                name.startswith(".") and last in _ONION
                and (bare.startswith("memory.") or "memory" in str(path))
            ):
                importers.setdefault(path.relative_to(REPO).as_posix(), set()).add(last)
        # The spelling the census above cannot see: ``from ..memory import
        # librarian`` names the onion module as the imported name, not as
        # the module. The routes import it that way, inside their handlers.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "memory":
                for alias in node.names:
                    if alias.name in _ONION:
                        importers.setdefault(path.relative_to(REPO).as_posix(), set()).add(alias.name)
    assert set(importers) == _USER_SURFACES, f"the two user surfaces import the onion, nothing else: {importers}"
    assert all(v == {"librarian"} for v in importers.values()), f"and each imports the librarian only: {importers}"
    for rel in _MODEL_TOOLS:
        text = (_PACKAGE / rel).read_text(encoding="utf-8")
        assert not any(m in text for m in ("core_store", "receipts", "librarian", "onion_store")), (
            f"{rel} names no onion module: the model's tools have no path to the Core"
        )
    facade = (_PACKAGE / "memory" / "__init__.py").read_text(encoding="utf-8")
    assert not any(f".{m} import" in facade or f"memory.{m}" in facade for m in _ONION), "the facade stays out of it"


# ---------------------------------------------------------------------------
# ZP7 -- the executor, the memory routes and the chat session import the librarian
# ---------------------------------------------------------------------------
_USER_SURFACES_WITH_SESSION = _USER_SURFACES | {"opti_oignon/cli/session.py"}


def _onion_importers():
    """Every module outside the onion that imports an onion module, with the names it imports."""
    importers = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path in _ONION_PATHS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _imports_of(tree):
            bare = name.lstrip(".")
            last = bare.split(".")[-1] if bare else ""
            if (
                bare.startswith("opti_oignon.memory.") and last in _ONION
            ) or (
                name.startswith(".") and last in _ONION
                and (bare.startswith("memory.") or "memory" in str(path))
            ):
                importers.setdefault(path.relative_to(REPO).as_posix(), set()).add(last)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "memory":
                for alias in node.names:
                    if alias.name in _ONION:
                        importers.setdefault(path.relative_to(REPO).as_posix(), set()).add(alias.name)
    return importers


def _module_scope_names_onion(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[-1] in _ONION or any(a.name in _ONION for a in node.names):
                return True
        if isinstance(node, ast.Import) and any(a.name.split(".")[-1] in _ONION for a in node.names):
            return True
    return False


def test_zp7_the_executor_the_memory_routes_and_the_chat_session_import_the_librarian_and_nothing_else():
    allowed = _STDLIB_ONLY_AT_SCOPE | {"pathlib", "threading", "logging", "contextlib", "datetime"}
    for path in sorted(_ONION_PATHS):
        assert path.is_file(), f"{path.name} is part of the onion"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = _imports_of(ast.Module(body=[s for s in tree.body if isinstance(s, (ast.Import, ast.ImportFrom))], type_ignores=[]))
        offenders = {n for n in top if n.split(".")[0] not in allowed and not n.startswith("__future__")}
        assert offenders == set(), f"{path.name} imports only the standard library at module scope: {offenders}"
    importers = _onion_importers()
    assert set(importers) == _USER_SURFACES_WITH_SESSION, f"the three user surfaces import the onion, nothing else: {importers}"
    assert all(v == {"librarian"} for v in importers.values()), f"and each imports the librarian only: {importers}"
    for rel in sorted(_USER_SURFACES_WITH_SESSION - {"opti_oignon/executor.py"}):
        assert not _module_scope_names_onion(REPO / rel), f"{rel} reaches the librarian inside a function, not at module scope"
    for rel in _MODEL_TOOLS:
        text = (_PACKAGE / rel).read_text(encoding="utf-8")
        assert not any(m in text for m in ("core_store", "receipts", "librarian", "onion_store")), (
            f"{rel} names no onion module: the model's tools have no path to the Core"
        )
    facade = (_PACKAGE / "memory" / "__init__.py").read_text(encoding="utf-8")
    assert not any(f".{m} import" in facade or f"memory.{m}" in facade for m in _ONION), "the facade stays out of it"
