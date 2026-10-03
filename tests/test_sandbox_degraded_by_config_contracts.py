#!/usr/bin/env python3
"""Degraded-mode contracts: the server decides, never the caller.

A sandbox on the tempdir backend gives no real isolation. Whether one may be
created is decided by the server configuration
(``require_degraded_confirmation``) and by the user's own confirmation
(``POST /api/sandbox/confirm-degraded``), never by a flag a caller passes:

  * DG1 -- unconfirmed, the manager refuses a degraded sandbox whatever the
    caller asks; once the user confirms, it creates one;
  * DG2 -- no production code passes a degraded flag, and no function
    accepts one;
  * DG3 -- ``POST /api/sandbox/create`` refuses ``allow_degraded: true`` with
    a 400 that names the confirmation route, and asks the manager nothing;
  * DG4 -- ``POST /api/coding/start`` refuses it the same way and starts no
    task.

The manager module is loaded from a copy of its file next to a configuration
that switches the import-time singleton off, so no test reaches the
configured workspace base. Local-only.
"""

import ast
import os
import sqlite3
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_PACKAGE = Path(__file__).resolve().parent.parent / "opti_oignon"


def _manager_module(tmp_path):
    """sandbox_manager.py loaded from a byte-exact copy whose configuration
    disables the singleton (it would reconcile the configured workspace base)."""
    package = tmp_path / "copy"
    (package / "config").mkdir(parents=True)
    (package / "config" / "sandbox.yaml").write_text("enabled: false\n", encoding="utf-8")
    copy = package / "sandbox_manager.py"
    copy.write_bytes(source("sandbox_manager.py").read_bytes())
    db_utils = types.ModuleType("opti_oignon.db_utils")
    db_utils.safe_connect = lambda path, *a, **k: sqlite3.connect(path)
    loaded, restore = isolate(
        targets={"opti_oignon.sandbox_manager": copy},
        seeded={"opti_oignon.db_utils": db_utils},
    )
    module = loaded["opti_oignon.sandbox_manager"]
    if module.sandbox_manager is not None:
        restore()
        raise AssertionError("the import-time singleton must stay off in this window")
    return module, restore


def test_dg1_degraded_mode_opens_only_by_the_users_confirmation(tmp_path):
    sm, restore = _manager_module(tmp_path)
    try:
        sm._detect_bwrap = lambda: (False, "stand-in: no bwrap")
        manager = sm.SandboxManager(sm.SandboxConfig(
            isolation_backend="auto",
            strict_mode=False,
            require_degraded_confirmation=True,
            workspace_base=str(tmp_path / "workspaces"),
            reconcile_on_start=False,
        ))
        assert manager.degraded_mode, "the window must stand for a degraded sandbox"
        outcomes = []
        for asked in ({}, {"allow_degraded": True}):
            try:
                manager.create_sandbox(f"dg1-{len(outcomes)}", **asked)
                outcomes.append("created")
            except (RuntimeError, TypeError) as exc:
                outcomes.append(type(exc).__name__)
        before = manager.active_session_count
        manager.confirm_degraded_mode()
        manager.create_sandbox("dg1-confirmed")
        after = manager.active_session_count
    finally:
        restore()
    assert "created" not in outcomes, f"a caller opened degraded mode: {outcomes}"
    assert (before, after) == (0, 1), (before, after)


def _production_sources():
    """Every Python file of the package, in a stable order. The package's data
    directory is never entered: it holds the user's data, not code."""
    found = []
    for root, dirs, files in os.walk(_PACKAGE):
        if Path(root) == _PACKAGE:
            dirs[:] = [d for d in dirs if d != "data"]
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        found.extend(Path(root) / f for f in sorted(files) if f.endswith(".py"))
    return found


def test_dg2_no_production_code_passes_or_accepts_a_degraded_flag():
    passes, accepts = [], []
    for path in _production_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_PACKAGE.parent).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(k.arg == "allow_degraded" for k in node.keywords):
                passes.append(f"{rel}:{node.lineno}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                if any(p.arg == "allow_degraded" for p in params):
                    accepts.append(f"{rel}:{node.lineno}")
    assert (passes, accepts) == ([], []), (passes, accepts)


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _route(filename, deps):
    """A REST route module loaded from its file, with the real schemas and a
    stand-in ``deps``."""
    name = "opti_oignon.api." + filename[:-3]
    loaded, restore = isolate(
        targets={
            "opti_oignon.api.schemas": source("api", "schemas.py"),
            name: source("api", filename),
        },
        seeded={"opti_oignon.api.deps": _module("opti_oignon.api.deps", **deps)},
        blocked=("ollama",),
        packages=("opti_oignon.api",),
    )
    return loaded[name], loaded["opti_oignon.api.schemas"], restore


def _refusal(call):
    """(status, names the confirmation route) of the refusal ``call`` raises."""
    try:
        call()
    except Exception as exc:  # the framework's HTTPException
        return getattr(exc, "status_code", None), "confirm-degraded" in str(getattr(exc, "detail", ""))
    return None


def test_dg3_the_sandbox_create_route_refuses_a_requested_degraded_mode():
    asked = []
    session = types.SimpleNamespace(
        session_id="dg3", workspace_path="/dg3",
        isolation_backend=types.SimpleNamespace(value="bwrap"), label="",
    )
    manager = types.SimpleNamespace(
        create_sandbox=lambda **kwargs: asked.append(kwargs) or session, degraded_mode=False,
    )
    deps = {"FILE_TOOLS_AVAILABLE": True, "SANDBOX_AVAILABLE": True, "sandbox_manager": manager}
    route, schemas, restore = _route("routes_sandbox.py", deps)
    try:
        route.SANDBOX_AVAILABLE, route.sandbox_manager, route._emergency_stop = True, manager, None
        refused = _refusal(lambda: route.create_sandbox(
            schemas.SandboxCreateRequest(allow_degraded=True), current_user={"sub": "local"}))
        asked_while_refused = list(asked)
        route.create_sandbox(schemas.SandboxCreateRequest(), current_user={"sub": "local"})
    finally:
        restore()
    assert refused == (400, True), refused
    assert asked_while_refused == [], "the manager was asked despite the refusal"
    assert len(asked) == 1 and "allow_degraded" not in asked[0], asked


def test_dg4_the_coding_start_route_refuses_a_requested_degraded_mode():
    started = []
    agent = types.SimpleNamespace(start_task=lambda **kwargs: started.append(kwargs) or "dg4")
    deps = {"CODING_AGENT_AVAILABLE": True, "CODING_HISTORY_AVAILABLE": False,
            "coding_agent_instance": None, "coding_history_store": None}
    route, schemas, restore = _route("routes_coding.py", deps)
    try:
        route._emergency_stop = None
        route._ensure_agent_with_callback = lambda: agent
        route._build_status_response = lambda _agent: {"status": "started"}
        refused = _refusal(lambda: route.start_coding_task(
            schemas.CodingTaskRequest(task="dg4", allow_degraded=True)))
        started_while_refused = list(started)
        route.start_coding_task(schemas.CodingTaskRequest(task="dg4"))
    finally:
        restore()
    assert refused == (400, True), refused
    assert started_while_refused == [], "a task started despite the refusal"
    assert len(started) == 1 and "allow_degraded" not in started[0], started
