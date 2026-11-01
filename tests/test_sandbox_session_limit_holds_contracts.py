"""Contracts for the sandbox's session limit: it holds when creations race.

``create_sandbox`` counted the active sessions against ``max_concurrent_sessions`` under its lock, let the lock go to
make the workspace, and took it again to register the session, checking only that its id was free: creations racing
for the last free places all registered, and the limit -- the one bound on how many workspaces, each with its quota
and its processes, the sandbox's callers may hold -- was exceeded by all but those places. Now the registration counts
again, under the lock that registers. And the notes' caption and transcription routes made a manager of their own for
each request, whose sessions no limit counted; they now share the routes' manager (round 1).

  * NL1 -- RACING CREATIONS NEVER EXCEED THE LIMIT: two creations held inside their workspaces' making until both
    have passed the first count, with a limit of one, and three with a limit of two: exactly the limit registered,
    the others refused with the limit's message, their workspace folders gone and their holds released.
  * NL2 -- WHAT STAYS: a creation made when the limit is reached is refused, before it makes a workspace; one after a
    destroy succeeds; a session no longer active counts against neither count (round 1); an id in use is refused; of
    two creations racing for one id, one registers and the other is refused as before.
  * NL3 -- THE NOTES ROUTES SHARE THE MANAGER (round 1): the caption and the transcription routes resolve the manager
    every sandbox route shares, never a manager of their own, and nothing in the package makes a ``SandboxManager``
    but the shared one.

Local-only (the public distribution ships no tests).
"""
from __future__ import annotations

import ast
import os
import sys
import threading
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_code_execution_in_the_sandbox_contracts as ce_t  # noqa: E402
from _allium_census import modules as package_modules  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_nl1_racing_creations_never_exceed_the_limit": 5.0,
    "test_nl2_what_the_session_limit_keeps": 5.0,
    "test_nl3_the_notes_routes_share_the_manager": 5.0,
}


def _race(manager, ids):
    """Create one session per id at once, each held inside its workspace's making until all have passed the first
    count; what each answered, and the workspaces and holds each made."""
    barrier = threading.Barrier(len(ids), timeout=5.0)
    real = manager._make_workspace
    made = []
    lock = threading.Lock()

    def held(session_id):
        barrier.wait()
        workspace, hold = real(session_id)
        with lock:
            made.append((workspace, hold))
        return workspace, hold

    manager._make_workspace = held
    answers = [None] * len(ids)

    def create(n, session_id):
        try:
            answers[n] = manager.create_sandbox(session_id, label=f"nl-{n}").session_id
        except Exception as exc:  # noqa: BLE001 -- the contract reads it
            answers[n] = exc

    threads = [threading.Thread(target=create, args=(n, sid)) for n, sid in enumerate(ids)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    manager._make_workspace = real
    return answers, made


def _closed(fd):
    try:
        os.fstat(fd)
    except OSError:
        return True
    return False


def test_nl1_racing_creations_never_exceed_the_limit(tmp_path):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        results = {}
        for limit, count in ((1, 2), (2, 3)):
            manager = ce_t._manager(sm, tmp_path / f"limit{limit}", max_concurrent_sessions=limit)
            answers, made = _race(manager, [None] * count)
            registered = [a for a in answers if isinstance(a, str)]
            refused = [a for a in answers if not isinstance(a, str)]
            kept = {manager.get_workspace_path(sid) for sid in registered}
            lost = [(workspace, hold) for workspace, hold in made if workspace not in kept]
            results[limit] = (registered, refused, lost, [os.path.exists(w) for w, _h in lost],
                              [h is None or _closed(h) for _w, h in lost], manager.active_session_count)
            for sid in registered:
                manager.destroy_sandbox(sid)
    finally:
        restore()
    for limit, count in ((1, 2), (2, 3)):
        registered, refused, lost, exists, closed, active = results[limit]
        assert len(registered) == limit and active == limit, (
            f"{count} creations raced past a limit of {limit}: {len(registered)} registered")
        assert [str(exc) for exc in refused] == [f"Maximum concurrent sessions reached ({limit})"] * (count - limit), (
            refused)
        assert len(lost) == count - limit and exists == [False] * len(lost), f"a refused workspace was left: {lost}"
        assert closed == [True] * len(lost), "a refused workspace's hold was left open"


def test_nl2_what_the_session_limit_keeps(tmp_path):
    sm, _ce, restore = ce_t._window(tmp_path, ce_t._ON)
    try:
        manager = ce_t._manager(sm, tmp_path, max_concurrent_sessions=1)
        first = manager.create_sandbox("nl-a").session_id
        real_make = manager._make_workspace
        made_at_limit = []
        manager._make_workspace = lambda sid: made_at_limit.append(sid) or real_make(sid)
        try:
            manager.create_sandbox("nl-b")
            second = None
        except ValueError as exc:
            second = str(exc)
        manager._make_workspace = real_make
        # A session no longer active (one a destroy has begun with) counts against neither count
        manager.get_session(first).active = False
        beside = manager.create_sandbox("nl-beside").session_id
        manager.get_session(first).active = True
        manager.destroy_sandbox(beside)
        manager.destroy_sandbox(first)
        again = manager.create_sandbox("nl-c").session_id
        try:
            manager.create_sandbox("nl-c")
            same = None
        except ValueError as exc:
            same = str(exc)
        manager.destroy_sandbox(again)
        manager2 = ce_t._manager(sm, tmp_path / "two", max_concurrent_sessions=5)
        answers, made = _race(manager2, ["nl-same", "nl-same"])
        left = [os.path.exists(w) for w, _h in made]
        for answer in answers:
            if isinstance(answer, str):
                manager2.destroy_sandbox(answer)
    finally:
        restore()
    assert second == "Maximum concurrent sessions reached (1)", second
    assert made_at_limit == [], f"a creation refused at the first count made a workspace first: {made_at_limit}"
    assert beside == "nl-beside", beside
    assert again == "nl-c" and same == "Session 'nl-c' already exists", (again, same)
    assert sorted(a if isinstance(a, str) else str(a) for a in answers) == [
        "Session 'nl-same' already exists", "nl-same"], answers
    assert sorted(left) == [False, True], f"the workspace of the creation that lost its id was left: {left}"


def test_nl3_the_notes_routes_share_the_manager():
    shared = object()
    made = []
    deps = types.ModuleType("opti_oignon.api.deps")
    deps.sandbox_manager = shared
    stand_in = types.ModuleType("opti_oignon.sandbox_manager")
    stand_in.SandboxManager = lambda *a, **k: made.append(a) or object()
    notes = {}
    for name, attrs in (("blob_store", ("get_notes_blob_store",)), ("notes_store", ("get_notes_store",)),
                        ("caption", ("build_live_captioner", "caption_attachment")),
                        ("transcription", ("build_live_transcriber", "transcribe_attachment"))):
        module = types.ModuleType(f"opti_oignon.notes.{name}")
        for attr in attrs:
            setattr(module, attr, lambda *a, **k: None)
        notes[f"opti_oignon.notes.{name}"] = module
    resolved = {}
    for route in ("routes_notes_caption", "routes_notes_transcription"):
        loaded, restore = isolate(
            targets={
                "opti_oignon.api.schemas": source("api", "schemas.py"),
                f"opti_oignon.api.{route}": source("api", f"{route}.py"),
            },
            seeded={"opti_oignon.api.deps": deps, "opti_oignon.sandbox_manager": stand_in, **notes},
            packages=("opti_oignon.api", "opti_oignon.notes"),
        )
        try:
            module = loaded[f"opti_oignon.api.{route}"]
            first = module._sandbox_dep()
            deps.sandbox_manager = None
            unavailable = module._sandbox_dep()
            deps.sandbox_manager = shared
            resolved[route] = (first is shared, unavailable)
        finally:
            restore()
    makers = []
    # The package's modules, its data directory never walked
    for path in sorted(package_modules().values()):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "SandboxManager":
                makers.append(str(path.relative_to(REPO)))
    assert resolved == {route: (True, None) for route in resolved}, f"a notes route made its own manager: {resolved}"
    assert made == [], f"a notes route made a SandboxManager: {made}"
    assert makers == ["opti_oignon/sandbox_manager.py"], f"a SandboxManager made outside the shared one: {makers}"
