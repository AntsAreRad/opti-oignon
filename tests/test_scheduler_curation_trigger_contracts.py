#!/usr/bin/env python3
"""Wiring contracts for the memory-curation trigger.

The conservative curation pass (``memory.curation``) is a proven chokepoint
with its own conservatism contracts; this suite pins WHERE it is triggered
from, not what it does. The trigger is the scheduler's manual dispatch,
reached through the authenticated security surface, and it must inherit the
proven posture rather than widen it:

  * Contract W1 -- CONSERVATIVE DELEGATION: the scheduler's curation trigger
    calls the pinned curation entry point with its conservative defaults --
    the hard-delete channel is never opened from this trigger (the pass
    stays recoverable), no force, the model pass allowed -- and surfaces the
    pass report;
  * Contract W2 -- ROUTE DISPATCH: the scheduler trigger endpoint accepts
    the curation task and dispatches it to the scheduler's curation trigger
    (not to any other task), returning its result;
  * Contract W3 -- FAIL-CLOSED STOP GATE: the curation task mutates the
    memory store, so an engaged emergency stop refuses it (503) before any
    dispatch, and an unavailable emergency-stop module refuses it too: an
    indeterminable stop state never opens a mutating trigger;
  * Contract W4 -- SENTINEL: an unknown task is still refused (400), and
    the two established tasks dispatch unchanged, without consulting the
    stop gate.

Loads the scheduler and the security REST facade in isolation under a
stand-in package; every ``opti_oignon.*`` entry plus the web-framework
entries is snapshotted and evicted first, and the seeds are deterministic
recorders (a curation entry recorder, a scheduler recorder, a controllable
emergency-stop stub, a minimal framework stand-in whose refusal type
carries the status code). A meta-path guard refuses any project submodule
that was not seeded, so the load behaves identically whether or not the
project is installed. Local-only. Runs under pytest or the __main__ runner.
"""

import asyncio
import sys
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


class _HTTPRefusal(Exception):
    """Framework stand-in refusal carrying the status code and detail."""

    def __init__(self, status_code, detail=""):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class _Router:
    """Framework stand-in router capturing the registered handlers."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.handlers = {}

    def _decorate(self, method, path):
        def deco(fn):
            self.handlers[(method, path)] = fn
            return fn
        return deco

    def get(self, path, **kwargs):
        return self._decorate("GET", path)

    def post(self, path, **kwargs):
        return self._decorate("POST", path)

    def put(self, path, **kwargs):
        return self._decorate("PUT", path)

    def delete(self, path, **kwargs):
        return self._decorate("DELETE", path)

    def websocket(self, path, **kwargs):
        return self._decorate("WS", path)


# ---------------------------------------------------------------------------
# Isolated loading of the scheduler with a curation-entry recorder


# The redteam config is reached conditionally by the scheduler; its ABSENCE
# selects the inert branch the clauses reason about, so it is declared here and
# proven unreachable before the scheduler loads.
_SCHED_BLOCKED = ("opti_oignon.redteam", "opti_oignon.redteam.config")


def _load_scheduler():
    """Load the scheduler alone against a recorder curation entry.

    Returns ``(scheduler_instance, curate_calls, restore)``: the recorder
    appends the ``(args, kwargs)`` of every curation-entry call and answers a
    deterministic report object.

    A guard at the head of the meta path refuses project names no finder should
    resolve, but a guard alone does not close the window: Python reads the
    module cache BEFORE it consults any finder, so a sibling some earlier suite
    imported for real answers straight out of the cache and the guard is never
    asked. The shared window evicts every project key it did not seed as well as
    guarding the namespace, so what the scheduler can reach here does not depend
    on what ran before it.
    """
    dep = types.ModuleType("opti_oignon.dep_monitor")

    class _DepMonitor:
        def __init__(self, severity_threshold="high", clock=None):
            self.severity_threshold = severity_threshold

        def get_summary(self):
            return {"last_audit": None, "audit_count": 0}

        def run_audit(self):
            return SimpleNamespace(
                filtered_count=0, filtered_findings=[], to_dict=lambda: {},
            )

    dep.DependencyMonitor = _DepMonitor

    curate_calls = []

    def _curate(*args, **kwargs):
        curate_calls.append((args, kwargs))
        return SimpleNamespace(
            skipped=False,
            fingerprint="fp-after",
            considered=4,
            consolidated=1,
            retired=2,
            retired_ids=["fact-a", "fact-b"],
        )

    curation = types.ModuleType("opti_oignon.memory.curation")
    curation.curate = _curate

    loaded, restore = isolate(
        targets={
            "opti_oignon.security_scheduler": source("security_scheduler.py"),
        },
        blocked=_SCHED_BLOCKED,
        seeded={
            "opti_oignon.dep_monitor": dep,
            "opti_oignon.memory.curation": curation,
        },
        packages=("opti_oignon.memory",),
    )

    config = SimpleNamespace(dep_severity_threshold="high")
    scheduler = loaded["opti_oignon.security_scheduler"].SecurityScheduler(
        config=config,
    )
    return scheduler, curate_calls, restore


# ---------------------------------------------------------------------------
# Isolated loading of the security REST facade with controlled siblings
# ---------------------------------------------------------------------------


class _RecorderScheduler:
    """Scheduler stand-in recording which trigger each dispatch reached."""

    def __init__(self):
        self.redteam_calls = 0
        self.dep_audit_calls = 0
        self.curation_calls = 0

    def trigger_redteam(self):
        self.redteam_calls += 1
        return {"status": "completed", "kind": "redteam"}

    def trigger_dep_audit(self):
        self.dep_audit_calls += 1
        return {"status": "completed", "kind": "dep_audit"}

    def trigger_curation(self):
        self.curation_calls += 1
        return {"status": "completed", "kind": "memory_curation"}


def _load_routes(*, seed_estop=True, estop_stopped=False):
    """Load the security routes alone against recorder siblings.

    ``seed_estop`` False leaves the emergency-stop module UNIMPORTABLE, so the
    facade's guarded import resolves it to None. That absence is the whole
    condition of the stop-free clauses, and it has to be manufactured: the stop
    module is one every other suite in the estate imports, so in a warm
    interpreter it sits in the module cache, where Python finds it before any
    guard is consulted. The shared window evicts it and then PROVES it
    unreachable, so a stop-free clause reports on the guarded import and not on
    the live stop.

    Returns ``(module, scheduler_recorder, guard_calls, restore)``.
    """
    fastapi = types.ModuleType("fastapi")
    fastapi.APIRouter = _Router
    fastapi.HTTPException = _HTTPRefusal
    fastapi.Depends = lambda fn=None: fn
    fastapi.Query = lambda default=None, **kwargs: default
    fastapi.Request = object

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = lambda default=None, **kwargs: default

    seeded = {"fastapi": fastapi, "pydantic": pydantic}
    blocked = []

    guard_calls = []
    if seed_estop:
        estop = types.ModuleType("opti_oignon.emergency_stop")

        def _guard_http():
            guard_calls.append(True)
            if estop_stopped:
                raise _HTTPRefusal(503, "Emergency stop engaged")

        estop.guard_http = _guard_http
        estop.is_stopped = lambda: bool(estop_stopped)
        estop.status = lambda: {"stopped": bool(estop_stopped)}
        seeded["opti_oignon.emergency_stop"] = estop
    else:
        blocked.append("opti_oignon.emergency_stop")

    scheduler = _RecorderScheduler()
    sched_mod = types.ModuleType("opti_oignon.security_scheduler")
    sched_mod.get_scheduler = lambda config=None: scheduler
    seeded["opti_oignon.security_scheduler"] = sched_mod

    loaded, restore = isolate(
        targets={
            "opti_oignon.api.routes_security": source(
                "api", "routes_security.py",
            ),
        },
        blocked=tuple(blocked),
        seeded=seeded,
        packages=("opti_oignon.api",),
    )
    mod = loaded["opti_oignon.api.routes_security"]

    assert getattr(mod, "router", None) is not None, (
        "the facade must build its router against the framework stand-in"
    )

    return mod, scheduler, guard_calls, restore


def _trigger_handler(mod):
    handler = mod.router.handlers.get(("POST", "/scheduler/trigger"))
    assert handler is not None, "the trigger handler must be registered"
    return handler


def _post(handler, task):
    return asyncio.run(handler(SimpleNamespace(task=task)))


# ---------------------------------------------------------------------------
# Contract W1 -- conservative delegation to the pinned curation entry
# ---------------------------------------------------------------------------
def test_w1_scheduler_trigger_delegates_conservatively():
    scheduler, curate_calls, restore = _load_scheduler()
    try:
        result = scheduler.trigger_curation()

        assert len(curate_calls) == 1, (
            "the curation trigger must call the pinned curation entry "
            "exactly once"
        )
        args, kwargs = curate_calls[0]
        assert kwargs.get("hard_delete") is False, (
            "the trigger must never open the hard-delete channel: the "
            "pass stays recoverable"
        )
        assert kwargs.get("force") is False, (
            "the trigger must keep the fingerprint gate (no forced pass)"
        )
        assert kwargs.get("use_llm") is True, (
            "the trigger must keep the model pass allowed (the entry's "
            "own confidence gate applies)"
        )
        assert not args, (
            "the trigger must not target a specific user (the entry "
            "resolves the local user itself)"
        )

        assert result.get("status") == "completed"
        assert result.get("retired") == 2
        assert result.get("retired_ids") == ["fact-a", "fact-b"]
        assert result.get("consolidated") == 1
        assert result.get("considered") == 4
        assert result.get("skipped") is False
        assert result.get("fingerprint") == "fp-after"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract W2 -- the route dispatches the curation task to the scheduler
# ---------------------------------------------------------------------------
def test_w2_route_dispatches_curation_task_to_the_curation_trigger():
    mod, scheduler, _guard_calls, restore = _load_routes()
    try:
        handler = _trigger_handler(mod)
        payload = _post(handler, "memory_curation")

        assert scheduler.curation_calls == 1, (
            "the curation task must reach the scheduler's curation trigger"
        )
        assert scheduler.redteam_calls == 0
        assert scheduler.dep_audit_calls == 0
        assert payload.get("task") == "memory_curation"
        assert payload.get("result", {}).get("kind") == "memory_curation", (
            "the endpoint must surface the curation trigger's own result"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract W3 -- the stop gate is fail-closed for the curation task
# ---------------------------------------------------------------------------
def test_w3_engaged_or_unavailable_stop_refuses_curation_before_dispatch():
    # Face 1: an engaged emergency stop refuses the mutating trigger.
    mod, scheduler, guard_calls, restore = _load_routes(estop_stopped=True)
    try:
        handler = _trigger_handler(mod)
        refusal = None
        try:
            _post(handler, "memory_curation")
        except _HTTPRefusal as exc:
            refusal = exc
        assert refusal is not None, (
            "an engaged emergency stop must refuse the curation task"
        )
        assert refusal.status_code == 503
        assert guard_calls, "the refusal must come from the stop gate"
        assert scheduler.curation_calls == 0, (
            "the curation trigger must never be dispatched under an "
            "engaged stop"
        )
    finally:
        restore()

    # Face 2: an unimportable emergency-stop module refuses too.
    mod, scheduler, _guard_calls, restore = _load_routes(seed_estop=False)
    try:
        handler = _trigger_handler(mod)
        refusal = None
        try:
            _post(handler, "memory_curation")
        except _HTTPRefusal as exc:
            refusal = exc
        assert refusal is not None, (
            "an indeterminable stop state must refuse the curation task "
            "(fail closed)"
        )
        assert refusal.status_code == 503
        assert scheduler.curation_calls == 0, (
            "the curation trigger must never be dispatched when the stop "
            "state cannot be determined"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract W4 -- sentinel: the established dispatch is unchanged
# ---------------------------------------------------------------------------
def test_w4_unknown_task_still_400_and_legacy_tasks_dispatch_stop_free():
    mod, scheduler, guard_calls, restore = _load_routes(estop_stopped=True)
    try:
        handler = _trigger_handler(mod)

        refusal = None
        try:
            _post(handler, "not-a-task")
        except _HTTPRefusal as exc:
            refusal = exc
        assert refusal is not None and refusal.status_code == 400, (
            "an unknown task must keep its established 400 refusal"
        )

        payload = _post(handler, "redteam")
        assert payload.get("result", {}).get("kind") == "redteam"
        payload = _post(handler, "dep_audit")
        assert payload.get("result", {}).get("kind") == "dep_audit"
        assert scheduler.redteam_calls == 1
        assert scheduler.dep_audit_calls == 1
        assert scheduler.curation_calls == 0
        assert not guard_calls, (
            "the established tasks must dispatch without consulting the "
            "stop gate (the gate belongs to the mutating curation task "
            "only)"
        )
    finally:
        restore()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except BaseException:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
