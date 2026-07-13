#!/usr/bin/env python3
"""Contracts for the model-capability gate on the agent run entry.

The agent loop is tool-bound by construction: its whole purpose is to
stream a model, dispatch the tool calls it emits, and feed observations
back. The run entry therefore poses the tool-calling requirement itself,
so a model that is explicitly unable to call tools never silently drives
the loop and answers tool-less on round one. These contracts pin that
gate at the run entry:

  * Contract G1 -- VERDICT GATE: a request naming a model with an
    explicit negative tool-calling verdict is refused by name (422 with a
    stable reason) and the run never starts; a capable model starts
    unchanged. The verdict is the capability manifest's public predicate,
    consulted with the request's model -- the single source of truth,
    never a local reimplementation.
  * Contract G2 -- INDETERMINABLE CAPABILITY: when the capability
    predicate cannot be imported, the intrinsic requirement fails secure:
    the run is refused by name (422 with its own stable reason) and never
    starts, instead of silently launching a loop whose model may not call
    tools.
  * Contract G3 -- SENTINEL PLACEMENT: an empty model still answers the
    established 503 (no model client) and the capability predicate is
    never consulted for it: the gate sits after client resolution and
    adds no behavior to the empty case.

Loads the agent REST facade in isolation under a stand-in package; every
``opti_oignon.*`` entry plus the web-framework entries is snapshotted and
evicted first, and the seeds are deterministic recorders: a minimal
framework stand-in whose refusal type carries the status code, empty
agent submodules (import-time only; the run manager is a recorder), an
emergency-stop stub, and a controllable capability predicate that records
every name it is asked about. A meta-path guard refuses any project
submodule that was not seeded, so the load behaves identically whether or
not the project is installed. Local-only. Runs under pytest or the
__main__ runner.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the
    test's back -- silently importing live code. This guard sits ahead of
    every finder and refuses the names that were not seeded, so a load
    behaves identically whether the project is installed or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


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

    def delete(self, path, **kwargs):
        return self._decorate("DELETE", path)

    def websocket(self, path, **kwargs):
        return self._decorate("WS", path)


class _RecorderManager:
    """Run-manager stand-in: records start calls, never runs anything."""

    def __init__(self):
        self.start_calls = []

    def start(self, task, **kwargs):
        self.start_calls.append((task, kwargs))
        return {"started": True}

    def status(self):
        return {"running": False, "rounds": 0, "stop_reason": ""}

    def cancel(self):
        return {"cancelled": False}

    def subscribe(self, callback):
        return callback

    def unsubscribe(self, callback):
        return None


# ---------------------------------------------------------------------------
# Isolated loading of the agent REST facade with controlled siblings
# ---------------------------------------------------------------------------
_FACADE_KEYS = (
    "fastapi", "pydantic",
    "opti_oignon", "opti_oignon.agent", "opti_oignon.agent.loop",
    "opti_oignon.agent.skills", "opti_oignon.agent.tools",
    "opti_oignon.emergency_stop", "opti_oignon.capability_manifest",
    "opti_oignon.api", "opti_oignon.api.routes_agent",
)


def _load_facade(*, non_capable_names):
    """Load the agent routes alone against a controllable predicate.

    ``non_capable_names`` is the set the stub predicate answers False for;
    every other name -- a model with no profile -- stays capable, matching
    the manifest's rule. The predicate records every name it is asked
    about. Returns ``(module, manager, predicate_calls, restore)``.
    """
    saved = {k: sys.modules.get(k) for k in _FACADE_KEYS}
    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    fastapi = types.ModuleType("fastapi")
    fastapi.APIRouter = _Router
    fastapi.HTTPException = _HTTPRefusal
    fastapi.WebSocket = object
    fastapi.WebSocketDisconnect = type(
        "WebSocketDisconnect", (Exception,), {},
    )
    sys.modules["fastapi"] = fastapi

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    sys.modules["pydantic"] = pydantic

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    agent_pkg = types.ModuleType("opti_oignon.agent")
    agent_pkg.__path__ = []
    for sub in ("loop", "skills", "tools"):
        mod = types.ModuleType(f"opti_oignon.agent.{sub}")
        sys.modules[f"opti_oignon.agent.{sub}"] = mod
        setattr(agent_pkg, sub, mod)
    sys.modules["opti_oignon.agent"] = agent_pkg
    pkg.agent = agent_pkg

    estop = types.ModuleType("opti_oignon.emergency_stop")
    estop.guard_http = lambda: None
    sys.modules["opti_oignon.emergency_stop"] = estop
    pkg.emergency_stop = estop

    non_capable = set(non_capable_names)
    predicate_calls = []

    def _predicate(name):
        predicate_calls.append(name)
        return name not in non_capable

    cm = types.ModuleType("opti_oignon.capability_manifest")
    cm.model_tool_capable = _predicate
    sys.modules["opti_oignon.capability_manifest"] = cm
    pkg.capability_manifest = cm

    api_pkg = types.ModuleType("opti_oignon.api")
    api_pkg.__path__ = []
    sys.modules["opti_oignon.api"] = api_pkg
    pkg.api = api_pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.api.routes_agent", _OO / "api" / "routes_agent.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.api.routes_agent"] = mod
    spec.loader.exec_module(mod)
    api_pkg.routes_agent = mod

    assert getattr(mod, "router", None) is not None, (
        "the facade must build its router against the framework stand-in"
    )

    manager = _RecorderManager()
    mod._MANAGER = manager

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, manager, predicate_calls, restore


def _run_handler(mod):
    handler = mod.router.handlers.get(("POST", "/run"))
    assert handler is not None, "the /run handler must be registered"
    return handler


def _request(model, task="list the workspace files"):
    return SimpleNamespace(
        task=task,
        mode="daily",
        model=model,
        conversation_id="",
        verify=False,
        consult=True,
    )


# ---------------------------------------------------------------------------
# Contract G1 -- the verdict gate at the run entry
# ---------------------------------------------------------------------------
def test_g1_explicit_negative_model_is_refused_and_capable_starts():
    mod, manager, predicate_calls, restore = _load_facade(
        non_capable_names=["nocap-model"],
    )
    try:
        reason = getattr(mod, "REASON_MODEL_NOT_TOOL_CAPABLE", None)
        assert isinstance(reason, str) and reason, (
            "the facade must expose a stable named reason for the "
            "non-capable refusal"
        )
        handler = _run_handler(mod)

        # (a) explicit negative verdict -> named 422, the run never starts.
        raised = None
        try:
            handler(_request("nocap-model"))
        except _HTTPRefusal as caught:
            raised = caught
        assert raised is not None, (
            "a model with an explicit negative verdict must be refused at "
            "the run entry, not handed the tool loop"
        )
        assert raised.status_code == 422, raised.status_code
        assert raised.detail == reason, (raised.detail, reason)
        assert manager.start_calls == [], (
            "the run must never start for a refused model"
        )
        assert predicate_calls == ["nocap-model"], (
            "the verdict must come from the manifest predicate consulted "
            f"with the request's model, got {predicate_calls}"
        )

        # (b) a capable model starts unchanged.
        del predicate_calls[:]
        result = handler(_request("cap-model"))
        assert result == {"started": True}, result
        assert len(manager.start_calls) == 1, manager.start_calls
        assert predicate_calls == ["cap-model"], predicate_calls
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract G2 -- indeterminable capability fails secure at the run entry
# ---------------------------------------------------------------------------
def test_g2_unimportable_predicate_refuses_and_never_starts():
    mod, manager, _calls, restore = _load_facade(non_capable_names=[])
    try:
        reason = getattr(mod, "REASON_TOOL_CAPABILITY_UNAVAILABLE", None)
        assert isinstance(reason, str) and reason, (
            "the facade must expose a stable named reason for the "
            "indeterminable-capability refusal"
        )
        handler = _run_handler(mod)

        pkg = sys.modules["opti_oignon"]
        saved_cm = sys.modules.get("opti_oignon.capability_manifest")
        sys.modules["opti_oignon.capability_manifest"] = None
        try:
            raised = None
            try:
                handler(_request("any-model"))
            except _HTTPRefusal as caught:
                raised = caught
            assert raised is not None, (
                "an unimportable capability predicate must refuse the run "
                "by name, never start a loop of indeterminable capability"
            )
            assert raised.status_code == 422, raised.status_code
            assert raised.detail == reason, (raised.detail, reason)
            assert manager.start_calls == [], manager.start_calls
        finally:
            if saved_cm is None:
                sys.modules.pop("opti_oignon.capability_manifest", None)
            else:
                sys.modules["opti_oignon.capability_manifest"] = saved_cm
            pkg.capability_manifest = saved_cm
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract G3 -- sentinel: the empty-model case is unchanged
# ---------------------------------------------------------------------------
def test_g3_empty_model_keeps_the_503_and_skips_the_predicate():
    mod, manager, predicate_calls, restore = _load_facade(
        non_capable_names=[""],
    )
    try:
        handler = _run_handler(mod)
        raised = None
        try:
            handler(_request(""))
        except _HTTPRefusal as caught:
            raised = caught
        assert raised is not None and raised.status_code == 503, (
            "an empty model must keep the established no-client 503, got "
            f"{raised!r}"
        )
        assert manager.start_calls == [], manager.start_calls
        assert predicate_calls == [], (
            "the gate sits after client resolution: the predicate is never "
            f"consulted for the empty case, got {predicate_calls}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("G1 negative refused, capable starts",
         test_g1_explicit_negative_model_is_refused_and_capable_starts),
        ("G2 unimportable predicate refuses",
         test_g2_unimportable_predicate_refuses_and_never_starts),
        ("G3 empty model keeps 503, predicate skipped",
         test_g3_empty_model_keeps_the_503_and_skips_the_predicate),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
