#!/usr/bin/env python3
"""Contracts for the security-mode floor at the agent run entry.

The machine's security mode is Daily or Bulbe. Leaving Bulbe for Daily is a
ceremony of its own (``security_mode``: four factors and a cooling period),
and the Bulbe tool set keeps the agent off the network and off persistent
memory and skills. A run request is a field in a JSON body: it must never be
the way around that ceremony. These contracts pin the mode a run is started
with:

  * Contract MF1 -- THE MACHINE DECIDES: a request that names no mode runs in
    the machine's current mode, Bulbe on a Bulbe machine and Daily on a Daily
    one, and the request model itself carries no mode of its own by default.
  * Contract MF2 -- NEVER LOOSER: a request that names Daily on a Bulbe machine
    runs in Bulbe; a request that names the stricter Bulbe on a Daily machine
    is honoured, and Daily on a Daily machine stays Daily.
  * Contract MF3 -- FAIL-SECURE: when the machine's mode cannot be read, or the
    request names a mode that does not exist, the run is started in Bulbe.

The agent REST facade and the real ``agent.allowlists`` are loaded through the
shared isolation window; the security mode is a seeded stand-in whose answer
the test sets (or makes raise), the run manager is a recorder, and the web
framework is a minimal stand-in. Local-only. Runs under pytest or the __main__
runner.
"""

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


_WEB_KEYS = ("fastapi", "pydantic")


def _load(machine_mode):
    """Load the facade with the machine in ``machine_mode``.

    ``machine_mode`` is the stand-in ``security_mode.get_current_mode``
    answer, or an exception instance it raises. Returns ``(module, manager,
    restore)``.
    """
    saved_web = {k: sys.modules.get(k) for k in _WEB_KEYS}
    fastapi = types.ModuleType("fastapi")
    fastapi.APIRouter = _Router
    fastapi.HTTPException = _HTTPRefusal
    fastapi.WebSocket = object
    fastapi.WebSocketDisconnect = type("WebSocketDisconnect", (Exception,), {})
    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    sys.modules["fastapi"] = fastapi
    sys.modules["pydantic"] = pydantic

    def _current_mode():
        if isinstance(machine_mode, BaseException):
            raise machine_mode
        return machine_mode

    security_mode = types.ModuleType("opti_oignon.security_mode")
    security_mode.get_current_mode = _current_mode
    estop = types.ModuleType("opti_oignon.emergency_stop")
    estop.guard_http = lambda: None
    capability = types.ModuleType("opti_oignon.capability_manifest")
    capability.model_tool_capable = lambda name: True
    seeded = {
        "opti_oignon.security_mode": security_mode,
        "opti_oignon.emergency_stop": estop,
        "opti_oignon.capability_manifest": capability,
    }
    for sub in ("loop", "skills", "tools"):
        seeded[f"opti_oignon.agent.{sub}"] = types.ModuleType(f"opti_oignon.agent.{sub}")
    try:
        loaded, restore_window = isolate(
            targets={
                "opti_oignon.agent.allowlists": source("agent", "allowlists.py"),
                "opti_oignon.api.routes_agent": source("api", "routes_agent.py"),
            },
            seeded=seeded,
            packages=("opti_oignon.agent", "opti_oignon.api"),
        )
    except BaseException:
        for key, value in saved_web.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        raise
    mod = loaded["opti_oignon.api.routes_agent"]
    manager = _RecorderManager()
    mod._MANAGER = manager

    def restore():
        restore_window()
        for key, value in saved_web.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, manager, restore


_OMITTED = object()


def _started_mode(machine_mode, requested):
    """The mode the run manager is started with for one request.

    ``_OMITTED`` stands for a body with no ``mode`` field: the request then
    carries the request model's own default, as the framework would fill it.
    """
    mod, manager, restore = _load(machine_mode)
    try:
        handler = mod.router.handlers.get(("POST", "/run"))
        assert handler is not None, "the /run handler must be registered"
        mode = getattr(mod.AgentRunRequest, "mode", None) if requested is _OMITTED else requested
        request = SimpleNamespace(task="list the workspace files", mode=mode, model="any-model",
                                  conversation_id="", verify=False, consult=True)
        result = handler(request)
        assert result == {"started": True}, result
        assert len(manager.start_calls) == 1, manager.start_calls
        return manager.start_calls[0][1].get("mode")
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract MF1 -- the machine's mode decides when the request names none
# ---------------------------------------------------------------------------
def test_mf1_a_request_without_a_mode_runs_in_the_machines_mode():
    assert _started_mode("daily", _OMITTED) == "daily", "control: a Daily machine runs a run in Daily"
    started = _started_mode("bulbe", _OMITTED)
    assert started == "bulbe", f"a request naming no mode on a Bulbe machine must run in Bulbe, got {started!r}"
    mod, _manager, restore = _load("bulbe")
    try:
        default = getattr(mod.AgentRunRequest, "mode", "absent")
    finally:
        restore()
    assert default is None, f"the run request must carry no mode of its own by default, got {default!r}"


# ---------------------------------------------------------------------------
# Contract MF2 -- never looser than the machine
# ---------------------------------------------------------------------------
def test_mf2_a_request_never_runs_looser_than_the_machine():
    assert _started_mode("daily", "daily") == "daily", "control: Daily on a Daily machine stays Daily"
    assert _started_mode("daily", "bulbe") == "bulbe", "the stricter mode a request names is honoured"
    started = _started_mode("bulbe", "daily")
    assert started == "bulbe", f"a request naming Daily on a Bulbe machine must run in Bulbe, got {started!r}"


# ---------------------------------------------------------------------------
# Contract MF3 -- an unreadable machine mode or an unknown mode is Bulbe
# ---------------------------------------------------------------------------
def test_mf3_an_unreadable_machine_mode_or_an_unknown_mode_runs_in_bulbe():
    for requested in (_OMITTED, "daily"):
        started = _started_mode(RuntimeError("lockfile unreadable"), requested)
        assert started == "bulbe", (
            f"an unreadable machine mode must run in Bulbe (request {'without a mode' if requested is _OMITTED else requested!r}), "
            f"got {started!r}")
    started = _started_mode("daily", "turbo")
    assert started == "bulbe", f"an unknown mode must run in Bulbe, got {started!r}"


def _run_all():
    cases = (
        ("MF1 the machine decides", test_mf1_a_request_without_a_mode_runs_in_the_machines_mode),
        ("MF2 never looser", test_mf2_a_request_never_runs_looser_than_the_machine),
        ("MF3 fail-secure", test_mf3_an_unreadable_machine_mode_or_an_unknown_mode_runs_in_bulbe),
    )
    failed = 0
    for name, case in cases:
        try:
            case()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
