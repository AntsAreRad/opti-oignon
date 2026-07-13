#!/usr/bin/env python3
"""Contracts for the turn guard of the quick sandbox lifecycle.

The quick sandbox auto-destroys after a wall-clock inactivity window.
Activity was only recorded when a tool call started, so a long
inference between tool calls let the window elapse and the workspace
was destroyed in the middle of a live turn. These contracts pin the
guard that makes the timeout a between-turns notion:

  * Contract 1 -- a live turn pins the workspace: while a turn is in
    flight, the session never reports expired (the full window is
    reported), the pool cleanup does not destroy it, and the pool keeps
    returning the same session.
  * Contract 2 -- the inactivity window restarts when the turn closes:
    right after end_turn the session is fresh, and it only expires a
    full window after the end of the turn.
  * Contract 3 -- the guard cannot leak or underflow: extra end_turn
    calls are harmless (a later begin_turn still pins), expiry outside
    any turn stays nominal, and the fresh-window semantics of
    set_auto_destroy_minutes are unchanged.
  * Contract 4 -- tool handlers refresh activity on completion, not
    only on entry: a handler whose execution outlives the window leaves
    a session that is still alive at return (protects direct callers
    that run outside any turn bracket).
  * Contract 5 -- the chat streaming layer brackets the whole
    generation: the turn opens before the executor runs and is released
    when the generation thread finishes, including when the executor
    raises; with no quick sandbox in play, no bracket is taken.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Two isolated loads are used: the quick
sandbox module with in-memory sandbox stand-ins and a controllable
clock, and the chat routes module with a stand-in dependency container,
a spy sandbox pool and a fake executor (fastapi/pydantic are the real
packages when installed, minimal stand-ins otherwise).
"""

import asyncio
import sys
import time as real_time
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402


# ---------------------------------------------------------------------------
# Controllable clock
# ---------------------------------------------------------------------------
class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# In-memory sandbox world
# ---------------------------------------------------------------------------
class FakeSandbox:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.active = True
        self.files: dict[str, str] = {}


class FakeManager:
    def __init__(self):
        self.sessions: dict[str, FakeSandbox] = {}
        self.create_calls: list[str] = []
        self.destroy_calls: list[str] = []

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def create_sandbox(self, session_id: str, allow_degraded: bool = True):
        self.create_calls.append(session_id)
        box = FakeSandbox(session_id)
        self.sessions[session_id] = box
        return box

    def destroy_sandbox(self, session_id: str) -> bool:
        self.destroy_calls.append(session_id)
        box = self.sessions.pop(session_id, None)
        if box is not None:
            box.active = False
            return True
        return False

    def extract_files(self, session_id: str):
        box = self.sessions.get(session_id)
        if box is None:
            return []
        return [{"path": name} for name in sorted(box.files)]


# ---------------------------------------------------------------------------
# Isolated loading of the quick sandbox module (fake clock + slow handlers)
# ---------------------------------------------------------------------------
def _load_quick_sandbox(clock: FakeClock, handler_delay: float = 0.0):
    """Load the quick sandbox against a fake clock and stubbed handlers.

    The sandbox manager and the file tools are handed in; every other project
    module must be UNREACHABLE, so the module under test cannot reach a live
    sandbox, a live audit sink or a live clock. A stand-in parent package whose
    path is empty does not achieve that wherever a finder answers on the module
    name, and the turn-guard clauses would then be reporting on live machinery.
    """
    sm = types.ModuleType("opti_oignon.sandbox_manager")
    sm.SANDBOX_AVAILABLE = True
    sm.SandboxManager = FakeManager
    sm.SandboxSession = FakeSandbox
    sm.sandbox_manager = None

    ft = types.ModuleType("opti_oignon.file_tools")
    ft.FILE_TOOLS_AVAILABLE = True

    def _bash(session_id, command, timeout=30, _sandbox_manager=None):
        clock.advance(handler_delay)
        return "Command success (return code: 0)"

    def _view(session_id, path, start_line=0, end_line=0,
              _sandbox_manager=None):
        clock.advance(handler_delay)
        box = _sandbox_manager.get_session(session_id)
        if box is None:
            return f"Error: unknown session {session_id}"
        if path in (".", ""):
            listing = "\n".join(sorted(box.files))
            return listing if listing else "(empty workspace)"
        if path in box.files:
            return box.files[path]
        return f"Error: Path not found: {path}"

    def _create_file(session_id, path, content, _sandbox_manager=None):
        clock.advance(handler_delay)
        box = _sandbox_manager.get_session(session_id)
        if box is None:
            return f"Error: unknown session {session_id}"
        box.files[path] = content
        return f"File created: {path}"

    ft._handle_sandbox_bash = _bash
    ft._handle_sandbox_view = _view
    ft._handle_sandbox_create_file = _create_file
    loaded, restore = isolate(
        targets={"opti_oignon.quick_sandbox": source("quick_sandbox.py")},
        seeded={
            "opti_oignon.sandbox_manager": sm,
            "opti_oignon.file_tools": ft,
        },
    )
    qs = loaded["opti_oignon.quick_sandbox"]

    if not qs.QUICK_SANDBOX_AVAILABLE:
        restore()
        raise RuntimeError("quick sandbox reports unavailable under stubs")

    # Route every wall-clock read of the module through the fake clock.
    qs.time = types.SimpleNamespace(time=clock.time)

    return qs, restore


# ---------------------------------------------------------------------------
# Contract 1 -- a live turn pins the workspace
# ---------------------------------------------------------------------------
def test_c1_live_turn_blocks_expiry_and_cleanup():
    clock = FakeClock()
    qs, restore = _load_quick_sandbox(clock)
    try:
        mgr = FakeManager()
        cfg = qs.QuickSandboxConfig(
            enabled=True, auto_destroy_minutes=1,
            max_concurrent_quick_sessions=3,
        )
        pool = qs.QuickSandboxManager(sandbox_mgr=mgr, config=cfg)
        session = pool.get_or_create_session(request_id="conv1")
        session.handle_write_file("a.txt", "x")

        session.begin_turn()
        clock.advance(3600.0)  # far beyond the 60 s window
        assert session.expired is False, "a live turn must pin the session"
        assert session.seconds_until_expiry == 60.0, (
            "the full window must be reported while a turn is in flight"
        )
        assert pool.cleanup_expired() == 0, (
            "cleanup must not destroy a session with a live turn"
        )
        assert mgr.destroy_calls == [], mgr.destroy_calls
        again = pool.get_or_create_session(request_id="conv1")
        assert again is session, "the pool must return the pinned session"
        session.end_turn()
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- the inactivity window restarts at the end of the turn
# ---------------------------------------------------------------------------
def test_c2_window_restarts_at_turn_end():
    clock = FakeClock()
    qs, restore = _load_quick_sandbox(clock)
    try:
        mgr = FakeManager()
        session = qs.QuickSandboxSession(
            "conv2", sandbox_mgr=mgr, auto_destroy_minutes=1,
        )
        session.begin_turn()
        clock.advance(3600.0)
        session.end_turn()
        assert session.expired is False, (
            "the countdown must restart from the end of the turn"
        )
        clock.advance(59.0)
        assert session.expired is False
        clock.advance(2.0)
        assert session.expired is True, (
            "one full window after the end of the turn must expire"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- floor, no leak, nominal expiry outside turns
# ---------------------------------------------------------------------------
def test_c3_counter_floor_and_nominal_expiry():
    clock = FakeClock()
    qs, restore = _load_quick_sandbox(clock)
    try:
        mgr = FakeManager()
        session = qs.QuickSandboxSession(
            "conv3", sandbox_mgr=mgr, auto_destroy_minutes=1,
        )
        # Extra end_turn calls must not underflow the counter: one later
        # begin_turn still pins the session.
        session.end_turn()
        session.end_turn()
        session.begin_turn()
        clock.advance(3600.0)
        assert session.expired is False, (
            "underflowed counter: a single live turn no longer pins"
        )
        session.end_turn()
        clock.advance(61.0)
        assert session.expired is True, (
            "expiry outside any turn must stay nominal"
        )
        # The fresh-window semantics of set_auto_destroy_minutes hold.
        session.set_auto_destroy_minutes(2)
        assert session.expired is False
        clock.advance(119.0)
        assert session.expired is False
        clock.advance(2.0)
        assert session.expired is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- handlers refresh activity on completion
# ---------------------------------------------------------------------------
def test_c4_handler_completion_refreshes_activity():
    clock = FakeClock()
    # Every stubbed low-level handler takes twice the 60 s window.
    qs, restore = _load_quick_sandbox(clock, handler_delay=120.0)
    try:
        mgr = FakeManager()
        session = qs.QuickSandboxSession(
            "conv4", sandbox_mgr=mgr, auto_destroy_minutes=1,
        )
        session.handle_execute_code("print('x')")
        assert session.expired is False, (
            "execute_code completion must refresh activity"
        )
        session.handle_write_file("b.txt", "y")
        assert session.expired is False, (
            "write_file completion must refresh activity"
        )
        session.handle_read_file("b.txt")
        assert session.expired is False, (
            "read_file completion must refresh activity"
        )
        session.handle_list_files(".")
        assert session.expired is False, (
            "list_files completion must refresh activity"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Stand-ins for the chat routes load (Contract 5)
# ---------------------------------------------------------------------------
def _pydantic_shim() -> types.ModuleType:
    mod = types.ModuleType("pydantic")

    class ValidationError(Exception):
        pass

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    class BaseModel:
        def __init__(self, **kwargs):
            for name in getattr(self.__class__, "__annotations__", {}):
                default = getattr(self.__class__, name, None)
                if isinstance(default, (list, dict)):
                    default = type(default)(default)
                setattr(self, name, default)
            for key, value in kwargs.items():
                setattr(self, key, value)

    mod.BaseModel = BaseModel
    mod.Field = Field
    mod.ValidationError = ValidationError
    return mod


def _fastapi_shim() -> types.ModuleType:
    mod = types.ModuleType("fastapi")

    class APIRouter:
        def __init__(self, **kwargs):
            pass

        def websocket(self, *args, **kwargs):
            def wrap(fn):
                return fn
            return wrap

        post = websocket
        get = websocket
        delete = websocket

    class WebSocket:
        pass

    class WebSocketDisconnect(Exception):
        pass

    responses = types.ModuleType("fastapi.responses")

    class JSONResponse:
        def __init__(self, *args, **kwargs):
            pass

    responses.JSONResponse = JSONResponse
    mod.APIRouter = APIRouter
    mod.WebSocket = WebSocket
    mod.WebSocketDisconnect = WebSocketDisconnect
    mod.responses = responses
    return mod


class SpyQuickSession:
    """Records the turn bracket taken by the streaming layer."""

    def __init__(self):
        self.session_id = "spy-session"
        # The servable id the done metadata carries (own id here: this
        # spy adopts nothing).
        self.effective_sandbox_id = "spy-session"
        self.events: list[str] = []

    @property
    def begins(self) -> int:
        return self.events.count("begin")

    @property
    def ends(self) -> int:
        return self.events.count("end")

    def begin_turn(self):
        self.events.append("begin")

    def end_turn(self):
        self.events.append("end")

    def get_sandbox_files(self):
        return []

    @property
    def files_created(self):
        return []


class SpyQuickPool:
    def __init__(self, session: SpyQuickSession):
        self.enabled = True
        self.available = True
        self._session = session

    def get_or_create_session(self, request_id=None, bound_sandbox_id=None):
        return self._session


class SpyToolRegistry:
    def __init__(self):
        self.modes: list[bool] = []

    def set_quick_sandbox_mode(self, flag, session=None):
        self.modes.append(bool(flag))


class FakeRouting:
    def __init__(self):
        self.model = "fake-model"
        self.task_type = "general"
        self.temperature = 0.7
        self.prompt_variant = "default"
        self.routing_reason = "stubbed"
        self.images = None


class FakeExecutor:
    def __init__(self, chunks=("ok",), raise_after=False):
        self._chunks = tuple(chunks)
        self._raise = raise_after
        self.last_vision_meta: dict = {}
        self.last_verification_results: list = []

    def reset(self):
        pass

    def cancel(self):
        pass

    def execute(self, **kwargs):
        yield from self._chunks
        if self._raise:
            raise RuntimeError("generation exploded")


class StreamFakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, data):
        self.sent.append(data)


# Conditional imports of the chat routes module. Their ABSENCE selects the
# inert branches, and that absence is what the route clauses reason about, so
# it is declared here and PROVEN by the window before the routes are loaded.
_ROUTE_CONDITIONAL = (
    "opti_oignon.tool_executor", "opti_oignon.emergency_stop",
    "opti_oignon.pipelines", "opti_oignon.agentic_executor",
    "opti_oignon.consensus", "opti_oignon.plugin_hooks",
    "opti_oignon.quick_sandbox", "opti_oignon.tool_registry",
    "opti_oignon.sandbox_workspace", "opti_oignon.tool_call_approval",
    "opti_oignon.security_mode", "opti_oignon.sse_backpressure",
    "opti_oignon.chat_coding_agent",
)


def _load_routes():
    """Load the chat routes over stubbed dependencies; returns (rc, schemas, restore).

    The third-party frameworks are shimmed only when they are genuinely absent;
    the project window is the isolation module's, so every conditional import
    above is neutralised AND proven unreachable, whatever a finder further down
    the meta path would have answered for it.
    """
    ext_keys = ("fastapi", "fastapi.responses", "pydantic")
    ext_saved = {k: sys.modules.get(k) for k in ext_keys}

    try:
        import fastapi  # noqa: F401
        import fastapi.responses  # noqa: F401
    except ImportError:
        shim = _fastapi_shim()
        sys.modules["fastapi"] = shim
        sys.modules["fastapi.responses"] = shim.responses
    try:
        import pydantic  # noqa: F401
    except ImportError:
        sys.modules["pydantic"] = _pydantic_shim()

    deps = types.ModuleType("opti_oignon.api.deps")
    deps.ANALYZER_AVAILABLE = False
    deps.CONVERSATION_AVAILABLE = False
    deps.EXECUTOR_AVAILABLE = False
    deps.PRESET_AVAILABLE = False
    deps.ROUTER_AVAILABLE = False
    deps.analyzer = None
    deps.conversation_manager = None
    deps.executor = None
    deps.preset_manager = None
    deps.router = None

    loaded, close = isolate(
        targets={
            "opti_oignon.api.schemas": source("api", "schemas.py"),
            "opti_oignon.api.routes_chat": source("api", "routes_chat.py"),
        },
        blocked=_ROUTE_CONDITIONAL,
        seeded={"opti_oignon.api.deps": deps},
        packages=("opti_oignon.api",),
    )

    def restore():
        close()
        for key, value in ext_saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return (
        loaded["opti_oignon.api.routes_chat"],
        loaded["opti_oignon.api.schemas"],
        restore,
    )


def _run_stream(rc, schemas, executor, spy_pool, spy_registry):
    rc.EXECUTOR_AVAILABLE = True
    rc.executor = executor
    rc._resolve_model_and_route = lambda message, request: (FakeRouting(), None)
    rc.QUICK_SANDBOX_AVAILABLE = True
    rc._quick_sandbox_manager = spy_pool
    rc._tool_registry = spy_registry
    rc._get_workspace_bindings = None
    request = schemas.ChatRequest(conversation_id="conv-live", message="run")
    ws = StreamFakeWebSocket()
    asyncio.run(rc._stream_response(ws, "conv-live", "run", request))
    return ws


def _wait_for(predicate, timeout=2.0):
    deadline = real_time.time() + timeout
    while not predicate() and real_time.time() < deadline:
        real_time.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# Contract 5 -- the streaming layer brackets the whole generation
# ---------------------------------------------------------------------------
def test_c5a_stream_brackets_the_turn():
    rc, schemas, restore = _load_routes()
    try:
        spy = SpyQuickSession()
        registry = SpyToolRegistry()
        ws = _run_stream(rc, schemas, FakeExecutor(), SpyQuickPool(spy), registry)
        assert _wait_for(lambda: spy.ends >= 1), (
            f"turn never released: {spy.events}"
        )
        assert spy.events == ["begin", "end"], (
            f"the turn must bracket the generation exactly once: {spy.events}"
        )
        types_sent = [d.get("type") for d in ws.sent]
        assert "done" in types_sent, f"stream did not complete: {types_sent}"
        assert registry.modes and registry.modes[0] is True
        assert registry.modes[-1] is False
    finally:
        restore()


def test_c5b_turn_released_when_executor_raises():
    rc, schemas, restore = _load_routes()
    try:
        spy = SpyQuickSession()
        registry = SpyToolRegistry()
        ws = _run_stream(
            rc, schemas, FakeExecutor(raise_after=True),
            SpyQuickPool(spy), registry,
        )
        assert _wait_for(lambda: spy.ends >= 1), (
            f"turn leaked on executor failure: {spy.events}"
        )
        assert spy.begins == 1 and spy.ends == 1, spy.events
        types_sent = [d.get("type") for d in ws.sent]
        assert "error" in types_sent, f"failure was not surfaced: {types_sent}"
    finally:
        restore()


def test_c5c_no_bracket_without_quick_sandbox():
    rc, schemas, restore = _load_routes()
    try:
        spy = SpyQuickSession()
        registry = SpyToolRegistry()
        rc.EXECUTOR_AVAILABLE = True
        rc.executor = FakeExecutor()
        rc._resolve_model_and_route = (
            lambda message, request: (FakeRouting(), None)
        )
        rc.QUICK_SANDBOX_AVAILABLE = False
        rc._quick_sandbox_manager = None
        rc._tool_registry = registry
        rc._get_workspace_bindings = None
        request = schemas.ChatRequest(conversation_id="conv-off", message="run")
        ws = StreamFakeWebSocket()
        asyncio.run(rc._stream_response(ws, "conv-off", "run", request))
        assert spy.events == [], f"bracket taken without a session: {spy.events}"
        types_sent = [d.get("type") for d in ws.sent]
        assert "done" in types_sent, f"stream did not complete: {types_sent}"
        assert registry.modes == [], registry.modes
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
