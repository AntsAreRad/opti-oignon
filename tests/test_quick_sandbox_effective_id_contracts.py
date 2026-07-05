#!/usr/bin/env python3
"""Contracts for the servable sandbox id carried by the done metadata.

The chat UI lists, previews, approves and downloads sandbox files
through ``/api/sandbox/...`` routes keyed by the MANAGER-side id. A
quick session adopting a bound workspace is served under the workspace
id, not under its own conversation-keyed id -- so the done metadata
must carry the id the API actually serves, or every chat-side file
action 404s for bound workspaces. These contracts pin that seam:

  * Contract 1 -- own session: with no bound workspace the servable id
    is the session's own id, and the files written through the session
    are listed under it.
  * Contract 2 -- adopted session: with a bound workspace the servable
    id is the WORKSPACE id (not the conversation-keyed session id), the
    files land under the workspace, and nothing is created under the
    session's own id.
  * Contract 3 -- the streaming layer emits the servable id: the done
    message's ``sandbox_session_id`` metadata is the servable id, not
    the session's own id.

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. Two isolated loads are used: the quick sandbox
module with in-memory sandbox stand-ins, and the chat routes module with
a stand-in dependency container, a spy sandbox pool and a fake executor
(fastapi/pydantic are the real packages when installed, minimal
stand-ins otherwise).
"""

import asyncio
import importlib.util
import sys
import time as real_time
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


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

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def create_sandbox(self, session_id: str, allow_degraded: bool = True):
        self.create_calls.append(session_id)
        box = FakeSandbox(session_id)
        self.sessions[session_id] = box
        return box

    def destroy_sandbox(self, session_id: str) -> bool:
        box = self.sessions.pop(session_id, None)
        if box is not None:
            box.active = False
            return True
        return False

    def extract_files(self, session_id: str):
        box = self.sessions.get(session_id)
        if box is None:
            raise ValueError(f"Session not found: {session_id}")
        return [
            {"path": name, "size": len(body), "modified": 0.0}
            for name, body in sorted(box.files.items())
        ]


def _load_quick_sandbox():
    keys = (
        "opti_oignon", "opti_oignon.sandbox_manager",
        "opti_oignon.file_tools", "opti_oignon.quick_sandbox",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    sm = types.ModuleType("opti_oignon.sandbox_manager")
    sm.SANDBOX_AVAILABLE = True
    sm.SandboxManager = FakeManager
    sm.SandboxSession = FakeSandbox
    sm.sandbox_manager = None
    sys.modules["opti_oignon.sandbox_manager"] = sm
    pkg.sandbox_manager = sm

    ft = types.ModuleType("opti_oignon.file_tools")
    ft.FILE_TOOLS_AVAILABLE = True

    def _bash(session_id, command, timeout=30, _sandbox_manager=None):
        return "Command success (return code: 0)"

    def _view(session_id, path, start_line=0, end_line=0,
              _sandbox_manager=None):
        box = _sandbox_manager.get_session(session_id)
        if box is None:
            return f"Error: unknown session {session_id}"
        if path in box.files:
            return box.files[path]
        return f"Error: Path not found: {path}"

    def _create_file(session_id, path, content, _sandbox_manager=None):
        box = _sandbox_manager.get_session(session_id)
        if box is None:
            return f"Error: unknown session {session_id}"
        box.files[path] = content
        return f"File created: {path}"

    ft._handle_sandbox_bash = _bash
    ft._handle_sandbox_view = _view
    ft._handle_sandbox_create_file = _create_file
    sys.modules["opti_oignon.file_tools"] = ft
    pkg.file_tools = ft

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.quick_sandbox", _OO / "quick_sandbox.py",
    )
    qs = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.quick_sandbox"] = qs
    spec.loader.exec_module(qs)
    pkg.quick_sandbox = qs

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return qs, restore


# ---------------------------------------------------------------------------
# Contract 1 -- own session: the servable id is the session's own id
# ---------------------------------------------------------------------------
def test_c1_own_session_servable_id_is_session_id():
    qs, restore = _load_quick_sandbox()
    try:
        mgr = FakeManager()
        session = qs.QuickSandboxSession(
            "conv-e1", sandbox_mgr=mgr, auto_destroy_minutes=30,
        )
        session.handle_write_file("a.txt", "x")
        got = getattr(session, "effective_sandbox_id", None)
        assert got == "conv-e1", (
            f"own session servable id must be the session id: {got!r}"
        )
        assert got == session.session_id
        listed = [f["path"] for f in mgr.extract_files(got)]
        assert listed == ["a.txt"], listed
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- adopted session: the servable id is the WORKSPACE id
# ---------------------------------------------------------------------------
def test_c2_adopted_session_servable_id_is_workspace_id():
    qs, restore = _load_quick_sandbox()
    try:
        mgr = FakeManager()
        mgr.create_sandbox("ws-42")
        session = qs.QuickSandboxSession(
            "conv-e2", sandbox_mgr=mgr, auto_destroy_minutes=30,
            existing_sandbox_id="ws-42",
        )
        session.handle_write_file("b.txt", "y")
        got = getattr(session, "effective_sandbox_id", None)
        assert got == "ws-42", (
            f"adopted session servable id must be the workspace id: {got!r}"
        )
        assert got != session.session_id
        listed = [f["path"] for f in mgr.extract_files("ws-42")]
        assert "b.txt" in listed, listed
        assert "conv-e2" not in mgr.sessions, (
            "no sandbox may be created under the session's own id on adoption"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Stand-ins for the chat routes load (Contract 3)
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
    """A live quick session with distinct own and servable ids."""

    def __init__(self):
        self.session_id = "conv-live"
        self.effective_sandbox_id = "ws-42"

    def begin_turn(self):
        pass

    def end_turn(self):
        pass

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
    def set_quick_sandbox_mode(self, flag, session=None):
        pass


class FakeRouting:
    def __init__(self):
        self.model = "fake-model"
        self.task_type = "general"
        self.temperature = 0.7
        self.prompt_variant = "default"
        self.routing_reason = "stubbed"
        self.images = None


class FakeExecutor:
    def __init__(self):
        self.last_vision_meta: dict = {}
        self.last_verification_results: list = []

    def reset(self):
        pass

    def cancel(self):
        pass

    def execute(self, **kwargs):
        yield "ok"


class StreamFakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, data):
        self.sent.append(data)


def _load_routes():
    keys = (
        "fastapi", "fastapi.responses", "pydantic",
        "opti_oignon", "opti_oignon.api", "opti_oignon.api.deps",
        "opti_oignon.api.schemas", "opti_oignon.api.routes_chat",
        # Conditional imports of the chat routes module: cleared so a warm
        # interpreter (full-suite run) cannot leak the real modules into
        # this isolated load -- their absence selects the inert branches.
        "opti_oignon.tool_executor", "opti_oignon.emergency_stop",
        "opti_oignon.pipelines", "opti_oignon.agentic_executor",
        "opti_oignon.consensus", "opti_oignon.plugin_hooks",
        "opti_oignon.quick_sandbox", "opti_oignon.tool_registry",
        "opti_oignon.sandbox_workspace", "opti_oignon.tool_call_approval",
        "opti_oignon.security_mode", "opti_oignon.sse_backpressure",
        "opti_oignon.chat_coding_agent",
    )
    saved = {k: sys.modules.get(k) for k in keys}
    for key in keys:
        if key.startswith("opti_oignon"):
            sys.modules.pop(key, None)

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

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg
    api_pkg = types.ModuleType("opti_oignon.api")
    api_pkg.__path__ = []
    sys.modules["opti_oignon.api"] = api_pkg
    pkg.api = api_pkg

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
    sys.modules["opti_oignon.api.deps"] = deps
    api_pkg.deps = deps

    def _real(dotted: str, path: Path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    schemas = _real("opti_oignon.api.schemas", _OO / "api" / "schemas.py")
    api_pkg.schemas = schemas
    rc = _real("opti_oignon.api.routes_chat", _OO / "api" / "routes_chat.py")
    api_pkg.routes_chat = rc

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return rc, schemas, restore


# ---------------------------------------------------------------------------
# Contract 3 -- the done metadata carries the servable id
# ---------------------------------------------------------------------------
def test_c3_done_metadata_carries_servable_id():
    rc, schemas, restore = _load_routes()
    try:
        spy = SpyQuickSession()
        rc.EXECUTOR_AVAILABLE = True
        rc.executor = FakeExecutor()
        rc._resolve_model_and_route = (
            lambda message, request: (FakeRouting(), None)
        )
        rc.QUICK_SANDBOX_AVAILABLE = True
        rc._quick_sandbox_manager = SpyQuickPool(spy)
        rc._tool_registry = SpyToolRegistry()
        rc._get_workspace_bindings = None
        request = schemas.ChatRequest(conversation_id="conv-live", message="run")
        ws = StreamFakeWebSocket()
        asyncio.run(rc._stream_response(ws, "conv-live", "run", request))

        deadline = real_time.time() + 2.0
        done = None
        while done is None and real_time.time() < deadline:
            done = next(
                (d for d in ws.sent if d.get("type") == "done"), None,
            )
            if done is None:
                real_time.sleep(0.01)
        assert done is not None, f"stream did not complete: {ws.sent}"
        meta = done.get("metadata") or {}
        got = meta.get("sandbox_session_id")
        assert got == "ws-42", (
            f"done metadata must carry the servable id, got {got!r} "
            f"(own id is {spy.session_id!r})"
        )
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
