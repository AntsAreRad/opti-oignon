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
the __main__ runner. Two loads through the shared isolation window, where
no other project module is reachable: the quick sandbox module with
in-memory sandbox stand-ins, and the chat routes module with a stand-in
dependency container, a spy sandbox pool and a fake executor
(fastapi/pydantic are the real packages when installed, minimal
stand-ins otherwise).
"""

import asyncio
import sys
import time as real_time
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_ABSENT = object()


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
    """The quick sandbox module in the shared window, over in-memory stand-ins."""
    sm = types.ModuleType("opti_oignon.sandbox_manager")
    sm.SANDBOX_AVAILABLE = True
    sm.SandboxManager = FakeManager
    sm.SandboxSession = FakeSandbox
    sm.sandbox_manager = None

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

    loaded, restore = isolate(
        targets={"opti_oignon.quick_sandbox": source("quick_sandbox.py")},
        seeded={"opti_oignon.sandbox_manager": sm, "opti_oignon.file_tools": ft},
    )
    return loaded["opti_oignon.quick_sandbox"], restore


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


def _shim_missing_packages():
    """Stand in for fastapi and pydantic only where they are not installed.

    They are not project modules, so the window does not hold them: what is
    put here is taken back by ``_put_back``, and a real package stays.
    """
    shims = {}
    try:
        import fastapi  # noqa: F401
        import fastapi.responses  # noqa: F401
    except ImportError:
        shim = _fastapi_shim()
        shims["fastapi"] = shim
        shims["fastapi.responses"] = shim.responses
    try:
        import pydantic  # noqa: F401
    except ImportError:
        shims["pydantic"] = _pydantic_shim()
    saved = {name: sys.modules.get(name, _ABSENT) for name in shims}
    sys.modules.update(shims)
    return saved


def _put_back(saved):
    for name, module in saved.items():
        if module is _ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_routes():
    """The chat routes module in the shared window, beside a stand-in container.

    No other project module is reachable, so each conditional import of the
    routes module takes its inert branch.
    """
    saved = _shim_missing_packages()

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

    try:
        loaded, close_window = isolate(
            targets={
                "opti_oignon.api.schemas": source("api", "schemas.py"),
                "opti_oignon.api.routes_chat": source("api", "routes_chat.py"),
            },
            seeded={"opti_oignon.api.deps": deps},
            packages=("opti_oignon.api",),
        )
    except BaseException:
        _put_back(saved)
        raise

    def restore():
        close_window()
        _put_back(saved)

    return loaded["opti_oignon.api.routes_chat"], loaded["opti_oignon.api.schemas"], restore


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
