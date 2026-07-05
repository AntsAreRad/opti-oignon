#!/usr/bin/env python3
"""Contracts for the model carried by the chat retry endpoint.

A retry regenerates the last answer of a conversation. The regeneration
must run on the model the conversation was using, not on the routing
default: losing the model silently reroutes the retry (and 404s when the
default resolves to a model that does not exist). These contracts pin
the recovery chain of the retry request:

  * Contract 1 -- the conversation's last used model is transported: a
    retry on a conversation whose record carries a model produces a
    downstream chat request forcing that model.
  * Contract 2 -- an explicit model in the retry request wins over the
    conversation record (the request schema accepts the override).
  * Contract 3 -- when the conversation record carries no model, the
    newest assistant message that recorded one is used (read before the
    history is rewound).
  * Contract 4 -- honesty: with no model anywhere, the downstream
    request carries none and routing keeps its default behavior.
  * Contract 5 -- the history rewind is unchanged: the last assistant
    message then the last user message are removed, and the user
    message is re-sent verbatim.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. The chat routes module is loaded in
isolation: the app dependency container and the auth layer are replaced
by stand-ins, the schemas module is the real one, and fastapi/pydantic
are the real packages when installed (minimal stand-ins otherwise). The
websocket endpoint itself is driven end to end with a fake socket; the
streaming layer is replaced by a spy that captures the rebuilt request.
"""

import asyncio
import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Minimal stand-ins (only when the real packages are absent)
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


# ---------------------------------------------------------------------------
# Conversation world stand-ins
# ---------------------------------------------------------------------------
class FakeMessage:
    def __init__(self, role: str, content: str, model: str | None = None):
        self.role = role
        self.content = content
        self.model = model


class FakeConversation:
    def __init__(self, conv_id: str, model: str | None = None):
        self.id = conv_id
        self.model = model


class FakeConversationManager:
    """Mirrors the pieces of the real manager the retry endpoint touches."""

    def __init__(self, conversation, messages):
        self._conversation = conversation
        self._messages = list(messages)
        self.deleted: list[tuple[str, str]] = []

    def get_conversation(self, conv_id: str):
        if self._conversation is not None and self._conversation.id == conv_id:
            return self._conversation
        return None

    def get_messages(self, conv_id: str):
        return list(self._messages)

    def delete_last_message(self, conv_id: str, role: str | None = None):
        if not self._messages:
            return False
        last = self._messages[-1]
        if role is not None and last.role != role:
            return False
        self._messages.pop()
        self.deleted.append((last.role, last.content))
        return True


class FakeWebSocket:
    def __init__(self, payload: dict):
        self._payload = payload
        self.sent: list[dict] = []
        self.close_calls = 0

    async def accept(self):
        pass

    async def receive_json(self):
        return self._payload

    async def send_json(self, data):
        self.sent.append(data)

    async def close(self, code=None):
        self.close_calls += 1


# ---------------------------------------------------------------------------
# Isolated loading
# ---------------------------------------------------------------------------
def _load():
    keys = (
        "fastapi", "fastapi.responses", "pydantic",
        "opti_oignon", "opti_oignon.api", "opti_oignon.api.deps",
        "opti_oignon.api.schemas", "opti_oignon.api.routes_auth",
        "opti_oignon.api.routes_chat",
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

    auth = types.ModuleType("opti_oignon.api.routes_auth")

    async def authenticate_websocket(websocket):
        return {"username": "local"}

    auth.authenticate_websocket = authenticate_websocket
    sys.modules["opti_oignon.api.routes_auth"] = auth
    api_pkg.routes_auth = auth

    rc = _real("opti_oignon.api.routes_chat", _OO / "api" / "routes_chat.py")
    api_pkg.routes_chat = rc

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return rc, schemas, restore


def _run_retry(rc, manager, payload):
    """Drive the real retry endpoint end to end; capture the rebuilt request."""
    captured: dict = {}

    async def _spy_stream(websocket, conversation_id, message, request):
        captured["conversation_id"] = conversation_id
        captured["message"] = message
        captured["request"] = request

    rc.conversation_manager = manager
    rc.CONVERSATION_AVAILABLE = True
    rc._emergency_stop = None
    rc._stream_response = _spy_stream
    ws = FakeWebSocket(payload)
    asyncio.run(rc.chat_retry(ws))
    return captured, ws


# ---------------------------------------------------------------------------
# Contract 1 -- the conversation's last used model is transported
# ---------------------------------------------------------------------------
def test_c1_conversation_model_is_transported():
    rc, _schemas, restore = _load()
    try:
        conv = FakeConversation("c1", model="conv-model")
        manager = FakeConversationManager(conv, [
            FakeMessage("user", "hello"),
            FakeMessage("assistant", "hi", model=None),
        ])
        captured, ws = _run_retry(rc, manager, {"conversation_id": "c1"})
        assert "request" in captured, f"stream never invoked: {ws.sent}"
        got = getattr(captured["request"], "model", None)
        assert got == "conv-model", f"retry lost the conversation model: {got!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- an explicit model in the retry request wins
# ---------------------------------------------------------------------------
def test_c2_request_override_wins():
    rc, _schemas, restore = _load()
    try:
        conv = FakeConversation("c1", model="conv-model")
        manager = FakeConversationManager(conv, [
            FakeMessage("user", "hello"),
            FakeMessage("assistant", "hi", model="conv-model"),
        ])
        captured, ws = _run_retry(
            rc, manager, {"conversation_id": "c1", "model": "override-x"},
        )
        assert "request" in captured, f"stream never invoked: {ws.sent}"
        got = getattr(captured["request"], "model", None)
        assert got == "override-x", f"request override was not honored: {got!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- fallback to the newest assistant message that has a model
# ---------------------------------------------------------------------------
def test_c3_last_assistant_message_model_fallback():
    rc, _schemas, restore = _load()
    try:
        conv = FakeConversation("c1", model=None)
        manager = FakeConversationManager(conv, [
            FakeMessage("user", "first question"),
            FakeMessage("assistant", "old answer", model="older-model"),
            FakeMessage("user", "second question"),
            FakeMessage("assistant", "new answer", model="msg-model"),
        ])
        captured, ws = _run_retry(rc, manager, {"conversation_id": "c1"})
        assert "request" in captured, f"stream never invoked: {ws.sent}"
        got = getattr(captured["request"], "model", None)
        assert got == "msg-model", (
            f"newest assistant model was not recovered: {got!r}"
        )
        assert captured["message"] == "second question"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- honesty: no model anywhere stays no model
# ---------------------------------------------------------------------------
def test_c4_no_model_anywhere_stays_none():
    rc, _schemas, restore = _load()
    try:
        conv = FakeConversation("c1", model=None)
        manager = FakeConversationManager(conv, [
            FakeMessage("user", "hello"),
            FakeMessage("assistant", "hi", model=None),
        ])
        captured, ws = _run_retry(rc, manager, {"conversation_id": "c1"})
        assert "request" in captured, f"stream never invoked: {ws.sent}"
        got = getattr(captured["request"], "model", None)
        assert got is None, f"a model was invented out of nothing: {got!r}"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 5 -- the history rewind is unchanged
# ---------------------------------------------------------------------------
def test_c5_history_rewind_preserved():
    rc, _schemas, restore = _load()
    try:
        conv = FakeConversation("c1", model="conv-model")
        manager = FakeConversationManager(conv, [
            FakeMessage("user", "the question"),
            FakeMessage("assistant", "the answer", model="conv-model"),
        ])
        captured, ws = _run_retry(rc, manager, {"conversation_id": "c1"})
        assert "request" in captured, f"stream never invoked: {ws.sent}"
        assert manager.deleted == [
            ("assistant", "the answer"),
            ("user", "the question"),
        ], f"unexpected rewind order: {manager.deleted}"
        assert captured["message"] == "the question"
        assert captured["conversation_id"] == "c1"
        assert getattr(captured["request"], "conversation_id", None) == "c1"
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
