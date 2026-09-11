#!/usr/bin/env python3
"""Contracts that a tool list travels through the registry and tool calls
come back on the unified response.

Native function calling is what makes an agent's tool selection reliable: the
model was trained on it, and parallel calls come for free. The only place it
existed was a direct client call in the tool executor, which the registry
never saw. A tool list now travels as an engine option, beside the schema
from the previous block, and every backend translates it into its own
dialect; the calls the model makes come back on the response, normalised, and
also in the client's own shape so the existing parser keeps working unchanged.

  * TR1 -- a tool list reaches Ollama as its own ``tools`` argument and does
    not leak into the engine options.
  * TR2 -- it reaches llama-server on the request body.
  * TR3 -- it reaches the in-process llama.cpp handle.
  * TR4 -- the calls a model makes come back on ``tool_calls``, normalised to
    name and arguments, from the client's shape and from the OpenAI-style
    shape alike; and ``to_dict`` still carries them where the parser looks.
  * TR5 -- no tools means nothing is added and ``tool_calls`` is empty.

Nothing here claims a model chooses the right tool; that is the model's
business and the host's to measure. The request leaves carrying the tools,
and the answer comes back carrying the calls.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; every transport is injected.
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"

_TOOLS = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "search the notes",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
}]

_OLLAMA_REPLY = {
    "message": {
        "content": "",
        "tool_calls": [{"function": {"name": "search", "arguments": {"q": "x"}}}],
    },
}
_OPENAI_REPLY = {
    "choices": [{
        "message": {
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "search", "arguments": json.dumps({"q": "x"})},
            }],
        },
    }],
    "model": "m",
}


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    def __init__(self):
        self.requests = []

    def urlopen(self, req, timeout=None):
        self.requests.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(json.dumps(_OPENAI_REPLY).encode("utf-8"))


class _FakeOllama:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools"):
            return _OLLAMA_REPLY
        return {"message": {"content": "ok"}}


class _FakeLlm:
    def __init__(self):
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools"):
            return _OPENAI_REPLY
        return {"choices": [{"message": {"content": "ok"}}]}


def _open():
    loaded, restore = isolate(
        targets={_BACKEND: source("inference_backend.py")},
        packages=("opti_oignon",),
    )
    return loaded[_BACKEND], restore


def _ollama(mod):
    fake = _FakeOllama()
    mod.OLLAMA_AVAILABLE = True
    mod._ollama_module = fake
    return mod.OllamaBackend(), fake


def _llama_server(mod):
    transport = _Transport()
    mod.urllib.request.urlopen = transport.urlopen
    return mod.LlamaServerBackend(host="http://fake:8080"), transport


def _llama_cpp(mod):
    backend = mod.LlamaCppBackend(model_dirs=[])
    llm = _FakeLlm()
    backend._get_or_load = lambda model: llm
    return backend, llm


_MESSAGES = [{"role": "user", "content": "find x"}]
_NORMALISED = [{"name": "search", "arguments": {"q": "x"}}]


# ---------------------------------------------------------------------------
# TR1 -- Ollama receives the tools as its own argument
# ---------------------------------------------------------------------------
def test_tr1_ollama_receives_the_tools_as_its_own_argument():
    mod, restore = _open()
    try:
        backend, fake = _ollama(mod)
        backend.generate("m", _MESSAGES, options={"temperature": 0.0, "tools": _TOOLS})
        call = fake.calls[0]
        assert call.get("tools") == _TOOLS, "the tool list is the client's tools argument"
        assert "tools" not in call["options"], "and does not leak into the engine options"
        assert call["options"]["temperature"] == 0.0, "which still travel"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TR2 -- llama-server receives them on the request body
# ---------------------------------------------------------------------------
def test_tr2_llama_server_receives_the_tools_on_the_body():
    mod, restore = _open()
    try:
        backend, transport = _llama_server(mod)
        backend.generate("m", _MESSAGES, options={"tools": _TOOLS})
        assert transport.requests[0].get("tools") == _TOOLS
    finally:
        restore()


# ---------------------------------------------------------------------------
# TR3 -- the in-process handle receives them
# ---------------------------------------------------------------------------
def test_tr3_llama_cpp_receives_the_tools():
    mod, restore = _open()
    try:
        backend, llm = _llama_cpp(mod)
        backend.generate("m.gguf", _MESSAGES, options={"tools": _TOOLS})
        assert llm.calls[0].get("tools") == _TOOLS
    finally:
        restore()


# ---------------------------------------------------------------------------
# TR4 -- the calls come back normalised, and where the parser looks
# ---------------------------------------------------------------------------
def test_tr4_tool_calls_come_back_normalised_from_every_shape():
    mod, restore = _open()
    try:
        backend, _fake = _ollama(mod)
        resp = backend.generate("m", _MESSAGES, options={"tools": _TOOLS})
        assert resp.tool_calls == _NORMALISED, (
            "the client's own shape is normalised to name and arguments"
        )
        echoed = resp.to_dict()["message"]["tool_calls"]
        assert echoed[0]["function"]["name"] == "search", (
            "and to_dict still carries the calls in the client's shape, where "
            "parse_native_tool_calls looks for them"
        )
        assert echoed[0]["function"]["arguments"] == {"q": "x"}

        server, _transport = _llama_server(mod)
        resp = server.generate("m", _MESSAGES, options={"tools": _TOOLS})
        assert resp.tool_calls == _NORMALISED, (
            "the OpenAI-style shape, with arguments as a JSON string, "
            "normalises to the same thing"
        )

        cpp, _llm = _llama_cpp(mod)
        resp = cpp.generate("m.gguf", _MESSAGES, options={"tools": _TOOLS})
        assert resp.tool_calls == _NORMALISED
    finally:
        restore()


# ---------------------------------------------------------------------------
# TR5 -- no tools adds nothing
# ---------------------------------------------------------------------------
def test_tr5_no_tools_adds_nothing_and_leaves_no_calls():
    mod, restore = _open()
    try:
        backend, fake = _ollama(mod)
        resp = backend.generate("m", _MESSAGES, options={"temperature": 0.0})
        assert "tools" not in fake.calls[0]
        assert resp.tool_calls == [], "an answer that called nothing says so"
        assert "tool_calls" not in resp.to_dict()["message"], (
            "and to_dict adds no empty key the parser would have to skip"
        )
    finally:
        restore()


def _run_all():
    tests = [
        ("TR1 ollama receives tools", test_tr1_ollama_receives_the_tools_as_its_own_argument),
        ("TR2 llama-server receives tools", test_tr2_llama_server_receives_the_tools_on_the_body),
        ("TR3 llama.cpp receives tools", test_tr3_llama_cpp_receives_the_tools),
        ("TR4 tool calls come back normalised", test_tr4_tool_calls_come_back_normalised_from_every_shape),
        ("TR5 no tools adds nothing", test_tr5_no_tools_adds_nothing_and_leaves_no_calls),
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
