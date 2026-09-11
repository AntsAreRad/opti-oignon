#!/usr/bin/env python3
"""Contracts that a constrained-decoding schema travels through the registry.

Constraining a model's output to a JSON schema already existed here, in
structured_output.py, which calls ``ollama.chat`` directly and never touches
BackendRegistry. So the one engine feature that most needs to be expressed
uniformly across backends was expressed for exactly one of them, outside the
abstraction built to hold it -- and a caller routed through the registry had
no way to ask for it at all.

A schema is an engine option, so it travels in ``options`` where temperature
and top_p already travel. That keeps the abstract interface untouched, leaves
every existing caller working, and means each backend translates the same
request into its own dialect rather than each caller learning three.

  * CD1 -- Ollama receives the schema as its own ``format`` argument, and the
    schema does not leak into the engine options it would not understand.
  * CD2 -- llama-server receives it as a response format on the request body,
    and it does not leak into the forwarded option whitelist.
  * CD3 -- the in-process llama.cpp backend receives it as a response format.
  * CD4 -- asking for nothing changes nothing: with no schema, all three send
    exactly what they sent before, so the feature is inert until used.
  * CD5 -- a schema that is not an object is refused loudly, rather than
    handed down to be misread by whichever engine receives it.
  * CD6 -- the caller's own options dict is never mutated.
  * CD7 -- the streaming path carries the same schema as the whole-answer
    path, on all three backends, so a constrained answer does not become
    unconstrained the moment a caller asks for it a token at a time.

Nothing here claims a model actually honours a schema; that is the engine's
business and the host's to verify. These contracts pin only that the request
leaves carrying what the caller asked for.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; every transport is injected and no backend is reached.
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"

_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
}


_SSE_LINES = [
    b'data: {"choices": [{"delta": {"content": "ok"}}]}\n',
    b"data: [DONE]\n",
]


class _FakeResponse:
    def __init__(self, body, lines=None):
        self._body = body
        self._lines = lines or []

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    """Records each llama-server request body."""

    def __init__(self):
        self.requests = []

    def urlopen(self, req, timeout=None):
        payload = json.loads(req.data.decode("utf-8")) if req.data else None
        self.requests.append(payload)
        return _FakeResponse(
            json.dumps(
                {"choices": [{"message": {"content": "ok"}}], "model": "m"}
            ).encode("utf-8"),
            lines=_SSE_LINES,
        )


class _FakeOllama:
    """Records the keyword arguments handed to ollama.chat."""

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [{"message": {"content": "ok"}, "done": True}]
        return {"message": {"content": "ok"}}


class _FakeLlm:
    """Records the keyword arguments handed to create_chat_completion."""

    def __init__(self):
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [{
                "choices": [
                    {"delta": {"content": "ok"}, "finish_reason": "stop"},
                ],
            }]
        return {"choices": [{"message": {"content": "ok"}}]}


def _open():
    loaded, restore = isolate(
        targets={_BACKEND: source("inference_backend.py")},
        packages=("opti_oignon",),
    )
    return loaded[_BACKEND], restore


def _ollama(mod):
    """An Ollama backend whose client is a recorder."""
    fake = _FakeOllama()
    mod.OLLAMA_AVAILABLE = True
    mod._ollama_module = fake
    return mod.OllamaBackend(), fake


def _llama_server(mod):
    """A llama-server backend whose transport is a recorder."""
    transport = _Transport()
    mod.urllib.request.urlopen = transport.urlopen
    return mod.LlamaServerBackend(host="http://fake:8080"), transport


def _llama_cpp(mod):
    """An in-process backend whose model handle is a recorder."""
    backend = mod.LlamaCppBackend(model_dirs=[])
    llm = _FakeLlm()
    backend._get_or_load = lambda model: llm
    return backend, llm


_MESSAGES = [{"role": "user", "content": "give me a verdict"}]


# ---------------------------------------------------------------------------
# CD1 -- Ollama receives the schema as its own argument
# ---------------------------------------------------------------------------
def test_cd1_ollama_receives_the_schema_as_format():
    mod, restore = _open()
    try:
        backend, fake = _ollama(mod)
        backend.generate(
            "m", _MESSAGES, options={"temperature": 0.1, "schema": _SCHEMA},
        )
        assert len(fake.calls) == 1, "the client was called once"
        call = fake.calls[0]
        assert call.get("format") == _SCHEMA, (
            "the schema arrives as Ollama's own constraining argument"
        )
        assert "schema" not in call["options"], (
            "and does not leak into the engine options, which would carry an "
            "option name Ollama does not know"
        )
        assert call["options"]["temperature"] == 0.1, (
            "the real engine options still travel, so the split is surgical"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CD2 -- llama-server receives it on the request body
# ---------------------------------------------------------------------------
def test_cd2_llama_server_receives_the_schema_as_response_format():
    mod, restore = _open()
    try:
        backend, transport = _llama_server(mod)
        backend.generate(
            "m", _MESSAGES, options={"temperature": 0.1, "schema": _SCHEMA},
        )
        assert len(transport.requests) == 1, "one request left"
        payload = transport.requests[0]
        fmt = payload.get("response_format")
        assert isinstance(fmt, dict), (
            "the schema arrives as a response format on the request body"
        )
        assert fmt.get("schema") == _SCHEMA, "carrying the schema verbatim"
        assert "schema" not in payload, (
            "and not as a bare top-level key the server would ignore"
        )
        assert payload.get("temperature") == 0.1, (
            "the whitelisted options still travel"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CD3 -- the in-process backend receives it too
# ---------------------------------------------------------------------------
def test_cd3_llama_cpp_receives_the_schema_as_response_format():
    mod, restore = _open()
    try:
        backend, llm = _llama_cpp(mod)
        backend.generate(
            "m.gguf", _MESSAGES, options={"temperature": 0.1, "schema": _SCHEMA},
        )
        assert len(llm.calls) == 1, "the model handle was called once"
        fmt = llm.calls[0].get("response_format")
        assert isinstance(fmt, dict), (
            "the schema arrives as a response format"
        )
        assert fmt.get("schema") == _SCHEMA, "carrying the schema verbatim"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CD4 -- asking for nothing changes nothing
# ---------------------------------------------------------------------------
def test_cd4_no_schema_adds_nothing_anywhere():
    mod, restore = _open()
    try:
        backend, fake = _ollama(mod)
        backend.generate("m", _MESSAGES, options={"temperature": 0.1})
        assert "format" not in fake.calls[0], (
            "Ollama is not handed a constraining argument nobody asked for"
        )

        server, transport = _llama_server(mod)
        server.generate("m", _MESSAGES, options={"temperature": 0.1})
        assert "response_format" not in transport.requests[0], (
            "nor is the server's request body given one"
        )

        cpp, llm = _llama_cpp(mod)
        cpp.generate("m.gguf", _MESSAGES, options={"temperature": 0.1})
        assert "response_format" not in llm.calls[0], (
            "nor the in-process handle"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CD5 -- a schema that is not an object is refused loudly
# ---------------------------------------------------------------------------
def test_cd5_a_malformed_schema_is_refused():
    mod, restore = _open()
    try:
        backend, fake = _ollama(mod)
        raised = ""
        try:
            backend.generate("m", _MESSAGES, options={"schema": "an object"})
        except ValueError as exc:
            raised = str(exc)
        assert "schema" in raised, (
            "a schema that is not an object is refused by name, instead of "
            "being handed to an engine that would quietly ignore it"
        )
        assert not fake.calls, (
            "and nothing was sent: the refusal happens before the request"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CD6 -- the caller's options dict is not mutated
# ---------------------------------------------------------------------------
def test_cd6_the_callers_options_are_not_mutated():
    mod, restore = _open()
    try:
        backend, _fake = _ollama(mod)
        options = {"temperature": 0.1, "schema": _SCHEMA}
        backend.generate("m", _MESSAGES, options=options)
        assert options == {"temperature": 0.1, "schema": _SCHEMA}, (
            "the caller's dict survives the call unchanged; taking the schema "
            "out of it in place would silently disarm every later reuse"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CD7 -- the streaming path carries the same schema
# ---------------------------------------------------------------------------
def test_cd7_the_streaming_path_carries_the_schema():
    mod, restore = _open()
    try:
        backend, fake = _ollama(mod)
        list(backend.stream("m", _MESSAGES, options={"schema": _SCHEMA}))
        assert fake.calls[0].get("format") == _SCHEMA, (
            "a streamed answer is constrained exactly as a whole one is"
        )
        assert "schema" not in fake.calls[0]["options"], (
            "and does not leak into the engine options there either"
        )

        cpp, llm = _llama_cpp(mod)
        list(cpp.stream("m.gguf", _MESSAGES, options={"schema": _SCHEMA}))
        fmt = llm.calls[0].get("response_format")
        assert isinstance(fmt, dict) and fmt.get("schema") == _SCHEMA, (
            "the in-process streaming path carries it too"
        )

        server, transport = _llama_server(mod)
        list(server.stream("m", _MESSAGES, options={"schema": _SCHEMA}))
        sfmt = transport.requests[0].get("response_format")
        assert isinstance(sfmt, dict) and sfmt.get("schema") == _SCHEMA, (
            "and so does the server's streaming request body"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("CD1 ollama receives the schema", test_cd1_ollama_receives_the_schema_as_format),
        ("CD2 llama-server receives the schema", test_cd2_llama_server_receives_the_schema_as_response_format),
        ("CD3 llama.cpp receives the schema", test_cd3_llama_cpp_receives_the_schema_as_response_format),
        ("CD4 no schema adds nothing", test_cd4_no_schema_adds_nothing_anywhere),
        ("CD5 malformed schema is refused", test_cd5_a_malformed_schema_is_refused),
        ("CD6 caller options are not mutated", test_cd6_the_callers_options_are_not_mutated),
        ("CD7 streaming carries the schema", test_cd7_the_streaming_path_carries_the_schema),
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
