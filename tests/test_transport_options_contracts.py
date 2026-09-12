#!/usr/bin/env python3
"""Contracts that a request timeout and the engine's reported counts travel
through the registry.

The benchmark transports, the judge and the reasoning engine each kept a
client of their own for two things the registry could not carry: a timeout
bound to the transport (a hung model used to block a pipeline for good, and
the fix was a per-timeout client), and the exact token count the engine
reports on its final chunk (the alternative is counting chunks and calling
it tokens). Sending those modules through the registry without carrying
both would have traded a bypass for a silent regression of two measured
fixes. So both travel: the timeout as an engine option, popped before the
options reach the engine and bound to the transport instead; the counts on
``extra`` of the response and of every chunk that reports them.

  * TX1 -- a timeout reaches Ollama as a client bound to that timeout, one
    client per timeout, and never enters the engine options; without one
    the module-level client is used, as before.
  * TX2 -- a timeout reaches llama-server as the request's own timeout on
    both heads and never enters the payload; a timeout that is not a
    number is refused before anything leaves.
  * TX3 -- Ollama's reported counts arrive on the response's ``extra`` and
    on the final chunk's ``extra``; a chunk that reports nothing carries
    an empty ``extra``, never an invented count.
  * TX4 -- the test bridge forwards ``timeout=`` to the scripted client only
    when one is given, and carries a chunk's reported count on ``extra``.

Local-only (the public distribution ships no tests). The backend module is
loaded through the shared isolation window with every client faked.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import ScriptedBackend  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"
_MSGS = [{"role": "user", "content": "hi"}]


class _FakeClient:
    instances = []

    def __init__(self, timeout=None):
        type(self).instances.append(timeout)
        self.timeout = timeout
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([
                {"message": {"content": "He"}},
                {"message": {"content": "llo"}, "done": True, "eval_count": 7, "prompt_eval_count": 3},
            ])
        return {"message": {"content": "ok"}, "eval_count": 5, "prompt_eval_count": 2, "total_duration": 11}


class _FakeOllama:
    def __init__(self):
        self.calls = []
        self.Client = _FakeClient

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"message": {"content": "a"}}, {"message": {"content": "b"}, "done": True}])
        return {"message": {"content": "module"}}


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


# ---------------------------------------------------------------------------
# TX1 -- a timeout binds the Ollama transport
# ---------------------------------------------------------------------------
def test_tx1_a_timeout_binds_an_ollama_client_per_timeout_and_leaves_the_options():
    _FakeClient.instances = []
    mod, restore = _open()
    try:
        backend, fake = _ollama(mod)
        response = backend.generate("m", _MSGS, options={"temperature": 0.1, "timeout": 9})
        assert response.content == "ok"
        assert fake.calls == [], "a timed-out request does not use the module-level client"
        assert _FakeClient.instances == [9]
        client = backend._client_for(9)
        assert client.calls[0]["options"] == {"temperature": 0.1}
        assert "timeout" not in client.calls[0]

        backend.generate("m", _MSGS, options={"timeout": 9})
        assert _FakeClient.instances == [9], "one client per timeout, cached"
        chunks = list(backend.stream("m", _MSGS, options={"timeout": 4, "num_predict": 3}))
        assert "".join(c.content for c in chunks) == "Hello"
        assert _FakeClient.instances == [9, 4]
        assert backend._client_for(4).calls[0]["options"] == {"num_predict": 3}

        plain = backend.generate("m", _MSGS, options={"temperature": 0.5})
        assert plain.content == "module" and fake.calls[0]["options"] == {"temperature": 0.5}
        assert _FakeClient.instances == [9, 4]
    finally:
        restore()


# ---------------------------------------------------------------------------
# TX2 -- a timeout binds the llama-server request
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def read(self):
        return b"".join(self._lines)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    def __init__(self):
        self.timeouts = []
        self.payloads = []

    def urlopen(self, req, timeout=None):
        self.timeouts.append(timeout)
        self.payloads.append(json.loads(req.data.decode("utf-8")))
        if self.payloads[-1].get("stream"):
            return _FakeResponse([
                b'data: {"choices": [{"delta": {"content": "x"}}]}\n',
                b"data: [DONE]\n",
            ])
        return _FakeResponse([json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")])


def test_tx2_a_timeout_binds_the_llama_server_request_and_a_bad_one_is_refused():
    mod, restore = _open()
    try:
        transport = _Transport()
        mod.urllib.request.urlopen = transport.urlopen
        backend = mod.LlamaServerBackend(host="http://fake:8080", timeout_s=120.0)
        backend.generate("m", _MSGS, options={"temperature": 0.2, "timeout": 7})
        assert transport.timeouts == [7.0]
        assert "timeout" not in transport.payloads[0] and transport.payloads[0]["temperature"] == 0.2
        list(backend.stream("m", _MSGS, options={"timeout": 3}))
        assert transport.timeouts == [7.0, 3.0]
        assert "timeout" not in transport.payloads[1]

        backend.generate("m", _MSGS, options={"temperature": 0.2})
        assert transport.timeouts[-1] == 120.0, "without a timeout the backend's own applies"

        with pytest.raises(ValueError, match="timeout"):
            backend.generate("m", _MSGS, options={"timeout": "soon"})
        assert len(transport.payloads) == 3, "a refused timeout sends nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# TX3 -- reported counts travel on extra
# ---------------------------------------------------------------------------
def test_tx3_ollama_reported_counts_arrive_on_extra_and_nothing_is_invented():
    mod, restore = _open()
    try:
        backend, fake = _ollama(mod)
        response = backend.generate("m", _MSGS, options={"timeout": 2})
        assert response.extra["eval_count"] == 5
        assert response.extra["prompt_eval_count"] == 2
        assert response.extra["total_duration"] == 11
        chunks = list(backend.stream("m", _MSGS, options={"timeout": 2}))
        assert chunks[0].extra == {}
        assert chunks[-1].extra["eval_count"] == 7 and chunks[-1].extra["prompt_eval_count"] == 3

        plain = backend.generate("m", _MSGS)
        assert plain.extra == {}, "a client that reports nothing yields no count"
        assert all(c.extra == {} for c in backend.stream("m", _MSGS))
        assert mod.StreamChunk().extra == {}
        assert mod.StreamChunk(extra={"eval_count": 1}).to_dict()["eval_count"] == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# TX4 -- the bridge carries both
# ---------------------------------------------------------------------------
class _Scripted:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"message": {"content": "a"}}, {"message": {"content": "b"}, "done": True, "eval_count": 4}])
        return {"message": {"content": "ok"}, "eval_count": 6}


def test_tx4_the_bridge_forwards_a_timeout_only_when_given_and_carries_counts():
    scripted = _Scripted()
    backend = ScriptedBackend(scripted)
    reply = backend.generate("m", _MSGS, options={"temperature": 0.1, "timeout": 5})
    assert scripted.calls[0]["timeout"] == 5
    assert scripted.calls[0]["options"] == {"temperature": 0.1}
    assert reply.extra["eval_count"] == 6

    chunks = list(backend.stream("m", _MSGS, options={"num_predict": 2}))
    assert "timeout" not in scripted.calls[1]
    assert chunks[0].extra == {} and chunks[-1].extra["eval_count"] == 4

    backend.generate("m", _MSGS)
    assert "timeout" not in scripted.calls[2] and scripted.calls[2]["options"] is None
