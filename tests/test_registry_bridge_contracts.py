#!/usr/bin/env python3
"""Contracts for the registry bridge the tool suites now stand on.

The bridge seeds a registry whose backend forwards each request to a
suite's scripted client with the kwargs the former direct call carried, and
carries the client's reply back in the shape the registry promises. Twenty
eight contracts across six suites read the scripted client through it, so
what it forwards and what it carries back are properties, not details.

  * RB1 -- native tools travel: ``options["tools"]`` reaches the client as
    ``tools=`` and only then; the reply's ``tool_calls`` are the client's,
    normalised, and ``to_dict()`` keeps the client's own shape.
  * RB2 -- a schema travels as ``format=`` and only then.
  * RB3 -- a stream is asked for as a stream and its chunks come back as
    the objects the hub reads.
  * RB4 -- the seeded registry registers, activates and unregisters like
    the real one, so a production seam can lay a backend over it for a
    block and take it away again.

Local-only (the public distribution ships no tests). The bridge is stdlib
and import-safe; no window is needed to test it.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _registry_bridge import ScriptedBackend, seed_registry  # noqa: E402

_MSGS = [{"role": "user", "content": "find x"}]
_TOOL = {"type": "function", "function": {"name": "search", "parameters": {}}}


class _Client:
    def __init__(self, reply=None, chunks=None):
        self.calls, self.reply, self.chunks = [], reply, chunks

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.chunks or [])
        return self.reply


# ---------------------------------------------------------------------------
# RB1 -- native tools travel, and the calls come back
# ---------------------------------------------------------------------------
def test_rb1_tools_travel_to_the_client_and_the_calls_come_back_in_both_shapes():
    raw = [SimpleNamespace(function=SimpleNamespace(name="search", arguments={"q": "x"}))]
    client = _Client(reply=SimpleNamespace(message=SimpleNamespace(content="", tool_calls=raw)))
    backend = ScriptedBackend(client)
    reply = backend.generate("m", _MSGS, options={"temperature": 0.0, "tools": [_TOOL]})
    call = client.calls[0]
    assert call["tools"] == [_TOOL], "tools reach the client as the direct call carried them"
    assert call["options"] == {"temperature": 0.0}, "the options carry no tools key of their own"
    assert "format" not in call
    assert reply.tool_calls == [{"name": "search", "arguments": {"q": "x"}}]
    assert reply.to_dict()["message"]["tool_calls"] == raw
    assert reply.to_dict()["message"]["tool_calls"][0] is raw[0], "the client's own objects, for the parser that reads them"
    assert reply.to_dict()["message"]["content"] == ""
    dict_client = _Client(reply={"message": {"content": "", "tool_calls": [{"function": {"name": "search", "arguments": "{\"q\": 1}"}}]}})
    reply = ScriptedBackend(dict_client).generate("m", _MSGS, options={"tools": [_TOOL]})
    assert reply.tool_calls == [{"name": "search", "arguments": {"q": 1}}], "a JSON-string argument is decoded"
    plain = _Client(reply=SimpleNamespace(message=SimpleNamespace(content="hi")))
    reply = ScriptedBackend(plain).generate("m", _MSGS, options={"temperature": 0.3})
    assert "tools" not in plain.calls[0], "no tools, no tools kwarg: suites branch on its presence"
    assert reply.tool_calls == [] and reply.content == "hi"
    assert reply.to_dict() == {"message": {"content": "hi", "tool_calls": []}}


# ---------------------------------------------------------------------------
# RB2 -- a schema travels as format
# ---------------------------------------------------------------------------
def test_rb2_a_schema_travels_as_format_and_only_then():
    schema = {"type": "object", "properties": {"tool_name": {"enum": ["search"]}}}
    client = _Client(reply={"message": {"content": "{\"tool_name\": \"search\"}"}})
    reply = ScriptedBackend(client).generate("m", _MSGS, options={"temperature": 0.0, "schema": schema})
    call = client.calls[0]
    assert call["format"] == schema
    assert "schema" not in (call["options"] or {})
    assert "tools" not in call
    assert reply.content == "{\"tool_name\": \"search\"}"
    ScriptedBackend(client).generate("m", _MSGS, options={"temperature": 0.0})
    assert "format" not in client.calls[1]


# ---------------------------------------------------------------------------
# RB3 -- a stream is a stream
# ---------------------------------------------------------------------------
def test_rb3_a_stream_is_asked_as_a_stream_and_read_as_chunks():
    client = _Client(chunks=[{"message": {"content": "fin"}}, {"message": {"content": "al"}, "done": True}])
    chunks = list(ScriptedBackend(client).stream("m", _MSGS, options={"temperature": 0.3}))
    assert client.calls[0]["stream"] is True
    assert "tools" not in client.calls[0]
    assert [c.content for c in chunks] == ["fin", "al"]
    assert [c.done for c in chunks] == [False, True]
    assert all(c.model == "m" for c in chunks)


# ---------------------------------------------------------------------------
# RB4 -- the seeded registry behaves like the real one
# ---------------------------------------------------------------------------
def test_rb4_the_seeded_registry_registers_activates_and_unregisters():
    seeded = {}
    client = _Client(reply={"message": {"content": "a"}})
    first = seed_registry(seeded, client)
    registry = seeded["opti_oignon.inference_backend"].get_backend_registry()
    assert registry.resolve_backend("any") is first and registry.active is first
    assert registry.active_name == first.name and registry.get(first.name) is first
    other = ScriptedBackend(_Client(reply={"message": {"content": "b"}}))
    other.name = "scripted-eval"
    registry.register(other)
    assert registry.get("scripted-eval") is other and registry.resolve_backend("any") is first, "registering does not activate"
    assert registry.activate("scripted-eval") is True
    assert registry.resolve_backend("any") is other and registry.active_name == "scripted-eval"
    assert registry.activate("nobody") is False
    assert registry.unregister("scripted-eval") is True and registry.get("scripted-eval") is None
    assert registry.unregister("scripted-eval") is False
    assert registry.active is first, "with the overlay gone, the first backend serves again"
    assert sorted(b.name for b in registry.backends()) == [first.name]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
