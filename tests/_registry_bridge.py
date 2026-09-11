#!/usr/bin/env python3
"""A registry stand-in that routes to a scripted client, for hub and tool suites.

Several executor suites drive the hub with a scripted ``ollama`` module and
assert on the messages that reach it -- memory envelopes, ledger records, a
stable prompt head. Six tool suites do the same with the tool executor and
assert on what it sends -- tool schemas, messages, a stream -- and read back
scripted tool calls. Those are properties of prompt assembly and dispatch,
not of transport. They were written when both modules called the client
directly; both refuse that now, so a suite that seeds only the client sees
its request go nowhere.

This bridge keeps every one of those assertions byte-identical: it seeds a
registry whose backend forwards each request to the suite's scripted
client with the kwargs the direct call used to carry -- ``tools=`` when the
options carry tools, ``format=`` when they carry a schema, and neither
otherwise, because suites branch on their presence -- and carries the
client's reply back in the shape the registry promises: ``content``,
normalised ``tool_calls``, and a ``to_dict()`` that keeps the client's own
objects for the parser that reads them. The request still reaches the
client -- through the registry, which is the point.

The registry it seeds registers, activates and unregisters like the real
one, so a production seam (the eval harness) can lay a backend over it for
a block and take it away again.

Import-safe and stdlib-only. Used by contract suites; never by the package.
"""

import json
import types
from types import SimpleNamespace


class _Chunk:
    """The shape the hub reads from a backend stream."""

    __slots__ = ("content", "thinking", "done", "model")

    def __init__(self, content="", thinking="", done=False, model=""):
        self.content = content
        self.thinking = thinking
        self.done = done
        self.model = model


def _normalise(raw_calls):
    out = []
    for call in raw_calls or []:
        fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
        if fn is None:
            continue
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        if not name:
            continue
        args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                args = {}
        out.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
    return out


class _Reply:
    """The shape the hub and the tool executor read from a backend generate."""

    def __init__(self, content, thinking=None, model="", raw_calls=None):
        self.content = content
        self.thinking = thinking
        self.model = model
        self.raw_calls = list(raw_calls or [])
        self.tool_calls = _normalise(self.raw_calls)

    def to_dict(self):
        return {"message": {"content": self.content, "tool_calls": self.raw_calls}}


def _message(reply):
    return reply.get("message", {}) if isinstance(reply, dict) else getattr(reply, "message", {})


def _field(chunk, name):
    msg = _message(chunk)
    value = msg.get(name, "") if isinstance(msg, dict) else getattr(msg, name, "")
    return value or ""


def _raw_calls(reply):
    msg = _message(reply)
    value = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
    return list(value or [])


class ScriptedBackend:
    """One backend, forwarding to a scripted client's ``chat``.

    The kwargs handed to the client are the ones the former direct call
    carried, so an assertion on ``scripted.calls[-1]["messages"]``,
    ``["options"]`` or ``"tools" in kw`` reads exactly what it always read.
    ``images`` is accepted and not embedded: no suite on this bridge asserts
    vision payloads.
    """

    name = "ollama"
    display_name = "scripted client (bridge)"

    def __init__(self, scripted):
        self._scripted = scripted

    def health_check(self):
        return True

    def model_info(self, model):
        return {"name": model}

    def slots(self):
        return []

    @staticmethod
    def _kwargs(model, messages, options, keep_alive, think, stream):
        opts = None if options is None else dict(options)
        tools = opts.pop("tools", None) if opts else None
        schema = opts.pop("schema", None) if opts else None
        kwargs = dict(model=model, messages=messages, options=opts, keep_alive=keep_alive)
        if stream:
            kwargs["stream"] = True
        if tools is not None:
            kwargs["tools"] = tools
        if schema is not None:
            kwargs["format"] = schema
        if think:
            kwargs["think"] = True
        return kwargs

    def stream(self, model, messages, options=None, keep_alive="30m",
               think=False, images=None):
        kwargs = self._kwargs(model, messages, options, keep_alive, think, stream=True)
        for chunk in self._scripted.chat(**kwargs):
            done = bool(chunk.get("done", False)) if isinstance(chunk, dict) else bool(getattr(chunk, "done", False))
            yield _Chunk(
                content=_field(chunk, "content"),
                thinking=_field(chunk, "thinking"),
                done=done,
                model=model,
            )

    def generate(self, model, messages, options=None, keep_alive="30m",
                 think=False, images=None):
        kwargs = self._kwargs(model, messages, options, keep_alive, think, stream=False)
        reply = self._scripted.chat(**kwargs)
        if isinstance(reply, dict) or hasattr(reply, "message"):
            return _Reply(_field(reply, "content"), _field(reply, "thinking") or None, model,
                          raw_calls=_raw_calls(reply))
        # A scripted client that only knows how to stream: join the chunks.
        content = "".join(_field(c, "content") for c in reply)
        return _Reply(content, None, model)


class StubRegistry:
    """The registry's surface, over the backends a suite hands it."""

    def __init__(self):
        self._backends = {}
        self._active_name = None

    def register(self, backend):
        self._backends[backend.name] = backend

    def unregister(self, name):
        if name not in self._backends:
            return False
        del self._backends[name]
        if self._active_name == name:
            self._active_name = None
        return True

    def get(self, name):
        return self._backends.get(name)

    def backends(self):
        return list(self._backends.values())

    @property
    def active(self):
        if self._active_name and self._active_name in self._backends:
            return self._backends[self._active_name]
        for backend in self._backends.values():
            return backend
        return None

    @property
    def active_name(self):
        return self._active_name

    def activate(self, name):
        if name not in self._backends:
            return False
        self._active_name = name
        return True

    def resolve_backend(self, model):
        return self.active


def seed_registry(seeded, scripted):
    """Seed ``opti_oignon.inference_backend`` with a registry over ``scripted``.

    Mutates ``seeded`` in place and returns the backend, so a suite can hand
    the mapping to the isolation window unchanged otherwise.
    """
    backend = ScriptedBackend(scripted)
    registry = StubRegistry()
    registry.register(backend)
    registry.activate(backend.name)
    module = types.ModuleType("opti_oignon.inference_backend")
    module.get_backend_registry = lambda: registry
    seeded["opti_oignon.inference_backend"] = module
    return backend
