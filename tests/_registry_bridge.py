#!/usr/bin/env python3
"""A registry stand-in that routes to a scripted client, for hub suites.

Several executor suites drive the hub with a scripted ``ollama`` module and
assert on the messages that reach it -- memory envelopes, ledger records, a
stable prompt head. Those are properties of prompt assembly, not of
transport. They were written when the hub, finding no registry in its
window, fell through to a direct client call; the hub refuses that now, so a
suite that seeds only the client sees its request go nowhere.

This bridge keeps every one of those assertions byte-identical: it seeds a
registry whose single backend forwards each request to the suite's scripted
client with the kwargs the direct call used to carry, and turns the client's
chunk dicts back into the stream objects the hub reads. The request still
reaches the client -- through the registry, which is the point.

Import-safe and stdlib-only. Used by contract suites; never by the package.
"""

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


class _Reply:
    """The shape the hub reads from a backend generate."""

    def __init__(self, content, thinking=None, model=""):
        self.content = content
        self.thinking = thinking
        self.model = model
        self.tool_calls = []


def _field(chunk, name):
    msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
    value = msg.get(name, "") if isinstance(msg, dict) else getattr(msg, name, "")
    return value or ""


class ScriptedBackend:
    """One backend, forwarding to a scripted client's ``chat``.

    The kwargs handed to the client are the ones the hub's former direct
    call carried, so an assertion on ``scripted.calls[-1]["messages"]`` or
    ``["options"]`` reads exactly what it always read. ``images`` is accepted
    and not embedded: no suite on this bridge asserts vision payloads.
    """

    name = "ollama"

    def __init__(self, scripted):
        self._scripted = scripted

    def health_check(self):
        return True

    def model_info(self, model):
        return {"name": model}

    def slots(self):
        return []

    def stream(self, model, messages, options=None, keep_alive="30m",
               think=False, images=None):
        kwargs = dict(
            model=model, messages=messages, options=options,
            stream=True, keep_alive=keep_alive,
        )
        if think:
            kwargs["think"] = True
        for chunk in self._scripted.chat(**kwargs):
            done = bool(chunk.get("done", False)) if isinstance(chunk, dict) else False
            yield _Chunk(
                content=_field(chunk, "content"),
                thinking=_field(chunk, "thinking"),
                done=done,
                model=model,
            )

    def generate(self, model, messages, options=None, keep_alive="30m",
                 think=False, images=None):
        kwargs = dict(
            model=model, messages=messages, options=options,
            keep_alive=keep_alive,
        )
        if think:
            kwargs["think"] = True
        reply = self._scripted.chat(**kwargs)
        if isinstance(reply, (dict,)) or hasattr(reply, "message"):
            return _Reply(_field(reply, "content"), _field(reply, "thinking") or None, model)
        # A scripted client that only knows how to stream: join the chunks.
        content = "".join(_field(c, "content") for c in reply)
        return _Reply(content, None, model)


def seed_registry(seeded, scripted):
    """Seed ``opti_oignon.inference_backend`` with a registry over ``scripted``.

    Mutates ``seeded`` in place and returns the backend, so a suite can hand
    the mapping to the isolation window unchanged otherwise.
    """
    backend = ScriptedBackend(scripted)
    registry = SimpleNamespace(
        active=backend,
        active_name=backend.name,
        resolve_backend=lambda model: backend,
        backends=lambda: [backend],
        get=lambda name: backend if name == backend.name else None,
    )
    module = types.ModuleType("opti_oignon.inference_backend")
    module.get_backend_registry = lambda: registry
    seeded["opti_oignon.inference_backend"] = module
    return backend
