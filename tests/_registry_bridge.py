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

    __slots__ = ("content", "thinking", "done", "model", "extra")

    def __init__(self, content="", thinking="", done=False, model="", extra=None):
        self.content = content
        self.thinking = thinking
        self.done = done
        self.model = model
        self.extra = extra or {}


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

    def __init__(self, content, thinking=None, model="", raw_calls=None, extra=None):
        self.content = content
        self.thinking = thinking
        self.model = model
        self.raw_calls = list(raw_calls or [])
        self.tool_calls = _normalise(self.raw_calls)
        self.extra = extra or {}

    def to_dict(self):
        return {"message": {"content": self.content, "tool_calls": self.raw_calls}}


def _message(reply):
    return reply.get("message", {}) if isinstance(reply, dict) else getattr(reply, "message", {})


def _field(chunk, name):
    msg = _message(chunk)
    value = msg.get(name, "") if isinstance(msg, dict) else getattr(msg, name, "")
    return value or ""


def _entry_field(entry, name):
    """A top-level field of a client entry, mapping or object; ``None`` when absent."""
    return entry.get(name) if isinstance(entry, dict) else getattr(entry, name, None)


def _reported(entry):
    """The counts a client reports on a reply or chunk, when it reports them."""
    out = {}
    for key in ("eval_count", "prompt_eval_count", "total_duration", "eval_duration"):
        value = _entry_field(entry, key)
        if value is not None:
            out[key] = value
    return out


def _raw_calls(reply):
    msg = _message(reply)
    value = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
    return list(value or [])


class ScriptedBackend:
    """One backend, forwarding to a scripted client's ``chat``.

    The kwargs handed to the client are the ones the former direct call
    carried, so an assertion on ``scripted.calls[-1]["messages"]``,
    ``["options"]`` or ``"tools" in kw`` reads exactly what it always read.
    ``images`` is forwarded as given, only when given, so a suite that
    asserts a vision payload reads it; before the vision pipeline went
    through the registry no suite on this bridge did, and the kwargs of a
    request without images are unchanged. It is not embedded: no suite
    vision payloads.
    """

    name = "ollama"
    display_name = "scripted client (bridge)"

    def __init__(self, scripted):
        self._scripted = scripted

    def health_check(self):
        return True

    def model_info(self, model):
        """The scripted client's ``show(model)`` as a record when it has one; ``{"name": model}`` without."""
        show = getattr(self._scripted, "show", None)
        if not callable(show):
            return {"name": model}
        info = show(model)
        details = _entry_field(info, "details") or {}
        mapping = _entry_field(info, "model_info") or _entry_field(info, "modelinfo") or {}
        ctx = None
        for key, value in dict(mapping).items():
            if "context_length" in str(key):
                ctx = int(value)
                break
        extra = {}
        families = _entry_field(details, "families")
        if isinstance(families, (list, tuple)):
            extra["families"] = [str(f) for f in families]
        for key in ("parameters", "digest", "template", "modelfile", "license"):
            value = _entry_field(info, key)
            if value:
                extra[key] = value
        if mapping:
            extra["model_info"] = dict(mapping)
        return SimpleNamespace(
            name=model, backend=self.name, size=None, modified_at=None, path=None,
            family=_entry_field(details, "family"),
            parameter_size=_entry_field(details, "parameter_size"),
            quantization_level=_entry_field(details, "quantization_level"),
            context_length=ctx, extra=extra,
        )

    def slots(self):
        return []

    def loaded_models(self):
        """The scripted client's ``ps()`` as records, ``None`` without one."""
        ps = getattr(self._scripted, "ps", None)
        if not callable(ps):
            return None
        payload = ps()
        models = payload.get("models", []) if isinstance(payload, dict) else getattr(payload, "models", [])
        out = []
        for m in models or []:
            name = _entry_field(m, "name") or _entry_field(m, "model")
            if not name:
                continue
            expires_at = _entry_field(m, "expires_at")
            if expires_at is not None and hasattr(expires_at, "timestamp"):
                expires_at = expires_at.timestamp()
            size_vram = _entry_field(m, "size_vram")
            out.append(SimpleNamespace(
                name=str(name), backend=self.name,
                size_vram=None if size_vram is None else int(size_vram),
                expires_at=expires_at,
                context_length=_entry_field(m, "context_length"),
                digest=_entry_field(m, "digest"),
            ))
        return out

    def embed(self, model, text):
        """The scripted client's ``embed(model=, input=)`` first vector, ``None`` without one."""
        embed = getattr(self._scripted, "embed", None)
        if not callable(embed):
            return None
        result = embed(model=model, input=text)
        vectors = _entry_field(result, "embeddings") or []
        return list(vectors[0]) if vectors else None

    def list_models(self):
        """The scripted client's ``list()`` as named entries; ``None`` without one."""
        listing = getattr(self._scripted, "list", None)
        if not callable(listing):
            return None
        payload = listing()
        models = payload.get("models", []) if isinstance(payload, dict) else getattr(payload, "models", [])
        out = []
        for m in models:
            if isinstance(m, dict):
                name = m.get("name", m.get("model", ""))
            else:
                name = getattr(m, "name", getattr(m, "model", str(m)))
            details = _entry_field(m, "details") or {}
            extra = {}
            size = _entry_field(m, "size")
            if size is not None:
                extra["size_bytes"] = int(size)
            digest = _entry_field(m, "digest")
            if digest:
                extra["digest"] = str(digest)
            families = _entry_field(details, "families")
            if isinstance(families, (list, tuple)):
                extra["families"] = [str(f) for f in families]
            out.append(SimpleNamespace(
                name=name, backend=self.name, path=None,
                size=None if size is None else str(size),
                modified_at=_entry_field(m, "modified_at"),
                family=_entry_field(details, "family"),
                parameter_size=_entry_field(details, "parameter_size"),
                quantization_level=_entry_field(details, "quantization_level"),
                context_length=None, extra=extra,
            ))
        return out

    @staticmethod
    def _kwargs(model, messages, options, keep_alive, think, stream):
        opts = None if options is None else dict(options)
        tools = opts.pop("tools", None) if opts else None
        schema = opts.pop("schema", None) if opts else None
        timeout = opts.pop("timeout", None) if opts else None
        kwargs = dict(model=model, messages=messages, options=opts, keep_alive=keep_alive)
        if stream:
            kwargs["stream"] = True
        if tools is not None:
            kwargs["tools"] = tools
        if schema is not None:
            kwargs["format"] = schema
        if timeout is not None:
            kwargs["timeout"] = timeout
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
                extra=_reported(chunk),
            )

    def generate(self, model, messages, options=None, keep_alive="30m",
                 think=False, images=None):
        kwargs = self._kwargs(model, messages, options, keep_alive, think, stream=False)
        if images:
            kwargs["images"] = list(images)
        reply = self._scripted.chat(**kwargs)
        if isinstance(reply, dict) or hasattr(reply, "message"):
            return _Reply(_field(reply, "content"), _field(reply, "thinking") or None, model,
                          raw_calls=_raw_calls(reply), extra=_reported(reply))
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
