#!/usr/bin/env python3
"""Contracts that the two summarisers ask the registry, not the client.

The conversation compressor and the context summariser are foundations of the
memory block, and the roadmap wants that block's librarian "placed by the
Forge". Both called the client library directly, so the placement, the
admission and the provenance the Forge attached to the registry never reached
either of them. They go through the registry now.

  * OC1 -- the compressor's LLM path sends through the registry's backend
    and reads its summary back.
  * OC2 -- the summariser does the same.
  * OC3 -- with no backend, each degrades the way it documents -- the
    compressor to its rule-based path, the summariser to None -- and the
    client is never reached.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the registry and the client are both stand-ins.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_SUMMARY = (
    "The user asked about placement and the assistant explained that a plan "
    "resolves to a reproducible command."
)


class _RecorderBackend:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=_SUMMARY, model=kwargs.get("model"))


class _ForbiddenClient(types.ModuleType):
    def __init__(self):
        super().__init__("ollama")
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("direct client call")


def _load(module, *, backend):
    registry = SimpleNamespace(
        active=backend, resolve_backend=lambda model: backend,
    )
    backends = types.ModuleType("opti_oignon.inference_backend")
    backends.get_backend_registry = lambda: registry
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: "stand-in",
        get_temperature=lambda *a, **k: 0.3,
        get=lambda *a, **k: None,
    )
    client = _ForbiddenClient()
    had = "ollama" in sys.modules
    prev = sys.modules.get("ollama")
    sys.modules["ollama"] = client
    loaded, win_restore = isolate(
        targets={f"opti_oignon.{module}": source(f"{module}.py")},
        seeded={
            "opti_oignon.inference_backend": backends,
            "opti_oignon.config": cfg,
        },
        packages=("opti_oignon",),
    )

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return loaded[f"opti_oignon.{module}"], client, restore


def _class_with(mod, method):
    """The class in the module that owns the method under contract."""
    for value in vars(mod).values():
        if isinstance(value, type) and method in vars(value):
            return value
    raise AssertionError(f"no class in {mod.__name__} defines {method}")


_MESSAGES = [
    {"role": "user", "content": "how is a model placed on the card"},
    {"role": "assistant", "content": "a plan resolves to a reproducible command"},
    {"role": "user", "content": "and the experts"},
    {"role": "assistant", "content": "they can be routed to system memory"},
]


# ---------------------------------------------------------------------------
# OC1 -- the compressor asks the registry
# ---------------------------------------------------------------------------
def test_oc1_the_compressor_summarises_through_the_registry():
    backend = _RecorderBackend()
    mod, client, restore = _load("conversation_compressor", backend=backend)
    try:
        compressor = _class_with(mod, "_compress_llm")()
        summary, mode = compressor._compress_llm(_MESSAGES, "stand-in")
        assert client.calls == [], "the client was never reached"
        assert len(backend.calls) == 1, "one request went through the registry"
        call = backend.calls[0]
        assert call["model"] == "stand-in"
        assert any(m["role"] == "user" for m in call["messages"]), (
            "the excerpt to summarise travels as the user message"
        )
        assert mode == "llm", "and the result is credited to the LLM path"
        assert _SUMMARY in summary, "carrying the backend's summary"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OC2 -- the summariser asks the registry
# ---------------------------------------------------------------------------
def test_oc2_the_summariser_summarises_through_the_registry():
    backend = _RecorderBackend()
    mod, client, restore = _load("context_summary", backend=backend)
    try:
        summariser = _class_with(mod, "summarize_messages")()
        out = summariser.summarize_messages(_MESSAGES, model="stand-in")
        assert client.calls == [], "the client was never reached"
        assert len(backend.calls) == 1, "one request went through the registry"
        assert backend.calls[0]["model"] == "stand-in"
        assert out and _SUMMARY.split(".")[0] in out, (
            "the summary the backend returned is the one handed back"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# OC3 -- with no backend, each degrades as documented, client untouched
# ---------------------------------------------------------------------------
def test_oc3_without_a_backend_each_degrades_as_documented():
    mod, client, restore = _load("conversation_compressor", backend=None)
    try:
        compressor = _class_with(mod, "_compress_llm")()
        _summary, mode = compressor._compress_llm(_MESSAGES, "stand-in")
        assert client.calls == [], "the client was never reached"
        assert mode == "rule_fallback", (
            "no backend means the rule-based path, which needs no model"
        )
    finally:
        restore()

    mod, client, restore = _load("context_summary", backend=None)
    try:
        summariser = _class_with(mod, "summarize_messages")()
        out = summariser.summarize_messages(_MESSAGES, model="stand-in")
        assert client.calls == [], "the client was never reached"
        assert out is None, "no backend means no summary, as documented"
    finally:
        restore()
