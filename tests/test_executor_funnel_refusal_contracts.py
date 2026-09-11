#!/usr/bin/env python3
"""Contracts that the funnel refuses when the registry has nothing, rather
than reaching for the client behind its back.

The executor asked the registry for a backend and, when the registry answered
None, fell through to a direct client call at three heads, each annotated
"Fallback: direct ollama call". The registry answers None only when no backend
is registered at all -- and the registry registers the Ollama backend the
moment the client library imports. So the fallback could run only when the
client library was absent, in which state the direct call fails too. A path
that can succeed only in a state where it fails is not a fallback; it is a
bypass with no reachable success, and it took every guarantee the registry
carries with it when it ran.

  * EF1 -- refine_question with no backend refuses, naming the registry, and
    never reaches the client.
  * EF2 -- execute_simple refuses the same way.
  * EF3 -- execute, the streaming head, refuses the same way.

The client stub in this file records any call and then raises, so a bypass
that fired would show up twice: as a recorded call and as an error that does
not name the registry.

Local-only (the public distribution ships no tests). Loaded through the shared
isolation window; the dependency modules are seeded and no client exists.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_EXECUTOR = "opti_oignon.executor"
_OPTIMIZER = "opti_oignon.context_optimizer"


class _ForbiddenClient(types.ModuleType):
    """A client that records the attempt and refuses to serve it."""

    def __init__(self):
        super().__init__("ollama")
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("direct client call")


def _load():
    registry = SimpleNamespace(active=None, resolve_backend=lambda model: None)
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: "test-model:1b",
        get_temperature=lambda *a, **k: 0.2,
    )
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    backends = types.ModuleType("opti_oignon.inference_backend")
    backends.get_backend_registry = lambda: registry

    client = _ForbiddenClient()
    had = "ollama" in sys.modules
    prev = sys.modules.get("ollama")
    sys.modules["ollama"] = client
    loaded, win_restore = isolate(
        targets={
            _OPTIMIZER: source("context_optimizer.py"),
            "opti_oignon.context_dedup": source("context_dedup.py"),
            _EXECUTOR: source("executor.py"),
        },
        seeded={
            "opti_oignon.config": cfg,
            "opti_oignon.router": router,
            "opti_oignon.inference_backend": backends,
        },
        packages=("opti_oignon",),
    )
    loaded[_OPTIMIZER].init_optimizer(
        config={"enabled": False, "stable_prefix": {"enabled": False}}
    )

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return loaded[_EXECUTOR], client, restore


def _names_the_registry(text):
    return "backend" in text.lower() and "registry" in text.lower()


# ---------------------------------------------------------------------------
# EF1 -- refine_question refuses instead of bypassing
# ---------------------------------------------------------------------------
def test_ef1_refine_question_refuses_without_a_backend():
    mod, client, restore = _load()
    try:
        refined, err = mod.Executor().refine_question("what is x", model="m")
        assert client.calls == [], (
            "the client was never reached: a registry with no backend is a "
            "refusal, not a reason to go around the registry"
        )
        assert refined == "what is x", "the question comes back untouched"
        assert err and _names_the_registry(err), (
            f"the error names the registry as the thing that has no backend, "
            f"got {err!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# EF2 -- execute_simple refuses the same way
# ---------------------------------------------------------------------------
def test_ef2_execute_simple_refuses_without_a_backend():
    mod, client, restore = _load()
    try:
        out = mod.Executor().execute_simple("q", "m", "system")
        assert client.calls == [], "the client was never reached"
        assert out.startswith("Error:") and _names_the_registry(out), (
            f"the error names the registry, got {out!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# EF3 -- the streaming head refuses the same way
# ---------------------------------------------------------------------------
def test_ef3_execute_refuses_without_a_backend():
    mod, client, restore = _load()
    try:
        routing = SimpleNamespace(
            model="m", task_type="general", temperature=0.2,
            prompt_variant="standard", timeout=30,
        )
        gen = mod.Executor().execute("q", routing, refine=False)
        streamed = []
        result = None
        try:
            while True:
                streamed.append(next(gen))
        except StopIteration as stop:
            result = stop.value
        assert client.calls == [], "the client was never reached"
        text = " ".join(str(s) for s in streamed) + " " + str(result)
        assert _names_the_registry(text), (
            f"the refusal names the registry somewhere in what the caller "
            f"receives, got streamed={streamed!r} result={result!r}"
        )
    finally:
        restore()
