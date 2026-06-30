#!/usr/bin/env python3
"""BR -- per-model backend routing: contracts for ``BackendRegistry.resolve_backend``.

``resolve_backend(model)`` lets one registry serve Ollama, llama.cpp, and an
external llama-server concurrently, choosing the backend from the requested model
instead of always using ``.active``. It health-gates a probe of each backend's
``model_info(model)``, prefers the active backend when it recognises the model
(stability), otherwise returns the first healthy recogniser, and falls back to
``.active`` when nobody recognises the model (so single-backend deployments are
unchanged). Contracts:

  * BR1 a model recognised by exactly one healthy backend resolves to THAT
    backend, not ``.active``.
  * BR2 a model no backend recognises falls back to ``.active``.
  * BR3 an unhealthy backend is health-gated out even when it recognises the
    model (the load is never routed to a dead backend).
  * BR4 when the active backend recognises the model it is preferred (no needless
    switch between backends that can both serve it).
  * BR5 (control) a single-backend registry resolves to that backend whether or
    not the model is "recognised" -- backward compatibility.

``inference_backend.py`` imports with stdlib-only top-level imports (the Ollama /
llama.cpp SDKs load lazily inside methods), so the registry is exercised in
isolation through ``spec_from_file_location`` with fake ``InferenceBackend``
subclasses whose ``health_check`` and ``model_info`` are fully controllable. No
network, no filesystem model probe, no real backend.

Local-only. Runs under pytest or the __main__ runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_IB_SRC = _REPO / "opti_oignon" / "inference_backend.py"


def _load_ib():
    """Load inference_backend.py in isolation (idempotent)."""
    cached = sys.modules.get("opti_oignon.inference_backend")
    if cached is not None:
        return cached
    if not isinstance(sys.modules.get("opti_oignon"), types.ModuleType) or (
        getattr(sys.modules.get("opti_oignon"), "__file__", "x") is not None
    ):
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(_REPO / "opti_oignon")]
        sys.modules["opti_oignon"] = pkg
    spec = importlib.util.spec_from_file_location(
        "opti_oignon.inference_backend", _IB_SRC
    )
    ib = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.inference_backend"] = ib
    spec.loader.exec_module(ib)
    return ib


def _fake_backend_cls(ib):
    """A minimal concrete InferenceBackend with controllable health/recognition."""

    class _FakeBackend(ib.InferenceBackend):
        def __init__(self, name, healthy=True, recognizes=()):
            self._name = name
            self._healthy = healthy
            self._recognizes = set(recognizes)

        @property
        def name(self):
            return self._name

        @property
        def display_name(self):
            return self._name

        def health_check(self):
            return self._healthy

        def list_models(self):
            return []

        def model_info(self, model_name):
            if model_name in self._recognizes:
                return ib.BackendModelInfo(name=model_name, backend=self._name)
            return None

        def generate(self, *args, **kwargs):
            raise NotImplementedError

        def stream(self, *args, **kwargs):
            raise NotImplementedError

    return _FakeBackend


def test_br1_recognizer_wins_over_active():
    """BR1 -- a model recognised by exactly one healthy backend resolves to that
    backend even though the active backend is a different one."""
    ib = _load_ib()
    Fake = _fake_backend_cls(ib)
    reg = ib.BackendRegistry()
    ollama = Fake("ollama", recognizes={"llama3"})
    llamacpp = Fake("llamacpp", recognizes={"mistral.gguf"})
    reg.register(ollama)
    reg.register(llamacpp)
    reg.activate("llamacpp")  # active does NOT recognise llama3
    assert reg.resolve_backend("llama3") is ollama


def test_br2_unknown_model_falls_back_to_active():
    """BR2 -- a model no backend recognises falls back to the active backend."""
    ib = _load_ib()
    Fake = _fake_backend_cls(ib)
    reg = ib.BackendRegistry()
    reg.register(Fake("ollama", recognizes={"llama3"}))
    reg.register(Fake("llamacpp", recognizes={"mistral.gguf"}))
    reg.activate("llamacpp")
    assert reg.resolve_backend("unknown-model") is reg.get("llamacpp")


def test_br3_unhealthy_recognizer_is_health_gated():
    """BR3 -- an unhealthy backend that recognises the model is skipped; the load
    is never routed to a dead backend (it falls back to the healthy active)."""
    ib = _load_ib()
    Fake = _fake_backend_cls(ib)
    reg = ib.BackendRegistry()
    sick = Fake("ollama", healthy=False, recognizes={"m"})  # recognises but dead
    other = Fake("llamacpp", healthy=True, recognizes=())  # healthy, not a recogniser
    reg.register(sick)
    reg.register(other)  # no active set -> active = first healthy = other
    resolved = reg.resolve_backend("m")
    assert resolved is not sick  # never the dead backend
    assert resolved is other  # the healthy fallback


def test_br4_active_recognizer_is_preferred():
    """BR4 -- when the active backend also recognises the model it is preferred
    over another recogniser (no needless switch)."""
    ib = _load_ib()
    Fake = _fake_backend_cls(ib)
    reg = ib.BackendRegistry()
    first = Fake("llamacpp", recognizes={"m"})  # registered first
    active = Fake("ollama", recognizes={"m"})  # also recognises
    reg.register(first)
    reg.register(active)
    reg.activate("ollama")  # both recognise; active is ollama
    assert reg.resolve_backend("m") is active


def test_br5_single_backend_unchanged():
    """BR5 (control) -- a single-backend registry resolves to that backend whether
    or not the model is recognised (backward compatibility)."""
    ib = _load_ib()
    Fake = _fake_backend_cls(ib)
    reg = ib.BackendRegistry()
    only = Fake("ollama", recognizes={"x"})
    reg.register(only)
    reg.activate("ollama")
    assert reg.resolve_backend("x") is only  # recognised
    assert reg.resolve_backend("y") is only  # unrecognised -> active fallback


if __name__ == "__main__":
    import traceback

    tests = [
        v
        for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)
