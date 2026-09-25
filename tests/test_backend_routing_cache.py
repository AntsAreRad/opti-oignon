#!/usr/bin/env python3
"""BR fast-follow -- the resolve_backend resolution cache.

``BackendRegistry.resolve_backend`` caches ``model -> backend name`` so the
per-call ``model_info`` probe (Ollama network / llama.cpp filesystem /
llama-server HTTP) is paid once per model on the hot path. Additive companion
(the S283 ``test_backend_routing.py`` is left byte-untouched; BR1-BR5 still hold
unchanged because the cache is transparent to first-call routing). Contracts:

  * CACHE1 a second resolve of the same model is a cache hit -- ``model_info`` is
    NOT probed again.
  * CACHE2 ``register`` clears the cache -- a topology change forces re-probing
    (a new backend may change routing).
  * CACHE3 a cache hit re-runs ``health_check`` -- a backend that became
    unhealthy since it was cached is never served from the cache; resolution
    falls through to a healthy backend.
  * CACHE4 ``unregister`` clears the cache -- same topology-change invariant.

Isolation reuses the S283 idiom: ``inference_backend.py`` imports with
stdlib-only top-level imports, so the registry is exercised with a fake
``InferenceBackend`` whose ``health_check`` is flippable and whose ``model_info``
counts its calls (the cache observable). No network, no filesystem, no real
backend. The module is loaded from its source through the shared window
(``tests/_isolation.py``), one window per contract, closed after it.

Local-only. Runs under pytest or the __main__ runner.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"


@pytest.fixture(autouse=True)
def _package_left_as_found():
    """Every contract here loads the backend module; none may leave a stand-in package behind."""
    before = sys.modules.get("opti_oignon")
    yield
    assert sys.modules.get("opti_oignon") is before, "the package entry is left as the contract found it"


def _open():
    """The backend module loaded from its source in the shared window, and its closer."""
    loaded, restore = isolate(targets={_BACKEND: source("inference_backend.py")}, packages=("opti_oignon",))
    return loaded[_BACKEND], restore


def _counting_backend_cls(ib):
    """A fake backend that counts model_info probes and has a flippable health."""

    class _CountingBackend(ib.InferenceBackend):
        def __init__(self, name, healthy=True, recognizes=()):
            self._name = name
            self.healthy = healthy  # flippable
            self._recognizes = set(recognizes)
            self.model_info_calls = 0

        @property
        def name(self):
            return self._name

        @property
        def display_name(self):
            return self._name

        def health_check(self):
            return self.healthy

        def list_models(self):
            return []

        def model_info(self, model_name):
            self.model_info_calls += 1
            if model_name in self._recognizes:
                return ib.BackendModelInfo(name=model_name, backend=self._name)
            return None

        def generate(self, *args, **kwargs):
            raise NotImplementedError

        def stream(self, *args, **kwargs):
            raise NotImplementedError

    return _CountingBackend


def test_cache1_second_resolve_is_a_cache_hit():
    """CACHE1 -- resolving the same model twice probes model_info only once; the
    second call is served from the cache."""
    ib, restore = _open()
    try:
        Counting = _counting_backend_cls(ib)
        reg = ib.BackendRegistry()
        a = Counting("ollama", recognizes={"m"})
        reg.register(a)
        reg.resolve_backend("m")
        probes_after_first = a.model_info_calls
        reg.resolve_backend("m")  # cache hit: no new probe
        assert a.model_info_calls == probes_after_first
    finally:
        restore()


def test_cache2_register_clears_cache():
    """CACHE2 -- registering a backend clears the cache, so the next resolve of a
    previously-cached model re-probes (the new backend may change routing)."""
    ib, restore = _open()
    try:
        Counting = _counting_backend_cls(ib)
        reg = ib.BackendRegistry()
        a = Counting("ollama", recognizes={"m"})
        reg.register(a)
        reg.resolve_backend("m")
        probes_after_first = a.model_info_calls
        reg.register(Counting("llamacpp", recognizes={"m"}))  # topology change
        reg.resolve_backend("m")
        assert a.model_info_calls > probes_after_first  # re-probed after register
    finally:
        restore()


def test_cache3_hit_rechecks_health():
    """CACHE3 -- a cache hit re-runs health_check; a backend that became unhealthy
    since being cached is not served, resolution falls through to a healthy one."""
    ib, restore = _open()
    try:
        Counting = _counting_backend_cls(ib)
        reg = ib.BackendRegistry()
        a = Counting("ollama", recognizes={"m"})
        b = Counting("llamacpp", recognizes={"m"})
        reg.register(a)
        reg.register(b)
        reg.activate("ollama")  # active recognises -> first resolve caches m -> a
        assert reg.resolve_backend("m") is a
        a.healthy = False  # a dies after being cached
        resolved = reg.resolve_backend("m")
        assert resolved is not a  # never served the dead cached backend
        assert resolved is b  # fell through to the healthy recogniser
    finally:
        restore()


def test_cache4_unregister_clears_cache():
    """CACHE4 -- unregistering a backend clears the cache, so a previously-cached
    model re-probes on the next resolve."""
    ib, restore = _open()
    try:
        Counting = _counting_backend_cls(ib)
        reg = ib.BackendRegistry()
        a = Counting("ollama", recognizes={"m"})
        b = Counting("llamacpp", recognizes=())
        reg.register(a)
        reg.register(b)
        reg.resolve_backend("m")  # caches m -> a
        probes_after_first = a.model_info_calls
        reg.unregister("llamacpp")  # topology change
        reg.resolve_backend("m")
        assert a.model_info_calls > probes_after_first  # re-probed after unregister
    finally:
        restore()


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
