#!/usr/bin/env python3
"""Contracts that a model listing says unknown when nobody could look.

The backend contract already writes the doctrine for its two observation
heads: ``loaded_models`` and ``embed`` answer ``None`` when the backend
cannot say, and an empty list only when it looked and found nothing. The
listing head predates that doctrine and every implementation answered an
empty list for a client that was absent, a request that failed, a daemon
that was down, or a directory that did not exist -- indistinguishable from
a backend that looked and found nothing, which is the silent zero this
repository treats as a defect. This suite pins the listing to the same
doctrine, on every backend, on the registry that aggregates them, on the
shared readers, and on the callers whose behaviour changes when unknown
stops reading as empty.

  * LU1 -- Ollama answers ``None`` with the client absent or the call
    failing, an empty list for an empty listing, and the contract's
    signature says so.
  * LU2 -- llama-server answers ``None`` on a transport error or a body
    that is not a listing, an empty list for an empty one, and its
    ``model_info`` reads an unknown listing as unknown without raising.
  * LU3 -- llama.cpp answers ``None`` when no configured directory exists
    (nobody could scan), an empty list after a real scan of an empty one.
  * LU4 -- the test bridge answers ``None`` without a scripted ``list``.
  * LU5 -- the registry's backend listing reports an unknown count as
    ``None`` rather than zero, and the aggregate listing skips an unknown
    backend without hiding the others.
  * LU6 -- the shared readers propagate unknown: a registered backend that
    cannot say is ``None`` on both, the same answer as no backend.
  * LU7 -- the callers whose answer changes: the extraction falls to its
    first fallback model when the listing is unknown instead of scanning
    nothing, and the health monitor reports the discovery as unknown and
    leaves its records alone instead of checking nothing.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window with every client faked; the transport of the
llama-server backend is replaced through the monkeypatch fixture so the
shared module is restored after each contract.
"""

import json
import sys
import types
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import ScriptedBackend, StubRegistry  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"
_CLIENTS = "opti_oignon.registry_clients"


class _Boom(Exception):
    pass


class _FakeOllama:
    def __init__(self, models=None, fail=False):
        self.calls = []
        self._models = models
        self._fail = fail

    def list(self):
        self.calls.append("list")
        if self._fail:
            raise _Boom("list down")
        return {"models": list(self._models or [])}


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transport:
    """A llama-server transport: a body per path, or a failure by name."""

    def __init__(self, body=None, fail=False):
        self.requests = []
        self._body = body
        self._fail = fail

    def urlopen(self, req, timeout=None):
        self.requests.append(req.full_url)
        if self._fail:
            raise urllib.error.URLError("server down")
        return _FakeResponse(json.dumps(self._body).encode("utf-8"))


def _open(seeded=None):
    loaded, restore = isolate(
        targets={_BACKEND: source("inference_backend.py")},
        seeded=seeded,
        packages=("opti_oignon",),
    )
    return loaded[_BACKEND], restore


def _ollama(mod, fake, available=True):
    mod.OLLAMA_AVAILABLE = available
    mod._ollama_module = fake if available else None
    return mod.OllamaBackend()


class _Listing:
    """A registered backend that answers whatever listing it was given."""

    name = "listing"
    display_name = "listing"

    def __init__(self, listing):
        self._listing = listing

    def health_check(self):
        return True

    def list_models(self):
        return self._listing

    def model_info(self, model):
        return None

    def generate(self, *a, **k):
        raise AssertionError("not asked")

    def stream(self, *a, **k):
        raise AssertionError("not asked")


class _Silent:
    """A scripted client with a ``list`` that cannot answer."""

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return {"message": {"content": "answer"}}


def _registry_over(backend):
    registry = StubRegistry()
    registry.register(backend)
    registry.activate(backend.name)
    module = types.ModuleType(_BACKEND)
    module.get_backend_registry = lambda: registry
    return module


def _unknown_window(rel, name, *, with_clients=True, packages=("opti_oignon",)):
    """Load ``rel`` over a registry whose only backend answers an unknown listing."""
    class _Unknown(ScriptedBackend):
        def list_models(self):
            return None

    seeds = {_BACKEND: _registry_over(_Unknown(_Silent()))}
    targets = {}
    if with_clients:
        targets[_CLIENTS] = source("registry_clients.py")
    targets[name] = source(*rel.split("/"))
    loaded, restore = isolate(targets=targets, seeded=seeds, packages=packages)
    return loaded[name], restore


# ---------------------------------------------------------------------------
# LU1 -- Ollama
# ---------------------------------------------------------------------------
def test_lu1_ollama_answers_unknown_when_the_client_is_absent_or_fails_and_empty_only_when_it_looked():
    mod, restore = _open()
    try:
        annotation = mod.InferenceBackend.list_models.__annotations__.get("return")
        assert "None" in str(annotation), "the contract's signature carries the unknown answer"
        assert _ollama(mod, _FakeOllama(), available=False).list_models() is None, (
            "an absent client is unknown, not an empty listing"
        )
        failing = _FakeOllama(fail=True)
        assert _ollama(mod, failing).list_models() is None, "a failed call is unknown"
        assert failing.calls == ["list"]
        empty = _FakeOllama(models=[])
        assert _ollama(mod, empty).list_models() == [], "a backend that looked and found nothing"
        assert [m.name for m in _ollama(mod, _FakeOllama(models=[{"model": "m"}])).list_models()] == ["m"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# LU2 -- llama-server
# ---------------------------------------------------------------------------
def test_lu2_llama_server_answers_unknown_on_a_transport_error_or_a_non_listing_body(monkeypatch):
    mod, restore = _open()
    try:
        backend = mod.LlamaServerBackend(host="http://fake:8080")
        monkeypatch.setattr(mod.urllib.request, "urlopen", _Transport(fail=True).urlopen)
        assert backend.list_models() is None, "an unreachable server is unknown"
        assert backend.model_info("served.gguf") is None, "an unknown listing describes nothing, and does not raise"
        monkeypatch.setattr(mod.urllib.request, "urlopen", _Transport(body={"object": "list"}).urlopen)
        assert backend.list_models() is None, "a body without a listing is unknown"
        monkeypatch.setattr(mod.urllib.request, "urlopen", _Transport(body={"data": []}).urlopen)
        assert backend.list_models() == [], "a listing with nothing in it is a known empty"
        monkeypatch.setattr(mod.urllib.request, "urlopen", _Transport(body={"data": [{"id": "served.gguf"}]}).urlopen)
        assert [m.name for m in backend.list_models()] == ["served.gguf"]
        assert backend.model_info("served.gguf").name == "served.gguf"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LU3 -- llama.cpp
# ---------------------------------------------------------------------------
def test_lu3_llama_cpp_answers_unknown_without_a_directory_to_scan_and_empty_after_a_real_scan(tmp_path):
    mod, restore = _open()
    try:
        assert mod.LlamaCppBackend(model_dirs=[]).list_models() is None, "nothing configured, nobody could scan"
        missing = tmp_path / "absent"
        assert mod.LlamaCppBackend(model_dirs=[str(missing)]).list_models() is None, "no configured directory exists"
        empty = tmp_path / "empty"
        empty.mkdir()
        assert mod.LlamaCppBackend(model_dirs=[str(missing), str(empty)]).list_models() == [], (
            "one directory was scanned and held nothing"
        )
        (empty / "tiny-q4_k_m.gguf").write_bytes(b"GGUF")
        assert [m.name for m in mod.LlamaCppBackend(model_dirs=[str(empty)]).list_models()] == ["tiny-q4_k_m.gguf"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# LU4 -- the bridge
# ---------------------------------------------------------------------------
def test_lu4_the_bridge_answers_unknown_without_a_scripted_list():
    assert ScriptedBackend(_Silent()).list_models() is None
    scripted = SimpleNamespace(chat=lambda **k: {"message": {"content": ""}}, list=lambda: {"models": []})
    assert ScriptedBackend(scripted).list_models() == []


# ---------------------------------------------------------------------------
# LU5 -- the registry
# ---------------------------------------------------------------------------
def test_lu5_the_registry_reports_an_unknown_count_as_none_and_skips_an_unknown_backend_in_the_aggregate():
    mod, restore = _open()
    try:
        registry = mod.BackendRegistry()
        unknown = _Listing(None)
        known = _Listing([mod.BackendModelInfo(name="m", backend="known")])
        known.name = known.display_name = "known"
        registry.register(unknown)
        registry.register(known)
        status = {row["name"]: row for row in registry.list_backends()}
        assert status["listing"]["healthy"] is True and status["listing"]["model_count"] is None, (
            "a healthy backend that cannot list reports an unknown count, not zero"
        )
        assert status["known"]["model_count"] == 1
        assert [m.name for m in registry.all_models()] == ["m"], "the unknown backend hides nothing else"
    finally:
        restore()


# ---------------------------------------------------------------------------
# LU6 -- the shared readers
# ---------------------------------------------------------------------------
def test_lu6_the_shared_readers_propagate_unknown_from_a_registered_backend():
    mod, restore = _unknown_window("registry_clients.py", _CLIENTS, with_clients=False)
    try:
        assert mod.backend_for() is not None, "control: a backend is registered"
        assert mod.installed_models() is None, "a backend that cannot say is unknown, not empty"
        assert mod.installed_model_names() is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# LU7 -- the callers whose answer changes
# ---------------------------------------------------------------------------
def test_lu7_the_extraction_falls_to_its_fallback_and_the_health_monitor_reports_unknown(tmp_path):
    mod, restore = _unknown_window("memory/extraction.py", "opti_oignon.memory.extraction",
                                   with_clients=False, packages=("opti_oignon", "opti_oignon.memory"))
    try:
        worker = mod.FactExtractor(fallback_models=["qwen3:8b", "other"])
        assert worker._resolve_model() == "qwen3:8b", "an unknown listing is the no-backend case, not an empty one"
    finally:
        restore()

    mod, restore = _unknown_window("model_health.py", "opti_oignon.model_health")
    try:
        monitor = mod.ModelHealthMonitor(config_path=tmp_path / "health.yaml")
        assert monitor._discover_models() is None, "discovery says unknown rather than an empty list"
        assert monitor.check_all() == {}, "nothing was checked, and nothing was recorded as failed"
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
