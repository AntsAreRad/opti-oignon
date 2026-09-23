#!/usr/bin/env python3
"""Contracts that Ollama's configured host is where the requests go.

``backends.yaml`` names an Ollama host and ``init_backends_from_config``
wrote it onto the backend, where no request read it: every head asked the
client library's module-level client, or a client built with a timeout
only, and both resolve their host from ``OLLAMA_HOST`` or the library's
default. The file now decides, unless ``OLLAMA_HOST`` is set, which wins as
the library documents. The shipped file names the library's own default,
so a machine that sets neither sends exactly where it sent before.

  * OH1 -- without a host and without ``OLLAMA_HOST`` every head asks the
    module-level client, as before, and no client is built.
  * OH2 -- with a host, every head -- health, listing, loaded set, model
    details, generate, stream, embeddings, eviction -- goes through one
    client built for that host, and the module-level client is not asked.
  * OH3 -- ``OLLAMA_HOST`` wins over the host: the module-level client
    answers, and a timeout builds a client without a host.
  * OH4 -- one client per host and timeout, built once and kept.
  * OH5 -- ``endpoint`` follows the host the requests use; the file's host
    reaches the backend through ``init_backends_from_config`` and routes a
    request; the shipped host is the one the library resolves by default.

Local-only (the public distribution ships no tests). The backend module is
loaded through the shared isolation window with the client library faked.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"
_MSGS = [{"role": "user", "content": "hi"}]
_HOST = "http://10.9.8.7:11434"


class _Transport:
    def __init__(self, label):
        self.label = label
        self.calls = []

    def list(self):
        self.calls.append("list")
        return {"models": []}

    def ps(self):
        self.calls.append("ps")
        return {"models": []}

    def show(self, name):
        self.calls.append("show")
        return {"details": {}, "model_info": {}}

    def generate(self, **kwargs):
        self.calls.append("generate")
        return {"response": ""}

    def chat(self, **kwargs):
        if kwargs.get("stream"):
            self.calls.append("chat-stream")
            return iter([{"message": {"content": "x"}, "done": True}])
        self.calls.append("chat")
        return {"message": {"content": "ok"}}

    def embed(self, **kwargs):
        self.calls.append("embed")
        texts = kwargs.get("input")
        n = len(texts) if isinstance(texts, list) else 1
        return {"embeddings": [[0.5]] * n}


class _FakeOllama(_Transport):
    def __init__(self):
        super().__init__("module")
        self.clients = []
        self.built = []

    def Client(self, **kwargs):  # noqa: N802 - the client library's spelling
        self.clients.append(dict(kwargs))
        client = _Transport(f"client{len(self.built) + 1}")
        self.built.append(client)
        return client


def _open():
    loaded, restore = isolate(targets={_BACKEND: source("inference_backend.py")}, packages=("opti_oignon",))
    mod = loaded[_BACKEND]
    fake = _FakeOllama()
    mod.OLLAMA_AVAILABLE = True
    mod._ollama_module = fake
    return mod, fake, restore


def _every_head(backend):
    assert backend.health_check() is True
    assert backend.list_models() == []
    assert backend.loaded_models() == []
    assert backend.model_info("m") is not None
    assert backend.generate("m", _MSGS).content == "ok"
    assert "".join(c.content for c in backend.stream("m", _MSGS)) == "x"
    assert backend.embed("m", "a") == [0.5]
    assert backend.embed_many("m", ["a", "b"]) == [[0.5], [0.5]]
    assert backend.unload_model("m") is True


_ALL_CALLS = ["list", "list", "ps", "show", "chat", "chat-stream", "embed", "embed", "generate"]


# ---------------------------------------------------------------------------
# OH1 -- no host, no variable: as before
# ---------------------------------------------------------------------------
def test_oh1_without_a_host_every_head_asks_the_module_level_client(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    mod, fake, restore = _open()
    try:
        _every_head(mod.OllamaBackend())
        assert fake.calls == _ALL_CALLS, "every head, on the module-level client"
        assert fake.clients == [], "and no client is built"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OH2 -- a host routes every head
# ---------------------------------------------------------------------------
def test_oh2_a_configured_host_routes_every_head_through_one_client(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    mod, fake, restore = _open()
    try:
        _every_head(mod.OllamaBackend(host=_HOST))
        assert fake.calls == [], "the module-level client is not asked"
        assert fake.clients == [{"host": _HOST}], "one client, for the configured host"
        assert fake.built[0].calls == _ALL_CALLS, "and every head went through it"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OH3 -- OLLAMA_HOST wins
# ---------------------------------------------------------------------------
def test_oh3_the_environment_variable_wins_over_the_configured_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "10.1.1.1:11434")
    mod, fake, restore = _open()
    try:
        backend = mod.OllamaBackend(host=_HOST)
        _every_head(backend)
        assert fake.calls == _ALL_CALLS and fake.clients == [], "the library resolves the variable itself"
        backend.generate("m", _MSGS, options={"timeout": 5})
        assert fake.clients == [{"timeout": 5}], "a timeout builds a client without a host"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OH4 -- one client per host and timeout
# ---------------------------------------------------------------------------
def test_oh4_one_client_per_host_and_timeout_built_once(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    mod, fake, restore = _open()
    try:
        backend = mod.OllamaBackend(host=_HOST)
        backend.generate("m", _MSGS)
        backend.generate("m", _MSGS, options={"timeout": 5})
        backend.generate("m", _MSGS, options={"timeout": 5})
        backend.embed("m", "a", timeout=5)
        backend.generate("m", _MSGS)
        assert fake.clients == [{"host": _HOST}, {"host": _HOST, "timeout": 5}]
        assert fake.built[1].calls == ["chat", "chat", "embed"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# OH5 -- the endpoint follows, the file reaches the backend, the shipped host
# ---------------------------------------------------------------------------
def test_oh5_the_endpoint_follows_the_host_and_the_file_reaches_the_backend(monkeypatch, tmp_path):
    mod, fake, restore = _open()
    try:
        seen = []
        mod._ollama_host_parser = lambda: (lambda host: seen.append(host) or f"parsed:{host}")
        monkeypatch.setenv("OLLAMA_HOST", "10.1.1.1:11434")
        assert mod.OllamaBackend(host=_HOST).endpoint() == "parsed:10.1.1.1:11434"
        monkeypatch.delenv("OLLAMA_HOST")
        assert mod.OllamaBackend(host=_HOST).endpoint() == f"parsed:{_HOST}"
        assert mod.OllamaBackend().endpoint() == "parsed:None"
        assert seen == ["10.1.1.1:11434", _HOST, None]

        config = tmp_path / "backends.yaml"
        config.write_text(yaml.safe_dump({"ollama": {"host": _HOST}}), encoding="utf-8")
        registry = mod.init_backends_from_config(str(config))
        ollama = registry.get("ollama")
        assert ollama is not None and ollama.generate("m", _MSGS).content == "ok"
        assert {"host": _HOST} in fake.clients, "the file's host routes the request"
    finally:
        restore()

    shipped = yaml.safe_load((REPO / "opti_oignon" / "config" / "backends.yaml").read_text(encoding="utf-8"))
    client = pytest.importorskip("ollama._client")
    assert client._parse_host(shipped["ollama"]["host"]) == client._parse_host(None), (
        "the shipped host is where the library sends by default: nothing moves without a setting"
    )
