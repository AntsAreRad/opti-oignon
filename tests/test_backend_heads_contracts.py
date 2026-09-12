#!/usr/bin/env python3
"""Contracts for the two observation heads on the backend contract: the
loaded set and the embedding.

Two modules still asked the client library for what the registry could not
answer: the warmup read the loaded set through ``ps()`` and the semantic
cache asked for an embedding directly. Both now have a head on the backend
contract, defaulted rather than abstract on purpose: a backend that cannot
answer says so with ``None``, and an unknown that announces itself is the
whole point -- an empty list would read as "nothing loaded" where the truth
is "nobody looked".

  * BH1 -- the base contract answers ``None`` to both heads for a backend
    that implements only the seven it always had.
  * BH2 -- Ollama reads the loaded set through ``ps()`` in both response
    forms, carries the five fields the warmup read, answers ``None`` when
    the client is absent or ``ps()`` fails, and evicts through the same
    read.
  * BH3 -- Ollama asks the client's ``embed`` and returns the first vector
    in both forms; ``None`` without the client; a client failure
    propagates; the governor is asked before the client.
  * BH4 -- llama.cpp answers its in-process set, empty when nothing is
    loaded (a known empty, not an unknown), and has no embedding.
  * BH5 -- llama-server and the remote core answer ``None`` to both: the
    slot listing does not name residency and the daemon has no route, and
    a reachable server still answers ``None`` rather than ``[]``.
  * BH6 -- the test bridge answers through the scripted client's ``ps`` and
    ``embed`` when it has them and ``None`` when it does not.

Local-only (the public distribution ships no tests). The backend module is
loaded through the shared isolation window with every client faked.
"""

import datetime as _dt
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import ScriptedBackend  # noqa: E402

_BACKEND = "opti_oignon.inference_backend"
_WHEN = _dt.datetime(2026, 9, 12, 12, 0, tzinfo=_dt.timezone.utc)


class _Boom(Exception):
    pass


class _FakeOllama:
    def __init__(self, ps=None, embedding=None, fail=False):
        self.calls = []
        self._ps = ps
        self._embedding = embedding
        self._fail = fail

    def ps(self):
        self.calls.append(("ps",))
        if self._fail:
            raise _Boom("ps down")
        return self._ps

    def embed(self, **kwargs):
        self.calls.append(("embed", kwargs))
        if self._fail:
            raise _Boom("embed down")
        return self._embedding

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return {"response": ""}


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


def _entry_dict(name):
    return {
        "model": name,
        "size_vram": 1024,
        "expires_at": _WHEN,
        "context_length": 8192,
        "digest": "sha256:" + name,
    }


class _Size:
    def __int__(self):
        return 2048


class _Entry:
    def __init__(self, name):
        self.name = name
        self.model = name
        self.size_vram = _Size()
        self.expires_at = 1700000000
        self.context_length = 4096
        self.digest = "sha256:" + name


class _PsObject:
    def __init__(self, models):
        self.models = models


class _EmbedObject:
    def __init__(self, vectors):
        self.embeddings = vectors


# ---------------------------------------------------------------------------
# BH1 -- the base contract answers unknown
# ---------------------------------------------------------------------------
def test_bh1_a_backend_with_only_the_seven_heads_answers_none_to_both():
    mod, restore = _open()
    try:
        class _Seven(mod.InferenceBackend):
            name = "seven"
            display_name = "seven"

            def health_check(self):
                return True

            def list_models(self):
                return []

            def model_info(self, model_name):
                return None

            def generate(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
                raise AssertionError("not asked")

            def stream(self, model, messages, options=None, keep_alive="30m", think=False, images=None):
                raise AssertionError("not asked")

        backend = _Seven()
        assert backend.loaded_models() is None
        assert backend.embed("m", "text") is None
        assert getattr(mod.InferenceBackend.loaded_models, "__isabstractmethod__", False) is False
        assert getattr(mod.InferenceBackend.embed, "__isabstractmethod__", False) is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# BH2 -- Ollama reads the loaded set through ps(), both forms
# ---------------------------------------------------------------------------
def test_bh2_ollama_reads_the_loaded_set_through_ps_in_both_forms_and_unknown_is_none():
    mod, restore = _open()
    try:
        fake = _FakeOllama(ps={"models": [_entry_dict("a"), _entry_dict("b")]})
        loaded = _ollama(mod, fake).loaded_models()
        assert fake.calls == [("ps",)]
        assert [m.name for m in loaded] == ["a", "b"]
        first = loaded[0]
        assert first.backend == "ollama"
        assert first.size_vram == 1024 and isinstance(first.size_vram, int)
        assert first.expires_at == _WHEN.timestamp()
        assert first.context_length == 8192
        assert first.digest == "sha256:a"
        assert first.to_dict() == {
            "name": "a", "backend": "ollama", "size_vram": 1024,
            "expires_at": _WHEN.timestamp(), "context_length": 8192, "digest": "sha256:a",
        }

        obj = _FakeOllama(ps=_PsObject([_Entry("c")]))
        loaded = _ollama(mod, obj).loaded_models()
        assert [m.name for m in loaded] == ["c"]
        assert loaded[0].size_vram == 2048 and isinstance(loaded[0].size_vram, int)
        assert loaded[0].expires_at == 1700000000.0 and loaded[0].context_length == 4096

        empty = _FakeOllama(ps={"models": []})
        assert _ollama(mod, empty).loaded_models() == []

        assert _ollama(mod, _FakeOllama(), available=False).loaded_models() is None
        down = _FakeOllama(fail=True)
        assert _ollama(mod, down).loaded_models() is None
        assert down.calls == [("ps",)]

        # Eviction reads the loaded set through the same head.
        fake = _FakeOllama(ps={"models": [_entry_dict("a"), _entry_dict("b")]})
        assert _ollama(mod, fake).unload_all() == 2
        evicted = [kw for kind, kw in fake.calls[1:] if kind == "generate"]
        assert [kw["model"] for kw in evicted] == ["a", "b"]
        assert all(kw["keep_alive"] == 0 for kw in evicted)
        assert _ollama(mod, _FakeOllama(fail=True)).unload_all() == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# BH3 -- Ollama embeds through the client, after the governor
# ---------------------------------------------------------------------------
def _governor(refuse):
    module = types.ModuleType("opti_oignon.resource_governor")

    class GovernorRefusal(RuntimeError):
        pass

    def backend_admission_gate(model, options):
        module.asked.append((model, options))
        if refuse:
            raise GovernorRefusal(f"refused {model}")

    module.FEATURE_AVAILABLE = True
    module.GovernorRefusal = GovernorRefusal
    module.backend_admission_gate = backend_admission_gate
    module.asked = []
    return module


def test_bh3_ollama_embeds_through_the_client_in_both_forms_after_the_governor():
    mod, restore = _open()
    try:
        fake = _FakeOllama(embedding={"embeddings": [[0.1, 0.2], [0.9, 0.9]]})
        vector = _ollama(mod, fake).embed("emb", "hello")
        assert vector == [0.1, 0.2]
        assert fake.calls == [("embed", {"model": "emb", "input": "hello"})]

        obj = _FakeOllama(embedding=_EmbedObject([[0.3]]))
        assert _ollama(mod, obj).embed("emb", "x") == [0.3]

        assert _ollama(mod, _FakeOllama(embedding={"embeddings": []})).embed("emb", "x") is None
        assert _ollama(mod, _FakeOllama(), available=False).embed("emb", "x") is None

        down = _FakeOllama(fail=True)
        with pytest.raises(_Boom):
            _ollama(mod, down).embed("emb", "x")
    finally:
        restore()

    governor = _governor(refuse=True)
    mod, restore = _open(seeded={"opti_oignon.resource_governor": governor})
    try:
        fake = _FakeOllama(embedding={"embeddings": [[1.0]]})
        with pytest.raises(governor.GovernorRefusal):
            _ollama(mod, fake).embed("emb", "x")
        assert governor.asked == [("emb", None)]
        assert fake.calls == []
    finally:
        restore()

    governor = _governor(refuse=False)
    mod, restore = _open(seeded={"opti_oignon.resource_governor": governor})
    try:
        fake = _FakeOllama(embedding={"embeddings": [[1.0]]})
        assert _ollama(mod, fake).embed("emb", "x") == [1.0]
        assert governor.asked == [("emb", None)] and len(fake.calls) == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# BH4 -- llama.cpp answers its own set
# ---------------------------------------------------------------------------
def test_bh4_llama_cpp_answers_its_in_process_set_and_has_no_embedding():
    mod, restore = _open()
    try:
        backend = mod.LlamaCppBackend(model_dirs=[])
        assert backend.loaded_models() == []
        backend._loaded_models["one.gguf"] = object()
        backend._loaded_models["two.gguf"] = object()
        loaded = backend.loaded_models()
        assert [m.name for m in loaded] == ["one.gguf", "two.gguf"]
        assert {m.backend for m in loaded} == {"llama_cpp"}
        assert all(m.size_vram is None and m.expires_at is None for m in loaded)
        assert backend.embed("one.gguf", "x") is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# BH5 -- llama-server and the remote core answer unknown
# ---------------------------------------------------------------------------
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
    def __init__(self):
        self.requests = []

    def urlopen(self, req, timeout=None):
        self.requests.append(req.full_url)
        return _FakeResponse(json.dumps({"data": [{"id": "served.gguf"}]}).encode("utf-8"))


def test_bh5_llama_server_and_the_remote_core_answer_unknown_not_empty():
    mod, restore = _open()
    try:
        transport = _Transport()
        mod.urllib.request.urlopen = transport.urlopen
        backend = mod.LlamaServerBackend(host="http://fake:8080")
        assert backend.health_check() is True
        assert [m.name for m in backend.list_models()] == ["served.gguf"]
        assert backend.loaded_models() is None
        assert backend.embed("served.gguf", "x") is None
    finally:
        restore()

    class _Base:
        pass

    seeded_backend = types.ModuleType(_BACKEND)
    seeded_backend.get_backend_registry = lambda: None
    seeded_backend.InferenceBackend = _Base
    seeded_backend.ChatResponse = object
    seeded_backend.StreamChunk = object
    seeded_backend.BackendModelInfo = object
    loaded, restore = isolate(
        targets={"opti_oignon.core_client": source("core_client.py")},
        seeded={_BACKEND: seeded_backend},
        packages=("opti_oignon",),
    )
    try:
        remote = loaded["opti_oignon.core_client"].RemoteCoreBackend("http://127.0.0.1:1")
        assert remote.loaded_models() is None
        assert remote.embed("m", "x") is None
    finally:
        restore()


# ---------------------------------------------------------------------------
# BH6 -- the bridge answers through the scripted client
# ---------------------------------------------------------------------------
class _ScriptedWithBoth:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        return {"message": {"content": ""}}

    def ps(self):
        self.calls.append(("ps",))
        return {"models": [_entry_dict("a")]}

    def embed(self, **kwargs):
        self.calls.append(("embed", kwargs))
        return {"embeddings": [[0.5, 0.5]]}


class _ScriptedChatOnly:
    def chat(self, **kwargs):
        return {"message": {"content": ""}}


def test_bh6_the_bridge_answers_ps_and_embed_when_the_scripted_client_has_them():
    scripted = _ScriptedWithBoth()
    backend = ScriptedBackend(scripted)
    loaded = backend.loaded_models()
    assert [m.name for m in loaded] == ["a"]
    assert loaded[0].size_vram == 1024 and loaded[0].digest == "sha256:a"
    assert loaded[0].expires_at == _WHEN.timestamp() and loaded[0].context_length == 8192
    assert backend.embed("emb", "hello") == [0.5, 0.5]
    assert scripted.calls == [("ps",), ("embed", {"model": "emb", "input": "hello"})]

    plain = ScriptedBackend(_ScriptedChatOnly())
    assert plain.loaded_models() is None
    assert plain.embed("emb", "hello") is None
