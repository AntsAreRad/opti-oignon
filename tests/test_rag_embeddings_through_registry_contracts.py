#!/usr/bin/env python3
"""Contracts that every embedding the application computes goes through the registry.

``rag.embeddings.OllamaEmbeddings`` is the embedder of the RAG store, the
project context and the memory's semantic recall. It posted to the
inference server's embedding endpoints with its own HTTP transport, so no
vector was ever admitted by the governor. It now asks the registry's
backend for the configured model, one text through ``embed`` and a batch
through ``embed_many``, and keeps the two guards it had: a batch answer
that does not align with its texts is redone text by text, and a text
that fails is ``None`` in its own slot.

  * EM1 -- one text: the backend's ``embed`` with the model and the
    configured timeout; no backend, or a backend that fails, is ``None``.
  * EM2 -- a batch: one ``embed_many`` with twice the timeout; a
    misaligned answer, a failure, or a backend without a batch endpoint is
    redone text by text, one slot per text.
  * EM3 -- the model is verified against the registry's catalogue:
    configured name completed with its tag, then the fast model, then any
    embedding model; a catalogue nobody could read verifies nothing and no
    vector is asked for.
  * EM4 -- the status report reads the registry: running is the backend's
    health, the models are its catalogue, and an unreadable catalogue is
    said by name.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window with the HTTP transport declared unreachable.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_EMB = "opti_oignon.rag.embeddings"


class _Backend:
    def __init__(self, *, models=("mxbai-embed-large:latest", "llama3:8b"), healthy=True,
                 batch="aligned", fail_single=False):
        self.models = None if models is None else [SimpleNamespace(name=m) for m in models]
        self.healthy = healthy
        self.batch = batch
        self.fail_single = fail_single
        self.calls = []

    def health_check(self):
        return self.healthy

    def list_models(self):
        self.calls.append(("list",))
        return self.models

    def embed(self, model, text, timeout=None):
        self.calls.append(("embed", model, text, timeout))
        if self.fail_single:
            raise RuntimeError("embedding down")
        return [float(len(text))]

    def embed_many(self, model, texts, timeout=None):
        self.calls.append(("embed_many", model, list(texts), timeout))
        if self.batch == "aligned":
            return [[float(len(t))] for t in texts]
        if self.batch == "short":
            raise ValueError(f"1 vector(s) for {len(texts)} text(s)")
        if self.batch == "none":
            return None
        raise RuntimeError("batch down")


class _Registry:
    def __init__(self, backend):
        self.backend = backend
        self.asked = []

    def resolve_backend(self, model):
        self.asked.append(model)
        return self.backend


def _open(registry):
    backend_mod = types.ModuleType("opti_oignon.inference_backend")
    backend_mod.get_backend_registry = lambda: registry
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda it, **kw: it
    loaded, restore = isolate(
        targets={
            "opti_oignon.rag.config": source("rag", "config.py"),
            _EMB: source("rag", "embeddings.py"),
        },
        blocked=("requests",),
        seeded={"opti_oignon.inference_backend": backend_mod, "tqdm": tqdm_mod},
        packages=("opti_oignon.rag",),
    )
    mod = loaded[_EMB]
    config = loaded["opti_oignon.rag.config"].EmbeddingConfig(timeout=30)
    return mod, config, restore


# ---------------------------------------------------------------------------
# EM1 -- one text
# ---------------------------------------------------------------------------
def test_em1_one_text_is_embedded_by_the_registrys_backend_with_the_configured_timeout():
    backend = _Backend()
    mod, config, restore = _open(_Registry(backend))
    try:
        embedder = mod.OllamaEmbeddings(config)
        assert embedder.embed_single("onion") == [5.0]
        assert ("embed", "mxbai-embed-large:latest", "onion", 30) in backend.calls, (
            "the verified model with its tag, and the configured timeout"
        )
    finally:
        restore()

    mod, config, restore = _open(_Registry(_Backend(fail_single=True)))
    try:
        assert mod.OllamaEmbeddings(config).embed_single("onion") is None, "a failing backend is None, not a raise"
    finally:
        restore()

    mod, config, restore = _open(_Registry(None))
    try:
        assert mod.OllamaEmbeddings(config).embed_single("onion") is None, "no backend, no vector"
    finally:
        restore()


# ---------------------------------------------------------------------------
# EM2 -- a batch, and its guards
# ---------------------------------------------------------------------------
def test_em2_a_batch_is_one_request_and_a_misaligned_or_failed_one_is_redone_text_by_text():
    backend = _Backend()
    mod, config, restore = _open(_Registry(backend))
    try:
        embedder = mod.OllamaEmbeddings(config)
        assert embedder.embed_batch(["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]]
        batches = [c for c in backend.calls if c[0] == "embed_many"]
        assert batches == [("embed_many", "mxbai-embed-large:latest", ["a", "bb", "ccc"], 60)], (
            "one request for the batch, with twice the timeout"
        )
        assert not [c for c in backend.calls if c[0] == "embed"], "and no per-text request"
        assert embedder.embed(["a", "bb", "ccc", "dddd"], show_progress=False, batch_size=2) == [[1.0], [2.0], [3.0], [4.0]]
    finally:
        restore()

    for mode in ("short", "down", "none"):
        backend = _Backend(batch=mode)
        mod, config, restore = _open(_Registry(backend))
        try:
            out = mod.OllamaEmbeddings(config).embed_batch(["a", "bb"])
            assert out == [[1.0], [2.0]], f"{mode}: redone text by text, one slot per text"
            assert [c[2] for c in backend.calls if c[0] == "embed"] == ["a", "bb"]
        finally:
            restore()


# ---------------------------------------------------------------------------
# EM3 -- the model is verified against the registry's catalogue
# ---------------------------------------------------------------------------
def test_em3_the_model_is_verified_against_the_catalogue_and_unknown_verifies_nothing():
    backend = _Backend(models=("nomic-embed-text:v1.5", "llama3:8b"))
    mod, config, restore = _open(_Registry(backend))
    try:
        embedder = mod.OllamaEmbeddings(config)
        assert embedder.embed_single("x") == [1.0]
        assert embedder.config.model == "nomic-embed-text:v1.5", "the fast model, with its tag"
    finally:
        restore()

    backend = _Backend(models=("bge-m3:latest",))
    mod, config, restore = _open(_Registry(backend))
    try:
        embedder = mod.OllamaEmbeddings(config)
        assert embedder.embed_single("x") == [1.0] and embedder.config.model == "bge-m3:latest", "any embedding model"
    finally:
        restore()

    backend = _Backend(models=None)
    mod, config, restore = _open(_Registry(backend))
    try:
        embedder = mod.OllamaEmbeddings(config)
        assert embedder.embed_single("x") is None, "a catalogue nobody could read verifies nothing"
        assert embedder.embed_batch(["a", "b"]) == [None, None]
        assert [c for c in backend.calls if c[0] in ("embed", "embed_many")] == [], "and no vector is asked for"
    finally:
        restore()


# ---------------------------------------------------------------------------
# EM4 -- the status report reads the registry
# ---------------------------------------------------------------------------
def test_em4_the_status_report_reads_the_registrys_health_and_catalogue():
    backend = _Backend()
    mod, config, restore = _open(_Registry(backend))
    try:
        status = mod.check_ollama_status()
        assert status["ollama_running"] is True and status["embedding_model_available"] is True
        assert status["available_models"] == ["mxbai-embed-large:latest", "llama3:8b"] and status["error"] is None
    finally:
        restore()

    mod, config, restore = _open(_Registry(_Backend(models=None)))
    try:
        status = mod.check_ollama_status()
        assert status["ollama_running"] is True and status["embedding_model_available"] is False
        assert "could not list" in status["error"], "an unreadable catalogue is said by name"
    finally:
        restore()

    mod, config, restore = _open(_Registry(_Backend(healthy=False)))
    try:
        status = mod.check_ollama_status()
        assert status["ollama_running"] is False and status["error"], "an unhealthy backend is not running"
    finally:
        restore()

    mod, config, restore = _open(_Registry(None))
    try:
        status = mod.check_ollama_status()
        assert status["ollama_running"] is False and "no backend" in status["error"]
    finally:
        restore()
