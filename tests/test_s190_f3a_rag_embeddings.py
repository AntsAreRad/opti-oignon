"""S190 F3a -- rag/embeddings.py correctness fixes (RST-01, RST-02).

Loaded with the spec_from_file_location + sys.modules-stub idiom (register
before exec_module) so the heavy/optional deps of rag/embeddings.py
(requests, numpy, tqdm) and its rag.config dependency do not need to be
installed in the verification container.

RST-01: once OllamaEmbeddings has switched to the legacy /api/embeddings
        endpoint, embed_single must route through the legacy path (payload key
        "prompt", response key singular "embedding"); the /api/embed path sends
        "input" and parses plural "embeddings" and would return None for a
        legacy 200 response.
RST-02: on a batch count mismatch, embed_batch must fall back to sequential so
        the returned list stays 1:1 aligned with the inputs (a mismatched list
        returned as-is gets zip-truncated / mis-paired by the caller).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

RAG_DIR = Path(__file__).resolve().parent.parent / "opti_oignon" / "rag"


def _install_stub(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _make_requests_stub():
    """Minimal stand-in for the requests module with assignable post/get."""
    mod = types.ModuleType("requests")

    class _Exc(Exception):
        pass

    exceptions = types.SimpleNamespace(
        Timeout=type("Timeout", (_Exc,), {}),
        RequestException=type("RequestException", (_Exc,), {}),
        ConnectionError=type("ConnectionError", (_Exc,), {}),
    )
    mod.exceptions = exceptions
    mod.get = lambda *a, **k: (_ for _ in ()).throw(AssertionError("requests.get not stubbed"))
    mod.post = lambda *a, **k: (_ for _ in ()).throw(AssertionError("requests.post not stubbed"))
    return mod


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture()
def embeddings_mod():
    """Load rag.config then rag.embeddings with stubbed heavy deps."""
    # Stub heavy / optional third-party deps before loading the module.
    _install_stub("numpy", types.ModuleType("numpy"))
    sys.modules["numpy"].ndarray = object  # used in a return annotation
    sys.modules["numpy"].array = lambda *a, **k: None
    sys.modules["numpy"].linalg = types.SimpleNamespace(norm=lambda *a, **k: 1.0)
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda x, **k: x
    _install_stub("tqdm", tqdm_mod)
    requests_stub = _make_requests_stub()
    _install_stub("requests", requests_stub)

    # Synthetic parent package so embeddings.py's `from .config import ...`
    # relative import resolves to the loaded config module.
    pkg = types.ModuleType("rag_pkg_s190")
    pkg.__path__ = [str(RAG_DIR)]
    sys.modules["rag_pkg_s190"] = pkg

    # Load rag.config (dataclass module -- register before exec_module).
    spec_cfg = importlib.util.spec_from_file_location("rag_pkg_s190.config", RAG_DIR / "config.py")
    cfg_mod = importlib.util.module_from_spec(spec_cfg)
    sys.modules["rag_pkg_s190.config"] = cfg_mod
    spec_cfg.loader.exec_module(cfg_mod)

    # Load rag.embeddings as a child of the synthetic package.
    spec_emb = importlib.util.spec_from_file_location("rag_pkg_s190.embeddings", RAG_DIR / "embeddings.py")
    emb_mod = importlib.util.module_from_spec(spec_emb)
    sys.modules["rag_pkg_s190.embeddings"] = emb_mod
    spec_emb.loader.exec_module(emb_mod)

    yield emb_mod, cfg_mod, requests_stub

    for k in ("rag_pkg_s190.embeddings", "rag_pkg_s190.config", "rag_pkg_s190",
              "numpy", "tqdm", "requests"):
        sys.modules.pop(k, None)


def _new_embedder(emb_mod, cfg_mod):
    cfg = cfg_mod.EmbeddingConfig(model="mxbai-embed-large", ollama_url="http://localhost:11434")
    emb = emb_mod.OllamaEmbeddings(cfg)
    emb._model_verified = True  # short-circuit _verify_model (no /api/tags call)
    return emb


# --------------------------------------------------------------------------
# RST-01
# --------------------------------------------------------------------------

def test_rst01_legacy_single_returns_vector(embeddings_mod):
    emb_mod, cfg_mod, requests_stub = embeddings_mod
    emb = _new_embedder(emb_mod, cfg_mod)

    # Simulate the state after a prior 400 -> legacy switch.
    emb._use_legacy = True
    emb.url = "http://localhost:11434/api/embeddings"

    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["payload"] = json
        # Legacy endpoint responds 200 with the SINGULAR "embedding" key.
        return _FakeResponse(200, {"embedding": [0.11, 0.22, 0.33]})

    requests_stub.post = fake_post

    vec = emb.embed_single("hello")
    assert vec == [0.11, 0.22, 0.33], "legacy embed_single must return the vector, not None"
    # It must have used the legacy payload key "prompt", not "input".
    assert "prompt" in seen["payload"] and "input" not in seen["payload"]
    assert seen["url"].endswith("/api/embeddings")


def test_rst01_non_legacy_still_uses_embed_endpoint(embeddings_mod):
    emb_mod, cfg_mod, requests_stub = embeddings_mod
    emb = _new_embedder(emb_mod, cfg_mod)
    # Default (modern) mode: _use_legacy is False.
    assert emb._use_legacy is False

    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["payload"] = json
        return _FakeResponse(200, {"embeddings": [[0.5, 0.6]]})

    requests_stub.post = fake_post
    vec = emb.embed_single("hi")
    assert vec == [0.5, 0.6]
    assert "input" in seen["payload"]  # modern path uses "input"


# --------------------------------------------------------------------------
# RST-02
# --------------------------------------------------------------------------

def test_rst02_batch_mismatch_falls_back_to_sequential(embeddings_mod):
    emb_mod, cfg_mod, requests_stub = embeddings_mod
    emb = _new_embedder(emb_mod, cfg_mod)

    texts = ["a", "b", "c"]

    def fake_post(url, json=None, timeout=None):
        payload_input = json.get("input")
        if isinstance(payload_input, list):
            # Batch call: return a MISMATCHED count (1 vector for 3 inputs).
            return _FakeResponse(200, {"embeddings": [[9.0, 9.0]]})
        # Single call (sequential fallback): one vector per input.
        return _FakeResponse(200, {"embeddings": [[1.0, 1.0]]})

    requests_stub.post = fake_post

    out = emb.embed_batch(texts)
    # The fix guarantees 1:1 alignment with the inputs.
    assert len(out) == len(texts), "batch mismatch must not shrink/misalign the result"
    assert all(v == [1.0, 1.0] for v in out)


def test_rst02_batch_matching_count_unchanged(embeddings_mod):
    emb_mod, cfg_mod, requests_stub = embeddings_mod
    emb = _new_embedder(emb_mod, cfg_mod)
    texts = ["x", "y"]

    def fake_post(url, json=None, timeout=None):
        return _FakeResponse(200, {"embeddings": [[1, 1], [2, 2]]})

    requests_stub.post = fake_post
    out = emb.embed_batch(texts)
    assert out == [[1, 1], [2, 2]]  # matching count: returned directly, no fallback
