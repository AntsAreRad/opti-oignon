"""S186 audit fix IB-02 -- hold the GGUF model load under a per-model lock.

Before the fix, LlamaCppBackend._get_or_load checked the loaded-models cache and
constructed a Llama instance with no lock, while generate/stream took the coarse
self._lock only around create_chat_completion. Two concurrent first-use calls for
the same uncached model both missed the cache and both constructed a Llama (double
GGUF load -> double VRAM/RAM) while racing the dict write.

The fix adds a per-model load lock (double-checked) so a model is constructed
exactly once even under concurrent first use, keeps a lock-free fast path for an
already-loaded model, and replaces the coarse inference lock with a per-model
inference lock (a Llama instance is not safe for concurrent calls; distinct models
may generate in parallel).

The module is loaded in isolation: ``ollama`` is stubbed and ``opti_oignon`` is a
bare module, so the optional engine imports are absent and the telemetry lazy import
degrades to None. ``_LlamaCpp`` is replaced by a counting fake; no real llama-cpp.
"""

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

sys.modules.setdefault("opti_oignon", types.ModuleType("opti_oignon"))
sys.modules.setdefault("ollama", types.ModuleType("ollama"))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH = _REPO_ROOT / "opti_oignon" / "inference_backend.py"


def _load():
    spec = importlib.util.spec_from_file_location("inference_backend_ib02", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (3.12 dataclass ordering)
    spec.loader.exec_module(mod)
    return mod


ib = _load()


class _FakeLlama:
    """Counts constructions and widens the load race window with a small sleep."""

    construct_count = 0
    _count_lock = threading.Lock()

    def __init__(self, **kwargs):
        with _FakeLlama._count_lock:
            _FakeLlama.construct_count += 1
        # Widen the window during which a second thread could race a load.
        time.sleep(0.05)

    def create_chat_completion(self, **kwargs):
        if kwargs.get("stream"):
            def _gen():
                yield {
                    "choices": [
                        {"delta": {"content": "ok"}, "finish_reason": None}
                    ]
                }
                yield {
                    "choices": [
                        {"delta": {"content": ""}, "finish_reason": "stop"}
                    ]
                }
            return _gen()
        return {"choices": [{"message": {"content": "ok"}}]}


def _install_fake():
    _FakeLlama.construct_count = 0
    ib.LLAMA_CPP_AVAILABLE = True
    ib._LlamaCpp = _FakeLlama


# ---------------------------------------------------------------------------
# Core property: single construction under concurrent first use (_get_or_load)
# ---------------------------------------------------------------------------

def test_single_construction_under_concurrent_first_use(tmp_path):
    (tmp_path / "test-model.gguf").write_bytes(b"GGUF")
    _install_fake()
    backend = ib.LlamaCppBackend(model_dirs=[str(tmp_path)])

    barrier = threading.Barrier(2)
    results: list = []
    errors: list = []

    def worker():
        try:
            barrier.wait(timeout=5)
            results.append(backend._get_or_load("test-model"))
        except Exception as exc:  # pragma: no cover - surfaced via assert
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(not t.is_alive() for t in threads), "load deadlocked"
    assert not errors, errors
    assert _FakeLlama.construct_count == 1
    assert len(results) == 2 and results[0] is results[1]


# ---------------------------------------------------------------------------
# Fast path: a cache hit returns the same instance without reconstructing
# ---------------------------------------------------------------------------

def test_cache_hit_returns_same_instance_without_reconstruct(tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"GGUF")
    _install_fake()
    backend = ib.LlamaCppBackend(model_dirs=[str(tmp_path)])

    first = backend._get_or_load("m")
    second = backend._get_or_load("m")

    assert first is second
    assert _FakeLlama.construct_count == 1


# ---------------------------------------------------------------------------
# Granularity: locks are per model, load and inference registries independent
# ---------------------------------------------------------------------------

def test_locks_are_per_model_and_registries_independent():
    backend = ib.LlamaCppBackend(model_dirs=[])

    inf_a1 = backend._lock_for(backend._inference_locks, "a")
    inf_a2 = backend._lock_for(backend._inference_locks, "a")
    inf_b = backend._lock_for(backend._inference_locks, "b")

    assert inf_a1 is inf_a2          # same model -> same lock
    assert inf_a1 is not inf_b       # distinct models -> distinct locks

    load_a = backend._lock_for(backend._load_locks, "a")
    assert load_a is not inf_a1      # load and inference registries are distinct


# ---------------------------------------------------------------------------
# generate() end to end: one construction under concurrent first use
# ---------------------------------------------------------------------------

def test_generate_concurrent_first_use_single_construction(tmp_path):
    (tmp_path / "g.gguf").write_bytes(b"GGUF")
    _install_fake()
    backend = ib.LlamaCppBackend(model_dirs=[str(tmp_path)])

    barrier = threading.Barrier(2)
    contents: list = []
    errors: list = []

    def worker():
        try:
            barrier.wait(timeout=5)
            resp = backend.generate("g", [{"role": "user", "content": "hi"}])
            contents.append(resp.content)
        except Exception as exc:  # pragma: no cover - surfaced via assert
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(not t.is_alive() for t in threads), "generate deadlocked"
    assert not errors, errors
    assert _FakeLlama.construct_count == 1
    assert contents == ["ok", "ok"]


# ---------------------------------------------------------------------------
# Docstring no longer overclaims that the coarse lock protects loading
# ---------------------------------------------------------------------------

def test_docstring_no_longer_overclaims_load_protection():
    doc = ib.LlamaCppBackend.__doc__ or ""
    assert "per-model load lock" in doc
    assert "protects model loading" not in doc


# ---------------------------------------------------------------------------
# Source assertion: _get_or_load takes the per-model load lock, fast path uses get
# ---------------------------------------------------------------------------

def test_get_or_load_source_uses_per_model_load_lock():
    src = _PATH.read_text(encoding="utf-8")
    start = src.index("def _get_or_load(")
    end = src.index("def unload_model", start)
    body = src[start:end]
    assert "_lock_for(self._load_locks" in body
    assert "self._loaded_models.get(" in body
