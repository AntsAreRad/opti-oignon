"""S189 phase F2 -- Model lifecycle & health (item 6) regression tests.

Covers the two applied fixes:

- MH-01: ``ModelHealthMonitor.get_all_health`` / ``get_healthy_models`` /
  ``get_degraded_models`` / ``get_unavailable_models`` iterated ``_records`` without the
  lock while the health-check thread inserts records under the lock -> a "dictionary
  changed size during iteration" race (MM-02 class), on the smart_router failover read
  path. The four iterating readers now acquire ``self._lock``. (model_health imports
  ollama at load, so this is asserted at source/AST level.)
- WRM-01: ``ModelWarmup.get_loaded_models`` called ``ps_response.get("models", [])``
  unconditionally, which raised ``AttributeError`` on the object-form ``ProcessResponse``
  (newer ollama), so it returned [] and ``is_model_loaded`` was always False (warmup never
  skipped). Now dispatches on dict vs object. (model_warmup has no relative imports and
  ``get_loaded_models`` uses no instance state, so it is exercised in isolation.)
"""

import ast
import importlib.util
import pathlib
import sys
import types

_REPO = pathlib.Path(__file__).resolve().parents[1]
_HEALTH = _REPO / "opti_oignon" / "model_health.py"
_WARMUP = _REPO / "opti_oignon" / "model_warmup.py"


# --- MH-01: source/AST assertions ---

_HSRC = _HEALTH.read_text(encoding="utf-8")
_HTREE = ast.parse(_HSRC)


def _method_body(tree, src, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"method {name} not found")


def test_mh01_iterating_readers_acquire_lock():
    for m in ("get_all_health", "get_healthy_models",
              "get_degraded_models", "get_unavailable_models"):
        seg = _method_body(_HTREE, _HSRC, m)
        assert "with self._lock:" in seg, m


# --- WRM-01: functional, isolated load ---

def _load_warmup_isolated():
    ollama_stub = types.ModuleType("ollama")
    ollama_stub.ps = lambda: {"models": []}
    sys.modules["ollama"] = ollama_stub
    spec = importlib.util.spec_from_file_location(
        "opti_oignon_model_warmup_probe", _WARMUP
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OLLAMA_AVAILABLE = True
    module._ollama = ollama_stub
    return module


class _PSObjModel:
    def __init__(self, name, size_vram):
        self.name = name
        self.size_vram = size_vram
        self.expires_at = None
        self.context_length = None
        self.digest = None


class _PSObjResponse:
    """Mimics a newer ollama ProcessResponse (no dict .get)."""

    def __init__(self, models):
        self.models = models


def test_wrm01_get_loaded_models_handles_object_ps():
    m = _load_warmup_isolated()
    m._ollama.ps = lambda: _PSObjResponse([_PSObjModel("qwen3:1b", 123)])
    loaded = m.ModelWarmup.get_loaded_models(object())
    assert [x.name for x in loaded] == ["qwen3:1b"]
    assert loaded[0].size_vram == 123


def test_wrm01_get_loaded_models_handles_dict_ps():
    m = _load_warmup_isolated()
    m._ollama.ps = lambda: {"models": [{"name": "qwen3:7b", "size_vram": 456}]}
    loaded = m.ModelWarmup.get_loaded_models(object())
    assert [x.name for x in loaded] == ["qwen3:7b"]


def test_wrm01_get_loaded_models_empty_when_unavailable():
    m = _load_warmup_isolated()
    m.OLLAMA_AVAILABLE = False
    loaded = m.ModelWarmup.get_loaded_models(object())
    assert loaded == []
