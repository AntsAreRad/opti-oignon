"""S190 F3c -- rag_hybrid_search.py config wiring (RHS-03).

rag_hybrid_search.py is all-stdlib at module scope (the heavy deps rag_store /
chromadb are imported lazily inside methods), so it loads directly via
spec_from_file_location. `_load_config` does a local `import yaml`; we stub it
in sys.modules so the test feeds a non-default bm25 config without needing a
real rag.yaml or PyYAML.

RHS-03: config/rag.yaml [hybrid_search] bm25_k1 / bm25_b must be applied to the
        BM25 scorer. Before the fix they were loaded into the config dict but
        never wired to self._bm25, so a non-default value was silently ignored.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MOD_PATH = Path(__file__).resolve().parent.parent / "opti_oignon" / "rag_hybrid_search.py"


def _load_engine_module(hybrid_cfg):
    """Load rag_hybrid_search with a stubbed yaml returning hybrid_cfg."""
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda f: {"hybrid_search": hybrid_cfg}
    sys.modules["yaml"] = yaml_stub

    spec = importlib.util.spec_from_file_location("rag_hybrid_search_s190", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rag_hybrid_search_s190"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    for k in ("rag_hybrid_search_s190", "yaml"):
        sys.modules.pop(k, None)


def test_rhs03_bm25_params_wired_from_config():
    # config/rag.yaml exists in the repo, so config_path.exists() is True; the
    # stubbed yaml.safe_load supplies non-default bm25 params regardless.
    mod = _load_engine_module({"bm25_k1": 2.3, "bm25_b": 0.9})
    engine = mod.HybridSearchEngine()
    # Constructor defaults before config load.
    assert engine._bm25.k1 == 1.5
    assert engine._bm25.b == 0.75

    engine._load_config()

    # After loading config, the scorer must reflect the configured values.
    assert engine._bm25.k1 == 2.3, "bm25_k1 from rag.yaml must reach the scorer"
    assert engine._bm25.b == 0.9, "bm25_b from rag.yaml must reach the scorer"
    # And get_config reports the live scorer params.
    cfg = engine.get_config()
    assert cfg["bm25_k1"] == 2.3 and cfg["bm25_b"] == 0.9


def test_rhs03_defaults_preserved_when_config_matches():
    mod = _load_engine_module({"bm25_k1": 1.5, "bm25_b": 0.75})
    engine = mod.HybridSearchEngine()
    engine._load_config()
    # Behaviour unchanged when config matches the constructor defaults.
    assert engine._bm25.k1 == 1.5
    assert engine._bm25.b == 0.75


def test_rhs03_invalid_config_keeps_current_values():
    mod = _load_engine_module({"bm25_k1": "not-a-number", "bm25_b": None})
    engine = mod.HybridSearchEngine()
    engine._load_config()
    # A malformed value must not crash and must keep the current scorer params.
    assert engine._bm25.k1 == 1.5
    assert engine._bm25.b == 0.75
