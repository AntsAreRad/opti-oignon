"""S189 phase F2 -- System presets & presets (item 8) regression test.

Covers the one applied fix:

- PRS-02: ``SystemPresetsManager.detect_and_recommend`` reached the final ``else``
  ("Could not determine model sizes") only for a single large model (every detected model
  is categorized, so all-zero counts is impossible with models present), giving a wrong
  reason. A dedicated ``counts["large"] == 1`` branch now gives a correct reason; the tier
  is unchanged (balanced). PRS-01 (recommend ignores hardware) is recorded, not fixed.

``system_presets`` imports ``.config``; it is loaded in isolation with a stubbed
``opti_oignon.config``, and ``detect_ollama_models`` is monkeypatched (``detect_and_recommend``
uses module-level ``detect_ollama_models`` and ``ModelInfo.size_category`` only, no instance
state, so it is called with a dummy ``self``).
"""

import importlib.util
import pathlib
import sys
import types

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SP = _REPO / "opti_oignon" / "system_presets.py"


def _load_system_presets_isolated():
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules.setdefault("opti_oignon", pkg)

    cfg = types.ModuleType("opti_oignon.config")
    cfg.CONFIG_DIR = pathlib.Path("/tmp")
    cfg.DATA_DIR = pathlib.Path("/tmp")
    cfg.load_yaml = lambda p: {}
    cfg.save_yaml = lambda p, d: True
    sys.modules["opti_oignon.config"] = cfg

    spec = importlib.util.spec_from_file_location("opti_oignon.system_presets", _SP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.system_presets"] = module
    spec.loader.exec_module(module)
    return module


def _recommend_for(module, models):
    module.detect_ollama_models = lambda: models
    return module.SystemPresetsManager.detect_and_recommend(object())


def test_prs02_single_large_model_has_correct_reason():
    m = _load_system_presets_isolated()
    models = [m.ModelInfo(name="big:70b", parameter_count_b=70.0)]
    result = _recommend_for(m, models)
    # Tier unchanged from the prior else-fallthrough behaviour.
    assert result.recommended_preset == "balanced"
    # The misleading "could not determine sizes" reason no longer applies here.
    assert "could not determine" not in result.reason.lower()
    assert "large" in result.reason.lower()


def test_two_large_models_recommend_power():
    m = _load_system_presets_isolated()
    models = [
        m.ModelInfo(name="big:70b", parameter_count_b=70.0),
        m.ModelInfo(name="huge:40b", parameter_count_b=40.0),
    ]
    result = _recommend_for(m, models)
    assert result.recommended_preset == "power"


def test_medium_model_recommends_balanced():
    m = _load_system_presets_isolated()
    models = [m.ModelInfo(name="mid:14b", parameter_count_b=14.0)]
    result = _recommend_for(m, models)
    assert result.recommended_preset == "balanced"


def test_small_model_recommends_minimal():
    m = _load_system_presets_isolated()
    models = [m.ModelInfo(name="tiny:3b", parameter_count_b=3.0)]
    result = _recommend_for(m, models)
    assert result.recommended_preset == "minimal"


def test_no_models_recommends_minimal():
    m = _load_system_presets_isolated()
    result = _recommend_for(m, [])
    assert result.recommended_preset == "minimal"
