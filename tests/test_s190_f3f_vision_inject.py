"""S190 F3f -- VIS-02: build_augmented_message must fall back on a malformed
inject_format template (including ValueError from a stray "{"), not crash.

vision_pipeline.py is all-stdlib at module scope (yaml / ollama / vision_config
are optionally imported with guards), so it loads via spec_from_file_location.
A non-existent config_path makes _load_delegation_config a no-op, keeping the
default inject_format; the test then injects malformed templates directly.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

MOD_PATH = Path(__file__).resolve().parent.parent / "opti_oignon" / "vision_pipeline.py"


@pytest.fixture()
def vp_mod():
    spec = importlib.util.spec_from_file_location("vision_pipeline_s190", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vision_pipeline_s190"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("vision_pipeline_s190", None)


def _pipeline(mod):
    # config_path points nowhere -> _load_delegation_config is a no-op, defaults kept.
    return mod.VisionPipeline(
        vision_config=None,
        ollama_module=object(),
        config_path=Path("/nonexistent/vision.yaml"),
    )


def test_vis02_stray_brace_template_falls_back(vp_mod):
    vp = _pipeline(vp_mod)
    vp._inject_format = "broken { template {description}"  # stray "{" -> ValueError
    out = vp.build_augmented_message("my question", "a cat on a mat")
    # Must not raise; must fall back and keep both pieces.
    assert "a cat on a mat" in out
    assert "my question" in out
    assert out.startswith("[Image analysis:")


def test_vis02_unknown_placeholder_falls_back(vp_mod):
    vp = _pipeline(vp_mod)
    vp._inject_format = "{nonexistent_key}"  # KeyError
    out = vp.build_augmented_message("q", "desc")
    assert "desc" in out and "q" in out


def test_vis02_positional_ref_falls_back(vp_mod):
    vp = _pipeline(vp_mod)
    vp._inject_format = "{0} {1}"  # IndexError (no positional args supplied)
    out = vp.build_augmented_message("q", "desc")
    assert "desc" in out and "q" in out


def test_vis02_valid_template_unchanged(vp_mod):
    vp = _pipeline(vp_mod)
    vp._inject_format = "IMG: {description} || Q: {message}"
    out = vp.build_augmented_message("what is this", "a diagram")
    assert out == "IMG: a diagram || Q: what is this"


def test_vis02_empty_description_returns_original(vp_mod):
    vp = _pipeline(vp_mod)
    out = vp.build_augmented_message("original message", "")
    assert out == "original message"
