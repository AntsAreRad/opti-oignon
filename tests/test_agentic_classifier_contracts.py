#!/usr/bin/env python3
"""Contracts for the request classifier and the tool-model default.

The pipeline selector runs a keyword heuristic over the user message before
any model call. These contracts pin two properties of that seam:

  * Contract 1 -- accented French phrasings trigger the same detections as
    their unaccented keyword entries (web, reasoning, tools), including the
    typographic apostrophe.
  * Contract 2 -- ASCII and English phrasings keep triggering (no regression).
  * Contract 3 -- when no default model is passed, the tool executor resolves
    it from the configuration instead of a hardcoded name, and falls back to
    the legacy name only when the configuration seam is absent.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Modules load in isolation with stubbed
heavy dependencies.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


def _pydantic_shim() -> types.ModuleType:
    mod = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for name in getattr(self.__class__, "__annotations__", {}):
                default = getattr(self.__class__, name, None)
                if isinstance(default, (list, dict)):
                    default = type(default)(default)
                setattr(self, name, default)
            for key, value in kwargs.items():
                setattr(self, key, value)

    mod.BaseModel = BaseModel
    return mod


_KEYS = (
    "pydantic", "ollama", "opti_oignon", "opti_oignon.tool_calling",
    "opti_oignon.tool_registry", "opti_oignon.structured_output",
    "opti_oignon.response_hygiene", "opti_oignon.tool_executor",
    "opti_oignon.agentic_executor", "opti_oignon.config",
)


def _snapshot():
    return {k: sys.modules.get(k) for k in _KEYS}


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _prime_package():
    try:
        import pydantic  # noqa: F401
    except ImportError:
        sys.modules["pydantic"] = _pydantic_shim()
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(
        f"opti_oignon.{name}", _OO / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"opti_oignon.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_agentic():
    """The classifier module alone; every heavy import degrades by design.

    The hygiene helper is preloaded when present so the nominal
    normalization path is the one under contract (its absence degrades to
    plain lowercasing by design, which these contracts do not pin).
    """
    _prime_package()
    rh_path = _OO / "response_hygiene.py"
    if rh_path.exists():
        _load_module("response_hygiene", "response_hygiene.py")
    return _load_module("agentic_executor", "agentic_executor.py")


def _load_tool_executor(config_stub):
    """The tool executor with a controllable configuration seam."""
    _prime_package()
    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **kw: None
    sys.modules["ollama"] = ollama_stub

    _load_module("tool_calling", "tool_calling.py")

    reg = types.ModuleType("opti_oignon.tool_registry")
    reg.ToolRegistry = object
    reg.tool_registry = None
    sys.modules["opti_oignon.tool_registry"] = reg

    so = types.ModuleType("opti_oignon.structured_output")
    so.StructuredOutputEngine = object
    so.structured_engine = None
    so.STRUCTURED_OUTPUT_AVAILABLE = False
    sys.modules["opti_oignon.structured_output"] = so

    rh_path = _OO / "response_hygiene.py"
    if rh_path.exists():
        _load_module("response_hygiene", "response_hygiene.py")

    if config_stub is not None:
        sys.modules["opti_oignon.config"] = config_stub
    else:
        sys.modules.pop("opti_oignon.config", None)

    return _load_module("tool_executor", "tool_executor.py")


# ---------------------------------------------------------------------------
# Contract 1: accented French phrasings trigger detection
# ---------------------------------------------------------------------------
def test_classifier_matches_accented_french():
    saved = _snapshot()
    try:
        ae = _load_agentic()
        c = ae._quick_classify("Quelle est l'actualité aujourd'hui ?")
        assert c["needs_web"] is True, c
        # Typographic apostrophe variant.
        c2 = ae._quick_classify("Quelle est l\u2019actualit\u00e9 du jour ?")
        assert c2["needs_web"] is True, c2
        c3 = ae._quick_classify(
            "Explique-moi \u00e9tape par \u00e9tape la strat\u00e9gie \u00e0 suivre")
        assert c3["needs_reasoning"] is True, c3
        assert c3["is_complex"] is True, c3
        c4 = ae._quick_classify("Ex\u00e9cute ce code s'il te pla\u00eet")
        assert c4["needs_tools"] is True, c4
    finally:
        _restore(saved)


# ---------------------------------------------------------------------------
# Contract 2: ASCII / English phrasings keep triggering
# ---------------------------------------------------------------------------
def test_classifier_ascii_paths_unchanged():
    saved = _snapshot()
    try:
        ae = _load_agentic()
        assert ae._quick_classify("search the latest news")["needs_web"] is True
        assert ae._quick_classify(
            "compare the pros and cons of both designs")["is_complex"] is True
        assert ae._quick_classify(
            "break down the plan step by step")["needs_reasoning"] is True
        neutral = ae._quick_classify("Bonjour, tout va bien.")
        assert neutral["needs_web"] is False
        assert neutral["needs_tools"] is False
    finally:
        _restore(saved)


# ---------------------------------------------------------------------------
# Contract 3: the default tool model resolves from configuration
# ---------------------------------------------------------------------------
def test_default_model_resolves_from_configuration():
    saved = _snapshot()
    try:
        cfg = types.ModuleType("opti_oignon.config")
        calls = []

        def get_model(model_type, priority="primary"):
            calls.append((model_type, priority))
            return "configured-model:latest"

        cfg.get_model = get_model
        te = _load_tool_executor(cfg)
        ex = te.ToolExecutor(registry=None, structured_engine=None)
        assert ex.default_model == "configured-model:latest", ex.default_model
        assert calls, "configuration seam was never consulted"

        # An explicit model always wins over the configuration.
        ex2 = te.ToolExecutor(
            registry=None, structured_engine=None, default_model="explicit:1",
        )
        assert ex2.default_model == "explicit:1"
    finally:
        _restore(saved)

    saved = _snapshot()
    try:
        # Configuration seam absent: the legacy fallback name holds.
        te = _load_tool_executor(None)
        ex = te.ToolExecutor(registry=None, structured_engine=None)
        assert ex.default_model == "qwen3:32b", ex.default_model
    finally:
        _restore(saved)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print("-" * 48)
    print(f"{len(tests)} selected, {failed} failed")
    sys.exit(1 if failed else 0)
