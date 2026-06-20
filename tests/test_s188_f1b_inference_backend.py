#!/usr/bin/env python3
"""S188 phase F1-B: inference backend.

Source-level / AST assertions. ``inference_backend.py`` imports ollama and
(optionally) llama_cpp at module load, so the backend is inspected from its
source rather than imported.

Covers:
- the explicit F1 item: ``LlamaCppBackend._get_or_load_draft`` had zero callers
  (code and tests), an unused ``main_model_name`` parameter, and its return value
  was never consumed by ``generate`` / ``stream`` (the in-process llama-cpp-python
  path implements no draft-verify loop; the real speculative path for llama.cpp is
  the external llama-server, built by ``build_server_flags``). It was dead and is
  removed here. Its removal is internal: it was never executed, so there is no
  user-facing behaviour change.
- a regression guard that the IB-02 lock story still holds end to end: per-model
  double-checked load lock, lock-free fast-path hit, per-model inference lock.
"""

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
BACKEND = REPO / "opti_oignon" / "inference_backend.py"
SRC = BACKEND.read_text(encoding="utf-8")
TREE = ast.parse(SRC)  # also asserts the edited file parses


def _class(name: str) -> ast.ClassDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _has_method(cls_name: str, method: str) -> bool:
    cls = _class(cls_name)
    return any(
        isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == method
        for f in cls.body
    )


def _method_segment(cls_name: str, method: str) -> str:
    cls = _class(cls_name)
    for f in cls.body:
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == method:
            seg = ast.get_source_segment(SRC, f)
            assert seg is not None
            return seg
    raise AssertionError(f"{cls_name}.{method} not found")


def test_dead_draft_loader_removed():
    assert not _has_method("LlamaCppBackend", "_get_or_load_draft")


def test_removal_was_surgical():
    # only the dead draft loader was removed; the rest of LlamaCppBackend stands
    for m in ("_get_or_load", "generate", "stream", "build_server_flags",
              "unload_model", "unload_all", "set_speculative_config"):
        assert _has_method("LlamaCppBackend", m), m


def test_loaded_draft_models_now_has_no_writer():
    # the removed method held the only subscript assignment into the dict; it is
    # now vestigial (only init, clear, membership/del remain). Documents the
    # post-removal state and guards against a writer reappearing without wiring
    # an actual speculative path (which would be an F2 design item).
    assert re.search(r"self\._loaded_draft_models\[[^\]]+\]\s*=", SRC) is None


def test_ib02_load_lock_invariant_holds():
    seg = _method_segment("LlamaCppBackend", "_get_or_load")
    # per-model load lock + double-checked cache (fast-path get, then re-check
    # under the lock) so a model is constructed exactly once
    assert "with self._lock_for(self._load_locks, model_name):" in seg
    assert seg.count("self._loaded_models.get(model_name)") >= 2


def test_ib02_inference_lock_is_per_model():
    for m in ("generate", "stream"):
        seg = _method_segment("LlamaCppBackend", m)
        # a Llama instance is not concurrency-safe; calls are serialized per model
        assert "with self._lock_for(self._inference_locks, model):" in seg


def test_lock_guard_not_held_during_load_or_inference():
    # the locks_guard only brackets lock creation in _lock_for, never a load or
    # an inference call (so it cannot serialize the hot path)
    seg = _method_segment("LlamaCppBackend", "_lock_for")
    assert "with self._locks_guard:" in seg
    for m in ("_get_or_load", "generate", "stream"):
        assert "_locks_guard" not in _method_segment("LlamaCppBackend", m)
