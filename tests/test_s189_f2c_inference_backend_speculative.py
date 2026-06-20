"""S189 phase F2 -- IB-03 decision: the dead in-process speculative surface on
``LlamaCppBackend`` is removed.

This RE-ASSERTS the post-removal reality and supersedes
``tests/test_s188_f1b_inference_backend.py::test_removal_was_surgical`` (which asserted
``build_server_flags`` and ``set_speculative_config`` were still present). That S188 test
is left untouched and ``--deselect``-ed when this suite runs; see
``SESSION_TRACKING_S65_S189.md``.

Decision (axes a/b): the surface (`set_speculative_config`, `speculative_enabled`,
`get_speculative_info`, `build_server_flags`, the `_speculative_*` fields and
`_loaded_draft_models`) had zero production callers and duplicated
``speculative_decoding.SpeculativeDecodingManager`` (which owns the config and the live
``build_llama_cpp_flags``). It was removed wholesale rather than wired; wiring real
speculative inference is a future cycle driven from ``speculative_decoding``.

``inference_backend.py`` imports ollama / optional llama_cpp at load, so it is inspected
from source via AST (same approach as the S188 suite).
"""

import ast
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[1]
_BACKEND = _REPO / "opti_oignon" / "inference_backend.py"
_SRC = _BACKEND.read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)  # also asserts the edited file parses


def _class(name):
    for node in ast.walk(_TREE):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _has_method(cls_name, method):
    cls = _class(cls_name)
    return any(
        isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == method
        for f in cls.body
    )


def test_dead_speculative_methods_removed():
    for m in ("set_speculative_config", "speculative_enabled",
              "get_speculative_info", "build_server_flags"):
        assert not _has_method("LlamaCppBackend", m), m


def test_general_backend_methods_intact():
    for m in ("_get_or_load", "generate", "stream", "unload_model",
              "unload_all", "list_models", "_lock_for"):
        assert _has_method("LlamaCppBackend", m), m


def test_no_residual_speculative_state_references():
    for token in ("_loaded_draft_models", "_speculative_enabled",
                  "_speculative_draft_model", "_speculative_draft_max"):
        assert token not in _SRC, token


def test_live_flag_builder_lives_in_speculative_decoding():
    sd = (_REPO / "opti_oignon" / "speculative_decoding.py").read_text(encoding="utf-8")
    assert "def build_llama_cpp_flags(" in sd
