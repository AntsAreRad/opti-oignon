"""S189 phase F2 -- Speculative decoding (item 5) regression tests.

Covers the two applied fixes in the live S70 path (speculative.py):

- SPD-01: ``_call_draft`` / ``_call_verify`` parsed the ollama response with
  ``response["message"]["content"]`` (dict-only); the object-form ``ChatResponse``
  raised, which ``generate()`` swallowed as a failed draft/verify. A shared
  ``_extract_message_content`` now handles both forms.
- SPD-02: on acceptance ``generate()`` set ``final_response = draft_response`` (the
  original phase-1 draft) even when convergence happened on a later iteration, where the
  original draft had already been replaced. It now serves ``current_draft`` (the converged
  text) -- identical when convergence is on iteration 1, correct otherwise.

``speculative`` is loaded in isolation: only ``opti_oignon`` is stubbed (its
``self_correction`` and ``ollama`` imports are guarded and allowed to fail). ``generate``
is driven via injected ``draft_call`` / ``verify_call`` callables, so no ollama is needed.

The IB-03 removal of the dead backend speculative surface is asserted in
``test_s189_f2c_inference_backend_speculative.py``; not duplicated here.
"""

import importlib.util
import pathlib
import sys
import types

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SPEC = _REPO / "opti_oignon" / "speculative.py"


def _load_speculative_isolated():
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules.setdefault("opti_oignon", pkg)
    spec = importlib.util.spec_from_file_location("opti_oignon.speculative", _SPEC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.speculative"] = module
    spec.loader.exec_module(module)
    return module


class _ObjMsg:
    def __init__(self, content):
        self.content = content


class _ObjResp:
    def __init__(self, content):
        self.message = _ObjMsg(content)


def test_extract_content_dict_and_object():
    m = _load_speculative_isolated()
    assert m._extract_message_content({"message": {"content": "d"}}) == "d"
    assert m._extract_message_content(_ObjResp("o")) == "o"

    class _ObjDictMsg:
        message = {"content": "mixed"}

    assert m._extract_message_content(_ObjDictMsg()) == "mixed"
    assert m._extract_message_content(_ObjResp(None)) == ""
    assert m._extract_message_content(object()) == ""


def test_accept_serves_converged_text_not_original_draft():
    m = _load_speculative_isolated()
    gen = m.SpeculativeGenerator(config_path=None)
    # Force a multi-iteration run with a high threshold.
    gen._max_iterations = 3
    gen._convergence_threshold = 0.9
    gen._draft_model = "draft:1b"
    gen._verify_model = "verify:8b"

    original = "alpha beta gamma delta epsilon zeta"
    refined = "lorem ipsum dolor amet consectetur adipiscing"  # disjoint -> low sim

    def draft_call(query):
        return original

    def verify_call(query, draft):
        # iteration 1 (draft == original): diverge to `refined`
        # iteration 2 (draft == refined): reproduce it -> convergence == 1.0
        return refined if draft == original else draft

    result = gen.generate("q", draft_call=draft_call, verify_call=verify_call)

    assert result.draft_accepted is True
    assert result.iterations == 2
    assert result.draft_response == original
    # The fix: serve the converged text, not the rejected original draft.
    assert result.final_response == refined


def test_iteration1_acceptance_still_serves_the_draft():
    m = _load_speculative_isolated()
    gen = m.SpeculativeGenerator(config_path=None)
    gen._max_iterations = 3
    gen._convergence_threshold = 0.5
    gen._draft_model = "draft:1b"
    gen._verify_model = "verify:8b"

    draft_text = "the quick brown fox jumps over"

    def draft_call(query):
        return draft_text

    def verify_call(query, draft):
        return draft_text  # identical -> convergence 1.0 on iteration 1

    result = gen.generate("q", draft_call=draft_call, verify_call=verify_call)
    assert result.draft_accepted is True
    assert result.iterations == 1
    # current_draft == draft_response on iteration 1, so behaviour is preserved.
    assert result.final_response == draft_text
