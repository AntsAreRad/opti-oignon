"""S189 phase F2 -- Cascading & consensus (item 4) regression test.

Covers the one applied fix:

- CNS-01: ``consensus.ConsensusEngine._call_llm`` parsed the ollama response with
  ``response.get("message", {})``, assuming a dict. Newer ollama clients return a
  ``ChatResponse`` object exposing ``.message.content``; on that form the dict-only
  access raised ``AttributeError`` (swallowed by ``_query_model`` -> every consensus
  model query failed). It now handles both forms, mirroring
  ``CascadingInference._call_llm``.

``consensus`` is loaded in isolation with stubbed ``ollama`` and
``opti_oignon.model_profiles`` (all of consensus's imports are guarded), registered as
``opti_oignon.consensus`` before ``exec_module`` per the test-loader idiom. ``_call_llm``
uses no instance state, so it is invoked with a dummy ``self``.
"""

import importlib.util
import pathlib
import sys
import types

_REPO = pathlib.Path(__file__).resolve().parents[1]
_CONSENSUS = _REPO / "opti_oignon" / "consensus.py"


def _load_consensus_isolated():
    # Stub the only third-party / relative imports consensus pulls.
    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **kwargs: {"message": {"content": ""}}
    sys.modules["ollama"] = ollama_stub

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []  # mark as a package so the relative import resolves
    sys.modules.setdefault("opti_oignon", pkg)

    profiles_stub = types.ModuleType("opti_oignon.model_profiles")
    profiles_stub.profile_manager = None
    profiles_stub.ModelProfile = None
    sys.modules["opti_oignon.model_profiles"] = profiles_stub

    spec = importlib.util.spec_from_file_location("opti_oignon.consensus", _CONSENSUS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.consensus"] = module  # register before exec (dataclasses)
    spec.loader.exec_module(module)
    return module


class _ObjMessage:
    def __init__(self, content):
        self.content = content


class _ObjResponse:
    """Mimics a newer ollama ChatResponse object (no dict .get)."""

    def __init__(self, content):
        self.message = _ObjMessage(content)


def test_call_llm_handles_object_response():
    module = _load_consensus_isolated()
    assert module.OLLAMA_AVAILABLE is True
    module.ollama.chat = lambda **kwargs: _ObjResponse("  object-form answer  ")
    out = module.ConsensusEngine._call_llm(
        object(), messages=[{"role": "user", "content": "q"}], model="m"
    )
    assert out == "object-form answer"


def test_call_llm_handles_dict_response():
    module = _load_consensus_isolated()
    module.ollama.chat = lambda **kwargs: {"message": {"content": "  dict-form answer  "}}
    out = module.ConsensusEngine._call_llm(
        object(), messages=[{"role": "user", "content": "q"}], model="m"
    )
    assert out == "dict-form answer"


def test_call_llm_handles_object_with_dict_message():
    module = _load_consensus_isolated()

    class _ObjDictMsg:
        message = {"content": "mixed answer"}

    module.ollama.chat = lambda **kwargs: _ObjDictMsg()
    out = module.ConsensusEngine._call_llm(
        object(), messages=[{"role": "user", "content": "q"}], model="m"
    )
    assert out == "mixed answer"


def test_call_llm_empty_content_is_safe():
    module = _load_consensus_isolated()
    module.ollama.chat = lambda **kwargs: _ObjResponse(None)
    out = module.ConsensusEngine._call_llm(
        object(), messages=[{"role": "user", "content": "q"}], model="m"
    )
    assert out == ""
