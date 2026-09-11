#!/usr/bin/env python3
"""Contract for the consensus engine's model queries through the registry.

Consensus queried each model through the client directly and reported
itself available whenever the client library imported. It now asks the
registry's backend for each query and is available exactly when the
registry can serve one.

  * CN1 -- a query reaches the registry's backend with its model, messages
    and temperature; the text comes back; availability follows the
    registry; and with no backend a query raises by name.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window over the registry bridge.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

_MSGS = [{"role": "user", "content": "2+2"}]


class _Scripted:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "  4  "}}


def _open(*, registry=True):
    seeded = {}
    scripted = _Scripted()
    if registry:
        seed_registry(seeded, scripted)
    loaded, restore = isolate(
        targets={"opti_oignon.consensus": source("consensus.py")},
        blocked=("opti_oignon.model_profiles",) + (() if registry else ("opti_oignon.inference_backend",)),
        seeded=seeded,
        packages=("opti_oignon",),
    )
    return loaded["opti_oignon.consensus"], scripted, restore


def test_cn1_a_query_goes_through_the_registry_and_availability_follows_it():
    mod, scripted, restore = _open()
    try:
        engine = mod.ConsensusEngine(config=mod.ConsensusConfig())
        assert engine.available is True
        assert engine._call_llm(_MSGS, "m", temperature=0.4) == "4"
        call = scripted.calls[0]
        assert call["model"] == "m" and call["messages"] == _MSGS and call["options"]["temperature"] == 0.4
        assert "import ollama" not in source("consensus.py").read_text(encoding="utf-8")
    finally:
        restore()
    mod, scripted, restore = _open(registry=False)
    try:
        engine = mod.ConsensusEngine(config=mod.ConsensusConfig())
        assert engine.available is False
        with pytest.raises(RuntimeError, match="registry"):
            engine._call_llm(_MSGS, "m")
        assert scripted.calls == []
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
