#!/usr/bin/env python3
"""Contract for the agent-eval runner's chat client through the registry.

The eval runner drove the agent loop with a client built on the model
library directly. It now asks the registry's backend for each turn -- the
tool schemas as an engine option -- and hands the loop one chunk in the
shape it reads, content and tool calls included. With no backend the turn
raises by name rather than reaching for a client.

  * ER1 -- a turn reaches the registry's backend with the messages and
    tools, comes back as one loop-shaped chunk, and no backend is a
    refusal by name.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window over the registry bridge with the runner's
neighbours declared unreachable.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

_MSGS = [{"role": "user", "content": "create a file"}]
_TOOLS = [{"type": "function", "function": {"name": "create_file", "parameters": {}}}]


class _Scripted:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "on it", "tool_calls": [
            SimpleNamespace(function=SimpleNamespace(name="create_file", arguments={"path": "a.txt"}))
        ]}}


def _open(*, registry=True):
    store = types.ModuleType("opti_oignon.agent_eval.store")
    store.FAILURE_CLASSES = ("ok",)
    store.EvalResultsStore = object
    seeded = {"opti_oignon.agent_eval.store": store}
    scripted = _Scripted()
    if registry:
        seed_registry(seeded, scripted)
    blocked = ("opti_oignon.agent", "opti_oignon.file_tools", "opti_oignon.sandbox_tools", "opti_oignon.resource_governor")
    if not registry:
        blocked += ("opti_oignon.inference_backend",)
    loaded, restore = isolate(
        targets={
            "opti_oignon.agent_eval.tasks": source("agent_eval", "tasks.py"),
            "opti_oignon.agent_eval.runner": source("agent_eval", "runner.py"),
        },
        blocked=blocked,
        seeded=seeded,
        packages=("opti_oignon", "opti_oignon.agent_eval"),
    )
    return loaded["opti_oignon.agent_eval.runner"], scripted, restore


def test_er1_a_turn_goes_through_the_registry_and_comes_back_loop_shaped():
    mod, scripted, restore = _open()
    try:
        client = mod.RegistryChatClient("m")
        chunks = list(client.stream(_MSGS, _TOOLS))
        assert len(chunks) == 1
        message = chunks[0]["message"]
        assert message["content"] == "on it"
        assert message["tool_calls"][0].function.name == "create_file"
        call = scripted.calls[0]
        assert call["model"] == "m" and call["messages"] == _MSGS and call["tools"] == _TOOLS
        assert "import ollama" not in source("agent_eval", "runner.py").read_text(encoding="utf-8")
        assert mod.OllamaChatClient is mod.RegistryChatClient, "the exported name stays, as an alias"
        assert mod._default_client_factory("m").__class__ is mod.RegistryChatClient
    finally:
        restore()
    mod, scripted, restore = _open(registry=False)
    try:
        with pytest.raises(RuntimeError, match="registry"):
            list(mod.RegistryChatClient("m").stream(_MSGS, _TOOLS))
        assert scripted.calls == []
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
