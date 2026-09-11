#!/usr/bin/env python3
"""Contracts for the executor's two seams into the onion memory.

Reading: when the onion is enabled and the librarian has a block for the
conversation, that block is what reaches the prompt, inside the untrusted
envelope, and the working-memory composer of today is not consulted; when
the onion is off, or has nothing, today's path runs unchanged. Writing:
after a turn is saved, the librarian is offered the conversation beside the
auto-capture, and a librarian that is absent or failing never breaks the
turn. The librarian itself is a fake seam here; its own behaviour has its
own contracts.

  * XW1 -- onion on with a block: the block arrives wrapped as memory, once,
    and the legacy composer is not consulted.
  * XW2 -- onion off: the librarian is never asked and the legacy block
    arrives as before.
  * XW3 -- onion on with nothing: the legacy block arrives; fail-closed to
    the path that exists.
  * XW4 -- after the turn is saved, the librarian is offered the
    conversation with its id and messages.
  * XW5 -- a librarian that is absent leaves the executor importable and the
    legacy path intact; one that raises is swallowed.

Local-only (the public distribution ships no tests). Same window as the
memory-wrap suite: the real hub over a scripted registry.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

_EXECUTOR = "opti_oignon.executor"
_WRAPPER = "opti_oignon.agent.untrusted_context"
_LIBRARIAN = "opti_oignon.memory.librarian"
_LEGACY = "Remembered facts:\n- the user tends their onion garden at dawn"
_ONION = "[data layer=core]\nThe user is called Alice.\n[/data]"


class _Scripted:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return iter([{"message": {"content": "Hello"}}, {"message": {"content": " world"}, "done": True}])


class _Composer:
    def __init__(self, block):
        self.block, self.calls = block, 0

    def build(self, question, **kwargs):
        self.calls += 1
        return self.block


class _Librarian:
    """The fake seam: counts what the hub asks of it."""

    def __init__(self, *, enabled, block, raise_on_block=False, raise_on_curate=False):
        self.enabled, self.block = enabled, block
        self.block_calls, self.curate_calls, self.enabled_calls = [], [], 0
        self.raise_on_block, self.raise_on_curate = raise_on_block, raise_on_curate

    def onion_enabled(self, path=None):
        self.enabled_calls += 1
        return self.enabled

    def memory_block(self, conversation_id, question=None, **kwargs):
        self.block_calls.append((conversation_id, question))
        if self.raise_on_block:
            raise RuntimeError("librarian down")
        return self.block

    def maybe_curate(self, conversation_id, messages, **kwargs):
        self.curate_calls.append((conversation_id, list(messages or [])))
        if self.raise_on_curate:
            raise RuntimeError("librarian down")
        return True


class _Conversations:
    """A conversation store that remembers what it was handed."""

    def __init__(self):
        self.messages = {}

    def add_message(self, conversation_id, role, content, **kwargs):
        self.messages.setdefault(conversation_id, []).append({"role": role, "content": content})

    def get_context_messages(self, conversation_id, **kwargs):
        return list(self.messages.get(conversation_id, []))

    def get_conversation(self, conversation_id):
        return SimpleNamespace(id=conversation_id, messages=self.get_context_messages(conversation_id), metadata={})

    def update_conversation_metadata(self, conversation_id, *args, **kwargs):
        return None


def _routing(**overrides):
    fields = {"model": "test-model:1b", "task_type": "general", "temperature": 0.2, "prompt_variant": "standard", "timeout": 30}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _load(*, librarian=None, librarian_absent=False):
    composer = _Composer(_LEGACY)
    scripted = _Scripted()
    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = scripted.chat
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(get_model=lambda *a, **k: "test-model:1b", get_temperature=lambda *a, **k: 0.2)
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    retrieval = types.ModuleType("opti_oignon.memory.retrieval")
    retrieval.build_memory_block = composer.build
    retrieval.working_memory_block = composer.build
    conversations = _Conversations()
    conversation = types.ModuleType("opti_oignon.conversation")
    conversation.conversation_manager = conversations
    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.memory.retrieval": retrieval,
        "opti_oignon.conversation": conversation,
    }
    blocked = []
    if librarian_absent:
        blocked.append(_LIBRARIAN)
    else:
        module = types.ModuleType(_LIBRARIAN)
        module.onion_enabled = librarian.onion_enabled
        module.memory_block = librarian.memory_block
        module.maybe_curate = librarian.maybe_curate
        seeded[_LIBRARIAN] = module
    targets = {
        _WRAPPER: source("agent", "untrusted_context.py"),
        "opti_oignon.context_dedup": source("context_dedup.py"),
        _EXECUTOR: source("executor.py"),
    }
    seed_registry(seeded, scripted)
    had, prev = "ollama" in sys.modules, sys.modules.get("ollama")
    sys.modules["ollama"] = ollama_stub
    loaded, win_restore = isolate(targets=targets, blocked=blocked, seeded=seeded,
                                  packages=("opti_oignon.agent", "opti_oignon.memory"))

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    return loaded[_EXECUTOR], loaded[_WRAPPER], scripted, composer, conversations, restore


def _drive(gen):
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def _system(scripted):
    assert scripted.calls, "the request must reach the scripted registry"
    system = [m for m in scripted.calls[0]["messages"] if m.get("role") == "system"]
    assert system
    return system[0]["content"]


# ---------------------------------------------------------------------------
# XW1 -- onion on, with a block
# ---------------------------------------------------------------------------
def test_xw1_the_onion_block_arrives_wrapped_and_the_legacy_composer_is_not_consulted():
    lib = _Librarian(enabled=True, block=_ONION)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        content = _system(scripted)
        assert wrapper.wrap(_ONION, source=wrapper.SOURCE_MEMORY) in content
        assert content.count("The user is called Alice.") == 1
        assert _LEGACY not in content
        assert composer.calls == 0, "today's composer is not consulted when the onion answers"
        assert lib.block_calls == [("conv-1", "What is my name?")]
    finally:
        restore()


# ---------------------------------------------------------------------------
# XW2 -- onion off
# ---------------------------------------------------------------------------
def test_xw2_with_the_onion_off_the_librarian_is_never_asked():
    lib = _Librarian(enabled=False, block=_ONION)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        content = _system(scripted)
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in content
        assert _ONION not in content
        assert lib.block_calls == [] and lib.enabled_calls >= 1
        assert composer.calls == 1
        assert lib.curate_calls == [], "off means off on the write side too"
    finally:
        restore()


# ---------------------------------------------------------------------------
# XW3 -- onion on, with nothing
# ---------------------------------------------------------------------------
def test_xw3_with_the_onion_on_but_empty_the_legacy_block_arrives():
    lib = _Librarian(enabled=True, block="")
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        content = _system(scripted)
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in content
        assert lib.block_calls == [("conv-1", "What is my name?")]
        assert composer.calls == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# XW4 -- the librarian is offered the saved conversation
# ---------------------------------------------------------------------------
def test_xw4_after_the_turn_the_librarian_is_offered_the_conversation():
    lib = _Librarian(enabled=True, block=_ONION)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        assert conversations.messages.get("conv-1"), "control: the turn was saved"
        assert len(lib.curate_calls) == 1
        cid, messages = lib.curate_calls[0]
        assert cid == "conv-1"
        assert messages == conversations.get_context_messages("conv-1")
        assert messages[-1]["role"] == "assistant" and messages[-1]["content"] == "Hello world"
    finally:
        restore()


# ---------------------------------------------------------------------------
# XW5 -- absent or failing librarian
# ---------------------------------------------------------------------------
def test_xw5_an_absent_or_failing_librarian_never_breaks_the_turn():
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian_absent=True)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in _system(scripted)
        assert composer.calls == 1
    finally:
        restore()
    lib = _Librarian(enabled=True, block=_ONION, raise_on_block=True, raise_on_curate=True)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in _system(scripted)
        assert lib.block_calls and lib.curate_calls, "both seams were tried and both failures were swallowed"
        assert conversations.messages.get("conv-1")
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
