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
  * XW11 -- onion on: before it asks for the block, the turn hands the
    librarian its own counter, for the model it is sent to, so the block is
    counted as the request it rides; nothing is handed over with the onion
    off or with no conversation, and a seam that fails leaves the legacy
    block riding the turn.
  * XW12 -- onion off, the request the executor sends is the one it sent
    before the onion window had rungs, byte for byte: its digest was taken
    on the tree before the change.

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


def _load(*, librarian=None, librarian_absent=False, conversations=None):
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
    if conversations is None:
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
        if hasattr(librarian, "use_counter"):
            module.use_counter = librarian.use_counter
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


# ---------------------------------------------------------------------------
# The block rides the user turn. XW6 to XW9 keep what XW1, XW2, XW3 and XW5
# pinned, read where the block now arrives: in front of the question, in the
# user role. The onion's window keeps the frames its composer wrote.
# ---------------------------------------------------------------------------
def _turn(scripted):
    assert scripted.calls, "the request must reach the scripted registry"
    messages = scripted.calls[0]["messages"]
    assert messages and messages[-1].get("role") == "user"
    return messages[-1]["content"]


def test_xw6_the_onion_block_arrives_in_the_turn_with_its_frames_and_the_legacy_composer_is_not_consulted():
    lib = _Librarian(enabled=True, block=_ONION)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        turn = _turn(scripted)
        assert wrapper.wrap(_ONION, source=wrapper.SOURCE_MEMORY, frames=True) in turn
        assert turn.count("The user is called Alice.") == 1
        assert _LEGACY not in turn and "The user is called Alice." not in _system(scripted)
        assert composer.calls == 0, "today's composer is not consulted when the onion answers"
        assert lib.block_calls == [("conv-1", "What is my name?")]
    finally:
        restore()


def test_xw7_with_the_onion_off_the_librarian_is_never_asked_and_the_legacy_block_rides_the_turn():
    lib = _Librarian(enabled=False, block=_ONION)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        turn = _turn(scripted)
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in turn
        assert "The user is called Alice." not in turn
        assert lib.block_calls == [] and lib.enabled_calls >= 1
        assert composer.calls == 1
        assert lib.curate_calls == [], "off means off on the write side too"
    finally:
        restore()


def test_xw8_with_the_onion_on_but_empty_the_legacy_block_rides_the_turn():
    lib = _Librarian(enabled=True, block="")
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in _turn(scripted)
        assert lib.block_calls == [("conv-1", "What is my name?")]
        assert composer.calls == 1
    finally:
        restore()


def test_xw9_an_absent_or_failing_librarian_never_breaks_the_turn_and_the_legacy_block_rides_it():
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian_absent=True)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in _turn(scripted)
        assert composer.calls == 1
    finally:
        restore()
    lib = _Librarian(enabled=True, block=_ONION, raise_on_block=True, raise_on_curate=True)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in _turn(scripted)
        assert lib.block_calls and lib.curate_calls, "both seams were tried and both failures were swallowed"
        assert conversations.messages.get("conv-1")
    finally:
        restore()


# ---------------------------------------------------------------------------
# XW10 supersedes XW9. The librarian now mirrors the conversation through the
# store's origin read, which the store XW9 seeds does not carry, so its
# failing librarian was never offered the turn. Same properties over the
# store as it stands. XW4 is superseded by OT16, in the turn-origin suite.
# ---------------------------------------------------------------------------
class _OriginConversations(_Conversations):
    """The store as it stands: the mirror reads who wrote each turn."""

    def get_mirror_messages(self, conversation_id):
        return [dict(m, origin="legacy", segments=[]) for m in self.get_context_messages(conversation_id)]


def test_xw10_an_absent_or_failing_librarian_never_breaks_the_turn_and_the_legacy_block_rides_it():
    mod, wrapper, scripted, composer, conversations, restore = _load(
        librarian_absent=True, conversations=_OriginConversations())
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in _turn(scripted)
        assert composer.calls == 1
        assert conversations.messages.get("conv-1"), "the turn is saved with no librarian at all"
    finally:
        restore()
    lib = _Librarian(enabled=True, block=_ONION, raise_on_block=True, raise_on_curate=True)
    mod, wrapper, scripted, composer, conversations, restore = _load(
        librarian=lib, conversations=_OriginConversations())
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in _turn(scripted)
        assert lib.block_calls and lib.curate_calls, "both seams were tried and both failures were swallowed"
        assert conversations.messages.get("conv-1")
    finally:
        restore()


# ---------------------------------------------------------------------------
# XW11 -- the turn hands the onion its own counter
# ---------------------------------------------------------------------------
class _CountingLibrarian(_Librarian):
    """The fake seam with the counter seam: it keeps what each turn hands over, in order with the block."""

    def __init__(self, *, raise_on_counter=False, **kwargs):
        super().__init__(**kwargs)
        self.counters, self.order, self.raise_on_counter = [], [], raise_on_counter

    def use_counter(self, conversation_id, count, model):
        self.counters.append((conversation_id, count, model))
        self.order.append("counter")
        if self.raise_on_counter:
            raise TypeError("counter refused")

    def memory_block(self, conversation_id, question=None, **kwargs):
        self.order.append("block")
        return super().memory_block(conversation_id, question, **kwargs)


_COUNTED = ("Alice tends her garden at dawn.", "def f(x):\n    return {'k': [x, x]}\n",
            "Le jardin d'Alice fleurit \u00e0 l'aube, pr\u00e8s du vieux mur.", "x" * 97)


def test_xw11_the_turn_hands_the_onion_its_own_counter_for_the_model_it_is_sent_to():
    lib = _CountingLibrarian(enabled=True, block=_ONION)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        # The executor's own count, read for the model the turn is sent to: a
        # third more for this one than for any other.
        ex._estimate_tokens = lambda text, model="": len(text) // (3 if model == "counted-model:7b" else 4)
        _drive(ex.execute("What is my name?", _routing(model="counted-model:7b"), refine=False,
                          conversation_id="conv-1"))
        assert lib.order == ["counter", "block"], "the counter is handed over before the block is asked"
        ((cid, count, model),) = lib.counters
        assert (cid, model) == ("conv-1", "counted-model:7b")
        for text in _COUNTED:
            assert count(text) == ex._estimate_tokens(text, "counted-model:7b") == len(text) // 3, (
                f"the turn's own count, for the model it is sent to: {text!r}")
        assert any(count(text) != ex._estimate_tokens(text) for text in _COUNTED), (
            "control: a count blind to the model gives other numbers")
        assert any(count(text) != max(1, int(len(text.split()) * 1.3)) for text in _COUNTED), (
            "control: the turn's count is not the onion's default")
        assert wrapper.wrap(_ONION, source=wrapper.SOURCE_MEMORY, frames=True) in _turn(scripted)
    finally:
        restore()
    for enabled, conversation_id in ((False, "conv-1"), (True, None)):
        lib = _CountingLibrarian(enabled=enabled, block=_ONION)
        mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
        try:
            _drive(mod.Executor().execute("What is my name?", _routing(), refine=False, conversation_id=conversation_id))
            assert lib.counters == [], f"nothing handed over (onion {'on' if enabled else 'off'}, {conversation_id})"
            assert lib.order == (["block"] if enabled else []), lib.order
        finally:
            restore()
    lib = _CountingLibrarian(enabled=True, block=_ONION, raise_on_counter=True)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        _drive(mod.Executor().execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        assert lib.counters and "block" not in lib.order, "a seam that fails: no block counted another way"
        assert wrapper.wrap(_LEGACY, source=wrapper.SOURCE_MEMORY) in _turn(scripted), "the legacy block rides it"
        assert conversations.messages.get("conv-1"), "and the turn is saved"
    finally:
        restore()


# ---------------------------------------------------------------------------
# XW12 -- onion off, the request of before, byte for byte
# ---------------------------------------------------------------------------
# The digest of what the executor sent, onion off, for the two turns below,
# taken on the tree before the onion window had rungs and one count.
_BEFORE_RUNGS = "d1780f443e749215b4957c5281c645c9025d050a6f18c21687bdcbc9e574742a"


def _requests_onion_off():
    """What the executor sends, onion off, for two turns of one conversation, as canonical JSON."""
    import json

    lib = _CountingLibrarian(enabled=False, block=_ONION)
    mod, wrapper, scripted, composer, conversations, restore = _load(librarian=lib)
    try:
        ex = mod.Executor()
        _drive(ex.execute("What is my name?", _routing(), refine=False, conversation_id="conv-1"))
        _drive(ex.execute("And where does the garden grow?", _routing(), refine=False, conversation_id="conv-1"))
        assert len(scripted.calls) == 2 and lib.counters == [] and lib.order == []
        return json.dumps(scripted.calls, sort_keys=True, ensure_ascii=False)
    finally:
        restore()


def test_xw12_with_the_onion_off_the_request_is_the_one_sent_before_the_window_had_rungs():
    import hashlib

    first, second = _requests_onion_off(), _requests_onion_off()
    assert first == second, "control: the request is the same from one run to the next"
    assert "What is my name?" in first and "And where does the garden grow?" in first and "Hello world" in first, (
        "control: both turns were sent, the second with the first's answer")
    assert hashlib.sha256(first.encode("utf-8")).hexdigest() == _BEFORE_RUNGS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
