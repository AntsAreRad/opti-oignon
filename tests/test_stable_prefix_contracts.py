#!/usr/bin/env python3
"""What the context pipeline promises about a byte-stable prompt prefix.

A local model reuses its KV cache only up to the first byte that differs
from the previous request. Today every per-turn block -- ranked memory,
web results, archive snippets, project retrieval -- is appended to the
LEADING system message, so the very first message changes on every turn
and the whole history behind it is re-processed. These contracts pin the
opt-in relocation that fixes the layout without changing a single byte of
what the model is ultimately told.

Relocated, not rewritten. Under the flag, the volatile blocks leave the
leading system message and ride ONE trailing context message placed after
the history and before the current turn: leading system message byte-equal
to the stable head, capability block and history in front, volatile block
behind, current user message last. Every block keeps the exact glue bytes
it carries today, so the concatenation of head and tail reproduces the
historical composed prompt byte for byte.

Identity is the whole envelope. The response and semantic caches key on
the fully assembled context. The relocation must not move that key: the
fingerprint the cache seam receives is computed over head plus tail, so a
flag flip can never alias two different contexts -- or split one.

Stability is the point. Across two turns of one conversation the leading
messages -- stable head, capability block, the shared history -- are byte
identical; only the trailing context and the user turn may differ.

Off means off. With the flag absent or false, every path is byte-identical
to the historical behaviour: volatile blocks stay in the leading system
message, the pipeline signature is unchanged, and the trailing context
message never appears.

Loaded through the shared isolation window; a scripted inference client
records every message; no model, no database, no network is ever reached.
"""

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_OPTIMIZER = "opti_oignon.context_optimizer"
_EXECUTOR = "opti_oignon.executor"
_WRAPPER = "opti_oignon.agent.untrusted_context"

_STABLE_HEAD = "You are the assistant. Stable head." + " pad" * 20


# ---------------------------------------------------------------------------
# Optimizer-level window
# ---------------------------------------------------------------------------

class _ProjectBuilder:
    """Deterministic stand-in for the project retrieval collaborator."""

    def __init__(self, text="PROJECT CONTEXT: onions prefer loam"):
        self.text = text
        self.available = True

    def build_context(self, project_id, query, budget_tokens=None, **kwargs):
        return SimpleNamespace(
            context_text=self.text,
            chunks_used=1,
            total_tokens_estimate=8,
        )

    def build_system_instructions_only(self, project_id):
        return SimpleNamespace(
            context_text=self.text,
            chunks_used=0,
            total_tokens_estimate=8,
        )


def _load_optimizer(project_text=None):
    loaded, restore = isolate(
        targets={_OPTIMIZER: source("context_optimizer.py")},
        seeded={},
        packages=("opti_oignon",),
    )
    mod = loaded[_OPTIMIZER]
    opt = mod.ContextOptimizer()
    if project_text is not None:
        opt._project_builder = _ProjectBuilder(project_text)
    return mod, opt, restore


def _history(n=2):
    out = []
    for i in range(n // 2):
        out.append({"role": "user", "content": f"question {i}"})
        out.append({"role": "assistant", "content": f"answer {i}"})
    return out


# ---------------------------------------------------------------------------
# sp1 -- control: the historical call shape is untouched
# ---------------------------------------------------------------------------

def test_sp1_legacy_call_keeps_volatile_inside_the_leading_system_message():
    mod, opt, restore = _load_optimizer(project_text="PROJECT CONTEXT: loam")
    try:
        composed = _STABLE_HEAD + "\n\nVOLATILE: ranked memory for turn one"
        result = opt.optimize(
            model="test-model:1b",
            system_prompt=composed,
            user_message="What is a monoid?",
            conversation_history=_history(),
            project_id="p-1",
        )
        roles = [m["role"] for m in result.messages]
        assert roles[0] == "system" and roles[-1] == "user"
        assert "VOLATILE: ranked memory" in result.messages[0]["content"]
        assert "PROJECT CONTEXT: loam" in result.messages[0]["content"]
        assert result.system_prompt == result.messages[0]["content"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# sp2 -- relocation: head stays bare, one trailing context message appears
# ---------------------------------------------------------------------------

def test_sp2_volatile_block_rides_one_trailing_message_and_head_stays_bare():
    mod, opt, restore = _load_optimizer(project_text="PROJECT CONTEXT: loam")
    try:
        tail = "\n\nVOLATILE: ranked memory for turn one"
        result = opt.optimize(
            model="test-model:1b",
            system_prompt=_STABLE_HEAD,
            user_message="What is a monoid?",
            conversation_history=_history(),
            project_id="p-1",
            manifest_block="CAPABILITIES: tools A, B",
            volatile_block=tail,
        )
        msgs = result.messages
        assert msgs[0]["content"] == _STABLE_HEAD, "head must stay byte-bare"
        assert msgs[1]["content"] == "CAPABILITIES: tools A, B"
        assert msgs[-1] == {"role": "user", "content": "What is a monoid?"}
        trailing = msgs[-2]
        assert trailing["role"] == "system"
        assert "VOLATILE: ranked memory" in trailing["content"]
        assert "PROJECT CONTEXT: loam" in trailing["content"]
        body = [m["content"] for m in msgs[:-2]]
        assert all("VOLATILE" not in c and "PROJECT CONTEXT" not in c for c in body)
    finally:
        restore()


# ---------------------------------------------------------------------------
# sp3 -- identity: head + tail reproduces the historical composed prompt
# ---------------------------------------------------------------------------

def test_sp3_reported_system_prompt_is_head_plus_tail_byte_for_byte():
    mod, opt_a, restore_a = _load_optimizer(project_text="PROJECT CONTEXT: loam")
    try:
        tail = "\n\nVOLATILE: ranked memory for turn one"
        relocated = opt_a.optimize(
            model="test-model:1b",
            system_prompt=_STABLE_HEAD,
            user_message="q",
            conversation_history=_history(),
            project_id="p-1",
            volatile_block=tail,
        )
    finally:
        restore_a()
    mod, opt_b, restore_b = _load_optimizer(project_text="PROJECT CONTEXT: loam")
    try:
        legacy = opt_b.optimize(
            model="test-model:1b",
            system_prompt=_STABLE_HEAD + tail,
            user_message="q",
            conversation_history=_history(),
            project_id="p-1",
        )
        assert relocated.system_prompt == legacy.system_prompt, (
            "the identity view must be byte-equal across modes"
        )
    finally:
        restore_b()


# ---------------------------------------------------------------------------
# sp4 -- stability: two turns share every leading message byte for byte
# ---------------------------------------------------------------------------

def test_sp4_two_turns_keep_head_capability_and_shared_history_identical():
    mod, opt, restore = _load_optimizer()
    try:
        hist1 = _history(2)
        turn1 = opt.optimize(
            model="test-model:1b",
            system_prompt=_STABLE_HEAD,
            user_message="What is a monoid?",
            conversation_history=hist1,
            manifest_block="CAPABILITIES: tools A, B",
            volatile_block="\n\nVOLATILE: facts ranked for monoids",
        )
        hist2 = hist1 + [
            {"role": "user", "content": "What is a monoid?"},
            {"role": "assistant", "content": "A monoid is a set with..."},
        ]
        turn2 = opt.optimize(
            model="test-model:1b",
            system_prompt=_STABLE_HEAD,
            user_message="And a functor?",
            conversation_history=hist2,
            manifest_block="CAPABILITIES: tools A, B",
            volatile_block="\n\nVOLATILE: facts ranked for functors",
        )
        shared = 2 + len(hist1)  # head, capability block, shared history
        assert turn2.messages[:shared] == turn1.messages[:shared], (
            "every leading message must be byte-identical across turns"
        )
        s1 = json.dumps(turn1.messages[:shared], ensure_ascii=False)
        s2 = json.dumps(turn2.messages, ensure_ascii=False)
        assert s2.startswith(s1[:-1]), "the serialized prefix must be shared"
    finally:
        restore()


# ---------------------------------------------------------------------------
# sp5 -- empty tail still relocates project retrieval out of the head
# ---------------------------------------------------------------------------

def test_sp5_empty_tail_still_moves_project_retrieval_behind_the_history():
    mod, opt, restore = _load_optimizer(project_text="PROJECT CONTEXT: loam")
    try:
        result = opt.optimize(
            model="test-model:1b",
            system_prompt=_STABLE_HEAD,
            user_message="q",
            conversation_history=_history(),
            project_id="p-1",
            volatile_block="",
        )
        msgs = result.messages
        assert msgs[0]["content"] == _STABLE_HEAD
        assert msgs[-2]["role"] == "system"
        assert "PROJECT CONTEXT: loam" in msgs[-2]["content"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# sp12 -- degraded collaborators keep the relocated order sane
# ---------------------------------------------------------------------------

def test_sp12_degraded_collaborators_keep_the_relocated_order():
    mod, opt, restore = _load_optimizer()
    try:
        opt._project_builder = None
        opt._budget_manager = None
        result = opt.optimize(
            model="test-model:1b",
            system_prompt=_STABLE_HEAD,
            user_message="q",
            conversation_history=_history(),
            volatile_block="\n\nVOLATILE: tail",
        )
        roles = [m["role"] for m in result.messages]
        assert roles[0] == "system" and roles[-1] == "user"
        assert result.messages[0]["content"] == _STABLE_HEAD
        assert result.messages[-2]["content"] == "\n\nVOLATILE: tail"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Executor-level window (the real hub, scripted world)
# ---------------------------------------------------------------------------

class _ScriptedOllama:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return iter([{"message": {"content": "ok"}}])


class _Composer:
    def __init__(self):
        self.next_block = "remembered: onions at dawn"

    def build(self, question, **kwargs):
        return self.next_block


class _ConvStub:
    def __init__(self):
        self.history = []

    def get_context_messages(self, cid):
        return list(self.history)

    def get_conversation(self, cid):
        return SimpleNamespace(metadata={})

    def add_message(self, *a, **k):
        pass

    def update_conversation_metadata(self, *a, **k):
        pass


class _CacheRecorder:
    enabled = True

    def __init__(self):
        self.fingerprints = []

    def get(self, *a, **k):
        self.fingerprints.append(k.get("context_fingerprint"))
        return None

    def put(self, *a, **k):
        return None


def _load_executor(*, flag_on, with_web=False, with_cache=False):
    scripted = _ScriptedOllama()
    composer = _Composer()
    conv = _ConvStub()
    cache = _CacheRecorder()

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = scripted.chat

    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: "test-model:1b",
        get_temperature=lambda *a, **k: 0.2,
    )
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    retrieval = types.ModuleType("opti_oignon.memory.retrieval")
    retrieval.build_memory_block = composer.build
    retrieval.working_memory_block = composer.build
    convmod = types.ModuleType("opti_oignon.conversation")
    convmod.conversation_manager = conv

    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.memory.retrieval": retrieval,
        "opti_oignon.conversation": convmod,
    }
    if with_web:
        web = types.ModuleType("opti_oignon.web_search")
        web.web_search_engine = SimpleNamespace(
            search=lambda q, max_results=5: [
                {"title": "T1", "snippet": "S1", "url": "http://local/1"}
            ]
        )
        seeded["opti_oignon.web_search"] = web
    if with_cache:
        sem = types.ModuleType("opti_oignon.semantic_cache")
        sem.semantic_cache = cache
        seeded["opti_oignon.semantic_cache"] = sem

    targets = {
        _WRAPPER: source("agent", "untrusted_context.py"),
        _OPTIMIZER: source("context_optimizer.py"),
        # The deduplicator rides along as a real target: the hub imports
        # it plainly, and it is pure and standard-library only.
        "opti_oignon.context_dedup": source("context_dedup.py"),
        _EXECUTOR: source("executor.py"),
    }
    had = "ollama" in sys.modules
    prev = sys.modules.get("ollama")
    sys.modules["ollama"] = ollama_stub
    loaded, win_restore = isolate(
        targets=targets,
        seeded=seeded,
        packages=("opti_oignon.agent", "opti_oignon.memory"),
    )
    opt_mod = loaded[_OPTIMIZER]
    opt_mod.init_optimizer(
        config={
            "enabled": False,
            "stable_prefix": {"enabled": bool(flag_on)},
        }
    )

    def restore():
        win_restore()
        if had:
            sys.modules["ollama"] = prev
        else:
            sys.modules.pop("ollama", None)

    world = SimpleNamespace(
        mod=loaded[_EXECUTOR],
        wrapper=loaded[_WRAPPER],
        scripted=scripted,
        composer=composer,
        conv=conv,
        cache=cache,
    )
    return world, restore


def _routing():
    return SimpleNamespace(
        model="test-model:1b",
        task_type="general",
        temperature=0.2,
        prompt_variant="standard",
        timeout=30,
    )


def _drive(world, question, conversation_id=None, web_search=False):
    ex = getattr(world, "_executor", None)
    if ex is None:
        ex = world.mod.Executor()
        world._executor = ex
    gen = ex.execute(
        question,
        _routing(),
        refine=False,
        conversation_id=conversation_id,
        web_search=web_search,
    )
    try:
        while True:
            next(gen)
    except StopIteration:
        pass
    return world.scripted.calls[-1]["messages"]


# ---------------------------------------------------------------------------
# sp6 -- control: flag off, memory stays wrapped inside the leading message
# ---------------------------------------------------------------------------

def test_sp6_flag_off_keeps_wrapped_memory_in_the_leading_system_message():
    world, restore = _load_executor(flag_on=False)
    try:
        msgs = _drive(world, "What is a monoid?")
        expected = world.wrapper.wrap(
            world.composer.next_block, source=world.wrapper.SOURCE_MEMORY
        )
        assert expected in msgs[0]["content"]
        assert [m["role"] for m in msgs] == ["system", "user"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# sp7 -- flag on, single turn: memory leaves the head, envelope intact
# ---------------------------------------------------------------------------

def test_sp7_flag_on_moves_wrapped_memory_to_the_trailing_context_message():
    world, restore = _load_executor(flag_on=True)
    try:
        msgs = _drive(world, "What is a monoid?")
        expected = world.wrapper.wrap(
            world.composer.next_block, source=world.wrapper.SOURCE_MEMORY
        )
        assert [m["role"] for m in msgs] == ["system", "system", "user"]
        assert expected not in msgs[0]["content"]
        assert expected in msgs[1]["content"], "the envelope must survive the move"
        assert world.composer.next_block not in msgs[0]["content"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# sp8 -- flag on, conversation: the head is byte-stable across two turns
# ---------------------------------------------------------------------------

def test_sp8_flag_on_conversation_head_is_byte_stable_across_turns():
    world, restore = _load_executor(flag_on=True)
    try:
        world.composer.next_block = "facts ranked for monoids"
        m1 = _drive(world, "What is a monoid?", conversation_id="c-1")
        world.conv.history += [
            {"role": "user", "content": "What is a monoid?"},
            {"role": "assistant", "content": "ok"},
        ]
        world.composer.next_block = "facts ranked for functors"
        m2 = _drive(world, "And a functor?", conversation_id="c-1")

        assert m1[0] == m2[0], "the leading system message must not move"
        assert "facts ranked" not in m2[0]["content"]
        assert m2[-2]["role"] == "system"
        assert "facts ranked for functors" in m2[-2]["content"]
        assert m2[-1]["role"] == "user"
    finally:
        restore()


# ---------------------------------------------------------------------------
# sp9 -- the cache fingerprint does not move when the flag flips
# ---------------------------------------------------------------------------

def test_sp9_context_fingerprint_is_identical_across_flag_states():
    world_off, restore_off = _load_executor(flag_on=False, with_cache=True)
    try:
        _drive(world_off, "What is a monoid?")
        fp_off = world_off.cache.fingerprints[-1]
        sys_off = world_off.scripted.calls[-1]["messages"][0]["content"]
    finally:
        restore_off()

    world_on, restore_on = _load_executor(flag_on=True, with_cache=True)
    try:
        _drive(world_on, "What is a monoid?")
        fp_on = world_on.cache.fingerprints[-1]
        msgs_on = world_on.scripted.calls[-1]["messages"]
    finally:
        restore_on()

    assert fp_off is not None and fp_on is not None
    assert fp_on == fp_off, "relocation must not move the cache identity"
    joined = msgs_on[0]["content"] + msgs_on[1]["content"]
    assert joined == sys_off, "head plus tail must reproduce the composed prompt"
    assert fp_on == hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# sp10 -- manual trim math counts the relocated tail as fixed cost
# ---------------------------------------------------------------------------

def test_sp10_manual_build_counts_the_tail_in_its_fixed_token_cost():
    world, restore = _load_executor(flag_on=True)
    try:
        ex = world.mod.Executor()
        tail = "\n\nVOLATILE: " + "x" * 400
        msgs, total, stats = ex._build_conversation_messages(
            system_prompt=_STABLE_HEAD,
            conversation_id="c-1",
            current_message="q",
            model="test-model:1b",
            volatile_block=tail,
        )
        assert msgs[-2] == {"role": "system", "content": tail}
        assert msgs[-1] == {"role": "user", "content": "q"}
        tail_tokens = ex._estimate_tokens(tail, "test-model:1b")
        assert total >= tail_tokens, "the tail must be priced into the total"
        assert stats["total_tokens"] == total
    finally:
        restore()


# ---------------------------------------------------------------------------
# sp11 -- flag on, web results ride the trailing message, never the head
# ---------------------------------------------------------------------------

def test_sp11_flag_on_web_results_ride_the_trailing_context_message():
    world, restore = _load_executor(flag_on=True, with_web=True)
    try:
        msgs = _drive(world, "What is a monoid?", web_search=True)
        assert "Web Search Results" not in msgs[0]["content"]
        trailing = msgs[-2]["content"]
        assert "--- Web Search Results ---" in trailing
        assert "http://local/1" in trailing
    finally:
        restore()
