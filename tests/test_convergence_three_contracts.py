#!/usr/bin/env python3
"""Contracts that the last eleven direct callers ask the inference registry.

After two convergence blocks the registry-funnel ledger still owed eleven
modules at twelve sites: the warmup reading the loaded set and pinging
residency, the semantic cache asking for an embedding, two benchmark
transports, the judge, the cascade, the humanizer's rewrite pass, the
reasoning engine, the fine-tune comparison and two model listings. Each now
resolves its backend from the registry at the moment of the call, degrades
by name when none is registered, and carries what it used to get from the
client -- a transport timeout, a reported token count, the loaded set --
as the registry now carries them. The two benchmark transports and the two
suites that pinned direct clients are contracted in their own files; the
rest are here.

  * CX1 -- the warmup reads the loaded set through the registry, reports
    unknown as ``None`` and never as an empty list, warms with one user
    message and a single token, and renews residency with no messages.
  * CX2 -- the governor's S1 read treats an unknown loaded set as
    unreachable and a known one, empty or not, as reachable.
  * CX3 -- the benchmark route lists models from the registry's backends
    and runs a task through the backend with the timeout as an option;
    without a backend the task is recorded as an error naming the registry.
  * CX4 -- the fine-tune comparison builds its inference function over the
    registry, one user message per prompt, and builds none without one.
  * CX5 -- a cascade tier asks the registry with its system and user
    messages, its options and its keep-alive, and refuses by name without
    a backend.
  * CX6 -- the reasoning engine is available exactly when the registry has
    a backend for its default model, sends its per-step timeout as an
    option, and refuses by name without a backend.
  * CX7 -- the routing benchmark runs a task through the backend with its
    timeout as an option and lists models from the registry; without a
    backend the task is an error naming the registry.
  * CX8 -- the semantic cache asks the backend's embedding head and answers
    ``None`` without a backend, without an endpoint, or on failure.
  * CX9 -- none of the eleven names the client any more, and the counter
    that says so still counts a synthetic direct site.

Local-only (the public distribution ships no tests). Every module is loaded
through the shared isolation window, with the registry seeded over a
scripted client or left unreachable.
"""

import sys
import tempfile
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402
from _registry_bridge import StubRegistry, seed_registry  # noqa: E402

_REGISTRY = "opti_oignon.inference_backend"
_GUARD = REPO / ".github" / "scripts" / "registry_funnel_guard.py"

_ELEVEN = [
    "api/routes_benchmark.py", "api/routes_fine_tune.py", "benchmark_judge.py",
    "benchmark_runner.py", "cascading.py", "humanizer.py", "model_warmup.py",
    "pre_cache.py", "reasoning.py", "routing/benchmark.py", "semantic_cache.py",
]


class _Scripted:
    """A scripted client: chat answers, a ps listing, an embedding."""

    def __init__(self, reply="answer", ps=None, embedding=None, fail=False):
        self.calls = []
        self.reply = reply
        self._ps = ps
        self._embedding = embedding
        self._fail = fail

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        if self._fail:
            raise RuntimeError("scripted failure")
        return {"message": {"content": self.reply}}

    def ps(self):
        self.calls.append(("ps",))
        return {"models": self._ps or []}

    def embed(self, **kwargs):
        self.calls.append(("embed", kwargs))
        if self._fail:
            raise RuntimeError("scripted failure")
        return {"embeddings": [self._embedding] if self._embedding is not None else []}


def _empty_registry_module():
    module = types.ModuleType(_REGISTRY)
    module.get_backend_registry = lambda: StubRegistry()
    return module


def _window(rel, name, *, scripted=None, empty=False, seeded=None, packages=("opti_oignon",)):
    """Load ``rel`` as ``name`` with the registry seeded over ``scripted``,
    seeded empty, or left unreachable."""
    seeds = dict(seeded or {})
    if scripted is not None:
        seed_registry(seeds, scripted)
    elif empty:
        seeds[_REGISTRY] = _empty_registry_module()
    loaded, restore = isolate(targets={name: source(*rel.split("/"))}, seeded=seeds, packages=packages)
    return loaded[name], restore


def _chat_calls(scripted):
    return [call[1] for call in scripted.calls if call[0] == "chat"]


# ---------------------------------------------------------------------------
# CX1 -- the warmup through the registry
# ---------------------------------------------------------------------------
def test_cx1_the_warmup_reads_warms_and_pings_through_the_registry_and_says_unknown():
    scripted = _Scripted(ps=[{"model": "big", "size_vram": 2048, "context_length": 4096, "digest": "sha256:b"}])
    mod, restore = _window("model_warmup.py", "opti_oignon.model_warmup", scripted=scripted)
    try:
        warmup = mod.ModelWarmup()
        loaded = warmup.get_loaded_models()
        assert [m.name for m in loaded] == ["big"]
        assert loaded[0].size_vram == 2048 and loaded[0].context_length == 4096 and loaded[0].digest == "sha256:b"
        assert warmup.is_model_loaded("big") is True and warmup.is_model_loaded("small") is False
        assert warmup.get_vram_summary()["known"] is True

        result = warmup.warmup("small")
        assert result.success is True and result.already_loaded is False
        asked = _chat_calls(scripted)[0]
        assert asked["model"] == "small"
        assert asked["messages"] == [{"role": "user", "content": mod.WARMUP_PROMPT}]
        assert asked["options"] == {"num_predict": 1}
        assert asked["keep_alive"] == warmup._keep_alive

        assert warmup.warmup("big").already_loaded is True, "an already loaded model is not asked again"
        assert len(_chat_calls(scripted)) == 1

        assert warmup.send_keepalive("big") is True
        ping = _chat_calls(scripted)[1]
        assert ping["messages"] == [] and ping["options"] == {"num_predict": 0}
        assert ping["keep_alive"] == warmup._keep_alive
    finally:
        restore()

    mod, restore = _window("model_warmup.py", "opti_oignon.model_warmup")
    try:
        warmup = mod.ModelWarmup()
        assert warmup.get_loaded_models() is None, "no backend: unknown, not empty"
        assert warmup.is_model_loaded("big") is False
        summary = warmup.get_vram_summary()
        assert summary["known"] is False and summary["model_count"] == 0
        assert "unknown" in warmup.get_warmup_report()
        result = warmup.warmup("big")
        assert result.success is False and "registry" in result.error
        assert warmup.send_keepalive("big") is False
    finally:
        restore()

    class _NoLoadedSet:
        name = "blind"

        def loaded_models(self):
            return None

    mod, restore = _window("model_warmup.py", "opti_oignon.model_warmup", empty=True)
    try:
        seeded = sys.modules[_REGISTRY]
        stub = StubRegistry()
        stub.register(_NoLoadedSet())
        seeded.get_backend_registry = lambda: stub
        assert mod.ModelWarmup().get_loaded_models() is None, "a backend that does not observe its set is unknown too"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CX2 -- the governor's S1 read
# ---------------------------------------------------------------------------
def test_cx2_the_governor_treats_an_unknown_loaded_set_as_unreachable():
    loaded, restore = isolate(
        targets={"opti_oignon.resource_governor": source("resource_governor.py")},
        packages=("opti_oignon",),
    )
    try:
        rg = loaded["opti_oignon.resource_governor"]

        class _Warmup:
            def __init__(self, answer):
                self.answer = answer

            def get_loaded_models(self):
                return self.answer

        gov = rg.ResourceGovernor.__new__(rg.ResourceGovernor)
        gov._warmup = _Warmup(None)
        assert gov._read_s1() == ([], False), "unknown is unreachable, not an empty loaded set"

        gov._warmup = _Warmup([])
        assert gov._read_s1() == ([], True), "a known empty set is reachable"

        gov._warmup = _Warmup([types.SimpleNamespace(name="big", size_vram=1024, expires_at=None, context_length=None, digest="d")])
        views, reachable = gov._read_s1()
        assert reachable is True and [v.name for v in views] == ["big"] and views[0].size_vram_bytes == 1024
    finally:
        restore()


# ---------------------------------------------------------------------------
# CX3 -- the benchmark route
# ---------------------------------------------------------------------------
class _Listing(_Scripted):
    def list(self):
        return {"models": [{"name": "a"}, {"model": "b"}]}


def test_cx3_the_benchmark_route_lists_and_runs_through_the_registry_with_the_timeout():
    scripted = _Listing(reply="the answer with keyword")
    mod, restore = _window("api/routes_benchmark.py", "opti_oignon.api.routes_benchmark",
                           scripted=scripted, packages=("opti_oignon", "opti_oignon.api"))
    try:
        assert mod._get_installed_models() == ["a", "b"]
        row = mod._execute_single_test(
            "m", "t1", {"prompt": "say keyword", "expected_keywords": ["keyword"], "name": "T1"},
            0.3, 17, 64, {},
        )
        assert row["status"] == "success", row
        asked = _chat_calls(scripted)[0]
        assert asked["model"] == "m"
        assert asked["messages"] == [{"role": "user", "content": "say keyword"}]
        assert asked["options"] == {"temperature": 0.3, "num_predict": 64}
        assert asked["timeout"] == 17
    finally:
        restore()

    mod, restore = _window("api/routes_benchmark.py", "opti_oignon.api.routes_benchmark",
                           packages=("opti_oignon", "opti_oignon.api"))
    try:
        assert mod._get_installed_models() == []
        row = mod._execute_single_test("m", "t1", {"prompt": "p", "name": "T1"}, 0.3, 17, 64, {})
        assert row["status"] == "error" and "registry" in row["error_message"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# CX4 -- the fine-tune comparison
# ---------------------------------------------------------------------------
def test_cx4_the_fine_tune_comparison_infers_through_the_registry_or_builds_nothing():
    scripted = _Scripted(reply="compared")
    mod, restore = _window("api/routes_fine_tune.py", "opti_oignon.api.routes_fine_tune",
                           scripted=scripted, packages=("opti_oignon", "opti_oignon.api"))
    try:
        infer = mod._get_inference_fn()
        assert callable(infer)
        assert infer("base", "the prompt") == "compared"
        asked = _chat_calls(scripted)[0]
        assert asked["model"] == "base" and asked["messages"] == [{"role": "user", "content": "the prompt"}]
    finally:
        restore()

    mod, restore = _window("api/routes_fine_tune.py", "opti_oignon.api.routes_fine_tune",
                           empty=True, packages=("opti_oignon", "opti_oignon.api"))
    try:
        assert mod._get_inference_fn() is None, "an empty registry builds no inference function"
    finally:
        restore()

    mod, restore = _window("api/routes_fine_tune.py", "opti_oignon.api.routes_fine_tune",
                           packages=("opti_oignon", "opti_oignon.api"))
    try:
        assert mod._get_inference_fn() is None, "no registry builds no inference function"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CX5 -- a cascade tier
# ---------------------------------------------------------------------------
def test_cx5_a_cascade_tier_asks_the_registry_with_its_messages_and_refuses_by_name():
    scripted = _Scripted(reply="tier answer")
    mod, restore = _window("cascading.py", "opti_oignon.cascading", scripted=scripted)
    try:
        engine = mod.CascadingInference.__new__(mod.CascadingInference)
        tier = mod.CascadeTierConfig(name="fast", model="small", threshold=0.5, max_tokens=128, temperature=0.2)
        assert engine._call_llm("the query", tier) == "tier answer"
        asked = _chat_calls(scripted)[0]
        assert asked["model"] == "small"
        assert [m["role"] for m in asked["messages"]] == ["system", "user"]
        assert asked["messages"][1]["content"] == "the query"
        assert asked["options"] == {"temperature": 0.2, "num_predict": 128}
        assert asked["keep_alive"] == "30m"
    finally:
        restore()

    mod, restore = _window("cascading.py", "opti_oignon.cascading")
    try:
        engine = mod.CascadingInference.__new__(mod.CascadingInference)
        tier = mod.CascadeTierConfig(name="fast", model="small", threshold=0.5)
        with pytest.raises(RuntimeError, match="registry"):
            engine._call_llm("the query", tier)
    finally:
        restore()


# ---------------------------------------------------------------------------
# CX6 -- the reasoning engine
# ---------------------------------------------------------------------------
def test_cx6_the_reasoning_engine_is_available_with_a_backend_and_sends_its_timeout():
    scripted = _Scripted(reply="  reasoned  ")
    mod, restore = _window("reasoning.py", "opti_oignon.reasoning", scripted=scripted)
    try:
        engine = mod.ReasoningEngine(config=mod.ReasoningConfig(timeout_per_step=23), default_model="think")
        assert engine.available is True
        messages = [{"role": "user", "content": "why"}]
        assert engine._call_llm(messages, temperature=0.4) == "reasoned"
        asked = _chat_calls(scripted)[0]
        assert asked["model"] == "think" and asked["messages"] == messages
        assert asked["options"] == {"temperature": 0.4}
        assert asked["timeout"] == 23
        engine._call_llm(messages, model="other", timeout=5)
        assert _chat_calls(scripted)[1]["model"] == "other" and _chat_calls(scripted)[1]["timeout"] == 5
    finally:
        restore()

    mod, restore = _window("reasoning.py", "opti_oignon.reasoning")
    try:
        engine = mod.ReasoningEngine(config=mod.ReasoningConfig(), default_model="think")
        assert engine.available is False
        with pytest.raises(RuntimeError, match="registry"):
            engine._call_llm([{"role": "user", "content": "why"}])
    finally:
        restore()


# ---------------------------------------------------------------------------
# CX7 -- the routing benchmark
# ---------------------------------------------------------------------------
def test_cx7_the_routing_benchmark_runs_and_lists_through_the_registry():
    scripted = _Listing(reply="an answer naming the keyword")
    mod, restore = _window("routing/benchmark.py", "opti_oignon.routing.benchmark",
                           scripted=scripted, packages=("opti_oignon", "opti_oignon.routing"))
    try:
        bench = mod.ModelBenchmark.__new__(mod.ModelBenchmark)
        bench.temperature = 0.1
        bench.timeout = 31
        result = bench._run_single_test("m", "t1", {"prompt": "p", "name": "T1", "expected_keywords": ["keyword"]})
        assert result.status == "success" and result.keywords_found == ["keyword"]
        asked = _chat_calls(scripted)[0]
        assert asked["model"] == "m" and asked["messages"] == [{"role": "user", "content": "p"}]
        assert asked["options"] == {"temperature": 0.1, "num_predict": 1000}
        assert asked["timeout"] == 31
        assert bench._get_available_models() == ["a", "b"]
    finally:
        restore()

    mod, restore = _window("routing/benchmark.py", "opti_oignon.routing.benchmark",
                           packages=("opti_oignon", "opti_oignon.routing"))
    try:
        bench = mod.ModelBenchmark.__new__(mod.ModelBenchmark)
        bench.temperature = 0.1
        bench.timeout = 31
        result = bench._run_single_test("m", "t1", {"prompt": "p", "name": "T1"})
        assert result.status == "error" and "registry" in result.error_message
        assert bench._get_available_models() == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# CX8 -- the semantic cache's embedding
# ---------------------------------------------------------------------------
def _cache_seeds():
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = Path(tempfile.mkdtemp(prefix="cache_data_"))
    return {"opti_oignon.config": cfg}


def test_cx8_the_semantic_cache_embeds_through_the_registry_or_answers_none():
    scripted = _Scripted(embedding=[0.25, 0.5])
    mod, restore = _window("semantic_cache.py", "opti_oignon.semantic_cache", scripted=scripted, seeded=_cache_seeds())
    try:
        assert mod._get_embedding("hello", model="emb") == [0.25, 0.5]
        assert scripted.calls == [("embed", {"model": "emb", "input": "hello"})]
    finally:
        restore()

    mod, restore = _window("semantic_cache.py", "opti_oignon.semantic_cache",
                           scripted=_Scripted(embedding=None), seeded=_cache_seeds())
    try:
        assert mod._get_embedding("hello", model="emb") is None, "a backend with no vector answers None"
    finally:
        restore()

    mod, restore = _window("semantic_cache.py", "opti_oignon.semantic_cache",
                           scripted=_Scripted(fail=True), seeded=_cache_seeds())
    try:
        assert mod._get_embedding("hello", model="emb") is None, "a failing backend answers None, never raises"
    finally:
        restore()

    mod, restore = _window("semantic_cache.py", "opti_oignon.semantic_cache", seeded=_cache_seeds())
    try:
        assert mod._get_embedding("hello", model="emb") is None, "no registry answers None"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CX9 -- the eleven carry no direct site, and the counter still counts
# ---------------------------------------------------------------------------
def test_cx9_the_eleven_carry_no_direct_site_and_the_counter_still_counts():
    loaded, restore = isolate(targets={"registry_funnel_guard": _GUARD})
    try:
        count = loaded["registry_funnel_guard"].count_sites
    finally:
        restore()
    for rel in _ELEVEN:
        text = (REPO / "opti_oignon" / rel).read_text(encoding="utf-8")
        assert count(text) == 0, f"{rel} still calls the client directly"
        assert "import ollama" not in text, f"{rel} still names the client"
    assert count("import ollama\n\ndef f():\n    return ollama.chat(model='m', messages=[])\n") == 1, "control: capable"
    assert count("import ollama as _o\n\ndef f():\n    return _o.ps()\n") == 1, "control: a ps() read is a site too"
