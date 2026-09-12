#!/usr/bin/env python3
"""Contracts for the second convergence: nine more modules ask the inference
registry instead of the client behind it.

Each seam below used to call the client library directly -- a completion
call, a chat, a stream, a model list, a one-shot client built per route.
Each now resolves a backend through the registry, so admission,
provenance, schema and tool options apply to it, and each degrades by
name when no backend serves the model rather than reaching for a client.
The last contract is the census: the funnel guard's own counter finds no
direct site in any migrated module, and names the one that still owes.

  * CV1 -- self-correction's fact check asks the registry with a single
    user message and the options it always sent; no backend, the
    heuristic result.
  * CV2 -- the dynamic planner asks the registry with its planning model;
    no backend, the fallback plan and no request.
  * CV3 -- an agent executes through the registry, reads the model list
    from the active backend, and absorbs a backend failure as text.
  * CV4 -- the speculative draft asks the registry with the draft model;
    no backend, a refusal by name.
  * CV5 -- legacy fact extraction asks the registry and parses the reply;
    no backend, no facts.
  * CV6 -- the shared one-shot client answers with the registry's text and
    refuses by name without a backend.
  * CV7 -- the shared stream client forwards tools and yields one chunk in
    the loop's shape.
  * CV8 -- the census: zero direct sites in the nine modules, the five
    routes standing on the shared clients, and the one owed module named.

Local-only (the public distribution ships no tests). Every module is
loaded through the shared isolation window over the registry bridge.
"""

import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402
from _registry_bridge import ScriptedBackend, StubRegistry, seed_registry  # noqa: E402

_GUARD = REPO / ".github" / "scripts" / "registry_funnel_guard.py"


class _Scripted:
    def __init__(self, reply="ok", stream=("ok",), models=("m1", "mX"), tool_calls=None):
        self.calls, self.reply, self.stream_parts, self.models, self.tool_calls = [], reply, list(stream), list(models), tool_calls

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([{"message": {"content": p}} for p in self.stream_parts] + [{"message": {"content": ""}, "done": True}])
        return {"message": {"content": self.reply, "tool_calls": list(self.tool_calls or [])}}

    def list(self):
        self.calls.append({"list": True})
        return {"models": [{"name": m} for m in self.models]}


def _window(targets, scripted=None, *, registry=True, seeded=None, blocked=()):
    seeded = dict(seeded or {})
    scripted = scripted or _Scripted()
    if registry:
        seed_registry(seeded, scripted)
    else:
        empty = StubRegistry()
        module = types.ModuleType("opti_oignon.inference_backend")
        module.get_backend_registry = lambda: empty
        seeded["opti_oignon.inference_backend"] = module
    loaded, restore = isolate(targets=targets, seeded=seeded, blocked=blocked, packages=("opti_oignon", "opti_oignon.agents", "opti_oignon.memory", "opti_oignon.api"))
    return loaded, scripted, restore


def _count_sites():
    spec = importlib.util.spec_from_file_location("registry_funnel_guard", str(_GUARD))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.count_sites


# ---------------------------------------------------------------------------
# CV1 -- self-correction
# ---------------------------------------------------------------------------
def test_cv1_the_fact_check_asks_the_registry_and_degrades_to_the_heuristic():
    reply = json.dumps({"flags": [{"claim": "Rome", "concern": "wrong city", "severity": "high"}], "confidence": 0.4})
    loaded, scripted, restore = _window({"opti_oignon.self_correction": source("self_correction.py")}, _Scripted(reply=reply))
    try:
        mod = loaded["opti_oignon.self_correction"]
        engine = mod.SelfCorrectionEngine(config=mod.SelfCorrectionConfig(check_facts=True, correction_model="corr:1b"))
        assert engine.available is True
        result = engine.check_facts("The Eiffel tower is in Rome.")
        assert result.flag_count == 1 and result.flags[0].claim == "Rome"
        call = scripted.calls[0]
        assert call["model"] == "corr:1b"
        assert [m["role"] for m in call["messages"]] == ["user"], "the completion prompt becomes one user message"
        assert "Eiffel" in call["messages"][0]["content"]
        assert call["options"] == {"temperature": 0.1, "num_predict": 1024}
    finally:
        restore()
    loaded, scripted, restore = _window({"opti_oignon.self_correction": source("self_correction.py")}, registry=False)
    try:
        mod = loaded["opti_oignon.self_correction"]
        engine = mod.SelfCorrectionEngine(config=mod.SelfCorrectionConfig(check_facts=True))
        assert engine.available is False
        result = engine.check_facts("The Eiffel tower is in Rome.")
        assert result.confidence == 0.5 and result.flag_count == 0, "no backend: the heuristic result, as before"
        assert scripted.calls == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# CV2 -- the dynamic planner
# ---------------------------------------------------------------------------
def test_cv2_the_planner_asks_the_registry_and_falls_back_without_a_backend():
    plan = json.dumps({"steps": [{"agent_type": "coder", "description": "write it", "model": "mX"}], "complexity": "simple"})
    loaded, scripted, restore = _window({"opti_oignon.agents.dynamic_pipeline": source("agents", "dynamic_pipeline.py")}, _Scripted(reply=plan))
    try:
        mod = loaded["opti_oignon.agents.dynamic_pipeline"]
        planner = mod.DynamicPipelinePlanner(config={"planning_model": "planner:1b"})
        out = planner.plan("build a sieve")
        assert out is not None
        assert scripted.calls, "the planner asked the registry"
        call = scripted.calls[0]
        assert call["model"] == "planner:1b"
        assert [m["role"] for m in call["messages"]] == ["system", "user"]
        assert "build a sieve" in call["messages"][1]["content"]
    finally:
        restore()
    loaded, scripted, restore = _window({"opti_oignon.agents.dynamic_pipeline": source("agents", "dynamic_pipeline.py")}, registry=False)
    try:
        mod = loaded["opti_oignon.agents.dynamic_pipeline"]
        planner = mod.DynamicPipelinePlanner(config={"planning_model": "planner:1b"})
        out = planner.plan("build a sieve")
        assert out is not None and scripted.calls == [], "the fallback plan, and no request went anywhere"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CV3 -- an agent
# ---------------------------------------------------------------------------
def test_cv3_an_agent_executes_through_the_registry_and_absorbs_a_failure_as_text():
    loaded, scripted, restore = _window({"opti_oignon.agents.base": source("agents", "base.py")}, _Scripted(reply="stub-answer"))
    try:
        mod = loaded["opti_oignon.agents.base"]

        class _Agent(mod.BaseAgent):
            def get_system_prompt(self, role, context):
                return "sys"

        agent = _Agent("a", {"models": {"primary": "m1"}, "temperature": 0.3, "timeout": 5})
        assert agent.is_available("m1") is True and agent.is_available("nope") is False
        output = agent.execute(prompt="hello", role=mod.AgentRole.GENERATOR, context={})
        assert output.content == "stub-answer" and output.model_used == "m1"
        chats = [c for c in scripted.calls if "messages" in c]
        assert len(chats) == 1 and chats[0]["model"] == "m1" and chats[0]["options"] == {"temperature": 0.3}
        scripted.reply = None

        def boom(**kwargs):
            raise RuntimeError("client down")
        scripted.chat = boom
        output = agent.execute(prompt="hello", role=mod.AgentRole.GENERATOR, context={})
        assert output.content.startswith("Error:") and "client down" in output.content
    finally:
        restore()


# ---------------------------------------------------------------------------
# CV4 -- the speculative draft
# ---------------------------------------------------------------------------
def test_cv4_the_speculative_draft_asks_the_registry_and_refuses_without_a_backend():
    loaded, scripted, restore = _window({"opti_oignon.speculative": source("speculative.py")}, _Scripted(reply="a draft"))
    try:
        mod = loaded["opti_oignon.speculative"]
        gen = mod.SpeculativeGenerator()
        assert gen._call_draft("q") == "a draft"
        call = scripted.calls[0]
        assert call["model"] == gen._draft_model and call["messages"] == [{"role": "user", "content": "q"}]
        assert set(call["options"]) == {"temperature", "num_predict"}
        assert gen._call_verify("q", "a draft") == "a draft"
        assert scripted.calls[1]["model"] == gen._verify_model and scripted.calls[1]["messages"][0]["role"] == "system"
    finally:
        restore()
    loaded, scripted, restore = _window({"opti_oignon.speculative": source("speculative.py")}, registry=False)
    try:
        gen = loaded["opti_oignon.speculative"].SpeculativeGenerator()
        with pytest.raises(RuntimeError, match="registry"):
            gen._call_draft("q")
    finally:
        restore()


# ---------------------------------------------------------------------------
# CV5 -- legacy extraction
# ---------------------------------------------------------------------------
def test_cv5_legacy_extraction_asks_the_registry_and_extracts_nothing_without_one(tmp_path):
    cfg = types.ModuleType("opti_oignon.config")
    cfg.DATA_DIR = tmp_path
    cfg.config = SimpleNamespace(get=lambda *a, **k: None)
    db = types.ModuleType("opti_oignon.db_utils")
    db.safe_connect = lambda p, **kw: sqlite3.connect(str(p), **kw)
    seeded = {"opti_oignon.config": cfg, "opti_oignon.db_utils": db}
    reply = json.dumps([{"fact": "Alice lives in Berlin", "category": "identity"}])
    loaded, scripted, restore = _window({"opti_oignon.memory.legacy": source("memory", "legacy.py")}, _Scripted(reply=reply), seeded=seeded)
    try:
        mod = loaded["opti_oignon.memory.legacy"]
        manager = mod.MemoryManager(db_path=tmp_path / "memories.db")
        # Two turns: the extractor declines a conversation shorter than that
        # before it asks anyone, and this contract is about the asking.
        turns = [
            {"role": "user", "content": "I am Alice and I live in Berlin."},
            {"role": "assistant", "content": "Noted, Alice in Berlin."},
        ]
        facts = manager.extract_facts_from_messages(turns, model="x:1b")
        assert facts and facts[0]["fact"] == "Alice lives in Berlin"
        call = scripted.calls[0]
        assert call["model"] == "x:1b" and [m["role"] for m in call["messages"]] == ["system", "user"]
    finally:
        restore()
    loaded, scripted, restore = _window({"opti_oignon.memory.legacy": source("memory", "legacy.py")}, registry=False, seeded=seeded)
    try:
        mod = loaded["opti_oignon.memory.legacy"]
        manager = mod.MemoryManager(db_path=tmp_path / "memories2.db")
        turns = [
            {"role": "user", "content": "I am Alice."},
            {"role": "assistant", "content": "Hello Alice."},
        ]
        assert manager.extract_facts_from_messages(turns, model="x:1b") == []
        assert scripted.calls == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# CV6 -- the shared one-shot client
# ---------------------------------------------------------------------------
def test_cv6_the_one_shot_client_answers_with_the_registrys_text_and_refuses_by_name():
    loaded, scripted, restore = _window({"opti_oignon.registry_clients": source("registry_clients.py")}, _Scripted(reply="one shot"))
    try:
        mod = loaded["opti_oignon.registry_clients"]
        client = mod.OneShotChatClient("m", host="http://ignored")
        assert client([{"role": "user", "content": "hi"}]) == "one shot"
        call = scripted.calls[0]
        assert call["model"] == "m" and call["messages"] == [{"role": "user", "content": "hi"}]
        assert "host" not in call, "the per-route host override is retired: the registry's backend is the funnel"
    finally:
        restore()
    loaded, scripted, restore = _window({"opti_oignon.registry_clients": source("registry_clients.py")}, registry=False)
    try:
        client = loaded["opti_oignon.registry_clients"].OneShotChatClient("m")
        with pytest.raises(RuntimeError, match="registry"):
            client([{"role": "user", "content": "hi"}])
    finally:
        restore()


# ---------------------------------------------------------------------------
# CV7 -- the shared stream client
# ---------------------------------------------------------------------------
def test_cv7_the_stream_client_forwards_tools_and_yields_one_loop_shaped_chunk():
    raw = [SimpleNamespace(function=SimpleNamespace(name="create_file", arguments={"path": "a"}))]
    loaded, scripted, restore = _window({"opti_oignon.registry_clients": source("registry_clients.py")}, _Scripted(reply="on it", tool_calls=raw))
    try:
        mod = loaded["opti_oignon.registry_clients"]
        tools = [{"type": "function", "function": {"name": "create_file"}}]
        chunks = list(mod.ModelStreamClient("m").stream([{"role": "user", "content": "go"}], tools))
        assert len(chunks) == 1
        assert chunks[0]["message"]["content"] == "on it"
        assert chunks[0]["message"]["tool_calls"][0].function.name == "create_file"
        assert scripted.calls[0]["tools"] == tools and scripted.calls[0]["model"] == "m"
        chunks = list(mod.ModelStreamClient("m").stream([{"role": "user", "content": "go"}]))
        assert "tools" not in scripted.calls[1], "no tools, no tools kwarg"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CV8 -- the census
# ---------------------------------------------------------------------------
def test_cv8_the_migrated_modules_carry_no_direct_site_and_the_owed_one_is_named():
    count = _count_sites()
    migrated = [
        "self_correction.py", "agents/dynamic_pipeline.py", "agents/base.py", "speculative.py", "memory/legacy.py",
        "registry_clients.py", "api/routes_note_actions.py", "api/routes_claim_verification.py",
        "api/routes_citation_verification.py", "api/routes_answer_verification.py", "api/routes_agent.py",
    ]
    for rel in migrated:
        text = (REPO / "opti_oignon" / rel).read_text(encoding="utf-8")
        assert count(text) == 0, f"{rel} still calls the client directly"
        assert "import ollama" not in text, f"{rel} still names the client"
    for rel in ("api/routes_note_actions.py", "api/routes_claim_verification.py", "api/routes_citation_verification.py",
                "api/routes_answer_verification.py", "api/routes_agent.py"):
        assert "registry_clients" in (REPO / "opti_oignon" / rel).read_text(encoding="utf-8"), f"{rel} stands on the shared client"
    owed = (REPO / "opti_oignon" / "model_warmup.py").read_text(encoding="utf-8")
    assert count(owed) >= 1, "control: the counter still counts, and the one module that owes is the one named"
    assert count("import ollama\n\ndef f():\n    return ollama.chat(model='m', messages=[])\n") == 1, "control: capable"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
