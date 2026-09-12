#!/usr/bin/env python3
"""Multi-agent exec-surface contracts: text generation only, bounded, off by default.

The multi-agent package orchestrates several models over the local model
client and nothing else: no filesystem step, no shell step, no network
beyond that client. Its execution must stay contained (a client failure
becomes text, never an exception), bounded (per-agent timeout, capped
inter-step context, single-step fallbacks), and opt-in (the enable switch
reads as off when unset). This suite pins that surface:

  * OX1 -- an agent run goes through the model client's chat call and
    returns its content, with no other execution surface touched;
  * OX2 -- a client failure is absorbed into an error text, never raised;
  * OX3 -- the streaming path stops at the per-agent timeout bound instead
    of consuming an unbounded stream;
  * OX4 -- the enable switch is off when the configuration is empty or the
    flag is missing, and on only when explicitly set;
  * OX5 -- the dynamic step executor calls only the client when it is
    available and yields a plain error marker (zero calls) when it is not.
    Superseded: it read a client flag the module no longer holds;
  * OX7 -- the same property over the registry: one request on the step
    model through the registry's backend, and a plain error marker with
    zero requests when no backend serves the step model;
  * OX6 -- planning degrades to exactly one bounded step (fallback plan,
    empty pipeline, unknown agent normalized) and the step prompt keeps
    only the last two previous outputs, each truncated.

Loads the package modules through the shared isolation window over the
registry bridge: the inference registry holds one backend whose client is a
recording stub, so every request the agents send is seen, and a window
without a backend proves the degraded path. Local-only. Runs under pytest
or the __main__ runner.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import StubRegistry, seed_registry  # noqa: E402


class _ClientRecorder:
    """The client behind the registry's backend; behavior injectable."""

    def __init__(self):
        self.calls = []
        self.chat_impl = lambda **kw: {"message": {"content": "stub-answer"}}

    def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return self.chat_impl(**kwargs)

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return {"models": [{"name": "m1"}, {"name": "mX"}]}


def _load(module_name, *, registry=True):
    """Load one package module over the registry bridge.

    With ``registry`` the registry holds one backend over the recording
    client; without it the registry is empty, so no backend serves any
    model. Returns ``(module, client, restore)``.
    """
    client = _ClientRecorder()
    seeded = {}
    if registry:
        seed_registry(seeded, client)
    else:
        empty = StubRegistry()
        module = types.ModuleType("opti_oignon.inference_backend")
        module.get_backend_registry = lambda: empty
        seeded["opti_oignon.inference_backend"] = module
    full = f"opti_oignon.agents.{module_name}"
    loaded, restore = isolate(
        targets={full: source("agents", f"{module_name}.py")},
        seeded=seeded,
        packages=("opti_oignon", "opti_oignon.agents"),
    )
    return loaded[full], client, restore


def _concrete_agent(mod, timeout=120):
    """A minimal concrete agent over the abstract base."""

    class _Probe(mod.BaseAgent):
        def get_system_prompt(self, role, context):
            return "probe system prompt"

    return _Probe(
        name="probe",
        config={"models": {"primary": "m1"}, "timeout": timeout},
    )


# ---------------------------------------------------------------------------
# OX1 -- an agent run is a client chat call and nothing else
# ---------------------------------------------------------------------------
def test_ox1_agent_execution_goes_through_the_client_only():
    mod, client, restore = _load("base")
    try:
        agent = _concrete_agent(mod)
        output = agent.execute(
            prompt="hello", role=mod.AgentRole.GENERATOR, context={},
        )
        assert output.content == "stub-answer", (
            f"the client answer must be returned verbatim, got {output.content!r}"
        )
        assert output.model_used == "m1"
        kinds = {kind for kind, _ in client.calls}
        assert kinds <= {"chat", "list"}, (
            f"only the client surface may be touched, saw {kinds}"
        )
        chats = [kw for kind, kw in client.calls if kind == "chat"]
        assert len(chats) == 1 and chats[0].get("model") == "m1", (
            f"exactly one chat call on the selected model, got {chats}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# OX2 -- a client failure becomes error text, never an exception
# ---------------------------------------------------------------------------
def test_ox2_client_failure_is_absorbed_as_text():
    mod, client, restore = _load("base")
    try:
        def _boom(**kwargs):
            raise RuntimeError("client down")

        client.chat_impl = _boom
        agent = _concrete_agent(mod)
        output = agent.execute(
            prompt="hello", role=mod.AgentRole.GENERATOR, context={},
        )
        assert output.content.startswith("Error:"), (
            f"a client failure must surface as error text, got {output.content!r}"
        )
        assert "client down" in output.content
    finally:
        restore()


# ---------------------------------------------------------------------------
# OX3 -- streaming stops at the per-agent timeout bound
# ---------------------------------------------------------------------------
def test_ox3_streaming_is_bounded_by_the_agent_timeout():
    mod, client, restore = _load("base")
    try:
        consumed = {"n": 0}

        def _endless(**kwargs):
            def _gen():
                while True:
                    consumed["n"] += 1
                    yield {"message": {"content": "x"}}
            return _gen()

        client.chat_impl = _endless
        agent = _concrete_agent(mod, timeout=0)
        tokens = []
        output = agent.execute(
            prompt="hello",
            role=mod.AgentRole.GENERATOR,
            context={},
            stream=True,
            on_token=tokens.append,
        )
        assert output.content.endswith("[Timeout]"), (
            f"the run must stop with the timeout marker, got {output.content!r}"
        )
        assert consumed["n"] <= 2, (
            f"an endless stream must not be consumed past the bound, "
            f"got {consumed['n']} chunks"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# OX4 -- the enable switch is off unless explicitly set
# ---------------------------------------------------------------------------
def test_ox4_enable_switch_reads_off_by_default():
    mod, _client, restore = _load("base")
    try:
        original = mod._agent_config
        try:
            mod._agent_config = {}
            assert mod.is_multi_agent_enabled() is False, (
                "an empty configuration must read as disabled"
            )
            mod._agent_config = {"global": {}}
            assert mod.is_multi_agent_enabled() is False, (
                "a missing enable flag must read as disabled"
            )
            mod._agent_config = {"global": {"enabled": True}}
            assert mod.is_multi_agent_enabled() is True, (
                "an explicit enable flag must read as enabled"
            )
        finally:
            mod._agent_config = original
    finally:
        restore()


# ---------------------------------------------------------------------------
# OX5 -- the dynamic step executor: client-only when up, marker when down
# ---------------------------------------------------------------------------
def test_ox5_dynamic_step_calls_only_the_client_or_yields_a_marker():
    mod, client, restore = _load("dynamic_pipeline")
    try:
        assert mod.OLLAMA_AVAILABLE is True, (
            "the client stub must be importable in this load"
        )
        executor = mod.DynamicPipelineExecutor()
        step = mod.PipelineStep(
            step_number=1,
            agent_type="coder",
            model="mX",
            task_description="do it",
            expected_output="result",
        )
        out = "".join(executor._execute_step(step, "prompt", stream=False))
        assert out == "stub-answer"
        chats = [kw for kind, kw in client.calls if kind == "chat"]
        assert len(chats) == 1 and chats[0].get("model") == "mX", (
            f"exactly one chat call on the step model, got {chats}"
        )

        client.calls.clear()
        original_flag = mod.OLLAMA_AVAILABLE
        try:
            mod.OLLAMA_AVAILABLE = False
            down = list(executor._execute_step(step, "prompt", stream=False))
        finally:
            mod.OLLAMA_AVAILABLE = original_flag
        assert down and down[0].startswith("[ERROR]"), (
            f"an unavailable client must yield a plain marker, got {down}"
        )
        assert client.calls == [], (
            "an unavailable client must never be called"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# OX7 -- the dynamic step executor over the registry: one request, or a marker
# ---------------------------------------------------------------------------
def test_ox7_dynamic_step_asks_the_registry_once_or_yields_a_marker():
    mod, client, restore = _load("dynamic_pipeline")
    try:
        executor = mod.DynamicPipelineExecutor()
        step = mod.PipelineStep(
            step_number=1,
            agent_type="coder",
            model="mX",
            task_description="do it",
            expected_output="result",
        )
        out = "".join(executor._execute_step(step, "prompt", stream=False))
        assert out == "stub-answer"
        chats = [kw for kind, kw in client.calls if kind == "chat"]
        assert len(chats) == 1 and chats[0].get("model") == "mX", (
            f"exactly one request on the step model through the registry, got {chats}"
        )
    finally:
        restore()

    mod, client, restore = _load("dynamic_pipeline", registry=False)
    try:
        executor = mod.DynamicPipelineExecutor()
        step = mod.PipelineStep(
            step_number=1,
            agent_type="coder",
            model="mX",
            task_description="do it",
            expected_output="result",
        )
        down = list(executor._execute_step(step, "prompt", stream=False))
        assert down and down[0].startswith("[ERROR]"), (
            f"a step model no backend serves must yield a plain marker, got {down}"
        )
        assert client.calls == [], "no backend, no request"
    finally:
        restore()


# ---------------------------------------------------------------------------
# OX6 -- planning degrades to one bounded step; step context is capped
# ---------------------------------------------------------------------------
def test_ox6_plans_degrade_bounded_and_step_context_is_capped():
    mod, _client, restore = _load("dynamic_pipeline")
    try:
        planner = mod.DynamicPipelinePlanner(config={})
        fallback = planner._create_fallback_plan("write code please", "", 0.0)
        assert fallback.step_count == 1, (
            f"the fallback plan must hold exactly one step, got "
            f"{fallback.step_count}"
        )
        assert fallback.single_model_sufficient is True

        normalized = planner._build_plan(
            {
                "recommended_pipeline": [
                    {"agent": "wizard", "model": "auto"},
                ],
                "complexity": "unheard-of",
            },
            "",
            0.0,
        )
        assert normalized.recommended_pipeline[0].agent_type == "coder", (
            "an unknown agent must normalize to the coder"
        )
        assert (
            normalized.recommended_pipeline[0].model
            == planner.agent_models["coder"]
        )
        assert normalized.complexity == mod.PlanComplexity.MEDIUM

        empty = planner._build_plan({"recommended_pipeline": []}, "", 0.0)
        assert empty.step_count == 1, (
            "an empty pipeline must degrade to one default step"
        )

        executor = mod.DynamicPipelineExecutor()
        step = mod.PipelineStep(
            step_number=4,
            agent_type="coder",
            model="mX",
            task_description="finish",
            expected_output="result",
        )
        previous = [
            {"step": 1, "agent": "a", "output": "alpha " * 10},
            {"step": 2, "agent": "b", "output": "beta"},
            {"step": 3, "agent": "c", "output": "z" * 2500},
        ]
        prompt = executor._build_step_prompt(step, "orig", previous, {})
        assert "[Step 1 " not in prompt, (
            "only the last two previous outputs may enter the step prompt"
        )
        assert "[Step 2 " in prompt and "[Step 3 " in prompt
        assert "(truncated)" in prompt, (
            "an over-long previous output must be truncated"
        )
        assert "z" * 2001 not in prompt, (
            "the truncated output must not exceed the cap"
        )
    finally:
        restore()


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {_e!r}")
    print(f"\n{'OK' if _failures == 0 else str(_failures) + ' FAILED'}")
    sys.exit(1 if _failures else 0)
