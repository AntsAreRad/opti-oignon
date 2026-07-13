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
    available and yields a plain error marker (zero calls) when it is not;
  * OX6 -- planning degrades to exactly one bounded step (fallback plan,
    empty pipeline, unknown agent normalized) and the step prompt keeps
    only the last two previous outputs, each truncated.

Loads the package modules in isolation under a stand-in package with a
recording client stub; every ``opti_oignon.*`` entry plus the client entry
is snapshotted and evicted first. A meta-path guard refuses any project
submodule that was not seeded, so the load behaves identically whether or
not the project is installed (an editable install resolves submodules by
name and would otherwise bypass the stand-in package). Local-only. Runs
under pytest or the __main__ runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_AGENTS = _REPO / "opti_oignon" / "agents"


class _ClientRecorder:
    """Module-shaped stub for the model client; behavior injectable."""

    def __init__(self):
        self.module = types.ModuleType("ollama")
        self.calls = []
        self.chat_impl = lambda **kw: {"message": {"content": "stub-answer"}}
        self.module.chat = self._chat
        self.module.list = self._list

    def _chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return self.chat_impl(**kwargs)

    def _list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return {"models": [{"name": "m1"}, {"name": "mX"}]}


class _IsolationGuard:
    """Refuse every project submodule the test did not seed.

    A stand-in package whose ``__path__`` is empty isolates the tree only
    while the parent path is the sole way to resolve a submodule. That
    assumption breaks wherever the project is installed in editable mode:
    such an install registers a finder that answers on the module NAME and
    ignores the parent path, so a real submodule resolves behind the test's
    back -- silently importing live code and reopening real databases. This
    guard sits ahead of every finder and refuses the names that were not
    seeded, so a load behaves identically whether the project is installed
    or not.
    """

    _PREFIX = "opti_oignon."

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self._PREFIX):
            raise ModuleNotFoundError(
                f"not seeded in the isolation window: {fullname}",
                name=fullname,
            )
        return None


def _load(module_name):
    """Load one package module under a stand-in package with a client stub."""
    keys = ["ollama"] + [
        k
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    ]
    saved = {k: sys.modules[k] for k in keys if k in sys.modules}
    for k in keys:
        sys.modules.pop(k, None)

    client = _ClientRecorder()
    sys.modules["ollama"] = client.module

    root = types.ModuleType("opti_oignon")
    root.__path__ = []
    pkg = types.ModuleType("opti_oignon.agents")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = root
    sys.modules["opti_oignon.agents"] = pkg
    root.agents = pkg

    guard = _IsolationGuard()
    sys.meta_path.insert(0, guard)

    def restore():
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        for k in list(sys.modules):
            if k == "opti_oignon" or k.startswith("opti_oignon."):
                del sys.modules[k]
        sys.modules.pop("ollama", None)
        for k, v in saved.items():
            sys.modules[k] = v

    full = f"opti_oignon.agents.{module_name}"
    spec = importlib.util.spec_from_file_location(
        full, _AGENTS / f"{module_name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    setattr(pkg, module_name, mod)
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        restore()
        raise

    return mod, client, restore


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
