#!/usr/bin/env python3
"""What a chat turn owns: its stop, its callbacks and its results.

A chat turn used to share all three with every other turn in the process.
One stop flag lived on the executor singleton, set by any conversation's
Stop and cleared by the next call of any conversation. The agentic
executor kept a turn's four callbacks on itself, so a tool call made for
one conversation reached whichever conversation had started last. The
route built a turn's ``done`` payload from the singletons' last results,
which were another turn's whenever two turns overlapped. These contracts
pin the other side of each of those facts:

* FC1 -- one turn's stop never reaches another: the Stop route, a later
  turn and a closing socket. FC13 is its second half, split for time:
  calls that belong to no turn, anonymous turns and two turns of one
  conversation.
* FC2 -- callbacks belong to a turn: tool calls, reasoning steps, consensus
  responses and correction steps reach only their own turn, on the socket
  too, and the agentic executor's default hooks stay defaults.
* FC3 -- results belong to a turn: the ``done`` metadata, the vision and
  verification events, the tool loop's direct answer and native response,
  and the agentic executor's decision to verify code.
* FC6 -- a census of the code: no process-level slot carries a turn, no
  stream function blocks the event loop, and the emergency stop reaches
  every live chat turn.
* FC8 -- a closed socket stops its turn at once and never holds the loop.

Every window comes from the shared isolation module and loads the real
modules under contract; the model client is a scripted stand-in behind the
registry bridge, gated by events. Interleavings are ordered by
``threading.Event`` gates and asyncio events; a wait on a gate times out
after three seconds and fails the clause by name. Every thread a clause
starts, or makes the code start, is joined before the window closes, and
the clause asserts that none is still alive. Nothing real is reached: no
model, no socket, no database.
"""

import ast
import asyncio
import inspect
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

BUDGET_S = {
    "test_fc1_one_turns_stop_never_reaches_another_turn": 2.0,
    "test_fc13_calls_of_no_turn_anonymous_turns_and_shared_turns_are_kept_apart": 2.0,
    "test_fc2_callbacks_belong_to_a_turn": 2.0,
    "test_fc3_results_belong_to_a_turn": 2.0,
    "test_fc6_no_process_slot_carries_a_turn_and_the_emergency_stop_reaches_every_turn": 2.0,
    "test_fc8_a_closed_socket_stops_its_turn_at_once_and_never_holds_the_loop": 2.0,
}

_WAIT = 3.0
_N = 6
_CANCEL = "[Generation cancelled]"


# ---------------------------------------------------------------------------
# Shared material
# ---------------------------------------------------------------------------

def _routing():
    return SimpleNamespace(
        model="m", task_type="general", temperature=0.2,
        prompt_variant="standard", timeout=30,
        routing_reason="contract", images=None,
    )


def _deps_stub():
    deps = types.ModuleType("opti_oignon.api.deps")
    for name in ("ANALYZER_AVAILABLE", "CONVERSATION_AVAILABLE",
                 "EXECUTOR_AVAILABLE", "PRESET_AVAILABLE", "ROUTER_AVAILABLE"):
        setattr(deps, name, False)
    for name in ("analyzer", "conversation_manager", "executor",
                 "preset_manager", "router"):
        setattr(deps, name, None)
    return deps


def _tokens(ws):
    return "".join(d.get("content", "") for d in ws.sent if d.get("type") == "token")


def _done(ws):
    frames = [d for d in ws.sent if d.get("type") == "done"]
    return frames[-1]["metadata"] if frames else None


def _frames(ws, kind):
    return [d for d in ws.sent if d.get("type") == kind]


class _Socket:
    """A chat socket stand-in.

    ``fail_on`` -- a token frame containing this text raises, and ``failed``
    is set when it does. ``hold`` -- token frames wait on this asyncio event,
    and ``holding`` is set when the first one starts waiting.
    """

    def __init__(self, fail_on=None, hold=None, holding=None):
        self.sent = []
        self.fail_on = fail_on
        self.failed = None
        self.hold = hold
        self.holding = holding

    async def send_json(self, data):
        if data.get("type") == "token":
            if self.fail_on is not None and self.fail_on in data.get("content", ""):
                if self.failed is not None:
                    self.failed.set()
                raise ConnectionError("socket closed by the client")
            if self.hold is not None and not self.hold.is_set():
                if self.holding is not None:
                    self.holding.set()
                await self.hold.wait()
        self.sent.append(data)


async def _until(predicate, timeout=_WAIT):
    """Wait on a condition set from another thread; False at the deadline."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.005)
    return True


def _settled(before, timeout=_WAIT):
    """Join every thread started since ``before``; name the ones still alive."""
    deadline = time.monotonic() + timeout
    for thread in [t for t in threading.enumerate() if t not in before]:
        thread.join(max(0.0, deadline - time.monotonic()))
    return [t.name for t in threading.enumerate() if t not in before and t.is_alive()]


class _Dispatch:
    """The client behind the registry: forwards to the clause's client."""

    def __init__(self):
        self.current = None

    def chat(self, **kwargs):
        return self.current.chat(**kwargs)


class _GatedClient:
    """One scripted stream per key: a first chunk, a gate, five more.

    It counts the chunks it produced, records in its ``finally`` whether it
    was closed before the sixth, and sets that key's end event.
    """

    def __init__(self, *keys):
        self.keys = keys
        self.gates = {k: threading.Event() for k in keys}
        self.ended = {k: threading.Event() for k in keys}
        self.produced = {k: 0 for k in keys}
        self.closed_early = {k: None for k in keys}

    def chat(self, **kwargs):
        text = kwargs["messages"][-1]["content"]
        key = next(k for k in self.keys if k in text)
        return self._stream(key)

    def _stream(self, key):
        try:
            self.produced[key] += 1
            yield {"message": {"content": f"{key}0 "}}
            self.gates[key].wait(_WAIT)
            for i in range(1, _N):
                self.produced[key] += 1
                yield {"message": {"content": f"{key}{i} "}}
        finally:
            self.closed_early[key] = self.produced[key] < _N
            self.ended[key].set()

    def release(self, *keys):
        for key in keys or self.keys:
            self.gates[key].set()

    def all_ended(self, *keys):
        return all(self.ended[k].wait(_WAIT) for k in (keys or self.keys))


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _load_route_exec():
    """The chat routes over the real executor, a scripted client behind them."""
    dispatch = _Dispatch()
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(
        get_model=lambda *a, **k: "m", get_temperature=lambda *a, **k: 0.2,
    )
    router = types.ModuleType("opti_oignon.router")
    router.RoutingResult = type("RoutingResult", (), {})
    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.api.deps": _deps_stub(),
    }
    seed_registry(seeded, dispatch)
    loaded, close = isolate(
        targets={
            "opti_oignon.context_dedup": source("context_dedup.py"),
            "opti_oignon.executor": source("executor.py"),
            "opti_oignon.api.schemas": source("api", "schemas.py"),
            "opti_oignon.api.routes_chat": source("api", "routes_chat.py"),
        },
        seeded=seeded,
        packages=("opti_oignon.api",),
    )
    rc = loaded["opti_oignon.api.routes_chat"]
    rc.EXECUTOR_AVAILABLE = True
    rc._resolve_model_and_route = lambda message, request: (_routing(), None)
    world = SimpleNamespace(
        rc=rc,
        schemas=loaded["opti_oignon.api.schemas"],
        exmod=loaded["opti_oignon.executor"],
        dispatch=dispatch,
    )
    return world, close


def _load_routes_only(extra_targets=None, seeded=None):
    """The chat routes alone, their executor to be set by the clause."""
    targets = dict(extra_targets or {})
    targets["opti_oignon.api.schemas"] = source("api", "schemas.py")
    targets["opti_oignon.api.routes_chat"] = source("api", "routes_chat.py")
    seeds = {"opti_oignon.api.deps": _deps_stub()}
    seeds.update(seeded or {})
    loaded, close = isolate(targets=targets, seeded=seeds, packages=("opti_oignon.api",))
    rc = loaded["opti_oignon.api.routes_chat"]
    rc.EXECUTOR_AVAILABLE = True
    rc._resolve_model_and_route = lambda message, request: (_routing(), None)
    return loaded, rc, loaded["opti_oignon.api.schemas"], close


def _load_agentic():
    loaded, close = isolate(
        targets={"opti_oignon.agentic_executor": source("agentic_executor.py")},
        seeded={},
    )
    return loaded["opti_oignon.agentic_executor"], close


def _load_route_agentic():
    """The chat routes over the real agentic executor (stages are stand-ins)."""
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = None
    loaded, rc, schemas, close = _load_routes_only(
        extra_targets={"opti_oignon.agentic_executor": source("agentic_executor.py")},
        seeded={"opti_oignon.conversation": conv},
    )
    ae = loaded["opti_oignon.agentic_executor"]
    rc.AGENTIC_EXECUTOR_AVAILABLE = True
    rc.executor = SimpleNamespace(
        reset=lambda: None, cancel=lambda: None,
        last_vision_meta={}, last_verification_results=[],
        execute=lambda **kw: iter(()),
    )
    return SimpleNamespace(rc=rc, schemas=schemas, ae=ae), close


def _agent(ae, **engines):
    agent = ae.AgenticExecutor(
        executor=engines.pop("executor", SimpleNamespace()),
        default_model="m", **engines,
    )
    agent._classify_message = lambda *a, **k: {"needs_tools": False}
    return agent


async def _open(world, ws, conv, message, **fields):
    request = world.schemas.ChatRequest(
        conversation_id=conv or None, message=message, **fields,
    )
    return asyncio.ensure_future(
        world.rc._stream_response(ws, conv, message, request)
    )


async def _stop(world, conv):
    return await world.rc.cancel_generation(
        world.schemas.ChatCancelRequest(conversation_id=conv)
    )


def _status(answer):
    return getattr(answer, "status_code", 200)


# ---------------------------------------------------------------------------
# FC1 -- one turn's stop never reaches another turn
# ---------------------------------------------------------------------------

def _fc1_fresh(world, *keys):
    client = _GatedClient(*keys)
    world.dispatch.current = client
    world.rc.executor = world.exmod.Executor()
    return client


def _fc1_c1(world):
    """The Stop route stops its own conversation and no other."""
    client = _fc1_fresh(world, "alpha", "beta")
    before = set(threading.enumerate())
    wa, wb = _Socket(), _Socket()

    async def scenario():
        ta = await _open(world, wa, "conv-a", "alpha question")
        assert await _until(lambda: "alpha0" in _tokens(wa)), "A never streamed"
        tb = await _open(world, wb, "conv-b", "beta question")
        assert await _until(lambda: "beta0" in _tokens(wb)), "B never streamed"
        await _stop(world, "conv-a")
        client.release()
        await asyncio.wait_for(asyncio.gather(ta, tb), _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        client.release()
    assert client.all_ended(), "a scripted stream never ended"
    assert _settled(before) == [], "a thread outlived the clause"
    assert _tokens(wa).endswith(_CANCEL), f"A was not stopped: {_tokens(wa)!r}"
    assert _done(wa)["cancelled"] is True
    assert client.produced["alpha"] < _N, "A's model kept generating after its stop"
    for i in range(_N):
        assert f"beta{i}" in _tokens(wb), f"B lost beta{i}: {_tokens(wb)!r}"
    assert _CANCEL not in _tokens(wb), "A's stop reached B's socket"
    assert _done(wb)["cancelled"] is False, "B's done says cancelled"
    assert client.produced["beta"] == _N, f"B's model was cut: {client.produced}"


def _fc1_c2(world):
    """Witness: with no stop, both turns complete, twelve chunks in all."""
    client = _fc1_fresh(world, "alpha", "beta")
    before = set(threading.enumerate())
    wa, wb = _Socket(), _Socket()

    async def scenario():
        ta = await _open(world, wa, "conv-a", "alpha question")
        tb = await _open(world, wb, "conv-b", "beta question")
        assert await _until(lambda: "alpha0" in _tokens(wa) and "beta0" in _tokens(wb))
        client.release()
        await asyncio.wait_for(asyncio.gather(ta, tb), _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        client.release()
    assert client.all_ended()
    assert _settled(before) == []
    assert client.produced["alpha"] + client.produced["beta"] == 2 * _N
    assert _done(wa)["cancelled"] is False and _done(wb)["cancelled"] is False


def _fc1_c3(world):
    """A turn that starts later never cancels an earlier turn's stop."""
    client = _fc1_fresh(world, "alpha", "beta")
    before = set(threading.enumerate())
    wa, wb = _Socket(), _Socket()

    async def scenario():
        ta = await _open(world, wa, "conv-a", "alpha question")
        assert await _until(lambda: "alpha0" in _tokens(wa)), "A never streamed"
        await _stop(world, "conv-a")
        tb = await _open(world, wb, "conv-b", "beta question")
        assert await _until(lambda: "beta0" in _tokens(wb)), "B never streamed"
        client.release()
        await asyncio.wait_for(asyncio.gather(ta, tb), _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        client.release()
    assert client.all_ended()
    assert _settled(before) == []
    assert client.produced["alpha"] < _N, (
        f"B's start erased A's stop: A's model produced {client.produced['alpha']} of {_N}"
    )


def _fc1_c4(world):
    """A closing socket stops its own turn and no other."""
    client = _fc1_fresh(world, "alpha", "beta")
    before = set(threading.enumerate())
    wb = _Socket()
    wa = _Socket(fail_on="alpha0")

    async def scenario():
        wa.failed = asyncio.Event()
        ta = await _open(world, wa, "conv-a", "alpha question")
        tb = await _open(world, wb, "conv-b", "beta question")
        await asyncio.wait_for(wa.failed.wait(), _WAIT)
        assert await _until(lambda: "beta0" in _tokens(wb)), "B never streamed"
        client.release("alpha")
        client.release("beta")
        await asyncio.wait_for(asyncio.gather(ta, tb), _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        client.release()
    assert client.all_ended()
    assert _settled(before) == []
    assert client.produced["beta"] == _N, f"A's closed socket cut B: {client.produced}"
    assert _done(wb)["cancelled"] is False
    assert client.produced["alpha"] < _N, "A's closed socket did not stop A"


def _drainer(world, drained):
    def drain(key):
        gen = world.rc.executor.execute(
            question=f"{key} question", routing=_routing(), refine=False,
        )
        drained[key] = "".join(c for c in gen if isinstance(c, str))
    return drain


def _fc1_c5i(world):
    """A call that belongs to no turn is untouched by a turn's Stop."""
    client = _fc1_fresh(world, "alpha", "gamma")
    before = set(threading.enumerate())
    wa = _Socket()
    drained = {}
    drain = _drainer(world, drained)

    async def scenario():
        ta = await _open(world, wa, "conv-a", "alpha question")
        assert await _until(lambda: "alpha0" in _tokens(wa)), "A never streamed"
        tg = threading.Thread(target=drain, args=("gamma",), name="fc1-gamma")
        tg.start()
        assert await _until(lambda: client.produced["gamma"] >= 1), "gamma never streamed"
        await _stop(world, "conv-a")
        client.release()
        await asyncio.wait_for(ta, _WAIT)
        assert await _until(lambda: not tg.is_alive()), "gamma's drain never ended"

    try:
        asyncio.run(scenario())
    finally:
        client.release()
    assert client.all_ended()
    assert _settled(before) == []
    assert client.produced["gamma"] == _N, (
        f"a turn's Stop cut a call that belongs to no turn: {client.produced}"
    )
    assert _CANCEL not in drained["gamma"]


def _fc1_c5ii(world):
    """With no turn live, the executor's cancel() still stops a call of its own."""
    client = _fc1_fresh(world, "sigma")
    before = set(threading.enumerate())
    drained = {}
    ts = threading.Thread(target=_drainer(world, drained), args=("sigma",), name="fc1-sigma")
    ts.start()
    try:
        deadline = time.monotonic() + _WAIT
        while client.produced["sigma"] < 1 and time.monotonic() < deadline:
            client.ended["sigma"].wait(0.005)
        assert client.produced["sigma"] >= 1, "sigma never streamed"
        world.rc.executor.cancel()
        client.release()
        ts.join(_WAIT)
        assert not ts.is_alive(), "sigma's drain never ended"
    finally:
        client.release()
    assert client.all_ended()
    assert _settled(before) == []
    assert client.produced["sigma"] < _N, "the executor's cancel() reached no call"
    assert _CANCEL in drained["sigma"]


def _fc1_c6i(world):
    """An anonymous turn cannot be named by a Stop."""
    client = _fc1_fresh(world, "delta")
    before = set(threading.enumerate())
    wd = _Socket()
    answers = {}

    async def scenario():
        td = await _open(world, wd, "", "delta question")
        assert await _until(lambda: "delta0" in _tokens(wd)), "the anonymous turn never streamed"
        answers["anonymous"] = await _stop(world, "")
        client.release()
        await asyncio.wait_for(td, _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        client.release()
    assert client.all_ended()
    assert _settled(before) == []
    assert _status(answers["anonymous"]) == 404, "a Stop named the anonymous turn"
    assert client.produced["delta"] == _N and _done(wd)["cancelled"] is False


def _fc1_c6ii(world):
    """A Stop of a conversation stops both of its live turns, then finds none."""
    client = _fc1_fresh(world, "omega", "kappa")
    before = set(threading.enumerate())
    wo, wk = _Socket(), _Socket()
    answers = {}

    async def scenario():
        to = await _open(world, wo, "conv-s", "omega question")
        tk = await _open(world, wk, "conv-s", "kappa question")
        assert await _until(lambda: "omega0" in _tokens(wo) and "kappa0" in _tokens(wk))
        answers["shared"] = await _stop(world, "conv-s")
        client.release()
        await asyncio.wait_for(asyncio.gather(to, tk), _WAIT)
        answers["after"] = await _stop(world, "conv-s")

    try:
        asyncio.run(scenario())
    finally:
        client.release()
    assert client.all_ended()
    assert _settled(before) == []
    assert _status(answers["shared"]) == 200
    assert client.produced["omega"] < _N and client.produced["kappa"] < _N, (
        f"a Stop of conv-s left one of its turns running: {client.produced}"
    )
    assert _done(wo)["cancelled"] is True and _done(wk)["cancelled"] is True
    assert _status(answers["after"]) == 404, "a closed turn still answers a Stop"


def test_fc1_one_turns_stop_never_reaches_another_turn():
    world, close = _load_route_exec()
    try:
        for clause in (_fc1_c1, _fc1_c2, _fc1_c3, _fc1_c4):
            clause(world)
    finally:
        close()


def test_fc13_calls_of_no_turn_anonymous_turns_and_shared_turns_are_kept_apart():
    world, close = _load_route_exec()
    try:
        for clause in (_fc1_c5i, _fc1_c5ii, _fc1_c6i, _fc1_c6ii):
            clause(world)
    finally:
        close()


# ---------------------------------------------------------------------------
# FC2 -- callbacks belong to a turn
# ---------------------------------------------------------------------------

class _Stage:
    """A stage stand-in: turn A parks on the gate, turn B runs through."""

    available = True
    enabled = True

    def __init__(self):
        self.gate = threading.Event()
        self.reached = {"a": threading.Event(), "b": threading.Event()}

    def _enter(self, text):
        who = "a" if "alpha" in text else "b"
        self.reached[who].set()
        if who == "a":
            self.gate.wait(_WAIT)
        return who


class _ToolStage(_Stage):
    def __init__(self, calls=None):
        super().__init__()
        self.calls = calls or {"a": 1, "b": 1}

    def should_use_tools(self, *a, **k):
        return True

    def _call(self, who, i):
        return SimpleNamespace(
            tool_name="read_file", arguments={"path": f"{who}-secret.txt"},
            result=f"{who} private {i}", success=True, execution_time=0,
            reasoning="",
        )

    def stream_with_tools(self, **kw):
        who = self._enter(kw["message"])
        made = [self._call(who, i) for i in range(self.calls[who])]
        for call in made:
            kw["on_tool_call"](call)
        yield f"{who} answer"
        return SimpleNamespace(tool_calls=made, response=f"{who} answer", verification_hints=0)


class _ReasoningStage(_Stage):
    def execute_reasoning(self, **kw):
        who = self._enter(kw["question"])
        kw["on_step"](SimpleNamespace(title=f"{who}-step"))
        result = SimpleNamespace(
            strategy=f"strategy-{who}", steps=[], confidence=0.5,
            total_duration_ms=1,
        )
        yield ("reasoning_done", result)
        yield f"{who} answer"


class _ConsensusStage(_Stage):
    def execute_consensus(self, **kw):
        who = self._enter(kw["query"])
        kw["on_model_done"](SimpleNamespace(model=f"{who}-model"))
        yield f"{who} answer"


class _CorrectionStage(_Stage):
    def execute_self_correction(self, **kw):
        who = self._enter(kw["user_message"])
        yield ("correction_step", {"who": who})
        result = SimpleNamespace(
            was_corrected=True, iterations_performed=7 if who == "b" else 3,
            compliance_before=0.1, compliance_after=0.9, quality_before=0.1,
            quality_after=0.9, total_duration_ms=1,
        )
        yield ("correction_done", result)
        yield f"{who} answer"


def _drafting_executor():
    return SimpleNamespace(execute=lambda **kw: iter(["draft"]))


def _interleave(stage, run_a, run_b):
    """A enters and parks; B runs to its end; then A is released."""
    ta = threading.Thread(target=run_a, name="fc-turn-a")
    tb = threading.Thread(target=run_b, name="fc-turn-b")
    ta.start()
    try:
        assert stage.reached["a"].wait(_WAIT), "turn A never reached its stage"
        tb.start()
        tb.join(_WAIT)
        assert not tb.is_alive(), "turn B never ended"
    finally:
        stage.gate.set()
        ta.join(_WAIT)
    assert not ta.is_alive(), "turn A never ended"


def _two_turns(agent, stage, hook, pick, **flags):
    sinks = {"a": [], "b": []}

    def run(who, message):
        kwargs = dict(flags)
        kwargs[hook] = lambda item: sinks[who].append(pick(item))
        for _ in agent.execute(message, _routing(), **kwargs):
            pass

    _interleave(stage, lambda: run("a", "alpha"), lambda: run("b", "beta"))
    return sinks


def _fc2_c1(ae):
    """Tool calls reach only their own turn."""
    stage = _ToolStage()
    ae.TOOL_EXECUTOR_AVAILABLE = True
    ae._select_pipeline = lambda **kw: ae.PIPELINE_TOOLS
    agent = _agent(ae, tool_executor=stage)
    sinks = _two_turns(agent, stage, "on_tool_call", lambda r: r.arguments["path"])
    assert sinks["b"], "turn B's sink is empty: the comparison met nothing"
    assert sinks["a"] == ["a-secret.txt"], f"A's tool call went elsewhere: {sinks}"
    assert sinks["b"] == ["b-secret.txt"], f"B received another turn's call: {sinks}"


def _fc2_c2(ae):
    """Reasoning steps reach only their own turn."""
    stage = _ReasoningStage()
    ae.REASONING_AVAILABLE = True
    ae._select_pipeline = lambda **kw: ae.PIPELINE_REASONING
    agent = _agent(ae, reasoning_engine=stage)
    sinks = _two_turns(agent, stage, "on_reasoning_step", lambda s: s.title)
    assert sinks["b"], "turn B's sink is empty"
    assert sinks == {"a": ["a-step"], "b": ["b-step"]}, sinks


def _fc2_c3(ae):
    """Consensus responses reach only their own turn."""
    stage = _ConsensusStage()
    ae.CONSENSUS_AVAILABLE = True
    agent = _agent(ae, consensus_engine=stage)
    sinks = _two_turns(agent, stage, "on_consensus_model", lambda r: r.model, consensus=True)
    assert sinks["b"], "turn B's sink is empty"
    assert sinks == {"a": ["a-model"], "b": ["b-model"]}, sinks


def _fc2_c4(ae):
    """Correction steps reach only their own turn."""
    stage = _CorrectionStage()
    ae.SELF_CORRECTION_AVAILABLE = True
    agent = _agent(ae, executor=_drafting_executor(), self_correction_engine=stage)
    sinks = _two_turns(agent, stage, "on_correction_step", lambda s: s["who"], self_correct=True)
    assert sinks["b"], "turn B's sink is empty"
    assert sinks == {"a": ["a"], "b": ["b"]}, sinks


def _socket_pair(world, stage, fields_a=None, fields_b=None):
    """A parks in its stage; B runs through and holds its token; A ends; B ends."""
    before = set(threading.enumerate())
    wa = _Socket()

    async def scenario():
        hold, holding = asyncio.Event(), asyncio.Event()
        wb.hold, wb.holding = hold, holding
        ta = await _open(world, wa, "conv-a", "alpha", **(fields_a or {}))
        assert await _until(stage.reached["a"].is_set), "A never reached its stage"
        tb = await _open(world, wb, "conv-b", "beta", **(fields_b or {}))
        await asyncio.wait_for(holding.wait(), _WAIT)
        stage.gate.set()
        await asyncio.wait_for(ta, _WAIT)
        hold.set()
        await asyncio.wait_for(tb, _WAIT)

    wb = _Socket()
    try:
        asyncio.run(scenario())
    finally:
        stage.gate.set()
    assert _settled(before) == [], "a thread outlived the clause"
    return wa, wb


def _tool_pair(world):
    stage = _ToolStage(calls={"a": 3, "b": 1})
    world.ae.TOOL_EXECUTOR_AVAILABLE = True
    world.ae._select_pipeline = lambda **kw: world.ae.PIPELINE_TOOLS
    world.rc._agentic_executor = _agent(world.ae, tool_executor=stage)
    return _socket_pair(world, stage)


def _fc2_c5(world):
    """On the socket: each turn's tool calls, and only its own."""
    wa, wb = _tool_pair(world)
    calls_a, calls_b = _frames(wa, "tool_call"), _frames(wb, "tool_call")
    assert calls_b, "B's socket carried no tool call: the comparison met nothing"
    assert len(calls_a) == 3, f"A's socket carried {len(calls_a)} tool calls, not 3"
    assert len(calls_b) == 1, f"B's socket carried {len(calls_b)} tool calls, not 1"
    leaked = [f for f in calls_b if "a-secret" in str(f["metadata"]["arguments"])]
    assert leaked == [], f"B's socket carried A's arguments: {leaked}"


def _fc2_c6(ae):
    """The default hooks stay defaults; an internal call without a turn uses them."""
    names = ("_on_tool_call", "_on_reasoning_step", "_on_consensus_model", "_on_correction_step")
    stage = _ToolStage()
    stage.gate.set()
    ae.TOOL_EXECUTOR_AVAILABLE = True
    ae._select_pipeline = lambda **kw: ae.PIPELINE_TOOLS
    agent = _agent(ae, tool_executor=stage)
    fresh = {n: getattr(agent, n, "missing") for n in names}
    assert all(v is None for v in fresh.values()), f"a fresh instance's hooks: {fresh}"
    for _ in agent.execute("alpha", _routing(), on_tool_call=lambda r: None,
                           on_reasoning_step=lambda s: None,
                           on_consensus_model=lambda r: None,
                           on_correction_step=lambda s: None):
        pass
    after = {n: getattr(agent, n, "missing") for n in names}
    assert all(v is None for v in after.values()), f"a turn left its hooks on the instance: {after}"
    relayed = []
    agent._on_tool_call = lambda r: relayed.append(r.arguments["path"])
    for _ in agent._execute_tools_pipeline("beta", _routing(), None, None):
        pass
    assert relayed == ["b-secret.txt"], f"a call without a turn lost its default hook: {relayed}"


def test_fc2_callbacks_belong_to_a_turn():
    for clause in (_fc2_c1, _fc2_c2, _fc2_c3, _fc2_c4, _fc2_c6):
        ae, close = _load_agentic()
        try:
            clause(ae)
        finally:
            close()
    world, close = _load_route_agentic()
    try:
        _fc2_c5(world)
    finally:
        close()


# ---------------------------------------------------------------------------
# FC3 -- results belong to a turn
# ---------------------------------------------------------------------------

def _fc3_c1(world):
    """The done metadata of the agentic path is each turn's own."""
    wa, wb = _tool_pair(world)
    da, db = _done(wa), _done(wb)
    assert da["pipeline"] == "tools", da
    assert da["tool_calls_count"] == 3, f"A's done counts {da['tool_calls_count']}"
    assert db["tool_calls_count"] == 1, f"B's done counts {db['tool_calls_count']}"

    ae = world.ae
    reasoning, correction = _ReasoningStage(), _CorrectionStage()
    ae.REASONING_AVAILABLE = True
    ae.SELF_CORRECTION_AVAILABLE = True
    ae._select_pipeline = lambda **kw: ae.PIPELINE_REASONING
    world.rc._agentic_executor = _agent(
        ae, executor=_drafting_executor(),
        reasoning_engine=reasoning, self_correction_engine=correction,
    )
    wa, wb = _socket_pair(world, reasoning, fields_b={"self_correct": True})
    da, db = _done(wa), _done(wb)
    assert da.get("reasoning", {}).get("strategy") == "strategy-a", da
    assert "correction" not in da, f"A's done carries another turn's correction: {da}"
    assert db.get("correction", {}).get("iterations_performed") == 7, db
    assert "reasoning" not in db, f"B's done carries another turn's reasoning: {db}"


class _RecordingExecutor:
    """A plain-path executor stand-in that keeps the real result contract.

    It writes its run's ``results`` when it is given a run, and always its
    own ``last_*`` mirrors, as the real executor does. Turn A writes, then
    parks after its chunk; turn B runs through and writes last.
    """

    def __init__(self):
        self.last_vision_meta = {}
        self.last_verification_results = []
        self.gate = threading.Event()
        self.wrote = {"a": threading.Event(), "b": threading.Event()}

    def reset(self):
        pass

    def cancel(self):
        pass

    def execute(self, **kw):
        who = "a" if "alpha" in kw["question"] else "b"
        vision = {"delegated": True, "vision_model": f"vision-{who}",
                  "description_length": 1, "duration_ms": 1}
        checked = [SimpleNamespace(
            status=f"status-{who}", iterations=1, language=f"lang-{who}",
            errors_encountered=[], fixes_applied=[], execution_output="",
        )]
        run = kw.get("run")
        if run is not None:
            run.results["vision_meta"] = vision
            run.results["verification_results"] = checked
        self.last_vision_meta = vision
        self.last_verification_results = checked
        self.wrote[who].set()
        yield f"{who} answer"
        if who == "a":
            self.gate.wait(_WAIT)


def _fc3_c2(world):
    """Vision and verification of the plain path are each turn's own."""
    rec = _RecordingExecutor()
    world.rc.executor = rec
    before = set(threading.enumerate())
    wa, wb = _Socket(), _Socket()

    async def scenario():
        ta = await _open(world, wa, "conv-a", "alpha question")
        assert await _until(rec.wrote["a"].is_set), "A never wrote"
        tb = await _open(world, wb, "conv-b", "beta question")
        await asyncio.wait_for(tb, _WAIT)
        rec.gate.set()
        await asyncio.wait_for(ta, _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        rec.gate.set()
    assert _settled(before) == []
    da, db = _done(wa), _done(wb)
    assert db["vision_delegation"]["vision_model"] == "vision-b"
    assert da["vision_delegation"]["vision_model"] == "vision-a", (
        f"A's done names another turn's vision model: {da['vision_delegation']}"
    )
    langs_a = [f["metadata"]["language"] for f in _frames(wa, "verification")]
    langs_b = [f["metadata"]["language"] for f in _frames(wb, "verification")]
    assert langs_b == ["lang-b"], langs_b
    assert langs_a == ["lang-a"], f"A's verification events are another turn's: {langs_a}"
    live_a, live_b = (
        [f["metadata"].get("vision_model") for f in _frames(ws, "vision_delegation")
         if f["metadata"].get("status") == "done"]
        for ws in (wa, wb)
    )
    assert live_b == ["vision-b"], live_b
    assert live_a == ["vision-a"], f"A's live vision event names another turn's model: {live_a}"


def _load_tool_executor():
    """The real tool executor chain over a scripted client (sibling shape)."""
    class Scripted:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            who = "A" if "alpha" in str(kwargs["messages"]) else "B"
            return SimpleNamespace(message=SimpleNamespace(
                content=f"Answer meant for conversation {who}.", tool_calls=[],
            ))

    scripted = Scripted()
    sm = types.ModuleType("opti_oignon.security_mode")
    sm.is_bulbe = lambda: False
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(get_user_preference=lambda key, default=None: default)
    cfg.get_model = lambda *a, **k: "scripted-model"
    so = types.ModuleType("opti_oignon.structured_output")
    so.StructuredOutputEngine = object
    so.ToolCallRequest = object
    seeded = {
        "opti_oignon.security_mode": sm,
        "opti_oignon.config": cfg,
        "opti_oignon.structured_output": so,
    }
    seed_registry(seeded, scripted)
    loaded, close = isolate(
        targets={
            "opti_oignon.tool_calling": source("tool_calling.py"),
            "opti_oignon.response_hygiene": source("response_hygiene.py"),
            "opti_oignon.tool_registry": source("tool_registry.py"),
            "opti_oignon.tool_executor": source("tool_executor.py"),
        },
        seeded=seeded,
        packages=("opti_oignon",),
    )
    te = loaded["opti_oignon.tool_executor"]
    te.model_supports_native_tools = lambda model, capability_lookup=None: True
    tr = loaded["opti_oignon.tool_registry"]
    registry = tr.ToolRegistry()
    registry.register(tr.ToolDefinition(
        name="echo", description="Echo the call back. Local test tool.",
        parameters={}, handler=lambda **kwargs: "echoed",
    ))
    executor = te.ToolExecutor(registry=registry)
    return te, executor, scripted, close


def _two_tool_turns(executor, pause_a, resume_a):
    """Run turn A and turn B on the named threads; A pauses where told."""
    replies = {}

    def run(who, message):
        replies[who] = executor.execute_with_tools(message, model="scripted").response

    before = set(threading.enumerate())
    ta = threading.Thread(target=run, args=("A", "alpha question"), name="fc3-A")
    tb = threading.Thread(target=run, args=("B", "beta question"), name="fc3-B")
    ta.start()
    try:
        assert pause_a.wait(_WAIT), "turn A never reached its pause"
        tb.start()
        tb.join(_WAIT)
        assert not tb.is_alive(), "turn B never ended"
    finally:
        resume_a.set()
        ta.join(_WAIT)
    assert _settled(before) == []
    return replies


def _fc3_c3(ctx):
    """The tool loop's direct answer is its own turn's."""
    te, executor, scripted, _ = ctx
    at_take, b_decided, a_took = threading.Event(), threading.Event(), threading.Event()
    take = executor._take_direct_answer_candidate
    decide = executor._decide_tools

    def decide_wrapped(*a, **k):
        out = decide(*a, **k)
        if threading.current_thread().name == "fc3-B":
            b_decided.set()
        return out

    def take_wrapped():
        if threading.current_thread().name == "fc3-A":
            at_take.set()
            b_decided.wait(_WAIT)
            try:
                return take()
            finally:
                a_took.set()
        a_took.wait(_WAIT)
        return take()

    executor._decide_tools = decide_wrapped
    executor._take_direct_answer_candidate = take_wrapped
    replies = _two_tool_turns(executor, at_take, b_decided)
    assert replies.get("B") == "Answer meant for conversation B.", replies
    assert replies.get("A") == "Answer meant for conversation A.", (
        f"A's reply is another turn's answer: {replies}"
    )


def _fc3_c4(ctx):
    """The native response a turn reads is its own."""
    te, executor, scripted, _ = ctx
    paused, b_done = threading.Event(), threading.Event()
    native = executor._native_tool_decision

    def native_wrapped(*a, **k):
        out = native(*a, **k)
        if threading.current_thread().name == "fc3-A":
            paused.set()
            b_done.wait(_WAIT)
        return out

    executor._native_tool_decision = native_wrapped
    original_execute = executor.execute_with_tools

    def execute_marked(*a, **k):
        try:
            return original_execute(*a, **k)
        finally:
            if threading.current_thread().name == "fc3-B":
                b_done.set()

    executor.execute_with_tools = execute_marked
    replies = _two_tool_turns(executor, paused, b_done)
    assert replies.get("B") == "Answer meant for conversation B.", replies
    assert replies.get("A") == "Answer meant for conversation A.", (
        f"A stored another turn's native answer: {replies}"
    )
    assert executor._last_native_response is None, (
        "the native response slot is readable from a thread that never decided"
    )


def _fc3_c5(ae):
    """The agentic executor decides to verify from its own call's results."""
    class Base:
        last_verification_results = [SimpleNamespace(language="another turn")]

        def execute(self, **kw):
            run = kw.get("run")
            if run is not None:
                run.results["verification_results"] = []
            yield "```python\nprint(1)\n```"

    class Engine:
        available = True

        def __init__(self):
            self.calls = 0
            self.results = [SimpleNamespace(language="python", status="ok", iterations=1)]

        def verify_response_code_blocks(self, **kw):
            self.calls += 1
            return self.results

    engine = Engine()
    ae.VERIFICATION_AVAILABLE = True
    ae._select_pipeline = lambda **kw: ae.PIPELINE_CODE_VERIFY
    agent = _agent(ae, executor=Base(), verification_engine=engine)
    for _ in agent.execute("write code", _routing()):
        pass
    assert engine.calls == 1, (
        f"the turn decided from another turn's results: engine called {engine.calls} time(s)"
    )
    assert agent.last_verification_results == engine.results


def test_fc3_results_belong_to_a_turn():
    world, close = _load_route_agentic()
    try:
        _fc3_c1(world)
    finally:
        close()
    loaded, rc, schemas, close = _load_routes_only()
    try:
        _fc3_c2(SimpleNamespace(rc=rc, schemas=schemas))
    finally:
        close()
    for clause in (_fc3_c3, _fc3_c4):
        ctx = _load_tool_executor()
        try:
            clause(ctx)
        finally:
            ctx[3]()
    ae, close = _load_agentic()
    try:
        _fc3_c5(ae)
    finally:
        close()


# ---------------------------------------------------------------------------
# FC6 -- a census of the code, and the emergency stop
# ---------------------------------------------------------------------------

def _text(*parts):
    return source(*parts).read_text(encoding="utf-8")


def _methods(tree, cls):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return []


def _is_property(fn):
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "property":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr in ("setter", "getter", "deleter"):
            return True
    return False


def _self_attr(node, prefix):
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name) and node.value.id == "self"
        and node.attr.startswith(prefix)
    )


def _lastish(name):
    return name.startswith("last_") or name.startswith("_last_")


def _hook_census(text, cls="AgenticExecutor"):
    """Writers of self._on_* other than __init__/reset; readers other than _legacy_turn."""
    out = []
    for fn in _methods(ast.parse(text), cls):
        for node in ast.walk(fn):
            if not _self_attr(node, "_on_"):
                continue
            if isinstance(node.ctx, ast.Store) and fn.name not in ("__init__", "reset"):
                out.append(f"{fn.name} assigns self.{node.attr}")
            if isinstance(node.ctx, ast.Load) and fn.name != "_legacy_turn":
                out.append(f"{fn.name} reads self.{node.attr}")
    return out


def _mirror_read_census(text, cls="AgenticExecutor"):
    """Methods that are not properties and load self._last_*."""
    out = []
    for fn in _methods(ast.parse(text), cls):
        if _is_property(fn):
            continue
        for node in ast.walk(fn):
            if _self_attr(node, "_last_") and isinstance(node.ctx, ast.Load):
                out.append(f"{fn.name} reads self.{node.attr}")
    return out


def _route_census(text, names=("executor", "_agentic_executor")):
    """The routes' reads of a singleton's per-call state, and its cancel/reset."""
    out = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            recv = node.func.value
            if isinstance(recv, ast.Name) and recv.id in names and node.func.attr in ("cancel", "reset"):
                out.append(f"{recv.id}.{node.func.attr}()")
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Name) and node.value.id in names and _lastish(node.attr):
                out.append(f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("getattr", "hasattr") and len(node.args) >= 2:
            recv, const = node.args[0], node.args[1]
            if isinstance(recv, ast.Name) and recv.id in names and isinstance(const, ast.Constant) \
                    and isinstance(const.value, str) and _lastish(const.value):
                out.append(f"{node.func.id}({recv.id}, {const.value!r})")
    return out


_PATH_MODULES = ("path", "os", "posixpath", "ntpath")


def _blocks_the_thread(call):
    """Whether a call, left un-awaited, waits on a thread (by its spelling).

    ``.wait(`` and ``.join(`` (a string or path join excepted), ``time.sleep(``,
    and ``.get(`` / ``.put(`` / ``.result(`` given a timeout (a queue or a
    concurrent future). The census reads names, not types: a blocking wait
    spelled otherwise is not seen.
    """
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    recv = func.value
    timed = any(kw.arg in ("timeout", "block") for kw in call.keywords)
    if func.attr == "wait":
        return not isinstance(recv, ast.Constant)
    if func.attr == "join":
        if isinstance(recv, (ast.Constant, ast.JoinedStr)) or len(call.args) >= 2:
            return False
        name = recv.attr if isinstance(recv, ast.Attribute) else getattr(recv, "id", "")
        return name not in _PATH_MODULES
    if func.attr == "sleep":
        return isinstance(recv, ast.Name) and recv.id == "time"
    if func.attr in ("get", "put"):
        return timed
    if func.attr == "result":
        return timed or bool(call.args)
    return False


def _blocking_census(text):
    """Calls that wait on a thread, un-awaited, whose nearest enclosing function is async.

    A call is awaited when it is the awaited expression itself or a direct
    argument of the awaited call (``await asyncio.wait_for(event.wait(), 1)``).
    """
    out = []

    def visit(node, owner, awaited):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            owner = node
        if isinstance(node, ast.Await):
            awaited = {id(node.value)}
            if isinstance(node.value, ast.Call):
                awaited |= {id(a) for a in node.value.args}
                awaited |= {id(kw.value) for kw in node.value.keywords}
        if (
            isinstance(node, ast.Call)
            and _blocks_the_thread(node)
            and isinstance(owner, ast.AsyncFunctionDef)
            and id(node) not in awaited
        ):
            out.append(f"{owner.name}: line {node.lineno} .{node.func.attr}(")
        for child in ast.iter_child_nodes(node):
            keep = awaited if isinstance(node, (ast.Await, ast.Call)) and awaited else set()
            visit(child, owner, keep)

    visit(ast.parse(text), None, set())
    return out


def _executor_mirror_census(text, cls="AgenticExecutor"):
    """Non-property methods that decide from the executor's per-call mirrors.

    A load of ``self._executor.<last...>`` and a ``getattr`` of such a name
    are reads. A ``hasattr`` reads no value; it is left out only when the
    same method stores that name and never loads it: a presence test that
    guards a write of the executor's mirror (the cascading and speculative
    pipelines keep it for property access) carries no other turn's result.
    Every other ``hasattr`` of such a name is counted.
    """
    out = []
    for fn in _methods(ast.parse(text), cls):
        if _is_property(fn):
            continue
        loads, stores, probes = set(), set(), []
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and _self_attr(node.value, "_executor") \
                    and node.value.attr == "_executor" and _lastish(node.attr):
                (loads if isinstance(node.ctx, ast.Load) else stores).add(node.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in ("getattr", "hasattr") and len(node.args) >= 2:
                recv, const = node.args[0], node.args[1]
                if _self_attr(recv, "_executor") and recv.attr == "_executor" \
                        and isinstance(const, ast.Constant) and isinstance(const.value, str) \
                        and _lastish(const.value):
                    probes.append((node.func.id, const.value))
        for name in sorted(loads):
            out.append(f"{fn.name} reads self._executor.{name}")
        for kind, name in probes:
            if kind == "getattr" or name in loads or name not in stores:
                out.append(f"{fn.name} {kind}(self._executor, {name!r})")
    return out


def _executor_census(text):
    """The executor's process flag, its execute()'s slot reads, and the budget."""
    out = []
    if "_cancel_event" in text:
        out.append("names _cancel_event")
    tree = ast.parse(text)
    for fn in _methods(tree, "Executor"):
        if fn.name != "execute":
            continue
        builds = 0
        for node in ast.walk(fn):
            if _self_attr(node, "_last_") and isinstance(node.ctx, ast.Load):
                out.append(f"execute reads self.{node.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "clear" and isinstance(node.func.value, ast.Attribute) \
                        and isinstance(node.func.value.value, ast.Name) \
                        and node.func.value.value.id == "self":
                    out.append(f"execute clears self.{node.func.value.attr}")
                if node.func.attr == "_build_conversation_messages":
                    builds += 1
                    if "prompt_budget" not in {kw.arg for kw in node.keywords}:
                        out.append(f"line {node.lineno}: a build without prompt_budget=")
        if builds != 2:
            out.append(f"execute builds the conversation {builds} time(s), not 2")
    return out


def _slot_census(cls):
    return [
        name for name in ("_pending_direct_answer", "_last_native_response")
        if not isinstance(inspect.getattr_static(cls, name, None), property)
    ]


def _fc6_c1():
    offenders = _hook_census(_text("agentic_executor.py"))
    assert offenders == [], f"a hook slot carries a turn: {offenders}"


def _fc6_c1b():
    offenders = _mirror_read_census(_text("agentic_executor.py"))
    assert offenders == [], f"a pipeline reads a process mirror: {offenders}"


def _fc6_c2():
    text = _text("api", "routes_chat.py")
    assert _route_census(text) == [], f"the routes read a singleton: {_route_census(text)}"


def _fc6_c2b():
    offenders = _blocking_census(_text("api", "routes_chat.py"))
    assert offenders == [], f"a stream function blocks the event loop: {offenders}"


def _fc6_c3():
    offenders = _executor_mirror_census(_text("agentic_executor.py"))
    assert offenders == [], f"the agentic executor reads the executor's mirror: {offenders}"


def _fc6_c4():
    offenders = _executor_census(_text("executor.py"))
    assert offenders == [], f"the executor keeps a turn on itself: {offenders}"


def _fc6_c5():
    te, executor, scripted, close = _load_tool_executor()
    try:
        offenders = _slot_census(te.ToolExecutor)
    finally:
        close()
    assert offenders == [], f"a per-turn slot of the tool executor is shared: {offenders}"


def _fc6_c6():
    """Witnesses: each census flags a planted text (and passes its negative)."""
    hooks = (
        "class AgenticExecutor:\n"
        "    def execute(self, on_tool_call=None):\n"
        "        self._on_tool_call = on_tool_call\n"
    )
    assert _hook_census(hooks) == ["execute assigns self._on_tool_call"]
    mirrors = (
        "class AgenticExecutor:\n"
        "    def _pipe(self):\n"
        "        return self._last_x\n"
        "    @property\n"
        "    def last_x(self):\n"
        "        return self._last_x\n"
    )
    assert _mirror_read_census(mirrors) == ["_pipe reads self._last_x"]
    assert _route_census("def f():\n    executor.cancel()\n") == ["executor.cancel()"]
    assert _route_census("v = getattr(executor, 'last_vision_meta', {})\n") == [
        "getattr(executor, 'last_vision_meta')"
    ]
    blocking = (
        "async def f(event):\n"
        "    event.wait(5)\n"
        "async def g():\n"
        "    await asyncio.sleep(0.05)\n"
        "    ', '.join(['a'])\n"
    )
    assert _blocking_census(blocking) == ["f: line 2 .wait("]
    spelled = (
        "async def h(t, q, fut, ev, a):\n"
        "    t.join(1)\n"
        "    time.sleep(5)\n"
        "    q.get(timeout=5)\n"
        "    fut.result(timeout=5)\n"
        "    await asyncio.wait_for(ev.wait(), 1)\n"
        "    await f(g(ev.wait()))\n"
        "    os.path.join(a, 'b')\n"
        "    d = {}.get('k')\n"
        "def k(t):\n"
        "    t.join(1)\n"
    )
    assert _blocking_census(spelled) == [
        "h: line 2 .join(", "h: line 3 .sleep(", "h: line 4 .get(",
        "h: line 5 .result(", "h: line 7 .wait(",
    ]
    reads = (
        "class AgenticExecutor:\n"
        "    def _pipe(self):\n"
        "        return self._executor.last_x\n"
        "    def _mirror(self, r):\n"
        "        if hasattr(self._executor, '_last_y'):\n"
        "            self._executor._last_y = r\n"
        "    @property\n"
        "    def last_x(self):\n"
        "        return self._executor.last_x\n"
    )
    assert _executor_mirror_census(reads) == ["_pipe reads self._executor.last_x"]
    probes = (
        "class AgenticExecutor:\n"
        "    def _pipe(self):\n"
        "        return getattr(self._executor, 'last_x', None)\n"
        "    def _probe(self):\n"
        "        return hasattr(self._executor, '_last_y')\n"
    )
    assert _executor_mirror_census(probes) == [
        "_pipe getattr(self._executor, 'last_x')", "_probe hasattr(self._executor, '_last_y')",
    ]
    one_build = (
        "class Executor:\n"
        "    def execute(self):\n"
        "        self._build_conversation_messages(prompt_budget=None)\n"
    )
    assert _executor_census(one_build) == ["execute builds the conversation 1 time(s), not 2"]
    moved = (
        "class Executor:\n"
        "    def execute(self):\n"
        "        self._helper()\n"
        "    def _helper(self):\n"
        "        self._build_conversation_messages(prompt_budget=None)\n"
        "        self._build_conversation_messages(prompt_budget=None)\n"
    )
    assert _executor_census(moved) == ["execute builds the conversation 0 time(s), not 2"]
    execute = (
        "_cancel_event = None\n"
        "class Executor:\n"
        "    def execute(self):\n"
        "        self._x.clear()\n"
        "        self._build_conversation_messages(a=1)\n"
        "        self._build_conversation_messages(prompt_budget=None)\n"
    )
    assert _executor_census(execute) == [
        "names _cancel_event", "execute clears self._x",
        "line 5: a build without prompt_budget=",
    ]
    planted = type("Planted", (), {"_pending_direct_answer": None, "_last_native_response": None})
    assert _slot_census(planted) == ["_pending_direct_answer", "_last_native_response"]


def _load_estop_window():
    calls = []

    def singleton(module_name, cls_name, attr):
        module = types.ModuleType(module_name)
        cls = type(cls_name, (), {"cancel": lambda self: calls.append(cls_name)})
        setattr(module, attr, cls())
        return module

    seeded = {
        "opti_oignon.executor": singleton("opti_oignon.executor", "Executor", "executor"),
        "opti_oignon.agentic_executor": singleton(
            "opti_oignon.agentic_executor", "AgenticExecutor", "agentic_executor",
        ),
    }
    loaded, rc, schemas, close = _load_routes_only(
        extra_targets={"opti_oignon.emergency_stop": source("emergency_stop.py")},
        seeded=seeded,
    )
    return loaded["opti_oignon.emergency_stop"], rc, schemas, calls, close


class _Parked:
    """The route's executor stand-in: each call parks on its run's stop."""

    last_vision_meta = {}
    last_verification_results = []

    def __init__(self):
        self.entered = threading.Event()
        self.released = threading.Event()
        self.private = threading.Event()
        self.cancels = 0

    def reset(self):
        pass

    def cancel(self):
        self.cancels += 1

    def execute(self, **kw):
        self.entered.set()
        run = kw.get("run")
        (run.stop if run is not None else self.private).wait(_WAIT)
        self.released.set()
        yield "parked answer"


def _fc6_c7(ctx):
    """The emergency stop reaches a live chat turn parked outside any executor call."""
    es, rc, schemas, calls, _ = ctx
    parked = _Parked()
    rc.executor = parked
    world = SimpleNamespace(rc=rc, schemas=schemas)
    before = set(threading.enumerate())
    ws = _Socket()
    record = {}

    async def scenario():
        task = await _open(world, ws, "conv-e", "parked question")
        assert await _until(parked.entered.is_set), "the turn never parked"
        started = time.monotonic()
        record["first"] = es._step_cancel_generations()
        record["released"] = await _until(parked.released.is_set, 1.0)
        record["after"] = time.monotonic() - started
        await asyncio.wait_for(task, _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        parked.private.set()
    assert _settled(before) == []
    assert record["released"], (
        f"the emergency stop did not reach the live turn within 1.0 s ({record['after']:.2f} s)"
    )
    assert record["first"]["cancelled"] == ["Executor", "AgenticExecutor", "LiveChatTurns"], record
    saved = sys.modules.pop("opti_oignon.api.routes_chat")
    try:
        record["alone"] = es._step_cancel_generations()
    finally:
        sys.modules["opti_oignon.api.routes_chat"] = saved
    assert record["alone"]["cancelled"] == ["Executor", "AgenticExecutor"], record


def test_fc6_no_process_slot_carries_a_turn_and_the_emergency_stop_reaches_every_turn():
    for clause in (_fc6_c1, _fc6_c1b, _fc6_c2, _fc6_c2b, _fc6_c3, _fc6_c4, _fc6_c5, _fc6_c6):
        clause()
    ctx = _load_estop_window()
    try:
        _fc6_c7(ctx)
    finally:
        ctx[4]()


# ---------------------------------------------------------------------------
# FC8 -- a closed socket stops its turn at once and never holds the loop
# ---------------------------------------------------------------------------

class _ParkingToolStage:
    """A tools stage that yields ``first`` (when given) and then parks."""

    def __init__(self, first=None):
        self.first = first
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.polls = 0
        self.saw_stop_at = None

    def should_use_tools(self, *a, **k):
        return True

    def stream_with_tools(self, **kw):
        self.entered.set()
        if self.first is not None:
            yield self.first
        should_stop = kw.get("should_stop")
        deadline = time.monotonic() + _WAIT
        while time.monotonic() < deadline and not self.gate.is_set():
            self.polls += 1
            if should_stop is not None and should_stop():
                self.saw_stop_at = time.monotonic()
                break
            self.gate.wait(0.01)
        return SimpleNamespace(tool_calls=[], response="", verification_hints=0)


def _fc8_world(world, stage):
    world.ae.TOOL_EXECUTOR_AVAILABLE = True
    world.ae._select_pipeline = lambda **kw: world.ae.PIPELINE_TOOLS
    world.rc._agentic_executor = _agent(world.ae, tool_executor=stage)


def _fc8_c1(world):
    """A dead socket stops the turn and the loop keeps serving meanwhile."""
    stage = _ParkingToolStage(first="first chunk")
    _fc8_world(world, stage)
    parks = stage.stream_with_tools

    def ignoring_the_stop(**kw):
        kw.pop("should_stop", None)
        return parks(**kw)

    stage.stream_with_tools = ignoring_the_stop
    before = set(threading.enumerate())
    ws = _Socket(fail_on="first chunk")
    ticks = []
    seen = {}

    async def ticker():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    async def scenario():
        ws.failed = asyncio.Event()
        tick = asyncio.ensure_future(ticker())
        task = await _open(world, ws, "conv-a", "alpha")
        await asyncio.wait_for(ws.failed.wait(), _WAIT)
        mark = len(ticks)
        assert await _until(lambda: len(ticks) >= mark + 5), "the ticker stopped"
        stage.gate.set()
        await asyncio.wait_for(task, _WAIT)
        seen["ended"] = task.done()
        tick.cancel()

    try:
        asyncio.run(scenario())
    finally:
        stage.gate.set()
    assert _settled(before) == []
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert seen.get("ended"), "the turn never ended"
    assert max(gaps) <= 1.0, f"the event loop froze for {max(gaps):.2f} s on a dead socket"


class _DisconnectingSocket(_Socket):
    """Accepts sends; ``receive()`` answers a disconnect when told to."""

    def __init__(self, never=False):
        super().__init__()
        self.never = never
        self.go = None

    async def receive(self):
        await self.go.wait()
        if self.never:
            await asyncio.Event().wait()
        return {"type": "websocket.disconnect", "code": 1001}


def _fc8_c2(world):
    """A closed tab is noticed at once and stops the stage that sends nothing."""
    stage = _ParkingToolStage()
    _fc8_world(world, stage)
    before = set(threading.enumerate())
    ws = _DisconnectingSocket()
    seen = {}

    async def scenario():
        ws.go = asyncio.Event()
        task = await _open(world, ws, "conv-a", "alpha")
        assert await _until(stage.entered.is_set), "the stage never started"
        seen["closed_at"] = time.monotonic()
        ws.go.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), 1.5)
            seen["returned"] = time.monotonic() - seen["closed_at"]
        except asyncio.TimeoutError:
            seen["returned"] = None
        stage.gate.set()
        await asyncio.wait_for(task, _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        stage.gate.set()
    assert _settled(before) == []
    assert stage.saw_stop_at is not None, "the stage never saw the closed socket's stop"
    assert stage.saw_stop_at - seen["closed_at"] <= 1.0, (
        f"the stop took {stage.saw_stop_at - seen['closed_at']:.2f} s"
    )
    assert seen["returned"] is not None, "the route did not return within 1.5 s"


def _fc8_c3(world):
    """Witness: a socket that never closes never stops the stage."""
    stage = _ParkingToolStage()
    _fc8_world(world, stage)
    before = set(threading.enumerate())
    ws = _DisconnectingSocket(never=True)

    async def scenario():
        ws.go = asyncio.Event()
        ws.go.set()
        task = await _open(world, ws, "conv-a", "alpha")
        assert await _until(lambda: stage.polls >= 20), "the stage never polled"
        stage.gate.set()
        await asyncio.wait_for(task, _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        stage.gate.set()
    assert _settled(before) == []
    assert stage.saw_stop_at is None, "a live socket stopped its turn"
    assert _done(ws)["cancelled"] is False


def test_fc8_a_closed_socket_stops_its_turn_at_once_and_never_holds_the_loop():
    world, close = _load_route_agentic()
    try:
        for clause in (_fc8_c1, _fc8_c2, _fc8_c3):
            clause(world)
    finally:
        close()
