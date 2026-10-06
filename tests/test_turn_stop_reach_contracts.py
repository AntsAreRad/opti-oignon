#!/usr/bin/env python3
"""How far a turn's stop reaches, and what a stopped call leaves behind.

A Stop used to reach one thing: the executor's streaming loop, and only
when a chunk arrived. Everything that runs between model calls -- the
coding agent's phases, the reasoning strategies, a consensus wait, the
self-correction loop, the tool loop, the pipeline runner, a tool waiting
for approval -- went on as if nothing had happened, and a caller that read
a stopped call to its end got it saved, captured and cached. These
contracts pin the reach of the stop and its aftermath:

* FC4 -- the stop reaches the chat coding agent: between phases, before
  files are written, in the fix loop, and the stopped turn is recorded; the
  next turn of the conversation uses its own model callback.
* FC5 -- the stop reaches the model stages: every reasoning strategy, the
  consensus wait and merge, and both self-correction phases, whose failure
  after a stop brings no fallback reply.
* FC7 -- the executor stops its own call, during prefill too, before
  generation starts, during its admission and before its stream opens.
* FC9 -- a stopped call leaves no answer behind: nothing saved, captured,
  curated, cached or measured, and exactly one cancelled ledger record.
* FC10 -- the executor's results and prompt budget belong to its call.
* FC11 -- the stop reaches the tool stages: the tool loop (its decisions,
  the tools it salvages from a narrated answer, its final answer), the
  second phase of think+tools, the approval wait, its withdrawal and a
  decision that lands after the stop, the pipeline runner and code
  verification; and the tools that ran stay recorded when the turn is
  stopped while its answer streams.
* FC12 -- two coding turns of one conversation stay apart, and a turn with
  no conversation never runs on the coding session kept under the empty id.
* FC13 -- supersedes FC9 over the store as it stands, whose origin read
  feeds the onion's mirror.

Every window comes from the shared isolation module and loads the real
modules under contract, with scripted stand-ins behind the registry
bridge. Interleavings are ordered by events, never by sleeps; a wait on a
gate times out after three seconds and fails the clause by name. Every
thread a clause starts, or makes the code start, is joined before the
window closes. Each "no model call after the stop" count comes with a
witness run without the stop, which does make the calls.
"""

import ast
import asyncio
import json
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
    "test_fc4_the_stop_reaches_the_chat_coding_agent": 2.0,
    "test_fc5_the_stop_reaches_the_model_stages": 2.0,
    "test_fc7_the_executor_stops_its_own_call": 2.0,
    "test_fc9_a_stopped_call_leaves_no_answer_behind": 2.0,
    "test_fc10_the_executors_results_belong_to_its_call": 2.0,
    "test_fc11_the_stop_reaches_the_tool_stages": 2.0,
    "test_fc12_two_coding_turns_of_one_conversation_stay_apart": 2.0,
    "test_fc13_a_stopped_call_leaves_no_answer_behind_over_the_store_with_its_origin_read": 2.0,
}

_WAIT = 3.0
_CANCEL = "[Generation cancelled]"


# ---------------------------------------------------------------------------
# Shared material
# ---------------------------------------------------------------------------

def _routing(model="m"):
    return SimpleNamespace(
        model=model, task_type="general", temperature=0.2,
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


def _settled(before, timeout=_WAIT):
    """Join every thread started since ``before``; name the ones still alive."""
    deadline = time.monotonic() + timeout
    for thread in [t for t in threading.enumerate() if t not in before]:
        thread.join(max(0.0, deadline - time.monotonic()))
    return [t.name for t in threading.enumerate() if t not in before and t.is_alive()]


async def _until(predicate, timeout=_WAIT):
    """Wait on a condition set from another thread; False at the deadline."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.005)
    return True


class _Socket:
    """A chat socket stand-in; ``fail_type`` frames raise and set ``failed``."""

    def __init__(self, fail_type=None):
        self.sent = []
        self.fail_type = fail_type
        self.failed = None

    async def send_json(self, data):
        if self.fail_type is not None and data.get("type") == self.fail_type:
            if self.failed is not None:
                self.failed.set()
            raise ConnectionError("socket closed by the client")
        self.sent.append(data)


def _frames(ws, kind):
    return [d for d in ws.sent if d.get("type") == kind]


def _done(ws):
    frames = _frames(ws, "done")
    return frames[-1]["metadata"] if frames else None


def _load_routes_only(extra_targets=None, seeded=None):
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


async def _open(world, ws, conv, message, **fields):
    request = world.schemas.ChatRequest(conversation_id=conv or None, message=message, **fields)
    return asyncio.ensure_future(world.rc._stream_response(ws, conv, message, request))


async def _stop(world, conv):
    return await world.rc.cancel_generation(
        world.schemas.ChatCancelRequest(conversation_id=conv)
    )


# ---------------------------------------------------------------------------
# FC4 and FC12 -- the chat coding agent
# ---------------------------------------------------------------------------

_LONG = (
    "/code build a small python package with one module and one test file "
    "that prints a single greeting line when it runs please"
)


class _Sandbox:
    """Sandbox handlers: writes and commands are recorded, the listing is the writes."""

    def __init__(self, test_output="1 failed"):
        self.writes = []
        self.commands = []
        self.test_output = test_output
        self.gate = None
        self.parked = threading.Event()

    def bash(self, sid, cmd, timeout, _sandbox_manager=None):
        self.commands.append(cmd)
        if self.gate is not None and len(self.commands) == 1:
            self.parked.set()
            self.gate.wait(_WAIT)
        return self.test_output

    def create(self, sid, path, content, _sandbox_manager=None):
        self.writes.append(path)
        return "ok"

    def view(self, sid, path, a, b, _sandbox_manager=None):
        return "\n".join(dict.fromkeys(self.writes))


class _Box:
    active = True


class _SandboxManager:
    def create_sandbox(self, sid, allow_degraded=True):
        return _Box()

    def extract_files(self, sid):
        return []

    def destroy_sandbox(self, sid):
        return True


class _Conversation:
    def __init__(self):
        self.added = []

    def get_context_messages(self, conv_id):
        return []

    def add_message(self, conv_id, role, content, **kw):
        self.added.append((role, content))


class _CodingModel:
    """The route's executor stand-in on the coding path.

    Each call records its routing model and the run it was handed, yields a
    numbered plan line and a FILE block, and parks after its text when its
    index is in ``park`` (the gate ignores the stop).
    """

    last_vision_meta = {}
    last_verification_results = []

    def __init__(self, park=()):
        self.calls = []
        self.park = set(park)
        self.gate = threading.Event()
        self.parked = threading.Event()

    def reset(self):
        pass

    def cancel(self):
        pass

    def execute(self, **kw):
        index = len(self.calls) + 1
        self.calls.append((kw["routing"].model, kw.get("run")))
        yield "1. write a.py\n--- FILE: a.py ---\nprint(1)\n--- END FILE ---\n"
        if index in self.park:
            self.parked.set()
            self.gate.wait(_WAIT)


def _coding_window():
    """The chat routes over the real chat coding agent (loaded once per contract)."""
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = None
    loaded, rc, schemas, close = _load_routes_only(
        extra_targets={"opti_oignon.chat_coding_agent": source("chat_coding_agent.py")},
        seeded={"opti_oignon.conversation": conv},
    )
    cca = loaded["opti_oignon.chat_coding_agent"]
    cca.CHAT_CODING_AVAILABLE = True
    rc.CHAT_CODING_AVAILABLE = True
    rc._resolve_model_and_route = lambda message, request: (
        _routing("m2" if "second" in message else "m1"), None,
    )
    return SimpleNamespace(rc=rc, schemas=schemas, cca=cca), close


def _coding_world(window, park=(), test_output="1 failed"):
    """Fresh sandbox, conversation, session manager and model for one clause."""
    cca, rc = window.cca, window.rc
    conversation = _Conversation()
    cca._conversation_manager = conversation
    sandbox = _Sandbox(test_output)
    cca._handle_sandbox_bash = sandbox.bash
    cca._handle_sandbox_create_file = sandbox.create
    cca._handle_sandbox_view = sandbox.view
    manager = cca.ChatCodingManager(
        sandbox_mgr=_SandboxManager(),
        config=cca.ChatCodingConfig(enabled=True, max_fix_retries=3, auto_test=True),
    )
    rc._chat_coding_manager = manager
    model = _CodingModel(park)
    rc.executor = model
    return SimpleNamespace(
        rc=rc, schemas=window.schemas, cca=cca, sandbox=sandbox, model=model,
        manager=manager, conversation=conversation,
    )


def _load_route_coding(park=(), test_output="1 failed"):
    window, close = _coding_window()
    return _coding_world(window, park, test_output), close


def _coding_done(ws):
    frames = _frames(ws, "coding_done")
    return frames[-1]["metadata"] if frames else {}


def _run_coding(world, ws, stop, release_after=None, message=_LONG):
    """One coding turn: park, optionally Stop through the route, release."""
    before = set(threading.enumerate())

    async def scenario():
        if ws.fail_type is not None:
            ws.failed = asyncio.Event()
        task = await _open(world, ws, "conv-c", message)
        if world.model.park:
            assert await _until(world.model.parked.is_set), "the model call never parked"
        if release_after is not None:
            await asyncio.wait_for(release_after(ws), _WAIT)
        if stop:
            await _stop(world, "conv-c")
        world.model.gate.set()
        await asyncio.wait_for(task, _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        world.model.gate.set()
    assert _settled(before) == [], "a thread outlived the clause"


def _fc4_c1(world, stop=True):
    """A stop during the plan's model call ends the turn before implement."""
    ws = _Socket()
    _run_coding(world, ws, stop=stop)
    if not stop:
        assert len(world.model.calls) >= 2 and "a.py" in world.sandbox.writes
        return
    assert len(world.model.calls) == 1, (
        f"the coding agent made {len(world.model.calls)} model calls after its stop"
    )
    assert world.sandbox.writes == [], f"files were written after the stop: {world.sandbox.writes}"
    implementing = [f for f in _frames(ws, "coding_status") if f.get("content") == "Implementing..."]
    assert implementing == [], "the implement phase started after the stop"
    done = _coding_done(ws)
    assert done.get("stopped") is True and done.get("stopped_during") == "plan", done
    assert _done(ws)["cancelled"] is True, "the coding turn's done says not cancelled"


def _fc4_c2(world, stop=True):
    """A stopped implement call writes no file, even with a FILE block in hand."""
    ws = _Socket()
    _run_coding(world, ws, stop=stop)
    if not stop:
        assert "a.py" in world.sandbox.writes, "the witness wrote no file"
        return
    assert world.sandbox.writes == [], f"a stopped call's files were applied: {world.sandbox.writes}"
    assert _coding_done(ws).get("stopped_during") == "implement", _coding_done(ws)


def _fc4_c3(world, closed=True):
    """A socket closed during the fix loop stops it: no further attempt, no command."""
    ws = _Socket(fail_type="coding_fix" if closed else None)

    async def after_failure(sock):
        if closed:
            await sock.failed.wait()

    _run_coding(world, ws, stop=False, release_after=after_failure)
    if not closed:
        assert len(world.sandbox.commands) >= 3, "the witness made no second fix attempt"
        return
    assert len(world.sandbox.commands) == 1, (
        f"sandbox commands ran after the socket closed: {world.sandbox.commands}"
    )
    assert len(world.model.calls) == 3, (
        f"a further fix attempt ran after the socket closed: {len(world.model.calls)} calls"
    )


def _fc4_c4(world):
    """The stopped turn is recorded, with the files it wrote."""
    ws = _Socket()
    _run_coding(world, ws, stop=True)
    session = world.manager.get_session("conv-c")
    summary = session.sandbox_state.cumulative_summary
    assert "Stopped during fix" in summary, f"the session did not record the stop: {summary!r}"
    replies = [c for role, c in world.conversation.added if role == "assistant"]
    assert replies and "Stopped during fix" in replies[-1] and "a.py" in replies[-1], replies


def _fc4_c5(world):
    """A later turn of the conversation uses its own model callback and run."""
    ws1, ws2 = _Socket(), _Socket()
    _run_coding(world, ws1, stop=True, message=_LONG.replace("build", "first build"))
    first = list(world.model.calls)
    assert first and first[0][0] == "m1"
    world.model.park = set()
    world.sandbox.test_output = "1 passed"
    _run_coding(world, ws2, stop=False, message=_LONG.replace("build", "second build"))
    second = world.model.calls[len(first):]
    assert second, "turn 2 made no model call"
    assert {m for m, _ in second} == {"m2"}, f"turn 2's calls carried another turn's routing: {second}"
    runs = {id(r) for _, r in second}
    assert len(runs) == 1 and second[0][1] is not None, "turn 2's calls did not carry one run"
    assert all(r is not second[0][1] for _, r in first), "turn 2 reused turn 1's run"


def test_fc4_the_stop_reaches_the_chat_coding_agent():
    window, close = _coding_window()
    try:
        _fc4_clauses(window)
    finally:
        close()


def _fc4_clauses(window):
    for clause, park, output, kwargs in (
        (_fc4_c1, (1,), "1 passed", {}),
        (_fc4_c1, (), "1 passed", {"stop": False}),
        (_fc4_c2, (2,), "1 passed", {}),
        (_fc4_c2, (2,), "1 passed", {"stop": False}),
        (_fc4_c3, (3,), "1 failed", {}),
        (_fc4_c3, (3,), "1 failed", {"closed": False}),
        (_fc4_c4, (3,), "1 failed", {}),
        (_fc4_c5, (1,), "1 passed", {}),
    ):
        clause(_coding_world(window, park, output), **kwargs)


def _fc12_c1(world):
    """Turn 2 waits for turn 1 to end, then runs on its own callback and run."""
    before = set(threading.enumerate())
    world.sandbox.gate = threading.Event()
    world.model.park = {3}
    ws1, ws2 = _Socket(), _Socket()
    seen = {}

    async def scenario():
        t1 = await _open(world, ws1, "conv-c", _LONG.replace("build", "first build"))
        assert await _until(world.sandbox.parked.is_set), "turn 1 never reached its test command"
        await _stop(world, "conv-c")
        t2 = await _open(world, ws2, "conv-c", _LONG.replace("build", "second build"))
        assert await _until(lambda: any(
            "Waiting for the previous coding turn" in f.get("content", "")
            for f in _frames(ws2, "coding_status")
        )), "turn 2 never said it was waiting"
        seen["calls_while_waiting"] = len(world.model.calls)
        world.sandbox.gate.set()
        await asyncio.wait_for(t1, _WAIT)
        assert await _until(world.model.parked.is_set), "turn 2 never reached its plan call"
        await _stop(world, "conv-c")
        world.model.gate.set()
        await asyncio.wait_for(t2, _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        world.sandbox.gate.set()
        world.model.gate.set()
    assert _settled(before) == []
    assert seen["calls_while_waiting"] == 2, (
        f"turn 2 made a model call while turn 1 was live: {world.model.calls}"
    )
    done1 = _coding_done(ws1)
    assert done1.get("stopped") is True and done1.get("stopped_during") == "test", done1
    assert _frames(ws1, "coding_fix") == [], "turn 1 made a fix attempt after its stop"
    first, second = world.model.calls[:2], world.model.calls[2:]
    run1 = first[0][1]
    assert run1 is not None and all(r is run1 for _, r in first), "turn 1's calls carried another run"
    assert _done(ws1)["turn_count"] == 1, _done(ws1)
    assert second and {m for m, _ in second} == {"m2"}, f"turn 2's calls: {second}"
    run2 = second[0][1]
    assert run2 is not None and run2 is not run1 and all(r is run2 for _, r in second)
    assert _coding_done(ws2).get("stopped") is True, "turn 2's own stop did not reach it"


def _fc12_c2(world, conv=""):
    """A turn with no conversation never runs on a coding session: every such
    turn would share the one kept under the empty id."""
    before = set(threading.enumerate())
    ws = _Socket()

    async def scenario():
        task = await _open(world, ws, conv, _LONG)
        await asyncio.wait_for(task, _WAIT)

    try:
        asyncio.run(scenario())
    finally:
        world.model.gate.set()
    assert _settled(before) == [], "a thread outlived the clause"
    sessions = [s["conversation_id"] for s in world.manager.list_sessions()]
    if conv:
        assert sessions == [conv] and _coding_done(ws), "the witness ran no coding session"
        return
    assert sessions == [], f"a turn with no conversation opened a coding session: {sessions}"
    assert _frames(ws, "coding_status") == [] and world.sandbox.writes == [], (
        "a turn with no conversation ran the coding agent"
    )
    assert _frames(ws, "metadata")[0]["metadata"]["chat_coding"] is False
    assert len(world.model.calls) == 1 and _done(ws)["cancelled"] is False, (
        "the turn did not run as a plain reply"
    )


def test_fc12_two_coding_turns_of_one_conversation_stay_apart():
    window, close = _coding_window()
    try:
        _fc12_c1(_coding_world(window, (), "1 failed"))
        _fc12_c2(_coding_world(window, (), "1 passed"), conv="conv-c")
        _fc12_c2(_coding_world(window, (), "1 passed"))
    finally:
        close()


# ---------------------------------------------------------------------------
# FC5 -- the stop reaches the model stages
# ---------------------------------------------------------------------------

class _StageBackend:
    """A registry backend for the stage engines: counts calls, parks one."""

    def __init__(self, answer, park_at=None):
        self.calls = []
        self.answer = answer
        self.park_at = park_at
        self.entered = threading.Event()
        self.gate = threading.Event()

    def generate(self, model=None, messages=None, options=None, **kw):
        self.calls.append(model)
        if self.park_at is not None and len(self.calls) == self.park_at:
            self.entered.set()
            self.gate.wait(_WAIT)
        return SimpleNamespace(content=self.answer(len(self.calls), model, messages))


def _fc5_c1(stop=True):
    """A reasoning turn stopped during a step makes no further model call."""
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = None
    loaded, rc, schemas, close = _load_routes_only(
        extra_targets={
            "opti_oignon.reasoning": source("reasoning.py"),
            "opti_oignon.agentic_executor": source("agentic_executor.py"),
        },
        seeded={"opti_oignon.conversation": conv},
    )
    try:
        rs, ae = loaded["opti_oignon.reasoning"], loaded["opti_oignon.agentic_executor"]
        steps = json.dumps([{"title": f"s{i}", "question": f"q{i}"} for i in range(4)])
        backend = _StageBackend(
            lambda n, model, messages: steps if n == 1 else "partial answer",
            park_at=2 if stop else None,
        )
        rs._resolve_backend = lambda model: backend
        engine = rs.ReasoningEngine(config=rs.ReasoningConfig())
        ae.REASONING_AVAILABLE = True
        ae._select_pipeline = lambda **kw: ae.PIPELINE_REASONING
        agent = ae.AgenticExecutor(executor=SimpleNamespace(), reasoning_engine=engine, default_model="m")
        agent._classify_message = lambda *a, **k: {"needs_tools": False}
        rc.AGENTIC_EXECUTOR_AVAILABLE = True
        rc._agentic_executor = agent
        rc.executor = SimpleNamespace(reset=lambda: None, cancel=lambda: None,
                                      last_vision_meta={}, last_verification_results=[])
        world = SimpleNamespace(rc=rc, schemas=schemas)
        before = set(threading.enumerate())
        ws = _Socket()

        async def scenario():
            task = await _open(world, ws, "conv-r", "why")
            if stop:
                assert await _until(backend.entered.is_set), "step 1 never started"
                await _stop(world, "conv-r")
                backend.gate.set()
            await asyncio.wait_for(task, _WAIT)

        try:
            asyncio.run(scenario())
        finally:
            backend.gate.set()
        assert _settled(before) == []
        if not stop:
            assert len(backend.calls) == 6, f"the witness made {len(backend.calls)} calls, not 6"
            return
        assert len(backend.calls) <= 2, (
            f"the reasoning engine made {len(backend.calls)} model calls after a stop at step 1"
        )
        assert _done(ws)["cancelled"] is True
    finally:
        close()


def _load_leaf(name, filename):
    loaded, close = isolate(targets={f"opti_oignon.{name}": source(filename)}, seeded={},
                            packages=("opti_oignon",))
    return loaded[f"opti_oignon.{name}"], close


def _fc5_c2():
    """Every reasoning strategy checks the stop before each model call."""
    rs, close = _load_leaf("reasoning", "reasoning.py")
    try:
        for strategy in ("tree_of_thought", "self_consistency"):
            for stopped in (False, True):
                backend = _StageBackend(lambda n, model, messages: "[{\"approach\": \"a\"}, {\"approach\": \"b\"}]")
                rs._resolve_backend = lambda model: backend
                engine = rs.ReasoningEngine(config=rs.ReasoningConfig())
                kwargs = {"should_stop": (lambda: len(backend.calls) >= 1)} if stopped else {}
                getattr(engine, strategy)("why", **kwargs)
                if stopped:
                    assert len(backend.calls) == 1, (
                        f"{strategy} made {len(backend.calls)} calls after its stop"
                    )
                else:
                    assert len(backend.calls) > 1, f"the {strategy} witness made one call"
    finally:
        close()


def _fc5_c3():
    """A stopped consensus stops waiting for the slow model and merges nothing."""
    cs, close = _load_leaf("consensus", "consensus.py")
    try:
        for stopped in (False, True):
            gate, judge = threading.Event(), []

            class Backend:
                def generate(self, model=None, messages=None, options=None, **kw):
                    if model == "judge":
                        judge.append(model)
                        return SimpleNamespace(content="merged answer from the judge")
                    if model == "m3":
                        gate.wait(_WAIT)
                    return SimpleNamespace(content=f"answer from {model} with several words")

            backend = Backend()
            cs._resolve_backend = lambda model: backend
            engine = cs.ConsensusEngine(config=cs.ConsensusConfig(
                default_models=["m1", "m2", "m3"], judge_model="judge", timeout_per_model=5,
                strategy="llm_merge",
            ))
            if not stopped:
                gate.set()
                result = engine.run_consensus("q", models=["m1", "m2", "m3"], strategy="llm_merge")
                assert len(result.individual_responses) == 3 and judge == ["judge"], (
                    "the witness did not query three models and merge once"
                )
                continue
            before = set(threading.enumerate())
            stop, done_at = threading.Event(), {}

            def on_done(resp):
                done_at.setdefault("n", 0)
                done_at["n"] += 1
                if done_at["n"] == 2:
                    stop.set()
                    done_at["stopped"] = time.monotonic()

            try:
                responses = engine.query_models_parallel(
                    [{"role": "user", "content": "q"}], models=["m1", "m2", "m3"],
                    on_model_done=on_done, should_stop=stop.is_set,
                )
                returned = time.monotonic()
                stop_merge, counted = threading.Event(), []

                def on_done_merge(resp):
                    counted.append(resp)
                    if len(counted) == 2:
                        stop_merge.set()

                result = engine.run_consensus(
                    "q", models=["m1", "m2", "m3"], strategy="llm_merge",
                    on_model_done=on_done_merge, should_stop=stop_merge.is_set,
                )
            finally:
                gate.set()
            assert _settled(before) == []
            assert len(responses) == 2, f"the stopped query returned {len(responses)} responses"
            assert returned - done_at["stopped"] <= 1.0, (
                f"the query waited {returned - done_at['stopped']:.2f} s after its stop"
            )
            assert judge == [], "a judge merged the responses of a stopped consensus"
            assert result.metadata.get("stopped") is True, result.metadata
    finally:
        close()


def _fc5_c4i():
    """Self-correction: a stop during phase 1 means the engine is never called."""
    ae, close = _load_leaf("agentic_executor", "agentic_executor.py")
    try:
        for stopped in (False, True):
            entered = threading.Event()

            class Base:
                def execute(self, **kw):
                    yield "a first draft of the answer"
                    entered.set()
                    if stopped:
                        kw["run"].stop.wait(_WAIT)

            class Engine:
                available = True
                calls = 0

                def execute_self_correction(self, **kw):
                    Engine.calls += 1
                    yield "corrected"

            ae.SELF_CORRECTION_AVAILABLE = True
            agent = ae.AgenticExecutor(executor=Base(), self_correction_engine=Engine(), default_model="m")
            run = SimpleNamespace(stop=threading.Event(), results={})
            before = set(threading.enumerate())
            worker = threading.Thread(
                target=lambda: list(agent.execute("q", _routing(), self_correct=True, run=run)),
                name="fc5-self-correct",
            )
            worker.start()
            try:
                assert entered.wait(_WAIT), "phase 1 never produced its draft"
                if stopped:
                    run.stop.set()
                worker.join(_WAIT)
            finally:
                run.stop.set()
            assert _settled(before) == []
            if stopped:
                assert Engine.calls == 0, "the correction engine ran after a stop in phase 1"
            else:
                assert Engine.calls == 1, "the witness never reached the correction engine"
    finally:
        close()


def _fc5_c4iv():
    """Self-correction: an engine that fails after the stop brings no fallback reply."""
    ae, close = _load_leaf("agentic_executor", "agentic_executor.py")
    try:
        for stopped in (False, True):
            run = SimpleNamespace(stop=threading.Event(), results={})

            class Base:
                def execute(self, **kw):
                    yield "a first draft of the answer"

            class Engine:
                available = True

                def execute_self_correction(self, **kw):
                    if stopped:
                        run.stop.set()
                    raise RuntimeError("the correction engine failed")
                    yield "never"  # a generator, as the real engine is

            ae.SELF_CORRECTION_AVAILABLE = True
            agent = ae.AgenticExecutor(executor=Base(), self_correction_engine=Engine(), default_model="m")
            out = [c for c in agent.execute("q", _routing(), self_correct=True, run=run)
                   if isinstance(c, str)]
            if stopped:
                assert out == [], f"a stopped turn got the fallback reply: {out}"
            else:
                assert out == ["a first draft of the answer"], f"the witness had no fallback: {out}"
    finally:
        close()


def _load_self_correction():
    class Client:
        def chat(self, **kwargs):
            return {"message": {"content": "ok"}}

    seeded = {}
    seed_registry(seeded, Client())
    loaded, close = isolate(
        targets={"opti_oignon.self_correction": source("self_correction.py")},
        seeded=seeded, packages=("opti_oignon",),
    )
    mod = loaded["opti_oignon.self_correction"]
    engine = mod.SelfCorrectionEngine(config=mod.SelfCorrectionConfig(
        enable_auto=True, max_iterations=2, compliance_threshold=0.7,
        quality_threshold=0.6, check_instructions=True, check_facts=True,
        check_quality=True,
    ))
    return mod, engine, close


def _fc5_c4ii():
    """correct() checks the stop before each model check."""
    for stopped in (False, True):
        mod, engine, close = _load_self_correction()
        try:
            checks = []
            engine.check_compliance = lambda *a, **k: checks.append("compliance") or mod.ComplianceResult(score=1.0)
            engine.check_facts = lambda *a, **k: checks.append("facts") or mod.FactCheckResult(confidence=1.0)
            engine.check_quality = lambda *a, **k: checks.append("quality") or mod.QualityResult(overall_score=1.0)
            kwargs = {"should_stop": (lambda: len(checks) >= 1)} if stopped else {}
            engine.correct("answer in a list", "an answer", use_llm=True, **kwargs)
            if stopped:
                assert checks == ["compliance"], f"checks ran after the stop: {checks}"
            else:
                assert checks == ["compliance", "facts", "quality"], checks
        finally:
            close()


def _fc5_c4iii():
    """correct() checks the stop at the top of each correction iteration."""
    for stopped in (False, True):
        mod, engine, close = _load_self_correction()
        try:
            engine.check_compliance = lambda *a, **k: mod.ComplianceResult(score=0.3)
            engine.check_facts = lambda *a, **k: mod.FactCheckResult(confidence=1.0)
            engine.check_quality = lambda *a, **k: mod.QualityResult(overall_score=0.3)
            mod.compute_heuristic_compliance = lambda *a, **k: mod.ComplianceResult(score=0.3)
            mod.compute_heuristic_quality = lambda *a, **k: mod.QualityResult(overall_score=0.3)
            corrections = []

            def generate(*a, **k):
                corrections.append(1)
                return f"a corrected answer, still failing, number {len(corrections)}"

            engine._generate_correction = generate
            kwargs = {"should_stop": (lambda: len(corrections) >= 1)} if stopped else {}
            engine.correct("answer in a list", "an answer", use_llm=True, **kwargs)
            if stopped:
                assert corrections == [1], f"{len(corrections)} correction calls after the stop"
            else:
                assert len(corrections) == 2, "the witness made no second correction call"
        finally:
            close()


def test_fc5_the_stop_reaches_the_model_stages():
    _fc5_c1(stop=False)
    _fc5_c1()
    for clause in (_fc5_c2, _fc5_c3, _fc5_c4i, _fc5_c4ii, _fc5_c4iii, _fc5_c4iv):
        clause()


# ---------------------------------------------------------------------------
# FC7, FC9, FC10 -- the executor's call
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.records = []

    def record(self, **fields):
        self.records.append(fields)
        return True


class _ExecClient:
    """The client behind the registry: plays ``factory(kwargs)`` per call."""

    def __init__(self):
        self.calls = []
        self.factory = lambda kw: iter([{"message": {"content": "Hello"}},
                                        {"message": {"content": " world"}}])

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.factory(kwargs)


class _ParkedStream:
    """A stream that parks before its first chunk; its end is observable."""

    def __init__(self, chunks=("Hello", " world"), raise_after_gate=False):
        self.chunks = chunks
        self.raise_after_gate = raise_after_gate
        self.gate = threading.Event()
        self.parked = threading.Event()
        self.ended = threading.Event()
        self.produced = 0
        self.closed_early = None

    def __call__(self, kw):
        return self._stream()

    def _stream(self):
        try:
            self.parked.set()
            self.gate.wait(_WAIT)
            if self.raise_after_gate:
                raise RuntimeError("the backend failed after the stop")
            for chunk in self.chunks:
                self.produced += 1
                yield {"message": {"content": chunk}}
        finally:
            self.closed_early = self.produced < len(self.chunks)
            self.ended.set()


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_exec(extra_seeded=None, packages=()):
    recorder, client = _Recorder(), _ExecClient()
    cfg = _module("opti_oignon.config", config=SimpleNamespace(
        get_model=lambda *a, **k: "test-model:1b", get_temperature=lambda *a, **k: 0.2,
    ))
    router = _module("opti_oignon.router", RoutingResult=type("RoutingResult", (), {}))
    seeded = {
        "opti_oignon.config": cfg,
        "opti_oignon.router": router,
        "opti_oignon.context_ledger": _module(
            "opti_oignon.context_ledger", get_context_ledger=lambda: recorder,
        ),
    }
    seeded.update(extra_seeded or {})
    seed_registry(seeded, client)
    loaded, close = isolate(
        targets={
            "opti_oignon.context_dedup": source("context_dedup.py"),
            "opti_oignon.executor": source("executor.py"),
        },
        seeded=seeded,
        packages=packages,
    )
    return SimpleNamespace(mod=loaded["opti_oignon.executor"], client=client, recorder=recorder), close


def _drive(gen):
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as stop:
        return chunks, stop.value


def _outcomes(world):
    return [r["outcome"] for r in world.recorder.records]


def _fc7_c1(world):
    """cancel() reaches a call parked in prefill; the call returns at once."""
    stream = _ParkedStream()
    world.client.factory = stream
    ex = world.mod.Executor()
    before = set(threading.enumerate())
    marks = {}

    def cancel_when_parked():
        if stream.parked.wait(_WAIT):
            marks["cancel"] = time.monotonic()
            ex.cancel()

    canceller = threading.Thread(target=cancel_when_parked, name="fc7-cancel")
    canceller.start()
    try:
        chunks, (refined, response) = _drive(ex.execute("q", _routing(), refine=False))
        marks["returned"] = time.monotonic()
    finally:
        stream.gate.set()
    assert stream.ended.wait(_WAIT)
    assert _settled(before) == []
    assert _CANCEL in response, response
    assert marks["returned"] - marks["cancel"] <= 1.0, (
        f"the call returned {marks['returned'] - marks['cancel']:.2f} s after its stop"
    )
    assert _outcomes(world) == ["cancelled"], _outcomes(world)
    assert stream.produced == 1 and stream.closed_early is True, (
        f"the stream after the stop: produced {stream.produced}, closed early {stream.closed_early}"
    )


def _gov_module(admissions, on_admit=None, on_ticket=None):
    """A recording governor; ``on_admit`` runs inside the admission and
    ``on_ticket`` when the producer takes its ticket, just before its stream."""
    decision = SimpleNamespace(admitted=True, action="admit", reason="", num_ctx=None,
                               keep_alive=None, conditional_on_eviction=False, load_expected=False)

    def admit(model, requested_ctx=None, caller="chat", extra_models=None):
        admissions.append(model)
        if on_admit is not None:
            on_admit()
        return decision

    def set_active_ticket(decision):
        if on_ticket is not None:
            on_ticket()

    return _module(
        "opti_oignon.resource_governor",
        get_resource_governor=lambda: SimpleNamespace(
            config=SimpleNamespace(enabled=True), admit=admit,
        ),
        set_active_ticket=set_active_ticket,
        clear_active_ticket=lambda: None,
        ticket_scope=lambda decision: None,
    )


def _fc7_c2():
    """A stop that arrives before generation: no stream, no admission, one record."""
    for stopped in (False, True):
        admissions, holder = [], {}

        def get(*a, **k):
            if stopped:
                holder["ex"].cancel()
            return None

        sc = _module("opti_oignon.semantic_cache", semantic_cache=SimpleNamespace(enabled=True, get=get))
        world, close = _load_exec({
            "opti_oignon.semantic_cache": sc,
            "opti_oignon.resource_governor": _gov_module(admissions),
        })
        try:
            ex = world.mod.Executor()
            holder["ex"] = ex
            before = set(threading.enumerate())
            chunks, (refined, response) = _drive(ex.execute("q", _routing(), refine=False))
            assert _settled(before) == [], "a thread outlived the call"
            if not stopped:
                assert len(world.client.calls) == 1, "the witness never reached the backend"
                assert admissions == ["m"], f"the witness took no admission: {admissions}"
                continue
            assert world.client.calls == [], "a call stopped before generation opened a stream"
            assert admissions == [], "a call stopped before generation took an admission"
            assert response == "[Cancelled]", response
            assert _outcomes(world) == ["cancelled"], _outcomes(world)
        finally:
            close()


def _fc7_c5():
    """A stop during the admission, or as the producer takes its ticket, opens no stream."""
    for where in (None, "admission", "ticket"):
        admissions, holder = [], {}

        def stop_at(place):
            return (lambda: holder["ex"].cancel()) if where == place else None

        world, close = _load_exec({
            "opti_oignon.resource_governor": _gov_module(
                admissions, on_admit=stop_at("admission"), on_ticket=stop_at("ticket"),
            ),
        })
        try:
            ex = world.mod.Executor()
            holder["ex"] = ex
            before = set(threading.enumerate())
            chunks, (refined, response) = _drive(ex.execute("q", _routing(), refine=False))
            assert _settled(before) == [], "a thread outlived the call"
            assert admissions == ["m"], f"the admission was not asked once: {admissions}"
            if where is None:
                assert len(world.client.calls) == 1 and response == "Hello world", (
                    "the witness never streamed"
                )
                assert _outcomes(world) == ["completed"], _outcomes(world)
                continue
            assert world.client.calls == [], f"a stop during the {where} still opened a stream"
            if where == "admission":
                assert response == "[Cancelled]", (
                    f"a stop during the admission went on to the producer: {response!r}"
                )
            else:
                assert _CANCEL in response, response
            assert _outcomes(world) == ["cancelled"], _outcomes(world)
        finally:
            close()


def _fc7_c3(world):
    """A run's stop stops that call; a run-less call on the same instance completes."""
    alpha = _ParkedStream(chunks=("a1", "a2", "a3"))
    beta = _ParkedStream(chunks=("b1", "b2", "b3"))
    world.client.factory = lambda kw: (alpha if "alpha" in kw["messages"][-1]["content"] else beta)(kw)
    ex = world.mod.Executor()
    run = SimpleNamespace(stop=threading.Event(), results={})
    out = {}
    before = set(threading.enumerate())

    def drain(key, **kwargs):
        chunks, value = _drive(ex.execute(f"{key} q", _routing(), refine=False, **kwargs))
        out[key] = value[1]

    ta = threading.Thread(target=drain, args=("alpha",), kwargs={"run": run}, name="fc7-alpha")
    tb = threading.Thread(target=drain, args=("beta",), name="fc7-beta")
    ta.start()
    tb.start()
    try:
        assert alpha.parked.wait(_WAIT) and beta.parked.wait(_WAIT), "a stream never started"
        run.stop.set()
        beta.gate.set()
        tb.join(_WAIT)
        ta.join(_WAIT)
    finally:
        alpha.gate.set()
        beta.gate.set()
    assert _settled(before) == []
    assert "alpha" in out and "beta" in out, f"a call never returned: {out}"
    assert _CANCEL in out["alpha"], out
    assert out["beta"] == "b1b2b3", f"another call's stop reached the run-less call: {out}"


def _fc7_c4(world):
    """Witness: the parked stream with no stop completes its chunks."""
    stream = _ParkedStream()
    world.client.factory = stream
    ex = world.mod.Executor()
    before = set(threading.enumerate())
    opener = threading.Thread(target=lambda: stream.parked.wait(_WAIT) and stream.gate.set(),
                              name="fc7-open")
    opener.start()
    try:
        chunks, (refined, response) = _drive(ex.execute("q", _routing(), refine=False))
    finally:
        stream.gate.set()
    assert _settled(before) == []
    assert response == "Hello world" and stream.produced == 2
    assert _outcomes(world) == ["completed"]


def test_fc7_the_executor_stops_its_own_call():
    for clause in (_fc7_c1, _fc7_c3, _fc7_c4):
        world, close = _load_exec()
        try:
            clause(world)
        finally:
            close()
    _fc7_c2()
    _fc7_c5()


class _Sinks:
    """Recording stand-ins for everything a finished answer is kept in."""

    def __init__(self):
        self.added, self.captured, self.puts, self.sem_puts = [], [], [], []
        self.embeds, self.perf, self.curated = [], [], []

    def seeded(self):
        sinks = self
        conv = SimpleNamespace(
            get_context_messages=lambda conv_id: [],
            add_message=lambda conv_id, role, content, **kw: sinks.added.append(role),
            update_conversation_metadata=lambda *a, **k: None,
            get_conversation=lambda conv_id: None,
        )
        response_cache = SimpleNamespace(
            enabled=True,
            make_cache_key=lambda *a: "exact-key",
            make_conversation_cache_key=lambda *a: "turn-key",
            get=lambda key: None,
            put=lambda **kw: sinks.puts.append(kw["explicit_key"]),
        )
        semantic_cache = SimpleNamespace(
            enabled=True,
            get=lambda *a, **k: None,
            get_with_fallback=lambda *a, **k: (None, 0.0, "miss"),
            put=lambda **kw: sinks.sem_puts.append(kw["query"]) or "semantic-key",
            store_embedding=lambda **kw: sinks.embeds.append(kw["cache_key"]),
        )
        return {
            "opti_oignon.conversation": _module("opti_oignon.conversation", conversation_manager=conv),
            "opti_oignon.memory.auto_capture": _module(
                "opti_oignon.memory.auto_capture",
                maybe_capture=lambda conv_id, messages: sinks.captured.append(conv_id),
            ),
            # The onion switched on: the librarian is offered the saved turn.
            "opti_oignon.memory.librarian": _module(
                "opti_oignon.memory.librarian",
                maybe_curate=lambda conv_id, messages: sinks.curated.append(conv_id),
                memory_block=lambda conv_id, question: "",
                onion_enabled=lambda: True,
            ),
            "opti_oignon.response_cache": _module("opti_oignon.response_cache", response_cache=response_cache),
            "opti_oignon.semantic_cache": _module("opti_oignon.semantic_cache", semantic_cache=semantic_cache),
            "opti_oignon.performance_monitor": _module(
                "opti_oignon.performance_monitor",
                performance_monitor=SimpleNamespace(
                    enabled=True, record_execution=lambda **kw: sinks.perf.append(kw["model"]),
                ),
            ),
        }

    def kept(self):
        return {
            "added": self.added, "captured": self.captured, "curated": self.curated,
            "puts": self.puts, "sem_puts": self.sem_puts, "embeds": self.embeds,
            "perf": self.perf,
        }


def _cancelling_stream(holder):
    def factory(kw):
        def stream():
            yield {"message": {"content": "Hi"}}
            holder["ex"].cancel()
            yield {"message": {"content": " more"}}
            yield {"message": {"content": " again"}}
        return stream()
    return factory


def _fc9_c1():
    """A drained cancelled call saves, captures, curates, caches and measures nothing."""
    for conversation_id in ("conv-9", None):
        for stopped in (False, True):
            sinks, holder = _Sinks(), {}
            world, close = _load_exec(sinks.seeded(), packages=("opti_oignon.memory",))
            try:
                ex = world.mod.Executor()
                holder["ex"] = ex
                if stopped:
                    world.client.factory = _cancelling_stream(holder)
                before = set(threading.enumerate())
                chunks, (refined, response) = _drive(
                    ex.execute("q", _routing(), refine=False, conversation_id=conversation_id)
                )
                assert _settled(before) == [], "a thread outlived the call"
                record = world.recorder.records[-1]
                kept = sinks.kept()
                if not stopped:
                    if conversation_id:
                        expected = {"added": ["user", "assistant"], "captured": ["conv-9"],
                                    "curated": ["conv-9"], "puts": ["turn-key"], "sem_puts": [],
                                    "embeds": [], "perf": ["m"]}
                    else:
                        expected = {"added": [], "captured": [], "curated": [], "puts": ["exact-key"],
                                    "sem_puts": ["q"], "embeds": ["exact-key"], "perf": ["m"]}
                    assert kept == expected, f"the witness kept {kept}"
                    assert record["cache_stored"] is True
                    continue
                assert _CANCEL in response, response
                assert all(v == [] for v in kept.values()), (
                    f"a cancelled call left an answer behind: {kept}"
                )
                assert record["outcome"] == "cancelled" and record["cache_stored"] is False, record
            finally:
                close()


def _fc9_c2i(world):
    """A call closed by its caller after a stop leaves exactly one cancelled record."""
    holder = {}
    ex = world.mod.Executor()
    holder["ex"] = ex
    world.client.factory = _cancelling_stream(holder)
    before = set(threading.enumerate())
    gen = ex.execute("q", _routing(), refine=False)
    for chunk in gen:
        if isinstance(chunk, str) and _CANCEL in chunk:
            break
    gen.close()
    assert _settled(before) == [], "a thread outlived the call"
    assert _outcomes(world) == ["cancelled"], f"a closed stopped call left {_outcomes(world)}"


def _fc9_c2ii(world):
    """A call whose run is already stopped, closed at once, leaves one cancelled record."""
    ex = world.mod.Executor()
    run = SimpleNamespace(stop=threading.Event(), results={})
    run.stop.set()
    gen = ex.execute("q", _routing(), refine=False, run=run)
    first = next(gen)
    gen.close()
    assert first == "[Cancelled]", first
    assert _outcomes(world) == ["cancelled"], _outcomes(world)


def _fc9_c2_witness(world):
    """Witness: a completed call leaves one completed record; a closed live call none."""
    ex = world.mod.Executor()
    before = set(threading.enumerate())
    _drive(ex.execute("q", _routing(), refine=False))
    assert _outcomes(world) == ["completed"]
    gen = ex.execute("q", _routing(), refine=False)
    assert next(gen) == "Hello"
    gen.close()
    assert _settled(before) == []
    assert _outcomes(world) == ["completed"], "a call closed without a stop left a record"


def _fc9_c3(world):
    """An error the backend raises after the stop is never reported."""
    stream = _ParkedStream(raise_after_gate=True)
    world.client.factory = stream
    ex = world.mod.Executor()
    before = set(threading.enumerate())
    canceller = threading.Thread(target=lambda: stream.parked.wait(_WAIT) and ex.cancel(),
                                 name="fc9-cancel")
    canceller.start()
    gen = ex.execute("q", _routing(), refine=False)
    seen = []
    try:
        deadline = time.monotonic() + _WAIT
        while time.monotonic() < deadline:
            chunk = next(gen)
            seen.append(chunk)
            if isinstance(chunk, str) and _CANCEL in chunk:
                break
        assert any(_CANCEL in c for c in seen if isinstance(c, str)), (
            f"the stopped call never yielded its marker: {seen}"
        )
    finally:
        stream.gate.set()
    assert stream.ended.wait(_WAIT)
    assert _settled(before) == [], "a thread outlived the call"
    chunks, (refined, response) = _drive(gen)
    assert "[ERR]" not in response, f"an error after the stop was reported: {response!r}"
    assert _outcomes(world) == ["cancelled"], _outcomes(world)


def test_fc9_a_stopped_call_leaves_no_answer_behind():
    _fc9_c1()
    for clause in (_fc9_c2i, _fc9_c2ii, _fc9_c2_witness, _fc9_c3):
        world, close = _load_exec()
        try:
            clause(world)
        finally:
            close()


# FC13 supersedes FC9. The onion's mirror now reads the conversation through
# the store's origin read, which the store FC9's witness seeds does not carry,
# so its witness was never curated. The same clauses over the store as it
# stands; the four that never reach the store are FC9's own.
class _OriginSinks(_Sinks):
    """The same sinks, over a store that carries the mirror's read."""

    def seeded(self):
        seeded = super().seeded()
        seeded["opti_oignon.conversation"].conversation_manager.get_mirror_messages = lambda conv_id: []
        return seeded


def _fc13_c1():
    """A drained cancelled call saves, captures, curates, caches and measures nothing."""
    for conversation_id in ("conv-9", None):
        for stopped in (False, True):
            sinks, holder = _OriginSinks(), {}
            world, close = _load_exec(sinks.seeded(), packages=("opti_oignon.memory",))
            try:
                ex = world.mod.Executor()
                holder["ex"] = ex
                if stopped:
                    world.client.factory = _cancelling_stream(holder)
                before = set(threading.enumerate())
                chunks, (refined, response) = _drive(
                    ex.execute("q", _routing(), refine=False, conversation_id=conversation_id)
                )
                assert _settled(before) == [], "a thread outlived the call"
                record = world.recorder.records[-1]
                kept = sinks.kept()
                if not stopped:
                    if conversation_id:
                        expected = {"added": ["user", "assistant"], "captured": ["conv-9"],
                                    "curated": ["conv-9"], "puts": ["turn-key"], "sem_puts": [],
                                    "embeds": [], "perf": ["m"]}
                    else:
                        expected = {"added": [], "captured": [], "curated": [], "puts": ["exact-key"],
                                    "sem_puts": ["q"], "embeds": ["exact-key"], "perf": ["m"]}
                    assert kept == expected, f"the witness kept {kept}"
                    assert record["cache_stored"] is True
                    continue
                assert _CANCEL in response, response
                assert all(v == [] for v in kept.values()), (
                    f"a cancelled call left an answer behind: {kept}"
                )
                assert record["outcome"] == "cancelled" and record["cache_stored"] is False, record
            finally:
                close()


def test_fc13_a_stopped_call_leaves_no_answer_behind_over_the_store_with_its_origin_read():
    _fc13_c1()
    for clause in (_fc9_c2i, _fc9_c2ii, _fc9_c2_witness, _fc9_c3):
        world, close = _load_exec()
        try:
            clause(world)
        finally:
            close()


def _fc10_c1():
    """Each run gets its own vision and verification results."""
    vision = SimpleNamespace(
        detect_needs_delegation=lambda **kw: False,
        process=lambda message, images, current_model, on_status=None: (
            message, None,
            {"delegated": True, "vision_model": "vision-" + message.split()[0]},
        ),
    )
    engine = SimpleNamespace(
        available=True,
        verify_response_code_blocks=lambda **kw: [SimpleNamespace(
            language="lang-" + kw["original_question"].split()[0], status="ok", iterations=1,
        )],
    )
    world, close = _load_exec({
        "opti_oignon.vision_pipeline": _module("opti_oignon.vision_pipeline", vision_pipeline=vision),
        "opti_oignon.verification": _module("opti_oignon.verification", verification_engine=engine),
    })
    try:
        alpha = _ParkedStream(chunks=("a1", "a2"))
        beta = _ParkedStream(chunks=("b1", "b2"))
        beta.gate.set()
        world.client.factory = lambda kw: (alpha if "alpha" in kw["messages"][-1]["content"] else beta)(kw)
        ex = world.mod.Executor()
        run_a = SimpleNamespace(stop=threading.Event(), results={})
        run_b = SimpleNamespace(stop=threading.Event(), results={})
        before = set(threading.enumerate())
        ta = threading.Thread(
            target=lambda: _drive(ex.execute("alpha q", _routing(), refine=False,
                                             images=["img"], run=run_a)),
            name="fc10-alpha",
        )
        ta.start()
        try:
            assert alpha.parked.wait(_WAIT), "call A never reached its stream"
            _drive(ex.execute("beta q", _routing(), refine=False, images=["img"], run=run_b))
        finally:
            alpha.gate.set()
            ta.join(_WAIT)
        assert _settled(before) == []
        assert run_a.results.get("vision_meta", {}).get("vision_model") == "vision-alpha", run_a.results
        assert run_b.results.get("vision_meta", {}).get("vision_model") == "vision-beta", run_b.results
        langs = {k: [v.language for v in r.results.get("verification_results") or []]
                 for k, r in (("a", run_a), ("b", run_b))}
        assert langs["a"] == ["lang-alpha"], f"call A's verification results: {langs}"
        assert langs["b"] == ["lang-beta"], f"call B's verification results: {langs}"
        assert ex.last_vision_meta["vision_model"] == "vision-beta"
        assert ex.last_verification_results[0].language == "lang-alpha"
    finally:
        close()


def _fc10_c2():
    """A call whose own budget is None never takes the instance's budget."""
    compressions = []
    history = [{"role": "user", "content": "x " * 300} for _ in range(6)]
    conv = SimpleNamespace(get_context_messages=lambda conv_id: list(history),
                           get_conversation=lambda conv_id: None)
    compressor = SimpleNamespace(
        enabled=True,
        get_config=lambda: {},
        compress=lambda **kw: compressions.append(kw["budget_tokens"]) or SimpleNamespace(
            compressed_count=1, summary="summary", recent_messages=[],
            strategy_used="contract", original_count=6, tokens_saved=1,
        ),
    )
    world, close = _load_exec({
        "opti_oignon.conversation": _module("opti_oignon.conversation", conversation_manager=conv),
        "opti_oignon.conversation_compressor": _module(
            "opti_oignon.conversation_compressor", CompressedContext=object,
            check_retrieval_trigger=lambda *a, **k: False, conversation_compressor=compressor,
        ),
    })
    try:
        ex = world.mod.Executor()
        ex._last_prompt_budget = SimpleNamespace(history_tokens=5)
        args = dict(system_prompt="sys", conversation_id="c", current_message="q", model="m")
        ex._build_conversation_messages(**args, prompt_budget=None)
        assert compressions == [], "a call without a budget compressed on another call's budget"
        ex._build_conversation_messages(**args, prompt_budget=SimpleNamespace(history_tokens=7))
        assert compressions == [7], "a call's own budget did not reach the compressor"
    finally:
        close()


def test_fc10_the_executors_results_belong_to_its_call():
    _fc10_c1()
    _fc10_c2()


# ---------------------------------------------------------------------------
# FC11 -- the stop reaches the tool stages
# ---------------------------------------------------------------------------

def _decision(content, calls=()):
    return SimpleNamespace(message=SimpleNamespace(
        content=content,
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name=n, arguments=dict(a))) for n, a in calls],
    ))


def _load_tools():
    """The real tool executor chain over a scripted client (sibling shape)."""
    class Scripted:
        def __init__(self):
            self.script, self.calls = [], []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if not self.script:
                raise AssertionError("the scripted client was called past its script")
            return self.script.pop(0)

    scripted = Scripted()
    cfg = _module("opti_oignon.config",
                  config=SimpleNamespace(get_user_preference=lambda key, default=None: default),
                  get_model=lambda *a, **k: "scripted-model")
    seeded = {
        "opti_oignon.security_mode": _module("opti_oignon.security_mode", is_bulbe=lambda: False),
        "opti_oignon.config": cfg,
        "opti_oignon.structured_output": _module("opti_oignon.structured_output",
                                                 StructuredOutputEngine=object, ToolCallRequest=object),
    }
    seed_registry(seeded, scripted)
    loaded, close = isolate(
        targets={
            "opti_oignon.tool_calling": source("tool_calling.py"),
            "opti_oignon.response_hygiene": source("response_hygiene.py"),
            "opti_oignon.tool_registry": source("tool_registry.py"),
            "opti_oignon.tool_executor": source("tool_executor.py"),
        },
        seeded=seeded, packages=("opti_oignon",),
    )
    te, tr = loaded["opti_oignon.tool_executor"], loaded["opti_oignon.tool_registry"]
    te.model_supports_native_tools = lambda model, capability_lookup=None: True
    ran = []
    registry = tr.ToolRegistry()
    handler = {"fn": lambda **kwargs: ran.append("echo") or "echoed"}
    registry.register(tr.ToolDefinition(
        name="echo", description="Echo the call back. Local test tool.",
        parameters={}, handler=lambda **kwargs: handler["fn"](**kwargs),
    ))
    executor = te.ToolExecutor(registry=registry)
    return SimpleNamespace(te=te, tr=tr, executor=executor, scripted=scripted, ran=ran,
                           handler=handler), close


def _full_script(front):
    final = _decision("Synthesized final.")
    if front == "stream":
        final = [{"message": {"content": "Synthesized final."}}]
    return [_decision("", calls=(("echo", {}),)), _decision("Leftover narration.", calls=()), final]


def _fc11_c1():
    """A stop after the first decision: no tool, no final generation, both fronts."""
    for front in ("execute", "stream"):
        for stopped in (False, True):
            world, close = _load_tools()
            try:
                world.scripted.script = _full_script(front)
                kwargs = {"should_stop": (lambda: len(world.scripted.calls) >= 1)} if stopped else {}
                if front == "execute":
                    result = world.executor.execute_with_tools("do it", model="scripted", **kwargs)
                    yielded = []
                else:
                    yielded, result = _drive(world.executor.stream_with_tools("do it", model="scripted", **kwargs))
                if not stopped:
                    assert world.ran == ["echo"] and len(world.scripted.calls) == 3, (
                        "the witness did not run the tool and the final generation"
                    )
                    continue
                assert world.ran == [], f"a tool ran after the stop ({front})"
                assert len(world.scripted.calls) == 1, (
                    f"{len(world.scripted.calls)} model calls after the stop ({front})"
                )
                assert result.response == "" and yielded == [], (front, result.response, yielded)
            finally:
                close()


def _fc11_c2():
    """A stop inside the first tool: no second decision, no final generation."""
    world, close = _load_tools()
    try:
        world.scripted.script = _full_script("execute")
        flag = threading.Event()
        world.handler["fn"] = lambda **kwargs: (world.ran.append("echo"), flag.set(), "echoed")[2]
        result = world.executor.execute_with_tools("do it", model="scripted", should_stop=flag.is_set)
        assert world.ran == ["echo"], world.ran
        assert len(world.scripted.calls) == 1, (
            f"{len(world.scripted.calls)} model calls after a stop inside the tool"
        )
        assert [c.tool_name for c in result.tool_calls] == ["echo"]
    finally:
        close()


def _fc11_c3():
    """Think+tools: a stopped turn runs no tool phase, or saves nothing after it."""
    for case in ("witness", "after_phase_1", "inside_phase_2"):
        added = []
        conv = _module("opti_oignon.conversation", conversation_manager=SimpleNamespace(
            get_messages=lambda conv_id: [],
            add_message=lambda **kw: added.append(kw["role"]),
        ))
        loaded, close = isolate(
            targets={"opti_oignon.agentic_executor": source("agentic_executor.py")},
            seeded={"opti_oignon.conversation": conv},
        )
        try:
            ae = loaded["opti_oignon.agentic_executor"]
            run = SimpleNamespace(stop=threading.Event(), results={})
            seen = {}
            call = SimpleNamespace(tool_name="echo", arguments={}, result="echoed", success=True)

            class Base:
                def execute(self, **kw):
                    yield "thinking it through"
                    if case == "after_phase_1":
                        run.stop.set()

            class Tools:
                def should_use_tools(self, *a, **k):
                    return True

                def execute_with_tools(self, **kw):
                    seen["kw"] = kw
                    if case == "inside_phase_2":
                        run.stop.set()
                    return SimpleNamespace(tool_calls=[call], response="tool output", verification_hints=0)

            ae.TOOL_EXECUTOR_AVAILABLE = True
            ae._select_pipeline = lambda **kw: ae.PIPELINE_THINK_TOOLS
            agent = ae.AgenticExecutor(executor=Base(), tool_executor=Tools(), default_model="m")
            agent._classify_message = lambda *a, **k: {"needs_tools": False}
            for _ in agent.execute("q", _routing(), conversation_id="conv-t", run=run):
                pass
            if case == "witness":
                assert "kw" in seen and added == ["user", "assistant"], (seen.keys(), added)
            elif case == "after_phase_1":
                assert "kw" not in seen, "the tool phase ran after a stop in phase 1"
                assert added == [], f"a stopped turn was saved: {added}"
            else:
                should_stop = seen["kw"].get("should_stop")
                assert should_stop is not None and should_stop() is True, (
                    "phase 2 was not handed the turn's stop"
                )
                assert agent.get_tool_history("conv-t") == [call], "the tool that ran was not recorded"
                assert added == [], f"a stopped turn was saved: {added}"
        finally:
            close()


def _fc11_c4():
    """The approval wait ends on the turn's stop, and the request is withdrawn."""
    loaded, rc, schemas, close = _load_routes_only(
        extra_targets={"opti_oignon.tool_call_approval": source("tool_call_approval.py")},
    )
    try:
        tca = loaded["opti_oignon.tool_call_approval"]
        before = set(threading.enumerate())
        for stop_it in (True, False):
            done, stop, out = threading.Event(), threading.Event(), {}
            waiter = threading.Thread(
                target=lambda: out.setdefault("at", (rc._await_approval(done, stop, 30), time.monotonic())),
                name="fc11-approval",
            )
            waiter.start()
            began = time.monotonic()
            (stop if stop_it else done).set()
            waiter.join(1.5)
            assert "at" in out and out["at"][1] - began <= 1.0, (
                f"the approval wait did not end on its {'stop' if stop_it else 'decision'}"
            )
        assert _settled(before) == []

        manager = tca.ToolCallApprovalManager()
        manager._reaper_active = True
        aid, event = manager.submit(conversation_id="conv-b", tool_name="echo", arguments={})
        assert manager.withdraw(aid, "turn_stopped") is True
        assert event.is_set(), "the withdrawn request's waiter was not released"
        assert manager.get_status(aid) == tca.ApprovalStatus.DENIED
        assert manager.audit_log(1)[0]["resolved_by"] == "turn_stopped"
        assert manager.approve(aid) is False, "a withdrawn request could still be approved"
        live, _ = manager.submit(conversation_id="conv-b", tool_name="echo", arguments={})
        assert manager.approve(live) is True, "the witness request could not be approved"

        tree = ast.parse(source("api", "routes_chat.py").read_text(encoding="utf-8"))
        hooks = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_approval_hook"]
        assert len(hooks) == 1, "no single approval hook"
        calls = [n for n in ast.walk(hooks[0]) if isinstance(n, ast.Call)]
        waits = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "_await_approval"]
        assert waits and any(
            isinstance(a, ast.Attribute) and a.attr == "stop" and isinstance(a.value, ast.Name)
            and a.value.id == "turn" for a in waits[0].args
        ), "the hook does not wait on the turn's stop"
        assert any("withdraw" in ast.dump(c.func) for c in calls), "the hook never withdraws"
        assert not any(
            isinstance(c.func, ast.Attribute) and c.func.attr == "wait"
            and isinstance(c.func.value, ast.Name) and c.func.value.id == "event" for c in calls
        ), "the hook still waits on the request's event alone"
    finally:
        close()


def _fc11_c4_hook():
    """The route's approval hook itself: a stopped turn's request is withdrawn,
    and a decision that lands after the stop never runs the tool."""
    loaded, rc, schemas, close = _load_routes_only(
        extra_targets={"opti_oignon.tool_call_approval": source("tool_call_approval.py")},
    )
    try:
        tca = loaded["opti_oignon.tool_call_approval"]
        outcomes = {}
        for case in ("approved", "stopped_while_pending", "approved_after_the_stop"):
            manager = tca.ToolCallApprovalManager()
            manager._reaper_active = True
            turn = rc.ChatTurn("conv-h")
            emitted = []

            def emit(event, manager=manager, turn=turn, case=case, emitted=emitted):
                # Runs on the hook's own thread, before it waits: the person
                # and the Stop act here, so no interleaving is left to chance.
                emitted.append(event)
                if event[0] != "tool_call_pending":
                    return
                aid = event[1]["approval_id"]
                if case in ("approved", "approved_after_the_stop"):
                    manager.approve(aid)
                if case in ("stopped_while_pending", "approved_after_the_stop"):
                    turn.stop.set()

            hook = rc._make_approval_hook(manager, turn, "conv-h", emit, 30)
            began = time.monotonic()
            allowed = hook("echo", {})
            outcomes[case] = allowed
            assert time.monotonic() - began <= 1.0, f"the hook waited out its deadline ({case})"
            resolved = [e[1] for e in emitted if e[0] == "tool_call_resolved"]
            assert resolved and resolved[-1]["approved"] is allowed, (case, emitted)
            if case == "stopped_while_pending":
                assert manager.pending() == [], (
                    "a stopped turn's request is still waiting for a person"
                )
                record = manager.audit_log(1)
                assert record and record[0]["resolved_by"] == "turn_stopped", (
                    f"a stopped turn's request was not withdrawn on its behalf: {record}"
                )
        assert outcomes["approved"] is True, "the witness approval did not run the tool"
        assert outcomes["stopped_while_pending"] is False, "a stopped turn's tool ran"
        assert outcomes["approved_after_the_stop"] is False, (
            "an approval that landed after the stop ran the tool"
        )
    finally:
        close()


def _fc11_c5():
    """The pipeline runner checks the turn's stop between steps and forwards its run."""
    pl, close = _load_leaf("pipelines", "pipelines.py")
    try:
        pl._resolve_emergency_stop = lambda: None
        pl._resolve_resource_governor = lambda: None
        run = SimpleNamespace(stop=threading.Event(), results={})

        class Recording:
            def __init__(self):
                self.calls = []

            def execute(self, **kw):
                self.calls.append(kw)
                if len(self.calls) == 1:
                    run.stop.set()
                yield "step output"

        router = SimpleNamespace(enabled=False, override_routing=lambda routing, step_type: routing)
        pipe = pl.ExecutionPipeline(id="p", name="P", steps=[
            pl.ExecutionStep("direct", label="one"), pl.ExecutionStep("direct", label="two"),
        ])
        rec = Recording()
        list(pl.PipelineRunner(agentic_executor=rec, smart_router=router).execute(
            pipe, "hello", SimpleNamespace(model="m"), run=run,
        ))
        assert len(rec.calls) == 1, f"the runner started step 2 after the stop: {len(rec.calls)} steps"
        assert rec.calls[0].get("run") is run, "the step was not handed the turn's run"
        plain = Recording()
        run.stop.clear()
        list(pl.PipelineRunner(agentic_executor=plain, smart_router=router).execute(
            pipe, "hello", SimpleNamespace(model="m"),
        ))
        assert plain.calls and all("run" not in kw for kw in plain.calls), (
            "a runner call without a run passed a run keyword"
        )
    finally:
        close()


def _fc11_c6():
    """Code verification never starts on a stopped turn."""
    ae, close = _load_leaf("agentic_executor", "agentic_executor.py")
    try:
        calls = []
        engine = SimpleNamespace(available=True,
                                 verify_response_code_blocks=lambda **kw: calls.append(1) or [])
        ae.VERIFICATION_AVAILABLE = True
        agent = ae.AgenticExecutor(executor=SimpleNamespace(), verification_engine=engine, default_model="m")
        stopped = SimpleNamespace(stop=threading.Event(), results={})
        stopped.stop.set()
        agent._run_code_verification("```python\nprint(1)\n```", "q", "m", turn=ae._Turn(stopped))
        assert calls == [], "verification started on a stopped turn"
        live = SimpleNamespace(stop=threading.Event(), results={})
        agent._run_code_verification("```python\nprint(1)\n```", "q", "m", turn=ae._Turn(live))
        assert calls == [1], "the witness never reached the verification engine"
    finally:
        close()


_NARRATED = "Here it is:\n```python\nprint('hi')\n```\nRun it."


def _fc11_c7():
    """A stop during the salvage's candidate: no salvaged tool and no final generation."""
    for front in ("execute", "stream"):
        for stopped in (False, True):
            world, close = _load_tools()
            try:
                salvaged = []
                for name in ("write_file", "execute_code"):
                    world.executor.registry.register(world.tr.ToolDefinition(
                        name=name, description=f"{name}, a local test tool.", parameters={},
                        handler=(lambda tool: lambda **kwargs: salvaged.append(tool) or "ok")(name),
                    ))
                final = _decision("Summary of the run.")
                if front == "stream":
                    final = [{"message": {"content": "Summary of the run."}}]
                # The decision calls nothing and carries no answer, so the
                # salvage generates a candidate, which narrates code.
                world.scripted.script = [_decision(""), _decision(_NARRATED), final]
                kwargs = {"should_stop": (lambda: len(world.scripted.calls) >= 2)} if stopped else {}
                message = "write and run a hello script"
                if front == "execute":
                    result = world.executor.execute_with_tools(message, model="scripted", **kwargs)
                    yielded = []
                else:
                    yielded, result = _drive(
                        world.executor.stream_with_tools(message, model="scripted", **kwargs)
                    )
                if not stopped:
                    assert salvaged == ["write_file", "execute_code"], (
                        f"the witness salvaged nothing ({front}): {salvaged}"
                    )
                    assert len(world.scripted.calls) == 3, "the witness made no final generation"
                    continue
                assert salvaged == [], f"a salvaged tool ran after the stop ({front}): {salvaged}"
                assert len(world.scripted.calls) == 2, (
                    f"{len(world.scripted.calls)} model calls after a stop in the salvage ({front})"
                )
                assert result.response == "" and yielded == [], (front, result.response, yielded)
            finally:
                close()


def _fc11_c8():
    """A stop during the native decision: that decision makes no second model call."""
    for stopped in (False, True):
        world, close = _load_tools()
        try:
            asked = []
            world.te.STRUCTURED_OUTPUT_AVAILABLE = True
            world.executor.structured_engine = SimpleNamespace(
                generate_structured=lambda **kw: asked.append(kw["model"]) or SimpleNamespace(
                    success=True, data=SimpleNamespace(tool_name="none", arguments={}),
                ),
            )

            def failing_chat(**kwargs):
                world.scripted.calls.append(kwargs)
                raise RuntimeError("the native decision failed")

            world.scripted.chat = failing_chat
            kwargs = {"should_stop": (lambda: len(world.scripted.calls) >= 1)} if stopped else {}
            result = world.executor.execute_with_tools("do it", model="scripted", **kwargs)
            if not stopped:
                assert asked == ["scripted"], f"the witness never asked the fallback: {asked}"
                continue
            assert asked == [], "a stopped decision asked its fallback model"
            assert len(world.scripted.calls) == 1 and result.response == "", (
                len(world.scripted.calls), result.response,
            )
        finally:
            close()


class _StreamingToolStage:
    """A tools stage: its tool runs, then its answer streams and parks half-way."""

    def __init__(self):
        self.gate = threading.Event()
        self.parked = threading.Event()
        self.call = SimpleNamespace(
            tool_name="write_file", arguments={"path": "x.txt"}, result="written",
            success=True, execution_time=0, reasoning="",
        )

    def should_use_tools(self, *a, **k):
        return True

    def stream_with_tools(self, **kw):
        kw["on_tool_call"](self.call)
        yield "the answer begins"
        self.parked.set()
        self.gate.wait(_WAIT)
        yield " and goes on"
        return SimpleNamespace(tool_calls=[self.call], response="the answer begins and goes on",
                               verification_hints=0)


def _fc11_c9():
    """A turn stopped while its answer streams keeps the record of the tools that ran."""
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = None
    loaded, rc, schemas, close = _load_routes_only(
        extra_targets={"opti_oignon.agentic_executor": source("agentic_executor.py")},
        seeded={"opti_oignon.conversation": conv},
    )
    try:
        ae = loaded["opti_oignon.agentic_executor"]
        ae.TOOL_EXECUTOR_AVAILABLE = True
        ae._select_pipeline = lambda **kw: ae.PIPELINE_TOOLS
        rc.AGENTIC_EXECUTOR_AVAILABLE = True
        rc.executor = SimpleNamespace(reset=lambda: None, cancel=lambda: None,
                                      last_vision_meta={}, last_verification_results=[])
        world = SimpleNamespace(rc=rc, schemas=schemas)
        for stop in (False, True):
            stage = _StreamingToolStage()
            agent = ae.AgenticExecutor(executor=SimpleNamespace(), tool_executor=stage,
                                       default_model="m")
            agent._classify_message = lambda *a, **k: {"needs_tools": False}
            rc._agentic_executor = agent
            before = set(threading.enumerate())
            ws = _Socket()

            async def scenario():
                task = await _open(world, ws, "conv-k", "write x")
                assert await _until(stage.parked.is_set), "the answer never started"
                if stop:
                    await _stop(world, "conv-k")
                stage.gate.set()
                await asyncio.wait_for(task, _WAIT)

            try:
                asyncio.run(scenario())
            finally:
                stage.gate.set()
            assert _settled(before) == [], "a thread outlived the clause"
            done = _done(ws)
            assert done["cancelled"] is stop, done
            assert agent.get_tool_history("conv-k") == [stage.call], (
                f"the tool that ran is missing from the conversation's tool history (stop={stop})"
            )
            assert done["tool_calls_count"] == 1, (
                f"the done of a turn whose tool ran counts {done['tool_calls_count']} (stop={stop})"
            )
    finally:
        close()


def test_fc11_the_stop_reaches_the_tool_stages():
    for clause in (_fc11_c1, _fc11_c2, _fc11_c3, _fc11_c4, _fc11_c4_hook, _fc11_c5, _fc11_c6,
                   _fc11_c7, _fc11_c8, _fc11_c9):
        clause()
