#!/usr/bin/env python3
"""The chat relay and the step frames of a reply.

The recorder of a turn puts its frames in the relay's queue; the relay
sends them, keeps them through backpressure, closes every step the run
left open, and hands the last state of every step to ``done``. These
contracts pin the relay's part:

* PS6 -- the relay closes every open step: on a Stop, running to
  cancelled and pending to not_run (with ``Stop all engaged`` under the
  emergency stop); on an error, running to failed and pending to not_run,
  before the error frame; no ``done`` leaves while a step is open.
* PS7 -- ``pipeline_step`` is critical: a backlog of 150 events trimmed to
  its cap drops non-critical events and never a step frame.
* PS8 -- the relay sends ``pipeline_step`` frames and no longer sends
  "Step k done"; a tuple of an unknown kind never reaches the reply text.
* PS14 -- ``done`` carries ``steps``, the last state of every step emitted
  in the reply, and no ``steps`` when none was.
* PS15 -- only a status that starts an image analysis is tagged
  ``analyzing``; one that merely names vision is not.
* PS23 -- the backlog trim never loses an event appended while it runs.
* PS10 -- a step the live hook already queued reaches the socket once: the
  replay of the reasoning steps and of the consensus models after the run
  is dropped; a consensus model that failed is reported live too.

Every stream loads the real route module and the real schemas in the
shared isolation window, with the real recorder module given to it; the
runner is scripted, or the real runner behind a scripted agentic
executor. The socket is a recording stand-in.
"""

import asyncio
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

import opti_oignon.pipelines as P  # noqa: E402
from opti_oignon import pipeline_step as ps  # noqa: E402

BUDGET_S = {
    "test_ps6_the_relay_closes_every_open_step": 2.0,
    "test_ps7_a_step_frame_survives_backpressure": 2.0,
    "test_ps8_the_relay_sends_step_frames_and_no_step_done_status": 2.0,
    "test_ps14_done_carries_the_last_state_of_every_step": 2.0,
    "test_ps15_only_an_image_analysis_is_tagged_analyzing": 2.0,
    "test_ps23_the_backlog_trim_loses_no_event_appended_while_it_runs": 2.0,
    "test_ps10_a_live_step_reaches_the_socket_once": 2.0,
}

_TERMINAL = ("done", "failed", "skipped", "cancelled", "not_run")


# ---------------------------------------------------------------------------
# The relay's world
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


class _Socket:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)


class _Estop:
    """The emergency stop module as the relay reads it."""

    def __init__(self):
        self.engaged = False

    def is_stopped(self):
        return self.engaged

    def refusal_payload(self):
        return {"message": "Stop all engaged"}


class _Runner:
    """A scripted pipeline runner: ``script(run, on_status)`` is the stream."""

    def __init__(self, script):
        self.script = script

    def execute(self, *, pipeline, message, routing, run=None, on_status=None, **kwargs):
        return self.script(run, on_status)


_PIPE = SimpleNamespace(id="research", name="Research", steps=[])


def _load(runner, estop=None):
    seeded = {"opti_oignon.api.deps": _deps_stub(), "opti_oignon.pipeline_step": ps}
    if estop is not None:
        seeded["opti_oignon.emergency_stop"] = estop
    loaded, close = isolate(
        targets={
            "opti_oignon.api.schemas": source("api", "schemas.py"),
            "opti_oignon.api.routes_chat": source("api", "routes_chat.py"),
        },
        seeded=seeded,
        packages=("opti_oignon.api",),
    )
    rc = loaded["opti_oignon.api.routes_chat"]
    rc.EXECUTOR_AVAILABLE = True
    rc.executor = SimpleNamespace()
    rc._resolve_model_and_route = lambda message, request: (_routing(), None)
    rc.AGENTIC_EXECUTOR_AVAILABLE = True
    rc._agentic_executor = SimpleNamespace(available=True)
    rc.PIPELINE_DIRECT = "direct"
    rc.EXEC_PIPELINES_AVAILABLE = True
    rc.get_pipeline_store = lambda: SimpleNamespace(get=lambda pid: _PIPE)
    rc.get_pipeline_runner = lambda: runner
    return SimpleNamespace(rc=rc, schemas=loaded["opti_oignon.api.schemas"], close=close)


def _stream(runner, estop=None):
    """One chat turn through the real route; returns the frames sent."""
    world = _load(runner, estop)
    ws = _Socket()
    try:
        request = world.schemas.ChatRequest(
            conversation_id="c1", message="What does an onion need?", exec_pipeline="research")
        asyncio.run(world.rc._stream_response(ws, "c1", request.message, request))
    finally:
        world.close()
    return ws.sent


def _steps(sent):
    return [d["metadata"] for d in sent if d.get("type") == "pipeline_step"]


def _last_states(sent):
    last = {}
    for md in _steps(sent):
        last[(md["run"], md["index"])] = md
    return last


def _position(sent, predicate):
    for i, frame in enumerate(sent):
        if predicate(frame):
            return i
    return None


def _open_run(run, total=3):
    """A pipeline run with step 0 done, step 1 running, the rest pending."""
    r = run.steps.start_run("exec_pipeline", "Research", total=total, pipeline_id="research")
    for index in range(total):
        run.steps.pending(r, index, f"Step {index + 1}")
    run.steps.running(r, 0)
    run.steps.end(r, 0, "done", ran_as="direct")
    run.steps.running(r, 1)
    return r


# ---------------------------------------------------------------------------
# PS6 -- the relay closes every open step
# ---------------------------------------------------------------------------

def _ps6_c1():
    """On a Stop: running -> cancelled, pending -> not_run, before the
    cancellation; under the emergency stop with its reason."""
    for emergency in (False, True):
        estop = _Estop()

        def script(run, on_status):
            _open_run(run)
            yield "Onions like "
            if emergency:
                estop.engaged = True
            run.stop.set()
            yield "sun."
        sent = _stream(_Runner(script), estop)
        last = _last_states(sent)
        reason = "Stop all engaged" if emergency else None
        assert [(md["state"], md["reason"]) for md in last.values()] == [
            ("done", None), ("cancelled", reason), ("not_run", reason)]
        closed = _position(sent, lambda f: f.get("type") == "pipeline_step"
                           and f["metadata"]["state"] == "not_run")
        cancelled = _position(sent, lambda f: f.get("type") == "token"
                              and "[Generation cancelled]" in f.get("content", ""))
        assert closed is not None and cancelled is not None
        assert closed < cancelled


def _ps6_c2():
    """On an error: running -> failed, pending -> not_run, before the
    error frame."""
    def script(run, on_status):
        _open_run(run)
        yield "Onions like "
        raise RuntimeError("the backend went away")
    sent = _stream(_Runner(script))
    last = _last_states(sent)
    assert [(md["state"], md["reason"]) for md in last.values()] == [
        ("done", None), ("failed", "the backend went away"), ("not_run", "the backend went away")]
    error = _position(sent, lambda f: f.get("type") == "error")
    assert error is not None
    assert max(i for i, f in enumerate(sent) if f.get("type") == "pipeline_step") < error


def _ps6_c3():
    """No ``done`` while a step is open: a step the run never ended is
    closed as on an error, with its reason, before ``done``."""
    def script(run, on_status):
        _open_run(run)
        yield "Onions like sun."
    sent = _stream(_Runner(script))
    done = _position(sent, lambda f: f.get("type") == "done")
    assert done is not None
    before = _last_states(sent[:done])
    assert [md["state"] for md in before.values()] == ["done", "failed", "not_run"]
    assert {md["reason"] for md in list(before.values())[1:]} == {"No end was reported"}


def test_ps6_the_relay_closes_every_open_step():
    _ps6_c1()
    _ps6_c2()
    _ps6_c3()


# ---------------------------------------------------------------------------
# PS7 and PS23 -- the backlog trim
# ---------------------------------------------------------------------------

def _frame(index):
    return ("pipeline_step", {"seq": index + 1, "index": index})


def test_ps7_a_step_frame_survives_backpressure():
    world = _load(_Runner(lambda run, on_status: iter(())))
    try:
        trim, cap = world.rc._trim_backlog, world.rc._BP_MAX_SIZE
    finally:
        world.close()
    kinds = ("chunk", "status", "reasoning_step", "thinking", "tool_call")
    chunks, frames = [], []
    for i in range(150):
        if i % 10 == 0:
            frames.append(_frame(i))
            chunks.append(frames[-1])
        else:
            chunks.append((kinds[i % len(kinds)], f"event {i}"))
    dropped = trim(chunks, 0)
    assert cap <= 100
    assert dropped >= 1
    assert [c for c in chunks if c[0] == "pipeline_step"] == frames


class _Growing(list):
    """A queue the generation thread appends to while the trim reads it:
    the first read that touches the pending window appends ``late``."""

    def __init__(self, items, sent_index, late):
        super().__init__(items)
        self.sent_index = sent_index
        self.late = list(late)
        self.fired = False

    def __getitem__(self, key):
        touches = (key >= self.sent_index if isinstance(key, int)
                   else key.stop is None or key.stop > self.sent_index)
        if touches and not self.fired:
            self.fired = True
            for item in self.late:
                self.append(item)
        return super().__getitem__(key)


def test_ps23_the_backlog_trim_loses_no_event_appended_while_it_runs():
    world = _load(_Runner(lambda run, on_status: iter(())))
    try:
        trim = world.rc._trim_backlog
    finally:
        world.close()
    # Kinds that were critical before step frames were, so the clause holds
    # on the race alone.
    sent_index = 20
    waiting = ("tool_call_pending", {"id": "t1"})
    items = [("chunk", f"sent {i}") for i in range(sent_index)]
    items += [("chunk", f"pending {i}") for i in range(130)]
    items.insert(sent_index + 5, waiting)
    late = [("error", "the backend went away"), ("chunk", "late text")]
    chunks = _Growing(items, sent_index, late)
    dropped = trim(chunks, sent_index)
    assert chunks.fired
    assert dropped >= 1
    kept = list(list.__iter__(chunks))
    assert [c for c in kept if c[0] in ("tool_call_pending", "error")] == [waiting, late[0]]
    assert late[1] in kept
    assert kept[:sent_index] == items[:sent_index]


# ---------------------------------------------------------------------------
# PS8 -- step frames, and no "Step k done"
# ---------------------------------------------------------------------------

class _Agentic:
    def execute(self, *, message, routing, run=None, **kwargs):
        run.results["pipeline"] = "direct"
        yield "An answer."


class _NoRouter:
    enabled = False

    def override_routing(self, routing, step_type):
        return routing


def _ps8_c1(monkeypatch):
    """The real runner behind the relay: step frames are sent, the
    runner's "Step i/N" statuses still flow, and no "Step k done"."""
    monkeypatch.setattr(P, "_resolve_emergency_stop", lambda: None)
    monkeypatch.setattr(P, "_resolve_resource_governor", lambda: None)
    pipe = P.ExecutionPipeline(id="research", name="Research", steps=[
        P.ExecutionStep(step_type="direct", label="Gather"),
        P.ExecutionStep(step_type="think", label="Write")])
    monkeypatch.setattr(sys.modules[__name__], "_PIPE", pipe)
    runner = P.PipelineRunner(agentic_executor=_Agentic(), smart_router=_NoRouter())
    sent = _stream(runner)
    statuses = [d["metadata"]["message"] for d in sent if d.get("type") == "status"]
    assert len(_steps(sent)) >= 6
    assert [md["state"] for md in _last_states(sent).values()] == ["done", "done"]
    assert len([s for s in statuses if s.startswith("Step 1/2")]) >= 1
    assert [s for s in statuses if s.endswith(" done") and s.startswith("Step ")] == []


def _ps8_c2():
    """A tuple of an unknown kind (the cascade and speculative results)
    never reaches the reply text, and the reply still ends ``done``."""
    def script(run, on_status):
        yield "Onions like "
        yield ("cascade_done", SimpleNamespace(final="ignored"))
        yield ("speculative_done", {"accepted": 3})
        yield "sun."
    sent = _stream(_Runner(script))
    assert [f for f in sent if f.get("type") == "error"] == []
    done = [f for f in sent if f.get("type") == "done"]
    assert len(done) == 1
    assert done[0]["content"] == "Onions like sun."


def test_ps8_the_relay_sends_step_frames_and_no_step_done_status(monkeypatch):
    _ps8_c1(monkeypatch)
    _ps8_c2()


# ---------------------------------------------------------------------------
# PS14 -- done carries the steps
# ---------------------------------------------------------------------------

def test_ps14_done_carries_the_last_state_of_every_step():
    def script(run, on_status):
        r = run.steps.start_run("exec_pipeline", "Research", total=3, pipeline_id="research")
        for index in range(3):
            run.steps.pending(r, index, f"Step {index + 1}")
        run.steps.running(r, 0)
        run.steps.end(r, 0, "done", ran_as="direct")
        run.steps.end(r, 1, "skipped")
        run.steps.running(r, 2)
        run.steps.end(r, 2, "failed", reason="Connection refused")
        yield "Onions like sun."
    sent = _stream(_Runner(script))
    done = [f for f in sent if f.get("type") == "done"][-1]["metadata"]
    assert [md["state"] for md in done["steps"]] == ["done", "skipped", "failed"]
    assert done["steps"] == list(_last_states(sent).values())

    # Witness: a reply that emitted no step carries no ``steps``.
    sent = _stream(_Runner(lambda run, on_status: iter(["Onions like sun."])))
    done = [f for f in sent if f.get("type") == "done"][-1]["metadata"]
    assert _steps(sent) == []
    assert "steps" not in done


# ---------------------------------------------------------------------------
# PS15 -- the vision tag
# ---------------------------------------------------------------------------

def test_ps15_only_an_image_analysis_is_tagged_analyzing():
    messages = (
        "No vision-capable model found",
        "[!] Resource admission refused for the vision model",
        "Analyzing image with llava:7b",
    )

    def script(run, on_status):
        for message in messages:
            on_status(message)
        yield "An answer."
    sent = _stream(_Runner(script))
    statuses = [d["metadata"]["message"] for d in sent if d.get("type") == "status"]
    tagged = [d["metadata"]["message"] for d in sent if d.get("type") == "vision_delegation"
              and d["metadata"].get("status") == "analyzing"]
    assert statuses == list(messages)
    assert tagged == ["Analyzing image with llava:7b"]


class _Backend:
    """A registry backend for the reasoning engine: ``answer(n)`` is the
    reply to the n-th call, counted from 1."""

    def __init__(self, answer):
        self.calls = 0
        self.answer = answer

    def generate(self, model=None, messages=None, options=None, **kw):
        self.calls += 1
        return SimpleNamespace(content=self.answer(self.calls))


def _agentic_stream(engine, wire, **request):
    """One agentic turn through the real route, the real agentic executor
    and the real ``engine`` module; ``wire(module, ae)`` puts a scripted
    backend behind the engine and returns the agentic executor. ``request``
    adds fields to the chat request. Returns the frames sent."""
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = None
    loaded, close = isolate(
        targets={
            f"opti_oignon.{engine}": source(f"{engine}.py"),
            "opti_oignon.agentic_executor": source("agentic_executor.py"),
            "opti_oignon.api.schemas": source("api", "schemas.py"),
            "opti_oignon.api.routes_chat": source("api", "routes_chat.py"),
        },
        seeded={"opti_oignon.api.deps": _deps_stub(), "opti_oignon.pipeline_step": ps,
                "opti_oignon.conversation": conv},
        packages=("opti_oignon.api",),
    )
    ws = _Socket()
    try:
        module, ae = loaded[f"opti_oignon.{engine}"], loaded["opti_oignon.agentic_executor"]
        rc, schemas = loaded["opti_oignon.api.routes_chat"], loaded["opti_oignon.api.schemas"]
        agent = wire(module, ae)
        agent._classify_message = lambda *a, **k: {"needs_tools": False}
        rc.EXECUTOR_AVAILABLE = True
        rc.executor = SimpleNamespace(reset=lambda: None, cancel=lambda: None,
                                      last_vision_meta={}, last_verification_results=[])
        rc._resolve_model_and_route = lambda message, request: (_routing(), None)
        rc.AGENTIC_EXECUTOR_AVAILABLE = True
        rc._agentic_executor = agent
        chat = schemas.ChatRequest(conversation_id="c1", message="Why loose soil?", **request)
        asyncio.run(rc._stream_response(ws, "c1", chat.message, chat))
    finally:
        close()
    return ws.sent


def _reasoning_stream(answer):
    """One reasoning turn, the real engine behind a scripted backend."""
    def wire(rs, ae):
        backend = _Backend(answer)
        rs._resolve_backend = lambda model: backend
        engine = rs.ReasoningEngine(config=rs.ReasoningConfig(), default_model="m")
        ae.REASONING_AVAILABLE = True
        ae._select_pipeline = lambda **kw: ae.PIPELINE_REASONING
        return ae.AgenticExecutor(executor=SimpleNamespace(), reasoning_engine=engine,
                                  default_model="m")
    return _agentic_stream("reasoning", wire)


class _Models:
    """A registry backend for the consensus engine: ``answer(model)`` is the
    model's reply; an exception it returns is raised by the call."""

    def __init__(self, answer):
        self.answer = answer

    def generate(self, model=None, messages=None, options=None, **kw):
        reply = self.answer(model)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply)


def _consensus_stream(answer, models):
    """One consensus turn asked for ``models``, the real engine behind a
    scripted backend."""
    def wire(cs, ae):
        backend = _Models(answer)
        cs._resolve_backend = lambda model: backend
        engine = cs.ConsensusEngine(config=cs.ConsensusConfig(), default_model="m")
        ae.CONSENSUS_AVAILABLE = True
        return ae.AgenticExecutor(executor=SimpleNamespace(), consensus_engine=engine,
                                  default_model="m")
    return _agentic_stream("consensus", wire, consensus=True, consensus_models=models)


def _ps10_c1():
    """Each reasoning step reaches the socket once, from the live hook."""
    plan = '[{"title": "Soil", "question": "q1"}, {"title": "Water", "question": "q2"}]'
    sent = _reasoning_stream(lambda n: plan if n == 1 else f"answer {n}")
    numbers = [d["metadata"]["step_number"] for d in sent if d.get("type") == "reasoning_step"]
    assert numbers == [1, 2], f"the reasoning steps reached the socket as {numbers}"


def _ps10_c2():
    """Each consensus model reaches the socket once, from the live hook,
    the model that failed too."""
    sent = _consensus_stream(lambda model: RuntimeError("model gone") if model == "b"
                             else f"{model}: loose soil", ["a", "b", "c"])
    models = [d["metadata"]["model"] for d in sent if d.get("type") == "consensus_model_done"]
    assert sorted(models) == ["a", "b", "c"], f"the consensus models reached the socket as {models}"
    failed = [d["metadata"]["model"] for d in sent if d.get("type") == "consensus_model_done"
              and d["metadata"]["success"] is False]
    assert failed == ["b"], f"the failed models reached the socket as {failed}"


def test_ps10_a_live_step_reaches_the_socket_once():
    _ps10_c1()
    _ps10_c2()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
