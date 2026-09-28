#!/usr/bin/env python3
"""The step event of a reply and the recorder that emits it.

A reply that runs an execution pipeline, a reasoning strategy, a consensus
or a self-correction is made of steps, and the interface draws each step
from ``pipeline_step`` frames. The frame is built in one place and sent by
one recorder per reply, so every frame that leaves is well formed and every
step follows the same transition rule. These contracts pin both:

* PS1 -- the builder is closed and total: every field of the schema is
  present and typed, the state is one of the seven, progress only on a
  running step, a reason only on a failed, not-run or cancelled step, the
  label and the reason cut to their caps, and anything else raises. The
  kinds and units that have no emitter yet raise too.
* PS25 -- the recorder is the transition rule: an illegal transition, a
  second terminal state and any event after the recorder is closed raise
  and send nothing; ``seq`` strictly increases across every run of the
  reply.

The execution pipeline runner emits through the turn's recorder:

* PS2 -- a run sends one ``pending`` per step, in index order, before any
  ``running``; every step reaches exactly one terminal state; ``seq``
  strictly increases; a runner that cannot start sends nothing.
* PS3 -- a step that raises ends ``failed`` with the exception's text, is
  never ``done``, and the next step still runs.
* PS4 -- a step whose condition is false ends ``skipped`` and never runs.
* PS5 -- the emergency stop between steps ends the rest ``not_run``; the
  runner's own admission refused at step k never sends ``running`` for k,
  and k and the later steps end ``not_run``; a stop engaged during a step
  ends it ``cancelled``, never ``done``.
* PS13 -- a step's terminal frame carries ``ran_as``, the pipeline the
  turn reports it ran, never the shared instance's last pipeline.
* PS17 -- a step whose call reports one of the server's own fixed errors
  ends ``failed`` with it; a model reply equal to that text is ``done``;
  the fixed errors are a closed list of sites that say so structurally.
* PS21 -- a safety mechanism never ends a step ``failed``: the executor's
  admission refusal, read from a structured signal, ends it ``cancelled``
  with the governor's message.
* PS24 -- after a refusal inside step k, the later steps end ``not_run``
  with the same message, none starts, and no prompt holds the refusal.

The reasoning strategies and the self-correction emit through it too:

* PS9 -- decompose runs its first step with no total, then announces one
  step per planned sub-question and the combine step, all with the total
  n + 2; tree-of-thought has three steps and self-consistency one sample
  per run, both with their total from the first frame; a sub-step whose
  call raised ends ``failed``.
* PS16 -- a self-correction is two steps with no ``progress``: its passes
  stop early, so the pass number is said in words, by ``correction_step``.
* PS11 -- a consensus is two steps: the query carries ``progress`` in
  models, its total the resolved model count and its ``done`` the models
  that finished, answered or failed, never going back; the compare step
  starts after the last model. The words ("2 of 3 models finished") are
  the interface's; the frame carries the unit they are said in.
* PS22 -- ``progress`` only when its units are the step's whole work: a
  pipeline ``web_search`` step and the consensus compare step carry none;
  a step that runs a strategy of its own and the consensus query do.
* PS12 -- in every fixture stream, ``done`` is the number of sub-unit ends
  emitted before its frame; the progress path reads no clock.

The builder and the recorder are standard library only, so they are
imported directly; the recorder's sink is a list. Each "raises after the
close" is paired with a witness recorder, not closed, on which the same
event is accepted. The runner runs for real behind a scripted agentic
executor, with the emergency stop and the governor as seams; the one
clause that needs the executor's own admission loads the real executor in
the shared isolation window, with a refusing governor behind it.
"""

import ast
import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

import opti_oignon.pipelines as P  # noqa: E402
from opti_oignon import pipeline_step as ps  # noqa: E402

BUDGET_S = {
    "test_ps1_the_step_event_builder_is_closed_and_total": 2.0,
    "test_ps25_the_step_recorder_is_the_transition_rule": 2.0,
    "test_ps2_a_pipeline_run_announces_every_step_and_ends_each_once": 2.0,
    "test_ps3_a_step_that_raises_ends_failed_and_the_next_step_runs": 2.0,
    "test_ps4_a_step_whose_condition_is_false_ends_skipped": 2.0,
    "test_ps5_the_stops_and_the_runners_admission_end_steps_honestly": 2.0,
    "test_ps13_a_step_ends_with_the_pipeline_its_turn_ran": 2.0,
    "test_ps17_a_fixed_server_error_ends_its_step_failed": 2.0,
    "test_ps21_a_safety_mechanism_never_ends_a_step_failed": 2.0,
    "test_ps24_a_refusal_inside_a_step_ends_the_run": 2.0,
    "test_ps9_a_reasoning_strategy_announces_its_steps": 2.0,
    "test_ps16_a_self_correction_is_two_steps_with_no_progress": 2.0,
    "test_ps11_the_consensus_query_counts_the_models_that_finished": 2.0,
    "test_ps22_progress_only_when_its_units_are_the_whole_step": 2.0,
    "test_ps12_progress_counts_the_ends_and_reads_no_clock": 2.0,
}

_FIELDS = (
    "v", "seq", "run", "kind", "name", "pipeline_id", "parent", "index",
    "total", "label", "step_type", "state", "progress", "reason", "ran_as",
    "duration_ms",
)
_REQUIRED = ("seq", "run", "kind", "name", "index", "label", "state")
_TERMINAL = ("done", "failed", "skipped", "cancelled", "not_run")


def _fields(**over):
    """A valid pending pipeline step, with ``over`` laid on top."""
    fields = dict(seq=1, run="r1", kind="exec_pipeline", name="Research",
                  index=0, total=3, label="Search the web", state="pending")
    fields.update(over)
    return fields


def _without(key):
    fields = _fields()
    del fields[key]
    return fields


# ---------------------------------------------------------------------------
# PS1 -- the builder is closed and total
# ---------------------------------------------------------------------------

def _ps1_c1():
    """Total: every field present, in the schema's order, and each state,
    first-wave kind and first-wave unit buildable with its own fields."""
    frame = ps.build(**_fields())
    assert tuple(frame) == _FIELDS
    assert frame["v"] == 1
    assert [key for key in _FIELDS if frame[key] is None] == [
        "pipeline_id", "parent", "step_type", "progress", "reason", "ran_as",
        "duration_ms"]

    kinds = (("exec_pipeline", "Research"), ("reasoning", "decompose"),
             ("reasoning", "tree_of_thought"), ("reasoning", "self_consistency"),
             ("consensus", "consensus"), ("self_correct", "self_correct"))
    for kind, name in kinds:
        frame = ps.build(**_fields(kind=kind, name=name))
        assert (frame["kind"], frame["name"]) == (kind, name)

    per_state = {
        "pending": {},
        "running": {"progress": {"done": 1, "total": 3, "unit": "sub_step"}},
        "done": {"ran_as": "tool_calling", "duration_ms": 12},
        "failed": {"reason": "Connection refused", "duration_ms": 3},
        "skipped": {},
        "cancelled": {"reason": "Stop all engaged", "duration_ms": 0},
        "not_run": {"reason": "Resources refused for m"},
    }
    for state, extra in per_state.items():
        frame = ps.build(**_fields(state=state, **extra))
        assert tuple(frame) == _FIELDS
        assert frame["state"] == state
        for key, value in extra.items():
            assert frame[key] == value

    for unit in ("sub_step", "model", "sample"):
        progress = {"done": 0, "total": 1, "unit": unit}
        assert ps.build(**_fields(state="running", progress=progress))["progress"] == progress

    frame = ps.build(**_fields(pipeline_id="research", step_type="web_search",
                               parent={"run": "r0", "index": 2}, total=None))
    assert frame["pipeline_id"] == "research"
    assert frame["step_type"] == "web_search"
    assert frame["parent"] == {"run": "r0", "index": 2}
    assert frame["total"] is None


def _invalid():
    """Every input the builder must refuse, each with its reason."""
    cases = [(f"missing {key}", _without(key)) for key in _REQUIRED]
    cases += [
        ("unknown field", _fields(colour="red")),
        ("seq zero", _fields(seq=0)),
        ("seq a bool", _fields(seq=True)),
        ("seq a string", _fields(seq="1")),
        ("run empty", _fields(run="")),
        ("run not a string", _fields(run=1)),
        ("kind coding", _fields(kind="coding", name="coding")),
        ("kind research", _fields(kind="research", name="research")),
        ("kind search", _fields(kind="search", name="search")),
        ("kind unknown", _fields(kind="plan")),
        ("name empty", _fields(name="")),
        ("reasoning name outside the three", _fields(kind="reasoning", name="plan")),
        ("consensus named otherwise", _fields(kind="consensus", name="vote")),
        ("self-correction named otherwise", _fields(kind="self_correct", name="fix")),
        ("pipeline_id outside a pipeline", _fields(kind="consensus", name="consensus",
                                                   pipeline_id="research")),
        ("pipeline_id not a string", _fields(pipeline_id=5)),
        ("step_type outside a pipeline", _fields(kind="reasoning", name="decompose",
                                                 step_type="generate")),
        ("step_type not a string", _fields(step_type=5)),
        ("parent without an index", _fields(parent={"run": "r0"})),
        ("parent with a negative index", _fields(parent={"run": "r0", "index": -1})),
        ("parent with an extra key", _fields(parent={"run": "r0", "index": 0, "x": 1})),
        ("parent not a mapping", _fields(parent="r0")),
        ("index negative", _fields(index=-1)),
        ("index a bool", _fields(index=False)),
        ("index at the total", _fields(index=3, total=3)),
        ("total zero", _fields(total=0)),
        ("total a string", _fields(total="3")),
        ("total a bool", _fields(total=True)),
        ("label not a string", _fields(label=None)),
        ("state unknown", _fields(state="unknown")),
        ("state capitalised", _fields(state="Running")),
        ("progress with done above total", _fields(
            state="running", progress={"done": 4, "total": 3, "unit": "model"})),
        ("progress with a negative done", _fields(
            state="running", progress={"done": -1, "total": 3, "unit": "model"})),
        ("progress with a zero total", _fields(
            state="running", progress={"done": 0, "total": 0, "unit": "model"})),
        ("progress in searches", _fields(
            state="running", progress={"done": 0, "total": 1, "unit": "search"})),
        ("progress in an unknown unit", _fields(
            state="running", progress={"done": 0, "total": 1, "unit": "file"})),
        ("progress without a unit", _fields(
            state="running", progress={"done": 0, "total": 1})),
        ("progress with an extra key", _fields(
            state="running", progress={"done": 0, "total": 1, "unit": "model", "x": 1})),
        ("progress with a bool done", _fields(
            state="running", progress={"done": True, "total": 1, "unit": "model"})),
        ("progress not a mapping", _fields(state="running", progress=[0, 1, "model"])),
        ("reason not a string", _fields(state="failed", reason=5)),
        ("ran_as outside a pipeline", _fields(kind="reasoning", name="decompose",
                                              state="done", ran_as="direct")),
        ("ran_as not a string", _fields(state="done", ran_as=5)),
        ("duration_ms negative", _fields(state="done", duration_ms=-1)),
        ("duration_ms a float", _fields(state="done", duration_ms=1.5)),
    ]
    for state in ("pending", "done", "failed", "skipped", "cancelled", "not_run"):
        cases.append((f"progress on {state}", _fields(
            state=state, progress={"done": 1, "total": 2, "unit": "sub_step"})))
    for state in ("pending", "running", "done", "skipped"):
        cases.append((f"reason on {state}", _fields(state=state, reason="why")))
    for state in ("pending", "running", "skipped", "not_run"):
        cases.append((f"ran_as on {state}", _fields(state=state, ran_as="direct")))
        cases.append((f"duration_ms on {state}", _fields(state=state, duration_ms=5)))
    return cases


def _ps1_c2():
    """Closed: every input outside the schema raises ValueError."""
    cases = _invalid()
    assert len(cases) >= 60
    accepted = []
    for why, fields in cases:
        try:
            ps.build(**fields)
        except ValueError:
            continue
        accepted.append(why)
    assert accepted == []


def _ps1_c3():
    """The label is cut to 120 characters and the reason to 300; at the cap
    nothing is cut."""
    assert ps.build(**_fields(label="x" * 200))["label"] == "x" * 120
    assert ps.build(**_fields(label="z" * 120))["label"] == "z" * 120
    frame = ps.build(**_fields(state="failed", reason="y" * 400, duration_ms=1))
    assert frame["reason"] == "y" * 300
    frame = ps.build(**_fields(state="not_run", reason="w" * 300))
    assert frame["reason"] == "w" * 300


def test_ps1_the_step_event_builder_is_closed_and_total():
    _ps1_c1()
    _ps1_c2()
    _ps1_c3()


# ---------------------------------------------------------------------------
# PS25 -- the recorder is the transition rule
# ---------------------------------------------------------------------------

def _recorder():
    frames = []
    return ps.StepRecorder(frames.append), frames


_EVENTS = ("pending", "running", "progress") + _TERMINAL

# The one legal next event of each state of a step, per the transition rule;
# ``None`` is a step never emitted.
_LEGAL = {
    None: {"pending"},
    "pending": {"running", "skipped", "not_run"},
    "running": {"progress", "done", "failed", "cancelled"},
    "done": set(), "failed": set(), "skipped": set(),
    "cancelled": set(), "not_run": set(),
}
_PATH = {
    None: (),
    "pending": ("pending",),
    "running": ("pending", "running"),
    "done": ("pending", "running", "done"),
    "failed": ("pending", "running", "failed"),
    "cancelled": ("pending", "running", "cancelled"),
    "skipped": ("pending", "skipped"),
    "not_run": ("pending", "not_run"),
}


def _apply(rec, run, event):
    if event == "pending":
        rec.pending(run, 0, "Read the notes")
    elif event == "running":
        rec.running(run, 0)
    elif event == "progress":
        rec.progress(run, 0, 1, 2, "sub_step")
    else:
        reason = "why" if event in ("failed", "cancelled", "not_run") else None
        rec.end(run, 0, event, reason=reason)


def _ps25_c1():
    """From every state of a step, exactly the events of the transition rule
    are accepted, each sending one frame; every other one -- a second
    terminal state among them -- raises and sends nothing."""
    wrong = []
    tried = 0
    for state, path in _PATH.items():
        for event in _EVENTS:
            rec, frames = _recorder()
            run = rec.start_run("exec_pipeline", "Research", total=1)
            for step in path:
                _apply(rec, run, step)
            before = len(frames)
            tried += 1
            try:
                _apply(rec, run, event)
            except ValueError:
                if event in _LEGAL[state]:
                    wrong.append((state, event, "refused"))
                if len(frames) != before:
                    wrong.append((state, event, "sent a frame and raised"))
                continue
            if event not in _LEGAL[state]:
                wrong.append((state, event, "accepted"))
            elif len(frames) != before + 1:
                wrong.append((state, event, f"sent {len(frames) - before} frames"))
            elif frames[-1]["state"] != ("running" if event == "progress" else event):
                wrong.append((state, event, f"sent {frames[-1]['state']}"))
    assert tried == len(_PATH) * len(_EVENTS)
    assert wrong == []


def _open_reply():
    """A reply with open steps: a pipeline run, step 0 running and step 1
    pending, index 2 never emitted, and a nested decompose run whose total
    is not known yet."""
    rec, frames = _recorder()
    run = rec.start_run("exec_pipeline", "Research", total=3, pipeline_id="research")
    rec.pending(run, 0, "Plan")
    rec.pending(run, 1, "Write")
    rec.running(run, 0)
    child = rec.start_run("reasoning", "decompose")
    rec.pending(child, 0, "Break the question down")
    rec.running(child, 0)
    return rec, frames, run, child


def _ps25_c2():
    """After ``close`` every event raises and sends nothing, while the same
    event is accepted by a witness reply that was not closed."""
    events = {
        "start_run": lambda rec, run, child: rec.start_run("consensus", "consensus"),
        "pending": lambda rec, run, child: rec.pending(run, 2, "Check"),
        "set_total": lambda rec, run, child: rec.set_total(child, 3),
        "close": lambda rec, run, child: rec.close("error", "Connection refused"),
        "running": lambda rec, run, child: rec.running(run, 1),
        "progress": lambda rec, run, child: rec.progress(run, 0, 1, 2, "sub_step"),
        "end": lambda rec, run, child: rec.end(run, 0, "done"),
    }
    accepted_after_close = []
    refused_by_witness = []
    for name, event in events.items():
        for cause in ("stop", "error"):
            rec, frames, run, child = _open_reply()
            rec.close(cause)
            before = len(frames)
            try:
                event(rec, run, child)
            except ValueError:
                if len(frames) != before:
                    accepted_after_close.append((cause, name, "sent a frame"))
            else:
                accepted_after_close.append((cause, name, "accepted"))
        witness, _, run, child = _open_reply()
        try:
            event(witness, run, child)
        except ValueError:
            refused_by_witness.append(name)
    assert accepted_after_close == []
    assert refused_by_witness == []


def _ps25_c3():
    """``seq`` starts at 1 and strictly increases, in the sink's order,
    across a pipeline run, a nested run, a later run and the close."""
    rec, frames = _recorder()
    run = rec.start_run("exec_pipeline", "Research", total=2, pipeline_id="research")
    rec.pending(run, 0, "Plan")
    rec.pending(run, 1, "Write")
    rec.running(run, 0)
    child = rec.start_run("reasoning", "decompose")
    rec.pending(child, 0, "Break the question down")
    rec.running(child, 0)
    rec.set_total(child, 3)
    rec.pending(child, 1, "What the notes say")
    rec.pending(child, 2, "Combine")
    rec.end(child, 0, "done")
    rec.running(child, 1)
    rec.end(child, 1, "done")
    rec.running(child, 2)
    rec.end(child, 2, "done")
    rec.end(run, 0, "done", ran_as="reasoning")
    rec.running(run, 1)
    later = rec.start_run("self_correct", "self_correct", total=2)
    rec.pending(later, 0, "Review")
    rec.close("stop")
    seqs = [frame["seq"] for frame in frames]
    assert len({frame["run"] for frame in frames}) >= 3
    assert len(seqs) >= 15
    assert seqs == list(range(1, len(seqs) + 1))


def test_ps25_the_step_recorder_is_the_transition_rule():
    _ps25_c1()
    _ps25_c2()
    _ps25_c3()


# ---------------------------------------------------------------------------
# The runner's world: a scripted agentic executor, the two seams, a turn
# ---------------------------------------------------------------------------

_MESSAGE = "What does an onion need to grow?"
_REFUSAL = "Not enough memory to load m"
_FIXED = (
    "[ERR] Executor not available",
    "[ERR] No initial response generated",
    "[ERR] Context exceeds model limit: 9000 tokens for 8192",
)


def _routing(model="m"):
    return SimpleNamespace(
        model=model, task_type="general", temperature=0.2,
        prompt_variant="standard", timeout=30,
        routing_reason="contract", images=None,
    )


class _Estop:
    def __init__(self):
        self.engaged = False

    def is_stopped(self):
        return self.engaged


class _Governor:
    """The runner's admission seam; refuses the admissions numbered in
    ``refuse`` (0-based, in the order they are asked)."""

    def __init__(self, refuse=()):
        self.refuse = set(refuse)
        self.asked = []

    def get_resource_governor(self):
        return self

    def admit(self, model, requested_ctx=None, caller="chat"):
        refused = len(self.asked) in self.refuse
        self.asked.append(model)
        return SimpleNamespace(admitted=not refused,
                               reason="the model does not fit" if refused else "")


class _NoRouter:
    enabled = False

    def override_routing(self, routing, step_type):
        return routing


class _Scripted:
    """The agentic executor behind the runner: one scripted act per call.
    The shared instance's last pipeline is always another value."""

    def __init__(self, world, acts):
        self.world = world
        self.acts = list(acts)
        self.prompts = []
        self.last_pipeline = self._last_pipeline = "consensus"

    def execute(self, *, message, routing, run=None, **kwargs):
        self.prompts.append(message)
        return self.acts[len(self.prompts) - 1](self.world, run, message, routing)


def _answers(pipeline="direct", text="Sun, water and loose soil."):
    def act(world, run, message, routing):
        if pipeline is not None:
            run.results["pipeline"] = pipeline
        yield text
    return act


def _raises(text="executor boom"):
    def act(world, run, message, routing):
        raise RuntimeError(text)
        yield ""  # a generator, as the executor's execute is
    return act


def _refused(message=_REFUSAL):
    """The executor's own admission refused inside the step."""
    def act(world, run, message_, routing):
        run.results["admission_refused"] = message
        yield f"[ERR] {message}"
    return act


def _fixed_error(literal):
    def act(world, run, message, routing):
        run.results["step_error"] = literal
        yield literal
    return act


def _stops(emergency=False):
    """A stop engaged while the step runs; the stage then ends."""
    def act(world, run, message, routing):
        if emergency:
            world.estop.engaged = True
        else:
            run.stop.set()
        yield "[Cancelled]"
    return act


class _World:
    def __init__(self, monkeypatch, acts, refuse=()):
        self.estop = _Estop()
        self.governor = _Governor(refuse)
        monkeypatch.setattr(P, "_resolve_emergency_stop", lambda: self.estop)
        monkeypatch.setattr(P, "_resolve_resource_governor", lambda: self.governor)
        self.executor = _Scripted(self, acts)
        self.runner = P.PipelineRunner(agentic_executor=self.executor, smart_router=_NoRouter())
        self.frames = []
        self.run = SimpleNamespace(stop=threading.Event(), results={},
                                   steps=ps.StepRecorder(self.frames.append))
        self.out = []

    def go(self, steps, message=_MESSAGE, on_step_end=None):
        pipeline = P.ExecutionPipeline(id="research", name="Research", steps=steps)
        self.out = list(self.runner.execute(
            pipeline=pipeline, message=message, routing=_routing(),
            on_step_end=on_step_end, run=self.run))
        return self.frames

    def states(self, index):
        return [f["state"] for f in self.frames
                if f["kind"] == "exec_pipeline" and f["index"] == index]

    def last(self, index):
        frames = [f for f in self.frames
                  if f["kind"] == "exec_pipeline" and f["index"] == index]
        return frames[-1] if frames else None


def _steps(*types_):
    return [P.ExecutionStep(step_type=t, label=f"Step {i + 1}") for i, t in enumerate(types_)]


# ---------------------------------------------------------------------------
# PS2 -- a run announces every step and ends each once
# ---------------------------------------------------------------------------

def _ps2_c1(monkeypatch):
    """One pending per step, in index order, before any running."""
    world = _World(monkeypatch, [_answers(), _answers(), _answers()])
    frames = world.go(_steps("direct", "think", "tools"))
    assert len(frames) >= 9
    head = frames[:3]
    assert [f["state"] for f in head] == ["pending"] * 3
    assert [f["index"] for f in head] == [0, 1, 2]
    assert [f["step_type"] for f in head] == ["direct", "think", "tools"]
    assert [f["label"] for f in head] == ["Step 1", "Step 2", "Step 3"]
    assert {(f["kind"], f["name"], f["pipeline_id"], f["total"]) for f in head} == {
        ("exec_pipeline", "Research", "research", 3)}
    assert [world.states(i) for i in range(3)] == [["pending", "running", "done"]] * 3


def _ps2_c2(monkeypatch):
    """Every step reaches exactly one terminal state, whatever ends it."""
    long_answer = "x" * 600
    world = _World(monkeypatch, [_answers(text="short"), _raises(), _answers(text=long_answer)])
    steps = _steps("direct", "think", "direct", "tools")
    steps[2].condition = "if_long_input"
    world.go(steps)
    terminal = [[s for s in world.states(i) if s in _TERMINAL] for i in range(4)]
    assert terminal == [["done"], ["failed"], ["skipped"], ["done"]]


def _ps2_c3(monkeypatch):
    """``seq`` starts at 1 and strictly increases over the whole run."""
    world = _World(monkeypatch, [_answers(), _raises(), _answers()])
    frames = world.go(_steps("direct", "think", "tools"))
    seqs = [f["seq"] for f in frames]
    assert len(seqs) >= 9
    assert seqs == list(range(1, len(seqs) + 1))


def _ps2_c4(monkeypatch):
    """A runner that cannot start (no executor, no steps) sends nothing;
    the witness pipeline of the first clause sends frames."""
    world = _World(monkeypatch, [])
    world.go([])
    assert world.frames == []
    world = _World(monkeypatch, [_answers()])
    monkeypatch.setattr(world.runner, "_get_executor", lambda: None)
    world.go(_steps("direct"))
    assert world.frames == []
    assert world.executor.prompts == []


def test_ps2_a_pipeline_run_announces_every_step_and_ends_each_once(monkeypatch):
    _ps2_c1(monkeypatch)
    _ps2_c2(monkeypatch)
    _ps2_c3(monkeypatch)
    _ps2_c4(monkeypatch)


# ---------------------------------------------------------------------------
# PS3 -- a step that raises
# ---------------------------------------------------------------------------

def test_ps3_a_step_that_raises_ends_failed_and_the_next_step_runs(monkeypatch):
    world = _World(monkeypatch, [_answers(), _raises("executor boom"), _answers()])
    world.go(_steps("direct", "think", "tools"))
    assert world.states(1) == ["pending", "running", "failed"]
    assert world.last(1)["reason"] == "executor boom"
    assert world.last(1)["duration_ms"] is not None
    assert len(world.executor.prompts) == 3
    assert world.states(2) == ["pending", "running", "done"]


# ---------------------------------------------------------------------------
# PS4 -- a false condition
# ---------------------------------------------------------------------------

def test_ps4_a_step_whose_condition_is_false_ends_skipped(monkeypatch):
    for answer, ran in (("short", False), ("x" * 600, True)):
        world = _World(monkeypatch, [_answers(text=answer), _answers(), _answers()])
        steps = _steps("direct", "think", "tools")
        steps[1].condition = "if_long_input"
        world.go(steps)
        if ran:
            # Witness: the same condition met, the step runs.
            assert world.states(1) == ["pending", "running", "done"]
            assert len(world.executor.prompts) == 3
        else:
            assert world.states(1) == ["pending", "skipped"]
            assert len(world.executor.prompts) == 2
        assert world.states(2) == ["pending", "running", "done"]


# ---------------------------------------------------------------------------
# PS5 -- the stops and the runner's admission
# ---------------------------------------------------------------------------

def _ps5_c1(monkeypatch):
    """The emergency stop engaged between steps: the rest end not_run."""
    world = _World(monkeypatch, [_answers(), _answers(), _answers()])

    def engage(index, step, output):
        world.estop.engaged = True
    world.go(_steps("direct", "think", "tools"), on_step_end=engage)
    assert world.states(0) == ["pending", "running", "done"]
    for index in (1, 2):
        assert world.states(index) == ["pending", "not_run"]
        assert world.last(index)["reason"] == "Stop all engaged"
    assert len(world.executor.prompts) == 1


def _ps5_c2(monkeypatch):
    """The runner's admission refused at step k: running never sent for k,
    and k and the later steps end not_run with the governor's reason."""
    world = _World(monkeypatch, [_answers(), _answers(), _answers()], refuse={1})
    world.go(_steps("direct", "think", "tools"))
    assert world.governor.asked == ["m", "m"]
    assert world.states(0) == ["pending", "running", "done"]
    for index in (1, 2):
        assert world.states(index) == ["pending", "not_run"]
        assert world.last(index)["reason"] == "Resources refused for m"
    assert len(world.executor.prompts) == 1


def _ps5_c3(monkeypatch):
    """A stop engaged during a step ends that step cancelled, never done;
    under the emergency stop with its reason, the rest not_run."""
    world = _World(monkeypatch, [_answers(), _stops(), _answers()])
    world.go(_steps("direct", "think", "tools"))
    assert world.states(1) == ["pending", "running", "cancelled"]
    assert world.last(1)["reason"] is None
    assert "running" not in world.states(2)
    assert len(world.executor.prompts) == 2

    world = _World(monkeypatch, [_answers(), _stops(emergency=True), _answers()])
    world.go(_steps("direct", "think", "tools"))
    assert world.states(1) == ["pending", "running", "cancelled"]
    assert world.last(1)["reason"] == "Stop all engaged"
    assert world.states(2) == ["pending", "not_run"]
    assert world.last(2)["reason"] == "Stop all engaged"


def test_ps5_the_stops_and_the_runners_admission_end_steps_honestly(monkeypatch):
    _ps5_c1(monkeypatch)
    _ps5_c2(monkeypatch)
    _ps5_c3(monkeypatch)


# ---------------------------------------------------------------------------
# PS13 -- ran_as is the turn's
# ---------------------------------------------------------------------------

def _ps13_c1(monkeypatch):
    """``ran_as`` is the pipeline the turn reports, also when it differs
    from the step's type; a step that reports none has none."""
    world = _World(monkeypatch, [_answers("tools"), _answers("direct"), _answers(None)])
    world.go(_steps("think", "direct", "web_search"))
    assert [world.last(i)["ran_as"] for i in range(3)] == ["tools", "direct", None]


def _ps13_c2(monkeypatch):
    """With the shared instance's last pipeline set to another value,
    ``ran_as`` is still the turn's."""
    world = _World(monkeypatch, [_answers("think")])
    assert world.executor.last_pipeline == "consensus"
    world.go(_steps("think"))
    assert world.last(0)["ran_as"] == "think"


def test_ps13_a_step_ends_with_the_pipeline_its_turn_ran(monkeypatch):
    _ps13_c1(monkeypatch)
    _ps13_c2(monkeypatch)


# ---------------------------------------------------------------------------
# PS17 -- the server's own fixed errors
# ---------------------------------------------------------------------------

_PKG = Path(__file__).resolve().parent.parent / "opti_oignon"


def _writes(key):
    """Every ``<results>["key"] = ...`` in the package, with its block:
    ``(file name, line, statements of the block that holds it)``."""
    found = []
    for path in sorted(_PKG.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if f'"{key}"' not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if not isinstance(block, list):
                    continue
                for stmt in block:
                    if isinstance(stmt, ast.Assign) and any(
                            isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                            and t.slice.value == key for t in stmt.targets):
                        found.append((path.name, stmt.lineno, block))
    return found


def _yielded_literals(block):
    """The leading literal text of every ``yield`` of an ``[ERR]`` line in a
    block."""
    out = []
    for stmt in block:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
            value = stmt.value.value
            if isinstance(value, ast.JoinedStr) and value.values and isinstance(value.values[0], ast.Constant):
                text = value.values[0].value
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                text = value.value
            else:
                continue
            if text.startswith("[ERR]"):
                out.append(text)
    return out


def _block_key(name, block):
    return (name, block[0].lineno)


def _ps17_c1(monkeypatch):
    """Each fixed error, reported by its call, ends the step failed with
    that text, never done; the next step runs and is not handed it."""
    for literal in _FIXED:
        world = _World(monkeypatch, [_answers(), _fixed_error(literal), _answers()])
        world.go(_steps("direct", "think", "tools"))
        assert world.states(1) == ["pending", "running", "failed"]
        assert world.last(1)["reason"] == literal
        assert world.states(2) == ["pending", "running", "done"]
        assert literal not in world.executor.prompts[2]


def _ps17_c2(monkeypatch):
    """A model reply equal to a fixed error, with no report from its call,
    is the model's text: the step ends done."""
    for literal in _FIXED:
        world = _World(monkeypatch, [_answers(text=literal)])
        world.go(_steps("direct"))
        assert world.states(0) == ["pending", "running", "done"]


def _ps17_c3():
    """The fixed errors are a closed list of sites: exactly three writes of
    the signal in the package, each beside the literal it reports."""
    writes = _writes("step_error")
    sites = sorted((name, lit) for name, _, block in writes for lit in _yielded_literals(block))
    assert len(writes) == 3
    assert sites == [
        ("agentic_executor.py", "[ERR] Executor not available"),
        ("agentic_executor.py", "[ERR] No initial response generated"),
        ("executor.py", "[ERR] Context exceeds model limit: "),
    ]


def test_ps17_a_fixed_server_error_ends_its_step_failed(monkeypatch):
    _ps17_c1(monkeypatch)
    _ps17_c2(monkeypatch)
    _ps17_c3()


# ---------------------------------------------------------------------------
# PS21 -- a safety mechanism never ends a step failed
# ---------------------------------------------------------------------------

def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _Ledger:
    def __init__(self):
        self.records = []

    def record(self, **fields):
        self.records.append(fields)
        return True


class _Client:
    """The client behind the registry; a refused call never reaches it."""

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return iter([{"message": {"content": "Hello"}}])


def _refusing_governor(asked):
    decision = SimpleNamespace(
        admitted=False, action="refuse", reason="memory", num_ctx=None,
        keep_alive=None, conditional_on_eviction=False, load_expected=False,
        refusal_payload=lambda: {"message": _REFUSAL})

    def admit(model, requested_ctx=None, caller="chat", extra_models=None):
        asked.append(model)
        return decision

    return _module(
        "opti_oignon.resource_governor",
        get_resource_governor=lambda: SimpleNamespace(
            config=SimpleNamespace(enabled=True), admit=admit),
        set_active_ticket=lambda decision: None,
        clear_active_ticket=lambda: None,
        ticket_scope=lambda decision: None,
    )


def _real_executor(asked):
    ledger, client = _Ledger(), _Client()
    seeded = {
        "opti_oignon.config": _module("opti_oignon.config", config=SimpleNamespace(
            get_model=lambda *a, **k: "test-model:1b",
            get_temperature=lambda *a, **k: 0.2)),
        "opti_oignon.router": _module("opti_oignon.router",
                                      RoutingResult=type("RoutingResult", (), {})),
        "opti_oignon.context_ledger": _module("opti_oignon.context_ledger",
                                              get_context_ledger=lambda: ledger),
        "opti_oignon.resource_governor": _refusing_governor(asked),
        # The runner imports the recorder's module while the window is open.
        "opti_oignon.pipeline_step": ps,
    }
    seed_registry(seeded, client)
    loaded, close = isolate(
        targets={
            "opti_oignon.context_dedup": source("context_dedup.py"),
            "opti_oignon.executor": source("executor.py"),
        },
        seeded=seeded,
    )
    return loaded["opti_oignon.executor"].Executor(), client, ledger, close


def _ps21_c1(monkeypatch):
    """The executor's main admission refused, with a fake governor: the
    call reports it structurally, and the runner ends the step cancelled
    with the governor's message."""
    asked = []
    executor, client, ledger, close = _real_executor(asked)
    try:
        def real(world, run, message, routing):
            return (yield from executor.execute(message, routing, refine=False, run=run))

        world = _World(monkeypatch, [_answers(), real, _answers()])
        world.go(_steps("direct", "think", "tools"))
    finally:
        close()
    assert asked == ["m"]
    assert client.calls == []
    assert [r.get("outcome") for r in ledger.records] == ["governor_refused"]
    assert world.run.results.get("admission_refused") == _REFUSAL
    assert world.states(1) == ["pending", "running", "cancelled"]
    assert world.last(1)["reason"] == _REFUSAL


def _ps21_c2(monkeypatch):
    """No step ended by Stop, Stop all or the governor is failed, in any
    fixture stream; the witness stream with a raising step is."""
    streams = {}
    world = _World(monkeypatch, [_answers(), _stops(), _answers()])
    streams["stop"] = world.go(_steps("direct", "think", "tools"))
    world = _World(monkeypatch, [_answers(), _stops(emergency=True), _answers()])
    streams["stop all during a step"] = world.go(_steps("direct", "think", "tools"))
    world = _World(monkeypatch, [_answers(), _answers(), _answers()])

    def engage(index, step, output):
        world.estop.engaged = True
    streams["stop all between steps"] = world.go(_steps("direct", "think", "tools"),
                                                 on_step_end=engage)
    world = _World(monkeypatch, [_answers(), _answers(), _answers()], refuse={1})
    streams["runner admission"] = world.go(_steps("direct", "think", "tools"))
    world = _World(monkeypatch, [_answers(), _refused(), _answers()])
    streams["inner admission"] = world.go(_steps("direct", "think", "tools"))
    failed = {name: [f["index"] for f in frames if f["state"] == "failed"]
              for name, frames in streams.items()}
    assert failed == {name: [] for name in streams}
    ended = {name: [f["state"] for f in frames if f["state"] in _TERMINAL]
             for name, frames in streams.items()}
    assert all(len(states) >= 2 for states in ended.values())
    world = _World(monkeypatch, [_answers(), _raises(), _answers()])
    assert [f["index"] for f in world.go(_steps("direct", "think", "tools"))
            if f["state"] == "failed"] == [1]


def _ps21_c3():
    """The governor's refusal is not a fixed error: the two refusal sites
    report ``admission_refused`` beside their ledger write, and no block
    reports both signals."""
    refusals = _writes("admission_refused")
    ledgered = set()
    for name, _, block in refusals:
        for stmt in block:
            call = stmt.value if isinstance(stmt, ast.Expr) else None
            if (isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_emit_ledger"
                    and call.args and isinstance(call.args[0], ast.Constant)):
                ledgered.add((name, call.args[0].value))
    assert len(refusals) == 2
    assert ledgered == {("executor.py", "vision_refused"), ("executor.py", "governor_refused")}
    fixed_blocks = {_block_key(name, block) for name, _, block in _writes("step_error")}
    assert len(fixed_blocks) >= 3
    assert [line for name, line, block in refusals
            if _block_key(name, block) in fixed_blocks] == []


def test_ps21_a_safety_mechanism_never_ends_a_step_failed(monkeypatch):
    _ps21_c1(monkeypatch)
    _ps21_c2(monkeypatch)
    _ps21_c3()


# ---------------------------------------------------------------------------
# PS24 -- a refusal inside a step ends the run
# ---------------------------------------------------------------------------

def test_ps24_a_refusal_inside_a_step_ends_the_run(monkeypatch):
    # Witness: with no refusal, all four steps run, each handed the last.
    world = _World(monkeypatch, [_answers(text=f"answer {i}") for i in range(4)])
    world.go(_steps("direct", "think", "tools", "direct"))
    assert len(world.executor.prompts) == 4
    assert "answer 1" in world.executor.prompts[2]

    world = _World(monkeypatch, [_answers(), _refused(), _answers(), _answers()])
    world.go(_steps("direct", "think", "tools", "direct"))
    assert world.states(1) == ["pending", "running", "cancelled"]
    assert world.last(1)["reason"] == _REFUSAL
    for index in (2, 3):
        assert world.states(index) == ["pending", "not_run"]
        assert world.last(index)["reason"] == _REFUSAL
    assert len(world.executor.prompts) == 2
    assert [p for p in world.executor.prompts if _REFUSAL in p] == []


# ---------------------------------------------------------------------------
# PS9, PS16 -- the reasoning strategies and the self-correction
# ---------------------------------------------------------------------------

class _Backend:
    """A registry backend for the reasoning engine: ``answer(n)`` is the
    reply to the n-th call, counted from 1; an exception it returns is
    raised by the call."""

    def __init__(self, answer):
        self.calls = 0
        self.answer = answer

    def generate(self, model=None, messages=None, options=None, **kw):
        self.calls += 1
        reply = self.answer(self.calls)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=reply)


def _agentic(backend):
    """The real engine and the real agentic executor in the shared window."""
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = None
    loaded, close = isolate(
        targets={
            "opti_oignon.reasoning": source("reasoning.py"),
            "opti_oignon.agentic_executor": source("agentic_executor.py"),
        },
        seeded={
            "opti_oignon.conversation": conv,
            # The executor imports the recorder's module while the window is open.
            "opti_oignon.pipeline_step": ps,
        },
    )
    rs, ae = loaded["opti_oignon.reasoning"], loaded["opti_oignon.agentic_executor"]
    rs._resolve_backend = lambda model: backend
    return rs, ae, close


def _turn_run(frames):
    return SimpleNamespace(stop=threading.Event(), results={},
                           steps=ps.StepRecorder(frames.append))


def _reason(answer, **config):
    """One reasoning pipeline under a turn; returns its frames."""
    frames = []
    rs, ae, close = _agentic(_Backend(answer))
    try:
        ae.REASONING_AVAILABLE = True
        engine = rs.ReasoningEngine(config=rs.ReasoningConfig(**config), default_model="m")
        agent = ae.AgenticExecutor(executor=SimpleNamespace(), reasoning_engine=engine,
                                   default_model="m")
        turn = ae._Turn(_turn_run(frames))
        for _ in agent._execute_reasoning_pipeline(_MESSAGE, _routing(), None, None, turn=turn):
            pass
    finally:
        close()
    return frames


def _by_index(frames):
    steps = {}
    for frame in frames:
        steps.setdefault(frame["index"], []).append(frame)
    return steps


def _plan(*titles):
    return json.dumps([{"title": t, "question": f"{t}?"} for t in titles])


def _ps9_c1():
    """decompose: step 0 runs with no total; the plan brings n + 2."""
    frames = _reason(lambda n: _plan("Soil", "Water") if n == 1 else f"answer {n}")
    assert frames, "a decompose run sent no step frame"
    assert {(f["kind"], f["name"], f["run"]) for f in frames} == {("reasoning", "decompose", "r1")}
    steps = _by_index(frames)
    assert sorted(steps) == [0, 1, 2, 3]
    assert [f["state"] for f in steps[0]] == ["pending", "running", "done"]
    assert steps[0][0]["label"] == "Break the question down"
    assert steps[0][1]["total"] is None, "step 0 runs before the plan: its total is unknown"
    assert [steps[i][0]["label"] for i in (1, 2, 3)] == ["Soil", "Water", "Combine the sub-answers"]
    for index in (1, 2, 3):
        assert steps[index][0]["state"] == "pending"
        assert steps[index][0]["seq"] > steps[0][1]["seq"]
        assert [f["total"] for f in steps[index]] == [4] * len(steps[index]), (
            f"step {index} does not carry the planned total n + 2 = 4")
        assert [f["state"] for f in steps[index]] == ["pending", "running", "done"]


def _ps9_c2():
    """tree_of_thought: three steps, the total known from the first frame."""
    scores = {2: '{"score": 0.4}', 3: '{"score": 0.8}'}
    frames = _reason(
        lambda n: '[{"approach": "a"}, {"approach": "b"}]' if n == 1 else scores.get(n, "developed"),
        default_strategy="tree_of_thought")
    assert frames, "a tree-of-thought run sent no step frame"
    assert {f["name"] for f in frames} == {"tree_of_thought"}
    assert [f["total"] for f in frames] == [3] * len(frames)
    steps = _by_index(frames)
    assert {i: s[0]["label"] for i, s in steps.items()} == {
        0: "Draft approaches", 1: "Score the approaches", 2: "Develop the best approach"}
    for index in (0, 1, 2):
        assert [f["state"] for f in steps[index]] == ["pending", "running", "done"]


def _ps9_c3():
    """self_consistency: the total is the number of runs, one sample each."""
    frames = _reason(lambda n: "Loose soil and sun.", default_strategy="self_consistency",
                     self_consistency_runs=4)
    assert frames, "a self-consistency run sent no step frame"
    assert {f["name"] for f in frames} == {"self_consistency"}
    assert [f["total"] for f in frames] == [4] * len(frames)
    steps = _by_index(frames)
    assert [steps[i][0]["label"] for i in sorted(steps)] == [f"Sample {k}" for k in (1, 2, 3, 4)]
    for index in range(4):
        assert [f["state"] for f in steps[index]] == ["pending", "running", "done"]


def _ps9_c4():
    """decompose: a sub-step whose call raised ends failed; the rest runs."""
    frames = _reason(lambda n: _plan("Soil", "Water") if n == 1
                     else RuntimeError("model gone") if n == 2 else f"answer {n}")
    steps = _by_index(frames)
    assert [f["state"] for f in steps.get(1, [])] == ["pending", "running", "failed"]
    assert steps[1][-1]["reason"] == "model gone"
    for index in (2, 3):
        assert [f["state"] for f in steps[index]] == ["pending", "running", "done"]


def test_ps9_a_reasoning_strategy_announces_its_steps():
    _ps9_c1()
    _ps9_c2()
    _ps9_c3()
    _ps9_c4()


class _Corrector:
    """A self-correction engine: two passes, then the corrected text."""

    available = True

    def execute_self_correction(self, user_message, response, model=None, should_stop=None):
        for iteration in (1, 2):
            yield ("correction_step", {"iteration": iteration, "compliance_score": 0.9,
                                       "quality_score": 0.8, "improvements": [],
                                       "duration_ms": 1})
        yield ("correction_done", SimpleNamespace(final_response="Corrected."))
        yield "Corrected."


def test_ps16_a_self_correction_is_two_steps_with_no_progress():
    frames, passes = [], []
    rs, ae, close = _agentic(_Backend(lambda n: "unused"))
    try:
        ae.SELF_CORRECTION_AVAILABLE = True
        draft = SimpleNamespace(execute=lambda **kw: iter(["A first draft."]))
        agent = ae.AgenticExecutor(executor=draft, self_correction_engine=_Corrector(),
                                   default_model="m")
        turn = ae._Turn(_turn_run(frames), on_correction_step=passes.append)
        text = "".join(c for c in agent._execute_self_correct_pipeline(
            _MESSAGE, _routing(), None, None, turn=turn) if isinstance(c, str))
    finally:
        close()
    assert text == "Corrected."
    assert frames, "a self-correction sent no step frame"
    assert {(f["kind"], f["name"]) for f in frames} == {("self_correct", "self_correct")}
    assert [f["total"] for f in frames] == [2] * len(frames)
    steps = _by_index(frames)
    assert {i: s[0]["label"] for i, s in steps.items()} == {
        0: "Write a first answer", 1: "Check and correct"}
    for index in (0, 1):
        assert [f["state"] for f in steps[index]] == ["pending", "running", "done"]
    assert [f for f in frames if f["progress"] is not None] == [], (
        "a self-correction pass is not progress: its iterations stop early")
    assert [p["iteration"] for p in passes] == [1, 2], "the pass is said in words"


# ---------------------------------------------------------------------------
# PS11, PS22, PS12 -- the consensus query and the progress of a step
# ---------------------------------------------------------------------------

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


def _consensus(answer, models, **config):
    """One consensus pipeline under a turn, through the real engine and the
    real agentic executor in the shared window. Returns the stream: every
    step frame, and a ``("model", name)`` marker where the turn's live
    per-model hook fired."""
    stream = []
    conv = types.ModuleType("opti_oignon.conversation")
    conv.conversation_manager = None
    loaded, close = isolate(
        targets={
            "opti_oignon.consensus": source("consensus.py"),
            "opti_oignon.agentic_executor": source("agentic_executor.py"),
        },
        seeded={"opti_oignon.conversation": conv, "opti_oignon.pipeline_step": ps},
    )
    try:
        cs, ae = loaded["opti_oignon.consensus"], loaded["opti_oignon.agentic_executor"]
        backend = _Models(answer)
        cs._resolve_backend = lambda model: backend
        engine = cs.ConsensusEngine(config=cs.ConsensusConfig(**config), default_model="m")
        ae.CONSENSUS_AVAILABLE = True
        agent = ae.AgenticExecutor(executor=SimpleNamespace(), consensus_engine=engine,
                                   default_model="m")
        turn = ae._Turn(_turn_run(stream),
                        on_consensus_model=lambda resp: stream.append(("model", resp.model)))
        for _ in agent._execute_consensus_pipeline(_MESSAGE, _routing(), None, None,
                                                   models=models, turn=turn):
            pass
    finally:
        close()
    return stream


def _frames(stream):
    return [item for item in stream if isinstance(item, dict)]


def _nests(sub_steps):
    """A step that runs a strategy of its own: a nested run on the turn's
    recorder, every sub-step announced, then run and ended done."""
    def act(world, run, message, routing):
        nested = run.steps.start_run("reasoning", "tree_of_thought", total=sub_steps)
        for index in range(sub_steps):
            run.steps.pending(nested, index, f"Part {index + 1}")
        for index in range(sub_steps):
            run.steps.running(nested, index)
            run.steps.end(nested, index, "done")
        run.results["pipeline"] = "reasoning"
        yield "Loose soil."
    return act


def test_ps11_the_consensus_query_counts_the_models_that_finished():
    """Four models asked, three resolved, one of them failing: the query's
    progress is in models, of 3; at every progress frame ``done`` is the
    number of models the live hook reported before it, so it never goes
    back and a failed model counts; the compare step starts after the last."""
    stream = _consensus(lambda model: RuntimeError("model gone") if model == "b"
                        else f"{model}: sun, water and loose soil",
                        ["a", "b", "c", "d"], max_models=3)
    frames = _frames(stream)
    assert frames, "a consensus sent no step frame"
    assert {(f["kind"], f["name"], f["total"]) for f in frames} == {("consensus", "consensus", 2)}
    steps = _by_index(frames)
    assert {i: s[0]["label"] for i, s in steps.items()} == {
        0: "Query the models", 1: "Compare the answers"}
    reported = [item[1] for item in stream if isinstance(item, tuple)]
    assert sorted(reported) == ["a", "b", "c"], (
        f"the live hook reported {reported}: every resolved model, the failed one too, once")
    finished, dones = 0, []
    for item in stream:
        if isinstance(item, tuple):
            finished += 1
        elif item["index"] == 0 and item["progress"] is not None:
            progress = item["progress"]
            assert (progress["total"], progress["unit"]) == (3, "model"), (
                f"the query's progress is {progress}, not in the 3 resolved models")
            assert progress["done"] == finished, (
                f"done is {progress['done']} after {finished} models finished")
            dones.append(progress["done"])
    assert dones[-1:] == [3], f"the query's progress went {dones}, not up to 3 of 3"
    assert dones == sorted(dones), f"done went back: {dones}"
    assert [f["state"] for f in steps[0] if f["progress"] is None] == ["pending", "running", "done"]
    assert [f["state"] for f in steps[1]] == ["pending", "running", "done"]
    start = next(i for i, item in enumerate(stream) if isinstance(item, dict)
                 and item["index"] == 1 and item["state"] == "running")
    assert [item for item in stream[start:] if isinstance(item, tuple)] == [], (
        "the compare step started before the last model finished")
    assert steps[0][-1]["seq"] < stream[start]["seq"], "the compare step started before the query ended"


def _ps22_c1(monkeypatch):
    """A pipeline web_search step is one search: it carries no progress.
    Its witness in the same pipeline, a step that runs a strategy of its
    own, carries the strategy's ends as its progress."""
    world = _World(monkeypatch, [_answers(), _nests(2)])
    frames = world.go(_steps("web_search", "reasoning"))
    assert world.states(0) == ["pending", "running", "done"]
    assert [f for f in frames if f["kind"] == "exec_pipeline" and f["index"] == 0
            and f["progress"] is not None] == [], "a web_search step carried progress"
    parent = [f["progress"] for f in frames if f["kind"] == "exec_pipeline" and f["index"] == 1
              and f["progress"] is not None]
    assert parent == [{"done": 1, "total": 2, "unit": "sub_step"},
                      {"done": 2, "total": 2, "unit": "sub_step"}], (
        f"the step that ran a strategy carried {parent}")


def _ps22_c2():
    """The consensus query's units, the models, are its whole work: it
    carries progress; the compare step, one comparison, carries none."""
    steps = _by_index(_frames(_consensus(lambda model: "Sun and water.", ["a", "b"])))
    assert [f for f in steps.get(0, []) if f["progress"] is not None], (
        "the consensus query step carried no progress")
    assert [f["state"] for f in steps.get(1, [])] == ["pending", "running", "done"]
    assert [f for f in steps[1] if f["progress"] is not None] == [], (
        "the compare step carried progress")


def test_ps22_progress_only_when_its_units_are_the_whole_step(monkeypatch):
    _ps22_c1(monkeypatch)
    _ps22_c2()


def _ends_before(stream, position, frame):
    """The sub-unit ends emitted before ``stream[position]``: for a
    consensus query the models reported, for any other step the terminal
    frames of the runs nested under it."""
    before = stream[:position]
    if frame["kind"] == "consensus":
        return sum(1 for item in before if isinstance(item, tuple))
    step = {"run": frame["run"], "index": frame["index"]}
    return sum(1 for item in before if isinstance(item, dict)
               and item["parent"] == step and item["state"] in _TERMINAL)


def _ps12_c1(monkeypatch):
    """done equals the sub-unit ends before it, in every fixture stream;
    each stream shows the probe at least two progress frames."""
    world = _World(monkeypatch, [_nests(3), _answers()])
    world.go(_steps("reasoning", "direct"))
    streams = {
        "a nested run": world.frames,
        "a consensus": _consensus(lambda model: "Sun.", ["a", "b", "c"]),
        "a consensus with a failed model": _consensus(
            lambda model: RuntimeError("model gone") if model == "a" else "Sun.", ["a", "b"]),
    }
    for name, stream in streams.items():
        checked = 0
        for position, item in enumerate(stream):
            if isinstance(item, dict) and item["progress"] is not None:
                ends = _ends_before(stream, position, item)
                assert item["progress"]["done"] == ends, (
                    f"{name}: done is {item['progress']['done']} after {ends} ends")
                checked += 1
        assert checked >= 2, f"{name}: the probe saw {checked} progress frames"


_CLOCKS = frozenset({
    "time", "time_ns", "monotonic", "monotonic_ns", "perf_counter",
    "perf_counter_ns", "process_time", "now", "utcnow", "today",
})


def _function(path, qualname):
    """The definition of ``qualname`` (``Class.method`` or ``function``)."""
    scope = ast.parse(path.read_text(encoding="utf-8"))
    for part in qualname.split("."):
        found = [node for node in scope.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == part]
        assert len(found) == 1, f"{path.name} defines {len(found)} {part!r}"
        scope = found[0]
    return scope


def _clock_calls(node):
    """The calls on a clock in a definition: ``time.monotonic()``,
    ``datetime.now()`` and their kin, as source text."""
    calls = []
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _CLOCKS:
                calls.append(ast.unparse(call))
    return calls


def _ps12_c2():
    """No function on the progress path calls a clock; the probe finds the
    recorder's own duration clock in ``running`` and ``_finish``."""
    path = {
        _PKG / "pipeline_step.py": ("_progress", "StepRecorder.progress",
                                    "StepRecorder._update_progress", "StepRecorder._feed_parent"),
        _PKG / "agentic_executor.py": ("_consensus_progress",),
    }
    for source_path, names in path.items():
        for name in names:
            calls = _clock_calls(_function(source_path, name))
            assert calls == [], f"{name} reads a clock: {calls}"
    for witness in ("StepRecorder.running", "StepRecorder._finish"):
        assert _clock_calls(_function(_PKG / "pipeline_step.py", witness)), (
            f"the probe found no clock in {witness}, which reads one")


def test_ps12_progress_counts_the_ends_and_reads_no_clock(monkeypatch):
    _ps12_c1(monkeypatch)
    _ps12_c2()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
