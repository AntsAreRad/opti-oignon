#!/usr/bin/env python3
"""Integration tests for PipelineRunner.execute (pipelines.py).

execute() is the streaming, executor-backed driver that runs a pipeline step by
step (it is what routes_chat invokes when a pipeline is selected). It is covered
here with a fake AgenticExecutor so the orchestration logic can be asserted
without a real model:

  * step iteration + the ("pipeline_step_start"/"pipeline_step_end", ...) tuple
    protocol interleaved with string chunks;
  * step_type -> executor flags (think / web_search / consensus / self_correct);
  * explicit model_override;
  * step conditions (skip when false);
  * previous-output chaining (PIP-01) and last-good-context-on-failure (PIP-02);
  * the status/step callbacks;
  * the error paths: no executor, empty pipeline, a failing step, and a
    mid-run emergency stop.

An autouse fixture neutralizes the module-level emergency-stop and resource-
governor resolvers so execution is deterministic; individual tests re-point the
emergency-stop resolver when they exercise that path.
"""

import pytest

import opti_oignon.pipelines as P
from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep, PipelineRunner


class FakeExecutor:
    """Records each call and yields one deterministic chunk per step."""

    def __init__(self, fail_on=None):
        self.calls = []
        self._fail_on = fail_on  # 0-based step index that should raise

    def execute(
        self, *, message, routing, conversation_id=None, think=None,
        web_search=None, consensus=None, consensus_models=None,
        consensus_strategy=None, self_correct=None, on_status=None,
        on_tool_call=None, on_reasoning_step=None, on_consensus_model=None,
        on_correction_step=None, approval_fn=None,
    ):
        idx = len(self.calls)
        self.calls.append({
            "message": message,
            "model": getattr(routing, "model", None),
            "think": think,
            "web_search": web_search,
            "consensus": consensus,
            "self_correct": self_correct,
        })
        if self._fail_on is not None and idx == self._fail_on:
            raise RuntimeError("executor boom")
        yield f"out[{getattr(routing, 'model', '?')}]"


class FakeRouting:
    def __init__(self, model="base"):
        self.model = model


class FakeRouter:
    enabled = False  # disables smart-routing overrides

    def override_routing(self, routing, step_type):
        return routing


def _runner(executor):
    return PipelineRunner(agentic_executor=executor, smart_router=FakeRouter())


def _pipe(steps):
    return ExecutionPipeline(id="p", name="P", steps=steps)


def _types(events):
    return [e[0] if isinstance(e, tuple) else "str" for e in events]


@pytest.fixture(autouse=True)
def _neutralize_resolvers(monkeypatch):
    monkeypatch.setattr(P, "_resolve_emergency_stop", lambda: None)
    monkeypatch.setattr(P, "_resolve_resource_governor", lambda: None)


# ---------------------------------------------------------------------------
# Iteration + tuple protocol
# ---------------------------------------------------------------------------

def test_execute_iterates_steps_and_yields_boundaries():
    ex = FakeExecutor()
    events = list(_runner(ex).execute(
        _pipe([ExecutionStep("direct", label="A"), ExecutionStep("direct", label="B")]),
        "hello", FakeRouting(),
    ))
    assert _types(events) == [
        "pipeline_step_start", "str", "pipeline_step_end",
        "pipeline_step_start", "str", "pipeline_step_end",
    ]
    assert len(ex.calls) == 2
    starts = [e for e in events if isinstance(e, tuple) and e[0] == "pipeline_step_start"]
    assert starts[0][1] == 0 and isinstance(starts[0][2], ExecutionStep)
    ends = [e for e in events if isinstance(e, tuple) and e[0] == "pipeline_step_end"]
    assert ends[0][2] == "out[base]"          # the step output is captured


def test_execute_collects_string_chunks():
    ex = FakeExecutor()
    events = list(_runner(ex).execute(_pipe([ExecutionStep("direct")]), "x", FakeRouting()))
    assert [e for e in events if isinstance(e, str)] == ["out[base]"]


# ---------------------------------------------------------------------------
# step_type -> executor flags + model override
# ---------------------------------------------------------------------------

def test_execute_applies_model_override():
    ex = FakeExecutor()
    list(_runner(ex).execute(
        _pipe([ExecutionStep("direct", model_override="big")]), "x", FakeRouting("base"),
    ))
    assert ex.calls[0]["model"] == "big"


def test_execute_think_step_sets_think_flag():
    ex = FakeExecutor()
    list(_runner(ex).execute(
        _pipe([ExecutionStep("direct"), ExecutionStep("think")]), "x", FakeRouting(),
    ))
    assert ex.calls[0]["think"] is None       # direct step
    assert ex.calls[1]["think"] is True       # think step


def test_execute_web_search_step_sets_flag():
    ex = FakeExecutor()
    list(_runner(ex).execute(_pipe([ExecutionStep("web_search")]), "x", FakeRouting()))
    assert ex.calls[0]["web_search"] is True


def test_execute_consensus_and_self_correct_flags():
    ex = FakeExecutor()
    list(_runner(ex).execute(
        _pipe([ExecutionStep("consensus"), ExecutionStep("self_correct")]), "x", FakeRouting(),
    ))
    assert ex.calls[0]["consensus"] is True
    assert ex.calls[1]["self_correct"] is True


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def test_execute_skips_step_with_false_condition():
    ex = FakeExecutor()
    events = list(_runner(ex).execute(
        _pipe([
            ExecutionStep("direct", label="A"),
            ExecutionStep("direct", label="B", condition="if_long_input"),
        ]),
        "short", FakeRouting(),                # input < 500 chars -> B is skipped
    ))
    assert len(ex.calls) == 1
    assert _types(events).count("pipeline_step_start") == 1


def test_execute_runs_step_with_true_condition():
    ex = FakeExecutor()
    list(_runner(ex).execute(
        _pipe([ExecutionStep("direct", condition="always")]), "x", FakeRouting(),
    ))
    assert len(ex.calls) == 1


# ---------------------------------------------------------------------------
# Previous-output chaining (PIP-01) + failure isolation (PIP-02)
# ---------------------------------------------------------------------------

def test_execute_passes_previous_output_when_enabled():
    ex = FakeExecutor()
    list(_runner(ex).execute(
        _pipe([ExecutionStep("direct", label="A"), ExecutionStep("direct", label="B")]),
        "original question", FakeRouting(),
    ))
    msg = ex.calls[1]["message"]
    assert "previous analysis" in msg.lower()
    assert "out[base]" in msg                  # step A's output is embedded


def test_execute_ignores_previous_output_when_disabled():
    ex = FakeExecutor()
    list(_runner(ex).execute(
        _pipe([
            ExecutionStep("direct", label="A"),
            ExecutionStep("direct", label="B", pass_previous_output=False),
        ]),
        "the original message", FakeRouting(),
    ))
    assert ex.calls[1]["message"] == "the original message"


def test_execute_isolates_failing_step():
    # Step 0 fails; step 1 must still run and must NOT receive the error text
    # as "previous analysis" -- it keeps the last good context (PIP-02).
    ex = FakeExecutor(fail_on=0)
    events = list(_runner(ex).execute(
        _pipe([ExecutionStep("direct", label="A"), ExecutionStep("direct", label="B")]),
        "the original", FakeRouting(),
    ))
    assert any(isinstance(e, str) and "Step 1 failed" in e for e in events)
    assert len(ex.calls) == 2                       # step B still ran
    assert ex.calls[1]["message"] == "the original"  # not the error text


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def test_execute_fires_step_callbacks():
    ex = FakeExecutor()
    starts, ends, statuses = [], [], []
    list(_runner(ex).execute(
        _pipe([ExecutionStep("direct", label="A")]),
        "x", FakeRouting(),
        on_step_start=lambda i, s: starts.append(i),
        on_step_end=lambda i, s, out: ends.append((i, out)),
        on_status=lambda msg: statuses.append(msg),
    ))
    assert starts == [0]
    assert ends == [(0, "out[base]")]
    assert any("Step 1" in s for s in statuses)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_execute_no_executor_yields_error(monkeypatch):
    runner = _runner(FakeExecutor())
    monkeypatch.setattr(runner, "_get_executor", lambda: None)
    events = list(runner.execute(_pipe([ExecutionStep("direct")]), "x", FakeRouting()))
    assert events == ["[ERR] AgenticExecutor not available"]


def test_execute_empty_steps_yields_error():
    events = list(_runner(FakeExecutor()).execute(_pipe([]), "x", FakeRouting()))
    assert events == ["[ERR] Pipeline has no steps"]


def test_execute_emergency_stop_aborts(monkeypatch):
    class _Stopped:
        def is_stopped(self):
            return True

    monkeypatch.setattr(P, "_resolve_emergency_stop", lambda: _Stopped())
    ex = FakeExecutor()
    events = list(_runner(ex).execute(_pipe([ExecutionStep("direct")]), "x", FakeRouting()))
    assert any(isinstance(e, str) and "emergency stop" in e.lower() for e in events)
    assert len(ex.calls) == 0                       # nothing executed
