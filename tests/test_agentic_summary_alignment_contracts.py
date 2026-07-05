#!/usr/bin/env python3
"""Contracts aligning the agentic summary with the corrective hints.

The ReAct loop's verification pass injects a corrective observation
when the last artifact-producing call failed, and runs one more
iteration -- that is what actually played on the field. The agentic
summary line reported ``verifications=`` from a DIFFERENT subsystem
(the code-verification engine), so a run whose hint fired twice logged
``verifications=0``: misleading instrumentation. These contracts pin
the aligned chain:

  * Contract 1 -- the tool loop counts its injections and the count
    travels into the returned result: a run whose artifact call failed
    reports ``verification_hints == 1`` (single corrective iteration by
    design) and still threads the final answer; a clean run reports 0.
  * Contract 2 -- the result schema carries the count additively:
    default 0 (every existing construction stays valid), explicit
    values round-trip.
  * Contract 3 -- the agentic summary tells both truths: after a tools
    run whose result carries hints, the summary line names
    ``correction_hints=N`` next to the existing ``verifications=``
    field (the grep-stable runbook chain is untouched), the public
    accessor exposes the same number, and a subsequent non-tools run
    reports 0 again (the entry reset -- a stale count never leaks into
    the next summary).

Local-only (the public distribution ships no tests). Runs under pytest or
the __main__ runner. Two isolated loads: the tool executor with the
robust-suite stand-ins (stubbed ollama, fake registry, scripted
decisions), and the agentic executor under a bare stub package with
fully injected fakes.
"""

import importlib.util
import logging
import sys
import traceback
import types
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Isolated loading -- tool executor (robust-suite idiom)
# ---------------------------------------------------------------------------
def _load_tool_executor():
    keys = (
        "ollama", "opti_oignon", "opti_oignon.tool_calling",
        "opti_oignon.tool_registry", "opti_oignon.structured_output",
        "opti_oignon.tool_executor",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **kw: None
    sys.modules["ollama"] = ollama_stub

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec_tc = importlib.util.spec_from_file_location(
        "opti_oignon.tool_calling", _OO / "tool_calling.py",
    )
    tc = importlib.util.module_from_spec(spec_tc)
    sys.modules["opti_oignon.tool_calling"] = tc
    spec_tc.loader.exec_module(tc)

    reg = types.ModuleType("opti_oignon.tool_registry")
    reg.ToolRegistry = object
    reg.tool_registry = None
    sys.modules["opti_oignon.tool_registry"] = reg

    so = types.ModuleType("opti_oignon.structured_output")
    so.StructuredOutputEngine = object
    so.structured_engine = None
    so.STRUCTURED_OUTPUT_AVAILABLE = False
    sys.modules["opti_oignon.structured_output"] = so

    spec_te = importlib.util.spec_from_file_location(
        "opti_oignon.tool_executor", _OO / "tool_executor.py",
    )
    te = importlib.util.module_from_spec(spec_te)
    sys.modules["opti_oignon.tool_executor"] = te
    spec_te.loader.exec_module(te)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return te, restore


class _FakeRegistry:
    def list_available(self):
        return []

    def get(self, name):
        return None

    def get_tools_prompt(self):
        return "tools: execute_code"


def _scripted_executor(te, decisions: list, call_result_text: str):
    """A real ToolExecutor with scripted decisions and canned tool runs."""
    ex = te.ToolExecutor(
        registry=_FakeRegistry(), structured_engine=None,
        default_model="contract-model",
    )
    queue = list(decisions)

    def fake_decide(message, _model, context_messages, tool_results_context,
                    native_transcript=None):
        return queue.pop(0) if queue else []

    def fake_execute(tool_name, arguments, reasoning, approval_fn=None):
        return te.ToolCallResult(
            tool_name=tool_name,
            arguments=dict(arguments),
            result=call_result_text,
            success=True,
        )

    ex._decide_tools = fake_decide
    ex._execute_tool = fake_execute
    ex._generate_final_response = lambda *a, **kw: "final answer"
    return ex


# ---------------------------------------------------------------------------
# Contract 1 -- the loop counts injections and threads them into the result
# ---------------------------------------------------------------------------
def test_c1_loop_counts_hint_injections_into_the_result():
    te, restore = _load_tool_executor()
    try:
        failing = _scripted_executor(
            te,
            decisions=[[("execute_code", {"code": "boom"})], [], []],
            call_result_text="Execution Failed (return code: 1)",
        )
        result = failing.execute_with_tools(message="run it", model="m")
        got = getattr(result, "verification_hints", None)
        assert got == 1, (
            f"one corrective injection must be counted into the result: {got!r}"
        )
        assert result.response == "final answer", result.response
        assert len(result.tool_calls) == 1

        clean = _scripted_executor(
            te,
            decisions=[[("execute_code", {"code": "ok"})], []],
            call_result_text="Command success (return code: 0)",
        )
        result = clean.execute_with_tools(message="run it", model="m")
        assert getattr(result, "verification_hints", None) == 0, (
            f"a clean run must report zero injections: {result!r}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- additive schema: default 0, explicit round-trip
# ---------------------------------------------------------------------------
def test_c2_result_schema_carries_the_count_additively():
    te, restore = _load_tool_executor()
    try:
        bare = te.ToolExecutionResult()
        assert getattr(bare, "verification_hints", None) == 0, (
            "the field must default to 0 so every existing construction "
            f"site stays valid: {bare!r}"
        )
        explicit = te.ToolExecutionResult(verification_hints=3)
        assert explicit.verification_hints == 3
    finally:
        restore()


# ---------------------------------------------------------------------------
# Isolated loading -- agentic executor (bare stub package, injected fakes)
# ---------------------------------------------------------------------------
def _load_agentic():
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "opti_oignon" or k.startswith("opti_oignon.")
    }
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.agentic_executor", _OO / "agentic_executor.py",
    )
    ag = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.agentic_executor"] = ag
    spec.loader.exec_module(ag)
    pkg.agentic_executor = ag
    # The availability property consults the module import flag as well as
    # the injected instance; under the bare stub package the flag resolves
    # False and every tools run would silently fall back to direct. Force
    # it so the injected fake tool executor is honored.
    ag.TOOL_EXECUTOR_AVAILABLE = True

    def restore():
        for key in list(sys.modules):
            if key == "opti_oignon" or key.startswith("opti_oignon."):
                sys.modules.pop(key, None)
        sys.modules.update(saved)

    return ag, restore


class _FakeBaseExecutor:
    last_verification_results: list = []

    def execute(self, **kwargs):
        yield "direct-out"


class _FakeToolExecutorResult:
    def __init__(self, hints: int):
        self.response = "tools-out"
        self.tool_calls: list[Any] = []
        self.verification_hints = hints
        self.model = "contract-model"
        self.total_time = 0.1


class _FakeToolExecutor:
    """Deliberately has no stream_with_tools: the non-stream path runs."""

    def __init__(self, hints: int):
        self._hints = hints
        self.calls = 0

    def execute_with_tools(self, **kwargs):
        self.calls += 1
        return _FakeToolExecutorResult(self._hints)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _summaries(handler: _ListHandler) -> list[str]:
    return [
        m for m in handler.messages
        if m.startswith("AgenticExecutor: finished")
    ]


# ---------------------------------------------------------------------------
# Contract 3 -- the summary tells both truths; the entry reset holds
# ---------------------------------------------------------------------------
def test_c3_summary_reports_hints_and_resets_between_runs():
    ag, restore = _load_agentic()
    handler = _ListHandler()
    original_select = ag._select_pipeline
    try:
        ag.logger.addHandler(handler)
        ag.logger.setLevel(logging.INFO)

        executor = ag.AgenticExecutor(
            executor=_FakeBaseExecutor(),
            tool_executor=_FakeToolExecutor(hints=2),
            default_model="contract-model",
        )
        routing = types.SimpleNamespace(
            model="contract-model", task_type="general",
            temperature=0.7, prompt_variant="default",
        )

        # Run 1: a tools run whose result carries two injections.
        ag._select_pipeline = lambda **kw: ag.PIPELINE_TOOLS
        out = "".join(
            c for c in executor.execute("do the tools", routing)
            if isinstance(c, str)
        )
        assert "tools-out" in out, out
        summaries = _summaries(handler)
        assert summaries, f"no summary line captured: {handler.messages}"
        line = summaries[-1]
        assert "pipeline=tools" in line, line
        assert "verifications=0" in line, (
            f"the code-verification field must stay grep-stable: {line}"
        )
        assert "correction_hints=2" in line, (
            f"the summary must name the injections that played: {line}"
        )
        assert executor.last_verification_hints == 2

        # Run 2: a non-tools run right after -- the entry reset must hold,
        # a stale count never leaks into the next summary.
        ag._select_pipeline = lambda **kw: ag.PIPELINE_DIRECT
        out = "".join(
            c for c in executor.execute("just answer", routing)
            if isinstance(c, str)
        )
        assert "direct-out" in out, out
        line = _summaries(handler)[-1]
        assert "pipeline=direct" in line, line
        assert "correction_hints=0" in line, (
            f"a stale hint count leaked across runs: {line}"
        )
        assert executor.last_verification_hints == 0
    finally:
        ag._select_pipeline = original_select
        ag.logger.removeHandler(handler)
        restore()


# ---------------------------------------------------------------------------
# Runner (pytest picks up the test_ functions; direct execution works too)
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    names = sorted(n for n in globals() if n.startswith("test_"))
    selected = [
        n for n in names if not argv or any(fragment in n for fragment in argv)
    ]
    failures = 0
    for name in selected:
        try:
            globals()[name]()
        except Exception as exc:
            failures += 1
            print(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
