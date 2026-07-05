#!/usr/bin/env python3
"""Contracts for the chat tool-loop eval harness.

The harness drives the ToolExecutor chat surface with a scripted
deterministic backend and a fixture-backed registry, scores runs with
built-in checkers, and writes JSONL traces. These contracts pin the
properties that make its numbers trustworthy:

  * Contract 1 -- determinism: two runs of the same suite produce
    identical report rows and identical trace task rows (wall-clock
    fields live only on the run header and footer rows).
  * Contract 2 -- the checkers are actually wired into the verdict: a
    task whose scripted answer misattributes the tool actions to the
    user and quotes a runtime scaffold line FAILS with those checkers
    named (and non-empty details), while a clean task passes.
  * Contract 3 -- measurement and ordering on the streaming front: the
    recorded backend calls separate decision rounds from the streamed
    final generation, decision input size grows strictly as results
    accumulate, and every tool event fires before the first answer
    chunk.
  * Contract 4 -- trace schema: every row is valid JSON in the
    kind/round/data shape, kinds stay within the published set, the
    file is framed by run_start/run_end, and task_end rows carry the
    verdict keys consumers rely on.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. The harness and the chat surface load in
isolation under their canonical dotted names, so relative and package
imports resolve against the loaded copies; when pydantic is absent a
minimal attribute-bag stand-in is installed (the real package, when
installed, is left untouched).
"""

import importlib.util
import json
import sys
import tempfile
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Minimal pydantic stand-in (only when the real package is absent)
# ---------------------------------------------------------------------------
def _pydantic_shim() -> types.ModuleType:
    mod = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for name in getattr(self.__class__, "__annotations__", {}):
                default = getattr(self.__class__, name, None)
                if isinstance(default, (list, dict)):
                    default = type(default)(default)
                setattr(self, name, default)
            for key, value in kwargs.items():
                setattr(self, key, value)

    mod.BaseModel = BaseModel
    return mod


# ---------------------------------------------------------------------------
# Isolated loading: the real chat surface and the real harness under their
# canonical dotted names, heavy seams stubbed. Saves/restores sys.modules
# for sibling suites.
# ---------------------------------------------------------------------------
def _load(runner_path: Path | None = None):
    keys = (
        "pydantic", "ollama", "opti_oignon", "opti_oignon.tool_calling",
        "opti_oignon.tool_registry", "opti_oignon.structured_output",
        "opti_oignon.response_hygiene", "opti_oignon.tool_executor",
        "opti_oignon.config", "opti_oignon.agent_eval",
        "opti_oignon.agent_eval.tasks", "opti_oignon.agent_eval.chat_runner",
    )
    saved = {k: sys.modules.get(k) for k in keys}

    try:
        import pydantic  # noqa: F401
    except ImportError:
        sys.modules["pydantic"] = _pydantic_shim()

    ollama_stub = types.ModuleType("ollama")
    ollama_stub.chat = lambda **kw: None
    sys.modules["ollama"] = ollama_stub

    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    def _real(dotted: str, path: Path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    pkg.tool_calling = _real(
        "opti_oignon.tool_calling", _OO / "tool_calling.py",
    )
    pkg.tool_registry = _real(
        "opti_oignon.tool_registry", _OO / "tool_registry.py",
    )

    so = types.ModuleType("opti_oignon.structured_output")
    so.StructuredOutputEngine = object
    so.ToolCallRequest = object
    so.structured_output_engine = None
    so.STRUCTURED_OUTPUT_AVAILABLE = False
    sys.modules["opti_oignon.structured_output"] = so
    pkg.structured_output = so

    pkg.response_hygiene = _real(
        "opti_oignon.response_hygiene", _OO / "response_hygiene.py",
    )
    te = _real("opti_oignon.tool_executor", _OO / "tool_executor.py")
    pkg.tool_executor = te

    ae = types.ModuleType("opti_oignon.agent_eval")
    ae.__path__ = []
    sys.modules["opti_oignon.agent_eval"] = ae
    pkg.agent_eval = ae
    ae.tasks = _real(
        "opti_oignon.agent_eval.tasks", _OO / "agent_eval" / "tasks.py",
    )
    target = runner_path or (_OO / "agent_eval" / "chat_runner.py")
    cr = _real("opti_oignon.agent_eval.chat_runner", target)
    ae.chat_runner = cr

    if not cr.FEATURE_AVAILABLE:
        raise RuntimeError("harness reports FEATURE_AVAILABLE False")

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return cr, te, restore


# ---------------------------------------------------------------------------
# Suite material (temp YAML files; the loader accepts filesystem paths)
# ---------------------------------------------------------------------------
FINAL_MULTI = (
    "I listed the workspace, read status.txt, and wrote a one-line summary "
    "to summary.txt with my file tools. Everything is nominal with a queue "
    "depth of three."
)
FINAL_DIRECT = (
    "Sunlight scatters off air molecules, and the shorter blue wavelengths "
    "scatter the most, so the sky looks blue."
)

_A1_BLOCK = """\
  - id: a1-multi
    title: Multi-step chain
    prompt: >
      List the workspace files, read status.txt, and write a one-line
      summary to summary.txt.
    fixture:
      status.txt: "All services nominal.\\nQueue depth: 3.\\n"
    checks:
      - "expect_tool:write_file"
      - "expect_file:summary.txt"
      - "max_calls:3"
      - "final_nonempty"
      - "no_misattribution"
      - "no_internal_markers"
      - "tools_before_stream"
    script:
      - tool_calls:
          - name: list_files
            arguments: {}
      - tool_calls:
          - name: read_file
            arguments: {filename: status.txt}
      - tool_calls:
          - name: write_file
            arguments:
              filename: summary.txt
              content: "Services nominal; queue depth 3.\\n"
      - content: ""
      - content: "__FINAL_MULTI__"
"""

_A2_BLOCK = """\
  - id: a2-direct
    title: Direct answer
    prompt: In one sentence, explain why the sky is blue.
    checks:
      - "expect_no_tools"
      - "final_nonempty"
      - "no_misattribution"
    script:
      - content: ""
      - content: "__FINAL_DIRECT__"
"""

SUITE_MULTI = (
    "suite: harness-multi\n\ntasks:\n" + _A1_BLOCK + _A2_BLOCK
).replace("__FINAL_MULTI__", FINAL_MULTI).replace(
    "__FINAL_DIRECT__", FINAL_DIRECT,
)

SUITE_STREAM = (
    "suite: harness-stream\n\ntasks:\n" + _A1_BLOCK
).replace("__FINAL_MULTI__", FINAL_MULTI)

SUITE_TRAP = """\
suite: harness-trap

tasks:
  - id: b1-clean
    title: Clean single read
    prompt: Read notes.txt and report it.
    fixture:
      notes.txt: "alpha\\n"
    checks:
      - "expect_tool:read_file"
      - "final_nonempty"
      - "no_misattribution"
      - "no_internal_markers"
    script:
      - tool_calls:
          - name: read_file
            arguments: {filename: notes.txt}
      - content: ""
      - content: >
          I read notes.txt with my read_file tool; it contains the word
          alpha.

  - id: b2-trap
    title: Leaky misattributed answer
    prompt: Write done to report.txt and confirm.
    checks:
      - "no_misattribution"
      - "no_internal_markers"
      - "final_nonempty"
    script:
      - tool_calls:
          - name: write_file
            arguments: {filename: report.txt, content: "done\\n"}
      - content: ""
      - content: |
          You created the file report.txt yourself.
          ```
          [environment] tool call by assistant: write_file
          ```
          All done.
"""


def _suite_file(directory: str, name: str, text: str) -> str:
    path = Path(directory) / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _read_trace(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _task_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["kind"] not in ("run_start", "run_end")]


# ---------------------------------------------------------------------------
# Contract 1 -- determinism of report rows and trace task rows
# ---------------------------------------------------------------------------
def test_c1_determinism():
    cr, _te, restore = _load()
    try:
        base = tempfile.mkdtemp(prefix="chat-eval-c1-")
        suite = _suite_file(base, "multi", SUITE_MULTI)
        first = cr.run_suite(suite, trace_dir=str(Path(base) / "t1"))
        second = cr.run_suite(suite, trace_dir=str(Path(base) / "t2"))

        rows_first = [record.to_row() for record in first.records]
        rows_second = [record.to_row() for record in second.records]
        assert rows_first == rows_second, (
            "report rows differ between two identical runs"
        )
        assert first.passed and second.passed, rows_first

        trace_first = _task_rows(_read_trace(first.trace_path))
        trace_second = _task_rows(_read_trace(second.trace_path))
        assert trace_first, "trace has no task rows"
        assert trace_first == trace_second, (
            "trace task rows differ between two identical runs"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- checkers are wired into the verdict, with named failures
# ---------------------------------------------------------------------------
def test_c2_checkers_wired():
    cr, _te, restore = _load()
    try:
        base = tempfile.mkdtemp(prefix="chat-eval-c2-")
        suite = _suite_file(base, "trap", SUITE_TRAP)
        report = cr.run_suite(
            suite, fronts=("execute",), trace_dir=base,
        )
        by_id = {record.task_id: record for record in report.records}

        clean = by_id["b1-clean"]
        assert clean.passed, [c.to_dict() for c in clean.checks]

        trap = by_id["b2-trap"]
        assert not trap.passed, "trap task must fail its hygiene checks"
        failing = {c.check for c in trap.checks if not c.ok}
        assert "no_misattribution" in failing, failing
        assert "no_internal_markers" in failing, failing
        for outcome in trap.checks:
            if not outcome.ok:
                assert outcome.detail, (
                    f"failing check {outcome.check} carries no detail"
                )
        ok_map = {c.check: c.ok for c in trap.checks}
        assert ok_map["final_nonempty"], ok_map
        assert ok_map["script_fidelity"], ok_map
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- streaming-front measurement and event ordering
# ---------------------------------------------------------------------------
def test_c3_measurement_and_order():
    cr, _te, restore = _load()
    try:
        base = tempfile.mkdtemp(prefix="chat-eval-c3-")
        suite = _suite_file(base, "stream", SUITE_STREAM)
        report = cr.run_suite(suite, fronts=("stream",), trace_dir=base)
        assert len(report.records) == 1
        record = report.records[0]

        calls = record.model_calls
        assert len(calls) == 5, [
            (c["index"], c["stream"], c["tools_param"]) for c in calls
        ]
        for call in calls[:4]:
            assert call["stream"] is False, call
            assert call["tools_param"] is True, call
        assert calls[4]["stream"] is True, calls[4]
        assert calls[4]["tools_param"] is False, calls[4]

        growth = [c["chars_in"] for c in calls[:4]]
        assert all(b > a for a, b in zip(growth, growth[1:])), growth

        kinds = [event[0] for event in record.events]
        assert kinds.count("tool_call") == 3, record.events
        assert "chunk" in kinds, record.events
        first_chunk = kinds.index("chunk")
        assert "tool_call" not in kinds[first_chunk:], record.events
        assert kinds.count("chunk") >= 2, record.events

        ok_map = {c.check: c.ok for c in record.checks}
        assert ok_map["tools_before_stream"], record.checks
        assert ok_map["script_fidelity"], record.checks
        assert record.passed, [c.to_dict() for c in record.checks]
        assert record.final_text == FINAL_MULTI, record.final_text
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- trace rows are valid JSON in the published schema
# ---------------------------------------------------------------------------
def test_c4_trace_schema():
    cr, _te, restore = _load()
    try:
        base = tempfile.mkdtemp(prefix="chat-eval-c4-")
        suite = _suite_file(base, "multi", SUITE_MULTI)
        report = cr.run_suite(suite, trace_dir=base)
        rows = _read_trace(report.trace_path)

        assert rows, "trace file is empty"
        assert rows[0]["kind"] == "run_start", rows[0]
        assert rows[-1]["kind"] == "run_end", rows[-1]
        allowed = set(cr.TRACE_KINDS)
        for row in rows:
            assert set(row) == {"kind", "round", "data"}, row
            assert row["kind"] in allowed, row
            assert isinstance(row["round"], int), row
            assert isinstance(row["data"], dict), row

        ends = [row for row in rows if row["kind"] == "task_end"]
        assert len(ends) == 4, len(ends)  # two tasks on two fronts
        for row in ends:
            data = row["data"]
            for key in ("task", "front", "passed", "checks"):
                assert key in data, data
            assert isinstance(data["checks"], list) and data["checks"]
            for check in data["checks"]:
                assert {"check", "ok"} <= set(check), check

        model_calls = [row for row in rows if row["kind"] == "model_call"]
        assert model_calls, "no model_call rows in the trace"
        for row in model_calls:
            assert isinstance(row["data"]["index"], int), row
            assert isinstance(row["data"]["chars_in"], int), row

        assert rows[-1]["data"]["passed"] is True, rows[-1]
    finally:
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
