#!/usr/bin/env python3
"""
Chat tool-loop eval harness.

Drives the ToolExecutor chat surface end to end -- the exact ReAct loop the
chat routes use -- with a scripted deterministic model client and a fake,
fixture-backed tool registry. Runs are fully deterministic and offline: no
model process, no network, no real filesystem. The registry handlers close
over an in-memory workspace seeded from the task fixture, so nothing an
eval run does can touch host state.

This module complements the sandboxed agent-surface harness (runner.py):
that one measures a real model driving the agent loop inside a disposable
sandbox and is host territory by design. This one measures the CHAT
pipeline's own mechanics -- decision framing, live tool progress, streaming
hygiene, retry feedback, context growth across decision rounds -- and is
provable anywhere because the model is a script.

Task format: the standard TaskSpec suites (tasks.py) are reused verbatim
for id/title/prompt/fixture/checks validation, extended per task by one
additive key the standard loader ignores::

    script:                     # ordered model turns, one per chat call
      - tool_calls:             # a decision turn returning native calls
          - name: read_file
            arguments: {filename: notes.txt}
      - content: ""             # a decision turn with no calls (stop)
      - content: |              # the LAST turn is the user-facing answer
          The final answer text.

The last turn must be a non-empty text answer with no tool calls; every
call to the backend consumes exactly one turn, and any mismatch (leftover
turns, calls past the end) fails the task through the always-on
script_fidelity check instead of being silently absorbed.

Checks are names of built-in checkers, not sandbox commands:

    expect_tool:<name>      a successful call to <name> happened
    expect_no_tools         no tool call happened at all
    expect_file:<relpath>   <relpath> exists in the workspace after the run
    max_calls:<n>           at most <n> tool calls (failed ones included)
    final_nonempty          the final answer has visible content
    no_misattribution       no second-person action claim in the answer
    no_internal_markers     no runtime scaffold line in the answer
    tools_before_stream     on the streaming front, every tool event fired
                            before the first answer chunk

TaskSpec fields that only make sense with a sandbox (timeout_s,
requested_ctx) are accepted and inert here; max_rounds caps the executor's
tool budget.

Traces: one JSONL file per run, rows in the agent loop's observable-event
shape (kind / round / data; harness rows use round 0 and carry an explicit
index where ordering matters). Wall-clock fields appear only on the
run_start and run_end rows, so two runs of the same suite produce
byte-identical task rows -- determinism is checkable by diff.
"""

import argparse
import copy
import json
import logging
import threading
import time
import types
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module conventions (project-wide).
checkpoint_before_apply = True

try:
    import yaml

    _YAML_OK = True
except ImportError:  # pragma: no cover - PyYAML is a core dependency
    yaml = None  # type: ignore[assignment]
    _YAML_OK = False

# The chat surface under evaluation and the pieces the fake registry, the
# suite loader and the checkers build on. Guarded so a partial build
# degrades to FEATURE_AVAILABLE False instead of an import error.
try:
    from opti_oignon import tool_executor as _texec_mod
    from opti_oignon.agent_eval.tasks import (
        _coerce_task,
        resolve_suite_path,
    )
    from opti_oignon.response_hygiene import detect_misattribution
    from opti_oignon.tool_executor import ToolExecutor
    from opti_oignon.tool_registry import (
        ToolDefinition,
        ToolParam,
        ToolRegistry,
    )

    _CHAT_SURFACE_OK = True
except Exception:  # pragma: no cover - partial build degradation
    _texec_mod = None  # type: ignore[assignment]
    _coerce_task = None  # type: ignore[assignment]
    resolve_suite_path = None  # type: ignore[assignment]
    detect_misattribution = None  # type: ignore[assignment]
    ToolExecutor = None  # type: ignore[assignment]
    ToolDefinition = ToolParam = ToolRegistry = None  # type: ignore
    _CHAT_SURFACE_OK = False

FEATURE_AVAILABLE = _CHAT_SURFACE_OK and _YAML_OK

# The scripted backend advertises a model name inside a native
# function-calling family, so the decision path under measurement is the
# production one (native tool schemas), not the format= fallback.
SCRIPTED_MODEL = "qwen3-scripted"

# The counterpart: a model name outside every native function-calling
# family, so an executor built with it routes every decision through the
# structured-output engine -- the format= fallback path. Pair it with
# ScriptedDecisionEngine to script and measure that path deterministically.
SCRIPTED_FALLBACK_MODEL = "scripted-fallback"

# Streaming turns are split into fixed-size chunks so incremental consumers
# (the marker filter, the chunk relay) are exercised across boundaries.
STREAM_CHUNK_CHARS = 12

# The two execution fronts of the chat surface.
FRONTS = ("execute", "stream")

# Default location for run traces (runtime data, never shipped).
TRACE_DIR_DEFAULT = Path(__file__).parent.parent / "data" / "chat_eval_traces"

# Row kinds a trace file may contain (schema anchor for consumers).
TRACE_KINDS = (
    "run_start",
    "task_start",
    "model_call",
    "tool_call",
    "task_end",
    "run_end",
)

# Scaffold line prefixes that must never surface in a final answer. The
# scan is deliberately fence-agnostic: a suite answer never legitimately
# quotes runtime scaffolding, so a marker inside a code fence is still a
# leak for evaluation purposes.
_LEAK_PREFIXES = (
    "[environment]",
    "[tool:",
    "[prior tool call",
    "[verification]",
    "[untrusted",
    "[reminder",
)

# One scripted run at a time: the backend swap below rebinds module-level
# state on the executor module, mirroring how the sibling runner serializes
# its runs.
_RUN_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Script model
# ---------------------------------------------------------------------------
@dataclass
class ScriptTurn:
    """One scripted model turn: optional native calls plus message text."""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class ChatTask:
    """A validated suite entry: the standard spec plus its model script."""

    spec: Any
    script: list[ScriptTurn]


class ScriptedChatClient:
    """Deterministic stand-in for the chat backend: replays a script.

    One instance per task run. Each ``chat()`` call consumes the next turn;
    the turn's tool_calls become native function calls, its content the
    message text. ``stream=True`` splits the content into fixed-size chunks
    so incremental consumers are exercised across chunk boundaries.

    Every call is recorded (message count, input characters, role sequence,
    whether native tool schemas were offered) -- the recording IS the
    context-growth measurement, taken from outside the surface under test.
    Calls past the end of the script return an empty message and are
    counted, never invented, so a script/loop mismatch stays visible.
    """

    def __init__(self, turns: list[ScriptTurn]):
        self._turns = list(turns)
        self._next = 0
        self.calls: list[dict] = []
        self.overrun = 0

    @property
    def leftover(self) -> int:
        """Turns the run never consumed (a mismatch when non-zero)."""
        return max(0, len(self._turns) - self._next)

    def chat(
        self,
        model: str | None = None,
        messages: list[dict] | None = None,
        tools: Any = None,
        options: Any = None,
        stream: bool = False,
        **_ignored: Any,
    ):
        msgs = list(messages or [])
        self.calls.append({
            "index": len(self.calls),
            "model": str(model or ""),
            "stream": bool(stream),
            "tools_param": tools is not None,
            "messages_count": len(msgs),
            "chars_in": sum(
                len(str(m.get("content") or "")) for m in msgs
            ),
            "roles": [str(m.get("role", "")) for m in msgs],
        })
        if self._next >= len(self._turns):
            self.overrun += 1
            turn = ScriptTurn()
        else:
            turn = self._turns[self._next]
            self._next += 1
        if stream:
            return self._chunks(turn.content)
        # Non-streaming responses use the attribute form the real client
        # returns (response.message.content), which the surface reads with
        # attribute access on the final-answer path.
        native_calls = [
            types.SimpleNamespace(
                function=types.SimpleNamespace(
                    name=call["name"],
                    arguments=dict(call.get("arguments") or {}),
                ),
            )
            for call in turn.tool_calls
        ]
        return types.SimpleNamespace(
            message=types.SimpleNamespace(
                content=turn.content,
                tool_calls=native_calls,
            ),
        )

    @staticmethod
    def _chunks(text: str):
        for start in range(0, len(text), STREAM_CHUNK_CHARS):
            yield {
                "message": {
                    "content": text[start:start + STREAM_CHUNK_CHARS],
                },
            }


class _NoDecisionEngine:
    """Structured-output stand-in that always declines.

    The scripted client answers every decision through the native path, so
    the format= fallback must never be consulted; if it ever is, declining
    keeps the run deterministic and offline instead of reaching a real
    engine through the module singleton.
    """

    def generate_structured(self, **_kwargs: Any):
        return type("Result", (), {"success": False, "data": None})()


class ScriptedDecisionEngine:
    """Structured-output stand-in replaying scripted tool decisions.

    The counterpart of the declining stand-in above, for the format=
    fallback path: an executor built with a model outside the native
    function-calling families (see ``SCRIPTED_FALLBACK_MODEL``) routes
    every decision through the structured engine, so scripting the
    engine drives -- and measures -- the non-native decision path
    deterministically. Turns are ``(tool_name, arguments)`` pairs; a
    ``"none"`` turn scripts the stop. Every call is captured (deep-copied
    messages and the exact structured parameters) for assertions.
    Exhaustion answers ``"none"`` so the loop always stops, with the
    ``overrun`` counter recording the excess -- the same fidelity
    accounting as the scripted chat client.
    """

    def __init__(self, turns: list[tuple[str, dict]]):
        self._turns = list(turns)
        self._next = 0
        self.captured: list[dict] = []
        self.overrun = 0

    @property
    def leftover(self) -> int:
        return max(0, len(self._turns) - self._next)

    def generate_structured(
        self,
        messages: list[dict] | None = None,
        schema: Any = None,
        model: str | None = None,
        extra_system_prompt: str | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
        **_ignored: Any,
    ):
        self.captured.append({
            "messages": copy.deepcopy(list(messages or [])),
            "schema": schema,
            "model": str(model or ""),
            "extra_system_prompt": str(extra_system_prompt or ""),
            "temperature": temperature,
            "max_retries": max_retries,
        })
        if self._next >= len(self._turns):
            self.overrun += 1
            name, arguments = "none", {}
        else:
            name, arguments = self._turns[self._next]
            self._next += 1
        data = types.SimpleNamespace(
            tool_name=str(name),
            arguments=dict(arguments or {}),
            reasoning="",
        )
        return types.SimpleNamespace(success=True, data=data)


# ---------------------------------------------------------------------------
# Fixture-backed tool surface
# ---------------------------------------------------------------------------
class VirtualWorkspace:
    """In-memory file store the scripted tool handlers close over."""

    def __init__(self, fixture: dict[str, str] | None = None):
        self.files: dict[str, str] = dict(fixture or {})

    def read(self, filename: str) -> str:
        if filename not in self.files:
            raise ValueError(f"file not found: {filename}")
        return self.files[filename]

    def write(self, filename: str, content: str) -> str:
        text = str(content)
        self.files[str(filename)] = text
        return f"Wrote {len(text)} characters to {filename}"

    def listing(self) -> str:
        return "\n".join(sorted(self.files)) if self.files else "(no files)"


# Reserved write path for scripted failures. A write_file call whose
# filename starts with this prefix returns the executor's textual write
# failure marker instead of writing. It is the one deliberate exception to
# the marker-free canned outputs: a task can script a failed write (and its
# recovery) so the executor's verification pass has something real to see.
# The handler still returns text and never raises, exactly like the
# production write handler, which captures failures as text too.
SCRIPTED_READONLY_PREFIX = "readonly/"

# Reserved code sentinel for scripted execution failures. An execute_code
# call whose code contains this sentinel returns the executor's textual
# execution failure marker (production shape) instead of the canned
# success line. Together with SCRIPTED_READONLY_PREFIX it is the second
# deliberate exception to the marker-free canned outputs: a task can
# script a failed run (and its correction) so the verification pass has
# something real to see on the execution branch. The handler still
# returns text and never raises, exactly like the production execution
# handler, which captures failures as text too.
SCRIPTED_EXEC_FAILURE_SENTINEL = "# scripted: fail"


def build_scripted_registry(workspace: VirtualWorkspace):
    """A fresh registry with deterministic, offline tool stand-ins.

    The file tools close over the in-memory workspace; execute_code and
    web_search return canned deterministic text. Nothing runs, nothing
    leaves the process: the handlers exist so the loop's decision, retry
    and progress mechanics can be measured, not to perform work. The
    canned outputs deliberately avoid the executor's failure markers, so
    the verification pass stays inert unless a task scripts a failure
    through the reserved ``readonly/`` write prefix (see
    ``SCRIPTED_READONLY_PREFIX``) or the reserved execution-failure
    sentinel (see ``SCRIPTED_EXEC_FAILURE_SENTINEL``).
    """
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="read_file",
        description="Read a file from the workspace.",
        parameters={
            "filename": ToolParam(
                "filename", "string", "Name of the file to read.",
            ),
        },
        handler=workspace.read,
    ))
    def _scripted_write(filename: str, content: str) -> str:
        if str(filename).startswith(SCRIPTED_READONLY_PREFIX):
            return (
                f"Write file error: {filename} is under the reserved "
                f"read-only prefix"
            )
        return workspace.write(filename, content)

    registry.register(ToolDefinition(
        name="write_file",
        description="Write content to a file in the workspace.",
        parameters={
            "filename": ToolParam(
                "filename", "string", "Name of the file to write.",
            ),
            "content": ToolParam(
                "content", "string", "Content to write.",
            ),
        },
        handler=_scripted_write,
    ))
    registry.register(ToolDefinition(
        name="list_files",
        description="List the files in the workspace.",
        parameters={},
        handler=workspace.listing,
    ))
    def _scripted_execute(code: str) -> str:
        if SCRIPTED_EXEC_FAILURE_SENTINEL in str(code):
            return "Execution Failed (return code: 1)"
        return f"exit code 0\n[scripted run of {len(str(code))} characters]"

    registry.register(ToolDefinition(
        name="execute_code",
        description="Run a code snippet and return its output.",
        parameters={
            "code": ToolParam("code", "string", "The code to run."),
        },
        handler=_scripted_execute,
    ))
    registry.register(ToolDefinition(
        name="web_search",
        description="Search the web for a query.",
        parameters={
            "query": ToolParam("query", "string", "Search query."),
        },
        handler=lambda query: f"[scripted results for: {query}]",
    ))
    return registry


@contextmanager
def scripted_chat_backend(client: ScriptedChatClient):
    """Route the executor module's chat backend to ``client`` for the block.

    The executor resolves its backend at module level; the harness swaps
    that binding in and restores the previous one afterwards, under the
    module lock so concurrent runs cannot interleave their swaps.
    """
    if _texec_mod is None:
        raise RuntimeError("chat surface unavailable")
    with _RUN_LOCK:
        had_client = hasattr(_texec_mod, "ollama")
        prev_client = getattr(_texec_mod, "ollama", None)
        prev_flag = getattr(_texec_mod, "OLLAMA_AVAILABLE", False)
        _texec_mod.ollama = client
        _texec_mod.OLLAMA_AVAILABLE = True
        try:
            yield
        finally:
            if had_client:
                _texec_mod.ollama = prev_client
            else:  # pragma: no cover - backend package absent entirely
                try:
                    delattr(_texec_mod, "ollama")
                except AttributeError:
                    pass
            _texec_mod.OLLAMA_AVAILABLE = prev_flag


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------
def _coerce_script(raw: Any, task_id: str) -> list[ScriptTurn]:
    """Validate one task's script block into ScriptTurn objects."""
    where = f"task {task_id!r}"
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"{where}: 'script' must be a non-empty list of model turns"
        )
    turns: list[ScriptTurn] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{where}: script turn #{index} must be a mapping"
            )
        content = entry.get("content", "")
        if not isinstance(content, str):
            raise ValueError(
                f"{where}: script turn #{index} 'content' must be a string"
            )
        calls_raw = entry.get("tool_calls", []) or []
        if not isinstance(calls_raw, list):
            raise ValueError(
                f"{where}: script turn #{index} 'tool_calls' must be a list"
            )
        calls: list[dict] = []
        for call in calls_raw:
            if not isinstance(call, dict):
                raise ValueError(
                    f"{where}: script turn #{index} tool call must be a "
                    f"mapping"
                )
            name = call.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"{where}: script turn #{index} tool call needs a "
                    f"non-empty 'name'"
                )
            arguments = call.get("arguments", {}) or {}
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"{where}: script turn #{index} 'arguments' must be a "
                    f"mapping"
                )
            calls.append({
                "name": name.strip(),
                "arguments": dict(arguments),
            })
        turns.append(ScriptTurn(content=content, tool_calls=calls))
    last = turns[-1]
    if last.tool_calls or not last.content.strip():
        raise ValueError(
            f"{where}: the last script turn must be a non-empty text "
            f"answer with no tool calls"
        )
    return turns


def load_chat_suite(suite: str) -> tuple[str, list[ChatTask]]:
    """Load a chat suite: standard TaskSpec validation plus the script key.

    The standard task fields go through the standard coercion verbatim; the
    additive ``script`` key (which the standard loader ignores) is read from
    the same entries and validated here.
    """
    if not FEATURE_AVAILABLE:
        raise RuntimeError(
            "chat eval harness unavailable: chat surface or PyYAML missing"
        )
    path = resolve_suite_path(suite)
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"suite {path.name}: top level must be a mapping")
    entries = raw.get("tasks")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"suite {path.name}: 'tasks' must be a non-empty list"
        )
    seen: set[str] = set()
    tasks: list[ChatTask] = []
    for index, entry in enumerate(entries):
        spec = _coerce_task(entry, index, seen)
        script = _coerce_script(
            entry.get("script") if isinstance(entry, dict) else None,
            spec.id,
        )
        tasks.append(ChatTask(spec=spec, script=script))
    suite_name = raw.get("suite")
    if not isinstance(suite_name, str) or not suite_name.strip():
        suite_name = path.stem
    return suite_name, tasks


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class CheckOutcome:
    """One checker verdict for one task run."""

    check: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"check": self.check, "ok": self.ok, "detail": self.detail}


@dataclass
class TaskRunRecord:
    """Everything one task run produced, measured from outside the loop."""

    task_id: str
    front: str
    passed: bool
    checks: list[CheckOutcome]
    tool_calls: list[dict]
    model_calls: list[dict]
    events: list[list]
    final_text: str
    script_leftover: int
    script_overrun: int
    workspace_files: list[str]

    def to_row(self) -> dict:
        return {
            "task": self.task_id,
            "front": self.front,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "tool_calls": list(self.tool_calls),
            "events": [list(e) for e in self.events],
            "final_chars": len(self.final_text),
            "final_preview": self.final_text[:120],
            "script_leftover": self.script_leftover,
            "script_overrun": self.script_overrun,
            "workspace_files": list(self.workspace_files),
        }


@dataclass
class ChatEvalReport:
    """The outcome of a full suite run across the requested fronts."""

    suite: str
    fronts: tuple
    records: list[TaskRunRecord]
    trace_path: str | None = None

    @property
    def passed(self) -> bool:
        return all(record.passed for record in self.records)


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------
def scan_internal_markers(text: str) -> list[str]:
    """Lines in ``text`` that start with a runtime scaffold prefix."""
    hits: list[str] = []
    for line in (text or "").splitlines():
        lowered = line.lstrip().lower()
        for prefix in _LEAK_PREFIXES:
            if lowered.startswith(prefix):
                hits.append(line.strip()[:80])
                break
    return hits


def _apply_check(name: str, record: TaskRunRecord) -> tuple[bool, str]:
    """Evaluate one named checker against a task run record."""
    if name == "final_nonempty":
        ok = bool(record.final_text.strip())
        return ok, "" if ok else "final answer is empty"
    if name == "no_misattribution":
        hits = detect_misattribution(record.final_text)
        return (not hits), "; ".join(hits[:3])
    if name == "no_internal_markers":
        hits = scan_internal_markers(record.final_text)
        return (not hits), "; ".join(hits[:3])
    if name == "expect_no_tools":
        ok = not record.tool_calls
        return ok, "" if ok else f"{len(record.tool_calls)} tool call(s)"
    if name == "tools_before_stream":
        if record.front != "stream":
            return True, "not a streaming run"
        first_chunk = None
        for position, event in enumerate(record.events):
            if event[0] == "chunk":
                first_chunk = position
                break
        if first_chunk is None:
            return True, "no chunk emitted"
        late = [
            event[1]
            for event in record.events[first_chunk:]
            if event[0] == "tool_call"
        ]
        return (not late), "; ".join(str(item) for item in late[:3])
    if name.startswith("expect_tool:"):
        wanted = name.split(":", 1)[1].strip()
        ok = any(
            call["name"] == wanted and call["success"]
            for call in record.tool_calls
        )
        return ok, "" if ok else f"no successful call to {wanted}"
    if name.startswith("expect_file:"):
        wanted = name.split(":", 1)[1].strip()
        ok = wanted in record.workspace_files
        return ok, "" if ok else f"{wanted} not in workspace"
    if name.startswith("max_calls:"):
        try:
            limit = int(name.split(":", 1)[1])
        except ValueError:
            return False, "max_calls needs an integer bound"
        ok = len(record.tool_calls) <= limit
        return ok, "" if ok else f"{len(record.tool_calls)} > {limit}"
    return False, "unknown check"


def _run_checks(task: ChatTask, record: TaskRunRecord) -> list[CheckOutcome]:
    outcomes: list[CheckOutcome] = []
    for raw_name in task.spec.checks:
        name = raw_name.strip()
        ok, detail = _apply_check(name, record)
        outcomes.append(CheckOutcome(check=name, ok=ok, detail=detail))
    fidelity_ok = record.script_leftover == 0 and record.script_overrun == 0
    outcomes.append(CheckOutcome(
        check="script_fidelity",
        ok=fidelity_ok,
        detail="" if fidelity_ok else (
            f"leftover={record.script_leftover} "
            f"overrun={record.script_overrun}"
        ),
    ))
    return outcomes


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def _run_task(
    task: ChatTask,
    front: str,
    executor_kwargs: dict | None = None,
) -> TaskRunRecord:
    """Run one task on one front with a fresh client, workspace, executor.

    ``executor_kwargs`` are applied on top of the harness defaults, so the
    same suite can be measured across executor configurations (for example
    a transcript mode) without touching the harness itself.
    """
    client = ScriptedChatClient(task.script)
    workspace = VirtualWorkspace(task.spec.fixture)
    registry = build_scripted_registry(workspace)
    kwargs: dict[str, Any] = dict(
        registry=registry,
        structured_engine=_NoDecisionEngine(),
        max_tool_calls=task.spec.max_rounds,
        default_model=SCRIPTED_MODEL,
    )
    kwargs.update(executor_kwargs or {})
    executor = ToolExecutor(**kwargs)

    events: list[list] = []
    tool_rows: list[dict] = []

    def on_tool_call(result: Any) -> None:
        tool_rows.append({
            "name": str(getattr(result, "tool_name", "")),
            "success": bool(getattr(result, "success", False)),
        })
        events.append(["tool_call", str(getattr(result, "tool_name", ""))])

    final_text = ""
    with scripted_chat_backend(client):
        if front == "stream":
            generator = executor.stream_with_tools(
                message=task.spec.prompt,
                on_tool_call=on_tool_call,
            )
            result = None
            try:
                while True:
                    chunk = next(generator)
                    if chunk:
                        final_text += chunk
                        events.append(["chunk", len(chunk)])
            except StopIteration as stop:
                result = stop.value
            if result is not None and getattr(result, "response", ""):
                final_text = result.response
        else:
            result = executor.execute_with_tools(
                message=task.spec.prompt,
                on_tool_call=on_tool_call,
            )
            final_text = getattr(result, "response", "") or ""

    record = TaskRunRecord(
        task_id=task.spec.id,
        front=front,
        passed=False,
        checks=[],
        tool_calls=tool_rows,
        model_calls=list(client.calls),
        events=events,
        final_text=final_text,
        script_leftover=client.leftover,
        script_overrun=client.overrun,
        workspace_files=sorted(workspace.files),
    )
    record.checks = _run_checks(task, record)
    record.passed = all(outcome.ok for outcome in record.checks)
    return record


class _TraceWriter:
    """JSONL rows in the agent loop's observable-event shape."""

    def __init__(self, path: Path):
        self._handle = open(path, "w", encoding="utf-8")

    def row(self, kind: str, rnd: int, data: dict) -> None:
        payload = {"kind": kind, "round": rnd, "data": data}
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def close(self) -> None:
        self._handle.flush()
        self._handle.close()


def run_suite(
    suite: str,
    fronts: tuple = FRONTS,
    trace_dir: str | Path | None = None,
    write_traces: bool = True,
    executor_kwargs: dict | None = None,
) -> ChatEvalReport:
    """Run every task of ``suite`` on every requested front.

    Each (task, front) pair gets a fresh scripted client, workspace,
    registry and executor: nothing persists between runs except the record
    rows and the trace file. ``executor_kwargs`` overlay the executor
    defaults for every run (see ``_run_task``), enabling side-by-side
    measurement of executor configurations on identical tasks.
    """
    if not FEATURE_AVAILABLE:
        raise RuntimeError(
            "chat eval harness unavailable: chat surface or PyYAML missing"
        )
    suite_name, tasks = load_chat_suite(suite)
    fronts = tuple(fronts)
    for front in fronts:
        if front not in FRONTS:
            raise ValueError(f"unknown front: {front!r}")
    if not fronts:
        raise ValueError("at least one front is required")

    writer = None
    trace_path: str | None = None
    if write_traces:
        base = Path(trace_dir) if trace_dir else TRACE_DIR_DEFAULT
        base.mkdir(parents=True, exist_ok=True)
        run_id = "{}-{}".format(
            time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8],
        )
        trace_path = str(base / f"{run_id}.jsonl")
        writer = _TraceWriter(Path(trace_path))
        writer.row("run_start", 0, {
            "suite": suite_name,
            "fronts": list(fronts),
            "ts": time.time(),
        })

    records: list[TaskRunRecord] = []
    try:
        for task in tasks:
            for front in fronts:
                record = _run_task(task, front, executor_kwargs)
                records.append(record)
                if writer is not None:
                    writer.row("task_start", 0, {
                        "task": record.task_id, "front": front,
                    })
                    for call in record.model_calls:
                        writer.row("model_call", 0, {
                            "task": record.task_id,
                            "front": front,
                            **call,
                        })
                    for call in record.tool_calls:
                        writer.row("tool_call", 0, {
                            "task": record.task_id,
                            "front": front,
                            **call,
                        })
                    writer.row("task_end", 0, record.to_row())
    finally:
        if writer is not None:
            writer.row("run_end", 0, {
                "passed": all(r.passed for r in records),
                "tasks": len(records),
                "ts": time.time(),
            })
            writer.close()

    return ChatEvalReport(
        suite=suite_name,
        fronts=fronts,
        records=records,
        trace_path=trace_path,
    )


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a chat tool-loop eval suite against the scripted backend."
        ),
    )
    parser.add_argument(
        "suite", nargs="?", default="chat_tools",
        help="suite name or YAML path (default: chat_tools)",
    )
    parser.add_argument(
        "--front", choices=[*FRONTS, "both"], default="both",
        help="which execution front(s) to run (default: both)",
    )
    parser.add_argument(
        "--trace-dir", default=None,
        help="directory for the JSONL trace (default: data/chat_eval_traces)",
    )
    parser.add_argument(
        "--no-traces", action="store_true",
        help="skip writing the JSONL trace",
    )
    args = parser.parse_args(argv)

    if not FEATURE_AVAILABLE:
        print("chat eval harness unavailable: chat surface or PyYAML missing")
        return 2

    selected = FRONTS if args.front == "both" else (args.front,)
    report = run_suite(
        args.suite,
        fronts=selected,
        trace_dir=args.trace_dir,
        write_traces=not args.no_traces,
    )
    for record in report.records:
        status = "PASS" if record.passed else "FAIL"
        line = (
            f"{status} {record.task_id} [{record.front}] "
            f"calls={len(record.tool_calls)} "
            f"final_chars={len(record.final_text)}"
        )
        failing = [c.check for c in record.checks if not c.ok]
        if failing:
            line += " failing: " + ", ".join(failing)
        print(line)
    print(
        f"suite={report.suite} runs={len(report.records)} "
        f"passed={report.passed}"
    )
    if report.trace_path:
        print(f"trace: {report.trace_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
