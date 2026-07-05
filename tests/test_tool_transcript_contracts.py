#!/usr/bin/env python3
"""Contracts for the tool-transcript mode of the chat tool loop.

The executor can replay executed tool calls to the model in two shapes:
the historical "flat" reconstruction (every result folded as text into a
single rebuilt user message) and the "native" transcript (assistant
tool_calls echoes plus role "tool" messages -- the format function-calling
models are trained on). These contracts pin what makes the flag safe to
ship and worth measuring:

  * Contract 1 -- the default is flat and the flat shapes are byte-exact:
    with no flag (and no configuration), decision and final messages are
    the exact historical strings; an unknown mode value falls back to
    flat; an empty transcript falls back to the flat body even under the
    native flag.
  * Contract 2 -- the native transcript is exactly the trained shape: one
    assistant echo per executed call (post-repair arguments), one role
    "tool" message per result with the environment/untrusted framing
    INSIDE the content, the final generation led by the attribution
    system message and ending on the last tool message with no synthetic
    user turn, and no flat results header anywhere.
  * Contract 3 -- coherent degradation: when the native decision path
    fails mid-loop under the native flag, the format= fallback still
    receives the FLAT results context (both representations are
    accumulated), so the run completes instead of losing its history.
  * Contract 4 -- the measurement: on identical tasks driven through the
    eval harness, native decision input is smaller in total than flat
    (no re-folded reconstruction), native rounds actually carry
    assistant/tool roles, and the hygiene checkers stay green.
  * Contract 5 -- the harness override is inert by default: no override,
    an empty override and an explicit flat override produce identical
    report rows.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
contracts: canonical dotted names so package and relative imports resolve
against the loaded copies, with a minimal pydantic stand-in only when the
real package is absent.
"""

import copy
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
# Isolated loading (sibling-harness idiom)
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
# Local material
# ---------------------------------------------------------------------------
MSG = "Read notes.txt, then write out.txt, then report."

FINAL_TEXT = (
    "I read notes.txt and wrote a short summary to out.txt with my file "
    "tools; the source said alpha."
)

FINAL_MEASURE = (
    "I listed the workspace, read status.txt, and wrote a one-line summary "
    "to report.txt with my file tools; everything reported nominal."
)

SUITE_MEASURE = """\
suite: transcript-measure

tasks:
  - id: m1-chain
    title: Three-step chain
    prompt: >
      List the workspace files, read status.txt, and write a one-line
      summary to report.txt.
    fixture:
      status.txt: "All services nominal.\\nQueue depth: 3.\\n"
    checks:
      - "expect_tool:write_file"
      - "expect_file:report.txt"
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
              filename: report.txt
              content: "Services nominal; queue depth 3.\\n"
      - content: ""
      - content: "__FINAL__"
""".replace("__FINAL__", FINAL_MEASURE)


def _executor(te, cr, registry, **overrides):
    kwargs = dict(
        registry=registry,
        structured_engine=cr._NoDecisionEngine(),
        max_tool_calls=6,
        default_model=cr.SCRIPTED_MODEL,
    )
    kwargs.update(overrides)
    return te.ToolExecutor(**kwargs)


def _turn(cr, **fields):
    return cr.ScriptTurn(**fields)


def _suite_file(directory: str, name: str, text: str) -> str:
    path = Path(directory) / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Contract 1 -- flat is the default and its shapes are byte-exact
# ---------------------------------------------------------------------------
def test_c1_flat_default_and_golden_bytes():
    cr, te, restore = _load()
    try:
        registry = cr.build_scripted_registry(cr.VirtualWorkspace({}))

        flat = _executor(te, cr, registry)
        assert flat.tool_transcript == te.TOOL_TRANSCRIPT_FLAT, (
            flat.tool_transcript
        )
        native = _executor(te, cr, registry, tool_transcript="native")
        assert native.tool_transcript == te.TOOL_TRANSCRIPT_NATIVE
        weird = _executor(te, cr, registry, tool_transcript="weird")
        assert weird.tool_transcript == te.TOOL_TRANSCRIPT_FLAT, (
            weird.tool_transcript
        )

        prev = [
            "[environment] tool call by assistant: read_file\n"
            "Arguments: {'filename': 'notes.txt'}\n"
            "Result: alpha\n"
        ]
        expected_decision = [{
            "role": "user",
            "content": (
                f"Q?\n\n{te.ENV_RESULTS_HEADER}\n{prev[0]}\n\n"
                "Call the next tool if needed; otherwise answer directly. "
                "Never attribute these tool actions to the user."
            ),
        }]
        got = flat._build_decision_messages("Q?", [], prev)
        assert got == expected_decision, got

        expected_final = [
            {"role": "system", "content": te.FINAL_ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Q?\n\n{te.ENV_RESULTS_HEADER}\n{prev[0]}\n\n"
                    "Write the final user-facing answer, reporting in "
                    "first person what you did and what the results were."
                ),
            },
        ]
        got_final = flat._final_messages("Q?", prev, [])
        assert got_final == expected_final, got_final

        # Native flag with an empty transcript falls back to the flat body.
        assert native._final_messages("Q?", prev, []) == expected_final
        assert native._build_decision_messages("Q?", [], prev) == (
            expected_decision
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- the native transcript is exactly the trained shape
# ---------------------------------------------------------------------------
class _RecordingClient:
    """ScriptedChatClient wrapper capturing the full messages of each call."""

    def __init__(self, inner):
        self._inner = inner
        self.captured: list[dict] = []

    def chat(self, model=None, messages=None, tools=None, options=None,
             stream=False, **kwargs):
        self.captured.append({
            "stream": bool(stream),
            "tools": tools is not None,
            "messages": copy.deepcopy(list(messages or [])),
        })
        return self._inner.chat(
            model=model, messages=messages, tools=tools, options=options,
            stream=stream, **kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_c2_native_transcript_exact():
    cr, te, restore = _load()
    try:
        workspace = cr.VirtualWorkspace({"notes.txt": "alpha\n"})
        registry = cr.build_scripted_registry(workspace)
        turns = [
            _turn(cr, tool_calls=[
                {"name": "read_file", "arguments": {"filename": "notes.txt"}},
            ]),
            _turn(cr, tool_calls=[
                {"name": "write_file", "arguments": {
                    "filename": "out.txt", "content": "done\n",
                }},
            ]),
            _turn(cr, content=""),
            _turn(cr, content=FINAL_TEXT),
        ]
        client = _RecordingClient(cr.ScriptedChatClient(turns))
        executor = _executor(te, cr, registry, tool_transcript="native")

        with cr.scripted_chat_backend(client):
            result = executor.execute_with_tools(message=MSG)

        calls = client.captured
        assert len(calls) == 4, [(c["stream"], c["tools"]) for c in calls]
        for call in calls[:3]:
            assert call["tools"] is True and call["stream"] is False, call
        assert calls[3]["tools"] is False and calls[3]["stream"] is False

        # Decision round 1: just the user message.
        assert calls[0]["messages"] == [{"role": "user", "content": MSG}]

        # Decision round 2: user, assistant echo, tool result.
        m = calls[1]["messages"]
        assert [x["role"] for x in m] == ["user", "assistant", "tool"], m
        echo = m[1]
        assert echo["content"] == ""
        assert echo["tool_calls"][0]["function"]["name"] == "read_file"
        assert echo["tool_calls"][0]["function"]["arguments"] == {
            "filename": "notes.txt",
        }
        expected_tool = (
            "[environment] tool result (untrusted content): read_file\n"
            "Result: alpha\n"
        )
        assert m[2] == {"role": "tool", "content": expected_tool}, m[2]

        # Decision round 3: both pairs, in execution order.
        m = calls[2]["messages"]
        assert [x["role"] for x in m] == [
            "user", "assistant", "tool", "assistant", "tool",
        ], m
        assert m[3]["tool_calls"][0]["function"]["name"] == "write_file"
        assert "Wrote 5 characters to out.txt" in m[4]["content"], m[4]

        # Final generation: attribution system first, ends on the last tool
        # message, no synthetic trailing user, no flat results header.
        m = calls[3]["messages"]
        assert m[0] == {
            "role": "system", "content": te.FINAL_ANSWER_SYSTEM_PROMPT,
        }
        assert [x["role"] for x in m] == [
            "system", "user", "assistant", "tool", "assistant", "tool",
        ], m
        assert m[-1]["role"] == "tool"
        joined = json.dumps([c["messages"] for c in calls])
        assert te.ENV_RESULTS_HEADER not in joined

        assert result.response == FINAL_TEXT, result.response
        assert [t.success for t in result.tool_calls] == [True, True]
        assert client.leftover == 0 and client.overrun == 0
        assert workspace.files.get("out.txt") == "done\n"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3 -- native outage mid-loop degrades to the flat fallback
# ---------------------------------------------------------------------------
class _OutageClient(_RecordingClient):
    """Raises on one specific backend call without consuming a turn."""

    def __init__(self, inner, raise_at: int):
        super().__init__(inner)
        self._raise_at = raise_at

    def chat(self, model=None, messages=None, tools=None, options=None,
             stream=False, **kwargs):
        if len(self.captured) == self._raise_at:
            self.captured.append({
                "stream": bool(stream),
                "tools": tools is not None,
                "messages": copy.deepcopy(list(messages or [])),
            })
            raise RuntimeError("scripted native outage")
        return super().chat(
            model=model, messages=messages, tools=tools, options=options,
            stream=stream, **kwargs,
        )


class _SpyEngine:
    """format= fallback stand-in: records the messages, always declines."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def generate_structured(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs.get("messages") or []))
        return types.SimpleNamespace(success=False, data=None)


def test_c3_native_outage_falls_back_to_flat_context():
    cr, te, restore = _load()
    try:
        workspace = cr.VirtualWorkspace({"notes.txt": "alpha\n"})
        registry = cr.build_scripted_registry(workspace)
        turns = [
            _turn(cr, tool_calls=[
                {"name": "read_file", "arguments": {"filename": "notes.txt"}},
            ]),
            _turn(cr, content="Recovered answer."),
        ]
        client = _OutageClient(cr.ScriptedChatClient(turns), raise_at=1)
        spy = _SpyEngine()
        executor = _executor(
            te, cr, registry,
            tool_transcript="native", structured_engine=spy,
        )

        with cr.scripted_chat_backend(client):
            result = executor.execute_with_tools(message=MSG)

        # The fallback was consulted once, with the FLAT results context:
        # the rebuilt user message carries the header and the first result.
        assert len(spy.calls) == 1, len(spy.calls)
        fallback_user = spy.calls[0][-1]
        assert fallback_user["role"] == "user"
        assert te.ENV_RESULTS_HEADER in fallback_user["content"]
        assert "read_file" in fallback_user["content"]
        assert "alpha" in fallback_user["content"]

        # The run completed: the executed call survived, the final answer
        # was generated (native transcript path) after the outage.
        assert [t.success for t in result.tool_calls] == [True]
        assert result.response == "Recovered answer.", result.response
        assert client.leftover == 0 and client.overrun == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4 -- measured on identical tasks: native decisions are smaller
# ---------------------------------------------------------------------------
def test_c4_native_measured_smaller_than_flat():
    cr, _te, restore = _load()
    try:
        base = tempfile.mkdtemp(prefix="transcript-c4-")
        suite = _suite_file(base, "measure", SUITE_MEASURE)
        flat = cr.run_suite(
            suite, fronts=("stream",), trace_dir=str(Path(base) / "flat"),
        )
        native = cr.run_suite(
            suite, fronts=("stream",), trace_dir=str(Path(base) / "native"),
            executor_kwargs={"tool_transcript": "native"},
        )
        rec_flat = flat.records[0]
        rec_native = native.records[0]
        assert rec_flat.passed, [c.to_dict() for c in rec_flat.checks]
        assert rec_native.passed, [c.to_dict() for c in rec_native.checks]

        flat_dec = [c["chars_in"] for c in rec_flat.model_calls[:4]]
        native_dec = [c["chars_in"] for c in rec_native.model_calls[:4]]
        assert len(rec_flat.model_calls) == 5
        assert len(rec_native.model_calls) == 5
        assert sum(native_dec) < sum(flat_dec), (native_dec, flat_dec)
        assert all(b > a for a, b in zip(flat_dec, flat_dec[1:])), flat_dec
        assert all(b > a for a, b in zip(native_dec, native_dec[1:])), (
            native_dec
        )

        # Native rounds really carry the trained roles; flat ones do not.
        assert "tool" in rec_native.model_calls[2]["roles"]
        assert "assistant" in rec_native.model_calls[2]["roles"]
        assert rec_flat.model_calls[2]["roles"] == ["user"]
        assert rec_native.model_calls[4]["roles"][0] == "system"
        assert rec_native.model_calls[4]["roles"][-1] == "tool"
        assert rec_flat.model_calls[4]["roles"] == ["system", "user"]

        assert rec_native.final_text == FINAL_MEASURE
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 5 -- the harness override is inert by default
# ---------------------------------------------------------------------------
def test_c5_harness_override_inert_by_default():
    cr, _te, restore = _load()
    try:
        base = tempfile.mkdtemp(prefix="transcript-c5-")
        suite = _suite_file(base, "measure", SUITE_MEASURE)
        rows = []
        for index, kwargs in enumerate((
            None, {}, {"tool_transcript": "flat"},
        )):
            report = cr.run_suite(
                suite,
                trace_dir=str(Path(base) / f"run{index}"),
                executor_kwargs=kwargs,
            )
            rows.append([record.to_row() for record in report.records])
        assert rows[0] == rows[1] == rows[2], (
            "default, empty and explicit-flat overrides must be identical"
        )
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
