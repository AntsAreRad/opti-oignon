#!/usr/bin/env python3
"""Contracts for the scripted execution-failure affordance and the
execution branch of the verification pass.

  * Contract 1 -- the affordance and its French suite: an execute_code
    call whose code carries the reserved sentinel returns the executor's
    execution-failure marker in production shape, any other code keeps
    the canned success line byte-identical, and the dedicated French
    suite (scripted failure, corrective run, first-person answer) is
    green end to end on both fronts with exact script fidelity.
  * Contract 2 -- the execution branch of the verification pass: a
    failing execution yields the corrective observation verbatim, a
    successful execution yields None even when an older failed write
    sits behind it (only the most recent artifact-producing call is
    judged), non-artifact histories stay inert, and through the real
    loop on the flat path the hint enters the rebuilt user message
    exactly once per subsequent model call, never before the failure.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
contracts: canonical dotted names so package and relative imports resolve
against the loaded copies, with a minimal pydantic stand-in only when the
real package is absent.
"""

import copy
import importlib.util
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
def _load():
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
    cr = _real(
        "opti_oignon.agent_eval.chat_runner",
        _OO / "agent_eval" / "chat_runner.py",
    )
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
SUITE_NAME = "chat_tools_fr_exec"
TASK_ID = "e01-execution-corrigee"

SENTINEL = "# scripted: fail"

EXEC_HINT = (
    "[environment] verification: the last code execution reported an "
    "error. Inspect the output above, fix the code, and run it again."
)
WRITE_HINT_PREFIX = "[environment] verification: the last file write"
EXEC_FAILURE_MARKER = "Execution Failed (return code:"

EXPECTED_FINAL_FR = (
    "Ma premi\u00e8re ex\u00e9cution du script de contr\u00f4le a "
    "\u00e9chou\u00e9 avec un code de retour non nul ; j\u2019ai "
    "corrig\u00e9 le script puis je l\u2019ai relanc\u00e9 : il "
    "s\u2019est termin\u00e9 correctement et le contr\u00f4le affiche "
    "la valeur 42."
)

MSG = "Run the control script; if it fails, fix it and run it, then report."
FAIL_CODE = SENTINEL + "\nprint('check')\n"
OK_CODE = "print('check: 42')\n"
FINAL_TEXT = (
    "My first run of the control script failed with a non-zero return "
    "code; I fixed the script, ran it again, and it completed with the "
    "expected output."
)


def _ns(tool_name: str, result: str):
    return types.SimpleNamespace(tool_name=tool_name, result=result)


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


def _turns(cr):
    return [
        cr.ScriptTurn(tool_calls=[
            {"name": "execute_code", "arguments": {"code": FAIL_CODE}},
        ]),
        cr.ScriptTurn(content=""),
        cr.ScriptTurn(tool_calls=[
            {"name": "execute_code", "arguments": {"code": OK_CODE}},
        ]),
        cr.ScriptTurn(content=""),
        cr.ScriptTurn(content=FINAL_TEXT),
    ]


# ---------------------------------------------------------------------------
# Contract 1 -- the affordance and its French suite are exact
# ---------------------------------------------------------------------------
def test_c1_exec_affordance_and_fr_suite_green():
    cr, _te, restore = _load()
    try:
        assert cr.SCRIPTED_EXEC_FAILURE_SENTINEL == SENTINEL, (
            cr.SCRIPTED_EXEC_FAILURE_SENTINEL
        )

        registry = cr.build_scripted_registry(cr.VirtualWorkspace({}))
        handler = registry.get("execute_code").handler
        failed = handler(code=FAIL_CODE)
        assert EXEC_FAILURE_MARKER in failed, failed
        assert failed == "Execution Failed (return code: 1)", failed
        clean = handler(code=OK_CODE)
        assert clean == (
            f"exit code 0\n[scripted run of {len(OK_CODE)} characters]"
        ), clean

        with tempfile.TemporaryDirectory(prefix="exec-fr-c1-") as tmp:
            report = cr.run_suite(SUITE_NAME, trace_dir=tmp)
        assert report.passed, [
            (r.task_id, r.front, [c.check for c in r.checks if not c.ok])
            for r in report.records
        ]
        assert len(report.records) == 2, len(report.records)
        assert sorted(r.front for r in report.records) == [
            "execute", "stream",
        ]
        for rec in report.records:
            assert rec.task_id == TASK_ID
            assert [c["name"] for c in rec.tool_calls] == [
                "execute_code", "execute_code",
            ], rec.tool_calls
            assert all(c["success"] for c in rec.tool_calls)
            assert rec.script_leftover == 0 and rec.script_overrun == 0
            assert rec.final_text.strip() == EXPECTED_FINAL_FR, (
                rec.final_text
            )
            assert rec.workspace_files == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- the execution branch of the verification pass
# ---------------------------------------------------------------------------
def test_c2_exec_verification_branch_exact():
    cr, te, restore = _load()
    try:
        hint = te._verification_hint

        assert hint([_ns("execute_code", "Execution Failed (return code: 1)")
                     ]) == EXEC_HINT
        assert hint([_ns(
            "execute_code",
            "STDERR:\nTraceback (most recent call last):\n  boom",
        )]) == EXEC_HINT

        # Scan-back: only the most recent artifact call is judged. A clean
        # run in front of an older failed write must stay silent.
        assert hint([
            _ns("write_file", "Write file error: locked"),
            _ns("execute_code", "exit code 0\n[scripted run of 9 chars]"),
        ]) is None
        assert hint([_ns("execute_code", "exit code 0")]) is None
        assert hint([]) is None
        assert hint([_ns("read_file", "whatever")]) is None
        judged = hint([
            _ns("execute_code", "exit code 0"),
            _ns("write_file", "Write file error: locked"),
        ])
        assert judged is not None and judged.startswith(WRITE_HINT_PREFIX)

        # End to end on the flat path: the hint enters the rebuilt user
        # message exactly once per model call after the failure, never
        # before it, and the loop settles on the scripted final.
        registry = cr.build_scripted_registry(cr.VirtualWorkspace({}))
        client = _RecordingClient(cr.ScriptedChatClient(_turns(cr)))
        executor = te.ToolExecutor(
            registry=registry,
            structured_engine=cr._NoDecisionEngine(),
            max_tool_calls=6,
            default_model=cr.SCRIPTED_MODEL,
        )
        assert executor.tool_transcript == te.TOOL_TRANSCRIPT_FLAT
        seen: list[str] = []
        with cr.scripted_chat_backend(client):
            result = executor.execute_with_tools(
                message=MSG,
                on_tool_call=lambda r: seen.append(r.tool_name),
            )
        assert seen == ["execute_code", "execute_code"], seen
        assert [c.tool_name for c in result.tool_calls] == seen
        assert all(c.success for c in result.tool_calls)
        assert result.response == FINAL_TEXT, result.response
        assert client.leftover == 0 and client.overrun == 0
        assert len(client.captured) == 5, len(client.captured)

        contents = [
            "\n".join(
                str(m.get("content") or "") for m in call["messages"]
            )
            for call in client.captured
        ]
        assert [c.count(EXEC_HINT) for c in contents] == [0, 0, 1, 1, 1], (
            [c.count(EXEC_HINT) for c in contents]
        )
        assert contents[1].count(EXEC_FAILURE_MARKER) == 1
        assert EXEC_FAILURE_MARKER not in result.response
        assert EXEC_HINT not in result.response
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
