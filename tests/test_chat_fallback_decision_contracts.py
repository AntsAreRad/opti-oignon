#!/usr/bin/env python3
"""Contracts for the format= fallback decision path of the tool loop.

  * Contract 1 -- exact wiring of the fallback path: an executor built
    with a model outside the native function-calling families never
    offers native tool schemas to the chat backend and takes every
    decision through the structured engine, with the exact structured
    parameters (the registry tools prompt as the extra system prompt,
    the ToolDecision schema, temperature 0.0, two retries), a bare-user
    first decision, a byte-exact rebuilt second decision carrying the
    fallback instruction tail, a scripted "none" stopping the loop, and
    the final answer flowing through the chat backend alone -- on both
    fronts. The cross-control: with the native scripted model the
    engine is never consulted.
  * Contract 2 -- the verification pass and the growth of the fallback
    path: a scripted execution failure makes the corrective observation
    enter the engine's rebuilt user message exactly once per decision
    after the failure and once in the final generation, the corrected
    run settles on the scripted answer, and on a four-call chain the
    engine's decision context grows strictly at every round -- the
    measurement of the non-native path.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Loading follows the sibling harness
contracts: canonical dotted names so package and relative imports resolve
against the loaded copies, with a minimal pydantic stand-in only when the
real package is absent.
"""

import copy
import importlib.util
import sys
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
FALLBACK_TAIL = (
    "Based on these results, do you need another tool? If not, set "
    'tool_name to "none". Never attribute these tool actions to the user.'
)

EXEC_HINT = (
    "[environment] verification: the last code execution reported an "
    "error. Inspect the output above, fix the code, and run it again."
)
EXEC_FAILURE_MARKER = "Execution Failed (return code:"

MSG = "Read notes.txt and report its content."
FINAL_TEXT = (
    "I read notes.txt with my file tools; the source said alpha."
)

HINT_MSG = "Run the control script; fix it if it fails, then report."
HINT_FINAL = (
    "My first run of the control script failed; I fixed the script, ran "
    "it again, and it completed with the expected output."
)

CHAIN_MSG = "Read the four status files and report."
CHAIN_FINAL = (
    "I read the four status files with my file tools; all four report "
    "nominal."
)


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


def _fallback_executor(cr, te, registry, engine):
    return te.ToolExecutor(
        registry=registry,
        structured_engine=engine,
        max_tool_calls=6,
        default_model=cr.SCRIPTED_FALLBACK_MODEL,
    )


def _contents(call: dict) -> str:
    return "\n".join(
        str(m.get("content") or "") for m in call["messages"]
    )


# ---------------------------------------------------------------------------
# Contract 1 -- exact wiring of the fallback decision path
# ---------------------------------------------------------------------------
def test_c1_fallback_wiring_exact():
    cr, te, restore = _load()
    try:
        tc = sys.modules["opti_oignon.tool_calling"]
        assert cr.SCRIPTED_FALLBACK_MODEL == "scripted-fallback"
        assert not tc.model_supports_native_tools(cr.SCRIPTED_FALLBACK_MODEL)
        assert tc.model_supports_native_tools(cr.SCRIPTED_MODEL)

        # Execute front: one read, a scripted "none", the scripted final.
        workspace = cr.VirtualWorkspace({"notes.txt": "alpha\n"})
        registry = cr.build_scripted_registry(workspace)
        engine = cr.ScriptedDecisionEngine([
            ("read_file", {"filename": "notes.txt"}),
            ("none", {}),
        ])
        client = _RecordingClient(cr.ScriptedChatClient([
            cr.ScriptTurn(content=FINAL_TEXT),
        ]))
        executor = _fallback_executor(cr, te, registry, engine)
        seen: list[str] = []
        with cr.scripted_chat_backend(client):
            result = executor.execute_with_tools(
                message=MSG,
                on_tool_call=lambda r: seen.append(r.tool_name),
            )

        assert seen == ["read_file"], seen
        assert [c.tool_name for c in result.tool_calls] == ["read_file"]
        assert all(c.success for c in result.tool_calls)
        assert result.response == FINAL_TEXT, result.response
        assert engine.leftover == 0 and engine.overrun == 0
        assert client.leftover == 0 and client.overrun == 0

        # The chat backend saw no decision call and no native schemas:
        # exactly one call, the final generation, without tools.
        assert len(client.captured) == 1, len(client.captured)
        final_call = client.captured[0]
        assert final_call["tools"] is False
        assert final_call["stream"] is False
        roles = [m.get("role") for m in final_call["messages"]]
        assert roles == ["system", "user"], roles
        assert final_call["messages"][0]["content"] == (
            te.FINAL_ANSWER_SYSTEM_PROMPT
        )
        assert "Result: alpha" in final_call["messages"][1]["content"]

        # Both decisions went through the engine with the exact
        # structured parameters.
        assert len(engine.captured) == 2, len(engine.captured)
        tools_prompt = registry.get_tools_prompt()
        assert tools_prompt and "read_file" in tools_prompt
        for call in engine.captured:
            assert call["schema"] is te.ToolDecision
            assert call["model"] == cr.SCRIPTED_FALLBACK_MODEL
            assert call["extra_system_prompt"] == tools_prompt
            assert call["temperature"] == 0.0
            assert call["max_retries"] == 2
        first, second = engine.captured
        assert first["messages"] == [{"role": "user", "content": MSG}]
        prev = (
            "[environment] tool call by assistant: read_file\n"
            "Arguments: {'filename': 'notes.txt'}\n"
            "Result: alpha\n"
        )
        expected_second = (
            f"{MSG}\n\n{te.ENV_RESULTS_HEADER}\n{prev}\n\n{FALLBACK_TAIL}"
        )
        assert [m.get("role") for m in second["messages"]] == ["user"]
        assert second["messages"][0]["content"] == expected_second, (
            second["messages"][0]["content"]
        )

        # Stream front: the fallback final flows through the streaming
        # chat call, decisions still through the engine alone.
        engine2 = cr.ScriptedDecisionEngine([
            ("read_file", {"filename": "notes.txt"}),
            ("none", {}),
        ])
        client2 = _RecordingClient(cr.ScriptedChatClient([
            cr.ScriptTurn(content=FINAL_TEXT),
        ]))
        executor2 = _fallback_executor(
            cr, te, cr.build_scripted_registry(
                cr.VirtualWorkspace({"notes.txt": "alpha\n"}),
            ), engine2,
        )
        chunks: list[str] = []
        with cr.scripted_chat_backend(client2):
            gen = executor2.stream_with_tools(message=MSG)
            try:
                while True:
                    chunks.append(next(gen))
            except StopIteration:
                pass
        assert "".join(chunks) == FINAL_TEXT
        assert len(chunks) > 1, len(chunks)
        assert len(engine2.captured) == 2
        assert len(client2.captured) == 1
        assert client2.captured[0]["stream"] is True
        assert client2.captured[0]["tools"] is False
        assert engine2.leftover == 0 and client2.leftover == 0

        # Cross-control: with the native scripted model the engine is
        # never consulted, even when one is wired in.
        engine3 = cr.ScriptedDecisionEngine([("read_file", {})])
        client3 = cr.ScriptedChatClient([
            cr.ScriptTurn(tool_calls=[{
                "name": "read_file",
                "arguments": {"filename": "notes.txt"},
            }]),
            cr.ScriptTurn(content=""),
            cr.ScriptTurn(content=FINAL_TEXT),
        ])
        executor3 = te.ToolExecutor(
            registry=cr.build_scripted_registry(
                cr.VirtualWorkspace({"notes.txt": "alpha\n"}),
            ),
            structured_engine=engine3,
            max_tool_calls=6,
            default_model=cr.SCRIPTED_MODEL,
        )
        with cr.scripted_chat_backend(client3):
            native_result = executor3.execute_with_tools(message=MSG)
        assert native_result.response == FINAL_TEXT
        assert engine3.captured == [], engine3.captured
        assert engine3.overrun == 0 and engine3.leftover == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2 -- verification pass and growth on the fallback path
# ---------------------------------------------------------------------------
def test_c2_fallback_hint_and_growth():
    cr, te, restore = _load()
    try:
        # Verification interplay: a scripted execution failure through the
        # reserved sentinel, the scripted stop, the corrective decision.
        registry = cr.build_scripted_registry(cr.VirtualWorkspace({}))
        fail_code = cr.SCRIPTED_EXEC_FAILURE_SENTINEL + "\nprint('check')\n"
        ok_code = "print('check: 42')\n"
        engine = cr.ScriptedDecisionEngine([
            ("execute_code", {"code": fail_code}),
            ("none", {}),
            ("execute_code", {"code": ok_code}),
            ("none", {}),
        ])
        client = _RecordingClient(cr.ScriptedChatClient([
            cr.ScriptTurn(content=HINT_FINAL),
        ]))
        executor = _fallback_executor(cr, te, registry, engine)
        seen: list[str] = []
        with cr.scripted_chat_backend(client):
            result = executor.execute_with_tools(
                message=HINT_MSG,
                on_tool_call=lambda r: seen.append(r.tool_name),
            )

        assert seen == ["execute_code", "execute_code"], seen
        assert result.response == HINT_FINAL
        assert engine.leftover == 0 and engine.overrun == 0
        assert client.leftover == 0 and client.overrun == 0
        assert len(engine.captured) == 4, len(engine.captured)
        engine_contents = [_contents(c) for c in engine.captured]
        assert [c.count(EXEC_HINT) for c in engine_contents] == [
            0, 0, 1, 1,
        ], [c.count(EXEC_HINT) for c in engine_contents]
        assert engine_contents[1].count(EXEC_FAILURE_MARKER) == 1
        assert len(client.captured) == 1
        assert _contents(client.captured[0]).count(EXEC_HINT) == 1
        assert EXEC_HINT not in result.response
        assert EXEC_FAILURE_MARKER not in result.response

        # Growth: on a four-call chain the engine's decision context
        # grows strictly at every round -- the non-native path measured.
        fixture = {
            f"status_{i}.txt": f"segment {i}: nominal\n" for i in (1, 2, 3, 4)
        }
        engine4 = cr.ScriptedDecisionEngine(
            [
                ("read_file", {"filename": f"status_{i}.txt"})
                for i in (1, 2, 3, 4)
            ] + [("none", {})],
        )
        client4 = _RecordingClient(cr.ScriptedChatClient([
            cr.ScriptTurn(content=CHAIN_FINAL),
        ]))
        executor4 = _fallback_executor(
            cr, te, cr.build_scripted_registry(cr.VirtualWorkspace(fixture)),
            engine4,
        )
        with cr.scripted_chat_backend(client4):
            chain_result = executor4.execute_with_tools(message=CHAIN_MSG)
        assert chain_result.response == CHAIN_FINAL
        assert [c.tool_name for c in chain_result.tool_calls] == [
            "read_file",
        ] * 4
        assert engine4.leftover == 0 and engine4.overrun == 0
        sizes = [len(_contents(c)) for c in engine4.captured]
        assert len(sizes) == 5, sizes
        assert all(b > a for a, b in zip(sizes, sizes[1:])), sizes
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
