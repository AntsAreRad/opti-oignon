#!/usr/bin/env python3
"""Tests for the robust tool-calling cycle (Lot 1 + Lot 2).

Two layers of coverage:

  * Primitives (`tool_calling.py`): native schema building, native tool-call
    parsing across response forms, enum-forcing, the intent transpiler, and the
    capability gate. Pure stdlib -- no Ollama, no registry.

  * Wiring (`tool_executor.execute_with_tools`): the ReAct loop is loaded in
    isolation with a controllable fake Ollama and a fake registry, and three
    paths are exercised end to end -- native function-call fires; native
    declines and the salvage transpiles narrated code into real tool calls; a
    non-capable model falls back to the format= path with no regression.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Isolated loading: real tool_calling + tool_executor, stubbed heavy deps
# ---------------------------------------------------------------------------
def _load():
    """Return (tool_calling, tool_executor, ollama_stub) loaded in isolation.

    Saves/restores sys.modules so the suite stays clean for sibling tests.
    """
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

    return tc, te, ollama_stub, restore


def _param(required=True):
    return types.SimpleNamespace(
        type="string", required=required, default=None, description="",
    )


# ---------------------------------------------------------------------------
# Primitive tests
# ---------------------------------------------------------------------------
def test_native_tool_schemas():
    tc, _, _, restore = _load()
    try:
        tools = [
            types.SimpleNamespace(
                name="write_file", description="Write",
                parameters={"filename": _param(), "content": _param()},
            ),
            types.SimpleNamespace(
                name="execute_code", description="Run",
                parameters={"filename": _param(), "timeout": _param(False)},
            ),
            types.SimpleNamespace(
                name="list_files", description="List",
                parameters={"paths": types.SimpleNamespace(
                    type="list", required=False, default=None, description="")},
            ),
        ]
        s = tc.native_tool_schemas(tools)
        assert s[0]["function"]["name"] == "write_file"
        assert s[0]["function"]["parameters"]["required"] == ["filename", "content"]
        assert s[1]["function"]["parameters"]["required"] == ["filename"]
        paths = s[2]["function"]["parameters"]["properties"]["paths"]
        assert paths["type"] == "array" and paths["items"]["type"] == "string"
    finally:
        restore()


def test_parse_native_tool_calls():
    tc, _, _, restore = _load()
    try:
        obj = types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=[
            types.SimpleNamespace(function=types.SimpleNamespace(
                name="write_file", arguments={"filename": "a.py"})),
            types.SimpleNamespace(function=types.SimpleNamespace(
                name="execute_code", arguments='{"filename": "a.py"}')),
        ]))
        assert tc.parse_native_tool_calls(obj) == [
            ("write_file", {"filename": "a.py"}),
            ("execute_code", {"filename": "a.py"}),
        ]
        d = {"message": {"tool_calls": [
            {"function": {"name": "list_files", "arguments": {}}},
            {"function": {"name": "", "arguments": {}}},
        ]}}
        assert tc.parse_native_tool_calls(d) == [("list_files", {})]
        assert tc.parse_native_tool_calls({"message": {}}) == []
    finally:
        restore()


def test_forced_decision_schema():
    tc, _, _, restore = _load()
    try:
        fs = tc.forced_decision_schema(["write_file", "execute_code"])
        assert fs["properties"]["tool_name"]["enum"] == ["write_file", "execute_code"]
        assert "none" not in fs["properties"]["tool_name"]["enum"]
        try:
            tc.forced_decision_schema([])
            raise AssertionError("empty tool list must raise")
        except ValueError:
            pass
    finally:
        restore()


def test_transpile_intent():
    tc, _, _, restore = _load()
    try:
        sample = (
            "### Fichier `sieve.py`\n```python\nimport math\nprint(1)\n```\nVoila."
        )
        got = tc.transpile_intent(
            sample, "cree sieve.py et execute-le",
            available={"write_file", "execute_code"},
        )
        assert got[0] == ("write_file", {
            "filename": "sieve.py", "content": "import math\nprint(1)\n"})
        assert got[1] == ("execute_code", {"filename": "sieve.py"})
        # available filtering
        only_w = tc.transpile_intent(sample, "run it", available={"write_file"})
        assert [c[0] for c in only_w] == ["write_file"]
        # no code -> nothing
        assert tc.transpile_intent("prose only", "x") == []
    finally:
        restore()


def test_model_supports_native_tools():
    tc, _, _, restore = _load()
    try:
        assert tc.model_supports_native_tools("qwen2.5-coder:14b-16k") is True
        assert tc.model_supports_native_tools("qwen3.6:35b-a3b-16k") is True
        assert tc.model_supports_native_tools("random-model:7b") is False
        assert tc.model_supports_native_tools("x", capability_lookup=lambda m: True) is True
        assert tc.model_supports_native_tools("qwen2.5", capability_lookup=lambda m: False) is False
        assert tc.model_supports_native_tools("qwen2.5", capability_lookup=lambda m: None) is True
    finally:
        restore()


# ---------------------------------------------------------------------------
# Wiring integration
# ---------------------------------------------------------------------------
def _make_executor(te, ollama_stub, executed):
    p = _param

    def wf(**kw):
        executed.append(("write_file", kw))
        return f"wrote {kw['filename']}"

    def ec(**kw):
        executed.append(("execute_code", kw))
        return f"ran {kw['filename']}"

    tools = {
        "write_file": types.SimpleNamespace(
            name="write_file", description="write", enabled=True,
            parameters={"filename": p(), "content": p()}, handler=wf),
        "execute_code": types.SimpleNamespace(
            name="execute_code", description="run", enabled=True,
            parameters={"filename": p()}, handler=ec),
    }

    class FakeRegistry:
        def list_available(self):
            return list(tools.values())

        def get(self, name):
            return tools.get(name)

        def get_tools_prompt(self):
            return "tools: write_file, execute_code"

    return te.ToolExecutor(
        registry=FakeRegistry(), structured_engine=None,
        default_model="qwen2.5-coder:14b-16k",
    )


def test_wiring_native_fires():
    _, te, ollama_stub, restore = _load()
    try:
        executed = []
        ex = _make_executor(te, ollama_stub, executed)
        seq = [
            types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=[
                types.SimpleNamespace(function=types.SimpleNamespace(
                    name="write_file",
                    arguments={"filename": "sieve.py", "content": "print(1)\n"})),
            ], content="")),
            types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=[], content="done")),
        ]

        def chat(**kw):
            if "tools" in kw:
                return seq.pop(0) if seq else types.SimpleNamespace(
                    message=types.SimpleNamespace(tool_calls=[], content="done"))
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content="created sieve.py"))

        ollama_stub.chat = chat
        r = ex.execute_with_tools("create sieve.py", model="qwen2.5-coder:14b-16k")
        assert [c.tool_name for c in r.tool_calls] == ["write_file"]
        assert executed == [("write_file", {"filename": "sieve.py", "content": "print(1)\n"})]
    finally:
        restore()


def test_wiring_salvage_on_narration():
    _, te, ollama_stub, restore = _load()
    try:
        executed = []
        ex = _make_executor(te, ollama_stub, executed)

        def chat(**kw):
            if "tools" in kw:  # native declines
                return types.SimpleNamespace(
                    message=types.SimpleNamespace(tool_calls=[], content=""))
            return types.SimpleNamespace(message=types.SimpleNamespace(content=(
                "### Fichier `sieve.py`\n```python\nimport math\nprint(2)\n```\nVoila.")))

        ollama_stub.chat = chat
        r = ex.execute_with_tools(
            "cree sieve.py et execute-le", model="qwen2.5-coder:14b-16k")
        assert [c.tool_name for c in r.tool_calls] == ["write_file", "execute_code"]
        assert executed[0][1]["filename"] == "sieve.py"
        assert "import math" in executed[0][1]["content"]
    finally:
        restore()


def test_wiring_non_native_fallback_no_regression():
    _, te, ollama_stub, restore = _load()
    try:
        executed = []
        ex = _make_executor(te, ollama_stub, executed)
        ollama_stub.chat = lambda **kw: types.SimpleNamespace(
            message=types.SimpleNamespace(content="plain answer, no code"))
        r = ex.execute_with_tools("hello", model="some-unknown-model:7b")
        assert r.tool_calls == [] and executed == []
        assert r.response == "plain answer, no code"
    finally:
        restore()


def test_repair_arguments():
    tc, _, _, restore = _load()
    try:
        names = ["filename", "content"]
        got = tc.repair_arguments(names, {"path": "a.py", "code": "x"})
        assert got["filename"] == "a.py" and got["content"] == "x"
        # normalized exact match (case / separators)
        assert tc.repair_arguments(["filename"], {"FileName": "a.py"})["filename"] == "a.py"
        # already present -> not overwritten by an alias
        keep = tc.repair_arguments(["filename"], {"filename": "ok", "path": "no"})
        assert keep["filename"] == "ok"
        # non-dict input
        assert tc.repair_arguments(["x"], None) == {}
    finally:
        restore()


def _flaky_executor(te, tools):
    class Reg:
        def list_available(self):
            return list(tools.values())

        def get(self, n):
            return tools.get(n)

        def get_tools_prompt(self):
            return "tools"

    return te.ToolExecutor(
        registry=Reg(), structured_engine=None,
        default_model="qwen2.5-coder:14b-16k",
    )


def _native_call(name, args):
    return types.SimpleNamespace(message=types.SimpleNamespace(
        tool_calls=[types.SimpleNamespace(function=types.SimpleNamespace(
            name=name, arguments=args))], content=""))


def test_wiring_arg_repair():
    _, te, ollama_stub, restore = _load()
    try:
        executed = []
        ex = _make_executor(te, ollama_stub, executed)
        seq = [
            _native_call("write_file", {"path": "a.py", "code": "print(1)\n"}),
            types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=[], content="done")),
        ]

        def chat(**kw):
            if "tools" in kw:
                return seq.pop(0) if seq else types.SimpleNamespace(
                    message=types.SimpleNamespace(tool_calls=[], content="done"))
            return types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))

        ollama_stub.chat = chat
        r = ex.execute_with_tools("make a.py", model="qwen2.5-coder:14b-16k")
        # wrong keys (path/code) repaired to filename/content -> call succeeds
        assert [c.tool_name for c in r.tool_calls] == ["write_file"]
        assert r.tool_calls[0].success is True
        assert executed == [("write_file", {"filename": "a.py", "content": "print(1)\n"})]
    finally:
        restore()


def test_wiring_error_feedback_retry():
    _, te, ollama_stub, restore = _load()
    try:
        executed = []
        state = {"n": 0}

        def wf(**kw):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("boom")
            executed.append(("write_file", kw))
            return "wrote"

        tools = {"write_file": types.SimpleNamespace(
            name="write_file", description="w", enabled=True,
            parameters={"filename": _param(), "content": _param()}, handler=wf)}
        ex = _flaky_executor(te, tools)
        seq = [
            _native_call("write_file", {"filename": "a.py", "content": "x"}),
            _native_call("write_file", {"filename": "a.py", "content": "x"}),
            types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=[], content="done")),
        ]

        def chat(**kw):
            if "tools" in kw:
                return seq.pop(0) if seq else types.SimpleNamespace(
                    message=types.SimpleNamespace(tool_calls=[], content="done"))
            return types.SimpleNamespace(message=types.SimpleNamespace(content="final"))

        ollama_stub.chat = chat
        r = ex.execute_with_tools("make a.py", model="qwen2.5-coder:14b-16k")
        # first failed (retryable) but the loop continued and the retry succeeded
        assert [c.success for c in r.tool_calls] == [False, True]
        assert r.tool_calls[0].retryable is True
        assert executed == [("write_file", {"filename": "a.py", "content": "x"})]
    finally:
        restore()


def test_wiring_hard_failure_stops():
    _, te, ollama_stub, restore = _load()
    try:
        executed = []
        ex = _make_executor(te, ollama_stub, executed)
        seq = [
            _native_call("write_file", {"filename": "a.py", "content": "x"}),
            _native_call("write_file", {"filename": "a.py", "content": "x"}),
        ]

        def chat(**kw):
            if "tools" in kw:
                return seq.pop(0) if seq else types.SimpleNamespace(
                    message=types.SimpleNamespace(tool_calls=[], content="done"))
            return types.SimpleNamespace(message=types.SimpleNamespace(content="final"))

        ollama_stub.chat = chat
        # approval denies -> hard (non-retryable) failure -> stop after one call
        r = ex.execute_with_tools(
            "make a.py", model="qwen2.5-coder:14b-16k",
            approval_fn=lambda n, a: False)
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].success is False and r.tool_calls[0].retryable is False
        assert executed == []
    finally:
        restore()


def test_wiring_retry_budget():
    _, te, ollama_stub, restore = _load()
    try:
        def wf(**kw):
            raise RuntimeError("always")

        tools = {"write_file": types.SimpleNamespace(
            name="write_file", description="w", enabled=True,
            parameters={"filename": _param(), "content": _param()}, handler=wf)}
        ex = _flaky_executor(te, tools)
        ex.max_tool_retries = 2
        call = _native_call("write_file", {"filename": "a.py", "content": "x"})

        def chat(**kw):
            if "tools" in kw:
                return call  # every decision returns the failing call
            return types.SimpleNamespace(message=types.SimpleNamespace(content="final"))

        ollama_stub.chat = chat
        r = ex.execute_with_tools("make a.py", model="qwen2.5-coder:14b-16k")
        # bounded: stops after max_tool_retries consecutive retryable failures
        assert len(r.tool_calls) == 2
        assert all(c.retryable for c in r.tool_calls)
    finally:
        restore()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'} - {failures} failure(s)")
    sys.exit(1 if failures else 0)
