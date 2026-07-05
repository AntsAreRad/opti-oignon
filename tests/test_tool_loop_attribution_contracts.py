#!/usr/bin/env python3
"""Contracts for the tool loop's user-facing hygiene and progress reporting.

The ReAct loop feeds tool results back to the model as plain chat text. These
contracts pin the properties that keep that seam honest for the user:

  * Contract 1 -- the final answer is generated under an explicit attribution
    system message, and tool results are framed as environment output produced
    by the assistant's own calls (never as something the user said or did).
  * Contract 2 -- the per-round decision messages use the same environment
    framing for prior results.
  * Contract 3 -- a caller-provided progress callback fires for each tool call
    DURING the loop, before the final answer exists.
  * Contract 4 -- the no-progress guard catches an alternating repeat
    (A-B-A-B), not only an immediate one, while a normal non-repeating
    multi-tool run still completes.
  * Contract 5 -- the response hygiene helpers: internal marker lines are
    stripped outside fenced code blocks and preserved inside them, and the
    misattribution detector flags second-person action claims in French and
    English without flagging first-person or neutral text.
  * Contract 6 -- the loop's final response is passed through the marker
    stripper, so an echoed internal line never reaches the user.
  * Contract 7 -- the keyword gate matches accented French phrasings.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. Heavy dependencies are stubbed so the loop
loads in isolation; when pydantic is absent a minimal attribute-bag stand-in
is installed (the real package, when installed, is left untouched).
"""

import importlib.util
import sys
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
# Isolated loading: real tool_calling + tool_executor + response_hygiene,
# stubbed heavy deps. Saves/restores sys.modules for sibling suites.
# ---------------------------------------------------------------------------
def _load():
    keys = (
        "pydantic", "ollama", "opti_oignon", "opti_oignon.tool_calling",
        "opti_oignon.tool_registry", "opti_oignon.structured_output",
        "opti_oignon.response_hygiene", "opti_oignon.tool_executor",
        "opti_oignon.config",
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

    rh = None
    rh_path = _OO / "response_hygiene.py"
    if rh_path.exists():
        spec_rh = importlib.util.spec_from_file_location(
            "opti_oignon.response_hygiene", rh_path,
        )
        rh = importlib.util.module_from_spec(spec_rh)
        sys.modules["opti_oignon.response_hygiene"] = rh
        spec_rh.loader.exec_module(rh)

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

    return tc, te, rh, ollama_stub, restore


def _param(required=True):
    return types.SimpleNamespace(
        type="string", required=required, default=None, description="",
    )


def _make_registry(executed, extra_tools=()):
    def wf(**kw):
        executed.append(("write_file", dict(kw)))
        return f"wrote {kw.get('filename', '?')}"

    def ec(**kw):
        executed.append(("execute_code", dict(kw)))
        return f"ran {kw.get('filename', '?')}"

    tools = {
        "write_file": types.SimpleNamespace(
            name="write_file", description="write", enabled=True,
            parameters={"filename": _param(), "content": _param(False)},
            handler=wf),
        "execute_code": types.SimpleNamespace(
            name="execute_code", description="run", enabled=True,
            parameters={"filename": _param()}, handler=ec),
    }
    for name in extra_tools:
        def _h(_name=name, **kw):
            executed.append((_name, dict(kw)))
            return f"ok {_name}"
        tools[name] = types.SimpleNamespace(
            name=name, description=name, enabled=True,
            parameters={"filename": _param(False)}, handler=_h)

    class FakeRegistry:
        def list_available(self):
            return list(tools.values())

        def get(self, name):
            return tools.get(name)

        def get_tools_prompt(self):
            return "tools: " + ", ".join(sorted(tools))

        def is_available(self, name):
            return name in tools

    return FakeRegistry()


def _native_call(name, arguments):
    return types.SimpleNamespace(
        message=types.SimpleNamespace(
            tool_calls=[types.SimpleNamespace(
                function=types.SimpleNamespace(name=name, arguments=arguments),
            )],
            content="",
        ),
    )


def _native_done(content=""):
    return types.SimpleNamespace(
        message=types.SimpleNamespace(tool_calls=[], content=content),
    )


_MODEL = "qwen2.5-coder:14b-16k"


# ---------------------------------------------------------------------------
# Contract 1: final answer under attribution system message + environment
# framing (no "Here are the tool results:" user-voiced block)
# ---------------------------------------------------------------------------
def test_final_answer_generated_under_attribution_system_message():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        executed = []
        ex = te.ToolExecutor(
            registry=_make_registry(executed), structured_engine=None,
            default_model=_MODEL,
        )
        captured_final = {}
        seq = [
            _native_call("write_file", {"filename": "sieve.py", "content": "x"}),
            _native_done(),
        ]

        def chat(**kw):
            if "tools" in kw:
                return seq.pop(0) if seq else _native_done()
            captured_final["messages"] = kw.get("messages", [])
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content="I created sieve.py."))

        ollama_stub.chat = chat
        result = ex.execute_with_tools("create sieve.py", model=_MODEL)

        msgs = captured_final.get("messages", [])
        assert msgs, "final generation was never reached"
        sys_msgs = [m for m in msgs if m.get("role") == "system"]
        assert len(sys_msgs) == 1, f"expected one system message, got {len(sys_msgs)}"
        guard = sys_msgs[0]["content"].lower()
        assert "never attribute" in guard and "tool" in guard, guard
        assert "do not mention" in guard, guard

        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert user_msgs, "no user message in final generation"
        body = user_msgs[-1]["content"]
        assert "Here are the tool results:" not in body, body
        assert "[environment]" in body, body
        assert "assistant" in body.lower(), body
        assert result.response == "I created sieve.py."
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 2: decision rounds frame prior results as environment output
# ---------------------------------------------------------------------------
def test_decision_round_uses_environment_framing_for_prior_results():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        executed = []
        ex = te.ToolExecutor(
            registry=_make_registry(executed), structured_engine=None,
            default_model=_MODEL,
        )
        captured_rounds = []

        def chat(**kw):
            if "tools" in kw:
                captured_rounds.append(kw.get("messages", []))
                if len(captured_rounds) == 1:
                    return _native_call(
                        "write_file", {"filename": "a.py", "content": "x"})
                return _native_done()
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content="done"))

        ollama_stub.chat = chat
        ex.execute_with_tools("make a.py", model=_MODEL)

        assert len(captured_rounds) >= 2, "second decision round never reached"
        body = captured_rounds[1][-1]["content"]
        assert "Previous tool results:" not in body, body
        assert "[environment]" in body, body
        assert "assistant" in body.lower(), body
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 3: progress callback fires per tool call, during the loop
# ---------------------------------------------------------------------------
def test_progress_callback_fires_during_loop_before_final_answer():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        executed = []
        ex = te.ToolExecutor(
            registry=_make_registry(executed), structured_engine=None,
            default_model=_MODEL,
        )
        events = []
        seq = [
            _native_call("write_file", {"filename": "a.py", "content": "x"}),
            _native_call("execute_code", {"filename": "a.py"}),
            _native_done(),
        ]

        def chat(**kw):
            if "tools" in kw:
                return seq.pop(0) if seq else _native_done()
            events.append(("final", None))
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content="done"))

        ollama_stub.chat = chat
        result = ex.execute_with_tools(
            "make and run a.py", model=_MODEL,
            on_tool_call=lambda call: events.append(("tool", call.tool_name)),
        )

        assert events == [
            ("tool", "write_file"), ("tool", "execute_code"), ("final", None),
        ], events
        assert [c.tool_name for c in result.tool_calls] == [
            "write_file", "execute_code",
        ]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 4: alternating repeat (A-B-A) stops; distinct sequence completes
# ---------------------------------------------------------------------------
def test_no_progress_guard_catches_alternating_repeat():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        executed = []
        ex = te.ToolExecutor(
            registry=_make_registry(executed), structured_engine=None,
            default_model=_MODEL, max_tool_calls=6,
        )
        call_a = ("write_file", {"filename": "a.py", "content": "x"})
        call_b = ("execute_code", {"filename": "a.py"})
        seq = [
            _native_call(*call_a),
            _native_call(*call_b),
            _native_call(*call_a),
            _native_call(*call_b),
            _native_call(*call_a),
            _native_done(),
        ]

        def chat(**kw):
            if "tools" in kw:
                return seq.pop(0) if seq else _native_done()
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content="done"))

        ollama_stub.chat = chat
        result = ex.execute_with_tools("loop", model=_MODEL)
        assert len(result.tool_calls) == 2, (
            f"alternating repeat not stopped: "
            f"{[c.tool_name for c in result.tool_calls]}"
        )
        assert [name for name, _ in executed] == ["write_file", "execute_code"]

        # A distinct three-tool sequence is NOT over-blocked.
        executed2 = []
        ex2 = te.ToolExecutor(
            registry=_make_registry(executed2, extra_tools=("list_files",)),
            structured_engine=None, default_model=_MODEL, max_tool_calls=6,
        )
        seq2 = [
            _native_call("write_file", {"filename": "b.py", "content": "y"}),
            _native_call("execute_code", {"filename": "b.py"}),
            _native_call("list_files", {}),
            _native_done(),
        ]

        def chat2(**kw):
            if "tools" in kw:
                return seq2.pop(0) if seq2 else _native_done()
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content="done"))

        ollama_stub.chat = chat2
        result2 = ex2.execute_with_tools("three steps", model=_MODEL)
        assert len(result2.tool_calls) == 3, (
            f"distinct sequence over-blocked: "
            f"{[c.tool_name for c in result2.tool_calls]}"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 5: hygiene helpers -- marker stripping and misattribution detection
# ---------------------------------------------------------------------------
def test_hygiene_helpers_strip_markers_and_detect_misattribution():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        assert rh is not None, "response_hygiene module is missing"

        text = (
            "Here is the outcome.\n"
            "[environment] tool call by assistant: write_file\n"
            "[Tool: write_file] Arguments: {'filename': 'a.py'}\n"
            "[Verification] The last code execution reported an error.\n"
            "[2 rounds remain; converge]\n"
            "The script now runs cleanly.\n"
            "```\n"
            "[Tool: kept_inside_fence]\n"
            "```\n"
        )
        cleaned, dropped = rh.strip_internal_markers(text)
        assert dropped == 4, (dropped, cleaned)
        assert "Here is the outcome." in cleaned
        assert "The script now runs cleanly." in cleaned
        assert "[environment]" not in cleaned
        assert "[Verification]" not in cleaned
        assert "rounds remain" not in cleaned
        assert "[Tool: kept_inside_fence]" in cleaned
        assert "[Tool: write_file]" not in cleaned

        clean_text = "All good.\nNothing internal here."
        same, zero = rh.strip_internal_markers(clean_text)
        assert zero == 0 and same == clean_text

        hits_fr = rh.detect_misattribution(
            "Bravo, tu as créé le fichier et il fonctionne.")
        assert hits_fr, "French second-person action claim not detected"
        hits_en = rh.detect_misattribution(
            "Great, you created the file and it works.")
        assert hits_en, "English second-person action claim not detected"
        assert rh.detect_misattribution(
            "I created the file for you and ran it.") == []
        assert rh.detect_misattribution(
            "Tu as demandé un script, le voici.") == []
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 6: the loop strips echoed internal markers from the final answer
# ---------------------------------------------------------------------------
def test_final_answer_is_stripped_of_echoed_internal_markers():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        executed = []
        ex = te.ToolExecutor(
            registry=_make_registry(executed), structured_engine=None,
            default_model=_MODEL,
        )
        seq = [
            _native_call("write_file", {"filename": "a.py", "content": "x"}),
            _native_done(),
        ]

        def chat(**kw):
            if "tools" in kw:
                return seq.pop(0) if seq else _native_done()
            leaked = (
                "Done.\n"
                "[Tool: write_file] Result: wrote a.py\n"
                "The file is ready."
            )
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content=leaked))

        ollama_stub.chat = chat
        result = ex.execute_with_tools("make a.py", model=_MODEL)
        assert "[Tool:" not in result.response, result.response
        assert "Done." in result.response
        assert "The file is ready." in result.response
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 7: the keyword gate matches accented French phrasings
# ---------------------------------------------------------------------------
def test_keyword_gate_matches_accented_french():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        executed = []
        ex = te.ToolExecutor(
            registry=_make_registry(executed), structured_engine=None,
            default_model=_MODEL,
        )
        assert ex.should_use_tools("Exécute ce code s'il te plaît") is True
        # ASCII phrasing keeps working (no regression).
        assert ex.should_use_tools("execute ce code maintenant") is True
        # A message with no tool intent stays False.
        assert ex.should_use_tools("Bonjour, comment vas-tu ?") is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 8: the incremental stream filter drops scaffold lines across
# arbitrary chunk boundaries and never stalls long prose
# ---------------------------------------------------------------------------
def test_stream_filter_drops_markers_across_chunk_boundaries():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        assert rh is not None, "response_hygiene module is missing"
        assert hasattr(rh, "StreamMarkerFilter"), "StreamMarkerFilter missing"

        filt = rh.StreamMarkerFilter()
        emitted = filt.feed("Done.\n[To")
        emitted += filt.feed("ol: write_file] Result: leaked\nAll go")
        emitted += filt.feed("od.")
        emitted += filt.flush()
        assert emitted == "Done.\nAll good.", repr(emitted)
        assert filt.dropped == 1, filt.dropped

        # Fenced content passes verbatim, including scaffold-looking lines.
        filt2 = rh.StreamMarkerFilter()
        text = "```\n[Tool: kept]\n```\nok"
        out = filt2.feed(text) + filt2.flush()
        assert out == text, repr(out)
        assert filt2.dropped == 0

        # Long prose without a newline starts flowing before the line ends.
        filt3 = rh.StreamMarkerFilter()
        head = filt3.feed("This is a long ordinary sentence that keeps going "
                          "well past the hold window without any newline")
        assert head.startswith("This is a long"), repr(head)

        # A bracketed but non-scaffold line is kept once resolved.
        filt4 = rh.StreamMarkerFilter()
        out4 = filt4.feed("[note] hello\n") + filt4.flush()
        assert out4 == "[note] hello\n", repr(out4)
        assert filt4.dropped == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 9: stream_with_tools streams the final answer in multiple chunks
# under the same attribution framing, filters echoed markers, and returns a
# result whose response equals exactly the emitted text
# ---------------------------------------------------------------------------
def test_stream_with_tools_streams_filtered_final_answer():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        executed = []
        ex = te.ToolExecutor(
            registry=_make_registry(executed), structured_engine=None,
            default_model=_MODEL,
        )
        assert hasattr(ex, "stream_with_tools"), "stream_with_tools missing"

        captured_final = {}
        decision_seq = [
            _native_call("write_file", {"filename": "a.py", "content": "x"}),
            _native_done(),
        ]
        final_pieces = ["I created", " a.py.\n[Tool: write_file] echo\n", "It runs."]

        def chat(**kw):
            if "tools" in kw:
                return decision_seq.pop(0) if decision_seq else _native_done()
            if kw.get("stream"):
                captured_final["messages"] = kw.get("messages", [])

                def gen():
                    for piece in final_pieces:
                        yield types.SimpleNamespace(
                            message=types.SimpleNamespace(content=piece))
                return gen()
            return types.SimpleNamespace(
                message=types.SimpleNamespace(content="".join(final_pieces)))

        ollama_stub.chat = chat

        events = []
        chunks = []
        gen = ex.stream_with_tools(
            "make a.py", model=_MODEL,
            on_tool_call=lambda call: events.append(call.tool_name),
        )
        result = None
        try:
            while True:
                chunks.append(next(gen))
        except StopIteration as stop:
            result = stop.value

        emitted = "".join(chunks)
        assert len(chunks) >= 2, chunks
        assert events == ["write_file"], events
        assert "[Tool:" not in emitted, emitted
        assert "I created a.py." in emitted and "It runs." in emitted, emitted
        assert result is not None, "no result returned by the generator"
        assert result.response == emitted, (result.response, emitted)
        assert [c.tool_name for c in result.tool_calls] == ["write_file"]

        msgs = captured_final.get("messages", [])
        assert msgs and msgs[0].get("role") == "system", msgs
        assert "never attribute" in msgs[0]["content"].lower()
        assert "[environment]" in msgs[-1]["content"], msgs[-1]["content"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Contract 10: the tools pipeline streams chunks live when the tool executor
# offers streaming, and keeps the single-shot path otherwise
# ---------------------------------------------------------------------------
def test_tools_pipeline_streams_when_available_else_single_shot():
    tc, te, rh, ollama_stub, restore = _load()
    try:
        spec_ae = importlib.util.spec_from_file_location(
            "opti_oignon.agentic_executor", _OO / "agentic_executor.py",
        )
        ae = importlib.util.module_from_spec(spec_ae)
        sys.modules["opti_oignon.agentic_executor"] = ae
        spec_ae.loader.exec_module(ae)

        call_shape = types.SimpleNamespace(
            tool_name="write_file", arguments={}, result="ok",
            success=True, execution_time=0.0, reasoning="", retryable=False,
        )

        class StreamingFake:
            default_model = _MODEL

            def stream_with_tools(self, **kw):
                notify = kw.get("on_tool_call")
                if notify is not None:
                    notify(call_shape)
                yield "part one "
                yield "part two"
                return types.SimpleNamespace(
                    response="part one part two",
                    tool_calls=[call_shape], model=_MODEL, total_time=0.0,
                )

        agent = ae.AgenticExecutor(
            executor=None, tool_executor=StreamingFake(),
            structured_engine=None, verification_engine=None,
        )
        routing = types.SimpleNamespace(model=_MODEL)
        chunks = list(agent._execute_tools_pipeline(
            "make a.py", routing, None, None,
        ))
        assert chunks == ["part one ", "part two"], chunks
        assert [c.tool_name for c in agent.last_tool_calls] == ["write_file"]

        class LegacyFake:
            default_model = _MODEL

            def execute_with_tools(self, **kw):
                return types.SimpleNamespace(
                    response="single shot", tool_calls=[call_shape],
                    model=_MODEL, total_time=0.0,
                )

        agent2 = ae.AgenticExecutor(
            executor=None, tool_executor=LegacyFake(),
            structured_engine=None, verification_engine=None,
        )
        chunks2 = list(agent2._execute_tools_pipeline(
            "make a.py", routing, None, None,
        ))
        assert chunks2 == ["single shot"], chunks2
    finally:
        sys.modules.pop("opti_oignon.agentic_executor", None)
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print("-" * 48)
    print(f"{len(tests)} selected, {failed} failed")
    sys.exit(1 if failed else 0)
