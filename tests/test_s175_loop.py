#!/usr/bin/env python3
"""Tests for S175 -- the agent loop (Theme 3 / Odysseus Core).

Exercises ``opti_oignon/agent/loop.py`` and the package facade
``opti_oignon/agent/__init__.py``: the multi-turn streaming loop (round cap
honoured, terminates on a no-tool-call final answer, content accumulated across
chunks), the sandbox dispatch invariant inside the loop (tools route through the
injected session; refusal when bwrap is unavailable), per-mode gating (Bulbe
approval), untrusted-context wrapping of observations and the memory working
block, the bounded verifier (never exceeds the reference cap), and the
guarantee that nothing raises into the conversation path. The model client and
sandbox are injected. Loaded in isolation via ``spec_from_file_location`` with
``opti_oignon`` stubbed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"


def _ensure_pkg():
    if "opti_oignon" not in sys.modules:
        pkg = types.ModuleType("opti_oignon")
        pkg.__path__ = [str(OO)]
        sys.modules["opti_oignon"] = pkg
    if "opti_oignon.agent" not in sys.modules:
        apkg = types.ModuleType("opti_oignon.agent")
        apkg.__path__ = [str(AGENT)]
        sys.modules["opti_oignon.agent"] = apkg


def _ensure_agent(name: str):
    full = f"opti_oignon.agent.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(AGENT / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg()
_ensure_agent("tool_parsing")
_ensure_agent("allowlists")
_ensure_agent("dispatch")
_ensure_agent("untrusted_context")
L = _ensure_agent("loop")


# Fakes


class FakeManager:
    def __init__(self, bwrap: bool):
        self.bwrap_available = bwrap


class FakeSession:
    def __init__(self, bwrap: bool = True, active: bool = True):
        self.sandbox_manager = FakeManager(bwrap)
        self.active = active
        self.calls: list[tuple] = []

    def bash(self, command, timeout=30):
        self.calls.append(("bash", command))
        return f"out: {command}"

    def view(self, path, start_line=0, end_line=0):
        self.calls.append(("view", path))
        return f"view {path}"

    def create_file(self, path, content):
        self.calls.append(("create_file", path))
        return f"wrote {path}"

    def str_replace(self, path, old_str, new_str=""):
        self.calls.append(("str_replace", path))
        return "ok"


def _native(name, args):
    return [{"function": {"name": name, "arguments": args}}]


class ScriptedClient:
    """Yields chunks per round from a script; splits content across two chunks."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.seen = []

    def stream(self, messages, tools=None):
        self.seen.append(list(messages))
        step = self.script[self.i] if self.i < len(self.script) else {"content": "done.", "tool_calls": None}
        self.i += 1
        content = step.get("content", "")
        half = len(content) // 2
        yield {"message": {"content": content[:half]}}
        yield {"message": {"content": content[half:], "tool_calls": step.get("tool_calls")}}


class AlwaysToolClient:
    def stream(self, messages, tools=None):
        yield {"message": {"content": "again", "tool_calls": _native("view", {"path": "/x"})}}


class BoomClient:
    def stream(self, messages, tools=None):
        raise RuntimeError("model crashed")


class SingleDictClient:
    """Returns a single response dict (non-streaming) instead of an iterator."""

    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def stream(self, messages, tools=None):
        return {"message": {"content": self.content, "tool_calls": self.tool_calls}}


# Module shape


class TestModuleShape:
    def test_sentinels(self):
        assert L.checkpoint_before_apply is True
        assert L.FEATURE_AVAILABLE is True

    def test_round_caps(self):
        assert L.MAX_AGENT_ROUNDS == 20
        assert L._VERIFIER_MAX_ROUNDS == 2

    def test_stop_constants(self):
        assert L.STOP_DONE == "done"
        assert L.STOP_MAX_ROUNDS == "max_rounds"
        assert L.STOP_ERROR == "error"


# Loop basics


class TestLoopBasics:
    def test_single_final_answer_no_tools(self):
        client = ScriptedClient([{"content": "Here is the answer.", "tool_calls": None}])
        res = L.run("q", model_client=client, sandbox=FakeSession(), mode="daily")
        assert res.stop_reason == L.STOP_DONE
        assert res.rounds == 1
        assert res.final_text == "Here is the answer."
        assert res.tool_results == []

    def test_multi_round_tool_then_final(self):
        script = [
            {"content": "running", "tool_calls": _native("bash", {"command": "ls"})},
            {"content": "All done.", "tool_calls": None},
        ]
        sess = FakeSession()
        res = L.run("list", model_client=ScriptedClient(script), sandbox=sess, mode="daily")
        assert res.rounds == 2
        assert res.stop_reason == L.STOP_DONE
        assert res.final_text == "All done."
        assert [(r.tool_name, r.executed) for r in res.tool_results] == [("bash", True)]
        assert sess.calls == [("bash", "ls")]

    def test_observation_wrapped_as_untrusted_user_message(self):
        script = [
            {"content": "x", "tool_calls": _native("view", {"path": "/a"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=FakeSession(), mode="daily")
        obs = [m for m in res.messages if m["role"] == "user" and "untrusted_data" in m.get("content", "")]
        assert obs, "tool observation must be wrapped as an untrusted user message"
        assert 'trusted="false"' in obs[0]["content"]
        assert "view" in obs[0]["content"]

    def test_content_accumulated_across_chunks(self):
        # ScriptedClient splits content into two chunks; the loop reassembles it.
        client = ScriptedClient([{"content": "abcdefgh", "tool_calls": None}])
        res = L.run("q", model_client=client, sandbox=FakeSession(), mode="daily")
        assert res.final_text == "abcdefgh"

    def test_single_dict_response_supported(self):
        res = L.run("q", model_client=SingleDictClient("final"), sandbox=FakeSession(), mode="daily")
        assert res.final_text == "final" and res.stop_reason == L.STOP_DONE

    def test_parser_path_tool_call_from_text(self):
        # No native tool_calls; the model writes a bracketed tool block.
        script = [
            {"content": '[TOOL_CALL]{"name":"view","args":{"path":"/p"}}[/TOOL_CALL]', "tool_calls": None},
            {"content": "done", "tool_calls": None},
        ]
        sess = FakeSession()
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess, mode="daily")
        assert sess.calls == [("view", "/p")]
        assert res.tool_results[0].source == "bracketed"

    def test_transcript_roles(self):
        script = [
            {"content": "x", "tool_calls": _native("bash", {"command": "id"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=FakeSession(), mode="daily", include_memory=False)
        roles = [m["role"] for m in res.messages]
        assert roles == ["user", "assistant", "user", "assistant"]


# Round cap


class TestRoundCap:
    def test_cap_honoured(self):
        res = L.run("q", model_client=AlwaysToolClient(), sandbox=FakeSession(), mode="daily", max_rounds=3)
        assert res.rounds == 3
        assert res.stop_reason == L.STOP_MAX_ROUNDS

    def test_clamp_rounds(self):
        assert L._clamp_rounds(-5) == 1
        assert L._clamp_rounds(0) == 1
        assert L._clamp_rounds(5) == 5
        assert L._clamp_rounds("nope") == L.MAX_AGENT_ROUNDS
        assert L._clamp_rounds(10_000) == L._HARD_ROUND_CEILING

    def test_default_cap_is_reference_value(self):
        # The default max_rounds is the Odysseus reference value.
        import inspect

        sig = inspect.signature(L.run)
        assert sig.parameters["max_rounds"].default == L.MAX_AGENT_ROUNDS == 20


# Never raises into the conversation path


class TestNeverRaises:
    def test_model_stream_error_is_observation(self):
        res = L.run("q", model_client=BoomClient(), sandbox=FakeSession(), mode="daily")
        assert res.stop_reason == L.STOP_ERROR
        assert res.rounds == 1
        assert any("model error" in m.get("content", "") for m in res.messages)

    def test_no_model_client_returns_error_result(self):
        res = L.run("q", model_client=None)
        assert res.stop_reason == L.STOP_ERROR
        assert res.rounds == 0

    def test_tool_error_does_not_propagate(self):
        sess = FakeSession()
        sess.bash = lambda command, timeout=30: (_ for _ in ()).throw(RuntimeError("boom"))
        script = [
            {"content": "x", "tool_calls": _native("bash", {"command": "ls"})},
            {"content": "recovered", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess, mode="daily")
        assert res.stop_reason == L.STOP_DONE
        assert res.tool_results[0].reason == "error"

    def test_on_event_exception_is_swallowed(self):
        def bad_observer(event):
            raise RuntimeError("observer broke")

        res = L.run(
            "q",
            model_client=ScriptedClient([{"content": "ok", "tool_calls": None}]),
            sandbox=FakeSession(),
            mode="daily",
            on_event=bad_observer,
        )
        assert res.stop_reason == L.STOP_DONE  # loop completes despite the observer


# Sandbox invariant inside the loop


class TestSandboxInvariantInLoop:
    def test_refusal_when_bwrap_unavailable(self):
        sess = FakeSession(bwrap=False)
        script = [
            {"content": "x", "tool_calls": _native("bash", {"command": "rm -rf /"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=sess, mode="daily")
        assert [r.reason for r in res.tool_results] == ["sandbox_unavailable"]
        assert sess.calls == []  # the sandbox method was never called
        assert res.stop_reason == L.STOP_DONE

    def test_no_sandbox_session_refuses(self):
        script = [
            {"content": "x", "tool_calls": _native("view", {"path": "/a"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run("q", model_client=ScriptedClient(script), sandbox=None, mode="daily")
        assert res.tool_results[0].reason == "sandbox_unavailable"


# Per-mode gating inside the loop


class TestGatingInLoop:
    def test_bulbe_denied_tool_not_executed(self):
        sess = FakeSession()
        script = [
            {"content": "x", "tool_calls": _native("bash", {"command": "ls"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run(
            "q", model_client=ScriptedClient(script), sandbox=sess, mode="bulbe",
            approval_fn=lambda c, t, a: False,
        )
        assert res.tool_results[0].reason == "denied_by_human"
        assert sess.calls == []

    def test_bulbe_approved_tool_executed(self):
        sess = FakeSession()
        script = [
            {"content": "x", "tool_calls": _native("view", {"path": "/a"})},
            {"content": "done", "tool_calls": None},
        ]
        res = L.run(
            "q", model_client=ScriptedClient(script), sandbox=sess, mode="bulbe",
            approval_fn=lambda c, t, a: True,
        )
        assert res.tool_results[0].executed is True
        assert sess.calls == [("view", "/a")]


# Memory working-block injection


class TestMemoryInjection:
    def test_provider_block_injected_as_untrusted_before_task(self):
        res = L.run(
            "kubuntu?",
            model_client=ScriptedClient([{"content": "answer", "tool_calls": None}]),
            sandbox=FakeSession(),
            mode="daily",
            memory_provider=lambda q, *, user_id=None: "Relevant memories:\n- Leon uses Kubuntu",
        )
        mem = [m for m in res.messages if m["role"] == "user" and "Kubuntu" in m.get("content", "")]
        assert mem, "memory working block must be injected"
        assert 'trusted="false"' in mem[0]["content"]
        # It precedes the task user message.
        idx_mem = res.messages.index(mem[0])
        idx_task = next(i for i, m in enumerate(res.messages) if m["role"] == "user" and m["content"] == "kubuntu?")
        assert idx_mem < idx_task

    def test_empty_block_no_memory_message(self):
        res = L.run(
            "q",
            model_client=ScriptedClient([{"content": "a", "tool_calls": None}]),
            sandbox=FakeSession(),
            mode="daily",
            memory_provider=lambda q, *, user_id=None: "",
        )
        assert not any("untrusted_data" in m.get("content", "") for m in res.messages if m["role"] == "user")

    def test_include_memory_false_skips_provider(self):
        called = {"n": 0}

        def provider(q, *, user_id=None):
            called["n"] += 1
            return "block"

        L.run(
            "q",
            model_client=ScriptedClient([{"content": "a", "tool_calls": None}]),
            sandbox=FakeSession(),
            mode="daily",
            include_memory=False,
            memory_provider=provider,
        )
        assert called["n"] == 0


# Bounded verifier


class TestVerifier:
    def test_verifier_pass_verdict(self):
        # Main loop finishes in one round; the verifier returns PASS.
        client = ScriptedClient([
            {"content": "Final answer.", "tool_calls": None},   # main round
            {"content": "PASS, the result is correct.", "tool_calls": None},  # verifier round
        ])
        res = L.run("q", model_client=client, sandbox=FakeSession(), mode="daily", verify=True)
        assert res.verifier is not None
        assert res.verifier.verdict == "pass"
        assert res.verifier.rounds == 1
        assert res.verifier.bounded is True

    def test_verifier_never_exceeds_cap(self):
        # Main loop finishes; the verifier model always asks for a tool.
        script = [{"content": "Final.", "tool_calls": None}] + [
            {"content": "v", "tool_calls": _native("view", {"path": "/x"})}
        ] * 6
        res = L.run("q", model_client=ScriptedClient(script), sandbox=FakeSession(), mode="daily", verify=True)
        assert res.verifier.rounds == L._VERIFIER_MAX_ROUNDS == 2
        assert res.verifier.bounded is True

    def test_no_verifier_when_disabled(self):
        res = L.run(
            "q", model_client=ScriptedClient([{"content": "a", "tool_calls": None}]),
            sandbox=FakeSession(), mode="daily", verify=False,
        )
        assert res.verifier is None

    def test_verifier_skipped_on_max_rounds(self):
        res = L.run(
            "q", model_client=AlwaysToolClient(), sandbox=FakeSession(), mode="daily",
            max_rounds=2, verify=True,
        )
        assert res.stop_reason == L.STOP_MAX_ROUNDS
        assert res.verifier is None

    def test_extract_verdict(self):
        assert L._extract_verdict("This FAILS the check") == "fail"
        assert L._extract_verdict("PASS") == "pass"
        assert L._extract_verdict("the result is correct") == "pass"
        assert L._extract_verdict("hmm, not sure") == "unknown"


# Events


class TestEvents:
    def test_events_emitted(self):
        kinds = []
        script = [
            {"content": "x", "tool_calls": _native("bash", {"command": "ls"})},
            {"content": "done", "tool_calls": None},
        ]
        L.run(
            "q", model_client=ScriptedClient(script), sandbox=FakeSession(), mode="daily",
            on_event=lambda e: kinds.append(e.kind),
        )
        kinds_set = set(kinds)
        assert "round_start" in kinds_set
        assert "model_output" in kinds_set
        assert "tool_result" in kinds_set
        assert "done" in kinds_set


# Package facade


class TestFacade:
    def test_init_source_re_exports(self):
        src = (AGENT / "__init__.py").read_text(encoding="utf-8")
        for name in ("run", "ToolCall", "evaluate", "untrusted_message", "parse_tool_blocks",
                     "DAILY_ALLOWLIST", "MAX_AGENT_ROUNDS"):
            assert name in src, f"facade should re-export {name}"
        assert "checkpoint_before_apply" in src
        assert "FEATURE_AVAILABLE" in src

    def test_init_parses(self):
        import ast

        ast.parse((AGENT / "__init__.py").read_text(encoding="utf-8"))

    def test_facade_runtime_exports(self):
        # Load the real package facade, then restore the stub so other test
        # files that rely on the stub are unaffected.
        saved = sys.modules.get("opti_oignon.agent")
        try:
            spec = importlib.util.spec_from_file_location(
                "opti_oignon.agent",
                str(AGENT / "__init__.py"),
                submodule_search_locations=[str(AGENT)],
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["opti_oignon.agent"] = mod
            spec.loader.exec_module(mod)
            for name in ("run", "ToolCall", "DispatchResult", "evaluate", "is_tool_allowed",
                         "untrusted_message", "parse_tool_blocks", "DAILY_ALLOWLIST",
                         "BULBE_ALLOWLIST", "MAX_AGENT_ROUNDS"):
                assert hasattr(mod, name), f"facade missing runtime export {name}"
            assert callable(mod.run)
        finally:
            if saved is not None:
                sys.modules["opti_oignon.agent"] = saved
