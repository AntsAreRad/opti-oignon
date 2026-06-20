#!/usr/bin/env python3
"""Tests for S175 -- agent tool-call dispatch (Theme 3 / Odysseus Core).

Three areas (ODYSSEUS_SPEC.md Section 5.2, Section 5.3, Section 5.4):

- Dual dispatch: native function-calling calls and parser-recovered text calls
  both normalise to one ``ToolCall``; native takes precedence; the round
  reports which path it used.
- The sandbox invariant: a filesystem / shell / code tool dispatches only
  through the injected sandbox session, there is no host-execution path in the
  module, and the dispatch refuses when bwrap is unavailable (the session
  method is never called).
- Per-mode gating wiring: the allowlist is consulted, and in Bulbe the human
  gate decides; a refused or failed tool is an observation, never an exception.

Loaded in isolation via ``spec_from_file_location`` with ``opti_oignon``
stubbed; the sandbox session is a recording fake.
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
d = _ensure_agent("dispatch")


class FakeManager:
    def __init__(self, bwrap: bool):
        self.bwrap_available = bwrap


class FakeSession:
    """Recording stand-in for sandbox_tools.SandboxToolSession."""

    def __init__(self, bwrap: bool = True, active: bool = True):
        self.sandbox_manager = FakeManager(bwrap)
        self.active = active
        self.calls: list[tuple] = []

    def bash(self, command, timeout=30):
        self.calls.append(("bash", command, timeout))
        return f"[sandbox] ran: {command}"

    def view(self, path, start_line=0, end_line=0):
        self.calls.append(("view", path, start_line, end_line))
        return f"[sandbox] view {path}"

    def create_file(self, path, content):
        self.calls.append(("create_file", path, content))
        return f"[sandbox] wrote {path}"

    def str_replace(self, path, old_str, new_str=""):
        self.calls.append(("str_replace", path, old_str, new_str))
        return "[sandbox] replaced"


def _tc(name, args=None, source="native"):
    return d.ToolCall(name=name, arguments=args or {}, source=source)


# Module shape


class TestModuleShape:
    def test_sentinels(self):
        assert d.checkpoint_before_apply is True
        assert d.FEATURE_AVAILABLE is True

    def test_path_constants(self):
        assert d.PATH_NATIVE == "native"
        assert d.PATH_TEXT == "text"

    def test_guard_flags_are_bool(self):
        assert isinstance(d.JSON_REPAIR_AVAILABLE, bool)
        assert isinstance(d.STRUCTURED_OUTPUT_AVAILABLE, bool)

    def test_tool_call_to_dict(self):
        c = _tc("bash", {"command": "ls"})
        assert c.to_dict() == {"name": "bash", "arguments": {"command": "ls"}, "source": "native"}


# Dual dispatch / normalisation


class TestNativePath:
    def test_ollama_shape_dict_args(self):
        resp = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "bash", "arguments": {"command": "ls"}}},
                    {"function": {"name": "view", "arguments": {"path": "/x"}}},
                ],
            }
        }
        calls, path = d.resolve_tool_calls(resp)
        assert path == d.PATH_NATIVE
        assert [(c.name, c.arguments, c.source) for c in calls] == [
            ("bash", {"command": "ls"}, "native"),
            ("view", {"path": "/x"}, "native"),
        ]

    def test_openai_shape_json_string_args(self):
        resp = {
            "message": {
                "tool_calls": [
                    {"id": "1", "type": "function",
                     "function": {"name": "bash", "arguments": '{"command": "id"}'}}
                ]
            }
        }
        calls, path = d.resolve_tool_calls(resp)
        assert path == d.PATH_NATIVE
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "id"}

    def test_top_level_tool_calls(self):
        resp = {"tool_calls": [{"function": {"name": "view", "arguments": {"path": "/a"}}}]}
        calls, path = d.resolve_tool_calls(resp)
        assert path == d.PATH_NATIVE
        assert calls[0].name == "view"

    def test_attribute_style_response(self):
        msg = types.SimpleNamespace(
            content="",
            tool_calls=[types.SimpleNamespace(
                function=types.SimpleNamespace(name="bash", arguments={"command": "pwd"})
            )],
        )
        resp = types.SimpleNamespace(message=msg)
        calls, path = d.resolve_tool_calls(resp)
        assert path == d.PATH_NATIVE
        assert calls[0].name == "bash" and calls[0].arguments == {"command": "pwd"}

    def test_native_call_missing_name_is_skipped(self):
        resp = {"message": {"tool_calls": [{"function": {"arguments": {"x": 1}}}]}}
        calls, path = d.resolve_tool_calls(resp)
        assert path == d.PATH_NATIVE
        assert calls == []


class TestTextPath:
    def test_bracketed_text_call(self):
        resp = {"message": {"content": '[TOOL_CALL]{"name":"create_file","args":{"path":"/x","content":"hi"}}[/TOOL_CALL]'}}
        calls, path = d.resolve_tool_calls(resp)
        assert path == d.PATH_TEXT
        assert calls[0].name == "create_file"
        assert calls[0].arguments == {"path": "/x", "content": "hi"}
        assert calls[0].source == "bracketed"

    def test_fenced_and_xml_sources_preserved(self):
        resp = {"message": {"content": (
            '```json\n{"tool":"bash","arguments":{"command":"ls"}}\n```\n'
            '<invoke name="view"><param name="path">/y</param></invoke>'
        )}}
        calls, path = d.resolve_tool_calls(resp)
        assert path == d.PATH_TEXT
        assert [(c.name, c.source) for c in calls] == [("bash", "fenced"), ("view", "xml")]

    def test_no_calls_in_text(self):
        resp = {"message": {"content": "I cannot help with that."}}
        calls, path = d.resolve_tool_calls(resp)
        assert path == d.PATH_TEXT
        assert calls == []

    def test_empty_response(self):
        calls, path = d.resolve_tool_calls({"message": {"content": ""}})
        assert path == d.PATH_TEXT and calls == []


class TestPrecedenceAndExtraction:
    def test_native_preferred_over_text(self):
        resp = {"message": {
            "content": '```json\n{"tool":"view","arguments":{"path":"/y"}}\n```',
            "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "pwd"}}}],
        }}
        calls, path = d.resolve_tool_calls(resp)
        assert path == d.PATH_NATIVE
        assert [c.name for c in calls] == ["bash"]

    def test_extract_text_helper(self):
        assert d.extract_text({"message": {"content": "hello"}}) == "hello"
        assert d.extract_text({"content": "top"}) == "top"
        assert d.extract_text({"message": {}}) == ""

    def test_extract_native_calls_returns_none_without_calls(self):
        assert d.extract_native_calls({"message": {"content": "x"}}) is None


# The sandbox invariant


class TestSandboxInvariant:
    def test_no_host_execution_primitives_in_source(self):
        src = (AGENT / "dispatch.py").read_text(encoding="utf-8")
        forbidden = [
            "subprocess", "os.system", "os.popen", "Popen", "pty.spawn",
            "socket.socket", "commands.getoutput", "tempfile", "mkdtemp",
            "allow_degraded", "confirm_degraded",
        ]
        present = [t for t in forbidden if t in src]
        assert present == [], f"host-execution primitives must not appear in dispatch: {present}"
        assert "import os" not in src
        assert "import subprocess" not in src

    def test_sandbox_ready_true_with_bwrap(self):
        assert d.sandbox_ready(FakeSession(bwrap=True)) is True

    def test_sandbox_ready_false_without_bwrap(self):
        assert d.sandbox_ready(FakeSession(bwrap=False)) is False

    def test_sandbox_ready_false_without_session_or_manager(self):
        assert d.sandbox_ready(None) is False
        assert d.sandbox_ready(types.SimpleNamespace(sandbox_manager=None)) is False

    def test_sandbox_tool_dispatches_through_session(self):
        sess = FakeSession()
        r = d.dispatch_tool_call(_tc("bash", {"command": "ls -la"}), mode="daily", sandbox=sess)
        assert r.executed is True and r.reason == d.REASON_EXECUTED
        assert r.observation == "[sandbox] ran: ls -la"
        assert sess.calls == [("bash", "ls -la", 30)]

    def test_all_four_sandbox_tools_route_through_session(self):
        sess = FakeSession()
        for call, expect in [
            (_tc("view", {"path": "/a"}), "view"),
            (_tc("create_file", {"path": "/b", "content": "x"}), "create_file"),
            (_tc("str_replace", {"path": "/c", "old_str": "a", "new_str": "b"}), "str_replace"),
        ]:
            r = d.dispatch_tool_call(call, mode="daily", sandbox=sess)
            assert r.executed is True
        assert [c[0] for c in sess.calls] == ["view", "create_file", "str_replace"]

    def test_refusal_when_bwrap_unavailable_does_not_execute(self):
        sess = FakeSession(bwrap=False)
        r = d.dispatch_tool_call(_tc("bash", {"command": "rm -rf /"}), mode="daily", sandbox=sess)
        assert r.executed is False
        assert r.reason == d.REASON_SANDBOX_UNAVAILABLE
        assert sess.calls == []  # the tool method was never called

    def test_refusal_when_no_session(self):
        r = d.dispatch_tool_call(_tc("bash", {"command": "x"}), mode="daily", sandbox=None)
        assert r.executed is False and r.reason == d.REASON_SANDBOX_UNAVAILABLE

    def test_refusal_when_session_inactive(self):
        sess = FakeSession(bwrap=True, active=False)
        r = d.dispatch_tool_call(_tc("view", {"path": "/a"}), mode="daily", sandbox=sess)
        assert r.executed is False and r.reason == d.REASON_SANDBOX_UNAVAILABLE
        assert sess.calls == []


# Per-mode gating wiring


class TestGatingWiring:
    def test_daily_no_per_call_approval(self):
        sess = FakeSession()
        called = {"n": 0}

        def gate(c, t, a):
            called["n"] += 1
            return True

        r = d.dispatch_tool_call(_tc("bash", {"command": "ls"}), mode="daily", sandbox=sess, approval_fn=gate)
        assert r.executed is True
        assert called["n"] == 0  # Daily does not consult the per-call gate

    def test_bulbe_denied_does_not_execute(self):
        sess = FakeSession()
        r = d.dispatch_tool_call(
            _tc("bash", {"command": "x"}), mode="bulbe", sandbox=sess, approval_fn=lambda c, t, a: False
        )
        assert r.executed is False and r.reason == "denied_by_human"
        assert sess.calls == []

    def test_bulbe_approved_executes(self):
        sess = FakeSession()
        r = d.dispatch_tool_call(
            _tc("view", {"path": "/z"}), mode="bulbe", sandbox=sess, approval_fn=lambda c, t, a: True
        )
        assert r.executed is True and sess.calls == [("view", "/z", 0, 0)]

    def test_disallowed_tool_refused(self):
        r = d.dispatch_tool_call(_tc("web_search", {"q": "x"}), mode="bulbe", sandbox=FakeSession())
        assert r.executed is False and r.reason == "not_in_allowlist"

    def test_non_sandbox_allowed_tool_without_handler(self):
        r = d.dispatch_tool_call(_tc("web_search", {"q": "x"}), mode="daily")
        assert r.executed is False and r.reason == d.REASON_NO_EXECUTOR

    def test_non_sandbox_allowed_tool_with_handler(self):
        r = d.dispatch_tool_call(
            _tc("web_search", {"q": "cats"}),
            mode="daily",
            tool_handlers={"web_search": lambda a: f"results for {a['q']}"},
        )
        assert r.executed is True and r.observation == "results for cats"

    def test_unknown_tool_refused(self):
        r = d.dispatch_tool_call(_tc("rm_rf_host", {}), mode="daily", sandbox=FakeSession())
        assert r.executed is False and r.reason == "not_in_allowlist"


# Never raises into the conversation path


class TestNeverRaises:
    def test_sandbox_tool_error_becomes_observation(self):
        sess = FakeSession()
        sess.bash = lambda command, timeout=30: (_ for _ in ()).throw(RuntimeError("kaboom"))
        r = d.dispatch_tool_call(_tc("bash", {"command": "x"}), mode="daily", sandbox=sess)
        assert r.executed is False and r.reason == d.REASON_ERROR
        assert "kaboom" in r.observation

    def test_handler_error_becomes_observation(self):
        def boom(a):
            raise ValueError("nope")

        r = d.dispatch_tool_call(
            _tc("web_search", {}), mode="daily", tool_handlers={"web_search": boom}
        )
        assert r.executed is False and r.reason == d.REASON_ERROR
        assert "nope" in r.observation

    def test_malformed_arguments_do_not_raise(self):
        sess = FakeSession()
        # Non-numeric timeout is coerced to the default rather than raising.
        r = d.dispatch_tool_call(
            _tc("bash", {"command": "ls", "timeout": "soon"}), mode="daily", sandbox=sess
        )
        assert r.executed is True and sess.calls[0] == ("bash", "ls", 30)


# Round-level convenience


class TestDispatchRound:
    def test_round_native_two_calls(self):
        sess = FakeSession()
        resp = {"message": {"tool_calls": [
            {"function": {"name": "bash", "arguments": {"command": "ls"}}},
            {"function": {"name": "view", "arguments": {"path": "/x"}}},
        ]}}
        results, path = d.dispatch_round(resp, mode="daily", sandbox=sess)
        assert path == d.PATH_NATIVE
        assert [r.tool_name for r in results] == ["bash", "view"]
        assert all(r.executed for r in results)
        assert [c[0] for c in sess.calls] == ["bash", "view"]

    def test_round_text_path(self):
        sess = FakeSession()
        resp = {"message": {"content": '[TOOL_CALL]{"name":"view","args":{"path":"/x"}}[/TOOL_CALL]'}}
        results, path = d.dispatch_round(resp, mode="daily", sandbox=sess)
        assert path == d.PATH_TEXT
        assert results[0].executed is True and results[0].source == "bracketed"

    def test_round_no_calls(self):
        results, path = d.dispatch_round({"message": {"content": "nothing"}}, mode="daily")
        assert results == [] and path == d.PATH_TEXT
