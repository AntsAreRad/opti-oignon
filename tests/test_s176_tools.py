#!/usr/bin/env python3
"""Tests for S176 -- the concrete agent tool set (Theme 3 / Odysseus Core).

Covers ODYSSEUS_SPEC.md Section 5.3 / Section 6 surfaces that are not skills:

- The schema model and the six tool schemas (the four sandboxed file / shell /
  code tools plus ``web_search`` and ``manage_memory``); the sandbox tools'
  argument names match the S175 ``dispatch._SANDBOX_DISPATCH`` seam.
- The per-mode registry: what it exposes is always a subset of the active
  mode's allowlist; Daily exposes all six, Bulbe exposes exactly the sandboxed
  four (the network and state-mutation tools are unreachable there).
- The non-sandbox handlers (``web_search``, ``manage_memory``): each returns an
  observation string and never raises, with the backend injected for tests.
- Integration with the S175 dispatch seam: a Daily handler tool executes, a
  Bulbe network / state tool is refused, a sandbox tool still routes through the
  injected session.
- Cartography: ``tools.py`` is registered in ODYSSEUS_SPEC.md Section 10.

Loaded in isolation via ``spec_from_file_location`` with ``opti_oignon``
stubbed, so the runtime collects without the backend.
"""

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OO = ROOT / "opti_oignon"
AGENT = OO / "agent"
SPEC = ROOT / "ODYSSEUS_SPEC.md"


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
al = _ensure_agent("allowlists")
disp = _ensure_agent("dispatch")
t = _ensure_agent("tools")


@pytest.fixture(autouse=True)
def _reset_registry():
    t.reset_tool_registry()
    yield
    t.reset_tool_registry()


# Recording sandbox session, mirroring sandbox_tools.SandboxToolSession.


class _FakeManager:
    def __init__(self, bwrap: bool = True):
        self.bwrap_available = bwrap


class _FakeSession:
    def __init__(self, bwrap: bool = True, active: bool = True):
        self.sandbox_manager = _FakeManager(bwrap)
        self.active = active
        self.calls: list[tuple] = []

    def bash(self, command, timeout=30):
        self.calls.append(("bash", command, timeout))
        return f"[sandbox] {command}"

    def view(self, path, start_line=0, end_line=0):
        self.calls.append(("view", path, start_line, end_line))
        return f"[sandbox] view {path}"

    def create_file(self, path, content):
        self.calls.append(("create_file", path, content))
        return f"[sandbox] wrote {path}"

    def str_replace(self, path, old_str, new_str=""):
        self.calls.append(("str_replace", path, old_str, new_str))
        return "[sandbox] replaced"


# Fake memory store, mirroring MemoryStore's used surface.


class _Rec:
    def __init__(self, rid="m1", category="fact", text="a fact"):
        self.id = rid
        self.category = category
        self.text = text


class _Dec:
    def __init__(self, action="add"):
        self.action = action


class _FakeStore:
    def __init__(self):
        self.added: list[tuple] = []
        self.deleted: list[str] = []
        self.hard_deleted: list[str] = []

    def add(self, text, category="fact", *, source="", **kw):
        self.added.append((text, category, source))
        return _Rec(text=text, category=category), _Dec("add")

    def list(self, *, category=None, limit=20, **kw):
        return [_Rec(rid="m1", text="first"), _Rec(rid="m2", text="second")]

    def get(self, fact_id, **kw):
        return _Rec(rid=fact_id) if fact_id == "m1" else None

    def update(self, fact_id, *, text=None, category=None, **kw):
        return _Rec(rid=fact_id, text=text or "updated") if fact_id == "m1" else None

    def soft_delete(self, fact_id, **kw):
        self.deleted.append(fact_id)
        return fact_id == "m1"

    def hard_delete(self, fact_id, **kw):
        self.hard_deleted.append(fact_id)
        return True


def _tc(name, args=None, source="native"):
    return disp.ToolCall(name=name, arguments=dict(args or {}), source=source)


# Module conventions


class TestModuleConventions:
    def test_sentinels_present(self):
        assert t.checkpoint_before_apply is True
        assert t.FEATURE_AVAILABLE is True

    def test_registry_singleton(self):
        a = t.get_tool_registry()
        b = t.get_tool_registry()
        assert a is b

    def test_reset_drops_singleton(self):
        a = t.get_tool_registry()
        t.reset_tool_registry()
        b = t.get_tool_registry()
        assert a is not b


# Schemas


class TestSchemas:
    def test_six_schemas(self):
        assert len(t.ALL_SCHEMAS) == 6

    def test_sandbox_four_match_allowlist(self):
        sandboxed = {s.name for s in t.ALL_SCHEMAS if s.sandboxed}
        assert sandboxed == set(al.SANDBOX_TOOL_NAMES)

    def test_handler_two_not_sandboxed(self):
        non = {s.name for s in t.ALL_SCHEMAS if not s.sandboxed}
        assert non == {t.TOOL_WEB_SEARCH, t.TOOL_MANAGE_MEMORY}

    def test_bash_args_match_dispatch_seam(self):
        names = {p.name for p in t.BASH_SCHEMA.parameters}
        assert names == {"command", "timeout"}
        assert t.BASH_SCHEMA.required_names() == ["command"]

    def test_view_args_match_dispatch_seam(self):
        names = {p.name for p in t.VIEW_SCHEMA.parameters}
        assert names == {"path", "start_line", "end_line"}
        assert t.VIEW_SCHEMA.required_names() == ["path"]

    def test_create_file_args_match_dispatch_seam(self):
        names = {p.name for p in t.CREATE_FILE_SCHEMA.parameters}
        assert names == {"path", "content"}
        assert set(t.CREATE_FILE_SCHEMA.required_names()) == {"path", "content"}

    def test_str_replace_args_match_dispatch_seam(self):
        names = {p.name for p in t.STR_REPLACE_SCHEMA.parameters}
        assert names == {"path", "old_str", "new_str"}
        assert set(t.STR_REPLACE_SCHEMA.required_names()) == {"path", "old_str"}

    def test_web_search_schema(self):
        names = {p.name for p in t.WEB_SEARCH_SCHEMA.parameters}
        assert names == {"query", "max_results"}
        assert t.WEB_SEARCH_SCHEMA.required_names() == ["query"]
        assert t.WEB_SEARCH_SCHEMA.sandboxed is False

    def test_manage_memory_schema(self):
        names = {p.name for p in t.MANAGE_MEMORY_SCHEMA.parameters}
        assert "action" in names
        assert t.MANAGE_MEMORY_SCHEMA.required_names() == ["action"]
        assert t.MANAGE_MEMORY_SCHEMA.sandboxed is False

    def test_sandbox_argument_names_cover_dispatch_lambdas(self):
        # Every key the dispatch sandbox lambdas read from arguments must be a
        # declared parameter, so the schema cannot drift from the seam.
        expected = {
            "bash": {"command", "timeout"},
            "view": {"path", "start_line", "end_line"},
            "create_file": {"path", "content"},
            "str_replace": {"path", "old_str", "new_str"},
        }
        for schema in t.ALL_SCHEMAS:
            if schema.sandboxed:
                got = {p.name for p in schema.parameters}
                assert got == expected[schema.name]


# Native schema and prompt rendering


class TestNativeSchema:
    def test_native_function_shape(self):
        native = t.BASH_SCHEMA.to_native()
        assert native["type"] == "function"
        assert native["function"]["name"] == "bash"
        params = native["function"]["parameters"]
        assert params["type"] == "object"
        assert "command" in params["properties"]
        assert params["required"] == ["command"]

    def test_native_property_types(self):
        native = t.VIEW_SCHEMA.to_native()
        props = native["function"]["parameters"]["properties"]
        assert props["path"]["type"] == "string"
        assert props["start_line"]["type"] == "integer"

    def test_to_prompt_has_name_and_args(self):
        line = t.WEB_SEARCH_SCHEMA.to_prompt()
        assert "web_search" in line
        assert "query" in line

    def test_to_prompt_marks_sandboxed(self):
        assert "[sandboxed]" in t.BASH_SCHEMA.to_prompt()
        assert "[sandboxed]" not in t.WEB_SEARCH_SCHEMA.to_prompt()


# Per-mode registry


class TestRegistryPerMode:
    def test_daily_exposes_all_six(self):
        ts = t.build_tool_set("daily")
        assert set(ts.names) == {s.name for s in t.ALL_SCHEMAS}

    def test_bulbe_exposes_sandbox_only(self):
        ts = t.build_tool_set("bulbe")
        assert set(ts.names) == set(al.SANDBOX_TOOL_NAMES)

    def test_daily_handlers_are_non_sandbox_two(self):
        ts = t.build_tool_set("daily")
        assert set(ts.tool_handlers) == {t.TOOL_WEB_SEARCH, t.TOOL_MANAGE_MEMORY}

    def test_bulbe_has_no_handlers(self):
        ts = t.build_tool_set("bulbe")
        assert ts.tool_handlers == {}

    def test_registry_subset_of_allowlist_daily(self):
        ts = t.build_tool_set("daily")
        for name in ts.names:
            assert al.is_tool_allowed(name, "daily")

    def test_registry_subset_of_allowlist_bulbe(self):
        ts = t.build_tool_set("bulbe")
        for name in ts.names:
            assert al.is_tool_allowed(name, "bulbe")

    def test_bulbe_names_are_subset_of_daily(self):
        daily = set(t.build_tool_set("daily").names)
        bulbe = set(t.build_tool_set("bulbe").names)
        assert bulbe < daily

    def test_include_handlers_false_yields_no_handlers(self):
        ts = t.build_tool_set("daily", include_handlers=False)
        assert ts.tool_handlers == {}
        assert set(ts.names) == {s.name for s in t.ALL_SCHEMAS}

    def test_unknown_mode_is_fail_secure_bulbe(self):
        ts = t.build_tool_set("nonsense")
        assert set(ts.names) == set(al.SANDBOX_TOOL_NAMES)

    def test_native_tools_for_daily(self):
        native = t.native_tools_for("daily")
        names = {n["function"]["name"] for n in native}
        assert names == {s.name for s in t.ALL_SCHEMAS}

    def test_system_prompt_section_lists_tools(self):
        section = t.system_prompt_section_for("daily")
        for name in (t.TOOL_BASH, t.TOOL_WEB_SEARCH, t.TOOL_MANAGE_MEMORY):
            assert name in section

    def test_sandbox_names_helper(self):
        ts = t.build_tool_set("daily")
        assert set(ts.sandbox_names) == set(al.SANDBOX_TOOL_NAMES)


# Allowlist discipline (network / state tools out of Bulbe)


class TestAllowlistDiscipline:
    def test_web_search_is_network_and_out_of_bulbe(self):
        assert t.TOOL_WEB_SEARCH in al.NETWORK_TOOLS
        assert t.TOOL_WEB_SEARCH not in al.BULBE_ALLOWLIST

    def test_manage_memory_is_state_and_out_of_bulbe(self):
        assert t.TOOL_MANAGE_MEMORY in al.STATE_MUTATION_TOOLS
        assert t.TOOL_MANAGE_MEMORY not in al.BULBE_ALLOWLIST

    def test_sandbox_tools_in_both_modes(self):
        for name in al.SANDBOX_TOOL_NAMES:
            assert name in al.DAILY_ALLOWLIST
            assert name in al.BULBE_ALLOWLIST


# web_search handler


class TestWebSearchHandler:
    def test_returns_formatted_results(self):
        h = t.make_web_search_handler(lambda q, max_results=3: f"[{max_results}] {q}")
        out = h({"query": "pandas", "max_results": 2})
        assert "pandas" in out
        assert "[2]" in out

    def test_empty_query_message(self):
        h = t.make_web_search_handler(lambda q, max_results=3: "x")
        assert "query" in h({"query": "   "}).lower()

    def test_unavailable_when_no_resolver(self, monkeypatch):
        # When the search function cannot be resolved at all, the handler
        # reports it unavailable rather than raising.
        monkeypatch.setattr(t, "_default_web_search_fn", lambda: None)
        h = t.make_web_search_handler()
        assert "unavailable" in h({"query": "anything"}).lower()

    def test_default_handler_is_graceful(self):
        # No injected fn: whether the backend resolves and errors, or is
        # absent, the default handler returns a non-empty observation and never
        # raises.
        h = t.make_web_search_handler()
        out = h({"query": "anything"})
        assert isinstance(out, str) and out.strip()

    def test_never_raises_on_backend_error(self):
        def boom(q, max_results=3):
            raise RuntimeError("network down")

        h = t.make_web_search_handler(boom)
        out = h({"query": "x"})
        assert "failed" in out.lower()

    def test_no_results_message(self):
        h = t.make_web_search_handler(lambda q, max_results=3: "")
        assert "no results" in h({"query": "x"}).lower()


# manage_memory handler


class TestManageMemoryHandler:
    def test_list(self):
        store = _FakeStore()
        h = t.make_manage_memory_handler(store)
        out = h({"action": "list"})
        assert "m1" in out and "m2" in out

    def test_get_found(self):
        h = t.make_manage_memory_handler(_FakeStore())
        out = h({"action": "get", "fact_id": "m1"})
        assert "m1" in out

    def test_get_missing(self):
        h = t.make_manage_memory_handler(_FakeStore())
        out = h({"action": "get", "fact_id": "zzz"})
        assert "no memory" in out.lower()

    def test_add(self):
        store = _FakeStore()
        h = t.make_manage_memory_handler(store)
        out = h({"action": "add", "text": "Leon uses Kubuntu", "category": "fact"})
        assert "added" in out.lower()
        assert store.added and store.added[0][0] == "Leon uses Kubuntu"

    def test_add_requires_text(self):
        h = t.make_manage_memory_handler(_FakeStore())
        assert "text" in h({"action": "add"}).lower()

    def test_update(self):
        h = t.make_manage_memory_handler(_FakeStore())
        out = h({"action": "update", "fact_id": "m1", "text": "new text"})
        assert "updated" in out.lower()

    def test_delete_is_soft_only(self):
        store = _FakeStore()
        h = t.make_manage_memory_handler(store)
        out = h({"action": "delete", "fact_id": "m1"})
        assert "archived" in out.lower()
        assert store.deleted == ["m1"]
        assert store.hard_deleted == []  # the agent never hard-deletes

    def test_unknown_action(self):
        h = t.make_manage_memory_handler(_FakeStore())
        assert "must be one of" in h({"action": "frobnicate"})

    def test_unavailable_store(self):
        h = t.make_manage_memory_handler()  # no injected store; backend absent
        assert "unavailable" in h({"action": "list"}).lower()

    def test_never_raises_on_store_error(self):
        class Boom:
            def list(self, **kw):
                raise RuntimeError("db locked")

        h = t.make_manage_memory_handler(Boom())
        assert "failed" in h({"action": "list"}).lower()


# Integration with the S175 dispatch seam


class TestDispatchIntegration:
    def test_daily_web_search_executes_via_handler(self):
        ts = t.build_tool_set("daily")
        # inject a deterministic search fn through a fresh registry
        t.reset_tool_registry()
        reg = t.ToolRegistry(web_search_fn=lambda q, max_results=3: f"hit:{q}")
        handlers = reg.build("daily").tool_handlers
        res = disp.dispatch_tool_call(
            _tc("web_search", {"query": "rust"}),
            mode="daily",
            tool_handlers=handlers,
        )
        assert res.executed is True
        assert "hit:rust" in res.observation

    def test_bulbe_web_search_refused(self):
        ts = t.build_tool_set("bulbe")
        res = disp.dispatch_tool_call(
            _tc("web_search", {"query": "rust"}),
            mode="bulbe",
            tool_handlers=ts.tool_handlers,
            approval_fn=lambda *a: True,
        )
        assert res.executed is False
        assert res.reason == al.REASON_NOT_ALLOWED

    def test_daily_manage_memory_executes_via_handler(self):
        store = _FakeStore()
        reg = t.ToolRegistry(memory_store=store)
        handlers = reg.build("daily").tool_handlers
        res = disp.dispatch_tool_call(
            _tc("manage_memory", {"action": "add", "text": "fact one"}),
            mode="daily",
            tool_handlers=handlers,
        )
        assert res.executed is True
        assert store.added

    def test_sandbox_tool_routes_through_session(self):
        # A sandboxed tool from the tool set still executes through the session.
        session = _FakeSession(bwrap=True, active=True)
        res = disp.dispatch_tool_call(
            _tc("bash", {"command": "ls"}),
            mode="bulbe",
            sandbox=session,
            approval_fn=lambda *a: True,
        )
        assert res.executed is True
        assert session.calls and session.calls[0][0] == "bash"

    def test_sandbox_tool_refused_without_bwrap(self):
        session = _FakeSession(bwrap=False, active=True)
        res = disp.dispatch_tool_call(
            _tc("create_file", {"path": "a.txt", "content": "x"}),
            mode="bulbe",
            sandbox=session,
            approval_fn=lambda *a: True,
        )
        assert res.executed is False
        assert res.reason == disp.REASON_SANDBOX_UNAVAILABLE
        assert session.calls == []  # the session method was never called


# Cartography


class TestCartography:
    def test_tools_registered_in_spec(self):
        text = SPEC.read_text(encoding="utf-8")
        assert "opti_oignon/agent/tools.py" in text

    def test_tools_file_on_disk(self):
        assert (AGENT / "tools.py").exists()
