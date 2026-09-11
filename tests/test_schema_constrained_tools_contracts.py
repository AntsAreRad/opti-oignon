#!/usr/bin/env python3
"""Contracts for schema-constrained tool calls.

A tool call is structurally valid or it does not run. The forced decision
no longer leaves the arguments free: its schema is one branch per tool,
the tool's name as a constant and the tool's own parameter schema for the
arguments, so the sampler cannot produce a call the tool cannot take. At
execution, an argument of the wrong type is refused before the handler,
named, and marked retryable so the model can fix its call. Both schemas --
native and constrained -- come from one builder, so they cannot disagree.

  * SC1 -- the constrained schema is one branch per tool, the name a
    constant, the arguments the tool's parameter schema closed to unknown
    keys, and it agrees with the native schema on every parameter.
  * SC2 -- the validator is capable and precise: nothing on a good call,
    one named error per wrong type or missing parameter, booleans are not
    integers, integers are numbers, list items are strings.
  * SC3 -- the forced decision sends the constrained schema through the
    registry when it has the definitions, and the enum schema when it has
    only names.
  * SC4 -- a wrong-typed argument is refused before the handler, named,
    retryable; a well-typed call runs; the missing-required message stays.
  * SC5 -- end to end: a forced decision whose payload fits the constrained
    schema is executed, and the schema that reached the engine was the
    constrained one.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window over the registry bridge.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402
from _registry_bridge import seed_registry  # noqa: E402

_MSGS = [{"role": "user", "content": "find x"}]


class _Scripted:
    """Replies by kwarg: tools -> the native turn, format -> the forced turn."""

    def __init__(self, native=None, forced=None):
        self.calls, self.native, self.forced = [], native, forced

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if "format" in kwargs:
            return {"message": {"content": json.dumps(self.forced or {})}}
        if "tools" in kwargs:
            return {"message": {"content": "", "tool_calls": self.native or []}}
        return {"message": {"content": "done"}}


def _param(name, ptype, required=True, default=None):
    return SimpleNamespace(name=name, type=ptype, description=f"the {name}", required=required, default=default)


def _search_tool(handler=None):
    return SimpleNamespace(
        name="search", description="search the web", enabled=True, handler=handler,
        parameters={
            "query": _param("query", "string"),
            "max_results": _param("max_results", "int", required=False, default=5),
            "verbose": _param("verbose", "bool", required=False),
            "tags": _param("tags", "list", required=False),
            "ratio": _param("ratio", "float", required=False),
        },
    )


class _Registry:
    def __init__(self, tools):
        self._tools = {t.name: t for t in tools}

    def list_available(self):
        return list(self._tools.values())

    def get(self, name):
        return self._tools.get(name)

    def is_available(self, name):
        return name in self._tools

    def get_tools_prompt(self, tools=None):
        return "tools: " + ", ".join(self._tools)


def _open(scripted=None):
    so = types.ModuleType("opti_oignon.structured_output")
    so.StructuredOutputEngine = object
    so.ToolCallRequest = object
    so.structured_output_engine = None
    so.STRUCTURED_OUTPUT_AVAILABLE = False
    cfg = types.ModuleType("opti_oignon.config")
    cfg.config = SimpleNamespace(get_model=lambda *a, **k: "m", get_temperature=lambda *a, **k: 0.0,
                                 get=lambda *a, **k: None, get_user_preference=lambda k, d=None: d)
    cfg.get_model = lambda *a, **k: "m"
    seeded = {"opti_oignon.structured_output": so, "opti_oignon.config": cfg}
    scripted = scripted or _Scripted()
    seed_registry(seeded, scripted)
    loaded, restore = isolate(
        targets={
            "opti_oignon.tool_calling": source("tool_calling.py"),
            "opti_oignon.tool_registry": source("tool_registry.py"),
            "opti_oignon.response_hygiene": source("response_hygiene.py"),
            "opti_oignon.tool_executor": source("tool_executor.py"),
        },
        seeded=seeded,
        packages=("opti_oignon",),
    )
    return loaded["opti_oignon.tool_calling"], loaded["opti_oignon.tool_executor"], scripted, restore


# ---------------------------------------------------------------------------
# SC1 -- the constrained schema
# ---------------------------------------------------------------------------
def test_sc1_the_constrained_schema_is_one_closed_branch_per_tool_and_agrees_with_the_native_one():
    tc, te, scripted, restore = _open()
    try:
        tools = [_search_tool(), SimpleNamespace(name="ping", description="ping", enabled=True, handler=None,
                                                 parameters={"host": _param("host", "string")})]
        schema = tc.constrained_decision_schema(tools)
        branches = schema["oneOf"]
        assert [b["properties"]["tool_name"]["const"] for b in branches] == ["search", "ping"]
        for tool, branch in zip(tools, branches):
            args = branch["properties"]["arguments"]
            assert args["type"] == "object" and args["additionalProperties"] is False
            assert set(args["properties"]) == set(tool.parameters)
            assert args["required"] == [p.name for p in tool.parameters.values() if p.required]
            assert branch["required"] == ["tool_name", "arguments"]
            native = tc.native_tool_schemas([tool])[0]["function"]["parameters"]
            assert args["properties"] == native["properties"] and args["required"] == native["required"], (
                "one builder behind both schemas"
            )
        search = branches[0]["properties"]["arguments"]["properties"]
        assert search["max_results"]["type"] == "integer" and search["verbose"]["type"] == "boolean"
        assert search["tags"] == {"type": "array", "description": "the tags", "items": {"type": "string"}}
        assert search["ratio"]["type"] == "number"
        with pytest.raises(ValueError):
            tc.constrained_decision_schema([])
    finally:
        restore()


# ---------------------------------------------------------------------------
# SC2 -- the validator
# ---------------------------------------------------------------------------
def test_sc2_the_validator_names_every_wrong_type_and_nothing_on_a_good_call():
    tc, te, scripted, restore = _open()
    try:
        tool = _search_tool()
        good = {"query": "onions", "max_results": 3, "verbose": True, "tags": ["a", "b"], "ratio": 0.5}
        assert tc.validate_arguments(tool, good) == []
        assert tc.validate_arguments(tool, {"query": "onions"}) == [], "optional parameters may be absent"
        assert tc.validate_arguments(tool, {"query": "onions", "ratio": 3}) == [], "an integer is a number"
        cases = {
            "query": ({"query": 5}, "string"),
            "max_results": ({"query": "x", "max_results": "ten"}, "integer"),
            "verbose": ({"query": "x", "verbose": "false"}, "boolean"),
            "tags": ({"query": "x", "tags": ["a", 1]}, "string"),
            "ratio": ({"query": "x", "ratio": "half"}, "number"),
        }
        for name, (args, expected) in cases.items():
            errors = tc.validate_arguments(tool, args)
            assert len(errors) == 1, f"{name}: exactly one error, got {errors}"
            assert name in errors[0] and expected in errors[0], f"{name}: the error names the parameter and the type: {errors[0]}"
        errors = tc.validate_arguments(tool, {"query": "x", "max_results": True})
        assert len(errors) == 1 and "max_results" in errors[0], "a boolean is not an integer"
        errors = tc.validate_arguments(tool, {"max_results": 3})
        assert len(errors) == 1 and "query" in errors[0] and "required" in errors[0]
        errors = tc.validate_arguments(tool, {"query": 5, "max_results": "ten"})
        assert len(errors) == 2, "one error per wrong parameter, none swallowed"
        assert tc.validate_arguments(tool, None) != [] and tc.validate_arguments(tool, "query=x") != []
    finally:
        restore()


# ---------------------------------------------------------------------------
# SC3 -- the forced decision sends the constrained schema
# ---------------------------------------------------------------------------
def test_sc3_the_forced_decision_sends_the_constrained_schema_when_it_has_definitions():
    scripted = _Scripted(forced={"tool_name": "search", "arguments": {"query": "x"}, "reasoning": "r"})
    tc, te, scripted, restore = _open(scripted)
    try:
        tool = _search_tool()
        holder = SimpleNamespace()
        decision = te.ToolExecutor._enum_force_tool(holder, _MSGS, "m", ["search"], tools=[tool])
        assert decision == ("search", {"query": "x"})
        sent = scripted.calls[-1]["format"]
        assert sent == tc.constrained_decision_schema([tool])
        assert "oneOf" in sent
        decision = te.ToolExecutor._enum_force_tool(holder, _MSGS, "m", ["search"])
        assert decision == ("search", {"query": "x"})
        assert scripted.calls[-1]["format"] == tc.forced_decision_schema(["search"]), "names only: the enum schema, as before"
    finally:
        restore()


# ---------------------------------------------------------------------------
# SC4 -- refused before the handler
# ---------------------------------------------------------------------------
def test_sc4_a_wrong_typed_argument_is_refused_before_the_handler_and_named():
    tc, te, scripted, restore = _open()
    try:
        seen = []

        def handler(**kwargs):
            seen.append(kwargs)
            return "ok"

        ex = te.ToolExecutor(registry=_Registry([_search_tool(handler)]), structured_engine=None, default_model="m")
        result = ex._execute_tool("search", {"query": "x", "max_results": "ten"})
        assert result.success is False and result.retryable is True
        assert "max_results" in result.result and "integer" in result.result
        assert seen == [], "the handler never ran"
        result = ex._execute_tool("search", {"query": "x", "max_results": 3})
        assert result.success is True and seen == [{"query": "x", "max_results": 3}]
        result = ex._execute_tool("search", {"max_results": 3})
        assert result.success is False and result.retryable is True
        assert result.result.startswith("Missing required parameter: query"), "the existing message stays"
        assert len(seen) == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# SC5 -- end to end under force
# ---------------------------------------------------------------------------
def test_sc5_a_well_typed_forced_decision_is_executed_end_to_end():
    scripted = _Scripted(native=[], forced={"tool_name": "search", "arguments": {"query": "x", "max_results": 2}})
    tc, te, scripted, restore = _open(scripted)
    try:
        te.model_supports_native_tools = lambda model, capability_lookup=None: True
        tool = _search_tool()
        ex = te.ToolExecutor(registry=_Registry([tool]), structured_engine=None, default_model="m")
        calls = ex._decide_tools("find x", "m", [], [], force=True)
        assert calls == [("search", {"query": "x", "max_results": 2})]
        kinds = [("tools" in c, "format" in c) for c in scripted.calls]
        assert kinds == [(True, False), (False, True)], "native first, then the forced head"
        assert scripted.calls[1]["format"] == tc.constrained_decision_schema([tool])
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
