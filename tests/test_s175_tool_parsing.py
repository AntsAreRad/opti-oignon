#!/usr/bin/env python3
"""Tests for S175 -- agent tool-block parsing (Theme 3 / Odysseus Core).

Exercises ``opti_oignon/agent/tool_parsing.py`` directly: the three call
formats the local Ollama models emit (fenced code blocks, bracketed
``[TOOL_CALL]`` blocks, XML-style ``<invoke>`` / ``<param>`` blocks), the
normalisation pass into a single ``ParsedToolCall`` shape, exact-duplicate
deduplication, and the guarantee that ordinary content never misfires as a
tool call. Loaded in isolation via ``spec_from_file_location`` with the
``opti_oignon`` package stubbed, so the suite collects without the backend.
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
tp = _ensure_agent("tool_parsing")


def _names(calls):
    return [c.name for c in calls]


# Module shape


class TestModuleShape:
    def test_sentinels(self):
        assert tp.checkpoint_before_apply is True
        assert tp.FEATURE_AVAILABLE is True

    def test_supported_formats(self):
        assert tp.SUPPORTED_FORMATS == ("fenced", "bracketed", "xml")

    def test_json_repair_guard_flag_is_bool(self):
        assert isinstance(tp.JSON_REPAIR_AVAILABLE, bool)

    def test_parsed_tool_call_to_dict(self):
        c = tp.ParsedToolCall(name="bash", arguments={"command": "ls"}, source="fenced")
        d = c.to_dict()
        assert d == {"name": "bash", "arguments": {"command": "ls"}, "source": "fenced"}


# Fenced code blocks


class TestFenced:
    def test_tool_key(self):
        text = '```json\n{"tool": "bash", "arguments": {"command": "ls -la"}}\n```'
        calls = tp.parse_fenced_blocks(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "ls -la"}
        assert calls[0].source == "fenced"

    def test_name_key_and_no_language_hint(self):
        text = '```\n{"name": "view", "args": {"path": "/workspace/x"}}\n```'
        calls = tp.parse_fenced_blocks(text)
        assert calls[0].name == "view"
        assert calls[0].arguments == {"path": "/workspace/x"}

    def test_tool_name_key_and_parameters_key(self):
        text = '```tool\n{"tool_name": "str_replace", "parameters": {"path": "a", "old_str": "x"}}\n```'
        calls = tp.parse_fenced_blocks(text)
        assert calls[0].name == "str_replace"
        assert calls[0].arguments == {"path": "a", "old_str": "x"}

    def test_nested_openai_function_object(self):
        text = '```json\n{"function": {"name": "bash", "arguments": {"command": "id"}}}\n```'
        calls = tp.parse_fenced_blocks(text)
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "id"}

    def test_arguments_as_json_string(self):
        text = '```json\n{"tool": "bash", "arguments": "{\\"command\\": \\"pwd\\"}"}\n```'
        calls = tp.parse_fenced_blocks(text)
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "pwd"}

    def test_plain_code_fence_does_not_misfire(self):
        text = "```python\nprint('hello {world}')\nx = {1: 2}\n```"
        assert tp.parse_fenced_blocks(text) == []

    def test_json_without_tool_name_is_ignored(self):
        text = '```json\n{"result": 42, "items": [1, 2, 3]}\n```'
        assert tp.parse_fenced_blocks(text) == []

    def test_multiple_fenced_blocks(self):
        text = (
            '```json\n{"tool": "bash", "arguments": {"command": "ls"}}\n```\n'
            "some prose\n"
            '```json\n{"tool": "view", "arguments": {"path": "/x"}}\n```\n'
        )
        assert _names(tp.parse_fenced_blocks(text)) == ["bash", "view"]

    def test_malformed_json_is_repaired_or_skipped(self):
        # Trailing comma: json.loads fails; the guarded repairer recovers it.
        text = '```json\n{"tool": "bash", "arguments": {"command": "ls",}}\n```'
        calls = tp.parse_fenced_blocks(text)
        if tp.JSON_REPAIR_AVAILABLE:
            assert calls and calls[0].name == "bash"
        else:
            assert calls == []


# Bracketed [TOOL_CALL] blocks


class TestBracketed:
    def test_closed_block(self):
        text = '[TOOL_CALL]{"name": "view", "args": {"path": "/workspace/a"}}[/TOOL_CALL]'
        calls = tp.parse_bracketed_blocks(text)
        assert calls[0].name == "view"
        assert calls[0].arguments == {"path": "/workspace/a"}
        assert calls[0].source == "bracketed"

    def test_unclosed_block(self):
        text = 'I will call: [TOOL_CALL]{"tool": "bash", "arguments": {"command": "uname -a"}}'
        calls = tp.parse_bracketed_blocks(text)
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "uname -a"}

    def test_nested_braces_in_arguments(self):
        text = '[TOOL_CALL]{"name": "x", "args": {"a": {"b": [1, 2]}, "c": "}"}}[/TOOL_CALL]'
        calls = tp.parse_bracketed_blocks(text)
        assert calls[0].arguments == {"a": {"b": [1, 2]}, "c": "}"}

    def test_case_insensitive_marker(self):
        text = '[tool_call]{"tool": "bash", "arguments": {"command": "ls"}}[/tool_call]'
        calls = tp.parse_bracketed_blocks(text)
        assert calls[0].name == "bash"

    def test_no_json_after_marker_is_skipped(self):
        text = "[TOOL_CALL] please run something [/TOOL_CALL]"
        assert tp.parse_bracketed_blocks(text) == []


# XML invoke / param blocks


class TestXml:
    def test_single_param(self):
        text = '<invoke name="bash"><param name="command">ls -la</param></invoke>'
        calls = tp.parse_xml_blocks(text)
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "ls -la"}
        assert calls[0].source == "xml"

    def test_multiple_params_and_parameter_alias(self):
        text = (
            '<invoke name="create_file">'
            '<param name="path">/workspace/x.py</param>'
            '<parameter name="timeout">30</parameter>'
            "</invoke>"
        )
        calls = tp.parse_xml_blocks(text)
        assert calls[0].name == "create_file"
        assert calls[0].arguments == {"path": "/workspace/x.py", "timeout": 30}

    def test_scalar_coercion(self):
        text = (
            '<invoke name="t">'
            '<param name="n">42</param>'
            '<param name="b">true</param>'
            '<param name="obj">{"k": 1}</param>'
            '<param name="s">a plain string</param>'
            "</invoke>"
        )
        args = tp.parse_xml_blocks(text)[0].arguments
        assert args["n"] == 42
        assert args["b"] is True
        assert args["obj"] == {"k": 1}
        assert args["s"] == "a plain string"

    def test_single_quoted_name_attr(self):
        text = "<invoke name='view'><param name='path'>/x</param></invoke>"
        calls = tp.parse_xml_blocks(text)
        assert calls[0].name == "view"
        assert calls[0].arguments == {"path": "/x"}

    def test_invoke_without_name_is_skipped(self):
        text = "<invoke><param name='path'>/x</param></invoke>"
        assert tp.parse_xml_blocks(text) == []


# Combined dispatch-facing entry point


class TestParseToolBlocks:
    def test_mixed_message_preserves_order(self):
        text = (
            '```json\n{"tool": "bash", "arguments": {"command": "ls"}}\n```\n'
            '[TOOL_CALL]{"name": "view", "args": {"path": "/x"}}[/TOOL_CALL]\n'
            '<invoke name="str_replace"><param name="path">a</param></invoke>'
        )
        calls = tp.parse_tool_blocks(text)
        assert _names(calls) == ["bash", "view", "str_replace"]
        assert [c.source for c in calls] == ["fenced", "bracketed", "xml"]

    def test_exact_duplicate_across_formats_is_deduped(self):
        text = (
            '```json\n{"tool": "bash", "arguments": {"command": "ls"}}\n```\n'
            '[TOOL_CALL]{"name": "bash", "args": {"command": "ls"}}[/TOOL_CALL]'
        )
        calls = tp.parse_tool_blocks(text)
        assert len(calls) == 1
        # The first match wins and keeps its source.
        assert calls[0].source == "fenced"

    def test_distinct_arguments_not_deduped(self):
        text = (
            '[TOOL_CALL]{"name": "bash", "args": {"command": "ls"}}[/TOOL_CALL]\n'
            '[TOOL_CALL]{"name": "bash", "args": {"command": "pwd"}}[/TOOL_CALL]'
        )
        calls = tp.parse_tool_blocks(text)
        assert len(calls) == 2

    def test_no_tool_call_returns_empty(self):
        assert tp.parse_tool_blocks("just a normal sentence with no tools.") == []

    def test_empty_input_returns_empty(self):
        assert tp.parse_tool_blocks("") == []

    def test_has_tool_call(self):
        assert tp.has_tool_call('[TOOL_CALL]{"tool": "bash", "arguments": {}}[/TOOL_CALL]')
        assert not tp.has_tool_call("nothing here")


# Normalisation


class TestNormalisation:
    def test_name_whitespace_stripped(self):
        text = '```json\n{"tool": "  bash  ", "arguments": {}}\n```'
        assert tp.parse_fenced_blocks(text)[0].name == "bash"

    def test_arguments_always_dict_when_missing(self):
        text = '```json\n{"tool": "bash"}\n```'
        calls = tp.parse_fenced_blocks(text)
        assert calls[0].arguments == {}
        assert isinstance(calls[0].arguments, dict)

    def test_non_object_arguments_become_empty_dict(self):
        text = '```json\n{"tool": "bash", "arguments": [1, 2, 3]}\n```'
        assert tp.parse_fenced_blocks(text)[0].arguments == {}

    def test_raw_substring_is_retained(self):
        text = 'pre [TOOL_CALL]{"name": "bash", "args": {}}[/TOOL_CALL] post'
        c = tp.parse_bracketed_blocks(text)[0]
        assert "[TOOL_CALL]" in c.raw and "[/TOOL_CALL]" in c.raw
