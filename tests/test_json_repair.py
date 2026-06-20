#!/usr/bin/env python3
"""
Tests for JSON Repair module (S80).

Covers: markdown fence stripping, JSON substring extraction,
trailing comma fix, single quote fix, comment stripping,
unescaped newline fix, missing closing bracket fix,
numbered list fallback, main repair pipeline.
"""

import importlib.util
import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Module loading (test isolation -- no ollama needed)
# ---------------------------------------------------------------------------

_base = os.path.join(os.path.dirname(__file__), os.pardir, "opti_oignon")

_jr_path = os.path.join(_base, "json_repair.py")
_jr_spec = importlib.util.spec_from_file_location("json_repair", _jr_path)
_jr_mod = importlib.util.module_from_spec(_jr_spec)
_jr_spec.loader.exec_module(_jr_mod)

strip_markdown_fences = _jr_mod.strip_markdown_fences
extract_json_substring = _jr_mod.extract_json_substring
fix_trailing_commas = _jr_mod.fix_trailing_commas
fix_single_quotes = _jr_mod.fix_single_quotes
strip_comments = _jr_mod.strip_comments
fix_unescaped_newlines = _jr_mod.fix_unescaped_newlines
fix_missing_closing = _jr_mod.fix_missing_closing
parse_numbered_list = _jr_mod.parse_numbered_list
repair_json = _jr_mod.repair_json
repair_json_or_list = _jr_mod.repair_json_or_list
JSON_RETRY_SUFFIX = _jr_mod.JSON_RETRY_SUFFIX
SIMPLIFIED_PLAN_SUFFIX = _jr_mod.SIMPLIFIED_PLAN_SUFFIX

_infer_step_type = _jr_mod._infer_step_type
_extract_file_path = _jr_mod._extract_file_path
_extract_command = _jr_mod._extract_command


# ===================================================================
# Markdown fence stripping
# ===================================================================

class TestStripMarkdownFences:
    """Tests for strip_markdown_fences()."""

    def test_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_json_uppercase_fence(self):
        text = '```JSON\n{"a": 1}\n```'
        assert strip_markdown_fences(text) == '{"a": 1}'

    def test_bare_fence(self):
        text = '```\n{"a": 1}\n```'
        assert strip_markdown_fences(text) == '{"a": 1}'

    def test_js_fence(self):
        text = '```js\n{"a": 1}\n```'
        assert strip_markdown_fences(text) == '{"a": 1}'

    def test_javascript_fence(self):
        text = '```javascript\n{"a": 1}\n```'
        assert strip_markdown_fences(text) == '{"a": 1}'

    def test_no_fence(self):
        text = '{"key": "value"}'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_multiline_content(self):
        text = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        result = strip_markdown_fences(text)
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_whitespace_around_fences(self):
        text = '  ```json\n{"a": 1}\n```  '
        result = strip_markdown_fences(text)
        assert json.loads(result) == {"a": 1}

    def test_opening_fence_no_closing(self):
        text = '```json\n{"a": 1}'
        result = strip_markdown_fences(text)
        assert json.loads(result) == {"a": 1}

    def test_multiple_fenced_blocks_returns_first(self):
        text = '```json\n{"first": true}\n```\nSome text\n```json\n{"second": true}\n```'
        result = strip_markdown_fences(text)
        assert json.loads(result) == {"first": True}

    def test_empty_fence(self):
        text = '```json\n\n```'
        result = strip_markdown_fences(text)
        assert result == ""


# ===================================================================
# JSON substring extraction
# ===================================================================

class TestExtractJsonSubstring:
    """Tests for extract_json_substring()."""

    def test_embedded_object(self):
        text = 'Here is the plan: {"a": 1} and more text'
        result = extract_json_substring(text)
        assert json.loads(result) == {"a": 1}

    def test_embedded_array(self):
        text = 'Result: [1, 2, 3] done'
        result = extract_json_substring(text)
        assert json.loads(result) == [1, 2, 3]

    def test_nested_objects(self):
        text = 'Output: {"a": {"b": {"c": 1}}} end'
        result = extract_json_substring(text)
        assert json.loads(result) == {"a": {"b": {"c": 1}}}

    def test_mixed_brackets(self):
        text = 'Data: {"items": [1, 2, 3]} trailing'
        result = extract_json_substring(text)
        assert json.loads(result) == {"items": [1, 2, 3]}

    def test_string_with_brackets(self):
        text = 'Output: {"msg": "hello {world}"} end'
        result = extract_json_substring(text)
        assert json.loads(result) == {"msg": "hello {world}"}

    def test_no_json(self):
        text = "This is plain text without any JSON"
        assert extract_json_substring(text) is None

    def test_object_before_array(self):
        text = 'A {"x": 1} B [2] C'
        result = extract_json_substring(text)
        assert json.loads(result) == {"x": 1}

    def test_array_before_object(self):
        text = 'A [1, 2] B {"x": 3} C'
        result = extract_json_substring(text)
        assert json.loads(result) == [1, 2]

    def test_escaped_quotes_in_string(self):
        text = r'Res: {"msg": "say \"hello\""} done'
        result = extract_json_substring(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["msg"] == 'say "hello"'

    def test_truncated_json_returns_rest(self):
        text = 'Start: {"a": 1, "b": '
        result = extract_json_substring(text)
        # Should return from { to end since brackets never balance
        assert result is not None
        assert result.startswith("{")


# ===================================================================
# Trailing comma fix
# ===================================================================

class TestFixTrailingCommas:
    """Tests for fix_trailing_commas()."""

    def test_object_trailing_comma(self):
        text = '{"a": 1, "b": 2,}'
        result = fix_trailing_commas(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_array_trailing_comma(self):
        text = '[1, 2, 3,]'
        result = fix_trailing_commas(text)
        assert json.loads(result) == [1, 2, 3]

    def test_nested_trailing_commas(self):
        text = '{"a": [1, 2,], "b": {"c": 3,},}'
        result = fix_trailing_commas(text)
        assert json.loads(result) == {"a": [1, 2], "b": {"c": 3}}

    def test_trailing_comma_with_whitespace(self):
        text = '{"a": 1 ,  \n}'
        result = fix_trailing_commas(text)
        assert json.loads(result) == {"a": 1}

    def test_no_trailing_comma(self):
        text = '{"a": 1, "b": 2}'
        assert fix_trailing_commas(text) == text

    def test_comma_in_string_preserved(self):
        text = '{"msg": "a, b, c,"}'
        # The comma inside the string value is not a trailing comma
        # but the pattern might match it; verify the output is valid
        result = fix_trailing_commas(text)
        # The regex only targets commas before } or ], so string commas
        # not followed by }] are safe
        assert "a, b, c" in result


# ===================================================================
# Single quote fix
# ===================================================================

class TestFixSingleQuotes:
    """Tests for fix_single_quotes()."""

    def test_single_quoted_keys_and_values(self):
        text = "{'key': 'value'}"
        result = fix_single_quotes(text)
        assert json.loads(result) == {"key": "value"}

    def test_mixed_quotes(self):
        text = '{\'a\': "b", "c": \'d\'}'
        result = fix_single_quotes(text)
        parsed = json.loads(result)
        assert parsed["a"] == "b"
        assert parsed["c"] == "d"

    def test_nested_single_quotes(self):
        text = "{'items': ['x', 'y']}"
        result = fix_single_quotes(text)
        assert json.loads(result) == {"items": ["x", "y"]}

    def test_no_single_quotes(self):
        text = '{"a": "b"}'
        result = fix_single_quotes(text)
        assert json.loads(result) == {"a": "b"}

    def test_double_quote_inside_single(self):
        text = "{'msg': 'He said \"hi\"'}"
        result = fix_single_quotes(text)
        parsed = json.loads(result)
        assert "hi" in parsed["msg"]


# ===================================================================
# Comment stripping
# ===================================================================

class TestStripComments:
    """Tests for strip_comments()."""

    def test_line_comment(self):
        text = '{"a": 1, // this is a comment\n"b": 2}'
        result = strip_comments(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_block_comment(self):
        text = '{"a": 1, /* block */ "b": 2}'
        result = strip_comments(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_comment_inside_string_preserved(self):
        text = '{"url": "http://example.com"}'
        result = strip_comments(text)
        assert json.loads(result) == {"url": "http://example.com"}

    def test_multiline_block_comment(self):
        text = '{"a": 1,\n/* this\nis\na\ncomment */\n"b": 2}'
        result = strip_comments(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_no_comments(self):
        text = '{"a": 1}'
        assert strip_comments(text) == text

    def test_double_slash_in_string_preserved(self):
        text = '{"path": "C://files//data"}'
        result = strip_comments(text)
        assert json.loads(result)["path"] == "C://files//data"


# ===================================================================
# Unescaped newline fix
# ===================================================================

class TestFixUnescapedNewlines:
    """Tests for fix_unescaped_newlines()."""

    def test_newline_in_string(self):
        text = '{"msg": "line1\nline2"}'
        result = fix_unescaped_newlines(text)
        parsed = json.loads(result)
        assert parsed["msg"] == "line1\nline2"

    def test_tab_in_string(self):
        text = '{"msg": "col1\tcol2"}'
        result = fix_unescaped_newlines(text)
        parsed = json.loads(result)
        assert parsed["msg"] == "col1\tcol2"

    def test_no_special_chars(self):
        text = '{"a": "normal"}'
        result = fix_unescaped_newlines(text)
        assert result == text

    def test_already_escaped(self):
        text = '{"a": "line1\\nline2"}'
        result = fix_unescaped_newlines(text)
        # Already escaped, should not double-escape
        assert json.loads(result)["a"] == "line1\nline2"

    def test_newlines_outside_strings_preserved(self):
        text = '{\n"a": 1,\n"b": 2\n}'
        result = fix_unescaped_newlines(text)
        assert json.loads(result) == {"a": 1, "b": 2}


# ===================================================================
# Missing closing bracket fix
# ===================================================================

class TestFixMissingClosing:
    """Tests for fix_missing_closing()."""

    def test_missing_closing_brace(self):
        text = '{"a": 1, "b": 2'
        result = fix_missing_closing(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_missing_closing_bracket(self):
        text = '[1, 2, 3'
        result = fix_missing_closing(text)
        assert json.loads(result) == [1, 2, 3]

    def test_missing_nested_closings(self):
        text = '{"a": [1, 2'
        result = fix_missing_closing(text)
        assert json.loads(result) == {"a": [1, 2]}

    def test_trailing_comma_before_close(self):
        text = '{"a": 1,'
        result = fix_missing_closing(text)
        assert json.loads(result) == {"a": 1}

    def test_already_complete(self):
        text = '{"a": 1}'
        result = fix_missing_closing(text)
        assert result == text

    def test_deeply_nested(self):
        text = '{"a": {"b": {"c": [1'
        result = fix_missing_closing(text)
        assert json.loads(result) == {"a": {"b": {"c": [1]}}}


# ===================================================================
# Numbered list fallback
# ===================================================================

class TestParseNumberedList:
    """Tests for parse_numbered_list() and helpers."""

    def test_simple_list(self):
        text = "1. Create file main.py\n2. Edit utils.py\n3. Run tests"
        result = parse_numbered_list(text)
        assert result is not None
        assert len(result) == 3
        assert result[0]["step_type"] == "create"
        assert result[1]["step_type"] == "edit"
        assert result[2]["step_type"] == "test"

    def test_list_with_file_paths(self):
        text = "1. Create file opti_oignon/helper.py with utility functions"
        result = parse_numbered_list(text)
        assert result is not None
        assert result[0]["file_path"] == "opti_oignon/helper.py"

    def test_list_with_commands(self):
        text = "1. Run `pytest tests/` to verify"
        result = parse_numbered_list(text)
        assert result is not None
        assert result[0]["command"] == "pytest tests/"

    def test_no_numbered_list(self):
        text = "This is just plain text without any steps."
        assert parse_numbered_list(text) is None

    def test_list_with_preamble(self):
        text = "Here is my plan:\n1. First step\n2. Second step"
        result = parse_numbered_list(text)
        assert result is not None
        assert len(result) == 2

    def test_infer_bash_type(self):
        assert _infer_step_type("install dependencies") == "bash"

    def test_infer_create_type(self):
        assert _infer_step_type("Create new file") == "create"
        assert _infer_step_type("Write helper module") == "create"

    def test_infer_edit_type(self):
        assert _infer_step_type("Edit main.py to add import") == "edit"
        assert _infer_step_type("Modify the config") == "edit"
        assert _infer_step_type("Fix the bug in utils") == "edit"

    def test_infer_test_type(self):
        assert _infer_step_type("Run pytest to verify") == "test"
        assert _infer_step_type("Test the new feature") == "test"

    def test_extract_file_path_py(self):
        assert _extract_file_path("Create opti_oignon/helper.py") == "opti_oignon/helper.py"

    def test_extract_file_path_yaml(self):
        assert _extract_file_path("Edit config/settings.yaml") == "config/settings.yaml"

    def test_extract_file_path_none(self):
        assert _extract_file_path("Run all tests") == ""

    def test_extract_command(self):
        assert _extract_command("Run `pytest -x` to verify") == "pytest -x"

    def test_extract_command_none(self):
        assert _extract_command("Create a new file") == ""


# ===================================================================
# Main repair pipeline
# ===================================================================

class TestRepairJson:
    """Tests for repair_json() main pipeline."""

    def test_valid_json_passthrough(self):
        text = '{"summary": "test", "steps": []}'
        assert repair_json(text) == {"summary": "test", "steps": []}

    def test_fenced_json(self):
        text = '```json\n{"summary": "plan", "steps": []}\n```'
        assert repair_json(text) == {"summary": "plan", "steps": []}

    def test_embedded_json(self):
        text = 'Sure! Here is the plan:\n{"summary": "test", "steps": []}\nHope that helps!'
        assert repair_json(text) == {"summary": "test", "steps": []}

    def test_trailing_comma_repair(self):
        text = '{"summary": "test", "steps": [],}'
        assert repair_json(text) == {"summary": "test", "steps": []}

    def test_single_quote_repair(self):
        text = "{'summary': 'test', 'steps': []}"
        assert repair_json(text) == {"summary": "test", "steps": []}

    def test_comment_repair(self):
        text = '{"summary": "test", // planning\n"steps": []}'
        assert repair_json(text) == {"summary": "test", "steps": []}

    def test_truncated_json_repair(self):
        text = '{"summary": "test", "steps": [{"step_type": "create"'
        result = repair_json(text)
        assert result["summary"] == "test"

    def test_combined_issues(self):
        text = "```json\n{'summary': 'plan', 'steps': [1, 2,],}\n```"
        result = repair_json(text)
        assert result["summary"] == "plan"
        assert result["steps"] == [1, 2]

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="Empty input"):
            repair_json("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="Empty input"):
            repair_json("   \n\n  ")

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="Failed to repair"):
            repair_json("this is not json at all")

    def test_realistic_llm_plan(self):
        text = '''Sure, here's the plan:

```json
{
  "summary": "Add logging to utils module",
  "estimated_files": 2,
  "steps": [
    {
      "step_type": "edit",
      "description": "Add import logging to utils.py",
      "file_path": "utils.py",
      "old_str": "import os",
      "new_str": "import os\\nimport logging"
    },
    {
      "step_type": "test",
      "description": "Run tests to verify",
      "command": "pytest tests/"
    }
  ]
}
```

Let me know if you want changes!'''
        result = repair_json(text)
        assert result["summary"] == "Add logging to utils module"
        assert len(result["steps"]) == 2
        assert result["steps"][0]["step_type"] == "edit"

    def test_array_result(self):
        text = '[{"a": 1}, {"b": 2}]'
        result = repair_json(text)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_fenced_with_leading_text(self):
        text = 'I will create a plan:\n```json\n{"a": 1}\n```\nDone.'
        assert repair_json(text) == {"a": 1}


# ===================================================================
# Combined repair + list fallback
# ===================================================================

class TestRepairJsonOrList:
    """Tests for repair_json_or_list()."""

    def test_valid_json_returns_json(self):
        json_result, list_result = repair_json_or_list('{"a": 1}')
        assert json_result == {"a": 1}
        assert list_result is None

    def test_numbered_list_fallback(self):
        text = "1. Create main.py\n2. Edit config.yaml\n3. Run tests"
        json_result, list_result = repair_json_or_list(text)
        assert json_result is None
        assert list_result is not None
        assert len(list_result) == 3

    def test_both_none_on_garbage(self):
        json_result, list_result = repair_json_or_list("random garbage")
        assert json_result is None
        assert list_result is None

    def test_json_preferred_over_list(self):
        # If text contains both JSON and a numbered list, JSON wins
        text = '{"steps": []}\n1. First step\n2. Second step'
        json_result, list_result = repair_json_or_list(text)
        assert json_result is not None
        assert list_result is None


# ===================================================================
# Prompt constants
# ===================================================================

class TestPromptConstants:
    """Tests for retry prompt constants."""

    def test_retry_suffix_exists(self):
        assert "ONLY" in JSON_RETRY_SUFFIX
        assert "JSON" in JSON_RETRY_SUFFIX

    def test_simplified_suffix_exists(self):
        assert "numbered list" in SIMPLIFIED_PLAN_SUFFIX
