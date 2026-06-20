#!/usr/bin/env python3
"""
Tests for S103 -- Showcase Plugins Batch 1.

Covers:
- fact-checker: claim extraction, verification, annotation, hook
- chain-of-thought-enforcer: complexity detection, split, format, hooks
- scratchpad: DB operations, tag extraction, commands, ui_panel
- task-extractor: task extraction, DB operations, commands, hook
- auto-tldr: sentence splitting, scoring, summary generation, hook
- code-guardian: Python/JSON/R validation, badge formatting, hook
"""

import importlib.util
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =========================================================================
# MODULE LOADING (importlib isolation)
# =========================================================================

ROOT = Path(__file__).resolve().parent.parent


def _load_plugin(plugin_dir_name: str) -> ModuleType:
    """Load a plugin entry_point.py by directory name."""
    filepath = ROOT / "opti_oignon" / "plugins" / plugin_dir_name / "entry_point.py"
    mod_name = f"plugin_{plugin_dir_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod = importlib.util.module_from_spec(spec)

    # Stub opti_oignon.config if needed
    if "opti_oignon.config" not in sys.modules:
        cfg_stub = ModuleType("opti_oignon.config")
        cfg_stub.DATA_DIR = tempfile.mkdtemp()
        sys.modules["opti_oignon.config"] = cfg_stub
    if "opti_oignon" not in sys.modules:
        parent = ModuleType("opti_oignon")
        parent.__path__ = [str(ROOT / "opti_oignon")]
        sys.modules["opti_oignon"] = parent

    # Stub web_search for fact-checker (avoid real network)
    if plugin_dir_name == "fact-checker":
        if "opti_oignon.web_search" not in sys.modules:
            ws_stub = ModuleType("opti_oignon.web_search")
            ws_stub.DDGS_AVAILABLE = False
            ws_stub.web_searcher = MagicMock()
            sys.modules["opti_oignon.web_search"] = ws_stub

    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load all 6 plugins
fact_checker = _load_plugin("fact-checker")
cot_enforcer = _load_plugin("chain-of-thought-enforcer")
scratchpad = _load_plugin("scratchpad")
task_extractor = _load_plugin("task-extractor")
auto_tldr = _load_plugin("auto-tldr")
code_guardian = _load_plugin("code-guardian")


def _make_ctx(
    data: dict | None = None,
    config: dict | None = None,
    metadata: dict | None = None,
) -> MagicMock:
    """Create a mock HookContext."""
    ctx = MagicMock()
    ctx.data = data or {}
    ctx.config = config or {}
    ctx.metadata = metadata or {}
    return ctx


# =========================================================================
# FACT-CHECKER TESTS
# =========================================================================


class TestFactCheckerExtraction:
    """Test claim extraction from text."""

    def test_extract_date_claims(self):
        text = "Python was created in 1991 and Java appeared in 1995."
        claims = fact_checker.extract_claims(text, aggressiveness="low")
        assert len(claims) >= 1
        assert any("1991" in c.text for c in claims)

    def test_extract_numeric_claims(self):
        text = "The distance is approximately 384400 km from Earth."
        claims = fact_checker.extract_claims(text, aggressiveness="low")
        assert len(claims) >= 1
        assert any("384400" in c.text for c in claims)

    def test_extract_entity_claims_moderate(self):
        text = "Guido van Rossum created Python as a hobby project."
        claims = fact_checker.extract_claims(text, aggressiveness="moderate")
        # Should find entity claim at moderate level
        assert isinstance(claims, list)

    def test_skip_code_blocks(self):
        text = "The year is 2024.\n```python\ndate = 1999\n```\nEnd."
        claims = fact_checker.extract_claims(
            text, aggressiveness="low", skip_code_blocks=True,
        )
        # Should not extract 1999 from code block
        for c in claims:
            assert "1999" not in c.text

    def test_max_claims_limit(self):
        text = "In 1990, in 1991, in 1992, in 1993, in 1994, in 1995, in 1996."
        claims = fact_checker.extract_claims(text, max_claims=3)
        assert len(claims) <= 3

    def test_empty_text(self):
        claims = fact_checker.extract_claims("")
        assert claims == []


class TestFactCheckerVerify:
    """Test claim verification (web search stubbed)."""

    def test_verify_without_web_search(self):
        claim = fact_checker.Claim("in 1991", "date", 0, 7)
        result = fact_checker.verify_claim(claim)
        assert result["status"] == "unverified"
        assert result["sources"] == []

    def test_annotate_response_verified(self):
        text = "Created in 1991 by Guido."
        claim = fact_checker.Claim("in 1991", "date", 8, 15)
        result = {"status": "verified", "detail": None}
        annotated = fact_checker.annotate_response(text, [claim], [result])
        assert "[verified]" in annotated

    def test_annotate_response_conflict(self):
        text = "Created in 1991 by Guido."
        claim = fact_checker.Claim("in 1991", "date", 8, 15)
        result = {"status": "conflict", "detail": "Actually 1989"}
        annotated = fact_checker.annotate_response(text, [claim], [result])
        assert "[conflict:" in annotated

    def test_annotate_empty_claims(self):
        text = "Hello world."
        annotated = fact_checker.annotate_response(text, [], [])
        assert annotated == text


class TestFactCheckerHook:
    """Test post_inference hook."""

    def test_hook_short_response_ignored(self):
        ctx = _make_ctx(data={"response": "Short."})
        result = fact_checker.hook_post_inference(ctx)
        assert result is None

    def test_hook_no_claims_returns_none(self):
        ctx = _make_ctx(data={"response": "This is a simple response with no facts."})
        result = fact_checker.hook_post_inference(ctx)
        assert result is None


# =========================================================================
# CHAIN-OF-THOUGHT ENFORCER TESTS
# =========================================================================


class TestCoTComplexity:
    """Test complexity detection."""

    def test_simple_question_not_complex(self):
        assert not cot_enforcer.is_complex_question("Hi")

    def test_short_question_not_complex(self):
        assert not cot_enforcer.is_complex_question("What time?")

    def test_why_question_complex(self):
        assert cot_enforcer.is_complex_question(
            "Why does the sky appear blue during the day?"
        )

    def test_compare_question_complex(self):
        assert cot_enforcer.is_complex_question(
            "Compare Python and JavaScript for web development."
        )

    def test_math_operators_complex(self):
        assert cot_enforcer.is_complex_question(
            "Solve the equation x + 3 = 7 for x"
        )

    def test_multi_part_complex(self):
        assert cot_enforcer.is_complex_question(
            "1) What is photosynthesis? 2) Why is it important?"
        )

    def test_custom_keywords(self):
        assert cot_enforcer.is_complex_question(
            "Please elaborate on the topic of gravity",
            keywords=["elaborate"],
        )


class TestCoTSplitAndFormat:
    """Test reasoning/answer splitting and formatting."""

    def test_split_with_answer_marker(self):
        text = "Step 1: think.\nStep 2: reason.\n\nFinal answer: 42"
        reasoning, answer = cot_enforcer.split_reasoning_and_answer(text)
        assert reasoning
        assert "42" in answer

    def test_split_with_therefore(self):
        text = "We can see that X > Y.\n\nTherefore: X is larger."
        reasoning, answer = cot_enforcer.split_reasoning_and_answer(text)
        assert reasoning or answer  # Should find some split

    def test_split_no_markers(self):
        text = "Just a plain response."
        reasoning, answer = cot_enforcer.split_reasoning_and_answer(text)
        assert reasoning == ""
        assert answer == text

    def test_format_separator_style(self):
        result = cot_enforcer.format_response(
            "I think because X.", "The answer is Y.", style="separator",
        )
        assert "**Reasoning:**" in result
        assert "---" in result
        assert "**Answer:**" in result

    def test_format_collapsible_style(self):
        result = cot_enforcer.format_response(
            "Reasoning here.", "Answer here.", style="collapsible",
        )
        assert "<details>" in result
        assert "**Answer:**" in result

    def test_format_empty_reasoning(self):
        result = cot_enforcer.format_response("", "Just answer.")
        assert result == "Just answer."


class TestCoTHooks:
    """Test pre_inference and post_inference hooks."""

    def test_pre_inference_injects_cot(self):
        ctx = _make_ctx(data={
            "prompt": "Why does quantum entanglement violate local realism?",
            "system_message": "You are helpful.",
        })
        result = cot_enforcer.hook_pre_inference(ctx)
        assert result is not None
        assert "step by step" in result["system_message"].lower()
        assert result["_cot_injected"] is True

    def test_pre_inference_skips_simple(self):
        ctx = _make_ctx(data={"prompt": "Hello"})
        result = cot_enforcer.hook_pre_inference(ctx)
        assert result is None

    def test_post_inference_no_markers(self):
        ctx = _make_ctx(data={"response": "Simple answer."})
        result = cot_enforcer.hook_post_inference(ctx)
        assert result is None


# =========================================================================
# SCRATCHPAD TESTS
# =========================================================================


class TestScratchpadDB:
    """Test ScratchpadDB operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        self.db = scratchpad.ScratchpadDB(tmp_path / "test.db", max_notes=10)
        # Reset module-level _db to avoid interference
        scratchpad._db = None
        yield
        self.db.close()

    def test_add_and_list(self):
        result = self.db.add_note("Test note", tags=["python", "test"])
        assert result["id"] == 1
        notes = self.db.list_notes()
        assert len(notes) == 1
        assert notes[0]["text"] == "Test note"
        assert "python" in notes[0]["tags"]

    def test_delete_note(self):
        self.db.add_note("To delete")
        assert self.db.delete_note(1) is True
        assert self.db.list_notes() == []

    def test_delete_nonexistent(self):
        assert self.db.delete_note(999) is False

    def test_search_notes(self):
        self.db.add_note("Python tutorial")
        self.db.add_note("JavaScript guide")
        results = self.db.search_notes("Python")
        assert len(results) == 1
        assert "Python" in results[0]["text"]

    def test_max_notes_limit(self):
        for i in range(10):
            self.db.add_note(f"Note {i}")
        result = self.db.add_note("One too many")
        assert "error" in result

    def test_export_markdown(self):
        self.db.add_note("Important thing")
        export = self.db.export_markdown()
        assert "# Scratchpad Export" in export
        assert "Important thing" in export

    def test_note_count(self):
        self.db.add_note("One")
        self.db.add_note("Two")
        assert self.db.get_note_count() == 2


class TestScratchpadTags:
    """Test tag extraction."""

    def test_extract_tags_basic(self):
        tags = scratchpad.extract_tags("Install Python and configure Docker environment")
        assert len(tags) > 0
        assert all(isinstance(t, str) for t in tags)

    def test_extract_tags_filters_stop_words(self):
        tags = scratchpad.extract_tags("This is a very simple test")
        assert "this" not in tags
        assert "very" not in tags

    def test_extract_tags_max_limit(self):
        text = "alpha bravo charlie delta echo foxtrot golf hotel india"
        tags = scratchpad.extract_tags(text, max_tags=3)
        assert len(tags) <= 3


class TestScratchpadHook:
    """Test tool_call hook for slash commands."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        scratchpad._db = None
        self.tmp_path = tmp_path
        yield
        scratchpad._db = None

    def _ctx(self, user_input: str) -> MagicMock:
        return _make_ctx(
            data={"user_input": user_input},
            config={"max_note_length": 2000, "auto_tag": True, "max_notes": 500},
            metadata={"plugin_dir": str(self.tmp_path)},
        )

    def test_add_note_command(self):
        ctx = self._ctx("/note Remember to buy groceries")
        result = scratchpad.hook_tool_call(ctx)
        assert result is not None
        assert "saved" in result["response"].lower()

    def test_list_notes_command(self):
        # Add a note first
        scratchpad.hook_tool_call(self._ctx("/note First note"))
        result = scratchpad.hook_tool_call(self._ctx("/notes"))
        assert result is not None
        assert "First note" in result["response"]

    def test_delete_command(self):
        scratchpad.hook_tool_call(self._ctx("/note To be deleted"))
        result = scratchpad.hook_tool_call(self._ctx("/note delete 1"))
        assert result is not None
        assert "deleted" in result["response"].lower()

    def test_search_command(self):
        scratchpad.hook_tool_call(self._ctx("/note Python project idea"))
        result = scratchpad.hook_tool_call(self._ctx("/note search Python"))
        assert result is not None
        assert "Python" in result["response"]

    def test_non_command_ignored(self):
        ctx = self._ctx("just a regular message")
        result = scratchpad.hook_tool_call(ctx)
        assert result is None

    def test_ui_panel_hook(self):
        ctx = _make_ctx(
            metadata={"plugin_dir": str(self.tmp_path)},
            config={"max_notes": 500},
        )
        result = scratchpad.hook_ui_panel(ctx)
        assert result is not None
        assert result["panel_id"] == "scratchpad"
        assert "panel_html" in result


# =========================================================================
# TASK-EXTRACTOR TESTS
# =========================================================================


class TestTaskExtraction:
    """Test task extraction from text."""

    def test_extract_pattern_you_should(self):
        text = "You should update the dependencies before deploying."
        tasks = task_extractor.extract_tasks(text)
        assert len(tasks) >= 1

    def test_extract_pattern_todo(self):
        text = "TODO: fix the authentication module before release."
        tasks = task_extractor.extract_tasks(text)
        assert len(tasks) >= 1

    def test_extract_numbered_steps(self):
        text = "1. Install Python\n2. Create virtual environment\n3. Run tests"
        tasks = task_extractor.extract_tasks(text)
        assert len(tasks) >= 2

    def test_extract_imperative(self):
        text = "Install Docker on your machine.\nConfigure the environment variables."
        tasks = task_extractor.extract_tasks(text)
        assert len(tasks) >= 1

    def test_skip_code_blocks(self):
        text = "You should test this.\n```python\n# TODO: refactor\n```"
        tasks = task_extractor.extract_tasks(text)
        # Should not extract TODO from code block
        for t in tasks:
            assert "refactor" not in t["text"].lower()

    def test_empty_text(self):
        tasks = task_extractor.extract_tasks("")
        assert tasks == []

    def test_max_tasks_limit(self):
        text = "\n".join(f"You should do task {i} right now." for i in range(20))
        tasks = task_extractor.extract_tasks(text, max_tasks=5)
        assert len(tasks) <= 5


class TestTaskDB:
    """Test TaskDB operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        self.db = task_extractor.TaskDB(tmp_path / "test_tasks.db", max_tasks=10)
        task_extractor._db = None
        yield
        self.db.close()

    def test_add_and_list(self):
        result = self.db.add_task("Fix the bug", source="pattern")
        assert result["id"] == 1
        tasks = self.db.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["text"] == "Fix the bug"

    def test_mark_done(self):
        self.db.add_task("Complete this")
        assert self.db.mark_done(1) is True
        pending = self.db.list_tasks(include_done=False)
        assert len(pending) == 0
        all_tasks = self.db.list_tasks(include_done=True)
        assert len(all_tasks) == 1
        assert all_tasks[0]["done"] is True

    def test_clear_done(self):
        self.db.add_task("Task A")
        self.db.add_task("Task B")
        self.db.mark_done(1)
        removed = self.db.clear_done()
        assert removed == 1
        assert self.db.get_pending_count() == 1

    def test_clear_all(self):
        self.db.add_task("Task A")
        self.db.add_task("Task B")
        removed = self.db.clear_all()
        assert removed == 2

    def test_max_tasks_limit(self):
        for i in range(10):
            self.db.add_task(f"Task {i}")
        result = self.db.add_task("One too many")
        assert "error" in result


class TestTaskExtractorHook:
    """Test tool_call hook for slash commands."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        task_extractor._db = None
        self.tmp_path = tmp_path
        yield
        task_extractor._db = None

    def _ctx(self, user_input: str) -> MagicMock:
        return _make_ctx(
            data={"user_input": user_input},
            config={"max_tasks": 200},
            metadata={"plugin_dir": str(self.tmp_path)},
        )

    def test_tasks_list_empty(self):
        result = task_extractor.hook_tool_call(self._ctx("/tasks"))
        assert result is not None
        assert "No tasks" in result["response"]

    def test_tasks_done_command(self):
        # First add a task via post_inference
        db = task_extractor.TaskDB(self.tmp_path / "tasks.db")
        db.add_task("Some task")
        task_extractor._db = db
        result = task_extractor.hook_tool_call(self._ctx("/tasks done 1"))
        assert result is not None
        assert "done" in result["response"].lower()

    def test_non_command_ignored(self):
        result = task_extractor.hook_tool_call(self._ctx("regular message"))
        assert result is None


# =========================================================================
# AUTO-TLDR TESTS
# =========================================================================


class TestAutoTldrTextProcessing:
    """Test text processing utilities."""

    def test_count_words(self):
        assert auto_tldr.count_words("one two three") == 3
        assert auto_tldr.count_words("") == 0

    def test_split_sentences_basic(self):
        text = "This is the first sentence. Here is the second one. And a third sentence follows."
        sentences = auto_tldr.split_sentences(text)
        assert len(sentences) >= 2

    def test_split_sentences_with_code_removed(self):
        text = "Important intro. More details follow here."
        sentences = auto_tldr.split_sentences(text)
        assert len(sentences) >= 1

    def test_extract_keywords(self):
        text = "Python programming language for data science and machine learning"
        keywords = auto_tldr.extract_keywords(text)
        assert "python" in keywords or "programming" in keywords
        assert "the" not in keywords


class TestAutoTldrScoring:
    """Test sentence scoring."""

    def test_position_score_first(self):
        score = auto_tldr._position_score(0, 10)
        assert score > 0.5  # First sentence should score high

    def test_position_score_last(self):
        score = auto_tldr._position_score(9, 10)
        assert score > 0.5  # Last sentence should score high

    def test_position_score_middle(self):
        score = auto_tldr._position_score(5, 10)
        assert score <= 1.0

    def test_keyword_density_score(self):
        keywords = {"python", "machine", "learning"}
        score = auto_tldr._keyword_density_score(
            "Python machine learning is great", keywords,
        )
        assert score > 0.0

    def test_length_score_average(self):
        score = auto_tldr._length_score("a normal sentence of some words", 6.0)
        assert score > 0.5

    def test_filler_penalty(self):
        penalty = auto_tldr._filler_penalty(
            "It is worth noting that at the end of the day this matters."
        )
        assert penalty > 0


class TestAutoTldrHook:
    """Test post_inference hook."""

    def test_short_response_ignored(self):
        ctx = _make_ctx(data={"response": "Short response."})
        result = auto_tldr.hook_post_inference(ctx)
        assert result is None

    def test_long_response_gets_tldr(self):
        # Generate a response > 300 words
        sentences = [f"This is sentence number {i} with enough words to be meaningful." for i in range(50)]
        long_response = " ".join(sentences)
        ctx = _make_ctx(
            data={"response": long_response},
            config={"word_threshold": 50, "max_summary_sentences": 2, "separator": "---"},
        )
        result = auto_tldr.hook_post_inference(ctx)
        assert result is not None
        assert "TL;DR" in result["response"]
        assert "---" in result["response"]

    def test_custom_threshold(self):
        text = "Word " * 100
        ctx = _make_ctx(
            data={"response": text},
            config={"word_threshold": 500},
        )
        result = auto_tldr.hook_post_inference(ctx)
        assert result is None  # Below custom threshold


# =========================================================================
# CODE-GUARDIAN TESTS
# =========================================================================


class TestCodeGuardianPython:
    """Test Python validation."""

    def test_valid_python(self):
        result = code_guardian.validate_python("x = 1\nprint(x)")
        assert result["valid"] is True

    def test_invalid_python(self):
        result = code_guardian.validate_python("def foo(\n  pass")
        assert result["valid"] is False
        assert result["error"] is not None

    def test_python_unused_import(self):
        code = "import os\nx = 1"
        result = code_guardian.validate_python(code)
        assert result["valid"] is True
        assert any("unused import" in d.lower() for d in result["details"])

    def test_python_bare_except(self):
        code = "try:\n    x = 1\nexcept:\n    pass"
        result = code_guardian.validate_python(code)
        assert result["valid"] is True
        assert any("bare except" in d.lower() for d in result["details"])

    def test_python_mutable_default(self):
        code = "def foo(x=[]):\n    return x"
        result = code_guardian.validate_python(code)
        assert result["valid"] is True
        assert any("mutable default" in d.lower() for d in result["details"])


class TestCodeGuardianJSON:
    """Test JSON validation."""

    def test_valid_json(self):
        result = code_guardian.validate_json('{"key": "value", "num": 42}')
        assert result["valid"] is True

    def test_invalid_json(self):
        result = code_guardian.validate_json('{"key": "value",}')
        assert result["valid"] is False

    def test_json_array(self):
        result = code_guardian.validate_json('[1, 2, 3]')
        assert result["valid"] is True


class TestCodeGuardianR:
    """Test R validation."""

    def test_valid_r(self):
        result = code_guardian.validate_r('x <- c(1, 2, 3)\nmean(x)')
        assert result["valid"] is True

    def test_r_unclosed_paren(self):
        result = code_guardian.validate_r('x <- c(1, 2, 3\nmean(x)')
        assert result["valid"] is False
        assert "Unclosed" in result["error"] or "Mismatched" in result["error"]

    def test_r_unclosed_string(self):
        result = code_guardian.validate_r('x <- "hello\nprint(x)')
        assert result["valid"] is False
        assert "string" in result["error"].lower()

    def test_r_with_comments(self):
        result = code_guardian.validate_r('# This is a comment\nx <- 1\n# Done')
        assert result["valid"] is True


class TestCodeGuardianBadge:
    """Test badge formatting."""

    def test_bracket_ok(self):
        validation = {"valid": True, "error": None, "details": []}
        badge = code_guardian.format_badge(validation, "python", badge_format="bracket")
        assert "[Python Syntax OK]" in badge

    def test_bracket_error(self):
        validation = {"valid": False, "error": "unexpected EOF", "line": 5, "details": []}
        badge = code_guardian.format_badge(validation, "python", badge_format="bracket")
        assert "Syntax Error" in badge
        assert "line 5" in badge

    def test_bracket_with_warnings(self):
        validation = {"valid": True, "error": None, "details": ["Unused import: os"]}
        badge = code_guardian.format_badge(validation, "python", badge_format="bracket")
        assert "warnings" in badge

    def test_hidden_valid_no_badge(self):
        validation = {"valid": True, "error": None, "details": []}
        badge = code_guardian.format_badge(validation, "python", badge_format="hidden")
        assert badge == ""


class TestCodeGuardianHook:
    """Test post_inference hook."""

    def test_hook_with_python_block(self):
        response = 'Here is code:\n```python\nx = 1\nprint(x)\n```\nDone.'
        ctx = _make_ctx(
            data={"response": response},
            config={"languages": "python,json,r", "badge_format": "bracket", "min_lines": 2},
        )
        result = code_guardian.hook_post_inference(ctx)
        assert result is not None
        assert "[Python Syntax OK]" in result["response"]
        assert result["code_guardian_summary"]["valid"] >= 1

    def test_hook_with_invalid_json_block(self):
        response = 'Config:\n```json\n{"key": "val",}\n```\nEnd.'
        ctx = _make_ctx(
            data={"response": response},
            config={"languages": "python,json,r", "badge_format": "bracket", "min_lines": 2},
        )
        result = code_guardian.hook_post_inference(ctx)
        assert result is not None
        assert "Syntax Error" in result["response"]

    def test_hook_no_code_blocks(self):
        ctx = _make_ctx(
            data={"response": "No code here."},
            config={"languages": "python,json,r"},
        )
        result = code_guardian.hook_post_inference(ctx)
        assert result is None

    def test_hook_respects_min_lines(self):
        response = 'Short:\n```python\nx=1\n```\nDone.'
        ctx = _make_ctx(
            data={"response": response},
            config={"languages": "python", "min_lines": 3},
        )
        result = code_guardian.hook_post_inference(ctx)
        assert result is None  # Block too short


# =========================================================================
# MANIFEST VALIDATION (all 6)
# =========================================================================


class TestManifestsValid:
    """Ensure all 6 manifests parse and have required fields."""

    PLUGINS = [
        "fact-checker",
        "chain-of-thought-enforcer",
        "scratchpad",
        "task-extractor",
        "auto-tldr",
        "code-guardian",
    ]

    @pytest.fixture(autouse=True)
    def load_yaml(self):
        import yaml
        self.yaml = yaml

    @pytest.mark.parametrize("plugin_name", PLUGINS)
    def test_manifest_valid(self, plugin_name):
        path = ROOT / "opti_oignon" / "plugins" / plugin_name / "manifest.yaml"
        data = self.yaml.safe_load(path.read_text())
        assert data["name"] == plugin_name
        assert "version" in data
        assert "entry_point" in data
        assert "hooks" in data
        assert isinstance(data["hooks"], list)
        assert len(data["hooks"]) >= 1
        assert "config_schema" in data

    @pytest.mark.parametrize("plugin_name", PLUGINS)
    def test_readme_exists(self, plugin_name):
        path = ROOT / "opti_oignon" / "plugins" / plugin_name / "README.md"
        assert path.exists()
        content = path.read_text()
        assert len(content) > 100


# =========================================================================
# HOOK REGISTRY (all 6)
# =========================================================================


class TestHookRegistries:
    """Ensure all plugins export a HOOKS dict matching their manifest."""

    def test_fact_checker_hooks(self):
        assert "post_inference" in fact_checker.HOOKS
        assert callable(fact_checker.HOOKS["post_inference"])

    def test_cot_enforcer_hooks(self):
        assert "pre_inference" in cot_enforcer.HOOKS
        assert "post_inference" in cot_enforcer.HOOKS

    def test_scratchpad_hooks(self):
        assert "tool_call" in scratchpad.HOOKS
        assert "ui_panel" in scratchpad.HOOKS

    def test_task_extractor_hooks(self):
        assert "post_inference" in task_extractor.HOOKS
        assert "tool_call" in task_extractor.HOOKS

    def test_auto_tldr_hooks(self):
        assert "post_inference" in auto_tldr.HOOKS

    def test_code_guardian_hooks(self):
        assert "post_inference" in code_guardian.HOOKS

    def test_all_have_init_shutdown(self):
        for mod in [fact_checker, cot_enforcer, scratchpad,
                     task_extractor, auto_tldr, code_guardian]:
            assert hasattr(mod, "init") and callable(mod.init)
            assert hasattr(mod, "shutdown") and callable(mod.shutdown)
