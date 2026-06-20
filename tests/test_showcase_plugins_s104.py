#!/usr/bin/env python3
"""
Tests for S104 -- Showcase Plugins Batch 2.

Covers:
- tone-shifter: mode rules, code block protection, custom rules, hook
- response-stats: syllable counting, FK readability, token estimation, hook
- markdown-beautifier: header spacing, list formatting, table alignment,
  fence repair, code block spacing, full pipeline, hook
- session-summarizer: extractive summarization, /summary command,
  /summary reset, post_inference tracking, background thread
- diff-tracker: code block extraction, name extraction, similarity,
  diff generation, history matching, hook
- github-connector: TokenStore CRUD, ref detection, URL detection,
  code block exclusion, command routing, footnote formatting, hook
"""

import importlib.util
import json
import math
import os
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch, Mock

import pytest

# =========================================================================
# MODULE LOADING (importlib isolation)
# =========================================================================

ROOT = Path(__file__).resolve().parent.parent


def _load_plugin(plugin_dir_name: str) -> ModuleType:
    """Load a plugin entry_point.py by directory name."""
    filepath = ROOT / "opti_oignon" / "plugins" / plugin_dir_name / "entry_point.py"
    mod_name = f"plugin_s104_{plugin_dir_name.replace('-', '_')}"
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

    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load all 6 plugins
tone_shifter = _load_plugin("tone-shifter")
response_stats = _load_plugin("response-stats")
markdown_beautifier = _load_plugin("markdown-beautifier")
session_summarizer = _load_plugin("session-summarizer")
diff_tracker = _load_plugin("diff-tracker")
github_connector = _load_plugin("github-connector")


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
# TONE-SHIFTER TESTS
# =========================================================================


class TestToneShifterModes:
    """Test individual tone transformation modes."""

    def test_academic_hedging(self):
        text = "This is definitely the best approach."
        result = tone_shifter.transform_tone(text, "academic")
        assert "likely" in result.lower() or "appears" in result.lower()

    def test_academic_formalize(self):
        text = "There are a lot of things to consider."
        result = tone_shifter.transform_tone(text, "academic")
        assert "significant number" in result or "elements" in result

    def test_casual_contractions(self):
        text = "I do not know. It is not clear."
        result = tone_shifter.transform_tone(text, "casual")
        assert "don't" in result
        assert "isn't" in result

    def test_casual_connectors(self):
        text = "Furthermore, we should consider this. However, there are issues."
        result = tone_shifter.transform_tone(text, "casual")
        assert "Also" in result
        assert "But" in result

    def test_eli5_simplification(self):
        text = "The algorithm optimizes the configuration parameters."
        result = tone_shifter.transform_tone(text, "eli5")
        assert "algorithm" not in result.lower() or "recipe" in result.lower()

    def test_formal_expand_contractions(self):
        text = "I don't think it's a good idea."
        result = tone_shifter.transform_tone(text, "formal")
        assert "do not" in result
        assert "it is" in result

    def test_concise_strip_filler(self):
        text = "It is worth noting that this approach works well."
        result = tone_shifter.transform_tone(text, "concise")
        assert "it is worth noting that" not in result.lower()

    def test_concise_compress(self):
        text = "She has the ability to solve complex problems."
        result = tone_shifter.transform_tone(text, "concise")
        assert "can" in result.lower()

    def test_verbose_expand(self):
        text = "Use this config e.g. for the repo."
        result = tone_shifter.transform_tone(text, "verbose")
        assert "for example" in result
        assert "configuration" in result


class TestToneShifterProtection:
    """Test code block protection."""

    def test_code_blocks_untouched(self):
        text = "Text outside. ```python\nThis is definitely code\n``` End."
        result = tone_shifter.transform_tone(text, "academic")
        assert "This is definitely code" in result

    def test_inline_code_untouched(self):
        text = "Use `definitely` as a variable name."
        result = tone_shifter.transform_tone(text, "academic")
        assert "`definitely`" in result

    def test_none_mode_passthrough(self):
        text = "This is definitely the best."
        result = tone_shifter.transform_tone(text, "none")
        assert result == text


class TestToneShifterHook:
    """Test the post_inference hook."""

    def test_hook_transforms(self):
        ctx = _make_ctx(
            data={"response": "I do not know the answer."},
            config={"active_mode": "casual"},
        )
        result = tone_shifter.hook_post_inference(ctx)
        assert result is not None
        assert "don't" in result["response"]

    def test_hook_none_mode(self):
        ctx = _make_ctx(
            data={"response": "Some text."},
            config={"active_mode": "none"},
        )
        assert tone_shifter.hook_post_inference(ctx) is None

    def test_hook_empty_response(self):
        ctx = _make_ctx(data={"response": ""}, config={"active_mode": "casual"})
        assert tone_shifter.hook_post_inference(ctx) is None

    def test_hook_custom_rules(self):
        ctx = _make_ctx(
            data={"response": "The foo is bar."},
            config={"active_mode": "none", "custom_rules": {"foo": "baz"}},
        )
        result = tone_shifter.hook_post_inference(ctx)
        assert result is not None
        assert "baz" in result["response"]

    def test_hook_unknown_mode(self):
        ctx = _make_ctx(
            data={"response": "Some text."},
            config={"active_mode": "nonexistent_mode"},
        )
        assert tone_shifter.hook_post_inference(ctx) is None


# =========================================================================
# RESPONSE-STATS TESTS
# =========================================================================


class TestResponseStatsSyllables:
    """Test syllable counting."""

    def test_monosyllable(self):
        assert response_stats.count_syllables("cat") == 1

    def test_multisyllable(self):
        assert response_stats.count_syllables("algorithm") >= 3

    def test_silent_e(self):
        # "code" should be 1 syllable (silent e removed)
        assert response_stats.count_syllables("code") == 1

    def test_empty(self):
        assert response_stats.count_syllables("") == 0


class TestResponseStatsMetrics:
    """Test readability metrics."""

    def test_flesch_kincaid_simple(self):
        text = "The cat sat on the mat. It was a nice day."
        grade = response_stats.flesch_kincaid_grade(text)
        assert 0 <= grade <= 25
        assert grade < 8  # simple text

    def test_flesch_kincaid_complex(self):
        text = (
            "The implementation of distributed consensus algorithms "
            "necessitates sophisticated Byzantine fault tolerance mechanisms "
            "to ensure consistency across heterogeneous replicated state machines."
        )
        grade = response_stats.flesch_kincaid_grade(text)
        assert grade > 10  # complex text

    def test_reading_ease_bounds(self):
        text = "Simple words here. Easy to read."
        ease = response_stats.flesch_reading_ease(text)
        assert 0 <= ease <= 100

    def test_complexity_labels(self):
        assert response_stats.complexity_label(3.0) == "simple"
        assert response_stats.complexity_label(8.0) == "moderate"
        assert response_stats.complexity_label(12.0) == "complex"
        assert response_stats.complexity_label(16.0) == "advanced"

    def test_token_estimation(self):
        text = "Hello world this is a test sentence."
        tokens = response_stats.estimate_tokens(text)
        words = response_stats.count_words(text)
        # Tokens should be roughly 1.3x words
        assert tokens >= words
        assert tokens < words * 3

    def test_reading_time(self):
        assert response_stats.reading_time_seconds(0) == 0
        secs = response_stats.reading_time_seconds(238)
        assert 55 <= secs <= 65  # ~1 minute at 238 WPM

    def test_format_reading_time_short(self):
        assert response_stats.format_reading_time(10) == "< 1 min"

    def test_format_reading_time_minutes(self):
        result = response_stats.format_reading_time(120)
        assert "2 min" in result


class TestResponseStatsHook:
    """Test the post_inference hook."""

    def test_hook_appends_footer(self):
        text = " ".join(["word"] * 50)
        ctx = _make_ctx(
            data={"response": text},
            config={"min_words": 20, "style": "compact"},
        )
        result = response_stats.hook_post_inference(ctx)
        assert result is not None
        assert "words" in result["response"]
        assert result["stats"]["words"] == 50

    def test_hook_below_threshold(self):
        ctx = _make_ctx(
            data={"response": "Short text."},
            config={"min_words": 100},
        )
        assert response_stats.hook_post_inference(ctx) is None

    def test_hook_detailed_style(self):
        text = " ".join(["word"] * 50)
        ctx = _make_ctx(
            data={"response": text},
            config={"style": "detailed", "min_words": 10},
        )
        result = response_stats.hook_post_inference(ctx)
        assert "**Response Statistics:**" in result["response"]

    def test_hook_header_position(self):
        text = " ".join(["word"] * 50)
        ctx = _make_ctx(
            data={"response": text},
            config={"position": "header", "min_words": 10},
        )
        result = response_stats.hook_post_inference(ctx)
        # Stats should come before the original text
        assert result["response"].index("words") < result["response"].index(text)


# =========================================================================
# MARKDOWN-BEAUTIFIER TESTS
# =========================================================================


class TestMarkdownBeautifierHeaders:
    """Test header normalization."""

    def test_header_spacing_after_hash(self):
        result = markdown_beautifier.fix_header_spacing("##No space")
        assert "## No space" in result

    def test_blank_line_before_header(self):
        result = markdown_beautifier.fix_header_spacing("Some text\n## Header")
        lines = result.split("\n")
        header_idx = next(i for i, l in enumerate(lines) if l.startswith("## Header"))
        assert lines[header_idx - 1].strip() == ""

    def test_blank_line_after_header(self):
        result = markdown_beautifier.fix_header_spacing("## Header\nContent right after")
        lines = result.split("\n")
        header_idx = next(i for i, l in enumerate(lines) if l.startswith("## Header"))
        assert lines[header_idx + 1].strip() == ""


class TestMarkdownBeautifierLists:
    """Test list formatting."""

    def test_normalize_indent(self):
        result = markdown_beautifier.fix_list_formatting("   - Item", "normal")
        assert result.startswith("  - Item")

    def test_strict_marker_normalize(self):
        result = markdown_beautifier.fix_list_formatting("* Item\n+ Other", "strict")
        assert "- Item" in result
        assert "- Other" in result

    def test_numbered_list_preserved(self):
        result = markdown_beautifier.fix_list_formatting("1. First\n2. Second")
        assert "1. First" in result
        assert "2. Second" in result


class TestMarkdownBeautifierTables:
    """Test table alignment."""

    def test_column_alignment(self):
        text = "| A | B |\n|---|---|\n| foo | 1 |\n| barbaz | 2 |"
        result = markdown_beautifier.fix_table_alignment(text)
        lines = result.strip().split("\n")
        # All rows should have same length
        lengths = [len(l) for l in lines]
        assert len(set(lengths)) == 1  # all same length

    def test_non_table_passthrough(self):
        text = "Just some text.\nNo tables here."
        result = markdown_beautifier.fix_table_alignment(text)
        assert result == text


class TestMarkdownBeautifierFences:
    """Test fence repair and code block spacing."""

    def test_close_unclosed_fence(self):
        text = "```python\nsome code"
        result = markdown_beautifier.fix_unclosed_fences(text)
        assert result.rstrip().endswith("```")

    def test_already_closed(self):
        text = "```python\ncode\n```"
        result = markdown_beautifier.fix_unclosed_fences(text)
        # Should not add extra fences
        assert result.count("```") == 2

    def test_code_block_spacing(self):
        text = "Text before\n```python\ncode\n```\nText after"
        result = markdown_beautifier.fix_code_block_spacing(text)
        # Should have blank lines around the code block
        assert "\n\n```python" in result
        assert "```\n\n" in result


class TestMarkdownBeautifierHook:
    """Test the post_inference hook."""

    def test_hook_beautifies(self):
        ctx = _make_ctx(
            data={"response": "##Bad header\nText right under"},
            config={"rules": ["header_spacing"]},
        )
        result = markdown_beautifier.hook_post_inference(ctx)
        assert result is not None
        assert "## Bad header" in result["response"]

    def test_hook_no_change(self):
        ctx = _make_ctx(
            data={"response": "## Good Header\n\nText with space."},
            config={"rules": ["header_spacing"]},
        )
        # Already well-formatted, should return None
        result = markdown_beautifier.hook_post_inference(ctx)
        # May or may not be None depending on trailing whitespace
        if result is not None:
            assert "## Good Header" in result["response"]

    def test_hook_empty_response(self):
        ctx = _make_ctx(data={"response": ""}, config={})
        assert markdown_beautifier.hook_post_inference(ctx) is None


# =========================================================================
# SESSION-SUMMARIZER TESTS
# =========================================================================


class TestSessionSummarizerExtraction:
    """Test extractive summarization."""

    def test_summarize_long_text(self):
        text = (
            "Python is a popular programming language. "
            "It was created by Guido van Rossum. "
            "Python supports multiple paradigms. "
            "The standard library is extensive. "
            "Many data science tools use Python. "
            "Performance can be improved with Cython."
        )
        summary = session_summarizer.extractive_summarize(text, max_sentences=2)
        assert len(summary) > 0
        assert summary.count(".") <= 3  # at most 2 sentences + possible partial

    def test_summarize_short_text(self):
        text = "Short text only."
        summary = session_summarizer.extractive_summarize(text, max_sentences=2)
        # Should return the text as-is (fewer sentences than max)
        assert len(summary) > 0

    def test_summarize_empty(self):
        summary = session_summarizer.extractive_summarize("", max_sentences=2)
        assert summary == ""


class TestSessionSummarizerCommands:
    """Test /summary commands."""

    def setup_method(self):
        """Reset state before each test."""
        session_summarizer.shutdown()

    def test_summary_no_messages(self):
        ctx = _make_ctx(
            data={"user_input": "/summary"},
            config={"interval": 5},
        )
        result = session_summarizer.hook_tool_call(ctx)
        assert result is not None
        assert result["handled"] is True
        assert "No summary available" in result["response"]

    def test_summary_reset(self):
        ctx = _make_ctx(data={"user_input": "/summary reset"}, config={})
        result = session_summarizer.hook_tool_call(ctx)
        assert result["response"] == "Session summary reset."

    def test_non_command_passthrough(self):
        ctx = _make_ctx(data={"user_input": "hello world"}, config={})
        assert session_summarizer.hook_tool_call(ctx) is None

    def test_post_inference_tracking(self):
        session_summarizer.shutdown()
        ctx = _make_ctx(
            data={"prompt": "Hello", "response": "Hi there"},
            config={"interval": 100},  # high interval, no trigger
            metadata={},
        )
        session_summarizer.hook_post_inference(ctx)
        assert session_summarizer._message_count == 1
        assert len(session_summarizer._conversation_buffer) == 2

    def test_background_summary_trigger(self):
        session_summarizer.shutdown()
        # Set interval to 1 so every message triggers
        for i in range(2):
            ctx = _make_ctx(
                data={
                    "prompt": f"Question {i} about Python performance",
                    "response": f"Answer {i} about caching and memoization techniques",
                },
                config={"interval": 1, "max_summary_length": 50},
                metadata={},
            )
            session_summarizer.hook_post_inference(ctx)

        # Wait for background thread
        time.sleep(0.5)
        assert session_summarizer._current_summary != ""


# =========================================================================
# DIFF-TRACKER TESTS
# =========================================================================


class TestDiffTrackerExtraction:
    """Test code block and name extraction."""

    def test_extract_python_block(self):
        text = "Here:\n```python\ndef hello():\n    pass\n```"
        blocks = diff_tracker.extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert "def hello" in blocks[0]["code"]

    def test_extract_multiple_blocks(self):
        text = "```js\nconst x = 1;\n```\nText\n```python\ny = 2\n```"
        blocks = diff_tracker.extract_code_blocks(text)
        assert len(blocks) == 2

    def test_extract_python_names(self):
        code = "def foo():\n    pass\n\nclass Bar:\n    pass\n"
        names = diff_tracker.extract_names(code, "python")
        assert "foo" in names
        assert "Bar" in names

    def test_extract_js_names(self):
        code = "function greet() {}\nconst helper = () => {}\nclass App {}\n"
        names = diff_tracker.extract_names(code, "javascript")
        assert "greet" in names
        assert "helper" in names
        assert "App" in names

    def test_extract_r_names(self):
        code = "my_func <- function(x) {\n  x + 1\n}\n"
        names = diff_tracker.extract_names(code, "r")
        assert "my_func" in names


class TestDiffTrackerSimilarity:
    """Test similarity computation."""

    def test_identical_code(self):
        code = "def hello():\n    return 'world'\n"
        sim = diff_tracker.compute_similarity(code, code)
        assert sim >= 0.99

    def test_similar_code(self):
        a = "def greet(name):\n    print(f'Hello {name}')\n"
        b = "def greet(name, greeting='Hi'):\n    print(f'{greeting} {name}')\n"
        sim = diff_tracker.compute_similarity(a, b)
        assert 0.3 < sim < 0.95

    def test_unrelated_code(self):
        a = "import os\nos.listdir('.')\n"
        b = "class Database:\n    def connect(self):\n        pass\n"
        sim = diff_tracker.compute_similarity(a, b)
        assert sim < 0.4


class TestDiffTrackerDiff:
    """Test diff generation."""

    def test_generate_diff_with_stats(self):
        old = "def hello():\n    pass\n"
        new = "def hello(name):\n    print(name)\n    return True\n"
        diff = diff_tracker.generate_diff(old, new, show_stats=True)
        assert "```diff" in diff
        assert "additions" in diff
        assert "deletions" in diff

    def test_generate_diff_no_stats(self):
        old = "x = 1\n"
        new = "x = 2\n"
        diff = diff_tracker.generate_diff(old, new, show_stats=False)
        assert "```diff" in diff
        assert "additions" not in diff

    def test_identical_no_diff(self):
        code = "x = 1\n"
        diff = diff_tracker.generate_diff(code, code)
        assert diff == ""


class TestDiffTrackerHook:
    """Test the post_inference hook."""

    def setup_method(self):
        diff_tracker._code_history.clear()

    def test_hook_no_code_blocks(self):
        ctx = _make_ctx(data={"response": "Just plain text."}, config={})
        assert diff_tracker.hook_post_inference(ctx) is None

    def test_hook_first_code_block_no_diff(self):
        ctx = _make_ctx(
            data={"response": "```python\ndef foo():\n    pass\n```"},
            config={},
        )
        result = diff_tracker.hook_post_inference(ctx)
        # First block: no history to match against
        assert result is None
        # But it should be added to history
        assert len(diff_tracker._code_history) == 1

    def test_hook_detects_iteration(self):
        # Pre-seed history
        diff_tracker._code_history.append({
            "language": "python",
            "code": "def process(data):\n    return data\n",
            "names": {"process"},
        })
        ctx = _make_ctx(
            data={
                "response": "```python\ndef process(data, validate=True):\n    if validate:\n        check(data)\n    return data\n```"
            },
            config={"similarity_threshold": 0.3},
        )
        result = diff_tracker.hook_post_inference(ctx)
        assert result is not None
        assert result["diffs_found"] >= 1
        assert "```diff" in result["response"]


# =========================================================================
# GITHUB-CONNECTOR TESTS
# =========================================================================


class TestGitHubTokenStore:
    """Test SQLite token storage."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = github_connector.TokenStore(
            Path(self.tmpdir) / "test_auth.db"
        )

    def teardown_method(self):
        self.store.close()

    def test_no_token_initially(self):
        assert self.store.get_token() is None
        assert self.store.get_auth_info() is None

    def test_store_and_retrieve(self):
        self.store.store_token("ghp_abc123", "testuser", "repo,gist")
        assert self.store.get_token() == "ghp_abc123"
        info = self.store.get_auth_info()
        assert info["username"] == "testuser"
        assert info["scopes"] == "repo,gist"

    def test_overwrite_token(self):
        self.store.store_token("ghp_old", "user1", "repo")
        self.store.store_token("ghp_new", "user2", "repo,gist")
        assert self.store.get_token() == "ghp_new"
        assert self.store.get_auth_info()["username"] == "user2"

    def test_revoke_token(self):
        self.store.store_token("ghp_abc", "user", "repo")
        assert self.store.revoke_token() is True
        assert self.store.get_token() is None

    def test_revoke_nonexistent(self):
        assert self.store.revoke_token() is False


class TestGitHubRefDetection:
    """Test GitHub reference pattern detection."""

    def test_owner_repo_issue(self):
        refs = github_connector.detect_github_refs("See torvalds/linux#42")
        assert len(refs) == 1
        assert refs[0]["repo"] == "torvalds/linux"
        assert refs[0]["number"] == "42"

    def test_full_url_issue(self):
        refs = github_connector.detect_github_refs(
            "Link: https://github.com/foo/bar/issues/99"
        )
        assert len(refs) == 1
        assert refs[0]["repo"] == "foo/bar"
        assert refs[0]["number"] == "99"

    def test_full_url_pr(self):
        refs = github_connector.detect_github_refs(
            "PR at https://github.com/org/repo/pull/7"
        )
        assert len(refs) == 1
        assert refs[0]["number"] == "7"

    def test_bare_issue_with_default(self):
        refs = github_connector.detect_github_refs(
            "Fix #123 and #456", default_repo="my/repo"
        )
        assert len(refs) == 2
        assert all(r["repo"] == "my/repo" for r in refs)

    def test_bare_issue_without_default(self):
        refs = github_connector.detect_github_refs("Fix #123")
        # No default repo, bare refs should be ignored
        assert len(refs) == 0

    def test_dedup_same_ref(self):
        refs = github_connector.detect_github_refs(
            "See foo/bar#1 and foo/bar#1 again"
        )
        assert len(refs) == 1

    def test_code_block_exclusion(self):
        refs = github_connector.detect_github_refs(
            "Before\n```\nfoo/bar#42\n```\nAfter"
        )
        assert len(refs) == 0

    def test_inline_code_exclusion(self):
        refs = github_connector.detect_github_refs("Use `owner/repo#10` in code")
        assert len(refs) == 0

    def test_multiple_refs_mixed(self):
        text = (
            "Check torvalds/linux#1 and "
            "https://github.com/python/cpython/issues/99"
        )
        refs = github_connector.detect_github_refs(text)
        assert len(refs) == 2


class TestGitHubCommandRouting:
    """Test /gh command parsing and routing."""

    def _make_gh_ctx(self, user_input, config=None):
        tmpdir = tempfile.mkdtemp()
        ctx = _make_ctx(
            data={"user_input": user_input},
            config=config or {},
            metadata={"plugin_dir": tmpdir},
        )
        return ctx

    def test_non_gh_passthrough(self):
        ctx = self._make_gh_ctx("hello world")
        assert github_connector.hook_tool_call(ctx) is None

    def test_auth_status_no_token(self):
        github_connector._token_store = None
        ctx = self._make_gh_ctx("/gh auth status")
        result = github_connector.hook_tool_call(ctx)
        assert result is not None
        assert "Not authenticated" in result["response"]

    def test_auth_revoke_no_token(self):
        github_connector._token_store = None
        ctx = self._make_gh_ctx("/gh auth revoke")
        result = github_connector.hook_tool_call(ctx)
        assert "No token to revoke" in result["response"]

    def test_auth_invalid_token(self):
        github_connector._token_store = None
        ctx = self._make_gh_ctx("/gh auth abc")
        result = github_connector.hook_tool_call(ctx)
        assert "Invalid token" in result["response"]

    def test_issues_no_token(self):
        github_connector._token_store = None
        ctx = self._make_gh_ctx("/gh issues foo/bar")
        result = github_connector.hook_tool_call(ctx)
        assert "No GitHub token" in result["response"]

    def test_issues_no_repo(self):
        # Pre-store a token
        tmpdir = tempfile.mkdtemp()
        store = github_connector.TokenStore(Path(tmpdir) / "auth.db")
        store.store_token("ghp_test", "user", "repo")
        github_connector._token_store = store
        ctx = self._make_gh_ctx("/gh issues", config={})
        result = github_connector.hook_tool_call(ctx)
        assert "Usage" in result["response"] or "default_repo" in result["response"]
        github_connector._token_store = None
        store.close()

    def test_command_regex_all_commands(self):
        commands = [
            "/gh auth status",
            "/gh auth revoke",
            "/gh auth ghp_testtoken12345",
            "/gh issues torvalds/linux",
            "/gh pr list torvalds/linux",
            "/gh repo info torvalds/linux",
            "/gh search python web framework",
            "/gh gist create my description",
        ]
        for cmd in commands:
            m = github_connector._CMD_RE.match(cmd)
            assert m is not None, f"Failed to parse: {cmd}"


class TestGitHubFootnotes:
    """Test footnote formatting."""

    def test_format_enriched(self):
        enriched = [
            {
                "repo": "foo/bar",
                "number": "42",
                "title": "Fix memory leak",
                "state": "open",
                "ref_type": "Issue",
            },
        ]
        footnotes = github_connector.format_footnotes(enriched)
        assert "foo/bar#42" in footnotes
        assert "Fix memory leak" in footnotes
        assert "[open]" in footnotes

    def test_format_error_ref(self):
        enriched = [
            {"repo": "a/b", "number": "1", "error": "not found"},
        ]
        footnotes = github_connector.format_footnotes(enriched)
        assert "could not fetch" in footnotes

    def test_format_empty(self):
        assert github_connector.format_footnotes([]) == ""


class TestGitHubPostInferenceHook:
    """Test the post_inference auto-link hook."""

    def test_hook_disabled(self):
        ctx = _make_ctx(
            data={"response": "See torvalds/linux#1"},
            config={"auto_link": False},
        )
        assert github_connector.hook_post_inference(ctx) is None

    def test_hook_no_refs(self):
        ctx = _make_ctx(
            data={"response": "No references here."},
            config={"auto_link": True},
        )
        assert github_connector.hook_post_inference(ctx) is None

    def test_hook_no_token(self):
        github_connector._token_store = None
        tmpdir = tempfile.mkdtemp()
        ctx = _make_ctx(
            data={"response": "Check torvalds/linux#1"},
            config={"auto_link": True},
            metadata={"plugin_dir": tmpdir},
        )
        # No token stored, should return None
        assert github_connector.hook_post_inference(ctx) is None


class TestGitHubApiHelper:
    """Test the _github_api helper with mocked urllib."""

    def test_api_network_error(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
            result = github_connector._github_api("GET", "/user", "token")
            assert result["_status"] == 0
            assert "Network error" in result["_error"]

    def test_api_http_error(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            error = urllib.error.HTTPError(
                url="https://api.github.com/user",
                code=401,
                msg="Unauthorized",
                hdrs=MagicMock(get=lambda *a: "?"),
                fp=MagicMock(read=lambda: b'{"message": "Bad credentials"}'),
            )
            mock_urlopen.side_effect = error
            result = github_connector._github_api("GET", "/user", "bad_token")
            assert result["_status"] == 401
            assert "Bad credentials" in result["_error"]

    def test_api_success(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"login": "testuser"}'
        mock_resp.headers = MagicMock()
        mock_resp.headers.get = lambda k, d="?": {"X-RateLimit-Remaining": "59", "X-RateLimit-Limit": "60"}.get(k, d)
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = lambda s, *a: None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = github_connector._github_api("GET", "/user", "ghp_valid")
            assert result["login"] == "testuser"
            assert result["_status"] == 200

    def test_rate_limit_format_low(self):
        msg = github_connector._format_rate_limit(
            {"remaining": "5", "limit": "60"}, show=True
        )
        assert "Warning" in msg

    def test_rate_limit_format_hidden(self):
        msg = github_connector._format_rate_limit(
            {"remaining": "50", "limit": "60"}, show=False
        )
        assert msg == ""
