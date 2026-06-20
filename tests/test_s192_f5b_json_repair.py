#!/usr/bin/env python3
"""
S192 F5b tests -- json_repair (JRP-01, JRP-02, JRP-03).

Functional parity is additionally backed by the pre-existing
tests/test_json_repair.py suite (83 tests, all passing after the edits).
This file pins the three fixes specifically: the fence-pattern rewrite
(parity + timing on the empirically confirmed pathological inputs), the
incremental single-quote pass (parity + timing), and the
RecursionError-to-ValueError contract on adversarial deep nesting.
"""

import importlib.util
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_json_repair():
    name = "oo_s192_json_repair"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "opti_oignon" / "json_repair.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# JRP-01 -- fence pattern: parity + no catastrophic backtracking
# =============================================================================

class TestJrp01FencePattern:
    def test_parity_normal_cases(self):
        jr = _load_json_repair()
        assert jr.strip_markdown_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
        assert jr.strip_markdown_fences("```\n[1, 2]\n```") == "[1, 2]"
        assert jr.strip_markdown_fences('```JSON\n{"a": 1}\n```') == '{"a": 1}'
        # Surrounding prose: the first fenced block is extracted.
        text = 'Sure:\n```json\n{"a": 1}\n```\nDone.'
        assert jr.strip_markdown_fences(text) == '{"a": 1}'
        # Two blocks: first wins (documented behaviour).
        two = '```json\n{"a": 1}\n```\nand\n```json\n{"b": 2}\n```'
        assert jr.strip_markdown_fences(two) == '{"a": 1}'
        # Extra blank lines inside the fence: stripped, same final value.
        assert (
            jr.strip_markdown_fences('```json\n\n\n{"a": 1}\n\n```')
            == '{"a": 1}'
        )
        # No fence: stripped passthrough.
        assert jr.strip_markdown_fences('  {"a": 1}  ') == '{"a": 1}'

    def test_unclosed_fence_fallback_branch(self):
        jr = _load_json_repair()
        # Opening fence, no closing fence: the line-based fallback applies.
        assert jr.strip_markdown_fences('```json\n{"a": 1}') == '{"a": 1}'

    def test_no_catastrophic_backtracking(self):
        jr = _load_json_repair()
        # Pre-fix: both inputs exceeded 3s at 5k chars and effectively hung
        # at 20k (empirically confirmed in S192). Post-fix: linear.
        for payload in (
            "```json" + "\n" * 20000,
            "```json\n" + " " * 20000,
            "```json" + " \n" * 10000,
        ):
            t0 = time.time()
            jr.strip_markdown_fences(payload)
            assert time.time() - t0 < 0.5


# =============================================================================
# JRP-02 -- single-quote pass: parity + linear time
# =============================================================================

class TestJrp02SingleQuotes:
    def test_parity_cases(self):
        jr = _load_json_repair()
        assert jr.fix_single_quotes("{'a': 'b'}") == '{"a": "b"}'
        assert jr.fix_single_quotes("['x', 'y']") == '["x", "y"]'
        # Apostrophe inside a double-quoted string is preserved.
        assert jr.fix_single_quotes('{"k": "it\'s"}') == '{"k": "it\'s"}'
        # Double quote inside a single-quoted string gets escaped.
        assert (
            jr.fix_single_quotes("{'k': 'say \"hi\"'}")
            == '{"k": "say \\"hi\\""}'
        )
        # Already-valid JSON passes through unchanged.
        assert jr.fix_single_quotes('{"a": 1}') == '{"a": 1}'

    def test_quote_dense_input_is_fast(self):
        jr = _load_json_repair()
        # Pre-fix: 1.3s at 20k quotes (join+rstrip per quote, O(n^2)).
        t0 = time.time()
        jr.fix_single_quotes("'" * 20000)
        assert time.time() - t0 < 0.3


# =============================================================================
# JRP-03 -- ValueError contract on adversarial deep nesting
# =============================================================================

class TestJrp03RecursionContract:
    def test_deep_nesting_raises_valueerror(self):
        jr = _load_json_repair()
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(1000)
        try:
            for payload in ("[" * 50000, "{" * 50000):
                try:
                    jr.repair_json(payload)
                except ValueError:
                    pass  # The documented contract.
                except RecursionError:
                    raise AssertionError(
                        "RecursionError escaped repair_json (JRP-03)"
                    )
        finally:
            sys.setrecursionlimit(old_limit)

    def test_adversarial_battery_fail_secure(self):
        jr = _load_json_repair()
        # Everything here must either parse or raise ValueError -- never
        # another exception type.
        cases = [
            '{"a": 1,}',                       # trailing comma
            "{'a': 1}",                        # single quotes
            '// note\n{"a": 1}',               # comment
            '{"a": "trunc',                    # truncated string
            '{"a": {"b": [1, 2',               # truncated nesting
            'Sure! {"a": 1} hope this helps',  # prose-wrapped
            "",                                # empty
            "no json at all",                  # hopeless
            '{"a": ' * 3000,                   # repeated open pattern
        ]
        for text in cases:
            try:
                jr.repair_json(text)
            except ValueError:
                pass

    def test_numbered_list_fallback(self):
        jr = _load_json_repair()
        text = "1. Create file utils.py\n2. Edit main.py\n3. Run tests"
        parsed, steps = jr.repair_json_or_list(text)
        assert parsed is None
        assert steps is not None and len(steps) == 3
        assert steps[0]["step_type"] == "create"
        assert steps[0]["file_path"] == "utils.py"
