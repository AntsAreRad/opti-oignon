"""
S194 F7c -- humanizer fix lot tests.

Covers:
- HUM-01: code segments (fenced + inline) survive the rule pass intact;
  rewrite prompts carry the code-preservation instruction.
- HUM-02: contraction injection preserves newlines / paragraph breaks.
- HUM-03: hedging runs before contractions (multiword hedges with
  "it is" are stripped even when contractions are enabled).
- HUM-04: leading word boundary on banned/hedging patterns; whitespace
  cleanup conditional on a removal and scoped off line starts; start-of-
  text capitalization after hedge removal.
- HUM-05: yaml import guarded.
- Idempotency and the JRP-01-style regex timing battery (adversarial
  inputs stay linear in the container).
"""

import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent


def _load_module(name, rel_path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(_PROJECT / rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hum = _load_module("s194c_humanizer", "opti_oignon/humanizer.py")
# Engine feedback DB must never land in the repo data dir during tests.
hum._DATA_DIR = Path(tempfile.mkdtemp(prefix="s194c_data_"))

_SHIPPED = hum._load_config(_PROJECT / "opti_oignon" / "config" / "humanizer.yaml")
VOCAB = _SHIPPED["vocabulary_replacements"]
BANNED = _SHIPPED["banned_phrases"]
CONTR = _SHIPPED["contractions"]
HEDGE = _SHIPPED["hedging_excess"]

CODE_TEXT = (
    "It's worth noting, the function below is robust.\n\n"
    "```python\n"
    "def leverage(data):\n"
    "    if data is not None:\n"
    "        return data\n"
    "    return None\n"
    "```\n\n"
    "Use `data is not None` to leverage the check inline.\n"
)


def _engine():
    eng = hum.HumanizerEngine(
        config_path=_PROJECT / "opti_oignon" / "config" / "humanizer.yaml"
    )
    return eng


class TestHUM01CodeProtection(unittest.TestCase):
    """HUM-01: code blocks and inline code are untouched by the rules."""

    def test_fenced_block_survives_full_rule_pass(self):
        eng = _engine()
        eng._config.formality = "casual"
        out, strategies, count = eng._apply_rules(CODE_TEXT)
        self.assertIn("def leverage(data):", out)
        self.assertIn("    if data is not None:", out)
        self.assertIn("        return data", out)
        self.assertGreater(count, 0)
        self.assertIn("filler_reduction", strategies)

    def test_inline_code_survives(self):
        eng = _engine()
        eng._config.formality = "casual"
        out, _, _ = eng._apply_rules(CODE_TEXT)
        self.assertIn("`data is not None`", out)

    def test_prose_outside_code_still_transformed(self):
        eng = _engine()
        eng._config.formality = "casual"
        out, _, _ = eng._apply_rules(CODE_TEXT)
        # "robust" -> "strong" in prose, banned opener stripped
        self.assertIn("strong", out.split("```")[0])
        self.assertNotIn("It's worth noting", out)
        # prose "leverage" (outside code spans) replaced
        tail = out.split("`data is not None`")[-1]
        self.assertNotIn("leverage", tail)

    def test_no_sentinel_leakage(self):
        eng = _engine()
        out, _, _ = eng._apply_rules(CODE_TEXT)
        self.assertNotIn("\ue000", out)
        self.assertNotIn("OOSEG", out)

    def test_engine_fallback_path_preserves_code(self):
        # mode "rewrite" without ollama falls back to rules end-to-end
        eng = _engine()
        result = eng.humanize(CODE_TEXT, formality="casual")
        self.assertIn("def leverage(data):", result.humanized)
        self.assertIn("    if data is not None:", result.humanized)

    def test_prompts_carry_code_instruction(self):
        for intensity, template in hum._REWRITE_PROMPTS.items():
            self.assertIn(
                "Keep all code blocks and inline code exactly unchanged",
                template,
                f"missing in {intensity}",
            )

    def test_unterminated_fence_left_unmasked(self):
        masked, segs = hum._protect_code_segments("intro\n```python\nx = 1\n")
        self.assertEqual(segs, [])
        self.assertIn("x = 1", masked)


class TestHUM02StructurePreservation(unittest.TestCase):
    """HUM-02: contractions never flatten newlines or paragraphs."""

    def test_paragraph_breaks_preserved(self):
        text = "First point here. It is fine.\n\nSecond paragraph. Do not worry.\n- item one\n- item two"
        out, count = hum._apply_contractions(text, CONTR, "casual")
        self.assertIn("\n\n", out)
        self.assertIn("\n- item one\n- item two", out)
        self.assertIn("It's fine.", out)
        self.assertIn("Don't worry.", out)
        self.assertGreaterEqual(count, 2)

    def test_neutral_skips_first_sentence_only(self):
        text = "It is the intro. It is the body."
        out, _ = hum._apply_contractions(text, CONTR, "neutral")
        self.assertTrue(out.startswith("It is the intro."))
        self.assertIn("It's the body.", out)

    def test_exact_separator_roundtrip_without_matches(self):
        text = "Alpha.\n\n\nBeta!   Gamma?\tDelta."
        out, count = hum._apply_contractions(text, {"zz qq": "z'q"}, "casual")
        self.assertEqual(out, text)
        self.assertEqual(count, 0)


class TestHUM03StrategyOrder(unittest.TestCase):
    """HUM-03: hedging strips before contractions can defeat it."""

    def test_hedge_with_it_is_stripped_under_casual(self):
        eng = _engine()
        eng._config.formality = "casual"
        text = "Some intro sentence here. It is possible that the cache is stale."
        out, strategies, _ = eng._apply_rules(text)
        self.assertNotIn("possible that", out)
        self.assertIn("hedging_calibration", strategies)

    def test_order_in_source(self):
        src = (_PROJECT / "opti_oignon" / "humanizer.py").read_text(encoding="utf-8")
        rules = src.split("def _apply_rules")[1].split("def ")[0]
        self.assertLess(
            rules.index("hedging_calibration"),
            rules.index("contraction_injection"),
        )


class TestHUM04CleanupCorrectness(unittest.TestCase):
    """HUM-04: boundaries, scoped collapse, start capitalization."""

    def test_no_substring_phrase_match(self):
        text = "We denote that x holds."
        out, count = hum._strip_banned_phrases(text, ["note that"])
        self.assertEqual(count, 0)
        self.assertEqual(out, text)

    def test_no_substring_hedge_match(self):
        text = "maybe later"
        out, count = hum._reduce_hedging(text, ["may"])
        self.assertEqual(count, 0)
        self.assertEqual(out, text)

    def test_cleanup_only_on_removal(self):
        text = "columns  aligned  here"
        out, count = hum._strip_banned_phrases(text, ["absent phrase"])
        self.assertEqual(count, 0)
        self.assertEqual(out, text)

    def test_line_leading_spaces_preserved(self):
        text = "It's worth noting this list:\n  - nested  item\n  - other"
        out, count = hum._strip_banned_phrases(text, ["It's worth noting"])
        self.assertEqual(count, 1)
        self.assertIn("\n  - nested item\n  - other", out)

    def test_start_of_text_capitalized_after_hedge(self):
        text = "It is possible that the answer is 42."
        out, count = hum._reduce_hedging(text, ["It is possible that"])
        self.assertEqual(count, 1)
        self.assertTrue(out.startswith("The answer is 42."))


class TestHUM05YamlGuard(unittest.TestCase):
    """HUM-05: yaml import guarded."""

    def test_guard_in_source(self):
        src = (_PROJECT / "opti_oignon" / "humanizer.py").read_text(encoding="utf-8")
        self.assertIn("YAML_AVAILABLE = True", src)
        self.assertIn("except ImportError", src)
        self.assertIn("if not YAML_AVAILABLE or not config_path.exists()", src)
        self.assertTrue(hasattr(hum, "YAML_AVAILABLE"))


class TestIdempotencyAndTiming(unittest.TestCase):
    """Idempotency on prose; JRP-01-style adversarial timing battery."""

    def _rules(self, text):
        eng = _engine()
        eng._config.formality = "casual"
        out, _, _ = eng._apply_rules(text)
        return out

    def test_rules_idempotent_on_prose(self):
        prose = (
            "Furthermore, we will leverage a robust methodology. "
            "It is possible that we are correct. It's worth noting the rest."
        )
        once = self._rules(prose)
        twice = self._rules(once)
        self.assertEqual(once, twice)

    def test_rules_idempotent_with_code(self):
        once = self._rules(CODE_TEXT)
        twice = self._rules(once)
        self.assertEqual(once, twice)

    def test_adversarial_timing_linear(self):
        cases = {
            "spaces": " " * 50000 + "end",
            "phrase_spam": "It's worth noting " * 2000,
            "punct_runs": "a." * 20000,
        }
        eng = _engine()
        eng._config.formality = "casual"
        for name, payload in cases.items():
            start = time.perf_counter()
            eng._apply_rules(payload)
            elapsed = time.perf_counter() - start
            self.assertLess(
                elapsed, 2.0, f"{name} took {elapsed:.2f}s (expected linear)"
            )


if __name__ == "__main__":
    unittest.main()
