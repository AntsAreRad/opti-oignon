#!/usr/bin/env python3
"""What the calibrated token estimator promises about its arithmetic.

Every budget in the tree rests on this module-level estimator: the
family table turns a model name into a chars-per-token ratio, the code
detector applies a language multiplier on top, and the batch helper
resolves the ratio once for a whole message list. A silent change to any
of these constants shifts every zone budget downstream, so the figures
are pinned at the digit.

Families are matched by ordered prefixes so that the specific names win
over the generic ones they contain (the code variants before their base
families), only the part before the colon is consulted, and anything
unrecognized lands on the neutral default ratio. The reference figures
for a 450-character plain text are pinned per family and for the
default. Code content multiplies the base estimate by a per-language
factor, with a generic factor when indicators fire without a language
match, and the dominant language wins the scoring. The single-text path
floors at one token for any non-empty text; the batch path deliberately
does not -- it truncates per message, so a tiny non-empty message can
contribute zero. That asymmetry is pinned as it stands, not judged.

Loaded through the shared isolation window from the module source; the
functions under contract are pure, so no stand-ins are needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.context_manager"

_PLAIN_450 = "a" * 450
_PY_TEXT = "def f():\n" + "    import os\n" + "a" * 430  # 453 chars, python markers
_FENCE_450 = "```\n" + "z" * 446  # indicators fire, no language pattern matches
_MIXED = "def a\nimport b\nself.x\nconst y"  # 29 chars, python outweighs javascript


def _load():
    """Load the real module alone on the shared window."""
    loaded, restore = isolate(targets={_TARGET: source("context_manager.py")})
    return loaded[_TARGET], restore


# ---------------------------------------------------------------------------
# n1 -- family detection: ordered prefixes, colon split, unknown fallback
# ---------------------------------------------------------------------------

def test_n1_family_detection_orders_specific_before_generic():
    module, restore = _load()
    try:
        assert module.detect_model_family("codellama:13b") == "codellama"
        assert module.detect_model_family("codegemma:7b") == "codegemma"
        assert module.detect_model_family("codestral:22b") == "codestral"
        assert module.detect_model_family("starcoder2:15b") == "starcoder"
        assert module.detect_model_family("llama3.3:latest") == "llama"
        # Only the part before the colon is consulted: a family name in the
        # tag does not leak into detection.
        assert module.detect_model_family("mymodel:qwen-flavored") == "unknown"
        assert module.detect_model_family("") == "unknown"
        assert module.detect_model_family("totally-new:7b") == "unknown"
    finally:
        restore()


# ---------------------------------------------------------------------------
# n2 -- per-family ratios and the neutral default
# ---------------------------------------------------------------------------

def test_n2_family_ratios_and_default_are_the_calibrated_constants():
    module, restore = _load()
    try:
        assert module.get_family_chars_per_token("qwen3:32b") == 3.2
        assert module.get_family_chars_per_token("llama3.3:latest") == 3.5
        assert module.get_family_chars_per_token("mistral-small3.2:latest") == 3.8
        assert module.get_family_chars_per_token("codellama:13b") == 3.0
        assert module.get_family_chars_per_token("zz-nope:1b") == 3.7
    finally:
        restore()


# ---------------------------------------------------------------------------
# n3 -- reference figures for 450 plain characters
# ---------------------------------------------------------------------------

def test_n3_reference_figures_for_450_plain_characters():
    module, restore = _load()
    try:
        assert module.estimate_tokens_calibrated(_PLAIN_450, "qwen3:32b") == 140
        assert module.estimate_tokens_calibrated(_PLAIN_450, "llama3.3:latest") == 128
        assert module.estimate_tokens_calibrated(_PLAIN_450) == 121
        assert module.estimate_tokens_calibrated(_PLAIN_450, "zzz-model:1b") == 121
    finally:
        restore()


# ---------------------------------------------------------------------------
# n4 -- an explicit ratio override beats family detection
# ---------------------------------------------------------------------------

def test_n4_explicit_override_beats_family_detection():
    module, restore = _load()
    try:
        got = module.estimate_tokens_calibrated(
            _PLAIN_450, "qwen3:32b", chars_per_token_override=2.0
        )
        assert got == 225
    finally:
        restore()


# ---------------------------------------------------------------------------
# n5 -- empty text is zero; any non-empty text floors at one
# ---------------------------------------------------------------------------

def test_n5_empty_is_zero_and_nonempty_floors_at_one():
    module, restore = _load()
    try:
        assert module.estimate_tokens_calibrated("") == 0
        assert module.estimate_tokens_calibrated("a") == 1
    finally:
        restore()


# ---------------------------------------------------------------------------
# n6 -- python content applies its language multiplier on the family base
# ---------------------------------------------------------------------------

def test_n6_python_content_applies_its_language_multiplier():
    module, restore = _load()
    try:
        assert module._detect_code_language(_PY_TEXT) == "python"
        assert module.estimate_tokens_calibrated(_PY_TEXT, "qwen3:32b") == 158
    finally:
        restore()


# ---------------------------------------------------------------------------
# n7 -- indicators without a language match use the generic multiplier
# ---------------------------------------------------------------------------

def test_n7_indicators_without_language_use_the_generic_multiplier():
    module, restore = _load()
    try:
        assert module._has_code_content(_FENCE_450) is True
        assert module._detect_code_language(_FENCE_450) == "default"
        assert module.estimate_tokens_calibrated(_FENCE_450, "qwen3:32b") == 161
    finally:
        restore()


# ---------------------------------------------------------------------------
# n8 -- the dominant language wins the scoring
# ---------------------------------------------------------------------------

def test_n8_dominant_language_wins_the_scoring():
    module, restore = _load()
    try:
        assert module._detect_code_language(_MIXED) == "python"
        assert module.estimate_tokens_calibrated(_MIXED) == 8
    finally:
        restore()


# ---------------------------------------------------------------------------
# n9 -- batch truncation: a tiny non-empty message can contribute zero
# ---------------------------------------------------------------------------

def test_n9_batch_truncates_where_the_single_path_floors():
    module, restore = _load()
    try:
        assert module.estimate_tokens_calibrated("abc") == 1
        assert module.estimate_messages_tokens([{"role": "u", "content": "abc"}]) == 0
        assert module.estimate_messages_tokens([]) == 0
        assert module.estimate_messages_tokens([{"role": "u", "content": ""}]) == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# n10 -- the batch path matches the single path on real-sized content
# ---------------------------------------------------------------------------

def test_n10_batch_matches_single_path_on_real_sized_content():
    module, restore = _load()
    try:
        plain = [{"role": "u", "content": _PLAIN_450}]
        assert module.estimate_messages_tokens(plain, "qwen3:32b") == 140
        code = [{"role": "u", "content": _PY_TEXT}]
        assert module.estimate_messages_tokens(code, "qwen3:32b") == 158
    finally:
        restore()


# ---------------------------------------------------------------------------
# n11 -- an unknown family in the batch path uses the neutral default
# ---------------------------------------------------------------------------

def test_n11_batch_unknown_family_uses_the_neutral_default():
    module, restore = _load()
    try:
        msgs = [{"role": "u", "content": _PLAIN_450}]
        assert module.estimate_messages_tokens(msgs, "zzz-model:1b") == 121
    finally:
        restore()


# ---------------------------------------------------------------------------
# n12 -- detection is case-insensitive
# ---------------------------------------------------------------------------

def test_n12_family_detection_is_case_insensitive():
    module, restore = _load()
    try:
        assert module.detect_model_family("QWEN3:32B") == "qwen"
        assert module.get_family_chars_per_token("LLAMA3.3:LATEST") == 3.5
    finally:
        restore()
