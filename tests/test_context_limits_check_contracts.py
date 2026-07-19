#!/usr/bin/env python3
"""What the context manager promises about limits, checks and truncation.

The manager resolves a model's window through a fixed ladder -- the
user's config first, then the verified table, then a live model query,
then a prefix-similar table entry, then the universal fallback -- and
caches what it resolved. The validation arithmetic and the truncation
shapes built on top are what the rest of the tree trusts, so both are
pinned here at the digit.

The live query goes through a subprocess seam; these contracts replace
that seam with a scripted stand-in so the ladder is deterministic on any
machine, with or without a local model runtime, and so the live branch
itself can be exercised on demand. The output parser is pinned directly
on a crafted transcript. The check arithmetic pins the available-input
formula, the suggested-removal margin and the warning tiers with their
styling classes. Truncation pins the pass-through, the middle-cut shape
with its exact marker, and the degenerate head-cut whose removal figure
is measured on the kept text alone, marker excluded. The module-level
singleton and its reset are pinned, and the config loader is proven to
degrade to defaults when its import is unreachable.

Loaded through the shared isolation window from the module source.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.context_manager"


class _ScriptedRun:
    """Subprocess stand-in: counts calls, replays one scripted result."""

    def __init__(self, returncode=1, stdout="", stderr="unavailable"):
        self.calls = 0
        self._result = types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        )

    def run(self, *args, **kwargs):
        self.calls += 1
        return self._result


def _load(*, returncode=1, stdout=""):
    """Load the real module and replace its subprocess seam."""
    loaded, win_restore = isolate(targets={_TARGET: source("context_manager.py")})
    module = loaded[_TARGET]
    seam = _ScriptedRun(returncode=returncode, stdout=stdout)
    module.subprocess = types.SimpleNamespace(
        run=seam.run, TimeoutExpired=Exception
    )
    return module, seam, win_restore


# ---------------------------------------------------------------------------
# k1 -- the user's config wins, and a malformed entry is skipped
# ---------------------------------------------------------------------------

def test_k1_config_entry_wins_and_malformed_entry_is_skipped():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager(models_config={
            "cfg-model": {
                "context_window": 1000, "max_output": 200,
                "recommended_input": 700, "chars_per_token": 2.0,
                "display_name": "CfgModel",
            },
            "half-entry": {"note": "no window key"},
        })
        limits = manager.get_model_limits("cfg-model")
        assert limits.source == "config"
        assert (limits.context_window, limits.max_output) == (1000, 200)
        assert limits.safe_input == 800
        # An entry without a window key falls through the ladder instead of
        # producing a half-formed limit.
        fallback = manager.get_model_limits("half-entry")
        assert fallback.source == "default"
    finally:
        restore()


# ---------------------------------------------------------------------------
# k2 -- the verified table answers for known models
# ---------------------------------------------------------------------------

def test_k2_verified_table_answers_for_known_models():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()
        limits = manager.get_model_limits("qwen3-coder:30b")
        assert limits.source == "verified"
        assert (limits.context_window, limits.max_output) == (262144, 8192)
        assert limits.safe_input == 262144 - 8192
        assert seam.calls == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# k3 -- a prefix-similar name resolves to the verified entry itself
# ---------------------------------------------------------------------------

def test_k3_prefix_similar_name_resolves_to_the_verified_entry():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()
        exact = manager.get_model_limits("qwen3-coder:30b")
        similar = manager.get_model_limits("qwen3-coder:30b-zz-probe")
        assert similar is exact
        assert seam.calls == 1  # the live query was tried once and said no
    finally:
        restore()


# ---------------------------------------------------------------------------
# k4 -- a fully unknown model lands on the universal fallback
# ---------------------------------------------------------------------------

def test_k4_unknown_model_lands_on_the_universal_fallback():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()
        limits = manager.get_model_limits("zz-nope:1b")
        assert limits is module.DEFAULT_MODEL_LIMITS["_default"]
        assert (limits.context_window, limits.max_output) == (8192, 4096)
        assert limits.source == "default"
    finally:
        restore()


# ---------------------------------------------------------------------------
# k5 -- the transcript parser extracts the four fields
# ---------------------------------------------------------------------------

def test_k5_transcript_parser_extracts_the_four_fields():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()
        transcript = (
            "  architecture        qwen2\n"
            "  parameters          7.6B\n"
            "  context length      32768\n"
            "  quantization        Q4_K_M\n"
        )
        info = manager._parse_ollama_output(transcript)
        assert info["context_length"] == 32768
        assert info["parameters"] == 7600000000
        assert info["architecture"] == "qwen2"
        assert info["quantization"] == "Q4_K_M"
    finally:
        restore()


# ---------------------------------------------------------------------------
# k6 -- the live branch derives output and recommended sizes
# ---------------------------------------------------------------------------

def test_k6_live_branch_derives_output_and_recommended_sizes():
    module, seam, restore = _load(
        returncode=0, stdout="  context length      40960\n"
    )
    try:
        manager = module.ContextManager()
        limits = manager.get_model_limits("brandnew-model:9b")
        assert limits.source == "ollama"
        assert limits.context_window == 40960
        assert limits.max_output == 10240  # a quarter, capped at 32768
        assert limits.recommended_input == 40960 - 10240 - 1000
        assert limits.chars_per_token == 3.5
    finally:
        restore()


# ---------------------------------------------------------------------------
# k7 -- resolution is cached; clearing the cache resolves again
# ---------------------------------------------------------------------------

def test_k7_resolution_is_cached_until_cleared():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()
        first = manager.get_model_limits("zz-nope:1b")
        assert seam.calls == 1
        second = manager.get_model_limits("zz-nope:1b")
        assert second is first
        assert seam.calls == 1
        manager.clear_cache()
        manager.get_model_limits("zz-nope:1b")
        assert seam.calls == 2
    finally:
        restore()


# ---------------------------------------------------------------------------
# k8 -- the check arithmetic: available input and suggested removal
# ---------------------------------------------------------------------------

def test_k8_check_arithmetic_available_input_and_suggested_removal():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()
        check = manager.check_context(
            prompt="p" * 400, document="d" * 40000,
            system_prompt="s" * 400, model="zz-nope:1b",
        )
        assert check.total_tokens == 10200
        assert check.available_for_input == 4096
        assert check.exceeds_limit is True
        assert check.truncation_needed is True
        assert check.suggested_truncation == 10200 - 4096 + 500
        assert check.is_safe is False
        assert check.status_class == "danger"
    finally:
        restore()


# ---------------------------------------------------------------------------
# k9 -- the warning tiers and their styling classes
# ---------------------------------------------------------------------------

def test_k9_warning_tiers_and_styling_classes():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()

        over = manager.check_context(prompt="p" * 400, document="d" * 40000,
                                     system_prompt="s" * 400, model="zz-nope:1b")
        assert over.warning_message.startswith("Context exceeds limit!")

        crowded = manager.check_context(prompt="p" * 400, document="d" * 15400,
                                        system_prompt="s" * 400, model="zz-nope:1b")
        assert crowded.exceeds_limit is False
        assert crowded.exceeds_recommended is True
        assert crowded.warning_message.startswith("Context exceeds recommended size")
        assert crowded.status_class == "warning"

        busy = manager.check_context(prompt="p" * 400, document="d" * 15000,
                                     system_prompt="s" * 400, model="zz-nope:1b")
        assert busy.exceeds_recommended is False
        assert busy.usage_percent > 75
        assert "Consider summarizing" in busy.warning_message
        assert busy.status_class == "warning"

        light = manager.check_context(prompt="p" * 400, document="d" * 4000,
                                      system_prompt="", model="zz-nope:1b")
        assert light.warning_message is None
        assert light.status_class == "safe"
    finally:
        restore()


# ---------------------------------------------------------------------------
# k10 -- truncation passes text through when it already fits
# ---------------------------------------------------------------------------

def test_k10_truncation_passes_through_when_text_fits():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()
        text, removed = manager.smart_truncate("hello world", 100, model="zz-nope:1b")
        assert text == "hello world"
        assert removed == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# k11 -- the middle cut keeps both ends and carries the exact marker
# ---------------------------------------------------------------------------

def test_k11_middle_cut_keeps_both_ends_with_the_exact_marker():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()
        text = "".join(f"L{i:05d} " for i in range(5000))
        out, removed = manager.smart_truncate(
            text, max_tokens=1000, model="zz-nope:1b",
            preserve_start=100, preserve_end=50,
        )
        assert out.startswith(text[:400])   # 100 tokens at 4 chars each
        assert out.endswith(text[-200:])    # 50 tokens at 4 chars each
        marker = "\n\n[... content truncated to fit context window ...]\n\n"
        assert marker in out
        recount = manager.estimate_tokens(out, "zz-nope:1b")
        assert removed == manager.estimate_tokens(text, "zz-nope:1b") - recount
    finally:
        restore()


# ---------------------------------------------------------------------------
# k12 -- the degenerate head cut measures removal on the kept text alone
# ---------------------------------------------------------------------------

def test_k12_degenerate_head_cut_measures_removal_on_kept_text():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager()
        text = "ab" * 300  # 600 chars; preserve windows exceed the text
        out, removed = manager.smart_truncate(
            text, max_tokens=50, model="zz-nope:1b",
            preserve_start=500, preserve_end=200,
        )
        assert out.endswith("\n\n[... truncated ...]")
        assert out.startswith(text[:200])   # 50 tokens at 4 chars each
        assert removed == 100               # 150 estimated, 50 kept; marker excluded
    finally:
        restore()


# ---------------------------------------------------------------------------
# k13 -- one module-level instance, reset makes a fresh one, facades share it
# ---------------------------------------------------------------------------

def test_k13_module_singleton_reset_and_facade_delegation():
    module, seam, restore = _load()
    try:
        first = module.get_context_manager()
        assert module.get_context_manager() is first
        module.reset_context_manager()
        fresh = module.get_context_manager()
        assert fresh is not first
        via_facade = module.check_context(prompt="p" * 400, model="zz-nope:1b")
        direct = fresh.check_context(prompt="p" * 400, model="zz-nope:1b")
        assert via_facade.total_tokens == direct.total_tokens
        assert via_facade.available_for_input == direct.available_for_input
    finally:
        restore()


# ---------------------------------------------------------------------------
# k14 -- the estimation method's three paths, and config degradation
# ---------------------------------------------------------------------------

def test_k14_estimation_method_paths_and_config_degradation():
    module, seam, restore = _load()
    try:
        manager = module.ContextManager(models_config={
            "mystery-model": {
                "context_window": 4096, "max_output": 512,
                "recommended_input": 3000, "chars_per_token": 2.0,
            },
        })
        # A known family goes through the calibrated path.
        assert manager.estimate_tokens("a" * 450, "qwen3:32b") == 140
        # An unknown family uses the resolved limits' own ratio.
        assert manager.estimate_tokens("a" * 450, "mystery-model") == 225
        # No model at all uses the plain four-chars ratio.
        assert manager.estimate_tokens("a" * 450) == 112
        # The config-backed constructor degrades to defaults when its import
        # is unreachable inside the window.
        degraded = module.ContextManager.from_config()
        assert degraded._config_limits == {}
    finally:
        restore()
