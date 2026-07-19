#!/usr/bin/env python3
"""What the context window manager promises about budgets and survival.

Two managers share this module and one obligation: when the history has
to shrink, what survives is decided by explicit rules, not by accident.
These contracts pin those rules at the reference literals so a quiet
retune must surface as a red.

The budget arithmetic. The three zones split the context window at their
ratios and never exceed it; what the history may actually use is the
window minus the generation reserve minus the real system prompt, floored
at zero. A profile whose generation share would squeeze the history below
one fifth of the window is rebalanced: the history floor holds at 0.20
and generation gives the difference back. A known model resolves its
exact profile, a quantised variant resolves through the prefix, and an
unknown model with no client to ask falls back to the default window.

The survival rules. When everything fits, everything is kept and the
strategy says so. When even the newest pairs overflow the budget, only
they survive. In between, the newest pairs are untouchable, the old
messages compete on the weighted importance score -- recency, code,
artifact, length, role at the frozen weights -- and the winners are
re-assembled in chronological order, never in score order. The recent
boundary counts clean user/assistant pairs from the end and degrades one
message at a time when the tail has no clean pairs to give. A summary
message is scored at the frozen 0.95 regardless of its size or position,
and the token estimate is the frozen words-times-1.3 heuristic with a
floor of one token for any non-empty text.

Loaded through the shared isolation window. The module reaches no other
project module; the inference client its unknown-model fallback would
ask is proven absent for that contract, so the default is the default
and not a lucky answer. Nothing real is reached.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_TARGET = "opti_oignon.context_window"

# Reference literals, read off the module in the reference environment and
# frozen here so a change to any of them must update this file.
_DEFAULT_WINDOW = 8192
_HISTORY_FLOOR = 0.20
_SUMMARY_IMPORTANCE = 0.95
_TOKENS_PER_WORD = 1.3
_WEIGHTS = {"recency": 0.40, "code": 0.25, "artifact": 0.15,
            "length": 0.10, "user": 0.10}


def _load(*, ollama_absent=False):
    had = "ollama" in sys.modules
    prev = sys.modules.get("ollama")
    if ollama_absent:
        sys.modules["ollama"] = None

    loaded, win_restore = isolate(targets={_TARGET: source("context_window.py")})

    def restore():
        win_restore()
        if ollama_absent:
            if had:
                sys.modules["ollama"] = prev
            else:
                sys.modules.pop("ollama", None)

    return loaded[_TARGET], restore


def _msg(role, words):
    return {"role": role, "content": " ".join(["word"] * words)}


def _pairs(n, words=20):
    out = []
    for _ in range(n):
        out.append(_msg("user", words))
        out.append(_msg("assistant", words))
    return out


# ---------------------------------------------------------------------------
# v1 -- the three zones split the window and the history floor is zero
# ---------------------------------------------------------------------------

def test_v1_budget_zones_split_the_window_at_their_ratios():
    module, restore = _load()
    try:
        budget = module.TokenBudget(model="m", context_window=8192)
        assert budget.system_budget == int(8192 * 0.10)
        assert budget.history_budget == int(8192 * 0.60)
        assert budget.generation_budget == int(8192 * 0.30)
        assert budget.total_allocated <= budget.context_window

        # The history gets what the generation reserve and the REAL system
        # prompt leave behind, never a negative number.
        assert budget.available_for_history(1000) == 8192 - budget.generation_budget - 1000
        assert budget.available_for_history(10_000) == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# v2 -- a generation-heavy profile is rebalanced onto the history floor
# ---------------------------------------------------------------------------

def test_v2_history_floor_rebalances_a_generation_heavy_profile():
    module, restore = _load()
    try:
        manager = module.TokenBudgetManager(
            custom_profiles={
                "greedy-model": {"context_window": 8192, "generation_ratio": 0.85}
            }
        )
        budget = manager.get_budget("greedy-model")
        assert budget.history_ratio == pytest.approx(_HISTORY_FLOOR)
        assert budget.generation_ratio == pytest.approx(1.0 - _HISTORY_FLOOR - 0.10)
        assert (
            budget.system_ratio + budget.history_ratio + budget.generation_ratio
            == pytest.approx(1.0)
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# v3 -- a known model resolves its exact profile
# ---------------------------------------------------------------------------

def test_v3_known_model_resolves_its_exact_profile():
    module, restore = _load()
    try:
        budget = module.TokenBudgetManager().get_budget("qwen3:32b")
        assert budget.context_window == 32768
        assert budget.generation_ratio == pytest.approx(0.25)
        assert budget.history_ratio == pytest.approx(1.0 - 0.25 - 0.10)
    finally:
        restore()


# ---------------------------------------------------------------------------
# v4 -- a quantised variant resolves through the profile prefix
# ---------------------------------------------------------------------------

def test_v4_variant_model_resolves_through_the_prefix():
    module, restore = _load()
    try:
        manager = module.TokenBudgetManager()
        exact = manager.get_budget("qwen3:32b")
        variant = manager.get_budget("qwen3:32b-q4_0")
        assert variant.context_window == exact.context_window
        assert variant.generation_ratio == pytest.approx(exact.generation_ratio)
    finally:
        restore()


# ---------------------------------------------------------------------------
# v5 -- an unknown model with no client to ask gets the default window
# ---------------------------------------------------------------------------

def test_v5_unknown_model_without_a_client_falls_back_to_the_default():
    module, restore = _load(ollama_absent=True)
    try:
        budget = module.TokenBudgetManager().get_budget("nobody-knows-this:1b")
        assert budget.context_window == _DEFAULT_WINDOW
        assert budget.generation_ratio == pytest.approx(0.30)
        assert budget.history_ratio == pytest.approx(0.60)
    finally:
        restore()


# ---------------------------------------------------------------------------
# v6 -- when everything fits, everything is kept and says so
# ---------------------------------------------------------------------------

def test_v6_history_inside_the_budget_is_kept_whole():
    module, restore = _load()
    try:
        manager = module.SlidingWindowManager()
        messages = _pairs(3, words=10)
        kept, stats = manager.prepare_messages(
            messages, "unused", context_window_override=100_000
        )
        assert stats["strategy"] == "keep_all"
        assert kept == messages
        assert stats["kept"] == len(messages) and stats["dropped"] == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# v7 -- when even the newest pairs overflow, only they survive
# ---------------------------------------------------------------------------

def test_v7_overflowing_recent_pairs_leave_only_the_recent_window():
    module, restore = _load()
    try:
        manager = module.SlidingWindowManager(min_recent_pairs=3)
        messages = _pairs(5, words=20)  # 10 messages, 26 tokens each
        kept, stats = manager.prepare_messages(
            messages, "unused", context_window_override=100
        )
        # available = 100 - 30 (generation) = 70; the three newest pairs
        # alone weigh 6 * 26 = 156 >= 70, so only they survive.
        assert stats["strategy"] == "recent_only"
        assert stats["kept"] == 6 and stats["dropped"] == 4
        assert kept == messages[4:]
    finally:
        restore()


# ---------------------------------------------------------------------------
# v8 -- old messages compete on importance and a code block wins the seat
# ---------------------------------------------------------------------------

def test_v8_code_bearing_old_message_outscores_a_plain_newer_one():
    module, restore = _load()
    try:
        manager = module.SlidingWindowManager(min_recent_pairs=3)
        code_words = " ".join(["word"] * 12)
        old_code = {"role": "user", "content": f"```python\n{code_words}\n```"}
        old_plain = _msg("user", 15)
        messages = [old_code, old_plain] + _pairs(3, words=20)

        # available = 250 - 75 = 175; the six recent messages weigh 156,
        # leaving a 19-token seat. The code message (14 words with its fences,
        # 18 tokens) and the plain one (15 words, 19 tokens) each fit alone
        # but not together, so the seat goes to the higher importance score.
        kept, stats = manager.prepare_messages(
            messages, "unused", context_window_override=250
        )
        assert stats["strategy"] == "sliding_window"
        assert stats["kept_recent"] == 6 and stats["kept_old"] == 1
        assert kept[0] is old_code, (
            "the code-bearing message must win the remaining seat on weight"
        )
        assert old_plain not in kept
    finally:
        restore()


# ---------------------------------------------------------------------------
# v9 -- the recent boundary counts clean pairs and degrades one by one
# ---------------------------------------------------------------------------

def test_v9_recent_boundary_counts_clean_pairs_from_the_end():
    module, restore = _load()
    try:
        manager = module.SlidingWindowManager(min_recent_pairs=3)

        four_pairs = _pairs(4)
        assert manager._identify_recent_boundary(four_pairs) == 2, (
            "three clean pairs from the end of four start at index 2"
        )

        no_pairs = [_msg("user", 5) for _ in range(4)]
        assert manager._identify_recent_boundary(no_pairs) == 0, (
            "with no clean pairs the boundary walks back one message at a time"
        )

        assert manager._identify_recent_boundary([]) == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# v10 -- a summary message scores the frozen importance regardless of size
# ---------------------------------------------------------------------------

def test_v10_summary_message_scores_the_frozen_importance():
    module, restore = _load()
    try:
        manager = module.SlidingWindowManager()
        summary = {
            "role": "system",
            "content": "[Summary of earlier conversation] two words",
        }
        score = manager._score_message(summary, index=0, total=10)
        assert score.is_summary is True
        assert score.importance == pytest.approx(_SUMMARY_IMPORTANCE)

        plain_system = {"role": "system", "content": "You are helpful."}
        other = manager._score_message(plain_system, index=0, total=10)
        assert other.is_summary is False
        assert other.importance < _SUMMARY_IMPORTANCE
    finally:
        restore()


# ---------------------------------------------------------------------------
# v11 -- survivors are re-assembled in chronological order, not score order
# ---------------------------------------------------------------------------

def test_v11_kept_messages_come_back_in_chronological_order():
    module, restore = _load()
    try:
        manager = module.SlidingWindowManager(min_recent_pairs=3)
        old_plain_first = _msg("user", 15)   # 19 tokens, lowest score
        old_code_second = {                  # 18 tokens, highest score
            "role": "user",
            "content": "```python\n" + " ".join(["word"] * 12) + "\n```",
        }
        # 26 tokens, middle score; distinctive words so the absence check
        # below cannot collide with an equal-content recent message.
        old_plain_third = {"role": "user", "content": " ".join(["extra"] * 20)}
        messages = [old_plain_first, old_code_second, old_plain_third] + _pairs(
            3, words=20
        )

        # available = 280 - 84 = 196; recents weigh 156, leaving a 40-token
        # seat row. Greedy order by score is code (18), third (26: no seat
        # left), first (19: fits) -- so the survivors are the FIRST and the
        # SECOND message, picked out of chronological order.
        kept, stats = manager.prepare_messages(
            messages, "unused", context_window_override=280
        )
        assert stats["strategy"] == "sliding_window"
        assert stats["kept_old"] == 2 and stats["dropped"] == 1
        assert old_plain_third not in kept
        assert kept[0] is old_plain_first and kept[1] is old_code_second, (
            "survivors must be re-assembled in their original order, not in "
            "the score order that selected them"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# v12 -- the token estimate is the frozen words-times-1.3 heuristic
# ---------------------------------------------------------------------------

def test_v12_token_estimate_is_the_frozen_heuristic():
    module, restore = _load()
    try:
        estimate = module.SlidingWindowManager._estimate_tokens
        assert estimate("") == 0
        assert estimate("word") == 1, "any non-empty text is at least one token"
        assert estimate("one two three four") == int(4 * _TOKENS_PER_WORD)
        assert estimate(" ".join(["w"] * 100)) == int(100 * _TOKENS_PER_WORD)

        weights = module.SlidingWindowManager
        assert weights.WEIGHT_RECENCY == pytest.approx(_WEIGHTS["recency"])
        assert weights.WEIGHT_CODE == pytest.approx(_WEIGHTS["code"])
        assert weights.WEIGHT_ARTIFACT == pytest.approx(_WEIGHTS["artifact"])
        assert weights.WEIGHT_LENGTH == pytest.approx(_WEIGHTS["length"])
        assert weights.WEIGHT_USER == pytest.approx(_WEIGHTS["user"])
    finally:
        restore()
