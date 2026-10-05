#!/usr/bin/env python3
"""What the summary layer promises about where its settings come from.

The live summarizer kept its model, fallback chain, sampling temperature,
output cap, timeout, input cap and message threshold as literals in its
class, and the tier layer kept its segment budget and verbatim tail the same
way: tunables a user could not reach. They live in ``compression.yaml`` now,
beside the compressor's own, under ``live_summary`` and ``summary_tiers``.
Each value is checked when it is read, and a value that cannot be right is
refused by its full name. A summarizer whose settings were refused asks no
model at all: it does not guess a value in place of the one it could not
trust. The shipped file carries the values the literals carried, so moving
them changed nothing a user had.

A recording engine stands in for the model; every assertion reads what the
summarizer or the tier manager did with the file it was given.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_SUMMARY = "opti_oignon.context_summary"
_TIERS = "opti_oignon.context_summary_tiers"

_LIVE = {
    "model": "yaml-model:2b",
    "fallback_models": ["yaml-model:2b", "yaml-fallback:1b"],
    "temperature": 0.55,
    "max_summary_tokens": 123,
    "timeout_s": 7,
    "max_input_tokens": 999,
    "min_messages": 6,
}
_TIER = {"segment_budget_tokens": 20, "tail_keep_messages": 2, "compose_budget_tokens": 90, "compose_share": 0.25}


def _write(tmp_path, live=None, tier=None):
    import yaml

    path = tmp_path / "compression.yaml"
    path.write_text(
        yaml.safe_dump(
            {"live_summary": dict(_LIVE, **(live or {})), "summary_tiers": dict(_TIER, **(tier or {}))}
        ),
        encoding="utf-8",
    )
    return path


class _Backend:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content="A faithful summary of the turns above.")


def _summarizer(path):
    loaded, restore = isolate(targets={_SUMMARY: source("context_summary.py")})
    mod = loaded[_SUMMARY]
    try:
        summarizer = mod.ContextSummarizer(settings_path=path)
    except Exception:
        restore()
        raise
    backend = _Backend()
    summarizer._resolve_backend = lambda model: backend
    summarizer._installed_model_names = lambda: {"yaml-model:2b", "yaml-fallback:1b", "qwen3:8b"}
    return mod, summarizer, backend, restore


def _turns(n=6):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} about onions"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# ss1 -- the model call takes its model, temperature and cap from the file
# ---------------------------------------------------------------------------

def test_ss1_the_live_summarizer_asks_the_file_s_model_with_the_file_s_sampling(tmp_path):
    mod, summarizer, backend, restore = _summarizer(_write(tmp_path))
    try:
        assert summarizer.summarize_messages(_turns()) is not None
        call = backend.calls[-1]
        assert call["model"] == "yaml-model:2b"
        assert call["options"]["temperature"] == 0.55
        assert call["options"]["num_predict"] == 123
    finally:
        restore()


# ---------------------------------------------------------------------------
# ss2 -- the thresholds come from the file
# ---------------------------------------------------------------------------

def test_ss2_the_live_summarizer_s_thresholds_come_from_the_file(tmp_path):
    mod, summarizer, backend, restore = _summarizer(_write(tmp_path))
    try:
        assert summarizer.SUMMARY_THRESHOLD == 6
        assert summarizer.MAX_INPUT_TOKENS == 999
        assert summarizer.SUMMARY_TIMEOUT == 7
        assert summarizer.FALLBACK_MODELS == ["yaml-model:2b", "yaml-fallback:1b"]
    finally:
        restore()


# ---------------------------------------------------------------------------
# ss3 -- the tier manager's budgets come from the file
# ---------------------------------------------------------------------------

def test_ss3_the_tier_manager_freezes_and_composes_by_the_file_s_budgets(tmp_path):
    path = _write(tmp_path)
    loaded, restore = isolate(targets={_TIERS: source("context_summary_tiers.py")})
    try:
        tiers = loaded[_TIERS]
        archive = [
            {"id": i + 1, "role": "user" if i % 2 == 0 else "assistant", "content": "w " * 40}
            for i in range(12)
        ]
        manager = tiers.TierManager(lambda cid: archive, settings_path=path, estimate=lambda t: len(t.split()))
        update = manager.advance("conv-1", {}, summarize_fn=lambda msgs: "s " * 40)
        state = tiers.TierState.from_metadata(update)
        assert [(s.first_id, s.last_id) for s in state.segments] == [(i, i) for i in range(1, 11)], (
            "a 20-token budget freezes each 40-word turn alone, and the last two stay verbatim"
        )
        composed = manager.compose("conv-1", update)
        assert len(composed.split()) <= 90 + 20, "the composition keeps to the file's budget"
    finally:
        restore()


# ---------------------------------------------------------------------------
# ss4 -- a value that cannot be right is refused by its full name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "live, tier, name",
    [
        ({"temperature": "warm"}, None, "live_summary.temperature"),
        ({"max_summary_tokens": 0}, None, "live_summary.max_summary_tokens"),
        ({"model": ""}, None, "live_summary.model"),
        ({"fallback_models": "qwen3:8b"}, None, "live_summary.fallback_models"),
        (None, {"segment_budget_tokens": -1}, "summary_tiers.segment_budget_tokens"),
        (None, {"compose_budget_tokens": True}, "summary_tiers.compose_budget_tokens"),
        (None, {"compose_share": 1.5}, "summary_tiers.compose_share"),
    ],
)
def test_ss4_a_value_that_cannot_be_right_is_refused_by_its_full_name(tmp_path, live, tier, name):
    path = _write(tmp_path, live=live, tier=tier)
    loaded, restore = isolate(
        targets={_SUMMARY: source("context_summary.py"), _TIERS: source("context_summary_tiers.py")}
    )
    try:
        loader = (
            loaded[_SUMMARY].load_settings if name.startswith("live_summary")
            else loaded[_TIERS].load_tier_settings
        )
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            loader(path)
    finally:
        restore()


# ---------------------------------------------------------------------------
# ss5 -- a summarizer whose settings were refused asks no model
# ---------------------------------------------------------------------------

def test_ss5_a_summarizer_over_refused_settings_asks_no_model_and_says_why(tmp_path):
    mod, summarizer, backend, restore = _summarizer(_write(tmp_path, live={"temperature": "warm"}))
    try:
        assert summarizer.summarize_messages(_turns(), model="yaml-model:2b") is None
        assert backend.calls == []
        assert "live_summary.temperature" in (summarizer.settings_error or "")
    finally:
        restore()


# ---------------------------------------------------------------------------
# ss6 -- the shipped file carries the values the literals carried
# ---------------------------------------------------------------------------

def test_ss6_the_shipped_file_carries_the_former_literals():
    loaded, restore = isolate(
        targets={_SUMMARY: source("context_summary.py"), _TIERS: source("context_summary_tiers.py")}
    )
    try:
        live = loaded[_SUMMARY].load_settings()
        assert (live.model, live.temperature, live.max_summary_tokens, live.timeout_s,
                live.max_input_tokens, live.min_messages) == ("qwen3:8b", 0.3, 400, 15, 4000, 4)
        assert live.fallback_models == ("qwen3:8b", "nemotron-3-nano:8b", "qwen3:4b", "qwen3:1.7b")
        tier = loaded[_TIERS].load_tier_settings()
        assert (tier.segment_budget_tokens, tier.tail_keep_messages) == (1200, 4)
        assert tier.compose_budget_tokens >= tier.segment_budget_tokens // 4
    finally:
        restore()


# ---------------------------------------------------------------------------
# ss7, ss8 -- a file saved in another encoding is refused by name, never raised
# ---------------------------------------------------------------------------

def _undecodable(tmp_path):
    """The shipped settings, with a comment saved as Latin-1: valid YAML once decoded, not UTF-8."""
    import yaml

    body = yaml.safe_dump({"live_summary": _LIVE, "summary_tiers": _TIER}).encode("ascii")
    path = tmp_path / "compression.yaml"
    path.write_bytes(b"# r" + bytes([0xE9]) + b"glages\n" + body)
    return path


def test_ss7_a_summarizer_over_a_file_it_cannot_decode_is_unavailable_and_names_the_section(tmp_path):
    path = _undecodable(tmp_path)
    loaded, restore = isolate(targets={_SUMMARY: source("context_summary.py")})
    try:
        mod = loaded[_SUMMARY]
        with pytest.raises(mod.SummarySettingsError, match="live_summary"):
            mod.load_settings(path)
        summarizer = mod.ContextSummarizer(settings_path=path)
        assert summarizer.available is False
        assert "live_summary" in (summarizer.settings_error or "")
    finally:
        restore()


def test_ss8_the_tier_settings_over_a_file_they_cannot_decode_are_refused_by_name(tmp_path):
    path = _undecodable(tmp_path)
    loaded, restore = isolate(targets={_TIERS: source("context_summary_tiers.py")})
    try:
        mod = loaded[_TIERS]
        with pytest.raises(mod.TierSettingsError, match="summary_tiers"):
            mod.load_tier_settings(path)
    finally:
        restore()


# ---------------------------------------------------------------------------
# ss9, ss10 -- a file the YAML reader cannot turn into values is refused too
# ---------------------------------------------------------------------------

def _unconstructible(tmp_path):
    """Valid UTF-8 and valid YAML syntax, but a date the reader cannot build."""
    import yaml

    body = yaml.safe_dump({"live_summary": _LIVE, "summary_tiers": _TIER})
    path = tmp_path / "compression.yaml"
    path.write_text("note: 2026-02-30\n" + body, encoding="utf-8")
    return path


def test_ss9_a_summarizer_over_a_file_whose_values_cannot_be_built_is_unavailable_and_says_why(tmp_path):
    path = _unconstructible(tmp_path)
    loaded, restore = isolate(targets={_SUMMARY: source("context_summary.py")})
    try:
        mod = loaded[_SUMMARY]
        with pytest.raises(mod.SummarySettingsError, match="live_summary"):
            mod.load_settings(path)
        summarizer = mod.ContextSummarizer(settings_path=path)
        assert summarizer.available is False
        assert "live_summary" in (summarizer.settings_error or "")
    finally:
        restore()


def test_ss10_the_tier_settings_over_a_file_whose_values_cannot_be_built_are_refused_by_name(tmp_path):
    path = _unconstructible(tmp_path)
    loaded, restore = isolate(targets={_TIERS: source("context_summary_tiers.py")})
    try:
        mod = loaded[_TIERS]
        with pytest.raises(mod.TierSettingsError, match="summary_tiers"):
            mod.load_tier_settings(path)
    finally:
        restore()
