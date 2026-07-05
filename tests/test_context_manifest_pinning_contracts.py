#!/usr/bin/env python3
"""Contracts for the pinned capability block in the context assembly.

The capability prompt block tells the model what it can call this turn.
Once it enters the assembled context it must behave as a PINNED segment:
present in full on every assembly that carries it, never trimmed by the
compression ladder, never evicted by the optimizer, and paid for by the
compressed-history zone -- the priority order under budget is: current
turn, then the capability block, then the compressed summary, then the
retrieval zone (which keeps its own cap and behavior unchanged). These
contracts pin the properties that make the segment trustworthy:

  * Contract 1 -- small window, heavy history: the block survives verbatim
    as its own system segment while the history compresses around it; the
    history budget handed to the compressor is the base history budget
    minus the measured block, and the block's zone reports zero trimming.
  * Contract 2 -- large window: nothing is trimmed anywhere and the block
    sits exactly once between the system prompt and the history, with the
    current user turn last.
  * Contract 3 -- priority order proven by differential assembly: with and
    without the block, only the history budget moves (by exactly the
    measured block size); the user zone and the retrieval zone are
    byte-identical between the two runs.
  * Contract 4 -- the assembled total is measured, not assumed: the
    reported total equals an independent re-measure with the same
    estimator and never exceeds the profile context window (threaded
    through the existing window override), even when the emergency
    truncation has to fire -- and the block still survives that.
  * Contract 5 -- strictly additive: with no block, the assembly is
    identical to the historical one (same messages, same zones, same
    totals); nothing regresses for callers that never pass a block.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. The optimizer is driven through its
injectable seams (budget manager, compressor, token estimator), so no
module stubbing is needed and every measurement is deterministic.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_OO = _ROOT / "opti_oignon"


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
_KEYS = ("opti_oignon", "opti_oignon.context_optimizer")


def _load_optimizer_module():
    """Load the context optimizer alone under a package stub."""
    saved = {k: sys.modules.get(k) for k in _KEYS}
    pkg = types.ModuleType("opti_oignon")
    pkg.__path__ = []
    sys.modules["opti_oignon"] = pkg

    spec = importlib.util.spec_from_file_location(
        "opti_oignon.context_optimizer", _OO / "context_optimizer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opti_oignon.context_optimizer"] = mod
    spec.loader.exec_module(mod)
    pkg.context_optimizer = mod

    def restore():
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

    return mod, restore


# ---------------------------------------------------------------------------
# Controllable stand-ins (deterministic measurements)
# ---------------------------------------------------------------------------
def _tokens(text: str) -> int:
    """The deterministic yardstick every stand-in shares: len // 4."""
    return max(1, len(text or "") // 4) if text else 0


class _FakeContextManager:
    """Token estimation seam: deterministic, model-independent."""

    def estimate_tokens(self, text, model=None):
        return _tokens(text)


class _Budget:
    def __init__(self, total):
        self.total_window = total
        self.system_tokens = int(total * 0.10)
        self.project_tokens = int(total * 0.25)
        self.history_tokens = int(total * 0.40)
        self.user_tokens = int(total * 0.10)
        self.reserve_tokens = int(total * 0.15)
        self.fingerprint_tokens = 0
        self.model = ""


class _FakeBudgetManager:
    """Budget seam: honors the window override, fixed ratio split."""

    def __init__(self, default_window=2000):
        self.default_window = default_window

    def calculate_budget(self, **kwargs):
        override = kwargs.get("context_window_override") or 0
        return _Budget(override if override > 0 else self.default_window)


class _CompressResult:
    def __init__(self, summary, recent):
        self.summary = summary
        self.recent_messages = recent
        self.compressed_count = 1
        self.original_count = 1 + len(recent)
        self.tokens_saved = 0
        self.strategy_used = "stand-in"


class _FakeCompressor:
    """Compression seam: records the budget it received, returns a
    summary plus the last message so the shrink is real and visible."""

    def __init__(self):
        self.received_budget = None

    def compress(self, messages, budget_tokens, model, strategy):
        self.received_budget = budget_tokens
        summary = "summary:" + "s" * 40
        recent = messages[-1:]
        return _CompressResult(summary, recent)


def _make_optimizer(mod, compressor=None, window=2000):
    return mod.ContextOptimizer(
        config={"emergency": {"enabled": True}},
        budget_manager=_FakeBudgetManager(default_window=window),
        conversation_compressor=compressor,
        context_manager=_FakeContextManager(),
    )


def _history(n, chars):
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"m{i}:" + "x" * chars})
    return out


def _zone(report, name):
    for z in report.zones:
        if z.zone == name:
            return z
    return None


_BLOCK = (
    "Capabilities available this turn:\n"
    "- web_search: look information up on the web\n"
    "- execute_code: run code in the disposable sandbox"
)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
def test_c1_block_survives_while_history_compresses():
    mod, restore = _load_optimizer_module()
    try:
        compressor = _FakeCompressor()
        opt = _make_optimizer(mod, compressor=compressor, window=1200)
        block_tokens = _tokens(_BLOCK)
        result = opt.optimize(
            model="tiny",
            system_prompt="sys:" + "p" * 60,
            user_message="turn:" + "u" * 60,
            conversation_history=_history(8, 400),
            manifest_block=_BLOCK,
        )
        # The block is present verbatim, exactly once, as a system segment.
        hits = [
            m for m in result.messages
            if m.get("role") == "system" and m.get("content") == _BLOCK
        ]
        assert len(hits) == 1, f"pinned block present {len(hits)} times"
        # The history really compressed around it.
        hist_zone = _zone(result.report, "history")
        assert hist_zone is not None
        assert hist_zone.strategy.startswith("compressed"), hist_zone.strategy
        # The compressor budget is the base history budget minus the block.
        base_history = _Budget(1200).history_tokens
        assert compressor.received_budget == base_history - block_tokens, (
            compressor.received_budget, base_history, block_tokens,
        )
        # The pinned zone never trims.
        pin = _zone(result.report, "manifest")
        assert pin is not None, "no pinned zone reported"
        assert pin.trimmed_tokens == 0
        assert pin.actual_tokens == block_tokens
        assert pin.strategy == "pinned", pin.strategy
    finally:
        restore()


def test_c2_large_window_keeps_everything_and_orders_segments():
    mod, restore = _load_optimizer_module()
    try:
        opt = _make_optimizer(mod, compressor=_FakeCompressor(), window=100000)
        history = _history(4, 80)
        result = opt.optimize(
            model="big",
            system_prompt="sys prompt",
            user_message="the current turn",
            conversation_history=history,
            manifest_block=_BLOCK,
        )
        msgs = result.messages
        assert msgs[0]["role"] == "system" and msgs[0]["content"] == "sys prompt"
        assert msgs[1]["role"] == "system" and msgs[1]["content"] == _BLOCK
        assert msgs[-1] == {"role": "user", "content": "the current turn"}
        assert msgs[2:-1] == history, "history altered on a roomy window"
        hist_zone = _zone(result.report, "history")
        assert hist_zone.trimmed_tokens == 0
        assert result.report.overflow is False
    finally:
        restore()


def test_c3_priority_order_only_the_history_budget_moves():
    mod, restore = _load_optimizer_module()
    try:
        kwargs = dict(
            model="tiny",
            system_prompt="sys:" + "p" * 40,
            user_message="turn:" + "u" * 40,
            conversation_history=_history(6, 200),
        )
        comp_a, comp_b = _FakeCompressor(), _FakeCompressor()
        opt_a = _make_optimizer(mod, compressor=comp_a, window=1200)
        opt_b = _make_optimizer(mod, compressor=comp_b, window=1200)
        with_block = opt_a.optimize(manifest_block=_BLOCK, **kwargs)
        without = opt_b.optimize(**kwargs)
        block_tokens = _tokens(_BLOCK)
        # The summary cedes exactly the block's measure...
        zb = _zone(with_block.report, "history")
        zn = _zone(without.report, "history")
        assert zn.budgeted_tokens - zb.budgeted_tokens == block_tokens, (
            zn.budgeted_tokens, zb.budgeted_tokens, block_tokens,
        )
        # ...while the current turn and the retrieval zone never move.
        for name in ("user", "project"):
            za, zc = _zone(with_block.report, name), _zone(without.report, name)
            assert (za.budgeted_tokens, za.actual_tokens, za.trimmed_tokens) \
                == (zc.budgeted_tokens, zc.actual_tokens, zc.trimmed_tokens), name
    finally:
        restore()


def test_c4_total_is_measured_and_capped_even_under_emergency():
    mod, restore = _load_optimizer_module()
    try:
        # No compressor and no sliding window: the only way down is the
        # emergency truncation, which must spare the pinned block.
        opt = _make_optimizer(mod, compressor=None, window=2000)
        window = 1000
        result = opt.optimize(
            model="tiny",
            system_prompt="sys:" + "p" * 196,
            user_message="turn:" + "u" * 195,
            conversation_history=_history(30, 96),
            manifest_block=_BLOCK,
            context_window_override=window,
        )
        assert result.report.overflow is True, "emergency did not fire"
        hits = [
            m for m in result.messages
            if m.get("role") == "system" and m.get("content") == _BLOCK
        ]
        assert len(hits) == 1, "the pinned block did not survive emergency"
        remeasured = sum(_tokens(m.get("content", "")) for m in result.messages)
        assert result.total_tokens == remeasured, (
            result.total_tokens, remeasured,
        )
        assert result.total_tokens <= window, (result.total_tokens, window)
        assert result.report.total_window == window
    finally:
        restore()


def test_c5_no_block_means_the_historical_assembly_exactly():
    mod, restore = _load_optimizer_module()
    try:
        kwargs = dict(
            model="tiny",
            system_prompt="sys prompt",
            user_message="the turn",
            conversation_history=_history(4, 120),
        )
        opt_a = _make_optimizer(mod, compressor=_FakeCompressor(), window=5000)
        opt_b = _make_optimizer(mod, compressor=_FakeCompressor(), window=5000)
        legacy = opt_a.optimize(**kwargs)
        explicit_none = opt_b.optimize(manifest_block=None, **kwargs)
        assert explicit_none.messages == legacy.messages
        assert explicit_none.total_tokens == legacy.total_tokens
        names_a = [z.zone for z in legacy.report.zones]
        names_b = [z.zone for z in explicit_none.report.zones]
        assert names_a == names_b
        assert "manifest" not in names_b
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
_TESTS = (
    "test_c1_block_survives_while_history_compresses",
    "test_c2_large_window_keeps_everything_and_orders_segments",
    "test_c3_priority_order_only_the_history_budget_moves",
    "test_c4_total_is_measured_and_capped_even_under_emergency",
    "test_c5_no_block_means_the_historical_assembly_exactly",
)


def _main() -> int:
    passed = 0
    for name in _TESTS:
        try:
            globals()[name]()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
            passed += 1
    print(f"{passed}/{len(_TESTS)} passed")
    return 0 if passed == len(_TESTS) else 1


if __name__ == "__main__":
    sys.exit(_main())
