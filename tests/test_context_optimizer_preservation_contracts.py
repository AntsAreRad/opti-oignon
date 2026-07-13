#!/usr/bin/env python3
"""Contracts for the context optimizer's message-assembly invariants.

The optimizer trims and compresses conversation history to fit a model's
context window. Its output message list is what the executor sends to the
model, so the safety-relevant property is not how well it compresses but
what it can never drop: the constraints that ride in the system prompt, the
current user turn, and any pinned capability block. These contracts pin the
preservation guard-rails without pinning the learned or heuristic trimming
magnitudes (ratios, compression strategy, token estimation).

  * CO1 -- the base system prompt is always emitted in full as the first
    message, even under heavy overflow that forces emergency truncation. A
    policy constraint carried in the system prompt can never be trimmed away.
  * CO2 -- the current user turn is always emitted in full as the last
    message; it is never a truncation target.
  * CO3 -- a pinned capability block survives every trim, including emergency
    truncation: it is present in the final messages whenever the caller
    supplies one, even when history is being cut hard.
  * CO4 -- emergency truncation only drops the oldest history and always
    keeps at least the configured recent minimum; whatever history remains is
    a tail of the original order.
  * CO5 -- fallback preserves: with no compressor and no sliding window
    available, over-budget history is returned unchanged rather than being
    silently emptied.

Local-only (the public distribution ships no tests). Runs under pytest or the
__main__ runner. Loading follows the sibling-harness idiom: the real module
is loaded under a stand-in package, and every collaborator (budget manager,
compressor, sliding window) is injected as a deterministic fake, so no
inference backend and no sibling module are required.
"""

import importlib.util
import sys
import traceback
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OO = _REPO / "opti_oignon"


# ---------------------------------------------------------------------------
# Isolated loading (sibling-harness idiom)
# ---------------------------------------------------------------------------
def _load():
    """Load the real context optimizer under a stand-in package.

    Returns (module, restore). Collaborators are injected per test, so the
    stand-in package needs no submodules; the module's own calibrated-token
    import is guarded and falls through to the local estimator here.
    """
    keys = ("opti_oignon", "opti_oignon.context_optimizer")
    saved = {k: sys.modules.get(k) for k in keys}

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
# Deterministic collaborators
# ---------------------------------------------------------------------------
class _Budget:
    """A tiny fixed budget so any real history forces overflow handling."""

    def __init__(self, model="", window=160):
        self.system_tokens = 50
        self.project_tokens = 0
        self.history_tokens = 40
        self.user_tokens = 30
        self.reserve_tokens = 20
        self.total_window = window
        self.model = model
        self.fingerprint_tokens = 0


class _BudgetManager:
    """Returns a fixed budget; ratio fields exist for the override dance."""

    _system_ratio = 0.1
    _project_ratio = 0.1
    _history_ratio = 0.1
    _user_ratio = 0.1
    _reserve_ratio = 0.1

    def __init__(self, window=160):
        self._window = window

    def calculate_budget(
        self, model, project_active, context_window_override, fingerprint_active
    ):
        return _Budget(model=model, window=self._window)


class _CompressResult:
    """A no-op compression result: nothing compressed, history untouched."""

    compressed_count = 0
    summary = ""
    recent_messages = ()
    strategy_used = "none"
    original_count = 0
    tokens_saved = 0


class _NoopCompressor:
    """A present-but-inert compressor: exercises the compressor branch
    without altering the message content the later assertions inspect."""

    def compress(self, messages, budget_tokens, model, strategy):
        return _CompressResult()


_SYS = "SYSTEM-POLICY-CONSTRAINT-DO-NOT-DROP"
_USER = "CURRENT-USER-TURN-KEEP-INTACT"
_MANIFEST = "CAPABILITY-BLOCK-TOOL-CONSTRAINT"


def _flood_history(n=12, filler=200):
    """An oversized history guaranteed to blow a 160-token window."""
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"MSG{i} " + ("filler " * filler),
        }
        for i in range(n)
    ]


def _optimizer(mod, window=160, compressor=None):
    return mod.ContextOptimizer(
        config={
            "enabled": True,
            "emergency": {
                "enabled": True,
                "min_recent_messages": 2,
                "max_block_chars": 2000,
            },
            "report": {"max_retained": 5},
        },
        budget_manager=_BudgetManager(window=window),
        conversation_compressor=compressor,
    )


def _history_msgs(messages):
    """The subset of a message list that is flood history (MSG-tagged)."""
    return [m for m in messages if m["content"].startswith("MSG")]


# ---------------------------------------------------------------------------
# CO1 -- system prompt is preserved in full under overflow
# ---------------------------------------------------------------------------
def test_co1_system_prompt_preserved_in_full():
    mod, restore = _load()
    try:
        opt = _optimizer(mod, compressor=_NoopCompressor())
        res = opt.optimize(
            model="fake",
            system_prompt=_SYS,
            user_message=_USER,
            conversation_history=_flood_history(),
            manifest_block=_MANIFEST,
        )
        msgs = res.messages
        assert msgs, "the optimizer must emit at least a system and a user turn"
        assert msgs[0]["role"] == "system", "the first message is the system role"
        assert _SYS in msgs[0]["content"], (
            "the system prompt is emitted in full as the first message even "
            "when history is being truncated"
        )
        assert res.system_prompt.startswith(_SYS)
        assert res.report.overflow is True, (
            "this fixture must actually exercise the overflow path"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CO2 -- the current user turn is preserved in full
# ---------------------------------------------------------------------------
def test_co2_user_turn_preserved_in_full():
    mod, restore = _load()
    try:
        opt = _optimizer(mod, compressor=_NoopCompressor())
        res = opt.optimize(
            model="fake",
            system_prompt=_SYS,
            user_message=_USER,
            conversation_history=_flood_history(),
            manifest_block=_MANIFEST,
        )
        msgs = res.messages
        assert msgs[-1]["role"] == "user", "the last message is the user turn"
        assert msgs[-1]["content"] == _USER, (
            "the current user message is never trimmed and appears verbatim"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CO3 -- a pinned capability block survives emergency truncation
# ---------------------------------------------------------------------------
def test_co3_manifest_block_survives_truncation():
    mod, restore = _load()
    try:
        opt = _optimizer(mod, compressor=_NoopCompressor())
        res = opt.optimize(
            model="fake",
            system_prompt=_SYS,
            user_message=_USER,
            conversation_history=_flood_history(),
            manifest_block=_MANIFEST,
        )
        assert res.report.overflow is True, "the fixture must trigger truncation"
        assert any(_MANIFEST in m["content"] for m in res.messages), (
            "a supplied capability block is pinned and survives every trim"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CO4 -- emergency truncation drops the oldest, keeps a recent tail
# ---------------------------------------------------------------------------
def test_co4_emergency_truncation_drops_oldest_keeps_tail():
    mod, restore = _load()
    try:
        opt = _optimizer(mod, compressor=_NoopCompressor())
        res = opt.optimize(
            model="fake",
            system_prompt=_SYS,
            user_message=_USER,
            conversation_history=_flood_history(n=12),
            manifest_block=_MANIFEST,
        )
        kept = [m["content"].split()[0] for m in _history_msgs(res.messages)]
        original = [f"MSG{i}" for i in range(12)]
        assert len(kept) >= 1, "truncation keeps at least the recent minimum"
        assert len(kept) < len(original), "the fixture must actually drop some history"
        assert kept == original[len(original) - len(kept):], (
            "the surviving history is a tail of the original order: the oldest "
            "messages are dropped, never the most recent"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# CO5 -- fallback preserves over-budget history when no trimmer is available
# ---------------------------------------------------------------------------
def test_co5_fallback_preserves_history_without_trimmers():
    mod, restore = _load()
    try:
        # A large window so nothing overflows the window, but a tiny history
        # budget so the history exceeds its zone with no compressor to shrink
        # it and no sliding window to fall back to.
        opt = _optimizer(mod, window=100000, compressor=None)
        history = [
            {"role": "user", "content": "MSG0 " + ("word " * 50)},
            {"role": "assistant", "content": "MSG1 " + ("word " * 50)},
            {"role": "user", "content": "MSG2 " + ("word " * 50)},
        ]
        res = opt.optimize(
            model="fake",
            system_prompt="S",
            user_message="U",
            conversation_history=history,
        )
        kept = [m["content"].split()[0] for m in _history_msgs(res.messages)]
        assert kept == ["MSG0", "MSG1", "MSG2"], (
            "with no compressor and no sliding window, over-budget history is "
            "returned unchanged rather than silently emptied"
        )
        assert res.report.overflow is False, (
            "the window is not exceeded, so no emergency truncation fires"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        ("CO1 system prompt preserved in full", test_co1_system_prompt_preserved_in_full),
        ("CO2 user turn preserved in full", test_co2_user_turn_preserved_in_full),
        ("CO3 manifest block survives truncation", test_co3_manifest_block_survives_truncation),
        ("CO4 truncation drops oldest keeps tail", test_co4_emergency_truncation_drops_oldest_keeps_tail),
        ("CO5 fallback preserves history", test_co5_fallback_preserves_history_without_trimmers),
    ]
    passed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"PASS  {label}")
            passed += 1
        except Exception:  # noqa: BLE001 -- report and continue
            print(f"FAIL  {label}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
