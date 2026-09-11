#!/usr/bin/env python3
"""Contracts for the window composer of the onion memory.

The composer is a pure function from a registry state, a token budget and a
retrieval set to a prompt. It assembles in a fixed order -- Core, receipts
digest, selected Peels, Flesh, current turn -- under per-layer caps and a
hard total cap, and it never rewrites the Core: the Core bytes in every
assembled prompt are the registry bytes, anchored by their hash. Recalled
content is framed as data with its provenance, never as instruction.

  * CW1 -- layers appear in the declared order, each present when its input is.
  * CW2 -- every layer is within its cap and the total is within the hard
    cap; a retrieval set that overflows its cap is cut, and the cut is
    counted (the instrument reads non-zero).
  * CW3 -- Core byte-identity: the Core segment equals the registry bytes,
    its hash equals the registry root, and a tampered Core is refused.
  * CW4 -- purity: equal inputs give equal prompts, and the inputs are not
    mutated.
  * CW5 -- recalled content is data: receipts, Peels and Flesh segments carry
    provenance and are not instruction-bearing; Core and the current turn are.
  * CW6 -- the composer refuses rather than cuts what it may not cut: a Core
    over its cap, a Flesh over its cap (a turn leaves the window only with a
    receipt, never by a silent drop here), a current turn over its cap.
  * CW7 -- the budget is the YAML's, read from the file, and an
    oversubscribed budget is refused.
  * CW8 -- the token estimator is a seam: an injected estimator changes the
    count, so the default is a fallback and not a hidden constant.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source; nothing else in the package is reached.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import REPO, isolate, source  # noqa: E402

_ONION_YAML = REPO / "opti_oignon" / "config" / "onion.yaml"


def _open():
    loaded, restore = isolate(
        targets={
            "opti_oignon.memory.core_store": source("memory", "core_store.py"),
            "opti_oignon.memory.receipts": source("memory", "receipts.py"),
            "opti_oignon.memory.composer": source("memory", "composer.py"),
        },
        packages=("opti_oignon.memory",),
    )
    return (
        loaded["opti_oignon.memory.composer"],
        loaded["opti_oignon.memory.core_store"],
        loaded["opti_oignon.memory.receipts"],
        restore,
    )


def _turn(i, role="user"):
    return {"turn_id": f"t{i:02d}", "role": role, "text": f"Turn {i} says that service {i} moved to the new cluster."}


def _state(composer, core_store, receipts, *, peels=3):
    """A registry state with every layer populated."""
    core = core_store.CoreStore()
    core.add("The user is called Alice.", actor=core_store.USER)
    core.add("Answers are concise and in English.", actor=core_store.USER)
    cellar = receipts.Cellar()
    ledger = receipts.ReceiptLedger()
    flesh = receipts.Flesh([_turn(i) for i in range(1, 7)])
    flesh.evict_oldest(cellar, ledger)
    flesh.evict_oldest(cellar, ledger)
    retrieval = [
        composer.Peel(text=f"Episode {i}: the migration of service {i} was reviewed.", provenance=f"peel:{i}")
        for i in range(1, peels + 1)
    ]
    budget = composer.Budget(window=600, reserve=60, core=60, receipts=60, peels=120, flesh=200, turn=100)
    return dict(core=core, ledger=ledger, cellar=cellar, retrieval=retrieval, flesh=flesh,
                turn="Which service moved last?", budget=budget)


# ---------------------------------------------------------------------------
# CW1 -- order
# ---------------------------------------------------------------------------
def test_cw1_layers_appear_in_the_declared_order():
    composer, core_store, receipts, restore = _open()
    try:
        prompt = composer.compose(**_state(composer, core_store, receipts))
        layers = [s.layer for s in prompt.segments]
        for layer in composer.LAYERS:
            assert layers.count(layer) >= 1, f"{layer} is present when its input is"
        first = [layers.index(layer) for layer in composer.LAYERS]
        assert first == sorted(first), f"assembly order is {composer.LAYERS}, got {layers}"
        assert composer.LAYERS == ("core", "receipts", "peels", "flesh", "turn")
    finally:
        restore()


# ---------------------------------------------------------------------------
# CW2 -- caps, and a counted cut
# ---------------------------------------------------------------------------
def test_cw2_every_layer_within_its_cap_and_the_cut_is_counted():
    composer, core_store, receipts, restore = _open()
    try:
        state = _state(composer, core_store, receipts, peels=40)
        prompt = composer.compose(**state)
        budget = state["budget"]
        assert prompt.tokens <= budget.window - budget.reserve
        per_layer = {}
        for seg in prompt.segments:
            per_layer[seg.layer] = per_layer.get(seg.layer, 0) + seg.tokens
        for layer in composer.LAYERS:
            assert per_layer.get(layer, 0) <= getattr(budget, layer), f"{layer} within its cap"
        assert prompt.dropped_peels >= 1, "forty peels do not fit in 120 tokens: the cut is counted"
        assert per_layer["peels"] >= 1, "and something was kept"
        assert prompt.dropped_peels + len([s for s in prompt.segments if s.layer == "peels"]) == 40
    finally:
        restore()


# ---------------------------------------------------------------------------
# CW3 -- Core byte-identity
# ---------------------------------------------------------------------------
def test_cw3_core_bytes_are_the_registry_bytes_and_a_tampered_core_is_refused():
    composer, core_store, receipts, restore = _open()
    try:
        state = _state(composer, core_store, receipts)
        core = state["core"]
        prompt = composer.compose(**state)
        core_segments = [s for s in prompt.segments if s.layer == "core"]
        assert len(core_segments) == 1
        assert core_segments[0].text.encode("utf-8") == core.text().encode("utf-8")
        assert hashlib.sha256(core.text().encode("utf-8")).hexdigest() == core.root()
        assert prompt.core_root == core.root()
        # Tamper one entry's bytes behind the store's back: the composer refuses.
        entry = core.active()[0]
        object.__setattr__(entry, "text", entry.text + " ")
        with pytest.raises(core_store.CoreIntegrityError) as caught:
            composer.compose(**state)
        assert entry.id in str(caught.value), "the refusal names the entry"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CW4 -- purity
# ---------------------------------------------------------------------------
def test_cw4_equal_inputs_give_equal_prompts_and_inputs_are_untouched():
    composer, core_store, receipts, restore = _open()
    try:
        state = _state(composer, core_store, receipts)
        retrieval_before = list(state["retrieval"])
        flesh_before = list(state["flesh"].turns())
        open_before = list(state["ledger"].open())
        first = composer.compose(**state)
        second = composer.compose(**state)
        assert first == second
        assert first.render() == second.render()
        assert state["retrieval"] == retrieval_before
        assert state["flesh"].turns() == flesh_before
        assert state["ledger"].open() == open_before, "composing resolves nothing and evicts nothing"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CW5 -- recalled content is data
# ---------------------------------------------------------------------------
def test_cw5_recalled_content_is_data_with_provenance_never_instruction():
    composer, core_store, receipts, restore = _open()
    try:
        prompt = composer.compose(**_state(composer, core_store, receipts))
        data = [s for s in prompt.segments if s.layer in ("receipts", "peels", "flesh")]
        assert len(data) >= 3
        for seg in data:
            assert seg.instruction_bearing is False, f"{seg.layer} is data"
            assert seg.provenance, f"{seg.layer} carries provenance"
        bearing = {s.layer for s in prompt.segments if s.instruction_bearing}
        assert bearing == {"core", "turn"}
        rendered = prompt.render()
        for seg in data:
            assert seg.provenance in rendered, "provenance survives rendering"
    finally:
        restore()


# ---------------------------------------------------------------------------
# CW6 -- refuse rather than cut
# ---------------------------------------------------------------------------
def test_cw6_what_may_not_be_cut_is_refused_by_name():
    composer, core_store, receipts, restore = _open()
    try:
        state = _state(composer, core_store, receipts)
        b = state["budget"]
        over_core = composer.Budget(window=b.window, reserve=b.reserve, core=1, receipts=b.receipts,
                                    peels=b.peels, flesh=b.flesh, turn=b.turn + b.core - 1)
        with pytest.raises(composer.BudgetError, match="core"):
            composer.compose(**{**state, "budget": over_core})
        over_flesh = composer.Budget(window=b.window, reserve=b.reserve, core=b.core, receipts=b.receipts,
                                     peels=b.peels + b.flesh - 1, flesh=1, turn=b.turn)
        with pytest.raises(composer.BudgetError, match="flesh"):
            composer.compose(**{**state, "budget": over_flesh})
        with pytest.raises(composer.BudgetError, match="turn"):
            composer.compose(**{**state, "turn": " ".join(["word"] * 200)})
    finally:
        restore()


# ---------------------------------------------------------------------------
# CW7 -- the budget is the YAML's
# ---------------------------------------------------------------------------
def test_cw7_the_budget_is_read_from_the_yaml_and_oversubscription_is_refused():
    import yaml

    composer, core_store, receipts, restore = _open()
    try:
        raw = yaml.safe_load(_ONION_YAML.read_text(encoding="utf-8"))
        budget = composer.load_budget()
        assert budget.window == int(raw["window"])
        assert budget.reserve == int(raw["reserve"])
        for layer in composer.LAYERS:
            assert getattr(budget, layer) == int(raw["layers"][layer]), f"{layer} is the file's"
        assert budget.validate() == []
        assert sum(getattr(budget, layer) for layer in composer.LAYERS) + budget.reserve <= budget.window
        bad = composer.Budget(window=100, reserve=50, core=30, receipts=10, peels=10, flesh=10, turn=10)
        assert bad.validate() != []
        with pytest.raises(composer.BudgetError):
            composer.compose(**{**_state(composer, core_store, receipts), "budget": bad})
    finally:
        restore()


# ---------------------------------------------------------------------------
# CW8 -- the estimator is a seam
# ---------------------------------------------------------------------------
def test_cw8_an_injected_estimator_changes_the_count():
    composer, core_store, receipts, restore = _open()
    try:
        state = _state(composer, core_store, receipts)
        default = composer.compose(**state)
        by_words = composer.compose(**state, estimate=lambda text: len(text.split()))
        assert default.tokens >= 1
        assert by_words.tokens != default.tokens
        assert composer.estimate_tokens("one two three") == int(3 * 1.3)
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
