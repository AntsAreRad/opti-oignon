#!/usr/bin/env python3
"""How the context optimizer gates the unified retrieval layer.

The optimizer owns the project zone of the assembled prompt. A unified
retrieval layer can fill that zone instead of the per-project builder, but
only under an explicit configuration key, and only as long as it actually
delivers. These clauses pin the gate from both sides:

  * Off by default. A configuration that never mentions the key keeps the
    exact historical pipeline, and the layer's module is never touched --
    proven with a seeded spy that records every approach.
  * On means on: the layer is asked with the query, the conversation, the
    project and the text already composed, its block lands in the project
    zone, and the zone report names the strategy.
  * The zone budget is handed through unchanged, so the layer trims to the
    same allowance the per-project builder would have had.
  * A layer that raises costs nothing: the optimizer falls back to the
    per-project builder in the same call and no exception escapes.

The optimizer is loaded on the shared isolation window; the retrieval layer
is a seeded stand-in module, so these clauses hold whatever that layer does.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_MOD = "opti_oignon.context_optimizer"
_LAYER = "opti_oignon.unified_retrieval"


class _SpyRetriever:
    """Records every approach; answers with a fixed block or a failure."""

    def __init__(self, text="UNIFIED-BLOCK", fail=False):
        self.text = text
        self.fail = fail
        self.retrieve_calls = []
        self.format_calls = []

    def retrieve(self, *, query, conversation_id=None, project_id=None,
                 already_composed=""):
        self.retrieve_calls.append({
            "query": query,
            "conversation_id": conversation_id,
            "project_id": project_id,
            "already_composed": already_composed,
        })
        if self.fail:
            raise RuntimeError("layer down")
        return types.SimpleNamespace(
            query=query, items=["marker"], per_source={"rag": 1},
            dropped_duplicates=0, ordering="fused-order", failures={},
            elapsed_ms=0.0,
        )

    def format_for_injection(self, items, *, budget_tokens, estimate=None):
        self.format_calls.append({"items": list(items),
                                  "budget_tokens": budget_tokens})
        return self.text


class _Builder:
    """The per-project builder the legacy path relies on."""

    def __init__(self):
        self.calls = []

    def build_context(self, project_id, query, budget_tokens=0):
        self.calls.append((project_id, query, budget_tokens))
        return types.SimpleNamespace(
            context_text="LEGACY-BLOCK", chunks_used=1, total_tokens_estimate=2,
        )


def _load(spy):
    layer = types.ModuleType(_LAYER)
    layer.get_unified_retriever = lambda: spy
    loaded, restore = isolate(
        targets={_MOD: source("context_optimizer.py")},
        seeded={_LAYER: layer},
        blocked=("ollama",),
    )
    return loaded[_MOD], restore


def _optimize(mod, config, builder, **kwargs):
    optimizer = mod.ContextOptimizer(
        config=config, project_context_builder=builder,
    )
    return optimizer.optimize(
        model="m", system_prompt="SYS", user_message="ask", project_id="p1",
        **kwargs,
    )


def _zone(result, name):
    return next(z for z in result.report.zones if z.zone == name)


def test_g1_the_layer_is_off_by_default_and_never_touched():
    spy = _SpyRetriever()
    mod, restore = _load(spy)
    try:
        builder = _Builder()
        result = _optimize(mod, {}, builder)
        assert spy.retrieve_calls == []
        assert builder.calls, "the legacy builder was expected to run"
        assert _zone(result, "project").strategy == "rag"
        assert "LEGACY-BLOCK" in result.system_prompt
    finally:
        restore()


def test_g2_enabled_means_the_layer_fills_the_project_zone():
    spy = _SpyRetriever(text="UNIFIED-BLOCK")
    mod, restore = _load(spy)
    try:
        builder = _Builder()
        config = {"unified_retrieval": {"enabled": True}}
        result = _optimize(mod, config, builder, conversation_id="c9")
        assert len(spy.retrieve_calls) == 1
        call = spy.retrieve_calls[0]
        assert call["query"] == "ask"
        assert call["conversation_id"] == "c9"
        assert call["project_id"] == "p1"
        assert call["already_composed"] == "SYS"
        assert builder.calls == []
        zone = _zone(result, "project")
        assert zone.strategy == "unified"
        assert "UNIFIED-BLOCK" in result.system_prompt
    finally:
        restore()


def test_g3_a_failing_layer_falls_back_to_the_builder_in_the_same_call():
    spy = _SpyRetriever(fail=True)
    mod, restore = _load(spy)
    try:
        builder = _Builder()
        config = {"unified_retrieval": {"enabled": True}}
        result = _optimize(mod, config, builder)
        assert len(spy.retrieve_calls) == 1
        assert builder.calls, "the fallback was expected to run"
        assert _zone(result, "project").strategy == "rag"
        assert "LEGACY-BLOCK" in result.system_prompt
    finally:
        restore()


def test_g4_the_project_budget_is_handed_through_unchanged():
    spy = _SpyRetriever()
    mod, restore = _load(spy)
    try:
        config = {"unified_retrieval": {"enabled": True}}
        _optimize(mod, config, _Builder())
        assert len(spy.format_calls) == 1
        assert spy.format_calls[0]["budget_tokens"] == 2000
        assert spy.format_calls[0]["items"] == ["marker"]
    finally:
        restore()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
