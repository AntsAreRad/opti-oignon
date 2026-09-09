#!/usr/bin/env python3
"""What the unified retrieval front promises to the prompt it feeds.

Three retrieval sources compose one prompt -- project documents, personal
memory and the conversation archive -- and each one runs its own engine with
its own score scale. This layer stands in front of all three, and these
clauses pin the properties that make that front honest:

  * Every source is reached through the callable it was registered under,
    with the query and the per-source limit and nothing else. The layer
    never writes anywhere; the archive stays as read-only here as it is
    everywhere else.
  * A failing source is named in the report and costs only its own
    contribution. One broken engine never empties the prompt.
  * Provenance is stamped from the registry, not read off the item, so a
    source cannot claim another's name for its own material.
  * Cross-source duplicates are dropped before injection and the survivor
    is the source's own object, byte-for-byte -- the higher-ranked copy
    wins because it would have been injected first.
  * Scores are compared through per-source normalisation: each source's
    best is worth as much as every other source's best, and the order
    within a source is the source's own. Ties fall to the source's own
    score first, then to source name, origin and content, so the order is
    reproducible either way.
  * The announced ordering is the ordering actually applied. Without a
    local reranker the report says so; a reranker that is present but not
    enabled is never consulted; a reranker that fails is named and the
    fused order stands. Absence is reported, never papered over.
  * Injection formatting drops whole items when the budget runs out. It
    never truncates inside an item, because half a snippet reads as a
    whole one.

The module under test is loaded on the shared isolation window together
with the deduplication helper it leans on; everything else in the package
is manufactured absence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_DEDUP = "opti_oignon.context_dedup"
_MOD = "opti_oignon.unified_retrieval"


def _load():
    """Open the window and load the layer plus the real deduplicator."""
    loaded, restore = isolate(
        targets={
            _DEDUP: source("context_dedup.py"),
            _MOD: source("unified_retrieval.py"),
        },
        blocked=("ollama",),
    )
    return loaded[_MOD], restore


def _item(mod, source_name, content, score, origin=""):
    return mod.RetrievedItem(
        source=source_name, content=content, score=score, origin=origin,
    )


class _Recorder:
    """A source that records how it was called and returns fixed items."""

    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def __call__(self, query, limit):
        self.calls.append((query, limit))
        return list(self.items)


def _raiser(query, limit):
    raise RuntimeError("engine down")


# --- The clauses -------------------------------------------------------------


def test_c0_window_holds_and_the_module_is_import_pure():
    """The layer loads with only the deduplicator seeded and pulls nothing."""
    M, restore = _load()
    try:
        assert sys.modules.get("ollama", "absent") is None
        project = {
            k for k in sys.modules
            if k.split(".")[0] == "opti_oignon" and sys.modules[k] is not None
        }
        assert project == {"opti_oignon", _DEDUP, _MOD}, project
    finally:
        restore()


def test_c1_without_a_reranker_the_report_says_fused_order():
    """No reranker: the label is honest and the order is the score order."""
    M, restore = _load()
    try:
        rag = _Recorder([
            _item(M, "rag", "alpha facts", 0.9, "a.md"),
            _item(M, "rag", "beta facts", 0.3, "b.md"),
        ])
        r = M.UnifiedRetriever(sources={"rag": rag}).retrieve(query="facts")
        assert r.ordering == "fused-order"
        assert [it.content for it in r.items] == ["alpha facts", "beta facts"]
        assert r.failures == {}
    finally:
        restore()


def test_c2_a_cross_source_duplicate_is_dropped_and_the_survivor_untouched():
    """The higher-ranked copy survives as its source's own object."""
    M, restore = _load()
    try:
        winner = _item(M, "rag", "the quick brown fox jumps over the lazy dog",
                       0.9, "doc.md")
        copy = _item(M, "archive", "The quick brown fox jumps over the lazy dog.",
                     0.8, "msg-4")
        rag = _Recorder([winner])
        archive = _Recorder([copy])
        r = M.UnifiedRetriever(
            sources={"rag": rag, "archive": archive},
        ).retrieve(query="fox")
        assert r.dropped_duplicates == 1
        assert len(r.items) == 1
        assert r.items[0] is winner
        assert r.items[0].content == "the quick brown fox jumps over the lazy dog"
    finally:
        restore()


def test_c3_a_failing_source_is_named_and_the_others_still_deliver():
    """One broken engine costs its own contribution, nothing more."""
    M, restore = _load()
    try:
        memory = _Recorder([_item(M, "memory", "a remembered fact", 0.7, "m-1")])
        r = M.UnifiedRetriever(
            sources={"rag": _raiser, "memory": memory},
        ).retrieve(query="fact")
        assert [it.content for it in r.items] == ["a remembered fact"]
        assert set(r.failures) == {"rag"}
        assert "RuntimeError" in r.failures["rag"]
        assert r.per_source == {"memory": 1}
    finally:
        restore()


def test_c4_each_sources_best_outranks_every_sources_second():
    """Per-source normalisation: tops compare as tops, and ties are stable."""
    M, restore = _load()
    try:
        rag = _Recorder([
            _item(M, "rag", "top document", 10.0, "a.md"),
            _item(M, "rag", "second document", 4.0, "b.md"),
        ])
        memory = _Recorder([
            _item(M, "memory", "top memory", 0.6, "m-1"),
            _item(M, "memory", "second memory", 0.24, "m-2"),
        ])
        retriever = M.UnifiedRetriever(sources={"rag": rag, "memory": memory})
        first = [it.content for it in retriever.retrieve(query="q").items]
        assert first[:2] == ["top document", "top memory"]
        assert first[2:] == ["second document", "second memory"]
        again = [it.content for it in retriever.retrieve(query="q").items]
        assert again == first
    finally:
        restore()


def test_c5_an_enabled_reranker_is_applied_and_named():
    """The reranker sees the survivors and its order is adopted under its name."""
    M, restore = _load()
    try:
        seen = {}

        def flip(query, items):
            seen["query"] = query
            seen["items"] = list(items)
            return list(reversed(items))

        rag = _Recorder([
            _item(M, "rag", "first", 0.9, "a"),
            _item(M, "rag", "unrelated second", 0.4, "b"),
        ])
        r = M.UnifiedRetriever(
            sources={"rag": rag},
            reranker=("local-cross-encoder", flip),
            reranker_enabled=True,
        ).retrieve(query="which")
        assert r.ordering == "local-cross-encoder"
        assert [it.content for it in r.items] == ["unrelated second", "first"]
        assert seen["query"] == "which"
        assert [it.content for it in seen["items"]] == ["first", "unrelated second"]
    finally:
        restore()


def test_c6_a_reranker_that_is_present_but_not_enabled_is_never_consulted():
    """Governance: presence is not consent. Disabled means untouched."""
    M, restore = _load()
    try:
        calls = []

        def spy(query, items):
            calls.append(query)
            return list(items)

        rag = _Recorder([_item(M, "rag", "only", 0.5, "a")])
        r = M.UnifiedRetriever(
            sources={"rag": rag},
            reranker=("local-cross-encoder", spy),
            reranker_enabled=False,
        ).retrieve(query="q")
        assert calls == []
        assert r.ordering == "fused-order"
    finally:
        restore()


def test_c7_injection_formatting_drops_whole_items_never_truncates():
    """When the budget runs out, the next item is absent, not cut."""
    M, restore = _load()
    try:
        first = _item(M, "rag", "kept text", 0.9, "a.md")
        second = _item(M, "rag", "this one does not fit at all", 0.4, "b.md")
        retriever = M.UnifiedRetriever(sources={})
        block = retriever.format_for_injection(
            [first, second], budget_tokens=6, estimate=lambda text: len(text.split()),
        )
        assert "kept text" in block
        assert "does not fit" not in block
        assert block == retriever.format_for_injection(
            [first], budget_tokens=6, estimate=lambda text: len(text.split()),
        )
    finally:
        restore()


def test_c8_provenance_is_stamped_from_the_registry_not_the_item():
    """A source cannot sign its material with another source's name."""
    M, restore = _load()
    try:
        liar = _Recorder([_item(M, "rag", "borrowed words", 0.5, "m-9")])
        r = M.UnifiedRetriever(sources={"memory": liar}).retrieve(query="q")
        assert [it.source for it in r.items] == ["memory"]
    finally:
        restore()


def test_c9_checkpoint_before_apply_is_hardcoded_true():
    """The safety posture is a module constant, not a configuration."""
    M, restore = _load()
    try:
        assert M.checkpoint_before_apply is True
    finally:
        restore()


def test_c10_sources_receive_the_query_and_the_limit_and_nothing_else():
    """The layer's whole demand on a source is (query, limit), read-only."""
    M, restore = _load()
    try:
        rag = _Recorder([_item(M, "rag", "one", 1.0, "a")])
        M.UnifiedRetriever(
            sources={"rag": rag}, per_source_limit=3,
        ).retrieve(query="exact words")
        assert rag.calls == [("exact words", 3)]
    finally:
        restore()


def test_c11_a_failing_reranker_is_named_and_the_fused_order_stands():
    """Absence and failure degrade the same way: honestly."""
    M, restore = _load()
    try:
        def broken(query, items):
            raise ValueError("model missing")

        rag = _Recorder([
            _item(M, "rag", "first", 0.9, "a"),
            _item(M, "rag", "unrelated second", 0.4, "b"),
        ])
        r = M.UnifiedRetriever(
            sources={"rag": rag},
            reranker=("local-cross-encoder", broken),
            reranker_enabled=True,
        ).retrieve(query="q")
        assert r.ordering == "fused-order"
        assert "reranker" in r.failures
        assert "ValueError" in r.failures["reranker"]
        assert [it.content for it in r.items] == ["first", "unrelated second"]
    finally:
        restore()


def test_c12_a_snippet_the_prompt_already_carries_is_not_injected_again():
    """The already-composed text is part of the corpus a candidate faces."""
    M, restore = _load()
    try:
        rag = _Recorder([
            _item(M, "rag", "the deadline moved to friday afternoon", 0.9, "a"),
        ])
        r = M.UnifiedRetriever(sources={"rag": rag}).retrieve(
            query="deadline",
            already_composed="Note: the deadline moved to Friday afternoon.",
        )
        assert r.items == []
        assert r.dropped_duplicates == 1
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
