#!/usr/bin/env python3
"""Contracts for the tiered conversation summary and its proven invalidation.

The live summarizer keeps one cumulative paragraph and re-merges it forever;
nothing bounds it, nothing detects that the ground under it moved, and no
suite pins any of it. The tier layer gives the summary levels backed by the
archive: frozen SEGMENTS over exact message spans, one ROLLUP built from the
segments alone, and a live partial for whatever is not frozen yet. These
clauses hold the machinery from every side that matters:

  * A span digest is a statement about exact bytes: same span, same digest;
    one changed byte, one changed role, one changed id, one changed order --
    a different digest.
  * Freezing is fail-safe: a summarizer that declines produces no segment,
    and never destroys the segments already frozen.
  * The rollup is built from segment texts ONLY. Raw messages never reach
    the rollup builder a second time, and the rollup names the exact
    segments it was built from.
  * Staleness is refused, not hoped away: a segment whose archived span no
    longer matches its digest is dropped, and so is any rollup that was
    built on it. What still matches survives byte for byte.
  * An unreadable archive proves nothing, so nothing is composed from tiers
    and nothing is written: refusing to verify must never destroy state.
  * The tier record round-trips losslessly through conversation metadata,
    and an absent or malformed record is an empty state, never a crash.
  * The module never opens the conversation store itself and never writes
    anywhere: it reads through an injected reader and hands metadata back
    to its caller.
  * Composition is ordered and total: rollup first, then the live partial;
    without a rollup, the segment texts stand in; with nothing, the partial
    stands alone.
  * The executor actually reaches the seam: the advance call rides the
    summary path's metadata write, and the restore path consults the
    composed tiers before the legacy cumulative key.
  * Advancing is oldest-first, never covers the verbatim tail, produces
    contiguous non-overlapping spans, and is idempotent on an unchanged
    archive.

The tier module is loaded from its source file inside the shared isolation
window; readers and summarizers are injected, so no project engine and no
model is ever consulted here.
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_MOD = "opti_oignon.context_summary_tiers"
_MOD_PATH = _ROOT / "opti_oignon" / "context_summary_tiers.py"
_EXECUTOR_PATH = _ROOT / "opti_oignon" / "executor.py"


def _load():
    """Load the tier module from source inside the shared window."""
    loaded, restore = isolate(targets={_MOD: source("context_summary_tiers.py")})
    return loaded[_MOD], restore


def _messages(n, *, start_id=1, size=150):
    """An archive of ``n`` alternating messages with ascending ids.

    Every content line carries its own id so any two messages differ and a
    digest over the span is sensitive to each of them.
    """
    out = []
    for k in range(n):
        mid = start_id + k
        role = "user" if k % 2 == 0 else "assistant"
        body = f"message {mid} " + ("x" * max(0, size - 12))
        out.append({"id": mid, "role": role, "content": body})
    return out


class _Reader:
    """An injected archive reader over a mutable in-memory message list."""

    def __init__(self, messages):
        self.messages = messages
        self.calls = 0

    def __call__(self, conversation_id):
        self.calls += 1
        if self.messages is None:
            return None
        return [dict(m) for m in self.messages]


class _SummarizeSpy:
    """Records every span it is asked to summarize; answers by a template."""

    def __init__(self, template="SEG<{ids}>", fail_after=None):
        self.template = template
        self.fail_after = fail_after
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            return None
        ids = ",".join(str(m.get("id", "?")) for m in messages)
        return self.template.format(ids=ids)


def _manager(module, reader, *, budget=100, tail=2):
    return module.TierManager(
        archive_reader=reader,
        segment_budget_tokens=budget,
        tail_keep_messages=tail,
    )


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def test_t1_span_digest_is_exact_and_sensitive():
    module, restore = _load()
    try:
        span = _messages(4)
        base = module.span_digest(span)
        assert base == module.span_digest([dict(m) for m in span])

        changed = [dict(m) for m in span]
        changed[2]["content"] = changed[2]["content"] + "!"
        assert module.span_digest(changed) != base

        role_flip = [dict(m) for m in span]
        role_flip[1]["role"] = "user"
        assert module.span_digest(role_flip) != base

        renumbered = [dict(m) for m in span]
        renumbered[0]["id"] = 99
        assert module.span_digest(renumbered) != base

        reordered = [span[1], span[0], span[2], span[3]]
        assert module.span_digest(reordered) != base
    finally:
        restore()


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def test_t2_a_declining_summarizer_freezes_nothing_and_destroys_nothing():
    module, restore = _load()
    try:
        reader = _Reader(_messages(10))
        manager = _manager(module, reader)

        first_only = _SummarizeSpy(fail_after=1)
        update = manager.advance("conv", {}, summarize_fn=first_only)
        state = module.TierState.from_metadata(update)
        assert len(state.segments) == 1, (
            "the first freeze succeeded and must be kept; the declined one "
            "must simply not exist"
        )

        never = _SummarizeSpy(fail_after=0)
        update2 = manager.advance("conv", update, summarize_fn=never)
        state2 = module.TierState.from_metadata(update2)
        assert [s.digest for s in state2.segments] == [
            s.digest for s in state.segments
        ], "declining to summarize must never touch what is already frozen"
        assert state2.segments[0].text == state.segments[0].text
    finally:
        restore()


def test_t3_the_rollup_is_built_from_segment_texts_only():
    module, restore = _load()
    try:
        sentinel = "RAW-ONLY-9481"
        archive = _messages(10)
        for m in archive:
            m["content"] += f" {sentinel}"
        reader = _Reader(archive)
        manager = _manager(module, reader)

        spy = _SummarizeSpy(template="TIER<{ids}>")
        update = manager.advance("conv", {}, summarize_fn=spy)
        state = module.TierState.from_metadata(update)
        assert len(state.segments) >= module.ROLLUP_MIN_SEGMENTS
        assert state.rollup is not None

        rollup_inputs = [
            call for call in spy.calls
            if all(m.get("role") == "summary" for m in call)
        ]
        assert rollup_inputs, "the rollup builder was never asked"
        fed = [m["content"] for m in rollup_inputs[-1]]
        assert fed == [s.text for s in state.segments], (
            "the rollup must be built from the segment texts, in order"
        )
        assert all(sentinel not in content for content in fed), (
            "raw message content reached the rollup builder a second time"
        )
        assert state.rollup.built_from == [s.digest for s in state.segments]
    finally:
        restore()


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_t4_a_stale_span_is_refused_and_takes_its_rollup_with_it():
    module, restore = _load()
    try:
        archive = _messages(10)
        reader = _Reader(archive)
        manager = _manager(module, reader)
        update = manager.advance("conv", {}, summarize_fn=_SummarizeSpy())
        state = module.TierState.from_metadata(update)
        assert len(state.segments) == 2 and state.rollup is not None

        intact = state.segments[1]
        victim_id = state.segments[0].first_id
        for m in archive:
            if m["id"] == victim_id:
                m["content"] = "rewritten behind the digest's back"

        verified = manager.verify(state, "conv")
        digests = [s.digest for s in verified.segments]
        assert state.segments[0].digest not in digests, (
            "a span that no longer matches its digest must be dropped"
        )
        assert verified.rollup is None, (
            "a rollup built on a dropped segment is itself stale"
        )
        kept = [s for s in verified.segments if s.digest == intact.digest]
        assert kept and kept[0].text == intact.text, (
            "what still matches must survive byte for byte"
        )
    finally:
        restore()


def test_t5_an_unreadable_archive_composes_nothing_and_writes_nothing():
    module, restore = _load()
    try:
        reader = _Reader(_messages(10))
        manager = _manager(module, reader)
        update = manager.advance("conv", {}, summarize_fn=_SummarizeSpy())
        assert module.TIERS_METADATA_KEY in update

        reader.messages = None
        assert manager.advance("conv", update, summarize_fn=_SummarizeSpy()) == {}, (
            "refusing to verify must never persist anything"
        )
        composed = manager.compose("conv", update, live_partial="partial only")
        assert composed == "partial only", (
            "tiers that cannot be verified must not be injected"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def test_t6_the_record_round_trips_and_tolerates_garbage():
    module, restore = _load()
    try:
        reader = _Reader(_messages(10))
        manager = _manager(module, reader)
        update = manager.advance("conv", {}, summarize_fn=_SummarizeSpy())

        state = module.TierState.from_metadata(update)
        again = module.TierState.from_metadata(
            {module.TIERS_METADATA_KEY: state.to_metadata_value()}
        )
        assert again.to_metadata_value() == state.to_metadata_value()
        assert again.version == module.TIERS_VERSION

        for garbage in (None, {}, {"other": 1},
                        {module.TIERS_METADATA_KEY: "not a mapping"},
                        {module.TIERS_METADATA_KEY: {"version": 99}},
                        {module.TIERS_METADATA_KEY: {"version": 1,
                                                     "segments": "nope"}}):
            empty = module.TierState.from_metadata(garbage)
            assert empty.segments == [] and empty.rollup is None
    finally:
        restore()


def test_t7_the_module_never_opens_the_store_and_never_writes():
    tree = ast.parse(_MOD_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "sqlite3" not in imported, (
        "the tier layer must read through the conversation API, never open "
        "the store itself"
    )
    called_attrs = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = {"execute", "executescript", "executemany", "commit",
                 "add_message", "update_conversation_metadata",
                 "create_conversation"}
    assert not (called_attrs & forbidden), (
        f"write-capable calls found in the tier layer: "
        f"{sorted(called_attrs & forbidden)}"
    )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_t8_composition_is_ordered_and_total():
    module, restore = _load()
    try:
        reader = _Reader(_messages(10))
        manager = _manager(module, reader)
        update = manager.advance("conv", {}, summarize_fn=_SummarizeSpy())
        state = module.TierState.from_metadata(update)
        assert state.rollup is not None

        block = manager.compose("conv", update, live_partial="LIVE-TAIL")
        assert block.startswith(state.rollup.text)
        assert block.endswith("LIVE-TAIL")
        assert manager.compose("conv", update) == state.rollup.text

        headless = module.TierState(
            segments=list(state.segments), rollup=None,
        )
        meta = {module.TIERS_METADATA_KEY: headless.to_metadata_value()}
        block2 = manager.compose("conv", meta, live_partial="LIVE-TAIL")
        for segment in state.segments:
            assert segment.text in block2
        assert block2.index(state.segments[0].text) < block2.index(
            state.segments[1].text
        )
        assert block2.endswith("LIVE-TAIL")

        assert manager.compose("conv", {}, live_partial="LIVE-TAIL") == "LIVE-TAIL"
        assert manager.compose("conv", {}) == ""
    finally:
        restore()


# ---------------------------------------------------------------------------
# Reach
# ---------------------------------------------------------------------------


def test_t9_the_executor_reaches_the_seam_on_both_paths():
    tree = ast.parse(_EXECUTOR_PATH.read_text(encoding="utf-8"))

    guarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for stmt in ast.walk(node):
                if (isinstance(stmt, ast.ImportFrom)
                        and stmt.module == "context_summary_tiers"):
                    guarded = True
    assert guarded, (
        "the tier import must be guarded like every sibling feature, so an "
        "install without the module keeps the exact historical pipeline"
    )

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert "_summarize_old_messages" in functions
    assert "_build_conversation_messages" in functions

    def _attr_call_lines(fn, name):
        return [
            node.lineno
            for node in ast.walk(functions[fn])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        ]

    assert _attr_call_lines("_summarize_old_messages", "advance"), (
        "the summary path never advances the tiers"
    )
    assert _attr_call_lines("_summarize_old_messages",
                            "update_conversation_metadata"), (
        "the summary path lost its metadata write"
    )

    compose_lines = _attr_call_lines("_build_conversation_messages", "compose")
    assert compose_lines, "the restore path never consults the composed tiers"
    legacy_reads = [
        node.lineno
        for node in ast.walk(functions["_build_conversation_messages"])
        if isinstance(node, ast.Constant) and node.value == "context_summary"
    ]
    assert legacy_reads, "the legacy cumulative key must remain the fallback"
    assert min(compose_lines) < min(legacy_reads), (
        "the composed tiers must be consulted before the legacy key"
    )


def test_t10_the_checkpoint_posture_is_hardcoded():
    tree = ast.parse(_MOD_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id == "checkpoint_before_apply"):
                    assert ast.literal_eval(node.value) is True
                    return
    raise AssertionError("checkpoint_before_apply is not declared True")


# ---------------------------------------------------------------------------
# Advancing
# ---------------------------------------------------------------------------


def test_t11_advancing_is_bounded_ordered_and_idempotent():
    module, restore = _load()
    try:
        archive = _messages(12)
        reader = _Reader(archive)
        manager = _manager(module, reader, budget=100, tail=3)

        update = manager.advance("conv", {}, summarize_fn=_SummarizeSpy())
        state = module.TierState.from_metadata(update)
        assert state.segments, "a long archive must freeze at least one span"

        tail_ids = {m["id"] for m in archive[-3:]}
        for segment in state.segments:
            assert segment.first_id <= segment.last_id
            covered = set(range(segment.first_id, segment.last_id + 1))
            assert not (covered & tail_ids), (
                "the verbatim tail must never be frozen"
            )
        firsts = [s.first_id for s in state.segments]
        assert firsts == sorted(firsts), "freezing must be oldest-first"
        for before, after in zip(state.segments, state.segments[1:]):
            assert after.first_id == before.last_id + 1, (
                "spans must be contiguous and non-overlapping"
            )
        assert module.covered_up_to(state) == state.segments[-1].last_id

        again = manager.advance("conv", update, summarize_fn=_SummarizeSpy())
        assert again == update, (
            "an unchanged archive must advance to the identical record"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                import traceback

                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
