#!/usr/bin/env python3
"""What cross-source deduplication promises before a snippet is injected.

Several retrieval sources compose one prompt and none of them can see what
the others already contributed, so the same passage arrives more than once.
Dropping the repeat is only safe if the drop is the ONLY thing that ever
happens -- these clauses pin that, from both directions:

  * Survivors are the source's own objects, handed back untouched and in
    order. Nothing is rewritten, merged, reordered, or moved from one
    source into another, so a caller keeps every field its own retriever
    attached and the model reads the source's own bytes.
  * A passage the prompt already carries is recognised across the
    formatting differences that separate two sources quoting the same
    thing -- case, punctuation, line wrapping, a role prefix.
  * Near-duplicates are dropped only above a stated threshold, and the
    threshold is the caller's to set. Below it the candidate survives:
    over-dropping removes evidence silently, which is the worse failure.
  * Survivors accumulate. A candidate that repeats an earlier survivor of
    the same batch is dropped too, because by the time it would be injected
    the earlier one already says it.
  * The comparison covers BOTH composition layouts. A turn puts some blocks
    in the system prompt and holds others back as volatile parts; looking
    at only one of them would be blind exactly when the other is in use.
  * It is pure. Same inputs, same survivors, every time, with the caller's
    own list left alone -- no clock, no network, no persistence, no model,
    and no optional dependency to be absent.
  * The hub wires it on the archive path BEFORE injection, not after, and
    that wiring is derived from the tree here rather than assumed.

Local-only (the public distribution ships no tests). Runs under pytest or
directly via the __main__ runner. The module under test is loaded from its
source file inside the shared isolation window; the hub is read, never
imported, so no model, database or network is reached.
"""

import ast
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_OO = _ROOT / "opti_oignon"

_DEDUP = "opti_oignon.context_dedup"


class _Snippet:
    """A retrieved candidate, carrying the fields its own source attached."""

    def __init__(self, text, score=0.0, role="assistant"):
        self.snippet = text
        self.score = score
        self.role = role


def _module():
    return isolate(targets={_DEDUP: source("context_dedup.py")})


def _text(candidate):
    return candidate.snippet


# ---------------------------------------------------------------------------
# What survives, and in what state
# ---------------------------------------------------------------------------

def test_d1_survivors_are_the_callers_own_objects_in_order():
    """Nothing is rewritten, reordered, merged, or rebuilt."""
    loaded, restore = _module()
    try:
        dedup = loaded[_DEDUP]
        first = _Snippet("the quokka census ran over three separate islands", 0.9)
        second = _Snippet("rainfall that winter was the lowest on record", 0.4)
        candidates = [first, second]

        kept, dropped = dedup.drop_duplicates(candidates, "", key=_text)

        assert kept == [first, second]
        assert kept[0] is first and kept[1] is second, (
            "a survivor must be the caller's own object, not a copy"
        )
        assert kept[0].score == 0.9 and kept[0].role == "assistant", (
            "fields the source attached must survive untouched"
        )
        assert dropped == []
        assert candidates == [first, second], "the caller's list must be left alone"
    finally:
        restore()


def test_d2_a_passage_the_prompt_already_carries_is_dropped():
    """Containment is enough; a genuinely new passage is kept."""
    loaded, restore = _module()
    try:
        dedup = loaded[_DEDUP]
        composed = (
            "Earlier context: the quokka census ran over three separate "
            "islands and the counts were reconciled by hand."
        )
        repeat = _Snippet("the quokka census ran over three separate islands")
        fresh = _Snippet("the reconciliation was later automated in a spreadsheet")

        kept, dropped = dedup.drop_duplicates([repeat, fresh], composed, key=_text)

        assert kept == [fresh]
        assert dropped == [repeat]
    finally:
        restore()


def test_d3_the_same_passage_is_recognised_across_formatting():
    """Case, punctuation, wrapping and a role prefix do not hide a repeat."""
    loaded, restore = _module()
    try:
        dedup = loaded[_DEDUP]
        composed = "the quokka census ran over three separate islands"
        reformatted = _Snippet(
            "[assistant] THE QUOKKA CENSUS -- ran over\n  three, separate islands!"
        )

        assert dedup.is_already_present(reformatted.snippet, composed)

        kept, dropped = dedup.drop_duplicates([reformatted], composed, key=_text)
        assert kept == [] and dropped == [reformatted]
    finally:
        restore()


def test_d4_near_duplicates_are_dropped_only_above_the_stated_threshold():
    """The threshold is the caller's, and below it the candidate survives."""
    loaded, restore = _module()
    try:
        dedup = loaded[_DEDUP]
        composed = (
            "the census ran over three separate islands during the dry season "
            "and the counts were reconciled by hand afterwards"
        )
        # Shares most of its wording with the composed text, then diverges.
        partial = _Snippet(
            "the census ran over three separate islands during the dry season "
            "and the results were published the following spring"
        )

        overlap_high = dedup.drop_duplicates(
            [partial], composed, key=_text, threshold=0.3,
        )
        assert overlap_high[0] == [] and overlap_high[1] == [partial], (
            "a lenient threshold must drop a heavily overlapping candidate"
        )

        overlap_low = dedup.drop_duplicates(
            [partial], composed, key=_text, threshold=0.95,
        )
        assert overlap_low[0] == [partial] and overlap_low[1] == [], (
            "a strict threshold must keep it: over-dropping loses evidence"
        )

        assert dedup.DEFAULT_OVERLAP >= 0.5, (
            "the shipped default must sit on the conservative side"
        )
    finally:
        restore()


def test_d5_survivors_accumulate_within_the_batch():
    """A repeat of an earlier survivor is dropped, not injected twice."""
    loaded, restore = _module()
    try:
        dedup = loaded[_DEDUP]
        first = _Snippet("the counts were reconciled by hand every evening")
        echo = _Snippet("The counts were reconciled by hand, every evening.")
        other = _Snippet("the weather station reported a dry spell in July")

        kept, dropped = dedup.drop_duplicates([first, echo, other], "", key=_text)

        assert kept == [first, other]
        assert dropped == [echo]
    finally:
        restore()


def test_d6_the_comparison_covers_both_composition_layouts():
    """Volatile parts count as already injected, not as unseen."""
    loaded, restore = _module()
    try:
        dedup = loaded[_DEDUP]
        head = "You are a helpful assistant."
        volatile = ["\n\nmemory: the counts were reconciled by hand every evening"]

        composed = dedup.compose_already_injected(head, volatile)
        assert head in composed and volatile[0] in composed

        repeat = _Snippet("the counts were reconciled by hand every evening")
        kept, dropped = dedup.drop_duplicates([repeat], composed, key=_text)
        assert kept == [] and dropped == [repeat], (
            "a block held back as a volatile part is still in the prompt"
        )

        # And the head alone would not have caught it.
        head_only = dedup.compose_already_injected(head, [])
        assert dedup.drop_duplicates([repeat], head_only, key=_text)[0] == [repeat]
    finally:
        restore()


def test_d7_the_decision_is_pure_and_repeatable():
    """Same inputs, same survivors -- no clock, no state, no side effect."""
    loaded, restore = _module()
    try:
        dedup = loaded[_DEDUP]
        composed = "the counts were reconciled by hand every evening"
        batch = [
            _Snippet("the counts were reconciled by hand every evening"),
            _Snippet("the weather station reported a dry spell in July"),
            _Snippet("the weather station reported a dry spell in July"),
        ]

        runs = [dedup.drop_duplicates(batch, composed, key=_text) for _ in range(5)]
        assert all(run == runs[0] for run in runs), "the decision must not drift"
        assert len(runs[0][0]) == 1 and len(runs[0][1]) == 2
        assert [c.snippet for c in batch] == [
            "the counts were reconciled by hand every evening",
            "the weather station reported a dry spell in July",
            "the weather station reported a dry spell in July",
        ], "candidates must not be mutated"
    finally:
        restore()


def test_d8_blank_is_dropped_and_a_short_new_passage_is_kept():
    """The window logic must not silently eat a short original snippet."""
    loaded, restore = _module()
    try:
        dedup = loaded[_DEDUP]
        blank = _Snippet("   \n  ")
        short = _Snippet("rain stopped")

        kept, dropped = dedup.drop_duplicates(
            [blank, short], "the counts were reconciled by hand", key=_text,
        )
        assert dropped == [blank], "an empty candidate injects nothing"
        assert kept == [short], (
            "a short passage the prompt does not carry must survive"
        )

        assert dedup.shingles(["a", "b"], 5) == set(), (
            "a passage shorter than one window yields no windows"
        )
    finally:
        restore()


def test_d10_short_passages_are_deduplicated_by_containment():
    """A passage too short for one window still has to be caught.

    Window overlap is the route for reformatted prose, but a passage
    shorter than one window produces no windows at all, so containment is
    the only route left for it. A batch that repeats a short line would
    otherwise inject it twice while the long lines beside it were being
    deduplicated correctly -- the failure would be invisible in any test
    written with sentences.
    """
    loaded, restore = _module()
    try:
        dedup = loaded[_DEDUP]
        first = _Snippet("rain stopped")
        echo = _Snippet("Rain stopped.")
        already = _Snippet("dry spell")

        kept, dropped = dedup.drop_duplicates(
            [first, echo, already], "the station logged a dry spell", key=_text,
        )

        assert kept == [first], "a short line must survive exactly once"
        assert dropped == [echo, already], (
            "a short repeat within the batch, and a short passage the prompt "
            "already carries, must both be dropped"
        )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Where the hub wires it
# ---------------------------------------------------------------------------

def test_d9_the_hub_deduplicates_before_it_injects():
    """Derived from the tree: the drop happens ahead of the injection."""
    hub = _OO / "executor.py"
    text = hub.read_text(encoding="utf-8")
    tree = ast.parse(text)

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "context_dedup" in imported, "the hub never imports the deduplicator"

    def lines_calling(name):
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if called == name:
                    found.append(node.lineno)
        return found

    retrieved = lines_calling("retrieve_from_archive")
    composed = lines_calling("compose_already_injected")
    dropped = lines_calling("drop_duplicates")
    assert retrieved, "the archive path moved; this clause is looking at nothing"
    assert composed and dropped, "the archive path injects without deduplicating"

    injection = next(
        (
            index + 1
            for index, line in enumerate(text.splitlines())
            if "--- Retrieved from conversation archive ---" in line
        ),
        None,
    )
    assert injection, "the archive injection point moved"

    assert min(retrieved) < min(composed) < injection, (
        "the corpus must be composed after retrieval and before injection"
    )
    assert min(dropped) < injection, (
        "deduplication after injection would drop nothing from the prompt"
    )

    # The corpus has to be built from BOTH layouts, or the drop is blind on
    # whichever one is active.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "compose_already_injected":
            continue
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        assert {"system_prompt"} <= names, "the composed prompt is not consulted"
        assert any("volatile" in n for n in names), (
            "the volatile parts are not consulted"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# Derived from the module, not listed beside it.
_CLAUSES = sorted(name for name in dict(globals()) if name.startswith("test_d"))


def _main() -> int:
    passed = 0
    for name in _CLAUSES:
        try:
            globals()[name]()
        except Exception:
            print(f"FAIL {name}:")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
            passed += 1
    total = len(_CLAUSES)
    print(f"{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(_main())
