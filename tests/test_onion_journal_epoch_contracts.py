#!/usr/bin/env python3
"""Contracts for the journal's epochs: why a stable prefix begins again, and the recall seam that keeps it.

A turn's window is compared with the conversation's last: while the bytes
before the last window's current turn are kept, there is no epoch; when they
are not, an epoch begins, read by why from a closed set, and anything else
is counted unexplained.

  * JW4 -- a scripted conversation: each event begins exactly one epoch with
    its motive -- the first window, a system prompt changed, a manifest
    added, a pin, an accepted proposal, an eviction, a resolved receipt, a
    withdrawal, a take-back, a model switched, a rung climbed and left, a
    region past its cap, a take-back that supersedes a receipt -- every
    motive at least once, each epoch's chain begun again from its own
    prefix; a resolved receipt moves its count by one; a turn that only
    appends begins none, nor one mirrored again under a new id with the
    same bytes; a withdrawal that reaches nothing laid leaves a take-back in
    the same turn its own motive; nothing is counted unexplained, and an
    item of the region changed by no cause the librarian reads is. A window
    is cold -- no prefix in common with the conversation's last -- when it
    is the first or its first byte changed (the system prompt), never when
    it only appends, nor when an epoch begins past its first byte.
  * JW18 -- the spans a turn recalls ride in the turn's message as data
    from their receipt, with the context and lineage of its turns: the
    region keeps the recalled span's peel at its place and the bytes before
    the turn are the window's without recall; a turn a withdrawal reached is
    not shown, said by number, and a span under a superseded receipt not at
    all; no module of the package passes a recall yet.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import ast
import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_journal as oj  # noqa: E402

BUDGET_S = {
    "test_jw4_each_event_begins_one_epoch_with_its_motive_and_an_append_begins_none": 10.0,
    "test_jw18_a_recalled_span_rides_in_the_turn_and_keeps_the_prefix": 5.0,
}

_DECISION = "We decided to keep Redis on the new cluster."
_DOCUMENT = "Service 99 runs on the new cluster since May."


def _start(loaded):
    """A conversation with a held span whose decision the queue offers, and a peel over a document turn."""
    lib = loaded[oj.LIBRARIAN]
    core_store = loaded["opti_oignon.memory.core_store"]
    cfg = oj.config(lib)
    msgs = [oj.typed(_DECISION), oj.answer("Noted: Redis stays on the new cluster."),
            oj.with_document("Please keep this in mind.", _DOCUMENT), oj.answer("Noted: service 99.")]
    msgs += oj.conversation(30, start=5)
    state = lib.state_for("c1", cfg)
    state.mirror(msgs)
    held = oj.hold(state, 2, anchors=(("t0001", 0, len(_DECISION)),))
    pid = core_store.proposal_id(held.key, "t0001", 0, len(_DECISION))
    state.proposals.append(core_store.Proposal(pid, held.key, "t0001", 0, len(_DECISION), "typed", "2026-10-15",
                                               "open"))
    document = oj.accept(loaded, state, 2, "The document says service 99 runs on the new cluster.")
    return lib, cfg, msgs, state, held, pid, document


def test_jw4_each_event_begins_one_epoch_with_its_motive_and_an_append_begins_none():
    loaded, restore = oj.open_window()
    try:
        lib, cfg, msgs, state, held, pid, document = _start(loaded)
        stored = msgs[:12]
        asked = {"system": oj.SYSTEM, "model": "model-a", "withdrawn": (), "question": "What did Alice review?"}
        seen, epochs, colds = [], [0], []

        def turn(expected):
            question = asked["question"]
            win = oj.window(loaded, "c1", question, messages=list(stored), config=cfg, system=asked["system"],
                            model=asked["model"], withdrawn=asked["withdrawn"])
            assert win.refusal is None, (expected, win.refusal)
            assert win.motive == expected, (expected, win.motive)
            assert win.epoch == epochs[-1] + (expected is not None), (expected, win.epoch, epochs[-1])
            if expected is not None:
                assert win.chain == hashlib.sha256(
                    hashlib.sha256(oj.canonical(win)[: state.journal.stable_end]).digest()).hexdigest(), (
                    "an epoch begins its chain again from its own prefix", expected)
            epochs.append(win.epoch)
            seen.append(win.motive)
            colds.append((win.motive, win.cold, win.first_diff))
            stored.extend([oj.typed(question), oj.answer(f"An answer to: {question}")])
            return win

        turn("first")
        turn(None)
        asked["system"] = ("You are a terse assistant.",)
        turn("system")
        asked["system"] = ("You are a terse assistant.", oj.MANIFEST)
        turn("system")
        turn(None)
        lib.pin("c1", "Answers cite their source.", actor="user", config=cfg)
        turn("core")
        lib.accept_proposal("c1", pid, actor="user", config=cfg)
        turn("proposal")
        oj.accept(loaded, state, 2, "Alice reviewed service 5; it stays.")
        turn("eviction")
        resolved = dict(state.journal.causes)["resolved"]
        lib.resolve_receipt("c1", held.key, actor="user", config=cfg)
        turn("receipt")
        assert dict(state.journal.causes)["resolved"] == resolved + 1, "a resolved receipt moves its count by one"
        turn(None)
        # The region past its cap before any withdrawal: once a source is withdrawn, a peel over an answer, whose
        # lineage was never recorded, is laid as a note, and a note does not fill the region.
        long = " ".join(["Alice reviewed the service and it stays on the new cluster."] * 40)
        while True:
            oj.accept(loaded, state, 2, long + f" Span {len(state.tree.all())}.")
            # The region the next window lays, read by a plan that composes nothing.
            probe = oj.plan(lib, "c1", asked["question"], wrap=loaded[oj.WRAPPER].wrap, messages=list(stored),
                            config=cfg, system=asked["system"], model=asked["model"], withdrawn=asked["withdrawn"])
            if loaded[oj.COMPOSER].region_cut(probe._held.region, probe._held.budget, oj.count) > 0:
                break
            turn("eviction")
        turn("peels_full")
        turn(None)
        lineage = [entry for turn_ in state.cellar.get(document.key)
                   for entry in loaded[oj.PEELS].turn_label(turn_, state.lineage)[1] if entry.startswith("document:")]
        assert lineage, "control: the span holds a document's turn"
        asked["withdrawn"] = tuple(lineage[:1])
        turn("withdrawn")
        # The last pair the window mirrored, asked and answered again; the pair after it was never mirrored.
        stored[-4:-2] = [oj.typed("Asked again: what did Alice review?"), oj.answer("Answered again.")]
        turn("take_back")
        asked["model"] = "model-b"
        turn("model")
        asked["question"] = oj.with_document("Read this.", " ".join(["cluster"] * 400))
        turn("rung")
        asked["question"] = "And service 9?"
        turn("rung")
        # A turn under a receipt asked again: the receipt superseded, the region it laid gone.
        stored[4] = oj.typed("Turn 5, asked again: did Alice review service 5?")
        turn("take_back")
        turn(None)
        # A withdrawal that reaches nothing laid, with a take-back in the same turn: the take-back's motive.
        asked["withdrawn"] = asked["withdrawn"] + ("document:" + "0" * 64,)
        stored[-4:-2] = [oj.typed("Asked once more: what did Alice review?"), oj.answer("Answered once more.")]
        turn("take_back")
        # A turn the last window laid, mirrored again under a new id with the same bytes -- a withdrawal that
        # reached only its context -- keeps the prefix: no epoch, the first change past the last window's prefix.
        prefix, mirrored = state.journal.stable_end, [t["turn_id"] for t in state.flesh.turns()]
        stored[-3] = dict(stored[-3], context=["withdrawn"])
        assert turn(None).first_diff >= prefix, "the same bytes keep the prefix"
        assert [t["turn_id"] for t in state.flesh.turns()][:len(mirrored) - 1] == mirrored[:-1] and (
            [t["turn_id"] for t in state.flesh.turns()][len(mirrored) - 1] != mirrored[-1]), (
            "control: the answer was mirrored again under a new id")
        motives = {motive for motive in seen if motive is not None}
        assert motives == set(loaded[oj.LIBRARIAN].EPOCH_MOTIVES), sorted(motives)
        assert "unexplained" not in lib.counters().get("journal", {}), lib.counters().get("journal")
        # The witness: an item of the region changed behind the librarian's back, by no cause it reads, begins an
        # epoch counted unexplained -- the count of none above can say otherwise.
        other = lib.state_for("c2", cfg)
        story = oj.conversation(6)
        other.mirror(story)
        oj.accept(loaded, other, 2, "Alice reviewed service 1; it stays.")
        oj.window(loaded, "c2", "What did Alice review?", messages=story, config=cfg)
        peel = other.tree.all()[0]
        other.tree.add(replace(peel, text="Alice reviewed service 1; it moved."))
        odd = oj.window(loaded, "c2", "What did Alice review?", messages=story, config=cfg)
        assert odd.motive == "unexplained" and lib.counters()["journal"].get("unexplained") == 1, (
            odd.motive, lib.counters()["journal"])
        assert all(cold == (first_diff == 0) for _motive, cold, first_diff in colds), colds
        assert [motive for motive, cold, _diff in colds if cold] == ["first", "system"], colds
        assert sum(1 for motive, _cold, _diff in colds if motive == "system") == 2, "control: a warm system epoch"
        assert not any(cold for motive, cold, _diff in colds if motive is None), "an append is never cold"
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW18 -- the recall seam
# ---------------------------------------------------------------------------
def _modules(package):
    """The package's modules, its data directories pruned, never entered."""
    for root, dirs, files in os.walk(package):
        dirs[:] = sorted(name for name in dirs if name not in ("data", "__pycache__"))
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(root) / name


def _passes_recalled(path):
    """The calls of a module that pass ``recalled``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
            and any(keyword.arg == "recalled" for keyword in node.keywords)]


def test_jw18_a_recalled_span_rides_in_the_turn_and_keeps_the_prefix():
    loaded, restore = oj.open_window()
    try:
        lib = loaded[oj.LIBRARIAN]
        cfg = oj.config(lib)
        msgs = oj.conversation(10)
        state = lib.state_for("c1", cfg)
        state.mirror(msgs)
        lib.pin("c1", "Answers cite their source.", actor="user", config=cfg)
        first = oj.accept(loaded, state, 2, "Alice reviewed service 1; it stays.")
        oj.accept(loaded, state, 2, "Alice reviewed service 3; it stays.")
        plain = oj.window(loaded, "c1", "What did Alice say?", messages=msgs, config=cfg)
        before = state.journal
        recalled = oj.window(loaded, "c1", "What did Alice say?", messages=msgs, config=cfg, recalled=(first.key,))
        after = state.journal
        assert plain.refusal is None and recalled.refusal is None
        assert recalled.motive is None and recalled.first_diff >= before.stable_end, (recalled.motive,
                                                                                      recalled.first_diff)
        assert after.stable_end == before.stable_end and (
            [piece for piece in after.pieces if piece[2] <= after.stable_end]
            == [piece for piece in before.pieces if piece[2] <= before.stable_end]), (
            "the bytes before the turn are the window's without recall")
        peel = next(p for p in state.tree.all() if first.key in p.sources)
        m2 = recalled.messages[1][1]
        assert f"provenance=peel:{peel.id[:12]}" in m2, "the recalled span's peel stays in the region"
        turn = recalled.messages[-1][1]
        assert f"[data layer=recalled provenance=receipt:{first.key[:12]}]" in turn, turn[-400:]
        assert "Turn 1: did Alice review service 1" in turn and "Turn 1:" not in plain.messages[-1][1]
        # A span with a document: recalled with both halves of its label; a withdrawal that reached its turns
        # shows none of them, said by number; under a superseded receipt, nothing.
        story = [oj.with_document("Please keep this.", "Service 77 moves to the old cluster."),
                 oj.answer("Noted: service 77.")] + oj.conversation(6)
        state2 = lib.state_for("c2", cfg)
        state2.mirror(story)
        under = oj.accept(loaded, state2, 2, "The document says service 77 moves.")
        source = next(entry for t in state2.cellar.get(under.key)
                      for entry in loaded[oj.PEELS].turn_label(t, state2.lineage)[1] if entry.startswith("document:"))
        shown = oj.window(loaded, "c2", "Where does service 77 move?", messages=story, config=cfg,
                          recalled=(under.key,))
        assert "Service 77 moves to the old cluster." in shown.messages[-1][1], "control: recalled, the span is laid"
        context, lineage = shown.messages[-1][2]
        assert "document" in context and source in lineage, ("both halves of its label", shown.messages[-1][2])
        reached = oj.window(loaded, "c2", "Where does service 77 move?", messages=story, config=cfg,
                            recalled=(under.key,), withdrawn=(source,))
        assert "Service 77 moves to the old cluster." not in reached.messages[-1][1], "a withdrawal reached it"
        assert "of its turns not shown: a withdrawal reached them" in reached.messages[-1][1], (
            reached.messages[-1][1][-300:])
        story[0] = oj.with_document("Please keep this, said again.", "Service 77 moves to the old cluster.")
        retracted = oj.window(loaded, "c2", "Where does service 77 move?", messages=story, config=cfg,
                              recalled=(under.key,))
        assert under.key in {r.key for r in state2.ledger.all() if r.kind == "superseded"}, "control: superseded"
        assert f"provenance=receipt:{under.key[:12]}" not in retracted.messages[-1][1], (
            "a span under a superseded receipt is recalled nowhere")
        package = oj.REPO / "opti_oignon"
        passing = {str(path.relative_to(package)): _passes_recalled(path) for path in _modules(package)
                   if path.name != "librarian.py"}
        assert not any(passing.values()), ("no module passes a recall yet", {k: v for k, v in passing.items() if v})
        assert _passes_recalled(oj.REPO / "opti_oignon" / "memory" / "librarian.py"), (
            "control: the reader finds the librarian's own pass")
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
