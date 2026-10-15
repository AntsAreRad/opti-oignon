#!/usr/bin/env python3
"""Property contracts for the journal, under seeded generated conversations.

A contract here states one property and holds it over conversations drawn
from a generator of the standard library, seeded: the same conversations on
every run, so a red names the seed and the turn that broke it. The user asks
about the services a region of peels summarises, at times with a document;
each turn brings retrieval and web results of its own.

  * JW2 -- through turns that only append -- no eviction, pin, take-back or
    withdrawal, the system prompt, the model and the rung unchanged -- each
    window keeps the bytes before the last window's current turn: no epoch,
    its first change at or past them, never past the first byte that
    differs, every byte before it the last window's, and its chain extended
    by the digest of the bytes it adds to them; every window's pieces tile
    its bytes, each the digest of its own.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _onion_journal as oj  # noqa: E402

_SEEDS = (4, 11, 23, 47)
_NAMES = ("Alice", "Bob", "Carol")
_SERVICES = ("the gateway", "the queue", "the cache", "the cluster")

BUDGET_S = {
    "test_jw2_a_window_that_only_appends_keeps_its_prefix_and_extends_its_chain": 10.0,
}


def _question(rng):
    words = f"Did {rng.choice(_NAMES)} move {rng.choice(_SERVICES)} on day {rng.randint(1, 28)}?"
    if rng.random() < 0.3:
        return oj.with_document(words, f"{rng.choice(_NAMES)} wrote that {rng.choice(_SERVICES)} stays.")
    return oj.typed(words)


def _tail(rng):
    tail = [oj.rag(*(f"Section {i}: {rng.choice(_SERVICES)} stays." for i in range(rng.randint(0, 2))))]
    if rng.random() < 0.5:
        tail.append(oj.web(f"Result: {rng.choice(_NAMES)} moved {rng.choice(_SERVICES)}.", suffix="Sources: [1]"))
    return tuple(tail)


def test_jw2_a_window_that_only_appends_keeps_its_prefix_and_extends_its_chain():
    checked = 0
    for seed in _SEEDS:
        rng = random.Random(seed)
        loaded, restore = oj.open_window()
        try:
            lib = loaded[oj.LIBRARIAN]
            cfg = oj.config(lib)
            cid = f"c{seed}"
            stored = oj.conversation(8)
            state = lib.state_for(cid, cfg)
            state.mirror(stored)
            lib.pin(cid, "Answers cite their source.", actor="user", config=cfg)
            oj.accept(loaded, state, 2, "Alice reviewed the gateway; it stays.")
            oj.accept(loaded, state, 2, "Bob moved the queue to the old cluster.")
            oj.bare(state, 2)
            prev = None
            for step in range(12):
                question = _question(rng)
                win = oj.window(loaded, cid, question, messages=list(stored), config=cfg,
                                system=oj.SYSTEM + (oj.MANIFEST,), tail=_tail(rng))
                record = state.journal
                assert win.refusal is None, (seed, step, win.refusal)
                data = oj.canonical(win)
                if prev is not None:
                    before, last = prev
                    assert win.motive is None and win.epoch == last.epoch, (seed, step, win.motive)
                    assert data[: last.stable_end] == before[: last.stable_end], (seed, step, "the prefix moved")
                    byte = next((i for i, (a, b) in enumerate(zip(before, data)) if a != b), min(len(before), len(data)))
                    assert last.stable_end <= win.first_diff <= byte and data[:win.first_diff] == before[:win.first_diff], (
                        seed, step, win.first_diff, byte, last.stable_end)
                    added = hashlib.sha256(data[last.stable_end: record.stable_end]).digest()
                    assert win.chain == hashlib.sha256(bytes.fromhex(last.chain) + added).hexdigest(), (seed, step)
                    assert record.stable_end > last.stable_end, (seed, step, "control: the turn's pair was added")
                    checked += 1
                ends = [piece[2] for piece in record.pieces]
                assert ends[-1] == len(data) and all(
                    hashlib.sha256(data[start:end]).hexdigest() == piece[3]
                    for start, end, piece in zip([0] + ends[:-1], ends, record.pieces)), (seed, step, "the tiling")
                prev = (data, record)
                stored += [question, oj.answer(f"An answer to step {step}.")]
        finally:
            restore()
    assert checked == len(_SEEDS) * 11, checked
