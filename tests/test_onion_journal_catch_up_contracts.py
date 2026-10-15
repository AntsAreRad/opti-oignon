#!/usr/bin/env python3
"""Contracts for the journal's hold on the conversation: what a window lays is what the store holds now.

A turn's plan mirrors the conversation as the store holds it before it reads
anything, so a window never lays a turn the conversation no longer holds, nor
leaves out one it holds.

  * JW10 -- the Flesh a window lays is the stored conversation's turns, in
    order: after a retry took the last pair back and wrote another, after
    two turns were saved with no burst between them, and with the turns
    handed over as an iterator, read once; after a roll back to an earlier
    point whose question is asked again in its own words, the window a
    prefix of the last one's, a take-back; after the first turn was
    retried, when the conversation holds nothing, the window lays nothing
    of it but the Core -- no Flesh, nor what an evicted pair left -- and is
    a take-back.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_journal as oj  # noqa: E402

BUDGET_S = {
    "test_jw10_a_window_lays_the_conversation_the_store_holds_now": 5.0,
}


def _flesh(win):
    """The contents a window lays between its system prompt and the current turn's message."""
    return [content for _role, content, _label in win.messages[1:-1]]


def test_jw10_a_window_lays_the_conversation_the_store_holds_now():
    loaded, restore = oj.open_window()
    try:
        lib = loaded[oj.LIBRARIAN]
        cfg = oj.config(lib)
        stored = oj.conversation(8)
        first = oj.window(loaded, "c1", "What did Alice review?", messages=stored, config=cfg)
        assert first.refusal is None and _flesh(first) == [m["content"] for m in stored], "control: as stored"
        retried = stored[:6] + [oj.typed("Turn 7, asked again: did Alice review service 7?"),
                                oj.answer("Turn 8, answered again: she did.")]
        again = oj.window(loaded, "c1", "What did Alice review?", messages=retried, config=cfg)
        assert again.refusal is None and _flesh(again) == [m["content"] for m in retried], (
            "a retry's pair, taken back and written again", _flesh(again)[-2:])
        assert "Turn 8: yes" not in "\n".join(_flesh(again)), "the pair the retry took back is laid nowhere"
        saved = retried + [oj.typed("Turn 9: did Alice review service 9?"), oj.answer("Turn 10: she did.")]
        grown = oj.window(loaded, "c1", "What did Alice review?", messages=saved, config=cfg)
        assert grown.refusal is None and _flesh(grown) == [m["content"] for m in saved], (
            "two turns saved with no burst between", _flesh(grown)[-2:])
        assert grown.motive is None, ("control: turns appended keep the prefix", grown.motive)
        # The first turn retried: the conversation holds nothing, so no Flesh, whatever the state still mirrors.
        oj.window(loaded, "c2", "Did Alice review service 1?", messages=[], config=cfg)
        state = lib.state_for("c2", cfg)
        oj.window(loaded, "c2", "And service 2?", config=cfg,
                  messages=[oj.typed("Did Alice review service 1?"), oj.answer("She did.")])
        assert len(state.flesh.turns()) == 2, "control: the state mirrors the first pair"
        retried = oj.window(loaded, "c2", "Did Alice review service 1?", messages=[], config=cfg)
        assert retried.refusal is None and _flesh(retried) == [], ("a first turn retried lays no Flesh",
                                                                   _flesh(retried))
        # The only pair evicted, then the first turn retried: nothing its turns left is laid, and it is a take-back.
        oj.window(loaded, "c3", "Did Alice review service 1?", messages=[], config=cfg)
        evicted = lib.state_for("c3", cfg)
        pair = [oj.typed("Did Alice review service 1?"), oj.answer("She did.")]
        oj.window(loaded, "c3", "And service 2?", messages=pair, config=cfg)
        oj.accept(loaded, evicted, 2, "Alice reviewed service 1; it stays.")
        laid = oj.window(loaded, "c3", "And service 2?", messages=pair, config=cfg)
        assert "Alice reviewed service 1; it stays." in laid.messages[1][1], "control: the region holds the pair's peel"
        again = oj.window(loaded, "c3", "Did Alice review service 1?", messages=[], config=cfg)
        assert [(role, content) for role, content, _label in again.messages[1:]] == [
            ("user", "Did Alice review service 1?")], ("nothing of the conversation", again.messages[1:])
        assert again.motive == "take_back", again.motive
        # Rolled back to an earlier point, its question asked again in its own words: the window is a prefix of the
        # last one's, laid as the store holds the conversation, and a take-back.
        oj.window(loaded, "c5", "Did Alice review service 9?", messages=stored, config=cfg)
        back = oj.window(loaded, "c5", stored[6]["content"], messages=stored[:6], config=cfg)
        assert back.refusal is None and _flesh(back) == [m["content"] for m in stored[:6]], _flesh(back)[-2:]
        assert back.motive == "take_back", back.motive
        # The turns handed over as an iterator: read once, whole.
        streamed = oj.window(loaded, "c4", "What did Alice review?", messages=iter(stored), config=cfg)
        assert streamed.refusal is None and _flesh(streamed) == [m["content"] for m in stored], (
            "an iterator of turns is read whole", _flesh(streamed)[:2])
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
