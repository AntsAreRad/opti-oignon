#!/usr/bin/env python3
"""Contracts for the room the memory a question brings takes in a turn's window.

The tail of a window holds each kind of source a turn brings -- the memory,
the retrieval, the web results, a tool turn's earlier calls -- to its share
of the tail's cap, its envelope included. At 4096 the memory's share, 0.35
of a tail of 192 tokens, did not hold its envelope alone (174 tokens by the
chat's counter for a Qwen model): every memory item was cut, and the turn
was answered from no memory. Measured on the drift A/B at 4096, a 27B model
invented every answer. The plan also chose the memory by its bare words, and
the composer cut silently what passed the cap once envelope and frames were
counted.

  * MR1 -- THE MEMORY'S CAPS: its own share of the tail's cap, and the whole
    it may take: with the shares of the kinds after it the turn brings
    nothing of and the room the region of accepted peels leaves of the
    peels' cap; none of either with no share or no tail.
  * MR2 -- AT 4096 THE MEMORY A QUESTION REACHES IS LAID: a held fact the
    question reaches is in the window's tail, nothing cut, a retrieval the
    turn brings beside it or not.
  * MR3 -- THE PLAN HANDS THE COMPOSER WHAT IT LAYS: the plan keeps of the
    memory it chose, a recalled span with it, what the composer's own loop
    lays (``fit_kind``), envelope and frames counted as it counts them, so
    the composer lays all it is handed -- under a counter that is not
    additive too -- and the window counts what the plan kept back as cut;
    it chooses on the shares of the kinds the turn brings nothing of, and
    not on the share of one it brings.
  * MR4 -- THE ROOM MOVES, IT IS NOT ADDED: the peels, the tail and the
    parts a window lays whole together never pass their caps; a region that
    fills the peels' cap, beside parts laid whole that fill theirs, leaves
    the memory its shares alone, which at 4096 do not hold its envelope.
  * MR5 -- THE COMPOSER LENDS AS THE CAP SAYS: with no room from the region,
    the memory takes the shares of the kinds the turn brings nothing of --
    fourteen items laid where its own share holds four -- and not the share
    of a kind the turn brings; what it borrowed is said (``lent``); and from
    what its items want (``memory_wants``: the heaviest of the loop's tries,
    under a count an item can lower) it lays all of them, below it not.
  * MR6 -- THE BORROWED ROOM GIVES WAY FIRST: a window at its prompt's cap
    keeps its Flesh whole, sets nothing aside over a shadow, and the memory
    takes what the window leaves; turns with memory and without keep the
    window's stable prefix.
  * MR7 -- UNDER A TAIL'S ROOM THE MEMORY BORROWS NOTHING: a window cut
    to a tail's room gave the borrowed room up first; the memory holds to
    its own share whatever ``lend`` says, the kinds after it as before.
  * MR8 -- A BURST IS NOT AIMED ON BORROWED ROOM: a tool turn's record aims
    the next burst at the Flesh room the window leaves as if the memory had
    borrowed nothing.
  * MR9 -- THE FLESH GIVES WHAT THE WINDOW PASSES ITS CAP BY WITH NOTHING
    BORROWED, NO MORE: where a window passes its cap even so, it is the
    window a memory that borrows nothing lays.

The onion's modules and the wrapper come from source through the shared
isolation window (``_onion_journal``); the counter is the journal suite's,
two tokens a word, under which the envelope costs what a Qwen model's does.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _onion_journal as oj  # noqa: E402

BUDGET_S = {
    "test_mr1_the_memorys_caps_take_the_shares_and_the_room_nothing_else_claims": 5.0,
    "test_mr2_at_4096_the_memory_a_question_reaches_is_laid": 5.0,
    "test_mr3_the_plan_hands_the_composer_what_it_lays": 5.0,
    "test_mr4_the_room_moves_to_the_tail_and_is_never_added": 5.0,
    "test_mr5_the_composer_lends_the_shares_the_cap_says": 5.0,
    "test_mr6_the_borrowed_room_gives_way_before_the_flesh": 10.0,
    "test_mr7_under_a_tails_room_the_memory_borrows_nothing": 5.0,
    "test_mr8_a_burst_is_not_aimed_on_borrowed_room": 10.0,
    "test_mr9_the_flesh_gives_what_the_window_passes_its_cap_by_with_nothing_borrowed": 10.0,
}

_FACT = "Alice moved the build to Lisbon on 2026-03-04."


def _held_facts(lib, cid, facts, cfg, after=6):
    """A conversation whose first spans, each a typed fact and an answer, were held, each fact its anchor."""
    messages = []
    for fact in facts:
        messages += [oj.typed(fact), oj.answer("Noted.")]
    messages += oj.conversation(after, start=len(messages) + 1)
    state = lib.state_for(cid, cfg)
    state.mirror(messages)
    for fact in facts:
        oj.hold(state, 2, anchors=((state.flesh.turns()[0]["turn_id"], 0, len(fact)),))
    return state, messages


def _code_word(text):
    """A counter that is not additive: the journal suite's, the whole raised a fifth when it holds a code word."""
    base = oj.count(text)
    return base + base // 5 if "package " in text else base


# ---------------------------------------------------------------------------
# MR1 -- the memory's caps
# ---------------------------------------------------------------------------
def test_mr1_the_memorys_caps_take_the_shares_and_the_room_nothing_else_claims():
    """MR1: own share; whole with the shares of kinds absent and the region's spare room; none with no share."""
    loaded, restore = oj.open_window()
    composer = loaded[oj.COMPOSER]
    try:
        budget = composer.load_budgets().at(4096)
        shares = composer.load_journal().shares()
        assert (budget.tail, budget.peels) == (192, 256), ("control: the shipped table at 4096", budget)
        spare = budget.peels - 100
        assert composer.memory_cap(budget, shares, set(), 100) == (67, int(0.8 * 192) + spare), (
            "its own share; the retrieval's, the web's and the tool's shares, and the room the region leaves")
        assert composer.memory_cap(budget, shares, {"rag"}, 100) == (67, int(0.55 * 192) + spare), (
            "a kind the turn brings keeps its share")
        assert composer.memory_cap(budget, shares, {"rag", "web", "tool"}, budget.peels) == (67, 67), (
            "every kind brought and the region full: its own share alone")
        assert composer.memory_cap(budget, shares, set(), budget.peels + 50) == (67, int(0.8 * 192)), (
            "a region past the peels' cap leaves no room, never a negative one")
        assert composer.memory_cap(budget, dict(shares, memory=0.0), set(), 0) == (0, 0), "no share keeps it out"
        assert composer.memory_cap(replace(budget, tail=0), shares, set(), 0) == (0, 0), "no tail keeps it out"
        region = (("p1", "Alice reviewed the gateway.", "peel:1", None), ("p2", "Bob reviewed the queue.", "peel:2", None))
        assert composer.memory_room(budget, shares, set(), region, oj.count) == composer.memory_cap(
            budget, shares, set(), oj.count("Alice reviewed the gateway.") + oj.count("Bob reviewed the queue.")), (
            "the region counted as the journal lays it")
        long = " ".join(["Alice reviewed the gateway."] * 12)
        crowded = tuple((f"p{n}", long, f"peel:{n}", None) for n in range(3))
        each = oj.count(long)
        assert 2 * each <= budget.peels < 3 * each and composer.region_cut(crowded, budget, oj.count) == 1, (
            "control: the oldest peel falls out of the peels' cap", each)
        assert composer.memory_room(budget, shares, set(), crowded, oj.count) == composer.memory_cap(
            budget, shares, set(), 2 * each), "the region counted as laid, its oldest peel cut out"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MR2 -- at 4096 the memory a question reaches is laid
# ---------------------------------------------------------------------------
def test_mr2_at_4096_the_memory_a_question_reaches_is_laid():
    """MR2: a held fact the question reaches is laid at 4096, nothing cut, a retrieval the turn brings beside it."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        wrap = loaded[oj.WRAPPER].wrap
        envelope = oj.count(wrap(".", source="memory", frames=True)) - oj.count(".")
        assert envelope > int(0.35 * 192), ("control: the memory's own share does not hold its envelope", envelope)
        _state, messages = _held_facts(lib, "c1", [_FACT], cfg)
        win = oj.window(loaded, "c1", "Where did Alice move the build?", messages=messages, config=cfg,
                        model_window=4096)
        assert win.refusal is None and win.rung == 4096, (win.refusal, win.rung)
        assert _FACT in win.messages[-1][1], ("the fact the question reaches, in the tail", win.messages[-1][1][-400:])
        assert dict(win.cut)["memory"] == 0, ("nothing cut", win.cut)
        brought = oj.window(loaded, "c1", "Where did Alice move the build?", messages=messages, config=cfg,
                            model_window=4096, tail=(oj.rag("Section A: the build farm runs in Lisbon."),))
        assert brought.refusal is None and _FACT in brought.messages[-1][1], ("the memory is laid beside it", brought)
        assert dict(brought.cut)["memory"] == 0, ("nothing of it cut", brought.cut)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MR3 -- the plan hands the composer what it lays
# ---------------------------------------------------------------------------
def test_mr3_the_plan_hands_the_composer_what_it_lays():
    """MR3: many anchors reached, a counter that is not additive, a span recalled: all the plan hands is laid and
    what it kept back is counted as cut, at 4096 and at 8192; with no room from the region it chooses on the shares
    the turn leaves, a retrieval's share left to it."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        wrap = loaded[oj.WRAPPER].wrap
        items = [("memory", f"m{n}", f"fact {n} of the package build", ("peels", f"anchor:{n}"), None) for n in range(3)]
        taken, _kept, placed, dropped = composer.fit_kind("memory", items, 10 ** 6, wrap, _code_word)
        assert taken == [0, 1, 2] and dropped == 0 and all(item[2] in placed for item in items), (
            "control: the loop lays what fits, in its order")
        exact = _code_word(composer.fit_kind("memory", items[:1], 10 ** 6, wrap, _code_word)[2])
        assert composer.fit_kind("memory", items[:1], exact, wrap, _code_word)[0] == [0], "laid at its cap"
        assert composer.fit_kind("memory", items[:1], exact - 1, wrap, _code_word)[0] == [], "never a token past it"
        sources = "Sources: [1] builds.example.org"
        cited = composer.fit_kind("web", items[:1], 10 ** 6, wrap, oj.count, citation=sources)[2]
        assert cited.endswith(sources), ("the citation closes the kind's envelope", cited[-80:])
        assert composer.fit_kind("web", items[:1], oj.count(cited) - 1, wrap, oj.count, citation=sources)[0] == [], (
            "the citation counted against the cap")
        facts = [f"Alice moved build {n} to Lisbon on 2026-03-{n % 28 + 1:02d}." for n in range(1, 101)]
        facts[-1] = facts[-1][:-1] + " with the package of the release."
        state, messages = _held_facts(lib, "c1", facts, cfg)
        recalled = oj.bare(state, 2)

        def composed(window, **extra):
            planned = oj.plan(lib, "c1", "Where did Alice move the builds to Lisbon?", wrap=wrap, messages=messages,
                              config=cfg, model_window=window, **extra)
            return planned._held, lib.compose_window(planned)

        for window in (4096, 8192):
            held, win = composed(window, counter=_code_word, recalled=(recalled.key,))
            assert win.refusal is None and win.rung == window, (window, win.refusal)
            handed = [item[2] for item in held.tail if item[0] == "memory"]
            assert handed and all(text in win.messages[-1][1] for text in handed), ("all it hands is laid", window)
            assert window > 4096 or "layer=recalled" not in win.messages[-1][1], (
                "control: at 4096 the recalled span, last, kept back")
            assert dict(win.cut)["memory"] == held.trimmed, (
                "what the plan chose and kept back, the recalled span with it, counted as cut, no more", window,
                win.cut, held.trimmed)
            assert len(handed) < len(facts) and (window > 4096 or held.trimmed > 0), (
                "control: the caps bind, the plan keeping back at 4096", window, len(handed), held.trimmed)
        # The parts laid whole fill their caps (``_onion_journal.snug``): the caps bind as written.
        shipped = oj.snug(loaded, "Where did Alice move the builds to Lisbon?")
        at = shipped.at(8192)
        bare = composer.Budgets(reserve=shipped.reserve, home=8192, tables=(
            (8192, replace(at, peels=0, flesh=at.flesh + at.peels)),))
        own = int(composer.load_journal().shares()["memory"] * at.tail)
        laid = {}
        for name, tail in (("alone", ()), ("rag", (oj.rag("Section A: the build farm runs in Lisbon."),))):
            held, win = composed(8192, budgets=bare, tail=tail)
            assert win.refusal is None and dict(win.cut)["memory"] == held.trimmed, (name, win.refusal, win.cut)
            laid[name] = sum(1 for fact in facts if fact in win.messages[-1][1])
            assert name == "alone" or "Section A:" in win.messages[-1][1], "the retrieval laid at its share"
            assert name == "rag" or dict(win.layers)["tail"] > own, (
                "no room from the region: the plan chooses on the shares the turn leaves, past its own", win.layers)
        assert 0 < laid["rag"] < laid["alone"], ("the retrieval's share is not the memory's", laid)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MR4 -- the room moves to the tail and is never added
# ---------------------------------------------------------------------------
def test_mr4_the_room_moves_to_the_tail_and_is_never_added():
    """MR4: the peels, the tail and the parts laid whole hold within their caps; a full region and full parts leave
    the memory its shares."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        budget = composer.load_budgets().at(4096)
        pool = [f"Alice moved build {n} to Lisbon on 2026-04-{n % 28 + 1:02d}." for n in range(1, 61)]
        _state, messages = _held_facts(lib, "c1", [_FACT] + pool, cfg)
        win = oj.window(loaded, "c1", "Where did Alice move the build?", messages=messages, config=cfg,
                        model_window=4096)
        layers = dict(win.layers)
        assert layers["tail"] > budget.tail, ("control: the tail took room the region left", layers)
        whole = ("system", "core", "turn")
        assert layers["peels"] + layers["tail"] + sum(layers[name] for name in whole) <= (
            budget.peels + budget.tail + sum(getattr(budget, name) for name in whole)), ("moved, never added", layers)
        full = lib.state_for("c2", cfg)
        messages = [oj.typed(_FACT), oj.answer("Noted.")] + oj.conversation(12, start=3)
        full.mirror(messages)
        oj.hold(full, 2, anchors=((full.flesh.turns()[0]["turn_id"], 0, len(_FACT)),))
        for n in range(2):
            oj.accept(loaded, full, 2, " ".join([f"Alice reviewed service {n} and it stays on the new cluster."] * 4))
        region = dict(oj.window(loaded, "c2", "Where did Alice move the build?", messages=messages, config=cfg,
                                model_window=4096).layers)["peels"]
        shipped = composer.load_budgets()
        exact = oj.snug(loaded, "Where did Alice move the build?", budgets=composer.Budgets(
            reserve=shipped.reserve, home=4096,
            tables=((4096, replace(budget, peels=region, flesh=budget.flesh + budget.peels - region)),)))
        crowded = oj.window(loaded, "c2", "Where did Alice move the build?", messages=messages, config=cfg,
                            model_window=4096, budgets=exact)
        assert crowded.refusal is None and dict(crowded.layers)["peels"] == region, (
            "control: the region fills the peels' cap", crowded.refusal, crowded.layers)
        assert _FACT not in crowded.messages[-1][1] and dict(crowded.layers)["tail"] == 0, (
            "no room moved: the memory's shares alone, which do not hold its envelope", crowded.layers)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MR5 -- the composer lends the shares the cap says
# ---------------------------------------------------------------------------
def test_mr5_the_composer_lends_the_shares_the_cap_says():
    """MR5: no region room: fourteen items laid on the lent shares, its own holding four; a brought kind's kept."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        wrap = loaded[oj.WRAPPER].wrap
        held = oj.plan(lib, None, "Where did Alice move the builds?", wrap=wrap)._held
        # The parts laid whole fill their caps (``_onion_journal.snug``): the shares bind as written.
        budget = replace(oj.snug(loaded, "Where did Alice move the builds?").at(8192), peels=0)
        shares = held.shares
        items = tuple(("memory", f"m{n}", f"Alice moved build {n} to Lisbon on 2026-03-{n:02d}, said the user.",
                       ("peels", f"anchor:{n:012d}:t{n:04d}"), (("typed",), ())) for n in range(1, 15))

        def laid(tail, **extra):
            journal = composer.compose_journal(system=oj.SYSTEM, pinned=held.pinned, region=(), digest="",
                                               flesh=(), tail=tail, turn=held.turn, budget=budget, count=oj.count,
                                               overhead=held.overhead, wrap=wrap, shares=shares, **extra)
            return sum(1 for item in items if item[2] in journal.messages[-1].content), dict(journal.cut), journal

        own = int(shares["memory"] * budget.tail)
        envelope = oj.count(wrap(".", source="memory", frames=True)) - oj.count(".")
        each = oj.count(items[0][2] + " [data layer=peels provenance=anchor:x:t0001] [/data]")
        assert (own - envelope) // each <= 4, ("control: the memory's own share holds four items at most", own)
        count, cut, journal = laid(items)
        assert count == 14 and cut["memory"] == 0, ("fourteen laid on the shares the turn leaves", count, cut)
        assert journal.lent > 0, ("what it borrowed past its own share is said", journal.lent)
        alone, _cut, held_back = laid(items, lend=0)
        assert alone <= 4 and held_back.lent == 0, ("lend 0: its own share alone", alone, held_back.lent)
        rag = (("rag", "rag:0", "Section A: the build farm runs in Lisbon.", None, (("file",), ("file:a",))),)
        brought, cut, _journal = laid(items + rag)
        assert 4 < brought < 14 and cut["memory"] == 14 - brought, ("the retrieval's share is not lent", brought, cut)

        def flip(text):
            """A count an item can lower: raised a fifth by a code word, unless a query line comes with it."""
            base = oj.count(text)
            return base + base // 5 if "package " in text and "SELECT " not in text else base

        code = (("memory", "c1", "The build moved in package main to Lisbon.", ("peels", "anchor:c1"), (("typed",), ())),
                ("memory", "c2", "SELECT id FROM builds WHERE city = 'Lisbon';", ("peels", "anchor:c2"),
                 (("typed",), ())))
        trials = []
        composer.fit_kind("memory", code, 10 ** 6, wrap, flip, trials=trials)
        assert trials[0] > trials[1], ("control: the count falls as the query line is added", trials)
        flipped = dict(shares, memory=(trials[1] - 5) / budget.tail)
        own = int(flipped["memory"] * budget.tail)
        wants = composer.memory_wants(code, own, wrap, flip)
        assert wants == trials[0] - own, ("the heaviest try, not the last", wants, trials, own)

        def code_laid(lend):
            return composer.compose_journal(system=oj.SYSTEM, pinned=held.pinned, region=(), digest="", flesh=(),
                                            tail=code, turn=held.turn, budget=budget, count=flip,
                                            overhead=held.overhead, wrap=wrap, shares=flipped, lend=lend).messages

        assert code_laid(wants) == code_laid(None) != code_laid(wants - 1), "from what it wants on, all; below, not"
    finally:
        restore()


# ---------------------------------------------------------------------------
# MR6 -- the borrowed room gives way before the Flesh
# ---------------------------------------------------------------------------
def test_mr6_the_borrowed_room_gives_way_before_the_flesh():
    """MR6: at the prompt's cap the Flesh stays whole, no shadow, the memory takes what is left; no epoch churns."""
    loaded, restore = oj.open_window()
    lib = loaded[oj.LIBRARIAN]
    try:
        cfg = oj.config(lib)
        facts = [f"Alice moved build {n} to Lisbon on 2026-03-{n:02d}." for n in range(1, 9)]
        _state, messages = _held_facts(lib, "c1", facts, cfg, after=30)
        lib.pin("c1", " ".join(["alpha"] * 80), actor="user", config=cfg)
        system = (" ".join(["rule"] * 230),)
        long_question = "Where did Alice move the builds to Lisbon? " + " ".join(["please"] * 110)
        win = oj.window(loaded, "c1", long_question, messages=messages, config=cfg, model_window=4096, system=system)
        assert win.refusal is None, win.refusal
        layers = dict(win.layers)
        assert layers["flesh"] >= 500 and "shadow" not in lib.counters(), ("the Flesh whole, nothing set aside", layers)
        assert "shadow:receipts" not in win.messages[-1][1], "no fold"
        assert layers["tail"] > 0 and any(fact in win.messages[-1][1] for fact in facts), (
            "the memory takes what the window leaves", layers)
        assert win.tokens <= 2048, ("within the prompt's cap", win.tokens)
        for question in ("What is the weather like?", "Where did Alice move the builds to Lisbon?",
                         "What is the weather like?"):
            again = oj.window(loaded, "c1", question, messages=messages, config=cfg, model_window=4096, system=system)
            assert again.refusal is None and again.motive is None, ("the stable prefix kept", question, again.motive)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MR7 -- under a tail's room the memory borrows nothing
# ---------------------------------------------------------------------------
def test_mr7_under_a_tails_room_the_memory_borrows_nothing():
    """MR7: a tail's room: the memory holds to its own share whatever ``lend`` says, the retrieval after it."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        wrap = loaded[oj.WRAPPER].wrap
        held = oj.plan(lib, None, "Where did Alice move the builds?", wrap=wrap)._held
        budget = composer.load_budgets().at(8192)
        items = tuple(("memory", f"m{n}", f"Alice moved build {n} to Lisbon on 2026-03-{n:02d}, said the user.",
                       ("peels", f"anchor:{n:012d}:t{n:04d}"), (("typed",), ())) for n in range(1, 25))
        rag = (("rag", "rag:0", "Section A: the build farm runs in Lisbon.", None, (("file",), ("file:a",))),)

        def laid(room, lend=None):
            journal = composer.compose_journal(system=oj.SYSTEM, pinned=held.pinned, region=(), digest="",
                                               flesh=(), tail=items + rag, turn=held.turn, budget=budget,
                                               count=oj.count, overhead=held.overhead, wrap=wrap,
                                               shares=held.shares, tail_room=room, lend=lend)
            content = journal.messages[-1].content
            return sum(1 for item in items if item[2] in content), "Section A:" in content, journal

        free, rag_free, journal = laid(None)
        assert rag_free and free > 4 and journal.lent > 0, ("control: with no tail's room it borrows", free)
        alone = laid(700, 0)[2]
        for lend in (None, 0, 10 ** 6):
            under, rag_under, journal = laid(700, lend)
            assert under <= 4 and journal.lent == 0, ("its own share alone", lend, under, journal.lent)
            assert rag_under and journal.messages == alone.messages, ("the same window whatever lend says", lend)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MR8 -- a burst is not aimed on borrowed room
# ---------------------------------------------------------------------------
def test_mr8_a_burst_is_not_aimed_on_borrowed_room():
    """MR8: a tool turn's next burst is aimed as if the memory had borrowed nothing."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        facts = [f"Alice moved build {n} to Lisbon on 2026-03-{n % 28 + 1:02d} after the long review."
                 for n in range(1, 31)]
        schema = '{"name": "lookup", "description": "' + " ".join(["word"] * 300) + '"}'
        system = (" ".join(["rule"] * 360),)
        aims = {}
        for cid, question in (("c1", "Where did Alice move the builds to Lisbon?"), ("c2", "What is the weather?")):
            state, messages = _held_facts(lib, cid, facts, cfg, after=20)
            lib.pin(cid, " ".join(["alpha"] * 150), actor="user", config=cfg)
            win = oj.window(loaded, cid, question + " " + " ".join(["please"] * 150), messages=messages, config=cfg,
                            model_window=8192, system=system, tools=(schema,))
            assert win.refusal is None, win.refusal
            aims[cid] = (state.journal.burst[0].flesh, dict(win.layers)["tail"])
        own = int(composer.load_journal().shares()["memory"] * composer.load_budgets().at(8192).tail)
        (reached, tail_reached), (none, tail_none) = aims["c1"], aims["c2"]
        assert tail_reached > own and tail_none == 0, ("control: the memory borrowed in one, nothing in the other",
                                                      aims)
        assert reached >= none - own, ("the burst lowered by the memory's own share at most, never by what it borrowed",
                                       aims, own)
    finally:
        restore()


# ---------------------------------------------------------------------------
# MR9 -- the Flesh gives what the window passes its cap by with nothing borrowed
# ---------------------------------------------------------------------------
def test_mr9_the_flesh_gives_what_the_window_passes_its_cap_by_with_nothing_borrowed():
    """MR9: tool turns at 4096 past their cap even with nothing borrowed: each is the window a memory that borrows
    nothing lays."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    real = composer.compose_journal
    try:
        cfg = oj.config(lib)
        facts = [f"Alice moved build {n} to Lisbon on 2026-03-{n:02d}." for n in range(1, 9)]
        schema = '{"name": "lookup", "description": "' + " ".join(["word"] * 40) + '"}'
        system = (" ".join(["rule"] * 232),)
        question = "Where did Alice move the builds to Lisbon? " + " ".join(["please"] * 110)

        def laid(cid, after, core_words, peels):
            state, messages = _held_facts(lib, cid, facts, cfg, after=after)
            for n in range(peels):
                oj.accept(loaded, state, 2, " ".join([f"Bob checked queue {n} and it stays green."] * 3))
            lib.pin(cid, " ".join(["alpha"] * core_words), actor="user", config=cfg)
            return oj.window(loaded, cid, question, messages=messages, config=cfg, model_window=4096, system=system,
                             tools=(schema,))

        def borrowing_nothing(**kw):
            return real(**dict(kw, lend=0))

        for n, (after, core_words, peels) in enumerate(((60, 100, 2), (60, 110, 3), (40, 100, 2))):
            win = laid(f"a{n}", after, core_words, peels)
            shaded = sum(lib.counters().get("shadow", {}).values())
            composer.compose_journal = borrowing_nothing
            try:
                nothing = laid(f"b{n}", after, core_words, peels)
            finally:
                composer.compose_journal = real
            assert win.refusal is None and nothing.refusal is None, (n, win.refusal, nothing.refusal)
            assert sum(lib.counters().get("shadow", {}).values()) > shaded, (
                "control: with nothing borrowed the window passes its cap, the Flesh giving", n)
            assert (win.layers, win.tokens) == (nothing.layers, nothing.tokens), (
                "no more Flesh given for the room the memory borrowed", n, win.layers, nothing.layers)
    finally:
        composer.compose_journal = real
        restore()
