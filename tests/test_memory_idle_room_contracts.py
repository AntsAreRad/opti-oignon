#!/usr/bin/env python3
"""Contracts for the room the parts a window lays whole leave to the memory a question brings.

The system prompt, the Core and the current turn are laid whole, never cut,
each within its cap at the rung a plan chooses; what they leave of their caps
no other part claimed. At 4096 the tail's cap is 192 tokens and the memory's
envelope alone costs 174 by the chat's counter for a Qwen model: borrowing
the shares of the kinds a turn brings nothing of and the room the region
leaves of the peels' cap, the memory still laid few items, and none once the
region filled its cap, while a window took about 550 of the 2048 tokens its
prompt may take. Measured with the onion state of each drift A/B probe
recorded and each probe's window composed again on it, a 4B model missed
three probes at 4096 because the copies of a fact and its update, or the
facts themselves, had no room.

  * IR1 -- THE ROOM THE PARTS LAID WHOLE LEAVE: what the system prompt, the
    Core under its header and the turn's words leave of their caps, each
    counted as the window counts it -- a document of the turn not among its
    words -- none of them below nothing nor taken from another; the memory's
    whole cap adds it, its own share does not; no share keeps it out still.
  * IR2 -- AT 4096 THE MEMORY TAKES IT: a region that fills the peels' cap,
    the memory's shares alone short of its envelope: the fact the question
    reaches is laid, nothing cut, on the room the parts laid whole leave;
    parts that fill their caps leave it none.
  * IR3 -- THE PLAN HANDS WHAT THE WINDOW LAYS: on that room too, a Core
    pinned -- all it hands laid, what it kept back counted as cut -- and the
    memory laid past what the shares and the region lend; a window planned
    at 8192 and sent at 4096 lays on that room as well.
  * IR4 -- IT GIVES WAY FIRST: a window past its prompt's cap keeps its
    Flesh whole and sets nothing aside over a shadow; the memory keeps some
    of the room the parts laid whole leave, within the cap.
  * IR5 -- THE MEMORY'S ALONE, TO ITS WHOLE CAP: a retrieval the turn
    brings keeps to its share; the memory takes the room, to its whole cap
    and no further.

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
    "test_ir1_the_room_the_parts_laid_whole_leave": 5.0,
    "test_ir2_at_4096_the_memory_takes_the_room_the_parts_laid_whole_leave": 5.0,
    "test_ir3_the_plan_hands_what_the_window_lays_on_that_room": 5.0,
    "test_ir4_it_gives_way_first_before_the_flesh": 10.0,
    "test_ir5_the_room_is_the_memorys_alone": 5.0,
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


# ---------------------------------------------------------------------------
# IR1 -- the room the parts laid whole leave
# ---------------------------------------------------------------------------
def test_ir1_the_room_the_parts_laid_whole_leave():
    """IR1: each part's cap less what it lays, never below nothing; the memory's whole cap adds it, not its own."""
    loaded, restore = oj.open_window()
    composer = loaded[oj.COMPOSER]
    try:
        wrap = loaded[oj.WRAPPER].wrap
        budget = composer.load_budgets().at(4096)
        assert (budget.system, budget.core, budget.turn) == (512, 256, 256), ("control: the shipped table", budget)
        system = ("You are a careful assistant.", "Answer from what you are shown.")
        pinned = ("0123456789abcdef", "Never push to main on Atlas.")
        words = "Where did Alice move the build?"
        turn = ((("bare", words), ("file", " ".join(["page"] * 300))), (("typed",), ()))
        laid = sum(oj.count(text) for text in system)
        core = oj.count(composer._core_part(pinned))
        idle = composer.idle_room(budget, system, pinned, turn, oj.count, wrap)
        assert idle == (512 - laid) + (256 - core) + (256 - oj.count(words)), (
            "each part's cap less what it lays, the turn's document not among its words", idle)
        assert composer.idle_room(budget, system, ("", ""), turn, oj.count, wrap) == idle + core, (
            "an empty Core leaves its whole cap")
        long = (" ".join(["rule"] * 400),)
        assert oj.count(long[0]) > 512, "control: a system prompt past its cap"
        assert composer.idle_room(budget, long, pinned, turn, oj.count, wrap) == idle - (512 - laid), (
            "a part past its cap leaves nothing, and takes nothing from the others")
        shares = composer.load_journal().shares()
        spare = budget.peels - 100
        assert composer.memory_cap(budget, shares, set(), 100, idle) == (67, int(0.8 * 192) + spare + idle), (
            "the whole cap adds the room, the memory's own share does not")
        assert composer.memory_cap(budget, dict(shares, memory=0.0), set(), 0, idle) == (0, 0), "no share keeps it out"
        region = (("p1", "Alice reviewed the gateway.", "peel:1", None),)
        assert composer.memory_room(budget, shares, set(), region, oj.count, idle) == composer.memory_cap(
            budget, shares, set(), oj.count("Alice reviewed the gateway."), idle), "the room passed on whole"
    finally:
        restore()


# ---------------------------------------------------------------------------
# IR2 -- at 4096 the memory takes it
# ---------------------------------------------------------------------------
def test_ir2_at_4096_the_memory_takes_the_room_the_parts_laid_whole_leave():
    """IR2: a region filling the peels' cap: the fact laid, nothing cut, on the room the parts laid whole leave;
    parts filling their caps leave it none."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        question = "Where did Alice move the build?"
        full = lib.state_for("c1", cfg)
        messages = [oj.typed(_FACT), oj.answer("Noted.")] + oj.conversation(12, start=3)
        full.mirror(messages)
        oj.hold(full, 2, anchors=((full.flesh.turns()[0]["turn_id"], 0, len(_FACT)),))
        for n in range(2):
            oj.accept(loaded, full, 2, " ".join([f"Alice reviewed service {n} and it stays on the new cluster."] * 4))
        budget = composer.load_budgets().at(4096)
        first = oj.window(loaded, "c1", question, messages=messages, config=cfg, model_window=4096)
        assert first.refusal is None, ("the window lays the memory it borrows room for", first.refusal)
        region = dict(first.layers)["peels"]
        shipped = composer.load_budgets()
        exact = composer.Budgets(reserve=shipped.reserve, home=4096, tables=(
            (4096, replace(budget, peels=region, flesh=budget.flesh + budget.peels - region)),))
        roomy = oj.window(loaded, "c1", question, messages=messages, config=cfg, model_window=4096, budgets=exact)
        assert roomy.refusal is None and dict(roomy.layers)["peels"] == region, (
            "control: the region fills the peels' cap", roomy.refusal, roomy.layers)
        assert _FACT in roomy.messages[-1][1] and dict(roomy.cut)["memory"] == 0, (
            "the fact laid on the room the parts laid whole leave, nothing cut", roomy.layers, roomy.cut)
        snug = oj.window(loaded, "c1", question, messages=messages, config=cfg, model_window=4096,
                         budgets=oj.snug(loaded, question, budgets=exact))
        assert snug.refusal is None and _FACT not in snug.messages[-1][1], (
            "parts that fill their caps lend it nothing it can use: the shares alone do not hold the envelope",
            snug.layers)
    finally:
        restore()


# ---------------------------------------------------------------------------
# IR3 -- the plan hands what the window lays, on that room
# ---------------------------------------------------------------------------
def test_ir3_the_plan_hands_what_the_window_lays_on_that_room():
    """IR3: a hundred facts reached at 4096, a Core pinned: all the plan hands laid, what it kept back cut, past
    what the shares and the region lend; planned at 8192 and sent at 4096, laid on that room too."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        wrap = loaded[oj.WRAPPER].wrap
        facts = [f"Alice moved build {n} to Lisbon on 2026-03-{n % 28 + 1:02d}." for n in range(1, 101)]
        _state, messages = _held_facts(lib, "c1", facts, cfg)
        lib.pin("c1", " ".join(["Answers cite the build they name."] * 8), actor="user", config=cfg)
        question = "Where did Alice move the builds to Lisbon?"
        planned = oj.plan(lib, "c1", question, wrap=wrap, messages=messages, config=cfg, model_window=4096)
        held, win = planned._held, lib.compose_window(planned)
        assert win.refusal is None and win.rung == 4096, win.refusal
        assert dict(win.layers)["core"] > 0, ("control: a Core pinned", win.layers)
        handed = [item[2] for item in held.tail if item[0] == "memory"]
        assert handed and all(text in win.messages[-1][1] for text in handed), "all the plan hands is laid"
        assert dict(win.cut)["memory"] == held.trimmed, ("what it kept back counted as cut", win.cut, held.trimmed)
        _own, lent = composer.memory_room(composer.load_budgets().at(4096), held.shares, set(), held.region, oj.count)
        assert dict(win.layers)["tail"] > lent, ("laid past what the shares and the region lend", win.layers, lent)
        wide = oj.plan(lib, "c1", question, wrap=wrap, messages=messages, config=cfg, model_window=8192)
        sent = lib.compose_window(wide, num_ctx=4096)
        assert wide.rung == 8192 and sent.refusal is None and sent.num_ctx == 4096, (wide.rung, sent.refusal)
        assert dict(sent.layers)["tail"] > lent, ("sent at 4096, laid on that room too", sent.layers, lent)
    finally:
        restore()


# ---------------------------------------------------------------------------
# IR4 -- it gives way first, before the Flesh
# ---------------------------------------------------------------------------
def test_ir4_it_gives_way_first_before_the_flesh():
    """IR4: past the prompt's cap: the Flesh whole, no shadow; the memory keeps some of the room the parts laid
    whole leave, within the cap."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        cfg = oj.config(lib)
        wrap = loaded[oj.WRAPPER].wrap
        facts = [f"Alice moved build {n} to Lisbon on 2026-03-{n % 28 + 1:02d}." for n in range(1, 61)]
        state, messages = _held_facts(lib, "c1", facts, cfg, after=30)
        for n in range(3):
            oj.accept(loaded, state, 2, " ".join([f"Bob reviewed service {n} and it stays on the old cluster."] * 3))
        system = (" ".join(["rule"] * 150),)
        question = "Where did Alice move the builds to Lisbon?"
        planned = oj.plan(lib, "c1", question, wrap=wrap, messages=messages, config=cfg, model_window=4096,
                          system=system)
        held = planned._held

        def laid(tail):
            return composer.compose_journal(system=held.system, pinned=held.pinned, region=held.region,
                                            digest=held.digest, flesh=held.laid_turns, tail=tail, turn=held.turn,
                                            budget=held.budget, count=oj.count, overhead=held.overhead, wrap=wrap,
                                            shares=held.shares)

        try:
            laid(held.tail)
        except composer.BudgetError as exc:
            assert exc.motive == "window", ("control: with all it borrows the window passes its prompt's cap", exc)
        else:
            raise AssertionError("control: with all it borrows the window passes its prompt's cap")
        win = lib.compose_window(planned)
        assert win.refusal is None and win.tokens <= 2048, ("within the prompt's cap", win.refusal, win.tokens)
        layers = dict(win.layers)
        assert layers["flesh"] == dict(laid(()).layers)["flesh"] and "shadow" not in lib.counters(), (
            "the Flesh whole, as the plan holds it, nothing set aside", layers)
        _own, lent = composer.memory_room(held.budget, held.shares, set(), held.region, oj.count)
        assert layers["tail"] > lent, ("the memory keeps some of the room the parts laid whole leave", layers, lent)
    finally:
        restore()


# ---------------------------------------------------------------------------
# IR5 -- the room is the memory's alone
# ---------------------------------------------------------------------------
def test_ir5_the_room_is_the_memorys_alone():
    """IR5: a retrieval the turn brings laid to its share; the memory laid to its whole cap, the room among it,
    no further."""
    loaded, restore = oj.open_window()
    lib, composer = loaded[oj.LIBRARIAN], loaded[oj.COMPOSER]
    try:
        wrap = loaded[oj.WRAPPER].wrap
        held = oj.plan(lib, None, "Where did Alice move the builds?", wrap=wrap)._held
        budget = replace(composer.load_budgets().at(8192), peels=0)
        items = tuple(("memory", f"m{n}", f"Alice moved build {n} to Lisbon on 2026-03-{n:02d}, said the user.",
                       ("peels", f"anchor:{n:012d}:t{n:04d}"), (("typed",), ())) for n in range(1, 121))
        rag = tuple(("rag", f"rag:{n}", f"Section {n}: the build farm {n} runs in Lisbon.", None,
                     (("file",), (f"file:{n}",))) for n in range(20))
        journal = composer.compose_journal(system=oj.SYSTEM, pinned=held.pinned, region=(), digest="", flesh=(),
                                           tail=items + rag, turn=held.turn, budget=budget, count=oj.count,
                                           overhead=held.overhead, wrap=wrap, shares=held.shares)
        content = journal.messages[-1].content
        share = int(held.shares["rag"] * budget.tail)
        fits = composer.fit_kind("rag", rag, share, wrap, oj.count)[0]
        assert 0 < len(fits) < len(rag), ("control: the retrieval's share holds some of it", len(fits))
        assert [n for n, item in enumerate(rag) if item[2] in content] == fits, "the retrieval laid to its share"
        idle = composer.idle_room(budget, oj.SYSTEM, held.pinned, held.turn, oj.count, wrap)
        _own, whole = composer.memory_cap(budget, held.shares, {"rag"}, 0, idle)
        _own, lent = composer.memory_cap(budget, held.shares, {"rag"}, 0)
        capped = composer.fit_kind("memory", items, whole, wrap, oj.count)[0]
        assert len(composer.fit_kind("memory", items, lent, wrap, oj.count)[0]) < len(capped) < len(items), (
            "control: the whole cap binds, past what the shares lend", len(capped))
        assert [n for n, item in enumerate(items) if item[2] in content] == capped, (
            "the memory laid to its whole cap, the room among it, no further")
    finally:
        restore()
