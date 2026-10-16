#!/usr/bin/env python3
"""Contracts for the executor with the onion's journal: a conversation's turn planned, admitted, composed and sent.

With the onion on, a turn of a conversation the store keeps is laid as the
onion's journal: the librarian plans the window, the governor admits it at
the plan's rung, the librarian composes it under the context admitted, and
the executor sends those messages as they were composed, with the context
they were composed for and the answer held to the room they keep. A window
that cannot be laid is refused by name and leaves nothing behind.

  * JW5 -- an onion turn reaches no mechanism of the legacy window: no
    optimizer, no manual build, no coalescing, no memory composer, no
    archive, no compressor, no block of the onion's read side; at every rung
    of the ladder and through a descent, what is sent is the window's
    messages, their labels left behind. The same world with the onion off
    reaches every one of them. The sources the turn brings reach the plan
    raw: the retrieval asked for within its room, the web results with the
    sentence that cites them, the room read within the model's window.
  * JW6 -- the admission stands between the plan and the composition: per
    turn the plan, the network, the admission, the composition, the cache,
    the ticket and the stream, in that order; one admission, asked for the
    plan's rung, a rung of the governor's ladder; a context admitted under
    the rung is a descent, composed under it, sent with it, the next plan
    back at the rung; and the composition of a descent leaves the
    conversation's state as it found it. Offline, the turn is queued after
    its plan, never admitted, and the network is read once.
  * JW7 -- no truncation is left to the engine: every request carries the
    window's context and its answer's room, and what the window is
    accounted plus that room fits the context; a llama-server engine also
    gets the room as max_tokens and keeps the prefix, and its head forwards
    exactly those; the slot's own context bounds the window, and a slot
    that cannot be read is said so; an in-process engine with no governor
    to admit it bounds the window by the context it loads at; an answer
    that used all its room, to the last token, is the outcome "length",
    with a status line and no cache, and one from an engine that reports
    no count is not cached either; the engine's count of a cold window's
    prompt is kept, of a warm one never.
  * JW8 -- a refusal by name leaves nothing: a plan or a composition
    refused, a guard failed on any of its terms, a seam that raised, the
    governor's refusal -- one line yielded, the refusal's fixed words and
    figures and never an exception's text, nothing sent, saved, captured,
    curated or cached, one ledger row; with the governor itself, no
    admission and no load is left behind by any exit, a cache hit, a stop,
    or a cache lookup that raises. A curation that raises after the save is
    swallowed: the turn completes, saved and cached.
  * JW14 -- the binding: the journal's three names come from the
    librarian in an import of their own, which the curation's never shares;
    the librarian defines them; a librarian of three names, on and with a
    block, leaves the request the onion-off one, byte for byte, and its
    block is never asked; with no librarian at all the turn is saved and
    the legacy block rides it.
  * JW26 -- the counter hand-over: the plan gets the model the turn is
    routed to and the turn's own counter for it; nothing is planned with
    the onion off or without a conversation; the real librarian holds that
    counter for the conversation's verbs and none for the next block; the
    executor names no counter seam. (JW21 is the journal's answer floor.)
  * JW27 -- the retrieval is asked for in the builder's own units, four
    characters a token and no envelope, and what it returns is fitted to
    its share as the window counts it, code read heavier and the chunks'
    names included: it lands in its file envelope. A share that holds no
    chunk asks nothing of the trigger or the builder and is said left out;
    a retrieval the window lets go is said so, in a status line and in the
    ledger row's cut.
  * JW28 -- the onion's switch is read once the call's stop is live: an
    emergency stop fired while it is read ends the call, onion on or off.

Local-only (the public distribution ships no tests).
"""

import ast
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _onion_journal as oj  # noqa: E402
import _onion_off_golden as golden  # noqa: E402
import _onion_window_world as w  # noqa: E402
import test_executor_onion_off_golden_contracts as golden_suite  # noqa: E402
import test_executor_onion_wiring_contracts as wiring  # noqa: E402
from _isolation import REPO, isolate, source  # noqa: E402

BUDGET_S = {
    "test_jw5_an_onion_turn_reaches_no_legacy_mechanism_and_sends_the_window_as_composed": 8.0,
    "test_jw6_the_admission_stands_between_the_plan_and_the_composition_and_a_descent_is_composed_under_it": 8.0,
    "test_jw7_every_request_carries_the_windows_context_and_its_answers_room": 6.0,
    "test_jw8_a_refusal_by_name_leaves_nothing_sent_saved_cached_or_admitted": 8.0,
    "test_jw14_the_journal_is_bound_through_the_librarian_alone_and_a_three_name_librarian_changes_nothing": 4.0,
    "test_jw26_the_plan_gets_the_turns_own_counter_for_the_model_it_is_routed_to": 4.0,
    "test_jw27_the_retrieval_is_asked_for_in_the_builders_units_and_lands_whole_or_is_said_left_out": 4.0,
    "test_jw28_the_onion_switch_is_read_once_the_calls_stop_is_live": 3.0,
}

_TWO = (("conv-1", w.FIRST, {}), ("conv-1", w.SECOND, {}))
_LADDER = tuple(yaml.safe_load((REPO / "opti_oignon" / "config" / "resource_governor.yaml")
                               .read_text(encoding="ascii"))["ctx_ladder"])
_REFUSED = re.compile(r"^\[ERR\] The onion window was refused \(([a-z_]+)\): (.+) \[(\d+(?: \d+)*)?\]$")


def _sent_as_composed(world):
    """Each request's messages and the window composed for it, as the engine was sent them and as composed."""
    return [(call["messages"], [{"role": role, "content": content} for role, content, _label in window.messages])
            for call, window in zip(w.sent(world), world.journal.composed)]


def _events(record, kind):
    return [event[1] if len(event) > 1 else {} for event in record["trace"] if event[0] == kind]


def _first(kinds, kind):
    return kinds.index(kind) if kind in kinds else None


# ---------------------------------------------------------------------------
# JW5 -- no legacy mechanism on the onion path
# ---------------------------------------------------------------------------
# Every mechanism of the legacy window, by the trace it leaves; the onion's
# read side of before is a tripwire of its own.
_LEGACY = ("legacy.optimize", "legacy.build", "legacy.coalesce", "memory.compose", "memory.build", "archive.trigger",
           "archive.retrieve", "compress")
_TRIPWIRES = ("trip.memory_block", "trip.use_counter")


def _traced_legacy(world, executor):
    """The optimizer, the manual build and the coalescing, each traced where the executor reaches it."""
    trace = world.trace

    def tracing(kind, call):
        def traced(*args, **kwargs):
            trace.add(kind)
            return call(*args, **kwargs)
        return traced

    executor._build_conversation_messages = tracing("legacy.build", executor._build_conversation_messages)
    world.executor_module._coalesce_user_turns = tracing("legacy.coalesce", world.executor_module._coalesce_user_turns)
    if getattr(world, "optimizer", None) is not None:
        world.optimizer.optimize = tracing("legacy.optimize", world.optimizer.optimize)


def _reached(*, enabled, integrated, optimizer=True, journal=None, gov=None):
    """The kinds a two-turn conversation reaches, and the world, closed."""
    world, close = w.open_world(journal=dict(journal or {}, enabled=enabled), gov=gov or {}, integrated=integrated,
                                optimizer=optimizer, history={"conv-1": golden.history_of(8, size=600)},
                                projects=True, web=True)
    try:
        executor = world.executor_module.Executor()
        _traced_legacy(world, executor)
        records = w.run_turns(world, (("conv-1", "As you said before, what grows in the east beds?",
                                       {"web_search": True}), ("conv-1", w.SECOND, {})), executor=executor)
        return {kind for record in records for kind in w.kinds(record)}, world
    finally:
        close()


def test_jw5_an_onion_turn_reaches_no_legacy_mechanism_and_sends_the_window_as_composed():
    seen_off = set()
    for integrated, optimizer in ((True, True), (True, False), (False, True)):
        fired, _world = _reached(enabled=False, integrated=integrated, optimizer=optimizer)
        seen_off |= fired
        assert "plan" not in fired and "compose" not in fired, "control: the onion off lays no journal"
    assert set(_LEGACY) <= seen_off, ("control: the onion off reaches every legacy mechanism",
                                      sorted(set(_LEGACY) - seen_off))
    # Every rung of the ladder and a descent, the optimizer on; the optimizer off, and the plain window, at home.
    cases = [(rung, admitted, True, True)
             for rung, admitted in ((4096, 4096), (8192, 8192), (16384, 16384), (32768, 32768), (16384, 4096))]
    cases += [(16384, 16384, True, False), (16384, 16384, False, True)]
    for case in cases:
        rung, admitted, integrated, optimizer = case
        fired, world = _reached(enabled=True, integrated=integrated, optimizer=optimizer,
                                journal={"rung": rung}, gov={"num_ctx": admitted})
        assert "plan" in fired and "compose" in fired, ("control: the journal laid the window", case)
        assert not fired & set(_LEGACY + _TRIPWIRES), (case, sorted(fired & set(_LEGACY + _TRIPWIRES)))
        pairs = _sent_as_composed(world)
        assert len(pairs) == 2, (case, len(pairs))
        for sent, composed in pairs:
            assert sent == composed, case
        assert [call["options"]["num_ctx"] for call in w.sent(world)] == [admitted, admitted], case
        # The sources the turn brings, as the plan is handed them.
        first = world.trace.turns[0]
        assert [event[1]["model_window"] for event in first if event[0] == "tail_room"] == [32768], case
        plan = world.journal.plans[0]
        assert [part["kind"] for part in plan["tail"]] == ["rag", "web"], case
        assert plan["tail"][1].get("suffix") == world.executor_module._WEB_CITATION, case
        (built,) = [event[1] for event in first if event[0] == "project.build"]
        asked = built["kwargs"]
        assert set(asked) == {"budget_tokens"} and type(asked["budget_tokens"]) is int, (case, asked)
        budget = asked["budget_tokens"]
        assert 0 < budget < w.ROOM["rag"], (case, budget)
        filled = world.wrapper.wrap("y" * 4 * budget, source=world.wrapper.SOURCE_FILE)
        assert world.executor._estimate_tokens(filled, w.MODEL) <= w.ROOM["rag"], (
            "what the builder fills at four characters a token fits the room once wrapped", case, budget)
    world, close = w.open_world()
    try:
        world.journal.memory_block("conv-1")
        world.journal.use_counter("conv-1", len, w.MODEL)
        assert [event[0] for event in world.trace.turns[-1]] == list(_TRIPWIRES), "control: the tripwires are live"
    finally:
        close()


# ---------------------------------------------------------------------------
# JW6 -- the admission between the plan and the composition
# ---------------------------------------------------------------------------
_ORDER = ("plan", "network.online", "admit", "compose", "cache.key", "cache.get", "ticket.set", "stream")


def _state_digest(state):
    """The conversation's onion as it stands: its Core, its receipts and spans, its peels, its Flesh, its lineage."""
    with state.lock:
        held = (state.core.root(), state.core.text(), state.ledger.digest(state.cellar),
                [repr(receipt) for receipt in state.ledger.all()], sorted(state.cellar.keys()),
                [repr(peel) for peel in state.tree.all()], [sorted(turn.items()) for turn in state.flesh.turns()],
                state.seen, sorted(state.lineage.items()))
    return hashlib.sha256(repr(held).encode("utf-8")).hexdigest()


def test_jw6_the_admission_stands_between_the_plan_and_the_composition_and_a_descent_is_composed_under_it():
    world, close = w.open_world(gov={}, projects=True)
    try:
        records = w.run_turns(world, _TWO)
    finally:
        close()
    for index, record in enumerate(records):
        kinds = w.kinds(record)
        places = [_first(kinds, kind) for kind in _ORDER]
        assert None not in places and places == sorted(places), (index, list(zip(_ORDER, places)))
        assert [kind for kind in kinds if kind in _ORDER] == list(_ORDER), (index, kinds)
        (admitted,) = _events(record, "admit")
        assert admitted["requested_ctx"] == world.journal.rung and admitted["requested_ctx"] in _LADDER
        assert kinds.count("admit") == 1 and kinds.count("compose") == 1, index
    world, close = w.open_world(gov={}, journal={"plan_refusal": ("turn", (900, 512))})
    try:
        (record,) = w.run_turns(world, _TWO[:1])
    finally:
        close()
    assert "plan" in w.kinds(record) and "admit" not in w.kinds(record), "a plan refused is never admitted"
    assert "compose" not in w.kinds(record)
    world, close = w.open_world(gov={}, offline=True)
    try:
        (record,) = w.run_turns(world, _TWO[:1])
    finally:
        close()
    kinds = w.kinds(record)
    assert kinds.count("network.online") == 1 and kinds.count("network.enqueue") == 1, kinds
    assert kinds.index("plan") < kinds.index("network.online"), kinds
    assert not {"admit", "compose", "cache.get", "stream"} & set(kinds), kinds
    assert [row["outcome"] for row in world.rows] == ["offline_queued"], world.rows
    assert record["yields"] and record["yields"][0].startswith("[Offline]"), record["yields"]

    # A descent: the context admitted under the plan's rung, composed and sent under it.
    world, close = w.open_world(gov={"num_ctx": (8192, 16384)})
    try:
        records = w.run_turns(world, _TWO)
    finally:
        close()
    assert [_events(record, "compose")[0]["num_ctx"] for record in records] == [8192, 16384]
    assert [call["options"]["num_ctx"] for call in w.sent(world)] == [8192, 16384]
    assert [(row["win_plan_rung"], row["win_num_ctx"], row["win_descent"]) for row in world.rows] == [
        (16384, 8192, True), (16384, 16384, False)]

    # The same with the onion itself: the descent laid over a shadow, the conversation's state untouched, the
    # next plan back at the rung.
    history = {"conv-1": golden.history_of(10, size=450)}
    world, close = w.open_real(gov={"num_ctx": (8192, 16384)}, history=history)
    try:
        lib = world.librarian
        composing = world.executor_module._onion_compose_window
        digests = []

        def digesting(plan, *, num_ctx=None):
            state = lib._states["conv-1"]
            before = _state_digest(state)
            window = composing(plan, num_ctx=num_ctx)
            digests.append((num_ctx, before, _state_digest(state), window))
            return window

        world.executor_module._onion_compose_window = digesting
        records = w.run_turns(world, _TWO)
        rows = world.rows
        assert [(row["outcome"], row["win_plan_rung"], row["win_num_ctx"], row["win_descent"]) for row in rows] == [
            ("completed", 16384, 8192, True), ("completed", 16384, 16384, False)], rows
        assert all(row["win_plan_rung"] in _LADDER for row in rows)
        assert [_events(record, "admit")[0]["requested_ctx"] for record in records] == [16384, 16384]
        (descent, before, after, window), (_ctx, _before, _after, whole) = digests
        assert descent == 8192 and before == after, "the composition of a descent leaves the state as it was"
        assert len(window.messages) < len(whole.messages) - 2, "control: the descent laid the Flesh over a shadow"
        state = lib._states["conv-1"]
        held = _state_digest(state)
        oj.bare(state, 2)
        assert _state_digest(state) != held, "control: an eviction moves the digest"
    finally:
        close()


# ---------------------------------------------------------------------------
# JW7 -- no truncation left to the engine
# ---------------------------------------------------------------------------
def _replayed(options):
    """What the llama-server heads send, stream and generate, for ``options``."""
    loaded, close = isolate(targets={golden_suite._BACKEND: source("inference_backend.py")}, seeded={},
                            packages=("opti_oignon",))
    module = loaded[golden_suite._BACKEND]
    sent = []

    def urlopen(request, timeout=None):
        sent.append(json.loads(request.data.decode("utf-8")))
        return golden_suite._Response(golden_suite._BODY, golden_suite._SSE)

    real = module.urllib.request.urlopen
    module.urllib.request.urlopen = urlopen
    try:
        module._live_mode = lambda: "daily"
        backend = module.LlamaServerBackend(host="http://fake:8080")
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        list(backend.stream(model=w.MODEL, messages=messages, options=dict(options), keep_alive="30m"))
        backend.generate(model=w.MODEL, messages=messages, options=dict(options), keep_alive="30m")
    finally:
        module.urllib.request.urlopen = real
        close()
    return sent


def _carries_the_window(world):
    for call, window in zip(w.sent(world), world.journal.composed):
        options = call["options"]
        assert options.get("num_ctx") == window.num_ctx and options.get("num_predict") == window.num_predict, options
        assert window.accounted + window.num_predict <= window.num_ctx and window.num_predict >= 1


def test_jw7_every_request_carries_the_windows_context_and_its_answers_room():
    for gov in ({}, None):
        world, close = w.open_world(gov=gov)
        try:
            w.run_turns(world, _TWO)
        finally:
            close()
        assert len(w.sent(world)) == 2, gov
        _carries_the_window(world)
        assert all("max_tokens" not in call["options"] and "cache_prompt" not in call["options"]
                   for call in w.sent(world)), "an Ollama engine gets neither max_tokens nor cache_prompt"
        assert [row["win_slot_unknown"] for row in world.rows] == [False, False]

    # llama-server: the slot's context bounds the window; the room goes as max_tokens, the prefix is kept.
    slots = [{"id": 0, "n_ctx": 16384}, {"id": 1, "n_ctx": 8192}, {"id": 2, "n_ctx": "8192"}]
    world, close = w.open_world(gov={"num_ctx": 16384}, backend="llama_server", slots=slots)
    try:
        (record,) = w.run_turns(world, _TWO[:1])
    finally:
        close()
    assert _events(record, "plan")[0]["model_window"] == 8192, "the window is planned within the slot's context"
    assert _events(record, "tail_room")[0]["model_window"] == 8192, "the sources' room is read within it too"
    assert _events(record, "compose")[0]["num_ctx"] == 8192, "the context admitted is capped by the slot's"
    _carries_the_window(world)
    (call,) = w.sent(world)
    window = world.journal.composed[0]
    assert call["options"]["max_tokens"] == window.num_predict and call["options"]["cache_prompt"] is True
    assert world.rows[-1]["win_slot_unknown"] is False and world.rows[-1]["win_num_ctx"] == 8192
    for payload in _replayed(call["options"]):
        assert payload["max_tokens"] == window.num_predict and payload["cache_prompt"] is True, payload
        assert "num_ctx" not in payload and "num_predict" not in payload, sorted(payload)
    world, close = w.open_world(gov={"num_ctx": 16384}, backend="llama_server")
    try:
        (record,) = w.run_turns(world, _TWO[:1])
    finally:
        close()
    assert _events(record, "compose")[0]["num_ctx"] == 16384 and world.rows[-1]["win_slot_unknown"] is True, (
        "a slot that lists no context leaves the bound unknown, said so in the row")
    # The scripted engine reports no count: its answer is never cached, and is said not stored.
    assert "cache.put" not in w.kinds(record) and world.rows[-1]["cache_stored"] is False, w.kinds(record)

    # An in-process engine loads at its own context when no admission names one: with no governor, that context
    # bounds the window; with one, the admission names it.
    for gov, room, sent in ((None, 4096, 4096), ({"num_ctx": 16384}, 32768, 16384)):
        world, close = w.open_world(gov=gov, backend="llama_cpp", engine_context=4096)
        try:
            (record,) = w.run_turns(world, _TWO[:1])
        finally:
            close()
        assert _events(record, "tail_room")[0]["model_window"] == room, (gov, _events(record, "tail_room"))
        assert _events(record, "plan")[0]["model_window"] == room, gov
        assert _events(record, "compose")[0]["num_ctx"] == sent, (gov, _events(record, "compose"))
        _carries_the_window(world)

    # The answer that used all its room, to its last token: the outcome, a status line, no cache; the answer is
    # kept. One token short of it is completed, and cached.
    for counted, outcome in ((1024, "length"), (1023, "completed")):
        world, close = w.open_world(gov={}, journal={"num_predict": 1024}, engine_extra={"eval_count": counted})
        try:
            (record,) = w.run_turns(world, _TWO[:1])
        finally:
            close()
        kinds = w.kinds(record)
        room = world.journal.composed[0].num_predict
        assert [row["outcome"] for row in world.rows] == [outcome], world.rows
        assert ("cache.put" in kinds) is (outcome == "completed"), kinds
        assert world.rows[-1]["cache_stored"] is (outcome == "completed"), world.rows[-1]
        said = [line for line in record["statuses"] if "used all its room" in line]
        assert said == ([f"[!] The answer used all its room ({room} tokens)"] if outcome == "length" else []), said
        assert record["yields"] == ["Hello", " world"] and kinds.count("store.add") == 2, outcome

    # The engine's count of the prompt, kept for a cold window only.
    world, close = w.open_world(gov={}, engine_extra={"prompt_eval_count": 77})
    try:
        w.run_turns(world, _TWO)
    finally:
        close()
    assert [window.cold for window in world.journal.composed] == [True, False], "control: one cold, one warm"
    assert [row.get("engine_prompt_tokens") for row in world.rows] == [77, None], world.rows


# ---------------------------------------------------------------------------
# JW8 -- a refusal by name leaves nothing
# ---------------------------------------------------------------------------
_REFUSALS = (
    ("a plan refused", {"plan_refusal": ("turn", (900, 512))}, "turn", (900, 512), False),
    ("a composition refused", {"compose_refusal": ("window", (17000, 16384))}, "window", (17000, 16384), True),
    ("the guard failed", {"accounted": 16000, "num_predict": 1024}, "count", (16000, 1024, 16384), True),
    ("a window composed for another context", {"sent": 32768, "accounted": 100, "num_predict": 1024}, "count",
     (100, 1024, 32768), True),
    ("an answer left no room", {"accounted": 100, "num_predict": 0}, "count", (100, 0, 16384), True),
    ("the plan raised", {"raises": ("plan",)}, "raised", (), False),
    ("the composition raised", {"raises": ("compose",)}, "raised", (), True),
)
_LEFT_NOTHING = ("stream", "generate", "store.add", "store.metadata", "capture", "curate", "cache.put")


def _real_exits():
    """Each way a turn leaves before its stream after its admission, as ``(name, world keywords, hook, turns)``."""

    def stop_during_admission(world):
        admit = world.governor.admit

        def admitting(*args, **kwargs):
            decision = admit(*args, **kwargs)
            world.run.stop.set()
            return decision

        world.governor.admit = admitting

    def stop_on_the_thread(world):
        hold = world.executor_module._governor_set_ticket

        def holding(decision):
            hold(decision)
            world.run.stop.set()

        world.executor_module._governor_set_ticket = holding

    plain = (("conv-1", w.FIRST, {}),)
    return (
        ("a composition refused", {"journal": {"compose_refusal": ("window", (17000, 16384))}}, None, plain),
        ("the guard failed", {"journal": {"accounted": 16000, "num_predict": 1024}}, None, plain),
        ("the composition raised", {"journal": {"raises": ("compose",)}}, None, plain),
        ("the governor refused", {"real_gov": {"capacity": 2.0, "weights": {w.MODEL: 9.0}, "ram_mb": 1000.0}}, None,
         plain),
        ("a stop during the admission", {}, stop_during_admission, plain),
        ("a stop at the lookup", {"stop_at": {"lookup": {0}}}, None, plain),
        ("a stop before the stream", {"stop_at": {"generating": {0}}}, None, plain),
        ("a stop on the stream's thread", {}, stop_on_the_thread, plain),
        ("a cache hit", {"hit": True, "engine_extra": {"eval_count": 3}}, None,
         (("conv-1", w.FIRST, {}), ("conv-2", w.FIRST, {}))),
    )


def test_jw8_a_refusal_by_name_leaves_nothing_sent_saved_cached_or_admitted():
    for name, journal, motive, figures, admitted in _REFUSALS:
        world, close = w.open_world(journal=journal, gov={}, integrated=True)
        try:
            (record,) = w.run_turns(world, _TWO[:1])
            words = world.executor_module._ONION_REFUSAL_WORDS[motive]
        finally:
            close()
        assert len(record["yields"]) == 1, ("a refusal is one line, and nothing is sent", name, record["yields"])
        (said,) = record["yields"]
        match = _REFUSED.match(said)
        assert match and match.group(1) == motive and match.group(2) == words, (name, said)
        assert tuple(int(value) for value in (match.group(3) or "").split()) == figures, (name, said)
        assert "never to be shown" not in said and "RuntimeError" not in said, name
        assert record["results"].get("step_error") == said and record["returned"][1] == said, name
        assert "admission_refused" not in record["results"], name
        assert not set(w.kinds(record)) & set(_LEFT_NOTHING), (name, w.kinds(record))
        assert [(row["outcome"], row.get("win_refusal"), row.get("win_figures")) for row in world.rows] == [
            ("window_refused", motive, list(figures))], (name, world.rows)
        assert ("admit" in w.kinds(record)) is admitted, name
        assert world.gov_state.admitted == set() and world.gov_state.pending == set(), (name, world.gov_state)
        if admitted:
            kinds = w.kinds(record)
            assert kinds.index("forgo.hold") < kinds.index("forgo.end") < kinds.index("yield"), (name, kinds)
    world, close = w.open_world(gov={"admitted": False}, integrated=True)
    try:
        (record,) = w.run_turns(world, _TWO[:1])
    finally:
        close()
    assert record["yields"] == ["[ERR] refused: not enough memory"]
    assert record["results"].get("admission_refused") == "refused: not enough memory"
    assert "step_error" not in record["results"]
    assert not set(w.kinds(record)) & set(_LEFT_NOTHING + ("compose", "forgo.hold"))
    assert [(row["outcome"], row["gov_requested_ctx"], row["win_plan_rung"]) for row in world.rows] == [
        ("governor_refused", 16384, 16384)]
    world, close = w.open_world(gov={}, integrated=True, engine_extra={"eval_count": 3})
    try:
        (record,) = w.run_turns(world, _TWO[:1])
    finally:
        close()
    assert {"stream", "store.add", "capture", "curate", "cache.put"} <= set(w.kinds(record)), (
        "control: a turn laid reaches every seam a refusal never reaches")

    # A curation that raises after the save is the librarian's problem, never the turn's.
    world, close = w.open_world(gov={}, journal={"raises": ("curate",)}, engine_extra={"eval_count": 3})
    try:
        try:
            (record,) = w.run_turns(world, _TWO[:1])
        except RuntimeError as exc:
            raise AssertionError(f"a curation that raises took the turn with it: {exc}") from exc
    finally:
        close()
    kinds = w.kinds(record)
    assert "curate" in kinds and kinds.count("store.add") == 2 and "cache.put" in kinds, kinds
    assert record["yields"] == ["Hello", " world"] and [row["outcome"] for row in world.rows] == ["completed"]

    # The governor itself: no admission held, no load pending, no background held by either, after any exit.
    world, close = w.open_world(real_gov={})
    try:
        gov, forgo = world.governor, world.executor_module._onion_forgo
        decision = gov.admit(w.MODEL, 16384, caller="chat")
        assert decision.admitted and decision.load_expected, "control: the model is loaded nowhere"
        assert gov._admitted and gov.pending_loads() and gov._background_hold() == "interactive_admitted", (
            "control: an admission left alone holds the background and claims its load")
        forgo(decision)
        assert gov._admitted == {} and gov.pending_loads() == [] and gov._background_hold() is None
    finally:
        close()
    for name, options, hook, turns in _real_exits():
        options = dict(options)
        options.setdefault("real_gov", {})
        world, close = w.open_world(**options)
        try:
            gov = world.governor
            if hook is not None:
                hook(world)
            records = []
            for index, turn in enumerate(turns):
                if index:
                    for ticket in list(gov._pending_loads):
                        gov.end_pending_load(ticket)
                records += w.run_turns(world, (turn,), executor=getattr(world, "executor", None))
            assert len(w.sent(world)) == (1 if name == "a cache hit" else 0), (name, len(w.sent(world)))
            assert gov._admitted == {}, (name, gov._admitted)
            assert gov.pending_loads() == [], (name, gov.pending_loads())
            assert gov._background_hold() != "interactive_admitted", name
            expected = {"a cache hit": "cache_hit", "the governor refused": "governor_refused"}.get(name)
            if expected is None:
                expected = "cancelled" if name.startswith("a stop") else "window_refused"
            assert world.rows[-1]["outcome"] == expected, (name, [row["outcome"] for row in world.rows])
        finally:
            close()

    # A cache lookup that raises: the turn fails as the off path does, and leaves no admission behind.
    world, close = w.open_world(real_gov={})
    try:
        gov = world.governor

        def locked(key):
            raise sqlite3.OperationalError("database is locked")

        world.executor_module._response_cache.get = locked
        with pytest.raises(sqlite3.OperationalError):
            w.run_turns(world, _TWO[:1])
        assert gov._admitted == {} and gov.pending_loads() == [] and gov._background_hold() is None, (
            gov._admitted, gov.pending_loads())
        assert w.sent(world) == [], "control: the turn never reached the stream"
    finally:
        close()
    # A hit told to a caller whose status line fails: the admission was given back before it was told.
    world, close = w.open_world(real_gov={}, hit=True, engine_extra={"eval_count": 3})
    try:
        gov = world.governor
        w.run_turns(world, (("conv-1", w.FIRST, {}),))
        for ticket in list(gov._pending_loads):
            gov.end_pending_load(ticket)
        with pytest.raises(RuntimeError, match="status line"):
            w.run_turns(world, (("conv-2", w.FIRST, {"raise_at": "[CACHE]"}),), executor=world.executor)
        assert gov._admitted == {} and gov.pending_loads() == [], (gov._admitted, gov.pending_loads())
        assert len(w.sent(world)) == 1, "control: the second turn hit the cache"
    finally:
        close()


# ---------------------------------------------------------------------------
# JW14 -- the binding
# ---------------------------------------------------------------------------
_JOURNAL = {"plan_window", "compose_window", "tail_room"}
_ONION_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian")


def _imported_from(tree, module):
    """Each import of ``module`` in ``tree``, as the set of names it imports, with the Try that holds it if any."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for statement in node.body:
                if isinstance(statement, ast.ImportFrom) and statement.module == module and statement.level == 1:
                    found.append((node, {alias.name for alias in statement.names}))
    return found


def test_jw14_the_journal_is_bound_through_the_librarian_alone_and_a_three_name_librarian_changes_nothing():
    tree = ast.parse(source("executor.py").read_text(encoding="utf-8"))
    tries = {}
    for node, names in _imported_from(tree, "memory.librarian"):
        tries.setdefault(id(node), set()).update(names)
    journal = [names for names in tries.values() if names & _JOURNAL]
    curation = [names for names in tries.values() if "maybe_curate" in names]
    assert journal == [_JOURNAL], ("one Try imports exactly the journal's three names", journal)
    assert len(curation) == 1 and not curation[0] & _JOURNAL, curation
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level == 1}
    assert {f"memory.{name}" for name in _ONION_MODULES} & imported == {"memory.librarian"}, sorted(imported)
    library = ast.parse(source("memory", "librarian.py").read_text(encoding="utf-8"))
    defined = {node.name for node in library.body if isinstance(node, ast.FunctionDef)}
    assert _JOURNAL <= defined, sorted(_JOURNAL - defined)

    # A librarian of three names, on and with a block: the request of the onion off, byte for byte.
    world, close = w.open_world(three_names={"enabled": True})
    try:
        records = w.run_turns(world, tuple((cid, question, {"claim": False}) for cid, question, _kw in _TWO))
    finally:
        close()
    kinds = [kind for record in records for kind in w.kinds(record)]
    calls = json.dumps(world.scripted.calls, sort_keys=True, ensure_ascii=False)
    assert golden.sha(calls) == golden._WIRING_DIGEST, "the request is the onion-off one, byte for byte"
    assert "trip.memory_block" not in kinds and "plan" not in kinds and "compose" not in kinds, kinds
    assert kinds.count("curate") == 2, ("the write side is offered each saved turn", kinds.count("curate"))
    world.three_names.memory_block("conv-1")
    assert world.trace.turns[-1][-1][0] == "trip.memory_block", "control: the block's tripwire is live"

    # No librarian at all: the turn is saved and the legacy block rides it (XW10's first half, word for word).
    mod, wrapper, scripted, composer, conversations, restore = wiring._load(
        librarian_absent=True, conversations=wiring._OriginConversations())
    try:
        ex = mod.Executor()
        wiring._drive(ex.execute("What is my name?", wiring._routing(), refine=False, conversation_id="conv-1"))
        assert wrapper.wrap(wiring._LEGACY, source=wrapper.SOURCE_MEMORY) in wiring._turn(scripted)
        assert composer.calls == 1
        assert conversations.messages.get("conv-1"), "the turn is saved with no librarian at all"
    finally:
        restore()


# ---------------------------------------------------------------------------
# JW26 -- the counter hand-over
# ---------------------------------------------------------------------------
_COUNTED_MODEL = "counted-model:7b"


def _model_sensitive(text, model=""):
    """The executor's own count, read for the model the turn is sent to: a third more for this one."""
    return len(text) // (3 if model == _COUNTED_MODEL else 4)


def _names(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update(alias.name for alias in node.names)
            out.update(alias.asname for alias in node.names if alias.asname)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
    return out


def test_jw26_the_plan_gets_the_turns_own_counter_for_the_model_it_is_routed_to():
    world, close = w.open_world(gov={})
    try:
        ex = world.executor_module.Executor()
        ex._estimate_tokens = _model_sensitive
        w.run_turns(world, (("conv-1", w.FIRST, {"model": _COUNTED_MODEL}),), executor=ex)
    finally:
        close()
    (plan,) = world.journal.plans
    assert plan["model"] == _COUNTED_MODEL
    for text in wiring._COUNTED:
        assert plan["count"](text) == ex._estimate_tokens(text, _COUNTED_MODEL) == len(text) // 3, text
    assert any(plan["count"](text) != ex._estimate_tokens(text) for text in wiring._COUNTED), (
        "control: a count blind to the model gives other numbers")
    assert any(plan["count"](text) != max(1, int(len(text.split()) * 1.3)) for text in wiring._COUNTED), (
        "control: the turn's count is not the onion's default")
    for enabled, conversation_id in ((False, "conv-1"), (True, None)):
        world, close = w.open_world(gov={}, journal={"enabled": enabled})
        try:
            (record,) = w.run_turns(world, ((conversation_id, w.FIRST, {}),))
        finally:
            close()
        kinds = w.kinds(record)
        assert "stream" in kinds, "control: the turn ran"
        assert "plan" not in kinds and "trip.use_counter" not in kinds, (enabled, conversation_id, kinds)

    # The onion itself: the counter held for the conversation's verbs, none handed to a next block.
    world, close = w.open_real(gov={})
    try:
        lib = world.librarian
        counted = []

        def counting(text, model=""):
            counted.append((str(text), model))
            return _model_sensitive(text, model)

        ex = world.executor_module.Executor()
        ex._estimate_tokens = counting
        w.run_turns(world, (("conv-1", w.FIRST, {"model": _COUNTED_MODEL}),), executor=ex)
        assert lib._counters["conv-1"][1] == _COUNTED_MODEL
        assert getattr(lib._handed, "counter", None) is None, "nothing is handed to a next block"
        counted.clear()
        lib.pin("conv-1", "Answers cite their source.", actor="user")
        assert any("Answers cite their source." in text and model == _COUNTED_MODEL for text, model in counted), (
            "a later pin counts with the turn's counter")
    finally:
        close()
    assert "use_counter" not in _names(ast.parse(source("executor.py").read_text(encoding="utf-8")))
    assert "use_counter" in _names(ast.parse(source("memory", "librarian.py").read_text(encoding="utf-8"))), (
        "control: the probe finds the seam where it is")


# ---------------------------------------------------------------------------
# JW27 -- the retrieval in the builder's units
# ---------------------------------------------------------------------------
_RAG = "SENTINEL-RAG-6f0c"
_ENVELOPE = re.compile(r'<untrusted_data source="([a-z_]+)" trusted="false">\n(.*?)\n</untrusted_data>', re.S)


def _filling(world, asked, *, ignore=False):
    """A builder that fills the budget it is handed as the project builder does, four characters a token and no
    envelope, from relevant chunks enough to fill any; ``ignore`` fills its own budget whatever it is handed."""

    def build_context(project_id, question, budget_tokens=None):
        asked.append(budget_tokens)
        budget = 4096 if budget_tokens is None or ignore else budget_tokens
        text = (f"[beds.md] {_RAG} " + "Bed notes: onions in the east beds, garlic by the west wall. " * 800)
        return SimpleNamespace(context_text=text[: budget * 4], chunks_used=3, total_tokens_estimate=budget)

    world.projects.build_context = build_context


def _in_file_envelope(messages, sentinel):
    for message in messages:
        for found in _ENVELOPE.finditer(message["content"]):
            if sentinel in found.group(2):
                return found.group(1) == "file"
    return False


# Chunks of code as a project holds them: R, its functions assigned, and Rust.
_R = ("fit_beds <- function(data, formula) {\n  model <- lm(formula, data = data)\n  summary(model)\n}\n"
      "plot_beds <- function(fit) {\n  plot(fit$residuals, main = \"Beds\")\n}\n") * 3
_RUST = ("fn parse_beds(input: &str) -> Result<Vec<u32>, String> {\n    input.split(',')\n"
         "        .map(|s| s.trim().parse::<u32>().map_err(|e| e.to_string()))\n        .collect()\n}\n") * 3


def _code_builder(asked):
    """A builder that fills the budget it is handed as the project builder does: each chunk under its
    ``[name (section)]`` name, the names and the frame left out of its four characters a token, the last chunk cut
    when more than fifty of its tokens fit."""
    chunks = [("beds.R", "model", f"# {_RAG}\n" + _R)] + [("parse.rs", "beds", _RUST), ("beds.R", "plots", _R)] * 4

    def build_context(project_id, question, budget_tokens=None):
        asked.append(budget_tokens)
        available = (4096 if budget_tokens is None else budget_tokens) - 32
        parts, used = [], 0
        for name, section, content in chunks:
            tokens = len(content) // 4
            if used + tokens > available:
                if available - used > 50:
                    parts.append(f"[{name}] {content[:(available - used) * 4]}...")
                break
            parts.append(f"[{name} ({section})]\n{content}")
            used += tokens
        text = "\n\n".join(["--- Project Context ---", *parts, "--- End Project Context ---"]) if parts else ""
        return SimpleNamespace(context_text=text, chunks_used=len(parts), total_tokens_estimate=len(text) // 4)

    return build_context


def _calibrated():
    """The application's own count: by the model's family, and code read heavier -- the whole text, the envelope's
    policy included, once it holds any."""
    loaded, close = isolate(targets={"opti_oignon.context_manager": source("context_manager.py")}, seeded={})
    try:
        calibrated = loaded["opti_oignon.context_manager"].estimate_tokens_calibrated
    finally:
        close()
    return lambda text, model=None: calibrated(str(text), model or "")


def test_jw27_the_retrieval_is_asked_for_in_the_builders_units_and_lands_whole_or_is_said_left_out():
    # Prose: the room the builder is handed is one its fill fits once wrapped.
    world, close = w.open_real(gov={}, projects=True)
    try:
        asked = []
        _filling(world, asked)
        share = world.librarian.tail_room(32768)["rag"]
        (record,) = w.run_turns(world, (("conv-1", w.FIRST, {}),))
        filled = world.wrapper.wrap("y" * 4 * (asked or [0])[0], source=world.wrapper.SOURCE_FILE)
        counted = world.executor._estimate_tokens(filled, w.MODEL)
    finally:
        close()
    assert len(asked) == 1 and type(asked[0]) is int and asked[0] > 0, asked
    assert counted <= share, ("what the builder fills at four characters a token fits its share once wrapped",
                              asked, counted, share)
    assert _in_file_envelope(w.sent(world)[0]["messages"], _RAG), "the retrieval lands whole, in its file envelope"
    assert [line for line in record["statuses"] if "left out" in line] == [], record["statuses"]
    assert world.rows[-1]["win_cut"].get("rag") == 0, world.rows[-1]["win_cut"]

    # Code, counted as the application counts it: what the builder returned is fitted to the share.
    world, close = w.open_real(gov={}, projects=True, estimate=_calibrated())
    try:
        asked = []
        world.projects.build_context = _code_builder(asked)
        (record,) = w.run_turns(world, (("conv-1", w.FIRST, {"model": "qwen3:8b"}),))
    finally:
        close()
    assert len(asked) == 1 and asked[0] > 0, asked
    assert _in_file_envelope(w.sent(world)[0]["messages"], _RAG), (
        "a retrieval of code lands in its file envelope", world.rows[-1]["win_cut"])
    assert world.rows[-1]["win_cut"].get("rag") == 0, world.rows[-1]["win_cut"]

    # A share that holds no chunk: neither the trigger nor the builder is asked, and the turn says so.
    world, close = w.open_real(gov={}, projects=True, limits=(8192, 4096))
    try:
        asked = []
        _filling(world, asked)
        (record,) = w.run_turns(world, (("conv-1", w.FIRST, {}),))
    finally:
        close()
    kinds = w.kinds(record)
    assert asked == [] and "project.detect" not in kinds, (asked, kinds)
    assert "project.lookup" in kinds, "control: the conversation's project was looked up"
    assert [line for line in record["statuses"] if "left out" in line] == [
        "[!] Project context left out: its share of the window holds no chunk"], record["statuses"]

    # A retrieval the window let go is said so, and counted in the row.
    for cut, said in (((("rag", 1), ("web", 0)), ["[!] Project context left out: over its share of the window"]),
                      ((), [])):
        world, close = w.open_world(gov={}, projects=True, journal={"cut": cut})
        try:
            (record,) = w.run_turns(world, (("conv-1", w.FIRST, {}),))
        finally:
            close()
        assert [line for line in record["statuses"] if "left out" in line] == said, record["statuses"]
        assert world.rows[-1]["win_cut"] == dict(cut), world.rows[-1]["win_cut"]


# ---------------------------------------------------------------------------
# JW28 -- the switch read once the stop is live
# ---------------------------------------------------------------------------
def test_jw28_the_onion_switch_is_read_once_the_calls_stop_is_live():
    for enabled in (True, False):
        world, close = w.open_world(gov={}, journal={"enabled": enabled})
        try:
            executor = world.executor_module.Executor()
            reading = world.executor_module._onion_enabled
            fired = []

            def read_under_a_stop(path=None):
                if not fired:
                    fired.append(True)
                    executor.cancel()
                return reading(path)

            world.executor_module._onion_enabled = read_under_a_stop
            (record,) = w.run_turns(world, _TWO[:1], executor=executor)
        finally:
            close()
        assert fired, "control: the switch was read"
        assert record["yields"] == ["[Cancelled]"], (enabled, record["yields"])
        assert [row["outcome"] for row in world.rows] == ["cancelled"], (enabled, world.rows)
        assert "stream" not in w.kinds(record) and "plan" not in w.kinds(record), (enabled, w.kinds(record))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
