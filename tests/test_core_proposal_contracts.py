"""Contracts for the proposals the queue offers the Core.

A span the queue holds keeps its probe-bearing units as anchors in the
Cellar. When one of them is a decision the user typed, it is offered to
the Core as a proposal: never written there by itself, accepted or
declined by the user alone, one at a time, its exact words shown with the
turn and the origin they came from.

  * CP1 -- a typed decision a held span keeps becomes one open proposal:
    its exact words, its turn and its origin; the Core is unchanged.
  * CP2 -- an anchor that carries no typed decision proposes nothing: a
    date, a name, the assistant's words.
  * CP3 -- a proposal is accepted by the user alone, one at a time, with its
    exact words: another actor, a list of ids and a proposal no longer open
    are refused by name, and nothing is pinned.
  * CP4 -- a declined proposal leaves the open list and the Core unchanged.
  * CP5 -- beyond the day's cap no proposal is made; the next day they are
    made again.
  * CP6 -- proposals come back from the store as they were saved.
  * CP7 -- a decision is anchored and offered in its own words beside a
    neighbour that answers it -- the question before it, a remark on it:
    the neighbour never takes its place. (The probes still read some such
    neighbours as decisions of their own; that is the generator's to fix.)
  * CP8 -- a decision past the day's cap is deferred, not lost: kept with
    the state, it is offered at the first step of a later day with room.
  * CP9 -- a proposal moved in the file is refused by name: its id no
    longer answers to its place.
  * CP10 -- a proposal whose place holds no typed decision of its turn is
    neither offered nor accepted, whatever id it carries.
  * CP11 -- a deferred decision is offered on a later day with no step
    since: listing the proposals opens what the day's cap has room for.
  * CP12 -- what a listing opened is saved: after a restart the proposal is
    taken by the id the listing showed.
  * CP13 -- a deferred proposal whose place no longer reads as a typed
    decision is not opened: it takes no room, and the next one is offered.

Local-only (the public distribution ships no tests). Loaded through the
shared isolation window from source; the registry is blocked and the clock
is the contract's.
"""

import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate, source  # noqa: E402

_MODULES = ("probes", "core_store", "receipts", "composer", "peels", "librarian", "onion_store")

_TYPED = [
    {"role": "user", "origin": "typed", "segments": [],
     "content": "Alice moved the build to Berlin on 2026-03-04. We keep Docker on the build server."},
    {"role": "assistant", "origin": "assistant", "segments": [],
     "content": "Noted: the Berlin build runs 12 jobs a day, a sensible load for that machine. "
                "Bob checks the logs every morning."},
]
_DECISION = "We keep Docker on the build server."
_LOSSY = "Alice moved the build to Berlin on 2026-03-04. The Berlin build runs 12 jobs a day. Bob checks the logs every morning."


def _open():
    loaded, restore = isolate(
        targets={f"opti_oignon.memory.{m}": source("memory", f"{m}.py") for m in _MODULES},
        blocked=("opti_oignon.inference_backend", "opti_oignon.db_utils"),
        packages=("opti_oignon.memory",),
    )
    lib = loaded["opti_oignon.memory.librarian"]
    lib.reset_librarian()
    return lib, loaded, restore


def _held(lib, loaded, cid="c1", spans=1, day="2026-10-06", per_day=5):
    """A conversation whose ``spans`` spans of the typed fixture were each held, on ``day``."""
    peels, composer = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.composer"]
    gate = replace(peels.load_gate(), span_turns=2)
    ladder = replace(peels.load_ladder(), rho=0.5, proposals_per_day=per_day)
    tiny = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)
    lib._today = lambda: day
    state = lib.state_for(cid)
    state.mirror(_TYPED * spans)
    for _ in range(spans):
        outcome = lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=ladder)
        assert outcome.rung == "held", "control: the span is held"
    return state


def test_cp1_a_typed_decision_a_held_span_keeps_becomes_one_open_proposal():
    lib, loaded, restore = _open()
    try:
        state = _held(lib, loaded)
        core_before = state.core.root()
        (offered,) = lib.proposals("c1")
        assert offered["text"] == _DECISION, "its exact words, read from the Cellar"
        assert offered["turn_id"] == "t0001" and offered["origin"] == "typed"
        assert state.core.root() == core_before and state.core.all() == [], "never written to the Core by itself"
    finally:
        restore()


def test_cp2_an_anchor_that_carries_no_typed_decision_proposes_nothing():
    lib, loaded, restore = _open()
    try:
        state = _held(lib, loaded)
        (receipt,) = state.ledger.all()
        assert len(receipt.anchors) == 2, "control: two typed anchors, a date with a name, and the decision"
        assert [p["text"] for p in lib.proposals("c1")] == [_DECISION], "the decision alone"
        assert len(state.proposals) == 1, "and no record of the other anchor: it takes none of the day's room"
    finally:
        restore()


def test_cp3_a_proposal_is_accepted_by_the_user_alone_one_at_a_time_with_its_exact_words():
    lib, loaded, restore = _open()
    try:
        state = _held(lib, loaded)
        (offered,) = lib.proposals("c1")
        with pytest.raises(PermissionError):
            lib.accept_proposal("c1", offered["id"], actor="librarian")
        with pytest.raises(TypeError, match="one proposal"):
            lib.accept_proposal("c1", [offered["id"]], actor="user")
        assert state.core.all() == [], "nothing pinned by a refusal"
        entry_id = lib.accept_proposal("c1", offered["id"], actor="user")
        assert [e.text for e in state.core.all()] == [_DECISION] and state.core.all()[0].id == entry_id
        assert lib.proposals("c1") == [], "an accepted proposal leaves the open list"
        with pytest.raises(KeyError, match="not open"):
            lib.accept_proposal("c1", offered["id"], actor="user")
        assert len(state.core.all()) == 1
    finally:
        restore()


def test_cp4_a_declined_proposal_leaves_the_open_list_and_the_core_unchanged():
    lib, loaded, restore = _open()
    try:
        state = _held(lib, loaded)
        (offered,) = lib.proposals("c1")
        with pytest.raises(PermissionError):
            lib.decline_proposal("c1", offered["id"], actor="assistant")
        assert len(lib.proposals("c1")) == 1, "control: a refused decline leaves it open"
        lib.decline_proposal("c1", offered["id"], actor="user")
        assert lib.proposals("c1") == [] and state.core.all() == []
    finally:
        restore()


def test_cp5_beyond_the_day_s_cap_no_proposal_is_made_and_the_next_day_they_are():
    lib, loaded, restore = _open()
    try:
        state = _held(lib, loaded, spans=2, per_day=1)
        peels, composer = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.composer"]
        gate = replace(peels.load_gate(), span_turns=2)
        ladder = replace(peels.load_ladder(), rho=0.5, proposals_per_day=1)
        tiny = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)
        assert len(lib.proposals("c1", ladder=ladder)) == 1, "two held decisions on one day, a cap of one"
        lib._today = lambda: "2026-10-07"
        state.mirror(_TYPED * 3)
        assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=ladder).rung == "held"
        assert [p["made_on"] for p in lib.proposals("c1", ladder=ladder)] == ["2026-10-06", "2026-10-07"]
    finally:
        restore()


def test_cp6_proposals_come_back_from_the_store_as_they_were_saved(tmp_path):
    lib, loaded, restore = _open()
    try:
        store_mod = loaded["opti_oignon.memory.onion_store"]
        state = _held(lib, loaded, spans=2)
        first, _second = lib.proposals("c1")
        lib.decline_proposal("c1", first["id"], actor="user")
        store = store_mod.OnionStore(tmp_path / "onion.db", connect=lambda p: sqlite3.connect(str(p)),
                                     require_encryption=False)
        store.save("c1", state)
        back = store.load("c1", lib.OnionState())
        assert back.proposals == state.proposals and len(back.proposals) == 2
        assert [q.status for q in back.proposals] == ["declined", "open"]
    finally:
        restore()


def _ladder_parts(loaded, per_day=5):
    peels, composer = loaded["opti_oignon.memory.peels"], loaded["opti_oignon.memory.composer"]
    gate = replace(peels.load_gate(), span_turns=2)
    ladder = replace(peels.load_ladder(), rho=0.5, proposals_per_day=per_day)
    tiny = composer.Budget(window=780, reserve=60, core=60, receipts=300, peels=160, flesh=1, turn=60)
    return gate, ladder, tiny


def test_cp7_a_decision_is_anchored_and_offered_in_its_own_words_beside_a_neighbour_that_answers_it():
    lib, loaded, restore = _open()
    try:
        gate, ladder, tiny = _ladder_parts(loaded)
        for neighbour in ("Do we keep Docker on the build server for Alice?",
                          "Alice thinks it is odd that we keep Docker on the build server in Berlin."):
            lib.reset_librarian()
            lib._today = lambda: "2026-10-06"
            state = lib.state_for("c1")
            state.mirror([
                {"role": "user", "origin": "typed", "segments": [], "content": neighbour + " " + _DECISION},
                {"role": "assistant", "origin": "assistant", "segments": [],
                 "content": "Noted, Bob checks the logs every morning."},
            ])
            outcome = lib.curate(state, lambda turns: "Bob checks the logs every morning.", gate=gate, budget=tiny,
                                 ladder=ladder)
            assert outcome.rung == "held", "control: the span is held"
            text = {t["turn_id"]: t["text"] for t in state.cellar.get(outcome.receipt.key)}
            kept = [text[turn][start:stop] for turn, start, stop in outcome.receipt.anchors]
            assert _DECISION in kept, "the decision is anchored in its own words"
            assert _DECISION in [p["text"] for p in lib.proposals("c1")], "and offered in them"
    finally:
        restore()


def test_cp8_a_decision_past_the_day_s_cap_is_offered_on_a_later_day_not_lost():
    lib, loaded, restore = _open()
    try:
        gate, ladder, tiny = _ladder_parts(loaded, per_day=1)
        podman = [dict(_TYPED[0], content=_TYPED[0]["content"].replace("Docker", "Podman")), _TYPED[1]]
        lib._today = lambda: "2026-10-06"
        state = lib.state_for("c1")
        state.mirror(_TYPED + podman)
        for _ in range(2):
            assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=ladder).rung == "held"
        assert [p["text"] for p in lib.proposals("c1", ladder=ladder)] == [_DECISION], "control: the cap holds the second back"
        lib._today = lambda: "2026-10-07"
        state.mirror(_TYPED + podman + [{"role": "user", "content": "ok."}, {"role": "assistant", "content": "sure."}])
        assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=ladder).rung == "bare"
        assert [q.status for q in state.proposals] == ["open", "open"], "the step itself opened it, before any listing"
        offered = lib.proposals("c1", ladder=ladder)
        assert [p["text"] for p in offered] == [_DECISION, _DECISION.replace("Docker", "Podman")]
        assert offered[1]["made_on"] == "2026-10-07", "it counts against the day it is offered on"
    finally:
        restore()


def test_cp9_a_proposal_moved_in_the_file_is_refused_by_name(tmp_path):
    lib, loaded, restore = _open()
    try:
        store_mod = loaded["opti_oignon.memory.onion_store"]
        state = _held(lib, loaded)
        q = state.proposals[0]
        text = {t["turn_id"]: t["text"] for t in state.cellar.get(q.span_key)}[q.turn_id]
        start = text.index("Alice moved")
        path = tmp_path / "onion.db"
        store = store_mod.OnionStore(path, connect=lambda p: sqlite3.connect(str(p)), require_encryption=False)
        store.save("c1", state)
        conn = sqlite3.connect(str(path))
        conn.execute("UPDATE onion_proposals SET start = ?, stop = ?", (start, start + len("Alice moved the build")))
        conn.commit()
        conn.close()
        with pytest.raises(store_mod.OnionIntegrityError, match="no longer answers to its place"):
            store.load("c1", lib.OnionState())
    finally:
        restore()


def test_cp10_a_proposal_whose_place_holds_no_typed_decision_is_neither_offered_nor_accepted():
    import hashlib

    lib, loaded, restore = _open()
    try:
        state = _held(lib, loaded)
        q = state.proposals[0]
        text = {t["turn_id"]: t["text"] for t in state.cellar.get(q.span_key)}[q.turn_id]
        start = text.index("Alice moved")
        stop = start + len("Alice moved the build to Berlin on 2026-03-04.")
        forged = hashlib.sha256(f"{q.span_key}:{q.turn_id}:{start}:{stop}".encode()).hexdigest()
        state.proposals[0] = replace(q, id=forged, start=start, stop=stop)
        assert lib.proposals("c1") == [], "a place that holds no typed decision is not offered"
        with pytest.raises(KeyError, match="typed decision"):
            lib.accept_proposal("c1", forged, actor="user")
        assert state.core.all() == [], "nothing pinned"
    finally:
        restore()


def test_cp11_a_deferred_decision_is_offered_on_a_later_day_with_no_step_since():
    lib, loaded, restore = _open()
    try:
        gate, ladder, tiny = _ladder_parts(loaded, per_day=1)
        podman = [dict(_TYPED[0], content=_TYPED[0]["content"].replace("Docker", "Podman")), _TYPED[1]]
        lib._today = lambda: "2026-10-06"
        state = lib.state_for("c1")
        state.mirror(_TYPED + podman)
        for _ in range(2):
            assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=ladder).rung == "held"
        assert state.flesh.turns() == [], "control: nothing left for a later step to run on"
        assert [p["text"] for p in lib.proposals("c1", ladder=ladder)] == [_DECISION], "control: one deferred"
        lib._today = lambda: "2026-10-08"
        offered = lib.proposals("c1", ladder=ladder)
        assert [p["text"] for p in offered] == [_DECISION, _DECISION.replace("Docker", "Podman")]
        assert offered[1]["made_on"] == "2026-10-08", "it counts against the day it is offered on"
    finally:
        restore()


def _variants(*tools):
    """The typed fixture once per tool, each a span whose typed decision keeps that tool."""
    return [turn for tool in tools
            for turn in (dict(_TYPED[0], content=_TYPED[0]["content"].replace("Docker", tool)), _TYPED[1])]


def test_cp12_a_proposal_a_listing_opened_is_saved_and_taken_after_a_restart(tmp_path):
    lib, loaded, restore = _open()
    try:
        store_mod = loaded["opti_oignon.memory.onion_store"]
        path = tmp_path / "onion.db"

        def register():
            lib._store[(str(path), False)] = store_mod.OnionStore(path, connect=lambda p: sqlite3.connect(str(p)),
                                                                  require_encryption=False)

        register()
        config = lib.LibrarianConfig(enabled=True, model="fake:1b", keep_alive="0", min_new_turns=1, temperature=0.0,
                                     num_predict=64, persist_path=str(path), require_encryption=False)
        gate, ladder, tiny = _ladder_parts(loaded, per_day=1)
        lib._today = lambda: "2026-10-06"
        state = lib.state_for("c1", config)
        state.mirror(_variants("Docker", "Podman"))
        for _ in range(2):
            assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=ladder).rung == "held"
        lib._save_state("c1", state, config)
        lib._today = lambda: "2026-10-08"
        offered = lib.proposals("c1", config=config, ladder=ladder)
        assert len(offered) == 2, "control: the listing opened the deferred one"
        lib.reset_librarian()
        register()
        assert lib.accept_proposal("c1", offered[1]["id"], actor="user", config=config), "taken by the id it was shown with"
    finally:
        restore()


def test_cp13_a_deferred_proposal_whose_place_no_longer_reads_gives_way_to_the_next():
    import hashlib

    lib, loaded, restore = _open()
    try:
        gate, ladder, tiny = _ladder_parts(loaded, per_day=1)
        lib._today = lambda: "2026-10-06"
        state = lib.state_for("c1")
        state.mirror(_variants("Docker", "Podman", "Nginx"))
        for _ in range(3):
            assert lib.curate(state, lambda turns: _LOSSY, gate=gate, budget=tiny, ladder=ladder).rung == "held"
        assert [q.status for q in state.proposals] == ["open", "deferred", "deferred"], "control: two deferred"
        q = state.proposals[1]
        text = {t["turn_id"]: t["text"] for t in state.cellar.get(q.span_key)}[q.turn_id]
        start = text.index("Alice moved")
        stop = start + len("Alice moved the build to Berlin on 2026-03-04.")
        forged = hashlib.sha256(f"{q.span_key}:{q.turn_id}:{start}:{stop}".encode()).hexdigest()
        state.proposals[1] = replace(q, id=forged, start=start, stop=stop)
        lib._today = lambda: "2026-10-07"
        offered = [p["text"] for p in lib.proposals("c1", ladder=ladder)]
        assert offered == [_DECISION, _DECISION.replace("Docker", "Nginx")], "the next one takes the day's room"
    finally:
        restore()
